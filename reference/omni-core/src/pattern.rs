//! Name selectors: globs with captures, and a small bounded regular expression
//! engine.
//!
//! §08.3 selects base tensors "by glob, regex, `semantic`, `role`, or `axes`
//! pattern", and uses the wildcard captures (`{1}`, `{2}`, …) to index the
//! adapter's own tensors. That makes pattern matching part of the format rather
//! than a convenience, so it is implemented here with the same care as the
//! parser: patterns come out of untrusted containers.
//!
//! The regex engine is a deliberately small subset — literals, `.`, character
//! classes, `*`, `+`, `?`, bounded repetition, alternation, groups and anchors,
//! with no backreferences and no lookaround. It backtracks, so it has a **step
//! budget**: a pattern like `(a+)+b` against a long string of `a`s is a denial
//! of service in most backtracking engines, and a format that lets a published
//! model choose the pattern cannot afford that. Exceeding the budget is
//! reported as [`Error::Budget`], never as "no match".

/// Maximum backtracking steps for one match attempt.
pub const STEP_BUDGET: u32 = 100_000;

/// Maximum pattern length and nesting, so parsing is bounded too.
pub const MAX_PATTERN: usize = 4096;
const MAX_NEST: usize = 32;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Syntax(String),
    /// The match exceeded [`STEP_BUDGET`]. Indeterminate, not "no".
    Budget,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Syntax(m) => write!(f, "pattern syntax: {m}"),
            Error::Budget => write!(
                f,
                "pattern matching exceeded {STEP_BUDGET} steps; the result is unknown, not negative"
            ),
        }
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------- glob --

/// Matches a glob against a name, returning one capture per wildcard.
///
/// `*` matches any run of characters that does not cross a `.`, `?` matches one
/// character, and `**` matches anything including separators. Restricting `*` at
/// the separator is what makes `model.layers.*.attn.q_proj.weight` select one
/// layer's projection rather than half the model.
pub fn glob_captures(pattern: &str, name: &str) -> Option<Vec<String>> {
    if pattern.len() > MAX_PATTERN {
        return None;
    }
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    let mut caps = Vec::new();
    if glob_walk(&p, 0, &n, 0, &mut caps) {
        Some(caps)
    } else {
        None
    }
}

pub fn glob_match(pattern: &str, name: &str) -> bool {
    glob_captures(pattern, name).is_some()
}

fn glob_walk(p: &[char], pi: usize, n: &[char], ni: usize, caps: &mut Vec<String>) -> bool {
    if pi == p.len() {
        return ni == n.len();
    }
    match p[pi] {
        '*' => {
            let deep = p.get(pi + 1) == Some(&'*');
            let next = pi + if deep { 2 } else { 1 };
            // Try the longest match first, shrinking; greedy is what callers
            // expect from a glob.
            let mut end = n.len();
            loop {
                let seg = &n[ni..end];
                if deep || !seg.contains(&'.') {
                    let at = caps.len();
                    caps.push(seg.iter().collect());
                    if glob_walk(p, next, n, end, caps) {
                        return true;
                    }
                    caps.truncate(at);
                }
                if end == ni {
                    return false;
                }
                end -= 1;
            }
        }
        '?' => {
            if ni < n.len() {
                let at = caps.len();
                caps.push(n[ni].to_string());
                if glob_walk(p, pi + 1, n, ni + 1, caps) {
                    return true;
                }
                caps.truncate(at);
            }
            false
        }
        c => ni < n.len() && n[ni] == c && glob_walk(p, pi + 1, n, ni + 1, caps),
    }
}

/// Substitutes `{1}`, `{2}`, … in a template with the corresponding capture.
/// §08.3 uses this to turn a matched base tensor name into the adapter's own
/// tensor names, which is how an adapter attaches to a base it has never seen.
pub fn substitute(template: &str, captures: &[String]) -> Result<String, Error> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '{' {
            out.push(c);
            continue;
        }
        let mut num = String::new();
        loop {
            match chars.next() {
                Some('}') => break,
                Some(d) if d.is_ascii_digit() => num.push(d),
                Some(other) => {
                    return Err(Error::Syntax(format!(
                        "unexpected `{other}` in capture reference"
                    )))
                }
                None => return Err(Error::Syntax("unterminated capture reference".into())),
            }
        }
        let i: usize = num
            .parse()
            .map_err(|_| Error::Syntax(format!("`{{{num}}}` is not a capture number")))?;
        if i == 0 || i > captures.len() {
            return Err(Error::Syntax(format!(
                "capture {{{i}}} does not exist; the pattern produced {} \
                 (a mismatch here would bind the wrong adapter tensor)",
                captures.len()
            )));
        }
        out.push_str(&captures[i - 1]);
    }
    Ok(out)
}

// --------------------------------------------------------------------- regex --

#[derive(Clone, Debug, PartialEq, Eq)]
enum Node {
    Empty,
    Char(char),
    Any,
    Class {
        set: Vec<(char, char)>,
        negated: bool,
    },
    Start,
    End,
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    /// `{min, max}` repetition; `max == None` is unbounded.
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    Group {
        node: Box<Node>,
        index: Option<usize>,
    },
}

/// A compiled regular expression over the documented subset.
#[derive(Clone, Debug)]
pub struct Regex {
    root: Node,
    groups: usize,
    source: String,
}

impl Regex {
    pub fn parse(pattern: &str) -> Result<Regex, Error> {
        if pattern.len() > MAX_PATTERN {
            return Err(Error::Syntax(format!(
                "pattern longer than {MAX_PATTERN} bytes"
            )));
        }
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser {
            c: &chars,
            i: 0,
            groups: 0,
            depth: 0,
        };
        let root = p.alternation()?;
        if p.i != chars.len() {
            return Err(Error::Syntax(format!(
                "unexpected `{}` at offset {}",
                chars[p.i], p.i
            )));
        }
        Ok(Regex {
            root,
            groups: p.groups,
            source: pattern.to_string(),
        })
    }

    pub fn as_str(&self) -> &str {
        &self.source
    }

    pub fn group_count(&self) -> usize {
        self.groups
    }

    /// True when the pattern matches anywhere in `text`.
    pub fn is_match(&self, text: &str) -> Result<bool, Error> {
        Ok(self.captures(text)?.is_some())
    }

    /// The byte range of the leftmost match, or `None` when the pattern matches
    /// nowhere in `text`.
    ///
    /// Semantics are leftmost-first, as in any backtracking engine: the
    /// earliest start position that can match wins, and the end is wherever the
    /// pattern's own greediness lands. Splitting text needs the offsets, which
    /// [`Regex::is_match`] cannot give — asking "does it match?" of successive
    /// substrings finds the wrong boundaries, because an unanchored match
    /// inside a substring says nothing about where that substring begins.
    pub fn find(&self, text: &str) -> Result<Option<(usize, usize)>, Error> {
        let t: Vec<char> = text.chars().collect();
        // char index -> byte offset, so callers can slice `text` directly.
        let mut byte_of: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        byte_of.push(text.len());
        for start in 0..=t.len() {
            let mut st = State {
                text: &t,
                steps: 0,
                caps: vec![None; self.groups],
            };
            let mut end = start;
            let hit = st.matches(&self.root, start, &mut |_, e| {
                end = e;
                true
            })?;
            if hit {
                return Ok(Some((byte_of[start], byte_of[end])));
            }
        }
        Ok(None)
    }

    /// The capture groups of the leftmost match, in group order. An empty
    /// vector means the pattern matched but has no groups.
    pub fn captures(&self, text: &str) -> Result<Option<Vec<String>>, Error> {
        let t: Vec<char> = text.chars().collect();
        for start in 0..=t.len() {
            let mut st = State {
                text: &t,
                steps: 0,
                caps: vec![None; self.groups],
            };
            if st.matches(&self.root, start, &mut |_, _| true)? {
                let mut out = Vec::with_capacity(self.groups);
                for c in &st.caps {
                    out.push(match c {
                        Some((a, b)) => t[*a..*b].iter().collect(),
                        None => String::new(),
                    });
                }
                return Ok(Some(out));
            }
        }
        Ok(None)
    }
}

struct Parser<'a> {
    c: &'a [char],
    i: usize,
    groups: usize,
    depth: usize,
}

impl Parser<'_> {
    fn alternation(&mut self) -> Result<Node, Error> {
        let mut branches = vec![self.concat()?];
        while self.peek() == Some('|') {
            self.i += 1;
            branches.push(self.concat()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap()
        } else {
            Node::Alt(branches)
        })
    }

    fn concat(&mut self) -> Result<Node, Error> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            items.push(self.repeat()?);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().unwrap(),
            _ => Node::Concat(items),
        })
    }

    fn repeat(&mut self) -> Result<Node, Error> {
        let atom = self.atom()?;
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, None)
            }
            Some('+') => {
                self.i += 1;
                (1, None)
            }
            Some('?') => {
                self.i += 1;
                (0, Some(1))
            }
            Some('{') => {
                // Bounded repetition. An unbounded upper limit here would be a
                // way to smuggle a very large match past the step budget, so
                // the count is capped.
                let save = self.i;
                self.i += 1;
                let lo = self.number();
                let hi =
                    if self.peek() == Some(',') {
                        self.i += 1;
                        if self.peek() == Some('}') {
                            None
                        } else {
                            Some(self.number().ok_or_else(|| {
                                Error::Syntax("expected a number after `,`".into())
                            })?)
                        }
                    } else {
                        lo
                    };
                if self.peek() != Some('}') || lo.is_none() {
                    // Not a repetition after all: treat `{` as a literal.
                    self.i = save;
                    return Ok(atom);
                }
                self.i += 1;
                let lo = lo.unwrap();
                if lo > 1024 || hi.is_some_and(|h| h > 1024 || h < lo) {
                    return Err(Error::Syntax(
                        "repetition counts must be <= 1024 and increasing".into(),
                    ));
                }
                (lo, hi)
            }
            _ => return Ok(atom),
        };
        let greedy = if self.peek() == Some('?') {
            self.i += 1;
            false
        } else {
            true
        };
        Ok(Node::Repeat {
            node: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        if self.i == start {
            return None;
        }
        self.c[start..self.i]
            .iter()
            .collect::<String>()
            .parse()
            .ok()
    }

    fn atom(&mut self) -> Result<Node, Error> {
        let c = self
            .peek()
            .ok_or_else(|| Error::Syntax("unexpected end of pattern".into()))?;
        self.i += 1;
        Ok(match c {
            '.' => Node::Any,
            '^' => Node::Start,
            '$' => Node::End,
            '(' => {
                self.depth += 1;
                if self.depth > MAX_NEST {
                    return Err(Error::Syntax(format!("nesting deeper than {MAX_NEST}")));
                }
                // `(?:` is a non-capturing group; nothing else after `(?` is
                // supported, and pretending otherwise would silently change
                // what a selector means.
                let index = if self.peek() == Some('?') {
                    if self.c.get(self.i + 1) == Some(&':') {
                        self.i += 2;
                        None
                    } else {
                        return Err(Error::Syntax(
                            "only `(?:` groups are supported; lookaround and flags are not".into(),
                        ));
                    }
                } else {
                    self.groups += 1;
                    Some(self.groups - 1)
                };
                let inner = self.alternation()?;
                if self.peek() != Some(')') {
                    return Err(Error::Syntax("unclosed group".into()));
                }
                self.i += 1;
                self.depth -= 1;
                Node::Group {
                    node: Box::new(inner),
                    index,
                }
            }
            ')' => return Err(Error::Syntax("unmatched `)`".into())),
            '[' => self.class()?,
            '\\' => {
                let e = self
                    .peek()
                    .ok_or_else(|| Error::Syntax("trailing backslash".into()))?;
                self.i += 1;
                match e {
                    'd' => Node::Class {
                        set: vec![('0', '9')],
                        negated: false,
                    },
                    'D' => Node::Class {
                        set: vec![('0', '9')],
                        negated: true,
                    },
                    'w' => Node::Class {
                        set: word_class(),
                        negated: false,
                    },
                    'W' => Node::Class {
                        set: word_class(),
                        negated: true,
                    },
                    's' => Node::Class {
                        set: space_class(),
                        negated: false,
                    },
                    'S' => Node::Class {
                        set: space_class(),
                        negated: true,
                    },
                    'n' => Node::Char('\n'),
                    't' => Node::Char('\t'),
                    'r' => Node::Char('\r'),
                    'f' => Node::Char('\u{c}'),
                    '0' => Node::Char('\0'),
                    other if other.is_ascii_digit() => {
                        return Err(Error::Syntax(
                            "backreferences are not supported: they are what make matching \
                             super-linear, and a selector is not worth that"
                                .into(),
                        ))
                    }
                    other => Node::Char(literal_escape(other)?),
                }
            }
            other => Node::Char(other),
        })
    }

    fn class(&mut self) -> Result<Node, Error> {
        let negated = if self.peek() == Some('^') {
            self.i += 1;
            true
        } else {
            false
        };
        let mut set = Vec::new();
        let mut first = true;
        loop {
            let c = self
                .peek()
                .ok_or_else(|| Error::Syntax("unclosed character class".into()))?;
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let lo = if c == '\\' {
                let e = self
                    .peek()
                    .ok_or_else(|| Error::Syntax("trailing backslash in class".into()))?;
                self.i += 1;
                match e {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    'f' => '\u{c}',
                    other => literal_escape(other)?,
                }
            } else {
                c
            };
            if self.peek() == Some('-') && self.c.get(self.i + 1).is_some_and(|c| *c != ']') {
                self.i += 1;
                let hi = self.c[self.i];
                self.i += 1;
                if hi < lo {
                    return Err(Error::Syntax(format!("class range {lo}-{hi} is inverted")));
                }
                set.push((lo, hi));
            } else {
                set.push((lo, lo));
            }
        }
        Ok(Node::Class { set, negated })
    }

    fn peek(&self) -> Option<char> {
        self.c.get(self.i).copied()
    }
}

/// Resolves `\c` for a `c` the engine has no rule for.
///
/// Escaping punctuation means the literal character in every regex flavour, so
/// `\.` and `\+` are safe. An escaped *letter* is a different matter: `\p{L}`,
/// `\b`, `\A` and friends are constructs this engine does not implement, and
/// reading `\p` as the letter `p` would match a completely different language
/// without saying so. A pattern this engine cannot honour is an error, so the
/// caller can report it as indeterminate (§15.1) instead of acting on a wrong
/// answer.
fn literal_escape(c: char) -> Result<char, Error> {
    if c.is_alphanumeric() {
        return Err(Error::Syntax(format!(
            "`\\{c}` is not a construct this engine implements; it is not treated as the \
             literal `{c}`, because that would silently match a different language"
        )));
    }
    Ok(c)
}

fn word_class() -> Vec<(char, char)> {
    vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')]
}

fn space_class() -> Vec<(char, char)> {
    vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')]
}

struct State<'a> {
    text: &'a [char],
    steps: u32,
    caps: Vec<Option<(usize, usize)>>,
}

/// A continuation: "the rest of the pattern matched, starting here".
type Cont<'c> = dyn FnMut(&mut State<'_>, usize) -> bool + 'c;

impl State<'_> {
    fn step(&mut self) -> Result<(), Error> {
        self.steps += 1;
        if self.steps > STEP_BUDGET {
            return Err(Error::Budget);
        }
        Ok(())
    }

    fn matches(&mut self, node: &Node, at: usize, k: &mut Cont<'_>) -> Result<bool, Error> {
        self.step()?;
        match node {
            Node::Empty => Ok(k(self, at)),
            Node::Char(c) => Ok(self.text.get(at) == Some(c) && k(self, at + 1)),
            Node::Any => Ok(at < self.text.len() && k(self, at + 1)),
            Node::Class { set, negated } => {
                let Some(c) = self.text.get(at) else {
                    return Ok(false);
                };
                let inside = set.iter().any(|(lo, hi)| c >= lo && c <= hi);
                Ok(inside != *negated && k(self, at + 1))
            }
            Node::Start => Ok(at == 0 && k(self, at)),
            Node::End => Ok(at == self.text.len() && k(self, at)),
            Node::Concat(items) => self.concat_from(items, 0, at, k),
            Node::Alt(branches) => {
                for b in branches {
                    if self.matches(b, at, k)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Node::Group { node, index } => {
                let idx = *index;
                let start = at;
                let saved = idx.map(|i| self.caps[i]);
                let mut inner = |st: &mut State<'_>, end: usize| -> bool {
                    if let Some(i) = idx {
                        st.caps[i] = Some((start, end));
                    }
                    k(st, end)
                };
                let ok = self.matches(node, at, &mut inner)?;
                if !ok {
                    if let (Some(i), Some(s)) = (idx, saved) {
                        self.caps[i] = s;
                    }
                }
                Ok(ok)
            }
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => self.repeat(node, *min, *max, *greedy, 0, at, k),
        }
    }

    fn concat_from(
        &mut self,
        items: &[Node],
        i: usize,
        at: usize,
        k: &mut Cont<'_>,
    ) -> Result<bool, Error> {
        if i == items.len() {
            return Ok(k(self, at));
        }
        // The continuation for item `i` is "match items i+1.. then k".
        let mut err = None;
        let mut rest = |st: &mut State<'_>, next: usize| -> bool {
            match st.concat_from(items, i + 1, next, k) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        };
        let out = self.matches(&items[i], at, &mut rest)?;
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn repeat(
        &mut self,
        node: &Node,
        min: u32,
        max: Option<u32>,
        greedy: bool,
        done: u32,
        at: usize,
        k: &mut Cont<'_>,
    ) -> Result<bool, Error> {
        self.step()?;
        let can_more = max.is_none_or(|m| done < m);
        if greedy {
            if can_more && self.one_more(node, min, max, greedy, done, at, k)? {
                return Ok(true);
            }
            Ok(done >= min && k(self, at))
        } else {
            if done >= min && k(self, at) {
                return Ok(true);
            }
            if can_more {
                self.one_more(node, min, max, greedy, done, at, k)
            } else {
                Ok(false)
            }
        }
    }

    /// Matches one more iteration of a repetition, then the rest of it.
    #[allow(clippy::too_many_arguments)]
    fn one_more(
        &mut self,
        node: &Node,
        min: u32,
        max: Option<u32>,
        greedy: bool,
        done: u32,
        at: usize,
        k: &mut Cont<'_>,
    ) -> Result<bool, Error> {
        let mut err = None;
        let mut again = |st: &mut State<'_>, next: usize| -> bool {
            // An iteration that consumed nothing would loop forever.
            if next == at {
                return false;
            }
            match st.repeat(node, min, max, greedy, done + 1, next, k) {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        };
        let out = self.matches(node, at, &mut again)?;
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs_capture_their_wildcards() {
        let caps = glob_captures(
            "model.layers.*.attn.q_proj.weight",
            "model.layers.7.attn.q_proj.weight",
        )
        .unwrap();
        assert_eq!(caps, vec!["7".to_string()]);
        assert_eq!(
            substitute("lora.{1}.q_proj.A", &caps).unwrap(),
            "lora.7.q_proj.A"
        );
        // A single star does not cross a separator, so it selects one layer.
        assert!(!glob_match(
            "model.layers.*.weight",
            "model.layers.7.attn.q_proj.weight"
        ));
        // A double star does.
        assert!(glob_match(
            "model.layers.**.weight",
            "model.layers.7.attn.q_proj.weight"
        ));
        assert_eq!(
            glob_captures("*.mlp.experts.?.*", "model.layers.0.mlp.experts.3.w1"),
            None
        );
        assert_eq!(
            glob_captures("**.mlp.experts.?.*", "model.layers.0.mlp.experts.3.w1").unwrap(),
            vec!["model.layers.0".to_string(), "3".into(), "w1".into()]
        );
    }

    #[test]
    fn substitution_refuses_a_capture_that_does_not_exist() {
        let caps = vec!["7".to_string()];
        assert!(substitute("lora.{2}.A", &caps).is_err());
        assert!(substitute("lora.{0}.A", &caps).is_err());
        assert!(substitute("lora.{1", &caps).is_err());
        assert_eq!(substitute("plain", &caps).unwrap(), "plain");
    }

    fn m(p: &str, s: &str) -> bool {
        Regex::parse(p).unwrap().is_match(s).unwrap()
    }

    #[test]
    fn the_regex_subset_behaves() {
        assert!(m("abc", "xxabcxx"));
        assert!(!m("^abc", "xxabc"));
        assert!(m("^abc$", "abc"));
        assert!(m("a.c", "abc"));
        assert!(!m("a.c", "ac"));
        assert!(m("ab*c", "ac"));
        assert!(m("ab*c", "abbbc"));
        assert!(m("ab+c", "abc"));
        assert!(!m("ab+c", "ac"));
        assert!(m("ab?c", "ac"));
        assert!(m("a|b", "b"));
        assert!(m("(ab)+", "abab"));
        assert!(m("[a-c]+", "bbb"));
        assert!(!m("[^a-c]+", "abc"));
        assert!(m(r"\d{2,3}", "layers.42."));
        assert!(!m(r"^\d{4}$", "123"));
        assert!(m(r"q_proj\.weight$", "model.layers.0.attn.q_proj.weight"));
        assert!(m("(?:ab)+c", "ababc"));
        // Non-greedy repetition.
        let r = Regex::parse("<(.+?)>").unwrap();
        assert_eq!(
            r.captures("<a><b>").unwrap().unwrap(),
            vec!["a".to_string()]
        );
    }

    #[test]
    fn regex_captures_index_adapter_tensors() {
        let r = Regex::parse(r"^model\.layers\.(\d+)\.attn\.(q|k|v)_proj\.weight$").unwrap();
        assert_eq!(r.group_count(), 2);
        let caps = r
            .captures("model.layers.31.attn.v_proj.weight")
            .unwrap()
            .unwrap();
        assert_eq!(caps, vec!["31".to_string(), "v".to_string()]);
        assert_eq!(
            substitute("lora.{1}.{2}_proj.B", &caps).unwrap(),
            "lora.31.v_proj.B"
        );
        assert!(r
            .captures("model.layers.x.attn.v_proj.weight")
            .unwrap()
            .is_none());
    }

    #[test]
    fn syntax_errors_are_errors_not_silent_mismatches() {
        for bad in [
            "(", ")", "[a", "[z-a]", "a{2,1}", "a\\", "(?=x)", "(a)\\1", "a{2000}",
        ] {
            assert!(Regex::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }

    #[test]
    fn catastrophic_backtracking_is_budgeted_not_survived() {
        // The classic: nested unbounded repetition against a long non-matching
        // string. A published container can choose this pattern, so the engine
        // must refuse rather than hang.
        let r = Regex::parse("(a+)+b").unwrap();
        let s = "a".repeat(60);
        match r.is_match(&s) {
            Err(Error::Budget) => {}
            other => panic!("expected a budget error, got {other:?}"),
        }
        // The same pattern on a matching string is fine and fast.
        assert!(r.is_match(&format!("{}b", "a".repeat(20))).unwrap());
        // And an empty-body repetition terminates instead of looping.
        assert!(Regex::parse("(a*)*b").unwrap().is_match("b").unwrap());
    }

    #[test]
    fn an_unimplemented_escape_is_an_error_not_a_literal() {
        // `\p{L}` is the pre-tokenizer pattern every GPT-2 descendant uses. An
        // engine without Unicode property classes that reads `\p` as the letter
        // `p` matches a different language and produces different token ids
        // without a word of complaint, which is the worst of the three outcomes.
        for bad in [r"\p{L}+", r"[^\p{N}]", r"\bword\b", r"\A.\z", r"\Qa.b\E"] {
            assert!(Regex::parse(bad).is_err(), "`{bad}` must not parse");
        }
        // Escaped punctuation means the literal in every flavour, so it stays.
        assert!(m(r"a\.b", "a.b"));
        assert!(!m(r"^a\.b$", "axb"));
        assert!(m(r"a\\b", r"a\b"));
        // And the escapes the engine does implement keep working.
        assert!(m(r"\d\s\w", "1 a"));
        assert!(m("a\\tb", "a\tb"));
        assert!(m("a\\rb", "a\rb"));
    }

    #[test]
    fn find_reports_where_the_leftmost_match_is() {
        let r = Regex::parse("[ab]+").unwrap();
        assert_eq!(r.find("cabc").unwrap(), Some((1, 3)));
        assert_eq!(r.find("ccc").unwrap(), None);
        // Offsets are byte offsets into the input, so multi-byte text slices
        // correctly: `é` is two bytes, so the match starts at 2, not 1.
        assert_eq!(r.find("¿ab").unwrap(), Some((2, 4)));
        let s = "¿ab";
        let (a, b) = r.find(s).unwrap().unwrap();
        assert_eq!(&s[a..b], "ab");
        // An anchor is honoured rather than ignored while scanning forward.
        assert_eq!(Regex::parse("^b").unwrap().find("ab").unwrap(), None);
        // A pattern that can match empty reports an empty span at the front.
        assert_eq!(
            Regex::parse("a*").unwrap().find("bb").unwrap(),
            Some((0, 0))
        );
    }

    #[test]
    fn very_long_patterns_are_refused() {
        let long = "a".repeat(MAX_PATTERN + 1);
        assert!(Regex::parse(&long).is_err());
        assert!(glob_captures(&long, "a").is_none());
    }
}
