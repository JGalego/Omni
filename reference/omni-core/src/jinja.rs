//! Jinja2 → OMNI-CT translation (§06.9).
//!
//! Every chat template in the wild today is a Jinja2 string that the runtime
//! `exec`s. §06.9's answer is OMNI-CT, a language that is total by construction.
//! That answer is only useful if real templates can get there, so this module is
//! the bridge — and, more importantly, the *measurement*: the roadmap's Gate 2
//! asks for translation to succeed on 95 % of a hub snapshot, and a translator
//! that cannot say which templates it fails on is not evidence for anything.
//!
//! ## What translation means here
//!
//! Not "parse Jinja2". Jinja2 is Turing-complete — macros, recursion, arbitrary
//! Python method calls, `{% include %}` — and the whole point of §06.9 is that a
//! chat template does not need any of it. So this translates the *subset chat
//! templates actually use* and **refuses the rest by name**, with the construct
//! and its position, because a refusal a maintainer can act on is worth more
//! than a partial translation nobody can trust.
//!
//! The refusals are the interesting output. A template that needs `{% macro %}`
//! is telling you something about either the template or §06.9's grammar, and
//! [`Report`] keeps the reason so the two can be told apart.
//!
//! ## Where the semantics differ, and what is done about it
//!
//! * **Undefined variables.** Jinja renders them as the empty string; OMNI-CT
//!   makes them an error (§06.9, and the `ct` module's own note on why). A
//!   translation cannot preserve both, so it preserves OMNI-CT's: a template
//!   that silently dropped a variable was already shipping a subtly wrong
//!   prompt. `is defined` and `| default(…)` translate faithfully, so a template
//!   that *handles* the absent case keeps handling it.
//! * **Method calls.** `s.strip()`, `s.split(x)`, `s.startswith(x)` and friends
//!   are Python methods on a host object. The first two have exact OMNI-CT
//!   equivalents (`trim`, `split`) and are rewritten; anything else is refused
//!   rather than approximated.
//! * **`loop.index0` / `loop.first` / `loop.last`.** OMNI-CT's `for` has no loop
//!   variable. These are translated by binding the loop over an index range —
//!   which OMNI-CT cannot express either — so they are **refused**, and that is
//!   a genuine gap in §06.9 rather than in this translator. It is the single
//!   most common refusal, and §06.9 should grow a loop variable.

use crate::ct::{BinOp, Expr, Node, Template};

/// Why a template could not be translated.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    /// The Jinja construct, named the way a template author would recognise it.
    pub construct: String,
    /// Why it has no OMNI-CT form.
    pub reason: String,
    /// Byte offset into the source, so a maintainer can find it.
    pub at: usize,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` at byte {}: {}",
            self.construct, self.at, self.reason
        )
    }
}

#[derive(Debug)]
pub enum Error {
    /// The source is not valid Jinja2, or not valid in a way this reads.
    Syntax(String, usize),
    /// Valid Jinja2 that OMNI-CT has no form for.
    Unsupported(Refusal),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Syntax(m, at) => write!(f, "syntax error at byte {at}: {m}"),
            Error::Unsupported(r) => write!(f, "{r}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// Longest source accepted. Chat templates are a few kilobytes; anything vastly
/// larger is not a chat template.
pub const MAX_SOURCE: usize = 64 << 10;

/// A successful translation, and what it cost.
#[derive(Clone, Debug)]
pub struct Translated {
    pub template: Template,
    /// Constructs rewritten rather than carried across, so a reviewer can see
    /// which lines are not literally what the author wrote.
    pub rewrites: Vec<String>,
}

// ---------------------------------------------------------------------- lexer --

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    /// Literal text between tags, with whitespace control already applied.
    Text(String),
    /// `{{ … }}`
    Out(String, usize),
    /// `{% … %}`
    Tag(String, usize),
}

/// Splits Jinja source into text, `{{ }}` and `{% %}`, honouring the `-`
/// whitespace-control markers.
///
/// Whitespace control is not cosmetic in a chat template: `{%- if -%}` versus
/// `{% if %}` is the difference between a prompt with a leading newline and one
/// without, and a tokenizer will notice. So it is applied here rather than left
/// for a reader to approximate.
fn lex(src: &str) -> Res<Vec<Tok>> {
    let b = src.as_bytes();
    let mut out: Vec<Tok> = Vec::new();
    let mut text = String::new();
    let mut i = 0usize;
    // Set when a tag ended with `-%}` or `-}}`: the *next* text run loses its
    // leading whitespace.
    let mut trim_next = false;
    while i < b.len() {
        if b[i] == b'{' && i + 1 < b.len() && matches!(b[i + 1], b'{' | b'%' | b'#') {
            let kind = b[i + 1];
            let close: &[u8] = if kind == b'{' {
                b"}}"
            } else if kind == b'%' {
                b"%}"
            } else {
                b"#}"
            };
            let mut start = i + 2;
            let mut trim_before = false;
            if start < b.len() && b[start] == b'-' {
                trim_before = true;
                start += 1;
            }
            let end = find(b, close, start)
                .ok_or_else(|| Error::Syntax(format!("unclosed `{}`", kind as char), i))?;
            let mut inner_end = end;
            let mut trim_after = false;
            if inner_end > start && b[inner_end - 1] == b'-' {
                trim_after = true;
                inner_end -= 1;
            }
            if trim_before {
                while text.ends_with([' ', '\t', '\n', '\r']) {
                    text.pop();
                }
            }
            if !text.is_empty() {
                out.push(Tok::Text(std::mem::take(&mut text)));
            }
            let inner = std::str::from_utf8(&b[start..inner_end])
                .map_err(|_| Error::Syntax("non-UTF-8 in a tag".into(), i))?
                .trim()
                .to_string();
            match kind {
                // A comment produces nothing, which is also what it means.
                b'#' => {}
                b'{' => out.push(Tok::Out(inner, i)),
                _ => out.push(Tok::Tag(inner, i)),
            }
            trim_next = trim_after;
            i = end + 2;
            continue;
        }
        let ch = src[i..].chars().next().unwrap();
        if trim_next && ch.is_whitespace() {
            i += ch.len_utf8();
            continue;
        }
        trim_next = false;
        text.push(ch);
        i += ch.len_utf8();
    }
    if !text.is_empty() {
        out.push(Tok::Text(text));
    }
    Ok(out)
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    (from..hay.len().saturating_sub(needle.len() - 1))
        .find(|&k| &hay[k..k + needle.len()] == needle)
}

// --------------------------------------------------------------------- parsing --

struct P<'a> {
    toks: &'a [Tok],
    i: usize,
    rewrites: Vec<String>,
}

/// Translates a Jinja2 chat template into OMNI-CT.
pub fn translate(src: &str) -> Res<Translated> {
    if src.len() > MAX_SOURCE {
        return Err(Error::Syntax(
            format!("{} bytes exceeds the {MAX_SOURCE}-byte limit", src.len()),
            0,
        ));
    }
    let toks = lex(src)?;
    let mut p = P {
        toks: &toks,
        i: 0,
        rewrites: Vec::new(),
    };
    let nodes = p.block(&[])?;
    if p.i < p.toks.len() {
        if let Tok::Tag(t, at) = &p.toks[p.i] {
            return Err(Error::Syntax(format!("unexpected `{{% {t} %}}`"), *at));
        }
    }
    // Round-trips through OMNI-CT's own parser rather than being handed back as
    // an AST: the printed form is what goes in the container, so if it does not
    // parse the translation is not usable and saying so here is better than
    // discovering it at render time.
    let printed = print(&nodes);
    let template = Template::parse(&printed).map_err(|e| {
        Error::Syntax(
            format!("the translation does not parse as OMNI-CT: {e} — in `{printed}`"),
            0,
        )
    })?;
    Ok(Translated {
        template,
        rewrites: p.rewrites,
    })
}

impl P<'_> {
    fn block(&mut self, stop: &[&str]) -> Res<Vec<Node>> {
        let mut out = Vec::new();
        while self.i < self.toks.len() {
            match &self.toks[self.i] {
                Tok::Text(t) => {
                    out.push(Node::Text(t.clone()));
                    self.i += 1;
                }
                Tok::Out(e, at) => {
                    let ex = self.expr(e, *at)?;
                    out.push(Node::Out(ex));
                    self.i += 1;
                }
                Tok::Tag(t, at) => {
                    let head = t.split_whitespace().next().unwrap_or("");
                    if stop.contains(&head) {
                        return Ok(out);
                    }
                    let at = *at;
                    let t = t.clone();
                    self.i += 1;
                    out.push(self.tag(&t, head, at)?);
                }
            }
        }
        Ok(out)
    }

    fn tag(&mut self, whole: &str, head: &str, at: usize) -> Res<Node> {
        match head {
            "if" => {
                let cond = self.expr(whole[2..].trim(), at)?;
                let mut arms = vec![(cond, self.block(&["elif", "else", "endif"])?)];
                let mut otherwise = Vec::new();
                loop {
                    let Some(Tok::Tag(t, tat)) = self.toks.get(self.i) else {
                        return Err(Error::Syntax("unclosed `{% if %}`".into(), at));
                    };
                    let h = t.split_whitespace().next().unwrap_or("");
                    let (t, tat) = (t.clone(), *tat);
                    self.i += 1;
                    match h {
                        "elif" => {
                            let c = self.expr(t[4..].trim(), tat)?;
                            arms.push((c, self.block(&["elif", "else", "endif"])?));
                        }
                        "else" => {
                            otherwise = self.block(&["endif"])?;
                        }
                        "endif" => break,
                        _ => return Err(Error::Syntax(format!("unexpected `{t}`"), tat)),
                    }
                }
                Ok(Node::If { arms, otherwise })
            }
            "for" => {
                // `for x in y` only. `for k, v in m.items()` needs tuple
                // unpacking, which OMNI-CT's `for` does not have.
                let rest = whole[3..].trim();
                let Some((var, iter)) = rest.split_once(" in ") else {
                    return Err(Error::Syntax("`for` without `in`".into(), at));
                };
                let var = var.trim();
                if var.contains(',') {
                    return Err(unsupported(
                        "for x, y in …",
                        "OMNI-CT's `for` binds one variable; tuple unpacking would \
                         need a destructuring form §06.9 does not have",
                        at,
                    ));
                }
                if !is_name(var) {
                    return Err(Error::Syntax(format!("`{var}` is not a loop variable"), at));
                }
                let iter = self.expr(iter.trim(), at)?;
                let body = self.block(&["else", "endfor"])?;
                let mut otherwise = Vec::new();
                let Some(Tok::Tag(t, tat)) = self.toks.get(self.i) else {
                    return Err(Error::Syntax("unclosed `{% for %}`".into(), at));
                };
                let (t, tat) = (t.clone(), *tat);
                self.i += 1;
                if t.starts_with("else") {
                    otherwise = self.block(&["endfor"])?;
                    let Some(Tok::Tag(e, _)) = self.toks.get(self.i) else {
                        return Err(Error::Syntax("unclosed `{% for %}`".into(), at));
                    };
                    if !e.starts_with("endfor") {
                        return Err(Error::Syntax(format!("unexpected `{e}`"), tat));
                    }
                    self.i += 1;
                } else if !t.starts_with("endfor") {
                    return Err(Error::Syntax(format!("unexpected `{t}`"), tat));
                }
                Ok(Node::For {
                    var: var.to_string(),
                    iter,
                    body,
                    otherwise,
                })
            }
            "set" => {
                let rest = whole[3..].trim();
                let Some((name, value)) = rest.split_once('=') else {
                    // `{% set x %}…{% endset %}` is a block assignment.
                    return Err(unsupported(
                        "set … endset",
                        "a block assignment captures rendered output into a \
                         variable; OMNI-CT's `set` takes an expression",
                        at,
                    ));
                };
                let name = name.trim();
                if !is_name(name) {
                    return Err(unsupported(
                        "set a.b = …",
                        "assignment into a field mutates a structure; OMNI-CT's \
                         `set` binds a name",
                        at,
                    ));
                }
                Ok(Node::Set {
                    name: name.to_string(),
                    value: self.expr(value.trim(), at)?,
                })
            }
            "macro" | "endmacro" | "call" | "filter" | "include" | "import" | "extends"
            | "block" | "endblock" | "raw" | "endraw" | "with" | "endwith" | "do" => {
                Err(unsupported(
                    head,
                    match head {
                        "macro" | "endmacro" | "call" => {
                            "a macro is a function, and a function can recurse — \
                             which is exactly what makes a template non-total (§06.9)"
                        }
                        "include" | "import" | "extends" => {
                            "pulling in another template makes rendering depend on \
                             files outside the container"
                        }
                        "raw" | "endraw" => {
                            "raw blocks are a lexer feature this translator does \
                             not implement; the contents would need escaping"
                        }
                        _ => "no OMNI-CT equivalent (§06.9's grammar is closed)",
                    },
                    at,
                ))
            }
            _ => Err(Error::Syntax(format!("unknown tag `{head}`"), at)),
        }
    }

    // ------------------------------------------------------------ expressions --

    fn expr(&mut self, src: &str, at: usize) -> Res<Expr> {
        let mut e = E {
            s: src.as_bytes(),
            i: 0,
            at,
            rewrites: &mut self.rewrites,
        };
        let out = e.ternary()?;
        e.ws();
        if e.i < e.s.len() {
            return Err(Error::Syntax(
                format!("trailing `{}` in expression", &src[e.i..]),
                at,
            ));
        }
        Ok(out)
    }
}

fn unsupported(construct: &str, reason: &str, at: usize) -> Error {
    Error::Unsupported(Refusal {
        construct: construct.to_string(),
        reason: reason.to_string(),
        at,
    })
}

fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

struct E<'a> {
    s: &'a [u8],
    i: usize,
    at: usize,
    rewrites: &'a mut Vec<String>,
}

impl E<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        self.ws();
        let w = word.as_bytes();
        if self.i + w.len() > self.s.len() || &self.s[self.i..self.i + w.len()] != w {
            return false;
        }
        // A word operator must not be the prefix of an identifier: `internal`
        // starts with `in`.
        let alpha = w[0].is_ascii_alphabetic();
        if alpha {
            let after = self.s.get(self.i + w.len()).copied().unwrap_or(b' ');
            if after.is_ascii_alphanumeric() || after == b'_' {
                return false;
            }
        }
        self.i += w.len();
        true
    }

    fn err<T>(&self, m: impl Into<String>) -> Res<T> {
        Err(Error::Syntax(m.into(), self.at))
    }

    /// `a if c else b`, the lowest-precedence form.
    fn ternary(&mut self) -> Res<Expr> {
        let then = self.or()?;
        if self.eat("if") {
            let cond = self.or()?;
            if !self.eat("else") {
                // Jinja allows `a if c` with an implicit undefined; OMNI-CT has
                // no undefined, so there is nothing to translate it to.
                return Err(unsupported(
                    "a if c",
                    "a conditional without `else` yields Jinja's undefined, which \
                     OMNI-CT does not have",
                    self.at,
                ));
            }
            let otherwise = self.ternary()?;
            return Ok(Expr::Cond {
                cond: Box::new(cond),
                then: Box::new(then),
                otherwise: Box::new(otherwise),
            });
        }
        Ok(then)
    }

    fn or(&mut self) -> Res<Expr> {
        let mut l = self.and()?;
        while self.eat("or") {
            let r = self.and()?;
            l = bin(BinOp::Or, l, r);
        }
        Ok(l)
    }

    fn and(&mut self) -> Res<Expr> {
        let mut l = self.not()?;
        while self.eat("and") {
            let r = self.not()?;
            l = bin(BinOp::And, l, r);
        }
        Ok(l)
    }

    fn not(&mut self) -> Res<Expr> {
        if self.eat("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.compare()
    }

    fn compare(&mut self) -> Res<Expr> {
        let l = self.concat()?;
        // `is` tests come first: `x is defined` binds tighter than a comparison
        // would and is the form chat templates actually use.
        if self.eat("is") {
            let negated = self.eat("not");
            let name = self.ident()?;
            let t = self.test(&name, l, self.at)?;
            return Ok(if negated { Expr::Not(Box::new(t)) } else { t });
        }
        for (word, op) in [
            ("==", BinOp::Eq),
            ("!=", BinOp::Ne),
            ("<=", BinOp::Le),
            (">=", BinOp::Ge),
            ("<", BinOp::Lt),
            (">", BinOp::Gt),
        ] {
            if self.eat(word) {
                let r = self.concat()?;
                return Ok(bin(op, l, r));
            }
        }
        if self.eat("not") {
            if !self.eat("in") {
                return self.err("expected `in` after `not`");
            }
            let r = self.concat()?;
            return Ok(bin(BinOp::NotIn, l, r));
        }
        if self.eat("in") {
            let r = self.concat()?;
            return Ok(bin(BinOp::In, l, r));
        }
        Ok(l)
    }

    /// A Jinja `is <test>`, mapped onto an OMNI-CT expression where one exists.
    fn test(&mut self, name: &str, l: Expr, at: usize) -> Res<Expr> {
        Ok(match name {
            // OMNI-CT has no undefined, so "is defined" is "is not null" — which
            // is what a chat template means by it, since the variable is either
            // passed or it is not.
            "defined" => {
                self.rewrites
                    .push("`is defined` became a null check: OMNI-CT has no undefined".into());
                bin(BinOp::Ne, l, Expr::Null)
            }
            "undefined" | "none" => bin(BinOp::Eq, l, Expr::Null),
            "string" | "number" | "mapping" | "sequence" | "iterable" | "boolean" => {
                return Err(unsupported(
                    &format!("is {name}"),
                    "a type test needs runtime type introspection; OMNI-CT's values \
                     are typed but not reflective",
                    at,
                ))
            }
            other => {
                return Err(unsupported(
                    &format!("is {other}"),
                    "not a test this translator knows",
                    at,
                ))
            }
        })
    }

    fn concat(&mut self) -> Res<Expr> {
        let mut l = self.add()?;
        while self.eat("~") {
            let r = self.add()?;
            l = bin(BinOp::Concat, l, r);
        }
        Ok(l)
    }

    fn add(&mut self) -> Res<Expr> {
        let mut l = self.mul()?;
        loop {
            if self.eat("+") {
                let r = self.mul()?;
                l = bin(BinOp::Add, l, r);
            } else if self.eat("-") {
                let r = self.mul()?;
                l = bin(BinOp::Sub, l, r);
            } else {
                return Ok(l);
            }
        }
    }

    fn mul(&mut self) -> Res<Expr> {
        let mut l = self.unary()?;
        loop {
            if self.eat("*") {
                let r = self.unary()?;
                l = bin(BinOp::Mul, l, r);
            } else if self.eat("//") {
                // Floor division, which is what OMNI-CT's `/` is: its value
                // domain has integers and no floats.
                let r = self.unary()?;
                l = bin(BinOp::Div, l, r);
            } else if self.eat("/") {
                // Jinja's `/` is *true* division: `5 / 2` is 2.5, and §06.9's
                // value domain has no float to hold that. Mapping it onto
                // integer division would silently turn 2.5 into 2, which is the
                // kind of quiet difference this whole translator exists to
                // avoid. `//` is right there and means what OMNI-CT can do.
                return Err(unsupported(
                    "a / b",
                    "Jinja's `/` is true division and yields a float; §06.9's \
                     values are strings, integers, booleans, lists and maps. Use \
                     `//` for the integer division OMNI-CT has",
                    self.at,
                ));
            } else if self.eat("%") {
                let r = self.unary()?;
                l = bin(BinOp::Rem, l, r);
            } else {
                return Ok(l);
            }
        }
    }

    fn unary(&mut self) -> Res<Expr> {
        self.ws();
        if self.eat("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.postfix()
    }

    /// A primary followed by any number of `.name`, `[expr]`, `(args)` and
    /// `| filter` suffixes.
    fn postfix(&mut self) -> Res<Expr> {
        let mut e = self.primary()?;
        loop {
            self.ws();
            if self.i < self.s.len() && self.s[self.i] == b'.' {
                self.i += 1;
                let name = self.ident()?;
                self.ws();
                // A `.name(` is a Python method call, not a field.
                if self.i < self.s.len() && self.s[self.i] == b'(' {
                    let args = self.args()?;
                    e = self.method(e, &name, args)?;
                } else {
                    e = Expr::Field(Box::new(e), name);
                }
                continue;
            }
            if self.i < self.s.len() && self.s[self.i] == b'[' {
                self.i += 1;
                let idx = self.ternary()?;
                self.ws();
                if self.i >= self.s.len() || self.s[self.i] != b']' {
                    // A slice: `messages[1:]` is common and OMNI-CT has no
                    // slicing form.
                    if self.i < self.s.len() && self.s[self.i] == b':' {
                        return Err(unsupported(
                            "a[b:c]",
                            "slicing a list needs a slice form §06.9 does not have",
                            self.at,
                        ));
                    }
                    return self.err("expected `]`");
                }
                self.i += 1;
                e = Expr::Index(Box::new(e), Box::new(idx));
                continue;
            }
            if self.i < self.s.len() && self.s[self.i] == b'|' {
                self.i += 1;
                let name = self.ident()?;
                self.ws();
                let mut args = if self.i < self.s.len() && self.s[self.i] == b'(' {
                    self.args()?
                } else {
                    Vec::new()
                };
                args.insert(0, e);
                e = self.filter(&name, args)?;
                continue;
            }
            return Ok(e);
        }
    }

    /// Jinja filters, mapped onto the closed OMNI-CT standard library.
    fn filter(&mut self, name: &str, args: Vec<Expr>) -> Res<Expr> {
        let direct = [
            "upper", "lower", "trim", "join", "default", "tojson", "length", "first", "last",
            "replace", "split", "string", "int", "abs",
        ];
        if direct.contains(&name) {
            return Ok(Expr::Call {
                name: name.to_string(),
                args,
            });
        }
        Ok(match name {
            // Aliases with identical meaning.
            "striptags" | "e" | "escape" | "safe" => {
                self.rewrites.push(format!(
                    "`| {name}` dropped: OMNI-CT does not escape output"
                ));
                args.into_iter().next().unwrap_or(Expr::Null)
            }
            "count" => Expr::Call {
                name: "length".into(),
                args,
            },
            "list" => args.into_iter().next().unwrap_or(Expr::Null),
            "capitalize" | "title" | "sort" | "reverse" | "unique" | "map" | "select"
            | "reject" | "selectattr" | "rejectattr" | "groupby" | "batch" | "slice" | "round"
            | "sum" | "min" | "max" | "indent" | "wordwrap" | "truncate" => {
                return Err(unsupported(
                    &format!("| {name}"),
                    "no OMNI-CT equivalent; §06.9's standard library is closed and \
                     every entry in it is pure and total",
                    self.at,
                ))
            }
            other => {
                return Err(unsupported(
                    &format!("| {other}"),
                    "not a filter this translator knows",
                    self.at,
                ))
            }
        })
    }

    /// Python method calls on a value. Two have exact equivalents; the rest do
    /// not, and approximating them would change what a prompt says.
    fn method(&mut self, recv: Expr, name: &str, mut args: Vec<Expr>) -> Res<Expr> {
        args.insert(0, recv);
        Ok(match name {
            "strip" if args.len() == 1 => {
                self.rewrites.push("`.strip()` became `trim(…)`".into());
                Expr::Call {
                    name: "trim".into(),
                    args,
                }
            }
            "split" if args.len() == 2 => Expr::Call {
                name: "split".into(),
                args,
            },
            "upper" if args.len() == 1 => Expr::Call {
                name: "upper".into(),
                args,
            },
            "lower" if args.len() == 1 => Expr::Call {
                name: "lower".into(),
                args,
            },
            "replace" if args.len() == 3 => Expr::Call {
                name: "replace".into(),
                args,
            },
            "get" if args.len() >= 2 => {
                // Python's two-argument `get` returns None when absent; OMNI-CT's
                // takes the fallback explicitly, so the missing one is written.
                if args.len() == 2 {
                    args.push(Expr::Null);
                    self.rewrites.push(
                        "`.get(k)` became `get(m, k, null)`: the fallback is explicit".into(),
                    );
                }
                Expr::Call {
                    name: "get".into(),
                    args,
                }
            }
            "items" | "keys" | "values" if args.len() == 1 => {
                if name == "items" {
                    return Err(unsupported(
                        ".items()",
                        "iterating a map's pairs needs tuple unpacking in the `for`, \
                         which OMNI-CT does not have",
                        self.at,
                    ));
                }
                Expr::Call {
                    name: name.to_string(),
                    args,
                }
            }
            other => {
                return Err(unsupported(
                    &format!(".{other}()"),
                    "a Python method on a host value; OMNI-CT calls only its own \
                     closed standard library",
                    self.at,
                ))
            }
        })
    }

    fn args(&mut self) -> Res<Vec<Expr>> {
        // Assumes the caller checked for `(`.
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.i < self.s.len() && self.s[self.i] == b')' {
            self.i += 1;
            return Ok(out);
        }
        loop {
            // A keyword argument would need named parameters, which the closed
            // standard library does not have.
            let save = self.i;
            if let Ok(name) = self.ident() {
                self.ws();
                if self.i < self.s.len()
                    && self.s[self.i] == b'='
                    && self.s.get(self.i + 1) != Some(&b'=')
                {
                    return Err(unsupported(
                        &format!("{name}=…"),
                        "a keyword argument; OMNI-CT's standard library takes \
                         positional arguments only",
                        self.at,
                    ));
                }
            }
            self.i = save;
            out.push(self.ternary()?);
            self.ws();
            if self.i < self.s.len() && self.s[self.i] == b',' {
                self.i += 1;
                continue;
            }
            if self.i < self.s.len() && self.s[self.i] == b')' {
                self.i += 1;
                return Ok(out);
            }
            return self.err("expected `,` or `)`");
        }
    }

    fn ident(&mut self) -> Res<String> {
        self.ws();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
        {
            self.i += 1;
        }
        if self.i == start {
            return self.err("expected a name");
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    fn primary(&mut self) -> Res<Expr> {
        self.ws();
        if self.i >= self.s.len() {
            return self.err("expected an expression");
        }
        let c = self.s[self.i];
        if c == b'(' {
            self.i += 1;
            let e = self.ternary()?;
            self.ws();
            if self.i < self.s.len() && self.s[self.i] == b',' {
                return Err(unsupported(
                    "(a, b)",
                    "a tuple; OMNI-CT has lists and maps and no tuple type",
                    self.at,
                ));
            }
            if self.i >= self.s.len() || self.s[self.i] != b')' {
                return self.err("expected `)`");
            }
            self.i += 1;
            return Ok(e);
        }
        if c == b'[' {
            self.i += 1;
            let mut items = Vec::new();
            self.ws();
            if self.i < self.s.len() && self.s[self.i] == b']' {
                self.i += 1;
                return Ok(Expr::List(items));
            }
            loop {
                items.push(self.ternary()?);
                self.ws();
                if self.i < self.s.len() && self.s[self.i] == b',' {
                    self.i += 1;
                    self.ws();
                    if self.i < self.s.len() && self.s[self.i] == b']' {
                        self.i += 1;
                        return Ok(Expr::List(items));
                    }
                    continue;
                }
                if self.i < self.s.len() && self.s[self.i] == b']' {
                    self.i += 1;
                    return Ok(Expr::List(items));
                }
                return self.err("expected `,` or `]`");
            }
        }
        if c == b'{' {
            return Err(unsupported(
                "{…}",
                "a dict literal; OMNI-CT reads maps from its input and does not \
                 construct them",
                self.at,
            ));
        }
        if c == b'\'' || c == b'"' {
            return self.string(c);
        }
        if c.is_ascii_digit() {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i < self.s.len() && self.s[self.i] == b'.' {
                return Err(unsupported(
                    "a float literal",
                    "§06.9's value domain is strings, integers, booleans, lists \
                     and maps — a float would render differently in different \
                     implementations",
                    self.at,
                ));
            }
            let text = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
            return Ok(Expr::Int(text.parse().map_err(|_| {
                Error::Syntax(format!("`{text}` is not an integer"), self.at)
            })?));
        }
        let name = self.ident()?;
        self.ws();
        // A bare `name(` is a global function call. A name that has no OMNI-CT
        // form is refused here, before its arguments are parsed: `namespace(x=1)`
        // should be refused for being a namespace, not for the keyword argument
        // inside it, and the caller can only act on the outer reason.
        if self.i < self.s.len() && self.s[self.i] == b'(' {
            if !crate::ct::STDLIB.iter().any(|(n, _)| *n == name) {
                return self.global(&name, Vec::new());
            }
            let args = self.args()?;
            return self.global(&name, args);
        }
        Ok(match name.as_str() {
            "true" | "True" => Expr::Bool(true),
            "false" | "False" => Expr::Bool(false),
            "none" | "None" => Expr::Null,
            // `loop` is the loop variable Jinja binds inside `{% for %}`.
            // OMNI-CT's `for` binds none, so every use of it is a refusal — and
            // this is the most common one by a wide margin.
            "loop" => {
                return Err(unsupported(
                    "loop.*",
                    "OMNI-CT's `for` has no loop variable, so `loop.index0`, \
                     `loop.first` and `loop.last` have nothing to translate to. \
                     This is a gap in §06.9's grammar rather than in the template",
                    self.at,
                ))
            }
            _ => Expr::Var(name),
        })
    }

    /// A bare `name(...)` call. Every one of these is a refusal today, with a
    /// different reason each time — which is the useful part, because the reasons
    /// are what a template author or a §06.9 editor would act on.
    fn global(&mut self, name: &str, args: Vec<Expr>) -> Res<Expr> {
        // A name in OMNI-CT's own standard library is a call to it. Jinja
        // templates write these as filters, but this module's *printed* form
        // writes them as calls, so refusing them would mean a translation that
        // does not round-trip through its own printer.
        if crate::ct::STDLIB.iter().any(|(n, _)| *n == name) {
            return Ok(Expr::Call {
                name: name.to_string(),
                args,
            });
        }
        Err(match name {
            // Every HF template's error path. OMNI-CT is total and has no way to
            // fail deliberately, so the branch cannot be carried across.
            "raise_exception" => unsupported(
                "raise_exception(…)",
                "a template that raises is asserting something about its \
                     input; OMNI-CT is total and has no failure form, so the \
                     assertion would be silently dropped",
                self.at,
            ),
            "namespace" => unsupported(
                "namespace(…)",
                "a namespace exists so a loop body can mutate state that \
                     outlives it, which is exactly the feedback OMNI-CT's `set` \
                     is scoped to prevent",
                self.at,
            ),
            "range" => unsupported(
                "range(…)",
                "iterating a computed range rather than a finite structure \
                     already in memory; §06.9's `for` takes the latter",
                self.at,
            ),
            "strftime_now" => unsupported(
                "strftime_now(…)",
                "reading the clock makes rendering non-deterministic; \
                     OMNI-CT's `strftime` takes the epoch as an argument, so pass \
                     the time in",
                self.at,
            ),
            other => unsupported(
                &format!("{other}(…)"),
                "a global function; OMNI-CT calls only its own closed \
                     standard library",
                self.at,
            ),
        })
    }

    fn string(&mut self, quote: u8) -> Res<Expr> {
        self.i += 1;
        let mut out = String::new();
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c == quote {
                self.i += 1;
                return Ok(Expr::Str(out));
            }
            if c == b'\\' && self.i + 1 < self.s.len() {
                let e = self.s[self.i + 1];
                out.push(match e {
                    b'n' => '\n',
                    b't' => '\t',
                    b'r' => '\r',
                    b'\\' => '\\',
                    b'\'' => '\'',
                    b'"' => '"',
                    other => other as char,
                });
                self.i += 2;
                continue;
            }
            // Multi-byte characters pass through whole.
            let rest = String::from_utf8_lossy(&self.s[self.i..]);
            let ch = rest.chars().next().unwrap_or('\u{fffd}');
            out.push(ch);
            self.i += ch.len_utf8();
        }
        self.err("unterminated string")
    }
}

fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
    Expr::Bin {
        op,
        l: Box::new(l),
        r: Box::new(r),
    }
}

// -------------------------------------------------------------------- printing --

/// Prints an OMNI-CT AST back to source.
///
/// The translation goes out as *text* rather than as an AST because that is what
/// a `ChatTemplate` object holds, and because a human reviewing a converted model
/// should be able to read what their template became.
pub fn print(nodes: &[Node]) -> String {
    let mut out = String::new();
    for n in nodes {
        print_node(n, &mut out);
    }
    out
}

fn print_node(n: &Node, out: &mut String) {
    match n {
        Node::Text(t) => out.push_str(t),
        Node::Out(e) => {
            out.push_str("{{ ");
            print_expr(e, out);
            out.push_str(" }}");
        }
        Node::Set { name, value } => {
            out.push_str("{% set ");
            out.push_str(name);
            out.push_str(" = ");
            print_expr(value, out);
            out.push_str(" %}");
        }
        Node::If { arms, otherwise } => {
            for (k, (cond, body)) in arms.iter().enumerate() {
                out.push_str(if k == 0 { "{% if " } else { "{% elif " });
                print_expr(cond, out);
                out.push_str(" %}");
                for b in body {
                    print_node(b, out);
                }
            }
            if !otherwise.is_empty() {
                out.push_str("{% else %}");
                for b in otherwise {
                    print_node(b, out);
                }
            }
            out.push_str("{% endif %}");
        }
        Node::For {
            var,
            iter,
            body,
            otherwise,
        } => {
            out.push_str("{% for ");
            out.push_str(var);
            out.push_str(" in ");
            print_expr(iter, out);
            out.push_str(" %}");
            for b in body {
                print_node(b, out);
            }
            if !otherwise.is_empty() {
                out.push_str("{% else %}");
                for b in otherwise {
                    print_node(b, out);
                }
            }
            out.push_str("{% endfor %}");
        }
    }
}

fn print_expr(e: &Expr, out: &mut String) {
    match e {
        Expr::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    other => out.push(other),
                }
            }
            out.push('"');
        }
        Expr::Int(n) => out.push_str(&n.to_string()),
        Expr::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Expr::Null => out.push_str("none"),
        Expr::Var(v) => out.push_str(v),
        Expr::List(xs) => {
            out.push('[');
            for (k, x) in xs.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                print_expr(x, out);
            }
            out.push(']');
        }
        Expr::Field(b, name) => {
            print_expr(b, out);
            out.push('.');
            out.push_str(name);
        }
        Expr::Index(b, i) => {
            print_expr(b, out);
            out.push('[');
            print_expr(i, out);
            out.push(']');
        }
        // Parenthesized without exception: the printed form has to reparse to
        // the same tree, and guessing which parentheses are redundant is how a
        // round trip stops being one.
        Expr::Not(x) => {
            out.push_str("(not ");
            print_expr(x, out);
            out.push(')');
        }
        Expr::Neg(x) => {
            out.push_str("(-");
            print_expr(x, out);
            out.push(')');
        }
        Expr::Bin { op, l, r } => {
            out.push('(');
            print_expr(l, out);
            out.push(' ');
            out.push_str(op_text(*op));
            out.push(' ');
            print_expr(r, out);
            out.push(')');
        }
        Expr::Cond {
            cond,
            then,
            otherwise,
        } => {
            out.push('(');
            print_expr(then, out);
            out.push_str(" if ");
            print_expr(cond, out);
            out.push_str(" else ");
            print_expr(otherwise, out);
            out.push(')');
        }
        Expr::Call { name, args } => {
            out.push_str(name);
            out.push('(');
            for (k, a) in args.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                print_expr(a, out);
            }
            out.push(')');
        }
    }
}

fn op_text(op: BinOp) -> &'static str {
    match op {
        BinOp::Or => "or",
        BinOp::And => "and",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::In => "in",
        BinOp::NotIn => "not in",
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Concat => "~",
    }
}

// -------------------------------------------------------------- the corpus --

/// One template in the corpus [`coverage`] measures against.
pub struct Sample {
    /// The model family whose template this is.
    pub name: &'static str,
    pub source: &'static str,
}

/// Chat templates from real model families, written out in the syntax those
/// families publish.
///
/// **This is not the hub snapshot Gate 2 asks for.** It is a small corpus, so
/// the percentage it produces is a percentage of *this*, not of the hub, and
/// `coverage` says so in as many words. What it is good for is the thing a
/// percentage cannot do on its own: it names which construct blocks which
/// family, and those names are the same ones a hub snapshot would produce.
pub const CORPUS: &[Sample] = &[
    Sample {
        name: "chatml",
        source: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + \
                 message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}\
                 {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
    },
    Sample {
        name: "vicuna",
        source: "{{ system }}{% for message in messages %}{% if message['role'] == 'user' %}\
                 {{ 'USER: ' + message['content'] + '\\n' }}{% else %}\
                 {{ 'ASSISTANT: ' + message['content'] + '</s>\\n' }}{% endif %}{% endfor %}\
                 {% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
    },
    Sample {
        name: "alpaca",
        source: "{{ 'Below is an instruction.\\n\\n' }}{% for message in messages %}\
                 {% if message['role'] == 'user' %}{{ '### Instruction:\\n' + \
                 message['content'] + '\\n\\n' }}{% else %}{{ '### Response:\\n' + \
                 message['content'] + '\\n\\n' }}{% endif %}{% endfor %}",
    },
    Sample {
        name: "zephyr",
        source: "{% for message in messages %}\\n{% if message['role'] == 'user' %}\
                 {{ '<|user|>\\n' + message['content'] + eos_token }}\
                 {% elif message['role'] == 'system' %}\
                 {{ '<|system|>\\n' + message['content'] + eos_token }}\
                 {% else %}{{ '<|assistant|>\\n' + message['content'] + eos_token }}\
                 {% endif %}{% if loop.last and add_generation_prompt %}\
                 {{ '<|assistant|>' }}{% endif %}{% endfor %}",
    },
    Sample {
        name: "llama-2",
        source: "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}\
                 {% set system_message = messages[0]['content'] %}{% else %}\
                 {% set loop_messages = messages %}{% endif %}\
                 {% for message in loop_messages %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
                 {% endfor %}",
    },
    Sample {
        name: "llama-3",
        source: "{% set loop_messages = messages %}{% for message in loop_messages %}\
                 {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' \
                 + message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}\
                 {% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\
                 {% endif %}",
    },
    Sample {
        name: "mistral",
        source: "{{ bos_token }}{% for message in messages %}\
                 {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
                 {% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}\
                 {% endif %}{% endfor %}",
    },
    Sample {
        name: "gemma",
        source: "{% for message in messages %}\
                 {{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] | trim + \
                 '<end_of_turn>\\n' }}{% endfor %}\
                 {% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}",
    },
    Sample {
        name: "phi-3",
        source: "{% for message in messages %}{% if message['role'] == 'system' %}\
                 {{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}\
                 {% elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + \
                 '<|end|>\\n' }}{% else %}{{ '<|assistant|>\\n' + message['content'] + \
                 '<|end|>\\n' }}{% endif %}{% endfor %}\
                 {% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}",
    },
    Sample {
        name: "falcon",
        source: "{% for message in messages %}{{ message['role'] | capitalize + ': ' + \
                 message['content'] + '\\n' }}{% endfor %}",
    },
    Sample {
        name: "openchat",
        source: "{{ bos_token }}{% for message in messages %}\
                 {{ 'GPT4 Correct ' + message['role'] | title + ': ' + message['content'] + \
                 '<|end_of_turn|>' }}{% endfor %}",
    },
    Sample {
        name: "strict-alternation",
        source: "{% for message in messages %}\
                 {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\
                 {{ raise_exception('Conversation roles must alternate') }}{% endif %}\
                 {{ message['content'] }}{% endfor %}",
    },
    Sample {
        name: "tool-calling",
        source: "{% for message in messages %}{% if message['role'] == 'tool' %}\
                 {{ '<tool_response>' + message['content'] | tojson + '</tool_response>' }}\
                 {% else %}{{ message['content'] }}{% endif %}{% endfor %}",
    },
    Sample {
        name: "system-default",
        source: "{% set system = messages[0]['content'] if messages[0]['role'] == 'system' \
                 else 'You are a helpful assistant.' %}{{ system }}\
                 {% for message in messages %}{{ message['content'] }}{% endfor %}",
    },
    Sample {
        name: "optional-system",
        source: "{% if system_prompt is defined %}{{ system_prompt }}{% endif %}\
                 {% for message in messages %}{{ message['content'] | default('') }}{% endfor %}",
    },
];

/// What translating the whole corpus produced.
pub struct Coverage {
    pub total: usize,
    pub translated: Vec<&'static str>,
    /// Family, and why it could not be translated.
    pub refused: Vec<(&'static str, String)>,
}

impl Coverage {
    pub fn rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.translated.len() as f64 / self.total as f64
    }

    /// The constructs that blocked a translation, most frequent first. This is
    /// the output worth acting on: a construct that blocks four families is a
    /// case for growing §06.9's grammar, and one that blocks one is a case for
    /// changing that template.
    pub fn blockers(&self) -> Vec<(String, usize)> {
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for (_, why) in &self.refused {
            let construct = why
                .split_once('`')
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(c, _)| c.to_string())
                .unwrap_or_else(|| why.clone());
            *counts.entry(construct).or_default() += 1;
        }
        let mut v: Vec<(String, usize)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }
}

/// Translates [`CORPUS`] and reports what happened.
pub fn coverage() -> Coverage {
    let mut translated = Vec::new();
    let mut refused = Vec::new();
    for s in CORPUS {
        match translate(s.source) {
            Ok(_) => translated.push(s.name),
            Err(e) => refused.push((s.name, e.to_string())),
        }
    }
    Coverage {
        total: CORPUS.len(),
        translated,
        refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cbor::Value;

    fn msg(role: &str, content: &str) -> Value {
        Value::map(vec![
            ("content", Value::text(content)),
            ("role", Value::text(role)),
        ])
    }

    fn render(src: &str, input: Value) -> String {
        let t = translate(src).unwrap_or_else(|e| panic!("did not translate: {e}"));
        t.template
            .render(&input)
            .unwrap_or_else(|e| panic!("did not render: {e}"))
    }

    #[test]
    fn a_chatml_template_renders_what_jinja_would() {
        // The test that matters: not that it translates, but that the
        // translation produces the same string. Worked out by hand from the
        // template, since the point is to check the translation and not to check
        // it against itself.
        let src = CORPUS.iter().find(|s| s.name == "chatml").unwrap().source;
        let input = Value::map(vec![
            ("add_generation_prompt", Value::Bool(true)),
            (
                "messages",
                Value::Array(vec![msg("user", "hi"), msg("assistant", "hello")]),
            ),
        ]);
        assert_eq!(
            render(src, input),
            "<|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\nhello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn whitespace_control_is_carried_across() {
        // `{%- … -%}` is not cosmetic: it decides whether a prompt has a leading
        // newline, and a tokenizer notices.
        assert_eq!(
            render("a\n  {%- if true -%}  \nb{%- endif %}", Value::map(vec![])),
            "ab"
        );
        // And without the markers the whitespace stays.
        assert_eq!(
            render("a\n  {% if true %}  \nb{% endif %}", Value::map(vec![])),
            "a\n    \nb"
        );
    }

    #[test]
    fn the_filters_and_methods_with_exact_equivalents_are_rewritten() {
        let input = Value::map(vec![("s", Value::text("  Hi  "))]);
        assert_eq!(render("{{ s | trim }}", input.clone()), "Hi");
        assert_eq!(render("{{ s.strip() }}", input.clone()), "Hi");
        assert_eq!(render("{{ s | upper | trim }}", input.clone()), "HI");
        // The rewrite is reported rather than done silently, so a reviewer can
        // see which lines are not literally what the author wrote.
        let t = translate("{{ s.strip() }}").unwrap();
        assert!(
            t.rewrites.iter().any(|r| r.contains("trim")),
            "{:?}",
            t.rewrites
        );
        // `is defined` has no exact equivalent — OMNI-CT has no undefined — so
        // it becomes a null check, and that is recorded too.
        let t = translate("{% if x is defined %}y{% endif %}").unwrap();
        assert!(
            t.rewrites.iter().any(|r| r.contains("undefined")),
            "{:?}",
            t.rewrites
        );
        assert_eq!(
            render(
                "{% if x is defined %}{{ x }}{% else %}none{% endif %}",
                Value::map(vec![("x", Value::Null)])
            ),
            "none"
        );
    }

    #[test]
    fn a_comment_renders_nothing_and_a_conditional_picks_its_arm() {
        assert_eq!(render("a{# a note #}b", Value::map(vec![])), "ab");
        let pick = |v: bool| {
            render(
                "{% if flag %}yes{% elif other %}maybe{% else %}no{% endif %}",
                Value::map(vec![
                    ("flag", Value::Bool(v)),
                    ("other", Value::Bool(false)),
                ]),
            )
        };
        assert_eq!(pick(true), "yes");
        assert_eq!(pick(false), "no");
        // The ternary form too.
        assert_eq!(
            render(
                "{{ 'y' if flag else 'n' }}",
                Value::map(vec![("flag", Value::Bool(false))])
            ),
            "n"
        );
    }

    #[test]
    fn what_cannot_be_translated_is_named_with_its_position() {
        for (src, construct) in [
            ("{% macro x() %}{% endmacro %}", "macro"),
            ("{% include 'other.jinja' %}", "include"),
            ("{{ range(5) }}", "range(…)"),
            ("{{ raise_exception('no') }}", "raise_exception(…)"),
            ("{{ namespace(x=1) }}", "namespace(…)"),
            ("{{ messages[1:] }}", "a[b:c]"),
            ("{{ x | capitalize }}", "| capitalize"),
            ("{{ x.startswith('a') }}", ".startswith()"),
            ("{% for k, v in m.items() %}{% endfor %}", "for x, y in …"),
            ("{{ loop.index0 }}", "loop.*"),
            ("{{ 1.5 }}", "a float literal"),
            ("{{ a / b }}", "a / b"),
            ("{{ x is string }}", "is string"),
            ("{{ strftime_now('%Y') }}", "strftime_now(…)"),
        ] {
            let e = translate(src).expect_err("should be refused");
            match e {
                Error::Unsupported(r) => assert_eq!(r.construct, construct, "for `{src}`"),
                other => panic!("`{src}` gave a syntax error rather than a refusal: {other}"),
            }
        }
    }

    #[test]
    fn a_translation_round_trips_through_the_omni_ct_parser() {
        // `translate` reparses its own output, so a printed form that does not
        // parse is caught here rather than at render time. This checks the
        // printer's parenthesization, which is where that would break.
        for src in [
            "{{ (a + b) * c }}",
            "{{ not a and b }}",
            "{{ a if b else c if d else e }}",
            "{{ -x + y }}",
            "{{ a ~ b ~ c }}",
            "{{ x not in y }}",
            "{{ get(m, 'k', 'fallback') }}",
        ] {
            let t = translate(src).unwrap_or_else(|e| panic!("`{src}`: {e}"));
            let printed = print(&t.template.nodes);
            let again = crate::ct::Template::parse(&printed)
                .unwrap_or_else(|e| panic!("`{printed}` from `{src}`: {e}"));
            assert_eq!(again.nodes, t.template.nodes, "`{src}` -> `{printed}`");
        }
    }

    #[test]
    fn the_corpus_coverage_is_measured_and_its_blockers_named() {
        let c = coverage();
        assert_eq!(c.total, CORPUS.len());
        assert_eq!(c.translated.len() + c.refused.len(), c.total);
        // Every family that translates must also render, or "translated" is not
        // a useful word.
        let input = Value::map(vec![
            ("add_generation_prompt", Value::Bool(true)),
            ("bos_token", Value::text("<s>")),
            ("eos_token", Value::text("</s>")),
            ("system", Value::text("sys")),
            ("system_prompt", Value::text("sys")),
            (
                "messages",
                Value::Array(vec![msg("user", "hi"), msg("assistant", "yo")]),
            ),
        ]);
        for name in &c.translated {
            let s = CORPUS.iter().find(|s| &s.name == name).unwrap();
            let t = translate(s.source).unwrap();
            t.template
                .render(&input)
                .unwrap_or_else(|e| panic!("{name} translated but did not render: {e}"));
        }
        // The blockers are the deliverable, so the shape of that list is part of
        // the contract: most frequent first, and every refusal accounted for.
        let blockers = c.blockers();
        assert_eq!(
            blockers.iter().map(|(_, n)| n).sum::<usize>(),
            c.refused.len()
        );
        for w in blockers.windows(2) {
            assert!(w[0].1 >= w[1].1, "blockers are not sorted: {blockers:?}");
        }
        // Gate 2 wants 95 % of a hub snapshot. This is a 15-template corpus and
        // it is not that, so the assertion here is a floor that catches a
        // regression rather than a claim about the hub.
        assert!(
            c.rate() >= 0.6,
            "coverage fell to {:.0}%: {:?}",
            c.rate() * 100.0,
            c.refused
        );
    }
}
