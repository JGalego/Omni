//! §06.9 — OMNI-CT, a total template language for chat templates.
//!
//! A chat template is the last place in a model artifact where arbitrary code
//! still runs. Today it ships as a Jinja2 string that the runtime executes,
//! which means a downloaded file gets to run code in the loader of a format
//! whose entire premise is safe loading. §06.9 replaces that with OMNI-CT: a
//! language that is *total* — every well-formed template terminates, on every
//! input, without a sandbox.
//!
//! Totality is not a restriction bolted on afterwards; it is what the grammar
//! permits. There is no `while`, no recursion, no macro, no include, no import,
//! no attribute access into host objects and no method call on an arbitrary
//! value. `{% for %}` iterates a finite structure that is already in memory.
//! Rendering is therefore O(input size × template size), and the step and
//! output budgets below exist only as a backstop against a pathological
//! product of the two — never as a way to truncate output.
//!
//! Two of the language's forms are there because measuring the translator
//! (§06.9's own note, and `jinja`'s corpus) found templates that needed them and
//! nothing about totality that objected. `loop` is bound inside `{% for %}` to
//! seven values computed from the sequence's length and the current position —
//! and to nothing that remembers a previous iteration, which is the line
//! between a loop variable and loop state. `a[b:c]` slices lists and strings
//! with bounds clamped and negative bounds counted from the end, so a slice
//! cannot fail where an index can. Both are total for the same reason the rest
//! is: they are functions of data already in memory.
//!
//! Two consequences worth having beyond safety:
//!
//! - **The required inputs are computable.** [`Template::free_vars`] returns
//!   exactly the variables a template reads, statically. A runtime can check it
//!   has them before rendering, and `omni inspect` can print them.
//! - **A regression is detectable.** §06.9's `vectors` pair an input with the
//!   string it must render to, so `omni verify --template` turns "the chat
//!   template changed during conversion" into a build failure.
//!
//! ## Departures from §06.9, stated rather than hidden
//!
//! - §06.9 describes `vectors` as "input JSON → expected string". This
//!   implementation reads them as **canonical OMNI-CBOR** (§03.2): an array of
//!   `{"input": <map>, "want": <text>}`. A reader that can open the container
//!   already has a CBOR decoder and would otherwise need a second parser for a
//!   second encoding of the same data. The pairing is identical.
//! - An undefined variable is an **error**, not the empty string. Jinja
//!   silently renders nothing, which is how a template that lost an input ships
//!   a subtly wrong prompt. Since the free variables are computable, a caller
//!   can always know what to pass; `get(map, key, fallback)` and
//!   `default(value, fallback)` cover genuinely optional inputs.
//! - The value domain is §06.9's exactly — strings, integers, booleans, lists
//!   and maps. A byte string, a tag or a float in the input is refused rather
//!   than coerced into something with different rendering rules.

use crate::cbor::Value;
use crate::expr::Ref;
use std::collections::{BTreeMap, BTreeSet};

/// Longest template source accepted, so parsing is bounded.
pub const MAX_SOURCE: usize = 64 << 10;
/// Maximum nesting of blocks and of expression operators.
pub const MAX_DEPTH: usize = 32;
/// Maximum rendering steps. A backstop, not a policy: a template that hits it
/// is reported as over budget rather than rendered partially.
pub const STEP_BUDGET: u64 = 1 << 22;
/// Maximum rendered output.
pub const MAX_OUTPUT: usize = 4 << 20;

/// The language identifier §06.9 requires in `lang`.
pub const LANG: &str = "omni-ct/1";

// ---------------------------------------------------------------------- error --

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Syntax { line: usize, msg: String },
    Type(String),
    Undefined(String),
    Budget(String),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Syntax { line, msg } => write!(f, "line {line}: {msg}"),
            Error::Type(m) => write!(f, "type error: {m}"),
            Error::Undefined(m) => write!(f, "undefined: {m}"),
            Error::Budget(m) => write!(f, "over budget: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::expr::Error> for Error {
    fn from(e: crate::expr::Error) -> Error {
        Error::Unsupported(e.to_string())
    }
}

type Res<T> = Result<T, Error>;

fn syn<T>(line: usize, msg: impl Into<String>) -> Res<T> {
    Err(Error::Syntax {
        line,
        msg: msg.into(),
    })
}

// ------------------------------------------------------------------------ ast --

/// A template statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Text(String),
    /// `{{ expr }}`
    Out(Expr),
    /// `{% if %}` with its `elif` arms and optional `else`.
    If {
        arms: Vec<(Expr, Vec<Node>)>,
        otherwise: Vec<Node>,
    },
    /// `{% for x in expr %}`, with the `else` branch taken on an empty
    /// sequence. The iterated value is finite and already in memory, which is
    /// what makes rendering total.
    For {
        var: String,
        iter: Expr,
        body: Vec<Node>,
        otherwise: Vec<Node>,
    },
    /// `{% set name = expr %}`. Local to the enclosing block: a `set` inside a
    /// loop body does not escape it, so there is no way to build a counter that
    /// feeds back into the loop's own bound.
    Set {
        name: String,
        value: Expr,
    },
}

/// An expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Str(String),
    Int(i64),
    Bool(bool),
    Null,
    Var(String),
    List(Vec<Expr>),
    /// `a.b` — a map lookup, and only ever a map lookup. There is no host
    /// object to reach into.
    Field(Box<Expr>, String),
    /// `a[b]`
    Index(Box<Expr>, Box<Expr>),
    /// `a[b:c]`, with either bound optional. A slice of a list or a string,
    /// with the out-of-range and negative-index behaviour every language that
    /// has slices agrees on: bounds are clamped, a reversed range is empty, and
    /// a negative bound counts from the end. It cannot fail, which is why it can
    /// be in a total language at all.
    Slice {
        base: Box<Expr>,
        from: Option<Box<Expr>>,
        to: Option<Box<Expr>>,
    },
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Bin {
        op: BinOp,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// `a if c else b`
    Cond {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    /// A call into the fixed standard library. `a | upper` parses to
    /// `Call { name: "upper", args: [a] }`; there is no other kind of call.
    Call {
        name: String,
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    /// `~`: string concatenation, coercing both sides.
    Concat,
}

impl BinOp {
    fn name(self) -> &'static str {
        match self {
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

    fn parse(s: &str) -> Option<BinOp> {
        Some(match s {
            "or" => BinOp::Or,
            "and" => BinOp::And,
            "==" => BinOp::Eq,
            "!=" => BinOp::Ne,
            "<" => BinOp::Lt,
            "<=" => BinOp::Le,
            ">" => BinOp::Gt,
            ">=" => BinOp::Ge,
            "in" => BinOp::In,
            "not in" => BinOp::NotIn,
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            "%" => BinOp::Rem,
            "~" => BinOp::Concat,
            _ => return None,
        })
    }
}

/// The closed standard library. Every entry is pure and total: same inputs,
/// same output, no I/O, no failure mode other than a type error.
pub const STDLIB: &[(&str, &str)] = &[
    ("upper", "upper(s)"),
    ("lower", "lower(s)"),
    (
        "capitalize",
        "capitalize(s) — first character upper, the rest lower",
    ),
    (
        "title",
        "title(s) — first character of each word upper, the rest lower",
    ),
    ("trim", "trim(s)"),
    ("join", "join(list, sep)"),
    ("default", "default(v, fallback) — fallback when v is null"),
    ("get", "get(map_or_list, key, fallback)"),
    ("tojson", "tojson(v)"),
    ("strftime", "strftime(format, epoch_seconds)"),
    ("length", "length(s_or_list_or_map)"),
    ("first", "first(list_or_s)"),
    ("last", "last(list_or_s)"),
    ("replace", "replace(s, from, to)"),
    ("split", "split(s, sep)"),
    ("keys", "keys(map)"),
    ("values", "values(map)"),
    ("string", "string(v)"),
    ("int", "int(v)"),
    ("abs", "abs(i)"),
];

// --------------------------------------------------------------------- parser --

/// A parsed template.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub nodes: Vec<Node>,
    pub source: String,
}

struct Parser<'a> {
    s: &'a [char],
    i: usize,
    depth: usize,
    /// Set by `-%}` / `-}}`: strip the whitespace that follows the tag.
    trim_next: bool,
}

impl Template {
    /// Parses OMNI-CT source.
    pub fn parse(source: &str) -> Res<Template> {
        if source.len() > MAX_SOURCE {
            return Err(Error::Budget(format!(
                "template source is {} bytes; the limit is {MAX_SOURCE}",
                source.len()
            )));
        }
        let chars: Vec<char> = source.chars().collect();
        let mut p = Parser {
            s: &chars,
            i: 0,
            depth: 0,
            trim_next: false,
        };
        let (nodes, term) = p.block(&[])?;
        if !term.is_empty() {
            return syn(p.line(), format!("unexpected `{term}` at the top level"));
        }
        Ok(Template {
            nodes,
            source: source.to_string(),
        })
    }

    /// The variables this template reads, computed statically.
    ///
    /// This is the payoff of a total language: a caller can be told exactly what
    /// to supply, rather than discovering it from a rendering failure.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut bound = BTreeSet::new();
        free_in_block(&self.nodes, &mut bound, &mut out);
        out
    }

    /// Renders the template against an input map.
    pub fn render(&self, input: &Value) -> Res<String> {
        check_domain(input, 0)?;
        let mut st = Render {
            out: String::new(),
            steps: 0,
            scopes: vec![BTreeMap::new()],
            input,
        };
        st.block(&self.nodes)?;
        Ok(st.out)
    }
}

impl Parser<'_> {
    fn line(&self) -> usize {
        1 + self.s[..self.i.min(self.s.len())]
            .iter()
            .filter(|c| **c == '\n')
            .count()
    }

    fn starts(&self, pat: &str) -> bool {
        pat.chars()
            .enumerate()
            .all(|(k, c)| self.s.get(self.i + k) == Some(&c))
    }

    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.i += 1;
        }
    }

    /// Parses statements until one of `terms` opens, or end of input when
    /// `terms` is empty. Returns the terminator's name, already consumed.
    fn block(&mut self, terms: &[&str]) -> Res<(Vec<Node>, String)> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return syn(
                self.line(),
                format!("blocks nested deeper than {MAX_DEPTH}"),
            );
        }
        let mut out: Vec<Node> = Vec::new();
        let mut text = String::new();
        let flush = |text: &mut String, out: &mut Vec<Node>| {
            if !text.is_empty() {
                out.push(Node::Text(std::mem::take(text)));
            }
        };
        loop {
            if self.i >= self.s.len() {
                flush(&mut text, &mut out);
                if terms.is_empty() {
                    self.depth -= 1;
                    return Ok((out, String::new()));
                }
                return syn(
                    self.line(),
                    format!("unclosed block; expected `{}`", terms.join("` or `")),
                );
            }
            if !(self.starts("{{") || self.starts("{%") || self.starts("{#")) {
                if self.trim_next {
                    self.trim_next = false;
                    while matches!(self.peek(), Some(c) if c.is_whitespace()) {
                        self.i += 1;
                    }
                    continue;
                }
                text.push(self.s[self.i]);
                self.i += 1;
                continue;
            }
            self.trim_next = false;
            let kind = self.s[self.i + 1];
            let dash = self.s.get(self.i + 2) == Some(&'-');
            if dash {
                while text.ends_with(char::is_whitespace) {
                    text.pop();
                }
            }
            self.i += if dash { 3 } else { 2 };
            match kind {
                '#' => {
                    // A comment. Nothing inside it is parsed, so a comment can
                    // never introduce a construct.
                    while self.i < self.s.len() && !self.starts("#}") && !self.starts("-#}") {
                        self.i += 1;
                    }
                    if self.i >= self.s.len() {
                        return syn(self.line(), "unterminated comment");
                    }
                    if self.starts("-#}") {
                        self.trim_next = true;
                        self.i += 3;
                    } else {
                        self.i += 2;
                    }
                }
                '{' => {
                    let e = self.expr()?;
                    self.close("}}")?;
                    flush(&mut text, &mut out);
                    out.push(Node::Out(e));
                }
                _ => {
                    self.ws();
                    let name = self.ident()?;
                    if terms.contains(&name.as_str()) {
                        flush(&mut text, &mut out);
                        self.depth -= 1;
                        return Ok((out, name));
                    }
                    flush(&mut text, &mut out);
                    let node = self.statement(&name)?;
                    out.push(node);
                }
            }
        }
    }

    fn statement(&mut self, name: &str) -> Res<Node> {
        match name {
            "if" => {
                let mut arms = Vec::new();
                let mut cond = self.expr()?;
                self.close("%}")?;
                let mut otherwise = Vec::new();
                loop {
                    let (body, term) = self.block(&["elif", "else", "endif"])?;
                    arms.push((cond, body));
                    match term.as_str() {
                        "elif" => {
                            cond = self.expr()?;
                            self.close("%}")?;
                        }
                        "else" => {
                            self.close("%}")?;
                            let (body, _) = self.block(&["endif"])?;
                            otherwise = body;
                            self.close("%}")?;
                            break;
                        }
                        _ => {
                            self.close("%}")?;
                            break;
                        }
                    }
                }
                Ok(Node::If { arms, otherwise })
            }
            "for" => {
                let var = self.ident()?;
                let kw = self.ident()?;
                if kw != "in" {
                    return syn(self.line(), format!("expected `in`, found `{kw}`"));
                }
                let iter = self.expr()?;
                self.close("%}")?;
                let (body, term) = self.block(&["else", "endfor"])?;
                let mut otherwise = Vec::new();
                if term == "else" {
                    self.close("%}")?;
                    let (b, _) = self.block(&["endfor"])?;
                    otherwise = b;
                }
                self.close("%}")?;
                Ok(Node::For {
                    var,
                    iter,
                    body,
                    otherwise,
                })
            }
            "set" => {
                let name = self.ident()?;
                self.ws();
                if self.peek() != Some('=') {
                    return syn(self.line(), "expected `=` in `set`");
                }
                self.i += 1;
                let value = self.expr()?;
                self.close("%}")?;
                Ok(Node::Set { name, value })
            }
            other => syn(
                self.line(),
                format!(
                    "`{other}` is not an OMNI-CT statement; the set is `if`, `elif`, `else`, \
                     `endif`, `for`, `endfor`, `set` — deliberately closed, because every \
                     addition is a new way for a downloaded file to do something"
                ),
            ),
        }
    }

    /// Consumes the tag close, honouring `-%}` / `-}}` whitespace control.
    fn close(&mut self, pat: &str) -> Res<()> {
        self.ws();
        if self.peek() == Some('-') && self.s.get(self.i + 1) == Some(&pat.chars().next().unwrap())
        {
            self.trim_next = true;
            self.i += 1;
        }
        if !self.starts(pat) {
            return syn(
                self.line(),
                format!(
                    "expected `{pat}`, found `{}`",
                    self.peek()
                        .map(String::from)
                        .unwrap_or_else(|| "end".into())
                ),
            );
        }
        self.i += pat.len();
        Ok(())
    }

    fn ident(&mut self) -> Res<String> {
        self.ws();
        let start = self.i;
        while matches!(self.peek(), Some(c) if c.is_alphanumeric() || c == '_') {
            self.i += 1;
        }
        if self.i == start {
            return syn(
                self.line(),
                format!(
                    "expected a name, found `{}`",
                    self.peek()
                        .map(String::from)
                        .unwrap_or_else(|| "end".into())
                ),
            );
        }
        Ok(self.s[start..self.i].iter().collect())
    }

    /// Peeks an identifier without consuming it.
    fn peek_ident(&self) -> Option<String> {
        let mut j = self.i;
        while matches!(self.s.get(j), Some(c) if c.is_whitespace()) {
            j += 1;
        }
        let start = j;
        while matches!(self.s.get(j), Some(c) if c.is_alphanumeric() || *c == '_') {
            j += 1;
        }
        if j == start {
            None
        } else {
            Some(self.s[start..j].iter().collect())
        }
    }

    /// Consumes `kw` if it is the next identifier.
    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.peek_ident().as_deref() == Some(kw) {
            self.ws();
            self.i += kw.chars().count();
            true
        } else {
            false
        }
    }

    fn eat(&mut self, op: &str) -> bool {
        self.ws();
        if self.starts(op) {
            self.i += op.chars().count();
            true
        } else {
            false
        }
    }

    // Expression grammar, loosest binding first.
    fn expr(&mut self) -> Res<Expr> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return syn(
                self.line(),
                format!("expression nested deeper than {MAX_DEPTH}"),
            );
        }
        let out = self.conditional();
        self.depth -= 1;
        out
    }

    fn conditional(&mut self) -> Res<Expr> {
        let then = self.or_expr()?;
        if self.eat_kw("if") {
            let cond = self.or_expr()?;
            if !self.eat_kw("else") {
                return syn(
                    self.line(),
                    "a conditional expression needs `else`; there is no implicit empty branch",
                );
            }
            let otherwise = self.expr()?;
            return Ok(Expr::Cond {
                cond: Box::new(cond),
                then: Box::new(then),
                otherwise: Box::new(otherwise),
            });
        }
        Ok(then)
    }

    fn or_expr(&mut self) -> Res<Expr> {
        let mut l = self.and_expr()?;
        while self.eat_kw("or") {
            let r = self.and_expr()?;
            l = Expr::Bin {
                op: BinOp::Or,
                l: Box::new(l),
                r: Box::new(r),
            };
        }
        Ok(l)
    }

    fn and_expr(&mut self) -> Res<Expr> {
        let mut l = self.not_expr()?;
        while self.eat_kw("and") {
            let r = self.not_expr()?;
            l = Expr::Bin {
                op: BinOp::And,
                l: Box::new(l),
                r: Box::new(r),
            };
        }
        Ok(l)
    }

    fn not_expr(&mut self) -> Res<Expr> {
        // `not in` is a comparison operator, not a `not` applied to `in`, so it
        // must not be consumed here.
        if self.peek_ident().as_deref() == Some("not") {
            let save = self.i;
            self.eat_kw("not");
            if self.peek_ident().as_deref() == Some("in") {
                self.i = save;
            } else {
                return Ok(Expr::Not(Box::new(self.not_expr()?)));
            }
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Res<Expr> {
        let l = self.additive()?;
        // Two-character operators first, so `<=` is not read as `<` then `=`.
        for op in ["==", "!=", "<=", ">="] {
            if self.eat(op) {
                let r = self.additive()?;
                return Ok(Expr::Bin {
                    op: BinOp::parse(op).unwrap(),
                    l: Box::new(l),
                    r: Box::new(r),
                });
            }
        }
        for op in ["<", ">"] {
            if self.eat(op) {
                let r = self.additive()?;
                return Ok(Expr::Bin {
                    op: BinOp::parse(op).unwrap(),
                    l: Box::new(l),
                    r: Box::new(r),
                });
            }
        }
        if self.peek_ident().as_deref() == Some("not") {
            let save = self.i;
            self.eat_kw("not");
            if self.eat_kw("in") {
                let r = self.additive()?;
                return Ok(Expr::Bin {
                    op: BinOp::NotIn,
                    l: Box::new(l),
                    r: Box::new(r),
                });
            }
            self.i = save;
        }
        if self.eat_kw("in") {
            let r = self.additive()?;
            return Ok(Expr::Bin {
                op: BinOp::In,
                l: Box::new(l),
                r: Box::new(r),
            });
        }
        Ok(l)
    }

    fn additive(&mut self) -> Res<Expr> {
        let mut l = self.multiplicative()?;
        loop {
            let op = if self.eat("+") {
                BinOp::Add
            } else if self.eat("~") {
                BinOp::Concat
            } else if self.peek_after_ws() == Some('-') && !self.starts_close_after_ws() {
                self.eat("-");
                BinOp::Sub
            } else {
                return Ok(l);
            };
            let r = self.multiplicative()?;
            l = Expr::Bin {
                op,
                l: Box::new(l),
                r: Box::new(r),
            };
        }
    }

    fn peek_after_ws(&self) -> Option<char> {
        let mut j = self.i;
        while matches!(self.s.get(j), Some(c) if c.is_whitespace()) {
            j += 1;
        }
        self.s.get(j).copied()
    }

    /// True when the `%` at the cursor opens `%}` — the close of a statement
    /// tag — rather than being the remainder operator. Checked *before*
    /// consuming, because consuming it and backing out would leave the cursor
    /// past the tag close.
    fn percent_closes_tag(&self) -> bool {
        let mut j = self.i;
        while matches!(self.s.get(j), Some(c) if c.is_whitespace()) {
            j += 1;
        }
        self.s.get(j) == Some(&'%') && self.s.get(j + 1) == Some(&'}')
    }

    /// True when the `-` at the cursor is the whitespace-control dash of a tag
    /// close rather than a subtraction.
    fn starts_close_after_ws(&self) -> bool {
        let mut j = self.i;
        while matches!(self.s.get(j), Some(c) if c.is_whitespace()) {
            j += 1;
        }
        self.s.get(j) == Some(&'-')
            && matches!(self.s.get(j + 1), Some('}') | Some('%'))
            && matches!(self.s.get(j + 2), Some('}'))
    }

    fn multiplicative(&mut self) -> Res<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = if self.eat("*") {
                BinOp::Mul
            } else if self.eat("/") {
                BinOp::Div
            } else if self.peek_after_ws() == Some('%') && !self.percent_closes_tag() {
                self.eat("%");
                BinOp::Rem
            } else {
                return Ok(l);
            };
            let r = self.unary()?;
            l = Expr::Bin {
                op,
                l: Box::new(l),
                r: Box::new(r),
            };
        }
    }

    fn unary(&mut self) -> Res<Expr> {
        if self.peek_after_ws() == Some('-') && !self.starts_close_after_ws() {
            self.eat("-");
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Res<Expr> {
        let mut e = self.primary()?;
        loop {
            self.ws();
            if self.eat(".") {
                let name = self.ident()?;
                e = Expr::Field(Box::new(e), name);
            } else if self.eat("[") {
                // `[a]`, `[a:b]`, `[a:]`, `[:b]` and `[:]` — the index form and
                // the slice form share their opening bracket, so which one this
                // is is only known after the first expression (or its absence).
                self.ws();
                let lo = if self.starts(":") {
                    None
                } else {
                    Some(Box::new(self.expr()?))
                };
                self.ws();
                if self.eat(":") {
                    self.ws();
                    let hi = if self.starts("]") {
                        None
                    } else {
                        Some(Box::new(self.expr()?))
                    };
                    if !self.eat("]") {
                        return syn(self.line(), "expected `]`");
                    }
                    e = Expr::Slice {
                        base: Box::new(e),
                        from: lo,
                        to: hi,
                    };
                } else {
                    let Some(idx) = lo else {
                        return syn(self.line(), "expected an index or a slice");
                    };
                    if !self.eat("]") {
                        return syn(self.line(), "expected `]`");
                    }
                    e = Expr::Index(Box::new(e), idx);
                }
            } else if self.starts("|") && self.s.get(self.i + 1) != Some(&'|') {
                self.i += 1;
                let name = self.ident()?;
                let mut args = vec![e];
                if self.eat("(") {
                    args.extend(self.call_args()?);
                }
                e = self.checked_call(name, args)?;
            } else if self.eat("(") {
                // A call is only ever a call of a standard-library name; there
                // are no methods on values and no callables in the data model.
                let Expr::Var(name) = e else {
                    return syn(
                        self.line(),
                        "only the standard library is callable; there are no method calls on \
                         values in OMNI-CT",
                    );
                };
                let args = self.call_args()?;
                e = self.checked_call(name, args)?;
            } else {
                return Ok(e);
            }
        }
    }

    fn checked_call(&self, name: String, args: Vec<Expr>) -> Res<Expr> {
        if !STDLIB.iter().any(|(n, _)| *n == name) {
            return syn(
                self.line(),
                format!(
                    "`{name}` is not in the OMNI-CT standard library; it is closed, and the \
                     members are: {}",
                    STDLIB
                        .iter()
                        .map(|(n, _)| *n)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }
        Ok(Expr::Call { name, args })
    }

    fn call_args(&mut self) -> Res<Vec<Expr>> {
        let mut args = Vec::new();
        self.ws();
        if self.eat(")") {
            return Ok(args);
        }
        loop {
            args.push(self.expr()?);
            if self.eat(",") {
                continue;
            }
            if self.eat(")") {
                return Ok(args);
            }
            return syn(self.line(), "expected `,` or `)`");
        }
    }

    fn primary(&mut self) -> Res<Expr> {
        self.ws();
        match self.peek() {
            None => syn(self.line(), "expression ended early"),
            Some('\'') | Some('"') => self.string(),
            Some('(') => {
                self.i += 1;
                let e = self.expr()?;
                if !self.eat(")") {
                    return syn(self.line(), "expected `)`");
                }
                Ok(e)
            }
            Some('[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Ok(Expr::List(items));
                }
                loop {
                    items.push(self.expr()?);
                    if self.eat(",") {
                        self.ws();
                        if self.eat("]") {
                            return Ok(Expr::List(items));
                        }
                        continue;
                    }
                    if self.eat("]") {
                        return Ok(Expr::List(items));
                    }
                    return syn(self.line(), "expected `,` or `]`");
                }
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.i;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                }
                if self.peek() == Some('.') {
                    return syn(
                        self.line(),
                        "OMNI-CT has no floating-point literals; §06.9's value domain is \
                         strings, integers, booleans, lists and maps",
                    );
                }
                let text: String = self.s[start..self.i].iter().collect();
                match text.parse::<i64>() {
                    Ok(n) => Ok(Expr::Int(n)),
                    Err(_) => syn(self.line(), format!("`{text}` does not fit in an i64")),
                }
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let name = self.ident()?;
                Ok(match name.as_str() {
                    "true" => Expr::Bool(true),
                    "false" => Expr::Bool(false),
                    "none" | "null" => Expr::Null,
                    _ => Expr::Var(name),
                })
            }
            Some(c) => syn(self.line(), format!("unexpected `{c}` in an expression")),
        }
    }

    fn string(&mut self) -> Res<Expr> {
        let quote = self.s[self.i];
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return syn(self.line(), "unterminated string");
            };
            self.i += 1;
            if c == quote {
                return Ok(Expr::Str(out));
            }
            if c != '\\' {
                out.push(c);
                continue;
            }
            let Some(e) = self.peek() else {
                return syn(self.line(), "trailing backslash in string");
            };
            self.i += 1;
            out.push(match e {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '"' => '"',
                'u' => {
                    let mut n = 0u32;
                    for _ in 0..4 {
                        let Some(h) = self.peek().and_then(|c| c.to_digit(16)) else {
                            return syn(self.line(), "`\\u` needs four hex digits");
                        };
                        self.i += 1;
                        n = n * 16 + h;
                    }
                    match char::from_u32(n) {
                        Some(c) => c,
                        None => return syn(self.line(), format!("U+{n:04X} is not a character")),
                    }
                }
                other => {
                    return syn(
                        self.line(),
                        format!(
                            "`\\{other}` is not an escape OMNI-CT defines; it is refused rather \
                             than read as the literal `{other}`"
                        ),
                    )
                }
            });
        }
    }
}

// ---------------------------------------------------------------- free vars --

fn free_in_block(nodes: &[Node], bound: &mut BTreeSet<String>, out: &mut BTreeSet<String>) {
    for n in nodes {
        match n {
            Node::Text(_) => {}
            Node::Out(e) => free_in_expr(e, bound, out),
            Node::If { arms, otherwise } => {
                for (c, body) in arms {
                    free_in_expr(c, bound, out);
                    free_in_block(body, bound, out);
                }
                free_in_block(otherwise, bound, out);
            }
            Node::For {
                var,
                iter,
                body,
                otherwise,
            } => {
                free_in_expr(iter, bound, out);
                let mut inner = bound.clone();
                inner.insert(var.clone());
                inner.insert("loop".into());
                free_in_block(body, &mut inner, out);
                free_in_block(otherwise, bound, out);
            }
            Node::Set { name, value } => {
                free_in_expr(value, bound, out);
                bound.insert(name.clone());
            }
        }
    }
}

fn free_in_expr(e: &Expr, bound: &BTreeSet<String>, out: &mut BTreeSet<String>) {
    match e {
        Expr::Str(_) | Expr::Int(_) | Expr::Bool(_) | Expr::Null => {}
        Expr::Var(n) => {
            if !bound.contains(n) {
                out.insert(n.clone());
            }
        }
        Expr::List(items) => items.iter().for_each(|x| free_in_expr(x, bound, out)),
        Expr::Field(b, _) => free_in_expr(b, bound, out),
        Expr::Index(a, b) => {
            free_in_expr(a, bound, out);
            free_in_expr(b, bound, out);
        }
        Expr::Slice { base, from, to } => {
            free_in_expr(base, bound, out);
            for e in [from, to].into_iter().flatten() {
                free_in_expr(e, bound, out);
            }
        }
        Expr::Not(a) | Expr::Neg(a) => free_in_expr(a, bound, out),
        Expr::Bin { l, r, .. } => {
            free_in_expr(l, bound, out);
            free_in_expr(r, bound, out);
        }
        Expr::Cond {
            cond,
            then,
            otherwise,
        } => {
            free_in_expr(cond, bound, out);
            free_in_expr(then, bound, out);
            free_in_expr(otherwise, bound, out);
        }
        Expr::Call { args, .. } => args.iter().for_each(|x| free_in_expr(x, bound, out)),
    }
}

// ---------------------------------------------------------------- rendering --

struct Render<'a> {
    out: String,
    steps: u64,
    scopes: Vec<BTreeMap<String, Value>>,
    input: &'a Value,
}

impl Render<'_> {
    fn step(&mut self) -> Res<()> {
        self.steps += 1;
        if self.steps > STEP_BUDGET {
            return Err(Error::Budget(format!(
                "rendering exceeded {STEP_BUDGET} steps"
            )));
        }
        Ok(())
    }

    fn push(&mut self, s: &str) -> Res<()> {
        if self.out.len() + s.len() > MAX_OUTPUT {
            return Err(Error::Budget(format!(
                "rendered output would exceed {MAX_OUTPUT} bytes"
            )));
        }
        self.out.push_str(s);
        Ok(())
    }

    fn block(&mut self, nodes: &[Node]) -> Res<()> {
        for n in nodes {
            self.step()?;
            match n {
                Node::Text(t) => self.push(t)?,
                Node::Out(e) => {
                    let v = self.eval(e)?;
                    let s = to_text(&v)?;
                    self.push(&s)?;
                }
                Node::If { arms, otherwise } => {
                    let mut taken = false;
                    for (c, body) in arms {
                        if truthy(&self.eval(c)?) {
                            self.block(body)?;
                            taken = true;
                            break;
                        }
                    }
                    if !taken {
                        self.block(otherwise)?;
                    }
                }
                Node::For {
                    var,
                    iter,
                    body,
                    otherwise,
                } => {
                    let seq = self.eval(iter)?;
                    let items = sequence(&seq)?;
                    if items.is_empty() {
                        self.block(otherwise)?;
                        continue;
                    }
                    let n = items.len();
                    for (i, item) in items.into_iter().enumerate() {
                        self.step()?;
                        let mut scope = BTreeMap::new();
                        scope.insert(var.clone(), item);
                        scope.insert(
                            "loop".into(),
                            Value::map(vec![
                                ("index", Value::U(i as u64 + 1)),
                                ("index0", Value::U(i as u64)),
                                ("revindex", Value::U((n - i) as u64)),
                                ("revindex0", Value::U((n - i - 1) as u64)),
                                ("first", Value::Bool(i == 0)),
                                ("last", Value::Bool(i + 1 == n)),
                                ("length", Value::U(n as u64)),
                            ]),
                        );
                        self.scopes.push(scope);
                        let r = self.block(body);
                        self.scopes.pop();
                        r?;
                    }
                }
                Node::Set { name, value } => {
                    let v = self.eval(value)?;
                    if let Some(top) = self.scopes.last_mut() {
                        top.insert(name.clone(), v);
                    }
                }
            }
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Res<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
        }
        if let Some(v) = self.input.get(name) {
            return Ok(v.clone());
        }
        Err(Error::Undefined(format!(
            "`{name}` was not supplied; the template's inputs are computable in advance \
             (`omni inspect` prints them), so this is a missing input rather than an empty one"
        )))
    }

    fn eval(&mut self, e: &Expr) -> Res<Value> {
        self.step()?;
        Ok(match e {
            Expr::Str(s) => Value::text(s.clone()),
            Expr::Int(n) => int_value(*n),
            Expr::Bool(b) => Value::Bool(*b),
            Expr::Null => Value::Null,
            Expr::Var(n) => self.lookup(n)?,
            Expr::List(items) => Value::Array(
                items
                    .iter()
                    .map(|x| self.eval(x))
                    .collect::<Res<Vec<_>>>()?,
            ),
            Expr::Field(b, name) => {
                let base = self.eval(b)?;
                index(&base, &Value::text(name.clone()))?
            }
            Expr::Index(a, b) => {
                let base = self.eval(a)?;
                let key = self.eval(b)?;
                index(&base, &key)?
            }
            Expr::Slice { base, from, to } => {
                let b = self.eval(base)?;
                let lo = match from {
                    Some(e) => Some(as_int(&self.eval(e)?)?),
                    None => None,
                };
                let hi = match to {
                    Some(e) => Some(as_int(&self.eval(e)?)?),
                    None => None,
                };
                slice(&b, lo, hi)?
            }
            Expr::Not(a) => Value::Bool(!truthy(&self.eval(a)?)),
            Expr::Neg(a) => {
                let v = self.eval(a)?;
                let n = as_int(&v)?;
                int_value(
                    n.checked_neg().ok_or_else(|| {
                        Error::Type("negating this integer overflows an i64".into())
                    })?,
                )
            }
            Expr::Cond {
                cond,
                then,
                otherwise,
            } => {
                if truthy(&self.eval(cond)?) {
                    self.eval(then)?
                } else {
                    self.eval(otherwise)?
                }
            }
            // `and` and `or` short-circuit, and return a boolean rather than one
            // of their operands: OMNI-CT has no truthy-value-passing idiom, so
            // returning the operand would only invite surprise.
            Expr::Bin {
                op: BinOp::And,
                l,
                r,
            } => Value::Bool(truthy(&self.eval(l)?) && truthy(&self.eval(r)?)),
            Expr::Bin {
                op: BinOp::Or,
                l,
                r,
            } => Value::Bool(truthy(&self.eval(l)?) || truthy(&self.eval(r)?)),
            Expr::Bin { op, l, r } => {
                let a = self.eval(l)?;
                let b = self.eval(r)?;
                binop(*op, &a, &b)?
            }
            Expr::Call { name, args } => {
                let vals = args.iter().map(|x| self.eval(x)).collect::<Res<Vec<_>>>()?;
                call(name, &vals)?
            }
        })
    }
}

/// Integers are `U` when non-negative, as canonical CBOR requires (§03.2 D1).
fn int_value(n: i64) -> Value {
    if n >= 0 {
        Value::U(n as u64)
    } else {
        Value::I(n)
    }
}

fn as_int(v: &Value) -> Res<i64> {
    match v {
        Value::U(n) => i64::try_from(*n)
            .map_err(|_| Error::Type(format!("{n} does not fit in a signed integer"))),
        Value::I(n) => Ok(*n),
        other => Err(Error::Type(format!(
            "expected an integer, found {}",
            kind_of(other)
        ))),
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Text(_) => "a string",
        Value::U(_) | Value::I(_) => "an integer",
        Value::Bool(_) => "a boolean",
        Value::Array(_) => "a list",
        Value::Map(_) => "a map",
        Value::Null => "null",
        Value::Bytes(_) => "a byte string",
        Value::F64(_) => "a float",
        Value::Tag(_, _) => "a tagged value",
    }
}

/// Rejects anything outside §06.9's value domain, before rendering rather than
/// on first use, so a bad input fails the same way every time.
fn check_domain(v: &Value, depth: usize) -> Res<()> {
    if depth > MAX_DEPTH {
        return Err(Error::Budget(format!(
            "input nested deeper than {MAX_DEPTH}"
        )));
    }
    match v {
        Value::Text(_) | Value::U(_) | Value::I(_) | Value::Bool(_) | Value::Null => Ok(()),
        Value::Array(a) => a.iter().try_for_each(|x| check_domain(x, depth + 1)),
        Value::Map(m) => m.iter().try_for_each(|(k, val)| {
            if k.as_str().is_none() {
                return Err(Error::Type(
                    "map keys in a template input must be strings".into(),
                ));
            }
            check_domain(val, depth + 1)
        }),
        other => Err(Error::Type(format!(
            "{} is outside §06.9's value domain (strings, integers, booleans, lists, maps)",
            kind_of(other)
        ))),
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Null => false,
        Value::U(n) => *n != 0,
        Value::I(n) => *n != 0,
        Value::Text(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Map(m) => !m.is_empty(),
        _ => true,
    }
}

/// The string a `{{ … }}` renders to. A list or map has no one obvious
/// rendering, so it is an error rather than a guess; `tojson` is the explicit
/// form.
fn to_text(v: &Value) -> Res<String> {
    Ok(match v {
        Value::Text(s) => s.clone(),
        Value::U(n) => n.to_string(),
        Value::I(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => {
            return Err(Error::Type(format!(
                "{} has no single obvious rendering; use `tojson` or `join` to say which one",
                kind_of(other)
            )))
        }
    })
}

fn sequence(v: &Value) -> Res<Vec<Value>> {
    Ok(match v {
        Value::Array(a) => a.clone(),
        // Iterating a map yields its keys, as in Python and Jinja.
        Value::Map(m) => m.iter().map(|(k, _)| k.clone()).collect(),
        Value::Text(s) => s.chars().map(|c| Value::text(c.to_string())).collect(),
        Value::Null => Vec::new(),
        other => return Err(Error::Type(format!("{} is not iterable", kind_of(other)))),
    })
}

fn index(base: &Value, key: &Value) -> Res<Value> {
    match base {
        Value::Map(m) => {
            let Some(k) = key.as_str() else {
                return Err(Error::Type(format!(
                    "a map is indexed by a string, not {}",
                    kind_of(key)
                )));
            };
            m.iter()
                .find(|(mk, _)| mk.as_str() == Some(k))
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    Error::Undefined(format!(
                        "the map has no key `{k}`; use `get(map, '{k}', fallback)` for an \
                         optional one"
                    ))
                })
        }
        Value::Array(a) => {
            let i = as_int(key)?;
            let n = a.len() as i64;
            // Negative indices count from the end, as everywhere else.
            let at = if i < 0 { n + i } else { i };
            if at < 0 || at >= n {
                return Err(Error::Undefined(format!(
                    "index {i} is outside a list of {n}"
                )));
            }
            Ok(a[at as usize].clone())
        }
        other => Err(Error::Type(format!("{} cannot be indexed", kind_of(other)))),
    }
}

/// `base[from:to]` over a list or a string.
///
/// Total by construction: bounds are clamped into range after negative ones are
/// counted from the end, and a range that runs backwards is empty. There is no
/// input for which this fails, which is what lets §06.9 have it — indexing can
/// fail and does, but a slice of the wrong end of a short list is a shorter
/// list, not an error.
fn slice(base: &Value, from: Option<i64>, to: Option<i64>) -> Res<Value> {
    let n = match base {
        Value::Array(a) => a.len(),
        Value::Text(s) => s.chars().count(),
        other => {
            return Err(Error::Type(format!("{} cannot be sliced", kind_of(other))));
        }
    } as i64;
    let clamp = |i: i64| -> usize {
        let at = if i < 0 { n + i } else { i };
        at.clamp(0, n) as usize
    };
    let lo = clamp(from.unwrap_or(0));
    let hi = clamp(to.unwrap_or(n));
    let hi = hi.max(lo);
    Ok(match base {
        Value::Array(a) => Value::Array(a[lo..hi].to_vec()),
        Value::Text(s) => Value::text(s.chars().skip(lo).take(hi - lo).collect::<String>()),
        _ => unreachable!("checked above"),
    })
}

fn binop(op: BinOp, a: &Value, b: &Value) -> Res<Value> {
    use BinOp::*;
    Ok(match op {
        Or | And => unreachable!("short-circuited in eval"),
        Eq => Value::Bool(equal(a, b)),
        Ne => Value::Bool(!equal(a, b)),
        Lt | Le | Gt | Ge => {
            let ord = compare(a, b)?;
            Value::Bool(match op {
                Lt => ord.is_lt(),
                Le => ord.is_le(),
                Gt => ord.is_gt(),
                _ => ord.is_ge(),
            })
        }
        In => Value::Bool(contains(b, a)?),
        NotIn => Value::Bool(!contains(b, a)?),
        Concat => Value::text(format!("{}{}", to_text(a)?, to_text(b)?)),
        Add => match (a, b) {
            (Value::Text(x), Value::Text(y)) => Value::text(format!("{x}{y}")),
            (Value::Array(x), Value::Array(y)) => {
                Value::Array(x.iter().chain(y.iter()).cloned().collect())
            }
            _ => {
                let (x, y) = (as_int(a)?, as_int(b)?);
                int_value(
                    x.checked_add(y)
                        .ok_or_else(|| Error::Type("integer addition overflows".into()))?,
                )
            }
        },
        Sub | Mul | Div | Rem => {
            let (x, y) = (as_int(a)?, as_int(b)?);
            let r = match op {
                Sub => x.checked_sub(y),
                Mul => x.checked_mul(y),
                Div => {
                    if y == 0 {
                        return Err(Error::Type("division by zero".into()));
                    }
                    x.checked_div(y)
                }
                _ => {
                    if y == 0 {
                        return Err(Error::Type("remainder by zero".into()));
                    }
                    x.checked_rem(y)
                }
            };
            int_value(r.ok_or_else(|| Error::Type(format!("integer `{}` overflows", op.name())))?)
        }
    })
}

fn equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::U(_), Value::I(_)) | (Value::I(_), Value::U(_)) => match (as_int(a), as_int(b)) {
            (Ok(x), Ok(y)) => x == y,
            _ => false,
        },
        _ => a == b,
    }
}

fn compare(a: &Value, b: &Value) -> Res<std::cmp::Ordering> {
    match (a, b) {
        (Value::Text(x), Value::Text(y)) => Ok(x.cmp(y)),
        _ => Ok(as_int(a)?.cmp(&as_int(b)?)),
    }
}

fn contains(haystack: &Value, needle: &Value) -> Res<bool> {
    Ok(match haystack {
        Value::Array(a) => a.iter().any(|x| equal(x, needle)),
        // Membership in a map tests its keys, matching iteration.
        Value::Map(m) => m.iter().any(|(k, _)| equal(k, needle)),
        Value::Text(s) => {
            let Some(n) = needle.as_str() else {
                return Err(Error::Type(format!(
                    "`in` on a string needs a string, not {}",
                    kind_of(needle)
                )));
            };
            s.contains(n)
        }
        other => {
            return Err(Error::Type(format!(
                "`in` needs a list, map or string, not {}",
                kind_of(other)
            )))
        }
    })
}

// ---------------------------------------------------------------- stdlib --

fn arity(name: &str, args: &[Value], want: std::ops::RangeInclusive<usize>) -> Res<()> {
    if want.contains(&args.len()) {
        return Ok(());
    }
    let sig = STDLIB
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| *s)
        .unwrap_or(name);
    Err(Error::Type(format!(
        "`{name}` takes {}, given {} — signature: {sig}",
        if want.start() == want.end() {
            format!("{} argument(s)", want.start())
        } else {
            format!("{} to {} arguments", want.start(), want.end())
        },
        args.len()
    )))
}

fn text_arg(name: &str, v: &Value) -> Res<String> {
    v.as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Type(format!("`{name}` needs a string, given {}", kind_of(v))))
}

fn call(name: &str, args: &[Value]) -> Res<Value> {
    match name {
        "upper" => {
            arity(name, args, 1..=1)?;
            Ok(Value::text(text_arg(name, &args[0])?.to_uppercase()))
        }
        "lower" => {
            arity(name, args, 1..=1)?;
            Ok(Value::text(text_arg(name, &args[0])?.to_lowercase()))
        }
        "trim" => {
            arity(name, args, 1..=1)?;
            Ok(Value::text(text_arg(name, &args[0])?.trim().to_string()))
        }
        // `capitalize` and `title` lower-case what they do not upper-case,
        // which is what Jinja's filters of the same name do and is the
        // behaviour a translated template depends on.
        "capitalize" => {
            arity(name, args, 1..=1)?;
            let s = text_arg(name, &args[0])?;
            let mut out = String::with_capacity(s.len());
            for (i, c) in s.chars().enumerate() {
                if i == 0 {
                    out.extend(c.to_uppercase());
                } else {
                    out.extend(c.to_lowercase());
                }
            }
            Ok(Value::text(out))
        }
        "title" => {
            arity(name, args, 1..=1)?;
            let s = text_arg(name, &args[0])?;
            let mut out = String::with_capacity(s.len());
            // A word starts at the beginning and after any of the characters
            // Jinja's `title` treats as a boundary. Stating the set is the
            // point: "which characters start a word" is exactly the kind of
            // question a template must not have two answers to.
            let mut at_start = true;
            for c in s.chars() {
                if at_start {
                    out.extend(c.to_uppercase());
                } else {
                    out.extend(c.to_lowercase());
                }
                at_start = c.is_whitespace() || matches!(c, '-' | '(' | '{' | '[' | '<');
            }
            Ok(Value::text(out))
        }
        "join" => {
            arity(name, args, 1..=2)?;
            let sep = match args.get(1) {
                Some(v) => text_arg(name, v)?,
                None => String::new(),
            };
            let items = sequence(&args[0])?;
            let parts = items.iter().map(to_text).collect::<Res<Vec<_>>>()?;
            Ok(Value::text(parts.join(&sep)))
        }
        "default" => {
            arity(name, args, 2..=2)?;
            Ok(if matches!(args[0], Value::Null) {
                args[1].clone()
            } else {
                args[0].clone()
            })
        }
        "get" => {
            arity(name, args, 3..=3)?;
            Ok(index(&args[0], &args[1]).unwrap_or_else(|_| args[2].clone()))
        }
        "tojson" => {
            arity(name, args, 1..=1)?;
            let mut out = String::new();
            to_json(&args[0], &mut out)?;
            Ok(Value::text(out))
        }
        "strftime" => {
            arity(name, args, 2..=2)?;
            let fmt = text_arg(name, &args[0])?;
            strftime(&fmt, as_int(&args[1])?).map(Value::text)
        }
        "length" => {
            arity(name, args, 1..=1)?;
            Ok(Value::U(match &args[0] {
                Value::Text(s) => s.chars().count() as u64,
                Value::Array(a) => a.len() as u64,
                Value::Map(m) => m.len() as u64,
                other => {
                    return Err(Error::Type(format!("`length` of {}", kind_of(other))));
                }
            }))
        }
        "first" | "last" => {
            arity(name, args, 1..=1)?;
            let items = sequence(&args[0])?;
            items
                .into_iter()
                .nth(if name == "first" { 0 } else { usize::MAX })
                .or_else(|| sequence(&args[0]).ok().and_then(|v| v.last().cloned()))
                .ok_or_else(|| Error::Undefined(format!("`{name}` of an empty sequence")))
        }
        "replace" => {
            arity(name, args, 3..=3)?;
            Ok(Value::text(text_arg(name, &args[0])?.replace(
                &text_arg(name, &args[1])?,
                &text_arg(name, &args[2])?,
            )))
        }
        "split" => {
            arity(name, args, 2..=2)?;
            let sep = text_arg(name, &args[1])?;
            if sep.is_empty() {
                return Err(Error::Type("`split` needs a non-empty separator".into()));
            }
            Ok(Value::Array(
                text_arg(name, &args[0])?
                    .split(sep.as_str())
                    .map(Value::text)
                    .collect(),
            ))
        }
        "keys" | "values" => {
            arity(name, args, 1..=1)?;
            let Value::Map(m) = &args[0] else {
                return Err(Error::Type(format!(
                    "`{name}` needs a map, given {}",
                    kind_of(&args[0])
                )));
            };
            Ok(Value::Array(
                m.iter()
                    .map(|(k, v)| if name == "keys" { k.clone() } else { v.clone() })
                    .collect(),
            ))
        }
        "string" => {
            arity(name, args, 1..=1)?;
            Ok(Value::text(to_text(&args[0])?))
        }
        "int" => {
            arity(name, args, 1..=1)?;
            Ok(match &args[0] {
                Value::Text(s) => {
                    int_value(s.trim().parse::<i64>().map_err(|_| {
                        Error::Type(format!("`int` cannot read {s:?} as an integer"))
                    })?)
                }
                Value::Bool(b) => Value::U(*b as u64),
                other => int_value(as_int(other)?),
            })
        }
        "abs" => {
            arity(name, args, 1..=1)?;
            let n = as_int(&args[0])?;
            Ok(int_value(n.checked_abs().ok_or_else(|| {
                Error::Type("absolute value overflows an i64".into())
            })?))
        }
        other => Err(Error::Unsupported(format!(
            "`{other}` is not in the OMNI-CT standard library"
        ))),
    }
}

/// Serializes a value as JSON, deterministically. Map keys keep the order they
/// have in the value, which for anything read from a container is the canonical
/// CBOR order (§03.2 D6).
fn to_json(v: &Value, out: &mut String) -> Res<()> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::U(n) => out.push_str(&n.to_string()),
        Value::I(n) => out.push_str(&n.to_string()),
        Value::Text(s) => json_string(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                to_json(x, out)?;
            }
            out.push(']');
        }
        Value::Map(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let Some(key) = k.as_str() else {
                    return Err(Error::Type("`tojson` needs string map keys".into()));
                };
                json_string(key, out);
                out.push(':');
                to_json(val, out)?;
            }
            out.push('}');
        }
        other => {
            return Err(Error::Type(format!(
                "`tojson` of {} is outside the value domain",
                kind_of(other)
            )))
        }
    }
    Ok(())
}

fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Civil date from days since 1970-01-01. Howard Hinnant's algorithm, valid
/// across the whole i64 range this needs.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 }.div_euclid(400);
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// `strftime` on an explicit epoch second, in UTC.
///
/// Pure by construction: there is no way to ask this for the current time, so a
/// template cannot render differently on two machines. A chat template that
/// wants today's date takes it as an input, which is also the only way a
/// conformance vector could pin it down.
pub fn strftime(fmt: &str, epoch: i64) -> Res<String> {
    let days = epoch.div_euclid(86_400);
    let secs = epoch.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (secs / 3600, (secs / 60) % 60, secs % 60);
    let weekday = (days + 4).rem_euclid(7) as usize;
    let yday = days - days_from_civil(y, 1, 1) + 1;
    let mut out = String::new();
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(spec) = chars.next() else {
            return Err(Error::Type("`strftime` format ends with `%`".into()));
        };
        match spec {
            'Y' => out.push_str(&y.to_string()),
            'y' => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            'm' => out.push_str(&format!("{m:02}")),
            'd' => out.push_str(&format!("{d:02}")),
            'e' => out.push_str(&format!("{d:2}")),
            'H' => out.push_str(&format!("{hh:02}")),
            'M' => out.push_str(&format!("{mm:02}")),
            'S' => out.push_str(&format!("{ss:02}")),
            'j' => out.push_str(&format!("{yday:03}")),
            'B' => out.push_str(MONTHS[(m - 1) as usize]),
            'b' => out.push_str(&MONTHS[(m - 1) as usize][..3]),
            'A' => out.push_str(DAYS[weekday]),
            'a' => out.push_str(&DAYS[weekday][..3]),
            'F' => out.push_str(&format!("{y}-{m:02}-{d:02}")),
            'T' => out.push_str(&format!("{hh:02}:{mm:02}:{ss:02}")),
            's' => out.push_str(&epoch.to_string()),
            '%' => out.push('%'),
            other => {
                return Err(Error::Unsupported(format!(
                    "`%{other}` is not a conversion this `strftime` defines; the set is \
                     YymdeHMSjBbAaFTs%% — locale- and zone-dependent ones are excluded because \
                     they would make rendering machine-dependent"
                )))
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------- compiled AST form --

impl Template {
    /// The AST as canonical CBOR — §06.9's optional `compiled` field.
    ///
    /// It is a *derived* form, so it is checkable: `omni verify --level 6`
    /// reparses the source and compares, which is the only thing that makes a
    /// cache safe to trust.
    pub fn to_value(&self) -> Value {
        Value::map(vec![
            ("t", Value::text("omni.tok/ct-ast")),
            ("v", Value::U(1)),
            ("lang", Value::text(LANG)),
            (
                "body",
                Value::Array(self.nodes.iter().map(node_to_value).collect()),
            ),
        ])
    }
}

fn node_to_value(n: &Node) -> Value {
    match n {
        Node::Text(t) => Value::map(vec![
            ("k", Value::text("text")),
            ("s", Value::text(t.clone())),
        ]),
        Node::Out(e) => Value::map(vec![("k", Value::text("out")), ("e", expr_to_value(e))]),
        Node::If { arms, otherwise } => Value::map(vec![
            ("k", Value::text("if")),
            (
                "arms",
                Value::Array(
                    arms.iter()
                        .map(|(c, b)| {
                            Value::map(vec![
                                ("c", expr_to_value(c)),
                                ("b", Value::Array(b.iter().map(node_to_value).collect())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "else",
                Value::Array(otherwise.iter().map(node_to_value).collect()),
            ),
        ]),
        Node::For {
            var,
            iter,
            body,
            otherwise,
        } => Value::map(vec![
            ("k", Value::text("for")),
            ("var", Value::text(var.clone())),
            ("iter", expr_to_value(iter)),
            (
                "body",
                Value::Array(body.iter().map(node_to_value).collect()),
            ),
            (
                "else",
                Value::Array(otherwise.iter().map(node_to_value).collect()),
            ),
        ]),
        Node::Set { name, value } => Value::map(vec![
            ("k", Value::text("set")),
            ("name", Value::text(name.clone())),
            ("e", expr_to_value(value)),
        ]),
    }
}

fn expr_to_value(e: &Expr) -> Value {
    match e {
        Expr::Str(s) => Value::map(vec![
            ("k", Value::text("str")),
            ("s", Value::text(s.clone())),
        ]),
        Expr::Int(n) => Value::map(vec![("k", Value::text("int")), ("n", int_value(*n))]),
        Expr::Bool(b) => Value::map(vec![("k", Value::text("bool")), ("b", Value::Bool(*b))]),
        Expr::Null => Value::map(vec![("k", Value::text("null"))]),
        Expr::Var(n) => Value::map(vec![
            ("k", Value::text("var")),
            ("name", Value::text(n.clone())),
        ]),
        Expr::List(items) => Value::map(vec![
            ("k", Value::text("list")),
            (
                "items",
                Value::Array(items.iter().map(expr_to_value).collect()),
            ),
        ]),
        Expr::Field(b, name) => Value::map(vec![
            ("k", Value::text("field")),
            ("base", expr_to_value(b)),
            ("name", Value::text(name.clone())),
        ]),
        Expr::Index(a, b) => Value::map(vec![
            ("k", Value::text("index")),
            ("base", expr_to_value(a)),
            ("key", expr_to_value(b)),
        ]),
        Expr::Slice { base, from, to } => {
            let mut f = vec![("k", Value::text("slice")), ("base", expr_to_value(base))];
            if let Some(e) = from {
                f.push(("from", expr_to_value(e)));
            }
            if let Some(e) = to {
                f.push(("to", expr_to_value(e)));
            }
            Value::map(f)
        }
        Expr::Not(a) => Value::map(vec![("k", Value::text("not")), ("a", expr_to_value(a))]),
        Expr::Neg(a) => Value::map(vec![("k", Value::text("neg")), ("a", expr_to_value(a))]),
        Expr::Bin { op, l, r } => Value::map(vec![
            ("k", Value::text("bin")),
            ("op", Value::text(op.name())),
            ("l", expr_to_value(l)),
            ("r", expr_to_value(r)),
        ]),
        Expr::Cond {
            cond,
            then,
            otherwise,
        } => Value::map(vec![
            ("k", Value::text("cond")),
            ("c", expr_to_value(cond)),
            ("t", expr_to_value(then)),
            ("f", expr_to_value(otherwise)),
        ]),
        Expr::Call { name, args } => Value::map(vec![
            ("k", Value::text("call")),
            ("name", Value::text(name.clone())),
            (
                "args",
                Value::Array(args.iter().map(expr_to_value).collect()),
            ),
        ]),
    }
}

// -------------------------------------------------------------- the object --

/// A `ChatTemplate` object (§06.9).
#[derive(Clone, Debug)]
pub struct ChatTemplate {
    pub lang: String,
    pub template: Template,
    /// The pre-parsed AST, if the writer cached one. Derived, and checked.
    pub compiled: Option<Ref>,
    /// A Jinja2 rendering of the same template, for legacy runtimes. Carried
    /// verbatim and never executed — this implementation has no Jinja engine,
    /// which is the point.
    pub jinja_compat: Option<String>,
    pub capabilities: Vec<String>,
    pub vectors: Option<Ref>,
}

/// The capability names §06.9 lists.
pub const CAPABILITIES: &[&str] = &["tools", "system", "thinking", "multimodal"];

impl ChatTemplate {
    pub fn from_value(v: &Value) -> Res<ChatTemplate> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.tok/chat-template") {
            return Err(Error::Type(
                "R-O02: object is not an omni.tok/chat-template".into(),
            ));
        }
        let lang = v
            .get("lang")
            .and_then(|x| x.as_str())
            .unwrap_or(LANG)
            .to_string();
        if lang != LANG {
            return Err(Error::Unsupported(format!(
                "`{lang}` is not a template language this build implements; only `{LANG}` is, and \
                 a Jinja2 template is not run — executing one is the problem §06.9 exists to fix"
            )));
        }
        let source = v
            .get("source")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::Type("a chat template needs a `source`".into()))?;
        Ok(ChatTemplate {
            lang,
            template: Template::parse(source)?,
            compiled: match v.get("compiled") {
                Some(r) => Some(crate::expr::parse_ref_value(r)?),
                None => None,
            },
            jinja_compat: v
                .get("jinja_compat")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            capabilities: v
                .get("capabilities")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            vectors: match v.get("vectors") {
                Some(r) => Some(crate::expr::parse_ref_value(r)?),
                None => None,
            },
        })
    }

    pub fn load(ctx: &crate::expr::Ctx<'_>, r: &Ref) -> Res<ChatTemplate> {
        ChatTemplate::from_value(&ctx.value(&r.1)?)
    }

    /// Checks the claims a chat template makes about itself: that a declared
    /// capability corresponds to an input the template actually reads, that a
    /// declared capability is one §06.9 names, and that a cached AST agrees
    /// with the source it was compiled from.
    pub fn check(&self, ctx: &crate::expr::Ctx<'_>) -> Vec<String> {
        let mut out = Vec::new();
        let free = self.template.free_vars();
        for c in &self.capabilities {
            if !CAPABILITIES.contains(&c.as_str()) {
                out.push(format!(
                    "declares capability `{c}`, which is not one of {}",
                    CAPABILITIES.join(", ")
                ));
                continue;
            }
            // A template that says it supports tools but never reads `tools`
            // will silently drop them.
            let reads = match c.as_str() {
                "tools" => free.contains("tools"),
                "system" => true, // system messages arrive inside `messages`
                "thinking" => free.iter().any(|v| v.contains("thinking")),
                _ => true,
            };
            if !reads {
                out.push(format!(
                    "declares capability `{c}` but never reads a matching input; the inputs it \
                     reads are: {}",
                    free.iter().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
        if let Some(r) = self.compiled {
            // §06.9 stores `compiled` as a Blob: it is the CBOR encoding of the
            // AST, not an object with a schema of its own. Storing it under the
            // ChatTemplate otype would make it contradict its own `t` (R-O02).
            match ctx.bytes(&r.1).map_err(Error::from).and_then(|b| {
                crate::cbor::decode(&b)
                    .map_err(|e| Error::Type(format!("`compiled` is not canonical CBOR: {e:?}")))
            }) {
                Ok(cached) => {
                    let fresh = self.template.to_value();
                    if cached.encode() != fresh.encode() {
                        out.push(
                            "the cached `compiled` AST does not match the AST of `source`; a \
                             derived form that disagrees with its input is worse than no cache"
                                .into(),
                        );
                    }
                }
                Err(e) => out.push(format!("`compiled` is present but unreadable: {e}")),
            }
        }
        out
    }

    /// Runs the §06.9 vectors: each input must render to its recorded string.
    pub fn check_vectors(&self, ctx: &crate::expr::Ctx<'_>) -> Res<VectorReport> {
        let Some(r) = self.vectors else {
            return Ok(VectorReport::default());
        };
        let bytes = ctx.bytes(&r.1)?;
        let v = crate::cbor::decode(&bytes)
            .map_err(|e| Error::Type(format!("template vectors are not canonical CBOR: {e:?}")))?;
        let cases = v
            .as_array()
            .ok_or_else(|| Error::Type("template vectors must be a CBOR array".into()))?;
        let mut report = VectorReport::default();
        for (i, case) in cases.iter().enumerate() {
            let Some(input) = case.get("input") else {
                report.malformed += 1;
                continue;
            };
            let Some(want) = case.get("want").and_then(|x| x.as_str()) else {
                report.malformed += 1;
                continue;
            };
            report.total += 1;
            match self.template.render(input) {
                Ok(got) if got == want => report.passed += 1,
                Ok(got) => report.failures.push(VectorFailure {
                    case: i,
                    want: want.to_string(),
                    got: Some(got),
                    error: None,
                }),
                Err(e) => report.failures.push(VectorFailure {
                    case: i,
                    want: want.to_string(),
                    got: None,
                    error: Some(e.to_string()),
                }),
            }
        }
        Ok(report)
    }
}

/// Encodes template conformance vectors as the blob [`ChatTemplate::check_vectors`]
/// reads: a CBOR array of `{"input": …, "want": …}`.
pub fn encode_vectors(cases: &[(Value, String)]) -> Vec<u8> {
    Value::Array(
        cases
            .iter()
            .map(|(input, want)| {
                Value::map(vec![
                    ("input", input.clone()),
                    ("want", Value::text(want.clone())),
                ])
            })
            .collect::<Vec<_>>(),
    )
    .encode()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorFailure {
    pub case: usize,
    pub want: String,
    pub got: Option<String>,
    pub error: Option<String>,
}

impl std::fmt::Display for VectorFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "case {}: ", self.case)?;
        match (&self.got, &self.error) {
            (Some(got), _) => write!(f, "rendered {got:?}, expected {:?}", self.want),
            (None, Some(e)) => write!(f, "failed to render: {e}"),
            (None, None) => write!(f, "did not render {:?}", self.want),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VectorReport {
    pub total: usize,
    pub passed: usize,
    pub malformed: usize,
    pub failures: Vec<VectorFailure>,
}

impl VectorReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty() && self.malformed == 0
    }
}

impl std::fmt::Display for VectorReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} vectors pass", self.passed, self.total)?;
        if self.malformed > 0 {
            write!(f, ", {} malformed", self.malformed)?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::HashAlgo;
    use crate::expr::Ctx;
    use crate::store::{MemoryStore, WritableStore};

    fn render(src: &str, input: Value) -> Res<String> {
        Template::parse(src)?.render(&input)
    }

    fn msgs(pairs: &[(&str, &str)]) -> Value {
        Value::Array(
            pairs
                .iter()
                .map(|(r, c)| {
                    Value::map(vec![
                        ("role", Value::text(*r)),
                        ("content", Value::text(*c)),
                    ])
                })
                .collect(),
        )
    }

    /// The shape of a real chat template, in OMNI-CT.
    const CHAT: &str = "\
{%- for m in messages -%}
<|{{ m.role }}|>
{{ m.content }}<|end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}";

    #[test]
    fn a_realistic_chat_template_renders() {
        let input = Value::map(vec![
            ("messages", msgs(&[("system", "Be nice."), ("user", "Hi")])),
            ("add_generation_prompt", Value::Bool(true)),
        ]);
        assert_eq!(
            render(CHAT, input).unwrap(),
            "<|system|>\nBe nice.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>"
        );
    }

    #[test]
    fn the_required_inputs_are_computable_without_rendering() {
        let t = Template::parse(CHAT).unwrap();
        assert_eq!(
            t.free_vars().into_iter().collect::<Vec<_>>(),
            vec!["add_generation_prompt", "messages"]
        );
        // The loop variable and `loop` are bound, not required.
        assert!(!t.free_vars().contains("m"));
        assert!(!t.free_vars().contains("loop"));
        // A `set` binds for what follows it.
        let t = Template::parse("{% set x = 1 %}{{ x }}{{ y }}").unwrap();
        assert_eq!(
            t.free_vars().into_iter().collect::<Vec<_>>(),
            vec!["y".to_string()]
        );
    }

    #[test]
    fn a_missing_input_is_an_error_not_an_empty_string() {
        // Jinja renders nothing here, which is how a template that lost an
        // input ships a subtly wrong prompt for months.
        let e = render("{{ bos }}x", Value::map(vec![])).unwrap_err();
        assert!(matches!(e, Error::Undefined(_)), "{e}");
        // A missing map key likewise.
        let input = Value::map(vec![("m", Value::map(vec![("a", Value::U(1))]))]);
        assert!(matches!(
            render("{{ m.b }}", input.clone()).unwrap_err(),
            Error::Undefined(_)
        ));
        // And the explicit escape hatches work.
        assert_eq!(render("{{ get(m, 'b', 'z') }}", input).unwrap(), "z");
        assert_eq!(
            render(
                "{{ default(x, 'z') }}",
                Value::map(vec![("x", Value::Null)])
            )
            .unwrap(),
            "z"
        );
    }

    #[test]
    fn there_is_no_way_to_write_a_nonterminating_template() {
        // No `while`, no recursion, no macro, no include, no import. Each of
        // these is a syntax error naming the closed statement set.
        for src in [
            "{% while true %}{% endwhile %}",
            "{% macro f() %}{% endmacro %}",
            "{% include 'other' %}",
            "{% import 'x' as y %}",
            "{% raw %}{% endraw %}",
        ] {
            let e = Template::parse(src).unwrap_err();
            assert!(
                matches!(&e, Error::Syntax { msg, .. } if msg.contains("not an OMNI-CT statement")),
                "{src} -> {e}"
            );
        }
        // A `for` iterates a value already in memory, so its bound is the
        // input's size; a `set` inside the body cannot feed back into it.
        let input = Value::map(vec![("xs", Value::Array(vec![Value::U(1), Value::U(2)]))]);
        assert_eq!(
            render(
                "{% for x in xs %}{% set xs = [1,2,3] %}{{ x }}{% endfor %}",
                input
            )
            .unwrap(),
            "12"
        );
    }

    #[test]
    fn no_host_access_and_no_method_calls() {
        // `.` is a map lookup and nothing else, so there is no attribute to
        // reach through into the host.
        let input = Value::map(vec![("s", Value::text("hi"))]);
        assert!(matches!(
            render("{{ s.upper() }}", input.clone()).unwrap_err(),
            Error::Syntax { .. }
        ));
        // A name outside the closed library is refused at parse time, so it
        // cannot even be reached.
        let e = Template::parse("{{ eval('1') }}").unwrap_err();
        assert!(
            matches!(&e, Error::Syntax { msg, .. } if msg.contains("standard library")),
            "{e}"
        );
        // The library itself works, both as a call and as a filter.
        assert_eq!(render("{{ upper(s) }}", input.clone()).unwrap(), "HI");
        assert_eq!(render("{{ s | upper }}", input).unwrap(), "HI");
    }

    #[test]
    fn whitespace_control_matches_the_dashes() {
        assert_eq!(
            render("a\n  {{ 1 }}  \nb", Value::map(vec![])).unwrap(),
            "a\n  1  \nb"
        );
        assert_eq!(
            render("a\n  {{- 1 -}}  \nb", Value::map(vec![])).unwrap(),
            "a1b"
        );
        assert_eq!(
            render("a {%- if true -%} b {%- endif -%} c", Value::map(vec![])).unwrap(),
            "abc"
        );
        // A `-` that closes a tag is not a subtraction.
        assert_eq!(render("{{ 5 - 2 -}} x", Value::map(vec![])).unwrap(), "3x");
        assert_eq!(render("{{ 5-2 }}", Value::map(vec![])).unwrap(), "3");
    }

    #[test]
    fn loop_metadata_is_finite_and_correct() {
        let input = Value::map(vec![(
            "xs",
            Value::Array(vec![Value::text("a"), Value::text("b"), Value::text("c")]),
        )]);
        assert_eq!(
            render(
                "{% for x in xs %}{{ loop.index }}{{ x }}{% if not loop.last %},{% endif %}\
                 {% endfor %}",
                input.clone()
            )
            .unwrap(),
            "1a,2b,3c"
        );
        assert_eq!(
            render("{% for x in xs %}{{ loop.revindex }}{% endfor %}", input).unwrap(),
            "321"
        );
        // The `else` branch of a `for` fires on an empty sequence.
        assert_eq!(
            render(
                "{% for x in xs %}{{ x }}{% else %}none{% endfor %}",
                Value::map(vec![("xs", Value::Array(vec![]))])
            )
            .unwrap(),
            "none"
        );
    }

    #[test]
    fn expressions_follow_the_documented_precedence() {
        let e = Value::map(vec![]);
        for (src, want) in [
            ("{{ 1 + 2 * 3 }}", "7"),
            ("{{ (1 + 2) * 3 }}", "9"),
            ("{{ -2 + 5 }}", "3"),
            ("{{ 7 % 3 }}", "1"),
            ("{{ 7 / 2 }}", "3"),
            ("{{ 'a' ~ 1 ~ true }}", "a1true"),
            ("{{ 'a' + 'b' }}", "ab"),
            ("{{ 1 < 2 and 2 <= 2 }}", "true"),
            ("{{ not 1 > 2 }}", "true"),
            ("{{ 'b' in ['a', 'b'] }}", "true"),
            ("{{ 'c' not in ['a', 'b'] }}", "true"),
            ("{{ 'yes' if 1 else 'no' }}", "yes"),
            ("{{ join(['a','b'], '-') }}", "a-b"),
            ("{{ length('héllo') }}", "5"),
            ("{{ ['a','b'][-1] }}", "b"),
            ("{{ tojson(['a', 1, true, none]) }}", "[\"a\",1,true,null]"),
        ] {
            assert_eq!(render(src, e.clone()).unwrap(), want, "{src}");
        }
        // Division and remainder by zero are errors, not panics.
        assert!(render("{{ 1 / 0 }}", e.clone()).is_err());
        assert!(render("{{ 1 % 0 }}", e).is_err());
    }

    #[test]
    fn the_value_domain_is_enforced_on_the_way_in() {
        // A float or a byte string in the input is refused rather than coerced:
        // §06.9's domain is strings, integers, booleans, lists and maps.
        for bad in [
            Value::F64(1.5),
            Value::Bytes(vec![1]),
            Value::Tag(0, Box::new(Value::U(1))),
        ] {
            let input = Value::map(vec![("x", bad.clone())]);
            let e = render("{{ x }}", input).unwrap_err();
            assert!(matches!(e, Error::Type(_)), "{bad:?} -> {e}");
        }
        assert!(Template::parse("{{ 1.5 }}").is_err());
        // Printing a list has no one obvious answer, so it is refused.
        let input = Value::map(vec![("x", Value::Array(vec![Value::U(1)]))]);
        assert!(matches!(
            render("{{ x }}", input).unwrap_err(),
            Error::Type(_)
        ));
    }

    #[test]
    fn an_unknown_string_escape_is_refused() {
        // The same rule as the regex engine: reading `\d` as `d` would silently
        // change what the template emits.
        assert!(Template::parse(r"{{ '\d' }}").is_err());
        assert_eq!(
            render(r"{{ 'é\n\t' }}", Value::map(vec![])).unwrap(),
            "é\n\t"
        );
    }

    #[test]
    fn strftime_is_pure_and_matches_known_dates() {
        // There is no way to ask for "now", so two machines cannot disagree.
        assert_eq!(
            strftime("%Y-%m-%d %H:%M:%S", 0).unwrap(),
            "1970-01-01 00:00:00"
        );
        assert_eq!(
            strftime("%A %d %B %Y", 0).unwrap(),
            "Thursday 01 January 1970"
        );
        // A leap day, and a date before the epoch.
        assert_eq!(
            strftime("%F %a %j", 1_709_164_800).unwrap(),
            "2024-02-29 Thu 060"
        );
        assert_eq!(strftime("%F", -1).unwrap(), "1969-12-31");
        assert_eq!(
            strftime("%d/%m/%y %T", 1_767_225_600).unwrap(),
            "01/01/26 00:00:00"
        );
        // Round-trip the civil-date conversions across a wide range.
        for d in [-100_000i64, -1, 0, 1, 19_000, 100_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d, "{d}");
        }
        // A locale- or zone-dependent conversion is refused, not approximated.
        assert!(strftime("%Z", 0).is_err());
        assert!(strftime("%c", 0).is_err());
    }

    #[test]
    fn a_runaway_product_of_template_and_input_hits_the_budget() {
        // Totality is not the same as cheapness: nested loops over a large
        // input are finite but can still be enormous, so the budget is reported
        // rather than the output truncated.
        let n = 400u64;
        let xs = Value::Array((0..n).map(Value::U).collect());
        let input = Value::map(vec![("xs", xs)]);
        let src = "{% for a in xs %}{% for b in xs %}{% for c in xs %}x{% endfor %}\
                   {% endfor %}{% endfor %}";
        match render(src, input).unwrap_err() {
            Error::Budget(_) => {}
            other => panic!("expected a budget error, got {other}"),
        }
    }

    #[test]
    fn nesting_and_source_size_are_bounded() {
        let deep = "{% if true %}".repeat(MAX_DEPTH + 2);
        assert!(Template::parse(&deep).is_err());
        let long = "x".repeat(MAX_SOURCE + 1);
        assert!(matches!(
            Template::parse(&long).unwrap_err(),
            Error::Budget(_)
        ));
        assert!(Template::parse(
            "{{ ((((((((((((((((((((((((((((((((((1))))))))))))))))))))))))))))))))))  }}"
        )
        .is_err());
    }

    #[test]
    fn unclosed_blocks_are_syntax_errors() {
        for bad in [
            "{% if true %}x",
            "{% for x in y %}",
            "{{ 1",
            "{# comment",
            "{% if true %}{% endfor %}",
            "{% set %}",
            "{% for x y %}{% endfor %}",
        ] {
            assert!(Template::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }

    fn chat_object(s: &mut MemoryStore, source: &str, cases: &[(Value, String)]) -> Value {
        let blob = s.put(&encode_vectors(cases)).unwrap();
        Value::map(vec![
            ("t", Value::text("omni.tok/chat-template")),
            ("v", Value::U(1)),
            ("lang", Value::text(LANG)),
            ("source", Value::text(source)),
            (
                "vectors",
                Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
            ),
        ])
    }

    #[test]
    fn conformance_vectors_turn_a_regression_into_a_failure() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let input = Value::map(vec![
            ("messages", msgs(&[("user", "Hi")])),
            ("add_generation_prompt", Value::Bool(false)),
        ]);
        let good = "<|user|>\nHi<|end|>\n".to_string();
        let v = chat_object(&mut s, CHAT, &[(input.clone(), good.clone())]);
        let ctx = Ctx::new(&s);
        let t = ChatTemplate::from_value(&v).unwrap();
        let r = t.check_vectors(&ctx).unwrap();
        assert!(r.ok(), "{r}: {:?}", r.failures);
        assert_eq!(r.passed, 1);

        // Now change the template the way a conversion would, and the vector
        // catches it.
        let drifted = CHAT.replace("<|end|>", "</s>");
        let v = chat_object(&mut s, &drifted, &[(input, good)]);
        let t = ChatTemplate::from_value(&v).unwrap();
        let r = t.check_vectors(&Ctx::new(&s)).unwrap();
        assert!(!r.ok());
        assert!(
            r.failures[0].to_string().contains("expected"),
            "{}",
            r.failures[0]
        );
    }

    #[test]
    fn a_jinja_template_is_carried_but_never_run() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = chat_object(&mut s, CHAT, &[]).as_map().unwrap().to_vec();
        v.push((
            Value::text("jinja_compat"),
            Value::text("{% for m in messages %}{{ m.content }}{% endfor %}"),
        ));
        let t = ChatTemplate::from_value(&Value::Map(v)).unwrap();
        assert!(t.jinja_compat.is_some());
        // The derived Jinja form is not what renders.
        let out = t
            .template
            .render(&Value::map(vec![
                ("messages", msgs(&[("user", "Hi")])),
                ("add_generation_prompt", Value::Bool(false)),
            ]))
            .unwrap();
        assert!(out.contains("<|user|>"));

        // And a container claiming Jinja as the template *language* is refused,
        // because running it is the problem §06.9 exists to fix.
        let mut v = chat_object(&mut s, CHAT, &[]).as_map().unwrap().to_vec();
        v.retain(|(k, _)| k.as_str() != Some("lang"));
        v.push((Value::text("lang"), Value::text("jinja2")));
        let e = ChatTemplate::from_value(&Value::Map(v)).unwrap_err();
        assert!(
            matches!(&e, Error::Unsupported(m) if m.contains("not run")),
            "{e}"
        );
    }

    #[test]
    fn a_cached_ast_that_disagrees_with_its_source_is_reported() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let good = Template::parse(CHAT).unwrap();
        let other = Template::parse("{{ 1 }}").unwrap();
        let ctx_blob = |s: &mut MemoryStore, t: &Template| s.put(&t.to_value().encode()).unwrap();
        // `compiled` is a Blob (otype 0), not an object with its own schema.
        let right = ctx_blob(&mut s, &good);
        let wrong = ctx_blob(&mut s, &other);
        let build = |d: &[u8; 32]| {
            let mut v = Value::map(vec![
                ("t", Value::text("omni.tok/chat-template")),
                ("v", Value::U(1)),
                ("lang", Value::text(LANG)),
                ("source", Value::text(CHAT)),
            ])
            .as_map()
            .unwrap()
            .to_vec();
            v.push((
                Value::text("compiled"),
                Value::Array(vec![Value::U(0), Value::Bytes(d.to_vec())]),
            ));
            Value::Map(v)
        };
        let ctx = Ctx::new(&s);
        assert!(ChatTemplate::from_value(&build(&right))
            .unwrap()
            .check(&ctx)
            .is_empty());
        let findings = ChatTemplate::from_value(&build(&wrong))
            .unwrap()
            .check(&ctx);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("does not match"), "{}", findings[0]);
    }

    #[test]
    fn a_declared_capability_must_correspond_to_an_input() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = chat_object(&mut s, CHAT, &[]).as_map().unwrap().to_vec();
        v.push((
            Value::text("capabilities"),
            Value::Array(vec![Value::text("tools"), Value::text("telepathy")]),
        ));
        let t = ChatTemplate::from_value(&Value::Map(v)).unwrap();
        let f = t.check(&Ctx::new(&s));
        assert_eq!(f.len(), 2, "{f:?}");
        // `tools` is declared but the template never reads it, so tools would
        // be silently dropped.
        assert!(f.iter().any(|x| x.contains("never reads")), "{f:?}");
        assert!(f.iter().any(|x| x.contains("telepathy")), "{f:?}");
    }

    #[test]
    fn the_compiled_ast_round_trips_through_canonical_cbor() {
        let t = Template::parse(CHAT).unwrap();
        let bytes = t.to_value().encode();
        let back = crate::cbor::decode(&bytes).unwrap();
        assert_eq!(back.encode(), bytes);
        // And it is stable: parsing the same source twice gives the same AST,
        // which is what makes it usable as a content-addressed cache.
        assert_eq!(Template::parse(CHAT).unwrap().to_value().encode(), bytes);
    }

    #[test]
    fn the_wrong_object_type_is_refused() {
        let v = Value::map(vec![("t", Value::text("omni.tok/tokenizer"))]);
        assert!(matches!(
            ChatTemplate::from_value(&v).unwrap_err(),
            Error::Type(_)
        ));
    }
}
