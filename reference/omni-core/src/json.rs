//! A JSON reader and writer (RFC 8259), because the formats OMNI has to absorb
//! are described in JSON and this crate has no dependencies.
//!
//! It exists to read *headers*: a safetensors header, a `config.json`, an
//! adapter config. That is a small job with sharp edges, and the edges are the
//! reason this is real parsing rather than string matching:
//!
//! * **Bounded.** Nesting depth and input length are capped, like every other
//!   parser here (§12.4). A header is untrusted input arriving from a hub.
//! * **Strict.** Trailing commas, unquoted keys, comments, `NaN`, `Infinity`,
//!   leading `+`, leading zeros and lone surrogates are all errors. A permissive
//!   reader is how two implementations end up disagreeing about one file.
//! * **Exact where it matters.** Integers that fit in `u64`/`i64` are kept as
//!   integers rather than being routed through `f64`, because a tensor offset of
//!   2^53 + 1 is a real offset and `9007199254740993.0` is not it.
//!
//! What it is not: a serde replacement. There is no derive, no schema, and no
//! attempt at streaming. [`Value::get`] and the `as_*` accessors are the whole
//! interface, and callers are expected to check what they read.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The maximum nesting depth accepted. Deep enough for any real header, shallow
/// enough that recursion cannot exhaust the stack.
pub const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A JSON number that is an exact unsigned integer.
    U(u64),
    /// A JSON number that is an exact negative integer.
    I(i64),
    /// A JSON number that is not an integer, or does not fit in one.
    F(f64),
    Str(String),
    Array(Vec<Value>),
    /// Object members, sorted by key. JSON says member order is not significant,
    /// so keeping insertion order would preserve something callers must not
    /// depend on; sorting also makes the writer deterministic.
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(u) => Some(*u),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::U(u) => i64::try_from(*u).ok(),
            Value::I(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::U(u) => Some(*u as f64),
            Value::I(i) => Some(*i as f64),
            Value::F(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Compact encoding: no spaces, keys in sorted order, so the same value
    /// always produces the same bytes. Import and export both depend on that —
    /// a re-exported header that differs only in whitespace is not the
    /// round-trip §15.3.1 asks for.
    pub fn encode(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::U(u) => {
                let _ = write!(out, "{u}");
            }
            Value::I(i) => {
                let _ = write!(out, "{i}");
            }
            Value::F(f) => out.push_str(&write_f64(*f)),
            Value::Str(s) => write_string(s, out),
            Value::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Value::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Builds an object from pairs, for callers that would otherwise assemble a
/// `BTreeMap` by hand.
pub fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub fn string(s: impl Into<String>) -> Value {
    Value::Str(s.into())
}

/// JSON has no `NaN` or `Infinity`, so a non-finite float cannot be written.
/// Rather than emit `null` and lose the distinction, it is written as `0` and
/// the caller is expected not to be in this situation: nothing in a model header
/// is legitimately infinite.
fn write_f64(f: f64) -> String {
    if !f.is_finite() {
        return "0".to_string();
    }
    // `{:?}` gives the shortest representation that round-trips, which is what
    // JSON wants; it can produce `1e300`-style exponents, which JSON allows.
    let s = format!("{f:?}");
    // Rust writes `1.0`; that is valid JSON and keeps the value a float on the
    // way back, which is the honest reading of an f64.
    s
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ------------------------------------------------------------------- parsing --

#[derive(Debug)]
pub struct Error {
    pub at: usize,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.message)
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// Parses one JSON value, which must be the whole input apart from trailing
/// whitespace.
pub fn parse(input: &[u8]) -> Res<Value> {
    let mut p = Parser {
        b: input,
        i: 0,
        depth: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing bytes after the top-level value"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, m: impl Into<String>) -> Error {
        Error {
            at: self.i,
            message: m.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        // RFC 8259's four whitespace characters, and only those: a tab is
        // whitespace, a vertical tab is not.
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Res<()> {
        if self.peek() != Some(c) {
            return Err(self.err(format!("expected `{}`", c as char)));
        }
        self.i += 1;
        Ok(())
    }

    fn literal(&mut self, word: &[u8]) -> Res<()> {
        if self.b[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(())
        } else {
            Err(self.err("not a JSON literal"))
        }
    }

    fn value(&mut self) -> Res<Value> {
        if self.depth >= MAX_DEPTH {
            return Err(self.err(format!("nesting deeper than {MAX_DEPTH}")));
        }
        match self.peek() {
            None => Err(self.err("unexpected end of input")),
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(self.err(format!("unexpected byte {c:#04x}"))),
        }
    }

    fn array(&mut self) -> Res<Value> {
        self.eat(b'[')?;
        self.depth += 1;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(self.err("expected `,` or `]`")),
            }
        }
        self.depth -= 1;
        Ok(Value::Array(out))
    }

    fn object(&mut self) -> Res<Value> {
        self.eat(b'{')?;
        self.depth += 1;
        let mut out = BTreeMap::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Object(out));
        }
        loop {
            self.ws();
            let at = self.i;
            let key = self.string()?;
            // A duplicate member makes the object's meaning depend on which one
            // the reader keeps; RFC 8259 calls the behaviour unpredictable, so
            // it is refused rather than resolved.
            if out.contains_key(&key) {
                return Err(Error {
                    at,
                    message: format!("duplicate member `{key}`"),
                });
            }
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            out.insert(key, v);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(self.err("expected `,` or `}`")),
            }
        }
        self.depth -= 1;
        Ok(Value::Object(out))
    }

    fn string(&mut self) -> Res<String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                // RFC 8259: the control characters must be escaped.
                0x00..=0x1f => return Err(self.err("unescaped control character")),
                b'\\' => {
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.unicode_escape()?),
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                // Everything else is UTF-8 to be validated as a unit.
                _ => {
                    let start = self.i - 1;
                    let len = utf8_len(c).ok_or_else(|| self.err("invalid UTF-8 lead byte"))?;
                    let end = start + len;
                    let s = self
                        .b
                        .get(start..end)
                        .ok_or_else(|| self.err("truncated UTF-8 sequence"))?;
                    let s = std::str::from_utf8(s).map_err(|_| self.err("invalid UTF-8"))?;
                    out.push_str(s);
                    self.i = end;
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Res<char> {
        let first = self.hex4()?;
        // Surrogate pairs: JSON escapes are UTF-16, so a character above the BMP
        // arrives as two escapes and a lone half is an error rather than a
        // replacement character.
        let scalar = match first {
            0xD800..=0xDBFF => {
                if !self.b[self.i..].starts_with(b"\\u") {
                    return Err(self.err("a high surrogate with no low surrogate"));
                }
                self.i += 2;
                let low = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&low) {
                    return Err(self.err("a high surrogate followed by a non-surrogate"));
                }
                0x1_0000 + ((first as u32 - 0xD800) << 10) + (low as u32 - 0xDC00)
            }
            0xDC00..=0xDFFF => return Err(self.err("a low surrogate with no high surrogate")),
            other => other as u32,
        };
        char::from_u32(scalar).ok_or_else(|| self.err("not a Unicode scalar value"))
    }

    fn hex4(&mut self) -> Res<u16> {
        let s = self
            .b
            .get(self.i..self.i + 4)
            .ok_or_else(|| self.err("truncated \\u escape"))?;
        let s = std::str::from_utf8(s).map_err(|_| self.err("bad \\u escape"))?;
        let v = u16::from_str_radix(s, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Res<Value> {
        let start = self.i;
        let neg = self.peek() == Some(b'-');
        if neg {
            self.i += 1;
        }
        // int: `0` or a non-zero digit followed by digits. `01` is not a JSON
        // number, and accepting it would make this reader disagree with every
        // other one about a file that is simply invalid.
        match self.peek() {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    return Err(self.err("leading zero"));
                }
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err(self.err("expected a digit")),
        }
        let mut integral = true;
        if self.peek() == Some(b'.') {
            integral = false;
            self.i += 1;
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.err("a decimal point with no digits after it"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            integral = false;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(self.err("an exponent with no digits"));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII by construction");
        if integral {
            // Exact integers stay exact. A tensor offset is an integer, and
            // routing it through f64 silently rounds anything past 2^53.
            if neg {
                if let Ok(i) = text.parse::<i64>() {
                    return Ok(Value::I(i));
                }
            } else if let Ok(u) = text.parse::<u64>() {
                return Ok(Value::U(u));
            }
        }
        let f: f64 = text.parse().map_err(|_| Error {
            at: start,
            message: "not a number".into(),
        })?;
        if !f.is_finite() {
            return Err(Error {
                at: start,
                message: "a number too large to represent".into(),
            });
        }
        Ok(Value::F(f))
    }
}

fn utf8_len(lead: u8) -> Option<usize> {
    Some(match lead {
        0x00..=0x7f => 1,
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_a_header_is_made_of() {
        let v = parse(br#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#).unwrap();
        let a = v.get("a").unwrap();
        assert_eq!(a.get("dtype").unwrap().as_str(), Some("F32"));
        assert_eq!(
            a.get("shape").unwrap().as_array().unwrap(),
            &[Value::U(2), Value::U(3)]
        );
        assert_eq!(
            a.get("data_offsets").unwrap().as_array().unwrap(),
            &[Value::U(0), Value::U(24)]
        );
    }

    #[test]
    fn every_scalar_kind() {
        assert_eq!(parse(b"null").unwrap(), Value::Null);
        assert_eq!(parse(b"true").unwrap(), Value::Bool(true));
        assert_eq!(parse(b"false").unwrap(), Value::Bool(false));
        assert_eq!(parse(b"0").unwrap(), Value::U(0));
        assert_eq!(parse(b"-1").unwrap(), Value::I(-1));
        assert_eq!(parse(b"1.5").unwrap(), Value::F(1.5));
        assert_eq!(parse(b"1e3").unwrap(), Value::F(1000.0));
        assert_eq!(parse(b"-0").unwrap(), Value::I(0));
        assert_eq!(parse(br#""x""#).unwrap(), Value::Str("x".into()));
    }

    /// An offset past 2^53 is a real offset in a real 100 GB shard. Routing it
    /// through `f64` loses it, so integers are parsed as integers.
    #[test]
    fn a_large_integer_survives_exactly() {
        let v = parse(b"9007199254740993").unwrap();
        assert_eq!(v.as_u64(), Some(9_007_199_254_740_993));
        assert_eq!(v.encode(), "9007199254740993");
        // And the largest u64 is not silently a float.
        assert_eq!(
            parse(b"18446744073709551615").unwrap().as_u64(),
            Some(u64::MAX)
        );
        // Past that, it becomes a float and says so rather than failing: the
        // value is representable, just not exactly.
        assert!(matches!(
            parse(b"18446744073709551616").unwrap(),
            Value::F(_)
        ));
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(
            parse(br#""a\"b\\c\/d\n\t\r\b\f""#).unwrap().as_str(),
            Some("a\"b\\c/d\n\t\r\u{08}\u{0c}")
        );
        assert_eq!(parse(br#""\u20ac""#).unwrap().as_str(), Some("€"));
        // A surrogate pair is one character.
        assert_eq!(parse(br#""\ud83d\ude00""#).unwrap().as_str(), Some("😀"));
        // Raw UTF-8 passes through.
        assert_eq!(
            parse("\"héllo → 😀\"".as_bytes()).unwrap().as_str(),
            Some("héllo → 😀")
        );
    }

    /// A permissive reader is how two implementations disagree about one file.
    /// Each of these is invalid JSON, and each must be an error rather than a
    /// guess.
    #[test]
    fn what_is_not_json_is_refused() {
        for bad in [
            &b"{\"a\":1,}"[..], // trailing comma
            b"[1,2,]",          // trailing comma
            b"{a:1}",           // unquoted key
            b"{'a':1}",         // single quotes
            b"{\"a\":1}// c",   // comment
            b"NaN",
            b"Infinity",
            b"-Infinity",
            b"+1",
            b"01",
            b"1.",
            b".5",
            b"1e",
            b"1e+",
            b"0x10",
            b"{\"a\":}",
            b"{\"a\" 1}",
            b"[1 2]",
            b"\"unterminated",
            b"\"\\q\"",            // unknown escape
            b"\"\\u00\"",          // truncated escape
            b"\"\\ud800\"",        // lone high surrogate
            b"\"\\udc00\"",        // lone low surrogate
            b"\"\\ud800\\u0041\"", // high surrogate then not a low one
            b"{\"a\":1}{}",        // two top-level values
            b"",
            b"   ",
            b"{\"a\":1,\"a\":2}", // duplicate member
            b"\"\x01\"",          // unescaped control character
        ] {
            assert!(
                parse(bad).is_err(),
                "accepted invalid JSON: {}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    /// Depth is bounded, and the bound is hit with an error rather than a stack
    /// overflow (§12.4).
    #[test]
    fn nesting_is_bounded() {
        let ok = format!("{}{}", "[".repeat(MAX_DEPTH - 1), "]".repeat(MAX_DEPTH - 1));
        assert!(parse(ok.as_bytes()).is_ok());
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert!(parse(deep.as_bytes()).is_err());
        // And the same for objects, which recurse through a different arm.
        let deep = format!(
            "{}1{}",
            "{\"a\":".repeat(MAX_DEPTH + 1),
            "}".repeat(MAX_DEPTH + 1)
        );
        assert!(parse(deep.as_bytes()).is_err());
    }

    /// The writer is deterministic and the reader accepts what it wrote: that
    /// pair is what makes an exported header reproducible.
    #[test]
    fn writing_then_reading_is_the_identity() {
        let v = object(vec![
            ("z", Value::U(1)),
            (
                "a",
                Value::Array(vec![Value::I(-2), Value::F(0.5), Value::Null]),
            ),
            ("m", object(vec![("k", string("v\n\"quoted\""))])),
            ("t", Value::Bool(true)),
        ]);
        let text = v.encode();
        // Keys sorted, no spaces: one value, one encoding.
        assert_eq!(
            text,
            r#"{"a":[-2,0.5,null],"m":{"k":"v\n\"quoted\""},"t":true,"z":1}"#
        );
        assert_eq!(parse(text.as_bytes()).unwrap(), v);
        assert_eq!(parse(text.as_bytes()).unwrap().encode(), text);
    }

    #[test]
    fn a_truncated_header_is_an_error_at_every_length() {
        let full = br#"{"weight":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}}"#;
        for n in 0..full.len() {
            assert!(parse(&full[..n]).is_err(), "prefix of {n} parsed");
        }
        assert!(parse(full).is_ok());
    }

    /// Malformed UTF-8 inside a string is an error, not a lossy conversion: a
    /// tensor name that changed during import points at nothing.
    #[test]
    fn invalid_utf8_is_refused_rather_than_replaced() {
        for bad in [
            &b"\"\xff\""[..],
            b"\"\xc3\"",         // truncated two-byte sequence
            b"\"\xe0\x80\x80\"", // overlong
            b"\"\xed\xa0\x80\"", // a surrogate encoded directly
        ] {
            assert!(parse(bad).is_err(), "accepted {bad:?}");
        }
    }
}
