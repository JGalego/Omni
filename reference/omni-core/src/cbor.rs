//! OMNI-CBOR: the deterministic CBOR profile of §03.2.
//!
//! Encoding enforces rules D1–D8 by construction:
//!   D1 shortest integer form           D5 shortest round-tripping float
//!   D2 definite lengths only           D6 canonical NaN
//!   D3 map keys sorted by encoded bytes D7 registered tags only
//!   D4 no duplicate keys               D8 no trailing bytes
//!
//! Decoding is strict: any input that would not have been produced by this
//! encoder is rejected, so `decode(encode(v)) == v` and
//! `encode(decode(b)) == b` both hold. That round-trip property is what makes
//! object digests stable across implementations.

/// Maximum nesting depth (§02.10 / §12.4).
pub const MAX_DEPTH: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U(u64),
    I(i64), // negative integers only; non-negative must use U
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Tag(u64, Box<Value>),
    Bool(bool),
    Null,
    F64(f64),
}

#[derive(Debug)]
pub enum Error {
    Eof,
    Trailing(usize),
    DepthExceeded,
    NonCanonicalInt,
    NonCanonicalFloat,
    IndefiniteLength,
    DuplicateKey,
    UnsortedKeys,
    BadUtf8,
    Reserved(u8),
    UnregisteredTag(u64),
    LengthOverflow,
    TypeMismatch(&'static str),
    Missing(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Eof => write!(f, "unexpected end of input"),
            Error::Trailing(n) => write!(f, "{n} trailing bytes after top-level item (D8)"),
            Error::DepthExceeded => write!(f, "nesting depth exceeds {MAX_DEPTH}"),
            Error::NonCanonicalInt => write!(f, "integer not in shortest form (D1)"),
            Error::NonCanonicalFloat => write!(f, "float not in shortest form (D5/D6)"),
            Error::IndefiniteLength => write!(f, "indefinite-length item (D2)"),
            Error::DuplicateKey => write!(f, "duplicate map key (D4)"),
            Error::UnsortedKeys => write!(f, "map keys not in canonical order (D3)"),
            Error::BadUtf8 => write!(f, "invalid UTF-8 in text string"),
            Error::Reserved(b) => write!(f, "reserved/unsupported initial byte 0x{b:02x}"),
            Error::UnregisteredTag(t) => write!(f, "unregistered tag {t} (D7)"),
            Error::LengthOverflow => write!(f, "declared length exceeds available input"),
            Error::TypeMismatch(w) => write!(f, "expected {w}"),
            Error::Missing(k) => write!(f, "missing required key `{k}`"),
        }
    }
}

impl std::error::Error for Error {}

/// Tags registered by §03.3.
pub const TAG_BIGNUM_POS: u64 = 2;
pub const TAG_BIGNUM_NEG: u64 = 3;
pub const TAG_RATIONAL: u64 = 30;
pub const TAG_REF: u64 = 1001;
pub const TAG_DIGEST: u64 = 1002;
pub const TAG_DTYPE: u64 = 1003;
pub const TAG_SHAPE: u64 = 1004;
pub const TAG_EXPR: u64 = 1005;
pub const TAG_URI: u64 = 1006;
pub const TAG_DECIMAL: u64 = 1007;

fn tag_is_registered(t: u64) -> bool {
    matches!(
        t,
        TAG_BIGNUM_POS
            | TAG_BIGNUM_NEG
            | TAG_RATIONAL
            | TAG_REF
            | TAG_DIGEST
            | TAG_DTYPE
            | TAG_SHAPE
            | TAG_EXPR
            | TAG_URI
            | TAG_DECIMAL
    )
}

// ---------------------------------------------------------------- encoding --

fn put_head(out: &mut Vec<u8>, major: u8, arg: u64) {
    let m = major << 5;
    if arg < 24 {
        out.push(m | arg as u8);
    } else if arg <= u8::MAX as u64 {
        out.push(m | 24);
        out.push(arg as u8);
    } else if arg <= u16::MAX as u64 {
        out.push(m | 25);
        out.extend_from_slice(&(arg as u16).to_be_bytes());
    } else if arg <= u32::MAX as u64 {
        out.push(m | 26);
        out.extend_from_slice(&(arg as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&arg.to_be_bytes());
    }
}

/// IEEE 754 binary32 → binary16 bits, or `None` if not exactly representable.
fn f32_to_f16_exact(f: f32) -> Option<u16> {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // inf / NaN
        return if mant == 0 {
            Some(sign | 0x7c00)
        } else {
            // Only the canonical quiet NaN is permitted (D6).
            None
        };
    }
    if exp == 0 && mant == 0 {
        return Some(sign); // ±0
    }
    let unbiased = exp - 127;
    if !(-24..=15).contains(&unbiased) {
        return None;
    }
    if unbiased >= -14 {
        // normal in f16: needs the low 13 mantissa bits clear
        if mant & 0x1fff != 0 {
            return None;
        }
        Some(sign | (((unbiased + 15) as u16) << 10) | ((mant >> 13) as u16))
    } else {
        // subnormal in f16
        let shift = (-14 - unbiased) as u32;
        let full = mant | 0x0080_0000;
        let denom = 13 + shift;
        if denom >= 32 || (full & ((1u32 << denom) - 1)) != 0 {
            return None;
        }
        Some(sign | (full >> denom) as u16)
    }
}

fn encode_float(out: &mut Vec<u8>, v: f64) {
    if v.is_nan() {
        // D6: the only permitted NaN encoding.
        out.push(0xf9);
        out.extend_from_slice(&0x7e00u16.to_be_bytes());
        return;
    }
    let as32 = v as f32;
    if (as32 as f64).to_bits() == v.to_bits() {
        if let Some(h) = f32_to_f16_exact(as32) {
            out.push(0xf9);
            out.extend_from_slice(&h.to_be_bytes());
            return;
        }
        out.push(0xfa);
        out.extend_from_slice(&as32.to_be_bytes());
        return;
    }
    out.push(0xfb);
    out.extend_from_slice(&v.to_be_bytes());
}

impl Value {
    /// Canonical encoding. Map keys are sorted by their encoded bytes (D3).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            Value::U(n) => put_head(out, 0, *n),
            Value::I(n) => {
                debug_assert!(*n < 0, "non-negative integers must use Value::U (D1)");
                put_head(out, 1, (-1 - *n) as u64)
            }
            Value::Bytes(b) => {
                put_head(out, 2, b.len() as u64);
                out.extend_from_slice(b);
            }
            Value::Text(s) => {
                put_head(out, 3, s.len() as u64);
                out.extend_from_slice(s.as_bytes());
            }
            Value::Array(a) => {
                put_head(out, 4, a.len() as u64);
                for v in a {
                    v.encode_into(out);
                }
            }
            Value::Map(m) => {
                // Sort by encoded key bytes; this is the whole of D3.
                let mut items: Vec<(Vec<u8>, &Value)> =
                    m.iter().map(|(k, v)| (k.encode(), v)).collect();
                items.sort_by(|a, b| a.0.cmp(&b.0));
                put_head(out, 5, items.len() as u64);
                for (k, v) in items {
                    out.extend_from_slice(&k);
                    v.encode_into(out);
                }
            }
            Value::Tag(t, inner) => {
                put_head(out, 6, *t);
                inner.encode_into(out);
            }
            Value::Bool(false) => out.push(0xf4),
            Value::Bool(true) => out.push(0xf5),
            Value::Null => out.push(0xf6),
            Value::F64(f) => encode_float(out, *f),
        }
    }
}

// ---------------------------------------------------------------- decoding --

struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Dec<'a> {
    fn byte(&mut self) -> Result<u8, Error> {
        let v = *self.b.get(self.p).ok_or(Error::Eof)?;
        self.p += 1;
        Ok(v)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if n > self.b.len().saturating_sub(self.p) {
            return Err(Error::LengthOverflow);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    /// Reads a head, enforcing shortest-form (D1) and rejecting indefinite
    /// lengths (D2).
    fn head(&mut self) -> Result<(u8, u64), Error> {
        let ib = self.byte()?;
        let major = ib >> 5;
        let ai = ib & 0x1f;
        let arg = match ai {
            0..=23 => ai as u64,
            24 => {
                let v = self.byte()? as u64;
                if v < 24 {
                    return Err(Error::NonCanonicalInt);
                }
                v
            }
            25 => {
                let s = self.take(2)?;
                let v = u16::from_be_bytes([s[0], s[1]]) as u64;
                if v <= u8::MAX as u64 {
                    return Err(Error::NonCanonicalInt);
                }
                v
            }
            26 => {
                let s = self.take(4)?;
                let v = u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as u64;
                if v <= u16::MAX as u64 {
                    return Err(Error::NonCanonicalInt);
                }
                v
            }
            27 => {
                let s = self.take(8)?;
                let v = u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
                if v <= u32::MAX as u64 {
                    return Err(Error::NonCanonicalInt);
                }
                v
            }
            31 => return Err(Error::IndefiniteLength),
            _ => return Err(Error::Reserved(ib)),
        };
        Ok((major, arg))
    }

    fn value(&mut self, depth: usize) -> Result<Value, Error> {
        if depth > MAX_DEPTH {
            return Err(Error::DepthExceeded);
        }
        let start = self.p;
        let ib = *self.b.get(self.p).ok_or(Error::Eof)?;

        // Major type 7 needs special handling before the generic head reader.
        if ib >> 5 == 7 {
            self.p += 1;
            return match ib & 0x1f {
                20 => Ok(Value::Bool(false)),
                21 => Ok(Value::Bool(true)),
                22 => Ok(Value::Null),
                25 => {
                    let s = self.take(2)?;
                    let h = u16::from_be_bytes([s[0], s[1]]);
                    let v = f16_to_f64(h);
                    // Canonicity: re-encoding must reproduce these bytes.
                    let mut re = Vec::new();
                    encode_float(&mut re, v);
                    if re != self.b[start..self.p] {
                        return Err(Error::NonCanonicalFloat);
                    }
                    Ok(Value::F64(v))
                }
                26 => {
                    let s = self.take(4)?;
                    let v = f32::from_be_bytes([s[0], s[1], s[2], s[3]]) as f64;
                    let mut re = Vec::new();
                    encode_float(&mut re, v);
                    if re != self.b[start..self.p] {
                        return Err(Error::NonCanonicalFloat);
                    }
                    Ok(Value::F64(v))
                }
                27 => {
                    let s = self.take(8)?;
                    let v = f64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]);
                    let mut re = Vec::new();
                    encode_float(&mut re, v);
                    if re != self.b[start..self.p] {
                        return Err(Error::NonCanonicalFloat);
                    }
                    Ok(Value::F64(v))
                }
                _ => Err(Error::Reserved(ib)),
            };
        }

        let (major, arg) = self.head()?;
        match major {
            0 => Ok(Value::U(arg)),
            1 => {
                if arg > i64::MAX as u64 {
                    return Err(Error::LengthOverflow);
                }
                Ok(Value::I(-1 - arg as i64))
            }
            2 => Ok(Value::Bytes(self.take(arg as usize)?.to_vec())),
            3 => {
                let s = self.take(arg as usize)?;
                Ok(Value::Text(
                    std::str::from_utf8(s)
                        .map_err(|_| Error::BadUtf8)?
                        .to_string(),
                ))
            }
            4 => {
                let n = arg as usize;
                // Cheap sanity bound: each element is ≥1 byte.
                if n > self.b.len() - self.p {
                    return Err(Error::LengthOverflow);
                }
                let mut a = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    a.push(self.value(depth + 1)?);
                }
                Ok(Value::Array(a))
            }
            5 => {
                let n = arg as usize;
                if n > (self.b.len() - self.p) / 2 + 1 {
                    return Err(Error::LengthOverflow);
                }
                let mut m: Vec<(Value, Value)> = Vec::with_capacity(n.min(1024));
                let mut prev: Option<Vec<u8>> = None;
                for _ in 0..n {
                    let kstart = self.p;
                    let k = self.value(depth + 1)?;
                    let kbytes = self.b[kstart..self.p].to_vec();
                    if let Some(p) = &prev {
                        match kbytes.cmp(p) {
                            std::cmp::Ordering::Less => return Err(Error::UnsortedKeys),
                            std::cmp::Ordering::Equal => return Err(Error::DuplicateKey),
                            std::cmp::Ordering::Greater => {}
                        }
                    }
                    prev = Some(kbytes);
                    let v = self.value(depth + 1)?;
                    m.push((k, v));
                }
                Ok(Value::Map(m))
            }
            6 => {
                if !tag_is_registered(arg) {
                    return Err(Error::UnregisteredTag(arg));
                }
                Ok(Value::Tag(arg, Box::new(self.value(depth + 1)?)))
            }
            _ => unreachable!("major 7 handled above"),
        }
    }
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as f64;
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        31 => {
            if mant == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// Strict canonical decode. Rejects any non-canonical input and any trailing
/// bytes (D8).
pub fn decode(bytes: &[u8]) -> Result<Value, Error> {
    let mut d = Dec { b: bytes, p: 0 };
    let v = d.value(0)?;
    if d.p != bytes.len() {
        return Err(Error::Trailing(bytes.len() - d.p));
    }
    Ok(v)
}

// ------------------------------------------------------------- ergonomics --

impl Value {
    pub fn map(pairs: Vec<(&str, Value)>) -> Value {
        Value::Map(
            pairs
                .into_iter()
                .map(|(k, v)| (Value::Text(k.to_string()), v))
                .collect(),
        )
    }

    pub fn text(s: impl Into<String>) -> Value {
        Value::Text(s.into())
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m
                .iter()
                .find(|(k, _)| matches!(k, Value::Text(t) if t == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&[(Value, Value)]> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    /// Renders CBOR diagnostic notation (RFC 8949 §8), as used throughout the
    /// specification's examples.
    pub fn diag(&self) -> String {
        match self {
            Value::U(n) => n.to_string(),
            Value::I(n) => n.to_string(),
            Value::Bytes(b) => format!("h'{}'", crate::sha256::hex(b)),
            Value::Text(s) => format!("{:?}", s),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".into(),
            Value::F64(f) => {
                if f.fract() == 0.0 && f.abs() < 1e15 {
                    format!("{:.1}", f)
                } else {
                    format!("{}", f)
                }
            }
            Value::Tag(t, v) => format!("{}({})", t, v.diag()),
            Value::Array(a) => {
                let inner: Vec<String> = a.iter().map(|v| v.diag()).collect();
                format!("[{}]", inner.join(", "))
            }
            Value::Map(m) => {
                let mut items: Vec<(Vec<u8>, &(Value, Value))> =
                    m.iter().map(|kv| (kv.0.encode(), kv)).collect();
                items.sort_by(|a, b| a.0.cmp(&b.0));
                let inner: Vec<String> = items
                    .iter()
                    .map(|(_, (k, v))| format!("{}: {}", k.diag(), v.diag()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip through the canonical form. Note that `Value::Map` preserves
    /// insertion order in memory while encoding sorts (D3), so equality is
    /// checked on the canonical bytes rather than on the in-memory value.
    fn rt(v: Value) {
        let e = v.encode();
        let d = decode(&e).expect("decode");
        assert_eq!(d.encode(), e, "re-encode must be byte-identical");
        assert_eq!(decode(&d.encode()).unwrap(), d, "decode is idempotent");
        if !matches!(v, Value::Map(_)) {
            assert_eq!(d, v);
        }
    }

    #[test]
    fn roundtrips() {
        rt(Value::U(0));
        rt(Value::U(23));
        rt(Value::U(24));
        rt(Value::U(u64::MAX));
        rt(Value::I(-1));
        rt(Value::I(-1000));
        rt(Value::Bytes(vec![1, 2, 3]));
        rt(Value::Text("hello".into()));
        rt(Value::Bool(true));
        rt(Value::Null);
        rt(Value::F64(1.5));
        rt(Value::F64(0.0));
        rt(Value::F64(-0.0));
        rt(Value::F64(1e300));
        rt(Value::F64(0.1));
        rt(Value::Array(vec![Value::U(1), Value::Text("x".into())]));
        rt(Value::map(vec![
            ("z", Value::U(1)),
            ("a", Value::U(2)),
            ("mm", Value::U(3)),
        ]));
    }

    #[test]
    fn rfc8949_appendix_a_vectors() {
        assert_eq!(Value::U(0).encode(), vec![0x00]);
        assert_eq!(Value::U(10).encode(), vec![0x0a]);
        assert_eq!(Value::U(100).encode(), vec![0x18, 0x64]);
        assert_eq!(Value::U(1000).encode(), vec![0x19, 0x03, 0xe8]);
        assert_eq!(Value::I(-1).encode(), vec![0x20]);
        assert_eq!(Value::I(-100).encode(), vec![0x38, 0x63]);
        assert_eq!(Value::Text("a".into()).encode(), vec![0x61, 0x61]);
        assert_eq!(Value::F64(1.0).encode(), vec![0xf9, 0x3c, 0x00]);
        assert_eq!(Value::F64(1.5).encode(), vec![0xf9, 0x3e, 0x00]);
        assert_eq!(
            Value::F64(100000.0).encode(),
            vec![0xfa, 0x47, 0xc3, 0x50, 0x00]
        );
        assert_eq!(
            Value::F64(1.1).encode(),
            vec![0xfb, 0x3f, 0xf1, 0x99, 0x99, 0x99, 0x99, 0x99, 0x9a]
        );
    }

    #[test]
    fn map_keys_are_sorted_by_encoded_bytes() {
        let v = Value::map(vec![
            ("bb", Value::U(1)),
            ("a", Value::U(2)),
            ("ccc", Value::U(3)),
        ]);
        let e = v.encode();
        // Shorter text strings encode to shorter byte sequences, which sort
        // first: "a" (0x61 61) < "bb" (0x62 62 62) < "ccc" (0x63 ...).
        let d = decode(&e).unwrap();
        let m = d.as_map().unwrap();
        assert_eq!(m[0].0.as_str(), Some("a"));
        assert_eq!(m[1].0.as_str(), Some("bb"));
        assert_eq!(m[2].0.as_str(), Some("ccc"));
    }

    #[test]
    fn rejects_non_canonical() {
        // D1: 0 encoded in two bytes.
        assert!(matches!(decode(&[0x18, 0x00]), Err(Error::NonCanonicalInt)));
        // D2: indefinite-length array.
        assert!(matches!(
            decode(&[0x9f, 0x01, 0xff]),
            Err(Error::IndefiniteLength)
        ));
        // D8: trailing bytes.
        assert!(matches!(decode(&[0x00, 0x00]), Err(Error::Trailing(1))));
        // D3: unsorted map keys {"b":1,"a":2}.
        assert!(matches!(
            decode(&[0xa2, 0x61, 0x62, 0x01, 0x61, 0x61, 0x02]),
            Err(Error::UnsortedKeys)
        ));
        // D4: duplicate keys {"a":1,"a":2}.
        assert!(matches!(
            decode(&[0xa2, 0x61, 0x61, 0x01, 0x61, 0x61, 0x02]),
            Err(Error::DuplicateKey)
        ));
        // D5: 1.0 encoded as f64 instead of f16.
        assert!(matches!(
            decode(&[0xfb, 0x3f, 0xf0, 0, 0, 0, 0, 0, 0]),
            Err(Error::NonCanonicalFloat)
        ));
        // D7: unregistered tag.
        assert!(matches!(
            decode(&[0xc1, 0x00]),
            Err(Error::UnregisteredTag(1))
        ));
    }

    #[test]
    fn depth_is_bounded() {
        let mut bytes = vec![0x81u8; MAX_DEPTH + 5];
        bytes.push(0x00);
        assert!(matches!(decode(&bytes), Err(Error::DepthExceeded)));
    }

    #[test]
    fn length_overflow_is_caught_not_allocated() {
        // Byte string declaring 2^32 bytes in a 5-byte input.
        assert!(matches!(
            decode(&[0x5a, 0xff, 0xff, 0xff, 0xff]),
            Err(Error::LengthOverflow)
        ));
        // Array declaring 2^32 elements.
        assert!(matches!(
            decode(&[0x9a, 0xff, 0xff, 0xff, 0xff]),
            Err(Error::LengthOverflow)
        ));
    }
}
