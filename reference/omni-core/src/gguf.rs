//! GGUF import and export (§05.2.4; `docs/design/import-export.md` §3).
//!
//! GGUF is a header of key/value metadata followed by a table of tensors, each
//! of which is either a dense array or a sequence of fixed-size *blocks*: a few
//! bytes of scale, then the quantized values, then — for the K-quants — a
//! second level of per-sub-block scales packed six bits at a time. §05.2.4
//! claims all of it is expressible in the core algebra, with no new mechanism
//! and no special case in the evaluator. This module is where that claim is
//! either true or false.
//!
//! ## The shape of the mapping
//!
//! A block is a struct, and a struct of arrays is what the algebra can read. So
//! an imported tensor keeps **every source byte, regrouped by field**: the
//! `d` scales of all blocks in one literal, the `dmin` minima in another, the
//! packed quantized values in a third. Each field is read back with the dtype
//! and [`Layout::Packed`] that names its bit width, and the arithmetic on top
//! is one or two `dequantize` nodes from §05.1's closed set of formulas.
//!
//! Nothing is re-encoded. The 6-bit sub-scales of `Q4_K`, which llama.cpp
//! unpacks with four masked shifts, are read as three literals over the same
//! twelve bytes at three different bit widths — `u6` for the four that are a
//! whole low six bits, `u4` and `u2` for the eight that straddle — and
//! recombined with `add` and `scale`. That matters for more than tidiness: it
//! means an export is a byte-for-byte re-interleave of what was stored, so the
//! round trip is exact by construction rather than by careful rounding.
//!
//! ## What the element order costs
//!
//! Every GGML type interleaves its values differently — `Q4_0` puts elements
//! *i* and *i+16* in one byte, `Q6_K` scatters four elements across a nibble
//! pair and a two-bit field, `Q2_K` runs a shift as its slowest axis. Each of
//! those is a `permute` of a packed literal whose axes are (word, slot), which
//! is the same trick GPTQ's and AWQ's importers use for their packings. The
//! permutations are stated once, in [`value_expr`], next to the loop from
//! `ggml-quants.c` they undo.
//!
//! ## What is refused rather than approximated
//!
//! The `IQ*` types, whose values are indices into a fixed lattice codebook that
//! is compiled into llama.cpp rather than stored in the file: §05.6 rule 1 says
//! an importer that does not know the exact dequantization must not guess one,
//! and a codebook this build would have to hardcode from memory is exactly that
//! case. `Q8_K`, which llama.cpp uses as an intermediate and never writes to a
//! file. GGUF v1, whose string and array lengths are 32-bit. The repacked
//! `Q4_0_4_4`/`_4_8`/`_8_8` types, which are a CPU-specific reordering of
//! `Q4_0` that the format itself calls deprecated. Each is refused by name.
//!
//! ## What is verified
//!
//! Two checks, both counted in the fidelity report (I4):
//!
//! 1. **Block reassembly.** Every quantized tensor is rebuilt from the fields
//!    the import wrote and compared with the source bytes. This is the export
//!    path, run at import time, so "the round trip is bit-exact" is a measured
//!    statement about this file rather than a property claimed for the format.
//! 2. **Independent dequantization.** Every block of every quantized tensor is
//!    dequantized twice: once by evaluating the expression graph, and once by
//!    [`reference_dequant`], scalar code written from the GGML block layouts
//!    that shares nothing with the evaluator. A wrong permutation, a swapped
//!    nibble or a sub-scale read at the wrong bit offset produces plausible
//!    numbers and is invisible to check 1; this is the check that sees it.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::expr::{dims, BinOp, Ctx, Expr, Scalar};
use crate::json;
use crate::layout::{BitOrder, Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{Fidelity, Note};
use crate::tensor::{Materialize, TensorDesc, TensorTable};

pub const IMPORTER: &str = "omni-import-gguf";
pub const EXPORTER: &str = "omni-export-gguf";

/// The GGUF magic, little-endian: `GGUF`.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// The alignment GGUF assumes when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    /// Well-formed, and says something this build will not approximate.
    Unsupported(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed gguf: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// ----------------------------------------------------------------- ggml types --

/// A GGML tensor type, by its wire number.
///
/// The numbers are the format's, not this build's: an unknown one is carried as
/// [`Type::Other`] so that a file from a newer llama.cpp is *reported* rather
/// than mis-read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Other(u32),
}

impl Type {
    pub fn from_u32(v: u32) -> Type {
        match v {
            0 => Type::F32,
            1 => Type::F16,
            2 => Type::Q4_0,
            3 => Type::Q4_1,
            6 => Type::Q5_0,
            7 => Type::Q5_1,
            8 => Type::Q8_0,
            9 => Type::Q8_1,
            10 => Type::Q2K,
            11 => Type::Q3K,
            12 => Type::Q4K,
            13 => Type::Q5K,
            14 => Type::Q6K,
            15 => Type::Q8K,
            24 => Type::I8,
            25 => Type::I16,
            26 => Type::I32,
            27 => Type::I64,
            28 => Type::F64,
            30 => Type::BF16,
            other => Type::Other(other),
        }
    }

    pub fn to_u32(self) -> u32 {
        match self {
            Type::F32 => 0,
            Type::F16 => 1,
            Type::Q4_0 => 2,
            Type::Q4_1 => 3,
            Type::Q5_0 => 6,
            Type::Q5_1 => 7,
            Type::Q8_0 => 8,
            Type::Q8_1 => 9,
            Type::Q2K => 10,
            Type::Q3K => 11,
            Type::Q4K => 12,
            Type::Q5K => 13,
            Type::Q6K => 14,
            Type::Q8K => 15,
            Type::I8 => 24,
            Type::I16 => 25,
            Type::I32 => 26,
            Type::I64 => 27,
            Type::F64 => 28,
            Type::BF16 => 30,
            Type::Other(v) => v,
        }
    }

    pub fn name(self) -> String {
        match self {
            Type::F32 => "F32".into(),
            Type::F16 => "F16".into(),
            Type::Q4_0 => "Q4_0".into(),
            Type::Q4_1 => "Q4_1".into(),
            Type::Q5_0 => "Q5_0".into(),
            Type::Q5_1 => "Q5_1".into(),
            Type::Q8_0 => "Q8_0".into(),
            Type::Q8_1 => "Q8_1".into(),
            Type::Q2K => "Q2_K".into(),
            Type::Q3K => "Q3_K".into(),
            Type::Q4K => "Q4_K".into(),
            Type::Q5K => "Q5_K".into(),
            Type::Q6K => "Q6_K".into(),
            Type::Q8K => "Q8_K".into(),
            Type::I8 => "I8".into(),
            Type::I16 => "I16".into(),
            Type::I32 => "I32".into(),
            Type::I64 => "I64".into(),
            Type::F64 => "F64".into(),
            Type::BF16 => "BF16".into(),
            Type::Other(v) => match v {
                4 => "Q4_2 (removed)".into(),
                5 => "Q4_3 (removed)".into(),
                16 => "IQ2_XXS".into(),
                17 => "IQ2_XS".into(),
                18 => "IQ3_XXS".into(),
                19 => "IQ1_S".into(),
                20 => "IQ4_NL".into(),
                21 => "IQ3_S".into(),
                22 => "IQ2_S".into(),
                23 => "IQ4_XS".into(),
                29 => "IQ1_M".into(),
                31 => "Q4_0_4_4".into(),
                32 => "Q4_0_4_8".into(),
                33 => "Q4_0_8_8".into(),
                34 => "TQ1_0".into(),
                35 => "TQ2_0".into(),
                other => format!("ggml type {other}"),
            },
        }
    }

    /// The dense dtype, for the types that are not blocked.
    pub fn dense_dtype(self) -> Option<DType> {
        Some(match self {
            Type::F32 => DType::F32,
            Type::F16 => DType::F16,
            Type::F64 => DType::F64,
            Type::BF16 => DType::BF16,
            Type::I8 => DType::I8,
            Type::I16 => DType::I16,
            Type::I32 => DType::I32,
            Type::I64 => DType::I64,
            _ => return None,
        })
    }

    /// Elements per block, and bytes per block. `None` for the types this build
    /// refuses; the refusal names them one at a time in [`check_supported`].
    pub fn block(self) -> Option<(u64, u64)> {
        Some(match self {
            Type::F32 => (1, 4),
            Type::F16 | Type::BF16 => (1, 2),
            Type::F64 => (1, 8),
            Type::I8 => (1, 1),
            Type::I16 => (1, 2),
            Type::I32 => (1, 4),
            Type::I64 => (1, 8),
            Type::Q4_0 => (32, 18),
            Type::Q4_1 => (32, 20),
            Type::Q5_0 => (32, 22),
            Type::Q5_1 => (32, 24),
            Type::Q8_0 => (32, 34),
            Type::Q8_1 => (32, 36),
            Type::Q2K => (256, 84),
            Type::Q3K => (256, 110),
            Type::Q4K => (256, 144),
            Type::Q5K => (256, 176),
            Type::Q6K => (256, 210),
            Type::Q8K | Type::Other(_) => return None,
        })
    }

    pub fn is_quantized(self) -> bool {
        self.dense_dtype().is_none()
    }

    /// The stored size of `numel` elements of this type.
    pub fn stored_bytes(self, numel: u64) -> Option<u64> {
        let (be, bb) = self.block()?;
        Some(numel.div_ceil(be) * bb)
    }
}

/// Why this build will not read a type, said one type at a time.
fn check_supported(t: Type) -> Res<()> {
    if t.block().is_some() {
        return Ok(());
    }
    let why = match t {
        Type::Q8K => "`Q8_K` is llama.cpp's intermediate quantization for dot \
                      products and is never written to a file; a file that \
                      contains one is either from a tool this build does not \
                      know or corrupt"
            .to_string(),
        Type::Other(v @ (16..=23 | 29)) => format!(
            "`{}` indexes a fixed lattice codebook that lives in llama.cpp's \
             source rather than in the file. §05.6 rule 1 forbids inventing a \
             dequantization, and a table reproduced from memory is exactly \
             that; importing it needs the codebook as data (§05.4), which the \
             file does not carry",
            Type::Other(v).name()
        ),
        Type::Other(v @ (31..=33)) => format!(
            "`{}` is `Q4_0` reordered for one CPU's kernels; the format calls \
             it deprecated and llama.cpp repacks at load time, so an importer \
             that read it would be preserving a machine's cache layout as if \
             it were the model",
            Type::Other(v).name()
        ),
        Type::Other(v @ (34 | 35)) => format!(
            "`{}` is a ternary format whose block layout this build has not \
             implemented; §05.2.9 says how it would be expressed",
            Type::Other(v).name()
        ),
        Type::Other(v @ (4 | 5)) => format!(
            "`{}` was removed from GGML before GGUF existed",
            Type::Other(v).name()
        ),
        other => format!("{} is not a type this build knows", other.name()),
    };
    Err(Error::Unsupported(why))
}

// ------------------------------------------------------------------ metadata --

/// A GGUF metadata value. The wire types are kept distinct — a `u32` and an
/// `i32` are different keys' worth of meaning, and an export has to write back
/// the one it read.
#[derive(Clone, Debug, PartialEq)]
pub enum Meta {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// `(element type, values)`. Empty arrays keep their element type, which is
    /// why it is stored rather than derived.
    Arr(u32, Vec<Meta>),
}

impl Meta {
    pub fn wire_type(&self) -> u32 {
        match self {
            Meta::U8(_) => 0,
            Meta::I8(_) => 1,
            Meta::U16(_) => 2,
            Meta::I16(_) => 3,
            Meta::U32(_) => 4,
            Meta::I32(_) => 5,
            Meta::F32(_) => 6,
            Meta::Bool(_) => 7,
            Meta::Str(_) => 8,
            Meta::Arr(..) => 9,
            Meta::U64(_) => 10,
            Meta::I64(_) => 11,
            Meta::F64(_) => 12,
        }
    }

    /// The value as an unsigned integer, when it is one.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Meta::U8(v) => *v as u64,
            Meta::U16(v) => *v as u64,
            Meta::U32(v) => *v as u64,
            Meta::U64(v) => *v,
            Meta::I8(v) if *v >= 0 => *v as u64,
            Meta::I16(v) if *v >= 0 => *v as u64,
            Meta::I32(v) if *v >= 0 => *v as u64,
            Meta::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match self {
            Meta::F32(v) => *v as f64,
            Meta::F64(v) => *v,
            other => other.as_u64()? as f64,
        })
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Meta::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Meta]> {
        match self {
            Meta::Arr(_, v) => Some(v),
            _ => None,
        }
    }

    /// The CBOR form written into the `Foreign` object, so that an export can
    /// reproduce the key exactly: `[wire type, value]`.
    pub fn to_value(&self) -> Value {
        let v = match self {
            Meta::U8(v) => Value::U(*v as u64),
            Meta::U16(v) => Value::U(*v as u64),
            Meta::U32(v) => Value::U(*v as u64),
            Meta::U64(v) => Value::U(*v),
            Meta::I8(v) => Value::I(*v as i64),
            Meta::I16(v) => Value::I(*v as i64),
            Meta::I32(v) => Value::I(*v as i64),
            Meta::I64(v) => Value::I(*v),
            Meta::F32(v) => Value::F64(*v as f64),
            Meta::F64(v) => Value::F64(*v),
            Meta::Bool(v) => Value::Bool(*v),
            Meta::Str(s) => Value::text(s.clone()),
            Meta::Arr(_, xs) => Value::Array(xs.iter().map(|x| x.to_value()).collect()),
        };
        match self {
            Meta::Arr(elem, _) => Value::Array(vec![
                Value::U(self.wire_type() as u64),
                v,
                Value::U(*elem as u64),
            ]),
            _ => Value::Array(vec![Value::U(self.wire_type() as u64), v]),
        }
    }

    /// The inverse of [`Meta::to_value`].
    pub fn from_value(v: &Value) -> Res<Meta> {
        let a = v
            .as_array()
            .ok_or_else(|| Error::Malformed("metadata value must be [type, value]".into()))?;
        let t = a
            .first()
            .and_then(|x| x.as_u64())
            .ok_or_else(|| Error::Malformed("metadata value has no wire type".into()))?;
        let val = a
            .get(1)
            .ok_or_else(|| Error::Malformed("metadata value has no value".into()))?;
        let int = |x: &Value| -> Res<i64> {
            match x {
                Value::U(u) => Ok(*u as i64),
                Value::I(i) => Ok(*i),
                _ => Err(Error::Malformed("metadata integer expected".into())),
            }
        };
        Ok(match t {
            0 => Meta::U8(int(val)? as u8),
            1 => Meta::I8(int(val)? as i8),
            2 => Meta::U16(int(val)? as u16),
            3 => Meta::I16(int(val)? as i16),
            4 => Meta::U32(int(val)? as u32),
            5 => Meta::I32(int(val)? as i32),
            6 => Meta::F32(
                num_f64(val).ok_or_else(|| Error::Malformed("f32 expected".into()))? as f32,
            ),
            7 => Meta::Bool(matches!(val, Value::Bool(true))),
            8 => Meta::Str(
                val.as_str()
                    .ok_or_else(|| Error::Malformed("string expected".into()))?
                    .to_string(),
            ),
            9 => {
                let elem = a
                    .get(2)
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| Error::Malformed("array has no element type".into()))?;
                let xs = val
                    .as_array()
                    .ok_or_else(|| Error::Malformed("array expected".into()))?;
                Meta::Arr(
                    elem as u32,
                    xs.iter().map(Meta::from_value).collect::<Res<Vec<_>>>()?,
                )
            }
            10 => Meta::U64(int(val)? as u64),
            11 => Meta::I64(int(val)?),
            12 => Meta::F64(num_f64(val).ok_or_else(|| Error::Malformed("f64 expected".into()))?),
            other => {
                return Err(Error::Malformed(format!(
                    "metadata wire type {other} is not one of GGUF's twelve"
                )))
            }
        })
    }
}

fn num_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::U(u) => Some(*u as f64),
        Value::I(i) => Some(*i as f64),
        _ => None,
    }
}

// -------------------------------------------------------------------- parsing --

/// One tensor's entry in the GGUF tensor table.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    /// Extents in GGUF order: `dims[0]` is the fastest-varying axis.
    pub dims: Vec<u64>,
    pub ty: Type,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
}

impl Entry {
    /// The row-major shape, which is GGUF's reversed: llama.cpp's `ne[0]` is the
    /// number of columns.
    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }

    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A parsed GGUF file. Borrowing rather than copying: a 40 GB model is not
/// going to be read into a second buffer to be looked at.
pub struct File<'a> {
    pub version: u32,
    pub kv: Vec<(String, Meta)>,
    pub tensors: Vec<Entry>,
    pub alignment: u64,
    /// Offset of the tensor data section.
    pub data_start: u64,
    /// Offset just past the tensor table, before alignment padding. The gap
    /// between the two is padding whose content GGUF does not specify, so an
    /// export has to say whether it is reproducing it or writing zeros.
    pub header_end: u64,
    bytes: &'a [u8],
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or_else(|| {
            Error::Malformed(format!("a length of {n} bytes at {} overflows", self.at))
        })?;
        if end > self.b.len() {
            return Err(Error::Malformed(format!(
                "wants {n} bytes at {} and the file has {}",
                self.at,
                self.b.len()
            )));
        }
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Res<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Res<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// A GGUF string: a 64-bit length and that many bytes of UTF-8.
    fn string(&mut self, limit: u64) -> Res<String> {
        let n = self.u64()?;
        if n > limit {
            return Err(Error::Malformed(format!(
                "a {n}-byte string in a {limit}-byte file"
            )));
        }
        let b = self.take(n as usize)?;
        String::from_utf8(b.to_vec())
            .map_err(|_| Error::Malformed("a metadata string is not UTF-8".into()))
    }

    fn meta(&mut self, ty: u32, limit: u64, depth: u32) -> Res<Meta> {
        Ok(match ty {
            0 => Meta::U8(self.u8()?),
            1 => Meta::I8(self.u8()? as i8),
            2 => Meta::U16(self.u16()?),
            3 => Meta::I16(self.u16()? as i16),
            4 => Meta::U32(self.u32()?),
            5 => Meta::I32(self.u32()? as i32),
            6 => Meta::F32(self.f32()?),
            7 => Meta::Bool(self.u8()? != 0),
            8 => Meta::Str(self.string(limit)?),
            9 => {
                // §12.4: an array of arrays is legal on the wire and is not
                // something any producer writes, so the recursion is bounded
                // rather than trusted.
                if depth > 0 {
                    return Err(Error::Unsupported(
                        "a nested metadata array; GGUF allows the encoding and \
                         nothing writes it, so this build refuses it rather \
                         than recursing on a length it has not checked"
                            .into(),
                    ));
                }
                let elem = self.u32()?;
                let n = self.u64()?;
                if n > limit {
                    return Err(Error::Malformed(format!(
                        "an array of {n} elements in a {limit}-byte file"
                    )));
                }
                let mut xs = Vec::with_capacity((n as usize).min(1 << 16));
                for _ in 0..n {
                    xs.push(self.meta(elem, limit, depth + 1)?);
                }
                Meta::Arr(elem, xs)
            }
            10 => Meta::U64(self.u64()?),
            11 => Meta::I64(self.u64()? as i64),
            12 => Meta::F64(self.f64()?),
            other => {
                return Err(Error::Malformed(format!(
                    "metadata type {other} is not one of GGUF's twelve"
                )))
            }
        })
    }
}

impl<'a> File<'a> {
    pub fn parse(bytes: &'a [u8]) -> Res<File<'a>> {
        let mut c = Cursor { b: bytes, at: 0 };
        if c.take(4)? != MAGIC {
            return Err(Error::Malformed(
                "no `GGUF` magic; this is not a GGUF file".into(),
            ));
        }
        let version = c.u32()?;
        if version < 2 {
            return Err(Error::Unsupported(format!(
                "GGUF v{version}: its string and array lengths are 32-bit, and \
                 no model published since 2023 uses it"
            )));
        }
        if version > 3 {
            return Err(Error::Unsupported(format!(
                "GGUF v{version} is newer than the v3 this build implements; \
                 refusing rather than reading a v3 header out of it"
            )));
        }
        let limit = bytes.len() as u64;
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        // Both counts bound allocations, so both are checked against the file
        // size before anything is reserved (§12.4).
        if n_tensors > limit / 8 || n_kv > limit / 8 {
            return Err(Error::Malformed(format!(
                "{n_tensors} tensors and {n_kv} metadata keys do not fit in \
                 {limit} bytes"
            )));
        }

        let mut kv = Vec::with_capacity((n_kv as usize).min(1 << 12));
        for _ in 0..n_kv {
            let key = c.string(limit)?;
            let ty = c.u32()?;
            let val = c.meta(ty, limit, 0)?;
            kv.push((key, val));
        }

        let mut tensors = Vec::with_capacity((n_tensors as usize).min(1 << 16));
        for _ in 0..n_tensors {
            let name = c.string(limit)?;
            let n_dims = c.u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err(Error::Malformed(format!(
                    "`{name}` has {n_dims} dimensions; GGML tensors have one to four"
                )));
            }
            let mut d = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                d.push(c.u64()?);
            }
            let ty = Type::from_u32(c.u32()?);
            let offset = c.u64()?;
            tensors.push(Entry {
                name,
                dims: d,
                ty,
                offset,
            });
        }

        let alignment = kv
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Malformed(format!(
                "`general.alignment` is {alignment}, and an alignment is a power of two"
            )));
        }
        let data_start = (c.at as u64).div_ceil(alignment) * alignment;
        if data_start > limit {
            return Err(Error::Malformed(
                "the header runs past the end of the file".into(),
            ));
        }

        let f = File {
            version,
            kv,
            tensors,
            alignment,
            data_start,
            header_end: c.at as u64,
            bytes,
        };
        // Every tensor's extent is checked here rather than at read time, so a
        // caller that iterates the table cannot be handed a slice that is short.
        for e in &f.tensors {
            let (be, _) = match e.ty.block() {
                Some(b) => b,
                // An unsupported type still has to be *located* to be reported,
                // and its size is unknown, so the check is deferred to the point
                // of use, which refuses it by name.
                None => continue,
            };
            if be > 1 && e.dims[0] % be != 0 {
                return Err(Error::Malformed(format!(
                    "`{}` is {} with {} columns, which is not a multiple of its \
                     {be}-element block",
                    e.name,
                    e.ty.name(),
                    e.dims[0]
                )));
            }
            let n =
                e.ty.stored_bytes(e.numel())
                    .ok_or_else(|| Error::Malformed(format!("`{}` has no size", e.name)))?;
            let end = f
                .data_start
                .checked_add(e.offset)
                .and_then(|s| s.checked_add(n))
                .ok_or_else(|| Error::Malformed(format!("`{}` overflows the file", e.name)))?;
            if end > limit {
                return Err(Error::Malformed(format!(
                    "`{}` ends at {end} and the file is {limit} bytes",
                    e.name
                )));
            }
        }
        Ok(f)
    }

    /// The stored bytes of one tensor.
    pub fn tensor(&self, e: &Entry) -> Res<&'a [u8]> {
        check_supported(e.ty)?;
        let n = e.ty.stored_bytes(e.numel()).expect("checked") as usize;
        let at = (self.data_start + e.offset) as usize;
        self.bytes
            .get(at..at + n)
            .ok_or_else(|| Error::Malformed(format!("`{}` is outside the file", e.name)))
    }

    pub fn get(&self, key: &str) -> Option<&Meta> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `general.architecture`, which every other architectural key is prefixed
    /// with.
    pub fn arch(&self) -> Option<&str> {
        self.get("general.architecture").and_then(|m| m.as_str())
    }
}

// ------------------------------------------------------------- block geometry --

/// One field of a block: where it is, how wide, and what it means.
///
/// The table below *is* the mapping of §05.2.4 for this build. Every offset is
/// from `ggml-common.h`'s block structs, and the order of fields is the order
/// they appear in memory, which is what makes reassembly a concatenation.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    /// Byte offset within the block.
    pub off: usize,
    /// Byte length within the block.
    pub len: usize,
}

/// The fields of a quantized block type, in memory order.
///
/// Two of these are finer-grained than the C struct: `Q3_K`'s twelve scale
/// bytes are split at the boundary between the four-bit and two-bit halves of
/// the packing, and `Q4_K`/`Q5_K`'s at the three four-byte groups
/// `get_scale_min_k4` reads. Splitting them costs nothing — a field is a byte
/// range and reassembly is a copy either way — and it is what lets the scales be
/// *read* by a layout instead of unpacked by code.
pub fn fields(t: Type) -> &'static [Field] {
    macro_rules! fs {
        ($(($n:literal, $o:literal, $l:literal)),* $(,)?) => {
            &[$(Field { name: $n, off: $o, len: $l }),*]
        };
    }
    match t {
        // block_q4_0 { ggml_half d; uint8_t qs[16]; }
        Type::Q4_0 => fs![("d", 0, 2), ("qs", 2, 16)],
        // block_q4_1 { ggml_half d, m; uint8_t qs[16]; }
        Type::Q4_1 => fs![("d", 0, 2), ("m", 2, 2), ("qs", 4, 16)],
        // block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; }
        Type::Q5_0 => fs![("d", 0, 2), ("qh", 2, 4), ("qs", 6, 16)],
        // block_q5_1 { ggml_half d, m; uint8_t qh[4]; uint8_t qs[16]; }
        Type::Q5_1 => fs![("d", 0, 2), ("m", 2, 2), ("qh", 4, 4), ("qs", 8, 16)],
        // block_q8_0 { ggml_half d; int8_t qs[32]; }
        Type::Q8_0 => fs![("d", 0, 2), ("qs", 2, 32)],
        // block_q8_1 { ggml_half d, s; int8_t qs[32]; }
        Type::Q8_1 => fs![("d", 0, 2), ("s", 2, 2), ("qs", 4, 32)],
        // block_q2_K { uint8_t scales[16]; uint8_t qs[64]; ggml_half d, dmin; }
        Type::Q2K => fs![
            ("scales", 0, 16),
            ("qs", 16, 64),
            ("d", 80, 2),
            ("dmin", 82, 2)
        ],
        // block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12];
        //              ggml_half d; }
        Type::Q3K => fs![
            ("hmask", 0, 32),
            ("qs", 32, 64),
            ("scales_l", 96, 8),
            ("scales_h", 104, 4),
            ("d", 108, 2),
        ],
        // block_q4_K { ggml_half d, dmin; uint8_t scales[12]; uint8_t qs[128]; }
        Type::Q4K => fs![
            ("d", 0, 2),
            ("dmin", 2, 2),
            ("scales_a", 4, 4),
            ("scales_b", 8, 4),
            ("scales_c", 12, 4),
            ("qs", 16, 128),
        ],
        // block_q5_K { ggml_half d, dmin; uint8_t scales[12]; uint8_t qh[32];
        //              uint8_t qs[128]; }
        Type::Q5K => fs![
            ("d", 0, 2),
            ("dmin", 2, 2),
            ("scales_a", 4, 4),
            ("scales_b", 8, 4),
            ("scales_c", 12, 4),
            ("qh", 16, 32),
            ("qs", 48, 128),
        ],
        // block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
        //              ggml_half d; }
        Type::Q6K => fs![
            ("ql", 0, 128),
            ("qh", 128, 64),
            ("scales", 192, 16),
            ("d", 208, 2),
        ],
        _ => &[],
    }
}

/// Pulls one field out of every block: the transpose that turns an array of
/// structs into a struct of arrays.
fn gather(blocks: &[u8], block_bytes: usize, fld: &Field, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * fld.len);
    for b in 0..n {
        let at = b * block_bytes + fld.off;
        out.extend_from_slice(&blocks[at..at + fld.len]);
    }
    out
}

/// The inverse of [`gather`]: the export path, and the check that the import
/// path lost nothing.
pub fn reassemble(t: Type, parts: &[(&'static str, Vec<u8>)], n: usize) -> Res<Vec<u8>> {
    let (_, bb) = t
        .block()
        .ok_or_else(|| Error::Unsupported(format!("{} has no block layout", t.name())))?;
    let bb = bb as usize;
    let mut out = vec![0u8; n * bb];
    for fld in fields(t) {
        let Some((_, bytes)) = parts.iter().find(|(k, _)| *k == fld.name) else {
            return Err(Error::Malformed(format!(
                "{}: field `{}` is missing",
                t.name(),
                fld.name
            )));
        };
        if bytes.len() != n * fld.len {
            return Err(Error::Malformed(format!(
                "{}: field `{}` has {} bytes and {n} blocks need {}",
                t.name(),
                fld.name,
                bytes.len(),
                n * fld.len
            )));
        }
        for b in 0..n {
            let at = b * bb + fld.off;
            out[at..at + fld.len].copy_from_slice(&bytes[b * fld.len..(b + 1) * fld.len]);
        }
    }
    Ok(out)
}

// ------------------------------------------------------- the value expression --

/// A packed layout: `elems_per_word` values of the literal's dtype in each
/// `word_bits`-bit word, first value in the low bits — which is what every GGML
/// packing does.
fn packed(elems_per_word: u32, word_bits: u32) -> Layout {
    Layout::Packed {
        elems_per_word,
        word_bits,
        bit_order: BitOrder::LsbFirst,
        order: Order::RowMajor,
    }
}

fn uint(w: u16) -> DType {
    DType::Int { w, signed: false }
}

fn reshape(x: Expr, shape: &[u64]) -> Expr {
    Expr::Reshape {
        x: Box::new(x),
        shape: dims(shape),
    }
}

fn permute(x: Expr, perm: &[usize]) -> Expr {
    Expr::Permute {
        x: Box::new(x),
        perm: perm.to_vec(),
    }
}

fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::Bin {
        op,
        a: Box::new(a),
        b: Box::new(b),
    }
}

fn scale(x: Expr, k: i64) -> Expr {
    Expr::Scale {
        x: Box::new(x),
        k: Scalar::Int(k),
    }
}

fn cast(x: Expr, d: DType) -> Expr {
    Expr::Cast {
        x: Box::new(x),
        dtype: d,
        round: crate::dtype::Round::Rne,
    }
}

fn full(v: i64, d: DType) -> Expr {
    Expr::Full {
        value: Scalar::Int(v),
        dtype: d,
        shape: dims(&[1, 1]),
    }
}

/// One slot of a packed field, as a tensor of one fewer axis: `x[.., .., slot]`.
fn slot(x: Expr, shape: &[u64], axis_len: u64, which: u64) -> Expr {
    let mut starts = vec![0u64; shape.len()];
    let mut sizes = shape.to_vec();
    starts[shape.len() - 1] = which;
    sizes[shape.len() - 1] = 1;
    let _ = axis_len;
    reshape(
        Expr::Slice {
            x: Box::new(x),
            starts,
            sizes,
            steps: vec![1; shape.len()],
        },
        &shape[..shape.len() - 1],
    )
}

/// The stored fields of one imported tensor, by name.
pub struct Stored {
    pub refs: Vec<(&'static str, crate::expr::Ref)>,
}

impl Stored {
    fn lit(&self, name: &str, dtype: DType, shape: &[u64], layout: Layout) -> Expr {
        let chunks = self
            .refs
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, r)| *r)
            .expect("every field named here was stored by `import_tensor`");
        Expr::Literal {
            chunks,
            dtype,
            shape: dims(shape),
            layout,
        }
    }

    fn f16(&self, name: &str, nb: u64) -> Expr {
        self.lit(name, DType::F16, &[nb], Layout::default())
    }
}

/// The dequantization expression for one block type (§05.2.4), over the fields
/// [`import_tensor`] stored, producing a `[nb·block]`-element f32 tensor.
///
/// Every permutation below undoes an interleave from `ggml-quants.c`'s
/// `dequantize_row_*`, and the comment above it is the loop it undoes. Reading
/// them side by side is the point: this is the whole of §05.2.4's claim, and if
/// one of these is wrong the check in [`import`] says so with an element index.
pub fn value_expr(t: Type, s: &Stored, nb: u64) -> Res<Expr> {
    let f32_out = DType::F32.to_value();
    // A dequantize node with `block` over the last axis (or last two, for the
    // super-block types), whose scale and zero are expressions over the stored
    // fields rather than numbers baked in here.
    let deq = |x: Expr, formula: &str, block: &[u64], sc: Expr, zero: Option<Expr>| -> Expr {
        let mut m = vec![
            (
                "scheme",
                Value::text(if formula == "sym" { "sym" } else { "affine" }),
            ),
            ("formula", Value::text(formula.to_string())),
            ("out", f32_out.clone()),
            ("axis", Value::U(block.len() as u64 - 1)),
            (
                "block",
                Value::Array(block.iter().map(|d| Value::U(*d)).collect()),
            ),
            ("scale", sc.to_value()),
        ];
        match zero {
            Some(z) => m.push(("zero", z.to_value())),
            None => m.push(("sym", Value::Bool(true))),
        }
        Expr::Dequantize {
            x: Box::new(x),
            scheme: Value::map(m),
        }
    };

    // The 4-bit values of a 32-element block: `qs[i] & 0xF` is element `i`,
    // `qs[i] >> 4` is element `i + 16`, so the slot is the *slow* axis of the
    // output and a permute is the whole of the un-interleaving.
    let nibbles32 = |field: &str| -> Expr {
        reshape(
            permute(
                s.lit(field, uint(4), &[nb, 16, 2], packed(2, 8)),
                &[0, 2, 1],
            ),
            &[nb, 32],
        )
    };
    // One bit per element, in element order: bit `j` of byte `j / 8`.
    let bits32 = |field: &str| -> Expr { s.lit(field, uint(1), &[nb, 32], packed(8, 8)) };
    // The scale of a 32-element block, one f16 per block.
    let d = || s.f16("d", nb);

    Ok(match t {
        // for (j = 0; j < qk/2; ++j) {
        //     y[j]        = (x[i].qs[j] & 0x0F) - 8) * d;
        //     y[j + qk/2] = (x[i].qs[j] >>   4) - 8) * d; }
        Type::Q4_0 => deq(
            nibbles32("qs"),
            "affine-sub",
            &[1, 32],
            d(),
            Some(full(8, uint(8))),
        ),
        // …the same, with a stored minimum instead of the constant −8:
        //     y[j] = (x[i].qs[j] & 0x0F) * d + m;
        Type::Q4_1 => deq(
            nibbles32("qs"),
            "affine-add",
            &[1, 32],
            d(),
            Some(s.f16("m", nb)),
        ),
        // y[j] = x[i].qs[j] * d, int8, nothing packed.
        Type::Q8_0 | Type::Q8_1 => deq(
            s.lit("qs", DType::I8, &[nb, 32], Layout::default()),
            "sym",
            &[1, 32],
            d(),
            None,
        ),
        // The fifth bit lives in a 32-bit plane, one bit per element:
        //     xh_0 = ((qh >> j) << 4) & 0x10;
        //     y[j] = (((qs[j] & 0xF) | xh_0) - 16) * d;
        // The two halves are added *before* dequantizing, which is both what
        // the format means — one 5-bit value, not two — and what keeps the
        // arithmetic identical to llama.cpp's, which rounds once.
        Type::Q5_0 | Type::Q5_1 => {
            let q = bin(
                BinOp::Add,
                cast(nibbles32("qs"), uint(8)),
                scale(cast(bits32("qh"), uint(8)), 16),
            );
            if t == Type::Q5_0 {
                deq(q, "affine-sub", &[1, 32], d(), Some(full(16, uint(8))))
            } else {
                deq(q, "affine-add", &[1, 32], d(), Some(s.f16("m", nb)))
            }
        }
        // block_q2_K: 16 sub-blocks of 16, each with a 4-bit scale in the low
        // nibble of scales[j] and a 4-bit minimum in the high nibble.
        //     dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
        //     y = dl * q - ml;
        // The 2-bit values run shift-major: for each of two 128-element halves,
        // four shifts, each covering 32 bytes.
        Type::Q2K => {
            let sc4 = s.lit("scales", uint(4), &[nb, 16, 2], packed(2, 8));
            let sc = slot(sc4.clone(), &[nb, 16, 2], 2, 0);
            let mn = slot(sc4, &[nb, 16, 2], 2, 1);
            let q = reshape(
                permute(
                    s.lit("qs", uint(2), &[nb, 2, 32, 4], packed(4, 8)),
                    &[0, 1, 3, 2],
                ),
                &[nb, 16, 16],
            );
            let d1 = cast(bin(BinOp::Mul, reshape(d(), &[nb, 1]), sc), DType::F32);
            let m1 = scale(
                cast(
                    bin(BinOp::Mul, reshape(s.f16("dmin", nb), &[nb, 1]), mn),
                    DType::F32,
                ),
                -1,
            );
            deq(q, "affine-add", &[1, 1, 16], d1, Some(m1))
        }
        // block_q3_K: 2 bits in `qs` and an *inverted* third in `hmask`:
        //     y = d * (sc - 32) * (((q >> shift) & 3) - (hm & m ? 0 : 4));
        // and the sixteen 6-bit scales are four low nibbles plus two high bits,
        // read here as two literals over the same twelve bytes.
        Type::Q3K => {
            let low4 = reshape(
                permute(
                    s.lit("scales_l", uint(4), &[nb, 8, 2], packed(2, 8)),
                    &[0, 2, 1],
                ),
                &[nb, 16],
            );
            let high2 = reshape(
                permute(
                    s.lit("scales_h", uint(2), &[nb, 4, 4], packed(4, 8)),
                    &[0, 2, 1],
                ),
                &[nb, 16],
            );
            let sc6 = bin(
                BinOp::Add,
                cast(low4, uint(8)),
                scale(cast(high2, uint(8)), 16),
            );
            let sc = cast(
                bin(
                    BinOp::Mul,
                    reshape(d(), &[nb, 1]),
                    bin(BinOp::Sub, sc6, full(32, uint(8))),
                ),
                DType::F32,
            );
            let q2 = reshape(
                permute(
                    s.lit("qs", uint(2), &[nb, 2, 32, 4], packed(4, 8)),
                    &[0, 1, 3, 2],
                ),
                &[nb, 16, 16],
            );
            let hm = reshape(
                permute(
                    s.lit("hmask", uint(1), &[nb, 32, 8], packed(8, 8)),
                    &[0, 2, 1],
                ),
                &[nb, 16, 16],
            );
            let q = bin(BinOp::Add, cast(q2, uint(8)), scale(cast(hm, uint(8)), 4));
            deq(q, "affine-sub", &[1, 1, 16], sc, Some(full(4, uint(8))))
        }
        // block_q4_K and block_q5_K: eight sub-blocks of 32, whose 6-bit scales
        // and minima are packed twelve bytes at a time by `get_scale_min_k4`:
        //     j < 4: sc = q[j] & 63,               m = q[j+4] & 63
        //     j >= 4: sc = (q[j+4] & 0xF) | (q[j-4] >> 6) << 4,
        //             m  = (q[j+4] >> 4)  | (q[j]   >> 6) << 4
        // which is three literals over the same bytes at three bit widths.
        Type::Q4K | Type::Q5K => {
            let a = s.lit("scales_a", uint(6), &[nb, 4], packed(1, 8));
            let b = s.lit("scales_b", uint(6), &[nb, 4], packed(1, 8));
            let c = s.lit("scales_c", uint(4), &[nb, 4, 2], packed(2, 8));
            let hi = |field: &str| -> Expr {
                slot(
                    s.lit(field, uint(2), &[nb, 4, 4], packed(4, 8)),
                    &[nb, 4, 4],
                    4,
                    3,
                )
            };
            let join = |lo: Expr, high: Expr| -> Expr {
                bin(
                    BinOp::Add,
                    cast(lo, uint(8)),
                    scale(cast(high, uint(8)), 16),
                )
            };
            let sc = Expr::Concat {
                xs: vec![
                    cast(a, uint(8)),
                    join(slot(c.clone(), &[nb, 4, 2], 2, 0), hi("scales_a")),
                ],
                axis: 1,
            };
            let mn = Expr::Concat {
                xs: vec![
                    cast(b, uint(8)),
                    join(slot(c, &[nb, 4, 2], 2, 1), hi("scales_b")),
                ],
                axis: 1,
            };
            let d1 = cast(bin(BinOp::Mul, reshape(d(), &[nb, 1]), sc), DType::F32);
            let m1 = scale(
                cast(
                    bin(BinOp::Mul, reshape(s.f16("dmin", nb), &[nb, 1]), mn),
                    DType::F32,
                ),
                -1,
            );
            // The 4-bit values: four chunks of 32 bytes, low nibbles first.
            let ql = reshape(
                permute(
                    s.lit("qs", uint(4), &[nb, 4, 32, 2], packed(2, 8)),
                    &[0, 1, 3, 2],
                ),
                &[nb, 8, 32],
            );
            let q = if t == Type::Q4K {
                ql
            } else {
                // Q5_K's fifth bit: bit `2c + half` of qh[l], which is the
                // sub-block index and the position within it — so the plane
                // needs one permute and no arithmetic on indices.
                let qh = permute(s.lit("qh", uint(1), &[nb, 32, 8], packed(8, 8)), &[0, 2, 1]);
                bin(BinOp::Add, cast(ql, uint(8)), scale(cast(qh, uint(8)), 16))
            };
            deq(q, "affine-add", &[1, 1, 32], d1, Some(m1))
        }
        // block_q6_K: six bits per value — four in `ql`, two in `qh` — with a
        // signed 8-bit scale per sixteen:
        //     q = (ql & 0xF) | ((qh >> 2k) & 3) << 4;  y = d * sc * (q - 32);
        // and the four `k` slots interleave the two 32-byte halves of `ql`.
        Type::Q6K => {
            let ql = reshape(
                permute(
                    s.lit("ql", uint(4), &[nb, 2, 2, 32, 2], packed(2, 8)),
                    &[0, 1, 4, 2, 3],
                ),
                &[nb, 16, 16],
            );
            let qh = reshape(
                permute(
                    s.lit("qh", uint(2), &[nb, 2, 32, 4], packed(4, 8)),
                    &[0, 1, 3, 2],
                ),
                &[nb, 16, 16],
            );
            let q = bin(BinOp::Add, cast(ql, uint(8)), scale(cast(qh, uint(8)), 16));
            let sc = cast(
                bin(
                    BinOp::Mul,
                    reshape(d(), &[nb, 1]),
                    s.lit("scales", DType::I8, &[nb, 16], Layout::default()),
                ),
                DType::F32,
            );
            deq(q, "affine-sub", &[1, 1, 16], sc, Some(full(32, uint(8))))
        }
        other => {
            check_supported(other)?;
            return Err(Error::Unsupported(format!(
                "{} is dense; it has no dequantization expression",
                other.name()
            )));
        }
    })
}

// --------------------------------------------------------------------- import --

#[derive(Clone, Debug)]
pub struct ImportOpts {
    pub name: String,
    pub source_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    /// The OMNI architecture family, when the caller knows it. GGUF's
    /// `general.architecture` is a GGML family name and not one of §07.8's, so
    /// it is recorded as metadata and never silently promoted to `arch.family`.
    pub arch: Option<String>,
    /// The largest tensor to dequantize whole for the I4 check, in elements.
    /// What it skipped is reported rather than implied.
    pub max_verify_elems: u64,
    /// Also attach §05.2.4's *opaque* form: each quantized tensor's blocks,
    /// verbatim, as a `RuntimeCache` a runtime can map without conversion.
    ///
    /// Off by default because it doubles the stored bytes of every quantized
    /// tensor, and the structural form already preserves them. §05.2.4 says a
    /// well-formed import produces the structural form as canonical and *may*
    /// attach the opaque one; this is the may.
    pub opaque_cache: bool,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/gguf".into(),
            source_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
            max_verify_elems: 1 << 22,
            opaque_cache: false,
        }
    }
}

/// One imported tensor, as the exporter will need it back.
#[derive(Clone, Debug)]
pub struct Imported1 {
    pub name: String,
    pub ty: Type,
    pub dims: Vec<u64>,
    /// The source file's offset for this tensor, from the start of the data
    /// section. Kept so that an export reproduces the file's own padding rather
    /// than a tidier packing of its tensors.
    pub offset: u64,
    pub fields: Vec<(&'static str, crate::expr::Ref)>,
    /// Elements compared against [`reference_dequant`]; zero for dense tensors
    /// and for the ones too large to materialize.
    pub checked: u64,
}

pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
    pub tensors: Vec<Imported1>,
    /// How many tensors of each type, in the order the types first appear.
    pub histogram: Vec<(Type, usize)>,
}

impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Imported {{ {} objects, root {}, {} tensor(s), lossless {} }}",
            self.objects.len(),
            crate::sha256::hex(&self.root[..6]),
            self.tensors.len(),
            self.report.lossless
        )
    }
}

/// Imports a GGUF file into an OMNI object graph.
pub fn import(bytes: &[u8], opts: &ImportOpts) -> Res<Imported> {
    let f = File::parse(bytes)?;
    let hash = opts.hash;
    let mut report = Fidelity {
        format: "gguf",
        importer: IMPORTER,
        source_path: opts.source_path.clone(),
        source_digest: hash.digest(bytes),
        source_size: bytes.len() as u64,
        lossless: true,
        represented: vec![
            "tensors".into(),
            "dtypes".into(),
            "shapes".into(),
            "metadata".into(),
        ],
        verify_method: "block-reassembly + independent-dequant",
        ..Default::default()
    };

    // Every type in the file is checked before anything is written, so a file
    // with one IQ2_XS tensor is refused by name instead of half-imported.
    for e in &f.tensors {
        check_supported(e.ty).map_err(|err| match err {
            Error::Unsupported(m) => Error::Unsupported(format!("`{}`: {m}", e.name)),
            other => other,
        })?;
    }

    let mut b = ModelBuilder::new(opts.name.clone())
        .hash(hash)
        .chunk_size(opts.chunk_size);
    if let Some(spdx) = &opts.license {
        b = b.license(spdx.clone());
    }
    if let Some(family) = &opts.arch {
        b = b.arch(family.clone(), arch_params(&f));
    }

    let mut imported: Vec<Imported1> = Vec::with_capacity(f.tensors.len());
    let mut histogram: Vec<(Type, usize)> = Vec::new();
    for e in &f.tensors {
        match histogram.iter_mut().find(|(t, _)| *t == e.ty) {
            Some((_, n)) => *n += 1,
            None => histogram.push((e.ty, 1)),
        }
        let raw = f.tensor(e)?;
        let shape = e.shape();
        if let Some(dtype) = e.ty.dense_dtype() {
            b = b.tensor(TensorSpec {
                name: e.name.clone(),
                shape: shape.clone(),
                dtype,
                axes: None,
                semantic: "",
                data: raw.to_vec(),
                layout: None,
            });
            imported.push(Imported1 {
                name: e.name.clone(),
                ty: e.ty,
                dims: e.dims.clone(),
                offset: e.offset,
                fields: Vec::new(),
                checked: 0,
            });
            continue;
        }

        let (be, bb) = e.ty.block().expect("checked above");
        let nb = e.numel() / be;
        let mut refs: Vec<(&'static str, crate::expr::Ref)> = Vec::new();
        for fld in fields(e.ty) {
            let part = gather(raw, bb as usize, fld, nb as usize);
            refs.push((fld.name, b.chunk_list(&part)));
        }
        let stored = Stored { refs: refs.clone() };
        let value = reshape(value_expr(e.ty, &stored, nb)?, &shape);
        b = b.derived(
            e.name.clone(),
            TensorDesc {
                shape: dims(&shape),
                dtype: DType::F32,
                layout: Layout::default(),
                value,
                semantic: None,
                role: Some("quantized".into()),
                axes: None,
                device_hint: None,
                materialize: Materialize::Lazy,
                stats: None,
                digest_materialized: None,
            },
        );
        imported.push(Imported1 {
            name: e.name.clone(),
            ty: e.ty,
            dims: e.dims.clone(),
            offset: e.offset,
            fields: refs,
            checked: 0,
        });
    }

    // §05.2.4's second representation, when it was asked for.
    let mut caches: Vec<Object> = Vec::new();
    if opts.opaque_cache {
        for (t, entry) in imported.iter().zip(&f.tensors) {
            if !t.ty.is_quantized() {
                continue;
            }
            let raw = f.tensor(entry)?;
            let (be, bb) = t.ty.block().expect("quantized");
            let payload = b.chunk_list(raw);
            // §05.2.4: the cache is keyed by the structural expression's
            // digest, so a reader can tell whether it caches *this* tensor
            // rather than an older version of it (§10.6 rule 2).
            let structural = b
                .derived
                .iter()
                .find(|(n, _)| n == &t.name)
                .map(|(_, d)| hash.digest(&d.value.to_value().encode()))
                .ok_or_else(|| Error::Core(format!("`{}` has no structural form", t.name)))?;
            caches.push(opaque_cache(
                t,
                be,
                bb,
                payload,
                structural,
                raw.len() as u64,
            ));
        }
        report
            .represented
            .push(format!("opaque block cache ({} tensor(s))", caches.len()));
    }

    // I2: the header, in full, in a form an export can write back. Not a
    // summary — a key this build does not model is still a key the file had.
    if !caches.is_empty() {
        b = b.manifest_key(
            "caches",
            Value::Array(
                caches
                    .iter()
                    .map(|o| {
                        Value::Array(vec![
                            Value::U(otype::RUNTIME_CACHE as u64),
                            Value::Bytes(o.digest(hash).to_vec()),
                        ])
                    })
                    .collect(),
            ),
        );
    }
    let foreign = foreign_object(&f, &imported);
    b = b.manifest_key(
        "foreign",
        Value::Array(vec![Value::Array(vec![
            Value::U(otype::FOREIGN as u64),
            Value::Bytes(foreign.digest(hash).to_vec()),
        ])]),
    );

    // §06.9: the chat template is the one part of GGUF's tokenizer block that
    // is self-contained, so it is the one part that becomes an OMNI object.
    let mut notes: Vec<Note> = Vec::new();
    if let Some(raw) = f.get("tokenizer.chat_template").and_then(|m| m.as_str()) {
        match crate::jinja::translate(raw) {
            Ok(t) => {
                report.represented.push("chat_template (OMNI-CT)".into());
                let asset = crate::hf::chat_template_asset(&mut b, t, &mut notes);
                b = b.asset("chat_template", otype::CHAT_TEMPLATE, asset);
            }
            Err(crate::jinja::Error::Unsupported(r)) => notes.push(Note {
                item: "tokenizer.chat_template".into(),
                reason: format!("Jinja2 `{}` — {}", r.construct, r.reason),
                action: "left in the foreign metadata, untranslated; §06.9 will \
                         not ship a template it cannot express totally"
                    .into(),
            }),
            Err(e) => notes.push(Note {
                item: "tokenizer.chat_template".into(),
                reason: format!("the Jinja2 source did not parse: {e}"),
                action: "left in the foreign metadata, untranslated".into(),
            }),
        }
    }
    if f.get("tokenizer.ggml.tokens").is_some() {
        // The finding, rather than a shrug: GGUF carries a vocabulary but not
        // the pre-tokenizer that decides what a token boundary *is*. §06.7
        // needs both, so building one from these keys would produce a tokenizer
        // that decodes correctly and encodes differently from the model.
        notes.push(Note {
            item: "tokenizer".into(),
            reason: "`tokenizer.ggml.pre` names a pre-tokenizer (its regexes live \
                     in llama.cpp's source, not in the file), so the vocabulary \
                     here does not determine the token boundaries"
                .into(),
            action: "the vocabulary, merges, scores and token types are preserved \
                     in the foreign metadata; no §06.7 tokenizer is synthesized, \
                     because one built from these keys would encode differently \
                     from the model it came with"
                .into(),
        });
    }
    report.unrepresented.extend(notes);

    report.assumptions.push(Note {
        item: "arch.family".into(),
        reason: format!(
            "`general.architecture` is {}, which names a GGML family rather than \
             one of §07.8's",
            match f.arch() {
                Some(a) => format!("`{a}`"),
                None => "absent".into(),
            }
        ),
        action: match &opts.arch {
            Some(family) => format!("supplied by the caller as `{family}`"),
            None => "field omitted; the GGUF key is kept as metadata".into(),
        },
    });
    report.assumptions.push(Note {
        item: "license".into(),
        reason: "GGUF's `general.license` is a free-text string, not an SPDX id".into(),
        action: match &opts.license {
            Some(spdx) => format!("supplied by the caller as `{spdx}`"),
            None => "field omitted".into(),
        },
    });
    for (t, n) in &histogram {
        report
            .represented
            .push(format!("{} ({n} tensor(s))", t.name()));
    }

    // ---- verification, before the report is written ------------------------
    let (probe, _) = b.build();
    let mut mem = crate::store::MemoryStore::new(hash);
    for o in &probe {
        let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
    }
    let ctx = Ctx::new(&mem);
    let tensors_ref = probe
        .iter()
        .find(|o| o.otype == otype::TENSOR_TABLE)
        .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
        .ok_or_else(|| Error::Core("the builder produced no tensor table".into()))?;
    let table = TensorTable::load(&ctx, &tensors_ref).map_err(|e| Error::Core(e.to_string()))?;

    let mut skipped: Vec<String> = Vec::new();
    for (t, e) in imported.iter_mut().zip(&f.tensors) {
        let raw = f.tensor(e)?;
        let r = table
            .get(&t.name)
            .ok_or_else(|| Error::Core(format!("`{}` did not reach the table", t.name)))?;
        let d = TensorDesc::load(&ctx, r).map_err(|err| Error::Core(err.to_string()))?;

        // 1. The bytes. For a dense tensor that is the literal; for a blocked
        //    one it is the export path, run here.
        let got = match t.fields.is_empty() {
            true => {
                let Expr::Literal { chunks, .. } = &d.value else {
                    return Err(Error::Core(format!("`{}` is not a literal", t.name)));
                };
                ctx.chunk_bytes(chunks)
                    .map_err(|err| Error::Core(err.to_string()))?
            }
            false => {
                let mut parts = Vec::with_capacity(t.fields.len());
                for (name, r) in &t.fields {
                    parts.push((
                        *name,
                        ctx.chunk_bytes(r)
                            .map_err(|err| Error::Core(err.to_string()))?,
                    ));
                }
                let (be, _) = t.ty.block().expect("quantized");
                reassemble(t.ty, &parts, (e.numel() / be) as usize)?
            }
        };
        if got != raw {
            return Err(Error::Core(format!(
                "I4: `{}` does not reassemble to the bytes it was read from; the \
                 round trip this import claims would not be exact",
                t.name
            )));
        }
        report.verified_tensors += 1;
        report.verified_bytes += raw.len() as u64;

        // 2. The values, against scalar code that shares nothing with the
        //    evaluator.
        if t.ty.is_quantized() {
            if e.numel() > opts.max_verify_elems {
                skipped.push(t.name.clone());
                continue;
            }
            let want =
                reference_dequant(t.ty, raw, (e.numel() / t.ty.block().unwrap().0) as usize)?;
            let evaluated = d
                .value
                .eval(&ctx)
                .map_err(|err| Error::Core(format!("{}: {err}", t.name)))?;
            if evaluated.data.len() != want.len() {
                return Err(Error::Core(format!(
                    "I4: `{}` evaluates to {} elements and dequantizes to {}",
                    t.name,
                    evaluated.data.len(),
                    want.len()
                )));
            }
            for (i, (a, w)) in evaluated.data.iter().zip(&want).enumerate() {
                let w = *w as f64;
                if *a != w && !(a.is_nan() && w.is_nan()) {
                    return Err(Error::Core(format!(
                        "I4: `{}`[{i}] is {a} through the expression graph and {w} \
                         from the block layout; the permutation, the bit offset or \
                         the formula is wrong",
                        t.name
                    )));
                }
            }
            t.checked = e.numel();
            report.dequant_checked += e.numel();
        }
    }
    for name in &skipped {
        report.warnings.push(format!(
            "{name}: too large to dequantize whole (> {} elements); its bytes are \
             still reassembled and compared",
            opts.max_verify_elems
        ));
    }

    let (mut objects, root) = b
        .asset("provenance", otype::PROVENANCE, report.to_value())
        .build();
    objects.push(foreign);
    objects.extend(caches);
    Ok(Imported {
        objects,
        root,
        report,
        tensors: imported,
        histogram,
    })
}

/// The `<arch>.*` keys that mean the same thing as an OMNI `arch.params` entry.
///
/// Only the ones whose meaning is identical are mapped; `rope.dimension_count`,
/// for instance, is GGML's partial-rotary count and is *not* the head dimension,
/// so it is left in the foreign metadata rather than renamed into something that
/// looks familiar.
pub fn arch_params(f: &File<'_>) -> Vec<(&'static str, Value)> {
    let Some(arch) = f.arch() else {
        return Vec::new();
    };
    let get = |suffix: &str| f.get(&format!("{arch}.{suffix}")).and_then(|m| m.as_u64());
    let getf = |suffix: &str| f.get(&format!("{arch}.{suffix}")).and_then(|m| m.as_f64());
    let mut out: Vec<(&'static str, Value)> = Vec::new();
    for (gguf, omni) in [
        ("embedding_length", "hidden_size"),
        ("block_count", "n_layers"),
        ("attention.head_count", "n_heads"),
        ("attention.head_count_kv", "n_kv_heads"),
        ("feed_forward_length", "intermediate_size"),
        ("context_length", "context_length"),
        ("vocab_size", "vocab_size"),
    ] {
        if let Some(v) = get(gguf) {
            out.push((omni, Value::U(v)));
        }
    }
    let eps = getf("attention.layer_norm_rms_epsilon");
    if let Some(eps) = eps {
        out.push((
            "norm",
            Value::map(vec![("kind", Value::text("rms")), ("eps", Value::F64(eps))]),
        ));
    } else if let Some(eps) = getf("attention.layer_norm_epsilon") {
        out.push((
            "norm",
            Value::map(vec![
                ("kind", Value::text("layer")),
                ("eps", Value::F64(eps)),
            ]),
        ));
    }
    if let Some(theta) = getf("rope.freq_base") {
        out.push(("rope", Value::map(vec![("theta", Value::F64(theta))])));
    }
    out
}

/// One tensor's blocks, verbatim, as the §10.6 `RuntimeCache` §05.2.4 calls the
/// *opaque* representation.
///
/// The point of it is that `llama.cpp` can map these bytes and run with no
/// conversion at all, while the canonical form of the same tensor stays an
/// expression a reader can see through. Two things make that safe rather than a
/// second source of truth: the cache is keyed by the structural expression's
/// digest (§10.6 rule 2), so a stale one is detectable, and it is flagged
/// `CACHEABLE`, so deleting every cache leaves the same model (§10.6 rule 1).
///
/// The dtype is `opaque`, which §04.3.5 restricts to `literal`, `slice` and
/// cast-to-opaque: this build will not do arithmetic on bytes whose element
/// layout it has declined to describe. The layout says how long a block is and
/// what its fields are, which is the *sizing* information §04.3.5 does allow.
fn opaque_cache(
    t: &Imported1,
    block_elems: u64,
    block_bytes: u64,
    payload: crate::expr::Ref,
    structural: Digest,
    size: u64,
) -> Object {
    let id = format!("org.ggml/{}", t.ty.name());
    let dtype = DType::Opaque {
        id: id.clone(),
        block_elems,
        block_bytes,
    };
    let groups: Vec<Vec<crate::layout::Field>> = vec![fields(t.ty)
        .iter()
        .map(|f| crate::layout::Field {
            name: f.name.to_string(),
            dtype: None,
            count: Some(f.len as u64),
        })
        .collect()];
    let layout = Layout::Interleaved {
        groups,
        stride_bytes: block_bytes,
    };
    let numel: u64 = t.dims.iter().product();
    let value = Expr::Literal {
        chunks: payload,
        dtype,
        shape: dims(&[numel / block_elems]),
        layout,
    };
    let mut o = Object::structure(
        otype::RUNTIME_CACHE,
        &Value::map(vec![
            ("t", Value::text("omni.rt/cache")),
            ("v", Value::U(1)),
            ("kind", Value::text("materialized-tensor")),
            ("tensor", Value::text(t.name.clone())),
            ("key", Value::Bytes(structural.to_vec())),
            (
                "target",
                Value::map(vec![(
                    "runtime",
                    Value::map(vec![("name", Value::text("ggml"))]),
                )]),
            ),
            ("payload", value.to_value()),
            ("size", Value::U(size)),
            ("executable", Value::Bool(false)),
            ("reproducible", Value::Bool(true)),
        ]),
    );
    // §10.6 rule 1: every cache is droppable, and the flag is how a reader
    // knows it without understanding what is cached.
    o.oflags |= crate::container::oflags::CACHEABLE;
    o
}

/// The `Foreign` object (§01.9): the whole GGUF header, and where every
/// tensor's bytes went, so that an export writes the file back rather than a
/// file like it.
fn foreign_object(f: &File<'_>, tensors: &[Imported1]) -> Object {
    let kv = Value::Array(
        f.kv.iter()
            .map(|(k, v)| Value::Array(vec![Value::text(k.clone()), v.to_value()]))
            .collect(),
    );
    let ts = Value::Array(
        tensors
            .iter()
            .map(|t| {
                Value::map(vec![
                    ("name", Value::text(t.name.clone())),
                    ("type", Value::U(t.ty.to_u32() as u64)),
                    (
                        "dims",
                        Value::Array(t.dims.iter().map(|d| Value::U(*d)).collect()),
                    ),
                    ("offset", Value::U(t.offset)),
                    (
                        "fields",
                        Value::Array(
                            t.fields
                                .iter()
                                .map(|(n, r)| {
                                    Value::Array(vec![
                                        Value::text((*n).to_string()),
                                        Value::Array(vec![
                                            Value::U(r.0 as u64),
                                            Value::Bytes(r.1.to_vec()),
                                        ]),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect(),
    );
    Object::structure(
        otype::FOREIGN,
        &Value::map(vec![
            ("t", Value::text("omni.core/foreign")),
            ("v", Value::U(1)),
            ("format", Value::text("gguf")),
            ("version", Value::U(f.version as u64)),
            ("alignment", Value::U(f.alignment)),
            ("padding_zero", Value::Bool(f.padding_is_zero())),
            ("kv", kv),
            ("tensors", ts),
        ]),
    )
}

// -------------------------------------------------- the independent reference --

fn half(b: &[u8], at: usize) -> f32 {
    crate::dtype::DType::F16
        .decode(&b[at..at + 2], 0)
        .unwrap_or(f64::NAN) as f32
}

/// Dequantizes a run of blocks the way `ggml-quants.c` does: scalar code,
/// written from the block layouts, sharing nothing with the expression
/// evaluator.
///
/// This is the oracle for the I4 check in [`import`]. It is deliberately a
/// transcription — loop structure, shift order and arithmetic order all follow
/// `dequantize_row_*`, because the point is to disagree with the evaluator
/// whenever the evaluator is wrong, and code factored for elegance would drift
/// towards agreeing with it for the wrong reason.
pub fn reference_dequant(t: Type, blocks: &[u8], n: usize) -> Res<Vec<f32>> {
    let (be, bb) = t
        .block()
        .ok_or_else(|| Error::Unsupported(format!("{} has no block layout", t.name())))?;
    let (be, bb) = (be as usize, bb as usize);
    if blocks.len() < n * bb {
        return Err(Error::Malformed(format!(
            "{n} blocks of {} need {} bytes and {} were given",
            t.name(),
            n * bb,
            blocks.len()
        )));
    }
    let mut y = Vec::with_capacity(n * be);
    for i in 0..n {
        let x = &blocks[i * bb..(i + 1) * bb];
        match t {
            Type::Q4_0 => {
                let d = half(x, 0);
                for j in 0..16 {
                    y.push(((x[2 + j] & 0x0F) as i32 - 8) as f32 * d);
                }
                for j in 0..16 {
                    y.push(((x[2 + j] >> 4) as i32 - 8) as f32 * d);
                }
            }
            Type::Q4_1 => {
                let (d, m) = (half(x, 0), half(x, 2));
                for j in 0..16 {
                    y.push((x[4 + j] & 0x0F) as f32 * d + m);
                }
                for j in 0..16 {
                    y.push((x[4 + j] >> 4) as f32 * d + m);
                }
            }
            Type::Q5_0 => {
                let d = half(x, 0);
                let qh = u32::from_le_bytes(x[2..6].try_into().unwrap());
                for j in 0..16 {
                    let h = ((qh >> j) & 1) as i32;
                    y.push((((x[6 + j] & 0x0F) as i32 | (h << 4)) - 16) as f32 * d);
                }
                for j in 0..16 {
                    let h = ((qh >> (j + 16)) & 1) as i32;
                    y.push((((x[6 + j] >> 4) as i32 | (h << 4)) - 16) as f32 * d);
                }
            }
            Type::Q5_1 => {
                let (d, m) = (half(x, 0), half(x, 2));
                let qh = u32::from_le_bytes(x[4..8].try_into().unwrap());
                for j in 0..16 {
                    let h = ((qh >> j) & 1) as i32;
                    y.push(((x[8 + j] & 0x0F) as i32 | (h << 4)) as f32 * d + m);
                }
                for j in 0..16 {
                    let h = ((qh >> (j + 16)) & 1) as i32;
                    y.push(((x[8 + j] >> 4) as i32 | (h << 4)) as f32 * d + m);
                }
            }
            Type::Q8_0 | Type::Q8_1 => {
                let d = half(x, 0);
                let off = if t == Type::Q8_0 { 2 } else { 4 };
                for j in 0..32 {
                    y.push(x[off + j] as i8 as f32 * d);
                }
            }
            Type::Q2K => {
                let (d, dmin) = (half(x, 80), half(x, 82));
                let (scales, q) = (&x[0..16], &x[16..80]);
                let mut is = 0usize;
                for nn in 0..2 {
                    let q = &q[nn * 32..];
                    for j in 0..4 {
                        let shift = 2 * j;
                        for sub in 0..2 {
                            let sc = scales[is];
                            is += 1;
                            let dl = d * (sc & 0xF) as f32;
                            let ml = dmin * (sc >> 4) as f32;
                            for l in 0..16 {
                                let v = (q[sub * 16 + l] >> shift) & 3;
                                y.push(dl * v as f32 - ml);
                            }
                        }
                    }
                }
            }
            Type::Q3K => {
                let d_all = half(x, 108);
                let (hmask, q, sraw) = (&x[0..32], &x[32..96], &x[96..108]);
                // The four masked shifts of `dequantize_row_q3_K`, written out.
                let mut sc = [0u8; 16];
                for k in 0..4 {
                    sc[k] = (sraw[k] & 0x0F) | ((sraw[8 + k] & 3) << 4);
                    sc[4 + k] = (sraw[4 + k] & 0x0F) | (((sraw[8 + k] >> 2) & 3) << 4);
                    sc[8 + k] = ((sraw[k] >> 4) & 0x0F) | (((sraw[8 + k] >> 4) & 3) << 4);
                    sc[12 + k] = ((sraw[4 + k] >> 4) & 0x0F) | (((sraw[8 + k] >> 6) & 3) << 4);
                }
                let mut is = 0usize;
                let mut m = 0u32;
                for nn in 0..2 {
                    let q = &q[nn * 32..];
                    for j in 0..4 {
                        let shift = 2 * j;
                        for sub in 0..2 {
                            let dl = d_all * (sc[is] as i32 - 32) as f32;
                            is += 1;
                            for l in 0..16 {
                                let idx = sub * 16 + l;
                                let bit = (hmask[idx] >> m) & 1;
                                let v =
                                    ((q[idx] >> shift) & 3) as i32 - if bit == 1 { 0 } else { 4 };
                                y.push(dl * v as f32);
                            }
                        }
                        m += 1;
                    }
                }
            }
            Type::Q4K | Type::Q5K => {
                let (d, dmin) = (half(x, 0), half(x, 2));
                let sraw = &x[4..16];
                let (qh, ql) = match t {
                    Type::Q4K => (None, &x[16..144]),
                    _ => (Some(&x[16..48]), &x[48..176]),
                };
                // get_scale_min_k4, both branches.
                let scale_min = |j: usize| -> (u8, u8) {
                    if j < 4 {
                        (sraw[j] & 63, sraw[j + 4] & 63)
                    } else {
                        (
                            (sraw[j + 4] & 0x0F) | ((sraw[j - 4] >> 6) << 4),
                            (sraw[j + 4] >> 4) | ((sraw[j] >> 6) << 4),
                        )
                    }
                };
                for c in 0..4 {
                    for half_ in 0..2 {
                        let is = c * 2 + half_;
                        let (sc, mn) = scale_min(is);
                        let d1 = d * sc as f32;
                        let m1 = dmin * mn as f32;
                        for l in 0..32 {
                            let lo = if half_ == 0 {
                                ql[c * 32 + l] & 0x0F
                            } else {
                                ql[c * 32 + l] >> 4
                            } as i32;
                            let hi = match qh {
                                Some(qh) => ((qh[l] >> is) & 1) as i32,
                                None => 0,
                            };
                            y.push(d1 * (lo + 16 * hi) as f32 - m1);
                        }
                    }
                }
            }
            Type::Q6K => {
                // The four values `dequantize_row_q6_K` computes for one `l`
                // are 32 elements apart in the output, so this block is written
                // into place rather than pushed.
                let d = half(x, 208);
                let (ql, qh, sc) = (&x[0..128], &x[128..192], &x[192..208]);
                let base = y.len();
                y.resize(base + 256, 0.0);
                for nn in 0..2 {
                    let (ql, qh, sc) = (&ql[nn * 64..], &qh[nn * 32..], &sc[nn * 8..]);
                    for l in 0..32 {
                        let is = l / 16;
                        for k in 0..4usize {
                            let byte = ql[(k % 2) * 32 + l];
                            let low = if k < 2 { byte & 0x0F } else { byte >> 4 } as i32;
                            let high = ((qh[l] >> (2 * k)) & 3) as i32;
                            let q = (low | (high << 4)) - 32;
                            y[base + nn * 128 + k * 32 + l] =
                                d * sc[k * 2 + is] as i8 as f32 * q as f32;
                        }
                    }
                }
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "{} has no reference dequantization here",
                    other.name()
                )))
            }
        }
    }
    Ok(y)
}

impl File<'_> {
    /// Whether every byte the tensors do not cover is zero.
    ///
    /// GGUF leaves padding unspecified, so an export can only reproduce a file
    /// byte for byte if the padding it is reproducing is the padding it writes.
    /// This is checked at import so that the answer is recorded while the source
    /// is still in hand, rather than assumed at export time.
    pub fn padding_is_zero(&self) -> bool {
        let mut covered: Vec<(u64, u64)> = Vec::with_capacity(self.tensors.len());
        for e in &self.tensors {
            let Some(n) = e.ty.stored_bytes(e.numel()) else {
                return false;
            };
            covered.push((self.data_start + e.offset, n));
        }
        covered.sort_unstable();
        let mut at = self.header_end;
        for (start, len) in covered {
            if start > at
                && self.bytes[at as usize..start as usize]
                    .iter()
                    .any(|b| *b != 0)
            {
                return false;
            }
            at = at.max(start + len);
        }
        !self.bytes[at as usize..].iter().any(|b| *b != 0)
    }
}

// --------------------------------------------------------------------- export --

/// One thing an export would lose (E1, `import-export.md` §5.1).
#[derive(Clone, Debug)]
pub struct Loss {
    pub item: String,
    pub reason: String,
}

/// What an export would produce and what it would cost, computed without
/// writing anything.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// `(name, ggml type, dims, stored bytes)`, in the order they will be
    /// written.
    pub tensors: Vec<(String, String, Vec<u64>, u64)>,
    pub loss: Vec<Loss>,
    pub bytes: u64,
    /// Whether this is a re-export of a file this build imported, in which case
    /// the header is the source's own rather than one composed here.
    pub from_gguf: bool,
    /// Tensors whose §05.2.4 opaque cache was found and checked against the
    /// bytes the structural form reassembles to. Zero when the container
    /// carries no caches, which is the default.
    pub opaque_checked: usize,
}

impl Plan {
    pub fn lossless(&self) -> bool {
        self.loss.is_empty()
    }

    pub fn loss_report(&self, source: &Digest, hash: HashAlgo) -> String {
        json::object(vec![
            ("target", json::string("gguf")),
            (
                "source",
                json::object(vec![
                    ("format", json::string("omni")),
                    (
                        "digest",
                        json::string(format!("{}:{}", hash.prefix(), crate::sha256::hex(source))),
                    ),
                    ("round_trip", json::Value::Bool(self.from_gguf)),
                ]),
            ),
            ("lossless", json::Value::Bool(self.lossless())),
            ("tensors", json::Value::U(self.tensors.len() as u64)),
            (
                "opaque_caches_verified",
                json::Value::U(self.opaque_checked as u64),
            ),
            ("bytes", json::Value::U(self.bytes)),
            (
                "lost",
                json::Value::Array(
                    self.loss
                        .iter()
                        .map(|l| {
                            json::object(vec![
                                ("item", json::string(l.item.clone())),
                                ("reason", json::string(l.reason.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
        .encode()
    }
}

/// One tensor as it will be written: name, type, GGUF-order extents, offset in
/// the data section, and the bytes themselves.
type Written = (String, Type, Vec<u64>, u64, Vec<u8>);

/// The opaque block caches (§05.2.4, §10.6) a container carries, by tensor name.
///
/// A cache is an *assertion* that these bytes are what the structural form
/// computes, and §10.6 rule 2 says a consumer must check it rather than trust
/// it. So the export path reads them and compares; a stale one is a defect,
/// not a fallback.
fn opaque_caches(ctx: &Ctx<'_>, manifest: &Value) -> Vec<(String, crate::expr::Ref)> {
    let mut out = Vec::new();
    for item in manifest
        .get("caches")
        .and_then(|v| v.as_array())
        .unwrap_or(&[])
    {
        let Ok(r) = crate::expr::parse_ref_value(item) else {
            continue;
        };
        let Ok(v) = ctx.value(&r.1) else { continue };
        let (Some(name), Some(payload)) = (
            v.get("tensor").and_then(|x| x.as_str()),
            v.get("payload").and_then(|x| Expr::from_value(x).ok()),
        ) else {
            continue;
        };
        if let Expr::Literal { chunks, .. } = payload {
            out.push((name.to_string(), chunks));
        }
    }
    out
}

/// The GGUF header a container carries, when it came from one.
struct Preserved {
    version: u32,
    alignment: u64,
    padding_zero: bool,
    kv: Vec<(String, Meta)>,
    tensors: Vec<PreservedTensor>,
}

struct PreservedTensor {
    name: String,
    ty: Type,
    dims: Vec<u64>,
    offset: u64,
    fields: Vec<(String, crate::expr::Ref)>,
}

/// Reads the `Foreign` object an import left behind. `None` when this container
/// did not come from a GGUF file — which is not an error, only a different
/// export.
fn preserved(ctx: &Ctx<'_>, manifest: &Value) -> Res<Option<Preserved>> {
    let Some(list) = manifest.get("foreign").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    for item in list {
        let Ok(r) = crate::expr::parse_ref_value(item) else {
            continue;
        };
        let Ok(v) = ctx.value(&r.1) else { continue };
        if v.get("format").and_then(|x| x.as_str()) != Some("gguf") {
            continue;
        }
        let version = v.get("version").and_then(|x| x.as_u64()).unwrap_or(3) as u32;
        let alignment = v
            .get("alignment")
            .and_then(|x| x.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        let padding_zero = matches!(v.get("padding_zero"), Some(Value::Bool(true)));
        let mut kv = Vec::new();
        for pair in v.get("kv").and_then(|x| x.as_array()).unwrap_or(&[]) {
            let a = pair
                .as_array()
                .ok_or_else(|| Error::Malformed("a metadata pair is not [key, value]".into()))?;
            let key = a
                .first()
                .and_then(|x| x.as_str())
                .ok_or_else(|| Error::Malformed("a metadata key is not a string".into()))?;
            let val = Meta::from_value(
                a.get(1)
                    .ok_or_else(|| Error::Malformed("a metadata pair has no value".into()))?,
            )?;
            kv.push((key.to_string(), val));
        }
        let mut tensors = Vec::new();
        for t in v.get("tensors").and_then(|x| x.as_array()).unwrap_or(&[]) {
            let name = t
                .get("name")
                .and_then(|x| x.as_str())
                .ok_or_else(|| Error::Malformed("a preserved tensor has no name".into()))?
                .to_string();
            let ty = Type::from_u32(t.get("type").and_then(|x| x.as_u64()).unwrap_or(0) as u32);
            let dims = t
                .get("dims")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|d| d.as_u64()).collect::<Vec<u64>>())
                .unwrap_or_default();
            let offset = t.get("offset").and_then(|x| x.as_u64()).unwrap_or(0);
            let mut fields = Vec::new();
            for fld in t.get("fields").and_then(|x| x.as_array()).unwrap_or(&[]) {
                let a = fld
                    .as_array()
                    .ok_or_else(|| Error::Malformed("a field is not [name, ref]".into()))?;
                let n = a
                    .first()
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error::Malformed("a field has no name".into()))?
                    .to_string();
                let r = crate::expr::parse_ref_value(
                    a.get(1)
                        .ok_or_else(|| Error::Malformed("a field has no ref".into()))?,
                )
                .map_err(|e| Error::Malformed(e.to_string()))?;
                fields.push((n, r));
            }
            tensors.push(PreservedTensor {
                name,
                ty,
                dims,
                offset,
                fields,
            });
        }
        return Ok(Some(Preserved {
            version,
            alignment,
            padding_zero,
            kv,
            tensors,
        }));
    }
    Ok(None)
}

/// E1: what an export would lose, without producing bytes.
pub fn plan(ctx: &Ctx<'_>, manifest: &Value, table: &TensorTable) -> Res<Plan> {
    let mut p = Plan::default();
    if let Some(pres) = preserved(ctx, manifest)? {
        p.from_gguf = true;
        if !pres.padding_zero {
            p.loss.push(Loss {
                item: "padding".into(),
                reason: "the source file's padding was not all zero, and GGUF does \
                         not define what padding contains, so this export writes \
                         zeros and the file will differ from the original there"
                    .into(),
            });
        }
        let mut end = 0u64;
        for t in &pres.tensors {
            let numel: u64 = t.dims.iter().product();
            let bytes =
                t.ty.stored_bytes(numel)
                    .ok_or_else(|| Error::Unsupported(format!("{} has no size", t.ty.name())))?;
            p.tensors
                .push((t.name.clone(), t.ty.name(), t.dims.clone(), bytes));
            end = end.max(t.offset + bytes);
        }
        p.bytes = end;
        return Ok(p);
    }

    // No GGUF header to reproduce: a file is composed from the tensor table,
    // and everything GGUF has no field for is named.
    let mut names: Vec<String> = table
        .order
        .iter()
        .filter(|n| table.tensors.contains_key(*n))
        .cloned()
        .collect();
    for n in table.tensors.keys() {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    let mut at = 0u64;
    for name in names {
        let r = &table.tensors[&name];
        let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
        let Some(shape) = crate::expr::concrete(&desc.shape) else {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: "a symbolic shape has no fixed extent to write".into(),
            });
            continue;
        };
        if shape.len() > 4 {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: format!("{} dimensions; a GGML tensor has at most four", shape.len()),
            });
            continue;
        }
        let numel: u64 = shape.iter().product();
        let known = [
            (DType::F32, Type::F32),
            (DType::F16, Type::F16),
            (DType::BF16, Type::BF16),
            (DType::F64, Type::F64),
            (DType::I8, Type::I8),
            (DType::I16, Type::I16),
            (DType::I32, Type::I32),
            (DType::I64, Type::I64),
        ];
        let (ty, note) = match known.iter().find(|(d, _)| *d == desc.dtype) {
            Some((_, t)) => (*t, None),
            // The values can still be written — as F32 — but that is a change
            // of type, so it is loss and is reported rather than done quietly.
            None => (
                Type::F32,
                Some(format!(
                    "{} has no GGML type; its values are written as F32",
                    desc.dtype.label()
                )),
            ),
        };
        if let Some(reason) = note {
            p.loss.push(Loss {
                item: format!("tensor `{name}` dtype"),
                reason,
            });
        }
        let bytes = ty.stored_bytes(numel).unwrap_or(0);
        at = at.div_ceil(DEFAULT_ALIGNMENT) * DEFAULT_ALIGNMENT + bytes;
        p.tensors.push((
            name.clone(),
            ty.name(),
            shape.iter().rev().copied().collect(),
            bytes,
        ));
    }
    p.bytes = at;
    p.loss.push(Loss {
        item: "metadata".into(),
        reason: "this container did not come from GGUF, so there is no \
                 `general.architecture` or hyper-parameter block to write, and \
                 llama.cpp will not load a file without one"
            .into(),
    });
    Ok(p)
}

// ------------------------------------------------------------- writing a file --

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_meta(out: &mut Vec<u8>, m: &Meta) {
    match m {
        Meta::U8(v) => out.push(*v),
        Meta::I8(v) => out.push(*v as u8),
        Meta::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Meta::Bool(v) => out.push(u8::from(*v)),
        Meta::Str(s) => put_str(out, s),
        Meta::Arr(elem, xs) => {
            out.extend_from_slice(&elem.to_le_bytes());
            out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                put_meta(out, x);
            }
        }
    }
}

/// Writes a GGUF file from a container.
///
/// For a container this build imported from GGUF, this is the exact inverse of
/// [`import`]: the header is the source's, key for key and in order, and each
/// tensor's bytes are its stored fields re-interleaved. For any other container
/// it composes a file from the tensor table, and [`plan`] has already said what
/// that cannot carry.
pub fn export(ctx: &Ctx<'_>, manifest: &Value, table: &TensorTable) -> Res<(Vec<u8>, Plan)> {
    let mut p = plan(ctx, manifest, table)?;
    let pres = preserved(ctx, manifest)?;
    let caches = opaque_caches(ctx, manifest);

    // (name, ggml type, dims, bytes) in file order, with the data already in
    // hand: a tensor's size is not a number to trust from a header when the
    // bytes are right here.
    let mut entries: Vec<Written> = Vec::new();
    let (version, alignment, kv) = match &pres {
        Some(pres) => {
            for t in &pres.tensors {
                let data = if t.fields.is_empty() {
                    let r = table.tensors.get(&t.name).ok_or_else(|| {
                        Error::Core(format!("`{}` is not in the tensor table", t.name))
                    })?;
                    let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
                    let Expr::Literal { chunks, .. } = &desc.value else {
                        return Err(Error::Core(format!(
                            "`{}` was imported dense and is no longer a literal",
                            t.name
                        )));
                    };
                    ctx.chunk_bytes(chunks)
                        .map_err(|e| Error::Core(e.to_string()))?
                } else {
                    let mut parts: Vec<(&'static str, Vec<u8>)> = Vec::new();
                    for fld in fields(t.ty) {
                        let (_, r) =
                            t.fields
                                .iter()
                                .find(|(n, _)| n == fld.name)
                                .ok_or_else(|| {
                                    Error::Malformed(format!(
                                        "`{}`: the container does not carry field `{}`",
                                        t.name, fld.name
                                    ))
                                })?;
                        parts.push((
                            fld.name,
                            ctx.chunk_bytes(r).map_err(|e| Error::Core(e.to_string()))?,
                        ));
                    }
                    let numel: u64 = t.dims.iter().product();
                    let (be, _) =
                        t.ty.block()
                            .expect("a preserved field list implies a block");
                    reassemble(t.ty, &parts, (numel / be) as usize)?
                };
                // §05.2.4 permits both representations, and this is where they
                // are made to agree: the cache is compared with what the
                // structural form reassembles to, and a disagreement is an
                // error rather than a preference between two answers.
                if let Some((_, r)) = caches.iter().find(|(n, _)| *n == t.name) {
                    let cached = ctx.chunk_bytes(r).map_err(|e| Error::Core(e.to_string()))?;
                    if cached != data {
                        return Err(Error::Core(format!(
                            "`{}`: the opaque cache holds {} bytes and the \
                             structural form reassembles to {}; §10.6 rule 2 \
                             makes that a stale cache rather than a choice",
                            t.name,
                            cached.len(),
                            data.len()
                        )));
                    }
                    p.opaque_checked += 1;
                }
                entries.push((t.name.clone(), t.ty, t.dims.clone(), t.offset, data));
            }
            (pres.version, pres.alignment, pres.kv.clone())
        }
        None => {
            let mut at = 0u64;
            for (name, tyname, dims, _) in &p.tensors {
                let r = table
                    .tensors
                    .get(name)
                    .ok_or_else(|| Error::Core(format!("`{name}` is not in the tensor table")))?;
                let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
                let ty = match tyname.as_str() {
                    "F16" => Type::F16,
                    "BF16" => Type::BF16,
                    "F64" => Type::F64,
                    "I8" => Type::I8,
                    "I16" => Type::I16,
                    "I32" => Type::I32,
                    "I64" => Type::I64,
                    _ => Type::F32,
                };
                let dtype = ty.dense_dtype().expect("dense");
                let t = desc
                    .value
                    .eval(ctx)
                    .map_err(|e| Error::Core(e.to_string()))?;
                let mut data = vec![0u8; dtype.packed_bytes(t.data.len() as u64) as usize];
                for (i, v) in t.data.iter().enumerate() {
                    dtype.encode(&mut data, i as u64, *v, crate::dtype::Round::Rne);
                }
                at = at.div_ceil(alignment_of(&pres)) * alignment_of(&pres);
                entries.push((name.clone(), ty, dims.clone(), at, data));
                at += dtype.packed_bytes(t.data.len() as u64);
            }
            (3u32, DEFAULT_ALIGNMENT, Vec::new())
        }
    };

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
    for (k, v) in &kv {
        put_str(&mut out, k);
        out.extend_from_slice(&v.wire_type().to_le_bytes());
        put_meta(&mut out, v);
    }
    for (name, ty, dims, offset, _) in &entries {
        put_str(&mut out, name);
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ty.to_u32().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
    let data_start = (out.len() as u64).div_ceil(alignment) * alignment;
    out.resize(data_start as usize, 0);
    for (name, _, _, offset, data) in &entries {
        let at = (data_start + offset) as usize;
        if at < out.len() {
            return Err(Error::Core(format!(
                "`{name}` would be written at {at}, before the end of what is \
                 already written ({}); the preserved offsets overlap",
                out.len()
            )));
        }
        out.resize(at, 0);
        out.extend_from_slice(data);
    }
    Ok((out, p))
}

fn alignment_of(p: &Option<Preserved>) -> u64 {
    p.as_ref().map(|p| p.alignment).unwrap_or(DEFAULT_ALIGNMENT)
}

// ---------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryStore, WritableStore};

    /// A tiny GGUF writer, used only by these tests. It exists so the fixtures
    /// are built from the format's own definition rather than from this
    /// module's parser — a file written by `File::parse`'s inverse would agree
    /// with it about any mistake they shared.
    struct W {
        out: Vec<u8>,
        kv: Vec<(String, Meta)>,
        tensors: Vec<(String, Type, Vec<u64>, Vec<u8>)>,
    }

    impl W {
        fn new() -> W {
            W {
                out: Vec::new(),
                kv: Vec::new(),
                tensors: Vec::new(),
            }
        }
        fn kv(mut self, k: &str, v: Meta) -> W {
            self.kv.push((k.to_string(), v));
            self
        }
        fn tensor(mut self, name: &str, ty: Type, dims: &[u64], data: Vec<u8>) -> W {
            self.tensors.push((name.into(), ty, dims.to_vec(), data));
            self
        }
        fn finish(mut self) -> Vec<u8> {
            self.out.extend_from_slice(b"GGUF");
            self.out.extend_from_slice(&3u32.to_le_bytes());
            self.out
                .extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            self.out
                .extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (k, v) in &self.kv {
                put_str(&mut self.out, k);
                self.out.extend_from_slice(&v.wire_type().to_le_bytes());
                put_meta(&mut self.out, v);
            }
            let mut at = 0u64;
            let mut offsets = Vec::new();
            for (_, _, _, data) in &self.tensors {
                offsets.push(at);
                at += data.len() as u64;
                at = at.div_ceil(32) * 32;
            }
            for ((name, ty, dims, _), off) in self.tensors.iter().zip(&offsets) {
                put_str(&mut self.out, name);
                self.out
                    .extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    self.out.extend_from_slice(&d.to_le_bytes());
                }
                self.out.extend_from_slice(&ty.to_u32().to_le_bytes());
                self.out.extend_from_slice(&off.to_le_bytes());
            }
            let start = (self.out.len() as u64).div_ceil(32) * 32;
            self.out.resize(start as usize, 0);
            for ((_, _, _, data), off) in self.tensors.iter().zip(&offsets) {
                self.out.resize((start + off) as usize, 0);
                self.out.extend_from_slice(data);
            }
            self.out
        }
    }

    /// Deterministic pseudo-random block bytes: enough structure to exercise
    /// every nibble, shift and sign, and the same on every machine.
    fn junk(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    /// Block bytes whose `ggml_half` scale fields are finite: random bytes make
    /// a NaN scale about one time in 2000, and a NaN scale makes every value in
    /// its block NaN, which would hide a wrong permutation.
    fn blocks(t: Type, n: usize, seed: u64) -> Vec<u8> {
        let (_, bb) = t.block().unwrap();
        let mut b = junk(n * bb as usize, seed);
        for i in 0..n {
            for fld in fields(t) {
                if !matches!(fld.name, "d" | "dmin" | "m" | "s") {
                    continue;
                }
                let at = i * bb as usize + fld.off;
                // An f16 with a small exponent: finite, non-zero, and not
                // subnormal.
                b[at + 1] = (b[at + 1] & 0x83) | 0x0c;
            }
        }
        b
    }

    fn import_one(t: Type, nb: usize) -> (Vec<u8>, Imported) {
        let (be, _) = t.block().unwrap();
        let data = blocks(t, nb, 7 + t.to_u32() as u64);
        let src = W::new()
            .kv("general.architecture", Meta::Str("llama".into()))
            .tensor("w", t, &[be * nb as u64], data)
            .finish();
        let imported = import(
            &src,
            &ImportOpts {
                name: "test/gguf".into(),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{} import: {e}", t.name()));
        (src, imported)
    }

    #[test]
    fn every_block_type_dequantizes_the_way_the_block_layout_says() {
        // The I4 check runs inside `import`, comparing the expression graph
        // against `reference_dequant` element by element, so reaching this
        // assertion at all is the result. What is asserted here is that it
        // really ran over every element.
        for t in [
            Type::Q4_0,
            Type::Q4_1,
            Type::Q5_0,
            Type::Q5_1,
            Type::Q8_0,
            Type::Q8_1,
            Type::Q2K,
            Type::Q3K,
            Type::Q4K,
            Type::Q5K,
            Type::Q6K,
        ] {
            let (be, _) = t.block().unwrap();
            let nb = 3;
            let (_, imported) = import_one(t, nb);
            assert_eq!(
                imported.report.dequant_checked,
                be * nb as u64,
                "{}: not every element was checked",
                t.name()
            );
            assert!(imported.report.lossless, "{}", t.name());
        }
    }

    #[test]
    fn a_wrong_permutation_is_caught_rather_than_plausible() {
        // The point of the second check: byte identity holds no matter how the
        // values are read, so this swaps two axes of one type's expression and
        // shows the import refuses it. Q4_0's nibble order is the classic
        // mistake — elements i and i+16 share a byte, not i and i+1.
        let t = Type::Q4_0;
        let data = blocks(t, 2, 11);
        let mut b = ModelBuilder::new("wrong").chunk_size(1 << 20);
        let mut refs = Vec::new();
        for fld in fields(t) {
            refs.push((fld.name, b.chunk_list(&gather(&data, 18, fld, 2))));
        }
        let s = Stored { refs };
        // The correct expression, and the one that forgets the permute.
        let right = value_expr(t, &s, 2).unwrap();
        let wrong = Expr::Dequantize {
            x: Box::new(reshape(
                s.lit("qs", uint(4), &[2, 16, 2], packed(2, 8)),
                &[2, 32],
            )),
            scheme: match &right {
                Expr::Dequantize { scheme, .. } => scheme.clone(),
                _ => unreachable!(),
            },
        };
        let (objs, _) = b.build();
        let mut mem = MemoryStore::new(HashAlgo::default());
        for o in &objs {
            let _ = mem.put(&o.payload);
        }
        let ctx = Ctx::new(&mem);
        let want = reference_dequant(t, &data, 2).unwrap();
        let a = right.eval(&ctx).unwrap();
        let bad = wrong.eval(&ctx).unwrap();
        assert!(a.data.iter().zip(&want).all(|(x, y)| *x == *y as f64));
        assert!(
            bad.data.iter().zip(&want).any(|(x, y)| *x != *y as f64),
            "the un-permuted reading is supposed to disagree"
        );
    }

    #[test]
    fn a_gguf_file_survives_the_round_trip_byte_for_byte() {
        // Every supported type in one file, with dense tensors between them so
        // the offsets and the padding are exercised too.
        let mut w = W::new()
            .kv("general.architecture", Meta::Str("llama".into()))
            .kv("general.name", Meta::Str("round trip".into()))
            .kv("llama.block_count", Meta::U32(2))
            .kv("llama.embedding_length", Meta::U32(64))
            .kv("llama.attention.head_count", Meta::U32(4))
            .kv("llama.attention.layer_norm_rms_epsilon", Meta::F32(1e-5))
            .kv(
                "tokenizer.ggml.tokens",
                Meta::Arr(8, vec![Meta::Str("a".into()), Meta::Str("b".into())]),
            )
            .kv("general.file_type", Meta::U32(15))
            .kv("split.count", Meta::U16(1))
            .kv("general.quantized", Meta::Bool(true));
        w = w.tensor("norm.weight", Type::F32, &[8], junk(32, 3));
        for (i, t) in [
            Type::Q4_0,
            Type::Q4_1,
            Type::Q5_0,
            Type::Q5_1,
            Type::Q8_0,
            Type::Q2K,
            Type::Q3K,
            Type::Q4K,
            Type::Q5K,
            Type::Q6K,
        ]
        .into_iter()
        .enumerate()
        {
            let (be, _) = t.block().unwrap();
            w = w.tensor(
                &format!("blk.{i}.weight"),
                t,
                &[be, 2],
                blocks(t, 2, 100 + i as u64),
            );
        }
        let src = w
            .tensor("out.weight", Type::F16, &[4, 2], junk(16, 9))
            .finish();

        let imported = import(
            &src,
            &ImportOpts {
                name: "test/round-trip".into(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(imported.report.lossless);

        let mut mem = MemoryStore::new(HashAlgo::default());
        for o in &imported.objects {
            let _ = mem.put(&o.payload);
        }
        let ctx = Ctx::new(&mem);
        let manifest = ctx.value(&imported.root).unwrap();
        let model = crate::expr::parse_ref_value(
            manifest.get("assets").and_then(|a| a.get("model")).unwrap(),
        )
        .unwrap();
        let mv = ctx.value(&model.1).unwrap();
        let tref = crate::expr::parse_ref_value(mv.get("tensors").unwrap()).unwrap();
        let table = TensorTable::load(&ctx, &tref).unwrap();

        let (out, plan) = export(&ctx, &manifest, &table).unwrap();
        assert!(plan.from_gguf);
        assert!(plan.lossless(), "{:?}", plan.loss);
        assert_eq!(out.len(), src.len(), "the file changed length");
        assert!(out == src, "the round trip is not byte-exact");
    }

    #[test]
    fn the_opaque_form_is_attached_on_request_and_checked_on_the_way_out() {
        // §05.2.4 permits both representations. The structural one is canonical
        // and the opaque one is an attachment, so what has to be true is that
        // they agree — and that dropping the attachment changes nothing.
        let t = Type::Q4K;
        let (be, _) = t.block().unwrap();
        let src = W::new()
            .kv("general.architecture", Meta::Str("llama".into()))
            .tensor("w", t, &[be, 2], blocks(t, 2, 41))
            .finish();
        let plain = import(&src, &ImportOpts::default()).unwrap();
        let with = import(
            &src,
            &ImportOpts {
                opaque_cache: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(with.objects.len() > plain.objects.len());
        assert!(with
            .report
            .represented
            .iter()
            .any(|s| s.contains("opaque block cache")));

        let mut mem = MemoryStore::new(HashAlgo::default());
        for o in &with.objects {
            let _ = mem.put(&o.payload);
        }
        let ctx = Ctx::new(&mem);
        let manifest = ctx.value(&with.root).unwrap();
        // §10.6 rule 1: every cache object is flagged droppable.
        let cache = with
            .objects
            .iter()
            .find(|o| o.otype == otype::RUNTIME_CACHE)
            .expect("a cache was attached");
        assert!(cache.oflags & crate::container::oflags::CACHEABLE != 0);
        // Rule 2: it is keyed by what produced it, and that key is the
        // structural expression's digest rather than a name or a date.
        let v = crate::cbor::decode(&cache.payload).unwrap();
        let key = v.get("key").and_then(|k| k.as_bytes()).unwrap().to_vec();
        let model = crate::expr::parse_ref_value(
            manifest.get("assets").and_then(|a| a.get("model")).unwrap(),
        )
        .unwrap();
        let mv = ctx.value(&model.1).unwrap();
        let tref = crate::expr::parse_ref_value(mv.get("tensors").unwrap()).unwrap();
        let table = TensorTable::load(&ctx, &tref).unwrap();
        let desc = TensorDesc::load(&ctx, table.get("w").unwrap()).unwrap();
        assert_eq!(
            key,
            HashAlgo::default()
                .digest(&desc.value.to_value().encode())
                .to_vec()
        );

        // And the export checks the two against each other rather than
        // preferring one.
        let (out, p) = export(&ctx, &manifest, &table).unwrap();
        assert_eq!(p.opaque_checked, 1);
        assert!(out == src);
    }

    #[test]
    fn an_iq_type_is_refused_by_name_rather_than_guessed() {
        let src = W::new()
            .tensor("w", Type::Other(16), &[256], vec![0u8; 256])
            .finish();
        let e = import(&src, &ImportOpts::default()).unwrap_err();
        let m = e.to_string();
        assert!(m.contains("IQ2_XXS"), "{m}");
        assert!(m.contains("codebook"), "{m}");
    }

    #[test]
    fn a_truncated_file_is_refused_before_anything_is_allocated() {
        let src = W::new()
            .tensor("w", Type::Q4_0, &[64], blocks(Type::Q4_0, 2, 1))
            .finish();
        for cut in [4, 16, 24, src.len() - 1] {
            assert!(
                File::parse(&src[..cut]).is_err(),
                "a file cut at {cut} parsed"
            );
        }
        // A tensor count that would need more bytes than the file has.
        let mut lie = src.clone();
        lie[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(File::parse(&lie).is_err());
    }

    #[test]
    fn the_architecture_keys_that_mean_the_same_thing_are_the_only_ones_mapped() {
        let src = W::new()
            .kv("general.architecture", Meta::Str("llama".into()))
            .kv("llama.embedding_length", Meta::U32(4096))
            .kv("llama.block_count", Meta::U32(32))
            .kv("llama.attention.head_count", Meta::U32(32))
            .kv("llama.attention.head_count_kv", Meta::U32(8))
            .kv("llama.rope.dimension_count", Meta::U32(128))
            .kv("llama.rope.freq_base", Meta::F32(10000.0))
            .tensor("w", Type::F32, &[4], junk(16, 2))
            .finish();
        let f = File::parse(&src).unwrap();
        let p = arch_params(&f);
        let keys: Vec<&str> = p.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"hidden_size"));
        assert!(keys.contains(&"n_layers"));
        assert!(keys.contains(&"n_kv_heads"));
        assert!(keys.contains(&"rope"));
        // `rope.dimension_count` is GGML's partial-rotary count, which is not
        // the head dimension; renaming it would be inventing a fact.
        assert!(!keys.contains(&"head_dim"));
    }

    #[test]
    fn a_tokenizer_without_its_pre_tokenizer_is_reported_rather_than_synthesized() {
        let src = W::new()
            .kv("tokenizer.ggml.model", Meta::Str("gpt2".into()))
            .kv("tokenizer.ggml.pre", Meta::Str("llama-bpe".into()))
            .kv(
                "tokenizer.ggml.tokens",
                Meta::Arr(8, vec![Meta::Str("a".into())]),
            )
            .tensor("w", Type::F32, &[4], junk(16, 5))
            .finish();
        let r = import(&src, &ImportOpts::default()).unwrap().report;
        let n = r
            .unrepresented
            .iter()
            .find(|n| n.item == "tokenizer")
            .expect("the tokenizer gap is named");
        assert!(n.reason.contains("pre-tokenizer"));
        assert!(!r.represented.iter().any(|s| s.contains("tokenizer")));
    }

    #[test]
    fn a_chat_template_in_the_metadata_becomes_omni_ct() {
        let src = W::new()
            .kv(
                "tokenizer.chat_template",
                Meta::Str(
                    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}".into(),
                ),
            )
            .tensor("w", Type::F32, &[4], junk(16, 6))
            .finish();
        let r = import(&src, &ImportOpts::default()).unwrap().report;
        assert!(r.represented.iter().any(|s| s.contains("chat_template")));
    }
}
