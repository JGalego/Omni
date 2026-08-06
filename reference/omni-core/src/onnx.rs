//! ONNX import and export (§07; `docs/design/import-export.md` §3, §5.2).
//!
//! ONNX is the only widely adopted *graph* interchange format, and it is the
//! one row of the capability matrix that exercises §07 rather than §04: a
//! safetensors file is tensors, a GGUF file is tensors plus an architecture
//! enum, and an ONNX file is a computation. So this module is where OMNI-IR
//! either absorbs somebody else's graph or does not.
//!
//! ## What ONNX is, on the wire
//!
//! A single protobuf message. There is no framing, no index, no alignment and
//! no content addressing: `ModelProto` contains a `GraphProto`, which contains
//! nodes, initializers (the weights) and value declarations, and a reader has
//! to parse all of it to find any of it. Protobuf's 2 GB message limit is why
//! the *external data* mechanism exists — weights moved into sibling files and
//! referenced by path — which is a second, weaker format inside the first, and
//! [`External`] is what reads it.
//!
//! Nothing here depends on a protobuf library: the wire format is seven kinds
//! of field and a varint, and [`Reader`] implements it in about a hundred
//! lines. The same rule as everywhere else in this crate — untrusted input,
//! `#![forbid(unsafe_code)]`, no dependencies.
//!
//! ## The mapping, and the line it draws
//!
//! §07.1 says ONNX's mistake is *a single abstraction level*: `attention`
//! becomes fifteen primitives, and every backend pattern-matches to get the
//! intent back. An importer can repeat that mistake in the other direction. If
//! `Relu` were imported as `maximum(x, 0)` — which it is, exactly — then the
//! export would have to pattern-match two OMNI ops back into one ONNX op, and
//! the round trip would depend on a peephole matcher rather than on a table.
//!
//! So the rule here is narrow and mechanical: **an ONNX op is translated only
//! when one OMNI op means exactly what it means**. [`MAP`] is that table, it is
//! read in both directions, and it is the only place the correspondence is
//! written. Everything else is carried in a *compat dialect* named after the
//! ONNX domain it came from (`ai.onnx`, `ai.onnx.ml`, `com.microsoft`, …) with
//! its attributes intact, which §11.3 makes a first-class outcome: a reader
//! that does not know `ai.onnx/Relu` may still validate, copy, sign and
//! partially execute the model, and this build's interpreter refuses that one
//! op by name instead of guessing at it.
//!
//! The compat dialect's *version* is the opset the file imported, and that is
//! the most faithful thing that can be recorded: ONNX versions its whole opset
//! monolithically, so every op in one file shares one number. §07.4.1 exists to
//! avoid precisely that, and an import that spread the opset over per-op
//! versions it does not know would be inventing information.
//!
//! ## What is refused rather than approximated
//!
//! * **`STRING`, `COMPLEX64`, `COMPLEX128` initializers.** §04.3 has no
//!   equivalent for the first and no complex *storage* layout this could claim
//!   to have checked for the others.
//! * **The `FNUZ` float8 variants.** They differ from `FLOAT8E4M3FN` and
//!   `FLOAT8E5M2` by an exponent bias, and a dtype whose bias is guessed is
//!   §05.6 rule 1's exact prohibition applied to §04.3.
//! * **Subgraph attributes** (`If`, `Loop`, `Scan` bodies). They are regions in
//!   OMNI-IR (§07.3) and translating a subgraph's scope rules into a region's
//!   is a mapping with a right answer this build has not worked out.
//! * **Training fields** (`TrainingInfoProto`) and **sparse initializers**, each
//!   named where it is found.
//! * **An external-data path that escapes its directory.** `../` and absolute
//!   paths are refused before anything is opened (§12.4).
//!
//! ## What is verified
//!
//! I4 is two checks here, both counted in the report:
//!
//! 1. **Every initializer is re-read through the object graph** and compared
//!    with the bytes the file held, which is what makes "the weights survived"
//!    a measurement.
//! 2. **Every value the imported graph produces is typed by OMNI's own shape
//!    inference and compared with ONNX's**, when the file carries one. Two
//!    independent shape functions disagreeing about a concrete dimension is a
//!    finding, and it is an error rather than a warning: one of the two readers
//!    is wrong about what the model computes.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::expr::{concrete, Ctx, Dim, Expr};
use crate::ir::{self, Block, Function, Level, Module, Op, Region, Type};
use crate::layout::{BitOrder, Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{Fidelity, Note};
use crate::tensor::{TensorDesc, TensorTable};

pub const IMPORTER: &str = "omni-import-onnx";
pub const EXPORTER: &str = "omni-export-onnx";

/// The ONNX IR version this build writes (`onnx.IR_VERSION` 10, ONNX 1.16).
///
/// An import records the file's own and an export reproduces it; this is only
/// what a container with nothing to say gets.
pub const IR_VERSION: i64 = 10;

/// The default opset an export claims when the container names none.
pub const DEFAULT_OPSET: i64 = 17;

/// The domain ONNX spells as the empty string.
pub const AI_ONNX: &str = "ai.onnx";

/// The largest single protobuf field this will materialize, so a declared
/// length cannot be used to allocate the machine's memory (§12.4).
pub const MAX_FIELD: u64 = 1 << 34;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    /// Well-formed, and says something this build will not represent.
    Unsupported(String),
    /// An export would lose something and consent was not given (E2).
    Lossy(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed ONNX: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Lossy(m) => write!(f, "{m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

fn malformed<T>(m: impl Into<String>) -> Res<T> {
    Err(Error::Malformed(m.into()))
}

// ------------------------------------------------------------------ protobuf --

/// One protobuf field's payload.
#[derive(Clone, Debug)]
pub enum Wire<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

impl Wire<'_> {
    fn as_u64(&self) -> Res<u64> {
        match self {
            Wire::Varint(v) => Ok(*v),
            other => malformed(format!("expected a varint, found {}", other.kind())),
        }
    }

    fn as_i64(&self) -> Res<i64> {
        Ok(self.as_u64()? as i64)
    }

    fn as_i32(&self) -> Res<i32> {
        Ok(self.as_u64()? as i32)
    }

    fn as_bytes(&self) -> Res<&[u8]> {
        match self {
            Wire::Bytes(b) => Ok(b),
            other => malformed(format!(
                "expected a length-delimited field, found {}",
                other.kind()
            )),
        }
    }

    fn as_str(&self) -> Res<String> {
        let b = self.as_bytes()?;
        String::from_utf8(b.to_vec())
            .map_err(|_| Error::Malformed("a string field is not UTF-8".into()))
    }

    fn kind(&self) -> &'static str {
        match self {
            Wire::Varint(_) => "a varint",
            Wire::Fixed64(_) => "a 64-bit field",
            Wire::Fixed32(_) => "a 32-bit field",
            Wire::Bytes(_) => "a length-delimited field",
        }
    }
}

/// A protobuf message reader.
///
/// The whole of the wire format that ONNX uses: varints, two fixed widths, and
/// length-delimited bytes. Groups (wire types 3 and 4) were removed from proto3
/// and are refused rather than skipped, because skipping one silently would
/// mean reading the fields after it out of a message this reader does not
/// actually understand.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0 }
    }

    pub fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Res<u64> {
        let mut out = 0u64;
        let mut shift = 0u32;
        loop {
            let b = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| Error::Malformed("a varint runs past the end".into()))?;
            self.pos += 1;
            if shift >= 64 {
                return malformed("a varint is longer than 64 bits");
            }
            out |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
        }
    }

    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|e| *e <= self.buf.len())
            .ok_or_else(|| {
                Error::Malformed(format!(
                    "a field of {n} bytes at {} runs past the {}-byte message",
                    self.pos,
                    self.buf.len()
                ))
            })?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// The next `(field number, payload)`, or `None` at the end of the message.
    pub fn next_field(&mut self) -> Res<Option<(u32, Wire<'a>)>> {
        if self.done() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        if field == 0 {
            return malformed("field number 0 is not a field");
        }
        let payload = match key & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                let b = self.take(8)?;
                Wire::Fixed64(u64::from_le_bytes(b.try_into().unwrap()))
            }
            2 => {
                let n = self.varint()?;
                if n > MAX_FIELD {
                    return malformed(format!(
                        "a field declares {n} bytes, past the {MAX_FIELD}-byte bound this \
                         reader will materialize"
                    ));
                }
                Wire::Bytes(self.take(n as usize)?)
            }
            5 => {
                let b = self.take(4)?;
                Wire::Fixed32(u32::from_le_bytes(b.try_into().unwrap()))
            }
            t @ (3 | 4) => {
                return malformed(format!(
                    "wire type {t} is a group, which proto3 removed; field {field} cannot \
                     be read without knowing where it ends"
                ))
            }
            t => return malformed(format!("wire type {t} does not exist")),
        };
        Ok(Some((field, payload)))
    }
}

/// Repeated scalars, which protobuf may write either packed into one
/// length-delimited field or one field per element. Both appear in real ONNX
/// files — the reference writer packs, older ones do not — so both are read.
fn packed_varints(w: &Wire<'_>) -> Res<Vec<u64>> {
    match w {
        Wire::Varint(v) => Ok(vec![*v]),
        Wire::Bytes(b) => {
            let mut r = Reader::new(b);
            let mut out = Vec::new();
            while !r.done() {
                out.push(r.varint()?);
            }
            Ok(out)
        }
        other => malformed(format!(
            "expected repeated integers, found {}",
            other.kind()
        )),
    }
}

fn packed_f32(w: &Wire<'_>) -> Res<Vec<f32>> {
    match w {
        Wire::Fixed32(v) => Ok(vec![f32::from_bits(*v)]),
        Wire::Bytes(b) => {
            if b.len() % 4 != 0 {
                return malformed("a packed float field is not a multiple of 4 bytes");
            }
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        other => malformed(format!("expected repeated floats, found {}", other.kind())),
    }
}

fn packed_f64(w: &Wire<'_>) -> Res<Vec<f64>> {
    match w {
        Wire::Fixed64(v) => Ok(vec![f64::from_bits(*v)]),
        Wire::Bytes(b) => {
            if b.len() % 8 != 0 {
                return malformed("a packed double field is not a multiple of 8 bytes");
            }
            Ok(b.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        other => malformed(format!("expected repeated doubles, found {}", other.kind())),
    }
}

// -------------------------------------------------------------------- writing --

/// A protobuf message writer.
///
/// Fields are written in ascending number with packed repeated scalars, which
/// is what protobuf's own writers emit and what makes an export of an import
/// comparable byte for byte with its source. The wire format permits any order,
/// so that comparison is a property of two *writers* agreeing, not of the
/// format — which is why the round-trip test says which file it round-trips.
#[derive(Default)]
pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer { buf: Vec::new() }
    }

    fn key(&mut self, field: u32, wire: u32) {
        self.varint(u64::from(field) << 3 | u64::from(wire));
    }

    fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.buf.push(b);
                return;
            }
            self.buf.push(b | 0x80);
        }
    }

    /// A varint field, omitted when zero: proto3 does not write defaults, and
    /// writing them would make an export differ from its source in fields
    /// nobody set.
    pub fn int(&mut self, field: u32, v: i64) {
        if v != 0 {
            self.key(field, 0);
            self.varint(v as u64);
        }
    }

    pub fn int_always(&mut self, field: u32, v: i64) {
        self.key(field, 0);
        self.varint(v as u64);
    }

    pub fn f32(&mut self, field: u32, v: f32) {
        if v != 0.0 {
            self.key(field, 5);
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn bytes(&mut self, field: u32, v: &[u8]) {
        if v.is_empty() {
            return;
        }
        self.raw_bytes(field, v);
    }

    /// A length-delimited field written even when empty — which matters for a
    /// repeated string like a node's inputs, where an empty entry means "this
    /// optional operand was not supplied".
    pub fn raw_bytes(&mut self, field: u32, v: &[u8]) {
        self.key(field, 2);
        self.varint(v.len() as u64);
        self.buf.extend_from_slice(v);
    }

    pub fn text(&mut self, field: u32, v: &str) {
        self.bytes(field, v.as_bytes());
    }

    pub fn message(&mut self, field: u32, w: Writer) {
        self.raw_bytes(field, &w.buf);
    }

    pub fn packed_ints(&mut self, field: u32, v: &[i64]) {
        if v.is_empty() {
            return;
        }
        let mut inner = Writer::new();
        for x in v {
            inner.varint(*x as u64);
        }
        self.raw_bytes(field, &inner.buf);
    }

    pub fn packed_f32(&mut self, field: u32, v: &[f32]) {
        if v.is_empty() {
            return;
        }
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        self.raw_bytes(field, &b);
    }
}

// -------------------------------------------------------------------- the proto --

/// `TensorProto.DataType`, with the OMNI dtype each denotes and how ONNX
/// arranges it in bytes.
///
/// This is the whole correspondence, written once so the importer and the
/// exporter cannot disagree about it. The types with no row are refused by
/// name: [`dtype_of`] says which and why.
pub const DTYPES: &[(i32, &str, &str)] = &[
    (1, "FLOAT", "f32"),
    (2, "UINT8", "u8"),
    (3, "INT8", "i8"),
    (4, "UINT16", "u16"),
    (5, "INT16", "i16"),
    (6, "INT32", "i32"),
    (7, "INT64", "i64"),
    (9, "BOOL", "bool"),
    (10, "FLOAT16", "f16"),
    (11, "DOUBLE", "f64"),
    (12, "UINT32", "u32"),
    (13, "UINT64", "u64"),
    (16, "BFLOAT16", "bf16"),
    (17, "FLOAT8E4M3FN", "f8e4m3"),
    (19, "FLOAT8E5M2", "f8e5m2"),
    (21, "UINT4", "u4"),
    (22, "INT4", "i4"),
    (23, "FLOAT4E2M1", "f4e2m1"),
];

/// The ONNX name of a data type, including the ones this build refuses, so an
/// error can say what it refused rather than a number.
pub fn dtype_name(code: i32) -> &'static str {
    match code {
        0 => "UNDEFINED",
        8 => "STRING",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        18 => "FLOAT8E4M3FNUZ",
        20 => "FLOAT8E5M2FNUZ",
        other => DTYPES
            .iter()
            .find(|(c, _, _)| *c == other)
            .map(|(_, n, _)| *n)
            .unwrap_or("(a data type this build has no name for)"),
    }
}

/// The OMNI dtype an ONNX data type denotes.
pub fn dtype_of(code: i32) -> Res<DType> {
    match DTYPES.iter().find(|(c, _, _)| *c == code) {
        Some((_, _, alias)) => DType::from_alias(alias)
            .ok_or_else(|| Error::Core(format!("no dtype for the alias `{alias}`"))),
        None => Err(Error::Unsupported(match code {
            8 => "an initializer of type STRING: §04.3 is a numeric type algebra and \
                  has no string element type"
                .into(),
            14 | 15 => format!(
                "an initializer of type {}: §04.3 has a complex dtype, but no layout \
                 that states how ONNX interleaves the two halves, and a guess about \
                 interleaving is a guess about every value",
                dtype_name(code)
            ),
            18 | 20 => format!(
                "an initializer of type {}: the FNUZ float8 formats differ from \
                 {} by an exponent bias, and importing one as the other would \
                 change every value it holds",
                dtype_name(code),
                if code == 18 {
                    "FLOAT8E4M3FN"
                } else {
                    "FLOAT8E5M2"
                }
            ),
            other => format!("data type {} ({other})", dtype_name(other)),
        })),
    }
}

/// The ONNX data type an OMNI dtype maps back to, if the format has one.
pub fn code_of(d: &DType) -> Option<i32> {
    DTYPES
        .iter()
        .find(|(_, _, alias)| DType::from_alias(alias).as_ref() == Some(d))
        .map(|(c, _, _)| *c)
}

/// How ONNX arranges a tensor of this dtype.
///
/// Dense row-major for everything except the sub-byte types, which ONNX packs
/// exactly as §04.4's `packed` layout describes: `BOOL` gets a whole byte per
/// element, and the 4-bit types get two elements per byte, low nibble first.
pub fn layout_of(d: &DType) -> Layout {
    let packed = |elems_per_word: u32| Layout::Packed {
        elems_per_word,
        word_bits: 8,
        bit_order: BitOrder::LsbFirst,
        order: Order::RowMajor,
    };
    if d == &DType::Bool {
        packed(1)
    } else if d == &DType::I4 || d == &DType::U4 || d == &DType::F4E2M1 {
        packed(2)
    } else {
        Layout::row_major()
    }
}

/// Bytes ONNX uses for a tensor of this dtype and shape.
pub fn stored_bytes(d: &DType, shape: &[u64]) -> u64 {
    let n: u64 = shape.iter().product();
    layout_of(d)
        .stored_bytes(shape, d)
        .unwrap_or_else(|| d.packed_bytes(n))
}

/// `AttributeProto.AttributeType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttrType {
    #[default]
    Undefined,
    Float,
    Int,
    String,
    Tensor,
    Graph,
    SparseTensor,
    TypeProto,
    Floats,
    Ints,
    Strings,
    Tensors,
    Graphs,
    SparseTensors,
    TypeProtos,
}

impl AttrType {
    pub fn from_code(c: i64) -> AttrType {
        match c {
            1 => AttrType::Float,
            2 => AttrType::Int,
            3 => AttrType::String,
            4 => AttrType::Tensor,
            5 => AttrType::Graph,
            6 => AttrType::Floats,
            7 => AttrType::Ints,
            8 => AttrType::Strings,
            9 => AttrType::Tensors,
            10 => AttrType::Graphs,
            11 => AttrType::SparseTensor,
            12 => AttrType::SparseTensors,
            13 => AttrType::TypeProto,
            14 => AttrType::TypeProtos,
            _ => AttrType::Undefined,
        }
    }

    pub fn code(self) -> i64 {
        match self {
            AttrType::Undefined => 0,
            AttrType::Float => 1,
            AttrType::Int => 2,
            AttrType::String => 3,
            AttrType::Tensor => 4,
            AttrType::Graph => 5,
            AttrType::Floats => 6,
            AttrType::Ints => 7,
            AttrType::Strings => 8,
            AttrType::Tensors => 9,
            AttrType::Graphs => 10,
            AttrType::SparseTensor => 11,
            AttrType::SparseTensors => 12,
            AttrType::TypeProto => 13,
            AttrType::TypeProtos => 14,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AttrType::Undefined => "UNDEFINED",
            AttrType::Float => "FLOAT",
            AttrType::Int => "INT",
            AttrType::String => "STRING",
            AttrType::Tensor => "TENSOR",
            AttrType::Graph => "GRAPH",
            AttrType::SparseTensor => "SPARSE_TENSOR",
            AttrType::TypeProto => "TYPE_PROTO",
            AttrType::Floats => "FLOATS",
            AttrType::Ints => "INTS",
            AttrType::Strings => "STRINGS",
            AttrType::Tensors => "TENSORS",
            AttrType::Graphs => "GRAPHS",
            AttrType::SparseTensors => "SPARSE_TENSORS",
            AttrType::TypeProtos => "TYPE_PROTOS",
        }
    }
}

/// One `AttributeProto`. Only the forms this build can carry are kept as
/// values; a subgraph or a sparse tensor is recorded as *present* so the import
/// can refuse the node by name rather than dropping the attribute silently.
#[derive(Clone, Debug, Default)]
pub struct Attr {
    pub name: String,
    pub kind: AttrType,
    pub f: f32,
    pub i: i64,
    pub s: Vec<u8>,
    pub t: Option<Tensor>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<Vec<u8>>,
    /// A subgraph, a sparse tensor or a type: present, and not represented.
    pub opaque: bool,
    pub doc_string: String,
    pub ref_attr_name: String,
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<Attr>,
    pub doc_string: String,
}

impl Node {
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|a| a.name == name)
    }

    /// The dialect namespace this node's domain names. ONNX spells its own
    /// domain as the empty string; everything else is already reverse-DNS,
    /// which is what §11.2 asks a namespace to be.
    pub fn dialect(&self) -> &str {
        if self.domain.is_empty() {
            AI_ONNX
        } else {
            &self.domain
        }
    }

    /// How an error names this node: ONNX node names are optional, so the
    /// output it defines is the identifier that always exists.
    pub fn label(&self) -> String {
        if !self.name.is_empty() {
            format!("`{}` ({})", self.name, self.op_type)
        } else if let Some(o) = self.outputs.first() {
            format!("the {} producing `{o}`", self.op_type)
        } else {
            format!("a {} node", self.op_type)
        }
    }
}

/// One `TensorProto`: an initializer, or a `Constant` node's value.
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub data_type: i32,
    pub raw: Option<Vec<u8>>,
    pub floats: Vec<f32>,
    pub int32s: Vec<i32>,
    pub int64s: Vec<i64>,
    pub doubles: Vec<f64>,
    pub uint64s: Vec<u64>,
    pub strings: usize,
    /// `external_data` key/value pairs, in the file's order.
    pub external: Vec<(String, String)>,
    pub data_location: i32,
    pub has_segment: bool,
    pub doc_string: String,
}

impl Tensor {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A shape dimension: a size, a symbol, or nothing said at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PDim {
    Value(i64),
    Param(String),
    Unknown,
}

/// A `TypeProto`. Only tensor types are modelled; the others are recorded by
/// name, because a graph whose input is a sequence is not a graph this build
/// can give a §07.3.1 type to.
#[derive(Clone, Debug)]
pub enum PType {
    Tensor {
        elem: i32,
        /// `None` when the type declares no shape at all, which is different
        /// from declaring a shape whose dimensions are unknown.
        shape: Option<Vec<PDim>>,
    },
    Other(&'static str),
}

#[derive(Clone, Debug, Default)]
pub struct ValueInfo {
    pub name: String,
    pub ty: Option<PType>,
    pub doc_string: String,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub initializers: Vec<Tensor>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
    pub doc_string: String,
    pub sparse_initializers: usize,
    pub quantization_annotations: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: Graph,
    /// `(domain, version)` in the file's order. The empty domain is ONNX's own.
    pub opsets: Vec<(String, i64)>,
    pub metadata: Vec<(String, String)>,
    pub training_infos: usize,
    pub functions: usize,
}

impl Model {
    /// The opset version a domain was imported at.
    pub fn opset(&self, domain: &str) -> Option<i64> {
        let want = if domain == AI_ONNX { "" } else { domain };
        self.opsets
            .iter()
            .find(|(d, _)| d == want || (want.is_empty() && d == AI_ONNX))
            .map(|(_, v)| *v)
    }
}

// ------------------------------------------------------------------- decoding --

fn string_pairs(b: &[u8]) -> Res<(String, String)> {
    let mut r = Reader::new(b);
    let (mut k, mut v) = (String::new(), String::new());
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => k = w.as_str()?,
            2 => v = w.as_str()?,
            _ => {}
        }
    }
    Ok((k, v))
}

fn decode_tensor(b: &[u8]) -> Res<Tensor> {
    let mut t = Tensor::default();
    let mut r = Reader::new(b);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => {
                for d in packed_varints(&w)? {
                    let n = d as i64;
                    if n < 0 {
                        return malformed(format!("a tensor declares a dimension of {n}"));
                    }
                    t.dims.push(n as u64);
                }
            }
            2 => t.data_type = w.as_i32()?,
            3 => t.has_segment = true,
            4 => t.floats.extend(packed_f32(&w)?),
            5 => t
                .int32s
                .extend(packed_varints(&w)?.into_iter().map(|v| v as i32)),
            6 => t.strings += 1,
            7 => t
                .int64s
                .extend(packed_varints(&w)?.into_iter().map(|v| v as i64)),
            8 => t.name = w.as_str()?,
            9 => t.raw = Some(w.as_bytes()?.to_vec()),
            10 => t.doubles.extend(packed_f64(&w)?),
            11 => t.uint64s.extend(packed_varints(&w)?),
            12 => t.doc_string = w.as_str()?,
            13 => t.external.push(string_pairs(w.as_bytes()?)?),
            14 => t.data_location = w.as_i32()?,
            _ => {}
        }
    }
    Ok(t)
}

fn decode_attr(b: &[u8]) -> Res<Attr> {
    let mut a = Attr::default();
    let mut r = Reader::new(b);
    let mut explicit_kind = None;
    // Which payload fields were present, so an attribute with no `type` — legal
    // in early ONNX files — can still be classified.
    let (mut has_f, mut has_i, mut has_s, mut has_t) = (false, false, false, false);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => a.name = w.as_str()?,
            2 => {
                a.f = match w {
                    Wire::Fixed32(v) => f32::from_bits(v),
                    other => return malformed(format!("attribute `f` is {}", other.kind())),
                };
                has_f = true;
            }
            3 => {
                a.i = w.as_i64()?;
                has_i = true;
            }
            4 => {
                a.s = w.as_bytes()?.to_vec();
                has_s = true;
            }
            5 => {
                a.t = Some(decode_tensor(w.as_bytes()?)?);
                has_t = true;
            }
            6 | 11 | 14 | 15 | 22 | 23 => a.opaque = true,
            7 => a.floats.extend(packed_f32(&w)?),
            8 => a
                .ints
                .extend(packed_varints(&w)?.into_iter().map(|v| v as i64)),
            9 => a.strings.push(w.as_bytes()?.to_vec()),
            10 => a.opaque = true,
            13 => a.doc_string = w.as_str()?,
            20 => explicit_kind = Some(AttrType::from_code(w.as_i64()?)),
            21 => a.ref_attr_name = w.as_str()?,
            _ => {}
        }
    }
    a.kind = match explicit_kind {
        Some(AttrType::Undefined) | None => {
            if has_f {
                AttrType::Float
            } else if has_i {
                AttrType::Int
            } else if has_s {
                AttrType::String
            } else if has_t {
                AttrType::Tensor
            } else if !a.floats.is_empty() {
                AttrType::Floats
            } else if !a.ints.is_empty() {
                AttrType::Ints
            } else if !a.strings.is_empty() {
                AttrType::Strings
            } else {
                AttrType::Undefined
            }
        }
        Some(k) => k,
    };
    Ok(a)
}

fn decode_node(b: &[u8]) -> Res<Node> {
    let mut n = Node::default();
    let mut r = Reader::new(b);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => n.inputs.push(w.as_str()?),
            2 => n.outputs.push(w.as_str()?),
            3 => n.name = w.as_str()?,
            4 => n.op_type = w.as_str()?,
            5 => n.attrs.push(decode_attr(w.as_bytes()?)?),
            6 => n.doc_string = w.as_str()?,
            7 => n.domain = w.as_str()?,
            _ => {}
        }
    }
    Ok(n)
}

fn decode_type(b: &[u8]) -> Res<PType> {
    let mut r = Reader::new(b);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => {
                let mut tr = Reader::new(w.as_bytes()?);
                let mut elem = 0i32;
                let mut shape = None;
                while let Some((tf, tw)) = tr.next_field()? {
                    match tf {
                        1 => elem = tw.as_i32()?,
                        2 => {
                            let mut sr = Reader::new(tw.as_bytes()?);
                            let mut dims = Vec::new();
                            while let Some((sf, sw)) = sr.next_field()? {
                                if sf != 1 {
                                    continue;
                                }
                                let mut dr = Reader::new(sw.as_bytes()?);
                                let mut d = PDim::Unknown;
                                while let Some((df, dw)) = dr.next_field()? {
                                    match df {
                                        1 => d = PDim::Value(dw.as_i64()?),
                                        2 => d = PDim::Param(dw.as_str()?),
                                        _ => {}
                                    }
                                }
                                dims.push(d);
                            }
                            shape = Some(dims);
                        }
                        _ => {}
                    }
                }
                return Ok(PType::Tensor { elem, shape });
            }
            4 => return Ok(PType::Other("a sequence")),
            5 => return Ok(PType::Other("a map")),
            8 => return Ok(PType::Other("a sparse tensor")),
            9 => return Ok(PType::Other("an optional")),
            _ => {}
        }
    }
    Ok(PType::Other("a type with no kind"))
}

fn decode_value_info(b: &[u8]) -> Res<ValueInfo> {
    let mut v = ValueInfo::default();
    let mut r = Reader::new(b);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => v.name = w.as_str()?,
            2 => v.ty = Some(decode_type(w.as_bytes()?)?),
            3 => v.doc_string = w.as_str()?,
            _ => {}
        }
    }
    Ok(v)
}

fn decode_graph(b: &[u8]) -> Res<Graph> {
    let mut g = Graph::default();
    let mut r = Reader::new(b);
    while let Some((f, w)) = r.next_field()? {
        match f {
            1 => g.nodes.push(decode_node(w.as_bytes()?)?),
            2 => g.name = w.as_str()?,
            5 => g.initializers.push(decode_tensor(w.as_bytes()?)?),
            10 => g.doc_string = w.as_str()?,
            11 => g.inputs.push(decode_value_info(w.as_bytes()?)?),
            12 => g.outputs.push(decode_value_info(w.as_bytes()?)?),
            13 => g.value_info.push(decode_value_info(w.as_bytes()?)?),
            14 => g.quantization_annotations += 1,
            15 => g.sparse_initializers += 1,
            _ => {}
        }
    }
    Ok(g)
}

impl Model {
    /// Parses a `ModelProto`.
    pub fn parse(bytes: &[u8]) -> Res<Model> {
        let mut m = Model::default();
        let mut r = Reader::new(bytes);
        let mut saw_graph = false;
        while let Some((f, w)) = r.next_field()? {
            match f {
                1 => m.ir_version = w.as_i64()?,
                2 => m.producer_name = w.as_str()?,
                3 => m.producer_version = w.as_str()?,
                4 => m.domain = w.as_str()?,
                5 => m.model_version = w.as_i64()?,
                6 => m.doc_string = w.as_str()?,
                7 => {
                    m.graph = decode_graph(w.as_bytes()?)?;
                    saw_graph = true;
                }
                8 => {
                    let (d, v) = {
                        let mut orr = Reader::new(w.as_bytes()?);
                        let (mut d, mut v) = (String::new(), 0i64);
                        while let Some((of, ow)) = orr.next_field()? {
                            match of {
                                1 => d = ow.as_str()?,
                                2 => v = ow.as_i64()?,
                                _ => {}
                            }
                        }
                        (d, v)
                    };
                    m.opsets.push((d, v));
                }
                14 => m.metadata.push(string_pairs(w.as_bytes()?)?),
                20 => m.training_infos += 1,
                25 => m.functions += 1,
                _ => {}
            }
        }
        if m.ir_version == 0 {
            return malformed(
                "no `ir_version`: every ONNX file states one, and a file that does not \
                 is not one this reader will guess at",
            );
        }
        if !saw_graph {
            return malformed("no `graph`: a ModelProto without one describes nothing");
        }
        Ok(m)
    }
}

// ------------------------------------------------------------- external data --

/// Where an initializer with `data_location: EXTERNAL` gets its bytes.
///
/// ONNX moved weights out of the message because protobuf refuses to encode
/// more than 2 GB, and the mechanism it reached for is a *path in a string*.
/// That is an untrusted path (§12.4), so resolution is a caller's decision
/// rather than something this module does to the filesystem on its own.
pub trait External {
    /// The bytes of `path`, which is relative to the model file's directory.
    fn read(&self, path: &str) -> Option<Vec<u8>>;
}

/// No external data is available. An initializer that needs it fails by name
/// rather than importing as zeros.
pub struct NoExternal;

impl External for NoExternal {
    fn read(&self, _path: &str) -> Option<Vec<u8>> {
        None
    }
}

/// External data resolved against the directory the model file is in.
///
/// A path that escapes that directory is refused before anything is opened: an
/// ONNX file names its own weight files, and a file from the internet naming
/// `../../.ssh/id_ed25519` is the reason this check is not a nicety.
pub struct DirExternal {
    pub dir: std::path::PathBuf,
}

impl DirExternal {
    /// Whether a path stays inside the model's directory. Refuses absolute
    /// paths, drive letters, backslashes and any `..` component.
    pub fn is_contained(path: &str) -> bool {
        if path.is_empty() || path.starts_with('/') || path.contains('\\') {
            return false;
        }
        if path.len() >= 2 && path.as_bytes()[1] == b':' {
            return false;
        }
        !path.split('/').any(|c| c == ".." || c.is_empty())
    }
}

impl External for DirExternal {
    fn read(&self, path: &str) -> Option<Vec<u8>> {
        if !DirExternal::is_contained(path) {
            return None;
        }
        std::fs::read(self.dir.join(path)).ok()
    }
}

/// The dense little-endian bytes of an initializer, in the arrangement §04.4
/// says its layout has.
///
/// ONNX stores a tensor three ways — `raw_data`, a typed array, or a sibling
/// file — and all three mean the same values. What this refuses is the fourth
/// possibility: a tensor that says one length and carries another.
pub fn tensor_bytes(t: &Tensor, ext: &dyn External) -> Res<Vec<u8>> {
    let dtype = dtype_of(t.data_type)?;
    let want = stored_bytes(&dtype, &t.dims);
    let n = t.numel();

    if t.data_location == 1 {
        let get = |k: &str| {
            t.external
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        let location = get("location").ok_or_else(|| {
            Error::Malformed(format!("`{}` is external data with no `location`", t.name))
        })?;
        if !DirExternal::is_contained(location) {
            return Err(Error::Unsupported(format!(
                "`{}` names the external data file `{location}`, which leaves the \
                 model's own directory; §12.4 does not let a file decide which of \
                 the reader's files to open",
                t.name
            )));
        }
        let bytes = ext.read(location).ok_or_else(|| {
            Error::Unsupported(format!(
                "`{}` keeps its data in `{location}`, which is not available to this \
                 import; ONNX external data is a second file and it has to be beside \
                 the first",
                t.name
            ))
        })?;
        let parse = |k: &str| -> Res<Option<u64>> {
            match get(k) {
                None => Ok(None),
                Some(v) => v.parse::<u64>().map(Some).map_err(|_| {
                    Error::Malformed(format!("`{}` has a non-numeric external `{k}`", t.name))
                }),
            }
        };
        let offset = parse("offset")?.unwrap_or(0);
        let length =
            parse("length")?.unwrap_or(bytes.len() as u64 - offset.min(bytes.len() as u64));
        let end = offset.saturating_add(length);
        if end > bytes.len() as u64 {
            return Err(Error::Malformed(format!(
                "`{}` claims bytes {offset}..{end} of `{location}`, which is {} bytes",
                t.name,
                bytes.len()
            )));
        }
        let slice = bytes[offset as usize..end as usize].to_vec();
        if slice.len() as u64 != want {
            return Err(Error::Malformed(format!(
                "`{}` is {n} elements of {} = {want} bytes, and its external range is {}",
                t.name,
                dtype_name(t.data_type),
                slice.len()
            )));
        }
        return Ok(slice);
    }

    if let Some(raw) = &t.raw {
        if raw.len() as u64 != want {
            return Err(Error::Malformed(format!(
                "`{}` is {n} elements of {} = {want} bytes, and its `raw_data` is {}",
                t.name,
                dtype_name(t.data_type),
                raw.len()
            )));
        }
        return Ok(raw.clone());
    }

    // A typed array. Every ONNX type has exactly one field it may use, and the
    // sub-byte types have none: their `int32_data` form stores one element per
    // int32 while `raw_data` packs two per byte, so a reader that accepted both
    // would have to decide which packing the tensor meant.
    let mut out = Vec::with_capacity(want as usize);
    let count: u64 = match t.data_type {
        1 => {
            for x in &t.floats {
                out.extend_from_slice(&x.to_le_bytes());
            }
            t.floats.len() as u64
        }
        11 => {
            for x in &t.doubles {
                out.extend_from_slice(&x.to_le_bytes());
            }
            t.doubles.len() as u64
        }
        7 => {
            for x in &t.int64s {
                out.extend_from_slice(&x.to_le_bytes());
            }
            t.int64s.len() as u64
        }
        12 => {
            for x in &t.uint64s {
                out.extend_from_slice(&(*x as u32).to_le_bytes());
            }
            t.uint64s.len() as u64
        }
        13 => {
            for x in &t.uint64s {
                out.extend_from_slice(&x.to_le_bytes());
            }
            t.uint64s.len() as u64
        }
        6 => {
            for x in &t.int32s {
                out.extend_from_slice(&x.to_le_bytes());
            }
            t.int32s.len() as u64
        }
        // The narrow types travel in `int32_data` as their own bit pattern.
        4 | 5 | 10 | 16 => {
            for x in &t.int32s {
                out.extend_from_slice(&(*x as u16).to_le_bytes());
            }
            t.int32s.len() as u64
        }
        2 | 3 | 9 | 17 | 19 => {
            for x in &t.int32s {
                out.push(*x as u8);
            }
            t.int32s.len() as u64
        }
        21..=23 => {
            return Err(Error::Unsupported(format!(
                "`{}` is {} and stores its values in `int32_data`: the packed form \
                 puts two elements in a byte and the typed form puts one in a word, \
                 and this build reads the packed one",
                t.name,
                dtype_name(t.data_type)
            )))
        }
        other => return Err(dtype_of(other).map(|_| unreachable!()).unwrap_err()),
    };
    if count != n {
        return Err(Error::Malformed(format!(
            "`{}` declares {:?} = {n} elements and carries {count}",
            t.name, t.dims
        )));
    }
    if out.len() as u64 != want {
        return Err(Error::Malformed(format!(
            "`{}` encodes to {} bytes where its shape and dtype need {want}",
            t.name,
            out.len()
        )));
    }
    Ok(out)
}

// -------------------------------------------------------------- the op table --

/// ONNX ops that are exactly one OMNI op, with nothing to translate but the
/// name.
///
/// Read in both directions. Anything not here — and not one of the attributed
/// forms [`map_node`] handles — is carried in the compat dialect rather than
/// approximated, which is the rule this module's header states and the reason
/// `Relu` is not in the table: it is `maximum(x, 0)`, and writing that would
/// make the export a pattern matcher.
pub const MAP: &[(&str, &str, &str)] = &[
    ("Add", "omni.tensor", "add"),
    ("Sub", "omni.tensor", "sub"),
    ("Mul", "omni.tensor", "mul"),
    ("Div", "omni.tensor", "div"),
    ("Neg", "omni.tensor", "neg"),
    ("Exp", "omni.tensor", "exp"),
    ("Log", "omni.tensor", "log"),
    ("Sqrt", "omni.tensor", "sqrt"),
    ("Tanh", "omni.tensor", "tanh"),
    ("Sigmoid", "omni.tensor", "sigmoid"),
    ("Erf", "omni.tensor", "erf"),
    ("MatMul", "omni.tensor", "matmul"),
    ("Where", "omni.tensor", "where"),
];

/// The ONNX reductions, paired with `omni.tensor/reduce`'s `kind`.
pub const REDUCTIONS: &[(&str, &str)] = &[
    ("ReduceSum", "sum"),
    ("ReduceMean", "mean"),
    ("ReduceMax", "max"),
    ("ReduceMin", "min"),
    ("ReduceProd", "prod"),
];

// ------------------------------------------------------------- graph mapping --

/// A dimension as §07.3.1 states it. A negative `dim_value` is what some
/// exporters write for "unknown", and it means the same as saying nothing.
fn pdim(d: &PDim) -> Dim {
    match d {
        PDim::Value(n) if *n >= 0 => Dim::N(*n as u64),
        PDim::Value(_) | PDim::Unknown => Dim::Dynamic,
        PDim::Param(s) => Dim::Sym(s.clone()),
    }
}

/// The §07.3.1 type an ONNX `TypeProto` denotes.
fn ptype(p: &PType, what: &str) -> Res<Type> {
    match p {
        PType::Tensor { elem, shape } => {
            let dtype = dtype_of(*elem).map_err(|e| match e {
                Error::Unsupported(m) => Error::Unsupported(format!("{what}: {m}")),
                other => other,
            })?;
            match shape {
                Some(dims) => Ok(Type::tensor(dims.iter().map(pdim).collect(), dtype)),
                None => Err(Error::Unsupported(format!(
                    "{what} declares an element type and no shape, so its rank is \
                     unknown; §07.3.1 has dynamic dimensions but not dynamic rank, and \
                     a rank invented here would be invented for every op downstream"
                ))),
            }
        }
        PType::Other(kind) => Err(Error::Unsupported(format!(
            "{what} is {kind}; this build gives §07.3.1 types to tensors, and a \
             sequence or map input is a different graph"
        ))),
    }
}

/// What one ONNX node becomes.
enum Mapped {
    /// One OMNI op that means exactly what the node means.
    Native {
        dialect: &'static str,
        name: &'static str,
        attrs: Vec<(String, Value)>,
        /// How many of the node's inputs stay operands. The rest were constants
        /// folded into attributes.
        operands: usize,
    },
    /// Carried in the compat dialect, attributes intact.
    Compat {
        attrs: Vec<(String, Value)>,
        /// Why it was not translated, for the report.
        reason: String,
    },
}

fn int_value(n: i64) -> Value {
    if n < 0 {
        Value::I(n)
    } else {
        Value::U(n as u64)
    }
}

fn int_array(v: &[i64]) -> Value {
    Value::Array(v.iter().map(|n| int_value(*n)).collect())
}

/// One ONNX attribute, in a form that round-trips: the key is the proto field
/// the value came from, so the export puts it back where it was.
fn compat_attr(a: &Attr) -> Res<Value> {
    let text_or_bytes = |b: &Vec<u8>| match std::str::from_utf8(b) {
        Ok(s) => Value::map(vec![("s", Value::text(s.to_string()))]),
        Err(_) => Value::map(vec![("b", Value::Bytes(b.clone()))]),
    };
    Ok(match a.kind {
        AttrType::Float => Value::map(vec![("f", Value::F64(f64::from(a.f)))]),
        AttrType::Int => Value::map(vec![("i", int_value(a.i))]),
        AttrType::String => text_or_bytes(&a.s),
        AttrType::Floats => Value::map(vec![(
            "floats",
            Value::Array(a.floats.iter().map(|x| Value::F64(f64::from(*x))).collect()),
        )]),
        AttrType::Ints => Value::map(vec![("ints", int_array(&a.ints))]),
        AttrType::Strings => Value::map(vec![(
            "strings",
            Value::Array(a.strings.iter().map(text_or_bytes).collect()),
        )]),
        other => {
            return Err(Error::Unsupported(format!(
                "an attribute of type {}: a {} is a graph object rather than a value, \
                 and §07.3 makes it a region or a tensor rather than an attribute",
                other.name(),
                other.name()
            )))
        }
    })
}

/// The importer's decisions about one node.
///
/// Every branch that declines to translate says *why* in a sentence that names
/// the difference, because "unsupported" is the answer that teaches nobody
/// anything about a format's semantics.
fn map_node(
    node: &Node,
    opset: i64,
    ins: &[Type],
    const_ints: &dyn Fn(&str) -> Option<Vec<i64>>,
) -> Res<Mapped> {
    let compat = |reason: &str| -> Res<Mapped> {
        let mut attrs = Vec::new();
        for a in &node.attrs {
            if !a.ref_attr_name.is_empty() {
                return Err(Error::Unsupported(format!(
                    "{} refers to the function attribute `{}`; a node whose attributes \
                     come from a caller is a function body, and this build imports \
                     graphs",
                    node.label(),
                    a.ref_attr_name
                )));
            }
            if a.opaque {
                return Err(Error::Unsupported(format!(
                    "{} carries a subgraph, a sparse tensor or a type as the attribute \
                     `{}`; §07.3 makes a subgraph a region, and translating ONNX's \
                     scope rules into a region's is a mapping this build has not made",
                    node.label(),
                    a.name
                )));
            }
            if a.kind == AttrType::Tensor {
                return Err(Error::Unsupported(format!(
                    "{} carries the tensor-valued attribute `{}`; a tensor in OMNI is \
                     an object in the table, not a value inside an op",
                    node.label(),
                    a.name
                )));
            }
            attrs.push((a.name.clone(), compat_attr(a)?));
        }
        Ok(Mapped::Compat {
            attrs,
            reason: reason.to_string(),
        })
    };
    let native = |dialect, name, attrs: Vec<(String, Value)>, operands| {
        Ok(Mapped::Native {
            dialect,
            name,
            attrs,
            operands,
        })
    };
    let attr_i = |k: &str| node.attr(k).map(|a| a.i);
    let attr_ints = |k: &str| node.attr(k).map(|a| a.ints.clone());
    let rank = |i: usize| ins.get(i).and_then(|t| t.as_tensor()).map(|(s, _)| s.len());

    // The unattributed one-for-one ops.
    if let Some((_, d, n)) = MAP.iter().find(|(o, _, _)| *o == node.op_type) {
        // MatMul promotes a rank-1 operand and OMNI's matmul does not, so the
        // two ops agree exactly only above rank 1.
        if node.op_type == "MatMul" && (rank(0) == Some(1) || rank(1) == Some(1)) {
            return compat(
                "ONNX MatMul promotes a rank-1 operand to a matrix and removes the \
                 added dimension afterwards; omni.tensor/matmul takes rank ≥ 2, so the \
                 two are the same op only above rank 1",
            );
        }
        return native(d, n, Vec::new(), node.inputs.len());
    }

    match node.op_type.as_str() {
        "Max" | "Min" if node.inputs.len() == 2 => {
            let n = if node.op_type == "Max" {
                "maximum"
            } else {
                "minimum"
            };
            native("omni.tensor", n, Vec::new(), 2)
        }
        "Max" | "Min" => compat(
            "ONNX Max and Min are variadic; omni.tensor's take two operands, and \
             chaining them would be a lowering rather than a translation",
        ),
        "Transpose" => {
            let Some(r) = rank(0) else {
                return compat(
                    "its operand's rank is unknown, and a default `perm` is \
                               the reversal of a rank this import does not have",
                );
            };
            // ONNX's default is the reversed axes; OMNI's `perm` is required, so
            // the default is written out rather than left implicit.
            let perm = attr_ints("perm").unwrap_or_else(|| (0..r as i64).rev().collect());
            native(
                "omni.tensor",
                "transpose",
                vec![("perm".into(), int_array(&perm))],
                1,
            )
        }
        "Concat" => match attr_i("axis") {
            Some(axis) => native(
                "omni.tensor",
                "concat",
                vec![("axis".into(), int_value(axis))],
                node.inputs.len(),
            ),
            None => Err(Error::Malformed(format!(
                "{} has no `axis`, which ONNX requires",
                node.label()
            ))),
        },
        "Softmax" if opset >= 13 => native(
            "omni.tensor",
            "softmax",
            vec![("axis".into(), int_value(attr_i("axis").unwrap_or(-1)))],
            1,
        ),
        "Softmax" => compat(
            "before opset 13 Softmax flattens its operand to two dimensions and \
             normalizes the second, which is a reshape and a softmax rather than a \
             softmax",
        ),
        "Cast" => {
            let Some(to) = attr_i("to") else {
                return Err(Error::Malformed(format!(
                    "{} has no `to`, which ONNX requires",
                    node.label()
                )));
            };
            if attr_i("saturate") == Some(0) {
                return compat(
                    "`saturate: 0` makes an out-of-range cast produce an infinity \
                     rather than the type's maximum, and §04.3's rounding modes do \
                     not name that behaviour",
                );
            }
            let dtype = dtype_of(to as i32)?;
            native(
                "omni.tensor",
                "cast",
                vec![("dtype".into(), dtype.to_value())],
                1,
            )
        }
        "Gather" => {
            // ONNX Gather takes the indices' shape into the output at `axis`;
            // omni.tensor/gather does the same, and its shape function is
            // written for axis 0.
            let axis = attr_i("axis").unwrap_or(0);
            native(
                "omni.tensor",
                "gather",
                vec![("axis".into(), int_value(axis))],
                2,
            )
        }
        "CumSum" => {
            if attr_i("exclusive").unwrap_or(0) != 0 || attr_i("reverse").unwrap_or(0) != 0 {
                return compat(
                    "`exclusive` and `reverse` change which elements the sum at a \
                     position includes, and omni.tensor/cumsum has neither",
                );
            }
            let Some(axis) = node.inputs.get(1).and_then(|n| const_ints(n)) else {
                return compat(
                    "its axis is an operand rather than a constant, so it is not an \
                     attribute this import can write",
                );
            };
            let [axis] = axis[..] else {
                return Err(Error::Malformed(format!(
                    "{}'s axis is not a single integer",
                    node.label()
                )));
            };
            native(
                "omni.tensor",
                "cumsum",
                vec![("axis".into(), int_value(axis))],
                1,
            )
        }
        "Reshape" => {
            let Some(shape) = node.inputs.get(1).and_then(|n| const_ints(n)) else {
                return compat(
                    "its target shape is an operand rather than a constant; a graph \
                     that computes its own shapes is one OMNI states with a symbolic \
                     dimension, and rewriting it that way is not a translation",
                );
            };
            if shape.contains(&0) && attr_i("allowzero").unwrap_or(0) == 0 {
                return compat(
                    "a 0 in ONNX's target shape means `copy this dimension from the \
                     operand` unless `allowzero` is set, and omni.tensor/reshape reads \
                     it as a dimension of zero",
                );
            }
            native(
                "omni.tensor",
                "reshape",
                vec![("shape".into(), int_array(&shape))],
                1,
            )
        }
        "Expand" => compat(
            "ONNX Expand broadcasts to a shape given as an operand; \
             omni.tensor/broadcast takes it as an attribute, and a computed shape is \
             not one",
        ),
        r if REDUCTIONS.iter().any(|(o, _)| *o == r) => {
            let kind = REDUCTIONS.iter().find(|(o, _)| *o == r).unwrap().1;
            if attr_i("noop_with_empty_axes").unwrap_or(0) != 0 {
                return compat(
                    "`noop_with_empty_axes` makes an empty axis list mean `reduce \
                     nothing` instead of `reduce everything`, and omni.tensor/reduce \
                     requires the axes it reduces",
                );
            }
            // Opset 18 moved `axes` from an attribute to an operand. Both forms
            // say the same thing when the operand is a constant.
            let axes = match attr_ints("axes") {
                Some(a) => Some(a),
                None => match node.inputs.get(1) {
                    Some(n) => const_ints(n),
                    None => rank(0).map(|r| (0..r as i64).collect()),
                },
            };
            let Some(axes) = axes else {
                return compat(
                    "its axes are an operand rather than a constant, and an axis this \
                     import cannot read is one it will not guess",
                );
            };
            let keep = attr_i("keepdims").unwrap_or(1) != 0;
            native(
                "omni.tensor",
                "reduce",
                vec![
                    ("kind".into(), Value::text(kind)),
                    ("axes".into(), int_array(&axes)),
                    ("keepdims".into(), Value::Bool(keep)),
                ],
                1,
            )
        }
        "QuantizeLinear" | "DequantizeLinear" => {
            let scalar_scale = ins
                .get(1)
                .and_then(|t| t.as_tensor())
                .map(|(s, _)| s.iter().all(|d| matches!(d, Dim::N(1))))
                .unwrap_or(false);
            if !scalar_scale || node.attr("block_size").is_some() {
                return compat(
                    "its scale is per-axis or per-block, and stating which elements \
                     share a scale needs §05.1's block shape over an operand whose \
                     shape a dynamic graph does not fix",
                );
            }
            let name = if node.op_type == "QuantizeLinear" {
                "quantize"
            } else {
                "dequantize"
            };
            // §05.1's `formula` is a closed enumeration precisely so that this
            // is not a guess: ONNX defines y = (x - zero_point) * scale.
            let out = if node.op_type == "QuantizeLinear" {
                match ins.get(2).and_then(|t| t.as_tensor()) {
                    Some((_, d)) => d.clone(),
                    None => match attr_i("output_dtype") {
                        Some(c) => dtype_of(c as i32)?,
                        None => DType::U8,
                    },
                }
            } else {
                match ins.get(1).and_then(|t| t.as_tensor()) {
                    Some((_, d)) => d.clone(),
                    None => DType::F32,
                }
            };
            let scheme = Value::map(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", out.to_value()),
            ]);
            native(
                "omni.quant",
                name,
                vec![("scheme".into(), scheme)],
                node.inputs.len(),
            )
        }
        _ => compat("no OMNI op means exactly what it means"),
    }
}

// -------------------------------------------------------------- the importer --

/// What a caller may add that the file does not say. Every field is absent
/// unless given (I1).
#[derive(Clone, Debug)]
pub struct ImportOpts {
    pub name: String,
    pub source_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    pub arch: Option<(String, Vec<(String, Value)>)>,
    /// Whether to translate the graph. A file whose graph this build cannot
    /// carry still has weights worth importing, and `false` says so explicitly
    /// rather than dropping the graph quietly.
    pub graph: bool,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/onnx".into(),
            source_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
            graph: true,
        }
    }
}

/// One initializer, as the container will hold it.
struct Weight {
    name: String,
    shape: Vec<u64>,
    dtype: DType,
    data: Vec<u8>,
    /// A `Constant` node's value, which ONNX keeps inside the graph and OMNI
    /// keeps in the table like any other tensor.
    from_node: bool,
}

/// How one op was spelled in ONNX. OMNI-IR numbers its values, so the names are
/// exactly the information the IR has no room for (I2).
#[derive(Clone, Debug, Default)]
struct OpNames {
    node: String,
    doc: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Inputs folded into attributes: constants the op no longer reads, whose
    /// names an export needs so it can write them back.
    folded: Vec<String>,
}

/// The result of an import.
pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
    /// The translated graph, when one was asked for and produced.
    pub module: Option<Module>,
    /// `(ONNX op, count)` for nodes one OMNI op could express.
    pub native: Vec<(String, usize)>,
    /// `(ONNX op, count, why)` for nodes carried in a compat dialect.
    pub compat: Vec<(String, usize, String)>,
    pub initializers: usize,
    pub opset: i64,
}

impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Imported {{ {} object(s), {} initializer(s), {} native op kind(s), {} \
             carried in a compat dialect }}",
            self.objects.len(),
            self.initializers,
            self.native.len(),
            self.compat.len()
        )
    }
}

/// Reads a constant integer vector out of an initializer, for the ONNX ops that
/// take a shape or an axis list as an operand.
fn ints_of(w: &Weight) -> Option<Vec<i64>> {
    match w.dtype {
        DType::I64 => Some(
            w.data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DType::I32 => Some(
            w.data
                .chunks_exact(4)
                .map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())))
                .collect(),
        ),
        _ => None,
    }
}

/// What translating one graph produced.
struct Built {
    module: Module,
    names: Vec<OpNames>,
    native: Vec<(String, usize)>,
    compat: Vec<(String, usize, String)>,
    notes: Vec<Note>,
}

/// Builds the OMNI-IR module for one ONNX graph.
struct GraphBuilder<'a> {
    g: &'a Graph,
    weights: &'a [Weight],
    opsets: &'a Model,
    /// value name → SSA id
    ids: Vec<(String, u32)>,
    /// SSA id → type
    types: Vec<Type>,
    next: u32,
    ops: Vec<Op>,
    names: Vec<OpNames>,
    /// The types ONNX declares, for the ops OMNI cannot type itself.
    declared: Vec<(String, Type)>,
    native: Vec<(String, usize)>,
    compat: Vec<(String, usize, String)>,
    dialects: Vec<(String, u32)>,
    notes: Vec<Note>,
}

impl<'a> GraphBuilder<'a> {
    fn id_of(&self, name: &str) -> Option<u32> {
        self.ids.iter().find(|(n, _)| n == name).map(|(_, i)| *i)
    }

    fn weight(&self, name: &str) -> Option<&'a Weight> {
        self.weights.iter().find(|w| w.name == name)
    }

    fn ty(&self, id: u32) -> Type {
        self.types[id as usize].clone()
    }

    fn define(&mut self, name: &str, ty: Type) -> u32 {
        let id = self.next;
        self.next += 1;
        self.ids.push((name.to_string(), id));
        self.types.push(ty);
        id
    }

    fn use_dialect(&mut self, ns: &str, version: u32) {
        if !self.dialects.iter().any(|(n, _)| n == ns) {
            self.dialects.push((ns.to_string(), version));
        }
    }

    /// The id of a value, materializing a `core.constant` for an initializer the
    /// first time one is read.
    fn operand(&mut self, name: &str, by: &Node) -> Res<u32> {
        if let Some(id) = self.id_of(name) {
            return Ok(id);
        }
        let Some(w) = self.weight(name) else {
            return Err(Error::Malformed(format!(
                "{} reads `{name}`, which no input, initializer or earlier node \
                 defines; ONNX requires a graph's nodes to be in topological order",
                by.label()
            )));
        };
        let ty = Type::tensor(
            w.shape.iter().map(|d| Dim::N(*d)).collect(),
            w.dtype.clone(),
        );
        let id = self.define(name, ty.clone());
        self.ops.push(
            Op::new("omni.core", "constant", 1)
                .with_output(id, ty)
                .with_attr("tensor", Value::text(name.to_string())),
        );
        self.names.push(OpNames {
            outputs: vec![name.to_string()],
            ..Default::default()
        });
        Ok(id)
    }

    fn declared_type(&self, name: &str) -> Option<Type> {
        self.declared
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.clone())
    }

    /// The result types of an op: OMNI's own shape function where it has one,
    /// ONNX's declaration where it does not — and an error when the two
    /// disagree about a dimension both of them state.
    fn result_types(&self, op: &Op, node: &Node, ins: &[Type]) -> Res<Vec<Type>> {
        let declared: Vec<Option<Type>> =
            node.outputs.iter().map(|n| self.declared_type(n)).collect();
        match ir::infer(op, ins) {
            ir::Inferred::Ill(msg) => Err(Error::Malformed(format!(
                "{}: OMNI's shape function rejects the translation — {msg}",
                node.label()
            ))),
            ir::Inferred::Types(ts) => {
                for (i, t) in ts.iter().enumerate() {
                    let (Some(d), Some((got, gd))) =
                        (declared.get(i).and_then(|x| x.as_ref()), t.as_tensor())
                    else {
                        continue;
                    };
                    let Some((want, wd)) = d.as_tensor() else {
                        continue;
                    };
                    // Two independent shape functions on the same node. A
                    // disagreement about a dimension both of them state is a
                    // finding: one of the two readers is wrong about the model.
                    if want.len() != got.len() {
                        return Err(Error::Malformed(format!(
                            "{}: ONNX declares `{}` with rank {} and OMNI's shape \
                             function computes rank {}",
                            node.label(),
                            node.outputs[i],
                            want.len(),
                            got.len()
                        )));
                    }
                    for (k, (a, b)) in want.iter().zip(got).enumerate() {
                        if let (Dim::N(x), Dim::N(y)) = (a, b) {
                            if x != y {
                                return Err(Error::Malformed(format!(
                                    "{}: ONNX declares axis {k} of `{}` as {x} and \
                                     OMNI's shape function computes {y}",
                                    node.label(),
                                    node.outputs[i]
                                )));
                            }
                        }
                    }
                    if wd != gd {
                        return Err(Error::Malformed(format!(
                            "{}: ONNX declares `{}` as {} and the translation \
                             produces {}",
                            node.label(),
                            node.outputs[i],
                            wd.label(),
                            gd.label()
                        )));
                    }
                }
                Ok(ts)
            }
            ir::Inferred::Unchecked(_) => {
                let mut out = Vec::new();
                for (i, name) in node.outputs.iter().enumerate() {
                    match declared.get(i).and_then(|x| x.clone()) {
                        Some(t) => out.push(t),
                        None => {
                            return Err(Error::Unsupported(format!(
                                "{} produces `{name}`, and neither OMNI's shape \
                                 functions nor the file itself says what its type is. \
                                 §07.3 requires every result to declare one, and a \
                                 declaration invented here would be a claim about \
                                 the model. Run ONNX's own shape inference over the \
                                 file, or import it without its graph",
                                node.label()
                            )))
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    fn count_native(&mut self, op: &str) {
        match self.native.iter_mut().find(|(n, _)| n == op) {
            Some((_, c)) => *c += 1,
            None => self.native.push((op.to_string(), 1)),
        }
    }

    fn count_compat(&mut self, op: &str, why: &str) {
        match self.compat.iter_mut().find(|(n, _, _)| n == op) {
            Some((_, c, _)) => *c += 1,
            None => self.compat.push((op.to_string(), 1, why.to_string())),
        }
    }

    fn build(mut self) -> Res<Built> {
        // Declared types, from every value the file gives one to.
        for vi in self
            .g
            .inputs
            .iter()
            .chain(&self.g.outputs)
            .chain(&self.g.value_info)
        {
            if let Some(p) = &vi.ty {
                if let Ok(t) = ptype(p, &vi.name) {
                    self.declared.push((vi.name.clone(), t));
                }
            }
        }

        // Parameters: the graph inputs that are not also initializers. ONNX
        // before IR version 4 let an initializer appear as an input to mean "a
        // default a caller may override", and OMNI has no such thing: a value
        // the model carries is a constant.
        let mut params: Vec<(String, Type)> = Vec::new();
        for vi in &self.g.inputs {
            if self.weights.iter().any(|w| w.name == vi.name) {
                self.notes.push(Note {
                    item: format!("graph input `{}`", vi.name),
                    reason: "it is also an initializer, which before IR version 4 meant \
                             a default a caller could override"
                        .into(),
                    action: "imported as a constant, not as a parameter".into(),
                });
                continue;
            }
            let p = vi.ty.as_ref().ok_or_else(|| {
                Error::Malformed(format!("the graph input `{}` declares no type", vi.name))
            })?;
            let t = ptype(p, &format!("the graph input `{}`", vi.name))?;
            let id = self.define(&vi.name, t.clone());
            debug_assert_eq!(id as usize, params.len());
            params.push((vi.name.clone(), t));
        }

        for node in &self.g.nodes {
            self.node(node)?;
        }

        // The outputs, named as ONNX named them — which is what `omni.io/output`
        // is for (§07.8) — and then returned, so the function has results a
        // caller can bind.
        let mut results = Vec::new();
        for vi in &self.g.outputs {
            let id = self.id_of(&vi.name).ok_or_else(|| {
                Error::Malformed(format!(
                    "the graph declares the output `{}`, which no node produces",
                    vi.name
                ))
            })?;
            self.use_dialect("omni.io", 1);
            self.ops.push(
                Op::new("omni.io", "output", 1)
                    .with_inputs(&[id])
                    .with_attr("name", Value::text(vi.name.clone())),
            );
            self.names.push(OpNames {
                inputs: vec![vi.name.clone()],
                ..Default::default()
            });
            results.push(self.ty(id));
        }
        let returned: Vec<u32> = self
            .g
            .outputs
            .iter()
            .filter_map(|vi| self.id_of(&vi.name))
            .collect();
        self.ops
            .push(Op::new("omni.core", "return", 1).with_inputs(&returned));
        self.names.push(OpNames::default());

        self.use_dialect("omni.core", 1);
        let mut module = Module::new(Level::Primitive, "main");
        module.dialects = self
            .dialects
            .iter()
            .map(|(ns, v)| ir::DialectUse {
                ns: ns.clone(),
                version: *v,
                reference: None,
            })
            .collect();
        module.dialects.sort_by(|a, b| a.ns.cmp(&b.ns));
        module.functions.push((
            "main".into(),
            Function {
                params,
                results,
                attrs: vec![("kind".into(), Value::text("forward"))],
                body: Region {
                    blocks: vec![Block {
                        args: Vec::new(),
                        ops: self.ops,
                    }],
                },
                constraints: Vec::new(),
            },
        ));
        Ok(Built {
            module,
            names: self.names,
            native: self.native,
            compat: self.compat,
            notes: self.notes,
        })
    }

    fn node(&mut self, node: &Node) -> Res<()> {
        // `Constant` is the one ONNX node that is a tensor rather than a
        // computation. It became an entry in the table before this ran, so what
        // is left is the op that reads it.
        if node.op_type == "Constant" && node.domain.is_empty() {
            let out = node
                .outputs
                .first()
                .ok_or_else(|| Error::Malformed(format!("{} produces nothing", node.label())))?;
            self.operand(out, node)?;
            // `operand` recorded the synthetic constant; give it the node's own
            // spelling so an export writes the node back rather than an
            // initializer.
            if let Some(n) = self.names.last_mut() {
                n.node = node.name.clone();
                n.doc = node.doc_string.clone();
                n.outputs = vec![out.clone()];
                // `Constant` marks this as a node rather than an initializer —
                // ONNX stores the same tensor two ways and the difference is
                // visible in the file — and the second entry is the name the
                // inner TensorProto carried, which OMNI's table has no room for
                // because the table is keyed by the value's name.
                n.folded = vec![
                    "Constant".into(),
                    node.attr("value")
                        .and_then(|a| a.t.as_ref())
                        .map(|t| t.name.clone())
                        .unwrap_or_default(),
                ];
            }
            self.count_native("Constant");
            self.use_dialect("omni.core", 1);
            return Ok(());
        }

        // The operands are *typed* before any of them is materialized, because
        // whether an operand stays an operand is what [`map_node`] decides: a
        // constant folded into an attribute is not read by anything afterwards,
        // and emitting a `core.constant` for it would leave the graph carrying a
        // value nothing uses.
        let mut supplied = Vec::new();
        let mut in_types: Vec<Type> = Vec::new();
        let mut trailing_gap = false;
        for name in &node.inputs {
            if name.is_empty() {
                // ONNX spells "this optional operand was not given" as an empty
                // name. An absent operand in the middle changes which operand
                // every later one is, so it is refused rather than compacted.
                trailing_gap = true;
                continue;
            }
            if trailing_gap {
                return Err(Error::Unsupported(format!(
                    "{} leaves an optional operand empty in the middle of its input \
                     list, and OMNI-IR has no way to say `not this one`",
                    node.label()
                )));
            }
            let t = match self.id_of(name) {
                Some(id) => self.ty(id),
                None => match self.weight(name) {
                    Some(w) => Type::tensor(
                        w.shape.iter().map(|d| Dim::N(*d)).collect(),
                        w.dtype.clone(),
                    ),
                    None => {
                        return Err(Error::Malformed(format!(
                            "{} reads `{name}`, which no input, initializer or earlier \
                             node defines; ONNX requires a graph's nodes to be in \
                             topological order",
                            node.label()
                        )))
                    }
                },
            };
            in_types.push(t);
            supplied.push(name.clone());
        }

        let opset = self.opsets.opset(node.dialect()).ok_or_else(|| {
            Error::Malformed(format!(
                "{} is in the domain `{}`, which the file does not import an opset \
                 for; an op's meaning is the opset's to state",
                node.label(),
                node.dialect()
            ))
        })?;
        let weights = self.weights;
        let const_ints = |n: &str| -> Option<Vec<i64>> {
            weights.iter().find(|w| w.name == n).and_then(ints_of)
        };
        let mapped = map_node(node, opset, &in_types, &const_ints)?;

        // Now the operands that survived, in order: this is where an
        // initializer becomes a `core.constant`.
        let keep = match &mapped {
            Mapped::Native { operands, .. } => (*operands).min(supplied.len()),
            Mapped::Compat { .. } => supplied.len(),
        };
        let mut ins = Vec::with_capacity(keep);
        for name in &supplied[..keep] {
            ins.push(self.operand(name, node)?);
        }
        let in_types = &in_types[..keep];

        let (op, folded) = match mapped {
            Mapped::Native {
                dialect,
                name,
                attrs,
                ..
            } => {
                self.use_dialect(dialect, 1);
                self.count_native(&node.op_type);
                let mut op = Op::new(dialect, name, 1).with_inputs(&ins);
                for (k, v) in attrs {
                    op = op.with_attr(&k, v);
                }
                (op, supplied[keep..].to_vec())
            }
            Mapped::Compat { attrs, reason } => {
                let ns = node.dialect().to_string();
                if !(0..=i64::from(u32::MAX)).contains(&opset) {
                    return malformed(format!("the opset version {opset} is not a version"));
                }
                self.use_dialect(&ns, opset as u32);
                self.count_compat(&node.op_type, &reason);
                let mut op = Op::new(&ns, &node.op_type, opset as u32).with_inputs(&ins);
                for (k, v) in attrs {
                    op = op.with_attr(&k, v);
                }
                (op, Vec::new())
            }
        };

        let mut op = op;
        if !node.name.is_empty() {
            op.loc = Some(node.name.clone());
        }
        // Result ids are allocated before inference so an op that produces two
        // values numbers them in ONNX's order.
        let types = self.result_types(&op, node, in_types)?;
        if types.len() != node.outputs.len() {
            return Err(Error::Unsupported(format!(
                "{} declares {} result(s) and the OMNI op it maps to produces {}",
                node.label(),
                node.outputs.len(),
                types.len()
            )));
        }
        for (name, t) in node.outputs.iter().zip(types) {
            let id = self.define(name, t.clone());
            op = op.with_output(id, t);
        }
        self.ops.push(op);
        self.names.push(OpNames {
            node: node.name.clone(),
            doc: node.doc_string.clone(),
            inputs: supplied[..keep].to_vec(),
            outputs: node.outputs.clone(),
            folded,
        });
        Ok(())
    }
}

// ------------------------------------------------------------- the envelope --

/// The ONNX spellings OMNI has no field for, kept verbatim so an export can put
/// them back (I2).
///
/// This is deliberately *only* the incidental half of the file: producer
/// strings, doc strings, node names, and the names ONNX gives the values
/// OMNI-IR numbers. The ops and their attributes are not here — they are in the
/// graph, and an export that read them from this object would be copying a file
/// rather than translating a model.
fn envelope(m: &Model, names: &[OpNames], weights: &[Weight]) -> Value {
    let pairs = |v: &[(String, String)]| {
        Value::Array(
            v.iter()
                .map(|(k, x)| Value::Array(vec![Value::text(k.clone()), Value::text(x.clone())]))
                .collect(),
        )
    };
    let vinfo = |v: &[ValueInfo]| {
        Value::Array(
            v.iter()
                .map(|x| {
                    Value::Array(vec![
                        Value::text(x.name.clone()),
                        Value::text(x.doc_string.clone()),
                    ])
                })
                .collect(),
        )
    };
    let ops = Value::Array(
        names
            .iter()
            .map(|n| {
                Value::map(vec![
                    ("node", Value::text(n.node.clone())),
                    ("doc", Value::text(n.doc.clone())),
                    (
                        "in",
                        Value::Array(n.inputs.iter().map(|s| Value::text(s.clone())).collect()),
                    ),
                    (
                        "out",
                        Value::Array(n.outputs.iter().map(|s| Value::text(s.clone())).collect()),
                    ),
                    (
                        "folded",
                        Value::Array(n.folded.iter().map(|s| Value::text(s.clone())).collect()),
                    ),
                ])
            })
            .collect(),
    );
    Value::map(vec![
        ("t", Value::text("omni.core/foreign")),
        ("v", Value::U(1)),
        ("format", Value::text("onnx")),
        ("ir_version", Value::U(m.ir_version as u64)),
        ("producer_name", Value::text(m.producer_name.clone())),
        ("producer_version", Value::text(m.producer_version.clone())),
        ("domain", Value::text(m.domain.clone())),
        ("model_version", Value::U(m.model_version as u64)),
        ("doc_string", Value::text(m.doc_string.clone())),
        (
            "opsets",
            Value::Array(
                m.opsets
                    .iter()
                    .map(|(d, v)| Value::Array(vec![Value::text(d.clone()), Value::U(*v as u64)]))
                    .collect(),
            ),
        ),
        ("metadata_props", pairs(&m.metadata)),
        (
            "graph",
            Value::map(vec![
                ("name", Value::text(m.graph.name.clone())),
                ("doc_string", Value::text(m.graph.doc_string.clone())),
                ("inputs", vinfo(&m.graph.inputs)),
                ("outputs", vinfo(&m.graph.outputs)),
                ("value_info", vinfo(&m.graph.value_info)),
                (
                    "initializers",
                    Value::Array(
                        weights
                            .iter()
                            .filter(|w| !w.from_node)
                            .map(|w| Value::text(w.name.clone()))
                            .collect(),
                    ),
                ),
            ]),
        ),
        ("ops", ops),
    ])
}

/// Assembles the object graph: tensors, the module, the report and the
/// envelope. Runs twice, for the reason [`crate::safetensors`] runs its own
/// twice — the report counts what verification checked, and verification needs
/// a graph to check against.
#[allow(clippy::too_many_arguments)]
fn assemble(
    weights: &[Weight],
    module: &Option<Module>,
    envelope: &Value,
    opts: &ImportOpts,
    report: &Fidelity,
) -> (Vec<Object>, Digest) {
    let mut b = ModelBuilder::new(opts.name.clone())
        .hash(opts.hash)
        .chunk_size(opts.chunk_size);
    if let Some(spdx) = &opts.license {
        b = b.license(spdx.clone());
    }
    if let Some((family, params)) = &opts.arch {
        b = b.arch(
            family.clone(),
            params
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect(),
        );
    }
    for w in weights {
        b = b.tensor(TensorSpec {
            name: w.name.clone(),
            shape: w.shape.clone(),
            dtype: w.dtype.clone(),
            axes: None,
            // ONNX says nothing about what a tensor *is*: an initializer is a
            // value a node reads, and calling them all weights would be a guess
            // §04.2 does not need made.
            semantic: "",
            data: w.data.clone(),
            layout: Some(layout_of(&w.dtype)),
        });
    }
    if let Some(m) = module {
        b = b.graph(m.clone(), Vec::new());
    }
    b = b.asset("provenance", otype::PROVENANCE, report.to_value());

    let foreign = Object::structure(otype::FOREIGN, envelope);
    let d = foreign.digest(opts.hash);
    b = b.manifest_key(
        "foreign",
        Value::Array(vec![Value::Array(vec![
            Value::U(otype::FOREIGN as u64),
            Value::Bytes(d.to_vec()),
        ])]),
    );
    let (mut objects, root) = b.build();
    objects.push(foreign);
    (objects, root)
}

/// Imports an ONNX file into an OMNI object graph.
pub fn import(bytes: &[u8], opts: &ImportOpts, ext: &dyn External) -> Res<Imported> {
    let m = Model::parse(bytes)?;
    let mut report = Fidelity {
        format: "onnx",
        importer: IMPORTER,
        source_path: opts.source_path.clone(),
        source_digest: opts.hash.digest(bytes),
        source_size: bytes.len() as u64,
        lossless: false,
        represented: vec!["tensors".into(), "dtypes".into(), "shapes".into()],
        ..Default::default()
    };

    // The weights: initializers first, in the file's order, then the tensors
    // `Constant` nodes carry inside the graph.
    let mut weights: Vec<Weight> = Vec::new();
    for t in &m.graph.initializers {
        if t.name.is_empty() {
            return malformed("an initializer has no name, so nothing can refer to it");
        }
        if weights.iter().any(|w| w.name == t.name) {
            return malformed(format!("the initializer `{}` appears twice", t.name));
        }
        if t.has_segment {
            return Err(Error::Unsupported(format!(
                "`{}` is a segmented tensor: ONNX's `segment` splits one tensor over \
                 several TensorProtos and this build has never seen a writer emit one",
                t.name
            )));
        }
        weights.push(Weight {
            name: t.name.clone(),
            shape: t.dims.clone(),
            dtype: dtype_of(t.data_type)?,
            data: tensor_bytes(t, ext)?,
            from_node: false,
        });
    }
    if opts.graph {
        for node in &m.graph.nodes {
            if node.op_type != "Constant" || !node.domain.is_empty() {
                continue;
            }
            let out = node
                .outputs
                .first()
                .ok_or_else(|| Error::Malformed(format!("{} produces nothing", node.label())))?;
            let Some(a) = node.attr("value") else {
                return Err(Error::Unsupported(format!(
                    "{}: this build reads a Constant's `value` tensor; the \
                     `value_float`, `value_ints` and `sparse_value` forms each state \
                     the same thing differently and are refused rather than \
                     half-read",
                    node.label()
                )));
            };
            let t = a.t.as_ref().ok_or_else(|| {
                Error::Malformed(format!("{}'s `value` is not a tensor", node.label()))
            })?;
            if weights.iter().any(|w| w.name == *out) {
                return malformed(format!(
                    "`{out}` is both an initializer and a Constant node's output"
                ));
            }
            weights.push(Weight {
                name: out.clone(),
                shape: t.dims.clone(),
                dtype: dtype_of(t.data_type)?,
                data: tensor_bytes(t, ext)?,
                from_node: true,
            });
        }
    }

    // I1: what the file does not state is not written.
    report.assumptions.push(Note {
        item: "license".into(),
        reason: "ONNX has no license field".into(),
        action: match &opts.license {
            Some(spdx) => format!("supplied by the caller as `{spdx}`"),
            None => "field omitted".into(),
        },
    });
    report.assumptions.push(Note {
        item: "arch.family".into(),
        reason: "an ONNX graph is primitives; the architecture that produced it is \
                 not recorded anywhere in the file"
            .into(),
        action: match &opts.arch {
            Some((family, _)) => format!("supplied by the caller as `{family}`"),
            None => "field omitted".into(),
        },
    });

    // I2/I3: what the file has and OMNI does not model, named one at a time.
    if m.training_infos > 0 {
        return Err(Error::Unsupported(format!(
            "the file carries {} TrainingInfoProto: an initialization graph and an \
             update graph are two more graphs, and §09 states training state rather \
             than the computation that produces it",
            m.training_infos
        )));
    }
    if m.functions > 0 {
        return Err(Error::Unsupported(format!(
            "the file defines {} local function(s): a function whose attributes come \
             from its callers is a template, and §07.3's functions are not",
            m.functions
        )));
    }
    if m.graph.sparse_initializers > 0 {
        return Err(Error::Unsupported(format!(
            "the graph has {} sparse initializer(s): §04.6 has a sparsity catalogue \
             and ONNX's COO form maps onto it, but this build has not written that \
             mapping and will not approximate it",
            m.graph.sparse_initializers
        )));
    }
    if m.graph.quantization_annotations > 0 {
        report.unrepresented.push(Note {
            item: "quantization_annotation".into(),
            reason: "it names the scale and zero-point tensors of a value the graph \
                     quantizes, which §05 states in the expression instead"
                .into(),
            action: "dropped; the QDQ nodes themselves are imported".into(),
        });
    }
    if !m.metadata.is_empty() {
        report.unrepresented.push(Note {
            item: "metadata_props".into(),
            reason: "a free-form string map with no OMNI schema".into(),
            action: "preserved verbatim in a Foreign object".into(),
        });
    }

    // The graph.
    let mut module = None;
    let mut names: Vec<OpNames> = Vec::new();
    let mut native: Vec<(String, usize)> = Vec::new();
    let mut compat: Vec<(String, usize, String)> = Vec::new();
    if opts.graph {
        let b = GraphBuilder {
            g: &m.graph,
            weights: &weights,
            opsets: &m,
            ids: Vec::new(),
            types: Vec::new(),
            next: 0,
            ops: Vec::new(),
            names: Vec::new(),
            declared: Vec::new(),
            native: Vec::new(),
            compat: Vec::new(),
            dialects: Vec::new(),
            notes: Vec::new(),
        };
        let built = b.build()?;
        report
            .represented
            .push("graph (§07, primitive level)".into());
        for (op, count, why) in &built.compat {
            report.unrepresented.push(Note {
                item: format!("{count} × `{op}`"),
                reason: why.clone(),
                action: format!(
                    "carried in the `{}` compat dialect with its attributes",
                    AI_ONNX
                ),
            });
        }
        report.assumptions.extend(built.notes);
        module = Some(built.module);
        names = built.names;
        native = built.native;
        compat = built.compat;
    } else {
        report.unrepresented.push(Note {
            item: "graph".into(),
            reason: "the caller asked for the weights only".into(),
            action: "not imported".into(),
        });
    }

    let env = envelope(&m, &names, &weights);

    // I4: every initializer is read back out of the graph this import built and
    // compared with the bytes the file held.
    let (probe, probe_root) = assemble(&weights, &module, &env, opts, &report);
    let store = store_of(&probe, opts.hash);
    let ctx = Ctx::new(&store);
    let table = table_of(&probe, &probe_root, opts.hash)?;
    for w in &weights {
        let r = table
            .tensors
            .get(&w.name)
            .ok_or_else(|| Error::Core(format!("`{}` did not reach the table", w.name)))?;
        let desc = TensorDesc::from_value(
            &crate::cbor::decode(
                &crate::store::Store::resolve(&store, &r.1)
                    .map_err(|e| Error::Core(e.to_string()))?
                    .ok_or_else(|| Error::Core("a descriptor went missing".into()))?,
            )
            .map_err(|e| Error::Core(e.to_string()))?,
        )
        .map_err(|e| Error::Core(e.to_string()))?;
        let got = materialize(&desc, &ctx)?;
        if got != w.data {
            return Err(Error::Core(format!(
                "I4: `{}` did not survive import byte for byte ({} bytes in, {} out)",
                w.name,
                w.data.len(),
                got.len()
            )));
        }
        report.verified_tensors += 1;
        report.verified_bytes += w.data.len() as u64;
    }
    // Losslessness is a finding, not an intention: the weights survive, and the
    // graph does only when every node was one OMNI op could state.
    report.lossless = opts.graph && compat.is_empty() && report.unrepresented.is_empty();

    let opset = m.opset(AI_ONNX).unwrap_or(0);
    let (objects, root) = assemble(&weights, &module, &env, opts, &report);
    Ok(Imported {
        objects,
        root,
        report,
        module,
        native,
        compat,
        initializers: weights.len(),
        opset,
    })
}

/// A store over a freshly built object list, for reading the graph back.
fn store_of(objects: &[Object], hash: HashAlgo) -> crate::store::MemoryStore {
    let mut store = crate::store::MemoryStore::new(hash);
    for o in objects {
        let _ = crate::store::WritableStore::put(&mut store, &o.payload);
    }
    store
}

/// Walks manifest → model → table in a freshly built graph.
fn table_of(objects: &[Object], root: &Digest, hash: HashAlgo) -> Res<TensorTable> {
    let find = |d: &Digest| {
        objects
            .iter()
            .find(|o| &o.digest(hash) == d)
            .map(|o| o.payload.clone())
    };
    let decode = |d: &Digest| -> Res<Value> {
        let bytes = find(d).ok_or_else(|| Error::Core("a just-built object is missing".into()))?;
        crate::cbor::decode(&bytes).map_err(|e| Error::Core(e.to_string()))
    };
    let manifest = decode(root)?;
    let model_ref = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(|r| crate::expr::parse_ref_value(r).ok())
        .ok_or_else(|| Error::Core("no model asset".into()))?;
    let model = decode(&model_ref.1)?;
    let table_ref = model
        .get("tensors")
        .and_then(|r| crate::expr::parse_ref_value(r).ok())
        .ok_or_else(|| Error::Core("no tensor table".into()))?;
    TensorTable::from_value(&decode(&table_ref.1)?).map_err(|e| Error::Core(e.to_string()))
}

/// The stored bytes of a tensor, in the arrangement ONNX wants.
fn materialize(desc: &TensorDesc, ctx: &Ctx<'_>) -> Res<Vec<u8>> {
    let shape = concrete(&desc.shape).ok_or_else(|| {
        Error::Unsupported(
            "a symbolic shape has to be bound before it can be written to a flat \
             buffer (§04.7.3)"
                .into(),
        )
    })?;
    let want = stored_bytes(&desc.dtype, &shape);
    if let Expr::Literal { chunks, .. } = &desc.value {
        if desc.layout == Layout::row_major() || desc.layout == layout_of(&desc.dtype) {
            let bytes = ctx
                .chunk_bytes(chunks)
                .map_err(|e| Error::Core(e.to_string()))?;
            if bytes.len() as u64 >= want {
                return Ok(bytes[..want as usize].to_vec());
            }
        }
    }
    // Anything else — a quantized weight, a LoRA-merged one — is evaluated and
    // re-encoded, which is what lets an export write a tensor OMNI stores as an
    // expression.
    let t = desc
        .value
        .eval(ctx)
        .map_err(|e| Error::Core(e.to_string()))?;
    let mut out = vec![0u8; want as usize];
    for (i, x) in t.data.iter().enumerate() {
        if !write_element(&desc.dtype, &mut out, i as u64, *x) {
            return Err(Error::Unsupported(format!(
                "dtype `{}` has no element encoding, so it cannot be written to a \
                 flat buffer",
                desc.dtype
                    .alias()
                    .unwrap_or("(a type with no registered alias)")
            )));
        }
    }
    Ok(out)
}

/// Writes element `i` the way ONNX arranges it: one byte for a `bool`, the
/// dtype's own packing otherwise.
fn write_element(d: &DType, out: &mut [u8], i: u64, x: f64) -> bool {
    if d == &DType::Bool {
        return match out.get_mut(i as usize) {
            Some(b) => {
                *b = u8::from(x != 0.0);
                true
            }
            None => false,
        };
    }
    d.encode(out, i, x, crate::dtype::Round::Rne)
}

// -------------------------------------------------------------- the exporter --

/// One thing an export would lose (§5.1).
#[derive(Clone, Debug)]
pub struct Loss {
    pub item: String,
    pub reason: String,
}

/// What an export would produce and what it would cost, computed without
/// writing anything (E1).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// `(name, ONNX type name, shape, bytes)` for every initializer.
    pub initializers: Vec<(String, &'static str, Vec<u64>, u64)>,
    /// `(op_type, domain, count)` for the nodes this export would write.
    pub nodes: Vec<(String, String, usize)>,
    /// Ops with no ONNX spelling, named individually. An export with any of
    /// these does not happen: §5.2 says an unmapped op fails with a precise
    /// list, and unlike lost metadata there is nothing to consent to.
    pub unmapped: Vec<(String, usize)>,
    pub loss: Vec<Loss>,
    pub bytes: u64,
    pub opset: i64,
    /// Whether the ONNX spellings came from a file this container was imported
    /// from, or were composed here.
    pub from_onnx: bool,
}

impl Plan {
    pub fn lossless(&self) -> bool {
        self.loss.is_empty()
    }

    pub fn writable(&self) -> bool {
        self.unmapped.is_empty()
    }

    /// The loss report of §5.1, as JSON, for writing beside the artifact (E3).
    pub fn loss_report(&self, source: &Digest, hash: HashAlgo) -> String {
        use crate::json;
        json::object(vec![
            ("target", json::string("onnx")),
            (
                "source",
                json::object(vec![
                    ("format", json::string("omni")),
                    (
                        "digest",
                        json::string(format!("{}:{}", hash.prefix(), crate::sha256::hex(source))),
                    ),
                ]),
            ),
            ("opset", json::Value::U(self.opset as u64)),
            ("lossless", json::Value::Bool(self.lossless())),
            (
                "initializers",
                json::Value::U(self.initializers.len() as u64),
            ),
            (
                "nodes",
                json::Value::U(self.nodes.iter().map(|(_, _, c)| *c as u64).sum()),
            ),
            ("bytes", json::Value::U(self.bytes)),
            (
                "unmapped",
                json::Value::Array(
                    self.unmapped
                        .iter()
                        .map(|(op, n)| {
                            json::object(vec![
                                ("op", json::string(op.clone())),
                                ("count", json::Value::U(*n as u64)),
                            ])
                        })
                        .collect(),
                ),
            ),
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

/// The ONNX spellings an import preserved, when this container came from one.
#[derive(Clone, Debug, Default)]
pub struct Envelope {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub opsets: Vec<(String, i64)>,
    pub metadata: Vec<(String, String)>,
    pub graph_name: String,
    pub graph_doc: String,
    pub inputs: Vec<(String, String)>,
    pub outputs: Vec<(String, String)>,
    pub value_info: Vec<(String, String)>,
    pub initializers: Vec<String>,
    pub ops: Vec<OpNamesOwned>,
}

/// One op's ONNX spelling, as [`Envelope`] carries it.
#[derive(Clone, Debug, Default)]
pub struct OpNamesOwned {
    pub node: String,
    pub doc: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub folded: Vec<String>,
}

/// Reads the envelope an ONNX import left in the container's `foreign` list.
pub fn preserved(ctx: &Ctx<'_>, manifest: &Value) -> Res<Option<Envelope>> {
    let Some(list) = manifest.get("foreign").and_then(|v| v.as_array()) else {
        return Ok(None);
    };
    for item in list {
        let Ok(r) = crate::expr::parse_ref_value(item) else {
            continue;
        };
        let Ok(v) = ctx.value(&r.1) else { continue };
        if v.get("format").and_then(|x| x.as_str()) != Some("onnx") {
            continue;
        }
        let text = |m: &Value, k: &str| {
            m.get(k)
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let int = |m: &Value, k: &str| m.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as i64;
        let pairs = |m: &Value, k: &str| -> Vec<(String, String)> {
            m.get(k)
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|p| {
                            let p = p.as_array()?;
                            Some((
                                p.first()?.as_str()?.to_string(),
                                p.get(1)?.as_str()?.to_string(),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let strings = |m: &Value, k: &str| -> Vec<String> {
            m.get(k)
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default()
        };
        let graph = v
            .get("graph")
            .cloned()
            .unwrap_or_else(|| Value::Map(Vec::new()));
        let ops = v
            .get("ops")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|o| OpNamesOwned {
                        node: text(o, "node"),
                        doc: text(o, "doc"),
                        inputs: strings(o, "in"),
                        outputs: strings(o, "out"),
                        folded: strings(o, "folded"),
                    })
                    .collect()
            })
            .unwrap_or_default();
        return Ok(Some(Envelope {
            ir_version: int(&v, "ir_version"),
            producer_name: text(&v, "producer_name"),
            producer_version: text(&v, "producer_version"),
            domain: text(&v, "domain"),
            model_version: int(&v, "model_version"),
            doc_string: text(&v, "doc_string"),
            opsets: v
                .get("opsets")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|p| {
                            let p = p.as_array()?;
                            Some((p.first()?.as_str()?.to_string(), p.get(1)?.as_u64()? as i64))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            metadata: pairs(&v, "metadata_props"),
            graph_name: text(&graph, "name"),
            graph_doc: text(&graph, "doc_string"),
            inputs: pairs(&graph, "inputs"),
            outputs: pairs(&graph, "outputs"),
            value_info: pairs(&graph, "value_info"),
            initializers: strings(&graph, "initializers"),
            ops,
        }));
    }
    Ok(None)
}

/// The ONNX op an OMNI op maps back to, or `None` when the format has no
/// spelling for it.
///
/// The reverse of [`map_node`], and deliberately a *table lookup* rather than a
/// pattern match over the graph: an export that recognised `maximum(x, 0)` as
/// `Relu` would be doing the thing §07.1 criticises ONNX's backends for having
/// to do.
fn onnx_op(op: &Op) -> Option<&'static str> {
    if let Some((o, _, _)) = MAP
        .iter()
        .find(|(_, d, n)| *d == op.dialect && *n == op.name)
    {
        return Some(o);
    }
    Some(match (op.dialect.as_str(), op.name.as_str()) {
        ("omni.tensor", "maximum") => "Max",
        ("omni.tensor", "minimum") => "Min",
        ("omni.tensor", "transpose") => "Transpose",
        ("omni.tensor", "concat") => "Concat",
        ("omni.tensor", "softmax") => "Softmax",
        ("omni.tensor", "cast") => "Cast",
        ("omni.tensor", "gather") => "Gather",
        ("omni.tensor", "cumsum") => "CumSum",
        ("omni.tensor", "reshape") => "Reshape",
        ("omni.tensor", "reduce") => {
            let kind = op.attr("kind").and_then(|v| v.as_str())?;
            REDUCTIONS
                .iter()
                .find(|(_, k)| *k == kind)
                .map(|(o, _)| *o)?
        }
        ("omni.quant", "quantize") => "QuantizeLinear",
        ("omni.quant", "dequantize") => "DequantizeLinear",
        _ => return None,
    })
}

/// The inner tensor name of a `Constant` node, when this op was one.
///
/// ONNX stores a constant either as an initializer or as a `Constant` node
/// carrying a tensor attribute, and both import to the same `core.constant`.
/// The difference is visible in the file, so the envelope records it rather
/// than letting an export pick one.
fn constant_node<'a>(names: Option<&'a [OpNamesOwned]>, i: usize, op: &Op) -> Option<&'a str> {
    if op.dialect != "omni.core" || op.name != "constant" {
        return None;
    }
    let n = names?.get(i)?;
    if n.folded.first().map(String::as_str) != Some("Constant") {
        return None;
    }
    Some(n.folded.get(1).map(String::as_str).unwrap_or(""))
}

/// The preserved spellings, when they still line up with the graph they were
/// recorded for. A container whose graph has been rewritten since is named by
/// this export rather than by its source, and [`Plan::from_onnx`] says so.
fn aligned<'a>(env: &'a Option<Envelope>, ops: &[Op]) -> Option<&'a [OpNamesOwned]> {
    env.as_ref()
        .map(|e| e.ops.as_slice())
        .filter(|o| o.len() == ops.len())
}

/// Whether an op is one an export handles without writing a node: constants
/// become initializers and terminators become the graph's own output list.
fn structural(op: &Op) -> bool {
    matches!(
        (op.dialect.as_str(), op.name.as_str()),
        ("omni.core", "constant") | ("omni.core", "return") | ("omni.io", "output")
    )
}

/// The entry function of a module, which is the only one an ONNX graph can be.
fn entry(m: &Module) -> Res<&Function> {
    if m.functions.len() > 1 {
        return Err(Error::Unsupported(format!(
            "the module defines {} functions; an ONNX graph is one, and inlining the \
             rest would be a rewrite rather than an export",
            m.functions.len()
        )));
    }
    m.function(&m.entry)
        .ok_or_else(|| Error::Malformed(format!("the module names no function `{}`", m.entry)))
}

/// The ops of the entry function's single block.
fn body(f: &Function) -> Res<&[Op]> {
    let blocks = &f.body.blocks;
    if blocks.len() != 1 {
        return Err(Error::Unsupported(format!(
            "the entry function's body has {} blocks; ONNX graphs are one",
            blocks.len()
        )));
    }
    Ok(&blocks[0].ops)
}

/// E1: what an export would write, and what it would lose, with nothing
/// written.
pub fn plan(
    ctx: &Ctx<'_>,
    manifest: &Value,
    table: &TensorTable,
    module: Option<&Module>,
) -> Res<Plan> {
    let mut p = Plan::default();
    let env = preserved(ctx, manifest)?;
    p.from_onnx = env.is_some();
    p.opset = env
        .as_ref()
        .and_then(|e| {
            e.opsets
                .iter()
                .find(|(d, _)| d.is_empty() || d == AI_ONNX)
                .map(|(_, v)| *v)
        })
        .unwrap_or(DEFAULT_OPSET);

    // §5.2: ONNX requires a graph. A weights-only container is a perfectly good
    // OMNI model and not an ONNX one, and saying so is more use than writing a
    // graph with no nodes.
    let Some(m) = module else {
        return Err(Error::Unsupported(
            "this container carries no execution graph, and an ONNX file is a graph. \
             §07.5 makes the graph optional in OMNI precisely because most models \
             ship without one; `omni graph synthesize` builds one for a registered \
             architecture"
                .into(),
        ));
    };
    if m.level != Level::Primitive {
        return Err(Error::Unsupported(format!(
            "the graph is at the `{}` level and ONNX's opset is primitives. §07.2 \
             makes that a lowering, and `omni graph lower` is where it happens — an \
             export that lowered silently would be choosing an abstraction level on \
             the model's behalf",
            m.level.name()
        )));
    }
    let f = entry(m)?;
    let ops = body(f)?;

    // The nodes.
    let count = |list: &mut Vec<(String, String, usize)>, op: &str, dom: &str| match list
        .iter_mut()
        .find(|(o, d, _)| o == op && d == dom)
    {
        Some((_, _, c)) => *c += 1,
        None => list.push((op.to_string(), dom.to_string(), 1)),
    };
    let names = aligned(&env, ops);
    for (i, op) in ops.iter().enumerate() {
        if constant_node(names, i, op).is_some() {
            count(&mut p.nodes, "Constant", "");
            continue;
        }
        if structural(op) {
            continue;
        }
        match onnx_op(op) {
            Some(name) => count(&mut p.nodes, name, ""),
            None if ir::dialect(&op.dialect).is_none() => {
                // A compat-dialect op is an ONNX op that never stopped being
                // one, and it exports as itself.
                let domain = if op.dialect == AI_ONNX {
                    String::new()
                } else {
                    op.dialect.clone()
                };
                count(&mut p.nodes, &op.name, &domain);
            }
            None => match p.unmapped.iter_mut().find(|(o, _)| *o == op.qualified()) {
                Some((_, c)) => *c += 1,
                None => p.unmapped.push((op.qualified(), 1)),
            },
        }
    }

    // The initializers: every tensor the graph reads by name, plus the ones the
    // source file carried and nothing reads.
    let mut initializers: Vec<String> = Vec::new();
    if let Some(e) = &env {
        initializers.extend(e.initializers.iter().cloned());
    }
    for (i, op) in ops.iter().enumerate() {
        if constant_node(names, i, op).is_some() {
            continue;
        }
        if let Some(n) = op.attr("tensor").and_then(|v| v.as_str()) {
            if !initializers.iter().any(|x| x == n) {
                initializers.push(n.to_string());
            }
        }
    }
    for name in initializers {
        let Some(r) = table.tensors.get(&name) else {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: "the graph names it and the table does not hold it".into(),
            });
            continue;
        };
        let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
        let Some(shape) = concrete(&desc.shape) else {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: "a symbolic shape has no fixed extent to write".into(),
            });
            continue;
        };
        match code_of(&desc.dtype) {
            Some(c) => {
                let bytes = stored_bytes(&desc.dtype, &shape);
                p.bytes += bytes;
                p.initializers
                    .push((name.clone(), dtype_name(c), shape, bytes));
            }
            None => {
                let bytes = DType::F32.packed_bytes(shape.iter().product());
                p.bytes += bytes;
                p.initializers.push((name.clone(), "FLOAT", shape, bytes));
                p.loss.push(Loss {
                    item: format!("tensor `{name}` dtype"),
                    reason: format!(
                        "`{}` has no ONNX equivalent; it would be written as FLOAT",
                        desc.dtype
                            .alias()
                            .unwrap_or("(a type with no registered alias)")
                    ),
                });
            }
        }
        if !matches!(desc.value, Expr::Literal { .. }) {
            p.loss.push(Loss {
                item: format!("tensor `{name}` structure"),
                reason: "its value is an expression; ONNX stores materialized bytes, \
                         so the derivation is lost even though the values are not"
                    .into(),
            });
        }
    }

    // Everything ONNX has no room for, named one at a time rather than
    // summarised.
    let assets = manifest.get("assets");
    for (slot, what) in [
        ("tokenizer", "the tokenizer (§06.7)"),
        ("chat_template", "the chat template (§06.9)"),
        ("provenance", "provenance and the import history (§06.4)"),
        ("signatures", "the signatures (§12.5)"),
    ] {
        if assets.and_then(|a| a.get(slot)).is_some() {
            p.loss.push(Loss {
                item: slot.to_string(),
                reason: format!("{what} has no ONNX representation"),
            });
        }
    }
    if manifest.get("parents").is_some() {
        p.loss.push(Loss {
            item: "parents[]".into(),
            reason: "the delta chain (§01.7) has no ONNX representation".into(),
        });
    }
    if !m.rewrites.is_empty() {
        p.loss.push(Loss {
            item: "rewrites".into(),
            reason: format!(
                "{} shipped lowering(s) (§07.7): the one thing ONNX's frozen opset \
                 cannot carry, since a rule that rewrites an unknown op is what makes \
                 an unknown op survivable",
                m.rewrites.len()
            ),
        });
    }
    // A named axis is information about what a dimension *is*, and an ONNX
    // tensor has none.
    let axes: usize = table
        .tensors
        .values()
        .filter(|r| {
            TensorDesc::load(ctx, r)
                .map(|d| d.axes.is_some())
                .unwrap_or(false)
        })
        .count();
    if axes > 0 {
        p.loss.push(Loss {
            item: "tensor axis names".into(),
            reason: format!("{axes} tensor(s) name their axes (§04.2); ONNX does not"),
        });
    }
    Ok(p)
}

/// Writes a `TensorProto`. Always as `raw_data`: it is what every ONNX writer
/// emits for a tensor of any size, and picking one encoding means an export of
/// an import differs from its source only where the source used the other one.
fn write_tensor(name: &str, dtype: &DType, shape: &[u64], data: &[u8]) -> Res<Writer> {
    let code = code_of(dtype).ok_or_else(|| {
        Error::Unsupported(format!(
            "`{name}` has dtype `{}`, which ONNX does not name",
            dtype.alias().unwrap_or("(unnamed)")
        ))
    })?;
    let mut w = Writer::new();
    w.packed_ints(1, &shape.iter().map(|d| *d as i64).collect::<Vec<i64>>());
    w.int(2, i64::from(code));
    w.text(8, name);
    w.bytes(9, data);
    Ok(w)
}

fn write_type(t: &Type) -> Res<Writer> {
    let (shape, dtype) = t.as_tensor().ok_or_else(|| {
        Error::Unsupported(format!(
            "the type {} is not a tensor; ONNX's graph inputs and outputs are",
            t.print()
        ))
    })?;
    let code = code_of(dtype).ok_or_else(|| {
        Error::Unsupported(format!(
            "dtype `{}` has no ONNX equivalent",
            dtype.alias().unwrap_or("(unnamed)")
        ))
    })?;
    let mut dims = Writer::new();
    for d in shape {
        let mut dim = Writer::new();
        match d {
            Dim::N(n) => dim.int_always(1, *n as i64),
            Dim::Sym(s) => dim.text(2, s),
            Dim::Dynamic => {}
        }
        dims.message(1, dim);
    }
    let mut tensor = Writer::new();
    tensor.int(1, i64::from(code));
    tensor.message(2, dims);
    let mut ty = Writer::new();
    ty.message(1, tensor);
    Ok(ty)
}

fn write_value_info(name: &str, doc: &str, t: Option<&Type>) -> Res<Writer> {
    let mut w = Writer::new();
    w.text(1, name);
    if let Some(t) = t {
        w.message(2, write_type(t)?);
    }
    w.text(3, doc);
    Ok(w)
}

/// One attribute, written back into the proto field it came from.
fn write_attr(name: &str, v: &Value) -> Res<Writer> {
    let mut w = Writer::new();
    w.text(1, name);
    let kind = if let Some(f) = v.get("f").and_then(as_f64) {
        w.f32(2, f as f32);
        AttrType::Float
    } else if let Some(i) = as_int(v.get("i")) {
        w.int(3, i);
        AttrType::Int
    } else if let Some(s) = v.get("s").and_then(|x| x.as_str()) {
        w.bytes(4, s.as_bytes());
        AttrType::String
    } else if let Some(b) = v.get("b").and_then(|x| x.as_bytes()) {
        w.bytes(4, b);
        AttrType::String
    } else if let Some(a) = v.get("floats").and_then(|x| x.as_array()) {
        let f: Vec<f32> = a.iter().filter_map(as_f64).map(|x| x as f32).collect();
        w.packed_f32(7, &f);
        AttrType::Floats
    } else if let Some(a) = v.get("ints").and_then(|x| x.as_array()) {
        let ints: Vec<i64> = a.iter().filter_map(|x| as_int(Some(x))).collect();
        w.packed_ints(8, &ints);
        AttrType::Ints
    } else if let Some(a) = v.get("strings").and_then(|x| x.as_array()) {
        for s in a {
            match (
                s.get("s").and_then(|x| x.as_str()),
                s.get("b").and_then(|x| x.as_bytes()),
            ) {
                (Some(t), _) => w.raw_bytes(9, t.as_bytes()),
                (_, Some(b)) => w.raw_bytes(9, b),
                _ => return Err(Error::Core("a string attribute member is neither".into())),
            }
        }
        AttrType::Strings
    } else {
        return Err(Error::Core(format!(
            "the attribute `{name}` is in no form this export knows how to write"
        )));
    };
    w.int(20, kind.code());
    Ok(w)
}

/// A CBOR number as a float, whichever of the three number forms it took.
fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::U(n) => Some(*n as f64),
        Value::I(n) => Some(*n as f64),
        _ => None,
    }
}

fn as_int(v: Option<&Value>) -> Option<i64> {
    match v {
        Some(Value::U(n)) => Some(*n as i64),
        Some(Value::I(n)) => Some(*n),
        _ => None,
    }
}

fn ints_attr(op: &Op, key: &str) -> Res<Vec<i64>> {
    match op.attr(key) {
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                as_int(Some(x)).ok_or_else(|| Error::Core(format!("`{key}` holds a non-integer")))
            })
            .collect(),
        _ => Err(Error::Core(format!(
            "{} has no `{key}`, which its own dialect requires",
            op.qualified()
        ))),
    }
}

/// A synthesized initializer: the constant an ONNX op takes as an operand where
/// the OMNI op took it as an attribute.
struct Folded {
    name: String,
    values: Vec<i64>,
}

/// The nodes and attributes one OMNI op becomes.
struct NodeOut {
    op_type: String,
    domain: String,
    extra_inputs: Vec<String>,
    attrs: Vec<Writer>,
    folded: Vec<Folded>,
}

/// Turns one OMNI op back into an ONNX node's op type, attributes and any
/// operands ONNX takes where OMNI took an attribute.
fn node_of(op: &Op, opset: i64, names: Option<&OpNamesOwned>, seq: usize) -> Res<NodeOut> {
    let mut out = NodeOut {
        op_type: String::new(),
        domain: String::new(),
        extra_inputs: Vec::new(),
        attrs: Vec::new(),
        folded: Vec::new(),
    };
    // A folded operand keeps the name it had when it was imported, and gets a
    // generated one when this container never came from ONNX.
    let folded_name = |i: usize, suffix: &str| -> String {
        match names.and_then(|n| n.folded.get(i)) {
            Some(n) if !n.is_empty() => n.clone(),
            _ => format!("omni_{seq}_{suffix}"),
        }
    };

    if let Some(o) = onnx_op(op) {
        out.op_type = o.to_string();
        match (op.dialect.as_str(), op.name.as_str()) {
            ("omni.tensor", "transpose") => {
                out.attrs.push(write_attr(
                    "perm",
                    &Value::map(vec![(
                        "ints",
                        op.attr("perm").cloned().unwrap_or(Value::Array(vec![])),
                    )]),
                )?);
            }
            ("omni.tensor", "concat" | "softmax" | "gather") => {
                if let Some(a) = op.attr("axis") {
                    out.attrs
                        .push(write_attr("axis", &Value::map(vec![("i", a.clone())]))?);
                }
            }
            ("omni.tensor", "cast") => {
                let d = op
                    .attr("dtype")
                    .ok_or_else(|| Error::Core("cast has no `dtype`".into()))?;
                let dtype = DType::from_value(d).map_err(|e| Error::Core(e.to_string()))?;
                let code = code_of(&dtype).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "a cast to `{}`, which ONNX does not name",
                        dtype.alias().unwrap_or("(unnamed)")
                    ))
                })?;
                out.attrs.push(write_attr(
                    "to",
                    &Value::map(vec![("i", Value::U(u64::from(code as u32)))]),
                )?);
            }
            ("omni.tensor", "reshape") => {
                let shape = ints_attr(op, "shape")?;
                let name = folded_name(0, "shape");
                out.extra_inputs.push(name.clone());
                out.folded.push(Folded {
                    name,
                    values: shape,
                });
            }
            ("omni.tensor", "cumsum") => {
                let axis = as_int(op.attr("axis")).unwrap_or(-1);
                let name = folded_name(0, "axis");
                out.extra_inputs.push(name.clone());
                out.folded.push(Folded {
                    name,
                    values: vec![axis],
                });
            }
            ("omni.tensor", "reduce") => {
                let axes = ints_attr(op, "axes")?;
                if opset >= 18 {
                    let name = folded_name(0, "axes");
                    out.extra_inputs.push(name.clone());
                    out.folded.push(Folded { name, values: axes });
                } else {
                    out.attrs.push(write_attr(
                        "axes",
                        &Value::map(vec![("ints", int_array(&axes))]),
                    )?);
                }
                let keep = matches!(op.attr("keepdims"), Some(Value::Bool(true)));
                out.attrs.push(write_attr(
                    "keepdims",
                    &Value::map(vec![("i", Value::U(u64::from(keep)))]),
                )?);
            }
            _ => {}
        }
        return Ok(out);
    }

    // A compat-dialect op: an ONNX op that never stopped being one.
    if ir::dialect(&op.dialect).is_none() {
        out.op_type = op.name.clone();
        out.domain = if op.dialect == AI_ONNX {
            String::new()
        } else {
            op.dialect.clone()
        };
        for (k, v) in &op.attrs {
            out.attrs.push(write_attr(k, v)?);
        }
        return Ok(out);
    }

    Err(Error::Unsupported(format!(
        "{} has no ONNX spelling",
        op.qualified()
    )))
}

/// Writes an ONNX file from a container (E2, E3).
pub fn export(
    ctx: &Ctx<'_>,
    manifest: &Value,
    table: &TensorTable,
    module: Option<&Module>,
) -> Res<(Vec<u8>, Plan)> {
    let p = plan(ctx, manifest, table, module)?;
    if !p.writable() {
        let list = p
            .unmapped
            .iter()
            .map(|(o, n)| format!("  {o} ×{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(Error::Unsupported(format!(
            "these ops have no ONNX equivalent, so there is no file to write:\n{list}\n\
             This is not something --allow-lossy covers: an unwritable op is not lost \
             metadata, it is the computation. §07.7's shipped lowerings are what turn \
             a semantic op into ones ONNX has"
        )));
    }
    let m = module.expect("plan refuses a container with no graph");
    let f = entry(m)?;
    let ops = body(f)?;
    let env = preserved(ctx, manifest)?;
    // The preserved spellings line up with the graph only if the graph is still
    // the one they were recorded for. If it is not, the export names values
    // itself and says so by writing no node names.
    let names = aligned(&env, ops);

    // Value names. Three of the four kinds are in the graph itself — a
    // parameter names itself, a constant names its tensor, an output names
    // itself — and only the intermediates need the envelope.
    let mut value_names: Vec<(u32, String)> = Vec::new();
    let set = |list: &mut Vec<(u32, String)>, id: u32, name: String| match list
        .iter_mut()
        .find(|(i, _)| *i == id)
    {
        Some((_, n)) => *n = name,
        None => list.push((id, name)),
    };
    for (i, (name, _)) in f.params.iter().enumerate() {
        set(&mut value_names, i as u32, name.clone());
    }
    for (i, op) in ops.iter().enumerate() {
        if let Some(n) = op.attr("tensor").and_then(|v| v.as_str()) {
            if let Some((id, _)) = op.outputs.first() {
                set(&mut value_names, *id, n.to_string());
            }
            continue;
        }
        for (j, (id, _)) in op.outputs.iter().enumerate() {
            let name = names
                .and_then(|n| n[i].outputs.get(j))
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("omni_v{id}"));
            set(&mut value_names, *id, name);
        }
    }
    // An output's declared name wins over whatever produced it, since that is
    // the name the graph promises its callers.
    for op in ops {
        if op.dialect == "omni.io" && op.name == "output" {
            if let (Some(id), Some(n)) =
                (op.inputs.first(), op.attr("name").and_then(|v| v.as_str()))
            {
                set(&mut value_names, *id, n.to_string());
            }
        }
    }
    let name_of = |id: u32| -> String {
        value_names
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("omni_v{id}"))
    };
    let type_of = |id: u32| -> Option<Type> {
        if let Some((_, t)) = f.params.iter().enumerate().find(|(i, _)| *i as u32 == id) {
            return Some(t.1.clone());
        }
        ops.iter()
            .flat_map(|o| o.outputs.iter())
            .find(|(i, _)| *i == id)
            .map(|(_, t)| t.clone())
    };

    // The nodes, and the constants ONNX takes as operands.
    let mut nodes: Vec<Writer> = Vec::new();
    let mut folded: Vec<Folded> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        // A `Constant` node: the tensor goes back inside the graph rather than
        // into the initializer list, because that is where the file had it.
        if let Some(inner) = constant_node(names, i, op) {
            let table_name = op
                .attr("tensor")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Core("a constant names no tensor".into()))?;
            let r = table
                .tensors
                .get(table_name)
                .ok_or_else(|| Error::Core(format!("`{table_name}` is not in the table")))?;
            let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
            let shape = concrete(&desc.shape).ok_or_else(|| {
                Error::Unsupported(format!("`{table_name}` has a symbolic shape"))
            })?;
            let data = materialize(&desc, ctx)?;
            let mut a = Writer::new();
            a.text(1, "value");
            a.message(5, write_tensor(inner, &desc.dtype, &shape, &data)?);
            a.int(20, AttrType::Tensor.code());
            let mut w = Writer::new();
            for (id, _) in &op.outputs {
                w.raw_bytes(2, name_of(*id).as_bytes());
            }
            w.text(3, &names.map(|x| x[i].node.clone()).unwrap_or_default());
            w.text(4, "Constant");
            w.message(5, a);
            w.text(6, &names.map(|x| x[i].doc.clone()).unwrap_or_default());
            nodes.push(w);
            continue;
        }
        if structural(op) {
            continue;
        }
        let n = node_of(op, p.opset, names.map(|n| &n[i]), i)?;
        let mut w = Writer::new();
        for id in &op.inputs {
            w.raw_bytes(1, name_of(*id).as_bytes());
        }
        for name in &n.extra_inputs {
            w.raw_bytes(1, name.as_bytes());
        }
        for (id, _) in &op.outputs {
            w.raw_bytes(2, name_of(*id).as_bytes());
        }
        let node_name = names
            .map(|x| x[i].node.clone())
            .filter(|s| !s.is_empty())
            .or_else(|| op.loc.clone())
            .unwrap_or_default();
        w.text(3, &node_name);
        w.text(4, &n.op_type);
        for a in n.attrs {
            w.message(5, a);
        }
        if let Some(d) = names.map(|x| x[i].doc.clone()) {
            w.text(6, &d);
        }
        w.text(7, &n.domain);
        nodes.push(w);
        folded.extend(n.folded);
    }

    // The initializers, in the order the plan settled — which is the source
    // file's own order when there was one.
    let mut initializers: Vec<Writer> = Vec::new();
    for (name, _, shape, _) in &p.initializers {
        let r = table.tensors.get(name).ok_or_else(|| {
            Error::Core(format!("`{name}` left the table between plan and write"))
        })?;
        let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
        let data = materialize(&desc, ctx)?;
        initializers.push(write_tensor(name, &desc.dtype, shape, &data)?);
    }
    for x in &folded {
        // A folded constant that is already an initializer is the one the
        // import took the attribute from; writing it twice would make the file
        // invalid.
        if p.initializers.iter().any(|(n, _, _, _)| n == &x.name) {
            continue;
        }
        let mut data = Vec::with_capacity(x.values.len() * 8);
        for v in &x.values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        initializers.push(write_tensor(
            &x.name,
            &DType::I64,
            &[x.values.len() as u64],
            &data,
        )?);
    }

    // The graph.
    let mut g = Writer::new();
    for n in nodes {
        g.message(1, n);
    }
    let graph_name = env
        .as_ref()
        .map(|e| e.graph_name.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "main".into());
    g.text(2, &graph_name);
    for t in initializers {
        g.message(5, t);
    }
    if let Some(e) = &env {
        g.text(10, &e.graph_doc);
    }
    let doc_of = |list: &[(String, String)], name: &str| -> String {
        list.iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.clone())
            .unwrap_or_default()
    };
    for (i, (name, t)) in f.params.iter().enumerate() {
        let _ = i;
        let doc = env
            .as_ref()
            .map(|e| doc_of(&e.inputs, name))
            .unwrap_or_default();
        g.message(11, write_value_info(name, &doc, Some(t))?);
    }
    for op in ops {
        if op.dialect != "omni.io" || op.name != "output" {
            continue;
        }
        let name = op
            .attr("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Core("omni.io/output has no `name`".into()))?;
        let t = op.inputs.first().and_then(|id| type_of(*id));
        let doc = env
            .as_ref()
            .map(|e| doc_of(&e.outputs, name))
            .unwrap_or_default();
        g.message(12, write_value_info(name, &doc, t.as_ref())?);
    }
    if let Some(e) = &env {
        for (name, doc) in &e.value_info {
            let Some((id, _)) = value_names.iter().find(|(_, n)| n == name) else {
                continue;
            };
            let t = type_of(*id);
            g.message(13, write_value_info(name, doc, t.as_ref())?);
        }
    }

    // The model.
    let mut w = Writer::new();
    w.int_always(
        1,
        env.as_ref()
            .map(|e| e.ir_version)
            .filter(|v| *v > 0)
            .unwrap_or(IR_VERSION),
    );
    match &env {
        Some(e) => {
            w.text(2, &e.producer_name);
            w.text(3, &e.producer_version);
            w.text(4, &e.domain);
            w.int(5, e.model_version);
            w.text(6, &e.doc_string);
        }
        None => {
            w.text(2, EXPORTER);
            w.text(3, env!("CARGO_PKG_VERSION"));
        }
    }
    w.message(7, g);
    let opsets: Vec<(String, i64)> = match &env {
        Some(e) if !e.opsets.is_empty() => e.opsets.clone(),
        _ => {
            let mut v = vec![(String::new(), p.opset)];
            // Every compat dialect the graph uses is a domain the file has to
            // import an opset for, or a reader cannot know what its ops mean.
            for d in &m.dialects {
                if ir::dialect(&d.ns).is_none() && d.ns != AI_ONNX {
                    v.push((d.ns.clone(), i64::from(d.version)));
                }
            }
            v
        }
    };
    for (domain, version) in &opsets {
        let mut o = Writer::new();
        o.text(1, domain);
        o.int_always(2, *version);
        w.message(8, o);
    }
    if let Some(e) = &env {
        for (k, v) in &e.metadata {
            let mut kv = Writer::new();
            kv.text(1, k);
            kv.text(2, v);
            w.message(14, kv);
        }
    }
    Ok((w.buf, p))
}

// ---------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Tensor as ETensor;

    /// A minimal ONNX writer for the tests, built on the same [`Writer`] the
    /// exporter uses. It writes fields in ascending order, which is what makes
    /// "the export is byte-identical to its source" a statement about two
    /// writers agreeing rather than about one writer.
    #[derive(Default)]
    struct Build {
        nodes: Vec<Writer>,
        inits: Vec<Writer>,
        inputs: Vec<Writer>,
        outputs: Vec<Writer>,
        value_info: Vec<Writer>,
        opset: i64,
        extra_opsets: Vec<(String, i64)>,
    }

    fn ty(elem: i32, dims: &[Option<i64>]) -> Writer {
        let mut shape = Writer::new();
        for d in dims {
            let mut dim = Writer::new();
            match d {
                Some(n) => dim.int_always(1, *n),
                None => dim.text(2, "B"),
            }
            shape.message(1, dim);
        }
        let mut t = Writer::new();
        t.int(1, i64::from(elem));
        t.message(2, shape);
        let mut out = Writer::new();
        out.message(1, t);
        out
    }

    fn value_info(name: &str, elem: i32, dims: &[Option<i64>]) -> Writer {
        let mut w = Writer::new();
        w.text(1, name);
        w.message(2, ty(elem, dims));
        w
    }

    fn f32_init(name: &str, dims: &[u64], data: &[f32]) -> Writer {
        let mut raw = Vec::new();
        for x in data {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        let mut w = Writer::new();
        w.packed_ints(1, &dims.iter().map(|d| *d as i64).collect::<Vec<i64>>());
        w.int(2, 1);
        w.text(8, name);
        w.bytes(9, &raw);
        w
    }

    fn i64_init(name: &str, data: &[i64]) -> Writer {
        let mut raw = Vec::new();
        for x in data {
            raw.extend_from_slice(&x.to_le_bytes());
        }
        let mut w = Writer::new();
        w.packed_ints(1, &[data.len() as i64]);
        w.int(2, 7);
        w.text(8, name);
        w.bytes(9, &raw);
        w
    }

    fn attr_int(name: &str, v: i64) -> Writer {
        let mut w = Writer::new();
        w.text(1, name);
        w.int(3, v);
        w.int(20, 2);
        w
    }

    fn attr_ints(name: &str, v: &[i64]) -> Writer {
        let mut w = Writer::new();
        w.text(1, name);
        w.packed_ints(8, v);
        w.int(20, 7);
        w
    }

    fn attr_float(name: &str, v: f32) -> Writer {
        let mut w = Writer::new();
        w.text(1, name);
        w.f32(2, v);
        w.int(20, 1);
        w
    }

    fn node(op_type: &str, ins: &[&str], outs: &[&str], name: &str, attrs: Vec<Writer>) -> Writer {
        let mut w = Writer::new();
        for i in ins {
            w.raw_bytes(1, i.as_bytes());
        }
        for o in outs {
            w.raw_bytes(2, o.as_bytes());
        }
        w.text(3, name);
        w.text(4, op_type);
        for a in attrs {
            w.message(5, a);
        }
        w
    }

    impl Build {
        fn model(self) -> Vec<u8> {
            let mut g = Writer::new();
            for n in self.nodes {
                g.message(1, n);
            }
            g.text(2, "main");
            for t in self.inits {
                g.message(5, t);
            }
            for i in self.inputs {
                g.message(11, i);
            }
            for o in self.outputs {
                g.message(12, o);
            }
            for v in self.value_info {
                g.message(13, v);
            }
            let mut w = Writer::new();
            w.int_always(1, IR_VERSION);
            w.text(2, "omni-test");
            w.text(3, "1.0");
            w.message(7, g);
            let mut o = Writer::new();
            o.int_always(2, if self.opset == 0 { 17 } else { self.opset });
            w.message(8, o);
            for (d, v) in &self.extra_opsets {
                let mut o = Writer::new();
                o.text(1, d);
                o.int_always(2, *v);
                w.message(8, o);
            }
            w.buf
        }
    }

    /// x: [B,4] · w: [4,3] + b: [3], then tanh. Small enough to check by hand
    /// and big enough to exercise a parameter, an initializer, three nodes and
    /// a symbolic dimension.
    fn linear_model() -> Vec<u8> {
        Build {
            nodes: vec![
                node("MatMul", &["x", "w"], &["h"], "mm", vec![]),
                node("Add", &["h", "b"], &["z"], "bias", vec![]),
                node("Tanh", &["z"], &["y"], "act", vec![]),
            ],
            inits: vec![
                f32_init(
                    "w",
                    &[4, 3],
                    &[1.0, 0.0, -1.0, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, -2.0, 0.0, 0.5],
                ),
                f32_init("b", &[3], &[0.25, -0.5, 0.0]),
            ],
            inputs: vec![value_info("x", 1, &[None, Some(4)])],
            outputs: vec![value_info("y", 1, &[None, Some(3)])],
            ..Default::default()
        }
        .model()
    }

    fn imported(bytes: &[u8]) -> Imported {
        import(bytes, &ImportOpts::default(), &NoExternal).expect("import")
    }

    #[test]
    fn varints_round_trip_at_the_edges() {
        for v in [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut w = Writer::new();
            w.int_always(1, v as i64);
            let mut r = Reader::new(&w.buf);
            let (f, x) = r.next_field().unwrap().unwrap();
            assert_eq!(f, 1);
            assert_eq!(x.as_u64().unwrap(), v);
            assert!(r.done());
        }
    }

    #[test]
    fn a_group_is_refused_rather_than_skipped() {
        // Wire type 3 with no length: a reader that skipped it would read the
        // rest of the message from the wrong offset.
        let bytes = [0x0bu8, 0x08, 0x01];
        let mut r = Reader::new(&bytes);
        let e = r.next_field().unwrap_err();
        assert!(format!("{e}").contains("group"), "{e}");
    }

    #[test]
    fn a_truncated_field_does_not_allocate_the_machine() {
        // A length-delimited field claiming 2^40 bytes, in a 3-byte message.
        let mut w = Writer::new();
        w.buf.push(0x0a);
        w.varint(1 << 40);
        let mut r = Reader::new(&w.buf);
        let e = r.next_field().unwrap_err();
        assert!(format!("{e}").contains("bound"), "{e}");
    }

    #[test]
    fn a_linear_graph_imports_with_its_weights_and_its_computation() {
        let file = linear_model();
        let im = imported(&file);
        assert_eq!(im.initializers, 2);
        assert_eq!(im.report.verified_tensors, 2);
        assert!(im.compat.is_empty(), "{:?}", im.compat);
        let m = im.module.expect("a graph");
        assert_eq!(m.level, Level::Primitive);

        // The ops, in order: two constants (materialized where first read),
        // three computations, one io/output and a return.
        let f = m.function("main").unwrap();
        let ops = &f.body.blocks[0].ops;
        let seq: Vec<String> = ops.iter().map(|o| o.qualified()).collect();
        assert_eq!(
            seq,
            vec![
                "omni.core/constant@1",
                "omni.tensor/matmul@1",
                "omni.core/constant@1",
                "omni.tensor/add@1",
                "omni.tensor/tanh@1",
                "omni.io/output@1",
                "omni.core/return@1",
            ]
        );
        // The parameter kept ONNX's symbolic batch dimension.
        assert_eq!(
            f.params[0].1,
            Type::tensor(vec![Dim::Sym("B".into()), Dim::N(4)], DType::F32)
        );
    }

    #[test]
    fn the_imported_graph_verifies_against_its_own_tensors() {
        let im = imported(&linear_model());
        let m = im.module.unwrap();
        let lookup = |name: &str| -> Option<(Vec<u64>, DType)> {
            match name {
                "w" => Some((vec![4, 3], DType::F32)),
                "b" => Some((vec![3], DType::F32)),
                _ => None,
            }
        };
        let cx = ir::Context {
            tensor: Some(&lookup),
            rewrites: &[],
        };
        let report = ir::verify(&m, &cx);
        assert!(
            report.findings.iter().all(|f| !f.is_invalid()),
            "{:?}",
            report.findings
        );
    }

    #[test]
    fn the_imported_graph_computes_what_the_file_computes() {
        let im = imported(&linear_model());
        let m = im.module.unwrap();
        let w = ETensor::new(
            vec![4, 3],
            DType::F32,
            vec![1.0, 0.0, -1.0, 0.5, 2.0, 0.0, 0.0, 1.0, 1.0, -2.0, 0.0, 0.5],
        );
        let b = ETensor::new(vec![3], DType::F32, vec![0.25, -0.5, 0.0]);
        let weights: Vec<(String, ETensor)> = vec![("w".into(), w), ("b".into(), b)];
        let x = ETensor::new(
            vec![2, 4],
            DType::F32,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, -1.0],
        );
        let out =
            crate::interp::run(&m, &[x], &weights, &crate::interp::Limits::default()).expect("run");
        // Row 0: [1,2,3,4]·W = [1*1+2*0.5+3*0+4*-2, 2*2+3*1, -1+3+4*0.5] = [-6, 7, 4]
        // then + b = [-5.75, 6.5, 4], then tanh.
        let got = &out.returned[0];
        assert_eq!(got.shape, vec![2, 3]);
        let want = [(-5.75f64).tanh(), (6.5f64).tanh(), (4.0f64).tanh()];
        for (g, w) in got.data.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        // And the graph's own output name survived the trip through §07.8.
        assert_eq!(out.outputs[0].0, "y");
    }

    #[test]
    fn an_export_of_an_import_is_the_same_file() {
        let file = linear_model();
        let im = imported(&file);
        let store = store_of(&im.objects, HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = table_of(&im.objects, &im.root, HashAlgo::default()).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &im.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let (out, p) = export(&ctx, &manifest, &table, im.module.as_ref()).expect("export");
        assert!(p.writable());
        assert_eq!(
            out, file,
            "an ONNX file this build imported did not come back byte for byte"
        );
    }

    #[test]
    fn an_op_omni_has_no_word_for_is_carried_rather_than_guessed_at() {
        let file = Build {
            nodes: vec![
                node("Relu", &["x"], &["r"], "relu", vec![]),
                node(
                    "LeakyRelu",
                    &["r"],
                    &["y"],
                    "leaky",
                    vec![attr_float("alpha", 0.125)],
                ),
            ],
            inputs: vec![value_info("x", 1, &[Some(2), Some(2)])],
            outputs: vec![value_info("y", 1, &[Some(2), Some(2)])],
            value_info: vec![value_info("r", 1, &[Some(2), Some(2)])],
            ..Default::default()
        }
        .model();
        let im = imported(&file);
        assert_eq!(im.native, Vec::<(String, usize)>::new());
        let ops: Vec<String> = im.compat.iter().map(|(o, _, _)| o.clone()).collect();
        assert_eq!(ops, vec!["Relu", "LeakyRelu"]);
        let m = im.module.clone().unwrap();
        // The compat dialect is the ONNX domain, at the opset the file imported.
        let f = m.function("main").unwrap();
        let relu = &f.body.blocks[0].ops[0];
        assert_eq!(relu.qualified(), "ai.onnx/Relu@17");
        assert_eq!(
            f.body.blocks[0].ops[1].attr("alpha"),
            Some(&Value::map(vec![("f", Value::F64(0.125))]))
        );

        // §15.1: an op from a dialect this reader does not know is
        // *indeterminate*, and reporting it as invalid would itself be a
        // conformance violation.
        let cx = ir::Context::default();
        let report = ir::verify(&m, &cx);
        assert!(
            report.findings.iter().all(|f| !f.is_invalid()),
            "{:?}",
            report.findings
        );
        assert!(report
            .findings
            .iter()
            .any(|f| f.message().contains("ai.onnx")));

        // And it exports as itself, attribute intact.
        let store = store_of(&im.objects, HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = table_of(&im.objects, &im.root, HashAlgo::default()).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &im.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let (out, _) = export(&ctx, &manifest, &table, im.module.as_ref()).unwrap();
        assert_eq!(out, file);
    }

    #[test]
    fn an_attribute_folded_into_an_operand_comes_back_as_an_operand() {
        let file = Build {
            nodes: vec![
                node("Reshape", &["x", "newshape"], &["r"], "rs", vec![]),
                node(
                    "ReduceSum",
                    &["r"],
                    &["y"],
                    "sum",
                    vec![attr_ints("axes", &[1]), attr_int("keepdims", 0)],
                ),
            ],
            inits: vec![i64_init("newshape", &[2, 6])],
            inputs: vec![value_info("x", 1, &[Some(3), Some(4)])],
            outputs: vec![value_info("y", 1, &[Some(2)])],
            opset: 13,
            ..Default::default()
        }
        .model();
        let im = imported(&file);
        assert!(im.compat.is_empty(), "{:?}", im.compat);
        let m = im.module.clone().unwrap();
        let ops = &m.function("main").unwrap().body.blocks[0].ops;
        // The shape operand became an attribute, so the constant that held it
        // is not read by anything and no `core.constant` was emitted for it.
        assert_eq!(ops[0].qualified(), "omni.tensor/reshape@1");
        assert_eq!(
            ops[0].attr("shape"),
            Some(&Value::Array(vec![Value::U(2), Value::U(6)]))
        );
        assert_eq!(ops[1].qualified(), "omni.tensor/reduce@1");

        let store = store_of(&im.objects, HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = table_of(&im.objects, &im.root, HashAlgo::default()).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &im.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let (out, _) = export(&ctx, &manifest, &table, im.module.as_ref()).unwrap();
        assert_eq!(out, file);
    }

    #[test]
    fn a_constant_node_is_a_node_again_and_not_an_initializer() {
        // ONNX stores a constant two ways — an initializer, or a `Constant`
        // node with a tensor attribute — and both import to one
        // `omni.core/constant`. Which one the file had is not something the
        // export gets to pick.
        let mut value = Writer::new();
        value.packed_ints(1, &[2]);
        value.int(2, 1);
        value.text(8, "the_value");
        value.bytes(9, &[0, 0, 0, 64, 0, 0, 128, 63]);
        let mut a = Writer::new();
        a.text(1, "value");
        a.message(5, value);
        a.int(20, 4);
        let file = Build {
            nodes: vec![
                node("Constant", &[], &["c"], "k", vec![a]),
                node("Mul", &["x", "c"], &["y"], "m", vec![]),
            ],
            inputs: vec![value_info("x", 1, &[Some(2)])],
            outputs: vec![value_info("y", 1, &[Some(2)])],
            value_info: vec![value_info("c", 1, &[Some(2)])],
            ..Default::default()
        }
        .model();
        let im = imported(&file);
        assert_eq!(im.initializers, 1);
        assert_eq!(
            im.native,
            vec![("Constant".to_string(), 1), ("Mul".to_string(), 1)]
        );

        let store = store_of(&im.objects, HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = table_of(&im.objects, &im.root, HashAlgo::default()).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &im.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let p = plan(&ctx, &manifest, &table, im.module.as_ref()).unwrap();
        assert!(p.initializers.is_empty(), "{:?}", p.initializers);
        let (out, _) = export(&ctx, &manifest, &table, im.module.as_ref()).unwrap();
        assert_eq!(out, file);
    }

    #[test]
    fn two_shape_functions_disagreeing_is_an_error_and_not_a_shrug() {
        // ONNX says the product of [2,4] and [4,3] is [2,5]. One of the two
        // readers is wrong about this model, and importing it either way would
        // be picking one without saying so.
        let file = Build {
            nodes: vec![node("MatMul", &["x", "w"], &["y"], "mm", vec![])],
            inits: vec![f32_init("w", &[4, 3], &[0.0; 12])],
            inputs: vec![value_info("x", 1, &[Some(2), Some(4)])],
            outputs: vec![value_info("y", 1, &[Some(2), Some(5)])],
            ..Default::default()
        }
        .model();
        let e = import(&file, &ImportOpts::default(), &NoExternal).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("ONNX declares axis 1"), "{msg}");
    }

    #[test]
    fn an_untypeable_value_is_named_rather_than_invented() {
        // A compat op with no `value_info`: nothing in the file or in this
        // build says what `r` is.
        let file = Build {
            nodes: vec![
                node("Relu", &["x"], &["r"], "relu", vec![]),
                node("Tanh", &["r"], &["y"], "t", vec![]),
            ],
            inputs: vec![value_info("x", 1, &[Some(2)])],
            outputs: vec![value_info("y", 1, &[Some(2)])],
            ..Default::default()
        }
        .model();
        let e = import(&file, &ImportOpts::default(), &NoExternal).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("shape inference"), "{msg}");
        // The weights are still importable without the graph.
        let opts = ImportOpts {
            graph: false,
            ..Default::default()
        };
        assert!(import(&file, &opts, &NoExternal).is_ok());
    }

    #[test]
    fn a_dtype_omni_will_not_guess_at_is_refused_by_name() {
        for (code, needle) in [(8, "STRING"), (18, "FLOAT8E4M3FNUZ"), (14, "COMPLEX64")] {
            let mut t = Writer::new();
            t.packed_ints(1, &[1]);
            t.int(2, i64::from(code));
            t.text(8, "k");
            t.bytes(9, &[0u8; 8]);
            let file = Build {
                nodes: vec![node("Identity", &["k"], &["y"], "id", vec![])],
                inits: vec![t],
                outputs: vec![value_info("y", 1, &[Some(1)])],
                ..Default::default()
            }
            .model();
            let e = import(&file, &ImportOpts::default(), &NoExternal).unwrap_err();
            assert!(format!("{e}").contains(needle), "{e} did not name {needle}");
        }
    }

    #[test]
    fn external_data_stays_inside_the_models_directory() {
        for bad in ["../secret.bin", "/etc/passwd", "a/../../b", "C:/x"] {
            assert!(!DirExternal::is_contained(bad), "{bad} was allowed");
        }
        for good in ["weights.bin", "data/weights.bin"] {
            assert!(DirExternal::is_contained(good), "{good} was refused");
        }

        let mut t = Writer::new();
        t.packed_ints(1, &[2]);
        t.int(2, 1);
        t.text(8, "w");
        let mut kv = Writer::new();
        kv.text(1, "location");
        kv.text(2, "../../etc/passwd");
        t.message(13, kv);
        t.int(14, 1);
        let file = Build {
            nodes: vec![node("Identity", &["w"], &["y"], "id", vec![])],
            inits: vec![t],
            outputs: vec![value_info("y", 1, &[Some(2)])],
            ..Default::default()
        }
        .model();
        let e = import(&file, &ImportOpts::default(), &NoExternal).unwrap_err();
        assert!(
            format!("{e}").contains("leaves the model's own directory"),
            "{e}"
        );
    }

    #[test]
    fn an_export_refuses_an_op_onnx_cannot_spell_and_says_which() {
        let mut m = Module::new(Level::Primitive, "main");
        m.dialects = vec![
            ir::DialectUse {
                ns: "omni.core".into(),
                version: 1,
                reference: None,
            },
            ir::DialectUse {
                ns: "omni.nn".into(),
                version: 1,
                reference: None,
            },
        ];
        let t = Type::tensor(vec![Dim::N(1), Dim::N(2)], DType::F32);
        m.functions.push((
            "main".into(),
            Function {
                params: vec![
                    ("q".into(), t.clone()),
                    ("k".into(), t.clone()),
                    ("v".into(), t.clone()),
                ],
                results: vec![t.clone()],
                attrs: Vec::new(),
                body: Region {
                    blocks: vec![Block {
                        args: Vec::new(),
                        ops: vec![
                            Op::new("omni.nn", "attention", 1)
                                .with_inputs(&[0, 1, 2])
                                .with_output(3, t.clone()),
                            Op::new("omni.core", "return", 1).with_inputs(&[3]),
                        ],
                    }],
                },
                constraints: Vec::new(),
            },
        ));
        let store = crate::store::MemoryStore::new(HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = TensorTable::default();
        let manifest = Value::map(vec![("t", Value::text("omni.core/manifest"))]);
        let p = plan(&ctx, &manifest, &table, Some(&m)).unwrap();
        assert_eq!(p.unmapped, vec![("omni.nn/attention@1".to_string(), 1)]);
        let e = export(&ctx, &manifest, &table, Some(&m)).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("omni.nn/attention@1"), "{msg}");
        assert!(msg.contains("--allow-lossy"), "{msg}");
    }

    #[test]
    fn an_export_of_a_container_with_no_graph_says_so() {
        let store = crate::store::MemoryStore::new(HashAlgo::default());
        let ctx = Ctx::new(&store);
        let table = TensorTable::default();
        let manifest = Value::map(vec![("t", Value::text("omni.core/manifest"))]);
        let e = plan(&ctx, &manifest, &table, None).unwrap_err();
        assert!(format!("{e}").contains("no execution graph"), "{e}");
    }

    #[test]
    fn quantize_and_dequantize_become_the_quantization_dialect() {
        let file = Build {
            nodes: vec![
                node("QuantizeLinear", &["x", "s", "z"], &["q"], "q", vec![]),
                node("DequantizeLinear", &["q", "s", "z"], &["y"], "dq", vec![]),
            ],
            inits: vec![f32_init("s", &[], &[0.5]), {
                let mut w = Writer::new();
                w.int(2, 2);
                w.text(8, "z");
                w.bytes(9, &[128u8]);
                w
            }],
            inputs: vec![value_info("x", 1, &[Some(4)])],
            outputs: vec![value_info("y", 1, &[Some(4)])],
            value_info: vec![value_info("q", 2, &[Some(4)])],
            ..Default::default()
        }
        .model();
        let im = imported(&file);
        assert!(im.compat.is_empty(), "{:?}", im.compat);
        let m = im.module.clone().unwrap();
        let ops = &m.function("main").unwrap().body.blocks[0].ops;
        let q = ops
            .iter()
            .find(|o| o.name == "quantize")
            .expect("a quantize");
        // §05.1's closed enumeration is why this is a mapping rather than a
        // guess: ONNX defines y = (x - zero_point) * scale.
        assert_eq!(
            q.attr("scheme")
                .and_then(|s| s.get("formula"))
                .and_then(|f| f.as_str()),
            Some("affine-sub")
        );

        // It runs, and it round-trips a value through u8 and back.
        let weights: Vec<(String, ETensor)> = vec![
            ("s".into(), ETensor::new(vec![], DType::F32, vec![0.5])),
            ("z".into(), ETensor::new(vec![], DType::U8, vec![128.0])),
        ];
        let x = ETensor::new(vec![4], DType::F32, vec![0.0, 1.0, -2.0, 3.5]);
        let out =
            crate::interp::run(&m, &[x], &weights, &crate::interp::Limits::default()).expect("run");
        assert_eq!(out.returned[0].data, vec![0.0, 1.0, -2.0, 3.5]);
    }
}
