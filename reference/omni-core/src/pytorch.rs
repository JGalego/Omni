//! PyTorch checkpoint import (`.bin`, `.pt`, `.pth`) — §12.10 and
//! `docs/design/import-export.md` §3.
//!
//! A `torch.save` file is a ZIP archive containing one **pickle** and a data
//! file per storage. The pickle is the problem: unpickling is executing, and
//! `torch.load` on an untrusted file has been a remote-code-execution primitive
//! for as long as the format has existed. §12.10 is the specification's answer
//! and this module is its implementation:
//!
//! 1. **A restricted unpickler with an opcode allowlist and a class
//!    allowlist.** Every opcode this module does not implement is an error
//!    naming the opcode. Every `GLOBAL` outside [`ALLOWED_GLOBALS`] is an error
//!    naming the symbol. `REDUCE` calls one of six known tensor-reconstruction
//!    functions or fails.
//! 2. §12.10 also asks for a confined child process. This build does not
//!    provide one, and the reason is worth stating rather than hiding: there is
//!    nothing here to confine. The unpickler is not a Python interpreter with a
//!    filter bolted on — it has no `import`, no attribute access, no call
//!    mechanism beyond a match on six names, and no way to reach the
//!    filesystem or the network. Process isolation defends against a *general*
//!    evaluator; this is a parser for a data language that happens to share
//!    pickle's byte encoding. A production importer that ever grows a general
//!    evaluator needs the sandbox back.
//! 3. The `Provenance` object records that the source was an unsafe format,
//!    what was refused, and the digest of the file.
//! 4. Nothing here ever writes pickle.
//!
//! ## What is represented
//!
//! Tensors: dtype, shape, strides, and the bytes, exactly. A PyTorch tensor is
//! a *view* — an offset and a stride tuple over a flat storage — and §04.4's
//! `strided` layout is the same idea, so a transposed or otherwise
//! non-contiguous tensor keeps its strides rather than being silently
//! materialized into a different array.
//!
//! ## What is not, and is said out loud
//!
//! Everything else in a checkpoint is arbitrary Python: optimizer state,
//! learning-rate schedules, epoch counters, `argparse.Namespace` objects. Scalar
//! leaves are preserved verbatim in a `Foreign` object so nothing is lost (I2);
//! anything requiring a Python class to reconstruct is refused by name, because
//! the alternative is running it.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::expr::Ctx;
use crate::layout::{Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{Fidelity, ImportOpts, Imported, Note};
use crate::tensor::{TensorDesc, TensorTable};

use std::collections::BTreeMap;

pub const IMPORTER: &str = "omni-core/pytorch";

#[derive(Debug)]
pub enum Error {
    /// The archive framing is wrong.
    Zip(String),
    /// The pickle stream is malformed.
    Pickle(String),
    /// The pickle is well formed and asks for something this importer refuses
    /// to do. This is the §12.10 case and it is deliberately a *hard* error.
    Refused(String),
    /// The checkpoint is fine and this build cannot represent part of it.
    Unsupported(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Zip(m) => write!(f, "not a readable torch archive: {m}"),
            Error::Pickle(m) => write!(f, "malformed pickle: {m}"),
            Error::Refused(m) => write!(f, "refused (§12.10): {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}
impl std::error::Error for Error {}

pub type Res<T> = Result<T, Error>;

// --------------------------------------------------------------------- zip --

/// The bound on a single decompressed archive member. Data files are stored,
/// not deflated, in every `torch.save` archive; a deflated member is a pickle
/// or a version string, and those are small. A declared size is untrusted input
/// (§12.4), so it does not get to drive an allocation.
const MAX_INFLATED: usize = 1 << 28;

#[derive(Clone, Debug)]
pub struct ZipEntry {
    pub name: String,
    /// Offset of the local file header.
    pub header_off: u64,
    pub compressed: u64,
    pub size: u64,
    pub method: u16,
}

/// A ZIP archive, read from its central directory.
///
/// Zip64 is not optional here: a 7 B model in fp16 is 14 GB, which is three
/// orders of magnitude past what the 32-bit fields can express, so the 64-bit
/// end-of-central-directory record and the 0x0001 extra field are the normal
/// path rather than an edge case.
pub struct Zip<'a> {
    bytes: &'a [u8],
    pub entries: Vec<ZipEntry>,
}

fn u16le(b: &[u8], at: usize) -> Res<u16> {
    b.get(at..at + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| Error::Zip(format!("truncated at byte {at}")))
}

fn u32le(b: &[u8], at: usize) -> Res<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| Error::Zip(format!("truncated at byte {at}")))
}

fn u64le(b: &[u8], at: usize) -> Res<u64> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| Error::Zip(format!("truncated at byte {at}")))
}

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const EOCD64_LOC_SIG: u32 = 0x0706_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

impl<'a> Zip<'a> {
    pub fn open(bytes: &'a [u8]) -> Res<Zip<'a>> {
        // The end-of-central-directory record is at the end, behind a comment
        // of up to 65 535 bytes, so it is found by scanning backwards.
        let window = bytes.len().min(22 + 0xFFFF);
        let start = bytes.len() - window;
        let mut eocd = None;
        for i in (start..bytes.len().saturating_sub(21)).rev() {
            if u32le(bytes, i)? == EOCD_SIG {
                eocd = Some(i);
                break;
            }
        }
        let eocd = eocd.ok_or_else(|| {
            Error::Zip("no end-of-central-directory record; not a ZIP archive".into())
        })?;

        let mut count = u16le(bytes, eocd + 10)? as u64;
        let mut cd_off = u32le(bytes, eocd + 16)? as u64;
        let mut cd_size = u32le(bytes, eocd + 12)? as u64;

        // Zip64: the 32-bit fields saturate, and the real ones live in a
        // separate record found through a locator immediately before the EOCD.
        if count == 0xFFFF || cd_off == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF {
            if eocd < 20 {
                return Err(Error::Zip(
                    "the archive saturates the 32-bit fields but has no Zip64 locator".into(),
                ));
            }
            let loc = eocd - 20;
            if u32le(bytes, loc)? != EOCD64_LOC_SIG {
                return Err(Error::Zip("Zip64 locator signature is wrong".into()));
            }
            let z64 = u64le(bytes, loc + 8)? as usize;
            if u32le(bytes, z64)? != EOCD64_SIG {
                return Err(Error::Zip(
                    "Zip64 end-of-central-directory is not there".into(),
                ));
            }
            count = u64le(bytes, z64 + 32)?;
            cd_size = u64le(bytes, z64 + 40)?;
            cd_off = u64le(bytes, z64 + 48)?;
        }
        if cd_off.saturating_add(cd_size) > bytes.len() as u64 {
            return Err(Error::Zip(
                "the central directory runs past the file".into(),
            ));
        }
        if count > 1 << 22 {
            return Err(Error::Zip(format!("{count} entries is not a checkpoint")));
        }

        let mut entries = Vec::with_capacity(count.min(4096) as usize);
        let mut p = cd_off as usize;
        for _ in 0..count {
            if u32le(bytes, p)? != CDIR_SIG {
                return Err(Error::Zip(format!(
                    "central directory entry at {p} has the wrong signature"
                )));
            }
            let method = u16le(bytes, p + 10)?;
            let mut compressed = u32le(bytes, p + 20)? as u64;
            let mut size = u32le(bytes, p + 24)? as u64;
            let name_len = u16le(bytes, p + 28)? as usize;
            let extra_len = u16le(bytes, p + 30)? as usize;
            let comment_len = u16le(bytes, p + 32)? as usize;
            let mut header_off = u32le(bytes, p + 42)? as u64;
            let name_at = p + 46;
            let name = bytes
                .get(name_at..name_at + name_len)
                .ok_or_else(|| Error::Zip("a name runs past the file".into()))?;
            let name = std::str::from_utf8(name)
                .map_err(|_| Error::Zip("a member name is not UTF-8".into()))?
                .to_string();

            // The 0x0001 extra field carries whichever of the four fields
            // saturated, in that fixed order and only those.
            let extra_at = name_at + name_len;
            let extra = bytes
                .get(extra_at..extra_at + extra_len)
                .ok_or_else(|| Error::Zip("an extra field runs past the file".into()))?;
            let mut q = 0usize;
            while q + 4 <= extra.len() {
                let id = u16le(extra, q)?;
                let len = u16le(extra, q + 2)? as usize;
                if q + 4 + len > extra.len() {
                    break;
                }
                if id == 0x0001 {
                    let f = &extra[q + 4..q + 4 + len];
                    let mut r = 0usize;
                    for target in [&mut size, &mut compressed, &mut header_off] {
                        if *target == 0xFFFF_FFFF && r + 8 <= f.len() {
                            *target = u64le(f, r)?;
                            r += 8;
                        }
                    }
                }
                q += 4 + len;
            }
            entries.push(ZipEntry {
                name,
                header_off,
                compressed,
                size,
                method,
            });
            p = extra_at + extra_len + comment_len;
        }
        Ok(Zip { bytes, entries })
    }

    pub fn find(&self, name: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// A member's bytes. Stored members are borrowed from the archive, which is
    /// the case that matters: `torch.save` stores its data files uncompressed,
    /// so a multi-gigabyte checkpoint is never inflated into a second copy.
    pub fn read(&self, e: &ZipEntry) -> Res<std::borrow::Cow<'a, [u8]>> {
        let h = e.header_off as usize;
        if u32le(self.bytes, h)? != LOCAL_SIG {
            return Err(Error::Zip(format!(
                "`{}` does not start with a local file header",
                e.name
            )));
        }
        let name_len = u16le(self.bytes, h + 26)? as usize;
        let extra_len = u16le(self.bytes, h + 28)? as usize;
        let at = h + 30 + name_len + extra_len;
        let raw = self
            .bytes
            .get(at..at + e.compressed as usize)
            .ok_or_else(|| Error::Zip(format!("`{}` runs past the file", e.name)))?;
        match e.method {
            0 => {
                if e.compressed != e.size {
                    return Err(Error::Zip(format!(
                        "`{}` is stored but its two lengths disagree ({} vs {})",
                        e.name, e.compressed, e.size
                    )));
                }
                Ok(std::borrow::Cow::Borrowed(raw))
            }
            8 => {
                let cap = (e.size as usize).min(MAX_INFLATED);
                let out = crate::codec::inflate(raw, cap)
                    .map_err(|err| Error::Zip(format!("`{}`: {err}", e.name)))?;
                if out.len() as u64 != e.size {
                    return Err(Error::Zip(format!(
                        "`{}` inflates to {} bytes, not the {} it declares",
                        e.name,
                        out.len(),
                        e.size
                    )));
                }
                Ok(std::borrow::Cow::Owned(out))
            }
            m => Err(Error::Unsupported(format!(
                "`{}` uses ZIP compression method {m}; only stored and deflate are read",
                e.name
            ))),
        }
    }
}

// ------------------------------------------------------------------ pickle --

/// Every symbol this unpickler will resolve. Anything else is a hard error
/// naming it, which is §12.10 clause 1.
///
/// The list is short because it is the whole attack surface. `torch.load`'s own
/// `weights_only` allowlist is far longer; this one covers what a `state_dict`
/// actually contains and refuses the rest rather than growing to meet each new
/// checkpoint.
pub const ALLOWED_GLOBALS: &[&str] = &[
    "collections.OrderedDict",
    "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter",
    "torch.Size",
    "torch.BFloat16Storage",
    "torch.BoolStorage",
    "torch.ByteStorage",
    "torch.CharStorage",
    "torch.ComplexDoubleStorage",
    "torch.ComplexFloatStorage",
    "torch.DoubleStorage",
    "torch.Float8_e4m3fnStorage",
    "torch.Float8_e5m2Storage",
    "torch.FloatStorage",
    "torch.HalfStorage",
    "torch.IntStorage",
    "torch.LongStorage",
    "torch.ShortStorage",
];

/// The dtype behind a storage class name.
pub fn storage_dtype(global: &str) -> Option<DType> {
    Some(match global {
        "torch.FloatStorage" => DType::F32,
        "torch.DoubleStorage" => DType::F64,
        "torch.HalfStorage" => DType::F16,
        "torch.BFloat16Storage" => DType::BF16,
        "torch.Float8_e4m3fnStorage" => DType::F8E4M3,
        "torch.Float8_e5m2Storage" => DType::F8E5M2,
        "torch.LongStorage" => DType::Int {
            w: 64,
            signed: true,
        },
        "torch.IntStorage" => DType::Int {
            w: 32,
            signed: true,
        },
        "torch.ShortStorage" => DType::Int {
            w: 16,
            signed: true,
        },
        "torch.CharStorage" => DType::Int { w: 8, signed: true },
        "torch.ByteStorage" => DType::Int {
            w: 8,
            signed: false,
        },
        "torch.BoolStorage" => DType::Bool,
        "torch.ComplexFloatStorage" => DType::Complex {
            re: Box::new(DType::F32),
        },
        "torch.ComplexDoubleStorage" => DType::Complex {
            re: Box::new(DType::F64),
        },
        _ => return None,
    })
}

/// One PyTorch tensor: a view over a storage, which is what §04.4's `strided`
/// layout already describes.
#[derive(Clone, Debug, PartialEq)]
pub struct TorchTensor {
    pub storage_key: String,
    pub dtype: DType,
    /// Elements, not bytes.
    pub storage_offset: u64,
    pub size: Vec<u64>,
    pub stride: Vec<u64>,
    pub requires_grad: bool,
}

impl TorchTensor {
    pub fn numel(&self) -> u64 {
        self.size.iter().product()
    }

    /// Whether the strides are the dense row-major ones for this shape, in
    /// which case the layout does not need to state them.
    pub fn is_contiguous(&self) -> bool {
        self.stride == Order::RowMajor.strides(&self.size)
    }

    /// Elements from `storage_offset` to the last one this view reaches.
    pub fn span(&self) -> u64 {
        if self.size.contains(&0) {
            return 0;
        }
        let mut last = 0u64;
        for (d, s) in self.size.iter().zip(&self.stride) {
            last += (d - 1) * s;
        }
        last + 1
    }
}

/// A Python value, restricted to what a checkpoint can contain.
#[derive(Clone, Debug)]
pub enum PyVal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PyVal>),
    List(std::rc::Rc<std::cell::RefCell<Vec<PyVal>>>),
    Dict(std::rc::Rc<std::cell::RefCell<Vec<(PyVal, PyVal)>>>),
    /// A resolved allowlisted symbol, before it is called.
    Global(&'static str),
    Storage {
        dtype: DType,
        key: String,
        location: String,
        numel: u64,
    },
    Tensor(TorchTensor),
}

impl PyVal {
    fn kind(&self) -> &'static str {
        match self {
            PyVal::None => "None",
            PyVal::Bool(_) => "bool",
            PyVal::Int(_) => "int",
            PyVal::Float(_) => "float",
            PyVal::Str(_) => "str",
            PyVal::Bytes(_) => "bytes",
            PyVal::Tuple(_) => "tuple",
            PyVal::List(_) => "list",
            PyVal::Dict(_) => "dict",
            PyVal::Global(_) => "symbol",
            PyVal::Storage { .. } => "storage",
            PyVal::Tensor(_) => "tensor",
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            PyVal::Int(i) if *i >= 0 => Some(*i as u64),
            PyVal::Bool(b) => Some(*b as u64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            PyVal::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A scalar's text form, for the `Foreign` object that keeps what OMNI does
    /// not model (I2).
    fn scalar_text(&self) -> Option<String> {
        Some(match self {
            PyVal::None => "null".into(),
            PyVal::Bool(b) => b.to_string(),
            PyVal::Int(i) => i.to_string(),
            PyVal::Float(f) => format!("{f:?}"),
            PyVal::Str(s) => s.clone(),
            PyVal::Bytes(b) => format!("<{} bytes>", b.len()),
            _ => return None,
        })
    }
}

/// The name of an opcode this build does not implement, so a refusal can say
/// *which*. Opcodes absent from this table are not pickle at all.
fn opcode_name(op: u8) -> &'static str {
    match op {
        b'F' => "FLOAT",
        b'I' => "INT",
        b'L' => "LONG",
        b'S' => "STRING",
        b'V' => "UNICODE",
        b'P' => "PERSID",
        b'g' => "GET",
        b'p' => "PUT",
        b'i' => "INST",
        b'o' => "OBJ",
        0x82 => "EXT1",
        0x83 => "EXT2",
        0x84 => "EXT4",
        0x8f => "EMPTY_SET",
        0x90 => "ADDITEMS",
        0x91 => "FROZENSET",
        0x92 => "NEWOBJ_EX",
        0x96 => "BYTEARRAY8",
        0x97 => "NEXT_BUFFER",
        0x98 => "READONLY_BUFFER",
        _ => "an unassigned opcode",
    }
}

/// The restricted unpickler.
///
/// It is a stack machine over a data language, not an interpreter. There is no
/// `import`, no attribute lookup, no `__reduce__` dispatch and no user code
/// path: `REDUCE` matches on six names and everything else is an error.
pub struct Unpickler<'a> {
    input: &'a [u8],
    pos: usize,
    stack: Vec<PyVal>,
    marks: Vec<usize>,
    memo: BTreeMap<u32, PyVal>,
    /// Storage keys seen, in the order they appeared.
    pub storages: Vec<StorageRef>,
    /// Symbols the file asked for and did not get, for the report.
    pub refused: Vec<String>,
    ops: u64,
}

/// A cap on the number of opcodes executed (§12.4: nothing unbounded driven by
/// untrusted input). A 1 000-tensor checkpoint's pickle is a few tens of
/// thousands of opcodes.
const MAX_OPS: u64 = 8_000_000;
const MAX_STACK: usize = 1 << 20;

impl<'a> Unpickler<'a> {
    pub fn new(input: &'a [u8]) -> Unpickler<'a> {
        Unpickler {
            input,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: BTreeMap::new(),
            storages: Vec::new(),
            refused: Vec::new(),
            ops: 0,
        }
    }

    fn byte(&mut self) -> Res<u8> {
        let b = *self
            .input
            .get(self.pos)
            .ok_or_else(|| Error::Pickle("ran off the end".into()))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self
            .input
            .get(self.pos..self.pos + n)
            .ok_or_else(|| Error::Pickle(format!("wanted {n} bytes, the stream ends")))?;
        self.pos += n;
        Ok(s)
    }

    fn len_prefixed(&mut self, width: usize) -> Res<&'a [u8]> {
        let raw = self.take(width)?;
        let mut n = 0u64;
        for (i, b) in raw.iter().enumerate() {
            n |= (*b as u64) << (8 * i);
        }
        if n > self.input.len() as u64 {
            return Err(Error::Pickle(format!(
                "a {n}-byte string in a {}-byte stream",
                self.input.len()
            )));
        }
        self.take(n as usize)
    }

    /// A newline-terminated field, used by the text opcodes.
    fn line(&mut self) -> Res<&'a str> {
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.input.len() {
            return Err(Error::Pickle("an unterminated line".into()));
        }
        let s = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|_| Error::Pickle("a line is not UTF-8".into()))?;
        self.pos += 1;
        Ok(s)
    }

    fn push(&mut self, v: PyVal) -> Res<()> {
        if self.stack.len() >= MAX_STACK {
            return Err(Error::Pickle("the stack grew past its bound".into()));
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self) -> Res<PyVal> {
        self.stack
            .pop()
            .ok_or_else(|| Error::Pickle("an opcode popped an empty stack".into()))
    }

    fn pop_mark(&mut self) -> Res<Vec<PyVal>> {
        let at = self
            .marks
            .pop()
            .ok_or_else(|| Error::Pickle("an opcode wanted a mark and there was none".into()))?;
        if at > self.stack.len() {
            return Err(Error::Pickle("the mark is above the stack".into()));
        }
        Ok(self.stack.split_off(at))
    }

    fn memo_put(&mut self, i: u32) -> Res<()> {
        let v = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| Error::Pickle("PUT on an empty stack".into()))?;
        self.memo.insert(i, v);
        Ok(())
    }

    fn memo_get(&mut self, i: u32) -> Res<()> {
        let v = self
            .memo
            .get(&i)
            .cloned()
            .ok_or_else(|| Error::Pickle(format!("GET {i} was never PUT")))?;
        self.push(v)
    }

    fn global(&mut self, module: &str, name: &str) -> Res<PyVal> {
        let full = format!("{module}.{name}");
        match ALLOWED_GLOBALS.iter().find(|g| **g == full) {
            Some(g) => Ok(PyVal::Global(g)),
            None => {
                self.refused.push(full.clone());
                Err(Error::Refused(format!(
                    "the pickle asks for `{full}`, which is not one of the {} \
                     tensor-reconstruction symbols this importer resolves. \
                     Loading it would mean running it",
                    ALLOWED_GLOBALS.len()
                )))
            }
        }
    }

    /// `REDUCE`: the only call mechanism, over six known functions.
    fn reduce(&mut self, f: PyVal, args: Vec<PyVal>) -> Res<PyVal> {
        let PyVal::Global(name) = f else {
            return Err(Error::Refused(format!(
                "REDUCE on a {}; only an allowlisted symbol is callable",
                f.kind()
            )));
        };
        match name {
            "collections.OrderedDict" => Ok(PyVal::Dict(Default::default())),
            "torch.Size" => Ok(match args.into_iter().next() {
                Some(PyVal::Tuple(t)) => PyVal::Tuple(t),
                Some(PyVal::List(l)) => PyVal::Tuple(l.borrow().clone()),
                _ => PyVal::Tuple(Vec::new()),
            }),
            "torch._utils._rebuild_parameter" => args
                .into_iter()
                .next()
                .ok_or_else(|| Error::Pickle("_rebuild_parameter with no data".into())),
            "torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2" => {
                self.rebuild_tensor(name, args)
            }
            other => Err(Error::Refused(format!(
                "`{other}` is allowlisted as a value but is not callable"
            ))),
        }
    }

    fn rebuild_tensor(&mut self, which: &str, args: Vec<PyVal>) -> Res<PyVal> {
        if args.len() < 4 {
            return Err(Error::Pickle(format!(
                "{which} takes at least 4 arguments, got {}",
                args.len()
            )));
        }
        let (dtype, key) = match &args[0] {
            PyVal::Storage { dtype, key, .. } => (dtype.clone(), key.clone()),
            other => {
                return Err(Error::Pickle(format!(
                    "{which}'s first argument is a {}, not a storage",
                    other.kind()
                )))
            }
        };
        let storage_offset = args[1].as_u64().ok_or_else(|| {
            Error::Pickle(format!("{which}'s storage offset is not a whole number"))
        })?;
        let ints = |v: &PyVal, what: &str| -> Res<Vec<u64>> {
            let items = match v {
                PyVal::Tuple(t) => t.clone(),
                PyVal::List(l) => l.borrow().clone(),
                other => {
                    return Err(Error::Pickle(format!(
                        "{which}'s {what} is a {}, not a tuple",
                        other.kind()
                    )))
                }
            };
            items
                .iter()
                .map(|x| match x {
                    PyVal::Int(i) if *i >= 0 => Ok(*i as u64),
                    PyVal::Int(i) => Err(Error::Unsupported(format!(
                        "{what} contains {i}; a negative stride is a reversed view and \
                         §04.4's `strided` layout has no encoding for one"
                    ))),
                    other => Err(Error::Pickle(format!(
                        "{what} contains a {}, not an int",
                        other.kind()
                    ))),
                })
                .collect()
        };
        let size = ints(&args[2], "size")?;
        let stride = ints(&args[3], "stride")?;
        if size.len() != stride.len() {
            return Err(Error::Pickle(format!(
                "{which} got {} sizes and {} strides",
                size.len(),
                stride.len()
            )));
        }
        let requires_grad = matches!(args.get(4), Some(PyVal::Bool(true)));
        Ok(PyVal::Tensor(TorchTensor {
            storage_key: key,
            dtype,
            storage_offset,
            size,
            stride,
            requires_grad,
        }))
    }

    /// `BINPERSID`: the storage references. The tuple is
    /// `('storage', <Class>Storage, key, location, numel)`.
    fn persistent(&mut self, pid: PyVal) -> Res<PyVal> {
        let PyVal::Tuple(t) = &pid else {
            return Err(Error::Pickle(format!(
                "a persistent id that is a {}, not a tuple",
                pid.kind()
            )));
        };
        if t.first().and_then(|x| x.as_str()) != Some("storage") || t.len() < 5 {
            return Err(Error::Refused(format!(
                "a persistent id this importer does not know: {:?}",
                t.first().map(|x| x.kind())
            )));
        }
        let PyVal::Global(cls) = &t[1] else {
            return Err(Error::Pickle("a storage id with no storage class".into()));
        };
        let dtype = storage_dtype(cls).ok_or_else(|| {
            Error::Unsupported(format!(
                "`{cls}` is allowlisted but carries no dtype this build maps; \
                 an untyped storage does not say what its elements are"
            ))
        })?;
        let key = t[2]
            .as_str()
            .ok_or_else(|| Error::Pickle("a storage key that is not text".into()))?
            .to_string();
        let location = t[3].as_str().unwrap_or("cpu").to_string();
        let numel = t[4]
            .as_u64()
            .ok_or_else(|| Error::Pickle("a storage length that is not a count".into()))?;
        if !self.storages.iter().any(|(k, _, _)| *k == key) {
            self.storages.push((key.clone(), dtype.clone(), numel));
        }
        Ok(PyVal::Storage {
            dtype,
            key,
            location,
            numel,
        })
    }

    /// `BUILD`: only for the containers this importer knows. Anything with a
    /// `__setstate__` is a Python object being reconstructed, which is the case
    /// §12.10 exists to refuse.
    fn build(&mut self, state: PyVal, obj: PyVal) -> Res<PyVal> {
        match (&obj, &state) {
            (_, PyVal::None) => Ok(obj),
            (PyVal::Dict(d), PyVal::Dict(s)) => {
                d.borrow_mut().extend(s.borrow().iter().cloned());
                Ok(obj)
            }
            _ => Err(Error::Refused(format!(
                "BUILD on a {} with a {} state: reconstructing an object needs its \
                 class, and running that is what this importer will not do",
                obj.kind(),
                state.kind()
            ))),
        }
    }

    pub fn run(&mut self) -> Res<PyVal> {
        loop {
            self.ops += 1;
            if self.ops > MAX_OPS {
                return Err(Error::Pickle(format!(
                    "over {MAX_OPS} opcodes; this is not a checkpoint"
                )));
            }
            let op = self.byte()?;
            match op {
                0x80 => {
                    // PROTO
                    let v = self.byte()?;
                    if v > 5 {
                        return Err(Error::Pickle(format!("pickle protocol {v} does not exist")));
                    }
                }
                0x95 => {
                    // FRAME: an advisory length, checked rather than trusted.
                    let n = u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes"));
                    if n > self.input.len() as u64 {
                        return Err(Error::Pickle("a frame longer than the stream".into()));
                    }
                }
                b'(' => self.marks.push(self.stack.len()),
                b'.' => break,
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_mark()?;
                }
                b'2' => {
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| Error::Pickle("DUP on an empty stack".into()))?;
                    self.push(v)?;
                }
                b'N' => self.push(PyVal::None)?,
                0x88 => self.push(PyVal::Bool(true))?,
                0x89 => self.push(PyVal::Bool(false))?,
                b'J' => {
                    let b = self.take(4)?;
                    self.push(PyVal::Int(
                        i32::from_le_bytes(b.try_into().expect("4 bytes")) as i64,
                    ))?
                }
                b'K' => {
                    let b = self.byte()?;
                    self.push(PyVal::Int(b as i64))?
                }
                b'M' => {
                    let b = self.take(2)?;
                    self.push(PyVal::Int(
                        u16::from_le_bytes(b.try_into().expect("2 bytes")) as i64,
                    ))?
                }
                0x8a | 0x8b => {
                    // LONG1 / LONG4: a little-endian two's-complement integer.
                    let n = if op == 0x8a {
                        self.byte()? as usize
                    } else {
                        let b = self.take(4)?;
                        u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize
                    };
                    let raw = self.take(n)?;
                    if n > 8 {
                        return Err(Error::Unsupported(format!(
                            "a {n}-byte integer; nothing in a checkpoint is that large"
                        )));
                    }
                    let mut v: i64 = if raw.last().is_some_and(|b| b & 0x80 != 0) {
                        -1
                    } else {
                        0
                    };
                    for (i, b) in raw.iter().enumerate() {
                        v = (v & !(0xFFi64 << (8 * i))) | ((*b as i64) << (8 * i));
                    }
                    self.push(PyVal::Int(v))?
                }
                b'G' => {
                    // BINFLOAT is big-endian, unlike everything else here.
                    let b = self.take(8)?;
                    self.push(PyVal::Float(f64::from_be_bytes(
                        b.try_into().expect("8 bytes"),
                    )))?
                }
                b'X' | 0x8c | 0x8d => {
                    let w = match op {
                        b'X' => 4,
                        0x8c => 1,
                        _ => 8,
                    };
                    let raw = self.len_prefixed(w)?;
                    let s = std::str::from_utf8(raw)
                        .map_err(|_| Error::Pickle("a string is not UTF-8".into()))?;
                    self.push(PyVal::Str(s.to_string()))?
                }
                b'T' | b'U' => {
                    // BINSTRING / SHORT_BINSTRING: protocol-2 bytes, which torch
                    // uses for storage keys.
                    let raw = self.len_prefixed(if op == b'T' { 4 } else { 1 })?;
                    match std::str::from_utf8(raw) {
                        Ok(s) => self.push(PyVal::Str(s.to_string()))?,
                        Err(_) => self.push(PyVal::Bytes(raw.to_vec()))?,
                    }
                }
                b'B' | b'C' | 0x8e => {
                    let w = match op {
                        b'B' => 4,
                        b'C' => 1,
                        _ => 8,
                    };
                    let raw = self.len_prefixed(w)?;
                    self.push(PyVal::Bytes(raw.to_vec()))?
                }
                b')' => self.push(PyVal::Tuple(Vec::new()))?,
                b']' => self.push(PyVal::List(Default::default()))?,
                b'}' => self.push(PyVal::Dict(Default::default()))?,
                b't' => {
                    let items = self.pop_mark()?;
                    self.push(PyVal::Tuple(items))?
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    if self.stack.len() < n {
                        return Err(Error::Pickle("TUPLEn under an empty stack".into()));
                    }
                    let at = self.stack.len() - n;
                    let items = self.stack.split_off(at);
                    self.push(PyVal::Tuple(items))?
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.push(PyVal::List(std::rc::Rc::new(std::cell::RefCell::new(
                        items,
                    ))))?
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(Error::Pickle("DICT with an odd number of items".into()));
                    }
                    let pairs = items
                        .chunks(2)
                        .map(|c| (c[0].clone(), c[1].clone()))
                        .collect();
                    self.push(PyVal::Dict(std::rc::Rc::new(std::cell::RefCell::new(
                        pairs,
                    ))))?
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.stack.last() {
                        Some(PyVal::List(l)) => l.borrow_mut().push(v),
                        _ => return Err(Error::Pickle("APPEND onto something else".into())),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.stack.last() {
                        Some(PyVal::List(l)) => l.borrow_mut().extend(items),
                        _ => return Err(Error::Pickle("APPENDS onto something else".into())),
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last() {
                        Some(PyVal::Dict(d)) => d.borrow_mut().push((k, v)),
                        _ => return Err(Error::Pickle("SETITEM onto something else".into())),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(Error::Pickle("SETITEMS with an odd count".into()));
                    }
                    match self.stack.last() {
                        Some(PyVal::Dict(d)) => {
                            let mut m = d.borrow_mut();
                            for c in items.chunks(2) {
                                m.push((c[0].clone(), c[1].clone()));
                            }
                        }
                        _ => return Err(Error::Pickle("SETITEMS onto something else".into())),
                    }
                }
                b'q' => {
                    let i = self.byte()? as u32;
                    self.memo_put(i)?
                }
                b'r' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes(b.try_into().expect("4 bytes"));
                    self.memo_put(i)?
                }
                0x94 => {
                    let i = self.memo.len() as u32;
                    self.memo_put(i)?
                }
                b'h' => {
                    let i = self.byte()? as u32;
                    self.memo_get(i)?
                }
                b'j' => {
                    let b = self.take(4)?;
                    let i = u32::from_le_bytes(b.try_into().expect("4 bytes"));
                    self.memo_get(i)?
                }
                b'c' => {
                    let module = self.line()?.to_string();
                    let name = self.line()?.to_string();
                    let g = self.global(&module, &name)?;
                    self.push(g)?
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (m, n) = match (module.as_str(), name.as_str()) {
                        (Some(m), Some(n)) => (m.to_string(), n.to_string()),
                        _ => return Err(Error::Pickle("STACK_GLOBAL without two names".into())),
                    };
                    let g = self.global(&m, &n)?;
                    self.push(g)?
                }
                b'R' => {
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let args = match args {
                        PyVal::Tuple(t) => t,
                        other => {
                            return Err(Error::Pickle(format!(
                                "REDUCE with a {} for arguments",
                                other.kind()
                            )))
                        }
                    };
                    let v = self.reduce(f, args)?;
                    self.push(v)?
                }
                0x81 => {
                    // NEWOBJ(cls, args): only the containers, never a class.
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let args = match args {
                        PyVal::Tuple(t) => t,
                        _ => Vec::new(),
                    };
                    let v = self.reduce(cls, args)?;
                    self.push(v)?
                }
                b'b' => {
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    let v = self.build(state, obj)?;
                    self.push(v)?
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let v = self.persistent(pid)?;
                    self.push(v)?
                }
                other => {
                    return Err(Error::Refused(format!(
                        "opcode 0x{other:02x} ({}) is not implemented; \
                         this unpickler runs a data language, not Python",
                        opcode_name(other)
                    )))
                }
            }
        }
        if self.stack.len() != 1 {
            return Err(Error::Pickle(format!(
                "STOP with {} values on the stack, not 1",
                self.stack.len()
            )));
        }
        self.pop()
    }
}

/// A storage the pickle referenced: its key, its element type, and how many
/// elements the file says it holds.
pub type StorageRef = (String, DType, u64);

/// Loads one pickle stream under the §12.10 restrictions.
pub fn unpickle(bytes: &[u8]) -> Res<(PyVal, Vec<StorageRef>)> {
    let mut u = Unpickler::new(bytes);
    let v = u.run()?;
    Ok((v, std::mem::take(&mut u.storages)))
}

// ------------------------------------------------------------- checkpoint --

/// A parsed checkpoint: the tensors it contains, in the order the pickle named
/// them, and everything else it contained, preserved rather than dropped.
pub struct Checkpoint {
    pub tensors: Vec<(String, TorchTensor)>,
    /// Non-tensor leaves, as `path` → text. Optimizer counters, epoch numbers,
    /// config scalars: things with no OMNI schema that would be lost if they
    /// were simply skipped (I2).
    pub other: Vec<(String, String)>,
    /// Structures this build cannot walk, named.
    pub skipped: Vec<Note>,
    pub version: Option<String>,
    /// The `<prefix>/` every member of the archive shares.
    pub prefix: String,
}

fn walk(path: &str, v: &PyVal, ck: &mut Checkpoint, depth: usize) -> Res<()> {
    if depth > 32 {
        return Err(Error::Unsupported(format!(
            "`{path}` nests more than 32 deep"
        )));
    }
    match v {
        PyVal::Tensor(t) => ck.tensors.push((path.to_string(), t.clone())),
        PyVal::Dict(d) => {
            for (k, val) in d.borrow().iter() {
                let key = match k {
                    PyVal::Str(s) => s.clone(),
                    PyVal::Int(i) => i.to_string(),
                    other => {
                        ck.skipped.push(Note {
                            item: path.to_string(),
                            reason: format!("a {} key", other.kind()),
                            action: "entry skipped and named here".into(),
                        });
                        continue;
                    }
                };
                let child = if path.is_empty() {
                    key
                } else {
                    format!("{path}.{key}")
                };
                walk(&child, val, ck, depth + 1)?;
            }
        }
        PyVal::List(l) => {
            for (i, val) in l.borrow().iter().enumerate() {
                walk(&format!("{path}.{i}"), val, ck, depth + 1)?;
            }
        }
        PyVal::Tuple(t) => {
            for (i, val) in t.iter().enumerate() {
                walk(&format!("{path}.{i}"), val, ck, depth + 1)?;
            }
        }
        other => match other.scalar_text() {
            Some(text) => ck.other.push((path.to_string(), text)),
            None => ck.skipped.push(Note {
                item: path.to_string(),
                reason: format!("a {}", other.kind()),
                action: "not a tensor and not a scalar; named rather than dropped".into(),
            }),
        },
    }
    Ok(())
}

/// The legacy pre-1.6 `torch.save` magic. It is a bare pickle rather than an
/// archive, and its storages are interleaved with the stream.
const LEGACY_MAGIC: u64 = 0x1950_a86a_20f9_469c;

/// Reads the archive framing and the pickle, without touching a byte of tensor
/// payload.
pub fn parse(bytes: &[u8]) -> Res<(Checkpoint, Zip<'_>)> {
    if bytes.len() < 4 {
        return Err(Error::Zip("too short to be anything".into()));
    }
    if &bytes[..2] != b"PK" {
        // A protocol-2 pickle starts 0x80 0x02. The legacy format's first value
        // is the magic number, and saying so is more useful than "not a ZIP".
        if bytes[0] == 0x80 {
            let mut probe = Unpickler::new(bytes);
            let looks_legacy = matches!(probe.run(), Ok(PyVal::Int(m)) if m as u64 == LEGACY_MAGIC)
                || bytes.len() > 16;
            if looks_legacy {
                return Err(Error::Unsupported(
                    "this is a pre-1.6 `torch.save` file: a bare pickle with its storages \
                     interleaved into the stream, not a ZIP archive. Re-save it with a \
                     current PyTorch (`torch.save(sd, path)`), or better, convert it to \
                     safetensors"
                        .into(),
                ));
            }
        }
        return Err(Error::Zip(
            "does not begin with a ZIP local file header".into(),
        ));
    }
    let zip = Zip::open(bytes)?;

    // Every member shares one prefix, which is the archive's name.
    let pkl = zip
        .entries
        .iter()
        .find(|e| e.name.ends_with("/data.pkl"))
        .ok_or_else(|| {
            Error::Zip("no `data.pkl`; this archive is not a torch checkpoint".into())
        })?;
    let prefix = pkl.name[..pkl.name.len() - "data.pkl".len()].to_string();

    // §12.10: the byte order matters and is not guessed. Every dtype below is
    // read little-endian.
    if let Some(bo) = zip.find(&format!("{prefix}byteorder")) {
        let raw = zip.read(bo)?;
        let text = String::from_utf8_lossy(&raw).trim().to_string();
        if text != "little" {
            return Err(Error::Unsupported(format!(
                "the archive declares `{text}` byte order; this importer reads little-endian \
                 storages and will not byte-swap silently"
            )));
        }
    }

    let raw = zip.read(pkl)?;
    let (root, _storages) = unpickle(&raw)?;

    let version = zip
        .find(&format!("{prefix}version"))
        .and_then(|e| zip.read(e).ok())
        .map(|b| String::from_utf8_lossy(&b).trim().to_string());

    let mut ck = Checkpoint {
        tensors: Vec::new(),
        other: Vec::new(),
        skipped: Vec::new(),
        version,
        prefix,
    };
    walk("", &root, &mut ck, 0)?;
    if ck.tensors.is_empty() {
        return Err(Error::Unsupported(
            "the pickle contains no tensors; there is nothing here to import".into(),
        ));
    }
    Ok((ck, zip))
}

// ---------------------------------------------------------------- import --

fn elem_bytes(d: &DType) -> u64 {
    d.packed_bytes(1)
}

/// The bytes a tensor's view actually reaches, sliced out of its storage.
///
/// PyTorch shares one storage between views — tied embeddings are the common
/// case — and OMNI could model that with a layout offset. It does not, because
/// §04.4's `strided` sizing rule (R-T02) makes a literal's chunk exactly as long
/// as its view spans, so a view into the middle of a larger buffer would be
/// reported invalid. Slicing gives every tensor its own bytes; identical slices
/// still dedup, because the container is content-addressed.
fn slice_for(t: &TorchTensor, storage: &[u8]) -> Res<Vec<u8>> {
    let esz = elem_bytes(&t.dtype);
    let span = t.span();
    let start = t
        .storage_offset
        .checked_mul(esz)
        .ok_or_else(|| Error::Unsupported("a storage offset that overflows".into()))?;
    let len = span
        .checked_mul(esz)
        .ok_or_else(|| Error::Unsupported("a view longer than memory".into()))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::Unsupported("a view that overflows its storage".into()))?;
    storage
        .get(start as usize..end as usize)
        .map(|s| s.to_vec())
        .ok_or_else(|| {
            Error::Core(format!(
                "a tensor reaches bytes {start}..{end} of a {}-byte storage",
                storage.len()
            ))
        })
}

fn layout_for(t: &TorchTensor) -> Layout {
    if t.is_contiguous() {
        Layout::Strided {
            order: Order::RowMajor,
            strides: None,
            offset: 0,
        }
    } else {
        // The strides are kept rather than the tensor being materialized into a
        // dense array: a transposed weight *is* a transposed view, and §04.4
        // says so directly.
        Layout::Strided {
            order: Order::RowMajor,
            strides: Some(t.stride.clone()),
            offset: 0,
        }
    }
}

/// The `Foreign` object that preserves what has no OMNI schema (I2).
fn foreign_object(ck: &Checkpoint, path: &str, source: &Digest) -> Object {
    let mut kept = std::collections::BTreeMap::new();
    for (k, v) in &ck.other {
        kept.insert(k.clone(), crate::json::Value::Str(v.clone()));
    }
    let doc = crate::json::Value::Object(kept);
    Object::structure(
        otype::FOREIGN,
        &Value::map(vec![
            ("t", Value::text("omni.core/foreign")),
            ("v", Value::U(1)),
            ("format", Value::text("pytorch")),
            ("path", Value::text(path.to_string())),
            ("source", Value::Bytes(source.to_vec())),
            (
                "note",
                Value::text(
                    "non-tensor leaves of the checkpoint's pickle, as text. Anything needing a \
                 Python class to reconstruct was refused rather than represented (§12.10)",
                ),
            ),
            ("doc", Value::text(doc.encode())),
        ]),
    )
}

/// One tensor ready to go into the builder: name, dtype, shape, layout, bytes.
type Slice = (String, DType, Vec<u64>, Layout, Vec<u8>);

fn assemble(
    ck: &Checkpoint,
    slices: &[Slice],
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
    for (name, dtype, shape, layout, data) in slices {
        b = b.tensor(TensorSpec {
            name: name.clone(),
            shape: shape.clone(),
            dtype: dtype.clone(),
            axes: None,
            // A `state_dict` key is a convention, not a declaration. Calling
            // every entry a weight would be a guess.
            semantic: "".into(),
            data: data.clone(),
            layout: Some(layout.clone()),
        });
    }
    b = b.asset("provenance", otype::PROVENANCE, report.to_value());

    let foreign = (!ck.other.is_empty())
        .then(|| foreign_object(ck, &opts.source_path, &report.source_digest));
    if let Some(obj) = &foreign {
        let d = obj.digest(opts.hash);
        b = b.manifest_key(
            "foreign",
            Value::Array(vec![Value::Array(vec![
                Value::U(otype::FOREIGN as u64),
                Value::Bytes(d.to_vec()),
            ])]),
        );
    }
    let (mut objects, root) = b.build();
    objects.extend(foreign);
    (objects, root)
}

/// Imports a `torch.save` checkpoint into an OMNI object graph.
pub fn import(bytes: &[u8], opts: &ImportOpts) -> Res<Imported> {
    let (ck, zip) = parse(bytes)?;

    let mut report = Fidelity {
        format: "pytorch",
        importer: IMPORTER,
        source_path: opts.source_path.clone(),
        source_digest: opts.hash.digest(bytes),
        source_size: bytes.len() as u64,
        lossless: true,
        represented: vec![
            "tensors".into(),
            "dtypes".into(),
            "shapes".into(),
            "strides".into(),
        ],
        ..Default::default()
    };

    // §12.10 clause 3: the report says the source was an unsafe format, and
    // what the restriction actually was.
    report.assumptions.push(Note {
        item: "pickle".into(),
        reason: "the source is a code-bearing format (§12.10)".into(),
        action: format!(
            "read by a restricted unpickler: {} symbols resolvable, no call mechanism \
             beyond tensor reconstruction, no object reconstruction",
            ALLOWED_GLOBALS.len()
        ),
    });
    report.assumptions.push(Note {
        item: "license".into(),
        reason: "a torch checkpoint declares none".into(),
        action: match &opts.license {
            Some(spdx) => format!("supplied by the caller as `{spdx}`"),
            None => "field omitted".into(),
        },
    });
    report.assumptions.push(Note {
        item: "arch.family".into(),
        reason: "a state_dict's keys are a convention, not a declaration".into(),
        action: match &opts.arch {
            Some((family, _)) => format!("supplied by the caller as `{family}`"),
            None => "field omitted".into(),
        },
    });
    if let Some(v) = &ck.version {
        report
            .represented
            .push(format!("serialization version {v}"));
    }
    for note in &ck.skipped {
        report.lossless = false;
        report.unrepresented.push(note.clone());
    }
    if !ck.other.is_empty() {
        report.represented.push("non-tensor leaves".into());
        report.unrepresented.push(Note {
            item: format!("{} non-tensor leaf value(s)", ck.other.len()),
            reason: "optimizer counters, epochs and config scalars have no OMNI schema".into(),
            action: "preserved verbatim in a Foreign object".into(),
        });
    }

    // Read each storage once, then cut every view out of it.
    let mut storages: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut slices: Vec<Slice> = Vec::new();
    let mut shared = 0usize;
    let mut seen_keys: BTreeMap<String, usize> = BTreeMap::new();
    for (name, t) in &ck.tensors {
        let member = format!("{}data/{}", ck.prefix, t.storage_key);
        if !storages.contains_key(&t.storage_key) {
            let e = zip.find(&member).ok_or_else(|| {
                Error::Zip(format!(
                    "`{name}` names storage `{}`, and `{member}` is not in the archive",
                    t.storage_key
                ))
            })?;
            storages.insert(t.storage_key.clone(), zip.read(e)?.into_owned());
        }
        let count = seen_keys.entry(t.storage_key.clone()).or_insert(0);
        *count += 1;
        if *count == 2 {
            shared += 1;
        }
        let storage = &storages[&t.storage_key];
        let data = slice_for(t, storage)?;
        if !t.is_contiguous() {
            report.assumptions.push(Note {
                item: name.clone(),
                reason: format!("a non-contiguous view, strides {:?}", t.stride),
                action: "kept as a `strided` layout with explicit strides, not densified".into(),
            });
        }
        slices.push((
            name.clone(),
            t.dtype.clone(),
            t.size.clone(),
            layout_for(t),
            data,
        ));
    }
    if shared > 0 {
        report.assumptions.push(Note {
            item: format!("{shared} shared storage(s)"),
            reason: "PyTorch tensors are views, and more than one names the same buffer \
                     (tied embeddings, most often)"
                .into(),
            action: "each view gets its own bytes; identical ones dedup by digest".into(),
        });
    }

    // I4: every tensor is read back out of a graph built from this file and
    // compared with the bytes that went in. Without this, "lossless" is an
    // intention rather than a finding.
    let (probe, probe_root) = assemble(&ck, &slices, opts, &report);
    let store = store_of(&probe, opts.hash);
    let ctx = Ctx::new(&store);
    let table = table_of(&probe, &probe_root, opts.hash)?;
    for (name, _dtype, _shape, _layout, data) in &slices {
        let r = table
            .tensors
            .get(name)
            .ok_or_else(|| Error::Core(format!("`{name}` did not reach the table")))?;
        let raw = crate::store::Store::resolve(&store, &r.1)
            .map_err(|e| Error::Core(e.to_string()))?
            .ok_or_else(|| Error::Core("a descriptor went missing".into()))?;
        let desc = TensorDesc::from_value(
            &crate::cbor::decode(&raw).map_err(|e| Error::Core(e.to_string()))?,
        )
        .map_err(|e| Error::Core(e.to_string()))?;
        let crate::expr::Expr::Literal { chunks, .. } = &desc.value else {
            return Err(Error::Core(format!(
                "`{name}` is not a literal after import"
            )));
        };
        let got = ctx
            .chunk_bytes(chunks)
            .map_err(|e| Error::Core(e.to_string()))?;
        if &got != data {
            return Err(Error::Core(format!(
                "I4: `{name}` did not survive import byte for byte ({} in, {} out)",
                data.len(),
                got.len()
            )));
        }
        report.verified_tensors += 1;
        report.verified_bytes += data.len() as u64;
    }

    let (objects, root) = assemble(&ck, &slices, opts, &report);
    Ok(Imported {
        objects,
        root,
        report,
    })
}

fn store_of(objects: &[Object], hash: HashAlgo) -> crate::store::MemoryStore {
    let mut store = crate::store::MemoryStore::new(hash);
    for o in objects {
        let _ = crate::store::WritableStore::put(&mut store, &o.payload);
    }
    store
}

fn table_of(objects: &[Object], root: &Digest, hash: HashAlgo) -> Res<TensorTable> {
    let decode = |d: &Digest| -> Res<Value> {
        let bytes = objects
            .iter()
            .find(|o| &o.digest(hash) == d)
            .map(|o| o.payload.clone())
            .ok_or_else(|| Error::Core("a just-built object is missing".into()))?;
        crate::cbor::decode(&bytes).map_err(|e| Error::Core(e.to_string()))
    };
    let manifest = decode(root)?;
    let model_ref = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(|v| crate::expr::parse_ref_value(v).ok())
        .ok_or_else(|| Error::Core("the built manifest has no model asset".into()))?;
    let model = decode(&model_ref.1)?;
    let tt = model
        .get("tensors")
        .and_then(|v| crate::expr::parse_ref_value(v).ok())
        .ok_or_else(|| Error::Core("the built model has no tensor table".into()))?;
    TensorTable::from_value(&decode(&tt.1)?).map_err(|e| Error::Core(e.to_string()))
}

// ------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a minimal pickle writer, so the tests build their own input -------

    fn proto2() -> Vec<u8> {
        vec![0x80, 0x02]
    }

    fn bin_str(out: &mut Vec<u8>, s: &str) {
        // SHORT_BINSTRING, which is what torch's protocol-2 pickles use.
        out.push(b'U');
        out.push(s.len() as u8);
        out.extend_from_slice(s.as_bytes());
    }

    fn unicode(out: &mut Vec<u8>, s: &str) {
        out.push(b'X');
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn int(out: &mut Vec<u8>, n: u64) {
        if n < 256 {
            out.push(b'K');
            out.push(n as u8);
        } else {
            out.push(b'J');
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
    }

    fn global(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }

    /// `(storage, FloatStorage, key, 'cpu', numel)` followed by BINPERSID.
    fn storage(out: &mut Vec<u8>, cls: &str, key: &str, numel: u64) {
        out.push(b'(');
        bin_str(out, "storage");
        global(out, "torch", cls);
        bin_str(out, key);
        bin_str(out, "cpu");
        int(out, numel);
        out.push(b't');
        out.push(b'Q');
    }

    fn tuple_of_ints(out: &mut Vec<u8>, xs: &[u64]) {
        out.push(b'(');
        for x in xs {
            int(out, *x);
        }
        out.push(b't');
    }

    /// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad, {})`.
    fn tensor(
        out: &mut Vec<u8>,
        cls: &str,
        key: &str,
        numel: u64,
        offset: u64,
        size: &[u64],
        stride: &[u64],
    ) {
        global(out, "torch._utils", "_rebuild_tensor_v2");
        out.push(b'(');
        storage(out, cls, key, numel);
        int(out, offset);
        tuple_of_ints(out, size);
        tuple_of_ints(out, stride);
        out.push(0x89); // NEWFALSE
        out.push(b'}'); // an empty hooks dict
        out.push(b't');
        out.push(b'R');
    }

    /// A whole `data.pkl`: an OrderedDict of name → tensor.
    /// name, storage class, storage key, storage numel, offset, size, stride.
    type Entry<'a> = (&'a str, &'a str, &'a str, u64, u64, Vec<u64>, Vec<u64>);

    fn state_dict(entries: &[Entry<'_>]) -> Vec<u8> {
        let mut p = proto2();
        global(&mut p, "collections", "OrderedDict");
        p.push(b')'); // EMPTY_TUPLE
        p.push(b'R'); // REDUCE -> an ordered dict
        p.push(b'('); // MARK for SETITEMS
        for (name, cls, key, numel, offset, size, stride) in entries {
            unicode(&mut p, name);
            tensor(&mut p, cls, key, *numel, *offset, size, stride);
        }
        p.push(b'u'); // SETITEMS
        p.push(b'.'); // STOP
        p
    }

    // ---- a minimal ZIP writer ---------------------------------------------

    struct ZipWriter {
        out: Vec<u8>,
        central: Vec<u8>,
        count: u16,
    }

    impl ZipWriter {
        fn new() -> ZipWriter {
            ZipWriter {
                out: Vec::new(),
                central: Vec::new(),
                count: 0,
            }
        }

        fn add(&mut self, name: &str, data: &[u8]) {
            let off = self.out.len() as u32;
            self.out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            self.out.extend_from_slice(&[0; 4]); // version, flags
            self.out.extend_from_slice(&0u16.to_le_bytes()); // stored
            self.out.extend_from_slice(&[0; 8]); // time, date, crc — unchecked here
            self.out
                .extend_from_slice(&(data.len() as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(data.len() as u32).to_le_bytes());
            self.out
                .extend_from_slice(&(name.len() as u16).to_le_bytes());
            self.out.extend_from_slice(&0u16.to_le_bytes());
            self.out.extend_from_slice(name.as_bytes());
            self.out.extend_from_slice(data);

            self.central
                .extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            self.central.extend_from_slice(&[0; 6]);
            self.central.extend_from_slice(&0u16.to_le_bytes());
            self.central.extend_from_slice(&[0; 8]);
            self.central
                .extend_from_slice(&(data.len() as u32).to_le_bytes());
            self.central
                .extend_from_slice(&(data.len() as u32).to_le_bytes());
            self.central
                .extend_from_slice(&(name.len() as u16).to_le_bytes());
            self.central.extend_from_slice(&[0; 8]);
            self.central.extend_from_slice(&[0; 4]);
            self.central.extend_from_slice(&off.to_le_bytes());
            self.central.extend_from_slice(name.as_bytes());
            self.count += 1;
        }

        fn finish(mut self) -> Vec<u8> {
            let cd_off = self.out.len() as u32;
            let cd_len = self.central.len() as u32;
            self.out.extend_from_slice(&self.central);
            self.out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
            self.out.extend_from_slice(&[0; 4]);
            self.out.extend_from_slice(&self.count.to_le_bytes());
            self.out.extend_from_slice(&self.count.to_le_bytes());
            self.out.extend_from_slice(&cd_len.to_le_bytes());
            self.out.extend_from_slice(&cd_off.to_le_bytes());
            self.out.extend_from_slice(&0u16.to_le_bytes());
            self.out
        }
    }

    fn f32_bytes(n: usize, seed: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed + i as f32).to_le_bytes())
            .collect()
    }

    /// Two 2×3 f32 tensors, one of them a view of the other's storage.
    fn toy_archive() -> Vec<u8> {
        let mut z = ZipWriter::new();
        z.add(
            "toy/data.pkl",
            &state_dict(&[
                (
                    "a.weight",
                    "FloatStorage",
                    "0",
                    12,
                    0,
                    vec![2, 3],
                    vec![3, 1],
                ),
                (
                    "b.weight",
                    "FloatStorage",
                    "0",
                    12,
                    6,
                    vec![2, 3],
                    vec![3, 1],
                ),
                ("c.weight", "HalfStorage", "1", 4, 0, vec![2, 2], vec![2, 1]),
            ]),
        );
        z.add("toy/data/0", &f32_bytes(12, 1.0));
        z.add(
            "toy/data/1",
            &(0..4u16)
                .flat_map(|i| (i * 7).to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        z.add("toy/version", b"3\n");
        z.finish()
    }

    #[test]
    fn a_toy_archive_parses_into_tensors_and_their_views() {
        let bytes = toy_archive();
        let (ck, _zip) = parse(&bytes).map_err(|e| e.to_string()).expect("parses");
        assert_eq!(ck.prefix, "toy/");
        assert_eq!(ck.version.as_deref(), Some("3"));
        assert_eq!(ck.tensors.len(), 3);
        let (name, t) = &ck.tensors[0];
        assert_eq!(name, "a.weight");
        assert_eq!(t.size, vec![2, 3]);
        assert_eq!(t.stride, vec![3, 1]);
        assert_eq!(t.dtype, DType::F32);
        assert!(t.is_contiguous());
        // The second tensor is the same storage at a different offset, which is
        // exactly what a PyTorch view is.
        assert_eq!(ck.tensors[1].1.storage_key, "0");
        assert_eq!(ck.tensors[1].1.storage_offset, 6);
        assert_eq!(ck.tensors[2].1.dtype, DType::F16);
    }

    #[test]
    fn the_views_bytes_are_the_right_slice_of_their_storage() {
        let bytes = toy_archive();
        let opts = ImportOpts {
            name: "acme/toy".into(),
            ..Default::default()
        };
        let out = import(&bytes, &opts).expect("imports");
        assert_eq!(out.report.verified_tensors, 3);
        // 12 f32 + 4 f16 across three views: a and b are 24 bytes each, c is 8.
        assert_eq!(out.report.verified_bytes, 24 + 24 + 8);
        assert!(out.report.lossless);
        // Two views of one storage is reported, not silently duplicated.
        assert!(out
            .report
            .assumptions
            .iter()
            .any(|n| n.item.contains("shared storage")));
    }

    #[test]
    fn a_transposed_view_keeps_its_strides_instead_of_being_densified() {
        let mut z = ZipWriter::new();
        // A 3×2 view of a 2×3 buffer with swapped strides: `w.t()`.
        z.add(
            "m/data.pkl",
            &state_dict(&[("w", "FloatStorage", "0", 6, 0, vec![3, 2], vec![1, 3])]),
        );
        z.add("m/data/0", &f32_bytes(6, 1.0));
        let bytes = z.finish();
        let (ck, _) = parse(&bytes).map_err(|e| e.to_string()).expect("parses");
        let t = &ck.tensors[0].1;
        assert!(!t.is_contiguous());
        assert_eq!(t.span(), 6);
        let out = import(&bytes, &ImportOpts::default()).expect("imports");
        assert!(out
            .report
            .assumptions
            .iter()
            .any(|n| n.reason.contains("non-contiguous")));
        assert_eq!(out.report.verified_tensors, 1);
    }

    #[test]
    fn a_global_outside_the_allowlist_is_a_hard_error_naming_it() {
        // The whole point of §12.10. `posix.system` is the classic payload.
        let mut p = proto2();
        global(&mut p, "posix", "system");
        bin_str(&mut p, "echo pwned");
        p.push(b'\x85'); // TUPLE1
        p.push(b'R'); // REDUCE
        p.push(b'.');
        let err = unpickle(&p).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("posix.system"), "{msg}");
        assert!(msg.contains("running it"), "{msg}");
        assert!(matches!(err, Error::Refused(_)));
    }

    #[test]
    fn every_other_dangerous_global_is_refused_the_same_way() {
        for (module, name) in [
            ("os", "system"),
            ("subprocess", "Popen"),
            ("builtins", "eval"),
            ("__builtin__", "exec"),
            ("torch", "load"),
            ("torch.storage", "_load_from_bytes"),
            ("runpy", "_run_code"),
            ("operator", "attrgetter"),
        ] {
            let mut p = proto2();
            global(&mut p, module, name);
            p.push(b'.');
            let err = unpickle(&p).expect_err(&format!("{module}.{name} must be refused"));
            assert!(err.to_string().contains(name), "{err}");
        }
    }

    #[test]
    fn an_opcode_this_build_does_not_run_is_refused_by_name() {
        // INST is the protocol-0 way to build an arbitrary object.
        let mut p = proto2();
        p.push(b'i');
        let err = unpickle(&p).expect_err("must refuse");
        assert!(err.to_string().contains("INST"), "{err}");
        // And the extension registry, which resolves to arbitrary classes.
        for (op, name) in [(0x82u8, "EXT1"), (0x83, "EXT2"), (0x84, "EXT4")] {
            let mut p = proto2();
            p.push(op);
            let err = unpickle(&p).expect_err("must refuse");
            assert!(err.to_string().contains(name), "{err}");
        }
    }

    #[test]
    fn build_on_anything_but_a_dict_is_refused() {
        // BUILD is how `__setstate__` gets called, which needs the class.
        let mut p = proto2();
        global(&mut p, "torch", "Size");
        p.push(b')');
        p.push(b'R');
        p.push(b'}'); // a dict of state
        p.push(b'b'); // BUILD
        p.push(b'.');
        let err = unpickle(&p).expect_err("must refuse");
        assert!(err.to_string().contains("BUILD"), "{err}");
    }

    #[test]
    fn the_pre_16_format_is_named_rather_than_reported_as_not_a_zip() {
        let mut p = proto2();
        // The legacy magic, as a LONG1.
        p.push(0x8a);
        p.push(8);
        p.extend_from_slice(&LEGACY_MAGIC.to_le_bytes());
        p.push(b'.');
        p.extend_from_slice(&[0u8; 64]); // whatever follows
        let err = parse(&p).map(|_| ()).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("pre-1.6"), "{msg}");
        assert!(msg.contains("safetensors"), "{msg}");
    }

    #[test]
    fn a_truncated_archive_is_an_error_at_every_length() {
        let bytes = toy_archive();
        for cut in 1..bytes.len() {
            // Never a panic; the point of §12.4.
            let _ = parse(&bytes[..cut]);
        }
    }

    #[test]
    fn a_corrupted_pickle_is_an_error_rather_than_a_panic() {
        let bytes = toy_archive();
        // Flip one byte inside the pickle member, at many positions.
        for at in 40..140.min(bytes.len()) {
            let mut m = bytes.clone();
            m[at] ^= 0xFF;
            let _ = parse(&m);
        }
    }

    #[test]
    fn a_pickle_that_never_stops_hits_the_opcode_bound() {
        // A stream of NONE with no STOP runs off the end rather than forever;
        // a stream that loops through the memo would hit MAX_OPS. Both are
        // errors, and neither hangs.
        let mut p = proto2();
        p.extend(std::iter::repeat_n(b'N', 10_000));
        let err = unpickle(&p).expect_err("no STOP");
        assert!(err.to_string().contains("ran off the end"), "{err}");
    }

    #[test]
    fn a_storage_a_tensor_names_and_the_archive_lacks_is_reported() {
        let mut z = ZipWriter::new();
        z.add(
            "m/data.pkl",
            &state_dict(&[("w", "FloatStorage", "7", 6, 0, vec![2, 3], vec![3, 1])]),
        );
        z.add("m/data/0", &f32_bytes(6, 1.0));
        let bytes = z.finish();
        let err = import(&bytes, &ImportOpts::default())
            .map(|_| ())
            .expect_err("storage 7 is absent");
        assert!(err.to_string().contains("m/data/7"), "{err}");
    }

    #[test]
    fn a_view_that_runs_past_its_storage_is_caught() {
        let mut z = ZipWriter::new();
        z.add(
            "m/data.pkl",
            &state_dict(&[("w", "FloatStorage", "0", 6, 4, vec![2, 3], vec![3, 1])]),
        );
        z.add("m/data/0", &f32_bytes(6, 1.0));
        let bytes = z.finish();
        let err = import(&bytes, &ImportOpts::default())
            .map(|_| ())
            .expect_err("the view is too long");
        assert!(err.to_string().contains("24-byte storage"), "{err}");
    }

    #[test]
    fn non_tensor_leaves_are_preserved_rather_than_dropped() {
        let mut p = proto2();
        p.push(b'}'); // EMPTY_DICT
        p.push(b'(');
        unicode(&mut p, "w");
        tensor(&mut p, "FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
        unicode(&mut p, "epoch");
        int(&mut p, 7);
        unicode(&mut p, "note");
        unicode(&mut p, "trained on nothing");
        p.push(b'u');
        p.push(b'.');
        let mut z = ZipWriter::new();
        z.add("m/data.pkl", &p);
        z.add("m/data/0", &f32_bytes(6, 1.0));
        let bytes = z.finish();
        let (ck, _) = parse(&bytes).map_err(|e| e.to_string()).expect("parses");
        assert_eq!(ck.tensors.len(), 1);
        assert_eq!(
            ck.other,
            vec![
                ("epoch".to_string(), "7".to_string()),
                ("note".to_string(), "trained on nothing".to_string())
            ]
        );
        let out = import(&bytes, &ImportOpts::default()).expect("imports");
        // The Foreign object is in the graph, and the keys are in it.
        let foreign = out
            .objects
            .iter()
            .find(|o| o.otype == otype::FOREIGN)
            .expect("a Foreign object");
        let v = crate::cbor::decode(&foreign.payload).expect("decodes");
        let doc = v.get("doc").and_then(|x| x.as_str()).expect("a doc");
        assert!(doc.contains("epoch"), "{doc}");
        assert!(doc.contains("trained on nothing"), "{doc}");
    }

    #[test]
    fn a_nested_checkpoint_flattens_with_the_path_as_the_name() {
        let mut p = proto2();
        p.push(b'}');
        p.push(b'(');
        unicode(&mut p, "model");
        p.push(b'}');
        p.push(b'(');
        unicode(&mut p, "w");
        tensor(&mut p, "FloatStorage", "0", 6, 0, &[2, 3], &[3, 1]);
        p.push(b'u');
        p.push(b'u');
        p.push(b'.');
        let mut z = ZipWriter::new();
        z.add("m/data.pkl", &p);
        z.add("m/data/0", &f32_bytes(6, 1.0));
        let bytes = z.finish();
        let (ck, _) = parse(&bytes).map_err(|e| e.to_string()).expect("parses");
        assert_eq!(ck.tensors[0].0, "model.w");
    }

    #[test]
    fn zip64_sizes_are_read_from_the_extra_field() {
        // Build an archive whose central-directory entry saturates its 32-bit
        // size fields and puts the real ones in a 0x0001 extra field. This is
        // the normal case for a real checkpoint, so it cannot be untested.
        let data = f32_bytes(6, 1.0);
        let pkl = state_dict(&[("w", "FloatStorage", "0", 6, 0, vec![2, 3], vec![3, 1])]);
        let mut z = ZipWriter::new();
        z.add("m/data.pkl", &pkl);
        z.add("m/data/0", &data);
        let mut bytes = z.finish();

        // Rewrite the second central-directory entry to the Zip64 form.
        let cd = bytes
            .windows(4)
            .rposition(|w| w == 0x0201_4b50u32.to_le_bytes())
            .expect("a central directory entry");
        let name_len = u16::from_le_bytes([bytes[cd + 28], bytes[cd + 29]]) as usize;
        let real_size = u32::from_le_bytes(bytes[cd + 24..cd + 28].try_into().unwrap());
        let real_comp = u32::from_le_bytes(bytes[cd + 20..cd + 24].try_into().unwrap());
        bytes[cd + 20..cd + 24].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        bytes[cd + 24..cd + 28].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let mut extra = Vec::new();
        extra.extend_from_slice(&0x0001u16.to_le_bytes());
        extra.extend_from_slice(&16u16.to_le_bytes());
        extra.extend_from_slice(&(real_size as u64).to_le_bytes());
        extra.extend_from_slice(&(real_comp as u64).to_le_bytes());
        bytes[cd + 30..cd + 32].copy_from_slice(&(extra.len() as u16).to_le_bytes());
        let at = cd + 46 + name_len;
        // The extra field goes after the name; everything after it shifts, and
        // the EOCD's central-directory length has to move with it.
        let mut rebuilt = bytes[..at].to_vec();
        rebuilt.extend_from_slice(&extra);
        rebuilt.extend_from_slice(&bytes[at..]);
        let eocd = rebuilt
            .windows(4)
            .rposition(|w| w == 0x0605_4b50u32.to_le_bytes())
            .expect("an EOCD");
        let cd_len = u32::from_le_bytes(rebuilt[eocd + 12..eocd + 16].try_into().unwrap());
        rebuilt[eocd + 12..eocd + 16].copy_from_slice(&(cd_len + extra.len() as u32).to_le_bytes());

        let zip = Zip::open(&rebuilt).expect("opens");
        let e = zip.find("m/data/0").expect("the storage");
        assert_eq!(e.size, data.len() as u64);
        assert_eq!(zip.read(e).expect("reads").as_ref(), data.as_slice());
    }

    #[test]
    fn the_allowlist_holds_only_what_it_claims() {
        // A list that grows by accident stops being a security property, so its
        // contents are asserted rather than assumed.
        assert_eq!(ALLOWED_GLOBALS.len(), 19);
        for g in ALLOWED_GLOBALS {
            assert!(
                g.starts_with("torch.") || *g == "collections.OrderedDict",
                "{g} is neither torch nor the one container"
            );
        }
        // Every storage class in the list maps to a dtype, and nothing else
        // pretends to.
        let storages: Vec<_> = ALLOWED_GLOBALS
            .iter()
            .filter(|g| g.ends_with("Storage"))
            .collect();
        assert_eq!(storages.len(), 14);
        for g in &storages {
            assert!(storage_dtype(g).is_some(), "{g} has no dtype");
        }
        assert!(storage_dtype("torch.UntypedStorage").is_none());
    }
}
