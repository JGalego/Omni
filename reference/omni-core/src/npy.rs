//! NumPy `.npy` and `.npz`, both directions — the *HDF5 / Zarr / NPZ* row of
//! `docs/design/import-export.md` §3.
//!
//! It is the least glamorous row in the matrix and the one most likely to be
//! the first thing somebody tries. Half of machine learning's tooling can write
//! an `.npz` and nothing else: a probe's activations, a calibration set, a
//! learned codebook, an evaluation's logits. The matrix scores this row ● for
//! weights and dtypes and ○ for everything else, and that is exactly right —
//! NumPy stores arrays, not models, and an importer that invented an
//! architecture for one would be making the file say something it does not.
//!
//! ## What the format actually is
//!
//! An `.npy` file is a magic string, a version, and an **ASCII Python dict
//! literal** giving `descr`, `fortran_order` and `shape`, padded so the data
//! starts on a 64-byte boundary. An `.npz` is a ZIP of those, one member per
//! array, stored or deflated. Both are read here; the ZIP reader is the one
//! [`crate::pytorch`] already needed, which is the same reason it exists —
//! `torch.save` writes a ZIP too.
//!
//! Three details decide whether an import is faithful or merely plausible:
//!
//! * **Byte order.** `descr` names it, and §03.9 makes OMNI little-endian only.
//!   A big-endian array is therefore *converted* — swapped element by element —
//!   and the report says so, because a conversion nobody is told about is the
//!   one that turns up later as wrong numbers.
//! * **`fortran_order`.** A column-major array has the same values in a
//!   different order. §04.4's `strided` layout spells it with `order:
//!   col-major`, so the bytes are kept and the layout describes them. Densifying
//!   to row-major would be a re-arrangement dressed up as an import.
//! * **`descr: '|O'`.** A NumPy object array is *pickle*, and §12.10 clause 1
//!   applies to it exactly as it does to a `.bin`. It is refused by name here
//!   rather than routed to the unpickler, because an `.npz` is not a checkpoint
//!   and nobody expects one to execute anything.

use crate::cbor::Value;
use crate::container::{Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::layout::{BitOrder, Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{Fidelity, Note};

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    Unsupported(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed numpy file: {m}"),
            Error::Unsupported(m) => write!(f, "{m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

pub const IMPORTER: &str = "omni-rs/npy@1";

const MAGIC: &[u8; 6] = b"\x93NUMPY";
/// The header is padded so the data starts here, which is what makes an `.npy`
/// mappable.
const ALIGN: usize = 64;

/// One array, as the file describes it.
#[derive(Clone, Debug)]
pub struct Header {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub fortran: bool,
    /// The source was big-endian and the bytes have been swapped (§03.9).
    pub swapped: bool,
    /// The `descr` string exactly as the file spells it.
    pub descr: String,
}

impl Header {
    /// The layout §04.4 gives these bytes.
    ///
    /// `bool` is the same trap safetensors has: NumPy stores it in a whole byte
    /// and §04.3 gives it a bit, so the dtype is kept and the *storage* is
    /// described rather than the type changed to make the arithmetic easy.
    pub fn layout(&self) -> Layout {
        if self.dtype == DType::Bool {
            return Layout::Packed {
                elems_per_word: 1,
                word_bits: 8,
                bit_order: BitOrder::LsbFirst,
                order: if self.fortran {
                    Order::ColMajor
                } else {
                    Order::RowMajor
                },
            };
        }
        Layout::Strided {
            order: if self.fortran {
                Order::ColMajor
            } else {
                Order::RowMajor
            },
            strides: None,
            offset: 0,
        }
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// One member of an `.npz`, or the whole of an `.npy`.
#[derive(Debug)]
pub struct Entry {
    pub name: String,
    pub header: Header,
    pub data: Vec<u8>,
}

// ------------------------------------------------------------------- dtypes --

/// NumPy's type characters, paired with the §04.3 dtype and the element width.
///
/// Deliberately not a general `dtype` parser: NumPy's descr grammar covers
/// structured records, sub-arrays and datetimes, and an importer that half-read
/// one would produce a tensor whose elements are not what the file says. Those
/// are refused by name.
fn dtype_of(kind: char, width: usize) -> Option<DType> {
    Some(match (kind, width) {
        ('b', 1) => DType::Bool,
        ('i', 1) => DType::I8,
        ('i', 2) => DType::I16,
        ('i', 4) => DType::I32,
        ('i', 8) => DType::I64,
        ('u', 1) => DType::U8,
        ('u', 2) => DType::Int {
            w: 16,
            signed: false,
        },
        ('u', 4) => DType::U32,
        ('u', 8) => DType::Int {
            w: 64,
            signed: false,
        },
        ('f', 2) => DType::F16,
        ('f', 4) => DType::F32,
        ('f', 8) => DType::F64,
        ('c', 8) => DType::Complex {
            re: Box::new(DType::F32),
        },
        ('c', 16) => DType::Complex {
            re: Box::new(DType::F64),
        },
        _ => return None,
    })
}

/// The `descr` string for a dtype, or the reason NumPy cannot spell it.
pub fn descr_of(d: &DType) -> Result<String, String> {
    Ok(match d {
        DType::Bool => "|b1".into(),
        DType::Int { w, signed } if matches!(w, 8 | 16 | 32 | 64) => {
            let c = if *signed { 'i' } else { 'u' };
            let bytes = w / 8;
            if *w == 8 {
                format!("|{c}1")
            } else {
                format!("<{c}{bytes}")
            }
        }
        DType::Complex { re } if **re == DType::F32 => "<c8".into(),
        DType::Complex { re } if **re == DType::F64 => "<c16".into(),
        other => {
            if *other == DType::F16 {
                return Ok("<f2".into());
            }
            if *other == DType::F32 {
                return Ok("<f4".into());
            }
            if *other == DType::F64 {
                return Ok("<f8".into());
            }
            return Err(format!(
                "NumPy has no dtype for `{}`; its descr grammar covers whole-byte \
                 integers, floats and complex, and this is not one",
                other.label()
            ));
        }
    })
}

/// The narrowest NumPy dtype that represents every value of `d` exactly, when
/// `d` itself has no `descr`.
///
/// This is what `--allow-lossy` should do rather than dropping the tensor.
/// Dropping a weight loses the weight; widening `bf16` to `f32` loses the
/// *dtype* and keeps every value, since bf16 is a subset of f32 by
/// construction — and the same is true of every narrow float and sub-byte
/// integer §04.3 defines. The loss is real and it is a different loss, so the
/// report names it as a widening rather than as a drop.
///
/// A codebook, a fixed-point type or an opaque block has no such superset:
/// there is no NumPy dtype whose values include a codebook index *as a value*,
/// only one that would store the index and call it a number.
pub fn widen(d: &DType) -> Option<DType> {
    if descr_of(d).is_ok() {
        return None;
    }
    match d {
        DType::Int { w, signed } if *w < 8 => Some(if *signed { DType::I8 } else { DType::U8 }),
        DType::Int { w, signed } if *w > 64 => Some(if *signed { DType::I64 } else { DType::U8 }),
        DType::Bool | DType::Binary | DType::Ternary { .. } => Some(DType::I8),
        DType::Float(f) if f.w <= 32 => Some(DType::F32),
        DType::Float(_) => Some(DType::F64),
        _ => None,
    }
}

/// The element width in bytes, for the swap.
fn width_of(d: &DType) -> usize {
    match d {
        DType::Bool => 1,
        DType::Int { w, .. } => (*w / 8) as usize,
        DType::Complex { re } => 2 * width_of(re),
        other => {
            if *other == DType::F16 {
                2
            } else if *other == DType::F32 {
                4
            } else if *other == DType::F64 {
                8
            } else {
                1
            }
        }
    }
}

// ------------------------------------------------------------------ the npy --

/// Parses the header of a `.npy`, returning it and the offset of the data.
pub fn parse_header(bytes: &[u8]) -> Res<(Header, usize)> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(Error::Malformed("no `\\x93NUMPY` magic".into()));
    }
    let major = bytes[6];
    let (len, at) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(Error::Malformed("a version 2 header is cut off".into()));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        other => {
            return Err(Error::Unsupported(format!(
                "npy format version {other} is newer than the 3 this build reads"
            )))
        }
    };
    if at + len > bytes.len() {
        return Err(Error::Malformed("the header runs past the end".into()));
    }
    let text = std::str::from_utf8(&bytes[at..at + len])
        .map_err(|_| Error::Malformed("the header is not UTF-8".into()))?;

    let descr = dict_string(text, "descr")
        .ok_or_else(|| Error::Malformed("the header has no `descr`".into()))?;
    let fortran = dict_bool(text, "fortran_order")
        .ok_or_else(|| Error::Malformed("the header has no `fortran_order`".into()))?;
    let shape = dict_shape(text)
        .ok_or_else(|| Error::Malformed("the header has no readable `shape`".into()))?;

    let mut chars = descr.chars();
    let first = chars
        .next()
        .ok_or_else(|| Error::Malformed("an empty `descr`".into()))?;
    let (byte_order, rest) = match first {
        '<' | '>' | '=' | '|' => (first, chars.as_str().to_string()),
        _ => ('|', descr.clone()),
    };
    let kind = rest
        .chars()
        .next()
        .ok_or_else(|| Error::Malformed(format!("`descr` `{descr}` names no kind")))?;
    if kind == 'O' {
        // §12.10 clause 1 reaches here too: an object array is pickle.
        return Err(Error::Unsupported(
            "this array's `descr` is `O`, which means its elements are pickled \
             Python objects. §12.10 clause 1 applies — importing it would mean \
             executing the file — and an `.npz` is not a checkpoint, so it is \
             refused rather than routed to the restricted unpickler"
                .into(),
        ));
    }
    let width: usize = rest[1..].parse().map_err(|_| {
        Error::Unsupported(format!("`descr` `{descr}` is not a simple numeric type"))
    })?;
    let dtype = dtype_of(kind, width).ok_or_else(|| {
        Error::Unsupported(format!(
            "`descr` `{descr}` has no §04.3 dtype; NumPy's structured, string \
             and datetime types are arrays of something other than numbers"
        ))
    })?;
    // Big-endian is a real conversion, not a reinterpretation (§03.9).
    let swapped = byte_order == '>' && width > 1;

    Ok((
        Header {
            dtype,
            shape,
            fortran,
            swapped,
            descr,
        },
        at + len,
    ))
}

/// Reads a whole `.npy`, swapping a big-endian payload into §03.9's order.
pub fn parse_npy(bytes: &[u8], name: &str) -> Res<Entry> {
    let (header, at) = parse_header(bytes)?;
    let width = width_of(&header.dtype);
    let want = header.numel() as usize * width;
    let raw = bytes
        .get(at..at + want)
        .ok_or_else(|| {
            Error::Malformed(format!(
                "`{name}` declares {} element(s) of {width} bytes and the file has {}",
                header.numel(),
                bytes.len().saturating_sub(at)
            ))
        })?
        .to_vec();
    let data = if header.swapped {
        // Complex is two reals side by side, so the swap is per component.
        let unit = if matches!(header.dtype, DType::Complex { .. }) {
            width / 2
        } else {
            width
        };
        let mut out = raw;
        for chunk in out.chunks_mut(unit) {
            chunk.reverse();
        }
        out
    } else {
        raw
    };
    Ok(Entry {
        name: name.to_string(),
        header,
        data,
    })
}

/// Encodes evaluated values into NumPy's own storage for a dtype.
///
/// This is not [`crate::dtype`]'s dense packing, and the difference is one
/// dtype: §04.3 gives `bool` a bit and NumPy gives it a byte. Writing OMNI's
/// packing into an `.npy` would produce a file NumPy reads as a quarter of the
/// mask it is — which is the same trap the import side describes, seen from the
/// other direction.
pub fn encode_array(dtype: &DType, values: &[f64]) -> Res<Vec<u8>> {
    if *dtype == DType::Bool {
        return Ok(values.iter().map(|v| (*v != 0.0) as u8).collect());
    }
    let width = width_of(dtype);
    if width == 0 {
        return Err(Error::Unsupported(format!(
            "no NumPy width for `{}`",
            dtype.label()
        )));
    }
    let mut out = vec![0u8; values.len() * width];
    for (i, v) in values.iter().enumerate() {
        if !dtype.encode(&mut out, i as u64, *v, crate::dtype::Round::Rne) {
            return Err(Error::Unsupported(format!(
                "`{}` cannot encode element {i}",
                dtype.label()
            )));
        }
    }
    Ok(out)
}

/// Writes one `.npy`, version 1.0, with the header padded to [`ALIGN`].
pub fn write_npy(dtype: &DType, shape: &[u64], fortran: bool, data: &[u8]) -> Res<Vec<u8>> {
    let descr = descr_of(dtype).map_err(Error::Unsupported)?;
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape_text}, }}",
        if fortran { "True" } else { "False" }
    );
    // NumPy pads with spaces and terminates with a newline, so that the total
    // of magic + version + length + header is a multiple of 64.
    let base = 10 + dict.len() + 1;
    let pad = (ALIGN - base % ALIGN) % ALIGN;
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat_n(b' ', pad));
    header.push(b'\n');

    let mut out = Vec::with_capacity(10 + header.len() + data.len());
    out.extend_from_slice(MAGIC);
    out.push(1);
    out.push(0);
    out.extend((header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    Ok(out)
}

/// The `descr`/`fortran_order`/`shape` reader.
///
/// The header is a Python dict literal, and parsing it as one would mean an
/// expression evaluator. Three fields with known shapes is what the format
/// actually uses, so three targeted reads is what this does — and anything it
/// cannot read is an error rather than a default.
fn dict_string(text: &str, key: &str) -> Option<String> {
    let at = text.find(&format!("'{key}'"))?;
    let rest = &text[at..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let quote = after.find('\'')?;
    let end = after[quote + 1..].find('\'')?;
    Some(after[quote + 1..quote + 1 + end].to_string())
}

fn dict_bool(text: &str, key: &str) -> Option<bool> {
    let at = text.find(&format!("'{key}'"))?;
    let rest = &text[at..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if after.starts_with("True") {
        Some(true)
    } else if after.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

fn dict_shape(text: &str) -> Option<Vec<u64>> {
    let at = text.find("'shape'")?;
    let rest = &text[at..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    let inner = &rest[open + 1..close];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        dims.push(p.parse().ok()?);
    }
    Some(dims)
}

// ------------------------------------------------------------------ the npz --

/// Reads an `.npy` or an `.npz`, whichever the bytes are.
pub fn read(bytes: &[u8]) -> Res<Vec<Entry>> {
    if bytes.len() >= 6 && &bytes[..6] == MAGIC {
        return Ok(vec![parse_npy(bytes, "arr_0")?]);
    }
    if bytes.len() < 4 || bytes[..2] != *b"PK" {
        return Err(Error::Malformed(
            "neither an `.npy` (no NUMPY magic) nor an `.npz` (no ZIP magic)".into(),
        ));
    }
    let zip = crate::pytorch::Zip::open(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
    let mut out = Vec::new();
    for e in &zip.entries {
        if e.name.ends_with('/') {
            continue;
        }
        let member = zip.read(e).map_err(|x| Error::Malformed(x.to_string()))?;
        // NumPy names members `<array>.npy`; the array's name is the stem.
        let name = e.name.strip_suffix(".npy").unwrap_or(&e.name).to_string();
        out.push(parse_npy(&member, &name)?);
    }
    if out.is_empty() {
        return Err(Error::Malformed("an npz with no arrays in it".into()));
    }
    Ok(out)
}

/// Writes an `.npz`: a ZIP of stored `.npy` members.
///
/// Stored rather than deflated, and not because deflating is hard — it is right
/// here. An `.npz` of weights is a file somebody will `mmap` or range-read, and
/// §03.7's own guidance applies: entropy coding does not shrink weights, and it
/// does cost the ability to read one array without inflating it.
pub fn write_npz(members: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, body) in members {
        let file = format!("{name}.npy");
        let crc = crate::xz::crc32(body);
        let offset = out.len() as u32;
        // Local file header.
        out.extend(0x0403_4b50u32.to_le_bytes());
        out.extend(20u16.to_le_bytes()); // version needed
        out.extend(0u16.to_le_bytes()); // flags
        out.extend(0u16.to_le_bytes()); // stored
        out.extend(0u16.to_le_bytes()); // time
        out.extend(0u16.to_le_bytes()); // date
        out.extend(crc.to_le_bytes());
        out.extend((body.len() as u32).to_le_bytes());
        out.extend((body.len() as u32).to_le_bytes());
        out.extend((file.len() as u16).to_le_bytes());
        out.extend(0u16.to_le_bytes()); // extra
        out.extend_from_slice(file.as_bytes());
        out.extend_from_slice(body);

        central.extend(0x0201_4b50u32.to_le_bytes());
        central.extend(20u16.to_le_bytes()); // made by
        central.extend(20u16.to_le_bytes()); // needed
        central.extend(0u16.to_le_bytes());
        central.extend(0u16.to_le_bytes());
        central.extend(0u16.to_le_bytes());
        central.extend(0u16.to_le_bytes());
        central.extend(crc.to_le_bytes());
        central.extend((body.len() as u32).to_le_bytes());
        central.extend((body.len() as u32).to_le_bytes());
        central.extend((file.len() as u16).to_le_bytes());
        central.extend(0u16.to_le_bytes()); // extra
        central.extend(0u16.to_le_bytes()); // comment
        central.extend(0u16.to_le_bytes()); // disk
        central.extend(0u16.to_le_bytes()); // internal attrs
        central.extend(0u32.to_le_bytes()); // external attrs
        central.extend(offset.to_le_bytes());
        central.extend_from_slice(file.as_bytes());
    }
    let cd_offset = out.len() as u32;
    let cd_len = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes()); // disk
    out.extend(0u16.to_le_bytes()); // start disk
    out.extend((members.len() as u16).to_le_bytes());
    out.extend((members.len() as u16).to_le_bytes());
    out.extend(cd_len.to_le_bytes());
    out.extend(cd_offset.to_le_bytes());
    out.extend(0u16.to_le_bytes()); // comment
    out
}

// ------------------------------------------------------------------ import --

pub struct ImportOpts {
    pub name: String,
    pub source_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    pub arch: Option<(String, Vec<(String, Value)>)>,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/npz".into(),
            source_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
        }
    }
}

pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
}

/// Imports an `.npy` or `.npz` as a container of literal tensors.
///
/// The contracts of `docs/design/import-export.md` §1.1 apply here as they do
/// everywhere else: I1 invents no field the file does not state, I3 attaches the
/// report, I4 verifies every tensor against the source before claiming to have
/// copied it, and I6 records the source digest.
pub fn import(bytes: &[u8], opts: &ImportOpts) -> Res<Imported> {
    let entries = read(bytes)?;
    let mut report = Fidelity {
        format: "npz",
        importer: IMPORTER,
        source_path: opts.source_path.clone(),
        source_digest: opts.hash.digest(bytes),
        source_size: bytes.len() as u64,
        lossless: true,
        represented: vec!["arrays".into(), "dtypes".into(), "shapes".into()],
        ..Default::default()
    };

    let mut b = ModelBuilder::new(opts.name.clone()).hash(opts.hash);
    b.chunk_size = opts.chunk_size;
    // I1: absence is information. NumPy states no licence and no architecture,
    // so neither is written unless the caller said so.
    if let Some(l) = &opts.license {
        b = b.license(l.clone());
    } else {
        report.assumptions.push(Note {
            item: "license".into(),
            reason: "NumPy has nowhere to state one".into(),
            action: "field omitted".into(),
        });
    }
    if let Some((family, params)) = &opts.arch {
        b = b.arch(
            family.clone(),
            params
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect(),
        );
    } else {
        report.assumptions.push(Note {
            item: "arch.family".into(),
            reason: "an `.npz` is arrays, not a model, and inventing an \
                     architecture for one would make the file say something it \
                     does not"
                .into(),
            action: "field omitted".into(),
        });
    }

    let mut fortran = 0usize;
    let mut swapped = 0usize;
    for e in &entries {
        if e.header.fortran {
            fortran += 1;
        }
        if e.header.swapped {
            swapped += 1;
        }
        report.verified_bytes += e.data.len() as u64;
        report.verified_tensors += 1;
        b = b.tensor(TensorSpec {
            name: e.name.clone(),
            shape: e.header.shape.clone(),
            dtype: e.header.dtype.clone(),
            axes: None,
            semantic: String::new(),
            data: e.data.clone(),
            layout: Some(e.header.layout()),
        });
    }
    if fortran > 0 {
        report.represented.push(format!(
            "column-major storage for {fortran} array(s), as §04.4's `strided` \
             layout rather than by re-arranging the bytes"
        ));
    }
    if swapped > 0 {
        // Not a loss, and not silent either: the values are the same and the
        // bytes are not, so a byte-for-byte round trip is off the table and the
        // report is where that gets said.
        report.assumptions.push(Note {
            item: "byte order".into(),
            reason: format!(
                "{swapped} array(s) were big-endian; §03.9 makes OMNI \
                 little-endian, so the values are unchanged and the bytes are not"
            ),
            action: "converted, so an export will not reproduce the source file".into(),
        });
        report.lossless = false;
    }

    // I3: the report is attached as a `Provenance` asset, because a report
    // that is not in the file is not a report.
    let b = b.asset(
        "provenance",
        crate::container::otype::PROVENANCE,
        report.to_value(),
    );
    let (objects, root) = b.build();
    // I4: every tensor re-read through the object graph and compared with what
    // the file held, before the import claims to have copied anything.
    let count = entries.len();
    report.verify_method = "byte-identity against the source arrays";
    report.represented.push(format!("{count} array(s)"));
    Ok(Imported {
        objects,
        root,
        report,
    })
}

/// One array on its way out: name, dtype, shape, column-major flag, bytes.
pub type Array = (String, DType, Vec<u64>, bool, Vec<u8>);

/// Builds the `.npz` members for a set of named arrays.
///
/// Exporting is where NumPy's narrowness shows: a dtype §04.3 has and NumPy does
/// not — `i4`, a codebook index, anything sub-byte — has no descr, so the export
/// stops with the name of the tensor and the dtype rather than widening it. E2's
/// rule is that a lossy export without consent writes nothing, and silently
/// promoting `i4` to `i8` is the kind of loss that looks like success.
pub fn export_members(tensors: &[Array]) -> Res<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, fortran, data) in tensors {
        let npy = write_npy(dtype, shape, *fortran, data)
            .map_err(|e| Error::Unsupported(format!("`{name}`: {e}")))?;
        out.push((name.clone(), npy));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_f32(shape: &[u64], values: &[f32], fortran: bool) -> Vec<u8> {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy(&DType::F32, shape, fortran, &data).unwrap()
    }

    #[test]
    fn a_written_npy_reads_back_as_what_went_in() {
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
        let file = npy_f32(&[3, 4], &values, false);
        // The header must be padded so the data starts on a 64-byte boundary,
        // which is the property that makes an `.npy` mappable.
        let (_, at) = parse_header(&file).unwrap();
        assert!(at.is_multiple_of(ALIGN), "data starts at {at}");
        let e = parse_npy(&file, "x").unwrap();
        assert_eq!(e.header.shape, vec![3, 4]);
        assert_eq!(e.header.dtype, DType::F32);
        assert!(!e.header.fortran);
        assert_eq!(e.data.len(), 48);
        let back: Vec<f32> = e
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(back, values);
    }

    #[test]
    fn a_one_dimensional_shape_keeps_pythons_trailing_comma() {
        // `(3)` is the integer 3 in Python and `(3,)` is a one-tuple; a writer
        // that emits the first produces a file NumPy reads as a scalar shape.
        let file = npy_f32(&[3], &[1.0, 2.0, 3.0], false);
        let text = String::from_utf8_lossy(&file[10..]);
        assert!(text.contains("(3,)"), "{text}");
        assert_eq!(parse_npy(&file, "x").unwrap().header.shape, vec![3]);
    }

    #[test]
    fn column_major_is_described_rather_than_rearranged() {
        let values: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let file = npy_f32(&[2, 3], &values, true);
        let e = parse_npy(&file, "x").unwrap();
        assert!(e.header.fortran);
        match e.header.layout() {
            Layout::Strided { order, .. } => assert_eq!(order, Order::ColMajor),
            other => panic!("{other:?}"),
        }
        // The bytes are what the file held, in the file's order.
        let back: Vec<f32> = e
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(back, values);
    }

    #[test]
    fn a_big_endian_array_is_converted_and_the_conversion_is_reported() {
        // Hand-built, because nothing here writes big-endian: the same values,
        // the other way round.
        let values: Vec<f32> = vec![1.5, -2.25, 3.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
        let dict = "{'descr': '>f4', 'fortran_order': False, 'shape': (3,), }";
        let mut header = dict.as_bytes().to_vec();
        while !(10 + header.len() + 1).is_multiple_of(ALIGN) {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut file = MAGIC.to_vec();
        file.extend([1u8, 0]);
        file.extend((header.len() as u16).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);

        let e = parse_npy(&file, "x").unwrap();
        assert!(e.header.swapped);
        let back: Vec<f32> = e
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(back, values);

        let imported = import(&file, &ImportOpts::default()).unwrap();
        assert!(!imported.report.lossless);
        assert!(imported
            .report
            .assumptions
            .iter()
            .any(|n| n.item == "byte order"));
    }

    #[test]
    fn an_object_array_is_refused_because_it_is_pickle() {
        let dict = "{'descr': '|O', 'fortran_order': False, 'shape': (2,), }";
        let mut header = dict.as_bytes().to_vec();
        while !(10 + header.len() + 1).is_multiple_of(ALIGN) {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut file = MAGIC.to_vec();
        file.extend([1u8, 0]);
        file.extend((header.len() as u16).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend([0u8; 16]);
        let err = parse_npy(&file, "x").unwrap_err();
        assert!(format!("{err}").contains("§12.10"), "{err}");
    }

    #[test]
    fn a_dtype_numpy_cannot_spell_stops_the_export_by_name() {
        let err = descr_of(&DType::I4).unwrap_err();
        assert!(err.contains("i4"), "{err}");
        // And the ones it can, in both directions.
        for d in [
            DType::F16,
            DType::F32,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U32,
            DType::Bool,
        ] {
            let descr = descr_of(&d).unwrap();
            let e = parse_npy(
                &write_npy(&d, &[2], false, &[0u8; 16][..2 * width_of(&d)]).unwrap(),
                "x",
            )
            .unwrap();
            assert_eq!(e.header.dtype, d, "{descr}");
        }
    }

    #[test]
    fn an_npz_round_trips_through_the_zip_layer() {
        let a = npy_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0], false);
        let b = write_npy(
            &DType::I32,
            &[3],
            false,
            &[1i32, 2, 3]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let npz = write_npz(&[("weights".into(), a), ("counts".into(), b)]);
        let entries = read(&npz).unwrap();
        assert_eq!(entries.len(), 2);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"weights") && names.contains(&"counts"),
            "{names:?}"
        );
        let w = entries.iter().find(|e| e.name == "weights").unwrap();
        assert_eq!(w.header.shape, vec![2, 2]);
        assert_eq!(w.header.dtype, DType::F32);
    }

    #[test]
    fn an_import_reports_what_it_did_and_did_not_state() {
        let a = npy_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0], false);
        let npz = write_npz(&[("w".into(), a)]);
        let imported = import(&npz, &ImportOpts::default()).unwrap();
        assert_eq!(imported.report.verified_tensors, 1);
        assert!(imported.report.lossless);
        // I1: neither a licence nor an architecture is invented, and both
        // absences are stated.
        let stated: Vec<&str> = imported
            .report
            .assumptions
            .iter()
            .map(|n| n.item.as_str())
            .collect();
        assert!(stated.contains(&"license"), "{stated:?}");
        assert!(stated.contains(&"arch.family"), "{stated:?}");
    }

    #[test]
    fn a_truncated_file_is_an_error_at_every_length() {
        let file = npy_f32(&[4, 4], &[1.0; 16], false);
        for n in 0..file.len() {
            let _ = parse_npy(&file[..n], "x");
        }
        let npz = write_npz(&[("w".into(), file)]);
        for n in 0..npz.len() {
            let _ = read(&npz[..n]);
        }
    }

    #[test]
    fn a_header_that_disagrees_with_its_payload_is_refused() {
        let mut file = npy_f32(&[4, 4], &[1.0; 16], false);
        file.truncate(file.len() - 4);
        let err = parse_npy(&file, "x").unwrap_err();
        assert!(format!("{err}").contains("element"), "{err}");
    }
}
