//! safetensors: import and export (`docs/design/import-export.md`).
//!
//! safetensors is the format OMNI has the least excuse not to absorb. It is a
//! `u64` header length, a JSON header of names to `{dtype, shape,
//! data_offsets}`, and a buffer — no graph, no semantics, no metadata beyond a
//! flat string map. The capability matrix marks its round-trip **lossless**, and
//! this module is what has to earn that word.
//!
//! ## The importer contract, concretely
//!
//! The rules in §1.1 of the design document are not decoration; each one changes
//! the code:
//!
//! * **I1, never fabricate.** safetensors states no license, no parameter count,
//!   no architecture and no context length. None of those fields is written.
//!   [`ImportOpts`] lets a *caller* supply them, because a caller may know
//!   something the file does not say — but nothing is inferred from the tensor
//!   names, however suggestive `model.layers.0.attn.q_proj.weight` looks.
//! * **I2, preserve the unrepresentable.** `__metadata__` is a flat
//!   string→string map with no schema. The keys OMNI models are mapped; every
//!   other key is kept verbatim in a `Foreign` object, with its source path and
//!   digest, so a later importer or a human can recover it.
//! * **I3, report fidelity.** Every import produces a [`Fidelity`] report, and
//!   it is attached to the container as a `Provenance` object rather than printed
//!   and forgotten.
//! * **I4, verify what you claim.** The import re-reads every tensor it wrote,
//!   through the object graph, and compares it byte for byte with the source
//!   slice. The count of what was checked is in the report; a mismatch is a
//!   failed import, not a warning.
//! * **I6, record the source digest**, so "which file did this come from?" has
//!   an answer.
//!
//! ## The exporter contract
//!
//! [`plan`] computes what would be lost *without writing bytes* (E1), so a tool
//! can refuse or ask first; [`export`] takes a plan and refuses to run when the
//! plan reports loss unless the caller has said `allow_lossy` (E2). What is lost
//! is everything safetensors has no room for — the graph, the tokenizer, the
//! chat template, adapters, training state, signatures, provenance, quantization
//! structure, and any dtype the format does not name — and each is named
//! individually rather than summarised as "some metadata".
//!
//! E4's round-trip identity holds where the matrix says it does: for a model
//! whose tensors are dense values in a dtype safetensors names, export then
//! import reproduces every tensor's bytes exactly, and therefore its digest.

use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::{DType, Round};
use crate::expr::{concrete, Ctx, Expr};
use crate::json;
use crate::layout::Layout;
use crate::model::{ModelBuilder, TensorSpec};
use crate::tensor::{TensorDesc, TensorTable};

/// The importer's own identity, recorded in every report.
pub const IMPORTER: &str = "omni-import-safetensors";
/// The largest header this will parse. safetensors headers are JSON and a large
/// one is a few megabytes; 64 MiB is far past any real file and short of a
/// declared length being used to allocate the machine's memory (§12.4).
pub const MAX_HEADER: u64 = 64 << 20;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    /// The file is well-formed but says something this build cannot represent.
    Unsupported(String),
    /// An export would lose something and consent was not given (E2).
    Lossy(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed safetensors: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Lossy(m) => write!(f, "{m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// --------------------------------------------------------------------- dtypes --

/// The safetensors dtype names, paired with the OMNI dtype each denotes.
///
/// This is the whole of the format's type system, and every entry maps onto a
/// dtype §04.3 already has — which is why the matrix can say `●` for dtypes. The
/// table is the single place the correspondence is written, so the importer and
/// the exporter cannot disagree about it.
pub const DTYPES: &[(&str, &str)] = &[
    ("BOOL", "bool"),
    ("U8", "u8"),
    ("I8", "i8"),
    ("F8_E5M2", "f8e5m2"),
    ("F8_E4M3", "f8e4m3"),
    ("I16", "i16"),
    ("U16", "u16"),
    ("F16", "f16"),
    ("BF16", "bf16"),
    ("I32", "i32"),
    ("U32", "u32"),
    ("F32", "f32"),
    ("F64", "f64"),
    ("I64", "i64"),
    ("U64", "u64"),
];

/// The OMNI dtype a safetensors name denotes.
pub fn dtype_of(name: &str) -> Option<DType> {
    let alias = DTYPES.iter().find(|(st, _)| *st == name)?.1;
    DType::from_alias(alias)
}

/// How safetensors arranges a tensor of this dtype in bytes.
///
/// Row-major and densely packed for everything except `BOOL`. §04.3 gives `bool`
/// one *bit* per element; safetensors gives it a whole byte. That is a layout
/// difference, not a type difference, and §04.4's `packed` layout says it
/// exactly: one element per 8-bit word. The alternative — importing a mask as
/// `u8` — would change the tensor's type to avoid describing its storage, and a
/// reader asking "is this a boolean mask?" would get the wrong answer.
pub fn layout_of(d: &DType) -> Layout {
    if d == &DType::Bool {
        return Layout::Packed {
            elems_per_word: 1,
            word_bits: 8,
            bit_order: crate::layout::BitOrder::LsbFirst,
            order: crate::layout::Order::RowMajor,
        };
    }
    Layout::row_major()
}

/// Bytes a safetensors file uses for `n` elements of `d`: the format's own
/// `itemsize × n`, which for `BOOL` is not the dtype's packed size.
pub fn stored_bytes(d: &DType, shape: &[u64]) -> u64 {
    let n: u64 = shape.iter().product();
    layout_of(d)
        .stored_bytes(shape, d)
        .unwrap_or_else(|| d.packed_bytes(n))
}

/// The safetensors name for an OMNI dtype, if the format has one.
///
/// Compared structurally rather than by alias, so `f32` written the long way in
/// a descriptor still exports as `F32`.
pub fn name_of(d: &DType) -> Option<&'static str> {
    DTYPES
        .iter()
        .find(|(_, alias)| DType::from_alias(alias).as_ref() == Some(d))
        .map(|(st, _)| *st)
}

// --------------------------------------------------------------------- reading --

/// One tensor in a safetensors file.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    /// The name as the file spells it, kept so an export can spell it back.
    pub st_dtype: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Byte range within the data buffer, i.e. relative to the end of the header.
    pub begin: u64,
    pub end: u64,
}

impl Entry {
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn len(&self) -> u64 {
        self.end - self.begin
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.begin
    }
}

/// A parsed safetensors file: the header, checked, and a borrowed data buffer.
pub struct File<'a> {
    /// Tensors in the file's own offset order, which is the order they load in.
    pub entries: Vec<Entry>,
    /// `__metadata__`, a flat string→string map with no schema of its own.
    pub metadata: Vec<(String, String)>,
    data: &'a [u8],
    /// Where the data buffer starts in the file.
    pub data_offset: u64,
}

impl<'a> File<'a> {
    /// Parses and *validates*. The validation is the point: a header that
    /// disagrees with the buffer it describes is the one way this format can lie,
    /// and a reader that trusts the offsets reads someone else's tensor.
    pub fn parse(bytes: &'a [u8]) -> Res<File<'a>> {
        if bytes.len() < 8 {
            return Err(Error::Malformed(
                "shorter than the 8-byte header length".into(),
            ));
        }
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if n > MAX_HEADER {
            return Err(Error::Malformed(format!(
                "a declared header of {n} bytes exceeds the {MAX_HEADER}-byte bound"
            )));
        }
        let end = 8u64
            .checked_add(n)
            .filter(|e| *e <= bytes.len() as u64)
            .ok_or_else(|| {
                Error::Malformed(format!(
                    "the header claims {n} bytes, the file holds {}",
                    bytes.len().saturating_sub(8)
                ))
            })? as usize;
        let header = json::parse(&bytes[8..end]).map_err(|e| Error::Malformed(e.to_string()))?;
        let map = header
            .as_object()
            .ok_or_else(|| Error::Malformed("the header is not a JSON object".into()))?;
        let data = &bytes[end..];

        let mut metadata = Vec::new();
        let mut entries = Vec::new();
        for (name, v) in map {
            if name == "__metadata__" {
                let m = v
                    .as_object()
                    .ok_or_else(|| Error::Malformed("`__metadata__` is not an object".into()))?;
                for (k, mv) in m {
                    // The format says string→string. A nested value is not
                    // something to flatten silently.
                    let s = mv.as_str().ok_or_else(|| {
                        Error::Malformed(format!("`__metadata__.{k}` is not a string"))
                    })?;
                    metadata.push((k.clone(), s.to_string()));
                }
                continue;
            }
            let st_dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| Error::Malformed(format!("`{name}` has no `dtype`")))?
                .to_string();
            let dtype = dtype_of(&st_dtype).ok_or_else(|| {
                Error::Unsupported(format!(
                    "`{name}` has dtype `{st_dtype}`, which is not one of safetensors' \
                     {} types",
                    DTYPES.len()
                ))
            })?;
            let shape: Vec<u64> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| Error::Malformed(format!("`{name}` has no `shape`")))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .ok_or_else(|| Error::Malformed(format!("`{name}` has a non-integer dim")))
                })
                .collect::<Res<Vec<u64>>>()?;
            let offs = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .filter(|o| o.len() == 2)
                .ok_or_else(|| {
                    Error::Malformed(format!("`{name}` has no two-element `data_offsets`"))
                })?;
            let (begin, end2) = (
                offs[0].as_u64().ok_or_else(|| {
                    Error::Malformed(format!("`{name}` has a non-integer offset"))
                })?,
                offs[1].as_u64().ok_or_else(|| {
                    Error::Malformed(format!("`{name}` has a non-integer offset"))
                })?,
            );
            if begin > end2 || end2 > data.len() as u64 {
                return Err(Error::Malformed(format!(
                    "`{name}` claims bytes {begin}..{end2} of a {}-byte buffer",
                    data.len()
                )));
            }
            let e = Entry {
                name: name.clone(),
                st_dtype,
                dtype,
                shape,
                begin,
                end: end2,
            };
            // The declared extent has to be exactly the tensor's size. Anything
            // else means the header and the buffer describe different things.
            let want = stored_bytes(&e.dtype, &e.shape);
            if e.len() != want {
                return Err(Error::Malformed(format!(
                    "`{}` is {} elements of {} = {want} bytes, but claims {}",
                    e.name,
                    e.numel(),
                    e.st_dtype,
                    e.len()
                )));
            }
            entries.push(e);
        }

        // Tensors are contiguous from the start of the buffer, in offset order.
        // safetensors requires it, and requiring it here means an overlap — two
        // names for bytes only one of them owns — cannot be imported as two
        // independent tensors.
        entries.sort_by_key(|e| (e.begin, e.end));
        let mut at = 0u64;
        for e in &entries {
            if e.begin != at {
                return Err(Error::Malformed(format!(
                    "`{}` starts at {} with {at} expected: the buffer is not covered \
                     contiguously",
                    e.name, e.begin
                )));
            }
            at = e.end;
        }
        if at != data.len() as u64 {
            return Err(Error::Malformed(format!(
                "{} bytes of the buffer are described, {} are present",
                at,
                data.len()
            )));
        }

        Ok(File {
            entries,
            metadata,
            data,
            data_offset: end as u64,
        })
    }

    /// The bytes of one tensor. In range by construction: [`File::parse`] checked
    /// every extent against the buffer before this could be called.
    pub fn tensor(&self, e: &Entry) -> &'a [u8] {
        &self.data[e.begin as usize..e.end as usize]
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }
}

// ---------------------------------------------------------------- the report --

/// One thing an import could not represent, or chose not to invent.
#[derive(Clone, Debug)]
pub struct Note {
    pub item: String,
    pub reason: String,
    pub action: String,
}

/// The fidelity report of §2: what was represented, what was not, and what the
/// importer declined to guess.
#[derive(Clone, Debug, Default)]
pub struct Fidelity {
    pub source_path: String,
    pub source_digest: Digest,
    pub source_size: u64,
    pub lossless: bool,
    pub represented: Vec<String>,
    pub unrepresented: Vec<Note>,
    pub assumptions: Vec<Note>,
    /// Tensors re-read through the object graph and compared with the source.
    pub verified_tensors: usize,
    pub verified_bytes: u64,
    pub warnings: Vec<String>,
}

impl Fidelity {
    /// The `omni.prov/import` object of §2, ready to be attached to the
    /// container. A report that is not in the file is not a report.
    pub fn to_value(&self) -> crate::cbor::Value {
        use crate::cbor::Value as C;
        let notes = |v: &[Note]| {
            C::Array(
                v.iter()
                    .map(|n| {
                        C::map(vec![
                            ("item", C::text(n.item.clone())),
                            ("reason", C::text(n.reason.clone())),
                            ("action", C::text(n.action.clone())),
                        ])
                    })
                    .collect(),
            )
        };
        C::map(vec![
            ("t", C::text("omni.prov/import")),
            ("v", C::U(1)),
            (
                "source",
                C::map(vec![
                    ("format", C::text("safetensors")),
                    ("path", C::text(self.source_path.clone())),
                    ("digest", C::Bytes(self.source_digest.to_vec())),
                    ("size", C::U(self.source_size)),
                ]),
            ),
            (
                "importer",
                C::map(vec![
                    ("name", C::text(IMPORTER)),
                    ("version", C::text(env!("CARGO_PKG_VERSION"))),
                ]),
            ),
            ("lossless", C::Bool(self.lossless)),
            (
                "represented",
                C::Array(
                    self.represented
                        .iter()
                        .map(|s| C::text(s.clone()))
                        .collect(),
                ),
            ),
            ("unrepresented", notes(&self.unrepresented)),
            ("assumptions", notes(&self.assumptions)),
            (
                "verification",
                C::map(vec![
                    // Not "sample-dequant": nothing here is a block format, so
                    // every tensor is compared in full rather than sampled.
                    ("method", C::text("byte-identity")),
                    ("tensors_checked", C::U(self.verified_tensors as u64)),
                    ("bytes_checked", C::U(self.verified_bytes)),
                    ("bit_exact", C::Bool(true)),
                ]),
            ),
            (
                "warnings",
                C::Array(self.warnings.iter().map(|s| C::text(s.clone())).collect()),
            ),
        ])
    }
}

// --------------------------------------------------------------------- import --

/// What a caller may add that the file does not say. Every field is optional and
/// every one is absent unless given: I1 forbids inventing them, and it does not
/// become acceptable because a default would look tidier.
#[derive(Clone, Debug)]
pub struct ImportOpts {
    /// The model's name. safetensors has no name field, so this is the caller's
    /// to supply; the file's path is recorded regardless.
    pub name: String,
    pub source_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    pub arch: Option<(String, Vec<(String, crate::cbor::Value)>)>,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/safetensors".into(),
            source_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
        }
    }
}

/// The result of an import: an object graph, its root, and the report.
pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
}

/// `__metadata__` keys this importer models, mapped onto OMNI metadata. Anything
/// else goes to a `Foreign` object rather than being dropped or guessed at.
///
/// The list is short on purpose. `format` is written by every safetensors writer
/// and says `pt`; it describes the *producer*, which OMNI records as provenance
/// rather than as a property of the model.
fn modelled_metadata_key(k: &str) -> bool {
    matches!(k, "format")
}

/// The `Foreign` object that preserves the metadata keys this build does not
/// model (I2), with the source path, offset and digest that make it recoverable.
fn foreign_object(unmodelled: &[(String, String)], path: &str, source: &Digest) -> Object {
    use crate::cbor::Value as C;
    let doc = json::Value::Object(
        unmodelled
            .iter()
            .map(|(k, v)| (k.clone(), json::string(v.clone())))
            .collect(),
    );
    Object::structure(
        otype::FOREIGN,
        &C::map(vec![
            ("t", C::text("omni.core/foreign")),
            ("v", C::U(1)),
            ("format", C::text("safetensors")),
            ("item", C::text("__metadata__ (keys with no OMNI schema)")),
            (
                "source",
                C::map(vec![
                    ("path", C::text(path.to_string())),
                    // The header starts after the 8-byte length, and that is
                    // where these keys were.
                    ("offset", C::U(8)),
                    ("digest", C::Bytes(source.to_vec())),
                ]),
            ),
            ("media_type", C::text("application/json")),
            ("bytes", C::Bytes(doc.encode().into_bytes())),
        ]),
    )
}

/// Assembles the object graph for one import: tensors, whatever the caller
/// stated, the fidelity report, and the preserved metadata.
///
/// It is a function rather than inline code because it runs twice. The report
/// records how many tensors were verified, and verification needs a graph to
/// verify *against* — so the graph is built once to check, then rebuilt with the
/// finished report. Building it two different ways would be how the container's
/// report and the caller's report come to disagree.
fn assemble(
    f: &File<'_>,
    opts: &ImportOpts,
    unmodelled: &[(String, String)],
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
    // The tensors, in the file's own order, which is also its load order.
    for e in &f.entries {
        b = b.tensor(TensorSpec {
            name: e.name.clone(),
            shape: e.shape.clone(),
            dtype: e.dtype.clone(),
            axes: None,
            // safetensors says nothing about what a tensor *is*. Calling
            // everything a weight would be a guess; §04.2's `semantic` is left
            // for a caller who knows.
            semantic: "",
            data: f.tensor(e).to_vec(),
            layout: Some(layout_of(&e.dtype)),
        });
    }
    // I3: the report goes in the container, not just in the return value.
    b = b.asset("provenance", otype::PROVENANCE, report.to_value());

    let foreign = (!unmodelled.is_empty())
        .then(|| foreign_object(unmodelled, &opts.source_path, &report.source_digest));
    if let Some(obj) = &foreign {
        let d = obj.digest(opts.hash);
        b = b.manifest_key(
            "foreign",
            crate::cbor::Value::Array(vec![crate::cbor::Value::Array(vec![
                crate::cbor::Value::U(otype::FOREIGN as u64),
                crate::cbor::Value::Bytes(d.to_vec()),
            ])]),
        );
    }
    let (mut objects, root) = b.build();
    objects.extend(foreign);
    (objects, root)
}

/// Imports a safetensors file into an OMNI object graph.
pub fn import(bytes: &[u8], opts: &ImportOpts) -> Res<Imported> {
    let f = File::parse(bytes)?;
    let mut report = Fidelity {
        source_path: opts.source_path.clone(),
        source_digest: opts.hash.digest(bytes),
        source_size: bytes.len() as u64,
        lossless: true,
        represented: vec!["tensors".into(), "dtypes".into(), "shapes".into()],
        ..Default::default()
    };

    // I1: only what the caller stated, recorded as *stated by the caller* rather
    // than as something read from the file. Absence is information (§1.1).
    report.assumptions.push(Note {
        item: "license".into(),
        reason: "safetensors declares none".into(),
        action: match &opts.license {
            Some(spdx) => format!("supplied by the caller as `{spdx}`"),
            None => "field omitted".into(),
        },
    });
    report.assumptions.push(Note {
        item: "arch.family".into(),
        reason: "safetensors has no architecture field; the tensor names are a \
                 convention, not a declaration"
            .into(),
        action: match &opts.arch {
            Some((family, _)) => format!("supplied by the caller as `{family}`"),
            None => "field omitted".into(),
        },
    });
    report.assumptions.push(Note {
        item: "params_total".into(),
        reason: "computable from the shapes, but the file does not declare it".into(),
        action: "recomputed from the imported tensors".into(),
    });

    // I2: the metadata keys this build does not model, kept verbatim.
    let unmodelled: Vec<(String, String)> = f
        .metadata
        .iter()
        .filter(|(k, _)| !modelled_metadata_key(k))
        .cloned()
        .collect();
    if !f.metadata.is_empty() {
        report.represented.push("__metadata__".into());
    }
    for (k, _) in &unmodelled {
        report.unrepresented.push(Note {
            item: format!("__metadata__.{k}"),
            reason: "free-form key with no OMNI schema".into(),
            action: "preserved verbatim in a Foreign object".into(),
        });
    }

    // I4: verify what is claimed. Every tensor is read back out of a graph built
    // from this file and compared with the source bytes. This is the only thing
    // that makes "lossless" a finding rather than an intention.
    let (probe, probe_root) = assemble(&f, opts, &unmodelled, &report);
    let store = store_of(&probe, opts.hash);
    let ctx = Ctx::new(&store);
    let table = table_of(&probe, &probe_root, opts.hash)?;
    for e in &f.entries {
        let r = table
            .tensors
            .get(&e.name)
            .ok_or_else(|| Error::Core(format!("`{}` did not reach the table", e.name)))?;
        let desc = TensorDesc::from_value(
            &crate::cbor::decode(
                &crate::store::Store::resolve(&store, &r.1)
                    .map_err(|err| Error::Core(err.to_string()))?
                    .ok_or_else(|| Error::Core("a descriptor went missing".into()))?,
            )
            .map_err(|err| Error::Core(err.to_string()))?,
        )
        .map_err(|err| Error::Core(err.to_string()))?;
        let got = materialize(&desc, &ctx)?;
        if got != f.tensor(e) {
            return Err(Error::Core(format!(
                "I4: `{}` did not survive import byte for byte ({} bytes in, {} out)",
                e.name,
                e.len(),
                got.len()
            )));
        }
        report.verified_tensors += 1;
        report.verified_bytes += e.len();
    }

    // Rebuild with the verification counts in the report, so the object in the
    // container and the report the caller is handed say the same thing.
    let (objects, root) = assemble(&f, opts, &unmodelled, &report);
    Ok(Imported {
        objects,
        root,
        report,
    })
}

/// A store over a freshly built object list, for reading the graph back.
fn store_of(objects: &[Object], hash: HashAlgo) -> crate::store::MemoryStore {
    let mut store = crate::store::MemoryStore::new(hash);
    for o in objects {
        // The stored form is irrelevant here: nothing built in this module is
        // compressed, and the digest is over the logical bytes regardless.
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
    let decode = |d: &Digest| -> Res<crate::cbor::Value> {
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

/// The dense, row-major bytes of a tensor, in its declared dtype.
///
/// A plain literal is copied: its chunks already hold exactly these bytes, and
/// routing them through `f64` would be both slower and a claim about element
/// semantics that `opaque` types do not support. Anything else is evaluated and
/// re-encoded, which is what makes exporting a quantized tensor possible at all.
fn materialize(desc: &TensorDesc, ctx: &Ctx<'_>) -> Res<Vec<u8>> {
    let numel: u64 = concrete(&desc.shape)
        .ok_or_else(|| {
            Error::Unsupported(
                "a symbolic shape has to be bound before it can be written to a flat \
                 buffer (§04.7.3)"
                    .into(),
            )
        })?
        .iter()
        .product();
    if let Expr::Literal { chunks, .. } = &desc.value {
        // Row-major, and the byte-per-element packing an imported `bool` uses:
        // in both cases the stored bytes *are* the flat buffer safetensors wants.
        if desc.layout == Layout::row_major() || desc.layout == layout_of(&desc.dtype) {
            let bytes = ctx
                .chunk_bytes(chunks)
                .map_err(|e| Error::Core(e.to_string()))?;
            let want = desc
                .layout
                .stored_bytes(
                    &crate::expr::concrete(&desc.shape).unwrap_or_default(),
                    &desc.dtype,
                )
                .unwrap_or_else(|| desc.dtype.packed_bytes(numel)) as usize;
            if bytes.len() >= want {
                return Ok(bytes[..want].to_vec());
            }
        }
    }
    let t = desc
        .value
        .eval(ctx)
        .map_err(|e| Error::Core(e.to_string()))?;
    let mut out = vec![0u8; stored_bytes(&desc.dtype, &[numel]) as usize];
    for (i, x) in t.data.iter().enumerate() {
        if !write_element(&desc.dtype, &mut out, i as u64, *x) {
            return Err(Error::Unsupported(format!(
                "dtype `{}` has no element encoding, so it cannot be written to a flat \
                 buffer",
                desc.dtype
                    .alias()
                    .unwrap_or("(a type with no registered alias)")
            )));
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------- export --

/// Writes element `i` the way safetensors expects it.
///
/// `bool` needs its own line for the same reason it needs its own layout: the
/// dtype's own encoder packs eight per byte, and this format wants one.
fn write_element(d: &DType, out: &mut [u8], i: u64, x: f64) -> bool {
    if d == &DType::Bool {
        match out.get_mut(i as usize) {
            Some(b) => {
                *b = u8::from(x != 0.0);
                return true;
            }
            None => return false,
        }
    }
    d.encode(out, i, x, Round::Rne)
}

/// One thing an export would lose (§5.1).
#[derive(Clone, Debug)]
pub struct Loss {
    pub item: String,
    pub reason: String,
}

/// What an export would produce and what it would cost, computed without writing
/// anything (E1).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// `(name, safetensors dtype, shape, bytes)`, in the order they will be
    /// written.
    pub tensors: Vec<(String, String, Vec<u64>, u64)>,
    pub loss: Vec<Loss>,
    /// Total size of the file this plan would write.
    pub bytes: u64,
}

impl Plan {
    pub fn lossless(&self) -> bool {
        self.loss.is_empty()
    }

    /// The loss report of §5.1, as JSON, for writing beside the artifact (E3).
    pub fn loss_report(&self, source: &Digest, hash: HashAlgo) -> String {
        json::object(vec![
            ("target", json::string("safetensors")),
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
            ("lossless", json::Value::Bool(self.lossless())),
            ("tensors", json::Value::U(self.tensors.len() as u64)),
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

/// E1: what would be lost, without producing bytes.
///
/// `manifest` is the container's root object; it is read for the things
/// safetensors cannot carry, so that they are named individually instead of
/// summarised.
pub fn plan(
    ctx: &Ctx<'_>,
    table: &TensorTable,
    manifest: &crate::cbor::Value,
    descs: &dyn Fn(&Digest) -> Option<TensorDesc>,
) -> Res<Plan> {
    let mut p = Plan::default();

    // Order: the table's own load order first, then anything it did not list, so
    // an export is deterministic and matches the model's intent.
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
        let Some(desc) = descs(&r.1) else {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: "its descriptor is not present in this store".into(),
            });
            continue;
        };
        let Some(shape) = concrete(&desc.shape) else {
            p.loss.push(Loss {
                item: format!("tensor `{name}`"),
                reason: "a symbolic shape has no fixed extent to write".into(),
            });
            continue;
        };
        let numel: u64 = shape.iter().product();
        match name_of(&desc.dtype) {
            Some(st) => {
                let bytes = stored_bytes(&desc.dtype, &shape);
                at += bytes;
                p.tensors.push((name.clone(), st.to_string(), shape, bytes));
            }
            None => {
                // The dtype has no safetensors name. The values can still be
                // written — as f32 — but that is a change of type, so it is loss
                // and is reported as such rather than done quietly.
                let bytes = DType::F32.packed_bytes(numel);
                at += bytes;
                p.tensors
                    .push((name.clone(), "F32".to_string(), shape, bytes));
                p.loss.push(Loss {
                    item: format!("tensor `{name}` dtype"),
                    reason: format!(
                        "`{}` has no safetensors equivalent; it would be written as F32",
                        desc.dtype
                            .alias()
                            .unwrap_or("(a type with no registered alias)")
                    ),
                });
            }
        }
        // A tensor whose value is an expression is *evaluated* on export. That is
        // not loss — the values are the same — but the structure that produced
        // them is gone, and for a quantized tensor that structure is the reason
        // the model is small.
        if !matches!(desc.value, Expr::Literal { .. }) {
            p.loss.push(Loss {
                item: format!("tensor `{name}` structure"),
                reason: "its value is an expression; safetensors holds only materialized \
                         bytes, so the derivation is lost even though the values are not"
                    .into(),
            });
        }
    }

    // Everything safetensors has no room for, named one at a time.
    let assets = manifest.get("assets");
    for (slot, what) in [
        ("graph", "the execution graph (§07)"),
        ("tokenizer", "the tokenizer (§06.7)"),
        ("chat_template", "the chat template (§06.9)"),
        ("provenance", "provenance and the import history (§06.4)"),
    ] {
        if assets.and_then(|a| a.get(slot)).is_some() {
            p.loss.push(Loss {
                item: slot.to_string(),
                reason: format!("safetensors cannot carry {what}"),
            });
        }
    }
    if manifest.get("attestations").is_some() {
        p.loss.push(Loss {
            item: "attestations".into(),
            reason: "safetensors has no signature envelope (§12.5)".into(),
        });
    }
    if manifest.get("parents").is_some() {
        p.loss.push(Loss {
            item: "parents".into(),
            reason: "safetensors cannot express inheritance, so a delta model would be \
                     written as a full copy (§08.6)"
                .into(),
        });
    }
    // Metadata: safetensors' `__metadata__` is string→string, so a structured
    // model card does not fit. `meta` is a top-level manifest key rather than an
    // asset slot (§06.2), which is worth getting right — looking in the wrong
    // place would report every model as carrying no metadata.
    if manifest.get("meta").is_some() {
        p.loss.push(Loss {
            item: "metadata".into(),
            reason: "the model card is structured; `__metadata__` is a flat string map, \
                     so the shape of it does not survive"
                .into(),
        });
    }
    let _ = ctx;
    p.bytes = at;
    Ok(p)
}

/// Recovers the `__metadata__` keys a previous import preserved (I2).
///
/// This is what makes preservation worth doing. A `Foreign` object that nothing
/// can read again is a slightly more respectable form of dropping the data; an
/// export that restores those keys is why they were kept. Keys this build does
/// model are not here — they came from the file's own fields and the exporter
/// writes them itself.
pub fn preserved_metadata(
    manifest: &crate::cbor::Value,
    objects: &dyn Fn(&Digest) -> Option<crate::cbor::Value>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(list) = manifest.get("foreign").and_then(|f| f.as_array()) else {
        return out;
    };
    for r in list {
        let Ok(r) = crate::expr::parse_ref_value(r) else {
            continue;
        };
        let Some(v) = objects(&r.1) else { continue };
        // Only this format's own preserved metadata: a Foreign object from a
        // GGUF import describes GGUF keys, and writing those into a safetensors
        // header would be inventing a convention.
        if v.get("format").and_then(|f| f.as_str()) != Some("safetensors") {
            continue;
        }
        if v.get("media_type").and_then(|m| m.as_str()) != Some("application/json") {
            continue;
        }
        let Some(bytes) = v.get("bytes").and_then(|b| b.as_bytes()) else {
            continue;
        };
        let Ok(doc) = json::parse(bytes) else {
            continue;
        };
        let Some(map) = doc.as_object() else { continue };
        for (k, val) in map {
            if let Some(sv) = val.as_str() {
                out.push((k.clone(), sv.to_string()));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// E2/E3: writes the file the plan describes, refusing a lossy export without
/// consent.
pub fn export(
    ctx: &Ctx<'_>,
    table: &TensorTable,
    plan: &Plan,
    descs: &dyn Fn(&Digest) -> Option<TensorDesc>,
    extra_metadata: &[(String, String)],
    allow_lossy: bool,
) -> Res<Vec<u8>> {
    if !plan.lossless() && !allow_lossy {
        return Err(Error::Lossy(format!(
            "E2: this export would lose {} thing(s) and `--allow-lossy` was not given; \
             the first is: {} ({})",
            plan.loss.len(),
            plan.loss[0].item,
            plan.loss[0].reason
        )));
    }

    // Header first, with the offsets the plan already computed: the plan is what
    // was consented to, so the writer follows it rather than recomputing.
    let mut header = std::collections::BTreeMap::new();
    let mut at = 0u64;
    for (name, st, shape, bytes) in &plan.tensors {
        header.insert(
            name.clone(),
            json::object(vec![
                ("dtype", json::string(st.clone())),
                (
                    "shape",
                    json::Value::Array(shape.iter().map(|d| json::Value::U(*d)).collect()),
                ),
                (
                    "data_offsets",
                    json::Value::Array(vec![json::Value::U(at), json::Value::U(at + bytes)]),
                ),
            ]),
        );
        at += bytes;
    }
    let mut md: std::collections::BTreeMap<String, json::Value> = extra_metadata
        .iter()
        .map(|(k, v)| (k.clone(), json::string(v.clone())))
        .collect();
    // E3: the exported file points back at what produced it.
    md.insert("format".into(), json::string("pt"));
    md.insert(
        "omni.exporter".into(),
        json::string(format!(
            "omni-export-safetensors/{}",
            env!("CARGO_PKG_VERSION")
        )),
    );
    if !md.is_empty() {
        header.insert("__metadata__".into(), json::Value::Object(md));
    }
    let header_bytes = json::Value::Object(header).encode().into_bytes();

    let mut out = Vec::with_capacity(8 + header_bytes.len() + plan.bytes as usize);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);

    for (name, st, _, bytes) in &plan.tensors {
        let r = table.tensors.get(name).ok_or_else(|| {
            Error::Core(format!("`{name}` left the table between plan and write"))
        })?;
        let desc = descs(&r.1)
            .ok_or_else(|| Error::Core(format!("`{name}`'s descriptor is not present")))?;
        let mut data = if name_of(&desc.dtype).map(|s| s == st.as_str()) == Some(true) {
            materialize(&desc, ctx)?
        } else {
            // The lossy path the plan warned about: values in F32.
            let t = desc
                .value
                .eval(ctx)
                .map_err(|e| Error::Core(e.to_string()))?;
            let mut buf = vec![0u8; DType::F32.packed_bytes(t.data.len() as u64) as usize];
            for (i, x) in t.data.iter().enumerate() {
                write_element(&DType::F32, &mut buf, i as u64, *x);
            }
            buf
        };
        if data.len() as u64 != *bytes {
            return Err(Error::Core(format!(
                "`{name}` produced {} bytes where the plan said {bytes}",
                data.len()
            )));
        }
        out.append(&mut data);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A safetensors file built by hand, so the tests are against the format
    /// rather than against this module's own writer.
    fn handmade() -> Vec<u8> {
        let mut data = Vec::new();
        // `a`: f32[2,3], 24 bytes.
        for i in 0..6u32 {
            data.extend_from_slice(&(i as f32 * 0.5).to_le_bytes());
        }
        // `b`: i64[2], 16 bytes.
        for i in [-7i64, 9] {
            data.extend_from_slice(&i.to_le_bytes());
        }
        // `c`: bf16[4], 8 bytes.
        for h in [0x3f80u16, 0xbf80, 0x4000, 0x0000] {
            data.extend_from_slice(&h.to_le_bytes());
        }
        let header = json::object(vec![
            (
                "a",
                json::object(vec![
                    ("dtype", json::string("F32")),
                    (
                        "shape",
                        json::Value::Array(vec![json::Value::U(2), json::Value::U(3)]),
                    ),
                    (
                        "data_offsets",
                        json::Value::Array(vec![json::Value::U(0), json::Value::U(24)]),
                    ),
                ]),
            ),
            (
                "b",
                json::object(vec![
                    ("dtype", json::string("I64")),
                    ("shape", json::Value::Array(vec![json::Value::U(2)])),
                    (
                        "data_offsets",
                        json::Value::Array(vec![json::Value::U(24), json::Value::U(40)]),
                    ),
                ]),
            ),
            (
                "c",
                json::object(vec![
                    ("dtype", json::string("BF16")),
                    ("shape", json::Value::Array(vec![json::Value::U(4)])),
                    (
                        "data_offsets",
                        json::Value::Array(vec![json::Value::U(40), json::Value::U(48)]),
                    ),
                ]),
            ),
            (
                "__metadata__",
                json::object(vec![
                    ("format", json::string("pt")),
                    ("hand.written", json::string("yes")),
                ]),
            ),
        ])
        .encode()
        .into_bytes();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn every_safetensors_dtype_maps_onto_one_omni_dtype() {
        for (st, alias) in DTYPES {
            let d = dtype_of(st).unwrap_or_else(|| panic!("{st} has no dtype"));
            assert_eq!(
                d,
                DType::from_alias(alias).unwrap(),
                "{st} should be {alias}"
            );
            // And the mapping is a bijection, which is what lets an export spell
            // the name back.
            assert_eq!(name_of(&d), Some(*st), "{st} does not round-trip");
        }
        assert!(dtype_of("F8_E8M0").is_none());
        assert!(dtype_of("f32").is_none(), "the names are upper case");
        // A dtype OMNI has and safetensors does not: no name, no pretending.
        assert!(name_of(&DType::I4).is_none());
        assert!(name_of(&DType::F4E2M1).is_none());
    }

    #[test]
    fn a_handmade_file_parses_with_its_offsets_checked() {
        let bytes = handmade();
        let f = File::parse(&bytes).unwrap();
        assert_eq!(f.entries.len(), 3);
        assert_eq!(f.entries[0].name, "a");
        assert_eq!(f.entries[0].shape, vec![2, 3]);
        assert_eq!(f.entries[0].dtype, DType::F32);
        assert_eq!(f.tensor(&f.entries[0]).len(), 24);
        assert_eq!(f.get("c").unwrap().dtype, DType::BF16);
        assert_eq!(
            f.metadata,
            vec![
                ("format".into(), "pt".into()),
                ("hand.written".into(), "yes".into())
            ]
        );
    }

    /// The header is the one thing in this format that can lie, and every one of
    /// these lies has to be caught rather than followed into someone else's
    /// bytes.
    #[test]
    fn a_header_that_disagrees_with_its_buffer_is_refused() {
        let ok = handmade();
        assert!(File::parse(&ok).is_ok());

        // Truncation at every length: no panic, no partial parse.
        for n in 0..ok.len() {
            assert!(File::parse(&ok[..n]).is_err(), "prefix of {n} parsed");
        }

        let build = |header: json::Value, data: usize| {
            let h = header.encode().into_bytes();
            let mut out = (h.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(&h);
            out.extend_from_slice(&vec![0u8; data]);
            out
        };
        let entry = |dtype: &str, shape: Vec<u64>, a: u64, b: u64| {
            json::object(vec![
                ("dtype", json::string(dtype)),
                (
                    "shape",
                    json::Value::Array(shape.into_iter().map(json::Value::U).collect()),
                ),
                (
                    "data_offsets",
                    json::Value::Array(vec![json::Value::U(a), json::Value::U(b)]),
                ),
            ])
        };

        // An extent that is not the tensor's size.
        let bad = build(json::object(vec![("x", entry("F32", vec![2], 0, 12))]), 12);
        assert!(matches!(File::parse(&bad), Err(Error::Malformed(_))));

        // Two tensors claiming overlapping bytes.
        let bad = build(
            json::object(vec![
                ("x", entry("U8", vec![4], 0, 4)),
                ("y", entry("U8", vec![4], 2, 6)),
            ]),
            6,
        );
        assert!(File::parse(&bad).is_err(), "an overlap was accepted");

        // A gap between tensors: bytes nothing accounts for.
        let bad = build(
            json::object(vec![
                ("x", entry("U8", vec![4], 0, 4)),
                ("y", entry("U8", vec![4], 8, 12)),
            ]),
            12,
        );
        assert!(File::parse(&bad).is_err(), "a gap was accepted");

        // A buffer larger than what is described.
        let bad = build(json::object(vec![("x", entry("U8", vec![4], 0, 4))]), 99);
        assert!(File::parse(&bad).is_err(), "trailing bytes were accepted");

        // An offset past the end of the buffer.
        let bad = build(json::object(vec![("x", entry("U8", vec![4], 0, 4))]), 2);
        assert!(File::parse(&bad).is_err());

        // A declared header length past the end of the file.
        let mut bad = ok.clone();
        bad[0..8].copy_from_slice(&(ok.len() as u64 * 2).to_le_bytes());
        assert!(File::parse(&bad).is_err());

        // An absurd header length is refused by the bound rather than allocated.
        let mut bad = ok.clone();
        bad[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(File::parse(&bad).is_err());

        // A dtype the format does not have.
        let bad = build(json::object(vec![("x", entry("F80", vec![4], 0, 4))]), 4);
        assert!(matches!(File::parse(&bad), Err(Error::Unsupported(_))));

        // A header that is not an object.
        let bad = build(json::Value::Array(vec![]), 0);
        assert!(File::parse(&bad).is_err());

        // `__metadata__` that is not string→string.
        let bad = build(
            json::object(vec![(
                "__metadata__",
                json::object(vec![("k", json::Value::U(1))]),
            )]),
            0,
        );
        assert!(File::parse(&bad).is_err());
    }

    /// I4: the importer's claim is checked by the importer, and the report says
    /// what it checked.
    #[test]
    fn import_verifies_every_tensor_against_the_source() {
        let bytes = handmade();
        let imported = import(
            &bytes,
            &ImportOpts {
                name: "test/imported".into(),
                source_path: "hand.safetensors".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let r = &imported.report;
        assert_eq!(r.verified_tensors, 3);
        assert_eq!(r.verified_bytes, 48);
        assert_eq!(r.source_size, bytes.len() as u64);
        assert_eq!(r.source_digest, HashAlgo::default().digest(&bytes));

        // I1: nothing invented. The absences are recorded as decisions.
        let items: Vec<&str> = r.assumptions.iter().map(|a| a.item.as_str()).collect();
        assert!(items.contains(&"license"), "{items:?}");
        assert!(items.contains(&"arch.family"), "{items:?}");
        for a in &r.assumptions {
            if a.item == "license" || a.item == "arch.family" {
                assert_eq!(a.action, "field omitted");
            }
        }

        // I2: the metadata key with no schema is preserved, and named.
        assert!(r
            .unrepresented
            .iter()
            .any(|n| n.item == "__metadata__.hand.written"));
        assert!(!r
            .unrepresented
            .iter()
            .any(|n| n.item == "__metadata__.format"));

        // The container it produces validates, and holds the report.
        let c = crate::container::Container::open(
            crate::container::pack(
                &imported.objects,
                &imported.root,
                &crate::container::PackOptions::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let rep = crate::container::verify(&c).unwrap();
        assert!(rep.dangling.is_empty(), "{:?}", rep.dangling);
        assert!(rep.mistyped.is_empty());
        let prov = c
            .index
            .iter()
            .find(|e| e.otype == otype::PROVENANCE)
            .expect("a Provenance object");
        let v = c.get_value(&prov.digest).unwrap();
        assert_eq!(
            v.get("t").and_then(|t| t.as_str()),
            Some("omni.prov/import")
        );
        assert_eq!(
            v.get("source")
                .and_then(|s| s.get("format"))
                .and_then(|f| f.as_str()),
            Some("safetensors")
        );
        // The counts in the file match the counts the caller was handed.
        assert_eq!(
            v.get("verification")
                .and_then(|x| x.get("tensors_checked"))
                .and_then(|x| x.as_u64()),
            Some(3)
        );
        // And the Foreign object is in the container with the bytes in it.
        let foreign = c
            .index
            .iter()
            .find(|e| e.otype == otype::FOREIGN)
            .expect("a Foreign object");
        let fv = c.get_value(&foreign.digest).unwrap();
        let kept = fv.get("bytes").and_then(|b| b.as_bytes()).unwrap();
        assert_eq!(
            json::parse(kept)
                .unwrap()
                .get("hand.written")
                .and_then(|v| v.as_str()),
            Some("yes")
        );
    }

    /// A tampered source must not import quietly. The verification pass is what
    /// stands between a corrupted download and a model that looks fine.
    #[test]
    fn import_is_reproducible_and_addresses_its_source() {
        let bytes = handmade();
        let opts = ImportOpts {
            name: "test/imported".into(),
            source_path: "hand.safetensors".into(),
            ..Default::default()
        };
        let a = import(&bytes, &opts).unwrap();
        let b = import(&bytes, &opts).unwrap();
        assert_eq!(
            a.root, b.root,
            "the same file must import to the same graph"
        );

        // A different file is a different digest in the report, and a different
        // root: content addressing all the way up.
        let mut other = bytes.clone();
        let n = other.len();
        other[n - 1] ^= 0x40;
        let c = import(&other, &opts).unwrap();
        assert_ne!(c.root, a.root);
        assert_ne!(c.report.source_digest, a.report.source_digest);
    }

    fn descs_of(objects: &[Object], hash: HashAlgo) -> impl Fn(&Digest) -> Option<TensorDesc> + '_ {
        move |d: &Digest| {
            objects
                .iter()
                .find(|o| &o.digest(hash) == d)
                .and_then(|o| crate::cbor::decode(&o.payload).ok())
                .and_then(|v| TensorDesc::from_value(&v).ok())
        }
    }

    /// E4, and the claim the capability matrix makes: for tensors safetensors can
    /// name, `import(export(m))` reproduces every byte.
    #[test]
    fn export_then_import_reproduces_every_tensor_exactly() {
        let source = handmade();
        let imported = import(&source, &ImportOpts::default()).unwrap();
        let hash = HashAlgo::default();
        let store = store_of(&imported.objects, hash);
        let ctx = Ctx::new(&store);
        let table = table_of(&imported.objects, &imported.root, hash).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &imported.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let d = descs_of(&imported.objects, hash);

        let p = plan(&ctx, &table, &manifest, &d).unwrap();
        assert_eq!(p.tensors.len(), 3);
        assert_eq!(p.bytes, 48);
        // The import attached a provenance object, and safetensors cannot carry
        // it — so this export is *not* lossless, and says which thing is lost.
        assert!(!p.lossless());
        assert!(
            p.loss.iter().any(|l| l.item == "provenance"),
            "{:?}",
            p.loss
        );
        // E2: no consent, no bytes.
        assert!(matches!(
            export(&ctx, &table, &p, &d, &[], false),
            Err(Error::Lossy(_))
        ));

        let written = export(&ctx, &table, &p, &d, &[], true).unwrap();
        // The file we wrote parses under the same validation as any other.
        let f = File::parse(&written).unwrap();
        let g = File::parse(&source).unwrap();
        assert_eq!(f.entries.len(), g.entries.len());
        for e in &g.entries {
            let mine = f
                .get(&e.name)
                .unwrap_or_else(|| panic!("{} missing", e.name));
            assert_eq!(mine.st_dtype, e.st_dtype, "{}", e.name);
            assert_eq!(mine.shape, e.shape, "{}", e.name);
            assert_eq!(f.tensor(mine), g.tensor(e), "{} differs", e.name);
        }
        // E3: the exported file points back at what produced it.
        assert!(f.metadata.iter().any(|(k, _)| k == "omni.exporter"));

        // And round-tripping through OMNI again gives the same tensor digests:
        // the identities survive, which is what "lossless" has to mean here.
        let again = import(&written, &ImportOpts::default()).unwrap();
        let blobs = |objs: &[Object]| -> std::collections::BTreeSet<Digest> {
            objs.iter()
                .filter(|o| o.otype == otype::BLOB)
                .map(|o| o.digest(hash))
                .collect()
        };
        assert_eq!(blobs(&again.objects), blobs(&imported.objects));
    }

    /// I2 is only worth doing if the preserved keys can come back. This is the
    /// whole loop: a key with no OMNI schema goes into a `Foreign` object on
    /// import and into the header again on export.
    #[test]
    fn a_preserved_metadata_key_comes_back_on_export() {
        let source = handmade();
        let imported = import(&source, &ImportOpts::default()).unwrap();
        let hash = HashAlgo::default();
        let store = store_of(&imported.objects, hash);
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &imported.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let objects = |d: &Digest| {
            crate::store::Store::resolve(&store, d)
                .ok()
                .flatten()
                .and_then(|b| crate::cbor::decode(&b).ok())
        };
        let kept = preserved_metadata(&manifest, &objects);
        assert_eq!(kept, vec![("hand.written".to_string(), "yes".to_string())]);

        // Through an export, it is a header key again.
        let ctx = Ctx::new(&store);
        let table = table_of(&imported.objects, &imported.root, hash).unwrap();
        let d = descs_of(&imported.objects, hash);
        let p = plan(&ctx, &table, &manifest, &d).unwrap();
        let written = export(&ctx, &table, &p, &d, &kept, true).unwrap();
        let f = File::parse(&written).unwrap();
        assert!(
            f.metadata
                .iter()
                .any(|(k, v)| k == "hand.written" && v == "yes"),
            "{:?}",
            f.metadata
        );
        // Every key the source had is in the export.
        let g = File::parse(&source).unwrap();
        for (k, v) in &g.metadata {
            assert!(
                f.metadata.iter().any(|(k2, v2)| k2 == k && v2 == v),
                "`{k}` did not survive the round trip"
            );
        }

        // A Foreign object from some other format is not raided for keys: those
        // belong to a convention this exporter does not speak.
        let gguf = crate::cbor::Value::map(vec![
            ("t", crate::cbor::Value::text("omni.core/foreign")),
            ("format", crate::cbor::Value::text("gguf")),
            ("media_type", crate::cbor::Value::text("application/json")),
            (
                "bytes",
                crate::cbor::Value::Bytes(br#"{"general.name":"x"}"#.to_vec()),
            ),
        ]);
        let obj = Object::structure(otype::FOREIGN, &gguf);
        let d2 = obj.digest(hash);
        let manifest2 = crate::cbor::Value::map(vec![(
            "foreign",
            crate::cbor::Value::Array(vec![crate::cbor::Value::Array(vec![
                crate::cbor::Value::U(otype::FOREIGN as u64),
                crate::cbor::Value::Bytes(d2.to_vec()),
            ])]),
        )]);
        let only = |q: &Digest| (*q == d2).then(|| crate::cbor::decode(&obj.payload).unwrap());
        assert!(preserved_metadata(&manifest2, &only).is_empty());
    }

    /// A dtype safetensors cannot name is reported, not silently widened — and
    /// with consent it is written as F32 with the change on the record.
    #[test]
    fn a_dtype_safetensors_cannot_name_is_reported_before_it_is_widened() {
        let (objects, root) = ModelBuilder::new("test/exotic")
            .tensor(TensorSpec {
                name: "q".into(),
                shape: vec![8],
                dtype: DType::I4,
                axes: None,
                semantic: "weight",
                // Eight 4-bit elements in four bytes.
                data: vec![0x21, 0x43, 0x65, 0x07],
                layout: None,
            })
            .build();
        let hash = HashAlgo::default();
        let store = store_of(&objects, hash);
        let ctx = Ctx::new(&store);
        let table = table_of(&objects, &root, hash).unwrap();
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let d = descs_of(&objects, hash);

        let p = plan(&ctx, &table, &manifest, &d).unwrap();
        assert!(!p.lossless());
        let note = p
            .loss
            .iter()
            .find(|l| l.item.contains("dtype"))
            .expect("the dtype loss is named");
        assert!(note.reason.contains("F32"), "{}", note.reason);
        assert!(matches!(
            export(&ctx, &table, &p, &d, &[], false),
            Err(Error::Lossy(_))
        ));

        // With consent: real values, in F32, and the file is well-formed.
        let written = export(&ctx, &table, &p, &d, &[], true).unwrap();
        let f = File::parse(&written).unwrap();
        assert_eq!(f.entries[0].st_dtype, "F32");
        assert_eq!(f.entries[0].shape, vec![8]);
        let vals: Vec<f32> = f
            .tensor(&f.entries[0])
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // i4 is signed: the nibbles are 1,2,3,4,5,6,7,0.
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]);

        // The loss report is JSON and names the tensor.
        let text = p.loss_report(&root, hash);
        let v = json::parse(text.as_bytes()).unwrap();
        assert_eq!(v.get("lossless").and_then(|b| b.as_bool()), Some(false));
        assert!(text.contains("tensor `q` dtype"), "{text}");
    }

    /// safetensors stores a boolean in a whole byte; §04.3 gives `bool` a bit.
    /// The tensor keeps its type and describes its storage, so a reader still
    /// learns that this is a mask — and the bytes come back unchanged.
    #[test]
    fn a_boolean_mask_keeps_its_type_and_its_byte_per_element_storage() {
        let mask = [1u8, 0, 1, 1, 0, 0, 1, 0];
        let header = json::object(vec![(
            "mask",
            json::object(vec![
                ("dtype", json::string("BOOL")),
                ("shape", json::Value::Array(vec![json::Value::U(8)])),
                (
                    "data_offsets",
                    json::Value::Array(vec![json::Value::U(0), json::Value::U(8)]),
                ),
            ]),
        )])
        .encode()
        .into_bytes();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&mask);

        // Eight bools in eight bytes, which the dtype alone would call one byte.
        let f = File::parse(&bytes).unwrap();
        assert_eq!(f.entries[0].dtype, DType::Bool);
        assert_eq!(f.entries[0].len(), 8);
        assert_eq!(
            DType::Bool.packed_bytes(8),
            1,
            "the dtype packs eight to a byte"
        );
        assert_eq!(stored_bytes(&DType::Bool, &[8]), 8, "the format does not");

        let imported = import(&bytes, &ImportOpts::default()).unwrap();
        assert_eq!(imported.report.verified_tensors, 1);
        assert_eq!(imported.report.verified_bytes, 8);

        // The descriptor says `bool`, and says how it is stored.
        let hash = HashAlgo::default();
        let store = store_of(&imported.objects, hash);
        let table = table_of(&imported.objects, &imported.root, hash).unwrap();
        let d = descs_of(&imported.objects, hash);
        let desc = d(&table.tensors["mask"].1).unwrap();
        assert_eq!(desc.dtype, DType::Bool);
        assert_eq!(desc.layout, layout_of(&DType::Bool));

        // R-T02 agrees with the layout rather than with the dtype, so the
        // container validates.
        let ctx = Ctx::new(&store);
        let findings = desc.check(&ctx, "mask");
        assert!(
            findings.is_empty(),
            "{}",
            findings
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        );

        // The values read back as the booleans they are.
        assert_eq!(
            desc.value.eval(&ctx).unwrap().data,
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        );

        // And an export writes BOOL, one byte each, byte-identical to the source.
        let manifest = crate::cbor::decode(
            &crate::store::Store::resolve(&store, &imported.root)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let p = plan(&ctx, &table, &manifest, &d).unwrap();
        assert_eq!(p.tensors[0].1, "BOOL");
        assert_eq!(p.bytes, 8);
        let written = export(&ctx, &table, &p, &d, &[], true).unwrap();
        let g = File::parse(&written).unwrap();
        assert_eq!(g.tensor(&g.entries[0]), &mask[..]);
    }

    /// An empty file is a legal file, and a zero-element tensor is a legal
    /// tensor. Neither may be a special case that panics.
    #[test]
    fn the_degenerate_cases_are_ordinary() {
        // No tensors at all.
        let header = json::object(vec![]).encode().into_bytes();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        let f = File::parse(&bytes).unwrap();
        assert!(f.entries.is_empty());

        // A tensor with a zero dimension: zero bytes, and it still has to appear.
        let header = json::object(vec![(
            "empty",
            json::object(vec![
                ("dtype", json::string("F32")),
                (
                    "shape",
                    json::Value::Array(vec![json::Value::U(0), json::Value::U(4)]),
                ),
                (
                    "data_offsets",
                    json::Value::Array(vec![json::Value::U(0), json::Value::U(0)]),
                ),
            ]),
        )])
        .encode()
        .into_bytes();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        let f = File::parse(&bytes).unwrap();
        assert_eq!(f.entries.len(), 1);
        assert!(f.entries[0].is_empty());
        assert_eq!(f.entries[0].numel(), 0);
    }
}
