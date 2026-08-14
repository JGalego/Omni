//! The OMNI container binary format (§02).
//!
//! Implements the header, segment framing, fixed-layout object index, trailer
//! and the two-pass layout that lets a file carry byte-identical front and back
//! superblocks.

use crate::cbor::{self, Value};
use crate::crc32c::crc32c;
use crate::sha256::{hex, sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MAGIC: [u8; 8] = [0x89, b'O', b'M', b'N', b'I', 0x0d, 0x0a, 0x1a];
pub const MAGIC_END: [u8; 8] = [0x1a, 0x0a, 0x0d, b'I', b'N', b'M', b'O', 0x89];
pub const SEG_MAGIC: [u8; 4] = *b"OSEG";
pub const IDX_MAGIC: [u8; 4] = *b"OIDX";

pub const HEADER_SIZE: usize = 128;
pub const TRAILER_SIZE: usize = 64;
pub const SEG_HEADER_SIZE: usize = 32;
pub const IDX_HEADER_SIZE: usize = 64;
pub const IDX_ENTRY_SIZE: usize = 64;

/// Multicodec digest algorithm codes (§01.3).
pub const HASH_SHA256: u8 = 0x12;
pub const HASH_BLAKE3_256: u8 = 0x1e;

/// The digest algorithm a container is built with (§01.3, §03.5.1).
///
/// A container declares exactly one primary algorithm in its header, and every
/// digest in it — object identities, the root, the superblock digest, index
/// entries — uses that one. Mixing algorithms within a container would make
/// content addressing ambiguous: the same bytes would have two identities.
///
/// Agility lives *between* containers, not inside one. §12.11's hash-migration
/// story is "rehash the objects, rewrite the graph"; it costs one pass over the
/// data and no re-uploads, precisely because the algorithm is a single header
/// field rather than a per-object choice.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum HashAlgo {
    /// §03.5.1's default: parallel, tree-structured, Bao-verifiable.
    #[default]
    Blake3_256,
    /// Mandatory for interoperability with OCI, Sigstore and SLSA.
    Sha256,
}

impl HashAlgo {
    pub fn code(self) -> u8 {
        match self {
            HashAlgo::Blake3_256 => HASH_BLAKE3_256,
            HashAlgo::Sha256 => HASH_SHA256,
        }
    }

    pub fn from_code(code: u8) -> Option<HashAlgo> {
        match code {
            HASH_BLAKE3_256 => Some(HashAlgo::Blake3_256),
            HASH_SHA256 => Some(HashAlgo::Sha256),
            _ => None,
        }
    }

    /// The multihash-style name used in digest prefixes and CLI output.
    pub fn name(self) -> &'static str {
        match self {
            HashAlgo::Blake3_256 => "blake3-256",
            HashAlgo::Sha256 => "sha2-256",
        }
    }

    /// The short prefix printed before a digest, as in `b3:1a2b…`.
    pub fn prefix(self) -> &'static str {
        match self {
            HashAlgo::Blake3_256 => "b3",
            HashAlgo::Sha256 => "sha2",
        }
    }

    pub fn digest(self, data: &[u8]) -> Digest {
        match self {
            HashAlgo::Blake3_256 => crate::blake3::blake3(data),
            HashAlgo::Sha256 => sha256(data),
        }
    }

    /// A domain-separated digest for everything that is not an object
    /// (§03.5.3): expression identities, plan keys, derived encryption keys.
    ///
    /// BLAKE3 has a derive-key mode built for exactly this. SHA-256 has no
    /// such mode, so the context string is prefixed with a separator that
    /// cannot occur inside it; the property that matters is that a digest
    /// computed for one purpose can never be replayed as another.
    pub fn domain_digest(self, context: &str, data: &[u8]) -> Digest {
        match self {
            HashAlgo::Blake3_256 => crate::blake3::derive_key(context, data),
            HashAlgo::Sha256 => {
                let mut buf = Vec::with_capacity(context.len() + data.len() + 1);
                buf.extend_from_slice(context.as_bytes());
                buf.push(0x00);
                buf.extend_from_slice(data);
                crate::sha256::sha256(&buf)
            }
        }
    }

    /// Parses a CLI-facing name. Accepts both the multihash name and the
    /// common short form.
    pub fn parse(s: &str) -> Option<HashAlgo> {
        match s {
            "blake3" | "blake3-256" | "b3" => Some(HashAlgo::Blake3_256),
            "sha256" | "sha2-256" | "sha-256" => Some(HashAlgo::Sha256),
            _ => None,
        }
    }
}

/// The object-type registry of §01.9. Values `0x0000–0x7FFF` are reserved for
/// the specification; `0x8000` and above belong to plugin namespaces.
pub mod otype {
    pub const BLOB: u16 = 0x0000;
    pub const MANIFEST: u16 = 0x0001;
    pub const METADATA: u16 = 0x0002;
    pub const MODEL: u16 = 0x0003;
    pub const TENSOR_TABLE: u16 = 0x0004;
    pub const TENSOR_DESC: u16 = 0x0005;
    pub const CHUNK_LIST: u16 = 0x0006;
    pub const CODEBOOK: u16 = 0x0007;
    pub const GRAPH_MODULE: u16 = 0x0008;
    pub const DIALECT_REF: u16 = 0x0009;
    pub const TOKENIZER: u16 = 0x000A;
    pub const CHAT_TEMPLATE: u16 = 0x000B;
    pub const ADAPTER: u16 = 0x000C;
    pub const TRAINING_STATE: u16 = 0x000D;
    pub const SHARD_MAP: u16 = 0x000E;
    pub const RUNTIME_CACHE: u16 = 0x000F;
    pub const CAPABILITY_SET: u16 = 0x0010;
    pub const PLAN: u16 = 0x0011;
    pub const SIGNATURE: u16 = 0x0012;
    pub const PROVENANCE: u16 = 0x0013;
    pub const BAO_TREE: u16 = 0x0014;
    pub const OBJECT_INDEX: u16 = 0x0015;
    pub const NAME_INDEX: u16 = 0x0016;
    pub const SCHEMA: u16 = 0x0017;
    pub const ROSETTA: u16 = 0x0018;
    pub const FOREIGN: u16 = 0x0019;
    pub const DATASET: u16 = 0x001A;
    pub const PIN: u16 = 0x001B;
    pub const SHARDED_MAP: u16 = 0x001C;
    pub const ALT_DIGEST: u16 = 0x001D;
    pub const PLUGIN_MODULE: u16 = 0x001E;
    pub const EXTENSION: u16 = 0x001F;
    pub const EVALUATION: u16 = 0x0020;

    pub fn name(t: u16) -> &'static str {
        match t {
            BLOB => "Blob",
            MANIFEST => "Manifest",
            METADATA => "Metadata",
            MODEL => "Model",
            TENSOR_TABLE => "TensorTable",
            TENSOR_DESC => "TensorDesc",
            CHUNK_LIST => "ChunkList",
            CODEBOOK => "Codebook",
            GRAPH_MODULE => "GraphModule",
            DIALECT_REF => "DialectRef",
            TOKENIZER => "Tokenizer",
            CHAT_TEMPLATE => "ChatTemplate",
            ADAPTER => "Adapter",
            TRAINING_STATE => "TrainingState",
            SHARD_MAP => "ShardMap",
            RUNTIME_CACHE => "RuntimeCache",
            CAPABILITY_SET => "CapabilitySet",
            PLAN => "Plan",
            SIGNATURE => "Signature",
            PROVENANCE => "Provenance",
            BAO_TREE => "BaoTree",
            OBJECT_INDEX => "ObjectIndex",
            NAME_INDEX => "NameIndex",
            SCHEMA => "Schema",
            ROSETTA => "Rosetta",
            FOREIGN => "Foreign",
            DATASET => "Dataset",
            PIN => "Pin",
            SHARDED_MAP => "ShardedMap",
            ALT_DIGEST => "AltDigest",
            PLUGIN_MODULE => "PluginModule",
            EXTENSION => "Extension",
            EVALUATION => "Evaluation",
            _ if t >= 0x8000 => "Extension(plugin)",
            _ => "Unknown",
        }
    }

    /// The schema URI (`t`) a given object type must carry, where the
    /// specification fixes one. Used by R-O02: the index's `otype` and the
    /// object's own `t` must agree.
    pub fn schema_uri(t: u16) -> Option<&'static str> {
        Some(match t {
            MANIFEST => "omni.core/manifest",
            METADATA => "omni.meta/model",
            MODEL => "omni.core/model",
            TENSOR_TABLE => "omni.tensor/table",
            TENSOR_DESC => "omni.tensor/desc",
            CHUNK_LIST => "omni.tensor/chunklist",
            CODEBOOK => "omni.tensor/codebook",
            GRAPH_MODULE => "omni.ir/module",
            DIALECT_REF => "omni.ir/dialect",
            PLUGIN_MODULE => "omni.plugin/manifest",
            TOKENIZER => "omni.tok/tokenizer",
            CHAT_TEMPLATE => "omni.tok/chat-template",
            ADAPTER => "omni.adapt/adapter",
            TRAINING_STATE => "omni.train/state",
            SHARD_MAP => "omni.train/shardmap",
            CAPABILITY_SET => "omni.rt/capabilities",
            PLAN => "omni.rt/plan",
            SIGNATURE => "omni.sec/signature",
            BAO_TREE => "omni.stream/bao",
            DATASET => "omni.meta/dataset",
            EVALUATION => "omni.meta/evaluation",
            _ => return None,
        })
    }
}

pub mod seg {
    pub const BLOB: u16 = 1;
    pub const OBJ: u16 = 2;
    pub const INDEX: u16 = 3;
    pub const SUPER: u16 = 4;
    pub const SIG: u16 = 5;
    pub const PAD: u16 = 6;

    pub fn name(k: u16) -> &'static str {
        match k {
            BLOB => "BLOB",
            OBJ => "OBJ",
            INDEX => "INDEX",
            SUPER => "SUPER",
            SIG => "SIG",
            PAD => "PAD",
            _ => "UNKNOWN",
        }
    }
}

pub mod hflags {
    pub const SEALED: u32 = 1 << 0;
    pub const FRONT_SB: u32 = 1 << 1;
    pub const APPEND_LOG: u32 = 1 << 2;
    pub const SIGNED: u32 = 1 << 3;
    pub const ENCRYPTED: u32 = 1 << 4;
    pub const PARTIAL: u32 = 1 << 5;
    pub const DERIVED_ONLY: u32 = 1 << 6;
    pub const NO_MMAP_SAFE: u32 = 1 << 7;
}

pub mod oflags {
    pub const CRITICAL: u8 = 1 << 0;
    pub const CACHEABLE: u8 = 1 << 1;
    pub const EXTERNAL: u8 = 1 << 2;
    pub const LOSSY: u8 = 1 << 3;
    pub const ENCRYPTED: u8 = 1 << 4;
    pub const HAS_BAO: u8 = 1 << 5;
    pub const SAFE_TO_COPY: u8 = 1 << 6;
    pub const STRUCTURAL: u8 = 1 << 7;
}

/// §03.5.3 context string for the derived file UUID.
const UUID_CONTEXT: &str = "omni/1.0 uuid";

pub type Digest = [u8; 32];

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Rule(&'static str, String),
    Cbor(cbor::Error),
    NotFound(String),
    /// A codec problem: unsupported, malformed, or over a §03.7.4 bound.
    Codec(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Rule(id, msg) => write!(f, "{id}: {msg}"),
            Error::Cbor(e) => write!(f, "cbor: {e}"),
            Error::Codec(m) => write!(f, "codec: {m}"),
            Error::NotFound(d) => write!(f, "object not found: {d}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<crate::codec::Error> for Error {
    fn from(e: crate::codec::Error) -> Self {
        Error::Codec(e.to_string())
    }
}

impl From<cbor::Error> for Error {
    fn from(e: cbor::Error) -> Self {
        Error::Cbor(e)
    }
}

type Res<T> = Result<T, Error>;

fn rule(id: &'static str, msg: impl Into<String>) -> Error {
    Error::Rule(id, msg.into())
}

fn round_up(n: usize, a: usize) -> usize {
    debug_assert!(a.is_power_of_two());
    (n + a - 1) & !(a - 1)
}

// ------------------------------------------------------------------ object --

/// An object as held in memory before or after packing.
#[derive(Clone)]
pub struct Object {
    pub otype: u16,
    /// The object's *logical* bytes. Digests are over these, always (§03.5.2),
    /// which is why recompressing a container changes no identities.
    pub payload: Vec<u8>,
    pub oflags: u8,
    /// The stored form, when this copy is compressed: the codec id and the
    /// compressed bytes. Compression is a property of a stored copy, never of
    /// the object (§01.2).
    pub stored: Option<(u8, Vec<u8>)>,
}

impl Object {
    /// The bytes this copy occupies in a container.
    pub fn stored_bytes(&self) -> &[u8] {
        match &self.stored {
            Some((_, b)) => b,
            None => &self.payload,
        }
    }

    /// The codec id of this stored copy.
    pub fn codec_id(&self) -> u8 {
        match &self.stored {
            Some((c, _)) => *c,
            None => crate::codec::id::RAW,
        }
    }

    /// Returns this object with a compressed stored form. Keeps whichever
    /// encoding is smaller: a codec that expands an object is not a
    /// compression, and storing the expansion would be strictly worse.
    pub fn compressed(
        mut self,
        codec: &crate::codec::Codec,
    ) -> Result<Object, crate::codec::Error> {
        if matches!(codec, crate::codec::Codec::Raw) {
            self.stored = None;
            return Ok(self);
        }
        let bytes = codec.encode(&self.payload)?;
        if bytes.len() < self.payload.len() {
            self.stored = Some((codec.id(), bytes));
        } else {
            self.stored = None;
        }
        Ok(self)
    }

    pub fn structure(otype: u16, v: &Value) -> Object {
        Object {
            otype,
            payload: v.encode(),
            oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
            stored: None,
        }
    }

    pub fn blob(payload: Vec<u8>) -> Object {
        Object {
            otype: otype::BLOB,
            payload,
            oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
            stored: None,
        }
    }

    /// The object's identity under `algo` (§03.5.2: a data object hashes its
    /// logical bytes, a structure object its canonical CBOR).
    pub fn digest(&self, algo: HashAlgo) -> Digest {
        algo.digest(&self.payload)
    }
}

// ------------------------------------------------------------------- write --

pub struct PackOptions {
    pub log2_align: u8,
    pub creator: String,
    /// Zero timestamps and derive the UUID from the root digest (§01.10).
    pub reproducible: bool,
    /// The container's primary digest algorithm (§03.5.1).
    pub hash: HashAlgo,
    /// Codec for data objects (§03.7). Structure objects stay `raw`: they are
    /// tiny, they are on the parse hot path, and §03.1 already keeps tensor
    /// payloads out of them.
    pub codec: crate::codec::Codec,
}

impl Default for PackOptions {
    fn default() -> Self {
        PackOptions {
            log2_align: 12,
            creator: format!("omni-rs/{}", env!("CARGO_PKG_VERSION")),
            reproducible: true,
            hash: HashAlgo::default(),
            codec: crate::codec::Codec::Raw,
        }
    }
}

struct Placed {
    digest: Digest,
    otype: u16,
    oflags: u8,
    offset: u64,
    /// Stored length: what the file holds.
    len: u64,
    /// Logical length: what the digest covers.
    logical: u64,
    codec: u8,
}

/// Packs an object set into a sealed `core`-profile container.
///
/// Layout: header · SUPER(front) · OBJ · [PAD] · BLOB · [PAD] · INDEX ·
/// SUPER(back) · trailer. The front and back superblocks are byte-identical
/// (R-C10), which requires a fixed-point layout pass because the superblock
/// records the offsets of the segments that contain it.
pub fn pack(objects: &[Object], root: &Digest, opts: &PackOptions) -> Res<Vec<u8>> {
    pack_partial(objects, &[], root, opts)
}

/// [`pack`], plus objects that are *described but not present*.
///
/// An entry in `external` contributes an index entry with `EXTERNAL` set and no
/// extent: the digest, type and logical length are recorded, the bytes are not
/// in the file. That is §13.8's index-only container — a catalogue of a 700 GB
/// model in a few megabytes — and it is also what a partial mirror looks like
/// mid-transfer. The header gains `PARTIAL` so a reader knows before it plans
/// anything that this container is incomplete by construction rather than
/// truncated by accident.
///
/// An `external` entry whose digest is also among `objects` is dropped: a
/// present object beats an absent description of one, and an index cannot hold
/// both (R-C11 requires strictly increasing digests).
pub fn pack_partial(
    objects: &[Object],
    external: &[IndexEntry],
    root: &Digest,
    opts: &PackOptions,
) -> Res<Vec<u8>> {
    if !(6..=30).contains(&opts.log2_align) {
        return Err(rule("R-C04", "log2_align out of range"));
    }
    let align = 1usize << opts.log2_align;

    // Deduplicate by digest and order deterministically by (otype, digest).
    let mut uniq: BTreeMap<(u16, Digest), &Object> = BTreeMap::new();
    for o in objects {
        uniq.insert((o.otype, o.digest(opts.hash)), o);
    }
    // Apply the requested codec to data objects that do not already carry a
    // stored form. Digests are unaffected, so a compressed container dedups
    // against an uncompressed one object for object (§01.2).
    let compressed: Vec<Object> = if matches!(opts.codec, crate::codec::Codec::Raw) {
        Vec::new()
    } else {
        uniq.values()
            .filter(|o| o.otype == otype::BLOB && o.stored.is_none())
            .map(|o| (*o).clone().compressed(&opts.codec))
            .collect::<Result<Vec<Object>, crate::codec::Error>>()?
    };
    for c in &compressed {
        uniq.insert((c.otype, c.digest(opts.hash)), c);
    }
    let ordered: Vec<&Object> = uniq.values().copied().collect();
    let (blobs, structs): (Vec<&Object>, Vec<&Object>) =
        ordered.iter().partition(|o| o.otype == otype::BLOB);

    let present: BTreeSet<Digest> = uniq.keys().map(|(_, d)| *d).collect();
    let mut absent: Vec<IndexEntry> = external
        .iter()
        .filter(|e| !present.contains(&e.digest))
        .cloned()
        .collect();
    absent.sort_by_key(|e| e.digest);
    absent.dedup_by_key(|e| e.digest);

    // Fixed-point layout: the superblock's size affects the offsets it records.
    let mut sb_reserve = 4096usize;
    let (layout, sb_bytes) = loop {
        let l = compute_layout(&structs, &blobs, &absent, align, sb_reserve, opts.hash);
        let sb = superblock_value(&l, root, align, opts).encode();
        let need = round_up(SEG_HEADER_SIZE + sb.len(), 64);
        if need <= sb_reserve {
            break (l, sb);
        }
        sb_reserve = round_up(need, 64);
    };

    let mut out = vec![0u8; layout.file_size];

    // --- header ---------------------------------------------------------
    let uuid = if opts.reproducible {
        derive_uuid(opts.hash, root)
    } else {
        derive_uuid(
            opts.hash,
            &opts.hash.digest(&layout.file_size.to_le_bytes()),
        )
    };
    let mut flags = hflags::SEALED | hflags::FRONT_SB;
    if structs.iter().any(|o| o.otype == otype::SIGNATURE) {
        flags |= hflags::SIGNED;
    }
    if !layout.external.is_empty() {
        flags |= hflags::PARTIAL;
    }
    out[0..8].copy_from_slice(&MAGIC);
    out[8..10].copy_from_slice(&1u16.to_le_bytes());
    out[10..12].copy_from_slice(&0u16.to_le_bytes());
    out[12] = 0x01; // little-endian
    out[13] = opts.log2_align;
    out[14..16].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
    out[16..32].copy_from_slice(&uuid);
    out[32] = opts.hash.code();
    out[33] = 0; // profile: core
    out[34] = 32; // digest_len
    out[35] = 0;
    out[36..40].copy_from_slice(&flags.to_le_bytes());
    out[40..48].copy_from_slice(&(layout.front_sb_payload_off as u64).to_le_bytes());
    out[48..56].copy_from_slice(&(sb_bytes.len() as u64).to_le_bytes());
    out[56..64].copy_from_slice(&(layout.file_size as u64).to_le_bytes());
    out[64..96].copy_from_slice(root);
    let mut creator = [0u8; 16];
    let cb = opts.creator.as_bytes();
    creator[..cb.len().min(16)].copy_from_slice(&cb[..cb.len().min(16)]);
    out[96..112].copy_from_slice(&creator);
    out[112..120].copy_from_slice(&0u64.to_le_bytes()); // created: 0 (reproducible)
    out[120..124].copy_from_slice(&0u32.to_le_bytes());
    let hcrc = crc32c(&out[0..124]);
    out[124..128].copy_from_slice(&hcrc.to_le_bytes());

    // --- segments -------------------------------------------------------
    write_segment(&mut out, layout.front_sb_hdr_off, seg::SUPER, 0, &sb_bytes);

    let mut obj_payload = vec![0u8; layout.obj_payload_len];
    for p in &layout.structs {
        // Keyed lookup, not a linear scan: `uniq` is already keyed by exactly
        // this pair. Scanning here made packing O(n²) in the object count,
        // which is invisible at 49 objects and fatal at a million.
        let o = uniq
            .get(&(p.otype, p.digest))
            .expect("placed object exists");
        let rel = p.offset as usize - layout.obj_payload_off;
        let b = o.stored_bytes();
        obj_payload[rel..rel + b.len()].copy_from_slice(b);
    }
    write_segment(&mut out, layout.obj_hdr_off, seg::OBJ, 1, &obj_payload);

    if let Some(pad) = layout.pad_before_blob {
        write_segment(&mut out, pad.0, seg::PAD, 2, &vec![0u8; pad.1]);
    }

    let mut blob_payload = vec![0u8; layout.blob_payload_len];
    for p in &layout.blobs {
        let o = uniq
            .get(&(otype::BLOB, p.digest))
            .expect("placed blob exists");
        let rel = p.offset as usize - layout.blob_payload_off;
        let b = o.stored_bytes();
        blob_payload[rel..rel + b.len()].copy_from_slice(b);
    }
    write_segment(&mut out, layout.blob_hdr_off, seg::BLOB, 3, &blob_payload);

    if let Some(pad) = layout.pad_before_index {
        write_segment(&mut out, pad.0, seg::PAD, 4, &vec![0u8; pad.1]);
    }

    let idx = build_index(&layout, opts.hash);
    write_segment(&mut out, layout.index_hdr_off, seg::INDEX, 5, &idx);

    write_segment(&mut out, layout.back_sb_hdr_off, seg::SUPER, 6, &sb_bytes);

    // --- trailer --------------------------------------------------------
    let t = layout.file_size - TRAILER_SIZE;
    out[t..t + 8].copy_from_slice(&(layout.back_sb_payload_off as u64).to_le_bytes());
    out[t + 8..t + 16].copy_from_slice(&(sb_bytes.len() as u64).to_le_bytes());
    out[t + 16..t + 48].copy_from_slice(&opts.hash.digest(&sb_bytes));
    out[t + 48..t + 52].copy_from_slice(&flags.to_le_bytes());
    let tcrc = crc32c(&out[t..t + 52]);
    out[t + 52..t + 56].copy_from_slice(&tcrc.to_le_bytes());
    out[t + 56..t + 64].copy_from_slice(&MAGIC_END);

    Ok(out)
}

struct Layout {
    front_sb_hdr_off: usize,
    front_sb_payload_off: usize,
    obj_hdr_off: usize,
    obj_payload_off: usize,
    obj_payload_len: usize,
    pad_before_blob: Option<(usize, usize)>,
    blob_hdr_off: usize,
    blob_payload_off: usize,
    blob_payload_len: usize,
    pad_before_index: Option<(usize, usize)>,
    index_hdr_off: usize,
    back_sb_hdr_off: usize,
    back_sb_payload_off: usize,
    file_size: usize,
    structs: Vec<Placed>,
    blobs: Vec<Placed>,
    /// Index entries for objects this container describes but does not hold.
    external: Vec<Placed>,
    n_entries: usize,
    /// How many stored copies are not `raw` — decides what the superblock's
    /// codec list has to name (§02.5).
    compressed: usize,
}

fn compute_layout(
    structs: &[&Object],
    blobs: &[&Object],
    external: &[IndexEntry],
    align: usize,
    sb_reserve: usize,
    algo: HashAlgo,
) -> Layout {
    let front_sb_hdr_off = HEADER_SIZE;
    let front_sb_payload_off = front_sb_hdr_off + SEG_HEADER_SIZE;
    let mut off = front_sb_hdr_off + sb_reserve;

    // OBJ segment: structure objects, 8-byte aligned within the payload.
    let obj_hdr_off = round_up(off, 8);
    let obj_payload_off = obj_hdr_off + SEG_HEADER_SIZE;
    let mut placed_structs = Vec::new();
    let mut rel = 0usize;
    for o in structs {
        rel = round_up(rel, 8);
        placed_structs.push(Placed {
            digest: o.digest(algo),
            otype: o.otype,
            oflags: o.oflags,
            offset: (obj_payload_off + rel) as u64,
            len: o.stored_bytes().len() as u64,
            logical: o.payload.len() as u64,
            codec: o.codec_id(),
        });
        rel += o.stored_bytes().len();
    }
    let obj_payload_len = rel;
    off = obj_payload_off + obj_payload_len;

    // BLOB segment: its payload must start on an `align` boundary (R-C08).
    // Insert a PAD segment to consume the gap when one is needed.
    let mut pad_before_blob = None;
    let mut blob_hdr_off = round_up(off, 8);
    let want_payload = round_up(blob_hdr_off + SEG_HEADER_SIZE, align);
    if want_payload != blob_hdr_off + SEG_HEADER_SIZE {
        let target_hdr = want_payload - SEG_HEADER_SIZE;
        let gap = target_hdr - blob_hdr_off;
        if gap >= SEG_HEADER_SIZE {
            pad_before_blob = Some((blob_hdr_off, gap - SEG_HEADER_SIZE));
        } else {
            // Not enough room for a PAD segment header: push out one alignment unit.
            let target_hdr = want_payload + align - SEG_HEADER_SIZE;
            pad_before_blob = Some((blob_hdr_off, target_hdr - blob_hdr_off - SEG_HEADER_SIZE));
        }
        blob_hdr_off = pad_before_blob.unwrap().0 + SEG_HEADER_SIZE + pad_before_blob.unwrap().1;
    }
    let blob_payload_off = blob_hdr_off + SEG_HEADER_SIZE;
    debug_assert_eq!(blob_payload_off % align, 0);

    let mut placed_blobs = Vec::new();
    let mut rel = 0usize;
    for o in blobs {
        rel = round_up(rel, align); // every data object is align-aligned (R-C08)
        placed_blobs.push(Placed {
            digest: o.digest(algo),
            otype: otype::BLOB,
            oflags: o.oflags,
            offset: (blob_payload_off + rel) as u64,
            len: o.stored_bytes().len() as u64,
            logical: o.payload.len() as u64,
            codec: o.codec_id(),
        });
        rel += o.stored_bytes().len();
    }
    let blob_payload_len = rel;
    off = blob_payload_off + blob_payload_len;

    // INDEX segment: payload aligned to max(align, 64) (R-C: §2.9.4).
    let ialign = align.max(64);
    let mut pad_before_index = None;
    let mut index_hdr_off = round_up(off, 8);
    let want_payload = round_up(index_hdr_off + SEG_HEADER_SIZE, ialign);
    if want_payload != index_hdr_off + SEG_HEADER_SIZE {
        let target_hdr = want_payload - SEG_HEADER_SIZE;
        let gap = target_hdr - index_hdr_off;
        if gap >= SEG_HEADER_SIZE {
            pad_before_index = Some((index_hdr_off, gap - SEG_HEADER_SIZE));
        } else {
            let target_hdr = want_payload + ialign - SEG_HEADER_SIZE;
            pad_before_index = Some((index_hdr_off, target_hdr - index_hdr_off - SEG_HEADER_SIZE));
        }
        index_hdr_off = pad_before_index.unwrap().0 + SEG_HEADER_SIZE + pad_before_index.unwrap().1;
    }
    // Described-but-absent objects: an index entry, no extent, EXTERNAL set.
    let placed_external: Vec<Placed> = external
        .iter()
        .map(|e| Placed {
            digest: e.digest,
            otype: e.otype,
            oflags: e.oflags | oflags::EXTERNAL,
            offset: 0,
            len: 0,
            logical: e.logical_len,
            // There are no stored bytes, so there is no stored form to name.
            codec: crate::codec::id::RAW,
        })
        .collect();
    let n_entries = placed_structs.len() + placed_blobs.len() + placed_external.len();
    let compressed = placed_structs
        .iter()
        .chain(placed_blobs.iter())
        .filter(|p| p.codec != crate::codec::id::RAW)
        .count();
    let index_len = index_bytes(n_entries);
    off = index_hdr_off + SEG_HEADER_SIZE + index_len;

    let back_sb_hdr_off = round_up(off, 8);
    let back_sb_payload_off = back_sb_hdr_off + SEG_HEADER_SIZE;
    off = back_sb_hdr_off + sb_reserve;

    let file_size = off + TRAILER_SIZE;

    Layout {
        front_sb_hdr_off,
        front_sb_payload_off,
        obj_hdr_off,
        obj_payload_off,
        obj_payload_len,
        pad_before_blob,
        blob_hdr_off,
        blob_payload_off,
        blob_payload_len,
        pad_before_index,
        index_hdr_off,
        back_sb_hdr_off,
        back_sb_payload_off,
        file_size,
        structs: placed_structs,
        blobs: placed_blobs,
        external: placed_external,
        n_entries,
        compressed,
    }
}

fn write_segment(out: &mut [u8], hdr_off: usize, kind: u16, seq: u64, payload: &[u8]) {
    let h = hdr_off;
    out[h..h + 4].copy_from_slice(&SEG_MAGIC);
    out[h + 4..h + 6].copy_from_slice(&kind.to_le_bytes());
    out[h + 6..h + 8].copy_from_slice(&1u16.to_le_bytes()); // PADDED
    out[h + 8..h + 16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    out[h + 16..h + 24].copy_from_slice(&seq.to_le_bytes());
    out[h + 24..h + 28].copy_from_slice(&crc32c(payload).to_le_bytes());
    let hc = crc32c(&out[h..h + 28]);
    out[h + 28..h + 32].copy_from_slice(&hc.to_le_bytes());
    let p = h + SEG_HEADER_SIZE;
    out[p..p + payload.len()].copy_from_slice(payload);
}

/// Chooses a bucket width for `n` entries (§02.6.1: 0, 8 or 16 bits).
///
/// 16 bits, or none at all below ~1000 entries where a flat 256 KiB table
/// would dwarf the index it accelerates.
///
/// Wider tables are permitted by the format and readable by this
/// implementation, but they are not written, because they measured *worse*.
/// At 10^6 objects a 20-bit table gives roughly one entry per bucket and
/// should in theory be two memory accesses; measured, it was 695 ns p99
/// against 593 ns for 16 bits. The 4 MiB table stops fitting in L2, so the
/// bucket read becomes a miss of its own, and the win from scanning a
/// 16-bit bucket's handful of adjacent entries — which the prefetcher
/// handles — is lost. `omni bench` is what settled it; theory said
/// otherwise.
/// Total size of the index segment payload for `n` entries: header, entry
/// array and bucket table. The layout pass and the writer must agree on this,
/// so it lives in one place.
pub fn index_bytes(n: usize) -> usize {
    let bits = bucket_bits_for(n);
    let buckets = if bits == 0 { 0 } else { 1usize << bits };
    IDX_HEADER_SIZE + n * IDX_ENTRY_SIZE + buckets * 4
}

pub fn bucket_bits_for(n: usize) -> u32 {
    if n < 1024 {
        0
    } else {
        16
    }
}

/// The leading `bits` bits of a digest, which is what the entries are sorted
/// by. Entries sort byte-lexicographically, so a big-endian prefix is
/// monotonic in the same order.
fn bucket_of(d: &Digest, bits: u32) -> usize {
    if bits == 0 {
        return 0;
    }
    debug_assert!(bits <= 24);
    let top = ((d[0] as usize) << 16) | ((d[1] as usize) << 8) | d[2] as usize;
    top >> (24 - bits)
}

fn build_index(l: &Layout, algo: HashAlgo) -> Vec<u8> {
    let mut entries: Vec<&Placed> = l
        .structs
        .iter()
        .chain(l.blobs.iter())
        .chain(l.external.iter())
        .collect();
    entries.sort_by_key(|a| a.digest);

    let bits = bucket_bits_for(entries.len());
    let n_buckets = if bits == 0 { 0 } else { 1usize << bits };
    let bucket_off = IDX_HEADER_SIZE + entries.len() * IDX_ENTRY_SIZE;

    let mut out = vec![0u8; bucket_off + n_buckets * 4];
    out[0..4].copy_from_slice(&IDX_MAGIC);
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    out[6..8].copy_from_slice(&(IDX_ENTRY_SIZE as u16).to_le_bytes());
    out[8..16].copy_from_slice(&(entries.len() as u64).to_le_bytes());
    out[16..24].copy_from_slice(&(if bits == 0 { 0 } else { bucket_off as u64 }).to_le_bytes());
    out[24..28].copy_from_slice(&bits.to_le_bytes());
    out[28] = algo.code();
    out[29] = 32;
    out[30..32].copy_from_slice(&0b11u16.to_le_bytes()); // SORTED | COMPLETE
    out[32..40].copy_from_slice(&0u64.to_le_bytes());
    out[40..48].copy_from_slice(&0u64.to_le_bytes());
    let ic = crc32c(&out[0..60]);
    out[60..64].copy_from_slice(&ic.to_le_bytes());

    for (i, e) in entries.iter().enumerate() {
        let b = IDX_HEADER_SIZE + i * IDX_ENTRY_SIZE;
        out[b..b + 32].copy_from_slice(&e.digest);
        out[b + 32..b + 40].copy_from_slice(&e.offset.to_le_bytes());
        out[b + 40..b + 48].copy_from_slice(&e.len.to_le_bytes()); // stored_len
        out[b + 48..b + 56].copy_from_slice(&e.logical.to_le_bytes());
        out[b + 56..b + 58].copy_from_slice(&e.otype.to_le_bytes());
        out[b + 58] = e.codec;
        out[b + 59] = e.oflags;
        out[b + 60..b + 64].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // no aux
    }

    // bucket[i] is the first entry whose digest prefix is >= i, so the range
    // for prefix i is [bucket[i], bucket[i+1]) with entry_count as the final
    // bound. Filling backwards makes empty buckets collapse to zero-width
    // ranges without a second pass.
    if bits > 0 {
        let mut starts = vec![entries.len() as u32; n_buckets];
        for (i, e) in entries.iter().enumerate().rev() {
            starts[bucket_of(&e.digest, bits)] = i as u32;
        }
        let mut running = entries.len() as u32;
        for s in starts.iter_mut().rev() {
            if *s > running {
                *s = running;
            } else {
                running = *s;
            }
        }
        for (i, s) in starts.iter().enumerate() {
            let o = bucket_off + i * 4;
            out[o..o + 4].copy_from_slice(&s.to_le_bytes());
        }
    }
    out
}

fn superblock_value(l: &Layout, root: &Digest, align: usize, opts: &PackOptions) -> Value {
    let mut segments = vec![
        Value::Array(vec![
            Value::U(l.front_sb_hdr_off as u64),
            Value::U(seg::SUPER as u64),
        ]),
        Value::Array(vec![
            Value::U(l.obj_hdr_off as u64),
            Value::U(seg::OBJ as u64),
        ]),
    ];
    if let Some(p) = l.pad_before_blob {
        segments.push(Value::Array(vec![
            Value::U(p.0 as u64),
            Value::U(seg::PAD as u64),
        ]));
    }
    segments.push(Value::Array(vec![
        Value::U(l.blob_hdr_off as u64),
        Value::U(seg::BLOB as u64),
    ]));
    if let Some(p) = l.pad_before_index {
        segments.push(Value::Array(vec![
            Value::U(p.0 as u64),
            Value::U(seg::PAD as u64),
        ]));
    }
    segments.push(Value::Array(vec![
        Value::U(l.index_hdr_off as u64),
        Value::U(seg::INDEX as u64),
    ]));
    segments.push(Value::Array(vec![
        Value::U(l.back_sb_hdr_off as u64),
        Value::U(seg::SUPER as u64),
    ]));

    // §02.5's stats separate the two, and they differ exactly when a codec did
    // something: `logical` is what the digests cover, `stored` is what the file
    // costs.
    // Absent objects count towards `bytes_logical` and not towards
    // `bytes_stored`: an index-only container describes 700 GB and costs a few
    // megabytes, and the two numbers are what say so.
    let total_logical: u64 = l
        .structs
        .iter()
        .chain(l.blobs.iter())
        .chain(l.external.iter())
        .map(|p| p.logical)
        .sum();
    let total_stored: u64 = l.structs.iter().chain(l.blobs.iter()).map(|p| p.len).sum();

    Value::map(vec![
        ("t", Value::text("omni.core/superblock")),
        ("v", Value::U(1)),
        (
            "roots",
            Value::Array(vec![Value::Array(vec![
                Value::U(otype::MANIFEST as u64),
                Value::Bytes(root.to_vec()),
            ])]),
        ),
        (
            "index",
            Value::map(vec![
                ("off", Value::U((l.index_hdr_off + SEG_HEADER_SIZE) as u64)),
                ("len", Value::U(index_bytes(l.n_entries) as u64)),
                ("entries", Value::U(l.n_entries as u64)),
                ("fmt", Value::U(1)),
            ]),
        ),
        ("segments", Value::Array(segments)),
        // §02.5: the container's digest algorithm. The header carries it too, and
        // the two must agree — a superblock that names a different algorithm
        // than the file uses is a contradiction a reader cannot resolve.
        ("hash", Value::text(opts.hash.name())),
        (
            // §02.5: the codecs a reader may meet in this container. `raw` is
            // always among them — structure objects are never compressed — and
            // the requested codec joins it only when something actually used it.
            "codecs",
            Value::Array(if l.compressed > 0 {
                vec![
                    Value::map(vec![("id", Value::text("raw"))]),
                    opts.codec.to_value(),
                ]
            } else {
                vec![Value::map(vec![("id", Value::text("raw"))])]
            }),
        ),
        ("align", Value::U(align as u64)),
        ("profile", Value::text("core")),
        (
            "features",
            Value::map(vec![
                ("required", Value::Array(vec![Value::text("omni.core/1.0")])),
                ("optional", Value::Array(vec![])),
            ]),
        ),
        ("stats", {
            let mut stats = vec![
                ("objects", Value::U(l.n_entries as u64)),
                ("blobs", Value::U(l.blobs.len() as u64)),
                ("bytes_logical", Value::U(total_logical)),
                ("bytes_stored", Value::U(total_stored)),
            ];
            // Only when there are any: a complete container's superblock must
            // not change bytes because this field exists.
            if !l.external.is_empty() {
                stats.push(("external", Value::U(l.external.len() as u64)));
            }
            Value::map(stats)
        }),
        ("creator", Value::text(opts.creator.clone())),
    ])
}

/// Derives a deterministic UUIDv7-shaped identifier from the root digest
/// (§01.10 clause 4). Version and variant bits are set so the value is a
/// well-formed UUID even though it is not time-based.
fn derive_uuid(algo: HashAlgo, root: &Digest) -> [u8; 16] {
    // §03.5.3 domain separation. BLAKE3 has a derive-key mode built for
    // exactly this; SHA-256 has to prefix the context string instead.
    let d = match algo {
        HashAlgo::Blake3_256 => crate::blake3::derive_key(UUID_CONTEXT, root),
        HashAlgo::Sha256 => {
            let mut h = crate::sha256::Sha256::new();
            h.update(UUID_CONTEXT.as_bytes());
            h.update(root);
            h.finalize()
        }
    };
    let mut u = [0u8; 16];
    u.copy_from_slice(&d[..16]);
    u[6] = (u[6] & 0x0f) | 0x70; // version 7
    u[8] = (u[8] & 0x3f) | 0x80; // RFC 9562 variant
    u
}

// -------------------------------------------------------------------- read --

#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub digest: Digest,
    pub offset: u64,
    pub stored_len: u64,
    pub logical_len: u64,
    pub otype: u16,
    pub codec: u8,
    pub oflags: u8,
}

#[derive(Debug, Clone)]
pub struct Header {
    pub container_major: u16,
    pub container_minor: u16,
    pub log2_align: u8,
    pub header_size: u16,
    pub uuid: [u8; 16],
    /// The container's primary digest algorithm, already validated.
    pub hash: HashAlgo,
    pub profile: u8,
    pub flags: u32,
    pub front_sb_off: u64,
    pub front_sb_len: u64,
    pub file_size: u64,
    pub root_digest: Digest,
    pub creator: String,
}

/// A read-only view over a container's bytes.
///
/// In a production implementation `bytes` is an `mmap`; the parsing code below
/// is identical either way, which is the point of §02.9.
pub struct Container {
    pub bytes: Vec<u8>,
    pub header: Header,
    pub superblock: Value,
    pub index: Vec<IndexEntry>,
    /// Bucket table from the index (§02.6.1), if one is present. Turns a
    /// ~20-probe binary search over tens of megabytes into one bucket read
    /// plus a search of a dozen adjacent entries.
    pub buckets: Vec<u32>,
    pub bucket_bits: u32,
}

impl Container {
    /// Opens a sealed container the way a seek-capable reader should: trailer
    /// first, then one jump to the superblock, then the index (§02.7).
    pub fn open(bytes: Vec<u8>) -> Res<Container> {
        if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
            return Err(rule("R-C01", "file too small to be a container"));
        }
        let header = parse_header(&bytes)?;

        // Trailer.
        let t = bytes.len() - TRAILER_SIZE;
        if bytes[t + 56..t + 64] != MAGIC_END {
            return Err(rule("R-C09", "trailer magic mismatch"));
        }
        let tcrc = u32::from_le_bytes(bytes[t + 52..t + 56].try_into().unwrap());
        if crc32c(&bytes[t..t + 52]) != tcrc {
            return Err(rule("R-C09", "trailer CRC mismatch"));
        }
        let sb_off = u64::from_le_bytes(bytes[t..t + 8].try_into().unwrap()) as usize;
        let sb_len = u64::from_le_bytes(bytes[t + 8..t + 16].try_into().unwrap()) as usize;
        let sb_digest: Digest = bytes[t + 16..t + 48].try_into().unwrap();

        if sb_off.checked_add(sb_len).is_none_or(|e| e > bytes.len()) {
            return Err(rule("R-C12", "superblock extent out of range"));
        }
        let sb_bytes = &bytes[sb_off..sb_off + sb_len];
        if header.hash.digest(sb_bytes) != sb_digest {
            return Err(rule("R-C09", "superblock digest mismatch"));
        }
        let superblock = cbor::decode(sb_bytes)?;

        // R-C10: front and back superblocks must be byte-identical.
        if header.flags & hflags::FRONT_SB != 0 {
            let fo = header.front_sb_off as usize;
            let fl = header.front_sb_len as usize;
            if fo.checked_add(fl).is_none_or(|e| e > bytes.len()) {
                return Err(rule("R-C12", "front superblock extent out of range"));
            }
            if &bytes[fo..fo + fl] != sb_bytes {
                return Err(rule("R-C10", "front and back superblocks differ"));
            }
        }

        let idx = superblock
            .get("index")
            .ok_or_else(|| rule("R-E05", "superblock has no index"))?;
        let ioff = idx.get("off").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let ilen = idx.get("len").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let index = parse_index(&bytes, ioff, ilen)?;
        let (buckets, bucket_bits) = parse_buckets(&bytes, ioff, ilen, index.len())?;

        Ok(Container {
            bytes,
            header,
            superblock,
            index,
            buckets,
            bucket_bits,
        })
    }

    /// Locates an object — the hot path (§02.6.2).
    ///
    /// With a bucket table this is one bucket read followed by a search of the
    /// few entries sharing that digest prefix; without one it is a plain
    /// binary search over the whole index. The difference at a million objects
    /// is roughly an order of magnitude, because a 20-probe search over 61 MiB
    /// is 20 cache misses and nothing else.
    pub fn find(&self, d: &Digest) -> Option<&IndexEntry> {
        let mut probes = 0;
        self.locate::<false>(d, &mut probes).map(|i| &self.index[i])
    }

    /// How many index entries a lookup for `d` compares — the same walk as
    /// [`Container::find`], counted.
    ///
    /// This exists because a wall-clock p99 on a shared machine is not a
    /// measurement of the index. `docs/design/performance.md` §11 records a
    /// p99 that moves by 30 % between runs of the *same binary*, which is
    /// enough noise to hide any change worth making. Entries compared is the
    /// part of the cost this code decides, and it is the same on every machine.
    ///
    /// It is a separate entry point rather than an out-parameter on `find`
    /// because a counter in the hot path is a store per probe; `COUNT` is a
    /// const generic so the counting compiles away where it is not wanted.
    pub fn probe_cost(&self, d: &Digest) -> usize {
        let mut probes = 0;
        self.locate::<true>(d, &mut probes);
        probes
    }

    #[inline(always)]
    fn locate<const COUNT: bool>(&self, d: &Digest, probes: &mut usize) -> Option<usize> {
        macro_rules! probe {
            ($i:expr) => {{
                if COUNT {
                    *probes += 1;
                }
                &self.index[$i]
            }};
        }
        let (lo, hi) = if self.bucket_bits > 0 && !self.buckets.is_empty() {
            let b = bucket_of(d, self.bucket_bits);
            let lo = self.buckets[b] as usize;
            let hi = self
                .buckets
                .get(b + 1)
                .map(|x| *x as usize)
                .unwrap_or(self.index.len());
            (lo.min(self.index.len()), hi.min(self.index.len()))
        } else {
            (0, self.index.len())
        };
        if lo >= hi {
            return None;
        }
        // An `IndexEntry` is one cache line and the digest is at the front of
        // it, so the line that confirms a candidate is also the line that
        // answers the query. What is left to minimize is how many of those lines
        // get touched, and the way to touch one is to guess which.
        //
        // Digests are cryptographic hashes, so they are uniform — this is the
        // one place where "the keys are random" is a guarantee rather than a
        // hope. The bucket table has already spent the leading `bucket_bits`;
        // the bits immediately below them say where in that bucket the entry
        // falls, to within an entry or two. So: one guess, then a step in the
        // direction the comparison points.
        //
        // What this buys, measured: 2.20 entries compared per lookup against
        // 8.62 for scanning the bucket, at 10^6 objects — 141 bytes of index
        // touched instead of 552. What it does *not* buy, also measured: any
        // single-thread latency on the machine in `performance.md` §11, where
        // the scan's eight consecutive lines cost the same as these two or three
        // scattered ones because the prefetcher fetches them in parallel. The
        // claim here is bandwidth per lookup, which is arithmetic, and not a p99,
        // which would be a fiction on that hardware.
        //
        // The scan below is the fallback: it is what runs for a container with
        // no bucket table, where the range is the whole index and there is
        // nothing to interpolate within.
        const LINEAR_SCAN_LIMIT: usize = 64;
        let span = hi - lo;
        if self.bucket_bits > 0 && span > 1 {
            let p = u64::from_be_bytes(d[..8].try_into().unwrap());
            // What is left of the digest once the bucket has consumed its
            // prefix, as a 32-bit fraction of the bucket's own range.
            let frac = (p << self.bucket_bits) >> 32;
            let guess = lo + (((frac * span as u64) >> 32) as usize).min(span - 1);
            // Sorted, so the first comparison decides the direction and the walk
            // is monotone from there: it stops at the answer, or at the first
            // entry past where the answer would have been.
            return match probe!(guess).digest.cmp(d) {
                std::cmp::Ordering::Equal => Some(guess),
                std::cmp::Ordering::Less => {
                    for i in guess + 1..hi {
                        match probe!(i).digest.cmp(d) {
                            std::cmp::Ordering::Equal => return Some(i),
                            std::cmp::Ordering::Less => continue,
                            std::cmp::Ordering::Greater => return None,
                        }
                    }
                    None
                }
                std::cmp::Ordering::Greater => {
                    for i in (lo..guess).rev() {
                        match probe!(i).digest.cmp(d) {
                            std::cmp::Ordering::Equal => return Some(i),
                            std::cmp::Ordering::Greater => continue,
                            std::cmp::Ordering::Less => return None,
                        }
                    }
                    None
                }
            };
        }

        if span <= LINEAR_SCAN_LIMIT {
            return (lo..hi).find(|i| probe!(*i).digest == *d);
        }
        let mut a = lo;
        let mut b = hi;
        while a < b {
            let mid = a + (b - a) / 2;
            match probe!(mid).digest.cmp(d) {
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Less => a = mid + 1,
                std::cmp::Ordering::Greater => b = mid,
            }
        }
        None
    }

    /// Returns an object's bytes, verifying its digest (verification level L1).
    /// An object's logical bytes, borrowed. Only possible for an uncompressed
    /// copy; a compressed one has to be materialized, so use [`Container::read`]
    /// when the codec is not known to be `raw`.
    pub fn get(&self, d: &Digest) -> Res<&[u8]> {
        let e = self.entry(d)?;
        if e.codec != crate::codec::id::RAW {
            return Err(Error::Codec(format!(
                "{} is stored with codec {} and cannot be borrowed; use `read`",
                hex(d),
                crate::codec::Codec::from_id(e.codec).name()
            )));
        }
        let payload = self.stored_slice(e)?;
        if self.header.hash.digest(payload) != *d {
            return Err(rule("R-O01", format!("digest mismatch for {}", hex(d))));
        }
        Ok(payload)
    }

    /// An object's logical bytes, decompressing if this copy is compressed.
    ///
    /// The digest is checked over the *logical* bytes, which is what §03.5.2
    /// says it covers — so a corrupted compressed stream that happens to inflate
    /// to something is still caught.
    pub fn read(&self, d: &Digest) -> Res<Vec<u8>> {
        let e = self.entry(d)?;
        let stored = self.stored_slice(e)?;
        let codec = self.codec_for(e.codec);
        let logical = match codec {
            crate::codec::Codec::Raw => stored.to_vec(),
            other => other.decode(stored, e.logical_len, self.high_ratio())?,
        };
        if self.header.hash.digest(&logical) != *d {
            return Err(rule("R-O01", format!("digest mismatch for {}", hex(d))));
        }
        Ok(logical)
    }

    /// The codec for an index entry's codec byte, as described by the
    /// superblock.
    ///
    /// The index has room for an id, not for parameters, and some codecs need
    /// them: `bitshuffle+zstd` transposing by 2 bytes and by 4 produces
    /// different bytes from the same input. §03.7.1 is why this works — codec
    /// descriptors are required to be explicit and complete, and the superblock
    /// carries them. A container whose superblock omits the descriptor falls
    /// back to the registry default, which is the best a reader can do and is
    /// caught immediately by the digest check either way.
    pub fn codec_for(&self, id: u8) -> crate::codec::Codec {
        if id == crate::codec::id::RAW {
            return crate::codec::Codec::Raw;
        }
        if let Some(Value::Array(list)) = self.superblock.get("codecs") {
            for c in list {
                let declared = crate::codec::Codec::from_value(c);
                if declared.id() == id {
                    return declared;
                }
            }
        }
        crate::codec::Codec::from_id(id)
    }

    fn entry(&self, d: &Digest) -> Res<&IndexEntry> {
        let e = self.find(d).ok_or_else(|| Error::NotFound(hex(d)))?;
        if e.oflags & oflags::EXTERNAL != 0 {
            return Err(Error::NotFound(format!("{} (external)", hex(d))));
        }
        Ok(e)
    }

    fn stored_slice(&self, e: &IndexEntry) -> Res<&[u8]> {
        let s = e.offset as usize;
        let n = e.stored_len as usize;
        if s.checked_add(n).is_none_or(|x| x > self.bytes.len()) {
            return Err(rule("R-C12", "object extent out of range"));
        }
        Ok(&self.bytes[s..s + n])
    }

    /// Whether the container declares the high-ratio codec feature (§03.7.4).
    pub fn high_ratio(&self) -> bool {
        self.superblock
            .get("features")
            .and_then(|f| f.get("optional"))
            .and_then(|o| o.as_array())
            .is_some_and(|a| {
                a.iter()
                    .any(|x| x.as_str() == Some(crate::codec::HIGH_RATIO_FEATURE))
            })
    }

    pub fn get_value(&self, d: &Digest) -> Res<Value> {
        Ok(cbor::decode(&self.read(d)?)?)
    }

    pub fn root(&self) -> Res<Value> {
        self.get_value(&self.header.root_digest)
    }

    /// Walks every segment header, checking framing and CRCs (validation V0).
    pub fn segments(&self) -> Res<Vec<(usize, u16, u64)>> {
        let mut out = Vec::new();
        let mut off = self.header.header_size as usize;
        let end = self.bytes.len() - TRAILER_SIZE;
        while off + SEG_HEADER_SIZE <= end {
            if self.bytes[off..off + 4] != SEG_MAGIC {
                // Segments are padded; skip forward to the next 8-byte boundary
                // that carries a segment magic.
                off += 8;
                continue;
            }
            let hc = u32::from_le_bytes(self.bytes[off + 28..off + 32].try_into().unwrap());
            if crc32c(&self.bytes[off..off + 28]) != hc {
                return Err(rule("R-C05", format!("segment header CRC at {off:#x}")));
            }
            let kind = u16::from_le_bytes(self.bytes[off + 4..off + 6].try_into().unwrap());
            let plen = u64::from_le_bytes(self.bytes[off + 8..off + 16].try_into().unwrap());
            let p = off + SEG_HEADER_SIZE;
            if p as u64 + plen > end as u64 {
                return Err(rule(
                    "R-C05",
                    format!("segment payload overruns file at {off:#x}"),
                ));
            }
            let pc = u32::from_le_bytes(self.bytes[off + 24..off + 28].try_into().unwrap());
            if crc32c(&self.bytes[p..p + plen as usize]) != pc {
                return Err(rule("R-C05", format!("segment payload CRC at {off:#x}")));
            }
            out.push((off, kind, plen));
            off = p + plen as usize;
            off = round_up(off, 8);
        }
        Ok(out)
    }
}

/// Parses and CRC-checks the 128-byte file header.
///
/// Public because recovery (§02.8) needs it: the header is the only part of a
/// damaged container that must still be trusted, and it is the only part that
/// carries the digest algorithm, the alignment and the root digest.
/// Locates every segment by scanning for `OSEG` on 8-byte boundaries,
/// validating both CRCs (§02.8).
///
/// Deliberately does not consult the superblock, the trailer or the index:
/// this is the path a damaged container takes, and it must not depend on the
/// structures most likely to be damaged. A segment whose header CRC fails is
/// skipped rather than fatal — the goal is to salvage what is intact.
pub fn scan_segments(bytes: &[u8]) -> Res<Vec<(usize, u16, u64)>> {
    let header = parse_header(bytes)?;
    let mut out = Vec::new();
    let mut off = header.header_size as usize;
    // The trailer may itself be destroyed, so scan to the end of the file and
    // let the per-segment CRCs decide what is real.
    let end = bytes.len();
    while off + SEG_HEADER_SIZE <= end {
        if bytes[off..off + 4] != SEG_MAGIC {
            off += 8;
            continue;
        }
        let hc = u32::from_le_bytes(bytes[off + 28..off + 32].try_into().unwrap());
        if crc32c(&bytes[off..off + 28]) != hc {
            off += 8;
            continue;
        }
        let kind = u16::from_le_bytes(bytes[off + 4..off + 6].try_into().unwrap());
        let plen = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
        let p = off + SEG_HEADER_SIZE;
        if plen > (end - p) as u64 {
            off += 8;
            continue;
        }
        out.push((off, kind, plen));
        off = round_up(p + plen as usize, 8);
    }
    Ok(out)
}

/// [`parse_header`] against a buffer that holds only the header, with the file's
/// real length supplied separately — what a reader that has not read the file
/// has.
pub fn parse_header_bytes(head: &[u8], file_size: u64) -> Res<Header> {
    if head.len() < HEADER_SIZE {
        return Err(rule("R-C01", "header is shorter than 128 bytes"));
    }
    let mut h = parse_header(head)?;
    if h.file_size != file_size {
        return Err(rule(
            "R-C01",
            format!(
                "header declares {} bytes, the file has {file_size}",
                h.file_size
            ),
        ));
    }
    h.file_size = file_size;
    Ok(h)
}

/// [`parse_index`] against a buffer holding just the index and its header.
pub fn parse_index_bytes(b: &[u8], off: usize, len: usize) -> Res<Vec<IndexEntry>> {
    parse_index(b, off, len)
}

pub fn parse_header(b: &[u8]) -> Res<Header> {
    if b.len() < HEADER_SIZE {
        return Err(rule("R-C01", "file too small to hold a header"));
    }
    if b[0..8] != MAGIC {
        return Err(rule("R-C01", "bad magic"));
    }
    let hcrc = u32::from_le_bytes(b[124..128].try_into().unwrap());
    if crc32c(&b[0..124]) != hcrc {
        return Err(rule("R-C02", "header CRC mismatch"));
    }
    let header_size = u16::from_le_bytes(b[14..16].try_into().unwrap());
    if !(128..=4096).contains(&header_size) {
        return Err(rule("R-C03", "header_size out of range"));
    }
    if b[12] != 0x01 {
        return Err(rule("R-C01", "unsupported byte order"));
    }
    let log2_align = b[13];
    if !(6..=30).contains(&log2_align) {
        return Err(rule("R-C04", "log2_align out of range"));
    }
    // An unknown algorithm is fatal, not something to work around: every
    // digest in the file, including the root, would be uninterpretable.
    let hash = HashAlgo::from_code(b[32]).ok_or_else(|| {
        rule(
            "R-C05",
            format!("unsupported hash algorithm 0x{:02x}", b[32]),
        )
    })?;
    if b[34] as usize != std::mem::size_of::<Digest>() {
        return Err(rule("R-C05", "digest_len does not match the algorithm"));
    }
    let creator_raw = &b[96..112];
    let creator = String::from_utf8_lossy(creator_raw)
        .trim_end_matches('\0')
        .to_string();
    Ok(Header {
        container_major: u16::from_le_bytes(b[8..10].try_into().unwrap()),
        container_minor: u16::from_le_bytes(b[10..12].try_into().unwrap()),
        log2_align,
        header_size,
        uuid: b[16..32].try_into().unwrap(),
        hash,
        profile: b[33],
        flags: u32::from_le_bytes(b[36..40].try_into().unwrap()),
        front_sb_off: u64::from_le_bytes(b[40..48].try_into().unwrap()),
        front_sb_len: u64::from_le_bytes(b[48..56].try_into().unwrap()),
        file_size: u64::from_le_bytes(b[56..64].try_into().unwrap()),
        root_digest: b[64..96].try_into().unwrap(),
        creator,
    })
}

/// Reads the bucket table, if the index declares one.
///
/// The table is an accelerator, never authority: a bucket that points outside
/// the entry array, or is not monotonic, would let a lookup miss an object that
/// is present. Rather than trust it, an inconsistent table is dropped and the
/// reader falls back to a full binary search — slower, always correct.
fn parse_buckets(b: &[u8], off: usize, len: usize, entries: usize) -> Res<(Vec<u32>, u32)> {
    if len < IDX_HEADER_SIZE {
        return Ok((Vec::new(), 0));
    }
    let h = &b[off..off + IDX_HEADER_SIZE];
    let bucket_off = u64::from_le_bytes(h[16..24].try_into().unwrap()) as usize;
    let bits = u32::from_le_bytes(h[24..28].try_into().unwrap());
    if bucket_off == 0 || bits == 0 {
        return Ok((Vec::new(), 0));
    }
    if !matches!(bits, 8 | 16 | 20 | 24) {
        return Err(rule(
            "R-C11",
            format!("bucket_bits {bits} is not 0, 8, 16, 20 or 24"),
        ));
    }
    let n = 1usize << bits;
    let start = off + bucket_off;
    let end = start + n * 4;
    if bucket_off < IDX_HEADER_SIZE + entries * IDX_ENTRY_SIZE || end > off + len || end > b.len() {
        return Err(rule("R-C12", "bucket table extent out of range"));
    }
    let mut out = Vec::with_capacity(n);
    let mut prev = 0u32;
    for i in 0..n {
        let v = u32::from_le_bytes(b[start + i * 4..start + i * 4 + 4].try_into().unwrap());
        if v < prev || v as usize > entries {
            // Not monotonic or out of bounds: unusable as an accelerator.
            return Ok((Vec::new(), 0));
        }
        prev = v;
        out.push(v);
    }
    Ok((out, bits))
}

fn parse_index(b: &[u8], off: usize, len: usize) -> Res<Vec<IndexEntry>> {
    if off.checked_add(len).is_none_or(|e| e > b.len()) || len < IDX_HEADER_SIZE {
        return Err(rule("R-C12", "index extent out of range"));
    }
    let h = &b[off..off + IDX_HEADER_SIZE];
    if h[0..4] != IDX_MAGIC {
        return Err(rule("R-C11", "bad index magic"));
    }
    let icrc = u32::from_le_bytes(h[60..64].try_into().unwrap());
    if crc32c(&h[0..60]) != icrc {
        return Err(rule("R-C11", "index header CRC mismatch"));
    }
    let entry_size = u16::from_le_bytes(h[6..8].try_into().unwrap()) as usize;
    if entry_size != IDX_ENTRY_SIZE {
        return Err(rule("R-C11", "unsupported index entry size"));
    }
    let n = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
    if IDX_HEADER_SIZE + n * entry_size > len {
        return Err(rule("R-C11", "index entry count exceeds segment"));
    }

    let mut out = Vec::with_capacity(n);
    let mut prev: Option<Digest> = None;
    for i in 0..n {
        let p = off + IDX_HEADER_SIZE + i * entry_size;
        let digest: Digest = b[p..p + 32].try_into().unwrap();
        if let Some(pv) = prev {
            if digest <= pv {
                return Err(rule("R-C11", "index not strictly sorted"));
            }
        }
        prev = Some(digest);
        let e = IndexEntry {
            digest,
            offset: u64::from_le_bytes(b[p + 32..p + 40].try_into().unwrap()),
            stored_len: u64::from_le_bytes(b[p + 40..p + 48].try_into().unwrap()),
            logical_len: u64::from_le_bytes(b[p + 48..p + 56].try_into().unwrap()),
            otype: u16::from_le_bytes(b[p + 56..p + 58].try_into().unwrap()),
            codec: b[p + 58],
            oflags: b[p + 59],
        };
        // R-C13: decompression-ratio bound.
        if e.stored_len > 0 && e.logical_len / e.stored_len.max(1) > 1000 {
            return Err(rule("R-C13", "declared expansion ratio exceeds 1000:1"));
        }
        out.push(e);
    }
    Ok(out)
}

/// Structural + index + integrity validation (levels V0–V4 of §15.1).
pub struct Report {
    pub segments: Vec<(usize, u16, u64)>,
    pub objects_verified: usize,
    pub bytes_verified: u64,
    pub padding_ok: bool,
    pub alignment_ok: bool,
    pub reachable: usize,
    pub dangling: Vec<Digest>,
    /// R-O02 violations: the index says one object type, the object's own `t`
    /// says another. `(digest, index otype, the `t` it carries)`.
    pub mistyped: Vec<(Digest, u16, String)>,
}

pub fn verify(c: &Container) -> Res<Report> {
    let segments = c.segments()?;

    // R-C07 / R-C08: zero padding and data-object alignment.
    let align = 1usize << c.header.log2_align;
    let mut alignment_ok = true;
    for e in &c.index {
        if e.otype == otype::BLOB && e.offset % align as u64 != 0 {
            alignment_ok = false;
        }
    }

    // R-C07: every byte that is not part of the header, the trailer, a
    // segment header, a superblock, the index or an object must be zero.
    //
    // Note what this deliberately does *not* treat as covered: a PAD segment's
    // payload, and the gaps between objects inside OBJ and BLOB segments.
    // Those are padding by definition, and exempting them would leave the
    // format with several megabytes of unexamined space per container — a
    // place to hide data that no reader parses and no digest covers.
    let mut covered = vec![false; c.bytes.len()];
    covered[..HEADER_SIZE].fill(true);
    for (off, kind, plen) in &segments {
        covered[*off..*off + SEG_HEADER_SIZE].fill(true);
        // Superblock and index payloads are content; OBJ and BLOB payloads are
        // covered object by object below; PAD payloads are never covered.
        if matches!(*kind, seg::SUPER | seg::INDEX | seg::SIG) {
            let p = *off + SEG_HEADER_SIZE;
            covered[p..p + *plen as usize].fill(true);
        }
    }
    for e in &c.index {
        let s = e.offset as usize;
        let n = e.stored_len as usize;
        if s.checked_add(n).is_some_and(|x| x <= covered.len()) {
            covered[s..s + n].fill(true);
        }
    }
    covered[c.bytes.len() - TRAILER_SIZE..].fill(true);
    let padding_ok = covered
        .iter()
        .enumerate()
        .all(|(i, &c2)| c2 || c.bytes[i] == 0);

    // R-O01 over every present object.
    let mut objects_verified = 0usize;
    let mut bytes_verified = 0u64;
    for e in &c.index {
        if e.oflags & oflags::EXTERNAL != 0 {
            continue;
        }
        c.read(&e.digest)?;
        objects_verified += 1;
        bytes_verified += e.stored_len;
    }

    // R-O02: the index's `otype` and the object's own `t` must agree. An
    // object that lies about what it is defeats every type-directed decision a
    // reader makes before fetching it — which, in a format where refs carry the
    // type, is all of them.
    let mut mistyped = Vec::new();
    for e in &c.index {
        if e.otype == otype::BLOB || e.oflags & oflags::EXTERNAL != 0 {
            continue;
        }
        let Some(want) = otype::schema_uri(e.otype) else {
            continue;
        };
        let v = c.get_value(&e.digest)?;
        match v.get("t").and_then(|x| x.as_str()) {
            Some(got) if got == want => {}
            Some(got) => mistyped.push((e.digest, e.otype, got.to_string())),
            None => mistyped.push((e.digest, e.otype, String::new())),
        }
    }

    // V4: reachability from the root.
    let mut seen: std::collections::BTreeSet<Digest> = Default::default();
    let mut dangling = Vec::new();
    let mut stack = vec![c.header.root_digest];
    while let Some(d) = stack.pop() {
        if !seen.insert(d) {
            continue;
        }
        let Some(entry) = c.find(&d) else {
            dangling.push(d);
            continue;
        };
        if entry.otype == otype::BLOB {
            continue;
        }
        let v = c.get_value(&d)?;
        collect_refs(&v, &mut stack);
    }

    Ok(Report {
        segments,
        objects_verified,
        bytes_verified,
        padding_ok,
        alignment_ok,
        reachable: seen.len(),
        dangling,
        mistyped,
    })
}

/// Collects `[otype, digest]` references from a decoded structure object,
/// keeping the type. Object types live in refs rather than in objects, so this
/// is how a reader learns what an object is before fetching it.
pub fn collect_typed_refs(v: &Value, out: &mut Vec<(u16, Digest)>) {
    match v {
        Value::Array(a) => {
            if a.len() == 2 {
                if let (Some(t), Some(b)) = (a[0].as_u64(), a[1].as_bytes()) {
                    if b.len() == 32 && t <= u16::MAX as u64 {
                        let mut d = [0u8; 32];
                        d.copy_from_slice(b);
                        out.push((t as u16, d));
                        return;
                    }
                }
            }
            for x in a {
                collect_typed_refs(x, out);
            }
        }
        Value::Map(m) => {
            for (k, val) in m {
                collect_typed_refs(k, out);
                collect_typed_refs(val, out);
            }
        }
        Value::Tag(_, inner) => collect_typed_refs(inner, out),
        _ => {}
    }
}

/// Collects reference digests, discarding the types. Callers that only need
/// reachability — the verifier, `fsck` — do not care what an object is.
pub fn collect_refs(v: &Value, out: &mut Vec<Digest>) {
    let mut typed = Vec::new();
    collect_typed_refs(v, &mut typed);
    out.extend(typed.into_iter().map(|(_, d)| d));
}

#[cfg(test)]
mod tests {
    // Compression tests live here rather than in `codec` because what matters
    // is the container-level invariant of §01.2: a compressed copy is the same
    // object.
    /// §02.5: the superblock describes the container, so the algorithm it names
    /// has to be the one the container uses. Naming the other one is a
    /// contradiction a reader cannot resolve — and it was hardcoded here.
    #[test]
    fn the_superblock_names_the_container_s_own_hash() {
        for algo in [HashAlgo::Blake3_256, HashAlgo::Sha256] {
            let (objs, root) = crate::model::ModelBuilder::new("test/hash")
                .hash(algo)
                .tensor(crate::model::TensorSpec {
                    name: "w".into(),
                    shape: vec![4],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight".into(),
                    data: vec![0u8; 16],
                    layout: None,
                })
                .build();
            let c = Container::open(
                pack(
                    &objs,
                    &root,
                    &PackOptions {
                        hash: algo,
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(c.header.hash, algo);
            assert_eq!(
                c.superblock.get("hash").and_then(|v| v.as_str()),
                Some(algo.name()),
                "the superblock must name {}",
                algo.name()
            );
        }
    }

    mod compression {
        use super::super::*;
        use crate::codec::Codec;

        /// A model with a compressible payload: repetitive bytes, as a real
        /// tensor's exponent bytes are.
        fn objects() -> (Vec<Object>, Digest) {
            let data: Vec<u8> = std::iter::repeat_n([0x3f, 0x80, 0x00, 0x00], 4096)
                .flatten()
                .collect();
            crate::model::ModelBuilder::new("test/compressible")
                .chunk_size(1 << 20)
                .tensor(crate::model::TensorSpec {
                    name: "w".into(),
                    shape: vec![64, 64],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight".into(),
                    data,
                    layout: None,
                })
                .build()
        }

        #[test]
        fn a_compressed_container_holds_the_same_objects() {
            let (objs, root) = objects();
            let raw = pack(&objs, &root, &PackOptions::default()).unwrap();
            let deflated = pack(
                &objs,
                &root,
                &PackOptions {
                    codec: Codec::Deflate { level: 9 },
                    ..Default::default()
                },
            )
            .unwrap();
            // Smaller on disk. Not dramatically so at this size: alignment
            // padding and the index dominate a 16 KB model, which is itself a
            // fair illustration of §03.7's guidance about where the size wins
            // in OMNI actually come from.
            assert!(
                deflated.len() < raw.len(),
                "{} vs {}",
                deflated.len(),
                raw.len()
            );
            let a = Container::open(raw).unwrap();
            let b = Container::open(deflated).unwrap();
            // ...same root, same object identities, same bytes read back.
            assert_eq!(a.header.root_digest, b.header.root_digest);
            assert_eq!(a.index.len(), b.index.len());
            for e in &a.index {
                assert_eq!(a.read(&e.digest).unwrap(), b.read(&e.digest).unwrap());
            }
            // The data object really is stored compressed, and says so.
            let blob = b
                .index
                .iter()
                .find(|e| e.otype == otype::BLOB)
                .expect("a data object");
            assert_eq!(blob.codec, crate::codec::id::DEFLATE);
            assert!(
                blob.stored_len * 50 < blob.logical_len,
                "{} vs {}",
                blob.stored_len,
                blob.logical_len
            );
            // And it verifies: the digest covers the logical bytes.
            let r = verify(&b).unwrap();
            assert!(r.dangling.is_empty());
            assert!(r.padding_ok && r.alignment_ok);
        }

        #[test]
        fn borrowing_a_compressed_object_is_refused_rather_than_wrong() {
            let (objs, root) = objects();
            let c = Container::open(
                pack(
                    &objs,
                    &root,
                    &PackOptions {
                        codec: Codec::Deflate { level: 6 },
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
            let blob = c.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
            assert!(matches!(c.get(&blob.digest), Err(Error::Codec(_))));
            assert!(c.read(&blob.digest).is_ok());
            // Structure objects stay raw and can still be borrowed.
            let manifest = c.index.iter().find(|e| e.otype == otype::MANIFEST).unwrap();
            assert!(c.get(&manifest.digest).is_ok());
        }

        #[test]
        fn a_codec_that_would_expand_an_object_is_not_used() {
            // Incompressible data: keeping the compressed form would make the
            // file bigger for nothing.
            let data: Vec<u8> = (0..4096u64)
                .map(|i| (crate::expr::uniform01(0xc0de_beef_0000_0001, i) * 256.0) as u8)
                .collect();
            let o = Object::blob(data)
                .compressed(&Codec::Deflate { level: 9 })
                .unwrap();
            assert!(o.stored.is_none());
            assert_eq!(o.codec_id(), crate::codec::id::RAW);
        }

        #[test]
        fn an_unimplemented_codec_fails_the_pack_rather_than_silently_storing_raw() {
            let (objs, root) = objects();
            let r = pack(
                &objs,
                &root,
                &PackOptions {
                    codec: Codec::Unsupported("lz4"),
                    ..Default::default()
                },
            );
            assert!(matches!(r, Err(Error::Codec(_))));
        }

        /// The same properties under the codec §03.7.1 actually requires, plus
        /// the two the superblock is supposed to report about it.
        #[test]
        fn a_zstd_container_holds_the_same_objects_and_says_what_it_used() {
            let (objs, root) = objects();
            let raw = pack(&objs, &root, &PackOptions::default()).unwrap();
            for codec in [
                Codec::Zstd { level: 9 },
                Codec::BitshuffleZstd {
                    elem_size: 4,
                    level: 9,
                },
            ] {
                let id = codec.id();
                let packed = pack(
                    &objs,
                    &root,
                    &PackOptions {
                        codec,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(packed.len() < raw.len());
                let a = Container::open(raw.clone()).unwrap();
                let b = Container::open(packed).unwrap();
                assert_eq!(a.header.root_digest, b.header.root_digest);
                assert_eq!(a.index.len(), b.index.len());
                for e in &a.index {
                    assert_eq!(a.read(&e.digest).unwrap(), b.read(&e.digest).unwrap());
                }
                let blob = b
                    .index
                    .iter()
                    .find(|e| e.otype == otype::BLOB)
                    .expect("a data object");
                assert_eq!(blob.codec, id);
                assert!(blob.stored_len * 50 < blob.logical_len);
                // §02.5: the superblock names the codecs a reader may meet, and
                // separates what the digests cover from what the file costs.
                let codecs = b.superblock.get("codecs").expect("codecs");
                let names: Vec<&str> = match codecs {
                    Value::Array(a) => a
                        .iter()
                        .filter_map(|c| c.get("id").and_then(|v| v.as_str()))
                        .collect(),
                    _ => panic!("codecs is not an array"),
                };
                assert!(names.contains(&"raw"));
                assert!(names.iter().any(|n| n.contains("zstd")), "{names:?}");
                let stats = b.superblock.get("stats").unwrap();
                let logical = stats.get("bytes_logical").unwrap().as_u64().unwrap();
                let stored = stats.get("bytes_stored").unwrap().as_u64().unwrap();
                assert!(stored < logical, "{stored} vs {logical}");
                let r = verify(&b).unwrap();
                assert!(r.dangling.is_empty() && r.padding_ok && r.alignment_ok);
            }
            // A raw container still reports exactly one codec, and equal totals.
            let a = Container::open(raw).unwrap();
            let stats = a.superblock.get("stats").unwrap();
            assert_eq!(
                stats.get("bytes_logical").unwrap().as_u64(),
                stats.get("bytes_stored").unwrap().as_u64()
            );
            match a.superblock.get("codecs").unwrap() {
                Value::Array(v) => assert_eq!(v.len(), 1),
                _ => panic!("codecs is not an array"),
            }
        }

        #[test]
        fn tampering_with_a_compressed_stream_is_caught() {
            let (objs, root) = objects();
            let mut bytes = pack(
                &objs,
                &root,
                &PackOptions {
                    codec: Codec::Deflate { level: 9 },
                    ..Default::default()
                },
            )
            .unwrap();
            let c = Container::open(bytes.clone()).unwrap();
            let blob = c.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
            let at = blob.offset as usize + 4;
            bytes[at] ^= 0xff;
            let c = Container::open(bytes).unwrap();
            // Either the stream no longer inflates, or it inflates to something
            // whose digest is wrong. Both are errors; neither is silence.
            assert!(c.read(&blob.digest).is_err());
            assert!(verify(&c).is_err());
        }
    }

    use super::*;

    /// Every container-level test runs under both mandatory algorithms
    /// (§03.5.1). Anything that silently assumed 32-byte SHA-256 would pass
    /// under one and fail under the other.
    const ALGOS: [HashAlgo; 2] = [HashAlgo::Blake3_256, HashAlgo::Sha256];

    fn opts(hash: HashAlgo) -> PackOptions {
        PackOptions {
            hash,
            ..Default::default()
        }
    }

    fn tiny_model(algo: HashAlgo) -> (Vec<Object>, Digest) {
        let data: Vec<u8> = (0..8192u32).map(|i| (i % 256) as u8).collect();
        let blob = Object::blob(data);
        let blob_d = blob.digest(algo);

        let chunks = Object::structure(
            otype::CHUNK_LIST,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/chunklist")),
                ("v", Value::U(1)),
                ("total", Value::U(8192)),
                (
                    "chunks",
                    Value::Array(vec![Value::Array(vec![
                        Value::U(0),
                        Value::Bytes(blob_d.to_vec()),
                    ])]),
                ),
            ]),
        );
        let chunks_d = chunks.digest(algo);

        let desc = Object::structure(
            otype::TENSOR_DESC,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/desc")),
                ("v", Value::U(1)),
                ("shape", Value::Array(vec![Value::U(64), Value::U(64)])),
                ("semantic", Value::text("weight")),
                (
                    "value",
                    Value::map(vec![
                        ("op", Value::text("literal")),
                        (
                            "chunks",
                            Value::Array(vec![
                                Value::U(otype::CHUNK_LIST as u64),
                                Value::Bytes(chunks_d.to_vec()),
                            ]),
                        ),
                    ]),
                ),
            ]),
        );
        let desc_d = desc.digest(algo);

        let manifest = Object::structure(
            otype::MANIFEST,
            &Value::map(vec![
                ("t", Value::text("omni.core/manifest")),
                ("v", Value::U(1)),
                ("kind", Value::text("model")),
                (
                    "assets",
                    Value::map(vec![(
                        "w",
                        Value::Array(vec![
                            Value::U(otype::TENSOR_DESC as u64),
                            Value::Bytes(desc_d.to_vec()),
                        ]),
                    )]),
                ),
            ]),
        );
        let root = manifest.digest(algo);
        (vec![blob, chunks, desc, manifest], root)
    }

    #[test]
    fn pack_open_verify() {
        for algo in ALGOS {
            let (objs, root) = tiny_model(algo);
            let bytes = pack(&objs, &root, &opts(algo)).unwrap();
            let c = Container::open(bytes).unwrap();
            assert_eq!(c.header.root_digest, root);
            assert_eq!(c.header.container_major, 1);
            assert_eq!(c.header.hash, algo);
            let r = verify(&c).unwrap();
            assert!(r.padding_ok, "R-C07 zero padding");
            assert!(r.alignment_ok, "R-C08 data alignment");
            assert_eq!(r.objects_verified, 4);
            assert_eq!(r.reachable, 4);
            assert!(r.dangling.is_empty());
        }
    }

    #[test]
    fn packing_is_reproducible() {
        for algo in ALGOS {
            let (objs, root) = tiny_model(algo);
            let a = pack(&objs, &root, &opts(algo)).unwrap();
            let b = pack(&objs, &root, &opts(algo)).unwrap();
            assert_eq!(a, b, "W1: pack must be byte-reproducible");

            // Object order at the input must not matter.
            let mut shuffled = objs.clone();
            shuffled.reverse();
            let c = pack(&shuffled, &root, &opts(algo)).unwrap();
            assert_eq!(a, c, "emission order is (otype, digest), not input order");
        }
    }

    /// The same model under two algorithms is two different object graphs with
    /// two different roots — but the same logical content. Identity is a hash,
    /// so changing the hash changes every identity, which is exactly why the
    /// algorithm is a container-wide header field (§12.11).
    #[test]
    fn the_algorithm_changes_every_identity() {
        let (b3_objs, b3_root) = tiny_model(HashAlgo::Blake3_256);
        let (sha_objs, sha_root) = tiny_model(HashAlgo::Sha256);
        assert_ne!(b3_root, sha_root);

        let b3 = Container::open(pack(&b3_objs, &b3_root, &opts(HashAlgo::Blake3_256)).unwrap())
            .unwrap();
        let sha =
            Container::open(pack(&sha_objs, &sha_root, &opts(HashAlgo::Sha256)).unwrap()).unwrap();
        assert_eq!(b3.header.hash, HashAlgo::Blake3_256);
        assert_eq!(sha.header.hash, HashAlgo::Sha256);
        assert_eq!(b3.index.len(), sha.index.len());

        // The blob payload is identical in both; only its name differs.
        let b3_blob = b3.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
        let sha_blob = sha.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
        assert_eq!(b3_blob.logical_len, sha_blob.logical_len);
        assert_ne!(b3_blob.digest, sha_blob.digest);
        assert_eq!(
            b3.get(&b3_blob.digest).unwrap(),
            sha.get(&sha_blob.digest).unwrap()
        );
    }

    /// A reader that cannot compute the container's digests cannot verify
    /// anything in it, so an unknown algorithm must be refused at open time
    /// rather than tolerated as an unknown-but-skippable field.
    #[test]
    fn unknown_hash_algorithm_is_refused() {
        let (objs, root) = tiny_model(HashAlgo::Blake3_256);
        let mut bytes = pack(&objs, &root, &opts(HashAlgo::Blake3_256)).unwrap();
        bytes[32] = 0x99;
        let crc = crc32c(&bytes[0..124]);
        bytes[124..128].copy_from_slice(&crc.to_le_bytes());
        match Container::open(bytes) {
            Err(e) => assert!(e.to_string().contains("R-C05"), "got: {e}"),
            Ok(_) => panic!("unknown hash algorithm must be refused"),
        }
    }

    #[test]
    fn blob_payloads_are_page_aligned() {
        for algo in ALGOS {
            let (objs, root) = tiny_model(algo);
            let bytes = pack(&objs, &root, &opts(algo)).unwrap();
            let c = Container::open(bytes).unwrap();
            let align = 1u64 << c.header.log2_align;
            for e in &c.index {
                if e.otype == otype::BLOB {
                    assert_eq!(e.offset % align, 0, "R-C08");
                }
            }
        }
    }

    #[test]
    fn tampering_is_detected() {
        for algo in ALGOS {
            let (objs, root) = tiny_model(algo);
            let mut bytes = pack(&objs, &root, &opts(algo)).unwrap();
            // Flip a bit inside the blob payload.
            let c = Container::open(bytes.clone()).unwrap();
            let blob = c.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
            let pos = blob.offset as usize + 10;
            bytes[pos] ^= 0x01;
            let c2 = Container::open(bytes).unwrap();
            // Framing still parses (CRC covers the segment, so V0 fails first
            // here); object-level verification must fail regardless.
            let err = verify(&c2).err().expect("tampering must be detected");
            let msg = err.to_string();
            assert!(
                msg.contains("R-O01") || msg.contains("R-C05"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn truncation_is_detected() {
        for algo in ALGOS {
            let (objs, root) = tiny_model(algo);
            let bytes = pack(&objs, &root, &opts(algo)).unwrap();
            let truncated = bytes[..bytes.len() - 100].to_vec();
            assert!(Container::open(truncated).is_err());
        }
    }

    #[test]
    fn header_crc_is_checked() {
        let (objs, root) = tiny_model(HashAlgo::default());
        let mut bytes = pack(&objs, &root, &PackOptions::default()).unwrap();
        bytes[13] = 20; // change log2_align without fixing the CRC
        match Container::open(bytes) {
            Err(e) => assert!(e.to_string().contains("R-C02"), "got: {e}"),
            Ok(_) => panic!("header CRC must be checked"),
        }
    }

    /// The bucket table is an accelerator, so the property that matters is
    /// that it never changes an answer — only how fast it arrives.
    #[test]
    fn the_bucket_table_finds_every_object_and_no_others() {
        let algo = HashAlgo::default();
        let objects: Vec<Object> = (0..2000u32)
            .map(|i| {
                Object::structure(
                    otype::METADATA,
                    &Value::map(vec![
                        ("t", Value::text("omni.test/filler")),
                        ("v", Value::U(1)),
                        ("i", Value::U(i as u64)),
                    ]),
                )
            })
            .collect();
        let root = objects[0].digest(algo);
        let bytes = pack(&objects, &root, &opts(algo)).unwrap();
        let c = Container::open(bytes).unwrap();

        assert_eq!(c.bucket_bits, 16, "2000 entries should get a bucket table");
        assert_eq!(c.buckets.len(), 1 << 16);
        for o in &objects {
            let d = o.digest(algo);
            assert!(c.find(&d).is_some(), "{} not found", hex(&d));
        }
        // Digests that are not present must still miss, including ones landing
        // in populated buckets.
        let mut absent = objects[0].digest(algo);
        absent[31] ^= 0xff;
        assert!(c.find(&absent).is_none());
        assert!(c.find(&[0xff; 32]).is_none());
        assert!(c.find(&[0x00; 32]).is_none());
    }

    /// A damaged bucket table must cost speed, never correctness: a reader
    /// that trusted it would report objects as absent that are present.
    #[test]
    fn a_corrupt_bucket_table_falls_back_instead_of_losing_objects() {
        let algo = HashAlgo::default();
        let objects: Vec<Object> = (0..2000u32)
            .map(|i| {
                Object::structure(
                    otype::METADATA,
                    &Value::map(vec![
                        ("t", Value::text("omni.test/filler")),
                        ("v", Value::U(1)),
                        ("i", Value::U(i as u64)),
                    ]),
                )
            })
            .collect();
        let root = objects[0].digest(algo);
        let bytes = pack(&objects, &root, &opts(algo)).unwrap();
        let good = Container::open(bytes.clone()).unwrap();

        // Find the bucket table and make it non-monotonic.
        let segs = good.segments().unwrap();
        let (idx_hdr, _, ilen) = *segs.iter().find(|(_, k, _)| *k == seg::INDEX).unwrap();
        let p = idx_hdr + SEG_HEADER_SIZE;
        let bucket_off = u64::from_le_bytes(bytes[p + 16..p + 24].try_into().unwrap()) as usize;
        let mut damaged = bytes.clone();
        let at = p + bucket_off + 4 * 30_000;
        damaged[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        // Keep the CRCs consistent so this tests the fallback, not the CRC.
        let pc = crc32c(&damaged[p..p + ilen as usize]);
        damaged[idx_hdr + 24..idx_hdr + 28].copy_from_slice(&pc.to_le_bytes());
        let hc = crc32c(&damaged[idx_hdr..idx_hdr + 28]);
        damaged[idx_hdr + 28..idx_hdr + 32].copy_from_slice(&hc.to_le_bytes());

        let c = Container::open(damaged).unwrap();
        assert_eq!(c.bucket_bits, 0, "an unusable table must be dropped");
        for o in &objects {
            assert!(
                c.find(&o.digest(algo)).is_some(),
                "fallback must still find it"
            );
        }
    }

    #[test]
    fn missing_object_is_incomplete_not_invalid() {
        // A ref to an object that is not present is a dangling ref, which is
        // legal for a partial container (§01.4).
        let algo = HashAlgo::default();
        let (mut objs, _) = tiny_model(algo);
        objs.retain(|o| o.otype != otype::BLOB);
        let root = objs
            .iter()
            .find(|o| o.otype == otype::MANIFEST)
            .unwrap()
            .digest(algo);
        let bytes = pack(&objs, &root, &opts(algo)).unwrap();
        let c = Container::open(bytes).unwrap();
        let r = verify(&c).unwrap();
        assert_eq!(r.dangling.len(), 1);
    }

    #[test]
    fn an_object_that_lies_about_its_type_is_caught() {
        // R-O02: refs carry the object type, so a reader decides what to do with
        // an object *before* fetching it. An object whose own `t` contradicts
        // the index defeats every one of those decisions.
        let algo = HashAlgo::default();
        let (mut objs, root) = tiny_model(algo);
        objs.push(Object::structure(
            otype::TOKENIZER,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/table")),
                ("v", Value::U(1)),
            ]),
        ));
        // And one with no `t` at all.
        objs.push(Object::structure(
            otype::CHAT_TEMPLATE,
            &Value::map(vec![("v", Value::U(1))]),
        ));
        let bytes = pack(&objs, &root, &opts(algo)).unwrap();
        let c = Container::open(bytes).unwrap();
        let r = verify(&c).unwrap();
        assert_eq!(r.mistyped.len(), 2, "{:?}", r.mistyped);
        assert!(r
            .mistyped
            .iter()
            .any(|(_, ot, got)| *ot == otype::TOKENIZER && got == "omni.tensor/table"));
        assert!(r
            .mistyped
            .iter()
            .any(|(_, ot, got)| *ot == otype::CHAT_TEMPLATE && got.is_empty()));
        // Every object in the honest container agrees with its index entry, so
        // the registry is exercised in both directions.
        let (objs, root) = tiny_model(algo);
        let c = Container::open(pack(&objs, &root, &opts(algo)).unwrap()).unwrap();
        assert!(verify(&c).unwrap().mistyped.is_empty());
    }
    #[test]
    fn every_object_is_found_however_the_lookup_guesses() {
        // The bucket table turns on at 1024 entries, and above it `find`
        // *guesses* where in a bucket the entry is. A guess that is wrong must
        // still find the entry — otherwise the container has objects it says are
        // not there, which is the worst failure this index can have. So: every
        // object, looked up, at a size where the guessing path is live.
        let algo = HashAlgo::default();
        let objects: Vec<Object> = (0..4000u64)
            .map(|i| {
                Object::structure(
                    otype::METADATA,
                    &Value::map(vec![
                        ("t", Value::text("omni.bench/filler")),
                        ("v", Value::U(1)),
                        ("i", Value::U(i)),
                    ]),
                )
            })
            .collect();
        let root = objects[0].digest(algo);
        let c = Container::open(pack(&objects, &root, &opts(algo)).unwrap()).unwrap();
        assert!(c.bucket_bits > 0, "the guessing path needs a bucket table");

        let mut probes = 0usize;
        for o in &objects {
            let d = o.digest(algo);
            let e = c
                .find(&d)
                .unwrap_or_else(|| panic!("{} is missing", hex(&d)));
            assert_eq!(e.digest, d);
            probes += c.probe_cost(&d);
        }
        // And the count is the same walk: a lookup that finds something compared
        // at least one entry.
        assert!(
            probes >= objects.len(),
            "{probes} probes for {} objects",
            objects.len()
        );

        // Absence is absence, not a neighbour. Flipping one bit of a real digest
        // lands in the same bucket as it, which is exactly where a guess-then-walk
        // could wander into the wrong answer.
        for o in objects.iter().take(200) {
            let mut d = o.digest(algo);
            d[31] ^= 1;
            if objects.iter().any(|x| x.digest(algo) == d) {
                continue;
            }
            assert!(c.find(&d).is_none(), "{} should not be found", hex(&d));
        }
    }
}
