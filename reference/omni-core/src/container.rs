//! The OMNI container binary format (§02).
//!
//! Implements the header, segment framing, fixed-layout object index, trailer
//! and the two-pass layout that lets a file carry byte-identical front and back
//! superblocks.

use crate::cbor::{self, Value};
use crate::crc32c::crc32c;
use crate::sha256::{hex, sha256};
use std::collections::BTreeMap;

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

pub mod otype {
    pub const BLOB: u16 = 0x0000;
    pub const MANIFEST: u16 = 0x0001;
    pub const METADATA: u16 = 0x0002;
    pub const MODEL: u16 = 0x0003;
    pub const TENSOR_TABLE: u16 = 0x0004;
    pub const TENSOR_DESC: u16 = 0x0005;
    pub const CHUNK_LIST: u16 = 0x0006;
    pub const TOKENIZER: u16 = 0x000A;
    pub const SIGNATURE: u16 = 0x0012;
    pub const OBJECT_INDEX: u16 = 0x0015;
    pub const ROSETTA: u16 = 0x0018;

    pub fn name(t: u16) -> &'static str {
        match t {
            BLOB => "Blob",
            MANIFEST => "Manifest",
            METADATA => "Metadata",
            MODEL => "Model",
            TENSOR_TABLE => "TensorTable",
            TENSOR_DESC => "TensorDesc",
            CHUNK_LIST => "ChunkList",
            TOKENIZER => "Tokenizer",
            SIGNATURE => "Signature",
            OBJECT_INDEX => "ObjectIndex",
            ROSETTA => "Rosetta",
            _ if t >= 0x8000 => "Extension(plugin)",
            _ => "Unknown",
        }
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
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Rule(id, msg) => write!(f, "{id}: {msg}"),
            Error::Cbor(e) => write!(f, "cbor: {e}"),
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
    pub payload: Vec<u8>,
    pub oflags: u8,
}

impl Object {
    pub fn structure(otype: u16, v: &Value) -> Object {
        Object {
            otype,
            payload: v.encode(),
            oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
        }
    }

    pub fn blob(payload: Vec<u8>) -> Object {
        Object {
            otype: otype::BLOB,
            payload,
            oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
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
}

impl Default for PackOptions {
    fn default() -> Self {
        PackOptions {
            log2_align: 12,
            creator: format!("omni-rs/{}", env!("CARGO_PKG_VERSION")),
            reproducible: true,
            hash: HashAlgo::default(),
        }
    }
}

struct Placed {
    digest: Digest,
    otype: u16,
    oflags: u8,
    offset: u64,
    len: u64,
}

/// Packs an object set into a sealed `core`-profile container.
///
/// Layout: header · SUPER(front) · OBJ · [PAD] · BLOB · [PAD] · INDEX ·
/// SUPER(back) · trailer. The front and back superblocks are byte-identical
/// (R-C10), which requires a fixed-point layout pass because the superblock
/// records the offsets of the segments that contain it.
pub fn pack(objects: &[Object], root: &Digest, opts: &PackOptions) -> Res<Vec<u8>> {
    if !(6..=30).contains(&opts.log2_align) {
        return Err(rule("R-C04", "log2_align out of range"));
    }
    let align = 1usize << opts.log2_align;

    // Deduplicate by digest and order deterministically by (otype, digest).
    let mut uniq: BTreeMap<(u16, Digest), &Object> = BTreeMap::new();
    for o in objects {
        uniq.insert((o.otype, o.digest(opts.hash)), o);
    }
    let ordered: Vec<&Object> = uniq.values().copied().collect();
    let (blobs, structs): (Vec<&Object>, Vec<&Object>) =
        ordered.iter().partition(|o| o.otype == otype::BLOB);

    // Fixed-point layout: the superblock's size affects the offsets it records.
    let mut sb_reserve = 4096usize;
    let (layout, sb_bytes) = loop {
        let l = compute_layout(&structs, &blobs, align, sb_reserve, opts.hash);
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
        let o = uniq
            .values()
            .find(|o| o.digest(opts.hash) == p.digest && o.otype == p.otype)
            .expect("placed object exists");
        let rel = p.offset as usize - layout.obj_payload_off;
        obj_payload[rel..rel + o.payload.len()].copy_from_slice(&o.payload);
    }
    write_segment(&mut out, layout.obj_hdr_off, seg::OBJ, 1, &obj_payload);

    if let Some(pad) = layout.pad_before_blob {
        write_segment(&mut out, pad.0, seg::PAD, 2, &vec![0u8; pad.1]);
    }

    let mut blob_payload = vec![0u8; layout.blob_payload_len];
    for p in &layout.blobs {
        let o = uniq
            .values()
            .find(|o| o.digest(opts.hash) == p.digest && o.otype == otype::BLOB)
            .expect("placed blob exists");
        let rel = p.offset as usize - layout.blob_payload_off;
        blob_payload[rel..rel + o.payload.len()].copy_from_slice(&o.payload);
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
    n_entries: usize,
}

fn compute_layout(
    structs: &[&Object],
    blobs: &[&Object],
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
            len: o.payload.len() as u64,
        });
        rel += o.payload.len();
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
            len: o.payload.len() as u64,
        });
        rel += o.payload.len();
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
    let n_entries = placed_structs.len() + placed_blobs.len();
    let index_len = IDX_HEADER_SIZE + n_entries * IDX_ENTRY_SIZE;
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
        n_entries,
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

fn build_index(l: &Layout, algo: HashAlgo) -> Vec<u8> {
    let mut entries: Vec<&Placed> = l.structs.iter().chain(l.blobs.iter()).collect();
    entries.sort_by_key(|a| a.digest);

    let mut out = vec![0u8; IDX_HEADER_SIZE + entries.len() * IDX_ENTRY_SIZE];
    out[0..4].copy_from_slice(&IDX_MAGIC);
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    out[6..8].copy_from_slice(&(IDX_ENTRY_SIZE as u16).to_le_bytes());
    out[8..16].copy_from_slice(&(entries.len() as u64).to_le_bytes());
    out[16..24].copy_from_slice(&0u64.to_le_bytes()); // no bucket table
    out[24..28].copy_from_slice(&0u32.to_le_bytes());
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
        out[b + 48..b + 56].copy_from_slice(&e.len.to_le_bytes()); // logical_len (codec=raw)
        out[b + 56..b + 58].copy_from_slice(&e.otype.to_le_bytes());
        out[b + 58] = 0; // codec: raw
        out[b + 59] = e.oflags;
        out[b + 60..b + 64].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // no aux
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

    let total_logical: u64 = l.structs.iter().chain(l.blobs.iter()).map(|p| p.len).sum();

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
                (
                    "len",
                    Value::U((IDX_HEADER_SIZE + l.n_entries * IDX_ENTRY_SIZE) as u64),
                ),
                ("entries", Value::U(l.n_entries as u64)),
                ("fmt", Value::U(1)),
            ]),
        ),
        ("segments", Value::Array(segments)),
        ("hash", Value::text("sha2-256")),
        (
            "codecs",
            Value::Array(vec![Value::map(vec![("id", Value::text("raw"))])]),
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
        (
            "stats",
            Value::map(vec![
                ("objects", Value::U(l.n_entries as u64)),
                ("blobs", Value::U(l.blobs.len() as u64)),
                ("bytes_logical", Value::U(total_logical)),
                ("bytes_stored", Value::U(total_logical)),
            ]),
        ),
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

        Ok(Container {
            bytes,
            header,
            superblock,
            index,
        })
    }

    /// Binary search over the fixed-layout index — the hot path (§02.6.2).
    pub fn find(&self, d: &Digest) -> Option<&IndexEntry> {
        self.index
            .binary_search_by(|e| e.digest.as_slice().cmp(d.as_slice()))
            .ok()
            .map(|i| &self.index[i])
    }

    /// Returns an object's bytes, verifying its digest (verification level L1).
    pub fn get(&self, d: &Digest) -> Res<&[u8]> {
        let e = self.find(d).ok_or_else(|| Error::NotFound(hex(d)))?;
        if e.oflags & oflags::EXTERNAL != 0 {
            return Err(Error::NotFound(format!("{} (external)", hex(d))));
        }
        let s = e.offset as usize;
        let n = e.stored_len as usize;
        if s.checked_add(n).is_none_or(|x| x > self.bytes.len()) {
            return Err(rule("R-C12", "object extent out of range"));
        }
        let payload = &self.bytes[s..s + n];
        if self.header.hash.digest(payload) != *d {
            return Err(rule("R-O01", format!("digest mismatch for {}", hex(d))));
        }
        Ok(payload)
    }

    pub fn get_value(&self, d: &Digest) -> Res<Value> {
        Ok(cbor::decode(self.get(d)?)?)
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

fn parse_header(b: &[u8]) -> Res<Header> {
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

    // Every byte not covered by a segment header or payload must be zero.
    let mut covered = vec![false; c.bytes.len()];
    covered[..HEADER_SIZE].fill(true);
    for (off, _, plen) in &segments {
        let end = *off + SEG_HEADER_SIZE + *plen as usize;
        covered[*off..end].fill(true);
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
        c.get(&e.digest)?;
        objects_verified += 1;
        bytes_verified += e.stored_len;
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
    })
}

/// Collects `[otype, digest]` references from a decoded structure object.
pub fn collect_refs(v: &Value, out: &mut Vec<Digest>) {
    match v {
        Value::Array(a) => {
            if a.len() == 2 {
                if let (Some(_t), Some(b)) = (a[0].as_u64(), a[1].as_bytes()) {
                    if b.len() == 32 {
                        let mut d = [0u8; 32];
                        d.copy_from_slice(b);
                        out.push(d);
                        return;
                    }
                }
            }
            for x in a {
                collect_refs(x, out);
            }
        }
        Value::Map(m) => {
            for (k, val) in m {
                collect_refs(k, out);
                collect_refs(val, out);
            }
        }
        Value::Tag(_, inner) => collect_refs(inner, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
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
}
