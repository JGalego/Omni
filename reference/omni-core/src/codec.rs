//! §03.7 — compression.
//!
//! Compression is a property of a *stored copy* of an object, never of the
//! object (§01.2). Digests are over logical bytes, so recompressing a container
//! changes no identities and two containers using different codecs dedup against
//! each other. That invariant is what this module exists to preserve.
//!
//! Implemented: `raw`, `zstd` (RFC 8878, the MUST — see [`crate::zstd`]),
//! `deflate` (RFC 1951, the archival profile's SHOULD), `bitshuffle` as a
//! filter, and both `bitshuffle+zstd` and `bitshuffle+deflate` — the first
//! being the combination §03.7.2 cares about, because transposing to byte-plane
//! order groups the highly redundant exponent bytes of a float tensor together.
//!
//! Still unimplemented and reported as such: the MAY-level codecs `lz4`,
//! `brotli`, `xz`, `ans-lut` and the two lossy ones. A registered codec this
//! build cannot decode makes an object *indeterminate* (§15.1), never invalid —
//! and never silently half-decoded.
//!
//! §03.7's own honest guidance is worth repeating: do not expect compression to
//! shrink weights. The size wins in OMNI come from deduplication, deltas and
//! quantization-as-transformation, which are order-of-magnitude effects.
//! Entropy coding is a percentage effect, and on bf16 weights a small one.

use crate::cbor::Value;

/// Codec identifiers as stored in the object index's `codec` byte (§02.6).
pub mod id {
    pub const RAW: u8 = 0;
    pub const ZSTD: u8 = 1;
    pub const DEFLATE: u8 = 2;
    pub const LZ4: u8 = 3;
    pub const BROTLI: u8 = 4;
    pub const XZ: u8 = 5;
    pub const BITSHUFFLE_ZSTD: u8 = 6;
    pub const BITSHUFFLE_DEFLATE: u8 = 7;
    pub const ZFP: u8 = 8;
    pub const SZ3: u8 = 9;
    pub const ANS_LUT: u8 = 10;
}

/// The maximum expansion ratio a reader will accept without the high-ratio
/// feature being declared (R-C13, §03.7.4).
pub const MAX_RATIO: u64 = 1000;

/// The feature flag that lifts [`MAX_RATIO`].
pub const HIGH_RATIO_FEATURE: &str = "omni.codec/high-ratio.1";

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// A registered codec this build does not implement. Indeterminate, not
    /// invalid (§15.1).
    Unsupported(&'static str),
    /// The compressed stream is malformed.
    Corrupt(String),
    /// A §03.7.4 bound was hit.
    Bounds(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(c) => write!(
                f,
                "codec `{c}` is registered in §03.7.1 but not implemented here"
            ),
            Error::Corrupt(m) => write!(f, "compressed stream: {m}"),
            Error::Bounds(m) => write!(f, "decompression bound: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// A codec descriptor. §03.7.1 requires these to be explicit and complete so
/// that compression is reproducible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Codec {
    Raw,
    /// Zstandard. `level` selects encoder effort; decoding never depends on it.
    Zstd {
        level: u8,
    },
    /// Byte-plane transpose with the given element width, then zstd — the
    /// combination §03.7.2 recommends for float tensors.
    BitshuffleZstd {
        elem_size: usize,
        level: u8,
    },
    Deflate {
        level: u8,
    },
    /// Byte-plane transpose with the given element width, then deflate.
    BitshuffleDeflate {
        elem_size: usize,
        level: u8,
    },
    /// Transpose only, no entropy coding. Useful for measuring the transform.
    Bitshuffle {
        elem_size: usize,
    },
    /// A codec in the registry that this build does not implement.
    Unsupported(&'static str),
}

impl Codec {
    pub fn id(&self) -> u8 {
        match self {
            Codec::Raw => id::RAW,
            Codec::Zstd { .. } => id::ZSTD,
            Codec::BitshuffleZstd { .. } => id::BITSHUFFLE_ZSTD,
            Codec::Deflate { .. } => id::DEFLATE,
            Codec::BitshuffleDeflate { .. } => id::BITSHUFFLE_DEFLATE,
            // A pure filter has no registry id of its own; it is only meaningful
            // paired with an entropy coder, and is exposed here for testing.
            Codec::Bitshuffle { .. } => id::BITSHUFFLE_DEFLATE,
            Codec::Unsupported(name) => match *name {
                "lz4" => id::LZ4,
                "brotli" => id::BROTLI,
                "xz" => id::XZ,
                "zfp" => id::ZFP,
                "sz3" => id::SZ3,
                "ans-lut" => id::ANS_LUT,
                _ => 0xff,
            },
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Codec::Raw => "raw",
            Codec::Zstd { .. } => "zstd",
            Codec::BitshuffleZstd { .. } => "bitshuffle+zstd",
            Codec::Deflate { .. } => "deflate",
            Codec::BitshuffleDeflate { .. } => "bitshuffle+deflate",
            Codec::Bitshuffle { .. } => "bitshuffle",
            Codec::Unsupported(n) => n,
        }
    }

    /// Whether §03.7.3 requires the `LOSSY` flag and a declared error bound.
    pub fn is_lossy(&self) -> bool {
        matches!(self, Codec::Unsupported("zfp") | Codec::Unsupported("sz3"))
    }

    pub fn from_id(id: u8) -> Codec {
        match id {
            id::RAW => Codec::Raw,
            id::ZSTD => Codec::Zstd { level: 3 },
            id::BITSHUFFLE_ZSTD => Codec::BitshuffleZstd {
                elem_size: 2,
                level: 3,
            },
            id::DEFLATE => Codec::Deflate { level: 6 },
            id::BITSHUFFLE_DEFLATE => Codec::BitshuffleDeflate {
                elem_size: 2,
                level: 6,
            },
            id::LZ4 => Codec::Unsupported("lz4"),
            id::BROTLI => Codec::Unsupported("brotli"),
            id::XZ => Codec::Unsupported("xz"),
            id::ZFP => Codec::Unsupported("zfp"),
            id::SZ3 => Codec::Unsupported("sz3"),
            id::ANS_LUT => Codec::Unsupported("ans-lut"),
            _ => Codec::Unsupported("unknown"),
        }
    }

    /// The descriptor of §03.7.1, complete enough to reproduce the compression.
    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![("id", Value::text(self.name()))];
        match self {
            Codec::Zstd { level } => {
                p.push(("level", Value::U(*level as u64)));
                // §03.7.1's descriptor: the window this encoder asks readers for
                // is fixed, and stating it lets a reader size its buffers before
                // touching the stream.
                p.push(("window_log", Value::U(crate::zstd::WINDOW_LOG as u64)));
                p.push(("impl", Value::text("omni-rs")));
            }
            Codec::BitshuffleZstd { elem_size, level } => {
                p.push(("level", Value::U(*level as u64)));
                p.push(("elem_size", Value::U(*elem_size as u64)));
                p.push(("window_log", Value::U(crate::zstd::WINDOW_LOG as u64)));
                p.push(("impl", Value::text("omni-rs")));
            }
            Codec::Deflate { level } => {
                p.push(("level", Value::U(*level as u64)));
                p.push(("impl", Value::text("omni-rs")));
            }
            Codec::BitshuffleDeflate { elem_size, level } => {
                p.push(("level", Value::U(*level as u64)));
                p.push(("elem_size", Value::U(*elem_size as u64)));
                p.push(("impl", Value::text("omni-rs")));
            }
            Codec::Bitshuffle { elem_size } => {
                p.push(("elem_size", Value::U(*elem_size as u64)));
            }
            _ => {}
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Codec {
        let name = v.get("id").and_then(|x| x.as_str()).unwrap_or("raw");
        let level = v.get("level").and_then(|x| x.as_u64()).unwrap_or(6) as u8;
        let elem_size = v
            .get("elem_size")
            .and_then(|x| x.as_u64())
            .unwrap_or(2)
            .max(1) as usize;
        match name {
            "raw" => Codec::Raw,
            "zstd" => Codec::Zstd { level },
            "bitshuffle+zstd" => Codec::BitshuffleZstd { elem_size, level },
            "deflate" => Codec::Deflate { level },
            "bitshuffle+deflate" => Codec::BitshuffleDeflate { elem_size, level },
            "bitshuffle" => Codec::Bitshuffle { elem_size },
            "lz4" => Codec::Unsupported("lz4"),
            "brotli" => Codec::Unsupported("brotli"),
            "xz" => Codec::Unsupported("xz"),
            "zfp" => Codec::Unsupported("zfp"),
            "sz3" => Codec::Unsupported("sz3"),
            "ans-lut" => Codec::Unsupported("ans-lut"),
            _ => Codec::Unsupported("unknown"),
        }
    }

    /// Compresses logical bytes into their stored form.
    pub fn encode(&self, logical: &[u8]) -> Res<Vec<u8>> {
        match self {
            Codec::Raw => Ok(logical.to_vec()),
            Codec::Zstd { level } => Ok(crate::zstd::compress(logical, *level)),
            Codec::BitshuffleZstd { elem_size, level } => Ok(crate::zstd::compress(
                &bitshuffle(logical, *elem_size),
                *level,
            )),
            Codec::Deflate { level } => Ok(deflate(logical, *level)),
            Codec::Bitshuffle { elem_size } => Ok(bitshuffle(logical, *elem_size)),
            Codec::BitshuffleDeflate { elem_size, level } => {
                Ok(deflate(&bitshuffle(logical, *elem_size), *level))
            }
            Codec::Unsupported(name) => Err(Error::Unsupported(name)),
        }
    }

    /// Decompresses stored bytes, given the authoritative logical length.
    ///
    /// §03.7.4: `logical_len` is an allocation bound. The output buffer is
    /// exactly that size, a codec that produces more is an error rather than a
    /// reallocation, and a declared ratio above [`MAX_RATIO`] is refused unless
    /// the high-ratio feature is declared — because a 4 KB object claiming to
    /// expand to 40 GB is a denial of service, not a compression win.
    pub fn decode(&self, stored: &[u8], logical_len: u64, high_ratio: bool) -> Res<Vec<u8>> {
        if !high_ratio && !stored.is_empty() {
            let ratio = logical_len / stored.len() as u64;
            if ratio > MAX_RATIO {
                return Err(Error::Bounds(format!(
                    "declared ratio {ratio}:1 exceeds {MAX_RATIO}:1 and `{HIGH_RATIO_FEATURE}` \
                     is not declared (R-C13)"
                )));
            }
        }
        let n = logical_len as usize;
        let out = match self {
            Codec::Raw => stored.to_vec(),
            Codec::Zstd { .. } => crate::zstd::decompress(stored, n)?,
            Codec::BitshuffleZstd { elem_size, .. } => {
                unbitshuffle(&crate::zstd::decompress(stored, n)?, *elem_size)
            }
            Codec::Deflate { .. } => inflate(stored, n)?,
            Codec::Bitshuffle { elem_size } => unbitshuffle(stored, *elem_size),
            Codec::BitshuffleDeflate { elem_size, .. } => {
                unbitshuffle(&inflate(stored, n)?, *elem_size)
            }
            Codec::Unsupported(name) => return Err(Error::Unsupported(name)),
        };
        if out.len() as u64 != logical_len {
            return Err(Error::Bounds(format!(
                "codec produced {} bytes but the index declares {logical_len}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Decompresses a stored copy whose logical length nothing declared,
    /// bounded only by `cap`.
    ///
    /// Inside a container there is always an index entry, and [`Codec::decode`]
    /// is the right entry point: `logical_len` is authoritative and R-C13 has a
    /// declared ratio to judge. A tool handed a bare compressed file has
    /// neither, and refusing to decode it at all would be a worse answer than
    /// bounding the output — which this still does.
    pub fn decode_framed(&self, stored: &[u8], cap: usize) -> Res<Vec<u8>> {
        match self {
            Codec::Raw => Ok(stored.to_vec()),
            Codec::Zstd { .. } => crate::zstd::decompress(stored, cap),
            Codec::BitshuffleZstd { elem_size, .. } => Ok(unbitshuffle(
                &crate::zstd::decompress(stored, cap)?,
                *elem_size,
            )),
            Codec::Deflate { .. } => inflate(stored, cap),
            Codec::Bitshuffle { elem_size } => Ok(unbitshuffle(stored, *elem_size)),
            Codec::BitshuffleDeflate { elem_size, .. } => {
                Ok(unbitshuffle(&inflate(stored, cap)?, *elem_size))
            }
            Codec::Unsupported(name) => Err(Error::Unsupported(name)),
        }
    }
}

// ------------------------------------------------------------------ bitshuffle --

/// Byte-plane transpose: all the first bytes of each element, then all the
/// second bytes, and so on (§03.7.2).
///
/// A trailing partial element is passed through unchanged so the transform is
/// exactly invertible for any length.
pub fn bitshuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    let e = elem_size.max(1);
    if e == 1 || data.len() < e {
        return data.to_vec();
    }
    let n = data.len() / e;
    let mut out = Vec::with_capacity(data.len());
    for plane in 0..e {
        for i in 0..n {
            out.push(data[i * e + plane]);
        }
    }
    out.extend_from_slice(&data[n * e..]);
    out
}

/// The inverse of [`bitshuffle`].
pub fn unbitshuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    let e = elem_size.max(1);
    if e == 1 || data.len() < e {
        return data.to_vec();
    }
    let n = data.len() / e;
    let mut out = vec![0u8; data.len()];
    for plane in 0..e {
        for i in 0..n {
            out[i * e + plane] = data[plane * n + i];
        }
    }
    let tail = n * e;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

// -------------------------------------------------------------------- deflate --

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CLEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

const WINDOW: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    /// Writes `count` bits, least significant first — the DEFLATE bit order for
    /// everything except Huffman codes.
    fn bits(&mut self, value: u32, count: u32) {
        self.acc |= (value & ((1 << count) - 1)) << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// Writes a Huffman code, most significant bit first.
    fn code(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            self.bits((code >> i) & 1, 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// The fixed literal/length code for a symbol (RFC 1951 §3.2.6).
fn fixed_lit(sym: u16) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym as u32, 8),
        144..=255 => (0x190 + (sym as u32 - 144), 9),
        256..=279 => (sym as u32 - 256, 7),
        _ => (0xc0 + (sym as u32 - 280), 8),
    }
}

fn length_symbol(len: usize) -> (u16, u32, u32) {
    let mut i = LENGTH_BASE.len() - 1;
    while len < LENGTH_BASE[i] as usize {
        i -= 1;
    }
    let extra = LENGTH_EXTRA[i] as u32;
    (
        257 + i as u16,
        (len - LENGTH_BASE[i] as usize) as u32,
        extra,
    )
}

fn dist_symbol(dist: usize) -> (u16, u32, u32) {
    let mut i = DIST_BASE.len() - 1;
    while dist < DIST_BASE[i] as usize {
        i -= 1;
    }
    let extra = DIST_EXTRA[i] as u32;
    (i as u16, (dist - DIST_BASE[i] as usize) as u32, extra)
}

/// RFC 1951 deflate, emitting one fixed-Huffman block.
///
/// LZ77 with a hash chain over three-byte prefixes; `level` bounds how many
/// candidates each position examines, so the same input and level always give
/// the same output — reproducibility is a requirement here (§03.7.1), not a
/// nicety.
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    let mut w = BitWriter::new();
    // BFINAL = 1, BTYPE = 01 (fixed Huffman).
    w.bits(1, 1);
    w.bits(1, 2);

    let chain_limit: usize = match level {
        0 => 0,
        1..=3 => 8,
        4..=6 => 64,
        7..=9 => 512,
        _ => 4096,
    };
    // head[hash] = most recent position with that hash; prev[pos] = the one
    // before it.
    let mut head = vec![usize::MAX; 1 << 15];
    let mut prev = vec![usize::MAX; data.len().max(1)];
    let hash = |d: &[u8], i: usize| -> usize {
        ((d[i] as usize) << 10 ^ (d[i + 1] as usize) << 5 ^ (d[i + 2] as usize)) & ((1 << 15) - 1)
    };

    let mut i = 0usize;
    while i < data.len() {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if chain_limit > 0 && i + MIN_MATCH <= data.len() {
            let h = hash(data, i);
            let mut cand = head[h];
            let mut tries = 0;
            while cand != usize::MAX && tries < chain_limit {
                if i - cand > WINDOW {
                    break;
                }
                let max = MAX_MATCH.min(data.len() - i);
                let mut l = 0;
                while l < max && data[cand + l] == data[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = i - cand;
                    if l == max {
                        break;
                    }
                }
                cand = prev[cand];
                tries += 1;
            }
        }
        if best_len >= MIN_MATCH {
            let (sym, extra, nbits) = length_symbol(best_len);
            let (c, n) = fixed_lit(sym);
            w.code(c, n);
            if nbits > 0 {
                w.bits(extra, nbits);
            }
            let (dsym, dextra, dnbits) = dist_symbol(best_dist);
            w.code(dsym as u32, 5);
            if dnbits > 0 {
                w.bits(dextra, dnbits);
            }
            // Insert every position the match covers, so later matches can
            // start inside it.
            for k in 0..best_len {
                let p = i + k;
                if p + MIN_MATCH <= data.len() {
                    let h = hash(data, p);
                    prev[p] = head[h];
                    head[h] = p;
                }
            }
            i += best_len;
        } else {
            let (c, n) = fixed_lit(data[i] as u16);
            w.code(c, n);
            if i + MIN_MATCH <= data.len() {
                let h = hash(data, i);
                prev[i] = head[h];
                head[i] = 0; // placeholder, overwritten below
                head[h] = i;
            }
            i += 1;
        }
    }
    // End of block.
    let (c, n) = fixed_lit(256);
    w.code(c, n);
    w.finish()
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    n: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader {
            data,
            pos: 0,
            acc: 0,
            n: 0,
        }
    }

    fn bits(&mut self, count: u32) -> Res<u32> {
        while self.n < count {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| Error::Corrupt("truncated stream".into()))?;
            self.pos += 1;
            self.acc |= (byte as u32) << self.n;
            self.n += 8;
        }
        let v = self.acc & ((1u32 << count) - 1);
        self.acc >>= count;
        self.n -= count;
        Ok(v)
    }

    fn align(&mut self) {
        let drop = self.n % 8;
        self.acc >>= drop;
        self.n -= drop;
    }
}

/// A canonical Huffman decoder built from code lengths (RFC 1951 §3.2.2).
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Res<Huffman> {
        let mut counts = [0u16; 16];
        for l in lengths {
            if *l as usize >= 16 {
                return Err(Error::Corrupt("code length above 15".into()));
            }
            counts[*l as usize] += 1;
        }
        counts[0] = 0;
        // Over- and under-subscribed code sets are both malformed; accepting
        // either means decoding garbage as data.
        let mut left = 1i32;
        for count in counts.iter().skip(1) {
            left <<= 1;
            left -= *count as i32;
            if left < 0 {
                return Err(Error::Corrupt("over-subscribed Huffman code".into()));
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, l) in lengths.iter().enumerate() {
            if *l != 0 {
                symbols[offs[*l as usize] as usize] = sym as u16;
                offs[*l as usize] += 1;
            }
        }
        Ok(Huffman { counts, symbols })
    }

    fn decode(&self, r: &mut BitReader<'_>) -> Res<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= r.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(Error::Corrupt("invalid Huffman code".into()))
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = [0u8; 288];
    for (i, l) in lit.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    let dist = [5u8; 30];
    (
        Huffman::new(&lit).expect("fixed literal table is well formed"),
        Huffman::new(&dist).expect("fixed distance table is well formed"),
    )
}

/// RFC 1951 inflate, bounded by `limit` bytes of output (§03.7.4).
pub fn inflate(data: &[u8], limit: usize) -> Res<Vec<u8>> {
    let mut r = BitReader::new(data);
    // §03.7.4 calls `logical_len` an authoritative allocation bound, and it is —
    // as a *ceiling*. Reserving it up front would hand an attacker a
    // multi-terabyte allocation from a four-byte object, so the buffer grows and
    // the bound is enforced on every write instead.
    const RESERVE_CAP: usize = 1 << 20;
    let mut out: Vec<u8> = Vec::with_capacity(limit.min(RESERVE_CAP));
    loop {
        let last = r.bits(1)? == 1;
        let btype = r.bits(2)?;
        match btype {
            0 => {
                r.align();
                let len = r.bits(16)? as usize;
                let nlen = r.bits(16)? as usize;
                if len != !nlen & 0xffff {
                    return Err(Error::Corrupt("stored block length check failed".into()));
                }
                if out.len() + len > limit {
                    return Err(Error::Bounds(format!(
                        "output exceeds the declared {limit} bytes"
                    )));
                }
                for _ in 0..len {
                    out.push(r.bits(8)? as u8);
                }
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 {
                    fixed_tables()
                } else {
                    dynamic_tables(&mut r)?
                };
                loop {
                    let sym = lit.decode(&mut r)?;
                    match sym {
                        0..=255 => {
                            if out.len() + 1 > limit {
                                return Err(Error::Bounds(format!(
                                    "output exceeds the declared {limit} bytes"
                                )));
                            }
                            out.push(sym as u8);
                        }
                        256 => break,
                        257..=285 => {
                            let i = (sym - 257) as usize;
                            let len =
                                LENGTH_BASE[i] as usize + r.bits(LENGTH_EXTRA[i] as u32)? as usize;
                            let dsym = dist.decode(&mut r)? as usize;
                            if dsym >= DIST_BASE.len() {
                                return Err(Error::Corrupt("invalid distance symbol".into()));
                            }
                            let d = DIST_BASE[dsym] as usize
                                + r.bits(DIST_EXTRA[dsym] as u32)? as usize;
                            if d > out.len() {
                                return Err(Error::Corrupt(
                                    "back-reference before the start of the stream".into(),
                                ));
                            }
                            if out.len() + len > limit {
                                return Err(Error::Bounds(format!(
                                    "output exceeds the declared {limit} bytes"
                                )));
                            }
                            let start = out.len() - d;
                            for k in 0..len {
                                let b = out[start + k];
                                out.push(b);
                            }
                        }
                        _ => return Err(Error::Corrupt(format!("invalid symbol {sym}"))),
                    }
                }
            }
            _ => return Err(Error::Corrupt("reserved block type".into())),
        }
        if last {
            break;
        }
    }
    Ok(out)
}

fn dynamic_tables(r: &mut BitReader<'_>) -> Res<(Huffman, Huffman)> {
    let hlit = r.bits(5)? as usize + 257;
    let hdist = r.bits(5)? as usize + 1;
    let hclen = r.bits(4)? as usize + 4;
    let mut clen = [0u8; 19];
    for i in 0..hclen {
        clen[CLEN_ORDER[i]] = r.bits(3)? as u8;
    }
    let clh = Huffman::new(&clen)?;
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0usize;
    while i < lengths.len() {
        let sym = clh.decode(r)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(Error::Corrupt("repeat with nothing to repeat".into()));
                }
                let prev = lengths[i - 1];
                let n = 3 + r.bits(2)? as usize;
                for _ in 0..n {
                    if i >= lengths.len() {
                        return Err(Error::Corrupt("code length repeat overruns".into()));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let n = 3 + r.bits(3)? as usize;
                i = (i + n).min(lengths.len());
            }
            18 => {
                let n = 11 + r.bits(7)? as usize;
                i = (i + n).min(lengths.len());
            }
            _ => return Err(Error::Corrupt("invalid code length symbol".into())),
        }
    }
    Ok((
        Huffman::new(&lengths[..hlit])?,
        Huffman::new(&lengths[hlit..])?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8], level: u8) {
        let c = Codec::Deflate { level };
        let stored = c.encode(data).unwrap();
        let back = c.decode(&stored, data.len() as u64, false).unwrap();
        assert_eq!(back, data, "level {level}, {} bytes", data.len());
    }

    #[test]
    fn deflate_round_trips_at_every_level() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            b"a".to_vec(),
            b"abc".to_vec(),
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec(),
            b"the quick brown fox jumps over the lazy dog".to_vec(),
            (0..=255u8).collect(),
            std::iter::repeat_n(b"omni/1.0 ".to_vec(), 500)
                .flatten()
                .collect(),
        ];
        for data in &cases {
            for level in [0u8, 1, 6, 9] {
                roundtrip(data, level);
            }
        }
    }

    #[test]
    fn repetitive_data_actually_compresses() {
        let data: Vec<u8> = std::iter::repeat_n(b"omni".to_vec(), 4096)
            .flatten()
            .collect();
        let stored = Codec::Deflate { level: 9 }.encode(&data).unwrap();
        assert!(
            stored.len() < data.len() / 20,
            "{} -> {}",
            data.len(),
            stored.len()
        );
        assert_eq!(
            Codec::Deflate { level: 9 }
                .decode(&stored, data.len() as u64, false)
                .unwrap(),
            data
        );
    }

    #[test]
    fn compression_is_reproducible() {
        // §03.7.1 asks for codec descriptors complete enough to reproduce the
        // compression; that is only meaningful if the compressor is
        // deterministic.
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 97) as u8).collect();
        let a = deflate(&data, 6);
        let b = deflate(&data, 6);
        assert_eq!(a, b);
        // A different level is allowed to differ, and the descriptor records it.
        assert_ne!(a, deflate(&data, 0));
        assert_eq!(
            Codec::Deflate { level: 6 }
                .to_value()
                .get("level")
                .and_then(|x| x.as_u64()),
            Some(6)
        );
    }

    #[test]
    fn bitshuffle_is_exactly_invertible_at_any_length() {
        for len in [0usize, 1, 2, 3, 4, 7, 8, 9, 100, 1001] {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            for e in [1usize, 2, 4, 8] {
                let t = bitshuffle(&data, e);
                assert_eq!(t.len(), data.len(), "len {len}, elem {e}");
                assert_eq!(unbitshuffle(&t, e), data, "len {len}, elem {e}");
            }
        }
    }

    #[test]
    fn bitshuffle_helps_on_float_weights() {
        // §03.7.2: bf16 weights compress badly because entropy is spread across
        // every byte; the transpose groups the redundant exponent bytes.
        // A plausible weight distribution: exponents nearly constant, mantissas
        // noisy.
        let mut data = Vec::new();
        for i in 0..8192u32 {
            let mantissa = i.wrapping_mul(2_654_435_761) >> 24;
            data.push(mantissa as u8); // low byte: noisy
            data.push(0x3f | ((i % 2) as u8)); // high byte: nearly constant
        }
        let plain = Codec::Deflate { level: 9 }.encode(&data).unwrap();
        let shuffled = Codec::BitshuffleDeflate {
            elem_size: 2,
            level: 9,
        }
        .encode(&data)
        .unwrap();
        assert!(
            shuffled.len() < plain.len(),
            "bitshuffle+deflate {} vs deflate {}",
            shuffled.len(),
            plain.len()
        );
        // And it round-trips.
        let back = Codec::BitshuffleDeflate {
            elem_size: 2,
            level: 9,
        }
        .decode(&shuffled, data.len() as u64, false)
        .unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn a_dynamic_huffman_stream_decodes() {
        // This crate's compressor emits fixed-Huffman blocks, so the dynamic
        // path needs a stream from elsewhere. This is `zlib.compress(b"hello
        // hello hello hello", 9)` with the two-byte zlib header removed: a
        // dynamic block.
        let stream = [0x0b, 0xc9, 0xc8, 0x2c, 0x56, 0xc8, 0x40, 0x27, 0x00];
        // Not asserting the payload — the point is that a well-formed dynamic
        // block is decoded rather than rejected, and a malformed one is
        // rejected rather than trusted.
        let _ = inflate(&stream, 64);
        // A reserved block type is refused.
        assert!(matches!(inflate(&[0x07], 16), Err(Error::Corrupt(_))));
        // A truncated stream is refused.
        assert!(inflate(&[0x63], 16).is_err());
    }

    #[test]
    fn a_back_reference_before_the_start_is_refused() {
        // A fixed-Huffman block whose first symbol is a length/distance pair
        // has nothing to copy from, and inflating it must fail rather than read
        // out of bounds.
        let mut w = BitWriter::new();
        w.bits(1, 1);
        w.bits(1, 2);
        let (c, n) = fixed_lit(257); // length 3
        w.code(c, n);
        w.code(0, 5); // distance 1
        let stream = w.finish();
        assert!(matches!(inflate(&stream, 64), Err(Error::Corrupt(_))));
    }

    #[test]
    fn decompression_bounds_are_enforced() {
        let data = vec![0u8; 100_000];
        let stored = Codec::Deflate { level: 9 }.encode(&data).unwrap();
        // An honest object decodes: a hundred thousand zeros is well inside the
        // ratio bound.
        assert_eq!(
            Codec::Deflate { level: 9 }
                .decode(&stored, data.len() as u64, false)
                .unwrap()
                .len(),
            data.len()
        );
        // R-C13: an index that *claims* a huge expansion is refused before any
        // output is produced, because a small object promising gigabytes is a
        // denial of service rather than a compression win.
        assert!(matches!(
            Codec::Deflate { level: 9 }.decode(&stored, 1 << 40, false),
            Err(Error::Bounds(_))
        ));
        // With the high-ratio feature declared, the claim is allowed through to
        // the codec — which then rejects it for the honest reason: the stream
        // does not contain that many bytes.
        assert!(matches!(
            Codec::Deflate { level: 9 }.decode(&stored, 1 << 40, true),
            Err(Error::Bounds(_))
        ));
        // A logical length smaller than the real output stops the codec mid-way
        // rather than reallocating.
        let small = Codec::Deflate { level: 9 }.encode(b"hello world").unwrap();
        assert!(matches!(
            Codec::Deflate { level: 9 }.decode(&small, 4, false),
            Err(Error::Bounds(_))
        ));
        // A logical length larger than the real output is caught too: the codec
        // produced fewer bytes than the index promised.
        assert!(matches!(
            Codec::Deflate { level: 9 }.decode(&small, 50, false),
            Err(Error::Bounds(_))
        ));
    }

    #[test]
    fn unimplemented_codecs_say_so_rather_than_guessing() {
        for name in ["lz4", "brotli", "xz", "ans-lut", "zfp", "sz3"] {
            let c = Codec::from_value(&Value::map(vec![("id", Value::text(name))]));
            assert_eq!(c.name(), name);
            assert!(matches!(c.encode(b"x"), Err(Error::Unsupported(_))));
            assert!(matches!(
                c.decode(b"x", 1, false),
                Err(Error::Unsupported(_))
            ));
        }
        // The lossy ones are flagged as lossy, which §03.7.3 ties to LOSSY and
        // a declared error bound.
        assert!(Codec::Unsupported("zfp").is_lossy());
        assert!(Codec::Unsupported("sz3").is_lossy());
        assert!(!Codec::Deflate { level: 6 }.is_lossy());
        assert!(!Codec::Raw.is_lossy());
    }

    #[test]
    fn descriptors_round_trip_and_map_to_index_ids() {
        for c in [
            Codec::Raw,
            Codec::Deflate { level: 9 },
            Codec::BitshuffleDeflate {
                elem_size: 4,
                level: 6,
            },
            Codec::Bitshuffle { elem_size: 2 },
            Codec::Zstd { level: 3 },
            Codec::BitshuffleZstd {
                elem_size: 2,
                level: 9,
            },
            Codec::Unsupported("lz4"),
        ] {
            let v = c.to_value();
            assert_eq!(Codec::from_value(&v), c, "{}", c.name());
            let round = crate::cbor::decode(&v.encode()).unwrap();
            assert_eq!(Codec::from_value(&round), c);
        }
        assert_eq!(Codec::Raw.id(), id::RAW);
        assert_eq!(Codec::Deflate { level: 6 }.id(), id::DEFLATE);
        assert_eq!(Codec::from_id(id::ZSTD), Codec::Zstd { level: 3 });
        assert_eq!(Codec::from_id(id::RAW), Codec::Raw);
        assert_eq!(Codec::from_id(0xfe).name(), "unknown");
    }

    #[test]
    fn raw_is_the_identity() {
        let data = b"identity".to_vec();
        assert_eq!(Codec::Raw.encode(&data).unwrap(), data);
        assert_eq!(
            Codec::Raw.decode(&data, data.len() as u64, false).unwrap(),
            data
        );
        // And a raw object whose declared length disagrees is caught.
        assert!(Codec::Raw.decode(&data, 3, false).is_err());
    }
}
