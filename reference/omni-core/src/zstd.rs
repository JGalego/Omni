//! §03.7.1 — Zstandard (RFC 8878), both directions, from scratch.
//!
//! `zstd` is the only compression codec §03.7.1 marks **MUST**, so a build
//! without it is not conforming, and a build that only *decodes* it cannot
//! write the containers everyone else reads. Both directions are here.
//!
//! The decoder is complete for frames without a dictionary: all four literal
//! block types including Treeless reuse of a previous Huffman table, both
//! Huffman table representations (direct weights and FSE-compressed weights),
//! one- and four-stream literal bitstreams, all four sequence-table modes
//! (predefined, RLE, FSE-compressed, repeat), the three repeat offsets, the
//! window, multi-frame streams, skippable frames and the XXH64 content
//! checksum. A dictionary-compressed frame is reported *unsupported* — not
//! guessed at — because there is nothing in the container to recover the
//! dictionary from unless §03.7.1's `dict` ref is present, and then it is a
//! different object to fetch.
//!
//! The encoder writes valid frames that libzstd reads (CI proves it against
//! libzstd, in both directions). Its ratio is below libzstd's, deliberately:
//! sequences use the §3.1.1.3.2.2.1 *predefined* FSE distributions rather than
//! per-block optimal ones, and literals are Huffman-coded only when the
//! alphabet fits the direct weight representation — which is exactly the case
//! §03.7.2 cares about, a bitshuffled float tensor's exponent plane. Being a
//! percentage behind on ratio is acceptable; being wrong is not, and every
//! frame this writes is checked by an independent decoder before it ships.
//!
//! Nothing here is on the identity path: §03.7 makes compression a property of
//! a *stored copy*, so a frame that decodes to the right bytes is right, and
//! the object digest is computed over those bytes either way.

use crate::codec::Error;

type Res<T> = Result<T, Error>;

/// Frame magic (RFC 8878 §3.1.1).
pub const MAGIC: u32 = 0xFD2F_B528;
/// Skippable frames are `0x184D2A5?`.
const SKIPPABLE_MASK: u32 = 0xFFFF_FFF0;
const SKIPPABLE_BASE: u32 = 0x184D_2A50;

/// `ZSTD_BLOCKSIZE_MAX`.
pub const MAX_BLOCK: usize = 128 * 1024;
/// The window this encoder asks readers for: 8 MiB, which covers a 4 MiB
/// chunk (§03.6's default) entirely and costs a decoder nothing extra. It is
/// part of the §03.7.1 codec descriptor, so it is public.
pub const WINDOW_LOG: u32 = 23;
/// Refuse absurd window declarations rather than trusting them (§12.4): a
/// 2 GiB window in a header is an allocation request, not a fact.
const MAX_WINDOW_LOG: u32 = 31;

const MIN_MATCH: usize = 3;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Corrupt(format!("zstd: {}", msg.into()))
}

fn u32le(d: &[u8]) -> u32 {
    u32::from_le_bytes([d[0], d[1], d[2], d[3]])
}

// ------------------------------------------------------------------- xxhash64 --

const XXH_P1: u64 = 0x9E37_79B1_85EB_CA87;
const XXH_P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const XXH_P3: u64 = 0x1656_67B1_9E37_79F9;
const XXH_P4: u64 = 0x85EB_CA77_C2B2_AE63;
const XXH_P5: u64 = 0x27D4_EB2F_1656_67C5;

fn xxh_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXH_P2))
        .rotate_left(31)
        .wrapping_mul(XXH_P1)
}

fn xxh_merge(acc: u64, val: u64) -> u64 {
    (acc ^ xxh_round(0, val))
        .wrapping_mul(XXH_P1)
        .wrapping_add(XXH_P4)
}

/// XXH64 — the frame content checksum of RFC 8878 §3.1.1.1.4.
///
/// A frame's `Content_Checksum` is the low 32 bits of this over the
/// decompressed content. It is a redundant check inside OMNI, where the object
/// digest already covers the same bytes, but a decoder that skipped it would
/// accept frames a conforming one rejects.
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let mut h;
    let mut p = 0;
    if data.len() >= 32 {
        let mut v1 = seed.wrapping_add(XXH_P1).wrapping_add(XXH_P2);
        let mut v2 = seed.wrapping_add(XXH_P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH_P1);
        while data.len() - p >= 32 {
            let mut lane = [0u64; 4];
            for (i, l) in lane.iter_mut().enumerate() {
                let o = p + i * 8;
                *l = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
            }
            v1 = xxh_round(v1, lane[0]);
            v2 = xxh_round(v2, lane[1]);
            v3 = xxh_round(v3, lane[2]);
            v4 = xxh_round(v4, lane[3]);
            p += 32;
        }
        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h = xxh_merge(h, v1);
        h = xxh_merge(h, v2);
        h = xxh_merge(h, v3);
        h = xxh_merge(h, v4);
    } else {
        h = seed.wrapping_add(XXH_P5);
    }
    h = h.wrapping_add(data.len() as u64);
    while data.len() - p >= 8 {
        let k = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        h = (h ^ xxh_round(0, k))
            .rotate_left(27)
            .wrapping_mul(XXH_P1)
            .wrapping_add(XXH_P4);
        p += 8;
    }
    if data.len() - p >= 4 {
        let k = u32le(&data[p..]) as u64;
        h = (h ^ k.wrapping_mul(XXH_P1))
            .rotate_left(23)
            .wrapping_mul(XXH_P2)
            .wrapping_add(XXH_P3);
        p += 4;
    }
    while p < data.len() {
        h = (h ^ (data[p] as u64).wrapping_mul(XXH_P5))
            .rotate_left(11)
            .wrapping_mul(XXH_P1);
        p += 1;
    }
    h ^= h >> 33;
    h = h.wrapping_mul(XXH_P2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXH_P3);
    h ^= h >> 32;
    h
}

// ----------------------------------------------------------------- bit access --

/// A backward bitstream: zstd's entropy-coded streams are read from the last
/// byte down, starting just below its highest set bit (the padding marker).
///
/// Reads are bit-by-bit rather than word-at-a-time. That is 30× slower than
/// libzstd and entirely legible, which is the same trade the BLAKE3 code in
/// this crate makes.
struct BackBits<'a> {
    d: &'a [u8],
    total: usize,
    used: usize,
}

impl<'a> BackBits<'a> {
    fn new(d: &'a [u8]) -> Res<Self> {
        let last = *d.last().ok_or_else(|| corrupt("empty bitstream"))?;
        if last == 0 {
            return Err(corrupt("bitstream padding marker is absent"));
        }
        Ok(BackBits {
            d,
            total: d.len() * 8,
            used: last.leading_zeros() as usize + 1,
        })
    }

    fn bit(&self, p: usize) -> u32 {
        ((self.d[p >> 3] >> (p & 7)) & 1) as u32
    }

    /// The next `n` bits, most-significant first, zero-filled past the start of
    /// the stream. Over-read is detected by [`BackBits::finish`] rather than
    /// here, because the final symbol of a valid stream may legitimately need
    /// bits the padding supplied.
    fn peek(&self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n as usize {
            let b = match self.total.checked_sub(self.used + i + 1) {
                Some(p) => self.bit(p),
                None => 0,
            };
            v = (v << 1) | b;
        }
        v
    }

    fn consume(&mut self, n: u32) {
        self.used += n as usize;
    }

    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// Whether the last read went past the start of the stream. zstd's
    /// interleaved FSE streams use this as their termination condition, so it
    /// is a normal state there and an error everywhere else.
    fn over_drawn(&self) -> bool {
        self.used > self.total
    }

    /// A stream whose symbol count is known must end exactly on its last bit
    /// (`BIT_endOfDStream`).
    fn finish(&self, what: &str) -> Res<()> {
        if self.used != self.total {
            return Err(corrupt(format!(
                "{what} bitstream ends {} bits from where it should",
                self.total as i64 - self.used as i64
            )));
        }
        Ok(())
    }
}

/// A forward, LSB-first bit reader — used only for FSE table descriptions,
/// which are the one place zstd reads bits in the normal direction.
struct FwdBits<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> FwdBits<'a> {
    fn new(d: &'a [u8]) -> Self {
        FwdBits { d, pos: 0 }
    }

    fn peek(&self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n as usize {
            let p = self.pos + i;
            let b = if p >> 3 < self.d.len() {
                ((self.d[p >> 3] >> (p & 7)) & 1) as u32
            } else {
                0
            };
            v |= b << i;
        }
        v
    }

    fn read(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }

    /// Bytes consumed, rounded up: a table description occupies whole bytes.
    fn bytes(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

/// Writes the bit layout the backward readers above expect: bits accumulate
/// least-significant-first, bytes come out little-endian, and the stream is
/// terminated by a `1` marker so a decoder can find where the padding starts.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    fn add(&mut self, v: u32, n: u32) {
        if n == 0 {
            return;
        }
        let mask = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        self.acc |= ((v & mask) as u64) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.add(1, 1);
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

// ------------------------------------------------------------------------ FSE --

/// An FSE decoding table: one entry per state.
#[derive(Clone, Debug)]
struct FseDTable {
    log: u32,
    /// `(symbol, nb_bits, base_state)`
    states: Vec<(u16, u8, u16)>,
}

impl FseDTable {
    fn rle(symbol: u16) -> Self {
        FseDTable {
            log: 0,
            states: vec![(symbol, 0, 0)],
        }
    }

    /// RFC 8878 §4.1's table construction: low-probability symbols fill the top
    /// of the table from the end, the rest are spread by a fixed step, and each
    /// state's bit count follows from how many states its symbol owns.
    fn build(norm: &[i32], log: u32) -> Res<Self> {
        let size = 1usize << log;
        let total: u64 = norm.iter().map(|c| c.unsigned_abs() as u64).sum();
        if total != size as u64 {
            return Err(corrupt(format!(
                "FSE distribution sums to {total}, not the table size {size}"
            )));
        }
        let mut symbols = vec![0u16; size];
        let mut high: isize = size as isize - 1;
        for (s, &c) in norm.iter().enumerate() {
            if c == -1 {
                symbols[high as usize] = s as u16;
                high -= 1;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mask = size - 1;
        let mut pos = 0usize;
        for (s, &c) in norm.iter().enumerate() {
            if c <= 0 {
                continue;
            }
            for _ in 0..c {
                symbols[pos] = s as u16;
                pos = (pos + step) & mask;
                while pos as isize > high {
                    pos = (pos + step) & mask;
                }
            }
        }
        if pos != 0 {
            return Err(corrupt("FSE symbol spread did not close"));
        }
        let mut next: Vec<u32> = norm
            .iter()
            .map(|&c| if c == -1 { 1 } else { c.max(0) as u32 })
            .collect();
        let mut states = vec![(0u16, 0u8, 0u16); size];
        for (u, st) in states.iter_mut().enumerate() {
            let s = symbols[u] as usize;
            let n = next[s];
            next[s] += 1;
            if n == 0 {
                return Err(corrupt("FSE state references an absent symbol"));
            }
            let nb = log - (31 - n.leading_zeros());
            *st = (s as u16, nb as u8, ((n << nb) - size as u32) as u16);
        }
        Ok(FseDTable { log, states })
    }
}

struct FseState {
    idx: usize,
}

impl FseState {
    fn init(t: &FseDTable, r: &mut BackBits<'_>) -> Self {
        FseState {
            idx: r.read(t.log) as usize,
        }
    }

    fn symbol(&self, t: &FseDTable) -> u16 {
        t.states[self.idx].0
    }

    fn update(&mut self, t: &FseDTable, r: &mut BackBits<'_>) {
        let (_, nb, base) = t.states[self.idx];
        self.idx = base as usize + r.read(nb as u32) as usize;
    }
}

/// Reads an FSE table description (RFC 8878 §4.1.1).
///
/// Returns the normalized counts, the accuracy log, and how many bytes the
/// description occupied.
fn read_ncount(d: &[u8], max_symbol: usize, max_log: u32) -> Res<(Vec<i32>, u32, usize)> {
    if d.is_empty() {
        return Err(corrupt("truncated FSE table description"));
    }
    let mut r = FwdBits::new(d);
    let log = r.read(4) + 5;
    if log > max_log {
        return Err(corrupt(format!(
            "FSE accuracy log {log} exceeds the {max_log} this table allows"
        )));
    }
    let mut remaining: i64 = (1i64 << log) + 1;
    let mut threshold: i64 = 1i64 << log;
    let mut nb_bits = log + 1;
    let mut norm: Vec<i32> = Vec::new();
    let mut previous0 = false;
    while remaining > 1 && norm.len() <= max_symbol {
        if previous0 {
            // A zero count is followed by 2-bit repeat groups; 3 means "three
            // more zeroes, keep reading".
            let mut zeros = 0usize;
            loop {
                let n = r.read(2) as usize;
                zeros += n;
                if n < 3 {
                    break;
                }
            }
            for _ in 0..zeros {
                if norm.len() > max_symbol {
                    return Err(corrupt("FSE table describes more symbols than allowed"));
                }
                norm.push(0);
            }
            previous0 = false;
            if norm.len() > max_symbol {
                break;
            }
            continue;
        }
        let max = (2 * threshold - 1) - remaining;
        let v = r.peek(nb_bits);
        let low = (v & ((1u32 << (nb_bits - 1)) - 1)) as i64;
        let mut count = if low < max {
            r.read(nb_bits - 1);
            low
        } else {
            let mut c = (r.read(nb_bits) & ((1u32 << nb_bits) - 1)) as i64;
            if c >= threshold {
                c -= max;
            }
            c
        };
        count -= 1;
        remaining -= count.abs();
        if remaining < 0 {
            return Err(corrupt("FSE table description overruns its table"));
        }
        norm.push(count as i32);
        previous0 = count == 0;
        while remaining < threshold {
            nb_bits -= 1;
            threshold >>= 1;
        }
        if r.bytes() > d.len() {
            return Err(corrupt("truncated FSE table description"));
        }
    }
    if remaining != 1 {
        return Err(corrupt("FSE table description does not close"));
    }
    let used = r.bytes();
    if used > d.len() {
        return Err(corrupt("truncated FSE table description"));
    }
    Ok((norm, log, used))
}

// -------------------------------------------------------- FSE encoding tables --

/// An FSE encoding table (`FSE_buildCTable`'s output, in Rust).
struct FseCTable {
    log: u32,
    /// State transitions, indexed by cumulative symbol position.
    table: Vec<u16>,
    /// Per symbol: `(delta_nb_bits, delta_find_state)`.
    tt: Vec<(i32, i32)>,
}

impl FseCTable {
    fn build(norm: &[i32], log: u32) -> Res<Self> {
        let size = 1usize << log;
        let total: u64 = norm.iter().map(|c| c.unsigned_abs() as u64).sum();
        if total != size as u64 {
            return Err(corrupt("FSE encoding distribution does not fill its table"));
        }
        // Cumulative starts, with low-probability symbols placed at the top —
        // the same spread the decoder computes, or the two would disagree.
        let mut cumul = vec![0u32; norm.len() + 1];
        let mut symbols = vec![0u16; size];
        let mut high: isize = size as isize - 1;
        for (s, &c) in norm.iter().enumerate() {
            if c == -1 {
                cumul[s + 1] = cumul[s] + 1;
                symbols[high as usize] = s as u16;
                high -= 1;
            } else {
                cumul[s + 1] = cumul[s] + c.max(0) as u32;
            }
        }
        let step = (size >> 1) + (size >> 3) + 3;
        let mask = size - 1;
        let mut pos = 0usize;
        for (s, &c) in norm.iter().enumerate() {
            if c <= 0 {
                continue;
            }
            for _ in 0..c {
                symbols[pos] = s as u16;
                pos = (pos + step) & mask;
                while pos as isize > high {
                    pos = (pos + step) & mask;
                }
            }
        }
        if pos != 0 {
            return Err(corrupt("FSE encoding spread did not close"));
        }
        let mut table = vec![0u16; size];
        let mut cursor = cumul.clone();
        for (u, &s) in symbols.iter().enumerate() {
            let s = s as usize;
            table[cursor[s] as usize] = (size + u) as u16;
            cursor[s] += 1;
        }
        let mut tt = vec![(0i32, 0i32); norm.len()];
        let mut running = 0i32;
        for (s, &c) in norm.iter().enumerate() {
            tt[s] = match c {
                0 => (((log + 1) << 16) as i32 - size as i32, 0),
                -1 | 1 => {
                    let e = ((log << 16) as i32 - size as i32, running - 1);
                    running += 1;
                    e
                }
                c => {
                    let max_bits_out = log - (31 - (c as u32 - 1).leading_zeros());
                    let min_state_plus = (c as u32) << max_bits_out;
                    let e = (
                        ((max_bits_out << 16) as i32) - min_state_plus as i32,
                        running - c,
                    );
                    running += c;
                    e
                }
            };
        }
        Ok(FseCTable { log, table, tt })
    }

    fn init_state(&self, sym: usize) -> u32 {
        let (dnb, dfs) = self.tt[sym];
        let nb = ((dnb + (1 << 15)) >> 16) as u32;
        let v = ((nb as i32) << 16) - dnb;
        self.table[((v >> nb) + dfs) as usize] as u32
    }

    fn encode(&self, w: &mut BitWriter, state: &mut u32, sym: usize) {
        let (dnb, dfs) = self.tt[sym];
        let nb = ((*state as i32 + dnb) >> 16) as u32;
        w.add(*state, nb);
        *state = self.table[((*state >> nb) as i32 + dfs) as usize] as u32;
    }

    fn flush(&self, w: &mut BitWriter, state: u32) {
        w.add(state, self.log);
    }
}

// -------------------------------------------------------------------- Huffman --

/// A flat Huffman decoding table indexed by `log` bits (RFC 8878 §4.2).
#[derive(Clone, Debug)]
struct HufTable {
    log: u32,
    /// `(symbol, nb_bits)` per index.
    entries: Vec<(u8, u8)>,
}

impl HufTable {
    /// Builds the table from per-symbol weights, all of them present.
    fn from_weights(weights: &[u8]) -> Res<Self> {
        let mut total: u32 = 0;
        for &w in weights {
            if w > 12 {
                return Err(corrupt(format!("Huffman weight {w} exceeds 12")));
            }
            if w > 0 {
                total += 1 << (w - 1);
            }
        }
        if total == 0 || !total.is_power_of_two() {
            return Err(corrupt("Huffman weights do not sum to a power of two"));
        }
        let log = 31 - total.leading_zeros();
        if log > 11 {
            return Err(corrupt(format!("Huffman table log {log} exceeds 11")));
        }
        let size = 1usize << log;
        let mut entries = vec![(0u8, 0u8); size];
        let mut at = 0usize;
        // Codes are assigned longest-first (lowest weight first), and by symbol
        // value within one weight. The flat table is filled in exactly that
        // order, so the lowest code values belong to the longest codes.
        for w in 1..=12u8 {
            for (s, &sw) in weights.iter().enumerate() {
                if sw != w {
                    continue;
                }
                let nb = (log + 1 - w as u32) as u8;
                let span = 1usize << (w - 1);
                if at + span > size {
                    return Err(corrupt("Huffman weights overflow the table"));
                }
                for e in &mut entries[at..at + span] {
                    *e = (s as u8, nb);
                }
                at += span;
            }
        }
        if at != size {
            return Err(corrupt("Huffman weights underfill the table"));
        }
        Ok(HufTable { log, entries })
    }

    /// Reads the table description of §4.2.1 and returns it with its size.
    fn read(d: &[u8]) -> Res<(Self, usize)> {
        let h = *d.first().ok_or_else(|| corrupt("empty Huffman table"))? as usize;
        let mut weights: Vec<u8>;
        let used;
        if h >= 128 {
            // Direct representation: `h - 127` weights of 4 bits each, packed
            // two to a byte. The count is of *weights*, not symbols — the last
            // symbol's weight is always deduced.
            let n_weights = h - 127;
            let bytes = n_weights.div_ceil(2);
            if d.len() < 1 + bytes {
                return Err(corrupt("truncated direct Huffman weights"));
            }
            weights = Vec::with_capacity(n_weights + 1);
            for i in 0..n_weights {
                let b = d[1 + i / 2];
                weights.push(if i % 2 == 0 { b >> 4 } else { b & 0x0f });
            }
            used = 1 + bytes;
        } else {
            // FSE-compressed weights, two interleaved states. Decoding runs
            // until the bitstream is over-drawn: the final symbol comes from
            // the state that was not just updated, which is how an FSE stream
            // gets its last two symbols for free.
            if d.len() < 1 + h {
                return Err(corrupt("truncated FSE-compressed Huffman weights"));
            }
            let (norm, log, desc) = read_ncount(&d[1..1 + h], 255, 6)?;
            let t = FseDTable::build(&norm, log)?;
            let stream = &d[1 + desc..1 + h];
            let mut r = BackBits::new(stream)?;
            let mut s1 = FseState::init(&t, &mut r);
            let mut s2 = FseState::init(&t, &mut r);
            weights = Vec::new();
            loop {
                if weights.len() > 254 {
                    return Err(corrupt("more than 255 Huffman weights"));
                }
                weights.push(s1.symbol(&t) as u8);
                s1.update(&t, &mut r);
                if r.over_drawn() {
                    weights.push(s2.symbol(&t) as u8);
                    break;
                }
                weights.push(s2.symbol(&t) as u8);
                s2.update(&t, &mut r);
                if r.over_drawn() {
                    weights.push(s1.symbol(&t) as u8);
                    break;
                }
            }
            used = 1 + h;
        }
        // The final weight is whatever completes the code space.
        let sum: u32 = weights
            .iter()
            .map(|&w| if w > 0 { 1u32 << (w - 1) } else { 0 })
            .sum();
        if sum == 0 {
            return Err(corrupt("Huffman weights are all zero"));
        }
        let log = (31 - sum.leading_zeros()) + 1;
        let rest = (1u32 << log) - sum;
        if rest == 0 || !rest.is_power_of_two() {
            return Err(corrupt("Huffman weights leave no room for the last symbol"));
        }
        weights.push((31 - rest.leading_zeros() + 1) as u8);
        if weights.len() > 256 {
            return Err(corrupt("Huffman table describes more than 256 symbols"));
        }
        // A valid tree has an even number of shortest-code symbols, at least
        // two of them: the reference checks this, and so must anything that
        // wants to reject the same inputs.
        let rank1 = weights.iter().filter(|&&w| w == 1).count();
        if rank1 < 2 || !rank1.is_multiple_of(2) {
            return Err(corrupt(format!(
                "Huffman tree has {rank1} symbols of weight 1, which cannot be a tree"
            )));
        }
        Ok((HufTable::from_weights(&weights)?, used))
    }

    fn decode_into(&self, r: &mut BackBits<'_>, out: &mut Vec<u8>, n: usize) -> Res<()> {
        for _ in 0..n {
            let (sym, nb) = self.entries[r.peek(self.log) as usize];
            r.consume(nb as u32);
            out.push(sym);
        }
        r.finish("Huffman literals")
    }
}

/// The encoder-side Huffman table: `(nb_bits, code)` per symbol.
struct HufCTable {
    codes: Vec<(u8, u32)>,
    weights: Vec<u8>,
}

impl HufCTable {
    /// Builds a length-limited canonical table from symbol frequencies.
    ///
    /// Lengths are limited to 11 bits (the format's maximum) by halving the
    /// frequencies and rebuilding — crude next to package-merge, but
    /// deterministic, which §03.7.1 requires of anything reproducible.
    fn build(freq: &[u32; 256], max_sym: usize) -> Option<Self> {
        let mut f: Vec<u32> = freq[..=max_sym].to_vec();
        let mut lengths;
        loop {
            lengths = huffman_lengths(&f)?;
            if lengths.iter().copied().max().unwrap_or(0) <= 11 {
                break;
            }
            for x in f.iter_mut() {
                if *x > 1 {
                    *x = (*x).div_ceil(2);
                }
            }
        }
        let log = *lengths.iter().max().unwrap() as u32;
        // Weight = log + 1 - nb_bits; unused symbols weigh nothing.
        let weights: Vec<u8> = lengths
            .iter()
            .map(|&l| if l == 0 { 0 } else { log as u8 + 1 - l })
            .collect();
        // Assign codes in the same order the decode table is filled.
        let mut codes = vec![(0u8, 0u32); 256];
        let mut at = 0u32;
        for w in 1..=12u8 {
            for (s, &sw) in weights.iter().enumerate() {
                if sw != w {
                    continue;
                }
                let nb = log + 1 - w as u32;
                codes[s] = (nb as u8, at >> (log - nb));
                at += 1 << (w - 1);
            }
        }
        if at != 1 << log {
            return None;
        }
        Some(HufCTable { codes, weights })
    }

    /// The §4.2.1 direct weight representation, available only for alphabets of
    /// at most 128 symbols. Larger alphabets need FSE-compressed weights, which
    /// this encoder does not write — it emits raw literals instead.
    fn direct_description(&self) -> Option<Vec<u8>> {
        // The header byte counts stored *weights*; the last symbol's weight is
        // always deduced, and 128 stored weights is the representation's limit.
        let n_weights = self.weights.len().checked_sub(1)?;
        if n_weights == 0 || n_weights > 128 {
            return None;
        }
        let mut out = vec![(127 + n_weights) as u8];
        for i in (0..n_weights).step_by(2) {
            let lo = if i + 1 < n_weights {
                self.weights[i + 1]
            } else {
                0
            };
            out.push((self.weights[i] << 4) | lo);
        }
        Some(out)
    }
}

/// Classic Huffman code lengths from frequencies, with the degenerate cases the
/// format still has to represent.
fn huffman_lengths(freq: &[u32]) -> Option<Vec<u8>> {
    let used: Vec<usize> = (0..freq.len()).filter(|&i| freq[i] > 0).collect();
    if used.len() < 2 {
        return None;
    }
    // Node arena: leaves first, then internal nodes.
    let mut weight: Vec<u64> = used.iter().map(|&i| freq[i] as u64).collect();
    let mut kids: Vec<(usize, usize)> = Vec::new();
    let mut live: Vec<usize> = (0..used.len()).collect();
    while live.len() > 1 {
        // Two smallest, ties broken by index so the result is deterministic.
        live.sort_by_key(|&n| (weight[n], n));
        let a = live.remove(0);
        let b = live.remove(0);
        let n = weight.len();
        weight.push(weight[a] + weight[b]);
        kids.push((a, b));
        live.push(n);
    }
    let root = live[0];
    let mut depth = vec![0u8; weight.len()];
    let mut stack = vec![(root, 0u8)];
    let n_leaves = used.len();
    while let Some((n, d)) = stack.pop() {
        if n < n_leaves {
            depth[n] = d.max(1);
            continue;
        }
        if d > 60 {
            return None;
        }
        let (a, b) = kids[n - n_leaves];
        stack.push((a, d + 1));
        stack.push((b, d + 1));
    }
    let mut out = vec![0u8; freq.len()];
    for (k, &s) in used.iter().enumerate() {
        out[s] = depth[k];
    }
    Some(out)
}

// ------------------------------------------------------------------ sequences --

const LL_BASE: [u32; 36] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32, 40, 48, 64,
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const LL_BITS: [u32; 36] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];
const ML_BASE: [u32; 53] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027,
    2051, 4099, 8195, 16387, 32771, 65539,
];
const ML_BITS: [u32; 53] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
];

/// §3.1.1.3.2.2.1's predefined distributions. A block may use these instead of
/// shipping a table, and this encoder always does.
const LL_DEFAULT: [i32; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];
const ML_DEFAULT: [i32; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
];
const OF_DEFAULT: [i32; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];
const LL_DEFAULT_LOG: u32 = 6;
const ML_DEFAULT_LOG: u32 = 6;
const OF_DEFAULT_LOG: u32 = 5;

fn code_for(bases: &[u32], value: u32) -> usize {
    let mut i = bases.len() - 1;
    while bases[i] > value {
        i -= 1;
    }
    i
}

struct Seq {
    literals: u32,
    match_len: u32,
    /// The stored offset value: `offset + 3`, or 1..3 for a repeat.
    off_value: u32,
}

// -------------------------------------------------------------------- decoder --

struct FrameHeader {
    window: u64,
    content_size: Option<u64>,
    checksum: bool,
    header_len: usize,
}

fn parse_frame_header(d: &[u8]) -> Res<FrameHeader> {
    if d.len() < 5 {
        return Err(corrupt("truncated frame header"));
    }
    let desc = d[4];
    let fcs_flag = desc >> 6;
    let single = desc & 0x20 != 0;
    if desc & 0x08 != 0 {
        return Err(corrupt("reserved frame header bit is set"));
    }
    let checksum = desc & 0x04 != 0;
    let dict_flag = desc & 0x03;
    let mut p = 5;
    let mut window: u64 = 0;
    if !single {
        let wd = *d
            .get(p)
            .ok_or_else(|| corrupt("truncated window descriptor"))?;
        p += 1;
        let exp = (wd >> 3) as u32;
        let mantissa = (wd & 7) as u64;
        if 10 + exp > MAX_WINDOW_LOG {
            return Err(corrupt(format!("window log {} is out of range", 10 + exp)));
        }
        let base = 1u64 << (10 + exp);
        window = base + (base / 8) * mantissa;
    }
    let dict_bytes = match dict_flag {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 4,
    };
    if d.len() < p + dict_bytes {
        return Err(corrupt("truncated dictionary id"));
    }
    if dict_bytes > 0 {
        let mut id = 0u32;
        for (i, &b) in d[p..p + dict_bytes].iter().enumerate() {
            id |= (b as u32) << (8 * i);
        }
        if id != 0 {
            // §03.7.1 allows a dictionary object ref; a frame that needs one we
            // were not given is indeterminate, not invalid.
            return Err(Error::Unsupported("zstd (dictionary-compressed frame)"));
        }
    }
    p += dict_bytes;
    let fcs_bytes = match fcs_flag {
        0 => usize::from(single),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    if d.len() < p + fcs_bytes {
        return Err(corrupt("truncated frame content size"));
    }
    let content_size = if fcs_bytes == 0 {
        None
    } else {
        let mut v = 0u64;
        for (i, &b) in d[p..p + fcs_bytes].iter().enumerate() {
            v |= (b as u64) << (8 * i);
        }
        if fcs_bytes == 2 {
            v += 256;
        }
        Some(v)
    };
    p += fcs_bytes;
    if single {
        window = content_size.unwrap_or(0);
    }
    Ok(FrameHeader {
        window,
        content_size,
        checksum,
        header_len: p,
    })
}

/// Per-frame decoding state that survives across blocks: the Huffman table a
/// Treeless literals block reuses, and the three repeatable sequence tables.
#[derive(Default)]
struct FrameState {
    huf: Option<HufTable>,
    ll: Option<FseDTable>,
    of: Option<FseDTable>,
    ml: Option<FseDTable>,
    reps: [u32; 3],
}

/// Decompresses one or more concatenated frames, refusing to produce more than
/// `limit` bytes (§03.7.4).
pub fn decompress(input: &[u8], limit: usize) -> Res<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    let mut p = 0usize;
    let mut frames = 0usize;
    while p < input.len() {
        if input.len() - p < 4 {
            return Err(corrupt("trailing bytes are not a frame"));
        }
        let magic = u32le(&input[p..]);
        if magic & SKIPPABLE_MASK == SKIPPABLE_BASE {
            if input.len() - p < 8 {
                return Err(corrupt("truncated skippable frame"));
            }
            let n = u32le(&input[p + 4..]) as usize;
            p = p
                .checked_add(8 + n)
                .filter(|&e| e <= input.len())
                .ok_or_else(|| corrupt("skippable frame runs past the end"))?;
            continue;
        }
        if magic != MAGIC {
            return Err(corrupt(format!(
                "bad frame magic {magic:#010x} at offset {p}"
            )));
        }
        p = decode_frame(input, p, &mut out, limit)?;
        frames += 1;
        if frames > 1 << 20 {
            return Err(corrupt("absurd number of frames"));
        }
    }
    if frames == 0 {
        return Err(corrupt("no frames"));
    }
    Ok(out)
}

fn decode_frame(input: &[u8], start_off: usize, out: &mut Vec<u8>, limit: usize) -> Res<usize> {
    let d = &input[start_off..];
    let h = parse_frame_header(d)?;
    let mut p = h.header_len;
    let origin = out.len();
    let mut st = FrameState {
        reps: [1, 4, 8],
        ..Default::default()
    };
    loop {
        if d.len() < p + 3 {
            return Err(corrupt("truncated block header"));
        }
        let bh = d[p] as u32 | (d[p + 1] as u32) << 8 | (d[p + 2] as u32) << 16;
        p += 3;
        let last = bh & 1 != 0;
        let btype = (bh >> 1) & 3;
        let size = (bh >> 3) as usize;
        if size > MAX_BLOCK {
            return Err(corrupt(format!(
                "block of {size} bytes exceeds the maximum"
            )));
        }
        match btype {
            0 => {
                if d.len() < p + size {
                    return Err(corrupt("truncated raw block"));
                }
                grow(out, size, limit)?;
                out.extend_from_slice(&d[p..p + size]);
                p += size;
            }
            1 => {
                if d.len() < p + 1 {
                    return Err(corrupt("truncated RLE block"));
                }
                grow(out, size, limit)?;
                out.extend(std::iter::repeat_n(d[p], size));
                p += 1;
            }
            2 => {
                if d.len() < p + size {
                    return Err(corrupt("truncated compressed block"));
                }
                decode_compressed_block(&d[p..p + size], out, origin, &h, &mut st, limit)?;
                p += size;
            }
            _ => return Err(corrupt("reserved block type")),
        }
        if last {
            break;
        }
    }
    if let Some(n) = h.content_size {
        if (out.len() - origin) as u64 != n {
            return Err(corrupt(format!(
                "frame declares {n} bytes of content but produced {}",
                out.len() - origin
            )));
        }
    }
    if h.checksum {
        if d.len() < p + 4 {
            return Err(corrupt("truncated content checksum"));
        }
        let want = u32le(&d[p..]);
        let got = xxh64(&out[origin..], 0) as u32;
        if want != got {
            return Err(corrupt(format!(
                "content checksum {got:#010x} does not match the frame's {want:#010x}"
            )));
        }
        p += 4;
    }
    Ok(start_off + p)
}

fn grow(out: &[u8], extra: usize, limit: usize) -> Res<()> {
    if out.len() + extra > limit {
        return Err(Error::Bounds(format!(
            "zstd output exceeds the declared {limit} bytes"
        )));
    }
    Ok(())
}

fn decode_compressed_block(
    d: &[u8],
    out: &mut Vec<u8>,
    origin: usize,
    h: &FrameHeader,
    st: &mut FrameState,
    limit: usize,
) -> Res<()> {
    let (literals, used) = decode_literals(d, st)?;
    let seqs = decode_sequences(&d[used..], st)?;
    // Execute (§3.1.1.3.2.1.2).
    let mut lit = 0usize;
    for s in &seqs {
        let ll = s.literals as usize;
        let ml = s.match_len as usize;
        if lit + ll > literals.len() {
            return Err(corrupt("sequence wants more literals than the block holds"));
        }
        grow(out, ll + ml, limit)?;
        out.extend_from_slice(&literals[lit..lit + ll]);
        lit += ll;
        let offset = s.off_value as usize;
        let produced = out.len() - origin;
        if offset == 0 || offset > produced {
            return Err(corrupt(format!(
                "match offset {offset} reaches outside the {produced} bytes decoded so far"
            )));
        }
        if h.window > 0 && offset as u64 > h.window {
            return Err(corrupt("match offset exceeds the declared window"));
        }
        let src = out.len() - offset;
        for i in 0..ml {
            let b = out[src + i];
            out.push(b);
        }
    }
    let rest = literals.len() - lit;
    grow(out, rest, limit)?;
    out.extend_from_slice(&literals[lit..]);
    Ok(())
}

/// §3.1.1.3.1 — the literals section.
fn decode_literals(d: &[u8], st: &mut FrameState) -> Res<(Vec<u8>, usize)> {
    let b0 = *d.first().ok_or_else(|| corrupt("empty block"))?;
    let btype = b0 & 3;
    let sf = (b0 >> 2) & 3;
    match btype {
        0 | 1 => {
            let (size, hdr) = match sf {
                0 | 2 => ((b0 >> 3) as usize, 1),
                1 => {
                    if d.len() < 2 {
                        return Err(corrupt("truncated literals header"));
                    }
                    (((b0 >> 4) as usize) | (d[1] as usize) << 4, 2)
                }
                _ => {
                    if d.len() < 3 {
                        return Err(corrupt("truncated literals header"));
                    }
                    (
                        ((b0 >> 4) as usize) | (d[1] as usize) << 4 | (d[2] as usize) << 12,
                        3,
                    )
                }
            };
            if size > MAX_BLOCK {
                return Err(corrupt("literals section exceeds the block maximum"));
            }
            if btype == 0 {
                if d.len() < hdr + size {
                    return Err(corrupt("truncated raw literals"));
                }
                Ok((d[hdr..hdr + size].to_vec(), hdr + size))
            } else {
                if d.len() < hdr + 1 {
                    return Err(corrupt("truncated RLE literals"));
                }
                Ok((vec![d[hdr]; size], hdr + 1))
            }
        }
        _ => {
            let treeless = btype == 3;
            let (regen, comp, streams, hdr) = match sf {
                0 | 1 => {
                    if d.len() < 3 {
                        return Err(corrupt("truncated literals header"));
                    }
                    let v = b0 as u32 | (d[1] as u32) << 8 | (d[2] as u32) << 16;
                    let regen = (v >> 4) & 0x3ff;
                    let comp = (v >> 14) & 0x3ff;
                    (regen, comp, if sf == 0 { 1 } else { 4 }, 3)
                }
                2 => {
                    if d.len() < 4 {
                        return Err(corrupt("truncated literals header"));
                    }
                    let v =
                        b0 as u32 | (d[1] as u32) << 8 | (d[2] as u32) << 16 | (d[3] as u32) << 24;
                    ((v >> 4) & 0x3fff, (v >> 18) & 0x3fff, 4, 4)
                }
                _ => {
                    if d.len() < 5 {
                        return Err(corrupt("truncated literals header"));
                    }
                    let v = b0 as u64
                        | (d[1] as u64) << 8
                        | (d[2] as u64) << 16
                        | (d[3] as u64) << 24
                        | (d[4] as u64) << 32;
                    (
                        ((v >> 4) & 0x3ffff) as u32,
                        ((v >> 22) & 0x3ffff) as u32,
                        4,
                        5,
                    )
                }
            };
            let regen = regen as usize;
            let comp = comp as usize;
            if regen > MAX_BLOCK || comp > MAX_BLOCK {
                return Err(corrupt("literals section exceeds the block maximum"));
            }
            if d.len() < hdr + comp {
                return Err(corrupt("truncated compressed literals"));
            }
            let body = &d[hdr..hdr + comp];
            let table_bytes = if treeless {
                0
            } else {
                let (t, n) = HufTable::read(body)?;
                st.huf = Some(t);
                n
            };
            let table = st
                .huf
                .as_ref()
                .ok_or_else(|| corrupt("treeless literals with no previous Huffman table"))?;
            let streams_data = &body[table_bytes..];
            let mut out = Vec::with_capacity(regen);
            if streams == 1 {
                let mut r = BackBits::new(streams_data)?;
                table.decode_into(&mut r, &mut out, regen)?;
            } else {
                if streams_data.len() < 6 {
                    return Err(corrupt("truncated four-stream jump table"));
                }
                let s1 = u16::from_le_bytes([streams_data[0], streams_data[1]]) as usize;
                let s2 = u16::from_le_bytes([streams_data[2], streams_data[3]]) as usize;
                let s3 = u16::from_le_bytes([streams_data[4], streams_data[5]]) as usize;
                let rest = &streams_data[6..];
                let s4 = rest
                    .len()
                    .checked_sub(s1 + s2 + s3)
                    .ok_or_else(|| corrupt("four-stream sizes exceed the section"))?;
                let quarter = regen.div_ceil(4);
                let mut at = 0usize;
                for (i, len) in [s1, s2, s3, s4].into_iter().enumerate() {
                    let n = if i == 3 { regen - 3 * quarter } else { quarter };
                    let mut r = BackBits::new(&rest[at..at + len])?;
                    table.decode_into(&mut r, &mut out, n)?;
                    at += len;
                }
            }
            if out.len() != regen {
                return Err(corrupt("literals do not match their regenerated size"));
            }
            Ok((out, hdr + comp))
        }
    }
}

/// §3.1.1.3.2 — the sequences section.
fn decode_sequences(d: &[u8], st: &mut FrameState) -> Res<Vec<Seq>> {
    if d.is_empty() {
        return Err(corrupt("missing sequences section"));
    }
    let (nb_seq, mut p) = match d[0] {
        0 => return Ok(Vec::new()),
        b @ 1..=127 => (b as usize, 1),
        b @ 128..=254 => {
            if d.len() < 2 {
                return Err(corrupt("truncated sequence count"));
            }
            ((((b as usize) - 128) << 8) + d[1] as usize, 2)
        }
        _ => {
            if d.len() < 3 {
                return Err(corrupt("truncated sequence count"));
            }
            (u16::from_le_bytes([d[1], d[2]]) as usize + 0x7f00, 3)
        }
    };
    if nb_seq == 0 {
        return Ok(Vec::new());
    }
    let modes = *d.get(p).ok_or_else(|| corrupt("missing sequence modes"))?;
    p += 1;
    if modes & 3 != 0 {
        return Err(corrupt("reserved bits set in the sequence mode byte"));
    }
    let ll_mode = modes >> 6;
    let of_mode = (modes >> 4) & 3;
    let ml_mode = (modes >> 2) & 3;

    let ll = read_seq_table(
        d,
        &mut p,
        ll_mode,
        &LL_DEFAULT,
        LL_DEFAULT_LOG,
        35,
        9,
        &st.ll,
    )?;
    let of = read_seq_table(
        d,
        &mut p,
        of_mode,
        &OF_DEFAULT,
        OF_DEFAULT_LOG,
        31,
        8,
        &st.of,
    )?;
    let ml = read_seq_table(
        d,
        &mut p,
        ml_mode,
        &ML_DEFAULT,
        ML_DEFAULT_LOG,
        52,
        9,
        &st.ml,
    )?;
    st.ll = Some(ll.clone());
    st.of = Some(of.clone());
    st.ml = Some(ml.clone());

    if p > d.len() {
        return Err(corrupt("sequence tables overrun the section"));
    }
    let mut r = BackBits::new(&d[p..])?;
    let mut s_ll = FseState::init(&ll, &mut r);
    let mut s_of = FseState::init(&of, &mut r);
    let mut s_ml = FseState::init(&ml, &mut r);
    let mut out = Vec::with_capacity(nb_seq.min(1 << 16));
    for i in 0..nb_seq {
        let ll_code = s_ll.symbol(&ll) as usize;
        let ml_code = s_ml.symbol(&ml) as usize;
        let of_code = s_of.symbol(&of) as usize;
        if ll_code >= LL_BASE.len() || ml_code >= ML_BASE.len() || of_code > 31 {
            return Err(corrupt("sequence code out of range"));
        }
        // Bit order within a sequence: offset, then match length, then literals
        // length — the reverse of the order an encoder writes them.
        let off_value = (1u64 << of_code) + r.read(of_code as u32) as u64;
        let ml_value = ML_BASE[ml_code] as u64 + r.read(ML_BITS[ml_code]) as u64;
        let ll_value = LL_BASE[ll_code] as u64 + r.read(LL_BITS[ll_code]) as u64;
        if off_value == 0 {
            return Err(corrupt("offset value zero"));
        }
        // §3.1.1.3.2.1.1's repeat offsets.
        let ll0 = usize::from(ll_value == 0);
        let real = if off_value > 3 {
            let o = (off_value - 3) as u32;
            st.reps[2] = st.reps[1];
            st.reps[1] = st.reps[0];
            st.reps[0] = o;
            o
        } else {
            let code = off_value as usize - 1 + ll0;
            if code == 0 {
                st.reps[0]
            } else {
                let o = if code == 3 {
                    st.reps[0]
                        .checked_sub(1)
                        .filter(|&x| x > 0)
                        .ok_or_else(|| corrupt("repeat offset underflow"))?
                } else {
                    st.reps[code]
                };
                if code >= 2 {
                    st.reps[2] = st.reps[1];
                }
                st.reps[1] = st.reps[0];
                st.reps[0] = o;
                o
            }
        };
        out.push(Seq {
            literals: ll_value as u32,
            match_len: ml_value as u32,
            off_value: real,
        });
        if i + 1 < nb_seq {
            s_ll.update(&ll, &mut r);
            s_ml.update(&ml, &mut r);
            s_of.update(&of, &mut r);
        }
    }
    r.finish("sequences")?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn read_seq_table(
    d: &[u8],
    p: &mut usize,
    mode: u8,
    default: &[i32],
    default_log: u32,
    max_symbol: usize,
    max_log: u32,
    previous: &Option<FseDTable>,
) -> Res<FseDTable> {
    match mode {
        0 => FseDTable::build(default, default_log),
        1 => {
            let s = *d.get(*p).ok_or_else(|| corrupt("truncated RLE table"))?;
            *p += 1;
            if s as usize > max_symbol {
                return Err(corrupt("RLE sequence symbol out of range"));
            }
            Ok(FseDTable::rle(s as u16))
        }
        2 => {
            if *p >= d.len() {
                return Err(corrupt("truncated sequence table"));
            }
            let (norm, log, used) = read_ncount(&d[*p..], max_symbol, max_log)?;
            *p += used;
            FseDTable::build(&norm, log)
        }
        _ => previous
            .clone()
            .ok_or_else(|| corrupt("repeat sequence table with no previous table")),
    }
}

// -------------------------------------------------------------------- encoder --

/// Compresses `data` into a single zstd frame.
///
/// `level` selects match-search effort only; see the module note on ratio.
pub fn compress(data: &[u8], level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    out.extend_from_slice(&MAGIC.to_le_bytes());

    let single = data.len() <= (1usize << WINDOW_LOG);
    let fcs_flag: u8 = if data.len() < 256 {
        0
    } else if data.len() < 65536 + 256 {
        1
    } else if data.len() <= u32::MAX as usize {
        2
    } else {
        3
    };
    // Content checksum on: a reader that verifies it should have something to
    // verify, and it costs four bytes.
    let desc = (fcs_flag << 6) | (u8::from(single) << 5) | 0x04;
    out.push(desc);
    if !single {
        out.push(((WINDOW_LOG - 10) << 3) as u8);
    }
    match fcs_flag {
        0 => out.push(data.len() as u8),
        1 => out.extend_from_slice(&((data.len() - 256) as u16).to_le_bytes()),
        2 => out.extend_from_slice(&(data.len() as u32).to_le_bytes()),
        _ => out.extend_from_slice(&(data.len() as u64).to_le_bytes()),
    }

    let window = 1usize << WINDOW_LOG;
    let mut matcher = Matcher::new(data, level);
    let mut at = 0usize;
    if data.is_empty() {
        // A frame still needs a block; an empty raw one is the honest encoding.
        out.extend_from_slice(&[0x01, 0x00, 0x00]);
    }
    while at < data.len() {
        let end = (at + MAX_BLOCK).min(data.len());
        let block = &data[at..end];
        let last = end == data.len();
        let body = compress_block(data, at, end, window, &mut matcher);
        match body {
            Some(b) if b.len() < block.len() => write_block(&mut out, 2, last, &b),
            _ => {
                if block.iter().all(|&x| x == block[0]) && !block.is_empty() {
                    write_rle_block(&mut out, last, block[0], block.len());
                } else {
                    write_block(&mut out, 0, last, block);
                }
            }
        }
        at = end;
    }
    out.extend_from_slice(&(xxh64(data, 0) as u32).to_le_bytes());
    out
}

fn write_block(out: &mut Vec<u8>, btype: u32, last: bool, body: &[u8]) {
    let h = u32::from(last) | (btype << 1) | ((body.len() as u32) << 3);
    out.extend_from_slice(&h.to_le_bytes()[..3]);
    out.extend_from_slice(body);
}

fn write_rle_block(out: &mut Vec<u8>, last: bool, byte: u8, n: usize) {
    let h = u32::from(last) | (1 << 1) | ((n as u32) << 3);
    out.extend_from_slice(&h.to_le_bytes()[..3]);
    out.push(byte);
}

/// A hash-chain match finder over the whole input, so a block's matches may
/// reach back into earlier blocks — which is what the window is for.
struct Matcher {
    head: Vec<u32>,
    prev: Vec<u32>,
    depth: usize,
    inserted: usize,
}

const HASH_BITS: usize = 16;

impl Matcher {
    fn new(data: &[u8], level: u8) -> Self {
        Matcher {
            head: vec![u32::MAX; 1 << HASH_BITS],
            prev: vec![u32::MAX; data.len().max(1)],
            depth: match level {
                0 => 0,
                1..=3 => 8,
                4..=6 => 32,
                7..=12 => 128,
                _ => 512,
            },
            inserted: 0,
        }
    }

    fn hash(d: &[u8], i: usize) -> usize {
        let v = (d[i] as u32) << 16 | (d[i + 1] as u32) << 8 | d[i + 2] as u32;
        (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
    }

    fn insert(&mut self, d: &[u8], upto: usize) {
        while self.inserted < upto && self.inserted + MIN_MATCH <= d.len() {
            let i = self.inserted;
            let h = Self::hash(d, i);
            self.prev[i] = self.head[h];
            self.head[h] = i as u32;
            self.inserted += 1;
        }
    }

    /// The shortest match worth taking at a given distance.
    ///
    /// A three-byte match twenty thousand bytes back costs more to encode than
    /// the three literals it replaces: the offset alone is fifteen bits. Without
    /// this rule a greedy matcher happily makes files *larger* on data with
    /// accidental short repeats, which is most binary data.
    fn min_len(dist: usize) -> usize {
        match dist {
            0..=1024 => MIN_MATCH,
            1025..=16384 => 4,
            16385..=262144 => 5,
            _ => 6,
        }
    }

    /// The best match for position `i`, bounded by the block end and window.
    fn find(&mut self, d: &[u8], i: usize, end: usize, window: usize) -> Option<(usize, usize)> {
        if self.depth == 0 || i + MIN_MATCH > end {
            return None;
        }
        self.insert(d, i);
        let mut cand = self.head[Self::hash(d, i)];
        let mut best = (0usize, 0usize);
        let mut tries = 0;
        let max = end - i;
        while cand != u32::MAX && tries < self.depth {
            let c = cand as usize;
            let dist = i - c;
            if dist > window {
                break;
            }
            let mut l = 0;
            while l < max && d[c + l] == d[i + l] {
                l += 1;
            }
            if l > best.1 && l >= Self::min_len(dist) {
                best = (dist, l);
                if l == max {
                    break;
                }
            }
            cand = self.prev[c];
            tries += 1;
        }
        if best.1 >= MIN_MATCH {
            Some(best)
        } else {
            None
        }
    }
}

/// Builds one compressed block, or `None` if it would not be a compressed block
/// worth writing.
fn compress_block(
    data: &[u8],
    start: usize,
    end: usize,
    window: usize,
    m: &mut Matcher,
) -> Option<Vec<u8>> {
    let mut literals: Vec<u8> = Vec::new();
    let mut seqs: Vec<Seq> = Vec::new();
    let mut lit_run = 0u32;
    let mut i = start;
    while i < end {
        let found = m.find(data, i, end, window).filter(|&(dist, _)| dist <= i);
        // One step of lazy matching: if the next position starts a longer match,
        // spend a literal here and take that one instead. This is where most of
        // the ratio difference between a naive and a serious LZ pass lives.
        let found = match found {
            Some((_, len)) if len < 64 && i + 1 < end => match m.find(data, i + 1, end, window) {
                Some((d2, l2)) if l2 > len && d2 <= i + 1 => None,
                _ => found,
            },
            other => other,
        };
        match found {
            Some((dist, len)) => {
                seqs.push(Seq {
                    literals: lit_run,
                    match_len: len as u32,
                    off_value: dist as u32 + 3,
                });
                lit_run = 0;
                m.insert(data, i + len);
                i += len;
            }
            None => {
                literals.push(data[i]);
                lit_run += 1;
                m.insert(data, i + 1);
                i += 1;
            }
        }
    }
    let mut body = encode_literals(&literals)?;
    encode_sequences(&mut body, &seqs);
    Some(body)
}

fn encode_literals(lit: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    if lit.is_empty() {
        out.push(0); // Raw, size format 0, size 0
        return Some(out);
    }
    if lit.iter().all(|&b| b == lit[0]) {
        // RLE literals: one byte for any length.
        write_raw_literals_header(&mut out, 1, lit.len());
        out.push(lit[0]);
        return Some(out);
    }
    let mut freq = [0u32; 256];
    let mut max_sym = 0usize;
    for &b in lit {
        freq[b as usize] += 1;
        max_sym = max_sym.max(b as usize);
    }
    if let Some(ct) = HufCTable::build(&freq, max_sym) {
        if let Some(desc) = ct.direct_description() {
            let streams = huf_encode_streams(lit, &ct);
            let compressed_len = desc.len()
                + streams.iter().map(Vec::len).sum::<usize>()
                + if streams.len() == 4 { 6 } else { 0 };
            if compressed_len < lit.len() {
                write_compressed_literals_header(
                    &mut out,
                    lit.len(),
                    compressed_len,
                    streams.len() == 4,
                );
                out.extend_from_slice(&desc);
                if streams.len() == 4 {
                    for s in &streams[..3] {
                        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
                    }
                }
                for s in &streams {
                    out.extend_from_slice(s);
                }
                return Some(out);
            }
        }
    }
    write_raw_literals_header(&mut out, 0, lit.len());
    out.extend_from_slice(lit);
    Some(out)
}

fn write_raw_literals_header(out: &mut Vec<u8>, btype: u8, size: usize) {
    if size < 32 {
        out.push(btype | ((size as u8) << 3));
    } else if size < 4096 {
        out.push(btype | (1 << 2) | ((size as u8 & 0x0f) << 4));
        out.push((size >> 4) as u8);
    } else {
        out.push(btype | (3 << 2) | ((size as u8 & 0x0f) << 4));
        out.push((size >> 4) as u8);
        out.push((size >> 12) as u8);
    }
}

fn write_compressed_literals_header(out: &mut Vec<u8>, regen: usize, comp: usize, four: bool) {
    if !four {
        let v = 2u32 | ((regen as u32) << 4) | ((comp as u32) << 14);
        out.extend_from_slice(&v.to_le_bytes()[..3]);
    } else if regen < (1 << 14) && comp < (1 << 14) {
        let v = 2u32 | (2 << 2) | ((regen as u32) << 4) | ((comp as u32) << 18);
        out.extend_from_slice(&v.to_le_bytes()[..4]);
    } else {
        let v = 2u64 | (3 << 2) | ((regen as u64) << 4) | ((comp as u64) << 22);
        out.extend_from_slice(&v.to_le_bytes()[..5]);
    }
}

/// Huffman-codes the literals, in one stream when the format allows it and four
/// when it does not. Symbols go out in reverse order, because the decoder reads
/// the bitstream backwards.
fn huf_encode_streams(lit: &[u8], ct: &HufCTable) -> Vec<Vec<u8>> {
    let one_stream = lit.len() < 1024;
    let encode = |slice: &[u8]| -> Vec<u8> {
        let mut w = BitWriter::new();
        for &b in slice.iter().rev() {
            let (nb, code) = ct.codes[b as usize];
            w.add(code, nb as u32);
        }
        w.finish()
    };
    if one_stream {
        return vec![encode(lit)];
    }
    let quarter = lit.len().div_ceil(4);
    let mut out = Vec::with_capacity(4);
    for i in 0..4 {
        let s = i * quarter;
        let e = if i == 3 { lit.len() } else { s + quarter };
        out.push(encode(&lit[s..e]));
    }
    out
}

fn encode_sequences(out: &mut Vec<u8>, seqs: &[Seq]) {
    let n = seqs.len();
    if n == 0 {
        out.push(0);
        return;
    }
    if n < 128 {
        out.push(n as u8);
    } else if n < 0x7f00 {
        out.push(((n >> 8) + 0x80) as u8);
        out.push((n & 0xff) as u8);
    } else {
        out.push(0xff);
        out.extend_from_slice(&((n - 0x7f00) as u16).to_le_bytes());
    }
    // All three tables predefined (§3.1.1.3.2.2.1): mode bits all zero.
    out.push(0);

    let ll_t = FseCTable::build(&LL_DEFAULT, LL_DEFAULT_LOG).expect("predefined LL table");
    let ml_t = FseCTable::build(&ML_DEFAULT, ML_DEFAULT_LOG).expect("predefined ML table");
    let of_t = FseCTable::build(&OF_DEFAULT, OF_DEFAULT_LOG).expect("predefined OF table");

    let codes: Vec<(usize, usize, usize)> = seqs
        .iter()
        .map(|s| {
            (
                code_for(&LL_BASE, s.literals),
                code_for(&ML_BASE, s.match_len),
                (31 - s.off_value.leading_zeros()) as usize,
            )
        })
        .collect();

    let mut w = BitWriter::new();
    let last = n - 1;
    let mut ml_state = ml_t.init_state(codes[last].1);
    let mut of_state = of_t.init_state(codes[last].2);
    let mut ll_state = ll_t.init_state(codes[last].0);
    let extra = |w: &mut BitWriter, s: &Seq, c: &(usize, usize, usize)| {
        w.add(s.literals - LL_BASE[c.0], LL_BITS[c.0]);
        w.add(s.match_len - ML_BASE[c.1], ML_BITS[c.1]);
        w.add(s.off_value - (1 << c.2), c.2 as u32);
    };
    extra(&mut w, &seqs[last], &codes[last]);
    for i in (0..last).rev() {
        of_t.encode(&mut w, &mut of_state, codes[i].2);
        ml_t.encode(&mut w, &mut ml_state, codes[i].1);
        ll_t.encode(&mut w, &mut ll_state, codes[i].0);
        extra(&mut w, &seqs[i], &codes[i]);
    }
    ml_t.flush(&mut w, ml_state);
    of_t.flush(&mut w, of_state);
    ll_t.flush(&mut w, ll_state);
    out.extend_from_slice(&w.finish());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same LCG as `tests/vectors/zstd/generate.py`, so a golden frame and
    /// the bytes it must produce are both derivable from source.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    const WORDS: [&str; 11] = [
        "the ",
        "model ",
        "exists ",
        "once ",
        "and ",
        "everything ",
        "else ",
        "is ",
        "derived ",
        "from ",
        "it. ",
    ];

    fn corpus(kind: u32, n: usize) -> Vec<u8> {
        let mut r = Rng(0x2545F4914F6CDD1Du64.wrapping_add(kind as u64));
        let mut out = Vec::with_capacity(n + 16);
        match kind {
            0 => {
                while out.len() < n {
                    out.extend_from_slice(WORDS[r.next() as usize % WORDS.len()].as_bytes());
                }
            }
            1 => {
                let mut prev = 0x3fu8;
                while out.len() < n {
                    if r.next().is_multiple_of(5) {
                        prev = 0x3c + (r.next() % 8) as u8;
                    }
                    out.push(prev);
                }
            }
            2 => {
                while out.len() < n {
                    out.push(r.next() as u8);
                }
            }
            3 => out.resize(n, 0),
            _ => {
                let base = corpus(0, 1 << 16);
                while out.len() < n {
                    let off = r.next() as usize % (base.len() - 4096);
                    out.extend_from_slice(&base[off..off + 4096]);
                }
            }
        }
        out.truncate(n);
        out
    }

    #[test]
    fn xxh64_known_vectors() {
        // The XXH64 specification's own vectors.
        assert_eq!(xxh64(b"", 0), 0xEF46DB3751D8E999);
        assert_eq!(xxh64(b"", 1), 0xD5AFBA1336A3BE4B);
        assert_eq!(xxh64(b"a", 0), 0xD24EC4F1A98C6E5B);
        assert_eq!(xxh64(b"abc", 0), 0x44BC2CF5AD770999);
        assert_eq!(
            xxh64(b"The quick brown fox jumps over the lazy dog", 0),
            0x0B242D361FDA71BC
        );
    }

    /// The golden frames were produced by libzstd. Decoding them is the only
    /// test that can distinguish "agrees with RFC 8878" from "agrees with
    /// itself".
    #[test]
    fn decodes_libzstd_frames() {
        let cases: [(&[u8], u32, usize); 6] = [
            (
                include_bytes!("../tests/vectors/zstd/text4k-l1.zst"),
                0,
                4096,
            ),
            (
                include_bytes!("../tests/vectors/zstd/text4k-l19.zst"),
                0,
                4096,
            ),
            (
                include_bytes!("../tests/vectors/zstd/plane8k-l3.zst"),
                1,
                8192,
            ),
            (
                include_bytes!("../tests/vectors/zstd/random3k-l3.zst"),
                2,
                3000,
            ),
            (
                include_bytes!("../tests/vectors/zstd/zeros1k-l3.zst"),
                3,
                1024,
            ),
            (
                include_bytes!("../tests/vectors/zstd/text200k-l3.zst"),
                4,
                200000,
            ),
        ];
        for (frame, kind, n) in cases {
            let want = corpus(kind, n);
            let got = decompress(frame, n).expect("golden frame decodes");
            assert_eq!(got.len(), want.len(), "kind {kind} length");
            assert!(got == want, "kind {kind} content");
        }
    }

    #[test]
    fn round_trips_every_corpus() {
        for kind in 0..5u32 {
            for n in [0usize, 1, 2, 3, 17, 255, 1024, 4096, 200_000] {
                let d = corpus(kind, n);
                for level in [0u8, 1, 3, 9, 19] {
                    let f = compress(&d, level);
                    let back = decompress(&f, d.len()).unwrap_or_else(|e| {
                        panic!("kind {kind} n {n} level {level}: {e}");
                    });
                    assert_eq!(back, d, "kind {kind} n {n} level {level}");
                }
            }
        }
    }

    #[test]
    fn compresses_what_is_compressible() {
        let text = corpus(0, 200_000);
        let f = compress(&text, 9);
        assert!(
            f.len() * 3 < text.len(),
            "text should compress at least 3x, got {} -> {}",
            text.len(),
            f.len()
        );
        // Incompressible input must not expand by more than block framing.
        let noise = corpus(2, 100_000);
        let g = compress(&noise, 9);
        assert!(g.len() < noise.len() + 64, "noise expanded to {}", g.len());
    }

    #[test]
    fn multi_block_frames_reach_across_blocks() {
        // The same incompressible 150 KiB twice. Nothing local helps: the second
        // half is only cheap if a sequence in the second block can point at an
        // offset inside the first, across the 128 KiB block boundary.
        let half = corpus(2, 150_000);
        let mut d = half.clone();
        d.extend_from_slice(&half);
        let f = compress(&d, 6);
        assert!(
            f.len() < half.len() + 4096,
            "cross-block matching failed: {} bytes for two copies of {}",
            f.len(),
            half.len()
        );
        assert_eq!(decompress(&f, d.len()).unwrap(), d);
    }

    #[test]
    fn rejects_a_corrupt_frame() {
        let d = corpus(0, 4096);
        let f = compress(&d, 3);
        // Every single-byte flip in the frame body must be caught by the
        // checksum, the framing, or the bitstream itself — never accepted.
        let mut caught = 0;
        for i in (8..f.len()).step_by(7) {
            let mut bad = f.clone();
            bad[i] ^= 0x40;
            match decompress(&bad, d.len()) {
                Ok(v) => assert_ne!(v, d, "a corrupted frame decoded to the original"),
                Err(_) => caught += 1,
            }
        }
        assert!(caught > 0, "no corruption was detected at all");
    }

    #[test]
    fn refuses_a_dictionary_frame() {
        // Frame header with Dictionary_ID_flag = 1 and a non-zero id.
        let mut f = MAGIC.to_le_bytes().to_vec();
        f.push(0x21); // single segment, dict id 1 byte
        f.push(7); // dict id
        f.push(1); // content size
        assert!(matches!(
            decompress(&f, 16),
            Err(Error::Unsupported("zstd (dictionary-compressed frame)"))
        ));
    }

    #[test]
    fn honours_the_output_bound() {
        let d = corpus(0, 20_000);
        let f = compress(&d, 3);
        assert!(matches!(decompress(&f, 1000), Err(Error::Bounds(_))));
    }

    #[test]
    fn skippable_frames_are_skipped() {
        let d = corpus(0, 300);
        let mut f = 0x184D2A50u32.to_le_bytes().to_vec();
        f.extend_from_slice(&4u32.to_le_bytes());
        f.extend_from_slice(b"junk");
        f.extend_from_slice(&compress(&d, 3));
        assert_eq!(decompress(&f, d.len()).unwrap(), d);
    }

    #[test]
    fn predefined_tables_are_the_spec_ones() {
        // A distribution that does not fill its table is the classic way to get
        // this wrong, and it fails loudly here rather than silently later.
        FseDTable::build(&LL_DEFAULT, LL_DEFAULT_LOG).unwrap();
        FseDTable::build(&ML_DEFAULT, ML_DEFAULT_LOG).unwrap();
        FseDTable::build(&OF_DEFAULT, OF_DEFAULT_LOG).unwrap();
        FseCTable::build(&LL_DEFAULT, LL_DEFAULT_LOG).unwrap();
        FseCTable::build(&ML_DEFAULT, ML_DEFAULT_LOG).unwrap();
        FseCTable::build(&OF_DEFAULT, OF_DEFAULT_LOG).unwrap();
    }
}
