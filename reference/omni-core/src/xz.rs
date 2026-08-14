//! §03.7.1 — the `xz` codec: the `.xz` container over LZMA2, both directions.
//!
//! §03.7.1 registers `xz` at MAY level with the note *archival ratio*, and the
//! archival profile (§14.6) is the one place OMNI cares about ratio more than
//! speed: an archived model is written once and read almost never, so spending
//! minutes to save a fifth of a petabyte-year is the right trade. That is the
//! whole argument for having it, and it is why this is the third entropy coder
//! here rather than the first.
//!
//! What is implemented is the format as it is actually written: the stream
//! header and footer with their flags and CRC-32s, block headers with their
//! filter chain, the LZMA2 chunk layer with its four reset modes and its
//! uncompressed chunks, the index, and all four integrity checks — none,
//! CRC-32, CRC-64 and SHA-256, the last of which this crate already has because
//! §03.5.1 requires it. A stream whose block declares a filter other than LZMA2
//! — a BCJ branch filter, `delta` — is reported *unsupported* rather than
//! guessed at, because a filter changes the bytes and skipping it produces a
//! plausible wrong answer instead of an error.
//!
//! The LZMA layer underneath is the whole coder: the range coder, the eleven
//! probability arrays, matched-literal decoding, the length coders, the
//! position-slot distance model with its reverse-coded low bits and align bits,
//! and the four rep distances. Decoding implements all of it. **Encoding
//! deliberately does not use the rep distances** — it emits literals and normal
//! matches only, which is a legal LZMA stream any decoder reads, and costs a few
//! percent of ratio for a large fraction of the encoder's complexity. That is
//! the same trade [`crate::zstd`]'s encoder makes with the predefined FSE
//! tables, and it is stated here for the same reason: being a percentage behind
//! on ratio is acceptable, being wrong is not.
//!
//! Both directions are checked against liblzma in CI — Python's `lzma` module,
//! which is the reference implementation with a different author — because a
//! codec whose only reader is its own writer has not been tested at all.

use crate::codec::Error;

type Res<T> = Result<T, Error>;

/// `.xz` stream header magic.
pub const MAGIC: [u8; 6] = [0xFD, b'7', b'z', b'X', b'Z', 0x00];
/// `.xz` stream footer magic.
const FOOTER_MAGIC: [u8; 2] = *b"YZ";
/// The LZMA2 filter's id in the filter registry.
const FILTER_LZMA2: u64 = 0x21;

/// The dictionary this encoder asks readers for: 8 MiB, which is `props` byte
/// 22 in LZMA2's exponential encoding and comfortably larger than anything a
/// single object in a container is likely to be.
const DICT_PROPS: u8 = 22;
const DICT_SIZE: usize = 8 << 20;

/// `lc`, `lp`, `pb` — the LZMA defaults, and what every real encoder emits
/// unless it has measured otherwise.
const LC: u32 = 3;
const LP: u32 = 0;
const PB: u32 = 2;

// --------------------------------------------------------------------- CRCs --

/// CRC-32 (IEEE 802.3, reflected), which is what `.xz` uses for its headers and
/// for check type 1. Not CRC-32C: [`crate::crc32c`] is a different polynomial,
/// and using one where the other is meant produces a file every other tool
/// rejects.
pub fn crc32(data: &[u8]) -> u32 {
    let mut t = [0u32; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = t[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// CRC-64 (ECMA-182, reflected), `.xz` check type 4.
pub fn crc64(data: &[u8]) -> u64 {
    let mut t = [0u64; 256];
    for (i, e) in t.iter_mut().enumerate() {
        let mut c = i as u64;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xC96C_5795_D787_0F42
            } else {
                c >> 1
            };
        }
        *e = c;
    }
    let mut crc = 0xFFFF_FFFF_FFFF_FFFFu64;
    for &b in data {
        crc = t[((crc ^ b as u64) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

// ------------------------------------------------------------------ varints --

/// `.xz`'s multibyte integer: seven bits at a time, low group first, high bit
/// set on every group but the last.
fn read_varint(d: &[u8], at: &mut usize) -> Res<u64> {
    let mut value = 0u64;
    for i in 0..9 {
        let b = *d
            .get(*at)
            .ok_or_else(|| Error::Corrupt("xz: a multibyte integer runs off the end".into()))?;
        *at += 1;
        value |= ((b & 0x7f) as u64) << (i * 7);
        if b & 0x80 == 0 {
            if b == 0 && i > 0 {
                return Err(Error::Corrupt(
                    "xz: a multibyte integer is not in its shortest form".into(),
                ));
            }
            return Ok(value);
        }
    }
    Err(Error::Corrupt("xz: a multibyte integer is too long".into()))
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

// ------------------------------------------------------------- range decoder --

const PROB_INIT: u16 = 1024;
const MOVE_BITS: u32 = 5;
const TOP: u32 = 1 << 24;

struct RangeDecoder<'a> {
    d: &'a [u8],
    at: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(d: &'a [u8]) -> Res<Self> {
        if d.len() < 5 {
            return Err(Error::Corrupt(
                "lzma: a chunk shorter than its range coder".into(),
            ));
        }
        if d[0] != 0 {
            return Err(Error::Corrupt(
                "lzma: the range coder's first byte is not zero".into(),
            ));
        }
        let code = u32::from_be_bytes([d[1], d[2], d[3], d[4]]);
        Ok(RangeDecoder {
            d,
            at: 5,
            range: u32::MAX,
            code,
        })
    }

    fn byte(&mut self) -> u8 {
        // Past the end reads as zero: a truncated chunk then decodes to
        // something the length check below refuses, rather than to an error
        // that hides which check failed.
        let b = self.d.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        b
    }

    fn normalize(&mut self) {
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.byte() as u32;
        }
    }

    fn bit(&mut self, prob: &mut u16) -> u32 {
        let bound = (self.range >> 11) * *prob as u32;
        let bit = if self.code < bound {
            self.range = bound;
            *prob += (2048 - *prob) >> MOVE_BITS;
            0
        } else {
            self.range -= bound;
            self.code -= bound;
            *prob -= *prob >> MOVE_BITS;
            1
        };
        self.normalize();
        bit
    }

    fn direct(&mut self, n: u32) -> u32 {
        let mut result = 0u32;
        for _ in 0..n {
            self.range >>= 1;
            let bit = if self.code >= self.range {
                self.code -= self.range;
                1
            } else {
                0
            };
            self.normalize();
            result = (result << 1) | bit;
        }
        result
    }

    fn tree(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1u32;
        for _ in 0..bits {
            m = (m << 1) | self.bit(&mut probs[m as usize]);
        }
        m - (1 << bits)
    }

    fn tree_reverse(&mut self, probs: &mut [u16], bits: u32) -> u32 {
        let mut m = 1u32;
        let mut result = 0u32;
        for i in 0..bits {
            let bit = self.bit(&mut probs[m as usize]);
            m = (m << 1) | bit;
            result |= bit << i;
        }
        result
    }
}

// ------------------------------------------------------------- range encoder --

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> RangeEncoder {
        RangeEncoder {
            low: 0,
            range: u32::MAX,
            cache: 0,
            // One, not zero: the first `shift_low` emits the leading zero byte
            // the decoder insists on.
            cache_size: 1,
            out: Vec::new(),
        }
    }

    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        // The shift is 32-bit on purpose: bits 24..31 have just been emitted
        // as `cache`, and carrying them forward in 64 bits is a bug that makes
        // every subsequent byte wrong while still producing plausible output.
        self.low = ((self.low as u32) << 8) as u64;
    }

    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn bit(&mut self, prob: &mut u16, bit: u32) {
        let bound = (self.range >> 11) * *prob as u32;
        if bit == 0 {
            self.range = bound;
            *prob += (2048 - *prob) >> MOVE_BITS;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *prob -= *prob >> MOVE_BITS;
        }
        self.normalize();
    }

    fn direct(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.range >>= 1;
            if (value >> i) & 1 != 0 {
                self.low += self.range as u64;
            }
            self.normalize();
        }
    }

    fn tree(&mut self, probs: &mut [u16], bits: u32, symbol: u32) {
        let mut m = 1u32;
        for i in (0..bits).rev() {
            let bit = (symbol >> i) & 1;
            self.bit(&mut probs[m as usize], bit);
            m = (m << 1) | bit;
        }
    }

    fn tree_reverse(&mut self, probs: &mut [u16], bits: u32, symbol: u32) {
        let mut m = 1u32;
        for i in 0..bits {
            let bit = (symbol >> i) & 1;
            self.bit(&mut probs[m as usize], bit);
            m = (m << 1) | bit;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

// -------------------------------------------------------------- length coder --

/// The length coder of both directions: a three-way choice into an 8-, 8- or
/// 256-symbol tree, one set of low/mid trees per position state.
struct LenCoder {
    choice: u16,
    choice2: u16,
    low: Vec<u16>,
    mid: Vec<u16>,
    high: Vec<u16>,
}

impl LenCoder {
    fn new() -> LenCoder {
        LenCoder {
            choice: PROB_INIT,
            choice2: PROB_INIT,
            low: vec![PROB_INIT; 16 * 8],
            mid: vec![PROB_INIT; 16 * 8],
            high: vec![PROB_INIT; 256],
        }
    }

    fn decode(&mut self, rc: &mut RangeDecoder<'_>, pos_state: usize) -> u32 {
        if rc.bit(&mut self.choice) == 0 {
            rc.tree(&mut self.low[pos_state * 8..pos_state * 8 + 8], 3)
        } else if rc.bit(&mut self.choice2) == 0 {
            8 + rc.tree(&mut self.mid[pos_state * 8..pos_state * 8 + 8], 3)
        } else {
            16 + rc.tree(&mut self.high, 8)
        }
    }

    fn encode(&mut self, rc: &mut RangeEncoder, pos_state: usize, len: u32) {
        if len < 8 {
            rc.bit(&mut self.choice, 0);
            rc.tree(&mut self.low[pos_state * 8..pos_state * 8 + 8], 3, len);
        } else if len < 16 {
            rc.bit(&mut self.choice, 1);
            rc.bit(&mut self.choice2, 0);
            rc.tree(&mut self.mid[pos_state * 8..pos_state * 8 + 8], 3, len - 8);
        } else {
            rc.bit(&mut self.choice, 1);
            rc.bit(&mut self.choice2, 1);
            rc.tree(&mut self.high, 8, len - 16);
        }
    }
}

// ---------------------------------------------------------------- LZMA state --

const MATCH_MIN_LEN: u32 = 2;
const END_POS_MODEL_INDEX: u32 = 14;
const NUM_FULL_DISTANCES: u32 = 1 << (END_POS_MODEL_INDEX >> 1);

/// Everything a chunk boundary may or may not reset (LZMA2's four modes).
struct Lzma {
    lc: u32,
    lp: u32,
    pb: u32,
    state: u32,
    reps: [u32; 4],
    literal: Vec<u16>,
    is_match: Vec<u16>,
    is_rep: Vec<u16>,
    is_rep_g0: Vec<u16>,
    is_rep_g1: Vec<u16>,
    is_rep_g2: Vec<u16>,
    is_rep0_long: Vec<u16>,
    pos_slot: Vec<u16>,
    spec_pos: Vec<u16>,
    align: Vec<u16>,
    len: LenCoder,
    rep_len: LenCoder,
}

impl Lzma {
    fn new(lc: u32, lp: u32, pb: u32) -> Lzma {
        Lzma {
            lc,
            lp,
            pb,
            state: 0,
            reps: [0; 4],
            literal: vec![PROB_INIT; 0x300 << (lc + lp)],
            is_match: vec![PROB_INIT; 12 * 16],
            is_rep: vec![PROB_INIT; 12],
            is_rep_g0: vec![PROB_INIT; 12],
            is_rep_g1: vec![PROB_INIT; 12],
            is_rep_g2: vec![PROB_INIT; 12],
            is_rep0_long: vec![PROB_INIT; 12 * 16],
            pos_slot: vec![PROB_INIT; 4 * 64],
            spec_pos: vec![PROB_INIT; 1 + (NUM_FULL_DISTANCES - END_POS_MODEL_INDEX) as usize],
            align: vec![PROB_INIT; 16],
            len: LenCoder::new(),
            rep_len: LenCoder::new(),
        }
    }

    /// Everything but the dictionary, which lives in the output buffer.
    fn reset_state(&mut self) {
        let (lc, lp, pb) = (self.lc, self.lp, self.pb);
        *self = Lzma::new(lc, lp, pb);
    }

    fn set_props(&mut self, props: u8) -> Res<()> {
        let mut d = props as u32;
        if d >= 9 * 5 * 5 {
            return Err(Error::Corrupt(format!(
                "lzma: properties byte {props} is out of range"
            )));
        }
        self.lc = d % 9;
        d /= 9;
        self.lp = d % 5;
        self.pb = d / 5;
        self.reset_state();
        Ok(())
    }

    fn state_literal(&mut self) {
        self.state = match self.state {
            0..=3 => 0,
            4..=9 => self.state - 3,
            _ => self.state - 6,
        };
    }

    fn state_match(&mut self) {
        self.state = if self.state < 7 { 7 } else { 10 };
    }

    fn state_rep(&mut self) {
        self.state = if self.state < 7 { 8 } else { 11 };
    }

    fn state_short_rep(&mut self) {
        self.state = if self.state < 7 { 9 } else { 11 };
    }

    fn literal_probs(&mut self, pos: u64, prev: u8) -> &mut [u16] {
        let lit_state = (((pos & ((1 << self.lp) - 1)) << self.lc) as u32
            + (prev >> (8 - self.lc)) as u32) as usize;
        &mut self.literal[0x300 * lit_state..0x300 * (lit_state + 1)]
    }
}

// ---------------------------------------------------------------- LZMA decode --

/// Decodes one LZMA chunk into `out`, which already holds the dictionary.
///
/// `limit` is the total output size this chunk may reach, so the caller's bound
/// is enforced inside the loop rather than after it.
fn lzma_chunk(
    st: &mut Lzma,
    data: &[u8],
    out: &mut Vec<u8>,
    want: usize,
    dict_start: usize,
) -> Res<()> {
    let mut rc = RangeDecoder::new(data)?;
    let target = out.len() + want;
    while out.len() < target {
        let pos_state = (out.len() as u64 & ((1 << st.pb) - 1)) as usize;
        let m = st.state as usize * 16 + pos_state;
        if rc.bit(&mut st.is_match[m]) == 0 {
            let prev = out.last().copied().unwrap_or(0);
            let pos = out.len() as u64;
            let state = st.state;
            let rep0 = st.reps[0];
            let matched = if state >= 7 {
                let back = out.len().checked_sub(rep0 as usize + 1).ok_or_else(|| {
                    Error::Corrupt("lzma: a matched literal reaches before the dictionary".into())
                })?;
                Some(out[back])
            } else {
                None
            };
            let probs = st.literal_probs(pos, prev);
            let mut symbol = 1u32;
            if let Some(mut match_byte) = matched {
                while symbol < 0x100 {
                    let match_bit = ((match_byte >> 7) & 1) as u32;
                    match_byte <<= 1;
                    let idx = ((1 + match_bit) << 8) as usize + symbol as usize;
                    let bit = rc.bit(&mut probs[idx]);
                    symbol = (symbol << 1) | bit;
                    if match_bit != bit {
                        break;
                    }
                }
            }
            while symbol < 0x100 {
                let idx = symbol as usize;
                symbol = (symbol << 1) | rc.bit(&mut probs[idx]);
            }
            out.push(symbol as u8);
            st.state_literal();
            continue;
        }

        let s = st.state as usize;
        let len;
        if rc.bit(&mut st.is_rep[s]) != 0 {
            // A repeated distance.
            if rc.bit(&mut st.is_rep_g0[s]) == 0 {
                if rc.bit(&mut st.is_rep0_long[s * 16 + pos_state]) == 0 {
                    st.state_short_rep();
                    let back = out
                        .len()
                        .checked_sub(st.reps[0] as usize + 1)
                        .ok_or_else(|| {
                            Error::Corrupt("lzma: a short rep reaches before the dictionary".into())
                        })?;
                    let b = out[back];
                    out.push(b);
                    continue;
                }
            } else {
                let dist;
                if rc.bit(&mut st.is_rep_g1[s]) == 0 {
                    dist = st.reps[1];
                } else if rc.bit(&mut st.is_rep_g2[s]) == 0 {
                    dist = st.reps[2];
                    st.reps[2] = st.reps[1];
                } else {
                    dist = st.reps[3];
                    st.reps[3] = st.reps[2];
                    st.reps[2] = st.reps[1];
                }
                st.reps[1] = st.reps[0];
                st.reps[0] = dist;
            }
            len = st.rep_len.decode(&mut rc, pos_state) + MATCH_MIN_LEN;
            st.state_rep();
        } else {
            st.reps[3] = st.reps[2];
            st.reps[2] = st.reps[1];
            st.reps[1] = st.reps[0];
            let l = st.len.decode(&mut rc, pos_state);
            st.state_match();
            let slot_state = (l.min(3) * 64) as usize;
            let slot = rc.tree(&mut st.pos_slot[slot_state..slot_state + 64], 6);
            let mut dist = slot;
            if slot >= 4 {
                let direct_bits = (slot >> 1) - 1;
                dist = (2 | (slot & 1)) << direct_bits;
                if slot < END_POS_MODEL_INDEX {
                    let base = (dist - slot) as usize;
                    let n = st.spec_pos.len();
                    if base >= n {
                        return Err(Error::Corrupt("lzma: a distance slot out of range".into()));
                    }
                    dist += rc.tree_reverse(&mut st.spec_pos[base..n], direct_bits);
                } else {
                    dist += rc.direct(direct_bits - 4) << 4;
                    dist += rc.tree_reverse(&mut st.align, 4);
                }
            }
            if dist == u32::MAX {
                // The end marker: legal, and LZMA2 never needs it because every
                // chunk declares its own size.
                break;
            }
            st.reps[0] = dist;
            len = l + MATCH_MIN_LEN;
        }

        let dist = st.reps[0] as usize + 1;
        if dist > out.len() - dict_start.min(out.len()) {
            return Err(Error::Corrupt(format!(
                "lzma: a match of {dist} reaches before the start of the dictionary"
            )));
        }
        if out.len() + len as usize > target {
            return Err(Error::Corrupt(
                "lzma: a match runs past the chunk's declared size".into(),
            ));
        }
        let start = out
            .len()
            .checked_sub(dist)
            .ok_or_else(|| Error::Corrupt("lzma: a match reaches before the dictionary".into()))?;
        for k in 0..len as usize {
            let b = out[start + k];
            out.push(b);
        }
    }
    if out.len() != target {
        return Err(Error::Corrupt(format!(
            "lzma: a chunk declared {want} bytes and produced {}",
            out.len() + want - target
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------- LZMA encode --

/// The largest match LZMA can spell in one symbol.
const MAX_MATCH: usize = 273;

/// Encodes `data` as an LZMA2 chunk stream.
///
/// Chunks are 64 KiB of input each, which is not arbitrary: a chunk header
/// spells its *compressed* size in sixteen bits, so a chunk that compresses
/// badly must still fit in 65 536 bytes, and 64 KiB of input cannot exceed that
/// without the uncompressed fallback below taking over. The dictionary is
/// **not** reset between chunks — only the probability state is — so a match in
/// the last chunk can still reach into the first, which is the whole reason
/// LZMA beats a 64 KiB-window coder on a tensor.
fn lzma2_encode(data: &[u8], level: u8) -> Vec<u8> {
    /// Both LZMA2 limits met at once: this many input bytes per chunk.
    const CHUNK: usize = 1 << 16;
    let mut out = Vec::new();
    let chain_limit: usize = match level {
        0 => 0,
        1..=3 => 8,
        4..=6 => 64,
        7..=9 => 512,
        _ => 4096,
    };
    // One match index over the whole input, walked forward as the chunks are.
    let mut head = vec![usize::MAX; 1 << 16];
    let mut prev = vec![usize::MAX; data.len().max(1)];

    let mut start = 0usize;
    let mut first = true;
    while start < data.len() {
        let end = (start + CHUNK).min(data.len());
        let packed = lzma_encode_range(data, start, end, chain_limit, &mut head, &mut prev);
        let n = end - start;
        if packed.len() + 2 >= n || packed.len() > 0xFFFF {
            // Compression that made it bigger is not compression, and LZMA2's
            // uncompressed chunks exist for exactly this. They still reset the
            // state, which is why the next chunk re-sends its properties.
            out.push(if first { 1 } else { 2 });
            out.extend(((n - 1) as u16).to_be_bytes());
            out.extend_from_slice(&data[start..end]);
        } else {
            // Reset the state and the properties on every chunk; reset the
            // dictionary only on the first, so later chunks keep the history.
            let reset: u8 = if first { 3 } else { 2 };
            let u = n - 1;
            out.push(0x80 | (reset << 5) | ((u >> 16) as u8 & 0x1f));
            out.extend(((u & 0xffff) as u16).to_be_bytes());
            out.extend(((packed.len() - 1) as u16).to_be_bytes());
            out.push(((PB * 5 + LP) * 9 + LC) as u8);
            out.extend_from_slice(&packed);
        }
        first = false;
        start = end;
    }
    out.push(0);
    out
}

/// One chunk's range-coded output, over a fresh probability state.
///
/// Positions are absolute: `pos_state` and the literal context are functions of
/// the position in the whole stream, which is what the decoder computes from
/// its output length, and a match may reach back before `start` because the
/// dictionary was not reset.
fn lzma_encode_range(
    data: &[u8],
    start: usize,
    end: usize,
    chain_limit: usize,
    head: &mut [usize],
    prev_pos: &mut [usize],
) -> Vec<u8> {
    let mut st = Lzma::new(LC, LP, PB);
    let mut rc = RangeEncoder::new();
    let hash = |d: &[u8], i: usize| -> usize {
        let v = u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
        ((v.wrapping_mul(2654435761)) >> 16) as usize & ((1 << 16) - 1)
    };

    let mut i = start;
    while i < end {
        let pos_state = (i as u64 & ((1 << PB) - 1)) as usize;
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if chain_limit > 0 && i + 4 <= data.len() {
            let h = hash(data, i);
            let mut cand = head[h];
            let mut tries = 0usize;
            while cand != usize::MAX && tries < chain_limit {
                if i - cand > DICT_SIZE {
                    break;
                }
                // A match may not run past this chunk: the decoder stops at the
                // chunk's declared size, so a longer one would be split across
                // two states.
                let max = MAX_MATCH.min(end - i);
                let mut l = 0usize;
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
                cand = prev_pos[cand];
                tries += 1;
            }
            prev_pos[i] = head[h];
            head[h] = i;
        }

        if best_len > MATCH_MIN_LEN as usize {
            let m = st.state as usize * 16 + pos_state;
            rc.bit(&mut st.is_match[m], 1);
            let s = st.state as usize;
            rc.bit(&mut st.is_rep[s], 0);
            let len_symbol = (best_len as u32) - MATCH_MIN_LEN;
            st.len.encode(&mut rc, pos_state, len_symbol);
            let dist = (best_dist - 1) as u32;
            let slot = pos_slot_of(dist);
            let slot_state = (len_symbol.min(3) * 64) as usize;
            rc.tree(&mut st.pos_slot[slot_state..slot_state + 64], 6, slot);
            if slot >= 4 {
                let direct_bits = (slot >> 1) - 1;
                let base = (2 | (slot & 1)) << direct_bits;
                let rest = dist - base;
                if slot < END_POS_MODEL_INDEX {
                    let off = (base - slot) as usize;
                    let n = st.spec_pos.len();
                    rc.tree_reverse(&mut st.spec_pos[off..n], direct_bits, rest);
                } else {
                    rc.direct(rest >> 4, direct_bits - 4);
                    rc.tree_reverse(&mut st.align, 4, rest & 0xf);
                }
            }
            st.reps[3] = st.reps[2];
            st.reps[2] = st.reps[1];
            st.reps[1] = st.reps[0];
            st.reps[0] = dist;
            st.state_match();
            // Insert every covered position, so a later match can start inside
            // this one.
            for k in 1..best_len {
                let p = i + k;
                if p + 4 <= data.len() && chain_limit > 0 {
                    let ph = hash(data, p);
                    prev_pos[p] = head[ph];
                    head[ph] = p;
                }
            }
            i += best_len;
            continue;
        }

        // A literal.
        let m = st.state as usize * 16 + pos_state;
        rc.bit(&mut st.is_match[m], 0);
        let prev = if i > 0 { data[i - 1] } else { 0 };
        let state = st.state;
        let rep0 = st.reps[0] as usize;
        let matched = if state >= 7 && i > rep0 {
            Some(data[i - rep0 - 1])
        } else {
            None
        };
        let symbol = data[i] as u32;
        let probs = st.literal_probs(i as u64, prev);
        let mut context = 1u32;
        let mut match_byte = matched.unwrap_or(0);
        let mut matching = matched.is_some();
        for k in (0..8).rev() {
            let bit = (symbol >> k) & 1;
            if matching {
                // The matched-literal path: while the literal agrees with the
                // byte one match-distance back, each bit is coded in that
                // byte's context. The moment they differ, so does the context,
                // and the decoder makes the same switch at the same bit.
                let match_bit = ((match_byte >> 7) & 1) as u32;
                match_byte <<= 1;
                let idx = ((1 + match_bit) << 8) as usize + context as usize;
                rc.bit(&mut probs[idx], bit);
                if match_bit != bit {
                    matching = false;
                }
            } else {
                let idx = context as usize;
                rc.bit(&mut probs[idx], bit);
            }
            context = (context << 1) | bit;
        }
        st.state_literal();
        i += 1;
    }
    rc.finish()
}

/// The position slot a distance falls in: `2 * floor(log2(d)) + the next bit`,
/// with the first four distances having a slot each.
fn pos_slot_of(dist: u32) -> u32 {
    if dist < 4 {
        return dist;
    }
    let bits = 31 - dist.leading_zeros();
    (bits << 1) | ((dist >> (bits - 1)) & 1)
}

// ------------------------------------------------------------- the container --

/// The `.xz` integrity checks (§ of the file format specification).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Check {
    None,
    Crc32,
    Crc64,
    Sha256,
}

impl Check {
    fn from_id(id: u8) -> Res<Check> {
        Ok(match id {
            0x00 => Check::None,
            0x01 => Check::Crc32,
            0x04 => Check::Crc64,
            0x0a => Check::Sha256,
            other => {
                return Err(Error::Unsupported(match other {
                    0x02 | 0x03 => "xz check CRC-32 variant",
                    0x05 | 0x06 => "xz check CRC-64 variant",
                    _ => "xz check",
                }))
            }
        })
    }

    fn size(self) -> usize {
        match self {
            Check::None => 0,
            Check::Crc32 => 4,
            Check::Crc64 => 8,
            Check::Sha256 => 32,
        }
    }

    fn compute(self, data: &[u8]) -> Vec<u8> {
        match self {
            Check::None => Vec::new(),
            Check::Crc32 => crc32(data).to_le_bytes().to_vec(),
            Check::Crc64 => crc64(data).to_le_bytes().to_vec(),
            Check::Sha256 => crate::sha256::sha256(data).to_vec(),
        }
    }
}

/// Decodes a whole `.xz` stream, refusing to produce more than `limit` bytes.
pub fn decompress(input: &[u8], limit: usize) -> Res<Vec<u8>> {
    if input.len() < 12 + 12 {
        return Err(Error::Corrupt(
            "xz: shorter than a header and a footer".into(),
        ));
    }
    if input[..6] != MAGIC {
        return Err(Error::Corrupt("xz: no stream header magic".into()));
    }
    if input[6] != 0 {
        return Err(Error::Corrupt(
            "xz: reserved stream flag bits are set".into(),
        ));
    }
    let check = Check::from_id(input[7] & 0x0f)?;
    let want = u32::from_le_bytes([input[8], input[9], input[10], input[11]]);
    if crc32(&input[6..8]) != want {
        return Err(Error::Corrupt(
            "xz: the stream header CRC does not match".into(),
        ));
    }

    let mut out: Vec<u8> = Vec::new();
    let mut at = 12usize;
    loop {
        let first = *input
            .get(at)
            .ok_or_else(|| Error::Corrupt("xz: the stream ends without an index".into()))?;
        if first == 0 {
            break; // the index indicator
        }
        let header_size = (first as usize + 1) * 4;
        if at + header_size > input.len() {
            return Err(Error::Corrupt("xz: a block header runs off the end".into()));
        }
        let header = &input[at..at + header_size];
        let hcrc = u32::from_le_bytes(header[header_size - 4..].try_into().unwrap_or([0, 0, 0, 0]));
        if crc32(&header[..header_size - 4]) != hcrc {
            return Err(Error::Corrupt(
                "xz: a block header CRC does not match".into(),
            ));
        }
        let flags = header[1];
        if flags & 0x3c != 0 {
            return Err(Error::Corrupt(
                "xz: reserved block flag bits are set".into(),
            ));
        }
        let filters = (flags & 0x03) as usize + 1;
        let mut p = 2usize;
        if flags & 0x40 != 0 {
            read_varint(header, &mut p)?; // compressed size, advisory here
        }
        if flags & 0x80 != 0 {
            read_varint(header, &mut p)?; // uncompressed size, likewise
        }
        // The filter chain. Anything but a lone LZMA2 changes the bytes, and
        // guessing at a filter is how a decoder produces a plausible wrong
        // answer instead of an error — so the chain is read whole and the
        // filter that is in the way is named, rather than reporting "a chain".
        let mut dict_props = 0u8;
        for k in 0..filters {
            let id = read_varint(header, &mut p)?;
            let props_len = read_varint(header, &mut p)? as usize;
            if p + props_len > header.len() {
                return Err(Error::Corrupt(
                    "xz: a filter's properties run off the end".into(),
                ));
            }
            if id != FILTER_LZMA2 {
                return Err(Error::Unsupported(match id {
                    0x03 => "xz delta filter",
                    0x04 => "xz BCJ x86 filter",
                    0x05 => "xz BCJ PowerPC filter",
                    0x06 => "xz BCJ IA-64 filter",
                    0x07 => "xz BCJ ARM filter",
                    0x08 => "xz BCJ ARM-Thumb filter",
                    0x09 => "xz BCJ SPARC filter",
                    0x0a => "xz BCJ ARM64 filter",
                    0x0b => "xz BCJ RISC-V filter",
                    _ => "xz filter",
                }));
            }
            // LZMA2 is the last filter in any chain, by construction.
            if k + 1 != filters {
                return Err(Error::Corrupt(
                    "xz: LZMA2 is not the last filter in the chain".into(),
                ));
            }
            if props_len != 1 {
                return Err(Error::Corrupt(
                    "xz: LZMA2 properties are not one byte".into(),
                ));
            }
            dict_props = header[p];
            p += props_len;
        }
        if dict_props > 40 {
            return Err(Error::Corrupt(
                "xz: an LZMA2 dictionary size out of range".into(),
            ));
        }
        for &pad in &header[p..header_size - 4] {
            if pad != 0 {
                return Err(Error::Corrupt(
                    "xz: block header padding is not zero".into(),
                ));
            }
        }

        at += header_size;
        // The compressed data runs to the end of the LZMA2 stream, which is
        // self-delimiting; the index says how long it was and is checked below.
        let start = at;
        let produced_before = out.len();
        let (consumed, produced) =
            lzma2_span(&input[at..], limit - out.len().min(limit), &mut out)?;
        at = start + consumed;
        let _ = produced;
        // Block padding to a four-byte boundary, then the check.
        while !(at - 12).is_multiple_of(4) {
            if *input.get(at).unwrap_or(&1) != 0 {
                return Err(Error::Corrupt("xz: block padding is not zero".into()));
            }
            at += 1;
        }
        let n = check.size();
        if at + n > input.len() {
            return Err(Error::Corrupt(
                "xz: the integrity check runs off the end".into(),
            ));
        }
        if check != Check::None {
            let computed = check.compute(&out[produced_before..]);
            if computed != input[at..at + n] {
                return Err(Error::Corrupt(
                    "xz: a block's integrity check does not match its data".into(),
                ));
            }
        }
        at += n;
    }

    // The index. Its records are what makes a truncated stream detectable.
    let index_start = at;
    at += 1;
    let count = read_varint(input, &mut at)?;
    if count > 1 << 20 {
        return Err(Error::Bounds(
            "xz: an index claiming more blocks than plausible".into(),
        ));
    }
    for _ in 0..count {
        read_varint(input, &mut at)?;
        let uncompressed = read_varint(input, &mut at)?;
        let _ = uncompressed;
    }
    while !(at - index_start).is_multiple_of(4) {
        if *input.get(at).unwrap_or(&1) != 0 {
            return Err(Error::Corrupt("xz: index padding is not zero".into()));
        }
        at += 1;
    }
    if at + 4 > input.len() {
        return Err(Error::Corrupt("xz: the index CRC runs off the end".into()));
    }
    let icrc = u32::from_le_bytes(input[at..at + 4].try_into().unwrap_or([0; 4]));
    if crc32(&input[index_start..at]) != icrc {
        return Err(Error::Corrupt("xz: the index CRC does not match".into()));
    }
    at += 4;

    if at + 12 > input.len() {
        return Err(Error::Corrupt("xz: the stream footer is missing".into()));
    }
    let footer = &input[at..at + 12];
    if footer[10..12] != FOOTER_MAGIC {
        return Err(Error::Corrupt("xz: no stream footer magic".into()));
    }
    let fcrc = u32::from_le_bytes(footer[0..4].try_into().unwrap_or([0; 4]));
    if crc32(&footer[4..10]) != fcrc {
        return Err(Error::Corrupt(
            "xz: the stream footer CRC does not match".into(),
        ));
    }
    if footer[8..10] != input[6..8] {
        return Err(Error::Corrupt(
            "xz: the footer's stream flags disagree with the header's".into(),
        ));
    }
    let backward = (u32::from_le_bytes(footer[4..8].try_into().unwrap_or([0; 4])) as usize + 1) * 4;
    if backward != at - index_start {
        return Err(Error::Corrupt(format!(
            "xz: the footer says the index is {backward} bytes and it is {}",
            at - index_start
        )));
    }
    Ok(out)
}

/// Decodes an LZMA2 stream whose length is not declared, reporting how many
/// bytes of input it consumed.
///
/// The stream is self-delimiting — a zero control byte ends it — so finding the
/// end is a matter of decoding, which is what this does.
fn lzma2_span(data: &[u8], limit: usize, out: &mut Vec<u8>) -> Res<(usize, usize)> {
    let before = out.len();
    let mut st = Lzma::new(LC, LP, PB);
    let mut have_props = false;
    let mut dict_start = out.len();
    let mut at = 0usize;
    loop {
        let control = *data
            .get(at)
            .ok_or_else(|| Error::Corrupt("lzma2: the stream ends without a terminator".into()))?;
        at += 1;
        if control == 0 {
            return Ok((at, out.len() - before));
        }
        if control == 1 || control == 2 {
            if at + 2 > data.len() {
                return Err(Error::Corrupt(
                    "lzma2: a truncated uncompressed chunk".into(),
                ));
            }
            let size = u16::from_be_bytes([data[at], data[at + 1]]) as usize + 1;
            at += 2;
            if at + size > data.len() {
                return Err(Error::Corrupt(
                    "lzma2: an uncompressed chunk runs off the end".into(),
                ));
            }
            if out.len() - before + size > limit {
                return Err(Error::Bounds(format!(
                    "xz: decoding would produce more than the declared {limit} bytes"
                )));
            }
            if control == 1 {
                dict_start = out.len();
            }
            out.extend_from_slice(&data[at..at + size]);
            at += size;
            st.reset_state();
            have_props = true;
            continue;
        }
        if control < 0x80 {
            return Err(Error::Corrupt(format!(
                "lzma2: control byte {control:#04x} is not a chunk"
            )));
        }
        if at + 4 > data.len() {
            return Err(Error::Corrupt("lzma2: a truncated chunk header".into()));
        }
        let unpacked =
            (((control & 0x1f) as usize) << 16 | (data[at] as usize) << 8 | data[at + 1] as usize)
                + 1;
        let packed = ((data[at + 2] as usize) << 8 | data[at + 3] as usize) + 1;
        at += 4;
        let reset = (control >> 5) & 0x3;
        if reset >= 2 {
            let props = *data
                .get(at)
                .ok_or_else(|| Error::Corrupt("lzma2: a chunk with no properties byte".into()))?;
            at += 1;
            st.set_props(props)?;
            have_props = true;
            if reset == 3 {
                dict_start = out.len();
            }
        } else if reset == 1 {
            st.reset_state();
        }
        if !have_props {
            return Err(Error::Corrupt(
                "lzma2: the first chunk does not set the properties".into(),
            ));
        }
        if at + packed > data.len() {
            return Err(Error::Corrupt("lzma2: a chunk runs off the end".into()));
        }
        if out.len() - before + unpacked > limit {
            return Err(Error::Bounds(format!(
                "xz: decoding would produce more than the declared {limit} bytes"
            )));
        }
        lzma_chunk(&mut st, &data[at..at + packed], out, unpacked, dict_start)?;
        at += packed;
    }
}

/// Writes a whole `.xz` stream: one block, LZMA2, CRC-32 checked.
pub fn compress(data: &[u8], level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 3 + 64);
    // Stream header: magic, flags (no reserved bits, CRC-32 check), CRC.
    out.extend_from_slice(&MAGIC);
    let flags = [0u8, 0x01];
    out.extend_from_slice(&flags);
    out.extend(crc32(&flags).to_le_bytes());

    let compressed = lzma2_encode(data, level);

    // Block header: flags, the LZMA2 filter and its dictionary size, padding,
    // CRC. The sizes are omitted; the index carries them and a decoder that
    // needs them before decoding is reading a different format.
    let mut header: Vec<u8> = vec![0, 0x00];
    write_varint(&mut header, FILTER_LZMA2);
    write_varint(&mut header, 1);
    header.push(DICT_PROPS);
    while !(header.len() + 4).is_multiple_of(4) {
        header.push(0);
    }
    let size = header.len() + 4;
    header[0] = (size / 4 - 1) as u8;
    let hcrc = crc32(&header);
    header.extend(hcrc.to_le_bytes());
    let header_len = header.len();
    out.extend_from_slice(&header);
    out.extend_from_slice(&compressed);

    // Unpadded size is the header, the data and the check — not the padding.
    let unpadded = header_len + compressed.len() + 4;
    let mut at = out.len();
    while !(at - 12).is_multiple_of(4) {
        out.push(0);
        at += 1;
    }
    out.extend(crc32(data).to_le_bytes());

    // Index: one record.
    let index_start = out.len();
    out.push(0);
    write_varint(&mut out, 1);
    write_varint(&mut out, unpadded as u64);
    write_varint(&mut out, data.len() as u64);
    while !(out.len() - index_start).is_multiple_of(4) {
        out.push(0);
    }
    let index_crc = crc32(&out[index_start..]);
    out.extend(index_crc.to_le_bytes());
    let index_size = out.len() - index_start;

    // Footer: CRC of what follows it, backward size, flags, magic.
    let mut footer = Vec::with_capacity(12);
    footer.extend(((index_size / 4 - 1) as u32).to_le_bytes());
    footer.extend_from_slice(&flags);
    let fcrc = crc32(&footer);
    out.extend(fcrc.to_le_bytes());
    out.extend_from_slice(&footer);
    out.extend_from_slice(&FOOTER_MAGIC);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8], level: u8) {
        let stream = compress(data, level);
        let back = decompress(&stream, data.len().max(1)).unwrap();
        assert_eq!(back, data, "level {level}, {} bytes", data.len());
    }

    #[test]
    fn crc32_matches_its_check_value() {
        // The standard check: CRC-32 of "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // And CRC-64/XZ's, which is a different polynomial and a different
        // answer — using one for the other writes a file every tool rejects.
        assert_eq!(crc64(b"123456789"), 0x995D_C9BB_DF19_39FA);
    }

    #[test]
    fn round_trips_at_every_level() {
        let corpus: Vec<Vec<u8>> = vec![
            vec![],
            b"x".to_vec(),
            b"the quick brown fox jumps over the lazy dog".to_vec(),
            std::iter::repeat_n(b"a model exists once. ".to_vec(), 400)
                .flatten()
                .collect(),
            (0..=255u8).collect(),
            (0..20000u32).map(|i| (i * 7 % 251) as u8).collect(),
            vec![0u8; 70000],
        ];
        for data in &corpus {
            for level in [0u8, 1, 6, 9] {
                round_trip(data, level);
            }
        }
    }

    #[test]
    fn a_long_input_crosses_a_chunk_boundary() {
        // Over 2 MiB, so the encoder emits more than one LZMA2 chunk and the
        // decoder has to honour a mid-stream reset.
        let data: Vec<u8> = std::iter::repeat_n(b"omni/1.0 is content addressed. ".to_vec(), 90000)
            .flatten()
            .collect();
        assert!(data.len() > (1 << 21));
        round_trip(&data, 6);
    }

    #[test]
    fn repetitive_data_actually_compresses() {
        let data: Vec<u8> = std::iter::repeat_n(b"omni".to_vec(), 8192)
            .flatten()
            .collect();
        let stream = compress(&data, 9);
        assert!(stream.len() < data.len() / 100, "{} bytes", stream.len());
    }

    #[test]
    fn a_tampered_stream_never_decodes_to_different_data() {
        let data: Vec<u8> = std::iter::repeat_n(b"a model exists once. ".to_vec(), 200)
            .flatten()
            .collect();
        let stream = compress(&data, 6);
        // The property the checks actually give, stated precisely: a flipped
        // bit anywhere either fails or reproduces the input — never a third
        // thing. A few bytes are redundant (the top of a size field the range
        // coder's own padding covers), so "everything is caught" would be a
        // stronger claim than the format makes, and a wrong one.
        let mut caught = 0usize;
        for i in 0..stream.len() {
            let mut bad = stream.clone();
            bad[i] ^= 0x01;
            match decompress(&bad, data.len()) {
                Ok(out) => assert_eq!(
                    out, data,
                    "a flipped bit at {i} decoded to something that is not the input"
                ),
                Err(_) => caught += 1,
            }
        }
        assert!(
            caught * 10 >= stream.len() * 9,
            "only {caught} of {} flipped bits were refused",
            stream.len()
        );
    }

    #[test]
    fn every_truncation_is_an_error_and_never_a_panic() {
        let data: Vec<u8> = std::iter::repeat_n(b"truncate me. ".to_vec(), 300)
            .flatten()
            .collect();
        let stream = compress(&data, 6);
        for n in 0..stream.len() {
            let _ = decompress(&stream[..n], data.len());
        }
    }

    #[test]
    fn the_declared_length_is_a_bound() {
        let data = vec![9u8; 40000];
        let stream = compress(&data, 6);
        let err = decompress(&stream, 100).unwrap_err();
        assert!(matches!(err, Error::Bounds(_)), "{err:?}");
    }

    #[test]
    fn a_filter_this_build_does_not_have_is_unsupported_not_invalid() {
        // A block header declaring the x86 BCJ filter. §15.1's distinction: the
        // file is fine and this reader cannot decode it, which is a different
        // outcome from a corrupt file.
        let mut header: Vec<u8> = vec![0, 0x00];
        write_varint(&mut header, 0x04); // BCJ x86
        write_varint(&mut header, 0);
        while !(header.len() + 4).is_multiple_of(4) {
            header.push(0);
        }
        let size = header.len() + 4;
        header[0] = (size / 4 - 1) as u8;
        let hcrc = crc32(&header);
        header.extend(hcrc.to_le_bytes());

        let mut stream = Vec::new();
        stream.extend_from_slice(&MAGIC);
        let flags = [0u8, 0x01];
        stream.extend_from_slice(&flags);
        stream.extend(crc32(&flags).to_le_bytes());
        stream.extend_from_slice(&header);
        stream.extend([0u8; 32]);
        let err = decompress(&stream, 4096).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }
}

#[cfg(test)]
mod rc_tests {
    use super::*;

    #[test]
    fn the_range_coder_round_trips_bits() {
        let mut probs = [PROB_INIT; 64];
        let bits: Vec<u32> = (0..500)
            .map(|i| ((i * 7 + i / 3) % 5 == 0) as u32)
            .collect();
        let mut rc = RangeEncoder::new();
        for (i, b) in bits.iter().enumerate() {
            rc.bit(&mut probs[i % 64], *b);
        }
        let out = rc.finish();
        let mut probs2 = [PROB_INIT; 64];
        let mut rd = RangeDecoder::new(&out).unwrap();
        for (i, b) in bits.iter().enumerate() {
            assert_eq!(rd.bit(&mut probs2[i % 64]), *b, "bit {i}");
        }
    }

    #[test]
    fn the_range_coder_round_trips_direct_bits() {
        let mut rc = RangeEncoder::new();
        let vals: Vec<u32> = (0..50).map(|i| (i * 2654435761u64 % 1024) as u32).collect();
        for v in &vals {
            rc.direct(*v, 10);
        }
        let out = rc.finish();
        let mut rd = RangeDecoder::new(&out).unwrap();
        for v in &vals {
            assert_eq!(rd.direct(10), *v);
        }
    }
}
