//! Brotli decompression (RFC 7932).
//!
//! §03.7.1 lists `brotli` as a MAY-level codec, and this crate declined it for a
//! long time on a specific and correct argument: a decoder without the 122 KiB
//! static dictionary answers *wrongly* on any stream that references it, rather
//! than refusing, and half of it is worse than none. That argument was about the
//! dictionary being unavailable, not about the work — and the dictionary is
//! available (google/brotli ships it under the MIT licence). So the reason
//! expired, and here is the decoder.
//!
//! It is the whole format: prefix codes both simple and complex, the
//! insert-and-copy command grammar of §5, block-type and block-count switching
//! with their move-to-front models, the context maps for literals and distances,
//! the postfix/direct distance parameterisation, and the static dictionary with
//! all 121 of §8's word transforms. What it decodes, it decodes because the
//! bitstream said so, not because a heuristic guessed.
//!
//! There is no independent second implementation of brotli to write from the
//! spec — it is one library — so the check is differential against that library:
//! `tools/brotli-fixture.py` compresses a corpus with `libbrotli` and this
//! decoder reproduces every byte. That is the strongest oracle available and the
//! one the format deserves.
//!
//! The **encoder** here emits only uncompressed meta-blocks. That is valid
//! brotli — libbrotli decodes it, and CI checks that it does — but it does not
//! compress, and it is not meant to: brotli's value in this format is *reading*
//! what other tools wrote, and a storage codec that re-compresses with a
//! worse-than-zstd encoder would be a footgun. `omni pack --codec brotli` is
//! refused for that reason; the codec exists to import, `omni repack` away from
//! brotli, and verify round trips.

use crate::brotli_tables as tbl;

type Res<T> = Result<T, Error>;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Corrupt(&'static str),
    TooLarge,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Truncated => write!(f, "brotli stream ends mid-symbol"),
            Error::Corrupt(w) => write!(f, "corrupt brotli stream: {w}"),
            Error::TooLarge => write!(f, "brotli output exceeds the declared bound"),
        }
    }
}

// ---------------------------------------------------------------------------
// Bit reader: LSB-first within each byte, like deflate (RFC 7932 §1.2).
// ---------------------------------------------------------------------------

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Bits<'a> {
        Bits {
            data,
            pos: 0,
            bit: 0,
        }
    }

    /// Reads `n` bits (0..=24), LSB first.
    fn take(&mut self, n: u32) -> Res<u32> {
        let mut v = 0u32;
        for i in 0..n {
            if self.pos >= self.data.len() {
                return Err(Error::Truncated);
            }
            let b = (self.data[self.pos] >> self.bit) & 1;
            v |= (b as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    fn bit1(&mut self) -> Res<u32> {
        self.take(1)
    }

    /// RFC 7932 §9.1's variable-length non-negative integer for MLEN-style
    /// fields is spelled out at each use; this is the plain fixed reader.
    fn byte_align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Prefix (Huffman) codes.
// ---------------------------------------------------------------------------

/// A canonical prefix code, decoded by a flat lookup table of `1 << max_len`
/// entries. Brotli's alphabets are small enough (≤ 704 for literals) that the
/// table is cheap and the decode is one shift and one index.
struct Prefix {
    /// `(symbol, length)` for each of the `1 << bits` bit patterns, LSB-first.
    table: Vec<(u16, u8)>,
    bits: u32,
}

impl Prefix {
    /// Builds a code from per-symbol lengths (0 = unused), the canonical way
    /// RFC 7932 §3.2 assigns codes: shorter lengths first, and within a length
    /// in symbol order, with the bits reversed because the stream is LSB-first.
    fn from_lengths(lengths: &[u8]) -> Res<Prefix> {
        let max_len = *lengths.iter().max().unwrap_or(&0);
        if max_len == 0 {
            return Err(Error::Corrupt("a prefix code with no symbols"));
        }
        // First code of each length (canonical Huffman).
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        // A single symbol of length... brotli uses length 1 for it; handled by
        // the general path below since count[1]==1 yields a 1-bit code.
        let mut next = [0u32; 16];
        let mut code = 0u32;
        count[0] = 0;
        for len in 1..16 {
            code = (code + count[len - 1]) << 1;
            next[len] = code;
        }
        let bits = max_len as u32;
        let mut table = vec![(0u16, 0u8); 1usize << bits];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let c = next[l as usize];
            next[l as usize] += 1;
            // Reverse the `l`-bit code so it reads LSB-first, then fill every
            // table slot whose low `l` bits match.
            let rev = reverse_bits(c, l as u32);
            let step = 1usize << l;
            let mut idx = rev as usize;
            while idx < table.len() {
                table[idx] = (sym as u16, l);
                idx += step;
            }
        }
        Ok(Prefix { table, bits })
    }

    /// A code with all symbols the same length — the degenerate simple-code case
    /// and the shape the code-length alphabet sometimes takes.
    fn decode(&self, br: &mut Bits<'_>) -> Res<u16> {
        // Peek `bits`, look up, then consume only the code's actual length.
        let peek = br.peek(self.bits)?;
        let (sym, len) = self.table[peek as usize];
        if len == 0 {
            return Err(Error::Corrupt("an unassigned prefix code was read"));
        }
        br.consume(len as u32);
        Ok(sym)
    }
}

impl Bits<'_> {
    /// Looks at the next `n` bits without consuming them, zero-padding at EOF —
    /// safe because a valid code never runs past the stream, and the caller
    /// consumes only the real length.
    fn peek(&self, n: u32) -> Res<u32> {
        let mut v = 0u32;
        let (mut pos, mut bit) = (self.pos, self.bit);
        for i in 0..n {
            if pos >= self.data.len() {
                // Zero pad. A real code will not depend on these bits.
                break;
            }
            let b = (self.data[pos] >> bit) & 1;
            v |= (b as u32) << i;
            bit += 1;
            if bit == 8 {
                bit = 0;
                pos += 1;
            }
        }
        let _ = &mut v;
        Ok(v)
    }

    fn consume(&mut self, n: u32) {
        let total = self.bit + n;
        self.pos += (total / 8) as usize;
        self.bit = total % 8;
    }
}

fn reverse_bits(v: u32, n: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..n {
        r |= ((v >> i) & 1) << (n - 1 - i);
    }
    r
}

/// The code-length code lengths are read in this fixed order (RFC 7932 §3.5).
const CODE_LENGTH_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Reads a prefix code for an alphabet of `alphabet_size` symbols (§3.4, §3.5).
fn read_prefix(br: &mut Bits<'_>, alphabet_size: usize) -> Res<Prefix> {
    let hskip = br.take(2)?;
    if hskip == 1 {
        return read_simple_prefix(br, alphabet_size);
    }
    // Complex code: read the code-length code, then run it over the alphabet.
    let mut cl_lengths = [0u8; 18];
    let mut space = 32u32;
    let mut nonzero = 0;
    // The 18 code-length symbols each get a code *length* in 0..=5, and those
    // lengths are themselves read through a fixed canonical Huffman code whose
    // own lengths are [2,4,3,2,2,4] over the values 0..=5 (RFC 7932 §3.5). It is
    // canonical, so it is built the same way every other code here is rather
    // than hand-decoded — hand-decoding it was the first bug this found.
    let cl_cl = Prefix::from_lengths(&[2, 4, 3, 2, 2, 4])?;
    for i in (hskip as usize)..18 {
        let len = cl_cl.decode(br)? as u8;
        cl_lengths[CODE_LENGTH_ORDER[i]] = len;
        if len != 0 {
            nonzero += 1;
            space -= 32 >> len;
            if space == 0 {
                break;
            }
        }
    }
    if nonzero != 1 && space != 0 {
        return Err(Error::Corrupt("code-length code is not complete"));
    }
    let cl_code = Prefix::from_lengths(&cl_lengths)?;

    // Now read the symbol lengths for the real alphabet, with the RLE symbols
    // 16 (repeat previous non-zero) and 17 (repeat zero).
    let mut lengths = vec![0u8; alphabet_size];
    let mut i = 0usize;
    let mut prev_len = 8u8;
    let mut prev_repeat = 0u32;
    let mut prev_repeat_code = 0u8;
    let mut total_space = 0u32;
    let target = 1u32 << 15;
    while i < alphabet_size && total_space < target {
        let sym = cl_code.decode(br)? as u8;
        match sym {
            0..=15 => {
                lengths[i] = sym;
                i += 1;
                prev_repeat = 0;
                if sym != 0 {
                    prev_len = sym;
                    total_space += target >> sym;
                }
            }
            16 | 17 => {
                let extra_bits = if sym == 16 { 2 } else { 3 };
                // §3.5: both run-length symbols start their repeat count at 3.
                let base = 3;
                if prev_repeat_code != sym {
                    prev_repeat = 0;
                }
                prev_repeat_code = sym;
                let mut repeat = prev_repeat;
                if repeat > 0 {
                    repeat -= 2;
                    repeat <<= extra_bits;
                }
                repeat += br.take(extra_bits)? + base;
                let new = repeat;
                let fill = new - prev_repeat;
                prev_repeat = new;
                let value = if sym == 16 { prev_len } else { 0 };
                for _ in 0..fill {
                    if i >= alphabet_size {
                        return Err(Error::Corrupt("run-length extends past the alphabet"));
                    }
                    lengths[i] = value;
                    i += 1;
                    if value != 0 {
                        total_space += target >> value;
                    }
                }
            }
            _ => return Err(Error::Corrupt("code-length symbol out of range")),
        }
    }
    Prefix::from_lengths(&lengths)
}

/// A simple prefix code: 1..=4 symbols listed explicitly (§3.4).
fn read_simple_prefix(br: &mut Bits<'_>, alphabet_size: usize) -> Res<Prefix> {
    let nsym = br.take(2)? + 1;
    let sym_bits = bit_width(alphabet_size as u32 - 1);
    let mut syms = Vec::new();
    for _ in 0..nsym {
        syms.push(br.take(sym_bits)? as u16);
    }
    // A simple code assigns codes to the symbols **in the order they were
    // listed**, not in symbol-value order. This is the whole difference from a
    // complex code, and getting it wrong desynchronises the decoder the moment a
    // simple code has two symbols of the same length — which was this decoder's
    // second bug. So the (symbol, length) pairs are built in list order and fed
    // to a builder that preserves it, rather than into `from_lengths`, which
    // sorts by symbol value.
    let pairs: Vec<(u16, u8)> = match nsym {
        1 => {
            // A single symbol: a zero-bit code that consumes nothing.
            return Ok(Prefix {
                table: vec![(syms[0], 0); 1],
                bits: 0,
            });
        }
        2 => vec![(syms[0], 1), (syms[1], 1)],
        3 => vec![(syms[0], 1), (syms[1], 2), (syms[2], 2)],
        4 => {
            if br.bit1()? == 0 {
                vec![(syms[0], 2), (syms[1], 2), (syms[2], 2), (syms[3], 2)]
            } else {
                vec![(syms[0], 1), (syms[1], 2), (syms[2], 3), (syms[3], 3)]
            }
        }
        _ => unreachable!(),
    };
    prefix_from_ordered(&pairs)
}

/// Builds a prefix code from `(symbol, length)` pairs, assigning canonical codes
/// shortest-length-first and, within a length, in the order given. `from_lengths`
/// assigns by symbol value instead; the two agree only when no length is shared,
/// which is why simple codes need this and complex codes do not.
fn prefix_from_ordered(pairs: &[(u16, u8)]) -> Res<Prefix> {
    let max_len = pairs.iter().map(|(_, l)| *l).max().unwrap_or(0);
    if max_len == 0 {
        return Err(Error::Corrupt("a simple code with no symbols"));
    }
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| pairs[i].1); // stable: ties keep list order
    let bits = max_len as u32;
    let mut table = vec![(0u16, 0u8); 1usize << bits];
    let mut code = 0u32;
    let mut prev_len = 0u8;
    for &i in &order {
        let (sym, len) = pairs[i];
        code <<= len - prev_len;
        prev_len = len;
        let rev = reverse_bits(code, len as u32);
        let step = 1usize << len;
        let mut idx = rev as usize;
        while idx < table.len() {
            table[idx] = (sym, len);
            idx += step;
        }
        code += 1;
    }
    Ok(Prefix { table, bits })
}

fn bit_width(mut v: u32) -> u32 {
    let mut n = 0;
    while v > 0 {
        n += 1;
        v >>= 1;
    }
    n.max(1)
}

// A zero-bit code returns its only symbol without consuming bits.
impl Prefix {
    fn decode_maybe_empty(&self, br: &mut Bits<'_>) -> Res<u16> {
        if self.bits == 0 {
            return Ok(self.table[0].0);
        }
        self.decode(br)
    }
}

// ---------------------------------------------------------------------------
// Block-switching: a symbol chosen by a prefix code, its type tracked with a
// move-to-front-like recency model (§6).
// ---------------------------------------------------------------------------

struct BlockSwitch {
    type_code: Prefix,
    count_code: Prefix,
    num_types: usize,
    cur_type: usize,
    prev_type: usize,
    remaining: u32,
}

impl BlockSwitch {
    fn read(br: &mut Bits<'_>, num_types: usize) -> Res<Option<BlockSwitch>> {
        if num_types < 2 {
            return Ok(None);
        }
        let type_code = read_prefix(br, num_types + 2)?;
        let count_code = read_prefix(br, 26)?;
        let sym = count_code.decode_maybe_empty(br)?;
        let remaining = read_block_count(br, sym)?;
        Ok(Some(BlockSwitch {
            type_code,
            count_code,
            num_types,
            cur_type: 0,
            prev_type: 1,
            remaining,
        }))
    }

    /// Consumes one command's worth of this block category, switching type when
    /// the current run is exhausted. Returns the current block type.
    fn step(&mut self, br: &mut Bits<'_>) -> Res<usize> {
        if self.remaining == 0 {
            let sym = self.type_code.decode_maybe_empty(br)? as usize;
            let next = match sym {
                0 => self.prev_type,
                1 => (self.cur_type + 1) % self.num_types,
                _ => sym - 2,
            };
            self.prev_type = self.cur_type;
            self.cur_type = next;
            let cs = self.count_code.decode_maybe_empty(br)?;
            self.remaining = read_block_count(br, cs)?;
        }
        self.remaining -= 1;
        Ok(self.cur_type)
    }
}

/// Block-count code (§6, §9.2): 26 symbols, each a base plus extra bits.
fn read_block_count(br: &mut Bits<'_>, sym: u16) -> Res<u32> {
    const BASE: [u32; 26] = [
        1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497, 753,
        1265, 2289, 4337, 8433, 16625,
    ];
    const EXTRA: [u32; 26] = [
        2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24,
    ];
    let s = sym as usize;
    if s >= 26 {
        return Err(Error::Corrupt("block-count symbol out of range"));
    }
    Ok(BASE[s] + br.take(EXTRA[s])?)
}

// ---------------------------------------------------------------------------
// Context maps (§7.3): a per-(block-type, context) index into the tree set,
// stored with run-length zeros and inverse move-to-front.
// ---------------------------------------------------------------------------

fn read_context_map(br: &mut Bits<'_>, num_trees: usize, size: usize) -> Res<Vec<u8>> {
    if num_trees < 2 {
        return Ok(vec![0u8; size]);
    }
    let rlemax = if br.bit1()? == 1 { br.take(4)? + 1 } else { 0 };
    let code = read_prefix(br, num_trees + rlemax as usize)?;
    let mut map = Vec::with_capacity(size);
    while map.len() < size {
        let sym = code.decode_maybe_empty(br)? as u32;
        if sym == 0 {
            map.push(0);
        } else if sym <= rlemax {
            let run = (1u32 << sym) + br.take(sym)?;
            for _ in 0..run {
                if map.len() >= size {
                    return Err(Error::Corrupt("context-map run overshoots"));
                }
                map.push(0);
            }
        } else {
            map.push((sym - rlemax) as u8);
        }
    }
    // Inverse move-to-front if the IMTF bit is set.
    if br.bit1()? == 1 {
        inverse_mtf(&mut map);
    }
    Ok(map)
}

fn inverse_mtf(v: &mut [u8]) {
    let mut table: Vec<u8> = (0..=255).collect();
    for x in v.iter_mut() {
        let idx = *x as usize;
        let val = table[idx];
        *x = val;
        table.copy_within(0..idx, 1);
        table[0] = val;
    }
}

// ---------------------------------------------------------------------------
// Insert-and-copy length codes (§5).
// ---------------------------------------------------------------------------

/// Splits an insert-and-copy command symbol into (insert_len, copy_len,
/// implicit_distance_zero) via RFC 7932 §5's table.
fn insert_copy(br: &mut Bits<'_>, sym: u16) -> Res<(u32, u32, bool)> {
    let s = sym as u32;
    // The 704-symbol space is grouped into 11 ranges of 64. Within a range the
    // top 3 of the low 6 bits pick an insert code and the low 3 a copy code
    // (§5). The per-range base of each code, and whether the command carries an
    // implicit last-distance, are what brotli's encoder packs with the
    // `CombineLengthCodes` magic constant 0x520D40; inverting that mapping gives
    // the eleven rows below. `dist0` (the implicit distance) holds only for the
    // first two ranges, where the encoder used the previous distance.
    let (insert_code, copy_code, dist0) = {
        let range = s >> 6;
        let sub = s & 0x3f;
        let (ins_hi, copy_hi, d0) = match range {
            0 => (0, 0, true),
            1 => (0, 8, true),
            2 => (0, 0, false),
            3 => (0, 8, false),
            4 => (8, 0, false),
            5 => (8, 8, false),
            6 => (0, 16, false),
            7 => (16, 0, false),
            8 => (8, 16, false),
            9 => (16, 8, false),
            10 => (16, 16, false),
            _ => return Err(Error::Corrupt("insert-copy symbol out of range")),
        };
        let ins = ins_hi + ((sub >> 3) & 7);
        let cpy = copy_hi + (sub & 7);
        (ins, cpy, d0)
    };
    let insert = insert_extra(br, insert_code)?;
    let copy = copy_extra(br, copy_code)?;
    Ok((insert, copy, dist0))
}

fn insert_extra(br: &mut Bits<'_>, code: u32) -> Res<u32> {
    const BASE: [u32; 24] = [
        0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114,
        6210, 22594,
    ];
    const EXTRA: [u32; 24] = [
        0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24,
    ];
    let c = code as usize;
    if c >= 24 {
        return Err(Error::Corrupt("insert length code out of range"));
    }
    Ok(BASE[c] + br.take(EXTRA[c])?)
}

fn copy_extra(br: &mut Bits<'_>, code: u32) -> Res<u32> {
    const BASE: [u32; 24] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094,
        2118,
    ];
    const EXTRA: [u32; 24] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
    ];
    let c = code as usize;
    if c >= 24 {
        return Err(Error::Corrupt("copy length code out of range"));
    }
    Ok(BASE[c] + br.take(EXTRA[c])?)
}

// ---------------------------------------------------------------------------
// The decoder proper.
// ---------------------------------------------------------------------------

/// Decompresses a brotli stream, bounded by `cap` output bytes.
pub fn decompress(data: &[u8], cap: usize) -> Res<Vec<u8>> {
    let mut br = Bits::new(data);
    let wbits = read_window_bits(&mut br)?;
    let window = 1usize << wbits;
    let _window = window;

    let mut out: Vec<u8> = Vec::new();
    // Ring of the last four distances (§4), initialised per the RFC.
    let mut dist_ring = [16i64, 15, 11, 4];
    let mut dist_rb_idx = 0usize;

    loop {
        let is_last = br.bit1()?;
        if is_last == 1 && br.bit1()? == 1 {
            // ISLASTEMPTY
            break;
        }
        // Meta-block length: MNIBBLES then the length, or metadata / empty.
        let mnibbles_sel = br.take(2)?;
        if mnibbles_sel == 3 {
            // Metadata block: skip.
            let reserved = br.bit1()?;
            if reserved != 0 {
                return Err(Error::Corrupt("reserved bit set in metadata block"));
            }
            let mskipbytes = br.take(2)?;
            let mut mskip = 0u32;
            for i in 0..mskipbytes {
                mskip |= br.take(8)? << (8 * i);
            }
            if mskipbytes > 1 && (mskip >> (8 * (mskipbytes - 1))) == 0 {
                return Err(Error::Corrupt("non-minimal metadata length"));
            }
            let mskip = if mskipbytes == 0 { 0 } else { mskip + 1 };
            br.byte_align();
            for _ in 0..mskip {
                br.take(8)?;
            }
            continue;
        }
        let mnibbles = 4 + mnibbles_sel;
        let mut mlen = 0u32;
        for i in 0..mnibbles {
            mlen |= br.take(4)? << (4 * i);
        }
        if mnibbles > 4 && (mlen >> (4 * (mnibbles - 1))) == 0 {
            return Err(Error::Corrupt("non-minimal meta-block length"));
        }
        let mlen = mlen as usize + 1;
        if is_last == 0 {
            // ISUNCOMPRESSED bit only appears on non-last blocks.
            let uncompressed = br.bit1()?;
            if uncompressed == 1 {
                br.byte_align();
                for _ in 0..mlen {
                    if out.len() >= cap {
                        return Err(Error::TooLarge);
                    }
                    out.push(br.take(8)? as u8);
                }
                continue;
            }
        }

        decode_meta_block(
            &mut br,
            mlen,
            cap,
            _window,
            &mut out,
            &mut dist_ring,
            &mut dist_rb_idx,
        )?;

        if is_last == 1 {
            break;
        }
    }
    if out.len() > cap {
        return Err(Error::TooLarge);
    }
    Ok(out)
}

fn read_window_bits(br: &mut Bits<'_>) -> Res<u32> {
    // WBITS encoding (§9.1).
    if br.bit1()? == 0 {
        return Ok(16);
    }
    let n = br.take(3)?;
    if n != 0 {
        return Ok(17 + n);
    }
    let n = br.take(3)?;
    match n {
        0 => Ok(17), // "large window" marker is not used by default streams
        _ => Ok(8 + n),
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_meta_block(
    br: &mut Bits<'_>,
    mlen: usize,
    cap: usize,
    window: usize,
    out: &mut Vec<u8>,
    dist_ring: &mut [i64; 4],
    dist_rb_idx: &mut usize,
) -> Res<()> {
    let start = out.len();

    let n_block_types_l = read_block_type_count(br)?;

    let mut sw_l = BlockSwitch::read(br, n_block_types_l)?;
    let n_block_types_i = read_block_type_count(br)?;
    let mut sw_i = BlockSwitch::read(br, n_block_types_i)?;
    let n_block_types_d = read_block_type_count(br)?;
    let mut sw_d = BlockSwitch::read(br, n_block_types_d)?;

    let npostfix = br.take(2)?;
    let ndirect = br.take(4)? << npostfix;

    // A context mode per literal block type.
    let mut context_modes = Vec::with_capacity(n_block_types_l);
    for _ in 0..n_block_types_l {
        context_modes.push(br.take(2)? as u8);
    }

    let n_trees_l = read_block_type_count(br)?;
    let ctx_map_l = read_context_map(br, n_trees_l, n_block_types_l * 64)?;
    let n_trees_d = read_block_type_count(br)?;
    let ctx_map_d = read_context_map(br, n_trees_d, n_block_types_d * 4)?;

    let literal_trees = read_prefix_set(br, n_trees_l, 256)?;
    let iac_trees = read_prefix_set(br, n_block_types_i, 704)?;
    let dist_alphabet = 16 + ndirect as usize + (48usize << npostfix);

    let dist_trees = read_prefix_set(br, n_trees_d, dist_alphabet)?;

    let mut cur_l = 0usize;
    let mut cur_i = 0usize;
    let mut cur_d = 0usize;

    while out.len() < start + mlen {
        if let Some(sw) = &mut sw_i {
            cur_i = sw.step(br)?;
        }
        let iac_sym = iac_trees[cur_i].decode_maybe_empty(br)?;
        let (insert, copy, dist0) = insert_copy(br, iac_sym)?;

        // Insert `insert` literals.
        for _ in 0..insert {
            if let Some(sw) = &mut sw_l {
                cur_l = sw.step(br)?;
            }
            let p1 = out.last().copied().unwrap_or(0);
            let p2 = if out.len() >= 2 {
                out[out.len() - 2]
            } else {
                0
            };
            let ctx = literal_context(context_modes[cur_l], p1, p2);
            let tree = ctx_map_l[cur_l * 64 + ctx as usize] as usize;
            let lit = literal_trees[tree].decode_maybe_empty(br)? as u8;
            if out.len() >= cap {
                return Err(Error::TooLarge);
            }
            out.push(lit);
            if out.len() >= start + mlen {
                break;
            }
        }
        if out.len() >= start + mlen {
            break;
        }

        // Distance. `code0` marks a command that reuses the last distance and
        // must not disturb the ring — either the implicit distance-0 of the
        // insert-and-copy command, or distance symbol 0.
        let (distance, code0) = if dist0 {
            (dist_ring[(*dist_rb_idx).wrapping_sub(1) & 3], true)
        } else {
            if let Some(sw) = &mut sw_d {
                cur_d = sw.step(br)?;
            }
            let dctx = copy_distance_context(copy);
            let tree = ctx_map_d[cur_d * 4 + dctx] as usize;
            let dsym = dist_trees[tree].decode_maybe_empty(br)? as usize;
            decode_distance(br, dsym, npostfix, ndirect, dist_ring, *dist_rb_idx)?
        };

        // The largest backward reference is min(what has been produced, the
        // window minus the RFC's 16-byte gap); a distance beyond it is a
        // static-dictionary reference (§8), and neither dictionary refs nor
        // last-distance reuses touch the ring.
        let max_back = out.len().min(window - 16) as i64;
        if distance > max_back {
            copy_dictionary_word(out, distance, max_back, copy, cap)?;
        } else {
            if distance <= 0 {
                return Err(Error::Corrupt("backward distance out of range"));
            }
            if !code0 {
                dist_ring[*dist_rb_idx & 3] = distance;
                *dist_rb_idx = dist_rb_idx.wrapping_add(1);
            }
            let src = out.len() - distance as usize;
            for k in 0..copy as usize {
                if out.len() >= cap {
                    return Err(Error::TooLarge);
                }
                let b = out[src + k];
                out.push(b);
                if out.len() >= start + mlen {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn read_block_type_count(br: &mut Bits<'_>) -> Res<usize> {
    // NBLTYPES / NTREES code (§9.2): 1 + a value with a prefix code of its own.
    if br.bit1()? == 0 {
        return Ok(1);
    }
    let nbits = br.take(3)?;
    Ok(1 + (1usize << nbits) + br.take(nbits)? as usize)
}

fn read_prefix_set(br: &mut Bits<'_>, count: usize, alphabet: usize) -> Res<Vec<Prefix>> {
    let mut v = Vec::with_capacity(count.max(1));
    for _ in 0..count.max(1) {
        v.push(read_prefix(br, alphabet)?);
    }
    Ok(v)
}

/// Literal context id (§7.1): a function of the two previous bytes and the mode.
fn literal_context(mode: u8, p1: u8, p2: u8) -> u8 {
    match mode {
        0 => p1 & 0x3f,                                                         // LSB6
        1 => p1 >> 2,                                                           // MSB6
        2 => tbl::CTX_UTF8[p1 as usize] | tbl::CTX_UTF8[256 + p2 as usize],     // UTF8
        _ => tbl::CTX_SIGNED[p1 as usize] | tbl::CTX_SIGNED[256 + p2 as usize], // SIGNED
    }
}

/// Distance context (§7.2): the copy length capped at 4, minus 2.
fn copy_distance_context(copy: u32) -> usize {
    (copy.min(4) - 2) as usize
}

/// Decodes a distance symbol into `(distance, code0)`, where `code0` is true
/// only for symbol 0 — the plain "last distance" that leaves the ring untouched
/// (§4). The ring is *not* mutated here; the caller pushes, because whether a
/// distance joins the ring also depends on whether it turned out to be a
/// dictionary reference, which is not known until the distance is compared
/// against the output length.
fn decode_distance(
    br: &mut Bits<'_>,
    dsym: usize,
    npostfix: u32,
    ndirect: u32,
    ring: &[i64; 4],
    rb_idx: usize,
) -> Res<(i64, bool)> {
    // The first 16 symbols are relative to the last four distances (§4). ring0
    // is the most recent; the RFC's table is codes 1..3 → the older three
    // distances, and 4..15 → ring0/ring1 with a ±1/±2/±3 adjustment.
    if dsym < 16 {
        let idx = rb_idx.wrapping_sub(1);
        let ring0 = ring[idx & 3];
        let ring1 = ring[idx.wrapping_sub(1) & 3];
        let d = match dsym {
            0 => ring0,
            1 => ring1,
            2 => ring[idx.wrapping_sub(2) & 3],
            3 => ring[idx.wrapping_sub(3) & 3],
            4 => ring0 - 1,
            5 => ring0 + 1,
            6 => ring0 - 2,
            7 => ring0 + 2,
            8 => ring0 - 3,
            9 => ring0 + 3,
            10 => ring1 - 1,
            11 => ring1 + 1,
            12 => ring1 - 2,
            13 => ring1 + 2,
            14 => ring1 - 3,
            _ => ring1 + 3,
        };
        return Ok((d, dsym == 0));
    }
    let dsym = dsym as u32;
    if dsym < 16 + ndirect {
        // A "direct" distance: the code is the distance, offset past the ring.
        return Ok(((dsym - 16 + 1) as i64, false));
    }
    // Otherwise a (postfix, extra-bits) distance (§4).
    let dnorm = dsym - ndirect - 16;
    let nbits = 1 + (dnorm >> (npostfix + 1));
    let postfix_mask = (1u32 << npostfix) - 1;
    let hcode = (dnorm & postfix_mask) as i64;
    let lcode = (dnorm >> npostfix) as i64;
    let extra = br.take(nbits)? as i64;
    let offset = ((2 + (lcode & 1)) << nbits) - 4;
    let d = ((offset + extra) << npostfix) + hcode + ndirect as i64 + 1;
    Ok((d, false))
}

/// A static-dictionary reference (§8): the excess distance selects a word by
/// length and index, then a transform.
fn copy_dictionary_word(
    out: &mut Vec<u8>,
    distance: i64,
    max_back: i64,
    copy: u32,
    cap: usize,
) -> Res<()> {
    let len = copy as usize;
    if !(4..=24).contains(&len) {
        return Err(Error::Corrupt("dictionary word length out of range"));
    }
    let word_bits = tbl::SIZE_BITS[len] as u32;
    if word_bits == 0 {
        return Err(Error::Corrupt("no dictionary words of that length"));
    }
    let offset = tbl::OFFSETS[len] as usize;
    // The distance beyond the buffer selects (transform, word index).
    let value = (distance - max_back - 1) as u64;
    let index = (value & ((1u64 << word_bits) - 1)) as usize;
    let transform_id = (value >> word_bits) as usize;
    if transform_id >= tbl::TRANSFORMS.len() {
        return Err(Error::Corrupt("dictionary transform id out of range"));
    }
    let word = &tbl::DICTIONARY[offset + index * len..offset + (index + 1) * len];
    let transformed = apply_transform(word, transform_id);
    for b in transformed {
        if out.len() >= cap {
            return Err(Error::TooLarge);
        }
        out.push(b);
    }
    Ok(())
}

/// Applies RFC 7932 §8 transform `id` to `word`.
fn apply_transform(word: &[u8], id: usize) -> Vec<u8> {
    let (prefix_id, ttype, suffix_id) = tbl::TRANSFORMS[id];
    let prefix = prefix_suffix(prefix_id);
    let suffix = prefix_suffix(suffix_id);

    // Apply the core transform to the word body.
    let body: Vec<u8> = match ttype {
        0 => word.to_vec(), // identity
        1..=9 => {
            // omit last N
            let n = ttype as usize;
            word[..word.len().saturating_sub(n)].to_vec()
        }
        10 => uppercase(word, 1),          // uppercase first
        11 => uppercase(word, word.len()), // uppercase all
        12..=20 => {
            // omit first N
            let n = (ttype - 11) as usize;
            if n >= word.len() {
                Vec::new()
            } else {
                word[n..].to_vec()
            }
        }
        _ => word.to_vec(),
    };
    let mut out = Vec::with_capacity(prefix.len() + body.len() + suffix.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(&body);
    out.extend_from_slice(suffix);
    out
}

fn prefix_suffix(id: u8) -> &'static [u8] {
    // Each entry in PREFIX_SUFFIX is a length byte followed by that many data
    // bytes (brotli's kPrefixSuffix layout); the map holds the offset of the
    // length byte, so the string proper begins one byte in and runs for exactly
    // that length. Emitting the length byte itself is what put a stray control
    // character in front of every space-prefixed word.
    let start = tbl::PREFIX_SUFFIX_MAP[id as usize] as usize;
    let len = tbl::PREFIX_SUFFIX[start] as usize;
    &tbl::PREFIX_SUFFIX[start + 1..start + 1 + len]
}

/// The UTF-8-aware uppercase transform of §8: it walks bytes, upcasing the first
/// `count` "words" (multi-byte sequences count as one step), the way the RFC's
/// pseudo-code does.
fn uppercase(word: &[u8], count: usize) -> Vec<u8> {
    let mut out = word.to_vec();
    let mut i = 0usize;
    let mut done = 0usize;
    while i < out.len() && done < count {
        if out[i] < 0xC0 {
            // One byte (ASCII or a stray continuation byte): upper-case a
            // lowercase letter, leave everything else — brotli's cut is 0xC0,
            // not 0x80, so continuation bytes advance by one untouched.
            if out[i].is_ascii_lowercase() {
                out[i] ^= 32;
            }
            i += 1;
        } else if out[i] < 0xE0 {
            // Two-byte: RFC 7932's UTF-8 uppercase toggles bit 5 of byte 2.
            if i + 1 < out.len() {
                out[i + 1] ^= 32;
            }
            i += 2;
        } else {
            // Three-byte (and beyond): toggles bit 0 and 2 of byte 3.
            if i + 2 < out.len() {
                out[i + 2] ^= 5;
            }
            i += 3;
        }
        done += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Encoder: uncompressed meta-blocks only. Valid brotli, does not compress.
// ---------------------------------------------------------------------------

/// Encodes `data` as a brotli stream of uncompressed meta-blocks. libbrotli
/// decodes it; this build's decoder decodes it; it does not compress. See the
/// module comment for why that is the deliberate choice rather than a stub.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    // WBITS = 22 (the default window), encoded as the 4-bit "0011" + "..." form.
    // The simplest legal encoding: bit 1, then 3 bits = 5 -> WBITS 22.
    w.push(1, 1);
    w.push(5, 3); // 17 + 5 = 22
    let max = 1usize << 24; // MLEN is a 24-bit field carrying len-1.
                            // Every data chunk is a *non-last* uncompressed meta-block, and the stream
                            // is closed by an empty last block. That side-steps the rule that a last
                            // meta-block may not be flagged uncompressed (ISUNCOMPRESSED is only read on
                            // non-last blocks) without ever emitting a compressed body.
    for chunk in data.chunks(max) {
        w.push(0, 1); // ISLAST = 0
        let mlen = (chunk.len() - 1) as u32; // MLEN is stored as length-1.
                                             // MNIBBLES must be minimal (§9.2): the top nibble may not be zero once
                                             // more than four nibbles are used. Pick the fewest that hold `mlen`.
        let nnib = if mlen < (1 << 16) {
            4u32
        } else if mlen < (1 << 20) {
            5
        } else {
            6
        };
        w.push(nnib - 4, 2); // MNIBBLES selector 0/1/2 -> 4/5/6 nibbles
        for i in 0..nnib {
            w.push((mlen >> (4 * i)) & 0xf, 4);
        }
        w.push(1, 1); // ISUNCOMPRESSED
        w.align();
        for &b in chunk {
            w.push(b as u32, 8);
        }
    }
    // Empty last meta-block: ISLAST=1, ISLASTEMPTY=1.
    w.push(1, 1);
    w.push(1, 1);
    w.finish()
}

struct BitWriter {
    bytes: Vec<u8>,
    cur: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter {
            bytes: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    fn push(&mut self, v: u32, n: u32) {
        self.cur |= (v & ((1u32 << n) - 1)) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.bytes.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }

    fn align(&mut self) {
        if self.nbits > 0 {
            self.bytes.push((self.cur & 0xff) as u8);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dictionary_and_tables_are_the_right_size() {
        assert_eq!(tbl::DICTIONARY.len(), 122_784);
        assert_eq!(tbl::TRANSFORMS.len(), 121);
        // Length-24 is the longest dictionary word, so the words of that length
        // run to the end of the dictionary: OFFSETS[25] is the total size.
        assert_eq!(tbl::OFFSETS[24], 122_016);
        assert_eq!(tbl::OFFSETS[25], tbl::DICTIONARY.len() as u32);
        // Offsets are consistent with size_bits: words[len] occupy
        // (1 << size_bits[len]) * len bytes.
        for len in 4..=24usize {
            if tbl::SIZE_BITS[len] == 0 {
                continue;
            }
            let count = 1usize << tbl::SIZE_BITS[len];
            let span = tbl::OFFSETS[len + 1] as usize - tbl::OFFSETS[len] as usize;
            assert_eq!(span, count * len, "length {len}");
        }
    }

    #[test]
    fn the_encoder_round_trips_through_this_decoder() {
        for case in [
            &b""[..],
            b"a",
            b"hello world",
            &[0u8; 5000][..],
            &(0..=255u8).collect::<Vec<u8>>(),
        ] {
            let enc = compress(case);
            let dec = decompress(&enc, case.len().max(1)).expect("round trip");
            assert_eq!(dec, case, "on {} bytes", case.len());
        }
    }

    #[test]
    fn a_truncated_stream_is_an_error_not_a_panic() {
        let enc = compress(b"some bytes to encode and then cut short");
        for cut in 1..enc.len() {
            let _ = decompress(&enc[..cut], 1024); // must not panic
        }
    }
}
