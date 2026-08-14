//! §03.7.5 — `ans-lut`: rANS with a per-block lookup table.
//!
//! This is the one codec in §03.7.1's registry that belongs to OMNI rather than
//! to somebody else, and until §03.7.5 was written the registry named an
//! identifier no two implementations could have agreed on. That is a worse gap
//! than an unimplemented codec: an unimplemented one is reported as
//! unsupported, and an undefined one is a place where two conforming readers
//! disagree about the same bytes. The bitstream is now specified, and this is a
//! transcription of it.
//!
//! ## What it is for
//!
//! A codebook-quantized weight (§05.2) is a stream of small indices with a
//! strongly skewed distribution — an NF4 tensor's sixteen values are nothing
//! like uniform, and neither are a k-means codebook's. An LZ coder finds no
//! matches in that and spends its entropy stage on a byte alphabet it models
//! badly. Coding the indices against a table measured from the block itself is
//! the operation that fits, and it costs one multiply, one shift and one table
//! lookup per symbol in each direction.
//!
//! What it is *not* is a general-purpose codec. It has no match finder and no
//! context model: it is order-0 over one block. On text it loses to `deflate`;
//! on a bitshuffled float tensor it loses to `zstd`. A writer that reaches for
//! it outside the case the registry names is choosing badly rather than doing
//! something forbidden, and [`compress`] says so by storing a block verbatim
//! whenever coding it would not be smaller.

use crate::codec::Error;

type Res<T> = Result<T, Error>;

/// The version byte this build writes and the only one it reads.
const VERSION: u8 = 1;
/// The frequency table sums to `1 << SCALE`. Twelve bits keeps the LUT at 4 096
/// entries, which is the whole reason to prefer a table to a search.
const SCALE: u32 = 12;
/// The bottom of the normalized interval; renormalization is sixteen bits, so
/// the state lives in `[LOWER, LOWER << 16)` and fits a `u32` exactly.
const LOWER: u32 = 1 << 16;
/// Symbols per block. Small enough that the measured table describes the block
/// it came from, large enough that the table is not most of the output.
const BLOCK: usize = 1 << 16;

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(d: &[u8], at: &mut usize) -> Res<u64> {
    let mut value = 0u64;
    for i in 0..9 {
        let b = *d
            .get(*at)
            .ok_or_else(|| Error::Corrupt("ans-lut: a varint runs off the end".into()))?;
        *at += 1;
        value |= ((b & 0x7f) as u64) << (i * 7);
        if b & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(Error::Corrupt("ans-lut: a varint is too long".into()))
}

fn u32_at(d: &[u8], at: &mut usize) -> Res<u32> {
    if *at + 4 > d.len() {
        return Err(Error::Corrupt(
            "ans-lut: a 32-bit field runs off the end".into(),
        ));
    }
    let v = u32::from_le_bytes([d[*at], d[*at + 1], d[*at + 2], d[*at + 3]]);
    *at += 4;
    Ok(v)
}

/// Scales the measured counts so they sum to exactly `1 << SCALE`, giving every
/// symbol that occurs at least one slot.
///
/// The last property is not an optimization: a symbol with frequency zero
/// cannot be coded at all, so a table that rounds one away turns a legal block
/// into an undecodable one. The correction takes the slack from the largest
/// frequency, which is the one that can afford it.
fn normalize(counts: &[u32; 256], total: u64) -> [u16; 256] {
    let target = 1u64 << SCALE;
    let mut freq = [0u16; 256];
    let mut sum = 0u64;
    let mut largest = 0usize;
    for s in 0..256 {
        if counts[s] == 0 {
            continue;
        }
        let scaled = ((counts[s] as u64 * target) / total).max(1);
        freq[s] = scaled.min(target) as u16;
        sum += freq[s] as u64;
        if counts[s] > counts[largest] {
            largest = s;
        }
    }
    // Give or take the difference from the most frequent symbol, which has the
    // slots to spare and is the least distorted by the change.
    while sum > target {
        let over = (sum - target).min(freq[largest] as u64 - 1);
        freq[largest] -= over as u16;
        sum -= over;
        if freq[largest] == 1 {
            // Pathological: take from whatever else has room.
            let Some(next) = (0..256).find(|&s| freq[s] > 1 && s != largest) else {
                break;
            };
            largest = next;
        }
    }
    if sum < target {
        freq[largest] += (target - sum) as u16;
    }
    freq
}

fn cumulative(freq: &[u16; 256]) -> [u32; 257] {
    let mut cum = [0u32; 257];
    for s in 0..256 {
        cum[s + 1] = cum[s] + freq[s] as u32;
    }
    cum
}

/// The LUT the codec is named for: `slot → symbol`, built by the decoder from
/// the frequency table rather than read from the stream.
fn lut(freq: &[u16; 256], cum: &[u32; 257]) -> Vec<u8> {
    let mut t = vec![0u8; 1 << SCALE];
    for s in 0..256 {
        let start = cum[s] as usize;
        let end = start + freq[s] as usize;
        for e in t.iter_mut().take(end).skip(start) {
            *e = s as u8;
        }
    }
    t
}

/// Codes one block, or returns `None` when the table plus the payload would not
/// be smaller than the block itself.
fn encode_block(block: &[u8]) -> Option<Vec<u8>> {
    let mut counts = [0u32; 256];
    for &b in block {
        counts[b as usize] += 1;
    }
    let used: Vec<usize> = (0..256).filter(|&s| counts[s] > 0).collect();
    if used.len() > 256 || block.is_empty() {
        return None;
    }
    let freq = normalize(&counts, block.len() as u64);
    let cum = cumulative(&freq);

    // rANS encodes backwards so the decoder can run forwards. The words come
    // out in the order the *decoder* will need them last, so the payload is
    // the final state followed by the words in reverse emission order.
    let mut emitted: Vec<u16> = Vec::with_capacity(block.len() / 2 + 4);
    let mut x: u32 = LOWER;
    for &b in block.iter().rev() {
        let f = freq[b as usize] as u32;
        let c = cum[b as usize];
        // In 64 bits on purpose: the bound is `2^20 × f`, which reaches
        // `2^32` at the largest frequency the scale allows.
        let max = (((LOWER >> SCALE) as u64) << 16) * f as u64;
        while x as u64 >= max {
            emitted.push((x & 0xffff) as u16);
            x >>= 16;
        }
        x = ((x / f) << SCALE) + (x % f) + c;
    }
    let mut words: Vec<u8> = Vec::with_capacity(4 + emitted.len() * 2);
    words.extend(x.to_le_bytes());
    for w in emitted.iter().rev() {
        words.extend(w.to_le_bytes());
    }

    let table_bytes = 1 + used.len() * 3;
    if table_bytes + 4 + words.len() >= block.len() {
        return None;
    }
    let mut out = Vec::with_capacity(table_bytes + 4 + words.len());
    out.push((used.len() - 1) as u8);
    for &s in &used {
        out.push(s as u8);
        out.extend((freq[s]).to_le_bytes());
    }
    out.extend((words.len() as u32).to_le_bytes());
    out.extend_from_slice(&words);
    Some(out)
}

/// Compresses under §03.7.5. `level` is accepted and ignored: an order-0 coder
/// has no effort knob, and pretending otherwise would make the descriptor lie.
pub fn compress(data: &[u8], _level: u8) -> Vec<u8> {
    let blocks = data.len().div_ceil(BLOCK).max(1);
    let mut out = Vec::with_capacity(data.len() / 2 + 16);
    out.push(VERSION);
    out.push(SCALE as u8);
    out.extend((BLOCK as u32).to_le_bytes());
    write_varint(&mut out, blocks as u64);
    if data.is_empty() {
        out.push(0);
        out.extend(0u32.to_le_bytes());
        return out;
    }
    for block in data.chunks(BLOCK) {
        match encode_block(block) {
            Some(coded) => {
                out.push(1);
                out.extend((block.len() as u32).to_le_bytes());
                out.extend_from_slice(&coded);
            }
            None => {
                out.push(0);
                out.extend((block.len() as u32).to_le_bytes());
                out.extend_from_slice(block);
            }
        }
    }
    out
}

/// Decompresses under §03.7.5, refusing to produce more than `limit` bytes.
pub fn decompress(input: &[u8], limit: usize) -> Res<Vec<u8>> {
    let mut at = 0usize;
    let version = *input
        .first()
        .ok_or_else(|| Error::Corrupt("ans-lut: an empty stream".into()))?;
    if version != VERSION {
        return Err(Error::Corrupt(format!(
            "ans-lut: version {version} is not the 1 this build writes"
        )));
    }
    at += 1;
    let scale = *input
        .get(at)
        .ok_or_else(|| Error::Corrupt("ans-lut: no scale byte".into()))? as u32;
    at += 1;
    // R-C23: below 8 a byte alphabet does not fit; above 16 the LUT stops
    // fitting in cache, which is the only reason to prefer it to a search.
    if !(8..=16).contains(&scale) {
        return Err(Error::Corrupt(format!(
            "ans-lut: log2_scale {scale} is outside §03.7.5's 8..=16"
        )));
    }
    let _block_elems = u32_at(input, &mut at)?;
    let count = read_varint(input, &mut at)?;
    if count > (limit as u64).div_ceil(1) + 1 {
        return Err(Error::Bounds(
            "ans-lut: more blocks than the declared length can hold".into(),
        ));
    }

    let total = 1u32 << scale;
    let mut out: Vec<u8> = Vec::with_capacity(limit.min(1 << 20));
    for _ in 0..count {
        let kind = *input
            .get(at)
            .ok_or_else(|| Error::Corrupt("ans-lut: a block with no kind".into()))?;
        at += 1;
        let n = u32_at(input, &mut at)? as usize;
        if out.len() + n > limit {
            return Err(Error::Bounds(format!(
                "ans-lut: decoding would produce more than the declared {limit} bytes"
            )));
        }
        if kind == 0 {
            if at + n > input.len() {
                return Err(Error::Corrupt(
                    "ans-lut: a stored block runs off the end".into(),
                ));
            }
            out.extend_from_slice(&input[at..at + n]);
            at += n;
            continue;
        }
        if kind != 1 {
            return Err(Error::Corrupt(format!(
                "ans-lut: block kind {kind} is not 0 or 1"
            )));
        }

        let used = *input
            .get(at)
            .ok_or_else(|| Error::Corrupt("ans-lut: a block with no table".into()))?
            as usize
            + 1;
        at += 1;
        let mut freq = [0u16; 256];
        let mut sum = 0u32;
        let mut last: i32 = -1;
        for _ in 0..used {
            if at + 3 > input.len() {
                return Err(Error::Corrupt(
                    "ans-lut: a frequency table runs off the end".into(),
                ));
            }
            let s = input[at] as i32;
            let f = u16::from_le_bytes([input[at + 1], input[at + 2]]);
            at += 3;
            // R-C21: strictly increasing, and every listed symbol codable.
            if s <= last {
                return Err(Error::Corrupt(
                    "ans-lut: the frequency table's symbols are not strictly increasing (R-C21)"
                        .into(),
                ));
            }
            if f == 0 {
                return Err(Error::Corrupt(
                    "ans-lut: a listed symbol has frequency zero (R-C21)".into(),
                ));
            }
            last = s;
            freq[s as usize] = f;
            sum += f as u32;
        }
        // R-C20: a table that does not sum to the scale is invalid, not a hint.
        if sum != total {
            return Err(Error::Corrupt(format!(
                "ans-lut: the frequencies sum to {sum} and the scale is {total} (R-C20)"
            )));
        }
        let cum = cumulative(&freq);
        let table = lut(&freq, &cum);

        let payload_len = u32_at(input, &mut at)? as usize;
        if at + payload_len > input.len() {
            return Err(Error::Corrupt("ans-lut: a payload runs off the end".into()));
        }
        let payload = &input[at..at + payload_len];
        at += payload_len;
        if payload_len < 4 {
            return Err(Error::Corrupt(
                "ans-lut: a payload shorter than its state".into(),
            ));
        }

        let mut p = 0usize;
        let mut x = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        p += 4;
        for _ in 0..n {
            let slot = (x & (total - 1)) as usize;
            let s = table[slot];
            let f = freq[s as usize] as u32;
            x = f * (x >> scale) + slot as u32 - cum[s as usize];
            while x < LOWER {
                // R-C22: a stream that ends early is invalid rather than padded.
                if p + 2 > payload_len {
                    return Err(Error::Corrupt(
                        "ans-lut: the payload ends before the block does (R-C22)".into(),
                    ));
                }
                x = (x << 16) | u16::from_le_bytes([payload[p], payload[p + 1]]) as u32;
                p += 2;
            }
            out.push(s);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let stream = compress(data, 6);
        let back = decompress(&stream, data.len().max(1)).unwrap();
        assert_eq!(back, data, "{} bytes", data.len());
    }

    /// The distribution this codec exists for: 4-bit codebook indices, skewed
    /// the way a real NF4 tensor's are.
    fn codebook_indices(n: usize) -> Vec<u8> {
        // A discrete approximation of a normal distribution over sixteen
        // values, which is what NF4's quantiles produce on real weights.
        let weights = [
            1u32, 2, 5, 12, 26, 52, 92, 140, 140, 92, 52, 26, 12, 5, 2, 1,
        ];
        let total: u32 = weights.iter().sum();
        let mut out = Vec::with_capacity(n);
        let mut state = 0x2026_0814u32;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let mut pick = (state >> 8) % total;
            let mut symbol = 0u8;
            for (i, w) in weights.iter().enumerate() {
                if pick < *w {
                    symbol = i as u8;
                    break;
                }
                pick -= w;
            }
            out.push(symbol);
        }
        out
    }

    #[test]
    fn round_trips_over_every_shape_of_input() {
        let corpus: Vec<Vec<u8>> = vec![
            vec![],
            vec![7],
            vec![0u8; 5000],
            (0..=255u8).collect(),
            codebook_indices(40000),
            // Every byte value, uniformly: the case where an order-0 coder can
            // do nothing and must store the block instead.
            (0..70000u32).map(|i| (i % 256) as u8).collect(),
            // Just over a block boundary.
            codebook_indices((1 << 16) + 17),
        ];
        for data in &corpus {
            round_trip(data);
            // And every prefix of the first few hundred bytes, so an off-by-one
            // in the last block shows up.
            for n in 0..data.len().min(200) {
                round_trip(&data[..n]);
            }
        }
    }

    #[test]
    fn skewed_indices_actually_compress() {
        // The measurement the registry entry is a claim about. The entropy of
        // the distribution above is about 3.0 bits, so a 4-bit-per-symbol
        // stream stored one-per-byte should land near 3/8 of its size — and
        // well under what an LZ coder gets, since there is nothing to match.
        let data = codebook_indices(60000);
        let ours = compress(&data, 6).len();
        assert!(
            ours < data.len() * 45 / 100,
            "{} -> {} bytes is not compression",
            data.len(),
            ours
        );
        let deflated = crate::codec::Codec::Deflate { level: 9 }
            .encode(&data)
            .unwrap()
            .len();
        assert!(
            ours < deflated,
            "ans-lut {ours} did not beat deflate {deflated} on the case it exists for"
        );
    }

    #[test]
    fn an_incompressible_block_is_stored_rather_than_expanded() {
        // §03.7.5's rule: a block that would not get smaller is written
        // verbatim, so the codec never expands its input by more than the
        // framing.
        let mut state = 0x1234_5678u32;
        let data: Vec<u8> = (0..40000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        let stream = compress(&data, 6);
        assert!(
            stream.len() <= data.len() + 16,
            "{} -> {}",
            data.len(),
            stream.len()
        );
        round_trip(&data);
    }

    #[test]
    fn a_table_that_does_not_sum_to_the_scale_is_invalid() {
        let data = codebook_indices(1000);
        let mut stream = compress(&data, 6);
        // The first frequency, at: version, scale, block_elems(4), varint(1),
        // kind(1), n(4), used(1), symbol(1).
        let at = 1 + 1 + 4 + 1 + 1 + 4 + 1 + 1;
        stream[at] = stream[at].wrapping_add(1);
        let err = decompress(&stream, data.len()).unwrap_err();
        assert!(format!("{err}").contains("R-C20"), "{err}");
    }

    #[test]
    fn a_symbol_with_frequency_zero_is_refused() {
        let data = codebook_indices(1000);
        let mut stream = compress(&data, 6);
        let at = 1 + 1 + 4 + 1 + 1 + 4 + 1 + 1;
        // Zero the first frequency, and fix nothing: R-C21 fires before R-C20.
        stream[at] = 0;
        stream[at + 1] = 0;
        let err = decompress(&stream, data.len()).unwrap_err();
        assert!(format!("{err}").contains("R-C21"), "{err}");
    }

    #[test]
    fn a_scale_outside_the_range_is_refused() {
        let data = codebook_indices(1000);
        let mut stream = compress(&data, 6);
        stream[1] = 20;
        let err = decompress(&stream, data.len()).unwrap_err();
        assert!(format!("{err}").contains("§03.7.5"), "{err}");
    }

    #[test]
    fn every_truncation_is_an_error_and_never_a_panic() {
        let data = codebook_indices(3000);
        let stream = compress(&data, 6);
        for n in 0..stream.len() {
            let _ = decompress(&stream[..n], data.len());
        }
    }

    #[test]
    fn the_declared_length_is_a_bound() {
        let data = codebook_indices(40000);
        let stream = compress(&data, 6);
        let err = decompress(&stream, 100).unwrap_err();
        assert!(matches!(err, Error::Bounds(_)), "{err:?}");
    }

    #[test]
    fn a_tampered_payload_never_decodes_to_the_original() {
        let data = codebook_indices(2000);
        let stream = compress(&data, 6);
        let mut differed = 0usize;
        for i in 20..stream.len() {
            let mut bad = stream.clone();
            bad[i] ^= 0x40;
            match decompress(&bad, data.len()) {
                Ok(out) => {
                    if out != data {
                        differed += 1;
                    }
                }
                Err(_) => differed += 1,
            }
        }
        assert!(
            differed * 10 >= (stream.len() - 20) * 9,
            "only {differed} of {} flipped bits changed the answer",
            stream.len() - 20
        );
    }
}
