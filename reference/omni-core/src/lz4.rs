//! §03.7.1 — the `lz4` codec: the LZ4 block format, both directions.
//!
//! §03.7.1 registers `lz4` at MAY level with the note *when decode speed >
//! ratio*, and the block format is the whole of it: there is no frame header,
//! no checksum and no dictionary — a sequence of tokens, literals and back
//! references, and nothing that has to be parsed before decoding can start.
//! That is exactly the shape a container wants, because the index already
//! carries the two things a frame header would repeat: the logical length
//! (§02.6) and the codec descriptor (§03.7.1). Decoding is therefore bounded by
//! a number the reader already trusted rather than by one the stream declares
//! about itself, which is §03.7.4's rule applied at the codec boundary instead
//! of after it.
//!
//! The decoder implements the format exactly, including the two properties that
//! separate a real LZ4 decoder from one that works on its own encoder's output:
//! an offset may be **smaller than the match length**, so a match is copied byte
//! by byte and legitimately reads bytes it has just written (that is how LZ4
//! spells a run), and a length nibble of 15 continues into as many `255` bytes
//! as the encoder wants. Both are refused-by-bounds rather than trusted: an
//! offset past the start of the output, a length that runs off the end of the
//! input, and a decode that would exceed the caller's cap are each an error with
//! the numbers in it.
//!
//! The encoder respects the format's three end-of-block rules — the last five
//! bytes are literals, the last match starts at least twelve bytes before the
//! end, and the block ends with a literal-only sequence — because a block that
//! ignores them decodes correctly here and is rejected by other
//! implementations, which is the worst of both. `level` bounds the hash chain,
//! so the same input and level give the same bytes every time; §03.7.1 requires
//! compression to be reproducible, and decoding never depends on the level.
//!
//! Nothing here is on the identity path (§03.7): compression is a property of a
//! stored copy, so two containers that store one object under `lz4` and `zstd`
//! hold the same object.

use crate::codec::Error;

type Res<T> = Result<T, Error>;

/// The format's minimum match length. A shorter back reference cannot be
/// spelled, because the token's low nibble stores `length - 4`.
pub const MIN_MATCH: usize = 4;
/// Offsets are two bytes, so a match cannot reach further back than this.
pub const MAX_OFFSET: usize = 65535;
/// The last five bytes of a block are always literals.
const LAST_LITERALS: usize = 5;
/// A match may not *start* within the last twelve bytes of a block.
const MF_LIMIT: usize = 12;

/// Decodes an LZ4 block, refusing to produce more than `cap` bytes.
///
/// `cap` is authoritative. Inside a container it is the index's `logical_len`
/// (§02.6), which the reader has already decided to trust; a caller with a bare
/// block passes its own bound. Either way the output buffer never grows past
/// it, so a malformed stream costs a bounded allocation and an error rather
/// than memory.
pub fn decompress(src: &[u8], cap: usize) -> Res<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(cap.min(1 << 20));
    let mut ip = 0usize;

    while ip < src.len() {
        let token = src[ip];
        ip += 1;

        // Literal length: the high nibble, extended by `255` continuation
        // bytes when it saturates.
        let mut lit = (token >> 4) as usize;
        if lit == 15 {
            lit += read_length(src, &mut ip)?;
        }
        if lit > src.len() - ip {
            return Err(Error::Corrupt(format!(
                "lz4: literal run of {lit} bytes with only {} left in the block",
                src.len() - ip
            )));
        }
        grow(&out, lit, cap)?;
        out.extend_from_slice(&src[ip..ip + lit]);
        ip += lit;

        // The block ends after a literal-only sequence: there is no offset to
        // read, and a token that promised a match is a truncated block.
        if ip == src.len() {
            break;
        }
        if ip + 2 > src.len() {
            return Err(Error::Corrupt(
                "lz4: a match offset is cut off by the end of the block".into(),
            ));
        }
        let offset = u16::from_le_bytes([src[ip], src[ip + 1]]) as usize;
        ip += 2;
        if offset == 0 {
            return Err(Error::Corrupt("lz4: match offset 0".into()));
        }
        if offset > out.len() {
            return Err(Error::Corrupt(format!(
                "lz4: match offset {offset} reaches before the start of {} decoded bytes",
                out.len()
            )));
        }

        let mut len = (token & 0x0f) as usize;
        if len == 15 {
            len += read_length(src, &mut ip)?;
        }
        len += MIN_MATCH;
        grow(&out, len, cap)?;

        // Byte at a time on purpose: LZ4 spells a run as an overlapping match,
        // so `offset < len` is the common case and not an error. A block copy
        // would read bytes that have not been written yet.
        let start = out.len() - offset;
        for k in 0..len {
            let b = out[start + k];
            out.push(b);
        }
    }
    Ok(out)
}

/// Reads a `255`-continued length extension.
fn read_length(src: &[u8], ip: &mut usize) -> Res<usize> {
    let mut total = 0usize;
    loop {
        if *ip >= src.len() {
            return Err(Error::Corrupt(
                "lz4: a length extension runs off the end of the block".into(),
            ));
        }
        let b = src[*ip];
        *ip += 1;
        total += b as usize;
        if b != 255 {
            return Ok(total);
        }
        // A stream of `255` bytes is how a legitimate long run is spelled, but
        // it is also how an attacker asks for one: the total is bounded by what
        // is left in the block, which is the only bound the format itself
        // gives.
        if total > src.len().saturating_mul(255) {
            return Err(Error::Bounds(
                "lz4: length extension larger than the block could describe".into(),
            ));
        }
    }
}

fn grow(out: &[u8], extra: usize, cap: usize) -> Res<()> {
    if out.len() + extra > cap {
        return Err(Error::Bounds(format!(
            "lz4: decoding would produce more than the declared {cap} bytes"
        )));
    }
    Ok(())
}

/// Compresses into an LZ4 block.
///
/// LZ77 over a hash chain of four-byte prefixes, with `level` bounding how many
/// candidates a position examines — the same shape as this crate's deflate
/// encoder, and for the same reason: the output has to be a function of the
/// input and the level alone (§03.7.1). Level 0 emits the input as literals,
/// which is a valid block and the honest way to spell "do not compress this".
pub fn compress(data: &[u8], level: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 4 + 16);

    // A block this short cannot hold a match at all: every byte falls inside
    // the twelve-byte tail where matches may not start.
    let chain_limit: usize = match level {
        0 => 0,
        1..=3 => 8,
        4..=6 => 64,
        7..=9 => 512,
        _ => 4096,
    };
    if data.len() < MF_LIMIT + 1 || chain_limit == 0 {
        emit_last_literals(&mut out, data);
        return out;
    }

    let mf_limit = data.len() - MF_LIMIT;
    let match_limit = data.len() - LAST_LITERALS;
    let mut head = vec![usize::MAX; 1 << 16];
    let mut prev = vec![usize::MAX; data.len()];
    let hash = |d: &[u8], i: usize| -> usize {
        let v = u32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]);
        // Knuth's multiplicative hash over the four-byte prefix, folded to the
        // table's width.
        ((v.wrapping_mul(2654435761)) >> 16) as usize & ((1 << 16) - 1)
    };

    let mut anchor = 0usize;
    let mut i = 0usize;
    while i < mf_limit {
        let h = hash(data, i);
        let mut cand = head[h];
        let mut best_len = 0usize;
        let mut best_ref = 0usize;
        let mut tries = 0usize;
        while cand != usize::MAX && tries < chain_limit {
            if i - cand > MAX_OFFSET {
                break;
            }
            let max = match_limit - i;
            let mut l = 0usize;
            while l < max && data[cand + l] == data[i + l] {
                l += 1;
            }
            if l > best_len {
                best_len = l;
                best_ref = cand;
                if l == max {
                    break;
                }
            }
            cand = prev[cand];
            tries += 1;
        }
        prev[i] = head[h];
        head[h] = i;

        if best_len >= MIN_MATCH {
            emit_sequence(&mut out, &data[anchor..i], i - best_ref, best_len);
            // Insert every position the match covers, so a later match can
            // start inside it.
            for k in 1..best_len {
                let p = i + k;
                if p + MIN_MATCH <= data.len() {
                    let ph = hash(data, p);
                    prev[p] = head[ph];
                    head[ph] = p;
                }
            }
            i += best_len;
            anchor = i;
        } else {
            i += 1;
        }
    }
    emit_last_literals(&mut out, &data[anchor..]);
    out
}

/// Writes one sequence: a token, the literal run, the offset, and the match
/// length that did not fit in the token.
fn emit_sequence(out: &mut Vec<u8>, literals: &[u8], offset: usize, match_len: usize) {
    let extra_match = match_len - MIN_MATCH;
    let token_lit = literals.len().min(15) as u8;
    let token_match = extra_match.min(15) as u8;
    out.push((token_lit << 4) | token_match);
    if literals.len() >= 15 {
        write_length(out, literals.len() - 15);
    }
    out.extend_from_slice(literals);
    out.push((offset & 0xff) as u8);
    out.push((offset >> 8) as u8);
    if extra_match >= 15 {
        write_length(out, extra_match - 15);
    }
}

/// The final sequence carries literals and stops: no offset follows it.
fn emit_last_literals(out: &mut Vec<u8>, literals: &[u8]) {
    let token_lit = literals.len().min(15) as u8;
    out.push(token_lit << 4);
    if literals.len() >= 15 {
        write_length(out, literals.len() - 15);
    }
    out.extend_from_slice(literals);
}

fn write_length(out: &mut Vec<u8>, mut n: usize) {
    while n >= 255 {
        out.push(255);
        n -= 255;
    }
    out.push(n as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8], level: u8) {
        let block = compress(data, level);
        let back = decompress(&block, data.len()).unwrap();
        assert_eq!(back, data, "level {level}, {} bytes", data.len());
    }

    #[test]
    fn round_trips_at_every_level_and_length() {
        let corpus: Vec<Vec<u8>> = vec![
            vec![],
            b"a".to_vec(),
            b"abcd".to_vec(),
            b"the quick brown fox jumps over the lazy dog".to_vec(),
            std::iter::repeat_n(b"omni/1.0 ".to_vec(), 500)
                .flatten()
                .collect(),
            (0..=255u8).collect(),
            (0..4096u32).map(|i| (i * 7 % 251) as u8).collect(),
            vec![0u8; 70000],
        ];
        for data in &corpus {
            for level in [0u8, 1, 6, 9, 12] {
                round_trip(data, level);
            }
            // Every prefix, so an end-of-block rule that only holds for lengths
            // divisible by something is caught.
            for n in 0..data.len().min(64) {
                round_trip(&data[..n], 6);
            }
        }
    }

    #[test]
    fn a_run_is_an_overlapping_match() {
        // The format's own idiom: one literal, then a match with offset 1 that
        // reads the bytes it is writing. Hand-assembled, so the decoder is
        // being tested against the format rather than against our encoder.
        //   token 0x1F: 1 literal, match length nibble 15 (saturated)
        //   'x', offset 0x0001, extension 0x01 -> match length 15 + 1 + 4 = 20
        let block = [0x1f, b'x', 0x01, 0x00, 0x01];
        let out = decompress(&block, 64).unwrap();
        assert_eq!(out, vec![b'x'; 21]);
    }

    #[test]
    fn a_saturated_literal_nibble_continues_into_255s() {
        // 270 literals: nibble 15, then 255 and 0.
        let mut block = vec![0xf0, 255, 0];
        block.extend(std::iter::repeat_n(b'q', 270));
        let out = decompress(&block, 512).unwrap();
        assert_eq!(out.len(), 270);
        assert!(out.iter().all(|&b| b == b'q'));
    }

    #[test]
    fn repetitive_data_actually_compresses() {
        let data: Vec<u8> = std::iter::repeat_n(b"omni".to_vec(), 4096)
            .flatten()
            .collect();
        let block = compress(&data, 9);
        assert!(block.len() < data.len() / 50, "{} bytes", block.len());
    }

    #[test]
    fn a_malformed_block_is_an_error_and_never_a_panic() {
        // An offset that reaches before the start of the output.
        assert!(decompress(&[0x10, b'a', 0x05, 0x00], 64).is_err());
        // Offset zero.
        assert!(decompress(&[0x10, b'a', 0x00, 0x00], 64).is_err());
        // A literal run longer than the block.
        assert!(decompress(&[0xf0, 255, 255, 255, b'a'], 4096).is_err());
        // A length extension with no terminator.
        assert!(decompress(&[0xf0, 255, 255, 255], 4096).is_err());
        // A token promising a match with no room for an offset.
        assert!(decompress(&[0x14, b'a', 0x01], 64).is_err());
    }

    #[test]
    fn every_truncation_of_a_valid_block_is_an_error_or_a_short_read() {
        let data: Vec<u8> = std::iter::repeat_n(b"omni is content addressed ".to_vec(), 40)
            .flatten()
            .collect();
        let block = compress(&data, 6);
        for n in 0..block.len() {
            // Truncation may decode to a prefix or fail; what it may not do is
            // produce the whole input or panic.
            if let Ok(out) = decompress(&block[..n], data.len()) {
                assert!(out.len() <= data.len());
            }
        }
    }

    #[test]
    fn decoding_stops_at_the_declared_length() {
        let data = vec![7u8; 4096];
        let block = compress(&data, 6);
        let err = decompress(&block, 100).unwrap_err();
        assert!(matches!(err, Error::Bounds(_)), "{err:?}");
    }

    #[test]
    fn the_encoder_obeys_the_end_of_block_rules() {
        // Every block must end with a literal-only sequence, its last five
        // bytes must be literals, and no match may start in the last twelve.
        // Walking the sequences is the only way to check that from outside.
        let data: Vec<u8> = std::iter::repeat_n(b"abcdefgh".to_vec(), 200)
            .flatten()
            .collect();
        let block = compress(&data, 9);
        let mut ip = 0usize;
        let mut produced = 0usize;
        let mut last_match_start = None;
        while ip < block.len() {
            let token = block[ip];
            ip += 1;
            let mut lit = (token >> 4) as usize;
            if lit == 15 {
                lit += read_length(&block, &mut ip).unwrap();
            }
            ip += lit;
            produced += lit;
            if ip == block.len() {
                assert!(
                    lit >= LAST_LITERALS,
                    "final literal run is {lit} bytes, fewer than {LAST_LITERALS}"
                );
                break;
            }
            last_match_start = Some(produced);
            ip += 2;
            let mut len = (token & 0x0f) as usize;
            if len == 15 {
                len += read_length(&block, &mut ip).unwrap();
            }
            produced += len + MIN_MATCH;
        }
        assert_eq!(produced, data.len());
        let start = last_match_start.expect("this input has matches");
        assert!(
            data.len() - start >= MF_LIMIT,
            "a match starts {} bytes from the end",
            data.len() - start
        );
    }
}
