//! Keccak-f[1600] and the two SHAKE extendable-output functions.
//!
//! SHA-2 and Keccak are unrelated constructions, so `sha256.rs` and `sha512.rs`
//! are no help here: a sponge is not a Merkle–Damgård chain and shares no code
//! with one. This exists because FIPS 204 and FIPS 205 both need it — ML-DSA for
//! its samplers and SLH-DSA for every hash it performs — and a primitive with two
//! callers belongs in one place.
//!
//! Output is squeezed incrementally rather than in one shot. ML-DSA's rejection
//! samplers need that: they cannot know in advance how many bytes they will
//! consume, because that is what rejection means.
//!
//! Checked against the published SHAKE outputs for the empty string, which is
//! the only test here that could catch a wrong permutation before it becomes a
//! wrong signature two modules away.

const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

fn keccak_f1600(a: &mut [u64; 25]) {
    for rc in RC {
        // theta
        let mut c = [0u64; 5];
        for (x, slot) in c.iter_mut().enumerate() {
            *slot = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // rho and pi, walked as one permutation cycle starting at lane 1
        let mut last = a[1];
        for i in 0..24 {
            let j = PI[i];
            let tmp = a[j];
            a[j] = last.rotate_left(RHO[i]);
            last = tmp;
        }
        // chi
        for y in 0..5 {
            let row = [
                a[5 * y],
                a[5 * y + 1],
                a[5 * y + 2],
                a[5 * y + 3],
                a[5 * y + 4],
            ];
            for x in 0..5 {
                a[x + 5 * y] = row[x] ^ ((!row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }
        // iota
        a[0] ^= rc;
    }
}

/// A SHAKE extendable-output function. Everything is absorbed at construction —
/// every caller here has its whole input in hand — and output is squeezed
/// incrementally, which the rejection samplers need because they cannot know in
/// advance how many bytes they will consume.
pub struct Xof {
    st: [u64; 25],
    rate: usize,
    pos: usize,
}

impl Xof {
    pub fn new(rate: usize, parts: &[&[u8]]) -> Xof {
        let mut st = [0u64; 25];
        let mut i = 0usize;
        for part in parts {
            for &b in *part {
                st[i / 8] ^= (b as u64) << (8 * (i % 8));
                i += 1;
                if i == rate {
                    keccak_f1600(&mut st);
                    i = 0;
                }
            }
        }
        // pad10*1 with the SHAKE domain separator
        st[i / 8] ^= 0x1f << (8 * (i % 8));
        st[(rate - 1) / 8] ^= 0x80 << (8 * ((rate - 1) % 8));
        keccak_f1600(&mut st);
        Xof { st, rate, pos: 0 }
    }

    pub fn shake128(parts: &[&[u8]]) -> Xof {
        Xof::new(168, parts)
    }

    pub fn shake256(parts: &[&[u8]]) -> Xof {
        Xof::new(136, parts)
    }

    pub fn squeeze(&mut self, out: &mut [u8]) {
        for byte in out.iter_mut() {
            if self.pos == self.rate {
                keccak_f1600(&mut self.st);
                self.pos = 0;
            }
            *byte = (self.st[self.pos / 8] >> (8 * (self.pos % 8))) as u8;
            self.pos += 1;
        }
    }

    pub fn squeeze_vec(&mut self, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        self.squeeze(&mut v);
        v
    }
}

/// SHAKE256 with a requested output length — FIPS 204's `H` and, for the SHAKE
/// parameter sets, every one of FIPS 205's six hash functions.
pub fn shake256(parts: &[&[u8]], n: usize) -> Vec<u8> {
    Xof::shake256(parts).squeeze_vec(n)
}

/// SHAKE128 with a requested output length — FIPS 204's `G`.
pub fn shake128(parts: &[&[u8]], n: usize) -> Vec<u8> {
    Xof::shake128(parts).squeeze_vec(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The permutation is what everything above it stands on, so it is checked
    /// against the published outputs for the empty string before any algorithm
    /// that uses it is looked at. A wrong permutation makes every later failure
    /// impossible to localise.
    #[test]
    fn shake_matches_the_published_empty_string_outputs() {
        assert_eq!(
            shake128(&[b""], 32),
            hex("7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26")
        );
        assert_eq!(
            shake256(&[b""], 32),
            hex("46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f")
        );
    }

    #[test]
    fn shake_squeezes_across_a_rate_boundary() {
        // Squeezing in one call and in pieces must agree, which is the property
        // the rejection samplers depend on and the one an off-by-one in `pos`
        // would break only after the first block.
        let whole = shake128(&[b"omni"], 400);
        let mut x = Xof::shake128(&[b"omni"]);
        let mut pieces = Vec::new();
        for n in [1usize, 7, 160, 32, 200] {
            pieces.extend_from_slice(&x.squeeze_vec(n));
        }
        assert_eq!(whole, pieces);
    }

    /// Absorbing in one part or several must be identical: SLH-DSA's hashes are
    /// naturally written as several pieces (`PK.seed`, an address, a message) and
    /// concatenating them first must not be necessary.
    #[test]
    fn absorbing_in_pieces_is_absorbing_the_concatenation() {
        let a = shake256(&[b"abc", b"defgh", b"ijklmnop"], 64);
        let b = shake256(&[b"abcdefghijklmnop"], 64);
        assert_eq!(a, b);
    }
}
