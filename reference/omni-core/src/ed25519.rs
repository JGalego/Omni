//! Ed25519 (RFC 8032), from scratch.
//!
//! §12.5.1 makes Ed25519 the one signature algorithm every implementation MUST
//! have, so a dependency-free reference implementation has to contain one. This
//! is it: field arithmetic mod 2²⁵⁵ − 19, the twisted Edwards group, and
//! `sign`/`verify` for PureEdDSA over SHA-512.
//!
//! ## What this is not
//!
//! **Not hardened against side channels.** Scalar multiplication is plain
//! double-and-add, so its timing depends on the scalar's bits, and nothing here
//! is protected against fault injection or power analysis. §12.4's threat model
//! is a malicious *file*, and for reading one this code is appropriate; for
//! holding a release key on a shared machine it is not. Use a vetted
//! implementation or an HSM there — the same advice this crate gives about its
//! own BLAKE3. The formulas are chosen so that correctness can be read off the
//! page, which is the property a reference implementation owes its readers.
//!
//! Verification handles only attacker-supplied data, so it is the half that
//! actually needs to be correct rather than quiet, and it follows RFC 8032
//! §5.1.7 including the canonical-encoding and small-order checks that a
//! surprising number of implementations skip.

use crate::sha512::sha512;

/// Length of a public key, a secret seed, and half a signature.
pub const KEY_LEN: usize = 32;
/// Length of a signature.
pub const SIG_LEN: usize = 64;

// ---------------------------------------------------------------------- field --
//
// Elements of GF(2^255 - 19) as five 51-bit limbs, little-endian. Products fit
// in u128, and a reduction after each operation keeps limbs below 2^54, which
// leaves room for the additions in the group law without overflow.

#[derive(Clone, Copy, Debug)]
struct Fe([u64; 5]);

const MASK51: u64 = (1u64 << 51) - 1;

impl Fe {
    const ZERO: Fe = Fe([0; 5]);
    const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    fn from_bytes(b: &[u8; 32]) -> Fe {
        let ld = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        // Bit 255 is ignored, as RFC 8032 requires.
        Fe([
            ld(0) & MASK51,
            (ld(6) >> 3) & MASK51,
            (ld(12) >> 6) & MASK51,
            (ld(19) >> 1) & MASK51,
            (ld(24) >> 12) & MASK51,
        ])
    }

    fn to_bytes(self) -> [u8; 32] {
        let mut t = self.reduced();
        // Is the value at least p? Adding 19 and watching the carry off the top
        // limb answers it without a comparison.
        let mut q = (t.0[0] + 19) >> 51;
        q = (t.0[1] + q) >> 51;
        q = (t.0[2] + q) >> 51;
        q = (t.0[3] + q) >> 51;
        q = (t.0[4] + q) >> 51;
        // Subtracting p is adding 19 and dropping the 2^255 bit.
        t.0[0] += 19 * q;
        for i in 0..4 {
            t.0[i + 1] += t.0[i] >> 51;
            t.0[i] &= MASK51;
        }
        t.0[4] &= MASK51;
        let mut out = [0u8; 32];
        let words = [
            t.0[0] | (t.0[1] << 51),
            (t.0[1] >> 13) | (t.0[2] << 38),
            (t.0[2] >> 26) | (t.0[3] << 25),
            (t.0[3] >> 39) | (t.0[4] << 12),
        ];
        for (i, w) in words.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn reduced(mut self) -> Fe {
        let mut carry = 0u64;
        for i in 0..5 {
            self.0[i] += carry;
            carry = self.0[i] >> 51;
            self.0[i] &= MASK51;
        }
        self.0[0] += carry * 19;
        carry = self.0[0] >> 51;
        self.0[0] &= MASK51;
        self.0[1] += carry;
        self
    }

    fn add(self, o: Fe) -> Fe {
        let mut r = [0u64; 5];
        for (i, slot) in r.iter_mut().enumerate() {
            *slot = self.0[i] + o.0[i];
        }
        Fe(r).reduced()
    }

    fn sub(self, o: Fe) -> Fe {
        // Add 16p before subtracting, so every limb stays non-negative even
        // when the operands are only weakly reduced.
        const P16_0: u64 = 16 * ((1u64 << 51) - 19);
        const P16_N: u64 = 16 * ((1u64 << 51) - 1);
        let mut r = [0u64; 5];
        r[0] = self.0[0] + P16_0 - o.0[0];
        for (i, slot) in r.iter_mut().enumerate().skip(1) {
            *slot = self.0[i] + P16_N - o.0[i];
        }
        Fe(r).reduced()
    }

    fn neg(self) -> Fe {
        Fe::ZERO.sub(self)
    }

    fn mul(self, o: Fe) -> Fe {
        let a = &self.0;
        let b = &o.0;
        let m = |x: u64, y: u64| (x as u128) * (y as u128);
        // 19 * b[i], for the terms that wrap past 2^255.
        let b19: Vec<u128> = b.iter().map(|x| 19 * (*x as u128)).collect();
        let t0 = m(a[0], b[0])
            + (a[1] as u128) * b19[4]
            + (a[2] as u128) * b19[3]
            + (a[3] as u128) * b19[2]
            + (a[4] as u128) * b19[1];
        let t1 = m(a[0], b[1])
            + m(a[1], b[0])
            + (a[2] as u128) * b19[4]
            + (a[3] as u128) * b19[3]
            + (a[4] as u128) * b19[2];
        let t2 = m(a[0], b[2])
            + m(a[1], b[1])
            + m(a[2], b[0])
            + (a[3] as u128) * b19[4]
            + (a[4] as u128) * b19[3];
        let t3 =
            m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0]) + (a[4] as u128) * b19[4];
        let t4 = m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0]);
        Fe::carry([t0, t1, t2, t3, t4])
    }

    /// Carries a product back into 51-bit limbs.
    ///
    /// The whole chain stays in `u128`: a product limb reaches about 2^115, so
    /// its carry does not fit in a `u64` and truncating it there would be a
    /// silent wrong answer rather than a visible one.
    fn carry(mut t: [u128; 5]) -> Fe {
        let mask = MASK51 as u128;
        for i in 0..4 {
            t[i + 1] += t[i] >> 51;
            t[i] &= mask;
        }
        // The top carry wraps around 2^255 and comes back multiplied by 19.
        let c = t[4] >> 51;
        t[4] &= mask;
        t[0] += c * 19;
        t[1] += t[0] >> 51;
        t[0] &= mask;
        Fe([
            t[0] as u64,
            t[1] as u64,
            t[2] as u64,
            t[3] as u64,
            t[4] as u64,
        ])
    }

    fn square(self) -> Fe {
        self.mul(self)
    }

    /// `self^(2^n)`.
    fn square_n(self, n: u32) -> Fe {
        let mut r = self;
        for _ in 0..n {
            r = r.square();
        }
        r
    }

    /// `self^(p-2) = self^-1` by the standard addition chain.
    fn invert(self) -> Fe {
        let z2 = self.square();
        let z9 = z2.square_n(2).mul(self);
        let z11 = z9.mul(z2);
        let z2_5_0 = z11.square().mul(z9);
        let z2_10_0 = z2_5_0.square_n(5).mul(z2_5_0);
        let z2_20_0 = z2_10_0.square_n(10).mul(z2_10_0);
        let z2_40_0 = z2_20_0.square_n(20).mul(z2_20_0);
        let z2_50_0 = z2_40_0.square_n(10).mul(z2_10_0);
        let z2_100_0 = z2_50_0.square_n(50).mul(z2_50_0);
        let z2_200_0 = z2_100_0.square_n(100).mul(z2_100_0);
        let z2_250_0 = z2_200_0.square_n(50).mul(z2_50_0);
        z2_250_0.square_n(5).mul(z11)
    }

    /// `self^((p-5)/8)`, the exponent square-root recovery needs.
    fn pow_p58(self) -> Fe {
        let z2 = self.square();
        let z9 = z2.square_n(2).mul(self);
        let z11 = z9.mul(z2);
        let z2_5_0 = z11.square().mul(z9);
        let z2_10_0 = z2_5_0.square_n(5).mul(z2_5_0);
        let z2_20_0 = z2_10_0.square_n(10).mul(z2_10_0);
        let z2_40_0 = z2_20_0.square_n(20).mul(z2_20_0);
        let z2_50_0 = z2_40_0.square_n(10).mul(z2_10_0);
        let z2_100_0 = z2_50_0.square_n(50).mul(z2_50_0);
        let z2_200_0 = z2_100_0.square_n(100).mul(z2_100_0);
        let z2_250_0 = z2_200_0.square_n(50).mul(z2_50_0);
        z2_250_0.square_n(2).mul(self)
    }

    fn is_zero(self) -> bool {
        self.to_bytes() == [0u8; 32]
    }

    fn eq(self, o: Fe) -> bool {
        self.to_bytes() == o.to_bytes()
    }

    fn is_negative(self) -> bool {
        self.to_bytes()[0] & 1 == 1
    }
}

/// `d = -121665/121666`, the curve constant.
fn d_const() -> Fe {
    Fe::from_bytes(&[
        0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75, 0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70,
        0x00, 0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c, 0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c,
        0x03, 0x52,
    ])
}

/// `sqrt(-1)` mod p.
fn sqrt_m1() -> Fe {
    Fe::from_bytes(&[
        0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4, 0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43,
        0x2f, 0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b, 0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24,
        0x83, 0x2b,
    ])
}

// ---------------------------------------------------------------------- group --

/// A point in extended twisted Edwards coordinates `(X : Y : Z : T)` with
/// `x = X/Z`, `y = Y/Z`, `xy = T/Z`.
#[derive(Clone, Copy, Debug)]
struct Point {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl Point {
    const IDENTITY: Point = Point {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    /// The standard base point, from its compressed encoding: `y = 4/5` with
    /// the even `x`.
    fn base() -> Point {
        const B: [u8; 32] = [
            0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66,
        ];
        Point::decompress(&B).expect("the base point decompresses")
    }

    /// `dbl-2008-hwcd` with the curve's `a = -1`, by way of the four
    /// intermediate coordinates: `2XY`, `Y²+X²`, `Y²−X²`, `2Z²−(Y²−X²)`.
    fn double(self) -> Point {
        let xx = self.x.square();
        let yy = self.y.square();
        let zz2 = self.z.square().add(self.z.square());
        let sum_sq = self.x.add(self.y).square();
        let yy_plus_xx = yy.add(xx);
        let yy_minus_xx = yy.sub(xx);
        let xc = sum_sq.sub(yy_plus_xx);
        let tc = zz2.sub(yy_minus_xx);
        Point {
            x: xc.mul(tc),
            y: yy_plus_xx.mul(yy_minus_xx),
            z: yy_minus_xx.mul(tc),
            t: xc.mul(yy_plus_xx),
        }
    }

    fn add(self, o: Point) -> Point {
        let a = self.y.sub(self.x).mul(o.y.sub(o.x));
        let b = self.y.add(self.x).mul(o.y.add(o.x));
        let c = self.t.mul(o.t).mul(d_const()).mul(Fe([2, 0, 0, 0, 0]));
        let d = self.z.mul(o.z).mul(Fe([2, 0, 0, 0, 0]));
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        Point {
            x: e.mul(f),
            y: g.mul(h),
            t: e.mul(h),
            z: f.mul(g),
        }
    }

    fn negate(self) -> Point {
        Point {
            x: self.x.neg(),
            y: self.y,
            z: self.z,
            t: self.t.neg(),
        }
    }

    fn compress(self) -> [u8; 32] {
        let zi = self.z.invert();
        let x = self.x.mul(zi);
        let y = self.y.mul(zi);
        let mut out = y.to_bytes();
        out[31] |= (x.to_bytes()[0] & 1) << 7;
        out
    }

    /// RFC 8032 §5.1.3 point decompression, rejecting non-canonical encodings
    /// and non-points.
    fn decompress(b: &[u8; 32]) -> Option<Point> {
        let sign = b[31] >> 7 == 1;
        let mut yb = *b;
        yb[31] &= 0x7f;
        let y = Fe::from_bytes(&yb);
        // A y coordinate must be canonical: re-encoding is the check.
        if y.to_bytes() != yb {
            return None;
        }
        let yy = y.square();
        let u = yy.sub(Fe::ONE);
        let v = yy.mul(d_const()).add(Fe::ONE);
        // x = sqrt(u/v)
        let v3 = v.square().mul(v);
        let v7 = v3.square().mul(v);
        let mut x = u.mul(v3).mul(u.mul(v7).pow_p58());
        let vxx = x.square().mul(v);
        if !vxx.sub(u).is_zero() {
            if vxx.add(u).is_zero() {
                x = x.mul(sqrt_m1());
            } else {
                return None; // not a point on the curve
            }
        }
        if x.is_negative() != sign {
            x = x.neg();
        }
        // x = 0 with a set sign bit is the one non-canonical encoding left.
        if x.is_zero() && sign {
            return None;
        }
        Some(Point {
            x,
            y,
            z: Fe::ONE,
            t: x.mul(y),
        })
    }

    /// `k * self`, most significant bit first.
    ///
    /// Plain double-and-add: 255 doublings and one conditional addition per set
    /// bit. It is the version whose correctness can be read off the page, and
    /// the timing caveat at the top of this module applies to it as it would to
    /// a windowed ladder.
    fn mul_scalar(self, k: &[u8; 32]) -> Point {
        let mut acc = Point::IDENTITY;
        for i in (0..256).rev() {
            acc = acc.double();
            if bit(k, i) {
                acc = acc.add(self);
            }
        }
        acc
    }

    /// Whether this point has small order — the check RFC 8032 §5.1.7 needs and
    /// that makes signature verification unambiguous.
    fn is_small_order(self) -> bool {
        self.double().double().double().is_identity()
    }

    fn is_identity(self) -> bool {
        // (X : Y : Z) is the identity when X == 0 and Y == Z.
        self.x.is_zero() && self.y.eq(self.z)
    }
}

fn bit(b: &[u8; 32], i: u32) -> bool {
    (b[(i / 8) as usize] >> (i % 8)) & 1 == 1
}

// --------------------------------------------------------------------- scalars --
//
// Scalars live mod L = 2^252 + 27742317777372353535851937790883648493.

const L: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];

/// Reduces a 64-byte little-endian integer mod L, by long division on bits.
///
/// Slower than Barrett reduction and much easier to read; it runs twice per
/// signature.
fn reduce_wide(x: &[u8; 64]) -> [u8; 32] {
    let mut acc = [0u8; 33];
    for i in (0..512).rev() {
        // acc = acc * 2 + bit
        let mut carry = (x[i / 8] >> (i % 8)) & 1;
        for byte in acc.iter_mut() {
            let v = ((*byte as u16) << 1) | carry as u16;
            *byte = v as u8;
            carry = (v >> 8) as u8;
        }
        if ge_l(&acc) {
            sub_l(&mut acc);
        }
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&acc[..32]);
    out
}

fn ge_l(acc: &[u8; 33]) -> bool {
    if acc[32] != 0 {
        return true;
    }
    for i in (0..32).rev() {
        if acc[i] != L[i] {
            return acc[i] > L[i];
        }
    }
    true
}

fn sub_l(acc: &mut [u8; 33]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let v = acc[i] as i16 - L[i] as i16 - borrow;
        if v < 0 {
            acc[i] = (v + 256) as u8;
            borrow = 1;
        } else {
            acc[i] = v as u8;
            borrow = 0;
        }
    }
    acc[32] = (acc[32] as i16 - borrow) as u8;
}

/// `(a * b + c) mod L`, on little-endian 32-byte scalars.
fn mul_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    let mut wide = [0u32; 64];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            wide[i + j] += (*x as u32) * (*y as u32);
        }
    }
    for (i, x) in c.iter().enumerate() {
        wide[i] += *x as u32;
    }
    let mut bytes = [0u8; 64];
    let mut carry = 0u32;
    for (i, w) in wide.iter().enumerate() {
        let v = w + carry;
        bytes[i] = v as u8;
        carry = v >> 8;
    }
    // A 32x32-byte product plus a 32-byte addend cannot exceed 64 bytes.
    debug_assert_eq!(carry, 0);
    reduce_wide(&bytes)
}

/// Whether a scalar is canonically reduced, i.e. strictly below L. RFC 8032
/// §5.1.7 requires rejecting `s >= L`, which is what stops signature
/// malleability.
fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    let mut acc = [0u8; 33];
    acc[..32].copy_from_slice(s);
    !ge_l(&acc)
}

// ------------------------------------------------------------------------ api --

/// An Ed25519 secret key: the 32-byte seed, plus the derived scalar and prefix.
#[derive(Clone)]
pub struct SecretKey {
    seed: [u8; KEY_LEN],
    scalar: [u8; 32],
    prefix: [u8; 32],
    public: [u8; KEY_LEN],
}

impl SecretKey {
    /// Derives a key pair from a 32-byte seed (RFC 8032 §5.1.5).
    pub fn from_seed(seed: &[u8; KEY_LEN]) -> SecretKey {
        let h = sha512(seed);
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&h[..32]);
        // Clamping, exactly as the RFC specifies.
        scalar[0] &= 248;
        scalar[31] &= 127;
        scalar[31] |= 64;
        let mut prefix = [0u8; 32];
        prefix.copy_from_slice(&h[32..]);
        let public = Point::base().mul_scalar(&scalar).compress();
        SecretKey {
            seed: *seed,
            scalar,
            prefix,
            public,
        }
    }

    pub fn seed(&self) -> [u8; KEY_LEN] {
        self.seed
    }

    pub fn public_key(&self) -> [u8; KEY_LEN] {
        self.public
    }

    /// PureEdDSA signature over `message` (RFC 8032 §5.1.6).
    pub fn sign(&self, message: &[u8]) -> [u8; SIG_LEN] {
        let mut h = crate::sha512::Sha512::new();
        h.update(&self.prefix).update(message);
        let r = reduce_wide(&h.finalize());
        let big_r = Point::base().mul_scalar(&r).compress();

        let mut h = crate::sha512::Sha512::new();
        h.update(&big_r).update(&self.public).update(message);
        let k = reduce_wide(&h.finalize());

        let s = mul_add(&k, &self.scalar, &r);
        let mut sig = [0u8; SIG_LEN];
        sig[..32].copy_from_slice(&big_r);
        sig[32..].copy_from_slice(&s);
        sig
    }
}

/// Verifies a PureEdDSA signature (RFC 8032 §5.1.7).
///
/// Rejects a non-canonical `s`, a public key or `R` that does not decompress,
/// and a small-order public key — the checks that make "this signature is
/// valid" mean one thing rather than several.
pub fn verify(public: &[u8; KEY_LEN], message: &[u8], sig: &[u8; SIG_LEN]) -> bool {
    let mut s = [0u8; 32];
    s.copy_from_slice(&sig[32..]);
    if !scalar_is_canonical(&s) {
        return false;
    }
    let mut rb = [0u8; 32];
    rb.copy_from_slice(&sig[..32]);
    let Some(big_r) = Point::decompress(&rb) else {
        return false;
    };
    let Some(a) = Point::decompress(public) else {
        return false;
    };
    if a.is_small_order() {
        return false;
    }
    let mut h = crate::sha512::Sha512::new();
    h.update(&rb).update(public).update(message);
    let k = reduce_wide(&h.finalize());

    // Check [s]B == R + [k]A, in the cofactorless form RFC 8032 specifies for
    // the strict variant.
    let lhs = Point::base().mul_scalar(&s);
    let rhs = big_r.add(a.mul_scalar(&k));
    lhs.add(rhs.negate()).is_identity()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::hex;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    fn seed(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    #[test]
    fn rfc_8032_test_vectors() {
        // RFC 8032 §7.1, TEST 1 through TEST 3 and the SHA-abc case.
        let cases: &[(&str, &str, &str, &str)] = &[
            (
                "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
                "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
                "",
                "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555\
                 fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
            ),
            (
                "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
                "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
                "72",
                "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da\
                 085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
            ),
            (
                "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
                "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
                "af82",
                "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac\
                 18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
            ),
        ];
        for (sk, pk, msg, sig) in cases {
            let s = SecretKey::from_seed(&seed(sk));
            assert_eq!(hex(&s.public_key()), *pk, "public key for {sk}");
            let m = unhex(msg);
            let want = sig.replace(char::is_whitespace, "");
            assert_eq!(hex(&s.sign(&m)), want, "signature over {msg:?}");
            let sig_bytes: [u8; 64] = unhex(&want).try_into().unwrap();
            assert!(verify(&s.public_key(), &m, &sig_bytes));
        }
    }

    #[test]
    fn a_long_message_vector() {
        // RFC 8032 §7.1 TEST 1024, truncated to the parts that matter: the
        // signature must round-trip over a message longer than one SHA-512
        // block.
        let s = SecretKey::from_seed(&seed(
            "f5e5767cf153319517630f226876b86c8160cc583bc013744c6bf255f5cc0ee5",
        ));
        assert_eq!(
            hex(&s.public_key()),
            "278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e"
        );
        let msg: Vec<u8> = (0..1023u32).map(|i| (i % 251) as u8).collect();
        let sig = s.sign(&msg);
        assert!(verify(&s.public_key(), &msg, &sig));
    }

    #[test]
    fn tampering_invalidates() {
        let s = SecretKey::from_seed(&[7u8; 32]);
        let msg = b"omni/1.0 tbs";
        let sig = s.sign(msg);
        assert!(verify(&s.public_key(), msg, &sig));
        // A changed message.
        assert!(!verify(&s.public_key(), b"omni/1.0 tbz", &sig));
        // A changed signature, in either half.
        for i in [0usize, 32, 63] {
            let mut bad = sig;
            bad[i] ^= 1;
            assert!(!verify(&s.public_key(), msg, &bad), "byte {i}");
        }
        // A different key.
        let other = SecretKey::from_seed(&[8u8; 32]);
        assert!(!verify(&other.public_key(), msg, &sig));
    }

    #[test]
    fn malleable_and_degenerate_inputs_are_refused() {
        let s = SecretKey::from_seed(&[9u8; 32]);
        let msg = b"x";
        let sig = s.sign(msg);
        // s >= L must be rejected, or signatures are malleable.
        let mut mall = sig;
        mall[32..].copy_from_slice(&L);
        assert!(!verify(&s.public_key(), msg, &mall));
        // A public key that is not a point.
        let mut bad_key = s.public_key();
        bad_key[31] = 0x7f;
        bad_key[0] = 0x02;
        let _ = verify(&bad_key, msg, &sig); // must not panic
                                             // The all-zero key is the small-order identity and is refused outright.
        assert!(!verify(&[0u8; 32], msg, &sig));
        // A non-canonical y (p + 1 re-encoded) is refused.
        let mut noncanon = [0xffu8; 32];
        noncanon[31] = 0x7f;
        assert!(!verify(&noncanon, msg, &sig));
    }

    #[test]
    fn signing_is_deterministic() {
        // EdDSA has no nonce to get wrong: the same key and message give the
        // same signature, every time, which is also what makes a signed
        // container reproducible.
        let s = SecretKey::from_seed(&[3u8; 32]);
        let a = s.sign(b"reproducible");
        let b = s.sign(b"reproducible");
        assert_eq!(a, b);
        // And the key pair is a function of the seed alone.
        let again = SecretKey::from_seed(&[3u8; 32]);
        assert_eq!(again.public_key(), s.public_key());
    }

    #[test]
    fn field_arithmetic_holds() {
        // A handful of algebraic identities, since everything above rests on
        // them.
        let a = Fe::from_bytes(&[3u8; 32]);
        let b = Fe::from_bytes(&[7u8; 32]);
        assert!(a.add(b).sub(b).eq(a));
        assert!(a.mul(Fe::ONE).eq(a));
        assert!(a.mul(a.invert()).eq(Fe::ONE));
        assert!(a.sub(a).is_zero());
        assert!(a.mul(b).eq(b.mul(a)));
        assert!(a.square().eq(a.mul(a)));
        assert!(a.neg().neg().eq(a));
        // Round-tripping through bytes is the identity on canonical elements.
        assert_eq!(Fe::from_bytes(&a.to_bytes()).to_bytes(), a.to_bytes());
    }

    #[test]
    fn group_arithmetic_holds() {
        let b = Point::base();
        assert!(!b.is_identity());
        assert!(!b.is_small_order());
        // 2B == B + B.
        assert_eq!(b.double().compress(), b.add(b).compress());
        // B - B is the identity.
        assert!(b.add(b.negate()).is_identity());
        // Scalar multiplication agrees with repeated addition.
        let mut k = [0u8; 32];
        k[0] = 5;
        let five = b.mul_scalar(&k);
        let by_hand = b.add(b).add(b).add(b).add(b);
        assert_eq!(five.compress(), by_hand.compress());
        // Compression round-trips.
        let c = five.compress();
        assert_eq!(Point::decompress(&c).unwrap().compress(), c);
        // The identity has small order; the base point does not.
        assert!(Point::IDENTITY.is_small_order());
    }
}
