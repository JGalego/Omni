//! NIST P-256 and ECDSA-with-SHA-256 — §12.5's `ES256`.
//!
//! §12.5 names three signature algorithms: Ed25519, ES256 and ML-DSA. Ed25519
//! is the default and is in [`crate::ed25519`]. ES256 is the one the rest of the
//! world already has: it is what a WebPKI certificate signs with, what an HSM
//! and a KMS offer without argument, what Sigstore's Fulcio issues, and what
//! COSE calls `-7`. A format that can only be signed by keys nobody's compliance
//! process recognises is a format with a queue in front of it.
//!
//! ## What is here
//!
//! The curve arithmetic (field `p = 2^256 − 2^224 + 2^192 + 2^96 − 1`, group
//! order `n`, Jacobian coordinates, double-and-add), ECDSA signing and
//! verification, and RFC 6979's deterministic nonce.
//!
//! **Deterministic `k` is not an optimization.** ECDSA with a repeated nonce
//! leaks the private key outright, and with a slightly biased one leaks it over
//! enough signatures; both have happened in the field to shipped software. RFC
//! 6979 derives `k` from the key and the message through HMAC-SHA-256, which
//! removes the entropy source from the threat model entirely — and, for OMNI,
//! has a second effect that matters here: signing the same manifest twice
//! produces the same bytes, so a signature is reproducible in the sense §01.10
//! uses the word.
//!
//! ## What this is not
//!
//! Constant-time. The scalar multiplication branches on bits of the scalar, so
//! a local attacker who can time it learns about the key. That is the same
//! caveat [`crate::blake3`] carries about SIMD and for the same reason:
//! auditability over performance, and a production signer belongs in an HSM or
//! behind a library with the countermeasures. Verification handles only public
//! data and is unaffected.

use crate::sha256::sha256;

/// A field element or scalar: four 64-bit limbs, least significant first.
type U256 = [u64; 4];

/// `p = 2^256 − 2^224 + 2^192 + 2^96 − 1`.
const P: U256 = [
    0xFFFF_FFFF_FFFF_FFFF,
    0x0000_0000_FFFF_FFFF,
    0x0000_0000_0000_0000,
    0xFFFF_FFFF_0000_0001,
];

/// The group order.
const N: U256 = [
    0xF3B9_CAC2_FC63_2551,
    0xBCE6_FAAD_A717_9E84,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_0000_0000,
];

/// `b` in `y² = x³ − 3x + b`.
const B: U256 = [
    0x3BCE_3C3E_27D2_604B,
    0x651D_06B0_CC53_B0F6,
    0xB3EB_BD55_7698_86BC,
    0x5AC6_35D8_AA3A_93E7,
];

/// The base point.
const GX: U256 = [
    0xF4A1_3945_D898_C296,
    0x7703_7D81_2DEB_33A0,
    0xF8BC_E6E5_63A4_40F2,
    0x6B17_D1F2_E12C_4247,
];
const GY: U256 = [
    0xCBB6_4068_37BF_51F5,
    0x2BCE_3357_6B31_5ECE,
    0x8EE7_EB4A_7C0F_9E16,
    0x4FE3_42E2_FE1A_7F9B,
];

/// The number of bytes in a coordinate, a scalar and half a signature.
pub const FIELD_LEN: usize = 32;
/// An uncompressed public key: `0x04 ‖ X ‖ Y`.
pub const PUBLIC_LEN: usize = 65;
/// A raw ECDSA signature: `R ‖ S`, which is what COSE carries.
pub const SIG_LEN: usize = 64;

// ------------------------------------------------------------- 256-bit maths --

fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = (a as u128) + (b as u128) + (carry as u128);
    (t as u64, (t >> 64) as u64)
}

fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let t = (a as u128)
        .wrapping_sub(b as u128)
        .wrapping_sub(borrow as u128);
    (t as u64, ((t >> 64) as u64) & 1)
}

fn add_carry(a: &U256, b: &U256) -> (U256, u64) {
    let mut out = [0u64; 4];
    let mut c = 0u64;
    for i in 0..4 {
        let (v, nc) = adc(a[i], b[i], c);
        out[i] = v;
        c = nc;
    }
    (out, c)
}

fn sub_borrow(a: &U256, b: &U256) -> (U256, u64) {
    let mut out = [0u64; 4];
    let mut br = 0u64;
    for i in 0..4 {
        let (v, nb) = sbb(a[i], b[i], br);
        out[i] = v;
        br = nb;
    }
    (out, br)
}

fn is_zero(a: &U256) -> bool {
    a.iter().all(|&x| x == 0)
}

/// `a < b`, unsigned.
fn lt(a: &U256, b: &U256) -> bool {
    sub_borrow(a, b).1 == 1
}

/// Modular addition, for any modulus the operands are already reduced under.
fn add_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    let (s, carry) = add_carry(a, b);
    let (r, borrow) = sub_borrow(&s, m);
    // Subtract the modulus when the sum reached it — which the carry out of the
    // addition also signals.
    if carry == 1 || borrow == 0 {
        r
    } else {
        s
    }
}

fn sub_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    let (d, borrow) = sub_borrow(a, b);
    if borrow == 1 {
        add_carry(&d, m).0
    } else {
        d
    }
}

/// Schoolbook 256×256 → 512.
fn mul_wide(a: &U256, b: &U256) -> [u64; 8] {
    let mut out = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let t = (a[i] as u128) * (b[j] as u128) + (out[i + j] as u128) + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
        out[i + 4] = carry as u64;
    }
    out
}

/// Reduces a 512-bit value modulo `m` by long division on bits.
///
/// Not fast, and deliberately not clever: a Solinas reduction for `p` and a
/// Montgomery ladder for `n` would each be a place to make a subtle mistake, and
/// this is a reference implementation whose job is to be checkable. Signing a
/// manifest is not on anybody's hot path.
fn reduce_wide(x: &[u64; 8], m: &U256) -> U256 {
    let mut r: U256 = [0; 4];
    for bit in (0..512).rev() {
        // r = r*2 + bit(x)
        let mut carry = (x[bit / 64] >> (bit % 64)) & 1;
        for limb in r.iter_mut() {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        // The shifted-out bit means r ≥ 2^256 > m, so the subtraction below is
        // the one that brings it back.
        if carry == 1 || !lt(&r, m) {
            r = sub_borrow(&r, m).0;
        }
    }
    r
}

fn mul_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    reduce_wide(&mul_wide(a, b), m)
}

/// `a^e mod m`, square-and-multiply over a fixed exponent.
fn pow_mod(a: &U256, e: &U256, m: &U256) -> U256 {
    let mut result: U256 = [1, 0, 0, 0];
    for bit in (0..256).rev() {
        result = mul_mod(&result, &result, m);
        if (e[bit / 64] >> (bit % 64)) & 1 == 1 {
            result = mul_mod(&result, a, m);
        }
    }
    result
}

/// The modular inverse, by Fermat: `a^(m−2) mod m`. Both moduli here are prime.
fn inv_mod(a: &U256, m: &U256) -> U256 {
    let two: U256 = [2, 0, 0, 0];
    let e = sub_borrow(m, &two).0;
    pow_mod(a, &e, m)
}

fn from_be(bytes: &[u8; 32]) -> U256 {
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&bytes[24 - i * 8..32 - i * 8]);
        out[i] = u64::from_be_bytes(limb);
    }
    out
}

fn to_be(v: &U256) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[24 - i * 8..32 - i * 8].copy_from_slice(&v[i].to_be_bytes());
    }
    out
}

// ----------------------------------------------------------------- the curve --

/// A point in Jacobian coordinates: `(X : Y : Z)` is `(X/Z², Y/Z³)`, with
/// `Z = 0` the point at infinity.
#[derive(Clone, Copy, Debug)]
struct Point {
    x: U256,
    y: U256,
    z: U256,
}

const INFINITY: Point = Point {
    x: [1, 0, 0, 0],
    y: [1, 0, 0, 0],
    z: [0, 0, 0, 0],
};

impl Point {
    fn generator() -> Point {
        Point {
            x: GX,
            y: GY,
            z: [1, 0, 0, 0],
        }
    }

    fn is_infinity(&self) -> bool {
        is_zero(&self.z)
    }

    /// The doubling formula for `a = −3`, which is why P-256 chose that `a`.
    fn double(&self) -> Point {
        if self.is_infinity() {
            return *self;
        }
        let f = |a: &U256, b: &U256| mul_mod(a, b, &P);
        let add = |a: &U256, b: &U256| add_mod(a, b, &P);
        let sub = |a: &U256, b: &U256| sub_mod(a, b, &P);

        let zz = f(&self.z, &self.z);
        // delta = 3(X − Z²)(X + Z²)
        let s1 = sub(&self.x, &zz);
        let s2 = add(&self.x, &zz);
        let m = f(&s1, &s2);
        let m = add(&add(&m, &m), &m);

        let yy = f(&self.y, &self.y);
        let s = f(&self.x, &yy);
        let s = add(&s, &s);
        let s = add(&s, &s);

        let x3 = sub(&f(&m, &m), &add(&s, &s));
        let yyyy = f(&yy, &yy);
        let eight_yyyy = {
            let t = add(&yyyy, &yyyy);
            let t = add(&t, &t);
            add(&t, &t)
        };
        let y3 = sub(&f(&m, &sub(&s, &x3)), &eight_yyyy);
        let z3 = {
            let t = f(&self.y, &self.z);
            add(&t, &t)
        };
        Point {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `self + other`, with `other` in affine form (`Z = 1`).
    fn add_affine(&self, ax: &U256, ay: &U256) -> Point {
        if self.is_infinity() {
            return Point {
                x: *ax,
                y: *ay,
                z: [1, 0, 0, 0],
            };
        }
        let f = |a: &U256, b: &U256| mul_mod(a, b, &P);
        let add = |a: &U256, b: &U256| add_mod(a, b, &P);
        let sub = |a: &U256, b: &U256| sub_mod(a, b, &P);

        let zz = f(&self.z, &self.z);
        let u2 = f(ax, &zz);
        let s2 = f(ay, &f(&zz, &self.z));
        let h = sub(&u2, &self.x);
        let r = sub(&s2, &self.y);
        if is_zero(&h) {
            if is_zero(&r) {
                return self.double();
            }
            return INFINITY;
        }
        let hh = f(&h, &h);
        let hhh = f(&hh, &h);
        let v = f(&self.x, &hh);
        let x3 = sub(&sub(&f(&r, &r), &hhh), &add(&v, &v));
        let y3 = sub(&f(&r, &sub(&v, &x3)), &f(&self.y, &hhh));
        let z3 = f(&self.z, &h);
        Point {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    fn add(&self, other: &Point) -> Point {
        if other.is_infinity() {
            return *self;
        }
        let (ax, ay) = other.to_affine();
        self.add_affine(&ax, &ay)
    }

    fn to_affine(self) -> (U256, U256) {
        if self.is_infinity() {
            return ([0; 4], [0; 4]);
        }
        let zinv = inv_mod(&self.z, &P);
        let zinv2 = mul_mod(&zinv, &zinv, &P);
        let zinv3 = mul_mod(&zinv2, &zinv, &P);
        (mul_mod(&self.x, &zinv2, &P), mul_mod(&self.y, &zinv3, &P))
    }

    /// `k · self`, double-and-add over the scalar's bits.
    fn mul(&self, k: &U256) -> Point {
        let (ax, ay) = self.to_affine();
        let mut acc = INFINITY;
        for bit in (0..256).rev() {
            acc = acc.double();
            if (k[bit / 64] >> (bit % 64)) & 1 == 1 {
                acc = acc.add_affine(&ax, &ay);
            }
        }
        acc
    }
}

/// Whether `(x, y)` satisfies `y² = x³ − 3x + b` and is not the point at
/// infinity.
///
/// Checked on every public key, because a point off the curve is how an
/// invalid-curve attack starts: the arithmetic still produces answers, on a
/// different and usually much weaker curve.
fn on_curve(x: &U256, y: &U256) -> bool {
    if !lt(x, &P) || !lt(y, &P) {
        return false;
    }
    let yy = mul_mod(y, y, &P);
    let xxx = mul_mod(&mul_mod(x, x, &P), x, &P);
    let three_x = add_mod(&add_mod(x, x, &P), x, &P);
    let rhs = add_mod(&sub_mod(&xxx, &three_x, &P), &B, &P);
    yy == rhs
}

// -------------------------------------------------------------------- ECDSA --

/// A P-256 secret key.
#[derive(Clone)]
pub struct SecretKey {
    d: U256,
    seed: [u8; 32],
}

impl SecretKey {
    /// Takes a 32-byte big-endian scalar. Refuses zero and anything ≥ `n`,
    /// which are not keys.
    pub fn from_bytes(seed: &[u8; 32]) -> Option<SecretKey> {
        let d = from_be(seed);
        if is_zero(&d) || !lt(&d, &N) {
            return None;
        }
        Some(SecretKey { d, seed: *seed })
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.seed
    }

    /// The uncompressed public point, `0x04 ‖ X ‖ Y`.
    pub fn public_key(&self) -> [u8; PUBLIC_LEN] {
        let (x, y) = Point::generator().mul(&self.d).to_affine();
        let mut out = [0u8; PUBLIC_LEN];
        out[0] = 0x04;
        out[1..33].copy_from_slice(&to_be(&x));
        out[33..].copy_from_slice(&to_be(&y));
        out
    }

    /// Signs a message, returning `R ‖ S`.
    ///
    /// `k` is RFC 6979's deterministic nonce, so this is a pure function of the
    /// key and the message: no entropy source in the threat model, and the same
    /// manifest signs to the same bytes twice.
    pub fn sign(&self, message: &[u8]) -> [u8; SIG_LEN] {
        let h = sha256(message);
        let e = reduce_scalar(&h);
        let mut attempt = 0u32;
        loop {
            let k = rfc6979_k(&self.seed, &h, attempt);
            attempt += 1;
            if is_zero(&k) || !lt(&k, &N) {
                continue;
            }
            let point = Point::generator().mul(&k);
            if point.is_infinity() {
                continue;
            }
            let (px, _) = point.to_affine();
            let r = reduce_scalar(&to_be(&px));
            if is_zero(&r) {
                continue;
            }
            let kinv = inv_mod(&k, &N);
            let s = mul_mod(&kinv, &add_mod(&e, &mul_mod(&r, &self.d, &N), &N), &N);
            if is_zero(&s) {
                continue;
            }
            // Low-S, as every modern verifier expects: `(r, s)` and `(r, n − s)`
            // are both valid, and leaving the choice open is a malleability
            // nobody wants in a signature that names a model.
            let s = if lt(&half_n(), &s) {
                sub_borrow(&N, &s).0
            } else {
                s
            };
            let mut out = [0u8; SIG_LEN];
            out[..32].copy_from_slice(&to_be(&r));
            out[32..].copy_from_slice(&to_be(&s));
            return out;
        }
    }
}

fn half_n() -> U256 {
    // (n − 1) / 2, computed once from N rather than written out twice.
    let mut h = N;
    let mut carry = 0u64;
    for limb in h.iter_mut().rev() {
        let next = *limb & 1;
        *limb = (*limb >> 1) | (carry << 63);
        carry = next;
    }
    h
}

/// A 32-byte hash as a scalar: reduced modulo `n`, which is what ECDSA's `e` is
/// for a curve whose order is 256 bits.
fn reduce_scalar(bytes: &[u8; 32]) -> U256 {
    let v = from_be(bytes);
    if lt(&v, &N) {
        v
    } else {
        sub_borrow(&v, &N).0
    }
}

/// HMAC-SHA-256.
fn hmac(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    if key.len() > 64 {
        k[..32].copy_from_slice(&sha256(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = ipad.to_vec();
    inner.extend_from_slice(data);
    let inner = sha256(&inner);
    let mut outer = opad.to_vec();
    outer.extend_from_slice(&inner);
    sha256(&outer)
}

/// RFC 6979 §3.2's deterministic nonce, with `attempt` extra rounds of the
/// generator for the (astronomically unlikely) case where the first `k` is out
/// of range.
fn rfc6979_k(secret: &[u8; 32], h1: &[u8; 32], attempt: u32) -> U256 {
    let mut v = [0x01u8; 32];
    let mut k = [0x00u8; 32];
    let bits2octets = to_be(&reduce_scalar(h1));

    let mut msg = Vec::with_capacity(97);
    msg.extend_from_slice(&v);
    msg.push(0x00);
    msg.extend_from_slice(secret);
    msg.extend_from_slice(&bits2octets);
    k = hmac(&k, &msg);
    v = hmac(&k, &v);

    let mut msg = Vec::with_capacity(97);
    msg.extend_from_slice(&v);
    msg.push(0x01);
    msg.extend_from_slice(secret);
    msg.extend_from_slice(&bits2octets);
    k = hmac(&k, &msg);
    v = hmac(&k, &v);

    for _ in 0..attempt {
        k = hmac(&k, &[v.as_slice(), &[0x00]].concat());
        v = hmac(&k, &v);
    }
    v = hmac(&k, &v);
    from_be(&v)
}

/// Verifies `R ‖ S` over a message with an uncompressed public key.
///
/// Every one of the checks below has been the subject of a real vulnerability in
/// somebody's ECDSA: a point off the curve, an `r` or `s` outside `[1, n)`, and
/// the point at infinity as a public key.
pub fn verify(public: &[u8], message: &[u8], sig: &[u8]) -> bool {
    if public.len() != PUBLIC_LEN || public[0] != 0x04 || sig.len() != SIG_LEN {
        return false;
    }
    let mut xb = [0u8; 32];
    let mut yb = [0u8; 32];
    xb.copy_from_slice(&public[1..33]);
    yb.copy_from_slice(&public[33..]);
    let (qx, qy) = (from_be(&xb), from_be(&yb));
    if !on_curve(&qx, &qy) {
        return false;
    }
    let mut rb = [0u8; 32];
    let mut sb = [0u8; 32];
    rb.copy_from_slice(&sig[..32]);
    sb.copy_from_slice(&sig[32..]);
    let (r, s) = (from_be(&rb), from_be(&sb));
    if is_zero(&r) || is_zero(&s) || !lt(&r, &N) || !lt(&s, &N) {
        return false;
    }

    let e = reduce_scalar(&sha256(message));
    let sinv = inv_mod(&s, &N);
    let u1 = mul_mod(&e, &sinv, &N);
    let u2 = mul_mod(&r, &sinv, &N);
    let q = Point {
        x: qx,
        y: qy,
        z: [1, 0, 0, 0],
    };
    let point = Point::generator().mul(&u1).add(&q.mul(&u2));
    if point.is_infinity() {
        return false;
    }
    let (px, _) = point.to_affine();
    reduce_scalar(&to_be(&px)) == r
}

/// A key pair from a seed, for `omni keygen --alg es256`.
///
/// The seed is hashed rather than used directly, so that any 32 bytes produce a
/// key: a raw seed can be zero or above the group order, and both are values a
/// user could hand over.
pub fn from_seed(seed: &[u8; 32]) -> SecretKey {
    let mut candidate = *seed;
    loop {
        if let Some(k) = SecretKey::from_bytes(&candidate) {
            return k;
        }
        candidate = sha256(&candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> SecretKey {
        from_seed(&[7u8; 32])
    }

    #[test]
    fn the_generator_is_on_the_curve_and_has_the_stated_order() {
        let (gx, gy) = Point::generator().to_affine();
        assert!(on_curve(&gx, &gy));
        // n·G is the point at infinity: the definition of the group order, and
        // the check that the constants were transcribed correctly.
        assert!(Point::generator().mul(&N).is_infinity());
        // And (n+1)·G is G again.
        let n1 = add_carry(&N, &[1, 0, 0, 0]).0;
        let (x, y) = Point::generator().mul(&n1).to_affine();
        assert_eq!((x, y), (gx, gy));
    }

    #[test]
    fn doubling_and_addition_agree() {
        let g = Point::generator();
        let two = g.double();
        let also_two = g.add(&g);
        assert_eq!(two.to_affine(), also_two.to_affine());
        let three = two.add(&g);
        let (x, y) = three.to_affine();
        assert!(on_curve(&x, &y));
        assert_eq!(g.mul(&[3, 0, 0, 0]).to_affine(), three.to_affine());
    }

    #[test]
    fn a_signature_verifies_and_a_tampered_one_does_not() {
        let k = key();
        let pk = k.public_key();
        let msg = b"a model exists once";
        let sig = k.sign(msg);
        assert!(verify(&pk, msg, &sig));
        // Every byte of the signature matters.
        for i in 0..SIG_LEN {
            let mut bad = sig;
            bad[i] ^= 1;
            assert!(!verify(&pk, msg, &bad), "byte {i}");
        }
        // And so does the message.
        assert!(!verify(&pk, b"a model exists twice", &sig));
        // A key that is not the signer's.
        let other = from_seed(&[9u8; 32]).public_key();
        assert!(!verify(&other, msg, &sig));
    }

    #[test]
    fn signing_is_deterministic() {
        // RFC 6979: the same key and message give the same bytes, which is what
        // makes a signed container reproducible (§01.10).
        let k = key();
        let a = k.sign(b"same");
        let b = k.sign(b"same");
        assert_eq!(a, b);
        assert_ne!(a, k.sign(b"different"));
    }

    #[test]
    fn signatures_are_low_s() {
        // Both `s` and `n − s` verify, so a signer that picks either produces
        // two valid signatures for one message. Every modern verifier expects
        // the low one; producing the high one is a malleability nobody wants.
        let k = key();
        for i in 0..4u8 {
            let sig = k.sign(&[i; 8]);
            let mut sb = [0u8; 32];
            sb.copy_from_slice(&sig[32..]);
            let s = from_be(&sb);
            assert!(!lt(&half_n(), &s), "signature {i} has a high s");
        }
    }

    #[test]
    fn a_public_key_off_the_curve_is_refused() {
        let k = key();
        let msg = b"x";
        let sig = k.sign(msg);
        let mut pk = k.public_key();
        // Move the point off the curve: the arithmetic would still produce
        // answers, on a different and much weaker curve.
        pk[64] ^= 1;
        assert!(!verify(&pk, msg, &sig));
        // A key that is not in the uncompressed form at all.
        assert!(!verify(&pk[..64], msg, &sig));
        pk[0] = 0x02;
        assert!(!verify(&pk, msg, &sig));
    }

    #[test]
    fn r_or_s_outside_the_group_is_refused() {
        let k = key();
        let msg = b"x";
        let mut sig = k.sign(msg);
        let zero = [0u8; 32];
        let mut z = sig;
        z[..32].copy_from_slice(&zero);
        assert!(!verify(&k.public_key(), msg, &z));
        let mut z = sig;
        z[32..].copy_from_slice(&zero);
        assert!(!verify(&k.public_key(), msg, &z));
        // s = n is out of range even though it is one past a valid value.
        sig[32..].copy_from_slice(&to_be(&N));
        assert!(!verify(&k.public_key(), msg, &sig));
    }

    #[test]
    fn a_seed_that_is_not_a_scalar_still_makes_a_key() {
        // Zero and the group order are both values a user could hand over, and
        // neither is a private key.
        let k = from_seed(&[0u8; 32]);
        assert!(!is_zero(&k.d) && lt(&k.d, &N));
        let k = from_seed(&to_be(&N));
        assert!(!is_zero(&k.d) && lt(&k.d, &N));
        // And a valid one is used as given, so a key round-trips.
        let raw = [5u8; 32];
        assert_eq!(from_seed(&raw).to_bytes(), raw);
    }

    #[test]
    fn the_deterministic_nonce_matches_rfc_6979() {
        // RFC 6979 §A.2.5's P-256 / SHA-256 vector for the message "sample".
        // This is the known-answer test that pins the whole module at once: the
        // field arithmetic, the scalar arithmetic, the point maths and the HMAC
        // ladder all have to be right for `r` to come out.
        let x = [
            0xc9, 0xaf, 0xa9, 0xd8, 0x45, 0xba, 0x75, 0x16, 0x6b, 0x5c, 0x21, 0x57, 0x67, 0xb1,
            0xd6, 0x93, 0x4e, 0x50, 0xc3, 0xdb, 0x36, 0xe8, 0x9b, 0x12, 0x7b, 0x8a, 0x62, 0x2b,
            0x12, 0x0f, 0x67, 0x21,
        ];
        let k = SecretKey::from_bytes(&x).expect("a valid scalar");
        let sig = k.sign(b"sample");
        assert_eq!(
            crate::sha256::hex(&sig[..32]),
            "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716",
            "r"
        );
        // The RFC's `s` is `f7cb1c94…`, which is above n/2. This signer
        // normalizes to low-S, so what it writes is `n − s` — the same
        // signature, in the form every modern verifier expects.
        let mut sb = [0u8; 32];
        sb.copy_from_slice(&sig[32..]);
        let s = from_be(&sb);
        let complement = sub_borrow(&N, &s).0;
        assert_eq!(
            crate::sha256::hex(&to_be(&complement)),
            "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8",
            "n - s"
        );
        // And the RFC's public key, which is the other half of the vector.
        let pk = k.public_key();
        assert_eq!(
            crate::sha256::hex(&pk[1..33]),
            "60fed4ba255a9d31c961eb74c6356d68c049b8923b61fa6ce669622e60f29fb6",
            "Ux"
        );
        assert_eq!(
            crate::sha256::hex(&pk[33..]),
            "7903fe1008b8bc99a41ae9e95628bc64f2f1b20c2d7e9f5177a3c294d4462299",
            "Uy"
        );
        assert!(verify(&pk, b"sample", &sig));
    }

    #[test]
    fn hmac_matches_its_published_vectors() {
        // RFC 4231 test case 1, so the HMAC RFC 6979 depends on is checked
        // against something outside this crate.
        let mac = hmac(&[0x0b; 20], b"Hi There");
        assert_eq!(
            crate::sha256::hex(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        // Case 2.
        let mac = hmac(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            crate::sha256::hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }
}
