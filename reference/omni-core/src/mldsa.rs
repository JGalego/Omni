//! ML-DSA (FIPS 204) — the post-quantum signature algorithm §12.5.1 names and
//! §12.11 exists for.
//!
//! §12.11 is about cryptographic agility against adversary A7, "future
//! cryptanalyst", and the honest position on a post-quantum algorithm is not
//! that it is better than Ed25519 today — it is that a container signed in 2026
//! and verified in 2046 cannot be re-signed by whoever wrote it. That is what
//! makes an archival format's choice of signature different from a TLS session's.
//!
//! This is ML-DSA-44, ML-DSA-65 and ML-DSA-87, from FIPS 204 and nothing else,
//! and it is checked against NIST's own ACVP known-answer vectors rather than
//! against itself — see `tools/acvp-vectors.py` for how the fixtures are
//! fetched and `tests/vectors/mldsa/` for what they are. That mattered more here
//! than anywhere else in this crate: a signature scheme that round-trips through
//! its own code proves only that its signer and verifier share an
//! interpretation, and for a scheme with rejection sampling, deterministic
//! nonces and a hint mechanism there are a great many interpretations that are
//! self-consistent and wrong.
//!
//! Arithmetic is plain and reduced with `%`, not Montgomery. That costs
//! throughput and buys the ability to read the file against the standard's
//! pseudocode line by line, which is the same trade the BLAKE3 in this crate
//! makes. A production implementation should use a reviewed library.

use core::fmt;

use crate::shake::Xof;

/// FIPS 204's `H`: SHAKE256 with a requested output length.
fn h(parts: &[&[u8]], n: usize) -> Vec<u8> {
    crate::shake::shake256(parts, n)
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const Q: i32 = 8_380_417;
const N: usize = 256;
const D: u32 = 13;
/// A primitive 512th root of unity mod q, from FIPS 204 §2.5.
const ZETA: i64 = 1753;

/// One ML-DSA parameter set. `lambda` is in bits; the challenge hash is
/// `lambda / 4` bytes, which is where FIPS 204 puts the collision strength.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub name: &'static str,
    pub k: usize,
    pub l: usize,
    pub eta: i32,
    pub tau: usize,
    pub beta: i32,
    pub gamma1: i32,
    pub gamma2: i32,
    pub omega: usize,
    pub lambda: usize,
}

pub const ML_DSA_44: Params = Params {
    name: "ML-DSA-44",
    k: 4,
    l: 4,
    eta: 2,
    tau: 39,
    beta: 78,
    gamma1: 1 << 17,
    gamma2: (Q - 1) / 88,
    omega: 80,
    lambda: 128,
};

pub const ML_DSA_65: Params = Params {
    name: "ML-DSA-65",
    k: 6,
    l: 5,
    eta: 4,
    tau: 49,
    beta: 196,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 55,
    lambda: 192,
};

pub const ML_DSA_87: Params = Params {
    name: "ML-DSA-87",
    k: 8,
    l: 7,
    eta: 2,
    tau: 60,
    beta: 120,
    gamma1: 1 << 19,
    gamma2: (Q - 1) / 32,
    omega: 75,
    lambda: 256,
};

pub const ALL: [Params; 3] = [ML_DSA_44, ML_DSA_65, ML_DSA_87];

impl Params {
    pub fn by_name(name: &str) -> Option<Params> {
        ALL.iter().copied().find(|p| p.name == name)
    }

    /// Bits per coefficient of `s1`/`s2`: `bitlen(2 * eta)`.
    fn eta_bits(&self) -> usize {
        bitlen((2 * self.eta) as u32)
    }

    /// Bits per coefficient of `z`: `1 + bitlen(gamma1 - 1)`.
    fn z_bits(&self) -> usize {
        1 + bitlen((self.gamma1 - 1) as u32)
    }

    /// Bits per coefficient of `w1`.
    fn w1_bits(&self) -> usize {
        bitlen(((Q - 1) / (2 * self.gamma2) - 1) as u32)
    }

    /// Lengths are derived rather than tabulated, so a wrong parameter shows up
    /// as a length that disagrees with FIPS 204's table (checked by a test)
    /// rather than as a subtly malformed key.
    pub fn public_key_len(&self) -> usize {
        32 + self.k * 32 * (bitlen((Q - 1) as u32) - D as usize)
    }

    pub fn secret_key_len(&self) -> usize {
        128 + 32 * self.eta_bits() * (self.k + self.l) + 32 * D as usize * self.k
    }

    pub fn signature_len(&self) -> usize {
        self.lambda / 4 + 32 * self.z_bits() * self.l + self.omega + self.k
    }
}

fn bitlen(mut v: u32) -> usize {
    let mut n = 0;
    while v > 0 {
        n += 1;
        v >>= 1;
    }
    n
}

// ---------------------------------------------------------------------------
// Modular arithmetic and the NTT
// ---------------------------------------------------------------------------

type Poly = [i32; N];

const ZERO: Poly = [0; N];

fn addq(a: i32, b: i32) -> i32 {
    let s = a + b;
    if s >= Q {
        s - Q
    } else {
        s
    }
}

fn subq(a: i32, b: i32) -> i32 {
    let s = a - b;
    if s < 0 {
        s + Q
    } else {
        s
    }
}

fn mulq(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64).rem_euclid(Q as i64)) as i32
}

fn powq(mut base: i64, mut exp: i64) -> i32 {
    let mut acc: i64 = 1;
    base = base.rem_euclid(Q as i64);
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base % Q as i64;
        }
        base = base * base % Q as i64;
        exp >>= 1;
    }
    acc as i32
}

/// `zetas[m] = ZETA^brv8(m) mod q`, computed rather than transcribed. A
/// 256-entry constant table is 256 chances to mistype a number that produces a
/// wrong signature no test outside the known-answer vectors would notice.
fn zetas() -> [i32; N] {
    let mut z = [0i32; N];
    for (m, slot) in z.iter_mut().enumerate() {
        *slot = powq(ZETA, (m as u8).reverse_bits() as i64);
    }
    z
}

fn ntt(w: &mut Poly, z: &[i32; N]) {
    let mut len = 128;
    let mut m = 0;
    while len >= 1 {
        let mut start = 0;
        while start < N {
            m += 1;
            let zeta = z[m];
            for j in start..start + len {
                let t = mulq(zeta, w[j + len]);
                w[j + len] = subq(w[j], t);
                w[j] = addq(w[j], t);
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

fn inv_ntt(w: &mut Poly, z: &[i32; N]) {
    let mut len = 1;
    let mut m = N;
    while len < N {
        let mut start = 0;
        while start < N {
            m -= 1;
            let zeta = Q - z[m];
            for j in start..start + len {
                let t = w[j];
                w[j] = addq(t, w[j + len]);
                w[j + len] = subq(t, w[j + len]);
                w[j + len] = mulq(zeta, w[j + len]);
            }
            start += 2 * len;
        }
        len *= 2;
    }
    let inv_n = powq(N as i64, Q as i64 - 2);
    for c in w.iter_mut() {
        *c = mulq(*c, inv_n);
    }
}

fn poly_mul_ntt(a: &Poly, b: &Poly) -> Poly {
    let mut out = ZERO;
    for i in 0..N {
        out[i] = mulq(a[i], b[i]);
    }
    out
}

fn poly_add(a: &Poly, b: &Poly) -> Poly {
    let mut out = ZERO;
    for i in 0..N {
        out[i] = addq(a[i], b[i]);
    }
    out
}

fn poly_sub(a: &Poly, b: &Poly) -> Poly {
    let mut out = ZERO;
    for i in 0..N {
        out[i] = subq(a[i], b[i]);
    }
    out
}

/// `r mod± alpha`, the centred representative in `(-alpha/2, alpha/2]`.
fn centered(r: i32, alpha: i32) -> i32 {
    let mut m = r.rem_euclid(alpha);
    if m > alpha / 2 {
        m -= alpha;
    }
    m
}

/// The infinity norm of a coefficient held in `[0, q)`, which is its distance
/// from zero in the centred representation.
fn norm(c: i32) -> i32 {
    let m = c.rem_euclid(Q);
    if m > Q / 2 {
        Q - m
    } else {
        m
    }
}

fn poly_norm(p: &Poly) -> i32 {
    p.iter().map(|&c| norm(c)).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Rounding (FIPS 204 §7.4)
// ---------------------------------------------------------------------------

fn power2round(r: i32) -> (i32, i32) {
    let r0 = centered(r, 1 << D);
    ((r - r0) >> D, r0)
}

fn decompose(r: i32, gamma2: i32) -> (i32, i32) {
    let rp = r.rem_euclid(Q);
    let r0 = centered(rp, 2 * gamma2);
    if rp - r0 == Q - 1 {
        (0, r0 - 1)
    } else {
        ((rp - r0) / (2 * gamma2), r0)
    }
}

fn high_bits(r: i32, gamma2: i32) -> i32 {
    decompose(r, gamma2).0
}

fn low_bits(r: i32, gamma2: i32) -> i32 {
    decompose(r, gamma2).1
}

fn make_hint(z: i32, r: i32, gamma2: i32) -> bool {
    high_bits(r.rem_euclid(Q), gamma2) != high_bits((r + z).rem_euclid(Q), gamma2)
}

fn use_hint(hint: bool, r: i32, gamma2: i32) -> i32 {
    let m = (Q - 1) / (2 * gamma2);
    let (r1, r0) = decompose(r, gamma2);
    if !hint {
        r1
    } else if r0 > 0 {
        (r1 + 1).rem_euclid(m)
    } else {
        (r1 - 1).rem_euclid(m)
    }
}

// ---------------------------------------------------------------------------
// Bit packing. FIPS 204's bitstreams are little-endian: bit i of the stream is
// bit (i mod 8) of byte (i / 8), and integers go in least-significant bit first.
// ---------------------------------------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    bit: usize,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter {
            out: Vec::new(),
            bit: 0,
        }
    }

    fn push(&mut self, value: u32, bits: usize) {
        for i in 0..bits {
            if self.bit.is_multiple_of(8) {
                self.out.push(0);
            }
            if (value >> i) & 1 == 1 {
                let last = self.out.len() - 1;
                self.out[last] |= 1 << (self.bit % 8);
            }
            self.bit += 1;
        }
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> BitReader<'a> {
        BitReader { data, bit: 0 }
    }

    fn take(&mut self, bits: usize) -> u32 {
        let mut v = 0u32;
        for i in 0..bits {
            let byte = self.data[self.bit / 8];
            if (byte >> (self.bit % 8)) & 1 == 1 {
                v |= 1 << i;
            }
            self.bit += 1;
        }
        v
    }
}

fn simple_bit_pack(w: &Poly, bits: usize) -> Vec<u8> {
    let mut bw = BitWriter::new();
    for &c in w.iter() {
        bw.push(c as u32, bits);
    }
    bw.out
}

fn bit_pack(w: &Poly, b: i32, bits: usize) -> Vec<u8> {
    let mut bw = BitWriter::new();
    for &c in w.iter() {
        bw.push((b - c) as u32, bits);
    }
    bw.out
}

fn simple_bit_unpack(data: &[u8], bits: usize) -> Poly {
    let mut br = BitReader::new(data);
    let mut out = ZERO;
    for c in out.iter_mut() {
        *c = br.take(bits) as i32;
    }
    out
}

fn bit_unpack(data: &[u8], b: i32, bits: usize) -> Poly {
    let mut br = BitReader::new(data);
    let mut out = ZERO;
    for c in out.iter_mut() {
        *c = b - br.take(bits) as i32;
    }
    out
}

// ---------------------------------------------------------------------------
// Sampling (FIPS 204 §7.3)
// ---------------------------------------------------------------------------

/// `RejNTTPoly`: three bytes at a time, keeping values below q. The sampler is
/// where a wrong byte order goes unnoticed, because a rejection sampler with the
/// bytes shuffled still produces a perfectly uniform-looking polynomial.
fn rej_ntt_poly(seed: &[u8], s: u8, r: u8) -> Poly {
    let mut xof = Xof::shake128(&[seed, &[s], &[r]]);
    let mut out = ZERO;
    let mut j = 0;
    let mut buf = [0u8; 3];
    while j < N {
        xof.squeeze(&mut buf);
        let val = buf[0] as i32 | ((buf[1] as i32) << 8) | (((buf[2] & 0x7f) as i32) << 16);
        if val < Q {
            out[j] = val;
            j += 1;
        }
    }
    out
}

fn coeff_from_half_byte(b: u8, eta: i32) -> Option<i32> {
    if eta == 2 {
        if b < 15 {
            Some(2 - (b as i32 % 5))
        } else {
            None
        }
    } else if b < 9 {
        Some(4 - b as i32)
    } else {
        None
    }
}

fn rej_bounded_poly(seed: &[u8], nonce: u16, eta: i32) -> Poly {
    let mut xof = Xof::shake256(&[seed, &nonce.to_le_bytes()]);
    let mut out = ZERO;
    let mut j = 0;
    let mut buf = [0u8; 1];
    while j < N {
        xof.squeeze(&mut buf);
        let b = buf[0];
        if let Some(c) = coeff_from_half_byte(b & 0x0f, eta) {
            out[j] = c.rem_euclid(Q);
            j += 1;
        }
        if j < N {
            if let Some(c) = coeff_from_half_byte(b >> 4, eta) {
                out[j] = c.rem_euclid(Q);
                j += 1;
            }
        }
    }
    out
}

fn expand_a(rho: &[u8], p: &Params) -> Vec<Vec<Poly>> {
    let mut a = Vec::with_capacity(p.k);
    for r in 0..p.k {
        let mut row = Vec::with_capacity(p.l);
        for s in 0..p.l {
            row.push(rej_ntt_poly(rho, s as u8, r as u8));
        }
        a.push(row);
    }
    a
}

fn expand_s(rhop: &[u8], p: &Params) -> (Vec<Poly>, Vec<Poly>) {
    let s1 = (0..p.l)
        .map(|i| rej_bounded_poly(rhop, i as u16, p.eta))
        .collect();
    let s2 = (0..p.k)
        .map(|i| rej_bounded_poly(rhop, (i + p.l) as u16, p.eta))
        .collect();
    (s1, s2)
}

fn expand_mask(rhopp: &[u8], kappa: u16, p: &Params) -> Vec<Poly> {
    let c = p.z_bits();
    (0..p.l)
        .map(|r| {
            let v = h(&[rhopp, &(kappa + r as u16).to_le_bytes()], 32 * c);
            let raw = bit_unpack(&v, p.gamma1 - 1, c);
            // BitUnpack yields (gamma1 - 1) - value; the mask wants gamma1 - value.
            let mut out = ZERO;
            for (o, &r) in out.iter_mut().zip(raw.iter()) {
                *o = (r + 1).rem_euclid(Q);
            }
            out
        })
        .collect()
}

/// `SampleInBall`: exactly `tau` coefficients are ±1 and the rest zero.
fn sample_in_ball(c_tilde: &[u8], tau: usize) -> Poly {
    let mut xof = Xof::shake256(&[c_tilde]);
    let mut sign_bytes = [0u8; 8];
    xof.squeeze(&mut sign_bytes);
    let mut signs = u64::from_le_bytes(sign_bytes);
    let mut c = ZERO;
    let mut buf = [0u8; 1];
    for i in (N - tau)..N {
        let j = loop {
            xof.squeeze(&mut buf);
            if buf[0] as usize <= i {
                break buf[0] as usize;
            }
        };
        c[i] = c[j];
        c[j] = if signs & 1 == 1 { Q - 1 } else { 1 };
        signs >>= 1;
    }
    c
}

// ---------------------------------------------------------------------------
// Key and signature encoding (FIPS 204 §7.2)
// ---------------------------------------------------------------------------

fn pk_encode(rho: &[u8], t1: &[Poly]) -> Vec<u8> {
    let bits = bitlen((Q - 1) as u32) - D as usize;
    let mut out = rho.to_vec();
    for p in t1 {
        out.extend_from_slice(&simple_bit_pack(p, bits));
    }
    out
}

fn pk_decode(pk: &[u8], p: &Params) -> (Vec<u8>, Vec<Poly>) {
    let bits = bitlen((Q - 1) as u32) - D as usize;
    let per = 32 * bits;
    let rho = pk[..32].to_vec();
    let t1 = (0..p.k)
        .map(|i| simple_bit_unpack(&pk[32 + i * per..32 + (i + 1) * per], bits))
        .collect();
    (rho, t1)
}

#[allow(clippy::too_many_arguments)]
fn sk_encode(
    rho: &[u8],
    key: &[u8],
    tr: &[u8],
    s1: &[Poly],
    s2: &[Poly],
    t0: &[Poly],
    p: &Params,
) -> Vec<u8> {
    let eb = p.eta_bits();
    let mut out = Vec::with_capacity(p.secret_key_len());
    out.extend_from_slice(rho);
    out.extend_from_slice(key);
    out.extend_from_slice(tr);
    for poly in s1.iter().chain(s2.iter()) {
        // Coefficients are held in [0, q); pack the centred value.
        let mut centred = ZERO;
        for (c, &raw) in centred.iter_mut().zip(poly.iter()) {
            *c = centered(raw, Q);
        }
        out.extend_from_slice(&bit_pack(&centred, p.eta, eb));
    }
    for poly in t0 {
        let mut centred = ZERO;
        for (c, &raw) in centred.iter_mut().zip(poly.iter()) {
            *c = centered(raw, Q);
        }
        out.extend_from_slice(&bit_pack(&centred, 1 << (D - 1), D as usize));
    }
    out
}

type DecodedSk = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<Poly>, Vec<Poly>, Vec<Poly>);

fn sk_decode(sk: &[u8], p: &Params) -> DecodedSk {
    let eb = p.eta_bits();
    let s_len = 32 * eb;
    let t_len = 32 * D as usize;
    let rho = sk[..32].to_vec();
    let key = sk[32..64].to_vec();
    let tr = sk[64..128].to_vec();
    let mut at = 128;
    let mut s1 = Vec::with_capacity(p.l);
    for _ in 0..p.l {
        let poly = bit_unpack(&sk[at..at + s_len], p.eta, eb);
        s1.push(reduce_poly(&poly));
        at += s_len;
    }
    let mut s2 = Vec::with_capacity(p.k);
    for _ in 0..p.k {
        let poly = bit_unpack(&sk[at..at + s_len], p.eta, eb);
        s2.push(reduce_poly(&poly));
        at += s_len;
    }
    let mut t0 = Vec::with_capacity(p.k);
    for _ in 0..p.k {
        let poly = bit_unpack(&sk[at..at + t_len], 1 << (D - 1), D as usize);
        t0.push(reduce_poly(&poly));
        at += t_len;
    }
    (rho, key, tr, s1, s2, t0)
}

fn reduce_poly(p: &Poly) -> Poly {
    let mut out = ZERO;
    for (o, &c) in out.iter_mut().zip(p.iter()) {
        *o = c.rem_euclid(Q);
    }
    out
}

fn w1_encode(w1: &[Poly], p: &Params) -> Vec<u8> {
    let bits = p.w1_bits();
    let mut out = Vec::new();
    for poly in w1 {
        out.extend_from_slice(&simple_bit_pack(poly, bits));
    }
    out
}

fn hint_bit_pack(hint: &[[bool; N]], p: &Params) -> Vec<u8> {
    let mut y = vec![0u8; p.omega + p.k];
    let mut index = 0usize;
    for (i, row) in hint.iter().enumerate() {
        for (j, &set) in row.iter().enumerate() {
            if set {
                y[index] = j as u8;
                index += 1;
            }
        }
        y[p.omega + i] = index as u8;
    }
    y
}

fn hint_bit_unpack(y: &[u8], p: &Params) -> Option<Vec<[bool; N]>> {
    let mut hint = vec![[false; N]; p.k];
    let mut index = 0usize;
    for i in 0..p.k {
        let end = y[p.omega + i] as usize;
        if end < index || end > p.omega {
            return None;
        }
        let first = index;
        while index < end {
            if index > first && y[index - 1] >= y[index] {
                return None;
            }
            hint[i][y[index] as usize] = true;
            index += 1;
        }
    }
    for &b in &y[index..p.omega] {
        if b != 0 {
            return None;
        }
    }
    Some(hint)
}

fn sig_encode(c_tilde: &[u8], z: &[Poly], hint: &[[bool; N]], p: &Params) -> Vec<u8> {
    let bits = p.z_bits();
    let mut out = c_tilde.to_vec();
    for poly in z {
        let mut centred = ZERO;
        for (c, &raw) in centred.iter_mut().zip(poly.iter()) {
            *c = centered(raw, Q);
        }
        out.extend_from_slice(&bit_pack(&centred, p.gamma1, bits));
    }
    out.extend_from_slice(&hint_bit_pack(hint, p));
    out
}

type DecodedSig = (Vec<u8>, Vec<Poly>, Vec<[bool; N]>);

fn sig_decode(sig: &[u8], p: &Params) -> Option<DecodedSig> {
    let bits = p.z_bits();
    let ct_len = p.lambda / 4;
    let z_len = 32 * bits;
    if sig.len() != p.signature_len() {
        return None;
    }
    let c_tilde = sig[..ct_len].to_vec();
    let mut at = ct_len;
    let mut z = Vec::with_capacity(p.l);
    for _ in 0..p.l {
        z.push(reduce_poly(&bit_unpack(
            &sig[at..at + z_len],
            p.gamma1,
            bits,
        )));
        at += z_len;
    }
    let hint = hint_bit_unpack(&sig[at..], p)?;
    Some((c_tilde, z, hint))
}

// ---------------------------------------------------------------------------
// The three operations
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error {
    BadLength {
        what: &'static str,
        want: usize,
        got: usize,
    },
    Malformed(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadLength { what, want, got } => {
                write!(f, "ML-DSA {what} must be {want} bytes, got {got}")
            }
            Error::Malformed(why) => write!(f, "malformed ML-DSA input: {why}"),
        }
    }
}

/// An ML-DSA key pair, held as the encoded forms FIPS 204 defines. Keeping the
/// encoded bytes rather than the decoded polynomials is deliberate: the encoded
/// form is what gets stored, transmitted and signed over, and a round trip
/// through it on every use is a standing check that the codec agrees with itself.
#[derive(Clone)]
pub struct KeyPair {
    pub params: Params,
    pub public: Vec<u8>,
    pub secret: Vec<u8>,
}

/// `ML-DSA.KeyGen_internal` (Algorithm 6).
pub fn keygen(params: &Params, xi: &[u8; 32]) -> KeyPair {
    let z = zetas();
    let seed = h(&[xi, &[params.k as u8], &[params.l as u8]], 128);
    let (rho, rest) = seed.split_at(32);
    let (rhop, key) = rest.split_at(64);

    let a = expand_a(rho, params);
    let (s1, s2) = expand_s(rhop, params);

    let mut s1_hat = s1.clone();
    for p in s1_hat.iter_mut() {
        ntt(p, &z);
    }

    let mut t1 = Vec::with_capacity(params.k);
    let mut t0 = Vec::with_capacity(params.k);
    for (row, s2i) in a.iter().zip(s2.iter()) {
        let mut acc = ZERO;
        for (aij, s1j) in row.iter().zip(s1_hat.iter()) {
            acc = poly_add(&acc, &poly_mul_ntt(aij, s1j));
        }
        inv_ntt(&mut acc, &z);
        let t = poly_add(&acc, s2i);
        let mut hi = ZERO;
        let mut lo = ZERO;
        for (n, &tc) in t.iter().enumerate() {
            let (a1, a0) = power2round(tc);
            hi[n] = a1;
            lo[n] = a0.rem_euclid(Q);
        }
        t1.push(hi);
        t0.push(lo);
    }

    let public = pk_encode(rho, &t1);
    let tr = h(&[&public], 64);
    let secret = sk_encode(rho, key, &tr, &s1, &s2, &t0, params);
    KeyPair {
        params: *params,
        public,
        secret,
    }
}

/// The `M'` of FIPS 204 Algorithm 2: a domain-separated wrapper carrying the
/// context string, so a signature over a message in one context cannot be
/// replayed in another.
fn message_prime(message: &[u8], ctx: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(2 + ctx.len() + message.len());
    m.push(0);
    m.push(ctx.len() as u8);
    m.extend_from_slice(ctx);
    m.extend_from_slice(message);
    m
}

/// `ML-DSA.Sign` (Algorithm 2) with the deterministic variant of
/// `Sign_internal`: `rnd` is 32 zero bytes.
///
/// Deterministic on purpose. §01.10's writer rule W1 says the same inputs
/// produce the same bytes, and a container whose signature changes on every
/// signing would break the reproducibility this format is built on — the same
/// reasoning that puts RFC 6979 nonces in `p256.rs`.
pub fn sign(kp: &KeyPair, message: &[u8], ctx: &[u8]) -> Result<Vec<u8>, Error> {
    if ctx.len() > 255 {
        return Err(Error::Malformed("context strings are limited to 255 bytes"));
    }
    sign_internal(kp, &message_prime(message, ctx))
}

fn sign_internal(kp: &KeyPair, mp: &[u8]) -> Result<Vec<u8>, Error> {
    let p = &kp.params;
    if kp.secret.len() != p.secret_key_len() {
        return Err(Error::BadLength {
            what: "secret key",
            want: p.secret_key_len(),
            got: kp.secret.len(),
        });
    }
    let z = zetas();
    let (rho, key, tr, s1, s2, t0) = sk_decode(&kp.secret, p);

    let a = expand_a(&rho, p);
    let mut s1_hat = s1;
    let mut s2_hat = s2;
    let mut t0_hat = t0;
    for poly in s1_hat
        .iter_mut()
        .chain(s2_hat.iter_mut())
        .chain(t0_hat.iter_mut())
    {
        ntt(poly, &z);
    }

    let mu = h(&[&tr, mp], 64);
    let rnd = [0u8; 32];
    let rhopp = h(&[&key, &rnd, &mu], 64);

    let mut kappa: u16 = 0;
    loop {
        let y = expand_mask(&rhopp, kappa, p);
        kappa += p.l as u16;

        let mut w = Vec::with_capacity(p.k);
        let mut y_hat = y.clone();
        for poly in y_hat.iter_mut() {
            ntt(poly, &z);
        }
        for row in &a {
            let mut acc = ZERO;
            for (aij, yj) in row.iter().zip(y_hat.iter()) {
                acc = poly_add(&acc, &poly_mul_ntt(aij, yj));
            }
            inv_ntt(&mut acc, &z);
            w.push(acc);
        }

        let mut w1 = Vec::with_capacity(p.k);
        for poly in &w {
            let mut hi = ZERO;
            for (n, &c) in poly.iter().enumerate() {
                hi[n] = high_bits(c, p.gamma2);
            }
            w1.push(hi);
        }

        let c_tilde = h(&[&mu, &w1_encode(&w1, p)], p.lambda / 4);
        let mut c_hat = sample_in_ball(&c_tilde, p.tau);
        ntt(&mut c_hat, &z);

        let mut cs1 = Vec::with_capacity(p.l);
        for poly in &s1_hat {
            let mut v = poly_mul_ntt(&c_hat, poly);
            inv_ntt(&mut v, &z);
            cs1.push(v);
        }
        let mut cs2 = Vec::with_capacity(p.k);
        for poly in &s2_hat {
            let mut v = poly_mul_ntt(&c_hat, poly);
            inv_ntt(&mut v, &z);
            cs2.push(v);
        }

        let zed: Vec<Poly> = y
            .iter()
            .zip(cs1.iter())
            .map(|(a, b)| poly_add(a, b))
            .collect();
        let r: Vec<Poly> = w
            .iter()
            .zip(cs2.iter())
            .map(|(a, b)| poly_sub(a, b))
            .collect();

        let z_norm = zed.iter().map(poly_norm).max().unwrap_or(0);
        let mut r0_norm = 0;
        for poly in &r {
            for &c in poly.iter() {
                r0_norm = r0_norm.max(low_bits(c, p.gamma2).abs());
            }
        }
        if z_norm >= p.gamma1 - p.beta || r0_norm >= p.gamma2 - p.beta {
            continue;
        }

        let mut ct0 = Vec::with_capacity(p.k);
        for poly in &t0_hat {
            let mut v = poly_mul_ntt(&c_hat, poly);
            inv_ntt(&mut v, &z);
            ct0.push(v);
        }
        let ct0_norm = ct0.iter().map(poly_norm).max().unwrap_or(0);

        let mut hint = vec![[false; N]; p.k];
        let mut ones = 0usize;
        for (i, (ct0i, ri)) in ct0.iter().zip(r.iter()).enumerate() {
            for n in 0..N {
                let zc = Q - ct0i[n] % Q;
                let rc = addq(ri[n], ct0i[n]);
                if make_hint(zc, rc, p.gamma2) {
                    hint[i][n] = true;
                    ones += 1;
                }
            }
        }
        if ct0_norm >= p.gamma2 || ones > p.omega {
            continue;
        }

        return Ok(sig_encode(&c_tilde, &zed, &hint, p));
    }
}

/// `ML-DSA.Verify` (Algorithm 3).
pub fn verify(params: &Params, public: &[u8], message: &[u8], ctx: &[u8], sig: &[u8]) -> bool {
    if public.len() != params.public_key_len() || ctx.len() > 255 {
        return false;
    }
    verify_internal(params, public, &message_prime(message, ctx), sig)
}

fn verify_internal(p: &Params, public: &[u8], mp: &[u8], sig: &[u8]) -> bool {
    let z = zetas();
    let Some((c_tilde, zed, hint)) = sig_decode(sig, p) else {
        return false;
    };
    let (rho, t1) = pk_decode(public, p);
    let a = expand_a(&rho, p);
    let tr = h(&[public], 64);
    let mu = h(&[&tr, mp], 64);

    if zed.iter().map(poly_norm).max().unwrap_or(0) >= p.gamma1 - p.beta {
        return false;
    }

    let mut c_hat = sample_in_ball(&c_tilde, p.tau);
    ntt(&mut c_hat, &z);

    let mut z_hat = zed.clone();
    for poly in z_hat.iter_mut() {
        ntt(poly, &z);
    }

    // t1 * 2^d, in the NTT domain
    let mut t1_hat = Vec::with_capacity(p.k);
    for poly in &t1 {
        let mut scaled = ZERO;
        for (n, &c) in poly.iter().enumerate() {
            scaled[n] = mulq(c, 1 << D);
        }
        ntt(&mut scaled, &z);
        t1_hat.push(scaled);
    }

    let mut w1 = Vec::with_capacity(p.k);
    for ((row, t1i), hi_row) in a.iter().zip(t1_hat.iter()).zip(hint.iter()) {
        let mut acc = ZERO;
        for (aij, zj) in row.iter().zip(z_hat.iter()) {
            acc = poly_add(&acc, &poly_mul_ntt(aij, zj));
        }
        acc = poly_sub(&acc, &poly_mul_ntt(&c_hat, t1i));
        inv_ntt(&mut acc, &z);
        let mut hi = ZERO;
        for (n, &c) in acc.iter().enumerate() {
            hi[n] = use_hint(hi_row[n], c, p.gamma2);
        }
        w1.push(hi);
    }

    let expected = h(&[&mu, &w1_encode(&w1, p)], p.lambda / 4);
    // Constant-time comparison is not the point here — both sides are public —
    // but a length check is, because a truncated c~ must not compare equal.
    expected.len() == c_tilde.len() && expected == c_tilde
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Keccak permutation is the foundation everything else here stands on,
    // so it is checked against the published SHAKE outputs for the empty string
    // before any ML-DSA vector is looked at. A wrong permutation makes every
    // later failure impossible to localise.

    #[test]
    fn the_derived_lengths_match_fips_204s_table() {
        for (p, pk, sk, sig) in [
            (ML_DSA_44, 1312, 2560, 2420),
            (ML_DSA_65, 1952, 4032, 3309),
            (ML_DSA_87, 2592, 4896, 4627),
        ] {
            assert_eq!(p.public_key_len(), pk, "{} public key", p.name);
            assert_eq!(p.secret_key_len(), sk, "{} secret key", p.name);
            assert_eq!(p.signature_len(), sig, "{} signature", p.name);
        }
    }

    #[test]
    fn the_ntt_is_invertible() {
        let z = zetas();
        let mut p = ZERO;
        for (i, c) in p.iter_mut().enumerate() {
            *c = ((i * 7919 + 13) % Q as usize) as i32;
        }
        let original = p;
        ntt(&mut p, &z);
        inv_ntt(&mut p, &z);
        assert_eq!(p, original);
    }

    #[test]
    fn the_ntt_agrees_with_schoolbook_negacyclic_multiplication() {
        // The NTT's whole job is to make this product cheap; if the two disagree
        // the transform is wrong in a way no round-trip test can see.
        let z = zetas();
        let mut a = ZERO;
        let mut b = ZERO;
        for i in 0..N {
            a[i] = ((i * 31 + 7) % 1000) as i32;
            b[i] = ((i * 17 + 3) % 1000) as i32;
        }
        let mut school = ZERO;
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let k = (i + j) % N;
                let term = mulq(ai, bj);
                if i + j < N {
                    school[k] = addq(school[k], term);
                } else {
                    school[k] = subq(school[k], term);
                }
            }
        }
        let mut an = a;
        let mut bn = b;
        ntt(&mut an, &z);
        ntt(&mut bn, &z);
        let mut prod = poly_mul_ntt(&an, &bn);
        inv_ntt(&mut prod, &z);
        assert_eq!(prod, school);
    }

    #[test]
    fn decompose_reconstructs_its_input() {
        for p in [ML_DSA_44, ML_DSA_65] {
            for r in [0, 1, 17, 95231, 95232, 4190208, Q - 2, Q - 1] {
                let (r1, r0) = decompose(r, p.gamma2);
                let back = (r1 * 2 * p.gamma2 + r0).rem_euclid(Q);
                assert_eq!(back, r.rem_euclid(Q), "{} r={r}", p.name);
            }
        }
    }

    #[test]
    fn power2round_reconstructs_its_input() {
        for r in [0, 1, 4095, 4096, 8191, 8192, Q - 1] {
            let (r1, r0) = power2round(r);
            assert_eq!(r1 * (1 << D) + r0, r);
        }
    }

    #[test]
    fn bit_packing_round_trips_at_every_width_used() {
        for bits in [3usize, 4, 6, 10, 13, 18, 20] {
            let mut p = ZERO;
            for (i, c) in p.iter_mut().enumerate() {
                *c = (i as i32) % (1 << bits);
            }
            let packed = simple_bit_pack(&p, bits);
            assert_eq!(packed.len(), 32 * bits);
            assert_eq!(simple_bit_unpack(&packed, bits), p);
        }
    }

    #[test]
    fn sample_in_ball_has_exactly_tau_nonzero_coefficients() {
        for p in ALL {
            let c = sample_in_ball(&[7u8; 64][..p.lambda / 4], p.tau);
            assert_eq!(c.iter().filter(|&&x| x != 0).count(), p.tau, "{}", p.name);
            for &x in c.iter() {
                assert!(x == 0 || x == 1 || x == Q - 1, "{} coefficient {x}", p.name);
            }
        }
    }

    #[test]
    fn a_signature_verifies_and_a_tampered_one_does_not() {
        let kp = keygen(&ML_DSA_44, &[42u8; 32]);
        assert_eq!(kp.public.len(), ML_DSA_44.public_key_len());
        assert_eq!(kp.secret.len(), ML_DSA_44.secret_key_len());

        let sig = sign(&kp, b"a model exists once", b"omni").unwrap();
        assert_eq!(sig.len(), ML_DSA_44.signature_len());
        assert!(verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists once",
            b"omni",
            &sig
        ));

        // A different context is a different message (Algorithm 2's M').
        assert!(!verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists once",
            b"other",
            &sig
        ));
        assert!(!verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists twice",
            b"omni",
            &sig
        ));

        let mut bad = sig.clone();
        bad[0] ^= 1;
        assert!(!verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists once",
            b"omni",
            &bad
        ));
        let mut bad = sig.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xff;
        assert!(!verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists once",
            b"omni",
            &bad
        ));

        assert!(!verify(
            &ML_DSA_44,
            &kp.public,
            b"a model exists once",
            b"omni",
            &sig[..sig.len() - 1]
        ));
    }

    #[test]
    fn signing_is_deterministic() {
        let kp = keygen(&ML_DSA_44, &[9u8; 32]);
        let a = sign(&kp, b"same", b"").unwrap();
        let b = sign(&kp, b"same", b"").unwrap();
        assert_eq!(a, b, "the deterministic variant must not vary");
    }

    #[test]
    fn a_wrong_key_does_not_verify() {
        let a = keygen(&ML_DSA_44, &[1u8; 32]);
        let b = keygen(&ML_DSA_44, &[2u8; 32]);
        let sig = sign(&a, b"m", b"").unwrap();
        assert!(!verify(&ML_DSA_44, &b.public, b"m", b"", &sig));
    }
}
