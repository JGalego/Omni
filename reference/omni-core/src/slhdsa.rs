//! SLH-DSA (FIPS 205) — the hash-based signature §12.5.1 marks MAY, and the last
//! of the three algorithms that section names.
//!
//! It is here for a reason ML-DSA does not cover. ML-DSA's security rests on a
//! lattice problem that is believed hard and is twenty years younger than the
//! discrete logarithm it replaces; SLH-DSA's rests on nothing but the hash
//! function, and §12.11's whole subject is what to do when a belief about a
//! primitive turns out to be wrong. For an archival format that is not a
//! hypothetical: a container signed in 2026 and verified in 2060 cannot be
//! re-signed by whoever wrote it. SLH-DSA is the conservative option to
//! dual-sign with (§12.5.1 recommends exactly that against adversary A7), and it
//! costs a 7 856-byte signature to have one.
//!
//! The six SHAKE parameter sets are implemented. The six SHA2 sets are not, and
//! are refused by name: they are not a smaller amount of work than these were,
//! they use a different address encoding and an MGF1 construction, and a
//! half-done second family would be the thing this crate keeps declining to do.
//!
//! Checked against NIST's own ACVP known-answer vectors, for the same reason
//! ML-DSA is: a hash-based signature has a great many self-consistent
//! interpretations. Every address field, every tree index, and the order the two
//! children of a node are concatenated in are all invisible to a round trip and
//! all fatal to interoperability.

use crate::shake::shake256;

// ---------------------------------------------------------------------------
// Parameters (FIPS 205 §11, Table 2)
// ---------------------------------------------------------------------------

/// One SLH-DSA parameter set. `s` sets sign slowly and produce short signatures;
/// `f` sets sign fast and produce long ones, which is the whole axis the suffix
/// names.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub name: &'static str,
    /// Security parameter, in bytes.
    pub n: usize,
    /// Total hypertree height.
    pub h: usize,
    /// Number of hypertree layers.
    pub d: usize,
    /// FORS tree height.
    pub a: usize,
    /// Number of FORS trees.
    pub k: usize,
    /// Message digest length, in bytes.
    pub m: usize,
}

/// The Winternitz parameter. FIPS 205 fixes `lg_w = 4` for every set, so `w` is
/// 16 and a message byte is exactly two chain indices.
const LG_W: usize = 4;
const W: usize = 1 << LG_W;

pub const SHAKE_128S: Params = Params {
    name: "SLH-DSA-SHAKE-128s",
    n: 16,
    h: 63,
    d: 7,
    a: 12,
    k: 14,
    m: 30,
};
pub const SHAKE_128F: Params = Params {
    name: "SLH-DSA-SHAKE-128f",
    n: 16,
    h: 66,
    d: 22,
    a: 6,
    k: 33,
    m: 34,
};
pub const SHAKE_192S: Params = Params {
    name: "SLH-DSA-SHAKE-192s",
    n: 24,
    h: 63,
    d: 7,
    a: 14,
    k: 17,
    m: 39,
};
pub const SHAKE_192F: Params = Params {
    name: "SLH-DSA-SHAKE-192f",
    n: 24,
    h: 66,
    d: 22,
    a: 8,
    k: 33,
    m: 42,
};
pub const SHAKE_256S: Params = Params {
    name: "SLH-DSA-SHAKE-256s",
    n: 32,
    h: 64,
    d: 8,
    a: 14,
    k: 22,
    m: 47,
};
pub const SHAKE_256F: Params = Params {
    name: "SLH-DSA-SHAKE-256f",
    n: 32,
    h: 68,
    d: 17,
    a: 9,
    k: 35,
    m: 49,
};

pub const ALL: [Params; 6] = [
    SHAKE_128S, SHAKE_128F, SHAKE_192S, SHAKE_192F, SHAKE_256S, SHAKE_256F,
];

impl Params {
    pub fn by_name(name: &str) -> Option<Params> {
        ALL.iter().copied().find(|p| p.name == name)
    }

    /// Height of one XMSS tree: `h / d`.
    pub fn hp(&self) -> usize {
        self.h / self.d
    }

    /// `len1 = ceil(8n / lg_w)`, the message part of a WOTS+ chain vector.
    fn len1(&self) -> usize {
        8 * self.n / LG_W
    }

    /// `len2`, the checksum part.
    fn len2(&self) -> usize {
        let max = self.len1() * (W - 1);
        // floor(log2(max) / lg_w) + 1, without floating point: log2 by bit width.
        let bits = usize::BITS as usize - max.leading_zeros() as usize; // = floor(log2)+1
        (bits - 1) / LG_W + 1
    }

    fn len(&self) -> usize {
        self.len1() + self.len2()
    }

    pub fn public_key_len(&self) -> usize {
        2 * self.n
    }

    pub fn secret_key_len(&self) -> usize {
        4 * self.n
    }

    pub fn signature_len(&self) -> usize {
        (1 + self.k * (1 + self.a) + self.h + self.d * self.len()) * self.n
    }
}

// ---------------------------------------------------------------------------
// Addresses (FIPS 205 §4.2)
// ---------------------------------------------------------------------------

const WOTS_HASH: u32 = 0;
const WOTS_PK: u32 = 1;
const TREE: u32 = 2;
const FORS_TREE: u32 = 3;
const FORS_ROOTS: u32 = 4;
const WOTS_PRF: u32 = 5;
const FORS_PRF: u32 = 6;

/// A 32-byte address, the SHAKE family's form. Every hash in SLH-DSA is
/// domain-separated by one of these, and getting a single field wrong produces a
/// scheme that is entirely self-consistent and interoperates with nothing —
/// which is why this is a distinct type with named setters rather than byte
/// arithmetic at each call site.
#[derive(Clone, Copy, Default)]
struct Adrs([u8; 32]);

impl Adrs {
    fn new() -> Adrs {
        Adrs([0; 32])
    }

    fn set_layer(&mut self, l: u32) {
        self.0[0..4].copy_from_slice(&l.to_be_bytes());
    }

    /// The tree address is 12 bytes; only the low 8 are ever non-zero for the
    /// heights FIPS 205 defines, and the high 4 stay zero rather than being
    /// omitted, because they are hashed.
    fn set_tree(&mut self, t: u64) {
        self.0[4..8].fill(0);
        self.0[8..16].copy_from_slice(&t.to_be_bytes());
    }

    /// Sets the type *and clears the three words after it*, which is what
    /// `setTypeAndClear` means and is load-bearing: a stale key-pair address
    /// left behind by the previous use of the address changes the hash.
    fn set_type_and_clear(&mut self, t: u32) {
        self.0[16..20].copy_from_slice(&t.to_be_bytes());
        self.0[20..32].fill(0);
    }

    fn set_key_pair(&mut self, i: u32) {
        self.0[20..24].copy_from_slice(&i.to_be_bytes());
    }

    fn key_pair(&self) -> u32 {
        u32::from_be_bytes(self.0[20..24].try_into().expect("4 bytes"))
    }

    fn set_chain(&mut self, i: u32) {
        self.0[24..28].copy_from_slice(&i.to_be_bytes());
    }

    fn set_hash(&mut self, i: u32) {
        self.0[28..32].copy_from_slice(&i.to_be_bytes());
    }

    fn set_tree_height(&mut self, i: u32) {
        self.0[24..28].copy_from_slice(&i.to_be_bytes());
    }

    fn set_tree_index(&mut self, i: u32) {
        self.0[28..32].copy_from_slice(&i.to_be_bytes());
    }

    fn tree_index(&self) -> u32 {
        u32::from_be_bytes(self.0[28..32].try_into().expect("4 bytes"))
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// The six hash functions (FIPS 205 §11.2.1). For the SHAKE sets they are all
// SHAKE256 over different inputs, which is why this family is the one to
// implement first.
// ---------------------------------------------------------------------------

fn h_msg(p: &Params, r: &[u8], pk_seed: &[u8], pk_root: &[u8], m: &[u8]) -> Vec<u8> {
    shake256(&[r, pk_seed, pk_root, m], p.m)
}

fn prf(p: &Params, pk_seed: &[u8], sk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    shake256(&[pk_seed, adrs.as_bytes(), sk_seed], p.n)
}

fn prf_msg(p: &Params, sk_prf: &[u8], opt_rand: &[u8], m: &[u8]) -> Vec<u8> {
    shake256(&[sk_prf, opt_rand, m], p.n)
}

fn f(p: &Params, pk_seed: &[u8], adrs: &Adrs, m1: &[u8]) -> Vec<u8> {
    shake256(&[pk_seed, adrs.as_bytes(), m1], p.n)
}

fn hh(p: &Params, pk_seed: &[u8], adrs: &Adrs, m2: &[u8]) -> Vec<u8> {
    shake256(&[pk_seed, adrs.as_bytes(), m2], p.n)
}

fn t_l(p: &Params, pk_seed: &[u8], adrs: &Adrs, m: &[u8]) -> Vec<u8> {
    shake256(&[pk_seed, adrs.as_bytes(), m], p.n)
}

// ---------------------------------------------------------------------------
// base_2b (FIPS 205 Algorithm 4)
// ---------------------------------------------------------------------------

/// Reads `out_len` big-endian `b`-bit integers out of `x`.
fn base_2b(x: &[u8], b: usize, out_len: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(out_len);
    let mut at = 0usize;
    let mut bits = 0usize;
    let mut total: u64 = 0;
    for _ in 0..out_len {
        while bits < b {
            total = (total << 8) | x[at] as u64;
            at += 1;
            bits += 8;
        }
        bits -= b;
        out.push(((total >> bits) & ((1u64 << b) - 1)) as u32);
    }
    out
}

// ---------------------------------------------------------------------------
// WOTS+ (FIPS 205 §5)
// ---------------------------------------------------------------------------

fn chain(p: &Params, x: &[u8], i: u32, s: u32, pk_seed: &[u8], adrs: &mut Adrs) -> Vec<u8> {
    let mut tmp = x.to_vec();
    for j in i..i + s {
        adrs.set_hash(j);
        tmp = f(p, pk_seed, adrs, &tmp);
    }
    tmp
}

fn wots_pk_gen(p: &Params, sk_seed: &[u8], pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let mut sk_adrs = *adrs;
    sk_adrs.set_type_and_clear(WOTS_PRF);
    sk_adrs.set_key_pair(adrs.key_pair());
    let mut tmp = Vec::with_capacity(p.len() * p.n);
    let mut wa = *adrs;
    for i in 0..p.len() as u32 {
        sk_adrs.set_chain(i);
        let sk = prf(p, pk_seed, sk_seed, &sk_adrs);
        wa.set_chain(i);
        tmp.extend_from_slice(&chain(p, &sk, 0, (W - 1) as u32, pk_seed, &mut wa));
    }
    let mut pk_adrs = *adrs;
    pk_adrs.set_type_and_clear(WOTS_PK);
    pk_adrs.set_key_pair(adrs.key_pair());
    t_l(p, pk_seed, &pk_adrs, &tmp)
}

/// The message plus its checksum, as `len` base-`w` digits.
fn wots_digits(p: &Params, m: &[u8]) -> Vec<u32> {
    let mut msg = base_2b(m, LG_W, p.len1());
    let csum: u32 = msg.iter().map(|d| (W as u32 - 1) - d).sum();
    // The checksum is left-shifted so that its `len2 * lg_w` bits sit at the top
    // of the bytes it is read back out of.
    let shift = (8 - (p.len2() * LG_W % 8)) % 8;
    let csum = (csum as u64) << shift;
    let bytes = (p.len2() * LG_W).div_ceil(8);
    let mut buf = vec![0u8; bytes];
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = (csum >> (8 * (bytes - 1 - i))) as u8;
    }
    msg.extend(base_2b(&buf, LG_W, p.len2()));
    msg
}

fn wots_sign(p: &Params, m: &[u8], sk_seed: &[u8], pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let msg = wots_digits(p, m);
    let mut sk_adrs = *adrs;
    sk_adrs.set_type_and_clear(WOTS_PRF);
    sk_adrs.set_key_pair(adrs.key_pair());
    let mut sig = Vec::with_capacity(p.len() * p.n);
    let mut wa = *adrs;
    for (i, d) in msg.iter().enumerate() {
        sk_adrs.set_chain(i as u32);
        let sk = prf(p, pk_seed, sk_seed, &sk_adrs);
        wa.set_chain(i as u32);
        sig.extend_from_slice(&chain(p, &sk, 0, *d, pk_seed, &mut wa));
    }
    sig
}

fn wots_pk_from_sig(p: &Params, sig: &[u8], m: &[u8], pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let msg = wots_digits(p, m);
    let mut tmp = Vec::with_capacity(p.len() * p.n);
    let mut wa = *adrs;
    for (i, d) in msg.iter().enumerate() {
        wa.set_chain(i as u32);
        let part = &sig[i * p.n..(i + 1) * p.n];
        tmp.extend_from_slice(&chain(p, part, *d, (W as u32 - 1) - *d, pk_seed, &mut wa));
    }
    let mut pk_adrs = *adrs;
    pk_adrs.set_type_and_clear(WOTS_PK);
    pk_adrs.set_key_pair(adrs.key_pair());
    t_l(p, pk_seed, &pk_adrs, &tmp)
}

// ---------------------------------------------------------------------------
// XMSS (FIPS 205 §6)
// ---------------------------------------------------------------------------

fn xmss_node(p: &Params, sk_seed: &[u8], i: u32, z: usize, pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    if z == 0 {
        let mut a = *adrs;
        a.set_type_and_clear(WOTS_HASH);
        a.set_key_pair(i);
        wots_pk_gen(p, sk_seed, pk_seed, &a)
    } else {
        let l = xmss_node(p, sk_seed, 2 * i, z - 1, pk_seed, adrs);
        let r = xmss_node(p, sk_seed, 2 * i + 1, z - 1, pk_seed, adrs);
        let mut a = *adrs;
        a.set_type_and_clear(TREE);
        a.set_tree_height(z as u32);
        a.set_tree_index(i);
        let mut both = l;
        both.extend_from_slice(&r);
        hh(p, pk_seed, &a, &both)
    }
}

fn xmss_sign(
    p: &Params,
    m: &[u8],
    sk_seed: &[u8],
    idx: u32,
    pk_seed: &[u8],
    adrs: &Adrs,
) -> Vec<u8> {
    let mut auth = Vec::with_capacity(p.hp() * p.n);
    for j in 0..p.hp() {
        let k = (idx >> j) ^ 1;
        auth.extend_from_slice(&xmss_node(p, sk_seed, k, j, pk_seed, adrs));
    }
    let mut a = *adrs;
    a.set_type_and_clear(WOTS_HASH);
    a.set_key_pair(idx);
    let mut sig = wots_sign(p, m, sk_seed, pk_seed, &a);
    sig.extend_from_slice(&auth);
    sig
}

fn xmss_pk_from_sig(
    p: &Params,
    idx: u32,
    sig: &[u8],
    m: &[u8],
    pk_seed: &[u8],
    adrs: &Adrs,
) -> Vec<u8> {
    let mut a = *adrs;
    a.set_type_and_clear(WOTS_HASH);
    a.set_key_pair(idx);
    let wots_len = p.len() * p.n;
    let mut node = wots_pk_from_sig(p, &sig[..wots_len], m, pk_seed, &a);
    let auth = &sig[wots_len..];

    a.set_type_and_clear(TREE);
    a.set_tree_index(idx);
    for k in 0..p.hp() {
        a.set_tree_height(k as u32 + 1);
        let sibling = &auth[k * p.n..(k + 1) * p.n];
        let mut both = Vec::with_capacity(2 * p.n);
        if (idx >> k).is_multiple_of(2) {
            a.set_tree_index(a.tree_index() / 2);
            both.extend_from_slice(&node);
            both.extend_from_slice(sibling);
        } else {
            a.set_tree_index((a.tree_index() - 1) / 2);
            both.extend_from_slice(sibling);
            both.extend_from_slice(&node);
        }
        node = hh(p, pk_seed, &a, &both);
    }
    node
}

// ---------------------------------------------------------------------------
// The hypertree (FIPS 205 §7)
// ---------------------------------------------------------------------------

fn ht_sign(
    p: &Params,
    m: &[u8],
    sk_seed: &[u8],
    pk_seed: &[u8],
    idx_tree: u64,
    idx_leaf: u32,
) -> Vec<u8> {
    let mut adrs = Adrs::new();
    adrs.set_layer(0);
    adrs.set_tree(idx_tree);
    let mut sig = xmss_sign(p, m, sk_seed, idx_leaf, pk_seed, &adrs);
    let mut root = xmss_pk_from_sig(p, idx_leaf, &sig, m, pk_seed, &adrs);
    let mut idx_tree = idx_tree;
    for j in 1..p.d {
        // Each layer up consumes h' bits of the tree index: the leaf this layer
        // signs at is the low h' bits, and the tree address is what is left.
        let leaf = (idx_tree % (1u64 << p.hp())) as u32;
        idx_tree >>= p.hp();
        adrs.set_layer(j as u32);
        adrs.set_tree(idx_tree);
        let part = xmss_sign(p, &root, sk_seed, leaf, pk_seed, &adrs);
        if j < p.d - 1 {
            root = xmss_pk_from_sig(p, leaf, &part, &root, pk_seed, &adrs);
        }
        sig.extend_from_slice(&part);
    }
    sig
}

fn ht_verify(
    p: &Params,
    m: &[u8],
    sig: &[u8],
    pk_seed: &[u8],
    idx_tree: u64,
    idx_leaf: u32,
    pk_root: &[u8],
) -> bool {
    let per = (p.hp() + p.len()) * p.n;
    let mut adrs = Adrs::new();
    adrs.set_layer(0);
    adrs.set_tree(idx_tree);
    let mut node = xmss_pk_from_sig(p, idx_leaf, &sig[..per], m, pk_seed, &adrs);
    let mut idx_tree = idx_tree;
    for j in 1..p.d {
        let leaf = (idx_tree % (1u64 << p.hp())) as u32;
        idx_tree >>= p.hp();
        adrs.set_layer(j as u32);
        adrs.set_tree(idx_tree);
        node = xmss_pk_from_sig(p, leaf, &sig[j * per..(j + 1) * per], &node, pk_seed, &adrs);
    }
    node == pk_root
}

// ---------------------------------------------------------------------------
// FORS (FIPS 205 §8)
// ---------------------------------------------------------------------------

fn fors_sk_gen(p: &Params, sk_seed: &[u8], pk_seed: &[u8], adrs: &Adrs, idx: u32) -> Vec<u8> {
    let mut a = *adrs;
    a.set_type_and_clear(FORS_PRF);
    a.set_key_pair(adrs.key_pair());
    a.set_tree_index(idx);
    prf(p, pk_seed, sk_seed, &a)
}

fn fors_node(p: &Params, sk_seed: &[u8], i: u32, z: usize, pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let mut a = *adrs;
    if z == 0 {
        let sk = fors_sk_gen(p, sk_seed, pk_seed, adrs, i);
        a.set_tree_height(0);
        a.set_tree_index(i);
        f(p, pk_seed, &a, &sk)
    } else {
        let l = fors_node(p, sk_seed, 2 * i, z - 1, pk_seed, adrs);
        let r = fors_node(p, sk_seed, 2 * i + 1, z - 1, pk_seed, adrs);
        a.set_tree_height(z as u32);
        a.set_tree_index(i);
        let mut both = l;
        both.extend_from_slice(&r);
        hh(p, pk_seed, &a, &both)
    }
}

fn fors_sign(p: &Params, md: &[u8], sk_seed: &[u8], pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let indices = base_2b(md, p.a, p.k);
    let mut sig = Vec::with_capacity(p.k * (1 + p.a) * p.n);
    for (i, &idx) in indices.iter().enumerate() {
        let base = (i as u32) << p.a;
        sig.extend_from_slice(&fors_sk_gen(p, sk_seed, pk_seed, adrs, base + idx));
        for j in 0..p.a {
            let s = (idx >> j) ^ 1;
            let node_index = ((i as u32) << (p.a - j)) + s;
            sig.extend_from_slice(&fors_node(p, sk_seed, node_index, j, pk_seed, adrs));
        }
    }
    sig
}

fn fors_pk_from_sig(p: &Params, sig: &[u8], md: &[u8], pk_seed: &[u8], adrs: &Adrs) -> Vec<u8> {
    let indices = base_2b(md, p.a, p.k);
    let per = (1 + p.a) * p.n;
    let mut roots = Vec::with_capacity(p.k * p.n);
    let mut a = *adrs;
    for (i, &idx) in indices.iter().enumerate() {
        let block = &sig[i * per..(i + 1) * per];
        let sk = &block[..p.n];
        let auth = &block[p.n..];
        a.set_tree_height(0);
        a.set_tree_index(((i as u32) << p.a) + idx);
        let mut node = f(p, pk_seed, &a, sk);
        for j in 0..p.a {
            a.set_tree_height(j as u32 + 1);
            let sibling = &auth[j * p.n..(j + 1) * p.n];
            let mut both = Vec::with_capacity(2 * p.n);
            if (idx >> j).is_multiple_of(2) {
                a.set_tree_index(a.tree_index() / 2);
                both.extend_from_slice(&node);
                both.extend_from_slice(sibling);
            } else {
                a.set_tree_index((a.tree_index() - 1) / 2);
                both.extend_from_slice(sibling);
                both.extend_from_slice(&node);
            }
            node = hh(p, pk_seed, &a, &both);
        }
        roots.extend_from_slice(&node);
    }
    let mut pk_adrs = *adrs;
    pk_adrs.set_type_and_clear(FORS_ROOTS);
    pk_adrs.set_key_pair(adrs.key_pair());
    t_l(p, pk_seed, &pk_adrs, &roots)
}

// ---------------------------------------------------------------------------
// The three operations (FIPS 205 §9, §10)
// ---------------------------------------------------------------------------

/// An SLH-DSA key pair, in the encoded forms FIPS 205 defines: the secret key is
/// `SK.seed ‖ SK.prf ‖ PK.seed ‖ PK.root` and the public key is
/// `PK.seed ‖ PK.root`.
#[derive(Clone)]
pub struct KeyPair {
    pub params: Params,
    pub public: Vec<u8>,
    pub secret: Vec<u8>,
}

/// `slh_keygen_internal` (Algorithm 18).
///
/// The three seeds are separate arguments rather than one, because that is how
/// FIPS 205 and the ACVP vectors present them, and deriving them from a single
/// seed here would make the vectors unusable.
pub fn keygen(p: &Params, sk_seed: &[u8], sk_prf: &[u8], pk_seed: &[u8]) -> KeyPair {
    assert_eq!(sk_seed.len(), p.n, "SK.seed is n bytes");
    assert_eq!(sk_prf.len(), p.n, "SK.prf is n bytes");
    assert_eq!(pk_seed.len(), p.n, "PK.seed is n bytes");
    let mut adrs = Adrs::new();
    adrs.set_layer(p.d as u32 - 1);
    let pk_root = xmss_node(p, sk_seed, 0, p.hp(), pk_seed, &adrs);

    let mut secret = Vec::with_capacity(p.secret_key_len());
    secret.extend_from_slice(sk_seed);
    secret.extend_from_slice(sk_prf);
    secret.extend_from_slice(pk_seed);
    secret.extend_from_slice(&pk_root);

    let mut public = Vec::with_capacity(p.public_key_len());
    public.extend_from_slice(pk_seed);
    public.extend_from_slice(&pk_root);

    KeyPair {
        params: *p,
        public,
        secret,
    }
}

/// The `M'` of Algorithm 22: a domain-separated wrapper carrying the context.
fn message_prime(message: &[u8], ctx: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(2 + ctx.len() + message.len());
    m.push(0);
    m.push(ctx.len() as u8);
    m.extend_from_slice(ctx);
    m.extend_from_slice(message);
    m
}

/// Splits the message digest into the FORS input and the two tree indices
/// (Algorithm 19, steps 6–9).
fn split_digest(p: &Params, digest: &[u8]) -> (Vec<u8>, u64, u32) {
    let md_len = (p.k * p.a).div_ceil(8);
    let tree_bits = p.h - p.hp();
    let tree_len = tree_bits.div_ceil(8);
    let leaf_len = p.hp().div_ceil(8);

    let md = digest[..md_len].to_vec();
    let tree_bytes = &digest[md_len..md_len + tree_len];
    let leaf_bytes = &digest[md_len + tree_len..md_len + tree_len + leaf_len];

    let mut idx_tree: u64 = 0;
    for b in tree_bytes {
        idx_tree = (idx_tree << 8) | *b as u64;
    }
    // `tree_bits` can be 63 or 64; shifting by 64 is undefined, so the full-width
    // case masks with `u64::MAX` instead of `(1 << 64) - 1`.
    idx_tree &= if tree_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << tree_bits) - 1
    };

    let mut idx_leaf: u64 = 0;
    for b in leaf_bytes {
        idx_leaf = (idx_leaf << 8) | *b as u64;
    }
    idx_leaf &= (1u64 << p.hp()) - 1;

    (md, idx_tree, idx_leaf as u32)
}

/// `slh_sign` (Algorithm 22) with the deterministic variant of
/// `slh_sign_internal`: `opt_rand` is `PK.seed`.
///
/// Deterministic for the same reason ML-DSA's signing is: writer rule W1
/// (§01.10) says the same inputs produce the same bytes, and a container whose
/// signature changed on every run would not be reproducible.
pub fn sign(kp: &KeyPair, message: &[u8], ctx: &[u8]) -> Result<Vec<u8>, String> {
    if ctx.len() > 255 {
        return Err("SLH-DSA context strings are limited to 255 bytes".into());
    }
    let p = &kp.params;
    if kp.secret.len() != p.secret_key_len() {
        return Err(format!(
            "{} secret keys are {} bytes, got {}",
            p.name,
            p.secret_key_len(),
            kp.secret.len()
        ));
    }
    let mp = message_prime(message, ctx);
    let (sk_seed, rest) = kp.secret.split_at(p.n);
    let (sk_prf, rest) = rest.split_at(p.n);
    let (pk_seed, pk_root) = rest.split_at(p.n);

    let opt_rand = pk_seed;
    let r = prf_msg(p, sk_prf, opt_rand, &mp);
    let digest = h_msg(p, &r, pk_seed, pk_root, &mp);
    let (md, idx_tree, idx_leaf) = split_digest(p, &digest);

    let mut adrs = Adrs::new();
    adrs.set_tree(idx_tree);
    adrs.set_type_and_clear(FORS_TREE);
    adrs.set_key_pair(idx_leaf);

    let sig_fors = fors_sign(p, &md, sk_seed, pk_seed, &adrs);
    let pk_fors = fors_pk_from_sig(p, &sig_fors, &md, pk_seed, &adrs);
    let sig_ht = ht_sign(p, &pk_fors, sk_seed, pk_seed, idx_tree, idx_leaf);

    let mut sig = r;
    sig.extend_from_slice(&sig_fors);
    sig.extend_from_slice(&sig_ht);
    debug_assert_eq!(sig.len(), p.signature_len());
    Ok(sig)
}

/// `slh_verify` (Algorithm 24).
pub fn verify(p: &Params, public: &[u8], message: &[u8], ctx: &[u8], sig: &[u8]) -> bool {
    if public.len() != p.public_key_len() || sig.len() != p.signature_len() || ctx.len() > 255 {
        return false;
    }
    let mp = message_prime(message, ctx);
    let (pk_seed, pk_root) = public.split_at(p.n);

    let fors_len = p.k * (1 + p.a) * p.n;
    let r = &sig[..p.n];
    let sig_fors = &sig[p.n..p.n + fors_len];
    let sig_ht = &sig[p.n + fors_len..];

    let digest = h_msg(p, r, pk_seed, pk_root, &mp);
    let (md, idx_tree, idx_leaf) = split_digest(p, &digest);

    let mut adrs = Adrs::new();
    adrs.set_tree(idx_tree);
    adrs.set_type_and_clear(FORS_TREE);
    adrs.set_key_pair(idx_leaf);

    let pk_fors = fors_pk_from_sig(p, sig_fors, &md, pk_seed, &adrs);
    ht_verify(p, &pk_fors, sig_ht, pk_seed, idx_tree, idx_leaf, pk_root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_derived_lengths_match_fips_205s_table() {
        // Table 2's signature sizes. These are the strongest cheap check there
        // is on `len1`, `len2` and the layer arithmetic: every one of them feeds
        // this number, and getting any wrong changes it.
        for (p, sig) in [
            (SHAKE_128S, 7856),
            (SHAKE_128F, 17088),
            (SHAKE_192S, 16224),
            (SHAKE_192F, 35664),
            (SHAKE_256S, 29792),
            (SHAKE_256F, 49856),
        ] {
            assert_eq!(p.signature_len(), sig, "{} signature", p.name);
            assert_eq!(p.public_key_len(), 2 * p.n, "{} public key", p.name);
            assert_eq!(p.secret_key_len(), 4 * p.n, "{} secret key", p.name);
        }
    }

    #[test]
    fn the_winternitz_lengths_are_what_fips_205_says() {
        assert_eq!(
            (SHAKE_128F.len1(), SHAKE_128F.len2(), SHAKE_128F.len()),
            (32, 3, 35)
        );
        assert_eq!(
            (SHAKE_192F.len1(), SHAKE_192F.len2(), SHAKE_192F.len()),
            (48, 3, 51)
        );
        assert_eq!(
            (SHAKE_256F.len1(), SHAKE_256F.len2(), SHAKE_256F.len()),
            (64, 3, 67)
        );
    }

    #[test]
    fn the_digest_split_consumes_exactly_m_bytes() {
        for p in ALL {
            let md_len = (p.k * p.a).div_ceil(8);
            let tree_len = (p.h - p.hp()).div_ceil(8);
            let leaf_len = p.hp().div_ceil(8);
            assert_eq!(md_len + tree_len + leaf_len, p.m, "{}", p.name);
        }
    }

    #[test]
    fn base_2b_reads_big_endian_fields() {
        // Two 4-bit digits per byte, high nibble first.
        assert_eq!(base_2b(&[0xAB, 0xCD], 4, 4), vec![0xA, 0xB, 0xC, 0xD]);
        // A width that straddles byte boundaries.
        assert_eq!(
            base_2b(&[0b1010_1010, 0b1100_1100], 6, 2),
            vec![0b101010, 0b101100]
        );
        assert_eq!(base_2b(&[0xFF, 0xFF, 0xFF], 12, 2), vec![0xFFF, 0xFFF]);
    }

    #[test]
    fn setting_a_type_clears_the_fields_after_it() {
        // The clearing is not tidiness: a stale key-pair address changes every
        // hash that follows, and nothing else in the scheme would notice.
        let mut a = Adrs::new();
        a.set_key_pair(0x1122_3344);
        a.set_tree_index(0x5566_7788);
        a.set_type_and_clear(TREE);
        assert_eq!(a.key_pair(), 0);
        assert_eq!(a.tree_index(), 0);
        assert_eq!(&a.as_bytes()[16..20], &TREE.to_be_bytes());
    }

    /// The fast parameter set, end to end. `f` rather than `s` because signing an
    /// `s` set walks seven trees of 512 WOTS+ key pairs and takes seconds in a
    /// debug build; the `s` sets are covered by the key-generation vectors and by
    /// CI, which runs in release.
    #[test]
    fn a_signature_verifies_and_a_tampered_one_does_not() {
        let p = SHAKE_128F;
        let kp = keygen(&p, &[1u8; 16], &[2u8; 16], &[3u8; 16]);
        assert_eq!(kp.public.len(), p.public_key_len());
        assert_eq!(kp.secret.len(), p.secret_key_len());

        let sig = sign(&kp, b"a model exists once", b"omni").unwrap();
        assert_eq!(sig.len(), p.signature_len());
        assert!(verify(
            &p,
            &kp.public,
            b"a model exists once",
            b"omni",
            &sig
        ));

        // The context is part of what is signed.
        assert!(!verify(
            &p,
            &kp.public,
            b"a model exists once",
            b"other",
            &sig
        ));
        assert!(!verify(
            &p,
            &kp.public,
            b"a model exists twice",
            b"omni",
            &sig
        ));

        // A flipped bit anywhere: in the randomiser, in FORS, in the hypertree.
        for at in [0usize, 20, 5000, 17087] {
            let mut bad = sig.clone();
            bad[at] ^= 1;
            assert!(
                !verify(&p, &kp.public, b"a model exists once", b"omni", &bad),
                "a bit flip at {at} verified"
            );
        }
        // And a truncated signature is a length check, not a panic.
        assert!(!verify(
            &p,
            &kp.public,
            b"a model exists once",
            b"omni",
            &sig[..sig.len() - 1]
        ));
    }

    #[test]
    fn signing_is_deterministic() {
        let p = SHAKE_128F;
        let kp = keygen(&p, &[7u8; 16], &[8u8; 16], &[9u8; 16]);
        assert_eq!(
            sign(&kp, b"same", b"").unwrap(),
            sign(&kp, b"same", b"").unwrap()
        );
    }

    #[test]
    fn a_wrong_key_does_not_verify() {
        let p = SHAKE_128F;
        let a = keygen(&p, &[1u8; 16], &[1u8; 16], &[1u8; 16]);
        let b = keygen(&p, &[2u8; 16], &[2u8; 16], &[2u8; 16]);
        let sig = sign(&a, b"m", b"").unwrap();
        assert!(!verify(&p, &b.public, b"m", b"", &sig));
    }
}
