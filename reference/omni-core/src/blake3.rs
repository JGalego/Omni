//! BLAKE3 — the specification's default digest algorithm (§03.5.1).
//!
//! Implemented from the BLAKE3 paper's reference pseudocode, single-threaded
//! and without SIMD: this crate optimises for auditability, not throughput. A
//! production implementation should use the upstream crate, which is roughly an
//! order of magnitude faster on the same hardware.
//!
//! Three keying modes are provided because §03.5.3 needs all three: plain
//! hashing for object digests, keyed hashing for MACs, and `derive_key` for the
//! domain-separated context strings (`omni/1.0 object`, `omni/1.0 plan-key`, …)
//! that stop a digest computed for one purpose being replayed as another.
//!
//! The tree is exposed, not just the root hash. [`chunk_chaining_value`] and
//! [`parent_chaining_value`] are what make Bao verified streaming (§13.3)
//! possible: an arbitrary byte range of a 140 GB object can be verified against
//! the root digest with a logarithmic proof. No flat hash can do that, and it
//! is the whole reason the specification prefers BLAKE3 over SHA-256.

/// Digest length in bytes. BLAKE3 is an XOF; 32 is the OMNI default.
pub const OUT_LEN: usize = 32;
/// Key length for keyed hashing.
pub const KEY_LEN: usize = 32;
/// Compression function block size.
pub const BLOCK_LEN: usize = 64;
/// Chunk size. The tree has one leaf per chunk.
pub const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const KEYED_HASH: u32 = 16;
const DERIVE_KEY_CONTEXT: u32 = 32;
const DERIVE_KEY_MATERIAL: u32 = 64;

/// The SHA-2 initialisation vector, reused by BLAKE3.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The maximum tree depth. A stack this deep covers 2^54 chunks — 2^64 bytes,
/// the largest input the 64-bit chunk counter can address.
const MAX_DEPTH: usize = 54;

#[allow(clippy::too_many_arguments)]
fn g(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

fn round(s: &mut [u32; 16], m: &[u32; 16]) {
    // Columns.
    g(s, 0, 4, 8, 12, m[0], m[1]);
    g(s, 1, 5, 9, 13, m[2], m[3]);
    g(s, 2, 6, 10, 14, m[4], m[5]);
    g(s, 3, 7, 11, 15, m[6], m[7]);
    // Diagonals.
    g(s, 0, 5, 10, 15, m[8], m[9]);
    g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7, 8, 13, m[12], m[13]);
    g(s, 3, 4, 9, 14, m[14], m[15]);
}

fn permute(m: &mut [u32; 16]) {
    let mut p = [0u32; 16];
    for (dst, &src) in p.iter_mut().zip(MSG_PERMUTATION.iter()) {
        *dst = m[src];
    }
    *m = p;
}

/// The compression function. Returns all 16 words: the first eight are the
/// chaining value, the full sixteen are the XOF output block.
fn compress(
    cv: &[u32; 8],
    block: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut s = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    let mut m = *block;
    for r in 0..7 {
        round(&mut s, &m);
        if r < 6 {
            permute(&mut m);
        }
    }
    for i in 0..8 {
        s[i] ^= s[i + 8];
        s[i + 8] ^= cv[i];
    }
    s
}

fn words_from_block(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for (word, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    w
}

fn words_from_key(key: &[u8; KEY_LEN]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for (word, bytes) in w.iter_mut().zip(key.chunks_exact(4)) {
        *word = u32::from_le_bytes(bytes.try_into().unwrap());
    }
    w
}

fn bytes_from_words(w: &[u32; 8]) -> [u8; OUT_LEN] {
    let mut out = [0u8; OUT_LEN];
    for (chunk, word) in out.chunks_exact_mut(4).zip(w.iter()) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// A node of the tree that has not yet been told whether it is the root.
///
/// This deferral is the reason BLAKE3 is a tree hash rather than a Merkle
/// construction bolted onto a flat hash: the ROOT flag is only set once, at
/// finalisation, so a subtree's chaining value can never be confused with the
/// hash of the whole input.
#[derive(Clone)]
struct Output {
    input_cv: [u32; 8],
    block: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        let s = compress(
            &self.input_cv,
            &self.block,
            self.counter,
            self.block_len,
            self.flags,
        );
        s[..8].try_into().unwrap()
    }

    fn root_bytes(&self, out: &mut [u8]) {
        for (i, block) in out.chunks_mut(2 * OUT_LEN).enumerate() {
            let words = compress(
                &self.input_cv,
                &self.block,
                i as u64,
                self.block_len,
                self.flags | ROOT,
            );
            for (dst, word) in block.chunks_mut(4).zip(words.iter()) {
                let le = word.to_le_bytes();
                dst.copy_from_slice(&le[..dst.len()]);
            }
        }
    }

    fn root_hash(&self) -> [u8; OUT_LEN] {
        let mut out = [0u8; OUT_LEN];
        self.root_bytes(&mut out);
        out
    }
}

#[derive(Clone)]
struct ChunkState {
    cv: [u32; 8],
    counter: u64,
    buf: [u8; BLOCK_LEN],
    buf_len: usize,
    blocks_compressed: u8,
    flags: u32,
}

impl ChunkState {
    fn new(key: [u32; 8], counter: u64, flags: u32) -> Self {
        ChunkState {
            cv: key,
            counter,
            buf: [0; BLOCK_LEN],
            buf_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.buf_len
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // The final block of a chunk is held back rather than compressed
            // eagerly, because only at finalisation is its length known and the
            // CHUNK_END flag applicable.
            if self.buf_len == BLOCK_LEN {
                let block = words_from_block(&self.buf);
                let s = compress(
                    &self.cv,
                    &block,
                    self.counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                );
                self.cv = s[..8].try_into().unwrap();
                self.blocks_compressed += 1;
                self.buf = [0; BLOCK_LEN];
                self.buf_len = 0;
            }
            let take = core::cmp::min(BLOCK_LEN - self.buf_len, input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        Output {
            input_cv: self.cv,
            block: words_from_block(&self.buf),
            counter: self.counter,
            block_len: self.buf_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(left: [u32; 8], right: [u32; 8], key: [u32; 8], flags: u32) -> Output {
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&left);
    block[8..].copy_from_slice(&right);
    Output {
        input_cv: key,
        block,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: flags | PARENT,
    }
}

/// Incremental BLAKE3 hasher.
#[derive(Clone)]
pub struct Hasher {
    chunk: ChunkState,
    key: [u32; 8],
    stack: [[u32; 8]; MAX_DEPTH],
    stack_len: usize,
    flags: u32,
}

impl Hasher {
    fn with_key(key: [u32; 8], flags: u32) -> Self {
        Hasher {
            chunk: ChunkState::new(key, 0, flags),
            key,
            stack: [[0; 8]; MAX_DEPTH],
            stack_len: 0,
            flags,
        }
    }

    /// Plain hashing.
    pub fn new() -> Self {
        Self::with_key(IV, 0)
    }

    /// Keyed hashing (a MAC).
    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        Self::with_key(words_from_key(key), KEYED_HASH)
    }

    /// Key derivation from a hardcoded, application-unique context string.
    ///
    /// The context is itself hashed in a separate mode, so two applications
    /// deriving from the same key material with different contexts can never
    /// collide. §03.5.3 lists the context strings OMNI uses.
    pub fn new_derive_key(context: &str) -> Self {
        let mut ctx = Self::with_key(IV, DERIVE_KEY_CONTEXT);
        ctx.update(context.as_bytes());
        let key = words_from_key(&ctx.finalize());
        Self::with_key(key, DERIVE_KEY_MATERIAL)
    }

    /// Merges completed subtrees. A new chaining value can be merged with its
    /// left sibling exactly when the number of chunks so far is even at that
    /// level, which is what the trailing-zero loop tests.
    fn push_chunk_cv(&mut self, mut cv: [u32; 8], mut total_chunks: u64) {
        while total_chunks & 1 == 0 {
            self.stack_len -= 1;
            let left = self.stack[self.stack_len];
            cv = parent_output(left, cv, self.key, self.flags).chaining_value();
            total_chunks >>= 1;
        }
        self.stack[self.stack_len] = cv;
        self.stack_len += 1;
    }

    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_LEN {
                let cv = self.chunk.output().chaining_value();
                let total = self.chunk.counter + 1;
                self.push_chunk_cv(cv, total);
                self.chunk = ChunkState::new(self.key, total, self.flags);
            }
            let take = core::cmp::min(CHUNK_LEN - self.chunk.len(), input.len());
            self.chunk.update(&input[..take]);
            input = &input[take..];
        }
        self
    }

    /// Writes an arbitrary number of output bytes (the XOF).
    pub fn finalize_xof(&self, out: &mut [u8]) {
        let mut output = self.chunk.output();
        // Fold the right edge of the tree into the left, which is the only
        // place where subtrees of unequal size are allowed to merge.
        for i in (0..self.stack_len).rev() {
            output = parent_output(self.stack[i], output.chaining_value(), self.key, self.flags);
        }
        output.root_bytes(out);
    }

    /// The 32-byte digest.
    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let mut out = [0u8; OUT_LEN];
        self.finalize_xof(&mut out);
        out
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot BLAKE3-256 (§03.5.1's default digest).
pub fn blake3(data: &[u8]) -> [u8; OUT_LEN] {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
}

/// One-shot keyed BLAKE3-256.
pub fn keyed_hash(key: &[u8; KEY_LEN], data: &[u8]) -> [u8; OUT_LEN] {
    let mut h = Hasher::new_keyed(key);
    h.update(data);
    h.finalize()
}

/// One-shot key derivation (§03.5.3 domain separation).
pub fn derive_key(context: &str, material: &[u8]) -> [u8; OUT_LEN] {
    let mut h = Hasher::new_derive_key(context);
    h.update(material);
    h.finalize()
}

/// Chaining value of a leaf that is **not** the root of its tree.
///
/// `chunk` must be at most [`CHUNK_LEN`] bytes and `counter` is its index.
/// Together with [`parent_chaining_value`] this is the whole interface Bao
/// needs to build an outboard tree (§13.3).
pub fn chunk_chaining_value(chunk: &[u8], counter: u64) -> [u8; OUT_LEN] {
    debug_assert!(chunk.len() <= CHUNK_LEN);
    let mut cs = ChunkState::new(IV, counter, 0);
    cs.update(chunk);
    bytes_from_words(&cs.output().chaining_value())
}

/// Root hash of a tree consisting of a single leaf.
pub fn chunk_root(chunk: &[u8], counter: u64) -> [u8; OUT_LEN] {
    debug_assert!(chunk.len() <= CHUNK_LEN);
    let mut cs = ChunkState::new(IV, counter, 0);
    cs.update(chunk);
    cs.output().root_hash()
}

/// Chaining value of an interior node that is **not** the root.
pub fn parent_chaining_value(left: &[u8; OUT_LEN], right: &[u8; OUT_LEN]) -> [u8; OUT_LEN] {
    let out = parent_output(words_from_key(left), words_from_key(right), IV, 0);
    bytes_from_words(&out.chaining_value())
}

/// Root hash of a tree whose top node has these two children.
pub fn parent_root(left: &[u8; OUT_LEN], right: &[u8; OUT_LEN]) -> [u8; OUT_LEN] {
    parent_output(words_from_key(left), words_from_key(right), IV, 0).root_hash()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::hex;

    /// The official BLAKE3 test vectors, as published in `test_vectors.json` in
    /// the BLAKE3 repository. The input of length *n* is the repeating byte
    /// pattern `i mod 251`; each expected value is 131 bytes of XOF output, so
    /// these check the extended output as well as the 32-byte digest.
    const KEY: &[u8; 32] = b"whats the Elvish word for friend";
    const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

    /// `(input_len, hash, keyed_hash, derive_key)`
    #[rustfmt::skip]
    const VECTORS: &[(usize, &str, &str, &str)] = &[
        (0,
         "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262e00f03e7b69af26b7faaf09fcd333050338ddfe085b8cc869ca98b206c08243a26f5487789e8f660afe6c99ef9e0c52b92e7393024a80459cf91f476f9ffdbda7001c22e159b402631f277ca96f2defdf1078282314e763699a31c5363165421cce14d",
         "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26b18171a2f22a4b94822c701f107153dba24918c4bae4d2945c20ece13387627d3b73cbf97b797d5e59948c7ef788f54372df45e45e4293c7dc18c1d41144a9758be58960856be1eabbe22c2653190de560ca3b2ac4aa692a9210694254c371e851bc8f",
         "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d905630c8be290dfcf3e6842f13bddd573c098c3f17361f1f206b8cad9d088aa4a3f746752c6b0ce6a83b0da81d59649257cdf8eb3e9f7d4998e41021fac119deefb896224ac99f860011f73609e6e0e4540f93b273e56547dfd3aa1a035ba6689d89a0"),
        (1,
         "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213c3a6cb8bf623e20cdb535f8d1a5ffb86342d9c0b64aca3bce1d31f60adfa137b358ad4d79f97b47c3d5e79f179df87a3b9776ef8325f8329886ba42f07fb138bb502f4081cbcec3195c5871e6c23e2cc97d3c69a613eba131e5f1351f3f1da786545e5",
         "6d7878dfff2f485635d39013278ae14f1454b8c0a3a2d34bc1ab38228a80c95b6568c0490609413006fbd428eb3fd14e7756d90f73a4725fad147f7bf70fd61c4e0cf7074885e92b0e3f125978b4154986d4fb202a3f331a3fb6cf349a3a70e49990f98fe4289761c8602c4e6ab1138d31d3b62218078b2f3ba9a88e1d08d0dd4cea11",
         "b3e2e340a117a499c6cf2398a19ee0d29cca2bb7404c73063382693bf66cb06c5827b91bf889b6b97c5477f535361caefca0b5d8c4746441c57617111933158950670f9aa8a05d791daae10ac683cbef8faf897c84e6114a59d2173c3f417023a35d6983f2c7dfa57e7fc559ad751dbfb9ffab39c2ef8c4aafebc9ae973a64f0c76551"),
        (2,
         "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63d8386b22e2ddc05836b7c1bb693d92af006deb5ffbc4c70fb44d0195d0c6f252faac61659ef86523aa16517f87cb5f1340e723756ab65efb2f91964e14391de2a432263a6faf1d146937b35a33621c12d00be8223a7f1919cec0acd12097ff3ab00ab1",
         "5392ddae0e0a69d5f40160462cbd9bd889375082ff224ac9c758802b7a6fd20a9ffbf7efd13e989a6c246f96d3a96b9d279f2c4e63fb0bdff633957acf50ee1a5f658be144bab0f6f16500dee4aa5967fc2c586d85a04caddec90fffb7633f46a60786024353b9e5cebe277fcd9514217fee2267dcda8f7b31697b7c54fab6a939bf8f",
         "1f166565a7df0098ee65922d7fea425fb18b9943f19d6161e2d17939356168e6daa59cae19892b2d54f6fc9f475d26031fd1c22ae0a3e8ef7bdb23f452a15e0027629d2e867b1bb1e6ab21c71297377750826c404dfccc2406bd57a83775f89e0b075e59a7732326715ef912078e213944f490ad68037557518b79c0086de6d6f6cdd2"),
        (3,
         "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f5b49b82f805a538c68915c1ae8035c900fd1d4b13902920fd05e1450822f36de9454b7e9996de4900c8e723512883f93f4345f8a58bfe64ee38d3ad71ab027765d25cdd0e448328a8e7a683b9a6af8b0af94fa09010d9186890b096a08471e4230a134",
         "39e67b76b5a007d4921969779fe666da67b5213b096084ab674742f0d5ec62b9b9142d0fab08e1b161efdbb28d18afc64d8f72160c958e53a950cdecf91c1a1bbab1a9c0f01def762a77e2e8545d4dec241e98a89b6db2e9a5b070fc110caae2622690bd7b76c02ab60750a3ea75426a6bb8803c370ffe465f07fb57def95df772c39f",
         "440aba35cb006b61fc17c0529255de438efc06a8c9ebf3f2ddac3b5a86705797f27e2e914574f4d87ec04c379e12789eccbfbc15892626042707802dbe4e97c3ff59dca80c1e54246b6d055154f7348a39b7d098b2b4824ebe90e104e763b2a447512132cede16243484a55a4e40a85790038bb0dcf762e8c053cabae41bbe22a5bff7"),
        (63,
         "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b1197012b1e7d9af4d7cb7bdd1f3bb49a90a9b5dec3ea2bbc6eaebce77f4e470cbf4687093b5352f04e4a4570fba233164e6acc36900e35d185886a827f7ea9bdc1e5c3ce88b095a200e62c10c043b3e9bc6cb9b6ac4dfa51794b02ace9f98779040755",
         "bb1eb5d4afa793c1ebdd9fb08def6c36d10096986ae0cfe148cd101170ce37aea05a63d74a840aecd514f654f080e51ac50fd617d22610d91780fe6b07a26b0847abb38291058c97474ef6ddd190d30fc318185c09ca1589d2024f0a6f16d45f11678377483fa5c005b2a107cb9943e5da634e7046855eaa888663de55d6471371d55d",
         "b6451e30b953c206e34644c6803724e9d2725e0893039cfc49584f991f451af3b89e8ff572d3da4f4022199b9563b9d70ebb616efff0763e9abec71b550f1371e233319c4c4e74da936ba8e5bbb29a598e007a0bbfa929c99738ca2cc098d59134d11ff300c39f82e2fce9f7f0fa266459503f64ab9913befc65fddc474f6dc1c67669"),
        (64,
         "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98fc9cc56cb831ffe33ea8e7e1d1df09b26efd2767670066aa82d023b1dfe8ab1b2b7fbb5b97592d46ffe3e05a6a9b592e2949c74160e4674301bc3f97e04903f8c6cf95b863174c33228924cdef7ae47559b10b294acd660666c4538833582b43f82d74",
         "ba8ced36f327700d213f120b1a207a3b8c04330528586f414d09f2f7d9ccb7e68244c26010afc3f762615bbac552a1ca909e67c83e2fd5478cf46b9e811efccc93f77a21b17a152ebaca1695733fdb086e23cd0eb48c41c034d52523fc21236e5d8c9255306e48d52ba40b4dac24256460d56573d1312319afcf3ed39d72d0bfc69acb",
         "a5c4a7053fa86b64746d4bb688d06ad1f02a18fce9afd3e818fefaa7126bf73e9b9493a9befebe0bf0c9509fb3105cfa0e262cde141aa8e3f2c2f77890bb64a4cca96922a21ead111f6338ad5244f2c15c44cb595443ac2ac294231e31be4a4307d0a91e874d36fc9852aeb1265c09b6e0cda7c37ef686fbbcab97e8ff66718be048bb"),
        (65,
         "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee0e16e0a4749d6811dd1d6d1265c29729b1b75a9ac346cf93f0e1d7296dfcfd4313b3a227faaaaf7757cc95b4e87a49be3b8a270a12020233509b1c3632b3485eef309d0abc4a4a696c9decc6e90454b53b000f456a3f10079072baaf7a981653221f2c",
         "c0a4edefa2d2accb9277c371ac12fcdbb52988a86edc54f0716e1591b4326e72d5e795f46a596b02d3d4bfb43abad1e5d19211152722ec1f20fef2cd413e3c22f2fc5da3d73041275be6ede3517b3b9f0fc67ade5956a672b8b75d96cb43294b9041497de92637ed3f2439225e683910cb3ae923374449ca788fb0f9bea92731bc26ad",
         "51fd05c3c1cfbc8ed67d139ad76f5cf8236cd2acd26627a30c104dfd9d3ff8a82b02e8bd36d8498a75ad8c8e9b15eb386970283d6dd42c8ae7911cc592887fdbe26a0a5f0bf821cd92986c60b2502c9be3f98a9c133a7e8045ea867e0828c7252e739321f7c2d65daee4468eb4429efae469a42763f1f94977435d10dccae3e3dce88d"),
        (127,
         "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640de3137d477156d1fde56b0cf36f8ef18b44b2d79897bece12227539ac9ae0a5119da47644d934d26e74dc316145dcb8bb69ac3f2e05c242dd6ee06484fcb0e956dc44355b452c5e2bbb5e2b66e99f5dd443d0cbcaaafd4beebaed24ae2f8bb672bcef78",
         "c64200ae7dfaf35577ac5a9521c47863fb71514a3bcad18819218b818de85818ee7a317aaccc1458f78d6f65f3427ec97d9c0adb0d6dacd4471374b621b7b5f35cd54663c64dbe0b9e2d95632f84c611313ea5bd90b71ce97b3cf645776f3adc11e27d135cbadb9875c2bf8d3ae6b02f8a0206aba0c35bfe42574011931c9a255ce6dc",
         "c91c090ceee3a3ac81902da31838012625bbcd73fcb92e7d7e56f78deba4f0c3feeb3974306966ccb3e3c69c337ef8a45660ad02526306fd685c88542ad00f759af6dd1adc2e50c2b8aac9f0c5221ff481565cf6455b772515a69463223202e5c371743e35210bbbbabd89651684107fd9fe493c937be16e39cfa7084a36207c99bea3"),
        (128,
         "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45efa69faba091427f9c5c4caa873aa07828651f19c55bad85c47d1368b11c6fd99e47ecba5820a0325984d74fe3e4058494ca12e3f1d3293d0010a9722f7dee64f71246f75e9361f44cc8e214a100650db1313ff76a9f93ec6e84edb7add1cb4a95019b0c",
         "b04fe15577457267ff3b6f3c947d93be581e7e3a4b018679125eaf86f6a628ecd86bbe0001f10bda47e6077b735016fca8119da11348d93ca302bbd125bde0db2b50edbe728a620bb9d3e6f706286aedea973425c0b9eedf8a38873544cf91badf49ad92a635a93f71ddfcee1eae536c25d1b270956be16588ef1cfef2f1d15f650bd5",
         "81720f34452f58a0120a58b6b4608384b5c51d11f39ce97161a0c0e442ca022550e7cd651e312f0b4c6afb3c348ae5dd17d2b29fab3b894d9a0034c7b04fd9190cbd90043ff65d1657bbc05bfdecf2897dd894c7a1b54656d59a50b51190a9da44db426266ad6ce7c173a8c0bbe091b75e734b4dadb59b2861cd2518b4e7591e4b83c9"),
        (129,
         "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12f96ffa7b36dd78ba321be7e842d364a62a42e3746681c8bace18a4a8a79649285c7127bf8febf125be9de39586d251f0d41da20980b70d35e3dac0eee59e468a894fa7e6a07129aaad09855f6ad4801512a116ba2b7841e6cfc99ad77594a8f2d181a7",
         "d4a64dae6cdccbac1e5287f54f17c5f985105457c1a2ec1878ebd4b57e20d38f1c9db018541eec241b748f87725665b7b1ace3e0065b29c3bcb232c90e37897fa5aaee7e1e8a2ecfcd9b51463e42238cfdd7fee1aecb3267fa7f2128079176132a412cd8aaf0791276f6b98ff67359bd8652ef3a203976d5ff1cd41885573487bcd683",
         "938d2d4435be30eafdbb2b7031f7857c98b04881227391dc40db3c7b21f41fc18d72d0f9c1de5760e1941aebf3100b51d64644cb459eb5d20258e233892805eb98b07570ef2a1787cd48e117c8d6a63a68fd8fc8e59e79dbe63129e88352865721c8d5f0cf183f85e0609860472b0d6087cefdd186d984b21542c1c780684ed6832d8d"),
        (1023,
         "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11a182d27a591b05592b15607500e1e8dd56bc6c7fc063715b7a1d737df5bad3339c56778957d870eb9717b57ea3d9fb68d1b55127bba6a906a4a24bbd5acb2d123a37b28f9e9a81bbaae360d58f85e5fc9d75f7c370a0cc09b6522d9c8d822f2f28f485",
         "c951ecdf03288d0fcc96ee3413563d8a6d3589547f2c2fb36d9786470f1b9d6e890316d2e6d8b8c25b0a5b2180f94fb1a158ef508c3cde45e2966bd796a696d3e13efd86259d756387d9becf5c8bf1ce2192b87025152907b6d8cc33d17826d8b7b9bc97e38c3c85108ef09f013e01c229c20a83d9e8efac5b37470da28575fd755a10",
         "74a16c1c3d44368a86e1ca6df64be6a2f64cce8f09220787450722d85725dea59c413264404661e9e4d955409dfe4ad3aa487871bcd454ed12abfe2c2b1eb7757588cf6cb18d2eccad49e018c0d0fec323bec82bf1644c6325717d13ea712e6840d3e6e730d35553f59eff5377a9c350bcc1556694b924b858f329c44ee64b884ef00d"),
        (1024,
         "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af71cf8107265ecdaf8505b95d8fcec83a98a6a96ea5109d2c179c47a387ffbb404756f6eeae7883b446b70ebb144527c2075ab8ab204c0086bb22b7c93d465efc57f8d917f0b385c6df265e77003b85102967486ed57db5c5ca170ba441427ed9afa684e",
         "75c46f6f3d9eb4f55ecaaee480db732e6c2105546f1e675003687c31719c7ba4a78bc838c72852d4f49c864acb7adafe2478e824afe51c8919d06168414c265f298a8094b1ad813a9b8614acabac321f24ce61c5a5346eb519520d38ecc43e89b5000236df0597243e4d2493fd626730e2ba17ac4d8824d09d1a4a8f57b8227778e2de",
         "7356cd7720d5b66b6d0697eb3177d9f8d73a4a5c5e968896eb6a6896843027066c23b601d3ddfb391e90d5c8eccdef4ae2a264bce9e612ba15e2bc9d654af1481b2e75dbabe615974f1070bba84d56853265a34330b4766f8e75edd1f4a1650476c10802f22b64bd3919d246ba20a17558bc51c199efdec67e80a227251808d8ce5bad"),
        (1025,
         "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444f4c4a22b4b399155358a994e52bf255de60035742ec71bd08ac275a1b51cc6bfe332b0ef84b409108cda080e6269ed4b3e2c3f7d722aa4cdc98d16deb554e5627be8f955c98e1d5f9565a9194cad0c4285f93700062d9595adb992ae68ff12800ab67a",
         "357dc55de0c7e382c900fd6e320acc04146be01db6a8ce7210b7189bd664ea69362396b77fdc0d2634a552970843722066c3c15902ae5097e00ff53f1e116f1cd5352720113a837ab2452cafbde4d54085d9cf5d21ca613071551b25d52e69d6c81123872b6f19cd3bc1333edf0c52b94de23ba772cf82636cff4542540a7738d5b930",
         "effaa245f065fbf82ac186839a249707c3bddf6d3fdda22d1b95a3c970379bcb5d31013a167509e9066273ab6e2123bc835b408b067d88f96addb550d96b6852dad38e320b9d940f86db74d398c770f462118b35d2724efa13da97194491d96dd37c3c09cbef665953f2ee85ec83d88b88d11547a6f911c8217cca46defa2751e7f3ad"),
        (2048,
         "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a9a60bf80001410ec9eea6698cd537939fad4749edd484cb541aced55cd9bf54764d063f23f6f1e32e12958ba5cfeb1bf618ad094266d4fc3c968c2088f677454c288c67ba0dba337b9d91c7e1ba586dc9a5bc2d5e90c14f53a8863ac75655461cea8f9",
         "879cf1fa2ea0e79126cb1063617a05b6ad9d0b696d0d757cf053439f60a99dd10173b961cd574288194b23ece278c330fbb8585485e74967f31352a8183aa782b2b22f26cdcadb61eed1a5bc144b8198fbb0c13abbf8e3192c145d0a5c21633b0ef86054f42809df823389ee40811a5910dcbd1018af31c3b43aa55201ed4edaac74fe",
         "7b2945cb4fef70885cc5d78a87bf6f6207dd901ff239201351ffac04e1088a23e2c11a1ebffcea4d80447867b61badb1383d842d4e79645d48dd82ccba290769caa7af8eaa1bd78a2a5e6e94fbdab78d9c7b74e894879f6a515257ccf6f95056f4e25390f24f6b35ffbb74b766202569b1d797f2d4bd9d17524c720107f985f4ddc583"),
        (2049,
         "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b687952256303096de31d71d74103403822a2e0bc1eb193e7aecc9643a76b7bbc0c9f9c52e8783aae98764ca468962b5c2ec92f0c74eb5448d519713e09413719431c802f948dd5d90425a4ecdadece9eb178d80f26efccae630734dff63340285adec2aed3b51073ad3",
         "9f29700902f7c86e514ddc4df1e3049f258b2472b6dd5267f61bf13983b78dd5f9a88abfefdfa1e00b418971f2b39c64ca621e8eb37fceac57fd0c8fc8e117d43b81447be22d5d8186f8f5919ba6bcc6846bd7d50726c06d245672c2ad4f61702c646499ee1173daa061ffe15bf45a631e2946d616a4c345822f1151284712f76b2b0e",
         "2ea477c5515cc3dd606512ee72bb3e0e758cfae7232826f35fb98ca1bcbdf27316d8e9e79081a80b046b60f6a263616f33ca464bd78d79fa18200d06c7fc9bffd808cc4755277a7d5e09da0f29ed150f6537ea9bed946227ff184cc66a72a5f8c1e4bd8b04e81cf40fe6dc4427ad5678311a61f4ffc39d195589bdbc670f63ae70f4b6"),
        (3072,
         "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd29a3f6b0b978d6608335c09dc94ccf682f9951cdfc501bfe47b9c9189a6fc7b404d120258506341a6d802857322fbd20d3e5dae05b95c88793fa83db1cb08e7d8008d1599b6209d78336e24839724c191b2a52a80448306e0daa84a3fdb566661a37e11",
         "044a0e7b172a312dc02a4c9a818c036ffa2776368d7f528268d2e6b5df19177022f302d0529e4174cc507c463671217975e81dab02b8fdeb0d7ccc7568dd22574c783a76be215441b32e91b9a904be8ea81f7a0afd14bad8ee7c8efc305ace5d3dd61b996febe8da4f56ca0919359a7533216e2999fc87ff7d8f176fbecb3d6f34278b",
         "050df97f8c2ead654d9bb3ab8c9178edcd902a32f8495949feadcc1e0480c46b3604131bbd6e3ba573b6dd682fa0a63e5b165d39fc43a625d00207607a2bfeb65ff1d29292152e26b298868e3b87be95d6458f6f2ce6118437b632415abe6ad522874bcd79e4030a5e7bad2efa90a7a7c67e93f0a18fb28369d0a9329ab5c24134ccb0"),
        (4096,
         "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e9690289e9409ddb1b99768eafe1623da896faf7e1114bebeadc1be30829b6f8af707d85c298f4f0ff4d9438aef948335612ae921e76d411c3a9111df62d27eaf871959ae0062b5492a0feb98ef3ed4af277f5395172dbe5c311918ea0074ce0036454f620",
         "befc660aea2f1718884cd8deb9902811d332f4fc4a38cf7c7300d597a081bfc0bbb64a36edb564e01e4b4aaf3b060092a6b838bea44afebd2deb8298fa562b7b597c757b9df4c911c3ca462e2ac89e9a787357aaf74c3b56d5c07bc93ce899568a3eb17d9250c20f6c5f6c1e792ec9a2dcb715398d5a6ec6d5c54f586a00403a1af1de",
         "1e0d7f3db8c414c97c6307cbda6cd27ac3b030949da8e23be1a1a924ad2f25b9d78038f7b198596c6cc4a9ccf93223c08722d684f240ff6569075ed81591fd93f9fff1110b3a75bc67e426012e5588959cc5a4c192173a03c00731cf84544f65a2fb9378989f72e9694a6a394a8a30997c2e67f95a504e631cd2c5f55246024761b245"),
        (8192,
         "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a635fe51a27db045a567c1ad51be5aa34c01c6651c4d9b5b5ac5d0fd58cf18dd61a47778566b797a8c67df7b1d60b97b19288d2d877bb2df417ace009dcb0241ca1257d62712b6a4043b4ff33f690d849da91ea3bf711ed583cb7b7a7da2839ba71309bbf",
         "dc9637c8845a770b4cbf76b8daec0eebf7dc2eac11498517f08d44c8fc00d58a4834464159dcbc12a0ba0c6d6eb41bac0ed6585cabfe0aca36a375e6c5480c22afdc40785c170f5a6b8a1107dbee282318d00d915ac9ed1143ad40765ec120042ee121cd2baa36250c618adaf9e27260fda2f94dea8fb6f08c04f8f10c78292aa46102",
         "ad01d7ae4ad059b0d33baa3c01319dcf8088094d0359e5fd45d6aeaa8b2d0c3d4c9e58958553513b67f84f8eac653aeeb02ae1d5672dcecf91cd9985a0e67f4501910ecba25555395427ccc7241d70dc21c190e2aadee875e5aae6bf1912837e53411dabf7a56cbf8e4fb780432b0d7fe6cec45024a0788cf5874616407757e9e6bef7"),
        (16384,
         "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde49d764c270176e53e97bdffa58d549073f2c660be0e81293767ed4e4929f9ad34bbb39a529334c57c4a381ffd2a6d4bfdbf1482651b172aa883cc13408fa67758a3e47503f93f87720a3177325f7823251b85275f64636a8f1d599c2e49722f42e93893",
         "9e9fc4eb7cf081ea7c47d1807790ed211bfec56aa25bb7037784c13c4b707b0df9e601b101e4cf63a404dfe50f2e1865bb12edc8fca166579ce0c70dba5a5c0fc960ad6f3772183416a00bd29d4c6e651ea7620bb100c9449858bf14e1ddc9ecd35725581ca5b9160de04060045993d972571c3e8f71e9d0496bfa744656861b169d65",
         "160e18b5878cd0df1c3af85eb25a0db5344d43a6fbd7a8ef4ed98d0714c3f7e160dc0b1f09caa35f2f417b9ef309dfe5ebd67f4c9507995a531374d099cf8ae317542e885ec6f589378864d3ea98716b3bbb65ef4ab5e0ab5bb298a501f19a41ec19af84a5e6b428ecd813b1a47ed91c9657c3fba11c406bc316768b58f6802c9e9b57"),
        (31744,
         "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47860cc51f2b0c28a7b77304bd55fe73af663c02d3f52ea053ba43431ca5bab7bfea2f5e9d7121770d88f70ae9649ea713087d1914f7f312147e247f87eb2d4ffef0ac978bf7b6579d57d533355aa20b8b77b13fd09748728a5cc327a8ec470f4013226f",
         "efa53b389ab67c593dba624d898d0f7353ab99e4ac9d42302ee64cbf9939a4193a7258db2d9cd32a7a3ecfce46144114b15c2fcb68a618a976bd74515d47be08b628be420b5e830fade7c080e351a076fbc38641ad80c736c8a18fe3c66ce12f95c61c2462a9770d60d0f77115bbcd3782b593016a4e728d4c06cee4505cb0c08a42ec",
         "39772aef80e0ebe60596361e45b061e8f417429d529171b6764468c22928e28e9759adeb797a3fbf771b1bcea30150a020e317982bf0d6e7d14dd9f064bc11025c25f31e81bd78a921db0174f03dd481d30e93fd8e90f8b2fee209f849f2d2a52f31719a490fb0ba7aea1e09814ee912eba111a9fde9d5c274185f7bae8ba85d300a2b"),
        (102400,
         "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085e01c59dab908c04c3342b816941a26d69c2605ebee5ec5291cc55e15b76146e6745f0601156c3596cb75065a9c57f35585a52e1ac70f69131c23d611ce11ee4ab1ec2c009012d236648e77be9295dd0426f29b764d65de58eb7d01dd42248204f45f8e",
         "1c35d1a5811083fd7119f5d5d1ba027b4d01c0c6c49fb6ff2cf75393ea5db4a7f9dbdd3e1d81dcbca3ba241bb18760f207710b751846faaeb9dff8262710999a59b2aa1aca298a032d94eacfadf1aa192418eb54808db23b56e34213266aa08499a16b354f018fc4967d05f8b9d2ad87a7278337be9693fc638a3bfdbe314574ee6fc4",
         "4652cff7a3f385a6103b5c260fc1593e13c778dbe608efb092fe7ee69df6e9c6d83a3e041bc3a48df2879f4a0a3ed40e7c961c73eff740f3117a0504c2dff4786d44fb17f1549eb0ba585e40ec29bf7732f0b7e286ff8acddc4cb1e23b87ff5d824a986458dcc6a04ac83969b80637562953df51ed1a7e90a7926924d2763778be8560"),
    ];

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn official_vectors() {
        for &(n, want_hash, want_keyed, want_derive) in VECTORS {
            let input = pattern(n);
            let xof_len = want_hash.len() / 2;

            let mut h = Hasher::new();
            h.update(&input);
            let mut out = vec![0u8; xof_len];
            h.finalize_xof(&mut out);
            assert_eq!(hex(&out), want_hash, "hash, len {n}");
            assert_eq!(
                hex(&blake3(&input)),
                want_hash[..64],
                "one-shot hash, len {n}"
            );

            let mut k = Hasher::new_keyed(KEY);
            k.update(&input);
            let mut out = vec![0u8; xof_len];
            k.finalize_xof(&mut out);
            assert_eq!(hex(&out), want_keyed, "keyed, len {n}");
            assert_eq!(
                hex(&keyed_hash(KEY, &input)),
                want_keyed[..64],
                "one-shot keyed, len {n}"
            );

            let mut d = Hasher::new_derive_key(CONTEXT);
            d.update(&input);
            let mut out = vec![0u8; xof_len];
            d.finalize_xof(&mut out);
            assert_eq!(hex(&out), want_derive, "derive_key, len {n}");
            assert_eq!(
                hex(&derive_key(CONTEXT, &input)),
                want_derive[..64],
                "one-shot derive_key, len {n}"
            );
        }
    }

    /// Streaming in arbitrary pieces must agree with hashing in one call.
    /// Chunk and block boundaries are where an incremental hasher goes wrong,
    /// so the split sizes deliberately straddle 64 and 1024.
    #[test]
    fn incremental_matches_one_shot() {
        let input = pattern(20_000);
        for &split in &[1usize, 7, 63, 64, 65, 127, 1023, 1024, 1025, 4096, 8191] {
            let mut h = Hasher::new();
            for piece in input.chunks(split) {
                h.update(piece);
            }
            assert_eq!(h.finalize(), blake3(&input), "split {split}");
        }
    }

    /// The XOF is a stream, not a family of unrelated outputs: a longer request
    /// must extend a shorter one rather than replace it.
    #[test]
    fn xof_is_a_prefix_extension() {
        let mut h = Hasher::new();
        h.update(b"omni");
        let mut short = [0u8; 32];
        let mut long = [0u8; 500];
        h.finalize_xof(&mut short);
        h.finalize_xof(&mut long);
        assert_eq!(short, long[..32]);
        assert_eq!(h.finalize(), short);
    }

    /// The three keying modes must be domain-separated: identical input under
    /// different modes must not collide. This is what §03.5.3 relies on.
    #[test]
    fn modes_are_domain_separated() {
        let m = b"same input";
        let a = blake3(m);
        let b = keyed_hash(KEY, m);
        let c = derive_key(CONTEXT, m);
        let d = derive_key("a different context", m);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
        assert_ne!(c, d, "the context string must actually separate keys");
    }

    /// The tree helpers Bao needs (§13.3) must reconstruct the same root hash
    /// the streaming hasher produces. Four chunks give a balanced tree; the
    /// odd tail exercises the unbalanced right edge.
    #[test]
    fn tree_helpers_reconstruct_the_root() {
        // Balanced: exactly four chunks.
        let input = pattern(4 * CHUNK_LEN);
        let cvs: Vec<[u8; 32]> = input
            .chunks(CHUNK_LEN)
            .enumerate()
            .map(|(i, c)| chunk_chaining_value(c, i as u64))
            .collect();
        let left = parent_chaining_value(&cvs[0], &cvs[1]);
        let right = parent_chaining_value(&cvs[2], &cvs[3]);
        assert_eq!(parent_root(&left, &right), blake3(&input));

        // Unbalanced: three chunks. The left subtree is a power of two; the
        // remainder attaches at the root.
        let input = pattern(3 * CHUNK_LEN);
        let cvs: Vec<[u8; 32]> = input
            .chunks(CHUNK_LEN)
            .enumerate()
            .map(|(i, c)| chunk_chaining_value(c, i as u64))
            .collect();
        let left = parent_chaining_value(&cvs[0], &cvs[1]);
        assert_eq!(parent_root(&left, &cvs[2]), blake3(&input));

        // A single chunk has no parent node at all.
        let input = pattern(500);
        assert_eq!(chunk_root(&input, 0), blake3(&input));
    }

    /// A chunk's chaining value depends on its position, so a reader cannot be
    /// tricked into accepting a validly-hashed chunk at the wrong offset.
    #[test]
    fn chunk_counter_is_bound_into_the_hash() {
        let c = pattern(CHUNK_LEN);
        assert_ne!(chunk_chaining_value(&c, 0), chunk_chaining_value(&c, 1));
    }
}
