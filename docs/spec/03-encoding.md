# OMNI/1.0 — §3 Encoding, Hashing, and Compression

## 3.1 Why CBOR

Structure objects are encoded in **OMNI-CBOR**, a deterministic profile of CBOR
(RFC 8949).

The requirements were:

1. **Self-describing.** A reader with no schema must still recover the full
   structure. This is the 50-year requirement (§00.9) and it eliminates
   Protobuf, FlatBuffers, Cap'n Proto, Thrift, and every schema-first format.
2. **Canonical.** Content addressing demands one byte encoding per value.
   JSON has no canonical form worth the name (number formatting, key order,
   Unicode escapes); Protobuf explicitly disclaims canonical serialization.
3. **Binary-native.** Digests, keys and small tensors are byte strings. JSON
   would base64 everything: +33 % size and a parse step.
4. **Streaming-capable.** A decoder must produce values incrementally.
5. **Boring and universal.** CBOR is an IETF Standards-Track RFC with
   independent implementations in every language that matters, is used by COSE,
   WebAuthn, and CoAP, and has no corporate owner.
6. **Integer- and float-exact.** RFC 8949's deterministic rules give exact,
   shortest-form encodings with no lossy float printing.

The cost of CBOR is parsing on the hot path. OMNI removes that cost where it
matters by making the object index a fixed-layout binary table (§02.6) and by
never putting tensor payloads inside CBOR. Structure objects are parsed once,
cached in memory, and are tiny relative to weights.

## 3.2 OMNI-CBOR: the deterministic profile

An OMNI-CBOR encoding MUST satisfy RFC 8949 §4.2.1 (core deterministic encoding)
plus:

| Rule | Requirement |
|---|---|
| D1 | Integers use the shortest form; no unnecessary 8/16/32/64-bit widths. |
| D2 | Definite-length encoding only. Indefinite-length strings, arrays and maps are forbidden. |
| D3 | Map keys MUST be text strings or unsigned integers, and MUST be sorted by their **encoded byte representation**, ascending, lexicographically. |
| D4 | Duplicate map keys are a hard error (encoder and decoder). |
| D5 | Floats use the shortest form that round-trips exactly (`f16` → `f32` → `f64`). |
| D6 | The only NaN encoding permitted is `f9 7e00` (quiet, positive, zero payload). `-0.0` is permitted and distinct from `0.0`. |
| D7 | Only registered tags may be used (§3.3). Unregistered tags are a hard error in structure objects. |
| D8 | No trailing bytes after the top-level item. |
| D9 | Text strings MUST be valid UTF-8 in NFC (Unicode Normalization Form C). |
| D10 | Top-level item MUST be a map with keys `t` (text, schema URI) and `v` (uint, schema version) present. |

D3's "sort by encoded bytes" makes integer keys sort before text keys and gives a
total order requiring no Unicode collation — a decision that would otherwise
haunt implementations in 30 languages.

D9's NFC requirement prevents two visually identical tensor names from hashing
differently, which would silently break dedup and adapter attachment.

**Verification:** a decoder in strict mode MUST re-encode and compare, or
equivalently validate the rules inline, and MUST reject non-canonical input in
any context where the bytes are hashed. Lenient mode (for importing
third-party CBOR) is permitted only outside the trust boundary.

## 3.3 Registered tags

| Tag | Meaning | Content |
|---:|---|---|
| 2 / 3 | bignum (positive/negative) | byte string — for parameter counts > 2⁶⁴, symbolic dims |
| 30 | rational | `[num, den]` — exact scale factors, RoPE ratios |
| 1001 | `omni.ref` | `[otype, digest]` or extended map (§01.4) |
| 1002 | `omni.digest` | byte string, multihash-prefixed |
| 1003 | `omni.dtype` | dtype descriptor map (§04.3) |
| 1004 | `omni.shape` | array of uint or text (symbolic dims) |
| 1005 | `omni.expr` | tensor expression node (§04.7) |
| 1006 | `omni.uri` | text — an identifier, **not** a fetch instruction |
| 1007 | `omni.decimal` | `[exp, mantissa]` — exact decimals for licensing/pricing/eval scores |

Tags are optional sugar: every tagged value has an untagged canonical form, and
decoders MUST accept both. Tags exist so that generic CBOR tooling (`cbor2diag`,
`jq`-for-CBOR) renders OMNI objects legibly.

## 3.4 Schemas

Every structure object carries `t`, a **schema URI**, and `v`, an integer version.

```
"t": "omni.core/manifest"        v: 1
"t": "omni.tensor/desc"          v: 1
"t": "org.acme/deployment-hints" v: 3
```

Schemas are described in **OSD** (OMNI Schema Description), itself a CBOR
document, expressive enough for: required/optional keys, types, ranges,
enumerations, ref target types, and cross-field constraints. OSD is deliberately
weaker than a general logic language — it is total, decidable, and cheap.

A container MAY embed `Schema` objects (`otype 0x0017`) for every schema URI it
uses. The archival profile (§14.8) **requires** this: an OMNI-A file can be fully
validated with no network and no registry.

Schema evolution rules:

- Adding an optional key: same `v`.
- Adding a required key, changing a type, or changing semantics: `v + 1`.
- A reader encountering `v` greater than it knows MUST apply the object's
  criticality rules (§11.3): ignore if non-critical, fail if critical.
- A reader encountering `v` *lower* than it knows MUST support it. Forever.
  There is no deprecation of *reading*; only of *writing* (§14.5).

## 3.5 Hashing

### 3.5.1 Algorithm choice

**BLAKE3-256 is the default**; SHA-256 MUST also be implemented for
interoperability with OCI, Sigstore, and every existing supply-chain tool.

BLAKE3 is chosen for properties that matter here and nowhere else:

1. **Parallel.** Hashing a 400 GB model at ~5–15 GB/s per core-group means
   verification is not the bottleneck; SHA-256 without hardware acceleration is
   ~1–2 GB/s single-core and would make full verification a minutes-long tax.
2. **Tree-structured (Bao).** BLAKE3's internal Merkle tree means an arbitrary
   byte range of a large object can be verified against the root hash with a
   logarithmic proof. This is what makes *verified partial download* and
   *verified `mmap` page faults* possible (§13.3). No flat hash can do this.
3. **XOF.** Extendable output gives us key derivation (§12.8) and longer digests
   for a post-quantum margin from the same primitive.

SHA-256 remains mandatory because the ecosystem OMNI must live in — OCI
registries, Sigstore/Rekor, SLSA, existing HF hashes — speaks SHA-256. A
container MAY carry both via `AltDigest` (§01.3).

### 3.5.2 What gets hashed

| Thing | Digest input |
|---|---|
| Structure object | its canonical OMNI-CBOR bytes |
| Data object (chunk) | its **logical** (uncompressed, unencrypted) bytes |
| Tensor value | the digest of its expression tree (§04.7.5) — *not* its materialization |
| Manifest | its canonical bytes with `attestations` removed (§12.5.2) |
| Container | `H(header ‖ all segments ‖ trailer[0..56])` — a whole-file digest for transport only |
| Plan | canonical bytes; used as the cache key for realizations (§10.4) |

### 3.5.3 Domain separation

All non-object hashing uses BLAKE3's keyed/derive-key modes with explicit context
strings, so a hash computed for one purpose can never be replayed as another:

```
omni/1.0 object
omni/1.0 expr-identity
omni/1.0 plan-key
omni/1.0 chunk-encryption-key
omni/1.0 uuid
```

## 3.6 Chunking

Data objects are produced from logical byte streams by a **chunker**. The choice
is recorded in the owning `ChunkList` (§04.5) so a reader never has to guess.

| Chunker | Parameters | Properties |
|---|---|---|
| `fixed` | `size` (default 4 MiB), `align` | **Default.** Aligned, `mmap`-friendly, O(1) offset→chunk math, perfect random access. No shift-resilience. |
| `cdc-gear` | `min`, `avg`, `max`, `mask` | FastCDC-style content-defined boundaries. Shift-resilient: an inserted row in a tensor re-chunks only locally. Costs a rolling hash pass and destroys offset math. |
| `row` | rows per chunk | Boundaries at tensor row/slice granularity — enables per-expert or per-head partial fetch. |
| `block` | quantization block count | Boundaries aligned to quantization blocks (§05), so a chunk is always a whole number of blocks and can be dequantized independently. |
| `none` | — | One chunk per tensor. Simplest; poor for dedup and resumption. |

**Guidance.** Use `fixed` for weights that are published once (the common case);
use `cdc-gear` when publishing many *near-identical* variants of a large tensor,
where its dedup wins outweigh the loss of offset arithmetic; use `block` for any
quantized tensor, so that partial loads never straddle a block; use `row` for MoE
expert weights so a runtime can fetch only the experts it will route to.

Chunk size involves competing pressures:

- Smaller chunks → better dedup, finer partial fetch, more resumable; but more
  objects (index size = 64 B × n), more syscalls, more per-chunk overhead.
- 4 MiB is the sweet spot for NVMe (deep-queue sequential throughput) and for
  object stores (S3 charges per request; 4 MiB keeps request count sane).
- 64 KiB–256 KiB is right for HTTP progressive loading on lossy links.
- For a 70 B model in bf16 (140 GB), 4 MiB chunks → 35 000 objects → 2.2 MiB
  index. Trivial.

## 3.7 Compression

Compression is a property of a **stored copy** of an object, never of the object
(§01.2). Consequences: recompressing a container changes no digests; two
containers using different codecs dedup against each other; and a lossy codec can
never silently change a model's identity.

### 3.7.1 Codec registry

| ID | Codec | Status | Notes |
|---|---|---|---|
| `raw` | none | **MUST** | |
| `zstd` | Zstandard (RFC 8878) | **MUST** | levels 1–22, optional dictionary object |
| `deflate` | RFC 1951 | SHOULD | required by archival profile |
| `lz4` | LZ4 block | MAY | when decode speed > ratio |
| `brotli` | RFC 7932 | MAY | text-heavy metadata |
| `xz` | LZMA2 | MAY | archival ratio |
| `bitshuffle+zstd` | byte/bit transpose then zstd | SHOULD | **the important one for tensors** |
| `zfp` | ZFP | MAY | **lossy**, must set `LOSSY` |
| `sz3` | SZ3 | MAY | **lossy**, must set `LOSSY` |
| `ans-lut` | rANS with per-block LUT | MAY | codebook-quantized weights |

Codec descriptors are explicit and complete so that compression is reproducible:

```cbor-diag
{"id":"zstd", "level":3, "window_log":27, "long":0,
 "dict": [0, h'…'],           ; optional dictionary object ref
 "impl":"libzstd", "ver":">=1.5.0 <2"}
```

### 3.7.2 Why bitshuffle matters

Floating-point weight tensors compress badly with general-purpose codecs (~1.05×
for bf16) because entropy is spread across every byte. Transposing to
bit-plane or byte-plane order groups the highly-redundant exponent bytes
together. Typical observed ratios (see §performance for methodology and
caveats): bf16 weights **1.15–1.35×** with `bitshuffle+zstd` vs. **1.02–1.08×**
with plain zstd. Not spectacular — weights are near-incompressible by
construction — but free, and the transform is cheap and SIMD-friendly.

The honest guidance: **do not expect compression to shrink weights.** OMNI's
size wins come from deduplication, deltas and quantization-as-transformation
(§05, §08), which are order-of-magnitude effects, not from entropy coding, which
is a percentage effect.

### 3.7.3 Lossy codecs

A lossy codec MAY be used **only** when:

1. The index entry sets `oflags.LOSSY`.
2. The aux record declares the error bound
   (`{"mode":"abs","bound":1e-3}` / `"rel"` / `"psnr"`).
3. The object is *not* on a canonical identity path — i.e. it is a `cacheable`
   object, or the tensor expression explicitly wraps it in `approx(...)` so the
   approximation is visible in the value algebra.

A lossy transformation that is invisible in the DAG is forbidden. This rule
exists because a format that lets you silently change weights is a format that
cannot be used to certify a model.

### 3.7.4 Decompression safety

`logical_len` in the index is an authoritative allocation bound. Readers MUST
pre-allocate exactly `logical_len`, MUST abort if the codec produces more, and
MUST reject a declared ratio above 1000:1 unless `features.optional` contains
`omni.codec/high-ratio.1`. See §12.4.

### 3.7.5 `ans-lut`

Every other codec in §3.7.1 is somebody else's format, and its bitstream is
defined by their specification. `ans-lut` is OMNI's own, and until it was
written down the registry named an identifier no two implementations could have
agreed on. This is the definition.

**What it is for.** A codebook-quantized weight (§5.2) is a stream of small
indices whose distribution is strongly skewed — an NF4 tensor's indices are not
uniform over sixteen values, and neither are a k-means codebook's. A
general-purpose LZ coder finds no matches in such a stream and spends its
entropy coder on a byte alphabet it models badly. Range-coding the indices
against a table measured from the block itself is the operation that fits, and
it is cheap in both directions: one multiply, one shift and one table lookup per
symbol.

**The stream.** All integers are little-endian.

```
u8      version               ; 1
u8      log2_scale            ; 8..16; the frequency table sums to 1 << log2_scale
u32     block_elems           ; symbols per block, the last block may be shorter
varint  block_count
block*                        ; block_count of them
```

Each block is:

```
u8      kind                  ; 0 = stored, 1 = coded
u32     symbol_count          ; symbols in this block
if kind == 0:
    u8  bytes[symbol_count]   ; the block verbatim
if kind == 1:
    u8  used                  ; distinct symbols, minus one (so 1..256)
    (u8 symbol, u16 freq)*    ; `used + 1` pairs, symbols strictly increasing
    u32 payload_len
    u8  payload[payload_len]  ; the rANS stream
```

A block whose entropy coding would not be smaller than the block MUST be written
`stored`, so the codec never expands its input by more than five bytes a block.

**The coder.** A single-state rANS over a 32-bit state, renormalizing sixteen
bits at a time, with `L = 1 << 16` as the lower bound of the normalized
interval — so the state lives in `[L, L << 16)` and fits a 32-bit word exactly. `freq[s]` is the block's measured frequency of `s`, scaled so the
frequencies sum to exactly `1 << log2_scale`, and every symbol that occurs has
`freq[s] ≥ 1`. `cum[s]` is the exclusive prefix sum. The **LUT** the name refers
to is the inverse map — `1 << log2_scale` entries, `slot → symbol` — which is
what makes decoding one lookup rather than a search, and which a decoder builds
from the frequency table rather than reading from the stream.

Encoding processes the block's symbols **in reverse**, so that decoding runs
forwards:

```
x ← L
for s in reverse(symbols):
    while x ≥ ((L >> log2_scale) << 16) × freq[s]:
        emit16(x & 0xFFFF); x ← x >> 16
    x ← ((x ÷ freq[s]) << log2_scale) + (x mod freq[s]) + cum[s]
emit32(x)
```

The payload is then **reversed byte-wise**, so a decoder reads it forwards:

```
x ← read32()
repeat symbol_count times:
    slot ← x & ((1 << log2_scale) − 1)
    s    ← lut[slot]
    x    ← freq[s] × (x >> log2_scale) + slot − cum[s]
    while x < L: x ← (x << 16) | read16()
    emit s
```

**Reader rules.**

* **R-C20** the frequencies MUST sum to exactly `1 << log2_scale`; a table that
  does not is invalid, not a decoding hint.
* **R-C21** a symbol listed in the table MUST have `freq ≥ 1`, and the listed
  symbols MUST be strictly increasing, so the table has one canonical form and
  two writers produce the same bytes for the same block.
* **R-C22** the decoder MUST stop after `symbol_count` symbols and MUST NOT
  read past `payload_len`; a stream that ends early is invalid rather than
  padded.
* **R-C23** `log2_scale` MUST be between 8 and 16. Below 8 a 256-symbol
  alphabet cannot be represented; above 16 the LUT stops fitting in cache,
  which is the only reason this codec is worth having.

**What it is not.** `ans-lut` has no match finder and no context modelling: it
is an order-0 coder over one block. On text it loses to `deflate`; on a
bitshuffled float tensor it loses to `zstd`. It exists for the case the registry
names, and a writer that applies it to anything else is choosing badly rather
than doing something forbidden.

## 3.8 Encryption (summary)

Optional; see §12.8. Applied per-object, after compression, using AEAD
(XChaCha20-Poly1305 default, AES-256-GCM-SIV permitted). Digests remain over
plaintext, so encrypted and plaintext copies of a model share identity — which is
a deliberate tradeoff: it preserves dedup and verification but reveals
*equality* of plaintexts to anyone holding both. A `random-key` mode is provided
for deployments where that leak is unacceptable, at the cost of dedup.

## 3.9 Endianness and floats

- The container is little-endian (§02.3). Big-endian hosts byte-swap on read.
  There is no big-endian container variant: the last big-endian platform in
  ML use is gone, and supporting both would double the test matrix and create
  two digests for one tensor.
- Tensor element encoding is defined bit-exactly per dtype (§04.3), including
  subnormal and NaN handling, so that a `cast` performed by two implementations
  yields identical bytes. Where a dtype admits multiple NaN payloads, the
  canonical encoder MUST emit the canonical NaN, and a validator MAY flag
  non-canonical NaNs (which are a known channel for hiding data in weights).

**Prev:** [§02 Container](02-container.md) · **Next:** [§04 Tensors](04-tensors.md)
