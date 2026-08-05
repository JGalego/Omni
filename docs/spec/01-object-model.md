# OMNI/1.0 — §1 Object Model

The object model is layer L1. Everything above it is expressed in terms of
objects and references; everything below it is a way of storing bytes.

## 1.1 Axioms

**A1. Immutability.** An object, once created, never changes. "Editing a model"
means creating new objects and a new root.

**A2. Content addressing.** An object's identity is the cryptographic digest of
its serialized bytes. Two objects with the same bytes are the same object,
everywhere, forever.

**A3. Acyclicity, by construction.** A reference contains the digest of its
target. Computing an object's digest requires its references' digests to already
exist. Therefore an object graph *cannot* contain a cycle. This is not a rule
validators enforce — it is arithmetically impossible to violate.

**A4. Locality of verification.** Verifying an object requires only that object's
bytes and its digest. Verifying a subgraph requires only the objects in it.
There is no global state, no central authority, and no ordering requirement.

**A5. Store independence.** Nothing in an object refers to a file, an offset, a
URL, or a machine. Objects can therefore be moved, copied, cached and mirrored
without rewriting.

These axioms are what make deduplication, delta models, resumable
downloads, partial loading, signing and reproducible builds all be *the same
mechanism* rather than five features.

## 1.2 Object

An object is the triple `(otype, payload, digest)` where:

- `otype` — a `u16` type code (§1.6) recorded in the index and, for structure
  objects, redundantly inside the payload as the `t` key.
- `payload` — the serialized bytes. For structure objects: canonical OMNI-CBOR
  (§03.2). For `Blob` objects: arbitrary bytes with no interpretation.
- `digest` — `H(payload)`, where `H` is the container's digest algorithm (§03.5).

> **The digest is over the *logical* payload, never over a compressed or
> encrypted encoding of it.** Storage-level codecs (§03.7) and at-rest encryption
> (§12.8) are properties of a *copy* of an object, not of the object. This one
> rule is what allows a zstd-compressed chunk in a container, a raw chunk in a
> page cache, and an AES-GCM chunk in S3 to be recognized as the same object and
> deduplicated against each other.

### 1.2.1 Structure objects vs. data objects

| | Structure object | Data object (`Blob`) |
|---|---|---|
| Encoding | canonical OMNI-CBOR map | opaque bytes |
| Typical size | 10² – 10⁵ bytes | 10⁵ – 10⁸ bytes |
| Contains refs | yes | never |
| Parsed on open | yes (lazily, by need) | never |
| Alignment | 8 bytes | container alignment (§02.9) |
| Digest tree | whole-object | whole-object + optional Bao tree (§13.3) |

The split exists because they have opposite access patterns: structure is small,
random, and must be parsed; data is large, sequential-or-mapped, and must never
be copied.

## 1.3 Digests

A digest is encoded as a byte string with a multihash-compatible prefix:

```
digest := varint(hash_algo) || varint(length) || raw_hash_bytes
```

Registered algorithms (§03.5):

| Code | Algorithm | Length | Status |
|---|---|---|---|
| `0x1e` | BLAKE3-256 | 32 | **MUST implement** (default) |
| `0x12` | SHA-256 | 32 | **MUST implement** |
| `0x13` | SHA-512 | 64 | MAY |
| `0x1f` | SHA-512/256 | 32 | MAY |
| `0x20` | SHA3-256 | 32 | MAY |
| `0x1e01` | BLAKE3-XOF-512 | 64 | MAY (future-margin profile) |

Codes follow the multicodec table where one exists, so OMNI digests are
interchangeable with IPFS/OCI tooling.

**Algorithm agility.** A container declares one *primary* algorithm in its header;
all index entries and internal refs use it. A container MAY additionally carry
`AltDigest` objects mapping primary → alternate digests, so a model signed under
BLAKE3 in 2030 can be re-attested under a post-quantum hash in 2045 without
rewriting a single tensor. See §14.6.

**Truncation.** The object index (§02.6) stores the first 32 bytes of the digest
for lookup only. Authoritative comparison MUST use the full digest from the ref.
A 32-byte lookup key gives ~2⁻¹²⁸ collision probability under the birthday bound
for any realistic object count; a collision would cause a lookup miss followed by
a full-digest mismatch, i.e. a clean error, never silent corruption.

## 1.4 References

A `Ref` is a CBOR array (compact) or map (extended):

```cbor-diag
; compact form: [otype, digest]
[3, h'1e20 3f2c…']

; extended form, used when hints are needed
{
  "o": 3,                     ; otype
  "d": h'1e20 3f2c…',         ; digest
  "n": 4194304,               ; logical byte length (hint, non-authoritative)
  "s": ["oci://ghcr.io/acme/llm@sha256:…"],  ; optional locator hints
  "c": 0                      ; criticality override (§11.3)
}
```

Rules:

- `d` is authoritative. `n` and `s` are **hints**: a reader MUST verify actual
  length and digest and MUST NOT trust `n` for allocation without bounds
  checking against the store's reported size (§12.4).
- `s` locators are *advisory mirrors*. A conforming reader MAY ignore them
  entirely; a security-conscious reader MUST NOT dereference them without policy
  (§12.9), since they are attacker-controlled URLs inside a downloaded file.
- A ref to an object not present in any reachable store is a **dangling ref**.
  Dangling refs are legal (that is what a partial/lazy container is) and are
  reported by `omni verify` at level L0 as `incomplete`, not `invalid`.

### 1.4.1 Ref sets and maps

Ordered collections use CBOR arrays. Name-keyed collections use CBOR maps with
text keys sorted per §03.2. Large name→ref maps (>4096 entries, e.g. tensor
tables of frontier models) SHOULD be split into a `ShardedMap` object:

```cbor-diag
{ "t": "omni.core/sharded-map", "v": 1,
  "hash": "blake3-256",         ; key hashing for bucket assignment
  "shards": 64,
  "buckets": [ [16, h'…'], [16, h'…'], … ]   ; refs to Map objects
}
```

This keeps any single structure object small enough to parse in one page-fault
burst and lets two models with 90 % identical tensor tables share buckets.

## 1.5 The DAG and reachability

The **root** of a container is a `Manifest` object identified in the superblock
(§02.5). *Reachable* means transitively referenced from the root.

- Objects that are reachable are part of the model.
- Objects that are present but unreachable are **loose** and MAY be garbage
  collected (`omni gc`). They are legal — that is what a checkpoint history or
  an append-log looks like mid-write.
- A container MAY declare additional roots in `superblock.roots[]` (multiple
  models per file, retained history, a base model and three fine-tunes).

**Pinning.** A `Pin` object records "keep these roots alive" with a reason and a
timestamp; `omni gc` treats pinned roots as GC roots. This is how a registry or a
local cache retains eviction policy inside the format rather than beside it.

## 1.6 Object type codes

`otype` is a `u16`. Values `0x0000–0x7FFF` are reserved for this specification;
`0x8000–0xFFFF` are available to plugins via the registry (§11.7) and are
guaranteed never to be assigned by the core spec.

| Code | Type | Section |
|---|---|---|
| `0x0000` | `Blob` (opaque data chunk) | §04.5 |
| `0x0001` | `Manifest` | §1.7 |
| `0x0002` | `Metadata` | §06 |
| `0x0003` | `Model` | §0.4 |
| `0x0004` | `TensorTable` | §04.2 |
| `0x0005` | `TensorDesc` | §04.2 |
| `0x0006` | `ChunkList` | §04.5 |
| `0x0007` | `Codebook` | §05.4 |
| `0x0008` | `GraphModule` | §07 |
| `0x0009` | `DialectRef` | §07.4 |
| `0x000A` | `Tokenizer` | §06.7 |
| `0x000B` | `ChatTemplate` | §06.9 |
| `0x000C` | `Adapter` | §08 |
| `0x000D` | `TrainingState` | §09 |
| `0x000E` | `ShardMap` (distributed checkpoint topology) | §09.4 |
| `0x000F` | `RuntimeCache` | §10.6 |
| `0x0010` | `CapabilitySet` | §10.2 |
| `0x0011` | `Plan` (realization) | §10.4 |
| `0x0012` | `Signature` | §12.5 |
| `0x0013` | `Provenance` (in-toto/SLSA attestation) | §12.6 |
| `0x0014` | `BaoTree` (verified-streaming outboard) | §13.3 |
| `0x0015` | `ObjectIndex` (derived accelerator) | §02.6 |
| `0x0016` | `NameIndex` (derived accelerator) | §02.6.4 |
| `0x0017` | `Schema` (embedded schema definition) | §03.4 |
| `0x0018` | `Rosetta` (embedded spec text, OMNI-A) | §14.8 |
| `0x0019` | `Foreign` (preserved source-format bytes) | import |
| `0x001A` | `Dataset` | §06.10 |
| `0x001B` | `Pin` | §1.5 |
| `0x001C` | `ShardedMap` | §1.4.1 |
| `0x001D` | `AltDigest` | §1.3 |
| `0x001E` | `PluginModule` (WASM) | §11.6 |
| `0x001F` | `Extension` (unknown-namespace container) | §11.3 |
| `0x0020` | `Evaluation` (benchmark results) | §06.8 |

Unassigned codes below `0x8000` MUST be treated as `Extension` with unknown
semantics, subject to criticality rules (§11.3).

## 1.7 Manifest

The `Manifest` is the object you sign, the object you name, and the only object
whose digest anyone needs to quote.

```cbor-diag
{
  "t": "omni.core/manifest", "v": 1,

  "id": "urn:omni:acme:llm-8b:2026-08-04",   ; optional stable human identifier
  "kind": "model",                            ; model | bundle | adapter | dataset | cache
  "created": 0,                               ; unix ms; 0 for reproducible builds

  "meta": [2, h'…'],                          ; -> Metadata

  "assets": {
    "text":      [3,  h'…'],                  ; -> Model
    "vision":    [3,  h'…'],                  ; -> Model
    "tokenizer": [10, h'…'],                  ; -> Tokenizer
    "README.md": [0,  h'…']                   ; -> Blob
  },
  "entry": "text",                            ; default asset for single-model tools

  "parents": [ [1, h'…'] ],                   ; -> Manifest (inheritance; §08.6)
  "compose": { … },                           ; how parents combine (§08.7)

  "features": {
    "required": ["omni.core/1.0",
                 "omni.tensor/expr.1",
                 "omni.quant/affine-block.1"],
    "optional": ["omni.rt/cuda-cache.1"]
  },

  "realizations": [ [17, h'…'] ],             ; -> Plan (precomputed; §10.5)
  "caches":       [ [15, h'…'] ],             ; -> RuntimeCache (droppable)
  "attestations": [ [18, h'…'], [19, h'…'] ], ; -> Signature / Provenance
  "ext": { "org.acme/deploy": [31, h'…'] }    ; namespaced extensions (§11)
}
```

Notes:

- A manifest with `kind: "bundle"` and several `Model` assets is how a diffusion
  pipeline (text encoder + UNet/DiT + VAE + scheduler config) or a multimodal
  stack ships as one artifact without merging unrelated tensor namespaces.
- `attestations` are inside the object that they sign only in the sense of being
  *listed* — the signature payload covers the manifest **with the
  `attestations` key removed** (§12.5.2). This resolves the self-reference
  paradox without a second manifest.

## 1.8 Stores

A **store** implements:

```
resolve(digest)              -> bytes | NotFound
resolve_range(digest, o, n)  -> bytes | NotFound | Unsupported
has(digest)                  -> bool
put(bytes) -> digest                  (writable stores)
iter(prefix)                 -> [digest]          (enumerable stores)
```

Store kinds defined by this specification:

| Store | Backing | Range reads | Enumerable | Writable |
|---|---|---|---|---|
| **Container** | one `.omni` file, `mmap` or `pread` | yes | yes | append-only |
| **Directory** | `.omnid/objects/ab/cdef…` | yes | yes | yes |
| **Pack set** | `.omnipack` + `.omni.idx` | yes | yes | append-only |
| **HTTP** | ranged GET on a container or per-object URLs | yes | no | no |
| **OCI** | registry blobs | partial | via manifest | yes |
| **Memory** | in-process map | yes | yes | yes |

Stores compose: a runtime typically layers
`Memory → Directory(local cache) → Container(mmap) → HTTP(remote)`, resolving in
order and back-filling. Because identity is a hash, this layering needs no
invalidation logic — a cache entry is either present and correct, or absent.

## 1.9 Packs

Storing 200 000 chunk objects as 200 000 files is hostile to filesystems and
fatal to container registries. OMNI adopts Git's answer: a **pack** is a
concatenation of objects plus a sorted index.

```
model.omnipack     : [ObjPack header][obj][obj]…[obj]
model.omni.idx     : sorted (digest → offset, length, otype, codec)
```

A `.omni` container is precisely a pack with a file header, superblock, segment
framing and a trailer (§02). Packs are therefore the interchange unit for
registries: one pack per "layer" bundles thousands of chunks into one blob a CDN
can cache, while the index preserves per-object addressability.

**Pack partitioning strategy** (advisory, `omni pack --strategy`):

| Strategy | Grouping | Best for |
|---|---|---|
| `linear` | write order | archival, single-file distribution |
| `by-tensor` | one pack per tensor | fine-grained partial fetch |
| `by-layer` | tensors grouped by graph depth | streaming / progressive load |
| `by-dtype` | group same-precision chunks | mixed-precision partial fetch |
| `by-novelty` | chunks unique to this model vs. inherited | delta distribution (§08.6) |

`by-novelty` is the important one: it puts everything a client already has (from
the base model) in packs the client can skip entirely.

## 1.10 Reproducible construction

`omni pack` MUST be deterministic: identical inputs produce byte-identical
output. Requirements:

1. Object emission ordered by (otype, digest) ascending — no map/hash iteration
   order.
2. All timestamps zero unless `--timestamp` is given explicitly.
3. Codec parameters fully specified (§03.7); "zstd level 3, no dictionary, no
   long mode" is part of the recipe, and codec implementations that do not
   guarantee bit-stable output across versions MUST be pinned by version in the
   `codec` descriptor.
4. UUIDv7 in the header derived deterministically from the root digest when
   `--reproducible` is set (`uuid = truncate(H("omni-uuid" || root_digest))` with
   version/variant bits fixed).
5. No padding byte is ever uninitialized; padding is zero (§02.9).

This makes "did these two build pipelines produce the same model?" a byte
comparison, and makes OMNI containers usable as SLSA build outputs (§12.6).

## 1.11 Why not just use Git / IPFS / OCI directly?

We use their ideas and interoperate with their stores, but none of them is
sufficient alone:

- **Git** has no typed objects beyond blob/tree/commit, no alignment guarantees,
  no `mmap`-friendly layout for multi-gigabyte values, no partial-object reads,
  and a hash algorithm migration that took a decade. Git-LFS is an admission of
  this.
- **IPFS/IPLD** gets addressing and typed links right but mandates a 256 KiB-ish
  block DAG structure that destroys `mmap` locality for tensors and imposes
  network semantics on local files.
- **OCI** gets distribution right (§13.5 maps OMNI onto it) but its manifests
  are JSON blobs with no object model — layers are tarballs, and tar has no
  random access, no alignment, and no dedup below file granularity.

OMNI is the missing middle: typed content-addressed objects with *alignment and
range semantics designed for tensors*, that can be **stored in** any of the
above.

**Prev:** [§00 Overview](00-overview.md) · **Next:** [§02 Container Binary Format](02-container.md)
