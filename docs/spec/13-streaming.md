# OMNI/1.0 — §13 Streaming, Transport, and Distribution

The unit of transfer is the **object**, not the file. Everything in this section
follows from that.

## 13.1 Access patterns to support

| Pattern | Requirement |
|---|---|
| Local `mmap` | alignment, no parsing on the hot path (§02.9, §02.6) |
| NVMe with deep queues | large contiguous chunk reads, batched |
| GPUDirect Storage / `O_DIRECT` | 4 KiB-aligned, page-aligned chunk boundaries |
| HTTP from a CDN | range requests, cacheable immutable URLs, no suffix-range dependency |
| S3 / GCS / Azure Blob | few large requests; range GETs; parallel prefix listing |
| Container registry (OCI) | content-addressed blobs, dedup across tags, existing infra |
| Peer-to-peer / LAN cache | verifiable partial data from untrusted peers |
| Resumable download over a bad link | chunk granularity, no restart |
| Progressive load (start before complete) | topological ordering + per-object verification |

## 13.2 Streaming order

In the `stream` container profile (§02.2), objects are emitted so that a
forward-only reader can act as early as possible:

```
1  FileHeader, front Superblock, ObjectIndex
2  Manifest, Metadata, features, signatures        → identity + policy decisions
3  TensorTable / shard directory, Tokenizer header  → planning possible
4  Tokenizer vocabulary + merges                    → can tokenize input NOW
5  GraphModule                                      → can compile/plan NOW
6  Tensors in TensorTable.order:
     embeddings → layer 0 → layer 1 → … → final norm → lm_head
7  Adapters, then caches, then everything droppable
```

A runtime can: decide whether it can run the model after step 2 (~50 KB);
tokenize the prompt after step 4 (~5 MB); begin prefill on layer 0 after the
first layer arrives; and produce a first token before the file finishes if it
executes layer-wise as weights arrive.

`TensorTable.order` is normative for this purpose: writers targeting `stream`
MUST emit tensors in an order consistent with it, and it MUST be a valid
execution order for the graph's first forward pass.

## 13.3 Verified streaming (the key primitive)

Verifying partial data is not optional — without it, "start executing before
download completes" means "execute unverified bytes".

BLAKE3 is a Merkle tree over 1 KiB chunks. The **Bao** outboard encoding stores
the interior tree nodes separately from the data (2 × 32 bytes per 1 KiB group,
≈6 % overhead at 1 KiB granularity, ≈0.1 % at 64 KiB granularity by pruning to a
declared verification granularity).

```cbor-diag
{ "t":"omni.stream/bao", "v":1,
  "target": h'…',          ; the object this tree verifies
  "granularity": 65536,    ; smallest independently verifiable span
  "size": 4294967296,
  "tree": [0, h'…'] }      ; -> Blob of interior hashes
```

With it:

- any byte range can be verified against the object's root digest with a
  ~log₂(n) proof, using only bytes already received;
- a `mmap`-backed reader can verify **on page fault**, so a 400 GB model is
  verified exactly to the extent it is actually read;
- data from an untrusted peer or CDN edge is safe without trusting the source;
- corruption is localized to a 64 KiB span rather than invalidating a whole
  object.

Bao trees are `CACHEABLE` derived objects: recomputable from the data, so they
are optional and may be dropped.

## 13.4 HTTP profile

### 13.4.1 Opening a remote container

```http
GET /models/llm-8b.omni
Range: bytes=-64                        → FileTrailer            (64 B)

GET /models/llm-8b.omni
Range: bytes=812993740800-812993744896  → Superblock             (~4 KB)

GET /models/llm-8b.omni
Range: bytes=…                          → ObjectIndex            (~2 MB, cacheable)

GET /models/llm-8b.omni
Range: bytes=…, …, …                    → Manifest+Metadata      (multipart ranges)
```

**A fully planned load of an arbitrarily large model in three round trips**, and
the index is immutable so a CDN caches it forever.

For servers or CDNs with poor suffix-range support, publish the sidecar:

```http
GET /models/llm-8b.omni.idx             → header + superblock + index, one object
```

which collapses the first three requests into one.

### 13.4.2 Fetching tensors

- Coalesce adjacent chunk ranges into one request; the index makes this a sort
  and a merge. Target 8–64 MiB per request over WAN, 4–16 MiB to object stores.
- Issue requests in `TensorTable.order` with a configurable window (default 8
  concurrent).
- Use `If-Match`/`ETag` or, better, rely on immutability: object URLs derived
  from digests never change, so `Cache-Control: public, max-age=31536000,
  immutable` is always correct.
- Resume: re-request only the ranges of chunks not yet complete. Because
  verification is per-object (and per-span with Bao), a partially received chunk
  is not wasted.

### 13.4.3 Per-object URLs

An alternative layout serves each object at `/objects/<multibase-digest>`:

```
GET /objects/b3-2f8a…        → the chunk, immutable, dedup across every model
```

This is optimal for CDN cache-hit rates across a hub with thousands of related
models (every model sharing a tokenizer or a base layer shares URLs) at the cost
of many small requests. A hub SHOULD serve both: packs for cold downloads,
per-object for deltas and cache fills.

## 13.5 OCI registry mapping

OMNI maps cleanly onto OCI distribution, which means it inherits registries,
mirrors, CDNs, auth, replication, and signing infrastructure that already exists
everywhere.

```
OCI image manifest
  mediaType: application/vnd.oci.image.manifest.v1+json
  artifactType: application/vnd.omni.model.v1
  config:  application/vnd.omni.manifest.v1+cbor      ← the OMNI Manifest object
  layers:
    - application/vnd.omni.pack.v1        (structure objects pack)
    - application/vnd.omni.pack.v1        (tensor pack: layers 0–7)
    - application/vnd.omni.pack.v1        (tensor pack: layers 8–15)
    - …
    - application/vnd.omni.index.v1       (object index)
  subject: <the base model's manifest>    ← OCI referrers: deltas point at bases
  annotations:
    dev.omni.canonical-digest: b3:…
    dev.omni.params: "8030261248"
```

Properties this buys:

- **Registry-level dedup**: a delta model's packs are new blobs; the base's packs
  are already present and are *not re-uploaded*. `docker push` semantics for
  model fine-tunes.
- **Referrers API** links adapters, signatures, SBOMs, provenance and evaluation
  results to a model without changing its digest.
- **Cosign/Notation** work unmodified on the OCI layer, alongside OMNI's own
  COSE signatures (belt and braces; the OMNI signature survives export out of
  the registry, the OCI signature does not).
- Pack partitioning (§01.9) is exactly the "layer" decision: `by-novelty` packing
  makes a fine-tune's push a few hundred MB.

**Caveat:** registries dislike very large individual blobs and very many small
ones. Target 100 MB – 2 GB packs. The reference packer defaults to 1 GiB packs
for registry export and single-file layout for direct distribution.

## 13.6 Object stores

For S3/GCS/Azure:

| Layout | Requests to load | Notes |
|---|---|---|
| Single `.omni` object + ranged GETs | 3 + ⌈bytes/range_size⌉ | simplest; use 16–64 MiB ranges, 16-way parallel |
| Sharded packs (one object per pack) | 1 per pack | better parallelism, better retry granularity |
| Per-object keys | 1 per chunk | pathological request counts; only for deltas |

Recommended: `log2_align = 16` (64 KiB) for object-store-primary distribution to
match minimum billable read sizes, and pack sizes ≥ 256 MiB.

Cost model (S3-class pricing, order of magnitude): 140 GB model, 4 MiB chunks =
35 000 GET requests ≈ \$0.014 if fetched individually; with 64 MiB coalescing,
2 200 requests ≈ \$0.001. Both negligible against egress — the reason to coalesce
is latency and connection overhead, not request cost.

## 13.7 `omni://` URIs

```
omni://acme/llm-8b@b3:2f8a…                 exact, immutable
omni://acme/llm-8b:2026.08.1                mutable tag → resolves to a digest
omni://acme/llm-8b:latest?plan=min-memory   resolution hint
omni+oci://ghcr.io/acme/llm-8b@sha256:…     explicit transport
omni+https://cdn.acme.com/m/llm-8b.omni     explicit transport
```

Resolution of a mutable tag MUST yield a digest, and consumers SHOULD pin the
digest thereafter. Tag resolution is the only mutable operation in the entire
system, and it is deliberately isolated at the edge.

## 13.8 Lazy and partial containers

A container may deliberately omit objects (`flags.PARTIAL`, index entries with
`offset = 0` and `EXTERNAL`). Uses:

- **Index-only containers**: a 3 MB file that fully describes a 700 GB model —
  inspectable, verifiable against signatures, plannable, with weights fetched on
  demand. This is what a model catalogue should ship.
- **Layer-subset containers**: pipeline-parallel serving where rank 3 holds only
  layers 24–31.
- **Cache warm-up sets**: the first N tensors, prefetched to a node.

`omni inspect` reports `complete: no (12 %, 291 tensors described, 34 local)`.
Nothing about this is exceptional in the object model — a partial container is
just a store that answers `NotFound` for some digests.

## 13.9 FUSE / virtual filesystem

`omni mount model.omni /mnt/m` exposes:

```
/mnt/m/
  manifest.json          rendered view
  metadata.json
  tensors/model.layers.0.attn.q_proj.weight     ← materialized on read, cached
  tensors.safetensors                            ← a synthesized safetensors view
  tokenizer.json                                 ← synthesized HF-compatible view
```

Reads are lazy and range-driven (§04.7.4), so `head -c 1024` on a 4 GB tensor
fetches one chunk. This is the compatibility bridge: **existing tools that only
know how to open safetensors can open an OMNI model** without conversion,
without a full copy, and with dedup and verification underneath.

## 13.10 Peer-assisted and offline distribution

Because objects are content-addressed and verifiable in spans:

- A LAN cache (or a peer) can serve chunks it has, verified by the client against
  the manifest's digests. Trust is not required.
- A cluster scheduler can seed a node from its neighbours instead of the origin
  — the classic "1000 nodes pull a 140 GB model" problem becomes a torrent-shaped
  problem with cryptographic integrity.
- Air-gapped transfer is `omni pack --bundle-parents` onto media; the receiving
  side deduplicates against whatever it already has.

## 13.11 Streaming a model that does not exist yet

The `append` profile (§02.2) supports writing a container while a training run
is still producing it: each flush appends objects and a new superblock. A reader
opening it mid-write finds the last valid superblock and sees a consistent
snapshot. Torn writes are detected by CRC and by the trailing superblock's
digest, and never produce a partially-valid model — the previous superblock
remains authoritative.

**Prev:** [§12 Security](12-security.md) · **Next:** [§14 Versioning & Migration](14-versioning.md)
