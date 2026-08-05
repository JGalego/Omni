# Performance Analysis

> **Methodological note.** Every number in this document is an **analytic model**
> derived from published device characteristics and from the format's structural
> properties, not a measurement of a running implementation — the reference
> implementation is at the stage described in the [roadmap](roadmap.md). Models
> are stated with their assumptions so they can be falsified. Where a claim is
> structural (e.g. "two round trips to open"), it follows from the specification
> and is not an estimate.

---

## 1 What determines model load time

```
T_load = T_open + T_index + T_plan + T_transfer + T_materialize + T_verify
```

| Term | OMNI cost | Dominated by |
|---|---|---|
| `T_open` | 2 reads (64 B + ~4 KB) | latency |
| `T_index` | mmap, lazy page faults | nothing (amortized) |
| `T_plan` | O(tensors), pure metadata | ~1 ms for 300 tensors |
| `T_transfer` | bytes actually needed / bandwidth | **this is everything** |
| `T_materialize` | 0 for direct-map; O(bytes) otherwise | expression choice |
| `T_verify` | O(bytes read) / hash throughput | BLAKE3 ≈ 5–15 GB/s multicore |

The design's entire load-time strategy is: **make `T_transfer` count only the
bytes you need, and make `T_materialize` zero whenever the hardware can consume
the stored form directly.**

## 2 Open latency (structural, not estimated)

| Operation | Requests | Bytes | Notes |
|---|---:|---:|---|
| Local sealed container | 2 `pread` | ~4 KiB | trailer → superblock |
| Local + index resident | +0 | +0 | index is mmap'd, faulted on demand |
| HTTP, suffix range supported | 3 | ~4 KiB + index | trailer, superblock, index |
| HTTP, `.omni.idx` sidecar | 1 | index | single request |
| S3/GCS | 3 | same | ranged GET |
| OCI registry | 2 | manifest + config | config *is* the OMNI manifest |

Comparison at the same task ("what is in this model?"):

| Format | Requests | Bytes read |
|---|---:|---:|
| **OMNI** | 2–3 | ~200 KiB |
| safetensors (single file) | 2 | 8 B + header (0.1–20 MiB, must read all of it) |
| safetensors (sharded, 30 shards) | 31+ | index JSON + 30 headers |
| GGUF | 1–2 | KV section (0.1–50 MiB; must scan sequentially) |
| PyTorch `.pt` | 1 | must read + unpickle the whole zip directory and pickle stream |
| ONNX | 1 | full protobuf parse (weights inline unless external-data) |

The structural difference: OMNI's metadata is a *bounded, fixed-position region*,
while safetensors' and GGUF's grow with tensor count and must be parsed linearly.

## 3 Zero-copy loading

For a tensor whose plan chose direct-map, the byte path is:

```
NVMe → page cache → (mmap) → consumer pointer
```

with **no copy** and no parse. Requirements met by §02.9: 4 KiB alignment, no
interleaved framing inside the payload, layout fully described in metadata.

Modeled cold-load of a 140 GiB bf16 model, direct-map, on 4× PCIe 5.0 NVMe in
RAID-0 (~48 GB/s sequential, realistically ~35 GB/s with filesystem overhead):

| Step | Time |
|---|---|
| open + plan | ~2 ms |
| transfer 140 GiB @ 35 GB/s | ~4.1 s |
| verify (BLAKE3, 16 cores, ~12 GB/s) | ~11.7 s, **overlapped** with transfer |
| materialize | 0 |
| **total (verify-overlapped, verify-bound)** | **≈ 12 s** |
| total with `--verify V0` (structural only) | ≈ 4.1 s |

Verification, not I/O, becomes the bottleneck on very fast storage — which is
exactly why BLAKE3 (parallel, ~10× SHA-256's single-core rate) was chosen over
SHA-256 as the default. With SHA-256 (~2 GB/s/core, 16 cores ≈ 25 GB/s with
perfect scaling, realistically ~15 GB/s) the gap narrows but BLAKE3 remains
ahead, and on a 4-core machine the difference is 3–4×.

### 3.1 GPUDirect Storage path

Because chunk offsets are known from the index before any parse, and chunks are
page-aligned, a DMA descriptor list can be built and handed to `cuFileRead`:

```
NVMe → (DMA) → GPU HBM
```

skipping host RAM entirely. Modeled: 140 GiB at ~28 GB/s effective GDS
throughput ≈ 5 s, with host CPU essentially idle. Verification in this mode must
happen on-GPU or be skipped; OMNI supports per-chunk verification on the GPU
(BLAKE3 parallelizes well on GPUs) or `V0` + signature-only trust.

**No existing model format can do this**, because none guarantees that a
tensor's bytes are contiguous, aligned, and locatable without parsing.

## 4 Random and partial access

| Task | OMNI | safetensors | GGUF | PyTorch |
|---|---|---|---|---|
| Read one 4096×4096 tensor from a 140 GB model | 1 index lookup + 8 chunk reads (32 MiB) | header parse + 1 read | sequential scan to find it, then read | read + unpickle everything |
| Read rows 100–163 of that tensor | 1 chunk (4 MiB) | 1 read of the row range (works: contiguous) | same as above | not possible |
| Read only layers 0–3 (pipeline stage) | ~8 % of bytes | possible with effort | possible with effort | no |
| Read only MoE experts 3, 17, 44 | ~5 % of bytes, if `row`/`blocklist` chunked | no (contiguous expert block) | no | no |
| Verify only what you read | yes (Bao) | no | no | no |

The MoE row is the interesting one. With `chunker: row` on expert weights, a
runtime that routes to 8 of 64 experts fetches ~12.5 % of the MLP weights. For a
Mixtral-class model that is the difference between 87 GB and 15 GB of transfer
per node.

## 5 The dedup and delta economics

This is where OMNI's gains are order-of-magnitude rather than percentage.

### 5.1 Modeled scenario: a model family on a hub

One 8 B base model, published as:
- fp16 base, 5 fine-tunes (full-parameter), 20 LoRA adapters,
- each of the 6 full models also in int8, int4-GPTQ, int4-AWQ, MXFP4 quantizations,
- GGUF Q4_K_M and Q8_0 for each of the 6.

**Status quo (independent artifacts):**

| Artifact class | Count | Each | Total |
|---|---:|---:|---:|
| fp16 models | 6 | 16.1 GB | 96.6 GB |
| int8 | 6 | 8.0 GB | 48.0 GB |
| int4 (GPTQ) | 6 | 4.5 GB | 27.0 GB |
| int4 (AWQ) | 6 | 4.5 GB | 27.0 GB |
| MXFP4 | 6 | 4.3 GB | 25.8 GB |
| GGUF Q4_K_M | 6 | 4.9 GB | 29.4 GB |
| GGUF Q8_0 | 6 | 8.5 GB | 51.0 GB |
| LoRA adapters | 20 | 0.17 GB | 3.4 GB |
| **Total** | | | **308.2 GB** |

**OMNI**, with quantization-as-transformation and deltas:

| Stored | Bytes |
|---|---:|
| base fp16 chunks | 16.1 GB |
| 5 full fine-tunes as quantized-residual deltas (int8 residual, ~6× smaller) | 13.4 GB |
| int4-GPTQ params (packed weights + scales + zeros; the only new bytes) | 4.5 GB × 6 = 27.0 GB |
| int4-AWQ params | 27.0 GB |
| MXFP4 params | 25.8 GB |
| int8: *derived from fp16*, stored as expression only | ~0 |
| GGUF Q8_0: derived from int8 expression, opaque cache optional | 0 (or 51.0 GB if cached) |
| GGUF Q4_K_M: structural = the GPTQ/AWQ family? No — distinct scheme, stored | 29.4 GB |
| 20 LoRAs | 3.4 GB |
| **Total (no opaque caches)** | **142.2 GB** (−54 %) |
| **Total (fp16 + int4-AWQ only, others derived on demand)** | **~60 GB** (−81 %) |

Two honest observations:

1. **Quantization schemes that are genuinely different bit-layouts still cost
   their bytes.** GPTQ, AWQ and MXFP4 weights are different numbers, not
   different views of the same numbers. OMNI does not magic that away.
2. **The large wins come from (a) not storing derivable representations
   (int8/fp8 casts, merged adapters, GGUF forms of already-stored schemes), and
   (b) delta-encoding fine-tunes.** Those are exactly the artifacts that
   currently dominate hub storage.

### 5.2 Fine-tune delta sizes (modeled)

For an 8 B model, 16.1 GB in fp16:

| Fine-tune type | Delta representation | Expected size | Ratio |
|---|---|---:|---:|
| LoRA r=16 on attention | low-rank, exact | 0.17 GB | 1.1 % |
| LoRA r=64 all-linear | low-rank, exact | 0.85 GB | 5.3 % |
| Full SFT, 1 epoch, small LR | int8 quantized residual | 2.2 GB | 13.7 % |
| Full SFT, large LR | int8 quantized residual | 3.1 GB | 19.3 % |
| Continued pretraining, frozen embeddings | chunk-level + residual | 12.0 GB | 74.5 % |
| Merge of 3 existing models | expression only | ~0 GB | 0 % |

The merge row is worth emphasizing: model merges — an enormous fraction of
published model artifacts today — become **recipes** with essentially zero
storage, and become reproducible in the process.

### 5.3 Chunk dedup across versions

Modeled with 4 MiB fixed chunking:

| Change | Chunks changed |
|---|---|
| Metadata edit (license, README) | 0 weight chunks |
| Tokenizer extended by 100 tokens | embedding + lm_head rows only (~0.4 %) |
| One layer retrained | that layer's chunks (~3 %) |
| Full fine-tune | ~100 % (every weight changes) |
| Requantization of the same weights | 100 % of the new scheme's bytes, 0 % of the old |

Fixed chunking gives near-zero dedup for full fine-tunes; this is a property of
the data, not the chunker. `cdc-gear` does not help either (the change is
everywhere, not shifted). The residual-delta path is the answer, and OMNI's
contribution is that it is *expressible in the format* rather than requiring a
bespoke pipeline.

## 6 Streaming and time-to-first-token

Modeled: 8 B model, int4 (4.5 GB), 200 Mbit/s link (25 MB/s).

| Milestone | Bytes | Time | Status quo (GGUF) |
|---|---:|---:|---|
| Model identified, plan made | 0.2 MB | 0.01 s | must download header (up to 50 MB) |
| Tokenizer ready | 3.4 MB | 0.14 s | after header |
| Layer 0 ready | 145 MB | 5.8 s | — |
| All layers ready | 4.5 GB | 180 s | 180 s |
| **First token possible** (layer-wise streaming execution) | ~145 MB | **≈6 s** | **180 s** |

The 30× improvement in perceived latency comes from three format properties:
declared load order, per-object verification, and the ability to start executing
layer 0 before layer 31 exists. None of them is available in a format whose
metadata is at the front and whose bytes are an undifferentiated blob.

## 7 Index and metadata overhead

| Model | Tensors | Chunks (4 MiB) | Index bytes | Structure bytes | Overhead |
|---|---:|---:|---:|---:|---:|
| 1 B fp16 (2 GB) | 150 | 512 | 42 KiB | ~90 KiB | 0.006 % |
| 8 B fp16 (16 GB) | 291 | 4 096 | 280 KiB | ~180 KiB | 0.003 % |
| 70 B fp16 (140 GB) | 723 | 35 840 | 2.3 MiB | ~450 KiB | 0.002 % |
| 671 B MoE fp8 (671 GB) | 12 000 | 171 776 | 10.5 MiB | ~6 MiB | 0.002 % |
| 70 B + Adam fp32 (1.7 TB) | 2 169 | 435 200 | 26.6 MiB | ~1.3 MiB | 0.002 % |

Padding overhead at 4 KiB alignment: worst case `4 KiB × n_objects`; for the 671 B
model that is 671 MiB on 671 GB = 0.1 %. At 64 KiB alignment it would be 1.6 %,
which is why 4 KiB is the default and 64 KiB is opt-in for object stores.

## 8 CPU costs

| Operation | Model | Cost |
|---|---|---|
| Index lookup | bucket + binary search | ~3 cache misses, <100 ns |
| CBOR parse of a TensorDesc | ~400 bytes | ~1 µs |
| Parse full structure for a 291-tensor model | ~180 KiB | ~2 ms |
| Plan resolution | O(tensors × candidates) | ~1 ms |
| BLAKE3 verify | 12 GB/s (16 cores) | I/O-comparable |
| int4 → bf16 dequantize | ~4 GB/s/core SIMD | 4.1 s/16 GB on 1 core, 0.3 s on 16 |
| LoRA merge (r=16, 4096²) | 2·4096·4096·16 FLOP/tensor | ~0.5 ms/tensor on CPU |

Materialization of an int4→bf16 model is the main new cost OMNI can introduce
relative to "the bytes are already bf16". It is 0.3–4 s for an 8 B model, paid
once and then cached (§10.6). The plan explicitly reports it (`est_load_ms`), and
`min-load-time` as an objective will prefer a direct-map representation when one
exists.

## 9 Where OMNI is *slower* than the alternatives

Stated plainly, because a proposal that claims to win everywhere is not credible:

1. **First load of a quantized-to-different-target model** pays materialization
   that a pre-materialized file does not. Mitigation: caches, `realizations[]`,
   `min-load-time` objective. Cost: seconds, once.
2. **Structure parsing is more work than safetensors' single JSON header** for
   tiny models. For a 10 MB model with 5 tensors, OMNI reads ~10 KiB of structure
   where safetensors reads ~500 bytes. Absolute difference: microseconds.
   Irrelevant, but real.
3. **Many small objects** cost index space and per-object overhead. A model with
   1 M tiny tensors (some GNN and recommender workloads) would have a 64 MB
   index. Mitigation: `ShardedMap`, grouping small tensors into one chunk with
   `slice` expressions. Design guidance: do not chunk below 64 KiB.
4. **Writing is slower than `torch.save`** because everything is hashed. BLAKE3
   at 12 GB/s makes this ~1.3 s per 16 GB, plus chunking. Acceptable; and
   `torch.save` was never the bottleneck in a training run.
5. **Deep parent chains** add one index lookup per level and, if parents are
   remote, one network round trip each. Mitigation: `omni flatten` for
   distribution.

## 10 Benchmark plan (to replace the models above with measurements)

The roadmap's Phase 3 gate is a published benchmark suite covering:

- cold/warm load of 1 B / 8 B / 70 B / 671 B models from NVMe, network FS, S3, HTTP;
- comparison against safetensors, GGUF, ONNX, `.pt` on identical hardware;
- GPUDirect path vs. host-staged;
- streaming TTFT over throttled links (10/100/1000 Mbit/s);
- dedup ratios measured over a real hub corpus snapshot;
- delta sizes over real fine-tune pairs;
- verification throughput across hash algorithms and core counts;
- p50/p99 index lookup latency at 10⁴/10⁵/10⁶ objects;
- writer determinism and throughput.

Results are to be published as OMNI containers with signed `Evaluation` objects
(§06.8) — the format should eat its own dog food for its own benchmarks.

---

**See also:** [Comparison](comparison.md) · [Roadmap](roadmap.md) · [§13 Streaming](../spec/13-streaming.md)
