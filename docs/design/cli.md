# `omni` — CLI Specification

One binary. Verbs that compose. Every command that reads a model works on a
local file, a directory store, an HTTP URL, or an OCI reference, because they are
all stores (§01.8).

```
omni <verb> [target] [flags]
```

Global flags: `--json` (machine-readable output on every verb), `--quiet`,
`--verbose`, `--store <path|url>` (additional stores to resolve refs from),
`--verify <level>` (V0–V8, default V3-selective), `--policy <file>`,
`--offline`, `--jobs N`, `--color`.

## 1 Inspection

### `omni inspect`

The flagship. **Reads no tensor payload.**

```
$ omni inspect llama31-8b-instruct.omni
llama31-8b-instruct.omni                                          143.2 GiB
  container   OMNI/1.0  profile=core  align=4096  hash=blake3-256  sealed
  canonical   b3:9c1f4a2e8d5b7c03…                        (identity of the model)
  root        b3:2f8a1c9e…                                (this file)

model  acme/llama-3.1-8b-instruct  v2026.08.1
  architecture   transformer.decoder            dialect omni.nn@1
  parameters     8,030,261,248  (8.03 B)        active/token 8.03 B
  layers 32   hidden 4096   heads 32/8 (gqa)    ffn 14336 (silu)
  context        trained 8192 · max 131072 (yarn ×8.0) · sliding none
  rope           theta 500000  dims 128  interleaved=false
  modality       text → text
  license        Llama-3.1-Community  (text embedded, 12.4 KiB)

tensors        291                                            140.1 GiB
  bf16                   291 tensors   140.1 GiB
  effective bits/param   16.00
  largest        model.embed_tokens.weight   [128256, 4096]   1.00 GiB
  chunking       fixed 4 MiB · 35,842 chunks · 0 shared with other models

graph          none (weights-only)
  portability  requires a runtime with built-in transformer.decoder support
               `omni graph synthesize` can generate OMNI-IR

tokenizer      bpe · vocab 128,256 · byte-fallback · 512 conformance vectors ✓
chat template  omni-ct/1 · caps: system, tools · 24 vectors ✓
generation     temp 0.6 · top_p 0.9 · eos [128001, 128009]

adapters       none
parents        none
plugins        none
realizations   3   (min-memory/int4, balanced/bf16, min-latency/f8e4m3)
caches         none
extensions     org.acme/deploy (non-critical, 1.1 KiB)

integrity      ✓ index consistent (V1)   ✓ 47 structure objects verified (V3)
signatures     ✓ EdDSA  release-bot@acme.com  (sigstore, rekor #88213741)
               ✓ ML-DSA-65  acme-release-2026  (hybrid)
provenance     4-step chain → base acme/llama-3.1-8b  ✓ verified
evaluations    11 benchmarks (self-reported)   mmlu 0.6821 · gsm8k 0.8412

read: 2 requests · 218 KiB · 41 ms
```

The last line is the point: everything above came from ~200 KiB.

Sub-modes: `--tensors` (full table), `--tensors=<glob>`, `--graph`, `--tokenizer`,
`--quant`, `--provenance`, `--caches`, `--ext`, `--raw <digest>`.

### `omni ls` / `omni tree` / `omni cat` / `omni stat`

```
$ omni ls model.omni --tensors --sort=size --limit 5
NAME                                 SHAPE          DTYPE  LAYOUT  BYTES     VALUE
model.embed_tokens.weight            [128256,4096]  bf16   strided 1.00 GiB  literal
lm_head.weight                       [128256,4096]  bf16   strided 1.00 GiB  literal
model.layers.0.mlp.gate_proj.weight  [14336,4096]   bf16   strided 112 MiB   literal
…

$ omni tree model.omni --objects --depth 2
Manifest b3:2f8a…
├── Metadata b3:71cd…                            4.1 KiB
├── assets/text → Model b3:5a09…
│   ├── tensors → TensorTable b3:c4e1…         182.3 KiB
│   └── tokenizer → Tokenizer b3:9182…           3.2 MiB
└── attestations → Signature b3:aa41…             312 B

$ omni cat model.omni --tensor model.layers.0.attn.q_proj.weight \
      --format npy --slice '[0:4, 0:8]'
$ omni stat model.omni --digest b3:c4e1…
```

### `omni diff`

```
$ omni diff base.omni tuned.omni
metadata
  ~ version           2026.07.2 → 2026.08.1
  + evaluations       +3 benchmarks
tensors  291 compared
  = identical         96                    0 B
  ~ changed          195              4.31 GiB changed content
      max |Δ|  0.0391   mean |Δ|  2.1e-4   rank(Δ) ≤ 32 for 160 tensors
  + added              0
  - removed            0
tokenizer  identical
graph      identical
storage    2.9 GiB of chunks shared (dedup 12.4 %)
suggestion  `omni delta` would produce ≈1.06 GiB (0.74 % of base)
```

## 2 Verification and security

```
$ omni verify model.omni --level V5
$ omni verify model.omni --level V8 --policy corp-policy.cbor
$ omni verify model.omni --recompute            # V6: check every derived object
$ omni verify model.omni --tokenizer --template --numeric

$ omni sign model.omni --key ed25519.key --purpose release
$ omni sign model.omni --keyless --oidc github   # sigstore
$ omni sign model.omni --key pq.key --alg ml-dsa-65 --hybrid

$ omni provenance model.omni --tree
acme/llama-3.1-8b-instruct  b3:9c1f…
└── fine-tune  step 4000  dataset acme/sft-mix-v7 (b3:31ab…)  ✓ attested
    └── acme/llama-3.1-8b-base  b3:77de…
        └── pretrain  15.0 T tokens  dataset acme/pretrain-v3  ✓ attested
            builder: acme.com/trainers/cluster-3  (SLSA L3)  ✓

$ omni policy check model.omni --policy corp.cbor
  ✗ require_signature_by: acme-release-*     → satisfied
  ✗ deny_executable_caches                   → satisfied (0 present)
  ✗ max_license_restriction: permissive      → VIOLATION: Llama-3.1-Community
```

## 3 Building and packing

```
$ omni pack ./model.omnid -o model.omni \
      --profile core --align 4096 --codec zstd:3 \
      --chunk fixed:4Mi --strategy by-layer --reproducible

$ omni unpack model.omni -o ./model.omnid
$ omni repack model.omni --codec bitshuffle+zstd:5 -o smaller.omni   # digests unchanged
$ omni gc ./store.omnid --keep-roots roots.txt --keep-every 10 --dry-run
$ omni strip model.omni --training --caches --executable -o infer.omni
$ omni flatten model.omni --depth 1 -o standalone.omni
```

`omni repack` is worth noting: it changes storage codecs, chunking and packing
**without changing a single object digest or the canonical model identity**.

## 4 Import / export / convert

```
$ omni import hf://meta-llama/Llama-3.1-8B-Instruct -o llama.omni
$ omni import ./ckpt.safetensors --config ./config.json -o m.omni
$ omni import model-Q4_K_M.gguf -o m.omni --keep-opaque
$ omni import --dcp ./checkpoints/step-128000 -o ckpt.omni
$ omni import model.pt -o m.omni --sandbox strict     # restricted unpickler

$ omni export m.omni --safetensors -o ./out/
$ omni export m.omni --gguf --quant q4_k_m -o m.gguf --allow-lossy
$ omni export m.omni --onnx --level primitive -o m.onnx
$ omni export m.omni --mlx -o m-mlx/

$ omni convert m.omni --requantize gptq:4:128 --calib c4:512 -o m-gptq.omni
$ omni convert m.omni --cast f8e4m3 --except '*.norm.*,lm_head*' -o m-fp8.omni
```

## 5 Adapters, deltas, merges

```
$ omni adapter check base.omni lora.omni
  ✓ base digest matches
  ✓ 128/128 selectors matched
  ✓ shapes consistent (rank 16, α 32)

$ omni adapter apply base.omni lora.omni -o merged.omni          # expression, no copy
$ omni adapter apply base.omni lora.omni -o merged.omni --materialize

$ omni delta base.omni tuned.omni -o delta.omni --max-err 5e-3
$ omni merge a.omni b.omni c.omni --mode ties --weights 1,0.7,0.3 -o merged.omni
$ omni lineage merged.omni                    # reproducible recipe
```

## 6 Planning and running

```
$ omni caps -o my-runtime.cbor                # detect local capabilities
$ omni plan model.omni --caps my-runtime.cbor --objective min-memory
PLAN  b3:5d1a…   objective=min-memory   feasible ✓
  graph          primitive (lowered from semantic via 1 shipped rule)
  tensors        291
    direct-map   96   (zero-copy)                       2.10 GiB
    dequantize   195  (int4 g128 → bf16, on demand)     4.31 GiB resident
  adapters       medical-v3 (runtime-applied)
  resident       6.41 GiB     load (est) 3.2 s     quality Δ −0.004 MMLU
  warnings       lm_head kept bf16 (int8 exceeded --max-err)

$ omni bench model.omni --load --plan min-latency
$ omni mount model.omni /mnt/m         # FUSE: synthesized safetensors/tokenizer views
$ omni serve ./store.omnid --port 8080 # object server with range support
```

## 7 Distribution

```
$ omni push model.omni oci://ghcr.io/acme/llama31-8b:2026.08.1
  layers: 4 new (1.06 GiB), 141 reused (139.0 GiB) — 0.75 % uploaded

$ omni pull omni://acme/llama31-8b@b3:9c1f… -o ./store.omnid
$ omni fetch omni://acme/llama31-8b --tensors 'model.layers.0.*' -o partial.omni
$ omni index model.omni -o model.omni.idx
$ omni mirror ./store.omnid s3://bucket/models/ --packs 1Gi
```

## 8 Schema, registry, plugins

```
$ omni schema list                       # schemas used by this build
$ omni schema show omni.tensor/desc@1
$ omni schema validate obj.cbor --schema omni.meta/model@1
$ omni registry search dtype mxfp4
$ omni plugin list model.omni
$ omni plugin verify org.acme/moe@2      # runs the plugin's own test vectors
$ omni plugin run org.acme/moe --op moe.pack --in a.npy   # WASM reference execution
```

## 9 Repair and forensics

```
$ omni fsck model.omni
$ omni fsck model.omni --rebuild -o repaired.omni   # scan segments, rebuild index
$ omni dump model.omni --header --hex
$ omni dump model.omni --object b3:71cd… --diag     # CBOR diagnostic notation
$ omni migrate model.omni --to omni/1.4 -o new.omni
```

## 10 Output discipline

What makes the output trustworthy:

1. **`--json` on everything.** Every verb emits a stable, schema'd JSON document
   with `--json`; the human rendering is a view of it. No scraping.
2. **Distinguish unknown from absent from zero.** `omni inspect --json` emits
   `null` for unknown and omits nothing that was present. A missing license
   prints `license: (not stated)`, never `license: unknown` in a way that looks
   like a value.
3. **Exit codes are semantic.**

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | invalid: the file violates the specification |
| 2 | usage error |
| 3 | indeterminate: valid, but this build cannot fully verify or execute it (unknown critical extension, unsupported signature algorithm, missing parent) |
| 4 | policy refusal (unsigned, lossy without consent, executable payload) |
| 5 | incomplete: required objects not available in any store |
| 6 | infeasible: capability negotiation found no valid plan |

Code 3 is the one most tools get wrong, and the one that keeps ecosystems from
fragmenting (§14.4).

**See also:** [SDK](sdk.md) · [Import/Export](import-export.md) · [§15 Conformance](../spec/15-conformance.md)
