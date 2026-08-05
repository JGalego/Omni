# Import / Export Architecture

OMNI's value depends entirely on whether it can absorb the existing world without
losing anything, and emit into it without lying about what was lost.

## 1 The two contracts

### 1.1 Importer contract

```rust
trait Importer {
    fn probe(&self, src: &Source) -> Option<Confidence>;
    fn import(&self, src: &Source, opts: &ImportOpts)
        -> Result<(ObjectGraph, FidelityReport)>;
}
```

Normative rules:

- **I1 — Never fabricate.** If the source does not state the parameter count,
  the license, the RoPE variant or the context length, the corresponding field
  is **absent**. Not zero, not "unknown", not a guess. Absence is information.
- **I2 — Preserve the unrepresentable.** Anything the importer cannot model
  structurally is preserved verbatim as a `Foreign` object with its source path,
  byte offset and digest, so that a future importer (or a human) can recover it.
- **I3 — Report fidelity.** Every import produces a machine-readable
  `FidelityReport`, attached to the container as a `Provenance` object.
- **I4 — Verify what you claim.** If the importer converts an opaque block format
  into a structural representation, it MUST verify by re-materializing a sample
  and comparing bit-exactly, and MUST record the result.
- **I5 — Sandbox unsafe sources.** Pickle and any other code-bearing format is
  parsed under the restrictions in §12.10.
- **I6 — Record the source digest**, so "which file did this come from?" is
  always answerable.

### 1.2 Exporter contract

```rust
trait Exporter {
    fn plan(&self, model: &Model, target: &Target) -> ExportPlan;   // no I/O
    fn export(&self, model: &Model, plan: &ExportPlan, sink: &mut Sink)
        -> Result<LossReport>;
}
```

- **E1 — Plan before writing.** `plan()` computes the loss report *without
  producing bytes*, so tools can refuse or ask first.
- **E2 — Lossy exports require consent.** If `LossReport` is non-empty and
  `--allow-lossy` is not given, `export` fails. No silent degradation, ever.
- **E3 — Emit the loss report** alongside the artifact (`model.gguf.loss.json`)
  and, where the target format supports metadata, embed a pointer to the OMNI
  source digest inside the exported file.
- **E4 — Round-trip identity.** For formats OMNI can represent losslessly,
  `import(export(m)) == m` at the canonical-digest level. This is a conformance
  test (§15.3.1 `roundtrip/`).

## 2 The fidelity report

```cbor-diag
{ "t":"omni.prov/import", "v":1,
  "source": {"format":"gguf","version":3,"path":"llama3-8b-Q4_K_M.gguf",
             "digest":h'…',"size":4920512512},
  "importer":{"name":"omni-import-gguf","version":"1.0.0"},
  "lossless": false,
  "represented":  ["tensors","quantization","tokenizer","chat_template",
                   "arch_params","rope","generation_defaults"],
  "unrepresented":[
     {"item":"general.source.huggingface.repository",
      "reason":"free-form key with no OMNI schema",
      "action":"preserved in ext:org.ggml/kv"} ],
  "assumptions": [
     {"item":"license", "reason":"GGUF file declares none",
      "action":"field omitted"} ],
  "verification": {"method":"sample-dequant","samples":4096,
                   "bit_exact":true,"blocks_checked":128},
  "warnings": ["tokenizer merges reconstructed from vocab ordering; \
                conformance vectors generated from source tokenizer: PASS (512/512)"] }
```

The `assumptions` list is the honest counterpart of "importers should preserve
everything": it records what the importer *chose not to invent*.

## 3 Per-format capability matrix

**Legend:** ● full · ◐ partial · ○ none · — not applicable

| Source | Weights | Dtypes | Quant | Graph | Tokenizer | Chat tmpl | Metadata | Training | Adapters | Sigs | Round-trip |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **safetensors** | ● | ● | ◐¹ | ○ | —² | — | ◐³ | ◐⁴ | —⁵ | ○ | ● lossless |
| **PyTorch `.pt`/`.bin`** | ● | ● | ◐¹ | ○⁶ | — | — | ◐ | ● | ◐ | ○ | ◐ (pickle side effects dropped) |
| **PyTorch DCP** | ● | ● | ◐ | ○ | — | — | ◐ | ● | — | ○ | ● |
| **GGUF** | ● | ● | ● | ○⁷ | ● | ● | ● | ○ | ◐⁸ | ○ | ● lossless (structural or opaque) |
| **GGML (legacy)** | ● | ◐ | ● | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ | ◐ |
| **ONNX** | ● | ● | ◐⁹ | ● | ◐¹⁰ | ○ | ◐ | ○ | ○ | ◐ | ◐ (opset semantics preserved as dialect) |
| **TensorFlow SavedModel** | ● | ● | ◐ | ● | ◐ | ○ | ◐ | ◐ | ○ | ○ | ◐ |
| **Keras `.keras`** | ● | ● | ○ | ● | — | — | ◐ | ◐ | ○ | ○ | ◐ |
| **Flax / JAX (msgpack, Orbax)** | ● | ● | ◐ | ○¹¹ | — | — | ◐ | ● | ◐ | ○ | ● |
| **TensorRT engine** | ◐¹² | ◐ | ◐ | ○¹² | — | — | ◐ | ○ | ○ | ○ | ○ (opaque cache only) |
| **CoreML `.mlpackage`** | ● | ● | ◐ | ● | ◐ | ○ | ◐ | ○ | ○ | ◐ | ◐ |
| **OpenVINO IR** | ● | ● | ◐ | ● | ○ | ○ | ◐ | ○ | ○ | ○ | ◐ |
| **TFLite** | ● | ● | ● | ● | ◐ | ○ | ◐ | ○ | ○ | ○ | ◐ |
| **ExecuTorch `.pte`** | ● | ● | ● | ● | ○ | ○ | ◐ | ○ | ○ | ○ | ◐ |
| **MLX** | ● | ● | ● | ○ | ◐ | ○ | ◐ | ◐ | ● | ○ | ● |
| **GPTQ (HF)** | ● | ● | ● | ○ | — | — | ◐ | ○ | — | ○ | ● |
| **AWQ (HF)** | ● | ● | ● | ○ | — | — | ◐ | ○ | — | ○ | ● |
| **EXL2** | ● | ● | ● | ○ | ◐ | ○ | ◐ | ○ | ◐ | ○ | ● |
| **bitsandbytes NF4/INT8** | ● | ● | ● | ○ | — | — | ◐ | ● | ● | ○ | ● |
| **PEFT LoRA/DoRA** | ● | ● | — | ○ | — | — | ● | ◐ | ● | ○ | ● lossless |
| **Ollama bundle** | ● | ● | ● | ○ | ● | ● | ● | ○ | ◐ | ◐¹³ | ● |
| **Hugging Face repo** | ● | ● | ● | ○ | ● | ● | ● | ◐ | ● | ◐ | ● (as a bundle) |
| **NeMo `.nemo`** | ● | ● | ◐ | ○ | ● | ◐ | ● | ● | ◐ | ○ | ◐ |
| **Megatron dist ckpt** | ● | ● | ◐ | ○ | ◐ | ○ | ◐ | ● | ○ | ○ | ● |
| **HDF5 / Zarr / NPZ** | ● | ● | ○ | ○ | — | — | ◐ | ◐ | ○ | ○ | ● |

Notes:
¹ only if quantization params are present as tensors + a config;
² tokenizer lives in a sibling `tokenizer.json`, imported as part of a repo;
³ safetensors' `__metadata__` is a flat string→string map;
⁴ optimizer states appear only if the writer put them there;
⁵ PEFT LoRA *is* safetensors + a config; imported as an `Adapter`;
⁶ TorchScript archives carry a graph, imported as a `Foreign` blob plus best-effort IR;
⁷ GGUF has no graph: architecture is an enum. OMNI records `arch.family` + params and can synthesize a graph (§07.5);
⁸ GGUF LoRA files exist as a separate convention;
⁹ ONNX QDQ nodes map onto `quantize`/`dequantize` cleanly; other quant conventions are opset-specific;
¹⁰ some ONNX models embed tokenizers as custom ops;
¹¹ JAX graphs are traced, not stored; `jaxpr`/StableHLO export imports as IR when present;
¹² TensorRT engines are compiled artifacts: weights are baked in and layer fusion is irreversible. Imported as an opaque `RuntimeCache` with `executable: true`, never as a canonical model;
¹³ Ollama uses OCI-ish manifests with digests.

## 4 Notable import paths in detail

### 4.1 Hugging Face repository → OMNI bundle

The most important path, since it is where most models live.

```
$ omni import hf://meta-llama/Llama-3.1-8B-Instruct -o llama31-8b.omni
```

| Repo file | Becomes |
|---|---|
| `model-*.safetensors` + `.index.json` | `TensorTable` + `ChunkList`s; **zero-copy**: chunk boundaries aligned to the safetensors payload so bytes are copied once, or referenced in place with `--link` |
| `config.json` | `meta.arch.params`, `context`, `rope` (with `interleaved` resolved from the architecture, and recorded as an *assumption* if inferred) |
| `generation_config.json` | `meta.generation` |
| `tokenizer.json` | `Tokenizer` IR (structural, not opaque) + conformance vectors generated by running the source tokenizer |
| `tokenizer_config.json` `chat_template` | `ChatTemplate` translated to OMNI-CT, with the Jinja2 source retained in `jinja_compat` |
| `special_tokens_map.json` | merged into `Tokenizer.added_tokens` |
| `README.md` (model card) | `Blob` + parsed YAML front-matter → `meta.license`, `meta.tags`, `meta.evaluations` (marked `self_reported`) |
| `LICENSE` | `Blob` referenced by `meta.license.text` |
| `*.py` (custom code) | `Foreign` blob, **never executed**, flagged prominently |
| `adapter_model.safetensors` + `adapter_config.json` | `Adapter` with `base` pinned by digest |

Sharded safetensors files are a particular win: OMNI's chunking makes shard
boundaries irrelevant, so the perpetual "shard 3 of 7 failed to download" problem
becomes a per-chunk retry.

### 4.2 GGUF → OMNI (dual representation)

```
$ omni import model-Q4_K_M.gguf -o model.omni --keep-opaque
```

Produces:
- canonical structural tensors (§05.2.4) — transparent, re-quantizable,
  composable with adapters;
- an opaque `RuntimeCache` holding the original block bytes, keyed by the
  structural expression's digest — so `llama.cpp` can `mmap` and run with **zero
  conversion**;
- verified equivalence between the two, recorded in the fidelity report.

This is the pattern for every legacy format: *be transparent, and also be fast
for the incumbent.*

### 4.3 Distributed checkpoint → OMNI

```
$ omni import --dcp ./checkpoints/step-128000 -o ckpt.omni --ranks 512
```

Ranks import in parallel, each writing its own `OBJ`/`BLOB` segments to a shared
directory store; a coordinator writes the `ShardMap` and superblock. Replicated
tensors across data-parallel ranks deduplicate automatically (§09.4) — typically
removing 30–60 % of a ZeRO-1 checkpoint's bytes with no configuration.

## 5 Export

### 5.1 Loss report

```
$ omni export model.omni --gguf -o model.gguf --quant q4_k_m
EXPORT PLAN → GGUF v3
  ✓ tensors            291 → 291
  ✓ quantization       affine-int4 g128 → Q4_K (bit-exact)
  ✓ tokenizer          bpe → gguf tokenizer (vectors: 512/512 PASS)
  ✓ chat template      omni-ct → jinja2 (vectors: 24/24 PASS)
  ✓ arch params        transformer.decoder → llama
  ⚠ LOSS: provenance chain (4 attestations)     — GGUF has no representation
  ⚠ LOSS: signatures (2)                        — GGUF has no representation
  ⚠ LOSS: evaluation results (11 benchmarks)    — GGUF has no representation
  ⚠ LOSS: adapter `medical-v3` (not merged)     — use --merge-adapters
  ⚠ LOSS: dataset descriptors, license text     — partially: license SPDX kept
  ⚠ LOSS: tensor axes names, materialization hints
  ⚠ DEGRADE: f32 → f16 for 3 norm tensors (max rel-err 4.1e-8)
refusing to write without --allow-lossy
```

Every one of those lines is information a user currently loses silently.

### 5.2 Export targets

| Target | Notes |
|---|---|
| **safetensors** | materialize each tensor's chosen representation; emit `.index.json` for shards; `__metadata__` gets the OMNI canonical digest so the lineage survives one hop |
| **GGUF** | map arch → enum (fails loudly for unmapped architectures), quantization → block type; can emit the opaque cache directly if present (zero-cost) |
| **ONNX** | requires a graph at `primitive` level; emits QDQ for quantized tensors; unmapped ops fail with a precise list rather than a generic error |
| **TensorRT** | build via TensorRT from the ONNX/graph path; result is stored back as a `RuntimeCache`, not as "the model" |
| **CoreML / MLX / OpenVINO / TFLite / ExecuTorch** | graph translation + weight materialization; each has a documented op-coverage table |
| **PyTorch** | `state_dict` of materialized tensors + a `config.json`; **never emits pickle by default** (writes safetensors and a loader stub) |
| **HF repo layout** | the inverse of §4.1 — the practical "publish this" path |

### 5.3 Lossless round-trip guarantees

Formats for which `import → export → import` is canonical-digest identical:
safetensors, PEFT adapters, GGUF (with `--keep-opaque`), GPTQ, AWQ, EXL2, MLX,
NPZ, Zarr, and PyTorch DCP. Each has a `roundtrip/` conformance case.

Formats where round-trip is explicitly *not* guaranteed, and why:
- **TensorRT / compiled engines**: fusion is irreversible.
- **ONNX**: opset lowering loses semantic intent; the reverse direction
  reconstructs `primitive` level only.
- **Pickle checkpoints**: arbitrary Python objects have no representation and are
  intentionally dropped.

## 6 Conversion as a first-class operation

```
$ omni convert model.omni --requantize awq:4:128 --calib c4:512 -o model-awq.omni
```

Conversion inside OMNI does not produce a new file's worth of bytes: it adds new
tensors expressed over the existing ones and a new manifest. The result is a
*delta*, and the provenance records the recipe (§05.5). A hub hosting fp16, int8,
int4-GPTQ, int4-AWQ and MXFP4 variants of a model stores the base once plus five
small manifests and the quantization parameters — instead of five full copies.

## 6.1 What the reference implementation has

safetensors, both directions, with the contracts above implemented rather than
described:

```console
$ omni import safetensors model.safetensors -o model.omni
$ omni export safetensors model.omni --plan            # E1: what would be lost
$ omni export safetensors model.omni -o out.safetensors --allow-lossy
```

The importer verifies every tensor against the source before it claims to have
copied it (I4), records the source digest (I6), omits every field safetensors does
not state instead of guessing it (I1), preserves `__metadata__` keys with no OMNI
schema in a `Foreign` object (I2) — and the exporter puts them back — and attaches
the fidelity report as a `Provenance` object (I3). `--plan` computes the loss
report without writing bytes (E1); a lossy export without `--allow-lossy` writes
nothing at all (E2); the report is written to `<out>.loss.json` (E3); and CI
checks E4 against a fixture built from the format's own definition, comparing
every tensor's dtype, shape and bytes, and the tensor object digests either side
of the round trip.

One detail worth naming, because it is the kind of thing that quietly corrupts a
mask: safetensors stores a boolean in a whole byte, while §04.3 gives `bool` one
bit. The importer keeps the dtype `bool` and describes the storage with §04.4's
`packed` layout — one element per 8-bit word — rather than importing masks as
`u8`. The type stays true and the bytes round-trip.

PEFT LoRA is the second row, and it is where the *digest* half of the object
model earns its keep:

```console
$ omni import peft ./lora --base model.omni -o lora.omni
$ omni adapter check model.omni lora.omni
```

`--base` is required rather than convenient. An `adapter_config.json` names its
base with a string; §08.1 pins it with a digest, so an adapter can never silently
attach to a different base. There is no honest way to synthesize the digest of a
model you were handed the *name* of, so the base container is an argument and
PEFT's name for it is kept as a name. `use_dora`, `fan_in_fan_out`, `use_rslora`,
`rank_pattern`, `alpha_pattern`, `modules_to_save` and any `peft_type` but `LORA`
each change what the update is, and each is refused by name.

One detail is worth recording because it took a real attach to find: the R-A03
rank-axis requirement is written **only** when the base names its axes. A base
imported from safetensors names none — safetensors says nothing about what a
dimension means — so asserting the requirement made every attach report as
*invalid* rather than merely unchecked. The shapes are still checked (R-A02),
which is what can actually be decided from the tensors.

GPTQ and AWQ are the third and fourth rows, and they are the ones that test §05's
central claim rather than the object model's:

```console
$ omni import gptq ./model-gptq -o model.omni
$ omni import awq  ./model-awq  -o model.omni
```

Quantization is a transformation, not a file type, so nothing is converted. The
packed 32-bit words go into the container unchanged and each layer's weight
becomes an expression over them:

```
weight = permute(dequantize(reshape(permute(qweight)), {affine-sub, block:[gs,1],
                                                       scale, zero}))
```

The int4-in-int32 packing is §04.4's `packed` layout. AWQ's GEMM interleave — its
kernel wants a word's columns in the order `0 2 4 6 1 3 5 7` — is a `gather`.
GPTQ's act-order is a `gather` too, over the scale and zero tensors rather than
over the weight, so the stored bytes are never rearranged. None of it is a special
case in the evaluator, which is what §05.2 claims and what this checks.

Two things are worth recording, because both are the kind of thing that produces a
container full of plausible wrong numbers rather than an error.

**Byte identity is not enough for a quantized import.** The packed words are
copied verbatim, so comparing them proves they were copied and says nothing about
whether they are being *read* correctly. So I4 here is two checks: byte identity,
and every layer dequantized through the expression graph and compared against
scalar code that shares nothing with the evaluator. The fidelity report says
`byte-identity + sample-dequant` and carries the element count. CI goes one step
further and compares against arithmetic done in Python on a fixture packed by the
formats' own rules, so the mapping is checked against GPTQ and AWQ rather than
against OMNI.

**The zero-point convention is the corruption §05.1 was written for.** §05.1 makes
`formula` a closed enumeration because *whether the zero point is subtracted
before or after scaling is a recurring source of silent corruption when converting
between GPTQ, AWQ and GGUF*. AutoGPTQ's original checkpoint format stores every
zero point one *less* than its true value and adds one back in
`QuantLinear.forward`; `checkpoint_format: "gptq_v2"` dropped that. The two differ
by exactly one quantization step in every weight and nothing in the tensors
distinguishes them. So the offset is read from `checkpoint_format`, written as an
explicit `+1` node in the expression rather than folded into a constant, and named
in the report — and an unrecognised `checkpoint_format` is **refused**, because
that is precisely the case where a guess corrupts the whole model. 3-bit GPTQ is
refused for a different reason: its values straddle the 32-bit word boundary, so
it is not a `packed` layout at all. AWQ's `gemv` and `marlin` versions interleave
differently from `gemm` and are refused by name.

Export back to GPTQ and AWQ does not exist yet, so §5.3's lossless round-trip
claim for these two rows is *not* demonstrated. What is demonstrated is that the
import is lossless: every source tensor is in the container, byte for byte, except
an ascending `g_idx`, which is checked to be `i / group_size` and left out because
`group_size` already says it.

Every other row of the matrix in §3 is unimplemented. A request to import one is
refused by name, with a pointer to this document, rather than half-attempted.

## 7 Implementation topology

```
omni-core         object model, container, CBOR, digests, expressions   (no I/O deps)
omni-io           stores: file, dir, http, oci, memory
omni-import-*     one crate per source format (feature-gated)
omni-export-*     one crate per target format
omni-convert      quantizers, delta extraction, merge algorithms
omni-cli          the `omni` binary
```

Importers and exporters are **out-of-tree-capable plugins**: the CLI discovers
`omni-import-<fmt>` binaries on `PATH` speaking a documented JSON-lines protocol,
so a vendor can ship an importer for a proprietary format without patching OMNI.
This is what keeps the core small and the ecosystem open.

**See also:** [CLI](cli.md) · [SDK](sdk.md) · [§05 Quantization](../spec/05-quantization.md)
