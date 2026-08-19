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
| **GGUF** | ● | ● | ● | ○⁷ | ◐¹⁴ | ● | ● | ○ | ◐⁸ | ○ | ● lossless (structural or opaque) |
| **GGML (legacy)** | ● | ◐ | ● | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ | ◐ |
| **ONNX** | ● | ● | ◐⁹ | ● | ◐¹⁰ | ○ | ◐ | ○ | ○ | ○¹⁵ | ◐ (opset semantics preserved as dialect) |
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
| **bitsandbytes NF4/INT8** | ● | ● | ● | ○ | — | — | ◐ | ● | ● | ○ | ● lossless¹⁷ |
| **PEFT LoRA/DoRA** | ● | ● | — | ○ | — | — | ● | ◐ | ● | ○ | ● lossless |
| **Ollama bundle** | ● | ● | ● | ○ | ● | ● | ● | ○ | ◐ | ◐¹³ | ● |
| **Hugging Face repo** | ● | ● | ● | ○ | ● | ● | ● | ◐ | ● | ◐ | ● (as a bundle) |
| **NeMo `.nemo`** | ● | ● | ◐ | ○ | ● | ◐ | ● | ● | ◐ | ○ | ◐ |
| **Megatron dist ckpt** | ● | ● | ◐ | ○ | ◐ | ○ | ◐ | ● | ○ | ○ | ● |
| **HDF5 / Zarr / NPZ** | ● | ●¹⁶ | ○ | ○ | — | — | ◐ | ◐ | ○ | ○ | ● lossless (NPZ) |

Notes:
¹ only if quantization params are present as tensors + a config;
² tokenizer lives in a sibling `tokenizer.json`, imported as part of a repo;
³ safetensors' `__metadata__` is a flat string→string map;
⁴ optimizer states appear only if the writer put them there;
⁵ PEFT LoRA *is* safetensors + a config; imported as an `Adapter`;
⁶ TorchScript archives carry a graph, imported as a `Foreign` blob plus best-effort IR;
⁷ GGUF has no graph: architecture is an enum. OMNI records `arch.family` + params and can synthesize a graph (§07.5);
⁸ GGUF LoRA files exist as a separate convention;
⁹ ONNX QDQ nodes map onto `quantize`/`dequantize` cleanly *per tensor* — §05.1's closed `formula` enumeration is what makes `(x − zero_point) × scale` a mapping rather than a guess. Downgraded in practice for the per-axis and blocked forms, where saying which elements share a scale needs §05.1's block shape over an operand whose shape a dynamic graph does not fix, so those are carried rather than translated;
¹⁰ some ONNX models embed tokenizers as custom ops;
¹¹ JAX graphs are traced, not stored; `jaxpr`/StableHLO export imports as IR when present;
¹² TensorRT engines are compiled artifacts: weights are baked in and layer fusion is irreversible. Imported as an opaque `RuntimeCache` with `executable: true`, never as a canonical model;
¹³ Ollama uses OCI-ish manifests with digests;
¹⁴ downgraded from ● on evidence, when the importer was written: GGUF carries the vocabulary, the merges and the scores, but `tokenizer.ggml.pre` names a pre-tokenizer whose regexes live in llama.cpp's source rather than in the file, so the keys present do not determine where a token begins. What is importable is a decoder, not an encoder.
¹⁵ corrected from ◐ when the importer was written: ONNX has no signature field
at all. The ◐ was a guess about the format, and the format says nothing.
¹⁶ NPZ is implemented and HDF5 and Zarr are not, which the single row hides:
they are three formats with one thing in common. NumPy's dtypes map onto §04.3
exactly in both directions *except* that OMNI has narrower floats and sub-byte
integers than NumPy can spell — `bf16`, `f8e4m3`, `i4` — so exporting one is a
widening to the narrowest NumPy type that holds every value, named in the loss
report and refused without `--allow-lossy`, rather than a drop. Column-major
arrays keep their bytes and are described with §04.4's `strided` layout; a
big-endian array is *converted*, because §03.9 makes OMNI little-endian, and the
report says so. `descr: '|O'` is pickle and is refused under §12.10 clause 1.

¹⁷ implemented, and checked against bitsandbytes itself rather than against this
repository's reading of the format — the library that defines NF4 is a pip install
away, so it writes the fixture and says what the numbers are. NF4 and FP4 import
as a §05.4 codebook with a per-block scale, and double quantization needs no new
formula because a scheme's `scale` is an *expression*: the outer dequantize's
scale is an inner dequantize plus the stored offset. Blocking is over the
flattened tensor, so the dequantize is built one-dimensional and reshaped rather
than forced into an axis-aligned block shape that would only divide some tensors.
Agreement is bit-exact for single-quantized scales and for LLM.int8, and one
float32 ULP for double-quantized scales, where bitsandbytes rounds the
reconstructed scale to f32 and this evaluator does not. What a checkpoint cannot
give back is the f16/bf16 weights it was quantized *from*: they are not in the
file, and the fidelity report says so rather than implying a round trip.

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

**Export exists, and §5.3's lossless round-trip claim for these two rows is
demonstrated rather than asserted.** `omni export gptq|awq` writes the checkpoint
back out, and it is byte-exact for a structural reason rather than a lucky one:
the import never converted anything, so exporting is a matter of finding the same
literals again. CI imports, exports, and compares tensor by tensor against the
file that went in — and then goes further, re-importing the result and checking it
builds the **identical tensor table**. That second check is the one with teeth:
sorting the tensors on the way out keeps every byte and still produces a different
graph, because §04.2's load order is information the source file carried.

The config is *reconstructed* from the container rather than remembered — bit
width from the packed dtype, group size from the scale grid, act-order from
whether the scale is gathered, and `checkpoint_format` from whether the `+1` node
is in the expression. A container that has been through `omni delta` or had its
provenance stripped still exports correctly, because none of that lives in the
provenance.

An ascending `g_idx` is the one tensor the import does not store, since
`group_size` already says it; the export recomputes it, which is writing down a
fact the container kept in a smaller form rather than inventing a tensor.

One thing is refused rather than approximated: a container whose layers disagree
about bit width, group size, act-order or checkpoint format. That is representable
in OMNI and not in a format whose config states each of those once, and writing a
config that is right for the first layer would be the quietest possible
corruption.

### PyTorch — import only, and the reason is §12.10

```console
$ omni import pytorch pytorch_model.bin -o model.omni
```

A `torch.save` file is a ZIP archive containing one pickle and a data member per
storage. The pickle is the whole problem: unpickling *is* executing, and
`torch.load` on a file from the internet has been a remote-code-execution
primitive for as long as the format has existed. §12.10 is the answer and the
importer implements it:

- **An opcode allowlist.** Every opcode this build does not run is an error
  naming the opcode — `INST`, `OBJ` and the three extension-registry opcodes
  among them, since each resolves to an arbitrary class.
- **A class allowlist of nineteen symbols**: `collections.OrderedDict`,
  `torch.Size`, three tensor-rebuild functions, and fourteen storage classes.
  A `GLOBAL` naming anything else is a hard error that quotes the symbol, and
  the CLI exits **4** — policy, not "malformed" — because the file is perfectly
  well formed and the answer is still no.
- **No call mechanism.** `REDUCE` matches on six names. There is no `import`, no
  attribute lookup, no `__reduce__` dispatch and no `__setstate__`, so `BUILD`
  on anything but a dict is refused too.

CI does not take this on faith: it builds a checkpoint with a `os.system`
payload, asserts that Python's own `pickle.loads` **does** run it, then asserts
that the importer refuses it by name and that nothing ran.

What comes across is what a tensor *is* in PyTorch: a view. `storage_offset`,
`size` and `stride` map onto §04.4's `strided` layout directly, so a transposed
weight keeps its strides instead of being densified into a different array, and
two views of one buffer are reported rather than silently duplicated. Non-tensor
leaves — epoch counters, config scalars — are preserved as text in a `Foreign`
object (I2); anything needing a Python class to reconstruct is refused, because
the alternative is running it.

Two things are deliberately absent. §12.10 also asks for a confined child
process, and this build does not provide one, on the argument that there is
nothing here to confine: the unpickler is a parser for a data language, not an
evaluator with a filter in front of it. And there is no PyTorch *exporter* —
§12.10 clause 4 says never to re-emit pickle, and "unless explicitly requested"
is not a request this build accepts. Export to safetensors instead.

### Hugging Face repo — the five files, together

```console
$ omni import hf ./Meta-Llama-3-8B-Instruct -o llama3-8b.omni
```

This is the row that matters most in practice, and the reason it is a row rather
than a footnote on safetensors: a model on the hub is `model*.safetensors` (often
sharded, with an index deciding which tensor is where), `config.json`,
`tokenizer.json`, `tokenizer_config.json` and `generation_config.json`, and none
of them means anything alone. Import them one at a time and you get five
containers that do not know about each other. Import them together and you get
the artifact the whole format is for: one content-addressed file where the
tokenizer that shipped with these weights is *in* it, addressed by digest, rather
than being a second download that might not match.

The mapping, and what it refuses to invent:

- **Weights**: every shard, in the order the `*.index.json` names them, through
  the safetensors or PyTorch importer. A repo with no index is its own order.
- **`config.json` → §06.2 `arch`**. `model_type` becomes `family`; the keys with
  a §06.2 spelling get it; everything else is kept verbatim under `hf`, so the
  file's own names survive and nothing is ambiguous about which name a value
  had.
- **RoPE**. §06.3 requires `interleaved` when `kind` is `rope`, and calls it the
  field that "has caused more silent output corruption in format conversions
  than any other". `config.json` never states it. So it is written *only* for
  families whose `transformers` implementation is unambiguous, recorded as an
  assumption naming where the value came from — and for an unfamiliar
  `model_type` no `rope` block is written at all, with `rope_theta` kept under
  `hf` so the number is not lost.
- **`tokenizer.json` → §06.7**. BPE, Unigram, WordPiece and WordLevel; merges
  converted from strings to **id pairs**, which is what §06.7 stores and is four
  times smaller. A merge naming a token the vocabulary lacks is an error, not a
  skipped line — it means the two halves of the file disagree and the ids would
  come out wrong. A vocabulary with a gap in its ids is an error for the same
  reason. Every normalizer, pre-tokenizer and decoder step outside the catalogue
  is carried **by name** as unsupported, so `omni verify --tokenizer` says
  *indeterminate* rather than producing plausible wrong ids.
- **`tokenizer_config.json` → §06.9**. The `chat_template` goes through the
  Jinja2 translator. If it cannot be expressed totally, no template is written
  and the blocking construct is named — which is §06.9's whole argument, applied
  to itself.
- **`generation_config.json`** is kept as the manifest's `generation` block.

Two absences are deliberate. No license is written, because `config.json` has no
license field and a repo's `LICENSE` is prose. And **no tokenizer conformance
vectors** (§06.7.1): vectors have to come from the source tokenizer, this
importer does not run `tokenizers`, and a vector generated by the encoder it is
meant to test cannot fail. The gap is in the fidelity report rather than papered
over.

GGUF is the fifth and sixth rows, and it is the one that decides whether §05.2.4
is a mapping or a wish:

```console
$ omni import gguf model-Q4_K_M.gguf -o model.omni
$ omni export gguf model.omni -o back.gguf          # the same file, byte for byte
```

Eleven block types — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`,
`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` — each as one or two `dequantize` nodes over
literals whose `packed` layouts name the bit widths, and nothing re-encoded. A
GGML block is a struct, so an imported tensor keeps every source byte regrouped
by field: the `d` scales of all blocks in one literal, the packed values in
another, the 6-bit sub-scales as three literals over the same twelve bytes at
three different widths. The interleaves — `Q4_0`'s elements *i* and *i+16* in
one byte, `Q6_K`'s four values scattered across a nibble pair and a two-bit
field — are `permute`s of a (word, slot) literal, which is the same mechanism
the GPTQ and AWQ importers use for their packings and not a new one.

Because nothing is re-encoded, the export is a re-interleave and the round trip
is byte-exact by construction. That is checked *at import time*, on the file in
hand: every tensor is reassembled from its stored fields and compared with the
source, which is the export path run early. What byte identity cannot see is a
value read the wrong way, so every block is also dequantized twice — once
through the expression graph, once by scalar code transcribed from the block
layouts — and compared element by element. CI adds a third implementation,
`tools/gguf-fixture.py`, written in Python from the format's own definitions,
and compares 5 760 float32 values bit for bit.

What is refused, each by name: the `IQ*` types, whose values index a lattice
codebook that lives in llama.cpp's source rather than in the file — §05.6 rule 1
says an importer that does not know the exact dequantization must not invent
one, and a table reproduced from memory is exactly that; `Q8_K`, which is
llama.cpp's intermediate and is never written to a file; the repacked
`Q4_0_4_4`/`_4_8`/`_8_8` types, which are one CPU's cache layout rather than a
model; and GGUF v1.

`--keep-opaque` adds §4.2's second representation: each quantized tensor's
blocks, verbatim, as a §10.6 `RuntimeCache` with an `opaque` dtype and an
`interleaved` layout — the form `llama.cpp` can map and run with no conversion.
It is off by default because it doubles the stored bytes of every quantized
tensor and the structural form already preserves them; §05.2.4 says the
structural form is canonical and the opaque one *may* be attached, and this is
that may. Two rules make it an attachment rather than a second source of truth:
the cache is keyed by the structural expression's digest, so a stale one is
detectable (§10.6 rule 2), and it is flagged `CACHEABLE`, so deleting every
cache leaves the same model (rule 1). The export path reads the cache and
*compares* it with what the structural form reassembles to — §4.2's "verified
equivalence between the two" — and a disagreement is an error rather than a
preference.

One thing §4.2 describes is not here. **No tokenizer** is
synthesized, which is a finding rather than an omission: GGUF stores the
vocabulary, the merges and the scores, but `tokenizer.ggml.pre` names a
pre-tokenizer whose regexes are compiled into llama.cpp. A §06.7 tokenizer built
from those keys would decode correctly and encode differently from the model it
shipped with, so the keys are preserved in the `Foreign` object and the gap is
named in the fidelity report. The `chat_template` key, which *is*
self-contained, goes through the §06.9 translator like any other.

ONNX is the seventh and eighth rows, and the only one where the *graph* is the
thing being imported rather than a table of weights:

```console
$ omni import onnx model.onnx -o model.omni
$ omni export onnx model.omni -o back.onnx --allow-lossy
```

The protobuf wire format is implemented here, since a dependency-free crate has
no library to call: seven kinds of field and a varint. Groups — wire types 3 and
4, removed from proto3 — are refused rather than skipped, because a reader that
skips a field it cannot find the end of reads everything after it from the wrong
offset.

**The line the mapping draws is the whole design.** §07.1's charge against ONNX
is that a single abstraction level forces every backend to pattern-match
`attention` back out of fifteen primitives. An importer can repeat that mistake
in the other direction: `Relu` *is* `maximum(x, 0)`, exactly, and importing it
that way would oblige the exporter to recognise the pair and fuse it back — a
peephole matcher over the graph, which is the thing being complained about. So
the rule is mechanical: **an ONNX op is translated only when one OMNI op means
exactly what it means.** One table, read in both directions, twenty-four op types
on it — the elementwise arithmetic, `MatMul`, `Transpose`, `Concat`, `Softmax`
from opset 13, `Cast`, `Gather`, `CumSum`, `Reshape`, the five reductions,
`Constant`, and the per-tensor QDQ pair.

Everything else is **carried** in a compat dialect named after the ONNX domain it
came from — `ai.onnx` for the default domain, `ai.onnx.ml` and `com.microsoft`
for the rest — with its attributes intact. The dialect's version is the opset the
file imported, which is the most faithful thing there is to record: ONNX versions
its whole opset at once, so every op in one file shares one number, and §07.4.1's
per-op versions exist precisely to avoid that. Spreading one opset number over
per-op versions this build does not know would be inventing information.

Carrying is not a failure, and §11.3 is the reason. A container full of
`ai.onnx` ops verifies, copies, signs, deduplicates and round-trips byte for
byte; `omni graph --verify` reports those ops **indeterminate** rather than
invalid, which §15.1 makes normative; and the one operation that actually needs
the semantics — execution — is refused by name. Each of those is a CI assertion
rather than a claim.

Several ops are *nearly* mappable and are carried instead, each for a stated
reason, because "nearly" is where silent corruption lives:

| Op | Why it is carried |
|---|---|
| `Relu`, `Identity`, `Gemm`, … | no single OMNI op means it |
| `Slice`, `Pad` | ONNX's bounds may be negative or past the end and are clamped; `omni.tensor`'s must be in range, so normalizing them is a *lowering* that needs the operand's shape |
| `MatMul` on a rank-1 operand | ONNX promotes it to a matrix and drops the added axis afterwards; `omni.tensor/matmul` takes rank ≥ 2 |
| `Max`, `Min` with more than two operands | they are variadic in ONNX; chaining is a lowering |
| `Softmax` before opset 13 | it flattens to two dimensions first, which is a reshape and a softmax |
| `Reshape` whose target shape holds a `0` | a `0` means "copy this dimension from the operand" unless `allowzero` is set |
| `Cast` with `saturate: 0` | an out-of-range cast then produces an infinity rather than the type's maximum, which §04.3's rounding modes do not name |
| per-axis and blocked `QuantizeLinear` | which elements share a scale needs §05.1's block shape over a shape a dynamic graph does not fix |
| anything whose shape is computed | a target shape or an axis list that is an operand rather than a constant is not an attribute an import can write |

**Two independent shape functions, on every value.** ONNX files usually carry
declared types for their intermediates, and OMNI has shape functions of its own.
The import runs both and compares them: a disagreement about a dimension *both*
state is an error naming the axis, not a warning, because one of the two readers
is then wrong about what the model computes. Where OMNI has no shape function —
every carried op — the file's declaration is used, and a value neither of them
can type stops the import with the remedy named (run ONNX's shape inference, or
`--no-graph` for the weights alone). Inventing a rank there would be inventing it
for every op downstream.

**External data is a second, weaker format inside the first.** Protobuf refuses
to encode more than 2 GB, so ONNX moved weights into sibling files referenced by
a path in a string. That is an untrusted path: resolution is the caller's
decision, `omni import onnx` resolves it against the model's own directory, and a
`location` that is absolute or contains `..` is refused before anything is opened
(§12.4).

Refused by name, each with its reason: `STRING`, `COMPLEX64` and `COMPLEX128`
initializers; the `FNUZ` float8 variants, which differ from the ones §04.3 names
by an exponent bias, so importing one as the other would change every value;
subgraph attributes, which are regions in §07.3 and whose scope rules this build
has not worked out how to translate; local functions; `TrainingInfoProto`; and
sparse initializers, which §04.6 has a catalogue for and this build has not
written the mapping to.

**The export refuses more than it writes, and that is the point.** A container
with no graph is not an ONNX file and says so — §07.5 makes the graph optional in
OMNI precisely because most models ship without one. A `semantic`-level graph is
refused with a pointer to `omni graph lower`, since an opset *is* the primitive
level and choosing an abstraction level on the model's behalf is not an
exporter's decision. And an op with no ONNX spelling stops the export with the
list of them: that is not a lossy export and `--allow-lossy` does not cover it,
because an op that cannot be written is the computation rather than lost
metadata.

Running that against this repository's own worked transformer is a measurement of
what the ONNX opset costs. Lowered to primitives, 49 nodes map — `MatMul`,
`Transpose`, `Reshape`, `Mul`, `Add`, `ReduceMean`, `Gather`, `Cast` — and three
kinds do not: `omni.nn/attention` and `omni.nn/rope`, which are the semantic ops
§07.2 exists to keep, and `omni.tensor/rsqrt`, which ONNX simply has no operator
for.

The round trip is byte-exact for a file this build imported, and the reason is
structural rather than lucky: the ops and their attributes come back out of the
graph, and everything OMNI-IR has no field for — producer strings, doc strings,
node names, and the names ONNX gives the values OMNI-IR numbers — is preserved in
a `Foreign` object and put back (I2). The one thing that object deliberately does
*not* hold is the ops themselves, because an export that read those from it would
be copying a file rather than translating a model. §5.3 still does not promise
this round trip in general, and the reason is unchanged: a file whose tensors use
the typed arrays rather than `raw_data`, or whose fields are not in ascending
order, comes back with the same values in a different encoding.

`tools/onnx-fixture.py` is the third implementation — Python, standard library
only, written from the wire format and the operator specifications. It writes the
file, parses back what the export wrote with its own reader, and computes what
the graph should produce. CI checks all three: the container's tensors are bit
identical to the bytes Python packed, the executed graph agrees with Python's
arithmetic on every output, and the exported file is the same bytes as the
imported one.

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
