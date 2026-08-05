# OMNI/1.0 — §5 Quantization

Quantization in OMNI is **a transformation, not a file type**. There is no
"quantized model"; there are tensors whose values are expressed as
`dequantize(integer_literal, scheme)`.

## 5.1 The quantization scheme descriptor

A scheme is data consumed by the `quantize` / `dequantize` nodes (§04.7.2).

```cbor-diag
{ "scheme": "affine",
  "out":    {"alias":"bf16"},        ; dequantized dtype
  "axis":   1,                       ; the quantized axis
  "block":  [1, 128],                ; group/block shape; [1,K] = per-K-group
  "scale":  <expr>,                  ; tensor of per-block scales
  "zero":   <expr>,                  ; optional zero-points
  "sym":    false,                   ; symmetric (no zero-point)
  "order":  <expr>,                  ; optional permutation (GPTQ act-order / desc_act)
  "clip":   [-8, 7],
  "formula":"(q - z) * s"            ; explicit, from a fixed enumeration
}
```

`formula` is drawn from a closed set so there is never ambiguity about whether
the zero point is subtracted before or after scaling — a genuine, recurring
source of silent corruption when converting between GPTQ, AWQ and GGUF today:

| id | formula |
|---|---|
| `affine-sub` | `(q − z) · s` |
| `affine-add` | `q · s + b` |
| `sym` | `q · s` |
| `codebook` | `book[q] · s` |
| `codebook-raw` | `book[q]` |
| `nested` | `(q − z) · (s_q − z_s) · s_ss` (double quantization) |

## 5.2 The scheme catalogue

Every scheme below is expressible with the *core* algebra — no plugins required.
This is the test of whether the algebra is adequate.

### 5.2.1 Uniform affine (the base case)

Covers: `bitsandbytes` int8, PyTorch quantized tensors, ONNX QDQ,
per-tensor / per-channel / per-group int8 and int4.

```
value = (q − zero) * scale
```
`block = [1, in_features]` → per-row; `block = [1, 128]` → group-wise;
`block = shape` → per-tensor.

### 5.2.2 GPTQ

GPTQ's *output* is uniform affine, group-wise, with a couple of wrinkles:

- **Column permutation** (`desc_act` / act-order). Expressed as
  `order`, a permutation tensor, applied via a `gather` node — no special case.
- **Packed int4 in int32 words**, layout `{"k":"packed","elems_per_word":8,
  "word_bits":32,"bit_order":"lsb-first"}`.

```cbor-diag
W = gather(dequantize(qweight, {scheme:"affine", formula:"affine-sub",
                                block:[1,128], axis:1,
                                scale:scales, zero:qzeros}),
           g_idx_inverse, axis=1)
```

Import is lossless; export back to GPTQ is lossless.

### 5.2.3 AWQ

Uniform affine group-wise plus a **pre-scaling** of activations/weights derived
from activation statistics. The per-channel smoothing factor is a real tensor and
is stored as one:

```cbor-diag
W = mul(dequantize(qweight, {…affine, block:[1,128]}), awq_scales)
```

AWQ's search procedure is metadata (`quant.method = "awq"`, plus its
hyperparameters and calibration-set reference in `Provenance`); its *result* is
two tensors and a multiply.

### 5.2.4 GGUF / GGML K-quants

The `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ*` families are block formats with
super-blocks, sub-block scales, and (for `IQ*`) codebooks. All are expressible:

| GGUF type | OMNI expression |
|---|---|
| `Q8_0` | `dequantize(int8, {sym, block:[1,32], scale:f16})` |
| `Q4_0` | `dequantize(int4, {sym, block:[1,32], scale:f16})` |
| `Q4_1` | `dequantize(int4, {affine-sub, block:[1,32], scale:f16, zero:f16})` |
| `Q4_K` | `nested`: 6-bit sub-scales quantized against a super-block f16 scale; `block:[1,32]` inner, `[1,256]` outer |
| `Q6_K` | as above with 6-bit elements and 8-bit sub-scales |
| `IQ2_XXS`… | `codebook` with a fixed 256-entry lattice book + per-block sign/scale |

Two representations are permitted and both are useful:

1. **Structural** (above): fully transparent, allows re-quantization, mixing
   with LoRA, and export to any target. **Preferred.**
2. **Opaque** (`{"k":"opaque","id":"org.ggml/q4_K","block_elems":256,
   "block_bytes":144}` with an `interleaved` layout): byte-identical passthrough
   so that `omni import model.gguf && omni export --gguf` is bit-exact and
   `llama.cpp` can `mmap` the payload with **zero conversion**.

A well-formed import produces the structural form as canonical and MAY attach the
opaque form as a `RuntimeCache` keyed by the structural expression's digest —
giving both transparency *and* zero-cost consumption by existing runtimes.

### 5.2.5 EXL2

Mixed-precision per-layer bit allocation with per-group scales. The "mixed" part
is not a new mechanism: each tensor (or each slice of a tensor) simply has its
own scheme, and a `concat` of differently-quantized slices is an ordinary
expression. EXL2's per-tensor bit budget is metadata; its output is
`concat([dequantize(slice_2bit,…), dequantize(slice_4bit,…), …])`.

### 5.2.6 HQQ

Affine with an optimized zero-point found by a half-quadratic solver, and
*optional double quantization* of scales/zeros — the `nested` formula. Storage is
identical to affine; the method is provenance.

### 5.2.7 NF4 / FP4 (bitsandbytes 4-bit)

`NF4` is a 16-entry codebook of normal-distribution quantiles:

```cbor-diag
{"k":"codebook","w":4,"book":<Codebook>,"shared":"per-tensor"}
```
with `formula:"codebook"` and a per-block absmax scale. Double quantization
(scales themselves quantized to int8 with a second-level scale) is `nested`.
The codebook object records `{"construct":"normal-float","bits":4}` so it is
reproducible rather than a magic constant table.

### 5.2.8 MX formats (MXFP8/6/4, OCP Microscaling)

Exactly what `blocked-scaled` layout + `e8m0` scale dtype were designed for:

```cbor-diag
dtype  = {"alias":"f4e2m1"}
layout = {"k":"blocked-scaled","block":[1,32],
          "scale_dtype":{"alias":"e8m0"},"interleave":"scales-after"}
value  = dequantize(literal(...), {scheme:"sym", formula:"sym",
                                   block:[1,32], scale:<e8m0 tensor>, out:bf16})
```

Because the scale dtype is a power-of-two exponent type, dequantization is
exact and bit-reproducible.

### 5.2.9 Ternary / 1-bit (BitNet class)

```cbor-diag
dtype  = {"k":"ternary","vals":[-1,0,1],"pack":"b3x5"}
value  = dequantize(literal(...), {scheme:"sym", block:[1,"row"], scale:absmean})
```
1.6 bits per weight actually stored, not 2. No other format can currently say
this without a private loader.

### 5.2.10 Custom quantizers

Anything not expressible above uses `plugin` nodes (§04.7.7) with a mandatory
`fallback` if the publisher wants C1 readers to work. Research quantizers should
start as plugins and graduate into the registry once stable — the same path
HTTP header extensions or PNG chunk types follow.

## 5.3 Mixed precision

A model may quantize different tensors, or different *parts* of a tensor,
differently. There is no "the model's quantization"; `omni inspect` reports a
histogram:

```
quantization:
  affine-int4 g128     : 226 tensors   6.71 GB
  affine-int8 per-row  :  32 tensors   0.51 GB
  bf16 (unquantized)   :  15 tensors   0.94 GB   (embeddings, norms, lm_head)
  effective bits/param : 4.31
```

`effective bits/param` is computed from stored bytes over parameter count and is
the honest number that "Q4_K_M" gestures at.

## 5.4 Codebook objects

```cbor-diag
{ "t":"omni.tensor/codebook", "v":1,
  "dtype": {"alias":"f32"}, "entries": 16, "dim": 1,
  "values": <expr>,                               ; the table itself
  "construct": {"method":"normal-float","bits":4},; optional reproducibility recipe
  "sorted": true }
```

Vector quantization (`dim > 1`) and product quantization (several codebooks over
sub-vectors, expressed as `concat` of per-subspace `dequantize` nodes) both fall
out without new machinery.

## 5.5 Calibration and quantization provenance

Quantization is a *measurement*, and measurements need provenance. A
`Provenance` object (§12.6) attached to a quantized model SHOULD record:

```cbor-diag
{ "t":"omni.prov/quantization", "v":1,
  "method":"gptq", "impl":"gptqmodel 2.1.0",
  "bits":4, "group_size":128, "act_order":true, "damp":0.01,
  "calibration": {"dataset":"c4/en", "digest":h'…', "samples":512, "seqlen":2048},
  "source_model": [1, h'…'],           ; the manifest this was derived from
  "eval": [32, h'…'],                  ; -> Evaluation object with before/after
  "produced": "2026-08-04T…Z" }
```

This makes "which calibration set produced this int4 model?" — currently
unanswerable for most published quantizations — a field lookup.

## 5.6 Import fidelity rules

1. An importer MUST NOT invent a scheme. If the source's exact dequantization
   formula is not known to the importer, it MUST use `opaque` dtype rather than
   guess a formula.
2. An importer MUST record, in the fidelity report (§import-export), whether the
   structural form was verified to reproduce the source bit-for-bit. The
   recommended verification is: dequantize both the structural and the opaque
   forms for a random sample of blocks and compare exactly.
3. `omni convert --verify` performs full re-materialization and comparison, and
   refuses to write a container whose structural form disagrees with its source.

## 5.7 Why quantization must not be a top-level format concept

The alternative — a `quantization: "Q4_K_M"` enum in the header, as GGUF does —
fails on five counts:

1. It cannot express **mixed** precision without a second mechanism.
2. It couples the format's release cycle to quantization research, which moves
   monthly.
3. It cannot represent the *relationship* between the fp16 and int4 versions, so
   they are unrelated files.
4. It cannot be composed with adapters: "Q4_K_M + LoRA" has no representation, so
   runtimes fuse at load time with ad-hoc code.
5. It hides the calibration data, which is exactly the information needed to
   evaluate whether the quantization is trustworthy.

Making quantization a transformation costs one evaluator and removes all five.

**Prev:** [§04 Tensors](04-tensors.md) · **Next:** [§06 Metadata & Tokenizers](06-metadata.md)
