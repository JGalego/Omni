# OMNI/1.0 — §4 Tensors

Layer L2. This section contains OMNI's central technical contribution: a tensor
is not a byte range, it is a **pure expression** whose byte range is one possible
leaf.

---

## 4.1 The idea

In every existing format, a tensor *is* its bytes. Therefore:

- a quantized model is a **new copy** of every tensor;
- a LoRA fine-tune is either a separate file the runtime must know how to fuse,
  or a **merged copy** of every tensor;
- an fp8 deployment variant is a **third copy**;
- and the relationship between them is documented in a README.

In OMNI a tensor's `value` is a node in a small, closed, pure algebra:

```
W_int4         = literal(chunks…)                       ; the only bytes stored
scales         = literal(chunks…)
W_bf16         = dequantize(W_int4, affine{scales, zeros, block=[1,128]})
W_lora         = add(W_bf16, scale(matmul(B, A), 0.0625))
W_deploy_fp8   = cast(W_lora, f8e4m3, rounding="rne")
```

Only `W_int4`, `scales`, `A` and `B` occupy storage. `W_bf16`, `W_lora` and
`W_deploy_fp8` are *definitions*. A runtime materializes whichever node its
hardware wants, and caches the result keyed by that node's digest (§10.6).

This single change collapses the N × M artifact explosion (§00.1) into N + M.

## 4.2 TensorTable and TensorDesc

```cbor-diag
; TensorTable  (otype 0x0004)
{ "t":"omni.tensor/table", "v":1,
  "tensors": {
     "model.layers.0.attn.q_proj.weight": [5, h'…'],   ; -> TensorDesc
     "model.layers.0.attn.k_proj.weight": [5, h'…'],
     …
  },
  "groups": {                       ; optional logical grouping for I/O planning
     "layer.0": ["model.layers.0.*"],
     "experts.3": ["model.layers.*.mlp.experts.3.*"]
  },
  "order": ["model.embed_tokens.weight", "model.layers.0.…", …]  ; load order hint
}
```

For very large tables use `ShardedMap` (§01.4.1).

```cbor-diag
; TensorDesc  (otype 0x0005)
{ "t":"omni.tensor/desc", "v":1,

  "shape":  [4096, 4096],           ; uint | text (symbolic) | -1 (dynamic)
  "dtype":  {"k":"float","w":16,"e":8,"m":7},        ; §4.3
  "layout": {"k":"strided","order":"row-major"},     ; §4.4
  "value":  <expr>,                                  ; §4.7

  "semantic": "weight",             ; weight|bias|scale|zero|index|state|buffer|
                                    ; embedding|codebook|mask|constant|opaque
  "role": "attn.q_proj",            ; free-form, dialect-defined
  "axes": ["out_features","in_features"],   ; named axes for shardability/adapters
  "device_hint": "gpu",
  "materialize": "lazy",            ; eager|lazy|stream
  "stats": {"min":-0.31,"max":0.29,"mean":1.2e-5,"absmax":0.31,
            "nan":0,"inf":0,"nonzero":16777003},     ; optional, verifiable
  "digest_materialized": h'…',      ; optional: digest of the evaluated bytes
  "ext": {}
}
```

`axes` names are load-bearing: they are how §08 attaches adapters without
hardcoding layouts, and how §09 describes sharding without hardcoding
frameworks.

`digest_materialized` is optional but powerful: it lets a runtime verify that its
own evaluation of an expression matched the publisher's, catching numerical
divergence between implementations. Because evaluation order can affect
floating-point results, it is only normative when the expression's `det` flag is
set (§4.7.6).

## 4.3 The numeric type algebra

A dtype is **not an enum**. Enums are why every format needs a new release for
every new precision. A dtype is a structured descriptor:

### 4.3.1 Float

```cbor-diag
{"k":"float", "w":16, "e":8, "m":7, "bias":127,
 "sub":true, "inf":true, "nan":"ieee", "sign":"lead"}
```

| Field | Meaning |
|---|---|
| `w` | total bit width |
| `e` | exponent bits |
| `m` | mantissa (trailing significand) bits |
| `bias` | exponent bias (default `2^(e-1)-1`) |
| `sub` | subnormals supported |
| `inf` | infinities encodable |
| `nan` | `"ieee"` (all-ones exp, non-zero mantissa) \| `"fn"` (finite-only, one NaN pattern) \| `"none"` |
| `sign` | `"lead"` (sign is MSB) \| `"none"` (unsigned float) |

Named aliases (registry §4.3.6) — all are *derived* from the above, never
primitive:

| Alias | w | e | m | Notes |
|---|---:|---:|---:|---|
| `f64` | 64 | 11 | 52 | IEEE 754 binary64 |
| `f32` | 32 | 8 | 23 | binary32 |
| `f16` | 16 | 5 | 10 | binary16 |
| `bf16` | 16 | 8 | 7 | bfloat16 |
| `tf32` | 19 | 8 | 10 | storage-padded to 32 in practice; layout flag |
| `f8e4m3` | 8 | 4 | 3 | `nan:"fn"`, no inf (OCP FP8 E4M3) |
| `f8e5m2` | 8 | 5 | 2 | `nan:"ieee"`, has inf |
| `f6e3m2` | 6 | 3 | 2 | OCP MX element type |
| `f6e2m3` | 6 | 2 | 3 | OCP MX element type |
| `f4e2m1` | 4 | 2 | 1 | OCP MX element type |
| `e8m0` | 8 | 8 | 0 | MX **scale** type (power-of-two exponent only) |

This table is data, not code. A 2035 format with `w:12,e:5,m:6` needs no spec
revision — only a registry alias if people want a short name.

### 4.3.2 Integer, fixed point, and small types

```cbor-diag
{"k":"int",   "w":8,  "signed":true}
{"k":"int",   "w":4,  "signed":false}
{"k":"int",   "w":2,  "signed":true}
{"k":"bool",  "w":1}
{"k":"fixed", "w":16, "signed":true, "frac":8}          ; Q8.8
{"k":"ternary","vals":[-1,0,1], "pack":"b3x5"}          ; 5 trits in 8 bits
{"k":"binary","vals":[-1,1]}                            ; 1-bit sign weights
```

`pack` names a bit-packing scheme from the registry: `"b3x5"` = base-3 encoding
of 5 ternary values in one byte (1.6 bits/value, vs. 2 bits naive) — the encoding
BitNet-class models need and which no existing format can express without a
custom loader.

### 4.3.3 Codebook (lookup) types

The general answer for NF4, AF4, vector quantization, k-means quantization, and
GGUF's LUT-ish variants:

```cbor-diag
{"k":"codebook", "w":4,                 ; 4-bit indices
 "book": [7, h'…'],                     ; -> Codebook object
 "dim": 1,                              ; 1 = scalar VQ; >1 = vector quantization
 "shared": "per-tensor"}                ; per-tensor|per-block|per-row
```

A `Codebook` object holds the values, their dtype, and optionally a
construction recipe (`kmeans{seed,iters}`, `normal-float{sigma}`) so the
codebook itself is reproducible.

### 4.3.4 Composite and exotic types

```cbor-diag
{"k":"complex", "re":{"k":"float","w":32,"e":8,"m":23}}
{"k":"struct",  "fields":[["r",{…}],["i",{…}]], "packed":true}
{"k":"opaque",  "id":"org.ggml/q4_K", "block_elems":256, "block_bytes":144}
{"k":"posit",   "w":16, "es":2}                    ; if posits ever matter
{"k":"logdom",  "w":8, "base":2, "frac":4}         ; log-number-system accelerators
{"k":"string",  "enc":"utf8"}                      ; vocabularies, labels
```

`opaque` is the pressure valve: it lets an importer preserve a foreign block
format bit-exactly, with correct sizes and correct partial-read behaviour, even
when OMNI has no semantic model of it. It is *not* a licence to be lazy — a
lossless import SHOULD express the format structurally (§05.6) and MAY carry the
opaque form alongside as a cache.

### 4.3.5 Dtype invariants

For any dtype `D`, the registry/descriptor MUST determine:

1. `bits(D)` — bits per element (may be fractional, e.g. 1.6 for `b3x5`).
2. `pack(D, n)` — bytes needed for `n` elements, including padding rules.
3. `decode(D, bytes, i) → ℝ ∪ {NaN, ±∞}` — bit-exact element semantics.
4. `encode(D, x, rounding) → bits` — for `rne`, `rtz`, `rup`, `rdown`,
   `stochastic{seed}`.

For registered dtypes these are normative and tested by the conformance suite
(§15). For `opaque` dtypes only (1) and (2) are known, and any operation other
than `literal`/`slice`/`cast-to-opaque` is undefined.

### 4.3.6 Registry

Aliases live in a registry file (`registry/dtypes.cbor`) shipped with the spec
and embeddable via `Schema` objects. An alias is a *name for a descriptor*;
readers MUST accept the expanded descriptor form even for known aliases, and
MUST treat an unknown alias with an inline descriptor as fully understood.
Writers SHOULD emit both (`{"alias":"bf16","k":"float","w":16,"e":8,"m":7}`) —
5 extra bytes buys total forward compatibility.

## 4.4 Layouts

Shape says *what*; layout says *where the bits are*.

```cbor-diag
{"k":"strided", "order":"row-major"}                     ; default, C-contiguous
{"k":"strided", "strides":[1,4096], "offset":0}          ; explicit, e.g. column-major
{"k":"tiled",   "tile":[128,64], "outer":"row-major", "inner":"row-major"}
{"k":"blocked-scaled", "block":[1,32],
 "scale_dtype":{"alias":"e8m0"}, "scale_layout":{"k":"strided"},
 "interleave":"scales-after"}                            ; MX / MXFP4
{"k":"packed",  "elems_per_word":8, "word_bits":32, "bit_order":"lsb-first"}
{"k":"interleaved", "groups":[["w","s","z"]], "stride_bytes":144}  ; GGUF-style blocks
{"k":"sharded", "spec": <ShardSpec>}                     ; §09.4
{"k":"opaque",  "id":"org.nvidia/tensorrt-weights.v10"}
```

Rules:

- The layout MUST be sufficient to compute the byte offset and bit position of
  element `(i₀…i_{n-1})` with no additional knowledge.
- `blocked-scaled` is the structural answer to MX formats, GPTQ groups, AWQ
  groups, and GGUF K-quants: an element block plus a per-block scale (and
  optionally zero-point) with a declared interleaving.
- `packed` is the answer to sub-byte types; `bit_order` and `elems_per_word` are
  where every existing implementation quietly disagrees, so OMNI makes them
  explicit and the conformance suite tests all four combinations.
- **Layout is orthogonal to dtype.** `int4` in `packed` layout and `int4` in
  `blocked-scaled` layout are the same numbers, different bits — and the expression
  algebra can convert between them with `relayout`.

## 4.5 Chunks and ChunkList

A tensor's bytes live in a `ChunkList`:

```cbor-diag
{ "t":"omni.tensor/chunklist", "v":1,
  "total": 33554432,                       ; logical bytes
  "chunker": {"k":"fixed","size":4194304},
  "chunks": [
    {"r":[0, h'…'], "n":4194304},          ; ref + logical length
    {"r":[0, h'…'], "n":4194304},
    …
  ],
  "bao": [20, h'…']                        ; optional verified-streaming tree
}
```

For `fixed` chunking a reader computes `chunk_index = offset / size` — no search.
For `cdc-gear` a prefix-sum array is included so the same lookup is a binary
search. Either way, **random access into a multi-gigabyte tensor costs one index
lookup and one page fault.**

Chunks are ordinary `Blob` objects, so:

- two models sharing a frozen embedding table share those chunk objects;
- a resumed download resumes at chunk granularity;
- a partial container can hold layers 0–7 and refer to the rest as `EXTERNAL`;
- a chunk shared by 300 fine-tunes exists once in a registry.

## 4.6 Sparsity

Sparse tensors are values, not a separate tensor kind — they are produced by a
`sparse` expression node.

| Scheme | Encoding | Use |
|---|---|---|
| `coo` | `indices` (n_dims × nnz), `values` (nnz) | general, GNNs |
| `csr` / `csc` | `indptr`, `indices`, `values` | classic sparse linear algebra |
| `bsr` | block CSR: `block=[r,c]` | block-sparse attention, pruned MLPs |
| `nm` | `n:m` structured (`{"n":2,"m":4}`), bitmask + values | NVIDIA sparse tensor cores |
| `bitmask` | dense bitmap + packed values | unstructured pruning |
| `ragged` | `offsets` + flat values | variable-length sequences, MoE token routing |
| `blocklist` | list of (index, dense block) | MoE experts, sparse deltas |

```cbor-diag
{"op":"sparse", "scheme":"nm", "n":2, "m":4,
 "mask":  <expr>,          ; -> tensor of dtype {k:"bool"}
 "values":<expr>,
 "shape":[4096,11008], "dtype":{"alias":"bf16"}, "fill":0.0}
```

Because sparsity is an expression node, a pruned fine-tune of a dense base is
naturally `add(base, sparse(...))` — the delta costs only its non-zeros (§08.6).

## 4.7 The tensor expression algebra (OTA)

### 4.7.1 Principles

1. **Pure.** No side effects, no I/O, no state. An expression denotes a value.
2. **Total.** Every node has a defined type and shape given its inputs, checked
   statically; there are no runtime-dependent shapes at this layer.
3. **Closed and small.** The core node set is fixed at 24 operations. Anything
   else is a `plugin` node (§4.7.7), which a reader may refuse.
4. **Deterministic identity.** The digest of the expression tree is the value's
   identity, independent of how or whether it is materialized.
5. **Not a compute graph.** OTA describes *weights*, not *inference*. It has no
   control flow, no batching, no state. Inference lives in §07. Keeping them
   separate is what keeps OTA analyzable and cheap.

### 4.7.2 Core nodes

**Leaves**

| Node | Args | Meaning |
|---|---|---|
| `literal` | `chunks`, `dtype`, `shape`, `layout` | bytes in the store |
| `extern` | `uri`, `digest`, `dtype`, `shape` | bytes elsewhere (never fetched implicitly) |
| `zeros` / `ones` / `full` | `value`, `dtype`, `shape` | generated constants |
| `arange` / `eye` | … | generated |
| `random` | `dist`, `seed`, `dtype`, `shape` | reproducible PRNG (§4.7.6) — for initialization and tests |

**Structural**

| Node | Meaning |
|---|---|
| `reshape` / `transpose` / `permute` / `squeeze` / `expand` | pure index remapping |
| `slice(x, starts, sizes, steps)` | sub-tensor |
| `concat(xs, axis)` / `split(x, axis, sizes)` | join / cut |
| `pad(x, pads, mode, value)` | padding |
| `gather(x, idx, axis)` | index selection (vocabulary pruning, expert selection) |
| `relayout(x, layout)` | same values, different bit placement |

**Numeric**

| Node | Meaning |
|---|---|
| `cast(x, dtype, rounding)` | precision change with explicit rounding |
| `add` / `sub` / `mul` / `div` (broadcasting) | elementwise |
| `scale(x, k)` | scalar multiply (kept distinct: it is the LoRA α/r case and enables exact rational scaling) |
| `matmul(a, b)` | contraction — the LoRA/DoRA case |
| `norm(x, axis, p)` | vector norms — the DoRA magnitude case |
| `clamp(x, lo, hi)` | saturation |

**Quantization**

| Node | Meaning |
|---|---|
| `dequantize(x, scheme)` | integer/codebook → float (§05) |
| `quantize(x, scheme, rounding)` | float → integer/codebook |
| `sparse(scheme, …)` | sparse → dense value |
| `approx(x, bound)` | marks an intentionally lossy subtree (§03.7.3) |

**Composition**

| Node | Meaning |
|---|---|
| `delta(base, patch, op)` | `op ∈ {add, xor, replace, sparse-add}` — inheritance (§08.6) |
| `select(cond_feature, a, b)` | capability-conditional value (§10.3) |
| `plugin(op, ns, args, attrs)` | extension point |

That is 24 core operations. The full set of quantization schemes, sparsity
schemes, dtypes and layouts is *data* consumed by four of them.

### 4.7.3 Typing

Every node has a static `(shape, dtype, layout)`. Shape inference is standard
(NumPy broadcasting for elementwise; standard rules for `matmul`, `concat`,
`slice`). A writer MUST record the resulting `shape` and `dtype` on the owning
`TensorDesc`, and a reader MUST verify that inference agrees. Disagreement is a
hard error — it is the cheapest possible detection of a malformed or malicious
file.

Symbolic dimensions (text entries in `shape`) may appear only for tensors whose
size genuinely varies (e.g. a vocabulary being extended); they must resolve
through the model's `dims` binding table before materialization.

### 4.7.4 Evaluation

An evaluator is a straightforward tree walk with three refinements:

1. **Lazy and range-driven.** `eval_range(expr, byte_range)` pushes the range
   through structural nodes so that reading rows 100–200 of
   `dequantize(literal(...))` fetches only the chunks covering those rows.
   Structural nodes (`slice`, `concat`, `gather`, `reshape`) all have exact
   inverse-range functions; numeric nodes are elementwise or block-local; only
   `matmul` and `norm` require a full contraction dimension, and the evaluator
   reports the true dependency set. **Partial loading is therefore automatic,
   not a special case.**
2. **Fusion.** `dequantize → cast → add` chains are fused into a single pass,
   so a LoRA-merged int4 model materializes in one traversal with one output
   buffer.
3. **Caching.** Any node's result may be cached under `H("omni/1.0 expr-identity" ‖
   canonical(expr))`. Cache correctness is automatic: a different expression is a
   different key. This is how §10.6 runtime caches stay honest.

### 4.7.5 Expression identity

The digest of a `TensorDesc.value` is computed over its canonical OMNI-CBOR
encoding **after normalization**:

- constant folding of pure structural chains that do not touch `literal` bytes;
- algebraic canonicalization: `scale(scale(x,a),b) → scale(x,a*b)` with exact
  rational arithmetic; `transpose(transpose(x))` elimination; `cast(cast(x,T),T)`
  collapse **only when provably lossless**;
- commutative argument sorting by sub-digest for `add`/`mul`.

Normalization MUST NOT change values. It exists so that two publishers who write
the same model in different but equivalent ways produce the same digest — which
is what makes cross-organization deduplication actually work.

### 4.7.6 Determinism

Nodes are marked `det: true` when their result is bit-reproducible across
conforming implementations. `literal`, all structural nodes, `cast`,
`dequantize` with integer/LUT schemes, and `random` (which uses a specified
counter-based PRNG — ChaCha20-based, defined bit-exactly in the registry) are
deterministic. `matmul`, `norm`, and float `add`/`mul` with more than pairwise
summation are **not** bit-deterministic across implementations, and OMNI says so
rather than pretending otherwise: a node may carry
`{"sum":"pairwise"|"sequential"|"kahan"}` to pin the reduction order when
bit-exactness is required (at a performance cost). `digest_materialized` is
normative only over fully-deterministic subtrees.

This is an area where existing formats are silently wrong; making it explicit
costs one key and prevents an entire class of "the merged weights don't match"
bug reports.

### 4.7.7 Plugin nodes

```cbor-diag
{"op":"plugin", "ns":"org.acme/quant", "name":"my-scheme", "v":2,
 "args":[…], "attrs":{…},
 "crit":true,
 "shape":[4096,4096], "dtype":{"alias":"bf16"},
 "fallback": <expr>}          ; optional: an equivalent core-only expression
```

`fallback` is the key to graceful degradation: a publisher using an exotic
quantizer can provide a (larger, slower, but core-only) equivalent so that any
C1 reader can still load the model. If `crit` is true and no fallback exists, a
reader without the plugin MUST refuse the tensor — but MAY still read the rest of
the model, report the missing plugin, and load a different realization (§10).

## 4.8 Worked example: one linear layer, four ways

All four share the *same stored bytes*.

```cbor-diag
; ---- stored objects ----
q      = literal(chunks=[…],   dtype={k:"int",w:4,signed:false},
                 shape=[4096,4096], layout={k:"blocked-scaled",block:[1,128],…})
scale  = literal(chunks=[…],   dtype=bf16,  shape=[4096,32])
zero   = literal(chunks=[…],   dtype={k:"int",w:4}, shape=[4096,32])
A      = literal(chunks=[…],   dtype=bf16,  shape=[16,4096])
B      = literal(chunks=[…],   dtype=bf16,  shape=[4096,16])

; ---- derived values, zero bytes stored ----
W_bf16   = dequantize(q, {scheme:"affine", scale:scale, zero:zero,
                          block:[1,128], axis:1, out:bf16})
W_lora   = add(W_bf16, scale(matmul(B, A), 30/16))       ; α=30, r=16, exact rational
W_fp8    = cast(W_lora, f8e4m3, "rne")
W_int8   = quantize(W_lora, {scheme:"affine", axis:0, out:{k:"int",w:8,signed:true}}, "rne")
```

| Consumer | Uses | Bytes it must fetch |
|---|---|---|
| llama.cpp-class int4 CPU runtime | `q, scale, zero` (+ fused LoRA at load) | 8.4 MB |
| bf16 GPU trainer | `W_lora` | 33.6 MB (materialized) |
| fp8 inference server | `W_fp8` | 16.8 MB (materialized, cached) |
| int8 edge NPU | `W_int8` | 16.8 MB (materialized, cached) |

Stored on the publisher's side: **8.4 MB + 0.26 MB of LoRA**, once. Under the
status quo those four consumers require four separate uploads totalling 75 MB
and no relationship between them beyond a naming convention.

## 4.9 Tradeoffs, stated honestly

| Benefit | Cost |
|---|---|
| N + M instead of N × M artifacts | Readers must implement an evaluator (profile C1). A C0 reader can only load tensors whose value is a bare `literal` — so publishers targeting maximum compatibility SHOULD include a `literal`-valued realization (§10.5). |
| Perfect cache keys | Expression normalization must be specified exactly, or two implementations disagree on identity. §4.7.5 is therefore normative and conformance-tested. |
| Automatic partial loading | Range pushdown is only exact through structural nodes; `matmul` in a LoRA chain forces the full low-rank factors to be read (they are tiny, so this is fine). |
| Adapters cost their delta | Materialization is now a runtime step with a latency cost; §10.6 caches exist precisely to amortize it, and `materialize:"eager"` opts out per tensor. |
| Lossy transforms are visible | Publishers who want to hide a lossy step cannot. This is the point. |

---

**Prev:** [§03 Encoding](03-encoding.md) · **Next:** [§05 Quantization](05-quantization.md)
