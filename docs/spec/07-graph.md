# OMNI/1.0 — §7 Execution Graph (OMNI-IR)

Layer L3. Optional: a model may be weights-only. When present, OMNI-IR is what
makes a model *self-describing* — executable by a runtime that has never heard of
its architecture.

## 7.1 The problem with existing model IRs

| IR | Failure mode |
|---|---|
| **ONNX** | One global opset, versioned monolithically. Adding an op requires committee approval; using a custom op makes the file non-portable. Worse: ONNX *lowers* semantics — attention becomes 15 primitive ops, so a TensorRT or NPU backend must **pattern-match** to recover the intent it needs for a fused kernel. The abstraction level is frozen at the wrong place. |
| **TorchScript** | A Python subset with dynamic semantics; effectively requires a PyTorch interpreter. Deprecated by its own authors. |
| **StableHLO / XLA** | Excellent at its level, but that level is "post-lowering compiler input"; high-level structure (which weights are an attention layer) is gone. |
| **TFLite / CoreML / OpenVINO** | Vendor IRs with vendor op sets. |
| **GGUF** | No graph at all: architecture is an enum, and the runtime hardcodes the rest. Works brilliantly until a new architecture appears. |

The unifying mistake is **a single abstraction level**. Producers want to write
`attention(q,k,v)`; hardware backends want intent; portable CPU runtimes want
primitives; compilers want machine-level ops. Freezing one level makes the other
three lossy.

## 7.2 Multi-level by design

An OMNI model MAY carry the *same computation* at several levels, linked by
lowering relations:

```
L2  semantic     nn.attention{gqa, rope, causal, window} (q,k,v,rope_cache)
      │  lowering: omni.nn/attention→primitive@1     (declarative, verifiable)
      ▼
L1  primitive    matmul · softmax · mul · add · rope_apply · mask
      │  lowering: fusion, layout selection
      ▼
L0  machine      target-specific fused ops, explicit memory, tiling
```

Rules:

- A `GraphModule` declares its `level` (`semantic` | `primitive` | `machine`) and
  its `dialects`.
- A model MAY provide multiple `GraphModule`s for the same function, related by
  `lowered_from` refs. Only one is canonical (the highest level present); lower
  levels are **derived and droppable** (§00.5).
- A runtime selects the **highest level it fully understands**, which is exactly
  the opposite of the ONNX experience where every backend must reconstruct intent
  from a soup of primitives.
- Lowerings may themselves be objects (declarative rewrite rules, §7.7), so a
  runtime that understands L1 but not `nn.attention` can *apply the shipped
  lowering* and proceed. **This is the key move**: an unknown high-level op is
  recoverable, not fatal, as long as the model ships its lowering.

## 7.3 Structure

```
GraphModule
  ├── dialects[]   : (ns, version, ref-to-DialectRef)
  ├── attrs        : module-level attributes
  ├── functions{}  : name -> Function
  └── entry        : name of the entry function
Function
  ├── signature    : params (name, type), results (type)
  ├── attrs        : e.g. {"kind":"forward","stateful":true}
  └── body         : Region
Region  = [Block]
Block   = (args: [(id, type)], ops: [Op])
Op      = { "d":dialect, "n":name, "v":version,
            "in":[value_id], "out":[(id,type)],
            "attrs":{…}, "regions":[Region], "loc":<SourceLoc> }
```

SSA form: every value has exactly one definition. Values are integers, local to
a function, assigned densely — so a graph is an array of records, not a pointer
graph, and deserializes in one pass with no fixups.

### 7.3.1 Types

```cbor-diag
{"k":"tensor","shape":[ "B", "S", 4096 ],"dtype":{"alias":"bf16"},
 "layout":{…},"device":null}
{"k":"tuple","elems":[…]}
{"k":"list","elem":…}
{"k":"state","id":"kv_cache","spec":{…}}      ; mutable runtime state
{"k":"stream","elem":…}                        ; streaming/online inputs
{"k":"token"}                                  ; ordering token for effects
{"k":"opaque","id":"org.acme/handle"}
```

Symbolic dimensions (`"B"`, `"S"`) are first-class, with a constraint system
(`S <= context.max`, `B >= 1`) attached to the function. Dynamic shapes are the
default, not an afterthought — every serious deployment has them.

### 7.3.2 Effects and state

Pure by default. Ops that touch state (`kv_cache` read/write, RNG, collectives)
declare effects, and ordering among effectful ops is expressed with `token`
values. This makes graph rewriting sound without a whole-program alias analysis
— the mistake that makes optimizing TorchScript miserable.

## 7.4 Dialects

A dialect is a namespaced, versioned set of ops.

| Dialect | Status | Contents |
|---|---|---|
| `omni.core` | **normative, frozen** | `func`, `call`, `return`, `if`, `while`, `scan`, `map`, `yield`, `region`, `constant`, `tuple`, `get`, `assert`, `debug` |
| `omni.tensor` | normative | elementwise math, `matmul`, `reduce`, `gather`/`scatter`, `slice`, `concat`, `pad`, `sort`, `topk`, `cumsum`, `einsum` |
| `omni.nn` | registry, versioned | `attention`, `conv`, `norm` (layer/rms/group/batch), `rope`, `activation`, `embedding`, `moe_route`, `ssm_scan`, `conv1d_causal`, `pool`, `interpolate` |
| `omni.quant` | registry | `quantize`, `dequantize`, `qmatmul`, `fake_quant` |
| `omni.rand` | registry | counter-based RNG, sampling |
| `omni.dist` | registry | `all_reduce`, `all_gather`, `reduce_scatter`, `send`/`recv`, shard annotations |
| `omni.io` | registry | model inputs/outputs, tokenization boundaries, streaming |
| `org.*` / `com.*` | third-party | anything |

> **`core.scan` has two results, and this sentence is here because it once did
> not.** A scan threads a carry and emits a value per step, so it produces the
> final carry *and* the emissions stacked along the scanned axis — what
> `lax.scan` and every other scan in the field produce. The reference
> implementation's op registry declared one result while its interpreter
> returned two, and nothing noticed until a synthesized LSTM used both: the same
> graph then ran correctly and failed verification. An op's arity is exactly the
> kind of thing two halves of one implementation can disagree about silently,
> which is the argument for writing the families down and executing them.
>
> `core.map` has one result: it does not thread a carry, so there is nothing
> else to return.

**`omni.core` is frozen for the life of OMNI/1.x.** Everything else, including
`omni.nn`, can be versioned, deprecated, or replaced without touching the spec.
An architecture is therefore a *dialect + weights*, and a new architecture in
2040 is a new dialect, not a new format.

### 7.4.1 Op versioning

Ops are identified by `(dialect_ns, name, version)`. Version is a monotonically
increasing integer per op, not per dialect — so bumping `nn.attention` from v1 to
v2 does not invalidate every other op, which is precisely ONNX's opset problem.

Compatibility rules:

- Adding an **optional** attribute with a specified default: same version.
- Adding a required input/output, changing semantics, changing a default:
  new version.
- A dialect MUST ship a `DialectRef` object listing, for each op, all versions it
  defines and the rewrite from version *n* to *n+1* when one exists (§7.7). A
  runtime supporting only v2 can then consume a v1 graph automatically.

### 7.4.2 DialectRef

```cbor-diag
{ "t":"omni.ir/dialect", "v":1,
  "ns":"omni.nn", "version":1,
  "ops": {
    "attention": {
      "versions":[1,2],
      "v2":{ "inputs":[{"name":"q","type":"tensor"},…],
             "attrs":{"causal":{"t":"bool","default":false},
                      "window":{"t":"int?|pair","default":null},
                      "softcap":{"t":"f64?","default":null},
                      "kv_groups":{"t":"int","default":1}},
             "shape_fn": {"wasm":[30,h'…'],"export":"attention_shape"},
             "verify_fn":{"wasm":[30,h'…'],"export":"attention_verify"},
             "ref_impl": {"wasm":[30,h'…'],"export":"attention_ref"},
             "lower_to": [ {"level":"primitive","rule":[0,h'…']} ],
             "doc": "…" } } },
  "requires": [{"ns":"omni.tensor","version":1}] }
```

`shape_fn`, `verify_fn` and `ref_impl` as **WebAssembly** modules is the
mechanism that makes a dialect genuinely self-describing (§11.6): a 2076 runtime
can validate and, if necessary, *execute* a 2026 custom op without the 2026
toolchain, because WASM is a frozen, formally-specified instruction set with a
deterministic profile.

Without them, an op from an unknown dialect is **indeterminate** (§15.1), which
is correct and is the weakest true statement a verifier can make. With them it is
*decided*. That is §7.2's key move applied to verification rather than to
execution, and it is the difference between "this reader cannot tell" and "this
graph is well-typed".

#### The calling convention

`shape_fn` and `verify_fn` take the same four arguments and return the same
three outcomes:

```
shape (in: i32, in_len: i32, out: i32, out_cap: i32) -> i32
verify(in: i32, in_len: i32, out: i32, out_cap: i32) -> i32
```

`in` points at canonical OMNI-CBOR of

```cbor-diag
{ "op": <the Op, exactly as §7.3 encodes it>,
  "in": [ <the operands' resolved §7.3.1 types> ] }
```

The op is passed whole rather than summarised, because a dialect's shape often
depends on its attributes and a summary is a decision about which ones matter.

| Return | `shape_fn` | `verify_fn` |
|---|---|---|
| `n > 0` | `n` bytes of CBOR `[<type>…]` written at `out`: the result types | `n` bytes of UTF-8 at `out`: **invalid**, and this is the reason |
| `0` | the op produces no results | **valid** |
| `n < 0` | the function **declines to decide**: indeterminate | the same |

The host allocates `out` through the module's own §11.6 `alloc`, so a module
never assumes a memory layout. A module that traps, exhausts its fuel, or
returns a length past `out_cap` is *indeterminate* as well, with the reason —
never invalid. **A plugin that will not answer says nothing about the graph**,
and reporting one as the other is precisely the conformance violation §15.1
names.

A `shape_fn` that answers decides R-I06 for that op: its result types are
compared with the declared ones exactly as a built-in shape function's are. A
`verify_fn` that objects makes the op **invalid**, in the dialect's own words.

#### `ref_impl`'s calling convention

`ref_impl` is the third slot and the one that *computes* rather than decides. It
takes the same four arguments, for the same reason — the host owns the buffers
and the module owns its memory:

```
ref_impl(in: i32, in_len: i32, out: i32, out_cap: i32) -> i32
```

`in` points at canonical OMNI-CBOR of

```cbor-diag
{ "op": <the Op, exactly as §7.3 encodes it>,
  "in": [ {"shape":[…], "dtype":<§4.3 descriptor>, "data":h'…'} … ] }
```

`data` holds the operand's elements in row-major order as little-endian IEEE
**binary64**, one value per element, whatever the tensor's own `dtype` says. The
dtype travels beside the data rather than in it because a reference
implementation is a correctness oracle, not a storage format: making every
`ref_impl` re-implement §4.3's packing would be asking each dialect author to
reimplement the part of OMNI that is already specified, and getting a different
answer from each of them is the failure this slot exists to prevent. An op that
genuinely depends on the stored representation — a bit-packing, a codebook — is
a §4.7.7 expression plugin, which is handed the bytes.

| Return | `ref_impl` |
|---|---|
| `n > 0` | `n` bytes of CBOR at `out`: `[{"shape":…,"dtype":…,"data":h'…'} …]`, the op's results in order |
| `0` | the op produces no results |
| `n < 0` | the function **declines**: the op is *unrun*, exactly as if there were no `ref_impl` |

The three outcomes of the other two functions apply unchanged: a module that
traps, exhausts its fuel or overruns `out_cap` has declined, with the reason. A
declined `ref_impl` never makes a graph invalid — it leaves the op unexecuted,
which is §15.1's indeterminate and the honest answer.

A runtime MUST prefer a `lower_to` rule to a `ref_impl` when both exist (§7.6's
ordering: a declarative composite runs at native speed and a WASM oracle does
not), and MUST NOT use `ref_impl` for an op it implements natively — the shipped
implementation is a fallback and a reference, not an override. A model cannot
change what `omni.tensor/add` means by shipping WebAssembly for it.

## 7.5 Weights-only models

A `Model` with `tensors` but no `graph` is legal and common (the safetensors
case). Interpretation then depends on `meta.arch.family` plus the runtime's
built-in knowledge. `omni inspect` reports:

```
graph: none (weights-only)
  architecture: transformer.decoder  (dialect omni.nn@1)
  portability: requires a runtime with built-in support for this family
```

This is an honest statement of a real limitation, not a hidden one. `omni graph
synthesize --family transformer.decoder` can generate an OMNI-IR graph from
`arch.params` for the registered families, upgrading a weights-only model to a
self-describing one.

## 7.6 Custom operators

In decreasing order of portability:

1. **Declarative composite** — the op is defined by a `lower_to` rule expressed
   in `omni.core` + `omni.tensor`. Fully portable; any C2 runtime can run it.
2. **WASM reference** — `ref_impl` gives executable semantics, under the calling
   convention in §7.4.2. Portable but slow; suitable as a correctness oracle and
   as a last-resort execution path.
3. **Native plugin** — a runtime-specific kernel identified by name. Not
   portable; the model MUST also provide tier 1 or 2 if it wants to be loadable
   elsewhere, and `omni verify --portable` fails if it does not.

## 7.7 Graph rewriting

Rewrites are **data**: a declarative pattern → replacement, with side conditions.

```cbor-diag
{ "t":"omni.ir/rewrite", "v":1,
  "name":"attention-v1-to-v2",
  "match": { "op":["omni.nn","attention",1], "binds":{"q":0,"k":1,"v":2} },
  "where": [ {"attr":"kv_groups","absent":true} ],
  "emit":  { "op":["omni.nn","attention",2],
             "in":["q","k","v"], "attrs":{"kv_groups":1} },
  "soundness": "semantics-preserving",     ; or "numeric-approximate"
  "tests": [0, h'…'] }
```

Uses: op version migration, dialect lowering, fusion, quantization insertion,
tensor-parallel sharding. Because rewrites are objects, a model can ship the
rules a runtime needs, and a compiler's optimization catalogue becomes portable,
reviewable, and testable rather than compiled-in.

`soundness: "numeric-approximate"` must be declared for rewrites that change
results (fusion changing accumulation order, math-mode relaxations), so that a
deployment with reproducibility requirements can refuse them.

## 7.8 Coverage check across architecture families

The test of "never hardcode architecture assumptions" is whether the core plus a
plausible `omni.nn` covers everything. Sketches:

| Family | How it is expressed |
|---|---|
| Transformer (dense) | `scan` over layers or unrolled; `nn.attention`, `nn.norm`, `tensor.matmul` |
| MoE | `nn.moe_route` produces routing weights + indices; `tensor.gather` over expert weights; ragged `map` region per expert. Expert weights are ordinary tensors with `blocklist` sparsity for partial fetch |
| Mamba / SSM | `nn.ssm_scan` (selective scan, §7.8.1), `nn.conv1d_causal`, `state` type for recurrent carry |
| RWKV | `core.scan` with an explicit recurrent state carry; WKV as a dialect op or as a composite |
| CNN | `nn.conv`, `nn.pool`, `nn.norm` |
| Diffusion / flow matching | The denoiser is a `Model`; the **sampler is a graph**: `core.while` over timesteps with a schedule tensor. Multi-model bundle (§01.7) holds text encoder + denoiser + VAE + scheduler graph |
| RNN / LSTM / GRU | `core.scan` with `state` |
| GNN | `tensor.gather` over edge index tensors; `tensor.scatter{reduction:"add"}` to aggregate per node; ragged types |
| Speech / audio | streaming `stream` types, `nn.conv1d_causal`, chunked `scan` |
| Video | temporal axis in shapes; `scan` over frames; state for caches |
| RL policies | ordinary graph + `omni.io` observation/action typing; value and policy heads as separate functions in one module |
| Retrieval-augmented | `omni.io` external-call op declared as an **effect**, so it is visible and refusable |

### 7.8.1 `nn.ssm_scan`

This op was **named but not defined** for one draft, and the reference
interpreter refused it by name rather than pick a reading — an implementation
that guessed would have been checking its guess against itself. What was missing
was not the arity but the *meaning*: which operand is the state transition and
which the input projection, whether the timestep is an operand or folded into
`A`, and whether the discretization is zero-order hold or bilinear. Those
readings produce different numbers from the same tensors, so two conforming
implementations could have disagreed about what a Mamba model computes while both
passing verification.

```
ssm_scan(u, Δ, A, B, C [, D]) -> y
```

| Operand | Shape | Meaning |
|---|---|---|
| `u` | `[…, L, P]` | the sequence, `P` channels, `L` positions |
| `Δ` | `[…, L, P]` | the timestep, **per channel and position** — this is what makes the model *selective* |
| `A` | `[P, N]` | the state transition, one row of `N` state dimensions per channel |
| `B` | `[…, L, N]` | the input projection, per position |
| `C` | `[…, L, N]` | the output projection, per position |
| `D` | `[P]` | optional skip connection |

Leading dimensions are batch and must agree across `u`, `Δ`, `B` and `C`. The
result has `u`'s shape and dtype.

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `delta_softplus` | bool | `false` | apply `softplus` to `Δ` before anything else |
| `reverse` | bool | `false` | scan from the last position toward the first |

Semantics, stated so that there is one reading:

```
Δ̂  = delta_softplus ? log(1 + exp(Δ)) : Δ
Ā_t = exp(Δ̂_t ⊗ A)                              [L, P, N]   zero-order hold on A
B̄_t = Δ̂_t ⊗ B_t                                 [L, P, N]   Euler on B
h_0 = 0                                          [P, N]
h_t = Ā_t ⊙ h_{t−1} + B̄_t · u_t
y_t = Σ_n C_t[n] · h_t[:, n]   ( + D ⊙ u_t )
```

`Ā` is the zero-order hold `exp(ΔA)`; `B̄` is the Euler form `ΔB` and **not** the
exact hold `(ΔA)⁻¹(exp(ΔA) − I)ΔB`. That asymmetry is not an oversight — it is
what every published implementation computes, and a specification that named the
exact form would describe a model nobody has trained. A future version may add
the exact discretization as a new op version; it must not silently change this
one.

> **The registered arity changed when this was defined, and that is not a
> compatibility break.** The op was declared with three to five operands, which
> was itself part of the underspecification. No conforming implementation could
> have existed to break: the meaning was absent, so any implementation was
> guessing, and §7.4.1's rule protects implementations rather than declarations.
> The arity is corrected in place at version 1 rather than bumped to a version 2
> whose version 1 means nothing.
>
> **`tensor.scatter` aggregates, and this paragraph is here because it once did
> not.** Synthesizing the GNN row found it: `scatter` was defined element for
> element — index *k* of the updates goes to position *k* with the scattered axis
> replaced — so when two edges arrived at the same node the second message
> overwrote the first and the aggregation silently lost it. `scatter` now takes
> an optional
>
> ```
> reduction : "replace" | "add" | "mul" | "max" | "min"     default "replace"
> ```
>
> applied between what is already at the destination and the update arriving
> there. `replace` is the previous behaviour and is the default, so this is an
> optional attribute with a specified default and **`scatter` stays at version
> 1** (§7.4.1). The spelling is ONNX's `ScatterElements(reduction)` rather than a
> new one, because the operation is not new — only its absence here was.
>
> An unrecognised `reduction` is a refusal, not a fallback to `replace`: the two
> produce different numbers from the same graph, which is exactly the case §15.1
> says a reader must report rather than resolve. `gnn.mpnn` now aggregates with
> `scatter{reduction:"add"}` over an edge list, and no longer carries the dense
> `[E, N]` incidence matrix that the workaround needed — a matrix quadratic in
> the thing the operation is sparse in.
| **Unknown, 2040** | New dialect. Core unchanged. |

The load-bearing claim is the last row, and it is supported by the fact that
`omni.core` contains no tensor mathematics at all — only regions, control flow,
and calls.

## 7.9 Serialization

Graphs are canonical OMNI-CBOR like every other structure object, with two
concessions to scale:

1. **Large graphs are split.** A `GraphModule` may reference per-function `Blob`
   objects containing the SSA op array in a fixed-layout binary encoding
   (op record: `dialect_id:u16, op_id:u16, version:u16, n_in:u16, n_out:u16,
   attr_off:u32, in_off:u32`), with a side table for attributes. A 100k-op graph
   parses in a single linear pass with no allocation per op.
2. **Constants never inline.** Any constant above 4 KiB is a tensor ref.

Forward compatibility: an unknown op with known *type signature* can be
skipped for structural validation, printed by tooling, and preserved on rewrite
(`safe-to-copy`, §11.3). Only *execution* requires understanding.

## 7.10 What OMNI-IR deliberately does not do

- **No training semantics in the graph.** Autodiff is a transformation over the
  graph, not a property of it. Gradients, when stored, are tensors (§09).
- **No memory planning, no scheduling, no device placement** in the canonical
  form. Those belong to `machine`-level derived graphs.
- **No Python.** Not a subset, not an embedding, not a fallback.
- **No implicit numerics.** Accumulation dtype, math mode and reduction order are
  explicit attributes where they matter; nothing is inferred from a global flag.

**Prev:** [§06 Metadata](06-metadata.md) · **Next:** [§08 Adapters & Delta Models](08-adapters.md)
