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
2. **WASM reference** — `ref_impl` gives executable semantics. Portable but
   slow; suitable as a correctness oracle and as a last-resort execution path.
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
| Mamba / SSM | `nn.ssm_scan` (associative scan) — **underspecified, see below** — `nn.conv1d_causal`, `state` type for recurrent carry |
| RWKV | `core.scan` with an explicit recurrent state carry; WKV as a dialect op or as a composite |
| CNN | `nn.conv`, `nn.pool`, `nn.norm` |
| Diffusion / flow matching | The denoiser is a `Model`; the **sampler is a graph**: `core.while` over timesteps with a schedule tensor. Multi-model bundle (§01.7) holds text encoder + denoiser + VAE + scheduler graph |
| RNN / LSTM / GRU | `core.scan` with `state` |
| GNN | `tensor.gather` over edge index tensors; aggregation is **not** `scatter` — see below; ragged types |
| Speech / audio | streaming `stream` types, `nn.conv1d_causal`, chunked `scan` |
| Video | temporal axis in shapes; `scan` over frames; state for caches |
| RL policies | ordinary graph + `omni.io` observation/action typing; value and policy heads as separate functions in one module |
| Retrieval-augmented | `omni.io` external-call op declared as an **effect**, so it is visible and refusable |

> **Known gap: `nn.ssm_scan` is named but not defined.** Writing a reference
> interpreter (`omni graph run`) surfaced this. The op's arity and attributes are
> registered — three to five operands and `delta_softplus` — but nothing here says
> which operand is the state transition and which the input projection, whether
> the timestep is an operand or already folded into `A`, or whether the
> discretization is zero-order hold or bilinear. Those readings produce different
> numbers from the same tensors, so two conforming implementations could disagree
> about what a Mamba model computes while both passing verification.
>
> Every other op in `omni.nn` is either pinned by a shape function or standard
> across every framework. This one is neither, and the reference implementation
> **refuses it by name** rather than picking a reading — an implementation that
> guessed would then be checking its guess against itself. Defining it is
> outstanding specification work, not outstanding implementation work.
>
> **Known gap: `tensor.scatter` cannot aggregate.** Synthesizing the GNN row
> surfaced this one. `scatter` is defined element for element — index *k* of the
> updates goes to position *k* with the scattered axis replaced — so when two
> edges arrive at the same node, the second message overwrites the first and the
> aggregation silently loses it. Message passing needs a scatter-*add*, and §07
> defines no reduction on `scatter`; ONNX spells it `ScatterElements(reduction)`
> and JAX spells it `segment_sum`, so the shape of the fix is not in doubt, only
> its spelling. Until it is spelled, `gnn.mpnn` takes the incidence matrix as an
> input and aggregates with `einsum`, which is the same arithmetic in an op that
> exists — and is dense where the operation is sparse, which is the cost of the
> gap.
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
