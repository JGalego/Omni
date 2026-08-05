# OMNI/1.0 — §10 Runtime Interface and Capability Negotiation

Layer L4. The canonical model is hardware-independent (§00.2). This section
defines how a *specific* runtime on *specific* hardware gets a representation it
can execute, and where the resulting artifacts live.

---

## 10.1 The negotiation problem

A runtime can do some things and not others. A model can be represented in many
ways. Today the matching is done by humans: you download `model-Q4_K_M.gguf`
because someone on a forum said it fits your GPU.

OMNI makes it a function:

```
resolve : (ModelDAG, CapabilitySet, Objective) → Plan | Failure(reasons)
```

`resolve` is pure, deterministic, and cheap (it touches metadata and tensor
descriptors, never tensor bytes). Its output, a `Plan`, is content-addressed —
so a plan computed once is reusable, cacheable and shareable.

## 10.2 CapabilitySet

```cbor-diag
{ "t":"omni.rt/capability", "v":1,
  "runtime": {"name":"vllm","version":"0.11.0"},
  "profiles": ["C0","C1","C2","C4"],

  "dtypes": {
    "compute": ["bf16","f16","f8e4m3"],
    "storage": ["bf16","f16","f8e4m3","int8","int4","f32"],
    "accumulate": ["f32"]
  },
  "quant_schemes": ["affine","sym","codebook","nested"],
  "layouts": ["strided","packed","blocked-scaled"],
  "sparsity": ["nm:2:4","bitmask"],

  "dialects": [{"ns":"omni.core","v":1},{"ns":"omni.tensor","v":1},
               {"ns":"omni.nn","v":1,"ops":{"attention":[2],"ssm_scan":[]}}],
  "graph_levels": ["semantic","primitive"],

  "features": ["omni.adapt/lora.1","omni.adapt/dora.1",
               "omni.tensor/expr.1","omni.stream/http-range.1"],
  "unsupported": ["omni.tensor/f4.1","omni.rt/kv-snapshot.1"],

  "devices": [ {"kind":"gpu","vendor":"nvidia","arch":"sm_90",
                "count":8,"memory":85899345920,
                "interconnect":"nvlink","fp8":true,"sparsity_2_4":true} ],
  "budget": {"memory_bytes":687194767360,"host_memory_bytes":2199023255552,
             "disk_bytes":10995116277760,"load_time_ms":60000},
  "policy": {"allow_lossy":false,"allow_plugins":["omni.*"],
             "allow_native_caches":false,"require_signature":true,
             "max_materialize_bytes":137438953472}
}
```

Notes:

- `unsupported` is explicit and distinct from "not listed". "Not listed" means
  *unknown*; "unsupported" means *do not attempt*. The distinction matters when
  a resolver must decide whether to try a representation optimistically.
- `policy` is where a deployment expresses trust decisions — notably
  `allow_native_caches: false`, which refuses precompiled engine blobs (§12.3).
- A runtime publishes its CapabilitySet as an object; `omni caps` emits one, and
  `omni plan --caps caps.cbor model.omni` resolves offline. This means a CI job
  can answer "will this model run on our fleet?" without the fleet.

## 10.3 Feature-conditional values

A tensor's value may branch on capability:

```cbor-diag
{"op":"select", "on":"omni.dtype/f8e4m3.1",
 "then": <cast(W, f8e4m3)>,
 "else": <W_bf16>}
```

`select` is resolved statically by the resolver — it never appears in an
executable plan. This lets a publisher express "use fp8 where available" once,
in the model, rather than shipping two models.

## 10.4 Plan

```cbor-diag
{ "t":"omni.rt/plan", "v":1,
  "model": [1, h'…'],                 ; the manifest this realizes
  "caps_digest": h'…',                ; the capability set it was resolved against
  "objective": "min-memory",          ; min-memory|max-quality|min-latency|
                                      ; min-load-time|balanced
  "tensors": {                        ; chosen expression per tensor
     "model.layers.0.attn.q_proj.weight":
        {"expr": <expr>, "materialize":"cache", "dtype":"f8e4m3",
         "bytes": 8388608, "cache": [15, h'…']} },
  "graph": [8, h'…'],                 ; chosen GraphModule (level)
  "adapters": [ [12, h'…'] ],
  "rewrites": [ [0, h'…'] ],
  "totals": {"resident_bytes":6.7e10,"materialize_bytes":0,
             "est_load_ms":4200,"est_quality_delta":-0.004},
  "warnings": ["lm_head kept at bf16: int8 quantization exceeded --max-err"],
  "unmet": []                          ; non-empty ⇒ this plan is infeasible
}
```

A `Plan` is:

- **verifiable** — a third party can re-run the resolver and get the same plan;
- **auditable** — it says exactly which representation of every tensor will be
  used, and its warnings are the honest list of compromises;
- **cacheable** — keyed by `H(model_digest ‖ caps_digest ‖ objective)`;
- **shippable** — a registry can precompute plans for the ten common
  configurations and include them in the manifest (`realizations[]`), so a
  client does no work at all.

## 10.5 The resolution algorithm

Deterministic, and specified so that two implementations agree:

```
resolve(model, caps, objective):
  1. FEATURE GATE
     for f in model.features.required:
         if f ∉ caps.features and f ∉ implied(caps): FAIL(f)
     optional features not in caps are simply disabled.

  2. GRAPH SELECTION
     levels = model.graphs ordered by level descending (semantic > primitive > machine)
     pick the highest level L such that every op in it is in caps.dialects
        (allowing shipped lowerings: if op unknown but a lower_to rule exists and
         its target ops are supported, accept and record the rewrite)
     if none: if model is weights-only and caps has built-in support for
              meta.arch.family → accept; else FAIL

  3. ADAPTER BINDING
     for each requested adapter: check base digest, run attach validation
     if a required adapter cannot bind: FAIL

  4. PER-TENSOR REPRESENTATION
     for each tensor t:
        candidates = enumerate(t.value)      # §10.5.1
        feasible   = { c ∈ candidates | dtype(c) ∈ caps.storage
                                      ∧ layout(c) ∈ caps.layouts
                                      ∧ scheme(c) ∈ caps.quant_schemes
                                      ∧ (¬lossy(c) ∨ policy.allow_lossy) }
        if feasible = ∅: FAIL(t)
        choose argmin/argmax over feasible by objective:
           min-memory   : resident bytes,          tie-break by est. quality
           max-quality  : est. quality,            tie-break by bytes
           min-load-time: bytes transferred + materialization cost
           min-latency  : compute dtype match with device, then bytes
           balanced     : normalized weighted sum with published weights

  5. BUDGET CHECK
     Σ resident bytes ≤ caps.budget.memory_bytes
     if exceeded: retry step 4 with objective=min-memory over the
        largest-first tensor list until it fits; if still exceeded: FAIL(budget)

  6. MATERIALIZATION PLANNING
     for each chosen expr: decide direct-map | materialize-on-load | cache-to-disk
        direct-map is chosen when the expr is a bare literal with a compatible
        layout — the zero-copy path
  7. EMIT Plan
```

### 10.5.1 Candidate enumeration

`enumerate(expr)` walks the expression tree and yields every node that is a
*valid standalone representation* of the tensor — i.e. every node whose value
equals the tensor's value up to a declared tolerance. Concretely: the root, plus
any node reachable through `cast`/`quantize`/`dequantize`/`select`/`approx`
chains, each annotated with its dtype, layout, storage cost and estimated quality
delta. This is why the algebra is small and pure: enumeration is a five-line tree
walk with no special cases.

### 10.5.2 Failure is informative

`FAIL` returns structured reasons, not a boolean:

```
$ omni plan model.omni --caps edge-npu.cbor
INFEASIBLE
  ✗ required feature omni.quant/nested.1 not supported by runtime
      → affects 226 tensors (6.7 GB)
      → remedy: `omni convert --requantize affine-int8` (est. quality -0.011 MMLU)
  ✗ budget: min feasible resident 11.4 GB > device memory 8.0 GB
      → remedy: `--objective min-memory --allow-lossy` yields 6.1 GB
  ✓ graph: primitive level satisfied
  ✓ tokenizer: satisfied
```

Actionable diagnostics are a feature of the format, not of a tool: they are
possible because the model declares what it needs in machine-readable form.

## 10.6 Runtime caches

```cbor-diag
{ "t":"omni.rt/cache", "v":1,
  "kind": "materialized-tensor",   ; materialized-tensor | trt-engine | coreml-package |
                                   ; openvino-ir | mlx-graph | kernel-autotune |
                                   ; kv-prefix | tokenizer-trie | plugin
  "key": h'…',                     ; digest of what produced it
  "produced_from": [ [5,h'…'], [17,h'…'] ],
  "target": {"vendor":"nvidia","arch":"sm_90","driver":">=560,<600",
             "runtime":{"name":"tensorrt","version":"10.4"},
             "os":"linux","abi":"gnu"},
  "payload": [6, h'…'],            ; -> ChunkList
  "validity": {"expires":null,"invalidated_by":["driver-major"]},
  "size": 4294967296,
  "executable": true,              ; §12.3 — this is CODE
  "signature": [18, h'…'],
  "reproducible": false,
  "build": {"cmd":"trtexec …","time_s":1840,"nondeterministic":true} }
```

Normative rules:

1. A cache object MUST set `oflags.CACHEABLE`. Deleting every cacheable object
   MUST leave a valid, semantically identical model.
2. A cache MUST record `key` = the digest of its inputs. A consumer MUST verify
   that `key` matches the plan it is about to execute; a mismatch means the cache
   is stale and MUST be ignored.
3. `executable: true` caches (TensorRT engines, compiled kernels, `.mlmodelc`)
   are **code from an untrusted file**. A runtime MUST NOT load one unless
   `policy.allow_native_caches` is set *and* the cache carries a signature the
   deployment trusts. Default is refuse. See §12.3.
4. Caches never participate in the model's identity. Two containers differing
   only in caches denote the same model, and `omni digest --canonical` returns
   the same value for both.

This is the mechanism that lets a container carry a TensorRT engine for the
fleet's GPUs *and* remain a portable, hardware-independent artifact — the engine
is an attachment with a verified provenance link, not the model.

## 10.7 Runtime loading interface

The reference API (§SDK) is deliberately narrow:

```rust
let store = Store::open("model.omni")?;              // mmap, 2 reads
let model = store.root_model()?;                     // parse manifest+meta
let caps  = Capabilities::detect()?;                 // or load from file
let plan  = model.resolve(&caps, Objective::MinLatency)?;
let inst  = plan.instantiate(&store)?;               // materializes / maps
for t in inst.tensors() {
    let view: TensorView = t.map()?;                 // zero-copy where possible
}
```

`instantiate` is where policy meets reality: it maps direct-map tensors, spawns
materialization for the rest (parallel, chunk-granular, with progress), consults
and populates caches, and returns a handle whose lifetime keeps the mapping
alive. Nothing above `instantiate` knows about files.

## 10.8 Progressive readiness

For streaming (§13), `instantiate` reports readiness incrementally:

```
ready(embed_tokens)          @ 1.2 s   (3.1 % downloaded)
ready(layers 0..3)           @ 4.8 s   (12 %)
ready(layers 0..31, lm_head) @ 41 s    (100 %)
first-token-possible         @ 4.8 s   (with layer-wise streaming execution)
```

Because tensor dependencies are explicit and load order is declared
(`TensorTable.order`), a runtime can begin prefill on layer 0 while layer 31 is
still in flight. The format supplies the ordering; the runtime supplies the
overlap.

---

**Prev:** [§09 Training](09-training.md) · **Next:** [§11 Plugin System](11-plugins.md)
