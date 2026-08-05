# OMNI/1.0 — §8 Adapters, Deltas, and Model Inheritance

Adapters and deltas are the same mechanism viewed at two scales: both are
**expressions over a parent's tensors** (§04.7). Nothing is merged, nothing is
copied.

---

## 8.1 Adapter object

```cbor-diag
{ "t":"omni.adapt/adapter", "v":1,

  "method": "lora",             ; lora|dora|ia3|loha|lokr|vera|adalora|bone|
                                ; bitfit|prompt|prefix|p-tuning|adapter-bottleneck|
                                ; control-vector|plugin
  "base": [1, h'…'],            ; -> Manifest of the base model (REQUIRED)
  "base_compat": {              ; what the adapter assumes about the base
     "tensors": { "model.layers.*.attn.q_proj.weight":
                    {"shape":[4096,4096],"axes":["out","in"]} },
     "arch_digest": h'…'        ; digest of base meta.arch, for fast checking
  },

  "rank": 16, "alpha": 32, "dropout": 0.0,
  "targets": ["*.attn.q_proj.weight","*.attn.v_proj.weight"],

  "tensors": [4, h'…'],         ; -> TensorTable holding A/B (or m, scale, …)

  "attach": [ … ],              ; §8.3 — the binding rules
  "scale_default": 1.0,
  "merge_policy": "runtime",    ; runtime|materialize|either
  "trained_on": [26, h'…'],     ; -> Dataset
  "provenance": [19, h'…'],
  "ext": {}
}
```

An adapter is a first-class publishable artifact: a `Manifest` with
`kind: "adapter"` whose only asset is an `Adapter`. It is typically 10–200 MB
against a 140 GB base, and it references the base by digest, so "which model is
this adapter for?" is verifiable rather than a filename convention.

## 8.2 Adapter methods as expressions

All of these use only core nodes:

| Method | Expression on target `W` |
|---|---|
| **LoRA** | `add(W, scale(matmul(B, A), α/r))` |
| **DoRA** | `mul(scale_col(add(W, scale(matmul(B,A), α/r))), div(m, norm(add(W, …), axis=0)))` — i.e. direction from LoRA-updated weight, magnitude from learned `m` |
| **IA³** | `mul(W, l)` where `l` broadcasts over one axis (or applied to activations at graph level) |
| **LoHa** | `add(W, scale(mul(matmul(B₁,A₁), matmul(B₂,A₂)), α/r))` (Hadamard product of two low-rank terms) |
| **LoKr** | `add(W, scale(kron(B,A), α/r))` — `kron` as a composite of `reshape`+`mul`+`reshape` |
| **VeRA** | `add(W, scale(matmul(mul(B, d), mul(A, b)), α/r))` with `A`,`B` **shared frozen random** matrices referenced from a *parent* adapter — dedup makes this nearly free |
| **AdaLoRA** | LoRA with a learned per-rank gate: `add(W, scale(matmul(mul(B,Λ), A), α/r))` |
| **BitFit** | `add(b, Δb)` on bias tensors only |
| **Bottleneck adapters** | *Graph-level*: new ops inserted into the module (§8.4) |
| **Prompt / prefix / P-tuning** | *Graph-level*: extra virtual tokens or per-layer KV prefixes, stored as tensors and bound to graph inputs (§8.4) |
| **Control vectors** | *Activation-level*: `add(h, scale(v, strength))` at declared graph points |

The first eight require **no format extension at all** — they are arithmetic. The
last three need graph-level attachment, which §8.4 provides.

## 8.3 Attachment rules

An adapter must be attachable to a base it has never seen at build time, without
string-matching fragility.

```cbor-diag
"attach": [
  { "select": {"glob":"model.layers.*.attn.q_proj.weight"},
    "kind":"tensor-transform",
    "apply":  {"op":"add",
               "with":{"op":"scale",
                       "x":{"op":"matmul","a":"$B","b":"$A"},
                       "k":[30,["ratio",32,16]]}},
    "bind": { "$A":"lora.{1}.q_proj.A", "$B":"lora.{1}.q_proj.B" },
    "require": {"axes":["out","in"], "rank_axis":"in"} }
]
```

- `select` matches base tensors by glob, regex, `semantic`, `role`, or `axes`
  pattern. Selecting by **`role` + `axes`** rather than by name is the robust
  option and is what tooling should emit: it survives renaming between model
  releases.
- `{1}` is the first wildcard capture, used to index the adapter's own tensors.
- `require` states the assumptions; a mismatch is a hard, early error with a
  clear message instead of silently wrong math.

Attachment is validated by `omni adapter check base.omni lora.omni`, which
reports unmatched selectors, shape mismatches, and any base tensor the adapter
expected but the base lacks — before any weights are loaded.

## 8.4 Graph-level adapters

Prompt tuning, prefix tuning and bottleneck adapters change the *computation*,
not just weights. They are expressed as graph rewrites (§07.7) shipped with the
adapter:

```cbor-diag
"graph_patches": [
  { "t":"omni.ir/rewrite", "v":1,
    "name":"prefix-kv",
    "match": {"op":["omni.nn","attention",2], "binds":{"k":1,"v":2}},
    "emit":  {"op":["omni.nn","attention",2],
              "in":[ "q",
                     {"op":["omni.tensor","concat"],"in":["$prefix_k","k"],"attrs":{"axis":2}},
                     {"op":["omni.tensor","concat"],"in":["$prefix_v","v"],"attrs":{"axis":2}} ]},
    "bind": {"$prefix_k":"prefix.{layer}.k", "$prefix_v":"prefix.{layer}.v"} }
]
```

Because rewrites are declarative objects, a runtime applies them mechanically. No
runtime needs to know what "prefix tuning" is.

## 8.5 Composition

Multiple adapters compose. Composition order matters and is therefore explicit.

```cbor-diag
"compose": {
  "order": ["safety","domain-medical","style-terse"],
  "mode": "sequential",           ; sequential | parallel-sum | ties | dare | slerp
  "weights": [1.0, 0.7, 0.3],
  "conflicts": "error"            ; error | last-wins | sum
}
```

| Mode | Semantics |
|---|---|
| `sequential` | apply transforms in order; each sees the previous result |
| `parallel-sum` | `W + Σ wᵢ·Δᵢ` — the usual multi-LoRA case |
| `ties` | TIES-merging: trim, elect sign, disjoint mean |
| `dare` | DARE: random drop + rescale (`seed` required for reproducibility) |
| `slerp` | spherical interpolation between two deltas |
| `plugin` | anything else |

`ties`, `dare` and `slerp` are *merge algorithms*, and OMNI expresses them as
expressions with declared seeds so that a merged model is **reproducible from its
parents** — which turns today's untraceable "merge soup" models into artifacts
with a verifiable recipe. `omni merge --recipe recipe.cbor` writes a manifest
whose tensors are merge expressions and whose storage is zero beyond the parents.

## 8.6 Delta models and inheritance

A *delta model* is a `Manifest` with `parents[]` whose tensors are expressions
over the parents' tensors.

```
foundation-8b            140 GB  (all chunks)
  └─ instruct-delta        1.1 GB  (only changed chunks + LoRA factors)
       └─ code-delta       0.9 GB
            └─ math-delta  0.4 GB
                 └─ medical-delta 0.6 GB
```

Total for all five: **143 GB**, not 700 GB. Five separate published artifacts
today: 700 GB, with no way to tell they are related.

Delta representations, chosen per tensor by `omni delta`:

| Representation | When | Expression |
|---|---|---|
| **identical** | tensor unchanged | reference the parent's `TensorDesc` directly — zero bytes |
| **chunk-level** | few chunks changed (common after continued pretraining on a subset) | new `ChunkList` reusing unchanged chunk refs — cost is only the changed chunks |
| **low-rank** | the change is (approximately) low-rank, e.g. it came from LoRA | `add(parent, scale(matmul(B,A), k))` |
| **sparse** | few weights changed materially | `add(parent, sparse(bitmask, values))` |
| **quantized-residual** | dense small change | `add(parent, dequantize(int8_residual, {…}))` — typically 4–8× smaller than a full copy, and *lossy only in the residual*, which is declared |
| **full** | everything changed | ordinary tensors |

`omni delta base.omni tuned.omni -o delta.omni` analyzes each tensor, picks the
cheapest representation subject to an error bound (`--max-err`), and reports:

```
tensors: 291
  identical              :  96   0 B
  chunk-level            :  12   184.5 MB
  low-rank (r≤32)        : 160   612.3 MB   max rel-err 0.0 (exact: source was LoRA)
  quantized-residual int8:  23   287.1 MB   max rel-err 3.2e-3
  full                   :   0   0 B
total delta: 1.06 GB   (0.76 % of base)
```

**Recursive resolution.** Reading a tensor from `medical-delta` walks the parent
chain until it hits `literal`s. Chain depth is bounded (`≤ 32` by default,
declared in the manifest) and a container MAY *flatten* a chain for distribution
(`omni flatten --depth 1`) while keeping the provenance links.

## 8.7 Parent resolution and pinning

```cbor-diag
"parents": [ { "ref":[1,h'…'],
               "role":"base",
               "name":"acme/llm-8b",
               "locators":["oci://ghcr.io/acme/llm-8b@sha256:…",
                           "hf://acme/llm-8b"],
               "required": true } ]
```

- Parents are pinned **by digest**. A delta can never silently attach to a
  different base — the failure is loud and immediate.
- `locators` are advisory (§01.4). A delta container with `required: true` and no
  resolvable parent is `incomplete`, and `omni inspect` says so on the first
  line.
- A container MAY **inline** its parent (`omni pack --bundle-parents`) to
  produce a self-contained artifact; dedup means this costs nothing when the
  parent is already present elsewhere in the same store.

## 8.8 Adapters and quantization interact correctly

The historically painful case — "QLoRA on a 4-bit base" — is ordinary here:

```
W_base_int4 = literal(...)                       ; 4-bit, stored once
W_base      = dequantize(W_base_int4, {…})       ; bf16, derived
W_tuned     = add(W_base, scale(matmul(B,A), α/r))
W_serve_int4= quantize(W_tuned, {…}, "rne")      ; re-quantized, derived+cached
```

A runtime that wants to serve int4 with the adapter merged materializes
`W_serve_int4` once and caches it (§10.6); a runtime that wants to apply the
adapter at runtime against the int4 base keeps `W_base_int4` mapped and applies
the low-rank term in the kernel. **Both read the same file.** Today these are two
different artifacts produced by two different toolchains.

## 8.9 Multi-tenant serving

A serving stack holding one base and 200 LoRAs benefits directly:

- one `mmap` of the base container, shared by all tenants;
- 200 small adapter containers, each ~50 MB, all referencing the base digest;
- attachment validated at load, not at first token;
- per-request adapter selection is `select(feature, expr_a, expr_b)` or simply
  choosing a different `Plan` (§10.4);
- adding a tenant is `omni pull` of one small object set, with dedup against
  everything already local.

## 8.10 Limits and honesty

- **Low-rank delta extraction is lossy** unless the change was genuinely
  produced by a low-rank update. `omni delta` reports measured error per tensor
  and refuses to exceed `--max-err` silently; the resulting expression is wrapped
  in `approx()` (§04.7.2) so the loss is visible in the DAG forever.
- **Chunk-level dedup across fine-tunes is usually poor** for fully fine-tuned
  models: every weight changes, so no chunk matches. This is a fact about full
  fine-tuning, not a format limitation, and it is why the low-rank and
  quantized-residual representations exist. Expect large dedup wins for
  LoRA-derived, partially-frozen, or continued-pretraining models, and near-zero
  chunk dedup for full-parameter fine-tunes — where the quantized-residual path
  still gives 4–8×.
- **Deep parent chains cost latency** on first load (one index lookup per level,
  all cheap, but network round trips if parents are remote). Flatten for
  distribution; keep the chain for provenance.

---

**Prev:** [§07 Execution Graph](07-graph.md) · **Next:** [§09 Training State](09-training.md)
