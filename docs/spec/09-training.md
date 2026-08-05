# OMNI/1.0 — §9 Training State

Training state is **optional and separable**. An inference container must not
pay for it — in bytes, in parse time, or in complexity.

## 9.1 Separability requirement

Normative rules:

1. `TrainingState` is referenced from `Model.training`, and every object
   reachable *only* through it MUST be marked with `oflags` such that
   `omni strip --training` removes them by reachability alone, producing a valid
   container with an unchanged set of weight tensors and **identical tensor
   digests**.
2. No inference-relevant object may reference a training object.
3. `omni inspect` reports training state size separately from weights.

A 70 B model checkpoint with Adam states is ~1.7 TB (fp32 params + 2 moments +
grads); the inference artifact is 140 GB. These must never be the same download,
and in OMNI they are the same *object graph* with a different root.

## 9.2 TrainingState

```cbor-diag
{ "t":"omni.train/state", "v":1,

  "framework": {"name":"pytorch","version":"2.9.0",
                "trainer":"megatron-core","trainer_version":"0.14"},
  "step": 128000, "epoch": 2, "samples_seen": 4194304000,
  "tokens_seen": 8589934592000,
  "wall_clock_s": 1382400,

  "optimizer": {
    "kind":"adamw",
    "hyper": {"lr":3e-4,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.1},
    "schedule": {"kind":"cosine","warmup":2000,"total":500000,"min_lr_ratio":0.1},
    "states": [4, h'…'],           ; -> TensorTable: exp_avg, exp_avg_sq, …
    "master_weights": [4, h'…'],   ; -> TensorTable (fp32 master copy, if any)
    "state_dtype": {"alias":"f32"},
    "step_counts": <expr>          ; per-parameter step counters if used
  },

  "gradients": [4, h'…'],          ; optional; usually absent
  "ema": [ {"decay":0.9999,"tensors":[4,h'…'],"step":128000} ],
  "grad_scaler": {"kind":"dynamic","scale":65536.0,"growth_interval":2000},

  "rng": [ … ],                    ; §9.3
  "shards": [14, h'…'],            ; -> ShardMap  §9.4
  "dataloader": { … },             ; §9.5
  "loss_history": <expr>,          ; f32 [steps] tensor
  "config": [0, h'…'],             ; verbatim training config blob
  "ext": {}
}
```

Optimizer states are ordinary `TensorTable`s of ordinary tensors — which means
they are chunked, deduplicated, compressible, quantizable (`add(m_int8_dequant,
…)` for 8-bit optimizers), and *delta-able against the previous checkpoint*. A
training run that writes a checkpoint every 500 steps and whose optimizer moments
change slowly gets substantial chunk-level dedup for free.

## 9.3 RNG state

Reproducibility requires capturing every stream:

```cbor-diag
"rng": [
  {"scope":"global","impl":"pytorch-cpu","state":[0,h'…']},
  {"scope":"cuda","device":0,"impl":"philox","seed":1234,"offset":98304},
  {"scope":"dataloader","worker":3,"impl":"numpy-pcg64","state":[0,h'…']},
  {"scope":"dropout","impl":"counter","key":[1234,0],"counter":8812345},
  {"scope":"jax","impl":"threefry","key":[0,1234]}
]
```

Counter-based generators (Philox, Threefry, ChaCha) are strongly preferred and
are the only ones for which OMNI can promise cross-implementation reproducibility
(`omni.rand`, §07.4). Stateful CPU generators are stored as opaque blobs with
their implementation identified — honest but non-portable, and flagged as such by
`omni verify --reproducible`.

## 9.4 Distributed and sharded checkpoints

The hard requirement: a checkpoint written by 512 ranks under FSDP must be
readable by 8 ranks under tensor parallelism, without a conversion script.

OMNI's answer: **store logical tensors; describe sharding as layout metadata.**

```cbor-diag
; ShardMap (otype 0x000E)
{ "t":"omni.train/shardmap", "v":1,
  "world": {"size":512, "mesh":{"dims":["dp","tp","pp"],"shape":[64,4,2]}},
  "strategy": "fsdp",           ; fsdp|zero1|zero2|zero3|tp|pp|ep|hybrid|megatron
  "placements": {
     "model.layers.0.attn.q_proj.weight": {
        "logical_shape":[4096,4096],
        "sharding":[{"axis":0,"mesh_dim":"tp","parts":4}],
        "shards":[ {"coord":{"tp":0},"range":[[0,1024],[0,4096]],"value":<expr>},
                   {"coord":{"tp":1},"range":[[1024,2048],[0,4096]],"value":<expr>} ] } },
  "flat_params": [ … ]          ; FSDP flat-parameter reconstruction info
}
```

Key properties:

- Each shard is an ordinary tensor expression; the **logical** tensor is
  `concat`/`scatter` of its shards. Therefore *any* resharding is an expression
  rewrite plus a range-driven read (§04.7.4), and a reader that wants rows
  2048–3071 fetches exactly the chunks holding them, regardless of how the writer
  sharded.
- FSDP's flat-parameter buffers (the thing that makes FSDP checkpoints
  notoriously non-portable) are described by their `(param, offset, numel,
  orig_shape)` table, so the flat buffer is a `literal` and each parameter is a
  `reshape(slice(flat, …))`. **Zero copy, zero conversion, full portability.**
- Writing is embarrassingly parallel: each rank writes its own objects and its
  own segment; a coordinator writes the superblock. Because objects are
  content-addressed, ranks writing *identical* shards (replicas under DP)
  deduplicate automatically — a property no existing distributed checkpoint
  format has, and one that removes the usual "rank 0 writes replicated tensors"
  special case.

### 9.4.1 Framework mapping

| Framework | Mapping |
|---|---|
| **PyTorch DCP** | `ShardMap.strategy` ∈ {fsdp, zero*}; DCP's `ChunkStorageMetadata` maps 1:1 onto `shards[].range` |
| **DeepSpeed ZeRO 1/2/3** | partition of optimizer states / grads / params along `dp`; `flat_params` for the fused buffers |
| **Megatron-LM** | `tp`/`pp`/`ep` mesh dims; layer→pipeline-stage map in `placements` |
| **NeMo** | Megatron mapping + NeMo config preserved as a `Foreign` blob |
| **JAX / Orbax** | `mesh` maps directly to `jax.sharding.Mesh`; `NamedSharding` specs become `sharding[]`; PyTree structure stored as a `Metadata` key |
| **TensorFlow** | variable name → tensor; `SaveSliceInfo` → `shards[].range` |

### 9.4.2 Resharding

`omni reshard ckpt.omni --mesh dp=8,tp=8 -o new.omni` rewrites only the
`ShardMap` — **no tensor bytes move** if the underlying chunking permits the new
ranges (which `fixed` chunking with a divisor-friendly size usually does). When
bytes must move, only the affected chunks are rewritten, and unchanged chunks are
shared with the original checkpoint.

## 9.5 Dataloader state

```cbor-diag
"dataloader": {
  "kind":"streaming", "position":{"shard":41,"offset":9182734},
  "seed":1234, "shuffle_buffer":10000, "epoch":2,
  "consumed_digest": h'…',       ; digest of the consumed-sample bitmap
  "sample_bitmap": [0, h'…']     ; optional exact resumption
}
```

Exact resumption of a data stream is the difference between "we restarted the run"
and "we restarted the run and it is statistically the same run".

## 9.6 Checkpoint chains

Checkpoints are ordinary manifests with `parents[]` pointing at the previous
checkpoint and a `Provenance` recording the step delta. A training run is then a
**chain of content-addressed manifests** — a Git history for model weights:

```
omni log run.omnid
  step 128000  2026-08-04T11:02Z  loss 1.842  Δ 4.1 GB  (of 1.7 TB)
  step 127500  2026-08-04T10:31Z  loss 1.849  Δ 4.3 GB
  step 127000  …
```

`Δ` is genuinely that small when optimizer moments are stored in bf16/fp8 and
chunk-level dedup catches the slow-moving portions. `omni gc --keep-every 10
--keep-last 5` implements retention policy at object granularity, which is
strictly better than deleting whole checkpoint directories.

## 9.7 Gradients and activations

- **Gradients** are stored only when explicitly requested (`--with-grads`); they
  are almost never worth persisting and their presence is reported prominently.
- **Activations / KV snapshots** are `RuntimeCache` objects (§10.6), never
  canonical. A KV-cache snapshot for prefix reuse is legal, droppable, and
  keyed by the digest of (model plan, token prefix).

## 9.8 What OMNI does not standardize here

Optimizer *algorithms* are not defined by OMNI: `kind: "adamw"` plus `hyper` is a
label and a parameter bag, interpreted by the framework. Attempting to specify
optimizer semantics would (a) be a research-tracking treadmill, and (b) provide
no interoperability benefit, since resuming a run in a different framework is
rarely meaningful even when the tensors transfer. What OMNI *does* guarantee is
that the tensors, their sharding, their RNG streams and their step counters
transfer losslessly — which is the part that is currently broken.

**Prev:** [§08 Adapters](08-adapters.md) · **Next:** [§10 Runtime & Capability Negotiation](10-runtime.md)
