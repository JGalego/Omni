# OMNI/1.0 — §0 Architecture Overview

**Status:** Draft
**Normative keywords:** MUST, MUST NOT, SHOULD, SHOULD NOT, MAY are to be
interpreted as in RFC 2119 / RFC 8174.

## 0.1 The problem

Every model format in use today conflates things that are logically independent:

| Format | Conflates |
|---|---|
| PyTorch `.pt` | model + Python object graph + arbitrary code |
| safetensors | weights + a flat name→range map, no graph, no semantics |
| GGUF | weights + quantization + architecture enum + tokenizer + hyperparameters, all in one hand-maintained schema |
| ONNX | graph + weights + a globally versioned opset that cannot be extended without committee approval |
| TensorRT engine | graph + weights + quantization + *one GPU architecture* + *one driver version* |
| CoreML `.mlpackage` | graph + weights + one vendor's runtime |

The consequences are structural, not cosmetic:

1. **N × M explosion.** A model published in 4 quantizations × 6 runtimes is 24
   full copies of the same weights.
2. **Metadata is lost at every hop.** License, provenance, RoPE parameters, chat
   template, evaluation results — each format keeps a different subset, and each
   conversion silently drops the rest.
3. **Nothing is verifiable.** A `.gguf` downloaded from a mirror is
   indistinguishable from a backdoored one without an out-of-band hash, and no
   format carries a signature or a provenance chain natively.
4. **Extension requires a new format.** Every new architecture (Mamba, RWKV,
   MoE routing, diffusion schedulers) requires either a spec revision or an
   out-of-band convention.
5. **The unit of transfer is the file.** Resuming, deduplicating, patching or
   partially loading a 400 GB model requires bespoke tooling above the format.

OMNI's thesis is that these are all the *same* problem: the format has no
**object model** underneath it.

## 0.2 The layer model

OMNI is defined as a stack of independent planes. Each may be implemented,
transported, versioned and cached separately. This separation is the
specification's single most important property; every other feature is a
consequence of it.

```
┌──────────────────────────────────────────────────────────────────────┐
│  L4  NEGOTIATION      capability sets, plans, realization selection  │  §10
│      "given this runtime, which representation should be used?"      │
├──────────────────────────────────────────────────────────────────────┤
│  L3  SEMANTIC         OMNI-IR graphs, dialects, tokenizers, metadata │  §06 §07 §11
│      "what does this model mean?"                                    │
├──────────────────────────────────────────────────────────────────────┤
│  L2  VALUE            tensor expressions, dtypes, layouts, quant     │  §04 §05 §08
│      "what are the numbers, and how are they computed?"              │
├──────────────────────────────────────────────────────────────────────┤
│  L1  OBJECT           immutable content-addressed objects, DAG, refs │  §01
│      "what is the identity of each piece?"                           │
├──────────────────────────────────────────────────────────────────────┤
│  L0  CONTAINER        bytes: header, segments, index, alignment      │  §02 §03
│      "how is this laid out on a disk, a socket, or S3?"              │
└──────────────────────────────────────────────────────────────────────┘
```

Rules that make the separation real, not decorative:

- **L0 is replaceable.** The `.omni` file is *one* serialization of an L1 object
  graph. A directory store, an OCI registry, an S3 prefix and an HTTP endpoint
  are equally valid L0 substrates. Nothing at L1 or above may depend on file
  offsets. *Consequence: the container can be revised in 2050 without
  invalidating any model.*
- **L1 knows nothing about tensors.** It stores typed, hashed byte strings and
  their references. *Consequence: new object types are additive forever.*
- **L2 knows nothing about architectures.** It knows numbers, shapes, layouts
  and a closed algebra of pure transformations. *Consequence: quantization
  schemes and adapter methods invented later need no format change.*
- **L3 knows nothing about hardware.** *Consequence: the canonical model is
  hardware-independent by construction, not by convention.*
- **L4 is pure function.** Given (model DAG, capability set) it produces a plan.
  It stores nothing that cannot be recomputed. *Consequence: every hardware
  artifact is a cache and may be deleted at any time without loss.*

## 0.3 Borrowed ideas

OMNI is deliberately unoriginal where prior art is strong:

| Borrowed from | Idea | Where used |
|---|---|---|
| **Git** | content-addressed immutable object DAG; packs | §01, §13 |
| **OCI** | manifest → layers; registries as CAS; media types | §01.9, §13.5 |
| **PNG** | criticality/safe-to-copy bits on unknown chunks | §11.3 |
| **ELF** | fixed header + section table + alignment for `mmap` | §02 |
| **Parquet** | footer index, column chunks, predicate-free random access | §02.6 |
| **MLIR** | dialects, progressive lowering, multi-level IR | §07 |
| **Matroska/EBML** | self-describing extensible element tree | §03.1 |
| **LLVM bitcode** | stable serialized IR with forward-compat wrappers | §07.9 |
| **ZIP** | end-of-file directory for append and single-seek open | §02.7 |
| **Bao/BLAKE3** | verified streaming of arbitrary byte ranges | §13.3 |
| **in-toto / SLSA** | attestations as first-class signed objects | §12.6 |
| **DLPack** | zero-copy tensor handoff across frameworks | SDK |
| **TUF** | repository-level key rotation and freshness | §12.7 |

The novel contributions are: the **tensor expression algebra** (§04.7) that makes
quantizations and adapters transformations rather than copies; the **multi-level
dialect IR** (§07.3) that fixes ONNX's frozen-abstraction problem; **capability
negotiation with derivable realizations** (§10); and the **archival profile**
(§14.8) that embeds its own decoder specification.

## 0.4 Anatomy of a model

A *model* in OMNI is a `Manifest` object plus everything reachable from it.

```
Manifest  (root; the thing you sign)
├── metadata        → Metadata            model card, license, provenance, hints
├── assets{}        → named entries, each one of:
│   ├── Model       → tensors, graph, tokenizer, defaults
│   ├── Tokenizer   → tokenizer IR
│   ├── Adapter     → LoRA/DoRA/… tensor deltas + attach points
│   ├── Dataset     → sample or reference-only dataset descriptor
│   └── Blob        → opaque payload (image, README, foreign artifact)
├── parents[]       → Manifest refs (delta / inheritance; §08.6)
├── realizations[]  → precomputed Plans for common capability sets (§10.5)
├── caches[]        → RuntimeCache refs (always droppable; §10.6)
├── attestations[]  → Signature / Provenance objects (§12)
└── features        → required[] / optional[] feature URIs (§14.3)
```

A `Model` asset is:

```
Model
├── tensors     → TensorTable  (name → TensorDesc; §04)
├── graph?      → GraphModule  (OMNI-IR; optional; §07)
├── arch?       → architecture descriptor + dialect refs (§07.2)
├── tokenizer?  → ref to Tokenizer asset
├── generation? → sampling / decoding defaults (§06.5)
└── training?   → TrainingState (optional; §09)
```

Every arrow above is a **content-addressed reference**. Two models that share a
base checkpoint share the referenced objects *physically*, in any store, without
coordination.

## 0.5 What is canonical and what is derived

This distinction is normative and is enforced by validators.

| Canonical (identity-bearing) | Derived (regenerable, droppable) |
|---|---|
| Tensor expression trees | Materialized tensor bytes for a given dtype |
| Chunk contents (uncompressed) | Chosen compression codec of a chunk |
| Graph at its highest declared level | Lowered graphs, fused kernels |
| Manifest, Metadata, Tokenizer | Object index, name index, Bao trees |
| Adapter tensors | Merged base+adapter weights |
| Feature declarations | Realization plans |
| Attestations | Runtime caches (TRT engines, `.mlmodelc`, autotune tables) |

Formally: an object is **derived** iff its content is a pure function of other
objects reachable from the same manifest, and it carries `flags.cacheable = 1`.
Removing every derived object from a container MUST leave a valid container that
denotes the same model. A validator MUST be able to verify this by recomputation
for every derived object type it understands. (`omni gc --canonical` performs
exactly this reduction; `omni verify --recompute` checks it.)

## 0.6 Conformance profiles

Implementations are large; not everyone needs all of OMNI. Profiles let a
minimal loader be *correct* rather than *partial*.

| Profile | Name | Requirement |
|---|---|---|
| **C0** | Reader-Core | Parse header/index/manifest; resolve refs; verify digests; read `literal` tensors in registered dtypes; `raw` + `zstd` codecs. |
| **C1** | Reader-Value | C0 + full tensor expression evaluation (§04.7), all registered quantization schemes, sparsity. |
| **C2** | Reader-Semantic | C1 + OMNI-IR parsing/validation, dialect resolution, tokenizer IR, chat templates. |
| **C3** | Writer | Produce canonical, reproducible containers; deterministic packing; signature emission. |
| **C4** | Negotiator | Capability negotiation and plan derivation (§10). |
| **CX** | Extended | WASM plugin execution (§11.6), encryption profile (§12.8), FUSE/mount, registry transport. |

A *Runtime* claiming "OMNI support" MUST state its profiles, e.g.
`OMNI/1.0 C0 C1 C2 C4`. `omni verify --profile C1 model.omni` checks that a file
is loadable under a stated profile.

Orthogonal **container profiles** (§02.2) constrain the file itself:
`core`, `stream`, `append`, `archive` (OMNI-A), `cache`.

## 0.7 Terminology

- **Object** — an immutable byte string with a type and a digest (§01.2).
- **Digest** — a multihash-style identifier; BLAKE3-256 by default (§03.5).
- **Ref** — a typed pointer to an object by digest (§01.4).
- **Chunk** — an object of type `Blob` holding raw payload bytes (§04.5).
- **ChunkList** — an ordered list of chunk refs forming a logical byte stream.
- **Container** — a `.omni` file: a pack of objects plus an index (§02).
- **Store** — anything that can resolve digest → bytes: a container, a
  directory, a registry, an HTTP endpoint (§01.8).
- **Realization / Plan** — a concrete choice of representations satisfying a
  capability set (§10.4).
- **Dialect** — a namespaced set of IR operations with versioned schemas (§07.4).
- **Feature** — a URI naming a capability a reader may need (§14.3).
- **Criticality** — whether an unknown object may be ignored (§11.3).

## 0.8 Non-goals

Stating these prevents scope collapse later:

1. **OMNI is not a runtime.** It defines no kernels, no scheduling, no memory
   planner. Reference WASM semantics (§11.6) exist to *define* operators, not to
   run them fast.
2. **OMNI is not a training framework.** It stores training state (§09); it does
   not define an optimizer.
3. **OMNI does not standardize architectures.** `nn.attention` lives in a
   dialect that could be replaced wholesale; the core never depends on it.
4. **OMNI is not a compression research vehicle.** Codecs are pluggable and
   deliberately boring in the mandatory set (§03.7).
5. **OMNI does not attempt semantic model equivalence checking.** Two models are
   "the same" iff their canonical DAGs hash equal. Numerical equivalence under
   different lowerings is a testing problem, not a format problem.
6. **OMNI does not police licensing or safety.** It gives them a structured,
   signed place to live (§06.6, §12.6) and makes tampering detectable.

## 0.9 The 50-year test

Every design decision in this specification was checked against a single
question: *can a competent engineer in 2076, with no access to us, no working
internet archive, and no legacy toolchain, read this file?*

The concrete answers:

1. **The header is fixed, tiny, and self-describing** — 128 bytes, documented in
   §02.3, with magic bytes that survive text-mode corruption.
2. **Structure is self-describing CBOR** (RFC 8949, an IETF Standards-Track
   binary format with two dozen independent implementations), not a
   schema-dependent format like Protobuf or FlatBuffers where losing the schema
   loses the data.
3. **The mandatory codec set is `raw` and `zstd`** — and the archival profile
   (§14.8) forbids everything except `raw` and `deflate` (RFC 1951), the most
   widely reimplemented compression algorithm in history.
4. **The archival profile embeds a Rosetta object** (§14.8.3): the full text of
   this specification, the schema definitions, and a plain-language description
   of the header layout, stored *inside the file*, uncompressed, in UTF-8.
5. **No external resolution is required.** Registry URIs are identifiers, not
   URLs that must be dereferenced; every object needed to interpret the file is
   either inside it or explicitly declared missing.
6. **Nothing executes on load.** A 2076 reader is not required to implement a
   2026 sandbox to read 2026 weights (§12.2).

**Next:** [§01 Object Model](01-object-model.md)
