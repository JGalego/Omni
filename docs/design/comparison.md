# Comparison Against Existing Formats

Each format below is compared on what it was *designed* to do, then on what it
cannot do. Most of them are good at their job; the argument for OMNI is not that
they are bad, but that their jobs are disjoint and nobody owns the union.

## 1 The capability matrix

**Legend:** ● yes · ◐ partial / by convention · ○ no

| | OMNI | safetensors | GGUF | ONNX | `.pt` | TF SavedModel | TensorRT | CoreML | MLX | TFLite | ExecuTorch | HDF5 | Zarr | NPZ | OCI/Ollama |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Safe to load (no code exec)** | ● | ● | ● | ● | ○ | ◐ | ○ | ◐ | ● | ● | ● | ● | ● | ● | ● |
| **Memory-mappable / zero-copy** | ● | ● | ● | ◐ | ○ | ○ | ◐ | ◐ | ● | ● | ● | ◐ | ◐ | ○ | ○ |
| **Guaranteed alignment** | ● | ◐¹ | ● | ○ | ○ | ○ | ● | ◐ | ● | ● | ● | ◐ | ◐ | ○ | ○ |
| **Random access to one tensor** | ● | ● | ◐² | ◐ | ○ | ◐ | ○ | ◐ | ● | ◐ | ◐ | ● | ● | ◐ | ○ |
| **Sub-tensor / range access** | ● | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ● | ● | ○ | ○ |
| **Content-addressed** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ● |
| **Deduplication across models** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐³ |
| **Delta / inheritance** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐³ |
| **Streaming / partial download** | ● | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ● | ○ | ◐ |
| **Verified partial reads** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| **Built-in digital signatures** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ◐⁴ | ○ | ○ | ○ | ○ | ○ | ○ | ◐⁵ |
| **Provenance / lineage chain** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ |
| **Quantization as transformation** | ● | ○ | ○ | ◐⁶ | ○ | ○ | ○ | ○ | ○ | ◐⁶ | ◐⁶ | ○ | ○ | ○ | ○ |
| **Arbitrary / future precisions** | ● | ◐⁷ | ○ | ○ | ◐ | ◐ | ○ | ○ | ◐ | ○ | ○ | ● | ● | ◐ | ○ |
| **Sub-byte packing declared** | ● | ○ | ● | ○ | ○ | ○ | ◐ | ◐ | ◐ | ● | ● | ○ | ○ | ○ | ○ |
| **Sparse tensors** | ● | ○ | ○ | ◐ | ◐ | ◐ | ● | ◐ | ○ | ◐ | ◐ | ◐ | ● | ○ | ○ |
| **Adapters as first-class** | ● | ◐⁸ | ◐ | ○ | ◐⁸ | ○ | ○ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ |
| **Execution graph** | ● | ○ | ○ | ● | ◐⁹ | ● | ● | ● | ○ | ● | ● | ○ | ○ | ○ | ○ |
| **Multi-level graph** | ● | — | — | ○ | ○ | ○ | ○ | ○ | — | ○ | ○ | — | — | — | — |
| **Extensible ops without spec change** | ● | — | ○ | ◐¹⁰ | ● | ● | ◐ | ◐ | — | ◐ | ◐ | — | — | — | — |
| **Unknown-extension tolerance** | ● | ○ | ◐ | ◐ | ○ | ◐ | ○ | ○ | ○ | ◐ | ◐ | ● | ● | ○ | ● |
| **Tokenizer included & structured** | ● | ○ | ● | ○ | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ● |
| **Chat template (safe)** | ● | ○ | ◐¹¹ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐¹¹ |
| **Rich model metadata** | ● | ◐ | ● | ◐ | ○ | ◐ | ○ | ● | ○ | ◐ | ◐ | ◐ | ● | ○ | ◐ |
| **Training state / optimizer** | ● | ◐ | ○ | ○ | ● | ◐ | ○ | ○ | ◐ | ○ | ○ | ● | ● | ◐ | ○ |
| **Distributed/sharded checkpoints** | ● | ◐ | ○ | ○ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ | ● | ○ | ○ |
| **Hardware-specific caches, droppable** | ● | ○ | ○ | ○ | ○ | ○ | —¹² | —¹² | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| **Capability negotiation** | ● | ○ | ○ | ◐¹³ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| **Reproducible byte-identical write** | ● | ● | ◐ | ◐ | ○ | ○ | ○ | ○ | ◐ | ◐ | ◐ | ○ | ○ | ● | ● |
| **Multi-model bundles** | ● | ○ | ○ | ◐ | ◐ | ● | ○ | ● | ○ | ○ | ○ | ● | ● | ◐ | ● |
| **Self-describing without a schema** | ● | ● | ● | ○ | ○ | ○ | ○ | ○ | ● | ○ | ○ | ● | ● | ● | ● |
| **Archival profile / embedded spec** | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |

¹ safetensors aligns to 8 bytes after a variable-length JSON header, so tensor
starts are not page-aligned; ² GGUF requires scanning the KV/tensor-info section;
³ OCI dedups whole layers, not tensor content; ⁴ CoreML supports code signing of
the package via macOS mechanisms; ⁵ cosign/notation on the registry, not the
format; ⁶ QDQ nodes represent quantization in the graph, but the *stored* weights
are one representation; ⁷ safetensors' dtype list is an enum requiring library
updates; ⁸ PEFT adapters are safetensors files plus a JSON config convention;
⁹ TorchScript archives; ¹⁰ ONNX custom domains exist but are not portable;
¹¹ Jinja2 template string — arbitrary code execution; ¹² these *are* the caches;
¹³ ONNX opset version is a coarse form of it.

## 2 Format-by-format

### safetensors
**Does well:** the correct core insight — a header plus a flat, contiguous,
mappable payload, with no code execution. Simple enough that everyone
implemented it. It deserves credit for ending the pickle era.

**Cannot do:** one monolithic JSON header (parse-all-or-nothing, and it grows
with tensor count); page alignment is not guaranteed because the header is
variable-length; dtypes are a closed enum; no graph, no tokenizer, no
architecture, no signatures, no dedup, no deltas, no quantization semantics, no
sparsity, no extension mechanism, `__metadata__` is `map<string,string>`.

**OMNI's relationship:** OMNI is what safetensors would be if you kept its
discipline and added an object model. `omni mount` exposes a safetensors view, so
migration costs nothing. Round-trip is lossless.

### GGUF
**Does well:** by far the most *complete* single-file format in practice —
weights, quantization, tokenizer, chat template, architecture parameters and
generation defaults in one mappable file. Excellent quantization schemes.
Pragmatic, and it won on the desktop for good reasons.

**Cannot do:** architecture is an enum, so every new architecture needs a
`llama.cpp` release; the KV schema is hand-maintained per architecture and is
the format's bottleneck; no graph; no dedup or deltas (each quantization is a
full copy — the single largest source of duplicated bytes on model hubs today);
no signatures or provenance; Jinja templates are executed; no partial or
verified streaming; extension is by convention.

**OMNI's relationship:** GGUF is the model to beat on completeness. OMNI imports
it losslessly in both structural and opaque form, so `llama.cpp` can consume an
OMNI container's cached GGUF payload with zero conversion while the canonical
model stays transparent.

### ONNX
**Does well:** the only widely adopted *graph* interchange format. Broad runtime
support. Genuine portability for classical models.

**Cannot do:** monolithic opset versioning makes extension a committee process;
custom domains are not portable; the abstraction level is fixed at primitives, so
backends pattern-match to recover intent (every serious inference engine has a
"fuse this 15-op subgraph back into attention" pass); protobuf's 2 GB message
limit forced the external-data mechanism, which is a second, weaker format;
no content addressing, no signatures, no adapters, no tokenizer, no
quantization-as-transformation, no streaming.

**OMNI's relationship:** OMNI-IR's multi-level design is a direct response to
ONNX's frozen-abstraction problem. ONNX imports as a `primitive`-level graph in
an `onnx`-compat dialect; export requires a graph and reports op-coverage gaps
precisely.

### PyTorch `.pt` / `.pth` / `.bin`
**Does well:** trivially convenient; stores anything Python can pickle,
including optimizer state and arbitrary objects.

**Cannot do:** *it executes arbitrary code on load.* Everything else is
secondary. Also: no random access, no alignment, no mmap, no dedup, no schema,
and it is not a format so much as a serialization of one language's object graph.

**OMNI's relationship:** import via a sandboxed restricted unpickler (§12.10),
never export by default. The ecosystem has already been migrating away; OMNI
should complete that migration rather than accommodate it.

### PyTorch DCP / DeepSpeed / Megatron / NeMo checkpoints
**Do well:** parallel writes at scale, which is genuinely hard.

**Cannot do:** each is a private layout tied to a parallelism strategy;
resharding requires framework-specific scripts; replicated tensors are stored
redundantly; no portable metadata; no integrity beyond the filesystem.

**OMNI's relationship:** §09.4 stores *logical* tensors with sharding as
metadata, so resharding is an expression rewrite and DP-replicated tensors
deduplicate automatically. This is one of OMNI's clearest wins and one of the
least glamorous.

### TensorRT engines / OpenVINO IR / CoreML / TFLite / ExecuTorch / MLX
**Do well:** each is optimal for its target. That is the point of them.

**Cannot do:** they are *outputs*, not sources. A TensorRT engine is tied to a
GPU architecture, TensorRT version and often a driver range; it is code, not
data; and it cannot be converted back.

**OMNI's relationship:** these are `RuntimeCache` objects (§10.6) — attached,
signed, verified against the plan digest that produced them, refused by default
because they are executable, and always droppable. OMNI's claim is not that it
replaces them but that it is the right *source* for producing them and the right
container for shipping them alongside a portable model.

### HDF5
**Does well:** chunking, compression filters, partial I/O, hierarchical
structure, decades of scientific use. Technically the closest prior art for the
storage layer, and it got a lot right in 1998.

**Cannot do:** one dominant, complex C implementation; historically fragile
under concurrency and prone to file corruption on abnormal termination; no
content addressing; no cryptographic story; a type system oriented to scientific
arrays rather than ML numerics (no fp8, no sub-byte packing, no quantization);
no graph, no model semantics.

### Zarr
**Does well:** chunked, cloud-native, extensible codecs, works over object
stores, excellent for large N-dimensional arrays. Zarr v3's extension model is
genuinely good and directly comparable to OMNI's codec/dtype registries.

**Cannot do:** no model semantics (it is an array store, by design); no content
addressing or dedup; no signatures; one chunk per key means a model with 10⁵
chunks is 10⁵ objects with no packing; no graph, tokenizer, or adapters.

**OMNI's relationship:** the closest peer for the tensor-storage layer, and a
source of good ideas. OMNI adds identity, semantics and packing.

### OCI artifacts / Ollama bundles
**Do well:** distribution. Content-addressed layers, registries, mirrors, CDNs,
auth, replication — all solved and deployed globally.

**Cannot do:** dedup at layer granularity only (a 1-byte change makes a new
layer); tar layers have no random access or alignment; no model semantics; the
manifest is JSON with no object model.

**OMNI's relationship:** not a competitor — a *transport*. §13.5 maps OMNI onto
OCI so it inherits the entire registry ecosystem while adding sub-layer dedup and
tensor-level addressing.

### NPZ / NPY, Flax msgpack, Keras
Simple array containers; fine for what they are. No alignment guarantees (NPZ is
zip-based), no semantics, no integrity. Import/export supported; nothing to learn
beyond simplicity.

### Historical: PMML, PFA, NNEF, CNTK, Caffe prototxt
Worth studying for how formats die: PMML was XML-verbose and model-type-enumerated
(a new model type required a spec revision — precisely the failure OMNI's plugin
system is designed to avoid); NNEF was well-engineered but arrived after ONNX had
network effects; Caffe's prototxt hardcoded layer types.

**The lesson taken:** enumerating model types in the specification is fatal, and
timing plus tooling beats elegance. OMNI's response is (a) enumerate nothing, and
(b) make adoption incremental — import losslessly, export everywhere, mount as
safetensors, and ride OCI for distribution, so nobody has to switch to benefit.

## 3 Non-ML formats OMNI learns from

| Format | Lesson taken | Lesson *not* taken |
|---|---|---|
| **ELF** | fixed header + section table + alignment; program vs. section views | its 1990s 32/64-bit split and per-architecture variants |
| **PNG** | criticality/safe-to-copy bits; a magic that detects text corruption; 30 years of forward compatibility with zero format breaks | fixed 4-byte chunk names, CRC per chunk (too small a unit at our scale) |
| **ZIP** | end-of-file directory enables append and one-seek open | per-entry framing, no alignment, legacy 4 GiB corners |
| **Parquet** | footer metadata, column chunks, statistics enabling skip | thrift-encoded metadata (schema dependence) |
| **Matroska/EBML** | self-describing extensible element tree, unknown elements skippable | verbose per-element IDs at tensor scale |
| **Git** | content-addressed immutable objects, packs, delta encoding, DAG history | SHA-1 (and how painful its migration was), no typed objects, no alignment |
| **OCI** | registries as CAS, media types, referrers | tar layers |
| **LLVM bitcode** | stable serialized IR, forward-compat wrappers, versioned dialects | its bit-level packing (unreadable without the toolchain) |
| **MLIR** | dialects and progressive lowering | in-memory-only design |
| **PDF/A** | archival profiles that forbid external dependencies | everything else about PDF |
| **FITS** | 30+ year archival readability through simplicity and embedded documentation | fixed 80-column card headers |
| **TUF** | repository-level key rotation, freshness, rollback protection | — |

## 4 Honest assessment: where OMNI is at a disadvantage

1. **Complexity.** OMNI is a much larger specification than safetensors. The
   mitigation is the C0 budget (§SDK.5): a *useful* reader is ~3 000 lines. But
   the full surface is large, and large specifications acquire dark corners. The
   conformance suite exists to make the corners testable rather than mythical.
2. **Network effects.** safetensors and GGUF have them; OMNI has none. This is
   the actual risk, and it is why the adoption strategy is parasitic rather than
   confrontational: lossless import, faithful export, a FUSE view that makes OMNI
   files openable by existing tools, and OCI transport so hubs need no new
   infrastructure.
3. **The evaluator requirement.** Tensor expressions mean a C0-only reader
   cannot load a model whose tensors are non-trivial expressions. Publishers who
   care about maximum reach must include a `literal` realization, which
   reintroduces duplication for those tensors. This is a genuine tension, not a
   solved problem; the guidance is that hubs should store expressions and serve
   materialized realizations on demand.
4. **Materialization latency** on first load of a derived representation
   (§performance.9).
5. **It is unproven.** Every claim here is a design claim. The roadmap's gates
   are structured so that the format is validated against real corpora before it
   asks anyone to adopt it.

**See also:** [Performance](performance.md) · [Import/Export](import-export.md) · [Rationale](../rationale/tradeoffs.md)
