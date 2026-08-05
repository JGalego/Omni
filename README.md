<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/omni-logo-anim-dark.svg">
  <img src="assets/omni-logo-anim.svg" alt="Omni" width="230">
</picture>

<p><em><strong>A model exists once. Everything else is derived.</strong></em></p>

[![CI](https://github.com/JGalego/Omni/actions/workflows/ci.yml/badge.svg)](https://github.com/JGalego/Omni/actions/workflows/ci.yml)
[![Spec](https://img.shields.io/badge/spec-OMNI%2F1.0--draft-6366f1)](docs/spec/00-overview.md)
[![Reference](https://img.shields.io/badge/reference-Rust-22d3ee?logo=rust&logoColor=white)](reference)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](reference/omni-core/Cargo.toml)
[![Unsafe](https://img.shields.io/badge/unsafe-forbidden-success)](reference/omni-core/src/lib.rs)
[![Spec license](https://img.shields.io/badge/spec-CC--BY--4.0-blue)](#license)
[![Code license](https://img.shields.io/badge/code-Apache--2.0%20OR%20MIT-blue)](#license)

</div>

**OMNI** — Open Model Neutral Interchange — is a vendor-neutral, content-addressed,
cryptographically verifiable container and object model for machine learning
models. It is designed to be the canonical, archival representation of a model:
the thing from which every other representation (safetensors, GGUF, ONNX,
TensorRT, CoreML, MLX, …) is *derived*.

This repository holds the engineering proposal, the normative specification, and
a reference implementation in Rust.

## The idea

Today a model exists as a dozen incompatible artifacts: a PyTorch checkpoint, a
safetensors shard set, four GGUF quantizations, an ONNX export, a TensorRT
engine, an MLX conversion, a CoreML package. Each is a full copy. Each carries
partial, contradictory metadata. None can prove where it came from.

OMNI says a model is a **directed acyclic graph of immutable, hash-addressed
objects**. Weights are chunk-addressed byte ranges. Quantizations, adapters and
deltas are **lazy algebraic transformations** over those chunks, not copies.
Architecture is described by a **multi-level, dialect-based IR** with no
hardcoded knowledge of transformers or anything else. Everything
hardware-specific is a **droppable cache**. The `.omni` file is just a *pack* —
one possible physical serialization of the object graph, alongside directory
stores, OCI registries and HTTP ranges.

## The design

**🧬 A model exists once.** Everything else is a view, a derivation, or a cache.

**#️⃣ Identity is a hash.** Objects are immutable and content-addressed, so
deduplication, deltas, resumable transfer and integrity all fall out of a single
mechanism instead of several unrelated ones.

**🧮 Weights are expressions, not files.** A tensor's value is a pure expression
tree — `dequantize`, `add-lora`, `concat`, `slice` — evaluated lazily by the
runtime. Fine-tunes and quantizations cost their delta, not a copy.

**🔌 The format knows nothing about architectures.** Transformers, Mamba, diffusion
and whatever comes in 2050 are all *dialect plugins* with versioned op schemas
and WebAssembly reference semantics.

**⚡ Hardware never touches the canonical model.** TensorRT engines, autotuned
kernels and materialized fp8 copies are cache objects keyed by the digest of
what produced them, and may always be deleted.

**🔭 Unknown things are not errors.** PNG-style criticality bits mean a reader from
2026 can validate, copy, sign and partially execute a file written in 2071.

## Getting started

The reference implementation has zero dependencies, so installing the CLI is a
clone and one `cargo install`:

```console
$ git clone https://github.com/JGalego/Omni && cd Omni
$ cargo install --path reference/omni-cli
```

That puts an `omni` binary on your `PATH`. Ask it for a model — a small but
complete two-layer transformer, byte-reproducible down to the last digest:

```console
$ omni example toy.omni
```

Look inside. `inspect` decides everything from metadata alone; no tensor
payload is ever read:

```console
$ omni inspect toy.omni
```

Now make it prove itself — validation levels V0 through V6, from framing and
hashes up to tensor semantics and recomputed derived objects:

```console
$ omni verify toy.omni --level 6
```

The same binary can quantize (`example --quantized`), evaluate tensor
expressions (`cat`), show what a read costs in real I/O (`open`), plan against a
runtime's capabilities (`plan`), sign and verify (`sign`), diff two models
(`delta`), attach adapters (`adapter`), tokenize (`tokenize`), render chat
templates (`render`), print, verify and lower an execution graph (`graph`), run
an embedded WebAssembly plugin (`plugin`), and separate a training checkpoint
from the weights it shares (`strip`, `log`, `reshard`), read a container over
HTTP one range at a time, serve one, or map one onto an OCI registry (`index`,
`fetch`, `serve`, `oci`), and absorb or emit safetensors with a
fidelity report rather than a shrug (`import`, `export`) — the full tour is in
[`reference/README.md`](reference/README.md).

The committed [`examples/toy.omni`](examples/toy.omni) is the exact container
`omni example` writes — CI proves it byte-for-byte on every push.
[`examples/README.md`](examples/README.md) dissects it with annotated hexdumps,
CBOR diagnostic listings of every object, and an independent digest
verification. To run the test suite: `cd reference && cargo test`.

## Repository map

| Path | Contents |
|---|---|
| [`docs/spec/`](docs/spec) | Normative specification (OMNI/1.0 draft) |
| [`docs/design/`](docs/design) | Engineering proposal: import/export, CLI, SDK, performance, comparisons, roadmap |
| [`docs/rationale/`](docs/rationale) | Design rationale and rejected alternatives |
| [`conformance/`](conformance) | Conformance corpus: valid, invalid and forward-compatibility cases |
| [`examples/`](examples) | Worked example files, CBOR diagnostic listings, hexdumps |
| [`reference/`](reference) | Rust reference implementation (`omni-core`, `omni-cli`) |

### Specification

| # | Document | Covers |
|---|---|---|
| 00 | [Architecture Overview](docs/spec/00-overview.md) | Layer model, conformance profiles, terminology |
| 01 | [Object Model](docs/spec/01-object-model.md) | Objects, digests, references, DAG, stores, packs |
| 02 | [Container Binary Format](docs/spec/02-container.md) | Header, segments, object index, trailer, alignment |
| 03 | [Encoding & Hashing](docs/spec/03-encoding.md) | OMNI-CBOR, canonicalization, codecs, digest algebra |
| 04 | [Tensors](docs/spec/04-tensors.md) | Numeric type algebra, layouts, chunking, sparsity, tensor expressions |
| 05 | [Quantization](docs/spec/05-quantization.md) | Quantization as transformation; GPTQ/AWQ/GGUF/EXL2/HQQ/NF4/MX |
| 06 | [Metadata & Tokenizers](docs/spec/06-metadata.md) | Model card, provenance, tokenizer IR, chat templates |
| 07 | [Execution Graph (OMNI-IR)](docs/spec/07-graph.md) | Multi-level IR, dialects, operator versioning, rewriting |
| 08 | [Adapters & Delta Models](docs/spec/08-adapters.md) | LoRA/DoRA/IA³/PEFT, composition, model inheritance |
| 09 | [Training State](docs/spec/09-training.md) | Optimizer state, sharded/distributed checkpoints, RNG |
| 10 | [Runtime & Capability Negotiation](docs/spec/10-runtime.md) | Caches, capability sets, plan resolution |
| 11 | [Plugin System](docs/spec/11-plugins.md) | Namespaces, criticality bits, WASM semantics, registry |
| 12 | [Security Model](docs/spec/12-security.md) | Threat model, verification levels, signatures, provenance |
| 13 | [Streaming & Transport](docs/spec/13-streaming.md) | HTTP ranges, Bao verification, OCI mapping, packs |
| 14 | [Versioning & Migration](docs/spec/14-versioning.md) | Feature flags, migration, deprecation, archival profile |
| 15 | [Conformance & Validation](docs/spec/15-conformance.md) | Validation rules, levels, test suite |

### Engineering proposal

- [Import / Export Architecture](docs/design/import-export.md) — fidelity contracts, per-format capability matrix, loss reports
- [CLI Specification](docs/design/cli.md) — `omni inspect`, `pack`, `verify`, `delta`, `mount`, …
- [SDK Design](docs/design/sdk.md) — Rust core, C ABI, Python/C++/Go/Java/Swift/JS bindings
- [Performance Analysis](docs/design/performance.md) — analytic models for NVMe, mmap, GDS, network, object stores
- [Format Comparison](docs/design/comparison.md) — OMNI against every major model format
- [Reference Implementation Roadmap](docs/design/roadmap.md) — phases, governance, conformance corpus
- [Design Rationale](docs/rationale/tradeoffs.md) — every major decision and what was rejected

## Status

**OMNI/1.0 — Draft.** This is a proposal, not a ratified standard. The binary
framing ([§02](docs/spec/02-container.md)) and object model
([§01](docs/spec/01-object-model.md)) are the parts that must be right on the
first attempt; everything above them is designed to evolve through the registry
and feature-flag mechanisms in [§14](docs/spec/14-versioning.md).

The reference implementation covers the container and object model, canonical
encoding and both mandatory hashes, the tensor expression algebra with its
typing and range pushdown, quantization and sparsity, compression (including
`zstd`, the one codec §03 marks MUST), tokenizers and chat templates, adapters
and deltas, OMNI-IR with its dialects and rewrites, capability negotiation,
signatures, training state, a WebAssembly host for plugins, HTTP range transport
with the detached index sidecar, an object server and the OCI mapping, and
lossless safetensors import and export. What is *not* implemented is stated in
the same place it is claimed: [`reference/README.md`](reference/README.md) lists
it — every importer and exporter except safetensors, `https://` (TLS needs a
dependency), the registry client behind §13.5's mapping, `mount`, SIMD in the
plugin host, and the MAY-level codecs — and every one of them is reported as
unsupported at run time rather than guessed at. See
[the roadmap](docs/design/roadmap.md) for what the gates have and have not
proven.

## File extension and media types

- `.omni` — sealed container (a *pack* of the object graph)
- `.omni.idx` — detached object index sidecar (CDN-friendly)
- `.omnid/` — directory-backed object store
- `application/vnd.omni.container.v1` — container
- `application/vnd.omni.object.v1+cbor` — single structure object
- `application/vnd.omni.chunk.v1` — single data chunk

[§14.7](docs/spec/14-versioning.md) has the reasoning behind keeping `.omni`.

## License

The specification is offered under CC BY 4.0; the reference implementation under
Apache-2.0 OR MIT. Standards need patent-safe, permissive licensing to be
adopted, and dual licensing removes the last excuse.
