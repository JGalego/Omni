# Reference Implementation Roadmap

The specification is worthless without an implementation that proves it and a
governance structure that outlives its authors. This is the plan, with gates
that can fail.

---

## Phase 0 — Prove the container (months 0–3)

**Goal:** the binary format is implementable, fast, and correct.

Deliverables:
- `omni-cbor`: canonical encoder/decoder, strict mode, OSD schema validation.
- `omni-hash`: BLAKE3 + SHA-256, Bao outboard encoding and range verification.
- `omni-core`: header/segment/index/trailer parse and write; object model; refs;
  `no_std + alloc`; `#![forbid(unsafe_code)]`.
- `omni-io`: file + mmap + memory + directory stores.
- `omni-cli`: `inspect`, `pack`, `unpack`, `verify`, `dump`, `fsck`.
- Conformance corpus v0: `valid/minimal`, `invalid/framing`, `invalid/encoding`.
- Continuous fuzzing (cargo-fuzz + AFL++) on header, index, and CBOR.

**Gate 0:** a 140 GB container opens in 2 reads; index lookup p99 < 500 ns at
10⁶ objects; `pack` is byte-reproducible; 72 h of fuzzing finds no crash, hang,
or OOM. *If the index cannot hit that latency, the index format changes now,
not later.*

## Phase 1 — Prove the value layer (months 3–7)

**Goal:** tensor expressions are implementable and actually save bytes.

Deliverables:
- Full dtype algebra with bit-exact encode/decode and all rounding modes.
- Layout math: strided, tiled, packed, blocked-scaled, interleaved.
- `omni-eval`: expression evaluation, range pushdown, fusion, caching.
- Quantization schemes: affine, sym, codebook, nested; GPTQ/AWQ/NF4/MX/GGUF-K
  structural mappings.
- Sparsity schemes.
- `omni-import-safetensors`, `-gguf`, `-gptq`, `-awq`, `-peft`.
- `omni-export-safetensors`, `-gguf`.
- `omni delta`, `omni adapter`, `omni convert`.
- Conformance corpus: `numeric/`, `roundtrip/`, `valid/features`.

**Gate 1:** lossless round-trip for safetensors, GGUF, GPTQ, AWQ, PEFT, verified
bit-exactly on ≥ 100 real models. A published delta-size study over ≥ 50 real
base/fine-tune pairs. Differential test against `llama.cpp`'s dequantization for
every GGUF K-quant type: zero mismatches. *If the structural GGUF mapping cannot
be made bit-exact, §05.2.4 is wrong and gets revised.*

## Phase 2 — Prove the semantic layer (months 7–12)

**Goal:** the model is self-describing.

Deliverables:
- OMNI-IR: parse, verify, print; `omni.core` + `omni.tensor` + `omni.nn` v1.
- Dialect mechanism, op versioning, declarative rewrites, lowering.
- `omni-plugin`: WASM host with the restricted profile, fuel metering.
- Tokenizer IR + conformance vectors + Jinja2 → OMNI-CT translator.
- `omni graph synthesize` for registered families.
- `omni-import-onnx`, `-export-onnx`.
- Capability negotiation and `omni plan`.

**Gate 2:** ten architecture families (§07.8) expressed end-to-end and executed
by a reference interpreter with outputs matching the source framework within
declared tolerance. Tokenizer vectors pass for ≥ 200 real tokenizers.
Jinja2 → OMNI-CT translation succeeds for ≥ 95 % of chat templates on a public
hub snapshot, with the failures analyzed and published. *If OMNI-CT cannot cover
95 %, the language grows or the fallback story changes.*

## Phase 3 — Prove the distribution layer (months 12–18)

**Goal:** it works at internet scale.

Deliverables:
- HTTP and object-store stores with range coalescing and resumption.
- OCI push/pull with `by-novelty` pack partitioning; referrers for adapters and
  signatures.
- Verified streaming (Bao) end to end; progressive load.
- `omni mount` (FUSE) with synthesized safetensors and tokenizer views.
- `omni serve`: an object server.
- Signatures (Ed25519, ES256, ML-DSA), Sigstore keyless, in-toto/SLSA
  provenance, revocation.
- Published benchmark suite (§performance.10).

**Gate 3:** a mirror of ≥ 10 000 real models, with measured dedup, delta sizes
and load times published as signed OMNI containers. TTFT over a throttled link
demonstrably better than the incumbent. A third-party security review of the
parser and the signature stack. *This gate is where the storage-savings claims
in this proposal are confirmed or retracted.*

## Phase 4 — Ecosystem (months 18–30)

- Bindings: Python, C, C++, Go (pure-Go C0 reader), Java, Swift, JS/WASM.
- Upstream integrations: PyTorch, JAX/Orbax, vLLM, SGLang, llama.cpp, MLX,
  Ollama, Transformers, PEFT, TensorRT-LLM.
- Training-side: DCP/DeepSpeed/Megatron/NeMo import and export; parallel writer.
- Hub-side reference: an OMNI-native model registry with dedup accounting.
- Conformance suite v1.0, published test reports, interoperability marks.

**Gate 4:** two independent, from-specification implementations pass the
conformance suite (the working group's Rust one does not count as one of them).
At least one production deployment outside the authoring organizations.

## Phase 5 — Standardization (months 30+)

- Working group charter, IPR policy (royalty-free, non-assertion), public
  archives.
- Registry operating procedures; mirrored, signed registry containers.
- Media-type registration with IANA; `magic(5)` entries upstreamed to `file`.
- OMNI/1.0 declared stable; container framing frozen.
- Archival deposit (Software Heritage + an academic library) in OMNI-A form.

---

## Engineering practices (non-negotiable)

| Practice | Why |
|---|---|
| `#![forbid(unsafe_code)]` in all parsing crates | untrusted input |
| Continuous fuzzing with a persistent corpus | untrusted input |
| Property-based tests for every dtype/layout round-trip | bit-exactness is the whole game |
| Differential testing against reference implementations | ambiguity detection |
| Every normative rule has a test case with its rule ID | spec↔code traceability |
| Reproducible builds of the toolchain itself | §01.10 credibility |
| Public benchmark methodology and raw data | §performance credibility |
| Semantic versioning with a documented MSRV policy | dependability |
| Every published claim in these docs marked as *modeled* or *measured* | honesty |

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Nobody adopts it** | high | Parasitic adoption: lossless import, faithful export, FUSE view, OCI transport. OMNI must be useful to someone who never publishes an `.omni`. |
| **The expression algebra is too complex for runtimes** | medium | C0/C1 profile split; `literal` realizations; the evaluator is ~1 500 lines and is provided in every binding |
| **Spec ambiguity causes divergent implementations** | medium | Differential testing from Phase 1; conformance suite is normative; WASM reference semantics for anything subtle |
| **A dominant vendor forks it** | medium | Permissive licensing + royalty-free IPR make forking legal but pointless; the conformance mark and mirrored registry are the coordination points |
| **Registry capture / politicization** | medium | `x.*` namespace, first-come-first-served for third parties, frozen core, mirrorable signed registry (§11.7) |
| **Hash algorithm break** | low, high impact | Agility designed in (§12.11); migration costs one hashing pass, zero re-uploads |
| **Container format needs a breaking change** | low, very high impact | `header_size` growth, skippable segments, index `fmt_version`, feature flags — the specific mechanisms that prevent it |
| **Scope creep** | high | The C0 budget (§SDK.5) is a hard gate on every proposal |

## Success criteria at 5 years

1. A model published once, consumed by CPU, GPU, NPU and browser runtimes from
   the same bytes.
2. Fine-tunes and quantizations distributed as deltas measured in percent of
   base size.
3. `omni verify --level V8` is a routine step in enterprise model deployment.
4. At least one major hub storing OMNI natively and reporting dedup savings.
5. A new architecture shipping as a dialect plugin, running on
   unmodified-but-older runtimes via its shipped lowering.
6. The container framing unchanged since 1.0.

---

**See also:** [Comparison](comparison.md) · [§15 Conformance](../spec/15-conformance.md)
