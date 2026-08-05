# Reference Implementation Roadmap

The specification is worthless without an implementation that proves it and a
governance structure that outlives its authors. This is the plan, with gates
that can fail.

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

**Gate 0 status.** Reproducible packing: met, and enforced by CI. Two-read open:
met by construction (§02.7), measured at 41 ms for a 10⁶-object index, dominated
by materialising the entry array rather than by I/O. Fuzzing: the in-CI mutation
fuzzer is clean over millions of iterations; the 72-hour libFuzzer run is a
release activity and has not been performed. **Index latency: not met** —
p99 ≈ 590 ns against a 500 ns target, after a 6.5× improvement from implementing
the bucket table §02.6.1 already specified. The measurement and what it implies
are in [`performance.md`](performance.md) §11. The gate has done its job: it
found the gap while the format is still a draft.

## Phase 1 — Prove the value layer (months 3–7)

**Goal:** tensor expressions are implementable and actually save bytes.

Deliverables:
- Full dtype algebra with bit-exact encode/decode and all rounding modes.
- Layout math: strided, tiled, packed, blocked-scaled, interleaved.
- `omni-eval`: expression evaluation, range pushdown, fusion, caching.
- Quantization schemes: affine, sym, codebook, nested; GPTQ/AWQ/NF4/MX/GGUF-K
  structural mappings.
- Sparsity schemes.
- `omni-import-safetensors` ✅, `-peft` ✅, `-gguf`, `-gptq`, `-awq`.
- `omni-export-safetensors` ✅, `-gguf`.
- `omni delta`, `omni adapter`, `omni convert`.
- Conformance corpus: `numeric/`, `roundtrip/`, `valid/features`.

**Gate 1:** lossless round-trip for safetensors, GGUF, GPTQ, AWQ, PEFT, verified
bit-exactly on ≥ 100 real models. A published delta-size study over ≥ 50 real
base/fine-tune pairs. Differential test against `llama.cpp`'s dequantization for
every GGUF K-quant type: zero mismatches. *If the structural GGUF mapping cannot
be made bit-exact, §05.2.4 is wrong and gets revised.*

**Gate 1 status.** Not met. The *format* side is done and tested: the dtype
algebra, the layouts, the expression algebra with range pushdown, the sparsity and
quantization catalogues, `omni delta` and `omni adapter`, and — as of the codec
work — `zstd` in both directions, checked against libzstd on every push. **Two importers and one exporter now exist**:
safetensors in both directions and PEFT LoRA in, with the I1–I6 and E1–E4
contracts of [`import-export.md`](import-export.md) implemented rather than
summarised — every tensor verified byte-for-byte against the source on
import, every loss named before an export writes anything, and a round-trip whose
tensor digests are checked against a fixture built from the format's own
definition in Python. That is two rows of a 25-row capability matrix. GGUF, PyTorch,
ONNX, GPTQ and AWQ do not exist, so the gate's actual asks are still
untouched: no round-trip over 100 real models, no delta-size study over 50 real
pairs, and no differential test against `llama.cpp`'s dequantization. The
catalogue and the one importer are necessary for that work and are not a
substitute for it.

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

**Gate 2 status.** OMNI-IR parses, verifies, prints and rewrites; `omni.core` is
frozen and `omni.tensor`/`omni.nn`/`omni.quant`/`omni.io` are defined with per-op
versions; `graph synthesize` builds a decoder graph from `arch.params` and
`graph lower` applies the shipped lowerings. The tokenizer IR and OMNI-CT run
their own conformance vectors. The WASM host of §11.6 exists and runs plugin
expression ops under the restricted profile. **Not met, and not close:** one
architecture family is synthesizable rather than ten, there is no reference
*interpreter* for the IR (verification and rewriting are not execution), the
tokenizer vectors are this repository's rather than 200 real ones, and no
Jinja2 → OMNI-CT translator exists, so the 95 % figure is untested. The
coverage numbers in this gate are the point of it; none of them has a value yet.

## Phase 3 — Prove the distribution layer (months 12–18)

**Goal:** it works at internet scale.

Deliverables:
- HTTP and object-store stores with range coalescing and resumption.
- OCI push/pull with `by-novelty` pack partitioning; referrers for adapters and
  signatures. ◐ (the mapping and a pushable layout exist; the registry client does not)
- Verified streaming (Bao) end to end; progressive load.
- `omni mount` (FUSE) with synthesized safetensors and tokenizer views.
- `omni serve`: an object server. ✅
- Signatures (Ed25519, ES256, ML-DSA), Sigstore keyless, in-toto/SLSA
  provenance, revocation.
- Published benchmark suite (§performance.10).

**Gate 3:** a mirror of ≥ 10 000 real models, with measured dedup, delta sizes
and load times published as signed OMNI containers. TTFT over a throttled link
demonstrably better than the incumbent. A third-party security review of the
parser and the signature stack. *This gate is where the storage-savings claims
in this proposal are confirmed or retracted.*

**Gate 3 status.** Not met. The *mechanisms* of §13.4 now exist and are tested
against a real socket: an HTTP/1.1 range store with coalescing, retry and
per-object digest verification; the `.omni.idx` sidecar, which turns a
three-request open into none; the index-only container of §13.8; Bao verified
streaming (§13.3); and `store::FileStore` for the local half. CI fetches a
container over HTTP by range and checks that the reassembled object graph is
byte-identical to the original. What the gate actually asks for is untouched:
there is no mirror of 10 000 models, no measured dedup or delta-size or
load-time figures, no TTFT comparison over a throttled link. `omni serve` implements §13.4.3's
per-object URLs alongside the pack, so CI exercises the client against a real
server rather than a mock. §13.5's mapping is implemented: a container becomes an
OCI image layout that `oras` can push, with the OMNI Manifest as the config and
the container cut into pack layers, and CI validates the layout against the
image-spec's rules and reassembles it byte for byte. `https://` is refused rather
than downgraded — TLS needs a dependency this crate does not have — there is no
registry *client* and no `mount`, and nothing has been pushed anywhere, so the
*distribution* claims remain claims. The signature stack of §12.5 is implemented
(Ed25519, COSE_Sign1, trust policies) and has had no third-party review.

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

**See also:** [Comparison](comparison.md) · [§15 Conformance](../spec/15-conformance.md)
