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

**Gate 0:** a 140 GB container opens in 2 reads; an index lookup compares **≤ 3
entries at 10⁶ objects**, with p99 latency under 1 µs on a shared cloud VM and
under 500 ns on a quiet machine; `pack` is byte-reproducible; 72 h of fuzzing
finds no crash, hang, or OOM. *If the index cannot hit that entry count, the
index format changes now, not later.*

**This gate was loosened, on evidence, and the original wording is kept here so
the change is visible rather than quiet.** It read *"index lookup p99 < 500 ns at
10⁶ objects … if the index cannot hit that latency, the index format changes
now"*. Two things were wrong with it. A latency in nanoseconds is not a property
of a format — it is a property of a machine, and the machine this was measured on
moves 30 % between runs of the same binary, which is more than any change to the
index would produce. And the criterion could not be *acted* on: the measurement
(`performance.md` §11.3) traced the gap to one DRAM access and a page walk over a
61 MiB working set, which is the floor for a 64-byte index entry — reachable only
with huge pages, which need `mmap` and therefore the `unsafe` this build forbids.
A gate that fails for a reason the format cannot fix is not a gate on the format.

So the normative half is now the structural number, which every implementation
measures identically and which the format *does* control, and the latency is
stated as two figures against named hardware instead of one figure against none.
Loosening it is a real concession: the original number was a promise about what a
lookup would cost, and this is a weaker promise. It is the one the evidence
supports.

**Gate 0 status.** Reproducible packing: met, and enforced by CI. Two-read open:
met by construction (§02.7), measured at 41 ms for a 10⁶-object index, dominated
by materialising the entry array rather than by I/O. Fuzzing: the in-CI mutation
fuzzer is clean over millions of iterations; the 72-hour libFuzzer run is a
release activity and has not been performed. **Index lookup: met, against the
restated criterion above.** An entry count of 2.20 against a bound of 3; p99 ≈ 690
ns on the shared VM this was measured on, inside the 1 µs the gate now names for
that class of machine, and outside the 500 ns it names for a quiet one — which is
recorded as unverified rather than claimed, because no quiet machine has run it.

How that number was reached, and why the gate changed shape around it: The first pass measured p99 ≈ 590 ns against a 500 ns target after a
6.5× improvement from implementing the bucket table §02.6.1 already specified. The
second pass found something more useful: that p99 is not reproducible on the test
machine — the same binary measures 640, 781 and 822 ns across three runs — so
every recorded improvement smaller than ~200 ns was unfalsifiable. `omni bench`
now reports rounds and their spread, and reports **entries compared per lookup**,
which is the part of the cost this code decides and is the same on every machine.
It is **2.20**, down from 8.62, from interpolating inside the bucket — digests are
cryptographic hashes, so where an entry sits in its bucket is predictable. That
change touches 3.9× less index per lookup and does *not* move the p99 on this
hardware, and both halves of that are recorded.

A lookup is now one L2-resident bucket read plus about one DRAM access and a page
walk over a 61 MiB working set, which is the floor for a 64-byte index entry. The
remaining gap is attributable to hardware and to the reference implementation's
no-`unsafe` rule — huge pages would remove most of the tail and need `mmap` — and
not to the index format, so the format does *not* change on the strength of this
gate. What replaces the deferred decision is a criterion that can be measured
without a quiet machine: any proposal to change the index must beat 2.20 entries
compared per lookup at 10⁶ objects. The reasoning and the two experiments, one
accepted and one rejected, are in [`performance.md`](performance.md) §11.

The gate has still done its job — arguably twice. It found the gap while the
format is a draft, then it found that the ruler was bent, and the second finding
is why the gate itself now reads differently. **Gate 0 is met on every criterion
except the 72-hour fuzz run**, which is a release activity.

## Phase 1 — Prove the value layer (months 3–7)

**Goal:** tensor expressions are implementable and actually save bytes.

Deliverables:
- Full dtype algebra with bit-exact encode/decode and all rounding modes.
- Layout math: strided, tiled, packed, blocked-scaled, interleaved.
- `omni-eval`: expression evaluation ✅, range pushdown ✅, fusion ✅, caching ✅.
- Quantization schemes: affine, sym, codebook, nested; GPTQ/AWQ/NF4/MX/GGUF-K
  structural mappings.
- Sparsity schemes.
- `omni-import-safetensors` ✅, `-pytorch` ✅, `-hf-repo` ✅, `-peft` ✅, `-gptq` ✅, `-awq` ✅, `-gguf` ✅.
- `omni-export-safetensors` ✅, `-gptq` ✅, `-awq` ✅, `-gguf` ✅.
- `omni delta` ✅, `omni adapter` ✅, `omni convert` ✅ (`--cast` and
  `--requantize`, the latter with §05.5's provenance).
- Conformance corpus: `numeric/` ✅, `roundtrip/` ✅, `valid/features` ✅.

**Gate 1:** lossless round-trip for safetensors, GGUF, GPTQ, AWQ, PEFT, verified
bit-exactly on ≥ 100 real models. A published delta-size study over ≥ 50 real
base/fine-tune pairs. Differential test against `llama.cpp`'s dequantization for
every GGUF K-quant type: zero mismatches. *If the structural GGUF mapping cannot
be made bit-exact, §05.2.4 is wrong and gets revised.*

**Gate 1 status.** Not met. The *format* side is done and tested: the dtype
algebra, the layouts, the expression algebra with all three of §04.7.4's
refinements — range pushdown, elementwise fusion, and a result cache keyed by the
§04.7.5 expression identity — the sparsity and
quantization catalogues, `omni delta`, `omni adapter` and `omni convert` — whose
`--requantize` is now a search rather than a refusal, with the round-to-nearest
baseline that needs no data and a clip search that weights its objective by a
calibration set's activation magnitudes, both recording §05.5's provenance so
"which calibration set produced this?" is a field lookup and "none" is an answer
rather than an empty field — and — as of the codec
work — `zstd` in both directions, checked against libzstd on every push. Fusion
and caching are the two the deliverable named and the code did not have; they are
switchable (`omni cat --no-fuse`, `--cache SIZE`) precisely because they are
claims about *cost*, so CI checks that the values are bit-identical either way
and that the counts the tool prints are real. **Seven
importers and five exporters now exist**: safetensors, GGUF, PEFT, GPTQ and AWQ in
both directions, PyTorch `.bin` in, and a whole Hugging Face repo in, with the
I1–I6 and E1–E4 contracts of
[`import-export.md`](import-export.md) implemented rather than summarised — every
tensor verified byte-for-byte against the source on import, every loss named
before an export writes anything, and a round-trip whose tensor digests are
checked against a fixture built from the format's own definition in Python. The
quantized importers go further, because byte identity cannot catch a
misread packing: every layer is dequantized through the expression graph and
compared against arithmetic done in Python, so §05.2.2's, §05.2.3's and §05.2.4's
structural mappings are checked against the formats and not against this
implementation.
The PyTorch importer is the §12.10 one: a restricted unpickler with an opcode
allowlist and nineteen resolvable symbols, checked in CI against a payload that
Python's own `pickle.loads` is first shown to execute. It imports only —
§12.10 clause 4 says never to re-emit pickle.

The repo importer is the one that makes the format usable rather than
demonstrable: `omni import hf <dir>` turns the five files a model on the hub
actually consists of — sharded weights, `config.json`, `tokenizer.json`, the
chat template, the generation defaults — into one container where the tokenizer
shipped with those weights is addressed by digest instead of downloaded
separately and hoped about.

**GGUF is the row this gate names, and it now exists in both directions.**
Eleven block types — `Q4_0` through `Q6_K` — as `dequantize` expressions over
literals whose `packed` layouts name the bit widths, with **no re-encoding**: a
block is a struct, so the import keeps every source byte regrouped by field and
the export re-interleaves them, which makes the round trip byte-exact by
construction rather than by careful rounding. CI checks it three ways on every
push: the file comes back identical, every block is dequantized twice inside the
import and compared element by element, and 5 760 float32 values are compared
bit for bit against `tools/gguf-fixture.py`, a Python implementation written from
the GGML block layouts. The `IQ*` types are refused by name — their codebooks are
compiled into llama.cpp rather than stored in the file, and §05.6 rule 1 forbids
inventing a dequantization.

Writing it produced one finding worth more than the row: **GGUF's tokenizer is
not self-contained.** The file carries the vocabulary, the merges and the scores,
but `tokenizer.ggml.pre` names a pre-tokenizer whose regexes live in llama.cpp's
source, so those keys do not determine where a token begins. A §06.7 tokenizer
built from them would decode correctly and encode differently from the model it
shipped with. So none is built, the keys are preserved, and the capability matrix
row moved from ● to ◐ on the evidence rather than staying at the number it was
first written with.

That is seven rows of a 25-row capability matrix — eight with ONNX, which
belongs to Phase 2's list rather than this one because it is a *graph* importer.
EXL2 does not exist.

**The gate's own two measurements now have an instrument.**
[`tools/corpus.py`](../../tools/corpus.py) runs them over whatever corpus it is
pointed at: `roundtrip` imports every model in a directory, exports it back and
compares **byte for byte**, and `delta` runs `omni delta` over a file of
base/fine-tune pairs and reports the distribution. Each prints the gate's own
threshold beside its figure and **exits 3 when the corpus is short of it**, so a
run is a check rather than a report somebody has to read. CI runs both over a
corpus this repository can build — two models, two pairs — which measures the
instrument rather than the gate, and says so.

What remains is the corpus: no round-trip over 100 real models, no delta-size
study over 50 real pairs, and no differential test against `llama.cpp`'s own
binary — the K-quant dequantization is checked against two independent
implementations of the *format*, which is the part that can be done without a
corpus, and not against the binary the gate names. The first two now need
downloading models and running one command; the third needs a build of
llama.cpp.

## Phase 2 — Prove the semantic layer (months 7–12)

**Goal:** the model is self-describing.

Deliverables:
- OMNI-IR: parse, verify, print; `omni.core` + `omni.tensor` + `omni.nn` v1.
- Dialect mechanism, op versioning, declarative rewrites, lowering.
- `omni-plugin`: WASM host with the restricted profile, fuel metering ✅, and
  §07.4.2's `shape_fn`/`verify_fn` execution ✅.
- Tokenizer IR + conformance vectors + Jinja2 → OMNI-CT translator.
- `omni graph synthesize` for registered families.
- `omni-import-onnx` ✅, `-export-onnx` ✅.
- Capability negotiation and `omni plan`.

**Gate 2:** ten architecture families (§07.8) expressed end-to-end and executed
by a reference interpreter with outputs matching the source framework within
declared tolerance. Tokenizer vectors pass for ≥ 200 real tokenizers.
Jinja2 → OMNI-CT translation succeeds for ≥ 95 % of chat templates on a public
hub snapshot, with the failures analyzed and published. *If OMNI-CT cannot cover
95 %, the language grows or the fallback story changes.*

**Gate 2 status.** Not met, and closer than it was: the gate wants ten
architecture families and this build now synthesizes **eleven** —
`transformer.decoder`, `transformer.encoder`, `transformer.moe`,
`cnn.classifier`, `mlp`, `rnn.lstm`, `rnn.gru`, `gnn.mpnn`, `rl.actor_critic`,
`audio.encoder` and `ssm.mamba`, the last of which needed §07.8.1 written before
it could exist. That is more than the *count* the gate names; what it
also asks for is outputs matching the source framework within a declared
tolerance, and there is no source framework here to compare against, so the
count is met and the comparison is not.

OMNI-IR parses, verifies, prints and rewrites; `omni.core` is frozen and
`omni.tensor`/`omni.nn`/`omni.quant`/`omni.io` are defined with per-op versions;
`graph synthesize` builds a graph from `arch.params` and `graph lower` applies
the shipped lowerings. The tokenizer IR and OMNI-CT run their own conformance
vectors. The WASM host of §11.6 exists and runs plugin expression ops under the
restricted profile — and, as of §07.4.2's calling convention being written down,
the `shape_fn` and `verify_fn` a dialect ships with the model. That closes the
gap between §07.2's claim about *execution* (an unknown op is recoverable if its
lowering is shipped) and the same claim about *verification*: a dialect this
build has never heard of is now decided rather than indeterminate when the model
brought its semantics. What is still absent is `ref_impl` execution — the
interpreter refuses an op it does not know by name rather than falling back to a
shipped reference implementation. A reference interpreter (`omni graph run`) executes all of
`omni.core` including its control flow, all 31 `omni.tensor` ops with a general
`einsum`, `omni.quant`'s four, and **all** of `omni.nn` — `ssm_scan` included,
which was refused for a draft because §07 named it without defining it. §07.8.1
now states the operand roles, the per-channel and per-position Δ, and the
discretization rule, so the op is decidable rather than a place where two
conforming readers could disagree about what a Mamba model computes.

**Every family is executed, not merely emitted**, and each is checked against a
property of *that* architecture rather than against "it produced numbers": the
encoder's position 0 must move when a later token changes and the decoder's must
not; the mixture's output must change when only the router changes; the
recurrence's first step must not see the last input and its last step must;
the graph network's node must move when its neighbour's features move and must
not when a non-neighbour's do; the causal audio encoder's earlier frames must not
move when a later frame does. That discipline has now paid three times. It found
the decoder attending across *heads* instead of positions while passing
verification. It found `core.scan` declared with one result in the op registry
and returning two in the interpreter — the same graph ran correctly and failed
verification, and nothing had used both results until an LSTM did. And
synthesizing the GNN row found that `tensor.scatter` could not aggregate at all:
it wrote element for element, so two edges into one node lost a message. That
one is now **fixed rather than recorded**: §07.4 spells `reduction` on `scatter`
— ONNX's `ScatterElements` spelling, not a new one — as an optional attribute
with a specified default, so the op stays at version 1, and `gnn.mpnn`
aggregates with a scatter-add over an edge list instead of carrying a dense
`[E, N]` incidence matrix that was quadratic in the thing the operation is
sparse in. An unrecognised reduction is refused rather than silently treated as
`replace`, because the two produce different models from one graph.

**Jinja2 → OMNI-CT now converts 14 of the 15-template corpus**, up from 10. The
three blockers §06.9 had recorded as gaps in itself — no loop variable, no slice
form, two missing standard-library entries — are closed, each in the form the
analysis named. The one still refused calls `raise_exception`, and it should be:
a total language has no failure form. The measurement is still of this
repository's corpus rather than of a hub snapshot.

**ONNX now imports and exports**, which is the deliverable this phase names and
the first time OMNI-IR has had to absorb somebody else's graph rather than
synthesize one. The mapping's rule is narrow on purpose: an ONNX op is translated
only when one OMNI op means exactly what it means (twenty-four op types), and
everything else is carried in a compat dialect named after its ONNX domain, at
the opset the file imported. Importing `Relu` as `maximum(x, 0)` would have made
the export a peephole matcher over the graph — the thing §07.1 criticises ONNX's
backends for having to do — so it is carried instead. CI checks both halves: a
file whose every node maps imports, verifies, *executes* against arithmetic done
in Python, and exports byte for byte; a file whose nodes do not still verifies,
copies, signs and round-trips, reports its ops **indeterminate** rather than
invalid, and refuses exactly one thing — execution — by name. That is §11.3's
forward-compatibility claim tested on a real format rather than a synthetic one.

Two things it found are worth more than the row. **ONNX and OMNI now check each
other's shape functions**: the import types every value both ways and treats a
disagreement about a concrete dimension as an error, because one of the two
readers is then wrong about the model. And **exporting this repository's own
transformer measures what the opset costs**: lowered to primitives, 49 nodes map
and three kinds do not — `omni.nn/attention` and `omni.nn/rope`, which are the
semantic ops §07.2 exists to keep, and `omni.tensor/rsqrt`, which ONNX has no
operator for.

**Gate 2's two corpus measurements have an instrument too.**
`omni jinja --coverage <dir>` runs the translation over a directory of chat
templates instead of the built-in fifteen, and
[`tools/corpus.py`](../../tools/corpus.py) `tokenizers` walks a directory of
`tokenizer.json` files, importing each and reporting how many carry conformance
vectors *separately from* how many imported — because that distinction is the
whole of what the gate asks. Vectors have to come from the source tokenizer: one
generated by the encoder it is meant to test cannot fail, so a tokenizer that
imported is not a tokenizer that passed.

**Still not met:** the eleven families are executed against arithmetic done in the
tests rather than against PyTorch, the tokenizer vectors are this repository's
rather than 200 real ones, and the translation figure is a percentage of a
15-template corpus and not of the public snapshot the gate names. ONNX's
round-trip is checked against a fixture written from the format's own definition
rather than against files a real exporter produced. Those need corpora and a
second framework; what they no longer need is code.

## Phase 3 — Prove the distribution layer (months 12–18)

**Goal:** it works at internet scale.

Deliverables:
- HTTP and object-store stores with range coalescing and resumption.
- OCI push/pull with `by-novelty` pack partitioning; referrers for adapters and
  signatures. ◐ (the mapping, the layout and the client exist and CI pushes to a
  real registry; the referrers API and `by-novelty` partitioning do not)
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
byte-identical to the original. `omni serve` implements §13.4.3's per-object URLs
alongside the pack, so CI exercises the client against a real server rather than
a mock.

**§13.5 is now end to end.** A container becomes an OCI image layout — the OMNI
Manifest as the config, the container cut into pack layers, validated against the
image-spec's own rules and reassembled byte for byte — and `omni oci push` and
`omni oci pull` carry it over the distribution API. CI runs a real `registry:2`
and pushes to it: the model goes up, comes back down byte-identical, verifies at
V6, and can be pulled by digest rather than by tag. The gate's phrase is
"published as signed OMNI containers", so a signed container makes the same
trip and its §12.5 attestation still verifies on the far side — which is the
thing a transport that changed one byte would break. `https://` is still refused
rather than downgraded, because TLS needs a dependency this crate does not have,
so what this reaches is a plaintext registry: a local one, a mirror, or anything
behind a terminator. A registry that answers `401` is told apart from one that
answers `404`, and the bearer-token realm it names is reported rather than
guessed at.

**The dedup claim now has a number attached, and the number says something more
specific than the claim did.** Pushing the same container under a second tag
uploads *nothing*: the registry answers 200 to every blob `HEAD`, which is
content addressing confirmed by the party that would know. Pushing a *modified*
model shares nothing at all — objects are placed in digest order (§02.4), so one
changed tensor moves the file offset of everything after it and every pack from
there on is a different blob. `oci.rs` said that in as many words before any of
this was pushed anywhere, and CI now measures it rather than repeating it. The
saving is in the layer above: the same fine-tune published as a **delta
container** (§08.6) uploads **20 %** of what the full model costs. That figure is
*modelled*, not measured at scale — these are toy containers, and what is being
demonstrated is the mechanism rather than a ratio that generalises.

**The TTFT comparison now exists, in the narrower form that is honest.** The
compute half of time-to-first-token is the same whatever the file looks like, so
what a *format* decides is the I/O half: how many bytes cross the wire before the
first tensor can be read. `omni serve --throttle` rate-limits the server, because
on a loopback socket every layout is instantaneous and the difference hides
exactly where it is supposed to matter, and `omni fetch --first-tensor` reads one
tensor two ways over that link — by range, and by fetching the whole file, which
is what a reader with no index has to do and what the hub tooling in fact does
today. On the worked example over a 200 KiB/s link: **37.7 KiB and 178 ms by
range against 111.2 KiB and 552 ms for the file**.

That ratio is a property of the *model* — one tensor's share of one small file —
and the number underneath it is the property of the *format*: **5.7 KiB of
framing and index to reach any tensor at all**, which is §02.7's two-read open
measured over a socket rather than asserted. On a real model the ratio is large
because that cost is nearly constant, so it is the second number that
generalises. The whole-file row is marked *modelled* in the output itself: it is
what the incumbent tooling does, not a measurement of any particular runtime, and
the gate's "demonstrably better than the incumbent" needs the incumbent, on a
real model, over a real link.

What the gate actually asks for is still untouched: there is no mirror of 10 000
models, and no dedup, delta-size, load-time or TTFT figures over real ones. There
is no `mount`, no referrers API and no `by-novelty` partitioning. The signature stack of §12.5 is implemented (Ed25519,
COSE_Sign1, trust policies) and has had no third-party review. The distribution
*mechanisms* are now demonstrated rather than claimed; the distribution
*measurements* are what remains, and they need a corpus rather than more code.

## Phase 4 — Ecosystem (months 18–30)

- Bindings: Python, C, C++, Go (pure-Go C0 reader), Java, Swift, JS/WASM.
  **The C ABI this phase depends on is built** (`reference/omni-ffi`,
  `omni.h`): opaque handles, no panic across the boundary, CLI-matching status
  codes, and DLPack out to PyTorch/JAX/NumPy without a copy, with a C program
  in CI driving open → verify → walk → bytes → plan → DLPack. It reads only;
  writing a container from C is still Rust-side. Every binding in this list is
  now a layer over an existing ABI rather than a new parser.
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
| **Spec ambiguity causes divergent implementations** | medium | Differential testing from Phase 1; conformance suite is normative; WASM reference semantics for anything subtle. A second reader in a second language (`bindings/python/omni.py`, pure Python, no dependencies) is checked against the Rust one on every push — two implementations agreeing on a byte is worth more than one asserting it |
| **A dominant vendor forks it** | medium | Permissive licensing + royalty-free IPR make forking legal but pointless; the conformance mark and mirrored registry are the coordination points |
| **Registry capture / politicization** | medium | `x.*` namespace, first-come-first-served for third parties, frozen core, mirrorable signed registry (§11.7) |
| **Hash algorithm break** | low, high impact | Agility designed in (§12.11); migration costs one hashing pass, zero re-uploads |
| **Container format needs a breaking change** | low, very high impact | `header_size` growth, skippable segments, index `fmt_version`, feature flags — the specific mechanisms that prevent it |
| **Scope creep** | high | The C0 budget (§SDK.5) is a hard gate on every proposal, and it is now *measured* rather than modelled: the pure-Python C0 reader is 878 lines against a ~3 000 line budget |

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
