# Design Rationale and Rejected Alternatives

Every significant decision, the alternatives considered, and what each one costs.
A specification that only lists what it chose is not reviewable.

---

## 1 Content addressing everywhere

**Chosen:** every object is identified by the digest of its bytes; the file is a
pack of objects.

**Alternatives:**
- *Named tensors with offsets (safetensors, GGUF).* Simpler, but every capability
  we want — dedup, deltas, resumption, partial fetch, signatures over subsets,
  cache keys — then needs its own mechanism.
- *UUIDs or sequence numbers.* Requires a naming authority and gives no
  integrity.
- *Names + separate hashes.* Two sources of truth, which diverge.

**Costs of the choice:**
- Writers must hash everything (BLAKE3 makes this ~1 s per 12 GB per 16 cores).
- Any edit changes identity — you cannot patch a manifest in place. This is a
  feature for integrity and an annoyance for tooling; the mitigation is that
  rewriting a manifest costs kilobytes, not gigabytes.
- Digest truncation in the index needs careful specification (§01.3).

**Verdict:** the single highest-leverage decision. Five features collapse into
one mechanism.

## 2 Tensor values as expressions

**Chosen:** a tensor's value is a pure expression over chunks and other tensors.

**Alternatives:**
- *Bytes only, with quantization as a separate file format.* The status quo;
  produces the N × M explosion.
- *Bytes plus an optional "recipe" side-channel.* Recipes that are not
  authoritative get out of sync with the bytes they describe.
- *A general compute graph for weights (reuse OMNI-IR).* Tempting for
  uniformity, but a weight expression must be total, analyzable, and cheap to
  type-check; a full IR with control flow is none of those. Keeping OTA
  separate and tiny is what makes range pushdown and candidate enumeration
  five-line functions.

**Costs:**
- Readers need an evaluator (profile C1). A pure-C0 reader cannot load models
  using non-trivial expressions.
- Expression normalization must be specified exactly or identity diverges
  (§04.7.5).
- First-load materialization latency for derived representations.
- Publishers can express things no runtime wants to evaluate; the plan's
  `warnings` and the resolver's failure diagnostics are the guardrails.

**Verdict:** the second-highest-leverage decision, and the one most likely to
draw objections. The C0/C1 split is the concession that makes it safe.

## 3 CBOR for structure, fixed binary for the index

**Chosen:** canonical CBOR for structure objects; a 64-byte-per-entry fixed array
for the object index.

**Alternatives considered and rejected:**

| Alternative | Why not |
|---|---|
| JSON everywhere | no binary, no canonical form, float formatting ambiguity, 33 % base64 bloat |
| Protobuf | no canonical serialization (explicitly disclaimed), schema-dependent (losing the schema loses the data), no self-description |
| FlatBuffers / Cap'n Proto everywhere | zero-copy is real, but schema evolution is by field-id only, canonical encoding is not guaranteed (breaks hashing), and unreadable without the schema |
| MessagePack | close to CBOR but no IETF standard, no tags, weaker canonical rules |
| A bespoke binary encoding | the 50-year test fails immediately |
| CBOR for the index too | parsing 10⁶ entries on every open is unacceptable |

**Costs:** two encodings in one format, which must be justified (they are: they
have opposite access patterns), and the index must be validated as a derived
structure rather than trusted.

## 4 Where the index lives

**Chosen:** front superblock (optional) + back superblock (authoritative) +
64-byte trailer.

**Alternatives:**
- *Front only.* Breaks single-pass streaming writers and appendability.
- *Back only (ZIP, Parquet).* Breaks pure-forward readers (`curl | omni`).
- *Both, with the front authoritative.* Requires a rewrite of the front after
  sealing, which breaks append-only media.

**Cost:** duplicated superblock bytes (kilobytes) and a consistency rule.
**Benefit:** both reader classes are first-class.

## 5 4 KiB default alignment

**Chosen:** `log2_align = 12`, configurable 6–30.

**Alternatives:** 8 bytes (safetensors-like — too small for `mmap`/`O_DIRECT`);
64 KiB (better for object stores, 16× the padding waste); 2 MiB (hugepages;
absurd for small objects).

**Cost:** up to 4 KiB wasted per object — 0.1 % on realistic models, and pathological
only for pathological chunkings (which §performance.9 warns against).

## 6 BLAKE3 default, SHA-256 mandatory

**Chosen:** both required; BLAKE3 primary.

**Alternatives:** SHA-256 only (loses parallel hashing and, critically, verified
range reads); BLAKE3 only (loses interop with OCI, Sigstore, SLSA and every
existing supply-chain tool); SHA-3 (slower, no tree mode).

**Cost:** two implementations. **Benefit:** verified partial reads, which are not
optional for a streaming format, plus ecosystem interop.

## 7 Multi-level IR instead of one abstraction level

**Chosen:** the same computation may exist at `semantic`, `primitive` and
`machine` levels, with shipped lowering rules.

**Alternatives:** ONNX's single primitive level (backends must pattern-match to
recover intent); a single semantic level (no portable execution path for unknown
ops); no graph at all (GGUF — works until a new architecture appears).

**Cost:** more objects, a lowering mechanism, and the risk that levels disagree.
Mitigation: lower levels are `CACHEABLE` and verified by recomputation at V6, so
disagreement is detectable.

## 8 WebAssembly for plugin semantics

**Chosen:** shape/verify/reference-implementation functions as WASM modules under
a deterministic, import-free profile.

**Alternatives:**
- *A custom DSL.* Another language to specify, implement and keep alive for 50
  years.
- *Native shared libraries.* Arbitrary code from a downloaded file. Absolutely not.
- *Nothing (prose only).* Guarantees divergent implementations.
- *A theorem-prover-friendly total language.* Better in principle; no ecosystem.

**Cost:** a WASM engine is a dependency for CX profile (though a minimal
interpreter is a few thousand lines and only needs to be *correct*, not fast),
and WASM's own 50-year survival is an assumption — mitigated by embedding the
WASM spec in the archival profile.

## 9 A total template language instead of Jinja2

**Chosen:** OMNI-CT — pure, total, no host access — with a Jinja2-compatible
subset and a `jinja_compat` rendering for legacy runtimes.

**Alternatives:** ship Jinja2 strings (the status quo: arbitrary code execution
at model load); ship a compiled AST only (loses human readability); no templates
(unusable for chat models).

**Cost:** ~5 % of existing templates may need manual translation (Gate 2 measures
this and will force a decision if the number is worse).
**Benefit:** loading a model can never execute code.

## 10 Optional graph, optional everything

**Chosen:** almost every layer is optional. A valid OMNI file may be
weights-only.

**Alternative:** require a graph, making every model self-describing.

**Rejected because:** it would make importing the existing world impossible
without fabricating graphs, violating "never fabricate" (I1). A weights-only
OMNI file is strictly better than a safetensors file and can be upgraded later;
requiring more would mean nothing imports cleanly and adoption stalls.

**Cost:** portability varies between files, so `omni inspect` must report it
honestly, and it does.

## 11 Criticality bits (the PNG rule)

**Chosen:** four bits — critical, cacheable, safe-to-copy, structural.

**Alternatives:** version gating only (too coarse — one unknown feature blocks
the whole file); "ignore everything unknown" (silently misinterprets models);
"reject everything unknown" (fragments the ecosystem within two releases).

**Cost:** writers must set the bits correctly; a wrong `critical=false` on
something that matters is a silent-corruption bug. Mitigation: the conformance
suite's `forward/` category, and the rule that under-declaring features is a
violation.

## 12 Encryption with plaintext digests

**Chosen:** digests over plaintext, with `convergent` and `random` key modes.

**Cost:** convergent mode leaks plaintext equality (confirmation-of-file
attacks). This is a genuine, unavoidable tradeoff between dedup and
confidentiality, and §12.8 states it rather than burying it. Default is `random`
for public releases.

## 13 Not building a runtime

**Chosen:** OMNI defines no kernels and no execution engine.

**Alternative:** ship a reference runtime that "just runs" any OMNI model.

**Rejected because:** a format that comes with a runtime becomes judged by that
runtime's performance, and its semantics drift toward whatever the runtime does.
ONNX Runtime's relationship to ONNX is the cautionary example. WASM reference
implementations give executable *semantics* without creating a performance
benchmark.

## 14 One endianness

**Chosen:** little-endian only.

**Cost:** big-endian hosts byte-swap. **Benefit:** one digest per tensor, half the
test matrix. The last big-endian platform in ML use is gone; supporting both
would create two identities for the same numbers, which is unacceptable in a
content-addressed format.

## 15 Registry governance

**Chosen:** IANA-style tiers, with `x.*` permanently unregistered and third-party
registration never deniable.

**Alternative:** a curated registry with quality control.

**Rejected because:** a registry that can say no becomes a political chokepoint,
and the format's adoption becomes hostage to committee throughput. PMML died
partly this way. Curation applies only to `omni.*`, where two independent
implementations and conformance vectors are required.

## 16 What we deliberately did not do

| Not done | Why |
|---|---|
| Model-type enumeration | PMML's fatal mistake |
| A "standard" architecture set in the core | makes maintainers a bottleneck on research |
| Built-in versioned "model registry" semantics (tags, branches) | that is a service concern; the format provides digests and lets services build on them |
| Compression research | codecs are pluggable and boring on purpose |
| A query language over metadata | JSON/CBOR output plus existing tools is enough |
| Differential privacy / watermarking primitives | active research; belongs in plugins, not the core |
| Enforcement of licenses or use restrictions | not technically possible; structured signed claims are what a format can honestly offer |
| Mutable state inside a container | breaks immutability, which everything else depends on |
| Ordering guarantees between unrelated objects | would constrain packers with no benefit |

## 17 The three things most likely to be wrong

Stated so reviewers know where to look hardest:

1. **The expression algebra's node set.** 24 operations is a guess informed by
   the schemes in §05 and §08. If a major quantization or adapter family cannot
   be expressed without a plugin, the set is wrong. Phase 1's gate tests this.
2. **The index entry layout.** 64 bytes with a 32-byte truncated digest, an
   8-byte offset and three length/type fields is tuned for today's scale. If
   models with 10⁸ objects become normal, a two-level or prefix-compressed index
   is needed — which is why the index carries its own `fmt_version`.
3. **OMNI-CT's coverage of real chat templates.** If it cannot cover the
   overwhelming majority, publishers will keep shipping Jinja2 and the safety
   property evaporates. Gate 2 measures this explicitly and the design commits to
   reacting rather than declaring victory.

---

**See also:** [§00 Overview](../spec/00-overview.md) · [Comparison](../design/comparison.md) · [Roadmap](../design/roadmap.md)
