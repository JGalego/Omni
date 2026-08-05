# OMNI/1.0 — §15 Conformance and Validation

A specification without an executable definition of "conforming" becomes folklore
within two years. This section is normative.

## 15.1 Validation levels

`omni verify` implements a strict ladder. Each level subsumes the previous.

| Level | Name | Checks | Cost |
|---|---|---|---|
| **V0** | Framing | magic, header CRC, `header_size`, trailer, segment chain, offsets/lengths in range, alignment, zero padding | O(segments) |
| **V1** | Index | index header, sortedness, bucket table consistency, entry bounds, no duplicate digests, aux table well-formed | O(entries) |
| **V2** | Structure | every parsed object is canonical OMNI-CBOR, schema-valid, `t`/`v` present, refs well-typed | O(structure bytes) |
| **V3** | Integrity | digest of every object read (V3-selective) or every object present (V3-complete) matches its index/ref | O(bytes hashed) |
| **V4** | Graph | reachability from roots, no dangling *required* refs, parent chain depth, DAG completeness, criticality resolution | O(objects) |
| **V5** | Semantics | tensor shape/dtype inference agrees with declarations; expression typing; quantization scheme consistency; adapter attach validation; IR verification via dialect `verify_fn` | O(descriptors) |
| **V6** | Derived | recompute every `CACHEABLE` object and compare (index, name index, Bao trees, materialized tensors, lowered graphs) | expensive |
| **V7** | Authenticity | signatures, trust policy, revocation, freshness | O(signatures) |
| **V8** | Provenance | full lineage, parent resolution, attestation chain, transparency-log inclusion, reproducibility replay | network |

Exit codes are distinct per level so CI can gate precisely, and the report
distinguishes three outcomes at every level: **valid**, **invalid**, and
**indeterminate** (e.g. unknown critical extension, missing parent, unsupported
signature algorithm). Reporting *indeterminate* as *invalid* is itself a
conformance violation.

## 15.2 Normative rules (checklist)

### Container (V0–V1)

- R-C01 `magic` exactly `89 4F 4D 4E 49 0D 0A 1A`.
- R-C02 `header_crc32c` correct over bytes `0..124`.
- R-C03 `header_size ∈ [128, 4096]`; readers skip to it.
- R-C04 `log2_align ∈ [6, 30]`.
- R-C05 Every segment: `seg_magic == "OSEG"`, both CRCs correct, `payload_len`
  within file.
- R-C06 Segments tile the file with no overlap and no gaps other than declared
  padding.
- R-C07 All padding bytes are `0x00`.
- R-C08 `BLOB` segment payloads and every data object within them begin at a
  multiple of `A`.
- R-C09 In a sealed file, trailer present, `magic_end` correct, superblock
  digest matches.
- R-C10 If both superblocks present, they are byte-identical.
- R-C11 Index sorted strictly ascending by `digest32`; no duplicates.
- R-C12 Every index `offset + stored_len ≤ file_size`, or `EXTERNAL` is set.
- R-C13 `logical_len / stored_len ≤ 1000` unless the high-ratio feature is
  declared.

### Encoding (V2)

- R-E01 Every structure object is canonical per §03.2 (D1–D10).
- R-E02 No duplicate map keys anywhere.
- R-E03 All text is valid UTF-8 in NFC.
- R-E04 Only registered tags appear.
- R-E05 `t` and `v` present on every structure object.
- R-E06 A schema is available (embedded or known) for every `t` that is
  `CRITICAL`.

### Objects and graph (V3–V4)

- R-O01 `H(payload) == digest` for every object checked.
- R-O02 `otype` in the index matches the object's `t`-implied type.
- R-O03 Every ref's target, when present, has the declared `otype`.
- R-O04 Roots exist and are `Manifest` objects.
- R-O05 No dangling ref that is `CRITICAL` and required by the selected plan.
- R-O06 Parent chain depth ≤ declared max (default 32).
- R-O07 Every object marked `CACHEABLE` is not the sole source of any canonical
  value (the "droppability" invariant, §00.5).

### Tensors (V5)

- R-T01 Declared `shape`/`dtype` equal the inferred shape/dtype of `value`.
- R-T02 `ChunkList.total` equals the sum of chunk logical lengths and equals the
  byte size implied by `shape`, `dtype` and `layout`.
- R-T03 Layout is sufficient to compute every element's bit position; strides,
  packing and blocking are internally consistent.
- R-T04 Quantization scheme tensors (`scale`, `zero`, `order`, codebooks) have
  shapes consistent with the declared block structure.
- R-T05 Expression trees are acyclic (guaranteed by content addressing for
  cross-object refs; checked explicitly for intra-object nesting), depth ≤ 256.
- R-T06 `approx` wraps every lossy subtree; no lossy codec on a non-cacheable,
  non-`approx` object.
- R-T07 Declared `stats`, when present and checked, match recomputation.

### Metadata and tokenizer (V5)

- R-M01 `params_total`, when present, equals the sum over weight-semantic tensors
  (`omni verify --strict` recomputes).
- R-M02 `rope.interleaved` present whenever `rope.kind == "rope"`.
- R-M03 Tokenizer conformance vectors, when present, reproduce exactly.
- R-M04 Chat template vectors, when present, render exactly.
- R-M05 Vocabulary size matches the embedding tensor's vocabulary axis when both
  are present.

### Adapters (V5)

- R-A01 `base` digest resolves (or is declared unresolvable).
- R-A02 Every `attach.select` matches at least one base tensor, or the adapter
  declares `allow_unmatched`.
- R-A03 Shapes and `require` constraints hold for every match.

### Training state (V5)

- R-N01 Every object reachable only through `TrainingState` is removable by
  reachability alone, leaving the weight tensors' digests unchanged (§09.1).
- R-N02 No inference-relevant object references a training object. The
  `Model.training` reference itself is the one permitted edge.
- R-N03 `omni inspect` reports training-state size separately from weights.
- R-N04 A `ShardMap` placement's shards tile its logical tensor exactly: no gaps,
  no overlaps.
- R-N05 Every sharding and shard coordinate names a mesh dimension that exists,
  with an extent that matches the declared number of parts.
- R-N06 A `flat_params` entry's `numel` equals the product of its `orig_shape`.

### Graph / OMNI-IR (V5)

- R-I01 Every value is defined exactly once within its function (SSA).
- R-I02 Every use refers to a value already in scope, i.e. defined earlier in the
  same block or in an enclosing one.
- R-I03 `entry` names a function the module defines.
- R-I04 Every op's dialect is declared in the module's `dialects`.
- R-I05 The op exists in that dialect at that version, or a shipped rewrite
  (§07.7) migrates or lowers it. Failing both is *indeterminate*, not invalid.
- R-I06 Declared result types equal the types inferred from the operands.
- R-I07 Operand and result counts, required attributes, region counts and block
  terminators match the op's contract.
- R-I08 A `token` value is consumed exactly once, so the effect order it
  expresses is total (§07.3.2).
- R-I09 The function's symbolic-dimension constraints are satisfiable.
- R-I10 A `core.constant` naming a tensor agrees with that tensor's declared
  shape and dtype.
- R-I11 A module carrying `lowered_from` is below the `semantic` level.

### Transport (V3)

- R-X01 A `.omni.idx` sidecar's header CRC and superblock digest verify, and its
  `hash_algo` agrees with the container header it carries (§13.4.1).
- R-X02 A sidecar's `file_size` equals the `file_size` of the container header it
  carries; and before an offset from it is used, the served container matches both
  that `file_size` and the sidecar's root digest.
- R-X03 Bytes received over a transport are verified against the object digest —
  or the Bao tree for a partial object — before use (§13.3, §13.7).

### Security (V7)

- R-S01 Signature covers the manifest with `attestations` removed.
- R-S02 `summary.canonical_digest` matches recomputation.
- R-S03 No `executable: true` object outside a `RuntimeCache`.
- R-S04 No object requires code execution to parse.

## 15.3 The conformance suite

Distributed as a versioned OMNI container (`omni-conformance-1.0.omni`) plus a
test-runner protocol.

### 15.3.1 Corpus categories

| Category | Count (target) | Purpose |
|---|---|---|
| `valid/minimal` | 12 | smallest legal files for each profile |
| `valid/features` | ~180 | one file per feature: every dtype, layout, quant scheme, sparsity scheme, expression node, adapter method, IR construct |
| `valid/scale` | 8 | 1 M objects; 4 GB single tensor; 100 k-op graph; 32-deep parent chain |
| `valid/architectures` | ~30 | transformer, MoE, Mamba, RWKV, CNN, diffusion bundle, GNN, RL policy, speech, video, multimodal |
| `invalid/framing` | ~80 | every R-C rule violated exactly once |
| `invalid/encoding` | ~60 | non-canonical CBOR, duplicate keys, bad UTF-8, denormalized Unicode |
| `invalid/semantic` | ~70 | shape/dtype mismatches, bad layouts, inconsistent quant params |
| `hostile/` | ~200 | fuzzer-derived: overflow offsets, decompression bombs, deep nesting, huge declared lengths, digest mismatches, truncation at every segment boundary, malicious locators |
| `roundtrip/` | ~50 | import→export→import fidelity for every supported source format |
| `numeric/` | ~120 | bit-exact dtype encode/decode vectors, rounding modes, dequantization vectors per scheme |
| `tokenizer/` | ~40 | encode/decode vectors incl. Unicode edge cases, ZWJ sequences, combining marks, byte fallback |
| `forward/` | ~25 | files using *fictional* future features, to test graceful degradation |

The `forward/` category is unusual and important: it contains files with unknown
segment kinds, unknown otypes, unknown critical and non-critical extensions,
future schema versions, and unknown dtypes — each with a declared expected
behaviour. A reader that "handles" these by rejecting the whole file fails
conformance.

### 15.3.2 Runner protocol

```
$ omni-conformance run --impl ./my-reader --profile C1
case valid/features/dtype-f4e2m1                 PASS
case invalid/framing/pad-nonzero                 PASS   (rejected, correct code)
case forward/unknown-critical-otype              FAIL
    expected: load succeeds, execution refused, exit 3 (INDETERMINATE)
    actual:   exit 1 (INVALID)
case hostile/bomb-ratio-1e6                      PASS   (refused, no OOM)
…
231/247 passed   profile C1: NOT CONFORMANT (3 required cases failed)
```

Implementations declare conformance as
`OMNI/1.0 C0 C1 C2 (suite 1.0.3, 247/247)`.

### 15.3.3 Differential testing

The suite includes a **cross-implementation** mode: run N implementations over
the corpus and diff their outputs (tensor bytes, tokenizations, template
renderings, plans). Any disagreement is a specification bug until proven
otherwise — which is how ambiguity gets found before it becomes entrenched.

## 15.4 Writer conformance

Writers are tested by round-tripping through the readers and by determinism:

- W1 `omni pack` twice from identical inputs produces byte-identical output.
- W2 Every emitted file passes V0–V5.
- W3 Declared features are complete (checked by an oracle reader that refuses any
  feature not declared).
- W4 Metadata is never fabricated: the suite includes sources with missing
  fields and asserts those fields are *absent*, not defaulted.
- W5 Loss reports on export match a reference computation (§import-export).

## 15.5 Interoperability marks

To prevent "OMNI-compatible" from becoming meaningless, the mark is defined:

| Mark | Requires |
|---|---|
| **OMNI Reader** | C0 conformance, suite passing, public test report |
| **OMNI Reader+** | C0+C1+C2 |
| **OMNI Writer** | C3 conformance including determinism |
| **OMNI Full** | C0–C4 + CX for at least one extended capability |
| **OMNI Archive** | can read and write OMNI-A with no external dependencies |

Claims must cite a suite version and a published report. The working group
maintains a list; the corpus is the arbiter, not the list.

## 15.6 What is deliberately not validated

- **Model quality.** OMNI verifies bytes and structure, never behaviour.
- **Metadata truthfulness.** A signed claim that a model scores 90 on MMLU is
  verifiable as *a claim by a specific signer*, not as a fact.
- **License compliance.** Structured and signed, not enforced.
- **Numerical equivalence between representations.** `omni verify --numeric`
  compares materializations against declared tolerances *when the model declares
  them*; it does not judge whether the tolerance is appropriate.

**Prev:** [§14 Versioning](14-versioning.md) · **Index:** [README](../../README.md)
