# OMNI/1.0 — §14 Versioning, Migration, and Longevity

The design target is: **a file written today opens in 2076, and a file written in
2076 does not break today's tooling more than necessary.**

## 14.1 Independent version axes

Conflating these is the mistake that kills formats.

| Axis | Where | Changes | Breaks |
|---|---|---|---|
| **Container version** | `FileHeader.container_major/minor` | ~never | major: everything |
| **Schema versions** | `v` in each structure object | per object type, independently | only that object type |
| **Feature flags** | `Manifest.features` | continuously | only readers lacking the feature |
| **Registry entries** | dtype/dialect/codec/op versions | continuously | only consumers of that entry |

The container version is the only global one, and the specification commits to
never incrementing `container_major` unless the 128-byte header itself must
change meaning — an event we do not anticipate and have designed to avoid
(`header_size` allows the header to *grow*; unknown segment kinds are skippable;
the index format has its own `fmt_version`).

## 14.2 Container version semantics

- **`container_minor` bump**: purely additive. New segment kinds, new header
  flags in reserved bits, new index format versions. A 1.0 reader MUST open a
  1.7 file, skipping what it does not know, and MUST report the file as
  `newer-minor` so users understand why some features are unavailable.
- **`container_major` bump**: a different format sharing the magic bytes. A
  reader that does not support the major version MUST refuse cleanly with a
  precise message and MUST NOT attempt heuristic parsing.
- `header_size` allows a future header to be longer; readers skip to
  `header_size` rather than assuming 128, so a 1.x reader tolerates a 2.x header
  well enough to report it accurately.

## 14.3 Feature flags: the real compatibility mechanism

```cbor-diag
"features": {
  "required": ["omni.core/1.0",
               "omni.tensor/expr.1",
               "omni.quant/nested.1",
               "org.hyperion/nn.1"],
  "optional": ["omni.rt/cuda-cache.1", "omni.stream/bao.1"]
}
```

Rules:

1. A reader encountering an unknown **required** feature MUST NOT execute the
   model. It MAY still inspect, verify, sign, copy, repack and garbage-collect
   it (§11.3).
2. Unknown **optional** features are ignored silently.
3. A writer MUST list every feature the file actually uses. Under-declaration is
   a conformance violation — it converts a clean refusal into a silent
   misinterpretation, which is the worst possible failure mode.
4. Features are URIs with a version suffix; there is no ordering or implication
   relation between them beyond what a registry entry explicitly states
   (`implies: [...]`).

**Why flags beat version numbers.** A version number says "you need at least
X"; a flag says "you need exactly this capability". A reader supporting 90 % of
OMNI/1.4 can load a 1.4 file that uses only features it has — which is the
common case, and which a monotonic version number would forbid.

## 14.4 Forward compatibility for readers

A reader from year *Y* facing a file from year *Y+n* must degrade predictably:

| Unknown thing | Behaviour |
|---|---|
| Segment kind | skip via `payload_len` |
| `otype` | treat per criticality bits; preserve on copy if `SAFE_TO_COPY` |
| Schema `v` higher than known | criticality rules; report precisely |
| Extension key in `ext` | read `crit`/`copy`, act, never parse `data` |
| dtype alias | expand the inline descriptor; if only an alias is given, fail *that tensor*, not the file |
| expression `plugin` node | use `fallback` if present; else fail that tensor |
| IR op | use shipped `lower_to`; else fail that graph level; try a lower level |
| codec | fail those objects; the rest of the file is still readable |
| signature algorithm | report "unverifiable by this reader", never "invalid" |

The distinction between **"I cannot verify this"** and **"this is invalid"** is
maintained everywhere. Conflating them is how old tooling starts rejecting valid
new files, which is how ecosystems fragment.

## 14.5 Backward compatibility for writers

- **Reading old versions is never deprecated.** A conforming 2050 reader still
  reads OMNI/1.0 files. This is an unconditional commitment; the cost is bounded
  because the container layer is frozen and old schemas are small.
- **Writing** old versions may be deprecated. Deprecation proceeds:
  `active → soft-deprecated (writers warn) → hard-deprecated (writers refuse
  without --force) → withdrawn from writers`. Minimum 3 years per stage, and
  **never withdrawn from readers**.
- Registry entries carry `status: active|deprecated|withdrawn` and
  `superseded_by`, so tooling can advise migration automatically.

## 14.6 Migration

`omni migrate` performs explicit, recorded transformations:

```
$ omni migrate model.omni --to omni/1.4 --hash blake3-256 -o new.omni
migration plan:
  ✓ container 1.0 → 1.4          (additive; no data change)
  ✓ omni.tensor/desc v1 → v2     (adds `axes`; inferred from role names)
  ⚠ omni.quant/affine v1 → v2    (v2 requires explicit `formula`; inferring
                                  "affine-sub" from v1 semantics — SAFE)
  ✓ digest sha256 → blake3-256   (re-index; payload bytes unchanged)
  ✗ org.legacy/thing v1          (no migration available; preserved verbatim,
                                  marked non-critical)
new canonical digest: b3:9c1f…
provenance recorded: omni.prov/migration linking old → new
```

Key properties:

- Migration **changes the digest**, therefore it produces a *new model*, linked
  to the old one by `Provenance`. There is no in-place mutation, ever.
- Migrations that cannot be proven semantics-preserving are refused unless
  `--allow-lossy`, and are recorded as approximate.
- Hash migration (§12.11) rewrites only the index and refs; payload bytes are
  untouched, so it is I/O-cheap and can be done incrementally across a corpus.

## 14.7 File extension

`.omni` is recommended and retained. It is unclaimed, pronounceable,
descriptive, and short. Alternatives considered:

| Candidate | Verdict |
|---|---|
| `.omni` | **chosen** |
| `.om` | too short; collides with Objective-C metadata and OpenMandriva packages |
| `.model` | generic; already used by SentencePiece and CoreML internals |
| `.ai` | catastrophic collision with Adobe Illustrator |
| `.mdl` | heavily used (Valve, Matlab, others) |
| `.omnipack` | used for pack files specifically, not containers |

Variants: `.omni` (container), `.omni.idx` (index sidecar), `.omnipack` (pack),
`.omnid/` (directory store), `.omni.sig` (detached signature).

Media types are registered as `application/vnd.omni.*` (§README).

## 14.8 The archival profile (OMNI-A)

Profile `3` (§02.2) is the "put it in a vault" profile. Constraints:

1. **Codecs**: `raw` and `deflate` (RFC 1951) only. No zstd, no dictionaries.
   Deflate because it is the most-reimplemented compression algorithm in
   history and is described completely in a 15-page RFC.
2. **Hash**: SHA-256 primary (the most durable choice by ubiquity), BLAKE3
   `AltDigest` optional.
3. **No external references.** Every parent, plugin, schema, codebook and
   tokenizer is embedded. `omni pack --archive` inlines the full parent chain.
4. **No runtime caches, no executable payloads, no encryption.**
5. **All text UTF-8, NFC.** No locale dependence.
6. **All schemas embedded** as `Schema` objects.
7. **A `Rosetta` object is required** (§14.8.3).
8. **Graphs at `semantic` level MUST ship `lower_to` rules to `primitive`**, so
   the model is executable from first principles.
9. **A WASM reference implementation of every plugin op is required** — with
   the WASM spec itself embedded in the Rosetta object.

### 14.8.3 The Rosetta object

```cbor-diag
{ "t":"omni.arch/rosetta", "v":1,
  "sections": [
    {"title":"How to read this file with no software",
     "media":"text/plain; charset=utf-8", "body":[0,h'…']},
    {"title":"OMNI/1.0 specification (full text)",
     "media":"text/markdown; charset=utf-8", "body":[0,h'…']},
    {"title":"Schema definitions (OSD) + human-readable rendering", …},
    {"title":"CBOR (RFC 8949) — full text", …},
    {"title":"SHA-256 (FIPS 180-4) reference and test vectors", …},
    {"title":"DEFLATE (RFC 1951) — full text", …},
    {"title":"WebAssembly Core 2.0 specification", …},
    {"title":"IEEE 754 numeric formats used, with bit-level examples", …},
    {"title":"Reference decoder source (C99, single file, ~2000 lines)", …}
  ],
  "uncompressed": true }
```

The first section is deliberately plain prose, stored uncompressed, beginning at
a known location, describing the 128-byte header field by field in words and
hexadecimal. A reader with a hex editor and patience can bootstrap from it.

Total Rosetta size: ~2–4 MB uncompressed. Against a 140 GB model, that is
0.003 %. Against a 50-year horizon, it is the difference between a readable
artifact and a mystery.

**Precedent and honesty:** this idea is borrowed from the Rosetta Disk and from
the practice of embedding format documentation in long-term scientific archives
(FITS headers, PDF/A). It does not guarantee readability; it removes the most
likely single point of failure, which is that the *specification* is lost while
the *data* survives.

## 14.9 Governance and the specification's own longevity

A format outlives its authors only with governance:

1. **Open working group** with a published charter, public archives, and a
   documented decision process (rough consensus + running code).
2. **Independent implementations** — at least two — required before any `omni.*` registration
   is finalized.
3. **Conformance suite is normative** (§15) and versioned alongside the spec; a
   claim of conformance is a claim about test results.
4. **Registry mirroring**: the registry is published as a signed OMNI container,
   mirrorable by anyone. If the working group disappears, the registry does not.
5. **Patent policy**: contributions under a royalty-free, non-assertion
   commitment. A format with patent ambiguity will not be adopted, and the
   industry has learned this repeatedly.
6. **Escrow**: the specification, the conformance corpus, and the reference
   implementation are deposited with a long-term archival institution (Software
   Heritage, an academic library) in OMNI-A form.

## 14.10 The commitments, stated as a contract

For adopters, the promises are:

1. Any OMNI/1.x file will be readable by any conforming OMNI/1.y reader (y ≥ x)
   to the extent its features are supported, and will always be at minimum
   inspectable, verifiable and copyable.
2. `omni.core`, the object model, the container framing and the digest rules are
   **frozen** for 1.x.
3. Reading support is never removed.
4. No feature will be added that requires code execution at load.
5. No registry entry may be denied for a non-`omni.*` namespace.
6. Every derived artifact is droppable, and the canonical model never depends on
   one.

**Prev:** [§13 Streaming](13-streaming.md) · **Next:** [§15 Conformance](15-conformance.md)
