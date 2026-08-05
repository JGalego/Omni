# OMNI/1.0 — §12 Security Model

Model files are downloaded from the internet, from mirrors, from community hubs,
and are loaded by processes with GPUs, credentials and network access. The
current state of the art — `pickle` in PyTorch checkpoints, executable engine
blobs, Jinja templates evaluated at load, no signatures — is indefensible.

---

## 12.1 Threat model

**Assets:** the loading process's memory and control flow; the host's
credentials, network position, and filesystem; the integrity of the model's
weights and behaviour; the confidentiality of proprietary weights.

**Adversaries and their capabilities:**

| # | Adversary | Can |
|---|---|---|
| A1 | Malicious publisher | craft arbitrary container bytes |
| A2 | Compromised mirror / CDN | substitute or modify bytes in transit or at rest |
| A3 | Network attacker | reorder, truncate, replay, downgrade |
| A4 | Malicious *contributor* to a legitimate model | insert an object into a legitimate build |
| A5 | Compromised build system | produce a signed but malicious artifact |
| A6 | Local co-tenant | read cached weights, observe dedup timing |
| A7 | Future cryptanalyst | break today's hash or signature scheme |

**Explicit non-goals:** OMNI cannot detect a *semantically* backdoored model
(poisoned weights that behave normally except on a trigger). No format can. What
OMNI provides is that the weights you run are exactly the weights someone
identifiable signed, and that loading them cannot execute code.

## 12.2 Rule zero: loading never executes

> **A conforming reader MUST NOT execute, interpret, JIT, deserialize-into-code,
> or otherwise transfer control to any data originating from a container as a
> consequence of opening, inspecting, verifying, or loading tensors from it.**

Consequences, each of which is a deliberate departure from an existing format:

| Banned | Instead |
|---|---|
| Python pickle / `torch.load` semantics | tensors are data; import from pickle uses a restricted unpickler *outside* the trust boundary (§12.10) |
| Jinja2 chat templates evaluated at load | OMNI-CT, total and pure (§06.9) |
| Loading TensorRT/CoreML/compiled engines by default | `executable: true` caches, refused unless explicitly trusted (§12.3) |
| Custom-op shared libraries loaded by name from the file | plugins are WASM-sandboxed (§11.6); native plugins require out-of-band installation and trust |
| Regex from the file compiled with a backtracking engine | tokenizer regexes are validated for linear-time execution or run under a step budget |
| URLs in the file dereferenced automatically | locators are hints; dereferencing requires policy (§12.9) |

## 12.3 Executable payloads

Some genuinely useful artifacts *are* code: TensorRT engines, `.mlmodelc`,
autotuned kernel libraries, native plugins. OMNI does not ban them; it isolates
them.

1. They may exist **only** as `RuntimeCache` objects with `executable: true`.
2. They MUST be marked `CACHEABLE` and MUST never be required for the model to
   be usable.
3. A runtime MUST refuse to load one unless **both**:
   (a) `policy.allow_native_caches` is true, and
   (b) the cache carries a `Signature` from a key in the deployment's trust
   store — *not* merely a signature from the model's publisher.
4. `omni inspect` prints a prominent warning listing every executable payload,
   its size, its target, and its signer.
5. `omni strip --executable` removes them all; the result is byte-verifiable as
   the same model.

The default posture is: **a model downloaded from the internet contains no code
your machine will run.**

## 12.4 Parser hardening

The parser is the largest attack surface. Normative requirements:

| Requirement | Rationale |
|---|---|
| Every length/offset validated against actual file size *before* use | integer-overflow and OOB reads |
| Arithmetic on offsets/lengths uses checked or 128-bit intermediates | overflow to a small value |
| `logical_len` is a hard allocation cap; decoders abort on overrun | decompression bombs |
| Declared expansion ratio > 1000:1 rejected unless feature-flagged | zip bombs |
| CBOR nesting depth ≤ 64; expression tree depth ≤ 256; parent chain ≤ 32 | stack exhaustion |
| Total objects, refs per object, index entries bounded (§02.10) | resource exhaustion |
| No recursion in parsers; explicit work stacks with capacity limits | stack overflow |
| Duplicate map keys and non-canonical CBOR rejected | parser-differential attacks |
| Digest verified **before** any parsed content is trusted for control flow | TOCTOU on the store |
| Shape inference verified against declared shapes (§04.7.3) | malformed tensors causing OOB in consumers |
| `mmap` reads treated as fallible (SIGBUS on truncation) — use `pread` or install a handler | truncated file after mapping |
| Zero-filled padding verified in strict mode | covert channels, reproducibility |

The reference implementation is continuously fuzzed (§roadmap) with structure-
aware fuzzers over the header, index, CBOR objects, and expression trees, and
ships a corpus of ~200 hostile files as part of the conformance suite.

**Memory safety.** The reference implementation is Rust with `#![forbid(unsafe_code)]`
in the parsing crate; `unsafe` appears only in the mapping layer, isolated behind
a reviewed API. C and C++ SDKs are thin wrappers over the Rust core rather than
independent parsers — reimplementing an untrusted-input parser in C is how every
media format got its CVEs.

## 12.5 Signatures

### 12.5.1 Scheme

Signatures are **COSE_Sign1** (RFC 9052) objects — CBOR-native, so no second
encoding appears in the trust path.

| Algorithm | Status |
|---|---|
| Ed25519 (`EdDSA`) | **MUST** |
| ECDSA P-256 (`ES256`) | SHOULD (hardware/HSM ubiquity) |
| ML-DSA-65 (FIPS 204) | SHOULD — post-quantum |
| SLH-DSA (FIPS 205) | MAY — hash-based, conservative long-term |
| RSA-PSS | MAY (legacy interop only) |

Hybrid signing (classical + PQ, both required to verify) is supported by
attaching two `Signature` objects with `policy: "all-required"`. Given A7, new
long-lived artifacts SHOULD be dual-signed.

### 12.5.2 What is signed

```
TBS = canonical_cbor({
   "t":"omni.sec/tbs", "v":1,
   "root": h'…',                 ; digest of Manifest with `attestations` removed
   "alg": "EdDSA",
   "purpose": "release",         ; release|internal|test|revocation
   "subject": {"name":"acme/llm-8b","version":"2026.08.1"},
   "not_before": "…", "not_after": null,
   "summary": {"tensors":291,"params":8030261248,
               "canonical_digest":h'…',   ; §12.5.3
               "executables":0},
   "counter": 3                  ; monotonic per-subject, replay/rollback defense
})
```

Removing `attestations` from the manifest before hashing resolves the
self-reference paradox: signatures list themselves in the object they sign,
without a second manifest.

`summary.executables` being part of the signed payload means a mirror cannot add
an executable cache to a signed model without detection, even though caches are
"droppable".

### 12.5.3 Canonical digest

`canonical_digest` = the digest of the manifest after removing *all* cacheable
objects and re-serializing canonically. It is the identity of the model *as a
model*, independent of which caches, indexes or packing a given file carries.
Two files with the same `canonical_digest` are the same model. This is the number
that belongs in a model card, a paper, or a compliance record.

### 12.5.4 Detached and multi-party signatures

- Signatures may be embedded (`attestations[]`) or detached (a `SIG` segment, or
  a separate `.omni.sig` file, or a Rekor entry).
- Multiple signatures are independent; a policy states how many and whose are
  required (`any-of`, `all-of`, `k-of-n`, `role-based`).
- Counter-signatures (an auditor signing a publisher's signature) are ordinary
  `Signature` objects whose `root` is the digest of another signature.

### 12.5.5 Identity and key distribution

Three interoperable models, because different ecosystems have settled
differently:

1. **Keys** — raw public keys pinned by the consumer (simplest, best for
   internal use).
2. **Keyless / Sigstore** — Fulcio-issued short-lived certificate bound to an
   OIDC identity, plus a Rekor transparency-log inclusion proof stored as a
   `Provenance` object. Gives "signed by `release-bot@acme.com` via GitHub
   Actions" with no key management.
3. **TUF** — for repositories serving many models: role separation, threshold
   signing, key rotation, and freshness guarantees against rollback and freeze
   attacks (A3). A model hub SHOULD run TUF over its OMNI object store; this is
   what protects consumers when the hub itself is compromised.

### 12.5.6 Revocation

Revocation is a signed statement, not an absence:

```cbor-diag
{ "t":"omni.sec/revocation", "v":1,
  "target": h'…',                 ; canonical_digest of the revoked model
  "reason": "weights-compromised", ; also: key-compromise|superseded|legal|safety
  "replacement": h'…',
  "issued":"…", "issuer":"…" }
```

Distributed through the same channels as signatures (transparency log, TUF
metadata, a well-known endpoint). A runtime with a freshness policy checks
revocations before load; an air-gapped one cannot, and OMNI says so plainly
rather than pretending otherwise.

## 12.6 Provenance

`Provenance` objects carry **in-toto attestations** with SLSA predicates:

```cbor-diag
{ "t":"omni.prov/attestation", "v":1,
  "predicate_type":"https://slsa.dev/provenance/v1",
  "subject":[{"name":"acme/llm-8b","digest":{"omni-canonical":h'…'}}],
  "predicate": {
    "buildDefinition": {
      "buildType":"https://omni.dev/build/train/v1",
      "externalParameters":{"config":h'…',"dataset":h'…',"base_model":h'…'},
      "resolvedDependencies":[{"uri":"omni://acme/llm-8b-base","digest":{…}}] },
    "runDetails":{"builder":{"id":"https://acme.com/trainers/cluster-3"},
                  "metadata":{"invocationId":"…","startedOn":"…","finishedOn":"…"}} },
  "chain": [ [19, h'…'] ]          ; -> parent Provenance: the full lineage
}
```

The **provenance chain** is the differentiating capability: because every parent
model is referenced by digest (§08.7), the lineage
`medical-delta → math-delta → code-delta → instruct → foundation` is
cryptographically verifiable end to end, including which dataset and which base
model each step used. `omni provenance --tree model.omni` prints it; `omni
provenance --verify` checks every signature and digest in the chain.

Also carried here: training compute disclosure, energy/carbon figures,
evaluation attestations (§06.8), red-team reports, and regulatory artifacts.
These are the fields regulators are converging on, and they belong in a signed,
machine-readable, tamper-evident structure rather than a PDF.

## 12.7 Verification levels

```
L0  STRUCTURAL   header, trailer, CRCs, index bounds, schema validation.
                 Cost: O(index). No hashing. Detects corruption and malformation.
L1  SELECTIVE    verify the digest of every object actually read.
                 Cost: O(bytes read). THE DEFAULT.
L2  COMPLETE     verify every object in the container + full reachability.
                 Cost: O(file). Minutes for a 400 GB model at BLAKE3 speeds.
L3  AUTHENTIC    L1/L2 + signature chain + policy (who signed, is it revoked,
                 is it fresh).
L4  PROVENANT    L3 + full lineage: every parent, every attestation, transparency
                 log inclusion, reproducibility check where claimed.
```

L1 is the default because it is the only level with the right cost/benefit shape:
you pay only for what you read, and you cannot be attacked through bytes you
never touch. Verified streaming (§13.3) is what makes L1 sound even for partial
reads — without a Merkle tree, "verify what you read" is impossible for a range.

## 12.8 Encryption (optional profile)

For proprietary weights at rest and in transit beyond TLS.

- **Granularity:** per object, applied *after* compression.
- **AEAD:** XChaCha20-Poly1305 (default) or AES-256-GCM-SIV. Associated data
  binds the object's digest and index position, preventing object substitution.
- **Key management:** envelope encryption. A `KeyEnvelope` object holds
  per-recipient wrapped keys via HPKE (RFC 9180), age, or a KMS reference. The
  container is decryptable by any authorized recipient without re-encrypting the
  payload.
- **Two modes:**

| Mode | Key derivation | Dedup | Leak |
|---|---|---|---|
| `convergent` | `key = KDF("omni/1.0 chunk-encryption-key", plaintext_digest)` | preserved | reveals plaintext *equality* to anyone with the ciphertext + a candidate plaintext (confirmation-of-file attack) |
| `random` | random per object | destroyed | none beyond size and structure |

The tradeoff is stated because it is real and often glossed over. Default is
`random` for `kind: "model"` releases; `convergent` is for internal object stores
where dedup economics dominate and the corpus is not attacker-guessable.

- **Metadata leakage:** even with encrypted payloads, the index reveals object
  count, sizes, and types, and the manifest reveals structure. A `sealed-index`
  sub-mode encrypts the index and manifest too, leaving only the header — at the
  cost of losing partial fetch and inspection.

## 12.9 Locators and SSRF

`Ref.s[]` locators and `extern` nodes contain URLs supplied by whoever wrote the
file. A reader:

- MUST NOT dereference them automatically;
- MUST require explicit policy (allowlist of schemes and hosts) before any
  fetch;
- MUST NOT follow redirects outside the allowlist;
- MUST treat fetched bytes as untrusted and verify their digest before use
  (which makes a malicious redirect harmless in the data path, though not in the
  SSRF path);
- SHOULD block link-local, loopback and metadata-service addresses by default.

## 12.10 Import from unsafe formats

Importing a PyTorch pickle checkpoint is the one place OMNI must touch dangerous
data. Rules:

1. Use a **restricted unpickler** with an opcode allowlist and a class allowlist
   limited to tensor-reconstruction primitives. `GLOBAL`/`REDUCE` to anything
   else is a hard error, not a warning.
2. Run the import **in a separate process** with no network, a read-only
   filesystem view, a memory cap, and a wall-clock cap. The reference importer
   uses a seccomp/landlock-confined child on Linux and the equivalent elsewhere.
3. Record in `Provenance` that the source was an unsafe format, what was
   rejected, and the digest of the source file.
4. Never re-emit pickle on export unless explicitly requested with a warning.

## 12.11 Cryptographic agility (adversary A7)

- Hash: the algorithm is a header field, `AltDigest` objects allow re-indexing an
  existing corpus under a new hash without rewriting payloads, and the
  conformance suite requires at least two hash implementations from day one.
- Signature: COSE algorithm identifiers; hybrid signing supported now.
- Migration path: publish `AltDigest` objects → re-sign under the new algorithm →
  consumers accept either during an overlap window → deprecate. Because payload
  bytes never change, migration costs one hashing pass over the corpus and zero
  re-uploads.
- The specification commits to *never* embedding an algorithm assumption in a
  fixed-width field beyond the 32-byte index lookup key (which is a
  non-cryptographic accelerator, §01.3).

## 12.12 Residual risks

Stated plainly:

1. **Semantic backdoors in weights** are undetectable by any format mechanism.
   OMNI's contribution is limited to making *who signed it* and *what it was
   trained on* verifiable.
2. **Convergent encryption** leaks equality (§12.8).
3. **The index is not authenticated at L0.** A corrupted index can misdirect a
   reader to wrong bytes — which then fail their digest check at L1. Deployments
   requiring L0 integrity against an active attacker must verify the superblock
   signature first, which covers the index digest.
4. **`mmap` + adversarial truncation** can produce SIGBUS; readers handling
   untrusted files on shared storage should prefer `pread` or install a handler.
5. **Timing side channels in dedup** (A6): a co-tenant can learn whether a chunk
   is already cached. Mitigation is deployment-level (per-tenant caches); OMNI
   flags it rather than solving it.
6. **WASM plugins can consume resources** up to their fuel/memory budget; the
   budget is the mitigation, and defaults are conservative.

---

**Prev:** [§11 Plugins](11-plugins.md) · **Next:** [§13 Streaming & Transport](13-streaming.md)
