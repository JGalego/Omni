# Security policy

OMNI is a container format for machine learning models, and its reference
implementation is a parser for files that arrive from other people. That is the
whole reason this file exists: [§12.4](docs/spec/12-security.md) calls the parser
"the largest attack surface" and makes twelve normative requirements of it, and a
project that writes that down owes you somewhere to send the case where it does
not hold.

## Reporting a vulnerability

**Use GitHub's private vulnerability reporting:**
<https://github.com/jgalego/Omni/security/advisories/new>

That is the right channel rather than a public issue, and it is the only one — a
`security@` address would be a promise that a mailbox is watched, and this
project cannot make that promise honestly. A private advisory reaches the
maintainers, keeps the report unindexed while it is being fixed, and becomes the
published advisory afterwards without anything being retyped.

If the report is about the *specification* rather than the implementation, say so
in the first line. It is a different and more serious class of problem — see
below — and it changes who needs to be in the thread.

Please include, if you can: the input that triggers it, the version or commit,
which of §12.1's adversaries (A1–A7) has to be assumed, and what an attacker
gets. A container that reproduces it is worth more than a description of one; if
it is large, a program that generates it is worth more still.

**What to expect.** An acknowledgement, a decision on whether it is in scope, and
a fix or a documented reason there will not be one. What you will not get is a
service-level agreement — this is a draft specification with a reference
implementation, not a supported product, and a stated response time nobody is
rostered to meet is worse than an honest silence about it. If a report goes
unacknowledged for two weeks, opening a public issue that says only "sent a
private advisory on <date>, no reply" is a reasonable escalation and will not be
treated as a disclosure violation.

**Disclosure.** Coordinated, with no fixed embargo clock. If you intend to publish
on a schedule, say so when you report and it will be worked to rather than argued
with. Credit is given by default under whatever name you ask for, and withheld if
you would rather it were.

## Which versions

`main`, and nothing else. There has been no release, the format is
`OMNI/1.0 Draft` ([§14](docs/spec/14-versioning.md)), and there are no
maintenance branches to backport to. A fix lands on `main`; if a release exists
by the time you read this and is not listed here, assume it is unsupported and
ask.

## What is in scope

The implementation, in roughly descending order of how much a report is worth:

- **Anything that violates a §12.4 row.** An out-of-bounds read, an offset
  trusted before it is checked against the file size, arithmetic on a length that
  overflows to a small value, a nesting limit that can be exceeded, a decoder
  that allocates past `logical_len`, or non-canonical CBOR accepted where §03.2
  requires rejection. Parser-differential findings — two readers here disagreeing
  about the same bytes, or this reader disagreeing with
  [`bindings/python/omni.py`](bindings/python/omni.py) — belong in this group even
  when neither side crashes, because a format whose readers disagree is a format
  where a signature covers one interpretation and a runtime executes another.
- **A panic, abort, unbounded allocation or non-terminating loop on untrusted
  input.** In scope, and not a lesser finding for being "only" a denial of
  service: this code is meant to run on files a stranger supplied, and
  `#![forbid(unsafe_code)]` converts what would be memory corruption elsewhere
  into exactly these symptoms. A panic on hostile input is the shape a memory-
  safety bug takes in this codebase, so it is treated as one.
- **Anything in `reference/omni-ffi`**, which is the one crate that uses `unsafe`
  because a C ABI cannot be written without it, and therefore the one crate where
  a bug can be memory-unsafe rather than merely fatal. Its own rules — no panic
  crossing the boundary, every handle checked, no pointer outliving its owner —
  are part of the surface.
- **Verification that returns the wrong answer.** A signature that verifies when
  it should not (§12.5), a digest mismatch reported as valid, a validation level
  claiming to have checked something it skipped, a Bao range proof accepted for
  bytes it does not cover (§13.3). `verify` answering "valid" wrongly is worse
  than `verify` crashing, because the crash is visible.
- **Loading that executes something.** §12.2 is "rule zero: loading never
  executes". Any path from opening a container to code running — the restricted
  unpickler in §12.10, a WASM plugin escaping its budget or its host functions, a
  dialect's shipped `ref_impl` reaching outside its buffers — is a rule-zero
  violation and the highest-severity implementation report there is.
- **SSRF and locator abuse** (§12.9): a container that makes the reader fetch
  something the user did not ask for, or reach a host they did not name.
- **Credential handling in the registry client.** A token or password logged, sent
  to a host other than the one it was scoped to, or sent over a plaintext
  connection.

## Specification vulnerabilities

A rule in `docs/spec/` that makes a *conforming* implementation insecure is a
worse problem than a bug in this implementation, because it cannot be fixed by
patching anything — every conforming reader has the bug, including ones not
written yet, and the fix has to go through [§14](docs/spec/14-versioning.md)'s
versioning machinery rather than a commit. Report these the same way and label
them clearly. Examples of what this looks like: a place where the normative text
permits an ambiguity two readers can resolve differently, a signature scope that
leaves something load-bearing unsigned, a criticality bit that lets a reader skip
something it must not skip, or a limit in §12.4 that is not actually sufficient.

## What is out of scope

Not because these do not matter, but because the answer is already written down
and a report will be closed with a pointer to it:

- **Semantically backdoored weights** — a model that behaves normally except on a
  trigger. §12.1 states this as an explicit non-goal and §12.12.1 repeats it: no
  format can detect this. What OMNI offers is that the weights you run are the
  ones somebody identifiable signed.
- **The other five residual risks in [§12.12](docs/spec/12-security.md)**, each of
  which is a known, stated, accepted limitation with its mitigation named:
  equality leakage under convergent encryption, the index being unauthenticated at
  L0, `mmap` plus adversarial truncation, dedup timing observable by a co-tenant,
  and WASM plugins consuming their whole budget. A report that one of these is
  *worse than §12.12 says* is in scope; a report that one of them exists is not.
- **Missing features reported as vulnerabilities.** `https://` is unimplemented
  and refused by name rather than downgraded to `http://`; the two network and
  filesystem confinements of §12.10 clause 2 are absent and
  [`reference/README.md`](reference/README.md) says so. Both are gaps, both are
  documented where they are claimed, and neither is a vulnerability. That a
  *documented* gap is reachable is not a finding; that it is reachable **in a way
  the documentation denies** very much is.
- **Findings against a fork's modifications**, or against a build with
  `--no-sandbox` where the flag's own help text says what it turns off.
- **Anything requiring an attacker who already has your keys or your process**,
  beyond what §12.1's adversary table grants.

## What is already done, so you know where to look

The point of listing this is not reassurance — it is that the interesting bugs
are the ones these do not catch, and knowing which they are saves you time:

- `#![forbid(unsafe_code)]` in every crate that parses, `omni-ffi` excepted.
- Three coverage-guided fuzz targets over the paths that see hostile bytes
  first — `container_open`, `cbor_decode`, `recover` — in
  [`reference/fuzz`](reference/fuzz), plus a mutation fuzzer in CI on every push.
- A conformance corpus of deliberately invalid containers
  ([`conformance/`](conformance)), each with the rule it violates named, run on
  every push.
- Differential testing against independent implementations wherever one exists:
  `zstd` against libzstd, `xz` against liblzma, digests against Python's
  `hashlib`, ES256 against `cryptography`, `lz4` and `ans-lut` against
  from-the-specification second implementations, and the whole read path against
  [`bindings/python/omni.py`](bindings/python/omni.py), which shares no code with
  the Rust.
- The §12.10 import path: a restricted unpickler with no call mechanism, run by
  default in a child process with an address-space cap and a wall clock.

None of that is a proof. It is a list of the places a bug has already been looked
for, which is a different thing.
