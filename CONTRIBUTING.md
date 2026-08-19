# Contributing

The rules this repository actually runs on are not obvious from reading it, and
the expensive way to learn them is to write a feature and then find out. This
file is the short version. The nine non-negotiable practices are in
[`docs/design/roadmap.md`](docs/design/roadmap.md#engineering-practices-non-negotiable);
what follows is how to satisfy them.

## The loop

```console
$ cd reference
$ cargo test --all --all-features          # 658 tests, about 40 seconds
$ cargo fmt --all
$ cargo clippy --all-targets --all-features -- -D warnings
```

Run all three before you commit. `-D warnings` is what CI uses, and clippy on a
newer stable finds things that passed yesterday — which is a good reason to run it
and a bad reason to be surprised by it.

Rust **1.87** is the floor and CI checks it from both sides: 1.87 must build and
pass, and 1.86 must *fail*. If you use a newer API, the `msrv` job will tell you,
and raising `rust-version` in `reference/Cargo.toml` is a deliberate change with a
reason in the commit message rather than a fix for a red build. Watch for this
where you least expect it: `is_multiple_of` is in this tree because clippy's
`manual_is_multiple_of` lint asked for it, one call site at a time, and it moved
the floor five releases without any commit meaning to.

CI is reproducible locally; there is very little that only fails on a runner. If
a step is unclear, read it — `.github/workflows/ci.yml` is long on purpose,
because every step says what it is checking and why.

## Zero dependencies

`reference/omni-core` has an empty `[dependencies]` table and that is a load-
bearing claim, not an aesthetic one: `docs/design/sdk.md` §5 says a conforming C0
reader needs nothing beyond a hash function, and this crate is the evidence. So
BLAKE3, SHA-256, SHA-512, CRC-32C, Ed25519, P-256, ChaCha20, deflate, Zstandard,
LZ4, LZMA2/xz, rANS, XXH64 and a canonical CBOR codec are all written out here.

A pull request that adds a dependency to `omni-core` will be asked to remove it.
That is not a rejection of the work — it is where the work goes instead.

## `#![forbid(unsafe_code)]`

Every crate that parses has it. The exception is `omni-ffi`, where a C ABI cannot
be written without `unsafe`, and it is confined there to three operations
(dereference a handle, read a C string, hand out an `Arc`-backed pointer). If you
find yourself wanting `unsafe` anywhere else, that is the signal to ask on an
issue first.

## Differential testing, not round-trip testing

This is the rule most often missed, and the one that catches the most.

**A round trip through your own code proves nothing about correctness.** An
encoder and a decoder that share a misunderstanding agree perfectly. The xz range
coder in this tree shifted `low` in 64 bits where the format wants 32; every
stream it produced round-tripped through itself and none of them was LZMA.

So anything that implements somebody else's format is checked against somebody
else's implementation:

| What | Checked against |
|---|---|
| `zstd` | libzstd, via the `zstd` CLI |
| `xz` | liblzma, via the `xz` CLI |
| digests | Python's `hashlib` |
| ES256 | Python's `cryptography` |
| safetensors, GGUF, NumPy, GPTQ, AWQ | the arithmetic done in Python |
| the whole read path | `bindings/python/omni.py`, which shares no code |

Where no other implementation exists — `lz4`'s specific match strategy, and
`ans-lut`, which is OMNI's own — a **second implementation written from the
specification text and nothing else** stands in, in `tools/`. That answers a
different and better question than "does my code agree with my code": it asks
whether the specification is enough to implement from. When the two disagree, the
first suspect is the section, not the code.

If you add a codec, an importer or a signature algorithm, the pull request needs
one of these. If you genuinely cannot find a second opinion, say so in the pull
request and propose what to do instead — that is a conversation, not a blocker.

## Say what you did not implement, where you claimed it

The rule is: **what is not implemented is stated in the same place it is
claimed.** `reference/README.md` has an implemented list and a not-implemented
list, and every entry in the second one says *why*, and every one of them is
reported as unsupported at run time rather than guessed at.

Partial support that answers wrongly is worse than no support. `brotli` is
unimplemented because RFC 7932 decoding needs the 122 KiB static dictionary and a
decoder without it produces a plausible wrong answer on any stream referencing
it, instead of refusing. `xz`'s BCJ filters are refused by the filter's own name
rather than skipped, because a filter changes the bytes.

So: implement it, or refuse it by name with a reason. Do not guess. And if your
change moves something from the second list to the first, move it — the lists
drifting behind the code is its own bug, and CI cannot catch that one.

## Specification changes

`docs/spec/` is normative and CC BY 4.0; the implementation is Apache-2.0 OR MIT.
[`LICENSE`](LICENSE) says which files are which.

A change to the normative text is a bigger change than a change to the code,
because every conforming implementation is affected including ones not yet
written. Rules get IDs (`R-C20`, `R-T02`, …) and every rule needs a test that
names its ID. If a rule is ambiguous enough that two readers could resolve it
differently, that is a defect in the section even when both readers work.

Adding a codec identifier, an op or a dialect means adding it to the registry in
the relevant section *and* implementing it — `ans-lut` had to be specified before
it could be implemented twice, because until §03.7.5 existed the registry named
an identifier no two implementations could have agreed on.

## Commits

Messages are prose, not bullet lists. Say what was wrong, what the change does,
and what you found out while doing it — particularly the thing that nearly went
wrong. Reference sections and rules by number (§03.7.5, R-T02) so the commit and
the specification can be read against each other.

One feature per commit. If a fix and a feature are in the same diff, they are two
commits.

The commits worth imitating are the ones that record a wrong turn. "The shift is
32-bit on purpose" is a comment somebody needed two hours to be able to write, and
the commit that added it says why.

## Reporting a security problem

Not here. [`SECURITY.md`](SECURITY.md) — private advisory, not a public issue.
The scope section there is worth reading first: a panic on hostile input is in
scope at full weight, and several plausible-looking reports are already answered
by §12.12.
