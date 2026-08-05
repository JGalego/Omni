# Coverage-guided fuzzing

```console
$ cargo install cargo-fuzz
$ cargo +nightly fuzz run container_open
$ cargo +nightly fuzz run cbor_decode
$ cargo +nightly fuzz run recover
```

Three targets, chosen for where untrusted bytes actually arrive:

| Target | Surface |
|---|---|
| `container_open` | header, trailer, superblock, index, segment walk, object verification |
| `cbor_decode` | canonical decoding, and the round-trip property that content addressing depends on |
| `recover` | segment scanning on damaged input, which trusts less and so reaches further |

Seed the corpora from the conformance suite, which is already a collection of
containers designed to be structurally interesting:

```console
$ cargo +nightly fuzz run container_open ../../conformance/valid ../../conformance/invalid
```

This crate is deliberately **not** a workspace member. `libfuzzer-sys` would
otherwise appear in the lockfile of a project whose central claim is that it has
no dependencies, and `cargo build` on stable would fail on a crate that needs
nightly.

The roadmap's Gate 0 asks for 72 hours of fuzzing with no crash, hang or OOM.
That is a release activity, not a per-commit one, so CI instead runs the
dependency-free mutation fuzzer in `omni-conformance fuzz`, which is weaker but
runs on stable in seconds and reproduces failures from a seed.
