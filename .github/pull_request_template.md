<!--
Delete whatever does not apply. This template exists because three of the
repository's rules are easy to satisfy and easy to forget, and finding out in
review costs more than finding out here. CONTRIBUTING.md has the reasoning.
-->

## What this changes

<!-- What was wrong, and what the change does. Prose is fine and preferred. -->

## What it is checked against

<!--
A round trip through your own code proves nothing: an encoder and a decoder that
share a misunderstanding agree perfectly. If this implements somebody else's
format, name the independent implementation it is checked against (libzstd,
liblzma, hashlib, cryptography, numpy, bindings/python/omni.py, or a
from-the-specification second implementation in tools/).

If there is no second opinion available, say so here and say what stands in
instead. That is a conversation, not a blocker.
-->

## What it does not do

<!--
What is not implemented is stated in the same place it is claimed. If this adds
partial support for something, say which part is absent, confirm it is refused by
name at run time rather than guessed at, and confirm reference/README.md's
not-implemented list still matches reality — including moving an entry off it if
this change implements one. CI cannot catch that list drifting.
-->

## Checks

- [ ] `cargo test --all --all-features` passes
- [ ] `cargo fmt --all` and `cargo clippy --all-targets --all-features -- -D warnings` are clean
- [ ] No new dependency in `omni-core` (its empty `[dependencies]` is a load-bearing claim, not a preference)
- [ ] No `unsafe` outside `omni-ffi`
- [ ] Rust 1.87 still builds this — the `msrv` job checks both that it does and that 1.86 does not
- [ ] Any new normative rule has an ID and a test that names it

<!--
Security problems do not belong in a pull request. SECURITY.md, private advisory.
-->
