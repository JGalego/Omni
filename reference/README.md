# OMNI reference implementation

A dependency-free Rust implementation of the OMNI/1.0 container, object model
and canonical encoding — enough to write, read, verify and inspect real `.omni`
files.

```console
$ cargo build --release
$ cargo test
$ ./target/release/omni example model.omni      # BLAKE3-256, the default
$ ./target/release/omni example --hash sha256 model-sha.omni
$ ./target/release/omni inspect model.omni
$ ./target/release/omni verify  model.omni
```

## What is here

| Crate | Contents | Spec |
|---|---|---|
| `omni-core` | container framing, object index, canonical CBOR, BLAKE3, SHA-256, CRC-32C, Bao trees, object stores, dtype algebra, model builder | §01–§04, §13 |
| `omni-cli` | `omni inspect · verify · ls · dump · cat · example` | design/cli.md |

## Deliberate constraints

- **Zero dependencies.** `docs/design/sdk.md` §5 claims a conforming C0 reader
  needs nothing beyond a hash function and fits in ~3 000 lines. This crate is
  the evidence rather than the assertion — BLAKE3, SHA-256, CRC-32C and a strict
  canonical CBOR codec are all implemented here.
- **`#![forbid(unsafe_code)]`.** This code parses untrusted binary input; §12.4
  requires memory safety, bounds checks on every length and offset, bounded
  nesting depth, and no allocation driven by an unvalidated declared size.
- **Both mandatory hashes, from scratch.** §03.5.1 requires BLAKE3-256 and
  SHA-256. Both are implemented here, BLAKE3 including the tree internals
  (chunk and parent chaining values) that Bao verified streaming (§13.3) is
  built on. The BLAKE3 code is single-threaded and SIMD-free — auditability
  over throughput; production implementations should use the upstream crate.
- **Reproducible packing.** `pack()` is deterministic: same inputs, same bytes,
  regardless of input ordering (§01.10, writer rule W1). Enforced by a test.

## Conformance status

Claims `OMNI/1.0 C0 C3` for the subset it implements — and *only* that. What is
implemented:

- §02 container: header, segments, index, trailer, alignment, padding, CRCs
- §02.7 two-read open (trailer → superblock → index)
- §02.6 fixed-layout object index with binary search
- §03.2 canonical CBOR (rules D1–D8) with strict rejection of non-canonical input
- §03.5 digests under both mandatory algorithms, content addressing, deduplication
- §13.3 Bao outboard trees: pruned encoding, range verification, proof sizing
- §01.8 stores: memory, `.omnid/` directory, container, layered resolution
- §01 object model, refs, reachability, dangling-ref detection
- §04.3 dtype descriptors and packed sizing
- §15.1 validation levels V0–V4

What is **not** implemented, and is reported as such rather than faked:

- §04.7 tensor expression evaluation (only bare `literal` values are read)
- §05 quantization · §07 OMNI-IR · §08 adapters and deltas · §09 training state
- §10 capability negotiation · §11 WASM plugins · §12.5 signatures
- §03.7 compression codecs (only `raw`) · §13 HTTP/OCI transport
- `mmap` (the reader takes a `Vec<u8>`; the parsing code is identical either way)

See [`docs/design/roadmap.md`](../docs/design/roadmap.md) for the plan.

## Tests

49 tests covering: SHA-256 against FIPS 180-4 vectors; BLAKE3 against the
official test vectors (all three keying modes, 131 bytes of XOF output each)
plus tree-reconstruction and domain-separation properties; CRC-32C against
standard check values; CBOR against RFC 8949 Appendix A vectors; canonical-form
rejection (each of D1–D8); depth and length-overflow bounds; pack/open/verify
round-trip; reproducibility including input-order independence; data-object
page alignment; tamper detection; truncation detection; header CRC checking;
rejection of an unknown hash algorithm; the
dangling-ref-is-incomplete-not-invalid rule; and, for Bao, that the outboard
root equals the object digest at every granularity, that each group verifies
alone, that corruption stays localised, and that a tampered tree, a
misdelivered range and an unverifiable request are all refused; and, for
stores, a container→directory→container round trip that is byte-exact,
type recovery from refs alone, detection of a file whose name lies about its
contents, and refusal to mix digest algorithms. Every container-level test runs
under both mandatory digest algorithms.

```console
$ cargo test
test result: ok. 49 passed; 0 failed
$ cargo clippy --all-targets -- -D warnings
    Finished (no warnings)
```

CI lints with whatever clippy ships in the current stable toolchain, which may
be newer than yours and may therefore know lints you do not have locally. If CI
flags something `cargo clippy` accepted on your machine, `rustup update` first.
Clippy runs on stable only; beta is an early-warning job and does not gate the
branch.
