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
| `omni-core` | container framing, object index, canonical CBOR, BLAKE3, SHA-256, CRC-32C, Bao trees, object stores, dtype algebra, layouts, the tensor expression algebra, sparsity and quantization schemes, model builder | §01–§05, §13 |
| `omni-cli` | `omni inspect · verify · ls · dump · cat · pack · unpack · fsck · example` | design/cli.md |
| `omni-conformance` | corpus generator, cross-implementation runner, mutation fuzzer | §15.3 |
| `fuzz` | coverage-guided fuzz targets (nightly; outside the workspace) | §12.4 |

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
- §02.8 recovery by segment scan (`omni fsck --rebuild`)
- §15.3 conformance corpus v0 and runner protocol
- §01 object model, refs, reachability, dangling-ref detection
- §04.3 the numeric type algebra: every dtype kind, bit-exact element decode and
  encode, all five rounding modes, the alias registry
- §04.4 layouts: strided, tiled, packed, blocked-scaled, interleaved — including
  the bit position of any element and the R-T03 sufficiency check
- §04.7 the tensor expression algebra: the closed core node set, static shape and
  dtype inference (R-T01), normalization and expression identity (§04.7.5),
  evaluation, declared determinism (§04.7.6), plugin fallbacks, and range
  pushdown so partial loading is automatic (§04.7.4)
- §04.6 sparsity: all eight schemes — coo, csr, csc, bsr, n:m, bitmask, ragged,
  blocklist — each validating its own structure rather than reading it
  optimistically
- §05 quantization: the closed formula set, per-block and per-tensor schemes,
  codebooks with reproducible construction recipes, double quantization, and the
  R-T04 consistency check; the catalogue of §05.2 is covered by tests built only
  from core nodes
- §15.1 validation levels V0–V4

What is **not** implemented, and is reported as such rather than faked:

- §07 OMNI-IR · §08 adapters and deltas · §09 training state
- §10 capability negotiation · §11 WASM plugins · §12.5 signatures
- §03.7 compression codecs (only `raw`) · §13 HTTP/OCI transport
- `mmap` (the reader takes a `Vec<u8>`; the parsing code is identical either way)

See [`docs/design/roadmap.md`](../docs/design/roadmap.md) for the plan.

## Tests

125 tests covering: SHA-256 against FIPS 180-4 vectors; BLAKE3 against the
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
contents, and refusal to mix digest algorithms; and, for recovery, that a
container stripped of its index, superblock and trailer rebuilds byte-identically
and that a corrupted data object is reported missing rather than accepted; and,
for the tensor layer, f32/f64 encoding against the host's own IEEE
implementation, the documented maxima of every OCP microscaling type, all four
directed rounding modes on a tie, element placement under each layout kind, a
round-trip case for every core expression node, that equivalent expression
trees normalize to one identity, that a range request through a structural
chain reads only the bytes it needs, and that ChaCha20 matches RFC 8439; and,
for quantization, that GPTQ's permutation applied inline agrees with the
equivalent `gather`, that GGUF's `Q8_0`/`Q4_0`/`Q4_1` blocks dequantize
correctly, that MX microscaling is exact, that the NF4 codebook is reproduced
from its recipe to within 1e-6 of the published quantiles, and that a symmetric
scheme carrying a zero point is refused rather than guessed at; and, for
sparsity, that each scheme densifies correctly and that a malformed one — an
index out of range, a non-monotone `indptr`, a 3-in-4 group in a 2:4 tensor, a
values array that disagrees with its mask — is refused. Every
container-level test runs under both mandatory digest algorithms.

```console
$ cargo test
test result: ok. 125 passed; 0 failed
$ cargo clippy --all-targets -- -D warnings
    Finished (no warnings)
```

CI lints with whatever clippy ships in the current stable toolchain, which may
be newer than yours and may therefore know lints you do not have locally. If CI
flags something `cargo clippy` accepted on your machine, `rustup update` first.
Clippy runs on stable only; beta is an early-warning job and does not gate the
branch.
