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
$ ./target/release/omni verify  model.omni --level 6
$ ./target/release/omni example --quantized quant.omni   # int4 + LoRA, as expressions
$ ./target/release/omni cat  quant.omni --tensor model.layers.0.attn.q_proj.weight.lora
$ ./target/release/omni deps quant.omni --tensor model.layers.0.attn.q_proj.weight.bf16 --range 0:64
$ ./target/release/omni keygen --out key.hex
$ ./target/release/omni sign model.omni --key <seed-hex> -o signed.omni
$ ./target/release/omni sign --verify signed.omni --key <public-hex>
$ ./target/release/omni delta base.omni tuned.omni -o delta.omni
$ ./target/release/omni adapter check base.omni lora.omni
$ ./target/release/omni example --tokenizer tok.omni
$ ./target/release/omni verify tok.omni --tokenizer     # runs its §06.7.1 vectors
$ ./target/release/omni tokenize tok.omni --text "hello"
$ ./target/release/omni example --graph graph.omni      # a synthesized OMNI-IR graph
$ ./target/release/omni graph graph.omni                # print the module (§07)
$ ./target/release/omni graph graph.omni --verify        # V5 IR rules, against the weights
$ ./target/release/omni graph lower graph.omni -o low.omni   # apply the shipped lowerings
$ ./target/release/omni pack model.omnid -o packed.omni --codec zstd:9
$ ./target/release/omni repack packed.omni -o smaller.omni --codec bitshuffle+zstd:9:2
$ ./target/release/omni example --chat-template ct.omni
$ ./target/release/omni verify ct.omni --template         # runs its §06.9 vectors
$ ./target/release/omni render ct.omni --message user:"Hi" --var add_generation_prompt=true
```

## What is here

| Crate | Contents | Spec |
|---|---|---|
| `omni-core` | container framing, object index, canonical CBOR, BLAKE3, SHA-256, CRC-32C, Bao trees, object stores, compression codecs (zstd, deflate, bitshuffle), dtype algebra, layouts, the tensor expression algebra, sparsity and quantization schemes, tokenizer IR, OMNI-CT, OMNI-IR, model builder | §01–§08, §10, §12, §13 |
| `omni-cli` | `omni inspect · verify · ls · dump · cat · deps · tokenize · render · graph · pack · unpack · repack · fsck · caps · plan · keygen · sign · delta · adapter · example` | design/cli.md |
| `omni-conformance` | corpus generator, cross-implementation runner, mutation fuzzer | §15.3 |
| `fuzz` | coverage-guided fuzz targets (nightly; outside the workspace) | §12.4 |

## Deliberate constraints

- **Zero dependencies.** `docs/design/sdk.md` §5 claims a conforming C0 reader
  needs nothing beyond a hash function and fits in ~3 000 lines. This crate is
  the evidence rather than the assertion — BLAKE3, SHA-256, SHA-512, CRC-32C,
  Ed25519, ChaCha20, deflate, Zstandard, XXH64 and a strict canonical CBOR codec
  are all implemented here.
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
- §04.2 reader-side `TensorTable` and `TensorDesc` views, with the V5 tensor
  rules: R-T01 (declared type equals inferred type), R-T02 (chunk sizing),
  R-T03, R-T04, R-T05, R-T06, R-T07 and R-M01
- §08 adapters and deltas: the object, selectors and attachment rules with
  R-A01–R-A03, the eight arithmetic methods built from core nodes, graph-level
  methods carried as rewrites, composition, the six delta representations with
  measured error, and parent chains with R-O06
- §03.7 compression: `raw`, `zstd` (RFC 8878, both directions — FSE, Huffman,
  sequences, repeat offsets, the window, multi-frame streams and the XXH64
  content checksum), `deflate` (RFC 1951, both directions), `bitshuffle`, and
  both `bitshuffle+zstd` and `bitshuffle+deflate`, with the §03.7.4
  decompression bounds; a compressed container holds the same object identities
  as an uncompressed one, the superblock names the codecs a reader will meet and
  reports stored bytes apart from logical ones, and `omni repack` changes the
  storage codec while proving every digest survived
- §07 OMNI-IR: the `GraphModule`/`Function`/`Region`/`Block`/`Op` structure in
  SSA form, the type system with symbolic dimensions and per-function
  constraints, the dialect mechanism with per-op versions and attribute defaults
  (`omni.core` frozen, plus `omni.tensor`, `omni.nn`, `omni.quant`, `omni.io`),
  `DialectRef` objects embedded in the container, shape and dtype inference for
  the ops this build knows, verification under rules R-I01–R-I11, rewrites as
  data (§07.7) driving both op-version migration and dialect lowering, the
  fixed-layout binary op array of §07.9, and `graph synthesize` (§07.5), which
  turns a weights-only transformer into a self-describing one from its
  `arch.params`. §07.2's load-bearing claim is exercised end to end: a runtime
  that knows only `omni.tensor` applies the lowering the *model* ships and
  proceeds, and `omni graph lower` names every op no shipped rule covers rather
  than pretending it lowered everything
- §10 capability negotiation: capability sets with the three-valued support of
  §10.2, candidate enumeration, the deterministic resolver of §10.5 under all
  five objectives, budget retry, and the informative failures of §10.5.2
- §12.5 signatures: COSE_Sign1 over the §12.5.2 payload, Ed25519, the
  `canonical_digest` of §12.5.3, trust policies (any-of, all-of, k-of-n,
  role-based), validity windows, rollback counters and revocation statements
- §06.7 the tokenizer IR: read structurally — the vocabulary as a string tensor,
  the merges as `u32` id pairs — with `encode`/`decode` for the bpe, wordpiece,
  unigram, wordlevel, char and byte kinds, the normalizer, pre-tokenizer and
  decoder pipelines, the template postprocessor, and the §06.7.1 conformance
  vectors run by `omni verify --tokenizer`. A step this build cannot honour —
  a plugin kind (§06.7.2), NFC composition, or a `regex-split` pattern needing
  Unicode property classes — makes encoding *indeterminate* rather than
  silently producing different token ids
- §06.9 OMNI-CT: a *total* template language, so a chat template renders with
  no sandbox because there is nothing to sandbox. No `while`, no recursion, no
  macro, no include, no import, no attribute access into host objects, no method
  call on a value — `{% for %}` iterates a finite structure already in memory,
  and the closed standard library is pure. Because it is total, a template's
  required inputs are computable statically (`omni render --inputs`), the
  optional `compiled` AST is a derived object that V6 recomputes, and the §06.9
  vectors are run by `omni verify --template`
- §01.5 R-O02 in `verify`: an object whose own `t` contradicts the index's
  `otype` is invalid. Refs carry the type, so a reader decides what to do with
  an object before fetching it; one that lies defeats all of those decisions
- §15.1 validation levels V0–V6 in the CLI; the V7 rules are implemented and
  reached through `omni sign --verify`

What is **not** implemented, and is reported as such rather than faked:

- §09 training state
- §11 WASM plugins
- §03.7's MAY-level codecs `lz4`, `brotli`, `xz`, `ans-lut` and the two lossy
  ones: reported as unsupported rather than half-decoded · §13 HTTP/OCI
  transport
- `mmap` (the reader takes a `Vec<u8>`; the parsing code is identical either way)

See [`docs/design/roadmap.md`](../docs/design/roadmap.md) for the plan.

## Tests

306 tests covering: SHA-256 against FIPS 180-4 vectors; BLAKE3 against the
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
values array that disagrees with its mask — is refused; and, for selectors,
that glob captures index adapter tensors correctly and that catastrophic regex
backtracking hits a step budget instead of hanging; and, for the V5 rules, that
a declared shape, chunk total, layout, quantization scheme or statistic that
disagrees with the value is reported as invalid while an unimplemented plugin or
an absent object is reported as indeterminate; and, for adapters, that a LoRA
attaches to a base it has never seen and binds each layer's own factors, that an
unmatched selector, an unsatisfiable binding, a shape that cannot work and a
violated `require` are each reported with their rule, that an absent base is
incomplete rather than invalid, and that TIES, DARE and SLERP are reproducible
from their recipes; and, for deltas, that an unchanged tensor costs nothing,
that a genuinely low-rank change is extracted exactly and reproducibly, that a
representation which would exceed `--max-err` is not chosen, that a dense small
change becomes a quantized residual wrapped in `approx`, and that a parent chain
is bounded and reports a missing required parent as incomplete; and, for
Ed25519, the RFC 8032 §7.1 vectors plus the checks that make a verdict mean one
thing — a non-canonical `s`, a non-canonical `y`, a small-order key and a
tampered signature are all refused; and, for §12.5, that attaching a signature
to the manifest it signs does not invalidate it, that the canonical digest
ignores caches but not assets, that a signature over another model does not
transfer, and that an unknown key or an unimplemented algorithm is indeterminate
rather than invalid; and, for compression, that zstd decodes six frames
produced by libzstd — covering Raw, RLE and compressed blocks, direct and
FSE-compressed Huffman weights, one- and four-stream literals, Treeless blocks
that reuse a previous table, and multi-block frames whose matches reach across a
block boundary — that our own frames round-trip at every level over every
corpus, that XXH64 matches its published vectors, that a dictionary frame is
refused as unsupported rather than guessed at, that a corrupted frame never
decodes to the original, that deflate round-trips at every
level and is reproducible, that bitshuffle is exactly invertible at any length
and helps on float weights, that a compressed container holds the same objects
and digests as an uncompressed one, and that a lying ratio, a back-reference
before the start of a stream and a tampered compressed object are all refused; and, for planning, that the
objective decides which realization is chosen, that a refused capability is
never attempted while an unknown one may be, that a dequantizable
representation needs the scheme, and that resolution is deterministic; and, for
the tokenizer, that BPE applies merges in priority order rather than
left-to-right, that added tokens are matched before normalization, that a merge
naming a token the vocabulary does not contain is an error, that the byte-level
mapping round-trips all 256 bytes, that WordPiece takes the longest match and
WordPiece and Unigram agree with their own definitions, that a conformance
vector which disagrees is a failure rather than a warning, and that an
unimplemented pre-tokenizer step refuses to encode instead of guessing; and that
an unimplemented regex escape — `\p{L}` above all — is a parse error rather
than the literal letter, because reading `\p` as `p` matches a different
language without saying so. For OMNI-CT: that a realistic chat
template renders exactly, that its required inputs are computed without
rendering it, that a missing input is an error rather than the empty string,
that `while`, `macro`, `include`, `import` and `raw` are each a syntax error
naming the closed statement set, that a `set` inside a loop cannot feed back
into the loop's own bound, that there is no method call and no name outside the
closed library, that whitespace-control dashes are not subtraction and `%}` is
not a remainder, that a float or byte string in the input is refused rather than
coerced, that `strftime` cannot ask for the current time and agrees with known
dates on both sides of the epoch, that a runaway product of template and input
size hits the budget instead of running, that a cached AST disagreeing with its
source is reported, that a capability declared but never read is reported, and
that a container claiming Jinja2 as its template *language* is refused —
executing one is the problem §06.9 exists to fix. And, for OMNI-IR: that a module round-trips
through canonical CBOR with every type kind present, that a value defined twice,
a use with no definition, a missing entry function, an undeclared dialect, a
declared type that contradicts inference, a wrong operand count, a missing
required attribute, a region that does not terminate, an effect token used twice,
contradictory dimension constraints and a constant that disagrees with the tensor
it names are each invalid and each name their rule; that an op from an unknown
dialect is *indeterminate* rather than invalid, and not even that when the model
ships a lowering for it; that the shipped lowerings produce primitive graphs
which verify on their own terms while a causal attention — which cannot be
lowered without ops for building a mask — keeps its high-level form and is
reported; that an approximate rewrite is refused unless allowed; that the §07.9
binary encoding round-trips nested regions and ten thousand ops and refuses a
truncated one; and that synthesis fails by naming the weight it wanted rather
than by inventing one. And that an object whose `t`
contradicts the index's otype is invalid (R-O02).
Every
container-level test runs under both mandatory digest algorithms.

```console
$ cargo test
test result: ok. 306 passed; 0 failed
$ cargo clippy --all-targets -- -D warnings
    Finished (no warnings)
```

CI lints with whatever clippy ships in the current stable toolchain, which may
be newer than yours and may therefore know lints you do not have locally. If CI
flags something `cargo clippy` accepted on your machine, `rustup update` first.
Clippy runs on stable only; beta is an early-warning job and does not gate the
branch.
