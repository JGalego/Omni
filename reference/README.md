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
$ ./target/release/omni example --plugin --tune 3 plug.omni   # an op only WASM can do
$ ./target/release/omni plugin list plug.omni
$ ./target/release/omni cat plug.omni --tensor model.layers.0.attn.q_proj.weight.scaled
$ ./target/release/omni open model.omni --tensor w --range 0:64   # what a read costs
$ ./target/release/omni example --training ckpt.omni    # a checkpoint with Adam state
$ ./target/release/omni strip ckpt.omni --training -o infer.omni   # weights, unchanged
$ ./target/release/omni reshard ckpt.omni --mesh tp=2 -o resharded.omni
$ ./target/release/omni log ckpt.omni --with prev.omni  # the checkpoint chain
$ ./target/release/omni example --graph graph.omni      # a synthesized OMNI-IR graph
$ ./target/release/omni graph graph.omni                # print the module (§07)
$ ./target/release/omni graph graph.omni --verify        # V5 IR rules, against the weights
$ ./target/release/omni graph lower graph.omni -o low.omni   # apply the shipped lowerings
$ ./target/release/omni pack model.omnid -o packed.omni --codec zstd:9
$ ./target/release/omni repack packed.omni -o smaller.omni --codec bitshuffle+zstd:9:2
$ ./target/release/omni example --chat-template ct.omni
$ ./target/release/omni verify ct.omni --template         # runs its §06.9 vectors
$ ./target/release/omni render ct.omni --message user:"Hi" --var add_generation_prompt=true
$ ./target/release/omni index model.omni                # the §13.4.1 index sidecar
$ ./target/release/omni fetch http://host/model.omni --sidecar model.omni.idx --all
$ ./target/release/omni strip model.omni --weights -o catalogue.omni   # §13.8
$ ./target/release/omni import safetensors w.safetensors -o w.omni  # with a report
$ ./target/release/omni export safetensors w.omni --plan            # what it would cost
$ ./target/release/omni import peft ./lora --base w.omni -o lora.omni  # a LoRA, pinned
$ ./target/release/omni graph run m.omni --tokens 1,2,3   # execute the §07 graph
$ ./target/release/omni import gptq ./model-gptq -o m.omni    # §05.2.2, as expressions
$ ./target/release/omni import awq ./model-awq -o m.omni      # §05.2.3
$ ./target/release/omni export gptq m.omni -o ./out-gptq      # byte-exact back
$ ./target/release/omni serve model.omni --port 8080    # §13.4.3 object server
$ ./target/release/omni oci export model.omni -o layout/ # §13.5, push with oras
```

## What is here

| Crate | Contents | Spec |
|---|---|---|
| `omni-core` | container framing, object index, canonical CBOR, BLAKE3, SHA-256, CRC-32C, Bao trees, object stores, compression codecs (zstd, deflate, bitshuffle), dtype algebra, layouts, the tensor expression algebra, sparsity and quantization schemes, tokenizer IR, OMNI-CT, OMNI-IR and an interpreter for it, training state, a WebAssembly host, an HTTP range store, object server and OCI mapping with the `.omni.idx` sidecar, a JSON codec, a Jinja2 translator, safetensors, PyTorch (ZIP + a restricted unpickler), PEFT, GPTQ and AWQ import and export, whole-Hugging-Face-repo import, model builder | §01–§13 |
| `omni-cli` | `omni inspect · verify · ls · dump · cat · deps · open · index · fetch · serve · oci · import · export · tokenize · render · graph · plugin · strip · log · reshard · pack · unpack · repack · fsck · caps · plan · keygen · sign · delta · adapter · example` | design/cli.md |
| `omni-ffi` | the C ABI (`omni.h`): opaque handles, panic-proof entry points, CLI-matching status codes, DLPack export. Built as `cdylib` + `staticlib`. The only crate here that uses `unsafe` | design/sdk.md §3 |
| `omni-conformance` | corpus generator, cross-implementation runner, mutation fuzzer | §15.3 |
| `fuzz` | coverage-guided fuzz targets (nightly; outside the workspace) | §12.4 |

## Deliberate constraints

- **Zero dependencies.** `docs/design/sdk.md` §5 claims a conforming C0 reader
  needs nothing beyond a hash function and fits in ~3 000 lines. This crate is
  the evidence rather than the assertion — BLAKE3, SHA-256, SHA-512, CRC-32C,
  Ed25519, ChaCha20, deflate, Zstandard, XXH64 and a strict canonical CBOR codec
  are all implemented here.
- **`#![forbid(unsafe_code)]`** in every crate that parses. This code reads
  untrusted binary input; §12.4 requires memory safety, bounds checks on every
  length and offset, bounded nesting depth, and no allocation driven by an
  unvalidated declared size. The single exception is `omni-ffi`, where the C ABI
  makes `unsafe` unavoidable — turning a caller's `const char *` into a `&str` is
  exactly the operation the compiler cannot check. It is confined there on
  purpose: that crate does no parsing, and its `unsafe` blocks are three kinds
  only (dereference a handle, read a C string, hand out an `Arc`-backed pointer).
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
  evaluation, declared determinism (§04.7.6), plugin fallbacks, and all three of
  §04.7.4's evaluator refinements — range pushdown so partial loading is
  automatic, **fusion** so a chain of elementwise nodes is one pass into one
  buffer rather than one buffer per node, and **caching** keyed by the §04.7.5
  identity, which is what makes "a different expression is a different key" a
  design rather than an invalidation problem. Both of the last two are
  switchable and counted: `omni cat --no-fuse` and `--cache SIZE` exist so the
  claim is measured on both sides, and CI checks that the values are bit-identical
  either way
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
- §09 training state: the `TrainingState` object with its optimizer, schedule,
  EMA, gradient scaler and step counters; RNG streams with the §09.3 distinction
  that decides what can be promised (counter-based streams reproduce anywhere,
  stateful ones are stored honestly as opaque blobs and reported as
  non-portable); the `ShardMap` of §09.4 with its mesh, strategy, per-tensor
  placements and FSDP flat-parameter table, checked under R-N04–R-N06; §09.4.2
  resharding, which rewrites the map alone when the boundaries permit and names
  the tensors that would need bytes moved when they do not; §09.5's dataloader
  position, including whether it is exact enough to continue a run rather than
  restart it; §09.6 checkpoint chains through `omni log`; and §09.1's
  separability, executed by `omni strip --training` and *proved* rather than
  asserted — the tensor digests are compared before and after, and a strip that
  would drop one refuses to write its output
- §11 plugins and the WebAssembly host: a from-scratch interpreter under §11.6's
  restricted profile — imports refused unless they are `omni_plugin/1`, fuel
  metered per instruction, memory capped, `memory.grow` failing rather than
  exceeding it, NaN results canonicalized so two runs cannot differ, and threads,
  exceptions, GC and SIMD refused by opcode at *load* time. The instruction set a
  plugin compiled from C or Rust uses is implemented: the whole i32/i64/f32/f64
  numeric set with conversions and saturating truncation, sign extension, every
  load and store, globals, structured control flow with `br_table`, `call` and
  `call_indirect`, `select` and the bulk-memory operations. On top of it, §11.5's
  plugin manifest as an embedded, content-addressed object, and the §04.7.7
  extension point wired through: `omni example --plugin` builds a container whose
  tensor is a `plugin` node with **no fallback**, and reading it runs the module
  the model shipped — which is §11.8's step 3, the property no other model format
  has
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
- §13.4 streaming and transport: the `.omni.idx` sidecar of §13.4.1, with framing
  of its own so a truncated or edited one is refused rather than half-believed;
  an HTTP/1.1 range store over a socket, with keep-alive, range coalescing,
  retry on a dropped connection, and every object checked against its digest
  before it is returned, because bytes from a CDN edge are bytes from a stranger;
  §02.7's open over a stateless transport, in three requests from the container
  and none at all from a sidecar; and §13.8's index-only container, which
  describes every object, holds no weights, and is *incomplete* rather than
  invalid — `omni fetch`, `omni index` and `omni strip --weights`, with the round
  trips counted so the claim is checkable
- **What a layout costs before the first number comes out.** `omni serve
  --throttle` rate-limits the server and `omni fetch --first-tensor` reads one
  tensor two ways over that link: by range, and by fetching the whole file the
  way a reader with no index has to. The throttle is the point — on a loopback
  socket every layout is instantaneous, which hides the difference exactly where
  §13 claims it matters. On the worked example over 200 KiB/s: 37.7 KiB and
  178 ms by range against 111.2 KiB and 552 ms for the file. The ratio belongs to
  the *model*; the number that belongs to the *format* is underneath it — 5.7 KiB
  of framing and index to reach any tensor at all, which is §02.7's two-read open
  measured over a socket. The whole-file row is marked *modelled* in the output,
  because it is what the hub tooling does rather than a measurement of a runtime
- §13.4.3 `omni serve`: the other half of the transport, read-only. The pack with
  range support, the sidecar generated from it, and every object at
  `/objects/<digest>` with the `immutable` cache header §13.4.2 says is always
  correct there. Writing is not a route and nothing is joined to a filesystem
  path, so there is nothing to traverse into. Having both halves means each is
  tested against a real implementation of the other rather than against a mock:
  CI runs `omni fetch` against `omni serve` and checks, with `hashlib` rather
  than with this crate, that every object served at a digest hashes to it
- §13.5 the OCI registry mapping, as an [OCI image layout] `oras`, `skopeo` and
  `crane` read: the OMNI Manifest object as the config, the container cut into
  `vnd.omni.pack.v1` layers at object boundaries, the object index as its own
  layer, and the annotations a registry UI can show without pulling anything.
  Concatenating the layers reproduces the container byte for byte, and importing
  verifies every blob against the digest that named it before anything becomes a
  file
- §13.5 **the registry client**, which was the missing half: `omni oci push` and
  `omni oci pull` over the distribution API, tested in CI against a real
  `registry:2` rather than a mock — because the argument for the OCI mapping is
  parasitic adoption, and adoption is not a property this repository can assert
  about itself. A model goes up, comes back down byte-identical, verifies at V6
  and can be pulled by digest instead of by tag; a signed container makes the
  same trip with its §12.5 attestation still valid on the far side. Every blob is
  `HEAD`ed first, so what is uploaded is what the registry does not already
  have — and that turns the dedup claim into a measurement made by the party that
  would know. It measures three things, and the middle one is the interesting
  one: the same container under a second tag uploads **nothing**; a *modified*
  model shares **no** blobs with the original, because objects are placed in
  digest order and one changed tensor moves everything after it; and the same
  fine-tune published as a delta container uploads **20 %** of the full model.
  The module docs said all three before anything had been pushed anywhere, and CI
  now measures them instead of repeating them. `https://` is still refused — TLS
  needs a dependency this crate does not have — so what this reaches is a
  plaintext registry: a local one, a mirror, or anything behind a terminator. A
  `401` is told apart from a `404` and the bearer realm it names is reported
  rather than guessed at
- **ONNX, both directions**, which is the row of the capability matrix that tests
  §07 rather than §04: a safetensors file is tensors, a GGUF file is tensors plus
  an architecture enum, and an ONNX file is a computation. The protobuf wire
  format is read and written here — seven kinds of field and a varint, no
  library — and the graph becomes OMNI-IR at the primitive level.

  The interesting part is the line it draws. §07.1's charge against ONNX is that
  its single abstraction level makes every backend pattern-match `attention` back
  out of fifteen primitives, and an importer can repeat that mistake in reverse:
  `Relu` *is* `maximum(x, 0)`, so importing it as two ops would make the export a
  peephole matcher over the graph. So an ONNX op is translated only when **one
  OMNI op means exactly what it means**, and one table is read in both
  directions. Twenty-four op types are on it. Everything else is carried in a
  compat dialect named after its ONNX domain, at the opset the file imported —
  which is the most faithful thing there is to record, since ONNX versions its
  whole opset at once and that is precisely what §07.4.1's per-op versions exist
  to avoid.

  Carrying rather than translating is not a failure, and §11.3 is why: CI checks
  that a container full of `ai.onnx` ops still verifies, still copies, still
  signs, still round-trips byte for byte, and refuses exactly one thing —
  execution — naming the op it refuses. `omni graph --verify` reports those ops
  **indeterminate**, because reporting them invalid would itself be a conformance
  violation (§15.1).

  Two checks make the claim measurable rather than stated. Every initializer is
  re-read through the object graph and compared with the source (I4). And every
  value the imported graph produces is typed by **both** shape functions —
  OMNI's own and the one the file carries — with a disagreement about a concrete
  dimension treated as an error rather than a warning, because one of the two
  readers is then wrong about what the model computes.
  `tools/onnx-fixture.py` is the third implementation, written in Python from the
  protobuf wire format and the operator specifications: it writes the file, reads
  back what the export wrote, and computes what the graph should produce. CI
  checks that the container's tensors are bit-identical to what Python packed,
  that the executed graph agrees with Python's arithmetic on every output, and
  that the exported file is the same bytes as the imported one.

  What the export refuses is worth as much as what it writes. A container with no
  graph is not an ONNX file and says so. A `semantic`-level graph is refused with
  a pointer to `omni graph lower`, because choosing an abstraction level is the
  caller's decision. And an op with no ONNX spelling stops the export with the
  *list* of them — not a lossy export, since an unwritable op is the computation
  rather than lost metadata, and `--allow-lossy` does not cover it. Lowering this
  repository's own worked transformer and exporting it is a measurement of what
  that costs: 49 nodes map, and `omni.nn/attention`, `omni.nn/rope` and
  `omni.tensor/rsqrt` — an op ONNX simply does not have — do not
- **PEFT LoRA import**, as the §08 `Adapter` the capability matrix says it is.
  The thing OMNI adds is in the one required argument: PEFT names its base with a
  *string* and §08.1 pins it with a *digest*, so `--base` is not optional and the
  name PEFT gave is kept as a name. Every field that would change what the update
  *is* — `use_dora`, `fan_in_fan_out`, `use_rslora`, `rank_pattern`,
  `alpha_pattern`, `modules_to_save`, any `peft_type` but `LORA` — is refused by
  name rather than quietly ignored. And the rank-axis requirement is written only
  when the base actually names its axes: a base imported from safetensors names
  none, because safetensors says nothing about them, and asserting a requirement
  the base cannot meet made every attach *invalid* instead of merely unchecked
- **A second reader, in pure Python** —
  [`bindings/python/omni.py`](../bindings/python/omni.py), 878 lines, standard
  library only, BLAKE3 included because Python does not ship it. It exists to test
  a claim this crate cannot test on its own: `docs/design/sdk.md` §5 says a
  conforming C0 reader fits in ~3 000 lines with no dependencies beyond a hash
  function, and the Rust implementation is also the program that *wrote* every
  container it reads, so on its own it cannot tell "the format is simple" from
  "these two programs share an author's assumptions". CI checks the two agree on
  every object digest, on the root digest in full, and on every literal tensor's
  bytes against what the Rust exporter writes — and that what is above C0
  (a compressed object, a `dequantize` expression, a packed layout) is refused by
  name rather than answered wrongly.

  Writing it found two places where the strictness is easy to get subtly wrong,
  both caught by reading real bytes rather than by reasoning: D5 is not "doubles
  only" but the *shortest float encoding that round-trips exactly*, so `1.0` must
  be a half and `0.1` a double; and D7 is "registered tags only" rather than "no
  tags", so refusing every tag refuses a valid container, because §04.3's exact
  rationals are one
- **A conformance suite that tests arithmetic**, not only framing. `numeric/`
  is seven containers that are structurally perfect and can still be read
  wrong: bf16's exponent range, ties-to-even at the f16 boundary, f16
  subnormals, f8e4m3's saturation where an infinity would be, int4's nibble
  order, e8m0's bare exponents — each carrying the publisher's own digest of
  what its tensor evaluates to (§04.3), and one whose declared digest is a lie,
  which a reader that ignores the field will pass and should not. `valid/features`
  is the opposite test — four containers that are valid *and* use something
  optional (a compressed segment, a tensor over sixteen chunks, a quantized
  weight as an expression, a model carrying its own graph), because a reader
  that refuses a feature the format defines today fails differently from one
  that mishandles a feature from tomorrow
- **A Jinja2 → OMNI-CT translator** (`omni jinja`), because §06.9 replaces an
  executed Jinja string with a total language and whether that trade is
  affordable is an empirical question. It converts 14 of a 15-template corpus of
  real families, and the refusals carry the construct, the reason and the byte
  offset — a maintainer can act on `` `loop.cycle(…)` at byte 284 `` and cannot
  act on "translation failed". It converted 10 when it was written: three of the
  four blockers were gaps in §06.9 itself — no loop variable, no slice form, two
  missing standard-library entries — and closing all three is what the other four
  templates were waiting for. The one that remains is `raise_exception`, and it
  should: a total language has no failure form, so a template asserting something
  about its input has to say so differently. Whitespace
  control is carried across, because `{%- … -%}` decides whether a prompt has a
  leading newline and a tokenizer notices
- **Eleven synthesizable architecture families**, one more than the count Gate 2
  asks for: `transformer.decoder`, `transformer.encoder`, `transformer.moe`,
  `cnn.classifier`, `mlp`, `rnn.lstm`, `rnn.gru`, `gnn.mpnn`, `rl.actor_critic`,
  `audio.encoder` and `ssm.mamba` — the last of which needed §07.8.1 written
  before it could exist, since its defining op was named and not defined. Each is
  *executed* in the tests over known weights, and
  each assertion is a property of that architecture rather than "it produced
  numbers": the mixture's output moves when only the router changes, the
  recurrence's first step cannot see the last input, the graph network's node
  moves when its neighbour does and not when a stranger does, the causal audio
  encoder's earlier frames do not move when a later frame changes, the selective
  scan's first output does not move when a later token changes and its last one
  does when an earlier token does. Running them
  is what found `core.scan` declared with one result and returning two, and
  found that `tensor.scatter` could not aggregate messages at all — now fixed in
  §07.4 with the `reduction` attribute ONNX already spells, so the GNN row
  aggregates with a scatter-add over an edge list rather than a dense incidence
  matrix. The count is met
  and the rest of the criterion is not: the gate also wants outputs matched
  against the source framework, and there is no source framework here
- **A reference interpreter for OMNI-IR** (`omni graph run`), which is where §07's
  claim gets tested rather than asserted: a model that describes its own
  computation can be executed by something that was never told its architecture.
  All of `omni.core` including `if`, `while`, `scan`, `map`, `call` and `tuple`;
  all 31 `omni.tensor` ops with a general `einsum` over explicit subscripts;
  `omni.quant`'s four, taking scale and zero from the op's *operands* because that
  is what the IR form does; and the `omni.nn` ops a decoder needs — `embedding`,
  `norm`, `rope`, `activation` and `attention` with `causal`, `window`, `softcap`
  and grouped queries, which the shipped lowering declines and `graph synthesize`
  emits. `conv`, `conv1d_causal`, `pool`, `interpolate` and `moe_route` too, which
  completes the dialect bar one op; a graph is bounded in ops, elements and loop
  iterations, because a graph is untrusted input.

  `ssm_scan` was the one refusal left, and it was a *specification* gap rather
  than an implementation one: §07 registered the op's arity and its
  `delta_softplus` attribute but never said which operand was the state
  transition, whether the timestep was an operand, or whether the discretization
  was zero-order hold or bilinear — readings that give different numbers from the
  same tensors. **§07.8.1 now defines it**: the operand roles, the per-channel
  *and* per-position Δ that makes the model selective, `Ā = exp(ΔA)` and
  `B̄ = ΔB` named separately because that asymmetry is what every published
  implementation computes, and a `reverse` attribute for the bidirectional case.
  The interpreter is a transcription of those five lines, and a test writes them
  out a second time by hand — an op that spent a draft undefined because two
  readings disagree needs a second implementation before its definition is worth
  anything. The registered arity changed with it, which is not a compatibility
  break for a reason §07.8.1 states: no conforming implementation could have
  existed to break

  It earned its keep on the first run. **`graph synthesize` was emitting a graph
  that verified and computed the wrong thing:** the projections were reshaped to
  `[B·S, heads, head_dim]` and handed to `attention`, whose last two axes are keys
  and head dimension — so it attended across the heads of a single token rather
  than across positions. Every shape agreed and `graph --verify` found nothing,
  because there was nothing wrong with the *types*. Running it and asking whether
  a later token could move position 0's logits is what found it. That is the
  difference between verifying a graph and executing one, and it is the whole
  argument for writing an interpreter
- **GPTQ and AWQ export**, which closes §5.3's lossless round-trip claim for two
  rows of the matrix. It is byte-exact for a structural reason: the import never
  converted anything, so exporting is finding the same literals again. CI imports,
  exports and compares tensor by tensor — then re-imports and checks the
  **identical tensor table** comes back, which is the check with teeth, because
  sorting the tensors on the way out keeps every byte and still builds a different
  graph. The config is reconstructed from the container rather than remembered:
  bit width from the packed dtype, group size from the scale grid, act-order from
  whether the scale is gathered, `checkpoint_format` from whether the `+1` node is
  in the expression — so a container that lost its provenance still exports
  correctly. A container whose layers disagree about any of those is refused,
  because a format whose config states them once has no faithful form for it
- **GPTQ and AWQ import**, which is where §05's claim gets tested: quantization is
  a transformation and not a file type, so a packed 4-bit weight becomes
  `permute(dequantize(reshape(permute(qweight)), scheme))` and needs nothing new
  in the evaluator. The int4-in-int32 packing is a `packed` layout, AWQ's GEMM
  interleave is a `gather`, GPTQ's act-order is a `gather`, and the arithmetic is
  one `dequantize` whose `formula` comes from §05.1's closed set. The packed words
  go in unchanged, so the container is expressions *over the source bytes* rather
  than a conversion of them.
  Byte identity is not enough to claim that, and the implementation says so: the
  words are copied verbatim, so comparing them proves nothing about whether they
  are being *read* right. Every layer is therefore dequantized through the
  expression graph and compared against scalar code that shares nothing with the
  evaluator — the check that catches a wrong interleave, a transposed axis, or a
  zero-point convention applied backwards. §05.1 says the closed `formula` set
  exists because *whether the zero point is subtracted before or after scaling is
  a recurring source of silent corruption when converting between GPTQ, AWQ and
  GGUF*; this is where that bites, because AutoGPTQ's original checkpoint format
  stores every zero point one low and nothing in the tensors says so. The offset
  is read from `checkpoint_format`, written as an explicit `+1` node, and named in
  the report — and an unrecognised `checkpoint_format` is refused rather than
  guessed, because guessing shifts every weight by one quantization step
- **safetensors, both directions**, with the importer and exporter contracts of
  `docs/design/import-export.md` §1 implemented rather than paraphrased: every
  tensor verified byte-for-byte against the source before the import claims to
  have copied it (I4), the source digest recorded (I6), every field safetensors
  does not state left *absent* rather than guessed (I1), `__metadata__` keys with
  no OMNI schema preserved in a `Foreign` object and put back on export (I2), and
  the fidelity report attached as a `Provenance` object (I3). On the way out:
  `--plan` computes the loss report without writing a byte (E1), a lossy export
  without `--allow-lossy` writes nothing (E2), and the report is written beside
  the artifact (E3). Includes the detail that quietly corrupts masks — the format
  stores a boolean in a byte where §04.3 gives `bool` a bit, so the import keeps
  the dtype and describes the storage with §04.4's `packed` layout instead of
  turning masks into `u8`
- A strict JSON codec (RFC 8259), because the formats OMNI absorbs are described
  in JSON and there are no dependencies to read it: bounded depth, exact
  integers past 2^53, and refusal of trailing commas, comments, `NaN`, lone
  surrogates and duplicate keys — a permissive reader is how two implementations
  come to disagree about one file
- §15.1 validation levels V0–V6 in the CLI; the V7 rules are implemented and
  reached through `omni sign --verify`

What is **not** implemented, and is reported as such rather than faked:

- §03.7's MAY-level codecs `lz4`, `brotli`, `xz`, `ans-lut` and the two lossy
  ones: reported as unsupported rather than half-decoded
- `https://`. §13.4's HTTP range store is here and speaks HTTP/1.1 over a
  socket, but TLS needs a cryptographic transport stack and this crate has no
  dependencies to provide one. An `https://` URL is refused with that reason
  rather than silently downgraded
- Registry *authentication*, and chunked blob uploads. `omni oci push` and
  `omni oci pull` speak the distribution API and CI pushes to a real
  `registry:2`, but a registry that answers `401` wants a token from a realm that
  is https on every registry this could reach — so the challenge is parsed and
  reported with the URL it would have fetched, and no anonymous retry is
  attempted, because an endpoint this build cannot reach is not made reachable by
  trying twice. Uploads are monolithic `PUT`s, which the specification allows at
  any size; chunking is a resumption story for a 5 GB layer over a bad link, and
  it is named here rather than half-implemented. The OCI referrers API, which is
  how an adapter or a signature would be linked to the model it belongs to, is
  also not here
- `omni convert --requantize`. `--cast` is here — it converts, measures the
  error it introduced, records the recipe as provenance and can re-read its own
  output to check it — but requantizing is a *search over a calibration set*
  (§05.5), and a build with no calibration data would have to invent either the
  data or the scales. It is refused with what it would need
- `omni mount` (§13.9), which needs FUSE
- Every importer and exporter except safetensors, PyTorch, GGUF, ONNX, PEFT,
  GPTQ, AWQ and a whole Hugging Face repo.
  The capability matrix in `docs/design/import-export.md` §3 has 25 rows and this
  build implements eight of them; EXL2 does not exist, and a
  request for one is refused by name rather than half-attempted. Export covers
  safetensors, GGUF, ONNX, PEFT, GPTQ and AWQ — not PyTorch, because §12.10
  clause 4 says never to re-emit pickle. 3-bit GPTQ and AWQ's `gemv`/`marlin`
  versions are refused for the reasons named above, and so are GGUF's `IQ*`
  types, whose codebooks are in llama.cpp's source rather than in the file
- §12.10's confined child process for the pickle import. The restricted
  unpickler is implemented in full — an opcode allowlist, 19 resolvable symbols,
  no call mechanism beyond tensor reconstruction — and the sandbox is not, on
  the argument that there is nothing to confine: it is a parser for a data
  language, not an evaluator with a filter in front of it. A build that ever
  grows a general evaluator needs the sandbox back
- The **writer** side of the C ABI. `omni-ffi` reads: open, verify, walk, bytes,
  values, plan, DLPack. A C caller cannot yet *build* a container, so
  `ModelBuilder` is Rust-only
- `mmap`, which needs `unsafe`. `store::FileStore` is the answer to what `mmap` was
  for here: a container opened and read one range at a time, counting its reads,
  so §02.7's two-read open and §04.7.4's partial reads are measurements
  (`omni open`) rather than constructions. What a production reader gains from
  `mmap` is the page cache doing the buffering; what it does not gain is a
  different parse

See [`docs/design/roadmap.md`](../docs/design/roadmap.md) for the plan.

## Tests

498 tests covering: SHA-256 against FIPS 180-4 vectors; BLAKE3 against the
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
stores, that opening a file-backed container costs four reads and under a percent
of the file, that a range read moves exactly its range and no more, that a
damaged trailer fails the open while a corrupted object body fails the *read*
where the digest is, and a container→directory→container round trip that is
byte-exact,
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
than by inventing one. And, for the two transformer
families: that an encoder emits `causal: false` on every attention op, returns
hidden states rather than logits, and does not demand a language-modelling head
— while a decoder built from the same weights still names `lm_head.weight` as
missing; and, run over real weights, that the encoder's output at position 0
*moves* when a later token changes, which is the only check that can tell an
encoder from a decoder once both verify. And, for §09: that a
training state and a shard map round-trip through canonical CBOR; that shards
which leave a gap, overlap, name a mesh dimension that does not exist or disagree
with its extent are each reported with their rule; that resharding four ranks to
two moves no bytes while four to eight says which tensor must be rechunked and a
mesh that cannot divide the tensor is refused with the numbers; that
reproducibility is *reported* rather than claimed, naming the stateful generator
and the vague dataloader position that limit it; and that stripping a checkpoint
removes the training state by reachability while every weight digest survives.
And, for §13: that a sidecar carries the same framing as the container it
describes and that a truncated one — at every length — and a single flipped bit
anywhere in its header are refused rather than half-parsed, as is a sidecar whose
superblock was rewritten; that a container and a sidecar each refuse to be
mis-read as the other; that an index-only container describes every object,
holds no weights, still validates, and reports the model's real size while being
a fraction of it; that range coalescing merges what is adjacent and within its
slack while honouring its cap; that `https://` is refused with its reason; and —
against a real HTTP server on a real socket — that a container opens in three
requests and a sidecar makes that zero, that many objects fetch in fewer
requests than there are objects, that a range of an object moves only its range,
that a dropped connection is retried rather than lost, that a chunked response
is dechunked, that a server which ignores `Range` is told apart from one that
fails, that an absent object is absent rather than an error, that a compressed
container decodes with the parameters its superblock declares, that a single
tampered byte from the wire is refused by the digest that covers it, and that a
sidecar for a *different* container is caught by R-X02 in one request rather than
used to plan ranges into the wrong offsets of the right file. And, for the object
server: that a request line names one of the four routes or nothing, with
traversal attempts, an unknown pack name and a non-digest path all reaching
`NotFound` and every write method refused; that a digest path is parsed strictly;
that every object is served at its own digest and hashes to it; that an unknown
digest is a counted 404; that the pack is range-readable by this crate's own
client in three requests; that the sidecar it generates opens the pack it
describes and passes R-X02; that an unsatisfiable range is refused rather than
answered short; that `HEAD` gives the length with no body and leaves the
connection usable; that an index-only container serves its structure and 404s its
weights; and that six kinds of malformed request end their own connection and
nothing else. And, for the OCI mapping: that the layout is one an OCI reader
recognises — the marker file, the index, the descriptors, every blob at the path
its own sha256 names, and no `..` anywhere; that the layers reassemble into the
same bytes for one, four and nine tensors; that the cuts cover the file exactly
once with no gap, overlap or empty pack at four pack sizes; that an index-only
container maps and says it is partial; that a tampered blob, a missing blob, a
missing marker, an unknown layout version, a missing index and reordered layers
are each refused; and — asserted so nobody writes it down wrong — that
re-exporting a model with one tensor changed does *not* share most of its blobs,
while a delta container's layout is a fraction of the base's, which is where the
dedup actually comes from. And, for PEFT: that a config's every
update-changing field is refused by name; that a regex `target_modules` and a
non-LoRA `peft_type` are refused; that factors are checked against the base they
claim to update, so a rank that disagrees, a target the base lacks and half a
factor pair are each errors; that the naming convention is *read* rather than
assumed; that a tensor nobody asked for makes the import lossy and says which;
that the imported adapter pins its base by digest, declares it as a non-required
parent that `delta::parents` can actually read back, and attaches to that base
binding each layer's own factors while touching nothing it did not train.

And, for the interpreter, the tests that check what it *computes* rather than that
it runs: that a convolution is a cross-correlation and would give the negatives of
its answers if the kernel were flipped; that padding puts zeros where it says and
a group never sees another group's channels; that a causal 1-D convolution's
earlier outputs do not move when a later input changes, which symmetric padding
would fail; that pooling defaults to a non-overlapping window; that the two
interpolation modes disagree and half-pixel centres put the samples where they
belong; that MoE routing softmaxes over every expert before taking the top k, that
`normalize` is what makes the chosen weights sum to one, and that a transposed
routing matrix is a named error rather than a silent transpose; that a causal attention's first position is exactly `v[0]` whatever the
scores are, and that removing the mask changes that, so the mask is load-bearing;
that grouped queries share the kv heads they are supposed to and a grouping the
shapes cannot support is an error rather than a modulo that produces numbers; that
a sliding window forgets and a soft cap keeps a logit of 1000 finite; that RMS norm
leaves unit mean-square and layer norm also centres, and that the two are
different functions; that both RoPE conventions preserve each pair's length and
disagree with each other; that `einsum` agrees with `matmul` and refuses an
ellipsis; that `scan` agrees with `cumsum`; that `while` terminates and a bound
stops one that would not; that a lowered graph computes what the graph it came
from computes, which is §07.2's claim; and — the whole thing — that a synthesized
decoder produces a distribution over the vocabulary and logits at position 0 that
no later token can move.

And, for GPTQ and AWQ, the tests that would notice a wrong answer rather than a
crash: that a fixture packed by the formats' own rules dequantizes to values
computed in the test, so the transpose, the interleave and the grouping are each
checked against arithmetic and not against the importer; that reading AWQ's
`qweight` with GPTQ's layout gives a *different* answer, because a test that
passed either way would be testing nothing; that the two `checkpoint_format`
conventions differ by exactly one scale in every weight, and that which one was
assumed is in the report; that `g_idx` decides whether act-order is a gather and
`desc_act` does not, with the disagreement reported; that an ascending `g_idx` is
checked to be ascending rather than ignored, and is the one source tensor not
stored because `group_size` already says it; that 3-bit GPTQ, 8-bit AWQ, `gemv`,
an unknown checkpoint format and a config naming the other method are each refused
by name; that a `scales` grid disagreeing with `group_size` is an error rather
than a plausible dequantization; and that a layer too large to dequantize whole
says so instead of being silently skipped.

[OCI image layout]: https://github.com/opencontainers/image-spec/blob/main/image-layout.md
And, for the WebAssembly host: that
arithmetic, locals, a real loop, memory loads and stores, `call`,
`call_indirect`, `br_table`, `if`/`else` and the bulk-memory operations all do
what WebAssembly says; that an out-of-bounds access, a division by zero, an
`unreachable` and an indirect call with the wrong signature each trap; that fuel
runs out instead of hanging, that the memory cap makes `memory.grow` return −1
rather than allocating, and that a module declaring more memory than the cap never
instantiates; that an import from anywhere but `omni_plugin/1` and an opcode from
a forbidden proposal are refused at load; that every NaN the host produces is the
same NaN and `min`/`max` follow WebAssembly's rules rather than Rust's; that
`read_object` sees only the objects it was given and returns −1 otherwise, and
that `abort` traps with the plugin's own message; that a malformed module, and the
same module truncated at every length, are errors rather than panics; and, end to
end, that a plugin manifest round-trips, that a missing or unrunnable module is
reported rather than hidden, and that the example module computes
`x × f` through the host and refuses the argument count it was not written for.
And, for JSON: that every scalar kind
parses, that an integer past 2^53 survives exactly rather than becoming a float,
that escapes and surrogate pairs decode, that a truncated header is an error at
every length, that malformed UTF-8 is refused rather than replaced, that nesting
is bounded through both the array and the object arm, that the writer is
deterministic and its output re-reads to the same value, and that thirty
non-JSON inputs — trailing commas, comments, `NaN`, `+1`, `01`, lone surrogates,
duplicate keys — are each an error. And, for safetensors: that every one of the
format's fifteen dtypes maps onto exactly one OMNI dtype and back; that a header
which disagrees with its buffer is refused, whether by a wrong extent, an
overlap, a gap, trailing bytes, an offset past the end, an absurd declared header
length, or truncation at any length; that a boolean mask keeps the dtype `bool`
and the byte-per-element layout, validates under R-T02, and exports as `BOOL`
byte-for-byte; that an import verifies every tensor against the source and
reports what it checked, invents no field the file does not state, preserves the
metadata keys it cannot model, and is reproducible and addressed by its source
digest; that a preserved key comes back on export while a `Foreign` object from
another format is not raided for keys; that an export refuses without consent and
names each thing it would lose; that a dtype safetensors cannot spell is
reported before it is widened to F32; and that export-then-import reproduces
every tensor object digest. And that an object whose `t`
contradicts the index's otype is invalid (R-O02).
Every
container-level test runs under both mandatory digest algorithms.

And, for the Hugging Face repo importer, the tests that are about the five files
meaning something only together: that a config maps onto §06.2's names while
every key it does not model survives under its own; that `rope.interleaved` —
which §06.3 calls the field responsible for the most silent corruption in format
conversions — is written only for families whose `transformers` implementation is
unambiguous, is reported as an assumption when it is, and is *omitted* for an
unknown family rather than guessed, with `rope_theta` kept anyway; that BPE
merges become id pairs and a merge naming a token the vocabulary lacks is an
error rather than a skipped line; that a vocabulary with a hole in it is an error,
because every id past the hole would be wrong; that a normalizer or pre-tokenizer
outside §06.7's catalogue is carried by name so encoding is *indeterminate*
rather than wrong; that a Unigram model keeps its scores; that a tokenizer model
§06.7 has no entry for is refused by name; that the imported tokenizer reads back
out of a packed container and reports nothing unsupported; that a chat template
§06.9 cannot express is left out and named instead of shipped; and that two
repos whose shards differ only in how they were cut get different source
digests.

And, for PyTorch, the tests that are about the threat rather than the format:
that a global outside the allowlist is a hard error naming it — `posix.system`,
`os.system`, `subprocess.Popen`, `builtins.eval`, `torch.load` and four more,
each checked individually; that `INST`, `OBJ` and the three extension-registry
opcodes are refused by the name of the opcode; that `BUILD` on anything but a
dict is refused, because `__setstate__` needs the class; that a pickle with no
`STOP` runs off the end rather than forever; that a truncated archive at every
length and a bit flipped at a hundred positions inside the pickle are errors and
never panics; and that the allowlist is exactly nineteen entries, asserted so it
cannot grow by accident and stop being a security property. Then the format:
that a transposed view keeps its strides instead of being densified into a
different array, that a second view of one storage is reported rather than
silently duplicated, that a view running past its storage is caught with the
byte counts, and that Zip64's 0x0001 extra field is read — which is not an edge
case when a 7 B model in fp16 is 14 GB.

And, for the C ABI, the tests that a C caller would find out the hard way: that a
null handle is a usage error rather than a segfault and freeing null does
nothing; that 512 zero bytes are *invalid* rather than a panic crossing the
boundary; that a tensor handle survives its store being closed, which is the
guarantee that makes a zero-copy binding possible at all; that a DLPack tensor
survives every OMNI handle being freed and then frees itself; that `bf16` goes
over DLPack as `kDLBfloat` with null strides while `i4` is refused by name rather
than passed off as `uint8`; that the C0 baseline reports this model infeasible
and names the feature it lacks instead of quietly planning less; and that the
status codes are the CLI's exit codes, checked value by value, because a C caller
and `omni` disagreeing about what happened is the failure mode that matters.

```console
$ cargo test
test result: ok. 498 passed; 0 failed
$ cargo clippy --all-targets -- -D warnings
    Finished (no warnings)
```

CI lints with whatever clippy ships in the current stable toolchain, which may
be newer than yours and may therefore know lints you do not have locally. If CI
flags something `cargo clippy` accepted on your machine, `rustup update` first.
Clippy runs on stable only; beta is an early-warning job and does not gate the
branch.
