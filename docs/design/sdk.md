# SDK Design

**Principle:** one implementation of the hard parts, many thin bindings. Parsing
untrusted binary input is written **once**, in Rust, and everything else calls
into it. Reimplementing an untrusted-input parser per language is how every media
format acquired its CVE list.

## 1 Crate topology

```
omni-cbor      canonical CBOR encode/decode + OSD schema validation   no_std-able
omni-hash      BLAKE3/SHA-2 + Bao verified streaming                  no_std-able
omni-core      objects, refs, container parse/serialize, index,       no_std + alloc
               tensor expressions, dtype/layout math
               #![forbid(unsafe_code)]
omni-io        stores: mmap file, dir, http, oci, s3, memory          std
omni-eval      expression evaluation, materialization, caching        std
omni-ir        OMNI-IR parse/verify/rewrite, dialect resolution       std
omni-plugin    WASM host (wasmtime or a minimal interpreter)          std
omni-convert   quantizers, delta extraction, merges                   std
omni-import-*  one crate per source format                            std
omni-export-*  one crate per target format
omni-ffi       C ABI (cdylib + staticlib) — the binding substrate
omni-cli       the `omni` binary
```

`omni-core` builds `no_std + alloc` so it can run in embedded loaders,
bootloaders, and WASM. It has no dependency that performs I/O — the parse layer
is given bytes and returns structures.

## 2 Rust API

The reference API. Everything else mirrors it.

```rust
use omni::{Store, Objective, Capabilities, TensorView};

// Open — 2 reads, mmap, no tensor payload touched.
let store = Store::open("model.omni")?
    .with_fallback(Store::http("https://cdn.acme.com/objects/")?)
    .verify_level(Verify::Selective);

let model = store.root()?.as_model()?;

// Metadata without touching weights.
println!("{} params, {} layers",
         model.meta().params_total().unwrap_or(0),
         model.meta().arch().get_u64("n_layers").unwrap_or(0));

// Negotiate.
let caps = Capabilities::detect()?;
let plan = model.resolve(&caps, Objective::MinLatency)?;
for w in plan.warnings() { eprintln!("warning: {w}"); }

// Instantiate: maps what it can, materializes what it must.
let inst = plan.instantiate(&store)?;

// Zero-copy where the plan chose direct-map.
let t: TensorView<'_> = inst.tensor("model.embed_tokens.weight")?;
assert_eq!(t.shape(), &[128256, 4096]);
match t.bytes() {
    Bytes::Mapped(s)  => feed_gpu(s),          // &[u8] into the mmap, no copy
    Bytes::Owned(b)   => feed_gpu(&b),         // materialized
}

// Range-driven partial read: fetches only the covering chunks.
let rows = t.slice(&[0..64, 0..4096])?.to_vec::<half::bf16>()?;
```

### 2.1 Lifetimes and the mapping

```rust
pub struct TensorView<'a> { /* borrows the Store's mapping */ }
pub struct OwnedTensor    { /* Arc<Mapping> or Arc<[u8]>  */ }
```

`TensorView<'a>` borrows; `OwnedTensor` holds an `Arc` to whatever backs it
(mapping or heap). The `Arc` variant is what crosses FFI and language
boundaries, because a Python object outliving a `&'a` is the classic
zero-copy-binding bug. **Rule: the FFI layer never exposes a borrow.**

### 2.2 Writing

```rust
let mut b = ModelBuilder::new()
    .reproducible(true)
    .align(4096)
    .codec(Codec::Zstd { level: 3, ..Default::default() })
    .chunker(Chunker::Fixed(4 << 20));

b.meta().name("acme/llm-8b").license_spdx("Apache-2.0");
b.tensor("model.embed_tokens.weight")
 .shape([128256, 4096]).dtype(DType::BF16)
 .axes(["vocab", "hidden"])
 .from_bytes(&data)?;                       // chunked + hashed streaming
let manifest = b.finish()?;
Container::write("model.omni", &manifest, &b.objects())?;
```

The builder streams: bytes are chunked, hashed and written as they arrive, so
writing a 1 TB checkpoint never requires 1 TB of RAM and never buffers a whole
tensor.

### 2.3 Async

`omni-io` exposes both sync and async store traits (`Store` / `AsyncStore`) via a
shared core, with `maybe-async` style codegen rather than two implementations.
HTTP and object-store backends are async-native; file and mmap are sync-native
and wrapped with `spawn_blocking`.

## 3 C ABI (`omni.h`)

The binding substrate. Stable, versioned, opaque handles, no callbacks into Rust
panics.

```c
typedef struct omni_store   omni_store;
typedef struct omni_model   omni_model;
typedef struct omni_plan    omni_plan;
typedef struct omni_tensor  omni_tensor;

omni_status omni_store_open(const char *uri, const omni_open_opts *opts,
                            omni_store **out);
omni_status omni_store_root(omni_store *s, omni_model **out);

omni_status omni_model_meta_json(omni_model *m, const char **json, size_t *len);
omni_status omni_model_resolve(omni_model *m, const omni_caps *caps,
                               omni_objective obj, omni_plan **out);

omni_status omni_plan_instantiate(omni_plan *p, omni_store *s, omni_inst **out);
omni_status omni_inst_tensor(omni_inst *i, const char *name, omni_tensor **out);

/* zero-copy access; buffer valid until omni_tensor_release */
omni_status omni_tensor_data(omni_tensor *t, const void **ptr, size_t *len);
omni_status omni_tensor_dlpack(omni_tensor *t, DLManagedTensor **out);

void        omni_tensor_release(omni_tensor *t);
const char *omni_last_error(void);          /* thread-local */
uint32_t    omni_abi_version(void);
```

Design rules:

- **Never unwind across FFI.** Every entry point catches panics and maps them to
  `OMNI_EINTERNAL`.
- **Explicit ownership**, one `*_release` per acquiring call.
- **`omni_status` mirrors the CLI exit codes** (§CLI.10), including the crucial
  `OMNI_INDETERMINATE`.
- **DLPack** (`omni_tensor_dlpack`) is the interop lingua franca: PyTorch, JAX,
  CuPy, TensorFlow, MLX and NumPy all consume it with zero copy.

### 3.1 What is built

This one exists: [`reference/omni-ffi`](../../reference/omni-ffi) and
[`omni.h`](../../reference/omni-ffi/include/omni.h), built as both a `cdylib` and
a `staticlib`, with [`examples/read.c`](../../reference/omni-ffi/examples/read.c)
driving the whole path in C and CI compiling it at `-Wall -Wextra -Werror`.

Four things about the build differ from the sketch above, and each is a decision
rather than a shortfall:

- **`unsafe` lives here and nowhere else.** `omni-core` stays
  `#![forbid(unsafe_code)]` because it parses untrusted input; a C ABI cannot be
  written without `unsafe`, so it is confined to a crate that does no parsing.
  The `unsafe` blocks are three kinds only: dereference a handle the caller
  handed back, read a caller's NUL-terminated string, hand out a pointer an
  `Arc` keeps alive.
- **No `omni_inst`.** Instantiation as a separate object buys nothing while
  every store is a single container: `omni_model_tensor` takes a tensor
  directly, and the plan is still there to negotiate with. The handle appears
  when there is more than one store behind it.
- **Handles are reference-counted, so `omni_store_close` is order-free.** A
  tensor outliving its store is the *supported* case — §2.1's "the FFI layer
  never exposes a borrow", enforced rather than documented. A DLPack tensor
  outlives every OMNI handle, which is what a Python consumer will actually do.
- **`omni_tensor_values` beside `omni_tensor_bytes`.** Stored bytes are the C0
  path and are handed over without a copy when the tensor is one raw object;
  a computed value — a `dequantize`, a `cast` — has no stored bytes at all and
  says `OMNI_INDETERMINATE` rather than handing back its operand, which would be
  the wrong array with the right length.

DLPack refuses what it cannot spell. It describes whole-byte lanes, and §04.3's
`i4`, ternary, fixed-point and codebook dtypes are not that, so
`omni_tensor_dlpack` returns `OMNI_INDETERMINATE` naming the dtype instead of
passing a 4-bit weight off as `uint8`. Same for a non-dense layout, because
`strides == NULL` in DLPack means dense row-major and nothing else.

## 4 Language bindings

### 4.1 Python (`omni-py`, PyO3)

```python
import omni

m = omni.open("model.omni")                     # or hf://, oci://, https://
print(m.meta.params_total, m.meta.arch.family)
print(m.tokenizer.encode("hello world"))

plan = m.resolve(omni.capabilities(), objective="min-memory")
inst = plan.instantiate()

w = inst["model.embed_tokens.weight"]           # lazy handle
t  = torch.from_dlpack(w)                       # zero-copy when direct-mapped
a  = np.asarray(w)                              # buffer protocol
jx = jax.dlpack.from_dlpack(w)

with omni.writer("out.omni", reproducible=True) as wr:
    wr.meta(name="acme/llm-8b", license="Apache-2.0")
    wr.tensor("w", torch_tensor, axes=("out", "in"))
```

- Implements `__dlpack__`/`__dlpack_device__` and the buffer protocol.
- Releases the GIL for all I/O and hashing.
- `omni.safetensors_compat` gives a drop-in `load_file`/`save_file` so existing
  code migrates by changing one import.
- Type stubs shipped; `mypy`-clean.

**What exists today is not this.** PyO3 needs a dependency, and the reference
implementation has none on purpose, so the binding above is a design and not a
build. Two things that are built stand where it will:

- [`reference/omni-ffi`](../../reference/omni-ffi), the C ABI of §3, which is
  what a PyO3 — or `ctypes`, or `cffi` — binding would call. It already returns
  DLPack, so `torch.from_dlpack` works today from any language that can call C.
- [`bindings/python/omni.py`](../../bindings/python/omni.py): a **C0 reader in
  pure Python, no dependencies**, which reads a container, verifies every digest,
  and hands back a literal tensor's bytes. No DLPack, no zero copy, no writer.
  Its purpose is different from a fast binding's, and §5.1 is that purpose.

### 4.2 C++ (`omni.hpp`)

Header-only RAII over the C ABI; C++17.

```cpp
auto store = omni::Store::open("model.omni");
auto model = store.root();
auto plan  = model.resolve(omni::Capabilities::detect(), omni::Objective::MinLatency);
auto inst  = plan.instantiate(store);
std::span<const std::byte> w = inst.tensor("lm_head.weight").bytes();   // zero-copy
```

`std::span`, `std::expected`-style error type (or `outcome` pre-C++23), no
exceptions across the ABI boundary, and an optional `mdspan` view over shaped
tensors.

### 4.3 Go

Layered: a **pure-Go reader** for `C0` (no cgo — matters enormously for
deployment) and a cgo binding for full functionality.

```go
st, _ := omni.Open("model.omni")
m, _  := st.Root()
fmt.Println(m.Meta().ParamsTotal())
t, _  := m.Tensor("lm_head.weight")
b, _  := t.Bytes()          // []byte aliasing the mmap
```

The pure-Go reader is viable precisely because C0 requires only: header parse,
index binary search, canonical CBOR decode, BLAKE3, and literal tensors. That is
~3 000 lines — a deliberate design goal (§ "the C0 budget").

### 4.4 Java / Kotlin

Panama FFM (JDK 22+), no JNI:

```java
try (var store = Omni.open(Path.of("model.omni"))) {
    var model = store.root();
    MemorySegment w = model.tensor("lm_head.weight").segment();  // zero-copy
    var buf = w.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
}
```

Falls back to a JNI shim on older JDKs. `MemorySegment` maps directly onto the
mmap with correct lifetime scoping — the first time the JVM has had a clean
answer for this.

### 4.5 Swift

```swift
let store = try OmniStore(path: "model.omni")
let model = try store.root()
let t = try model.tensor("lm_head.weight")
let mlx = try t.asMLXArray()          // zero-copy into MLX
let mlm = try t.asMLMultiArray()      // CoreML interop
```

Swift is where OMNI meets Apple silicon: `MLXArray` and `MLMultiArray` bridges,
plus `mmap` with `MADV_WILLNEED` tuned for unified memory.

### 4.6 JavaScript / TypeScript

Shipped as:

- **Node** (`napi-rs`): full functionality, native mmap.
- **Browser** (`wasm-bindgen`): C0/C1 reader over `fetch` + HTTP range requests,
  streaming and verifying with Bao (§13.3). Loads a model into WebGPU buffers
  progressively.

```ts
const model = await omni.open("https://cdn.acme.com/m/llm-8b.omni");
console.log(model.meta.paramsTotal);
for await (const t of model.stream(["model.layers.0.*"])) {
    device.queue.writeBuffer(gpuBuf, 0, t.bytes);
}
```

The browser case is a real design constraint, not an afterthought: it forced the
two-round-trip open (§02.7), range-friendly indexes, and verified partial reads.

### 4.7 Others

C# (P/Invoke over the C ABI, `Span<byte>`), Julia (`ccall` + `unsafe_wrap`), R,
Ruby, PHP — all straightforward over the C ABI. The specification's job is to
make them all *possible*; the working group maintains Rust, C, Python, and Go,
and the rest are community-maintained with the conformance suite as the arbiter.

## 5 The C0 budget

A hard design constraint, tracked as a metric:

> **A conforming C0 reader must be implementable in under 3 000 lines of
> straightforward code in any systems language, with no dependencies beyond a
> hash function.**

Breakdown (measured against the reference Rust implementation):

| Component | LoC |
|---|---:|
| Header + trailer + segment parsing | ~250 |
| Object index (binary search, buckets, aux) | ~250 |
| Canonical CBOR decoder (strict subset) | ~700 |
| Digest verification (BLAKE3 or SHA-256 via a library) | ~80 |
| Ref resolution + store abstraction | ~250 |
| Manifest / Metadata / TensorTable / TensorDesc structs | ~500 |
| dtype/layout offset math for registered dtypes | ~400 |
| `literal` tensor materialization + zstd | ~200 |
| Error handling, bounds checks, limits | ~300 |
| **Total** | **~2 930** |

This budget is why the container is boring, why the index is a fixed array, why
CBOR is used instead of a bespoke encoding, and why the expression evaluator is
C1 rather than C0. **If a proposed feature would push C0 over budget, it belongs
in a higher profile.** That rule has been applied throughout this specification.

### 5.1 Measured, in a second language

The table above is a breakdown of the Rust implementation, which is also the
program that *wrote* the containers it reads. On its own that cannot distinguish
"the format is simple" from "these two programs share an author's assumptions".

So there is a second reader: [`bindings/python/omni.py`](../../bindings/python/omni.py),
**878 lines of pure Python with no dependencies**, written from the specification
rather than from the Rust. It implements BLAKE3 from scratch — the one primitive
C0 needs that Python does not ship — plus CRC-32C, the two-read open, the index
with its bucket table, canonical OMNI-CBOR with D1–D8 enforced, the object graph,
and literal tensors.

CI runs it against every container the Rust implementation writes and checks that
the two agree: every object's digest, the root digest in full, and every literal
tensor's bytes compared against what the Rust *exporter* produces. It also checks
what C0 does not cover — a compressed object, a `dequantize` expression, a packed
layout — is refused **by name** rather than answered wrongly, because a floor is
only honest if what sits above it is named.

Two things the exercise found, which is the argument for doing it at all:

- The budget is comfortable. 878 lines against ~3 000, in a language with none of
  Rust's advantages for this kind of work, with room left over for the strictness
  checks a reader could technically skip.
- The strictness is load-bearing and easy to get subtly wrong. D5 is not "doubles
  only" — it is the *shortest float encoding that round-trips exactly*, so `1.0`
  must be a half and `0.1` must be a double, and a reader that accepts either
  form for either value has admitted two digests for one object. D7 is not
  "no tags" but "registered tags only", and refusing all of them would refuse a
  valid container, because §04.3's exact rationals are a tag. Both were written
  the wrong way first and caught by reading real bytes.

## 6 Error model

```rust
pub enum Error {
    Invalid(Rule, Span),      // violates a normative rule; cite the rule ID
    Indeterminate(Reason),    // valid but not fully understandable here
    Incomplete(Digest),       // object not available in any store
    Infeasible(Vec<Unmet>),   // capability negotiation failed
    Policy(PolicyViolation),  // refused by configuration
    Io(io::Error),
}
```

Every `Invalid` carries the conformance rule ID (`R-C07`) and a byte span, so a
user gets *"R-C07 non-zero padding at 0x4A000"* rather than *"parse error"*.
This is a small thing that dramatically changes how quickly an ecosystem
converges on correct writers.

## 7 Threading and memory

- Store handles are `Send + Sync`; readers are lock-free (the index is
  immutable).
- Materialization is parallel per tensor and per chunk, with a bounded work-stealing
  pool and a memory budget that applies backpressure rather than OOMing.
- `madvise(WILLNEED)` on planned extents, `MADV_DONTNEED` after use for
  streaming loads; `MAP_POPULATE` optional for latency-critical starts.
- Optional `O_DIRECT` / GPUDirect Storage path: because chunks are 4 KiB-aligned
  and their offsets are known before any parse, a DMA descriptor list can be
  built from the index alone and handed to `cuFileRead` — weights land in GPU
  memory without ever touching host RAM.

**See also:** [CLI](cli.md) · [Performance](performance.md) · [§10 Runtime](../spec/10-runtime.md)
