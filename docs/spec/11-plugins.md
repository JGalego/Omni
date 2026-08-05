# OMNI/1.0 — §11 Plugin and Extension System

The specification's job is to define a *substrate*, not a catalogue. Everything
domain-specific — architectures, quantizers, tokenizers, codecs, metadata
vocabularies — enters through this section.

---

## 11.1 Extension points

| Point | Mechanism | Section |
|---|---|---|
| Object types | `otype ≥ 0x8000` | §01.6 |
| Structure keys | `ext` maps with namespaced keys | throughout |
| Schemas | `t` URIs in any namespace | §03.4 |
| Dtypes | dtype descriptors + `plugin` kind | §04.3 |
| Layouts | `{"k":"opaque","id":"…"}` and registered layout kinds | §04.4 |
| Tensor operations | `plugin` expression node | §04.7.7 |
| Quantization schemes | scheme descriptors + `plugin` | §05.2.10 |
| IR dialects and ops | `DialectRef` | §07.4 |
| Compression codecs | codec descriptors | §03.7 |
| Tokenizers | `kind:"plugin"` + WASM | §06.7.2 |
| Adapter methods | `method:"plugin"` + attach rules | §08.1 |
| Signature schemes | COSE algorithm identifiers | §12.5 |
| Runtime cache kinds | `kind` string | §10.6 |
| Segment kinds | `kind ≥ 0x8000` | §02.4.2 |

## 11.2 Namespaces

Identifiers are `namespace/name` with an optional `@version`:

```
omni.core/manifest          reserved: this specification
omni.nn/attention           reserved: OMNI working group registries
org.acme/moe-router         third party, reverse-DNS
com.nvidia/trt-engine       vendor
x.local/experiment-3        the "private use" namespace: never registered,
                            never conflicting, never expected to be understood
```

Rules:

- `omni.*` is reserved for the working group.
- Everything else SHOULD use reverse-DNS or a registered short prefix.
- `x.*` is guaranteed unregistered forever — the equivalent of `X-` headers done
  right, for local experimentation. Tools MUST NOT warn about `x.*`.
- Namespace comparison is exact, byte-wise, case-sensitive, after NFC
  normalization (§03.2 D9).

## 11.3 Criticality: the PNG rule, generalized

PNG's masterstroke was encoding *how to treat what you do not understand* in the
chunk name itself. OMNI generalizes it to four independent bits, carried in the
object index (`oflags`) and, for inline extension values, in the extension
wrapper.

| Bit | Name | If the reader does not understand the item… |
|---|---|---|
| 0 | `CRITICAL` | …it MUST NOT execute the model. (It may still inspect, verify, copy, and re-serialize it.) |
| 1 | `CACHEABLE` | …it MAY delete it. Derived data. |
| 2 | `SAFE_TO_COPY` | …it MAY preserve it when rewriting the container. If clear, a rewrite that changes related content MUST drop it (it may have become stale). |
| 3 | `STRUCTURAL` | …it MUST understand it to validate integrity (e.g. a new index format). Failing this bit means "cannot verify", which is distinct from "invalid". |

The four combinations that matter:

| CRITICAL | SAFE_TO_COPY | Meaning | Example |
|:--:|:--:|---|---|
| 1 | 1 | Required to run; survives rewriting | a custom quantization scheme's parameter object |
| 1 | 0 | Required to run; invalidated by edits | a precomputed layout tied to a specific chunking |
| 0 | 1 | Ancillary, portable | author's private metadata, benchmark notes |
| 0 | 0 | Ancillary, ephemeral | a stale runtime cache from another machine |

**Normative consequence:** *a container containing objects a reader does not
understand is not thereby invalid.* A 2026 tool can verify, sign, repack,
deduplicate, transfer, garbage-collect and inspect a 2071 container, and will
correctly refuse only the specific operation (execution) that actually requires
the unknown semantics. This is the single most important forward-compatibility
property in the format.

## 11.4 Extension wrapper

For extension data appearing inside a known object:

```cbor-diag
"ext": {
  "org.acme/deploy": {
     "crit": false, "copy": true,
     "v": 2,
     "schema": [23, h'…'],            ; optional -> Schema object
     "data": { … }
  }
}
```

A reader that does not know `org.acme/deploy` reads `crit`/`copy` and acts
accordingly, without parsing `data`. Since `data` is canonical CBOR, it can be
copied byte-exactly and re-emitted in sorted position — preserving digests of the
enclosing object across tools that do not understand it.

> **Important subtlety.** Because a structure object's digest covers its
> extensions, a tool that *drops* a non-safe-to-copy extension necessarily
> changes the object's digest and hence the model's identity. That is correct and
> intended: the result is a different (derived) model, and `omni` records the
> relationship in `Provenance`. Silent identity-preserving mutation is impossible
> by construction.

## 11.5 Plugin manifest

A plugin is itself a content-addressed, signable artifact:

```cbor-diag
{ "t":"omni.plugin/manifest", "v":1,
  "ns":"org.acme/moe", "version":2,
  "provides": {
     "otypes":[32768,32769],
     "dtypes":["org.acme/mx-hybrid"],
     "expr_ops":["org.acme/moe.pack"],
     "dialects":[{"ns":"org.acme/moe","v":2}],
     "codecs":["org.acme/ans-moe"],
     "schemas":["org.acme/moe.config"] },
  "requires": [{"ns":"omni.core","v":1},{"ns":"omni.tensor","v":1}],
  "modules": {                          ; WASM implementations
     "validate":  {"ref":[30,h'…'],"export":"validate"},
     "shape":     {"ref":[30,h'…'],"export":"shape"},
     "reference": {"ref":[30,h'…'],"export":"eval"},
     "decode":    {"ref":[30,h'…'],"export":"decode"} },
  "native": [                            ; optional accelerated implementations
     {"target":"x86_64-linux-gnu","abi":"omni-plugin-c/1",
      "artifact":[0,h'…'],"signature":[18,h'…']} ],
  "docs": [0,h'…'], "tests": [0,h'…'],
  "license":"Apache-2.0", "homepage":"…" }
```

Plugins may be **embedded in the container**, so a model is self-contained.
`omni inspect` lists embedded plugins and whether the local runtime trusts them.

## 11.6 WebAssembly as portable semantics

Reference implementations of plugin behaviour are WebAssembly modules, executed
under a restricted profile:

| Constraint | Value |
|---|---|
| WASM version | Core 2.0 |
| Proposals allowed | multi-value, bulk-memory, sign-ext, reference types, SIMD (fixed-width, deterministic subset) |
| Proposals forbidden | threads, relaxed-SIMD (nondeterministic), exceptions with host interaction, GC (initially) |
| Imports | **none**, except the `omni_plugin/1` host ABI below |
| Host ABI | `alloc`, `dealloc`, `log`, `abort`, `read_object(digest_ptr) -> handle` (read-only, sandboxed to declared refs) |
| Determinism | required; float ops use the deterministic subset (no `relaxed_*`, no NaN-payload dependence) |
| Resource limits | fuel-metered; memory capped (default 256 MiB); wall-clock capped |
| Filesystem / network / clock / randomness | **unavailable** |

Why WASM and not "a C ABI" or "a Python function":

1. It is a **frozen, formally specified** instruction set with a mechanized
   semantics — the strongest 50-year portability guarantee available today.
2. It is **safe by construction**: a malicious plugin in a downloaded model can
   corrupt nothing outside its own linear memory, and cannot exfiltrate.
3. It is **deterministic** under the profile above, so a reference
   implementation is a genuine oracle for conformance testing.
4. It is **already** implementable in a few thousand lines (interpreter) or
   available from a dozen mature engines.

The cost is speed — a WASM reference kernel is 10–100× slower than a native one.
That is acceptable because WASM's role is *definition and fallback*, not
production execution. Production uses native code selected by the runtime; the
WASM module is what lets a runtime that lacks native code still be correct, and
what lets a validator check that the native code agrees.

## 11.7 The registry

A registry is a coordination mechanism, not a gatekeeper. OMNI's is modelled on
IANA's "Specification Required / Expert Review" tiers, and on PNG's
`iTXt`-style permanent openness.

| Space | Policy |
|---|---|
| `otype 0x0000–0x00FF` | Standards Action (spec revision) |
| `otype 0x0100–0x7FFF` | Expert Review |
| `otype 0x8000–0xFFFF` | First Come First Served (namespaced by plugin) |
| dtype aliases | Expert Review + a conformance vector set |
| `omni.*` dialect ops | Expert Review + reference WASM + tests |
| third-party dialects | registration optional; *discovery* only |
| codecs | Specification Required (a stable, documented bitstream) |
| feature URIs | First Come First Served |

Registration requirements for anything in `omni.*`: a written specification, a
WASM reference implementation, conformance vectors, and two independent
implementations. Registration for third-party namespaces requires only a name and
a contact — the registry there is a phone book, not a standards body.

**Anti-capture provisions**, because a format that becomes a standard becomes a
political object:

1. No registration may be *denied* for a non-`omni.*` namespace.
2. `x.*` exists and needs no permission, ever.
3. The core (`omni.core` dialect, container framing, object model) is frozen for
   the life of 1.x, so the registry cannot be used to force behavioural changes.
4. Registry data is itself distributed as a signed OMNI container, mirrorable by
   anyone, with no single point of failure.

## 11.8 Worked example: a new architecture in 2032

Suppose "Hyperion blocks" replace attention in 2032.

1. Someone publishes plugin `org.hyperion/nn@1` defining `hyperion.block` with a
   shape function, a verifier, a WASM reference implementation, and a `lower_to`
   rule expressed in `omni.tensor` primitives.
2. Models ship with the plugin embedded (a few hundred KB) and
   `features.required = ["org.hyperion/nn.1"]`.
3. A 2026-vintage runtime: reads the container fine; verifies signatures fine;
   `omni inspect` prints the full metadata, tensor list and parameter count;
   `resolve` finds it does not support the dialect, **falls back to the shipped
   lowering**, and executes the model at primitive level — slowly, but correctly.
4. A 2032 runtime with native Hyperion kernels selects the semantic level and
   runs fast.
5. Nothing about the specification changed. No new file extension, no
   "OMNI v2", no migration.

Step 3 is the property no existing format has. GGUF would report an unknown
architecture enum; ONNX would fail on an unknown op with no recourse;
safetensors would load the weights and leave the user to guess.

---

**Prev:** [§10 Runtime](10-runtime.md) · **Next:** [§12 Security Model](12-security.md)
