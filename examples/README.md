# Worked Examples

Everything on this page was **produced by the reference implementation** and can
be reproduced byte for byte:

```console
$ cd reference && cargo build --release
$ cd ../examples && ../reference/target/release/omni example toy.omni
```

`toy.omni` in this directory is a real, complete, valid OMNI container. It is
113 856 bytes and identical byte for byte across machines and runs, because
packing is deterministic (§01.10, writer rule W1).

It uses **BLAKE3-256**, the default of §03.5.1. Passing `--hash sha256` produces
a container that is byte-identical except for its digests — useful when you want
to check every hash in the file with `sha256sum`:

```console
$ omni example --hash sha256 toy-sha256.omni
```

That the two files differ *only* in their digests is worth stating plainly: the
algorithm is one header field, and §12.11's hash-migration story is "rehash and
rewrite the graph", not "redesign the format".

## 1 The example model

A toy 2-layer decoder: 12 tensors, 57 472 parameters, bf16 attention
projections, f32 norms, and — deliberately — **`lm_head.weight` tied to
`model.embed_tokens.weight`**, which is the single most common weight-sharing
pattern in real decoder models.

```console
$ omni example toy.omni
wrote toy.omni
  size           111.19 KiB
  root           b3:ef5e49b8b0c2faa7…
  objects        49
  reachable      49
  verified       49 objects, 86.54 KiB
  reproducible   ✓ (two packs byte-identical)
```

## 2 `omni inspect` — the whole model, no tensor payload

```console
$ omni inspect toy.omni
toy.omni                           111.19 KiB
  container   OMNI/1.0  profile=core  align=4096  hash=blake3-256  sealed
  creator     omni-rs/0.1.0
  uuid        cd5a1fe673227c0f9d739d3123dd0ccb
  root        b3:ef5e49b8b0c2faa7…

manifest    kind=model
model  omni/example-toy
  architecture  transformer.decoder
  params        rope {"dims": 16, "kind": "rope", "theta": 10000.0, "interleaved": false}
                n_heads 4  n_layers 2  activation "silu"  ffn_hidden 128
                n_kv_heads 2  hidden_size 64
  parameters    57,472
  license       Apache-2.0
  features      required: omni.core/1.0, omni.tensor/expr.1

tensors       12                       112.50 KiB
  lm_head.weight                               [256,64]       bf16      32.00 KiB
  model.embed_tokens.weight                    [256,64]       bf16      32.00 KiB
  model.layers.0.attn.o_proj.weight            [64,64]        bf16       8.00 KiB
  model.layers.0.attn.q_proj.weight            [64,64]        bf16       8.00 KiB
  model.layers.1.attn.o_proj.weight            [64,64]        bf16       8.00 KiB
  model.layers.1.attn.q_proj.weight            [64,64]        bf16       8.00 KiB
  model.layers.0.attn.k_proj.weight            [32,64]        bf16       4.00 KiB
  model.layers.0.attn.v_proj.weight            [32,64]        bf16       4.00 KiB
  … 4 more (use `omni ls`)
  dedup         112.50 KiB logical → 80.50 KiB stored (28.4% saved, shared chunk objects)

graph         none (weights-only)
tokenizer     (not present)
adapters      none
signatures    none

objects       49 in index (27 structure, 22 blob)
storage       86.54 KiB logical in objects · 24.65 KiB container overhead

read          header 128 B + trailer 64 B + superblock 337 B + index 3200 B + structure 6187 B
              = 9.68 KiB total, 0 tensor payload bytes
```

Several of the specification's claims are made concrete in that output:

1. **`dedup 112.50 KiB → 80.50 KiB (28.4 % saved)`.** `lm_head` and
   `embed_tokens` have *different* `TensorDesc` objects (different `semantic`),
   but their values resolve to the same `ChunkList`, so the 32 KiB payload is
   stored once. safetensors and GGUF both store it twice. This is §01 axiom A2
   doing real work with no special-casing anywhere.
2. **`0 tensor payload bytes`.** The entire summary came from 9.68 KiB. §06.12's
   discoverability guarantee is structural, not aspirational.
3. **`(not present)` for the tokenizer**, not a fabricated default. Importer
   rule I1.

## 3 `omni verify` — the validation ladder

```console
$ omni verify toy.omni
V0 framing     ✓ 7 segments
     0x00000080  SUPER           337 B
     0x00001080  OBJ            6294 B
     0x00002938  PAD            1672 B
     0x00002fe0  BLOB          90112 B
     0x00019000  PAD            4032 B
     0x00019fe0  INDEX          3200 B
     0x0001ac80  SUPER           337 B
V0 padding     ✓ (R-C07 zero fill)
V0 alignment   ✓ (R-C08 data objects on 4096-byte boundaries)
V1 index       ✓ 49 entries, sorted, complete
V2 structure   ✓ canonical CBOR, schemas present on 27 objects
V3 integrity   ✓ 49 objects, 86.54 KiB verified (R-O01)
V4 graph       ✓ 49 objects reachable from root

valid
```

Note the two `PAD` segments. Aligning data objects to 4 KiB in a file this small
costs 5.7 KiB of padding — the honest cost of `mmap`-ability, and the reason
§02.9 makes alignment configurable. On a real model, where chunks are megabytes,
this overhead disappears into the noise (§performance.7).

## 4 The file header, annotated

```console
$ omni dump toy.omni --header
OMNI FileHeader (§02.3)

    0  89 4f 4d 4e 49 0d 0a 1a                             magic  \x89 O M N I \r \n \x1a
    8  01 00                                               container_major
   10  00 00                                               container_minor
   12  01                                                  byte_order (01 = little)
   13  0c                                                  log2_align
   14  80 00                                               header_size
   16  cd 5a 1f e6 73 22 7c 0f 9d 73 9d 31 23 dd 0c cb     file_uuid (UUIDv7-shaped, derived)
   32  1e                                                  hash_algo (0x1e = blake3-256)
   33  00                                                  profile (0 = core)
   34  20                                                  digest_len
   35  00                                                  reserved0
   36  03 00 00 00                                         flags
   40  a0 00 00 00 00 00 00 00                             front_sb_off
   48  51 01 00 00 00 00 00 00                             front_sb_len
   56  c0 bc 01 00 00 00 00 00                             file_size
   64  ef 5e 49 b8 b0 c2 fa a7 54 5b d6 0a da 41 37 30 …   root_digest
   96  6f 6d 6e 69 2d 72 73 2f 30 2e 31 2e 30 00 00 00     creator
  112  00 00 00 00 00 00 00 00                             created_unix_ms
  120  00 00 00 00                                         reserved1
  124  f8 42 36 49                                         header_crc32c

raw:
00000000  89 4f 4d 4e 49 0d 0a 1a 01 00 00 00 01 0c 80 00  |.OMNI...........|
00000010  cd 5a 1f e6 73 22 7c 0f 9d 73 9d 31 23 dd 0c cb  |.Z..s"|..s.1#...|
00000020  1e 00 20 00 03 00 00 00 a0 00 00 00 00 00 00 00  |.. .............|
00000030  51 01 00 00 00 00 00 00 c0 bc 01 00 00 00 00 00  |Q...............|
00000040  ef 5e 49 b8 b0 c2 fa a7 54 5b d6 0a da 41 37 30  |.^I.....T[...A70|
00000050  fc c1 47 4f a7 3b e4 a5 b9 3f 5e 58 80 56 c8 d3  |..GO.;...?^X.V..|
00000060  6f 6d 6e 69 2d 72 73 2f 30 2e 31 2e 30 00 00 00  |omni-rs/0.1.0...|
00000070  00 00 00 00 00 00 00 00 00 00 00 00 f8 42 36 49  |.............B6I|
```

`created_unix_ms` is zero and the UUID is derived from the root digest, so the
build is reproducible (§01.10).

## 5 Independent verification

The root digest in the header is genuinely the BLAKE3-256 of the manifest object
at the offset the index reports — checkable against an implementation that has
nothing to do with this repository:

```console
$ pip install blake3   # the upstream implementation, not ours
$ python3 -c "
import blake3
d = open('toy.omni','rb').read()
print('manifest at 4256, 211 bytes:', blake3.blake3(d[4256:4256+211]).hexdigest())
print('root_digest in header      :', d[64:96].hex())"
manifest at 4256, 211 bytes: ef5e49b8b0c2faa7545bd60ada413730fcc1474fa73be4a5b93f5e588056c8d3
root_digest in header      : ef5e49b8b0c2faa7545bd60ada413730fcc1474fa73be4a5b93f5e588056c8d3
```

The `--hash sha256` container is checkable the same way with nothing but the
Python standard library, which is why CI does both.

## 6 Objects

```console
$ omni ls toy.omni
DIGEST               TYPE                   OFFSET        BYTES  FLAGS
ef5e49b8b0c2faa754   Manifest                 4256          211  critical,safe-to-copy
95484fe5f12f3daa08   Metadata                 4472          269  critical,safe-to-copy
9ecac5aba4b4a38f2f   Model                    4744           66  critical,safe-to-copy
020d6b6f0f20a8ed52   TensorTable              4816         1228  critical,safe-to-copy
17a7c710ff924215d5   TensorDesc               6048          220  critical,safe-to-copy
…
```

## 7 Structure objects in CBOR diagnostic notation

### Manifest (211 bytes)

```console
$ omni dump toy.omni --object ef5e49
```
```cbor-diag
{"t": "omni.core/manifest", "v": 1, "kind": "model",
 "meta": [2, h'95484fe5f12f3daa08c51f55383638f11d31c1ab55c27bc8635a04e874aa9237'],
 "entry": "model",
 "assets": {"model": [3, h'9ecac5aba4b4a38f2fa6c0797364e3571814a9e3b89f5e70bb98e0b4b3521fbe']},
 "created": 0,
 "features": {"optional": [], "required": ["omni.core/1.0", "omni.tensor/expr.1"]}}
```

Key order is canonical: sorted by *encoded key bytes* (rule D3), which puts
shorter keys first. That ordering is what makes the digest reproducible across
implementations.

### Metadata (269 bytes)

```cbor-diag
{"t": "omni.meta/model", "v": 1,
 "arch": {"family": "transformer.decoder",
          "params": {"rope": {"dims": 16, "kind": "rope", "theta": 10000.0,
                              "interleaved": false},
                     "n_heads": 4, "n_layers": 2, "activation": "silu",
                     "ffn_hidden": 128, "n_kv_heads": 2, "hidden_size": 64},
          "dialects": [{"v": 1, "ns": "omni.nn"}]},
 "name": "omni/example-toy",
 "license": {"spdx": "Apache-2.0"},
 "params_total": 57472}
```

`rope.interleaved` is present and explicit — §06.3's answer to the most common
silent-corruption bug in format conversion.

### Two TensorDescs, one value

These are `model.embed_tokens.weight` and `lm_head.weight`. Read them side by
side:

```cbor-diag
{"t": "omni.tensor/desc", "v": 1,
 "axes": ["vocab", "hidden"],
 "dtype": {"e": 8, "k": "float", "m": 7, "w": 16, "alias": "bf16"},
 "shape": [256, 64],
 "value": {"op": "literal",
           "chunks": [6, h'656ab9597b092cc71a6aa02cfebb0f306aae9fb1307a7205721ad20a15f1dcad']},
 "layout": {"k": "strided", "order": "row-major"},
 "semantic": "embedding",
 "materialize": "lazy"}

{"t": "omni.tensor/desc", "v": 1,
 "axes": ["vocab", "hidden"],
 "dtype": {"e": 8, "k": "float", "m": 7, "w": 16, "alias": "bf16"},
 "shape": [256, 64],
 "value": {"op": "literal",
           "chunks": [6, h'656ab9597b092cc71a6aa02cfebb0f306aae9fb1307a7205721ad20a15f1dcad']},
 "layout": {"k": "strided", "order": "row-major"},
 "semantic": "weight",
 "materialize": "lazy"}
```

They differ in exactly one field — `semantic` — and point at the same
`ChunkList`. Two names, two descriptors, two roles in the graph, one copy of the
32 KiB payload. Nothing in the writer special-cased tied embeddings; identity is
a hash, so equal values are the same object and that is the end of it.

Note the dtype: `alias: "bf16"` **and** the structural expansion
`{k: float, w: 16, e: 8, m: 7}`. Five extra bytes buy total forward
compatibility — a reader that has never heard of the name `bf16` still knows
exactly what the bits mean (§04.3.6).

`value` is a `literal` node — the simplest case of the tensor expression algebra
(§04.7). A quantized tensor would have `{"op":"dequantize", …}` here, and a
LoRA-merged one `{"op":"add", …}`, with no change to anything else in the file.

## 8 Tensor bytes

```console
$ omni cat toy.omni --tensor model.layers.0.norm.weight --limit 64
; model.layers.0.norm.weight
; {"t": "omni.tensor/desc", "v": 1, "axes": ["hidden"],
;  "dtype": {"e": 8, "k": "float", "m": 23, "w": 32, "alias": "f32"},
;  "shape": [64], "value": {"op": "literal", "chunks": [6, h'105a7bc6…']},
;  "layout": {"k": "strided", "order": "row-major"}, "semantic": "scale",
;  "materialize": "lazy"}
; chunk b3:dbdd7c1947b46931… @ file offset 94208
00017000  8b af 94 67 72 b7 57 06 45 d1 a1 1b 25 a9 54 8d  |...gr.W.E...%.T.|
00017010  62 b8 4e 29 c0 99 b3 e2 04 0d 58 c7 7a 9c 9d c7  |b.N)......X.z...|
00017020  8a 09 72 9e 92 06 5e 78 8b e6 aa 18 4b 3b 1b ab  |..r...^x....K;..|
00017030  53 95 e0 7f 7b d4 8e 51 fd c9 55 ea a2 0e d7 30  |S...{..Q..U....0|
… 192 more bytes
```

The payload bytes are the same as they were before the container switched hash
algorithms — only the chunk's *name* changed. Its file offset moved, from
`19 × 4096` to `23 × 4096`, because blobs are laid out in digest order; it is
still page-aligned, so this tensor can be handed to a consumer directly from an
`mmap` with no copy (R-C08).

## 9 A container and a directory are the same graph

`.omni` is one serialization of the object graph, not the graph itself
(§01 axiom A5). Exploding it into a directory store and packing it back has to
be a no-op, and it is:

```console
$ omni unpack toy.omni -o /tmp/toy.omnid
unpacked toy.omni -> /tmp/toy.omnid
  hash           blake3-256
  root           b3:ef5e49b8b0c2faa7…
  objects        49 written, 0 already present

$ omni pack /tmp/toy.omnid -o /tmp/repacked.omni
$ cmp /tmp/repacked.omni toy.omni && echo identical
identical
```

The directory has no index and no type column — just files named by digest:

```console
$ ls /tmp/toy.omnid
config  objects  root
$ ls /tmp/toy.omnid/objects/ef
5e49b8b0c2faa7545bd60ada413730fcc1474fa73be4a5b93f5e588056c8d3
```

Object types are not stored anywhere in it. They are recovered by walking the
`[otype, digest]` refs from a root whose type is known, which is why a plain
directory of files can round trip through a container that has a typed index.

## 10 Recovering a wrecked container

§02.8 says a container whose trailer, superblock and index are destroyed can be
rebuilt by scanning. Destroying the last 8 KiB of `toy.omni` removes all three:

```console
$ python3 -c "
d = bytearray(open('toy.omni','rb').read())
d[-8192:] = b'\x00' * 8192
open('wrecked.omni','wb').write(bytes(d))"

$ omni verify wrecked.omni
error: R-C09: trailer magic mismatch

$ omni fsck wrecked.omni --rebuild -o repaired.omni
normal open   ✗ R-C09: trailer magic mismatch
header        ✓ OMNI/1.0  hash=blake3-256  align=4096
root          b3:ef5e49b8b0c2faa7…
segment scan  5 segments with valid CRCs
     0x00000080  SUPER         337 B
     0x00001080  OBJ          6294 B
     0x00002938  PAD          1672 B
     0x00002fe0  BLOB        90112 B
     0x00019000  PAD          4032 B
structures    27 decoded from OBJ segments
data objects  22 located by alignment, confirmed by hashing
graph         49 objects reachable from the root

recoverable   ✓ complete

rebuilt       repaired.omni  111.19 KiB
              verifies: 49 objects, 49 reachable

$ cmp repaired.omni toy.omni && echo "byte-identical to the undamaged original"
byte-identical to the undamaged original
```

Two format decisions are doing the work. Structure objects are canonical CBOR,
which is self-delimiting, so decoding one finds the next without an index. Data
objects have no framing at all — but every one starts on a 4 KiB boundary
(R-C08) and the `ChunkList` objects record their lengths, so the search space is
small and every candidate is confirmed by hashing.

That confirmation is the part that matters. Recovery of most formats means
guessing, and a plausible guess is worse than a failure. Here a mis-assembled
object cannot pass: it fails its digest and is reported missing, so a damaged
file recovers to something *incomplete* rather than something *wrong*. Flipping
a byte inside one chunk produces exactly that — 21 objects recovered, one
reported missing, and the rebuilt container honestly carrying a dangling ref.

## 11 What these examples do *not* show

Honesty about scope. The reference implementation is at Phase 0/1 of the
[roadmap](../docs/design/roadmap.md); these examples exercise §01–§04's `literal`
case only. Not demonstrated here, because not yet implemented:

- tensor expression evaluation, quantization, adapters, deltas (§04.7, §05, §08)
- OMNI-IR graphs, dialects, WASM plugins (§07, §11)
- signatures, provenance, encryption (§12)
- compression codecs, HTTP/OCI transport, Bao verified streaming (§03.7, §13)
- capability negotiation and plans (§10)

The parts that *are* demonstrated are the ones that are expensive to change
later: the container framing, the object model, canonical encoding, the index,
alignment, digests, and reproducible packing.
