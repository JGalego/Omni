# Worked Examples

Everything on this page was **produced by the reference implementation** and can
be reproduced byte for byte:

```console
$ cd reference && cargo build --release
$ cd ../examples && ../reference/target/release/omni example toy.omni
```

`toy.omni` in this directory is a real, complete, valid OMNI container. It is
113 856 bytes and its SHA-256 is stable across machines and runs, because
packing is deterministic (§01.10, writer rule W1).

> The reference implementation uses **SHA-256** rather than the default
> BLAKE3-256 so that every digest here can be checked with `sha256sum` and the
> crate can stay dependency-free. Both algorithms are mandatory in §03.5.1.

---

## 1 The example model

A toy 2-layer decoder: 12 tensors, 57 472 parameters, bf16 attention
projections, f32 norms, and — deliberately — **`lm_head.weight` tied to
`model.embed_tokens.weight`**, which is the single most common weight-sharing
pattern in real decoder models.

```console
$ omni example toy.omni
wrote toy.omni
  size           111.19 KiB
  root           sha2:8e02117540106e18…
  objects        49
  reachable      49
  verified       49 objects, 86.54 KiB
  reproducible   ✓ (two packs byte-identical)
```

## 2 `omni inspect` — the whole model, no tensor payload

```console
$ omni inspect toy.omni
toy.omni                           111.19 KiB
  container   OMNI/1.0  profile=core  align=4096  hash=sha2-256  sealed
  creator     omni-rs/0.1.0
  uuid        586dd8a07b377bee8041aba8906c4557
  root        sha2:8e02117540106e18…

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

Three things in that output are the specification's claims made concrete:

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
   16  58 6d d8 a0 7b 37 7b ee 80 41 ab a8 90 6c 45 57     file_uuid (UUIDv7-shaped, derived)
   32  12                                                  hash_algo (0x12 = sha2-256)
   33  00                                                  profile (0 = core)
   34  20                                                  digest_len
   35  00                                                  reserved0
   36  03 00 00 00                                         flags
   40  a0 00 00 00 00 00 00 00                             front_sb_off
   48  51 01 00 00 00 00 00 00                             front_sb_len
   56  c0 bc 01 00 00 00 00 00                             file_size
   64  8e 02 11 75 40 10 6e 18 a8 f5 b9 8d 01 c2 8d 01 …   root_digest
   96  6f 6d 6e 69 2d 72 73 2f 30 2e 31 2e 30 00 00 00     creator
  112  00 00 00 00 00 00 00 00                             created_unix_ms
  120  00 00 00 00                                         reserved1
  124  95 10 10 b2                                         header_crc32c

raw:
00000000  89 4f 4d 4e 49 0d 0a 1a 01 00 00 00 01 0c 80 00  |.OMNI...........|
00000010  58 6d d8 a0 7b 37 7b ee 80 41 ab a8 90 6c 45 57  |Xm..{7{..A...lEW|
00000020  12 00 20 00 03 00 00 00 a0 00 00 00 00 00 00 00  |.. .............|
00000030  51 01 00 00 00 00 00 00 c0 bc 01 00 00 00 00 00  |Q...............|
00000040  8e 02 11 75 40 10 6e 18 a8 f5 b9 8d 01 c2 8d 01  |...u@.n.........|
00000050  c5 36 40 ff a6 f5 52 03 5c 8f 71 f0 3b 80 1d 46  |.6@...R.\.q.;..F|
00000060  6f 6d 6e 69 2d 72 73 2f 30 2e 31 2e 30 00 00 00  |omni-rs/0.1.0...|
00000070  00 00 00 00 00 00 00 00 00 00 00 00 95 10 10 b2  |................|
```

`created_unix_ms` is zero and the UUID is derived from the root digest, so the
build is reproducible (§01.10).

## 5 Independent verification

The root digest in the header is genuinely the SHA-256 of the manifest object at
the offset the index reports — checkable with any SHA-256 implementation:

```console
$ python3 -c "
import hashlib
d = open('toy.omni','rb').read()
print('manifest at 4256, 211 bytes:', hashlib.sha256(d[4256:4256+211]).hexdigest())
print('root_digest in header      :', d[64:96].hex())"
manifest at 4256, 211 bytes: 8e02117540106e18a8f5b98d01c28d01c53640ffa6f552035c8f71f03b801d46
root_digest in header      : 8e02117540106e18a8f5b98d01c28d01c53640ffa6f552035c8f71f03b801d46
```

## 6 Objects

```console
$ omni ls toy.omni
DIGEST               TYPE                   OFFSET        BYTES  FLAGS
8e02117540106e18a8   Manifest                 4256          211  critical,safe-to-copy
cbd309456924112801   Metadata                 4472          269  critical,safe-to-copy
b80ef4aa2725a41f76   Model                    4744           66  critical,safe-to-copy
79a283a39da3d3234a   TensorTable              4816         1228  critical,safe-to-copy
3ed8c98d646a7ece49   TensorDesc               6048          228  critical,safe-to-copy
…
```

## 7 Structure objects in CBOR diagnostic notation

### Manifest (211 bytes)

```console
$ omni dump toy.omni --object 8e0211
```
```cbor-diag
{"t": "omni.core/manifest", "v": 1, "kind": "model",
 "meta": [2, h'cbd3094569241128017692517b88283e951ab96cc6e7636284a6884e4e2b196c'],
 "entry": "model",
 "assets": {"model": [3, h'b80ef4aa2725a41f762c554dded845e883757be0bbedbe4abff82e3739cf2d8d']},
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

### TensorDesc (228 bytes)

```cbor-diag
{"t": "omni.tensor/desc", "v": 1,
 "axes": ["out_features", "in_features"],
 "dtype": {"e": 8, "k": "float", "m": 7, "w": 16, "alias": "bf16"},
 "shape": [32, 64],
 "value": {"op": "literal",
           "chunks": [6, h'c596356305a152261b716d2ca4c0ee2dacbe52c7e2cf6a2e02e2403145812376']},
 "layout": {"k": "strided", "order": "row-major"},
 "semantic": "weight",
 "materialize": "lazy"}
```

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
;  "shape": [64], "value": {"op": "literal", "chunks": [6, h'd5078e2e…']},
;  "layout": {"k": "strided", "order": "row-major"}, "semantic": "scale",
;  "materialize": "lazy"}
; chunk sha2:c16d36af535539a2… @ file offset 77824
00013000  8b af 94 67 72 b7 57 06 45 d1 a1 1b 25 a9 54 8d  |...gr.W.E...%.T.|
00013010  62 b8 4e 29 c0 99 b3 e2 04 0d 58 c7 7a 9c 9d c7  |b.N)......X.z...|
00013020  8a 09 72 9e 92 06 5e 78 8b e6 aa 18 4b 3b 1b ab  |..r...^x....K;..|
00013030  53 95 e0 7f 7b d4 8e 51 fd c9 55 ea a2 0e d7 30  |S...{..Q..U....0|
… 192 more bytes
```

File offset `0x13000` = 77 824, which is `19 × 4096` — page-aligned, so this
tensor can be handed to a consumer directly from an `mmap` with no copy (R-C08).

## 9 What these examples do *not* show

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
