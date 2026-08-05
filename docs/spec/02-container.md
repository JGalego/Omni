# OMNI/1.0 — §2 Container Binary Format

Layer L0. This is the only part of OMNI that is expensive to change, so it is
deliberately small, boring, and over-specified.

All multi-byte integers are **little-endian**. `u8/u16/u32/u64` denote unsigned
integers of that width. Offsets are absolute from the start of the file unless
stated otherwise.

## 2.1 File shape

```
┌────────────────────────────────────────────┐  offset 0
│ FileHeader                      128 bytes  │
├────────────────────────────────────────────┤
│ Superblock segment  (optional, "front")    │   ← present in `sealed` profile
├────────────────────────────────────────────┤
│ Segment 1  (OBJ:  structure objects)       │
│ Segment 2  (BLOB: tensor data chunks)      │
│ …                                          │
│ Segment n  (INDEX: object index)           │
├────────────────────────────────────────────┤
│ Superblock segment  (authoritative, "back")│
├────────────────────────────────────────────┤
│ FileTrailer                      64 bytes  │  ← last 64 bytes of the file
└────────────────────────────────────────────┘  offset = file_size
```

Both access patterns are first-class:

- **Seek-capable readers** (local file, S3, HTTP with ranges) read the last 64
  bytes, follow one pointer to the superblock, and one more to the index. Two
  round trips to a fully usable model, regardless of size.
- **Pure-forward readers** (a socket, a tape, `curl | omni`) rely on the *front*
  superblock when present, and otherwise on the fact that segments are
  self-framing and objects arrive in topological order (§13.2), so work can start
  before the file ends.

> **Design note.** ZIP, Parquet and ORC all put the directory at the end because
> writers stream. Content-addressed writers must hash everything before they can
> name it, so OMNI can *also* emit a front superblock cheaply. We do both, and
> require them to be identical, so neither reader class is a second-class citizen.

## 2.2 Container profiles

`FileHeader.profile` selects a profile. Profiles restrict, never extend.

| Value | Profile | Constraints |
|---|---|---|
| `0` | `core` | Sealed, indexed, front+back superblock. The default. |
| `1` | `stream` | Front superblock MUST be present; objects in topological order; index MAY be partial. Optimized for progressive load. |
| `2` | `append` | Append-log: superblock repeated after each flush; the last valid one wins; recovery by segment scan. For live training checkpoints. |
| `3` | `archive` (**OMNI-A**) | §14.8: `raw`/`deflate` codecs only, no external refs, no runtime caches, `Rosetta` object required, all metadata UTF-8. |
| `4` | `cache` | Contains only derived objects; MUST NOT be the sole source of a model. |

## 2.3 FileHeader (128 bytes, offset 0)

| Off | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 8 | `magic` | `89 4F 4D 4E 49 0D 0A 1A` = `\x89 O M N I \r \n \x1a` |
| 8 | 2 | `container_major` | `u16` — `1` for OMNI/1.x |
| 10 | 2 | `container_minor` | `u16` — `0` |
| 12 | 1 | `byte_order` | `0x01` = little-endian. Only value defined in 1.x. |
| 13 | 1 | `log2_align` | payload alignment exponent; `6..=30`; default `12` (4 KiB) |
| 14 | 2 | `header_size` | `u16` = `128`; readers MUST skip to `header_size` |
| 16 | 16 | `file_uuid` | RFC 9562 UUIDv7, or derived (§01.10) when reproducible |
| 32 | 1 | `hash_algo` | primary digest algorithm code (§01.3), e.g. `0x1e` |
| 33 | 1 | `profile` | §2.2 |
| 34 | 1 | `digest_len` | primary digest length in bytes (32 for BLAKE3-256) |
| 35 | 1 | `reserved0` | MUST be `0` |
| 36 | 4 | `flags` | §2.3.1 |
| 40 | 8 | `front_sb_off` | offset of front superblock segment; `0` if absent |
| 48 | 8 | `front_sb_len` | length of front superblock payload; `0` if absent |
| 56 | 8 | `file_size` | total file size, or `0` if unknown at write time |
| 64 | 32 | `root_digest` | digest of root `Manifest`, truncated/padded to 32 B; `0` if unknown |
| 96 | 16 | `creator` | UTF-8, NUL-padded, e.g. `omni-rs/1.0.0\0\0\0` |
| 112 | 8 | `created_unix_ms` | informational; `0` under `--reproducible` |
| 120 | 4 | `reserved1` | MUST be `0` |
| 124 | 4 | `header_crc32c` | CRC-32C (Castagnoli) over bytes `0..124` |

### 2.3.1 Header flags

| Bit | Name | Meaning |
|---:|---|---|
| 0 | `SEALED` | file is complete; `file_size` and trailer are valid |
| 1 | `FRONT_SB` | front superblock present |
| 2 | `APPEND_LOG` | multiple superblocks; last valid wins |
| 3 | `SIGNED` | at least one `Signature` object is reachable from root |
| 4 | `ENCRYPTED` | at least one object payload is AEAD-wrapped (§12.8) |
| 5 | `PARTIAL` | container knowingly omits some reachable objects (lazy/partial fetch) |
| 6 | `DERIVED_ONLY` | contains only `cacheable` objects |
| 7 | `NO_MMAP_SAFE` | alignment guarantees relaxed (e.g. produced by a streaming writer over a non-seekable sink) |
| 8–31 | reserved | MUST be `0` and MUST be ignored by readers |

### 2.3.2 Why these bytes

- **The magic.** Copied from PNG's structure for the same reasons: byte 0 has
  the high bit set (fails 7-bit-clean transfer), `\r\n` detects CRLF↔LF mangling,
  `\x1a` stops MS-DOS-lineage `TYPE`/`more` from dumping binary. It also makes
  `file(1)` detection trivial and unambiguous, and it is not a valid UTF-8
  prefix, so no tool will mistake a container for text.
- **`log2_align` in the header, not the superblock.** A reader must be able to
  compute alignment before parsing anything, including in a recovery scan.
- **`root_digest` duplicated in the header.** A ranged HTTP client that has only
  the first 128 bytes can already name the model, check a local cache, and
  potentially skip the download entirely.
- **CRC-32C, not a cryptographic hash, in the header.** The header describes
  where trust begins; it is not itself trusted. CRC catches truncation and bit
  rot; §12 handles adversaries. Castagnoli has hardware support on x86-64 and
  AArch64.

## 2.4 Segments

Everything after the header is a sequence of segments. A segment is
self-framing so that a damaged container can be recovered by scanning.

### 2.4.1 SegmentHeader (32 bytes, aligned to 8)

| Off | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | `seg_magic` | `'O' 'S' 'E' 'G'` = `4F 53 45 47` |
| 4 | 2 | `kind` | §2.4.2 |
| 6 | 2 | `seg_flags` | bit0 `PADDED`, bit1 `LAST`, bit2 `CODEC_APPLIED` |
| 8 | 8 | `payload_len` | bytes of payload, excluding padding |
| 16 | 8 | `seq` | monotonically increasing per file, from 0 |
| 24 | 4 | `payload_crc32c` | CRC-32C of payload bytes (integrity, not security) |
| 28 | 4 | `header_crc32c` | CRC-32C over bytes `0..28` |

Payload begins immediately after the header and is followed by zero padding to
the next `align` boundary when `PADDED` is set. Segment headers are 8-byte
aligned; segment *payloads* of kind `BLOB` start at an `align` boundary (§2.9).

### 2.4.2 Segment kinds

| Kind | Name | Payload |
|---:|---|---|
| 1 | `BLOB` | concatenated data-object payloads, each aligned |
| 2 | `OBJ` | concatenated structure-object payloads (8-byte aligned) |
| 3 | `INDEX` | an `ObjectIndex` (§2.6) |
| 4 | `SUPER` | a `Superblock` (§2.5) |
| 5 | `SIG` | detached signatures over preceding segments (§12.5.4) |
| 6 | `PAD` | explicit padding (used to reach a device-friendly boundary) |
| 7 | `FOREIGN` | verbatim bytes of an imported source artifact (§import) |
| 8 | `TOMB` | tombstone: marks preceding objects as superseded (append profile) |

Unknown segment kinds MUST be skipped using `payload_len`. This is the coarsest
forward-compatibility mechanism in the format, and it costs nothing.

## 2.5 Superblock

The superblock is a structure object (canonical OMNI-CBOR) carried in a `SUPER`
segment. It is the authoritative description of the container.

```cbor-diag
{
  "t": "omni.core/superblock", "v": 1,

  "roots": [ [1, h'…'] ],              ; root Manifest refs, primary first
  "index": { "off": 4294967296, "len": 8388608,
             "digest": h'…', "entries": 131072, "fmt": 1 },
  "name_index": { "off": …, "len": …, "digest": h'…' },   ; optional (§2.6.4)

  "segments": [                        ; segment directory (offset, len, kind)
     [128, 4096, 4], [4224, 268435456, 2], …
  ],

  "hash": "blake3-256",
  "codecs": [ {"id":"raw"},
              {"id":"zstd","level":3,"long":0,"ver":"1.5"} ],
  "align": 4096,

  "features": { "required": [...], "optional": [...] },
  "profile": "core",
  "stats": { "objects": 131072, "blobs": 130000, "bytes_logical": 1.6e11,
             "bytes_stored": 9.1e10, "params": 8030261248 },
  "prev": { "off": …, "digest": h'…' }   ; append profile: previous superblock
}
```

Rules:

- If both front and back superblocks exist, they MUST be byte-identical, and a
  reader MUST verify this when it has cheap access to both. If they differ, the
  **back** superblock wins and the file MUST be reported as `suspect`.
- `segments[]` lets a reader plan I/O (e.g. `madvise(WILLNEED)` over the BLOB
  extents) without scanning.
- `stats` is advisory; validators recompute.

## 2.6 Object index

The object index is the one structure in OMNI that is **not** CBOR. It is a
fixed-layout, cacheline-aligned array designed to be used *directly from an
`mmap` with zero parsing*.

### 2.6.1 Index header (64 bytes)

| Off | Size | Field |
|---:|---:|---|
| 0 | 4 | `'O''I''D''X'` |
| 4 | 2 | `fmt_version` (`1`) |
| 6 | 2 | `entry_size` (`64`) |
| 8 | 8 | `entry_count` |
| 16 | 8 | `bucket_table_off` (relative to index start; `0` if absent) |
| 24 | 4 | `bucket_bits` (0, 8, 16, 20 or 24) |
| 28 | 1 | `hash_algo` |
| 29 | 1 | `digest_len` |
| 30 | 2 | `flags` (bit0 `SORTED`, bit1 `COMPLETE`, bit2 `HAS_AUX`) |
| 32 | 8 | `aux_table_off` |
| 40 | 8 | `aux_table_len` |
| 48 | 12 | reserved (zero) |
| 60 | 4 | `crc32c` of the header |

### 2.6.2 Index entry (64 bytes, sorted ascending by `digest`)

| Off | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 32 | `digest32` | first 32 bytes of the object digest |
| 32 | 8 | `offset` | absolute file offset of stored bytes; `0` = not local |
| 40 | 8 | `stored_len` | bytes on disk (after codec/AEAD) |
| 48 | 8 | `logical_len` | bytes after decoding — **allocation bound** |
| 56 | 2 | `otype` | §01.6 |
| 58 | 1 | `codec` | index into `superblock.codecs`; `0xFF` → see aux |
| 59 | 1 | `oflags` | bit0 `CRITICAL`, bit1 `CACHEABLE`, bit2 `EXTERNAL`, bit3 `LOSSY`, bit4 `ENCRYPTED`, bit5 `HAS_BAO` |
| 60 | 4 | `aux` | index into the aux table, or `0xFFFFFFFF` |

Exactly 64 bytes: one cache line, one entry, no false sharing during a parallel
binary search.

**Lookup cost.** With a 16-bit bucket table (65 536 × `u32` = 256 KiB), locating
an object is one bucket read followed by a search among the entries sharing that
digest prefix. For a 1 M-object store that is ~15 entries, which at a 64-byte
stride is ~15 cache lines — so a *scan* of them, which the hardware prefetcher
handles, beats a binary search's random probes. Zero syscalls, zero allocation,
zero parsing. The index for that store is 61 MiB; it need not be resident, since
only touched pages fault in.

**Measured, not modelled.** `omni bench` reports p50 ≈ 200 ns and p99 ≈ 590 ns
at 10⁶ objects on a cloud VM. That is above the roadmap's Gate 0 target of
500 ns p99, and the gap is discussed in
[`docs/design/performance.md`](../design/performance.md) §11 rather than
rounded away here.

`bucket_bits` MAY be 0, 8, 16, 20 or 24. Wider is not automatically better: a
20-bit table gives about one entry per bucket, but its 4 MiB no longer fits in
L2, so the bucket read becomes a cache miss of its own and the adjacency of a
16-bit bucket's entries is lost. Measured, 20 bits was *slower* than 16.

### 2.6.3 Aux table

Variable-length records for the minority of objects needing more: extended codec
parameters, external locators, AEAD nonces/key ids, Bao tree refs. Encoded as
canonical OMNI-CBOR, one array indexed by `aux`. Keeping this out of the hot
array is what preserves the fixed 64-byte stride.

### 2.6.4 NameIndex (optional, derived)

A parallel fixed-layout table mapping FNV-1a-64 of a tensor/asset name to
`(digest32, otype)` for `omni inspect` and name-based random access without
parsing the tensor table. Purely derived; droppable; verified by recomputation.

### 2.6.5 Partial and remote indexes

- `flags.COMPLETE = 0` means the index does not list every reachable object;
  misses fall through to the next store in the chain.
- `offset = 0` with `EXTERNAL` set means "this object exists but is not in this
  file"; the aux record carries locator hints. This is how a 3 GB "index-only"
  container can describe a 700 GB model whose weights live in S3.
- The index MAY be shipped as a detached `.omni.idx` sidecar, byte-identical to
  the in-file `INDEX` segment payload, for CDNs that do not handle suffix ranges
  well.

## 2.7 FileTrailer (64 bytes, at `file_size - 64`)

| Off | Size | Field |
|---:|---:|---|
| 0 | 8 | `superblock_off` |
| 8 | 8 | `superblock_len` |
| 16 | 32 | `superblock_digest` (primary algorithm, truncated to 32) |
| 48 | 4 | `flags` (mirrors header flags at seal time) |
| 52 | 4 | `crc32c` over bytes `0..52` |
| 56 | 8 | `magic_end` = `1A 0A 0D 49 4E 4D 4F 89` (reversed magic) |

Opening a sealed container:

```
buf   = read(file_size - 64, 64)          # 1 read
check buf[56..64] == MAGIC_END, crc32c
sb    = read(superblock_off, superblock_len)   # 2nd read
verify H(sb) == superblock_digest
idx   = mmap(sb.index.off, sb.index.len)       # no read; page-faulted lazily
```

Over HTTP that is `Range: bytes=-64` then one more range request. **A 700 GB model
becomes queryable in two round trips and ~10 KiB of transfer.**

## 2.8 Recovery

A container whose trailer or superblock is damaged is recovered by scanning for
`OSEG` on 8-byte boundaries, validating `header_crc32c`, and rebuilding an index
from the object payloads (structure objects self-identify by their `t` key; blob
objects are identified by re-hashing and matching refs found in structure
objects). `omni fsck --rebuild` implements this. Because every object is
content-addressed, recovery cannot silently produce a *wrong* model: any
mis-assembled object fails its digest check.

## 2.9 Alignment

Let `A = 1 << log2_align` (default 4096).

1. Every `BLOB` segment payload starts at a multiple of `A`.
2. Every data object *within* a `BLOB` segment starts at a multiple of `A`.
3. Structure objects are 8-byte aligned; segment headers 8-byte aligned.
4. The `INDEX` segment payload starts at a multiple of `max(A, 64)`.
5. All padding bytes MUST be `0x00`. (Non-zero padding is a covert channel and a
   reproducibility hazard; validators check this.)
6. A writer MAY request larger alignment for specific objects via the aux table
   (`align_hint`), e.g. 2 MiB for transparent-hugepage-friendly weight blocks or
   4 KiB×N for O_DIRECT / GPUDirect Storage. Readers MUST NOT assume more than
   `A` unless the hint is present.

**Why 4 KiB default.** It is the page size everywhere that matters, the minimum
`O_DIRECT` granularity on Linux, the NVMe logical block size, and small enough
that padding waste on a model with 10⁵ objects is ~200 MiB worst case for 10⁵
objects — and near zero in practice because tensor chunks are megabytes. A 64 KiB
alignment is recommended (`log2_align = 16`) for object stores where the minimum
billable read is larger.

**Zero-copy requirement.** For any tensor whose dtype has a natural element
alignment ≤ `A` and whose layout is contiguous, a C0 reader MUST be able to hand
the mapped pointer directly to a consumer with no copy and no realignment. This
is the hard constraint that forced alignment into the container layer rather than
leaving it to writers' discretion — it is the single thing safetensors got most
right and PyTorch checkpoints got most wrong.

## 2.10 Size limits and defensive bounds

| Quantity | Limit | Rationale |
|---|---:|---|
| `header_size` | 128..4096 | future headers may grow within one page |
| segment `payload_len` | < 2⁶³ | signed-arithmetic safety in every language |
| structure object payload | ≤ 64 MiB | parse in one pass without streaming CBOR |
| index `entry_count` | ≤ 2³² | 256 GiB index ceiling; shard beyond |
| CBOR nesting depth | ≤ 64 | stack-overflow protection (§12.4) |
| refs per object | ≤ 2²⁴ | use `ShardedMap` beyond |
| decompression ratio | ≤ 1000:1 unless declared | zip-bomb protection |

A reader MUST reject a file violating a MUST-limit rather than attempting
best-effort parsing. Every length field MUST be validated against the actual file
size before it is used for allocation or indexing.

## 2.11 Worked byte layout

The bytes below are **generated by the reference implementation**
(`examples/toy.omni`, 113 856 bytes, 12 tensors), not hand-written:

```
00000000  89 4f 4d 4e 49 0d 0a 1a  01 00 00 00 01 0c 80 00  |.OMNI...........|
          └─ magic ─────────────┘  maj   min   bo al  hsz
00000010  cd 5a 1f e6 73 22 7c 0f  9d 73 9d 31 23 dd 0c cb  |.Z..s"|..s.1#...|
          └─ file_uuid (derived from root_digest; reproducible) ───────────┘
00000020  1e 00 20 00 03 00 00 00  a0 00 00 00 00 00 00 00  |.. .............|
          alg prf dl rs  └flags┘   └─ front_sb_off = 0xa0 ─┘
00000030  53 01 00 00 00 00 00 00  c0 bc 01 00 00 00 00 00  |S...............|
          └ front_sb_len = 339 ─┘  └─ file_size = 113856 ──┘
00000040  ef 5e 49 b8 b0 c2 fa a7  54 5b d6 0a da 41 37 30  |.^I.....T[...A70|
00000050  fc c1 47 4f a7 3b e4 a5  b9 3f 5e 58 80 56 c8 d3  |..GO.;...?^X.V..|
          └─ root_digest: blake3-256 of the Manifest object ──────────────┘
00000060  6f 6d 6e 69 2d 72 73 2f  30 2e 31 2e 30 00 00 00  |omni-rs/0.1.0...|
00000070  00 00 00 00 00 00 00 00  00 00 00 00 88 3e 26 92  |.............>&.|
          └ created = 0 (repro) ┘  └reserved┘  └─ crc32c ─┘
```

and the segment chain it describes:

```
offset      segment  payload
0x00000080  SUPER      339 B   front superblock
0x00001080  OBJ       6294 B   27 structure objects, 8-byte aligned
0x00002938  PAD       1672 B   so the next payload lands on a 4 KiB boundary
0x00002fe0  BLOB     90112 B   22 data chunks, each 4096-aligned
0x00019000  PAD       4032 B
0x00019fe0  INDEX     3200 B   64-byte header + 49 × 64-byte entries
0x0001ac80  SUPER      339 B   back superblock, byte-identical to the front
0x0001bd80  (trailer)   64 B
```

`flags = 0x03` is `SEALED | FRONT_SB`. `front_sb_off = 0xa0` = `0x80 + 32`, the
superblock segment's header plus its 32-byte frame. `alg = 0x1e` is BLAKE3-256,
the default of §03.5.1; the same file built with `--hash sha256` differs only in
that byte, the digests, and the digest-ordered placement of the data objects.

See [`examples/README.md`](../../examples/README.md) for the CBOR diagnostic
rendering of every object, an independent digest check, and the full `omni
verify` ladder.

## 2.12 Rejected container designs

| Alternative | Why rejected |
|---|---|
| **Tar-based (like OCI layers)** | No random access, no alignment, 512-byte headers interleaved with data, no index. Every consumer ends up extracting to disk first. |
| **ZIP-based (like `.mlpackage`, `.pt`)** | Directory-at-end is right, but per-entry compression framing, 4 GiB legacy corners, and the absence of alignment guarantees make zero-copy mapping impossible. PyTorch's ZIP checkpoints must copy every tensor. |
| **HDF5** | Genuinely capable (chunking, filters, partial I/O) but a single complex C implementation, a file format that has had silent corruption bugs, no content addressing, and a threading story that has burned every ML framework that tried it. |
| **Flat header + blob (safetensors)** | The right instinct, but a single JSON header means no streaming, no dedup, no extensibility, no signatures, and O(n) header re-parse on every open. |
| **FlatBuffers/Cap'n Proto for everything** | Zero-copy is attractive, but schema evolution is by-field-id only, canonical encoding is not guaranteed (so hashing is unstable), and losing the schema loses the data. We use the *idea* — a fixed-layout, zero-parse table — exactly where it pays (§2.6) and self-describing CBOR everywhere else. |
| **SQLite as container** | Tempting (indexes, transactions, ubiquity) but no alignment control, page-oriented storage defeats `mmap` of large values, and it puts a B-tree in the trust boundary. |

**Prev:** [§01 Object Model](01-object-model.md) · **Next:** [§03 Encoding & Hashing](03-encoding.md)
