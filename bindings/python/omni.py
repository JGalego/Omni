#!/usr/bin/env python3
"""A conforming OMNI/1.0 C0 reader, in pure Python with no dependencies.

`docs/design/sdk.md` §5 makes a specific claim about the format: a reader at the
C0 conformance level needs only

    header parse, index lookup, canonical CBOR decode, BLAKE3, literal tensors

and nothing else — no compression, no expression evaluation, no network. The
Rust reference implementation is one piece of evidence for that. It is also the
implementation that wrote the containers, so on its own it cannot distinguish
"the format is simple" from "these two programs share an author's assumptions".

This file is the second piece of evidence. It reads a container written by that
implementation, from the specification, in a language with different arithmetic
and no static types, using nothing outside the standard library — and it
implements BLAKE3 from scratch, because that is the one primitive C0 requires
that Python does not ship.

What it does:

  * §02.7's two-read open: trailer, then superblock, then index.
  * §02.6's index, including the §02.6.1 bucket table.
  * §03's canonical OMNI-CBOR decode, with the D1-D8 checks a *reader* is
    required to apply — a non-canonical encoding is an invalid container, so
    accepting one quietly would make this a different format's reader.
  * §03.5's digests: BLAKE3-256 from scratch, SHA-256 from `hashlib`.
  * §00.4's object graph, walked manifest -> model -> tensor table -> descriptor.
  * §04.5's literal tensors: chunk lists reassembled and the bytes decoded for
    the dense, row-major, non-packed case.

What it deliberately does not do, and reports by name rather than guessing:
compressed objects (§03.7), tensors whose value is an expression rather than a
literal (§04.7), non-dense layouts (§04.4), signature verification (§10), and
anything requiring the network. Every one of those is above C0.

Run it as a script for a summary of a container, or `--check` to verify every
object's digest:

    python3 bindings/python/omni.py examples/toy.omni
    python3 bindings/python/omni.py examples/toy.omni --check
    python3 bindings/python/omni.py examples/toy.omni --tensor model.norm.weight
"""

from __future__ import annotations

import hashlib
import struct
import sys

# --------------------------------------------------------------------- framing --

MAGIC = bytes([0x89, 0x4F, 0x4D, 0x4E, 0x49, 0x0D, 0x0A, 0x1A])
MAGIC_END = bytes([0x1A, 0x0A, 0x0D, 0x49, 0x4E, 0x4D, 0x4F, 0x89])
SEG_MAGIC = b"OSEG"
IDX_MAGIC = b"OIDX"

HEADER_SIZE = 128
TRAILER_SIZE = 64
SEG_HEADER_SIZE = 32
IDX_HEADER_SIZE = 64
IDX_ENTRY_SIZE = 64

HASH_SHA256 = 0x12
HASH_BLAKE3_256 = 0x1E

# §01.3's object type registry, for the numbers the index carries.
OTYPE = {
    0x0001: "Manifest",
    0x0002: "Metadata",
    0x0003: "Model",
    0x0004: "TensorTable",
    0x0005: "TensorDesc",
    0x0006: "ChunkList",
    0x0007: "Blob",
    0x0008: "Tokenizer",
    0x0009: "ChatTemplate",
    0x000A: "GraphModule",
    0x000B: "Adapter",
    0x000C: "Signature",
    0x000D: "Provenance",
    0x000E: "Evaluation",
    0x000F: "Plugin",
    0x0010: "DialectRef",
    0x0011: "Rewrite",
    0x0012: "TrainingState",
    0x0013: "Optimizer",
    0x0014: "ShardMap",
    0x0015: "Foreign",
    0x0016: "RuntimeCache",
    0x0017: "Codebook",
    0x0018: "Function",
    0x0019: "SparseTensor",
}

SEG_KIND = {1: "BLOB", 2: "OBJ", 3: "INDEX", 4: "SUPER", 5: "SIG", 6: "PAD"}


class Invalid(Exception):
    """A container that breaks a rule. Carries the rule id, as §15.1 requires."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule


class Unsupported(Exception):
    """Above C0. Not a wrong answer — a refusal to give one (§15.1)."""


# ----------------------------------------------------------------------- crc32c --

def _crc32c_table() -> list[int]:
    # The Castagnoli polynomial, reflected. Built rather than pasted so the
    # constant that matters is the polynomial and not 256 magic numbers.
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
        table.append(c)
    return table


_CRC32C = _crc32c_table()


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC32C[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


# ---------------------------------------------------------------------- blake3 --

_IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
_MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

_CHUNK_START, _CHUNK_END, _PARENT, _ROOT = 1, 2, 4, 8
_MASK = 0xFFFFFFFF


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


def _g(s: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    s[a] = (s[a] + s[b] + mx) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 7)


def _round(s: list[int], m: list[int]) -> None:
    _g(s, 0, 4, 8, 12, m[0], m[1])
    _g(s, 1, 5, 9, 13, m[2], m[3])
    _g(s, 2, 6, 10, 14, m[4], m[5])
    _g(s, 3, 7, 11, 15, m[6], m[7])
    _g(s, 0, 5, 10, 15, m[8], m[9])
    _g(s, 1, 6, 11, 12, m[10], m[11])
    _g(s, 2, 7, 8, 13, m[12], m[13])
    _g(s, 3, 4, 9, 14, m[14], m[15])


def _compress(cv: tuple[int, ...], block: bytes, counter: int, block_len: int,
              flags: int) -> list[int]:
    m = list(struct.unpack("<16I", block))
    s = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        _IV[0], _IV[1], _IV[2], _IV[3],
        counter & _MASK, (counter >> 32) & _MASK, block_len, flags,
    ]
    for r in range(7):
        _round(s, m)
        if r < 6:
            m = [m[i] for i in _MSG_PERMUTATION]
    # The feed-forward that makes the compression function one-way.
    return [s[i] ^ s[i + 8] for i in range(8)] + [s[i + 8] ^ cv[i] for i in range(8)]


def _chunk_cv(chunk: bytes, counter: int, extra_flags: int) -> tuple[int, ...]:
    """The chaining value of one chunk, up to 1024 bytes."""
    cv = _IV
    flags = _CHUNK_START
    # A chunk is 16 blocks of 64 bytes; the last one carries CHUNK_END and its
    # real length, and an empty input is one empty block rather than none.
    blocks = [chunk[i:i + 64] for i in range(0, len(chunk), 64)] or [b""]
    for i, blk in enumerate(blocks):
        if i == len(blocks) - 1:
            flags |= _CHUNK_END | extra_flags
        out = _compress(cv, blk.ljust(64, b"\0"), counter, len(blk), flags)
        cv = tuple(out[:8])
        flags = 0
    return cv


def _parent_cv(left: tuple[int, ...], right: tuple[int, ...],
               extra_flags: int) -> tuple[int, ...]:
    block = struct.pack("<8I", *left) + struct.pack("<8I", *right)
    return tuple(_compress(_IV, block, 0, 64, _PARENT | extra_flags)[:8])


def blake3_256(data: bytes) -> bytes:
    """BLAKE3, 32 bytes of output, unkeyed.

    The tree is built with the standard stack: a chunk's chaining value is merged
    with the one below it whenever the number of completed chunks makes the
    subtree full, which is exactly the binary counter's carry.
    """
    chunks = [data[i:i + 1024] for i in range(0, len(data), 1024)]
    if not chunks:
        chunks = [b""]
    if len(chunks) == 1:
        # A single chunk *is* the root, so ROOT goes on its last block. Empty
        # input included: the compression function makes that hash, not a case.
        return struct.pack("<8I", *_chunk_cv(chunks[0], 0, _ROOT))

    # Every chunk but the last is merged eagerly. The trailing-zero test on the
    # completed count is the binary counter's carry, and it is exactly when the
    # subtree to the left has become full.
    stack: list[tuple[int, ...]] = []
    for i, chunk in enumerate(chunks[:-1]):
        cv = _chunk_cv(chunk, i, 0)
        total = i + 1
        while total & 1 == 0:
            cv = _parent_cv(stack.pop(), cv, 0)
            total >>= 1
        stack.append(cv)

    # The last chunk is *not* pushed. It is folded against the stack, and the
    # final merge is the one that carries ROOT — which is why it has to be kept
    # out of the eager loop: a value merged there would have been finalized
    # without the flag, and the root of a two-chunk input is precisely that
    # merge.
    cv = _chunk_cv(chunks[-1], len(chunks) - 1, 0)
    while stack:
        left = stack.pop()
        flags = _ROOT if not stack else 0
        cv = _parent_cv(left, cv, flags)
    return struct.pack("<8I", *cv)


def digest(algo: int, data: bytes) -> bytes:
    if algo == HASH_BLAKE3_256:
        return blake3_256(data)
    if algo == HASH_SHA256:
        return hashlib.sha256(data).digest()
    # §02: an unknown algorithm makes every digest in the file uninterpretable,
    # including the root, so it is fatal rather than something to work around.
    raise Invalid("R-C05", f"unsupported hash algorithm 0x{algo:02x}")


ALGO_NAME = {HASH_BLAKE3_256: "blake3-256", HASH_SHA256: "sha2-256"}
ALGO_PREFIX = {HASH_BLAKE3_256: "b3", HASH_SHA256: "sha2"}


# ------------------------------------------------------------------------ cbor --

# §03.2 D7's closed list. A tag outside it is a semantic a reader would be
# inventing, so it is refused; `30` is an exact rational, which §04.3 needs
# because `b3x5` ternary is 8/5 bits per element and 1.6 is not that number.
TAG_BIGNUM_POS = 2
TAG_BIGNUM_NEG = 3
TAG_RATIONAL = 30
TAG_REF = 1001
TAG_DIGEST = 1002
TAG_DTYPE = 1003
TAG_SHAPE = 1004
TAG_EXPR = 1005
TAG_URI = 1006
TAG_DECIMAL = 1007

REGISTERED_TAGS = frozenset({
    TAG_BIGNUM_POS, TAG_BIGNUM_NEG, TAG_RATIONAL, TAG_REF, TAG_DIGEST,
    TAG_DTYPE, TAG_SHAPE, TAG_EXPR, TAG_URI, TAG_DECIMAL,
})


class Tag:
    """A tagged value, kept rather than unwrapped."""

    __slots__ = ("tag", "value")

    def __init__(self, tag: int, value) -> None:
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:
        return f"Tag({self.tag}, {self.value!r})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, Tag) and self.tag == other.tag
                and self.value == other.value)


def _shortest_float(v: float) -> bytes:
    """The one encoding §03.2 D5 permits for `v`: the shortest that is exact.

    Half if the value survives a round trip through it, single if not, double
    otherwise — and one fixed encoding for NaN, since a NaN with a payload would
    be a second encoding of the same value.
    """
    if v != v:
        return b"\xf9\x7e\x00"
    d = struct.pack(">d", v)
    try:
        f4 = struct.pack(">f", v)
    except OverflowError:
        return b"\xfb" + d
    if struct.pack(">d", struct.unpack(">f", f4)[0]) != d:
        return b"\xfb" + d
    as32 = struct.unpack(">f", f4)[0]
    try:
        f2 = struct.pack(">e", as32)
    except OverflowError:
        return b"\xfa" + f4
    if struct.pack(">f", struct.unpack(">e", f2)[0]) != f4:
        return b"\xfa" + f4
    return b"\xf9" + f2


class Decoder:
    """Canonical OMNI-CBOR (§03.2), decoded with its rules enforced.

    §03.2 makes canonical form part of validity, not a style: two encodings of
    the same value would be two digests for the same object, which is the one
    thing a content-addressed format cannot allow. So a reader that accepts a
    non-canonical encoding is not being lenient, it is reading a different
    format — and every rule below is a rule this decoder refuses to bend.
    """

    def __init__(self, data: bytes, max_depth: int = 64) -> None:
        self.d = data
        self.i = 0
        self.max_depth = max_depth

    def decode(self):
        v = self._value(0)
        if self.i != len(self.d):
            raise Invalid("D8", f"{len(self.d) - self.i} trailing byte(s)")
        return v

    def _byte(self) -> int:
        if self.i >= len(self.d):
            raise Invalid("R-E01", "unexpected end of input")
        b = self.d[self.i]
        self.i += 1
        return b

    def _take(self, n: int) -> bytes:
        if self.i + n > len(self.d):
            raise Invalid("R-E01", "declared length exceeds the available input")
        out = self.d[self.i:self.i + n]
        self.i += n
        return out

    def _argument(self, ai: int) -> int:
        """The head's argument, in the shortest form that fits it (D1)."""
        if ai < 24:
            return ai
        if ai == 24:
            n = self._byte()
            if n < 24:
                raise Invalid("D1", f"{n} encoded in 1 byte, not inline")
            return n
        if ai == 25:
            n = struct.unpack(">H", self._take(2))[0]
            if n <= 0xFF:
                raise Invalid("D1", f"{n} encoded in 2 bytes")
            return n
        if ai == 26:
            n = struct.unpack(">I", self._take(4))[0]
            if n <= 0xFFFF:
                raise Invalid("D1", f"{n} encoded in 4 bytes")
            return n
        if ai == 27:
            n = struct.unpack(">Q", self._take(8))[0]
            if n <= 0xFFFFFFFF:
                raise Invalid("D1", f"{n} encoded in 8 bytes")
            return n
        if ai == 31:
            # §03.2 D2: indefinite lengths are forbidden, because the same value
            # would have two encodings.
            raise Invalid("D2", "indefinite length")
        raise Invalid("R-E02", f"reserved additional information {ai}")

    def _value(self, depth: int):
        if depth > self.max_depth:
            raise Invalid("R-E04", f"nesting deeper than {self.max_depth} (§02.10)")
        head = self._byte()
        major, ai = head >> 5, head & 0x1F
        if major == 0:
            return self._argument(ai)
        if major == 1:
            return -1 - self._argument(ai)
        if major == 2:
            return self._take(self._argument(ai))
        if major == 3:
            raw = self._take(self._argument(ai))
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as e:
                raise Invalid("R-E03", f"invalid UTF-8 in a text string: {e}") from None
        if major == 4:
            return [self._value(depth + 1) for _ in range(self._argument(ai))]
        if major == 5:
            n = self._argument(ai)
            out = {}
            prev = None
            for _ in range(n):
                start = self.i
                key = self._value(depth + 1)
                raw = self.d[start:self.i]
                # D3: keys sorted by their encoded bytes, and D4: no duplicates.
                # Both matter for the same reason: one value, one encoding.
                if prev is not None and raw <= prev:
                    if raw == prev:
                        raise Invalid("D4", "duplicate map key")
                    raise Invalid("D3", "map keys are not in canonical order")
                prev = raw
                out[key if isinstance(key, (str, int)) else raw] = self._value(depth + 1)
            return out
        if major == 6:
            # D7: registered tags only. §03.2 keeps a closed list — an exact
            # rational, a reference, a digest, a dtype — so an unregistered tag is
            # a semantic this reader would be inventing.
            tag = self._argument(ai)
            if tag not in REGISTERED_TAGS:
                raise Invalid("D7", f"unregistered tag {tag}")
            inner = self._value(depth + 1)
            # The tag is kept, because dropping it would silently turn 8/5 into
            # the array [8, 5].
            return Tag(tag, inner)
        # major 7: simple values and floats.
        if ai == 20:
            return False
        if ai == 21:
            return True
        if ai == 22:
            return None
        if ai in (25, 26, 27):
            width = {25: 2, 26: 4, 27: 8}[ai]
            raw = self.d[self.i - 1:self.i + width]
            fmt = {25: ">e", 26: ">f", 27: ">d"}[ai]
            f = struct.unpack(fmt, self._take(width))[0]
            # D5 is not "doubles only": it is the *shortest* float encoding that
            # reproduces the value exactly — half, then single, then double. That
            # is what makes a value's encoding unique, and re-encoding is the only
            # honest way to check it.
            if _shortest_float(f) != raw:
                raise Invalid("D5", f"float {f} is not in its shortest exact form")
            return f
        raise Invalid("R-E02", f"reserved simple value {ai}")


def cbor_decode(data: bytes):
    return Decoder(data).decode()


# ------------------------------------------------------------------- container --

class Entry:
    __slots__ = ("digest", "offset", "stored_len", "logical_len", "otype",
                 "codec", "oflags")

    def __init__(self, raw: bytes) -> None:
        self.digest = raw[0:32]
        (self.offset, self.stored_len, self.logical_len) = struct.unpack(
            "<3Q", raw[32:56])
        self.otype = struct.unpack("<H", raw[56:58])[0]
        self.codec = raw[58]
        self.oflags = raw[59]

    @property
    def type_name(self) -> str:
        return OTYPE.get(self.otype, f"0x{self.otype:04x}")


class Container:
    """A container opened the way §02.7 says a seek-capable reader should."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < HEADER_SIZE + TRAILER_SIZE:
            raise Invalid("R-C01", "file too small to be a container")

        # --- header ---
        if data[0:8] != MAGIC:
            raise Invalid("R-C01", "bad magic")
        if crc32c(data[0:124]) != struct.unpack("<I", data[124:128])[0]:
            raise Invalid("R-C02", "header CRC mismatch")
        self.container_major, self.container_minor = struct.unpack("<2H", data[8:12])
        if data[12] != 0x01:
            raise Invalid("R-C01", "unsupported byte order")
        self.log2_align = data[13]
        if not 6 <= self.log2_align <= 30:
            raise Invalid("R-C04", "log2_align out of range")
        self.header_size = struct.unpack("<H", data[14:16])[0]
        if not 128 <= self.header_size <= 4096:
            raise Invalid("R-C03", "header_size out of range")
        self.uuid = data[16:32]
        self.hash = data[32]
        if self.hash not in ALGO_NAME:
            raise Invalid("R-C05", f"unsupported hash algorithm 0x{self.hash:02x}")
        self.profile = data[33]
        if data[34] != 32:
            raise Invalid("R-C05", "digest_len does not match the algorithm")
        self.flags = struct.unpack("<I", data[36:40])[0]
        self.front_sb_off, self.front_sb_len, self.file_size = struct.unpack(
            "<3Q", data[40:64])
        self.root_digest = data[64:96]
        self.creator = data[96:112].rstrip(b"\0").decode("utf-8", "replace")

        # --- trailer, then one jump to the superblock: the two-read open ---
        t = len(data) - TRAILER_SIZE
        if data[t + 56:t + 64] != MAGIC_END:
            raise Invalid("R-C09", "trailer magic mismatch")
        if crc32c(data[t:t + 52]) != struct.unpack("<I", data[t + 52:t + 56])[0]:
            raise Invalid("R-C09", "trailer CRC mismatch")
        sb_off, sb_len = struct.unpack("<2Q", data[t:t + 16])
        sb_digest = data[t + 16:t + 48]
        if sb_off + sb_len > len(data):
            raise Invalid("R-C12", "superblock extent out of range")
        sb_bytes = data[sb_off:sb_off + sb_len]
        if digest(self.hash, sb_bytes) != sb_digest:
            raise Invalid("R-C09", "superblock digest mismatch")
        self.superblock = cbor_decode(sb_bytes)

        # R-C10: a front superblock has to be byte-identical to the back one, or
        # the two reads a reader might make would disagree.
        if self.flags & 0x01:
            fo, fl = self.front_sb_off, self.front_sb_len
            if fo + fl > len(data):
                raise Invalid("R-C12", "front superblock extent out of range")
            if data[fo:fo + fl] != sb_bytes:
                raise Invalid("R-C10", "front and back superblocks differ")

        # §02.5: the superblock names the container's algorithm, and it has to be
        # the one the header names.
        named = self.superblock.get("hash")
        if named is not None and named != ALGO_NAME[self.hash]:
            raise Invalid(
                "R-C05",
                f"the header says {ALGO_NAME[self.hash]} and the superblock says {named}")

        # --- index ---
        idx = self.superblock.get("index") or {}
        self.index_off = int(idx.get("off", 0))
        self.index_len = int(idx.get("len", 0))
        self.index: list[Entry] = []
        self.buckets: list[int] = []
        self.bucket_bits = 0
        self._parse_index()

    # -- index ------------------------------------------------------------

    def _parse_index(self) -> None:
        off, length = self.index_off, self.index_len
        if off + length > len(self.data) or length < IDX_HEADER_SIZE:
            raise Invalid("R-C12", "index extent out of range")
        h = self.data[off:off + IDX_HEADER_SIZE]
        if h[0:4] != IDX_MAGIC:
            raise Invalid("R-C11", "bad index magic")
        if crc32c(h[0:60]) != struct.unpack("<I", h[60:64])[0]:
            raise Invalid("R-C11", "index header CRC mismatch")
        entry_size = struct.unpack("<H", h[6:8])[0]
        if entry_size != IDX_ENTRY_SIZE:
            raise Invalid("R-C11", f"unsupported index entry size {entry_size}")
        n = struct.unpack("<Q", h[8:16])[0]
        if IDX_HEADER_SIZE + n * entry_size > length:
            raise Invalid("R-C11", "index entry count exceeds segment")
        bucket_off = struct.unpack("<Q", h[16:24])[0]
        self.bucket_bits = struct.unpack("<I", h[24:28])[0]
        if h[28] != self.hash:
            raise Invalid("R-C11", "the index names a different hash than the header")

        prev = None
        for i in range(n):
            p = off + IDX_HEADER_SIZE + i * entry_size
            e = Entry(self.data[p:p + entry_size])
            # R-C11: strictly sorted. Not a nicety — the bucket table and every
            # binary search over the index depend on it.
            if prev is not None and e.digest <= prev:
                raise Invalid("R-C11", "index not strictly sorted")
            prev = e.digest
            # R-C13: a declared expansion ratio above 1000:1 is a decompression
            # bomb, refused before anything is read.
            if e.stored_len and e.logical_len // max(e.stored_len, 1) > 1000:
                raise Invalid("R-C13", "declared expansion ratio exceeds 1000:1")
            self.index.append(e)

        # §02.6.1's bucket table. An accelerator and never authority: a bucket
        # that points outside the index is ignored rather than trusted.
        if self.bucket_bits and bucket_off:
            count = 1 << self.bucket_bits
            base = off + bucket_off
            if base + count * 4 <= off + length:
                self.buckets = list(
                    struct.unpack(f"<{count}I", self.data[base:base + count * 4]))
                if any(b > n for b in self.buckets):
                    self.buckets = []

    def find(self, d: bytes) -> Entry | None:
        """§02.6.2's hot path: one bucket read, then a search of its entries."""
        lo, hi = 0, len(self.index)
        if self.buckets and self.bucket_bits:
            top = (d[0] << 16) | (d[1] << 8) | d[2]
            b = top >> (24 - self.bucket_bits)
            lo = self.buckets[b]
            hi = self.buckets[b + 1] if b + 1 < len(self.buckets) else len(self.index)
        while lo < hi:
            mid = (lo + hi) // 2
            got = self.index[mid].digest
            if got == d:
                return self.index[mid]
            if got < d:
                lo = mid + 1
            else:
                hi = mid
        return None

    # -- objects ----------------------------------------------------------

    def segments(self) -> list[tuple[int, str, int]]:
        """Every segment, walked by its own header chain (§02.4)."""
        out = []
        pos = self.header_size
        end = len(self.data) - TRAILER_SIZE
        while pos + SEG_HEADER_SIZE <= end:
            if self.data[pos:pos + 4] != SEG_MAGIC:
                # Segments are padded and a superblock may reserve more room than
                # its payload uses, so the chain is followed by scanning for the
                # next magic on an 8-byte boundary rather than by arithmetic.
                pos += 8
                continue
            h = self.data[pos:pos + SEG_HEADER_SIZE]
            if crc32c(h[0:28]) != struct.unpack("<I", h[28:32])[0]:
                raise Invalid("R-C05", f"segment header CRC mismatch at {pos:#x}")
            kind = struct.unpack("<H", h[4:6])[0]
            plen = struct.unpack("<Q", h[8:16])[0]
            p = pos + SEG_HEADER_SIZE
            if p + plen > end:
                raise Invalid("R-C05", f"segment payload overruns the file at {pos:#x}")
            if crc32c(self.data[p:p + plen]) != struct.unpack("<I", h[24:28])[0]:
                raise Invalid("R-C05", f"segment payload CRC mismatch at {pos:#x}")
            out.append((pos, SEG_KIND.get(kind, f"0x{kind:04x}"), plen))
            pos = (p + plen + 7) // 8 * 8
        return out

    def get(self, d: bytes) -> bytes:
        """An object's logical bytes, with its digest checked (R-O01)."""
        e = self.find(d)
        if e is None:
            raise Invalid("R-C14", f"no object {d.hex()[:16]} in the index")
        if e.codec != 0:
            # Above C0: §03.7's codecs are a C1 feature and this reader says so
            # rather than returning compressed bytes as if they were the object.
            raise Unsupported(
                f"object {d.hex()[:16]} is stored with codec {e.codec}; C0 reads "
                f"`raw` only (§03.7 is above it)")
        if e.offset + e.stored_len > len(self.data):
            raise Invalid("R-C12", "object extent out of range")
        payload = self.data[e.offset:e.offset + e.stored_len]
        if digest(self.hash, payload) != d:
            raise Invalid("R-O01", f"digest mismatch for {d.hex()[:16]}")
        return payload

    def get_value(self, d: bytes):
        return cbor_decode(self.get(d))

    def verify(self) -> tuple[int, int]:
        """Every object rehashed. Returns (objects, bytes)."""
        n = b = 0
        for e in self.index:
            if e.codec != 0:
                continue
            self.get(e.digest)
            n += 1
            b += e.stored_len
        return n, b

    # -- the object graph -------------------------------------------------

    def root(self):
        return self.get_value(self.root_digest)

    @staticmethod
    def ref(value) -> bytes | None:
        """A `[otype, digest]` reference, as §01.5 writes them."""
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], bytes):
            return value[1]
        return None

    def asset(self, slot: str):
        """An object reached from the manifest's `assets` (§03.4)."""
        manifest = self.root()
        got = (manifest.get("assets") or {}).get(slot)
        d = self.ref(got)
        return self.get_value(d) if d else None

    def meta(self):
        d = self.ref(self.root().get("meta"))
        return self.get_value(d) if d else None

    def tensors(self) -> dict[str, bytes]:
        """Tensor name -> descriptor digest, from the model's table."""
        model = self.asset("model")
        if model is None:
            return {}
        d = self.ref(model.get("tensors"))
        if d is None:
            return {}
        table = self.get_value(d)
        out = {}
        for name, r in (table.get("tensors") or {}).items():
            dd = self.ref(r)
            if dd:
                out[name] = dd
        return out

    def tensor_bytes(self, name: str) -> tuple[bytes, dict]:
        """A literal tensor's dense bytes, plus its descriptor (§04.5).

        C0's tensor requirement is `literal` and nothing else, so a tensor whose
        value is an expression is refused by name — with the node that made it
        one, which is the useful half of the message.
        """
        d = self.tensors().get(name)
        if d is None:
            raise Invalid("R-T01", f"no tensor `{name}` in the table")
        desc = self.get_value(d)
        value = desc.get("value") or {}
        op = value.get("op")
        if op != "literal":
            raise Unsupported(
                f"`{name}` is a `{op}` expression; C0 reads `literal` tensors "
                f"(§04.7's algebra is above it)")
        layout = desc.get("layout") or {}
        kind = layout.get("k", "strided")
        if kind not in ("strided", None):
            raise Unsupported(
                f"`{name}` has a `{kind}` layout; C0 reads dense row-major "
                f"(§04.4's other layouts are above it)")
        if layout.get("order") not in (None, "row-major", "c"):
            raise Unsupported(f"`{name}` is {layout.get('order')}, not row-major")
        cl = self.ref(value.get("chunks"))
        if cl is None:
            raise Invalid("R-T02", f"`{name}` has no chunk list")
        return self.chunk_bytes(cl), desc

    def chunk_bytes(self, d: bytes) -> bytes:
        """A ChunkList reassembled, with its declared total checked (R-T02)."""
        cl = self.get_value(d)
        out = bytearray()
        for c in cl.get("chunks") or []:
            r = self.ref(c.get("r"))
            if r is None:
                raise Invalid("R-T02", "a chunk with no reference")
            blob = self.get(r)
            n = c.get("n")
            if n is not None and len(blob) != n:
                raise Invalid(
                    "R-T02", f"a chunk declares {n} bytes and holds {len(blob)}")
            out += blob
        total = cl.get("total")
        if total is not None and len(out) != total:
            raise Invalid(
                "R-T02", f"the chunk list declares {total} bytes and holds {len(out)}")
        return bytes(out)


def open_file(path: str) -> Container:
    with open(path, "rb") as f:
        return Container(f.read())


# ------------------------------------------------------------------- the script --

def _dtype_of(desc: dict) -> str:
    dt = desc.get("dtype")
    if isinstance(dt, dict):
        return dt.get("alias") or dt.get("k") or "?"
    return str(dt)


def _shape_of(desc: dict) -> list:
    return [d if isinstance(d, int) else "?" for d in (desc.get("shape") or [])]


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = [a for a in argv[1:] if a.startswith("--")]
    if not args:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        print("usage: omni.py <file.omni> [--check] [--tensor NAME] [--json]",
              file=sys.stderr)
        return 2
    path = args[0]
    try:
        c = open_file(path)
    except (Invalid, Unsupported) as e:
        print(f"omni.py: {e}", file=sys.stderr)
        return 1

    want_tensor = None
    for i, o in enumerate(opts):
        if o == "--tensor" and i + 1 < len(opts):
            want_tensor = opts[i + 1]
    if want_tensor is None and "--tensor" in argv:
        k = argv.index("--tensor")
        if k + 1 < len(argv):
            want_tensor = argv[k + 1]
            if want_tensor in args:
                args.remove(want_tensor)

    if want_tensor:
        try:
            raw, desc = c.tensor_bytes(want_tensor)
        except (Invalid, Unsupported) as e:
            print(f"omni.py: {e}", file=sys.stderr)
            return 3 if isinstance(e, Unsupported) else 1
        print(f"{want_tensor}  {_dtype_of(desc)} {_shape_of(desc)}  {len(raw)} bytes")
        print(f"  {ALGO_PREFIX[c.hash]}:{digest(c.hash, raw).hex()}")
        return 0

    prefix = ALGO_PREFIX[c.hash]
    print(f"{path}")
    print(f"  omni            {c.container_major}.{c.container_minor}, "
          f"{ALGO_NAME[c.hash]}, {len(c.data)} bytes")
    print(f"  creator         {c.creator}")
    # In full, not truncated: the root digest is the file's identity, and a
    # cross-implementation check needs all of it.
    print(f"  root            {prefix}:{c.root_digest.hex()}")
    print(f"  objects         {len(c.index)}")
    segs = c.segments()
    print(f"  segments        {len(segs)}  "
          f"({', '.join(k for _, k, _ in segs)})")
    meta = c.meta()
    if meta:
        arch = (meta.get("arch") or {}).get("family")
        print(f"  name            {meta.get('name')}")
        if arch:
            print(f"  arch            {arch}")
        if meta.get("params_total") is not None:
            print(f"  params          {meta['params_total']:,}")
    tensors = c.tensors()
    print(f"  tensors         {len(tensors)}")
    literal = unsupported = 0
    for name in sorted(tensors):
        try:
            raw, desc = c.tensor_bytes(name)
            literal += 1
            if "--verbose" in opts:
                print(f"     {name:<40} {_dtype_of(desc)} {_shape_of(desc)} "
                      f"{len(raw)} B")
        except Unsupported as e:
            unsupported += 1
            if "--verbose" in opts:
                print(f"     {name:<40} above C0: {e}")
    print(f"     literal      {literal}")
    if unsupported:
        # C0 is a floor, not a claim to read everything. Counting what it cannot
        # read is the honest half of claiming what it can.
        print(f"     above C0     {unsupported} (expressions or non-dense layouts)")
    if "--check" in opts:
        n, b = c.verify()
        print(f"  verified        {n} object(s), {b} bytes rehashed — every digest")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
