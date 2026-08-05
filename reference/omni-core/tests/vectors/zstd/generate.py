#!/usr/bin/env python3
"""Regenerate the golden zstd frames in this directory using libzstd.

    pip install zstandard && python3 generate.py

These frames are the differential oracle for `omni_core::zstd`: the decoder in
this crate was written from RFC 8878, and the only way to know it agrees with
the format rather than with itself is to decode bytes produced by the reference
compressor. The payloads are generated from the same deterministic RNG that
`zstd.rs` uses in its tests, so a frame and the bytes it must produce are both
derivable from this file — nothing here is an opaque blob.

Each case targets a specific part of the format:

    text4k-l1     single block, Huffman literals, FSE sequences
    text4k-l19    the same payload at maximum effort: FSE-compressed Huffman
                  weights, four literal streams, checksum
    plane8k-l3    few distinct byte values, as a bitshuffled float plane has
    random3k-l3   incompressible: a Raw block
    zeros1k-l3    one byte repeated: an RLE block
    text200k-l3   multi-block: Treeless literal blocks reusing the previous
                  Huffman table, and matches reaching back into earlier blocks
"""
import pathlib
import zstandard as zstd

MASK = (1 << 64) - 1
WORDS = [b"the ", b"model ", b"exists ", b"once ", b"and ", b"everything ",
         b"else ", b"is ", b"derived ", b"from ", b"it. "]


class Rng:
    """The same LCG as `zstd.rs`'s test corpus generator."""

    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK
        return (self.s >> 33) & 0xFFFFFFFF


def corpus(kind, n):
    r = Rng(0x2545F4914F6CDD1D + kind)
    out = bytearray()
    if kind == 0:
        while len(out) < n:
            out += WORDS[r.next() % len(WORDS)]
    elif kind == 1:
        prev = 0x3F
        while len(out) < n:
            if r.next() % 5 == 0:
                prev = 0x3C + (r.next() % 8)
            out.append(prev)
    elif kind == 2:
        while len(out) < n:
            out.append(r.next() & 0xFF)
    elif kind == 3:
        out = bytearray(n)
    elif kind == 4:
        base = corpus(0, 1 << 16)
        while len(out) < n:
            off = r.next() % (len(base) - 4096)
            out += base[off:off + 4096]
    return bytes(out[:n])


CASES = [
    ("text4k-l1", 0, 4096, 1, False),
    ("text4k-l19", 0, 4096, 19, True),
    ("plane8k-l3", 1, 8192, 3, True),
    ("random3k-l3", 2, 3000, 3, False),
    ("zeros1k-l3", 3, 1024, 3, True),
    ("text200k-l3", 4, 200000, 3, True),
]

here = pathlib.Path(__file__).parent
for name, kind, n, level, checksum in CASES:
    data = corpus(kind, n)
    frame = zstd.ZstdCompressor(level=level, write_checksum=checksum,
                                write_content_size=True).compress(data)
    (here / f"{name}.zst").write_bytes(frame)
    print(f"{name}.zst  {n} B -> {len(frame)} B  (level {level}, "
          f"checksum {'on' if checksum else 'off'})")
