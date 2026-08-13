#!/usr/bin/env python3
"""An LZ4 block encoder and decoder, written from the block format definition.

This exists to disagree with `reference/omni-core/src/lz4.rs`. There is no
`liblz4` on the other side of this one — `zstd` gets checked against the library
everybody else links, and for `lz4` the runners have nothing to install that is
guaranteed present — so the second opinion is written here instead: the token
nibbles, the `255` continuation rule, the two-byte little-endian offset, the
`+4` on every match length and the three end-of-block rules, all from the
format's own definition in another language.

That is a weaker check than a library and it is worth saying so. What it can
still catch is the class of mistake that matters most for a codec: an encoder
whose output only its own decoder understands. So the check runs in both
directions — every block one side writes is read by the other — and the Python
encoder here is deliberately naive (a plain greedy hash table, no chain), so it
produces *different* sequences from the Rust one over the same input and the
agreement is about the format rather than about a shared strategy.

    tools/lz4-fixture.py encode in.bin out.lz4
    tools/lz4-fixture.py decode in.lz4 out.bin <logical-length>
    tools/lz4-fixture.py selftest

`selftest` checks this file against itself on a small corpus, so a bug here is
found before it is reported as a bug there.
"""

import struct
import sys

MIN_MATCH = 4
MAX_OFFSET = 65535
LAST_LITERALS = 5
MF_LIMIT = 12


def decode(src, cap):
    """Decodes an LZ4 block, refusing to produce more than `cap` bytes."""
    out = bytearray()
    ip = 0
    n = len(src)
    while ip < n:
        token = src[ip]
        ip += 1

        lit = token >> 4
        if lit == 15:
            extra, ip = _read_length(src, ip)
            lit += extra
        if ip + lit > n:
            raise ValueError(f"literal run of {lit} runs past the block")
        if len(out) + lit > cap:
            raise ValueError(f"decode exceeds the declared {cap} bytes")
        out += src[ip:ip + lit]
        ip += lit

        # A block ends with a literal-only sequence.
        if ip == n:
            break
        if ip + 2 > n:
            raise ValueError("match offset cut off by the end of the block")
        offset = struct.unpack_from('<H', src, ip)[0]
        ip += 2
        if offset == 0 or offset > len(out):
            raise ValueError(f"match offset {offset} into {len(out)} decoded bytes")

        mlen = token & 0x0F
        if mlen == 15:
            extra, ip = _read_length(src, ip)
            mlen += extra
        mlen += MIN_MATCH
        if len(out) + mlen > cap:
            raise ValueError(f"decode exceeds the declared {cap} bytes")
        # Byte at a time: an offset smaller than the length is how LZ4 spells a
        # run, and the match legitimately reads what it has just written.
        start = len(out) - offset
        for k in range(mlen):
            out.append(out[start + k])
    return bytes(out)


def _read_length(src, ip):
    """Reads a `255`-continued length extension, returning it and the new
    position."""
    total = 0
    while True:
        if ip >= len(src):
            raise ValueError("length extension runs off the end of the block")
        b = src[ip]
        ip += 1
        total += b
        if b != 255:
            return total, ip


def _write_length(out, n):
    while n >= 255:
        out.append(255)
        n -= 255
    out.append(n)


def encode(data):
    """A greedy single-candidate encoder — deliberately not the Rust one's
    strategy, so the two produce different sequences from the same input."""
    out = bytearray()
    n = len(data)
    if n < MF_LIMIT + 1:
        _emit_literals(out, data)
        return bytes(out)

    mf_limit = n - MF_LIMIT
    match_limit = n - LAST_LITERALS
    table = {}
    anchor = 0
    i = 0
    while i < mf_limit:
        key = data[i:i + 4]
        cand = table.get(key)
        table[key] = i
        if cand is None or i - cand > MAX_OFFSET:
            i += 1
            continue
        length = 0
        while i + length < match_limit and data[cand + length] == data[i + length]:
            length += 1
        if length < MIN_MATCH:
            i += 1
            continue
        _emit_sequence(out, data[anchor:i], i - cand, length)
        i += length
        anchor = i
    _emit_literals(out, data[anchor:])
    return bytes(out)


def _emit_sequence(out, literals, offset, match_len):
    extra = match_len - MIN_MATCH
    out.append((min(len(literals), 15) << 4) | min(extra, 15))
    if len(literals) >= 15:
        _write_length(out, len(literals) - 15)
    out += literals
    out += struct.pack('<H', offset)
    if extra >= 15:
        _write_length(out, extra - 15)


def _emit_literals(out, literals):
    out.append(min(len(literals), 15) << 4)
    if len(literals) >= 15:
        _write_length(out, len(literals) - 15)
    out += literals


def selftest():
    import random
    rng = random.Random(20260813)
    corpus = [
        b"",
        b"a",
        b"abcd",
        b"the quick brown fox jumps over the lazy dog",
        b"omni/1.0 " * 500,
        bytes(range(256)),
        bytes(rng.randrange(256) for _ in range(9000)),
        bytes(4096),
        b"a model exists once. everything else is derived. " * 300,
    ]
    for data in corpus:
        block = encode(data)
        back = decode(block, max(len(data), 1))
        assert back == data, f"self round trip failed on {len(data)} bytes"
    print(f"lz4-fixture: {len(corpus)} cases round-trip through this file alone")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "selftest":
        return selftest()
    if cmd == "encode" and len(argv) == 4:
        data = open(argv[2], 'rb').read()
        open(argv[3], 'wb').write(encode(data))
        return 0
    if cmd == "decode" and len(argv) == 5:
        data = open(argv[2], 'rb').read()
        open(argv[3], 'wb').write(decode(data, int(argv[4])))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
