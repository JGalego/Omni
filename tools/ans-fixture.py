#!/usr/bin/env python3
"""An `ans-lut` encoder and decoder, written from §03.7.5 and nothing else.

Every other codec in §03.7.1 belongs to somebody else, and CI checks ours
against theirs — zstd against libzstd, xz against liblzma. `ans-lut` is OMNI's
own, so there is no library to disagree with, and the question a second
implementation answers is a different and more important one: **is the
specification enough?** This file was written from the text of §03.7.5 — the
header layout, the block kinds, the frequency-table rules R-C20 through R-C23,
the rANS update and the renormalization bound — and not from
`reference/omni-core/src/ans.rs`. If the two agree on every byte in both
directions, the section says what it needs to say. If they do not, the section
is what gets fixed.

    tools/ans-fixture.py encode in.bin out.ans
    tools/ans-fixture.py decode in.ans out.bin <logical-length>
    tools/ans-fixture.py selftest
"""

import struct
import sys

VERSION = 1
SCALE = 12
LOWER = 1 << 16
BLOCK = 1 << 16


def _write_varint(out, v):
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)


def _read_varint(d, at):
    value = 0
    for i in range(9):
        if at >= len(d):
            raise ValueError("a varint runs off the end")
        b = d[at]
        at += 1
        value |= (b & 0x7F) << (i * 7)
        if not b & 0x80:
            return value, at
    raise ValueError("a varint is too long")


def _normalize(counts, total):
    """Scale the counts to sum to exactly 1 << SCALE, every used symbol ≥ 1."""
    target = 1 << SCALE
    freq = [0] * 256
    used = [s for s in range(256) if counts[s]]
    for s in used:
        freq[s] = max(1, min(target, counts[s] * target // total))
    largest = max(used, key=lambda s: counts[s])
    total_now = sum(freq)
    while total_now > target:
        over = min(total_now - target, freq[largest] - 1)
        freq[largest] -= over
        total_now -= over
        if freq[largest] == 1:
            candidates = [s for s in used if freq[s] > 1 and s != largest]
            if not candidates:
                break
            largest = candidates[0]
    if total_now < target:
        freq[largest] += target - total_now
    return freq


def _cumulative(freq):
    cum = [0] * 257
    for s in range(256):
        cum[s + 1] = cum[s] + freq[s]
    return cum


def _lut(freq, cum):
    table = bytearray(1 << SCALE)
    for s in range(256):
        for slot in range(cum[s], cum[s] + freq[s]):
            table[slot] = s
    return table


def _encode_block(block):
    counts = [0] * 256
    for b in block:
        counts[b] += 1
    used = [s for s in range(256) if counts[s]]
    freq = _normalize(counts, len(block))
    cum = _cumulative(freq)

    emitted = []
    x = LOWER
    for b in reversed(block):
        f = freq[b]
        maximum = ((LOWER >> SCALE) << 16) * f
        while x >= maximum:
            emitted.append(x & 0xFFFF)
            x >>= 16
        x = ((x // f) << SCALE) + (x % f) + cum[b]

    payload = bytearray(struct.pack('<I', x))
    for w in reversed(emitted):
        payload += struct.pack('<H', w)

    table_bytes = 1 + len(used) * 3
    if table_bytes + 4 + len(payload) >= len(block):
        return None
    out = bytearray()
    out.append(len(used) - 1)
    for s in used:
        out.append(s)
        out += struct.pack('<H', freq[s])
    out += struct.pack('<I', len(payload))
    out += payload
    return bytes(out)


def encode(data):
    blocks = max(1, (len(data) + BLOCK - 1) // BLOCK)
    out = bytearray()
    out.append(VERSION)
    out.append(SCALE)
    out += struct.pack('<I', BLOCK)
    _write_varint(out, blocks)
    if not data:
        out.append(0)
        out += struct.pack('<I', 0)
        return bytes(out)
    for start in range(0, len(data), BLOCK):
        block = data[start:start + BLOCK]
        coded = _encode_block(block)
        if coded is None:
            out.append(0)
            out += struct.pack('<I', len(block))
            out += block
        else:
            out.append(1)
            out += struct.pack('<I', len(block))
            out += coded
    return bytes(out)


def decode(data, limit):
    if not data:
        raise ValueError("an empty stream")
    if data[0] != VERSION:
        raise ValueError(f"version {data[0]} is not 1")
    scale = data[1]
    if not 8 <= scale <= 16:
        raise ValueError(f"log2_scale {scale} is outside 8..=16 (R-C23)")
    at = 2
    _block_elems = struct.unpack_from('<I', data, at)[0]
    at += 4
    count, at = _read_varint(data, at)

    total = 1 << scale
    out = bytearray()
    for _ in range(count):
        kind = data[at]
        at += 1
        n = struct.unpack_from('<I', data, at)[0]
        at += 4
        if len(out) + n > limit:
            raise ValueError(f"decoding would exceed the declared {limit} bytes")
        if kind == 0:
            out += data[at:at + n]
            at += n
            continue
        if kind != 1:
            raise ValueError(f"block kind {kind} is not 0 or 1")

        used = data[at] + 1
        at += 1
        freq = [0] * 256
        last = -1
        for _ in range(used):
            s = data[at]
            f = struct.unpack_from('<H', data, at + 1)[0]
            at += 3
            if s <= last:
                raise ValueError("the table's symbols are not increasing (R-C21)")
            if f == 0:
                raise ValueError("a listed symbol has frequency zero (R-C21)")
            last = s
            freq[s] = f
        if sum(freq) != total:
            raise ValueError(f"the frequencies sum to {sum(freq)}, not {total} (R-C20)")
        cum = _cumulative(freq)
        table = _lut(freq, cum)

        payload_len = struct.unpack_from('<I', data, at)[0]
        at += 4
        payload = data[at:at + payload_len]
        at += payload_len
        if len(payload) < 4:
            raise ValueError("a payload shorter than its state")

        x = struct.unpack_from('<I', payload, 0)[0]
        p = 4
        for _ in range(n):
            slot = x & (total - 1)
            s = table[slot]
            x = freq[s] * (x >> scale) + slot - cum[s]
            while x < LOWER:
                if p + 2 > payload_len:
                    raise ValueError("the payload ends before the block does (R-C22)")
                x = (x << 16) | struct.unpack_from('<H', payload, p)[0]
                p += 2
            out.append(s)
    return bytes(out)


def selftest():
    import random
    rng = random.Random(20260814)
    weights = [1, 2, 5, 12, 26, 52, 92, 140, 140, 92, 52, 26, 12, 5, 2, 1]
    population = [i for i, w in enumerate(weights) for _ in range(w)]
    corpus = [
        b"",
        b"\x07",
        bytes(5000),
        bytes(range(256)),
        bytes(rng.choice(population) for _ in range(40000)),
        bytes(rng.randrange(256) for _ in range(9000)),
        bytes(rng.choice(population) for _ in range((1 << 16) + 17)),
    ]
    for data in corpus:
        back = decode(encode(data), max(len(data), 1))
        assert back == data, f"self round trip failed on {len(data)} bytes"
    print(f"ans-fixture: {len(corpus)} cases round-trip through this file alone")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    if argv[1] == "selftest":
        return selftest()
    if argv[1] == "encode" and len(argv) == 4:
        open(argv[3], 'wb').write(encode(open(argv[2], 'rb').read()))
        return 0
    if argv[1] == "decode" and len(argv) == 5:
        open(argv[3], 'wb').write(decode(open(argv[2], 'rb').read(), int(argv[4])))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
