#!/usr/bin/env python3
"""A GGUF writer and dequantizer, written from the GGML block layouts.

This exists to disagree with `reference/omni-core/src/gguf.rs`. Every block
format here — the byte offsets, the nibble order, the shift order, the 6-bit
scale packing — was written from the format's own definition rather than from
that module, in another language, so that when the two agree on every value of
every type the agreement means something. The Rust side already checks itself
against a second scalar implementation inside the same file; this is a third,
in a different language, and it is the one that can catch a mistake the other
two share.

    tools/gguf-fixture.py write model.gguf values.txt
    tools/gguf-fixture.py check ./omni model.omni values.txt

`write` produces a GGUF file with one tensor of each supported block type, and
a text file of the float32 values each of them dequantizes to — in
little-endian hex, so the comparison is bit-exact rather than "close". `check`
evaluates the same tensors through a container and compares.
"""

import re
import struct
import subprocess
import sys
import random

QK_K = 256

def f16(x):  # float -> IEEE half bytes
    return struct.pack('<e', x)

def rnd_half(rng):
    # A finite, non-subnormal half in a modest range.
    return struct.unpack('<e', f16(rng.uniform(-2.0, 2.0)))[0]

def dq_half(b, at):
    return struct.unpack_from('<e', b, at)[0]

def blk_q4_0(rng):
    d = rnd_half(rng)
    qs = bytes(rng.randrange(256) for _ in range(16))
    y = []
    for j in range(16): y.append(((qs[j] & 0xF) - 8) * d)
    for j in range(16): y.append(((qs[j] >> 4) - 8) * d)
    return f16(d) + qs, y

def blk_q4_1(rng):
    d, m = rnd_half(rng), rnd_half(rng)
    qs = bytes(rng.randrange(256) for _ in range(16))
    y = [ (qs[j] & 0xF) * d + m for j in range(16) ] + [ (qs[j] >> 4) * d + m for j in range(16) ]
    return f16(d) + f16(m) + qs, y

def blk_q5_0(rng):
    d = rnd_half(rng)
    qh = rng.randrange(1 << 32)
    qs = bytes(rng.randrange(256) for _ in range(16))
    y = []
    for j in range(16): y.append((((qs[j] & 0xF) | (((qh >> j) & 1) << 4)) - 16) * d)
    for j in range(16): y.append((((qs[j] >> 4) | (((qh >> (j+16)) & 1) << 4)) - 16) * d)
    return f16(d) + struct.pack('<I', qh) + qs, y

def blk_q5_1(rng):
    d, m = rnd_half(rng), rnd_half(rng)
    qh = rng.randrange(1 << 32)
    qs = bytes(rng.randrange(256) for _ in range(16))
    y = []
    for j in range(16): y.append(((qs[j] & 0xF) | (((qh >> j) & 1) << 4)) * d + m)
    for j in range(16): y.append(((qs[j] >> 4) | (((qh >> (j+16)) & 1) << 4)) * d + m)
    return f16(d) + f16(m) + struct.pack('<I', qh) + qs, y

def blk_q8_0(rng):
    d = rnd_half(rng)
    qs = bytes(rng.randrange(256) for _ in range(32))
    y = [ (qs[j] - 256 if qs[j] > 127 else qs[j]) * d for j in range(32) ]
    return f16(d) + qs, y

def blk_q2_k(rng):
    scales = bytes(rng.randrange(256) for _ in range(16))
    qs = bytes(rng.randrange(256) for _ in range(64))
    d, dmin = rnd_half(rng), rnd_half(rng)
    y = []
    it = 0
    for n in range(2):
        q = qs[n*32:]
        for j in range(4):
            shift = 2*j
            for sub in range(2):
                sc = scales[it]; it += 1
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4)
                for l in range(16):
                    y.append(dl * ((q[sub*16+l] >> shift) & 3) - ml)
    return scales + qs + f16(d) + f16(dmin), y

def blk_q3_k(rng):
    hmask = bytes(rng.randrange(256) for _ in range(32))
    qs = bytes(rng.randrange(256) for _ in range(64))
    sraw = bytes(rng.randrange(256) for _ in range(12))
    d = rnd_half(rng)
    sc = [0]*16
    for k in range(4):
        sc[k]      = (sraw[k] & 0xF)        | ((sraw[8+k] & 3) << 4)
        sc[4+k]    = (sraw[4+k] & 0xF)      | (((sraw[8+k] >> 2) & 3) << 4)
        sc[8+k]    = ((sraw[k] >> 4) & 0xF) | (((sraw[8+k] >> 4) & 3) << 4)
        sc[12+k]   = ((sraw[4+k] >> 4) & 0xF) | (((sraw[8+k] >> 6) & 3) << 4)
    y = []
    it = 0; m = 0
    for n in range(2):
        q = qs[n*32:]
        for j in range(4):
            shift = 2*j
            for sub in range(2):
                dl = d * (sc[it] - 32); it += 1
                for l in range(16):
                    idx = sub*16 + l
                    bit = (hmask[idx] >> m) & 1
                    y.append(dl * (((q[idx] >> shift) & 3) - (0 if bit else 4)))
            m += 1
    return hmask + qs + sraw + f16(d), y

def scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j+4] & 63
    return ((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4))

def blk_q4_k(rng):
    d, dmin = rnd_half(rng), rnd_half(rng)
    sraw = bytes(rng.randrange(256) for _ in range(12))
    qs = bytes(rng.randrange(256) for _ in range(128))
    y = []
    for c in range(4):
        for h in range(2):
            sc, mn = scale_min_k4(c*2+h, sraw)
            d1, m1 = d*sc, dmin*mn
            for l in range(32):
                lo = (qs[c*32+l] & 0xF) if h == 0 else (qs[c*32+l] >> 4)
                y.append(d1*lo - m1)
    return f16(d) + f16(dmin) + sraw + qs, y

def blk_q5_k(rng):
    d, dmin = rnd_half(rng), rnd_half(rng)
    sraw = bytes(rng.randrange(256) for _ in range(12))
    qh = bytes(rng.randrange(256) for _ in range(32))
    qs = bytes(rng.randrange(256) for _ in range(128))
    y = []
    for c in range(4):
        for h in range(2):
            i = c*2+h
            sc, mn = scale_min_k4(i, sraw)
            d1, m1 = d*sc, dmin*mn
            for l in range(32):
                lo = (qs[c*32+l] & 0xF) if h == 0 else (qs[c*32+l] >> 4)
                hi = (qh[l] >> i) & 1
                y.append(d1*(lo + 16*hi) - m1)
    return f16(d) + f16(dmin) + sraw + qh + qs, y

def blk_q6_k(rng):
    ql = bytes(rng.randrange(256) for _ in range(128))
    qh = bytes(rng.randrange(256) for _ in range(64))
    sc = bytes(rng.randrange(256) for _ in range(16))
    d = rnd_half(rng)
    y = [0.0]*256
    for n in range(2):
        for l in range(32):
            i = l // 16
            for k in range(4):
                byte = ql[n*64 + (k % 2)*32 + l]
                low = (byte & 0xF) if k < 2 else (byte >> 4)
                high = (qh[n*32 + l] >> (2*k)) & 3
                q = (low | (high << 4)) - 32
                s = sc[n*8 + k*2 + i]
                s = s - 256 if s > 127 else s
                y[n*128 + k*32 + l] = d * s * q
    return ql + qh + sc + f16(d), y

TYPES = {
    'Q4_0': (2, 32, blk_q4_0), 'Q4_1': (3, 32, blk_q4_1),
    'Q5_0': (6, 32, blk_q5_0), 'Q5_1': (7, 32, blk_q5_1),
    'Q8_0': (8, 32, blk_q8_0),
    'Q2_K': (10, 256, blk_q2_k), 'Q3_K': (11, 256, blk_q3_k),
    'Q4_K': (12, 256, blk_q4_k), 'Q5_K': (13, 256, blk_q5_k),
    'Q6_K': (14, 256, blk_q6_k),
}

def gstr(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def write(path, kv, tensors):
    out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv)))
    for k, (t, enc) in kv.items():
        out += gstr(k) + struct.pack('<I', t) + enc
    off = 0
    infos = []
    for name, ty, dims, data in tensors:
        infos.append((name, ty, dims, off))
        off += len(data)
        off = (off + 31) // 32 * 32
    for name, ty, dims, o in infos:
        out += gstr(name) + struct.pack('<I', len(dims))
        for d in dims: out += struct.pack('<Q', d)
        out += struct.pack('<IQ', ty, o)
    start = (len(out) + 31) // 32 * 32
    out += b'\0' * (start - len(out))
    for (name, ty, dims, data), (_, _, _, o) in zip(tensors, infos):
        out += b'\0' * (start + o - len(out))
        out += data
    open(path, 'wb').write(bytes(out))

def write_fixture(path, valpath):
    rng = random.Random(20260806)
    kv = {
        'general.architecture': (8, gstr('llama')),
        'general.name': (8, gstr('differential')),
        'llama.block_count': (4, struct.pack('<I', 2)),
        'llama.embedding_length': (4, struct.pack('<I', 256)),
        'llama.attention.head_count': (4, struct.pack('<I', 8)),
        'tokenizer.chat_template': (8, gstr('{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}')),
    }
    tensors = []
    expected = {}
    for name, (ty, be, fn) in TYPES.items():
        nb = 4
        data = b''
        vals = []
        for _ in range(nb):
            b, y = fn(rng)
            data += b
            vals += y
        tname = name.lower() + '.weight'
        tensors.append((tname, ty, [be, nb], data))
        expected[tname] = vals
    write(path, kv, tensors)
    with open(valpath, 'w') as f:
        for name, vals in expected.items():
            # float32 hex, so the comparison is bit-exact rather than "close".
            f.write(name + ' ' + ' '.join(struct.pack('<f', v).hex() for v in vals) + '\n')

def check(omni, container, valpath):
    """Every value in the fixture, through the container, against Python."""
    bad = 0
    for line in open(valpath):
        name, rest = line.split(' ', 1)
        want = rest.split()
        out = subprocess.run(
            [omni, 'cat', container, '--tensor', name, '--hex',
             '--limit', str(4 * len(want))],
            capture_output=True, text=True, check=True).stdout
        raw = ''
        for l in out.splitlines():
            m = re.match(r'^[0-9a-f]{8}\s\s((?:[0-9a-f]{2} ?)+)', l)
            if m:
                raw += m.group(1).replace(' ', '')
        got = [raw[i * 8:(i + 1) * 8] for i in range(len(want))]
        if got == want:
            print(f'{name}: {len(want)} values agree bit-for-bit')
            continue
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                print(f'{name}[{i}]: omni {g}, python {w}')
                bad += 1
                break
    if bad:
        sys.exit(f'{bad} tensor(s) disagree')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('write', 'check'):
        sys.exit(__doc__)
    if sys.argv[1] == 'write':
        write_fixture(sys.argv[2], sys.argv[3])
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4])
