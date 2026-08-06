#!/usr/bin/env python3
"""An ONNX writer, reader and evaluator, written from the protobuf wire format
and the operator specifications.

This exists to disagree with `reference/omni-core/src/onnx.rs`. The field
numbers, the varint encoding, the packed repeated fields, the attribute type
tags and the arithmetic of each operator were written from ONNX's own
definitions rather than from that module, in another language, so that when the
two agree on every byte of a file and every value it computes the agreement
means something.

    tools/onnx-fixture.py write model.onnx values.txt
    tools/onnx-fixture.py compat compat.onnx
    tools/onnx-fixture.py check ./omni model.omni values.txt
    tools/onnx-fixture.py compare original.onnx exported.onnx

`write` produces a model whose every node maps onto exactly one OMNI op, plus a
file of the initializer bytes and the values the graph computes. `compat`
produces one whose nodes do not, for the other half of the mapping. `check`
reads the initializers back out of an OMNI container and compares them bit for
bit. `compare` parses two ONNX files with this reader and reports the first
structural difference, then whether the bytes are identical — which is a
stronger statement than "it round-tripped", because it says *where* it did not.
"""

import re
import struct
import subprocess
import sys

# ------------------------------------------------------------------ protobuf --

def varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)

def key(field, wire):
    return varint(field << 3 | wire)

def f_int(field, v):
    """A varint field. proto3 omits a default, and so does every ONNX writer."""
    return b'' if v == 0 else key(field, 0) + varint(v & (2**64 - 1))

def f_int_always(field, v):
    return key(field, 0) + varint(v & (2**64 - 1))

def f_f32(field, v):
    return b'' if v == 0.0 else key(field, 5) + struct.pack('<f', v)

def f_bytes(field, v):
    return b'' if len(v) == 0 else f_bytes_always(field, v)

def f_bytes_always(field, v):
    return key(field, 2) + varint(len(v)) + v

def f_text(field, s):
    return f_bytes(field, s.encode())

def f_msg(field, body):
    return f_bytes_always(field, body)

def f_packed_ints(field, xs):
    if not xs:
        return b''
    body = b''.join(varint(x & (2**64 - 1)) for x in xs)
    return f_bytes_always(field, body)

def f_packed_f32(field, xs):
    if not xs:
        return b''
    return f_bytes_always(field, b''.join(struct.pack('<f', x) for x in xs))

def read_varint(b, i):
    out, shift = 0, 0
    while True:
        x = b[i]
        i += 1
        out |= (x & 0x7F) << shift
        if not x & 0x80:
            return out, i
        shift += 7

def fields(b):
    """Every (number, wire type, payload) in a message, in the order written."""
    i = 0
    while i < len(b):
        k, i = read_varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = read_varint(b, i)
            yield f, w, v
        elif w == 1:
            yield f, w, b[i:i + 8]
            i += 8
        elif w == 2:
            n, i = read_varint(b, i)
            yield f, w, b[i:i + n]
            i += n
        elif w == 5:
            yield f, w, b[i:i + 4]
            i += 4
        else:
            raise ValueError(f'wire type {w} (a group) at {i}')

# --------------------------------------------------------------- onnx writing --

FLOAT, UINT8, INT8, INT32, INT64, BOOL = 1, 2, 3, 6, 7, 9

def tensor_proto(name, dtype, dims, raw):
    return (f_packed_ints(1, dims) + f_int(2, dtype) + f_text(8, name)
            + f_bytes(9, raw))

def f32_tensor(name, dims, values):
    return tensor_proto(name, FLOAT, dims,
                        b''.join(struct.pack('<f', v) for v in values))

def i64_tensor(name, dims, values):
    return tensor_proto(name, INT64, dims,
                        b''.join(struct.pack('<q', v) for v in values))

def attr_int(name, v):
    return f_text(1, name) + f_int(3, v) + f_int(20, 2)

def attr_ints(name, xs):
    return f_text(1, name) + f_packed_ints(8, xs) + f_int(20, 7)

def attr_float(name, v):
    return f_text(1, name) + f_f32(2, v) + f_int(20, 1)

def attr_tensor(name, t):
    return f_text(1, name) + f_msg(5, t) + f_int(20, 4)

def node(op_type, ins, outs, name='', attrs=(), domain=''):
    body = b''.join(f_bytes_always(1, i.encode()) for i in ins)
    body += b''.join(f_bytes_always(2, o.encode()) for o in outs)
    body += f_text(3, name) + f_text(4, op_type)
    body += b''.join(f_msg(5, a) for a in attrs)
    body += f_text(7, domain)
    return body

def type_proto(elem, dims):
    shape = b''
    for d in dims:
        shape += f_msg(1, f_int_always(1, d) if isinstance(d, int) else f_text(2, d))
    return f_msg(1, f_int(1, elem) + f_msg(2, shape))

def value_info(name, elem, dims):
    return f_text(1, name) + f_msg(2, type_proto(elem, dims))

def graph_proto(nodes, name, inits, inputs, outputs, value_infos):
    return (b''.join(f_msg(1, n) for n in nodes)
            + f_text(2, name)
            + b''.join(f_msg(5, t) for t in inits)
            + b''.join(f_msg(11, v) for v in inputs)
            + b''.join(f_msg(12, v) for v in outputs)
            + b''.join(f_msg(13, v) for v in value_infos))

def model_proto(graph, opsets, producer='onnx-fixture', version='1'):
    body = f_int_always(1, 10) + f_text(2, producer) + f_text(3, version)
    body += f_msg(7, graph)
    for domain, v in opsets:
        body += f_msg(8, f_text(1, domain) + f_int_always(2, v))
    return body

# ---------------------------------------------------------------- the fixture --

# Integers held as floats, so every value below is exact in f32 and in f64 and
# the comparison against the CLI's five decimals is not a tolerance in disguise.
W = [1.0, 0.0, -1.0,
     2.0, 3.0, 0.0,
     0.0, 1.0, 4.0,
     -2.0, 0.0, 1.0]      # [4, 3]
B = [1.0, -2.0, 3.0]      # [3]
TWO = [2.0]
X = [1.0, 2.0, 3.0, 4.0]  # the tokens passed to `omni graph run`

def expected():
    """What the graph computes, from the operator definitions."""
    mm = [sum(X[k] * W[k * 3 + j] for k in range(4)) for j in range(3)]
    y0 = [mm[j] + B[j] for j in range(3)]           # Add        [1,3]
    y1 = [v * TWO[0] for v in y0]                   # Mul        [1,3]
    y2 = list(y0)                                   # Transpose+Reshape [3]
    y3 = [sum(y0)]                                  # ReduceSum  [1]
    y4 = y0 + y0                                    # Concat     [1,6]
    return [y0, y1, y2, y3, y4]

def native_model():
    nodes = [
        node('MatMul', ['x', 'w'], ['mm'], 'matmul'),
        node('Add', ['mm', 'b'], ['y0'], 'bias'),
        node('Constant', [], ['two'], 'two_k',
             [attr_tensor('value', f32_tensor('two_value', [1], TWO))]),
        node('Mul', ['y0', 'two'], ['y1'], 'double'),
        node('Transpose', ['y0'], ['t'], 'flip', [attr_ints('perm', [1, 0])]),
        node('Reshape', ['t', 'flat'], ['y2'], 'flatten'),
        node('ReduceSum', ['y0'], ['y3'], 'total',
             [attr_ints('axes', [1]), attr_int('keepdims', 0)]),
        node('Concat', ['y0', 'y0'], ['y4'], 'twice', [attr_int('axis', 1)]),
    ]
    inits = [
        f32_tensor('w', [4, 3], W),
        f32_tensor('b', [3], B),
        i64_tensor('flat', [1], [3]),
    ]
    graph = graph_proto(
        nodes, 'main', inits,
        [value_info('x', FLOAT, [1, 4])],
        [value_info('y0', FLOAT, [1, 3]),
         value_info('y1', FLOAT, [1, 3]),
         value_info('y2', FLOAT, [3]),
         value_info('y3', FLOAT, [1]),
         value_info('y4', FLOAT, [1, 6])],
        [value_info('mm', FLOAT, [1, 3]),
         value_info('t', FLOAT, [3, 1])])
    return model_proto(graph, [('', 13)])

def compat_model():
    """The other half of the mapping: nodes no single OMNI op means."""
    nodes = [
        node('Relu', ['x'], ['r'], 'relu'),
        node('LeakyRelu', ['r'], ['l'], 'leaky', [attr_float('alpha', 0.125)]),
        node('Slice', ['l', 'starts', 'ends'], ['s'], 'cut'),
        node('Identity', ['s'], ['y'], 'id'),
    ]
    inits = [i64_tensor('starts', [1], [0]), i64_tensor('ends', [1], [2])]
    graph = graph_proto(
        nodes, 'main', inits,
        [value_info('x', FLOAT, [4])],
        [value_info('y', FLOAT, [2])],
        [value_info('r', FLOAT, [4]),
         value_info('l', FLOAT, [4]),
         value_info('s', FLOAT, [2])])
    return model_proto(graph, [('', 13)])

def write(path, values_path):
    open(path, 'wb').write(native_model())
    with open(values_path, 'w') as f:
        for name, dims, vals in [('w', [4, 3], W), ('b', [3], B)]:
            hexed = ' '.join(struct.pack('<f', v).hex() for v in vals)
            f.write(f'{name} {hexed}\n')
        for i, out in enumerate(expected()):
            f.write('# result {} {}\n'.format(i, ' '.join(f'{v:.5f}' for v in out)))
    print(f'wrote {path} ({len(native_model())} bytes) and {values_path}')

# ----------------------------------------------------------------- the checks --

def check(omni, container, values_path):
    """Every initializer, through the container, against Python."""
    bad = 0
    for line in open(values_path):
        if line.startswith('#'):
            continue
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
    return bad

def check_run(output, values_path):
    """The values `omni graph run` printed, against Python's arithmetic."""
    heads = re.findall(r'head\s+\[([^\]]*)\]', output)
    want = [line.split()[3:] for line in open(values_path)
            if line.startswith('# result ')]
    if len(heads) != len(want):
        print(f'omni printed {len(heads)} result(s), python computed {len(want)}')
        return 1
    bad = 0
    for i, (h, w) in enumerate(zip(heads, want)):
        got = [x.strip().rstrip(',') for x in h.split(',') if x.strip() not in ('', '…')]
        w = w[:6]
        if got != w:
            print(f'result {i}: omni {got}, python {w}')
            bad += 1
        else:
            print(f'result {i}: {len(w)} value(s) agree')
    return bad

# --------------------------------------------------------- the reader, again --

def parse_model(b):
    """A ModelProto, as a plain dict, read with this file's own reader."""
    m = {'opsets': [], 'graph': None}
    for f, w, v in fields(b):
        if f == 1:
            m['ir_version'] = v
        elif f == 2:
            m['producer'] = v.decode()
        elif f == 7:
            m['graph'] = parse_graph(v)
        elif f == 8:
            d, ver = '', 0
            for g, _, x in fields(v):
                if g == 1:
                    d = x.decode()
                elif g == 2:
                    ver = x
            m['opsets'].append((d, ver))
    return m

def parse_graph(b):
    g = {'nodes': [], 'inits': [], 'inputs': [], 'outputs': [], 'value_info': [],
         'name': ''}
    for f, w, v in fields(b):
        if f == 1:
            g['nodes'].append(parse_node(v))
        elif f == 2:
            g['name'] = v.decode()
        elif f == 5:
            g['inits'].append(parse_tensor(v))
        elif f == 11:
            g['inputs'].append(parse_value_info(v))
        elif f == 12:
            g['outputs'].append(parse_value_info(v))
        elif f == 13:
            g['value_info'].append(parse_value_info(v))
    return g

def parse_node(b):
    n = {'in': [], 'out': [], 'name': '', 'op': '', 'domain': '', 'attrs': []}
    for f, w, v in fields(b):
        if f == 1:
            n['in'].append(v.decode())
        elif f == 2:
            n['out'].append(v.decode())
        elif f == 3:
            n['name'] = v.decode()
        elif f == 4:
            n['op'] = v.decode()
        elif f == 5:
            n['attrs'].append(parse_attr(v))
        elif f == 7:
            n['domain'] = v.decode()
    return n

def parse_attr(b):
    a = {'name': '', 'type': 0, 'i': 0, 'f': 0.0, 'ints': [], 's': b'', 't': None}
    for f, w, v in fields(b):
        if f == 1:
            a['name'] = v.decode()
        elif f == 2:
            a['f'] = struct.unpack('<f', v)[0]
        elif f == 3:
            a['i'] = v
        elif f == 4:
            a['s'] = v
        elif f == 5:
            a['t'] = parse_tensor(v)
        elif f == 8:
            i = 0
            while i < len(v):
                x, i = read_varint(v, i)
                a['ints'].append(x)
        elif f == 20:
            a['type'] = v
    return a

def parse_tensor(b):
    t = {'dims': [], 'dtype': 0, 'name': '', 'raw': b''}
    for f, w, v in fields(b):
        if f == 1:
            i = 0
            while i < len(v):
                x, i = read_varint(v, i)
                t['dims'].append(x)
        elif f == 2:
            t['dtype'] = v
        elif f == 8:
            t['name'] = v.decode()
        elif f == 9:
            t['raw'] = v
    return t

def parse_value_info(b):
    vi = {'name': '', 'dims': [], 'elem': 0}
    for f, w, v in fields(b):
        if f == 1:
            vi['name'] = v.decode()
        elif f == 2:
            for g, _, x in fields(v):
                if g != 1:
                    continue
                for h, _, y in fields(x):
                    if h == 1:
                        vi['elem'] = y
                    elif h == 2:
                        for _, _, d in fields(y):
                            dim = None
                            for k, _, z in fields(d):
                                dim = z if k == 1 else z.decode()
                            vi['dims'].append(dim)
    return vi

def compare(a_path, b_path):
    """Two ONNX files, structure first and bytes second."""
    a, b = open(a_path, 'rb').read(), open(b_path, 'rb').read()
    ma, mb = parse_model(a), parse_model(b)
    problems = []
    if ma['opsets'] != mb['opsets']:
        problems.append(f"opsets {ma['opsets']} vs {mb['opsets']}")
    ga, gb = ma['graph'], mb['graph']
    if ga['name'] != gb['name']:
        problems.append(f"graph name {ga['name']!r} vs {gb['name']!r}")
    for key_ in ('inputs', 'outputs', 'value_info'):
        if ga[key_] != gb[key_]:
            problems.append(f'{key_}: {ga[key_]} vs {gb[key_]}')
    if len(ga['nodes']) != len(gb['nodes']):
        problems.append(f"{len(ga['nodes'])} nodes vs {len(gb['nodes'])}")
    for x, y in zip(ga['nodes'], gb['nodes']):
        if x != y:
            problems.append(f"node {x['name'] or x['op']}: {x} vs {y}")
    if len(ga['inits']) != len(gb['inits']):
        problems.append(f"{len(ga['inits'])} initializers vs {len(gb['inits'])}")
    for x, y in zip(ga['inits'], gb['inits']):
        if x != y:
            problems.append(f"initializer {x['name']}: dims/dtype/bytes differ")
    for p in problems:
        print(f'  differs: {p}')
    if problems:
        return len(problems)
    print(f'  structure identical: {len(ga["nodes"])} node(s), '
          f'{len(ga["inits"])} initializer(s)')
    if a != b:
        print(f'  bytes differ: {len(a)} vs {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f'  first difference at byte {i}')
                break
        return 1
    print(f'  bytes identical: {len(a)}')
    return 0

def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == 'write':
        write(argv[2], argv[3])
        return 0
    if cmd == 'compat':
        open(argv[2], 'wb').write(compat_model())
        print(f'wrote {argv[2]} ({len(compat_model())} bytes)')
        return 0
    if cmd == 'check':
        return 1 if check(argv[2], argv[3], argv[4]) else 0
    if cmd == 'check-run':
        return 1 if check_run(open(argv[2]).read(), argv[3]) else 0
    if cmd == 'compare':
        return 1 if compare(argv[2], argv[3]) else 0
    print(f'unknown command {cmd}')
    return 2

if __name__ == '__main__':
    sys.exit(main(sys.argv))
