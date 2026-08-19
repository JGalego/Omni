#!/usr/bin/env python3
"""A PyTorch reference for the §07.5 synthesized decoder, for Gate 2's first row.

Gate 2 asks that ten architecture families execute *and* that their outputs match
the source framework within a declared tolerance. This repository could do the
first half and not the second, for a reason that was honest and is no longer
true: there was no source framework here, so `tools/corpus.py families` took the
framework's answers as data and CI fed it this build's own answers — a comparison
that, as the CI comment admitted, could not fail on arithmetic.

It was worse than that, and this script is how it came out. The case CI used was
`omni example --graph`, whose weights are pseudo-random *bytes*; read as bf16 they
are NaNs and values around 1e24, and the graph over them evaluates to all zeros.
So the check compared 768 zeros against 768 zeros and passed. Every arithmetic
error in §04.7's evaluator was invisible to it.

This writes the other side. It builds the same computation in PyTorch from the
weights OMNI exports, runs it, and emits `<name>.expect.json` in the shape
`omni graph run --json` writes, so `corpus.py families` compares OMNI against a
framework that shares no code with it.

    tools/torch-reference.py <model.omni> --omni ./omni --tokens 1,2,3 \\
        --out cases/decoder

Requires torch and safetensors. Neither is a dependency of anything in
`reference/` — this is an instrument, like `tools/corpus.py`, and lives outside
the zero-dependency boundary on purpose.

What is deliberately *not* done here: no attempt to match OMNI's intermediate
rounding exactly. The graph declares bf16 at every step and this computes in
float32, rounding to bfloat16 at each op boundary the way the graph's types say
to. Two implementations of bf16 arithmetic that accumulate differently will not
agree to the last bit, and pretending otherwise by copying OMNI's accumulation
order would make the comparison circular again. The tolerance is declared instead,
and it is declared for bf16 rather than borrowed from f32.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile


def bf(x):
    """Round to bfloat16 and come back, mirroring a bf16-typed edge in the graph."""
    import torch

    return x.to(torch.bfloat16).to(torch.float32)


def rms_norm(x, weight, eps):
    """§07's `omni.nn/norm` with kind="rms": no mean subtraction, no bias."""
    import torch

    var = x.pow(2).mean(-1, keepdim=True)
    return bf(x * torch.rsqrt(var + eps) * weight)


def rope(x, theta=10000.0, interleaved=False):
    """`omni.nn/rope`. interleaved=false is the half-split rotation: dimension i
    pairs with i + d/2, which is what the graph's `interleaved=false` selects and
    is not the same thing as pairing (0,1), (2,3), ... — a wrong choice here
    produces plausible numbers and a different model."""
    import torch

    s, h, d = x.shape
    half = d // 2
    pos = torch.arange(s, dtype=torch.float32).unsqueeze(1)
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos * inv                      # (s, half)
    cos = torch.cos(ang).unsqueeze(1)    # (s, 1, half)
    sin = torch.sin(ang).unsqueeze(1)
    if interleaved:
        raise SystemExit("interleaved RoPE is not what the synthesizer emits")
    x1, x2 = x[..., :half], x[..., half:]
    return bf(torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1))


def attention(q, k, v, scale, causal, kv_groups):
    """`omni.nn/attention@2`. q is (heads, s, d); k and v are (kv_heads, s, d).

    `kv_groups` is the number of kv heads, so each of them serves
    `heads / kv_groups` query heads, consecutively — query head h reads kv head
    h // (heads / kv_groups). The other plausible mapping, h % kv_groups, is a
    different model that runs perfectly well, which is why this is spelled out.
    """
    import torch

    heads = q.shape[0]
    per = heads // kv_groups
    out = []
    for h in range(heads):
        kv = h // per
        scores = bf(q[h] @ k[kv].transpose(0, 1) * scale)
        if causal:
            s = scores.shape[0]
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        p = bf(torch.softmax(scores.to(torch.float32), dim=-1))
        out.append(bf(p @ v[kv]))
    return torch.stack(out, 0)


def forward(w, tokens, layers, heads, kv_heads, eps):
    """The graph of §07.5's `transformer.decoder`, op for op as `omni graph`
    prints it: embed, then per layer a pre-norm, GQA attention and a residual on
    the pre-norm input, then the language-model head. No MLP and no final norm —
    the synthesizer emits neither, and adding one here would be inventing a model
    OMNI is not running."""
    import torch

    x = bf(w["model.embed_tokens.weight"][tokens])          # (s, hidden)
    s, hidden = x.shape
    head_dim = hidden // heads
    for l in range(layers):
        p = f"model.layers.{l}"
        xn = rms_norm(x, bf(w[f"{p}.norm.weight"]), eps)
        q = bf(xn @ w[f"{p}.attn.q_proj.weight"].transpose(0, 1))
        k = bf(xn @ w[f"{p}.attn.k_proj.weight"].transpose(0, 1))
        v = bf(xn @ w[f"{p}.attn.v_proj.weight"].transpose(0, 1))
        q = rope(q.reshape(s, heads, head_dim))
        k = rope(k.reshape(s, kv_heads, head_dim))
        v = v.reshape(s, kv_heads, head_dim)
        a = attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                      scale=1.0 / math.sqrt(head_dim), causal=True,
                      kv_groups=kv_heads)
        a = a.transpose(0, 1).reshape(s, hidden)
        a = bf(a @ w[f"{p}.attn.o_proj.weight"].transpose(0, 1))
        x = bf(x + a)
    return bf(x @ w["lm_head.weight"].transpose(0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--omni", default="omni")
    ap.add_argument("--tokens", default="1,2,3")
    ap.add_argument("--out", required=True,
                    help="path prefix; writes <out>.omni, .inputs.json, .expect.json")
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1e-5)
    a = ap.parse_args()

    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as e:
        print(f"needs torch and safetensors: {e}", file=sys.stderr)
        return 2

    tokens = [int(t) for t in a.tokens.split(",")]
    tmp = tempfile.mkdtemp(prefix="omni-torch-")
    st = os.path.join(tmp, "w.safetensors")
    # `--allow-lossy` because the structured model card cannot survive
    # safetensors' flat string map; the tensors, which are all this needs, do.
    r = subprocess.run([a.omni, "export", "safetensors", a.model, "-o", st,
                        "--allow-lossy"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr, file=sys.stderr)
        return 1
    raw = load_file(st)
    w = {k: v.to(torch.float32) for k, v in raw.items()}

    logits = forward(w, tokens, a.layers, a.heads, a.kv_heads, a.eps)
    flat = [float(x) for x in logits.reshape(-1)]

    # bf16 carries about three significant decimal digits, and the two
    # implementations accumulate differently. The default is relative to the
    # magnitude actually present rather than a constant, because a constant that
    # passes on logits near 0.1 would be meaningless on logits near 100.
    scale = max(abs(x) for x in flat) or 1.0
    tol = a.tolerance if a.tolerance is not None else max(2e-2 * scale, 1e-3)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run(["cp", a.model, f"{a.out}.omni"], check=True)
    json.dump({"tokens": tokens, "tolerance": tol},
              open(f"{a.out}.inputs.json", "w"))
    json.dump({"source": f"pytorch {torch.__version__}",
               "note": "computed by tools/torch-reference.py, which shares no "
                       "code with the OMNI evaluator",
               "returned": [{"shape": [1, len(tokens), logits.shape[-1]],
                             "data": flat}]},
              open(f"{a.out}.expect.json", "w"))
    print(f"{a.out}.expect.json: {len(flat)} values from pytorch "
          f"{torch.__version__}, |max| {scale:.4f}, tolerance {tol:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
