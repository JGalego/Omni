#!/usr/bin/env python3
"""Write a small decoder's weights with the `safetensors` library.

The corpus needs a model whose weights are plausible floats rather than
plausible *bytes*. `omni example --graph` writes pseudo-random bytes, which read
as bf16 are NaNs and values near 1e24; a graph over them evaluates to all zeros,
and a comparison against all zeros cannot fail. So this writes real weights, with
the real library, for `tools/torch-reference.py` to run and OMNI to be checked
against.

    tools/decoder-weights.py <out.safetensors> [--layers 2] [--hidden 64]
        [--heads 4] [--kv-heads 2] [--vocab 256] [--seed 7]

Shapes match what §07.5's `transformer.decoder` synthesizer looks for. Magnitudes
are the point: 0.05 standard deviation for projections and near-one norm gains, so
logits land around 0.1–1 rather than saturating bf16 or vanishing into it.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"needs torch and safetensors: {e}", file=sys.stderr)
        return 2

    if a.hidden % a.heads:
        print(f"hidden {a.hidden} is not divisible by heads {a.heads}",
              file=sys.stderr)
        return 2
    head_dim = a.hidden // a.heads
    kv = a.kv_heads * head_dim

    torch.manual_seed(a.seed)

    def w(*shape):
        return (torch.randn(*shape) * 0.05).to(torch.bfloat16)

    t = {
        "model.embed_tokens.weight": w(a.vocab, a.hidden),
        "lm_head.weight": w(a.vocab, a.hidden),
    }
    for i in range(a.layers):
        p = f"model.layers.{i}"
        # The norm gain is f32 and near one, as a trained model's is; the graph
        # casts it to bf16 itself, and letting it do that is part of what is
        # being compared.
        t[f"{p}.norm.weight"] = (
            torch.ones(a.hidden) + torch.randn(a.hidden) * 0.02).to(torch.float32)
        t[f"{p}.attn.q_proj.weight"] = w(a.hidden, a.hidden)
        t[f"{p}.attn.k_proj.weight"] = w(kv, a.hidden)
        t[f"{p}.attn.v_proj.weight"] = w(kv, a.hidden)
        t[f"{p}.attn.o_proj.weight"] = w(a.hidden, a.hidden)

    save_file(t, a.out)
    print(f"{a.out}: {len(t)} tensors, {a.layers} layer(s), hidden {a.hidden}, "
          f"{a.heads} head(s) over {a.kv_heads} kv head(s), vocab {a.vocab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
