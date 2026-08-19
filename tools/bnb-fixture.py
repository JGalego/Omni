#!/usr/bin/env python3
"""Write a real bitsandbytes checkpoint, and what bitsandbytes says it means.

The other quantized importers in this crate are checked by dequantizing every
layer and comparing against arithmetic done in Python — arithmetic this
repository wrote, from the format's documentation. For bitsandbytes there is
something better available: the library that *defines* the format is a pip
install away, and it will both produce the checkpoint and tell you what the
numbers are.

So this writes three things:

    <out>/model.safetensors   a checkpoint quantized by bitsandbytes
    <out>/expect.json         what `bitsandbytes.dequantize_4bit` says it is
    <out>/meta.json           the shapes and the quant state, for the checker

and `omni import bitsandbytes` then has to agree with `expect.json`. Nothing in
the comparison comes from OMNI, which is the point: NF4 is a codebook, a block
size, a nibble order and — under double quantization — a second codebook, a
second block size and an offset. Six chances to be self-consistent and wrong.

    tools/bnb-fixture.py <out-dir> [--double] [--int8] [--quant nf4|fp4]

Requires torch, safetensors and bitsandbytes. Like the rest of `tools/`, this
lives outside the zero-dependency boundary on purpose.
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--blocksize", type=int, default=64)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--double", action="store_true",
                    help="quantize the block scales as well")
    ap.add_argument("--int8", action="store_true",
                    help="also write an LLM.int8 weight beside the 4-bit one")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    try:
        import torch
        import bitsandbytes.functional as F
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"needs torch, safetensors and bitsandbytes: {e}", file=sys.stderr)
        return 2

    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    w = torch.randn(a.rows, a.cols)
    q, state = F.quantize_4bit(w, blocksize=a.blocksize, quant_type=a.quant,
                               compress_statistics=a.double)
    # The library's own dequantization is the expectation. Not the original `w`:
    # quantizing loses information, and what the importer must reproduce is the
    # value the *checkpoint* holds, not the value it came from.
    ref = F.dequantize_4bit(q, state).float().reshape(-1).tolist()

    tensors = {}
    for k, v in state.as_dict(packed=True).items():
        # `as_dict` names the weight itself `weight`; the rest are its
        # companions, already suffixed the way a real checkpoint has them.
        tensors[f"w.{k}" if k != "weight" else "w"] = v
    tensors["w"] = q.reshape(-1, 1)
    expect = {"w": ref}
    meta = {
        "w": {
            "quant": a.quant,
            "blocksize": a.blocksize,
            "double": bool(a.double),
            "shape": [a.rows, a.cols],
        }
    }

    if a.int8:
        import bitsandbytes as bnb
        import bitsandbytes.nn as bnn

        lin = bnn.Linear8bitLt(a.cols, a.rows, bias=False, has_fp16_weights=False)
        lin.weight = bnb.nn.Int8Params(torch.randn(a.rows, a.cols),
                                       requires_grad=False)
        sd = lin.to("cpu").state_dict()
        tensors["m.weight"] = sd["weight"]
        tensors["m.SCB"] = sd["SCB"].float()
        if "weight_format" in sd:
            tensors["m.weight_format"] = sd["weight_format"]
        # LLM.int8's stored value is `q * SCB / 127`, and that is what the
        # importer has to reproduce — again the checkpoint's value, not the
        # tensor it was made from.
        deq = sd["weight"].float() * sd["SCB"].float().reshape(-1, 1) / 127.0
        expect["m.weight"] = deq.reshape(-1).tolist()
        meta["m.weight"] = {"quant": "int8", "shape": [a.rows, a.cols]}

    # A tensor nothing quantized, so the plain path is exercised too.
    tensors["norm.weight"] = torch.ones(a.rows)

    save_file(tensors, os.path.join(a.out, "model.safetensors"))
    json.dump(expect, open(os.path.join(a.out, "expect.json"), "w"))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))

    import bitsandbytes
    print(f"{a.out}: bitsandbytes {bitsandbytes.__version__}, {a.quant}, "
          f"blocksize {a.blocksize}"
          f"{', double-quantized' if a.double else ''}"
          f"{', plus int8' if a.int8 else ''}; "
          f"{len(expect)} expectation(s), {len(tensors)} tensor(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
