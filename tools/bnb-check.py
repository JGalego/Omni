#!/usr/bin/env python3
"""Compare an imported bitsandbytes container against what bitsandbytes said.

`tools/bnb-fixture.py` writes a checkpoint and the library's own dequantization
of it; `omni import bitsandbytes` turns the checkpoint into a container. This
reads the container's values back and checks them against the library's.

    tools/bnb-check.py <container.omni> <fixture-dir> [--tolerance 3e-7]

Values are read through the C ABI, not off `omni cat`, because `cat` prints six
decimals and the difference being measured is smaller than that.

The tolerance is not zero, and the reason is known rather than tolerated:
bitsandbytes reconstructs a block's scale in float32, and OMNI's evaluator works
in float64 and carries the extra precision into the product. Emulating the
library's rounding in Python reproduces its output *exactly*, which is what pins
the explanation — so the residue is one float32 ULP of OMNI being the more
precise of the two, and not an error in either.
"""

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("fixture")
    ap.add_argument("--tolerance", type=float, default=3e-7)
    ap.add_argument("--lib", default=os.environ.get("OMNI_LIB"))
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "bindings", "python"))
    if a.lib:
        os.environ["OMNI_LIB"] = a.lib
    import omni_ffi

    expect = json.load(open(os.path.join(a.fixture, "expect.json")))
    model = omni_ffi.open(a.container)

    worst = 0.0
    for name, want in sorted(expect.items()):
        got = list(model[name].values())
        if len(got) != len(want):
            print(f"{name}: {len(got)} values, expected {len(want)}", file=sys.stderr)
            return 1
        w = max(abs(x - y) for x, y in zip(got, want))
        worst = max(worst, w)
        print(f"  {name}: {len(want)} values, worst |diff| {w:.3e}")

    if worst > a.tolerance:
        print(f"worst difference {worst:.3e} exceeds {a.tolerance:.0e}, which is "
              f"more than one float32 ULP — this is not a rounding residue",
              file=sys.stderr)
        return 1
    print(f"agrees with bitsandbytes to {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
