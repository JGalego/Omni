#!/usr/bin/env python3
"""Compress arrays with zfpy, for the decoder in `zfp.rs` to reproduce.

zfp is one library rather than a specification with a field of independent
implementations, so the honest check is differential against that library: it
compresses, and OMNI's decoder must reproduce what *its own* decompressor
produces. Not the original array — zfp is lossy, and what a reader has to
reproduce is the value the stream holds, not the value it came from.

The corpus spans the axes that change the bitstream rather than just its size:
one, two and three dimensions; `float32` and `float64`; the fixed-rate,
fixed-precision and fixed-accuracy modes, which set the four stream parameters
differently; extents that are not multiples of four, so the partial-block path
is exercised; and arrays containing zero blocks, which take the one-bit form.

    tools/zfp-fixture.py <out-dir>

Writes `<name>.zfp` (the compressed stream) and `<name>.raw` (what zfpy's
decompressor makes of it, as little-endian scalars), plus `manifest.txt`.
Requires the `zfpy` package; like the rest of `tools/` it lives outside the
crate's zero-dependency boundary.
"""

import os
import sys


def corpus(np):
    """Arrays chosen for the paths a decoder gets wrong, not for size."""
    rng = np.random.default_rng(20260101)
    cases = {}

    # Smooth data, which is what zfp's transform is for: high compression and
    # every bit plane in use.
    cases["smooth1d"] = np.linspace(-3.0, 7.0, 64).astype(np.float32)
    cases["smooth2d"] = np.outer(
        np.linspace(0, 1, 16), np.linspace(0, 2, 16)
    ).astype(np.float32)
    cases["smooth3d"] = (
        np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8) / 97.0
    )
    # Noise: nothing to predict, so the coder spends its whole budget.
    cases["noise1d"] = rng.standard_normal(64).astype(np.float32)
    cases["noise2d"] = rng.standard_normal((12, 12)).astype(np.float32)
    cases["noise3d"] = rng.standard_normal((8, 8, 8)).astype(np.float32)
    # Extents that are not multiples of four: the partial-block path.
    cases["partial1d"] = rng.standard_normal(37).astype(np.float32)
    cases["partial2d"] = rng.standard_normal((7, 13)).astype(np.float32)
    cases["partial3d"] = rng.standard_normal((5, 6, 7)).astype(np.float32)
    # All zeros, and mostly zeros: the one-bit block form.
    cases["zeros2d"] = np.zeros((16, 16), dtype=np.float32)
    sparse = np.zeros((16, 16), dtype=np.float32)
    sparse[3, 4] = 12.5
    sparse[11, 9] = -0.25
    cases["sparse2d"] = sparse
    # Big magnitudes and tiny ones together, so the per-block exponent varies.
    spread = np.zeros((8, 8), dtype=np.float32)
    spread[:4, :4] = 1e18
    spread[4:, 4:] = 1e-18
    cases["spread2d"] = spread
    # float64, across the same shapes: a different exponent width and integer
    # width, which is the other half of the block codec.
    cases["d_smooth1d"] = np.linspace(-3.0, 7.0, 64)
    cases["d_noise2d"] = rng.standard_normal((12, 12))
    cases["d_noise3d"] = rng.standard_normal((8, 8, 8))
    cases["d_partial2d"] = rng.standard_normal((7, 13))
    return cases


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    try:
        import numpy as np
        import zfpy
    except ImportError as e:
        print(f"needs numpy and zfpy: {e}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    # The three lossy modes. Each sets a different pair of the four stream
    # parameters, and the decoder has to read all four out of the mode field.
    modes = [
        ("r8", dict(rate=8)),
        ("r16", dict(rate=16)),
        ("p14", dict(precision=14)),
        ("p22", dict(precision=22)),
        ("a1e-3", dict(tolerance=1e-3)),
        ("a1e-6", dict(tolerance=1e-6)),
    ]

    names = []
    for case, arr in sorted(corpus(np).items()):
        for tag, kw in modes:
            comp = zfpy.compress_numpy(arr, **kw)
            # zfpy's own decompression is the expectation: what the *stream*
            # holds, which is what a reader must reproduce.
            back = zfpy.decompress_numpy(comp)
            assert back.dtype == arr.dtype, (case, back.dtype, arr.dtype)
            assert back.shape == arr.shape, (case, back.shape, arr.shape)
            base = f"{case}-{tag}"
            open(os.path.join(out, f"{base}.zfp"), "wb").write(comp)
            raw = np.ascontiguousarray(back).tobytes()
            open(os.path.join(out, f"{base}.raw"), "wb").write(raw)
            names.append(
                f"{base}\t{arr.dtype}\t{'x'.join(str(n) for n in arr.shape)}"
                f"\t{len(raw)}\t{len(comp)}"
            )

    open(os.path.join(out, "manifest.txt"), "w").write("\n".join(names) + "\n")
    print(f"{out}: {len(names)} streams from zfpy "
          f"{getattr(zfpy, '__version__', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
