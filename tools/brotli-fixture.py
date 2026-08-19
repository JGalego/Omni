#!/usr/bin/env python3
"""Compress a corpus with libbrotli, for the decoder in `brotli.rs` to reproduce.

Brotli is one library, not a spec with many implementations, so the honest check
is differential against that library: it compresses, and OMNI's decoder has to
reproduce the original byte for byte. The corpus is chosen to exercise the parts
of RFC 7932 a lazy decoder gets wrong — the static dictionary and its transforms
above all, which is why there is English text in here and not only random bytes.

    tools/brotli-fixture.py <out-dir>

Writes, for each case, `<name>.br` (the compressed stream) and `<name>.raw`
(the original), plus `manifest.txt` listing them. `brotli.rs`'s vector test reads
that directory. Requires the `brotli` package; it lives outside the crate's
zero-dependency boundary like the rest of `tools/`.
"""

import os
import sys


def corpus():
    cases = {}
    # Text: the case that forces static-dictionary references and transforms,
    # because brotli's dictionary is English and web markup.
    cases["prose"] = (
        "The quick brown fox jumps over the lazy dog. " * 40
    ).encode()
    cases["html"] = (
        '<!DOCTYPE html><html><head><title>test</title></head>'
        '<body><p>the time has come, and this is the content of the page</p>'
        '<a href="http://example.com/">a link</a></body></html>' * 20
    ).encode()
    cases["json"] = (
        '{"name":"model","version":1,"layers":['
        + ",".join('{"kind":"linear","in":4096,"out":4096}' for _ in range(200))
        + "]}"
    ).encode()
    # Highly repetitive: long backward references and the distance ring.
    cases["repeat"] = (b"abcdefgh" * 4096)
    # Incompressible: forces uncompressed or near-uncompressed meta-blocks.
    import random
    rng = random.Random(20260101)
    cases["random"] = bytes(rng.randrange(256) for _ in range(20000))
    # Structured binary: what a tensor actually looks like.
    cases["floats"] = b"".join(
        int(x).to_bytes(4, "little")
        for x in (rng.random() * 1e6 for _ in range(4000))
    )
    cases["empty"] = b""
    cases["onebyte"] = b"Q"
    # A whole-dictionary probe: text likely to reference many transforms.
    cases["mixed"] = (
        "running runner runs ran. Testing tested tester tests. "
        "The DATA and the data, in UPPER and lower, at time.com and Time.org. "
    ).encode() * 30
    return cases


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    try:
        import brotli
    except ImportError:
        print("needs the `brotli` package", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    names = []
    for name, data in sorted(corpus().items()):
        # A spread of quality levels, since the encoder's choices — block
        # splitting, context modes, dictionary use — vary the bitstream the
        # decoder must handle.
        for q in (1, 5, 11):
            comp = brotli.compress(data, quality=q)
            base = f"{name}-q{q}"
            open(os.path.join(out, f"{base}.br"), "wb").write(comp)
            open(os.path.join(out, f"{base}.raw"), "wb").write(data)
            names.append(f"{base}\t{len(data)}\t{len(comp)}")
    open(os.path.join(out, "manifest.txt"), "w").write("\n".join(names) + "\n")
    import brotli as b
    print(f"{out}: {len(names)} cases from brotli {getattr(b, '__version__', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
