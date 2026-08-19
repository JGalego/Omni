#!/usr/bin/env python3
"""Turn NIST's ACVP vectors for ML-DSA into the fixtures `mldsa.rs` is tested on.

Every other cryptographic primitive in this crate has somebody else's
implementation to disagree with — digests against `hashlib`, ES256 against
`cryptography`. ML-DSA has something better: NIST publishes the known-answer
vectors for FIPS 204 itself, which is not a second implementation but the
authority both implementations are trying to match.

That distinction matters more for a signature scheme than anywhere else in this
codebase. ML-DSA has rejection sampling, a deterministic nonce, a hint mechanism
and four separate little-endian bit packings, and a signer and verifier that
share a misreading of any of them agree with each other perfectly and with
nobody else. A round-trip test would pass. The vectors are what make it possible
to be wrong and find out.

    tools/mldsa-vectors.py fetch     # download the ACVP JSON into a cache
    tools/mldsa-vectors.py build     # write reference/omni-core/tests/vectors/mldsa/
    tools/mldsa-vectors.py           # both

Provenance: https://github.com/usnistgov/ACVP-Server, gen-val/json-files/
ML-DSA-{keyGen,sigGen,sigVer}-FIPS204/internalProjection.json. The fixtures are
committed because CI has no network, and this script exists so anyone can
re-derive them from the source rather than taking the committed bytes on trust.

Only the cases this build implements are kept: the external signature interface,
`pure` (not pre-hashed) messages, deterministic signing, and no external mu.
Everything filtered out is counted and reported, because a fixture set that
quietly dropped the hard cases would be worse than no fixture set.
"""

import json
import os
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/"
        "gen-val/json-files")
CACHE = os.environ.get("ACVP_CACHE", "/tmp/acvp-mldsa")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "reference", "omni-core", "tests", "vectors", "mldsa")

# How many of each kind to keep, per parameter set. The whole ACVP set is 20 MB
# of JSON and the fixtures are hex, so this is a real trade rather than a detail:
# these counts exercise every code path in all three parameter sets while keeping
# the committed fixtures proportionate to the rest of `tests/vectors`. The
# numbers are stated here rather than buried so trimming further is a visible
# decision, and the build reports how many cases were left behind.
KEEP_KEYGEN = 3
KEEP_SIGGEN = 2
KEEP_SIGVER = 4

KINDS = ["keyGen", "sigGen", "sigVer"]
SETS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    for kind in KINDS:
        path = os.path.join(CACHE, f"{kind}.json")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"{kind}: cached")
            continue
        url = f"{BASE}/ML-DSA-{kind}-FIPS204/internalProjection.json"
        print(f"{kind}: fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        open(path, "wb").write(data)
        print(f"{kind}: {len(data)} bytes")
    return 0


def load(kind):
    return json.load(open(os.path.join(CACHE, f"{kind}.json"), encoding="utf-8"))


def implemented(group):
    """The subset of the ACVP surface this build claims."""
    return (group.get("signatureInterface", "external") == "external"
            and group.get("preHash", "pure") == "pure"
            and not group.get("externalMu", False)
            and group.get("deterministic", True))


def build():
    os.makedirs(OUT, exist_ok=True)
    total = {}

    # keyGen: seed -> (pk, sk). Exercises seed expansion, ExpandA, ExpandS, the
    # NTT, Power2Round and both key encodings.
    lines, kept, skipped = [], 0, 0
    for g in load("keyGen")["testGroups"]:
        if g["parameterSet"] not in SETS:
            skipped += len(g["tests"])
            continue
        for t in g["tests"][:KEEP_KEYGEN]:
            lines += [f"set {g['parameterSet']}", f"seed {t['seed']}",
                      f"pk {t['pk']}", f"sk {t['sk']}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - KEEP_KEYGEN)
    write("keygen.txt", lines)
    total["keyGen"] = (kept, skipped)

    # sigGen: (sk, message, context) -> signature, deterministic only.
    lines, kept, skipped = [], 0, 0
    for g in load("sigGen")["testGroups"]:
        if g["parameterSet"] not in SETS or not implemented(g):
            skipped += len(g["tests"])
            continue
        for t in g["tests"][:KEEP_SIGGEN]:
            lines += [f"set {g['parameterSet']}", f"sk {t['sk']}",
                      f"msg {t['message']}", f"ctx {t.get('context', '')}",
                      f"sig {t['signature']}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - KEEP_SIGGEN)
    write("siggen.txt", lines)
    total["sigGen"] = (kept, skipped)

    # sigVer: (pk, message, context, signature) -> pass/fail, with NIST's own
    # reason. The failing cases are the valuable half: they are what proves
    # `verify` rejects rather than merely accepts.
    lines, kept, skipped = [], 0, 0
    for g in load("sigVer")["testGroups"]:
        if g["parameterSet"] not in SETS or not implemented(g):
            skipped += len(g["tests"])
            continue
        chosen = pick_mixed(g["tests"], KEEP_SIGVER)
        for t in chosen:
            lines += [f"set {g['parameterSet']}", f"pk {t['pk']}",
                      f"msg {t['message']}", f"ctx {t.get('context', '')}",
                      f"sig {t['signature']}",
                      f"expect {'pass' if t['testPassed'] else 'fail'}",
                      f"reason {t.get('reason', '-')}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - len(chosen))
    write("sigver.txt", lines)
    total["sigVer"] = (kept, skipped)

    print()
    for kind, (kept, skipped) in total.items():
        print(f"{kind}: kept {kept}, skipped {skipped}")
    return 0


def pick_mixed(tests, n):
    """Keep both outcomes. Taking the first n would take almost only failures,
    since ACVP orders them that way, and a fixture set that cannot observe a
    valid signature being accepted is not testing verification."""
    good = [t for t in tests if t["testPassed"]]
    bad = [t for t in tests if not t["testPassed"]]
    half = max(1, n // 2)
    out = good[:half] + bad[:n - min(half, len(good))]
    return out[:n]


def write(name, lines):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generated by tools/mldsa-vectors.py from NIST ACVP-Server.\n"
                "# Source: gen-val/json-files/ML-DSA-*-FIPS204/internalProjection.json\n"
                "# Do not edit; re-run the script instead.\n")
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path)} bytes)")


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    if what in ("fetch", "all"):
        if fetch() != 0:
            return 1
    if what in ("build", "all"):
        return build()
    if what not in ("fetch", "build", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
