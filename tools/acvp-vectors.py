#!/usr/bin/env python3
"""Turn NIST's ACVP vectors into the fixtures `mldsa.rs` and `slhdsa.rs` are tested on.

Every other cryptographic primitive in this crate has somebody else's
implementation to disagree with — digests against `hashlib`, ES256 against
`cryptography`. ML-DSA has something better: NIST publishes the known-answer
vectors for FIPS 204 itself, which is not a second implementation but the
authority both implementations are trying to match.

That distinction matters more for a signature scheme than anywhere else in this
codebase. ML-DSA has rejection sampling, a deterministic nonce, a hint mechanism
and four separate little-endian bit packings; SLH-DSA has six hash functions
distinguished only by a 32-byte address whose every field is hashed, and two
places where the order the children of a node are concatenated in decides the
answer. In both, a signer and verifier that share a misreading agree with each
other perfectly and with nobody else. A round-trip test would pass. The vectors
are what make it possible to be wrong and find out.

    tools/acvp-vectors.py fetch [ml-dsa|slh-dsa]
    tools/acvp-vectors.py build [ml-dsa|slh-dsa]
    tools/acvp-vectors.py                          # fetch and build both

Provenance: https://github.com/usnistgov/ACVP-Server, gen-val/json-files/
{ML-DSA,SLH-DSA}-{keyGen,sigGen,sigVer}-FIPS{204,205}/internalProjection.json.
The fixtures are committed because CI has no network, and this script exists so
anyone can re-derive them from the source rather than taking the committed bytes
on trust.

Only the cases these builds implement are kept: for ML-DSA the external
signature interface, `pure` (not pre-hashed) messages, deterministic signing and
no external mu; for SLH-DSA the SHAKE parameter sets, deterministic signing and
the same external pure interface. Everything filtered out is counted and
reported, because a fixture set that quietly dropped the hard cases would be
worse than no fixture set.
"""

import json
import os
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/"
        "gen-val/json-files")
CACHE = os.environ.get("ACVP_CACHE", "/tmp/acvp-cache")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "reference", "omni-core", "tests", "vectors")

# How many of each kind to keep, per parameter set. The whole ACVP set is tens of
# megabytes of JSON and the fixtures are hex, so this is a real trade rather than
# a detail: these counts exercise every code path in every parameter set while
# keeping the committed fixtures proportionate to the rest of `tests/vectors`.
# The numbers are stated here rather than buried so trimming further is a visible
# decision, and the build reports how many cases were left behind.
#
# SLH-DSA's are smaller because its cost is not proportionate to its coverage.
# Key generation for an `s` set builds a top XMSS tree of 512 WOTS+ key pairs —
# roughly a million SHAKE calls, which is a minute in a debug build — and a second
# vector for the same set re-walks the same code with different bytes. So one per
# set, which is all the tree arithmetic needs, and `f` sets for signing, which is
# where the `s`/`f` axis costs two orders of magnitude.
KEEP = {
    "ml-dsa": {"keyGen": 3, "sigGen": 2, "sigVer": 4},
    "slh-dsa": {"keyGen": 1, "sigGen": 2, "sigVer": 3},
}

ALGS = {
    "ml-dsa": {
        "dir": "mldsa",
        "acvp": "ML-DSA",
        "fips": "FIPS204",
        "sets": ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"],
    },
    "slh-dsa": {
        "dir": "slhdsa",
        "acvp": "SLH-DSA",
        "fips": "FIPS205",
        # The SHAKE family only. The SHA2 sets use a different address encoding
        # and an MGF1 construction, and this build refuses them by name rather
        # than half-implementing a second family.
        "sets": [
            "SLH-DSA-SHAKE-128s", "SLH-DSA-SHAKE-128f",
            "SLH-DSA-SHAKE-192s", "SLH-DSA-SHAKE-192f",
            "SLH-DSA-SHAKE-256s", "SLH-DSA-SHAKE-256f",
        ],
        # Signing is where the parameter sets differ in cost by two orders of
        # magnitude, and a signature is also where the fixtures get large: a
        # 256f signature is 49 856 bytes, which is 99 712 hex digits. So `sigGen`
        # keeps the one set that is fast to sign and small to store, while
        # `sigVer` — which only verifies, and verification is cheap for every set
        # — keeps both 128 sets, including the `s` one whose signing is skipped.
        "siggen_sets": ["SLH-DSA-SHAKE-128f"],
        "sigver_sets": ["SLH-DSA-SHAKE-128s", "SLH-DSA-SHAKE-128f"],
    },
}

KINDS = ["keyGen", "sigGen", "sigVer"]


def cache_path(alg, kind):
    return os.path.join(CACHE, f"{ALGS[alg]['acvp']}-{kind}.json")


def fetch(alg):
    os.makedirs(CACHE, exist_ok=True)
    a = ALGS[alg]
    for kind in KINDS:
        path = cache_path(alg, kind)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"{alg} {kind}: cached")
            continue
        url = f"{BASE}/{a['acvp']}-{kind}-{a['fips']}/internalProjection.json"
        print(f"{alg} {kind}: fetching {url}")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        open(path, "wb").write(data)
        print(f"{alg} {kind}: {len(data)} bytes")
    return 0


def load(alg, kind):
    return json.load(open(cache_path(alg, kind), encoding="utf-8"))


def implemented(alg, group, kind=None):
    """The subset of the ACVP surface this build claims, and — for the kinds where
    cost or fixture size forces a choice — the subset this script keeps. Which
    sets were dropped is reported by the caller rather than left implicit."""
    allowed = ALGS[alg].get(f"{kind}_sets".lower()) if kind else None
    allowed = allowed or ALGS[alg]["sets"]
    if group["parameterSet"] not in allowed:
        return False
    return (group.get("signatureInterface", "external") == "external"
            and group.get("preHash", "pure") == "pure"
            and not group.get("externalMu", False)
            and group.get("deterministic", True))


def pick_mixed(tests, n):
    """Keep both outcomes. Taking the first n would take almost only failures,
    since ACVP orders them that way, and a fixture set that cannot observe a
    valid signature being accepted is not testing verification."""
    good = [t for t in tests if t["testPassed"]]
    bad = [t for t in tests if not t["testPassed"]]
    half = max(1, n // 2)
    out = good[:half] + bad[:n - min(half, len(good))]
    return out[:n]


def build(alg):
    out_dir = os.path.join(REF, ALGS[alg]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    keep = KEEP[alg]
    total = {}

    # keyGen. For ML-DSA one seed expands to both keys; SLH-DSA takes its three
    # seeds separately, as FIPS 205 and the vectors both present them.
    lines, kept, skipped = [], 0, 0
    for g in load(alg, "keyGen")["testGroups"]:
        if g["parameterSet"] not in ALGS[alg]["sets"]:
            skipped += len(g["tests"])
            continue
        for t in g["tests"][:keep["keyGen"]]:
            lines.append(f"set {g['parameterSet']}")
            if alg == "ml-dsa":
                lines.append(f"seed {t['seed']}")
            else:
                lines += [f"skSeed {t['skSeed']}", f"skPrf {t['skPrf']}",
                          f"pkSeed {t['pkSeed']}"]
            lines += [f"pk {t['pk']}", f"sk {t['sk']}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - keep["keyGen"])
    write(out_dir, "keygen.txt", lines)
    total["keyGen"] = (kept, skipped)

    # sigGen: (sk, message, context) -> signature, deterministic only.
    lines, kept, skipped = [], 0, 0
    for g in load(alg, "sigGen")["testGroups"]:
        if not implemented(alg, g, "siggen"):
            skipped += len(g["tests"])
            continue
        for t in g["tests"][:keep["sigGen"]]:
            lines += [f"set {g['parameterSet']}", f"sk {t['sk']}",
                      f"msg {t['message']}", f"ctx {t.get('context', '')}",
                      f"sig {t['signature']}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - keep["sigGen"])
    write(out_dir, "siggen.txt", lines)
    total["sigGen"] = (kept, skipped)

    # sigVer: pass/fail with NIST's own reason. The failing cases are the
    # valuable half: they are what proves `verify` rejects rather than accepts.
    lines, kept, skipped = [], 0, 0
    for g in load(alg, "sigVer")["testGroups"]:
        if not implemented(alg, g, "sigver"):
            skipped += len(g["tests"])
            continue
        chosen = pick_mixed(g["tests"], keep["sigVer"])
        for t in chosen:
            lines += [f"set {g['parameterSet']}", f"pk {t['pk']}",
                      f"msg {t['message']}", f"ctx {t.get('context', '')}",
                      f"sig {t['signature']}",
                      f"expect {'pass' if t['testPassed'] else 'fail'}",
                      f"reason {t.get('reason', '-')}", ""]
            kept += 1
        skipped += max(0, len(g["tests"]) - len(chosen))
    write(out_dir, "sigver.txt", lines)
    total["sigVer"] = (kept, skipped)

    print()
    for kind, (k, sk) in total.items():
        print(f"{alg} {kind}: kept {k}, skipped {sk}")
    return 0


def write(out_dir, name, lines):
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generated by tools/acvp-vectors.py from NIST ACVP-Server.\n"
                "# Source: gen-val/json-files/*/internalProjection.json\n"
                "# Do not edit; re-run the script instead.\n")
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"wrote {os.path.relpath(path)} ({os.path.getsize(path)} bytes)")


def main(argv):
    what = argv[1] if len(argv) > 1 else "all"
    algs = [argv[2]] if len(argv) > 2 else list(ALGS)
    for a in algs:
        if a not in ALGS:
            print(f"unknown algorithm {a}; known: {', '.join(ALGS)}")
            return 2
    if what in ("fetch", "all"):
        for a in algs:
            if fetch(a) != 0:
                return 1
    if what in ("build", "all"):
        for a in algs:
            if build(a) != 0:
                return 1
        return 0
    if what not in ("fetch", "build", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
