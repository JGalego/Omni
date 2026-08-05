#!/usr/bin/env python3
"""Builds `docs/explorer.html` from `examples/toy.omni`.

Every number, digest, offset and byte the explorer shows is read out of the real
container by the real tool — `omni ls`, `omni verify`, `omni dump` — or out of
the file itself. Nothing is typed in by hand, which is the point: a diagram of a
binary format that drifts from the format is worse than no diagram.

CI runs this and diffs the result against the committed page, so the two cannot
disagree. Regenerate with:

    python3 tools/build-explorer.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OMNI = ROOT / "reference" / "target" / "release" / "omni"
CONTAINER = ROOT / "examples" / "toy.omni"
OUT = ROOT / "docs" / "explorer.html"


def run(*args: str) -> str:
    r = subprocess.run(
        [str(OMNI), *args], capture_output=True, text=True, cwd=ROOT, check=True
    )
    return r.stdout


def objects() -> list[dict]:
    out = []
    for line in run("ls", str(CONTAINER), "--full").splitlines()[1:]:
        if not line.strip():
            continue
        digest, otype, offset, nbytes, *flags = line.split()
        out.append(
            {
                "digest": digest,
                "type": otype,
                "off": int(offset),
                "len": int(nbytes),
                "flags": flags[0] if flags else "",
            }
        )
    return out


def segments() -> list[dict]:
    out = []
    for line in run("verify", str(CONTAINER), "--level", "0").splitlines():
        m = re.match(r"\s+0x([0-9a-f]+)\s+(\w+)\s+(\d+) B", line)
        if m:
            out.append(
                {"off": int(m.group(1), 16), "kind": m.group(2), "len": int(m.group(3))}
            )
    return out


def diag(digest: str) -> str:
    text = run("dump", str(CONTAINER), "--object", digest, "--diag")
    # The first two lines are a `;` comment naming the object; the explorer shows
    # the type and offset in its own chrome, so only the value is kept.
    return "\n".join(l for l in text.splitlines() if not l.startswith(";")).strip()


def header_fields() -> list[dict]:
    """The annotated header dump, parsed back into (offset, len, name, value)."""
    fields = []
    for line in run("dump", str(CONTAINER), "--header").splitlines():
        # `   16  cd 5a … cb     file_uuid (UUIDv7-shaped, derived)`
        m = re.match(r"\s*(\d+)\s+((?:[0-9a-f]{2} )+)\s{2,}(\S+)(?:\s+(.*?))?\s*$", line)
        if not m:
            continue
        raw_bytes = m.group(2).split()
        fields.append(
            {
                "off": int(m.group(1)),
                "len": len(raw_bytes),
                "name": m.group(3),
                "value": (m.group(4) or "").strip(),
            }
        )
    return fields


def verify_lines() -> list[str]:
    """The V0–V6 ladder as the tool prints it."""
    out = []
    for line in run("verify", str(CONTAINER), "--level", "6").splitlines():
        if re.match(r"^(V\d|valid|invalid|incomplete)", line):
            out.append(line.rstrip())
    return out


def main() -> int:
    if not OMNI.exists():
        print(f"error: {OMNI} not built; run `cargo build --release` in reference/")
        return 2

    raw = CONTAINER.read_bytes()
    objs = objects()
    segs = segments()

    # Structure objects get their CBOR diagnostic notation; data objects are
    # bytes and have none. Refs are extracted from the diagnostics, which is how
    # the explorer draws edges: from what the objects actually say, not from a
    # hand-drawn graph.
    by_digest = {o["digest"]: o for o in objs}
    for o in objs:
        if o["type"] == "Blob":
            continue
        o["diag"] = diag(o["digest"])
        o["refs"] = [
            d for d in re.findall(r"h'([0-9a-f]{64})'", o["diag"]) if d in by_digest
        ]

    # Tensor names, so a node can be labelled with what a human calls it. The
    # table is the only place that mapping exists (§04.2).
    table = next(o for o in objs if o["type"] == "TensorTable")
    names = dict(
        (d, n)
        for n, d in re.findall(r'"([^"]+)": \[\d+, h\'([0-9a-f]{64})\'\]', table["diag"])
    )
    for o in objs:
        if o["digest"] in names:
            o["name"] = names[o["digest"]]

    data = {
        "file": CONTAINER.name,
        "size": len(raw),
        # The header says what the root is (bytes 64..96, §02.3). Taking the
        # object at the lowest offset would give the same answer for this file and
        # would be luck rather than a reading of the format.
        "root": raw[64:96].hex(),
        "segments": segs,
        "objects": objs,
        "header": header_fields(),
        "verify": verify_lines(),
        # The whole file, as hex. It doubles the page to ~300 KiB and it is
        # worth it: with anything less, clicking a data object shows a note
        # apologising instead of the weights, and an explorer with a hole in it
        # teaches the format's shape but not its substance.
        "hex": raw.hex(),
        "trailer_off": len(raw) - 64,
    }

    template = (ROOT / "tools" / "explorer.template.html").read_text()
    page = template.replace(
        "/*DATA*/", "const OMNI = " + json.dumps(data, separators=(",", ":")) + ";"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(
        f"wrote {OUT.relative_to(ROOT)}  "
        f"({len(page) / 1024:.0f} KiB, {len(objs)} objects, {len(segs)} segments)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
