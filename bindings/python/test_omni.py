#!/usr/bin/env python3
"""Tests for the pure-Python C0 reader.

Two kinds of check, and the second is the point of the whole file.

*Against known answers.* BLAKE3 against the official test vectors, CRC-32C
against the standard check value, and canonical CBOR against encodings that must
be refused. These would catch a reader that is wrong in the same way the Rust
implementation is wrong, which the cross-checks below cannot.

*Against the other implementation.* A container written by the Rust reference
implementation, read here, and compared: every object's digest, every literal
tensor's bytes, and the metadata. That is what makes this evidence for
`docs/design/sdk.md` §5's C0 budget rather than a second opinion from the same
author's assumptions.

Run: `python3 bindings/python/test_omni.py [container.omni …]`
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omni  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOY = os.path.join(REPO, "examples", "toy.omni")
OMNI_BIN = os.environ.get(
    "OMNI_BIN", os.path.join(REPO, "reference", "target", "release", "omni"))


def _vector_input(n: int) -> bytes:
    """The official BLAKE3 test-vector input: 0, 1, 2, … mod 251."""
    return bytes(i % 251 for i in range(n))


class TestPrimitives(unittest.TestCase):
    def test_crc32c_check_value(self):
        # The standard check value for CRC-32C over "123456789".
        self.assertEqual(omni.crc32c(b"123456789"), 0xE3069283)
        self.assertEqual(omni.crc32c(b""), 0)

    def test_blake3_official_vectors(self):
        # From the BLAKE3 repository's `test_vectors.json`, unkeyed, 32 bytes.
        # The sizes that matter are the ones that change the tree: a partial
        # block, a full chunk, one byte past a chunk, and enough chunks to make
        # the merge stack more than one deep.
        vectors = {
            0: "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            1: "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
            2: "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63",
            3: "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f",
            1023: "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11",
            1024: "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
            3072: "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2",
            4096: "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969",
        }
        for n, want in sorted(vectors.items()):
            with self.subTest(size=n):
                self.assertEqual(omni.blake3_256(_vector_input(n)).hex(), want)

    def test_blake3_tree_depth(self):
        # 1025 bytes is two chunks, which is the case a naive stack gets wrong:
        # the root is the *parent* of the two chunks, so the ROOT flag belongs to
        # that merge and not to either chunk.
        two = omni.blake3_256(_vector_input(1025))
        self.assertNotEqual(two, omni.blake3_256(_vector_input(1024)))
        # And the length is fixed whatever the depth.
        for n in (0, 1, 1024, 1025, 65537):
            self.assertEqual(len(omni.blake3_256(_vector_input(n))), 32)

    def test_sha256_is_the_other_mandatory_algorithm(self):
        self.assertEqual(
            omni.digest(omni.HASH_SHA256, b"abc").hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        with self.assertRaises(omni.Invalid):
            omni.digest(0x99, b"")


class TestCanonicalCbor(unittest.TestCase):
    """§03.2's canonical form is part of validity, so a reader enforces it.

    Every case below is a *different* encoding of a value the format already has
    an encoding for. Accepting one would mean two digests for one object, which
    is the one thing a content-addressed format cannot allow.
    """

    def test_it_reads_what_it_should(self):
        self.assertEqual(omni.cbor_decode(bytes([0x00])), 0)
        self.assertEqual(omni.cbor_decode(bytes([0x17])), 23)
        self.assertEqual(omni.cbor_decode(bytes([0x18, 0x18])), 24)
        self.assertEqual(omni.cbor_decode(bytes([0x20])), -1)
        self.assertEqual(omni.cbor_decode(bytes([0x41, 0xFF])), b"\xff")
        self.assertEqual(omni.cbor_decode(b"\x63abc"), "abc")
        self.assertEqual(omni.cbor_decode(bytes([0x82, 0x01, 0x02])), [1, 2])
        self.assertEqual(omni.cbor_decode(bytes([0xA1, 0x61, 0x61, 0x01])), {"a": 1})
        self.assertEqual(omni.cbor_decode(bytes([0xF4])), False)
        self.assertEqual(omni.cbor_decode(bytes([0xF6])), None)
        # D7's registered tags are kept, not unwrapped: 8/5 is not [8, 5], and
        # §04.3 needs the difference because `b3x5` ternary is exactly 8/5 bits.
        rational = omni.cbor_decode(bytes([0xD8, 0x1E, 0x82, 0x08, 0x05]))
        self.assertEqual(rational, omni.Tag(omni.TAG_RATIONAL, [8, 5]))
        # 1.0 in its shortest exact form is a half float.
        self.assertEqual(omni.cbor_decode(bytes([0xF9, 0x3C, 0x00])), 1.0)
        # 0.1 is not representable in half or single, so it is a double.
        self.assertAlmostEqual(
            omni.cbor_decode(bytes([0xFB, 0x3F, 0xB9, 0x99, 0x99, 0x99, 0x99,
                                    0x99, 0x9A])), 0.1)
        self.assertEqual(
            omni.cbor_decode(bytes([0xFA, 0x47, 0xC3, 0x50, 0x00])), 100000.0)

    def test_it_refuses_a_second_encoding_of_the_same_value(self):
        cases = [
            ("D1", bytes([0x18, 0x00]), "0 in two bytes"),
            ("D1", bytes([0x19, 0x00, 0x01]), "1 in three bytes"),
            ("D1", bytes([0x1A, 0x00, 0x00, 0x00, 0x01]), "1 in five bytes"),
            ("D2", bytes([0x9F, 0x01, 0xFF]), "an indefinite array"),
            ("D3", bytes([0xA2, 0x61, 0x62, 0x01, 0x61, 0x61, 0x02]), "unsorted keys"),
            ("D4", bytes([0xA2, 0x61, 0x61, 0x01, 0x61, 0x61, 0x02]), "duplicate keys"),
            ("D5", bytes([0xFB, 0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
             "1.0 as a double when a half would do"),
            ("D5", bytes([0xFA, 0x3F, 0x80, 0x00, 0x00]),
             "1.0 as a single when a half would do"),
            ("R-E03", bytes([0x62, 0xFF, 0xFE]), "text that is not UTF-8"),
            ("D7", bytes([0xC1, 0x00]), "an unregistered tag"),
            ("D8", bytes([0x01, 0x02]), "a trailing byte"),
            ("R-E01", bytes([0x41]), "a truncated byte string"),
        ]
        for rule, encoded, why in cases:
            with self.subTest(rule=rule, why=why):
                with self.assertRaises(omni.Invalid) as cm:
                    omni.cbor_decode(encoded)
                self.assertEqual(cm.exception.rule, rule, f"{why}: {cm.exception}")

    def test_depth_is_bounded(self):
        # A CBOR document is untrusted input, and 10^6 nested arrays is a stack
        # overflow in any language that recurses.
        deep = bytes([0x81]) * 200 + bytes([0x00])
        with self.assertRaises(omni.Invalid) as cm:
            omni.cbor_decode(deep)
        self.assertEqual(cm.exception.rule, "R-E04")

    def test_nan_has_one_encoding(self):
        v = omni.cbor_decode(bytes([0xF9, 0x7E, 0x00]))
        self.assertNotEqual(v, v, "that is a NaN")
        # Any other NaN payload is a second encoding of the same value.
        with self.assertRaises(omni.Invalid):
            omni.cbor_decode(bytes([0xF9, 0x7E, 0x01]))


class TestContainer(unittest.TestCase):
    """The committed example, read from the specification."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TOY):
            raise unittest.SkipTest(f"{TOY} is not there")
        cls.c = omni.open_file(TOY)

    def test_the_framing_is_what_section_02_says(self):
        c = self.c
        self.assertEqual((c.container_major, c.container_minor), (1, 0))
        self.assertEqual(c.hash, omni.HASH_BLAKE3_256)
        self.assertEqual(c.header_size, 128)
        self.assertEqual(c.file_size, len(c.data))
        self.assertEqual(c.creator, "omni-rs/0.1.0")
        kinds = [k for _, k, _ in c.segments()]
        # §02.4's order: a superblock at each end, objects, blobs, the index.
        self.assertEqual(kinds[0], "SUPER")
        self.assertEqual(kinds[-1], "SUPER")
        self.assertIn("OBJ", kinds)
        self.assertIn("BLOB", kinds)
        self.assertIn("INDEX", kinds)

    def test_every_object_verifies(self):
        n, b = self.c.verify()
        self.assertEqual(n, len(self.c.index))
        self.assertGreater(b, 0)
        # R-O01 is the whole point: an object's digest is its name, so reading it
        # without checking would make the name decorative.
        self.assertEqual(
            omni.digest(self.c.hash, self.c.get(self.c.root_digest)),
            self.c.root_digest)

    def test_the_index_is_sorted_and_findable(self):
        digests = [e.digest for e in self.c.index]
        self.assertEqual(digests, sorted(digests))
        for e in self.c.index:
            self.assertIs(self.c.find(e.digest), e)
        # And a digest that is not there is not there — not a neighbour.
        missing = bytearray(self.c.index[0].digest)
        missing[31] ^= 0xFF
        self.assertIsNone(self.c.find(bytes(missing)))

    def test_the_graph_walks_to_the_tensors(self):
        meta = self.c.meta()
        self.assertEqual(meta["name"], "omni/example-toy")
        self.assertEqual(meta["arch"]["family"], "transformer.decoder")
        tensors = self.c.tensors()
        self.assertIn("model.embed_tokens.weight", tensors)
        raw, desc = self.c.tensor_bytes("model.embed_tokens.weight")
        self.assertEqual(desc["shape"], [256, 64])
        # bf16: two bytes an element, and the chunk list has to hold exactly that.
        self.assertEqual(len(raw), 256 * 64 * 2)

    def test_the_embedding_and_the_head_share_their_bytes(self):
        # §01.2's claim, checked from the outside: `lm_head.weight` is tied to
        # the embedding, so the two descriptors differ and the payload exists
        # once. Two readers agreeing on that is worth more than one asserting it.
        a, _ = self.c.tensor_bytes("model.embed_tokens.weight")
        b, _ = self.c.tensor_bytes("lm_head.weight")
        self.assertEqual(a, b)

    def test_tampering_is_caught(self):
        data = bytearray(self.c.data)
        # Flip a byte inside the object segment. Which byte does not matter: it
        # belongs to some object, and that object's digest no longer matches.
        segs = omni.Container(bytes(data)).segments()
        obj = next(s for s in segs if s[1] == "OBJ")
        data[obj[0] + omni.SEG_HEADER_SIZE + 4] ^= 0x01
        with self.assertRaises(omni.Invalid):
            # The segment CRC catches it first, and if not, the digest does.
            c = omni.Container(bytes(data))
            c.segments()
            c.verify()

    def test_a_truncated_file_is_refused_at_every_length(self):
        for n in (0, 1, 100, 191, 192, len(self.c.data) - 1):
            with self.subTest(length=n):
                with self.assertRaises((omni.Invalid, struct.error, IndexError)):
                    omni.Container(self.c.data[:n])

    def test_a_corrupt_header_crc_is_refused(self):
        data = bytearray(self.c.data)
        data[13] ^= 0x01  # log2_align, covered by the header CRC
        with self.assertRaises(omni.Invalid) as cm:
            omni.Container(bytes(data))
        self.assertEqual(cm.exception.rule, "R-C02")


class TestAgainstRust(unittest.TestCase):
    """The cross-implementation check, which is what this file is for."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(OMNI_BIN):
            raise unittest.SkipTest(f"{OMNI_BIN} is not built")
        if not os.path.exists(TOY):
            raise unittest.SkipTest(f"{TOY} is not there")

    def test_blake3_agrees_over_every_tree_shape(self):
        # The Rust implementation is checked against the official vectors in its
        # own test suite, so agreeing with it over these sizes is agreeing with
        # the vectors — including tree depths no published vector covers.
        c = omni.open_file(TOY)
        for e in c.index:
            payload = c.data[e.offset:e.offset + e.stored_len]
            if e.codec == 0:
                self.assertEqual(
                    omni.blake3_256(payload), e.digest,
                    f"object {e.digest.hex()[:16]} ({e.stored_len} bytes)")

    def test_the_tensor_bytes_agree_with_the_rust_exporter(self):
        out = "/tmp/omni-py-crosscheck.safetensors"
        r = subprocess.run(
            [OMNI_BIN, "export", "safetensors", TOY, "-o", out, "--allow-lossy"],
            capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest(f"the exporter declined: {r.stderr.strip()}")
        raw = open(out, "rb").read()
        n = struct.unpack("<Q", raw[:8])[0]
        head = json.loads(raw[8:8 + n])
        c = omni.open_file(TOY)
        names = [k for k in head if k != "__metadata__"]
        self.assertGreater(len(names), 0)
        for name in names:
            with self.subTest(tensor=name):
                lo, hi = head[name]["data_offsets"]
                want = raw[8 + n + lo:8 + n + hi]
                got, _ = c.tensor_bytes(name)
                self.assertEqual(got, want, f"{name}: {len(got)} vs {len(want)} bytes")

    def test_both_mandatory_hash_algorithms_read(self):
        for algo, code in (("blake3", omni.HASH_BLAKE3_256),
                           ("sha256", omni.HASH_SHA256)):
            with self.subTest(hash=algo):
                out = f"/tmp/omni-py-{algo}.omni"
                r = subprocess.run([OMNI_BIN, "example", out, "--hash", algo],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                c = omni.open_file(out)
                self.assertEqual(c.hash, code)
                n, _ = c.verify()
                self.assertEqual(n, len(c.index))

    def test_what_is_above_c0_is_refused_by_name(self):
        # A quantized example's weights are `dequantize` expressions over packed
        # literals. C0 does not evaluate expressions, and saying so is the
        # difference between a conformance level and a bug.
        out = "/tmp/omni-py-quant.omni"
        r = subprocess.run([OMNI_BIN, "example", out, "--quantized"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("the quantized example is not available")
        c = omni.open_file(out)
        refused = []
        for name in c.tensors():
            try:
                c.tensor_bytes(name)
            except omni.Unsupported as e:
                refused.append(str(e))
        self.assertTrue(refused, "a quantized model should have something above C0")
        self.assertTrue(
            any("dequantize" in m for m in refused),
            f"the reason should name the node: {refused}")

    def test_a_compressed_object_is_refused_rather_than_returned(self):
        src = "/tmp/omni-py-zeros.safetensors"
        head = json.dumps(
            {"w": {"dtype": "F32", "shape": [1024], "data_offsets": [0, 4096]}}
        ).encode()
        open(src, "wb").write(struct.pack("<Q", len(head)) + head + b"\0" * 4096)
        plain, packed = "/tmp/omni-py-zeros.omni", "/tmp/omni-py-zstd.omni"
        for cmd in ([OMNI_BIN, "import", "safetensors", src, "-o", plain],
                    [OMNI_BIN, "repack", plain, "-o", packed, "--codec", "zstd"]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                self.skipTest(f"{cmd[1]} declined: {r.stderr.strip()}")
        c = omni.open_file(packed)
        compressed = [e for e in c.index if e.codec != 0]
        self.assertTrue(compressed, "zstd should have compressed the zero blob")
        with self.assertRaises(omni.Unsupported) as cm:
            c.get(compressed[0].digest)
        self.assertIn("codec", str(cm.exception))


class TestTheBudgetClaim(unittest.TestCase):
    def test_the_reader_is_small_enough_to_be_evidence(self):
        # `docs/design/sdk.md` §5 puts a pure-language C0 reader at ~3000 lines.
        # This one is the measurement for Python. It is not a limit to defend —
        # it is the number the claim is about, so it is asserted loosely and
        # printed either way.
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni.py")
        with open(path) as f:
            lines = f.readlines()
        code = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        print(f"\n  the C0 reader: {len(lines)} lines, {len(code)} non-comment")
        self.assertLess(len(lines), 3000, "the C0 budget claim is ~3000 lines")


if __name__ == "__main__":
    unittest.main(verbosity=2)
