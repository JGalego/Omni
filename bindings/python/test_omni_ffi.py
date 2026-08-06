#!/usr/bin/env python3
"""Tests for the ctypes binding over the C ABI.

The interesting checks here are the ones a binding gets wrong in ways that only
show up later: a handle freed in the wrong order, a memoryview outliving the
memory it aliases, a DLPack capsule nobody frees, and a status silently flattened
into "error". Each of those has a test.

Run: `python3 bindings/python/test_omni_ffi.py`
"""

from __future__ import annotations

import ctypes
import gc
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omni_ffi  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OMNI_BIN = os.environ.get(
    "OMNI_BIN", os.path.join(REPO, "reference", "target", "release", "omni")
)

_TMP = tempfile.mkdtemp(prefix="omni-ffi-")
PLAIN = os.path.join(_TMP, "plain.omni")
QUANT = os.path.join(_TMP, "quant.omni")


def setUpModule():
    if not os.path.exists(OMNI_BIN):
        raise unittest.SkipTest(f"no omni binary at {OMNI_BIN}")
    subprocess.run([OMNI_BIN, "example", PLAIN], check=True, capture_output=True)
    subprocess.run(
        [OMNI_BIN, "example", "--quantized", QUANT], check=True, capture_output=True
    )


class TestLibrary(unittest.TestCase):
    def test_the_module_and_the_library_agree_on_the_abi(self):
        lib = omni_ffi.load_library()
        self.assertEqual(lib.abi_version >> 16, omni_ffi.ABI_VERSION >> 16)
        self.assertTrue(lib.spec_version.startswith("OMNI/"))

    def test_a_missing_library_says_how_to_build_it(self):
        with self.assertRaises(omni_ffi.OmniError) as cm:
            omni_ffi.load_library("/nonexistent/libomni.so")
        self.assertIn("cargo build", str(cm.exception))


class TestReading(unittest.TestCase):
    def test_a_container_opens_verifies_and_lists_its_tensors(self):
        with omni_ffi.open(PLAIN) as model:
            store = model.store
            self.assertGreater(store.size, 0)
            self.assertGreater(store.object_count, 0)
            self.assertEqual(store.hash_name, "blake3-256")
            self.assertEqual(len(store.root_digest), 32)
            rep = store.verify()
            self.assertTrue(rep.ok)
            self.assertEqual(rep.dangling, 0)
            self.assertTrue(rep.padding_ok and rep.alignment_ok)
            names = model.names()
            self.assertEqual(len(names), len(model))
            self.assertIn("model.embed_tokens.weight", names)
            # §04.2 load order, not alphabetical: the embedding comes first
            # because the producer said so.
            self.assertEqual(names[0], "model.embed_tokens.weight")
            store.close()

    def test_the_manifest_round_trips_through_json(self):
        with omni_ffi.open(PLAIN) as model:
            meta = model.meta()
            self.assertEqual(meta["t"], "omni.core/manifest")
            # A ref is [otype, hex], not an array of 32 numbers.
            ref = meta["assets"]["model"]
            self.assertIsInstance(ref, list)
            self.assertIsInstance(ref[1], str)
            self.assertEqual(len(ref[1]), 64)

    def test_a_literal_tensors_bytes_are_mapped_not_copied(self):
        with omni_ffi.open(PLAIN) as model:
            t = model["model.layers.0.norm.weight"]
            self.assertEqual(t.dtype, "f32")
            self.assertEqual(t.shape, (64,))
            self.assertEqual(t.layout, "strided")
            self.assertEqual(t.value_op, "literal")
            self.assertFalse(t.symbolic)
            data = t.memory()
            self.assertEqual(len(data), 256)
            self.assertTrue(t.mapped, "a small raw literal should not be copied")
            # The first element read by hand agrees with the decoded values.
            first = struct.unpack("<f", bytes(data[:4]))[0]
            self.assertAlmostEqual(first, t.values()[0], places=5)
            t.close()

    def test_a_computed_tensor_says_so_instead_of_returning_its_operand(self):
        with omni_ffi.open(QUANT) as model:
            t = model["model.layers.0.attn.q_proj.weight.bf16"]
            self.assertEqual(t.value_op, "dequantize")
            with self.assertRaises(omni_ffi.OmniError) as cm:
                t.memory()
            self.assertTrue(cm.exception.indeterminate)
            self.assertIn("not stored bytes", str(cm.exception))
            # But the values are computable, and there are the right number.
            vals = t.values()
            self.assertEqual(len(vals), 32 * 64)
            self.assertTrue(any(v != 0 for v in vals))
            t.close()

    def test_an_unknown_tensor_name_is_a_keyerror(self):
        with omni_ffi.open(PLAIN) as model:
            with self.assertRaises(KeyError):
                model["no.such.tensor"]
            # …but the underlying call is a status, not an exception type
            # chosen by guesswork.
            with self.assertRaises(omni_ffi.OmniError) as cm:
                model.tensor("no.such.tensor")
            self.assertEqual(cm.exception.kind, "usage")


class TestLifetimes(unittest.TestCase):
    def test_a_tensor_outlives_the_store_it_came_from(self):
        # This is the guarantee that makes a zero-copy binding possible: a
        # Python object outliving a borrow is the classic crash, and the Rust
        # side reference-counts precisely so it cannot happen here.
        model = omni_ffi.open(PLAIN)
        t = model["model.layers.0.norm.weight"]
        data = t.memory()
        expected = bytes(data)
        model.store.close()
        model.close()
        gc.collect()
        self.assertEqual(bytes(t.memory()), expected)
        t.close()

    def test_closing_twice_and_using_after_close_are_both_defined(self):
        model = omni_ffi.open(PLAIN)
        t = model["model.layers.0.norm.weight"]
        t.close()
        t.close()
        with self.assertRaises(omni_ffi.OmniError) as cm:
            t.memory()
        self.assertIn("closed", str(cm.exception))
        model.close()

    def test_junk_bytes_are_invalid_rather_than_a_crash(self):
        with self.assertRaises(omni_ffi.OmniError) as cm:
            omni_ffi.open_bytes(b"\0" * 512)
        self.assertEqual(cm.exception.kind, "invalid")

    def test_a_truncated_container_never_takes_the_process_down(self):
        with open(PLAIN, "rb") as f:
            raw = f.read()
        for frac in (1, 2, 3, 7):
            cut = len(raw) * frac // 8
            try:
                model = omni_ffi.open_bytes(raw[:cut])
            except omni_ffi.OmniError:
                continue
            try:
                for name in model.names():
                    try:
                        model[name].memory()
                    except omni_ffi.OmniError:
                        pass
            finally:
                model.close()


class TestPlanning(unittest.TestCase):
    def test_the_c0_baseline_is_infeasible_and_names_the_feature(self):
        with omni_ffi.open(PLAIN) as model:
            plan = model.resolve()
            self.assertFalse(plan.feasible)
            self.assertTrue(plan.unmet)
            self.assertTrue(
                any("omni.tensor/expr.1" in u.get("what", "") for u in plan.unmet),
                plan.unmet,
            )
            plan.close()

    def test_capabilities_as_a_dict_make_it_feasible(self):
        caps = {
            "t": "omni.rt/capabilities",
            "v": 1,
            "runtime": {"name": "python-ctypes", "version": "0"},
            "profiles": ["C0", "C1"],
            "dtypes": {"storage": ["bf16", "f32"], "compute": ["f32"]},
            "layouts": ["strided"],
            "features": ["omni.core/1.0", "omni.tensor/expr.1"],
            "policy": {"allow_lossy": False},
        }
        with omni_ffi.open(PLAIN) as model:
            plan = model.resolve(caps, objective="min-memory")
            self.assertTrue(plan.feasible, plan.unmet)
            self.assertGreater(plan.resident_bytes, 0)
            self.assertEqual(plan.as_dict()["objective"], "min-memory")
            plan.close()

    def test_an_unknown_objective_is_refused_by_name(self):
        with omni_ffi.open(PLAIN) as model:
            with self.assertRaises(omni_ffi.OmniError) as cm:
                model.resolve(objective="go-fast")
            self.assertIn("min-memory", str(cm.exception))


class TestDLPack(unittest.TestCase):
    """The protocol, consumed by hand.

    Neither NumPy nor PyTorch is a dependency of this repository, so the capsule
    is unwrapped here the way a consumer would. That is a stronger check than
    calling `np.from_dlpack` anyway: it verifies the fields rather than trusting
    another library to.
    """

    @staticmethod
    def _consume(capsule):
        """Take the tensor the way DLPack says a consumer must: rename the
        capsule, then own the deleter. Reading the fields without renaming would
        leave the capsule's destructor to free it too — a double free, not a
        leak, which is why this is a method and not two inline lines."""
        return omni_ffi.consume_capsule(capsule)

    def test_a_bf16_tensor_goes_over_as_kdlbfloat_and_frees_itself(self):
        model = omni_ffi.open(PLAIN)
        t = model["model.layers.0.attn.k_proj.weight"]
        self.assertEqual(t.__dlpack_device__(), (omni_ffi.DLPACK_CPU, 0))
        capsule = t.__dlpack__()
        managed = self._consume(capsule)
        dl = managed.contents.dl_tensor
        self.assertEqual(dl.device.device_type, omni_ffi.DLPACK_CPU)
        self.assertEqual((dl.dtype.code, dl.dtype.bits, dl.dtype.lanes), (4, 16, 1))
        self.assertEqual(dl.ndim, len(t.shape))
        self.assertEqual([dl.shape[i] for i in range(dl.ndim)], list(t.shape))
        self.assertFalse(dl.strides, "dense row-major means null strides")

        # The consumer's data must be the tensor's data.
        n = t.shape[0] * t.shape[1] * 2
        seen = ctypes.string_at(dl.data, n)
        self.assertEqual(seen, bytes(t.memory()))

        # Everything on the OMNI side goes away, and the DLPack tensor is still
        # readable — then frees itself.
        t.close()
        model.store.close()
        model.close()
        gc.collect()
        self.assertEqual(ctypes.string_at(dl.data, n), seen)
        managed.contents.deleter(managed)

    def test_reading_a_capsule_without_consuming_it_still_frees_once(self):
        model = omni_ffi.open(PLAIN)
        t = model["model.layers.0.norm.weight"]
        capsule = t.__dlpack__()
        managed = omni_ffi.capsule_pointer(capsule)
        self.assertIsNotNone(managed)
        self.assertEqual(managed.contents.dl_tensor.ndim, 1)
        # Not consumed, so the capsule's own destructor owns the deleter.
        del capsule
        gc.collect()
        t.close()
        model.close()

    def test_a_capsule_nobody_consumes_is_freed_rather_than_leaked(self):
        model = omni_ffi.open(PLAIN)
        t = model["model.layers.0.norm.weight"]
        for _ in range(64):
            capsule = t.__dlpack__()
            del capsule
        gc.collect()
        t.close()
        model.close()

    def test_a_dtype_dlpack_cannot_spell_is_refused_by_name(self):
        with omni_ffi.open(QUANT) as model:
            t = model["model.layers.0.attn.q_proj.qweight"]
            self.assertEqual(t.dtype, "u4")
            with self.assertRaises(omni_ffi.OmniError) as cm:
                t.__dlpack__()
            self.assertTrue(cm.exception.indeterminate)
            self.assertIn("u4", str(cm.exception))
            t.close()

    def test_a_device_other_than_the_cpu_is_a_buffererror(self):
        with omni_ffi.open(PLAIN) as model:
            t = model["model.layers.0.norm.weight"]
            with self.assertRaises(BufferError):
                t.__dlpack__(dl_device=(2, 0))  # kDLCUDA
            t.close()


class TestAgainstTheOtherReader(unittest.TestCase):
    """The binding and the from-specification reader must agree.

    They share no code: one calls Rust through a C ABI, the other parses the
    bytes in Python. Agreement on a tensor's bytes is worth more than either
    asserting it alone.
    """

    def test_both_python_readers_see_the_same_tensor_bytes(self):
        try:
            import omni as pure
        except ImportError:  # pragma: no cover
            self.skipTest("the pure-Python reader is not importable")
        c = pure.open_file(PLAIN)
        with omni_ffi.open(PLAIN) as model:
            checked = 0
            for name in model.names():
                t = model[name]
                try:
                    mine = bytes(t.memory())
                except omni_ffi.OmniError:
                    t.close()
                    continue
                try:
                    theirs, _desc = c.tensor_bytes(name)
                except Exception:
                    t.close()
                    continue
                self.assertEqual(mine, theirs, name)
                checked += 1
                t.close()
            self.assertGreater(checked, 0, "no tensor was compared")
            print(f"\n  {checked} tensors agree byte for byte across both readers")


if __name__ == "__main__":
    unittest.main(verbosity=2)
