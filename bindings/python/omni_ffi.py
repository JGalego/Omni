#!/usr/bin/env python3
"""OMNI for Python, over the C ABI, with no dependencies outside the standard
library.

There are two Python readers in this directory and they answer different
questions.

`omni.py` is a **C0 reader written from the specification**, in pure Python,
implementing BLAKE3 by hand. Its job is evidence: `docs/design/sdk.md` §5 claims
a conforming reader at the lowest profile needs almost nothing, and a second
implementation in a second language is how that claim gets tested rather than
asserted. It is deliberately slow and deliberately limited.

This file is the other thing: a **binding**. It calls
[`reference/omni-ffi`](../../reference/omni-ffi) through `ctypes`, so it gets
everything the Rust implementation can do — compressed objects, expression
evaluation, quantized weights, capability negotiation — and it gets tensors out
through DLPack without a copy, which is what makes it useful to PyTorch, JAX,
CuPy and NumPy.

`ctypes` is in the standard library, so this stays dependency-free too. It is
not the PyO3 binding of `docs/design/sdk.md` §4.1 — that one would release the
GIL and skip a layer of marshalling — but it needs no build step and no compiler
on the user's machine, which for a *reader* is a better trade than it looks.

    import omni_ffi

    with omni_ffi.open("model.omni") as model:
        print(model.meta()["name"], len(model), "tensors")
        w = model["model.embed_tokens.weight"]
        print(w.dtype, w.shape, "zero copy" if w.mapped else "copied")
        t = torch.from_dlpack(w)              # no copy when w.mapped

The library is found via `$OMNI_LIBRARY`, then the repository's own
`reference/target/{release,debug}`, then the system loader. Build it with:

    cargo build --release -p omni-ffi --manifest-path reference/Cargo.toml
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import sys

__all__ = [
    "OmniError",
    "Library",
    "Store",
    "Model",
    "Plan",
    "Tensor",
    "Builder",
    "VerifyReport",
    "open",
    "open_bytes",
    "load_library",
    "capsule_pointer",
    "consume_capsule",
    "ABI_VERSION",
    "OBJECTIVES",
]

# The major half must match the library's; see omni.h. 1.1 added the writer.
ABI_VERSION = 0x0001_0001

OK = 0
EINVALID = 1
EUSAGE = 2
INDETERMINATE = 3
EPOLICY = 4
EINCOMPLETE = 5
EINFEASIBLE = 6
EINTERNAL = 7

_STATUS_NAMES = {
    OK: "ok",
    EINVALID: "invalid",
    EUSAGE: "usage",
    INDETERMINATE: "indeterminate",
    EPOLICY: "policy",
    EINCOMPLETE: "incomplete",
    EINFEASIBLE: "infeasible",
    EINTERNAL: "internal",
}

OBJECTIVES = {
    "min-memory": 0,
    "max-quality": 1,
    "min-load-time": 2,
    "min-latency": 3,
    "balanced": 4,
}


class OmniError(Exception):
    """A status the ABI returned, with the sentence it returned beside it.

    `status` is the numeric code and `kind` its name. The distinction that
    matters is `indeterminate`: the file is fine and this build cannot fully
    handle it. Catching that as if it were `invalid` is the bug §14.4 is about,
    so it is spelled out rather than folded into one exception type.
    """

    def __init__(self, status: int, message: str, what: str = ""):
        self.status = status
        self.kind = _STATUS_NAMES.get(status, "unknown")
        self.message = message
        prefix = f"{what}: " if what else ""
        super().__init__(f"{prefix}{self.kind}: {message}")

    @property
    def indeterminate(self) -> bool:
        return self.status == INDETERMINATE


# ------------------------------------------------------------------ DLPack --


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]

DLPACK_CPU = 1

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# These take a `PyObject *` in C. They are declared as taking a raw pointer
# rather than `py_object` on purpose: ctypes increfs a `py_object` argument and
# decrefs it on return, and doing that to a capsule that is *already being
# deallocated* frees it a second time. The destructor below runs in exactly that
# situation, so it never lets ctypes touch a reference count.
_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_PyErr_Clear = ctypes.pythonapi.PyErr_Clear
_PyErr_Clear.restype = None
_PyErr_Clear.argtypes = []

_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def capsule_pointer(capsule, name: bytes = b"dltensor"):
    """The pointer inside a PyCapsule, or None if it is not that capsule.

    Exposed because unwrapping a DLPack capsule by hand is how you test a
    producer without making a consumer a dependency.
    """
    addr = ctypes.c_void_p(id(capsule))
    if not _PyCapsule_IsValid(addr, name):
        _PyErr_Clear()
        return None
    ptr = _PyCapsule_GetPointer(addr, name)
    if not ptr:
        _PyErr_Clear()
        return None
    return ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor))


_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# PyCapsule_SetName stores the pointer, not a copy, so the name has to outlive
# every capsule that carries it. Module-level, therefore.
_USED_NAME = ctypes.c_char_p(b"used_dltensor")


def consume_capsule(capsule):
    """Takes ownership of a DLPack capsule, the way a consumer must.

    DLPack's rule is that whoever takes the tensor renames the capsule to
    `used_dltensor`, and from then on *they* are responsible for calling the
    deleter — exactly once. Skipping the rename means the capsule's own
    destructor also calls it, which is a double free rather than a leak.

    Returns the `DLManagedTensor *`. Call `managed.contents.deleter(managed)`
    when done.
    """
    managed = capsule_pointer(capsule)
    if managed is None:
        raise OmniError(EUSAGE, "not an unconsumed `dltensor` capsule")
    _PyCapsule_SetName(ctypes.c_void_p(id(capsule)), _USED_NAME)
    return managed


def _destroy_unconsumed_capsule(addr):
    """Frees a DLPack capsule nobody took.

    The DLPack protocol says a consumer renames the capsule to
    `used_dltensor` and becomes responsible for the deleter. If the name is
    still `dltensor` when the capsule dies, nobody did, and the producer has to
    free it or the tensor leaks. This is that path.
    """
    if not addr or not _PyCapsule_IsValid(addr, b"dltensor"):
        _PyErr_Clear()
        return
    ptr = _PyCapsule_GetPointer(addr, b"dltensor")
    if not ptr:
        _PyErr_Clear()
        return
    managed = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor))
    if managed.contents.deleter:
        managed.contents.deleter(managed)


_CAPSULE_DESTRUCTOR_REF = _CAPSULE_DESTRUCTOR(_destroy_unconsumed_capsule)


# ----------------------------------------------------------------- library --


class _VerifyReportC(ctypes.Structure):
    _fields_ = [
        ("segments", ctypes.c_uint64),
        ("objects_verified", ctypes.c_uint64),
        ("bytes_verified", ctypes.c_uint64),
        ("reachable", ctypes.c_uint64),
        ("dangling", ctypes.c_uint64),
        ("mistyped", ctypes.c_uint64),
        ("padding_ok", ctypes.c_int),
        ("alignment_ok", ctypes.c_int),
    ]


class _TensorInfoC(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_char_p),
        ("dtype_bits", ctypes.c_uint32),
        ("layout", ctypes.c_char_p),
        ("value_op", ctypes.c_char_p),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("numel", ctypes.c_uint64),
    ]


def _library_names():
    if sys.platform == "darwin":
        return ["libomni.dylib"]
    if os.name == "nt":
        return ["omni.dll"]
    return ["libomni.so"]


def _candidate_paths():
    env = os.environ.get("OMNI_LIBRARY")
    if env:
        yield pathlib.Path(env)
    # The repository's own build, so `python3 bindings/python/omni_ffi.py f.omni`
    # works from a checkout with nothing installed.
    root = pathlib.Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        for name in _library_names():
            yield root / "reference" / "target" / profile / name


def load_library(path=None) -> "Library":
    """Loads `libomni` and returns a `Library` with every prototype declared.

    Declaring argument and return types is not decoration: ctypes defaults a
    return to `int`, which silently truncates a pointer on 64-bit platforms.
    Every entry point below is spelled out for that reason.
    """
    tried = []
    for candidate in [pathlib.Path(path)] if path else _candidate_paths():
        if candidate.exists():
            return Library(ctypes.CDLL(str(candidate)))
        tried.append(str(candidate))
    for name in _library_names():
        try:
            return Library(ctypes.CDLL(name))
        except OSError:
            tried.append(name)
    raise OmniError(
        EINCOMPLETE,
        "cannot find libomni; build it with `cargo build --release -p omni-ffi "
        "--manifest-path reference/Cargo.toml`, or set $OMNI_LIBRARY. Tried: "
        + ", ".join(tried),
    )


_P = ctypes.POINTER
_VOID = ctypes.c_void_p
_CHARPP = _P(ctypes.c_char_p)


class Library:
    """The loaded shared object, with prototypes applied."""

    def __init__(self, dll: ctypes.CDLL):
        self.dll = dll
        d = dll

        def sig(name, restype, argtypes):
            fn = getattr(d, name)
            fn.restype = restype
            fn.argtypes = argtypes
            return fn

        sig("omni_abi_version", ctypes.c_uint32, [])
        sig("omni_spec_version", ctypes.c_char_p, [])
        sig("omni_last_error", ctypes.c_char_p, [])
        sig("omni_status_name", ctypes.c_char_p, [ctypes.c_int])

        sig("omni_store_open", ctypes.c_int, [ctypes.c_char_p, _P(_VOID)])
        sig(
            "omni_store_open_bytes",
            ctypes.c_int,
            [ctypes.c_char_p, ctypes.c_size_t, _P(_VOID)],
        )
        sig("omni_store_close", None, [_VOID])
        sig("omni_store_size", ctypes.c_uint64, [_VOID])
        sig("omni_store_object_count", ctypes.c_uint64, [_VOID])
        sig("omni_store_hash_name", ctypes.c_int, [_VOID, _CHARPP])
        sig("omni_store_root_digest", ctypes.c_int, [_VOID, _P(ctypes.c_uint8)])
        sig("omni_store_set_max_elems", ctypes.c_int, [_VOID, ctypes.c_uint64])
        sig("omni_store_verify", ctypes.c_int, [_VOID, _P(_VerifyReportC)])
        sig("omni_store_root", ctypes.c_int, [_VOID, _P(_VOID)])

        sig("omni_model_free", None, [_VOID])
        sig(
            "omni_model_meta_json",
            ctypes.c_int,
            [_VOID, _CHARPP, _P(ctypes.c_size_t)],
        )
        sig("omni_model_tensor_count", ctypes.c_size_t, [_VOID])
        sig(
            "omni_model_tensor_name",
            ctypes.c_int,
            [_VOID, ctypes.c_size_t, _CHARPP],
        )
        sig("omni_model_tensor", ctypes.c_int, [_VOID, ctypes.c_char_p, _P(_VOID)])
        sig(
            "omni_model_resolve",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, ctypes.c_int, _P(_VOID)],
        )

        sig("omni_plan_feasible", ctypes.c_int, [_VOID])
        sig("omni_plan_resident_bytes", ctypes.c_uint64, [_VOID])
        sig("omni_plan_read_bytes", ctypes.c_uint64, [_VOID])
        sig("omni_plan_json", ctypes.c_int, [_VOID, _CHARPP, _P(ctypes.c_size_t)])
        sig("omni_plan_free", None, [_VOID])

        sig("omni_tensor_name", ctypes.c_int, [_VOID, _CHARPP])
        sig("omni_tensor_get_info", ctypes.c_int, [_VOID, _P(_TensorInfoC)])
        sig(
            "omni_tensor_bytes",
            ctypes.c_int,
            [_VOID, _P(_VOID), _P(ctypes.c_size_t)],
        )
        sig("omni_tensor_mapped", ctypes.c_int, [_VOID])
        sig(
            "omni_tensor_values",
            ctypes.c_int,
            [_VOID, _P(_P(ctypes.c_double)), _P(ctypes.c_size_t)],
        )
        sig("omni_tensor_release", None, [_VOID])
        sig(
            "omni_tensor_dlpack",
            ctypes.c_int,
            [_VOID, _P(_P(_DLManagedTensor))],
        )

        sig("omni_builder_new", ctypes.c_int, [ctypes.c_char_p, _P(_VOID)])
        sig("omni_builder_free", None, [_VOID])
        sig("omni_builder_set_hash", ctypes.c_int, [_VOID, ctypes.c_char_p])
        sig("omni_builder_set_license", ctypes.c_int, [_VOID, ctypes.c_char_p])
        sig(
            "omni_builder_set_arch",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, ctypes.c_char_p],
        )
        sig(
            "omni_builder_add_meta",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, ctypes.c_char_p],
        )
        sig("omni_builder_set_chunk_size", ctypes.c_int, [_VOID, ctypes.c_uint64])
        sig("omni_builder_set_codec", ctypes.c_int, [_VOID, ctypes.c_char_p])
        sig("omni_builder_set_alignment", ctypes.c_int, [_VOID, ctypes.c_uint32])
        sig(
            "omni_builder_add_tensor",
            ctypes.c_int,
            [
                _VOID,
                ctypes.c_char_p,
                ctypes.c_char_p,
                _P(ctypes.c_uint64),
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ],
        )
        sig(
            "omni_builder_set_tensor_axes",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, ctypes.c_char_p],
        )
        sig(
            "omni_builder_set_tensor_semantic",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, ctypes.c_char_p],
        )
        sig(
            "omni_builder_add_dlpack",
            ctypes.c_int,
            [_VOID, ctypes.c_char_p, _P(_DLTensor)],
        )
        sig("omni_builder_tensor_count", ctypes.c_size_t, [_VOID])
        sig("omni_builder_root_digest", ctypes.c_int, [_VOID, _P(ctypes.c_uint8)])
        sig("omni_builder_write", ctypes.c_int, [_VOID, ctypes.c_char_p])
        sig(
            "omni_builder_write_bytes",
            ctypes.c_int,
            [_VOID, _P(_P(ctypes.c_uint8)), _P(ctypes.c_size_t)],
        )

        got = d.omni_abi_version()
        if got >> 16 != ABI_VERSION >> 16:
            raise OmniError(
                EINVALID,
                f"ABI major mismatch: this module expects {ABI_VERSION >> 16}, "
                f"the library is {got >> 16}",
            )

    @property
    def abi_version(self) -> int:
        return self.dll.omni_abi_version()

    @property
    def spec_version(self) -> str:
        return self.dll.omni_spec_version().decode()

    def last_error(self) -> str:
        raw = self.dll.omni_last_error()
        return raw.decode("utf-8", "replace") if raw else ""

    def check(self, status: int, what: str = "") -> None:
        if status != OK:
            raise OmniError(status, self.last_error(), what)


_DEFAULT: "Library | None" = None


def _default_library() -> Library:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_library()
    return _DEFAULT


# ------------------------------------------------------------------ handles --


class VerifyReport:
    """What a full verification found. `ok` is the three-valued answer."""

    __slots__ = (
        "segments",
        "objects_verified",
        "bytes_verified",
        "reachable",
        "dangling",
        "mistyped",
        "padding_ok",
        "alignment_ok",
        "status",
    )

    def __init__(self, c: _VerifyReportC, status: int):
        self.segments = c.segments
        self.objects_verified = c.objects_verified
        self.bytes_verified = c.bytes_verified
        self.reachable = c.reachable
        self.dangling = c.dangling
        self.mistyped = c.mistyped
        self.padding_ok = bool(c.padding_ok)
        self.alignment_ok = bool(c.alignment_ok)
        self.status = status

    @property
    def ok(self) -> bool:
        return self.status == OK

    def __repr__(self) -> str:
        return (
            f"VerifyReport({_STATUS_NAMES.get(self.status, self.status)}, "
            f"{self.objects_verified} objects, {self.bytes_verified} bytes, "
            f"{self.dangling} dangling)"
        )


class _Handle:
    """Common lifetime handling: idempotent close, context manager, no
    resurrection. The Rust side reference-counts, so the order these are closed
    in does not matter — a tensor may outlive its store."""

    _free_name = ""

    def __init__(self, lib: Library, ptr):
        self._lib = lib
        self._ptr = ptr

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            getattr(self._lib.dll, self._free_name)(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter teardown
            pass

    def _live(self):
        if not getattr(self, "_ptr", None):
            raise OmniError(EUSAGE, f"this {type(self).__name__} is closed")
        return self._ptr


class Store(_Handle):
    """An opened container."""

    _free_name = "omni_store_close"

    def __init__(self, lib: Library, ptr):
        super().__init__(lib, ptr)

    @property
    def size(self) -> int:
        return self._lib.dll.omni_store_size(self._live())

    @property
    def object_count(self) -> int:
        return self._lib.dll.omni_store_object_count(self._live())

    @property
    def hash_name(self) -> str:
        out = ctypes.c_char_p()
        self._lib.check(
            self._lib.dll.omni_store_hash_name(self._live(), ctypes.byref(out)),
            "hash name",
        )
        return out.value.decode()

    @property
    def root_digest(self) -> bytes:
        buf = (ctypes.c_uint8 * 32)()
        self._lib.check(
            self._lib.dll.omni_store_root_digest(self._live(), buf), "root digest"
        )
        return bytes(buf)

    def set_max_elems(self, n: int) -> None:
        """Raises the per-node materialization cap (§12.4). A declared size is
        untrusted input; loading something enormous is the caller's decision."""
        self._lib.check(
            self._lib.dll.omni_store_set_max_elems(self._live(), n), "max elems"
        )

    def verify(self, raise_on_error: bool = True) -> VerifyReport:
        rep = _VerifyReportC()
        status = self._lib.dll.omni_store_verify(self._live(), ctypes.byref(rep))
        if status != OK and raise_on_error:
            raise OmniError(status, self._lib.last_error(), "verify")
        return VerifyReport(rep, status)

    def root(self) -> "Model":
        out = ctypes.c_void_p()
        self._lib.check(
            self._lib.dll.omni_store_root(self._live(), ctypes.byref(out)), "root"
        )
        return Model(self._lib, out)


class Model(_Handle):
    """The model asset, its manifest, and its tensor table."""

    _free_name = "omni_model_free"

    def meta_json(self) -> str:
        out = ctypes.c_char_p()
        n = ctypes.c_size_t()
        self._lib.check(
            self._lib.dll.omni_model_meta_json(
                self._live(), ctypes.byref(out), ctypes.byref(n)
            ),
            "meta",
        )
        return ctypes.string_at(out, n.value).decode()

    def meta(self) -> dict:
        return json.loads(self.meta_json())

    def __len__(self) -> int:
        return self._lib.dll.omni_model_tensor_count(self._live())

    def names(self) -> "list[str]":
        """Tensor names in §04.2 load order — the order the producer asked for,
        not an alphabetisation of it."""
        out = ctypes.c_char_p()
        names = []
        for i in range(len(self)):
            self._lib.check(
                self._lib.dll.omni_model_tensor_name(
                    self._live(), i, ctypes.byref(out)
                ),
                f"tensor name {i}",
            )
            names.append(out.value.decode())
        return names

    def __iter__(self):
        return iter(self.names())

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def tensor(self, name: str) -> "Tensor":
        out = ctypes.c_void_p()
        self._lib.check(
            self._lib.dll.omni_model_tensor(
                self._live(), name.encode(), ctypes.byref(out)
            ),
            name,
        )
        return Tensor(self._lib, out)

    def __getitem__(self, name: str) -> "Tensor":
        try:
            return self.tensor(name)
        except OmniError as e:
            if e.status == EUSAGE:
                raise KeyError(name) from e
            raise

    def resolve(self, capabilities=None, objective: str = "min-memory") -> "Plan":
        """Negotiates a plan (§10.5).

        `capabilities` may be a dict, a JSON string, or None. None means the C0
        baseline — the floor every conforming reader meets — which deliberately
        does not include the expression feature, so a model that needs one comes
        back infeasible rather than quietly reduced. The plan is returned either
        way; `Plan.unmet` is the useful part.
        """
        if objective not in OBJECTIVES:
            raise OmniError(
                EUSAGE,
                f"`{objective}` is not an objective; try one of "
                + ", ".join(sorted(OBJECTIVES)),
            )
        if capabilities is None:
            caps = None
        elif isinstance(capabilities, str):
            caps = capabilities.encode()
        else:
            caps = json.dumps(capabilities).encode()
        out = ctypes.c_void_p()
        status = self._lib.dll.omni_model_resolve(
            self._live(), caps, OBJECTIVES[objective], ctypes.byref(out)
        )
        if status not in (OK, EINFEASIBLE):
            raise OmniError(status, self._lib.last_error(), "resolve")
        return Plan(self._lib, out)


class Plan(_Handle):
    """A resolved plan (§10.5)."""

    _free_name = "omni_plan_free"

    @property
    def feasible(self) -> bool:
        return bool(self._lib.dll.omni_plan_feasible(self._live()))

    @property
    def resident_bytes(self) -> int:
        return self._lib.dll.omni_plan_resident_bytes(self._live())

    @property
    def read_bytes(self) -> int:
        return self._lib.dll.omni_plan_read_bytes(self._live())

    def as_json(self) -> str:
        out = ctypes.c_char_p()
        n = ctypes.c_size_t()
        self._lib.check(
            self._lib.dll.omni_plan_json(
                self._live(), ctypes.byref(out), ctypes.byref(n)
            ),
            "plan json",
        )
        return ctypes.string_at(out, n.value).decode()

    def as_dict(self) -> dict:
        return json.loads(self.as_json())

    @property
    def unmet(self) -> list:
        return self.as_dict().get("unmet", []) or []

    @property
    def warnings(self) -> list:
        return self.as_dict().get("warnings", []) or []


class Tensor(_Handle):
    """One tensor: its description always, its bytes or values on request."""

    _free_name = "omni_tensor_release"

    def __init__(self, lib: Library, ptr):
        super().__init__(lib, ptr)
        info = _TensorInfoC()
        status = lib.dll.omni_tensor_get_info(ptr, ctypes.byref(info))
        if status not in (OK, INDETERMINATE):
            raise OmniError(status, lib.last_error(), "tensor info")
        self.dtype = info.dtype.decode()
        self.dtype_bits = info.dtype_bits
        self.layout = info.layout.decode()
        self.value_op = info.value_op.decode()
        self.ndim = info.ndim
        # A symbolic dimension is genuinely unknown until the model's `dims` are
        # bound (§04.7.3). `None` says that; a zero would be a lie shaped like a
        # value.
        self.shape = (
            tuple(info.shape[i] for i in range(info.ndim)) if info.shape else None
        )
        self.numel = info.numel if info.shape else None
        self.symbolic = status == INDETERMINATE

    @property
    def name(self) -> str:
        out = ctypes.c_char_p()
        self._lib.check(
            self._lib.dll.omni_tensor_name(self._live(), ctypes.byref(out)), "name"
        )
        return out.value.decode()

    @property
    def mapped(self) -> bool:
        """True when the last `memory()` handed back a pointer into the
        container itself rather than a copy of it."""
        return bool(self._lib.dll.omni_tensor_mapped(self._live()))

    def memory(self) -> memoryview:
        """The stored bytes, as §04.3.5 lays them out, without a copy when the
        tensor is one raw object.

        The view aliases the library's memory and is valid until this tensor is
        closed — the same contract the C header states. `to_bytes()` copies if
        you need something that outlives the handle.

        Raises `OmniError` with `indeterminate` for a tensor whose value is
        computed rather than stored: a `dequantize` has no bytes of its own, and
        returning its operand would be the wrong array at the right length. Use
        `values()` for those.
        """
        ptr = ctypes.c_void_p()
        n = ctypes.c_size_t()
        self._lib.check(
            self._lib.dll.omni_tensor_bytes(
                self._live(), ctypes.byref(ptr), ctypes.byref(n)
            ),
            "bytes",
        )
        buf = (ctypes.c_ubyte * n.value).from_address(ptr.value)
        # ctypes spells its formats `<B`/`<d`; `cast` normalises them to the
        # native ones that slicing and `list()` understand.
        return memoryview(buf).cast("B").toreadonly()

    def to_bytes(self) -> bytes:
        return bytes(self.memory())

    def values(self) -> memoryview:
        """Every element as a double, whatever the value expression is: a stored
        literal is decoded through its dtype and layout, a `dequantize` is
        computed.

        This is the C1 path and it costs 8 bytes an element, which is why it is
        a separate call from `memory()` rather than the only way to read.

        Unlike `memory()`, the returned view is a Python-owned *copy*, not an
        alias of the library's buffer. The C library ties that buffer to the
        tensor handle and frees it on release, so a view that aliased it would
        dangle the instant a caller wrote the obvious `model[name].values()` —
        the temporary tensor is collected the moment `.values()` returns, before
        anything reads the view. Copying here is what makes that idiom sound;
        `memory()` is the zero-copy path and documents its lifetime.
        """
        ptr = ctypes.POINTER(ctypes.c_double)()
        n = ctypes.c_size_t()
        self._lib.check(
            self._lib.dll.omni_tensor_values(
                self._live(), ctypes.byref(ptr), ctypes.byref(n)
            ),
            "values",
        )
        buf = (ctypes.c_double * n.value).from_address(
            ctypes.cast(ptr, ctypes.c_void_p).value
        )
        # Copy into Python-owned memory before the handle can be released.
        return memoryview(bytearray(memoryview(buf).cast("B"))).cast("d").toreadonly()

    def tolist(self) -> list:
        return list(self.values())

    # -- DLPack ------------------------------------------------------------

    def __dlpack_device__(self):
        return (DLPACK_CPU, 0)

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
        """The DLPack producer protocol: `torch.from_dlpack(tensor)`,
        `np.from_dlpack(tensor)`, `jax.dlpack.from_dlpack(tensor)`.

        Zero copy when `mapped` is true. The capsule owns what it needs, so this
        tensor may be closed immediately afterwards.

        `stream` is accepted and ignored because the data is on the CPU, where
        DLPack defines no stream. A dtype DLPack cannot spell — `i4`, ternary,
        codebook, fixed point — raises `OmniError` with `indeterminate` rather
        than being passed off as `uint8`.
        """
        if dl_device is not None and tuple(dl_device) != (DLPACK_CPU, 0):
            raise BufferError(f"omni tensors are on the CPU, not {dl_device}")
        out = ctypes.POINTER(_DLManagedTensor)()
        self._lib.check(
            self._lib.dll.omni_tensor_dlpack(self._live(), ctypes.byref(out)),
            "dlpack",
        )
        return _PyCapsule_New(
            ctypes.cast(out, ctypes.c_void_p),
            b"dltensor",
            ctypes.cast(_CAPSULE_DESTRUCTOR_REF, ctypes.c_void_p),
        )

    def __repr__(self) -> str:
        shape = "symbolic" if self.shape is None else list(self.shape)
        return (
            f"<Tensor {self.name!r} {self.dtype} {shape} "
            f"{self.layout} {self.value_op}>"
        )


class Builder(_Handle):
    """A container being written (`omni_builder_*`).

    The binding was a reader for a draft, which meant anything that wanted to
    *publish* an OMNI container from Python had to shell out to the CLI or go
    through Rust. This is the writer, over the same ABI.

    Arrays arrive either as raw bytes with a declared dtype and shape, or —
    more usefully — through DLPack, which is how a `torch.Tensor`, a
    `numpy.ndarray`, a JAX array or an MLX array gets in without being flattened
    by hand first:

        with omni_ffi.Builder("acme/tiny") as b:
            b.arch("transformer.decoder", {"n_layers": 2})
            b.add_dlpack("model.embed_tokens.weight", weights)
            b.write("tiny.omni")

    The bytes are copied on the way in. A zero-copy writer would have to keep
    the caller's array alive until the pack, and a binding that holds a
    reference to a tensor somebody else may mutate is a worse bargain than a
    copy.
    """

    _free_name = "omni_builder_free"

    def __init__(self, name: str, lib: "Library | None" = None):
        lib = lib or _default_library()
        out = ctypes.c_void_p()
        lib.check(lib.dll.omni_builder_new(name.encode(), ctypes.byref(out)), name)
        super().__init__(lib, out)

    def __len__(self) -> int:
        return self._lib.dll.omni_builder_tensor_count(self._live())

    # -- metadata ----------------------------------------------------------

    def hash(self, algo: str) -> "Builder":
        """`"blake3-256"` or `"sha2-256"` (§03.5.1)."""
        self._lib.check(
            self._lib.dll.omni_builder_set_hash(self._live(), algo.encode()), algo
        )
        return self

    def license(self, spdx: str) -> "Builder":
        self._lib.check(
            self._lib.dll.omni_builder_set_license(self._live(), spdx.encode()), spdx
        )
        return self

    def arch(self, family: str, params: "dict | None" = None) -> "Builder":
        blob = json.dumps(params).encode() if params is not None else None
        self._lib.check(
            self._lib.dll.omni_builder_set_arch(self._live(), family.encode(), blob),
            family,
        )
        return self

    def meta(self, key: str, value) -> "Builder":
        self._lib.check(
            self._lib.dll.omni_builder_add_meta(
                self._live(), key.encode(), json.dumps(value).encode()
            ),
            key,
        )
        return self

    def chunk_size(self, n: int) -> "Builder":
        self._lib.check(self._lib.dll.omni_builder_set_chunk_size(self._live(), n))
        return self

    def codec(self, spec: str) -> "Builder":
        """`"raw"`, `"zstd:9"`, `"lz4:9"`, `"bitshuffle+zstd:9:2"` (§03.7.1)."""
        self._lib.check(
            self._lib.dll.omni_builder_set_codec(self._live(), spec.encode()), spec
        )
        return self

    def alignment(self, log2: int) -> "Builder":
        self._lib.check(self._lib.dll.omni_builder_set_alignment(self._live(), log2))
        return self

    # -- tensors -----------------------------------------------------------

    def add_tensor(
        self,
        name: str,
        dtype: str,
        shape,
        data,
        axes=None,
        semantic: "str | None" = None,
    ) -> "Builder":
        """Adds a tensor from bytes laid out as §04.3.5 says.

        `dtype` is a §04.3.6 alias (`"bf16"`, `"i4"`) or a §04.3 structural
        descriptor as a JSON string.
        """
        dims = (ctypes.c_uint64 * len(shape))(*shape)
        buf = bytes(data)
        self._lib.check(
            self._lib.dll.omni_builder_add_tensor(
                self._live(),
                name.encode(),
                dtype.encode(),
                dims,
                len(shape),
                ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p),
                len(buf),
            ),
            name,
        )
        if axes is not None:
            self.axes(name, axes)
        if semantic is not None:
            self.semantic(name, semantic)
        return self

    def add_dlpack(
        self, name: str, array, axes=None, semantic: "str | None" = None
    ) -> "Builder":
        """Adds a tensor from anything that speaks DLPack.

        Takes an object with `__dlpack__` (torch, numpy ≥ 1.23, jax, cupy,
        mlx), a raw capsule, or a `DLManagedTensor *`. The capsule is consumed
        the way DLPack requires — renamed and deleted — so the caller does not
        also free it.
        """
        managed = None
        capsule = None
        if hasattr(array, "__dlpack__"):
            capsule = array.__dlpack__()
        elif isinstance(array, ctypes.POINTER(_DLManagedTensor)):
            managed = array
        else:
            capsule = array
        if capsule is not None:
            managed = consume_capsule(capsule)
        try:
            self._lib.check(
                self._lib.dll.omni_builder_add_dlpack(
                    self._live(),
                    name.encode(),
                    ctypes.byref(managed.contents.dl_tensor),
                ),
                name,
            )
        finally:
            if capsule is not None and managed.contents.deleter:
                managed.contents.deleter(managed)
        if axes is not None:
            self.axes(name, axes)
        if semantic is not None:
            self.semantic(name, semantic)
        return self

    def axes(self, name: str, axes) -> "Builder":
        csv = axes if isinstance(axes, str) else ",".join(axes)
        self._lib.check(
            self._lib.dll.omni_builder_set_tensor_axes(
                self._live(), name.encode(), csv.encode()
            ),
            name,
        )
        return self

    def semantic(self, name: str, semantic: str) -> "Builder":
        self._lib.check(
            self._lib.dll.omni_builder_set_tensor_semantic(
                self._live(), name.encode(), semantic.encode()
            ),
            name,
        )
        return self

    # -- writing -----------------------------------------------------------

    @property
    def root_digest(self) -> bytes:
        """The root digest these calls would produce, before anything is
        written: identity is a function of content (§01.2)."""
        buf = (ctypes.c_uint8 * 32)()
        self._lib.check(self._lib.dll.omni_builder_root_digest(self._live(), buf))
        return bytes(buf)

    def write(self, path) -> "Builder":
        self._lib.check(
            self._lib.dll.omni_builder_write(self._live(), str(path).encode()), str(path)
        )
        return self

    def to_bytes(self) -> bytes:
        ptr = ctypes.POINTER(ctypes.c_uint8)()
        n = ctypes.c_size_t()
        self._lib.check(
            self._lib.dll.omni_builder_write_bytes(
                self._live(), ctypes.byref(ptr), ctypes.byref(n)
            )
        )
        return bytes(bytearray(ctypes.cast(ptr, _P(ctypes.c_uint8 * n.value)).contents))

    def __repr__(self) -> str:
        return f"<Builder {len(self)} tensors>"


# ---------------------------------------------------------------- entry API --


def open(path, lib: "Library | None" = None) -> Model:  # noqa: A001
    """Opens a container and returns its model.

    The store is kept alive by the model, so the usual one-liner works:

        with omni_ffi.open("model.omni") as model: ...
    """
    lib = lib or _default_library()
    out = ctypes.c_void_p()
    lib.check(lib.dll.omni_store_open(str(path).encode(), ctypes.byref(out)), str(path))
    store = Store(lib, out)
    model = store.root()
    # The Rust side reference-counts, so closing the store here would be safe
    # and pointless; keeping it lets the caller reach `model.store.verify()`.
    model.store = store
    return model


def open_bytes(data: bytes, lib: "Library | None" = None) -> Model:
    lib = lib or _default_library()
    out = ctypes.c_void_p()
    lib.check(
        lib.dll.omni_store_open_bytes(data, len(data), ctypes.byref(out)), "bytes"
    )
    store = Store(lib, out)
    model = store.root()
    model.store = store
    return model


def _main(argv) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <file.omni> [tensor]", file=sys.stderr)
        return 2
    lib = _default_library()
    print(f"library abi {lib.abi_version:08x}, spec {lib.spec_version}")
    model = open(argv[1], lib)
    store = model.store
    rep = store.verify(raise_on_error=False)
    print(
        f"{argv[1]}: {store.size} bytes, {store.object_count} objects, "
        f"{store.hash_name}, root {store.root_digest.hex()[:16]}…"
    )
    print(f"verify: {rep}")
    meta = model.meta()
    print(f"name: {meta.get('name', '(not stated)')}")
    if len(argv) > 2:
        t = model[argv[2]]
        print(t)
        try:
            data = t.memory()
            print(f"  {len(data)} stored bytes, {'mapped' if t.mapped else 'copied'}")
        except OmniError as e:
            print(f"  no stored bytes: {e}")
        vals = t.values()
        print(f"  {len(vals)} values, first: {list(vals[:6])}")
        return 0
    for name in model:
        t = model[name]
        print(f"  {t}")
        t.close()
    plan = model.resolve()
    print(
        f"plan against C0: {'feasible' if plan.feasible else 'infeasible'}, "
        f"{plan.resident_bytes} resident"
    )
    for u in plan.unmet:
        print(f"  unmet: {u.get('what', u)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
