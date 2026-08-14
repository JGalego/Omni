//! # omni-ffi — the OMNI C ABI
//!
//! `docs/design/sdk.md` §3 calls this "the binding substrate", and the argument
//! for it is the same one that puts the parser in one place: a format with ten
//! language bindings and ten parsers has ten CVE lists. Everything here is a
//! thin, ownership-explicit shell over [`omni_core`], which is where the actual
//! reading happens.
//!
//! ## Why this crate exists separately
//!
//! `omni-core` is `#![forbid(unsafe_code)]` and that is not a slogan — it parses
//! untrusted binary input (§12.4). A C ABI cannot be written without `unsafe`,
//! because turning a `const char *` from a caller into a `&str` is exactly the
//! operation the compiler cannot check. So the `unsafe` lives *here*, in a crate
//! that does no parsing, holds no invariants of its own, and whose entire job is
//! to move ownership across a boundary. Every `unsafe` block in this file is one
//! of three things: dereferencing a handle the caller passed back, reading a
//! caller's NUL-terminated string, or handing out a pointer whose lifetime an
//! `Arc` guarantees.
//!
//! ## The rules the design doc states, and where they are kept
//!
//! * **Never unwind across FFI.** Every entry point runs inside [`guard`], which
//!   catches panics and returns [`OMNI_EINTERNAL`].
//! * **The FFI layer never exposes a borrow** (§2.1). Handles hold
//!   `Arc<ContainerStore>`; a model or tensor keeps its store alive by itself,
//!   so `omni_store_close` on a store with live children is safe and does not
//!   free anything still in use.
//! * **Explicit ownership**, one `*_release`/`*_free`/`*_close` per acquiring
//!   call. Freeing twice is the caller's bug; freeing `NULL` is a no-op.
//! * **Status codes are the CLI's exit codes** (`docs/design/cli.md` §10),
//!   including [`OMNI_INDETERMINATE`], plus one the CLI has no need for:
//!   a caught panic is a return value here and a crash there.
//!
//! ## Safety contract for every entry point
//!
//! All exported functions are `unsafe` and share one contract, stated once here
//! rather than restated twenty times below. A caller must ensure that:
//!
//! * every handle pointer is either null or was returned by the matching
//!   constructor in this library and has not been freed;
//! * every `const char *` is either null (where the signature allows it) or
//!   points to a NUL-terminated byte string;
//! * every out-parameter pointer is either null or points to writable storage
//!   of the right type;
//! * a handle is not used from two threads at once. Handles are not internally
//!   synchronised and [`omni_last_error`] is thread-local.
//!
//! Nothing here trusts the *contents* of anything: a path that does not exist,
//! bytes that are not a container, a tensor name that is not in the table and a
//! digest that does not match are all ordinary error returns.

#![deny(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Arc;

use omni_core::cbor::Value;
use omni_core::container::{otype, Container, Digest};
use omni_core::expr::{Ctx, Expr, Ref};
use omni_core::plan::{Capabilities, Objective, Plan};
use omni_core::store::{Borrowed, ContainerStore};
use omni_core::tensor::{TensorDesc, TensorTable};
use omni_core::{json, DType, Layout};

// ---------------------------------------------------------------- versions --

/// The ABI version. The high 16 bits are the incompatible-change counter; the
/// low 16 bits are the additive-change counter. A caller that finds a different
/// high half than it was compiled against must not proceed (§14).
///
/// 1.1 added the writer (`omni_builder_*`). Every symbol from 1.0 kept its
/// signature and its meaning, which is what the low half is for.
pub const OMNI_ABI_VERSION: u32 = 0x0001_0001;

// ---------------------------------------------------------------- statuses --

/// Success.
pub const OMNI_OK: c_int = 0;
/// The file violates the specification.
pub const OMNI_EINVALID: c_int = 1;
/// The call itself was malformed: a null handle, a bad enum, an unknown name.
pub const OMNI_EUSAGE: c_int = 2;
/// Valid, but this build cannot fully verify, represent, or execute it.
pub const OMNI_INDETERMINATE: c_int = 3;
/// A policy refused it.
pub const OMNI_EPOLICY: c_int = 4;
/// Required objects are not available in any store.
pub const OMNI_EINCOMPLETE: c_int = 5;
/// Capability negotiation found no valid plan.
pub const OMNI_EINFEASIBLE: c_int = 6;
/// A panic was caught at the boundary. Always a bug in this library.
pub const OMNI_EINTERNAL: c_int = 7;

/// Objectives, matching [`Objective`] and the `--objective` flag.
pub const OMNI_OBJ_MIN_MEMORY: c_int = 0;
pub const OMNI_OBJ_MAX_QUALITY: c_int = 1;
pub const OMNI_OBJ_MIN_LOAD_TIME: c_int = 2;
pub const OMNI_OBJ_MIN_LATENCY: c_int = 3;
pub const OMNI_OBJ_BALANCED: c_int = 4;

// ------------------------------------------------------------------ errors --

struct Fail {
    status: c_int,
    msg: String,
}

impl Fail {
    fn new(status: c_int, msg: impl Into<String>) -> Fail {
        Fail {
            status,
            msg: msg.into(),
        }
    }
}

type R<T> = Result<T, Fail>;

fn usage(msg: impl Into<String>) -> Fail {
    Fail::new(OMNI_EUSAGE, msg)
}

/// Maps an expression-layer error onto a status. This is the one mapping that
/// matters: §15.1 says a value this build cannot evaluate is *indeterminate*,
/// not invalid, and a missing object is *incomplete*, not invalid. Collapsing
/// the three into "error" is how ecosystems fragment (§14.4).
fn from_expr(e: omni_core::expr::Error) -> Fail {
    use omni_core::expr::Error as E;
    let status = match &e {
        E::Type(_) => OMNI_EINVALID,
        E::Unsupported(_) => OMNI_INDETERMINATE,
        E::Missing(_) | E::External(_) => OMNI_EINCOMPLETE,
        E::Store(_) => OMNI_EINVALID,
        E::Bounds(_) => OMNI_EPOLICY,
    };
    Fail::new(status, e.to_string())
}

fn from_container(e: omni_core::container::Error) -> Fail {
    use omni_core::container::Error as E;
    let status = match &e {
        E::Io(_) => OMNI_EINCOMPLETE,
        E::NotFound(_) => OMNI_EINCOMPLETE,
        // An unsupported codec is not a malformed file; §03.7.1 lets a reader
        // meet a codec it does not have.
        E::Codec(m) if m.contains("unsupported") || m.contains("not implemented") => {
            OMNI_INDETERMINATE
        }
        E::Codec(_) | E::Rule(..) | E::Cbor(_) => OMNI_EINVALID,
    };
    Fail::new(status, e.to_string())
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: &str) {
    // A NUL inside an error message would truncate it; replace rather than drop
    // the message, so the caller still sees something true.
    let cleaned: String = msg.replace('\0', "?");
    let c = CString::new(cleaned).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Runs `f` with no way for a panic to reach the caller's frame.
fn guard<F: FnOnce() -> R<()>>(f: F) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            set_error("");
            OMNI_OK
        }
        Ok(Err(e)) => {
            set_error(&e.msg);
            e.status
        }
        Err(_) => {
            set_error("a panic was caught at the ABI boundary; this is a bug in omni-ffi");
            OMNI_EINTERNAL
        }
    }
}

// ----------------------------------------------------------------- handles --

/// An opened container. Reference-counted internally: models and tensors taken
/// from it hold their own share, so the order of frees does not matter.
pub struct omni_store {
    st: Arc<ContainerStore>,
    max_elems: u64,
}

/// The model asset inside a container, with its manifest and tensor table
/// already decoded.
pub struct omni_model {
    st: Arc<ContainerStore>,
    max_elems: u64,
    manifest: Value,
    model_ref: Ref,
    table: TensorTable,
    /// Names in §04.2 load order first, then anything the order omitted, so
    /// index `i` is a stable, meaningful iteration order rather than an
    /// alphabetical accident.
    names: Vec<CString>,
    meta_json: RefCell<Option<CString>>,
}

/// A resolved plan (§10.5).
pub struct omni_plan {
    plan: Plan,
    json: RefCell<Option<CString>>,
}

/// One tensor's description, and its bytes once asked for.
pub struct omni_tensor {
    st: Arc<ContainerStore>,
    max_elems: u64,
    desc: TensorDesc,
    name: CString,
    dtype: CString,
    layout: CString,
    op: CString,
    /// Concrete extents, or empty when a dimension is symbolic (§04.7.3).
    shape: Vec<u64>,
    shape_i64: Vec<i64>,
    symbolic: bool,
    bytes: RefCell<Option<Bytes>>,
    values: RefCell<Option<Vec<f64>>>,
}

/// Where a tensor's bytes are. `Mapped` is a range inside the container the
/// store already holds — the zero-copy case, and the one `Bytes::Mapped` in
/// §2 of the SDK design is about. `Owned` is everything that had to be built:
/// decompressed, concatenated from chunks, or computed.
enum Bytes {
    Mapped { off: usize, len: usize },
    Owned(Arc<Vec<u8>>),
}

// ------------------------------------------------------------- conversions --

/// # Safety
/// See the module-level contract.
unsafe fn as_str<'a>(p: *const c_char, what: &str) -> R<&'a str> {
    if p.is_null() {
        return Err(usage(format!("{what} is null")));
    }
    // SAFETY: the contract requires a NUL-terminated string.
    let c = unsafe { CStr::from_ptr(p) };
    c.to_str()
        .map_err(|_| usage(format!("{what} is not valid UTF-8")))
}

/// # Safety
/// See the module-level contract.
unsafe fn handle<'a, T>(p: *mut T, what: &str) -> R<&'a T> {
    if p.is_null() {
        return Err(usage(format!("{what} is null")));
    }
    // SAFETY: the contract requires a live handle from this library.
    Ok(unsafe { &*p })
}

/// # Safety
/// See the module-level contract.
unsafe fn out<T>(p: *mut T, what: &str, v: T) -> R<()> {
    if p.is_null() {
        return Err(usage(format!("{what} is null")));
    }
    // SAFETY: the contract requires writable storage of this type.
    unsafe { p.write(v) };
    Ok(())
}

/// Projects a CBOR value onto JSON. The projection is lossy in exactly three
/// documented ways, and all three are visible in the output rather than silent:
/// byte strings become hex text, a tag becomes `{"@tag":n,"value":…}`, and a
/// §01.3 ref, *when it carries tag 1001*, becomes `{"@ref":{"t":otype,"d":hex}}`.
/// A ref written in its bare form stays the two-element array §01.3 defines it
/// as — `[otype, "<hex>"]` — because reinterpreting an untagged array by
/// guessing at its shape is how a projection starts lying.
/// Non-text map keys become their debug form, which no OMNI object produces.
fn to_json(v: &Value) -> json::Value {
    if let Value::Tag(omni_core::cbor::TAG_REF, inner) = v {
        if let Some(a) = inner.as_array() {
            if let (Some(t), Some(d)) = (
                a.first().and_then(|x| x.as_u64()),
                a.get(1).and_then(|x| x.as_bytes()),
            ) {
                return json::object(vec![(
                    "@ref",
                    json::object(vec![
                        ("t", json::Value::U(t)),
                        ("d", json::Value::Str(omni_core::hex(d))),
                    ]),
                )]);
            }
        }
    }
    match v {
        Value::U(u) => json::Value::U(*u),
        Value::I(i) => json::Value::I(*i),
        Value::Bool(b) => json::Value::Bool(*b),
        Value::Null => json::Value::Null,
        Value::F64(f) => json::Value::F(*f),
        Value::Text(s) => json::Value::Str(s.clone()),
        Value::Bytes(b) => json::Value::Str(omni_core::hex(b)),
        Value::Array(a) => json::Value::Array(a.iter().map(to_json).collect()),
        Value::Tag(t, inner) => json::object(vec![
            ("@tag", json::Value::U(*t)),
            ("value", to_json(inner)),
        ]),
        Value::Map(m) => {
            let mut o = std::collections::BTreeMap::new();
            for (k, val) in m {
                let key = match k {
                    Value::Text(s) => s.clone(),
                    other => format!("{other:?}"),
                };
                o.insert(key, to_json(val));
            }
            json::Value::Object(o)
        }
    }
}

/// The other direction, for capabilities handed in as JSON. JSON has no byte
/// strings and no tags, so this is total.
fn from_json(v: &json::Value) -> Value {
    match v {
        json::Value::Null => Value::Null,
        json::Value::Bool(b) => Value::Bool(*b),
        json::Value::U(u) => Value::U(*u),
        json::Value::I(i) => Value::I(*i),
        json::Value::F(f) => Value::F64(*f),
        json::Value::Str(s) => Value::Text(s.clone()),
        json::Value::Array(a) => Value::Array(a.iter().map(from_json).collect()),
        json::Value::Object(m) => Value::Map(
            m.iter()
                .map(|(k, val)| (Value::Text(k.clone()), from_json(val)))
                .collect(),
        ),
    }
}

fn cstring(s: impl Into<Vec<u8>>) -> CString {
    CString::new(s).unwrap_or_default()
}

// ------------------------------------------------------------------- store --

/// The ABI version this library was built with.
#[no_mangle]
pub extern "C" fn omni_abi_version() -> u32 {
    OMNI_ABI_VERSION
}

/// The specification version this build targets, as a static string.
#[no_mangle]
pub extern "C" fn omni_spec_version() -> *const c_char {
    // A `static` C string, so there is no allocation and no lifetime question.
    concat!("OMNI/1.0-draft", "\0").as_ptr() as *const c_char
}

/// The last error on *this thread*, as a NUL-terminated string. Never null;
/// empty when the last call succeeded. Valid until the next call on this
/// thread.
#[no_mangle]
pub extern "C" fn omni_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// A stable, human-readable name for a status code.
#[no_mangle]
pub extern "C" fn omni_status_name(status: c_int) -> *const c_char {
    let s = match status {
        OMNI_OK => "ok\0",
        OMNI_EINVALID => "invalid\0",
        OMNI_EUSAGE => "usage\0",
        OMNI_INDETERMINATE => "indeterminate\0",
        OMNI_EPOLICY => "policy\0",
        OMNI_EINCOMPLETE => "incomplete\0",
        OMNI_EINFEASIBLE => "infeasible\0",
        OMNI_EINTERNAL => "internal\0",
        _ => "unknown\0",
    };
    s.as_ptr() as *const c_char
}

fn open_bytes(bytes: Vec<u8>) -> R<*mut omni_store> {
    let c = Container::open(bytes).map_err(from_container)?;
    let st = Arc::new(ContainerStore::new(c));
    Ok(Box::into_raw(Box::new(omni_store {
        st,
        max_elems: 1 << 28,
    })))
}

/// Opens a container from a filesystem path.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_open(path: *const c_char, o: *mut *mut omni_store) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let p = unsafe { as_str(path, "path")? };
        let bytes = std::fs::read(p)
            .map_err(|e| Fail::new(OMNI_EINCOMPLETE, format!("cannot read `{p}`: {e}")))?;
        let s = open_bytes(bytes)?;
        // SAFETY: the contract.
        unsafe { out(o, "out", s) }
    })
}

/// Opens a container from bytes the caller already has. The bytes are copied;
/// the caller may free them on return.
///
/// # Safety
/// See the module-level contract. `bytes` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn omni_store_open_bytes(
    bytes: *const u8,
    len: usize,
    o: *mut *mut omni_store,
) -> c_int {
    guard(|| {
        if bytes.is_null() && len != 0 {
            return Err(usage("bytes is null"));
        }
        // SAFETY: the contract requires `len` readable bytes at `bytes`.
        let v = unsafe { std::slice::from_raw_parts(bytes, len) }.to_vec();
        let s = open_bytes(v)?;
        // SAFETY: the contract.
        unsafe { out(o, "out", s) }
    })
}

/// Releases a store handle. Any model or tensor taken from it stays valid;
/// the underlying container lives until the last of them is released.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_close(s: *mut omni_store) {
    if s.is_null() {
        return;
    }
    // SAFETY: the contract requires a handle from `omni_store_open*`.
    drop(unsafe { Box::from_raw(s) });
}

/// The container's total size in bytes.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_size(s: *mut omni_store) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*s).st.container().bytes.len() as u64 }
}

/// The number of objects in the container's index.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_object_count(s: *mut omni_store) -> u64 {
    if s.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*s).st.container().index.len() as u64 }
}

/// The container's hash algorithm, as its §03.5.1 name (`blake3-256`,
/// `sha2-256`). Static; valid for the process.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_hash_name(s: *mut omni_store, o: *mut *const c_char) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let s = unsafe { handle(s, "store")? };
        let name = match s.st.container().header.hash {
            omni_core::HashAlgo::Blake3_256 => "blake3-256\0",
            omni_core::HashAlgo::Sha256 => "sha2-256\0",
        };
        // SAFETY: the contract.
        unsafe { out(o, "out", name.as_ptr() as *const c_char) }
    })
}

/// Copies the root digest into a caller-provided 32-byte buffer.
///
/// # Safety
/// See the module-level contract. `d32` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn omni_store_root_digest(s: *mut omni_store, d32: *mut u8) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let s = unsafe { handle(s, "store")? };
        if d32.is_null() {
            return Err(usage("digest buffer is null"));
        }
        let root = s.st.container().header.root_digest;
        // SAFETY: the contract requires 32 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(root.as_ptr(), d32, 32) };
        Ok(())
    })
}

/// Raises this store's per-node materialization cap (§12.4). The default is
/// 2^28 elements. It exists because a declared size is untrusted input; a
/// caller that means to load a 70 B model is making that decision explicitly.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_set_max_elems(s: *mut omni_store, n: u64) -> c_int {
    guard(|| {
        if s.is_null() {
            return Err(usage("store is null"));
        }
        // SAFETY: the contract; no other reference to this handle exists per
        // the one-thread rule.
        unsafe { (*s).max_elems = n };
        Ok(())
    })
}

/// What a full verification found.
#[repr(C)]
pub struct omni_verify_report {
    pub segments: u64,
    pub objects_verified: u64,
    pub bytes_verified: u64,
    pub reachable: u64,
    pub dangling: u64,
    pub mistyped: u64,
    pub padding_ok: c_int,
    pub alignment_ok: c_int,
}

/// Verifies every object's digest, the framing, and reachability. Returns
/// `OMNI_OK` when clean, `OMNI_EINCOMPLETE` when a referenced object is absent,
/// and `OMNI_EINVALID` when the file breaks a rule. `report` may be null.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_verify(
    s: *mut omni_store,
    report: *mut omni_verify_report,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let s = unsafe { handle(s, "store")? };
        let r = omni_core::verify(s.st.container()).map_err(from_container)?;
        if !report.is_null() {
            // SAFETY: the contract.
            unsafe {
                report.write(omni_verify_report {
                    segments: r.segments.len() as u64,
                    objects_verified: r.objects_verified as u64,
                    bytes_verified: r.bytes_verified,
                    reachable: r.reachable as u64,
                    dangling: r.dangling.len() as u64,
                    mistyped: r.mistyped.len() as u64,
                    padding_ok: r.padding_ok as c_int,
                    alignment_ok: r.alignment_ok as c_int,
                })
            };
        }
        if !r.mistyped.is_empty() {
            return Err(Fail::new(
                OMNI_EINVALID,
                format!(
                    "R-O02: {} object(s) are not the type the index claims",
                    r.mistyped.len()
                ),
            ));
        }
        if !r.padding_ok || !r.alignment_ok {
            return Err(Fail::new(
                OMNI_EINVALID,
                "the container's padding or alignment breaks §02",
            ));
        }
        if !r.dangling.is_empty() {
            return Err(Fail::new(
                OMNI_EINCOMPLETE,
                format!("{} referenced object(s) are not present", r.dangling.len()),
            ));
        }
        Ok(())
    })
}

// ------------------------------------------------------------------- model --

fn model_of(s: &omni_store) -> R<*mut omni_model> {
    let c = s.st.container();
    let manifest = c.root().map_err(from_container)?;
    let model_ref = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(parse_ref)
        .ok_or_else(|| Fail::new(OMNI_EINVALID, "this container has no `model` asset"))?;
    let model = c.get_value(&model_ref.1).map_err(from_container)?;
    let tt = model
        .get("tensors")
        .and_then(parse_ref)
        .ok_or_else(|| Fail::new(OMNI_EINVALID, "the model has no tensor table"))?;
    let table =
        TensorTable::from_value(&c.get_value(&tt.1).map_err(from_container)?).map_err(from_expr)?;

    // Load order first (§04.2), then anything it left out. A reader iterating
    // by index gets the order the producer asked for, not an alphabetisation
    // of it.
    let mut names: Vec<CString> = Vec::with_capacity(table.tensors.len());
    let mut seen = std::collections::BTreeSet::new();
    for n in &table.order {
        if table.tensors.contains_key(n) && seen.insert(n.clone()) {
            names.push(cstring(n.as_str()));
        }
    }
    for n in table.tensors.keys() {
        if seen.insert(n.clone()) {
            names.push(cstring(n.as_str()));
        }
    }

    Ok(Box::into_raw(Box::new(omni_model {
        st: s.st.clone(),
        max_elems: s.max_elems,
        manifest,
        model_ref,
        table,
        names,
        meta_json: RefCell::new(None),
    })))
}

fn parse_ref(v: &Value) -> Option<Ref> {
    omni_core::expr::parse_ref_value(v).ok()
}

/// Takes the container's model asset.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_store_root(s: *mut omni_store, o: *mut *mut omni_model) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let s = unsafe { handle(s, "store")? };
        let m = model_of(s)?;
        // SAFETY: the contract.
        unsafe { out(o, "out", m) }
    })
}

/// Releases a model handle.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_free(m: *mut omni_model) {
    if m.is_null() {
        return;
    }
    // SAFETY: the contract.
    drop(unsafe { Box::from_raw(m) });
}

/// The manifest as JSON, cached on the handle and valid until it is freed.
/// `len` may be null.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_meta_json(
    m: *mut omni_model,
    o: *mut *const c_char,
    len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let m = unsafe { handle(m, "model")? };
        let mut slot = m.meta_json.borrow_mut();
        if slot.is_none() {
            *slot = Some(cstring(to_json(&m.manifest).encode()));
        }
        let c = slot.as_ref().expect("just filled");
        if !len.is_null() {
            // SAFETY: the contract.
            unsafe { len.write(c.as_bytes().len()) };
        }
        // SAFETY: the contract.
        unsafe { out(o, "out", c.as_ptr()) }
    })
}

/// How many tensors the model declares.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_tensor_count(m: *mut omni_model) -> usize {
    if m.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*m).names.len() }
}

/// The `i`th tensor name, in §04.2 load order. Valid until the model is freed.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_tensor_name(
    m: *mut omni_model,
    i: usize,
    o: *mut *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let m = unsafe { handle(m, "model")? };
        let n = m.names.get(i).ok_or_else(|| {
            usage(format!(
                "tensor index {i} is past the end ({})",
                m.names.len()
            ))
        })?;
        // SAFETY: the contract.
        unsafe { out(o, "out", n.as_ptr()) }
    })
}

/// Takes one tensor by name. Nothing is read from the payload yet: this decodes
/// the description only, which is the §02.1 promise that opening a model does
/// not touch its weights.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_tensor(
    m: *mut omni_model,
    name: *const c_char,
    o: *mut *mut omni_tensor,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let m = unsafe { handle(m, "model")? };
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        let r = m
            .table
            .get(name)
            .ok_or_else(|| usage(format!("no tensor named `{name}`")))?;
        let v = m.st.container().get_value(&r.1).map_err(from_container)?;
        let desc = TensorDesc::from_value(&v).map_err(from_expr)?;
        let shape = desc.sizes().unwrap_or_default();
        let symbolic = desc.sizes().is_none();
        let t = omni_tensor {
            st: m.st.clone(),
            max_elems: m.max_elems,
            name: cstring(name),
            dtype: cstring(desc.dtype.label()),
            layout: cstring(desc.layout.kind()),
            op: cstring(desc.value.op()),
            shape_i64: shape.iter().map(|&d| d as i64).collect(),
            shape,
            symbolic,
            desc,
            bytes: RefCell::new(None),
            values: RefCell::new(None),
        };
        // SAFETY: the contract.
        unsafe { out(o, "out", Box::into_raw(Box::new(t))) }
    })
}

// -------------------------------------------------------------------- plan --

/// Negotiates a plan (§10.5). `caps_json` may be null, which means the `C0`
/// baseline — the floor every conforming reader meets. `objective` is one of
/// the `OMNI_OBJ_*` constants.
///
/// Returns `OMNI_EINFEASIBLE` when no plan satisfies the capabilities. The plan
/// is still produced in that case, so the caller can read `omni_plan_json` to
/// find out what was unmet.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_model_resolve(
    m: *mut omni_model,
    caps_json: *const c_char,
    objective: c_int,
    o: *mut *mut omni_plan,
) -> c_int {
    let mut infeasible = false;
    let status = guard(|| {
        // SAFETY: the contract.
        let m = unsafe { handle(m, "model")? };
        let obj = match objective {
            OMNI_OBJ_MIN_MEMORY => Objective::MinMemory,
            OMNI_OBJ_MAX_QUALITY => Objective::MaxQuality,
            OMNI_OBJ_MIN_LOAD_TIME => Objective::MinLoadTime,
            OMNI_OBJ_MIN_LATENCY => Objective::MinLatency,
            OMNI_OBJ_BALANCED => Objective::Balanced,
            n => return Err(usage(format!("objective {n} is not one of OMNI_OBJ_*"))),
        };
        let caps = if caps_json.is_null() {
            Capabilities::c0()
        } else {
            // SAFETY: the contract.
            let s = unsafe { as_str(caps_json, "caps_json")? };
            let parsed =
                json::parse(s.as_bytes()).map_err(|e| usage(format!("capabilities: {e}")))?;
            Capabilities::from_value(&from_json(&parsed)).map_err(from_expr)?
        };
        let store = Borrowed(m.st.container());
        let ctx = Ctx::new(&store).max_elems(m.max_elems);
        let plan =
            omni_core::plan::resolve(&ctx, &m.manifest, m.model_ref, &m.table, &caps, obj, false)
                .map_err(from_expr)?;
        infeasible = !plan.is_feasible();
        let p = Box::into_raw(Box::new(omni_plan {
            plan,
            json: RefCell::new(None),
        }));
        // SAFETY: the contract.
        unsafe { out(o, "out", p) }
    });
    if status == OMNI_OK && infeasible {
        set_error("no representation satisfies these capabilities; see the plan's `unmet`");
        return OMNI_EINFEASIBLE;
    }
    status
}

/// Whether the plan covers every tensor.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_plan_feasible(p: *mut omni_plan) -> c_int {
    if p.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*p).plan.is_feasible() as c_int }
}

/// Bytes resident after instantiation, as the plan predicts them.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_plan_resident_bytes(p: *mut omni_plan) -> u64 {
    if p.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*p).plan.resident_bytes }
}

/// Bytes that must be read to instantiate.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_plan_read_bytes(p: *mut omni_plan) -> u64 {
    if p.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    unsafe { (*p).plan.read_bytes }
}

/// The plan as JSON, including its warnings and everything unmet. Cached on the
/// handle and valid until it is freed. `len` may be null.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_plan_json(
    p: *mut omni_plan,
    o: *mut *const c_char,
    len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let p = unsafe { handle(p, "plan")? };
        let mut slot = p.json.borrow_mut();
        if slot.is_none() {
            *slot = Some(cstring(to_json(&p.plan.to_value()).encode()));
        }
        let c = slot.as_ref().expect("just filled");
        if !len.is_null() {
            // SAFETY: the contract.
            unsafe { len.write(c.as_bytes().len()) };
        }
        // SAFETY: the contract.
        unsafe { out(o, "out", c.as_ptr()) }
    })
}

/// Releases a plan handle.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_plan_free(p: *mut omni_plan) {
    if p.is_null() {
        return;
    }
    // SAFETY: the contract.
    drop(unsafe { Box::from_raw(p) });
}

// ------------------------------------------------------------------ tensor --

/// Everything known about a tensor without reading its payload.
#[repr(C)]
pub struct omni_tensor_info {
    /// The §04.3 dtype label, e.g. `bf16`, `i4`, `q8.8`.
    pub dtype: *const c_char,
    /// Bits per element, rounded up.
    pub dtype_bits: u32,
    /// The §04.4 layout kind, e.g. `strided`, `packed`.
    pub layout: *const c_char,
    /// The §04.7 value node at the root, e.g. `literal`, `dequantize`.
    pub value_op: *const c_char,
    pub ndim: u32,
    /// `ndim` extents, or null when any dimension is symbolic.
    pub shape: *const u64,
    /// Product of the extents, or 0 when the shape is symbolic.
    pub numel: u64,
}

/// Fills in a tensor's description.
///
/// Returns `OMNI_INDETERMINATE`, with everything except `shape` and `numel`
/// filled in, when a dimension is symbolic (§04.7.3): the extent is genuinely
/// unknown until the model's `dims` are bound, and reporting 0 would be a lie
/// that looks like a value (`docs/design/cli.md` §10.2).
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_get_info(
    t: *mut omni_tensor,
    o: *mut omni_tensor_info,
) -> c_int {
    let mut symbolic = false;
    let status = guard(|| {
        // SAFETY: the contract.
        let t = unsafe { handle(t, "tensor")? };
        symbolic = t.symbolic;
        let info = omni_tensor_info {
            dtype: t.dtype.as_ptr(),
            dtype_bits: t.desc.dtype.bits(),
            layout: t.layout.as_ptr(),
            value_op: t.op.as_ptr(),
            ndim: t.desc.shape.len() as u32,
            shape: if t.symbolic {
                std::ptr::null()
            } else {
                t.shape.as_ptr()
            },
            numel: if t.symbolic {
                0
            } else {
                t.shape.iter().product::<u64>()
            },
        };
        // SAFETY: the contract.
        unsafe { out(o, "out", info) }
    });
    if status == OMNI_OK && symbolic {
        set_error("a dimension is symbolic; bind the model's `dims` before asking for extents");
        return OMNI_INDETERMINATE;
    }
    status
}

/// The tensor's own name, as the model's table spells it. Valid until the
/// tensor is released.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_name(t: *mut omni_tensor, o: *mut *const c_char) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let t = unsafe { handle(t, "tensor")? };
        // SAFETY: the contract.
        unsafe { out(o, "out", t.name.as_ptr()) }
    })
}

/// Loads a tensor's stored bytes, borrowing them from the container when they
/// are there contiguously and uncompressed.
fn load_bytes(t: &omni_tensor) -> R<()> {
    if t.bytes.borrow().is_some() {
        return Ok(());
    }
    let Expr::Literal { chunks, .. } = &t.desc.value else {
        return Err(Fail::new(
            OMNI_INDETERMINATE,
            format!(
                "`{}` is a `{}` expression, not stored bytes; ask for values instead",
                t.name.to_string_lossy(),
                t.desc.value.op()
            ),
        ));
    };
    let c = t.st.container();

    // The mappable case: one object, stored raw. §02.4's whole point is that
    // this is the common case, so it is worth detecting rather than always
    // copying.
    let single: Option<Digest> = if chunks.0 == otype::BLOB {
        Some(chunks.1)
    } else {
        c.get_value(&chunks.1).ok().and_then(|cl| {
            let arr = cl.get("chunks")?.as_array()?;
            if arr.len() != 1 {
                return None;
            }
            arr[0].get("r").and_then(parse_ref).map(|r| r.1)
        })
    };
    if let Some(d) = single {
        if let Ok(slice) = c.get(&d) {
            // Casting a pointer to an integer is not `unsafe`; the range is
            // recorded rather than the pointer so nothing here outlives the
            // `Arc` that keeps the container alive.
            let base = c.bytes.as_ptr() as usize;
            let off = slice.as_ptr() as usize - base;
            *t.bytes.borrow_mut() = Some(Bytes::Mapped {
                off,
                len: slice.len(),
            });
            return Ok(());
        }
    }

    let store = Borrowed(c);
    let ctx = Ctx::new(&store).max_elems(t.max_elems);
    let owned = ctx.chunk_bytes(chunks).map_err(from_expr)?;
    *t.bytes.borrow_mut() = Some(Bytes::Owned(Arc::new(owned)));
    Ok(())
}

fn byte_slice(t: &omni_tensor) -> (*const u8, usize) {
    let b = t.bytes.borrow();
    match b.as_ref() {
        Some(Bytes::Mapped { off, len }) => (t.st.container().bytes[*off..].as_ptr(), *len),
        Some(Bytes::Owned(v)) => (v.as_ptr(), v.len()),
        None => (std::ptr::null(), 0),
    }
}

/// The tensor's stored bytes, exactly as §04.3.5 lays them out. Valid until the
/// tensor is released. Fails with `OMNI_INDETERMINATE` for a tensor whose value
/// is computed rather than stored — a `dequantize` has no stored bytes of its
/// own, and pretending otherwise would hand back its operand.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_bytes(
    t: *mut omni_tensor,
    ptr: *mut *const c_void,
    len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let t = unsafe { handle(t, "tensor")? };
        load_bytes(t)?;
        let (p, n) = byte_slice(t);
        if !len.is_null() {
            // SAFETY: the contract.
            unsafe { len.write(n) };
        }
        // SAFETY: the contract.
        unsafe { out(ptr, "ptr", p as *const c_void) }
    })
}

/// 1 when the last `omni_tensor_bytes` handed back a pointer into the container
/// itself rather than a copy. This is the `Bytes::Mapped` / `Bytes::Owned`
/// distinction of the SDK design, made observable so a caller can tell whether
/// it got zero copy or paid for one.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_mapped(t: *mut omni_tensor) -> c_int {
    if t.is_null() {
        return 0;
    }
    // SAFETY: the contract.
    let t = unsafe { &*t };
    matches!(t.bytes.borrow().as_ref(), Some(Bytes::Mapped { .. })) as c_int
}

/// Evaluates the tensor to `f64` elements, whatever its value expression is:
/// a stored literal is decoded through its dtype and layout, and a
/// `dequantize` is computed. Cached on the handle; valid until it is released.
///
/// This is the C1 path. It allocates 8 bytes per element, which is why it is a
/// separate call from `omni_tensor_bytes` rather than the only way to read.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_values(
    t: *mut omni_tensor,
    ptr: *mut *const f64,
    len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let t = unsafe { handle(t, "tensor")? };
        if t.values.borrow().is_none() {
            let store = Borrowed(t.st.container());
            let ctx = Ctx::new(&store).max_elems(t.max_elems);
            let tensor = t.desc.value.eval(&ctx).map_err(from_expr)?;
            *t.values.borrow_mut() = Some(tensor.data);
        }
        let v = t.values.borrow();
        let v = v.as_ref().expect("just filled");
        if !len.is_null() {
            // SAFETY: the contract.
            unsafe { len.write(v.len()) };
        }
        // SAFETY: the contract.
        unsafe { out(ptr, "ptr", v.as_ptr()) }
    })
}

/// Releases a tensor handle.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_release(t: *mut omni_tensor) {
    if t.is_null() {
        return;
    }
    // SAFETY: the contract.
    drop(unsafe { Box::from_raw(t) });
}

// ------------------------------------------------------------------ DLPack --

/// `kDLCPU`.
pub const OMNI_DLPACK_CPU: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    /// `kDLInt` 0, `kDLUInt` 1, `kDLFloat` 2, `kDLBfloat` 4, `kDLBool` 6.
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// What a handed-out `DLManagedTensor` keeps alive. The consumer owns this
/// until it calls the deleter, which is the DLPack contract and the reason the
/// FFI layer never exposes a borrow (§2.1).
struct DlOwner {
    managed: DLManagedTensor,
    shape: Vec<i64>,
    _store: Arc<ContainerStore>,
    _owned: Option<Arc<Vec<u8>>>,
}

unsafe extern "C" fn dl_deleter(this: *mut DLManagedTensor) {
    if this.is_null() {
        return;
    }
    // SAFETY: `manager_ctx` was set from `Box::into_raw(Box<DlOwner>)` in
    // `omni_tensor_dlpack`, and DLPack says the deleter is called at most once.
    let ctx = unsafe { (*this).manager_ctx } as *mut DlOwner;
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx) });
    }
}

/// DLPack's encoding of a dtype, or the reason there isn't one.
///
/// DLPack describes whole-byte lanes. OMNI's §04.3 dtypes include 4-bit ints,
/// ternary, fixed point and codebook indices, and there is no honest DLPack
/// spelling for any of them — a `q4` handed over as `uint8` would be silently
/// wrong twice over. Those are refused by name; the caller still has
/// `omni_tensor_values`.
fn dl_dtype(d: &DType) -> Result<DLDataType, String> {
    let lanes = 1;
    if *d == DType::F16 {
        return Ok(DLDataType {
            code: 2,
            bits: 16,
            lanes,
        });
    }
    if *d == DType::BF16 {
        return Ok(DLDataType {
            code: 4,
            bits: 16,
            lanes,
        });
    }
    if *d == DType::F32 {
        return Ok(DLDataType {
            code: 2,
            bits: 32,
            lanes,
        });
    }
    if *d == DType::F64 {
        return Ok(DLDataType {
            code: 2,
            bits: 64,
            lanes,
        });
    }
    match d {
        DType::Bool => Ok(DLDataType {
            code: 6,
            bits: 8,
            lanes,
        }),
        DType::Int { w, signed } if matches!(w, 8 | 16 | 32 | 64) => Ok(DLDataType {
            code: if *signed { 0 } else { 1 },
            bits: *w as u8,
            lanes,
        }),
        other => Err(format!(
            "DLPack has no encoding for `{}`; it describes whole-byte lanes and \
             this dtype is not one",
            other.label()
        )),
    }
}

/// Hands the tensor to any DLPack consumer — PyTorch, JAX, CuPy, NumPy, MLX —
/// without a copy when the bytes were mappable.
///
/// The caller owns the returned `DLManagedTensor` and must call its `deleter`.
/// After this call the tensor handle may be released; the DLPack object keeps
/// what it needs alive on its own.
///
/// Refuses with `OMNI_INDETERMINATE`, naming the reason, when the dtype has no
/// DLPack spelling (`i4`, ternary, codebook, fixed point) or when the layout is
/// not dense row-major, because DLPack's `strides == NULL` means exactly that
/// and a tiled or packed buffer described as dense would be read wrongly.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_tensor_dlpack(
    t: *mut omni_tensor,
    o: *mut *mut DLManagedTensor,
) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let t = unsafe { handle(t, "tensor")? };
        if t.symbolic {
            return Err(Fail::new(
                OMNI_INDETERMINATE,
                "a dimension is symbolic; DLPack needs concrete extents",
            ));
        }
        let dtype = dl_dtype(&t.desc.dtype).map_err(|m| Fail::new(OMNI_INDETERMINATE, m))?;
        match &t.desc.layout {
            Layout::Strided {
                order: omni_core::layout::Order::RowMajor,
                strides: None,
                offset: 0,
            } => {}
            other => {
                return Err(Fail::new(
                    OMNI_INDETERMINATE,
                    format!(
                        "DLPack with null strides means dense row-major; this tensor is `{}`",
                        other.kind()
                    ),
                ))
            }
        }
        load_bytes(t)?;
        let (ptr, len) = byte_slice(t);
        let need = t.desc.dtype.packed_bytes(t.shape.iter().product::<u64>());
        if (len as u64) < need {
            return Err(Fail::new(
                OMNI_EINVALID,
                format!("R-T02: the shape needs {need} bytes and only {len} are stored"),
            ));
        }
        let owned = match t.bytes.borrow().as_ref() {
            Some(Bytes::Owned(v)) => Some(v.clone()),
            _ => None,
        };

        let mut owner = Box::new(DlOwner {
            managed: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: ptr as *mut c_void,
                    device: DLDevice {
                        device_type: OMNI_DLPACK_CPU,
                        device_id: 0,
                    },
                    ndim: t.shape.len() as i32,
                    dtype,
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dl_deleter),
            },
            shape: t.shape_i64.clone(),
            _store: t.st.clone(),
            _owned: owned,
        });
        owner.managed.dl_tensor.shape = owner.shape.as_mut_ptr();
        let raw = Box::into_raw(owner);
        // SAFETY: `raw` came from `Box::into_raw` and is uniquely owned here.
        // Taking an interior pointer is sound because the allocation does not
        // move for as long as the box exists, and the deleter frees it whole.
        let managed = unsafe {
            (*raw).managed.manager_ctx = raw as *mut c_void;
            std::ptr::addr_of_mut!((*raw).managed)
        };
        // SAFETY: the contract.
        unsafe { out(o, "out", managed) }
    })
}

// ------------------------------------------------------------------ writer --

/// A container being built (`docs/design/sdk.md` §3.4).
///
/// The read path above was the whole of this ABI for a draft, and the gap was
/// stated where it mattered: a C caller could open, verify, walk, evaluate and
/// hand out a model and could not *make* one, so every binding that wanted to
/// write went back through Rust. This is the other half. It is deliberately the
/// small half — [`ModelBuilder`] already turns named byte buffers into the
/// object graph of §00.4, and what a C ABI adds is a way to name the buffers.
///
/// Nothing here decides anything the format leaves to a writer. The digest
/// algorithm, the chunk size, the storage codec and the alignment are all the
/// caller's, and packing stays reproducible (§01.10, writer rule W1): the same
/// calls in the same order produce the same bytes, and so do the same calls in
/// a different order, because objects are placed by digest and not by arrival.
pub struct omni_builder {
    b: RefCell<omni_core::ModelBuilder>,
    codec: RefCell<omni_core::codec::Codec>,
    log2_align: std::cell::Cell<u8>,
    /// The last `omni_builder_write_bytes` result, kept alive for the caller.
    bytes: RefCell<Option<Arc<Vec<u8>>>>,
}

/// Parses a §04.3.6 alias, or a full §04.3 structural descriptor as JSON.
///
/// The alias is what a caller writing `"bf16"` means, and it covers the dtypes
/// that have one. A dtype without an alias — a fixed-point type, a codebook, a
/// struct — has to be spelled structurally, and §04.3.6 requires readers to
/// accept the expanded form anyway, so the JSON door is the same door and not a
/// second one.
fn dtype_from_c(spec: &str) -> R<DType> {
    let spec = spec.trim();
    if spec.starts_with('{') {
        let j = json::parse(spec.as_bytes())
            .map_err(|e| usage(format!("dtype: not valid JSON: {e}")))?;
        return DType::from_value(&from_json(&j)).map_err(|e| usage(format!("dtype: {e}")));
    }
    DType::from_alias(spec).ok_or_else(|| {
        usage(format!(
            "dtype: `{spec}` is not a §04.3.6 alias; pass the structural \
             descriptor of §04.3 as a JSON object instead"
        ))
    })
}

fn builder_mut<'a>(p: *mut omni_builder) -> R<&'a omni_builder> {
    if p.is_null() {
        return Err(usage("builder is null"));
    }
    // SAFETY: the contract.
    Ok(unsafe { &*p })
}

/// Starts a container for a model named `name` (§06.2's `meta.name`).
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_new(name: *const c_char, o: *mut *mut omni_builder) -> c_int {
    guard(|| {
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        let h = Box::new(omni_builder {
            b: RefCell::new(omni_core::ModelBuilder::new(name)),
            codec: RefCell::new(omni_core::codec::Codec::Raw),
            log2_align: std::cell::Cell::new(12),
            bytes: RefCell::new(None),
        });
        // SAFETY: the contract.
        unsafe { out(o, "out", Box::into_raw(h)) }
    })
}

/// Releases a builder. Freeing `NULL` does nothing.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_free(b: *mut omni_builder) {
    if !b.is_null() {
        // SAFETY: the contract: a live handle from `omni_builder_new`.
        drop(unsafe { Box::from_raw(b) });
    }
}

/// Sets the container's digest algorithm: `"blake3-256"` or `"sha2-256"`
/// (§03.5.1). Object identities depend on it, so it must be chosen before
/// anything is written and cannot be changed afterwards.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_hash(b: *mut omni_builder, algo: *const c_char) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let algo = unsafe { as_str(algo, "algo")? };
        let a = match algo {
            "blake3-256" | "blake3" | "b3" => omni_core::HashAlgo::Blake3_256,
            "sha2-256" | "sha256" => omni_core::HashAlgo::Sha256,
            other => {
                return Err(usage(format!(
                    "hash: `{other}` is not one of §03.5.1's two mandatory algorithms"
                )))
            }
        };
        h.b.borrow_mut().hash = a;
        Ok(())
    })
}

/// Sets `meta.license` to an SPDX identifier.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_license(
    b: *mut omni_builder,
    spdx: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let spdx = unsafe { as_str(spdx, "spdx")? };
        h.b.borrow_mut().license_spdx = Some(spdx.to_string());
        Ok(())
    })
}

/// Sets `meta.arch.family` and, from a JSON object, `meta.arch.params`
/// (§06.2). `params_json` may be null.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_arch(
    b: *mut omni_builder,
    family: *const c_char,
    params_json: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let family = unsafe { as_str(family, "family")? };
        let mut params: Vec<(String, Value)> = Vec::new();
        if !params_json.is_null() {
            // SAFETY: the contract.
            let text = unsafe { as_str(params_json, "params_json")? };
            let j = json::parse(text.as_bytes())
                .map_err(|e| usage(format!("params_json: not valid JSON: {e}")))?;
            match j {
                json::Value::Object(map) => {
                    for (k, v) in map {
                        params.push((k, from_json(&v)));
                    }
                }
                _ => return Err(usage("params_json: expected a JSON object")),
            }
        }
        let mut bb = h.b.borrow_mut();
        bb.arch_family = Some(family.to_string());
        bb.arch_params = params;
        Ok(())
    })
}

/// Adds one key to `meta` from a JSON value — the escape hatch for the §06
/// fields this ABI has no dedicated setter for.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_add_meta(
    b: *mut omni_builder,
    key: *const c_char,
    value_json: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let key = unsafe { as_str(key, "key")? };
        // SAFETY: the contract.
        let text = unsafe { as_str(value_json, "value_json")? };
        let j = json::parse(text.as_bytes())
            .map_err(|e| usage(format!("value_json: not valid JSON: {e}")))?;
        h.b.borrow_mut()
            .extra
            .push((key.to_string(), from_json(&j)));
        Ok(())
    })
}

/// Sets the chunk size in bytes (§03.6's `fixed` chunker). Deduplication and
/// range reads happen at this granularity, so it is the caller's decision.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_chunk_size(b: *mut omni_builder, bytes: u64) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        if bytes == 0 {
            return Err(usage("chunk size must not be zero"));
        }
        h.b.borrow_mut().chunk_size = bytes as usize;
        Ok(())
    })
}

/// Sets the storage codec for data objects, as `"raw"`, `"zstd:9"`,
/// `"bitshuffle+zstd:9:2"`, `"deflate:6"` or `"lz4:9"` (§03.7.1).
///
/// A codec in the registry this build cannot produce is refused by name rather
/// than silently downgraded to `raw`, because a caller that asked for
/// compression and got none should be told.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_codec(
    b: *mut omni_builder,
    codec: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let spec = unsafe { as_str(codec, "codec")? };
        let mut parts = spec.split(':');
        let id = parts.next().unwrap_or("");
        let level: u64 = match parts.next() {
            None => 3,
            Some(l) => l
                .parse()
                .map_err(|_| usage(format!("codec: `{l}` is not a level")))?,
        };
        let elem: u64 = match parts.next() {
            None => 2,
            Some(e) => e
                .parse()
                .map_err(|_| usage(format!("codec: `{e}` is not an element width")))?,
        };
        let c = omni_core::codec::Codec::from_value(&Value::map(vec![
            ("id", Value::text(id)),
            ("level", Value::U(level)),
            ("elem_size", Value::U(elem)),
        ]));
        if let omni_core::codec::Codec::Unsupported(name) = c {
            return Err(if name == "unknown" {
                usage(format!("codec: `{id}` is not in the §03.7.1 registry"))
            } else {
                Fail::new(
                    OMNI_INDETERMINATE,
                    format!("codec: `{name}` is registered but not implemented in this build"),
                )
            });
        }
        *h.codec.borrow_mut() = c;
        Ok(())
    })
}

/// Sets the page alignment as a power of two, 6 to 30 (§02.4). The default of
/// 12 is a 4 KiB page; 16 suits a device that reads in 64 KiB units.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_alignment(b: *mut omni_builder, log2: u32) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        if !(6..=30).contains(&log2) {
            return Err(usage(format!(
                "alignment: log2 {log2} is outside §02.4's 6..=30"
            )));
        }
        h.log2_align.set(log2 as u8);
        Ok(())
    })
}

/// Adds a tensor from bytes laid out exactly as §04.3.5 says: dense row-major,
/// little-endian, the dtype's own packing for sub-byte types.
///
/// `dtype` is a §04.3.6 alias (`"bf16"`, `"i4"`, `"f32"`) or the structural
/// descriptor of §04.3 as a JSON object. `data` is copied, so the caller may
/// free it on return. The declared shape and dtype must account for exactly
/// `len` bytes — R-T02 is checked here rather than at pack time, so a wrong
/// length is a usage error at the call that made it and not a failed write
/// afterwards.
///
/// # Safety
/// See the module-level contract; `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_add_tensor(
    b: *mut omni_builder,
    name: *const c_char,
    dtype: *const c_char,
    shape: *const u64,
    ndim: u32,
    data: *const c_void,
    len: usize,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        // SAFETY: the contract.
        let dtype = dtype_from_c(unsafe { as_str(dtype, "dtype")? })?;
        if shape.is_null() && ndim > 0 {
            return Err(usage("shape is null"));
        }
        if data.is_null() && len > 0 {
            return Err(usage("data is null"));
        }
        // SAFETY: the contract requires `ndim` readable extents.
        let dims: Vec<u64> = unsafe { std::slice::from_raw_parts(shape, ndim as usize) }.to_vec();
        // SAFETY: the contract requires `len` readable bytes.
        let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(data as *const u8, len) }.to_vec();
        add_tensor(h, name, dtype, dims, bytes, None)
    })
}

/// The shared tail of the two tensor entry points: check R-T02 before storing,
/// so a wrong byte count is reported by the call that made it.
fn add_tensor(
    h: &omni_builder,
    name: &str,
    dtype: DType,
    shape: Vec<u64>,
    data: Vec<u8>,
    layout: Option<Layout>,
) -> R<()> {
    if name.is_empty() {
        return Err(usage("tensor name is empty"));
    }
    {
        let bb = h.b.borrow();
        if bb.tensors.iter().any(|t| t.name == name) {
            return Err(usage(format!("tensor `{name}` was already added")));
        }
    }
    let numel: u64 = shape.iter().product();
    let lay = layout.clone().unwrap_or_default();
    let expected = lay
        .stored_bytes(&shape, &dtype)
        .unwrap_or_else(|| dtype.packed_bytes(numel));
    if expected != data.len() as u64 {
        return Err(Fail::new(
            OMNI_EINVALID,
            format!(
                "R-T02: tensor `{name}` of {} elements at `{}` needs {expected} bytes, got {}",
                numel,
                dtype.label(),
                data.len()
            ),
        ));
    }
    h.b.borrow_mut().tensors.push(omni_core::TensorSpec {
        name: name.to_string(),
        shape,
        dtype,
        axes: None,
        semantic: String::new(),
        data,
        layout,
    });
    Ok(())
}

/// Names a tensor's axes (§04.2). Comma-separated, one name per dimension.
///
/// Axis names are what make an adapter's rank requirement checkable (§08.2), so
/// they are worth stating — and stating them wrongly is worse than leaving them
/// out, which is why the count is checked against the tensor's rank.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_tensor_axes(
    b: *mut omni_builder,
    name: *const c_char,
    axes_csv: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        // SAFETY: the contract.
        let csv = unsafe { as_str(axes_csv, "axes_csv")? };
        let axes: Vec<String> = csv.split(',').map(|s| s.trim().to_string()).collect();
        let mut bb = h.b.borrow_mut();
        let t = bb
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| usage(format!("no tensor named `{name}` has been added")))?;
        if axes.len() != t.shape.len() {
            return Err(usage(format!(
                "tensor `{name}` has rank {} and {} axis names were given",
                t.shape.len(),
                axes.len()
            )));
        }
        t.axes = Some(axes);
        Ok(())
    })
}

/// Sets a tensor's §04.2 semantic role: `"weight"`, `"bias"`, `"embedding"`, …
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_set_tensor_semantic(
    b: *mut omni_builder,
    name: *const c_char,
    semantic: *const c_char,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        // SAFETY: the contract.
        let sem = unsafe { as_str(semantic, "semantic")? };
        let mut bb = h.b.borrow_mut();
        let t = bb
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| usage(format!("no tensor named `{name}` has been added")))?;
        t.semantic = sem.to_string();
        Ok(())
    })
}

/// Adds a tensor from a DLPack `DLTensor`, which is how a PyTorch, JAX, NumPy,
/// CuPy or MLX array gets into a container without the caller flattening it
/// first.
///
/// This is the inverse of [`omni_tensor_dlpack`] and it refuses the same three
/// things, for the same reasons: a device that is not the CPU (there are no
/// bytes here to read), a dtype DLPack spells and §04.3 does not (`lanes > 1`),
/// and a strided array that is not dense row-major — because copying it as if
/// it were dense would store a different tensor with the right length. The
/// caller keeps ownership of the array; the bytes are copied.
///
/// # Safety
/// See the module-level contract; `t` must point to a valid `DLTensor` whose
/// `data`, `shape` and `strides` are readable for its `ndim`.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_add_dlpack(
    b: *mut omni_builder,
    name: *const c_char,
    t: *const DLTensor,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let name = unsafe { as_str(name, "name")? };
        if t.is_null() {
            return Err(usage("dltensor is null"));
        }
        // SAFETY: the contract.
        let dl = unsafe { &*t };
        if dl.device.device_type != OMNI_DLPACK_CPU {
            return Err(usage(format!(
                "DLPack device type {} is not CPU; there are no bytes here to read",
                dl.device.device_type
            )));
        }
        if dl.ndim < 0 {
            return Err(usage("DLPack ndim is negative"));
        }
        let ndim = dl.ndim as usize;
        if dl.shape.is_null() && ndim > 0 {
            return Err(usage("DLPack shape is null"));
        }
        // SAFETY: the contract.
        let dims_i64 = unsafe { std::slice::from_raw_parts(dl.shape, ndim) };
        if dims_i64.iter().any(|d| *d < 0) {
            return Err(usage("DLPack shape has a negative extent"));
        }
        let dims: Vec<u64> = dims_i64.iter().map(|d| *d as u64).collect();

        // A null `strides` means dense row-major, which is what §04.3.5 wants.
        // A non-null one has to *be* dense row-major, checked rather than
        // assumed: an array that is a transposed view has the same bytes in a
        // different order, and storing it as dense would be a different tensor.
        if !dl.strides.is_null() {
            // SAFETY: the contract.
            let strides = unsafe { std::slice::from_raw_parts(dl.strides, ndim) };
            let mut want = 1i64;
            for i in (0..ndim).rev() {
                if dims[i] > 1 && strides[i] != want {
                    return Err(Fail::new(
                        OMNI_EUSAGE,
                        format!(
                            "DLPack strides {strides:?} are not dense row-major for shape \
                             {dims:?}; make the array contiguous first"
                        ),
                    ));
                }
                want *= dims[i].max(1) as i64;
            }
        }

        let dtype = dl_dtype_in(&dl.dtype)?;
        let numel: u64 = dims.iter().product();
        let len = dtype.packed_bytes(numel) as usize;
        if dl.data.is_null() && len > 0 {
            return Err(usage("DLPack data is null"));
        }
        // SAFETY: the contract: `data + byte_offset` is readable for the
        // array's extent, which for a dense array is `len` bytes.
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts((dl.data as *const u8).add(dl.byte_offset as usize), len)
        }
        .to_vec();
        add_tensor(h, name, dtype, dims, bytes, None)
    })
}

/// The inverse of [`dl_dtype`]: DLPack's encoding onto a §04.3 dtype.
///
/// DLPack's `bits` and `lanes` describe whole-byte lanes, so the mapping is
/// narrow on purpose. A `lanes > 1` vector type is refused rather than
/// flattened, because §04.3 would spell it as a different shape and silently
/// changing a caller's rank is not a conversion.
fn dl_dtype_in(d: &DLDataType) -> R<DType> {
    if d.lanes != 1 {
        return Err(usage(format!(
            "DLPack lanes {} has no §04.3 dtype; a vector lane is a shape there, not a type",
            d.lanes
        )));
    }
    let bits = d.bits as u16;
    Ok(match (d.code, bits) {
        (0, 8) | (0, 16) | (0, 32) | (0, 64) => DType::Int {
            w: bits,
            signed: true,
        },
        (1, 8) | (1, 16) | (1, 32) | (1, 64) => DType::Int {
            w: bits,
            signed: false,
        },
        (2, 16) => DType::F16,
        (2, 32) => DType::F32,
        (2, 64) => DType::F64,
        (4, 16) => DType::BF16,
        (6, 8) => DType::Bool,
        (code, bits) => {
            return Err(usage(format!(
                "DLPack dtype code {code} at {bits} bits has no §04.3 spelling here"
            )))
        }
    })
}

/// Copies the digest of the root manifest the current calls would produce.
///
/// It is available before anything is written because the identity is a
/// function of the content and nothing else (§01.2): building the graph twice
/// gives the same digest, and packing it decides only where the bytes sit.
///
/// # Safety
/// See the module-level contract; `d32` must point to 32 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_root_digest(b: *mut omni_builder, d32: *mut u8) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        if d32.is_null() {
            return Err(usage("d32 is null"));
        }
        let (_, root) = h.b.borrow().build();
        // SAFETY: the contract requires 32 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(root.as_ptr(), d32, 32) };
        Ok(())
    })
}

/// How many tensors have been added.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_tensor_count(b: *mut omni_builder) -> usize {
    match builder_mut(b) {
        Ok(h) => h.b.borrow().tensors.len(),
        Err(_) => 0,
    }
}

fn pack_options(h: &omni_builder) -> omni_core::PackOptions {
    omni_core::PackOptions {
        log2_align: h.log2_align.get(),
        hash: h.b.borrow().hash,
        codec: h.codec.borrow().clone(),
        ..Default::default()
    }
}

/// Writes the container to `path`.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_write(b: *mut omni_builder, path: *const c_char) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        // SAFETY: the contract.
        let path = unsafe { as_str(path, "path")? };
        let (objects, root) = h.b.borrow().build();
        let bytes = omni_core::pack(&objects, &root, &pack_options(h)).map_err(from_container)?;
        std::fs::write(path, &bytes)
            .map_err(|e| Fail::new(OMNI_EINVALID, format!("{path}: {e}")))?;
        Ok(())
    })
}

/// Writes the container into memory. The bytes stay valid until the next call
/// to this function on the same builder, or until the builder is freed.
///
/// # Safety
/// See the module-level contract.
#[no_mangle]
pub unsafe extern "C" fn omni_builder_write_bytes(
    b: *mut omni_builder,
    ptr: *mut *const u8,
    len: *mut usize,
) -> c_int {
    guard(|| {
        let h = builder_mut(b)?;
        let (objects, root) = h.b.borrow().build();
        let bytes = omni_core::pack(&objects, &root, &pack_options(h)).map_err(from_container)?;
        let arc = Arc::new(bytes);
        *h.bytes.borrow_mut() = Some(arc.clone());
        // SAFETY: the contract.
        unsafe { out(ptr, "ptr", arc.as_ptr())? };
        if !len.is_null() {
            // SAFETY: the contract.
            unsafe { out(len, "len", arc.len())? };
        }
        Ok(())
    })
}

// ------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the ABI the way C does: raw pointers, out-parameters, status
    /// codes. Calling it from Rust is the same code path a `dlopen`ing caller
    /// takes, minus the linker.
    fn example() -> Vec<u8> {
        let mut b = omni_core::ModelBuilder::new("acme/ffi-toy");
        let data: Vec<u8> = (0..64u16).flat_map(|i| (i * 3).to_le_bytes()).collect();
        b = b.tensor(omni_core::TensorSpec {
            name: "w".into(),
            shape: vec![8, 8],
            dtype: DType::BF16,
            axes: Some(vec!["out".into(), "in".into()]),
            semantic: "weight".into(),
            data,
            layout: None,
        });
        let (objects, root) = b.build();
        omni_core::pack(&objects, &root, &omni_core::PackOptions::default()).expect("packs")
    }

    fn open() -> *mut omni_store {
        let bytes = example();
        let mut s: *mut omni_store = std::ptr::null_mut();
        let rc = unsafe { omni_store_open_bytes(bytes.as_ptr(), bytes.len(), &mut s) };
        assert_eq!(rc, OMNI_OK, "{}", last());
        s
    }

    fn last() -> String {
        unsafe { CStr::from_ptr(omni_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    // ------------------------------------------------------------- writer --

    /// Builds a two-tensor model through the C entry points alone.
    fn build_through_c() -> *mut omni_builder {
        let mut b: *mut omni_builder = std::ptr::null_mut();
        let name = cstring("test/from-c");
        assert_eq!(unsafe { omni_builder_new(name.as_ptr(), &mut b) }, OMNI_OK);
        let spdx = cstring("Apache-2.0");
        assert_eq!(
            unsafe { omni_builder_set_license(b, spdx.as_ptr()) },
            OMNI_OK
        );
        let fam = cstring("transformer.decoder");
        let params = cstring("{\"n_layers\": 2, \"d_model\": 8}");
        assert_eq!(
            unsafe { omni_builder_set_arch(b, fam.as_ptr(), params.as_ptr()) },
            OMNI_OK,
            "{}",
            last()
        );
        let tname = cstring("w");
        let dtype = cstring("bf16");
        let shape = [4u64, 8];
        let data = [0x3fu8; 4 * 8 * 2];
        assert_eq!(
            unsafe {
                omni_builder_add_tensor(
                    b,
                    tname.as_ptr(),
                    dtype.as_ptr(),
                    shape.as_ptr(),
                    2,
                    data.as_ptr() as *const c_void,
                    data.len(),
                )
            },
            OMNI_OK,
            "{}",
            last()
        );
        let axes = cstring("out_features,in_features");
        assert_eq!(
            unsafe { omni_builder_set_tensor_axes(b, tname.as_ptr(), axes.as_ptr()) },
            OMNI_OK,
            "{}",
            last()
        );
        let sem = cstring("weight");
        assert_eq!(
            unsafe { omni_builder_set_tensor_semantic(b, tname.as_ptr(), sem.as_ptr()) },
            OMNI_OK
        );
        let bname = cstring("b");
        let f32ty = cstring("f32");
        let bshape = [4u64];
        let bdata = [0u8; 16];
        assert_eq!(
            unsafe {
                omni_builder_add_tensor(
                    b,
                    bname.as_ptr(),
                    f32ty.as_ptr(),
                    bshape.as_ptr(),
                    1,
                    bdata.as_ptr() as *const c_void,
                    bdata.len(),
                )
            },
            OMNI_OK,
            "{}",
            last()
        );
        b
    }

    #[test]
    fn a_container_built_from_c_opens_verifies_and_reads_back() {
        let b = build_through_c();
        assert_eq!(unsafe { omni_builder_tensor_count(b) }, 2);
        let mut ptr: *const u8 = std::ptr::null();
        let mut len: usize = 0;
        assert_eq!(
            unsafe { omni_builder_write_bytes(b, &mut ptr, &mut len) },
            OMNI_OK,
            "{}",
            last()
        );
        assert!(len > 0);
        // The digest the builder predicted must be the root of what it wrote:
        // identity is a function of content, not of when it was computed.
        let mut want = [0u8; 32];
        assert_eq!(
            unsafe { omni_builder_root_digest(b, want.as_mut_ptr()) },
            OMNI_OK
        );

        let mut s: *mut omni_store = std::ptr::null_mut();
        assert_eq!(
            unsafe { omni_store_open_bytes(ptr, len, &mut s) },
            OMNI_OK,
            "{}",
            last()
        );
        let mut got = [0u8; 32];
        assert_eq!(
            unsafe { omni_store_root_digest(s, got.as_mut_ptr()) },
            OMNI_OK
        );
        assert_eq!(got, want);
        assert_eq!(
            unsafe { omni_store_verify(s, std::ptr::null_mut()) },
            OMNI_OK,
            "{}",
            last()
        );

        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        assert_eq!(unsafe { omni_model_tensor_count(m) }, 2);
        let n = cstring("w");
        let mut t: *mut omni_tensor = std::ptr::null_mut();
        assert_eq!(
            unsafe { omni_model_tensor(m, n.as_ptr(), &mut t) },
            OMNI_OK,
            "{}",
            last()
        );
        let mut info = std::mem::MaybeUninit::<omni_tensor_info>::zeroed();
        assert_eq!(
            unsafe { omni_tensor_get_info(t, info.as_mut_ptr()) },
            OMNI_OK
        );
        let info = unsafe { info.assume_init() };
        assert_eq!(
            unsafe { CStr::from_ptr(info.dtype) }.to_str().unwrap(),
            "bf16"
        );
        assert_eq!(info.numel, 32);
        unsafe { omni_tensor_release(t) };
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
        unsafe { omni_builder_free(b) };
    }

    #[test]
    fn the_same_calls_produce_the_same_bytes() {
        // §01.10 writer rule W1. Two builders, same calls, byte-identical
        // output — the property that makes a container reproducible at all.
        let b1 = build_through_c();
        let b2 = build_through_c();
        let (mut p1, mut l1) = (std::ptr::null(), 0usize);
        let (mut p2, mut l2) = (std::ptr::null(), 0usize);
        assert_eq!(
            unsafe { omni_builder_write_bytes(b1, &mut p1, &mut l1) },
            OMNI_OK
        );
        assert_eq!(
            unsafe { omni_builder_write_bytes(b2, &mut p2, &mut l2) },
            OMNI_OK
        );
        let a = unsafe { std::slice::from_raw_parts(p1, l1) };
        let c = unsafe { std::slice::from_raw_parts(p2, l2) };
        assert_eq!(a, c);
        unsafe { omni_builder_free(b1) };
        unsafe { omni_builder_free(b2) };
    }

    #[test]
    fn a_wrong_byte_count_is_caught_by_the_call_that_made_it() {
        let mut b: *mut omni_builder = std::ptr::null_mut();
        let name = cstring("test/short");
        assert_eq!(unsafe { omni_builder_new(name.as_ptr(), &mut b) }, OMNI_OK);
        let tname = cstring("w");
        let dtype = cstring("f32");
        let shape = [4u64, 4];
        let data = [0u8; 8]; // 64 bytes are needed.
        let rc = unsafe {
            omni_builder_add_tensor(
                b,
                tname.as_ptr(),
                dtype.as_ptr(),
                shape.as_ptr(),
                2,
                data.as_ptr() as *const c_void,
                data.len(),
            )
        };
        assert_eq!(rc, OMNI_EINVALID);
        assert!(last().contains("R-T02"), "{}", last());
        // Nothing was added, so the builder is still usable.
        assert_eq!(unsafe { omni_builder_tensor_count(b) }, 0);
        unsafe { omni_builder_free(b) };
    }

    #[test]
    fn an_unspellable_dtype_and_an_unimplemented_codec_are_told_apart() {
        let mut b: *mut omni_builder = std::ptr::null_mut();
        let name = cstring("test/refusals");
        assert_eq!(unsafe { omni_builder_new(name.as_ptr(), &mut b) }, OMNI_OK);
        // A dtype with no alias is a usage error naming the way to spell it.
        let tname = cstring("w");
        let bogus = cstring("float37");
        let shape = [1u64];
        let data = [0u8; 8];
        let rc = unsafe {
            omni_builder_add_tensor(
                b,
                tname.as_ptr(),
                bogus.as_ptr(),
                shape.as_ptr(),
                1,
                data.as_ptr() as *const c_void,
                data.len(),
            )
        };
        assert_eq!(rc, OMNI_EUSAGE);
        assert!(last().contains("§04.3"), "{}", last());
        // A registered codec this build cannot produce is indeterminate, not
        // invalid, and never a silent downgrade to raw.
        let brotli = cstring("brotli");
        assert_eq!(
            unsafe { omni_builder_set_codec(b, brotli.as_ptr()) },
            OMNI_INDETERMINATE
        );
        let nonsense = cstring("gzip9000");
        assert_eq!(
            unsafe { omni_builder_set_codec(b, nonsense.as_ptr()) },
            OMNI_EUSAGE
        );
        // And one it can.
        let zstd = cstring("zstd:9");
        assert_eq!(unsafe { omni_builder_set_codec(b, zstd.as_ptr()) }, OMNI_OK);
        unsafe { omni_builder_free(b) };
    }

    #[test]
    fn a_dlpack_array_becomes_a_tensor_and_a_transposed_view_is_refused() {
        let mut b: *mut omni_builder = std::ptr::null_mut();
        let name = cstring("test/dlpack-in");
        assert_eq!(unsafe { omni_builder_new(name.as_ptr(), &mut b) }, OMNI_OK);
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut shape = [3i64, 4i64];
        let dl = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: OMNI_DLPACK_CPU,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let tn = cstring("x");
        assert_eq!(
            unsafe { omni_builder_add_dlpack(b, tn.as_ptr(), &dl) },
            OMNI_OK,
            "{}",
            last()
        );

        // The same buffer described as a transposed view: the bytes are the
        // same and the tensor is not, so it is refused rather than copied.
        let mut strides = [1i64, 3i64];
        let mut view = dl;
        view.strides = strides.as_mut_ptr();
        let tn2 = cstring("xT");
        assert_eq!(
            unsafe { omni_builder_add_dlpack(b, tn2.as_ptr(), &view) },
            OMNI_EUSAGE
        );
        assert!(last().contains("row-major"), "{}", last());

        // A lane count DLPack allows and §04.3 spells as a shape.
        let mut vec4 = view;
        vec4.strides = std::ptr::null_mut();
        vec4.dtype.lanes = 4;
        let tn3 = cstring("v");
        assert_eq!(
            unsafe { omni_builder_add_dlpack(b, tn3.as_ptr(), &vec4) },
            OMNI_EUSAGE
        );

        // What went in comes back out through the read path, values and all.
        let (mut p, mut l) = (std::ptr::null(), 0usize);
        assert_eq!(
            unsafe { omni_builder_write_bytes(b, &mut p, &mut l) },
            OMNI_OK
        );
        let mut s: *mut omni_store = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_open_bytes(p, l, &mut s) }, OMNI_OK);
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut t: *mut omni_tensor = std::ptr::null_mut();
        assert_eq!(
            unsafe { omni_model_tensor(m, tn.as_ptr(), &mut t) },
            OMNI_OK
        );
        let (mut vp, mut vl) = (std::ptr::null(), 0usize);
        assert_eq!(unsafe { omni_tensor_values(t, &mut vp, &mut vl) }, OMNI_OK);
        let values = unsafe { std::slice::from_raw_parts(vp, vl) };
        assert_eq!(values.len(), 12);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f64);
        }
        unsafe { omni_tensor_release(t) };
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
        unsafe { omni_builder_free(b) };
    }

    #[test]
    fn a_null_builder_handle_is_a_usage_error_and_freeing_null_does_nothing() {
        assert_eq!(
            unsafe { omni_builder_set_chunk_size(std::ptr::null_mut(), 4096) },
            OMNI_EUSAGE
        );
        assert_eq!(
            unsafe { omni_builder_tensor_count(std::ptr::null_mut()) },
            0
        );
        unsafe { omni_builder_free(std::ptr::null_mut()) };
    }

    #[test]
    fn the_abi_version_separates_breaking_from_additive() {
        assert_eq!(omni_abi_version() >> 16, 1);
    }

    #[test]
    fn a_null_handle_is_a_usage_error_and_not_a_crash() {
        assert_eq!(
            unsafe { omni_store_root(std::ptr::null_mut(), std::ptr::null_mut()) },
            OMNI_EUSAGE
        );
        assert!(last().contains("null"));
        // Freeing null is defined and does nothing.
        unsafe { omni_store_close(std::ptr::null_mut()) };
        unsafe { omni_model_free(std::ptr::null_mut()) };
        unsafe { omni_tensor_release(std::ptr::null_mut()) };
        unsafe { omni_plan_free(std::ptr::null_mut()) };
    }

    #[test]
    fn bytes_that_are_not_a_container_are_invalid_not_a_panic() {
        let junk = vec![0u8; 512];
        let mut s: *mut omni_store = std::ptr::null_mut();
        let rc = unsafe { omni_store_open_bytes(junk.as_ptr(), junk.len(), &mut s) };
        assert_eq!(rc, OMNI_EINVALID, "{}", last());
        assert!(s.is_null());
    }

    #[test]
    fn a_container_opens_verifies_and_names_its_tensors() {
        let s = open();
        let mut rep = omni_verify_report {
            segments: 0,
            objects_verified: 0,
            bytes_verified: 0,
            reachable: 0,
            dangling: 0,
            mistyped: 0,
            padding_ok: 0,
            alignment_ok: 0,
        };
        assert_eq!(
            unsafe { omni_store_verify(s, &mut rep) },
            OMNI_OK,
            "{}",
            last()
        );
        assert!(rep.objects_verified > 0);
        assert_eq!(rep.dangling, 0);
        assert_eq!(rep.padding_ok, 1);

        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK, "{}", last());
        assert_eq!(unsafe { omni_model_tensor_count(m) }, 1);
        let mut name: *const c_char = std::ptr::null();
        assert_eq!(unsafe { omni_model_tensor_name(m, 0, &mut name) }, OMNI_OK);
        assert_eq!(
            unsafe { CStr::from_ptr(name) }.to_str().expect("utf-8"),
            "w"
        );
        assert_eq!(
            unsafe { omni_model_tensor_name(m, 9, &mut name) },
            OMNI_EUSAGE
        );
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
    }

    #[test]
    fn a_tensor_reports_its_type_and_hands_back_mapped_bytes() {
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut t: *mut omni_tensor = std::ptr::null_mut();
        let n = cstring("w");
        assert_eq!(
            unsafe { omni_model_tensor(m, n.as_ptr(), &mut t) },
            OMNI_OK,
            "{}",
            last()
        );

        let mut info = omni_tensor_info {
            dtype: std::ptr::null(),
            dtype_bits: 0,
            layout: std::ptr::null(),
            value_op: std::ptr::null(),
            ndim: 0,
            shape: std::ptr::null(),
            numel: 0,
        };
        assert_eq!(
            unsafe { omni_tensor_get_info(t, &mut info) },
            OMNI_OK,
            "{}",
            last()
        );
        assert_eq!(
            unsafe { CStr::from_ptr(info.dtype) }
                .to_str()
                .expect("utf-8"),
            "bf16"
        );
        assert_eq!(info.dtype_bits, 16);
        assert_eq!(info.ndim, 2);
        assert_eq!(info.numel, 64);
        let shape = unsafe { std::slice::from_raw_parts(info.shape, 2) };
        assert_eq!(shape, &[8, 8]);

        let mut p: *const c_void = std::ptr::null();
        let mut len = 0usize;
        assert_eq!(
            unsafe { omni_tensor_bytes(t, &mut p, &mut len) },
            OMNI_OK,
            "{}",
            last()
        );
        assert_eq!(len, 128);
        // 8x8 bf16 stored raw in one blob is the mappable case, so no copy was
        // made and the pointer is inside the container itself.
        assert_eq!(unsafe { omni_tensor_mapped(t) }, 1);

        // Releasing the store first must not invalidate the tensor: the handle
        // holds its own share of the container.
        unsafe { omni_store_close(s) };
        unsafe { omni_model_free(m) };
        let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        assert_eq!(&bytes[..2], &0u16.to_le_bytes());
        assert_eq!(&bytes[2..4], &3u16.to_le_bytes());
        unsafe { omni_tensor_release(t) };
    }

    #[test]
    fn values_come_back_decoded_through_the_dtype() {
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut t: *mut omni_tensor = std::ptr::null_mut();
        let n = cstring("w");
        assert_eq!(unsafe { omni_model_tensor(m, n.as_ptr(), &mut t) }, OMNI_OK);
        let mut p: *const f64 = std::ptr::null();
        let mut len = 0usize;
        assert_eq!(
            unsafe { omni_tensor_values(t, &mut p, &mut len) },
            OMNI_OK,
            "{}",
            last()
        );
        assert_eq!(len, 64);
        let v = unsafe { std::slice::from_raw_parts(p, len) };
        // Element i is the bf16 whose bits are 3i, decoded as a subnormal.
        assert_eq!(v[0], 0.0);
        assert!(v[1] > 0.0 && v[1] < 1e-38);
        unsafe { omni_tensor_release(t) };
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
    }

    #[test]
    fn the_manifest_comes_out_as_json() {
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut p: *const c_char = std::ptr::null();
        let mut len = 0usize;
        assert_eq!(
            unsafe { omni_model_meta_json(m, &mut p, &mut len) },
            OMNI_OK
        );
        let j = unsafe { CStr::from_ptr(p) }.to_str().expect("utf-8");
        assert_eq!(j.len(), len);
        let parsed = json::parse(j.as_bytes()).expect("the ABI emits parseable JSON");
        assert_eq!(
            parsed.get("t").and_then(|x| x.as_str()),
            Some("omni.core/manifest")
        );
        // A ref is `[otype, digest]` (§01.3). Digests are bytes in CBOR and hex
        // in JSON, and the projection says so by producing text rather than an
        // array of 32 numbers, which is what a naive projection would emit.
        let asset = parsed
            .get("assets")
            .and_then(|a| a.get("model"))
            .and_then(|r| r.as_array())
            .expect("the manifest names its model asset");
        assert_eq!(asset[0].as_u64(), Some(u64::from(otype::MODEL)));
        assert_eq!(asset[1].as_str().map(str::len), Some(64));
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
    }

    #[test]
    fn the_c0_baseline_refuses_this_model_and_says_which_feature() {
        // The default capability set is C0, the floor every reader meets, and
        // it does *not* include the expression feature this model requires. So
        // the honest answer is infeasible with a named reason, not a plan that
        // quietly drops a tensor.
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut p: *mut omni_plan = std::ptr::null_mut();
        let rc = unsafe { omni_model_resolve(m, std::ptr::null(), OMNI_OBJ_MIN_MEMORY, &mut p) };
        assert_eq!(rc, OMNI_EINFEASIBLE, "{}", last());
        // The plan still comes back, because "why not" is the useful part.
        assert!(!p.is_null());
        assert_eq!(unsafe { omni_plan_feasible(p) }, 0);
        let mut j: *const c_char = std::ptr::null();
        assert_eq!(
            unsafe { omni_plan_json(p, &mut j, std::ptr::null_mut()) },
            OMNI_OK
        );
        let text = unsafe { CStr::from_ptr(j) }.to_string_lossy().into_owned();
        assert!(text.contains("omni.tensor/expr.1"), "{text}");
        unsafe { omni_plan_free(p) };
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
    }

    #[test]
    fn a_plan_resolves_against_capabilities_handed_in_as_json() {
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let caps = cstring(
            r#"{"t":"omni.rt/capabilities","v":1,
                "runtime":{"name":"c-caller","version":"0"},
                "profiles":["C0","C1"],
                "dtypes":{"storage":["bf16","f32"],"compute":["f32"]},
                "layouts":["strided"],
                "features":["omni.core/1.0","omni.tensor/expr.1"],
                "policy":{"allow_lossy":false}}"#,
        );
        let mut p: *mut omni_plan = std::ptr::null_mut();
        let rc = unsafe { omni_model_resolve(m, caps.as_ptr(), OMNI_OBJ_MIN_MEMORY, &mut p) };
        assert_eq!(rc, OMNI_OK, "{}", last());
        assert_eq!(unsafe { omni_plan_feasible(p) }, 1);
        assert_eq!(unsafe { omni_plan_resident_bytes(p) }, 128);
        let mut j: *const c_char = std::ptr::null();
        assert_eq!(
            unsafe { omni_plan_json(p, &mut j, std::ptr::null_mut()) },
            OMNI_OK
        );
        let text = unsafe { CStr::from_ptr(j) }.to_string_lossy().into_owned();
        assert!(text.contains("min-memory"), "{text}");
        unsafe { omni_plan_free(p) };

        // Capabilities that are not JSON, and an objective that is not one.
        let junk = cstring("{not json");
        let mut q: *mut omni_plan = std::ptr::null_mut();
        assert_eq!(
            unsafe { omni_model_resolve(m, junk.as_ptr(), OMNI_OBJ_BALANCED, &mut q) },
            OMNI_EUSAGE
        );
        assert!(last().contains("capabilities"), "{}", last());
        assert_eq!(
            unsafe { omni_model_resolve(m, std::ptr::null(), 99, &mut q) },
            OMNI_EUSAGE
        );
        assert!(last().contains("OMNI_OBJ"));
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
    }

    #[test]
    fn dlpack_describes_a_bf16_tensor_and_frees_itself() {
        let s = open();
        let mut m: *mut omni_model = std::ptr::null_mut();
        assert_eq!(unsafe { omni_store_root(s, &mut m) }, OMNI_OK);
        let mut t: *mut omni_tensor = std::ptr::null_mut();
        let n = cstring("w");
        assert_eq!(unsafe { omni_model_tensor(m, n.as_ptr(), &mut t) }, OMNI_OK);
        let mut dl: *mut DLManagedTensor = std::ptr::null_mut();
        assert_eq!(
            unsafe { omni_tensor_dlpack(t, &mut dl) },
            OMNI_OK,
            "{}",
            last()
        );
        let d = unsafe { &*dl };
        assert_eq!(d.dl_tensor.ndim, 2);
        // kDLBfloat, 16 bits, one lane.
        assert_eq!(d.dl_tensor.dtype.code, 4);
        assert_eq!(d.dl_tensor.dtype.bits, 16);
        assert_eq!(d.dl_tensor.dtype.lanes, 1);
        assert!(
            d.dl_tensor.strides.is_null(),
            "dense row-major means null strides"
        );
        let shape = unsafe { std::slice::from_raw_parts(d.dl_tensor.shape, 2) };
        assert_eq!(shape, &[8, 8]);

        // The consumer outliving every OMNI handle is the entire point of
        // §2.1's rule, so release them all before touching the data.
        unsafe { omni_tensor_release(t) };
        unsafe { omni_model_free(m) };
        unsafe { omni_store_close(s) };
        let bytes = unsafe { std::slice::from_raw_parts(d.dl_tensor.data as *const u8, 128) };
        assert_eq!(&bytes[2..4], &3u16.to_le_bytes());
        unsafe { (d.deleter.expect("DLPack requires a deleter"))(dl) };
    }

    #[test]
    fn a_dtype_dlpack_cannot_spell_is_refused_by_name() {
        // i4 is the case that matters: it is what every 4-bit quantized model
        // stores, and handing it over as uint8 would be wrong twice.
        let d = DType::Int { w: 4, signed: true };
        let err = dl_dtype(&d).expect_err("i4 has no DLPack spelling");
        assert!(err.contains("i4"), "{err}");
        assert!(dl_dtype(&DType::F32).is_ok());
        assert!(dl_dtype(&DType::Int {
            w: 32,
            signed: false
        })
        .is_ok());
    }

    #[test]
    fn status_codes_are_the_clis_exit_codes() {
        // docs/design/cli.md §10.3. If these drift, a C caller and the CLI
        // disagree about what happened.
        for (code, name) in [
            (OMNI_OK, "ok"),
            (OMNI_EINVALID, "invalid"),
            (OMNI_EUSAGE, "usage"),
            (OMNI_INDETERMINATE, "indeterminate"),
            (OMNI_EPOLICY, "policy"),
            (OMNI_EINCOMPLETE, "incomplete"),
            (OMNI_EINFEASIBLE, "infeasible"),
        ] {
            let s = unsafe { CStr::from_ptr(omni_status_name(code)) };
            assert_eq!(s.to_str().expect("utf-8"), name);
        }
        assert_eq!(OMNI_INDETERMINATE, 3);
    }
}
