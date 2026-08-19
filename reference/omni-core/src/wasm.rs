//! §11.6 — WebAssembly as portable semantics.
//!
//! §11 puts every domain-specific thing in OMNI behind a plugin, and §11.6 makes
//! the portable form of a plugin a WebAssembly module: "a frozen, formally
//! specified instruction set with a mechanized semantics — the strongest 50-year
//! portability guarantee available today". That is a claim about what a *host*
//! can do, and until now this implementation had none: an expression node with a
//! `plugin` op, a tokenizer of `kind: "plugin"`, a dialect's `shape_fn` — all
//! reported as unrunnable.
//!
//! This is the host, written to §11.6's restricted profile:
//!
//! | Constraint | Here |
//! |---|---|
//! | Imports | none, except `omni_plugin/1` — anything else is refused at load |
//! | Determinism | required: no clock, no randomness, no host state; NaN results are canonicalized so two runs cannot differ in a payload |
//! | Filesystem / network / clock | unavailable, because no import provides them |
//! | Fuel | metered per instruction; running out is a trap, not a hang |
//! | Memory | capped (default 256 MiB), and `memory.grow` fails rather than exceeding it |
//! | Forbidden proposals | threads, relaxed-SIMD, exceptions, GC — refused by opcode at *load* time, so a module cannot smuggle them past validation |
//!
//! What is implemented is the core instruction set a plugin compiled from C or
//! Rust actually uses: the full i32/i64/f32/f64 numeric set with conversions and
//! saturating truncation, sign extension, all memory loads and stores, globals,
//! structured control flow with `br_table`, `call` and `call_indirect`, `select`,
//! and the bulk-memory operations — plus **fixed-width SIMD**, the `v128` type
//! and the whole of §11.6's deterministic vector subset, which is what a plugin
//! compiled with `-msimd128` emits and what a hand-written kernel reaches for.
//!
//! Relaxed-SIMD lives in the same `0xfd` prefix and is *forbidden* rather than
//! merely absent, at load time, by opcode: its results are permitted to differ
//! between hosts, and a plugin whose output depends on the engine is the one
//! thing §11.6 rules out. The distinction matters enough to be a different error.
//!
//! The vector instructions are checked differentially against `wasmtime` rather
//! than against this host's own opinion — see `tests/wasm_simd_vectors.rs`. A
//! table of ~230 lane-wise operations is not something reading can verify: the
//! saturating forms, the two different NaN rules that separate `min` from
//! `pmin`, the rounding Q15 multiply and the narrowing saturations are each a
//! place where a plausible implementation is wrong and its own tests agree.
//!
//! Validation here is structural plus dynamic: the module's shape, its types,
//! its imports and its opcodes are checked when it is loaded, and everything
//! else — operand types, stack discipline, memory bounds — is checked as it
//! runs. A fully static validator is what a production engine wants for speed;
//! for a reference host the distinction that matters is that neither one can be
//! made to do something unsafe, and this one is written in Rust with
//! `forbid(unsafe_code)` over a `Vec<u8>` heap.

use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum Error {
    /// Not a WebAssembly module, or one whose framing is broken.
    Malformed(String),
    /// A feature §11.6 permits but this host does not implement. Indeterminate,
    /// not invalid: the plugin may be perfectly good.
    Unsupported(String),
    /// A feature §11.6 forbids.
    Forbidden(String),
    /// The module trapped: out of bounds, division by zero, an explicit
    /// `unreachable`, a failed indirect call.
    Trap(String),
    /// Out of fuel, or over the memory cap.
    Limit(String),
    /// The host ABI was used incorrectly by the module.
    Abi(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed wasm: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported wasm feature: {m}"),
            Error::Forbidden(m) => write!(f, "forbidden under §11.6: {m}"),
            Error::Trap(m) => write!(f, "trap: {m}"),
            Error::Limit(m) => write!(f, "limit: {m}"),
            Error::Abi(m) => write!(f, "plugin ABI: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

fn malformed<T>(m: impl Into<String>) -> Res<T> {
    Err(Error::Malformed(m.into()))
}

// --------------------------------------------------------------------- types --

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
    /// `v128`, the fixed-width SIMD vector §11.6 permits.
    V128,
    /// `funcref`, for tables. Represented as a function index.
    FuncRef,
}

impl ValType {
    fn parse(b: u8) -> Res<ValType> {
        Ok(match b {
            0x7f => ValType::I32,
            0x7e => ValType::I64,
            0x7d => ValType::F32,
            0x7c => ValType::F64,
            0x7b => ValType::V128,
            0x70 => ValType::FuncRef,
            0x6f => return Err(Error::Unsupported("externref".into())),
            other => return malformed(format!("value type {other:#04x}")),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    /// A 128-bit vector. Held as a `u128` whose least significant byte is lane
    /// 0, which is the order the vector has in memory — so a `v128.load` is a
    /// little-endian read and needs no lane shuffling.
    V128(u128),
    /// A function index, or `None` for a null reference.
    Ref(Option<u32>),
}

impl Value {
    fn ty(&self) -> ValType {
        match self {
            Value::I32(_) => ValType::I32,
            Value::I64(_) => ValType::I64,
            Value::F32(_) => ValType::F32,
            Value::F64(_) => ValType::F64,
            Value::V128(_) => ValType::V128,
            Value::Ref(_) => ValType::FuncRef,
        }
    }

    fn zero(t: ValType) -> Value {
        match t {
            ValType::I32 => Value::I32(0),
            ValType::I64 => Value::I64(0),
            ValType::F32 => Value::F32(0.0),
            ValType::F64 => Value::F64(0.0),
            ValType::V128 => Value::V128(0),
            ValType::FuncRef => Value::Ref(None),
        }
    }

    pub fn as_v128(&self) -> Res<u128> {
        match self {
            Value::V128(v) => Ok(*v),
            other => Err(Error::Trap(format!("expected v128, found {other:?}"))),
        }
    }

    pub fn as_i32(&self) -> Res<i32> {
        match self {
            Value::I32(n) => Ok(*n),
            other => Err(Error::Trap(format!("expected i32, found {other:?}"))),
        }
    }

    pub fn as_i64(&self) -> Res<i64> {
        match self {
            Value::I64(n) => Ok(*n),
            other => Err(Error::Trap(format!("expected i64, found {other:?}"))),
        }
    }

    pub fn as_f32(&self) -> Res<f32> {
        match self {
            Value::F32(n) => Ok(*n),
            other => Err(Error::Trap(format!("expected f32, found {other:?}"))),
        }
    }

    pub fn as_f64(&self) -> Res<f64> {
        match self {
            Value::F64(n) => Ok(*n),
            other => Err(Error::Trap(format!("expected f64, found {other:?}"))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

#[derive(Clone, Debug)]
struct Func {
    ty: u32,
    /// Declared local types, expanded from the compressed form.
    locals: Vec<ValType>,
    body: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
struct Global {
    ty: ValType,
    mutable: bool,
    init: Value,
}

/// A parsed, load-validated module.
pub struct Module {
    bytes: Vec<u8>,
    types: Vec<FuncType>,
    /// Host functions the module imports, in index order. Every one is from
    /// `omni_plugin/1`; anything else fails the load.
    imports: Vec<HostFn>,
    funcs: Vec<Func>,
    globals: Vec<Global>,
    exports: BTreeMap<String, Export>,
    memory: Option<(u32, Option<u32>)>,
    table: Vec<Option<u32>>,
    data: Vec<(Option<u32>, Vec<u8>)>,
    start: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Export {
    Func(u32),
    Memory,
    Global(u32),
    Table,
}

/// The host functions of `omni_plugin/1` (§11.6).
///
/// `alloc` and `dealloc` are the module's job — a host cannot sensibly allocate
/// inside someone else's linear memory — so they are *exports* the host calls,
/// and these three are what the module may *import*. Nothing here can observe a
/// clock, a random number or a file, which is what makes the profile
/// deterministic rather than merely sandboxed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostFn {
    /// `log(ptr: i32, len: i32)`
    Log,
    /// `abort(ptr: i32, len: i32)` — traps with the message.
    Abort,
    /// `read_object(digest_ptr: i32, out_ptr: i32, out_cap: i32) -> i32`
    /// Reads one of the refs the plugin was declared against; returns the
    /// object's length, or a negative code.
    ReadObject,
}

impl HostFn {
    fn from_name(name: &str) -> Option<HostFn> {
        match name {
            "log" => Some(HostFn::Log),
            "abort" => Some(HostFn::Abort),
            "read_object" => Some(HostFn::ReadObject),
            _ => None,
        }
    }

    fn signature(&self) -> FuncType {
        match self {
            HostFn::Log | HostFn::Abort => FuncType {
                params: vec![ValType::I32, ValType::I32],
                results: Vec::new(),
            },
            HostFn::ReadObject => FuncType {
                params: vec![ValType::I32, ValType::I32, ValType::I32],
                results: vec![ValType::I32],
            },
        }
    }
}

/// The one namespace §11.6 allows a plugin to import from.
pub const HOST_MODULE: &str = "omni_plugin/1";

const PAGE: usize = 65536;

// ------------------------------------------------------------------- parsing --

struct Reader<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Reader { d, at: 0 }
    }

    fn byte(&mut self) -> Res<u8> {
        let b = *self
            .d
            .get(self.at)
            .ok_or_else(|| Error::Malformed("unexpected end of module".into()))?;
        self.at += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self
            .d
            .get(self.at..self.at + n)
            .ok_or_else(|| Error::Malformed("unexpected end of module".into()))?;
        self.at += n;
        Ok(s)
    }

    /// LEB128, unsigned, bounded to 64 bits.
    fn u(&mut self) -> Res<u64> {
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.byte()?;
            if shift >= 64 {
                return malformed("overlong LEB128");
            }
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }

    fn u32(&mut self) -> Res<u32> {
        let v = self.u()?;
        u32::try_from(v).map_err(|_| Error::Malformed("index out of range".into()))
    }

    /// LEB128, signed.
    fn i(&mut self, bits: u32) -> Res<i64> {
        let mut v = 0i64;
        let mut shift = 0u32;
        loop {
            let b = self.byte()?;
            v |= ((b & 0x7f) as i64) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                if shift < bits && b & 0x40 != 0 {
                    v |= -1i64 << shift;
                }
                return Ok(v);
            }
            if shift > 64 {
                return malformed("overlong signed LEB128");
            }
        }
    }

    fn name(&mut self) -> Res<String> {
        let n = self.u()? as usize;
        let b = self.bytes(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| Error::Malformed("name is not UTF-8".into()))
    }

    fn f32(&mut self) -> Res<f32> {
        let b = self.bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f64(&mut self) -> Res<f64> {
        let b = self.bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn end(&self) -> bool {
        self.at >= self.d.len()
    }
}

impl Module {
    /// Parses and load-validates a module.
    pub fn load(bytes: &[u8]) -> Res<Module> {
        if bytes.len() < 8 || &bytes[0..4] != b"\0asm" {
            return malformed("bad magic");
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return malformed(format!("version {version} is not 1"));
        }
        let mut m = Module {
            bytes: bytes.to_vec(),
            types: Vec::new(),
            imports: Vec::new(),
            funcs: Vec::new(),
            globals: Vec::new(),
            exports: BTreeMap::new(),
            memory: None,
            table: Vec::new(),
            data: Vec::new(),
            start: None,
        };
        let mut r = Reader::new(&bytes[8..]);
        let base = 8usize;
        let mut func_types: Vec<u32> = Vec::new();
        while !r.end() {
            let id = r.byte()?;
            let len = r.u()? as usize;
            let body_start = r.at;
            let body = r.bytes(len)?;
            let mut s = Reader::new(body);
            match id {
                // custom: ignored, as any host may (§11.6 says nothing about it,
                // and a name section is not semantics).
                0 => {}
                1 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        if s.byte()? != 0x60 {
                            return malformed("a type is not a function type");
                        }
                        let np = s.u()? as usize;
                        let mut params = Vec::with_capacity(np.min(1024));
                        for _ in 0..np {
                            params.push(ValType::parse(s.byte()?)?);
                        }
                        let nr = s.u()? as usize;
                        let mut results = Vec::with_capacity(nr.min(1024));
                        for _ in 0..nr {
                            results.push(ValType::parse(s.byte()?)?);
                        }
                        m.types.push(FuncType { params, results });
                    }
                }
                2 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let module = s.name()?;
                        let field = s.name()?;
                        let kind = s.byte()?;
                        // §11.6: imports are none, except the host ABI. A module
                        // that wants anything else is refused here rather than
                        // discovering at run time that it cannot have it.
                        if module != HOST_MODULE {
                            return Err(Error::Forbidden(format!(
                                "import from `{module}`; only `{HOST_MODULE}` is permitted"
                            )));
                        }
                        if kind != 0 {
                            return Err(Error::Forbidden(format!(
                                "`{module}.{field}` is not a function import; a plugin may not \
                                 import memories, tables or globals"
                            )));
                        }
                        let ty = s.u32()?;
                        let host = HostFn::from_name(&field).ok_or_else(|| {
                            Error::Forbidden(format!("`{HOST_MODULE}` has no `{field}`"))
                        })?;
                        let declared = m
                            .types
                            .get(ty as usize)
                            .ok_or_else(|| Error::Malformed("import type index".into()))?;
                        if *declared != host.signature() {
                            return Err(Error::Malformed(format!(
                                "`{field}` is imported with the wrong signature"
                            )));
                        }
                        m.imports.push(host);
                    }
                }
                3 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        func_types.push(s.u32()?);
                    }
                }
                4 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let et = s.byte()?;
                        if et != 0x70 {
                            return Err(Error::Unsupported(
                                "a table of anything but funcref".into(),
                            ));
                        }
                        let flags = s.byte()?;
                        let min = s.u32()? as usize;
                        if flags == 1 {
                            let _max = s.u32()?;
                        }
                        if min > 1 << 20 {
                            return Err(Error::Limit("table larger than 2^20".into()));
                        }
                        m.table = vec![None; min];
                    }
                }
                5 => {
                    let n = s.u()?;
                    if n > 1 {
                        return Err(Error::Unsupported("more than one memory".into()));
                    }
                    for _ in 0..n {
                        let flags = s.byte()?;
                        if flags & 0x02 != 0 {
                            return Err(Error::Forbidden("a shared memory (threads)".into()));
                        }
                        let min = s.u32()?;
                        let max = if flags & 1 == 1 { Some(s.u32()?) } else { None };
                        m.memory = Some((min, max));
                    }
                }
                6 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let ty = ValType::parse(s.byte()?)?;
                        let mutable = match s.byte()? {
                            0 => false,
                            1 => true,
                            _ => return malformed("global mutability"),
                        };
                        let init = const_expr(&mut s, ty, &m.globals)?;
                        m.globals.push(Global { ty, mutable, init });
                    }
                }
                7 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let name = s.name()?;
                        let kind = s.byte()?;
                        let idx = s.u32()?;
                        let e = match kind {
                            0 => Export::Func(idx),
                            1 => Export::Table,
                            2 => Export::Memory,
                            3 => Export::Global(idx),
                            _ => return malformed("export kind"),
                        };
                        m.exports.insert(name, e);
                    }
                }
                8 => m.start = Some(s.u32()?),
                9 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let flags = s.u32()?;
                        if flags != 0 {
                            return Err(Error::Unsupported(
                                "passive or table-indexed element segments".into(),
                            ));
                        }
                        let offset = match const_expr(&mut s, ValType::I32, &m.globals)? {
                            Value::I32(n) => n as usize,
                            _ => return malformed("element offset is not i32"),
                        };
                        let count = s.u()? as usize;
                        for i in 0..count {
                            let f = s.u32()?;
                            let at = offset + i;
                            if at >= m.table.len() {
                                return Err(Error::Malformed(
                                    "an element segment runs past the table".into(),
                                ));
                            }
                            m.table[at] = Some(f);
                        }
                    }
                }
                10 => {
                    let n = s.u()? as usize;
                    if n != func_types.len() {
                        return malformed("function and code section lengths disagree");
                    }
                    for ty in func_types.iter().take(n) {
                        let size = s.u()? as usize;
                        let start = s.at;
                        let end = start + size;
                        if end > body.len() {
                            return malformed("a function body runs past the section");
                        }
                        let mut f = Reader::new(&body[start..end]);
                        let nlocal = f.u()?;
                        let mut locals = Vec::new();
                        for _ in 0..nlocal {
                            let count = f.u()? as usize;
                            let t = ValType::parse(f.byte()?)?;
                            if locals.len() + count > 1 << 16 {
                                return Err(Error::Limit("more than 65536 locals".into()));
                            }
                            for _ in 0..count {
                                locals.push(t);
                            }
                        }
                        // Body bytes, as an absolute range into `self.bytes`.
                        let abs = base + body_start + start + f.at;
                        m.funcs.push(Func {
                            ty: *ty,
                            locals,
                            body: abs..base + body_start + end,
                        });
                        s.at = end;
                    }
                }
                11 => {
                    let n = s.u()?;
                    for _ in 0..n {
                        let flags = s.u32()?;
                        let offset = match flags {
                            0 => match const_expr(&mut s, ValType::I32, &m.globals)? {
                                Value::I32(n) => Some(n as u32),
                                _ => return malformed("data offset is not i32"),
                            },
                            1 => None, // passive: only reachable via memory.init
                            _ => return Err(Error::Unsupported("memory-indexed data".into())),
                        };
                        let len = s.u()? as usize;
                        let bytes = s.bytes(len)?.to_vec();
                        m.data.push((offset, bytes));
                    }
                }
                12 => {
                    let _count = s.u()?;
                }
                13 => return Err(Error::Forbidden("a tag section (exceptions)".into())),
                other => return malformed(format!("unknown section {other}")),
            }
        }
        // Every function body is scanned once at load: this is where a forbidden
        // proposal is caught, so nothing can smuggle one past by hiding it
        // behind a branch that validation would not reach.
        for i in 0..m.funcs.len() {
            let range = m.funcs[i].body.clone();
            scan_body(&m.bytes[range])?;
        }
        if m.funcs.is_empty() && m.imports.is_empty() {
            return malformed("a module with no functions");
        }
        Ok(m)
    }

    pub fn exported_functions(&self) -> Vec<String> {
        self.exports
            .iter()
            .filter(|(_, e)| matches!(e, Export::Func(_)))
            .map(|(n, _)| n.clone())
            .collect()
    }

    pub fn func_type(&self, name: &str) -> Option<&FuncType> {
        match self.exports.get(name) {
            Some(Export::Func(i)) => {
                let i = *i as usize;
                if i < self.imports.len() {
                    return None;
                }
                self.types
                    .get(self.funcs.get(i - self.imports.len())?.ty as usize)
            }
            _ => None,
        }
    }

    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }
}

fn const_expr(r: &mut Reader<'_>, want: ValType, globals: &[Global]) -> Res<Value> {
    let op = r.byte()?;
    let v = match op {
        0x41 => Value::I32(r.i(32)? as i32),
        0x42 => Value::I64(r.i(64)?),
        0x43 => Value::F32(r.f32()?),
        0x44 => Value::F64(r.f64()?),
        0x23 => {
            let i = r.u32()? as usize;
            globals
                .get(i)
                .ok_or_else(|| Error::Malformed("global initializer index".into()))?
                .init
        }
        0xd0 => {
            let _t = r.byte()?;
            Value::Ref(None)
        }
        other => return malformed(format!("constant expression opcode {other:#04x}")),
    };
    if r.byte()? != 0x0b {
        return malformed("a constant expression does not end");
    }
    if v.ty() != want {
        return malformed("a constant expression has the wrong type");
    }
    Ok(v)
}

/// Walks a function body's opcodes, rejecting what §11.6 forbids and what this
/// host does not implement.
fn scan_body(body: &[u8]) -> Res<()> {
    let mut r = Reader::new(body);
    while !r.end() {
        let op = r.byte()?;
        match op {
            // Prefixes that carry whole proposals.
            0xfe => return Err(Error::Forbidden("atomic instructions (threads)".into())),
            0x06 | 0x07 | 0x08 | 0x09 | 0x18 | 0x19 => {
                return Err(Error::Forbidden("exception handling".into()))
            }
            0xfb => return Err(Error::Unsupported("GC instructions".into())),
            // Everything else: skip immediates so the walk stays aligned.
            _ => skip_immediates(&mut r, op)?,
        }
    }
    Ok(())
}

fn skip_immediates(r: &mut Reader<'_>, op: u8) -> Res<()> {
    match op {
        // No immediates.
        0x00 | 0x01 | 0x0b | 0x0f | 0x1a | 0x1b | 0x45..=0xc4 | 0xd1 => {}
        // Block types.
        0x02..=0x04 => {
            let b = r.byte()?;
            if b != 0x40 && ValType::parse(b).is_err() {
                // A multi-value block type is a type index, LEB-encoded, and the
                // byte we just read was its first byte.
                r.at -= 1;
                r.i(33)?;
            }
        }
        0x05 => {}
        0x0c | 0x0d => {
            r.u()?;
        }
        0x0e => {
            let n = r.u()?;
            for _ in 0..=n {
                r.u()?;
            }
        }
        0x10 | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 | 0x26 => {
            r.u()?;
        }
        0x11 => {
            r.u()?;
            r.u()?;
        }
        0x1c => {
            let n = r.u()?;
            for _ in 0..n {
                r.byte()?;
            }
        }
        0x28..=0x3e => {
            r.u()?;
            r.u()?;
        }
        0x3f | 0x40 => {
            r.byte()?;
        }
        0x41 => {
            r.i(32)?;
        }
        0x42 => {
            r.i(64)?;
        }
        0x43 => {
            r.bytes(4)?;
        }
        0x44 => {
            r.bytes(8)?;
        }
        0xd0 => {
            r.byte()?;
        }
        0xd2 => {
            r.u()?;
        }
        0xfc => {
            let sub = r.u()?;
            match sub {
                // saturating truncation: no immediates
                0..=7 => {}
                // memory.init, memory.copy, memory.fill, data.drop
                8 => {
                    r.u()?;
                    r.byte()?;
                }
                9 => {
                    r.u()?;
                }
                10 => {
                    r.byte()?;
                    r.byte()?;
                }
                11 => {
                    r.byte()?;
                }
                12..=17 => {
                    r.u()?;
                    r.u()?;
                }
                other => return Err(Error::Unsupported(format!("0xfc {other}"))),
            }
        }
        // The SIMD prefix. The walk has to know each form's immediates or it
        // loses alignment and starts reading operands as opcodes — which is how
        // a scanner reports a forbidden instruction that is not there.
        0xfd => {
            let sub = r.u()?;
            match sub {
                // Loads and stores: a memarg, and for the lane forms a lane index.
                0x00..=0x0b | 0x5c | 0x5d => {
                    r.u()?;
                    r.u()?;
                }
                0x54..=0x5b => {
                    r.u()?;
                    r.u()?;
                    r.byte()?;
                }
                // v128.const and i8x16.shuffle: sixteen immediate bytes each.
                0x0c | 0x0d => {
                    for _ in 0..16 {
                        r.byte()?;
                    }
                }
                // extract_lane / replace_lane: one lane index.
                0x15..=0x22 => {
                    r.byte()?;
                }
                // Everything else in the fixed-width set takes its operands from
                // the stack and carries no immediates.
                0x0e..=0x14 | 0x23..=0x53 | 0x5e..=0xff => {}
                0x100..=0x15f => {
                    return Err(Error::Forbidden(format!(
                        "relaxed-SIMD instruction {sub:#x} (nondeterministic)"
                    )))
                }
                other => return Err(Error::Unsupported(format!("SIMD opcode {other:#x}"))),
            }
        }
        other => return malformed(format!("opcode {other:#04x}")),
    }
    Ok(())
}

// --------------------------------------------------------------- the machine --

/// What the host lets a plugin see. §11.6's `read_object` is "read-only,
/// sandboxed to declared refs", so the set of objects a plugin may read is
/// supplied by the caller and nothing else is reachable.
pub struct Env<'a> {
    /// Objects the plugin was declared against, by digest.
    pub objects: &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>,
    /// Where `log` goes. Collected rather than printed: a host that writes to
    /// stderr from inside an evaluator is a surprise.
    pub log: std::cell::RefCell<Vec<String>>,
}

impl Default for Env<'_> {
    fn default() -> Self {
        Env {
            objects: &|_| None,
            log: std::cell::RefCell::new(Vec::new()),
        }
    }
}

/// Resource limits (§11.6).
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Instructions the module may execute before it traps.
    pub fuel: u64,
    /// Maximum linear memory, in bytes.
    pub memory: usize,
    /// Maximum call depth.
    pub depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            fuel: 100_000_000,
            memory: 256 << 20,
            depth: 512,
        }
    }
}

/// A module instance: its memory, globals and table, plus the fuel it has left.
pub struct Instance<'m, 'e> {
    m: &'m Module,
    env: &'e Env<'e>,
    limits: Limits,
    mem: Vec<u8>,
    globals: Vec<Value>,
    table: Vec<Option<u32>>,
    fuel: u64,
    depth: usize,
}

impl<'m, 'e> Instance<'m, 'e> {
    pub fn new(m: &'m Module, env: &'e Env<'e>, limits: Limits) -> Res<Instance<'m, 'e>> {
        let pages = m.memory.map(|(min, _)| min as usize).unwrap_or(0);
        if pages * PAGE > limits.memory {
            return Err(Error::Limit(format!(
                "the module wants {} bytes of memory, over the {} cap",
                pages * PAGE,
                limits.memory
            )));
        }
        let mut inst = Instance {
            m,
            env,
            limits,
            mem: vec![0u8; pages * PAGE],
            globals: m.globals.iter().map(|g| g.init).collect(),
            table: m.table.clone(),
            fuel: limits.fuel,
            depth: 0,
        };
        for (offset, bytes) in &m.data {
            if let Some(off) = offset {
                let at = *off as usize;
                if at + bytes.len() > inst.mem.len() {
                    return Err(Error::Malformed(
                        "a data segment runs past the declared memory".into(),
                    ));
                }
                inst.mem[at..at + bytes.len()].copy_from_slice(bytes);
            }
        }
        if let Some(start) = m.start {
            inst.call_index(start, &[])?;
        }
        Ok(inst)
    }

    pub fn fuel_used(&self) -> u64 {
        self.limits.fuel - self.fuel
    }

    pub fn fuel_left(&self) -> u64 {
        self.fuel
    }

    pub fn memory(&self) -> &[u8] {
        &self.mem
    }

    pub fn logs(&self) -> Vec<String> {
        self.env.log.borrow().clone()
    }

    /// Calls an exported function.
    pub fn call(&mut self, name: &str, args: &[Value]) -> Res<Vec<Value>> {
        let Some(Export::Func(i)) = self.m.exports.get(name).copied() else {
            return Err(Error::Abi(format!("no exported function `{name}`")));
        };
        self.call_index(i, args)
    }

    /// Writes bytes into the module's memory at an address the module allocated.
    pub fn write(&mut self, at: u32, bytes: &[u8]) -> Res<()> {
        let a = at as usize;
        if a.checked_add(bytes.len())
            .is_none_or(|e| e > self.mem.len())
        {
            return Err(Error::Abi(
                "a host write would run past linear memory".into(),
            ));
        }
        self.mem[a..a + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn read(&self, at: u32, n: usize) -> Res<Vec<u8>> {
        let a = at as usize;
        if a.checked_add(n).is_none_or(|e| e > self.mem.len()) {
            return Err(Error::Abi(
                "a host read would run past linear memory".into(),
            ));
        }
        Ok(self.mem[a..a + n].to_vec())
    }

    /// `alloc(n) -> ptr`, which §11.6 makes the module's responsibility.
    pub fn alloc(&mut self, n: u32) -> Res<u32> {
        let out = self.call("alloc", &[Value::I32(n as i32)])?;
        match out.first() {
            Some(Value::I32(p)) if *p > 0 => Ok(*p as u32),
            Some(Value::I32(_)) => Err(Error::Abi("alloc returned a null pointer".into())),
            _ => Err(Error::Abi("alloc did not return an i32".into())),
        }
    }

    pub fn dealloc(&mut self, ptr: u32, n: u32) -> Res<()> {
        if self.m.exports.contains_key("dealloc") {
            self.call("dealloc", &[Value::I32(ptr as i32), Value::I32(n as i32)])?;
        }
        Ok(())
    }

    fn burn(&mut self, n: u64) -> Res<()> {
        self.fuel = self.fuel.checked_sub(n).ok_or_else(|| {
            Error::Limit(format!(
                "out of fuel after {} instructions",
                self.limits.fuel
            ))
        })?;
        Ok(())
    }

    fn call_index(&mut self, idx: u32, args: &[Value]) -> Res<Vec<Value>> {
        let i = idx as usize;
        if i < self.m.imports.len() {
            return self.host_call(self.m.imports[i], args);
        }
        let f = self
            .m
            .funcs
            .get(i - self.m.imports.len())
            .ok_or_else(|| Error::Trap(format!("call to function {idx}, which does not exist")))?
            .clone();
        let ty = self
            .m
            .types
            .get(f.ty as usize)
            .ok_or_else(|| Error::Malformed("function type index".into()))?
            .clone();
        if args.len() != ty.params.len() {
            return Err(Error::Abi(format!(
                "function {idx} takes {} argument(s), given {}",
                ty.params.len(),
                args.len()
            )));
        }
        for (a, want) in args.iter().zip(&ty.params) {
            if a.ty() != *want {
                return Err(Error::Abi(format!(
                    "argument type mismatch: {:?} where {want:?} was declared",
                    a.ty()
                )));
            }
        }
        self.depth += 1;
        if self.depth > self.limits.depth {
            self.depth -= 1;
            return Err(Error::Limit(format!(
                "call depth exceeded {} frames",
                self.limits.depth
            )));
        }
        let mut locals: Vec<Value> = args.to_vec();
        locals.extend(f.locals.iter().map(|t| Value::zero(*t)));
        let body = self.m.bytes[f.body.clone()].to_vec();
        let r = self.exec(&body, &mut locals, &ty);
        self.depth -= 1;
        r
    }

    fn host_call(&mut self, f: HostFn, args: &[Value]) -> Res<Vec<Value>> {
        self.burn(10)?;
        match f {
            HostFn::Log => {
                let (ptr, len) = (args[0].as_i32()? as u32, args[1].as_i32()? as usize);
                let bytes = self.read(ptr, len.min(1 << 16))?;
                self.env
                    .log
                    .borrow_mut()
                    .push(String::from_utf8_lossy(&bytes).to_string());
                Ok(Vec::new())
            }
            HostFn::Abort => {
                let (ptr, len) = (args[0].as_i32()? as u32, args[1].as_i32()? as usize);
                let bytes = self.read(ptr, len.min(1 << 16)).unwrap_or_default();
                Err(Error::Trap(format!(
                    "the plugin aborted: {}",
                    String::from_utf8_lossy(&bytes)
                )))
            }
            HostFn::ReadObject => {
                let digest_ptr = args[0].as_i32()? as u32;
                let out_ptr = args[1].as_i32()? as u32;
                let cap = args[2].as_i32()?;
                if cap < 0 {
                    return Err(Error::Abi("read_object with a negative capacity".into()));
                }
                let d = self.read(digest_ptr, 32)?;
                let mut digest = [0u8; 32];
                digest.copy_from_slice(&d);
                match (self.env.objects)(&digest) {
                    // -1: no such object among the refs this plugin declared.
                    None => Ok(vec![Value::I32(-1)]),
                    Some(bytes) => {
                        if bytes.len() > cap as usize {
                            // -2: the caller's buffer is too small. The needed
                            // size is deliberately not returned as a positive
                            // number, because a plugin that ignored the sign
                            // would then read uninitialized memory.
                            return Ok(vec![Value::I32(-2)]);
                        }
                        self.write(out_ptr, &bytes)?;
                        Ok(vec![Value::I32(bytes.len() as i32)])
                    }
                }
            }
        }
    }
}

/// A structured control-flow label: where a branch out of the construct lands,
/// and what it leaves on the stack.
struct Label {
    /// Where a branch to this label jumps.
    target: usize,
    /// Stack height when the construct was entered.
    height: usize,
    /// How many values a branch to this label carries.
    arity: usize,
    /// Loops branch backwards to their own start; blocks and ifs forwards to
    /// their end.
    is_loop: bool,
}

/// The block type of a `block`/`loop`/`if`, resolved to its arity.
fn block_arity(m: &Module, r: &mut Reader<'_>) -> Res<(usize, usize)> {
    let b = r.byte()?;
    match b {
        0x40 => Ok((0, 0)),
        0x7f | 0x7e | 0x7d | 0x7c | 0x70 => Ok((0, 1)),
        _ => {
            r.at -= 1;
            let idx = r.i(33)?;
            if idx < 0 {
                return malformed("negative block type index");
            }
            let t = m
                .types
                .get(idx as usize)
                .ok_or_else(|| Error::Malformed("block type index".into()))?;
            Ok((t.params.len(), t.results.len()))
        }
    }
}

/// Finds the matching `else` and `end` for the construct starting at `from`.
fn matching(body: &[u8], from: usize) -> Res<(Option<usize>, usize)> {
    let mut r = Reader::new(body);
    r.at = from;
    let mut depth = 0i32;
    let mut else_at = None;
    while !r.end() {
        let here = r.at;
        let op = r.byte()?;
        match op {
            0x02..=0x04 => {
                depth += 1;
                skip_immediates(&mut r, op)?;
            }
            0x05 if depth == 0 => {
                else_at = Some(here + 1);
            }
            0x0b => {
                if depth == 0 {
                    return Ok((else_at, here));
                }
                depth -= 1;
            }
            _ => skip_immediates(&mut r, op)?,
        }
    }
    malformed("a block does not end")
}

macro_rules! binop {
    ($stack:expr, $as:ident, $wrap:path, $f:expr) => {{
        let b = pop($stack)?.$as()?;
        let a = pop($stack)?.$as()?;
        let f: fn(_, _) -> _ = $f;
        $stack.push($wrap(f(a, b)));
    }};
}

macro_rules! cmpop {
    ($stack:expr, $as:ident, $f:expr) => {{
        let b = pop($stack)?.$as()?;
        let a = pop($stack)?.$as()?;
        let f: fn(_, _) -> bool = $f;
        $stack.push(Value::I32(if f(a, b) { 1 } else { 0 }));
    }};
}

macro_rules! unop {
    ($stack:expr, $as:ident, $wrap:path, $f:expr) => {{
        let a = pop($stack)?.$as()?;
        let f: fn(_) -> _ = $f;
        $stack.push($wrap(f(a)));
    }};
}

fn pop(stack: &mut Vec<Value>) -> Res<Value> {
    stack
        .pop()
        .ok_or_else(|| Error::Trap("stack underflow".into()))
}

/// NaN canonicalization. §11.6 requires determinism and forbids "NaN-payload
/// dependence"; the simplest way to have neither is to produce one NaN.
fn canon32(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7fc0_0000)
    } else {
        x
    }
}

fn canon64(x: f64) -> f64 {
    if x.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        x
    }
}

impl Instance<'_, '_> {
    fn exec(&mut self, body: &[u8], locals: &mut [Value], ty: &FuncType) -> Res<Vec<Value>> {
        let mut stack: Vec<Value> = Vec::new();
        let mut labels: Vec<Label> = Vec::new();
        let mut r = Reader::new(body);
        loop {
            if r.end() {
                break;
            }
            self.burn(1)?;
            let op = r.byte()?;
            match op {
                0x00 => return Err(Error::Trap("unreachable".into())),
                0x01 => {}
                0x02 | 0x03 => {
                    let (params, results) = block_arity(self.m, &mut r)?;
                    let (_, end) = matching(body, r.at)?;
                    let is_loop = op == 0x03;
                    labels.push(Label {
                        target: if is_loop { r.at } else { end + 1 },
                        height: stack.len().saturating_sub(params),
                        arity: if is_loop { params } else { results },
                        is_loop,
                    });
                }
                0x04 => {
                    let (params, results) = block_arity(self.m, &mut r)?;
                    let (else_at, end) = matching(body, r.at)?;
                    let cond = pop(&mut stack)?.as_i32()?;
                    labels.push(Label {
                        target: end + 1,
                        height: stack.len().saturating_sub(params),
                        arity: results,
                        is_loop: false,
                    });
                    if cond == 0 {
                        match else_at {
                            Some(e) => r.at = e,
                            None => {
                                labels.pop();
                                r.at = end + 1;
                            }
                        }
                    }
                }
                0x05 => {
                    // Reached by falling out of the `then` arm: skip the `else`.
                    let l = labels
                        .last()
                        .ok_or_else(|| Error::Trap("else outside an if".into()))?;
                    r.at = l.target;
                    let arity = l.arity;
                    let height = l.height;
                    truncate_to(&mut stack, height, arity)?;
                    labels.pop();
                }
                0x0b => {
                    match labels.pop() {
                        Some(l) => {
                            truncate_to(&mut stack, l.height, l.arity)?;
                        }
                        // The function's own `end`.
                        None => break,
                    }
                }
                0x0c | 0x0d => {
                    let depth = r.u32()? as usize;
                    let taken = if op == 0x0c {
                        true
                    } else {
                        pop(&mut stack)?.as_i32()? != 0
                    };
                    if taken && branch(&mut stack, &mut labels, &mut r, depth, ty)? {
                        return finish(stack, ty);
                    }
                }
                0x0e => {
                    let n = r.u()? as usize;
                    let mut targets = Vec::with_capacity(n.min(1 << 16));
                    for _ in 0..n {
                        targets.push(r.u32()? as usize);
                    }
                    let default = r.u32()? as usize;
                    let i = pop(&mut stack)?.as_i32()?;
                    let depth = if i < 0 || i as usize >= targets.len() {
                        default
                    } else {
                        targets[i as usize]
                    };
                    if branch(&mut stack, &mut labels, &mut r, depth, ty)? {
                        return finish(stack, ty);
                    }
                }
                0x0f => return finish(stack, ty),
                0x10 => {
                    let idx = r.u32()?;
                    let sig = self.signature_of(idx)?;
                    let args = take_args(&mut stack, sig.params.len())?;
                    let out = self.call_index(idx, &args)?;
                    stack.extend(out);
                }
                0x11 => {
                    let type_idx = r.u32()? as usize;
                    let _table = r.u()?;
                    let want = self
                        .m
                        .types
                        .get(type_idx)
                        .ok_or_else(|| Error::Malformed("call_indirect type".into()))?
                        .clone();
                    let slot = pop(&mut stack)?.as_i32()?;
                    if slot < 0 || slot as usize >= self.table.len() {
                        return Err(Error::Trap("call_indirect: index out of bounds".into()));
                    }
                    let Some(f) = self.table[slot as usize] else {
                        return Err(Error::Trap("call_indirect: null function".into()));
                    };
                    let sig = self.signature_of(f)?;
                    if sig != want {
                        // The trap §11.6's determinism depends on: an indirect
                        // call whose type does not match must fail, not
                        // reinterpret the stack.
                        return Err(Error::Trap("call_indirect: signature mismatch".into()));
                    }
                    let args = take_args(&mut stack, sig.params.len())?;
                    let out = self.call_index(f, &args)?;
                    stack.extend(out);
                }
                0x1a => {
                    pop(&mut stack)?;
                }
                0x1b | 0x1c => {
                    if op == 0x1c {
                        let n = r.u()?;
                        for _ in 0..n {
                            r.byte()?;
                        }
                    }
                    let cond = pop(&mut stack)?.as_i32()?;
                    let b = pop(&mut stack)?;
                    let a = pop(&mut stack)?;
                    stack.push(if cond != 0 { a } else { b });
                }
                0x20 => {
                    let i = r.u32()? as usize;
                    stack.push(
                        *locals
                            .get(i)
                            .ok_or_else(|| Error::Trap("local.get out of range".into()))?,
                    );
                }
                0x21 | 0x22 => {
                    let i = r.u32()? as usize;
                    let v = if op == 0x21 {
                        pop(&mut stack)?
                    } else {
                        *stack
                            .last()
                            .ok_or_else(|| Error::Trap("local.tee on an empty stack".into()))?
                    };
                    let slot = locals
                        .get_mut(i)
                        .ok_or_else(|| Error::Trap("local.set out of range".into()))?;
                    if slot.ty() != v.ty() {
                        return Err(Error::Trap("local.set type mismatch".into()));
                    }
                    *slot = v;
                }
                0x23 => {
                    let i = r.u32()? as usize;
                    stack.push(
                        *self
                            .globals
                            .get(i)
                            .ok_or_else(|| Error::Trap("global.get out of range".into()))?,
                    );
                }
                0x24 => {
                    let i = r.u32()? as usize;
                    let v = pop(&mut stack)?;
                    let g = self
                        .m
                        .globals
                        .get(i)
                        .ok_or_else(|| Error::Trap("global.set out of range".into()))?;
                    if !g.mutable {
                        return Err(Error::Trap("global.set on an immutable global".into()));
                    }
                    if g.ty != v.ty() {
                        return Err(Error::Trap("global.set type mismatch".into()));
                    }
                    self.globals[i] = v;
                }
                // Loads.
                0x28..=0x35 => {
                    let _align = r.u()?;
                    let offset = r.u()?;
                    let base = pop(&mut stack)?.as_i32()? as u32 as u64;
                    let at = base
                        .checked_add(offset)
                        .ok_or_else(|| Error::Trap("address overflow".into()))?;
                    let v = self.load(op, at)?;
                    stack.push(v);
                }
                // Stores.
                0x36..=0x3e => {
                    let _align = r.u()?;
                    let offset = r.u()?;
                    let v = pop(&mut stack)?;
                    let base = pop(&mut stack)?.as_i32()? as u32 as u64;
                    let at = base
                        .checked_add(offset)
                        .ok_or_else(|| Error::Trap("address overflow".into()))?;
                    self.store(op, at, v)?;
                }
                0x3f => {
                    let _mem = r.byte()?;
                    stack.push(Value::I32((self.mem.len() / PAGE) as i32));
                }
                0x40 => {
                    let _mem = r.byte()?;
                    let pages = pop(&mut stack)?.as_i32()?;
                    let old = (self.mem.len() / PAGE) as i32;
                    if pages < 0 {
                        stack.push(Value::I32(-1));
                    } else {
                        let want = self.mem.len() + pages as usize * PAGE;
                        if want > self.limits.memory {
                            // §11.6 caps memory, and the cap is expressed the way
                            // WebAssembly expresses failure: -1, not a trap.
                            stack.push(Value::I32(-1));
                        } else {
                            self.mem.resize(want, 0);
                            stack.push(Value::I32(old));
                        }
                    }
                }
                0x41 => stack.push(Value::I32(r.i(32)? as i32)),
                0x42 => stack.push(Value::I64(r.i(64)?)),
                0x43 => stack.push(Value::F32(canon32(r.f32()?))),
                0x44 => stack.push(Value::F64(canon64(r.f64()?))),
                0x45 => {
                    let a = pop(&mut stack)?.as_i32()?;
                    stack.push(Value::I32(i32::from(a == 0)));
                }
                0x46 => cmpop!(&mut stack, as_i32, |a, b| a == b),
                0x47 => cmpop!(&mut stack, as_i32, |a, b| a != b),
                0x48 => cmpop!(&mut stack, as_i32, |a: i32, b: i32| a < b),
                0x49 => cmpop!(&mut stack, as_i32, |a: i32, b: i32| (a as u32) < b as u32),
                0x4a => cmpop!(&mut stack, as_i32, |a: i32, b: i32| a > b),
                0x4b => cmpop!(&mut stack, as_i32, |a: i32, b: i32| (a as u32) > b as u32),
                0x4c => cmpop!(&mut stack, as_i32, |a: i32, b: i32| a <= b),
                0x4d => cmpop!(&mut stack, as_i32, |a: i32, b: i32| (a as u32) <= b as u32),
                0x4e => cmpop!(&mut stack, as_i32, |a: i32, b: i32| a >= b),
                0x4f => cmpop!(&mut stack, as_i32, |a: i32, b: i32| (a as u32) >= b as u32),
                0x50 => {
                    let a = pop(&mut stack)?.as_i64()?;
                    stack.push(Value::I32(i32::from(a == 0)));
                }
                0x51 => cmpop!(&mut stack, as_i64, |a, b| a == b),
                0x52 => cmpop!(&mut stack, as_i64, |a, b| a != b),
                0x53 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| a < b),
                0x54 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| (a as u64) < b as u64),
                0x55 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| a > b),
                0x56 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| (a as u64) > b as u64),
                0x57 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| a <= b),
                0x58 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| (a as u64) <= b as u64),
                0x59 => cmpop!(&mut stack, as_i64, |a: i64, b: i64| a >= b),
                0x5a => cmpop!(&mut stack, as_i64, |a: i64, b: i64| (a as u64) >= b as u64),
                0x5b => cmpop!(&mut stack, as_f32, |a, b| a == b),
                0x5c => cmpop!(&mut stack, as_f32, |a, b| a != b),
                0x5d => cmpop!(&mut stack, as_f32, |a, b| a < b),
                0x5e => cmpop!(&mut stack, as_f32, |a, b| a > b),
                0x5f => cmpop!(&mut stack, as_f32, |a, b| a <= b),
                0x60 => cmpop!(&mut stack, as_f32, |a, b| a >= b),
                0x61 => cmpop!(&mut stack, as_f64, |a, b| a == b),
                0x62 => cmpop!(&mut stack, as_f64, |a, b| a != b),
                0x63 => cmpop!(&mut stack, as_f64, |a, b| a < b),
                0x64 => cmpop!(&mut stack, as_f64, |a, b| a > b),
                0x65 => cmpop!(&mut stack, as_f64, |a, b| a <= b),
                0x66 => cmpop!(&mut stack, as_f64, |a, b| a >= b),
                0x67 => unop!(&mut stack, as_i32, Value::I32, |a: i32| a.leading_zeros()
                    as i32),
                0x68 => unop!(&mut stack, as_i32, Value::I32, |a: i32| a.trailing_zeros()
                    as i32),
                0x69 => unop!(&mut stack, as_i32, Value::I32, |a: i32| a.count_ones()
                    as i32),
                0x6a => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .wrapping_add(b)),
                0x6b => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .wrapping_sub(b)),
                0x6c => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .wrapping_mul(b)),
                0x6d..=0x70 => {
                    let b = pop(&mut stack)?.as_i32()?;
                    let a = pop(&mut stack)?.as_i32()?;
                    if b == 0 {
                        return Err(Error::Trap("integer divide by zero".into()));
                    }
                    let v = match op {
                        0x6d => a
                            .checked_div(b)
                            .ok_or_else(|| Error::Trap("integer overflow".into()))?,
                        0x6e => ((a as u32) / (b as u32)) as i32,
                        0x6f => a
                            .checked_rem(b)
                            .ok_or_else(|| Error::Trap("integer overflow".into()))?,
                        _ => ((a as u32) % (b as u32)) as i32,
                    };
                    stack.push(Value::I32(v));
                }
                0x71 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a & b),
                0x72 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a | b),
                0x73 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a ^ b),
                0x74 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .wrapping_shl(b as u32)),
                0x75 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .wrapping_shr(b as u32)),
                0x76 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| ((a as u32)
                    .wrapping_shr(b as u32))
                    as i32),
                0x77 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .rotate_left(b as u32 & 31)),
                0x78 => binop!(&mut stack, as_i32, Value::I32, |a: i32, b: i32| a
                    .rotate_right(b as u32 & 31)),
                0x79 => unop!(&mut stack, as_i64, Value::I64, |a: i64| a.leading_zeros()
                    as i64),
                0x7a => unop!(&mut stack, as_i64, Value::I64, |a: i64| a.trailing_zeros()
                    as i64),
                0x7b => unop!(&mut stack, as_i64, Value::I64, |a: i64| a.count_ones()
                    as i64),
                0x7c => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .wrapping_add(b)),
                0x7d => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .wrapping_sub(b)),
                0x7e => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .wrapping_mul(b)),
                0x7f..=0x82 => {
                    let b = pop(&mut stack)?.as_i64()?;
                    let a = pop(&mut stack)?.as_i64()?;
                    if b == 0 {
                        return Err(Error::Trap("integer divide by zero".into()));
                    }
                    let v = match op {
                        0x7f => a
                            .checked_div(b)
                            .ok_or_else(|| Error::Trap("integer overflow".into()))?,
                        0x80 => ((a as u64) / (b as u64)) as i64,
                        0x81 => a
                            .checked_rem(b)
                            .ok_or_else(|| Error::Trap("integer overflow".into()))?,
                        _ => ((a as u64) % (b as u64)) as i64,
                    };
                    stack.push(Value::I64(v));
                }
                0x83 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a & b),
                0x84 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a | b),
                0x85 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a ^ b),
                0x86 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .wrapping_shl(b as u32)),
                0x87 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .wrapping_shr(b as u32)),
                0x88 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| ((a as u64)
                    .wrapping_shr(b as u32))
                    as i64),
                0x89 => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .rotate_left(b as u32 & 63)),
                0x8a => binop!(&mut stack, as_i64, Value::I64, |a: i64, b: i64| a
                    .rotate_right(b as u32 & 63)),
                0x8b => unop!(&mut stack, as_f32, Value::F32, |a: f32| a.abs()),
                0x8c => unop!(&mut stack, as_f32, Value::F32, |a: f32| -a),
                0x8d => unop!(&mut stack, as_f32, Value::F32, |a: f32| a.ceil()),
                0x8e => unop!(&mut stack, as_f32, Value::F32, |a: f32| a.floor()),
                0x8f => unop!(&mut stack, as_f32, Value::F32, |a: f32| a.trunc()),
                0x90 => unop!(&mut stack, as_f32, Value::F32, |a: f32| round_even32(a)),
                0x91 => unop!(&mut stack, as_f32, Value::F32, |a: f32| canon32(a.sqrt())),
                0x92 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| canon32(
                    a + b
                )),
                0x93 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| canon32(
                    a - b
                )),
                0x94 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| canon32(
                    a * b
                )),
                0x95 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| canon32(
                    a / b
                )),
                0x96 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| fmin32(
                    a, b
                )),
                0x97 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| fmax32(
                    a, b
                )),
                0x98 => binop!(&mut stack, as_f32, Value::F32, |a: f32, b: f32| a
                    .copysign(b)),
                0x99 => unop!(&mut stack, as_f64, Value::F64, |a: f64| a.abs()),
                0x9a => unop!(&mut stack, as_f64, Value::F64, |a: f64| -a),
                0x9b => unop!(&mut stack, as_f64, Value::F64, |a: f64| a.ceil()),
                0x9c => unop!(&mut stack, as_f64, Value::F64, |a: f64| a.floor()),
                0x9d => unop!(&mut stack, as_f64, Value::F64, |a: f64| a.trunc()),
                0x9e => unop!(&mut stack, as_f64, Value::F64, |a: f64| round_even64(a)),
                0x9f => unop!(&mut stack, as_f64, Value::F64, |a: f64| canon64(a.sqrt())),
                0xa0 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| canon64(
                    a + b
                )),
                0xa1 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| canon64(
                    a - b
                )),
                0xa2 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| canon64(
                    a * b
                )),
                0xa3 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| canon64(
                    a / b
                )),
                0xa4 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| fmin64(
                    a, b
                )),
                0xa5 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| fmax64(
                    a, b
                )),
                0xa6 => binop!(&mut stack, as_f64, Value::F64, |a: f64, b: f64| a
                    .copysign(b)),
                0xa7 => unop!(&mut stack, as_i64, Value::I32, |a: i64| a as i32),
                0xa8 | 0xa9 => {
                    let a = pop(&mut stack)?.as_f32()?;
                    stack.push(Value::I32(trunc_i32(a as f64, op == 0xa9)?));
                }
                0xaa | 0xab => {
                    let a = pop(&mut stack)?.as_f64()?;
                    stack.push(Value::I32(trunc_i32(a, op == 0xab)?));
                }
                0xac => unop!(&mut stack, as_i32, Value::I64, |a: i32| a as i64),
                0xad => unop!(&mut stack, as_i32, Value::I64, |a: i32| a as u32 as i64),
                0xae | 0xaf => {
                    let a = pop(&mut stack)?.as_f32()?;
                    stack.push(Value::I64(trunc_i64(a as f64, op == 0xaf)?));
                }
                0xb0 | 0xb1 => {
                    let a = pop(&mut stack)?.as_f64()?;
                    stack.push(Value::I64(trunc_i64(a, op == 0xb1)?));
                }
                0xb2 => unop!(&mut stack, as_i32, Value::F32, |a: i32| a as f32),
                0xb3 => unop!(&mut stack, as_i32, Value::F32, |a: i32| a as u32 as f32),
                0xb4 => unop!(&mut stack, as_i64, Value::F32, |a: i64| a as f32),
                0xb5 => unop!(&mut stack, as_i64, Value::F32, |a: i64| a as u64 as f32),
                0xb6 => unop!(&mut stack, as_f64, Value::F32, |a: f64| canon32(a as f32)),
                0xb7 => unop!(&mut stack, as_i32, Value::F64, |a: i32| a as f64),
                0xb8 => unop!(&mut stack, as_i32, Value::F64, |a: i32| a as u32 as f64),
                0xb9 => unop!(&mut stack, as_i64, Value::F64, |a: i64| a as f64),
                0xba => unop!(&mut stack, as_i64, Value::F64, |a: i64| a as u64 as f64),
                0xbb => unop!(&mut stack, as_f32, Value::F64, |a: f32| canon64(a as f64)),
                0xbc => unop!(&mut stack, as_f32, Value::I32, |a: f32| a.to_bits() as i32),
                0xbd => unop!(&mut stack, as_f64, Value::I64, |a: f64| a.to_bits() as i64),
                0xbe => unop!(&mut stack, as_i32, Value::F32, |a: i32| canon32(
                    f32::from_bits(a as u32)
                )),
                0xbf => unop!(&mut stack, as_i64, Value::F64, |a: i64| canon64(
                    f64::from_bits(a as u64)
                )),
                // Sign extension.
                0xc0 => unop!(&mut stack, as_i32, Value::I32, |a: i32| a as i8 as i32),
                0xc1 => unop!(&mut stack, as_i32, Value::I32, |a: i32| a as i16 as i32),
                0xc2 => unop!(&mut stack, as_i64, Value::I64, |a: i64| a as i8 as i64),
                0xc3 => unop!(&mut stack, as_i64, Value::I64, |a: i64| a as i16 as i64),
                0xc4 => unop!(&mut stack, as_i64, Value::I64, |a: i64| a as i32 as i64),
                0xd0 => {
                    let _t = r.byte()?;
                    stack.push(Value::Ref(None));
                }
                0xd1 => {
                    let v = pop(&mut stack)?;
                    stack.push(Value::I32(i32::from(matches!(v, Value::Ref(None)))));
                }
                0xd2 => {
                    let f = r.u32()?;
                    stack.push(Value::Ref(Some(f)));
                }
                0xfc => {
                    let sub = r.u()?;
                    self.misc(sub, &mut r, &mut stack)?;
                }
                0xfd => {
                    let sub = r.u()?;
                    self.simd(sub, &mut r, &mut stack)?;
                }
                other => return malformed(format!("opcode {other:#04x} at {}", r.at - 1)),
            }
        }
        finish(stack, ty)
    }

    fn signature_of(&self, idx: u32) -> Res<FuncType> {
        let i = idx as usize;
        if i < self.m.imports.len() {
            return Ok(self.m.imports[i].signature());
        }
        let f = self
            .m
            .funcs
            .get(i - self.m.imports.len())
            .ok_or_else(|| Error::Trap(format!("function {idx} does not exist")))?;
        self.m
            .types
            .get(f.ty as usize)
            .cloned()
            .ok_or_else(|| Error::Malformed("function type index".into()))
    }

    fn slice(&self, at: u64, n: usize) -> Res<&[u8]> {
        let end = at
            .checked_add(n as u64)
            .ok_or_else(|| Error::Trap("address overflow".into()))?;
        if end > self.mem.len() as u64 {
            return Err(Error::Trap(format!(
                "out of bounds memory access at {at} ({n} bytes, memory is {})",
                self.mem.len()
            )));
        }
        Ok(&self.mem[at as usize..at as usize + n])
    }

    /// A bounds-checked guest store. Distinct from `write`, which is the host
    /// ABI and reports `Abi` rather than trapping: an out-of-bounds store by the
    /// *module* is a trap, which is what the module is entitled to.
    fn store_bytes(&mut self, at: u64, bytes: &[u8]) -> Res<()> {
        let end = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::Trap("address overflow".into()))?;
        if end > self.mem.len() as u64 {
            return Err(Error::Trap(format!(
                "out of bounds memory access at {at} ({} bytes, memory is {})",
                bytes.len(),
                self.mem.len()
            )));
        }
        self.mem[at as usize..end as usize].copy_from_slice(bytes);
        Ok(())
    }

    fn load(&mut self, op: u8, at: u64) -> Res<Value> {
        let v = match op {
            0x28 => Value::I32(i32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap())),
            0x29 => Value::I64(i64::from_le_bytes(self.slice(at, 8)?.try_into().unwrap())),
            0x2a => Value::F32(canon32(f32::from_le_bytes(
                self.slice(at, 4)?.try_into().unwrap(),
            ))),
            0x2b => Value::F64(canon64(f64::from_le_bytes(
                self.slice(at, 8)?.try_into().unwrap(),
            ))),
            0x2c => Value::I32(self.slice(at, 1)?[0] as i8 as i32),
            0x2d => Value::I32(self.slice(at, 1)?[0] as i32),
            0x2e => Value::I32(i16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap()) as i32),
            0x2f => Value::I32(u16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap()) as i32),
            0x30 => Value::I64(self.slice(at, 1)?[0] as i8 as i64),
            0x31 => Value::I64(self.slice(at, 1)?[0] as i64),
            0x32 => Value::I64(i16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap()) as i64),
            0x33 => Value::I64(u16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap()) as i64),
            0x34 => Value::I64(i32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap()) as i64),
            0x35 => Value::I64(u32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap()) as i64),
            _ => return malformed("load opcode"),
        };
        Ok(v)
    }

    fn store(&mut self, op: u8, at: u64, v: Value) -> Res<()> {
        let bytes: Vec<u8> = match op {
            0x36 => v.as_i32()?.to_le_bytes().to_vec(),
            0x37 => v.as_i64()?.to_le_bytes().to_vec(),
            0x38 => v.as_f32()?.to_le_bytes().to_vec(),
            0x39 => v.as_f64()?.to_le_bytes().to_vec(),
            0x3a => vec![v.as_i32()? as u8],
            0x3b => (v.as_i32()? as u16).to_le_bytes().to_vec(),
            0x3c => vec![v.as_i64()? as u8],
            0x3d => (v.as_i64()? as u16).to_le_bytes().to_vec(),
            0x3e => (v.as_i64()? as u32).to_le_bytes().to_vec(),
            _ => return malformed("store opcode"),
        };
        let end = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::Trap("address overflow".into()))?;
        if end > self.mem.len() as u64 {
            return Err(Error::Trap(format!(
                "out of bounds memory access at {at} ({} bytes)",
                bytes.len()
            )));
        }
        self.mem[at as usize..at as usize + bytes.len()].copy_from_slice(&bytes);
        Ok(())
    }

    /// The `0xfd` family: fixed-width SIMD (§11.6's deterministic subset).
    ///
    /// One line per opcode against the specification's table, which is the only
    /// way ~230 instructions stay checkable by reading. Relaxed-SIMD lives above
    /// `0xff` in the same prefix and is *forbidden* rather than unimplemented,
    /// because its results are permitted to differ between hosts and §11.6's
    /// whole requirement is determinism.
    fn simd(&mut self, sub: u64, r: &mut Reader<'_>, stack: &mut Vec<Value>) -> Res<()> {
        // Memory operands, shared by the load and store forms.
        macro_rules! addr {
            () => {{
                let _align = r.u()?;
                let offset = r.u()?;
                let base = pop(stack)?.as_i32()? as u32 as u64;
                base.checked_add(offset)
                    .ok_or_else(|| Error::Trap("address overflow".into()))?
            }};
        }
        // A load whose address is popped *before* the vector operand, which is
        // the order `load_lane` and `store_lane` take their operands in.
        macro_rules! lane_addr {
            ($v:ident) => {{
                let _align = r.u()?;
                let offset = r.u()?;
                let lane = r.byte()? as usize;
                let $v = pop(stack)?.as_v128()?;
                let base = pop(stack)?.as_i32()? as u32 as u64;
                let at = base
                    .checked_add(offset)
                    .ok_or_else(|| Error::Trap("address overflow".into()))?;
                (at, lane, $v)
            }};
        }
        macro_rules! un {
            ($f:expr) => {{
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::V128($f(a)));
            }};
        }
        macro_rules! bin {
            ($f:expr) => {{
                let b = pop(stack)?.as_v128()?;
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::V128($f(a, b)));
            }};
        }
        // A shift takes a vector and an i32 count; the count is taken modulo the
        // lane width, which is why each arm masks it rather than trapping.
        macro_rules! shift {
            ($width:expr, $f:expr) => {{
                let n = pop(stack)?.as_i32()? as u32 % $width;
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::V128($f(a, n)));
            }};
        }
        macro_rules! test {
            ($f:expr) => {{
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::I32(i32::from($f(a))));
            }};
        }

        match sub {
            // -- loads and stores ------------------------------------------
            0x00 => {
                let at = addr!();
                let b = self.slice(at, 16)?;
                stack.push(Value::V128(u128::from_le_bytes(b.try_into().unwrap())));
            }
            0x01..=0x06 => {
                // Widening loads: eight bytes, four halves or two words, each
                // extended to the next lane width up.
                let at = addr!();
                let b: [u8; 8] = self.slice(at, 8)?.try_into().unwrap();
                let v = match sub {
                    0x01 => of_u16s(std::array::from_fn(|i| b[i] as i8 as i16 as u16)),
                    0x02 => of_u16s(std::array::from_fn(|i| b[i] as u16)),
                    0x03 => of_u32s(std::array::from_fn(|i| {
                        i16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as i32 as u32
                    })),
                    0x04 => of_u32s(std::array::from_fn(|i| {
                        u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32
                    })),
                    0x05 => of_u64s(std::array::from_fn(|i| {
                        i32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()) as i64 as u64
                    })),
                    _ => of_u64s(std::array::from_fn(|i| {
                        u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()) as u64
                    })),
                };
                stack.push(Value::V128(v));
            }
            0x07 => {
                let at = addr!();
                let x = self.slice(at, 1)?[0];
                stack.push(Value::V128(of_u8s([x; 16])));
            }
            0x08 => {
                let at = addr!();
                let x = u16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap());
                stack.push(Value::V128(of_u16s([x; 8])));
            }
            0x09 => {
                let at = addr!();
                let x = u32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap());
                stack.push(Value::V128(of_u32s([x; 4])));
            }
            0x0a => {
                let at = addr!();
                let x = u64::from_le_bytes(self.slice(at, 8)?.try_into().unwrap());
                stack.push(Value::V128(of_u64s([x; 2])));
            }
            0x0b => {
                let _align = r.u()?;
                let offset = r.u()?;
                let v = pop(stack)?.as_v128()?;
                let base = pop(stack)?.as_i32()? as u32 as u64;
                let at = base
                    .checked_add(offset)
                    .ok_or_else(|| Error::Trap("address overflow".into()))?;
                self.store_bytes(at, &v.to_le_bytes())?;
            }
            0x0c => {
                let mut b = [0u8; 16];
                for x in b.iter_mut() {
                    *x = r.byte()?;
                }
                stack.push(Value::V128(u128::from_le_bytes(b)));
            }
            0x0d => {
                // i8x16.shuffle: sixteen immediate lane indices into the pair.
                let mut idx = [0u8; 16];
                for x in idx.iter_mut() {
                    *x = r.byte()?;
                    if *x > 31 {
                        return malformed("a shuffle lane index above 31");
                    }
                }
                let b = pop(stack)?.as_v128()?;
                let a = pop(stack)?.as_v128()?;
                let (la, lb) = (u8s(a), u8s(b));
                stack.push(Value::V128(of_u8s(std::array::from_fn(|i| {
                    let k = idx[i] as usize;
                    if k < 16 {
                        la[k]
                    } else {
                        lb[k - 16]
                    }
                }))));
            }
            0x0e => {
                // i8x16.swizzle: a dynamic gather; an index past the end is 0.
                let b = pop(stack)?.as_v128()?;
                let a = pop(stack)?.as_v128()?;
                let (la, lb) = (u8s(a), u8s(b));
                stack.push(Value::V128(of_u8s(std::array::from_fn(|i| {
                    let k = lb[i] as usize;
                    if k < 16 {
                        la[k]
                    } else {
                        0
                    }
                }))));
            }
            // -- splats ----------------------------------------------------
            0x0f => {
                let x = pop(stack)?.as_i32()? as u8;
                stack.push(Value::V128(of_u8s([x; 16])));
            }
            0x10 => {
                let x = pop(stack)?.as_i32()? as u16;
                stack.push(Value::V128(of_u16s([x; 8])));
            }
            0x11 => {
                let x = pop(stack)?.as_i32()? as u32;
                stack.push(Value::V128(of_u32s([x; 4])));
            }
            0x12 => {
                let x = pop(stack)?.as_i64()? as u64;
                stack.push(Value::V128(of_u64s([x; 2])));
            }
            0x13 => {
                let x = pop(stack)?.as_f32()?;
                stack.push(Value::V128(of_f32s([x; 4])));
            }
            0x14 => {
                let x = pop(stack)?.as_f64()?;
                stack.push(Value::V128(of_f64s([x; 2])));
            }
            // -- lane access -----------------------------------------------
            0x15 | 0x16 | 0x18 | 0x19 | 0x1b | 0x1d | 0x1f | 0x21 => {
                let lane = r.byte()? as usize;
                let a = pop(stack)?.as_v128()?;
                let v = match sub {
                    0x15 => Value::I32(*at_lane(&u8s(a), lane)? as i8 as i32),
                    0x16 => Value::I32(*at_lane(&u8s(a), lane)? as i32),
                    0x18 => Value::I32(*at_lane(&u16s(a), lane)? as i16 as i32),
                    0x19 => Value::I32(*at_lane(&u16s(a), lane)? as i32),
                    0x1b => Value::I32(*at_lane(&u32s(a), lane)? as i32),
                    0x1d => Value::I64(*at_lane(&u64s(a), lane)? as i64),
                    0x1f => Value::F32(*at_lane(&f32s(a), lane)?),
                    _ => Value::F64(*at_lane(&f64s(a), lane)?),
                };
                stack.push(v);
            }
            0x17 | 0x1a | 0x1c | 0x1e | 0x20 | 0x22 => {
                let lane = r.byte()? as usize;
                let x = pop(stack)?;
                let a = pop(stack)?.as_v128()?;
                let v = match sub {
                    0x17 => {
                        let mut l = u8s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_i32()? as u8;
                        of_u8s(l)
                    }
                    0x1a => {
                        let mut l = u16s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_i32()? as u16;
                        of_u16s(l)
                    }
                    0x1c => {
                        let mut l = u32s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_i32()? as u32;
                        of_u32s(l)
                    }
                    0x1e => {
                        let mut l = u64s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_i64()? as u64;
                        of_u64s(l)
                    }
                    0x20 => {
                        let mut l = f32s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_f32()?;
                        of_f32s(l)
                    }
                    _ => {
                        let mut l = f64s(a);
                        *at_lane_mut(&mut l, lane)? = x.as_f64()?;
                        of_f64s(l)
                    }
                };
                stack.push(Value::V128(v));
            }
            // -- comparisons -----------------------------------------------
            0x23 => bin!(|a, b| mask8(a, b, |x, y| x == y)),
            0x24 => bin!(|a, b| mask8(a, b, |x, y| x != y)),
            0x25 => bin!(|a, b| mask8(a, b, |x, y| (x as i8) < (y as i8))),
            0x26 => bin!(|a, b| mask8(a, b, |x, y| x < y)),
            0x27 => bin!(|a, b| mask8(a, b, |x, y| (x as i8) > (y as i8))),
            0x28 => bin!(|a, b| mask8(a, b, |x, y| x > y)),
            0x29 => bin!(|a, b| mask8(a, b, |x, y| (x as i8) <= (y as i8))),
            0x2a => bin!(|a, b| mask8(a, b, |x, y| x <= y)),
            0x2b => bin!(|a, b| mask8(a, b, |x, y| (x as i8) >= (y as i8))),
            0x2c => bin!(|a, b| mask8(a, b, |x, y| x >= y)),
            0x2d => bin!(|a, b| mask16(a, b, |x, y| x == y)),
            0x2e => bin!(|a, b| mask16(a, b, |x, y| x != y)),
            0x2f => bin!(|a, b| mask16(a, b, |x, y| (x as i16) < (y as i16))),
            0x30 => bin!(|a, b| mask16(a, b, |x, y| x < y)),
            0x31 => bin!(|a, b| mask16(a, b, |x, y| (x as i16) > (y as i16))),
            0x32 => bin!(|a, b| mask16(a, b, |x, y| x > y)),
            0x33 => bin!(|a, b| mask16(a, b, |x, y| (x as i16) <= (y as i16))),
            0x34 => bin!(|a, b| mask16(a, b, |x, y| x <= y)),
            0x35 => bin!(|a, b| mask16(a, b, |x, y| (x as i16) >= (y as i16))),
            0x36 => bin!(|a, b| mask16(a, b, |x, y| x >= y)),
            0x37 => bin!(|a, b| mask32(a, b, |x, y| x == y)),
            0x38 => bin!(|a, b| mask32(a, b, |x, y| x != y)),
            0x39 => bin!(|a, b| mask32(a, b, |x, y| (x as i32) < (y as i32))),
            0x3a => bin!(|a, b| mask32(a, b, |x, y| x < y)),
            0x3b => bin!(|a, b| mask32(a, b, |x, y| (x as i32) > (y as i32))),
            0x3c => bin!(|a, b| mask32(a, b, |x, y| x > y)),
            0x3d => bin!(|a, b| mask32(a, b, |x, y| (x as i32) <= (y as i32))),
            0x3e => bin!(|a, b| mask32(a, b, |x, y| x <= y)),
            0x3f => bin!(|a, b| mask32(a, b, |x, y| (x as i32) >= (y as i32))),
            0x40 => bin!(|a, b| mask32(a, b, |x, y| x >= y)),
            0x41 => bin!(|a, b| fmask32(a, b, |x, y| x == y)),
            0x42 => bin!(|a, b| fmask32(a, b, |x, y| x != y)),
            0x43 => bin!(|a, b| fmask32(a, b, |x, y| x < y)),
            0x44 => bin!(|a, b| fmask32(a, b, |x, y| x > y)),
            0x45 => bin!(|a, b| fmask32(a, b, |x, y| x <= y)),
            0x46 => bin!(|a, b| fmask32(a, b, |x, y| x >= y)),
            0x47 => bin!(|a, b| fmask64(a, b, |x, y| x == y)),
            0x48 => bin!(|a, b| fmask64(a, b, |x, y| x != y)),
            0x49 => bin!(|a, b| fmask64(a, b, |x, y| x < y)),
            0x4a => bin!(|a, b| fmask64(a, b, |x, y| x > y)),
            0x4b => bin!(|a, b| fmask64(a, b, |x, y| x <= y)),
            0x4c => bin!(|a, b| fmask64(a, b, |x, y| x >= y)),
            // -- bitwise ---------------------------------------------------
            0x4d => un!(|a: u128| !a),
            0x4e => bin!(|a: u128, b: u128| a & b),
            0x4f => bin!(|a: u128, b: u128| a & !b),
            0x50 => bin!(|a: u128, b: u128| a | b),
            0x51 => bin!(|a: u128, b: u128| a ^ b),
            0x52 => {
                let m = pop(stack)?.as_v128()?;
                let b = pop(stack)?.as_v128()?;
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::V128((a & m) | (b & !m)));
            }
            0x53 => test!(|a: u128| a != 0),
            // -- lane loads and stores -------------------------------------
            0x54 => {
                let (at, lane, v) = lane_addr!(v);
                let x = self.slice(at, 1)?[0];
                let mut l = u8s(v);
                *at_lane_mut(&mut l, lane)? = x;
                stack.push(Value::V128(of_u8s(l)));
            }
            0x55 => {
                let (at, lane, v) = lane_addr!(v);
                let x = u16::from_le_bytes(self.slice(at, 2)?.try_into().unwrap());
                let mut l = u16s(v);
                *at_lane_mut(&mut l, lane)? = x;
                stack.push(Value::V128(of_u16s(l)));
            }
            0x56 => {
                let (at, lane, v) = lane_addr!(v);
                let x = u32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap());
                let mut l = u32s(v);
                *at_lane_mut(&mut l, lane)? = x;
                stack.push(Value::V128(of_u32s(l)));
            }
            0x57 => {
                let (at, lane, v) = lane_addr!(v);
                let x = u64::from_le_bytes(self.slice(at, 8)?.try_into().unwrap());
                let mut l = u64s(v);
                *at_lane_mut(&mut l, lane)? = x;
                stack.push(Value::V128(of_u64s(l)));
            }
            0x58 => {
                let (at, lane, v) = lane_addr!(v);
                let x = *at_lane(&u8s(v), lane)?;
                self.store_bytes(at, &[x])?;
            }
            0x59 => {
                let (at, lane, v) = lane_addr!(v);
                let x = *at_lane(&u16s(v), lane)?;
                self.store_bytes(at, &x.to_le_bytes())?;
            }
            0x5a => {
                let (at, lane, v) = lane_addr!(v);
                let x = *at_lane(&u32s(v), lane)?;
                self.store_bytes(at, &x.to_le_bytes())?;
            }
            0x5b => {
                let (at, lane, v) = lane_addr!(v);
                let x = *at_lane(&u64s(v), lane)?;
                self.store_bytes(at, &x.to_le_bytes())?;
            }
            0x5c => {
                let at = addr!();
                let x = u32::from_le_bytes(self.slice(at, 4)?.try_into().unwrap());
                stack.push(Value::V128(x as u128));
            }
            0x5d => {
                let at = addr!();
                let x = u64::from_le_bytes(self.slice(at, 8)?.try_into().unwrap());
                stack.push(Value::V128(x as u128));
            }
            // -- float width changes ---------------------------------------
            0x5e => un!(|a: u128| {
                let l = f64s(a);
                of_f32s([canon32(l[0] as f32), canon32(l[1] as f32), 0.0, 0.0])
            }),
            0x5f => un!(|a: u128| {
                let l = f32s(a);
                of_f64s([canon64(l[0] as f64), canon64(l[1] as f64)])
            }),
            // -- i8x16 -----------------------------------------------------
            0x60 => un!(|a| map8i(a, |x| x.wrapping_abs())),
            0x61 => un!(|a| map8i(a, |x| x.wrapping_neg())),
            0x62 => un!(|a| map8(a, |x| x.count_ones() as u8)),
            0x63 => test!(|a| u8s(a).iter().all(|x| *x != 0)),
            0x64 => {
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::I32(bitmask(u8s(a).iter().map(|x| (*x as i8) < 0))));
            }
            0x65 => bin!(narrow8_s),
            0x66 => bin!(narrow8_u),
            0x67 => un!(|a| mapf32(a, f32::ceil)),
            0x68 => un!(|a| mapf32(a, f32::floor)),
            0x69 => un!(|a| mapf32(a, f32::trunc)),
            0x6a => un!(|a| mapf32(a, nearest32)),
            0x6b => shift!(8, |a, n| map8(a, |x| x.wrapping_shl(n))),
            0x6c => shift!(8, |a, n| map8i(a, |x| x.wrapping_shr(n))),
            0x6d => shift!(8, |a, n| map8(a, |x| x.wrapping_shr(n))),
            0x6e => bin!(|a, b| zip8(a, b, u8::wrapping_add)),
            0x6f => bin!(|a, b| zip8i(a, b, i8::saturating_add)),
            0x70 => bin!(|a, b| zip8(a, b, u8::saturating_add)),
            0x71 => bin!(|a, b| zip8(a, b, u8::wrapping_sub)),
            0x72 => bin!(|a, b| zip8i(a, b, i8::saturating_sub)),
            0x73 => bin!(|a, b| zip8(a, b, u8::saturating_sub)),
            0x74 => un!(|a| mapf64(a, f64::ceil)),
            0x75 => un!(|a| mapf64(a, f64::floor)),
            0x76 => bin!(|a, b| zip8i(a, b, i8::min)),
            0x77 => bin!(|a, b| zip8(a, b, u8::min)),
            0x78 => bin!(|a, b| zip8i(a, b, i8::max)),
            0x79 => bin!(|a, b| zip8(a, b, u8::max)),
            0x7a => un!(|a| mapf64(a, f64::trunc)),
            0x7b => bin!(|a, b| zip8(a, b, |x, y| (x as u16 + y as u16).div_ceil(2) as u8)),
            0x7c => un!(|a| {
                let l = u8s(a);
                of_u16s(std::array::from_fn(|i| {
                    (l[2 * i] as i8 as i16).wrapping_add(l[2 * i + 1] as i8 as i16) as u16
                }))
            }),
            0x7d => un!(|a| {
                let l = u8s(a);
                of_u16s(std::array::from_fn(|i| {
                    l[2 * i] as u16 + l[2 * i + 1] as u16
                }))
            }),
            0x7e => un!(|a| {
                let l = u16s(a);
                of_u32s(std::array::from_fn(|i| {
                    (l[2 * i] as i16 as i32).wrapping_add(l[2 * i + 1] as i16 as i32) as u32
                }))
            }),
            0x7f => un!(|a| {
                let l = u16s(a);
                of_u32s(std::array::from_fn(|i| {
                    l[2 * i] as u32 + l[2 * i + 1] as u32
                }))
            }),
            // -- i16x8 -----------------------------------------------------
            0x80 => un!(|a| map16i(a, |x| x.wrapping_abs())),
            0x81 => un!(|a| map16i(a, |x| x.wrapping_neg())),
            0x82 => bin!(|a, b| zip16i(a, b, |x, y| {
                // Rounding Q15 multiply: (x*y + 2^14) >> 15, saturated.
                (((x as i32 * y as i32) + (1 << 14)) >> 15).clamp(-32768, 32767) as i16
            })),
            0x83 => test!(|a| u16s(a).iter().all(|x| *x != 0)),
            0x84 => {
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::I32(bitmask(u16s(a).iter().map(|x| (*x as i16) < 0))));
            }
            0x85 => bin!(narrow16_s),
            0x86 => bin!(narrow16_u),
            0x87 => un!(|a| of_u16s(half8(a, false).map(|x| x as i8 as i16 as u16))),
            0x88 => un!(|a| of_u16s(half8(a, true).map(|x| x as i8 as i16 as u16))),
            0x89 => un!(|a| of_u16s(half8(a, false).map(|x| x as u16))),
            0x8a => un!(|a| of_u16s(half8(a, true).map(|x| x as u16))),
            0x8b => shift!(16, |a, n| map16(a, |x| x.wrapping_shl(n))),
            0x8c => shift!(16, |a, n| map16i(a, |x| x.wrapping_shr(n))),
            0x8d => shift!(16, |a, n| map16(a, |x| x.wrapping_shr(n))),
            0x8e => bin!(|a, b| zip16(a, b, u16::wrapping_add)),
            0x8f => bin!(|a, b| zip16i(a, b, i16::saturating_add)),
            0x90 => bin!(|a, b| zip16(a, b, u16::saturating_add)),
            0x91 => bin!(|a, b| zip16(a, b, u16::wrapping_sub)),
            0x92 => bin!(|a, b| zip16i(a, b, i16::saturating_sub)),
            0x93 => bin!(|a, b| zip16(a, b, u16::saturating_sub)),
            0x94 => un!(|a| mapf64(a, nearest64)),
            0x95 => bin!(|a, b| zip16(a, b, u16::wrapping_mul)),
            0x96 => bin!(|a, b| zip16i(a, b, i16::min)),
            0x97 => bin!(|a, b| zip16(a, b, u16::min)),
            0x98 => bin!(|a, b| zip16i(a, b, i16::max)),
            0x99 => bin!(|a, b| zip16(a, b, u16::max)),
            0x9b => bin!(|a, b| zip16(a, b, |x, y| (x as u32 + y as u32).div_ceil(2) as u16)),
            0x9c => bin!(|a, b| extmul8(a, b, false, true)),
            0x9d => bin!(|a, b| extmul8(a, b, true, true)),
            0x9e => bin!(|a, b| extmul8(a, b, false, false)),
            0x9f => bin!(|a, b| extmul8(a, b, true, false)),
            // -- i32x4 -----------------------------------------------------
            0xa0 => un!(|a| map32i(a, |x| x.wrapping_abs())),
            0xa1 => un!(|a| map32i(a, |x| x.wrapping_neg())),
            0xa3 => test!(|a| u32s(a).iter().all(|x| *x != 0)),
            0xa4 => {
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::I32(bitmask(u32s(a).iter().map(|x| (*x as i32) < 0))));
            }
            0xa7 => un!(|a| of_u32s(half16(a, false).map(|x| x as i16 as i32 as u32))),
            0xa8 => un!(|a| of_u32s(half16(a, true).map(|x| x as i16 as i32 as u32))),
            0xa9 => un!(|a| of_u32s(half16(a, false).map(|x| x as u32))),
            0xaa => un!(|a| of_u32s(half16(a, true).map(|x| x as u32))),
            0xab => shift!(32, |a, n| map32(a, |x| x.wrapping_shl(n))),
            0xac => shift!(32, |a, n| map32i(a, |x| x.wrapping_shr(n))),
            0xad => shift!(32, |a, n| map32(a, |x| x.wrapping_shr(n))),
            0xae => bin!(|a, b| zip32(a, b, u32::wrapping_add)),
            0xb1 => bin!(|a, b| zip32(a, b, u32::wrapping_sub)),
            0xb5 => bin!(|a, b| zip32(a, b, u32::wrapping_mul)),
            0xb6 => bin!(|a, b| zip32i(a, b, i32::min)),
            0xb7 => bin!(|a, b| zip32(a, b, u32::min)),
            0xb8 => bin!(|a, b| zip32i(a, b, i32::max)),
            0xb9 => bin!(|a, b| zip32(a, b, u32::max)),
            0xba => bin!(|a, b| {
                // i32x4.dot_i16x8_s: adjacent pairs of products summed.
                let (x, y) = (u16s(a), u16s(b));
                of_u32s(std::array::from_fn(|i| {
                    let p0 = x[2 * i] as i16 as i32 * y[2 * i] as i16 as i32;
                    let p1 = x[2 * i + 1] as i16 as i32 * y[2 * i + 1] as i16 as i32;
                    p0.wrapping_add(p1) as u32
                }))
            }),
            0xbc => bin!(|a, b| extmul16(a, b, false, true)),
            0xbd => bin!(|a, b| extmul16(a, b, true, true)),
            0xbe => bin!(|a, b| extmul16(a, b, false, false)),
            0xbf => bin!(|a, b| extmul16(a, b, true, false)),
            // -- i64x2 -----------------------------------------------------
            0xc0 => un!(|a| map64i(a, |x| x.wrapping_abs())),
            0xc1 => un!(|a| map64i(a, |x| x.wrapping_neg())),
            0xc3 => test!(|a| u64s(a).iter().all(|x| *x != 0)),
            0xc4 => {
                let a = pop(stack)?.as_v128()?;
                stack.push(Value::I32(bitmask(u64s(a).iter().map(|x| (*x as i64) < 0))));
            }
            0xc7 => un!(|a| of_u64s(half32(a, false).map(|x| x as i32 as i64 as u64))),
            0xc8 => un!(|a| of_u64s(half32(a, true).map(|x| x as i32 as i64 as u64))),
            0xc9 => un!(|a| of_u64s(half32(a, false).map(|x| x as u64))),
            0xca => un!(|a| of_u64s(half32(a, true).map(|x| x as u64))),
            0xcb => shift!(64, |a, n| of_u64s(u64s(a).map(|x| x.wrapping_shl(n)))),
            0xcc => shift!(64, |a, n| map64i(a, |x| x.wrapping_shr(n))),
            0xcd => shift!(64, |a, n| of_u64s(u64s(a).map(|x| x.wrapping_shr(n)))),
            0xce => bin!(|a, b| zip64(a, b, u64::wrapping_add)),
            0xd1 => bin!(|a, b| zip64(a, b, u64::wrapping_sub)),
            0xd5 => bin!(|a, b| zip64(a, b, u64::wrapping_mul)),
            0xd6 => bin!(|a, b| mask64(a, b, |x, y| x == y)),
            0xd7 => bin!(|a, b| mask64(a, b, |x, y| x != y)),
            0xd8 => bin!(|a, b| mask64(a, b, |x, y| (x as i64) < (y as i64))),
            0xd9 => bin!(|a, b| mask64(a, b, |x, y| (x as i64) > (y as i64))),
            0xda => bin!(|a, b| mask64(a, b, |x, y| (x as i64) <= (y as i64))),
            0xdb => bin!(|a, b| mask64(a, b, |x, y| (x as i64) >= (y as i64))),
            0xdc => bin!(|a, b| extmul32(a, b, false, true)),
            0xdd => bin!(|a, b| extmul32(a, b, true, true)),
            0xde => bin!(|a, b| extmul32(a, b, false, false)),
            0xdf => bin!(|a, b| extmul32(a, b, true, false)),
            // -- f32x4 -----------------------------------------------------
            // `abs` and `neg` are bit operations, specified exactly even on a
            // NaN: they clear and flip the sign and touch nothing else. Routing
            // them through the float helpers would canonicalise the payload and
            // lose the sign, which is a real difference a real engine shows.
            0xe0 => un!(|a| map32(a, |x| x & 0x7fff_ffff)),
            0xe1 => un!(|a| map32(a, |x| x ^ 0x8000_0000)),
            0xe3 => un!(|a| mapf32(a, f32::sqrt)),
            0xe4 => bin!(|a, b| zipf32(a, b, |x, y| x + y)),
            0xe5 => bin!(|a, b| zipf32(a, b, |x, y| x - y)),
            0xe6 => bin!(|a, b| zipf32(a, b, |x, y| x * y)),
            0xe7 => bin!(|a, b| zipf32(a, b, |x, y| x / y)),
            0xe8 => bin!(|a, b| zipf32(a, b, fmin32)),
            0xe9 => bin!(|a, b| zipf32(a, b, fmax32)),
            0xea => bin!(|a, b| zipf32(a, b, pmin32)),
            0xeb => bin!(|a, b| zipf32(a, b, pmax32)),
            // -- f64x2 -----------------------------------------------------
            0xec => un!(|a| of_u64s(u64s(a).map(|x| x & 0x7fff_ffff_ffff_ffff))),
            0xed => un!(|a| of_u64s(u64s(a).map(|x| x ^ 0x8000_0000_0000_0000))),
            0xef => un!(|a| mapf64(a, f64::sqrt)),
            0xf0 => bin!(|a, b| zipf64(a, b, |x, y| x + y)),
            0xf1 => bin!(|a, b| zipf64(a, b, |x, y| x - y)),
            0xf2 => bin!(|a, b| zipf64(a, b, |x, y| x * y)),
            0xf3 => bin!(|a, b| zipf64(a, b, |x, y| x / y)),
            0xf4 => bin!(|a, b| zipf64(a, b, fmin64)),
            0xf5 => bin!(|a, b| zipf64(a, b, fmax64)),
            0xf6 => bin!(|a, b| zipf64(a, b, pmin64)),
            0xf7 => bin!(|a, b| zipf64(a, b, pmax64)),
            // -- conversions -----------------------------------------------
            0xf8 => un!(|a| of_u32s(f32s(a).map(|x| sat32_s(x) as u32))),
            0xf9 => un!(|a| of_u32s(f32s(a).map(sat32_u))),
            0xfa => un!(|a| of_f32s(u32s(a).map(|x| canon32(x as i32 as f32)))),
            0xfb => un!(|a| of_f32s(u32s(a).map(|x| canon32(x as f32)))),
            0xfc => un!(|a| {
                let l = f64s(a);
                of_u32s([sat64_s(l[0]) as u32, sat64_s(l[1]) as u32, 0, 0])
            }),
            0xfd => un!(|a| {
                let l = f64s(a);
                of_u32s([sat64_u(l[0]), sat64_u(l[1]), 0, 0])
            }),
            0xfe => un!(|a| {
                let l = u32s(a);
                of_f64s([canon64(l[0] as i32 as f64), canon64(l[1] as i32 as f64)])
            }),
            0xff => un!(|a| {
                let l = u32s(a);
                of_f64s([canon64(l[0] as f64), canon64(l[1] as f64)])
            }),
            // Relaxed-SIMD is a *different* proposal in the same prefix, and
            // §11.6 forbids it by name: its results may differ between hosts,
            // which is the one thing a deterministic plugin may not do.
            0x100..=0x15f => {
                return Err(Error::Forbidden(format!(
                    "relaxed-SIMD instruction {sub:#x} (nondeterministic)"
                )))
            }
            other => return malformed(format!("SIMD opcode {other:#x}")),
        }
        Ok(())
    }

    /// The `0xfc` family: saturating truncation and bulk memory.
    fn misc(&mut self, sub: u64, r: &mut Reader<'_>, stack: &mut Vec<Value>) -> Res<()> {
        match sub {
            0 | 1 => {
                let a = pop(stack)?.as_f32()? as f64;
                stack.push(Value::I32(sat_i32(a, sub == 1)));
            }
            2 | 3 => {
                let a = pop(stack)?.as_f64()?;
                stack.push(Value::I32(sat_i32(a, sub == 3)));
            }
            4 | 5 => {
                let a = pop(stack)?.as_f32()? as f64;
                stack.push(Value::I64(sat_i64(a, sub == 5)));
            }
            6 | 7 => {
                let a = pop(stack)?.as_f64()?;
                stack.push(Value::I64(sat_i64(a, sub == 7)));
            }
            // memory.init
            8 => {
                let seg = r.u32()? as usize;
                let _mem = r.byte()?;
                let n = pop(stack)?.as_i32()? as usize;
                let src = pop(stack)?.as_i32()? as usize;
                let dst = pop(stack)?.as_i32()? as usize;
                let data = self
                    .m
                    .data
                    .get(seg)
                    .ok_or_else(|| Error::Trap("memory.init: no such segment".into()))?
                    .1
                    .clone();
                if src + n > data.len() || dst + n > self.mem.len() {
                    return Err(Error::Trap("memory.init out of bounds".into()));
                }
                self.burn(n as u64 / 8 + 1)?;
                self.mem[dst..dst + n].copy_from_slice(&data[src..src + n]);
            }
            9 => {
                let _seg = r.u32()?;
            }
            // memory.copy
            10 => {
                let _d = r.byte()?;
                let _s = r.byte()?;
                let n = pop(stack)?.as_i32()? as usize;
                let src = pop(stack)?.as_i32()? as usize;
                let dst = pop(stack)?.as_i32()? as usize;
                if src.checked_add(n).is_none_or(|e| e > self.mem.len())
                    || dst.checked_add(n).is_none_or(|e| e > self.mem.len())
                {
                    return Err(Error::Trap("memory.copy out of bounds".into()));
                }
                self.burn(n as u64 / 8 + 1)?;
                // Overlapping copies are defined; `copy_within` does the right
                // thing in both directions.
                self.mem.copy_within(src..src + n, dst);
            }
            // memory.fill
            11 => {
                let _mem = r.byte()?;
                let n = pop(stack)?.as_i32()? as usize;
                let byte = pop(stack)?.as_i32()? as u8;
                let dst = pop(stack)?.as_i32()? as usize;
                if dst.checked_add(n).is_none_or(|e| e > self.mem.len()) {
                    return Err(Error::Trap("memory.fill out of bounds".into()));
                }
                self.burn(n as u64 / 8 + 1)?;
                self.mem[dst..dst + n].fill(byte);
            }
            other => return Err(Error::Unsupported(format!("0xfc {other}"))),
        }
        Ok(())
    }
}

fn truncate_to(stack: &mut Vec<Value>, height: usize, arity: usize) -> Res<()> {
    if stack.len() < height + arity {
        // A block that produced fewer values than its type declares is a
        // validation error a static validator would have caught; here it is a
        // trap, which is the same refusal one step later.
        return Err(Error::Trap("a block did not produce its results".into()));
    }
    let results: Vec<Value> = stack.split_off(stack.len() - arity);
    stack.truncate(height);
    stack.extend(results);
    Ok(())
}

/// Takes a branch. Returns true when the branch leaves the function.
fn branch(
    stack: &mut Vec<Value>,
    labels: &mut Vec<Label>,
    r: &mut Reader<'_>,
    depth: usize,
    ty: &FuncType,
) -> Res<bool> {
    if depth >= labels.len() {
        // A branch past the outermost label returns from the function.
        let arity = ty.results.len();
        if stack.len() < arity {
            return Err(Error::Trap(
                "branch out of a function without results".into(),
            ));
        }
        return Ok(true);
    }
    for _ in 0..depth {
        labels.pop();
    }
    let l = labels
        .last()
        .ok_or_else(|| Error::Trap("branch to a label that does not exist".into()))?;
    truncate_to(stack, l.height, l.arity)?;
    r.at = l.target;
    if !l.is_loop {
        labels.pop();
    }
    Ok(false)
}

fn take_args(stack: &mut Vec<Value>, n: usize) -> Res<Vec<Value>> {
    if stack.len() < n {
        return Err(Error::Trap("not enough arguments on the stack".into()));
    }
    Ok(stack.split_off(stack.len() - n))
}

fn finish(mut stack: Vec<Value>, ty: &FuncType) -> Res<Vec<Value>> {
    let n = ty.results.len();
    if stack.len() < n {
        return Err(Error::Trap(format!(
            "a function returning {n} value(s) left {} on the stack",
            stack.len()
        )));
    }
    let out = stack.split_off(stack.len() - n);
    for (v, want) in out.iter().zip(&ty.results) {
        if v.ty() != *want {
            return Err(Error::Trap(format!(
                "a function returned {:?} where {want:?} was declared",
                v.ty()
            )));
        }
    }
    Ok(out)
}

fn round_even32(a: f32) -> f32 {
    let r = a.round();
    if (a - a.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - a.signum()
    } else {
        r
    }
}

fn round_even64(a: f64) -> f64 {
    let r = a.round();
    if (a - a.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - a.signum()
    } else {
        r
    }
}

/// WebAssembly's `min`/`max` are not Rust's: NaN propagates, and −0 is smaller
/// than +0. Determinism (§11.6) is exactly this kind of detail.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return canon32(f32::NAN);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b {
        a
    } else {
        b
    }
}

fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return canon32(f32::NAN);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_positive() { a } else { b };
    }
    if a > b {
        a
    } else {
        b
    }
}

fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return canon64(f64::NAN);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b {
        a
    } else {
        b
    }
}

fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return canon64(f64::NAN);
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_positive() { a } else { b };
    }
    if a > b {
        a
    } else {
        b
    }
}

fn trunc_i32(a: f64, unsigned: bool) -> Res<i32> {
    if a.is_nan() {
        return Err(Error::Trap("invalid conversion to integer".into()));
    }
    let t = a.trunc();
    if unsigned {
        if !(0.0..=u32::MAX as f64).contains(&t) {
            return Err(Error::Trap("integer overflow".into()));
        }
        Ok(t as u32 as i32)
    } else {
        if !(i32::MIN as f64..=i32::MAX as f64).contains(&t) {
            return Err(Error::Trap("integer overflow".into()));
        }
        Ok(t as i32)
    }
}

fn trunc_i64(a: f64, unsigned: bool) -> Res<i64> {
    if a.is_nan() {
        return Err(Error::Trap("invalid conversion to integer".into()));
    }
    let t = a.trunc();
    if unsigned {
        if !(0.0..18446744073709551616.0).contains(&t) {
            return Err(Error::Trap("integer overflow".into()));
        }
        Ok(t as u64 as i64)
    } else {
        if !(i64::MIN as f64..9223372036854775808.0).contains(&t) {
            return Err(Error::Trap("integer overflow".into()));
        }
        Ok(t as i64)
    }
}

fn sat_i32(a: f64, unsigned: bool) -> i32 {
    if a.is_nan() {
        return 0;
    }
    let t = a.trunc();
    if unsigned {
        t.clamp(0.0, u32::MAX as f64) as u32 as i32
    } else {
        t.clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }
}

fn sat_i64(a: f64, unsigned: bool) -> i64 {
    if a.is_nan() {
        return 0;
    }
    let t = a.trunc();
    if unsigned {
        if t <= 0.0 {
            0
        } else if t >= 18446744073709551615.0 {
            -1
        } else {
            t as u64 as i64
        }
    } else if t <= i64::MIN as f64 {
        i64::MIN
    } else if t >= 9223372036854775807.0 {
        i64::MAX
    } else {
        t as i64
    }
}

// ------------------------------------------------------------- SIMD lanes --

// The lane helpers. A `v128` is a `u128` with lane 0 in the least significant
// bits, so every one of these is a byte-order-free reinterpretation of the same
// integer and `v128.load`/`store` are plain little-endian moves.
//
// Each family is a get/put pair plus the two shapes every opcode takes: a
// lane-wise map and a lane-wise zip. Writing them once is what keeps the
// opcode table below one line per instruction, which is the only way a table
// this size stays checkable by eye against the specification.

fn u8s(v: u128) -> [u8; 16] {
    v.to_le_bytes()
}
fn of_u8s(l: [u8; 16]) -> u128 {
    u128::from_le_bytes(l)
}
fn u16s(v: u128) -> [u16; 8] {
    let b = v.to_le_bytes();
    std::array::from_fn(|i| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
}
fn of_u16s(l: [u16; 8]) -> u128 {
    let mut b = [0u8; 16];
    for (i, x) in l.iter().enumerate() {
        b[2 * i..2 * i + 2].copy_from_slice(&x.to_le_bytes());
    }
    u128::from_le_bytes(b)
}
fn u32s(v: u128) -> [u32; 4] {
    let b = v.to_le_bytes();
    std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}
fn of_u32s(l: [u32; 4]) -> u128 {
    let mut b = [0u8; 16];
    for (i, x) in l.iter().enumerate() {
        b[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
    u128::from_le_bytes(b)
}
fn u64s(v: u128) -> [u64; 2] {
    [v as u64, (v >> 64) as u64]
}
fn of_u64s(l: [u64; 2]) -> u128 {
    (l[0] as u128) | ((l[1] as u128) << 64)
}
fn f32s(v: u128) -> [f32; 4] {
    u32s(v).map(f32::from_bits)
}
fn of_f32s(l: [f32; 4]) -> u128 {
    of_u32s(l.map(|x| x.to_bits()))
}
fn f64s(v: u128) -> [f64; 2] {
    u64s(v).map(f64::from_bits)
}
fn of_f64s(l: [f64; 2]) -> u128 {
    of_u64s(l.map(|x| x.to_bits()))
}

fn map8(v: u128, f: impl Fn(u8) -> u8) -> u128 {
    of_u8s(u8s(v).map(f))
}
fn map8i(v: u128, f: impl Fn(i8) -> i8) -> u128 {
    of_u8s(u8s(v).map(|x| f(x as i8) as u8))
}
fn zip8(a: u128, b: u128, f: impl Fn(u8, u8) -> u8) -> u128 {
    let (x, y) = (u8s(a), u8s(b));
    of_u8s(std::array::from_fn(|i| f(x[i], y[i])))
}
fn zip8i(a: u128, b: u128, f: impl Fn(i8, i8) -> i8) -> u128 {
    zip8(a, b, |x, y| f(x as i8, y as i8) as u8)
}
fn map16(v: u128, f: impl Fn(u16) -> u16) -> u128 {
    of_u16s(u16s(v).map(f))
}
fn map16i(v: u128, f: impl Fn(i16) -> i16) -> u128 {
    of_u16s(u16s(v).map(|x| f(x as i16) as u16))
}
fn zip16(a: u128, b: u128, f: impl Fn(u16, u16) -> u16) -> u128 {
    let (x, y) = (u16s(a), u16s(b));
    of_u16s(std::array::from_fn(|i| f(x[i], y[i])))
}
fn zip16i(a: u128, b: u128, f: impl Fn(i16, i16) -> i16) -> u128 {
    zip16(a, b, |x, y| f(x as i16, y as i16) as u16)
}
fn map32(v: u128, f: impl Fn(u32) -> u32) -> u128 {
    of_u32s(u32s(v).map(f))
}
fn map32i(v: u128, f: impl Fn(i32) -> i32) -> u128 {
    of_u32s(u32s(v).map(|x| f(x as i32) as u32))
}
fn zip32(a: u128, b: u128, f: impl Fn(u32, u32) -> u32) -> u128 {
    let (x, y) = (u32s(a), u32s(b));
    of_u32s(std::array::from_fn(|i| f(x[i], y[i])))
}
fn zip32i(a: u128, b: u128, f: impl Fn(i32, i32) -> i32) -> u128 {
    zip32(a, b, |x, y| f(x as i32, y as i32) as u32)
}
fn map64i(v: u128, f: impl Fn(i64) -> i64) -> u128 {
    of_u64s(u64s(v).map(|x| f(x as i64) as u64))
}
fn zip64(a: u128, b: u128, f: impl Fn(u64, u64) -> u64) -> u128 {
    let (x, y) = (u64s(a), u64s(b));
    of_u64s(std::array::from_fn(|i| f(x[i], y[i])))
}
fn mapf32(v: u128, f: impl Fn(f32) -> f32) -> u128 {
    of_f32s(f32s(v).map(|x| canon32(f(x))))
}
fn zipf32(a: u128, b: u128, f: impl Fn(f32, f32) -> f32) -> u128 {
    let (x, y) = (f32s(a), f32s(b));
    of_f32s(std::array::from_fn(|i| canon32(f(x[i], y[i]))))
}
fn mapf64(v: u128, f: impl Fn(f64) -> f64) -> u128 {
    of_f64s(f64s(v).map(|x| canon64(f(x))))
}
fn zipf64(a: u128, b: u128, f: impl Fn(f64, f64) -> f64) -> u128 {
    let (x, y) = (f64s(a), f64s(b));
    of_f64s(std::array::from_fn(|i| canon64(f(x[i], y[i]))))
}

/// A lane-wise comparison result: all ones for true, all zeros for false, which
/// is what makes `v128.bitselect` compose with a comparison.
fn mask8(a: u128, b: u128, f: impl Fn(u8, u8) -> bool) -> u128 {
    zip8(a, b, |x, y| if f(x, y) { 0xff } else { 0 })
}
fn mask16(a: u128, b: u128, f: impl Fn(u16, u16) -> bool) -> u128 {
    zip16(a, b, |x, y| if f(x, y) { 0xffff } else { 0 })
}
fn mask32(a: u128, b: u128, f: impl Fn(u32, u32) -> bool) -> u128 {
    zip32(a, b, |x, y| if f(x, y) { 0xffff_ffff } else { 0 })
}
fn mask64(a: u128, b: u128, f: impl Fn(u64, u64) -> bool) -> u128 {
    zip64(a, b, |x, y| if f(x, y) { u64::MAX } else { 0 })
}

// `f32x4.min`/`max` share the scalar rules above: NaN propagates and −0 <
// +0, so `fmin32`/`fmax64` and friends are reused rather than restated.

/// `pmin`/`pmax`: "pseudo-minimum", defined as `y < x ? y : x`, which returns
/// the *second* operand when either is NaN and does not distinguish zeros.
fn pmin32(x: f32, y: f32) -> f32 {
    if y < x {
        y
    } else {
        x
    }
}
fn pmax32(x: f32, y: f32) -> f32 {
    if x < y {
        y
    } else {
        x
    }
}
fn pmin64(x: f64, y: f64) -> f64 {
    if y < x {
        y
    } else {
        x
    }
}
fn pmax64(x: f64, y: f64) -> f64 {
    if x < y {
        y
    } else {
        x
    }
}

/// `iNxM.nearest`: round half to even, which is `f32::round_ties_even`.
fn nearest32(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        x
    } else {
        x.round_ties_even()
    }
}
fn nearest64(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        x
    } else {
        x.round_ties_even()
    }
}

/// Saturating float-to-int, the `trunc_sat` rule: NaN is 0, and anything past
/// the range clamps rather than wrapping or trapping.
fn sat32_s(x: f32) -> i32 {
    if x.is_nan() {
        0
    } else {
        let t = x.trunc();
        if t <= i32::MIN as f32 {
            i32::MIN
        } else if t >= i32::MAX as f32 {
            i32::MAX
        } else {
            t as i32
        }
    }
}
fn sat32_u(x: f32) -> u32 {
    if x.is_nan() {
        0
    } else {
        let t = x.trunc();
        if t <= 0.0 {
            0
        } else if t >= u32::MAX as f32 {
            u32::MAX
        } else {
            t as u32
        }
    }
}
fn sat64_s(x: f64) -> i32 {
    if x.is_nan() {
        0
    } else {
        let t = x.trunc();
        if t <= i32::MIN as f64 {
            i32::MIN
        } else if t >= i32::MAX as f64 {
            i32::MAX
        } else {
            t as i32
        }
    }
}
fn sat64_u(x: f64) -> u32 {
    if x.is_nan() {
        0
    } else {
        let t = x.trunc();
        if t <= 0.0 {
            0
        } else if t >= u32::MAX as f64 {
            u32::MAX
        } else {
            t as u32
        }
    }
}

/// A lane count's worth of high or low halves, for the `extend`/`extmul` family.
fn half8(v: u128, high: bool) -> [u8; 8] {
    let l = u8s(v);
    std::array::from_fn(|i| l[i + if high { 8 } else { 0 }])
}
fn half16(v: u128, high: bool) -> [u16; 4] {
    let l = u16s(v);
    std::array::from_fn(|i| l[i + if high { 4 } else { 0 }])
}
fn half32(v: u128, high: bool) -> [u32; 2] {
    let l = u32s(v);
    std::array::from_fn(|i| l[i + if high { 2 } else { 0 }])
}

/// `i8x16.bitmask` and friends: the sign bit of each lane, packed into an i32.
fn bitmask(signs: impl Iterator<Item = bool>) -> i32 {
    let mut m = 0i32;
    for (i, s) in signs.enumerate() {
        if s {
            m |= 1 << i;
        }
    }
    m
}

fn fmask32(a: u128, b: u128, f: impl Fn(f32, f32) -> bool) -> u128 {
    let (x, y) = (f32s(a), f32s(b));
    of_u32s(std::array::from_fn(|i| {
        if f(x[i], y[i]) {
            0xffff_ffff
        } else {
            0
        }
    }))
}
fn fmask64(a: u128, b: u128, f: impl Fn(f64, f64) -> bool) -> u128 {
    let (x, y) = (f64s(a), f64s(b));
    of_u64s(std::array::from_fn(|i| {
        if f(x[i], y[i]) {
            u64::MAX
        } else {
            0
        }
    }))
}

/// A lane index from an immediate. Out of range is a malformed module rather
/// than a trap: the index is in the code, not in the data.
fn at_lane<T>(l: &[T], i: usize) -> Res<&T> {
    l.get(i)
        .ok_or_else(|| Error::Malformed(format!("lane index {i} past {} lanes", l.len())))
}
fn at_lane_mut<T>(l: &mut [T], i: usize) -> Res<&mut T> {
    let n = l.len();
    l.get_mut(i)
        .ok_or_else(|| Error::Malformed(format!("lane index {i} past {n} lanes")))
}

/// `i8x16.narrow_i16x8_*`: two vectors of eight 16-bit lanes saturated down to
/// one vector of sixteen 8-bit lanes, `a`'s lanes first.
fn narrow8_s(a: u128, b: u128) -> u128 {
    let (x, y) = (u16s(a), u16s(b));
    of_u8s(std::array::from_fn(|i| {
        let v = if i < 8 { x[i] } else { y[i - 8] } as i16;
        v.clamp(-128, 127) as i8 as u8
    }))
}
fn narrow8_u(a: u128, b: u128) -> u128 {
    let (x, y) = (u16s(a), u16s(b));
    of_u8s(std::array::from_fn(|i| {
        let v = if i < 8 { x[i] } else { y[i - 8] } as i16;
        v.clamp(0, 255) as u8
    }))
}
fn narrow16_s(a: u128, b: u128) -> u128 {
    let (x, y) = (u32s(a), u32s(b));
    of_u16s(std::array::from_fn(|i| {
        let v = if i < 4 { x[i] } else { y[i - 4] } as i32;
        v.clamp(-32768, 32767) as i16 as u16
    }))
}
fn narrow16_u(a: u128, b: u128) -> u128 {
    let (x, y) = (u32s(a), u32s(b));
    of_u16s(std::array::from_fn(|i| {
        let v = if i < 4 { x[i] } else { y[i - 4] } as i32;
        v.clamp(0, 65535) as u16
    }))
}

/// The `extmul` family: multiply one half of each operand, widening as it goes.
fn extmul8(a: u128, b: u128, high: bool, signed: bool) -> u128 {
    let (x, y) = (half8(a, high), half8(b, high));
    of_u16s(std::array::from_fn(|i| {
        if signed {
            ((x[i] as i8 as i16).wrapping_mul(y[i] as i8 as i16)) as u16
        } else {
            (x[i] as u16).wrapping_mul(y[i] as u16)
        }
    }))
}
fn extmul16(a: u128, b: u128, high: bool, signed: bool) -> u128 {
    let (x, y) = (half16(a, high), half16(b, high));
    of_u32s(std::array::from_fn(|i| {
        if signed {
            ((x[i] as i16 as i32).wrapping_mul(y[i] as i16 as i32)) as u32
        } else {
            (x[i] as u32).wrapping_mul(y[i] as u32)
        }
    }))
}
fn extmul32(a: u128, b: u128, high: bool, signed: bool) -> u128 {
    let (x, y) = (half32(a, high), half32(b, high));
    of_u64s(std::array::from_fn(|i| {
        if signed {
            ((x[i] as i32 as i64).wrapping_mul(y[i] as i32 as i64)) as u64
        } else {
            (x[i] as u64).wrapping_mul(y[i] as u64)
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal WebAssembly encoder, so the tests are readable and the module
    /// bytes are derived from source rather than pasted in.
    ///
    /// A test that ships an opaque `.wasm` blob tests whatever that blob happens
    /// to be; one that builds the module here tests the instruction the test is
    /// named after.
    #[derive(Default)]
    struct Build {
        types: Vec<FuncType>,
        imports: Vec<(&'static str, u32)>,
        funcs: Vec<(u32, Vec<ValType>, Vec<u8>)>,
        exports: Vec<(String, u8, u32)>,
        memory: Option<(u32, Option<u32>)>,
        globals: Vec<(ValType, bool, Vec<u8>)>,
        table: Vec<u32>,
        data: Vec<(u32, Vec<u8>)>,
    }

    fn leb(mut n: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    fn sleb(mut n: i64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (n & 0x7f) as u8;
            n >>= 7;
            let done = (n == 0 && byte & 0x40 == 0) || (n == -1 && byte & 0x40 != 0);
            out.push(if done { byte } else { byte | 0x80 });
            if done {
                return out;
            }
        }
    }

    fn vt(t: ValType) -> u8 {
        match t {
            ValType::I32 => 0x7f,
            ValType::I64 => 0x7e,
            ValType::F32 => 0x7d,
            ValType::F64 => 0x7c,
            ValType::V128 => 0x7b,
            ValType::FuncRef => 0x70,
        }
    }

    impl Build {
        fn ty(&mut self, params: &[ValType], results: &[ValType]) -> u32 {
            let t = FuncType {
                params: params.to_vec(),
                results: results.to_vec(),
            };
            match self.types.iter().position(|x| *x == t) {
                Some(i) => i as u32,
                None => {
                    self.types.push(t);
                    (self.types.len() - 1) as u32
                }
            }
        }

        fn import(&mut self, field: &'static str, params: &[ValType], results: &[ValType]) -> u32 {
            let t = self.ty(params, results);
            self.imports.push((field, t));
            (self.imports.len() - 1) as u32
        }

        fn func(
            &mut self,
            name: &str,
            params: &[ValType],
            results: &[ValType],
            locals: &[ValType],
            body: Vec<u8>,
        ) -> u32 {
            let t = self.ty(params, results);
            self.funcs.push((t, locals.to_vec(), body));
            let idx = (self.imports.len() + self.funcs.len() - 1) as u32;
            if !name.is_empty() {
                self.exports.push((name.to_string(), 0, idx));
            }
            idx
        }

        fn memory(&mut self, min: u32, max: Option<u32>) {
            self.memory = Some((min, max));
            self.exports.push(("memory".into(), 2, 0));
        }

        fn finish(&self) -> Vec<u8> {
            let mut out = b"\0asm\x01\0\0\0".to_vec();
            let section = |out: &mut Vec<u8>, id: u8, body: Vec<u8>| {
                if body.is_empty() {
                    return;
                }
                out.push(id);
                out.extend(leb(body.len() as u64));
                out.extend(body);
            };
            // 1: types
            let mut b = leb(self.types.len() as u64);
            for t in &self.types {
                b.push(0x60);
                b.extend(leb(t.params.len() as u64));
                b.extend(t.params.iter().map(|x| vt(*x)));
                b.extend(leb(t.results.len() as u64));
                b.extend(t.results.iter().map(|x| vt(*x)));
            }
            section(&mut out, 1, b);
            // 2: imports
            if !self.imports.is_empty() {
                let mut b = leb(self.imports.len() as u64);
                for (field, t) in &self.imports {
                    b.extend(leb(HOST_MODULE.len() as u64));
                    b.extend(HOST_MODULE.as_bytes());
                    b.extend(leb(field.len() as u64));
                    b.extend(field.as_bytes());
                    b.push(0x00);
                    b.extend(leb(*t as u64));
                }
                section(&mut out, 2, b);
            }
            // 3: functions
            let mut b = leb(self.funcs.len() as u64);
            for (t, _, _) in &self.funcs {
                b.extend(leb(*t as u64));
            }
            section(&mut out, 3, b);
            // 4: table
            if !self.table.is_empty() {
                let mut b = leb(1);
                b.push(0x70);
                b.push(0x00);
                b.extend(leb(self.table.len() as u64));
                section(&mut out, 4, b);
            }
            // 5: memory
            if let Some((min, max)) = self.memory {
                let mut b = leb(1);
                match max {
                    Some(m) => {
                        b.push(0x01);
                        b.extend(leb(min as u64));
                        b.extend(leb(m as u64));
                    }
                    None => {
                        b.push(0x00);
                        b.extend(leb(min as u64));
                    }
                }
                section(&mut out, 5, b);
            }
            // 6: globals
            if !self.globals.is_empty() {
                let mut b = leb(self.globals.len() as u64);
                for (t, mutable, init) in &self.globals {
                    b.push(vt(*t));
                    b.push(u8::from(*mutable));
                    b.extend(init.clone());
                    b.push(0x0b);
                }
                section(&mut out, 6, b);
            }
            // 7: exports
            let mut b = leb(self.exports.len() as u64);
            for (name, kind, idx) in &self.exports {
                b.extend(leb(name.len() as u64));
                b.extend(name.as_bytes());
                b.push(*kind);
                b.extend(leb(*idx as u64));
            }
            section(&mut out, 7, b);
            // 9: elements
            if !self.table.is_empty() {
                let mut b = leb(1);
                b.extend(leb(0));
                b.push(0x41);
                b.extend(sleb(0));
                b.push(0x0b);
                b.extend(leb(self.table.len() as u64));
                for f in &self.table {
                    b.extend(leb(*f as u64));
                }
                section(&mut out, 9, b);
            }
            // 10: code
            let mut b = leb(self.funcs.len() as u64);
            for (_, locals, body) in &self.funcs {
                let mut f = Vec::new();
                // Locals, one group per type for simplicity.
                f.extend(leb(locals.len() as u64));
                for l in locals {
                    f.extend(leb(1));
                    f.push(vt(*l));
                }
                f.extend(body.clone());
                f.push(0x0b);
                b.extend(leb(f.len() as u64));
                b.extend(f);
            }
            section(&mut out, 10, b);
            // 11: data
            if !self.data.is_empty() {
                let mut b = leb(self.data.len() as u64);
                for (off, bytes) in &self.data {
                    b.extend(leb(0));
                    b.push(0x41);
                    b.extend(sleb(*off as i64));
                    b.push(0x0b);
                    b.extend(leb(bytes.len() as u64));
                    b.extend(bytes.clone());
                }
                section(&mut out, 11, b);
            }
            out
        }
    }

    fn i32c(n: i32) -> Vec<u8> {
        let mut v = vec![0x41];
        v.extend(sleb(n as i64));
        v
    }

    fn run(bytes: &[u8], name: &str, args: &[Value]) -> Res<Vec<Value>> {
        let m = Module::load(bytes)?;
        let env = Env::default();
        let mut i = Instance::new(&m, &env, Limits::default())?;
        i.call(name, args)
    }

    #[test]
    fn arithmetic_and_locals() {
        let mut b = Build::default();
        // (a + b) * 2 - 1
        let mut body = vec![0x20, 0, 0x20, 1, 0x6a];
        body.extend(i32c(2));
        body.push(0x6c);
        body.extend(i32c(1));
        body.push(0x6b);
        b.func(
            "f",
            &[ValType::I32, ValType::I32],
            &[ValType::I32],
            &[],
            body,
        );
        let out = run(&b.finish(), "f", &[Value::I32(20), Value::I32(1)]).unwrap();
        assert_eq!(out, vec![Value::I32(41)]);
    }

    #[test]
    fn a_loop_that_sums() {
        // A real loop with a block/br_if, which is what a compiler emits.
        let mut b = Build::default();
        let mut body = Vec::new();
        // local 1 = accumulator, local 2 = counter
        body.extend([0x02, 0x40]); // block
        body.extend([0x03, 0x40]); // loop
        body.extend([0x20, 2, 0x20, 0, 0x4e]); // counter >= n?
        body.extend([0x0d, 1]); // br_if out of the block
        body.extend([0x20, 1, 0x20, 2, 0x6a, 0x21, 1]); // acc += counter
        body.extend([0x20, 2]);
        body.extend(i32c(1));
        body.extend([0x6a, 0x21, 2]); // counter += 1
        body.extend([0x0c, 0]); // br to the loop
        body.push(0x0b); // end loop
        body.push(0x0b); // end block
        body.extend([0x20, 1]);
        b.func(
            "sum",
            &[ValType::I32],
            &[ValType::I32],
            &[ValType::I32, ValType::I32],
            body,
        );
        let bytes = b.finish();
        let out = run(&bytes, "sum", &[Value::I32(101)]).unwrap();
        assert_eq!(out, vec![Value::I32(5050)]);

        // Fuel is metered per instruction, so the same loop with a small budget
        // stops rather than running (§11.6).
        let m = Module::load(&bytes).unwrap();
        let env = Env::default();
        let mut i = Instance::new(
            &m,
            &env,
            Limits {
                fuel: 500,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(
            i.call("sum", &[Value::I32(1_000_000)]),
            Err(Error::Limit(_))
        ));
        // And a bigger budget gets a bigger answer, deterministically.
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        let a = i.call("sum", &[Value::I32(1000)]).unwrap();
        let mut j = Instance::new(&m, &env, Limits::default()).unwrap();
        let bb = j.call("sum", &[Value::I32(1000)]).unwrap();
        assert_eq!(a, bb);
        assert_eq!(i.fuel_used(), j.fuel_used());
    }

    #[test]
    fn memory_loads_stores_and_bounds() {
        let mut b = Build::default();
        b.memory(1, Some(2));
        // store f64 at ptr, then load it back doubled
        let mut body = vec![0x20, 0, 0x20, 1, 0x39, 0x03, 0x00]; // f64.store
        body.extend([0x20, 0, 0x2b, 0x03, 0x00]); // f64.load
        body.extend([0x20, 1, 0xa0]); // + x
        b.func(
            "roundtrip",
            &[ValType::I32, ValType::F64],
            &[ValType::F64],
            &[],
            body,
        );
        let bytes = b.finish();
        let out = run(&bytes, "roundtrip", &[Value::I32(64), Value::F64(1.5)]).unwrap();
        assert_eq!(out, vec![Value::F64(3.0)]);

        // Out of bounds traps rather than growing anything.
        assert!(matches!(
            run(&bytes, "roundtrip", &[Value::I32(65530), Value::F64(1.0)]),
            Err(Error::Trap(_))
        ));
    }

    #[test]
    fn memory_grow_respects_the_cap() {
        let mut b = Build::default();
        b.memory(1, None);
        let mut body = vec![0x20, 0, 0x40, 0x00];
        b.func("grow", &[ValType::I32], &[ValType::I32], &[], body.clone());
        body.clear();
        let bytes = b.finish();
        let m = Module::load(&bytes).unwrap();
        let env = Env::default();
        // A 4-page cap: growing by 2 works, growing by 100 returns -1 rather
        // than allocating (§11.6's memory cap).
        let mut i = Instance::new(
            &m,
            &env,
            Limits {
                memory: 4 * PAGE,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            i.call("grow", &[Value::I32(2)]).unwrap(),
            vec![Value::I32(1)]
        );
        assert_eq!(
            i.call("grow", &[Value::I32(100)]).unwrap(),
            vec![Value::I32(-1)]
        );
        assert_eq!(i.memory().len(), 3 * PAGE);

        // A module whose *declared* minimum is over the cap never instantiates.
        let mut b = Build::default();
        b.memory(64, None);
        b.func("f", &[], &[], &[], vec![]);
        let m = Module::load(&b.finish()).unwrap();
        assert!(matches!(
            Instance::new(
                &m,
                &env,
                Limits {
                    memory: PAGE,
                    ..Default::default()
                }
            ),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn call_and_call_indirect() {
        let mut b = Build::default();
        // double(x) = x * 2
        let mut d = vec![0x20, 0];
        d.extend(i32c(2));
        d.push(0x6c);
        let double = b.func("double", &[ValType::I32], &[ValType::I32], &[], d);
        // triple(x) = x * 3
        let mut t = vec![0x20, 0];
        t.extend(i32c(3));
        t.push(0x6c);
        let triple = b.func("triple", &[ValType::I32], &[ValType::I32], &[], t);
        b.table = vec![double, triple];
        // via(i, x) = table[i](x)
        let ty = b.ty(&[ValType::I32], &[ValType::I32]);
        let mut body = vec![0x20, 1, 0x20, 0, 0x11];
        body.extend(leb(ty as u64));
        body.push(0x00);
        b.func(
            "via",
            &[ValType::I32, ValType::I32],
            &[ValType::I32],
            &[],
            body,
        );
        // plus(x) = double(x) + triple(x)
        let mut p = vec![0x20, 0, 0x10];
        p.extend(leb(double as u64));
        p.extend([0x20, 0, 0x10]);
        p.extend(leb(triple as u64));
        p.push(0x6a);
        b.func("plus", &[ValType::I32], &[ValType::I32], &[], p);

        let bytes = b.finish();
        assert_eq!(
            run(&bytes, "via", &[Value::I32(0), Value::I32(21)]).unwrap(),
            vec![Value::I32(42)]
        );
        assert_eq!(
            run(&bytes, "via", &[Value::I32(1), Value::I32(21)]).unwrap(),
            vec![Value::I32(63)]
        );
        assert_eq!(
            run(&bytes, "plus", &[Value::I32(10)]).unwrap(),
            vec![Value::I32(50)]
        );
        // An index past the table traps.
        assert!(matches!(
            run(&bytes, "via", &[Value::I32(7), Value::I32(1)]),
            Err(Error::Trap(_))
        ));
    }

    #[test]
    fn the_host_abi_is_the_only_import_allowed() {
        // `log` works, and its message reaches the host rather than a terminal.
        let mut b = Build::default();
        b.memory(1, None);
        b.data.push((0, b"hello from a plugin".to_vec()));
        let log = b.import("log", &[ValType::I32, ValType::I32], &[]);
        let mut body = i32c(0);
        body.extend(i32c(19));
        body.push(0x10);
        body.extend(leb(log as u64));
        b.func("go", &[], &[], &[], body);
        let bytes = b.finish();
        let m = Module::load(&bytes).unwrap();
        let env = Env::default();
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        i.call("go", &[]).unwrap();
        assert_eq!(i.logs(), vec!["hello from a plugin".to_string()]);

        // An import from anywhere else is refused at load: §11.6 says none, and
        // a host that let one through would be a different sandbox.
        let mut bad = bytes.clone();
        let at = bad
            .windows(HOST_MODULE.len())
            .position(|w| w == HOST_MODULE.as_bytes())
            .unwrap();
        bad[at..at + 4].copy_from_slice(b"wasi");
        assert!(matches!(Module::load(&bad), Err(Error::Forbidden(_))));

        // As is a function `omni_plugin/1` does not have.
        let mut b = Build::default();
        b.import("open_file", &[ValType::I32, ValType::I32], &[]);
        b.func("f", &[], &[], &[], vec![]);
        assert!(matches!(
            Module::load(&b.finish()),
            Err(Error::Forbidden(_))
        ));
    }

    #[test]
    fn abort_traps_with_the_plugins_own_message() {
        let mut b = Build::default();
        b.memory(1, None);
        b.data.push((0, b"scheme not supported".to_vec()));
        let abort = b.import("abort", &[ValType::I32, ValType::I32], &[]);
        let mut body = i32c(0);
        body.extend(i32c(20));
        body.push(0x10);
        body.extend(leb(abort as u64));
        b.func("go", &[], &[], &[], body);
        match run(&b.finish(), "go", &[]) {
            Err(Error::Trap(m)) => assert!(m.contains("scheme not supported"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn read_object_is_sandboxed_to_the_declared_refs() {
        let mut b = Build::default();
        b.memory(1, None);
        // The digest the plugin asks for lives at 0; the answer goes to 64.
        b.data.push((0, vec![7u8; 32]));
        let read = b.import(
            "read_object",
            &[ValType::I32, ValType::I32, ValType::I32],
            &[ValType::I32],
        );
        let mut body = i32c(0);
        body.extend(i32c(64));
        body.extend(i32c(256));
        body.push(0x10);
        body.extend(leb(read as u64));
        b.func("go", &[], &[ValType::I32], &[], body);
        let bytes = b.finish();
        let m = Module::load(&bytes).unwrap();

        // Declared: the plugin gets the bytes.
        let objects = |d: &[u8; 32]| -> Option<Vec<u8>> {
            (*d == [7u8; 32]).then(|| b"tensor bytes".to_vec())
        };
        let env = Env {
            objects: &objects,
            log: Default::default(),
        };
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        assert_eq!(i.call("go", &[]).unwrap(), vec![Value::I32(12)]);
        assert_eq!(&i.memory()[64..76], b"tensor bytes");

        // Not declared: -1, and no way to find out anything else.
        let env = Env::default();
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        assert_eq!(i.call("go", &[]).unwrap(), vec![Value::I32(-1)]);
    }

    #[test]
    fn forbidden_and_unimplemented_proposals_are_refused_at_load() {
        // Fixed-width SIMD is *implemented*, so a `v128.load` — opcode, memarg
        // and all — loads rather than being refused. `wasm_simd_vectors.rs`
        // checks what it computes; this only checks that it is admitted.
        let mut b = Build::default();
        b.memory(1, None);
        b.func(
            "f",
            &[],
            &[],
            &[],
            vec![0x41, 0x00, 0xfd, 0x00, 0x04, 0x00, 0x1a, 0x0b],
        );
        assert!(Module::load(&b.finish()).is_ok());

        // Relaxed-SIMD shares that prefix and is forbidden rather than absent:
        // it is permitted to give different answers on different hosts, which is
        // the one thing §11.6 rules out. `0xfd 0x100` is relaxed_swizzle.
        let mut b = Build::default();
        b.memory(1, None);
        b.func("f", &[], &[], &[], vec![0xfd, 0x80, 0x02, 0x0b]);
        match Module::load(&b.finish()) {
            Err(Error::Forbidden(m)) => assert!(m.contains("relaxed"), "{m}"),
            Err(other) => panic!("relaxed-SIMD refused, but not as forbidden: {other}"),
            Ok(_) => panic!("a relaxed-SIMD module loaded"),
        }

        // An atomic: forbidden (threads).
        let mut b = Build::default();
        b.memory(1, None);
        b.func("f", &[], &[], &[], vec![0xfe, 0x10]);
        assert!(matches!(
            Module::load(&b.finish()),
            Err(Error::Forbidden(_))
        ));

        // A shared memory: also threads.
        let mut b = Build::default();
        b.func("f", &[], &[], &[], vec![]);
        let mut bytes = b.finish();
        // Splice in a memory section with the shared flag set.
        let mem_section = vec![0x05u8, 0x04, 0x01, 0x03, 0x01, 0x02];
        let at = bytes
            .windows(2)
            .position(|w| w[0] == 0x07)
            .unwrap_or(bytes.len());
        bytes.splice(at..at, mem_section);
        assert!(matches!(Module::load(&bytes), Err(Error::Forbidden(_))));
    }

    #[test]
    fn floats_are_deterministic() {
        // §11.6 requires determinism and forbids NaN-payload dependence. Every
        // NaN this host produces is the same NaN.
        let mut b = Build::default();
        let body = vec![0x20, 0, 0x20, 1, 0xa3]; // f64.div
        b.func(
            "div",
            &[ValType::F64, ValType::F64],
            &[ValType::F64],
            &[],
            body,
        );
        let out = run(&b.finish(), "div", &[Value::F64(0.0), Value::F64(0.0)]).unwrap();
        match out[0] {
            Value::F64(x) => {
                assert!(x.is_nan());
                assert_eq!(x.to_bits(), 0x7ff8_0000_0000_0000);
            }
            other => panic!("{other:?}"),
        }

        // min/max follow WebAssembly's rules, not Rust's: NaN propagates and
        // −0 < +0.
        let mut b = Build::default();
        b.func(
            "min",
            &[ValType::F64, ValType::F64],
            &[ValType::F64],
            &[],
            vec![0x20, 0, 0x20, 1, 0xa4],
        );
        let bytes = b.finish();
        let out = run(&bytes, "min", &[Value::F64(f64::NAN), Value::F64(1.0)]).unwrap();
        assert!(matches!(out[0], Value::F64(x) if x.is_nan()));
        let out = run(&bytes, "min", &[Value::F64(-0.0), Value::F64(0.0)]).unwrap();
        assert!(matches!(out[0], Value::F64(x) if x == 0.0 && x.is_sign_negative()));
    }

    #[test]
    fn traps_are_traps() {
        let mut b = Build::default();
        b.func("boom", &[], &[], &[], vec![0x00]);
        assert!(matches!(run(&b.finish(), "boom", &[]), Err(Error::Trap(_))));

        let mut b = Build::default();
        let body = vec![0x20, 0, 0x20, 1, 0x6d];
        b.func(
            "div",
            &[ValType::I32, ValType::I32],
            &[ValType::I32],
            &[],
            body,
        );
        assert!(matches!(
            run(&b.finish(), "div", &[Value::I32(1), Value::I32(0)]),
            Err(Error::Trap(_))
        ));

        // A malformed module is malformed, not a panic.
        assert!(Module::load(b"not wasm at all").is_err());
        assert!(Module::load(&[]).is_err());
        assert!(Module::load(b"\0asm\x09\0\0\0").is_err());
        // Truncation at every length: none of these may panic.
        let bytes = b.finish();
        for cut in 0..bytes.len() {
            let _ = Module::load(&bytes[..cut]);
        }
    }

    #[test]
    fn bulk_memory_works_and_stays_in_bounds() {
        let mut b = Build::default();
        b.memory(1, None);
        b.data.push((0, b"abcdefgh".to_vec()));
        // fill(ptr, byte, n) then copy 8 bytes from 0 to ptr+8
        let mut body = vec![0x20, 0, 0x20, 1, 0x20, 2, 0xfc, 0x0b, 0x00];
        body.extend([0x20, 0]);
        body.extend(i32c(0));
        body.extend(i32c(8));
        body.extend([0xfc, 0x0a, 0x00, 0x00]);
        b.func(
            "go",
            &[ValType::I32, ValType::I32, ValType::I32],
            &[],
            &[],
            body,
        );
        let bytes = b.finish();
        let m = Module::load(&bytes).unwrap();
        let env = Env::default();
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        i.call("go", &[Value::I32(1024), Value::I32(0x41), Value::I32(4)])
            .unwrap();
        assert_eq!(&i.memory()[1024..1032], b"abcdefgh");

        // A fill that runs off the end traps.
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        assert!(matches!(
            i.call("go", &[Value::I32(65535), Value::I32(0), Value::I32(1000)]),
            Err(Error::Trap(_))
        ));
    }

    #[test]
    fn br_table_and_if_else() {
        // A switch with void blocks and a local for the answer, which is the
        // shape that makes the label depths legible: index 0 exits the inner
        // block and runs the arm; anything else takes the default and exits the
        // outer one, skipping it.
        let mut b = Build::default();
        let mut body = Vec::new();
        body.extend([0x02, 0x40]); // block (void) — outer
        body.extend([0x02, 0x40]); // block (void) — inner
        body.extend([0x20, 0]);
        body.extend([0x0e, 0x01, 0x00, 0x01]); // br_table [0] default 1
        body.push(0x0b); // end inner: the arm for index 0
        body.extend(i32c(10));
        body.extend([0x21, 1]);
        body.push(0x0b); // end outer
        body.extend([0x20, 1]);
        b.func(
            "pick",
            &[ValType::I32],
            &[ValType::I32],
            &[ValType::I32],
            body,
        );
        let bytes = b.finish();
        assert_eq!(
            run(&bytes, "pick", &[Value::I32(0)]).unwrap(),
            vec![Value::I32(10)]
        );
        for other in [1, 2, 99] {
            assert_eq!(
                run(&bytes, "pick", &[Value::I32(other)]).unwrap(),
                vec![Value::I32(0)],
                "index {other} should have taken the default"
            );
        }

        // if/else, both arms.
        let mut b = Build::default();
        let mut body = vec![0x20, 0, 0x04, 0x7f];
        body.extend(i32c(1));
        body.push(0x05);
        body.extend(i32c(2));
        body.push(0x0b);
        b.func("cond", &[ValType::I32], &[ValType::I32], &[], body);
        let bytes = b.finish();
        assert_eq!(
            run(&bytes, "cond", &[Value::I32(1)]).unwrap(),
            vec![Value::I32(1)]
        );
        assert_eq!(
            run(&bytes, "cond", &[Value::I32(0)]).unwrap(),
            vec![Value::I32(2)]
        );
    }

    #[test]
    fn the_module_describes_its_own_exports() {
        let mut b = Build::default();
        b.memory(1, None);
        b.func("alloc", &[ValType::I32], &[ValType::I32], &[], i32c(1024));
        b.func("f", &[ValType::F64], &[ValType::F64], &[], vec![0x20, 0]);
        let m = Module::load(&b.finish()).unwrap();
        let mut names = m.exported_functions();
        names.sort();
        assert_eq!(names, vec!["alloc".to_string(), "f".to_string()]);
        assert_eq!(
            m.func_type("f"),
            Some(&FuncType {
                params: vec![ValType::F64],
                results: vec![ValType::F64]
            })
        );
        assert!(m.has_memory());

        // And its allocator is callable, which is how the host gets a buffer
        // inside someone else's memory.
        let env = Env::default();
        let mut i = Instance::new(&m, &env, Limits::default()).unwrap();
        assert_eq!(i.alloc(16).unwrap(), 1024);
        i.write(1024, &[1, 2, 3]).unwrap();
        assert_eq!(i.read(1024, 3).unwrap(), vec![1, 2, 3]);
        assert!(i.write(u32::MAX - 2, &[1, 2, 3, 4]).is_err());
    }
}
