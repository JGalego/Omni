//! §11.5 — plugins as content-addressed artifacts, and what running one means.
//!
//! [`crate::wasm`] is the engine; this is the part that connects it to OMNI. A
//! plugin is a `PluginModule` object (§11.5) declaring what it provides, plus
//! WebAssembly modules stored as ordinary blobs. Because the manifest is an
//! object like any other, a plugin can be *embedded in the container*, signed
//! with it, and deduplicated across every model that uses it — which is the
//! difference between "download this library" and "the model is self-contained".
//!
//! The calling convention below is this implementation's, not the
//! specification's: §11.6 fixes the host ABI (`alloc`, `dealloc`, `log`,
//! `abort`, `read_object`) and the determinism profile, and leaves the shape of
//! an *expression* entry point to the registry. Ours is the smallest thing that
//! works for a tensor op:
//!
//! ```text
//! op(argc: i32, argv: i32, attrs: i32, attrs_len: i32, out: i32, out_len: i32) -> i32
//! ```
//!
//! `argv` points at `argc` pairs of `(ptr, len)`, each an array of `f64`;
//! `attrs` is the node's attributes as canonical CBOR; `out` is a buffer of
//! `out_len` `f64`s the host allocated through the module's own `alloc`. A
//! non-zero return is the plugin refusing, and the host reports that as
//! *indeterminate* — the model may be fine and this runtime may simply be
//! wrong for it.

use crate::cbor::Value;
use crate::container::Digest;
use crate::expr::{Ref, Tensor};
use crate::wasm::{Env, Instance, Limits, Module};

pub const SCHEMA: &str = "omni.plugin/manifest";

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "plugin: {}", self.0)
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// What a plugin says it provides (§11.5).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provides {
    pub otypes: Vec<u64>,
    pub dtypes: Vec<String>,
    pub expr_ops: Vec<String>,
    pub dialects: Vec<(String, u32)>,
    pub codecs: Vec<String>,
    pub schemas: Vec<String>,
}

/// A WebAssembly implementation slot: which blob, and which export in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleSlot {
    pub reference: Ref,
    pub export: String,
}

/// The §11.5 manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub ns: String,
    pub version: u32,
    pub provides: Provides,
    pub requires: Vec<(String, u32)>,
    /// `validate`, `shape`, `reference`, `decode`, … by slot name.
    pub modules: Vec<(String, ModuleSlot)>,
    /// Optional native builds. This host never loads one — a native artifact is
    /// not portable and not sandboxable, and §11.6's whole argument is that the
    /// WASM module is what makes the plugin trustworthy.
    pub native_targets: Vec<String>,
    pub license: Option<String>,
}

fn ref_value(r: &Ref) -> Value {
    Value::Array(vec![Value::U(r.0 as u64), Value::Bytes(r.1.to_vec())])
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

impl Manifest {
    pub fn to_value(&self) -> Value {
        let mut provides: Vec<(&str, Value)> = Vec::new();
        if !self.provides.otypes.is_empty() {
            provides.push((
                "otypes",
                Value::Array(self.provides.otypes.iter().map(|n| Value::U(*n)).collect()),
            ));
        }
        for (key, list) in [
            ("dtypes", &self.provides.dtypes),
            ("expr_ops", &self.provides.expr_ops),
            ("codecs", &self.provides.codecs),
            ("schemas", &self.provides.schemas),
        ] {
            if !list.is_empty() {
                provides.push((
                    key,
                    Value::Array(list.iter().map(|s| Value::text(s.clone())).collect()),
                ));
            }
        }
        if !self.provides.dialects.is_empty() {
            provides.push((
                "dialects",
                Value::Array(
                    self.provides
                        .dialects
                        .iter()
                        .map(|(ns, v)| {
                            Value::map(vec![
                                ("ns", Value::text(ns.clone())),
                                ("v", Value::U(*v as u64)),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text(SCHEMA)),
            ("v", Value::U(1)),
            ("ns", Value::text(self.ns.clone())),
            ("version", Value::U(self.version as u64)),
            ("provides", Value::map(provides)),
            (
                "modules",
                Value::Map(
                    self.modules
                        .iter()
                        .map(|(slot, m)| {
                            (
                                Value::text(slot.clone()),
                                Value::map(vec![
                                    ("ref", ref_value(&m.reference)),
                                    ("export", Value::text(m.export.clone())),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if !self.requires.is_empty() {
            p.push((
                "requires",
                Value::Array(
                    self.requires
                        .iter()
                        .map(|(ns, v)| {
                            Value::map(vec![
                                ("ns", Value::text(ns.clone())),
                                ("v", Value::U(*v as u64)),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        if !self.native_targets.is_empty() {
            p.push((
                "native",
                Value::Array(
                    self.native_targets
                        .iter()
                        .map(|t| Value::map(vec![("target", Value::text(t.clone()))]))
                        .collect(),
                ),
            ));
        }
        if let Some(l) = &self.license {
            p.push(("license", Value::text(l.clone())));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Manifest> {
        if v.get("t").and_then(|x| x.as_str()) != Some(SCHEMA) {
            return Err(Error("not an omni.plugin/manifest object".into()));
        }
        let ns = v
            .get("ns")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("a plugin manifest has no `ns`".into()))?
            .to_string();
        let provides_v = v.get("provides");
        let provides = Provides {
            otypes: match provides_v.and_then(|p| p.get("otypes")) {
                Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).collect(),
                _ => Vec::new(),
            },
            dtypes: strings(provides_v.and_then(|p| p.get("dtypes"))),
            expr_ops: strings(provides_v.and_then(|p| p.get("expr_ops"))),
            codecs: strings(provides_v.and_then(|p| p.get("codecs"))),
            schemas: strings(provides_v.and_then(|p| p.get("schemas"))),
            dialects: match provides_v.and_then(|p| p.get("dialects")) {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|d| {
                        Some((
                            d.get("ns")?.as_str()?.to_string(),
                            d.get("v")?.as_u64()? as u32,
                        ))
                    })
                    .collect(),
                _ => Vec::new(),
            },
        };
        let mut modules = Vec::new();
        if let Some(Value::Map(m)) = v.get("modules") {
            for (slot, spec) in m {
                let slot = slot
                    .as_str()
                    .ok_or_else(|| Error("a module slot name is not text".into()))?
                    .to_string();
                let reference = spec
                    .get("ref")
                    .and_then(|r| crate::expr::parse_ref_value(r).ok())
                    .ok_or_else(|| Error(format!("module slot `{slot}` has no ref")))?;
                let export = spec
                    .get("export")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error(format!("module slot `{slot}` names no export")))?
                    .to_string();
                modules.push((slot, ModuleSlot { reference, export }));
            }
        }
        Ok(Manifest {
            ns,
            version: v.get("version").and_then(|x| x.as_u64()).unwrap_or(1) as u32,
            provides,
            requires: match v.get("requires") {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|d| {
                        Some((
                            d.get("ns")?.as_str()?.to_string(),
                            d.get("v")?.as_u64()? as u32,
                        ))
                    })
                    .collect(),
                _ => Vec::new(),
            },
            modules,
            native_targets: match v.get("native") {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|n| Some(n.get("target")?.as_str()?.to_string()))
                    .collect(),
                _ => Vec::new(),
            },
            license: v
                .get("license")
                .and_then(|x| x.as_str())
                .map(str::to_string),
        })
    }

    /// Whether this plugin claims the expression op `ns/name.v`.
    pub fn provides_op(&self, ns: &str, name: &str) -> bool {
        self.ns == ns
            && (self.provides.expr_ops.iter().any(|o| o == name)
                || self
                    .provides
                    .expr_ops
                    .iter()
                    .any(|o| o == &format!("{ns}/{name}")))
    }
}

// ------------------------------------------------------------------- running --

/// A loaded plugin: its manifest and the module bytes behind each slot.
pub struct Loaded {
    pub manifest: Manifest,
    modules: Vec<(String, Module)>,
}

impl Loaded {
    /// Loads a plugin's WebAssembly, resolving each slot through a store.
    ///
    /// A slot whose module is absent, or which uses something this host does not
    /// implement, is *left out* rather than failing the load: a plugin whose
    /// `decode` this host cannot run may still have a `reference` it can.
    pub fn load(
        manifest: Manifest,
        resolve: &dyn Fn(&Digest) -> Option<Vec<u8>>,
    ) -> (Loaded, Vec<String>) {
        let mut modules = Vec::new();
        let mut problems = Vec::new();
        for (slot, m) in &manifest.modules {
            let Some(bytes) = resolve(&m.reference.1) else {
                problems.push(format!("slot `{slot}`: the module object is not present"));
                continue;
            };
            match Module::load(&bytes) {
                Ok(module) => {
                    if module.func_type(&m.export).is_none() {
                        problems.push(format!(
                            "slot `{slot}`: the module exports no function `{}`",
                            m.export
                        ));
                        continue;
                    }
                    modules.push((slot.clone(), module));
                }
                Err(e) => problems.push(format!("slot `{slot}`: {e}")),
            }
        }
        (Loaded { manifest, modules }, problems)
    }

    pub fn slot(&self, name: &str) -> Option<&Module> {
        self.modules.iter().find(|(s, _)| s == name).map(|(_, m)| m)
    }

    pub fn slots(&self) -> Vec<&str> {
        self.modules.iter().map(|(s, _)| s.as_str()).collect()
    }
}

/// Runs plugin expression ops through the WebAssembly host.
pub struct Host<'a> {
    plugins: Vec<Loaded>,
    limits: Limits,
    /// Objects a plugin may read (§11.6's sandboxed `read_object`).
    pub objects: &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>,
    /// What the last call cost, so a caller can report it.
    pub last_fuel: std::cell::Cell<u64>,
    pub last_logs: std::cell::RefCell<Vec<String>>,
}

impl<'a> Host<'a> {
    pub fn new(plugins: Vec<Loaded>) -> Host<'a> {
        Host {
            plugins,
            limits: Limits::default(),
            objects: &|_| None,
            last_fuel: std::cell::Cell::new(0),
            last_logs: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn limits(mut self, l: Limits) -> Self {
        self.limits = l;
        self
    }

    /// The objects a plugin may read (§11.6's sandboxed `read_object`). Nothing
    /// outside this closure is reachable from inside a module.
    pub fn with_objects(mut self, objects: &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>) -> Self {
        self.objects = objects;
        self
    }

    pub fn ids(&self) -> Vec<String> {
        let mut out = Vec::new();
        for p in &self.plugins {
            for op in &p.manifest.provides.expr_ops {
                out.push(format!("{}/{}", p.manifest.ns, op));
            }
        }
        out
    }
}

impl crate::expr::PluginRunner for Host<'_> {
    fn run(
        &self,
        ns: &str,
        name: &str,
        _version: u64,
        attrs: &Value,
        args: &[Tensor],
        out_elems: usize,
    ) -> Result<Vec<f64>, String> {
        let Some(p) = self
            .plugins
            .iter()
            .find(|p| p.manifest.provides_op(ns, name))
        else {
            return Err(format!("no loaded plugin provides `{ns}/{name}`"));
        };
        let module = p
            .slot("reference")
            .ok_or_else(|| format!("`{ns}` has no runnable `reference` module"))?;
        let export = &p
            .manifest
            .modules
            .iter()
            .find(|(s, _)| s == "reference")
            .ok_or_else(|| "no reference slot".to_string())?
            .1
            .export;

        let env = Env {
            objects: self.objects,
            log: std::cell::RefCell::new(Vec::new()),
        };
        let mut inst = Instance::new(module, &env, self.limits).map_err(|e| e.to_string())?;

        // Arguments: each f64 array into the module's own allocator, then a
        // table of (ptr, len) pairs.
        let mut argv: Vec<u8> = Vec::with_capacity(args.len() * 8);
        for a in args {
            let bytes: Vec<u8> = a.data.iter().flat_map(|x| x.to_le_bytes()).collect();
            let ptr = inst
                .alloc(bytes.len().max(1) as u32)
                .map_err(|e| e.to_string())?;
            inst.write(ptr, &bytes).map_err(|e| e.to_string())?;
            argv.extend((ptr as i32).to_le_bytes());
            argv.extend((a.data.len() as i32).to_le_bytes());
        }
        let argv_ptr = inst
            .alloc(argv.len().max(1) as u32)
            .map_err(|e| e.to_string())?;
        inst.write(argv_ptr, &argv).map_err(|e| e.to_string())?;

        let attr_bytes = attrs.encode();
        let attr_ptr = inst
            .alloc(attr_bytes.len().max(1) as u32)
            .map_err(|e| e.to_string())?;
        inst.write(attr_ptr, &attr_bytes)
            .map_err(|e| e.to_string())?;

        let out_bytes = out_elems * 8;
        let out_ptr = inst
            .alloc(out_bytes.max(1) as u32)
            .map_err(|e| e.to_string())?;

        let rc = inst
            .call(
                export,
                &[
                    crate::wasm::Value::I32(args.len() as i32),
                    crate::wasm::Value::I32(argv_ptr as i32),
                    crate::wasm::Value::I32(attr_ptr as i32),
                    crate::wasm::Value::I32(attr_bytes.len() as i32),
                    crate::wasm::Value::I32(out_ptr as i32),
                    crate::wasm::Value::I32(out_elems as i32),
                ],
            )
            .map_err(|e| e.to_string())?;
        self.last_fuel.set(inst.fuel_used());
        *self.last_logs.borrow_mut() = inst.logs();
        match rc.first() {
            Some(crate::wasm::Value::I32(0)) => {}
            Some(crate::wasm::Value::I32(code)) => {
                return Err(format!(
                    "the plugin refused with code {code}{}",
                    match inst.logs().last() {
                        Some(l) => format!(": {l}"),
                        None => String::new(),
                    }
                ))
            }
            _ => return Err("the plugin's entry point returned no status".into()),
        }
        let raw = inst.read(out_ptr, out_bytes).map_err(|e| e.to_string())?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

// ------------------------------------------------------- an example plugin --

/// A tiny, complete plugin module: `scale(x, f) = x * f[0]`.
///
/// It exists so the whole path — an expression node naming an op no build knows,
/// a manifest declaring it, a WebAssembly module shipped in the container, and a
/// host that runs it under §11.6's limits — can be exercised end to end by
/// `omni example --plugin` and by CI. The bytes are assembled here rather than
/// committed as a blob so that what the test runs is derived from source.
///
/// The factor is a second *operand* rather than an attribute, because a
/// canonical-CBOR attribute map encodes 2.5 as an `f16` (rule D5 takes the
/// shortest form that round-trips) and a plugin that read a fixed-width float
/// out of the blob would be reading whatever D5 chose. A real plugin parses the
/// attributes it declares; this one takes numbers as numbers and says so.
pub fn example_module() -> Vec<u8> {
    let mut b = Enc::default();
    // A bump allocator over a fixed arena, which is all a pure function needs.
    // `alloc(n)`: ptr = *0; *0 = ptr + ((n + 7) & ~7); return ptr.
    let t_i32_i32 = b.ty(&[I32], &[I32]);
    let arena_start = 1024i32;
    let mut alloc = Vec::new();
    // if *0 == 0 { *0 = arena_start }
    alloc.extend(i32const(0));
    alloc.extend([0x28, 0x02, 0x00]); // i32.load
    alloc.extend([0x45]); // i32.eqz
    alloc.extend([0x04, 0x40]); // if
    alloc.extend(i32const(0));
    alloc.extend(i32const(arena_start));
    alloc.extend([0x36, 0x02, 0x00]); // i32.store
    alloc.push(0x0b);
    // ptr = *0; end = ptr + ((n + 7) & -8); *0 = end
    alloc.extend(i32const(0));
    alloc.extend([0x28, 0x02, 0x00]);
    alloc.extend([0x21, 1]);
    alloc.extend([0x20, 1]);
    alloc.extend([0x20, 0]);
    alloc.extend(i32const(7));
    alloc.extend([0x6a]);
    alloc.extend(i32const(-8));
    alloc.extend([0x71, 0x6a]); // and, add
    alloc.extend([0x21, 2]);
    alloc.extend(i32const(0));
    alloc.extend([0x20, 2]);
    alloc.extend([0x36, 0x02, 0x00]);
    // Grow linear memory until the arena fits. A plugin that wants more than the
    // host's cap gets -1 from memory.grow and returns null, which the host
    // reports rather than writing past the end of anything.
    alloc.extend([0x02, 0x40]); // block
    alloc.extend([0x03, 0x40]); // loop
    alloc.extend([0x20, 2]);
    alloc.extend([0x3f, 0x00]); // memory.size
    alloc.extend(i32const(65536));
    alloc.extend([0x6c]); // i32.mul
    alloc.extend([0x4c]); // i32.le_s
    alloc.extend([0x0d, 0x01]); // br_if 1 — the arena fits
    alloc.extend(i32const(1));
    alloc.extend([0x40, 0x00]); // memory.grow
    alloc.extend([0x21, 3]);
    alloc.extend([0x20, 3]);
    alloc.extend(i32const(-1));
    alloc.extend([0x46]); // i32.eq
    alloc.extend([0x04, 0x40]); // if the cap was hit
    alloc.extend(i32const(0));
    alloc.extend([0x0f]); // return null
    alloc.push(0x0b);
    alloc.extend([0x0c, 0x00]); // br 0
    alloc.push(0x0b); // end loop
    alloc.push(0x0b); // end block
    alloc.extend([0x20, 1]);
    b.func("alloc", t_i32_i32, &[I32, I32, I32], alloc);

    // The op. Signature: (argc, argv, attrs, attrs_len, out, out_len) -> i32.
    let t_op = b.ty(&[I32, I32, I32, I32, I32, I32], &[I32]);
    // locals: 6 = x_ptr, 7 = i, 8 = factor (f64)
    let mut op = Vec::new();
    // Refuse anything but two arguments: a plugin that guesses is worse than one
    // that says no.
    op.extend([0x20, 0]);
    op.extend(i32const(2));
    op.extend([0x47]); // i32.ne
    op.extend([0x04, 0x40]);
    op.extend(i32const(1));
    op.extend([0x0f]); // return 1
    op.push(0x0b);
    // x_ptr = argv[0].ptr
    op.extend([0x20, 1]);
    op.extend([0x28, 0x02, 0x00]);
    op.extend([0x21, 6]);
    // factor = *(argv[1].ptr)
    op.extend([0x20, 1]);
    op.extend(i32const(8));
    op.extend([0x6a]);
    op.extend([0x28, 0x02, 0x00]);
    op.extend([0x2b, 0x03, 0x00]); // f64.load
    op.extend([0x21, 8]);
    // for i in 0..out_len { out[i] = x[i] * factor }
    op.extend(i32const(0));
    op.extend([0x21, 7]);
    op.extend([0x02, 0x40]); // block
    op.extend([0x03, 0x40]); // loop
    op.extend([0x20, 7]);
    op.extend([0x20, 5]);
    op.extend([0x4e]); // i >= out_len (signed)
    op.extend([0x0d, 0x01]); // br_if 1
                             // out + i*8
    op.extend([0x20, 4]);
    op.extend([0x20, 7]);
    op.extend(i32const(8));
    op.extend([0x6c, 0x6a]);
    // x[i]
    op.extend([0x20, 6]);
    op.extend([0x20, 7]);
    op.extend(i32const(8));
    op.extend([0x6c, 0x6a]);
    op.extend([0x2b, 0x03, 0x00]);
    op.extend([0x20, 8]);
    op.extend([0xa2]); // f64.mul
    op.extend([0x39, 0x03, 0x00]); // f64.store
    op.extend([0x20, 7]);
    op.extend(i32const(1));
    op.extend([0x6a, 0x21, 7]);
    op.extend([0x0c, 0x00]);
    op.push(0x0b);
    op.push(0x0b);
    op.extend(i32const(0));
    b.func("scale", t_op, &[I32, I32, F64], op);
    b.memory(2);
    b.finish()
}

const I32: u8 = 0x7f;
const F64: u8 = 0x7c;

/// The smallest WebAssembly encoder that can express [`example_module`].
#[derive(Default)]
struct Enc {
    types: Vec<(Vec<u8>, Vec<u8>)>,
    funcs: Vec<(u32, Vec<u8>, Vec<u8>)>,
    exports: Vec<(String, u32)>,
    pages: u32,
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

fn i32const(n: i32) -> Vec<u8> {
    let mut v = vec![0x41];
    v.extend(sleb(n as i64));
    v
}

impl Enc {
    fn ty(&mut self, params: &[u8], results: &[u8]) -> u32 {
        self.types.push((params.to_vec(), results.to_vec()));
        (self.types.len() - 1) as u32
    }

    fn func(&mut self, name: &str, ty: u32, locals: &[u8], body: Vec<u8>) {
        self.funcs.push((ty, locals.to_vec(), body));
        self.exports
            .push((name.to_string(), (self.funcs.len() - 1) as u32));
    }

    fn memory(&mut self, pages: u32) {
        self.pages = pages;
    }

    fn finish(&self) -> Vec<u8> {
        let mut out = b"\0asm\x01\0\0\0".to_vec();
        let section = |out: &mut Vec<u8>, id: u8, body: Vec<u8>| {
            out.push(id);
            out.extend(leb(body.len() as u64));
            out.extend(body);
        };
        let mut b = leb(self.types.len() as u64);
        for (p, r) in &self.types {
            b.push(0x60);
            b.extend(leb(p.len() as u64));
            b.extend(p);
            b.extend(leb(r.len() as u64));
            b.extend(r);
        }
        section(&mut out, 1, b);
        let mut b = leb(self.funcs.len() as u64);
        for (t, _, _) in &self.funcs {
            b.extend(leb(*t as u64));
        }
        section(&mut out, 3, b);
        let mut b = leb(1);
        b.push(0x00);
        b.extend(leb(self.pages as u64));
        section(&mut out, 5, b);
        let mut b = leb((self.exports.len() + 1) as u64);
        for (name, idx) in &self.exports {
            b.extend(leb(name.len() as u64));
            b.extend(name.as_bytes());
            b.push(0x00);
            b.extend(leb(*idx as u64));
        }
        b.extend(leb(6));
        b.extend(b"memory");
        b.push(0x02);
        b.extend(leb(0));
        section(&mut out, 7, b);
        let mut b = leb(self.funcs.len() as u64);
        for (_, locals, body) in &self.funcs {
            let mut f = leb(locals.len() as u64);
            for l in locals {
                f.extend(leb(1));
                f.push(*l);
            }
            f.extend(body.clone());
            f.push(0x0b);
            b.extend(leb(f.len() as u64));
            b.extend(f);
        }
        section(&mut out, 10, b);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::expr::PluginRunner;

    fn manifest(module: Ref) -> Manifest {
        Manifest {
            ns: "org.acme/scale".into(),
            version: 1,
            provides: Provides {
                expr_ops: vec!["scale".into()],
                ..Default::default()
            },
            requires: vec![("omni.core".into(), 1)],
            modules: vec![(
                "reference".into(),
                ModuleSlot {
                    reference: module,
                    export: "scale".into(),
                },
            )],
            native_targets: vec!["x86_64-linux-gnu".into()],
            license: Some("Apache-2.0".into()),
        }
    }

    #[test]
    fn a_manifest_round_trips() {
        let m = manifest((0, [3u8; 32]));
        let bytes = m.to_value().encode();
        let back = Manifest::from_value(&crate::cbor::decode(&bytes).unwrap()).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.to_value().encode(), bytes);
        assert!(back.provides_op("org.acme/scale", "scale"));
        assert!(!back.provides_op("org.acme/scale", "unpack"));
        assert!(!back.provides_op("org.other", "scale"));
    }

    #[test]
    fn the_example_module_computes_through_the_host() {
        // The whole §11 path: a manifest, a module in a blob, a host under
        // §11.6's limits, and an op no part of this build knows how to do.
        let bytes = example_module();
        let digest = [9u8; 32];
        let resolve = |d: &Digest| -> Option<Vec<u8>> { (*d == digest).then(|| bytes.clone()) };
        let (loaded, problems) = Loaded::load(manifest((0, digest)), &resolve);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(loaded.slots(), vec!["reference"]);

        let host = Host::new(vec![loaded]);
        let attrs = Value::map(vec![("note", Value::text("unused by this example"))]);
        let x = Tensor::new(vec![4], DType::F64, vec![1.0, 2.0, 3.0, 4.0]);
        let f = Tensor::new(vec![1], DType::F64, vec![2.5]);
        let out = host
            .run("org.acme/scale", "scale", 1, &attrs, &[x, f.clone()], 4)
            .unwrap();
        assert_eq!(out, vec![2.5, 5.0, 7.5, 10.0]);
        assert!(host.last_fuel.get() > 0);

        // The plugin refuses what it was not written for, and the refusal is a
        // message rather than a wrong answer.
        let err = host
            .run(
                "org.acme/scale",
                "scale",
                1,
                &attrs,
                std::slice::from_ref(&f),
                1,
            )
            .unwrap_err();
        assert!(err.contains("refused"), "{err}");

        // An op nothing provides is not silently zero.
        assert!(host
            .run("org.acme/scale", "other", 1, &attrs, &[f], 1)
            .is_err());
    }

    #[test]
    fn a_missing_or_unrunnable_module_is_reported_not_hidden() {
        let (loaded, problems) = Loaded::load(manifest((0, [1u8; 32])), &|_| None);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("not present"), "{problems:?}");
        assert!(loaded.slot("reference").is_none());

        // A module that is not WebAssembly at all.
        let (_, problems) = Loaded::load(manifest((0, [2u8; 32])), &|_| Some(b"nope".to_vec()));
        assert_eq!(problems.len(), 1);

        // A module whose export the manifest names but does not have.
        let mut m = manifest((0, [9u8; 32]));
        m.modules[0].1.export = "absent".into();
        let bytes = example_module();
        let (_, problems) = Loaded::load(m, &|_| Some(bytes.clone()));
        assert!(problems[0].contains("no function `absent`"), "{problems:?}");
    }

    #[test]
    fn a_plugin_cannot_outrun_its_fuel() {
        let bytes = example_module();
        let resolve = |_: &Digest| -> Option<Vec<u8>> { Some(bytes.clone()) };
        let (loaded, _) = Loaded::load(manifest((0, [9u8; 32])), &resolve);
        let host = Host::new(vec![loaded]).limits(Limits {
            fuel: 200,
            ..Default::default()
        });
        let attrs = Value::map(vec![("note", Value::text("n/a"))]);
        let x = Tensor::new(vec![4096], DType::F64, vec![1.0; 4096]);
        let f = Tensor::new(vec![1], DType::F64, vec![2.0]);
        let err = host
            .run("org.acme/scale", "scale", 1, &attrs, &[x, f], 4096)
            .unwrap_err();
        assert!(err.contains("fuel"), "{err}");
    }
}
