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
    bump_allocator(&mut b);

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

/// §11.6's `alloc`: a bump allocator over a fixed arena, which is all a pure
/// function needs and what every module the host calls has to export, since the
/// host writes its arguments into the module's own memory rather than assuming
/// a layout.
fn bump_allocator(b: &mut Enc) {
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
}

/// A §07.4.2 function that answers with a fixed set of bytes.
///
/// It exists for the tests and for the CLI's worked example, and the reason it
/// is a *constant* answer is that the interesting part is the plumbing rather
/// than the arithmetic: a dialect that ships its semantics has to be loadable
/// from the container, runnable under the §11.6 limits, and believed when it
/// answers — and a shape function that decodes CBOR in hand-written
/// WebAssembly would be testing the decoder instead.
///
/// `answer` is what the function writes at `out`; the return value is its
/// length, or `code` when that is negative, which is how a function declines.
pub fn constant_answer_module(export: &str, answer: &[u8], code: i32) -> Vec<u8> {
    let mut b = Enc::default();
    bump_allocator(&mut b);
    // (in, in_len, out, out_cap) -> i32
    let t = b.ty(&[I32, I32, I32, I32], &[I32]);
    let mut f = Vec::new();
    if code < 0 {
        f.extend(i32const(code));
        f.extend([0x0f]);
    } else {
        // Refuse rather than overrun: a function that writes past the buffer
        // the host gave it is the one bug this ABI can actually have.
        f.extend([0x20, 3]);
        f.extend(i32const(answer.len() as i32));
        f.extend([0x48]); // i32.lt_s
        f.extend([0x04, 0x40]);
        f.extend(i32const(-2));
        f.extend([0x0f]);
        f.push(0x0b);
        for (i, byte) in answer.iter().enumerate() {
            f.extend([0x20, 2]); // out
            f.extend(i32const(i as i32));
            f.extend([0x6a]); // i32.add
            f.extend(i32const(*byte as i32));
            f.extend([0x3a, 0x00, 0x00]); // i32.store8
        }
        f.extend(i32const(answer.len() as i32));
    }
    b.func(export, t, &[], f);
    b.memory(1);
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

// ---------------------------------------------- §07.4.2 dialect semantics --

/// The two §07.4.2 functions a dialect may ship.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DialectFn {
    /// Computes an op's result types.
    Shape,
    /// Decides whether an op is well formed beyond its types.
    Verify,
}

impl DialectFn {
    fn key(self) -> &'static str {
        match self {
            DialectFn::Shape => "shape_fn",
            DialectFn::Verify => "verify_fn",
        }
    }
}

/// §07.4.2's shipped semantics, run in the §11.6 host.
///
/// A dialect this build has never heard of is *indeterminate* by default, and
/// that is the weakest true statement a verifier can make. §07.4.2 offers a
/// stronger one: the `DialectRef` may carry WebAssembly that computes an op's
/// result types and checks its wellformedness, so a reader can decide a graph
/// written against a dialect invented after it was built.
///
/// ## The calling convention
///
/// §07.4.2 names the slots and leaves their ABI to the registry, the way §11.6
/// leaves the expression entry point to it. This is the smallest thing that
/// works, and it is the same shape for both functions:
///
/// ```text
/// shape (in: i32, in_len: i32, out: i32, out_cap: i32) -> i32
/// verify(in: i32, in_len: i32, out: i32, out_cap: i32) -> i32
/// ```
///
/// `in` points at canonical OMNI-CBOR of `{"op": <the op>, "in": [<types>]}` —
/// the op exactly as §07.3 encodes it, with the operand types resolved. The
/// return value says what happened, and the three outcomes are §15.1's three:
///
/// | Return | `shape` | `verify` |
/// |---|---|---|
/// | `n > 0` | `n` bytes of CBOR `[<type>…]` at `out` | `n` bytes of a UTF-8 reason: **invalid** |
/// | `0` | no results | **valid** |
/// | `n < 0` | the function declines to decide: **indeterminate** | same |
///
/// A module that traps, runs out of fuel or writes past `out_cap` is
/// indeterminate too, with the reason. What it is not is invalid: a plugin that
/// will not answer says nothing about the graph.
pub struct Dialects<'a> {
    /// `(namespace, op name or `*`, which function, module, export)`.
    fns: Vec<(String, String, DialectFn, Module, String)>,
    limits: Limits,
    objects: &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>,
    /// Fuel spent and modules run, so a verifier can report what deciding cost.
    pub calls: std::cell::Cell<u64>,
    pub fuel: std::cell::Cell<u64>,
}

impl<'a> Dialects<'a> {
    pub fn new(objects: &'a dyn Fn(&[u8; 32]) -> Option<Vec<u8>>) -> Dialects<'a> {
        Dialects {
            fns: Vec::new(),
            limits: Limits::default(),
            objects,
            calls: std::cell::Cell::new(0),
            fuel: std::cell::Cell::new(0),
        }
    }

    pub fn limits(mut self, l: Limits) -> Self {
        self.limits = l;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.fns.is_empty()
    }

    pub fn len(&self) -> usize {
        self.fns.len()
    }

    /// Loads every `shape_fn` and `verify_fn` a `DialectRef` object declares.
    ///
    /// Returns the problems rather than failing: a dialect whose module is
    /// missing is one this reader cannot use, and that is a reason to fall back
    /// to *indeterminate* rather than to refuse the container.
    pub fn load(&mut self, dialect_ref: &Value) -> Vec<String> {
        let mut problems = Vec::new();
        let Some(ns) = dialect_ref.get("ns").and_then(|v| v.as_str()) else {
            return vec!["a DialectRef with no `ns`".into()];
        };
        let mut add = |op: &str, kind: DialectFn, spec: &Value, problems: &mut Vec<String>| {
            let (Some(w), Some(export)) = (
                spec.get("wasm")
                    .and_then(|v| crate::expr::parse_ref_value(v).ok()),
                spec.get("export").and_then(|v| v.as_str()),
            ) else {
                problems.push(format!("`{ns}/{op}`'s {} names no module", kind.key()));
                return;
            };
            let Some(bytes) = (self.objects)(&w.1) else {
                problems.push(format!(
                    "`{ns}/{op}`'s {} points at an object this container does not have",
                    kind.key()
                ));
                return;
            };
            match Module::load(&bytes) {
                Ok(m) => {
                    self.fns
                        .push((ns.to_string(), op.to_string(), kind, m, export.to_string()))
                }
                Err(e) => problems.push(format!("`{ns}/{op}`'s {}: {e}", kind.key())),
            }
        };
        // Dialect-wide functions, then per-op ones. A per-op function wins,
        // which is the order a reader would expect and the order they are
        // searched in below.
        for kind in [DialectFn::Shape, DialectFn::Verify] {
            if let Some(spec) = dialect_ref.get(kind.key()) {
                add("*", kind, spec, &mut problems);
            }
        }
        if let Some(Value::Map(ops)) = dialect_ref.get("ops") {
            for (name, spec) in ops {
                let Some(name) = name.as_str() else { continue };
                for kind in [DialectFn::Shape, DialectFn::Verify] {
                    // Either directly on the op, or under its version key —
                    // §07.4.2's example puts it under `v2`.
                    let found = spec.get(kind.key()).or_else(|| {
                        spec.get("versions")
                            .and_then(|v| v.as_array())
                            .and_then(|vs| vs.last())
                            .and_then(|v| v.as_u64())
                            .and_then(|v| spec.get(&format!("v{v}")))
                            .and_then(|s| s.get(kind.key()))
                    });
                    if let Some(f) = found {
                        add(name, kind, f, &mut problems);
                    }
                }
            }
        }
        problems
    }

    fn find(
        &self,
        op: &crate::ir::Op,
        kind: DialectFn,
    ) -> Option<&(String, String, DialectFn, Module, String)> {
        self.fns
            .iter()
            .find(|(ns, name, k, _, _)| *k == kind && ns == &op.dialect && name == &op.name)
            .or_else(|| {
                self.fns
                    .iter()
                    .find(|(ns, name, k, _, _)| *k == kind && ns == &op.dialect && name == "*")
            })
    }

    /// The call itself: encode, run, read back.
    fn call(
        &self,
        op: &crate::ir::Op,
        ins: &[crate::ir::Type],
        kind: DialectFn,
    ) -> Option<Result<(i32, Vec<u8>), String>> {
        let (_, _, _, module, export) = self.find(op, kind)?;
        // The op as §07.3 encodes it, with its operand types resolved. Encoding
        // the op itself rather than a summary means a shape function sees the
        // attributes, which is where a dialect keeps the thing its shape
        // depends on.
        let input = Value::map(vec![
            ("op", op.to_value()),
            (
                "in",
                Value::Array(ins.iter().map(|t| t.to_value()).collect()),
            ),
        ])
        .encode();
        let env = Env {
            objects: self.objects,
            log: std::cell::RefCell::new(Vec::new()),
        };
        let run = || -> Result<(i32, Vec<u8>), String> {
            let mut inst = Instance::new(module, &env, self.limits).map_err(|e| e.to_string())?;
            let in_ptr = inst
                .alloc(input.len().max(1) as u32)
                .map_err(|e| e.to_string())?;
            inst.write(in_ptr, &input).map_err(|e| e.to_string())?;
            // A generous but bounded answer buffer: a type is small and a
            // reason is a sentence.
            let cap = 4096u32;
            let out_ptr = inst.alloc(cap).map_err(|e| e.to_string())?;
            let rc = inst
                .call(
                    export,
                    &[
                        crate::wasm::Value::I32(in_ptr as i32),
                        crate::wasm::Value::I32(input.len() as i32),
                        crate::wasm::Value::I32(out_ptr as i32),
                        crate::wasm::Value::I32(cap as i32),
                    ],
                )
                .map_err(|e| e.to_string())?;
            self.calls.set(self.calls.get() + 1);
            self.fuel.set(self.fuel.get() + inst.fuel_used());
            let n = match rc.first() {
                Some(crate::wasm::Value::I32(n)) => *n,
                other => return Err(format!("the module returned {other:?}, not an i32")),
            };
            if n <= 0 {
                return Ok((n, Vec::new()));
            }
            if n as u32 > cap {
                return Err(format!(
                    "the module says it wrote {n} bytes into a {cap}-byte buffer"
                ));
            }
            let bytes = inst.read(out_ptr, n as usize).map_err(|e| e.to_string())?;
            Ok((n, bytes))
        };
        Some(run())
    }
}

impl crate::ir::DialectHost for Dialects<'_> {
    fn provides(&self, op: &crate::ir::Op) -> bool {
        self.find(op, DialectFn::Shape).is_some() || self.find(op, DialectFn::Verify).is_some()
    }

    fn shape(
        &self,
        op: &crate::ir::Op,
        ins: &[crate::ir::Type],
    ) -> Option<Result<Vec<crate::ir::Type>, String>> {
        match self.call(op, ins, DialectFn::Shape)? {
            Err(e) => Some(Err(e)),
            Ok((n, _)) if n < 0 => Some(Err(format!("it declined, with code {n}"))),
            Ok((_, bytes)) => {
                let v = match crate::cbor::decode(&bytes) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(format!("its answer is not canonical CBOR: {e}"))),
                };
                let Some(list) = v.as_array() else {
                    return Some(Err("its answer is not an array of types".into()));
                };
                let mut out = Vec::with_capacity(list.len());
                for t in list {
                    match crate::ir::Type::from_value(t) {
                        Ok(t) => out.push(t),
                        Err(e) => {
                            return Some(Err(format!("its answer is not a §07.3.1 type: {e}")))
                        }
                    }
                }
                Some(Ok(out))
            }
        }
    }

    fn check(&self, op: &crate::ir::Op, ins: &[crate::ir::Type]) -> Option<Result<(), String>> {
        match self.call(op, ins, DialectFn::Verify)? {
            Err(e) => Some(Err(e)),
            Ok((0, _)) => Some(Ok(())),
            Ok((n, _)) if n < 0 => Some(Err(format!("it declined, with code {n}"))),
            Ok((_, bytes)) => Some(Err(String::from_utf8_lossy(&bytes).to_string())),
        }
    }
}

#[cfg(test)]
mod dialect_tests {
    use super::*;
    use crate::dtype::DType;
    use crate::expr::Dim;
    use crate::ir::{self, Block, Function, Module as IrModule, Op, Region, Type};

    fn ty(shape: &[u64]) -> Type {
        Type::tensor(shape.iter().map(|d| Dim::N(*d)).collect(), DType::F32)
    }

    /// A `DialectRef` for `x.test` whose one op ships the two §07.4.2
    /// functions, pointing at blobs the caller holds.
    fn dialect_ref(shape: Option<[u8; 32]>, verify: Option<[u8; 32]>) -> Value {
        let mut op = vec![("versions", Value::Array(vec![Value::U(1)]))];
        if let Some(d) = shape {
            op.push((
                "shape_fn",
                Value::map(vec![
                    (
                        "wasm",
                        Value::Array(vec![
                            Value::U(crate::container::otype::BLOB as u64),
                            Value::Bytes(d.to_vec()),
                        ]),
                    ),
                    ("export", Value::text("shape")),
                ]),
            ));
        }
        if let Some(d) = verify {
            op.push((
                "verify_fn",
                Value::map(vec![
                    (
                        "wasm",
                        Value::Array(vec![
                            Value::U(crate::container::otype::BLOB as u64),
                            Value::Bytes(d.to_vec()),
                        ]),
                    ),
                    ("export", Value::text("verify")),
                ]),
            ));
        }
        Value::map(vec![
            ("t", Value::text("omni.ir/dialect")),
            ("v", Value::U(1)),
            ("ns", Value::text("x.test")),
            ("version", Value::U(1)),
            (
                "ops",
                Value::Map(vec![(Value::text("thing"), Value::map(op))]),
            ),
        ])
    }

    /// A module using one op of a dialect this build has never heard of.
    fn module_using(declared: Type) -> IrModule {
        let mut m = IrModule::new(ir::Level::Primitive, "main");
        m.dialects = vec![
            ir::DialectUse {
                ns: "omni.core".into(),
                version: 1,
                reference: None,
            },
            ir::DialectUse {
                ns: "x.test".into(),
                version: 1,
                reference: None,
            },
        ];
        m.functions.push((
            "main".into(),
            Function {
                params: vec![("x".into(), ty(&[2, 3]))],
                results: vec![declared.clone()],
                attrs: Vec::new(),
                body: Region {
                    blocks: vec![Block {
                        args: Vec::new(),
                        ops: vec![
                            Op::new("x.test", "thing", 1)
                                .with_inputs(&[0])
                                .with_output(1, declared),
                            Op::new("omni.core", "return", 1).with_inputs(&[1]),
                        ],
                    }],
                },
                constraints: Vec::new(),
            },
        ));
        m
    }

    /// The CBOR a shape function answers with: a list of §07.3.1 types.
    fn answer(types: &[Type]) -> Vec<u8> {
        Value::Array(types.iter().map(Type::to_value).collect()).encode()
    }

    #[test]
    fn a_dialect_that_ships_its_shape_function_is_decided_rather_than_unknown() {
        // §07.2's key move, applied to verification instead of execution: the
        // reader has never heard of `x.test`, and can still say the graph is
        // well-typed — because the model brought the semantics with it.
        let wasm = constant_answer_module("shape", &answer(&[ty(&[2, 3])]), 0);
        let digest = crate::HashAlgo::default().digest(&wasm);
        let objects = |d: &[u8; 32]| (d == &digest).then(|| wasm.clone());
        let mut host = Dialects::new(&objects);
        assert!(host.load(&dialect_ref(Some(digest), None)).is_empty());
        assert_eq!(host.len(), 1);

        let m = module_using(ty(&[2, 3]));
        // Without the semantics: indeterminate, which is correct and weak.
        let bare = ir::verify(&m, &ir::Context::default());
        assert_eq!(bare.unknown, 1);
        assert!(bare.is_indeterminate());
        assert!(bare.findings[0].message().contains("x.test"));

        // With them: decided.
        let cx = ir::Context {
            semantics: Some(&host),
            ..Default::default()
        };
        let r = ir::verify(&m, &cx);
        assert!(r.is_valid(), "{:?}", r.findings);
        assert_eq!(r.unknown, 0);
        assert_eq!(r.shipped, 1);
        assert!(host.calls.get() >= 1);
        assert!(host.fuel.get() > 0, "a module that ran for no fuel");
    }

    #[test]
    fn a_shipped_shape_function_can_say_the_graph_is_wrong() {
        // The other half, and the half that makes it worth running: a dialect's
        // own semantics disagreeing with the graph's declaration is R-I06, and
        // R-I06 is *invalid* rather than indeterminate.
        let wasm = constant_answer_module("shape", &answer(&[ty(&[2, 3])]), 0);
        let digest = crate::HashAlgo::default().digest(&wasm);
        let objects = |d: &[u8; 32]| (d == &digest).then(|| wasm.clone());
        let mut host = Dialects::new(&objects);
        host.load(&dialect_ref(Some(digest), None));

        let m = module_using(ty(&[2, 4]));
        let cx = ir::Context {
            semantics: Some(&host),
            ..Default::default()
        };
        let r = ir::verify(&m, &cx);
        assert!(r.is_invalid(), "{:?}", r.findings);
        let msg = r.findings[0].message().to_string();
        assert!(msg.contains("shipped shape function"), "{msg}");
        assert!(msg.contains("2×4") && msg.contains("2×3"), "{msg}");
    }

    #[test]
    fn a_verify_function_that_objects_is_invalid_and_one_that_declines_is_not() {
        let shape = constant_answer_module("shape", &answer(&[ty(&[2, 3])]), 0);
        let sd = crate::HashAlgo::default().digest(&shape);

        // A verify function that returns a reason: the op is invalid, and the
        // reason is the dialect's own words.
        let objected = constant_answer_module("verify", b"a thing needs an even width", 0);
        let od = crate::HashAlgo::default().digest(&objected);
        let objects = |d: &[u8; 32]| {
            if d == &sd {
                Some(shape.clone())
            } else if d == &od {
                Some(objected.clone())
            } else {
                None
            }
        };
        let mut host = Dialects::new(&objects);
        host.load(&dialect_ref(Some(sd), Some(od)));
        let m = module_using(ty(&[2, 3]));
        let r = ir::verify(
            &m,
            &ir::Context {
                semantics: Some(&host),
                ..Default::default()
            },
        );
        assert!(r.is_invalid(), "{:?}", r.findings);
        assert!(r.findings[0].message().contains("even width"));

        // A function that declines: indeterminate. A plugin that will not
        // answer says nothing about the graph, and reporting it as invalid
        // would be §15.1's exact prohibition.
        let declines = constant_answer_module("shape", &[], -1);
        let dd = crate::HashAlgo::default().digest(&declines);
        let objects2 = |d: &[u8; 32]| (d == &dd).then(|| declines.clone());
        let mut host2 = Dialects::new(&objects2);
        host2.load(&dialect_ref(Some(dd), None));
        let r2 = ir::verify(
            &m,
            &ir::Context {
                semantics: Some(&host2),
                ..Default::default()
            },
        );
        assert!(!r2.is_invalid(), "{:?}", r2.findings);
        assert!(r2.is_indeterminate());
        assert!(r2.findings[0].message().contains("did not answer"));
    }

    #[test]
    fn a_missing_or_broken_module_is_a_problem_rather_than_a_failure() {
        // A dialect whose module is not in the container is one this reader
        // cannot use, which is a reason to fall back to indeterminate — not a
        // reason to refuse the container.
        let objects = |_: &[u8; 32]| None;
        let mut host = Dialects::new(&objects);
        let problems = host.load(&dialect_ref(Some([9u8; 32]), None));
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("does not have"), "{}", problems[0]);
        assert!(host.is_empty());

        let junk = b"\0asm\x01\0\0\0not a module".to_vec();
        let d = crate::HashAlgo::default().digest(&junk);
        let objects = |x: &[u8; 32]| (x == &d).then(|| junk.clone());
        let mut host = Dialects::new(&objects);
        assert_eq!(host.load(&dialect_ref(Some(d), None)).len(), 1);
        assert!(host.is_empty());
    }
}
