//! §07 — OMNI-IR: the execution graph.
//!
//! A model with tensors but no graph is legal and common (§07.5), and until now
//! that was the only kind this implementation could read. The point of §07 is
//! the *other* kind: a model that describes its own computation, so a runtime
//! that has never heard of its architecture can still execute it.
//!
//! What is here:
//!
//! * The structure of §07.3 — `GraphModule` → `Function` → `Region` → `Block` →
//!   `Op` — in SSA form with densely numbered values, decoded from and encoded
//!   to canonical OMNI-CBOR.
//! * The type system of §07.3.1, including symbolic dimensions and the
//!   per-function constraints over them.
//! * Verification (§15.1 V5) with rule ids, and the three-valued outcome §15.1
//!   insists on: an op from a dialect this build does not know is
//!   **indeterminate**, not invalid — and if the model ships a lowering for it,
//!   it is not even that.
//! * The dialect mechanism of §07.4 with per-op versions, attribute defaults
//!   and shape functions for `omni.core`, `omni.tensor`, `omni.nn` and
//!   `omni.quant`, plus `DialectRef` objects.
//! * Rewrites as data (§07.7): pattern → replacement with side conditions,
//!   used for both op-version migration and dialect lowering. `nn.attention`
//!   lowers to `omni.tensor` primitives through a *shipped rule*, which is the
//!   load-bearing claim of §07.2 and the one thing no other model format can
//!   do.
//! * The fixed-layout binary op array of §07.9, so a large graph parses in one
//!   linear pass.
//! * `synthesize` (§07.5): a weights-only transformer becomes self-describing
//!   from its `arch.params` alone.
//!
//! What is not: WASM `shape_fn`/`verify_fn`/`ref_impl` execution (§11.6 — that
//! is the plugin host, and it is reported unimplemented rather than skipped),
//! machine-level graphs, and autodiff, which §07.10 puts outside the IR on
//! purpose.

use crate::cbor::Value;
use crate::container::Digest;
use crate::dtype::DType;
use crate::expr::{Dim, Ref};

// ------------------------------------------------------------------- errors --

/// A verification finding. §15.1 distinguishes three outcomes, and the
/// distinction is normative: reporting *indeterminate* as *invalid* is itself a
/// conformance violation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    /// The graph breaks a rule. Invalid.
    Invalid { rule: &'static str, msg: String },
    /// The graph may be perfectly good; this reader cannot tell.
    Indeterminate { rule: &'static str, msg: String },
}

impl Finding {
    pub fn rule(&self) -> &'static str {
        match self {
            Finding::Invalid { rule, .. } | Finding::Indeterminate { rule, .. } => rule,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Finding::Invalid { msg, .. } | Finding::Indeterminate { msg, .. } => msg,
        }
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, Finding::Invalid { .. })
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Finding::Invalid { rule, msg } => write!(f, "{rule}: {msg}"),
            Finding::Indeterminate { rule, msg } => write!(f, "{rule}: {msg} (indeterminate)"),
        }
    }
}

fn invalid(rule: &'static str, msg: impl Into<String>) -> Finding {
    Finding::Invalid {
        rule,
        msg: msg.into(),
    }
}

fn unknown(rule: &'static str, msg: impl Into<String>) -> Finding {
    Finding::Indeterminate {
        rule,
        msg: msg.into(),
    }
}

/// A structural decoding error: the object is not an OMNI-IR module at all.
#[derive(Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OMNI-IR: {}", self.0)
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

fn err<T>(msg: impl Into<String>) -> Res<T> {
    Err(Error(msg.into()))
}

// -------------------------------------------------------------------- types --

/// §07.3.1's type grammar.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Tensor {
        shape: Vec<Dim>,
        dtype: DType,
        layout: Option<Value>,
    },
    Tuple(Vec<Type>),
    List(Box<Type>),
    /// Mutable runtime state: a KV cache, an SSM carry.
    State {
        id: String,
        spec: Option<Box<Type>>,
    },
    Stream(Box<Type>),
    /// An ordering token for effects (§07.3.2).
    Token,
    Opaque(String),
}

impl Type {
    pub fn tensor(shape: Vec<Dim>, dtype: DType) -> Type {
        Type::Tensor {
            shape,
            dtype,
            layout: None,
        }
    }

    pub fn as_tensor(&self) -> Option<(&[Dim], &DType)> {
        match self {
            Type::Tensor { shape, dtype, .. } => Some((shape, dtype)),
            _ => None,
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            Type::Tensor {
                shape,
                dtype,
                layout,
            } => {
                let mut p = vec![
                    ("k", Value::text("tensor")),
                    ("shape", crate::expr::shape_to_value(shape)),
                    ("dtype", dtype.to_value()),
                ];
                if let Some(l) = layout {
                    p.push(("layout", l.clone()));
                }
                Value::map(p)
            }
            Type::Tuple(elems) => Value::map(vec![
                ("k", Value::text("tuple")),
                (
                    "elems",
                    Value::Array(elems.iter().map(Type::to_value).collect()),
                ),
            ]),
            Type::List(e) => Value::map(vec![("k", Value::text("list")), ("elem", e.to_value())]),
            Type::State { id, spec } => {
                let mut p = vec![("k", Value::text("state")), ("id", Value::text(id.clone()))];
                if let Some(s) = spec {
                    p.push(("spec", s.to_value()));
                }
                Value::map(p)
            }
            Type::Stream(e) => {
                Value::map(vec![("k", Value::text("stream")), ("elem", e.to_value())])
            }
            Type::Token => Value::map(vec![("k", Value::text("token"))]),
            Type::Opaque(id) => Value::map(vec![
                ("k", Value::text("opaque")),
                ("id", Value::text(id.clone())),
            ]),
        }
    }

    pub fn from_value(v: &Value) -> Res<Type> {
        let k = v
            .get("k")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("a type has no `k`".into()))?;
        match k {
            "tensor" => {
                let shape = match v.get("shape") {
                    Some(s) => crate::expr::parse_shape_value(s)
                        .map_err(|e: crate::expr::Error| Error(e.to_string()))?,
                    None => return err("a tensor type has no `shape`"),
                };
                let dtype = match v.get("dtype") {
                    Some(d) => DType::from_value(d).map_err(Error)?,
                    None => return err("a tensor type has no `dtype`"),
                };
                Ok(Type::Tensor {
                    shape,
                    dtype,
                    layout: v.get("layout").cloned(),
                })
            }
            "tuple" => match v.get("elems") {
                Some(Value::Array(a)) => Ok(Type::Tuple(
                    a.iter().map(Type::from_value).collect::<Res<Vec<_>>>()?,
                )),
                _ => err("a tuple type has no `elems`"),
            },
            "list" => match v.get("elem") {
                Some(e) => Ok(Type::List(Box::new(Type::from_value(e)?))),
                None => err("a list type has no `elem`"),
            },
            "state" => Ok(Type::State {
                id: v
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
                spec: match v.get("spec") {
                    Some(s) => Some(Box::new(Type::from_value(s)?)),
                    None => None,
                },
            }),
            "stream" => match v.get("elem") {
                Some(e) => Ok(Type::Stream(Box::new(Type::from_value(e)?))),
                None => err("a stream type has no `elem`"),
            },
            "token" => Ok(Type::Token),
            "opaque" => Ok(Type::Opaque(
                v.get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
            )),
            other => err(format!("unknown type kind `{other}`")),
        }
    }

    /// The textual form used by the printer: `tensor<B×S×4096, bf16>`.
    pub fn print(&self) -> String {
        match self {
            Type::Tensor { shape, dtype, .. } => {
                let dims: Vec<String> = shape
                    .iter()
                    .map(|d| match d {
                        Dim::N(n) => n.to_string(),
                        Dim::Sym(s) => s.clone(),
                        Dim::Dynamic => "?".into(),
                    })
                    .collect();
                format!("tensor<{}, {}>", dims.join("×"), dtype.label())
            }
            Type::Tuple(e) => format!(
                "tuple<{}>",
                e.iter().map(Type::print).collect::<Vec<_>>().join(", ")
            ),
            Type::List(e) => format!("list<{}>", e.print()),
            Type::State { id, .. } => format!("state<{id}>"),
            Type::Stream(e) => format!("stream<{}>", e.print()),
            Type::Token => "token".into(),
            Type::Opaque(id) => format!("opaque<{id}>"),
        }
    }
}

/// Whether two types are compatible, treating a symbolic dimension as matching
/// anything it could stand for.
///
/// A graph with dynamic shapes is the normal case, not the exception (§07.3.1),
/// so a verifier that demanded literal equality would reject every real model.
pub fn types_agree(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (
            Type::Tensor {
                shape: sa,
                dtype: da,
                ..
            },
            Type::Tensor {
                shape: sb,
                dtype: db,
                ..
            },
        ) => da == db && sa.len() == sb.len() && sa.iter().zip(sb).all(|(x, y)| dims_agree(x, y)),
        (Type::Tuple(x), Type::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| types_agree(a, b))
        }
        (Type::List(x), Type::List(y)) | (Type::Stream(x), Type::Stream(y)) => types_agree(x, y),
        (Type::State { id: x, .. }, Type::State { id: y, .. }) => x == y,
        (Type::Token, Type::Token) => true,
        (Type::Opaque(x), Type::Opaque(y)) => x == y,
        _ => false,
    }
}

pub fn dims_agree(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::N(x), Dim::N(y)) => x == y,
        (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
        (Dim::Sym(x), Dim::Sym(y)) => x == y,
        // A symbolic dimension may take any concrete value.
        (Dim::Sym(_), Dim::N(_)) | (Dim::N(_), Dim::Sym(_)) => true,
    }
}

// ---------------------------------------------------------------- structure --

/// Abstraction level (§07.2). A model may carry the same computation at several,
/// linked by `lowered_from`; only the highest is canonical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Machine,
    Primitive,
    Semantic,
}

impl Level {
    pub fn name(&self) -> &'static str {
        match self {
            Level::Semantic => "semantic",
            Level::Primitive => "primitive",
            Level::Machine => "machine",
        }
    }

    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "semantic" => Some(Level::Semantic),
            "primitive" => Some(Level::Primitive),
            "machine" => Some(Level::Machine),
            _ => None,
        }
    }
}

/// One operation. `d`/`n`/`v` are the `(dialect, name, version)` triple §07.4.1
/// makes the op's identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub dialect: String,
    pub name: String,
    pub version: u32,
    pub inputs: Vec<u32>,
    pub outputs: Vec<(u32, Type)>,
    pub attrs: Vec<(String, Value)>,
    pub regions: Vec<Region>,
    pub loc: Option<String>,
}

impl Op {
    pub fn new(dialect: &str, name: &str, version: u32) -> Op {
        Op {
            dialect: dialect.into(),
            name: name.into(),
            version,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
            loc: None,
        }
    }

    pub fn with_inputs(mut self, ins: &[u32]) -> Op {
        self.inputs = ins.to_vec();
        self
    }

    pub fn with_output(mut self, id: u32, t: Type) -> Op {
        self.outputs.push((id, t));
        self
    }

    pub fn with_attr(mut self, k: &str, v: Value) -> Op {
        self.attrs.push((k.to_string(), v));
        self
    }

    pub fn attr(&self, k: &str) -> Option<&Value> {
        self.attrs.iter().find(|(n, _)| n == k).map(|(_, v)| v)
    }

    /// `omni.nn/attention@2`
    pub fn qualified(&self) -> String {
        format!("{}/{}@{}", self.dialect, self.name, self.version)
    }

    pub fn to_value(&self) -> Value {
        let mut p = vec![
            ("d", Value::text(self.dialect.clone())),
            ("n", Value::text(self.name.clone())),
            ("v", Value::U(self.version as u64)),
            (
                "in",
                Value::Array(self.inputs.iter().map(|i| Value::U(*i as u64)).collect()),
            ),
            (
                "out",
                Value::Array(
                    self.outputs
                        .iter()
                        .map(|(id, t)| Value::Array(vec![Value::U(*id as u64), t.to_value()]))
                        .collect(),
                ),
            ),
        ];
        if !self.attrs.is_empty() {
            p.push((
                "attrs",
                Value::Map(
                    self.attrs
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if !self.regions.is_empty() {
            p.push((
                "regions",
                Value::Array(self.regions.iter().map(Region::to_value).collect()),
            ));
        }
        if let Some(l) = &self.loc {
            p.push(("loc", Value::text(l.clone())));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Op> {
        let dialect = v
            .get("d")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("an op has no dialect `d`".into()))?
            .to_string();
        let name = v
            .get("n")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("an op has no name `n`".into()))?
            .to_string();
        let version = v.get("v").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
        let inputs = match v.get("in") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as u32)
                        .ok_or_else(|| Error("an op input is not a value id".into()))
                })
                .collect::<Res<Vec<_>>>()?,
            None => Vec::new(),
            _ => return err("an op's `in` is not an array"),
        };
        let mut outputs = Vec::new();
        if let Some(Value::Array(a)) = v.get("out") {
            for o in a {
                match o {
                    Value::Array(pair) if pair.len() == 2 => {
                        let id = pair[0]
                            .as_u64()
                            .ok_or_else(|| Error("an op result id is not an integer".into()))?
                            as u32;
                        outputs.push((id, Type::from_value(&pair[1])?));
                    }
                    _ => return err("an op result is not a [id, type] pair"),
                }
            }
        }
        let attrs = match v.get("attrs") {
            Some(Value::Map(m)) => m
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.as_str()
                            .ok_or_else(|| Error("an attribute key is not text".into()))?
                            .to_string(),
                        v.clone(),
                    ))
                })
                .collect::<Res<Vec<_>>>()?,
            _ => Vec::new(),
        };
        let regions = match v.get("regions") {
            Some(Value::Array(a)) => a.iter().map(Region::from_value).collect::<Res<Vec<_>>>()?,
            _ => Vec::new(),
        };
        Ok(Op {
            dialect,
            name,
            version,
            inputs,
            outputs,
            attrs,
            regions,
            loc: v.get("loc").and_then(|x| x.as_str()).map(str::to_string),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Block {
    pub args: Vec<(u32, Type)>,
    pub ops: Vec<Op>,
}

impl Block {
    pub fn to_value(&self) -> Value {
        Value::map(vec![
            (
                "args",
                Value::Array(
                    self.args
                        .iter()
                        .map(|(id, t)| Value::Array(vec![Value::U(*id as u64), t.to_value()]))
                        .collect(),
                ),
            ),
            (
                "ops",
                Value::Array(self.ops.iter().map(Op::to_value).collect()),
            ),
        ])
    }

    pub fn from_value(v: &Value) -> Res<Block> {
        let mut args = Vec::new();
        if let Some(Value::Array(a)) = v.get("args") {
            for x in a {
                match x {
                    Value::Array(pair) if pair.len() == 2 => args.push((
                        pair[0]
                            .as_u64()
                            .ok_or_else(|| Error("a block argument id is not an integer".into()))?
                            as u32,
                        Type::from_value(&pair[1])?,
                    )),
                    _ => return err("a block argument is not a [id, type] pair"),
                }
            }
        }
        let ops = match v.get("ops") {
            Some(Value::Array(a)) => a.iter().map(Op::from_value).collect::<Res<Vec<_>>>()?,
            _ => return err("a block has no `ops`"),
        };
        Ok(Block { args, ops })
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Region {
    pub blocks: Vec<Block>,
}

impl Region {
    pub fn to_value(&self) -> Value {
        Value::Array(self.blocks.iter().map(Block::to_value).collect())
    }

    pub fn from_value(v: &Value) -> Res<Region> {
        match v {
            Value::Array(a) => Ok(Region {
                blocks: a.iter().map(Block::from_value).collect::<Res<Vec<_>>>()?,
            }),
            _ => err("a region is not an array of blocks"),
        }
    }
}

/// A constraint over a symbolic dimension (§07.3.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub dim: String,
    pub rel: Rel,
    pub bound: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rel {
    Le,
    Ge,
    Eq,
    Multiple,
}

impl Rel {
    fn name(&self) -> &'static str {
        match self {
            Rel::Le => "<=",
            Rel::Ge => ">=",
            Rel::Eq => "==",
            Rel::Multiple => "%",
        }
    }

    fn parse(s: &str) -> Option<Rel> {
        match s {
            "<=" => Some(Rel::Le),
            ">=" => Some(Rel::Ge),
            "==" => Some(Rel::Eq),
            "%" => Some(Rel::Multiple),
            _ => None,
        }
    }
}

impl Constraint {
    fn to_value(&self) -> Value {
        Value::map(vec![
            ("dim", Value::text(self.dim.clone())),
            ("rel", Value::text(self.rel.name())),
            ("n", Value::U(self.bound)),
        ])
    }

    fn from_value(v: &Value) -> Res<Constraint> {
        let dim = v
            .get("dim")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("a constraint has no `dim`".into()))?
            .to_string();
        let rel = v
            .get("rel")
            .and_then(|x| x.as_str())
            .and_then(Rel::parse)
            .ok_or_else(|| Error("a constraint has no recognized `rel`".into()))?;
        let bound = v
            .get("n")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| Error("a constraint has no bound `n`".into()))?;
        Ok(Constraint { dim, rel, bound })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub params: Vec<(String, Type)>,
    pub results: Vec<Type>,
    pub attrs: Vec<(String, Value)>,
    pub body: Region,
    pub constraints: Vec<Constraint>,
}

impl Function {
    pub fn to_value(&self) -> Value {
        let mut p = vec![
            (
                "params",
                Value::Array(
                    self.params
                        .iter()
                        .map(|(n, t)| Value::Array(vec![Value::text(n.clone()), t.to_value()]))
                        .collect(),
                ),
            ),
            (
                "results",
                Value::Array(self.results.iter().map(Type::to_value).collect()),
            ),
            ("body", self.body.to_value()),
        ];
        if !self.attrs.is_empty() {
            p.push((
                "attrs",
                Value::Map(
                    self.attrs
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if !self.constraints.is_empty() {
            p.push((
                "constraints",
                Value::Array(self.constraints.iter().map(Constraint::to_value).collect()),
            ));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Function> {
        let mut params = Vec::new();
        if let Some(Value::Array(a)) = v.get("params") {
            for x in a {
                match x {
                    Value::Array(pair) if pair.len() == 2 => params.push((
                        pair[0]
                            .as_str()
                            .ok_or_else(|| Error("a parameter name is not text".into()))?
                            .to_string(),
                        Type::from_value(&pair[1])?,
                    )),
                    _ => return err("a parameter is not a [name, type] pair"),
                }
            }
        }
        let results = match v.get("results") {
            Some(Value::Array(a)) => a.iter().map(Type::from_value).collect::<Res<Vec<_>>>()?,
            _ => Vec::new(),
        };
        let body = match v.get("body") {
            Some(b) => Region::from_value(b)?,
            None => return err("a function has no `body`"),
        };
        let attrs = match v.get("attrs") {
            Some(Value::Map(m)) => m
                .iter()
                .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                .collect(),
            _ => Vec::new(),
        };
        let constraints = match v.get("constraints") {
            Some(Value::Array(a)) => a
                .iter()
                .map(Constraint::from_value)
                .collect::<Res<Vec<_>>>()?,
            _ => Vec::new(),
        };
        Ok(Function {
            params,
            results,
            attrs,
            body,
            constraints,
        })
    }
}

/// A dialect this module uses, with the version it was written against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DialectUse {
    pub ns: String,
    pub version: u32,
    /// The `DialectRef` object carrying the definition, when embedded.
    pub reference: Option<Ref>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub level: Level,
    pub dialects: Vec<DialectUse>,
    pub attrs: Vec<(String, Value)>,
    pub functions: Vec<(String, Function)>,
    pub entry: String,
    /// For a derived (lowered) module: the module it came from (§07.2).
    pub lowered_from: Option<Ref>,
    /// Rewrites the model ships so a reader can lower what it does not know
    /// (§07.7). These are refs to `omni.ir/rewrite` objects.
    pub rewrites: Vec<Ref>,
}

pub const SCHEMA: &str = "omni.ir/module";

impl Module {
    pub fn new(level: Level, entry: &str) -> Module {
        Module {
            level,
            dialects: Vec::new(),
            attrs: Vec::new(),
            functions: Vec::new(),
            entry: entry.to_string(),
            lowered_from: None,
            rewrites: Vec::new(),
        }
    }

    pub fn function(&self, name: &str) -> Option<&Function> {
        self.functions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| f)
    }

    pub fn to_value(&self) -> Value {
        let mut p = vec![
            ("t", Value::text(SCHEMA)),
            ("v", Value::U(1)),
            ("level", Value::text(self.level.name())),
            (
                "dialects",
                Value::Array(
                    self.dialects
                        .iter()
                        .map(|d| {
                            let mut q = vec![
                                ("ns", Value::text(d.ns.clone())),
                                ("version", Value::U(d.version as u64)),
                            ];
                            if let Some(r) = &d.reference {
                                q.push(("ref", ref_value(r)));
                            }
                            Value::map(q)
                        })
                        .collect(),
                ),
            ),
            (
                "functions",
                Value::Map(
                    self.functions
                        .iter()
                        .map(|(n, f)| (Value::text(n.clone()), f.to_value()))
                        .collect(),
                ),
            ),
            ("entry", Value::text(self.entry.clone())),
        ];
        if !self.attrs.is_empty() {
            p.push((
                "attrs",
                Value::Map(
                    self.attrs
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if let Some(r) = &self.lowered_from {
            p.push(("lowered_from", ref_value(r)));
        }
        if !self.rewrites.is_empty() {
            p.push((
                "rewrites",
                Value::Array(self.rewrites.iter().map(ref_value).collect()),
            ));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Module> {
        match v.get("t").and_then(|x| x.as_str()) {
            Some(SCHEMA) => {}
            Some(other) => return err(format!("`{other}` is not an OMNI-IR module")),
            None => return err("no schema `t`"),
        }
        let level = v
            .get("level")
            .and_then(|x| x.as_str())
            .and_then(Level::parse)
            .ok_or_else(|| Error("a module has no recognized `level`".into()))?;
        let mut dialects = Vec::new();
        if let Some(Value::Array(a)) = v.get("dialects") {
            for d in a {
                let ns = d
                    .get("ns")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error("a dialect use has no `ns`".into()))?
                    .to_string();
                let version = d.get("version").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
                let reference = d.get("ref").and_then(parse_ref);
                dialects.push(DialectUse {
                    ns,
                    version,
                    reference,
                });
            }
        }
        let functions = match v.get("functions") {
            Some(Value::Map(m)) => {
                let mut out = Vec::new();
                for (k, f) in m {
                    let name = k
                        .as_str()
                        .ok_or_else(|| Error("a function name is not text".into()))?;
                    out.push((name.to_string(), Function::from_value(f)?));
                }
                out
            }
            _ => return err("a module has no `functions`"),
        };
        let entry = v
            .get("entry")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("a module has no `entry`".into()))?
            .to_string();
        let attrs = match v.get("attrs") {
            Some(Value::Map(m)) => m
                .iter()
                .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                .collect(),
            _ => Vec::new(),
        };
        let rewrites = match v.get("rewrites") {
            Some(Value::Array(a)) => a.iter().filter_map(parse_ref).collect(),
            _ => Vec::new(),
        };
        Ok(Module {
            level,
            dialects,
            attrs,
            functions,
            entry,
            lowered_from: v.get("lowered_from").and_then(parse_ref),
            rewrites,
        })
    }

    /// Every op in the module, outermost first, with the function it belongs to.
    pub fn ops(&self) -> Vec<(&str, &Op)> {
        let mut out = Vec::new();
        for (name, f) in &self.functions {
            collect_ops(&f.body, name.as_str(), &mut out);
        }
        out
    }

    pub fn op_count(&self) -> usize {
        self.ops().len()
    }

    /// The highest value id used anywhere, for allocating fresh ones.
    pub fn max_value_id(&self) -> u32 {
        let mut max = 0;
        for (_, f) in &self.functions {
            max = max.max(region_max_id(&f.body));
        }
        max
    }

    /// Whether the module declares a dialect, at any version.
    pub fn declares(&self, ns: &str) -> Option<u32> {
        self.dialects.iter().find(|d| d.ns == ns).map(|d| d.version)
    }
}

fn collect_ops<'a>(r: &'a Region, f: &'a str, out: &mut Vec<(&'a str, &'a Op)>) {
    for b in &r.blocks {
        for op in &b.ops {
            out.push((f, op));
            for sub in &op.regions {
                collect_ops(sub, f, out);
            }
        }
    }
}

fn region_max_id(r: &Region) -> u32 {
    let mut max = 0;
    for b in &r.blocks {
        for (id, _) in &b.args {
            max = max.max(*id);
        }
        for op in &b.ops {
            for (id, _) in &op.outputs {
                max = max.max(*id);
            }
            for sub in &op.regions {
                max = max.max(region_max_id(sub));
            }
        }
    }
    max
}

fn ref_value(r: &Ref) -> Value {
    Value::Array(vec![Value::U(r.0 as u64), Value::Bytes(r.1.to_vec())])
}

fn parse_ref(v: &Value) -> Option<Ref> {
    match v {
        Value::Array(a) if a.len() == 2 => {
            let t = a[0].as_u64()? as u16;
            match &a[1] {
                Value::Bytes(b) if b.len() == 32 => {
                    let mut d: Digest = [0; 32];
                    d.copy_from_slice(b);
                    Some((t, d))
                }
                _ => None,
            }
        }
        Value::Tag(1001, inner) => parse_ref(inner),
        _ => None,
    }
}

// ------------------------------------------------------------------ printing --

impl Module {
    /// A textual rendering, for `omni graph`.
    ///
    /// Not a parseable surface syntax and deliberately not one: OMNI-IR's
    /// interchange form is the CBOR, and a second normative syntax would be a
    /// second thing to keep in agreement. This exists so a human can read a
    /// graph.
    pub fn print(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "module @{} level={}\n",
            self.entry,
            self.level.name()
        ));
        for d in &self.dialects {
            s.push_str(&format!(
                "  dialect {}@{}{}\n",
                d.ns,
                d.version,
                if d.reference.is_some() {
                    " (embedded)"
                } else {
                    ""
                }
            ));
        }
        if let Some(r) = &self.lowered_from {
            s.push_str(&format!(
                "  lowered_from {}\n",
                &crate::sha256::hex(&r.1)[..16]
            ));
        }
        for (name, f) in &self.functions {
            let params: Vec<String> = f
                .params
                .iter()
                .map(|(n, t)| format!("{n}: {}", t.print()))
                .collect();
            let results: Vec<String> = f.results.iter().map(Type::print).collect();
            s.push_str(&format!(
                "\n  func @{name}({}) -> ({})\n",
                params.join(", "),
                results.join(", ")
            ));
            for c in &f.constraints {
                s.push_str(&format!(
                    "    constraint {} {} {}\n",
                    c.dim,
                    c.rel.name(),
                    c.bound
                ));
            }
            print_region(&f.body, 4, &mut s);
        }
        s
    }
}

fn print_region(r: &Region, indent: usize, s: &mut String) {
    let pad = " ".repeat(indent);
    for (i, b) in r.blocks.iter().enumerate() {
        if r.blocks.len() > 1 || !b.args.is_empty() {
            let args: Vec<String> = b
                .args
                .iter()
                .map(|(id, t)| format!("%{id}: {}", t.print()))
                .collect();
            s.push_str(&format!("{pad}^bb{i}({}):\n", args.join(", ")));
        }
        for op in &b.ops {
            let outs: Vec<String> = op.outputs.iter().map(|(id, _)| format!("%{id}")).collect();
            let ins: Vec<String> = op.inputs.iter().map(|i| format!("%{i}")).collect();
            let attrs: Vec<String> = op
                .attrs
                .iter()
                .map(|(k, v)| format!("{k}={}", short_value(v)))
                .collect();
            let lhs = if outs.is_empty() {
                String::new()
            } else {
                format!("{} = ", outs.join(", "))
            };
            let attr_str = if attrs.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", attrs.join(", "))
            };
            let ty = match op.outputs.first() {
                Some((_, t)) => format!(" : {}", t.print()),
                None => String::new(),
            };
            s.push_str(&format!(
                "{pad}{lhs}{} {}{attr_str}{ty}\n",
                op.qualified(),
                ins.join(", ")
            ));
            for sub in &op.regions {
                s.push_str(&format!("{pad}  region:\n"));
                print_region(sub, indent + 4, s);
            }
        }
    }
}

fn short_value(v: &Value) -> String {
    match v {
        Value::U(n) => n.to_string(),
        Value::I(n) => n.to_string(),
        Value::F64(f) => format!("{f}"),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".into(),
        Value::Text(t) => format!("\"{t}\""),
        Value::Bytes(b) => format!("h'{}'", crate::sha256::hex(b)),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(short_value).collect::<Vec<_>>().join(",")
        ),
        Value::Map(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}:{}", short_value(k), short_value(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Tag(t, inner) => format!("{t}({})", short_value(inner)),
    }
}

// ------------------------------------------------------------------ dialects --

/// What kind of value an attribute holds. Enough to check a graph without
/// running a WASM `verify_fn` (§07.4.2), which is the thorough answer this
/// build cannot give yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrKind {
    Int,
    Float,
    Bool,
    Text,
    IntList,
    Any,
}

impl AttrKind {
    fn accepts(&self, v: &Value) -> bool {
        match self {
            AttrKind::Int => matches!(v, Value::U(_) | Value::I(_)),
            AttrKind::Float => matches!(v, Value::F64(_) | Value::U(_) | Value::I(_)),
            AttrKind::Bool => matches!(v, Value::Bool(_)),
            AttrKind::Text => matches!(v, Value::Text(_)),
            AttrKind::IntList => {
                matches!(v, Value::Array(a) if a.iter().all(|x| matches!(x, Value::U(_) | Value::I(_))))
            }
            AttrKind::Any => true,
        }
    }
}

/// One operator's contract: how many inputs and results, how many regions, and
/// which attributes it understands.
#[derive(Clone, Debug)]
pub struct OpSpec {
    pub name: &'static str,
    /// Every version this dialect defines, ascending (§07.4.1).
    pub versions: &'static [u32],
    /// `(minimum, maximum)` inputs; `None` means variadic.
    pub inputs: (usize, Option<usize>),
    pub results: usize,
    pub regions: usize,
    /// Whether the op touches state, RNG or collectives (§07.3.2).
    pub effectful: bool,
    /// `(name, kind, required)`. An attribute that is not required has a
    /// specified default, which is why adding one does not bump the version.
    pub attrs: &'static [(&'static str, AttrKind, bool)],
}

#[derive(Clone, Debug)]
pub struct Dialect {
    pub ns: &'static str,
    pub version: u32,
    /// `omni.core` is frozen for the life of OMNI/1.x (§07.4).
    pub frozen: bool,
    pub ops: &'static [OpSpec],
}

impl Dialect {
    pub fn op(&self, name: &str) -> Option<&OpSpec> {
        self.ops.iter().find(|o| o.name == name)
    }
}

macro_rules! op {
    ($name:expr, $versions:expr, $inmin:expr, $inmax:expr, $out:expr, $reg:expr, $eff:expr,
     [$(($a:expr, $k:expr, $req:expr)),*]) => {
        OpSpec {
            name: $name,
            versions: $versions,
            inputs: ($inmin, $inmax),
            results: $out,
            regions: $reg,
            effectful: $eff,
            attrs: &[$(($a, $k, $req)),*],
        }
    };
}

/// `omni.core` — regions, control flow and calls, and no tensor mathematics at
/// all. §07.8's load-bearing claim (a 2040 architecture needs a new dialect, not
/// a new format) rests on that emptiness, so this list is frozen.
static CORE_OPS: &[OpSpec] = &[
    op!(
        "constant",
        &[1],
        0,
        Some(0),
        1,
        0,
        false,
        [
            ("value", AttrKind::Any, false),
            ("tensor", AttrKind::Text, false)
        ]
    ),
    op!(
        "call",
        &[1],
        0,
        None,
        1,
        0,
        false,
        [("callee", AttrKind::Text, true)]
    ),
    op!("return", &[1], 0, None, 0, 0, false, []),
    op!("yield", &[1], 0, None, 0, 0, false, []),
    op!("if", &[1], 1, Some(1), 1, 2, false, []),
    op!("while", &[1], 0, None, 1, 2, false, []),
    // Two results, not one: the threaded carry as it ended, and the emissions
    // stacked along the scanned axis — which is what `scan` means everywhere it
    // exists and what this build's interpreter has always returned. The spec
    // said one until a synthesized LSTM used both and verification disagreed
    // with execution about the same graph. See `docs/spec/07-graph.md` §7.3.
    op!(
        "scan",
        &[1],
        1,
        None,
        2,
        1,
        false,
        [
            ("axis", AttrKind::Int, false),
            ("reverse", AttrKind::Bool, false)
        ]
    ),
    op!(
        "map",
        &[1],
        1,
        None,
        1,
        1,
        false,
        [("axis", AttrKind::Int, false)]
    ),
    op!("region", &[1], 0, None, 1, 1, false, []),
    op!("tuple", &[1], 0, None, 1, 0, false, []),
    op!(
        "get",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("index", AttrKind::Int, true)]
    ),
    op!(
        "assert",
        &[1],
        1,
        Some(2),
        0,
        0,
        false,
        [("message", AttrKind::Text, false)]
    ),
    op!(
        "debug",
        &[1],
        0,
        None,
        0,
        0,
        false,
        [("label", AttrKind::Text, false)]
    ),
    op!(
        "func",
        &[1],
        0,
        None,
        0,
        1,
        false,
        [("name", AttrKind::Text, true)]
    ),
];

/// `omni.tensor` — the primitive level.
static TENSOR_OPS: &[OpSpec] = &[
    op!("add", &[1], 2, Some(2), 1, 0, false, []),
    op!("sub", &[1], 2, Some(2), 1, 0, false, []),
    op!("mul", &[1], 2, Some(2), 1, 0, false, []),
    op!("div", &[1], 2, Some(2), 1, 0, false, []),
    op!("neg", &[1], 1, Some(1), 1, 0, false, []),
    op!("exp", &[1], 1, Some(1), 1, 0, false, []),
    op!("log", &[1], 1, Some(1), 1, 0, false, []),
    op!("sqrt", &[1], 1, Some(1), 1, 0, false, []),
    op!("rsqrt", &[1], 1, Some(1), 1, 0, false, []),
    op!("tanh", &[1], 1, Some(1), 1, 0, false, []),
    op!("sigmoid", &[1], 1, Some(1), 1, 0, false, []),
    op!("erf", &[1], 1, Some(1), 1, 0, false, []),
    op!("maximum", &[1], 2, Some(2), 1, 0, false, []),
    op!("minimum", &[1], 2, Some(2), 1, 0, false, []),
    op!(
        "matmul",
        &[1],
        2,
        Some(2),
        1,
        0,
        false,
        [
            ("accum", AttrKind::Text, false),
            ("math", AttrKind::Text, false)
        ]
    ),
    op!(
        "einsum",
        &[1],
        1,
        None,
        1,
        0,
        false,
        [("equation", AttrKind::Text, true)]
    ),
    op!(
        "reduce",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("kind", AttrKind::Text, true),
            ("axes", AttrKind::IntList, true),
            ("keepdims", AttrKind::Bool, false)
        ]
    ),
    op!(
        "softmax",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("axis", AttrKind::Int, false)]
    ),
    op!(
        "slice",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("start", AttrKind::IntList, true),
            ("stop", AttrKind::IntList, true),
            ("step", AttrKind::IntList, false)
        ]
    ),
    op!(
        "concat",
        &[1],
        1,
        None,
        1,
        0,
        false,
        [("axis", AttrKind::Int, true)]
    ),
    op!(
        "pad",
        &[1],
        1,
        Some(2),
        1,
        0,
        false,
        [
            ("low", AttrKind::IntList, true),
            ("high", AttrKind::IntList, true),
            ("mode", AttrKind::Text, false)
        ]
    ),
    op!(
        "transpose",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("perm", AttrKind::IntList, true)]
    ),
    op!(
        "reshape",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("shape", AttrKind::IntList, true)]
    ),
    op!(
        "broadcast",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("shape", AttrKind::IntList, true)]
    ),
    op!(
        "gather",
        &[1],
        2,
        Some(2),
        1,
        0,
        false,
        [("axis", AttrKind::Int, false)]
    ),
    op!(
        "scatter",
        &[1],
        3,
        Some(3),
        1,
        0,
        false,
        [("axis", AttrKind::Int, false)]
    ),
    op!(
        "cumsum",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [("axis", AttrKind::Int, false)]
    ),
    op!(
        "sort",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("axis", AttrKind::Int, false),
            ("descending", AttrKind::Bool, false)
        ]
    ),
    op!(
        "topk",
        &[1],
        1,
        Some(1),
        2,
        0,
        false,
        [("k", AttrKind::Int, true), ("axis", AttrKind::Int, false)]
    ),
    op!(
        "cast",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("dtype", AttrKind::Any, true),
            ("round", AttrKind::Text, false)
        ]
    ),
    op!("where", &[1], 3, Some(3), 1, 0, false, []),
];

/// `omni.nn` — the semantic level. A registry dialect: versioned, extensible,
/// and never load-bearing for the format itself.
static NN_OPS: &[OpSpec] = &[
    op!(
        "attention",
        &[1, 2],
        3,
        Some(5),
        1,
        0,
        false,
        [
            ("causal", AttrKind::Bool, false),
            ("window", AttrKind::Any, false),
            ("softcap", AttrKind::Float, false),
            ("kv_groups", AttrKind::Int, false),
            ("scale", AttrKind::Float, false)
        ]
    ),
    op!(
        "norm",
        &[1],
        1,
        Some(3),
        1,
        0,
        false,
        [
            ("kind", AttrKind::Text, true),
            ("eps", AttrKind::Float, false),
            ("axis", AttrKind::Int, false)
        ]
    ),
    op!(
        "rope",
        &[1],
        1,
        Some(2),
        1,
        0,
        false,
        [
            ("theta", AttrKind::Float, false),
            ("interleaved", AttrKind::Bool, true)
        ]
    ),
    op!(
        "activation",
        &[1],
        1,
        Some(2),
        1,
        0,
        false,
        [("kind", AttrKind::Text, true)]
    ),
    op!("embedding", &[1], 2, Some(2), 1, 0, false, []),
    op!(
        "moe_route",
        &[1],
        2,
        Some(2),
        2,
        0,
        false,
        [
            ("top_k", AttrKind::Int, true),
            ("normalize", AttrKind::Bool, false)
        ]
    ),
    op!(
        "ssm_scan",
        &[1],
        3,
        Some(5),
        1,
        0,
        false,
        [("delta_softplus", AttrKind::Bool, false)]
    ),
    op!(
        "conv1d_causal",
        &[1],
        2,
        Some(3),
        1,
        0,
        false,
        [("groups", AttrKind::Int, false)]
    ),
    op!(
        "conv",
        &[1],
        2,
        Some(3),
        1,
        0,
        false,
        [
            ("stride", AttrKind::IntList, false),
            ("padding", AttrKind::IntList, false),
            ("dilation", AttrKind::IntList, false),
            ("groups", AttrKind::Int, false)
        ]
    ),
    op!(
        "pool",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("kind", AttrKind::Text, true),
            ("window", AttrKind::IntList, true),
            ("stride", AttrKind::IntList, false)
        ]
    ),
    op!(
        "interpolate",
        &[1],
        1,
        Some(1),
        1,
        0,
        false,
        [
            ("mode", AttrKind::Text, true),
            ("scale", AttrKind::Any, false)
        ]
    ),
];

/// `omni.quant` — quantization as ops, for graphs that carry it explicitly.
static QUANT_OPS: &[OpSpec] = &[
    op!(
        "quantize",
        &[1],
        1,
        Some(3),
        1,
        0,
        false,
        [("scheme", AttrKind::Any, true)]
    ),
    op!(
        "dequantize",
        &[1],
        1,
        Some(3),
        1,
        0,
        false,
        [("scheme", AttrKind::Any, true)]
    ),
    op!(
        "qmatmul",
        &[1],
        2,
        Some(4),
        1,
        0,
        false,
        [("scheme", AttrKind::Any, true)]
    ),
    op!(
        "fake_quant",
        &[1],
        1,
        Some(3),
        1,
        0,
        false,
        [("scheme", AttrKind::Any, true)]
    ),
];

/// `omni.io` — where a model meets the world: inputs, outputs, and external
/// calls that are visible *because* they are declared effectful (§07.8).
static IO_OPS: &[OpSpec] = &[
    op!(
        "input",
        &[1],
        0,
        Some(0),
        1,
        0,
        false,
        [("name", AttrKind::Text, true)]
    ),
    op!(
        "output",
        &[1],
        1,
        Some(1),
        0,
        0,
        false,
        [("name", AttrKind::Text, true)]
    ),
    op!(
        "external",
        &[1],
        0,
        None,
        1,
        0,
        true,
        [("id", AttrKind::Text, true)]
    ),
];

static DIALECTS: &[Dialect] = &[
    Dialect {
        ns: "omni.core",
        version: 1,
        frozen: true,
        ops: CORE_OPS,
    },
    Dialect {
        ns: "omni.tensor",
        version: 1,
        frozen: false,
        ops: TENSOR_OPS,
    },
    Dialect {
        ns: "omni.nn",
        version: 1,
        frozen: false,
        ops: NN_OPS,
    },
    Dialect {
        ns: "omni.quant",
        version: 1,
        frozen: false,
        ops: QUANT_OPS,
    },
    Dialect {
        ns: "omni.io",
        version: 1,
        frozen: false,
        ops: IO_OPS,
    },
];

/// The dialects this build knows. Anything else is *unknown*, which §11.3 makes
/// a reason to refuse execution and nothing more.
pub fn dialects() -> &'static [Dialect] {
    DIALECTS
}

pub fn dialect(ns: &str) -> Option<&'static Dialect> {
    DIALECTS.iter().find(|d| d.ns == ns)
}

pub fn op_spec(ns: &str, name: &str) -> Option<&'static OpSpec> {
    dialect(ns)?.op(name)
}

/// A `DialectRef` object (§07.4.2) for one of the dialects above.
///
/// The WASM `shape_fn` / `verify_fn` / `ref_impl` slots are deliberately absent
/// rather than empty: this build has no plugin host, and a `DialectRef` that
/// claimed to carry executable semantics it cannot run would be a lie a reader
/// could not detect.
pub fn dialect_ref_value(d: &Dialect) -> Value {
    let ops: Vec<(Value, Value)> = d
        .ops
        .iter()
        .map(|o| {
            let mut spec = vec![
                (
                    "versions",
                    Value::Array(o.versions.iter().map(|v| Value::U(*v as u64)).collect()),
                ),
                (
                    "inputs",
                    Value::Array(vec![
                        Value::U(o.inputs.0 as u64),
                        match o.inputs.1 {
                            Some(n) => Value::U(n as u64),
                            None => Value::Null,
                        },
                    ]),
                ),
                ("results", Value::U(o.results as u64)),
                ("regions", Value::U(o.regions as u64)),
            ];
            if o.effectful {
                spec.push(("effects", Value::Bool(true)));
            }
            if !o.attrs.is_empty() {
                spec.push((
                    "attrs",
                    Value::Map(
                        o.attrs
                            .iter()
                            .map(|(n, k, req)| {
                                (
                                    Value::text(*n),
                                    Value::map(vec![
                                        (
                                            "t",
                                            Value::text(match k {
                                                AttrKind::Int => "int",
                                                AttrKind::Float => "f64",
                                                AttrKind::Bool => "bool",
                                                AttrKind::Text => "text",
                                                AttrKind::IntList => "int[]",
                                                AttrKind::Any => "any",
                                            }),
                                        ),
                                        ("required", Value::Bool(*req)),
                                    ]),
                                )
                            })
                            .collect(),
                    ),
                ));
            }
            (Value::text(o.name), Value::map(spec))
        })
        .collect();
    let mut p = vec![
        ("t", Value::text("omni.ir/dialect")),
        ("v", Value::U(1)),
        ("ns", Value::text(d.ns)),
        ("version", Value::U(d.version as u64)),
        ("frozen", Value::Bool(d.frozen)),
        ("ops", Value::Map(ops)),
    ];
    if d.ns != "omni.core" {
        p.push((
            "requires",
            Value::Array(vec![Value::map(vec![
                ("ns", Value::text("omni.core")),
                ("version", Value::U(1)),
            ])]),
        ));
    }
    Value::map(p)
}

// ----------------------------------------------------------------- inference --

/// The result of asking an op what its results are.
#[derive(Clone, Debug, PartialEq)]
pub enum Inferred {
    /// The op's result types, computed from its inputs.
    Types(Vec<Type>),
    /// A known op with no shape function in this build: unchecked, not wrong.
    Unchecked(String),
    /// The inputs cannot produce any result: the graph is broken.
    Ill(String),
}

fn broadcast(a: &[Dim], b: &[Dim]) -> Result<Vec<Dim>, String> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(a.len().wrapping_sub(n - i)).cloned();
        let y = b.get(b.len().wrapping_sub(n - i)).cloned();
        let d = match (x, y) {
            (Some(Dim::N(1)), Some(other)) | (Some(other), Some(Dim::N(1))) => other,
            (Some(p), Some(q)) => {
                if !dims_agree(&p, &q) {
                    return Err(format!(
                        "cannot broadcast {} against {}",
                        dim_str(&p),
                        dim_str(&q)
                    ));
                }
                // A concrete dimension is more informative than a symbolic one.
                if matches!(p, Dim::N(_)) {
                    p
                } else {
                    q
                }
            }
            (Some(p), None) => p,
            (None, Some(q)) => q,
            (None, None) => unreachable!("index within the longer shape"),
        };
        out.push(d);
    }
    Ok(out)
}

fn dim_str(d: &Dim) -> String {
    match d {
        Dim::N(n) => n.to_string(),
        Dim::Sym(s) => s.clone(),
        Dim::Dynamic => "?".into(),
    }
}

fn axes_of(op: &Op, rank: usize) -> Result<Vec<usize>, String> {
    match op.attr("axes") {
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| match x {
                Value::U(n) if (*n as usize) < rank => Ok(*n as usize),
                Value::I(n) => {
                    let k = rank as i64 + *n;
                    if k >= 0 && (k as usize) < rank {
                        Ok(k as usize)
                    } else {
                        Err(format!("axis {n} is outside a rank-{rank} tensor"))
                    }
                }
                other => Err(format!("axis {} is out of range", short_value(other))),
            })
            .collect(),
        _ => Err("`axes` is missing".into()),
    }
}

fn int_list(op: &Op, key: &str) -> Option<Vec<i64>> {
    match op.attr(key) {
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| match x {
                Value::U(n) => Some(*n as i64),
                Value::I(n) => Some(*n),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn int_attr(op: &Op, key: &str) -> Option<i64> {
    match op.attr(key) {
        Some(Value::U(n)) => Some(*n as i64),
        Some(Value::I(n)) => Some(*n),
        _ => None,
    }
}

/// Shape and dtype inference for the ops this build understands (§07.4.2's
/// `shape_fn`, in Rust rather than WASM).
///
/// Getting this wrong in the permissive direction would be worse than not
/// having it: a verifier that accepts a graph whose shapes do not line up has
/// told the user their model is fine when it is not. So anything not modelled
/// here is [`Inferred::Unchecked`] and says so.
pub fn infer(op: &Op, ins: &[Type]) -> Inferred {
    let tensor = |i: usize| -> Result<(Vec<Dim>, DType), String> {
        match ins.get(i) {
            Some(Type::Tensor { shape, dtype, .. }) => Ok((shape.clone(), dtype.clone())),
            Some(other) => Err(format!("input {i} is {}, not a tensor", other.print())),
            None => Err(format!("input {i} is missing")),
        }
    };
    let elementwise_unary = || -> Inferred {
        match tensor(0) {
            Ok((s, d)) => Inferred::Types(vec![Type::tensor(s, d)]),
            Err(e) => Inferred::Ill(e),
        }
    };
    let elementwise_binary = || -> Inferred {
        match (tensor(0), tensor(1)) {
            (Ok((a, da)), Ok((b, db))) => {
                if da != db {
                    return Inferred::Ill(format!(
                        "elementwise operands differ in dtype: {} and {}",
                        da.label(),
                        db.label()
                    ));
                }
                match broadcast(&a, &b) {
                    Ok(s) => Inferred::Types(vec![Type::tensor(s, da)]),
                    Err(e) => Inferred::Ill(e),
                }
            }
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        }
    };

    match (op.dialect.as_str(), op.name.as_str()) {
        ("omni.tensor", "add" | "sub" | "mul" | "div" | "maximum" | "minimum") => {
            elementwise_binary()
        }
        (
            "omni.tensor",
            "neg" | "exp" | "log" | "sqrt" | "rsqrt" | "tanh" | "sigmoid" | "erf" | "softmax",
        ) => elementwise_unary(),
        ("omni.tensor", "cast") => match (tensor(0), op.attr("dtype")) {
            (Ok((s, _)), Some(d)) => match DType::from_value(d) {
                Ok(dt) => Inferred::Types(vec![Type::tensor(s, dt)]),
                Err(e) => Inferred::Ill(e),
            },
            (Err(e), _) => Inferred::Ill(e),
            (_, None) => Inferred::Ill("cast has no `dtype`".into()),
        },
        ("omni.tensor", "matmul") => match (tensor(0), tensor(1)) {
            (Ok((a, da)), Ok((b, db))) => {
                if da != db {
                    return Inferred::Ill(format!(
                        "matmul operands differ in dtype: {} and {}",
                        da.label(),
                        db.label()
                    ));
                }
                if a.len() < 2 || b.len() < 2 {
                    return Inferred::Ill("matmul needs rank ≥ 2 operands".into());
                }
                let (m, k1) = (a[a.len() - 2].clone(), a[a.len() - 1].clone());
                let (k2, n) = (b[b.len() - 2].clone(), b[b.len() - 1].clone());
                if !dims_agree(&k1, &k2) {
                    return Inferred::Ill(format!(
                        "matmul inner dimensions disagree: {} vs {}",
                        dim_str(&k1),
                        dim_str(&k2)
                    ));
                }
                let batch = match broadcast(&a[..a.len() - 2], &b[..b.len() - 2]) {
                    Ok(s) => s,
                    Err(e) => return Inferred::Ill(e),
                };
                let mut shape = batch;
                shape.push(m);
                shape.push(n);
                Inferred::Types(vec![Type::tensor(shape, da)])
            }
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        },
        ("omni.tensor", "reduce") => match tensor(0) {
            Ok((s, d)) => {
                let axes = match axes_of(op, s.len()) {
                    Ok(a) => a,
                    Err(e) => return Inferred::Ill(e),
                };
                let keep = matches!(op.attr("keepdims"), Some(Value::Bool(true)));
                let mut out = Vec::new();
                for (i, dim) in s.iter().enumerate() {
                    if axes.contains(&i) {
                        if keep {
                            out.push(Dim::N(1));
                        }
                    } else {
                        out.push(dim.clone());
                    }
                }
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            Err(e) => Inferred::Ill(e),
        },
        ("omni.tensor", "transpose") => match tensor(0) {
            Ok((s, d)) => {
                let Some(perm) = int_list(op, "perm") else {
                    return Inferred::Ill("transpose has no `perm`".into());
                };
                if perm.len() != s.len() {
                    return Inferred::Ill(format!(
                        "`perm` has {} entries for a rank-{} tensor",
                        perm.len(),
                        s.len()
                    ));
                }
                let mut seen = vec![false; s.len()];
                let mut out = Vec::with_capacity(s.len());
                for p in &perm {
                    let Ok(i) = usize::try_from(*p) else {
                        return Inferred::Ill("`perm` has a negative axis".into());
                    };
                    if i >= s.len() || seen[i] {
                        return Inferred::Ill("`perm` is not a permutation".into());
                    }
                    seen[i] = true;
                    out.push(s[i].clone());
                }
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            Err(e) => Inferred::Ill(e),
        },
        ("omni.tensor", "reshape" | "broadcast") => match tensor(0) {
            Ok((s, d)) => {
                let Some(shape) = int_list(op, "shape") else {
                    return Inferred::Ill(format!("{} has no `shape`", op.name));
                };
                // A -1 means "whatever is left", which is only computable when
                // every other dimension is concrete.
                let known: Option<u64> = s.iter().map(|x| x.size()).product::<Option<u64>>();
                let mut out: Vec<Dim> = Vec::with_capacity(shape.len());
                let mut infer_at = None;
                let mut fixed: u64 = 1;
                for (i, n) in shape.iter().enumerate() {
                    if *n == -1 {
                        if infer_at.is_some() {
                            return Inferred::Ill("`shape` has more than one -1".into());
                        }
                        infer_at = Some(i);
                        out.push(Dim::Dynamic);
                    } else if *n < 0 {
                        return Inferred::Ill(format!("`shape` has a negative dimension {n}"));
                    } else {
                        fixed = fixed.saturating_mul(*n as u64);
                        out.push(Dim::N(*n as u64));
                    }
                }
                if let (Some(i), Some(total)) = (infer_at, known) {
                    if fixed == 0 || total % fixed != 0 {
                        return Inferred::Ill(format!(
                            "cannot reshape {total} elements into the declared shape"
                        ));
                    }
                    out[i] = Dim::N(total / fixed);
                }
                if op.name == "reshape" && infer_at.is_none() {
                    if let Some(total) = known {
                        if total != fixed {
                            return Inferred::Ill(format!(
                                "reshape changes the element count: {total} to {fixed}"
                            ));
                        }
                    }
                }
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            Err(e) => Inferred::Ill(e),
        },
        ("omni.tensor", "slice") => match tensor(0) {
            Ok((s, d)) => {
                let (Some(start), Some(stop)) = (int_list(op, "start"), int_list(op, "stop"))
                else {
                    return Inferred::Ill("slice needs `start` and `stop`".into());
                };
                if start.len() != s.len() || stop.len() != s.len() {
                    return Inferred::Ill("slice bounds do not match the operand rank".into());
                }
                let step = int_list(op, "step").unwrap_or_else(|| vec![1; s.len()]);
                let mut out = Vec::with_capacity(s.len());
                for i in 0..s.len() {
                    if step[i] <= 0 {
                        return Inferred::Ill("slice `step` must be positive".into());
                    }
                    match s[i].size() {
                        Some(n) => {
                            if start[i] < 0 || stop[i] < start[i] || stop[i] as u64 > n {
                                return Inferred::Ill(format!(
                                    "slice [{}, {}) is outside axis {i} of length {n}",
                                    start[i], stop[i]
                                ));
                            }
                            let len = (stop[i] - start[i]) as u64;
                            out.push(Dim::N(len.div_ceil(step[i] as u64)));
                        }
                        None => out.push(Dim::Dynamic),
                    }
                }
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            Err(e) => Inferred::Ill(e),
        },
        ("omni.tensor", "concat") => {
            let Some(axis) = int_attr(op, "axis") else {
                return Inferred::Ill("concat has no `axis`".into());
            };
            let mut shape: Option<Vec<Dim>> = None;
            let mut dtype: Option<DType> = None;
            let mut total: Option<u64> = Some(0);
            for (i, t) in ins.iter().enumerate() {
                let Some((s, d)) = t.as_tensor() else {
                    return Inferred::Ill(format!("concat operand {i} is not a tensor"));
                };
                let ax = if axis < 0 {
                    s.len() as i64 + axis
                } else {
                    axis
                };
                if ax < 0 || ax as usize >= s.len() {
                    return Inferred::Ill(format!("concat axis {axis} is out of range"));
                }
                match &mut shape {
                    None => shape = Some(s.to_vec()),
                    Some(first) => {
                        if first.len() != s.len() {
                            return Inferred::Ill("concat operands differ in rank".into());
                        }
                        for (k, (a, b)) in first.iter().zip(s.iter()).enumerate() {
                            if k != ax as usize && !dims_agree(a, b) {
                                return Inferred::Ill(format!(
                                    "concat operands disagree on axis {k}: {} vs {}",
                                    dim_str(a),
                                    dim_str(b)
                                ));
                            }
                        }
                    }
                }
                match &dtype {
                    None => dtype = Some(d.clone()),
                    Some(x) if x == d => {}
                    Some(x) => {
                        return Inferred::Ill(format!(
                            "concat operands differ in dtype: {} and {}",
                            x.label(),
                            d.label()
                        ))
                    }
                }
                total = match (total, s[ax as usize].size()) {
                    (Some(t), Some(n)) => Some(t + n),
                    _ => None,
                };
            }
            match (shape, dtype) {
                (Some(mut s), Some(d)) => {
                    let ax = if axis < 0 {
                        (s.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    s[ax] = match total {
                        Some(n) => Dim::N(n),
                        None => Dim::Dynamic,
                    };
                    Inferred::Types(vec![Type::tensor(s, d)])
                }
                _ => Inferred::Ill("concat has no operands".into()),
            }
        }
        ("omni.tensor", "gather") => match (tensor(0), tensor(1)) {
            (Ok((table, d)), Ok((idx, _))) => {
                let axis = int_attr(op, "axis").unwrap_or(0);
                if axis != 0 {
                    return Inferred::Unchecked(format!(
                        "gather along axis {axis} has no shape function here"
                    ));
                }
                if table.is_empty() {
                    return Inferred::Ill("gather needs a rank ≥ 1 operand".into());
                }
                let mut out = idx;
                out.extend_from_slice(&table[1..]);
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        },
        ("omni.tensor", "where") => match (tensor(1), tensor(2)) {
            (Ok((a, da)), Ok((b, _))) => match broadcast(&a, &b) {
                Ok(s) => Inferred::Types(vec![Type::tensor(s, da)]),
                Err(e) => Inferred::Ill(e),
            },
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        },
        ("omni.tensor", "cumsum") => elementwise_unary(),
        ("omni.nn", "attention") => {
            // (q, k, v) with q: [..., S, Dh] and v: [..., Sk, Dv]; the result
            // takes q's positions and v's channels, which is what makes GQA and
            // MQA expressible without a different op (§07.4.1).
            match (tensor(0), tensor(2)) {
                (Ok((q, d)), Ok((v, _))) => {
                    if q.len() < 2 || v.len() < 2 {
                        return Inferred::Ill("attention needs rank ≥ 2 operands".into());
                    }
                    let mut out = q.clone();
                    let last = out.len() - 1;
                    out[last] = v[v.len() - 1].clone();
                    Inferred::Types(vec![Type::tensor(out, d)])
                }
                (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
            }
        }
        ("omni.nn", "norm" | "rope" | "activation") => elementwise_unary(),
        ("omni.nn", "embedding") => match (tensor(0), tensor(1)) {
            (Ok((ids, _)), Ok((table, d))) => {
                if table.len() != 2 {
                    return Inferred::Ill("an embedding table is rank 2".into());
                }
                let mut out = ids;
                out.push(table[1].clone());
                Inferred::Types(vec![Type::tensor(out, d)])
            }
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        },
        ("omni.nn", "moe_route") => match (tensor(0), tensor(1)) {
            (Ok((x, d)), Ok((w, _))) => {
                let Some(k) = int_attr(op, "top_k") else {
                    return Inferred::Ill("moe_route has no `top_k`".into());
                };
                if k <= 0 {
                    return Inferred::Ill("`top_k` must be positive".into());
                }
                if w.len() != 2 {
                    return Inferred::Ill("a routing matrix is rank 2".into());
                }
                let mut weights = x[..x.len() - 1].to_vec();
                weights.push(Dim::N(k as u64));
                let indices = weights.clone();
                Inferred::Types(vec![
                    Type::tensor(weights, d),
                    Type::tensor(
                        indices,
                        DType::Int {
                            w: 32,
                            signed: true,
                        },
                    ),
                ])
            }
            (Err(e), _) | (_, Err(e)) => Inferred::Ill(e),
        },
        ("omni.quant", "dequantize") => match (tensor(0), op.attr("scheme")) {
            (Ok((s, _)), Some(scheme)) => {
                // The output dtype is the scheme's declared output type.
                match scheme.get("out").and_then(|d| DType::from_value(d).ok()) {
                    Some(d) => Inferred::Types(vec![Type::tensor(s, d)]),
                    None => Inferred::Unchecked(
                        "the dequantization scheme does not declare an output dtype".into(),
                    ),
                }
            }
            (Err(e), _) => Inferred::Ill(e),
            (_, None) => Inferred::Ill("dequantize has no `scheme`".into()),
        },
        ("omni.core", "constant") => match op.outputs.first() {
            // A constant's type is declared, not derived: the value is either
            // inline or a tensor ref, and either way the declaration is what
            // §07.9 says a reader may trust after checking it against the
            // TensorTable (R-I10).
            Some((_, t)) => Inferred::Types(vec![t.clone()]),
            None => Inferred::Ill("a constant has no result".into()),
        },
        ("omni.core", "get") => match (ins.first(), int_attr(op, "index")) {
            (Some(Type::Tuple(elems)), Some(i)) => match elems.get(i as usize) {
                Some(t) => Inferred::Types(vec![t.clone()]),
                None => Inferred::Ill(format!("tuple index {i} is out of range")),
            },
            (Some(other), _) => Inferred::Ill(format!("get needs a tuple, not {}", other.print())),
            _ => Inferred::Ill("get has no `index`".into()),
        },
        ("omni.core", "tuple") => Inferred::Types(vec![Type::Tuple(ins.to_vec())]),
        // Terminators and effect-free annotations produce nothing, which is a
        // fact rather than an absence of information.
        ("omni.core", "return" | "yield" | "assert" | "debug") | ("omni.io", "output") => {
            Inferred::Types(Vec::new())
        }
        _ => Inferred::Unchecked(format!("{} has no shape function here", op.qualified())),
    }
}

// -------------------------------------------------------------- verification --

/// The outcome of verifying a module (§15.1 V5).
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    pub functions: usize,
    pub ops: usize,
    /// Ops whose declared result types were checked against inference.
    pub checked: usize,
    /// Ops this build has no shape function for.
    pub unchecked: usize,
    /// Ops from a dialect or version this build does not know.
    pub unknown: usize,
    /// Unknown ops for which the model ships a lowering, so a runtime can
    /// proceed anyway (§07.2). These are *not* counted as unknown.
    pub recoverable: usize,
}

impl Report {
    pub fn is_invalid(&self) -> bool {
        self.findings.iter().any(Finding::is_invalid)
    }

    pub fn is_indeterminate(&self) -> bool {
        !self.is_invalid() && !self.findings.is_empty()
    }

    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }
}

/// Resolves a tensor name to its declared shape and dtype.
///
/// The `TensorTable` maps names to *refs*, so answering "is this constant the
/// tensor it says it is" needs a store. Verification takes the lookup as a
/// function instead of a store so it stays usable from anywhere — a container,
/// a directory, a test.
pub type TensorLookup<'a> = &'a dyn Fn(&str) -> Option<(Vec<u64>, DType)>;

/// What a verifier may consult beyond the module itself.
#[derive(Default)]
pub struct Context<'a> {
    /// So a `constant` naming a weight can be checked against the weight that
    /// is actually there (R-I10).
    pub tensor: Option<TensorLookup<'a>>,
    /// Rewrites the model ships, which decide whether an unknown op is
    /// recoverable (§07.2) rather than merely unknown.
    pub rewrites: &'a [Rewrite],
}

/// Verifies a module.
///
/// Rules, in the numbering this implementation introduces for §07 (the
/// specification's §15.2 checklist stops at adapters):
///
/// * **R-I01** every value is defined exactly once in its function
/// * **R-I02** every use refers to a value already in scope
/// * **R-I03** `entry` names a function the module defines
/// * **R-I04** every op's dialect is declared by the module
/// * **R-I05** the op exists in that dialect at that version
/// * **R-I06** declared result types equal inferred result types
/// * **R-I07** region and terminator structure matches the op's contract
/// * **R-I08** `token` values are used exactly once, so effect order is total
/// * **R-I09** symbolic-dimension constraints are satisfiable
/// * **R-I10** a `constant` naming a tensor agrees with that tensor
/// * **R-I11** a derived (lowered) module sits below the module it came from
pub fn verify(m: &Module, cx: &Context<'_>) -> Report {
    let mut r = Report {
        functions: m.functions.len(),
        ..Default::default()
    };

    if m.function(&m.entry).is_none() {
        r.findings.push(invalid(
            "R-I03",
            format!(
                "`entry` names `{}`, which the module does not define",
                m.entry
            ),
        ));
    }

    for (name, f) in &m.functions {
        // R-I09: a constraint set that cannot be satisfied makes every shape
        // claim in the function meaningless.
        let mut lower: Vec<(&str, u64)> = Vec::new();
        let mut upper: Vec<(&str, u64)> = Vec::new();
        for c in &f.constraints {
            match c.rel {
                Rel::Ge => lower.push((&c.dim, c.bound)),
                Rel::Le => upper.push((&c.dim, c.bound)),
                Rel::Eq => {
                    lower.push((&c.dim, c.bound));
                    upper.push((&c.dim, c.bound));
                }
                Rel::Multiple => {
                    if c.bound == 0 {
                        r.findings.push(invalid(
                            "R-I09",
                            format!(
                                "@{name}: dimension {} is declared a multiple of zero",
                                c.dim
                            ),
                        ));
                    }
                }
            }
        }
        for (d, lo) in &lower {
            for (e, hi) in &upper {
                if d == e && lo > hi {
                    r.findings.push(invalid(
                        "R-I09",
                        format!("@{name}: dimension {d} must be ≥ {lo} and ≤ {hi}"),
                    ));
                }
            }
        }

        // Scope: parameters and the entry block's arguments are in scope for the
        // whole body. The IR is structured, so "defined earlier in this block or
        // an enclosing one" is exactly dominance.
        let mut scope: Vec<(u32, Type)> = Vec::new();
        let mut defined: Vec<u32> = Vec::new();
        for (i, (_, t)) in f.params.iter().enumerate() {
            // Parameters occupy the first value ids, by construction of §07.3's
            // dense numbering.
            scope.push((i as u32, t.clone()));
            defined.push(i as u32);
        }
        let mut tokens: Vec<(u32, usize)> = Vec::new();
        verify_region(
            &f.body,
            name,
            &mut scope,
            &mut defined,
            &mut tokens,
            m,
            cx,
            &mut r,
        );

        // R-I08: an effect token used twice is two orderings, which is none.
        for (id, uses) in tokens {
            if uses != 1 {
                r.findings.push(invalid(
                    "R-I08",
                    format!(
                        "@{name}: effect token %{id} is used {uses} times; \
                         a token orders exactly one successor"
                    ),
                ));
            }
        }
    }

    if let Some(_from) = &m.lowered_from {
        // R-I11: `lowered_from` means *this* module is the derived one, so it
        // must sit at a lower level than a semantic one. We cannot see the
        // parent from here, but a semantic module claiming to be lowered from
        // something is contradictory on its face.
        if m.level == Level::Semantic {
            r.findings.push(invalid(
                "R-I11",
                "a semantic module cannot be the result of lowering".to_string(),
            ));
        }
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn verify_region(
    region: &Region,
    fname: &str,
    scope: &mut Vec<(u32, Type)>,
    defined: &mut Vec<u32>,
    tokens: &mut Vec<(u32, usize)>,
    m: &Module,
    cx: &Context<'_>,
    r: &mut Report,
) {
    for block in &region.blocks {
        let outer = scope.len();
        for (id, t) in &block.args {
            define(*id, t.clone(), fname, scope, defined, tokens, r);
        }
        for (i, op) in block.ops.iter().enumerate() {
            r.ops += 1;

            // R-I04 / R-I05: is this an op we are allowed to reason about?
            let known = match dialect(&op.dialect) {
                None => {
                    if m.declares(&op.dialect).is_none() {
                        r.findings.push(invalid(
                            "R-I04",
                            format!(
                                "@{fname}: {} is used but its dialect is not declared",
                                op.qualified()
                            ),
                        ));
                    }
                    if cx.rewrites.iter().any(|w| w.matches_op(op)) {
                        r.recoverable += 1;
                    } else {
                        r.unknown += 1;
                        r.findings.push(unknown(
                            "R-I05",
                            format!(
                                "@{fname}: dialect `{}` is not known to this build and no \
                                 lowering for {} is shipped",
                                op.dialect,
                                op.qualified()
                            ),
                        ));
                    }
                    None
                }
                Some(d) => {
                    if m.declares(&op.dialect).is_none() {
                        r.findings.push(invalid(
                            "R-I04",
                            format!(
                                "@{fname}: {} is used but its dialect is not declared",
                                op.qualified()
                            ),
                        ));
                    }
                    match d.op(&op.name) {
                        None => {
                            r.unknown += 1;
                            r.findings.push(unknown(
                                "R-I05",
                                format!("@{fname}: `{}` defines no op `{}`", op.dialect, op.name),
                            ));
                            None
                        }
                        Some(spec) => {
                            if !spec.versions.contains(&op.version) {
                                if cx.rewrites.iter().any(|w| w.matches_op(op)) {
                                    r.recoverable += 1;
                                } else {
                                    r.unknown += 1;
                                    r.findings.push(unknown(
                                        "R-I05",
                                        format!(
                                            "@{fname}: {} — this build knows versions {:?}, and \
                                             no migration rewrite is shipped",
                                            op.qualified(),
                                            spec.versions
                                        ),
                                    ));
                                }
                                None
                            } else {
                                Some(spec)
                            }
                        }
                    }
                }
            };

            if let Some(spec) = known {
                check_arity(op, spec, fname, r);
                check_attrs(op, spec, fname, r);
                check_regions(op, spec, block, i, fname, r);
            }

            // R-I02: uses must be in scope.
            let mut input_types = Vec::with_capacity(op.inputs.len());
            for id in &op.inputs {
                match scope.iter().rev().find(|(v, _)| v == id) {
                    Some((_, t)) => {
                        if matches!(t, Type::Token) {
                            if let Some(e) = tokens.iter_mut().find(|(v, _)| v == id) {
                                e.1 += 1;
                            }
                        }
                        input_types.push(t.clone());
                    }
                    None => {
                        r.findings.push(invalid(
                            "R-I02",
                            format!(
                                "@{fname}: {} uses %{id}, which is not defined in scope",
                                op.qualified()
                            ),
                        ));
                        input_types.push(Type::Opaque("<undefined>".into()));
                    }
                }
            }

            // R-I06: declared types must equal inferred ones.
            if known.is_some() {
                match infer(op, &input_types) {
                    Inferred::Types(ts) => {
                        r.checked += 1;
                        if ts.len() != op.outputs.len() {
                            r.findings.push(invalid(
                                "R-I06",
                                format!(
                                    "@{fname}: {} declares {} result(s) but produces {}",
                                    op.qualified(),
                                    op.outputs.len(),
                                    ts.len()
                                ),
                            ));
                        } else {
                            for (k, (declared, inferred)) in
                                op.outputs.iter().map(|(_, t)| t).zip(ts.iter()).enumerate()
                            {
                                if !types_agree(declared, inferred) {
                                    r.findings.push(invalid(
                                        "R-I06",
                                        format!(
                                            "@{fname}: {} result {k} is declared {} but infers {}",
                                            op.qualified(),
                                            declared.print(),
                                            inferred.print()
                                        ),
                                    ));
                                }
                            }
                        }
                    }
                    Inferred::Unchecked(_) => r.unchecked += 1,
                    Inferred::Ill(msg) => r.findings.push(invalid(
                        "R-I06",
                        format!("@{fname}: {} — {msg}", op.qualified()),
                    )),
                }
            }

            // R-I10: a constant that names a tensor is a claim about that
            // tensor, and a claim that can be checked.
            if op.dialect == "omni.core" && op.name == "constant" {
                if let (Some(Value::Text(name)), Some(lookup), Some((_, declared))) =
                    (op.attr("tensor"), cx.tensor, op.outputs.first())
                {
                    check_constant(name, declared, lookup, fname, r);
                }
            }

            // Nested regions see the enclosing scope.
            for sub in &op.regions {
                verify_region(sub, fname, scope, defined, tokens, m, cx, r);
            }

            for (id, t) in &op.outputs {
                define(*id, t.clone(), fname, scope, defined, tokens, r);
            }
        }
        scope.truncate(outer);
    }
}

fn define(
    id: u32,
    t: Type,
    fname: &str,
    scope: &mut Vec<(u32, Type)>,
    defined: &mut Vec<u32>,
    tokens: &mut Vec<(u32, usize)>,
    r: &mut Report,
) {
    if defined.contains(&id) {
        r.findings.push(invalid(
            "R-I01",
            format!("@{fname}: %{id} is defined more than once; the IR is SSA"),
        ));
    } else {
        defined.push(id);
    }
    if matches!(t, Type::Token) {
        tokens.push((id, 0));
    }
    scope.push((id, t));
}

fn check_arity(op: &Op, spec: &OpSpec, fname: &str, r: &mut Report) {
    let n = op.inputs.len();
    let ok = n >= spec.inputs.0 && spec.inputs.1.is_none_or(|max| n <= max);
    if !ok {
        r.findings.push(invalid(
            "R-I07",
            format!(
                "@{fname}: {} takes {}{} inputs but was given {n}",
                op.qualified(),
                spec.inputs.0,
                match spec.inputs.1 {
                    Some(max) if max == spec.inputs.0 => String::new(),
                    Some(max) => format!("–{max}"),
                    None => "+".into(),
                }
            ),
        ));
    }
    if op.outputs.len() != spec.results {
        r.findings.push(invalid(
            "R-I07",
            format!(
                "@{fname}: {} produces {} result(s), not {}",
                op.qualified(),
                spec.results,
                op.outputs.len()
            ),
        ));
    }
}

fn check_attrs(op: &Op, spec: &OpSpec, fname: &str, r: &mut Report) {
    for (name, kind, required) in spec.attrs {
        match op.attr(name) {
            Some(v) => {
                if !kind.accepts(v) {
                    r.findings.push(invalid(
                        "R-I07",
                        format!(
                            "@{fname}: {}'s `{name}` is {}, which is not the declared kind",
                            op.qualified(),
                            short_value(v)
                        ),
                    ));
                }
            }
            None if *required => r.findings.push(invalid(
                "R-I07",
                format!("@{fname}: {} is missing required `{name}`", op.qualified()),
            )),
            None => {}
        }
    }
    // An attribute the op does not define is not an error — §07.4.1 allows
    // adding optional attributes without a version bump, so a graph written
    // against a newer minor dialect stays readable. It is reported, though,
    // because silently ignoring an attribute is how semantics get lost.
    for (k, _) in &op.attrs {
        if !spec.attrs.iter().any(|(n, _, _)| n == k) && k != "tensor" {
            r.findings.push(unknown(
                "R-I07",
                format!(
                    "@{fname}: {} carries `{k}`, which this build does not interpret",
                    op.qualified()
                ),
            ));
        }
    }
}

fn check_regions(op: &Op, spec: &OpSpec, block: &Block, at: usize, fname: &str, r: &mut Report) {
    if op.regions.len() != spec.regions {
        r.findings.push(invalid(
            "R-I07",
            format!(
                "@{fname}: {} takes {} region(s) but carries {}",
                op.qualified(),
                spec.regions,
                op.regions.len()
            ),
        ));
    }
    for region in &op.regions {
        if region.blocks.is_empty() {
            r.findings.push(invalid(
                "R-I07",
                format!("@{fname}: {} has an empty region", op.qualified()),
            ));
            continue;
        }
        for b in &region.blocks {
            match b.ops.last() {
                Some(last)
                    if last.dialect == "omni.core"
                        && (last.name == "yield" || last.name == "return") => {}
                _ => r.findings.push(invalid(
                    "R-I07",
                    format!(
                        "@{fname}: a region of {} does not end in core/yield or core/return",
                        op.qualified()
                    ),
                )),
            }
        }
    }
    // A terminator in the middle of a block is dead code at best.
    let terminates = op.dialect == "omni.core" && (op.name == "return" || op.name == "yield");
    if terminates && at + 1 != block.ops.len() {
        r.findings.push(invalid(
            "R-I07",
            format!(
                "@{fname}: {} is not the last op in its block",
                op.qualified()
            ),
        ));
    }
}

fn check_constant(
    name: &str,
    declared: &Type,
    lookup: TensorLookup<'_>,
    fname: &str,
    r: &mut Report,
) {
    let Some((tensor_shape, tensor_dtype)) = lookup(name) else {
        r.findings.push(invalid(
            "R-I10",
            format!("@{fname}: constant names tensor `{name}`, which the model does not define"),
        ));
        return;
    };
    let Some((shape, dtype)) = declared.as_tensor() else {
        r.findings.push(invalid(
            "R-I10",
            format!(
                "@{fname}: constant `{name}` is declared {}, not a tensor",
                declared.print()
            ),
        ));
        return;
    };
    if &tensor_dtype != dtype {
        r.findings.push(invalid(
            "R-I10",
            format!(
                "@{fname}: constant `{name}` is declared {} but the tensor is {}",
                dtype.label(),
                tensor_dtype.label()
            ),
        ));
    }
    if tensor_shape.len() != shape.len()
        || !tensor_shape
            .iter()
            .zip(shape)
            .all(|(a, b)| dims_agree(&Dim::N(*a), b))
    {
        let want: Vec<String> = tensor_shape.iter().map(|d| d.to_string()).collect();
        r.findings.push(invalid(
            "R-I10",
            format!(
                "@{fname}: constant `{name}` is declared {} but the tensor is shaped {}",
                declared.print(),
                want.join("×")
            ),
        ));
    }
}

// ------------------------------------------------------------------ rewrites --

/// How a rewrite affects results (§07.7). A deployment with reproducibility
/// requirements may refuse the approximate ones, which is only possible because
/// they have to say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Soundness {
    SemanticsPreserving,
    NumericApproximate,
}

impl Soundness {
    fn name(&self) -> &'static str {
        match self {
            Soundness::SemanticsPreserving => "semantics-preserving",
            Soundness::NumericApproximate => "numeric-approximate",
        }
    }

    fn parse(s: &str) -> Option<Soundness> {
        match s {
            "semantics-preserving" => Some(Soundness::SemanticsPreserving),
            "numeric-approximate" => Some(Soundness::NumericApproximate),
            _ => None,
        }
    }
}

/// A side condition on the matched op.
#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    AttrAbsent(String),
    AttrPresent(String),
    AttrEq(String, Value),
    AttrNe(String, Value),
    /// The attribute is absent, or equal to this value — which is how a
    /// condition talks about an attribute that has a default (§07.4.1).
    AttrDefaults(String, Value),
}

/// How an emitted op's result type is determined when inference cannot do it.
///
/// A rewrite is static data, so it cannot know the dtype of the tensor it will
/// be applied to. `DtypeLike` says "the dtype of the value bound to this name,
/// with this shape", which is exactly what a lowering needs to emit a scalar
/// constant into a graph whose element type it will only learn at match time.
#[derive(Clone, Debug, PartialEq)]
pub enum TypeSpec {
    Exact(Type),
    DtypeLike { name: String, shape: Vec<Dim> },
}

impl TypeSpec {
    fn to_value(&self) -> Value {
        match self {
            TypeSpec::Exact(t) => t.to_value(),
            TypeSpec::DtypeLike { name, shape } => Value::map(vec![
                ("dtype_like", Value::text(name.clone())),
                ("shape", crate::expr::shape_to_value(shape)),
            ]),
        }
    }

    fn from_value(v: &Value) -> Res<TypeSpec> {
        match v.get("dtype_like").and_then(|x| x.as_str()) {
            Some(name) => Ok(TypeSpec::DtypeLike {
                name: name.to_string(),
                shape: match v.get("shape") {
                    Some(s) => crate::expr::parse_shape_value(s)
                        .map_err(|e: crate::expr::Error| Error(e.to_string()))?,
                    None => Vec::new(),
                },
            }),
            None => Ok(TypeSpec::Exact(Type::from_value(v)?)),
        }
    }

    fn resolve(&self, types: &[(u32, Type)], env: &[(String, u32)]) -> Result<Type, String> {
        match self {
            TypeSpec::Exact(t) => Ok(t.clone()),
            TypeSpec::DtypeLike { name, shape } => {
                let id = env
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, id)| *id)
                    .ok_or_else(|| format!("`{name}` is not bound"))?;
                let t = types
                    .iter()
                    .find(|(v, _)| *v == id)
                    .map(|(_, t)| t)
                    .ok_or_else(|| format!("`{name}` has no known type"))?;
                let (_, dtype) = t
                    .as_tensor()
                    .ok_or_else(|| format!("`{name}` is not a tensor"))?;
                Ok(Type::tensor(shape.clone(), dtype.clone()))
            }
        }
    }
}

/// One op in a rewrite's replacement. Inputs and outputs are *names*: either a
/// name bound by the pattern or a fresh local one.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitOp {
    pub dialect: String,
    pub name: String,
    pub version: u32,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, Value)>,
    /// Attributes taken from the matched op, as `(source, destination)` — a
    /// lowering that turns `attention`'s `scale` into a `constant`'s `value`
    /// needs the rename.
    pub copy_attrs: Vec<(String, String)>,
    /// Declared result types, when the op has no inputs to infer from.
    pub out_types: Vec<TypeSpec>,
}

/// A declarative rewrite (§07.7): pattern → replacement, with side conditions.
///
/// The specification's example replaces one op with one op (a version
/// migration). Lowering needs a *sequence*, so `emit` is a list; a single-op
/// emit is the degenerate case and encodes identically.
#[derive(Clone, Debug, PartialEq)]
pub struct Rewrite {
    pub name: String,
    pub dialect: String,
    pub op: String,
    pub version: u32,
    /// Pattern operand names, in order: `["q", "k", "v"]` binds input 0, 1, 2.
    pub binds: Vec<String>,
    pub conditions: Vec<Cond>,
    pub emit: Vec<EmitOp>,
    /// Which emitted names take over the matched op's results, in order.
    pub results: Vec<String>,
    pub soundness: Soundness,
    /// The level the replacement is expressed at, when the rewrite is a
    /// lowering rather than a migration.
    pub to_level: Option<Level>,
}

pub const REWRITE_SCHEMA: &str = "omni.ir/rewrite";

impl Rewrite {
    pub fn matches_op(&self, op: &Op) -> bool {
        op.dialect == self.dialect && op.name == self.op && op.version == self.version
    }

    fn conditions_hold(&self, op: &Op) -> bool {
        self.conditions.iter().all(|c| match c {
            Cond::AttrAbsent(k) => op.attr(k).is_none(),
            Cond::AttrPresent(k) => op.attr(k).is_some(),
            Cond::AttrEq(k, v) => op.attr(k) == Some(v),
            Cond::AttrNe(k, v) => op.attr(k) != Some(v),
            Cond::AttrDefaults(k, v) => op.attr(k).is_none_or(|x| x == v),
        })
    }

    pub fn to_value(&self) -> Value {
        let mut p = vec![
            ("t", Value::text(REWRITE_SCHEMA)),
            ("v", Value::U(1)),
            ("name", Value::text(self.name.clone())),
            (
                "match",
                Value::map(vec![
                    (
                        "op",
                        Value::Array(vec![
                            Value::text(self.dialect.clone()),
                            Value::text(self.op.clone()),
                            Value::U(self.version as u64),
                        ]),
                    ),
                    (
                        "binds",
                        Value::Map(
                            self.binds
                                .iter()
                                .enumerate()
                                .map(|(i, n)| (Value::text(n.clone()), Value::U(i as u64)))
                                .collect(),
                        ),
                    ),
                ]),
            ),
            (
                "emit",
                Value::Array(
                    self.emit
                        .iter()
                        .map(|e| {
                            let mut q = vec![
                                (
                                    "op",
                                    Value::Array(vec![
                                        Value::text(e.dialect.clone()),
                                        Value::text(e.name.clone()),
                                        Value::U(e.version as u64),
                                    ]),
                                ),
                                (
                                    "in",
                                    Value::Array(
                                        e.inputs.iter().map(|n| Value::text(n.clone())).collect(),
                                    ),
                                ),
                                (
                                    "out",
                                    Value::Array(
                                        e.outputs.iter().map(|n| Value::text(n.clone())).collect(),
                                    ),
                                ),
                            ];
                            if !e.attrs.is_empty() {
                                q.push((
                                    "attrs",
                                    Value::Map(
                                        e.attrs
                                            .iter()
                                            .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                                            .collect(),
                                    ),
                                ));
                            }
                            if !e.copy_attrs.is_empty() {
                                q.push((
                                    "copy_attrs",
                                    Value::Map(
                                        e.copy_attrs
                                            .iter()
                                            .map(|(from, to)| {
                                                (Value::text(from.clone()), Value::text(to.clone()))
                                            })
                                            .collect(),
                                    ),
                                ));
                            }
                            if !e.out_types.is_empty() {
                                q.push((
                                    "out_types",
                                    Value::Array(
                                        e.out_types.iter().map(TypeSpec::to_value).collect(),
                                    ),
                                ));
                            }
                            Value::map(q)
                        })
                        .collect(),
                ),
            ),
            (
                "results",
                Value::Array(
                    self.results
                        .iter()
                        .map(|n| Value::text(n.clone()))
                        .collect(),
                ),
            ),
            ("soundness", Value::text(self.soundness.name())),
        ];
        if !self.conditions.is_empty() {
            p.push((
                "where",
                Value::Array(
                    self.conditions
                        .iter()
                        .map(|c| match c {
                            Cond::AttrAbsent(k) => Value::map(vec![
                                ("attr", Value::text(k.clone())),
                                ("absent", Value::Bool(true)),
                            ]),
                            Cond::AttrPresent(k) => Value::map(vec![
                                ("attr", Value::text(k.clone())),
                                ("absent", Value::Bool(false)),
                            ]),
                            Cond::AttrEq(k, v) => Value::map(vec![
                                ("attr", Value::text(k.clone())),
                                ("eq", v.clone()),
                            ]),
                            Cond::AttrNe(k, v) => Value::map(vec![
                                ("attr", Value::text(k.clone())),
                                ("ne", v.clone()),
                            ]),
                            Cond::AttrDefaults(k, v) => Value::map(vec![
                                ("attr", Value::text(k.clone())),
                                ("eq", v.clone()),
                                ("or_absent", Value::Bool(true)),
                            ]),
                        })
                        .collect(),
                ),
            ));
        }
        if let Some(l) = self.to_level {
            p.push((
                "lower_to",
                Value::map(vec![("level", Value::text(l.name()))]),
            ));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Rewrite> {
        if v.get("t").and_then(|x| x.as_str()) != Some(REWRITE_SCHEMA) {
            return err("not an omni.ir/rewrite object");
        }
        let name = v
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let m = v
            .get("match")
            .ok_or_else(|| Error("a rewrite has no `match`".into()))?;
        let triple = match m.get("op") {
            Some(Value::Array(a)) if a.len() == 3 => a,
            _ => return err("a rewrite's `match.op` is not a [dialect, name, version] triple"),
        };
        let dialect = triple[0]
            .as_str()
            .ok_or_else(|| Error("a rewrite's dialect is not text".into()))?
            .to_string();
        let op = triple[1]
            .as_str()
            .ok_or_else(|| Error("a rewrite's op name is not text".into()))?
            .to_string();
        let version = triple[2].as_u64().unwrap_or(1) as u32;
        // `binds` maps name -> operand index; the order that matters is the
        // index, so sort by it.
        let mut binds_idx: Vec<(usize, String)> = Vec::new();
        if let Some(Value::Map(b)) = m.get("binds") {
            for (k, idx) in b {
                if let (Some(n), Some(i)) = (k.as_str(), idx.as_u64()) {
                    binds_idx.push((i as usize, n.to_string()));
                }
            }
        }
        binds_idx.sort();
        let binds = binds_idx.into_iter().map(|(_, n)| n).collect();

        let mut emit = Vec::new();
        if let Some(Value::Array(a)) = v.get("emit") {
            for e in a {
                let triple = match e.get("op") {
                    Some(Value::Array(t)) if t.len() == 3 => t,
                    _ => return err("an emitted op is not a [dialect, name, version] triple"),
                };
                emit.push(EmitOp {
                    dialect: triple[0].as_str().unwrap_or_default().to_string(),
                    name: triple[1].as_str().unwrap_or_default().to_string(),
                    version: triple[2].as_u64().unwrap_or(1) as u32,
                    inputs: text_list(e.get("in")),
                    outputs: text_list(e.get("out")),
                    attrs: match e.get("attrs") {
                        Some(Value::Map(m)) => m
                            .iter()
                            .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                            .collect(),
                        _ => Vec::new(),
                    },
                    copy_attrs: match e.get("copy_attrs") {
                        Some(Value::Map(m)) => m
                            .iter()
                            .filter_map(|(k, v)| match (k.as_str(), v.as_str()) {
                                (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                                _ => None,
                            })
                            .collect(),
                        _ => Vec::new(),
                    },
                    out_types: match e.get("out_types") {
                        Some(Value::Array(a)) => a
                            .iter()
                            .map(TypeSpec::from_value)
                            .collect::<Res<Vec<_>>>()?,
                        _ => Vec::new(),
                    },
                });
            }
        }
        let mut conditions = Vec::new();
        if let Some(Value::Array(a)) = v.get("where") {
            for c in a {
                let Some(attr) = c.get("attr").and_then(|x| x.as_str()) else {
                    return err("a `where` clause has no `attr`");
                };
                if let Some(Value::Bool(absent)) = c.get("absent") {
                    conditions.push(if *absent {
                        Cond::AttrAbsent(attr.to_string())
                    } else {
                        Cond::AttrPresent(attr.to_string())
                    });
                } else if let Some(v) = c.get("eq") {
                    conditions.push(if matches!(c.get("or_absent"), Some(Value::Bool(true))) {
                        Cond::AttrDefaults(attr.to_string(), v.clone())
                    } else {
                        Cond::AttrEq(attr.to_string(), v.clone())
                    });
                } else if let Some(v) = c.get("ne") {
                    conditions.push(Cond::AttrNe(attr.to_string(), v.clone()));
                } else {
                    return err("a `where` clause states no condition");
                }
            }
        }
        Ok(Rewrite {
            name,
            dialect,
            op,
            version,
            binds,
            conditions,
            emit,
            results: text_list(v.get("results")),
            soundness: v
                .get("soundness")
                .and_then(|x| x.as_str())
                .and_then(Soundness::parse)
                .unwrap_or(Soundness::SemanticsPreserving),
            to_level: v
                .get("lower_to")
                .and_then(|l| l.get("level"))
                .and_then(|x| x.as_str())
                .and_then(Level::parse),
        })
    }
}

fn text_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// What applying rewrites produced.
#[derive(Clone, Debug, Default)]
pub struct Applied {
    /// `(rewrite name, times applied)`, in application order.
    pub applied: Vec<(String, usize)>,
    /// Rewrites that matched but could not be applied, with the reason.
    pub refused: Vec<(String, String)>,
    pub approximate: bool,
}

/// Applies rewrites to a module until no rule matches (§07.7).
///
/// This is the mechanism behind §07.2's key claim: a runtime that understands
/// `omni.tensor` but not `omni.nn/attention` applies the lowering the *model*
/// shipped and proceeds. So this function is what turns "unknown op" from fatal
/// into slow.
pub fn apply_rewrites(m: &Module, rules: &[Rewrite], allow_approximate: bool) -> (Module, Applied) {
    let mut out = m.clone();
    let mut info = Applied::default();
    let mut next_id = m.max_value_id() + 1;
    // Bounded: each pass must apply at least one rule, and a rule that produces
    // ops it also matches would otherwise never terminate.
    for _ in 0..64 {
        let mut changed = false;
        for (_, f) in out.functions.iter_mut() {
            let mut types = value_types(f);
            if rewrite_region(
                &mut f.body,
                rules,
                allow_approximate,
                &mut next_id,
                &mut types,
                &mut info,
            ) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The lowest level any applied rule targets becomes the module's level, and
    // the module records what it was derived from (§07.2).
    if !info.applied.is_empty() {
        let target = rules
            .iter()
            .filter(|r| info.applied.iter().any(|(n, _)| n == &r.name))
            .filter_map(|r| r.to_level)
            .min();
        if let Some(l) = target {
            out.level = l.min(out.level);
        }
        // A lowered module uses whatever dialects its replacement ops need.
        let mut used: Vec<String> = out
            .ops()
            .iter()
            .map(|(_, o)| o.dialect.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        used.retain(|ns| out.declares(ns).is_none());
        for ns in used {
            let version = dialect(&ns).map(|d| d.version).unwrap_or(1);
            out.dialects.push(DialectUse {
                ns,
                version,
                reference: None,
            });
        }
        // Dialects nothing uses any more are dropped: a lowered graph that still
        // claimed omni.nn would be asking a runtime for a capability it does not
        // need.
        let live: std::collections::BTreeSet<String> =
            out.ops().iter().map(|(_, o)| o.dialect.clone()).collect();
        out.dialects
            .retain(|d| live.contains(&d.ns) || d.ns == "omni.core");
    }
    (out, info)
}

/// The declared type of every value in a function, for inferring replacement
/// types as they are emitted.
fn value_types(f: &Function) -> Vec<(u32, Type)> {
    let mut out: Vec<(u32, Type)> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, (_, t))| (i as u32, t.clone()))
        .collect();
    fn walk(r: &Region, out: &mut Vec<(u32, Type)>) {
        for b in &r.blocks {
            for (id, t) in &b.args {
                out.push((*id, t.clone()));
            }
            for op in &b.ops {
                for (id, t) in &op.outputs {
                    out.push((*id, t.clone()));
                }
                for sub in &op.regions {
                    walk(sub, out);
                }
            }
        }
    }
    walk(&f.body, &mut out);
    out
}

fn rewrite_region(
    region: &mut Region,
    rules: &[Rewrite],
    allow_approximate: bool,
    next_id: &mut u32,
    types: &mut Vec<(u32, Type)>,
    info: &mut Applied,
) -> bool {
    let mut changed = false;
    for block in &mut region.blocks {
        let mut i = 0;
        while i < block.ops.len() {
            // Nested regions first, so a rewrite of an outer op sees a settled
            // interior.
            let mut subs = std::mem::take(&mut block.ops[i].regions);
            for sub in &mut subs {
                if rewrite_region(sub, rules, allow_approximate, next_id, types, info) {
                    changed = true;
                }
            }
            block.ops[i].regions = subs;

            let Some(rule) = rules
                .iter()
                .find(|r| r.matches_op(&block.ops[i]) && r.conditions_hold(&block.ops[i]))
            else {
                i += 1;
                continue;
            };
            if rule.soundness == Soundness::NumericApproximate && !allow_approximate {
                if !info.refused.iter().any(|(n, _)| n == &rule.name) {
                    info.refused.push((
                        rule.name.clone(),
                        "changes results and was not allowed to".into(),
                    ));
                }
                i += 1;
                continue;
            }
            match expand(&block.ops[i], rule, next_id, types) {
                Ok(new_ops) => {
                    let n = new_ops.len();
                    block.ops.splice(i..i + 1, new_ops);
                    match info.applied.iter_mut().find(|(n, _)| n == &rule.name) {
                        Some(e) => e.1 += 1,
                        None => info.applied.push((rule.name.clone(), 1)),
                    }
                    if rule.soundness == Soundness::NumericApproximate {
                        info.approximate = true;
                    }
                    changed = true;
                    i += n;
                }
                Err(why) => {
                    if !info.refused.iter().any(|(n, _)| n == &rule.name) {
                        info.refused.push((rule.name.clone(), why));
                    }
                    i += 1;
                }
            }
        }
    }
    changed
}

/// Turns one matched op into the rule's replacement ops.
fn expand(
    op: &Op,
    rule: &Rewrite,
    next_id: &mut u32,
    types: &mut Vec<(u32, Type)>,
) -> Result<Vec<Op>, String> {
    // Bind pattern operand names to the matched op's inputs.
    let mut env: Vec<(String, u32)> = Vec::new();
    for (i, name) in rule.binds.iter().enumerate() {
        match op.inputs.get(i) {
            Some(id) => env.push((name.clone(), *id)),
            None => {
                return Err(format!(
                    "the pattern binds `{name}` to a missing operand {i}"
                ))
            }
        }
    }
    // Names the rule declares as results take over the matched op's result ids,
    // so nothing downstream has to be rewritten.
    let mut reserved: Vec<(String, u32, Type)> = Vec::new();
    for (k, name) in rule.results.iter().enumerate() {
        match op.outputs.get(k) {
            Some((id, t)) => reserved.push((name.clone(), *id, t.clone())),
            None => {
                return Err(format!(
                    "the rule names result `{name}` but the op has none"
                ))
            }
        }
    }

    let mut out = Vec::with_capacity(rule.emit.len());
    for e in &rule.emit {
        let mut inputs = Vec::with_capacity(e.inputs.len());
        for n in &e.inputs {
            match env.iter().find(|(name, _)| name == n) {
                Some((_, id)) => inputs.push(*id),
                None => return Err(format!("`{n}` is used before it is produced")),
            }
        }
        let mut new_op = Op {
            dialect: e.dialect.clone(),
            name: e.name.clone(),
            version: e.version,
            inputs,
            outputs: Vec::new(),
            attrs: e.attrs.clone(),
            regions: Vec::new(),
            loc: op.loc.clone(),
        };
        for (from, to) in &e.copy_attrs {
            match op.attr(from) {
                Some(v) => new_op.attrs.push((to.clone(), v.clone())),
                None => {
                    return Err(format!(
                        "the rule copies `{from}`, which the matched op does not carry"
                    ))
                }
            }
        }
        // Result types come from inference, which is the only way a rewrite can
        // be *checked* rather than trusted.
        let in_types: Vec<Type> = new_op
            .inputs
            .iter()
            .map(|id| {
                types
                    .iter()
                    .find(|(v, _)| v == id)
                    .map(|(_, t)| t.clone())
                    .unwrap_or(Type::Opaque("<unknown>".into()))
            })
            .collect();
        let inferred = if e.out_types.is_empty() {
            match infer(&new_op, &in_types) {
                Inferred::Types(ts) => ts,
                Inferred::Unchecked(why) => {
                    return Err(format!(
                        "the replacement op {} cannot be typed here: {why}",
                        new_op.qualified()
                    ))
                }
                Inferred::Ill(why) => {
                    return Err(format!(
                        "the replacement op {} would be ill-typed: {why}",
                        new_op.qualified()
                    ))
                }
            }
        } else {
            e.out_types
                .iter()
                .map(|spec| spec.resolve(types, &env))
                .collect::<Result<Vec<_>, String>>()?
        };
        if inferred.len() != e.outputs.len() {
            return Err(format!(
                "{} produces {} result(s) but the rule names {}",
                new_op.qualified(),
                inferred.len(),
                e.outputs.len()
            ));
        }
        for (name, t) in e.outputs.iter().zip(inferred) {
            let id = match reserved.iter().find(|(n, _, _)| n == name) {
                Some((_, id, declared)) => {
                    if !types_agree(declared, &t) {
                        return Err(format!(
                            "the rule's result `{name}` would be {} where the original op \
                             declared {}",
                            t.print(),
                            declared.print()
                        ));
                    }
                    *id
                }
                None => {
                    let id = *next_id;
                    *next_id += 1;
                    id
                }
            };
            env.push((name.clone(), id));
            types.push((id, t.clone()));
            new_op.outputs.push((id, t));
        }
        out.push(new_op);
    }
    // Every result the original op produced must be produced by the expansion,
    // or the graph loses a value.
    for (name, id, _) in &reserved {
        if !out.iter().any(|o| o.outputs.iter().any(|(v, _)| v == id)) {
            return Err(format!("the expansion never produces result `{name}`"));
        }
    }
    Ok(out)
}

// -------------------------------------------------------- shipped lowerings --

fn emit(dialect: &str, name: &str, ins: &[&str], outs: &[&str]) -> EmitOp {
    EmitOp {
        dialect: dialect.into(),
        name: name.into(),
        version: 1,
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: Vec::new(),
        copy_attrs: Vec::new(),
        out_types: Vec::new(),
    }
}

fn with_attrs(mut e: EmitOp, attrs: Vec<(&str, Value)>) -> EmitOp {
    e.attrs = attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    e
}

/// The lowerings `omni.nn` ships (§07.4.2's `lower_to`, §07.6 tier 1).
///
/// These are what make an unknown high-level op recoverable rather than fatal:
/// a runtime that implements `omni.tensor` and nothing else can run a model
/// written in `omni.nn` by applying them. Each is expressed entirely in
/// `omni.core` + `omni.tensor`, so there is no third thing to understand.
///
/// The coverage is deliberately narrow and states its own edges: attention
/// lowers when it is not causal, not windowed, not softcapped and not grouped,
/// because a causal mask cannot be built without ops for constructing index
/// tensors, and pretending otherwise would produce a graph that computes the
/// wrong thing. An op that does not match keeps its high-level form and is
/// reported as needing a runtime that understands it.
pub fn shipped_lowerings() -> Vec<Rewrite> {
    vec![
        // attention(q,k,v) → matmul(softmax(matmul(q, kᵀ) · scale), v)
        Rewrite {
            name: "omni.nn/attention@2→primitive".into(),
            dialect: "omni.nn".into(),
            op: "attention".into(),
            version: 2,
            binds: vec!["q".into(), "k".into(), "v".into()],
            conditions: vec![
                Cond::AttrDefaults("causal".into(), Value::Bool(false)),
                Cond::AttrAbsent("window".into()),
                Cond::AttrAbsent("softcap".into()),
                Cond::AttrDefaults("kv_groups".into(), Value::U(1)),
                Cond::AttrPresent("scale".into()),
            ],
            emit: vec![
                EmitOp {
                    copy_attrs: vec![("scale".into(), "value".into())],
                    out_types: vec![TypeSpec::DtypeLike {
                        name: "q".into(),
                        shape: Vec::new(),
                    }],
                    ..emit("omni.core", "constant", &[], &["scale_t"])
                },
                with_attrs(
                    emit("omni.tensor", "transpose", &["k"], &["kt"]),
                    vec![(
                        "perm",
                        Value::Array(vec![Value::U(0), Value::U(1), Value::U(3), Value::U(2)]),
                    )],
                ),
                emit("omni.tensor", "matmul", &["q", "kt"], &["scores"]),
                emit("omni.tensor", "mul", &["scores", "scale_t"], &["scaled"]),
                with_attrs(
                    emit("omni.tensor", "softmax", &["scaled"], &["probs"]),
                    vec![("axis", Value::I(-1))],
                ),
                emit("omni.tensor", "matmul", &["probs", "v"], &["out"]),
            ],
            results: vec!["out".into()],
            soundness: Soundness::SemanticsPreserving,
            to_level: Some(Level::Primitive),
        },
        // norm(x, w){rms} → x · rsqrt(mean(x²) + eps) · w
        Rewrite {
            name: "omni.nn/norm@1[rms]→primitive".into(),
            dialect: "omni.nn".into(),
            op: "norm".into(),
            version: 1,
            binds: vec!["x".into(), "w".into()],
            conditions: vec![
                Cond::AttrEq("kind".into(), Value::text("rms")),
                Cond::AttrPresent("eps".into()),
            ],
            emit: vec![
                emit("omni.tensor", "mul", &["x", "x"], &["sq"]),
                with_attrs(
                    emit("omni.tensor", "reduce", &["sq"], &["ms"]),
                    vec![
                        ("kind", Value::text("mean")),
                        ("axes", Value::Array(vec![Value::I(-1)])),
                        ("keepdims", Value::Bool(true)),
                    ],
                ),
                EmitOp {
                    copy_attrs: vec![("eps".into(), "value".into())],
                    out_types: vec![TypeSpec::DtypeLike {
                        name: "x".into(),
                        shape: Vec::new(),
                    }],
                    ..emit("omni.core", "constant", &[], &["eps_t"])
                },
                emit("omni.tensor", "add", &["ms", "eps_t"], &["den"]),
                emit("omni.tensor", "rsqrt", &["den"], &["inv"]),
                emit("omni.tensor", "mul", &["x", "inv"], &["normed"]),
                emit("omni.tensor", "mul", &["normed", "w"], &["out"]),
            ],
            results: vec!["out".into()],
            soundness: Soundness::SemanticsPreserving,
            to_level: Some(Level::Primitive),
        },
        // activation(x){silu} → x · sigmoid(x)
        Rewrite {
            name: "omni.nn/activation@1[silu]→primitive".into(),
            dialect: "omni.nn".into(),
            op: "activation".into(),
            version: 1,
            binds: vec!["x".into()],
            conditions: vec![Cond::AttrEq("kind".into(), Value::text("silu"))],
            emit: vec![
                emit("omni.tensor", "sigmoid", &["x"], &["s"]),
                emit("omni.tensor", "mul", &["x", "s"], &["out"]),
            ],
            results: vec!["out".into()],
            soundness: Soundness::SemanticsPreserving,
            to_level: Some(Level::Primitive),
        },
        // embedding(ids, table) → gather(table, ids)
        Rewrite {
            name: "omni.nn/embedding@1→primitive".into(),
            dialect: "omni.nn".into(),
            op: "embedding".into(),
            version: 1,
            binds: vec!["ids".into(), "table".into()],
            conditions: Vec::new(),
            emit: vec![with_attrs(
                emit("omni.tensor", "gather", &["table", "ids"], &["out"]),
                vec![("axis", Value::U(0))],
            )],
            results: vec!["out".into()],
            soundness: Soundness::SemanticsPreserving,
            to_level: Some(Level::Primitive),
        },
    ]
}

/// The op-version migrations `omni.nn` ships (§07.4.1).
///
/// §07.7's own worked example: a runtime that implements `attention@2` and not
/// `@1` consumes a v1 graph by applying this, which is the difference between
/// versioning per op and ONNX's monolithic opset.
pub fn shipped_migrations() -> Vec<Rewrite> {
    vec![Rewrite {
        name: "attention-v1-to-v2".into(),
        dialect: "omni.nn".into(),
        op: "attention".into(),
        version: 1,
        binds: vec!["q".into(), "k".into(), "v".into()],
        conditions: vec![Cond::AttrAbsent("kv_groups".into())],
        emit: vec![EmitOp {
            version: 2,
            attrs: vec![("kv_groups".into(), Value::U(1))],
            ..emit("omni.nn", "attention", &["q", "k", "v"], &["out"])
        }],
        results: vec!["out".into()],
        soundness: Soundness::SemanticsPreserving,
        to_level: None,
    }]
}

// --------------------------------------------------- binary op array (§07.9) --

/// The fixed-layout binary encoding of §07.9, for graphs too large to want a
/// CBOR parse per op.
///
/// §07.9 specifies the op record as
/// `dialect_id:u16, op_id:u16, version:u16, n_in:u16, n_out:u16, attr_off:u32,
/// in_off:u32` and leaves regions unaddressed; a record with no region fields
/// cannot represent `core.while`, so this adds `n_reg:u16` and `reg_off:u32` and
/// keeps everything else as written. Blocks and regions are ranges into flat
/// arrays, so the whole body is four arrays and a string table: one linear pass,
/// no allocation per op.
pub mod binary {
    use super::*;

    pub const MAGIC: &[u8; 4] = b"OIRB";
    pub const VERSION: u16 = 1;
    /// `dialect:u16 name:u16 version:u16 n_in:u16 n_out:u16 n_reg:u16`
    /// `attr:u32 in_off:u32 out_off:u32 reg_off:u32`
    pub const OP_RECORD: usize = 28;
    /// Magic, version, and the `(offset, length)` section descriptors, with
    /// room to add a section without moving anything.
    const HEADER: usize = 128;
    const SECTIONS: usize = 11;

    #[derive(Default)]
    struct Writer {
        strings: Vec<String>,
        types: Vec<Vec<u8>>,
        attrs: Vec<Vec<u8>>,
        ops: Vec<u8>,
        blocks: Vec<u8>,
        regions: Vec<u8>,
        inputs: Vec<u8>,
        outputs: Vec<u8>,
        args: Vec<u8>,
        locs: Vec<u8>,
        n_ops: u32,
        n_blocks: u32,
        n_regions: u32,
    }

    impl Writer {
        fn intern(&mut self, s: &str) -> u16 {
            match self.strings.iter().position(|x| x == s) {
                Some(i) => i as u16,
                None => {
                    self.strings.push(s.to_string());
                    (self.strings.len() - 1) as u16
                }
            }
        }

        fn intern_type(&mut self, t: &Type) -> u32 {
            let bytes = t.to_value().encode();
            match self.types.iter().position(|x| *x == bytes) {
                Some(i) => i as u32,
                None => {
                    self.types.push(bytes);
                    (self.types.len() - 1) as u32
                }
            }
        }

        /// Regions, blocks and ops are each written into a contiguous run, so a
        /// block's ops and a region's blocks are ranges rather than lists. That
        /// is what makes a linear pass possible — and it means the records have
        /// to be *reserved* before anything nested is written, or a nested
        /// region's ops would land in the middle of its parent's range.
        fn region(&mut self, r: &Region) -> u32 {
            let idx = self.n_regions;
            self.n_regions += 1;
            self.regions.extend_from_slice(&[0u8; 8]);

            let base = self.n_blocks;
            self.n_blocks += r.blocks.len() as u32;
            self.blocks
                .extend(std::iter::repeat_n(0u8, r.blocks.len() * 16));
            let at = idx as usize * 8;
            self.regions[at..at + 4].copy_from_slice(&base.to_le_bytes());
            self.regions[at + 4..at + 8].copy_from_slice(&(r.blocks.len() as u32).to_le_bytes());

            for (i, b) in r.blocks.iter().enumerate() {
                self.block(base + i as u32, b);
            }
            idx
        }

        fn block(&mut self, idx: u32, b: &Block) {
            let args_off = (self.args.len() / 8) as u32;
            for (id, t) in &b.args {
                let ti = self.intern_type(t);
                self.args.extend_from_slice(&id.to_le_bytes());
                self.args.extend_from_slice(&ti.to_le_bytes());
            }
            let base = self.n_ops;
            self.n_ops += b.ops.len() as u32;
            self.ops
                .extend(std::iter::repeat_n(0u8, b.ops.len() * OP_RECORD));
            self.locs
                .extend(std::iter::repeat_n(0xffu8, b.ops.len() * 4));
            let at = idx as usize * 16;
            self.blocks[at..at + 4].copy_from_slice(&args_off.to_le_bytes());
            self.blocks[at + 4..at + 8].copy_from_slice(&(b.args.len() as u32).to_le_bytes());
            self.blocks[at + 8..at + 12].copy_from_slice(&base.to_le_bytes());
            self.blocks[at + 12..at + 16].copy_from_slice(&(b.ops.len() as u32).to_le_bytes());

            for (i, op) in b.ops.iter().enumerate() {
                self.op(base + i as u32, op);
            }
            // Nested regions come after the whole block's records, so the block
            // keeps its contiguous op range.
            for (i, op) in b.ops.iter().enumerate() {
                if op.regions.is_empty() {
                    continue;
                }
                let mut first = None;
                for r in &op.regions {
                    let id = self.region(r);
                    if first.is_none() {
                        first = Some(id);
                    }
                }
                let rec = (base as usize + i) * OP_RECORD;
                self.ops[rec + 24..rec + 28].copy_from_slice(&first.unwrap_or(0).to_le_bytes());
            }
        }

        fn op(&mut self, idx: u32, op: &Op) {
            let d = self.intern(&op.dialect);
            let n = self.intern(&op.name);
            let in_off = (self.inputs.len() / 4) as u32;
            for i in &op.inputs {
                self.inputs.extend_from_slice(&i.to_le_bytes());
            }
            let out_off = (self.outputs.len() / 8) as u32;
            for (id, t) in &op.outputs {
                let ti = self.intern_type(t);
                self.outputs.extend_from_slice(&id.to_le_bytes());
                self.outputs.extend_from_slice(&ti.to_le_bytes());
            }
            let attr = if op.attrs.is_empty() {
                u32::MAX
            } else {
                let v = Value::Map(
                    op.attrs
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                );
                self.attrs.push(v.encode());
                (self.attrs.len() - 1) as u32
            };
            // Source locations are debug information, not semantics, so they
            // live beside the records rather than in them: one string index per
            // op, `u32::MAX` for none.
            let loc = match &op.loc {
                Some(l) => self.intern(l) as u32,
                None => u32::MAX,
            };
            let lat = idx as usize * 4;
            if self.locs.len() < lat + 4 {
                self.locs.resize(lat + 4, 0xff);
            }
            self.locs[lat..lat + 4].copy_from_slice(&loc.to_le_bytes());
            let at = idx as usize * OP_RECORD;
            let rec = &mut self.ops[at..at + OP_RECORD];
            rec[0..2].copy_from_slice(&d.to_le_bytes());
            rec[2..4].copy_from_slice(&n.to_le_bytes());
            rec[4..6].copy_from_slice(&(op.version as u16).to_le_bytes());
            rec[6..8].copy_from_slice(&(op.inputs.len() as u16).to_le_bytes());
            rec[8..10].copy_from_slice(&(op.outputs.len() as u16).to_le_bytes());
            rec[10..12].copy_from_slice(&(op.regions.len() as u16).to_le_bytes());
            rec[12..16].copy_from_slice(&attr.to_le_bytes());
            rec[16..20].copy_from_slice(&in_off.to_le_bytes());
            rec[20..24].copy_from_slice(&out_off.to_le_bytes());
        }
    }

    fn section(out: &mut Vec<u8>, data: &[u8]) -> (u32, u32) {
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        let off = out.len() as u32;
        out.extend_from_slice(data);
        (off, data.len() as u32)
    }

    fn blob_table(items: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(items.len() as u32).to_le_bytes());
        for i in items {
            out.extend_from_slice(&(i.len() as u32).to_le_bytes());
            out.extend_from_slice(i);
        }
        out
    }

    /// Encodes one function's body as a `Blob` (§07.9's split representation).
    pub fn encode(f: &Function) -> Vec<u8> {
        let mut w = Writer::default();
        // Interning the function's own types first keeps the parameter types at
        // low indices, where a reader looks first.
        let mut head_types: Vec<u32> = Vec::with_capacity(f.params.len() + f.results.len());
        for (_, t) in &f.params {
            head_types.push(w.intern_type(t));
        }
        for t in &f.results {
            head_types.push(w.intern_type(t));
        }
        let root = w.region(&f.body);
        let head = Value::map(vec![
            (
                "params",
                Value::Array(
                    f.params
                        .iter()
                        .zip(head_types.iter())
                        .map(|((n, _), ti)| {
                            Value::Array(vec![Value::text(n.clone()), Value::U(*ti as u64)])
                        })
                        .collect(),
                ),
            ),
            (
                "results",
                Value::Array(
                    head_types[f.params.len()..]
                        .iter()
                        .map(|ti| Value::U(*ti as u64))
                        .collect(),
                ),
            ),
            (
                "attrs",
                Value::Map(
                    f.attrs
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ),
            (
                "constraints",
                Value::Array(f.constraints.iter().map(Constraint::to_value).collect()),
            ),
            ("root", Value::U(root as u64)),
        ])
        .encode();

        let strings = blob_table(
            &w.strings
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>(),
        );
        let types = blob_table(&w.types);
        let attrs = blob_table(&w.attrs);

        let mut out = vec![0u8; HEADER];
        out[0..4].copy_from_slice(MAGIC);
        out[4..6].copy_from_slice(&VERSION.to_le_bytes());
        let mut fields: Vec<(u32, u32)> = Vec::new();
        for data in [
            &head[..],
            &strings[..],
            &types[..],
            &attrs[..],
            &w.ops[..],
            &w.blocks[..],
            &w.regions[..],
            &w.inputs[..],
            &w.outputs[..],
            &w.args[..],
            &w.locs[..],
        ] {
            fields.push(section(&mut out, data));
        }
        for (i, (off, len)) in fields.iter().enumerate() {
            let at = 8 + i * 8;
            out[at..at + 4].copy_from_slice(&off.to_le_bytes());
            out[at + 4..at + 8].copy_from_slice(&len.to_le_bytes());
        }
        out
    }

    struct Reader<'a> {
        d: &'a [u8],
        strings: Vec<&'a str>,
        types: Vec<Type>,
        attrs: Vec<Vec<(String, Value)>>,
        ops: &'a [u8],
        blocks: &'a [u8],
        regions: &'a [u8],
        inputs: &'a [u8],
        outputs: &'a [u8],
        args: &'a [u8],
        locs: &'a [u8],
    }

    fn u32at(d: &[u8], at: usize) -> Res<u32> {
        d.get(at..at + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .ok_or_else(|| Error("binary graph is truncated".into()))
    }

    fn u16at(d: &[u8], at: usize) -> Res<u16> {
        d.get(at..at + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]))
            .ok_or_else(|| Error("binary graph is truncated".into()))
    }

    fn slice(d: &[u8], off: u32, len: u32) -> Res<&[u8]> {
        d.get(off as usize..off as usize + len as usize)
            .ok_or_else(|| Error("a binary graph section is out of range".into()))
    }

    fn blobs(d: &[u8]) -> Res<Vec<&[u8]>> {
        let n = u32at(d, 0)? as usize;
        let mut out = Vec::with_capacity(n.min(1 << 16));
        let mut at = 4;
        for _ in 0..n {
            let len = u32at(d, at)? as usize;
            at += 4;
            out.push(
                d.get(at..at + len)
                    .ok_or_else(|| Error("a binary graph table entry is truncated".into()))?,
            );
            at += len;
        }
        Ok(out)
    }

    /// Decodes what [`encode`] wrote.
    pub fn decode(d: &[u8]) -> Res<Function> {
        if d.len() < HEADER || &d[0..4] != MAGIC {
            return err("not a binary OMNI-IR body");
        }
        if u16at(d, 4)? != VERSION {
            return err(format!("binary graph version {} is not 1", u16at(d, 4)?));
        }
        let mut sections = Vec::new();
        for i in 0..SECTIONS {
            sections.push((u32at(d, 8 + i * 8)?, u32at(d, 12 + i * 8)?));
        }
        let head = crate::cbor::decode(slice(d, sections[0].0, sections[0].1)?)
            .map_err(|e| Error(format!("binary graph head: {e}")))?;
        let strings = blobs(slice(d, sections[1].0, sections[1].1)?)?
            .into_iter()
            .map(|b| std::str::from_utf8(b).map_err(|_| Error("a name is not UTF-8".into())))
            .collect::<Res<Vec<_>>>()?;
        let types = blobs(slice(d, sections[2].0, sections[2].1)?)?
            .into_iter()
            .map(|b| {
                let v = crate::cbor::decode(b).map_err(|e| Error(format!("a type: {e}")))?;
                Type::from_value(&v)
            })
            .collect::<Res<Vec<_>>>()?;
        let attrs = blobs(slice(d, sections[3].0, sections[3].1)?)?
            .into_iter()
            .map(|b| {
                let v = crate::cbor::decode(b).map_err(|e| Error(format!("attributes: {e}")))?;
                Ok(match v {
                    Value::Map(m) => m
                        .iter()
                        .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                        .collect(),
                    _ => Vec::new(),
                })
            })
            .collect::<Res<Vec<_>>>()?;
        let r = Reader {
            d,
            strings,
            types,
            attrs,
            ops: slice(d, sections[4].0, sections[4].1)?,
            blocks: slice(d, sections[5].0, sections[5].1)?,
            regions: slice(d, sections[6].0, sections[6].1)?,
            inputs: slice(d, sections[7].0, sections[7].1)?,
            outputs: slice(d, sections[8].0, sections[8].1)?,
            args: slice(d, sections[9].0, sections[9].1)?,
            locs: slice(d, sections[10].0, sections[10].1)?,
        };
        let _ = r.d;

        let type_at = |i: u64| -> Res<Type> {
            r.types
                .get(i as usize)
                .cloned()
                .ok_or_else(|| Error("a type index is out of range".into()))
        };
        let mut params = Vec::new();
        if let Some(Value::Array(a)) = head.get("params") {
            for p in a {
                match p {
                    Value::Array(pair) if pair.len() == 2 => params.push((
                        pair[0]
                            .as_str()
                            .ok_or_else(|| Error("a parameter name is not text".into()))?
                            .to_string(),
                        type_at(pair[1].as_u64().unwrap_or(u64::MAX))?,
                    )),
                    _ => return err("a binary parameter is malformed"),
                }
            }
        }
        let mut results = Vec::new();
        if let Some(Value::Array(a)) = head.get("results") {
            for t in a {
                results.push(type_at(t.as_u64().unwrap_or(u64::MAX))?);
            }
        }
        let root = head
            .get("root")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| Error("a binary graph has no root region".into()))?;
        let body = read_region(&r, root as u32)?;
        Ok(Function {
            params,
            results,
            attrs: match head.get("attrs") {
                Some(Value::Map(m)) => m
                    .iter()
                    .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                    .collect(),
                _ => Vec::new(),
            },
            body,
            constraints: match head.get("constraints") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(Constraint::from_value)
                    .collect::<Res<Vec<_>>>()?,
                _ => Vec::new(),
            },
        })
    }

    fn read_region(r: &Reader<'_>, idx: u32) -> Res<Region> {
        let at = idx as usize * 8;
        if at + 8 > r.regions.len() {
            return err("a region index is out of range");
        }
        let first = u32at(r.regions, at)?;
        let count = u32at(r.regions, at + 4)?;
        let mut blocks = Vec::with_capacity(count.min(1 << 16) as usize);
        for i in 0..count {
            blocks.push(read_block(r, first + i)?);
        }
        Ok(Region { blocks })
    }

    fn read_block(r: &Reader<'_>, idx: u32) -> Res<Block> {
        let at = idx as usize * 16;
        if at + 16 > r.blocks.len() {
            return err("a block index is out of range");
        }
        let args_off = u32at(r.blocks, at)? as usize;
        let n_args = u32at(r.blocks, at + 4)? as usize;
        let ops_off = u32at(r.blocks, at + 8)?;
        let n_ops = u32at(r.blocks, at + 12)?;
        let mut args = Vec::with_capacity(n_args);
        for i in 0..n_args {
            let a = (args_off + i) * 8;
            let id = u32at(r.args, a)?;
            let ti = u32at(r.args, a + 4)?;
            args.push((
                id,
                r.types
                    .get(ti as usize)
                    .cloned()
                    .ok_or_else(|| Error("a block argument type index is out of range".into()))?,
            ));
        }
        let mut ops = Vec::with_capacity(n_ops.min(1 << 20) as usize);
        for i in 0..n_ops {
            ops.push(read_op(r, ops_off + i)?);
        }
        Ok(Block { args, ops })
    }

    fn read_op(r: &Reader<'_>, idx: u32) -> Res<Op> {
        let at = idx as usize * OP_RECORD;
        if at + OP_RECORD > r.ops.len() {
            return err("an op index is out of range");
        }
        let rec = &r.ops[at..at + OP_RECORD];
        let name_of = |i: u16| -> Res<String> {
            r.strings
                .get(i as usize)
                .map(|s| s.to_string())
                .ok_or_else(|| Error("a name index is out of range".into()))
        };
        let n_in = u16at(rec, 6)? as usize;
        let n_out = u16at(rec, 8)? as usize;
        let n_reg = u16at(rec, 10)? as usize;
        let attr = u32at(rec, 12)?;
        let in_off = u32at(rec, 16)? as usize;
        let out_off = u32at(rec, 20)? as usize;
        let reg_off = u32at(rec, 24)?;
        let mut inputs = Vec::with_capacity(n_in);
        for i in 0..n_in {
            inputs.push(u32at(r.inputs, (in_off + i) * 4)?);
        }
        let mut outputs = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let a = (out_off + i) * 8;
            let id = u32at(r.outputs, a)?;
            let ti = u32at(r.outputs, a + 4)?;
            outputs.push((
                id,
                r.types
                    .get(ti as usize)
                    .cloned()
                    .ok_or_else(|| Error("a result type index is out of range".into()))?,
            ));
        }
        let mut regions = Vec::with_capacity(n_reg);
        for i in 0..n_reg {
            regions.push(read_region(r, reg_off + i as u32)?);
        }
        let loc = match u32at(r.locs, idx as usize * 4) {
            Ok(u32::MAX) | Err(_) => None,
            Ok(i) => r.strings.get(i as usize).map(|s| s.to_string()),
        };
        Ok(Op {
            dialect: name_of(u16at(rec, 0)?)?,
            name: name_of(u16at(rec, 2)?)?,
            version: u16at(rec, 4)? as u32,
            inputs,
            outputs,
            attrs: if attr == u32::MAX {
                Vec::new()
            } else {
                r.attrs
                    .get(attr as usize)
                    .cloned()
                    .ok_or_else(|| Error("an attribute index is out of range".into()))?
            },
            regions,
            loc,
        })
    }
}

// ------------------------------------------------------------- synthesis --

/// Builds an OMNI-IR module for a registered architecture family from its
/// `arch.params` (§07.5).
///
/// This is the upgrade path for the safetensors case: a weights-only model is
/// legal, common and *not portable*, and §07.5 says so out loud rather than
/// hiding it. `omni graph synthesize` closes the gap for families the working
/// group has registered, and refuses — naming what it wanted — for anything
/// else, because a graph that guesses at an architecture is worse than no graph.
///
/// `available` is the model's tensor names. Every weight the family needs is
/// looked up there, so a synthesized graph is checked against the weights that
/// exist before it is written, not after.
/// The families this build can synthesize a graph for.
///
/// Gate 2 wants ten. Each one added here has to be *executable* — `omni graph
/// run` over real weights — because a synthesizer that emits a well-typed graph
/// nobody has run is how `transformer.decoder` came to attend across heads
/// instead of positions and pass verification while doing it.
pub const FAMILIES: &[&str] = &[
    "transformer.decoder",
    "transformer.encoder",
    "transformer.moe",
    "cnn.classifier",
    "mlp",
    "rnn.lstm",
    "rnn.gru",
    "gnn.mpnn",
    "rl.actor_critic",
    "audio.encoder",
];

pub fn synthesize(family: &str, params: &Value, available: &[String]) -> Result<Module, String> {
    match family {
        "transformer.decoder" => synthesize_transformer(params, available, true),
        // The one difference that matters is the mask, and it is a difference in
        // *meaning* rather than in shape: without it every position sees the
        // future, which is what a bidirectional encoder is for and what a
        // decoder must never do. Everything else — the projections, the
        // grouping, the norm — is the same graph, so it is the same
        // synthesizer with the flag rather than a copy that can drift.
        "transformer.encoder" => synthesize_transformer(params, available, false),
        "transformer.moe" => synthesize_moe(params, available),
        "cnn.classifier" => synthesize_cnn(params, available),
        "mlp" => synthesize_mlp(params, available),
        // One synthesizer, two cells, for the same reason the encoder shares
        // the decoder's: the difference is the gate arithmetic, and a copy
        // would drift.
        "rnn.lstm" => synthesize_rnn(params, available, true),
        "rnn.gru" => synthesize_rnn(params, available, false),
        "gnn.mpnn" => synthesize_gnn(params, available),
        "rl.actor_critic" => synthesize_rl(params, available),
        "audio.encoder" => synthesize_audio(params, available),
        other => Err(format!(
            "no synthesizer for family `{other}`; this build knows {}",
            FAMILIES.join(", ")
        )),
    }
}

/// Looks up every weight a family needs, collecting the misses so the caller is
/// told all of them at once rather than one per attempt.
fn require(available: &[String], names: Vec<String>) -> Result<Vec<String>, String> {
    let missing: Vec<&String> = names.iter().filter(|n| !available.contains(n)).collect();
    if !missing.is_empty() {
        return Err(format!(
            "the model does not carry {} weight(s) this family needs: {}",
            missing.len(),
            missing
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(", ")
        ));
    }
    Ok(names)
}

fn synthesize_transformer(
    params: &Value,
    available: &[String],
    causal: bool,
) -> Result<Module, String> {
    let get = |k: &str| -> Option<u64> { params.get(k).and_then(|v| v.as_u64()) };
    let hidden = get("hidden_size").ok_or("arch.params has no `hidden_size`")?;
    let layers = get("n_layers").ok_or("arch.params has no `n_layers`")?;
    let heads = get("n_heads").ok_or("arch.params has no `n_heads`")?;
    let kv_heads = get("n_kv_heads").unwrap_or(heads);
    if heads == 0 || hidden % heads != 0 {
        return Err(format!(
            "hidden_size {hidden} is not divisible by n_heads {heads}"
        ));
    }
    let head_dim = hidden / heads;
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
        .to_string();
    let norm_kind = params
        .get("norm")
        .and_then(|v| v.get("kind"))
        .and_then(|v| v.as_str())
        .unwrap_or("rms")
        .to_string();
    let norm_eps = params
        .get("norm")
        .and_then(|v| v.get("eps"))
        .and_then(|v| match v {
            Value::F64(f) => Some(*f),
            Value::U(n) => Some(*n as f64),
            _ => None,
        })
        .unwrap_or(1e-5);
    let rope_theta = params
        .get("rope")
        .and_then(|v| v.get("theta"))
        .and_then(|v| match v {
            Value::F64(f) => Some(*f),
            Value::U(n) => Some(*n as f64),
            _ => None,
        });
    let rope_interleaved = params
        .get("rope")
        .and_then(|v| v.get("interleaved"))
        .and_then(|v| match v {
            Value::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(false);

    // The weights this family needs, by the names §06's metadata conventions
    // give them. Anything missing is named rather than assumed.
    let mut missing = Vec::new();
    let mut need = |name: String| -> String {
        if !available.contains(&name) {
            missing.push(name.clone());
        }
        name
    };
    let embed = need("model.embed_tokens.weight".into());
    // An encoder has no language-modelling head: its result is the hidden
    // states, and a classifier or a retriever puts its own head on top. Asking
    // for `lm_head.weight` and not finding it would refuse a perfectly good
    // BERT.
    let lm_head = causal.then(|| need("lm_head.weight".into()));
    let mut per_layer = Vec::new();
    for l in 0..layers {
        per_layer.push((
            need(format!("model.layers.{l}.norm.weight")),
            [
                need(format!("model.layers.{l}.attn.q_proj.weight")),
                need(format!("model.layers.{l}.attn.k_proj.weight")),
                need(format!("model.layers.{l}.attn.v_proj.weight")),
                need(format!("model.layers.{l}.attn.o_proj.weight")),
            ],
        ));
    }
    if !missing.is_empty() {
        return Err(format!(
            "the model does not carry {} weight(s) this family needs: {}",
            missing.len(),
            missing.join(", ")
        ));
    }

    // Symbolic batch and sequence dimensions: dynamic shapes are the default,
    // not an afterthought (§07.3.1).
    let b = Dim::Sym("B".into());
    let s = Dim::Sym("S".into());
    let dt = DType::BF16;
    let hid = |extra: Option<u64>| -> Type {
        let mut shape = vec![b.clone(), s.clone()];
        match extra {
            Some(n) => shape.push(Dim::N(n)),
            None => shape.push(Dim::N(hidden)),
        }
        Type::tensor(shape, dt.clone())
    };

    let mut next = 1u32; // 0 is the `tokens` parameter
    let mut fresh = || {
        let id = next;
        next += 1;
        id
    };
    let mut ops: Vec<Op> = Vec::new();

    // Weights enter the graph as constants naming tensors; R-I10 checks each
    // against the tensor table, so the graph cannot drift from the weights.
    let constant = |name: &str, shape: Vec<Dim>, id: u32| -> Op {
        Op::new("omni.core", "constant", 1)
            .with_attr("tensor", Value::text(name))
            .with_output(id, Type::tensor(shape, dt.clone()))
    };

    let embed_id = fresh();
    ops.push(constant(
        &embed,
        vec![Dim::Dynamic, Dim::N(hidden)],
        embed_id,
    ));
    let mut h = fresh();
    ops.push(
        Op::new("omni.nn", "embedding", 1)
            .with_inputs(&[0, embed_id])
            .with_output(h, hid(None)),
    );

    for (norm_w, projs) in &per_layer {
        let nw = fresh();
        ops.push(
            Op::new("omni.core", "constant", 1)
                .with_attr("tensor", Value::text(norm_w.clone()))
                .with_output(nw, Type::tensor(vec![Dim::N(hidden)], DType::F32)),
        );
        // §07.10 wants nothing implicit in the numerics, and a norm weight is
        // routinely stored in f32 while the activations are bf16. The cast is
        // therefore in the graph, where a reader can see it, rather than in a
        // runtime's assumptions.
        let nwc = fresh();
        ops.push(
            Op::new("omni.tensor", "cast", 1)
                .with_inputs(&[nw])
                .with_attr("dtype", dt.clone().to_value())
                .with_attr("round", Value::text("rne"))
                .with_output(nwc, Type::tensor(vec![Dim::N(hidden)], dt.clone())),
        );
        let normed = fresh();
        let mut norm_op = Op::new("omni.nn", "norm", 1)
            .with_inputs(&[h, nwc])
            .with_attr("kind", Value::text(norm_kind.clone()))
            .with_attr("eps", Value::F64(norm_eps))
            .with_output(normed, hid(None));
        norm_op.loc = Some("synthesized from arch.params".into());
        ops.push(norm_op);

        // q/k/v projections. k and v are narrower when the model uses grouped
        // query attention, which is why the head counts are separate params.
        let mut proj_out = Vec::new();
        for (i, name) in projs[..3].iter().enumerate() {
            let out_features = if i == 0 { hidden } else { head_dim * kv_heads };
            let w = fresh();
            ops.push(constant(
                name,
                vec![Dim::N(out_features), Dim::N(hidden)],
                w,
            ));
            let wt = fresh();
            ops.push(
                Op::new("omni.tensor", "transpose", 1)
                    .with_inputs(&[w])
                    .with_attr("perm", Value::Array(vec![Value::U(1), Value::U(0)]))
                    .with_output(
                        wt,
                        Type::tensor(vec![Dim::N(hidden), Dim::N(out_features)], dt.clone()),
                    ),
            );
            let p = fresh();
            ops.push(
                Op::new("omni.tensor", "matmul", 1)
                    .with_inputs(&[normed, wt])
                    .with_output(p, hid(Some(out_features))),
            );
            proj_out.push((p, out_features));
        }

        // Heads: [B, S, H·Dh] → [B, H, S, Dh].
        let mut headed = Vec::new();
        for (i, (p, out_features)) in proj_out.iter().enumerate() {
            let n_heads = out_features / head_dim;
            let r = fresh();
            ops.push(
                Op::new("omni.tensor", "reshape", 1)
                    .with_inputs(&[*p])
                    .with_attr(
                        "shape",
                        Value::Array(vec![Value::I(-1), Value::U(n_heads), Value::U(head_dim)]),
                    )
                    .with_output(
                        r,
                        Type::tensor(
                            vec![Dim::Dynamic, Dim::N(n_heads), Dim::N(head_dim)],
                            dt.clone(),
                        ),
                    ),
            );
            let mut cur = r;
            // RoPE applies to q and k only (§06's rope params).
            if i < 2 {
                if let Some(theta) = rope_theta {
                    let roped = fresh();
                    ops.push(
                        Op::new("omni.nn", "rope", 1)
                            .with_inputs(&[cur])
                            .with_attr("theta", Value::F64(theta))
                            .with_attr("interleaved", Value::Bool(rope_interleaved))
                            .with_output(
                                roped,
                                Type::tensor(
                                    vec![Dim::Dynamic, Dim::N(n_heads), Dim::N(head_dim)],
                                    dt.clone(),
                                ),
                            ),
                    );
                    cur = roped;
                }
            }
            headed.push((cur, n_heads));
        }

        // §07.8's `attention` contracts over the *last two* axes: keys against
        // queries, then head dimension. The projections above are
        // `[B*S, heads, head_dim]`, where the last two are heads and head
        // dimension — so without this transpose the op would attend across the
        // heads of a single token instead of across positions, which is not
        // attention at all. Found by running the graph (`crate::interp`) rather
        // than by verifying it: shapes and types agree either way.
        let mut posed = Vec::new();
        for (t, n_heads) in &headed {
            let p = fresh();
            ops.push(
                Op::new("omni.tensor", "transpose", 1)
                    .with_inputs(&[*t])
                    .with_attr(
                        "perm",
                        Value::Array(vec![Value::U(1), Value::U(0), Value::U(2)]),
                    )
                    .with_output(
                        p,
                        Type::tensor(
                            vec![Dim::N(*n_heads), Dim::Dynamic, Dim::N(head_dim)],
                            dt.clone(),
                        ),
                    ),
            );
            posed.push(p);
        }

        let attn = fresh();
        let mut attn_op = Op::new("omni.nn", "attention", 2)
            .with_inputs(&[posed[0], posed[1], posed[2]])
            .with_attr("causal", Value::Bool(causal))
            .with_attr("kv_groups", Value::U(heads / kv_heads.max(1)))
            // The scale is written explicitly: a shipped lowering cannot know
            // the head dimension, and §07.10 wants nothing implicit in the
            // numerics anyway.
            .with_attr("scale", Value::F64(1.0 / (head_dim as f64).sqrt()))
            .with_output(
                attn,
                Type::tensor(
                    vec![Dim::N(heads), Dim::Dynamic, Dim::N(head_dim)],
                    dt.clone(),
                ),
            );
        attn_op.loc = Some("synthesized from arch.params".into());
        ops.push(attn_op);

        // Back to position-major, so the heads concatenate along the hidden axis.
        let unposed = fresh();
        ops.push(
            Op::new("omni.tensor", "transpose", 1)
                .with_inputs(&[attn])
                .with_attr(
                    "perm",
                    Value::Array(vec![Value::U(1), Value::U(0), Value::U(2)]),
                )
                .with_output(
                    unposed,
                    Type::tensor(
                        vec![Dim::Dynamic, Dim::N(heads), Dim::N(head_dim)],
                        dt.clone(),
                    ),
                ),
        );

        let flat = fresh();
        ops.push(
            Op::new("omni.tensor", "reshape", 1)
                .with_inputs(&[unposed])
                .with_attr("shape", Value::Array(vec![Value::I(-1), Value::U(hidden)]))
                .with_output(
                    flat,
                    Type::tensor(vec![Dim::Dynamic, Dim::N(hidden)], dt.clone()),
                ),
        );
        let ow = fresh();
        ops.push(constant(
            &projs[3],
            vec![Dim::N(hidden), Dim::N(hidden)],
            ow,
        ));
        let owt = fresh();
        ops.push(
            Op::new("omni.tensor", "transpose", 1)
                .with_inputs(&[ow])
                .with_attr("perm", Value::Array(vec![Value::U(1), Value::U(0)]))
                .with_output(
                    owt,
                    Type::tensor(vec![Dim::N(hidden), Dim::N(hidden)], dt.clone()),
                ),
        );
        let projected = fresh();
        ops.push(
            Op::new("omni.tensor", "matmul", 1)
                .with_inputs(&[flat, owt])
                .with_output(
                    projected,
                    Type::tensor(vec![Dim::Dynamic, Dim::N(hidden)], dt.clone()),
                ),
        );
        // The residual stream.
        let sum = fresh();
        ops.push(
            Op::new("omni.tensor", "add", 1)
                .with_inputs(&[h, projected])
                .with_output(sum, hid(None)),
        );
        h = sum;
    }

    // The output head. `activation` is recorded on the module rather than
    // fabricated into a feed-forward block the weights do not describe: this
    // model has no FFN tensors, and inventing them would be exactly the
    // fabrication importer rule I1 forbids.
    let result = match &lm_head {
        Some(name) => {
            let hw = fresh();
            ops.push(constant(name, vec![Dim::Dynamic, Dim::N(hidden)], hw));
            let hwt = fresh();
            let vocab = Dim::Dynamic;
            ops.push(
                Op::new("omni.tensor", "transpose", 1)
                    .with_inputs(&[hw])
                    .with_attr("perm", Value::Array(vec![Value::U(1), Value::U(0)]))
                    .with_output(
                        hwt,
                        Type::tensor(vec![Dim::N(hidden), vocab.clone()], dt.clone()),
                    ),
            );
            let logits = fresh();
            ops.push(
                Op::new("omni.tensor", "matmul", 1)
                    .with_inputs(&[h, hwt])
                    .with_output(
                        logits,
                        Type::tensor(vec![b.clone(), s.clone(), vocab.clone()], dt.clone()),
                    ),
            );
            ops.push(Op::new("omni.core", "return", 1).with_inputs(&[logits]));
            Type::tensor(vec![b.clone(), s.clone(), vocab], dt.clone())
        }
        None => {
            ops.push(Op::new("omni.core", "return", 1).with_inputs(&[h]));
            Type::tensor(vec![b.clone(), s.clone(), Dim::N(hidden)], dt.clone())
        }
    };

    let f = Function {
        params: vec![(
            "tokens".into(),
            Type::tensor(
                vec![b, s],
                DType::Int {
                    w: 32,
                    signed: true,
                },
            ),
        )],
        results: vec![result],
        attrs: vec![
            ("kind".into(), Value::text("forward")),
            ("activation".into(), Value::text(activation)),
        ],
        body: Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops,
            }],
        },
        constraints: vec![
            // Exactly one, not "at least one". `omni.tensor/reshape` takes static
            // extents and at most one `-1`, and both B and S are symbolic — so
            // the only reshape this synthesizer can write collapses them into a
            // single axis. A batch of more than one would then let attention
            // attend across the boundary between sequences, so the graph declares
            // the shape it is actually correct for rather than the shape it looks
            // like it handles. Declaring `>= 1` here is how a wrong answer would
            // have become a silent one.
            Constraint {
                dim: "B".into(),
                rel: Rel::Eq,
                bound: 1,
            },
            Constraint {
                dim: "S".into(),
                rel: Rel::Ge,
                bound: 1,
            },
        ],
    };

    let mut m = Module::new(Level::Semantic, "forward");
    m.dialects = vec![
        DialectUse {
            ns: "omni.core".into(),
            version: 1,
            reference: None,
        },
        DialectUse {
            ns: "omni.tensor".into(),
            version: 1,
            reference: None,
        },
        DialectUse {
            ns: "omni.nn".into(),
            version: 1,
            reference: None,
        },
    ];
    m.attrs = vec![
        (
            "family".into(),
            Value::text(if causal {
                "transformer.decoder"
            } else {
                "transformer.encoder"
            }),
        ),
        ("synthesized".into(), Value::Bool(true)),
    ];
    m.functions = vec![("forward".into(), f)];
    Ok(m)
}

/// `mlp` — a stack of affine layers with an activation between them.
///
/// The simplest family that is still a model, and the one worth having first:
/// every mechanism a bigger graph uses (a weight named as a constant, a
/// transpose, a matmul, a bias, an activation) appears exactly once, so when a
/// bigger family goes wrong this is the graph that isolates which mechanism.
///
/// Weights are `mlp.layers.{i}.weight` (`[out, in]`, the orientation every
/// framework stores) and an optional `mlp.layers.{i}.bias`.
fn synthesize_mlp(params: &Value, available: &[String]) -> Result<Module, String> {
    let sizes: Vec<u64> = match params.get("hidden_sizes") {
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or("a non-integer in `hidden_sizes`".to_string())
            })
            .collect::<Result<Vec<u64>, String>>()?,
        _ => return Err("arch.params has no `hidden_sizes`".into()),
    };
    if sizes.len() < 2 {
        return Err("`hidden_sizes` needs at least an input and an output".into());
    }
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("relu")
        .to_string();
    let n = sizes.len() - 1;
    let weights = require(
        available,
        (0..n).map(|i| format!("mlp.layers.{i}.weight")).collect(),
    )?;
    // Biases are optional, and absent is a real answer: a layer without one is a
    // linear map, and inventing a zero bias would be inventing a tensor.
    let biases: Vec<Option<String>> = (0..n)
        .map(|i| {
            let name = format!("mlp.layers.{i}.bias");
            available.contains(&name).then_some(name)
        })
        .collect();

    let dt = DType::F32;
    let b = Dim::Sym("B".into());
    let mut next = 1u32;
    let mut fresh = || {
        let id = next;
        next += 1;
        id
    };
    let mut ops: Vec<Op> = Vec::new();
    let mut h = 0u32;
    for i in 0..n {
        let (fan_in, fan_out) = (sizes[i], sizes[i + 1]);
        let w = fresh();
        ops.push(
            Op::new("omni.core", "constant", 1)
                .with_attr("tensor", Value::text(weights[i].clone()))
                .with_output(
                    w,
                    Type::tensor(vec![Dim::N(fan_out), Dim::N(fan_in)], dt.clone()),
                ),
        );
        let wt = fresh();
        ops.push(
            Op::new("omni.tensor", "transpose", 1)
                .with_inputs(&[w])
                .with_attr("perm", Value::Array(vec![Value::U(1), Value::U(0)]))
                .with_output(
                    wt,
                    Type::tensor(vec![Dim::N(fan_in), Dim::N(fan_out)], dt.clone()),
                ),
        );
        let y = fresh();
        ops.push(
            Op::new("omni.tensor", "matmul", 1)
                .with_inputs(&[h, wt])
                .with_output(
                    y,
                    Type::tensor(vec![b.clone(), Dim::N(fan_out)], dt.clone()),
                ),
        );
        h = y;
        if let Some(name) = &biases[i] {
            let bw = fresh();
            ops.push(
                Op::new("omni.core", "constant", 1)
                    .with_attr("tensor", Value::text(name.clone()))
                    .with_output(bw, Type::tensor(vec![Dim::N(fan_out)], dt.clone())),
            );
            let sum = fresh();
            ops.push(
                Op::new("omni.tensor", "add", 1)
                    .with_inputs(&[h, bw])
                    .with_output(
                        sum,
                        Type::tensor(vec![b.clone(), Dim::N(fan_out)], dt.clone()),
                    ),
            );
            h = sum;
        }
        // No activation on the output layer: the last layer's values are logits
        // or regression targets, and squashing them would be a different model.
        if i + 1 < n {
            let act = fresh();
            ops.push(
                Op::new("omni.nn", "activation", 1)
                    .with_inputs(&[h])
                    .with_attr("kind", Value::text(activation.clone()))
                    .with_output(
                        act,
                        Type::tensor(vec![b.clone(), Dim::N(fan_out)], dt.clone()),
                    ),
            );
            h = act;
        }
    }
    ops.push(Op::new("omni.core", "return", 1).with_inputs(&[h]));

    let out = Type::tensor(vec![b.clone(), Dim::N(sizes[n])], dt.clone());
    let f = Function {
        params: vec![("x".into(), Type::tensor(vec![b, Dim::N(sizes[0])], dt))],
        results: vec![out],
        attrs: vec![
            ("kind".into(), Value::text("forward")),
            ("activation".into(), Value::text(activation)),
        ],
        body: Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops,
            }],
        },
        constraints: vec![Constraint {
            dim: "B".into(),
            rel: Rel::Ge,
            bound: 1,
        }],
    };
    Ok(finish_module(
        "mlp",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// `cnn.classifier` — a 2-D convolutional stack over `[N, C, H, W]`, then a
/// linear head over the globally pooled features.
///
/// Weights are `cnn.blocks.{i}.conv.weight` (`[out_c, in_c, kh, kw]`) with an
/// optional `.bias`, and `cnn.head.weight` (`[classes, features]`) with an
/// optional `cnn.head.bias`.
fn synthesize_cnn(params: &Value, available: &[String]) -> Result<Module, String> {
    let channels: Vec<u64> = match params.get("channels") {
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|v| v.as_u64().ok_or("a non-integer in `channels`".to_string()))
            .collect::<Result<Vec<u64>, String>>()?,
        _ => return Err("arch.params has no `channels`".into()),
    };
    if channels.len() < 2 {
        return Err("`channels` needs an input channel count and at least one block".into());
    }
    let kernel = params
        .get("kernel")
        .and_then(|v| v.as_u64())
        .ok_or("arch.params has no `kernel`")?;
    if kernel == 0 {
        return Err("`kernel` is 0".into());
    }
    let classes = params
        .get("num_classes")
        .and_then(|v| v.as_u64())
        .ok_or("arch.params has no `num_classes`")?;
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("relu")
        .to_string();
    let pool = params
        .get("pool")
        .and_then(|v| v.as_u64())
        .unwrap_or(2)
        .max(1);
    // Input spatial extent has to be concrete: every block halves it, and the
    // head's feature count depends on what is left. A symbolic H would make the
    // head's shape unknowable, which is exactly the kind of graph that verifies
    // and cannot run.
    let (h0, w0) = (
        params
            .get("height")
            .and_then(|v| v.as_u64())
            .ok_or("arch.params has no `height`; a classifier head needs a concrete one")?,
        params
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or("arch.params has no `width`; a classifier head needs a concrete one")?,
    );

    let n = channels.len() - 1;
    let mut names: Vec<String> = (0..n)
        .map(|i| format!("cnn.blocks.{i}.conv.weight"))
        .collect();
    names.push("cnn.head.weight".into());
    let names = require(available, names)?;
    let head_w = names[n].clone();

    let dt = DType::F32;
    let bdim = Dim::Sym("B".into());
    let mut next = 1u32;
    let mut fresh = || {
        let id = next;
        next += 1;
        id
    };
    let mut ops: Vec<Op> = Vec::new();
    let mut x = 0u32;
    let pad = (kernel - 1) / 2;
    let (mut cur_h, mut cur_w) = (h0, w0);
    for i in 0..n {
        let (cin, cout) = (channels[i], channels[i + 1]);
        let wt = fresh();
        ops.push(
            Op::new("omni.core", "constant", 1)
                .with_attr("tensor", Value::text(names[i].clone()))
                .with_output(
                    wt,
                    Type::tensor(
                        vec![Dim::N(cout), Dim::N(cin), Dim::N(kernel), Dim::N(kernel)],
                        dt.clone(),
                    ),
                ),
        );
        // "Same" padding when the kernel is odd, which is the only case where
        // symmetric padding preserves the extent exactly.
        let conv_h = cur_h + 2 * pad - (kernel - 1);
        let conv_w = cur_w + 2 * pad - (kernel - 1);
        let c = fresh();
        let bias_name = format!("cnn.blocks.{i}.conv.bias");
        let mut conv_in = vec![x, wt];
        if available.contains(&bias_name) {
            let bw = fresh();
            ops.push(
                Op::new("omni.core", "constant", 1)
                    .with_attr("tensor", Value::text(bias_name))
                    .with_output(bw, Type::tensor(vec![Dim::N(cout)], dt.clone())),
            );
            conv_in.push(bw);
        }
        ops.push(
            Op::new("omni.nn", "conv", 1)
                .with_inputs(&conv_in)
                .with_attr("padding", Value::Array(vec![Value::U(pad), Value::U(pad)]))
                .with_output(
                    c,
                    Type::tensor(
                        vec![bdim.clone(), Dim::N(cout), Dim::N(conv_h), Dim::N(conv_w)],
                        dt.clone(),
                    ),
                ),
        );
        let act = fresh();
        ops.push(
            Op::new("omni.nn", "activation", 1)
                .with_inputs(&[c])
                .with_attr("kind", Value::text(activation.clone()))
                .with_output(
                    act,
                    Type::tensor(
                        vec![bdim.clone(), Dim::N(cout), Dim::N(conv_h), Dim::N(conv_w)],
                        dt.clone(),
                    ),
                ),
        );
        let (ph, pw) = (conv_h / pool, conv_w / pool);
        if ph == 0 || pw == 0 {
            return Err(format!(
                "block {i} pools {conv_h}x{conv_w} by {pool}, which leaves nothing"
            ));
        }
        let p = fresh();
        ops.push(
            Op::new("omni.nn", "pool", 1)
                .with_inputs(&[act])
                .with_attr("kind", Value::text("max"))
                .with_attr("window", Value::Array(vec![Value::U(pool), Value::U(pool)]))
                .with_output(
                    p,
                    Type::tensor(
                        vec![bdim.clone(), Dim::N(cout), Dim::N(ph), Dim::N(pw)],
                        dt.clone(),
                    ),
                ),
        );
        x = p;
        cur_h = ph;
        cur_w = pw;
    }

    // Global average pool, written as a pool whose window is whatever is left,
    // so the head sees one feature per channel however deep the stack got.
    let last_c = channels[n];
    let gp = fresh();
    ops.push(
        Op::new("omni.nn", "pool", 1)
            .with_inputs(&[x])
            .with_attr("kind", Value::text("avg"))
            .with_attr(
                "window",
                Value::Array(vec![Value::U(cur_h), Value::U(cur_w)]),
            )
            .with_output(
                gp,
                Type::tensor(
                    vec![bdim.clone(), Dim::N(last_c), Dim::N(1), Dim::N(1)],
                    dt.clone(),
                ),
            ),
    );
    let flat = fresh();
    ops.push(
        Op::new("omni.tensor", "reshape", 1)
            .with_inputs(&[gp])
            .with_attr("shape", Value::Array(vec![Value::I(-1), Value::U(last_c)]))
            .with_output(
                flat,
                Type::tensor(vec![Dim::Dynamic, Dim::N(last_c)], dt.clone()),
            ),
    );
    let hw = fresh();
    ops.push(
        Op::new("omni.core", "constant", 1)
            .with_attr("tensor", Value::text(head_w))
            .with_output(
                hw,
                Type::tensor(vec![Dim::N(classes), Dim::N(last_c)], dt.clone()),
            ),
    );
    let hwt = fresh();
    ops.push(
        Op::new("omni.tensor", "transpose", 1)
            .with_inputs(&[hw])
            .with_attr("perm", Value::Array(vec![Value::U(1), Value::U(0)]))
            .with_output(
                hwt,
                Type::tensor(vec![Dim::N(last_c), Dim::N(classes)], dt.clone()),
            ),
    );
    let mut logits = fresh();
    ops.push(
        Op::new("omni.tensor", "matmul", 1)
            .with_inputs(&[flat, hwt])
            .with_output(
                logits,
                Type::tensor(vec![bdim.clone(), Dim::N(classes)], dt.clone()),
            ),
    );
    if available.contains(&"cnn.head.bias".to_string()) {
        let hb = fresh();
        ops.push(
            Op::new("omni.core", "constant", 1)
                .with_attr("tensor", Value::text("cnn.head.bias"))
                .with_output(hb, Type::tensor(vec![Dim::N(classes)], dt.clone())),
        );
        let sum = fresh();
        ops.push(
            Op::new("omni.tensor", "add", 1)
                .with_inputs(&[logits, hb])
                .with_output(
                    sum,
                    Type::tensor(vec![bdim.clone(), Dim::N(classes)], dt.clone()),
                ),
        );
        logits = sum;
    }
    ops.push(Op::new("omni.core", "return", 1).with_inputs(&[logits]));

    let f = Function {
        params: vec![(
            "images".into(),
            Type::tensor(
                vec![bdim.clone(), Dim::N(channels[0]), Dim::N(h0), Dim::N(w0)],
                dt.clone(),
            ),
        )],
        results: vec![Type::tensor(vec![bdim, Dim::N(classes)], dt)],
        attrs: vec![
            ("kind".into(), Value::text("forward")),
            ("activation".into(), Value::text(activation)),
        ],
        body: Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops,
            }],
        },
        constraints: vec![Constraint {
            dim: "B".into(),
            rel: Rel::Ge,
            bound: 1,
        }],
    };
    Ok(finish_module(
        "cnn.classifier",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

// ------------------------------------------------------- six smaller families --

/// A local emitter for the families below.
///
/// The synthesizers above build their ops by hand, which reads well when a
/// graph is a straight line. These are not straight lines — one routes tokens
/// to gathered expert weights, two carry a region, one has two heads — and
/// hand-building them would bury what each family *is* under identical
/// `with_output(Type::tensor(vec![…]))` boilerplate. So the boilerplate is here
/// once, and each family below reads as the graph it emits.
struct Emit {
    ops: Vec<Op>,
    next: u32,
    dt: DType,
}

impl Emit {
    fn new(first_free: u32, dt: DType) -> Emit {
        Emit {
            ops: Vec::new(),
            next: first_free,
            dt,
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    fn ty(&self, shape: &[Dim]) -> Type {
        Type::tensor(shape.to_vec(), self.dt.clone())
    }

    /// A weight, by the name the tensor table gives it. R-I10 checks that name
    /// against the table, so a synthesized graph cannot drift from the weights
    /// it was synthesized for.
    fn constant(&mut self, tensor: &str, shape: &[Dim]) -> u32 {
        let id = self.id();
        let t = self.ty(shape);
        self.ops.push(
            Op::new("omni.core", "constant", 1)
                .with_attr("tensor", Value::text(tensor.to_string()))
                .with_output(id, t),
        );
        id
    }

    fn op(&mut self, dialect: &str, name: &str, ins: &[u32], shape: &[Dim]) -> u32 {
        self.op_attrs(dialect, name, ins, Vec::new(), shape)
    }

    fn op_attrs(
        &mut self,
        dialect: &str,
        name: &str,
        ins: &[u32],
        attrs: Vec<(&str, Value)>,
        shape: &[Dim],
    ) -> u32 {
        let id = self.id();
        let t = self.ty(shape);
        let mut op = Op::new(dialect, name, 1).with_inputs(ins);
        for (k, v) in attrs {
            op = op.with_attr(k, v);
        }
        self.ops.push(op.with_output(id, t));
        id
    }

    fn act(&mut self, x: u32, kind: &str, shape: &[Dim]) -> u32 {
        self.op_attrs(
            "omni.nn",
            "activation",
            &[x],
            vec![("kind", Value::text(kind.to_string()))],
            shape,
        )
    }

    fn einsum(&mut self, eq: &str, ins: &[u32], shape: &[Dim]) -> u32 {
        self.op_attrs(
            "omni.tensor",
            "einsum",
            ins,
            vec![("equation", Value::text(eq.to_string()))],
            shape,
        )
    }

    /// `x[start..stop]` on every axis at once, which is the only form
    /// `tensor.slice` has. Every bound here is concrete: a symbolic axis has no
    /// number to slice at, and the families below slice features rather than
    /// batches for exactly that reason.
    fn slice(&mut self, x: u32, start: &[u64], stop: &[u64], shape: &[Dim]) -> u32 {
        self.op_attrs(
            "omni.tensor",
            "slice",
            &[x],
            vec![
                (
                    "start",
                    Value::Array(start.iter().map(|v| Value::U(*v)).collect()),
                ),
                (
                    "stop",
                    Value::Array(stop.iter().map(|v| Value::U(*v)).collect()),
                ),
            ],
            shape,
        )
    }

    fn reshape(&mut self, x: u32, shape: &[u64]) -> u32 {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::N(*d)).collect();
        self.op_attrs(
            "omni.tensor",
            "reshape",
            &[x],
            vec![(
                "shape",
                Value::Array(shape.iter().map(|d| Value::U(*d)).collect()),
            )],
            &dims,
        )
    }

    /// `x @ Wᵀ (+ b)` — most of what these families are made of. The transpose
    /// is a node rather than an assumption: `[out, in]` is what frameworks
    /// store, and turning it into `[in, out]` silently is how a graph comes to
    /// disagree with its weights.
    fn linear(
        &mut self,
        x: u32,
        weight: &str,
        bias: Option<&String>,
        rows: &[Dim],
        in_f: u64,
        out_f: u64,
    ) -> u32 {
        let w = self.constant(weight, &[Dim::N(out_f), Dim::N(in_f)]);
        let wt = self.op_attrs(
            "omni.tensor",
            "transpose",
            &[w],
            vec![("perm", Value::Array(vec![Value::U(1), Value::U(0)]))],
            &[Dim::N(in_f), Dim::N(out_f)],
        );
        let mut out_shape = rows.to_vec();
        out_shape.push(Dim::N(out_f));
        let y = self.op("omni.tensor", "matmul", &[x, wt], &out_shape);
        match bias {
            Some(name) => {
                let b = self.constant(name, &[Dim::N(out_f)]);
                self.op("omni.tensor", "add", &[y, b], &out_shape)
            }
            None => y,
        }
    }

    fn ret(&mut self, ids: &[u32]) {
        self.ops
            .push(Op::new("omni.core", "return", 1).with_inputs(ids));
    }
}

/// An optional weight: present is a name, absent stays absent. A zero bias that
/// the checkpoint does not contain is a tensor this build would be inventing.
fn optional(available: &[String], name: String) -> Option<String> {
    available.contains(&name).then_some(name)
}

fn function(
    params: Vec<(String, Type)>,
    results: Vec<Type>,
    attrs: Vec<(&str, Value)>,
    ops: Vec<Op>,
    constraints: Vec<Constraint>,
) -> Function {
    Function {
        params,
        results,
        attrs: attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        body: Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops,
            }],
        },
        constraints,
    }
}

fn at_least_one(sym: &str) -> Constraint {
    Constraint {
        dim: sym.into(),
        rel: Rel::Ge,
        bound: 1,
    }
}

/// `transformer.moe` — §07.8's MoE row: a router, gathered expert weights, and
/// a weighted sum of what the chosen experts computed.
///
/// The expert weights are one tensor per layer (`[experts, d_model, d_ff]`)
/// rather than one tensor per expert, and that is what makes `gather` the
/// mechanism §07.8 says it is: routing produces indices, `tensor.gather` turns
/// indices into the weight matrices themselves, and the per-token application is
/// an `einsum` over a batch of them. No new op, and no expert loop — a loop over
/// experts would make the graph's size depend on how many there are.
///
/// The cost of expressing it this way honestly: `gather` materialises
/// `[tokens, top_k, d_model, d_ff]`, which is the dense reading of a sparse
/// operation. A runtime lowers this to a grouped matmul; the canonical form's
/// job is to say what is computed, and this says it exactly.
fn synthesize_moe(params: &Value, available: &[String]) -> Result<Module, String> {
    let get = |k: &str| -> Option<u64> { params.get(k).and_then(|v| v.as_u64()) };
    let hidden = get("hidden_size").ok_or("arch.params has no `hidden_size`")?;
    let ff = get("intermediate_size").ok_or("arch.params has no `intermediate_size`")?;
    let experts = get("n_experts").ok_or("arch.params has no `n_experts`")?;
    let layers = get("n_layers").unwrap_or(1).max(1);
    let top_k = get("top_k").unwrap_or(2).max(1);
    if top_k > experts {
        return Err(format!(
            "`top_k` is {top_k} and there are {experts} expert(s)"
        ));
    }
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
        .to_string();
    let mut needed = Vec::new();
    for l in 0..layers {
        needed.push(format!("moe.layers.{l}.router.weight"));
        needed.push(format!("moe.layers.{l}.experts.w_in"));
        needed.push(format!("moe.layers.{l}.experts.w_out"));
    }
    require(available, needed)?;

    let t = Dim::Sym("T".into());
    let mut e = Emit::new(1, DType::F32);
    let mut h = 0u32;
    for l in 0..layers {
        // `[d_model, experts]` is the orientation `moe_route` takes. A
        // checkpoint that stores the router the other way is transposed in the
        // graph, where a reader can see it.
        let router = e.constant(
            &format!("moe.layers.{l}.router.weight"),
            &[Dim::N(hidden), Dim::N(experts)],
        );
        let (weights, idx) = (e.id(), e.id());
        let wt_ty = e.ty(&[t.clone(), Dim::N(top_k)]);
        let idx_ty = Type::tensor(
            vec![t.clone(), Dim::N(top_k)],
            DType::Int {
                w: 32,
                signed: true,
            },
        );
        e.ops.push(
            Op::new("omni.nn", "moe_route", 1)
                .with_inputs(&[h, router])
                .with_attr("top_k", Value::U(top_k))
                .with_output(weights, wt_ty)
                .with_output(idx, idx_ty),
        );
        let w_in = e.constant(
            &format!("moe.layers.{l}.experts.w_in"),
            &[Dim::N(experts), Dim::N(hidden), Dim::N(ff)],
        );
        let w_out = e.constant(
            &format!("moe.layers.{l}.experts.w_out"),
            &[Dim::N(experts), Dim::N(ff), Dim::N(hidden)],
        );
        let g_in = e.op_attrs(
            "omni.tensor",
            "gather",
            &[w_in, idx],
            vec![("axis", Value::U(0))],
            &[t.clone(), Dim::N(top_k), Dim::N(hidden), Dim::N(ff)],
        );
        let g_out = e.op_attrs(
            "omni.tensor",
            "gather",
            &[w_out, idx],
            vec![("axis", Value::U(0))],
            &[t.clone(), Dim::N(top_k), Dim::N(ff), Dim::N(hidden)],
        );
        let up = e.einsum(
            "td,tkdf->tkf",
            &[h, g_in],
            &[t.clone(), Dim::N(top_k), Dim::N(ff)],
        );
        let act = e.act(up, &activation, &[t.clone(), Dim::N(top_k), Dim::N(ff)]);
        let down = e.einsum(
            "tkf,tkfd->tkd",
            &[act, g_out],
            &[t.clone(), Dim::N(top_k), Dim::N(hidden)],
        );
        // The routing weights are what makes this a mixture rather than a
        // choice: each chosen expert's output is scaled by its gate.
        let mixed = e.einsum("tkd,tk->td", &[down, weights], &[t.clone(), Dim::N(hidden)]);
        h = e.op(
            "omni.tensor",
            "add",
            &[h, mixed],
            &[t.clone(), Dim::N(hidden)],
        );
    }
    e.ret(&[h]);
    let dt = DType::F32;
    let f = function(
        vec![(
            "tokens".into(),
            Type::tensor(vec![t.clone(), Dim::N(hidden)], dt.clone()),
        )],
        vec![Type::tensor(vec![t, Dim::N(hidden)], dt)],
        vec![
            ("kind", Value::text("forward")),
            ("experts", Value::U(experts)),
            ("top_k", Value::U(top_k)),
        ],
        e.ops,
        vec![at_least_one("T")],
    );
    Ok(finish_module(
        "transformer.moe",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// `rnn.lstm` and `rnn.gru` — §07.8's recurrent row: `core.scan` with an
/// explicit state carry.
///
/// One graph over one sequence, not over a batch. `scan` threads its first
/// operand as the carry and slices the second along the time axis, so a batch
/// dimension would have to be sliced inside the body — and slicing needs
/// concrete bounds, which a symbolic batch does not have. A batch is a `map`
/// over this function, which is where §07.10 says that belongs: the canonical
/// graph says what is computed, and how many of them run at once is a
/// scheduling question.
///
/// The carry is one tensor because `scan` threads one. For GRU that is the
/// hidden state; for LSTM it is `[h ‖ c]`, split inside the body — visible in
/// the graph rather than hidden in an op.
fn synthesize_rnn(params: &Value, available: &[String], lstm: bool) -> Result<Module, String> {
    let get = |k: &str| -> Option<u64> { params.get(k).and_then(|v| v.as_u64()) };
    let input = get("input_size").ok_or("arch.params has no `input_size`")?;
    let hidden = get("hidden_size").ok_or("arch.params has no `hidden_size`")?;
    let layers = get("n_layers").unwrap_or(1).max(1);
    let gates = if lstm { 4 } else { 3 };
    let carry = if lstm { 2 * hidden } else { hidden };
    let mut needed = Vec::new();
    for l in 0..layers {
        needed.push(format!("rnn.layers.{l}.weight_ih"));
        needed.push(format!("rnn.layers.{l}.weight_hh"));
    }
    require(available, needed)?;

    let time = Dim::Sym("T".into());
    let dt = DType::F32;
    // 0 and 1 are the function's parameters — the sequence and the initial
    // state — so the first value this synthesizer defines is 2.
    let mut next = 2u32;
    let mut ops: Vec<Op> = Vec::new();
    let mut seq = 0u32; // the input sequence, [T, input] then [T, 1, hidden]
    let mut states: Vec<u32> = Vec::new();
    for l in 0..layers {
        let in_f = if l == 0 { input } else { hidden };
        let bias_ih = optional(available, format!("rnn.layers.{l}.bias_ih"));
        let bias_hh = optional(available, format!("rnn.layers.{l}.bias_hh"));

        // This layer's slice of the initial state: `[layers, carry]` sliced on a
        // concrete axis, which is why the state is stored that way.
        let mut outer = Emit::new(next, dt.clone());
        let h0 = outer.slice(1, &[l, 0], &[l + 1, carry], &[Dim::N(1), Dim::N(carry)]);
        next = outer.next;
        ops.append(&mut outer.ops);

        // The body: one timestep. Its arguments are the carry and the slice
        // `scan` took, which is rank-1 on the first layer (`[input]`) and
        // rank-2 afterwards (`[1, hidden]`).
        let carry_id = next;
        let step_id = next + 1;
        let mut b = Emit::new(next + 2, dt.clone());
        let x = if l == 0 {
            b.reshape(step_id, &[1, in_f])
        } else {
            step_id
        };
        let row = [Dim::N(1)];
        let gi = b.linear(
            x,
            &format!("rnn.layers.{l}.weight_ih"),
            bias_ih.as_ref(),
            &row,
            in_f,
            gates * hidden,
        );
        let h_prev = if lstm {
            b.slice(
                carry_id,
                &[0, 0],
                &[1, hidden],
                &[Dim::N(1), Dim::N(hidden)],
            )
        } else {
            carry_id
        };
        let gh = b.linear(
            h_prev,
            &format!("rnn.layers.{l}.weight_hh"),
            bias_hh.as_ref(),
            &row,
            hidden,
            gates * hidden,
        );
        let gate = |b: &mut Emit, x: u32, k: u64| -> u32 {
            b.slice(
                x,
                &[0, k * hidden],
                &[1, (k + 1) * hidden],
                &[Dim::N(1), Dim::N(hidden)],
            )
        };
        let hshape = [Dim::N(1), Dim::N(hidden)];
        let (new_carry, emit) = if lstm {
            // The gate order is PyTorch's — i, f, g, o — because that is the
            // order the weights are stored in, and a different reading of the
            // same bytes is a different model that runs without complaining.
            let g = b.op(
                "omni.tensor",
                "add",
                &[gi, gh],
                &[Dim::N(1), Dim::N(gates * hidden)],
            );
            let (gi_, gf, gg, go) = (
                gate(&mut b, g, 0),
                gate(&mut b, g, 1),
                gate(&mut b, g, 2),
                gate(&mut b, g, 3),
            );
            let i = b.op("omni.tensor", "sigmoid", &[gi_], &hshape);
            let f = b.op("omni.tensor", "sigmoid", &[gf], &hshape);
            let gt = b.op("omni.tensor", "tanh", &[gg], &hshape);
            let o = b.op("omni.tensor", "sigmoid", &[go], &hshape);
            let c_prev = b.slice(
                carry_id,
                &[0, hidden],
                &[1, 2 * hidden],
                &[Dim::N(1), Dim::N(hidden)],
            );
            let fc = b.op("omni.tensor", "mul", &[f, c_prev], &hshape);
            let ig = b.op("omni.tensor", "mul", &[i, gt], &hshape);
            let c = b.op("omni.tensor", "add", &[fc, ig], &hshape);
            let tc = b.op("omni.tensor", "tanh", &[c], &hshape);
            let h = b.op("omni.tensor", "mul", &[o, tc], &hshape);
            let cat = b.op_attrs(
                "omni.tensor",
                "concat",
                &[h, c],
                vec![("axis", Value::U(1))],
                &[Dim::N(1), Dim::N(2 * hidden)],
            );
            (cat, Some(h))
        } else {
            // GRU, in PyTorch's formulation: the reset gate multiplies the
            // *hidden* half of the candidate only, which is why `gi` and `gh`
            // are kept apart until here instead of being added like the LSTM's.
            let (ir, iz, in_) = (
                gate(&mut b, gi, 0),
                gate(&mut b, gi, 1),
                gate(&mut b, gi, 2),
            );
            let (hr, hz, hn) = (
                gate(&mut b, gh, 0),
                gate(&mut b, gh, 1),
                gate(&mut b, gh, 2),
            );
            let rs = b.op("omni.tensor", "add", &[ir, hr], &hshape);
            let r = b.op("omni.tensor", "sigmoid", &[rs], &hshape);
            let zs = b.op("omni.tensor", "add", &[iz, hz], &hshape);
            let z = b.op("omni.tensor", "sigmoid", &[zs], &hshape);
            let rhn = b.op("omni.tensor", "mul", &[r, hn], &hshape);
            let ns = b.op("omni.tensor", "add", &[in_, rhn], &hshape);
            let n = b.op("omni.tensor", "tanh", &[ns], &hshape);
            // (1 − z)·n + z·h, written as n + z·(h − n): the same value, and it
            // needs no constant one — a graph's constants name tensors, and a
            // scalar this build invented would not be one of the model's.
            let diff = b.op("omni.tensor", "sub", &[carry_id, n], &hshape);
            let scaled = b.op("omni.tensor", "mul", &[z, diff], &hshape);
            let h = b.op("omni.tensor", "add", &[n, scaled], &hshape);
            (h, None)
        };
        let mut yields = vec![new_carry];
        if let Some(h) = emit {
            yields.push(h);
        }
        b.ops
            .push(Op::new("omni.core", "yield", 1).with_inputs(&yields));
        let body_ops = std::mem::take(&mut b.ops);
        next = b.next;

        let carry_ty = Type::tensor(vec![Dim::N(1), Dim::N(carry)], dt.clone());
        let step_ty = if l == 0 {
            Type::tensor(vec![Dim::N(in_f)], dt.clone())
        } else {
            Type::tensor(vec![Dim::N(1), Dim::N(hidden)], dt.clone())
        };
        let (final_id, hs_id) = (next, next + 1);
        next += 2;
        let mut scan = Op::new("omni.core", "scan", 1)
            .with_inputs(&[h0, seq])
            .with_attr("axis", Value::U(0))
            .with_output(final_id, carry_ty)
            .with_output(
                hs_id,
                Type::tensor(vec![time.clone(), Dim::N(1), Dim::N(hidden)], dt.clone()),
            );
        scan.regions = vec![Region {
            blocks: vec![Block {
                args: vec![
                    (
                        carry_id,
                        Type::tensor(vec![Dim::N(1), Dim::N(carry)], dt.clone()),
                    ),
                    (step_id, step_ty),
                ],
                ops: body_ops,
            }],
        }];
        ops.push(scan);
        seq = hs_id;
        states.push(final_id);
    }
    // The last layer's outputs, and every layer's final state — a caller
    // continuing the sequence needs all of them, and a caller that does not can
    // ignore the second result.
    let stacked = if states.len() == 1 {
        states[0]
    } else {
        let id = next;
        next += 1;
        ops.push(
            Op::new("omni.tensor", "concat", 1)
                .with_inputs(&states)
                .with_attr("axis", Value::U(0))
                .with_output(
                    id,
                    Type::tensor(vec![Dim::N(states.len() as u64), Dim::N(carry)], dt.clone()),
                ),
        );
        id
    };
    let _ = next;
    ops.push(Op::new("omni.core", "return", 1).with_inputs(&[seq, stacked]));

    let f = function(
        vec![
            (
                "x".into(),
                Type::tensor(vec![time.clone(), Dim::N(input)], dt.clone()),
            ),
            (
                "state".into(),
                Type::tensor(vec![Dim::N(layers), Dim::N(carry)], dt.clone()),
            ),
        ],
        vec![
            Type::tensor(vec![time, Dim::N(1), Dim::N(hidden)], dt.clone()),
            Type::tensor(vec![Dim::N(layers.max(1)), Dim::N(carry)], dt),
        ],
        vec![
            ("kind", Value::text("forward")),
            ("cell", Value::text(if lstm { "lstm" } else { "gru" })),
            ("gates", Value::U(gates)),
        ],
        ops,
        vec![at_least_one("T")],
    );
    Ok(finish_module(
        if lstm { "rnn.lstm" } else { "rnn.gru" },
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// `gnn.mpnn` — §07.8's GNN row: messages gathered along an edge index and
/// aggregated per node.
///
/// The gather half is exactly what §07.8 describes. The aggregation is not:
/// `tensor.scatter` writes element for element, last write wins, so two edges
/// into the same node leave one message and drop the other. Summing them needs
/// a scatter-*add*, and §07 defines no reduction on `scatter`. Rather than
/// invent one, this synthesizer takes the incidence matrix as an input and
/// aggregates with an `einsum`, which is the same arithmetic written in an op
/// that exists — and the gap is recorded in §07.8 next to `ssm_scan`'s.
fn synthesize_gnn(params: &Value, available: &[String]) -> Result<Module, String> {
    let get = |k: &str| -> Option<u64> { params.get(k).and_then(|v| v.as_u64()) };
    let in_f = get("input_size").ok_or("arch.params has no `input_size`")?;
    let hidden = get("hidden_size").ok_or("arch.params has no `hidden_size`")?;
    let layers = get("n_layers").unwrap_or(1).max(1);
    let classes = get("num_classes").ok_or("arch.params has no `num_classes`")?;
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("relu")
        .to_string();
    let mut needed = Vec::new();
    for l in 0..layers {
        needed.push(format!("gnn.layers.{l}.message.weight"));
        needed.push(format!("gnn.layers.{l}.self.weight"));
    }
    needed.push("gnn.head.weight".into());
    require(available, needed)?;

    let (n, edges) = (Dim::Sym("N".into()), Dim::Sym("E".into()));
    let dt = DType::F32;
    let mut e = Emit::new(3, dt.clone());
    let mut h = 0u32; // node features
    let mut width = in_f;
    for l in 0..layers {
        // `src` selects the source node of every edge; the result is one
        // message per edge, which is the shape the whole family is about.
        let msg = e.op_attrs(
            "omni.tensor",
            "gather",
            &[h, 1],
            vec![("axis", Value::U(0))],
            &[edges.clone(), Dim::N(width)],
        );
        let projected = e.linear(
            msg,
            &format!("gnn.layers.{l}.message.weight"),
            optional(available, format!("gnn.layers.{l}.message.bias")).as_ref(),
            std::slice::from_ref(&edges),
            width,
            hidden,
        );
        // Aggregation: incidence is `[E, N]`, one row per edge with a one in
        // the destination's column, so this sums the messages arriving at each
        // node. See the note above about why it is not a scatter.
        let agg = e.einsum("en,eh->nh", &[2, projected], &[n.clone(), Dim::N(hidden)]);
        let own = e.linear(
            h,
            &format!("gnn.layers.{l}.self.weight"),
            optional(available, format!("gnn.layers.{l}.self.bias")).as_ref(),
            std::slice::from_ref(&n),
            width,
            hidden,
        );
        let sum = e.op(
            "omni.tensor",
            "add",
            &[agg, own],
            &[n.clone(), Dim::N(hidden)],
        );
        h = e.act(sum, &activation, &[n.clone(), Dim::N(hidden)]);
        width = hidden;
    }
    let logits = e.linear(
        h,
        "gnn.head.weight",
        optional(available, "gnn.head.bias".into()).as_ref(),
        std::slice::from_ref(&n),
        width,
        classes,
    );
    e.ret(&[logits]);

    let i32t = DType::Int {
        w: 32,
        signed: true,
    };
    let f = function(
        vec![
            (
                "x".into(),
                Type::tensor(vec![n.clone(), Dim::N(in_f)], dt.clone()),
            ),
            ("src".into(), Type::tensor(vec![edges.clone()], i32t)),
            (
                "incidence".into(),
                Type::tensor(vec![edges, n.clone()], dt.clone()),
            ),
        ],
        vec![Type::tensor(vec![n, Dim::N(classes)], dt)],
        vec![
            ("kind", Value::text("forward")),
            ("aggregation", Value::text("sum")),
        ],
        e.ops,
        vec![at_least_one("N"), at_least_one("E")],
    );
    Ok(finish_module(
        "gnn.mpnn",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// `rl.actor_critic` — §07.8's RL row: one trunk, two heads, two results.
///
/// It is in the list because of the *two results*. A policy and a value are one
/// model with one set of shared weights, and a format that can only describe a
/// single output tensor forces them apart into two artifacts that then have to
/// be kept in step by convention.
fn synthesize_rl(params: &Value, available: &[String]) -> Result<Module, String> {
    let sizes: Vec<u64> = match params.get("hidden_sizes") {
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or("a non-integer in `hidden_sizes`".to_string())
            })
            .collect::<Result<Vec<u64>, String>>()?,
        _ => return Err("arch.params has no `hidden_sizes`".into()),
    };
    if sizes.len() < 2 {
        return Err("`hidden_sizes` needs an observation size and at least one layer".into());
    }
    let actions = params
        .get("n_actions")
        .and_then(|v| v.as_u64())
        .ok_or("arch.params has no `n_actions`")?;
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("tanh")
        .to_string();
    let n = sizes.len() - 1;
    let mut needed: Vec<String> = (0..n).map(|i| format!("rl.trunk.{i}.weight")).collect();
    needed.push("rl.policy.weight".into());
    needed.push("rl.value.weight".into());
    require(available, needed)?;

    let b = Dim::Sym("B".into());
    let dt = DType::F32;
    let mut e = Emit::new(1, dt.clone());
    let mut h = 0u32;
    for i in 0..n {
        let y = e.linear(
            h,
            &format!("rl.trunk.{i}.weight"),
            optional(available, format!("rl.trunk.{i}.bias")).as_ref(),
            std::slice::from_ref(&b),
            sizes[i],
            sizes[i + 1],
        );
        h = e.act(y, &activation, &[b.clone(), Dim::N(sizes[i + 1])]);
    }
    let features = sizes[n];
    let policy = e.linear(
        h,
        "rl.policy.weight",
        optional(available, "rl.policy.bias".into()).as_ref(),
        std::slice::from_ref(&b),
        features,
        actions,
    );
    // Logits, not probabilities: a softmax here would decide the sampling
    // temperature on the model's behalf.
    let value = e.linear(
        h,
        "rl.value.weight",
        optional(available, "rl.value.bias".into()).as_ref(),
        std::slice::from_ref(&b),
        features,
        1,
    );
    e.ret(&[policy, value]);

    let f = function(
        vec![(
            "obs".into(),
            Type::tensor(vec![b.clone(), Dim::N(sizes[0])], dt.clone()),
        )],
        vec![
            Type::tensor(vec![b.clone(), Dim::N(actions)], dt.clone()),
            Type::tensor(vec![b, Dim::N(1)], dt),
        ],
        vec![
            ("kind", Value::text("forward")),
            ("heads", Value::text("policy,value")),
        ],
        e.ops,
        vec![at_least_one("B")],
    );
    Ok(finish_module(
        "rl.actor_critic",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// `audio.encoder` — §07.8's speech row: a stack of causal 1-D convolutions.
///
/// Causality is the whole point of the row and the reason `conv1d_causal` is
/// its own op rather than `conv` with a padding attribute: a streaming encoder
/// that pads symmetrically sees one frame of the future, produces slightly
/// better numbers offline, and cannot be run live. The op with the padding
/// baked in is the one that cannot be got wrong.
fn synthesize_audio(params: &Value, available: &[String]) -> Result<Module, String> {
    let channels: Vec<u64> = match params.get("channels") {
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|v| v.as_u64().ok_or("a non-integer in `channels`".to_string()))
            .collect::<Result<Vec<u64>, String>>()?,
        _ => return Err("arch.params has no `channels`".into()),
    };
    if channels.len() < 2 {
        return Err("`channels` needs an input channel count and at least one block".into());
    }
    let kernel = params
        .get("kernel")
        .and_then(|v| v.as_u64())
        .ok_or("arch.params has no `kernel`")?;
    if kernel == 0 {
        return Err("`kernel` is 0".into());
    }
    let activation = params
        .get("activation")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu")
        .to_string();
    let n = channels.len() - 1;
    require(
        available,
        (0..n)
            .map(|i| format!("audio.blocks.{i}.conv.weight"))
            .collect(),
    )?;

    let (b, len) = (Dim::Sym("B".into()), Dim::Sym("L".into()));
    let dt = DType::F32;
    let mut e = Emit::new(1, dt.clone());
    let mut x = 0u32;
    for i in 0..n {
        let (cin, cout) = (channels[i], channels[i + 1]);
        let w = e.constant(
            &format!("audio.blocks.{i}.conv.weight"),
            &[Dim::N(cout), Dim::N(cin), Dim::N(kernel)],
        );
        let mut ins = vec![x, w];
        if let Some(bias) = optional(available, format!("audio.blocks.{i}.conv.bias")) {
            let bid = e.constant(&bias, &[Dim::N(cout)]);
            ins.push(bid);
        }
        let out = [b.clone(), Dim::N(cout), len.clone()];
        let y = e.op("omni.nn", "conv1d_causal", &ins, &out);
        // The last block's output is the encoding, and an activation on it
        // would be a choice about what the encoding means.
        x = if i + 1 < n {
            e.act(y, &activation, &out)
        } else {
            y
        };
    }
    e.ret(&[x]);

    let f = function(
        vec![(
            "audio".into(),
            Type::tensor(
                vec![b.clone(), Dim::N(channels[0]), len.clone()],
                dt.clone(),
            ),
        )],
        vec![Type::tensor(vec![b, Dim::N(channels[n]), len], dt)],
        vec![
            ("kind", Value::text("forward")),
            ("causal", Value::Bool(true)),
        ],
        e.ops,
        vec![at_least_one("B"), at_least_one("L")],
    );
    Ok(finish_module(
        "audio.encoder",
        f,
        &["omni.core", "omni.tensor", "omni.nn"],
    ))
}

/// The module wrapper every synthesizer ends with: one `forward` function, the
/// dialects it used, and the family it came from recorded so a reader can tell a
/// synthesized graph from an authored one (§07.5).
fn finish_module(family: &str, f: Function, dialects: &[&str]) -> Module {
    let mut m = Module::new(Level::Semantic, "forward");
    m.dialects = dialects
        .iter()
        .map(|ns| DialectUse {
            ns: (*ns).to_string(),
            version: 1,
            reference: None,
        })
        .collect();
    m.attrs = vec![
        ("family".into(), Value::text(family)),
        ("synthesized".into(), Value::Bool(true)),
    ];
    m.functions = vec![("forward".into(), f)];
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(shape: &[Dim]) -> Type {
        Type::tensor(shape.to_vec(), DType::BF16)
    }

    fn sym(s: &str) -> Dim {
        Dim::Sym(s.into())
    }

    /// A two-op module: one attention over three inputs, then a return.
    fn attention_module(version: u32, scale: bool) -> Module {
        let q = bf16(&[sym("B"), Dim::N(4), sym("S"), Dim::N(16)]);
        let mut attn = Op::new("omni.nn", "attention", version)
            .with_inputs(&[0, 1, 2])
            .with_output(3, q.clone());
        if scale {
            attn = attn.with_attr("scale", Value::F64(0.25));
        }
        let f = Function {
            params: vec![
                ("q".into(), q.clone()),
                ("k".into(), q.clone()),
                ("v".into(), q.clone()),
            ],
            results: vec![q],
            attrs: Vec::new(),
            body: Region {
                blocks: vec![Block {
                    args: Vec::new(),
                    ops: vec![attn, Op::new("omni.core", "return", 1).with_inputs(&[3])],
                }],
            },
            constraints: Vec::new(),
        };
        let mut m = Module::new(Level::Semantic, "forward");
        m.dialects = vec![
            DialectUse {
                ns: "omni.core".into(),
                version: 1,
                reference: None,
            },
            DialectUse {
                ns: "omni.nn".into(),
                version: 1,
                reference: None,
            },
        ];
        m.functions = vec![("forward".into(), f)];
        m
    }

    #[test]
    fn a_module_round_trips_through_cbor() {
        let mut m = attention_module(2, true);
        // Every type kind, so the encoding is exercised rather than the two
        // kinds a transformer happens to use.
        let f = &mut m.functions[0].1;
        f.params.push((
            "state".into(),
            Type::State {
                id: "kv_cache".into(),
                spec: Some(Box::new(bf16(&[Dim::N(2)]))),
            },
        ));
        f.params.push((
            "stream".into(),
            Type::Stream(Box::new(bf16(&[Dim::Dynamic]))),
        ));
        f.params.push(("tok".into(), Type::Token));
        f.params.push((
            "tup".into(),
            Type::Tuple(vec![Type::Token, bf16(&[Dim::N(1)])]),
        ));
        f.params.push((
            "lst".into(),
            Type::List(Box::new(Type::Opaque("org.acme/handle".into()))),
        ));
        f.constraints.push(Constraint {
            dim: "S".into(),
            rel: Rel::Le,
            bound: 4096,
        });
        let bytes = m.to_value().encode();
        let back = Module::from_value(&crate::cbor::decode(&bytes).unwrap()).unwrap();
        assert_eq!(back, m);
        // And the encoding is canonical, which every structure object must be.
        assert_eq!(back.to_value().encode(), bytes);
    }

    #[test]
    fn a_well_formed_module_verifies() {
        let m = attention_module(2, true);
        let r = verify(&m, &Context::default());
        assert!(r.is_valid(), "{:?}", r.findings);
        assert_eq!(r.ops, 2);
        assert!(r.checked >= 1);
    }

    #[test]
    fn ssa_violations_are_invalid() {
        // R-I01: one value, two definitions.
        let mut m = attention_module(2, true);
        let ops = &mut m.functions[0].1.body.blocks[0].ops;
        ops.insert(
            1,
            Op::new("omni.tensor", "add", 1)
                .with_inputs(&[3, 3])
                .with_output(3, bf16(&[sym("B"), Dim::N(4), sym("S"), Dim::N(16)])),
        );
        m.dialects.push(DialectUse {
            ns: "omni.tensor".into(),
            version: 1,
            reference: None,
        });
        let r = verify(&m, &Context::default());
        assert!(r
            .findings
            .iter()
            .any(|f| f.rule() == "R-I01" && f.is_invalid()));

        // R-I02: a use with no definition anywhere.
        let mut m = attention_module(2, true);
        m.functions[0].1.body.blocks[0].ops[0].inputs[2] = 99;
        let r = verify(&m, &Context::default());
        assert!(r
            .findings
            .iter()
            .any(|f| f.rule() == "R-I02" && f.is_invalid()));

        // R-I03: an entry that does not exist.
        let mut m = attention_module(2, true);
        m.entry = "nope".into();
        assert!(verify(&m, &Context::default())
            .findings
            .iter()
            .any(|f| f.rule() == "R-I03"));
    }

    #[test]
    fn a_declared_type_that_disagrees_with_inference_is_invalid() {
        let mut m = attention_module(2, true);
        // Attention's result takes q's positions and v's channels; claiming
        // something else is exactly the error a graph verifier exists to catch.
        m.functions[0].1.body.blocks[0].ops[0].outputs[0].1 = bf16(&[Dim::N(7)]);
        let r = verify(&m, &Context::default());
        assert!(
            r.findings
                .iter()
                .any(|f| f.rule() == "R-I06" && f.is_invalid()),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn arity_attribute_and_region_contracts_are_checked() {
        // Too few operands.
        let mut m = attention_module(2, true);
        m.functions[0].1.body.blocks[0].ops[0].inputs.truncate(2);
        assert!(verify(&m, &Context::default())
            .findings
            .iter()
            .any(|f| f.rule() == "R-I07"));

        // A missing required attribute.
        let mut m = attention_module(2, true);
        let norm = Op::new("omni.nn", "norm", 1)
            .with_inputs(&[0])
            .with_output(4, bf16(&[sym("B"), Dim::N(4), sym("S"), Dim::N(16)]));
        m.functions[0].1.body.blocks[0].ops.insert(1, norm);
        let r = verify(&m, &Context::default());
        assert!(r
            .findings
            .iter()
            .any(|f| f.rule() == "R-I07" && f.message().contains("`kind`")));

        // A region that does not terminate.
        let mut m = attention_module(2, true);
        let region = Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops: vec![Op::new("omni.core", "debug", 1)],
            }],
        };
        let mut whileop = Op::new("omni.core", "while", 1)
            .with_inputs(&[0])
            .with_output(9, Type::Token);
        whileop.regions = vec![region.clone(), region];
        m.functions[0].1.body.blocks[0].ops.insert(1, whileop);
        let r = verify(&m, &Context::default());
        assert!(r
            .findings
            .iter()
            .any(|f| f.rule() == "R-I07" && f.message().contains("yield")));
    }

    #[test]
    fn an_effect_token_orders_exactly_one_successor() {
        let mut m = attention_module(2, true);
        m.dialects.push(DialectUse {
            ns: "omni.io".into(),
            version: 1,
            reference: None,
        });
        let ops = &mut m.functions[0].1.body.blocks[0].ops;
        ops.insert(
            0,
            Op::new("omni.io", "external", 1)
                .with_attr("id", Value::text("org.acme/retrieve"))
                .with_output(10, Type::Token),
        );
        // Used twice: two orderings, which is none (R-I08).
        ops.insert(1, Op::new("omni.core", "assert", 1).with_inputs(&[10, 10]));
        let r = verify(&m, &Context::default());
        assert!(
            r.findings.iter().any(|f| f.rule() == "R-I08"),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn contradictory_dimension_constraints_are_invalid() {
        let mut m = attention_module(2, true);
        m.functions[0].1.constraints = vec![
            Constraint {
                dim: "S".into(),
                rel: Rel::Ge,
                bound: 4096,
            },
            Constraint {
                dim: "S".into(),
                rel: Rel::Le,
                bound: 512,
            },
        ];
        assert!(verify(&m, &Context::default())
            .findings
            .iter()
            .any(|f| f.rule() == "R-I09"));
    }

    #[test]
    fn an_unknown_dialect_is_indeterminate_and_a_shipped_lowering_recovers_it() {
        // §11.3: a container carrying something a reader does not understand is
        // not thereby invalid. Only *execution* needs the semantics.
        let mut m = attention_module(2, true);
        m.dialects.push(DialectUse {
            ns: "org.hyperion/nn".into(),
            version: 1,
            reference: None,
        });
        let ops = &mut m.functions[0].1.body.blocks[0].ops;
        ops.insert(
            0,
            Op::new("org.hyperion/nn", "hyperion.block", 1)
                .with_inputs(&[0])
                .with_output(20, bf16(&[sym("B"), Dim::N(4), sym("S"), Dim::N(16)])),
        );
        let r = verify(&m, &Context::default());
        assert!(!r.is_invalid(), "{:?}", r.findings);
        assert!(r.is_indeterminate());
        assert_eq!(r.unknown, 1);
        assert!(r.findings.iter().all(|f| !f.is_invalid()));

        // With a lowering shipped for it, the same graph is not even
        // indeterminate on that op: a runtime can apply the rule and proceed.
        let rule = Rewrite {
            name: "hyperion→primitive".into(),
            dialect: "org.hyperion/nn".into(),
            op: "hyperion.block".into(),
            version: 1,
            binds: vec!["x".into()],
            conditions: Vec::new(),
            emit: vec![emit("omni.tensor", "tanh", &["x"], &["out"])],
            results: vec!["out".into()],
            soundness: Soundness::SemanticsPreserving,
            to_level: Some(Level::Primitive),
        };
        let cx = Context {
            rewrites: std::slice::from_ref(&rule),
            ..Default::default()
        };
        let r = verify(&m, &cx);
        assert_eq!(r.recoverable, 1);
        assert_eq!(r.unknown, 0);
        assert!(r.is_valid(), "{:?}", r.findings);

        // And applying it really does produce a graph in a dialect this build
        // understands, which then verifies on its own terms.
        let (lowered, applied) = apply_rewrites(&m, std::slice::from_ref(&rule), false);
        assert_eq!(applied.applied, vec![("hyperion→primitive".to_string(), 1)]);
        assert!(lowered.declares("org.hyperion/nn").is_none());
        let r = verify(&lowered, &Context::default());
        assert!(r.is_valid(), "{:?}", r.findings);
        assert_eq!(r.unknown, 0);
    }

    #[test]
    fn an_op_version_migrates_by_shipped_rewrite() {
        // §07.7's own example: v1 attention consumed by a v2-only runtime.
        let m = attention_module(1, true);
        let (migrated, applied) = apply_rewrites(&m, &shipped_migrations(), false);
        assert_eq!(applied.applied, vec![("attention-v1-to-v2".to_string(), 1)]);
        let op = &migrated.functions[0].1.body.blocks[0].ops[0];
        assert_eq!(op.version, 2);
        assert_eq!(op.attr("kv_groups"), Some(&Value::U(1)));
        // The migration preserved the value id, so the return still reads it.
        assert_eq!(op.outputs[0].0, 3);
        assert!(verify(&migrated, &Context::default()).is_valid());
    }

    #[test]
    fn attention_lowers_to_primitives_that_verify() {
        let m = attention_module(2, true);
        let (low, applied) = apply_rewrites(&m, &shipped_lowerings(), false);
        assert_eq!(
            applied.applied,
            vec![("omni.nn/attention@2→primitive".to_string(), 1)]
        );
        assert!(applied.refused.is_empty(), "{:?}", applied.refused);
        // The point of §07.2: a runtime that knows only omni.tensor can run it.
        assert_eq!(low.level, Level::Primitive);
        assert!(low.declares("omni.nn").is_none());
        assert!(low.declares("omni.tensor").is_some());
        let ops: Vec<String> = low.ops().iter().map(|(_, o)| o.qualified()).collect();
        assert!(ops.iter().any(|o| o == "omni.tensor/softmax@1"), "{ops:?}");
        assert_eq!(
            ops.iter().filter(|o| *o == "omni.tensor/matmul@1").count(),
            2
        );
        let r = verify(&low, &Context::default());
        assert!(r.is_valid(), "{:?}", r.findings);
        // The lowered graph computes the same *type*, which is the strongest
        // statement a type checker can make about it.
        assert_eq!(low.functions[0].1.results, m.functions[0].1.results);
    }

    #[test]
    fn a_lowering_that_cannot_apply_says_so_instead_of_guessing() {
        // Causal attention has no lowering here: a mask cannot be built from
        // the ops available, and emitting the non-causal form would compute the
        // wrong thing silently.
        let mut m = attention_module(2, true);
        m.functions[0].1.body.blocks[0].ops[0] = m.functions[0].1.body.blocks[0].ops[0]
            .clone()
            .with_attr("causal", Value::Bool(true));
        let (low, applied) = apply_rewrites(&m, &shipped_lowerings(), false);
        assert!(applied.applied.is_empty());
        assert_eq!(low.level, Level::Semantic);
        assert!(low.declares("omni.nn").is_some());

        // And an attention op with no declared scale is not lowered either: the
        // rule needs the number, and inventing 1/√d for a graph that did not
        // say so is a numeric change dressed as a rewrite.
        let m = attention_module(2, false);
        let (_, applied) = apply_rewrites(&m, &shipped_lowerings(), false);
        assert!(applied.applied.is_empty());
    }

    #[test]
    fn an_approximate_rewrite_is_refused_unless_allowed() {
        let mut rule = shipped_lowerings().remove(0);
        rule.soundness = Soundness::NumericApproximate;
        let m = attention_module(2, true);
        let (kept, applied) = apply_rewrites(&m, std::slice::from_ref(&rule), false);
        assert!(applied.applied.is_empty());
        assert_eq!(applied.refused.len(), 1);
        assert_eq!(kept.level, Level::Semantic);
        let (low, applied) = apply_rewrites(&m, std::slice::from_ref(&rule), true);
        assert_eq!(applied.applied.len(), 1);
        assert!(applied.approximate);
        assert_eq!(low.level, Level::Primitive);
    }

    #[test]
    fn rewrites_round_trip_through_cbor() {
        for r in shipped_lowerings().into_iter().chain(shipped_migrations()) {
            let bytes = r.to_value().encode();
            let back = Rewrite::from_value(&crate::cbor::decode(&bytes).unwrap()).unwrap();
            // Attribute *order* is not part of a rewrite's meaning — canonical
            // CBOR sorts map keys (D3) — so the comparison is on the encoding.
            assert_eq!(back.to_value().encode(), bytes, "{}", r.name);
            assert_eq!(back.emit.len(), r.emit.len());
            assert_eq!(back.conditions, r.conditions);
        }
    }

    #[test]
    fn norm_activation_and_embedding_lower_too() {
        let x = bf16(&[sym("B"), sym("S"), Dim::N(64)]);
        let w = Type::tensor(vec![Dim::N(64)], DType::BF16);
        let ids = Type::tensor(
            vec![sym("B"), sym("S")],
            DType::Int {
                w: 32,
                signed: true,
            },
        );
        let table = bf16(&[Dim::N(256), Dim::N(64)]);
        let f = Function {
            params: vec![
                ("x".into(), x.clone()),
                ("w".into(), w),
                ("ids".into(), ids),
                ("table".into(), table),
            ],
            results: vec![x.clone()],
            attrs: Vec::new(),
            body: Region {
                blocks: vec![Block {
                    args: Vec::new(),
                    ops: vec![
                        Op::new("omni.nn", "norm", 1)
                            .with_inputs(&[0, 1])
                            .with_attr("kind", Value::text("rms"))
                            .with_attr("eps", Value::F64(1e-5))
                            .with_output(4, x.clone()),
                        Op::new("omni.nn", "activation", 1)
                            .with_inputs(&[4])
                            .with_attr("kind", Value::text("silu"))
                            .with_output(5, x.clone()),
                        Op::new("omni.nn", "embedding", 1)
                            .with_inputs(&[2, 3])
                            .with_output(6, x.clone()),
                        Op::new("omni.tensor", "add", 1)
                            .with_inputs(&[5, 6])
                            .with_output(7, x.clone()),
                        Op::new("omni.core", "return", 1).with_inputs(&[7]),
                    ],
                }],
            },
            constraints: Vec::new(),
        };
        let mut m = Module::new(Level::Semantic, "f");
        m.dialects = vec![
            DialectUse {
                ns: "omni.core".into(),
                version: 1,
                reference: None,
            },
            DialectUse {
                ns: "omni.tensor".into(),
                version: 1,
                reference: None,
            },
            DialectUse {
                ns: "omni.nn".into(),
                version: 1,
                reference: None,
            },
        ];
        m.functions = vec![("f".into(), f)];
        assert!(verify(&m, &Context::default()).is_valid());

        let (low, applied) = apply_rewrites(&m, &shipped_lowerings(), false);
        assert_eq!(applied.applied.len(), 3, "{:?}", applied.applied);
        assert!(low.declares("omni.nn").is_none());
        let r = verify(&low, &Context::default());
        assert!(r.is_valid(), "{:?}", r.findings);
        // rms norm becomes seven primitive ops; silu two; embedding one gather.
        assert!(low.op_count() > m.op_count());
    }

    #[test]
    fn a_constant_is_checked_against_the_tensor_it_names() {
        let t = bf16(&[Dim::N(256), Dim::N(64)]);
        let f = Function {
            params: Vec::new(),
            results: vec![t.clone()],
            attrs: Vec::new(),
            body: Region {
                blocks: vec![Block {
                    args: Vec::new(),
                    ops: vec![
                        Op::new("omni.core", "constant", 1)
                            .with_attr("tensor", Value::text("model.embed_tokens.weight"))
                            .with_output(0, t),
                        Op::new("omni.core", "return", 1).with_inputs(&[0]),
                    ],
                }],
            },
            constraints: Vec::new(),
        };
        let mut m = Module::new(Level::Semantic, "f");
        m.dialects = vec![DialectUse {
            ns: "omni.core".into(),
            version: 1,
            reference: None,
        }];
        m.functions = vec![("f".into(), f)];

        let good = |name: &str| -> Option<(Vec<u64>, DType)> {
            (name == "model.embed_tokens.weight").then(|| (vec![256, 64], DType::BF16))
        };
        let cx = Context {
            tensor: Some(&good),
            ..Default::default()
        };
        assert!(verify(&m, &cx).is_valid());

        // A tensor of a different shape, a different dtype, or none at all: all
        // R-I10, all invalid, because the graph and the weights must agree.
        for wrong in [(vec![128u64, 64], DType::BF16), (vec![256, 64], DType::F32)] {
            let lookup = move |name: &str| -> Option<(Vec<u64>, DType)> {
                (name == "model.embed_tokens.weight").then(|| wrong.clone())
            };
            let cx = Context {
                tensor: Some(&lookup),
                ..Default::default()
            };
            let r = verify(&m, &cx);
            assert!(
                r.findings
                    .iter()
                    .any(|f| f.rule() == "R-I10" && f.is_invalid()),
                "{:?}",
                r.findings
            );
        }
        let absent = |_: &str| -> Option<(Vec<u64>, DType)> { None };
        let cx = Context {
            tensor: Some(&absent),
            ..Default::default()
        };
        assert!(verify(&m, &cx).findings.iter().any(|f| f.rule() == "R-I10"));
    }

    #[test]
    fn the_binary_op_array_round_trips() {
        let m = attention_module(2, true);
        let f = &m.functions[0].1;
        let blob = binary::encode(f);
        assert_eq!(&blob[0..4], binary::MAGIC);
        assert_eq!(binary::decode(&blob).unwrap(), *f);

        // Nested regions, which §07.9's own record layout cannot express.
        let mut nested = f.clone();
        let inner = Region {
            blocks: vec![Block {
                args: vec![(50, Type::Token)],
                ops: vec![Op::new("omni.core", "yield", 1).with_inputs(&[50])],
            }],
        };
        let mut w = Op::new("omni.core", "while", 1)
            .with_inputs(&[0])
            .with_output(60, Type::Token);
        w.regions = vec![inner.clone(), inner];
        nested.body.blocks[0].ops.insert(0, w);
        let blob = binary::encode(&nested);
        assert_eq!(binary::decode(&blob).unwrap(), nested);

        // And at scale: a graph of ten thousand ops parses in one pass, with the
        // op array a flat 28 bytes per op as §07.9 requires.
        let mut big = f.clone();
        big.body.blocks[0].ops.clear();
        let t = bf16(&[sym("B"), Dim::N(4), sym("S"), Dim::N(16)]);
        let mut prev = 0u32;
        for i in 0..10_000u32 {
            big.body.blocks[0].ops.push(
                Op::new("omni.tensor", "tanh", 1)
                    .with_inputs(&[prev])
                    .with_output(i + 10, t.clone()),
            );
            prev = i + 10;
        }
        big.body.blocks[0]
            .ops
            .push(Op::new("omni.core", "return", 1).with_inputs(&[prev]));
        let blob = binary::encode(&big);
        assert_eq!(binary::decode(&blob).unwrap(), big);
        // One type, one dialect, two op names: the tables deduplicate, so a
        // large graph costs its records — 28 bytes of op, plus its operand,
        // result and location slots — and not its strings.
        assert!(
            blob.len() < 10_001 * (binary::OP_RECORD + 20),
            "{} bytes for {} ops",
            blob.len(),
            big.body.blocks[0].ops.len()
        );
    }

    #[test]
    fn a_truncated_binary_graph_is_refused() {
        let m = attention_module(2, true);
        let blob = binary::encode(&m.functions[0].1);
        for cut in [0, 4, 32, blob.len() / 2, blob.len() - 1] {
            assert!(binary::decode(&blob[..cut]).is_err(), "cut at {cut}");
        }
        let mut bad = blob.clone();
        bad[0] = b'X';
        assert!(binary::decode(&bad).is_err());
    }

    #[test]
    fn every_family_in_the_list_synthesizes_and_verifies() {
        // FAMILIES is what `omni graph synthesize` offers and what the roadmap
        // counts, so a name in it that does not build a graph — or builds one
        // with a finding — is a claim this build cannot support. The
        // architectures are tiny; what is checked is that each name produces a
        // module the verifier accepts against the weights it asked for.
        type Case = (&'static str, Value, Vec<(&'static str, Vec<u64>)>);
        let shapes: Vec<Case> = vec![
            (
                "transformer.moe",
                Value::map(vec![
                    ("hidden_size", Value::U(4)),
                    ("intermediate_size", Value::U(6)),
                    ("n_experts", Value::U(3)),
                    ("top_k", Value::U(2)),
                ]),
                vec![
                    ("moe.layers.0.router.weight", vec![4, 3]),
                    ("moe.layers.0.experts.w_in", vec![3, 4, 6]),
                    ("moe.layers.0.experts.w_out", vec![3, 6, 4]),
                ],
            ),
            (
                "rnn.lstm",
                Value::map(vec![
                    ("input_size", Value::U(3)),
                    ("hidden_size", Value::U(4)),
                ]),
                vec![
                    ("rnn.layers.0.weight_ih", vec![16, 3]),
                    ("rnn.layers.0.weight_hh", vec![16, 4]),
                ],
            ),
            (
                "rnn.gru",
                Value::map(vec![
                    ("input_size", Value::U(3)),
                    ("hidden_size", Value::U(4)),
                ]),
                vec![
                    ("rnn.layers.0.weight_ih", vec![12, 3]),
                    ("rnn.layers.0.weight_hh", vec![12, 4]),
                ],
            ),
            (
                "gnn.mpnn",
                Value::map(vec![
                    ("input_size", Value::U(2)),
                    ("hidden_size", Value::U(3)),
                    ("num_classes", Value::U(2)),
                ]),
                vec![
                    ("gnn.layers.0.message.weight", vec![3, 2]),
                    ("gnn.layers.0.self.weight", vec![3, 2]),
                    ("gnn.head.weight", vec![2, 3]),
                ],
            ),
            (
                "rl.actor_critic",
                Value::map(vec![
                    ("hidden_sizes", Value::Array(vec![Value::U(4), Value::U(5)])),
                    ("n_actions", Value::U(3)),
                ]),
                vec![
                    ("rl.trunk.0.weight", vec![5, 4]),
                    ("rl.policy.weight", vec![3, 5]),
                    ("rl.value.weight", vec![1, 5]),
                ],
            ),
            (
                "audio.encoder",
                Value::map(vec![
                    ("channels", Value::Array(vec![Value::U(2), Value::U(3)])),
                    ("kernel", Value::U(3)),
                ]),
                vec![("audio.blocks.0.conv.weight", vec![3, 2, 3])],
            ),
        ];
        for (family, params, weights) in &shapes {
            let names: Vec<String> = weights.iter().map(|(n, _)| n.to_string()).collect();
            let m = synthesize(family, params, &names).unwrap_or_else(|e| panic!("{family}: {e}"));
            assert_eq!(
                m.attrs
                    .iter()
                    .find(|(k, _)| k == "family")
                    .and_then(|(_, v)| v.as_str()),
                Some(*family)
            );
            let lookup = |name: &str| -> Option<(Vec<u64>, DType)> {
                weights
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, s)| (s.clone(), DType::F32))
            };
            let cx = Context {
                tensor: Some(&lookup),
                ..Default::default()
            };
            let r = verify(&m, &cx);
            assert!(r.is_valid(), "{family}: {:?}", r.findings);
            assert_eq!(r.unknown, 0, "{family} used an op no dialect declares");
            // A weight the model does not have is named rather than emitted.
            let short: Vec<String> = names[..names.len() - 1].to_vec();
            assert!(
                synthesize(family, params, &short).is_err(),
                "{family} synthesized a graph over a weight that is not there"
            );
        }
        // And the list itself is the thing the CLI offers, so nothing in it is
        // unreachable.
        for family in FAMILIES {
            assert!(
                synthesize(family, &Value::map(vec![]), &[]).is_err(),
                "{family} synthesized something out of no parameters at all"
            );
        }
    }

    #[test]
    fn synthesis_builds_a_graph_that_verifies_against_its_weights() {
        let params = Value::map(vec![
            ("hidden_size", Value::U(64)),
            ("n_layers", Value::U(2)),
            ("n_heads", Value::U(4)),
            ("n_kv_heads", Value::U(2)),
            ("activation", Value::text("silu")),
            (
                "rope",
                Value::map(vec![
                    ("kind", Value::text("rope")),
                    ("theta", Value::F64(10000.0)),
                    ("interleaved", Value::Bool(false)),
                ]),
            ),
        ]);
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for l in 0..2 {
            names.push(format!("model.layers.{l}.norm.weight"));
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                names.push(format!("model.layers.{l}.attn.{p}.weight"));
            }
        }
        let m = synthesize("transformer.decoder", &params, &names).unwrap();
        assert_eq!(m.level, Level::Semantic);

        // The weights the graph names are the weights the model has, checked by
        // the same rule a reader would apply.
        let lookup = |name: &str| -> Option<(Vec<u64>, DType)> {
            if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                return Some((vec![256, 64], DType::BF16));
            }
            if name.ends_with("norm.weight") {
                return Some((vec![64], DType::F32));
            }
            if name.ends_with("q_proj.weight") || name.ends_with("o_proj.weight") {
                return Some((vec![64, 64], DType::BF16));
            }
            if name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight") {
                return Some((vec![32, 64], DType::BF16));
            }
            None
        };
        let cx = Context {
            tensor: Some(&lookup),
            ..Default::default()
        };
        let r = verify(&m, &cx);
        assert!(r.is_valid(), "{:?}", r.findings);
        assert!(r.ops > 30, "{} ops", r.ops);
        assert!(r.unknown == 0);

        // A model missing a weight gets told which one, not a broken graph.
        let short = &names[..names.len() - 1];
        let e = synthesize("transformer.decoder", &params, short).unwrap_err();
        assert!(e.contains("o_proj"), "{e}");
        // And an unregistered family is refused rather than guessed at.
        assert!(synthesize("mamba", &params, &names).is_err());
        // As is an arithmetically impossible head split.
        let bad = Value::map(vec![
            ("hidden_size", Value::U(65)),
            ("n_layers", Value::U(1)),
            ("n_heads", Value::U(4)),
        ]);
        assert!(synthesize("transformer.decoder", &bad, &names).is_err());
    }

    #[test]
    fn an_encoder_is_the_same_graph_without_the_mask_and_without_the_head() {
        let params = Value::map(vec![
            ("hidden_size", Value::U(64)),
            ("n_layers", Value::U(2)),
            ("n_heads", Value::U(4)),
            ("norm", Value::map(vec![("kind", Value::text("layer"))])),
        ]);
        let mut names = vec!["model.embed_tokens.weight".to_string()];
        for l in 0..2 {
            names.push(format!("model.layers.{l}.norm.weight"));
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                names.push(format!("model.layers.{l}.attn.{p}.weight"));
            }
        }
        // No `lm_head.weight` in the list: an encoder's result is its hidden
        // states, and refusing a BERT for lacking a language-modelling head
        // would be refusing it for being what it is.
        let m = synthesize("transformer.encoder", &params, &names).expect("synthesizes");
        assert_eq!(
            m.attrs
                .iter()
                .find(|(k, _)| k == "family")
                .map(|(_, v)| v.clone()),
            Some(Value::text("transformer.encoder"))
        );

        // The mask is the whole difference, and it is asserted rather than
        // assumed: an encoder that emitted `causal: true` would verify, run,
        // and quietly be a decoder.
        let attn: Vec<&Op> = m.functions[0]
            .1
            .body
            .blocks
            .iter()
            .flat_map(|b| b.ops.iter())
            .filter(|o| o.name == "attention")
            .collect();
        assert_eq!(attn.len(), 2, "one attention per layer");
        for a in &attn {
            assert_eq!(a.attr("causal"), Some(&Value::Bool(false)));
        }

        // The result is [B, S, hidden], not [B, S, vocab].
        let out = &m.functions[0].1.results[0];
        assert!(
            format!("{out:?}").contains("N(64)"),
            "the encoder should return hidden states: {out:?}"
        );

        let lookup = |name: &str| -> Option<(Vec<u64>, DType)> {
            if name == "model.embed_tokens.weight" {
                return Some((vec![256, 64], DType::BF16));
            }
            if name.ends_with("norm.weight") {
                return Some((vec![64], DType::F32));
            }
            Some((vec![64, 64], DType::BF16))
        };
        let cx = Context {
            tensor: Some(&lookup),
            ..Default::default()
        };
        let r = verify(&m, &cx);
        assert!(r.is_valid(), "{:?}", r.findings);
        assert_eq!(r.unknown, 0);

        // And the decoder built from the same weights still wants its head.
        let e = synthesize("transformer.decoder", &params, &names).unwrap_err();
        assert!(e.contains("lm_head.weight"), "{e}");
    }

    #[test]
    fn the_printer_shows_the_structure() {
        let m = attention_module(2, true);
        let s = m.print();
        assert!(s.contains("module @forward level=semantic"));
        assert!(s.contains("dialect omni.nn@1"));
        assert!(s.contains("omni.nn/attention@2"));
        assert!(s.contains("tensor<B×4×S×16, bf16>"));
    }

    #[test]
    fn every_dialect_describes_itself() {
        // §07.4.2: a dialect ships a DialectRef listing its ops and versions.
        for d in dialects() {
            let v = dialect_ref_value(d);
            assert_eq!(v.get("t").and_then(|x| x.as_str()), Some("omni.ir/dialect"));
            let ops = v.get("ops").and_then(|x| x.as_map()).unwrap();
            assert_eq!(ops.len(), d.ops.len());
            // Canonical CBOR round-trip, like every other structure object.
            let bytes = v.encode();
            assert_eq!(crate::cbor::decode(&bytes).unwrap().encode(), bytes);
        }
        // omni.core is frozen; the others are not (§07.4).
        assert!(dialect("omni.core").unwrap().frozen);
        assert!(!dialect("omni.nn").unwrap().frozen);
        // And omni.core contains no tensor mathematics at all, which is the
        // property §07.8's last row depends on.
        for op in dialect("omni.core").unwrap().ops {
            assert!(
                !["add", "mul", "matmul", "softmax"].contains(&op.name),
                "omni.core must not contain {}",
                op.name
            );
        }
    }

    #[test]
    fn inference_rejects_shapes_that_cannot_work() {
        let a = bf16(&[Dim::N(4), Dim::N(8)]);
        let b = bf16(&[Dim::N(16), Dim::N(4)]);
        let op = Op::new("omni.tensor", "matmul", 1).with_inputs(&[0, 1]);
        assert!(matches!(infer(&op, &[a.clone(), b]), Inferred::Ill(_)));
        // A symbolic inner dimension is allowed to match: dynamic shapes are the
        // default, and a verifier that refused them would refuse every real
        // deployment.
        let sym_b = bf16(&[sym("K"), Dim::N(4)]);
        assert!(matches!(
            infer(&op, &[a.clone(), sym_b]),
            Inferred::Types(_)
        ));
        // Mixed dtypes in an elementwise op.
        let f32t = Type::tensor(vec![Dim::N(4), Dim::N(8)], DType::F32);
        let add = Op::new("omni.tensor", "add", 1).with_inputs(&[0, 1]);
        assert!(matches!(infer(&add, &[a.clone(), f32t]), Inferred::Ill(_)));
        // A reshape that changes the element count.
        let rs = Op::new("omni.tensor", "reshape", 1)
            .with_inputs(&[0])
            .with_attr("shape", Value::Array(vec![Value::U(7), Value::U(7)]));
        assert!(matches!(
            infer(&rs, std::slice::from_ref(&a)),
            Inferred::Ill(_)
        ));
        // And one that does not: -1 is resolved from what is left.
        let rs = Op::new("omni.tensor", "reshape", 1)
            .with_inputs(&[0])
            .with_attr("shape", Value::Array(vec![Value::I(-1), Value::U(4)]));
        match infer(&rs, &[a]) {
            Inferred::Types(t) => assert_eq!(t[0].as_tensor().unwrap().0, &[Dim::N(8), Dim::N(4)]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_op_with_no_shape_function_is_unchecked_not_wrong() {
        // §07.9: an unknown op with a known type signature can be skipped for
        // structural validation and preserved on rewrite. Same for a known op
        // this build has no shape function for.
        let x = bf16(&[Dim::N(2), Dim::N(3), Dim::N(4)]);
        let op = Op::new("omni.nn", "pool", 1)
            .with_inputs(&[0])
            .with_attr("kind", Value::text("max"))
            .with_attr("window", Value::Array(vec![Value::U(2)]))
            .with_output(1, x.clone());
        assert!(matches!(
            infer(&op, std::slice::from_ref(&x)),
            Inferred::Unchecked(_)
        ));
        let f = Function {
            params: vec![("x".into(), x.clone())],
            results: vec![x.clone()],
            attrs: Vec::new(),
            body: Region {
                blocks: vec![Block {
                    args: Vec::new(),
                    ops: vec![op, Op::new("omni.core", "return", 1).with_inputs(&[1])],
                }],
            },
            constraints: Vec::new(),
        };
        let mut m = Module::new(Level::Semantic, "f");
        m.dialects = vec![
            DialectUse {
                ns: "omni.core".into(),
                version: 1,
                reference: None,
            },
            DialectUse {
                ns: "omni.nn".into(),
                version: 1,
                reference: None,
            },
        ];
        m.functions = vec![("f".into(), f)];
        let r = verify(&m, &Context::default());
        assert!(r.is_valid(), "{:?}", r.findings);
        assert_eq!(r.unchecked, 1);
    }
}
