//! §07 — a reference interpreter for OMNI-IR.
//!
//! [`crate::ir`] parses, verifies, prints, rewrites and synthesizes graphs.
//! None of that is *execution*, and §07's whole claim is about execution: a
//! model that describes its own computation so that a runtime which has never
//! heard of its architecture can still run it. This module is where that claim
//! gets tested rather than asserted.
//!
//! It is a reference interpreter in the specification's sense — clarity and
//! honesty over speed. Every tensor is a dense `Vec<f64>`, every reduction is a
//! loop, and nothing is fused. The point is to be the thing an optimized runtime
//! is checked *against*.
//!
//! ## What runs
//!
//! * All of `omni.core`: `constant`, `call`, `return`, `yield`, `if`, `while`,
//!   `scan`, `map`, `region`, `tuple`, `get`, `assert`, `debug`.
//! * All 31 `omni.tensor` ops, including a general `einsum` over explicit
//!   subscripts.
//! * `omni.quant`'s `quantize`, `dequantize`, `fake_quant` and `qmatmul`, with
//!   scale and zero taken from the op's *inputs* — which is what the IR form of
//!   §05 does, as against the expression form where they are sub-expressions.
//! * The `omni.nn` ops a decoder needs: `embedding`, `norm`, `rope`,
//!   `activation`, `attention`. `attention` is interpreted directly rather than
//!   through the shipped lowering, because the lowering declines the cases that
//!   matter — `causal`, `kv_groups`, `window`, `softcap` — and an interpreter
//!   that could only run the easy configuration would not be able to execute
//!   `graph synthesize`'s own output.
//! * `omni.io`'s `input` and `output`, which name the graph's boundary.
//!
//! ## What does not, and says so
//!
//! `omni.nn`'s `moe_route`, `ssm_scan`, `conv`, `conv1d_causal`, `pool` and
//! `interpolate`; `omni.io/external`; a `constant` naming a tensor no
//! [`Weights`] provider has. Each is refused **by name**, which is the same
//! three-valued honesty §15.1 requires of verification: an op this build does
//! not implement is not a wrong answer, it is a refusal to give one.
//!
//! Symbolic dimensions are resolved from the arguments actually passed, not from
//! the declared types: `[B, S, H]` with a `[2, 3, 8]` argument binds `B = 2` and
//! `S = 3`, and a later `[B, S]` that disagrees is an error rather than a
//! reshape.

use crate::cbor::Value;
use crate::dtype::{DType, Round};
use crate::expr::Dim;
use crate::expr::{Sum, Tensor};
use crate::ir::{Function, Module, Op, Region, Type};
use crate::layout::numel;

#[derive(Debug)]
pub enum Error {
    /// An op this build does not implement, named.
    Unsupported(String),
    Type(String),
    Bounds(String),
    /// An `omni.core/assert` that failed, which is the graph's own claim about
    /// itself and therefore a first-class outcome.
    Assert(String),
    /// The fuel or element budget ran out. A graph is untrusted input.
    Budget(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Type(m) => write!(f, "type error: {m}"),
            Error::Bounds(m) => write!(f, "out of range: {m}"),
            Error::Assert(m) => write!(f, "assertion failed: {m}"),
            Error::Budget(m) => write!(f, "budget exhausted: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// --------------------------------------------------------------------- inputs --

/// Where a `constant` naming a tensor, or an `omni.io/external`, gets its bytes.
///
/// A graph refers to weights by name (§07.5 synthesizes exactly that), and the
/// interpreter is deliberately not given a container: it is handed a provider, so
/// the same graph can be run against a real model, against a fixture, or against
/// nothing at all — and in the last case the missing name is reported rather than
/// filled in with zeros.
pub trait Weights {
    fn tensor(&self, name: &str) -> Option<Tensor>;
}

/// No weights at all. A graph that needs one says which.
impl Weights for () {
    fn tensor(&self, _: &str) -> Option<Tensor> {
        None
    }
}

impl Weights for Vec<(String, Tensor)> {
    fn tensor(&self, name: &str) -> Option<Tensor> {
        self.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
    }
}

/// Bounds on an interpretation. A graph is untrusted input (§12), so the loops
/// it can ask for are bounded and the bound is reported when it is hit.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Elements in any one intermediate tensor.
    pub max_elems: u64,
    /// Ops executed, counting each iteration of a loop body separately.
    pub fuel: u64,
    /// Nesting depth of regions and calls.
    pub max_depth: usize,
    /// Iterations of a single `while`.
    pub max_iters: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_elems: 1 << 24,
            fuel: 1 << 22,
            max_depth: 32,
            max_iters: 1 << 16,
        }
    }
}

/// What an interpretation produced.
pub struct Outcome {
    /// The values `omni.core/return` returned, in order.
    pub returned: Vec<Tensor>,
    /// Values named by `omni.io/output`, in the order the ops appear.
    pub outputs: Vec<(String, Tensor)>,
    /// Ops executed, loop iterations counted separately.
    pub ops: u64,
    /// Symbolic dimensions, as the arguments bound them.
    pub dims: Vec<(String, u64)>,
    /// `omni.core/debug` labels reached, in order.
    pub debug: Vec<String>,
}

impl std::fmt::Debug for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Outcome {{ returned {:?}, {} output(s), {} ops, dims {:?} }}",
            self.returned.iter().map(|t| &t.shape).collect::<Vec<_>>(),
            self.outputs.len(),
            self.ops,
            self.dims
        )
    }
}

// -------------------------------------------------------------------- the run --

/// Runs a module's entry function over positional arguments.
///
/// The entry function's parameters are values `0..params.len()`, which is the
/// convention [`crate::ir::synthesize`] writes and §07.3 implies for a function
/// whose entry block declares no arguments of its own.
pub fn run(m: &Module, args: &[Tensor], w: &dyn Weights, limits: &Limits) -> Res<Outcome> {
    let entry = m.entry.clone();
    run_function(m, &entry, args, w, limits)
}

pub fn run_function(
    m: &Module,
    name: &str,
    args: &[Tensor],
    w: &dyn Weights,
    limits: &Limits,
) -> Res<Outcome> {
    let f = m
        .function(name)
        .ok_or_else(|| Error::Type(format!("no function `{name}` in this module")))?;
    let mut st = State {
        module: m,
        weights: w,
        limits: limits.clone(),
        ops: 0,
        dims: Vec::new(),
        debug: Vec::new(),
        outputs: Vec::new(),
    };
    let returned = st.call(f, args, 0)?;
    Ok(Outcome {
        returned,
        outputs: std::mem::take(&mut st.outputs),
        ops: st.ops,
        dims: st.dims,
        debug: st.debug,
    })
}

struct State<'a> {
    module: &'a Module,
    weights: &'a dyn Weights,
    limits: Limits,
    ops: u64,
    dims: Vec<(String, u64)>,
    debug: Vec<String>,
    outputs: Vec<(String, Tensor)>,
}

/// SSA values, indexed by id. Dense numbering (§07.3) makes this a `Vec`.
#[derive(Default)]
struct Env {
    slots: Vec<Option<Tensor>>,
    /// A `tuple`'s members, for the `get` that reads them. Tuples are the one
    /// non-tensor value in the IR, and giving them their own table keeps `slots`
    /// a `Tensor` rather than an enum every op has to unwrap.
    tuples: Vec<(u32, Vec<Tensor>)>,
}

impl Env {
    fn set(&mut self, id: u32, t: Tensor) {
        let i = id as usize;
        if self.slots.len() <= i {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = Some(t);
    }

    fn get(&self, id: u32) -> Res<&Tensor> {
        self.slots
            .get(id as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| Error::Type(format!("value %{id} is read before it is defined")))
    }

    fn tuple(&self, id: u32) -> Option<&[Tensor]> {
        self.tuples
            .iter()
            .find(|(k, _)| *k == id)
            .map(|(_, v)| v.as_slice())
    }
}

/// How a block finished.
enum Flow {
    /// Ran off the end.
    Fell,
    /// `omni.core/return`.
    Return(Vec<Tensor>),
    /// `omni.core/yield` — a region's result.
    Yield(Vec<Tensor>),
}

impl State<'_> {
    fn spend(&mut self, n: u64) -> Res<()> {
        self.ops += n;
        if self.ops > self.limits.fuel {
            return Err(Error::Budget(format!(
                "{} ops executed, over the limit of {}",
                self.ops, self.limits.fuel
            )));
        }
        Ok(())
    }

    fn check_size(&self, shape: &[u64]) -> Res<()> {
        let n = numel(shape);
        if n > self.limits.max_elems {
            return Err(Error::Budget(format!(
                "a {shape:?} tensor is {n} elements, over the limit of {}",
                self.limits.max_elems
            )));
        }
        Ok(())
    }

    /// Binds a symbolic dimension, or checks it against what it was already
    /// bound to. Two different sizes for one name is an error, not a reshape.
    fn bind_dim(&mut self, name: &str, size: u64) -> Res<()> {
        match self.dims.iter().find(|(n, _)| n == name) {
            Some((_, got)) if *got != size => Err(Error::Type(format!(
                "dimension `{name}` is {got} here and {size} there"
            ))),
            Some(_) => Ok(()),
            None => {
                self.dims.push((name.to_string(), size));
                Ok(())
            }
        }
    }

    /// Binds the symbolic dimensions of a declared type against a real tensor.
    fn bind_type(&mut self, t: &Type, got: &Tensor, what: &str) -> Res<()> {
        let Some((shape, _)) = t.as_tensor() else {
            return Ok(());
        };
        if shape.len() != got.shape.len() {
            return Err(Error::Type(format!(
                "{what}: declared rank {} but the value has rank {}",
                shape.len(),
                got.shape.len()
            )));
        }
        for (d, n) in shape.iter().zip(&got.shape) {
            match d {
                Dim::N(k) if k != n => {
                    return Err(Error::Type(format!(
                        "{what}: declared {shape:?} but the value is {:?}",
                        got.shape
                    )))
                }
                Dim::Sym(name) => self.bind_dim(name, *n)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn call(&mut self, f: &Function, args: &[Tensor], depth: usize) -> Res<Vec<Tensor>> {
        if depth > self.limits.max_depth {
            return Err(Error::Budget(format!("nested {depth} deep")));
        }
        if args.len() != f.params.len() {
            return Err(Error::Type(format!(
                "the function takes {} argument(s) and was given {}",
                f.params.len(),
                args.len()
            )));
        }
        let mut env = Env::default();
        for (i, ((name, ty), a)) in f.params.iter().zip(args).enumerate() {
            self.bind_type(ty, a, &format!("parameter `{name}`"))?;
            env.set(i as u32, a.clone());
        }
        // The constraints the function declares over its own dimensions (§07.3.1)
        // are checked against what the arguments bound, because a graph that says
        // `S >= 1` and is handed zero tokens should say so here rather than
        // produce an empty answer.
        for c in &f.constraints {
            if let Some((_, got)) = self.dims.iter().find(|(n, _)| n == &c.dim) {
                let ok = match c.rel {
                    crate::ir::Rel::Eq => *got == c.bound,
                    crate::ir::Rel::Ge => *got >= c.bound,
                    crate::ir::Rel::Le => *got <= c.bound,
                    crate::ir::Rel::Multiple => c.bound != 0 && got % c.bound == 0,
                };
                if !ok {
                    return Err(Error::Type(format!(
                        "the function declares {} {} {} and it is {got}",
                        c.dim,
                        match c.rel {
                            crate::ir::Rel::Eq => "==",
                            crate::ir::Rel::Ge => ">=",
                            crate::ir::Rel::Le => "<=",
                            crate::ir::Rel::Multiple => "a multiple of",
                        },
                        c.bound
                    )));
                }
            }
        }
        match self.region(&f.body, &mut env, depth)? {
            Flow::Return(v) | Flow::Yield(v) => Ok(v),
            Flow::Fell => Ok(Vec::new()),
        }
    }

    /// Runs a region's first block. §07.3 has no branch op, so a region is one
    /// block in practice; a second block is unreachable and reported as such
    /// rather than silently ignored.
    fn region(&mut self, r: &Region, env: &mut Env, depth: usize) -> Res<Flow> {
        let Some(b) = r.blocks.first() else {
            return Ok(Flow::Fell);
        };
        if r.blocks.len() > 1 {
            return Err(Error::Unsupported(format!(
                "a region with {} blocks: §07.3 has no branch op, so blocks after \
                 the first are unreachable and this build will not guess an order",
                r.blocks.len()
            )));
        }
        for op in &b.ops {
            self.spend(1)?;
            match self.step(op, env, depth)? {
                Flow::Fell => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Fell)
    }

    /// Runs one region with block arguments bound — the loop-body case.
    fn region_with(
        &mut self,
        r: &Region,
        args: &[Tensor],
        outer: &Env,
        depth: usize,
    ) -> Res<Vec<Tensor>> {
        let Some(b) = r.blocks.first() else {
            return Ok(args.to_vec());
        };
        // A region sees the values defined outside it (§07.3's regions are not
        // closures with their own scope), plus its own arguments.
        let mut env = Env {
            slots: outer.slots.clone(),
            tuples: outer.tuples.clone(),
        };
        if b.args.len() != args.len() {
            return Err(Error::Type(format!(
                "the region takes {} argument(s) and was given {}",
                b.args.len(),
                args.len()
            )));
        }
        for ((id, ty), a) in b.args.iter().zip(args) {
            self.bind_type(ty, a, &format!("region argument %{id}"))?;
            env.set(*id, a.clone());
        }
        match self.region(r, &mut env, depth + 1)? {
            Flow::Yield(v) | Flow::Return(v) => Ok(v),
            Flow::Fell => Ok(Vec::new()),
        }
    }

    fn step(&mut self, op: &Op, env: &mut Env, depth: usize) -> Res<Flow> {
        // `get` reads a tuple, which is the one value in the IR that is not a
        // tensor — so it is handled before the operands are resolved as tensors
        // rather than after, when that resolution has already failed.
        if op.dialect == "omni.core" && op.name == "get" {
            let k = int_attr(op, "index")? as usize;
            let src = op.inputs.first().copied().unwrap_or(0);
            let picked = match env.tuple(src) {
                Some(v) => v.get(k).cloned().ok_or_else(|| {
                    Error::Bounds(format!("get {k} of a {}-member tuple", v.len()))
                })?,
                // `get` of a non-tuple is how a multi-result op's results are
                // read, and those are already in their own slots.
                None => env.get(src)?.clone(),
            };
            env.set(out_id(op)?, picked);
            return Ok(Flow::Fell);
        }
        let ins: Vec<Tensor> = op
            .inputs
            .iter()
            .map(|i| env.get(*i).cloned())
            .collect::<Res<Vec<Tensor>>>()?;

        // Control flow first: these are the ops that do not simply produce a
        // tensor from tensors.
        if op.dialect == "omni.core" {
            match op.name.as_str() {
                "return" => return Ok(Flow::Return(ins)),
                "yield" => return Ok(Flow::Yield(ins)),
                "debug" => {
                    self.debug.push(
                        op.attr("label")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    );
                    return Ok(Flow::Fell);
                }
                "assert" => {
                    let t = ins
                        .first()
                        .ok_or_else(|| Error::Type("assert has no operand".into()))?;
                    if t.data.contains(&0.0) {
                        return Err(Error::Assert(
                            op.attr("message")
                                .and_then(|v| v.as_str())
                                .unwrap_or("an omni.core/assert operand was false")
                                .to_string(),
                        ));
                    }
                    return Ok(Flow::Fell);
                }
                "tuple" => {
                    let id = out_id(op)?;
                    env.tuples.retain(|(k, _)| *k != id);
                    env.tuples.push((id, ins));
                    return Ok(Flow::Fell);
                }
                "call" => {
                    let callee = op
                        .attr("callee")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::Type("call has no `callee`".into()))?;
                    let f = self.module.function(callee).ok_or_else(|| {
                        Error::Type(format!("call to `{callee}`, which this module has no"))
                    })?;
                    // Cloned so the borrow of `self.module` ends before `call`
                    // takes `&mut self`.
                    let f = f.clone();
                    let out = self.call(&f, &ins, depth + 1)?;
                    for ((id, _), t) in op.outputs.iter().zip(out) {
                        env.set(*id, t);
                    }
                    return Ok(Flow::Fell);
                }
                "if" => {
                    let cond = ins
                        .first()
                        .ok_or_else(|| Error::Type("if has no condition".into()))?;
                    if cond.numel() != 1 {
                        return Err(Error::Type(format!(
                            "if: the condition is {:?}, and a branch needs one value",
                            cond.shape
                        )));
                    }
                    let taken = if cond.data[0] != 0.0 { 0 } else { 1 };
                    let r = op.regions.get(taken).ok_or_else(|| {
                        Error::Type(format!("if has no region {taken}; §07.3 requires two"))
                    })?;
                    let out = self.region_with(r, &[], env, depth)?;
                    for ((id, _), t) in op.outputs.iter().zip(out) {
                        env.set(*id, t);
                    }
                    return Ok(Flow::Fell);
                }
                "while" => {
                    let (cr, br) = (
                        op.regions
                            .first()
                            .ok_or_else(|| Error::Type("while has no condition region".into()))?,
                        op.regions
                            .get(1)
                            .ok_or_else(|| Error::Type("while has no body region".into()))?,
                    );
                    let mut carried = ins;
                    let mut iters = 0u64;
                    loop {
                        iters += 1;
                        if iters > self.limits.max_iters {
                            return Err(Error::Budget(format!(
                                "while ran {iters} iterations, over the limit of {}",
                                self.limits.max_iters
                            )));
                        }
                        self.spend(1)?;
                        let c = self.region_with(cr, &carried, env, depth)?;
                        let go = c.first().map(|t| t.data.first() != Some(&0.0)) == Some(true);
                        if !go {
                            break;
                        }
                        carried = self.region_with(br, &carried, env, depth)?;
                    }
                    for ((id, _), t) in op.outputs.iter().zip(carried) {
                        env.set(*id, t);
                    }
                    return Ok(Flow::Fell);
                }
                "region" => {
                    let r = op
                        .regions
                        .first()
                        .ok_or_else(|| Error::Type("region has no region".into()))?;
                    let out = self.region_with(r, &ins, env, depth)?;
                    for ((id, _), t) in op.outputs.iter().zip(out) {
                        env.set(*id, t);
                    }
                    return Ok(Flow::Fell);
                }
                "map" | "scan" => {
                    let r = op
                        .regions
                        .first()
                        .ok_or_else(|| Error::Type("map/scan has no body".into()))?;
                    let axis = int_attr_or(op, "axis", 0)? as usize;
                    let reverse = matches!(op.attr("reverse"), Some(Value::Bool(true)));
                    let scan = op.name == "scan";
                    let out = self.map_or_scan(r, &ins, axis, reverse, scan, env, depth)?;
                    for ((id, _), t) in op.outputs.iter().zip(out) {
                        env.set(*id, t);
                    }
                    return Ok(Flow::Fell);
                }
                "func" => {
                    // A nested `func` is a definition, not a computation.
                    return Ok(Flow::Fell);
                }
                _ => {}
            }
        }

        // Everything else produces tensors from tensors.
        let outs = self.compute(op, &ins, env)?;
        if outs.len() < op.outputs.len() {
            return Err(Error::Type(format!(
                "{} produced {} result(s) for {} declared",
                op.qualified(),
                outs.len(),
                op.outputs.len()
            )));
        }
        for ((id, ty), t) in op.outputs.iter().zip(outs) {
            self.check_size(&t.shape)?;
            self.bind_type(ty, &t, &format!("{} result %{id}", op.qualified()))?;
            env.set(*id, t);
        }
        Ok(Flow::Fell)
    }

    /// `map` and `scan` (§07.3): the body runs once per slice along `axis`.
    /// `scan` threads the first operand as an accumulator; `map` does not.
    #[allow(clippy::too_many_arguments)]
    fn map_or_scan(
        &mut self,
        body: &Region,
        ins: &[Tensor],
        axis: usize,
        reverse: bool,
        scan: bool,
        env: &Env,
        depth: usize,
    ) -> Res<Vec<Tensor>> {
        let (carry, mapped) = if scan {
            let (a, b) = ins.split_at(1.min(ins.len()));
            (a.first().cloned(), b)
        } else {
            (None, ins)
        };
        let over = mapped
            .first()
            .ok_or_else(|| Error::Type("map/scan has nothing to iterate over".into()))?;
        if axis >= over.shape.len() {
            return Err(Error::Bounds(format!(
                "map/scan axis {axis} of a rank-{} tensor",
                over.shape.len()
            )));
        }
        let n = over.shape[axis];
        let mut acc = carry;
        let mut slices: Vec<Tensor> = Vec::new();
        let order: Vec<u64> = if reverse {
            (0..n).rev().collect()
        } else {
            (0..n).collect()
        };
        for i in order {
            let mut args: Vec<Tensor> = Vec::new();
            if let Some(a) = &acc {
                args.push(a.clone());
            }
            for t in mapped {
                args.push(take_index(t, axis, i)?);
            }
            self.spend(1)?;
            let out = self.region_with(body, &args, env, depth)?;
            if scan {
                acc = out.first().cloned();
                if let Some(rest) = out.get(1) {
                    slices.push(rest.clone());
                } else if let Some(a) = &acc {
                    slices.push(a.clone());
                }
            } else {
                slices.push(
                    out.into_iter()
                        .next()
                        .ok_or_else(|| Error::Type("a map body yielded nothing".into()))?,
                );
            }
        }
        if reverse {
            slices.reverse();
        }
        let stacked = stack(&slices, axis)?;
        Ok(match acc {
            Some(a) if scan => vec![a, stacked],
            _ => vec![stacked],
        })
    }
}

// --------------------------------------------------------------------- op impl --

impl State<'_> {
    fn compute(&mut self, op: &Op, ins: &[Tensor], env: &Env) -> Res<Vec<Tensor>> {
        let _ = env;
        let out_dtype = op
            .outputs
            .first()
            .and_then(|(_, t)| t.as_tensor().map(|(_, d)| d.clone()));
        match (op.dialect.as_str(), op.name.as_str()) {
            ("omni.core", "constant") => Ok(vec![self.constant(op)?]),
            ("omni.io", "input") => Err(Error::Unsupported(format!(
                "omni.io/input `{}`: this interpreter binds a graph's arguments \
                 through the entry function's parameters, and a graph that also \
                 declares io inputs is describing a second convention this build \
                 does not implement",
                op.attr("name").and_then(|v| v.as_str()).unwrap_or("")
            ))),
            ("omni.io", "output") => {
                let name = op
                    .attr("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let t = ins
                    .first()
                    .ok_or_else(|| Error::Type("output has no operand".into()))?;
                self.outputs.push((name, t.clone()));
                Ok(Vec::new())
            }
            ("omni.io", "external") => Err(Error::Unsupported(format!(
                "omni.io/external `{}`: an external operand is fetched, and this \
                 interpreter is given weights rather than a network",
                op.attr("id").and_then(|v| v.as_str()).unwrap_or("")
            ))),
            ("omni.tensor", _) => self.tensor_op(op, ins, out_dtype),
            ("omni.quant", _) => self.quant_op(op, ins, out_dtype),
            ("omni.nn", _) => self.nn_op(op, ins, out_dtype),
            _ => Err(Error::Unsupported(format!(
                "{}: this build implements omni.core, omni.tensor, omni.quant, \
                 omni.io and part of omni.nn. A dialect it does not know is a \
                 refusal and not a wrong answer (§15.1)",
                op.qualified()
            ))),
        }
    }

    fn constant(&mut self, op: &Op) -> Res<Tensor> {
        let (shape, dtype) = op
            .outputs
            .first()
            .and_then(|(_, t)| t.as_tensor())
            .ok_or_else(|| Error::Type("constant has no tensor result type".into()))?;
        if let Some(name) = op.attr("tensor").and_then(|v| v.as_str()) {
            let t = self.weights.tensor(name).ok_or_else(|| {
                Error::Unsupported(format!(
                    "omni.core/constant names the tensor `{name}`, which no weights \
                     provider has. A graph is not runnable without the weights it \
                     refers to, and inventing them would be worse than saying so"
                ))
            })?;
            // The declared shape may be partly dynamic — the vocabulary of an
            // embedding table usually is — so what the tensor says wins for
            // `Dynamic` and has to agree everywhere else.
            for (d, n) in shape.iter().zip(&t.shape) {
                if let Dim::N(k) = d {
                    if k != n {
                        return Err(Error::Type(format!(
                            "constant `{name}` is declared {shape:?} and the weights \
                             are {:?}",
                            t.shape
                        )));
                    }
                }
            }
            return Ok(t);
        }
        let v = op
            .attr("value")
            .ok_or_else(|| Error::Type("constant has neither `value` nor `tensor`".into()))?;
        let sizes = concrete(shape).ok_or_else(|| {
            Error::Type("a constant's shape must be concrete; a symbolic one has no bytes".into())
        })?;
        let n = numel(&sizes);
        self.check_size(&sizes)?;
        let data = match v {
            Value::Array(xs) => {
                let mut out = Vec::with_capacity(xs.len());
                for x in xs {
                    out.push(scalar_of(x)?);
                }
                if out.len() as u64 != n {
                    return Err(Error::Type(format!(
                        "constant has {} value(s) for a {sizes:?} result",
                        out.len()
                    )));
                }
                out
            }
            other => vec![scalar_of(other)?; n as usize],
        };
        Ok(Tensor::new(sizes, dtype.clone(), data))
    }
}

fn out_id(op: &Op) -> Res<u32> {
    op.outputs
        .first()
        .map(|(id, _)| *id)
        .ok_or_else(|| Error::Type(format!("{} has no result", op.qualified())))
}

fn scalar_of(v: &Value) -> Res<f64> {
    match v {
        Value::U(n) => Ok(*n as f64),
        Value::I(n) => Ok(*n as f64),
        Value::F64(f) => Ok(*f),
        Value::Bool(b) => Ok(f64::from(u8::from(*b))),
        other => Err(Error::Type(format!(
            "{other:?} is not a number a constant can hold"
        ))),
    }
}

fn concrete(shape: &[Dim]) -> Option<Vec<u64>> {
    shape
        .iter()
        .map(|d| match d {
            Dim::N(n) => Some(*n),
            _ => None,
        })
        .collect()
}

fn int_attr(op: &Op, key: &str) -> Res<i64> {
    op.attr(key)
        .and_then(as_int)
        .ok_or_else(|| Error::Type(format!("{} has no integer `{key}`", op.qualified())))
}

fn int_attr_or(op: &Op, key: &str, default: i64) -> Res<i64> {
    Ok(op.attr(key).and_then(as_int).unwrap_or(default))
}

fn as_int(v: &Value) -> Option<i64> {
    match v {
        Value::U(n) => i64::try_from(*n).ok(),
        Value::I(n) => Some(*n),
        _ => None,
    }
}

fn int_list(op: &Op, key: &str) -> Res<Vec<i64>> {
    match op.attr(key) {
        Some(Value::Array(xs)) => xs
            .iter()
            .map(|x| {
                as_int(x).ok_or_else(|| {
                    Error::Type(format!("{}: `{key}` has a non-integer", op.qualified()))
                })
            })
            .collect(),
        _ => Err(Error::Type(format!(
            "{} has no integer list `{key}`",
            op.qualified()
        ))),
    }
}

fn float_attr(op: &Op, key: &str) -> Option<f64> {
    match op.attr(key) {
        Some(Value::F64(f)) => Some(*f),
        Some(Value::U(n)) => Some(*n as f64),
        Some(Value::I(n)) => Some(*n as f64),
        _ => None,
    }
}

/// Resolves a possibly negative axis against a rank, as every array library does.
fn axis_of(a: i64, rank: usize) -> Res<usize> {
    let r = rank as i64;
    let k = if a < 0 { a + r } else { a };
    if k < 0 || k >= r {
        return Err(Error::Bounds(format!("axis {a} of a rank-{rank} tensor")));
    }
    Ok(k as usize)
}

/// One slice along `axis`, with that axis dropped.
fn take_index(t: &Tensor, axis: usize, i: u64) -> Res<Tensor> {
    if axis >= t.shape.len() || i >= t.shape[axis] {
        return Err(Error::Bounds(format!(
            "index {i} on axis {axis} of {:?}",
            t.shape
        )));
    }
    let mut shape: Vec<u64> = t.shape.clone();
    shape.remove(axis);
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        let mut src = idx.clone();
        src.insert(axis, i);
        data.push(t.at(&src));
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, t.dtype.clone(), data))
}

/// The inverse of [`take_index`]: slices stacked along a new `axis`.
fn stack(slices: &[Tensor], axis: usize) -> Res<Tensor> {
    let first = slices
        .first()
        .ok_or_else(|| Error::Type("nothing to stack".into()))?;
    for s in slices {
        if s.shape != first.shape {
            return Err(Error::Type(format!(
                "stacking {:?} with {:?}",
                first.shape, s.shape
            )));
        }
    }
    let axis = axis.min(first.shape.len());
    let mut shape = first.shape.clone();
    shape.insert(axis, slices.len() as u64);
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        let k = idx[axis] as usize;
        let mut src = idx.clone();
        src.remove(axis);
        data.push(slices[k].at(&src));
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, first.dtype.clone(), data))
}

/// NumPy broadcasting of two shapes.
fn broadcast(a: &[u64], b: &[u64]) -> Res<Vec<u64>> {
    let r = a.len().max(b.len());
    let mut out = Vec::with_capacity(r);
    for k in 0..r {
        let x = if k + a.len() >= r {
            a[k + a.len() - r]
        } else {
            1
        };
        let y = if k + b.len() >= r {
            b[k + b.len() - r]
        } else {
            1
        };
        out.push(match (x, y) {
            (1, y) => y,
            (x, 1) => x,
            (x, y) if x == y => x,
            _ => {
                return Err(Error::Type(format!(
                    "shapes {a:?} and {b:?} do not broadcast"
                )))
            }
        });
    }
    Ok(out)
}

fn zip_with(a: &Tensor, b: &Tensor, dtype: &DType, f: impl Fn(f64, f64) -> f64) -> Res<Tensor> {
    let shape = broadcast(&a.shape, &b.shape)?;
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        data.push(f(a.broadcast_at(&idx), b.broadcast_at(&idx)));
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, dtype.clone(), data))
}

fn map_with(t: &Tensor, dtype: &DType, f: impl Fn(f64) -> f64) -> Tensor {
    Tensor::new(
        t.shape.clone(),
        dtype.clone(),
        t.data.iter().copied().map(f).collect(),
    )
}

/// The error function, to double precision. `erf` is an `omni.tensor` op and
/// there is no `f64::erf` in the standard library, so it is here — Abramowitz &
/// Stegun 7.1.26 refined by one Newton step against the series, which is well
/// inside the tolerance any activation needs.
fn erf(x: f64) -> f64 {
    // erf is odd; compute for |x| and put the sign back.
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs();
    if z < 0.5 {
        // Maclaurin series: converges fast and is exact to f64 well past 0.5.
        let mut term = z;
        let mut sum = z;
        let zz = z * z;
        for k in 1..40 {
            term *= -zz / k as f64;
            let add = term / (2.0 * k as f64 + 1.0);
            sum += add;
            if add.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        return s * sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    // The tail, by the standard rational approximation to `erfc` (Numerical
    // Recipes §6.2): fractional error below 1.2e-7, which is four orders of
    // magnitude inside anything a bf16 activation can represent. Stated because
    // an approximation that is not declared is a lie about precision.
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let erfc = t * poly.exp();
    s * (1.0 - erfc)
}

/// §04.3's rounding modes are a closed set, so a name outside it is refused: a
/// cast that silently rounded the other way would be the quietest possible bug.
fn round_of(op: &Op) -> Res<Round> {
    Ok(match op.attr("round").and_then(|v| v.as_str()) {
        Some("rne") | None => Round::Rne,
        Some("rtz") => Round::Rtz,
        Some("rup") => Round::Rup,
        Some("rdown") => Round::Rdown,
        Some(other) => {
            return Err(Error::Unsupported(format!(
                "rounding mode `{other}`: §04.3 names rne, rtz, rup, rdown and \
                 stochastic, and stochastic needs a seed this op does not carry"
            )))
        }
    })
}

/// Rounds every element through `dtype`, which is what a cast *is*: the values a
/// runtime would actually hold, not the values it started with.
fn round_through(t: &Tensor, dtype: &DType, round: Round) -> Res<Tensor> {
    let mut buf = vec![0u8; dtype.packed_bytes(1).max(1) as usize];
    let mut data = Vec::with_capacity(t.data.len());
    for v in &t.data {
        if !dtype.encode(&mut buf, 0, *v, round) {
            return Err(Error::Unsupported(format!(
                "{} has no element encoder, so a cast to it cannot be performed",
                dtype.label()
            )));
        }
        data.push(dtype.decode(&buf, 0).unwrap_or(f64::NAN));
    }
    Ok(Tensor::new(t.shape.clone(), dtype.clone(), data))
}

// ------------------------------------------------------------- omni.tensor --

impl State<'_> {
    fn tensor_op(&mut self, op: &Op, ins: &[Tensor], out: Option<DType>) -> Res<Vec<Tensor>> {
        let a = |k: usize| -> Res<&Tensor> {
            ins.get(k)
                .ok_or_else(|| Error::Type(format!("{} has no operand {k}", op.qualified())))
        };
        // The result dtype the graph declared. An interpreter that computed in
        // f64 and reported f64 would be answering a different question than the
        // graph asked, so elementwise results are rounded through it.
        let dt = |t: &Tensor| out.clone().unwrap_or_else(|| t.dtype.clone());
        let one = |t: Tensor| -> Res<Vec<Tensor>> { Ok(vec![t]) };

        match op.name.as_str() {
            "add" | "sub" | "mul" | "div" | "maximum" | "minimum" => {
                let (x, y) = (a(0)?, a(1)?);
                let f: fn(f64, f64) -> f64 = match op.name.as_str() {
                    "add" => |p, q| p + q,
                    "sub" => |p, q| p - q,
                    "mul" => |p, q| p * q,
                    "div" => |p, q| p / q,
                    "maximum" => f64::max,
                    _ => f64::min,
                };
                one(zip_with(x, y, &dt(x), f)?)
            }
            "neg" | "exp" | "log" | "sqrt" | "rsqrt" | "tanh" | "sigmoid" | "erf" => {
                let x = a(0)?;
                let f: fn(f64) -> f64 = match op.name.as_str() {
                    "neg" => |v| -v,
                    "exp" => f64::exp,
                    "log" => f64::ln,
                    "sqrt" => f64::sqrt,
                    "rsqrt" => |v| 1.0 / v.sqrt(),
                    "tanh" => f64::tanh,
                    "sigmoid" => |v| 1.0 / (1.0 + (-v).exp()),
                    _ => erf,
                };
                one(map_with(x, &dt(x), f))
            }
            "where" => {
                let (c, x, y) = (a(0)?, a(1)?, a(2)?);
                let shape = broadcast(&broadcast(&c.shape, &x.shape)?, &y.shape)?;
                let n = numel(&shape);
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    data.push(if c.broadcast_at(&idx) != 0.0 {
                        x.broadcast_at(&idx)
                    } else {
                        y.broadcast_at(&idx)
                    });
                    crate::expr::bump(&mut idx, &shape);
                }
                one(Tensor::new(shape, dt(x), data))
            }
            "cast" => {
                let x = a(0)?;
                let to = op
                    .attr("dtype")
                    .ok_or_else(|| Error::Type("cast has no `dtype`".into()))
                    .and_then(|v| DType::from_value(v).map_err(Error::Type))?;
                one(round_through(x, &to, round_of(op)?)?)
            }
            "matmul" => {
                let (x, y) = (a(0)?, a(1)?);
                if x.shape.len() < 2 || y.shape.len() < 2 {
                    return Err(Error::Type(
                        "matmul: both operands must be rank >= 2".into(),
                    ));
                }
                let (m, k) = (x.shape[x.shape.len() - 2], x.shape[x.shape.len() - 1]);
                let (k2, nn) = (y.shape[y.shape.len() - 2], y.shape[y.shape.len() - 1]);
                if k != k2 {
                    return Err(Error::Type(format!(
                        "matmul: {:?} times {:?} — the contracted dimensions are {k} and {k2}",
                        x.shape, y.shape
                    )));
                }
                let batch =
                    broadcast(&x.shape[..x.shape.len() - 2], &y.shape[..y.shape.len() - 2])?;
                let mut shape = batch;
                shape.push(m);
                shape.push(nn);
                self.check_size(&shape)?;
                // `accum` names the summation order, which §07.9 makes explicit
                // because an unpinned order is not reproducible.
                let sum = match op.attr("accum").and_then(|v| v.as_str()) {
                    Some("sequential") => Sum::Sequential,
                    Some("pairwise") => Sum::Pairwise,
                    Some("kahan") => Sum::Kahan,
                    _ => Sum::Sequential,
                };
                let t = crate::expr::matmul(x, y, sum, &shape, &dt(x))
                    .map_err(|e| Error::Type(e.to_string()))?;
                one(t)
            }
            "reduce" => {
                let x = a(0)?;
                let kind = op
                    .attr("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("reduce has no `kind`".into()))?
                    .to_string();
                let axes: Vec<usize> = int_list(op, "axes")?
                    .into_iter()
                    .map(|k| axis_of(k, x.shape.len()))
                    .collect::<Res<Vec<usize>>>()?;
                let keep = matches!(op.attr("keepdims"), Some(Value::Bool(true)));
                one(reduce(x, &axes, &kind, keep, &dt(x))?)
            }
            "softmax" => {
                let x = a(0)?;
                let axis = axis_of(int_attr_or(op, "axis", -1)?, x.shape.len())?;
                one(softmax(x, axis, &dt(x)))
            }
            "cumsum" => {
                let x = a(0)?;
                let axis = axis_of(int_attr_or(op, "axis", -1)?, x.shape.len())?;
                let mut t = x.clone();
                t.dtype = dt(x);
                let stride = t.strides()[axis];
                let len = t.shape[axis];
                let outer = numel(&t.shape) / len.max(1);
                for o in 0..outer {
                    // Walk the positions whose index on `axis` is zero, then
                    // stride along it.
                    let base = position_skipping(&t.shape, axis, o);
                    let mut acc = 0.0;
                    for i in 0..len {
                        let p = (base + i * stride) as usize;
                        acc += t.data[p];
                        t.data[p] = acc;
                    }
                }
                one(t)
            }
            "transpose" => {
                let x = a(0)?;
                let perm: Vec<usize> = int_list(op, "perm")?
                    .into_iter()
                    .map(|k| axis_of(k, x.shape.len()))
                    .collect::<Res<Vec<usize>>>()?;
                if perm.len() != x.shape.len() {
                    return Err(Error::Type(format!(
                        "transpose: perm {perm:?} for a rank-{} tensor",
                        x.shape.len()
                    )));
                }
                let shape: Vec<u64> = perm.iter().map(|p| x.shape[*p]).collect();
                let n = numel(&shape);
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let mut src = vec![0u64; perm.len()];
                    for (k, p) in perm.iter().enumerate() {
                        src[*p] = idx[k];
                    }
                    data.push(x.at(&src));
                    crate::expr::bump(&mut idx, &shape);
                }
                one(Tensor::new(shape, dt(x), data))
            }
            "reshape" => {
                let x = a(0)?;
                let want = int_list(op, "shape")?;
                let shape = resolve_reshape(&want, x.numel())?;
                one(Tensor::new(shape, dt(x), x.data.clone()))
            }
            "broadcast" => {
                let x = a(0)?;
                let shape: Vec<u64> = int_list(op, "shape")?
                    .into_iter()
                    .map(|k| {
                        u64::try_from(k)
                            .map_err(|_| Error::Type("broadcast: a negative extent".into()))
                    })
                    .collect::<Res<Vec<u64>>>()?;
                self.check_size(&shape)?;
                let n = numel(&shape);
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    data.push(x.broadcast_at(&idx));
                    crate::expr::bump(&mut idx, &shape);
                }
                one(Tensor::new(shape, dt(x), data))
            }
            "slice" => {
                let x = a(0)?;
                let (start, stop) = (int_list(op, "start")?, int_list(op, "stop")?);
                let step = op
                    .attr("step")
                    .map(|_| int_list(op, "step"))
                    .transpose()?
                    .unwrap_or_else(|| vec![1; start.len()]);
                if start.len() != x.shape.len() || stop.len() != x.shape.len() {
                    return Err(Error::Type(format!(
                        "slice: {} start(s) and {} stop(s) for a rank-{} tensor",
                        start.len(),
                        stop.len(),
                        x.shape.len()
                    )));
                }
                let mut shape = Vec::with_capacity(start.len());
                for k in 0..start.len() {
                    let (lo, hi, st) = (start[k], stop[k], *step.get(k).unwrap_or(&1));
                    if st <= 0 {
                        return Err(Error::Unsupported(
                            "slice: a zero or negative step; reverse with a gather so the \
                             direction is visible"
                                .into(),
                        ));
                    }
                    if lo < 0 || hi < lo || hi as u64 > x.shape[k] {
                        return Err(Error::Bounds(format!(
                            "slice {lo}..{hi} of extent {} on axis {k}",
                            x.shape[k]
                        )));
                    }
                    shape.push(((hi - lo) as u64).div_ceil(st as u64));
                }
                let n = numel(&shape);
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let src: Vec<u64> = idx
                        .iter()
                        .enumerate()
                        .map(|(k, i)| start[k] as u64 + i * *step.get(k).unwrap_or(&1) as u64)
                        .collect();
                    data.push(x.at(&src));
                    crate::expr::bump(&mut idx, &shape);
                }
                one(Tensor::new(shape, dt(x), data))
            }
            "pad" => {
                let x = a(0)?;
                let (low, high) = (int_list(op, "low")?, int_list(op, "high")?);
                let fill = ins.get(1).and_then(|t| t.data.first().copied());
                let mode = op
                    .attr("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("constant");
                one(pad(x, &low, &high, mode, fill.unwrap_or(0.0), &dt(x))?)
            }
            "concat" => {
                let axis = axis_of(int_attr(op, "axis")?, a(0)?.shape.len())?;
                one(concat(ins, axis, &dt(a(0)?))?)
            }
            "gather" => {
                let (x, i) = (a(0)?, a(1)?);
                let axis = axis_of(int_attr_or(op, "axis", 0)?, x.shape.len())?;
                one(gather(x, i, axis, &dt(x))?)
            }
            "scatter" => {
                let (x, i, u) = (a(0)?, a(1)?, a(2)?);
                let axis = axis_of(int_attr_or(op, "axis", 0)?, x.shape.len())?;
                one(scatter(x, i, u, axis, &dt(x))?)
            }
            "sort" | "topk" => {
                let x = a(0)?;
                let axis = axis_of(int_attr_or(op, "axis", -1)?, x.shape.len())?;
                let desc =
                    op.name == "topk" || matches!(op.attr("descending"), Some(Value::Bool(true)));
                let k = if op.name == "topk" {
                    int_attr(op, "k")? as u64
                } else {
                    x.shape[axis]
                };
                if k > x.shape[axis] {
                    return Err(Error::Bounds(format!(
                        "topk k={k} on an axis of extent {}",
                        x.shape[axis]
                    )));
                }
                let (values, indices) = sort_axis(x, axis, desc, k, &dt(x))?;
                Ok(if op.name == "topk" {
                    vec![values, indices]
                } else {
                    vec![values]
                })
            }
            "einsum" => {
                let eq = op
                    .attr("equation")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("einsum has no `equation`".into()))?;
                one(einsum(eq, ins, &dt(a(0)?), self.limits.max_elems)?)
            }
            other => Err(Error::Unsupported(format!(
                "omni.tensor/{other}: not an op of this dialect at version {}",
                op.version
            ))),
        }
    }
}

/// The linear position of the `o`-th index whose coordinate on `axis` is zero.
fn position_skipping(shape: &[u64], axis: usize, o: u64) -> u64 {
    let strides = {
        let mut s = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1].max(1);
        }
        s
    };
    let mut rest = o;
    let mut pos = 0u64;
    for k in (0..shape.len()).rev() {
        if k == axis {
            continue;
        }
        let d = shape[k].max(1);
        pos += (rest % d) * strides[k];
        rest /= d;
    }
    pos
}

fn resolve_reshape(want: &[i64], n: u64) -> Res<Vec<u64>> {
    let mut shape = Vec::with_capacity(want.len());
    let mut hole = None;
    let mut known: u64 = 1;
    for (k, d) in want.iter().enumerate() {
        match *d {
            -1 if hole.is_none() => {
                hole = Some(k);
                shape.push(1);
            }
            -1 => return Err(Error::Type("reshape: more than one -1".into())),
            d if d >= 0 => {
                known = known.saturating_mul(d as u64);
                shape.push(d as u64);
            }
            d => return Err(Error::Type(format!("reshape: extent {d}"))),
        }
    }
    if let Some(k) = hole {
        if known == 0 || !n.is_multiple_of(known) {
            return Err(Error::Type(format!(
                "reshape: {n} elements do not divide by {known}"
            )));
        }
        shape[k] = n / known;
    } else if numel(&shape) != n {
        return Err(Error::Type(format!(
            "reshape: {shape:?} holds {} elements and the input has {n}",
            numel(&shape)
        )));
    }
    Ok(shape)
}

fn reduce(t: &Tensor, axes: &[usize], kind: &str, keep: bool, dtype: &DType) -> Res<Tensor> {
    let mut shape = Vec::with_capacity(t.shape.len());
    for (k, d) in t.shape.iter().enumerate() {
        if axes.contains(&k) {
            if keep {
                shape.push(1);
            }
        } else {
            shape.push(*d);
        }
    }
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    // The extent of the reduced region, walked for each output element.
    let red: Vec<u64> = axes.iter().map(|k| t.shape[*k]).collect();
    let count = numel(&red);
    for _ in 0..n {
        // Rebuild the full source index, leaving the reduced axes to the inner
        // walk.
        let mut full = vec![0u64; t.shape.len()];
        let mut it = idx.iter();
        for (k, f) in full.iter_mut().enumerate() {
            if axes.contains(&k) {
                if keep {
                    it.next();
                }
            } else {
                *f = *it.next().unwrap_or(&0);
            }
        }
        let mut acc = match kind {
            "max" => f64::NEG_INFINITY,
            "min" => f64::INFINITY,
            "prod" => 1.0,
            _ => 0.0,
        };
        let mut inner = vec![0u64; red.len()];
        for _ in 0..count {
            for (k, i) in axes.iter().zip(&inner) {
                full[*k] = *i;
            }
            let v = t.at(&full);
            acc = match kind {
                "sum" | "mean" => acc + v,
                "prod" => acc * v,
                "max" => acc.max(v),
                "min" => acc.min(v),
                "any" => f64::from(u8::from(acc != 0.0 || v != 0.0)),
                "all" => f64::from(u8::from(acc != 0.0 && v != 0.0)),
                other => {
                    return Err(Error::Unsupported(format!(
                        "reduce kind `{other}`: this build does sum, mean, prod, max, \
                         min, any and all"
                    )))
                }
            };
            if red.is_empty() {
                break;
            }
            crate::expr::bump(&mut inner, &red);
        }
        if kind == "all" && count == 0 {
            acc = 1.0;
        }
        data.push(if kind == "mean" && count > 0 {
            acc / count as f64
        } else {
            acc
        });
        if shape.is_empty() {
            break;
        }
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, dtype.clone(), data))
}

fn softmax(t: &Tensor, axis: usize, dtype: &DType) -> Tensor {
    let mut out = t.clone();
    out.dtype = dtype.clone();
    let strides = t.strides();
    let (stride, len) = (strides[axis], t.shape[axis]);
    let outer = numel(&t.shape) / len.max(1);
    for o in 0..outer {
        let base = position_skipping(&t.shape, axis, o);
        // Subtract the maximum first: the same guard every real implementation
        // uses, and without it a logit of 800 is an infinity.
        let mut m = f64::NEG_INFINITY;
        for i in 0..len {
            m = m.max(out.data[(base + i * stride) as usize]);
        }
        let mut sum = 0.0;
        for i in 0..len {
            let p = (base + i * stride) as usize;
            let e = (out.data[p] - m).exp();
            out.data[p] = e;
            sum += e;
        }
        for i in 0..len {
            out.data[(base + i * stride) as usize] /= sum;
        }
    }
    out
}

fn pad(t: &Tensor, low: &[i64], high: &[i64], mode: &str, fill: f64, dtype: &DType) -> Res<Tensor> {
    if low.len() != t.shape.len() || high.len() != t.shape.len() {
        return Err(Error::Type(format!(
            "pad: {} low and {} high for a rank-{} tensor",
            low.len(),
            high.len(),
            t.shape.len()
        )));
    }
    let mut shape = Vec::with_capacity(t.shape.len());
    for k in 0..t.shape.len() {
        let d = t.shape[k] as i64 + low[k] + high[k];
        if d < 0 {
            return Err(Error::Bounds(format!("pad: axis {k} ends up {d} wide")));
        }
        shape.push(d as u64);
    }
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        let mut src = Vec::with_capacity(idx.len());
        let mut outside = false;
        for (k, i) in idx.iter().enumerate() {
            let d = t.shape[k] as i64;
            let mut j = *i as i64 - low[k];
            if j < 0 || j >= d {
                match mode {
                    "constant" => outside = true,
                    "edge" => j = j.clamp(0, d - 1),
                    "reflect" => {
                        j = if j < 0 { -j } else { 2 * (d - 1) - j };
                        j = j.clamp(0, d - 1);
                    }
                    "wrap" => j = j.rem_euclid(d.max(1)),
                    other => {
                        return Err(Error::Unsupported(format!(
                            "pad mode `{other}`: this build does constant, edge, \
                             reflect and wrap"
                        )))
                    }
                }
            }
            src.push(j.max(0) as u64);
        }
        data.push(if outside { fill } else { t.at(&src) });
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, dtype.clone(), data))
}

fn concat(parts: &[Tensor], axis: usize, dtype: &DType) -> Res<Tensor> {
    let first = parts
        .first()
        .ok_or_else(|| Error::Type("concat has no operands".into()))?;
    let mut shape = first.shape.clone();
    let mut total = 0u64;
    for p in parts {
        if p.shape.len() != shape.len() {
            return Err(Error::Type("concat: mismatched ranks".into()));
        }
        for (k, (d, e)) in p.shape.iter().zip(&shape).enumerate() {
            if k != axis && d != e {
                return Err(Error::Type(format!(
                    "concat: {:?} and {:?} differ on axis {k}",
                    p.shape, shape
                )));
            }
        }
        total += p.shape[axis];
    }
    shape[axis] = total;
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        let mut at = idx[axis];
        let mut chosen = None;
        for p in parts {
            if at < p.shape[axis] {
                let mut src = idx.clone();
                src[axis] = at;
                chosen = Some(p.at(&src));
                break;
            }
            at -= p.shape[axis];
        }
        data.push(chosen.ok_or_else(|| Error::Bounds("concat index".into()))?);
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, dtype.clone(), data))
}

fn gather(t: &Tensor, i: &Tensor, axis: usize, dtype: &DType) -> Res<Tensor> {
    // The index tensor's own shape replaces `axis`, which is what makes an
    // embedding lookup `gather(table, tokens, axis=0)`.
    let mut shape: Vec<u64> = t.shape[..axis].to_vec();
    shape.extend_from_slice(&i.shape);
    shape.extend_from_slice(&t.shape[axis + 1..]);
    let n = numel(&shape);
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; shape.len()];
    let irank = i.shape.len();
    for _ in 0..n {
        let pick = i.at(&idx[axis..axis + irank]);
        if pick < 0.0 || pick as u64 >= t.shape[axis] {
            return Err(Error::Bounds(format!(
                "gather: index {pick} on an axis of extent {}",
                t.shape[axis]
            )));
        }
        let mut src: Vec<u64> = idx[..axis].to_vec();
        src.push(pick as u64);
        src.extend_from_slice(&idx[axis + irank..]);
        data.push(t.at(&src));
        crate::expr::bump(&mut idx, &shape);
    }
    Ok(Tensor::new(shape, dtype.clone(), data))
}

fn scatter(t: &Tensor, i: &Tensor, u: &Tensor, axis: usize, dtype: &DType) -> Res<Tensor> {
    let mut out = t.clone();
    out.dtype = dtype.clone();
    if i.shape != u.shape {
        return Err(Error::Type(format!(
            "scatter: indices are {:?} and updates are {:?}",
            i.shape, u.shape
        )));
    }
    // Element-for-element: index `k` of the updates goes to the same position
    // with `axis` replaced by the index.
    let mut idx = vec![0u64; i.shape.len()];
    for _ in 0..numel(&i.shape) {
        let pick = i.at(&idx);
        if pick < 0.0 || pick as u64 >= t.shape[axis] {
            return Err(Error::Bounds(format!(
                "scatter: index {pick} on an axis of extent {}",
                t.shape[axis]
            )));
        }
        let mut dst = idx.clone();
        if dst.len() != t.shape.len() {
            return Err(Error::Type(format!(
                "scatter: rank-{} indices into a rank-{} tensor",
                dst.len(),
                t.shape.len()
            )));
        }
        dst[axis] = pick as u64;
        let strides = out.strides();
        let lin: u64 = dst.iter().zip(&strides).map(|(a, b)| a * b).sum();
        out.data[lin as usize] = u.at(&idx);
        crate::expr::bump(&mut idx, &i.shape);
    }
    Ok(out)
}

/// Sorts along one axis, returning values and the indices they came from.
/// `topk` is this with `k` less than the extent.
fn sort_axis(t: &Tensor, axis: usize, desc: bool, k: u64, dtype: &DType) -> Res<(Tensor, Tensor)> {
    let mut shape = t.shape.clone();
    shape[axis] = k;
    let n = numel(&shape);
    let mut values = Vec::with_capacity(n as usize);
    let mut indices = Vec::with_capacity(n as usize);
    let len = t.shape[axis];
    let outer = numel(&t.shape) / len.max(1);
    let strides = t.strides();
    let stride = strides[axis];
    let mut columns: Vec<Vec<(f64, u64)>> = Vec::with_capacity(outer as usize);
    for o in 0..outer {
        let base = position_skipping(&t.shape, axis, o);
        let mut col: Vec<(f64, u64)> = (0..len)
            .map(|i| (t.data[(base + i * stride) as usize], i))
            .collect();
        // Stable, and ties broken by the original position — so the answer is a
        // function of the input and not of the sort's internals.
        col.sort_by(|a, b| {
            let ord = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
            if desc {
                ord.reverse().then(a.1.cmp(&b.1))
            } else {
                ord.then(a.1.cmp(&b.1))
            }
        });
        columns.push(col);
    }
    // Walk the output in row-major order, reading from the sorted columns.
    let mut idx = vec![0u64; shape.len()];
    for _ in 0..n {
        let mut o = 0u64;
        let mut mul = 1u64;
        for kk in (0..shape.len()).rev() {
            if kk == axis {
                continue;
            }
            o += idx[kk] * mul;
            mul *= shape[kk].max(1);
        }
        let (v, i) = columns[o as usize][idx[axis] as usize];
        values.push(v);
        indices.push(i as f64);
        crate::expr::bump(&mut idx, &shape);
    }
    Ok((
        Tensor::new(shape.clone(), dtype.clone(), values),
        Tensor::new(
            shape,
            DType::Int {
                w: 32,
                signed: true,
            },
            indices,
        ),
    ))
}

/// `einsum` over explicit subscripts, e.g. `bhqd,bhkd->bhqk`.
///
/// Ellipsis is refused rather than approximated: `...` means "whatever batch
/// dimensions there are", and a graph that relies on it is relying on a
/// broadcasting rule this build does not implement.
fn einsum(eq: &str, ins: &[Tensor], dtype: &DType, max_elems: u64) -> Res<Tensor> {
    if eq.contains("...") {
        return Err(Error::Unsupported(format!(
            "einsum `{eq}`: an ellipsis stands for batch dimensions this build does \
             not infer; write them out"
        )));
    }
    let (lhs, rhs) = eq.split_once("->").ok_or_else(|| {
        Error::Unsupported(format!(
            "einsum `{eq}`: implicit output subscripts are a convention, not a \
             specification; write the `->`"
        ))
    })?;
    let terms: Vec<&str> = lhs.split(',').map(str::trim).collect();
    let out_sub: Vec<char> = rhs.trim().chars().collect();
    if terms.len() != ins.len() {
        return Err(Error::Type(format!(
            "einsum `{eq}`: {} term(s) for {} operand(s)",
            terms.len(),
            ins.len()
        )));
    }
    // Every letter's extent, checked for agreement across the operands.
    let mut extent: Vec<(char, u64)> = Vec::new();
    for (term, t) in terms.iter().zip(ins) {
        let letters: Vec<char> = term.chars().collect();
        if letters.len() != t.shape.len() {
            return Err(Error::Type(format!(
                "einsum `{eq}`: `{term}` for a rank-{} operand",
                t.shape.len()
            )));
        }
        for (c, d) in letters.iter().zip(&t.shape) {
            match extent.iter().find(|(x, _)| x == c) {
                Some((_, e)) if e != d => {
                    return Err(Error::Type(format!("einsum `{eq}`: `{c}` is {e} and {d}")))
                }
                Some(_) => {}
                None => extent.push((*c, *d)),
            }
        }
    }
    let ext = |c: char| -> u64 {
        extent
            .iter()
            .find(|(x, _)| *x == c)
            .map(|(_, d)| *d)
            .unwrap_or(1)
    };
    for c in &out_sub {
        if !extent.iter().any(|(x, _)| x == c) {
            return Err(Error::Type(format!(
                "einsum `{eq}`: the output names `{c}`, which no operand has"
            )));
        }
    }
    // The letters not in the output are summed over.
    let summed: Vec<char> = extent
        .iter()
        .map(|(c, _)| *c)
        .filter(|c| !out_sub.contains(c))
        .collect();
    let out_shape: Vec<u64> = out_sub.iter().map(|c| ext(*c)).collect();
    if numel(&out_shape) > max_elems {
        return Err(Error::Budget(format!(
            "einsum `{eq}` would produce {} elements",
            numel(&out_shape)
        )));
    }
    let sum_shape: Vec<u64> = summed.iter().map(|c| ext(*c)).collect();
    let mut data = Vec::with_capacity(numel(&out_shape) as usize);
    let mut oidx = vec![0u64; out_shape.len()];
    let n = numel(&out_shape);
    for _ in 0..n {
        let mut acc = 0.0;
        let mut sidx = vec![0u64; sum_shape.len()];
        let count = numel(&sum_shape);
        for _ in 0..count {
            let mut term = 1.0;
            for (sub, t) in terms.iter().zip(ins) {
                let src: Vec<u64> = sub
                    .chars()
                    .map(|c| match out_sub.iter().position(|x| *x == c) {
                        Some(k) => oidx[k],
                        None => summed
                            .iter()
                            .position(|x| *x == c)
                            .map(|k| sidx[k])
                            .unwrap_or(0),
                    })
                    .collect();
                term *= t.at(&src);
            }
            acc += term;
            if sum_shape.is_empty() {
                break;
            }
            crate::expr::bump(&mut sidx, &sum_shape);
        }
        data.push(acc);
        if out_shape.is_empty() {
            break;
        }
        crate::expr::bump(&mut oidx, &out_shape);
    }
    Ok(Tensor::new(out_shape, dtype.clone(), data))
}

// -------------------------------------------------------------- omni.quant --

impl State<'_> {
    /// §05's quantization ops, in their IR form.
    ///
    /// The expression form of §04.7.2 carries scale and zero as
    /// sub-*expressions* inside the scheme; the IR form carries them as the op's
    /// *inputs*, because a graph's operands are how a graph names values. So the
    /// scheme here supplies the structure — formula, block, axis, output dtype —
    /// and the inputs supply the numbers. A scheme that also names scale or zero
    /// expressions is a second answer to the same question and is refused rather
    /// than silently preferred.
    fn quant_op(&mut self, op: &Op, ins: &[Tensor], out: Option<DType>) -> Res<Vec<Tensor>> {
        use crate::quant::{Formula, Scheme};
        let sv = op
            .attr("scheme")
            .ok_or_else(|| Error::Type(format!("{} has no `scheme`", op.qualified())))?;
        // Before parsing: a scheme carrying the scale as an expression is the
        // *other* form of §05, and reporting it as a malformed expression would
        // name the wrong problem.
        for key in ["scale", "zero", "book", "order"] {
            if sv.get(key).is_some() {
                return Err(Error::Unsupported(format!(
                    "{}: the scheme names `{key}` as an expression and the IR form \
                     takes it as an operand. Two sources for one value is not \
                     something to resolve by preference",
                    op.qualified()
                )));
            }
        }
        let s = Scheme::from_value(sv).map_err(|e| Error::Type(e.to_string()))?;
        if matches!(
            s.formula,
            Formula::Codebook | Formula::CodebookRaw | Formula::Nested
        ) {
            return Err(Error::Unsupported(format!(
                "{}: the `{}` formula needs a codebook or a second level of scales, \
                 which reach outside the graph; the expression form of §04.7.2 is \
                 what resolves those",
                op.qualified(),
                s.formula.id()
            )));
        }
        let x = ins
            .first()
            .ok_or_else(|| Error::Type(format!("{} has no operand", op.qualified())))?;
        let scale = ins.get(1);
        let zero = ins.get(2);
        if s.formula == Formula::AffineSub && zero.is_none() && op.name == "dequantize" {
            return Err(Error::Type(format!(
                "{}: the `affine-sub` formula subtracts a zero point and none was \
                 given as an operand",
                op.qualified()
            )));
        }

        let block = s.block_shape(&x.shape);
        let grid = s.grid_shape(&x.shape);
        let out_dtype = out
            .or_else(|| s.out.clone())
            .unwrap_or_else(|| x.dtype.clone());

        // The per-block lookup: an element's block index, then that block's scale.
        let pick = |t: Option<&Tensor>, g: &[u64], default: f64| -> f64 {
            match t {
                None => default,
                Some(t) if t.numel() == 1 => t.data[0],
                Some(t) if t.shape.len() == g.len() => {
                    let idx: Vec<u64> = t
                        .shape
                        .iter()
                        .zip(g)
                        .map(|(d, i)| if *d == 1 { 0 } else { *i })
                        .collect();
                    t.get(&idx).unwrap_or(default)
                }
                Some(t) => {
                    // A flat scale array, read in row-major block order: §05's own
                    // allowance for publishers who store `[rows * groups]`.
                    let mut lin = 0u64;
                    let mut mul = 1u64;
                    for k in (0..g.len()).rev() {
                        lin += g[k] * mul;
                        mul *= grid[k].max(1);
                    }
                    t.data.get(lin as usize).copied().unwrap_or(default)
                }
            }
        };

        let n = x.numel();
        let mut data = Vec::with_capacity(n as usize);
        let mut idx = vec![0u64; x.shape.len()];
        for i in 0..n {
            let g: Vec<u64> = idx
                .iter()
                .zip(&block)
                .map(|(i, b)| i / (*b).max(1))
                .collect();
            let sc = pick(scale, &g, 1.0);
            let z = pick(zero, &g, 0.0);
            let v = x.data[i as usize];
            data.push(match (op.name.as_str(), s.formula) {
                ("dequantize" | "qmatmul", Formula::Sym) => v * sc,
                ("dequantize" | "qmatmul", Formula::AffineSub) => (v - z) * sc,
                ("dequantize" | "qmatmul", Formula::AffineAdd) => v * sc + z,
                ("quantize", Formula::Sym) => v / sc,
                ("quantize", Formula::AffineSub) => v / sc + z,
                ("quantize", Formula::AffineAdd) => (v - z) / sc,
                // A round trip: the values a quantized weight would dequantize
                // to, which is what makes `fake_quant` a training op rather than
                // a storage one.
                ("fake_quant", Formula::Sym) => round_scalar(v / sc, &out_dtype, &s) * sc,
                ("fake_quant", Formula::AffineSub) => {
                    (round_scalar(v / sc + z, &out_dtype, &s) - z) * sc
                }
                ("fake_quant", Formula::AffineAdd) => {
                    round_scalar((v - z) / sc, &out_dtype, &s) * sc + z
                }
                (other, f) => {
                    return Err(Error::Unsupported(format!(
                        "omni.quant/{other} with formula `{}`",
                        f.id()
                    )))
                }
            });
            crate::expr::bump(&mut idx, &x.shape);
        }
        let mut t = Tensor::new(x.shape.clone(), out_dtype.clone(), data);
        if op.name == "quantize" {
            // Quantizing lands on integers, and `clip` is where the scheme says
            // which ones; without it the integer dtype's own range applies.
            for v in &mut t.data {
                *v = round_scalar(*v, &out_dtype, &s);
            }
        } else {
            t = round_through(&t, &out_dtype, Round::Rne)?;
        }
        if op.name == "qmatmul" {
            // Defined as dequantize-then-matmul: the operands are integers and
            // the product is not, so doing it any other way would be a different
            // op wearing the same name.
            let b = ins
                .get(1)
                .ok_or_else(|| Error::Type("qmatmul has no second operand".into()))?;
            if t.shape.len() < 2 || b.shape.len() < 2 {
                return Err(Error::Type("qmatmul: operands must be rank >= 2".into()));
            }
            let mut shape =
                broadcast(&t.shape[..t.shape.len() - 2], &b.shape[..b.shape.len() - 2])?;
            shape.push(t.shape[t.shape.len() - 2]);
            shape.push(b.shape[b.shape.len() - 1]);
            self.check_size(&shape)?;
            let p = crate::expr::matmul(&t, b, Sum::Sequential, &shape, &out_dtype)
                .map_err(|e| Error::Type(e.to_string()))?;
            return Ok(vec![p]);
        }
        Ok(vec![t])
    }
}

/// Rounds a quantized value onto the integer grid, honouring the scheme's `clip`.
fn round_scalar(x: f64, dtype: &DType, s: &crate::quant::Scheme) -> f64 {
    let r = x.round_ties_even();
    match s.clip {
        Some((lo, hi)) => r.clamp(lo, hi),
        None => match dtype {
            DType::Int { w, signed: true } if *w < 64 => {
                let hi = ((1i64 << (w - 1)) - 1) as f64;
                r.clamp(-hi - 1.0, hi)
            }
            DType::Int { w, signed: false } if *w < 64 => r.clamp(0.0, ((1u64 << w) - 1) as f64),
            _ => r,
        },
    }
}

// ----------------------------------------------------------------- omni.nn --

impl State<'_> {
    fn nn_op(&mut self, op: &Op, ins: &[Tensor], out: Option<DType>) -> Res<Vec<Tensor>> {
        let a = |k: usize| -> Res<&Tensor> {
            ins.get(k)
                .ok_or_else(|| Error::Type(format!("{} has no operand {k}", op.qualified())))
        };
        let dt = |t: &Tensor| out.clone().unwrap_or_else(|| t.dtype.clone());
        match op.name.as_str() {
            "embedding" => {
                // (tokens, table) → rows, which is the order §07.5's synthesizer
                // emits and the order the op's own shape function declares.
                let (tokens, table) = (a(0)?, a(1)?);
                if table.shape.len() != 2 {
                    return Err(Error::Type(format!(
                        "embedding: the table is {:?} and must be [vocab, hidden]",
                        table.shape
                    )));
                }
                let mut shape = tokens.shape.clone();
                shape.push(table.shape[1]);
                self.check_size(&shape)?;
                Ok(vec![gather(table, tokens, 0, &dt(table))?])
            }
            "norm" => {
                let x = a(0)?;
                let kind = op
                    .attr("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("norm has no `kind`".into()))?;
                let eps = float_attr(op, "eps").unwrap_or(1e-5);
                let axis = axis_of(int_attr_or(op, "axis", -1)?, x.shape.len())?;
                Ok(vec![norm(
                    x,
                    kind,
                    eps,
                    axis,
                    ins.get(1),
                    ins.get(2),
                    &dt(x),
                )?])
            }
            "activation" => {
                let x = a(0)?;
                let kind = op
                    .attr("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("activation has no `kind`".into()))?;
                Ok(vec![activation(x, kind, ins.get(1), &dt(x))?])
            }
            "rope" => {
                let x = a(0)?;
                let theta = float_attr(op, "theta").unwrap_or(10000.0);
                let interleaved = matches!(op.attr("interleaved"), Some(Value::Bool(true)));
                Ok(vec![rope(x, theta, interleaved, ins.get(1), &dt(x))?])
            }
            "attention" => {
                let (q, k, v) = (a(0)?, a(1)?, a(2)?);
                let d = dt(q);
                Ok(vec![self.attention(op, q, k, v, ins.get(3), &d)?])
            }
            "conv" => {
                let (x, w) = (a(0)?, a(1)?);
                let rank = x.shape.len().saturating_sub(2);
                let spatial = |key: &str, default: i64| -> Res<Vec<u64>> {
                    match op.attr(key) {
                        Some(_) => int_list(op, key)?
                            .into_iter()
                            .map(|v| {
                                u64::try_from(v).map_err(|_| {
                                    Error::Type(format!("conv: `{key}` has a negative entry"))
                                })
                            })
                            .collect(),
                        None => Ok(vec![default as u64; rank]),
                    }
                };
                let d = dt(x);
                Ok(vec![self.conv(
                    x,
                    w,
                    ins.get(2),
                    &spatial("stride", 1)?,
                    &spatial("padding", 0)?,
                    &spatial("dilation", 1)?,
                    int_attr_or(op, "groups", 1)?.max(1) as u64,
                    false,
                    &d,
                )?])
            }
            "conv1d_causal" => {
                // Causal means the padding is entirely on the left, so output
                // position t sees inputs up to t and never past it. Stride and
                // dilation are not attributes of this op: it exists to be the
                // one whose padding cannot be got wrong.
                let (x, w) = (a(0)?, a(1)?);
                if x.shape.len() != 3 || w.shape.len() != 3 {
                    return Err(Error::Type(format!(
                        "conv1d_causal: x is {:?} and w is {:?}; both are \
                         [batch, channels, length]",
                        x.shape, w.shape
                    )));
                }
                let d = dt(x);
                Ok(vec![self.conv(
                    x,
                    w,
                    ins.get(2),
                    &[1],
                    &[w.shape[2].saturating_sub(1)],
                    &[1],
                    int_attr_or(op, "groups", 1)?.max(1) as u64,
                    true,
                    &d,
                )?])
            }
            "pool" => {
                let x = a(0)?;
                let kind = op
                    .attr("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("pool has no `kind`".into()))?
                    .to_string();
                let window: Vec<u64> = int_list(op, "window")?
                    .into_iter()
                    .map(|v| v.max(1) as u64)
                    .collect();
                let stride: Vec<u64> = match op.attr("stride") {
                    Some(_) => int_list(op, "stride")?
                        .into_iter()
                        .map(|v| v.max(1) as u64)
                        .collect(),
                    // Non-overlapping by default, which is what every framework
                    // does and what "no stride given" has to mean for the window
                    // to partition the input.
                    None => window.clone(),
                };
                let d = dt(x);
                Ok(vec![self.pool(x, &kind, &window, &stride, &d)?])
            }
            "interpolate" => {
                let x = a(0)?;
                let mode = op
                    .attr("mode")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Type("interpolate has no `mode`".into()))?
                    .to_string();
                let d = dt(x);
                Ok(vec![self.interpolate(x, &mode, op.attr("scale"), &d)?])
            }
            "moe_route" => {
                let (x, w) = (a(0)?, a(1)?);
                let k = int_attr(op, "top_k")?;
                if k <= 0 {
                    return Err(Error::Type("moe_route: `top_k` must be positive".into()));
                }
                let normalize = matches!(op.attr("normalize"), Some(Value::Bool(true)));
                let d = dt(x);
                Ok(self.moe_route(x, w, k as u64, normalize, &d)?)
            }
            // `ssm_scan` is not refused for being hard. It is refused because
            // §07 names it — "(associative scan)" — without saying what it
            // computes: which operand is the state transition and which the
            // input projection, whether the timestep is an operand or folded
            // into `A`, and whether the discretization is zero-order hold or
            // bilinear. Those choices produce different numbers from the same
            // tensors. Every other op in this dialect either has a shape
            // function pinning its operands or a definition that is standard
            // across every framework; this one has neither, so implementing it
            // would mean inventing the semantics and then checking my own
            // invention. See `docs/spec/07-graph.md` §7.8.
            "ssm_scan" => Err(Error::Unsupported(
                "omni.nn/ssm_scan: §07 names this op but does not define it. The \
                 operand order, whether the timestep is an operand, and the \
                 discretization rule are all unstated, and different readings \
                 give different numbers — so this is a gap in the specification \
                 rather than a gap in this build, and filling it here would be \
                 inventing a semantics and then agreeing with myself"
                    .into(),
            )),
            other => Err(Error::Unsupported(format!(
                "omni.nn/{other}: not an op of this dialect at version {}",
                op.version
            ))),
        }
    }

    /// N-dimensional convolution, and the causal 1-D case that shares it.
    ///
    /// `x` is `[batch, in_channels, spatial…]`, `w` is
    /// `[out_channels, in_channels / groups, kernel…]`, and the optional third
    /// operand is a per-output-channel bias. This is the cross-correlation every
    /// framework calls convolution — the kernel is not flipped — which is worth
    /// saying because the mathematical convolution *does* flip it.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: &[u64],
        padding: &[u64],
        dilation: &[u64],
        groups: u64,
        causal: bool,
        dtype: &DType,
    ) -> Res<Tensor> {
        if x.shape.len() < 3 || w.shape.len() != x.shape.len() {
            return Err(Error::Type(format!(
                "conv: x is {:?} and w is {:?}; both are [n, c, spatial…]",
                x.shape, w.shape
            )));
        }
        let sp = x.shape.len() - 2;
        for (what, v) in [
            ("stride", stride),
            ("padding", padding),
            ("dilation", dilation),
        ] {
            if v.len() != sp {
                return Err(Error::Type(format!(
                    "conv: `{what}` has {} entries for {sp} spatial dimension(s)",
                    v.len()
                )));
            }
        }
        let (n, cin, cout) = (x.shape[0], x.shape[1], w.shape[0]);
        if groups == 0 || cin % groups != 0 || cout % groups != 0 {
            return Err(Error::Type(format!(
                "conv: {cin} in and {cout} out channels do not divide into {groups} group(s)"
            )));
        }
        if w.shape[1] != cin / groups {
            return Err(Error::Type(format!(
                "conv: the kernel takes {} input channel(s) and each group has {}",
                w.shape[1],
                cin / groups
            )));
        }
        let kernel: Vec<u64> = w.shape[2..].to_vec();
        let mut out_sp = Vec::with_capacity(sp);
        for k in 0..sp {
            // Causal padding is all on the left, so the output keeps the input's
            // length; the symmetric case pads both sides.
            let padded = x.shape[2 + k] + if causal { padding[k] } else { 2 * padding[k] };
            let span = dilation[k] * (kernel[k] - 1) + 1;
            if padded < span {
                return Err(Error::Bounds(format!(
                    "conv: a {span}-wide kernel does not fit {padded} on axis {k}"
                )));
            }
            out_sp.push((padded - span) / stride[k] + 1);
        }
        let mut shape = vec![n, cout];
        shape.extend_from_slice(&out_sp);
        self.check_size(&shape)?;
        // One unit of fuel per output element: a convolution is where a graph
        // can ask for arbitrarily much work with very few ops.
        self.spend(numel(&shape) / 64 + 1)?;

        let per_group_out = cout / groups;
        let per_group_in = cin / groups;
        let mut data = Vec::with_capacity(numel(&shape) as usize);
        let mut idx = vec![0u64; shape.len()];
        for _ in 0..numel(&shape) {
            let (b, oc) = (idx[0], idx[1]);
            let g = oc / per_group_out;
            let mut acc = 0.0;
            let mut kidx = vec![0u64; sp];
            let kn = numel(&kernel);
            for _ in 0..kn {
                // The input position this kernel tap reads, in each spatial axis.
                let mut src = Vec::with_capacity(x.shape.len());
                let mut inside = true;
                for k in 0..sp {
                    let p =
                        (idx[2 + k] * stride[k] + kidx[k] * dilation[k]) as i64 - padding[k] as i64;
                    if p < 0 || p as u64 >= x.shape[2 + k] {
                        inside = false;
                        break;
                    }
                    src.push(p as u64);
                }
                if inside {
                    for ic in 0..per_group_in {
                        let mut xi = vec![b, g * per_group_in + ic];
                        xi.extend_from_slice(&src);
                        let mut wi = vec![oc, ic];
                        wi.extend_from_slice(&kidx);
                        acc += x.at(&xi) * w.at(&wi);
                    }
                }
                if kernel.is_empty() {
                    break;
                }
                crate::expr::bump(&mut kidx, &kernel);
            }
            if let Some(bs) = bias {
                acc += bs.data.get(oc as usize).copied().unwrap_or(0.0);
            }
            data.push(acc);
            crate::expr::bump(&mut idx, &shape);
        }
        round_through(&Tensor::new(shape, dtype.clone(), data), dtype, Round::Rne)
    }

    /// `max` or `avg` pooling over the spatial axes of `[n, c, spatial…]`.
    fn pool(
        &mut self,
        x: &Tensor,
        kind: &str,
        window: &[u64],
        stride: &[u64],
        dtype: &DType,
    ) -> Res<Tensor> {
        if x.shape.len() < 3 {
            return Err(Error::Type(format!(
                "pool: {:?} is not [n, c, spatial…]",
                x.shape
            )));
        }
        let sp = x.shape.len() - 2;
        if window.len() != sp || stride.len() != sp {
            return Err(Error::Type(format!(
                "pool: a {}-wide window and {} stride(s) for {sp} spatial dimension(s)",
                window.len(),
                stride.len()
            )));
        }
        if !matches!(kind, "max" | "avg") {
            return Err(Error::Unsupported(format!(
                "pool kind `{kind}`: this build does `max` and `avg`"
            )));
        }
        let mut shape = vec![x.shape[0], x.shape[1]];
        for k in 0..sp {
            if x.shape[2 + k] < window[k] {
                return Err(Error::Bounds(format!(
                    "pool: a {}-wide window on an axis of extent {}",
                    window[k],
                    x.shape[2 + k]
                )));
            }
            shape.push((x.shape[2 + k] - window[k]) / stride[k] + 1);
        }
        self.check_size(&shape)?;
        let mut data = Vec::with_capacity(numel(&shape) as usize);
        let mut idx = vec![0u64; shape.len()];
        let wn = numel(window);
        for _ in 0..numel(&shape) {
            let mut acc = if kind == "max" {
                f64::NEG_INFINITY
            } else {
                0.0
            };
            let mut widx = vec![0u64; sp];
            for _ in 0..wn {
                let mut src = vec![idx[0], idx[1]];
                for k in 0..sp {
                    src.push(idx[2 + k] * stride[k] + widx[k]);
                }
                let v = x.at(&src);
                acc = if kind == "max" { acc.max(v) } else { acc + v };
                if window.is_empty() {
                    break;
                }
                crate::expr::bump(&mut widx, window);
            }
            data.push(if kind == "avg" { acc / wn as f64 } else { acc });
            crate::expr::bump(&mut idx, &shape);
        }
        round_through(&Tensor::new(shape, dtype.clone(), data), dtype, Round::Rne)
    }

    /// Resampling over the spatial axes of `[n, c, spatial…]`.
    ///
    /// `nearest` and `linear` only. The half-pixel convention is the one every
    /// framework's `align_corners=False` uses, and it is written out below
    /// because the other convention shifts every output by half a sample and
    /// both are called "linear".
    fn interpolate(
        &mut self,
        x: &Tensor,
        mode: &str,
        scale: Option<&Value>,
        dtype: &DType,
    ) -> Res<Tensor> {
        if x.shape.len() < 3 {
            return Err(Error::Type(format!(
                "interpolate: {:?} is not [n, c, spatial…]",
                x.shape
            )));
        }
        let sp = x.shape.len() - 2;
        let factors: Vec<f64> = match scale {
            Some(Value::Array(xs)) => xs
                .iter()
                .map(|v| match v {
                    Value::F64(f) => Ok(*f),
                    Value::U(n) => Ok(*n as f64),
                    Value::I(n) => Ok(*n as f64),
                    _ => Err(Error::Type("interpolate: a non-numeric `scale`".into())),
                })
                .collect::<Res<Vec<f64>>>()?,
            Some(Value::F64(f)) => vec![*f; sp],
            Some(Value::U(n)) => vec![*n as f64; sp],
            Some(Value::I(n)) => vec![*n as f64; sp],
            _ => return Err(Error::Type("interpolate has no `scale`".into())),
        };
        if factors.len() != sp {
            return Err(Error::Type(format!(
                "interpolate: {} scale(s) for {sp} spatial dimension(s)",
                factors.len()
            )));
        }
        if !matches!(mode, "nearest" | "linear") {
            return Err(Error::Unsupported(format!(
                "interpolate mode `{mode}`: this build does `nearest` and `linear`"
            )));
        }
        let mut shape = vec![x.shape[0], x.shape[1]];
        for (k, f) in factors.iter().enumerate() {
            if *f <= 0.0 {
                return Err(Error::Type(format!("interpolate: scale {f} on axis {k}")));
            }
            shape.push(((x.shape[2 + k] as f64) * f).floor().max(1.0) as u64);
        }
        self.check_size(&shape)?;
        let mut data = Vec::with_capacity(numel(&shape) as usize);
        let mut idx = vec![0u64; shape.len()];
        for _ in 0..numel(&shape) {
            let mut src = vec![idx[0], idx[1]];
            let mut frac = Vec::with_capacity(sp);
            for k in 0..sp {
                // Half-pixel centres: output sample j maps to input coordinate
                // (j + 0.5)/scale - 0.5, which is what keeps the resampled image
                // in the same place rather than shifted by half a pixel.
                let pos = ((idx[2 + k] as f64) + 0.5) / factors[k] - 0.5;
                let lo = pos.floor().max(0.0);
                src.push((lo as u64).min(x.shape[2 + k] - 1));
                frac.push((pos - lo).clamp(0.0, 1.0));
            }
            data.push(match mode {
                "nearest" => {
                    let mut s = src.clone();
                    for k in 0..sp {
                        if frac[k] >= 0.5 {
                            s[2 + k] = (s[2 + k] + 1).min(x.shape[2 + k] - 1);
                        }
                    }
                    x.at(&s)
                }
                // Multilinear: 2^sp corners, weighted by the fractional part in
                // each axis.
                _ => {
                    let mut acc = 0.0;
                    for corner in 0..(1u32 << sp) {
                        let mut s = src.clone();
                        let mut wgt = 1.0;
                        for k in 0..sp {
                            if corner >> k & 1 == 1 {
                                s[2 + k] = (s[2 + k] + 1).min(x.shape[2 + k] - 1);
                                wgt *= frac[k];
                            } else {
                                wgt *= 1.0 - frac[k];
                            }
                        }
                        if wgt != 0.0 {
                            acc += wgt * x.at(&s);
                        }
                    }
                    acc
                }
            });
            crate::expr::bump(&mut idx, &shape);
        }
        round_through(&Tensor::new(shape, dtype.clone(), data), dtype, Round::Rne)
    }

    /// §07.8's MoE router: which experts a token goes to, and how much of each.
    ///
    /// `x` is `[…, d_model]` and `w` is the routing matrix, so the logits are
    /// `x · w` and `w` is `[d_model, experts]`. When the two possible
    /// orientations are distinguishable, the wrong one is an error naming both,
    /// rather than a transpose applied on a guess.
    fn moe_route(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        k: u64,
        normalize: bool,
        dtype: &DType,
    ) -> Res<Vec<Tensor>> {
        if w.shape.len() != 2 || x.shape.is_empty() {
            return Err(Error::Type(format!(
                "moe_route: x is {:?} and the routing matrix is {:?}, which is rank 2",
                x.shape, w.shape
            )));
        }
        let d = x.shape[x.shape.len() - 1];
        if w.shape[0] != d {
            let hint = if w.shape[1] == d {
                "; it is [experts, d_model], and this op takes [d_model, experts] \
                 — transpose it in the graph where a reader can see it"
            } else {
                ""
            };
            return Err(Error::Type(format!(
                "moe_route: x has {d} feature(s) and the routing matrix is {:?}{hint}",
                w.shape
            )));
        }
        let experts = w.shape[1];
        if k > experts {
            return Err(Error::Bounds(format!(
                "moe_route: top_k {k} of {experts} expert(s)"
            )));
        }
        let rows = x.numel() / d.max(1);
        let mut shape = x.shape[..x.shape.len() - 1].to_vec();
        shape.push(k);
        self.check_size(&shape)?;
        let mut weights = Vec::with_capacity((rows * k) as usize);
        let mut indices = Vec::with_capacity((rows * k) as usize);
        for r in 0..rows {
            // Softmax over every expert first, then take the top k. Doing it the
            // other way — softmax over only the chosen k — is a different
            // routing, and `normalize` is what asks for that explicitly.
            let mut logits: Vec<f64> = (0..experts)
                .map(|e| {
                    (0..d)
                        .map(|i| x.data[(r * d + i) as usize] * w.data[(i * experts + e) as usize])
                        .sum()
                })
                .collect();
            let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for l in logits.iter_mut() {
                *l = (*l - m).exp();
                total += *l;
            }
            for l in logits.iter_mut() {
                *l /= total;
            }
            let mut order: Vec<usize> = (0..experts as usize).collect();
            // Ties broken by expert index, so routing is a function of the input
            // and not of the sort's internals.
            order.sort_by(|a, b| {
                logits[*b]
                    .partial_cmp(&logits[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            let chosen = &order[..k as usize];
            let picked: f64 = chosen.iter().map(|e| logits[*e]).sum();
            for e in chosen {
                weights.push(if normalize && picked > 0.0 {
                    logits[*e] / picked
                } else {
                    logits[*e]
                });
                indices.push(*e as f64);
            }
        }
        Ok(vec![
            round_through(
                &Tensor::new(shape.clone(), dtype.clone(), weights),
                dtype,
                Round::Rne,
            )?,
            Tensor::new(
                shape,
                DType::Int {
                    w: 32,
                    signed: true,
                },
                indices,
            ),
        ])
    }

    /// §07's `attention`, interpreted rather than lowered.
    ///
    /// The shipped lowering (`omni.nn/attention@2→primitive`) declines `causal`,
    /// `window`, `softcap` and grouped queries, and `graph synthesize` emits
    /// `causal` and `kv_groups` — so an interpreter that could only run the
    /// lowering could not execute this implementation's own graphs.
    fn attention(
        &mut self,
        op: &Op,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        dtype: &DType,
    ) -> Res<Tensor> {
        if q.shape.len() < 2 || k.shape.len() < 2 || v.shape.len() < 2 {
            return Err(Error::Type("attention: operands must be rank >= 2".into()));
        }
        let rank = q.shape.len();
        let (sq, dh) = (q.shape[rank - 2], q.shape[rank - 1]);
        let sk = k.shape[k.shape.len() - 2];
        if k.shape[k.shape.len() - 1] != dh {
            return Err(Error::Type(format!(
                "attention: q head dim {dh} and k head dim {}",
                k.shape[k.shape.len() - 1]
            )));
        }
        let dv = v.shape[v.shape.len() - 1];
        if v.shape[v.shape.len() - 2] != sk {
            return Err(Error::Type(format!(
                "attention: k has {sk} keys and v has {}",
                v.shape[v.shape.len() - 2]
            )));
        }
        let scale = float_attr(op, "scale").unwrap_or_else(|| 1.0 / (dh as f64).sqrt());
        let causal = matches!(op.attr("causal"), Some(Value::Bool(true)));
        let softcap = float_attr(op, "softcap");
        let groups = int_attr_or(op, "kv_groups", 1)?.max(1) as u64;
        let window = op.attr("window").and_then(as_int).map(|w| w.max(0) as u64);

        // Batch dimensions are everything before the last two. Grouped-query
        // attention has several query heads sharing one kv head, so a query
        // batch index divides down to its kv head rather than broadcasting.
        let qb: u64 = q.shape[..rank - 2].iter().product::<u64>().max(1);
        let kb: u64 = k.shape[..k.shape.len() - 2].iter().product::<u64>().max(1);
        if groups > 1 && qb != kb * groups {
            return Err(Error::Type(format!(
                "attention: {qb} query head(s), {kb} kv head(s) and kv_groups \
                 {groups}, which do not agree"
            )));
        }
        let mut shape = q.shape[..rank - 2].to_vec();
        shape.push(sq);
        shape.push(dv);
        self.check_size(&shape)?;
        self.spend(qb)?;

        let head_of = |b: u64| -> u64 {
            if groups > 1 {
                b / groups
            } else {
                b % kb
            }
        };
        let mut data = Vec::with_capacity(numel(&shape) as usize);
        let mut scores = vec![0.0f64; sk as usize];
        for b in 0..qb {
            let h = head_of(b);
            for i in 0..sq {
                let mut m = f64::NEG_INFINITY;
                // Keys run ahead of queries when there is a KV cache, so a
                // causal mask compares against the *offset* position rather than
                // the query index.
                let past = sk.saturating_sub(sq);
                for (j, sc) in scores.iter_mut().enumerate() {
                    let j = j as u64;
                    let hidden_causal = causal && j > i + past;
                    let hidden_window = window.is_some_and(|w| i + past > j + w);
                    if hidden_causal || hidden_window {
                        *sc = f64::NEG_INFINITY;
                        continue;
                    }
                    let mut acc = 0.0;
                    for p in 0..dh {
                        acc += flat3(q, b, i, p, sq, dh) * flat3(k, h, j, p, sk, dh);
                    }
                    acc *= scale;
                    if let Some(c) = softcap {
                        // §07.8's soft cap: a smooth bound on the logit, not a
                        // clamp, and the two differ in the gradient.
                        acc = c * (acc / c).tanh();
                    }
                    if let Some(msk) = mask {
                        acc += mask_at(msk, b, i, j);
                    }
                    *sc = acc;
                    m = m.max(acc);
                }
                let mut total = 0.0;
                for sc in scores.iter_mut() {
                    *sc = if sc.is_finite() { (*sc - m).exp() } else { 0.0 };
                    total += *sc;
                }
                if total == 0.0 {
                    // Every key hidden. A row of zeros, not a row of NaNs: the
                    // softmax of nothing has no answer and zeros say so without
                    // poisoning everything downstream.
                    data.extend(std::iter::repeat_n(0.0, dv as usize));
                    continue;
                }
                for p in 0..dv {
                    let mut acc = 0.0;
                    for (j, sc) in scores.iter().enumerate() {
                        acc += sc * flat3(v, h, j as u64, p, sk, dv);
                    }
                    data.push(acc / total);
                }
            }
        }
        let t = Tensor::new(shape, dtype.clone(), data);
        round_through(&t, dtype, Round::Rne)
    }
}

/// Reads `t[batch, i, j]` from a tensor whose last two axes are `rows x cols`.
fn flat3(t: &Tensor, batch: u64, i: u64, j: u64, rows: u64, cols: u64) -> f64 {
    let batches = (t.numel() / (rows * cols).max(1)).max(1);
    let b = batch % batches;
    t.data[((b * rows + i) * cols + j) as usize]
}

/// An additive attention mask, broadcast over whatever axes it does not have.
fn mask_at(m: &Tensor, batch: u64, i: u64, j: u64) -> f64 {
    match m.shape.len() {
        0 => m.data.first().copied().unwrap_or(0.0),
        1 => m.data.get(j as usize).copied().unwrap_or(0.0),
        2 => m
            .get(&[i.min(m.shape[0] - 1), j.min(m.shape[1] - 1)])
            .unwrap_or(0.0),
        _ => {
            let last = m.shape.len();
            let (rows, cols) = (m.shape[last - 2], m.shape[last - 1]);
            flat3(m, batch, i.min(rows - 1), j.min(cols - 1), rows, cols)
        }
    }
}

fn norm(
    x: &Tensor,
    kind: &str,
    eps: f64,
    axis: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    dtype: &DType,
) -> Res<Tensor> {
    let mut out = x.clone();
    out.dtype = dtype.clone();
    let strides = x.strides();
    let (stride, len) = (strides[axis], x.shape[axis]);
    let outer = numel(&x.shape) / len.max(1);
    for o in 0..outer {
        let base = position_skipping(&x.shape, axis, o);
        let read = |i: u64| x.data[(base + i * stride) as usize];
        let (centre, scale) = match kind {
            // RMS does not subtract the mean. That is the whole reason the two
            // are separate kinds rather than one with a flag.
            "rms" => {
                let mut ss = 0.0;
                for i in 0..len {
                    ss += read(i) * read(i);
                }
                (0.0, 1.0 / (ss / len as f64 + eps).sqrt())
            }
            "layer" => {
                let mut sum = 0.0;
                for i in 0..len {
                    sum += read(i);
                }
                let mean = sum / len as f64;
                let mut var = 0.0;
                for i in 0..len {
                    var += (read(i) - mean) * (read(i) - mean);
                }
                (mean, 1.0 / (var / len as f64 + eps).sqrt())
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "norm kind `{other}`: this build does `rms` and `layer`"
                )))
            }
        };
        for i in 0..len {
            let p = (base + i * stride) as usize;
            let mut val = (out.data[p] - centre) * scale;
            if let Some(w) = weight {
                val *= w
                    .data
                    .get((i % w.numel().max(1)) as usize)
                    .copied()
                    .unwrap_or(1.0);
            }
            if let Some(b) = bias {
                val += b
                    .data
                    .get((i % b.numel().max(1)) as usize)
                    .copied()
                    .unwrap_or(0.0);
            }
            out.data[p] = val;
        }
    }
    round_through(&out, dtype, Round::Rne)
}

fn activation(x: &Tensor, kind: &str, gate: Option<&Tensor>, dtype: &DType) -> Res<Tensor> {
    let f: fn(f64) -> f64 = match kind {
        "relu" => |v| v.max(0.0),
        // Exact gelu, through `erf`. `gelu_tanh` is a *different function* and
        // therefore a different name rather than the same one with a flag.
        "gelu" => |v| 0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2)),
        "gelu_tanh" | "gelu_new" => |v| {
            let c = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
        },
        "silu" | "swish" => |v| v / (1.0 + (-v).exp()),
        "sigmoid" => |v| 1.0 / (1.0 + (-v).exp()),
        "tanh" => f64::tanh,
        other => {
            return Err(Error::Unsupported(format!(
                "activation kind `{other}`: this build does relu, gelu, gelu_tanh, \
                 silu, sigmoid and tanh"
            )))
        }
    };
    let applied = map_with(x, dtype, f);
    // A second operand makes it gated — SwiGLU and friends — where the
    // activation applies to one half and multiplies the other.
    match gate {
        Some(g) => zip_with(&applied, g, dtype, |p, q| p * q),
        None => Ok(applied),
    }
}

/// Rotary position embeddings (§06's `rope` params).
///
/// Both pairing conventions are implemented because both are in the wild and
/// they produce different numbers: `interleaved` rotates adjacent elements
/// `(0,1)`, `(2,3)`, …, and the half-split convention rotates `i` against
/// `i + d/2`. A runtime that picks the wrong one produces a model that generates
/// fluent nonsense, which is why §06 makes it an explicit parameter.
fn rope(
    x: &Tensor,
    theta: f64,
    interleaved: bool,
    positions: Option<&Tensor>,
    dtype: &DType,
) -> Res<Tensor> {
    let rank = x.shape.len();
    if rank < 2 {
        return Err(Error::Type("rope: needs at least [seq, head_dim]".into()));
    }
    let dh = x.shape[rank - 1];
    if !dh.is_multiple_of(2) {
        return Err(Error::Type(format!(
            "rope: an odd head dimension {dh} has no pairs to rotate"
        )));
    }
    // The head dimension is last. Position is the axis before the heads for the
    // `[.., S, H, D]` layout §07.5 emits, and the axis before the head dimension
    // when there are no separate heads.
    let seq_axis = if rank >= 3 { rank - 3 } else { rank - 2 };
    let half = dh / 2;
    let mut out = x.clone();
    out.dtype = dtype.clone();
    let strides = x.strides();
    let n = numel(&x.shape);
    let mut idx = vec![0u64; rank];
    for _ in 0..n {
        let d = idx[rank - 1];
        // Each pair is rotated once, from its lower element.
        let partner = if interleaved {
            if d % 2 == 1 {
                crate::expr::bump(&mut idx, &x.shape);
                continue;
            }
            d + 1
        } else {
            if d >= half {
                crate::expr::bump(&mut idx, &x.shape);
                continue;
            }
            d + half
        };
        let pair = if interleaved { d / 2 } else { d };
        let pos = idx[seq_axis];
        let pos = match positions {
            Some(p) => p
                .data
                .get((pos % p.numel().max(1)) as usize)
                .copied()
                .unwrap_or(pos as f64),
            None => pos as f64,
        };
        let ang = pos * theta.powf(-2.0 * pair as f64 / dh as f64);
        let (c, s) = (ang.cos(), ang.sin());
        let lin: u64 = idx.iter().zip(&strides).map(|(a, b)| a * b).sum();
        let other = lin + (partner - d) * strides[rank - 1];
        let (a, b) = (out.data[lin as usize], out.data[other as usize]);
        out.data[lin as usize] = a * c - b * s;
        out.data[other as usize] = a * s + b * c;
        crate::expr::bump(&mut idx, &x.shape);
    }
    round_through(&out, dtype, Round::Rne)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Constraint, Level, Rel};

    fn f32t(shape: &[u64], data: &[f64]) -> Tensor {
        Tensor::new(shape.to_vec(), DType::F32, data.to_vec())
    }

    fn ty(shape: &[u64]) -> Type {
        Type::tensor(shape.iter().map(|d| Dim::N(*d)).collect(), DType::F32)
    }

    /// A module of one function whose body is `ops`, taking `params`.
    fn module(params: Vec<(String, Type)>, ops: Vec<Op>) -> Module {
        let mut m = Module::new(Level::Semantic, "f");
        m.functions.push((
            "f".into(),
            Function {
                params,
                results: Vec::new(),
                attrs: Vec::new(),
                body: Region {
                    blocks: vec![Block {
                        args: Vec::new(),
                        ops,
                    }],
                },
                constraints: Vec::new(),
            },
        ));
        m
    }

    /// One op over the given arguments, returning its result.
    fn one_op(op: Op, args: &[Tensor]) -> Res<Tensor> {
        let params: Vec<(String, Type)> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                (
                    format!("a{i}"),
                    Type::tensor(
                        a.shape.iter().map(|d| Dim::N(*d)).collect(),
                        a.dtype.clone(),
                    ),
                )
            })
            .collect();
        let n = args.len() as u32;
        let inputs: Vec<u32> = (0..n).collect();
        let mut op = op;
        op.inputs = inputs;
        let last = op.outputs.first().map(|(id, _)| *id).unwrap_or(n);
        let m = module(
            params,
            vec![op, Op::new("omni.core", "return", 1).with_inputs(&[last])],
        );
        let out = run(&m, args, &(), &Limits::default())?;
        out.returned
            .into_iter()
            .next()
            .ok_or_else(|| Error::Type("nothing returned".into()))
    }

    fn tensor_op(name: &str, attrs: Vec<(&str, Value)>, args: &[Tensor], shape: &[u64]) -> Tensor {
        let mut op = Op::new("omni.tensor", name, 1).with_output(args.len() as u32, ty(shape));
        for (k, v) in attrs {
            op = op.with_attr(k, v);
        }
        one_op(op, args).unwrap_or_else(|e| panic!("omni.tensor/{name}: {e}"))
    }

    fn ints(xs: &[i64]) -> Value {
        Value::Array(xs.iter().map(|x| Value::I(*x)).collect())
    }

    // -------------------------------------------------------- the basics --

    #[test]
    fn a_graph_of_constants_and_arithmetic_runs() {
        let c = Op::new("omni.core", "constant", 1)
            .with_attr(
                "value",
                Value::Array(vec![Value::F64(1.0), Value::F64(2.0)]),
            )
            .with_output(1, ty(&[2]));
        let add = Op::new("omni.tensor", "add", 1)
            .with_inputs(&[0, 1])
            .with_output(2, ty(&[2]));
        let m = module(
            vec![("x".into(), ty(&[2]))],
            vec![c, add, Op::new("omni.core", "return", 1).with_inputs(&[2])],
        );
        let out = run(&m, &[f32t(&[2], &[10.0, 20.0])], &(), &Limits::default()).unwrap();
        assert_eq!(out.returned[0].data, vec![11.0, 22.0]);
        assert_eq!(out.ops, 3);
    }

    #[test]
    fn every_elementwise_and_shape_op_does_what_it_says() {
        let x = f32t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            tensor_op("neg", vec![], std::slice::from_ref(&x), &[2, 3]).data,
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        );
        assert_eq!(
            tensor_op(
                "transpose",
                vec![("perm", ints(&[1, 0]))],
                std::slice::from_ref(&x),
                &[3, 2]
            )
            .data,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(
            tensor_op(
                "reshape",
                vec![("shape", ints(&[3, -1]))],
                std::slice::from_ref(&x),
                &[3, 2]
            )
            .shape,
            vec![3, 2]
        );
        assert_eq!(
            tensor_op(
                "slice",
                vec![("start", ints(&[0, 1])), ("stop", ints(&[2, 3]))],
                std::slice::from_ref(&x),
                &[2, 2]
            )
            .data,
            vec![2.0, 3.0, 5.0, 6.0]
        );
        assert_eq!(
            tensor_op(
                "reduce",
                vec![("kind", Value::text("sum")), ("axes", ints(&[1]))],
                std::slice::from_ref(&x),
                &[2]
            )
            .data,
            vec![6.0, 15.0]
        );
        assert_eq!(
            tensor_op(
                "reduce",
                vec![("kind", Value::text("max")), ("axes", ints(&[0]))],
                std::slice::from_ref(&x),
                &[3]
            )
            .data,
            vec![4.0, 5.0, 6.0]
        );
        assert_eq!(
            tensor_op(
                "reduce",
                vec![("kind", Value::text("mean")), ("axes", ints(&[0, 1]))],
                std::slice::from_ref(&x),
                &[]
            )
            .data,
            vec![3.5]
        );
        assert_eq!(
            tensor_op(
                "cumsum",
                vec![("axis", Value::I(1))],
                std::slice::from_ref(&x),
                &[2, 3]
            )
            .data,
            vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            tensor_op(
                "pad",
                vec![("low", ints(&[0, 1])), ("high", ints(&[0, 0]))],
                std::slice::from_ref(&x),
                &[2, 4]
            )
            .data,
            vec![0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            tensor_op(
                "broadcast",
                vec![("shape", ints(&[2, 2, 3]))],
                std::slice::from_ref(&x),
                &[2, 2, 3]
            )
            .data
            .len(),
            12
        );
        // A softmax row sums to one, which is the property and not the formula.
        let sm = tensor_op(
            "softmax",
            vec![("axis", Value::I(1))],
            std::slice::from_ref(&x),
            &[2, 3],
        );
        for r in 0..2 {
            let s: f64 = (0..3).map(|c| sm.get(&[r, c]).unwrap()).sum();
            assert!((s - 1.0).abs() < 1e-12, "row {r} sums to {s}");
        }
        // And it is shift-invariant: the guard against overflow is real.
        let big = f32t(&[1, 3], &[800.0, 801.0, 802.0]);
        let sb = tensor_op("softmax", vec![("axis", Value::I(1))], &[big], &[1, 3]);
        assert!(sb.data.iter().all(|v| v.is_finite()), "{:?}", sb.data);
    }

    #[test]
    fn gather_scatter_concat_sort_and_topk() {
        let table = f32t(&[3, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        let idx = Tensor::new(vec![2], DType::I32, vec![2.0, 0.0]);
        assert_eq!(
            tensor_op(
                "gather",
                vec![("axis", Value::I(0))],
                &[table.clone(), idx.clone()],
                &[2, 2]
            )
            .data,
            vec![20.0, 21.0, 0.0, 1.0]
        );
        let upd = f32t(&[2, 2], &[7.0, 7.0, 8.0, 8.0]);
        let i2 = Tensor::new(vec![2, 2], DType::I32, vec![2.0, 2.0, 0.0, 0.0]);
        assert_eq!(
            tensor_op(
                "scatter",
                vec![("axis", Value::I(0))],
                &[table.clone(), i2, upd],
                &[3, 2]
            )
            .data,
            vec![8.0, 8.0, 10.0, 11.0, 7.0, 7.0]
        );
        assert_eq!(
            tensor_op(
                "concat",
                vec![("axis", Value::I(0))],
                &[
                    f32t(&[1, 2], &[1.0, 2.0]),
                    f32t(&[2, 2], &[3.0, 4.0, 5.0, 6.0])
                ],
                &[3, 2]
            )
            .data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        let unsorted = f32t(&[1, 4], &[3.0, 1.0, 4.0, 1.0]);
        assert_eq!(
            tensor_op(
                "sort",
                vec![("axis", Value::I(1))],
                std::slice::from_ref(&unsorted),
                &[1, 4]
            )
            .data,
            vec![1.0, 1.0, 3.0, 4.0]
        );
        // topk has two results, and the indices are the second.
        let op = Op::new("omni.tensor", "topk", 1)
            .with_inputs(&[0])
            .with_attr("k", Value::U(2))
            .with_attr("axis", Value::I(1))
            .with_output(1, ty(&[1, 2]))
            .with_output(
                2,
                Type::tensor(
                    vec![Dim::N(1), Dim::N(2)],
                    DType::Int {
                        w: 32,
                        signed: true,
                    },
                ),
            );
        let m = module(
            vec![("x".into(), ty(&[1, 4]))],
            vec![op, Op::new("omni.core", "return", 1).with_inputs(&[1, 2])],
        );
        let out = run(&m, &[unsorted], &(), &Limits::default()).unwrap();
        assert_eq!(out.returned[0].data, vec![4.0, 3.0]);
        assert_eq!(
            out.returned[1].data,
            vec![2.0, 0.0],
            "indices, ties by position"
        );
    }

    #[test]
    fn einsum_agrees_with_matmul_and_refuses_what_it_cannot_do() {
        let a = f32t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f32t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let mm = tensor_op("matmul", vec![], &[a.clone(), b.clone()], &[2, 2]);
        let es = tensor_op(
            "einsum",
            vec![("equation", Value::text("ik,kj->ij"))],
            &[a.clone(), b.clone()],
            &[2, 2],
        );
        assert_eq!(mm.data, es.data);
        // A transpose, a trace and a batched contraction, all one op.
        assert_eq!(
            tensor_op(
                "einsum",
                vec![("equation", Value::text("ij->ji"))],
                std::slice::from_ref(&a),
                &[3, 2]
            )
            .data,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(
            tensor_op(
                "einsum",
                vec![("equation", Value::text("ij->"))],
                std::slice::from_ref(&a),
                &[]
            )
            .data,
            vec![21.0]
        );
        // And what it will not guess at.
        let bad = Op::new("omni.tensor", "einsum", 1)
            .with_attr("equation", Value::text("...ij,...jk->...ik"))
            .with_output(2, ty(&[2, 2]));
        let e = one_op(bad, &[a.clone(), b.clone()]).expect_err("ellipsis");
        assert!(e.to_string().contains("ellipsis"), "{e}");
        let implicit = Op::new("omni.tensor", "einsum", 1)
            .with_attr("equation", Value::text("ik,kj"))
            .with_output(2, ty(&[2, 2]));
        let e = one_op(implicit, &[a, b]).expect_err("implicit output");
        assert!(e.to_string().contains("->"), "{e}");
    }

    // ------------------------------------------------------ control flow --

    #[test]
    fn a_while_loop_counts_and_a_bound_stops_a_runaway_one() {
        // while (x < 4) { x = x + 1 }
        let cond = Region {
            blocks: vec![Block {
                args: vec![(10, ty(&[1]))],
                ops: vec![
                    Op::new("omni.core", "constant", 1)
                        .with_attr("value", Value::F64(4.0))
                        .with_output(11, ty(&[1])),
                    // `x < 4` as `maximum(4 - x, 0)`: the dialect has no compare
                    // op, and a test should use the ops that exist.
                    Op::new("omni.tensor", "sub", 1)
                        .with_inputs(&[11, 10])
                        .with_output(12, ty(&[1])),
                    Op::new("omni.tensor", "maximum", 1)
                        .with_inputs(&[12, 12])
                        .with_output(13, ty(&[1])),
                    Op::new("omni.core", "yield", 1).with_inputs(&[13]),
                ],
            }],
        };
        let body = Region {
            blocks: vec![Block {
                args: vec![(20, ty(&[1]))],
                ops: vec![
                    Op::new("omni.core", "constant", 1)
                        .with_attr("value", Value::F64(1.0))
                        .with_output(21, ty(&[1])),
                    Op::new("omni.tensor", "add", 1)
                        .with_inputs(&[20, 21])
                        .with_output(22, ty(&[1])),
                    Op::new("omni.core", "yield", 1).with_inputs(&[22]),
                ],
            }],
        };
        let mut w = Op::new("omni.core", "while", 1)
            .with_inputs(&[0])
            .with_output(1, ty(&[1]));
        w.regions = vec![cond, body];
        let m = module(
            vec![("x".into(), ty(&[1]))],
            vec![w, Op::new("omni.core", "return", 1).with_inputs(&[1])],
        );
        let out = run(&m, &[f32t(&[1], &[0.0])], &(), &Limits::default()).unwrap();
        assert_eq!(out.returned[0].data, vec![4.0]);

        // A graph is untrusted input, so the loop is bounded and the bound is
        // reported rather than the process hanging.
        let e = run(
            &m,
            &[f32t(&[1], &[-1e9])],
            &(),
            &Limits {
                max_iters: 100,
                ..Default::default()
            },
        )
        .expect_err("should hit the bound");
        assert!(e.to_string().contains("over the limit"), "{e}");
    }

    #[test]
    fn if_takes_the_branch_the_condition_names() {
        let branch = |v: f64, id: u32| Region {
            blocks: vec![Block {
                args: Vec::new(),
                ops: vec![
                    Op::new("omni.core", "constant", 1)
                        .with_attr("value", Value::F64(v))
                        .with_output(id, ty(&[1])),
                    Op::new("omni.core", "yield", 1).with_inputs(&[id]),
                ],
            }],
        };
        for (cond, want) in [(1.0, 7.0), (0.0, 9.0)] {
            let mut op = Op::new("omni.core", "if", 1)
                .with_inputs(&[0])
                .with_output(1, ty(&[1]));
            op.regions = vec![branch(7.0, 30), branch(9.0, 31)];
            let m = module(
                vec![("c".into(), ty(&[1]))],
                vec![op, Op::new("omni.core", "return", 1).with_inputs(&[1])],
            );
            let out = run(&m, &[f32t(&[1], &[cond])], &(), &Limits::default()).unwrap();
            assert_eq!(out.returned[0].data, vec![want], "condition {cond}");
        }
    }

    #[test]
    fn scan_threads_an_accumulator_and_map_does_not() {
        // scan(acc, xs) { acc + x } is a cumulative sum, which `cumsum` also
        // computes — so the two must agree, or one of them is wrong.
        let body = Region {
            blocks: vec![Block {
                args: vec![(40, ty(&[])), (41, ty(&[]))],
                ops: vec![
                    Op::new("omni.tensor", "add", 1)
                        .with_inputs(&[40, 41])
                        .with_output(42, ty(&[])),
                    Op::new("omni.core", "yield", 1).with_inputs(&[42]),
                ],
            }],
        };
        let mut sc = Op::new("omni.core", "scan", 1)
            .with_inputs(&[0, 1])
            .with_attr("axis", Value::U(0))
            .with_output(2, ty(&[]))
            .with_output(3, ty(&[4]));
        sc.regions = vec![body];
        let m = module(
            vec![("acc".into(), ty(&[])), ("xs".into(), ty(&[4]))],
            vec![sc, Op::new("omni.core", "return", 1).with_inputs(&[2, 3])],
        );
        let xs = f32t(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let out = run(
            &m,
            &[Tensor::new(vec![], DType::F32, vec![0.0]), xs.clone()],
            &(),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(out.returned[0].data, vec![10.0], "the accumulator");
        assert_eq!(
            out.returned[1].data,
            tensor_op("cumsum", vec![("axis", Value::I(0))], &[xs], &[4]).data
        );
    }

    #[test]
    fn a_tuple_is_built_and_read_back() {
        let t = Op::new("omni.core", "tuple", 1)
            .with_inputs(&[0, 1])
            .with_output(2, ty(&[1]));
        let g = Op::new("omni.core", "get", 1)
            .with_inputs(&[2])
            .with_attr("index", Value::U(1))
            .with_output(3, ty(&[1]));
        let m = module(
            vec![("a".into(), ty(&[1])), ("b".into(), ty(&[1]))],
            vec![t, g, Op::new("omni.core", "return", 1).with_inputs(&[3])],
        );
        let out = run(
            &m,
            &[f32t(&[1], &[5.0]), f32t(&[1], &[6.0])],
            &(),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(out.returned[0].data, vec![6.0]);
    }

    #[test]
    fn an_assert_is_the_graph_s_own_claim_and_failing_it_is_an_outcome() {
        let m = module(
            vec![("x".into(), ty(&[1]))],
            vec![Op::new("omni.core", "assert", 1)
                .with_inputs(&[0])
                .with_attr("message", Value::text("x must be nonzero"))],
        );
        assert!(run(&m, &[f32t(&[1], &[1.0])], &(), &Limits::default()).is_ok());
        let e = run(&m, &[f32t(&[1], &[0.0])], &(), &Limits::default()).expect_err("false");
        assert!(e.to_string().contains("x must be nonzero"), "{e}");
    }

    #[test]
    fn a_call_runs_another_function() {
        let mut m = module(
            vec![("x".into(), ty(&[2]))],
            vec![
                Op::new("omni.core", "call", 1)
                    .with_inputs(&[0])
                    .with_attr("callee", Value::text("double"))
                    .with_output(1, ty(&[2])),
                Op::new("omni.core", "return", 1).with_inputs(&[1]),
            ],
        );
        m.functions.push((
            "double".into(),
            Function {
                params: vec![("y".into(), ty(&[2]))],
                results: vec![ty(&[2])],
                attrs: Vec::new(),
                body: Region {
                    blocks: vec![Block {
                        args: Vec::new(),
                        ops: vec![
                            Op::new("omni.tensor", "add", 1)
                                .with_inputs(&[0, 0])
                                .with_output(1, ty(&[2])),
                            Op::new("omni.core", "return", 1).with_inputs(&[1]),
                        ],
                    }],
                },
                constraints: Vec::new(),
            },
        ));
        let out = run(&m, &[f32t(&[2], &[3.0, 4.0])], &(), &Limits::default()).unwrap();
        assert_eq!(out.returned[0].data, vec![6.0, 8.0]);
    }

    // ------------------------------------------------------------ omni.nn --

    #[test]
    fn a_causal_attention_sees_only_the_past() {
        // One head, three positions, one dimension. Position 0 can attend only to
        // key 0, so whatever the scores are its output is exactly v[0]. That is
        // the property a causal mask has, and it does not depend on the softmax.
        let q = f32t(&[1, 3, 1], &[1.0, 1.0, 1.0]);
        let k = f32t(&[1, 3, 1], &[1.0, 2.0, 3.0]);
        let v = f32t(&[1, 3, 1], &[10.0, 20.0, 30.0]);
        let op = Op::new("omni.nn", "attention", 2)
            .with_attr("causal", Value::Bool(true))
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty(&[1, 3, 1]));
        let out = one_op(op, &[q.clone(), k.clone(), v.clone()]).unwrap();
        assert_eq!(
            out.get(&[0, 0, 0]).unwrap(),
            10.0,
            "position 0 sees only v[0]"
        );
        // And later positions are a convex combination, so they lie in range and
        // are strictly increasing here because the scores are.
        let a = out.get(&[0, 1, 0]).unwrap();
        let b = out.get(&[0, 2, 0]).unwrap();
        assert!((10.0..=20.0).contains(&a), "{a}");
        assert!(a < b && b <= 30.0, "{a} then {b}");

        // Without the mask, position 0 sees everything and the answer changes.
        let open = Op::new("omni.nn", "attention", 2)
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty(&[1, 3, 1]));
        let out2 = one_op(open, &[q, k, v]).unwrap();
        assert!(
            out2.get(&[0, 0, 0]).unwrap() > 10.0,
            "the mask is load-bearing"
        );
    }

    #[test]
    fn grouped_query_attention_shares_kv_heads() {
        // Four query heads over two kv heads: heads 0 and 1 read kv head 0.
        let q = f32t(&[4, 1, 1], &[1.0, 1.0, 1.0, 1.0]);
        let k = f32t(&[2, 1, 1], &[1.0, 1.0]);
        let v = f32t(&[2, 1, 1], &[100.0, 200.0]);
        let op = Op::new("omni.nn", "attention", 2)
            .with_attr("kv_groups", Value::U(2))
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty(&[4, 1, 1]));
        let out = one_op(op, &[q.clone(), k.clone(), v]).unwrap();
        assert_eq!(out.data, vec![100.0, 100.0, 200.0, 200.0]);

        // And a grouping the shapes do not support is an error rather than a
        // modulo that happens to produce numbers.
        let bad = Op::new("omni.nn", "attention", 2)
            .with_attr("kv_groups", Value::U(3))
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty(&[4, 1, 1]));
        let e = one_op(bad, &[q, k.clone(), k]).expect_err("4 is not 2x3");
        assert!(e.to_string().contains("do not agree"), "{e}");
    }

    #[test]
    fn a_sliding_window_forgets_and_a_softcap_bounds() {
        let q = f32t(&[1, 4, 1], &[1.0, 1.0, 1.0, 1.0]);
        let k = f32t(&[1, 4, 1], &[0.0, 0.0, 0.0, 0.0]);
        let v = f32t(&[1, 4, 1], &[1.0, 2.0, 4.0, 8.0]);
        // Window 1 with a causal mask: each position sees itself and the one
        // before, so position 3 averages v[2] and v[3].
        let op = Op::new("omni.nn", "attention", 2)
            .with_attr("causal", Value::Bool(true))
            .with_attr("window", Value::U(1))
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty(&[1, 4, 1]));
        let out = one_op(op, &[q.clone(), k.clone(), v.clone()]).unwrap();
        assert_eq!(out.get(&[0, 3, 0]).unwrap(), 6.0, "(4 + 8) / 2");
        assert_eq!(out.get(&[0, 0, 0]).unwrap(), 1.0, "position 0 sees itself");

        // A soft cap keeps a large logit finite and smooth. With equal keys the
        // output is unchanged, which is the point: it caps the logits, not the
        // values.
        let capped = Op::new("omni.nn", "attention", 2)
            .with_attr("softcap", Value::F64(2.0))
            .with_attr("scale", Value::F64(1000.0))
            .with_output(3, ty(&[1, 4, 1]));
        let out = one_op(capped, &[q, k, v]).unwrap();
        assert!(out.data.iter().all(|x| x.is_finite()), "{:?}", out.data);
        assert!(
            (out.get(&[0, 0, 0]).unwrap() - 3.75).abs() < 1e-6,
            "{:?}",
            out.data
        );
    }

    #[test]
    fn rms_norm_normalizes_and_layer_norm_also_centres() {
        let x = f32t(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let rms = one_op(
            Op::new("omni.nn", "norm", 1)
                .with_attr("kind", Value::text("rms"))
                .with_attr("eps", Value::F64(0.0))
                .with_output(1, ty(&[1, 4])),
            std::slice::from_ref(&x),
        )
        .unwrap();
        let ms: f64 = rms.data.iter().map(|v| v * v).sum::<f64>() / 4.0;
        assert!(
            (ms - 1.0).abs() < 1e-5,
            "rms of the result is {}",
            ms.sqrt()
        );

        let ln = one_op(
            Op::new("omni.nn", "norm", 1)
                .with_attr("kind", Value::text("layer"))
                .with_attr("eps", Value::F64(0.0))
                .with_output(1, ty(&[1, 4])),
            std::slice::from_ref(&x),
        )
        .unwrap();
        let mean: f64 = ln.data.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5, "layer norm leaves mean {mean}");
        // The two are different functions, which is why they are two kinds.
        assert_ne!(rms.data, ln.data);

        let e = one_op(
            Op::new("omni.nn", "norm", 1)
                .with_attr("kind", Value::text("group"))
                .with_output(1, ty(&[1, 4])),
            &[x],
        )
        .expect_err("group norm");
        assert!(e.to_string().contains("norm kind `group`"), "{e}");
    }

    #[test]
    fn rope_rotates_pairs_and_the_two_conventions_disagree() {
        // A rotation preserves the length of each pair, whichever pairing is
        // used — that is what makes it a rotation.
        let x = f32t(
            &[3, 1, 4],
            &(0..12)
                .map(|i| (i as f64 + 1.0) * 0.1)
                .collect::<Vec<f64>>(),
        );
        let run_rope = |interleaved: bool| {
            one_op(
                Op::new("omni.nn", "rope", 1)
                    .with_attr("theta", Value::F64(10000.0))
                    .with_attr("interleaved", Value::Bool(interleaved))
                    .with_output(
                        1,
                        Type::tensor(vec![Dim::N(3), Dim::N(1), Dim::N(4)], DType::F32),
                    ),
                std::slice::from_ref(&x),
            )
            .unwrap()
        };
        for interleaved in [false, true] {
            let out = run_rope(interleaved);
            for s in 0..3u64 {
                let pairs: Vec<(u64, u64)> = if interleaved {
                    vec![(0, 1), (2, 3)]
                } else {
                    vec![(0, 2), (1, 3)]
                };
                for (a, b) in pairs {
                    let before =
                        x.get(&[s, 0, a]).unwrap().powi(2) + x.get(&[s, 0, b]).unwrap().powi(2);
                    let after =
                        out.get(&[s, 0, a]).unwrap().powi(2) + out.get(&[s, 0, b]).unwrap().powi(2);
                    assert!(
                        (before - after).abs() < 1e-4,
                        "interleaved={interleaved} s={s} pair ({a},{b}): {before} -> {after}"
                    );
                }
            }
            // Position 0 is the identity: the angle is zero.
            for d in 0..4u64 {
                assert!((out.get(&[0, 0, d]).unwrap() - x.get(&[0, 0, d]).unwrap()).abs() < 1e-6);
            }
        }
        // And they are genuinely different functions, which is why §06 makes the
        // convention an explicit parameter rather than a default.
        assert_ne!(run_rope(false).data, run_rope(true).data);
    }

    #[test]
    fn an_embedding_is_a_gather_and_activations_are_named() {
        let table = f32t(&[3, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        let tokens = Tensor::new(vec![2], DType::I32, vec![1.0, 2.0]);
        let out = one_op(
            Op::new("omni.nn", "embedding", 1).with_output(2, ty(&[2, 2])),
            &[tokens, table],
        )
        .unwrap();
        assert_eq!(out.data, vec![10.0, 11.0, 20.0, 21.0]);

        let x = f32t(&[3], &[-1.0, 0.0, 2.0]);
        let relu = one_op(
            Op::new("omni.nn", "activation", 1)
                .with_attr("kind", Value::text("relu"))
                .with_output(1, ty(&[3])),
            std::slice::from_ref(&x),
        )
        .unwrap();
        assert_eq!(relu.data, vec![0.0, 0.0, 2.0]);
        // gelu and its tanh approximation are close but not equal, and the
        // interpreter keeps them apart.
        let g = one_op(
            Op::new("omni.nn", "activation", 1)
                .with_attr("kind", Value::text("gelu"))
                .with_output(1, ty(&[3])),
            std::slice::from_ref(&x),
        )
        .unwrap();
        let gt = one_op(
            Op::new("omni.nn", "activation", 1)
                .with_attr("kind", Value::text("gelu_tanh"))
                .with_output(1, ty(&[3])),
            std::slice::from_ref(&x),
        )
        .unwrap();
        for (a, b) in g.data.iter().zip(&gt.data) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
        assert_ne!(g.data, gt.data);
        // erf against known values, since gelu rests on it.
        let e = tensor_op("erf", vec![], &[f32t(&[3], &[0.0, 0.5, 2.0])], &[3]);
        for (got, want) in e.data.iter().zip([0.0, 0.520_499_877_8, 0.995_322_265_0]) {
            assert!((got - want).abs() < 1e-6, "erf: {got} vs {want}");
        }
    }

    // --------------------------------------------------------- omni.quant --

    #[test]
    fn dequantize_takes_its_scale_from_the_operands() {
        // §05's affine-sub over two groups of two, with scale and zero as
        // operands rather than as expressions inside the scheme.
        let q = Tensor::new(
            vec![4, 2],
            DType::Int {
                w: 8,
                signed: false,
            },
            vec![8.0, 9.0, 7.0, 8.0, 0.0, 15.0, 8.0, 8.0],
        );
        let scale = f32t(&[2, 2], &[0.5, 0.25, 1.0, 2.0]);
        let zero = f32t(&[2, 2], &[8.0, 8.0, 8.0, 8.0]);
        let scheme = Value::map(vec![
            ("scheme", Value::text("affine")),
            ("formula", Value::text("affine-sub")),
            ("out", DType::F32.to_value()),
            ("axis", Value::U(0)),
            ("block", Value::Array(vec![Value::U(2), Value::U(1)])),
        ]);
        let out = one_op(
            Op::new("omni.quant", "dequantize", 1)
                .with_attr("scheme", scheme.clone())
                .with_output(3, ty(&[4, 2])),
            &[q.clone(), scale.clone(), zero.clone()],
        )
        .unwrap();
        assert_eq!(out.data, vec![0.0, 0.25, -0.5, 0.0, -8.0, 14.0, 0.0, 0.0]);

        // Quantizing back lands on the same integers, which is the round trip
        // §05 claims for a uniform affine scheme.
        let back = one_op(
            Op::new("omni.quant", "quantize", 1)
                .with_attr("scheme", scheme.clone())
                .with_output(
                    3,
                    Type::tensor(
                        vec![Dim::N(4), Dim::N(2)],
                        DType::Int {
                            w: 8,
                            signed: false,
                        },
                    ),
                ),
            &[out, scale, zero],
        )
        .unwrap();
        assert_eq!(back.data, q.data);

        // A scheme that also carries the scale as an expression is two answers to
        // one question.
        // `Value::map` puts the keys in canonical order, so building the
        // scheme afresh with the extra key is how it gets there legally.
        let both = Value::map(vec![
            ("scheme", Value::text("affine")),
            ("formula", Value::text("affine-sub")),
            ("out", DType::F32.to_value()),
            ("axis", Value::U(0)),
            ("block", Value::Array(vec![Value::U(2), Value::U(1)])),
            ("scale", Value::U(1)),
        ]);
        let e = one_op(
            Op::new("omni.quant", "dequantize", 1)
                .with_attr("scheme", both)
                .with_output(1, ty(&[4, 2])),
            &[q],
        )
        .expect_err("two sources");
        assert!(e.to_string().contains("Two sources"), "{e}");
    }

    // ------------------------------------------------------- the refusals --

    #[test]
    fn what_is_not_implemented_is_refused_by_name() {
        let x = f32t(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // What is left once conv, conv1d_causal, pool, interpolate and moe_route
        // were implemented: one op §07 names without defining, one that needs a
        // network, an op of no dialect, and a dialect nobody has heard of.
        for (dialect, name, needle) in [
            ("omni.nn", "ssm_scan", "ssm_scan"),
            ("omni.nn", "not_an_op", "omni.nn/not_an_op"),
            ("omni.io", "external", "external"),
            ("acme.secret", "op", "acme.secret/op"),
        ] {
            let mut op = Op::new(dialect, name, 1).with_output(1, ty(&[1, 2, 2]));
            if name == "external" {
                op = op.with_attr("id", Value::text("s3://weights"));
            }
            let e = one_op(op, std::slice::from_ref(&x)).expect_err("this op should be refused");
            assert!(
                matches!(e, Error::Unsupported(_)),
                "{dialect}/{name}: {e:?}"
            );
            assert!(e.to_string().contains(needle), "{dialect}/{name}: {e}");
        }
    }

    #[test]
    fn a_constant_naming_a_tensor_nobody_has_says_which() {
        let m = module(
            Vec::new(),
            vec![
                Op::new("omni.core", "constant", 1)
                    .with_attr("tensor", Value::text("model.embed_tokens.weight"))
                    .with_output(0, ty(&[2, 2])),
                Op::new("omni.core", "return", 1).with_inputs(&[0]),
            ],
        );
        let e = run(&m, &[], &(), &Limits::default()).expect_err("no weights");
        assert!(e.to_string().contains("model.embed_tokens.weight"), "{e}");

        // Given the weights, the same graph runs.
        let w: Vec<(String, Tensor)> = vec![(
            "model.embed_tokens.weight".into(),
            f32t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        )];
        let out = run(&m, &[], &w, &Limits::default()).unwrap();
        assert_eq!(out.returned[0].data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn budgets_are_enforced_and_reported() {
        let m = module(
            vec![("x".into(), ty(&[4]))],
            vec![
                Op::new("omni.tensor", "broadcast", 1)
                    .with_inputs(&[0])
                    .with_attr("shape", ints(&[1000, 1000, 4]))
                    .with_output(1, ty(&[1000, 1000, 4])),
                Op::new("omni.core", "return", 1).with_inputs(&[1]),
            ],
        );
        let e = run(
            &m,
            &[f32t(&[4], &[1.0, 2.0, 3.0, 4.0])],
            &(),
            &Limits {
                max_elems: 1000,
                ..Default::default()
            },
        )
        .expect_err("too big");
        assert!(e.to_string().contains("over the limit of 1000"), "{e}");
    }

    #[test]
    fn a_symbolic_dimension_binds_once_and_disagreement_is_an_error() {
        let sym = |names: &[&str]| {
            Type::tensor(
                names.iter().map(|n| Dim::Sym((*n).into())).collect(),
                DType::F32,
            )
        };
        let m = module(
            vec![
                ("a".into(), sym(&["B", "S"])),
                ("b".into(), sym(&["B", "S"])),
            ],
            vec![
                Op::new("omni.tensor", "add", 1)
                    .with_inputs(&[0, 1])
                    .with_output(2, sym(&["B", "S"])),
                Op::new("omni.core", "return", 1).with_inputs(&[2]),
            ],
        );
        let out = run(
            &m,
            &[f32t(&[2, 3], &[1.0; 6]), f32t(&[2, 3], &[2.0; 6])],
            &(),
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(out.dims, vec![("B".into(), 2), ("S".into(), 3)]);
        // The second argument disagrees about S, which a reshape would hide.
        let e = run(
            &m,
            &[f32t(&[2, 3], &[1.0; 6]), f32t(&[2, 4], &[2.0; 8])],
            &(),
            &Limits::default(),
        )
        .expect_err("S is 3 and 4");
        assert!(e.to_string().contains("`S` is 3 here and 4 there"), "{e}");
    }

    #[test]
    fn a_declared_constraint_is_checked_against_the_arguments() {
        let mut m = module(
            vec![(
                "x".into(),
                Type::tensor(vec![Dim::Sym("S".into())], DType::F32),
            )],
            vec![Op::new("omni.core", "return", 1).with_inputs(&[0])],
        );
        if let Some((_, f)) = m.functions.first_mut() {
            f.constraints = vec![Constraint {
                dim: "S".into(),
                rel: Rel::Ge,
                bound: 2,
            }];
        }
        assert!(run(&m, &[f32t(&[4], &[0.0; 4])], &(), &Limits::default()).is_ok());
        let e = run(&m, &[f32t(&[1], &[0.0])], &(), &Limits::default()).expect_err("S >= 2");
        assert!(e.to_string().contains("S >= 2"), "{e}");
    }
    // ------------------------------------------------- the whole thing --

    /// The claim §07 is for: a model that describes its own computation can be
    /// executed by something that was never told its architecture.
    ///
    /// `graph synthesize` builds a decoder from `arch.params` alone. Nothing here
    /// knows it is a transformer — the interpreter dispatches ops — and the
    /// result is checked as a probability distribution over the vocabulary,
    /// because that is the property a decoder's output has.
    #[test]
    fn a_synthesized_encoder_is_bidirectional_where_the_decoder_is_not() {
        // The mirror of the causality test below, and the only check that can
        // tell an encoder from a decoder: the encoder's output at position 0
        // *must* move when a later token changes. A synthesizer that emitted
        // `causal: true` by accident would verify, run, produce finite numbers
        // and pass every other assertion in this file.
        let (hidden, heads, layers, vocab) = (8u64, 2u64, 2u64, 5u64);
        let head_dim = hidden / heads;
        let w = |name: &str, shape: &[u64], k: f64| -> (String, Tensor) {
            let n = numel(shape);
            let data: Vec<f64> = (0..n).map(|i| (i as f64 * k).sin() * 0.3 + 0.05).collect();
            (
                name.to_string(),
                Tensor::new(shape.to_vec(), DType::F32, data),
            )
        };
        let mut weights: Vec<(String, Tensor)> =
            vec![w("model.embed_tokens.weight", &[vocab, hidden], 1.7)];
        for l in 0..layers {
            weights.push(w(&format!("model.layers.{l}.norm.weight"), &[hidden], 0.9));
            for (i, p) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
                let _ = head_dim;
                weights.push(w(
                    &format!("model.layers.{l}.attn.{p}.weight"),
                    &[hidden, hidden],
                    1.1 + i as f64 * 0.3 + l as f64,
                ));
            }
        }
        let available: Vec<String> = weights.iter().map(|(n, _)| n.clone()).collect();
        let params = Value::map(vec![
            ("hidden_size", Value::U(hidden)),
            ("n_layers", Value::U(layers)),
            ("n_heads", Value::U(heads)),
            ("activation", Value::text("gelu")),
            (
                "norm",
                Value::map(vec![
                    ("kind", Value::text("layer")),
                    ("eps", Value::F64(1e-5)),
                ]),
            ),
        ]);
        let m = crate::ir::synthesize("transformer.encoder", &params, &available)
            .expect("the synthesizer should build this");

        let toks = |a: f64, b: f64, c: f64| {
            Tensor::new(
                vec![1, 3],
                DType::Int {
                    w: 32,
                    signed: true,
                },
                vec![a, b, c],
            )
        };
        let out = run(&m, &[toks(0.0, 2.0, 4.0)], &weights, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized encoder did not run: {e}"));
        let h = &out.returned[0];
        // Hidden states, not logits: [B, S, hidden].
        assert_eq!(h.shape, vec![1, 3, hidden]);
        assert!(h.data.iter().all(|x| x.is_finite()), "{:?}", h.data);

        let out2 = run(&m, &[toks(0.0, 2.0, 1.0)], &weights, &Limits::default()).unwrap();
        let moved = (0..hidden)
            .any(|d| h.get(&[0, 0, d]).unwrap() != out2.returned[0].get(&[0, 0, d]).unwrap());
        assert!(
            moved,
            "position 0 did not move when a later token changed: this graph is causal"
        );
    }

    /// The six families added to reach §07.8's coverage, each *executed* rather
    /// than merely emitted.
    ///
    /// A synthesizer that produces a well-typed graph nobody has run is how the
    /// decoder came to attend across heads instead of positions and pass
    /// verification while doing it. So every family here is run over known
    /// weights, and each assertion is a property of *that* architecture — the
    /// mixture's output moves when the router changes, the recurrence's output
    /// at step t depends on step t−1, the graph convolution's node feature moves
    /// when a neighbour's does — rather than "it produced numbers".
    fn seeded_weights(specs: &[(&str, Vec<u64>, f64)]) -> Vec<(String, Tensor)> {
        specs
            .iter()
            .map(|(name, shape, k)| {
                let n = numel(shape);
                let data: Vec<f64> = (0..n).map(|i| (i as f64 * k).sin() * 0.4).collect();
                (
                    name.to_string(),
                    Tensor::new(shape.clone(), DType::F32, data),
                )
            })
            .collect()
    }

    fn synth_names(w: &[(String, Tensor)]) -> Vec<String> {
        w.iter().map(|(n, _)| n.clone()).collect()
    }

    #[test]
    fn a_synthesized_mixture_of_experts_routes_and_runs() {
        let (d, ff, experts, k) = (4u64, 6u64, 3u64, 2u64);
        let w = seeded_weights(&[
            ("moe.layers.0.router.weight", vec![d, experts], 1.3),
            ("moe.layers.0.experts.w_in", vec![experts, d, ff], 0.7),
            ("moe.layers.0.experts.w_out", vec![experts, ff, d], 1.1),
        ]);
        let params = Value::map(vec![
            ("hidden_size", Value::U(d)),
            ("intermediate_size", Value::U(ff)),
            ("n_experts", Value::U(experts)),
            ("top_k", Value::U(k)),
        ]);
        let m = crate::ir::synthesize("transformer.moe", &params, &synth_names(&w))
            .expect("the synthesizer should build this");
        let tokens = Tensor::new(
            vec![3, d],
            DType::F32,
            (0..3 * d).map(|i| (i as f64 * 0.37).cos()).collect(),
        );
        let out = run(&m, std::slice::from_ref(&tokens), &w, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized mixture did not run: {e}"));
        assert_eq!(out.returned[0].shape, vec![3, d]);
        assert!(out.returned[0].data.iter().all(|x| x.is_finite()));

        // The property that makes it a mixture rather than a stack: change the
        // router and the same tokens take a different path, so the output
        // changes even though every expert's weights are untouched.
        let mut rerouted = w.clone();
        rerouted[0].1 = Tensor::new(
            vec![d, experts],
            DType::F32,
            (0..d * experts)
                .map(|i| -(i as f64 * 1.3).sin() * 0.4)
                .collect(),
        );
        let out2 = run(&m, &[tokens], &rerouted, &Limits::default()).unwrap();
        assert_ne!(
            out.returned[0].data, out2.returned[0].data,
            "the routing made no difference, so nothing was routed"
        );
    }

    #[test]
    fn synthesized_recurrences_carry_state_across_time() {
        for (family, carry) in [("rnn.lstm", 2u64), ("rnn.gru", 1u64)] {
            let (input, hidden) = (3u64, 4u64);
            let gates = if family == "rnn.lstm" { 4 } else { 3 };
            let w = seeded_weights(&[
                ("rnn.layers.0.weight_ih", vec![gates * hidden, input], 0.9),
                ("rnn.layers.0.weight_hh", vec![gates * hidden, hidden], 1.4),
                ("rnn.layers.0.bias_ih", vec![gates * hidden], 0.3),
                ("rnn.layers.0.bias_hh", vec![gates * hidden], 0.6),
            ]);
            let params = Value::map(vec![
                ("input_size", Value::U(input)),
                ("hidden_size", Value::U(hidden)),
            ]);
            let m = crate::ir::synthesize(family, &params, &synth_names(&w))
                .unwrap_or_else(|e| panic!("{family}: {e}"));
            let x = |last: f64| {
                Tensor::new(
                    vec![3, input],
                    DType::F32,
                    (0..3 * input)
                        .map(|i| {
                            if i == 3 * input - 1 {
                                last
                            } else {
                                (i as f64 * 0.5).sin()
                            }
                        })
                        .collect(),
                )
            };
            let state = Tensor::new(
                vec![1, carry * hidden],
                DType::F32,
                vec![0.0; (carry * hidden) as usize],
            );
            let out = run(&m, &[x(0.2), state.clone()], &w, &Limits::default())
                .unwrap_or_else(|e| panic!("{family} did not run: {e}"));
            let hs = &out.returned[0];
            assert_eq!(hs.shape, vec![3, 1, hidden], "{family}");
            assert!(hs.data.iter().all(|v| v.is_finite()), "{family}");
            assert_eq!(out.returned[1].shape, vec![1, carry * hidden], "{family}");

            // Recurrence, stated as a property: changing the *last* input must
            // move the last output and leave the first alone. A body that
            // ignored its carry would pass the first half and fail the second,
            // and one that leaked the future would fail the first.
            let out2 = run(&m, &[x(-0.9), state], &w, &Limits::default()).unwrap();
            let h2 = &out2.returned[0];
            assert_eq!(
                hs.data[..hidden as usize],
                h2.data[..hidden as usize],
                "{family}: step 0 saw a later input"
            );
            assert_ne!(
                hs.data[(2 * hidden) as usize..],
                h2.data[(2 * hidden) as usize..],
                "{family}: the last step ignored its input"
            );
            // And the state really is threaded: the final carry is not the
            // zero it started from.
            assert!(
                out.returned[1].data.iter().any(|v| *v != 0.0),
                "{family}: the carry never changed"
            );
        }
    }

    #[test]
    fn a_synthesized_graph_network_passes_messages_along_its_edges() {
        let (fin, hidden, classes) = (2u64, 3u64, 2u64);
        let w = seeded_weights(&[
            ("gnn.layers.0.message.weight", vec![hidden, fin], 0.8),
            ("gnn.layers.0.self.weight", vec![hidden, fin], 1.2),
            ("gnn.head.weight", vec![classes, hidden], 1.9),
        ]);
        let params = Value::map(vec![
            ("input_size", Value::U(fin)),
            ("hidden_size", Value::U(hidden)),
            ("num_classes", Value::U(classes)),
        ]);
        let m = crate::ir::synthesize("gnn.mpnn", &params, &synth_names(&w)).expect("synthesizes");

        // Three nodes, two edges: 0 → 2 and 1 → 2. Node 2 therefore aggregates
        // *both* messages, which is the case a scatter would get wrong.
        let x = Tensor::new(
            vec![3, fin],
            DType::F32,
            vec![1.0, 0.5, -0.5, 0.25, 0.0, 0.0],
        );
        let src = Tensor::new(
            vec![2],
            DType::Int {
                w: 32,
                signed: true,
            },
            vec![0.0, 1.0],
        );
        let inc = Tensor::new(vec![2, 3], DType::F32, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let out = run(
            &m,
            &[x.clone(), src.clone(), inc.clone()],
            &w,
            &Limits::default(),
        )
        .unwrap_or_else(|e| panic!("the synthesized GNN did not run: {e}"));
        assert_eq!(out.returned[0].shape, vec![3, classes]);

        // Message passing, as a property: change node 0's features and node 2's
        // output must move, because there is an edge from 0 to 2 — while node
        // 1's output must not, because there is no edge from 0 to 1.
        let mut x2 = x.clone();
        x2.data[0] = -2.0;
        let out2 = run(&m, &[x2, src, inc], &w, &Limits::default()).unwrap();
        let row = |t: &Tensor, n: u64| -> Vec<f64> {
            (0..classes).map(|c| t.get(&[n, c]).unwrap()).collect()
        };
        assert_ne!(
            row(&out.returned[0], 2),
            row(&out2.returned[0], 2),
            "node 2 did not receive node 0's message"
        );
        assert_eq!(
            row(&out.returned[0], 1),
            row(&out2.returned[0], 1),
            "node 1 changed, and no edge reaches it from node 0"
        );
    }

    #[test]
    fn a_synthesized_policy_returns_both_of_its_heads() {
        let (obs, hidden, actions) = (4u64, 5u64, 3u64);
        let w = seeded_weights(&[
            ("rl.trunk.0.weight", vec![hidden, obs], 0.6),
            ("rl.trunk.0.bias", vec![hidden], 0.2),
            ("rl.policy.weight", vec![actions, hidden], 1.5),
            ("rl.value.weight", vec![1, hidden], 2.1),
        ]);
        let params = Value::map(vec![
            (
                "hidden_sizes",
                Value::Array(vec![Value::U(obs), Value::U(hidden)]),
            ),
            ("n_actions", Value::U(actions)),
        ]);
        let m = crate::ir::synthesize("rl.actor_critic", &params, &synth_names(&w))
            .expect("synthesizes");
        let x = Tensor::new(
            vec![2, obs],
            DType::F32,
            (0..2 * obs).map(|i| (i as f64 * 0.9).cos()).collect(),
        );
        let out = run(&m, &[x], &w, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized policy did not run: {e}"));
        // Two results from one graph over shared weights, which is the reason
        // this family is in the list at all.
        assert_eq!(out.returned.len(), 2);
        assert_eq!(out.returned[0].shape, vec![2, actions]);
        assert_eq!(out.returned[1].shape, vec![2, 1]);
        assert!(out
            .returned
            .iter()
            .all(|t| t.data.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn a_synthesized_audio_encoder_cannot_see_the_future() {
        let channels = [2u64, 3u64, 2u64];
        let kernel = 3u64;
        let w = seeded_weights(&[
            (
                "audio.blocks.0.conv.weight",
                vec![channels[1], channels[0], kernel],
                0.7,
            ),
            (
                "audio.blocks.1.conv.weight",
                vec![channels[2], channels[1], kernel],
                1.3,
            ),
        ]);
        let params = Value::map(vec![
            (
                "channels",
                Value::Array(channels.iter().map(|c| Value::U(*c)).collect()),
            ),
            ("kernel", Value::U(kernel)),
        ]);
        let m =
            crate::ir::synthesize("audio.encoder", &params, &synth_names(&w)).expect("synthesizes");
        let len = 6u64;
        let make = |last: f64| {
            Tensor::new(
                vec![1, channels[0], len],
                DType::F32,
                (0..channels[0] * len)
                    .map(|i| {
                        if i == channels[0] * len - 1 {
                            last
                        } else {
                            (i as f64 * 0.4).sin()
                        }
                    })
                    .collect(),
            )
        };
        let out = run(&m, &[make(0.1)], &w, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized encoder did not run: {e}"));
        assert_eq!(out.returned[0].shape, vec![1, channels[2], len]);

        // Causality, as a property rather than as an attribute: change the last
        // frame and every earlier output frame must be untouched. A symmetric
        // padding — the mistake `conv1d_causal` exists to prevent — moves them.
        let out2 = run(&m, &[make(-0.8)], &w, &Limits::default()).unwrap();
        for c in 0..channels[2] {
            for t in 0..len - 1 {
                assert_eq!(
                    out.returned[0].get(&[0, c, t]),
                    out2.returned[0].get(&[0, c, t]),
                    "frame {t} moved when a later frame changed"
                );
            }
        }
        assert_ne!(
            out.returned[0].get(&[0, 0, len - 1]),
            out2.returned[0].get(&[0, 0, len - 1]),
            "the last frame ignored the input that changed"
        );
    }

    #[test]
    fn a_synthesized_decoder_graph_executes_end_to_end() {
        let (hidden, heads, kv_heads, layers, vocab) = (8u64, 2u64, 1u64, 2u64, 5u64);
        let head_dim = hidden / heads;

        // Weights, made deterministically rather than randomly so a failure is
        // reproducible.
        let w = |name: &str, shape: &[u64], k: f64| -> (String, Tensor) {
            let n = numel(shape);
            let data: Vec<f64> = (0..n).map(|i| (i as f64 * k).sin() * 0.3 + 0.05).collect();
            (
                name.to_string(),
                Tensor::new(shape.to_vec(), DType::F32, data),
            )
        };
        let mut weights: Vec<(String, Tensor)> = vec![
            w("model.embed_tokens.weight", &[vocab, hidden], 1.7),
            w("lm_head.weight", &[vocab, hidden], 2.3),
        ];
        for l in 0..layers {
            weights.push(w(&format!("model.layers.{l}.norm.weight"), &[hidden], 0.9));
            for (i, p) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
                let out = if i == 1 || i == 2 {
                    head_dim * kv_heads
                } else {
                    hidden
                };
                weights.push(w(
                    &format!("model.layers.{l}.attn.{p}.weight"),
                    &[out, hidden],
                    1.1 + i as f64 * 0.3 + l as f64,
                ));
            }
        }
        let available: Vec<String> = weights.iter().map(|(n, _)| n.clone()).collect();

        let params = Value::map(vec![
            ("hidden_size", Value::U(hidden)),
            ("n_layers", Value::U(layers)),
            ("n_heads", Value::U(heads)),
            ("n_kv_heads", Value::U(kv_heads)),
            ("activation", Value::text("silu")),
            (
                "norm",
                Value::map(vec![
                    ("kind", Value::text("rms")),
                    ("eps", Value::F64(1e-5)),
                ]),
            ),
            (
                "rope",
                Value::map(vec![
                    ("theta", Value::F64(10000.0)),
                    ("interleaved", Value::Bool(false)),
                ]),
            ),
        ]);
        let m = crate::ir::synthesize("transformer.decoder", &params, &available)
            .expect("the synthesizer should build this");

        // It verifies first: executing a graph that does not verify would be
        // measuring the interpreter's tolerance rather than the graph.
        let table: Vec<(String, Vec<u64>)> = weights
            .iter()
            .map(|(n, t)| (n.clone(), t.shape.clone()))
            .collect();
        let _ = table;

        let tokens = Tensor::new(
            vec![1, 3],
            DType::Int {
                w: 32,
                signed: true,
            },
            vec![0.0, 2.0, 4.0],
        );
        let out = run(&m, &[tokens], &weights, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized graph did not run: {e}"));

        // [B, S, vocab] — the shape a decoder's logits have.
        assert_eq!(out.returned.len(), 1);
        let logits = &out.returned[0];
        assert_eq!(logits.shape, vec![1, 3, vocab], "logits are [B, S, vocab]");
        assert!(
            logits.data.iter().all(|x| x.is_finite()),
            "non-finite logits: {:?}",
            logits.data
        );
        // The symbolic dimensions were bound by the argument, not by the
        // declaration.
        assert!(out.dims.contains(&("B".into(), 1)));
        assert!(out.dims.contains(&("S".into(), 3)));
        // Two layers of a real decoder is a lot of ops, and the count is a
        // measurement rather than a guess.
        assert!(out.ops > 30, "{} ops for {layers} layers", out.ops);

        // Softmax of the logits is a distribution: that is what makes them
        // logits, and it is the end-to-end property worth asserting.
        let probs = softmax(logits, 2, &DType::F64);
        for b in 0..1u64 {
            for t in 0..3u64 {
                let total: f64 = (0..vocab).map(|v| probs.get(&[b, t, v]).unwrap()).sum();
                assert!((total - 1.0).abs() < 1e-9, "position {t} sums to {total}");
            }
        }

        // Causality, end to end and without knowing anything about attention: a
        // decoder's logits at position 0 cannot depend on a later token. Change
        // the last token and the first position's logits must not move.
        let other = Tensor::new(
            vec![1, 3],
            DType::Int {
                w: 32,
                signed: true,
            },
            vec![0.0, 2.0, 1.0],
        );
        let out2 = run(&m, &[other], &weights, &Limits::default()).unwrap();
        for v in 0..vocab {
            let a = logits.get(&[0, 0, v]).unwrap();
            let b = out2.returned[0].get(&[0, 0, v]).unwrap();
            assert_eq!(a, b, "position 0, vocab {v}: a later token changed it");
        }
        // And the last position *did* move, or the test above proves nothing.
        assert_ne!(
            (0..vocab)
                .map(|v| logits.get(&[0, 2, v]).unwrap())
                .collect::<Vec<f64>>(),
            (0..vocab)
                .map(|v| out2.returned[0].get(&[0, 2, v]).unwrap())
                .collect::<Vec<f64>>()
        );
    }

    /// §07.2's load-bearing claim: a model can ship a rewrite for an op the
    /// runtime does not know. So a graph whose `attention` has been lowered to
    /// primitives must compute what the unlowered one computes.
    #[test]
    fn a_lowered_graph_agrees_with_the_one_it_came_from() {
        // The shipped lowering declines `causal` and transposes the last two of
        // four axes, so the graph here is the configuration it accepts:
        // `[B, H, S, D]` with an explicit scale and nothing else.
        let q = f32t(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let k = f32t(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = f32t(&[1, 1, 2, 2], &[3.0, 4.0, 5.0, 6.0]);
        let ty3 = Type::tensor(vec![Dim::N(1), Dim::N(1), Dim::N(2), Dim::N(2)], DType::F32);
        let attn = Op::new("omni.nn", "attention", 2)
            .with_inputs(&[0, 1, 2])
            .with_attr("scale", Value::F64(1.0))
            .with_output(3, ty3.clone());
        let m = module(
            vec![
                ("q".into(), ty3.clone()),
                ("k".into(), ty3.clone()),
                ("v".into(), ty3.clone()),
            ],
            vec![attn, Op::new("omni.core", "return", 1).with_inputs(&[3])],
        );
        let direct = run(
            &m,
            &[q.clone(), k.clone(), v.clone()],
            &(),
            &Limits::default(),
        )
        .unwrap();

        let (lowered, applied) =
            crate::ir::apply_rewrites(&m, &crate::ir::shipped_lowerings(), false);
        assert!(
            !applied.applied.is_empty(),
            "the shipped lowering should apply; refused: {:?}",
            applied.refused
        );
        assert!(
            !lowered
                .ops()
                .iter()
                .any(|(_, o)| o.dialect == "omni.nn" && o.name == "attention"),
            "the nn op should be gone"
        );
        let via_rewrite = run(&lowered, &[q, k, v], &(), &Limits::default())
            .unwrap_or_else(|e| panic!("the lowered graph did not run: {e}"));
        for (a, b) in direct.returned[0]
            .data
            .iter()
            .zip(&via_rewrite.returned[0].data)
        {
            assert!(
                (a - b).abs() < 1e-6,
                "interpreted {a} against lowered {b}: §07.2 requires these to agree"
            );
        }
    }
    // ------------------------------------------------ the rest of omni.nn --

    /// An `omni.nn` op over the given operands, with a declared result shape.
    fn nn(name: &str, attrs: Vec<(&str, Value)>, args: &[Tensor], shape: &[u64]) -> Res<Tensor> {
        let mut op = Op::new("omni.nn", name, 1).with_output(args.len() as u32, ty(shape));
        for (k, v) in attrs {
            op = op.with_attr(k, v);
        }
        one_op(op, args)
    }

    #[test]
    fn a_convolution_is_a_cross_correlation_and_padding_is_where_it_says() {
        // [1, 1, 5] over a 3-tap kernel [1, 0, -1]: a first-difference filter, so
        // the answer is x[i] - x[i+2] at each valid position. Worked out here
        // rather than read off the implementation.
        let x = f32t(&[1, 1, 5], &[1.0, 2.0, 4.0, 8.0, 16.0]);
        let w = f32t(&[1, 1, 3], &[1.0, 0.0, -1.0]);
        let out = nn("conv", vec![], &[x.clone(), w.clone()], &[1, 1, 3]).unwrap();
        assert_eq!(out.data, vec![1.0 - 4.0, 2.0 - 8.0, 4.0 - 16.0]);
        // The kernel is *not* flipped: a true convolution would give the
        // negatives of these, and every framework calls this one convolution.
        assert_eq!(out.data[0], -3.0);

        // Padding widens the output at both ends, and the pad is zeros.
        let padded = nn(
            "conv",
            vec![("padding", ints(&[1]))],
            &[x.clone(), w.clone()],
            &[1, 1, 5],
        )
        .unwrap();
        assert_eq!(padded.shape, vec![1, 1, 5]);
        assert_eq!(padded.data[0], 0.0 - 2.0, "the left pad is a zero");
        assert_eq!(padded.data[4], 8.0 - 0.0, "and so is the right");

        // Stride and dilation, each changing the output the way they should.
        let strided = nn(
            "conv",
            vec![("stride", ints(&[2]))],
            &[x.clone(), w.clone()],
            &[1, 1, 2],
        )
        .unwrap();
        assert_eq!(strided.data, vec![1.0 - 4.0, 4.0 - 16.0]);
        let dilated = nn(
            "conv",
            vec![("dilation", ints(&[2]))],
            &[x.clone(), w.clone()],
            &[1, 1, 1],
        )
        .unwrap();
        assert_eq!(dilated.data, vec![1.0 - 16.0]);

        // A bias is per output channel.
        let biased = nn("conv", vec![], &[x, w, f32t(&[1], &[100.0])], &[1, 1, 3]).unwrap();
        assert_eq!(biased.data, vec![97.0, 94.0, 88.0]);
    }

    #[test]
    fn grouped_convolution_keeps_the_groups_apart() {
        // Two channels, two groups, one output channel each: group 1's kernel
        // must never see group 0's input, which is the whole point of groups.
        let x = f32t(&[1, 2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let w = f32t(&[2, 1, 1], &[1.0, 2.0]);
        let out = nn("conv", vec![("groups", Value::U(2))], &[x, w], &[1, 2, 3]).unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 20.0, 40.0, 60.0]);

        // A grouping the channels do not divide into is an error, not a modulo.
        let e = nn(
            "conv",
            vec![("groups", Value::U(3))],
            &[f32t(&[1, 2, 3], &[0.0; 6]), f32t(&[2, 1, 1], &[0.0; 2])],
            &[1, 2, 3],
        )
        .expect_err("2 channels, 3 groups");
        assert!(e.to_string().contains("do not divide"), "{e}");
    }

    #[test]
    fn a_causal_conv1d_never_reads_the_future() {
        // The property, not the arithmetic: change the last input and every
        // output before the last must be unmoved. A symmetric padding would
        // fail this, which is why the op exists separately.
        let w = f32t(&[1, 1, 3], &[1.0, 2.0, 4.0]);
        let run = |last: f64| {
            nn(
                "conv1d_causal",
                vec![],
                &[f32t(&[1, 1, 4], &[1.0, 2.0, 3.0, last]), w.clone()],
                &[1, 1, 4],
            )
            .unwrap()
        };
        let a = run(4.0);
        let b = run(99.0);
        assert_eq!(a.shape, vec![1, 1, 4], "causal padding keeps the length");
        assert_eq!(a.data[..3], b.data[..3], "an earlier output saw the future");
        assert_ne!(a.data[3], b.data[3], "and the last one should have moved");
        // Position 0 sees only itself, through the kernel's last tap.
        assert_eq!(a.data[0], 1.0 * 4.0);
        // Position 1 sees inputs 0 and 1.
        assert_eq!(a.data[1], 1.0 * 2.0 + 2.0 * 4.0);
    }

    #[test]
    fn pooling_reduces_by_the_window_and_defaults_to_not_overlapping() {
        let x = f32t(&[1, 1, 4], &[1.0, 3.0, 2.0, 8.0]);
        let mx = nn(
            "pool",
            vec![("kind", Value::text("max")), ("window", ints(&[2]))],
            std::slice::from_ref(&x),
            &[1, 1, 2],
        )
        .unwrap();
        assert_eq!(mx.data, vec![3.0, 8.0]);
        let avg = nn(
            "pool",
            vec![("kind", Value::text("avg")), ("window", ints(&[2]))],
            std::slice::from_ref(&x),
            &[1, 1, 2],
        )
        .unwrap();
        assert_eq!(avg.data, vec![2.0, 5.0]);
        // An explicit stride of 1 overlaps, and gives one more output.
        let overlap = nn(
            "pool",
            vec![
                ("kind", Value::text("max")),
                ("window", ints(&[2])),
                ("stride", ints(&[1])),
            ],
            std::slice::from_ref(&x),
            &[1, 1, 3],
        )
        .unwrap();
        assert_eq!(overlap.data, vec![3.0, 3.0, 8.0]);
        let e = nn(
            "pool",
            vec![("kind", Value::text("median")), ("window", ints(&[2]))],
            &[x],
            &[1, 1, 2],
        )
        .expect_err("median");
        assert!(e.to_string().contains("pool kind `median`"), "{e}");
    }

    #[test]
    fn interpolation_doubles_and_the_two_modes_differ() {
        let x = f32t(&[1, 1, 2], &[0.0, 4.0]);
        let near = nn(
            "interpolate",
            vec![("mode", Value::text("nearest")), ("scale", Value::U(2))],
            std::slice::from_ref(&x),
            &[1, 1, 4],
        )
        .unwrap();
        assert_eq!(near.data, vec![0.0, 0.0, 4.0, 4.0]);
        let lin = nn(
            "interpolate",
            vec![("mode", Value::text("linear")), ("scale", Value::U(2))],
            std::slice::from_ref(&x),
            &[1, 1, 4],
        )
        .unwrap();
        // Half-pixel centres: the outputs sit at input coordinates
        // -0.25, 0.25, 0.75, 1.25, clamped at the edges.
        assert_eq!(lin.data, vec![0.0, 1.0, 3.0, 4.0]);
        assert_ne!(near.data, lin.data);
        // Scaling down is the same map in reverse, not a separate op.
        let down = nn(
            "interpolate",
            vec![("mode", Value::text("nearest")), ("scale", Value::F64(0.5))],
            &[f32t(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0])],
            &[1, 1, 2],
        )
        .unwrap();
        assert_eq!(down.shape, vec![1, 1, 2]);
    }

    #[test]
    fn moe_routing_picks_the_top_experts_and_can_renormalize() {
        // Three experts, one token. The routing matrix's columns are the
        // experts, so a one-hot token picks out a column.
        let x = f32t(&[1, 2], &[1.0, 0.0]);
        let w = f32t(&[2, 3], &[0.0, 10.0, 5.0, 0.0, 0.0, 0.0]);
        let op = Op::new("omni.nn", "moe_route", 1)
            .with_inputs(&[0, 1])
            .with_attr("top_k", Value::U(2))
            .with_output(2, ty(&[1, 2]))
            .with_output(
                3,
                Type::tensor(
                    vec![Dim::N(1), Dim::N(2)],
                    DType::Int {
                        w: 32,
                        signed: true,
                    },
                ),
            );
        let m = module(
            vec![("x".into(), ty(&[1, 2])), ("w".into(), ty(&[2, 3]))],
            vec![op, Op::new("omni.core", "return", 1).with_inputs(&[2, 3])],
        );
        let out = run(&m, &[x.clone(), w.clone()], &(), &Limits::default()).unwrap();
        // Expert 1 has the largest logit, then expert 2.
        assert_eq!(out.returned[1].data, vec![1.0, 2.0]);
        // Weights are a softmax over *all* experts, so the chosen two do not
        // sum to one.
        let picked: f64 = out.returned[0].data.iter().sum();
        assert!(picked < 1.0, "{picked} should leave mass on expert 0");

        // `normalize` is what asks for them to sum to one.
        let mut op2 = Op::new("omni.nn", "moe_route", 1)
            .with_inputs(&[0, 1])
            .with_attr("top_k", Value::U(2))
            .with_attr("normalize", Value::Bool(true))
            .with_output(2, ty(&[1, 2]));
        op2.outputs.push((
            3,
            Type::tensor(
                vec![Dim::N(1), Dim::N(2)],
                DType::Int {
                    w: 32,
                    signed: true,
                },
            ),
        ));
        let m2 = module(
            vec![("x".into(), ty(&[1, 2])), ("w".into(), ty(&[2, 3]))],
            vec![op2, Op::new("omni.core", "return", 1).with_inputs(&[2, 3])],
        );
        let out2 = run(&m2, &[x.clone(), w], &(), &Limits::default()).unwrap();
        let total: f64 = out2.returned[0].data.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "normalized weights sum to {total}"
        );

        // The transposed routing matrix is a named error rather than a guess.
        let e = nn(
            "moe_route",
            vec![("top_k", Value::U(1))],
            &[x, f32t(&[3, 2], &[0.0; 6])],
            &[1, 1],
        )
        .expect_err("transposed");
        assert!(e.to_string().contains("[d_model, experts]"), "{e}");
    }

    #[test]
    fn ssm_scan_is_a_specification_gap_and_says_so() {
        // Refused not for being hard but for being undefined: §07 names the op
        // without saying what it computes. The message has to make that
        // distinction, or the next person implements a guess.
        let e = nn(
            "ssm_scan",
            vec![],
            &[
                f32t(&[1, 2], &[1.0, 2.0]),
                f32t(&[1, 2], &[1.0, 2.0]),
                f32t(&[1, 2], &[1.0, 2.0]),
            ],
            &[1, 2],
        )
        .expect_err("undefined");
        assert!(matches!(e, Error::Unsupported(_)), "{e:?}");
        let m = e.to_string();
        assert!(m.contains("does not define"), "{m}");
        assert!(m.contains("discretization"), "{m}");
    }
    // ------------------------------------------- the other synthesizers --

    /// Weights for a family, deterministic so a failure is reproducible.
    fn weights_for(names: &[(&str, Vec<u64>)]) -> Vec<(String, Tensor)> {
        names
            .iter()
            .enumerate()
            .map(|(k, (name, shape))| {
                let n = numel(shape);
                let data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64 + 1.0) * (k as f64 + 1.7)).sin() * 0.5)
                    .collect();
                (
                    (*name).to_string(),
                    Tensor::new(shape.clone(), DType::F32, data),
                )
            })
            .collect()
    }

    #[test]
    fn a_synthesized_mlp_runs_and_computes_the_affine_stack() {
        let w = weights_for(&[
            ("mlp.layers.0.weight", vec![3, 2]),
            ("mlp.layers.0.bias", vec![3]),
            ("mlp.layers.1.weight", vec![2, 3]),
        ]);
        let names: Vec<String> = w.iter().map(|(n, _)| n.clone()).collect();
        let params = Value::map(vec![
            (
                "hidden_sizes",
                Value::Array(vec![Value::U(2), Value::U(3), Value::U(2)]),
            ),
            ("activation", Value::text("relu")),
        ]);
        let m = crate::ir::synthesize("mlp", &params, &names).expect("should synthesize");
        let x = f32t(&[1, 2], &[1.0, -1.0]);
        let out = run(&m, std::slice::from_ref(&x), &w, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized mlp did not run: {e}"));
        assert_eq!(out.returned[0].shape, vec![1, 2]);

        // The same arithmetic, done here: relu(x·W0ᵀ + b0)·W1ᵀ.
        let get = |n: &str| w.iter().find(|(k, _)| k == n).unwrap().1.clone();
        let (w0, b0, w1) = (
            get("mlp.layers.0.weight"),
            get("mlp.layers.0.bias"),
            get("mlp.layers.1.weight"),
        );
        let mut hidden = Vec::new();
        for j in 0..3u64 {
            let mut acc = b0.get(&[j]).unwrap();
            for i in 0..2u64 {
                acc += x.get(&[0, i]).unwrap() * w0.get(&[j, i]).unwrap();
            }
            hidden.push(acc.max(0.0));
        }
        for j in 0..2u64 {
            let want: f64 = (0..3)
                .map(|i| hidden[i as usize] * w1.get(&[j, i]).unwrap())
                .sum();
            let got = out.returned[0].get(&[0, j]).unwrap();
            assert!((got - want).abs() < 1e-6, "logit {j}: {got} vs {want}");
        }

        // A missing weight is named before anything is written.
        let e =
            crate::ir::synthesize("mlp", &params, &names[..1]).expect_err("a weight is missing");
        assert!(e.contains("mlp.layers.1.weight"), "{e}");
    }

    #[test]
    fn a_synthesized_cnn_runs_over_the_conv_and_pool_ops() {
        // Two blocks over 8x8 inputs: conv -> relu -> 2x2 max pool, twice, then a
        // global average pool and a linear head.
        let w = weights_for(&[
            ("cnn.blocks.0.conv.weight", vec![4, 1, 3, 3]),
            ("cnn.blocks.0.conv.bias", vec![4]),
            ("cnn.blocks.1.conv.weight", vec![8, 4, 3, 3]),
            ("cnn.head.weight", vec![3, 8]),
            ("cnn.head.bias", vec![3]),
        ]);
        let names: Vec<String> = w.iter().map(|(n, _)| n.clone()).collect();
        let params = Value::map(vec![
            (
                "channels",
                Value::Array(vec![Value::U(1), Value::U(4), Value::U(8)]),
            ),
            ("kernel", Value::U(3)),
            ("num_classes", Value::U(3)),
            ("height", Value::U(8)),
            ("width", Value::U(8)),
            ("pool", Value::U(2)),
            ("activation", Value::text("relu")),
        ]);
        let m =
            crate::ir::synthesize("cnn.classifier", &params, &names).expect("should synthesize");
        let img = Tensor::new(
            vec![1, 1, 8, 8],
            DType::F32,
            (0..64).map(|i| (i as f64 * 0.37).sin()).collect(),
        );
        let out = run(&m, std::slice::from_ref(&img), &w, &Limits::default())
            .unwrap_or_else(|e| panic!("the synthesized cnn did not run: {e}"));
        assert_eq!(out.returned[0].shape, vec![1, 3], "one logit per class");
        assert!(
            out.returned[0].data.iter().all(|v| v.is_finite()),
            "{:?}",
            out.returned[0].data
        );

        // Every intermediate shape held: 8x8 -> conv same -> pool 4x4 -> conv
        // same -> pool 2x2 -> global avg -> [1, 8]. If any of that were wrong the
        // graph would not have run, because each op declares its result type and
        // the interpreter checks it.
        assert!(out.ops > 10, "{} ops", out.ops);

        // A concrete input extent is required rather than assumed: without it the
        // head's feature count is unknowable.
        let mut bad: Vec<(&str, Value)> = vec![
            (
                "channels",
                Value::Array(vec![Value::U(1), Value::U(4), Value::U(8)]),
            ),
            ("kernel", Value::U(3)),
            ("num_classes", Value::U(3)),
        ];
        bad.push(("width", Value::U(8)));
        let e = crate::ir::synthesize("cnn.classifier", &Value::map(bad), &names)
            .expect_err("no height");
        assert!(e.contains("height"), "{e}");
    }

    #[test]
    fn an_unregistered_family_is_refused_with_the_list() {
        let e =
            crate::ir::synthesize("mamba", &Value::map(vec![]), &[]).expect_err("no such family");
        assert!(e.contains("mamba"), "{e}");
        for known in crate::ir::FAMILIES {
            assert!(e.contains(known), "the refusal should list {known}: {e}");
        }
    }
}
