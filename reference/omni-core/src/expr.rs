//! §04.7 — the tensor expression algebra (OTA).
//!
//! A tensor's `value` is a node in a small, closed, pure algebra rather than a
//! byte range. This module is the algebra: the node set, static typing,
//! normalization and identity, evaluation, and range pushdown.
//!
//! Four properties from §04.7.1 shape the code:
//!
//! * **Pure.** [`Expr`] is a value type. Evaluation takes a [`Ctx`] for reading
//!   stored bytes and does nothing else.
//! * **Total.** [`Expr::infer`] gives every node a `(shape, dtype)` from its
//!   inputs alone, and a disagreement with the owning `TensorDesc` is a hard
//!   error (R-T01) — the cheapest possible detection of a malformed file.
//! * **Closed and small.** [`CORE_OPS`] is the whole set. Anything else is a
//!   `plugin` node, which a reader may refuse (§04.7.7).
//! * **Deterministic identity.** [`Expr::identity`] is the digest of the
//!   normalized tree, so two publishers who write the same model differently
//!   get the same identity and dedup actually works (§04.7.5).
//!
//! Evaluation here materializes `f64` buffers. That is the wrong choice for a
//! production runtime and the right one for a reference: it makes the *semantics*
//! testable without a kernel library. Range pushdown ([`Expr::deps`]) is what a
//! real implementation needs and is implemented properly, because partial
//! loading being automatic rather than a special case is the claim of §04.7.4.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo};
use crate::dtype::{DType, Round};
use crate::layout::{numel, Layout};
use crate::store::Store;
use std::collections::BTreeMap;

/// Maximum expression nesting depth (R-T05).
pub const MAX_DEPTH: usize = 256;

/// The closed core node set of §04.7.2. A reader that implements these and
/// refuses unknown `plugin` nodes is a conforming C1 evaluator.
pub const CORE_OPS: &[&str] = &[
    // leaves
    "literal",
    "extern",
    "zeros",
    "ones",
    "full",
    "arange",
    "eye",
    "random",
    // structural
    "reshape",
    "transpose",
    "permute",
    "squeeze",
    "expand",
    "slice",
    "concat",
    "split",
    "pad",
    "gather",
    "relayout",
    // numeric
    "cast",
    "add",
    "sub",
    "mul",
    "div",
    "scale",
    "matmul",
    "norm",
    "clamp",
    // quantization
    "dequantize",
    "quantize",
    "sparse",
    "approx",
    // composition
    "delta",
    "select",
    "plugin",
];

// -------------------------------------------------------------------- errors --

#[derive(Debug)]
pub enum Error {
    /// Static typing failed: the tree is malformed.
    Type(String),
    /// Well-formed but this build cannot evaluate it. Reported as
    /// *indeterminate*, never as invalid (§15.1).
    Unsupported(String),
    /// A referenced object is not in the store.
    Missing(Digest),
    /// An `extern` leaf was reached. Never fetched implicitly (§04.7.2).
    External(String),
    Store(String),
    /// A bound from §12.4 was hit: depth, element count, or a length that does
    /// not fit its buffer.
    Bounds(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Type(m) => write!(f, "expression type error: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Missing(d) => write!(f, "object {} not present", crate::sha256::hex(&d[..8])),
            Error::External(u) => write!(f, "external value at `{u}` is never fetched implicitly"),
            Error::Store(m) => write!(f, "store: {m}"),
            Error::Bounds(m) => write!(f, "bounds: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::container::Error> for Error {
    fn from(e: crate::container::Error) -> Self {
        Error::Store(e.to_string())
    }
}

impl From<crate::store::Error> for Error {
    fn from(e: crate::store::Error) -> Self {
        Error::Store(e.to_string())
    }
}

type Res<T> = Result<T, Error>;

// -------------------------------------------------------------------- shapes --

/// One dimension. Symbolic dimensions exist for tensors whose size genuinely
/// varies (a vocabulary being extended) and must resolve through the model's
/// `dims` binding table before materialization (§04.7.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    N(u64),
    Sym(String),
    Dynamic,
}

impl Dim {
    pub fn size(&self) -> Option<u64> {
        match self {
            Dim::N(n) => Some(*n),
            _ => None,
        }
    }
}

pub type Shape = Vec<Dim>;

pub fn concrete(shape: &[Dim]) -> Option<Vec<u64>> {
    shape.iter().map(|d| d.size()).collect()
}

pub fn dims(shape: &[u64]) -> Shape {
    shape.iter().map(|d| Dim::N(*d)).collect()
}

/// Encodes a shape (§03.3 tag 1004's untagged form).
pub fn shape_to_value(shape: &[Dim]) -> Value {
    Value::Array(
        shape
            .iter()
            .map(|d| match d {
                Dim::N(n) => Value::U(*n),
                Dim::Sym(s) => Value::text(s.clone()),
                Dim::Dynamic => Value::I(-1),
            })
            .collect(),
    )
}

/// Parses a shape, tagged or untagged.
pub fn parse_shape_value(v: &Value) -> Res<Shape> {
    let v = match v {
        Value::Tag(crate::cbor::TAG_SHAPE, inner) => inner.as_ref(),
        other => other,
    };
    let a = v
        .as_array()
        .ok_or_else(|| Error::Type("shape must be an array".into()))?;
    a.iter()
        .map(|d| match d {
            Value::U(n) => Ok(Dim::N(*n)),
            Value::I(-1) => Ok(Dim::Dynamic),
            Value::Text(s) => Ok(Dim::Sym(s.clone())),
            other => Err(Error::Type(format!(
                "shape entry must be uint, text or -1, got {}",
                other.diag()
            ))),
        })
        .collect()
}

/// NumPy broadcasting over possibly-symbolic dimensions.
fn broadcast(a: &[Dim], b: &[Dim]) -> Res<Shape> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(a.len().wrapping_sub(n - i)).cloned();
        let y = b.get(b.len().wrapping_sub(n - i)).cloned();
        let d = match (x, y) {
            (Some(x), None) => x,
            (None, Some(y)) => y,
            (Some(Dim::N(1)), Some(y)) => y,
            (Some(x), Some(Dim::N(1))) => x,
            (Some(x), Some(y)) if x == y => x,
            (Some(x), Some(y)) => {
                return Err(Error::Type(format!("cannot broadcast {x:?} against {y:?}")))
            }
            (None, None) => unreachable!(),
        };
        out.push(d);
    }
    Ok(out)
}

// -------------------------------------------------------------------- scalars --

/// A scalar constant. Rationals are exact, which is the point of keeping
/// `scale` distinct from `mul`: LoRA's α/r is a ratio, and rounding it to a
/// float would make two implementations disagree in the last bit forever.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
    Ratio(i64, i64),
}

impl Scalar {
    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Int(n) => n as f64,
            Scalar::Float(f) => f,
            Scalar::Ratio(n, d) => n as f64 / d as f64,
        }
    }

    /// Exact when both operands are exact (§04.7.5 requires exact rational
    /// arithmetic for `scale` folding).
    pub fn times(self, other: Scalar) -> Scalar {
        match (self, other) {
            (Scalar::Float(_), _) | (_, Scalar::Float(_)) => {
                Scalar::Float(self.as_f64() * other.as_f64())
            }
            (a, b) => {
                let (an, ad) = a.ratio();
                let (bn, bd) = b.ratio();
                match (an.checked_mul(bn), ad.checked_mul(bd)) {
                    (Some(n), Some(d)) => Scalar::Ratio(n, d).reduced(),
                    _ => Scalar::Float(self.as_f64() * other.as_f64()),
                }
            }
        }
    }

    fn ratio(self) -> (i64, i64) {
        match self {
            Scalar::Int(n) => (n, 1),
            Scalar::Ratio(n, d) => (n, d),
            Scalar::Float(f) => (f as i64, 1),
        }
    }

    /// Lowest terms, and an integer when the denominator divides out. §04.7.5
    /// requires exact rational arithmetic for `scale` canonicalization.
    pub fn reduced(self) -> Scalar {
        let (mut n, mut d) = self.ratio();
        if d == 0 {
            return Scalar::Float(f64::NAN);
        }
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i64;
        let (n, d) = if g > 1 { (n / g, d / g) } else { (n, d) };
        if d == 1 {
            Scalar::Int(n)
        } else {
            Scalar::Ratio(n, d)
        }
    }

    /// The canonical encoding: an integer, a float, or tag 30's exact rational.
    pub fn to_value(self) -> Value {
        match self {
            Scalar::Int(n) => int_value(n),
            Scalar::Float(f) => Value::F64(f),
            Scalar::Ratio(n, d) => Value::Tag(
                crate::cbor::TAG_RATIONAL,
                Box::new(Value::Array(vec![int_value(n), int_value(d)])),
            ),
        }
    }

    /// Parses a scalar, accepting tag 30 rationals.
    pub fn from_value(v: &Value) -> Res<Scalar> {
        Ok(match v {
            Value::U(n) => Scalar::Int(*n as i64),
            Value::I(n) => Scalar::Int(*n),
            Value::F64(f) => Scalar::Float(*f),
            Value::Tag(crate::cbor::TAG_RATIONAL, inner) => {
                let a = inner
                    .as_array()
                    .ok_or_else(|| Error::Type("rational must be [num, den]".into()))?;
                let g = |i: usize| -> Res<i64> {
                    match a.get(i) {
                        Some(Value::U(n)) => Ok(*n as i64),
                        Some(Value::I(n)) => Ok(*n),
                        _ => Err(Error::Type("rational must be [num, den]".into())),
                    }
                };
                Scalar::Ratio(g(0)?, g(1)?)
            }
            other => {
                return Err(Error::Type(format!(
                    "expected a scalar, got {}",
                    other.diag()
                )))
            }
        })
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

fn int_value(n: i64) -> Value {
    if n < 0 {
        Value::I(n)
    } else {
        Value::U(n as u64)
    }
}

// ---------------------------------------------------------------- node fields --

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
        }
    }
    fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
        }
    }
}

/// Reduction order for `matmul` and `norm`. Pinning it costs performance and
/// buys bit-exactness; §04.7.6 says so out loud rather than pretending float
/// reductions are associative.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Sum {
    /// Unpinned: fastest, and *not* bit-reproducible across implementations.
    #[default]
    Unspecified,
    Sequential,
    Pairwise,
    Kahan,
}

impl Sum {
    fn name(self) -> Option<&'static str> {
        Some(match self {
            Sum::Unspecified => return None,
            Sum::Sequential => "sequential",
            Sum::Pairwise => "pairwise",
            Sum::Kahan => "kahan",
        })
    }
    fn parse(s: &str) -> Option<Sum> {
        Some(match s {
            "sequential" => Sum::Sequential,
            "pairwise" => Sum::Pairwise,
            "kahan" => Sum::Kahan,
            _ => return None,
        })
    }
    /// Sums `terms` in the declared order.
    fn reduce(self, terms: &[f64]) -> f64 {
        match self {
            Sum::Pairwise => pairwise(terms),
            Sum::Kahan => {
                let mut s = 0.0f64;
                let mut c = 0.0f64;
                for t in terms {
                    let y = t - c;
                    let x = s + y;
                    c = (x - s) - y;
                    s = x;
                }
                s
            }
            // Unspecified evaluates sequentially here; the difference is that
            // it is not *promised* to.
            Sum::Sequential | Sum::Unspecified => terms.iter().sum(),
        }
    }
}

fn pairwise(t: &[f64]) -> f64 {
    if t.len() <= 8 {
        return t.iter().sum();
    }
    let mid = t.len() / 2;
    pairwise(&t[..mid]) + pairwise(&t[mid..])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Edge,
    Reflect,
    Wrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaOp {
    Add,
    Xor,
    Replace,
    SparseAdd,
}

/// The declared error bound of an `approx` subtree (§03.7.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bound {
    Abs(f64),
    Rel(f64),
    Psnr(f64),
}

/// The distribution of a `random` leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dist {
    Uniform { lo: f64, hi: f64 },
    Normal { mean: f64, std: f64 },
}

/// A typed reference: `[otype, digest]`.
pub type Ref = (u16, Digest);

fn ref_value(r: &Ref) -> Value {
    Value::Array(vec![Value::U(r.0 as u64), Value::Bytes(r.1.to_vec())])
}

/// Parses a typed reference `[otype, digest]`, tagged or untagged.
pub fn parse_ref_value(v: &Value) -> Res<Ref> {
    let v = match v {
        Value::Tag(crate::cbor::TAG_REF, inner) => inner.as_ref(),
        other => other,
    };
    let a = v
        .as_array()
        .ok_or_else(|| Error::Type("ref must be [otype, digest]".into()))?;
    let t = a
        .first()
        .and_then(|x| x.as_u64())
        .ok_or_else(|| Error::Type("ref otype".into()))?;
    let d: Digest = a
        .get(1)
        .and_then(|x| x.as_bytes())
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| Error::Type("ref digest must be 32 bytes".into()))?;
    Ok((t as u16, d))
}

// --------------------------------------------------------------------- nodes --

/// A node in the tensor expression algebra.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// The only node whose bytes are in the store.
    Literal {
        chunks: Ref,
        dtype: DType,
        shape: Shape,
        layout: Layout,
    },
    /// Bytes elsewhere. Never fetched implicitly.
    Extern {
        uri: String,
        digest: Option<Digest>,
        dtype: DType,
        shape: Shape,
    },
    Full {
        value: Scalar,
        dtype: DType,
        shape: Shape,
    },
    Arange {
        start: Scalar,
        step: Scalar,
        dtype: DType,
        shape: Shape,
    },
    Eye {
        rows: u64,
        cols: u64,
        dtype: DType,
    },
    Random {
        dist: Dist,
        seed: u64,
        dtype: DType,
        shape: Shape,
    },
    Reshape {
        x: Box<Expr>,
        shape: Shape,
    },
    Permute {
        x: Box<Expr>,
        perm: Vec<usize>,
    },
    Squeeze {
        x: Box<Expr>,
        axes: Vec<usize>,
    },
    Expand {
        x: Box<Expr>,
        shape: Shape,
    },
    Slice {
        x: Box<Expr>,
        starts: Vec<u64>,
        sizes: Vec<u64>,
        steps: Vec<u64>,
    },
    Concat {
        xs: Vec<Expr>,
        axis: usize,
    },
    Split {
        x: Box<Expr>,
        axis: usize,
        sizes: Vec<u64>,
        pick: usize,
    },
    Pad {
        x: Box<Expr>,
        pads: Vec<(u64, u64)>,
        mode: PadMode,
        value: Scalar,
    },
    Gather {
        x: Box<Expr>,
        idx: Box<Expr>,
        axis: usize,
    },
    Relayout {
        x: Box<Expr>,
        layout: Layout,
    },
    Cast {
        x: Box<Expr>,
        dtype: DType,
        round: Round,
    },
    Bin {
        op: BinOp,
        a: Box<Expr>,
        b: Box<Expr>,
    },
    Scale {
        x: Box<Expr>,
        k: Scalar,
    },
    MatMul {
        a: Box<Expr>,
        b: Box<Expr>,
        sum: Sum,
    },
    Norm {
        x: Box<Expr>,
        axis: usize,
        p: f64,
        sum: Sum,
    },
    Clamp {
        x: Box<Expr>,
        lo: Scalar,
        hi: Scalar,
    },
    /// Integer/codebook → float. The scheme is data (§05), interpreted at
    /// evaluation time.
    Dequantize {
        x: Box<Expr>,
        scheme: Value,
    },
    Quantize {
        x: Box<Expr>,
        scheme: Value,
        round: Round,
    },
    /// Sparse → dense (§04.6). The scheme names the encoding; `parts` carries
    /// its component tensors.
    Sparse {
        scheme: String,
        parts: Vec<(String, Expr)>,
        attrs: Value,
        shape: Shape,
        dtype: DType,
        fill: Scalar,
    },
    /// Marks an intentionally lossy subtree. §15.1 R-T06 requires it around
    /// every lossy transform, so the loss is visible in the DAG forever.
    Approx {
        x: Box<Expr>,
        bound: Bound,
    },
    Delta {
        base: Box<Expr>,
        patch: Box<Expr>,
        op: DeltaOp,
    },
    /// Capability-conditional value (§10.3).
    Select {
        feature: String,
        a: Box<Expr>,
        b: Box<Expr>,
    },
    Plugin {
        ns: String,
        name: String,
        v: u64,
        args: Vec<Expr>,
        attrs: Value,
        crit: bool,
        shape: Shape,
        dtype: DType,
        fallback: Option<Box<Expr>>,
    },
}

/// The static type of a node.
#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    pub shape: Shape,
    pub dtype: DType,
}

impl Expr {
    pub fn op(&self) -> &'static str {
        match self {
            Expr::Literal { .. } => "literal",
            Expr::Extern { .. } => "extern",
            Expr::Full { .. } => "full",
            Expr::Arange { .. } => "arange",
            Expr::Eye { .. } => "eye",
            Expr::Random { .. } => "random",
            Expr::Reshape { .. } => "reshape",
            Expr::Permute { .. } => "permute",
            Expr::Squeeze { .. } => "squeeze",
            Expr::Expand { .. } => "expand",
            Expr::Slice { .. } => "slice",
            Expr::Concat { .. } => "concat",
            Expr::Split { .. } => "split",
            Expr::Pad { .. } => "pad",
            Expr::Gather { .. } => "gather",
            Expr::Relayout { .. } => "relayout",
            Expr::Cast { .. } => "cast",
            Expr::Bin { op, .. } => op.name(),
            Expr::Scale { .. } => "scale",
            Expr::MatMul { .. } => "matmul",
            Expr::Norm { .. } => "norm",
            Expr::Clamp { .. } => "clamp",
            Expr::Dequantize { .. } => "dequantize",
            Expr::Quantize { .. } => "quantize",
            Expr::Sparse { .. } => "sparse",
            Expr::Approx { .. } => "approx",
            Expr::Delta { .. } => "delta",
            Expr::Select { .. } => "select",
            Expr::Plugin { .. } => "plugin",
        }
    }

    /// Immediate sub-expressions, in argument order.
    pub fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Literal { .. }
            | Expr::Extern { .. }
            | Expr::Full { .. }
            | Expr::Arange { .. }
            | Expr::Eye { .. }
            | Expr::Random { .. } => vec![],
            Expr::Reshape { x, .. }
            | Expr::Permute { x, .. }
            | Expr::Squeeze { x, .. }
            | Expr::Expand { x, .. }
            | Expr::Slice { x, .. }
            | Expr::Split { x, .. }
            | Expr::Pad { x, .. }
            | Expr::Relayout { x, .. }
            | Expr::Cast { x, .. }
            | Expr::Scale { x, .. }
            | Expr::Norm { x, .. }
            | Expr::Clamp { x, .. }
            | Expr::Dequantize { x, .. }
            | Expr::Quantize { x, .. }
            | Expr::Approx { x, .. } => vec![x],
            Expr::Concat { xs, .. } => xs.iter().collect(),
            Expr::Gather { x, idx, .. } => vec![x, idx],
            Expr::Bin { a, b, .. } | Expr::MatMul { a, b, .. } => vec![a, b],
            Expr::Sparse { parts, .. } => parts.iter().map(|(_, e)| e).collect(),
            Expr::Delta { base, patch, .. } => vec![base, patch],
            Expr::Select { a, b, .. } => vec![a, b],
            Expr::Plugin { args, fallback, .. } => {
                let mut v: Vec<&Expr> = args.iter().collect();
                if let Some(f) = fallback {
                    v.push(f);
                }
                v
            }
        }
    }

    pub fn depth(&self) -> usize {
        1 + self.children().iter().map(|c| c.depth()).max().unwrap_or(0)
    }

    /// Every `literal`/`extern` leaf the tree reads, deduplicated.
    pub fn leaves(&self) -> Vec<&Expr> {
        let mut out = Vec::new();
        fn walk<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
            if matches!(e, Expr::Literal { .. } | Expr::Extern { .. }) {
                out.push(e);
            }
            for c in e.children() {
                walk(c, out);
            }
        }
        walk(self, &mut out);
        out
    }

    // ------------------------------------------------------------- typing --

    /// Static shape and dtype inference (§04.7.3).
    pub fn infer(&self) -> Res<Type> {
        if self.depth() > MAX_DEPTH {
            return Err(Error::Bounds(format!(
                "expression depth {} exceeds {MAX_DEPTH} (R-T05)",
                self.depth()
            )));
        }
        self.infer_inner()
    }

    fn infer_inner(&self) -> Res<Type> {
        Ok(match self {
            Expr::Literal { dtype, shape, .. }
            | Expr::Extern { dtype, shape, .. }
            | Expr::Full { dtype, shape, .. }
            | Expr::Arange { dtype, shape, .. }
            | Expr::Random { dtype, shape, .. } => Type {
                shape: shape.clone(),
                dtype: dtype.clone(),
            },
            Expr::Eye { rows, cols, dtype } => Type {
                shape: vec![Dim::N(*rows), Dim::N(*cols)],
                dtype: dtype.clone(),
            },
            Expr::Reshape { x, shape } => {
                let t = x.infer_inner()?;
                if let (Some(from), Some(to)) = (concrete(&t.shape), concrete(shape)) {
                    if numel(&from) != numel(&to) {
                        return Err(Error::Type(format!(
                            "reshape changes element count: {} -> {}",
                            numel(&from),
                            numel(&to)
                        )));
                    }
                }
                Type {
                    shape: shape.clone(),
                    dtype: t.dtype,
                }
            }
            Expr::Permute { x, perm } => {
                let t = x.infer_inner()?;
                if perm.len() != t.shape.len() {
                    return Err(Error::Type("permute: rank mismatch".into()));
                }
                let mut seen = vec![false; perm.len()];
                for p in perm {
                    if *p >= perm.len() || seen[*p] {
                        return Err(Error::Type("permute: not a permutation".into()));
                    }
                    seen[*p] = true;
                }
                Type {
                    shape: perm.iter().map(|p| t.shape[*p].clone()).collect(),
                    dtype: t.dtype,
                }
            }
            Expr::Squeeze { x, axes } => {
                let t = x.infer_inner()?;
                let mut shape = Vec::new();
                for (i, d) in t.shape.iter().enumerate() {
                    if axes.contains(&i) {
                        if d != &Dim::N(1) {
                            return Err(Error::Type(format!(
                                "squeeze: axis {i} has extent {d:?}, not 1"
                            )));
                        }
                    } else {
                        shape.push(d.clone());
                    }
                }
                Type {
                    shape,
                    dtype: t.dtype,
                }
            }
            Expr::Expand { x, shape } => {
                let t = x.infer_inner()?;
                // Expansion is broadcasting to a declared shape.
                broadcast(&t.shape, shape)?;
                Type {
                    shape: shape.clone(),
                    dtype: t.dtype,
                }
            }
            Expr::Slice {
                x,
                starts,
                sizes,
                steps,
            } => {
                let t = x.infer_inner()?;
                if starts.len() != t.shape.len()
                    || sizes.len() != t.shape.len()
                    || steps.len() != t.shape.len()
                {
                    return Err(Error::Type("slice: one start/size/step per axis".into()));
                }
                if steps.contains(&0) {
                    return Err(Error::Type("slice: zero step".into()));
                }
                for (i, d) in t.shape.iter().enumerate() {
                    if let Dim::N(n) = d {
                        let last = starts[i] + (sizes[i].saturating_sub(1)) * steps[i];
                        if sizes[i] > 0 && last >= *n {
                            return Err(Error::Type(format!(
                                "slice: axis {i} reads element {last} of {n}"
                            )));
                        }
                    }
                }
                Type {
                    shape: sizes.iter().map(|s| Dim::N(*s)).collect(),
                    dtype: t.dtype,
                }
            }
            Expr::Concat { xs, axis } => {
                if xs.is_empty() {
                    return Err(Error::Type("concat: no inputs".into()));
                }
                let first = xs[0].infer_inner()?;
                if *axis >= first.shape.len() {
                    return Err(Error::Type("concat: axis out of range".into()));
                }
                let mut total = 0u64;
                for x in xs {
                    let t = x.infer_inner()?;
                    if t.dtype != first.dtype {
                        return Err(Error::Type("concat: mixed dtypes".into()));
                    }
                    if t.shape.len() != first.shape.len() {
                        return Err(Error::Type("concat: mixed ranks".into()));
                    }
                    for (i, (a, b)) in t.shape.iter().zip(&first.shape).enumerate() {
                        if i != *axis && a != b {
                            return Err(Error::Type(format!(
                                "concat: axis {i} differs: {a:?} vs {b:?}"
                            )));
                        }
                    }
                    match &t.shape[*axis] {
                        Dim::N(n) => total += n,
                        other => {
                            return Err(Error::Type(format!(
                                "concat: concatenated axis must be concrete, got {other:?}"
                            )))
                        }
                    }
                }
                let mut shape = first.shape.clone();
                shape[*axis] = Dim::N(total);
                Type {
                    shape,
                    dtype: first.dtype,
                }
            }
            Expr::Split {
                x,
                axis,
                sizes,
                pick,
            } => {
                let t = x.infer_inner()?;
                if *axis >= t.shape.len() {
                    return Err(Error::Type("split: axis out of range".into()));
                }
                if *pick >= sizes.len() {
                    return Err(Error::Type("split: pick out of range".into()));
                }
                if let Dim::N(n) = t.shape[*axis] {
                    if sizes.iter().sum::<u64>() != n {
                        return Err(Error::Type(format!(
                            "split: sizes sum to {} but axis is {n}",
                            sizes.iter().sum::<u64>()
                        )));
                    }
                }
                let mut shape = t.shape.clone();
                shape[*axis] = Dim::N(sizes[*pick]);
                Type {
                    shape,
                    dtype: t.dtype,
                }
            }
            Expr::Pad { x, pads, .. } => {
                let t = x.infer_inner()?;
                if pads.len() != t.shape.len() {
                    return Err(Error::Type("pad: one (before, after) pair per axis".into()));
                }
                let mut shape = Vec::new();
                for (d, (lo, hi)) in t.shape.iter().zip(pads) {
                    shape.push(match d {
                        Dim::N(n) => Dim::N(n + lo + hi),
                        other if *lo == 0 && *hi == 0 => other.clone(),
                        other => {
                            return Err(Error::Type(format!(
                                "pad: cannot pad symbolic dimension {other:?}"
                            )))
                        }
                    });
                }
                Type {
                    shape,
                    dtype: t.dtype,
                }
            }
            Expr::Gather { x, idx, axis } => {
                let t = x.infer_inner()?;
                let i = idx.infer_inner()?;
                if *axis >= t.shape.len() {
                    return Err(Error::Type("gather: axis out of range".into()));
                }
                if i.shape.len() != 1 {
                    return Err(Error::Type("gather: index must be one-dimensional".into()));
                }
                let mut shape = t.shape.clone();
                shape[*axis] = i.shape[0].clone();
                Type {
                    shape,
                    dtype: t.dtype,
                }
            }
            // Relayout changes bit placement, not values, so the type is
            // unchanged. That is the whole point of §04.4's orthogonality.
            Expr::Relayout { x, .. } => x.infer_inner()?,
            Expr::Cast { x, dtype, .. } => Type {
                shape: x.infer_inner()?.shape,
                dtype: dtype.clone(),
            },
            Expr::Bin { a, b, .. } => {
                let ta = a.infer_inner()?;
                let tb = b.infer_inner()?;
                let dtype = wider(&ta.dtype, &tb.dtype);
                Type {
                    shape: broadcast(&ta.shape, &tb.shape)?,
                    dtype,
                }
            }
            Expr::Scale { x, .. } | Expr::Clamp { x, .. } | Expr::Approx { x, .. } => {
                x.infer_inner()?
            }
            Expr::MatMul { a, b, .. } => {
                let ta = a.infer_inner()?;
                let tb = b.infer_inner()?;
                if ta.shape.len() < 2 || tb.shape.len() < 2 {
                    return Err(Error::Type("matmul: operands must be at least 2-D".into()));
                }
                let (am, ak) = (
                    ta.shape[ta.shape.len() - 2].clone(),
                    ta.shape[ta.shape.len() - 1].clone(),
                );
                let (bk, bn) = (
                    tb.shape[tb.shape.len() - 2].clone(),
                    tb.shape[tb.shape.len() - 1].clone(),
                );
                if ak != bk {
                    return Err(Error::Type(format!(
                        "matmul: contraction dimensions differ: {ak:?} vs {bk:?}"
                    )));
                }
                let batch = broadcast(
                    &ta.shape[..ta.shape.len() - 2],
                    &tb.shape[..tb.shape.len() - 2],
                )?;
                let mut shape = batch;
                shape.push(am);
                shape.push(bn);
                Type {
                    shape,
                    dtype: wider(&ta.dtype, &tb.dtype),
                }
            }
            Expr::Norm { x, axis, .. } => {
                let t = x.infer_inner()?;
                if *axis >= t.shape.len() {
                    return Err(Error::Type("norm: axis out of range".into()));
                }
                let mut shape = t.shape.clone();
                shape[*axis] = Dim::N(1);
                Type {
                    shape,
                    dtype: t.dtype,
                }
            }
            Expr::Dequantize { x, scheme } => {
                let t = x.infer_inner()?;
                let out = match scheme.get("out") {
                    Some(d) => DType::from_value(d).map_err(Error::Type)?,
                    None => DType::F32,
                };
                Type {
                    shape: t.shape,
                    dtype: out,
                }
            }
            Expr::Quantize { x, scheme, .. } => {
                let t = x.infer_inner()?;
                let out = match scheme.get("out") {
                    Some(d) => DType::from_value(d).map_err(Error::Type)?,
                    None => {
                        return Err(Error::Type(
                            "quantize: scheme must declare its output dtype".into(),
                        ))
                    }
                };
                Type {
                    shape: t.shape,
                    dtype: out,
                }
            }
            Expr::Sparse {
                shape,
                dtype,
                parts,
                ..
            } => {
                for (_, p) in parts {
                    p.infer_inner()?;
                }
                Type {
                    shape: shape.clone(),
                    dtype: dtype.clone(),
                }
            }
            Expr::Delta { base, patch, op } => {
                let tb = base.infer_inner()?;
                let tp = patch.infer_inner()?;
                match op {
                    DeltaOp::Replace => tp,
                    DeltaOp::Xor => {
                        if tb.dtype != tp.dtype {
                            return Err(Error::Type("delta xor: dtypes must match".into()));
                        }
                        Type {
                            shape: broadcast(&tb.shape, &tp.shape)?,
                            dtype: tb.dtype,
                        }
                    }
                    DeltaOp::Add | DeltaOp::SparseAdd => Type {
                        shape: broadcast(&tb.shape, &tp.shape)?,
                        dtype: wider(&tb.dtype, &tp.dtype),
                    },
                }
            }
            Expr::Select { a, b, .. } => {
                let ta = a.infer_inner()?;
                let tb = b.infer_inner()?;
                if ta != tb {
                    return Err(Error::Type(
                        "select: both branches must have the same type, or a runtime's choice \
                         would change a tensor's declared shape"
                            .into(),
                    ));
                }
                ta
            }
            Expr::Plugin {
                shape,
                dtype,
                args,
                fallback,
                ..
            } => {
                for a in args {
                    a.infer_inner()?;
                }
                if let Some(f) = fallback {
                    let tf = f.infer_inner()?;
                    if &tf.shape != shape || &tf.dtype != dtype {
                        return Err(Error::Type(
                            "plugin: fallback must have the declared type".into(),
                        ));
                    }
                }
                Type {
                    shape: shape.clone(),
                    dtype: dtype.clone(),
                }
            }
        })
    }

    /// R-T01: the owning `TensorDesc` must declare exactly what inference
    /// produces.
    pub fn check_declared(&self, shape: &[Dim], dtype: &DType) -> Res<()> {
        let t = self.infer()?;
        if t.shape != shape {
            return Err(Error::Type(format!(
                "R-T01: declared shape {:?} but the expression has {:?}",
                shape, t.shape
            )));
        }
        if &t.dtype != dtype {
            return Err(Error::Type(format!(
                "R-T01: declared dtype {} but the expression produces {}",
                dtype.label(),
                t.dtype.label()
            )));
        }
        Ok(())
    }

    /// Whether this node's result is bit-reproducible across conforming
    /// implementations (§04.7.6).
    pub fn deterministic(&self) -> bool {
        let own = match self {
            // Float reductions are not associative, so an unpinned order is
            // not reproducible. Saying so is the point.
            Expr::MatMul { sum, .. } | Expr::Norm { sum, .. } => *sum != Sum::Unspecified,
            // Elementwise float arithmetic is IEEE-exact per element, so
            // `add` and `mul` are deterministic; only reductions are at issue.
            Expr::Plugin { .. } => false,
            Expr::Extern { .. } => false,
            _ => true,
        };
        own && self.children().iter().all(|c| c.deterministic())
    }

    /// True when the tree contains an `approx` node, i.e. the value is
    /// intentionally lossy (R-T06).
    pub fn is_lossy(&self) -> bool {
        matches!(self, Expr::Approx { .. }) || self.children().iter().any(|c| c.is_lossy())
    }

    /// Plugin namespaces this tree needs and cannot fall back from. A reader
    /// missing one of these must refuse the tensor (§04.7.7) — but may still
    /// read the rest of the model.
    pub fn required_plugins(&self) -> Vec<String> {
        let mut out = Vec::new();
        fn walk(e: &Expr, out: &mut Vec<String>) {
            if let Expr::Plugin {
                ns,
                name,
                v,
                crit,
                fallback,
                ..
            } = e
            {
                if *crit && fallback.is_none() {
                    out.push(format!("{ns}/{name}.{v}"));
                }
            }
            for c in e.children() {
                walk(c, out);
            }
        }
        walk(self, &mut out);
        out.sort();
        out.dedup();
        out
    }

    /// Rewrites `plugin` nodes this reader cannot evaluate into their declared
    /// `fallback`, which is what makes graceful degradation work: a C1 reader
    /// loads an exotically-quantized model by the slower core-only path.
    pub fn with_fallbacks(&self, known: &dyn Fn(&str, &str, u64) -> bool) -> Expr {
        let mapped = self.map_children(&|c| c.with_fallbacks(known));
        if let Expr::Plugin {
            ns,
            name,
            v,
            fallback: Some(f),
            ..
        } = &mapped
        {
            if !known(ns, name, *v) {
                return (**f).clone();
            }
        }
        mapped
    }
}

fn wider(a: &DType, b: &DType) -> DType {
    // Elementwise arithmetic happens in the wider of the two operand types;
    // ties keep the left operand, so the result is a function of the inputs
    // and not of evaluation order.
    if b.bits() > a.bits() {
        b.clone()
    } else {
        a.clone()
    }
}

// ----------------------------------------------------------- child rewriting --

impl Expr {
    /// Applies `f` to every immediate child, rebuilding the node.
    pub fn map_children(&self, f: &dyn Fn(&Expr) -> Expr) -> Expr {
        let b = |e: &Expr| Box::new(f(e));
        match self {
            Expr::Literal { .. }
            | Expr::Extern { .. }
            | Expr::Full { .. }
            | Expr::Arange { .. }
            | Expr::Eye { .. }
            | Expr::Random { .. } => self.clone(),
            Expr::Reshape { x, shape } => Expr::Reshape {
                x: b(x),
                shape: shape.clone(),
            },
            Expr::Permute { x, perm } => Expr::Permute {
                x: b(x),
                perm: perm.clone(),
            },
            Expr::Squeeze { x, axes } => Expr::Squeeze {
                x: b(x),
                axes: axes.clone(),
            },
            Expr::Expand { x, shape } => Expr::Expand {
                x: b(x),
                shape: shape.clone(),
            },
            Expr::Slice {
                x,
                starts,
                sizes,
                steps,
            } => Expr::Slice {
                x: b(x),
                starts: starts.clone(),
                sizes: sizes.clone(),
                steps: steps.clone(),
            },
            Expr::Concat { xs, axis } => Expr::Concat {
                xs: xs.iter().map(f).collect(),
                axis: *axis,
            },
            Expr::Split {
                x,
                axis,
                sizes,
                pick,
            } => Expr::Split {
                x: b(x),
                axis: *axis,
                sizes: sizes.clone(),
                pick: *pick,
            },
            Expr::Pad {
                x,
                pads,
                mode,
                value,
            } => Expr::Pad {
                x: b(x),
                pads: pads.clone(),
                mode: *mode,
                value: *value,
            },
            Expr::Gather { x, idx, axis } => Expr::Gather {
                x: b(x),
                idx: b(idx),
                axis: *axis,
            },
            Expr::Relayout { x, layout } => Expr::Relayout {
                x: b(x),
                layout: layout.clone(),
            },
            Expr::Cast { x, dtype, round } => Expr::Cast {
                x: b(x),
                dtype: dtype.clone(),
                round: *round,
            },
            Expr::Bin { op, a, b: bb } => Expr::Bin {
                op: *op,
                a: b(a),
                b: b(bb),
            },
            Expr::Scale { x, k } => Expr::Scale { x: b(x), k: *k },
            Expr::MatMul { a, b: bb, sum } => Expr::MatMul {
                a: b(a),
                b: b(bb),
                sum: *sum,
            },
            Expr::Norm { x, axis, p, sum } => Expr::Norm {
                x: b(x),
                axis: *axis,
                p: *p,
                sum: *sum,
            },
            Expr::Clamp { x, lo, hi } => Expr::Clamp {
                x: b(x),
                lo: *lo,
                hi: *hi,
            },
            Expr::Dequantize { x, scheme } => Expr::Dequantize {
                x: b(x),
                scheme: scheme.clone(),
            },
            Expr::Quantize { x, scheme, round } => Expr::Quantize {
                x: b(x),
                scheme: scheme.clone(),
                round: *round,
            },
            Expr::Sparse {
                scheme,
                parts,
                attrs,
                shape,
                dtype,
                fill,
            } => Expr::Sparse {
                scheme: scheme.clone(),
                parts: parts.iter().map(|(k, e)| (k.clone(), f(e))).collect(),
                attrs: attrs.clone(),
                shape: shape.clone(),
                dtype: dtype.clone(),
                fill: *fill,
            },
            Expr::Approx { x, bound } => Expr::Approx {
                x: b(x),
                bound: *bound,
            },
            Expr::Delta { base, patch, op } => Expr::Delta {
                base: b(base),
                patch: b(patch),
                op: *op,
            },
            Expr::Select { feature, a, b: bb } => Expr::Select {
                feature: feature.clone(),
                a: b(a),
                b: b(bb),
            },
            Expr::Plugin {
                ns,
                name,
                v,
                args,
                attrs,
                crit,
                shape,
                dtype,
                fallback,
            } => Expr::Plugin {
                ns: ns.clone(),
                name: name.clone(),
                v: *v,
                args: args.iter().map(f).collect(),
                attrs: attrs.clone(),
                crit: *crit,
                shape: shape.clone(),
                dtype: dtype.clone(),
                fallback: fallback.as_ref().map(|e| Box::new(f(e))),
            },
        }
    }
}

// ------------------------------------------------------------------ encoding --

impl Expr {
    /// The canonical CBOR form of this node (§03.2 rules apply to the encoding;
    /// [`Value::encode`] enforces them).
    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![("op", Value::text(self.op()))];
        match self {
            Expr::Literal {
                chunks,
                dtype,
                shape,
                layout,
            } => {
                p.push(("chunks", ref_value(chunks)));
                p.push(("dtype", dtype.to_value()));
                p.push(("shape", shape_to_value(shape)));
                if layout != &Layout::default() {
                    p.push(("layout", layout.to_value()));
                }
            }
            Expr::Extern {
                uri,
                digest,
                dtype,
                shape,
            } => {
                p.push((
                    "uri",
                    Value::Tag(crate::cbor::TAG_URI, Box::new(Value::text(uri.clone()))),
                ));
                if let Some(d) = digest {
                    p.push(("digest", Value::Bytes(d.to_vec())));
                }
                p.push(("dtype", dtype.to_value()));
                p.push(("shape", shape_to_value(shape)));
            }
            Expr::Full {
                value,
                dtype,
                shape,
            } => {
                p.push(("value", value.to_value()));
                p.push(("dtype", dtype.to_value()));
                p.push(("shape", shape_to_value(shape)));
            }
            Expr::Arange {
                start,
                step,
                dtype,
                shape,
            } => {
                p.push(("start", start.to_value()));
                p.push(("step", step.to_value()));
                p.push(("dtype", dtype.to_value()));
                p.push(("shape", shape_to_value(shape)));
            }
            Expr::Eye { rows, cols, dtype } => {
                p.push(("rows", Value::U(*rows)));
                p.push(("cols", Value::U(*cols)));
                p.push(("dtype", dtype.to_value()));
            }
            Expr::Random {
                dist,
                seed,
                dtype,
                shape,
            } => {
                p.push((
                    "dist",
                    match dist {
                        Dist::Uniform { lo, hi } => Value::map(vec![
                            ("k", Value::text("uniform")),
                            ("lo", Value::F64(*lo)),
                            ("hi", Value::F64(*hi)),
                        ]),
                        Dist::Normal { mean, std } => Value::map(vec![
                            ("k", Value::text("normal")),
                            ("mean", Value::F64(*mean)),
                            ("std", Value::F64(*std)),
                        ]),
                    },
                ));
                p.push(("seed", Value::U(*seed)));
                p.push(("dtype", dtype.to_value()));
                p.push(("shape", shape_to_value(shape)));
            }
            Expr::Reshape { x, shape } | Expr::Expand { x, shape } => {
                p.push(("x", x.to_value()));
                p.push(("shape", shape_to_value(shape)));
            }
            Expr::Permute { x, perm } => {
                p.push(("x", x.to_value()));
                p.push((
                    "perm",
                    Value::Array(perm.iter().map(|i| Value::U(*i as u64)).collect()),
                ));
            }
            Expr::Squeeze { x, axes } => {
                p.push(("x", x.to_value()));
                p.push((
                    "axes",
                    Value::Array(axes.iter().map(|i| Value::U(*i as u64)).collect()),
                ));
            }
            Expr::Slice {
                x,
                starts,
                sizes,
                steps,
            } => {
                p.push(("x", x.to_value()));
                p.push(("starts", uarray(starts)));
                p.push(("sizes", uarray(sizes)));
                if steps.iter().any(|s| *s != 1) {
                    p.push(("steps", uarray(steps)));
                }
            }
            Expr::Concat { xs, axis } => {
                p.push((
                    "xs",
                    Value::Array(xs.iter().map(|x| x.to_value()).collect()),
                ));
                p.push(("axis", Value::U(*axis as u64)));
            }
            Expr::Split {
                x,
                axis,
                sizes,
                pick,
            } => {
                p.push(("x", x.to_value()));
                p.push(("axis", Value::U(*axis as u64)));
                p.push(("sizes", uarray(sizes)));
                p.push(("pick", Value::U(*pick as u64)));
            }
            Expr::Pad {
                x,
                pads,
                mode,
                value,
            } => {
                p.push(("x", x.to_value()));
                p.push((
                    "pads",
                    Value::Array(
                        pads.iter()
                            .map(|(a, b)| Value::Array(vec![Value::U(*a), Value::U(*b)]))
                            .collect(),
                    ),
                ));
                p.push((
                    "mode",
                    Value::text(match mode {
                        PadMode::Constant => "constant",
                        PadMode::Edge => "edge",
                        PadMode::Reflect => "reflect",
                        PadMode::Wrap => "wrap",
                    }),
                ));
                if *mode == PadMode::Constant {
                    p.push(("value", value.to_value()));
                }
            }
            Expr::Gather { x, idx, axis } => {
                p.push(("x", x.to_value()));
                p.push(("idx", idx.to_value()));
                p.push(("axis", Value::U(*axis as u64)));
            }
            Expr::Relayout { x, layout } => {
                p.push(("x", x.to_value()));
                p.push(("layout", layout.to_value()));
            }
            Expr::Cast { x, dtype, round } => {
                p.push(("x", x.to_value()));
                p.push(("dtype", dtype.to_value()));
                p.push(("rounding", Value::text(round.name())));
                if let Round::Stochastic { seed, .. } = round {
                    p.push(("seed", Value::U(*seed)));
                }
            }
            Expr::Bin { a, b, .. } => {
                p.push(("a", a.to_value()));
                p.push(("b", b.to_value()));
            }
            Expr::Scale { x, k } => {
                p.push(("x", x.to_value()));
                p.push(("k", k.to_value()));
            }
            Expr::MatMul { a, b, sum } => {
                p.push(("a", a.to_value()));
                p.push(("b", b.to_value()));
                if let Some(s) = sum.name() {
                    p.push(("sum", Value::text(s)));
                }
            }
            Expr::Norm {
                x,
                axis,
                p: ord,
                sum,
            } => {
                p.push(("x", x.to_value()));
                p.push(("axis", Value::U(*axis as u64)));
                p.push(("p", Value::F64(*ord)));
                if let Some(s) = sum.name() {
                    p.push(("sum", Value::text(s)));
                }
            }
            Expr::Clamp { x, lo, hi } => {
                p.push(("x", x.to_value()));
                p.push(("lo", lo.to_value()));
                p.push(("hi", hi.to_value()));
            }
            Expr::Dequantize { x, scheme } => {
                p.push(("x", x.to_value()));
                p.push(("scheme", scheme.clone()));
            }
            Expr::Quantize { x, scheme, round } => {
                p.push(("x", x.to_value()));
                p.push(("scheme", scheme.clone()));
                p.push(("rounding", Value::text(round.name())));
            }
            Expr::Sparse {
                scheme,
                parts,
                attrs,
                shape,
                dtype,
                fill,
            } => {
                p.push(("scheme", Value::text(scheme.clone())));
                for (k, e) in parts {
                    p.push((leak(k), e.to_value()));
                }
                if let Value::Map(m) = attrs {
                    for (k, v) in m {
                        if let Some(k) = k.as_str() {
                            p.push((leak(k), v.clone()));
                        }
                    }
                }
                p.push(("shape", shape_to_value(shape)));
                p.push(("dtype", dtype.to_value()));
                p.push(("fill", fill.to_value()));
            }
            Expr::Approx { x, bound } => {
                p.push(("x", x.to_value()));
                p.push((
                    "bound",
                    match bound {
                        Bound::Abs(v) => Value::map(vec![
                            ("mode", Value::text("abs")),
                            ("bound", Value::F64(*v)),
                        ]),
                        Bound::Rel(v) => Value::map(vec![
                            ("mode", Value::text("rel")),
                            ("bound", Value::F64(*v)),
                        ]),
                        Bound::Psnr(v) => Value::map(vec![
                            ("mode", Value::text("psnr")),
                            ("bound", Value::F64(*v)),
                        ]),
                    },
                ));
            }
            Expr::Delta { base, patch, op } => {
                p.push(("base", base.to_value()));
                p.push(("patch", patch.to_value()));
                p.push((
                    "dop",
                    Value::text(match op {
                        DeltaOp::Add => "add",
                        DeltaOp::Xor => "xor",
                        DeltaOp::Replace => "replace",
                        DeltaOp::SparseAdd => "sparse-add",
                    }),
                ));
            }
            Expr::Select { feature, a, b } => {
                p.push(("feature", Value::text(feature.clone())));
                p.push(("a", a.to_value()));
                p.push(("b", b.to_value()));
            }
            Expr::Plugin {
                ns,
                name,
                v,
                args,
                attrs,
                crit,
                shape,
                dtype,
                fallback,
            } => {
                p.push(("ns", Value::text(ns.clone())));
                p.push(("name", Value::text(name.clone())));
                p.push(("v", Value::U(*v)));
                p.push((
                    "args",
                    Value::Array(args.iter().map(|a| a.to_value()).collect()),
                ));
                p.push(("attrs", attrs.clone()));
                p.push(("crit", Value::Bool(*crit)));
                p.push(("shape", shape_to_value(shape)));
                p.push(("dtype", dtype.to_value()));
                if let Some(f) = fallback {
                    p.push(("fallback", f.to_value()));
                }
            }
        }
        Value::map(p)
    }

    /// Parses a node. Unknown `op` values are an error: the node set is closed,
    /// and an extension must announce itself as a `plugin` (§04.7.1 clause 3).
    pub fn from_value(v: &Value) -> Res<Expr> {
        Expr::parse_at(v, 0)
    }

    fn parse_at(v: &Value, depth: usize) -> Res<Expr> {
        if depth > MAX_DEPTH {
            return Err(Error::Bounds(format!(
                "expression nesting exceeds {MAX_DEPTH} (R-T05)"
            )));
        }
        let v = match v {
            Value::Tag(crate::cbor::TAG_EXPR, inner) => inner.as_ref(),
            other => other,
        };
        let op = v
            .get("op")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::Type("expression node has no `op`".into()))?;
        let sub = |key: &'static str| -> Res<Box<Expr>> {
            let c = v
                .get(key)
                .ok_or_else(|| Error::Type(format!("{op}: missing `{key}`")))?;
            Ok(Box::new(Expr::parse_at(c, depth + 1)?))
        };
        let dtype = |key: &'static str| -> Res<DType> {
            DType::from_value(
                v.get(key)
                    .ok_or_else(|| Error::Type(format!("{op}: missing `{key}`")))?,
            )
            .map_err(Error::Type)
        };
        let shape = |key: &'static str| -> Res<Shape> {
            parse_shape_value(
                v.get(key)
                    .ok_or_else(|| Error::Type(format!("{op}: missing `{key}`")))?,
            )
        };
        let uvec = |key: &str| -> Res<Vec<u64>> {
            v.get(key)
                .and_then(|x| x.as_array())
                .ok_or_else(|| Error::Type(format!("{op}: `{key}` must be an array")))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .ok_or_else(|| Error::Type(format!("{op}: `{key}` entries must be uint")))
                })
                .collect()
        };
        let scalar = |key: &str, default: Scalar| -> Res<Scalar> {
            match v.get(key) {
                Some(x) => Scalar::from_value(x),
                None => Ok(default),
            }
        };
        let axis = |key: &str| -> Res<usize> {
            Ok(v.get(key)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| Error::Type(format!("{op}: `{key}` must be a uint")))?
                as usize)
        };
        let round = || -> Res<Round> {
            let name = v.get("rounding").and_then(|x| x.as_str()).unwrap_or("rne");
            let r = Round::parse(name)
                .ok_or_else(|| Error::Type(format!("{op}: unknown rounding `{name}`")))?;
            Ok(match r {
                Round::Stochastic { .. } => Round::Stochastic {
                    seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(0),
                    index: 0,
                },
                other => other,
            })
        };
        let sum = || -> Res<Sum> {
            match v.get("sum").and_then(|x| x.as_str()) {
                Some(s) => {
                    Sum::parse(s).ok_or_else(|| Error::Type(format!("{op}: unknown sum `{s}`")))
                }
                None => Ok(Sum::Unspecified),
            }
        };
        Ok(match op {
            "literal" => Expr::Literal {
                chunks: parse_ref_value(
                    v.get("chunks")
                        .ok_or_else(|| Error::Type("literal: missing `chunks`".into()))?,
                )?,
                dtype: dtype("dtype")?,
                shape: shape("shape")?,
                layout: match v.get("layout") {
                    Some(l) => Layout::from_value(l).map_err(Error::Type)?,
                    None => Layout::default(),
                },
            },
            "extern" => Expr::Extern {
                uri: {
                    let u = v
                        .get("uri")
                        .ok_or_else(|| Error::Type("extern: missing `uri`".into()))?;
                    let u = match u {
                        Value::Tag(crate::cbor::TAG_URI, inner) => inner.as_ref(),
                        other => other,
                    };
                    u.as_str()
                        .ok_or_else(|| Error::Type("extern: `uri` must be text".into()))?
                        .to_string()
                },
                digest: v
                    .get("digest")
                    .and_then(|x| x.as_bytes())
                    .and_then(|b| b.try_into().ok()),
                dtype: dtype("dtype")?,
                shape: shape("shape")?,
            },
            "zeros" | "ones" | "full" => Expr::Full {
                value: match op {
                    "zeros" => Scalar::Int(0),
                    "ones" => Scalar::Int(1),
                    _ => scalar("value", Scalar::Int(0))?,
                },
                dtype: dtype("dtype")?,
                shape: shape("shape")?,
            },
            "arange" => Expr::Arange {
                start: scalar("start", Scalar::Int(0))?,
                step: scalar("step", Scalar::Int(1))?,
                dtype: dtype("dtype")?,
                shape: shape("shape")?,
            },
            "eye" => Expr::Eye {
                rows: v
                    .get("rows")
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| Error::Type("eye: missing `rows`".into()))?,
                cols: v
                    .get("cols")
                    .and_then(|x| x.as_u64())
                    .or_else(|| v.get("rows").and_then(|x| x.as_u64()))
                    .ok_or_else(|| Error::Type("eye: missing `cols`".into()))?,
                dtype: dtype("dtype")?,
            },
            "random" => {
                let d = v
                    .get("dist")
                    .ok_or_else(|| Error::Type("random: missing `dist`".into()))?;
                let f = |k: &str, def: f64| d.get(k).and_then(as_f64).unwrap_or(def);
                Expr::Random {
                    dist: match d.get("k").and_then(|x| x.as_str()) {
                        Some("uniform") => Dist::Uniform {
                            lo: f("lo", 0.0),
                            hi: f("hi", 1.0),
                        },
                        Some("normal") => Dist::Normal {
                            mean: f("mean", 0.0),
                            std: f("std", 1.0),
                        },
                        other => {
                            return Err(Error::Type(format!(
                                "random: unknown distribution {other:?}"
                            )))
                        }
                    },
                    seed: v.get("seed").and_then(|x| x.as_u64()).unwrap_or(0),
                    dtype: dtype("dtype")?,
                    shape: shape("shape")?,
                }
            }
            "reshape" => Expr::Reshape {
                x: sub("x")?,
                shape: shape("shape")?,
            },
            "transpose" => {
                // Sugar for the rank-2 permutation, and normalized to it so
                // that two spellings share one identity.
                let x = sub("x")?;
                let rank = x.infer_inner()?.shape.len();
                if rank < 2 {
                    return Err(Error::Type(
                        "transpose: operand must be at least 2-D".into(),
                    ));
                }
                let mut perm: Vec<usize> = (0..rank).collect();
                perm.swap(rank - 2, rank - 1);
                Expr::Permute { x, perm }
            }
            "permute" => Expr::Permute {
                x: sub("x")?,
                perm: uvec("perm")?.into_iter().map(|i| i as usize).collect(),
            },
            "squeeze" => Expr::Squeeze {
                x: sub("x")?,
                axes: uvec("axes")?.into_iter().map(|i| i as usize).collect(),
            },
            "expand" => Expr::Expand {
                x: sub("x")?,
                shape: shape("shape")?,
            },
            "slice" => {
                let starts = uvec("starts")?;
                let sizes = uvec("sizes")?;
                let steps = match v.get("steps") {
                    Some(_) => uvec("steps")?,
                    None => vec![1; starts.len()],
                };
                Expr::Slice {
                    x: sub("x")?,
                    starts,
                    sizes,
                    steps,
                }
            }
            "concat" => Expr::Concat {
                xs: v
                    .get("xs")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| Error::Type("concat: `xs` must be an array".into()))?
                    .iter()
                    .map(|x| Expr::parse_at(x, depth + 1))
                    .collect::<Res<Vec<Expr>>>()?,
                axis: axis("axis")?,
            },
            "split" => Expr::Split {
                x: sub("x")?,
                axis: axis("axis")?,
                sizes: uvec("sizes")?,
                pick: axis("pick")?,
            },
            "pad" => Expr::Pad {
                x: sub("x")?,
                pads: v
                    .get("pads")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| Error::Type("pad: `pads` must be an array".into()))?
                    .iter()
                    .map(|p| {
                        let a = p
                            .as_array()
                            .ok_or_else(|| Error::Type("pad: each pad is a pair".into()))?;
                        Ok((
                            a.first().and_then(|x| x.as_u64()).unwrap_or(0),
                            a.get(1).and_then(|x| x.as_u64()).unwrap_or(0),
                        ))
                    })
                    .collect::<Res<Vec<(u64, u64)>>>()?,
                mode: match v.get("mode").and_then(|x| x.as_str()) {
                    Some("constant") | None => PadMode::Constant,
                    Some("edge") => PadMode::Edge,
                    Some("reflect") => PadMode::Reflect,
                    Some("wrap") => PadMode::Wrap,
                    Some(m) => return Err(Error::Type(format!("pad: unknown mode `{m}`"))),
                },
                value: scalar("value", Scalar::Int(0))?,
            },
            "gather" => Expr::Gather {
                x: sub("x")?,
                idx: sub("idx")?,
                axis: axis("axis")?,
            },
            "relayout" => Expr::Relayout {
                x: sub("x")?,
                layout: Layout::from_value(
                    v.get("layout")
                        .ok_or_else(|| Error::Type("relayout: missing `layout`".into()))?,
                )
                .map_err(Error::Type)?,
            },
            "cast" => Expr::Cast {
                x: sub("x")?,
                dtype: dtype("dtype")?,
                round: round()?,
            },
            "add" | "sub" | "mul" | "div" => Expr::Bin {
                op: match op {
                    "add" => BinOp::Add,
                    "sub" => BinOp::Sub,
                    "mul" => BinOp::Mul,
                    _ => BinOp::Div,
                },
                a: sub("a")?,
                b: sub("b")?,
            },
            "scale" => Expr::Scale {
                x: sub("x")?,
                k: scalar("k", Scalar::Int(1))?,
            },
            "matmul" => Expr::MatMul {
                a: sub("a")?,
                b: sub("b")?,
                sum: sum()?,
            },
            "norm" => Expr::Norm {
                x: sub("x")?,
                axis: axis("axis")?,
                p: v.get("p").and_then(as_f64).unwrap_or(2.0),
                sum: sum()?,
            },
            "clamp" => Expr::Clamp {
                x: sub("x")?,
                lo: scalar("lo", Scalar::Float(f64::NEG_INFINITY))?,
                hi: scalar("hi", Scalar::Float(f64::INFINITY))?,
            },
            "dequantize" => Expr::Dequantize {
                x: sub("x")?,
                scheme: v
                    .get("scheme")
                    .cloned()
                    .ok_or_else(|| Error::Type("dequantize: missing `scheme`".into()))?,
            },
            "quantize" => Expr::Quantize {
                x: sub("x")?,
                scheme: v
                    .get("scheme")
                    .cloned()
                    .ok_or_else(|| Error::Type("quantize: missing `scheme`".into()))?,
                round: round()?,
            },
            "sparse" => {
                let scheme = v
                    .get("scheme")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error::Type("sparse: missing `scheme`".into()))?
                    .to_string();
                // Component tensors are whichever of the known part names are
                // present; everything else is an attribute.
                const PART_KEYS: &[&str] = &[
                    "values", "indices", "indptr", "mask", "offsets", "blocks", "index",
                ];
                let mut parts = Vec::new();
                let mut attrs: Vec<(Value, Value)> = Vec::new();
                for (k, val) in v.as_map().unwrap_or(&[]) {
                    let Some(k) = k.as_str() else { continue };
                    if PART_KEYS.contains(&k) {
                        parts.push((k.to_string(), Expr::parse_at(val, depth + 1)?));
                    } else if !matches!(k, "op" | "scheme" | "shape" | "dtype" | "fill") {
                        attrs.push((Value::text(k), val.clone()));
                    }
                }
                Expr::Sparse {
                    scheme,
                    parts,
                    attrs: Value::Map(attrs),
                    shape: shape("shape")?,
                    dtype: dtype("dtype")?,
                    fill: scalar("fill", Scalar::Int(0))?,
                }
            }
            "approx" => {
                let b = v
                    .get("bound")
                    .ok_or_else(|| Error::Type("approx: missing `bound`".into()))?;
                let val = b.get("bound").and_then(as_f64).unwrap_or(0.0);
                Expr::Approx {
                    x: sub("x")?,
                    bound: match b.get("mode").and_then(|x| x.as_str()) {
                        Some("abs") | None => Bound::Abs(val),
                        Some("rel") => Bound::Rel(val),
                        Some("psnr") => Bound::Psnr(val),
                        Some(m) => return Err(Error::Type(format!("approx: unknown mode `{m}`"))),
                    },
                }
            }
            "delta" => Expr::Delta {
                base: sub("base")?,
                patch: sub("patch")?,
                op: match v
                    .get("dop")
                    .or_else(|| v.get("op2"))
                    .and_then(|x| x.as_str())
                {
                    Some("add") | None => DeltaOp::Add,
                    Some("xor") => DeltaOp::Xor,
                    Some("replace") => DeltaOp::Replace,
                    Some("sparse-add") => DeltaOp::SparseAdd,
                    Some(o) => return Err(Error::Type(format!("delta: unknown op `{o}`"))),
                },
            },
            "select" => Expr::Select {
                feature: v
                    .get("feature")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error::Type("select: missing `feature`".into()))?
                    .to_string(),
                a: sub("a")?,
                b: sub("b")?,
            },
            "plugin" => Expr::Plugin {
                ns: v
                    .get("ns")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| Error::Type("plugin: missing `ns`".into()))?
                    .to_string(),
                name: v
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
                v: v.get("v").and_then(|x| x.as_u64()).unwrap_or(1),
                args: v
                    .get("args")
                    .and_then(|x| x.as_array())
                    .unwrap_or(&[])
                    .iter()
                    .map(|a| Expr::parse_at(a, depth + 1))
                    .collect::<Res<Vec<Expr>>>()?,
                attrs: v.get("attrs").cloned().unwrap_or(Value::Map(vec![])),
                crit: matches!(v.get("crit"), Some(Value::Bool(true))),
                shape: shape("shape")?,
                dtype: dtype("dtype")?,
                fallback: match v.get("fallback") {
                    Some(f) => Some(Box::new(Expr::parse_at(f, depth + 1)?)),
                    None => None,
                },
            },
            other => {
                return Err(Error::Unsupported(format!(
                    "unknown expression op `{other}`; the core node set is closed and \
                     extensions must be `plugin` nodes (§04.7.1)"
                )))
            }
        })
    }
}

/// The number of elements in one quantization block, from a scheme descriptor
/// (§05.1). Range pushdown needs it without knowing anything else about the
/// scheme: a block is the smallest independently-decodable unit, so a request
/// is widened to block boundaries and stays exact.
pub fn block_elems(scheme: &Value) -> Option<u64> {
    let b = scheme.get("block")?.as_array()?;
    let mut n = 1u64;
    for d in b {
        n = n.checked_mul(d.as_u64()?)?;
    }
    Some(n)
}

fn uarray(v: &[u64]) -> Value {
    Value::Array(v.iter().map(|x| Value::U(*x)).collect())
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::U(n) => Some(*n as f64),
        Value::I(n) => Some(*n as f64),
        _ => None,
    }
}

/// `Value::map` takes `&'static str` keys; sparse parts and attributes have
/// runtime names. Leaking them is bounded by the number of distinct keys in a
/// container, which is small and fixed by the schema registry.
fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

// ------------------------------------------------------- normal form, identity --

impl Expr {
    /// §04.7.5 normalization. Value-preserving by construction: every rule
    /// here either folds a structural chain over a generated constant, applies
    /// exact rational arithmetic, or reorders the operands of a commutative
    /// node.
    pub fn normalize(&self, algo: HashAlgo) -> Expr {
        let e = self.map_children(&|c| c.normalize(algo));
        match &e {
            // scale(scale(x, a), b) -> scale(x, a*b), exactly, and every
            // rational in lowest terms — so a publisher writing alpha/r as
            // 32/16 and one writing 2 produce the same identity.
            Expr::Scale { x, k } => {
                let k = k.reduced();
                match x.as_ref() {
                    Expr::Scale { x: inner, k: k2 } => Expr::Scale {
                        x: inner.clone(),
                        k: k.times(*k2),
                    },
                    _ if k == Scalar::Int(1) => (**x).clone(),
                    _ => Expr::Scale { x: x.clone(), k },
                }
            }
            // permute(permute(x, p), q) -> permute(x, p∘q); the identity
            // permutation disappears, which is how transpose(transpose(x))
            // collapses.
            Expr::Permute { x, perm } => {
                let composed = match x.as_ref() {
                    Expr::Permute { x: inner, perm: p2 } => Some((inner.clone(), {
                        let mut out = Vec::with_capacity(perm.len());
                        for i in perm {
                            out.push(p2[*i]);
                        }
                        out
                    })),
                    _ => None,
                };
                let (base, perm) = match composed {
                    Some((b, p)) => (b, p),
                    None => (x.clone(), perm.clone()),
                };
                if perm.iter().enumerate().all(|(i, p)| i == *p) {
                    (*base).clone()
                } else {
                    Expr::Permute { x: base, perm }
                }
            }
            // cast(cast(x, T), T) -> cast(x, T): provably lossless, because
            // the inner cast already landed in T.
            Expr::Cast { x, dtype, round } => match x.as_ref() {
                Expr::Cast {
                    x: inner,
                    dtype: d2,
                    ..
                } if d2 == dtype => Expr::Cast {
                    x: inner.clone(),
                    dtype: dtype.clone(),
                    round: *round,
                },
                _ => e.clone(),
            },
            // Commutative operands sort by sub-digest, so `add(a, b)` and
            // `add(b, a)` are one value with one identity.
            Expr::Bin {
                op: op @ (BinOp::Add | BinOp::Mul),
                a,
                b,
            } => {
                let (da, db) = (a.identity(algo), b.identity(algo));
                if db < da {
                    Expr::Bin {
                        op: *op,
                        a: b.clone(),
                        b: a.clone(),
                    }
                } else {
                    e.clone()
                }
            }
            // Structural chains over generated constants fold away; they touch
            // no stored bytes, so folding cannot change a value.
            Expr::Reshape { x, shape } | Expr::Expand { x, shape } => match x.as_ref() {
                Expr::Full { value, dtype, .. } => Expr::Full {
                    value: *value,
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                },
                _ => e.clone(),
            },
            _ => e.clone(),
        }
    }

    /// The expression's identity: the digest of its canonical encoding after
    /// normalization, domain-separated with `omni/1.0 expr-identity` (§03.5.3).
    ///
    /// This is the cache key of §04.7.4 clause 3 and the dedup key of §04.7.5.
    pub fn identity(&self, algo: HashAlgo) -> Digest {
        let bytes = self.normalize(algo).to_value().encode();
        algo.domain_digest("omni/1.0 expr-identity", &bytes)
    }
}

// ------------------------------------------------------------------- tensors --

/// A materialized tensor: dense, row-major, `f64` elements.
///
/// A production evaluator works in the target dtype and in blocks. This one
/// trades that for legibility: the semantics of every node are visible in one
/// place, which is what a reference implementation is for.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<u64>, dtype: DType, data: Vec<f64>) -> Tensor {
        Tensor { shape, dtype, data }
    }

    pub fn numel(&self) -> u64 {
        numel(&self.shape)
    }

    pub fn strides(&self) -> Vec<u64> {
        crate::layout::Order::RowMajor.strides(&self.shape)
    }

    /// Reads a tensor out of stored bytes through its dtype and layout.
    pub fn from_bytes(bytes: &[u8], dtype: &DType, layout: &Layout, shape: &[u64]) -> Res<Tensor> {
        if !dtype.is_numeric() {
            return Err(Error::Unsupported(format!(
                "dtype {} has no element semantics; only literal, slice and \
                 cast-to-opaque are defined for it (§04.3.5)",
                dtype.label()
            )));
        }
        let n = numel(shape);
        let mut data = Vec::with_capacity(n as usize);
        let mut idx = vec![0u64; shape.len()];
        for _ in 0..n {
            data.push(
                read_element(bytes, dtype, layout, shape, &idx).ok_or_else(|| {
                    Error::Bounds(format!(
                        "literal is {} bytes, too short for {:?} of {}",
                        bytes.len(),
                        shape,
                        dtype.label()
                    ))
                })?,
            );
            bump(&mut idx, shape);
        }
        Ok(Tensor {
            shape: shape.to_vec(),
            dtype: dtype.clone(),
            data,
        })
    }

    /// Writes this tensor into stored bytes under a dtype and layout — the
    /// inverse of [`Tensor::from_bytes`], and what `omni convert` needs.
    pub fn to_bytes(&self, dtype: &DType, layout: &Layout, round: Round) -> Res<Vec<u8>> {
        let size = layout.stored_bytes(&self.shape, dtype).ok_or_else(|| {
            Error::Unsupported(format!(
                "layout {} cannot size a {} tensor",
                layout.kind(),
                dtype.label()
            ))
        })?;
        let mut out = vec![0u8; size as usize];
        let mut idx = vec![0u64; self.shape.len()];
        for (i, x) in self.data.iter().enumerate() {
            let r = match round {
                Round::Stochastic { seed, .. } => Round::Stochastic {
                    seed,
                    index: i as u64,
                },
                other => other,
            };
            if !write_element(&mut out, dtype, layout, &self.shape, &idx, *x, r) {
                return Err(Error::Bounds(format!(
                    "cannot place element {idx:?} of a {} tensor in a {} layout",
                    dtype.label(),
                    layout.kind()
                )));
            }
            bump(&mut idx, &self.shape);
        }
        Ok(out)
    }

    /// Reads one element by multi-index, or `None` when out of range.
    pub fn get(&self, index: &[u64]) -> Option<f64> {
        if index.len() != self.shape.len() {
            return None;
        }
        let s = self.strides();
        let mut lin = 0u64;
        for ((i, st), d) in index.iter().zip(&s).zip(&self.shape) {
            if i >= d {
                return None;
            }
            lin += i * st;
        }
        self.data.get(lin as usize).copied()
    }

    fn at(&self, index: &[u64]) -> f64 {
        let s = self.strides();
        let mut lin = 0u64;
        for (i, st) in index.iter().zip(&s) {
            lin += i * st;
        }
        self.data[lin as usize]
    }

    /// Reads with NumPy broadcasting against `shape`.
    fn broadcast_at(&self, out_index: &[u64]) -> f64 {
        let off = out_index.len() - self.shape.len();
        let mut idx = Vec::with_capacity(self.shape.len());
        for (k, d) in self.shape.iter().enumerate() {
            idx.push(if *d == 1 { 0 } else { out_index[off + k] });
        }
        self.at(&idx)
    }
}

fn bump(idx: &mut [u64], shape: &[u64]) {
    for k in (0..idx.len()).rev() {
        idx[k] += 1;
        if idx[k] < shape[k] {
            return;
        }
        idx[k] = 0;
    }
}

fn read_element(
    bytes: &[u8],
    dtype: &DType,
    layout: &Layout,
    shape: &[u64],
    index: &[u64],
) -> Option<f64> {
    match dtype.bits_rational() {
        (_, 1) => {
            let bit = layout.bit_offset(shape, dtype, index)?;
            dtype.decode_bits(bytes, u64::try_from(bit).ok()?)
        }
        // Fractional-width types are packed in groups, so they are addressed by
        // element index rather than by bit position.
        _ => dtype.decode(bytes, layout.linear(shape, index)?),
    }
}

fn write_element(
    bytes: &mut [u8],
    dtype: &DType,
    layout: &Layout,
    shape: &[u64],
    index: &[u64],
    x: f64,
    round: Round,
) -> bool {
    match dtype.bits_rational() {
        (_, 1) => {
            let Some(bit) = layout.bit_offset(shape, dtype, index) else {
                return false;
            };
            let Ok(bit) = u64::try_from(bit) else {
                return false;
            };
            dtype.encode_bits(bytes, bit, x, round)
        }
        _ => match layout.linear(shape, index) {
            Some(lin) => dtype.encode(bytes, lin, x, round),
            None => false,
        },
    }
}

// ---------------------------------------------------------------- evaluation --

/// What an evaluator needs from the outside world: stored bytes, and the
/// runtime's answers to capability questions.
pub struct Ctx<'a> {
    store: &'a dyn Store,
    /// Answers for `select` nodes (§10.3).
    pub features: BTreeMap<String, bool>,
    /// Hard cap on elements materialized per node (§12.4: no allocation driven
    /// by an unvalidated declared size).
    pub max_elems: u64,
    /// Plugin implementations this evaluator has.
    pub plugins: Vec<String>,
}

impl<'a> Ctx<'a> {
    pub fn new(store: &'a dyn Store) -> Ctx<'a> {
        Ctx {
            store,
            features: BTreeMap::new(),
            max_elems: 1 << 28,
            plugins: Vec::new(),
        }
    }

    pub fn feature(mut self, name: &str, on: bool) -> Self {
        self.features.insert(name.to_string(), on);
        self
    }

    pub fn max_elems(mut self, n: u64) -> Self {
        self.max_elems = n;
        self
    }

    pub fn store(&self) -> &dyn Store {
        self.store
    }

    /// Reads an object's bytes.
    pub fn bytes(&self, d: &Digest) -> Res<Vec<u8>> {
        self.store.resolve(d)?.ok_or(Error::Missing(*d))
    }

    /// Reads a structure object.
    pub fn value(&self, d: &Digest) -> Res<Value> {
        let b = self.bytes(d)?;
        crate::cbor::decode(&b).map_err(|e| Error::Store(e.to_string()))
    }

    /// The logical bytes behind a `literal`'s `chunks` ref: either a
    /// `ChunkList` (§04.5) whose chunks are concatenated, or a bare `Blob` for
    /// a single-chunk tensor.
    pub fn chunk_bytes(&self, r: &Ref) -> Res<Vec<u8>> {
        if r.0 == otype::BLOB {
            return self.bytes(&r.1);
        }
        let cl = self.value(&r.1)?;
        let total = cl.get("total").and_then(|x| x.as_u64()).unwrap_or(0);
        if total > self.max_elems.saturating_mul(8) {
            return Err(Error::Bounds(format!(
                "ChunkList declares {total} bytes, above this evaluator's cap"
            )));
        }
        let chunks = cl
            .get("chunks")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::Type("ChunkList has no `chunks`".into()))?;
        let mut out = Vec::with_capacity(total as usize);
        for c in chunks {
            let cr = parse_ref_value(
                c.get("r")
                    .ok_or_else(|| Error::Type("chunk entry has no `r`".into()))?,
            )?;
            let n = c.get("n").and_then(|x| x.as_u64());
            let b = self.bytes(&cr.1)?;
            if let Some(n) = n {
                if b.len() as u64 != n {
                    return Err(Error::Bounds(format!(
                        "chunk declares {n} logical bytes but holds {}",
                        b.len()
                    )));
                }
            }
            out.extend_from_slice(&b);
        }
        if total != 0 && out.len() as u64 != total {
            return Err(Error::Bounds(format!(
                "ChunkList total is {total} but its chunks hold {} bytes (R-T02)",
                out.len()
            )));
        }
        Ok(out)
    }
}

impl Expr {
    /// Materializes the whole value.
    pub fn eval(&self, ctx: &Ctx<'_>) -> Res<Tensor> {
        let t = self.infer()?;
        let shape = concrete(&t.shape).ok_or_else(|| {
            Error::Type(
                "symbolic dimensions must be bound through the model's `dims` table before \
                 materialization (§04.7.3)"
                    .into(),
            )
        })?;
        if numel(&shape) > ctx.max_elems {
            return Err(Error::Bounds(format!(
                "{} elements exceeds this evaluator's cap of {}",
                numel(&shape),
                ctx.max_elems
            )));
        }
        self.eval_inner(ctx, &shape, &t.dtype)
    }

    fn eval_inner(&self, ctx: &Ctx<'_>, shape: &[u64], dtype: &DType) -> Res<Tensor> {
        let n = numel(shape);
        Ok(match self {
            Expr::Literal {
                chunks,
                dtype: d,
                layout,
                ..
            } => {
                let bytes = ctx.chunk_bytes(chunks)?;
                Tensor::from_bytes(&bytes, d, layout, shape)?
            }
            Expr::Extern { uri, .. } => return Err(Error::External(uri.clone())),
            Expr::Full { value, .. } => Tensor::new(
                shape.to_vec(),
                dtype.clone(),
                vec![value.as_f64(); n as usize],
            ),
            Expr::Arange { start, step, .. } => Tensor::new(
                shape.to_vec(),
                dtype.clone(),
                (0..n)
                    .map(|i| start.as_f64() + step.as_f64() * i as f64)
                    .collect(),
            ),
            Expr::Eye { rows, cols, .. } => {
                let mut data = vec![0.0; (rows * cols) as usize];
                for i in 0..(*rows).min(*cols) {
                    data[(i * cols + i) as usize] = 1.0;
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Random { dist, seed, .. } => Tensor::new(
                shape.to_vec(),
                dtype.clone(),
                (0..n).map(|i| random_at(*dist, *seed, i)).collect(),
            ),
            // Reshape, expand and squeeze are index remappings over a
            // row-major buffer, so they are free.
            Expr::Reshape { x, .. } | Expr::Squeeze { x, .. } => {
                let t = x.eval_child(ctx)?;
                Tensor::new(shape.to_vec(), dtype.clone(), t.data)
            }
            Expr::Expand { x, .. } => {
                let t = x.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    data.push(t.broadcast_at(&idx));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Permute { x, perm } => {
                let t = x.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    // out[i] = in[perm applied backwards]
                    let mut src = vec![0u64; perm.len()];
                    for (k, p) in perm.iter().enumerate() {
                        src[*p] = idx[k];
                    }
                    data.push(t.at(&src));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Slice {
                x, starts, steps, ..
            } => {
                let t = x.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let src: Vec<u64> = idx
                        .iter()
                        .enumerate()
                        .map(|(k, i)| starts[k] + i * steps[k])
                        .collect();
                    data.push(t.at(&src));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Concat { xs, axis } => {
                let parts: Vec<Tensor> = xs
                    .iter()
                    .map(|x| x.eval_child(ctx))
                    .collect::<Res<Vec<Tensor>>>()?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let mut k = idx[*axis];
                    let mut chosen = None;
                    for p in &parts {
                        if k < p.shape[*axis] {
                            let mut src = idx.clone();
                            src[*axis] = k;
                            chosen = Some(p.at(&src));
                            break;
                        }
                        k -= p.shape[*axis];
                    }
                    data.push(chosen.unwrap_or(0.0));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Split {
                x,
                axis,
                sizes,
                pick,
            } => {
                let t = x.eval_child(ctx)?;
                let start: u64 = sizes[..*pick].iter().sum();
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let mut src = idx.clone();
                    src[*axis] += start;
                    data.push(t.at(&src));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Pad {
                x,
                pads,
                mode,
                value,
            } => {
                let t = x.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let mut src = Vec::with_capacity(idx.len());
                    let mut outside = false;
                    for (k, i) in idx.iter().enumerate() {
                        let lo = pads[k].0 as i64;
                        let d = t.shape[k] as i64;
                        let mut j = *i as i64 - lo;
                        if j < 0 || j >= d {
                            match mode {
                                PadMode::Constant => outside = true,
                                PadMode::Edge => j = j.clamp(0, d - 1),
                                PadMode::Reflect => {
                                    j = if j < 0 { -j } else { 2 * (d - 1) - j };
                                    j = j.clamp(0, d - 1);
                                }
                                PadMode::Wrap => j = j.rem_euclid(d),
                            }
                        }
                        src.push(j.max(0) as u64);
                    }
                    data.push(if outside { value.as_f64() } else { t.at(&src) });
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Gather { x, idx: sel, axis } => {
                let t = x.eval_child(ctx)?;
                let s = sel.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let pick = s.data[idx[*axis] as usize];
                    if pick < 0.0 || pick as u64 >= t.shape[*axis] {
                        return Err(Error::Bounds(format!(
                            "gather: index {pick} out of range for axis {axis} of extent {}",
                            t.shape[*axis]
                        )));
                    }
                    let mut src = idx.clone();
                    src[*axis] = pick as u64;
                    data.push(t.at(&src));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            // Relayout is a change of bit placement; in a value-level evaluator
            // it is the identity, and the layout only matters when the result
            // is written back out.
            Expr::Relayout { x, .. } => x.eval_child(ctx)?,
            Expr::Cast {
                x,
                dtype: to,
                round,
            } => {
                let t = x.eval_child(ctx)?;
                // A cast is only meaningful if it goes through the target
                // encoding, so round-trip each element rather than copying it.
                let mut data = Vec::with_capacity(t.data.len());
                let mut buf = vec![0u8; to.packed_bytes(1).max(1) as usize];
                for (i, v) in t.data.iter().enumerate() {
                    let r = match round {
                        Round::Stochastic { seed, .. } => Round::Stochastic {
                            seed: *seed,
                            index: i as u64,
                        },
                        other => *other,
                    };
                    if !to.encode(&mut buf, 0, *v, r) {
                        return Err(Error::Unsupported(format!(
                            "cast to {} is not defined element-wise",
                            to.label()
                        )));
                    }
                    data.push(to.decode(&buf, 0).unwrap_or(f64::NAN));
                }
                Tensor::new(t.shape, to.clone(), data)
            }
            Expr::Bin { op, a, b } => {
                let ta = a.eval_child(ctx)?;
                let tb = b.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    data.push(op.apply(ta.broadcast_at(&idx), tb.broadcast_at(&idx)));
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Scale { x, k } => {
                let t = x.eval_child(ctx)?;
                let f = k.as_f64();
                Tensor::new(
                    t.shape,
                    dtype.clone(),
                    t.data.iter().map(|v| v * f).collect(),
                )
            }
            Expr::MatMul { a, b, sum } => {
                let ta = a.eval_child(ctx)?;
                let tb = b.eval_child(ctx)?;
                matmul(&ta, &tb, *sum, shape, dtype)?
            }
            Expr::Norm { x, axis, p, sum } => {
                let t = x.eval_child(ctx)?;
                let mut data = Vec::with_capacity(n as usize);
                let mut idx = vec![0u64; shape.len()];
                for _ in 0..n {
                    let mut terms = Vec::with_capacity(t.shape[*axis] as usize);
                    for j in 0..t.shape[*axis] {
                        let mut src = idx.clone();
                        src[*axis] = j;
                        let v = t.at(&src).abs();
                        terms.push(if *p == 1.0 {
                            v
                        } else if *p == 2.0 {
                            v * v
                        } else {
                            v.powf(*p)
                        });
                    }
                    data.push(if p.is_infinite() {
                        terms.iter().cloned().fold(0.0f64, f64::max)
                    } else if *p == 1.0 {
                        sum.reduce(&terms)
                    } else if *p == 2.0 {
                        sum.reduce(&terms).sqrt()
                    } else {
                        sum.reduce(&terms).powf(1.0 / *p)
                    });
                    bump(&mut idx, shape);
                }
                Tensor::new(shape.to_vec(), dtype.clone(), data)
            }
            Expr::Clamp { x, lo, hi } => {
                let t = x.eval_child(ctx)?;
                let (lo, hi) = (lo.as_f64(), hi.as_f64());
                Tensor::new(
                    t.shape,
                    dtype.clone(),
                    t.data.iter().map(|v| v.clamp(lo, hi)).collect(),
                )
            }
            // The scheme catalogue of §05 is data consumed by these two nodes.
            Expr::Dequantize { x, scheme } => {
                let t = x.eval_child(ctx)?;
                crate::quant::dequantize(ctx, &t, scheme, dtype)?
            }
            Expr::Quantize { x, scheme, round } => {
                let t = x.eval_child(ctx)?;
                crate::quant::quantize(ctx, &t, scheme, dtype, *round)?
            }
            // The sparsity schemes of §04.6 are data consumed by this node.
            Expr::Sparse {
                scheme,
                parts,
                attrs,
                fill,
                ..
            } => {
                let mut mat = Vec::with_capacity(parts.len());
                for (k, e) in parts {
                    mat.push((k.as_str(), e.eval_child(ctx)?));
                }
                crate::sparse::densify(scheme, &mat, attrs, shape, dtype, fill.as_f64())?
            }
            // `approx` is a marker, not a transform: it makes the loss visible
            // without changing the value.
            Expr::Approx { x, .. } => x.eval_child(ctx)?,
            Expr::Delta { base, patch, op } => {
                let tb = base.eval_child(ctx)?;
                let tp = patch.eval_child(ctx)?;
                match op {
                    DeltaOp::Replace => tp,
                    DeltaOp::Add | DeltaOp::SparseAdd => {
                        let mut data = Vec::with_capacity(n as usize);
                        let mut idx = vec![0u64; shape.len()];
                        for _ in 0..n {
                            data.push(tb.broadcast_at(&idx) + tp.broadcast_at(&idx));
                            bump(&mut idx, shape);
                        }
                        Tensor::new(shape.to_vec(), dtype.clone(), data)
                    }
                    DeltaOp::Xor => {
                        // A bit-level patch: encode both sides, xor the bytes,
                        // decode back. This is the representation that makes a
                        // delta of two int8 tensors exact.
                        let l = Layout::default();
                        let ba = tb.to_bytes(dtype, &l, Round::Rne)?;
                        let bb = tp.to_bytes(dtype, &l, Round::Rne)?;
                        let x: Vec<u8> = ba
                            .iter()
                            .zip(bb.iter().chain(std::iter::repeat(&0)))
                            .map(|(a, b)| a ^ b)
                            .collect();
                        Tensor::from_bytes(&x, dtype, &l, shape)?
                    }
                }
            }
            Expr::Select { feature, a, b } => {
                let on = ctx.features.get(feature).copied().unwrap_or(false);
                if on {
                    a.eval_child(ctx)?
                } else {
                    b.eval_child(ctx)?
                }
            }
            Expr::Plugin {
                ns,
                name,
                v,
                fallback,
                crit,
                ..
            } => {
                let id = format!("{ns}/{name}.{v}");
                if ctx.plugins.contains(&id) {
                    return Err(Error::Unsupported(format!(
                        "plugin `{id}` is registered but this build has no host to run it (§11)"
                    )));
                }
                match fallback {
                    Some(f) => f.eval_child(ctx)?,
                    None if *crit => {
                        return Err(Error::Unsupported(format!(
                            "critical plugin `{id}` has no fallback; this tensor must be \
                             refused, but the rest of the model is still readable (§04.7.7)"
                        )))
                    }
                    None => {
                        return Err(Error::Unsupported(format!(
                            "plugin `{id}` is not implemented and declares no fallback"
                        )))
                    }
                }
            }
        })
    }

    fn eval_child(&self, ctx: &Ctx<'_>) -> Res<Tensor> {
        let t = self.infer_inner()?;
        let shape = concrete(&t.shape)
            .ok_or_else(|| Error::Type("symbolic dimension reached evaluation".into()))?;
        if numel(&shape) > ctx.max_elems {
            return Err(Error::Bounds(format!(
                "{} elements exceeds this evaluator's cap",
                numel(&shape)
            )));
        }
        self.eval_inner(ctx, &shape, &t.dtype)
    }
}

fn matmul(a: &Tensor, b: &Tensor, sum: Sum, shape: &[u64], dtype: &DType) -> Res<Tensor> {
    let (m, k) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
    let nn = b.shape[b.shape.len() - 1];
    let batch: u64 = shape[..shape.len() - 2].iter().product();
    let mut data = Vec::with_capacity((batch * m * nn) as usize);
    let mut terms = vec![0.0f64; k as usize];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..nn {
                for (p, t) in terms.iter_mut().enumerate() {
                    *t = index_batched(a, bt, i, p as u64) * index_batched(b, bt, p as u64, j);
                }
                data.push(sum.reduce(&terms));
            }
        }
    }
    Ok(Tensor::new(shape.to_vec(), dtype.clone(), data))
}

/// Reads `t[batch, i, j]`, broadcasting the batch dimensions.
fn index_batched(t: &Tensor, batch: u64, i: u64, j: u64) -> f64 {
    let rank = t.shape.len();
    let (rows, cols) = (t.shape[rank - 2], t.shape[rank - 1]);
    let batches: u64 = t.shape[..rank - 2].iter().product();
    let b = if batches == 0 { 0 } else { batch % batches };
    t.data[((b * rows + i) * cols + j) as usize]
}

/// The `random` leaf's PRNG: ChaCha20 in counter mode, keyed by the declared
/// seed. §04.7.6 requires a counter-based PRNG defined bit-exactly, and a
/// counter-based one also means element `i` is generated without generating
/// elements `0..i` — so a partial load of a random tensor is still exact.
fn random_at(dist: Dist, seed: u64, index: u64) -> f64 {
    let (a, b) = chacha20_pair(seed, index);
    match dist {
        Dist::Uniform { lo, hi } => lo + (hi - lo) * a,
        Dist::Normal { mean, std } => {
            // Box–Muller from two uniforms. Deterministic given (seed, index).
            let u1 = if a <= f64::MIN_POSITIVE {
                f64::MIN_POSITIVE
            } else {
                a
            };
            mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }
}

/// One reproducible uniform in [0, 1) from a seed and an index. Exposed because
/// §08.5's DARE draws its drop mask from a declared seed, and that draw has to
/// be the same everywhere.
pub fn uniform01(seed: u64, index: u64) -> f64 {
    chacha20_pair(seed, index).0
}

/// Two uniforms in [0, 1) from one ChaCha20 block.
fn chacha20_pair(seed: u64, index: u64) -> (f64, f64) {
    let mut key = [0u8; 32];
    key[..8].copy_from_slice(&seed.to_le_bytes());
    let block = chacha20_block(&key, index / 8, index % 8);
    let w = |i: usize| u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap()) as f64;
    (w(0) / 4294967296.0, w(1) / 4294967296.0)
}

fn chacha20_block(key: &[u8; 32], counter: u64, nonce: u64) -> [u8; 64] {
    const SIGMA: &[u8; 16] = b"expand 32-byte k";
    let mut s = [0u32; 16];
    for i in 0..4 {
        s[i] = u32::from_le_bytes(SIGMA[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes(key[i * 4..i * 4 + 4].try_into().unwrap());
    }
    s[12] = counter as u32;
    s[13] = (counter >> 32) as u32;
    s[14] = nonce as u32;
    s[15] = (nonce >> 32) as u32;
    let mut x = s;
    for _ in 0..10 {
        quarter(&mut x, 0, 4, 8, 12);
        quarter(&mut x, 1, 5, 9, 13);
        quarter(&mut x, 2, 6, 10, 14);
        quarter(&mut x, 3, 7, 11, 15);
        quarter(&mut x, 0, 5, 10, 15);
        quarter(&mut x, 1, 6, 11, 12);
        quarter(&mut x, 2, 7, 8, 13);
        quarter(&mut x, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[i * 4..i * 4 + 4].copy_from_slice(&x[i].wrapping_add(s[i]).to_le_bytes());
    }
    out
}

fn quarter(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}

// -------------------------------------------------------------- range pushdown --

/// One source a range of a value depends on.
#[derive(Clone, Debug, PartialEq)]
pub struct Dep {
    /// The `literal`'s chunk reference, or `None` for an `extern` leaf.
    pub source: Option<Ref>,
    /// The `extern` URI, when that is what this dependency is.
    pub uri: Option<String>,
    /// Byte range needed from that source, half-open.
    pub bytes: (u64, u64),
    /// False when the range is a superset of what is strictly needed — a
    /// `matmul` contraction, a data-dependent `gather`, or a layout whose bits
    /// are not monotone in the element index.
    pub exact: bool,
    /// Why, when `exact` is false.
    pub reason: Option<&'static str>,
}

impl Expr {
    /// The dependency set of a range of *logical elements* — §04.7.4 clause 1.
    ///
    /// Structural nodes have exact inverse-range functions, so reading rows
    /// 100–200 of `dequantize(literal(…))` fetches only the chunks covering
    /// those rows. `matmul` and `norm` need a whole contraction dimension, and
    /// this reports that honestly rather than pretending otherwise.
    pub fn deps(&self, elems: (u64, u64)) -> Vec<Dep> {
        let mut out = Vec::new();
        self.deps_into(elems, true, None, &mut out);
        coalesce(out)
    }

    /// The dependency set of the whole value.
    pub fn deps_all(&self) -> Vec<Dep> {
        let n = self
            .infer()
            .ok()
            .and_then(|t| concrete(&t.shape))
            .map(|s| numel(&s))
            .unwrap_or(u64::MAX);
        self.deps((0, n))
    }

    fn deps_into(
        &self,
        elems: (u64, u64),
        exact: bool,
        reason: Option<&'static str>,
        out: &mut Vec<Dep>,
    ) {
        match self {
            Expr::Literal {
                chunks,
                dtype,
                shape,
                layout,
            } => {
                let (num, den) = dtype.bits_rational();
                let shape = concrete(shape).unwrap_or_default();
                let dense = matches!(
                    layout,
                    Layout::Strided {
                        strides: None,
                        offset: 0,
                        ..
                    }
                );
                let total = layout.stored_bytes(&shape, dtype).unwrap_or(u64::MAX);
                let (mut lo, mut hi, mut exact, mut reason) = (0u64, total, exact, reason);
                if dense {
                    // Bit positions are monotone in the element index, so the
                    // byte range is exact.
                    lo = (elems.0 * num as u64) / (8 * den as u64);
                    hi = ((elems.1 * num as u64).div_ceil(8 * den as u64)).min(total);
                } else if let (Some(a), Some(b)) = (
                    bit_of_linear(layout, &shape, dtype, elems.0),
                    bit_of_linear(layout, &shape, dtype, elems.1.saturating_sub(1)),
                ) {
                    // Non-dense layouts still bound the range, and the bound is
                    // reported as a bound.
                    lo = (a.min(b) / 8) as u64;
                    hi = ((a.max(b) / 8) as u64 + dtype.packed_bytes(1) + 8).min(total);
                    exact = false;
                    reason = Some("layout bit positions are not monotone in the element index");
                }
                out.push(Dep {
                    source: Some(*chunks),
                    uri: None,
                    bytes: (lo, hi.max(lo)),
                    exact,
                    reason,
                });
            }
            Expr::Extern {
                uri, dtype, shape, ..
            } => {
                let shape = concrete(shape).unwrap_or_default();
                out.push(Dep {
                    source: None,
                    uri: Some(uri.clone()),
                    bytes: (0, dtype.packed_bytes(numel(&shape))),
                    exact: false,
                    reason: Some("extern values are never fetched implicitly"),
                });
            }
            Expr::Full { .. } | Expr::Arange { .. } | Expr::Eye { .. } | Expr::Random { .. } => {}
            // Reshape and squeeze preserve row-major element order exactly.
            Expr::Reshape { x, .. } | Expr::Squeeze { x, .. } | Expr::Relayout { x, .. } => {
                x.deps_into(elems, exact, reason, out)
            }
            // Elementwise and marker nodes pass the range straight through.
            Expr::Cast { x, .. }
            | Expr::Scale { x, .. }
            | Expr::Clamp { x, .. }
            | Expr::Approx { x, .. } => x.deps_into(elems, exact, reason, out),
            // Block-local: a quantization block is the unit, so the range is
            // widened to block boundaries and stays exact.
            Expr::Dequantize { x, scheme } | Expr::Quantize { x, scheme, .. } => {
                let block = block_elems(scheme).unwrap_or(1).max(1);
                let lo = (elems.0 / block) * block;
                let hi = elems.1.div_ceil(block) * block;
                x.deps_into((lo, hi), exact, reason, out)
            }
            Expr::Bin { a, b, .. } => {
                // Broadcasting means an operand's own element range may be
                // smaller; the conservative-but-cheap answer is the same range
                // clipped to the operand's size, which is exact when shapes
                // match and a bound when they do not.
                for (side, t) in [(a, a.infer().ok()), (b, b.infer().ok())] {
                    let n = t
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| numel(&s))
                        .unwrap_or(elems.1);
                    if n >= elems.1 {
                        side.deps_into(elems, exact, reason, out);
                    } else {
                        side.deps_into(
                            (0, n),
                            false,
                            Some("broadcast operand is read in full"),
                            out,
                        );
                    }
                }
            }
            Expr::Concat { xs, axis } => {
                // Concatenation on the outermost axis splits the range exactly;
                // on an inner axis the parts interleave, so the range is a
                // bound.
                let mut base = 0u64;
                for x in xs {
                    let n = x
                        .infer()
                        .ok()
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| numel(&s))
                        .unwrap_or(0);
                    let (lo, hi) = (base, base + n);
                    if *axis == 0 {
                        if hi > elems.0 && lo < elems.1 {
                            x.deps_into(
                                (elems.0.saturating_sub(lo), (elems.1 - lo).min(n)),
                                exact,
                                reason,
                                out,
                            );
                        }
                    } else {
                        x.deps_into(
                            (0, n),
                            false,
                            Some("concat on an inner axis interleaves its inputs"),
                            out,
                        );
                    }
                    base = hi;
                }
            }
            Expr::Slice {
                x,
                starts,
                sizes,
                steps,
            } => {
                // Exact when the slice takes whole trailing axes with unit
                // steps: then a contiguous output range maps to a contiguous
                // input range.
                let inner: u64 = sizes[1..].iter().product();
                let whole_tail = starts[1..].iter().all(|s| *s == 0)
                    && steps.iter().all(|s| *s == 1)
                    && x.infer()
                        .ok()
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| s[1..] == sizes[1..])
                        .unwrap_or(false);
                if whole_tail {
                    let off = starts[0] * inner;
                    x.deps_into((elems.0 + off, elems.1 + off), exact, reason, out);
                } else {
                    let n = x
                        .infer()
                        .ok()
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| numel(&s))
                        .unwrap_or(0);
                    x.deps_into(
                        (0, n),
                        false,
                        Some("strided or inner-axis slice touches a non-contiguous input range"),
                        out,
                    );
                }
            }
            Expr::MatMul { a, b, .. } => {
                for side in [a, b] {
                    let n = side
                        .infer()
                        .ok()
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| numel(&s))
                        .unwrap_or(0);
                    side.deps_into(
                        (0, n),
                        false,
                        Some("matmul needs the whole contraction dimension"),
                        out,
                    );
                }
            }
            other => {
                // Everything else — permute, expand, pad, gather, split, norm,
                // sparse, delta, select, plugin — is either data-dependent or
                // reorders elements, so the honest answer is the full input.
                let why: &'static str = match other {
                    Expr::Gather { .. } => "gather is data-dependent on its index tensor",
                    Expr::Norm { .. } => "norm reduces a whole axis",
                    Expr::Permute { .. } => "permute reorders elements",
                    Expr::Sparse { .. } => "sparse encodings are read whole",
                    Expr::Delta { .. } => "a delta reads both parents",
                    Expr::Select { .. } => "select depends on a runtime capability",
                    _ => "no exact inverse-range function for this node",
                };
                for c in other.children() {
                    let n = c
                        .infer()
                        .ok()
                        .and_then(|t| concrete(&t.shape))
                        .map(|s| numel(&s))
                        .unwrap_or(0);
                    c.deps_into((0, n), false, Some(why), out);
                }
            }
        }
    }
}

fn bit_of_linear(layout: &Layout, shape: &[u64], dtype: &DType, linear: u64) -> Option<u128> {
    // Recover the multi-index of a row-major linear position, then ask the
    // layout where it lives.
    let strides = crate::layout::Order::RowMajor.strides(shape);
    let mut idx = Vec::with_capacity(shape.len());
    let mut rem = linear;
    for st in &strides {
        idx.push(rem / st.max(&1));
        rem %= st.max(&1);
    }
    layout.bit_offset(shape, dtype, &idx)
}

/// Merges dependencies on the same source, so a caller sees one range per
/// object rather than one per traversal path.
fn coalesce(mut deps: Vec<Dep>) -> Vec<Dep> {
    deps.sort_by(|a, b| (a.source, &a.uri, a.bytes.0).cmp(&(b.source, &b.uri, b.bytes.0)));
    let mut out: Vec<Dep> = Vec::new();
    for d in deps {
        match out.last_mut() {
            Some(prev) if prev.source == d.source && prev.uri == d.uri => {
                prev.bytes.0 = prev.bytes.0.min(d.bytes.0);
                prev.bytes.1 = prev.bytes.1.max(d.bytes.1);
                if !d.exact {
                    prev.exact = false;
                    prev.reason = prev.reason.or(d.reason);
                }
            }
            _ => out.push(d),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryStore, WritableStore};

    /// Stores a tensor's bytes as a single-chunk literal and returns the node.
    fn literal(store: &mut MemoryStore, t: &Tensor, dtype: &DType, layout: &Layout) -> Expr {
        let bytes = t.to_bytes(dtype, layout, Round::Rne).unwrap();
        let d = store.put(&bytes).unwrap();
        Expr::Literal {
            chunks: (otype::BLOB, d),
            dtype: dtype.clone(),
            shape: dims(&t.shape),
            layout: layout.clone(),
        }
    }

    fn f32_literal(store: &mut MemoryStore, shape: &[u64], data: &[f64]) -> Expr {
        let t = Tensor::new(shape.to_vec(), DType::F32, data.to_vec());
        literal(store, &t, &DType::F32, &Layout::default())
    }

    #[test]
    fn a_literal_round_trips_through_a_store() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = f32_literal(&mut s, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ctx = Ctx::new(&s);
        let t = e.eval(&ctx).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // And its type is inferred without reading a byte.
        assert_eq!(
            e.infer().unwrap(),
            Type {
                shape: dims(&[2, 3]),
                dtype: DType::F32
            }
        );
    }

    #[test]
    fn structural_nodes_are_index_remappings() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ctx = Ctx::new(&s);

        let t = Expr::Permute {
            x: Box::new(x.clone()),
            perm: vec![1, 0],
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let t = Expr::Reshape {
            x: Box::new(x.clone()),
            shape: dims(&[3, 2]),
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = Expr::Slice {
            x: Box::new(x.clone()),
            starts: vec![0, 1],
            sizes: vec![2, 2],
            steps: vec![1, 1],
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.data, vec![2.0, 3.0, 5.0, 6.0]);

        let t = Expr::Concat {
            xs: vec![x.clone(), x.clone()],
            axis: 0,
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.shape, vec![4, 3]);
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[6], 1.0);

        let t = Expr::Split {
            x: Box::new(x.clone()),
            axis: 1,
            sizes: vec![1, 2],
            pick: 1,
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![2.0, 3.0, 5.0, 6.0]);

        let t = Expr::Pad {
            x: Box::new(x.clone()),
            pads: vec![(1, 0), (0, 1)],
            mode: PadMode::Constant,
            value: Scalar::Float(-1.0),
        }
        .eval(&ctx)
        .unwrap();
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(&t.data[..4], &[-1.0, -1.0, -1.0, -1.0]);
        assert_eq!(&t.data[4..8], &[1.0, 2.0, 3.0, -1.0]);

        let idx = f32_literal(&mut s, &[2], &[2.0, 0.0]);
        let t = Expr::Gather {
            x: Box::new(x.clone()),
            idx: Box::new(idx),
            axis: 1,
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![3.0, 1.0, 6.0, 4.0]);

        let t = Expr::Expand {
            x: Box::new(f32_literal(&mut s, &[1, 3], &[1.0, 2.0, 3.0])),
            shape: dims(&[2, 3]),
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_worked_lora_example_of_section_04_8() {
        // W_lora = add(W, scale(matmul(B, A), alpha/r)) with alpha=30, r=16.
        let mut s = MemoryStore::new(HashAlgo::default());
        let w = f32_literal(&mut s, &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let bmat = f32_literal(&mut s, &[2, 1], &[1.0, 2.0]);
        let amat = f32_literal(&mut s, &[1, 2], &[3.0, 4.0]);
        let lora = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(w),
            b: Box::new(Expr::Scale {
                x: Box::new(Expr::MatMul {
                    a: Box::new(bmat),
                    b: Box::new(amat),
                    sum: Sum::Sequential,
                }),
                k: Scalar::Ratio(30, 16),
            }),
        };
        let t = lora.eval(&Ctx::new(&s)).unwrap();
        // B@A = [[3,4],[6,8]], times 30/16 = 1.875.
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(
            t.data,
            vec![
                1.0 + 3.0 * 1.875,
                4.0 * 1.875,
                6.0 * 1.875,
                1.0 + 8.0 * 1.875
            ]
        );
        // The ratio is exact: 30/16 reduces to 15/8, and folding two scales
        // multiplies exactly.
        assert_eq!(Scalar::Ratio(30, 16).reduced(), Scalar::Ratio(15, 8));
        assert_eq!(
            Scalar::Ratio(30, 16).times(Scalar::Ratio(16, 30)),
            Scalar::Int(1)
        );
    }

    #[test]
    fn casting_goes_through_the_target_encoding() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // 1.1 is not representable in bf16; a cast must actually lose the bits
        // rather than keeping the f64 value around.
        let x = f32_literal(&mut s, &[1], &[1.100_000_023_841_858]);
        let e = Expr::Cast {
            x: Box::new(x),
            dtype: DType::BF16,
            round: Round::Rne,
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(t.dtype, DType::BF16);
        assert_eq!(t.data[0], 1.1015625);
        assert_eq!(e.infer().unwrap().dtype, DType::BF16);
    }

    #[test]
    fn typing_rejects_what_evaluation_would_get_wrong() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2, 3], &[0.0; 6]);
        // Reshape must preserve the element count.
        assert!(Expr::Reshape {
            x: Box::new(x.clone()),
            shape: dims(&[4, 2])
        }
        .infer()
        .is_err());
        // Contraction dimensions must agree.
        assert!(Expr::MatMul {
            a: Box::new(x.clone()),
            b: Box::new(x.clone()),
            sum: Sum::Unspecified
        }
        .infer()
        .is_err());
        // Slices must stay in range.
        assert!(Expr::Slice {
            x: Box::new(x.clone()),
            starts: vec![0, 2],
            sizes: vec![2, 2],
            steps: vec![1, 1]
        }
        .infer()
        .is_err());
        // Concat needs matching non-axis extents.
        let y = f32_literal(&mut s, &[3, 3], &[0.0; 9]);
        assert!(Expr::Concat {
            xs: vec![x.clone(), y.clone()],
            axis: 1
        }
        .infer()
        .is_err());
        assert!(Expr::Concat {
            xs: vec![x.clone(), y],
            axis: 0
        }
        .infer()
        .is_ok());
        // Broadcasting rejects genuinely incompatible extents.
        let z = f32_literal(&mut s, &[4], &[0.0; 4]);
        assert!(Expr::Bin {
            op: BinOp::Add,
            a: Box::new(x.clone()),
            b: Box::new(z)
        }
        .infer()
        .is_err());
        // R-T01: the declared type must match the inferred one.
        assert!(x.check_declared(&dims(&[2, 3]), &DType::F32).is_ok());
        assert!(x.check_declared(&dims(&[3, 2]), &DType::F32).is_err());
        assert!(x.check_declared(&dims(&[2, 3]), &DType::BF16).is_err());
    }

    #[test]
    fn every_core_op_parses_and_re_encodes() {
        let d = [0u8; 32];
        let lit = Value::map(vec![
            ("op", Value::text("literal")),
            (
                "chunks",
                Value::Array(vec![Value::U(6), Value::Bytes(d.to_vec())]),
            ),
            ("dtype", DType::F32.to_value()),
            ("shape", Value::Array(vec![Value::U(2), Value::U(2)])),
        ]);
        let one = |op: &str, extra: Vec<(&'static str, Value)>| {
            let mut p = vec![("op", Value::text(op)), ("x", lit.clone())];
            p.extend(extra);
            Value::map(p)
        };
        let cases: Vec<Value> = vec![
            lit.clone(),
            Value::map(vec![
                ("op", Value::text("extern")),
                ("uri", Value::text("hf://acme/m/model.safetensors")),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            Value::map(vec![
                ("op", Value::text("zeros")),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            Value::map(vec![
                ("op", Value::text("ones")),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            Value::map(vec![
                ("op", Value::text("full")),
                ("value", Value::F64(0.5)),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            Value::map(vec![
                ("op", Value::text("arange")),
                ("start", Value::U(0)),
                ("step", Value::U(2)),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            Value::map(vec![
                ("op", Value::text("eye")),
                ("rows", Value::U(3)),
                ("cols", Value::U(3)),
                ("dtype", DType::F32.to_value()),
            ]),
            Value::map(vec![
                ("op", Value::text("random")),
                (
                    "dist",
                    Value::map(vec![
                        ("k", Value::text("normal")),
                        ("std", Value::F64(0.02)),
                    ]),
                ),
                ("seed", Value::U(7)),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(4)])),
            ]),
            one("reshape", vec![("shape", Value::Array(vec![Value::U(4)]))]),
            one("transpose", vec![]),
            Value::map(vec![
                ("op", Value::text("squeeze")),
                (
                    "x",
                    Value::map(vec![
                        ("op", Value::text("zeros")),
                        ("dtype", DType::F32.to_value()),
                        ("shape", Value::Array(vec![Value::U(1), Value::U(4)])),
                    ]),
                ),
                ("axes", Value::Array(vec![Value::U(0)])),
            ]),
            one(
                "permute",
                vec![("perm", Value::Array(vec![Value::U(1), Value::U(0)]))],
            ),
            one(
                "expand",
                vec![(
                    "shape",
                    Value::Array(vec![Value::U(3), Value::U(2), Value::U(2)]),
                )],
            ),
            one(
                "slice",
                vec![
                    ("starts", Value::Array(vec![Value::U(0), Value::U(0)])),
                    ("sizes", Value::Array(vec![Value::U(1), Value::U(2)])),
                ],
            ),
            Value::map(vec![
                ("op", Value::text("concat")),
                ("xs", Value::Array(vec![lit.clone(), lit.clone()])),
                ("axis", Value::U(0)),
            ]),
            one(
                "split",
                vec![
                    ("axis", Value::U(0)),
                    ("sizes", Value::Array(vec![Value::U(1), Value::U(1)])),
                    ("pick", Value::U(0)),
                ],
            ),
            one(
                "pad",
                vec![
                    (
                        "pads",
                        Value::Array(vec![
                            Value::Array(vec![Value::U(1), Value::U(1)]),
                            Value::Array(vec![Value::U(0), Value::U(0)]),
                        ]),
                    ),
                    ("mode", Value::text("edge")),
                ],
            ),
            Value::map(vec![
                ("op", Value::text("gather")),
                ("x", lit.clone()),
                (
                    "idx",
                    Value::map(vec![
                        ("op", Value::text("arange")),
                        ("dtype", DType::U32.to_value()),
                        ("shape", Value::Array(vec![Value::U(2)])),
                    ]),
                ),
                ("axis", Value::U(0)),
            ]),
            one(
                "relayout",
                vec![(
                    "layout",
                    Value::map(vec![
                        ("k", Value::text("strided")),
                        ("order", Value::text("col-major")),
                    ]),
                )],
            ),
            one(
                "cast",
                vec![
                    ("dtype", DType::BF16.to_value()),
                    ("rounding", Value::text("rtz")),
                ],
            ),
            Value::map(vec![
                ("op", Value::text("add")),
                ("a", lit.clone()),
                ("b", lit.clone()),
            ]),
            Value::map(vec![
                ("op", Value::text("sub")),
                ("a", lit.clone()),
                ("b", lit.clone()),
            ]),
            Value::map(vec![
                ("op", Value::text("mul")),
                ("a", lit.clone()),
                ("b", lit.clone()),
            ]),
            Value::map(vec![
                ("op", Value::text("div")),
                ("a", lit.clone()),
                ("b", lit.clone()),
            ]),
            one("scale", vec![("k", Value::F64(0.5))]),
            Value::map(vec![
                ("op", Value::text("matmul")),
                ("a", lit.clone()),
                ("b", lit.clone()),
                ("sum", Value::text("pairwise")),
            ]),
            one("norm", vec![("axis", Value::U(0)), ("p", Value::F64(2.0))]),
            one(
                "clamp",
                vec![("lo", Value::F64(-1.0)), ("hi", Value::F64(1.0))],
            ),
            one(
                "dequantize",
                vec![(
                    "scheme",
                    Value::map(vec![
                        ("scheme", Value::text("sym")),
                        ("out", DType::F32.to_value()),
                    ]),
                )],
            ),
            one(
                "quantize",
                vec![(
                    "scheme",
                    Value::map(vec![
                        ("scheme", Value::text("sym")),
                        ("out", DType::I8.to_value()),
                    ]),
                )],
            ),
            Value::map(vec![
                ("op", Value::text("sparse")),
                ("scheme", Value::text("bitmask")),
                ("mask", lit.clone()),
                ("values", lit.clone()),
                ("shape", Value::Array(vec![Value::U(2), Value::U(2)])),
                ("dtype", DType::F32.to_value()),
                ("fill", Value::F64(0.0)),
            ]),
            one(
                "approx",
                vec![(
                    "bound",
                    Value::map(vec![
                        ("mode", Value::text("rel")),
                        ("bound", Value::F64(1e-3)),
                    ]),
                )],
            ),
            Value::map(vec![
                ("op", Value::text("delta")),
                ("base", lit.clone()),
                ("patch", lit.clone()),
                ("dop", Value::text("add")),
            ]),
            Value::map(vec![
                ("op", Value::text("select")),
                ("feature", Value::text("omni.dtype/f8e4m3.1")),
                ("a", lit.clone()),
                ("b", lit.clone()),
            ]),
            Value::map(vec![
                ("op", Value::text("plugin")),
                ("ns", Value::text("org.acme/quant")),
                ("name", Value::text("my-scheme")),
                ("v", Value::U(2)),
                ("args", Value::Array(vec![lit.clone()])),
                ("attrs", Value::Map(vec![])),
                ("crit", Value::Bool(true)),
                ("shape", Value::Array(vec![Value::U(2), Value::U(2)])),
                ("dtype", DType::F32.to_value()),
                ("fallback", lit.clone()),
            ]),
        ];
        // Every op in the core set is exercised.
        let mut covered: Vec<String> = Vec::new();
        for c in &cases {
            let e = Expr::from_value(c).unwrap_or_else(|err| panic!("{}: {err}", c.diag()));
            e.infer()
                .unwrap_or_else(|err| panic!("{}: {err}", c.diag()));
            covered.push(c.get("op").unwrap().as_str().unwrap().to_string());
            // Re-encoding and re-parsing is a fixed point, which is what makes
            // the identity digest stable.
            let again = Expr::from_value(&e.to_value()).unwrap();
            assert_eq!(again, e, "{}", c.diag());
            // Through canonical CBOR the comparison is on bytes: decoding
            // sorts map keys (D3), and a node that carries an opaque scheme
            // map keeps whatever order it was handed. What must be stable is
            // the encoding, because that is what gets hashed.
            let bytes = e.to_value().encode();
            let decoded = crate::cbor::decode(&bytes).unwrap();
            assert_eq!(
                Expr::from_value(&decoded).unwrap().to_value().encode(),
                bytes,
                "{}",
                c.diag()
            );
        }
        for op in CORE_OPS {
            // `transpose` normalizes to `permute`, and zeros/ones are `full`.
            assert!(
                covered.iter().any(|c| c == op),
                "core op `{op}` has no round-trip case"
            );
        }
    }

    #[test]
    fn unknown_ops_are_refused_rather_than_ignored() {
        let v = Value::map(vec![
            ("op", Value::text("magic")),
            ("dtype", DType::F32.to_value()),
        ]);
        assert!(matches!(Expr::from_value(&v), Err(Error::Unsupported(_))));
    }

    #[test]
    fn normalization_gives_equivalent_trees_one_identity() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let y = f32_literal(&mut s, &[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let algo = HashAlgo::default();

        // add is commutative, so the two spellings are one value.
        let ab = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(x.clone()),
            b: Box::new(y.clone()),
        };
        let ba = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(y.clone()),
            b: Box::new(x.clone()),
        };
        assert_eq!(ab.identity(algo), ba.identity(algo));
        // sub is not.
        let sub = Expr::Bin {
            op: BinOp::Sub,
            a: Box::new(x.clone()),
            b: Box::new(y.clone()),
        };
        assert_ne!(sub.identity(algo), ab.identity(algo));

        // scale(scale(x, 2), 3) == scale(x, 6), exactly.
        let nested = Expr::Scale {
            x: Box::new(Expr::Scale {
                x: Box::new(x.clone()),
                k: Scalar::Int(2),
            }),
            k: Scalar::Int(3),
        };
        let flat = Expr::Scale {
            x: Box::new(x.clone()),
            k: Scalar::Int(6),
        };
        assert_eq!(nested.identity(algo), flat.identity(algo));
        assert_eq!(
            nested.normalize(algo).eval(&Ctx::new(&s)).unwrap(),
            flat.eval(&Ctx::new(&s)).unwrap()
        );

        // transpose(transpose(x)) is x.
        let tt = Expr::Permute {
            x: Box::new(Expr::Permute {
                x: Box::new(x.clone()),
                perm: vec![1, 0],
            }),
            perm: vec![1, 0],
        };
        assert_eq!(tt.identity(algo), x.identity(algo));
        assert_eq!(tt.normalize(algo), x);

        // cast(cast(x, bf16), bf16) is one cast.
        let cc = Expr::Cast {
            x: Box::new(Expr::Cast {
                x: Box::new(x.clone()),
                dtype: DType::BF16,
                round: Round::Rne,
            }),
            dtype: DType::BF16,
            round: Round::Rne,
        };
        let c = Expr::Cast {
            x: Box::new(x.clone()),
            dtype: DType::BF16,
            round: Round::Rne,
        };
        assert_eq!(cc.identity(algo), c.identity(algo));

        // scale by 1 is nothing; reshaping a constant folds.
        assert_eq!(
            Expr::Scale {
                x: Box::new(x.clone()),
                k: Scalar::Int(1)
            }
            .normalize(algo),
            x
        );
        let zr = Expr::Reshape {
            x: Box::new(Expr::Full {
                value: Scalar::Int(0),
                dtype: DType::F32,
                shape: dims(&[4]),
            }),
            shape: dims(&[2, 2]),
        };
        assert_eq!(
            zr.normalize(algo),
            Expr::Full {
                value: Scalar::Int(0),
                dtype: DType::F32,
                shape: dims(&[2, 2])
            }
        );

        // Identity is domain-separated: it is not the digest of the bytes.
        assert_ne!(
            x.identity(algo).to_vec(),
            algo.digest(&x.to_value().encode()).to_vec()
        );
        // And it does not depend on which digest algorithm names the objects
        // being equal — a different algorithm gives a different identity.
        assert_ne!(x.identity(algo), x.identity(HashAlgo::Sha256));
    }

    #[test]
    fn determinism_is_declared_not_assumed() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(x.deterministic());
        let unpinned = Expr::MatMul {
            a: Box::new(x.clone()),
            b: Box::new(x.clone()),
            sum: Sum::Unspecified,
        };
        assert!(!unpinned.deterministic());
        let pinned = Expr::MatMul {
            a: Box::new(x.clone()),
            b: Box::new(x.clone()),
            sum: Sum::Kahan,
        };
        assert!(pinned.deterministic());
        // Pinning the order changes the digest, because it changes the promise.
        assert_ne!(
            unpinned.identity(HashAlgo::default()),
            pinned.identity(HashAlgo::default())
        );
        // A pinned reduction is reproducible: Kahan and pairwise agree here
        // because the values are exact, and both are stated rather than hoped
        // for.
        let a = pinned.eval(&Ctx::new(&s)).unwrap();
        let b = Expr::MatMul {
            a: Box::new(x.clone()),
            b: Box::new(x.clone()),
            sum: Sum::Pairwise,
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn approx_marks_a_lossy_subtree() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2], &[1.0, 2.0]);
        assert!(!x.is_lossy());
        let a = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(x.clone()),
            b: Box::new(Expr::Approx {
                x: Box::new(x.clone()),
                bound: Bound::Rel(1e-3),
            }),
        };
        assert!(a.is_lossy());
        // The marker does not change the value.
        assert_eq!(a.eval(&Ctx::new(&s)).unwrap().data, vec![2.0, 4.0]);
    }

    #[test]
    fn a_plugin_without_a_fallback_is_refused_and_one_with_it_degrades() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = f32_literal(&mut s, &[2], &[1.0, 2.0]);
        let hard = Expr::Plugin {
            ns: "org.acme/quant".into(),
            name: "my-scheme".into(),
            v: 2,
            args: vec![x.clone()],
            attrs: Value::Map(vec![]),
            crit: true,
            shape: dims(&[2]),
            dtype: DType::F32,
            fallback: None,
        };
        assert_eq!(
            hard.required_plugins(),
            vec!["org.acme/quant/my-scheme.2".to_string()]
        );
        assert!(matches!(
            hard.eval(&Ctx::new(&s)),
            Err(Error::Unsupported(_))
        ));
        let soft = Expr::Plugin {
            ns: "org.acme/quant".into(),
            name: "my-scheme".into(),
            v: 2,
            args: vec![x.clone()],
            attrs: Value::Map(vec![]),
            crit: true,
            shape: dims(&[2]),
            dtype: DType::F32,
            fallback: Some(Box::new(x.clone())),
        };
        assert!(soft.required_plugins().is_empty());
        assert_eq!(soft.eval(&Ctx::new(&s)).unwrap().data, vec![1.0, 2.0]);
        // A reader that has no plugins rewrites to the fallback and evaluates
        // the core-only tree.
        let lowered = soft.with_fallbacks(&|_, _, _| false);
        assert_eq!(lowered, x);
    }

    #[test]
    fn select_reads_the_runtimes_capabilities() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let a = f32_literal(&mut s, &[2], &[1.0, 2.0]);
        let b = f32_literal(&mut s, &[2], &[3.0, 4.0]);
        let e = Expr::Select {
            feature: "omni.dtype/f8e4m3.1".into(),
            a: Box::new(a),
            b: Box::new(b),
        };
        let on = Ctx::new(&s).feature("omni.dtype/f8e4m3.1", true);
        assert_eq!(e.eval(&on).unwrap().data, vec![1.0, 2.0]);
        // Absent capability means the fallback branch, not an error.
        assert_eq!(e.eval(&Ctx::new(&s)).unwrap().data, vec![3.0, 4.0]);
    }

    #[test]
    fn delta_composes_with_its_parent() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let base = f32_literal(&mut s, &[3], &[1.0, 2.0, 3.0]);
        let patch = f32_literal(&mut s, &[3], &[0.5, 0.0, -1.0]);
        let e = Expr::Delta {
            base: Box::new(base.clone()),
            patch: Box::new(patch),
            op: DeltaOp::Add,
        };
        assert_eq!(e.eval(&Ctx::new(&s)).unwrap().data, vec![1.5, 2.0, 2.0]);

        // xor is a bit-level patch, exact on integers.
        let bi = Tensor::new(vec![3], DType::U8, vec![1.0, 2.0, 3.0]);
        let pi = Tensor::new(vec![3], DType::U8, vec![255.0, 0.0, 8.0]);
        let a = literal(&mut s, &bi, &DType::U8, &Layout::default());
        let b = literal(&mut s, &pi, &DType::U8, &Layout::default());
        let x = Expr::Delta {
            base: Box::new(a),
            patch: Box::new(b),
            op: DeltaOp::Xor,
        };
        assert_eq!(
            x.eval(&Ctx::new(&s)).unwrap().data,
            vec![(1u8 ^ 255) as f64, 2.0, (3u8 ^ 8) as f64]
        );
    }

    #[test]
    fn an_extern_leaf_is_never_fetched() {
        let s = MemoryStore::new(HashAlgo::default());
        let e = Expr::Extern {
            uri: "https://example.invalid/w.bin".into(),
            digest: None,
            dtype: DType::F32,
            shape: dims(&[4]),
        };
        assert!(matches!(e.eval(&Ctx::new(&s)), Err(Error::External(_))));
        // But it is visible as a dependency, so a planner can decide.
        let d = e.deps_all();
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].uri.as_deref(), Some("https://example.invalid/w.bin"));
        assert_eq!(d[0].bytes, (0, 16));
    }

    #[test]
    fn range_pushdown_reads_only_the_rows_asked_for() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // 100 rows of 10 f32 elements: 4000 bytes.
        let x = f32_literal(&mut s, &[100, 10], &vec![1.0; 1000]);
        let deps = x.deps((100, 200));
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].bytes, (400, 800));
        assert!(deps[0].exact);

        // Through a chain of structural and elementwise nodes the range is
        // still exact — this is the "partial loading is automatic" claim.
        let chain = Expr::Cast {
            x: Box::new(Expr::Scale {
                x: Box::new(Expr::Reshape {
                    x: Box::new(x.clone()),
                    shape: dims(&[10, 100]),
                }),
                k: Scalar::Float(2.0),
            }),
            dtype: DType::BF16,
            round: Round::Rne,
        };
        let deps = chain.deps((100, 200));
        assert_eq!(deps[0].bytes, (400, 800));
        assert!(deps[0].exact);

        // A matmul forces the whole of both operands, and says why.
        let mm = Expr::MatMul {
            a: Box::new(x.clone()),
            b: Box::new(f32_literal(&mut s, &[10, 4], &vec![1.0; 40])),
            sum: Sum::Sequential,
        };
        let deps = mm.deps((0, 4));
        assert!(deps.iter().all(|d| !d.exact));
        assert!(deps
            .iter()
            .any(|d| d.reason == Some("matmul needs the whole contraction dimension")));
        assert!(deps.iter().any(|d| d.bytes == (0, 4000)));

        // Concatenation on the outer axis splits the request exactly.
        let cat = Expr::Concat {
            xs: vec![x.clone(), x.clone()],
            axis: 0,
        };
        let deps = cat.deps((0, 10));
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].bytes, (0, 40));
    }

    #[test]
    fn sub_byte_literals_are_read_through_their_layout() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // int4 values packed eight to a 32-bit word, GPTQ style.
        let layout = Layout::Packed {
            elems_per_word: 8,
            word_bits: 32,
            bit_order: crate::layout::BitOrder::LsbFirst,
            order: crate::layout::Order::RowMajor,
        };
        let t = Tensor::new(
            vec![2, 8],
            DType::U4,
            (0..16).map(|i| (i % 16) as f64).collect(),
        );
        let e = literal(&mut s, &t, &DType::U4, &layout);
        let got = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(got.data, t.data);
        // Eight nibbles per word, two words: 8 bytes.
        assert_eq!(layout.stored_bytes(&[2, 8], &DType::U4), Some(8));
    }

    #[test]
    fn depth_is_bounded() {
        let mut e = Expr::Full {
            value: Scalar::Int(0),
            dtype: DType::F32,
            shape: dims(&[1]),
        };
        for _ in 0..MAX_DEPTH + 2 {
            e = Expr::Scale {
                x: Box::new(e),
                k: Scalar::Float(1.5),
            };
        }
        assert!(matches!(e.infer(), Err(Error::Bounds(_))));
    }

    #[test]
    fn the_evaluator_refuses_to_allocate_without_a_bound() {
        let s = MemoryStore::new(HashAlgo::default());
        let huge = Expr::Full {
            value: Scalar::Int(0),
            dtype: DType::F32,
            shape: dims(&[1 << 30, 1 << 30]),
        };
        let ctx = Ctx::new(&s).max_elems(1000);
        assert!(matches!(huge.eval(&ctx), Err(Error::Bounds(_))));
    }

    #[test]
    fn random_is_reproducible_and_addressable() {
        let s = MemoryStore::new(HashAlgo::default());
        let e = Expr::Random {
            dist: Dist::Normal {
                mean: 0.0,
                std: 0.02,
            },
            seed: 42,
            dtype: DType::F32,
            shape: dims(&[64]),
        };
        let a = e.eval(&Ctx::new(&s)).unwrap();
        let b = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(a.data, b.data);
        // Counter-based: element 7 does not depend on elements 0..7.
        let one = Expr::Slice {
            x: Box::new(e.clone()),
            starts: vec![7],
            sizes: vec![1],
            steps: vec![1],
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_eq!(one.data[0], a.data[7]);
        // A different seed is a different tensor.
        let c = Expr::Random {
            dist: Dist::Normal {
                mean: 0.0,
                std: 0.02,
            },
            seed: 43,
            dtype: DType::F32,
            shape: dims(&[64]),
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_ne!(a.data, c.data);
        // It generates no bytes and therefore has no dependencies.
        assert!(e.deps_all().is_empty());
    }

    #[test]
    fn chacha20_matches_rfc_8439() {
        // RFC 8439 §2.3.2 test vector: all-zero key and nonce, counter 0.
        let out = chacha20_block(&[0u8; 32], 0, 0);
        assert_eq!(
            &out[..16],
            &[
                0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90, 0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86,
                0xbd, 0x28
            ]
        );
    }
}
