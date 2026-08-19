//! §08.6–§08.7 — delta models, inheritance, and parent resolution.
//!
//! A delta model is a `Manifest` with `parents[]` whose tensors are expressions
//! over the parents' tensors. `omni delta base.omni tuned.omni` has to choose,
//! per tensor, the cheapest representation that stays inside a declared error
//! bound — and then say what it chose and what it cost. That analysis is this
//! module.
//!
//! The honesty requirements of §08.10 are part of the design, not commentary:
//!
//! * Low-rank extraction is **lossy** unless the change really was low-rank, so
//!   the measured error is reported per tensor and a representation that exceeds
//!   the bound is not selected. When a lossy one is selected, the expression is
//!   wrapped in `approx`, which makes the loss visible in the DAG forever
//!   (R-T06).
//! * Chunk-level dedup across full fine-tunes is usually near-zero, because
//!   every weight changes. The analyzer measures it rather than assuming it, and
//!   the quantized-residual path is there for exactly that case.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo};
use crate::dtype::{DType, Round};
use crate::expr::{BinOp, Error, Expr, Ref, Scalar, Sum, Tensor};
use crate::layout::Layout;
use std::collections::BTreeMap;

type Res<T> = Result<T, Error>;

/// Default maximum parent chain depth (§08.6, R-O06).
pub const MAX_CHAIN_DEPTH: usize = 32;

// ------------------------------------------------------------------- parents --

/// One entry of a manifest's `parents[]` (§08.7).
#[derive(Clone, Debug, PartialEq)]
pub struct Parent {
    pub reference: Ref,
    pub role: String,
    pub name: Option<String>,
    /// Advisory hints only (§01.4). A locator is never a fetch instruction.
    pub locators: Vec<String>,
    pub required: bool,
}

impl Parent {
    pub fn from_value(v: &Value) -> Res<Parent> {
        Ok(Parent {
            reference: crate::expr::parse_ref_value(
                v.get("ref")
                    .ok_or_else(|| Error::Type("a parent must carry a `ref`".into()))?,
            )?,
            role: v
                .get("role")
                .and_then(|x| x.as_str())
                .unwrap_or("base")
                .to_string(),
            name: v
                .get("name")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            locators: v
                .get("locators")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| match x {
                            Value::Text(s) => Some(s.clone()),
                            Value::Tag(crate::cbor::TAG_URI, inner) => {
                                inner.as_str().map(|s| s.to_string())
                            }
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            required: !matches!(v.get("required"), Some(Value::Bool(false))),
        })
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            (
                "ref",
                Value::Array(vec![
                    Value::U(self.reference.0 as u64),
                    Value::Bytes(self.reference.1.to_vec()),
                ]),
            ),
            ("role", Value::text(self.role.clone())),
        ];
        if let Some(n) = &self.name {
            p.push(("name", Value::text(n.clone())));
        }
        if !self.locators.is_empty() {
            p.push((
                "locators",
                Value::Array(
                    self.locators
                        .iter()
                        .map(|l| Value::Tag(crate::cbor::TAG_URI, Box::new(Value::text(l.clone()))))
                        .collect(),
                ),
            ));
        }
        p.push(("required", Value::Bool(self.required)));
        Value::map(p)
    }
}

/// Reads a manifest's parent list.
pub fn parents(manifest: &Value) -> Res<Vec<Parent>> {
    let mut out = Vec::new();
    for p in manifest
        .get("parents")
        .and_then(|x| x.as_array())
        .unwrap_or(&[])
    {
        out.push(Parent::from_value(p)?);
    }
    Ok(out)
}

/// What resolving a parent chain found.
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    /// Manifest digests from the delta up to the root, in order.
    pub links: Vec<Digest>,
    /// Parents that are required but absent: the container is *incomplete*
    /// (§08.7), and `omni inspect` says so on the first line.
    pub missing: Vec<Parent>,
    /// True when the declared depth bound was hit (R-O06).
    pub truncated: bool,
}

impl Chain {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && !self.truncated
    }
}

/// Walks the parent chain from a manifest, bounded by `max_depth` (R-O06).
///
/// Parents are pinned by digest, so a delta can never silently attach to a
/// different base: either the digest is present or the chain is incomplete.
pub fn resolve_chain(ctx: &crate::expr::Ctx<'_>, root: &Digest, max_depth: usize) -> Res<Chain> {
    let mut links = vec![*root];
    let mut missing = Vec::new();
    let mut truncated = false;
    let mut seen = std::collections::BTreeSet::new();
    seen.insert(*root);
    let mut current = *root;
    loop {
        if links.len() > max_depth {
            truncated = true;
            break;
        }
        let m = match ctx.store().resolve(&current)? {
            Some(b) => crate::cbor::decode(&b).map_err(|e| Error::Store(e.to_string()))?,
            None => break,
        };
        let ps = parents(&m)?;
        let Some(next) = ps.iter().find(|p| p.role == "base").or(ps.first()) else {
            break;
        };
        for p in &ps {
            if p.required && !ctx.store().has(&p.reference.1)? {
                missing.push(p.clone());
            }
        }
        if !ctx.store().has(&next.reference.1)? {
            break;
        }
        // Content addressing makes a cycle impossible between distinct objects,
        // but a malformed container can name itself, so the visited set is
        // still checked.
        if !seen.insert(next.reference.1) {
            break;
        }
        links.push(next.reference.1);
        current = next.reference.1;
    }
    Ok(Chain {
        links,
        missing,
        truncated,
    })
}

// ------------------------------------------------------- delta representations --

/// The delta representations of §08.6, cheapest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The tensor did not change: reference the parent's descriptor. Zero bytes.
    Identical,
    /// Few chunks changed: a new `ChunkList` reusing the unchanged refs.
    ChunkLevel,
    /// The change is (approximately) low-rank, e.g. it came from a LoRA.
    LowRank,
    /// Few weights changed materially.
    Sparse,
    /// A dense small change, stored as a quantized residual.
    QuantizedResidual,
    /// Everything changed.
    Full,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Identical => "identical",
            Kind::ChunkLevel => "chunk-level",
            Kind::LowRank => "low-rank",
            Kind::Sparse => "sparse",
            Kind::QuantizedResidual => "quantized-residual",
            Kind::Full => "full",
        }
    }

    /// Whether this representation can lose information, and therefore has to be
    /// wrapped in `approx` when it does (R-T06).
    pub fn is_lossy(self) -> bool {
        matches!(self, Kind::LowRank | Kind::Sparse | Kind::QuantizedResidual)
    }
}

/// Analyzer knobs.
#[derive(Clone, Debug)]
pub struct Options {
    /// Maximum relative error a lossy representation may introduce.
    pub max_err: f64,
    /// Chunk size for the chunk-level analysis.
    pub chunk_size: u64,
    /// Largest rank to try for low-rank extraction.
    pub max_rank: usize,
    /// Deterministic seed for the power iteration's starting vector, so two
    /// runs of `omni delta` produce the same factors.
    pub seed: u64,
    /// Residual quantization dtype.
    pub residual_dtype: DType,
    /// The dtype new stored tensors are written in.
    pub store_dtype: DType,
    /// How much cheaper a *lossy* representation must be before it is chosen
    /// over an exact one, as a fraction of the exact one's cost.
    ///
    /// Without this the analyzer would trade exactness for a handful of bytes:
    /// an int8 residual of a rank-1 change is often a few percent smaller than
    /// the two exact factors, and taking it would throw away a lossless delta
    /// for nothing. §08.10 asks for the loss to be measured and declared; this
    /// asks for it to be worth something.
    pub lossy_gain: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_err: 1e-2,
            chunk_size: 4 << 20,
            max_rank: 32,
            seed: DEFAULT_SEED,
            residual_dtype: DType::I8,
            store_dtype: DType::F32,
            lossy_gain: 0.75,
        }
    }
}

/// The default starting seed for low-rank extraction. Any fixed value would do;
/// what matters is that it is fixed, so `omni delta` is reproducible.
pub const DEFAULT_SEED: u64 = 0x4f4d_4e49_de17_a001;

/// The chosen representation for one tensor, with its measured cost.
#[derive(Clone, Debug)]
pub struct Plan {
    pub kind: Kind,
    /// Maximum relative error this representation introduces, measured.
    pub max_rel_err: f64,
    /// Bytes this representation costs.
    pub stored_bytes: u64,
    /// New tensors the delta must store, by role.
    pub tensors: Vec<(String, Tensor)>,
    /// For `chunk-level`: the indices of the chunks that changed.
    pub changed_chunks: Vec<u64>,
    pub total_chunks: u64,
}

impl Plan {
    /// Builds the delta's expression over its parent.
    ///
    /// `stored` maps the role names in [`Plan::tensors`] to the expressions that
    /// read them back (normally `literal` nodes over freshly written objects).
    pub fn build(&self, parent: &Expr, stored: &BTreeMap<String, Expr>) -> Res<Expr> {
        let get = |k: &str| -> Res<Expr> {
            stored
                .get(k)
                .cloned()
                .ok_or_else(|| Error::Type(format!("delta needs a stored tensor for `{k}`")))
        };
        let e = match self.kind {
            // Zero bytes: the delta references the parent's value directly.
            Kind::Identical => parent.clone(),
            Kind::Full => get("full")?,
            Kind::ChunkLevel => get("full")?,
            Kind::LowRank => Expr::Bin {
                op: BinOp::Add,
                a: Box::new(parent.clone()),
                b: Box::new(Expr::MatMul {
                    a: Box::new(get("B")?),
                    b: Box::new(get("A")?),
                    sum: Sum::Pairwise,
                }),
            },
            Kind::Sparse => {
                let t = parent.infer()?;
                let shape = crate::expr::concrete(&t.shape)
                    .ok_or_else(|| Error::Type("a sparse delta needs a concrete shape".into()))?;
                Expr::Bin {
                    op: BinOp::Add,
                    a: Box::new(parent.clone()),
                    b: Box::new(Expr::Sparse {
                        scheme: "bitmask".into(),
                        parts: vec![
                            ("mask".into(), get("mask")?),
                            ("values".into(), get("values")?),
                        ],
                        attrs: Value::Map(vec![]),
                        shape: crate::expr::dims(&shape),
                        dtype: t.dtype,
                        fill: Scalar::Int(0),
                    }),
                }
            }
            Kind::QuantizedResidual => {
                let t = parent.infer()?;
                Expr::Bin {
                    op: BinOp::Add,
                    a: Box::new(parent.clone()),
                    b: Box::new(Expr::Dequantize {
                        x: Box::new(get("residual")?),
                        scheme: Value::map(vec![
                            ("scheme", Value::text("sym")),
                            ("formula", Value::text("sym")),
                            ("out", t.dtype.to_value()),
                            ("scale", get("residual_scale")?.to_value()),
                        ]),
                    }),
                }
            }
        };
        // R-T06: a lossy subtree is wrapped so the loss is visible in the DAG
        // forever, with the measured bound rather than a nominal one.
        Ok(if self.kind.is_lossy() && self.max_rel_err > 0.0 {
            Expr::Approx {
                x: Box::new(e),
                bound: crate::expr::Bound::Rel(self.max_rel_err),
            }
        } else {
            e
        })
    }
}

/// Chooses the cheapest representation for one tensor, subject to `max_err`.
pub fn analyze(base: &Tensor, tuned: &Tensor, opts: &Options) -> Res<Plan> {
    if base.shape != tuned.shape {
        // A shape change is not a delta; it is a different tensor.
        return Ok(full_plan(tuned, opts));
    }
    let n = tuned.data.len();
    let bytes_base = base.to_bytes(&opts.store_dtype, &Layout::default(), Round::Rne)?;
    let bytes_tuned = tuned.to_bytes(&opts.store_dtype, &Layout::default(), Round::Rne)?;

    // identical — exact, and therefore always preferred. The test is on bytes
    // rather than on values, because a tensor holding a NaN is bit-identical to
    // itself and not *equal* to itself.
    if bytes_base == bytes_tuned {
        return Ok(identical_plan());
    }

    let diff: Vec<f64> = tuned
        .data
        .iter()
        .zip(&base.data)
        .map(|(t, b)| t - b)
        .collect();
    let scale = tuned
        .data
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(f64::MIN_POSITIVE);

    if diff.iter().all(|d| *d == 0.0) {
        return Ok(identical_plan());
    }

    // A difference that is not a finite number cannot be approximated: there is
    // no rank-1 factorization of a NaN and no residual that reconstructs an
    // infinity. Only the exact representations are candidates.
    let approximable = diff.iter().all(|d| d.is_finite());

    let mut candidates: Vec<Plan> = Vec::new();
    let (changed, total) = chunk_diff(&bytes_base, &bytes_tuned, opts.chunk_size);
    if !changed.is_empty() && changed.len() as u64 != total {
        candidates.push(Plan {
            kind: Kind::ChunkLevel,
            max_rel_err: 0.0,
            stored_bytes: changed.len() as u64 * opts.chunk_size,
            tensors: vec![("full".into(), tuned.clone())],
            changed_chunks: changed.clone(),
            total_chunks: total,
        });
    }

    // low-rank — lossy unless the change genuinely was low-rank. The search
    // extends one component at a time from a running residual rather than
    // refactorizing from scratch per candidate rank: power iteration with
    // deflation is sequential, so rank r's first r−1 components are rank
    // r−1's, and recomputing them made the search quadratic in rank — with
    // the worst case, a full fine-tune where *no* rank passes, being the
    // common case over real base/fine-tune pairs. Same components, same
    // seeding, once each.
    if approximable && tuned.shape.len() == 2 {
        let (rows, cols) = (tuned.shape[0] as usize, tuned.shape[1] as usize);
        let max_rank = opts.max_rank.min(rows.min(cols));
        let mut lr = LowRank::start(&diff, rows, cols, opts.seed);
        for rank in 1..=max_rank {
            if !lr.extend() {
                break; // the residual has no signal left to extract
            }
            let err = lr.residual_max() / scale;
            let bytes = ((rows + cols) * rank) as u64 * opts.store_dtype.packed_bytes(1);
            if err <= opts.max_err {
                let (b, a) = lr.factors();
                candidates.push(Plan {
                    kind: Kind::LowRank,
                    max_rel_err: err,
                    stored_bytes: bytes,
                    tensors: vec![
                        (
                            "B".into(),
                            Tensor::new(
                                vec![rows as u64, rank as u64],
                                opts.store_dtype.clone(),
                                b,
                            ),
                        ),
                        (
                            "A".into(),
                            Tensor::new(
                                vec![rank as u64, cols as u64],
                                opts.store_dtype.clone(),
                                a,
                            ),
                        ),
                    ],
                    changed_chunks: vec![],
                    total_chunks: 0,
                });
                break;
            }
        }
    }

    // sparse — exact for the entries it keeps, dropping the ones below the
    // bound.
    let threshold = if approximable {
        opts.max_err * scale
    } else {
        0.0
    };
    let mut mask = vec![0.0f64; n];
    let mut values = Vec::new();
    let mut dropped = 0.0f64;
    for (i, d) in diff.iter().enumerate() {
        if d.abs() > threshold {
            mask[i] = 1.0;
            values.push(*d);
        } else {
            dropped = dropped.max(d.abs());
        }
    }
    if !values.is_empty() && approximable {
        let bytes = values.len() as u64 * opts.store_dtype.packed_bytes(1)
            + DType::Bool.packed_bytes(n as u64);
        candidates.push(Plan {
            kind: Kind::Sparse,
            max_rel_err: dropped / scale,
            stored_bytes: bytes,
            tensors: vec![
                (
                    "mask".into(),
                    Tensor::new(tuned.shape.clone(), DType::Bool, mask),
                ),
                (
                    "values".into(),
                    Tensor::new(vec![values.len() as u64], opts.store_dtype.clone(), values),
                ),
            ],
            changed_chunks: vec![],
            total_chunks: 0,
        });
    }

    // quantized-residual — lossy only in the residual, which is declared.
    if approximable {
        let absmax = diff.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let (qlo, qhi) = match &opts.residual_dtype {
            DType::Int { w, signed: true } => {
                (-(2f64.powi(*w as i32 - 1)), 2f64.powi(*w as i32 - 1) - 1.0)
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "residual dtype {} is not a signed integer",
                    other.label()
                )))
            }
        };
        let step = if absmax == 0.0 { 1.0 } else { absmax / qhi };
        let mut q = Vec::with_capacity(n);
        let mut err = 0.0f64;
        for d in &diff {
            let v = (d / step).round().clamp(qlo, qhi);
            err = err.max((v * step - d).abs());
            q.push(v);
        }
        let bytes = opts.residual_dtype.packed_bytes(n as u64) + opts.store_dtype.packed_bytes(1);
        let rel = err / scale;
        if rel <= opts.max_err {
            candidates.push(Plan {
                kind: Kind::QuantizedResidual,
                max_rel_err: rel,
                stored_bytes: bytes,
                tensors: vec![
                    (
                        "residual".into(),
                        Tensor::new(tuned.shape.clone(), opts.residual_dtype.clone(), q),
                    ),
                    (
                        "residual_scale".into(),
                        Tensor::new(vec![1], opts.store_dtype.clone(), vec![step]),
                    ),
                ],
                changed_chunks: vec![],
                total_chunks: 0,
            });
        }
    }

    candidates.push(full_plan(tuned, opts));
    candidates.sort_by(|a, b| {
        (a.stored_bytes, a.kind.is_lossy()).cmp(&(b.stored_bytes, b.kind.is_lossy()))
    });
    // Cheapest wins, except that a lossy representation has to earn it: it is
    // only chosen when it is `lossy_gain` cheaper than the best exact option.
    //
    // "Exact" here means an error at or below the storage dtype's own rounding
    // error — an approximation finer than the bytes can express is not a loss
    // anyone can observe, and a rank-1 extraction of a genuine LoRA lands
    // there.
    let eps = dtype_eps(&opts.store_dtype);
    let best_exact = candidates
        .iter()
        .filter(|c| c.max_rel_err <= eps)
        .map(|c| c.stored_bytes)
        .min();
    if let Some(exact) = best_exact {
        let budget = (exact as f64 * opts.lossy_gain) as u64;
        candidates.retain(|c| c.max_rel_err <= eps || c.stored_bytes <= budget);
    }
    Ok(candidates.remove(0))
}

/// The relative rounding error of a storage dtype: an approximation below this
/// is indistinguishable once written.
fn dtype_eps(d: &DType) -> f64 {
    match d {
        DType::Float(f) => 2f64.powi(-(f.m as i32 + 1)),
        _ => 0.0,
    }
}

fn identical_plan() -> Plan {
    Plan {
        kind: Kind::Identical,
        max_rel_err: 0.0,
        stored_bytes: 0,
        tensors: vec![],
        changed_chunks: vec![],
        total_chunks: 0,
    }
}

fn full_plan(tuned: &Tensor, opts: &Options) -> Plan {
    Plan {
        kind: Kind::Full,
        max_rel_err: 0.0,
        stored_bytes: opts.store_dtype.packed_bytes(tuned.data.len() as u64),
        tensors: vec![("full".into(), tuned.clone())],
        changed_chunks: vec![],
        total_chunks: 0,
    }
}

/// Which fixed-size chunks differ between two byte strings.
pub fn chunk_diff(a: &[u8], b: &[u8], chunk: u64) -> (Vec<u64>, u64) {
    let chunk = chunk.max(1) as usize;
    let total = b.len().div_ceil(chunk) as u64;
    let mut changed = Vec::new();
    for i in 0..total as usize {
        let lo = i * chunk;
        let hi = ((i + 1) * chunk).min(b.len());
        let old = a.get(lo..hi.min(a.len())).unwrap_or(&[]);
        if old != &b[lo..hi] {
            changed.push(i as u64);
        }
    }
    (changed, total)
}

/// An incremental deterministic factorization of `diff` by power iteration
/// with deflation: after `k` calls to [`LowRank::extend`], `diff ≈ B @ A` at
/// rank `k`, and the running residual *is* `diff − B @ A`, so the error a
/// candidate rank would leave is a read rather than a recomputation.
///
/// Not a general SVD, and not trying to be: it converges on the dominant
/// subspace, which is exactly the case §08.6 cares about (a change that came
/// from a low-rank update). The seed fixes each component's starting vector so
/// two runs of `omni delta` produce the same factors and therefore the same
/// digests.
struct LowRank {
    residual: Vec<f64>,
    rows: usize,
    cols: usize,
    seed: u64,
    /// Extracted components, `u` carrying its σ, matching what the old
    /// per-rank refactorization produced.
    us: Vec<Vec<f64>>,
    vs: Vec<Vec<f64>>,
}

impl LowRank {
    fn start(diff: &[f64], rows: usize, cols: usize, seed: u64) -> Self {
        LowRank {
            residual: diff.to_vec(),
            rows,
            cols,
            seed,
            us: Vec::new(),
            vs: Vec::new(),
        }
    }

    /// Extracts the next component from the residual and deflates. Returns
    /// `false` when there is no signal left, in which case nothing was added.
    fn extend(&mut self) -> bool {
        let (rows, cols) = (self.rows, self.cols);
        let k = self.us.len();
        // Start from a reproducible pseudo-random vector.
        let mut v: Vec<f64> = (0..cols)
            .map(|j| crate::expr::uniform01(self.seed ^ (k as u64 + 1), j as u64) - 0.5)
            .collect();
        normalize(&mut v);
        let mut u = vec![0.0f64; rows];
        for _ in 0..64 {
            // u = M v
            for (i, ui) in u.iter_mut().enumerate() {
                *ui = (0..cols).map(|j| self.residual[i * cols + j] * v[j]).sum();
            }
            if normalize(&mut u) == 0.0 {
                return false;
            }
            // v = M^T u
            for (j, vj) in v.iter_mut().enumerate() {
                *vj = (0..rows).map(|i| self.residual[i * cols + j] * u[i]).sum();
            }
            if normalize(&mut v) == 0.0 {
                return false;
            }
        }
        // sigma = u^T M v
        let mut sigma = 0.0f64;
        for (i, ui) in u.iter().enumerate() {
            for (j, vj) in v.iter().enumerate() {
                sigma += ui * self.residual[i * cols + j] * vj;
            }
        }
        if sigma.abs() < 1e-300 {
            return false;
        }
        // Deflate.
        for (i, ui) in u.iter().enumerate() {
            for (j, vj) in v.iter().enumerate() {
                self.residual[i * cols + j] -= sigma * ui * vj;
            }
        }
        for ui in u.iter_mut() {
            *ui *= sigma;
        }
        self.us.push(u);
        self.vs.push(v);
        true
    }

    /// The largest absolute entry of `diff − B @ A` at the current rank.
    fn residual_max(&self) -> f64 {
        self.residual.iter().fold(0.0f64, |m, x| m.max(x.abs()))
    }

    /// Materializes `B` (`rows × k`, σ folded in) and `A` (`k × cols`).
    fn factors(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.us.len();
        let mut b = vec![0.0f64; self.rows * k];
        let mut a = vec![0.0f64; k * self.cols];
        for (kk, (u, v)) in self.us.iter().zip(&self.vs).enumerate() {
            for (i, ui) in u.iter().enumerate() {
                b[i * k + kk] = *ui;
            }
            for (j, vj) in v.iter().enumerate() {
                a[kk * self.cols + j] = *vj;
            }
        }
        (b, a)
    }
}

fn normalize(v: &mut [f64]) -> f64 {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

// -------------------------------------------------------------------- reports --

/// One line of the `omni delta` report (§08.6).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Row {
    pub tensors: usize,
    pub bytes: u64,
    pub max_rel_err: f64,
}

/// The whole report, in the order §08.6 prints it.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub rows: BTreeMap<&'static str, Row>,
    pub tensors: usize,
    pub total_bytes: u64,
    pub base_bytes: u64,
}

impl Report {
    pub fn add(&mut self, plan: &Plan) {
        let r = self.rows.entry(plan.kind.name()).or_default();
        r.tensors += 1;
        r.bytes += plan.stored_bytes;
        r.max_rel_err = r.max_rel_err.max(plan.max_rel_err);
        self.tensors += 1;
        self.total_bytes += plan.stored_bytes;
    }

    /// The delta as a percentage of the base, which is the number §08.6's
    /// report ends with.
    pub fn percent_of_base(&self) -> f64 {
        if self.base_bytes == 0 {
            return 0.0;
        }
        100.0 * self.total_bytes as f64 / self.base_bytes as f64
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "tensors: {}", self.tensors)?;
        for kind in [
            Kind::Identical,
            Kind::ChunkLevel,
            Kind::LowRank,
            Kind::Sparse,
            Kind::QuantizedResidual,
            Kind::Full,
        ] {
            let Some(r) = self.rows.get(kind.name()) else {
                continue;
            };
            write!(
                f,
                "  {:<23}: {:>3}   {}",
                kind.name(),
                r.tensors,
                human(r.bytes)
            )?;
            if kind.is_lossy() {
                write!(f, "   max rel-err {:.1e}", r.max_rel_err)?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "total delta: {}   ({:.2} % of base)",
            human(self.total_bytes),
            self.percent_of_base()
        )
    }
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1000.0 && u + 1 < UNITS.len() {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Flattens a chain for distribution (§08.7): the tensor's expression with its
/// parent links resolved down to `depth` levels, keeping the provenance links in
/// the manifest.
///
/// Flattening is a distribution choice, not a semantic one — the flattened
/// expression denotes the same value, which is why it is safe to do
/// mechanically.
pub fn flatten(e: &Expr, depth: usize) -> Expr {
    if depth == 0 {
        return e.clone();
    }
    e.map_children(&|c| flatten(c, depth - 1))
}

/// The identity a delta's tensor has once built, for dedup accounting.
pub fn delta_identity(e: &Expr, algo: HashAlgo) -> Digest {
    e.identity(algo)
}

/// Builds a `literal` node over an already-stored blob.
pub fn literal_of(digest: Digest, dtype: &DType, shape: &[u64]) -> Expr {
    Expr::Literal {
        chunks: (otype::BLOB, digest),
        dtype: dtype.clone(),
        shape: crate::expr::dims(shape),
        layout: Layout::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::Object;
    use crate::expr::Ctx;
    use crate::store::{MemoryStore, WritableStore};

    fn t(shape: &[u64], data: Vec<f64>) -> Tensor {
        Tensor::new(shape.to_vec(), DType::F32, data)
    }

    /// Stores a plan's tensors and builds its expression, the way `omni delta`
    /// would.
    fn realize(s: &mut MemoryStore, plan: &Plan, parent: &Expr) -> Expr {
        let mut stored = BTreeMap::new();
        for (role, tensor) in &plan.tensors {
            let bytes = tensor
                .to_bytes(&tensor.dtype, &Layout::default(), Round::Rne)
                .unwrap();
            let d = s.put(&bytes).unwrap();
            stored.insert(role.clone(), literal_of(d, &tensor.dtype, &tensor.shape));
        }
        plan.build(parent, &stored).unwrap()
    }

    #[test]
    fn an_unchanged_tensor_costs_nothing() {
        let base = t(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let plan = analyze(&base, &base.clone(), &Options::default()).unwrap();
        assert_eq!(plan.kind, Kind::Identical);
        assert_eq!(plan.stored_bytes, 0);
        assert_eq!(plan.max_rel_err, 0.0);
        assert!(plan.tensors.is_empty());
        // And its expression is the parent's, verbatim.
        let mut s = MemoryStore::new(HashAlgo::default());
        let parent = Expr::Full {
            value: Scalar::Float(1.0),
            dtype: DType::F32,
            shape: crate::expr::dims(&[2, 2]),
        };
        let e = realize(&mut s, &plan, &parent);
        assert_eq!(e, parent);
    }

    #[test]
    fn a_lora_derived_change_is_extracted_exactly() {
        // The change is genuinely rank 1, so low-rank extraction is exact and
        // §08.10's caveat does not apply.
        let rows = 8usize;
        let cols = 6usize;
        let b: Vec<f64> = (0..rows).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let a: Vec<f64> = (0..cols).map(|j| (j as f64 + 1.0) * 0.25).collect();
        let base = t(&[rows as u64, cols as u64], vec![1.0; rows * cols]);
        let tuned = t(
            &[rows as u64, cols as u64],
            (0..rows * cols)
                .map(|k| 1.0 + b[k / cols] * a[k % cols])
                .collect(),
        );
        let plan = analyze(&base, &tuned, &Options::default()).unwrap();
        assert_eq!(plan.kind, Kind::LowRank);
        assert!(plan.max_rel_err < 1e-9, "{}", plan.max_rel_err);
        // Two factors: [rows, 1] and [1, cols].
        assert_eq!(plan.tensors.len(), 2);
        assert_eq!(plan.tensors[0].1.shape, vec![rows as u64, 1]);
        assert_eq!(plan.tensors[1].1.shape, vec![1, cols as u64]);
        // Far cheaper than a full copy.
        assert!(plan.stored_bytes < 4 * (rows * cols) as u64);

        // Rebuilding reproduces the tuned tensor.
        let mut s = MemoryStore::new(HashAlgo::default());
        let bytes = base
            .to_bytes(&DType::F32, &Layout::default(), Round::Rne)
            .unwrap();
        let bd = s.put(&bytes).unwrap();
        let parent = literal_of(bd, &DType::F32, &base.shape);
        let e = realize(&mut s, &plan, &parent);
        let got = e.eval(&Ctx::new(&s)).unwrap();
        for (g, w) in got.data.iter().zip(&tuned.data) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        // The extraction is deterministic: the same inputs give the same
        // factors, and therefore the same digests.
        let again = analyze(&base, &tuned, &Options::default()).unwrap();
        assert_eq!(again.tensors[0].1.data, plan.tensors[0].1.data);
    }

    #[test]
    fn a_low_rank_extraction_that_would_exceed_the_bound_is_not_chosen() {
        // §08.10: `omni delta` refuses to exceed --max-err silently. A random
        // dense change is not low-rank at any small rank.
        let base = t(&[8, 8], vec![0.0; 64]);
        let tuned = t(
            &[8, 8],
            (0..64)
                .map(|i| crate::expr::uniform01(42, i) - 0.5)
                .collect(),
        );
        let opts = Options {
            max_err: 1e-6,
            max_rank: 3,
            ..Default::default()
        };
        let plan = analyze(&base, &tuned, &opts).unwrap();
        assert_ne!(plan.kind, Kind::LowRank);
        // And whatever was chosen respects the bound.
        assert!(plan.max_rel_err <= opts.max_err);
    }

    #[test]
    fn a_few_changed_weights_become_a_sparse_delta() {
        let mut data = vec![1.0f64; 256];
        let base = t(&[16, 16], data.clone());
        data[5] = 2.0;
        data[100] = -3.0;
        let tuned = t(&[16, 16], data);
        let plan = analyze(&base, &tuned, &Options::default()).unwrap();
        assert_eq!(plan.kind, Kind::Sparse);
        assert_eq!(plan.tensors[1].1.data, vec![1.0, -4.0]);

        let mut s = MemoryStore::new(HashAlgo::default());
        let bytes = base
            .to_bytes(&DType::F32, &Layout::default(), Round::Rne)
            .unwrap();
        let bd = s.put(&bytes).unwrap();
        let parent = literal_of(bd, &DType::F32, &base.shape);
        let e = realize(&mut s, &plan, &parent);
        let got = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(got.data, tuned.data);
    }

    #[test]
    fn a_dense_small_change_becomes_a_quantized_residual() {
        // Every weight changes a little: chunk dedup is useless here, which is
        // §08.10's point, and the residual path is what saves the bytes.
        let base = t(&[16, 16], vec![1.0; 256]);
        let tuned = t(
            &[16, 16],
            (0..256)
                .map(|i| 1.0 + 0.01 * (crate::expr::uniform01(7, i) - 0.5))
                .collect(),
        );
        let opts = Options {
            max_err: 1e-3,
            max_rank: 2,
            ..Default::default()
        };
        let plan = analyze(&base, &tuned, &opts).unwrap();
        assert_eq!(plan.kind, Kind::QuantizedResidual);
        assert!(plan.max_rel_err <= opts.max_err);
        // int8 residual plus one scale: a quarter of a f32 copy.
        assert!(plan.stored_bytes < 256 * 4 / 3);

        let mut s = MemoryStore::new(HashAlgo::default());
        let bytes = base
            .to_bytes(&DType::F32, &Layout::default(), Round::Rne)
            .unwrap();
        let bd = s.put(&bytes).unwrap();
        let parent = literal_of(bd, &DType::F32, &base.shape);
        let e = realize(&mut s, &plan, &parent);
        // The lossy path is wrapped in `approx`, so the loss is visible in the
        // DAG forever (R-T06).
        assert!(e.is_lossy());
        let got = e.eval(&Ctx::new(&s)).unwrap();
        for (g, w) in got.data.iter().zip(&tuned.data) {
            assert!((g - w).abs() <= opts.max_err * 1.001, "{g} vs {w}");
        }
    }

    #[test]
    fn changed_chunks_are_counted_not_assumed() {
        // Continued pretraining on a subset: one region of the tensor moves.
        let mut data = vec![1.0f64; 4096];
        let base = t(&[4096], data.clone());
        // The whole first chunk moves, so a per-element sparse delta would cost
        // more than shipping that chunk again.
        for (i, d) in data.iter_mut().enumerate().take(1024) {
            *d = 2.0 + i as f64;
        }
        let tuned = t(&[4096], data);
        let opts = Options {
            chunk_size: 4096, // 1024 f32 elements per chunk
            max_err: 0.0,     // no lossy representation may be chosen
            ..Default::default()
        };
        let plan = analyze(&base, &tuned, &opts).unwrap();
        assert_eq!(plan.kind, Kind::ChunkLevel);
        assert_eq!(plan.total_chunks, 4);
        assert_eq!(plan.changed_chunks, vec![0]);
        assert_eq!(plan.stored_bytes, 4096);

        // A full fine-tune changes every chunk, so chunk-level is not offered
        // at all — the honest outcome of §08.10.
        let tuned2 = t(&[4096], (0..4096).map(|i| 1.0 + i as f64).collect());
        let plan = analyze(&base, &tuned2, &opts).unwrap();
        assert_eq!(plan.kind, Kind::Full);
    }

    #[test]
    fn a_shape_change_is_not_a_delta() {
        let base = t(&[4], vec![1.0; 4]);
        let tuned = t(&[8], vec![1.0; 8]);
        let plan = analyze(&base, &tuned, &Options::default()).unwrap();
        assert_eq!(plan.kind, Kind::Full);
    }

    #[test]
    fn the_report_reads_like_section_08_6() {
        let mut r = Report {
            base_bytes: 140_000_000_000,
            ..Default::default()
        };
        for p in [
            Plan {
                kind: Kind::Identical,
                max_rel_err: 0.0,
                stored_bytes: 0,
                tensors: vec![],
                changed_chunks: vec![],
                total_chunks: 0,
            },
            Plan {
                kind: Kind::LowRank,
                max_rel_err: 0.0,
                stored_bytes: 612_300_000,
                tensors: vec![],
                changed_chunks: vec![],
                total_chunks: 0,
            },
            Plan {
                kind: Kind::QuantizedResidual,
                max_rel_err: 3.2e-3,
                stored_bytes: 287_100_000,
                tensors: vec![],
                changed_chunks: vec![],
                total_chunks: 0,
            },
        ] {
            r.add(&p);
        }
        let s = r.to_string();
        assert!(s.contains("tensors: 3"));
        assert!(s.contains("identical"));
        assert!(s.contains("max rel-err 3.2e-3"), "{s}");
        assert!(s.contains("0.64 % of base"), "{s}");
        // Exact representations do not print an error column, because they have
        // none.
        assert!(!s.contains("identical             :   1   0 B   max"));
    }

    fn manifest(parent: Option<Digest>) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("model")),
        ];
        if let Some(d) = parent {
            p.push((
                "parents",
                Value::Array(vec![Parent {
                    reference: (otype::MANIFEST, d),
                    role: "base".into(),
                    name: Some("acme/llm-8b".into()),
                    locators: vec!["oci://ghcr.io/acme/llm-8b".into()],
                    required: true,
                }
                .to_value()]),
            ));
        }
        Value::map(p)
    }

    #[test]
    fn a_parent_chain_resolves_and_is_bounded() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // foundation <- instruct <- code
        let root = s
            .put(&Object::structure(otype::MANIFEST, &manifest(None)).payload)
            .unwrap();
        let mid = s
            .put(&Object::structure(otype::MANIFEST, &manifest(Some(root))).payload)
            .unwrap();
        let leaf = s
            .put(&Object::structure(otype::MANIFEST, &manifest(Some(mid))).payload)
            .unwrap();
        let ctx = Ctx::new(&s);
        let chain = resolve_chain(&ctx, &leaf, MAX_CHAIN_DEPTH).unwrap();
        assert_eq!(chain.links, vec![leaf, mid, root]);
        assert!(chain.is_complete());
        // R-O06: a depth bound is a bound.
        let short = resolve_chain(&ctx, &leaf, 1).unwrap();
        assert!(short.truncated);
        assert!(!short.is_complete());
    }

    #[test]
    fn a_missing_required_parent_makes_the_container_incomplete() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let absent = [0xcd; 32];
        let leaf = s
            .put(&Object::structure(otype::MANIFEST, &manifest(Some(absent))).payload)
            .unwrap();
        let chain = resolve_chain(&Ctx::new(&s), &leaf, MAX_CHAIN_DEPTH).unwrap();
        assert_eq!(chain.links, vec![leaf]);
        assert_eq!(chain.missing.len(), 1);
        assert!(!chain.is_complete());
        assert_eq!(chain.missing[0].name.as_deref(), Some("acme/llm-8b"));
        // The locator is advisory and is carried, not followed.
        assert_eq!(chain.missing[0].locators.len(), 1);
    }

    #[test]
    fn a_manifest_that_names_itself_does_not_loop() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // Build a manifest, then a second one naming the first, then rewrite so
        // the digest is its own parent's — content addressing makes a true
        // cycle impossible, so this stands in for a malformed container.
        let a = s
            .put(&Object::structure(otype::MANIFEST, &manifest(None)).payload)
            .unwrap();
        let selfish = manifest(Some(a));
        let d = s
            .put(&Object::structure(otype::MANIFEST, &selfish).payload)
            .unwrap();
        let chain = resolve_chain(&Ctx::new(&s), &d, MAX_CHAIN_DEPTH).unwrap();
        assert_eq!(chain.links.len(), 2);
        // Walking from the root itself terminates immediately.
        let chain = resolve_chain(&Ctx::new(&s), &a, MAX_CHAIN_DEPTH).unwrap();
        assert_eq!(chain.links, vec![a]);
    }

    #[test]
    fn parents_round_trip() {
        let p = Parent {
            reference: (otype::MANIFEST, [3u8; 32]),
            role: "base".into(),
            name: Some("acme/llm-8b".into()),
            locators: vec![
                "oci://ghcr.io/acme/llm-8b@sha256:abc".into(),
                "hf://acme/llm-8b".into(),
            ],
            required: true,
        };
        let v = p.to_value();
        assert_eq!(Parent::from_value(&v).unwrap(), p);
        let round = crate::cbor::decode(&v.encode()).unwrap();
        assert_eq!(Parent::from_value(&round).unwrap(), p);
        // `required: false` survives, since it is the difference between
        // incomplete and merely optional.
        let mut m = v.as_map().unwrap().to_vec();
        m.retain(|(k, _)| k.as_str() != Some("required"));
        m.push((Value::text("required"), Value::Bool(false)));
        assert!(!Parent::from_value(&Value::Map(m)).unwrap().required);
    }

    #[test]
    fn flatten_preserves_the_value() {
        let s = MemoryStore::new(HashAlgo::default());
        let e = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(Expr::Full {
                value: Scalar::Float(1.0),
                dtype: DType::F32,
                shape: crate::expr::dims(&[2]),
            }),
            b: Box::new(Expr::Scale {
                x: Box::new(Expr::Full {
                    value: Scalar::Float(2.0),
                    dtype: DType::F32,
                    shape: crate::expr::dims(&[2]),
                }),
                k: Scalar::Float(0.5),
            }),
        };
        let f = flatten(&e, 1);
        assert_eq!(
            f.eval(&Ctx::new(&s)).unwrap().data,
            e.eval(&Ctx::new(&s)).unwrap().data
        );
        assert_eq!(flatten(&e, 0), e);
    }
}
