//! §05 — quantization.
//!
//! Quantization in OMNI is a transformation, not a file type: there is no
//! "quantized model", only tensors whose value is `dequantize(integer_literal,
//! scheme)`. A scheme is therefore *data* consumed by two expression nodes, and
//! this module is the interpreter for that data.
//!
//! The section's central claim — that every scheme in the catalogue (§05.2) is
//! expressible with the core algebra and no plugins — is what the tests here
//! check: uniform affine, GPTQ with its column permutation and int4-in-int32
//! packing, AWQ's pre-scaling, GGUF's `Q8_0`/`Q4_0`/`Q4_1` blocks, double
//! quantization, NF4, MX microscaling, and ternary BitNet weights, each built
//! from `dequantize` plus ordinary nodes.
//!
//! The `formula` field is drawn from a closed set for a specific reason stated
//! in §05.1: whether the zero point is subtracted before or after scaling is a
//! recurring source of silent corruption when converting between GPTQ, AWQ and
//! GGUF. Here it is one enum with one meaning each.

use crate::cbor::Value;
use crate::container::{otype, Digest};
use crate::dtype::{DType, Round};
use crate::expr::{Ctx, Dim, Error, Expr, Ref, Tensor};
use crate::layout::numel;

type Res<T> = Result<T, Error>;

/// The dequantization formulas of §05.1. Closed set, one meaning each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formula {
    /// `(q − z) · s`
    AffineSub,
    /// `q · s + b`
    AffineAdd,
    /// `q · s`
    Sym,
    /// `book[q] · s`
    Codebook,
    /// `book[q]`
    CodebookRaw,
    /// `(q − z) · (s_q − z_s) · s_ss` — double quantization.
    Nested,
}

impl Formula {
    pub fn id(self) -> &'static str {
        match self {
            Formula::AffineSub => "affine-sub",
            Formula::AffineAdd => "affine-add",
            Formula::Sym => "sym",
            Formula::Codebook => "codebook",
            Formula::CodebookRaw => "codebook-raw",
            Formula::Nested => "nested",
        }
    }

    pub fn parse(s: &str) -> Option<Formula> {
        Some(match s {
            "affine-sub" => Formula::AffineSub,
            "affine-add" => Formula::AffineAdd,
            "sym" => Formula::Sym,
            "codebook" => Formula::Codebook,
            "codebook-raw" => Formula::CodebookRaw,
            "nested" => Formula::Nested,
            _ => return None,
        })
    }

    fn uses_codebook(self) -> bool {
        matches!(self, Formula::Codebook | Formula::CodebookRaw)
    }
}

/// A quantization scheme descriptor (§05.1).
#[derive(Clone, Debug, PartialEq)]
pub struct Scheme {
    /// The named scheme (`affine`, `sym`, `codebook`, `nested`, …). This is
    /// documentation and dispatch convenience; `formula` is what defines the
    /// arithmetic.
    pub name: String,
    pub formula: Formula,
    /// Dequantized dtype.
    pub out: Option<DType>,
    /// The quantized axis, for reporting and for `order`.
    pub axis: Option<usize>,
    /// Group/block shape. `None` means per-tensor.
    pub block: Option<Vec<u64>>,
    pub scale: Option<Expr>,
    pub zero: Option<Expr>,
    /// Second-level scale terms for `nested` (double quantization).
    pub scale_zero: Option<Expr>,
    pub scale_scale: Option<Expr>,
    /// GPTQ act-order permutation. The canonical form applies it with an
    /// explicit `gather` node (§05.2.2); carrying it here is the equivalent
    /// shorthand, and it means the same thing.
    pub order: Option<Expr>,
    pub book: Option<Ref>,
    pub clip: Option<(f64, f64)>,
    pub sym: bool,
}

impl Scheme {
    pub fn from_value(v: &Value) -> Res<Scheme> {
        let name = v
            .get("scheme")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error::Type("quantization scheme has no `scheme`".into()))?
            .to_string();
        let sym = matches!(v.get("sym"), Some(Value::Bool(true)));
        // A scheme without an explicit formula gets the one its name implies.
        // An importer that does not know the source's exact formula must use an
        // opaque dtype instead of guessing (§05.6 rule 1), so guessing here
        // would defeat that rule — hence only the unambiguous names map.
        let formula = match v.get("formula").and_then(|x| x.as_str()) {
            Some(f) => {
                Formula::parse(f).ok_or_else(|| Error::Type(format!("unknown formula `{f}`")))?
            }
            None => match name.as_str() {
                "sym" => Formula::Sym,
                "affine" if sym => Formula::Sym,
                "affine" => Formula::AffineSub,
                "codebook" => Formula::Codebook,
                "nested" => Formula::Nested,
                other => {
                    return Err(Error::Type(format!(
                        "scheme `{other}` does not imply a formula; §05.1 requires one from the \
                         closed set (affine-sub, affine-add, sym, codebook, codebook-raw, nested)"
                    )))
                }
            },
        };
        let expr = |key: &str| -> Res<Option<Expr>> {
            match v.get(key) {
                Some(e) => Ok(Some(Expr::from_value(e)?)),
                None => Ok(None),
            }
        };
        let book = match v.get("book") {
            Some(b) => Some(parse_book_ref(b)?),
            None => None,
        };
        let clip = match v.get("clip").and_then(|x| x.as_array()) {
            Some(a) if a.len() == 2 => Some((
                num(&a[0]).ok_or_else(|| Error::Type("clip bounds must be numbers".into()))?,
                num(&a[1]).ok_or_else(|| Error::Type("clip bounds must be numbers".into()))?,
            )),
            Some(_) => return Err(Error::Type("clip must be [lo, hi]".into())),
            None => None,
        };
        Ok(Scheme {
            name,
            formula,
            out: match v.get("out") {
                Some(d) => Some(DType::from_value(d).map_err(Error::Type)?),
                None => None,
            },
            axis: v.get("axis").and_then(|x| x.as_u64()).map(|a| a as usize),
            block: v.get("block").and_then(|x| x.as_array()).map(|a| {
                a.iter()
                    .map(|d| d.as_u64().unwrap_or(1).max(1))
                    .collect::<Vec<u64>>()
            }),
            scale: expr("scale")?,
            zero: expr("zero")?,
            scale_zero: expr("scale_zero")?,
            scale_scale: expr("scale_scale")?,
            order: expr("order")?,
            book,
            clip,
            sym,
        })
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("scheme", Value::text(self.name.clone())),
            ("formula", Value::text(self.formula.id())),
        ];
        if let Some(d) = &self.out {
            p.push(("out", d.to_value()));
        }
        if let Some(a) = self.axis {
            p.push(("axis", Value::U(a as u64)));
        }
        if let Some(b) = &self.block {
            p.push((
                "block",
                Value::Array(b.iter().map(|x| Value::U(*x)).collect()),
            ));
        }
        for (k, e) in [
            ("scale", &self.scale),
            ("zero", &self.zero),
            ("scale_zero", &self.scale_zero),
            ("scale_scale", &self.scale_scale),
            ("order", &self.order),
        ] {
            if let Some(e) = e {
                p.push((k, e.to_value()));
            }
        }
        if let Some(b) = &self.book {
            p.push((
                "book",
                Value::Array(vec![Value::U(b.0 as u64), Value::Bytes(b.1.to_vec())]),
            ));
        }
        if let Some((lo, hi)) = self.clip {
            p.push(("clip", Value::Array(vec![Value::F64(lo), Value::F64(hi)])));
        }
        if self.sym {
            p.push(("sym", Value::Bool(true)));
        }
        Value::map(p)
    }

    /// The block shape for a tensor of `shape`: the declared one, or the whole
    /// tensor when the scheme is per-tensor.
    pub fn block_shape(&self, shape: &[u64]) -> Vec<u64> {
        match &self.block {
            Some(b) if b.len() == shape.len() => b.clone(),
            // A block shape of the wrong rank is treated as per-tensor rather
            // than silently mis-indexed; `check` reports it.
            _ => shape.to_vec(),
        }
    }

    /// The shape of the per-block scale (and zero) tensor implied by `shape`:
    /// one entry per block. R-T04 compares the declared scale tensor against
    /// this.
    pub fn grid_shape(&self, shape: &[u64]) -> Vec<u64> {
        let b = self.block_shape(shape);
        shape
            .iter()
            .zip(&b)
            .map(|(d, bb)| d.div_ceil((*bb).max(1)))
            .collect()
    }

    /// R-T04: the scheme's own tensors must be consistent with the block
    /// structure it declares.
    pub fn check(&self, shape: &[u64]) -> Res<()> {
        if let Some(b) = &self.block {
            if b.len() != shape.len() {
                return Err(Error::Type(format!(
                    "R-T04: block {:?} has rank {} but the tensor has rank {}",
                    b,
                    b.len(),
                    shape.len()
                )));
            }
        }
        let grid = self.grid_shape(shape);
        for (label, e) in [("scale", &self.scale), ("zero", &self.zero)] {
            let Some(e) = e else { continue };
            let t = e.infer()?;
            let Some(s) = crate::expr::concrete(&t.shape) else {
                continue;
            };
            // A scalar is always acceptable: that is the per-tensor case.
            if numel(&s) == 1 {
                continue;
            }
            if numel(&s) != numel(&grid) {
                return Err(Error::Type(format!(
                    "R-T04: `{label}` has {} entries but the block structure needs {} \
                     ({:?} blocks of {:?})",
                    numel(&s),
                    numel(&grid),
                    grid,
                    self.block_shape(shape)
                )));
            }
        }
        if self.formula.uses_codebook() && self.book.is_none() {
            return Err(Error::Type(format!(
                "formula `{}` needs a `book`",
                self.formula.id()
            )));
        }
        if self.formula == Formula::Nested && self.scale_scale.is_none() {
            return Err(Error::Type(
                "formula `nested` needs `scale_scale`: the second-level scale is what makes \
                 double quantization reconstructible"
                    .into(),
            ));
        }
        Ok(())
    }

    /// A one-line label for the mixed-precision histogram of §05.3.
    pub fn label(&self, q: &DType) -> String {
        let group = match &self.block {
            Some(b) => {
                let g: u64 = b.iter().product();
                format!("g{g}")
            }
            None => "per-tensor".to_string(),
        };
        format!("{}-{} {}", self.formula.id(), q.label(), group)
    }
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::U(n) => Some(*n as f64),
        Value::I(n) => Some(*n as f64),
        Value::F64(f) => Some(*f),
        _ => None,
    }
}

fn parse_book_ref(v: &Value) -> Res<Ref> {
    let v = match v {
        Value::Tag(crate::cbor::TAG_REF, inner) => inner.as_ref(),
        other => other,
    };
    let a = v
        .as_array()
        .ok_or_else(|| Error::Type("`book` must be [otype, digest]".into()))?;
    let t = a
        .first()
        .and_then(|x| x.as_u64())
        .unwrap_or(otype::CODEBOOK as u64);
    let d: Digest = a
        .get(1)
        .and_then(|x| x.as_bytes())
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| Error::Type("`book` digest must be 32 bytes".into()))?;
    Ok((t as u16, d))
}

// ------------------------------------------------------------------ codebook --

/// A `Codebook` object (§05.4): the table, or a recipe that reproduces it.
#[derive(Clone, Debug, PartialEq)]
pub struct Codebook {
    pub dtype: DType,
    pub entries: usize,
    pub dim: usize,
    pub values: Vec<f64>,
    pub sorted: bool,
}

impl Codebook {
    /// Reads a codebook, either from its stored `values` expression or by
    /// replaying its `construct` recipe. §05.2.7's point is that NF4 should be
    /// reproducible rather than a magic constant table, so the recipe path is
    /// the one that gets tested.
    pub fn load(ctx: &Ctx<'_>, r: &Ref) -> Res<Codebook> {
        let v = ctx.value(&r.1)?;
        let dtype = match v.get("dtype") {
            Some(d) => DType::from_value(d).map_err(Error::Type)?,
            None => DType::F32,
        };
        let entries = v.get("entries").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let dim = v.get("dim").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
        let sorted = matches!(v.get("sorted"), Some(Value::Bool(true)));
        let values = if let Some(e) = v.get("values") {
            let t = Expr::from_value(e)?.eval(ctx)?;
            t.data
        } else if let Some(c) = v.get("construct") {
            construct(c, entries)?
        } else {
            return Err(Error::Type(
                "Codebook has neither `values` nor a `construct` recipe".into(),
            ));
        };
        if entries != 0 && values.len() != entries * dim.max(1) {
            return Err(Error::Type(format!(
                "Codebook declares {entries} entries of dim {dim} but holds {} values",
                values.len()
            )));
        }
        Ok(Codebook {
            dtype,
            entries: if entries == 0 {
                values.len() / dim.max(1)
            } else {
                entries
            },
            dim: dim.max(1),
            values,
            sorted,
        })
    }

    /// The nearest entry to `x`, for quantizing into this book.
    fn nearest(&self, x: f64) -> u64 {
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (i, v) in self.values.iter().enumerate() {
            let d = (v - x).abs();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best as u64
    }
}

/// Replays a codebook construction recipe.
fn construct(c: &Value, entries: usize) -> Res<Vec<f64>> {
    let method = c
        .get("method")
        .or_else(|| c.get("construct"))
        .and_then(|x| x.as_str())
        .ok_or_else(|| Error::Type("construct recipe has no `method`".into()))?;
    match method {
        "normal-float" => {
            let bits = c
                .get("bits")
                .and_then(|x| x.as_u64())
                .unwrap_or_else(|| (entries.max(2) as f64).log2().round() as u64)
                as u32;
            let offset = c.get("offset").and_then(num).unwrap_or(0.967_708_3);
            Ok(normal_float(bits, offset))
        }
        "kmeans" => Err(Error::Unsupported(
            "a k-means codebook is only reproducible from its training data, which is not in \
             the container; store the `values` as well (§05.4)"
                .into(),
        )),
        other => Err(Error::Unsupported(format!(
            "codebook construction method `{other}` is not implemented"
        ))),
    }
}

/// The NormalFloat construction of §05.2.7: quantiles of the standard normal at
/// evenly spaced probabilities, normalized so the extremes are ±1.
///
/// The split is asymmetric — half the levels above zero, one at zero, the rest
/// below — which is what gives NF4 its exact zero and its published values. The
/// `offset` is part of the recipe rather than a constant in code, so a future
/// NF3 or NF5 is data.
pub fn normal_float(bits: u32, offset: f64) -> Vec<f64> {
    let k = 1usize << bits;
    let pos = k / 2;
    let neg = k / 2 - 1;
    let mut v: Vec<f64> = Vec::with_capacity(k);
    for i in 0..pos {
        let p = offset + (0.5 - offset) * (i as f64) / (pos as f64);
        v.push(probit(p));
    }
    v.push(0.0);
    for i in 0..neg {
        let p = offset + (0.5 - offset) * (i as f64) / (neg as f64);
        v.push(-probit(p));
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if max > 0.0 {
        for x in &mut v {
            *x /= max;
        }
    }
    v
}

/// The standard normal quantile function, by bisection on an exact CDF. Slow
/// and boring; a codebook has sixteen entries, and being able to state the
/// construction exactly is worth more here than speed.
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `erf` by its Maclaurin series, which converges quickly over the range a
/// quantile search needs, and by the complementary asymptotic form outside it.
fn erf(x: f64) -> f64 {
    let a = x.abs();
    if a > 6.0 {
        return x.signum();
    }
    if a < 3.0 {
        let mut term = a;
        let mut sum = a;
        let x2 = a * a;
        for n in 1..200 {
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        return x.signum() * sum * 2.0 / std::f64::consts::PI.sqrt();
    }
    // Continued fraction for erfc, evaluated backwards.
    let mut cf = 0.0f64;
    for n in (1..60).rev() {
        cf = (n as f64 / 2.0) / (a + cf);
    }
    let erfc = (-a * a).exp() / ((a + cf) * std::f64::consts::PI.sqrt());
    x.signum() * (1.0 - erfc)
}

// -------------------------------------------------------------- dequantize --

/// Evaluates a `dequantize` node (§04.7.2, §05).
pub fn dequantize(ctx: &Ctx<'_>, q: &Tensor, scheme: &Value, out: &DType) -> Res<Tensor> {
    let s = Scheme::from_value(scheme)?;
    s.check(&q.shape)?;
    let block = s.block_shape(&q.shape);
    let scale = eval_opt(ctx, &s.scale)?;
    let zero = eval_opt(ctx, &s.zero)?;
    let scale_zero = eval_opt(ctx, &s.scale_zero)?;
    let scale_scale = eval_opt(ctx, &s.scale_scale)?;
    let book = match &s.book {
        Some(r) => Some(Codebook::load(ctx, r)?),
        None => None,
    };

    if matches!(s.formula, Formula::Sym) && zero.is_some() {
        return Err(Error::Type(
            "formula `sym` has no zero point, but a `zero` tensor is present; one of the two is \
             wrong and guessing which is how conversions corrupt weights (§05.1)"
                .into(),
        ));
    }

    let n = q.numel();
    let mut data = Vec::with_capacity(n as usize);
    let mut idx = vec![0u64; q.shape.len()];
    let grid = s.grid_shape(&q.shape);
    for i in 0..n {
        let g = grid_index(&idx, &block);
        let sc = pick(&scale, &g, &grid, 1.0)?;
        let z = pick(&zero, &g, &grid, 0.0)?;
        let qv = q.data[i as usize];
        let v = match s.formula {
            Formula::AffineSub => (qv - z) * sc,
            // `b` is an additive bias, so the zero tensor holds it here.
            Formula::AffineAdd => qv * sc + z,
            Formula::Sym => qv * sc,
            Formula::Codebook | Formula::CodebookRaw => {
                let b = book.as_ref().unwrap();
                let e = qv as usize;
                if e >= b.values.len() {
                    return Err(Error::Bounds(format!(
                        "codebook index {e} out of range for {} entries",
                        b.entries
                    )));
                }
                if s.formula == Formula::Codebook {
                    b.values[e] * sc
                } else {
                    b.values[e]
                }
            }
            Formula::Nested => {
                // The stored scale is itself quantized: recover it with the
                // second-level terms before using it.
                let zs = pick(&scale_zero, &g, &grid, 0.0)?;
                let sss = pick(&scale_scale, &g, &grid, 1.0)?;
                (qv - z) * ((sc - zs) * sss)
            }
        };
        let v = match s.clip {
            Some((lo, hi)) => v.clamp(lo, hi),
            None => v,
        };
        data.push(v);
        bump(&mut idx, &q.shape);
    }

    // Round the result through the declared output dtype, so a dequantization
    // produces exactly the bytes a runtime would store — and so an MX block,
    // whose scales are powers of two, comes out bit-exact.
    let mut t = Tensor::new(q.shape.clone(), out.clone(), data);
    round_through(&mut t, out)?;

    // The canonical form of act-order applies the permutation with an explicit
    // `gather`; a scheme that carries it inline means the same thing.
    if let Some(order) = &s.order {
        let axis = s.axis.unwrap_or(t.shape.len() - 1);
        t = permute_axis(&t, &order.eval(ctx)?, axis)?;
    }
    Ok(t)
}

/// Evaluates a `quantize` node. When the scheme supplies no `scale`, one is
/// derived per block from the data — absmax for symmetric schemes, min/max for
/// affine ones — which is what makes `quantize` a total function on its inputs
/// rather than a call into a search procedure.
pub fn quantize(
    ctx: &Ctx<'_>,
    x: &Tensor,
    scheme: &Value,
    out: &DType,
    round: Round,
) -> Res<Tensor> {
    let s = Scheme::from_value(scheme)?;
    s.check(&x.shape)?;
    let block = s.block_shape(&x.shape);
    let grid = s.grid_shape(&x.shape);
    let book = match &s.book {
        Some(r) => Some(Codebook::load(ctx, r)?),
        None => None,
    };
    let (qlo, qhi) = match s.clip {
        Some(c) => c,
        None => dtype_range(out).ok_or_else(|| {
            Error::Unsupported(format!(
                "quantize: {} has no integer range; declare `clip`",
                out.label()
            ))
        })?,
    };

    let given_scale = eval_opt(ctx, &s.scale)?;
    let given_zero = eval_opt(ctx, &s.zero)?;
    // Derived per-block statistics, in row-major block order.
    let mut derived_scale = vec![0.0f64; numel(&grid) as usize];
    let mut derived_zero = vec![0.0f64; numel(&grid) as usize];
    if given_scale.is_none() {
        let mut lo = vec![f64::INFINITY; derived_scale.len()];
        let mut hi = vec![f64::NEG_INFINITY; derived_scale.len()];
        let mut idx = vec![0u64; x.shape.len()];
        for i in 0..x.numel() {
            let b = linear(&grid_index(&idx, &block), &grid) as usize;
            lo[b] = lo[b].min(x.data[i as usize]);
            hi[b] = hi[b].max(x.data[i as usize]);
            bump(&mut idx, &x.shape);
        }
        for b in 0..derived_scale.len() {
            match s.formula {
                Formula::Sym => {
                    let a = lo[b].abs().max(hi[b].abs());
                    derived_scale[b] = if a == 0.0 { 1.0 } else { a / qhi.abs() };
                }
                _ => {
                    let span = hi[b] - lo[b];
                    derived_scale[b] = if span == 0.0 { 1.0 } else { span / (qhi - qlo) };
                    derived_zero[b] = qlo - lo[b] / derived_scale[b];
                }
            }
        }
    }

    let mut data = Vec::with_capacity(x.numel() as usize);
    let mut idx = vec![0u64; x.shape.len()];
    for i in 0..x.numel() {
        let g = grid_index(&idx, &block);
        let b = linear(&g, &grid) as usize;
        let sc = match &given_scale {
            Some(t) => at_broadcast(t, &g, &grid)?,
            None => derived_scale[b],
        };
        let z = match &given_zero {
            Some(t) => at_broadcast(t, &g, &grid)?,
            None => derived_zero[b],
        };
        let v = x.data[i as usize];
        let qf =
            match s.formula {
                Formula::Sym => v / sc,
                Formula::AffineSub => v / sc + z,
                Formula::AffineAdd => (v - z) / sc,
                Formula::Codebook | Formula::CodebookRaw => {
                    let b = book.as_ref().unwrap();
                    b.nearest(if s.formula == Formula::Codebook {
                        v / sc
                    } else {
                        v
                    }) as f64
                }
                Formula::Nested => return Err(Error::Unsupported(
                    "quantizing into a `nested` scheme means quantizing the scales too; write it \
                     as a `quantize` of the scale tensor and a `quantize` of the values, which is \
                     what double quantization is"
                        .into(),
                )),
            };
        let r = match round {
            Round::Stochastic { seed, .. } => Round::Stochastic { seed, index: i },
            other => other,
        };
        data.push(round_to(qf, r).clamp(qlo, qhi));
        bump(&mut idx, &x.shape);
    }
    let mut t = Tensor::new(x.shape.clone(), out.clone(), data);
    round_through(&mut t, out)?;
    Ok(t)
}

/// The number of elements in one block, for range pushdown. Re-exported from
/// [`crate::expr`] so callers reading §05 find it here too.
pub fn block_elems(scheme: &Value) -> Option<u64> {
    crate::expr::block_elems(scheme)
}

fn eval_opt(ctx: &Ctx<'_>, e: &Option<Expr>) -> Res<Option<Tensor>> {
    match e {
        Some(e) => Ok(Some(e.eval(ctx)?)),
        None => Ok(None),
    }
}

fn grid_index(index: &[u64], block: &[u64]) -> Vec<u64> {
    index
        .iter()
        .zip(block)
        .map(|(i, b)| i / (*b).max(1))
        .collect()
}

fn linear(index: &[u64], shape: &[u64]) -> u64 {
    let strides = crate::layout::Order::RowMajor.strides(shape);
    index.iter().zip(&strides).map(|(i, s)| i * s).sum()
}

/// Reads a per-block tensor at a block index. A scalar covers the whole tensor
/// (the per-tensor case); a tensor with the grid's element count is read in
/// row-major block order even if its declared shape is flat, because publishers
/// legitimately store `[rows, groups]` and `[rows * groups]` alike.
fn at_broadcast(t: &Tensor, g: &[u64], grid: &[u64]) -> Res<f64> {
    if t.data.len() == 1 {
        return Ok(t.data[0]);
    }
    if t.shape.len() == g.len() {
        let mut idx = Vec::with_capacity(g.len());
        for (k, d) in t.shape.iter().enumerate() {
            idx.push(if *d == 1 { 0 } else { g[k] });
        }
        return t
            .get(&idx)
            .ok_or_else(|| Error::Bounds(format!("block index {idx:?} out of range")));
    }
    let lin = linear(g, grid) as usize;
    t.data.get(lin).copied().ok_or_else(|| {
        Error::Bounds(format!(
            "block {lin} out of range for {} entries",
            t.data.len()
        ))
    })
}

fn pick(t: &Option<Tensor>, g: &[u64], grid: &[u64], default: f64) -> Res<f64> {
    match t {
        Some(t) => at_broadcast(t, g, grid),
        None => Ok(default),
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

fn round_to(x: f64, r: Round) -> f64 {
    // The rounding modes live in the dtype algebra; reuse them by encoding into
    // a wide integer type rather than reimplementing tie-breaking here.
    let t = DType::Int {
        w: 64,
        signed: true,
    };
    let mut buf = [0u8; 8];
    if t.encode(&mut buf, 0, x, r) {
        t.decode(&buf, 0).unwrap_or(x)
    } else {
        x
    }
}

/// Rounds every element through `dtype`'s encoding, so the tensor holds exactly
/// the values that dtype can represent.
fn round_through(t: &mut Tensor, dtype: &DType) -> Res<()> {
    if !dtype.is_numeric() {
        return Err(Error::Unsupported(format!(
            "cannot materialize into {}: it has no element semantics (§04.3.5)",
            dtype.label()
        )));
    }
    let mut buf = vec![0u8; dtype.packed_bytes(1).max(1) as usize];
    for v in &mut t.data {
        if !dtype.encode(&mut buf, 0, *v, Round::Rne) {
            return Err(Error::Unsupported(format!(
                "{} has no element encoder",
                dtype.label()
            )));
        }
        *v = dtype.decode(&buf, 0).unwrap_or(f64::NAN);
    }
    Ok(())
}

fn dtype_range(d: &DType) -> Option<(f64, f64)> {
    match d {
        DType::Int { w, signed: true } => {
            Some((-(2f64.powi(*w as i32 - 1)), 2f64.powi(*w as i32 - 1) - 1.0))
        }
        DType::Int { w, signed: false } => Some((0.0, 2f64.powi(*w as i32) - 1.0)),
        DType::Codebook { w, .. } => Some((0.0, 2f64.powi(*w as i32) - 1.0)),
        DType::Ternary { .. } => Some((-1.0, 1.0)),
        DType::Binary => Some((-1.0, 1.0)),
        DType::Bool => Some((0.0, 1.0)),
        _ => None,
    }
}

/// Applies a permutation along one axis — GPTQ's act-order, and the same thing
/// a `gather` node does.
fn permute_axis(t: &Tensor, order: &Tensor, axis: usize) -> Res<Tensor> {
    if order.data.len() as u64 != t.shape[axis] {
        return Err(Error::Type(format!(
            "`order` has {} entries but axis {axis} has extent {}",
            order.data.len(),
            t.shape[axis]
        )));
    }
    let mut data = Vec::with_capacity(t.data.len());
    let mut idx = vec![0u64; t.shape.len()];
    for _ in 0..t.numel() {
        let pick = order.data[idx[axis] as usize];
        if pick < 0.0 || pick as u64 >= t.shape[axis] {
            return Err(Error::Bounds(format!("`order` entry {pick} out of range")));
        }
        let mut src = idx.clone();
        src[axis] = pick as u64;
        data.push(
            t.get(&src)
                .ok_or_else(|| Error::Bounds("order index out of range".into()))?,
        );
        bump(&mut idx, &t.shape);
    }
    Ok(Tensor::new(t.shape.clone(), t.dtype.clone(), data))
}

// ------------------------------------------------------------- mixed precision --

/// One row of the mixed-precision report of §05.3.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantStat {
    pub label: String,
    pub tensors: usize,
    pub params: u64,
    pub stored_bytes: u64,
}

/// Effective bits per parameter: stored bytes over parameter count. §05.3 calls
/// this "the honest number that Q4_K_M gestures at".
pub fn effective_bits(stats: &[QuantStat]) -> f64 {
    let params: u64 = stats.iter().map(|s| s.params).sum();
    if params == 0 {
        return 0.0;
    }
    let bytes: u64 = stats.iter().map(|s| s.stored_bytes).sum();
    (bytes as f64 * 8.0) / params as f64
}

/// Describes how a tensor's value is quantized, for the histogram. Walks the
/// expression rather than trusting a header field, because there is no header
/// field — which is the point of §05.7.
pub fn describe(e: &Expr) -> String {
    match e {
        Expr::Dequantize { x, scheme } => {
            let q = x.infer().map(|t| t.dtype).unwrap_or(DType::F32);
            match Scheme::from_value(scheme) {
                Ok(s) => s.label(&q),
                Err(_) => format!("dequantize({}) [unreadable scheme]", q.label()),
            }
        }
        Expr::Literal { dtype, .. } => format!("{} (unquantized)", dtype.label()),
        // A quantized value under a LoRA term, a cast or a gather is still
        // quantized; report the innermost description.
        other => {
            let kids = other.children();
            match kids.first() {
                Some(c) => describe(c),
                None => other.op().to_string(),
            }
        }
    }
}

/// Shape helper for callers building schemes programmatically.
pub fn dims(shape: &[u64]) -> Vec<Dim> {
    crate::expr::dims(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::HashAlgo;
    use crate::expr::{BinOp, Scalar, Sum};
    use crate::layout::{BitOrder, Interleave, Layout, Order};
    use crate::store::{MemoryStore, WritableStore};

    fn lit(s: &mut MemoryStore, t: &Tensor, dtype: &DType, layout: &Layout) -> Expr {
        let bytes = t.to_bytes(dtype, layout, Round::Rne).unwrap();
        let d = s.put(&bytes).unwrap();
        Expr::Literal {
            chunks: (otype::BLOB, d),
            dtype: dtype.clone(),
            shape: dims(&t.shape),
            layout: layout.clone(),
        }
    }

    fn dense(s: &mut MemoryStore, shape: &[u64], dtype: &DType, data: &[f64]) -> Expr {
        let t = Tensor::new(shape.to_vec(), dtype.clone(), data.to_vec());
        lit(s, &t, dtype, &Layout::default())
    }

    fn scheme(pairs: Vec<(&str, Value)>) -> Value {
        Value::map(pairs)
    }

    #[test]
    fn uniform_affine_is_the_base_case() {
        // §05.2.1: value = (q - zero) * scale, per group of 4.
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(
            &mut s,
            &[2, 4],
            &DType::U8,
            &[0.0, 128.0, 255.0, 64.0, 10.0, 20.0, 30.0, 40.0],
        );
        let scale = dense(&mut s, &[2, 1], &DType::F32, &[0.5, 0.25]);
        let zero = dense(&mut s, &[2, 1], &DType::F32, &[128.0, 0.0]);
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                ("scale", scale.to_value()),
                ("zero", zero.to_value()),
            ]),
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(
            t.data,
            vec![
                (0.0 - 128.0) * 0.5,
                0.0,
                (255.0 - 128.0) * 0.5,
                (64.0 - 128.0) * 0.5,
                10.0 * 0.25,
                20.0 * 0.25,
                30.0 * 0.25,
                40.0 * 0.25
            ]
        );
        // The type is known statically, from the scheme's `out`.
        assert_eq!(e.infer().unwrap().dtype, DType::F32);
    }

    #[test]
    fn a_symmetric_scheme_refuses_a_zero_point() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(&mut s, &[4], &DType::I8, &[1.0, -2.0, 3.0, -4.0]);
        let zero = dense(&mut s, &[1], &DType::F32, &[3.0]);
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.5),
                        dtype: DType::F32,
                        shape: dims(&[1]),
                    }
                    .to_value(),
                ),
                ("zero", zero.to_value()),
            ]),
        };
        // Guessing which of the two is wrong is exactly how conversions between
        // GPTQ, AWQ and GGUF corrupt weights.
        assert!(matches!(e.eval(&Ctx::new(&s)), Err(Error::Type(_))));
    }

    #[test]
    fn the_formula_must_be_stated_when_the_name_does_not_imply_one() {
        assert!(Scheme::from_value(&scheme(vec![("scheme", Value::text("gptq"))])).is_err());
        // Explicit formula, any name.
        let s = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("gptq")),
            ("formula", Value::text("affine-sub")),
        ]))
        .unwrap();
        assert_eq!(s.formula, Formula::AffineSub);
        // `affine` with sym:true is symmetric, and says so in one place.
        let s = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("affine")),
            ("sym", Value::Bool(true)),
        ]))
        .unwrap();
        assert_eq!(s.formula, Formula::Sym);
    }

    #[test]
    fn gptq_is_affine_plus_a_permutation_and_int4_in_int32_words() {
        // §05.2.2. qweight: int4 packed eight to a 32-bit word, lsb-first.
        let mut s = MemoryStore::new(HashAlgo::default());
        let packed = Layout::Packed {
            elems_per_word: 8,
            word_bits: 32,
            bit_order: BitOrder::LsbFirst,
            order: Order::RowMajor,
        };
        let qt = Tensor::new(
            vec![1, 8],
            DType::U4,
            vec![8.0, 9.0, 7.0, 0.0, 15.0, 1.0, 8.0, 8.0],
        );
        let q = lit(&mut s, &qt, &DType::U4, &packed);
        let scale = dense(&mut s, &[1, 1], &DType::F32, &[0.125]);
        let zero = dense(&mut s, &[1, 1], &DType::F32, &[8.0]);
        // g_idx_inverse: reverse the columns.
        let order = dense(
            &mut s,
            &[8],
            &DType::U32,
            &[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        );
        let deq = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::F32.to_value()),
                ("axis", Value::U(1)),
                ("block", Value::Array(vec![Value::U(1), Value::U(8)])),
                ("scale", scale.to_value()),
                ("zero", zero.to_value()),
            ]),
        };
        let plain = deq.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(
            plain.data,
            vec![0.0, 0.125, -0.125, -1.0, 0.875, -0.875, 0.0, 0.0]
        );

        // The canonical form applies act-order with a `gather`; the scheme's
        // own `order` field is the same value by a shorter route, and the two
        // must agree.
        let gathered = Expr::Gather {
            x: Box::new(deq.clone()),
            idx: Box::new(order.clone()),
            axis: 1,
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        let inline = Expr::Dequantize {
            x: match deq.clone() {
                Expr::Dequantize { x, .. } => x,
                _ => unreachable!(),
            },
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::F32.to_value()),
                ("axis", Value::U(1)),
                ("block", Value::Array(vec![Value::U(1), Value::U(8)])),
                ("scale", scale.to_value()),
                ("zero", zero.to_value()),
                ("order", order.to_value()),
            ]),
        }
        .eval(&Ctx::new(&s))
        .unwrap();
        assert_eq!(gathered.data, inline.data);
        assert_eq!(
            gathered.data,
            vec![0.0, 0.0, -0.875, 0.875, -1.0, -0.125, 0.125, 0.0]
        );
    }

    #[test]
    fn awq_is_a_dequantize_and_a_multiply() {
        // §05.2.3: the smoothing factor is a real tensor, so it is stored as
        // one and the "method" is provenance.
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(&mut s, &[1, 4], &DType::U8, &[10.0, 20.0, 30.0, 40.0]);
        let awq_scales = dense(&mut s, &[1, 4], &DType::F32, &[1.0, 2.0, 0.5, 4.0]);
        let e = Expr::Bin {
            op: BinOp::Mul,
            a: Box::new(Expr::Dequantize {
                x: Box::new(q),
                scheme: scheme(vec![
                    ("scheme", Value::text("sym")),
                    ("out", DType::F32.to_value()),
                    ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                    (
                        "scale",
                        Expr::Full {
                            value: Scalar::Float(0.1),
                            dtype: DType::F32,
                            shape: dims(&[1, 1]),
                        }
                        .to_value(),
                    ),
                ]),
            }),
            b: Box::new(awq_scales),
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        for (got, want) in t.data.iter().zip([1.0, 4.0, 1.5, 16.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn gguf_q8_0_q4_0_and_q4_1_blocks() {
        // §05.2.4's structural mappings, with the block layout that puts each
        // scale next to its own 32 elements.
        let mut s = MemoryStore::new(HashAlgo::default());
        let inline = |scale_dtype: DType| Layout::BlockedScaled {
            block: vec![1, 4],
            scale_dtype,
            scale_order: Order::RowMajor,
            interleave: Interleave::ScalesInline,
        };

        // Q8_0: sym, int8, one f16 scale per block.
        let q = Tensor::new(vec![1, 4], DType::I8, vec![-128.0, -1.0, 1.0, 127.0]);
        let qe = lit(&mut s, &q, &DType::I8, &inline(DType::F16));
        let e = Expr::Dequantize {
            x: Box::new(qe),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.5),
                        dtype: DType::F16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        assert_eq!(
            e.eval(&Ctx::new(&s)).unwrap().data,
            vec![-64.0, -0.5, 0.5, 63.5]
        );

        // Q4_0: sym int4 with a f16 scale; the classic offset-by-8 encoding is
        // the affine form with a constant zero, and both are expressible.
        let q4 = Tensor::new(vec![1, 4], DType::U4, vec![0.0, 8.0, 15.0, 4.0]);
        let q4e = lit(&mut s, &q4, &DType::U4, &inline(DType::F16));
        let e = Expr::Dequantize {
            x: Box::new(q4e),
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.25),
                        dtype: DType::F16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
                (
                    "zero",
                    Expr::Full {
                        value: Scalar::Int(8),
                        dtype: DType::F16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        assert_eq!(
            e.eval(&Ctx::new(&s)).unwrap().data,
            vec![-2.0, 0.0, 1.75, -1.0]
        );

        // Q4_1: affine-add — q * s + b, with a stored minimum rather than a
        // zero point. The two formulas are one field apart and mean different
        // things, which is the whole reason the enum exists.
        let e = Expr::Dequantize {
            x: Box::new(lit(&mut s, &q4, &DType::U4, &inline(DType::F16))),
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-add")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.25),
                        dtype: DType::F16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
                (
                    "zero",
                    Expr::Full {
                        value: Scalar::Float(-2.0),
                        dtype: DType::F16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        assert_eq!(
            e.eval(&Ctx::new(&s)).unwrap().data,
            vec![-2.0, 0.0, 1.75, -1.0]
        );
    }

    #[test]
    fn double_quantization_is_the_nested_formula() {
        // §05.2.6 / §05.2.7: the scales are themselves quantized to int8 with a
        // second-level scale.
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(&mut s, &[2, 2], &DType::U8, &[10.0, 20.0, 30.0, 40.0]);
        // Stored (quantized) scales, their zero and the second-level scale.
        let sq = dense(&mut s, &[2, 1], &DType::I8, &[100.0, 50.0]);
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("nested")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(2)])),
                ("scale", sq.to_value()),
                (
                    "zero",
                    Expr::Full {
                        value: Scalar::Int(0),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
                (
                    "scale_zero",
                    Expr::Full {
                        value: Scalar::Int(20),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
                (
                    "scale_scale",
                    Expr::Full {
                        value: Scalar::Float(0.01),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        // Row 0: scale = (100 - 20) * 0.01 = 0.8; row 1: (50 - 20) * 0.01 = 0.3.
        for (got, want) in t.data.iter().zip([8.0, 16.0, 9.0, 12.0]) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
        // `nested` without its second-level scale is not reconstructible, and
        // is refused rather than half-applied.
        let bad = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("nested")),
            ("out", DType::F32.to_value()),
        ]))
        .unwrap();
        assert!(bad.check(&[2, 2]).is_err());
    }

    #[test]
    fn nf4_is_a_reproducible_codebook_not_a_magic_table() {
        // §05.2.7: the recipe is stored so the book is reproducible. These are
        // the published NF4 quantiles.
        let want = [
            -1.0,
            -0.696_192_800_998_687_7,
            -0.525_073_051_452_636_7,
            -0.394_917_488_098_144_53,
            -0.284_441_381_692_886_35,
            -0.184_773_430_228_233_34,
            -0.091_050_036_251_544_95,
            0.0,
            0.079_580_299_556_255_34,
            0.160_930_201_411_247_25,
            0.246_112_301_945_686_34,
            0.337_915_241_718_292_24,
            0.440_709_829_330_444_34,
            0.562_617_003_917_694_1,
            0.722_956_836_223_602_3,
            1.0,
        ];
        let got = normal_float(4, 0.967_708_3);
        assert_eq!(got.len(), 16);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        // Exactly one zero, and the extremes are the unit interval's ends.
        assert_eq!(got.iter().filter(|x| **x == 0.0).count(), 1);
        assert_eq!(got[0], -1.0);
        assert_eq!(got[15], 1.0);
        // erf against known values, since everything above rests on it.
        assert!((erf(0.0) - 0.0).abs() < 1e-15);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
        assert!((erf(3.5) - 0.999_999_256_901_627_7).abs() < 1e-12);
        assert!((normal_cdf(1.96) - 0.975_002_104_851_780_2).abs() < 1e-12);
    }

    #[test]
    fn a_codebook_object_dequantizes_nf4_indices() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let book = crate::container::Object::structure(
            otype::CODEBOOK,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/codebook")),
                ("v", Value::U(1)),
                ("dtype", DType::F32.to_value()),
                ("entries", Value::U(16)),
                ("dim", Value::U(1)),
                (
                    "construct",
                    Value::map(vec![
                        ("method", Value::text("normal-float")),
                        ("bits", Value::U(4)),
                    ]),
                ),
                ("sorted", Value::Bool(true)),
            ]),
        );
        let bd = s.put(&book.payload).unwrap();
        // Indices 0, 7, 15 and a per-block absmax scale of 2.
        let q = dense(&mut s, &[1, 4], &DType::U4, &[0.0, 7.0, 15.0, 8.0]);
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("codebook")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "book",
                    Value::Array(vec![
                        Value::U(otype::CODEBOOK as u64),
                        Value::Bytes(bd.to_vec()),
                    ]),
                ),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Int(2),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(t.data[0], -2.0);
        assert_eq!(t.data[1], 0.0);
        assert_eq!(t.data[2], 2.0);
        assert!((t.data[3] - 2.0 * 0.079_580_3).abs() < 1e-6);
        // codebook-raw drops the scale.
        let raw = Expr::Dequantize {
            x: Box::new(dense(&mut s, &[1, 1], &DType::U4, &[15.0])),
            scheme: scheme(vec![
                ("scheme", Value::text("codebook")),
                ("formula", Value::text("codebook-raw")),
                ("out", DType::F32.to_value()),
                (
                    "book",
                    Value::Array(vec![
                        Value::U(otype::CODEBOOK as u64),
                        Value::Bytes(bd.to_vec()),
                    ]),
                ),
            ]),
        };
        assert_eq!(raw.eval(&Ctx::new(&s)).unwrap().data, vec![1.0]);
        // An out-of-range index is a bounds error, not a wrong weight.
        let bad = Expr::Dequantize {
            x: Box::new(dense(&mut s, &[1, 1], &DType::U8, &[200.0])),
            scheme: scheme(vec![
                ("scheme", Value::text("codebook")),
                ("out", DType::F32.to_value()),
                (
                    "book",
                    Value::Array(vec![
                        Value::U(otype::CODEBOOK as u64),
                        Value::Bytes(bd.to_vec()),
                    ]),
                ),
            ]),
        };
        assert!(matches!(bad.eval(&Ctx::new(&s)), Err(Error::Bounds(_))));
    }

    #[test]
    fn a_kmeans_codebook_without_values_is_refused() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let book = crate::container::Object::structure(
            otype::CODEBOOK,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/codebook")),
                ("v", Value::U(1)),
                ("entries", Value::U(4)),
                (
                    "construct",
                    Value::map(vec![
                        ("method", Value::text("kmeans")),
                        ("seed", Value::U(1)),
                        ("iters", Value::U(20)),
                    ]),
                ),
            ]),
        );
        let bd = s.put(&book.payload).unwrap();
        let ctx = Ctx::new(&s);
        assert!(matches!(
            Codebook::load(&ctx, &(otype::CODEBOOK, bd)),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn mx_dequantization_is_exact_because_its_scales_are_powers_of_two() {
        // §05.2.8: f4e2m1 elements, blocks of 4 here, e8m0 scales after the
        // data. Every product is exact, so the result is bit-reproducible.
        let mut s = MemoryStore::new(HashAlgo::default());
        let layout = Layout::BlockedScaled {
            block: vec![1, 4],
            scale_dtype: DType::E8M0,
            scale_order: Order::RowMajor,
            interleave: Interleave::ScalesAfter,
        };
        let q = Tensor::new(
            vec![2, 4],
            DType::F4E2M1,
            vec![0.5, 1.0, -6.0, 3.0, 1.5, -1.0, 2.0, 0.0],
        );
        let qe = lit(&mut s, &q, &DType::F4E2M1, &layout);
        // e8m0 scales: 2^2 and 2^-1.
        let scales = dense(&mut s, &[2, 1], &DType::E8M0, &[4.0, 0.5]);
        let e = Expr::Dequantize {
            x: Box::new(qe),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::BF16.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                ("scale", scales.to_value()),
            ]),
        };
        let t = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(t.dtype, DType::BF16);
        assert_eq!(t.data, vec![2.0, 4.0, -24.0, 12.0, 0.75, -0.5, 1.0, 0.0]);
        // Exactness: two evaluations agree bit-for-bit, and every value is
        // representable in the output dtype without rounding.
        assert_eq!(t.data, e.eval(&Ctx::new(&s)).unwrap().data);
    }

    #[test]
    fn ternary_bitnet_weights_dequantize_at_one_point_six_bits() {
        // §05.2.9: 1.6 bits per weight actually stored.
        let mut s = MemoryStore::new(HashAlgo::default());
        let tern = DType::Ternary {
            pack: crate::dtype::TernPack::B3x5,
        };
        let t = Tensor::new(
            vec![1, 10],
            tern.clone(),
            vec![-1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0],
        );
        assert_eq!(tern.packed_bytes(10), 2);
        let q = lit(&mut s, &t, &tern, &Layout::default());
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::BF16.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(10)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.25),
                        dtype: DType::BF16,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        let out = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(
            out.data,
            t.data.iter().map(|x| x * 0.25).collect::<Vec<f64>>()
        );
    }

    #[test]
    fn quantize_derives_its_scales_when_none_are_given() {
        // §04.8's `W_int8 = quantize(W_lora, {affine, axis:0, out:i8}, "rne")`
        // carries no scale tensor, so the node must derive one — deterministically,
        // or it would not be a value.
        let mut s = MemoryStore::new(HashAlgo::default());
        let x = dense(&mut s, &[1, 4], &DType::F32, &[-1.0, -0.5, 0.25, 1.0]);
        let e = Expr::Quantize {
            x: Box::new(x.clone()),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::I8.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
            ]),
            round: Round::Rne,
        };
        let q = e.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(q.dtype, DType::I8);
        // absmax 1.0 over a signed 8-bit range: scale = 1/127.
        assert_eq!(q.data, vec![-127.0, -64.0, 32.0, 127.0]);
        // Deterministic: the same input gives the same integers.
        assert_eq!(q.data, e.eval(&Ctx::new(&s)).unwrap().data);

        // Round-tripping through the derived scale recovers the input to within
        // the quantization step.
        let deq = Expr::Dequantize {
            x: Box::new(e.clone()),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(1.0 / 127.0),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        let back = deq.eval(&Ctx::new(&s)).unwrap();
        let orig = x.eval(&Ctx::new(&s)).unwrap();
        for (a, b) in back.data.iter().zip(&orig.data) {
            assert!((a - b).abs() <= 1.0 / 127.0, "{a} vs {b}");
        }
    }

    #[test]
    fn quantize_into_a_codebook_finds_the_nearest_entry() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let book = crate::container::Object::structure(
            otype::CODEBOOK,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/codebook")),
                ("v", Value::U(1)),
                ("entries", Value::U(16)),
                (
                    "construct",
                    Value::map(vec![
                        ("method", Value::text("normal-float")),
                        ("bits", Value::U(4)),
                    ]),
                ),
            ]),
        );
        let bd = s.put(&book.payload).unwrap();
        let x = dense(&mut s, &[1, 3], &DType::F32, &[-1.0, 0.0, 0.75]);
        let e = Expr::Quantize {
            x: Box::new(x),
            scheme: scheme(vec![
                ("scheme", Value::text("codebook")),
                ("formula", Value::text("codebook-raw")),
                ("out", DType::U4.to_value()),
                (
                    "book",
                    Value::Array(vec![
                        Value::U(otype::CODEBOOK as u64),
                        Value::Bytes(bd.to_vec()),
                    ]),
                ),
            ]),
            round: Round::Rne,
        };
        let q = e.eval(&Ctx::new(&s)).unwrap();
        // -1.0 is entry 0, 0.0 is entry 7, 0.75 is nearest 0.7229568 (entry 14).
        assert_eq!(q.data, vec![0.0, 7.0, 14.0]);
    }

    #[test]
    fn scheme_tensors_must_match_the_block_structure() {
        // R-T04.
        let mut s = MemoryStore::new(HashAlgo::default());
        let scale = dense(&mut s, &[3], &DType::F32, &[1.0, 2.0, 3.0]);
        let sc = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("sym")),
            ("out", DType::F32.to_value()),
            ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
            ("scale", scale.to_value()),
        ]))
        .unwrap();
        // A [2, 8] tensor in [1, 4] blocks needs 4 scales, not 3.
        assert_eq!(sc.grid_shape(&[2, 8]), vec![2, 2]);
        assert!(sc.check(&[2, 8]).is_err());
        let ok = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("sym")),
            ("out", DType::F32.to_value()),
            ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
            (
                "scale",
                dense(&mut s, &[2, 2], &DType::F32, &[1.0, 2.0, 3.0, 4.0]).to_value(),
            ),
        ]))
        .unwrap();
        assert!(ok.check(&[2, 8]).is_ok());
        // A block shape of the wrong rank is caught rather than mis-indexed.
        let wrong = Scheme::from_value(&scheme(vec![
            ("scheme", Value::text("sym")),
            ("block", Value::Array(vec![Value::U(4)])),
        ]))
        .unwrap();
        assert!(wrong.check(&[2, 8]).is_err());
    }

    #[test]
    fn range_pushdown_widens_to_block_boundaries_and_stays_exact() {
        // §04.7.4: dequantize is block-local, so a row request reads only the
        // blocks covering it.
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(&mut s, &[64, 8], &DType::U8, &vec![1.0; 512]);
        let e = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(8)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Float(0.5),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        // Elements 20..30 lie in blocks 2 and 3, i.e. bytes 16..32.
        let deps = e.deps((20, 30));
        let lit_dep = deps.iter().find(|d| d.source.is_some()).unwrap();
        assert_eq!(lit_dep.bytes, (16, 32));
        assert!(lit_dep.exact);
    }

    #[test]
    fn the_mixed_precision_report_of_section_05_3() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = dense(&mut s, &[2, 4], &DType::U4, &[0.0; 8]);
        let deq = Expr::Dequantize {
            x: Box::new(q),
            scheme: scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::BF16.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                (
                    "scale",
                    Expr::Full {
                        value: Scalar::Int(1),
                        dtype: DType::F32,
                        shape: dims(&[1, 1]),
                    }
                    .to_value(),
                ),
            ]),
        };
        assert_eq!(describe(&deq), "affine-sub-u4 g4");
        // A LoRA term over a quantized base is still a quantized base.
        let merged = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(deq.clone()),
            b: Box::new(Expr::Scale {
                x: Box::new(Expr::MatMul {
                    a: Box::new(dense(&mut s, &[2, 1], &DType::BF16, &[1.0, 1.0])),
                    b: Box::new(dense(&mut s, &[1, 4], &DType::BF16, &[1.0; 4])),
                    sum: Sum::Sequential,
                }),
                k: Scalar::Ratio(30, 16),
            }),
        };
        assert_eq!(describe(&merged), "affine-sub-u4 g4");
        assert_eq!(
            describe(&dense(&mut s, &[2], &DType::BF16, &[0.0, 0.0])),
            "bf16 (unquantized)"
        );

        let stats = vec![
            QuantStat {
                label: "affine-int4 g128".into(),
                tensors: 226,
                params: 13_400_000_000,
                stored_bytes: 6_700_000_000,
            },
            QuantStat {
                label: "bf16 (unquantized)".into(),
                tensors: 15,
                params: 500_000_000,
                stored_bytes: 1_000_000_000,
            },
        ];
        let b = effective_bits(&stats);
        assert!((b - 4.43).abs() < 0.01, "{b}");
        assert_eq!(effective_bits(&[]), 0.0);
    }

    #[test]
    fn schemes_round_trip_through_cbor() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let scale = dense(&mut s, &[2, 1], &DType::F32, &[1.0, 2.0]);
        for v in [
            scheme(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("affine-sub")),
                ("out", DType::BF16.to_value()),
                ("axis", Value::U(1)),
                ("block", Value::Array(vec![Value::U(1), Value::U(128)])),
                ("scale", scale.to_value()),
                ("clip", Value::Array(vec![Value::I(-8), Value::U(7)])),
            ]),
            scheme(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("sym", Value::Bool(true)),
            ]),
        ] {
            let parsed = Scheme::from_value(&v).unwrap();
            let again = Scheme::from_value(&parsed.to_value()).unwrap();
            assert_eq!(parsed, again);
            let round = crate::cbor::decode(&parsed.to_value().encode()).unwrap();
            assert_eq!(Scheme::from_value(&round).unwrap(), parsed);
        }
    }
}
