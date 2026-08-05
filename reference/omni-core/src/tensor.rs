//! Reader-side views of `TensorTable` and `TensorDesc` (§04.2), and the V5
//! tensor rules of §15.2.
//!
//! §04.7.3 makes one demand of a reader: "a writer MUST record the resulting
//! `shape` and `dtype` on the owning `TensorDesc`, and a reader MUST verify that
//! inference agrees. Disagreement is a hard error — it is the cheapest possible
//! detection of a malformed or malicious file." [`TensorDesc::check`] is that
//! verification, plus the rest of the R-T rules that can be decided without
//! materializing weights.

use crate::cbor::Value;
use crate::container::{otype, Digest};
use crate::dtype::DType;
use crate::expr::{concrete, Ctx, Error, Expr, Ref, Shape};
use crate::layout::{numel, Layout, Sufficiency};
use crate::pattern;
use std::collections::BTreeMap;

type Res<T> = Result<T, Error>;

/// Optional, verifiable tensor statistics (§04.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub absmax: Option<f64>,
    pub nan: Option<u64>,
    pub inf: Option<u64>,
    pub nonzero: Option<u64>,
}

impl Stats {
    fn from_value(v: &Value) -> Stats {
        let f = |k: &str| match v.get(k) {
            Some(Value::F64(x)) => Some(*x),
            Some(Value::U(n)) => Some(*n as f64),
            Some(Value::I(n)) => Some(*n as f64),
            _ => None,
        };
        Stats {
            min: f("min"),
            max: f("max"),
            mean: f("mean"),
            absmax: f("absmax"),
            nan: v.get("nan").and_then(|x| x.as_u64()),
            inf: v.get("inf").and_then(|x| x.as_u64()),
            nonzero: v.get("nonzero").and_then(|x| x.as_u64()),
        }
    }

    fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = Vec::new();
        for (k, x) in [
            ("min", self.min),
            ("max", self.max),
            ("mean", self.mean),
            ("absmax", self.absmax),
        ] {
            if let Some(x) = x {
                p.push((k, Value::F64(x)));
            }
        }
        for (k, n) in [
            ("nan", self.nan),
            ("inf", self.inf),
            ("nonzero", self.nonzero),
        ] {
            if let Some(n) = n {
                p.push((k, Value::U(n)));
            }
        }
        Value::map(p)
    }

    /// Recomputes statistics from materialized data.
    pub fn measure(data: &[f64]) -> Stats {
        let mut s = Stats {
            min: Some(f64::INFINITY),
            max: Some(f64::NEG_INFINITY),
            absmax: Some(0.0),
            nan: Some(0),
            inf: Some(0),
            nonzero: Some(0),
            mean: None,
        };
        let mut sum = 0.0f64;
        let mut finite = 0u64;
        for x in data {
            if x.is_nan() {
                s.nan = s.nan.map(|n| n + 1);
                continue;
            }
            if x.is_infinite() {
                s.inf = s.inf.map(|n| n + 1);
                continue;
            }
            finite += 1;
            sum += x;
            s.min = Some(s.min.unwrap().min(*x));
            s.max = Some(s.max.unwrap().max(*x));
            s.absmax = Some(s.absmax.unwrap().max(x.abs()));
            if *x != 0.0 {
                s.nonzero = s.nonzero.map(|n| n + 1);
            }
        }
        if finite > 0 {
            s.mean = Some(sum / finite as f64);
        } else {
            s.min = None;
            s.max = None;
        }
        s
    }
}

/// How a tensor should be materialized (§04.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Materialize {
    Eager,
    #[default]
    Lazy,
    Stream,
}

impl Materialize {
    fn name(self) -> &'static str {
        match self {
            Materialize::Eager => "eager",
            Materialize::Lazy => "lazy",
            Materialize::Stream => "stream",
        }
    }
    fn parse(s: &str) -> Option<Materialize> {
        Some(match s {
            "eager" => Materialize::Eager,
            "lazy" => Materialize::Lazy,
            "stream" => Materialize::Stream,
            _ => return None,
        })
    }
}

/// A `TensorDesc` (otype 0x0005) as a reader sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DType,
    pub layout: Layout,
    pub value: Expr,
    pub semantic: Option<String>,
    pub role: Option<String>,
    /// Named axes. §04.2 calls these load-bearing: they are how §08 attaches
    /// adapters without hardcoding layouts.
    pub axes: Option<Vec<String>>,
    pub device_hint: Option<String>,
    pub materialize: Materialize,
    pub stats: Option<Stats>,
    pub digest_materialized: Option<Digest>,
}

impl TensorDesc {
    pub fn from_value(v: &Value) -> Res<TensorDesc> {
        let t = v.get("t").and_then(|x| x.as_str());
        if t != Some("omni.tensor/desc") {
            return Err(Error::Type(format!(
                "R-O02: expected a TensorDesc, found `{}`",
                t.unwrap_or("<no t>")
            )));
        }
        let shape = crate::expr::parse_shape_value(
            v.get("shape")
                .ok_or_else(|| Error::Type("TensorDesc has no `shape`".into()))?,
        )?;
        let dtype = DType::from_value(
            v.get("dtype")
                .ok_or_else(|| Error::Type("TensorDesc has no `dtype`".into()))?,
        )
        .map_err(Error::Type)?;
        let layout = match v.get("layout") {
            Some(l) => Layout::from_value(l).map_err(Error::Type)?,
            None => Layout::default(),
        };
        let value = Expr::from_value(
            v.get("value")
                .ok_or_else(|| Error::Type("TensorDesc has no `value`".into()))?,
        )?;
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
        Ok(TensorDesc {
            shape,
            dtype,
            layout,
            value,
            semantic: s("semantic"),
            role: s("role"),
            axes: v.get("axes").and_then(|a| a.as_array()).map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or_default().to_string())
                    .collect()
            }),
            device_hint: s("device_hint"),
            materialize: v
                .get("materialize")
                .and_then(|x| x.as_str())
                .and_then(Materialize::parse)
                .unwrap_or_default(),
            stats: v.get("stats").map(Stats::from_value),
            digest_materialized: v
                .get("digest_materialized")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok()),
        })
    }

    pub fn load(ctx: &Ctx<'_>, r: &Ref) -> Res<TensorDesc> {
        if r.0 != otype::TENSOR_DESC {
            return Err(Error::Type(format!(
                "R-O03: ref declares otype {:#06x}, expected a TensorDesc",
                r.0
            )));
        }
        TensorDesc::from_value(&ctx.value(&r.1)?)
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.tensor/desc")),
            ("v", Value::U(1)),
            ("shape", crate::expr::shape_to_value(&self.shape)),
            ("dtype", self.dtype.to_value()),
            ("layout", self.layout.to_value()),
        ];
        if let Some(s) = &self.semantic {
            p.push(("semantic", Value::text(s.clone())));
        }
        if let Some(s) = &self.role {
            p.push(("role", Value::text(s.clone())));
        }
        p.push(("value", self.value.to_value()));
        p.push(("materialize", Value::text(self.materialize.name())));
        if let Some(a) = &self.axes {
            p.push((
                "axes",
                Value::Array(a.iter().map(|x| Value::text(x.clone())).collect()),
            ));
        }
        if let Some(d) = &self.device_hint {
            p.push(("device_hint", Value::text(d.clone())));
        }
        if let Some(s) = &self.stats {
            p.push(("stats", s.to_value()));
        }
        if let Some(d) = &self.digest_materialized {
            p.push(("digest_materialized", Value::Bytes(d.to_vec())));
        }
        Value::map(p)
    }

    /// The concrete shape, when it has no symbolic dimensions.
    pub fn sizes(&self) -> Option<Vec<u64>> {
        concrete(&self.shape)
    }

    pub fn numel(&self) -> Option<u64> {
        self.sizes().map(|s| numel(&s))
    }

    /// True when this tensor holds parameters, for the `params_total` check of
    /// R-M01.
    pub fn is_weight(&self) -> bool {
        matches!(
            self.semantic.as_deref(),
            Some("weight") | Some("bias") | Some("embedding") | None
        )
    }
}

/// A `TensorTable` (otype 0x0004).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorTable {
    pub tensors: BTreeMap<String, Ref>,
    /// Load-order hint (§04.2).
    pub order: Vec<String>,
    /// Logical groups for I/O planning, keyed by group name.
    pub groups: BTreeMap<String, Vec<String>>,
}

impl TensorTable {
    pub fn from_value(v: &Value) -> Res<TensorTable> {
        let t = v.get("t").and_then(|x| x.as_str());
        if t != Some("omni.tensor/table") {
            return Err(Error::Type(format!(
                "R-O02: expected a TensorTable, found `{}`",
                t.unwrap_or("<no t>")
            )));
        }
        let mut tensors = BTreeMap::new();
        for (k, val) in v
            .get("tensors")
            .and_then(|x| x.as_map())
            .ok_or_else(|| Error::Type("TensorTable has no `tensors` map".into()))?
        {
            let name = k
                .as_str()
                .ok_or_else(|| Error::Type("tensor names must be text".into()))?;
            tensors.insert(name.to_string(), crate::expr::parse_ref_value(val)?);
        }
        let order = v
            .get("order")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let mut groups = BTreeMap::new();
        for (k, val) in v.get("groups").and_then(|x| x.as_map()).unwrap_or(&[]) {
            let Some(name) = k.as_str() else { continue };
            let pats = val
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| x.as_str().unwrap_or_default().to_string())
                        .collect()
                })
                .unwrap_or_default();
            groups.insert(name.to_string(), pats);
        }
        Ok(TensorTable {
            tensors,
            order,
            groups,
        })
    }

    pub fn load(ctx: &Ctx<'_>, r: &Ref) -> Res<TensorTable> {
        TensorTable::from_value(&ctx.value(&r.1)?)
    }

    pub fn get(&self, name: &str) -> Option<&Ref> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Names matching a glob, in table order.
    pub fn select(&self, glob: &str) -> Vec<&String> {
        self.tensors
            .keys()
            .filter(|n| pattern::glob_match(glob, n))
            .collect()
    }

    /// The names in a declared group, resolved through its patterns.
    pub fn group(&self, name: &str) -> Vec<&String> {
        let Some(pats) = self.groups.get(name) else {
            return Vec::new();
        };
        self.tensors
            .keys()
            .filter(|n| pats.iter().any(|p| pattern::glob_match(p, n)))
            .collect()
    }

    /// The load order: the declared hint first, then anything it omits, so a
    /// partial `order` is a hint rather than a filter.
    pub fn load_order(&self) -> Vec<&String> {
        let mut out: Vec<&String> = Vec::with_capacity(self.tensors.len());
        for n in &self.order {
            if let Some((k, _)) = self.tensors.get_key_value(n) {
                out.push(k);
            }
        }
        for k in self.tensors.keys() {
            if !self.order.iter().any(|n| n == k) {
                out.push(k);
            }
        }
        out
    }
}

// ------------------------------------------------------------------ findings --

/// How bad a finding is. §15.1 requires the three outcomes to stay distinct:
/// reporting *indeterminate* as *invalid* is itself a conformance violation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// A normative rule is broken.
    Invalid,
    /// Cannot be decided by this build: an unimplemented feature, an absent
    /// object, an unsupported plugin.
    Indeterminate,
}

/// One V5 finding, carrying the rule ID so a report is traceable to the
/// specification (the engineering practice §roadmap calls spec↔code
/// traceability).
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub rule: &'static str,
    pub subject: String,
    pub message: String,
    pub severity: Severity,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {}: {}",
            match self.severity {
                Severity::Invalid => "invalid",
                Severity::Indeterminate => "indeterminate",
            },
            self.rule,
            self.subject,
            self.message
        )
    }
}

fn bad(rule: &'static str, subject: &str, message: impl Into<String>) -> Finding {
    Finding {
        rule,
        subject: subject.to_string(),
        message: message.into(),
        severity: Severity::Invalid,
    }
}

fn unknown(rule: &'static str, subject: &str, message: impl Into<String>) -> Finding {
    Finding {
        rule,
        subject: subject.to_string(),
        message: message.into(),
        severity: Severity::Indeterminate,
    }
}

impl TensorDesc {
    /// The V5 tensor rules that can be decided without materializing weights:
    /// R-T01 (declared type equals inferred type), R-T02 (chunk sizing),
    /// R-T03 (layout sufficiency), R-T04 (scheme consistency), R-T05 (depth)
    /// and R-T06 (lossy subtrees are marked).
    pub fn check(&self, ctx: &Ctx<'_>, name: &str) -> Vec<Finding> {
        let mut out = Vec::new();

        // R-T05 first: everything else walks the tree.
        if self.value.depth() > crate::expr::MAX_DEPTH {
            out.push(bad(
                "R-T05",
                name,
                format!(
                    "expression depth {} exceeds {}",
                    self.value.depth(),
                    crate::expr::MAX_DEPTH
                ),
            ));
            return out;
        }

        // R-T01.
        match self.value.check_declared(&self.shape, &self.dtype) {
            Ok(()) => {}
            Err(Error::Unsupported(m)) => out.push(unknown("R-T01", name, m)),
            Err(e) => out.push(bad("R-T01", name, e.to_string())),
        }

        // R-T03.
        if let Some(sizes) = self.sizes() {
            match self.layout.sufficiency(&sizes, &self.dtype) {
                Sufficiency::Sufficient => {}
                Sufficiency::Inconsistent(m) => out.push(bad("R-T03", name, m)),
                Sufficiency::NeedsContext(m) => out.push(unknown("R-T03", name, m)),
            }
        }

        // R-T02, over every `literal` leaf in the tree.
        for leaf in self.value.leaves() {
            if let Expr::Literal {
                chunks,
                dtype,
                shape,
                layout,
            } = leaf
            {
                let Some(sizes) = concrete(shape) else {
                    continue;
                };
                let Some(want) = layout.stored_bytes(&sizes, dtype) else {
                    continue;
                };
                match chunk_total(ctx, chunks) {
                    Ok(Some((total, summed))) => {
                        if total != summed {
                            out.push(bad(
                                "R-T02",
                                name,
                                format!(
                                    "ChunkList declares {total} bytes but its chunks hold {summed}"
                                ),
                            ));
                        }
                        if total != want {
                            out.push(bad(
                                "R-T02",
                                name,
                                format!(
                                    "ChunkList holds {total} bytes; {:?} of {} in a {} layout \
                                     needs {want}",
                                    sizes,
                                    dtype.label(),
                                    layout.kind()
                                ),
                            ));
                        }
                    }
                    Ok(None) => out.push(unknown(
                        "R-T02",
                        name,
                        "the tensor's chunks are not present in this store",
                    )),
                    Err(e) => out.push(bad("R-T02", name, e.to_string())),
                }
            }
        }

        // R-T04, over every quantization node.
        collect_quant(&self.value, &mut |x, scheme| {
            let sizes = x.infer().ok().and_then(|t| concrete(&t.shape));
            match crate::quant::Scheme::from_value(scheme) {
                Ok(s) => {
                    if let Some(sizes) = sizes {
                        if let Err(e) = s.check(&sizes) {
                            out.push(bad("R-T04", name, e.to_string()));
                        }
                    }
                }
                Err(e) => out.push(bad("R-T04", name, e.to_string())),
            }
        });

        // R-T06: a lossy subtree must be wrapped in `approx`, and this build
        // cannot tell whether an unmarked one is lossy — only that the marked
        // ones are declared. What it can check is the converse: an `approx`
        // node with a nonsensical bound.
        collect_approx(&self.value, &mut |bound| {
            let v = match bound {
                crate::expr::Bound::Abs(v) | crate::expr::Bound::Rel(v) => *v,
                crate::expr::Bound::Psnr(v) => *v,
            };
            if !v.is_finite() || v < 0.0 {
                out.push(bad(
                    "R-T06",
                    name,
                    format!("`approx` declares a bound of {v}, which states nothing"),
                ));
            }
        });

        // Plugins this build cannot evaluate make the tensor indeterminate,
        // not invalid — and the rest of the model is still readable (§04.7.7).
        for p in self.value.required_plugins() {
            out.push(unknown(
                "R-E06",
                name,
                format!("critical plugin `{p}` has no fallback and is not implemented here"),
            ));
        }

        out
    }

    /// R-T07: declared statistics must match recomputation. Separate from
    /// [`TensorDesc::check`] because it materializes the tensor.
    pub fn check_stats(&self, ctx: &Ctx<'_>, name: &str) -> Vec<Finding> {
        if self.stats.is_none() && self.digest_materialized.is_none() {
            return Vec::new();
        }
        let t = match self.value.eval(ctx) {
            Ok(t) => t,
            Err(Error::Unsupported(m)) | Err(Error::External(m)) => {
                return vec![unknown("R-T07", name, m)]
            }
            Err(e) => return vec![unknown("R-T07", name, e.to_string())],
        };
        let mut out = Vec::new();
        let declared = self.stats.clone().unwrap_or_default();
        let got = Stats::measure(&t.data);
        let mut cmp = |field: &str, a: Option<f64>, b: Option<f64>| {
            if let (Some(a), Some(b)) = (a, b) {
                // Declared statistics are usually written in a narrower type
                // than the accumulation, so the comparison is relative.
                let tol = 1e-6 * a.abs().max(1.0);
                if (a - b).abs() > tol {
                    out.push(bad(
                        "R-T07",
                        name,
                        format!("declared {field} {a} but measured {b}"),
                    ));
                }
            }
        };
        cmp("min", declared.min, got.min);
        cmp("max", declared.max, got.max);
        cmp("mean", declared.mean, got.mean);
        cmp("absmax", declared.absmax, got.absmax);
        for (field, a, b) in [
            ("nan", declared.nan, got.nan),
            ("inf", declared.inf, got.inf),
            ("nonzero", declared.nonzero, got.nonzero),
        ] {
            if let (Some(a), Some(b)) = (a, b) {
                if a != b {
                    out.push(bad(
                        "R-T07",
                        name,
                        format!("declared {field} {a} but measured {b}"),
                    ));
                }
            }
        }
        // `digest_materialized` is normative only over a fully deterministic
        // subtree (§04.2, §04.7.6), so a mismatch under an unpinned reduction
        // is indeterminate rather than invalid.
        if let Some(want) = self.digest_materialized {
            let bytes = t.to_bytes(&self.dtype, &self.layout, crate::dtype::Round::Rne);
            match bytes {
                Ok(b) => {
                    let got = ctx.store().hash().digest(&b);
                    if got != want {
                        let det = self.value.deterministic();
                        out.push(Finding {
                            rule: "R-T01",
                            subject: name.to_string(),
                            message: format!(
                                "digest_materialized is {} but this evaluation produced {}{}",
                                crate::sha256::hex(&want[..8]),
                                crate::sha256::hex(&got[..8]),
                                if det {
                                    ""
                                } else {
                                    " (the expression pins no reduction order, so the digest is \
                                     not normative — §04.7.6)"
                                }
                            ),
                            severity: if det {
                                Severity::Invalid
                            } else {
                                Severity::Indeterminate
                            },
                        });
                    }
                }
                Err(e) => out.push(unknown("R-T01", name, e.to_string())),
            }
        }
        out
    }
}

/// Reads a `ChunkList`'s declared total and the sum of its chunks' lengths.
/// `None` when the list itself is absent from the store.
fn chunk_total(ctx: &Ctx<'_>, r: &Ref) -> Res<Option<(u64, u64)>> {
    if r.0 == otype::BLOB {
        return match ctx.store().resolve(&r.1)? {
            Some(b) => Ok(Some((b.len() as u64, b.len() as u64))),
            None => Ok(None),
        };
    }
    let Some(bytes) = ctx.store().resolve(&r.1)? else {
        return Ok(None);
    };
    let cl = crate::cbor::decode(&bytes).map_err(|e| Error::Store(e.to_string()))?;
    let total = cl
        .get("total")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| Error::Type("ChunkList has no `total`".into()))?;
    let mut summed = 0u64;
    for c in cl
        .get("chunks")
        .and_then(|x| x.as_array())
        .ok_or_else(|| Error::Type("ChunkList has no `chunks`".into()))?
    {
        summed += c
            .get("n")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| Error::Type("chunk entry has no `n`".into()))?;
    }
    Ok(Some((total, summed)))
}

fn collect_quant(e: &Expr, f: &mut dyn FnMut(&Expr, &Value)) {
    match e {
        Expr::Dequantize { x, scheme } | Expr::Quantize { x, scheme, .. } => f(x, scheme),
        _ => {}
    }
    for c in e.children() {
        collect_quant(c, f);
    }
}

fn collect_approx(e: &Expr, f: &mut dyn FnMut(&crate::expr::Bound)) {
    if let Expr::Approx { bound, .. } = e {
        f(bound);
    }
    for c in e.children() {
        collect_approx(c, f);
    }
}

/// Validates every tensor in a table (V5, tensor rules).
pub fn validate_table(ctx: &Ctx<'_>, table: &TensorTable) -> Vec<Finding> {
    let mut out = Vec::new();
    for (name, r) in &table.tensors {
        match TensorDesc::load(ctx, r) {
            Ok(d) => out.extend(d.check(ctx, name)),
            Err(Error::Missing(_)) => out.push(unknown(
                "R-O05",
                name,
                "the tensor's descriptor is not present in this store",
            )),
            Err(e) => out.push(bad("R-O02", name, e.to_string())),
        }
    }
    out
}

/// R-M01: `params_total`, when present, equals the sum over weight-semantic
/// tensors.
pub fn check_params_total(ctx: &Ctx<'_>, table: &TensorTable, declared: u64) -> Vec<Finding> {
    let mut sum = 0u64;
    let mut unknown_shapes = 0usize;
    for r in table.tensors.values() {
        let Ok(d) = TensorDesc::load(ctx, r) else {
            unknown_shapes += 1;
            continue;
        };
        if !d.is_weight() {
            continue;
        }
        match d.numel() {
            Some(n) => sum += n,
            None => unknown_shapes += 1,
        }
    }
    if unknown_shapes > 0 {
        return vec![unknown(
            "R-M01",
            "metadata",
            format!("{unknown_shapes} tensors have no resolvable element count"),
        )];
    }
    if sum != declared {
        return vec![bad(
            "R-M01",
            "metadata",
            format!("params_total is {declared} but the weight tensors hold {sum}"),
        )];
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::HashAlgo;
    use crate::dtype::Round;
    use crate::expr::{dims, Scalar};
    use crate::store::{MemoryStore, WritableStore};

    fn desc(shape: &[u64], dtype: DType, value: Expr) -> TensorDesc {
        TensorDesc {
            shape: dims(shape),
            dtype,
            layout: Layout::default(),
            value,
            semantic: Some("weight".into()),
            role: None,
            axes: None,
            device_hint: None,
            materialize: Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        }
    }

    /// A literal whose ChunkList is properly formed.
    fn stored(s: &mut MemoryStore, shape: &[u64], dtype: &DType, data: &[f64]) -> Expr {
        let t = crate::expr::Tensor::new(shape.to_vec(), dtype.clone(), data.to_vec());
        let bytes = t.to_bytes(dtype, &Layout::default(), Round::Rne).unwrap();
        let blob = s.put(&bytes).unwrap();
        let cl = crate::container::Object::structure(
            otype::CHUNK_LIST,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/chunklist")),
                ("v", Value::U(1)),
                ("total", Value::U(bytes.len() as u64)),
                ("chunker", Value::map(vec![("k", Value::text("none"))])),
                (
                    "chunks",
                    Value::Array(vec![Value::map(vec![
                        (
                            "r",
                            Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
                        ),
                        ("n", Value::U(bytes.len() as u64)),
                    ])]),
                ),
            ]),
        );
        let d = s.put(&cl.payload).unwrap();
        Expr::Literal {
            chunks: (otype::CHUNK_LIST, d),
            dtype: dtype.clone(),
            shape: dims(shape),
            layout: Layout::default(),
        }
    }

    #[test]
    fn a_well_formed_descriptor_passes_every_rule_it_can() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(
            &mut s,
            &[2, 3],
            &DType::F32,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        let d = desc(&[2, 3], DType::F32, v);
        let ctx = Ctx::new(&s);
        assert_eq!(d.check(&ctx, "w"), vec![]);
        // And round-trips through its own encoding.
        let again = TensorDesc::from_value(&d.to_value()).unwrap();
        assert_eq!(again, d);
        let round = crate::cbor::decode(&d.to_value().encode()).unwrap();
        assert_eq!(TensorDesc::from_value(&round).unwrap(), d);
    }

    #[test]
    fn r_t01_catches_a_declared_shape_that_disagrees_with_the_value() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[2, 3], &DType::F32, &[0.0; 6]);
        let d = desc(&[3, 2], DType::F32, v.clone());
        let f = d.check(&Ctx::new(&s), "w");
        assert!(f.iter().any(|x| x.rule == "R-T01"
            && x.severity == Severity::Invalid
            && x.message.contains("declared shape")));
        // And a dtype that disagrees.
        let d = desc(&[2, 3], DType::BF16, v);
        let f = d.check(&Ctx::new(&s), "w");
        assert!(f
            .iter()
            .any(|x| x.rule == "R-T01" && x.message.contains("dtype")));
    }

    #[test]
    fn r_t02_catches_a_chunklist_that_lies_about_its_size() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let blob = s.put(&[0u8; 8]).unwrap();
        let cl = crate::container::Object::structure(
            otype::CHUNK_LIST,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/chunklist")),
                ("v", Value::U(1)),
                // Declares 24 bytes; holds 8.
                ("total", Value::U(24)),
                (
                    "chunks",
                    Value::Array(vec![Value::map(vec![
                        (
                            "r",
                            Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
                        ),
                        ("n", Value::U(8)),
                    ])]),
                ),
            ]),
        );
        let d = s.put(&cl.payload).unwrap();
        let value = Expr::Literal {
            chunks: (otype::CHUNK_LIST, d),
            dtype: DType::F32,
            shape: dims(&[2, 3]),
            layout: Layout::default(),
        };
        let f = desc(&[2, 3], DType::F32, value).check(&Ctx::new(&s), "w");
        let msgs: Vec<&str> = f.iter().map(|x| x.message.as_str()).collect();
        assert!(
            f.iter().filter(|x| x.rule == "R-T02").count() >= 1,
            "{msgs:?}"
        );
        assert!(msgs.iter().any(|m| m.contains("declares 24 bytes")));
    }

    #[test]
    fn r_t03_catches_an_impossible_layout() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[4], &DType::U4, &[0.0; 4]);
        let mut d = desc(&[4], DType::U4, v);
        d.layout = Layout::Packed {
            elems_per_word: 16,
            word_bits: 32,
            bit_order: crate::layout::BitOrder::LsbFirst,
            order: crate::layout::Order::RowMajor,
        };
        let f = d.check(&Ctx::new(&s), "w");
        assert!(f
            .iter()
            .any(|x| x.rule == "R-T03" && x.severity == Severity::Invalid));
        // An opaque layout is indeterminate, not invalid: it is legal and this
        // build cannot place its elements.
        d.layout = Layout::Opaque {
            id: "org.nvidia/tensorrt-weights.v10".into(),
        };
        let f = d.check(&Ctx::new(&s), "w");
        assert!(f
            .iter()
            .any(|x| x.rule == "R-T03" && x.severity == Severity::Indeterminate));
    }

    #[test]
    fn r_t04_catches_a_scale_tensor_that_does_not_match_its_blocks() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let q = stored(&mut s, &[2, 8], &DType::U8, &[1.0; 16]);
        let scale = stored(&mut s, &[3], &DType::F32, &[1.0, 2.0, 3.0]);
        let value = Expr::Dequantize {
            x: Box::new(q),
            scheme: Value::map(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(4)])),
                ("scale", scale.to_value()),
            ]),
        };
        let f = desc(&[2, 8], DType::F32, value).check(&Ctx::new(&s), "w");
        assert!(f.iter().any(|x| x.rule == "R-T04"), "{f:?}");
    }

    #[test]
    fn r_t07_compares_declared_statistics_against_recomputation() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[4], &DType::F32, &[-1.0, 0.0, 0.5, 2.0]);
        let mut d = desc(&[4], DType::F32, v);
        d.stats = Some(Stats {
            min: Some(-1.0),
            max: Some(2.0),
            mean: Some(0.375),
            absmax: Some(2.0),
            nan: Some(0),
            inf: Some(0),
            nonzero: Some(3),
        });
        let ctx = Ctx::new(&s);
        assert_eq!(d.check_stats(&ctx, "w"), vec![]);
        // A wrong non-zero count is caught exactly.
        d.stats.as_mut().unwrap().nonzero = Some(4);
        let f = d.check_stats(&ctx, "w");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].rule, "R-T07");
        assert_eq!(f[0].severity, Severity::Invalid);
    }

    #[test]
    fn digest_materialized_is_normative_only_over_a_deterministic_subtree() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let a = stored(&mut s, &[2, 2], &DType::F32, &[1.0, 2.0, 3.0, 4.0]);
        // A pinned reduction: the digest is normative, so a wrong one is
        // invalid.
        let pinned = Expr::MatMul {
            a: Box::new(a.clone()),
            b: Box::new(a.clone()),
            sum: crate::expr::Sum::Kahan,
        };
        let mut d = desc(&[2, 2], DType::F32, pinned);
        d.digest_materialized = Some([0u8; 32]);
        let f = d.check_stats(&Ctx::new(&s), "w");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Invalid);
        // Unpinned: the same mismatch is indeterminate, because §04.7.6 says
        // the digest is not normative there.
        let loose = Expr::MatMul {
            a: Box::new(a.clone()),
            b: Box::new(a),
            sum: crate::expr::Sum::Unspecified,
        };
        let mut d = desc(&[2, 2], DType::F32, loose);
        d.digest_materialized = Some([0u8; 32]);
        let f = d.check_stats(&Ctx::new(&s), "w");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Indeterminate);
    }

    #[test]
    fn an_unimplemented_plugin_is_indeterminate_not_invalid() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[2], &DType::F32, &[1.0, 2.0]);
        let value = Expr::Plugin {
            ns: "org.acme/quant".into(),
            name: "exotic".into(),
            v: 1,
            args: vec![v],
            attrs: Value::Map(vec![]),
            crit: true,
            shape: dims(&[2]),
            dtype: DType::F32,
            fallback: None,
        };
        let f = desc(&[2], DType::F32, value).check(&Ctx::new(&s), "w");
        assert!(
            f.iter()
                .any(|x| x.severity == Severity::Indeterminate
                    && x.message.contains("org.acme/quant"))
        );
        assert!(f.iter().all(|x| x.severity != Severity::Invalid));
    }

    #[test]
    fn an_approx_bound_that_states_nothing_is_invalid() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[2], &DType::F32, &[1.0, 2.0]);
        let value = Expr::Approx {
            x: Box::new(v),
            bound: crate::expr::Bound::Rel(f64::NAN),
        };
        let f = desc(&[2], DType::F32, value).check(&Ctx::new(&s), "w");
        assert!(f.iter().any(|x| x.rule == "R-T06"));
    }

    #[test]
    fn a_table_selects_groups_and_orders_its_load() {
        let v = Value::map(vec![
            ("t", Value::text("omni.tensor/table")),
            ("v", Value::U(1)),
            (
                "tensors",
                Value::Map(vec![
                    (
                        Value::text("model.embed_tokens.weight"),
                        Value::Array(vec![Value::U(5), Value::Bytes(vec![1u8; 32])]),
                    ),
                    (
                        Value::text("model.layers.0.attn.q_proj.weight"),
                        Value::Array(vec![Value::U(5), Value::Bytes(vec![2u8; 32])]),
                    ),
                    (
                        Value::text("model.layers.1.attn.q_proj.weight"),
                        Value::Array(vec![Value::U(5), Value::Bytes(vec![3u8; 32])]),
                    ),
                ]),
            ),
            (
                "groups",
                Value::Map(vec![(
                    Value::text("layer.0"),
                    Value::Array(vec![Value::text("model.layers.0.**")]),
                )]),
            ),
            (
                "order",
                Value::Array(vec![Value::text("model.embed_tokens.weight")]),
            ),
        ]);
        let t = TensorTable::from_value(&v).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(t.select("model.layers.*.attn.q_proj.weight").len(), 2);
        assert_eq!(t.group("layer.0").len(), 1);
        assert_eq!(t.group("nope").len(), 0);
        // The declared order comes first; the rest follow, so a partial hint is
        // a hint and not a filter.
        let order = t.load_order();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], "model.embed_tokens.weight");
    }

    #[test]
    fn the_wrong_object_type_is_a_ref_error_not_a_parse_crash() {
        let v = Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
        ]);
        assert!(TensorDesc::from_value(&v).is_err());
        assert!(TensorTable::from_value(&v).is_err());
    }

    #[test]
    fn params_total_is_recomputed() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = stored(&mut s, &[2, 3], &DType::F32, &[0.0; 6]);
        let d = desc(&[2, 3], DType::F32, v);
        let dd = s
            .put(&crate::container::Object::structure(otype::TENSOR_DESC, &d.to_value()).payload)
            .unwrap();
        let mut table = TensorTable::default();
        table.tensors.insert("w".into(), (otype::TENSOR_DESC, dd));
        let ctx = Ctx::new(&s);
        assert_eq!(check_params_total(&ctx, &table, 6), vec![]);
        let f = check_params_total(&ctx, &table, 7);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].rule, "R-M01");
        // And the whole-table walk is clean.
        assert_eq!(validate_table(&ctx, &table), vec![]);
    }

    #[test]
    fn a_missing_descriptor_is_indeterminate() {
        let s = MemoryStore::new(HashAlgo::default());
        let mut table = TensorTable::default();
        table
            .tensors
            .insert("w".into(), (otype::TENSOR_DESC, [9u8; 32]));
        let f = validate_table(&Ctx::new(&s), &table);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Indeterminate);
    }

    #[test]
    fn stats_measure_counts_nan_and_inf_separately() {
        let s = Stats::measure(&[1.0, -2.0, 0.0, f64::NAN, f64::INFINITY]);
        assert_eq!(s.nan, Some(1));
        assert_eq!(s.inf, Some(1));
        assert_eq!(s.nonzero, Some(2));
        assert_eq!(s.min, Some(-2.0));
        assert_eq!(s.max, Some(1.0));
        assert_eq!(s.absmax, Some(2.0));
        // The mean is over finite elements only, which is the only definition
        // that does not poison the whole field with one NaN.
        assert_eq!(s.mean, Some(-1.0 / 3.0));
        // An all-NaN tensor has no min or max to report, and says so.
        let s = Stats::measure(&[f64::NAN]);
        assert_eq!(s.min, None);
        assert_eq!(s.mean, None);
    }

    #[test]
    fn a_scalar_expression_with_no_literals_needs_no_chunks() {
        let s = MemoryStore::new(HashAlgo::default());
        let d = desc(
            &[4],
            DType::F32,
            Expr::Full {
                value: Scalar::Int(0),
                dtype: DType::F32,
                shape: dims(&[4]),
            },
        );
        assert_eq!(d.check(&Ctx::new(&s), "zeros"), vec![]);
    }
}
