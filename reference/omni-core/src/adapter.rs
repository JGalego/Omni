//! §08 — adapters, attachment, and composition.
//!
//! An adapter is an expression over a parent's tensors. Nothing is merged and
//! nothing is copied: `attach` produces, for each matched base tensor, a new
//! expression whose leaves are the base's stored bytes and the adapter's own
//! small factors.
//!
//! The eight arithmetic methods of §08.2 — LoRA, DoRA, IA³, LoHa, LoKr, VeRA,
//! AdaLoRA, BitFit — need no format extension at all, and this module builds
//! each of them out of core nodes to demonstrate it. The three graph-level
//! methods (prompt, prefix, bottleneck) are carried as declarative rewrites for
//! §07 to apply.
//!
//! Attachment is the part that has to be robust: an adapter must bind to a base
//! it has never seen, without string-matching fragility. §08.3's answer is
//! selectors plus a `require` block, and [`Adapter::check`] reports unmatched
//! selectors, shape mismatches and missing base tensors *before* any weights
//! load — R-A01 through R-A03.
//!
//! ## Where the specification and the algebra do not quite meet
//!
//! §08.5 lists `ties`, `dare` and `slerp` as composition modes and says OMNI
//! "expresses them as expressions with declared seeds". Two of the three cannot
//! be pure expressions over the core node set as it stands: TIES needs an
//! elementwise magnitude comparison to trim, and SLERP needs a dot product and
//! trigonometry, and §04.7.2 has neither. What *is* expressible — and what this
//! module does — is the result: SLERP folds into two exact `scale` coefficients
//! computed once from the parents, and TIES and DARE produce a sparse delta.
//! Both are reproducible from their recipe, which is the property §08.5 actually
//! wants; but a reader should know that the recipe is replayed by a tool rather
//! than evaluated by the expression evaluator.

use crate::cbor::Value;
use crate::container::Digest;
use crate::dtype::DType;
use crate::expr::{concrete, BinOp, Ctx, Dim, Error, Expr, Ref, Scalar, Sum, Tensor};
use crate::pattern::{self, Regex};
use crate::tensor::{Finding, Severity, TensorDesc, TensorTable};
use std::collections::BTreeMap;

type Res<T> = Result<T, Error>;

/// The adapter methods of §08.1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Method {
    Lora,
    Dora,
    Ia3,
    Loha,
    Lokr,
    Vera,
    AdaLora,
    Bone,
    BitFit,
    /// Graph-level: extra virtual tokens or per-layer KV prefixes (§08.4).
    Prompt,
    Prefix,
    PTuning,
    AdapterBottleneck,
    /// Activation-level (§08.2).
    ControlVector,
    Plugin(String),
}

impl Method {
    pub fn name(&self) -> &str {
        match self {
            Method::Lora => "lora",
            Method::Dora => "dora",
            Method::Ia3 => "ia3",
            Method::Loha => "loha",
            Method::Lokr => "lokr",
            Method::Vera => "vera",
            Method::AdaLora => "adalora",
            Method::Bone => "bone",
            Method::BitFit => "bitfit",
            Method::Prompt => "prompt",
            Method::Prefix => "prefix",
            Method::PTuning => "p-tuning",
            Method::AdapterBottleneck => "adapter-bottleneck",
            Method::ControlVector => "control-vector",
            Method::Plugin(n) => n,
        }
    }

    pub fn parse(s: &str) -> Method {
        match s {
            "lora" => Method::Lora,
            "dora" => Method::Dora,
            "ia3" => Method::Ia3,
            "loha" => Method::Loha,
            "lokr" => Method::Lokr,
            "vera" => Method::Vera,
            "adalora" => Method::AdaLora,
            "bone" => Method::Bone,
            "bitfit" => Method::BitFit,
            "prompt" => Method::Prompt,
            "prefix" => Method::Prefix,
            "p-tuning" => Method::PTuning,
            "adapter-bottleneck" => Method::AdapterBottleneck,
            "control-vector" => Method::ControlVector,
            other => Method::Plugin(other.to_string()),
        }
    }

    /// True when the method changes the computation rather than the weights, and
    /// therefore needs the graph-level attachment of §08.4.
    pub fn is_graph_level(&self) -> bool {
        matches!(
            self,
            Method::Prompt | Method::Prefix | Method::PTuning | Method::AdapterBottleneck
        )
    }
}

/// How a base tensor is selected (§08.3).
#[derive(Clone, Debug)]
pub enum Select {
    Glob(String),
    Regex(Regex),
    Semantic(String),
    Role(String),
    /// Matches when the base tensor's `axes` are exactly these names. §08.3
    /// calls selecting by role and axes the robust option, because it survives
    /// renaming between model releases.
    Axes(Vec<String>),
}

impl Select {
    fn from_value(v: &Value) -> Res<Select> {
        if let Some(g) = v.get("glob").and_then(|x| x.as_str()) {
            return Ok(Select::Glob(g.to_string()));
        }
        if let Some(r) = v.get("regex").and_then(|x| x.as_str()) {
            return Ok(Select::Regex(
                Regex::parse(r).map_err(|e| Error::Type(e.to_string()))?,
            ));
        }
        if let Some(s) = v.get("semantic").and_then(|x| x.as_str()) {
            return Ok(Select::Semantic(s.to_string()));
        }
        if let Some(s) = v.get("role").and_then(|x| x.as_str()) {
            return Ok(Select::Role(s.to_string()));
        }
        if let Some(a) = v.get("axes").and_then(|x| x.as_array()) {
            return Ok(Select::Axes(
                a.iter()
                    .map(|x| x.as_str().unwrap_or_default().to_string())
                    .collect(),
            ));
        }
        Err(Error::Type(
            "selector must be one of glob, regex, semantic, role or axes".into(),
        ))
    }

    fn to_value(&self) -> Value {
        match self {
            Select::Glob(g) => Value::map(vec![("glob", Value::text(g.clone()))]),
            Select::Regex(r) => Value::map(vec![("regex", Value::text(r.as_str().to_string()))]),
            Select::Semantic(s) => Value::map(vec![("semantic", Value::text(s.clone()))]),
            Select::Role(s) => Value::map(vec![("role", Value::text(s.clone()))]),
            Select::Axes(a) => Value::map(vec![(
                "axes",
                Value::Array(a.iter().map(|x| Value::text(x.clone())).collect()),
            )]),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Select::Glob(g) => format!("glob `{g}`"),
            Select::Regex(r) => format!("regex `{}`", r.as_str()),
            Select::Semantic(s) => format!("semantic `{s}`"),
            Select::Role(s) => format!("role `{s}`"),
            Select::Axes(a) => format!("axes {a:?}"),
        }
    }

    /// Matches a base tensor, returning its captures. Selectors that are not
    /// patterns produce no captures, so a rule that binds `{1}` against a
    /// `role` selector fails loudly rather than binding the wrong tensor.
    pub fn matches(&self, name: &str, desc: &TensorDesc) -> Result<Option<Vec<String>>, Error> {
        Ok(match self {
            Select::Glob(g) => pattern::glob_captures(g, name),
            Select::Regex(r) => r.captures(name).map_err(|e| Error::Type(e.to_string()))?,
            Select::Semantic(s) => {
                if desc.semantic.as_deref() == Some(s.as_str()) {
                    Some(vec![])
                } else {
                    None
                }
            }
            Select::Role(s) => {
                if desc.role.as_deref() == Some(s.as_str()) {
                    Some(vec![])
                } else {
                    None
                }
            }
            Select::Axes(a) => match &desc.axes {
                Some(axes) if axes == a => Some(vec![]),
                _ => None,
            },
        })
    }
}

/// What an attachment rule assumes about the base tensor (§08.3). A mismatch is
/// a hard, early error with a clear message instead of silently wrong math.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Require {
    pub axes: Option<Vec<String>>,
    pub rank_axis: Option<String>,
    pub shape: Option<Vec<u64>>,
    pub dtype: Option<DType>,
}

impl Require {
    fn from_value(v: &Value) -> Res<Require> {
        Ok(Require {
            axes: v.get("axes").and_then(|x| x.as_array()).map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or_default().to_string())
                    .collect()
            }),
            rank_axis: v
                .get("rank_axis")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            shape: v
                .get("shape")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect()),
            dtype: match v.get("dtype") {
                Some(d) => Some(DType::from_value(d).map_err(Error::Type)?),
                None => None,
            },
        })
    }

    fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = Vec::new();
        if let Some(a) = &self.axes {
            p.push((
                "axes",
                Value::Array(a.iter().map(|x| Value::text(x.clone())).collect()),
            ));
        }
        if let Some(r) = &self.rank_axis {
            p.push(("rank_axis", Value::text(r.clone())));
        }
        if let Some(s) = &self.shape {
            p.push((
                "shape",
                Value::Array(s.iter().map(|x| Value::U(*x)).collect()),
            ));
        }
        if let Some(d) = &self.dtype {
            p.push(("dtype", d.to_value()));
        }
        Value::map(p)
    }

    /// Checks the assumptions against a matched base tensor.
    fn check(&self, name: &str, desc: &TensorDesc) -> Vec<Finding> {
        let mut out = Vec::new();
        if let Some(want) = &self.axes {
            match &desc.axes {
                Some(got) if got == want => {}
                Some(got) => out.push(Finding {
                    rule: "R-A03",
                    subject: name.to_string(),
                    message: format!("adapter requires axes {want:?}; the base declares {got:?}"),
                    severity: Severity::Invalid,
                }),
                None => out.push(Finding {
                    rule: "R-A03",
                    subject: name.to_string(),
                    message: format!(
                        "adapter requires axes {want:?}; the base tensor declares none, so \
                         attachment cannot be checked"
                    ),
                    severity: Severity::Indeterminate,
                }),
            }
        }
        if let Some(axis) = &self.rank_axis {
            let known = desc
                .axes
                .as_ref()
                .is_some_and(|a| a.iter().any(|x| x == axis));
            if !known {
                out.push(Finding {
                    rule: "R-A03",
                    subject: name.to_string(),
                    message: format!(
                        "adapter names `{axis}` as its rank axis, but the base tensor has no \
                         such axis"
                    ),
                    severity: Severity::Invalid,
                });
            }
        }
        if let Some(want) = &self.shape {
            if desc.sizes().as_deref() != Some(want.as_slice()) {
                out.push(Finding {
                    rule: "R-A03",
                    subject: name.to_string(),
                    message: format!(
                        "adapter requires shape {want:?}; the base tensor is {:?}",
                        desc.shape
                    ),
                    severity: Severity::Invalid,
                });
            }
        }
        if let Some(want) = &self.dtype {
            if &desc.dtype != want {
                out.push(Finding {
                    rule: "R-A03",
                    subject: name.to_string(),
                    message: format!(
                        "adapter requires dtype {}; the base tensor is {}",
                        want.label(),
                        desc.dtype.label()
                    ),
                    severity: Severity::Invalid,
                });
            }
        }
        out
    }
}

/// What an attachment rule does to a matched tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachKind {
    /// Combine the base tensor with an expression: `apply.op(W, with)`.
    TensorTransform,
    /// Replace the base tensor's value outright.
    TensorReplace,
    /// A graph rewrite (§08.4), applied by §07 rather than here.
    GraphPatch,
}

/// One attachment rule (§08.3).
#[derive(Clone, Debug)]
pub struct Attach {
    pub select: Select,
    pub kind: AttachKind,
    /// `{"op": …, "with": <expression template>}`. Placeholders are `$name`
    /// strings resolved through `bind`.
    pub apply: Value,
    pub bind: Vec<(String, String)>,
    pub require: Require,
}

impl Attach {
    fn from_value(v: &Value) -> Res<Attach> {
        let select = Select::from_value(
            v.get("select")
                .ok_or_else(|| Error::Type("attach rule has no `select`".into()))?,
        )?;
        let kind = match v.get("kind").and_then(|x| x.as_str()) {
            Some("tensor-transform") | None => AttachKind::TensorTransform,
            Some("tensor-replace") => AttachKind::TensorReplace,
            Some("graph-patch") => AttachKind::GraphPatch,
            Some(k) => return Err(Error::Type(format!("unknown attach kind `{k}`"))),
        };
        let mut bind = Vec::new();
        for (k, val) in v.get("bind").and_then(|x| x.as_map()).unwrap_or(&[]) {
            let (Some(k), Some(val)) = (k.as_str(), val.as_str()) else {
                return Err(Error::Type("`bind` maps placeholder to tensor name".into()));
            };
            bind.push((k.to_string(), val.to_string()));
        }
        Ok(Attach {
            select,
            kind,
            apply: v
                .get("apply")
                .cloned()
                .unwrap_or_else(|| Value::Map(vec![])),
            bind,
            require: match v.get("require") {
                Some(r) => Require::from_value(r)?,
                None => Require::default(),
            },
        })
    }

    fn to_value(&self) -> Value {
        Value::map(vec![
            ("select", self.select.to_value()),
            (
                "kind",
                Value::text(match self.kind {
                    AttachKind::TensorTransform => "tensor-transform",
                    AttachKind::TensorReplace => "tensor-replace",
                    AttachKind::GraphPatch => "graph-patch",
                }),
            ),
            ("apply", self.apply.clone()),
            (
                "bind",
                Value::Map(
                    self.bind
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), Value::text(v.clone())))
                        .collect(),
                ),
            ),
            ("require", self.require.to_value()),
        ])
    }
}

/// The compatibility assumptions of §08.1's `base_compat`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BaseCompat {
    /// Per-tensor-pattern expectations about the base.
    pub tensors: BTreeMap<String, Require>,
    /// Digest of the base's `meta.arch`, for a fast check.
    pub arch_digest: Option<Digest>,
}

/// An `Adapter` object (otype 0x000C, §08.1).
#[derive(Clone, Debug)]
pub struct Adapter {
    pub method: Method,
    /// The base model's `Manifest`. Required, and pinned by digest, so an
    /// adapter can never silently attach to a different base.
    pub base: Ref,
    pub base_compat: BaseCompat,
    pub rank: Option<u64>,
    pub alpha: Option<f64>,
    pub dropout: Option<f64>,
    pub targets: Vec<String>,
    /// The adapter's own `TensorTable`.
    pub tensors: Ref,
    pub attach: Vec<Attach>,
    pub scale_default: f64,
    pub merge_policy: String,
    /// Declarative graph rewrites for §08.4 methods, carried verbatim.
    pub graph_patches: Vec<Value>,
    pub allow_unmatched: bool,
    pub trained_on: Option<Ref>,
    pub provenance: Option<Ref>,
}

impl Adapter {
    pub fn from_value(v: &Value) -> Res<Adapter> {
        let t = v.get("t").and_then(|x| x.as_str());
        if t != Some("omni.adapt/adapter") {
            return Err(Error::Type(format!(
                "R-O02: expected an Adapter, found `{}`",
                t.unwrap_or("<no t>")
            )));
        }
        let f = |k: &str| match v.get(k) {
            Some(Value::F64(x)) => Some(*x),
            Some(Value::U(n)) => Some(*n as f64),
            Some(Value::I(n)) => Some(*n as f64),
            _ => None,
        };
        let opt_ref = |k: &str| -> Res<Option<Ref>> {
            match v.get(k) {
                Some(r) => Ok(Some(crate::expr::parse_ref_value(r)?)),
                None => Ok(None),
            }
        };
        let mut compat = BaseCompat::default();
        if let Some(bc) = v.get("base_compat") {
            for (k, val) in bc.get("tensors").and_then(|x| x.as_map()).unwrap_or(&[]) {
                if let Some(k) = k.as_str() {
                    compat
                        .tensors
                        .insert(k.to_string(), Require::from_value(val)?);
                }
            }
            compat.arch_digest = bc
                .get("arch_digest")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok());
        }
        let mut attach = Vec::new();
        for a in v.get("attach").and_then(|x| x.as_array()).unwrap_or(&[]) {
            attach.push(Attach::from_value(a)?);
        }
        Ok(Adapter {
            method: Method::parse(v.get("method").and_then(|x| x.as_str()).unwrap_or("plugin")),
            base: crate::expr::parse_ref_value(
                v.get("base")
                    .ok_or_else(|| Error::Type("an adapter must name its `base` (§08.1)".into()))?,
            )?,
            base_compat: compat,
            rank: v.get("rank").and_then(|x| x.as_u64()),
            alpha: f("alpha"),
            dropout: f("dropout"),
            targets: v
                .get("targets")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .map(|x| x.as_str().unwrap_or_default().to_string())
                        .collect()
                })
                .unwrap_or_default(),
            tensors: crate::expr::parse_ref_value(
                v.get("tensors")
                    .ok_or_else(|| Error::Type("an adapter must carry `tensors`".into()))?,
            )?,
            attach,
            scale_default: f("scale_default").unwrap_or(1.0),
            merge_policy: v
                .get("merge_policy")
                .and_then(|x| x.as_str())
                .unwrap_or("runtime")
                .to_string(),
            graph_patches: v
                .get("graph_patches")
                .and_then(|x| x.as_array())
                .map(|a| a.to_vec())
                .unwrap_or_default(),
            allow_unmatched: matches!(v.get("allow_unmatched"), Some(Value::Bool(true))),
            trained_on: opt_ref("trained_on")?,
            provenance: opt_ref("provenance")?,
        })
    }

    pub fn load(ctx: &Ctx<'_>, r: &Ref) -> Res<Adapter> {
        Adapter::from_value(&ctx.value(&r.1)?)
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.adapt/adapter")),
            ("v", Value::U(1)),
            ("method", Value::text(self.method.name().to_string())),
            ("base", ref_value(&self.base)),
        ];
        if !self.base_compat.tensors.is_empty() || self.base_compat.arch_digest.is_some() {
            let mut bc: Vec<(&str, Value)> = Vec::new();
            if !self.base_compat.tensors.is_empty() {
                bc.push((
                    "tensors",
                    Value::Map(
                        self.base_compat
                            .tensors
                            .iter()
                            .map(|(k, r)| (Value::text(k.clone()), r.to_value()))
                            .collect(),
                    ),
                ));
            }
            if let Some(d) = self.base_compat.arch_digest {
                bc.push(("arch_digest", Value::Bytes(d.to_vec())));
            }
            p.push(("base_compat", Value::map(bc)));
        }
        if let Some(r) = self.rank {
            p.push(("rank", Value::U(r)));
        }
        if let Some(a) = self.alpha {
            p.push(("alpha", Value::F64(a)));
        }
        if let Some(d) = self.dropout {
            p.push(("dropout", Value::F64(d)));
        }
        if !self.targets.is_empty() {
            p.push((
                "targets",
                Value::Array(
                    self.targets
                        .iter()
                        .map(|x| Value::text(x.clone()))
                        .collect(),
                ),
            ));
        }
        p.push(("tensors", ref_value(&self.tensors)));
        p.push((
            "attach",
            Value::Array(self.attach.iter().map(|a| a.to_value()).collect()),
        ));
        p.push(("scale_default", Value::F64(self.scale_default)));
        p.push(("merge_policy", Value::text(self.merge_policy.clone())));
        if !self.graph_patches.is_empty() {
            p.push(("graph_patches", Value::Array(self.graph_patches.clone())));
        }
        if self.allow_unmatched {
            p.push(("allow_unmatched", Value::Bool(true)));
        }
        if let Some(r) = &self.trained_on {
            p.push(("trained_on", ref_value(r)));
        }
        if let Some(r) = &self.provenance {
            p.push(("provenance", ref_value(r)));
        }
        Value::map(p)
    }

    /// The exact rational α/r of §08.2, kept exact so two implementations agree
    /// in the last bit.
    pub fn lora_scale(&self) -> Scalar {
        match (self.alpha, self.rank) {
            (Some(a), Some(r)) if r > 0 && a.fract() == 0.0 => Scalar::Ratio(a as i64, r as i64),
            (Some(a), Some(r)) if r > 0 => Scalar::Float(a / r as f64),
            _ => Scalar::Float(self.scale_default),
        }
    }
}

fn ref_value(r: &Ref) -> Value {
    Value::Array(vec![Value::U(r.0 as u64), Value::Bytes(r.1.to_vec())])
}

// ----------------------------------------------------------------- attachment --

/// One attached tensor: the base tensor's name and the expression that replaces
/// its value.
#[derive(Clone, Debug)]
pub struct Binding {
    pub tensor: String,
    pub expr: Expr,
    /// The adapter tensors this binding consumed.
    pub used: Vec<String>,
}

/// The result of attaching an adapter to a base (§08.3).
#[derive(Clone, Debug, Default)]
pub struct AttachReport {
    pub bindings: Vec<Binding>,
    /// Selectors that matched nothing.
    pub unmatched: Vec<String>,
    pub findings: Vec<Finding>,
    /// Graph rewrites the adapter ships, for §07 to apply.
    pub graph_patches: Vec<Value>,
}

impl AttachReport {
    pub fn is_ok(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == Severity::Invalid)
    }
}

impl Adapter {
    /// Attaches to a base, producing one expression per matched tensor.
    ///
    /// `base` is the base model's tensor table; `base_ctx` resolves both the
    /// base's and the adapter's objects (a layered store, typically).
    pub fn attach(&self, ctx: &Ctx<'_>, base: &TensorTable) -> Res<AttachReport> {
        let own = TensorTable::load(ctx, &self.tensors)?;
        let mut report = AttachReport {
            graph_patches: self.graph_patches.clone(),
            ..Default::default()
        };

        // R-A01: the base is pinned by digest, so either it resolves or the
        // adapter is incomplete — never silently attached to something else.
        if !ctx.store().has(&self.base.1)? {
            report.findings.push(Finding {
                rule: "R-A01",
                subject: "base".into(),
                message: format!(
                    "the base manifest {} is not present; the adapter is incomplete, not invalid",
                    crate::sha256::hex(&self.base.1[..8])
                ),
                severity: Severity::Indeterminate,
            });
        }

        for rule in &self.attach {
            if rule.kind == AttachKind::GraphPatch {
                continue;
            }
            let mut matched = 0usize;
            for (name, r) in &base.tensors {
                let desc = match TensorDesc::load(ctx, r) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let Some(caps) = rule.select.matches(name, &desc)? else {
                    continue;
                };
                matched += 1;
                report.findings.extend(rule.require.check(name, &desc));

                // Resolve the bindings for this match.
                let mut binds: BTreeMap<String, Value> = BTreeMap::new();
                let mut used = Vec::new();
                let mut failed = false;
                for (placeholder, template) in &rule.bind {
                    let want = match pattern::substitute(template, &caps) {
                        Ok(n) => n,
                        Err(e) => {
                            report.findings.push(Finding {
                                rule: "R-A03",
                                subject: name.clone(),
                                message: e.to_string(),
                                severity: Severity::Invalid,
                            });
                            failed = true;
                            break;
                        }
                    };
                    let Some(ar) = own.get(&want) else {
                        report.findings.push(Finding {
                            rule: "R-A03",
                            subject: name.clone(),
                            message: format!(
                                "`{placeholder}` binds to adapter tensor `{want}`, which the \
                                 adapter does not have"
                            ),
                            severity: Severity::Invalid,
                        });
                        failed = true;
                        break;
                    };
                    let ad = TensorDesc::load(ctx, ar)?;
                    binds.insert(placeholder.clone(), ad.value.to_value());
                    used.push(want);
                }
                if failed {
                    continue;
                }

                match self.build(rule, &desc, &binds) {
                    Ok(expr) => {
                        // The merged expression must have the base tensor's own
                        // type, or loading it would change the model's shape.
                        match expr.check_declared(&desc.shape, &desc.dtype) {
                            Ok(()) => report.bindings.push(Binding {
                                tensor: name.clone(),
                                expr,
                                used,
                            }),
                            Err(e) => report.findings.push(Finding {
                                rule: "R-A03",
                                subject: name.clone(),
                                message: format!("attached expression does not fit the base: {e}"),
                                severity: Severity::Invalid,
                            }),
                        }
                    }
                    Err(e) => report.findings.push(Finding {
                        rule: "R-A03",
                        subject: name.clone(),
                        message: e.to_string(),
                        severity: Severity::Invalid,
                    }),
                }
            }
            // R-A02.
            if matched == 0 && !self.allow_unmatched {
                report.unmatched.push(rule.select.describe());
                report.findings.push(Finding {
                    rule: "R-A02",
                    subject: rule.select.describe(),
                    message: "selector matched no base tensor, and the adapter does not declare \
                              `allow_unmatched`"
                        .into(),
                    severity: Severity::Invalid,
                });
            }
        }
        Ok(report)
    }

    /// Builds the expression a rule produces for one matched tensor.
    fn build(
        &self,
        rule: &Attach,
        desc: &TensorDesc,
        binds: &BTreeMap<String, Value>,
    ) -> Res<Expr> {
        let with = rule
            .apply
            .get("with")
            .ok_or_else(|| Error::Type("`apply` needs a `with` expression".into()))?;
        let substituted = substitute(with, binds)?;
        let patch = Expr::from_value(&substituted)?;
        match rule.kind {
            AttachKind::TensorReplace => Ok(patch),
            AttachKind::GraphPatch => Err(Error::Type(
                "graph patches are applied by §07, not by tensor attachment".into(),
            )),
            AttachKind::TensorTransform => {
                let op = rule
                    .apply
                    .get("op")
                    .and_then(|x| x.as_str())
                    .unwrap_or("add");
                let base = Box::new(desc.value.clone());
                Ok(match op {
                    "add" => Expr::Bin {
                        op: BinOp::Add,
                        a: base,
                        b: Box::new(patch),
                    },
                    "sub" => Expr::Bin {
                        op: BinOp::Sub,
                        a: base,
                        b: Box::new(patch),
                    },
                    "mul" => Expr::Bin {
                        op: BinOp::Mul,
                        a: base,
                        b: Box::new(patch),
                    },
                    "div" => Expr::Bin {
                        op: BinOp::Div,
                        a: base,
                        b: Box::new(patch),
                    },
                    other => {
                        return Err(Error::Type(format!(
                            "`apply.op` must be an elementwise combiner, got `{other}`"
                        )))
                    }
                })
            }
        }
    }

    /// R-A01–R-A03 without building anything: what `omni adapter check` reports.
    pub fn check(&self, ctx: &Ctx<'_>, base: &TensorTable) -> Res<AttachReport> {
        let mut r = self.attach(ctx, base)?;
        // Base-compat patterns are assumptions about the base as a whole, not
        // per-rule, so they are checked here.
        for (pat, req) in &self.base_compat.tensors {
            let names: Vec<String> = base.select(pat).into_iter().cloned().collect();
            if names.is_empty() {
                r.findings.push(Finding {
                    rule: "R-A03",
                    subject: pat.clone(),
                    message: "base_compat names a tensor pattern the base does not have".into(),
                    severity: Severity::Invalid,
                });
                continue;
            }
            for n in names {
                if let Some(br) = base.get(&n) {
                    if let Ok(d) = TensorDesc::load(ctx, br) {
                        r.findings.extend(req.check(&n, &d));
                    }
                }
            }
        }
        if self.method.is_graph_level() && self.graph_patches.is_empty() {
            r.findings.push(Finding {
                rule: "R-A03",
                subject: self.method.name().to_string(),
                message: "a graph-level method must ship the rewrites that install it (§08.4)"
                    .into(),
                severity: Severity::Invalid,
            });
        }
        Ok(r)
    }
}

/// Substitutes `$name` placeholders in an expression template.
pub fn substitute(template: &Value, binds: &BTreeMap<String, Value>) -> Res<Value> {
    Ok(match template {
        Value::Text(s) if s.starts_with('$') => binds
            .get(s)
            .cloned()
            .ok_or_else(|| Error::Type(format!("`{s}` is not bound by this rule")))?,
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|x| substitute(x, binds))
                .collect::<Res<Vec<Value>>>()?,
        ),
        Value::Map(m) => Value::Map(
            m.iter()
                .map(|(k, v)| Ok((k.clone(), substitute(v, binds)?)))
                .collect::<Res<Vec<(Value, Value)>>>()?,
        ),
        Value::Tag(t, inner) => Value::Tag(*t, Box::new(substitute(inner, binds)?)),
        other => other.clone(),
    })
}

// -------------------------------------------------------- methods as templates --

/// The canonical attachment rule for each arithmetic method of §08.2, as an
/// expression over core nodes. Tooling emits these; a runtime only ever sees the
/// resulting expression.
pub fn method_template(method: &Method, scale: Scalar) -> Res<Value> {
    let k = scale.to_value();
    let matmul = |a: &str, b: &str| {
        Value::map(vec![
            ("op", Value::text("matmul")),
            ("a", Value::text(a)),
            ("b", Value::text(b)),
        ])
    };
    let scaled = |x: Value| {
        Value::map(vec![
            ("op", Value::text("scale")),
            ("x", x),
            ("k", k.clone()),
        ])
    };
    Ok(match method {
        // add(W, scale(matmul(B, A), alpha/r))
        Method::Lora => Value::map(vec![
            ("op", Value::text("add")),
            ("with", scaled(matmul("$B", "$A"))),
        ]),
        // AdaLoRA is LoRA with a learned per-rank gate.
        Method::AdaLora => Value::map(vec![
            ("op", Value::text("add")),
            (
                "with",
                scaled(matmul_of(
                    Value::map(vec![
                        ("op", Value::text("mul")),
                        ("a", Value::text("$B")),
                        ("b", Value::text("$Lambda")),
                    ]),
                    Value::text("$A"),
                )),
            ),
        ]),
        // VeRA shares frozen random factors and learns only two vectors.
        Method::Vera => Value::map(vec![
            ("op", Value::text("add")),
            (
                "with",
                scaled(matmul_of(
                    Value::map(vec![
                        ("op", Value::text("mul")),
                        ("a", Value::text("$B")),
                        ("b", Value::text("$d")),
                    ]),
                    Value::map(vec![
                        ("op", Value::text("mul")),
                        ("a", Value::text("$A")),
                        ("b", Value::text("$b")),
                    ]),
                )),
            ),
        ]),
        // LoHa: the Hadamard product of two low-rank terms.
        Method::Loha => Value::map(vec![
            ("op", Value::text("add")),
            (
                "with",
                scaled(Value::map(vec![
                    ("op", Value::text("mul")),
                    ("a", matmul("$B1", "$A1")),
                    ("b", matmul("$B2", "$A2")),
                ])),
            ),
        ]),
        // IA^3 rescales the weight along one axis.
        Method::Ia3 => Value::map(vec![
            ("op", Value::text("mul")),
            ("with", Value::text("$l")),
        ]),
        // BitFit shifts a bias.
        Method::BitFit => Value::map(vec![
            ("op", Value::text("add")),
            ("with", Value::text("$db")),
        ]),
        // Control vectors add a scaled direction at declared graph points; the
        // tensor-level form is the same arithmetic.
        Method::ControlVector => Value::map(vec![
            ("op", Value::text("add")),
            ("with", scaled(Value::text("$v"))),
        ]),
        // Bone and LoKr need a reshape/kron composition that depends on the
        // target's shape, so the template is emitted per tensor by the tool
        // rather than as a fixed shape-free form.
        Method::Lokr | Method::Bone | Method::Dora => {
            return Err(Error::Unsupported(format!(
                "`{}` needs the target tensor's shape to build its template; use \
                 `dora_template` or emit the rule per tensor",
                method.name()
            )))
        }
        m if m.is_graph_level() => {
            return Err(Error::Unsupported(format!(
                "`{}` is a graph-level method: it ships rewrites (§08.4), not a tensor transform",
                m.name()
            )))
        }
        m => {
            return Err(Error::Unsupported(format!(
                "no core-node template for method `{}`",
                m.name()
            )))
        }
    })
}

fn matmul_of(a: Value, b: Value) -> Value {
    Value::map(vec![("op", Value::text("matmul")), ("a", a), ("b", b)])
}

/// DoRA needs the base weight twice — once for direction, once for the norm it
/// is divided by — so its template is built against a specific tensor.
///
/// `mul(add(W, ΔW), div(m, norm(add(W, ΔW), axis)))`: direction from the
/// LoRA-updated weight, magnitude from the learned `m` (§08.2).
pub fn dora_template(scale: Scalar, axis: usize) -> Value {
    let update = Value::map(vec![
        ("op", Value::text("add")),
        ("a", Value::text("$W")),
        (
            "b",
            Value::map(vec![
                ("op", Value::text("scale")),
                ("x", matmul_of(Value::text("$B"), Value::text("$A"))),
                ("k", scale.to_value()),
            ]),
        ),
    ]);
    Value::map(vec![
        ("op", Value::text("mul")),
        (
            "with",
            Value::map(vec![
                ("op", Value::text("div")),
                ("a", Value::text("$m")),
                (
                    "b",
                    Value::map(vec![
                        ("op", Value::text("norm")),
                        ("x", update),
                        ("axis", Value::U(axis as u64)),
                        ("p", Value::F64(2.0)),
                        ("sum", Value::text("pairwise")),
                    ]),
                ),
            ]),
        ),
    ])
}

// ---------------------------------------------------------------- composition --

/// Composition modes of §08.5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Apply transforms in order; each sees the previous result.
    Sequential,
    /// `W + Σ wᵢ·Δᵢ` — the usual multi-LoRA case.
    ParallelSum,
    /// TIES-merging: trim, elect sign, disjoint mean.
    Ties,
    /// DARE: random drop and rescale, from a declared seed.
    Dare,
    /// Spherical interpolation between two deltas.
    Slerp,
    Plugin,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Sequential => "sequential",
            Mode::ParallelSum => "parallel-sum",
            Mode::Ties => "ties",
            Mode::Dare => "dare",
            Mode::Slerp => "slerp",
            Mode::Plugin => "plugin",
        }
    }
    pub fn parse(s: &str) -> Option<Mode> {
        Some(match s {
            "sequential" => Mode::Sequential,
            "parallel-sum" => Mode::ParallelSum,
            "ties" => Mode::Ties,
            "dare" => Mode::Dare,
            "slerp" => Mode::Slerp,
            "plugin" => Mode::Plugin,
            _ => return None,
        })
    }
    /// True when the mode is pure arithmetic over the parents and therefore an
    /// expression; false when it needs a data-dependent decision and is replayed
    /// by a tool instead (see the module docs).
    pub fn is_expressible(self) -> bool {
        matches!(self, Mode::Sequential | Mode::ParallelSum)
    }
}

/// How conflicting transforms on one tensor are resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conflicts {
    Error,
    LastWins,
    Sum,
}

/// A composition recipe (§08.5).
#[derive(Clone, Debug)]
pub struct Compose {
    pub order: Vec<String>,
    pub mode: Mode,
    pub weights: Vec<f64>,
    pub conflicts: Conflicts,
    /// Required for `dare`, and for any mode with a random component: without
    /// it the merge is not reproducible.
    pub seed: Option<u64>,
    /// TIES trim fraction: the proportion of entries kept, by magnitude.
    pub density: Option<f64>,
    /// SLERP interpolation parameter.
    pub t: Option<f64>,
}

impl Compose {
    pub fn from_value(v: &Value) -> Res<Compose> {
        let mode = match v.get("mode").and_then(|x| x.as_str()) {
            Some(m) => Mode::parse(m)
                .ok_or_else(|| Error::Type(format!("unknown composition mode `{m}`")))?,
            None => Mode::Sequential,
        };
        let f = |k: &str| match v.get(k) {
            Some(Value::F64(x)) => Some(*x),
            Some(Value::U(n)) => Some(*n as f64),
            _ => None,
        };
        let c = Compose {
            order: v
                .get("order")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .map(|x| x.as_str().unwrap_or_default().to_string())
                        .collect()
                })
                .unwrap_or_default(),
            mode,
            weights: v
                .get("weights")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .map(|x| match x {
                            Value::F64(f) => *f,
                            Value::U(n) => *n as f64,
                            Value::I(n) => *n as f64,
                            _ => 1.0,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            conflicts: match v.get("conflicts").and_then(|x| x.as_str()) {
                Some("last-wins") => Conflicts::LastWins,
                Some("sum") => Conflicts::Sum,
                Some("error") | None => Conflicts::Error,
                Some(c) => return Err(Error::Type(format!("unknown conflict policy `{c}`"))),
            },
            seed: v.get("seed").and_then(|x| x.as_u64()),
            density: f("density"),
            t: f("t"),
        };
        if c.mode == Mode::Dare && c.seed.is_none() {
            return Err(Error::Type(
                "`dare` drops entries at random, so it requires a `seed`; without one the merge \
                 is not reproducible from its parents (§08.5)"
                    .into(),
            ));
        }
        if !c.weights.is_empty() && !c.order.is_empty() && c.weights.len() != c.order.len() {
            return Err(Error::Type(format!(
                "{} weights for {} adapters",
                c.weights.len(),
                c.order.len()
            )));
        }
        Ok(c)
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            (
                "order",
                Value::Array(self.order.iter().map(|x| Value::text(x.clone())).collect()),
            ),
            ("mode", Value::text(self.mode.name())),
        ];
        if !self.weights.is_empty() {
            p.push((
                "weights",
                Value::Array(self.weights.iter().map(|x| Value::F64(*x)).collect()),
            ));
        }
        p.push((
            "conflicts",
            Value::text(match self.conflicts {
                Conflicts::Error => "error",
                Conflicts::LastWins => "last-wins",
                Conflicts::Sum => "sum",
            }),
        ));
        if let Some(s) = self.seed {
            p.push(("seed", Value::U(s)));
        }
        if let Some(d) = self.density {
            p.push(("density", Value::F64(d)));
        }
        if let Some(t) = self.t {
            p.push(("t", Value::F64(t)));
        }
        Value::map(p)
    }

    fn weight(&self, i: usize) -> f64 {
        self.weights.get(i).copied().unwrap_or(1.0)
    }
}

/// Composes several adapters' transforms on one tensor into one expression.
///
/// `patches` are the per-adapter *deltas* (the `with` side of each transform),
/// in composition order. Only the expressible modes are handled here; see
/// [`merge_values`] for the rest.
pub fn compose_expr(base: &Expr, patches: &[Expr], c: &Compose) -> Res<Expr> {
    if patches.is_empty() {
        return Ok(base.clone());
    }
    match c.mode {
        Mode::Sequential => {
            let mut acc = base.clone();
            for (i, p) in patches.iter().enumerate() {
                acc = Expr::Bin {
                    op: BinOp::Add,
                    a: Box::new(acc),
                    b: Box::new(weighted(p, c.weight(i))),
                };
            }
            Ok(acc)
        }
        Mode::ParallelSum => {
            let mut sum = weighted(&patches[0], c.weight(0));
            for (i, p) in patches.iter().enumerate().skip(1) {
                sum = Expr::Bin {
                    op: BinOp::Add,
                    a: Box::new(sum),
                    b: Box::new(weighted(p, c.weight(i))),
                };
            }
            Ok(Expr::Bin {
                op: BinOp::Add,
                a: Box::new(base.clone()),
                b: Box::new(sum),
            })
        }
        other => Err(Error::Unsupported(format!(
            "`{}` is a merge algorithm, not an expression: it needs an elementwise comparison \
             (ties), a dot product and trigonometry (slerp), or a drawn mask (dare), none of \
             which are core nodes. Replay it with `merge_values`, which records the recipe.",
            other.name()
        ))),
    }
}

fn weighted(e: &Expr, w: f64) -> Expr {
    if w == 1.0 {
        e.clone()
    } else {
        Expr::Scale {
            x: Box::new(e.clone()),
            k: Scalar::Float(w),
        }
    }
}

/// The outcome of replaying a merge algorithm over materialized deltas.
#[derive(Clone, Debug, PartialEq)]
pub struct Merged {
    /// The merged delta, elementwise.
    pub delta: Vec<f64>,
    /// For SLERP, the two exact coefficients the merge reduces to, so the result
    /// can be re-expressed as `add(scale(a, c0), scale(b, c1))` and evaluated by
    /// any conforming evaluator.
    pub coefficients: Option<(f64, f64)>,
    /// How many entries survived, for the merge report.
    pub kept: u64,
}

/// Replays `ties`, `dare` or `slerp` over materialized deltas (§08.5).
///
/// Every decision is a function of the inputs and the declared parameters — the
/// trim threshold from `density`, the drop mask from `seed` — so a merged model
/// is reproducible from its parents, which is what §08.5 is for.
pub fn merge_values(deltas: &[Tensor], c: &Compose) -> Res<Merged> {
    if deltas.is_empty() {
        return Err(Error::Type("nothing to merge".into()));
    }
    let n = deltas[0].data.len();
    for d in deltas {
        if d.data.len() != n {
            return Err(Error::Type(
                "merged deltas must have the same element count".into(),
            ));
        }
    }
    match c.mode {
        Mode::Ties => {
            let density = c.density.unwrap_or(0.2);
            if !(0.0..=1.0).contains(&density) {
                return Err(Error::Type(format!(
                    "ties: density {density} is not a fraction"
                )));
            }
            // Trim: keep the largest-magnitude entries of each delta.
            let mut trimmed: Vec<Vec<f64>> = Vec::with_capacity(deltas.len());
            for d in deltas {
                let mut mags: Vec<f64> = d.data.iter().map(|x| x.abs()).collect();
                mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let keep = ((n as f64) * density).ceil() as usize;
                let threshold = if keep == 0 || keep > n {
                    f64::INFINITY
                } else {
                    mags[keep - 1]
                };
                trimmed.push(
                    d.data
                        .iter()
                        .map(|x| if x.abs() >= threshold { *x } else { 0.0 })
                        .collect(),
                );
            }
            // Elect sign by summed magnitude, then take the disjoint mean of the
            // entries that agree with it.
            let mut out = vec![0.0f64; n];
            let mut kept = 0u64;
            for i in 0..n {
                let mut pos = 0.0f64;
                let mut neg = 0.0f64;
                for (j, t) in trimmed.iter().enumerate() {
                    let v = t[i] * c.weight(j);
                    if v > 0.0 {
                        pos += v;
                    } else {
                        neg += -v;
                    }
                }
                let sign = if pos >= neg { 1.0 } else { -1.0 };
                let mut sum = 0.0f64;
                let mut count = 0u32;
                for (j, t) in trimmed.iter().enumerate() {
                    let v = t[i] * c.weight(j);
                    if v != 0.0 && v.signum() == sign {
                        sum += v;
                        count += 1;
                    }
                }
                if count > 0 {
                    out[i] = sum / count as f64;
                    kept += 1;
                }
            }
            Ok(Merged {
                delta: out,
                coefficients: None,
                kept,
            })
        }
        Mode::Dare => {
            let p = 1.0 - c.density.unwrap_or(0.1);
            let seed = c.seed.expect("Compose::from_value requires a dare seed");
            let mut out = vec![0.0f64; n];
            let mut kept = 0u64;
            for (i, slot) in out.iter_mut().enumerate() {
                // The mask is drawn from the declared seed, so it is
                // reproducible and needs no storage.
                if crate::expr::uniform01(seed, i as u64) >= p {
                    let mut sum = 0.0;
                    for (j, d) in deltas.iter().enumerate() {
                        sum += d.data[i] * c.weight(j);
                    }
                    // Rescale by the survival probability, which is what makes
                    // DARE unbiased.
                    *slot = sum / (1.0 - p);
                    kept += 1;
                }
            }
            Ok(Merged {
                delta: out,
                coefficients: None,
                kept,
            })
        }
        Mode::Slerp => {
            if deltas.len() != 2 {
                return Err(Error::Type(
                    "slerp interpolates between exactly two deltas".into(),
                ));
            }
            let t = c.t.unwrap_or(0.5);
            let (a, b) = (&deltas[0].data, &deltas[1].data);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            let (c0, c1) = if na == 0.0 || nb == 0.0 {
                // Degenerate: fall back to linear interpolation rather than
                // dividing by zero.
                (1.0 - t, t)
            } else {
                let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
                let omega = cos.acos();
                if omega.abs() < 1e-9 {
                    (1.0 - t, t)
                } else {
                    let s = omega.sin();
                    (((1.0 - t) * omega).sin() / s, (t * omega).sin() / s)
                }
            };
            let delta: Vec<f64> = a.iter().zip(b).map(|(x, y)| c0 * x + c1 * y).collect();
            let kept = delta.iter().filter(|x| **x != 0.0).count() as u64;
            Ok(Merged {
                delta,
                coefficients: Some((c0, c1)),
                kept,
            })
        }
        other => Err(Error::Type(format!(
            "`{}` is an expression, not a replayed merge; use compose_expr",
            other.name()
        ))),
    }
}

/// The expression a SLERP merge reduces to, once its two coefficients are known:
/// exact, evaluable by any conforming reader, and no bytes stored.
pub fn slerp_expr(a: &Expr, b: &Expr, coefficients: (f64, f64)) -> Expr {
    Expr::Bin {
        op: BinOp::Add,
        a: Box::new(Expr::Scale {
            x: Box::new(a.clone()),
            k: Scalar::Float(coefficients.0),
        }),
        b: Box::new(Expr::Scale {
            x: Box::new(b.clone()),
            k: Scalar::Float(coefficients.1),
        }),
    }
}

/// Builds the LoRA delta expression `scale(matmul(B, A), k)` directly, for
/// tooling that has the factors in hand.
pub fn lora_delta(b: Expr, a: Expr, k: Scalar) -> Expr {
    Expr::Scale {
        x: Box::new(Expr::MatMul {
            a: Box::new(b),
            b: Box::new(a),
            sum: Sum::Pairwise,
        }),
        k,
    }
}

/// A minimal `Adapter` value for a LoRA over a glob of target tensors — what a
/// converter emits.
#[allow(clippy::too_many_arguments)]
pub fn lora_adapter_value(
    base: &Ref,
    tensors: &Ref,
    rank: u64,
    alpha: f64,
    targets: &[&str],
    bind_a: &str,
    bind_b: &str,
    // `rank_axis` is the base's own name for the axis the rank contracts over.
    // Naming it rather than assuming one is what makes `require` catch a
    // mismatch instead of the math being quietly wrong (§08.3).
    rank_axis: &str,
) -> Res<Value> {
    let scale = if alpha.fract() == 0.0 && rank > 0 {
        Scalar::Ratio(alpha as i64, rank as i64)
    } else {
        Scalar::Float(alpha / rank.max(1) as f64)
    };
    let template = method_template(&Method::Lora, scale)?;
    let attach: Vec<Value> = targets
        .iter()
        .map(|t| {
            Value::map(vec![
                (
                    "select",
                    Value::map(vec![("glob", Value::text((*t).to_string()))]),
                ),
                ("kind", Value::text("tensor-transform")),
                ("apply", template.clone()),
                (
                    "bind",
                    Value::Map(vec![
                        (Value::text("$A"), Value::text(bind_a.to_string())),
                        (Value::text("$B"), Value::text(bind_b.to_string())),
                    ]),
                ),
                (
                    "require",
                    Value::map(vec![("rank_axis", Value::text(rank_axis.to_string()))]),
                ),
            ])
        })
        .collect();
    Ok(Value::map(vec![
        ("t", Value::text("omni.adapt/adapter")),
        ("v", Value::U(1)),
        ("method", Value::text("lora")),
        ("base", ref_value(base)),
        ("rank", Value::U(rank)),
        ("alpha", Value::F64(alpha)),
        (
            "targets",
            Value::Array(
                targets
                    .iter()
                    .map(|t| Value::text((*t).to_string()))
                    .collect(),
            ),
        ),
        ("tensors", ref_value(tensors)),
        ("attach", Value::Array(attach)),
        ("merge_policy", Value::text("runtime")),
    ]))
}

/// Shape helper for callers.
pub fn dims(shape: &[u64]) -> Vec<Dim> {
    crate::expr::dims(shape)
}

/// Concrete shape helper.
pub fn sizes(shape: &[Dim]) -> Option<Vec<u64>> {
    concrete(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{otype, HashAlgo, Object};
    use crate::dtype::Round;
    use crate::layout::Layout;
    use crate::store::{MemoryStore, WritableStore};
    use crate::tensor::Materialize;

    struct Fixture {
        store: MemoryStore,
        base: TensorTable,
        adapter_tensors: Ref,
        base_manifest: Digest,
    }

    fn put_tensor(
        s: &mut MemoryStore,
        shape: &[u64],
        data: &[f64],
        axes: Option<Vec<&str>>,
        role: Option<&str>,
    ) -> Ref {
        let t = Tensor::new(shape.to_vec(), DType::F32, data.to_vec());
        let bytes = t
            .to_bytes(&DType::F32, &Layout::default(), Round::Rne)
            .unwrap();
        let blob = s.put(&bytes).unwrap();
        let desc = TensorDesc {
            shape: dims(shape),
            dtype: DType::F32,
            layout: Layout::default(),
            value: Expr::Literal {
                chunks: (otype::BLOB, blob),
                dtype: DType::F32,
                shape: dims(shape),
                layout: Layout::default(),
            },
            semantic: Some("weight".into()),
            role: role.map(|r| r.to_string()),
            axes: axes.map(|a| a.iter().map(|x| x.to_string()).collect()),
            device_hint: None,
            materialize: Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        };
        let d = s
            .put(&Object::structure(otype::TENSOR_DESC, &desc.to_value()).payload)
            .unwrap();
        (otype::TENSOR_DESC, d)
    }

    fn table(s: &mut MemoryStore, entries: &[(&str, Ref)]) -> Ref {
        let v = Value::map(vec![
            ("t", Value::text("omni.tensor/table")),
            ("v", Value::U(1)),
            (
                "tensors",
                Value::Map(
                    entries
                        .iter()
                        .map(|(n, r)| (Value::text((*n).to_string()), ref_value(r)))
                        .collect(),
                ),
            ),
        ]);
        let d = s
            .put(&Object::structure(otype::TENSOR_TABLE, &v).payload)
            .unwrap();
        (otype::TENSOR_TABLE, d)
    }

    /// A 2x2 base weight per layer, and a rank-1 LoRA for each.
    fn fixture() -> Fixture {
        let mut s = MemoryStore::new(HashAlgo::default());
        let w0 = put_tensor(
            &mut s,
            &[2, 2],
            &[1.0, 0.0, 0.0, 1.0],
            Some(vec!["out", "in"]),
            Some("attn.q_proj"),
        );
        let w1 = put_tensor(
            &mut s,
            &[2, 2],
            &[2.0, 0.0, 0.0, 2.0],
            Some(vec!["out", "in"]),
            Some("attn.q_proj"),
        );
        let base_ref = table(
            &mut s,
            &[
                ("model.layers.0.attn.q_proj.weight", w0),
                ("model.layers.1.attn.q_proj.weight", w1),
            ],
        );
        let base = TensorTable::load(&Ctx::new(&s), &base_ref).unwrap();
        // The adapter's own factors: B is 2x1, A is 1x2.
        let b0 = put_tensor(&mut s, &[2, 1], &[1.0, 2.0], None, None);
        let a0 = put_tensor(&mut s, &[1, 2], &[3.0, 4.0], None, None);
        let b1 = put_tensor(&mut s, &[2, 1], &[0.5, 0.5], None, None);
        let a1 = put_tensor(&mut s, &[1, 2], &[1.0, 1.0], None, None);
        let adapter_tensors = table(
            &mut s,
            &[
                ("lora.0.q_proj.B", b0),
                ("lora.0.q_proj.A", a0),
                ("lora.1.q_proj.B", b1),
                ("lora.1.q_proj.A", a1),
            ],
        );
        let base_manifest = s.put(b"a base manifest stands in here").unwrap();
        Fixture {
            store: s,
            base,
            adapter_tensors,
            base_manifest,
        }
    }

    fn lora_adapter(f: &Fixture, alpha: f64, rank: u64) -> Adapter {
        let v = lora_adapter_value(
            &(otype::MANIFEST, f.base_manifest),
            &f.adapter_tensors,
            rank,
            alpha,
            &["model.layers.*.attn.q_proj.weight"],
            "lora.{1}.q_proj.A",
            "lora.{1}.q_proj.B",
            "in",
        )
        .unwrap();
        Adapter::from_value(&v).unwrap()
    }

    #[test]
    fn a_lora_attaches_to_a_base_it_has_never_seen() {
        let f = fixture();
        let a = lora_adapter(&f, 30.0, 16);
        let ctx = Ctx::new(&f.store);
        let r = a.attach(&ctx, &f.base).unwrap();
        assert!(r.is_ok(), "{:?}", r.findings);
        assert_eq!(r.bindings.len(), 2);
        assert!(r.unmatched.is_empty());

        // Layer 0: W + (B@A) * 30/16 = I + [[3,4],[6,8]] * 1.875.
        let b0 = r
            .bindings
            .iter()
            .find(|b| b.tensor.contains("layers.0"))
            .unwrap();
        let t = b0.expr.eval(&ctx).unwrap();
        assert_eq!(
            t.data,
            vec![
                1.0 + 3.0 * 1.875,
                4.0 * 1.875,
                6.0 * 1.875,
                1.0 + 8.0 * 1.875
            ]
        );
        // The capture bound layer 0's factors, not layer 1's.
        assert_eq!(
            b0.used,
            vec!["lora.0.q_proj.A".to_string(), "lora.0.q_proj.B".to_string()]
        );
        // Layer 1 gets its own.
        let b1 = r
            .bindings
            .iter()
            .find(|b| b.tensor.contains("layers.1"))
            .unwrap();
        let t = b1.expr.eval(&ctx).unwrap();
        assert_eq!(t.data[0], 2.0 + 0.5 * 1.875);
        // The merged expression keeps the base tensor's type, so nothing about
        // the model's shape changes when an adapter is applied.
        assert_eq!(b1.expr.infer().unwrap().shape, dims(&[2, 2]));
    }

    #[test]
    fn alpha_over_rank_stays_an_exact_rational() {
        let f = fixture();
        let a = lora_adapter(&f, 32.0, 16);
        assert_eq!(a.lora_scale(), Scalar::Ratio(32, 16));
        // Two spellings of the same ratio give the same expression identity,
        // which is what makes a merged model dedup against its parents.
        let x = Expr::Full {
            value: Scalar::Int(1),
            dtype: DType::F32,
            shape: dims(&[1]),
        };
        let two = Expr::Scale {
            x: Box::new(x.clone()),
            k: Scalar::Ratio(32, 16),
        };
        let also_two = Expr::Scale {
            x: Box::new(x),
            k: Scalar::Int(2),
        };
        assert_eq!(
            two.identity(HashAlgo::default()),
            also_two.identity(HashAlgo::default())
        );
    }

    #[test]
    fn r_a02_reports_a_selector_that_matches_nothing() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.attach[0].select = Select::Glob("model.layers.*.mlp.gate.weight".into());
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(!r.is_ok());
        assert_eq!(r.unmatched.len(), 1);
        assert!(r.findings.iter().any(|x| x.rule == "R-A02"));
        // Unless the adapter says it expects that.
        a.allow_unmatched = true;
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.is_ok());
        assert!(r.bindings.is_empty());
    }

    #[test]
    fn r_a03_reports_a_binding_the_adapter_cannot_satisfy() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.attach[0].bind = vec![
            ("$A".into(), "lora.{1}.k_proj.A".into()),
            ("$B".into(), "lora.{1}.k_proj.B".into()),
        ];
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(!r.is_ok());
        assert!(r
            .findings
            .iter()
            .any(|x| x.rule == "R-A03" && x.message.contains("does not have")));
        assert!(r.bindings.is_empty());
    }

    #[test]
    fn r_a03_reports_a_shape_that_cannot_work() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        // Bind B to a 1x2 factor, which cannot multiply a 1x2 A.
        a.attach[0].bind = vec![
            ("$A".into(), "lora.{1}.q_proj.A".into()),
            ("$B".into(), "lora.{1}.q_proj.A".into()),
        ];
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(!r.is_ok());
        assert!(r.findings.iter().any(|x| x.rule == "R-A03"));
    }

    #[test]
    fn require_checks_the_assumptions_before_any_weights_load() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.attach[0].require = Require {
            axes: Some(vec!["out".into(), "in".into()]),
            rank_axis: Some("in".into()),
            shape: Some(vec![2, 2]),
            dtype: Some(DType::F32),
        };
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.is_ok(), "{:?}", r.findings);
        // A wrong axis name is caught with a clear message rather than
        // silently wrong math.
        a.attach[0].require.axes = Some(vec!["in".into(), "out".into()]);
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r
            .findings
            .iter()
            .any(|x| x.message.contains("requires axes")));
        // A rank axis the base does not have.
        a.attach[0].require = Require {
            rank_axis: Some("rank".into()),
            ..Default::default()
        };
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.findings.iter().any(|x| x.message.contains("rank axis")));
    }

    #[test]
    fn selecting_by_role_survives_renaming() {
        // §08.3: selecting by role and axes is the robust option because it
        // survives renaming between model releases.
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.attach[0].select = Select::Role("attn.q_proj".into());
        // A role selector produces no captures, so a rule that indexes {1}
        // must fail loudly.
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(!r.is_ok());
        assert!(r
            .findings
            .iter()
            .any(|x| x.message.contains("capture") || x.message.contains("does not exist")));
        // With a fixed binding it attaches to every tensor of that role.
        a.attach[0].bind = vec![
            ("$A".into(), "lora.0.q_proj.A".into()),
            ("$B".into(), "lora.0.q_proj.B".into()),
        ];
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.is_ok(), "{:?}", r.findings);
        assert_eq!(r.bindings.len(), 2);
    }

    #[test]
    fn a_regex_selector_captures_too() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.attach[0].select =
            Select::Regex(Regex::parse(r"^model\.layers\.(\d+)\.attn\.q_proj\.weight$").unwrap());
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.is_ok(), "{:?}", r.findings);
        assert_eq!(r.bindings.len(), 2);
    }

    #[test]
    fn semantic_and_axes_selectors_match() {
        let f = fixture();
        let ctx = Ctx::new(&f.store);
        let d = TensorDesc::load(
            &ctx,
            f.base.get("model.layers.0.attn.q_proj.weight").unwrap(),
        )
        .unwrap();
        assert!(Select::Semantic("weight".into())
            .matches("x", &d)
            .unwrap()
            .is_some());
        assert!(Select::Semantic("bias".into())
            .matches("x", &d)
            .unwrap()
            .is_none());
        assert!(Select::Axes(vec!["out".into(), "in".into()])
            .matches("x", &d)
            .unwrap()
            .is_some());
        assert!(Select::Axes(vec!["in".into()])
            .matches("x", &d)
            .unwrap()
            .is_none());
    }

    #[test]
    fn r_a01_an_absent_base_is_incomplete_not_invalid() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.base = (otype::MANIFEST, [0xab; 32]);
        let r = a.attach(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(
            r.is_ok(),
            "an unresolvable base must not be reported as invalid"
        );
        assert!(r
            .findings
            .iter()
            .any(|x| x.rule == "R-A01" && x.severity == Severity::Indeterminate));
    }

    #[test]
    fn every_arithmetic_method_of_section_08_2_is_core_nodes() {
        // The claim: the first eight methods need no format extension at all.
        for m in [
            Method::Lora,
            Method::AdaLora,
            Method::Vera,
            Method::Loha,
            Method::Ia3,
            Method::BitFit,
            Method::ControlVector,
        ] {
            let t = method_template(&m, Scalar::Ratio(32, 16)).unwrap();
            // The template's `with` side parses as an expression once its
            // placeholders are bound.
            let with = t.get("with").unwrap();
            let mut binds = BTreeMap::new();
            let leaf = Value::map(vec![
                ("op", Value::text("zeros")),
                ("dtype", DType::F32.to_value()),
                ("shape", Value::Array(vec![Value::U(2), Value::U(2)])),
            ]);
            for name in [
                "$A", "$B", "$A1", "$A2", "$B1", "$B2", "$l", "$db", "$v", "$d", "$b", "$Lambda",
                "$m",
            ] {
                binds.insert(name.to_string(), leaf.clone());
            }
            let sub = substitute(with, &binds).unwrap();
            let e = Expr::from_value(&sub).unwrap_or_else(|e| panic!("{}: {e}", m.name()));
            e.infer().unwrap_or_else(|e| panic!("{}: {e}", m.name()));
        }
        // DoRA needs the target's shape, and says so instead of emitting
        // something that does not type-check.
        assert!(method_template(&Method::Dora, Scalar::Int(1)).is_err());
        let t = dora_template(Scalar::Ratio(32, 16), 0);
        assert!(t.get("with").is_some());
        // Graph-level methods are not tensor transforms and refuse to pretend.
        assert!(method_template(&Method::Prefix, Scalar::Int(1)).is_err());
        assert!(Method::Prefix.is_graph_level());
        assert!(!Method::Lora.is_graph_level());
    }

    #[test]
    fn a_graph_level_adapter_must_ship_its_rewrites() {
        let f = fixture();
        let mut a = lora_adapter(&f, 30.0, 16);
        a.method = Method::Prefix;
        a.attach.clear();
        let r = a.check(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r
            .findings
            .iter()
            .any(|x| x.message.contains("ship the rewrites")));
        // With patches, it is accepted and they are handed on for §07 to apply.
        a.graph_patches = vec![Value::map(vec![
            ("t", Value::text("omni.ir/rewrite")),
            ("v", Value::U(1)),
            ("name", Value::text("prefix-kv")),
        ])];
        let r = a.check(&Ctx::new(&f.store), &f.base).unwrap();
        assert!(r.is_ok(), "{:?}", r.findings);
        assert_eq!(r.graph_patches.len(), 1);
    }

    #[test]
    fn multi_lora_composes_in_the_declared_order() {
        let f = fixture();
        let ctx = Ctx::new(&f.store);
        let base = TensorDesc::load(
            &ctx,
            f.base.get("model.layers.0.attn.q_proj.weight").unwrap(),
        )
        .unwrap()
        .value;
        let d1 = Expr::Full {
            value: Scalar::Float(1.0),
            dtype: DType::F32,
            shape: dims(&[2, 2]),
        };
        let d2 = Expr::Full {
            value: Scalar::Float(2.0),
            dtype: DType::F32,
            shape: dims(&[2, 2]),
        };
        let c = Compose::from_value(&Value::map(vec![
            (
                "order",
                Value::Array(vec![Value::text("safety"), Value::text("style")]),
            ),
            ("mode", Value::text("parallel-sum")),
            (
                "weights",
                Value::Array(vec![Value::F64(1.0), Value::F64(0.5)]),
            ),
        ]))
        .unwrap();
        let e = compose_expr(&base, &[d1.clone(), d2.clone()], &c).unwrap();
        let t = e.eval(&ctx).unwrap();
        // I + 1*1 + 0.5*2 = I + 2 elementwise.
        assert_eq!(t.data, vec![3.0, 2.0, 2.0, 3.0]);

        // Sequential gives the same total here but a different tree, and
        // therefore a different identity — the order is explicit because it
        // matters (§08.5).
        let seq = Compose {
            mode: Mode::Sequential,
            ..c.clone()
        };
        let e2 = compose_expr(&base, &[d1, d2], &seq).unwrap();
        assert_eq!(e2.eval(&ctx).unwrap().data, t.data);
        assert_ne!(
            e.identity(HashAlgo::default()),
            e2.identity(HashAlgo::default())
        );
    }

    #[test]
    fn dare_requires_a_seed_because_otherwise_it_is_not_reproducible() {
        let bad = Value::map(vec![("mode", Value::text("dare"))]);
        assert!(Compose::from_value(&bad).is_err());
        let good = Value::map(vec![
            ("mode", Value::text("dare")),
            ("seed", Value::U(7)),
            ("density", Value::F64(0.5)),
        ]);
        let c = Compose::from_value(&good).unwrap();
        let a = Tensor::new(vec![64], DType::F32, vec![1.0; 64]);
        let m1 = merge_values(std::slice::from_ref(&a), &c).unwrap();
        let m2 = merge_values(&[a], &c).unwrap();
        // Reproducible from the seed, and unbiased: survivors are rescaled.
        assert_eq!(m1, m2);
        assert!(m1.kept > 0 && m1.kept < 64);
        assert!(m1.delta.iter().all(|x| *x == 0.0 || *x == 2.0));
    }

    #[test]
    fn ties_trims_elects_a_sign_and_takes_a_disjoint_mean() {
        let c = Compose::from_value(&Value::map(vec![
            ("mode", Value::text("ties")),
            ("density", Value::F64(0.5)),
        ]))
        .unwrap();
        // Two deltas over four entries. Trimming to 50% keeps the two largest
        // magnitudes of each.
        let a = Tensor::new(vec![4], DType::F32, vec![1.0, -2.0, 0.1, 0.0]);
        let b = Tensor::new(vec![4], DType::F32, vec![3.0, 1.0, -0.2, 0.0]);
        let m = merge_values(&[a, b], &c).unwrap();
        // Entry 0: both keep (1.0 and 3.0), same sign, mean 2.0.
        assert_eq!(m.delta[0], 2.0);
        // Entry 1: a keeps -2.0, b keeps 1.0; the elected sign is negative
        // (magnitude 2 vs 1), so only a contributes.
        assert_eq!(m.delta[1], -2.0);
        // Entry 3 was zero in both and stays zero.
        assert_eq!(m.delta[3], 0.0);
        assert_eq!(m.kept, 2);
    }

    #[test]
    fn slerp_reduces_to_two_exact_scale_coefficients() {
        let c = Compose::from_value(&Value::map(vec![
            ("mode", Value::text("slerp")),
            ("t", Value::F64(0.5)),
        ]))
        .unwrap();
        // Two orthogonal unit deltas: at t = 0.5 the interpolation is
        // symmetric, so the coefficients are equal.
        let a = Tensor::new(vec![2], DType::F32, vec![1.0, 0.0]);
        let b = Tensor::new(vec![2], DType::F32, vec![0.0, 1.0]);
        let m = merge_values(&[a, b], &c).unwrap();
        let (c0, c1) = m.coefficients.unwrap();
        assert!((c0 - c1).abs() < 1e-12);
        // sin(45°)/sin(90°) = 0.7071…
        assert!((c0 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((m.delta[0] - c0).abs() < 1e-12);

        // The merge re-expresses as an ordinary expression, so any conforming
        // evaluator reproduces it with no stored bytes.
        let s = MemoryStore::new(HashAlgo::default());
        let ea = Expr::Full {
            value: Scalar::Float(1.0),
            dtype: DType::F32,
            shape: dims(&[2]),
        };
        let eb = Expr::Full {
            value: Scalar::Float(0.0),
            dtype: DType::F32,
            shape: dims(&[2]),
        };
        let e = slerp_expr(&ea, &eb, (c0, c1));
        assert_eq!(e.eval(&Ctx::new(&s)).unwrap().data, vec![c0, c0]);
        // Three parents is not slerp, and it says so.
        let t3 = Tensor::new(vec![2], DType::F32, vec![1.0, 1.0]);
        assert!(merge_values(&[t3.clone(), t3.clone(), t3], &c).is_err());
    }

    #[test]
    fn the_merge_algorithms_are_not_expressions_and_say_so() {
        let base = Expr::Full {
            value: Scalar::Int(0),
            dtype: DType::F32,
            shape: dims(&[2]),
        };
        for mode in ["ties", "slerp"] {
            let c = Compose::from_value(&Value::map(vec![("mode", Value::text(mode))])).unwrap();
            let e = compose_expr(&base, std::slice::from_ref(&base), &c);
            assert!(matches!(e, Err(Error::Unsupported(_))), "{mode}");
            assert!(!c.mode.is_expressible());
        }
        assert!(Mode::ParallelSum.is_expressible());
    }

    #[test]
    fn adapters_round_trip_through_cbor() {
        let f = fixture();
        let a = lora_adapter(&f, 30.0, 16);
        let v = a.to_value();
        let again = Adapter::from_value(&v).unwrap();
        assert_eq!(again.method, a.method);
        assert_eq!(again.base, a.base);
        assert_eq!(again.rank, a.rank);
        assert_eq!(again.attach.len(), a.attach.len());
        let round = crate::cbor::decode(&v.encode()).unwrap();
        let third = Adapter::from_value(&round).unwrap();
        assert_eq!(third.to_value().encode(), again.to_value().encode());
        // An adapter with no base is refused: "which model is this adapter for?"
        // must be answerable (§08.1).
        let mut bad = v.as_map().unwrap().to_vec();
        bad.retain(|(k, _)| k.as_str() != Some("base"));
        assert!(Adapter::from_value(&Value::Map(bad)).is_err());
    }

    #[test]
    fn a_composition_with_mismatched_weights_is_refused() {
        let v = Value::map(vec![
            (
                "order",
                Value::Array(vec![Value::text("a"), Value::text("b")]),
            ),
            ("mode", Value::text("parallel-sum")),
            ("weights", Value::Array(vec![Value::F64(1.0)])),
        ]);
        assert!(Compose::from_value(&v).is_err());
        let v = Value::map(vec![("mode", Value::text("telepathy"))]);
        assert!(Compose::from_value(&v).is_err());
    }
}
