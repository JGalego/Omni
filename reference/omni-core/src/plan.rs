//! §10 — the runtime interface: capability sets, plans, and the resolver.
//!
//! §10.1 states the whole point as a function:
//!
//! ```text
//! resolve : (ModelDAG, CapabilitySet, Objective) → Plan | Failure(reasons)
//! ```
//!
//! It is pure, deterministic and cheap: it reads metadata and tensor
//! descriptors and never a tensor byte. That is what makes a plan verifiable —
//! a third party re-runs the resolver and gets the same answer — and what lets a
//! CI job answer "will this model run on our fleet?" without the fleet.
//!
//! Two details from the section shape the code more than the rest:
//!
//! * `unsupported` is **not** the complement of the supported list. "Not
//!   listed" means unknown; "unsupported" means do not attempt. A resolver that
//!   collapses the two either refuses things that would have worked or attempts
//!   things a deployment has forbidden.
//! * Failure is informative (§10.5.2). `resolve` returns structured reasons with
//!   the tensors and bytes each one affects, because actionable diagnostics are
//!   a property of the format — the model declared what it needs in
//!   machine-readable form — rather than a nicety of a tool.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo};
use crate::dtype::DType;
use crate::expr::{concrete, Ctx, Error, Expr, Ref};
use crate::layout::Layout;
use crate::tensor::{TensorDesc, TensorTable};
use std::collections::{BTreeMap, BTreeSet};

type Res<T> = Result<T, Error>;

// ------------------------------------------------------------- capabilities --

/// What a runtime can do (§10.2).
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub runtime_name: String,
    pub runtime_version: String,
    pub profiles: Vec<String>,
    /// Dtypes the runtime can *store*: the ones a tensor may be represented in.
    pub storage_dtypes: BTreeSet<String>,
    /// Dtypes it can compute in.
    pub compute_dtypes: BTreeSet<String>,
    pub quant_schemes: BTreeSet<String>,
    pub layouts: BTreeSet<String>,
    pub sparsity: BTreeSet<String>,
    pub features: BTreeSet<String>,
    /// Explicitly refused: distinct from absent (§10.2).
    pub unsupported: BTreeSet<String>,
    pub memory_bytes: Option<u64>,
    pub allow_lossy: bool,
    pub allow_plugins: Vec<String>,
    pub require_signature: bool,
    pub max_materialize_bytes: Option<u64>,
}

impl Capabilities {
    /// A capability set that can hold a `literal`-only model and nothing else —
    /// the C0 reader of §00.6.
    pub fn c0() -> Capabilities {
        Capabilities {
            runtime_name: "c0".into(),
            runtime_version: "1".into(),
            profiles: vec!["C0".into()],
            storage_dtypes: ["f32", "f16", "bf16", "i8", "u8"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            compute_dtypes: ["f32"].iter().map(|s| s.to_string()).collect(),
            layouts: ["strided"].iter().map(|s| s.to_string()).collect(),
            features: ["omni.core/1.0"].iter().map(|s| s.to_string()).collect(),
            allow_lossy: false,
            ..Default::default()
        }
    }

    /// A capability set with the expression evaluator and the quantization
    /// schemes this crate implements — what `omni caps` emits for itself.
    pub fn reference() -> Capabilities {
        let s = |xs: &[&str]| -> BTreeSet<String> { xs.iter().map(|x| x.to_string()).collect() };
        Capabilities {
            runtime_name: "omni-rs".into(),
            runtime_version: env!("CARGO_PKG_VERSION").into(),
            profiles: crate::PROFILES.iter().map(|p| p.to_string()).collect(),
            storage_dtypes: s(&[
                "f64",
                "f32",
                "f16",
                "bf16",
                "tf32",
                "f8e4m3",
                "f8e5m2",
                "f6e3m2",
                "f6e2m3",
                "f4e2m1",
                "e8m0",
                "i8",
                "i16",
                "i32",
                "i64",
                "u8",
                "u16",
                "u32",
                "u64",
                "i4",
                "u4",
                "i2",
                "u2",
                "bool",
                "binary",
                "ternary-b3x5",
            ]),
            compute_dtypes: s(&["f64", "f32"]),
            quant_schemes: s(&["affine", "sym", "codebook", "nested"]),
            layouts: s(&[
                "strided",
                "tiled",
                "packed",
                "blocked-scaled",
                "interleaved",
            ]),
            sparsity: s(&[
                "coo",
                "csr",
                "csc",
                "bsr",
                "nm",
                "bitmask",
                "ragged",
                "blocklist",
            ]),
            features: s(&[
                "omni.core/1.0",
                "omni.tensor/expr.1",
                "omni.adapt/lora.1",
                "omni.stream/bao.1",
                "omni.codec/deflate.1",
            ]),
            unsupported: s(&["omni.plugin/wasm.1", "omni.codec/zstd.1"]),
            memory_bytes: None,
            allow_lossy: true,
            allow_plugins: vec![],
            require_signature: false,
            max_materialize_bytes: None,
        }
    }

    /// Whether the runtime can hold a tensor in this dtype.
    ///
    /// A dtype with no registered alias is `Unknown` rather than refused: the
    /// alias registry is data (§04.3.6), and a runtime that has not heard of a
    /// name has not said it cannot hold the type.
    fn has_dtype(&self, d: &DType) -> Support {
        match d.alias() {
            Some(a) if self.unsupported.contains(a) => Support::Refused,
            Some(a) if self.storage_dtypes.contains(a) => Support::Yes,
            _ => Support::Unknown,
        }
    }

    fn has_layout(&self, l: &Layout) -> Support {
        let k = l.kind();
        if self.unsupported.contains(k) {
            Support::Refused
        } else if self.layouts.contains(k) {
            Support::Yes
        } else {
            Support::Unknown
        }
    }

    fn has_scheme(&self, name: &str) -> Support {
        if self.unsupported.contains(name) {
            Support::Refused
        } else if self.quant_schemes.contains(name) {
            Support::Yes
        } else {
            Support::Unknown
        }
    }

    /// Feature support, including the implication of §10.5 step 1: a runtime
    /// that lists a profile implies the features that profile requires.
    pub fn has_feature(&self, f: &str) -> Support {
        if self.unsupported.contains(f) {
            return Support::Refused;
        }
        if self.features.contains(f) {
            return Support::Yes;
        }
        // `implied(caps)`: C1 and above imply the expression evaluator, since
        // that is what the profile means.
        if f == "omni.tensor/expr.1"
            && self
                .profiles
                .iter()
                .any(|p| matches!(p.as_str(), "C1" | "C2" | "C3" | "C4"))
        {
            return Support::Yes;
        }
        Support::Unknown
    }

    pub fn to_value(&self) -> Value {
        let arr = |xs: &BTreeSet<String>| {
            Value::Array(xs.iter().map(|x| Value::text(x.clone())).collect())
        };
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.rt/capabilities")),
            ("v", Value::U(1)),
            (
                "runtime",
                Value::map(vec![
                    ("name", Value::text(self.runtime_name.clone())),
                    ("version", Value::text(self.runtime_version.clone())),
                ]),
            ),
            (
                "profiles",
                Value::Array(
                    self.profiles
                        .iter()
                        .map(|x| Value::text(x.clone()))
                        .collect(),
                ),
            ),
            (
                "dtypes",
                Value::map(vec![
                    ("storage", arr(&self.storage_dtypes)),
                    ("compute", arr(&self.compute_dtypes)),
                ]),
            ),
            ("quant_schemes", arr(&self.quant_schemes)),
            ("layouts", arr(&self.layouts)),
            ("sparsity", arr(&self.sparsity)),
            ("features", arr(&self.features)),
            ("unsupported", arr(&self.unsupported)),
        ];
        let mut policy: Vec<(&str, Value)> = vec![
            ("allow_lossy", Value::Bool(self.allow_lossy)),
            ("require_signature", Value::Bool(self.require_signature)),
        ];
        if let Some(n) = self.max_materialize_bytes {
            policy.push(("max_materialize_bytes", Value::U(n)));
        }
        if !self.allow_plugins.is_empty() {
            policy.push((
                "allow_plugins",
                Value::Array(
                    self.allow_plugins
                        .iter()
                        .map(|x| Value::text(x.clone()))
                        .collect(),
                ),
            ));
        }
        p.push(("policy", Value::map(policy)));
        if let Some(m) = self.memory_bytes {
            p.push(("budget", Value::map(vec![("memory_bytes", Value::U(m))])));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Capabilities> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.rt/capabilities") {
            return Err(Error::Type(
                "R-O02: object is not an omni.rt/capabilities".into(),
            ));
        }
        let set = |key: Option<&Value>| -> BTreeSet<String> {
            key.and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default()
        };
        let policy = v.get("policy");
        let flag = |k: &str, default: bool| match policy.and_then(|p| p.get(k)) {
            Some(Value::Bool(b)) => *b,
            _ => default,
        };
        Ok(Capabilities {
            runtime_name: v
                .get("runtime")
                .and_then(|r| r.get("name"))
                .and_then(|x| x.as_str())
                .unwrap_or("(unnamed)")
                .to_string(),
            runtime_version: v
                .get("runtime")
                .and_then(|r| r.get("version"))
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            profiles: set(v.get("profiles")).into_iter().collect(),
            storage_dtypes: set(v.get("dtypes").and_then(|d| d.get("storage"))),
            compute_dtypes: set(v.get("dtypes").and_then(|d| d.get("compute"))),
            quant_schemes: set(v.get("quant_schemes")),
            layouts: set(v.get("layouts")),
            sparsity: set(v.get("sparsity")),
            features: set(v.get("features")),
            unsupported: set(v.get("unsupported")),
            memory_bytes: v
                .get("budget")
                .and_then(|b| b.get("memory_bytes"))
                .and_then(|x| x.as_u64()),
            allow_lossy: flag("allow_lossy", false),
            allow_plugins: set(policy.and_then(|p| p.get("allow_plugins")))
                .into_iter()
                .collect(),
            require_signature: flag("require_signature", false),
            max_materialize_bytes: policy
                .and_then(|p| p.get("max_materialize_bytes"))
                .and_then(|x| x.as_u64()),
        })
    }

    /// The digest a plan records, so a plan can be checked against the
    /// capabilities it was resolved for.
    pub fn digest(&self, algo: HashAlgo) -> Digest {
        algo.digest(&self.to_value().encode())
    }
}

/// Three-valued support, because §10.2 distinguishes absent from refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Yes,
    /// Not listed: the resolver may try it optimistically.
    Unknown,
    /// Listed as unsupported: do not attempt.
    Refused,
}

impl Support {
    fn usable(self, optimistic: bool) -> bool {
        match self {
            Support::Yes => true,
            Support::Unknown => optimistic,
            Support::Refused => false,
        }
    }
}

// --------------------------------------------------------------- objectives --

/// What the resolver optimizes for (§10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Objective {
    #[default]
    MinMemory,
    MaxQuality,
    MinLoadTime,
    MinLatency,
    Balanced,
}

impl Objective {
    pub fn name(self) -> &'static str {
        match self {
            Objective::MinMemory => "min-memory",
            Objective::MaxQuality => "max-quality",
            Objective::MinLoadTime => "min-load-time",
            Objective::MinLatency => "min-latency",
            Objective::Balanced => "balanced",
        }
    }
    pub fn parse(s: &str) -> Option<Objective> {
        Some(match s {
            "min-memory" => Objective::MinMemory,
            "max-quality" => Objective::MaxQuality,
            "min-load-time" => Objective::MinLoadTime,
            "min-latency" => Objective::MinLatency,
            "balanced" => Objective::Balanced,
            _ => return None,
        })
    }
}

// -------------------------------------------------------------- candidates --

/// How a chosen representation will be produced (§10.5 step 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Materialization {
    /// A bare literal in a compatible layout: the zero-copy path.
    DirectMap,
    /// Evaluated on load.
    OnLoad,
    /// Evaluated once and cached to disk.
    Cache,
}

impl Materialization {
    pub fn name(self) -> &'static str {
        match self {
            Materialization::DirectMap => "direct-map",
            Materialization::OnLoad => "materialize-on-load",
            Materialization::Cache => "cache-to-disk",
        }
    }
}

/// One valid standalone representation of a tensor (§10.5.1).
#[derive(Clone, Debug)]
pub struct Candidate {
    pub expr: Expr,
    pub dtype: DType,
    pub layout: Layout,
    /// Bytes resident once this representation is in memory.
    pub resident_bytes: u64,
    /// Bytes that must be read to produce it.
    pub read_bytes: u64,
    /// True when this representation loses information *relative to the best
    /// candidate available*.
    ///
    /// That is the only definition that means anything to a policy. A tensor
    /// published as int8 is not "lossy" in the abstract — int8 is what the
    /// publisher says it is — but if the container also carries the bf16 value
    /// the int8 was made from, then choosing int8 is a decision to lose
    /// something, and `allow_lossy: false` is a deployment saying it will not
    /// make that decision silently.
    pub lossy: bool,
    /// The quantization scheme a runtime must implement to consume this
    /// representation.
    ///
    /// The int4 literal under a `dequantize` node *is* a valid representation of
    /// the tensor — it is the one §04.8's int4 CPU runtime consumes, at a quarter
    /// of the memory — but only for a runtime that can do the dequantization
    /// itself. The requirement travels with the candidate rather than being read
    /// off its top node, which would miss it.
    pub requires_scheme: Option<String>,
    /// Quality relative to the tensor's declared value: `0.0` is the value as
    /// published, and *higher is better* — a candidate found by descending
    /// through a narrowing `cast` is the wider value the cast consumed.
    ///
    /// This is a *modeled* number, not a measured one, and it is only ever used
    /// to order candidates. §performance's rule about labelling claims applies:
    /// nothing here has been measured against an evaluation.
    pub quality_delta: f64,
    pub materialization: Materialization,
}

/// Walks an expression and yields every node that is a valid standalone
/// representation of the tensor (§10.5.1).
///
/// The reachable set is exactly what the section says: the root, plus anything
/// reachable through `cast`, `quantize`, `dequantize`, `select` and `approx`
/// chains. It is a short walk with no special cases, which is the payoff for
/// keeping the algebra small.
pub fn enumerate(root: &Expr, caps: &Capabilities) -> Vec<Candidate> {
    let mut out = Vec::new();
    walk(root, caps, 0.0, false, None, &mut out);
    // Lossiness is relative to the best candidate, so it can only be decided
    // once they are all known.
    let best = out
        .iter()
        .map(|c| c.quality_delta)
        .fold(f64::NEG_INFINITY, f64::max);
    for c in &mut out {
        c.lossy = c.lossy || c.quality_delta < best;
    }
    out
}

fn walk(
    e: &Expr,
    caps: &Capabilities,
    quality: f64,
    lossy: bool,
    scheme: Option<String>,
    out: &mut Vec<Candidate>,
) {
    let Ok(t) = e.infer() else { return };
    let Some(sizes) = concrete(&t.shape) else {
        return;
    };
    let layout = match e {
        Expr::Literal { layout, .. } => layout.clone(),
        Expr::Relayout { layout, .. } => layout.clone(),
        _ => Layout::default(),
    };
    let resident = layout
        .stored_bytes(&sizes, &t.dtype)
        .unwrap_or_else(|| t.dtype.packed_bytes(crate::layout::numel(&sizes)));
    let read: u64 = e.deps_all().iter().map(|d| d.bytes.1 - d.bytes.0).sum();
    // `approx` declares that its subtree *is* an approximation, so everything
    // at or below it is lossy however the rest of the tree compares.
    let lossy = lossy || matches!(e, Expr::Approx { .. });
    out.push(Candidate {
        expr: e.clone(),
        dtype: t.dtype.clone(),
        layout: layout.clone(),
        resident_bytes: resident,
        read_bytes: read,
        lossy,
        requires_scheme: scheme.clone().or_else(|| scheme_name(e)),
        quality_delta: quality,
        materialization: match e {
            // §10.5 step 6: direct-map is chosen when the expression is a bare
            // literal whose layout the runtime can use as it stands.
            Expr::Literal { .. } if caps.has_layout(&layout) == Support::Yes => {
                Materialization::DirectMap
            }
            Expr::Literal { .. } => Materialization::OnLoad,
            _ => Materialization::OnLoad,
        },
    });
    // Descend only through the nodes §10.5.1 names.
    match e {
        Expr::Cast { x, dtype, .. } => {
            // The value a narrowing cast consumed is the *better* of the two:
            // descending is how the resolver finds the higher-quality
            // representation the publisher also made available.
            let inner = x.infer().map(|t| t.dtype).unwrap_or(dtype.clone());
            let gain = if inner.bits() > dtype.bits() {
                ((inner.bits() - dtype.bits()) as f64 / inner.bits().max(1) as f64) * 0.05
            } else {
                0.0
            };
            walk(x, caps, quality + gain, lossy, scheme, out);
        }
        // Dequantization does not change the value, so its input is the same
        // quality in fewer bytes — for a runtime that can do the
        // dequantization, which the candidate now says it needs.
        Expr::Dequantize { x, .. } => walk(x, caps, quality, lossy, scheme_name(e).or(scheme), out),
        // The input of a `quantize` is the unquantized value: better, and
        // bigger.
        Expr::Quantize { x, .. } => walk(x, caps, quality + 0.01, lossy, scheme, out),
        // The subtree under an `approx` is the approximation, not a better
        // version of it — so descending finds a differently-shaped
        // representation of the same lossy value.
        Expr::Approx { x, .. } => walk(x, caps, quality, true, scheme, out),
        Expr::Select { a, b, .. } => {
            walk(a, caps, quality, lossy, scheme.clone(), out);
            walk(b, caps, quality, lossy, scheme, out);
        }
        _ => {}
    }
}

// ----------------------------------------------------------------- failures --

/// One structured reason a plan is infeasible (§10.5.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Unmet {
    pub what: String,
    /// Tensors this reason affects.
    pub tensors: usize,
    /// Bytes those tensors hold.
    pub bytes: u64,
    /// What the operator could do about it.
    pub remedy: Option<String>,
}

impl std::fmt::Display for Unmet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "✗ {}", self.what)?;
        if self.tensors > 0 {
            write!(f, "\n      → affects {} tensor(s)", self.tensors)?;
            if self.bytes > 0 {
                write!(f, " ({} bytes)", self.bytes)?;
            }
        }
        if let Some(r) = &self.remedy {
            write!(f, "\n      → remedy: {r}")?;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------- plan --

/// A per-tensor choice in a plan.
#[derive(Clone, Debug)]
pub struct Chosen {
    pub name: String,
    pub expr: Expr,
    pub dtype: DType,
    pub resident_bytes: u64,
    pub read_bytes: u64,
    pub materialization: Materialization,
    pub lossy: bool,
}

/// A realization plan (otype 0x0011, §10.4).
#[derive(Clone, Debug)]
pub struct Plan {
    pub model: Ref,
    pub caps_digest: Digest,
    pub objective: Objective,
    pub tensors: Vec<Chosen>,
    pub resident_bytes: u64,
    pub read_bytes: u64,
    pub warnings: Vec<String>,
    pub unmet: Vec<Unmet>,
}

impl Plan {
    pub fn is_feasible(&self) -> bool {
        self.unmet.is_empty()
    }

    /// §10.4: a plan is cacheable, keyed by the model, the capabilities and the
    /// objective. Domain-separated so a plan key can never be replayed as
    /// something else (§03.5.3).
    pub fn key(&self, algo: HashAlgo) -> Digest {
        let mut material = Vec::new();
        material.extend_from_slice(&self.model.1);
        material.extend_from_slice(&self.caps_digest);
        material.extend_from_slice(self.objective.name().as_bytes());
        algo.domain_digest("omni/1.0 plan-key", &material)
    }

    pub fn to_value(&self) -> Value {
        let mut tensors: Vec<(Value, Value)> = Vec::new();
        for c in &self.tensors {
            tensors.push((
                Value::text(c.name.clone()),
                Value::map(vec![
                    ("expr", c.expr.to_value()),
                    ("dtype", c.dtype.to_value()),
                    ("bytes", Value::U(c.resident_bytes)),
                    ("read", Value::U(c.read_bytes)),
                    ("materialize", Value::text(c.materialization.name())),
                ]),
            ));
        }
        Value::map(vec![
            ("t", Value::text("omni.rt/plan")),
            ("v", Value::U(1)),
            (
                "model",
                Value::Array(vec![
                    Value::U(self.model.0 as u64),
                    Value::Bytes(self.model.1.to_vec()),
                ]),
            ),
            ("caps_digest", Value::Bytes(self.caps_digest.to_vec())),
            ("objective", Value::text(self.objective.name())),
            ("tensors", Value::Map(tensors)),
            (
                "totals",
                Value::map(vec![
                    ("resident_bytes", Value::U(self.resident_bytes)),
                    ("read_bytes", Value::U(self.read_bytes)),
                ]),
            ),
            (
                "warnings",
                Value::Array(
                    self.warnings
                        .iter()
                        .map(|w| Value::text(w.clone()))
                        .collect(),
                ),
            ),
            (
                "unmet",
                Value::Array(
                    self.unmet
                        .iter()
                        .map(|u| {
                            Value::map(vec![
                                ("what", Value::text(u.what.clone())),
                                ("tensors", Value::U(u.tensors as u64)),
                                ("bytes", Value::U(u.bytes)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

// ----------------------------------------------------------------- resolver --

/// `resolve` of §10.1, following §10.5 step by step.
///
/// `optimistic` decides what to do with capabilities that are *unknown* rather
/// than refused: a resolver planning for itself may try them, one planning for a
/// remote fleet should not. §10.2 draws the distinction; this is where it is
/// acted on.
pub fn resolve(
    ctx: &Ctx<'_>,
    manifest: &Value,
    model_ref: Ref,
    table: &TensorTable,
    caps: &Capabilities,
    objective: Objective,
    optimistic: bool,
) -> Res<Plan> {
    let algo = ctx.store().hash();
    let mut plan = Plan {
        model: model_ref,
        caps_digest: caps.digest(algo),
        objective,
        tensors: Vec::new(),
        resident_bytes: 0,
        read_bytes: 0,
        warnings: Vec::new(),
        unmet: Vec::new(),
    };

    // 1. FEATURE GATE.
    for f in manifest
        .get("features")
        .and_then(|f| f.get("required"))
        .and_then(|x| x.as_array())
        .unwrap_or(&[])
    {
        let Some(name) = f.as_str() else { continue };
        match caps.has_feature(name) {
            Support::Yes => {}
            Support::Refused => plan.unmet.push(Unmet {
                what: format!("required feature `{name}` is refused by this runtime"),
                tensors: 0,
                bytes: 0,
                remedy: Some(format!(
                    "`omni convert` to a representation that does not need {name}"
                )),
            }),
            Support::Unknown if optimistic => plan.warnings.push(format!(
                "required feature `{name}` is not listed by the runtime; attempting anyway"
            )),
            Support::Unknown => plan.unmet.push(Unmet {
                what: format!("required feature `{name}` is not supported by this runtime"),
                tensors: 0,
                bytes: 0,
                remedy: Some(format!(
                    "add `{name}` to the capability set if the runtime does support it"
                )),
            }),
        }
    }
    // Optional features that are absent are simply disabled, and saying which
    // costs nothing.
    for f in manifest
        .get("features")
        .and_then(|f| f.get("optional"))
        .and_then(|x| x.as_array())
        .unwrap_or(&[])
    {
        if let Some(name) = f.as_str() {
            if caps.has_feature(name) != Support::Yes {
                plan.warnings
                    .push(format!("optional feature `{name}` disabled"));
            }
        }
    }

    // 3. ADAPTER BINDING is the caller's business (it needs the adapter's own
    //    container); `omni adapter check` is that step.

    // 4. PER-TENSOR REPRESENTATION.
    let mut refused: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for (name, r) in &table.tensors {
        let desc = match TensorDesc::load(ctx, r) {
            Ok(d) => d,
            Err(_) => {
                plan.unmet.push(Unmet {
                    what: format!("tensor `{name}` has no readable descriptor"),
                    tensors: 1,
                    bytes: 0,
                    remedy: None,
                });
                continue;
            }
        };
        let candidates = enumerate(&desc.value, caps);
        let mut feasible: Vec<Candidate> = Vec::new();
        for c in candidates {
            let mut why: Option<String> = None;
            if !caps.has_dtype(&c.dtype).usable(optimistic) {
                why = Some(format!("dtype {}", c.dtype.label()));
            } else if !caps.has_layout(&c.layout).usable(optimistic) {
                why = Some(format!("layout {}", c.layout.kind()));
            } else if let Some(scheme) = &c.requires_scheme {
                if !caps.has_scheme(scheme).usable(optimistic) {
                    why = Some(format!("quantization scheme {scheme}"));
                }
            }
            if why.is_none() && c.lossy && !caps.allow_lossy {
                why = Some("a lossy representation, which policy forbids".into());
            }
            match why {
                None => feasible.push(c),
                Some(w) => {
                    let e = refused.entry(w).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += c.resident_bytes;
                }
            }
        }
        if feasible.is_empty() {
            plan.unmet.push(Unmet {
                what: format!("no representation of `{name}` is supported"),
                tensors: 1,
                bytes: desc
                    .sizes()
                    .map(|s| desc.dtype.packed_bytes(crate::layout::numel(&s)))
                    .unwrap_or(0),
                remedy: Some("`omni convert --requantize affine-int8`".into()),
            });
            continue;
        }
        let best = pick(&feasible, objective);
        if best.lossy {
            plan.warnings.push(format!(
                "`{name}` uses a lossy representation ({}), which the policy allows",
                best.dtype.label()
            ));
        }
        plan.resident_bytes += best.resident_bytes;
        plan.read_bytes += best.read_bytes;
        plan.tensors.push(Chosen {
            name: name.clone(),
            expr: best.expr.clone(),
            dtype: best.dtype.clone(),
            resident_bytes: best.resident_bytes,
            read_bytes: best.read_bytes,
            materialization: best.materialization,
            lossy: best.lossy,
        });
    }

    // 5. BUDGET CHECK. Retrying under min-memory is the documented recovery,
    //    and it is only worth doing if the objective was not already that.
    if let Some(budget) = caps.memory_bytes {
        if plan.resident_bytes > budget && objective != Objective::MinMemory {
            let retry = resolve(
                ctx,
                manifest,
                model_ref,
                table,
                caps,
                Objective::MinMemory,
                optimistic,
            )?;
            if retry.resident_bytes <= budget && retry.is_feasible() {
                let mut retry = retry;
                retry.objective = objective;
                retry.warnings.push(format!(
                    "objective {} exceeded the memory budget; re-resolved under min-memory",
                    objective.name()
                ));
                return Ok(retry);
            }
        }
        if plan.resident_bytes > budget {
            plan.unmet.push(Unmet {
                what: format!(
                    "budget: minimum feasible resident {} bytes exceeds the device's {budget}",
                    plan.resident_bytes
                ),
                tensors: plan.tensors.len(),
                bytes: plan.resident_bytes,
                remedy: Some(
                    "`--objective min-memory --allow-lossy`, or a device with more memory".into(),
                ),
            });
        }
    }

    // The refusals that did not make a tensor infeasible are still worth
    // reporting: they are why a representation was not chosen.
    for (why, (tensors, bytes)) in refused {
        if plan.tensors.iter().any(|c| c.lossy) || !plan.unmet.is_empty() {
            plan.warnings.push(format!(
                "{tensors} candidate(s) rejected: {why} ({bytes} bytes)"
            ));
        }
    }
    Ok(plan)
}

/// The scheme name a `dequantize`/`quantize` node uses, if any.
fn scheme_name(e: &Expr) -> Option<String> {
    match e {
        Expr::Dequantize { scheme, .. } | Expr::Quantize { scheme, .. } => scheme
            .get("scheme")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

/// §10.5 step 4's `argmin/argmax over feasible by objective`, with the
/// tie-breaks the section names. Deterministic: two implementations that follow
/// this order produce the same plan.
fn pick(feasible: &[Candidate], objective: Objective) -> &Candidate {
    let cmp = |a: &&Candidate, b: &&Candidate| -> std::cmp::Ordering {
        match objective {
            Objective::MinMemory => a
                .resident_bytes
                .cmp(&b.resident_bytes)
                .then(total(b.quality_delta, a.quality_delta)),
            Objective::MaxQuality => total(b.quality_delta, a.quality_delta)
                .then(a.resident_bytes.cmp(&b.resident_bytes)),
            Objective::MinLoadTime => {
                (a.read_bytes + a.resident_bytes).cmp(&(b.read_bytes + b.resident_bytes))
            }
            // Without a device model, "compute dtype match" reduces to
            // preferring the representation that needs no materialization.
            Objective::MinLatency => materialization_rank(a)
                .cmp(&materialization_rank(b))
                .then(a.resident_bytes.cmp(&b.resident_bytes)),
            Objective::Balanced => {
                let score = |c: &Candidate| {
                    c.resident_bytes as f64 / (1 << 20) as f64 - c.quality_delta * 100.0
                };
                total(score(a), score(b))
            }
        }
        // Any remaining tie is broken by expression identity, so the choice does
        // not depend on map iteration order.
        .then_with(|| a.expr.to_value().encode().cmp(&b.expr.to_value().encode()))
    };
    feasible.iter().min_by(cmp).expect("feasible is non-empty")
}

fn materialization_rank(c: &Candidate) -> u8 {
    match c.materialization {
        Materialization::DirectMap => 0,
        Materialization::Cache => 1,
        Materialization::OnLoad => 2,
    }
}

fn total(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Builds the `Plan` object for storage.
pub fn plan_object(p: &Plan) -> crate::container::Object {
    crate::container::Object::structure(otype::PLAN, &p.to_value())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::Object;
    use crate::dtype::Round;
    use crate::expr::dims;
    use crate::store::{MemoryStore, WritableStore};
    use crate::tensor::Materialize;

    fn store_tensor(s: &mut MemoryStore, shape: &[u64], dtype: &DType, layout: &Layout) -> Expr {
        let n = crate::layout::numel(shape);
        let t = crate::expr::Tensor::new(shape.to_vec(), dtype.clone(), vec![1.0; n as usize]);
        let bytes = t.to_bytes(dtype, layout, Round::Rne).unwrap();
        let d = s.put(&bytes).unwrap();
        Expr::Literal {
            chunks: (otype::BLOB, d),
            dtype: dtype.clone(),
            shape: dims(shape),
            layout: layout.clone(),
        }
    }

    fn desc(value: Expr) -> TensorDesc {
        let t = value.infer().unwrap();
        TensorDesc {
            shape: t.shape,
            dtype: t.dtype,
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

    fn table_of(s: &mut MemoryStore, entries: Vec<(&str, TensorDesc)>) -> TensorTable {
        let mut t = TensorTable::default();
        for (name, d) in entries {
            let obj = Object::structure(otype::TENSOR_DESC, &d.to_value());
            let dig = s.put(&obj.payload).unwrap();
            t.tensors
                .insert(name.to_string(), (otype::TENSOR_DESC, dig));
        }
        t
    }

    fn manifest(required: &[&str]) -> Value {
        Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            (
                "features",
                Value::map(vec![
                    (
                        "required",
                        Value::Array(required.iter().map(|f| Value::text(*f)).collect()),
                    ),
                    ("optional", Value::Array(vec![Value::text("omni.rt/kv.1")])),
                ]),
            ),
        ])
    }

    /// The §04.8 model: int4 stored, bf16 derived, fp8 cast of that.
    fn three_realizations(s: &mut MemoryStore) -> Expr {
        let packed = Layout::Packed {
            elems_per_word: 8,
            word_bits: 32,
            bit_order: crate::layout::BitOrder::LsbFirst,
            order: crate::layout::Order::RowMajor,
        };
        let q = store_tensor(s, &[16, 32], &DType::U4, &packed);
        let scales = store_tensor(s, &[16, 1], &DType::BF16, &Layout::default());
        let deq = Expr::Dequantize {
            x: Box::new(q),
            scheme: Value::map(vec![
                ("scheme", Value::text("affine")),
                ("formula", Value::text("sym")),
                ("out", DType::BF16.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(32)])),
                ("scale", scales.to_value()),
            ]),
        };
        Expr::Cast {
            x: Box::new(deq),
            dtype: DType::F8E4M3,
            round: Round::Rne,
        }
    }

    #[test]
    fn enumeration_yields_every_standalone_representation() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let caps = Capabilities::reference();
        let cands = enumerate(&e, &caps);
        // fp8 (the root), bf16 (the dequantization) and int4 (the literal).
        let labels: Vec<String> = cands.iter().map(|c| c.dtype.label()).collect();
        assert!(labels.contains(&"f8e4m3".to_string()), "{labels:?}");
        assert!(labels.contains(&"bf16".to_string()), "{labels:?}");
        assert!(labels.contains(&"u4".to_string()), "{labels:?}");
        // The int4 one is the smallest and is a direct map; the bf16 one is not.
        let int4 = cands.iter().find(|c| c.dtype == DType::U4).unwrap();
        let bf16 = cands.iter().find(|c| c.dtype == DType::BF16).unwrap();
        assert!(int4.resident_bytes < bf16.resident_bytes);
        assert_eq!(int4.materialization, Materialization::DirectMap);
        assert_eq!(bf16.materialization, Materialization::OnLoad);
    }

    /// A tensor published as bf16 with an int8 quantization of it: the two
    /// objectives genuinely disagree here, which `three_realizations` does not —
    /// there, the int4 form has the *same* values in a quarter of the memory, so
    /// every objective rightly prefers it.
    fn lossy_pair(s: &mut MemoryStore) -> Expr {
        let w = store_tensor(s, &[16, 32], &DType::BF16, &Layout::default());
        Expr::Quantize {
            x: Box::new(w),
            scheme: Value::map(vec![
                ("scheme", Value::text("sym")),
                ("out", DType::I8.to_value()),
                ("block", Value::Array(vec![Value::U(1), Value::U(32)])),
            ]),
            round: Round::Rne,
        }
    }

    #[test]
    fn a_dequantizable_representation_needs_the_scheme() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        // A runtime that cannot do affine dequantization must not be handed the
        // int4 codes and told they are the weights.
        let mut caps = Capabilities::reference();
        caps.quant_schemes.clear();
        caps.unsupported.insert("affine".into());
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(p.is_feasible(), "{:?}", p.unmet);
        assert_ne!(p.tensors[0].dtype, DType::U4);
        // With the scheme available, the int4 form is chosen and it is a
        // quarter of the size.
        let caps = Capabilities::reference();
        let q = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert_eq!(q.tensors[0].dtype, DType::U4);
        assert!(q.resident_bytes < p.resident_bytes);
    }

    #[test]
    fn the_objective_decides_which_representation_is_chosen() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = lossy_pair(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0", "omni.tensor/expr.1"]);
        let ctx = Ctx::new(&s);
        let caps = Capabilities::reference();

        let mem = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(mem.is_feasible(), "{:?}", mem.unmet);
        // int8 is a quarter of the bf16 size, and lossy — which the plan says.
        assert_eq!(mem.tensors[0].dtype, DType::I8);
        assert!(mem.tensors[0].lossy);

        let qual = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MaxQuality,
            false,
        )
        .unwrap();
        // The unquantized bf16 value is the higher-quality candidate, and the
        // plan says so by choosing it and by not being lossy.
        assert_eq!(qual.tensors[0].dtype, DType::BF16);
        assert!(!qual.tensors[0].lossy);
        assert!(qual.resident_bytes > mem.resident_bytes);
    }

    #[test]
    fn resolution_is_deterministic_and_the_plan_is_keyed() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let caps = Capabilities::reference();
        let a = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [1u8; 32]),
            &table,
            &caps,
            Objective::Balanced,
            false,
        )
        .unwrap();
        let b = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [1u8; 32]),
            &table,
            &caps,
            Objective::Balanced,
            false,
        )
        .unwrap();
        assert_eq!(a.to_value().encode(), b.to_value().encode());
        assert_eq!(a.key(HashAlgo::default()), b.key(HashAlgo::default()));
        // The key is domain-separated: it is not the digest of the plan bytes.
        assert_ne!(
            a.key(HashAlgo::default()).to_vec(),
            HashAlgo::default().digest(&a.to_value().encode()).to_vec()
        );
        // A different objective is a different plan and a different key.
        let c = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [1u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert_ne!(a.key(HashAlgo::default()), c.key(HashAlgo::default()));
    }

    #[test]
    fn a_c0_runtime_cannot_hold_an_expression_model() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0", "omni.tensor/expr.1"]);
        let ctx = Ctx::new(&s);
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &Capabilities::c0(),
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(!p.is_feasible());
        // The failure names the feature and suggests something to do about it.
        let s = format!("{}", p.unmet[0]);
        assert!(s.contains("omni.tensor/expr.1"), "{s}");
        assert!(s.contains("remedy"), "{s}");
    }

    #[test]
    fn unknown_and_refused_capabilities_are_not_the_same_thing() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);

        // A runtime that lists neither `packed` nor `strided` layouts: unknown.
        // Pessimistically nothing is feasible; optimistically it is attempted.
        let mut caps = Capabilities::reference();
        caps.layouts.clear();
        let pessimistic = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(!pessimistic.is_feasible());
        let optimistic = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            true,
        )
        .unwrap();
        assert!(optimistic.is_feasible());

        // A runtime that *refuses* the packed layout: never attempted, at any
        // level of optimism.
        let mut caps = Capabilities::reference();
        caps.unsupported.insert("packed".into());
        for optimism in [false, true] {
            let p = resolve(
                &ctx,
                &m,
                (otype::MANIFEST, [0u8; 32]),
                &table,
                &caps,
                Objective::MinMemory,
                optimism,
            )
            .unwrap();
            // The int4 candidate is gone, so a wider one is chosen instead.
            assert_ne!(p.tensors[0].dtype, DType::U4, "optimistic={optimism}");
        }
    }

    #[test]
    fn a_policy_that_forbids_loss_refuses_a_lossy_representation() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let base = store_tensor(&mut s, &[8, 8], &DType::BF16, &Layout::default());
        let lossy = Expr::Approx {
            x: Box::new(base.clone()),
            bound: crate::expr::Bound::Rel(1e-2),
        };
        let table = table_of(&mut s, vec![("w", desc(lossy))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let mut caps = Capabilities::reference();
        caps.allow_lossy = false;
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        // Every candidate under an `approx` is lossy, so there is nothing left.
        assert!(!p.is_feasible());
        assert!(format!("{}", p.unmet[0]).contains("no representation"));
        // With the policy relaxed, it resolves.
        caps.allow_lossy = true;
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(p.is_feasible());
        assert!(p.warnings.iter().any(|w| w.contains("lossy")));
    }

    #[test]
    fn a_budget_forces_the_smaller_representation() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = lossy_pair(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let mut caps = Capabilities::reference();
        // Enough for the int8 representation (512 bytes) but not the bf16 one
        // (1024).
        caps.memory_bytes = Some(600);
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MaxQuality,
            false,
        )
        .unwrap();
        assert!(p.is_feasible(), "{:?}", p.unmet);
        assert_eq!(p.tensors[0].dtype, DType::I8);
        assert!(p.warnings.iter().any(|w| w.contains("min-memory")));

        // A budget nothing fits is an informative failure, not a silent
        // truncation.
        caps.memory_bytes = Some(16);
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(!p.is_feasible());
        assert!(format!("{}", p.unmet[0]).contains("budget"));
    }

    #[test]
    fn select_is_resolved_statically() {
        // §10.3: a feature-conditional value never reaches an executable plan.
        let mut s = MemoryStore::new(HashAlgo::default());
        let fp8 = store_tensor(&mut s, &[8, 8], &DType::F8E4M3, &Layout::default());
        let bf16 = store_tensor(&mut s, &[8, 8], &DType::BF16, &Layout::default());
        let sel = Expr::Select {
            feature: "omni.dtype/f8e4m3.1".into(),
            a: Box::new(Expr::Cast {
                x: Box::new(fp8),
                dtype: DType::BF16,
                round: Round::Rne,
            }),
            b: Box::new(bf16),
        };
        let table = table_of(&mut s, vec![("w", desc(sel))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let caps = Capabilities::reference();
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &caps,
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(p.is_feasible());
        // Both branches were enumerated, and the chosen expression is one of
        // them rather than the `select` itself.
        assert_ne!(p.tensors[0].expr.op(), "select");
        assert_eq!(p.tensors[0].dtype, DType::F8E4M3);
    }

    #[test]
    fn optional_features_that_are_absent_are_disabled_not_fatal() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = store_tensor(&mut s, &[4], &DType::F32, &Layout::default());
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [0u8; 32]),
            &table,
            &Capabilities::reference(),
            Objective::MinMemory,
            false,
        )
        .unwrap();
        assert!(p.is_feasible());
        assert!(p
            .warnings
            .iter()
            .any(|w| w.contains("omni.rt/kv.1") && w.contains("disabled")));
    }

    #[test]
    fn capability_sets_round_trip() {
        for caps in [Capabilities::c0(), Capabilities::reference()] {
            let v = caps.to_value();
            let back = Capabilities::from_value(&v).unwrap();
            assert_eq!(back.to_value().encode(), v.encode());
            let round = crate::cbor::decode(&v.encode()).unwrap();
            assert_eq!(
                Capabilities::from_value(&round)
                    .unwrap()
                    .to_value()
                    .encode(),
                v.encode()
            );
            // The digest is what a plan records.
            assert_eq!(
                caps.digest(HashAlgo::default()),
                back.digest(HashAlgo::default())
            );
        }
        assert!(Capabilities::from_value(&Value::map(vec![("t", Value::text("nope"))])).is_err());
    }

    #[test]
    fn plans_round_trip_through_cbor() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let e = three_realizations(&mut s);
        let table = table_of(&mut s, vec![("w", desc(e))]);
        let m = manifest(&["omni.core/1.0"]);
        let ctx = Ctx::new(&s);
        let p = resolve(
            &ctx,
            &m,
            (otype::MANIFEST, [2u8; 32]),
            &table,
            &Capabilities::reference(),
            Objective::MinMemory,
            false,
        )
        .unwrap();
        let v = p.to_value();
        let bytes = v.encode();
        let round = crate::cbor::decode(&bytes).unwrap();
        assert_eq!(round.encode(), bytes);
        assert_eq!(
            round.get("objective").and_then(|x| x.as_str()),
            Some("min-memory")
        );
        // And the object is storable.
        let obj = plan_object(&p);
        assert_eq!(obj.otype, otype::PLAN);
    }

    #[test]
    fn objectives_and_materializations_name_themselves() {
        for o in [
            Objective::MinMemory,
            Objective::MaxQuality,
            Objective::MinLoadTime,
            Objective::MinLatency,
            Objective::Balanced,
        ] {
            assert_eq!(Objective::parse(o.name()), Some(o));
        }
        assert!(Objective::parse("fastest").is_none());
        assert_eq!(Materialization::DirectMap.name(), "direct-map");
    }
}
