//! PEFT LoRA import (`docs/design/import-export.md` §3).
//!
//! The capability matrix has a footnote for this row: PEFT LoRA *is* safetensors
//! plus a config, imported as an `Adapter`. That is exactly what this does, and
//! the two halves it needs — [`crate::safetensors`] and [`crate::json`] — already
//! exist, as does the §08 adapter machinery it produces.
//!
//! ## The one thing PEFT cannot tell you
//!
//! An `adapter_config.json` names its base as a *string*:
//! `"base_model_name_or_path": "meta-llama/Llama-3-8B"`. §08.1 requires an
//! `Adapter` to pin its base by **digest**, so an adapter can never silently
//! attach to a different base — which is the guarantee OMNI adds and PEFT does
//! not have. So the base container is a required argument, not an optional one:
//! there is no honest way to synthesize the digest of a model you were handed the
//! name of. The name is preserved in the report, and if it disagrees with the
//! base's own name that becomes a warning rather than a silent mismatch.
//!
//! ## What is refused rather than approximated
//!
//! `use_dora`, `fan_in_fan_out`, `rank_pattern`, `alpha_pattern`,
//! `modules_to_save`, and any `peft_type` other than `LORA`. Each changes what
//! the update *is* — DoRA adds a magnitude vector, `fan_in_fan_out` transposes
//! the base, a rank pattern makes the rank per-module — and producing a LoRA that
//! quietly ignores them would be an adapter that computes something the source
//! did not. Every one is refused by name, with the field that caused it.

use crate::adapter::lora_adapter_value;
use crate::cbor::Value;
use crate::container::{otype, Container, Digest, HashAlgo, Object};
use crate::expr::{Ctx, Expr};
use crate::json;
use crate::model::ModelBuilder;
use crate::safetensors::{self, Fidelity, Note};
use crate::tensor::{Materialize, TensorDesc, TensorTable};

pub const IMPORTER: &str = "omni-import-peft";

/// The prefix PEFT puts on every tensor name. `base_model` is the wrapped model
/// and `model` is its first attribute; what follows is the base's own name for
/// the module.
pub const PREFIX: &str = "base_model.model.";

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    /// Well-formed, and says something this importer will not approximate.
    Unsupported(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed PEFT adapter: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// --------------------------------------------------------------------- config --

/// The fields of `adapter_config.json` this importer reads.
#[derive(Clone, Debug)]
pub struct Config {
    pub peft_type: String,
    pub r: u64,
    pub lora_alpha: f64,
    /// Module suffixes, e.g. `["q_proj", "v_proj"]`.
    pub target_modules: Vec<String>,
    pub lora_dropout: Option<f64>,
    pub base_model_name: Option<String>,
    pub task_type: Option<String>,
}

impl Config {
    pub fn parse(bytes: &[u8]) -> Res<Config> {
        let v = json::parse(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        let peft_type = s("peft_type").unwrap_or_default();
        if peft_type != "LORA" {
            return Err(Error::Unsupported(format!(
                "`peft_type` is {:?}; this importer does LORA and refuses the rest \
                 rather than approximating them",
                if peft_type.is_empty() {
                    "absent"
                } else {
                    &peft_type
                }
            )));
        }
        // Each of these changes what the update is, so each is a refusal.
        for (key, why) in [
            (
                "use_dora",
                "DoRA scales by a learned magnitude vector, which a plain \
                          LoRA update does not have",
            ),
            (
                "fan_in_fan_out",
                "the base weight is stored transposed, so the update \
                                is A·B rather than B·A",
            ),
            (
                "use_rslora",
                "rank-stabilised LoRA scales by alpha/sqrt(r), not alpha/r",
            ),
        ] {
            if v.get(key).and_then(|x| x.as_bool()) == Some(true) {
                return Err(Error::Unsupported(format!("`{key}` is true: {why}")));
            }
        }
        for key in ["rank_pattern", "alpha_pattern"] {
            if v.get(key)
                .and_then(|x| x.as_object())
                .is_some_and(|m| !m.is_empty())
            {
                return Err(Error::Unsupported(format!(
                    "`{key}` overrides the rank or alpha per module; this importer \
                     writes one rank and one alpha, and will not pick which modules \
                     to be wrong about"
                )));
            }
        }
        if v.get("modules_to_save")
            .and_then(|x| x.as_array())
            .is_some_and(|a| !a.is_empty())
        {
            return Err(Error::Unsupported(
                "`modules_to_save` replaces whole modules rather than adding a \
                 low-rank update; that is a different operation (§08.4)"
                    .into(),
            ));
        }

        let r = v
            .get("r")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| Error::Malformed("no `r`".into()))?;
        if r == 0 {
            return Err(Error::Malformed("`r` is 0".into()));
        }
        let lora_alpha = v
            .get("lora_alpha")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| Error::Malformed("no `lora_alpha`".into()))?;
        // PEFT accepts a list or, for a regex, a bare string.
        let target_modules = match v.get("target_modules") {
            Some(json::Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| Error::Malformed("a non-string target module".into()))
                })
                .collect::<Res<Vec<String>>>()?,
            Some(json::Value::Str(_)) => {
                return Err(Error::Unsupported(
                    "`target_modules` is a string, which PEFT treats as a regex over \
                     module paths; this importer takes the list form"
                        .into(),
                ))
            }
            _ => return Err(Error::Malformed("no `target_modules`".into())),
        };
        if target_modules.is_empty() {
            return Err(Error::Malformed("`target_modules` is empty".into()));
        }
        Ok(Config {
            peft_type,
            r,
            lora_alpha,
            target_modules,
            lora_dropout: v.get("lora_dropout").and_then(|x| x.as_f64()),
            base_model_name: s("base_model_name_or_path"),
            task_type: s("task_type"),
        })
    }
}

/// One imported factor: which base tensor it updates, and its two matrices.
#[derive(Debug)]
pub struct Factor {
    /// The base tensor this updates, by the base's own name.
    pub base_tensor: String,
    /// The target module suffix that matched, e.g. `q_proj`.
    pub target: String,
    pub a_name: String,
    pub b_name: String,
    pub a_shape: Vec<u64>,
    pub b_shape: Vec<u64>,
}

pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
    pub factors: Vec<Factor>,
}

/// A summary rather than a dump: a failing assertion wants to know what was
/// imported, not the bytes of a hundred objects.
impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Imported {{ {} objects, root {}, {} factors, lossless {} }}",
            self.objects.len(),
            crate::sha256::hex(&self.root[..6]),
            self.factors.len(),
            self.report.lossless
        )
    }
}

#[derive(Clone, Debug)]
pub struct ImportOpts {
    pub name: String,
    pub config_path: String,
    pub weights_path: String,
    pub chunk_size: usize,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/peft-lora".into(),
            config_path: String::new(),
            weights_path: String::new(),
            chunk_size: 1 << 20,
        }
    }
}

/// The base tensor a PEFT factor updates, and which target matched it.
///
/// `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` describes the
/// base's `model.layers.0.self_attn.q_proj.weight`. That mapping is a *convention*
/// of PEFT's, not a guarantee, so the result is looked up in the base's table and
/// an unmatched factor is reported rather than assumed.
fn base_tensor_of(peft_name: &str, targets: &[String]) -> Option<(String, String)> {
    let rest = peft_name.strip_prefix(PREFIX)?;
    let module = rest
        .strip_suffix(".lora_A.weight")
        .or_else(|| rest.strip_suffix(".lora_B.weight"))?;
    let target = targets
        .iter()
        .find(|t| module == t.as_str() || module.ends_with(&format!(".{t}")))?;
    Some((format!("{module}.weight"), target.clone()))
}

/// Imports a PEFT LoRA over a base container.
pub fn import(
    config_bytes: &[u8],
    weights: &[u8],
    base: &Container,
    opts: &ImportOpts,
) -> Res<Imported> {
    let cfg = Config::parse(config_bytes)?;
    let hash = base.header.hash;
    let f = safetensors::File::parse(weights).map_err(|e| Error::Malformed(e.to_string()))?;

    let mut report = Fidelity {
        format: "peft",
        importer: IMPORTER,
        source_path: opts.config_path.clone(),
        // Both halves of the adapter, hashed together in a fixed order, so
        // "which files did this come from?" has one answer (I6).
        source_digest: hash.digest(&[config_bytes, weights].concat()),
        source_size: (config_bytes.len() + weights.len()) as u64,
        lossless: true,
        represented: vec![
            "lora factors".into(),
            "rank".into(),
            "alpha".into(),
            "target_modules".into(),
        ],
        ..Default::default()
    };

    // The base's table, which is what decides whether a factor has anything to
    // attach to.
    let store = crate::store::Borrowed(base);
    let bctx = Ctx::new(&store);
    let btable = base_table(base)?;

    // Pair the A and B tensors by the base tensor they name.
    let mut pending: std::collections::BTreeMap<
        String,
        (
            String,
            Option<&safetensors::Entry>,
            Option<&safetensors::Entry>,
        ),
    > = Default::default();
    let mut ignored = Vec::new();
    for e in &f.entries {
        let Some((base_name, target)) = base_tensor_of(&e.name, &cfg.target_modules) else {
            ignored.push(e.name.clone());
            continue;
        };
        let slot = pending
            .entry(base_name)
            .or_insert_with(|| (target.clone(), None, None));
        if e.name.ends_with(".lora_A.weight") {
            slot.1 = Some(e);
        } else {
            slot.2 = Some(e);
        }
    }
    for name in &ignored {
        report.unrepresented.push(Note {
            item: name.clone(),
            reason: "not a `lora_A`/`lora_B` factor for a declared target module".into(),
            action: "not imported".into(),
        });
        report.lossless = false;
    }
    if pending.is_empty() {
        return Err(Error::Malformed(format!(
            "no LoRA factors for {:?} in {} tensor(s)",
            cfg.target_modules,
            f.entries.len()
        )));
    }

    // Build the adapter's own tensors, keeping the names PEFT gave them (minus its
    // prefix) so a human can match them back to the source file.
    let mut b = ModelBuilder::new(opts.name.clone())
        .hash(hash)
        .chunk_size(opts.chunk_size);
    let mut factors = Vec::new();
    let mut rank_axis = None;
    for (base_name, (target, a, bb)) in &pending {
        let (Some(a), Some(bb)) = (a, bb) else {
            return Err(Error::Malformed(format!(
                "`{base_name}` has only one of its two factors"
            )));
        };
        // R-A02: the factors have to fit the base tensor they claim to update.
        let Some(bref) = btable.get(base_name) else {
            return Err(Error::Malformed(format!(
                "`{base_name}` is not a tensor in the base container; PEFT's naming \
                 convention does not match this base"
            )));
        };
        let bdesc = TensorDesc::load(&bctx, bref).map_err(|e| Error::Core(e.to_string()))?;
        let bshape = bdesc
            .sizes()
            .ok_or_else(|| Error::Core(format!("`{base_name}` has a symbolic shape")))?;
        if bshape.len() != 2 {
            return Err(Error::Unsupported(format!(
                "`{base_name}` is {}-dimensional; a LoRA update is a matrix product",
                bshape.len()
            )));
        }
        if a.shape != vec![cfg.r, bshape[1]] {
            return Err(Error::Malformed(format!(
                "{}: lora_A is {:?}, expected [r={}, in={}]",
                a.name, a.shape, cfg.r, bshape[1]
            )));
        }
        if bb.shape != vec![bshape[0], cfg.r] {
            return Err(Error::Malformed(format!(
                "{}: lora_B is {:?}, expected [out={}, r={}]",
                bb.name, bb.shape, bshape[0], cfg.r
            )));
        }
        if rank_axis.is_none() {
            // The base's own name for the axis the rank contracts over. Naming it
            // is what lets `require` catch a mismatch rather than the
            // multiplication being quietly wrong (§08.3).
            rank_axis = bdesc.axes.as_ref().and_then(|a| a.last().cloned());
        }

        for (e, axes) in [(a, ["rank", "in"]), (bb, ["out", "rank"])] {
            let short = e.name.strip_prefix(PREFIX).unwrap_or(&e.name).to_string();
            let value = b.literal(
                f.tensor(e),
                e.dtype.clone(),
                &e.shape,
                safetensors::layout_of(&e.dtype),
            );
            b = b.derived(
                short,
                TensorDesc {
                    shape: crate::expr::dims(&e.shape),
                    dtype: e.dtype.clone(),
                    layout: safetensors::layout_of(&e.dtype),
                    value,
                    semantic: Some("weight".into()),
                    role: Some("lora".into()),
                    axes: Some(axes.iter().map(|x| x.to_string()).collect()),
                    device_hint: None,
                    materialize: Materialize::Lazy,
                    stats: None,
                    digest_materialized: None,
                },
            );
        }
        factors.push(Factor {
            base_tensor: base_name.clone(),
            target: target.clone(),
            a_name: a.name.clone(),
            b_name: bb.name.clone(),
            a_shape: a.shape.clone(),
            b_shape: bb.shape.clone(),
        });
    }

    // I1: what PEFT does not state stays absent, and what it states as a *name*
    // is recorded as a name rather than promoted to an identity.
    report.assumptions.push(Note {
        item: "base".into(),
        reason: match &cfg.base_model_name {
            Some(n) => format!("PEFT names its base `{n}`, which is a name and not a digest"),
            None => "PEFT declares no base at all".into(),
        },
        action: format!(
            "pinned to the base given on the command line, {}",
            crate::sha256::hex(&base.header.root_digest)
        ),
    });
    if let Some(d) = cfg.lora_dropout {
        if d > 0.0 {
            report.assumptions.push(Note {
                item: "lora_dropout".into(),
                reason: format!("{d} applies during training, not to the merged weights"),
                action: "recorded on the adapter, not applied".into(),
            });
        }
    }
    if let Some(t) = &cfg.task_type {
        report.assumptions.push(Note {
            item: "task_type".into(),
            reason: format!("PEFT says {t}, which describes the training head"),
            action: "not imported: OMNI has no field for it".into(),
        });
    }

    // The builder makes a whole model graph — manifest, metadata, model — and an
    // adapter needs only the tensor table out of it. The rest is pruned at the
    // end, once the manifest that decides what is reachable exists.
    let (mut objects, _) = b.build();
    let tensors_ref = objects
        .iter()
        .find(|o| o.otype == otype::TENSOR_TABLE)
        .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
        .ok_or_else(|| Error::Core("the builder produced no tensor table".into()))?;

    // I4 before the report is written, not after: the counts belong in the object
    // that goes into the container, and a report rewritten afterwards would change
    // its own digest and every digest above it. Verifying against a store over the
    // objects — rather than a packed container — is enough, because what is being
    // checked is the tensor bytes and not the framing.
    {
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let mctx = Ctx::new(&mem);
        let table =
            TensorTable::load(&mctx, &tensors_ref).map_err(|e| Error::Core(e.to_string()))?;
        for e in &f.entries {
            let short = e.name.strip_prefix(PREFIX).unwrap_or(&e.name);
            let Some(r) = table.get(short) else { continue };
            let d = TensorDesc::load(&mctx, r).map_err(|err| Error::Core(err.to_string()))?;
            let Expr::Literal { chunks, .. } = &d.value else {
                return Err(Error::Core(format!("`{short}` is not a literal")));
            };
            let got = mctx
                .chunk_bytes(chunks)
                .map_err(|err| Error::Core(err.to_string()))?;
            if got != f.tensor(e) {
                return Err(Error::Core(format!(
                    "I4: `{}` did not survive import byte for byte",
                    e.name
                )));
            }
            report.verified_tensors += 1;
            report.verified_bytes += e.len();
        }
    }

    // One attach rule per target module: a glob over the base's names, with the
    // adapter's own tensors named from what the glob captured.
    let targets: Vec<String> = cfg
        .target_modules
        .iter()
        .filter(|t| factors.iter().any(|f| &f.target == *t))
        // `**`, not `*`: §08.3's `*` stops at a `.`, and a target module lives
        // several dots deep in `model.layers.0.self_attn.q_proj.weight`.
        .map(|t| format!("**.{t}.weight"))
        .collect();
    if targets.len() < cfg.target_modules.len() {
        for t in &cfg.target_modules {
            if !factors.iter().any(|f| &f.target == t) {
                report.unrepresented.push(Note {
                    item: format!("target_modules.{t}"),
                    reason: "declared as a target but no factor in the weights file \
                             updates it"
                        .into(),
                    action: "no attach rule written for it".into(),
                });
                report.lossless = false;
            }
        }
    }
    let target_refs: Vec<&str> = targets.iter().map(String::as_str).collect();
    let adapter = lora_adapter_value(
        &(otype::MANIFEST, base.header.root_digest),
        &tensors_ref,
        cfg.r,
        cfg.lora_alpha,
        &target_refs,
        // `{1}` is what the glob captured — the module path without the target.
        // The bind names have to be the adapter's own tensor names, and they are:
        // PEFT's names with its prefix removed.
        "{1}.$T.lora_A.weight",
        "{1}.$T.lora_B.weight",
        rank_axis.as_deref().unwrap_or("in"),
    )
    .map_err(|e| Error::Core(e.to_string()))?;
    // `lora_adapter_value` writes one bind pair for every target, so `$T` has to
    // become the target each rule is for. Rewriting it here keeps that function
    // unaware of PEFT's naming.
    let mut adapter = specialise_binds(&adapter, &cfg.target_modules)?;

    // R-A03 checks that the base names the axis the rank contracts over — and a
    // base imported from safetensors names no axes at all, because safetensors
    // does not say what its dimensions mean. Asserting a requirement the base
    // cannot satisfy would make every attach *invalid* rather than merely
    // unchecked, so when there are no axes the requirement is not written.
    if rank_axis.is_none() {
        adapter = drop_requires(&adapter);
        report.assumptions.push(Note {
            item: "require.rank_axis".into(),
            reason: "the base declares no axes, so there is no name for the axis the \
                     rank contracts over"
                .into(),
            action: "no rank-axis requirement written; the shapes are still checked \
                     (R-A02)"
                .into(),
        });
    }

    let adapter_obj = Object::structure(otype::ADAPTER, &adapter);
    let adapter_ref = adapter_obj.digest(hash);
    objects.push(adapter_obj);

    let manifest = Object::structure(
        otype::MANIFEST,
        &Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("adapter")),
            ("created", Value::U(0)),
            (
                "assets",
                Value::map(vec![
                    (
                        "adapter",
                        Value::Array(vec![
                            Value::U(otype::ADAPTER as u64),
                            Value::Bytes(adapter_ref.to_vec()),
                        ]),
                    ),
                    ("provenance", {
                        let p = Object::structure(otype::PROVENANCE, &report.to_value());
                        let d = p.digest(hash);
                        objects.push(p);
                        Value::Array(vec![
                            Value::U(otype::PROVENANCE as u64),
                            Value::Bytes(d.to_vec()),
                        ])
                    }),
                ]),
            ),
            ("entry", Value::text("adapter")),
            (
                "parents",
                // Written by `delta::Parent` rather than by hand: the reader that
                // has to understand this is `delta::parents`, and spelling the
                // keys myself is how they come to disagree.
                Value::Array(vec![crate::delta::Parent {
                    reference: (otype::MANIFEST, base.header.root_digest),
                    role: "base".into(),
                    name: cfg.base_model_name.clone(),
                    locators: Vec::new(),
                    // The base is not in this container and does not have to be:
                    // §01.4 makes that incomplete rather than invalid, and an
                    // adapter is published on its own.
                    required: false,
                }
                .to_value()]),
            ),
            (
                "features",
                Value::map(vec![
                    (
                        "required",
                        Value::Array(vec![
                            Value::text("omni.core/1.0"),
                            Value::text("omni.adapt/1.0"),
                        ]),
                    ),
                    ("optional", Value::Array(vec![])),
                ]),
            ),
        ]),
    );
    let root = manifest.digest(hash);
    objects.push(manifest);

    // Prune to what the manifest can reach. The builder's own manifest, metadata
    // and model object describe a model this container is not — an adapter is not
    // a model — and leaving them in would put objects in the file that nothing
    // names.
    let keep = reachable(&objects, &root, hash);
    objects.retain(|o| keep.contains(&o.digest(hash)));

    Ok(Imported {
        objects,
        root,
        report,
        factors,
    })
}

/// Replaces `$T` in each attach rule's binds with the target that rule selects.
fn specialise_binds(adapter: &Value, targets: &[String]) -> Res<Value> {
    let Value::Map(pairs) = adapter else {
        return Err(Error::Core("the adapter is not a map".into()));
    };
    let mut out = Vec::new();
    for (k, v) in pairs {
        if k.as_str() != Some("attach") {
            out.push((k.clone(), v.clone()));
            continue;
        }
        let Value::Array(rules) = v else {
            return Err(Error::Core("`attach` is not an array".into()));
        };
        let mut fixed = Vec::new();
        for rule in rules {
            // The rule's glob is `*.<target>.weight`, so the target is in it.
            let glob = rule
                .get("select")
                .and_then(|s| s.get("glob"))
                .and_then(|g| g.as_str())
                .ok_or_else(|| Error::Core("an attach rule with no glob".into()))?
                .to_string();
            let target = targets
                .iter()
                .find(|t| glob == format!("**.{t}.weight"))
                .ok_or_else(|| Error::Core(format!("no target for glob `{glob}`")))?;
            fixed.push(substitute(rule, target));
        }
        out.push((k.clone(), Value::Array(fixed)));
    }
    Ok(Value::Map(out))
}

/// Removes the `require` clause from every attach rule.
///
/// Only for a base with no axes. What survives is R-A02 — the shapes still have
/// to work out — which is checked from the tensors themselves rather than from
/// something the base had to declare.
fn drop_requires(adapter: &Value) -> Value {
    let Value::Map(pairs) = adapter else {
        return adapter.clone();
    };
    Value::Map(
        pairs
            .iter()
            .map(|(k, v)| {
                if k.as_str() != Some("attach") {
                    return (k.clone(), v.clone());
                }
                let Value::Array(rules) = v else {
                    return (k.clone(), v.clone());
                };
                (
                    k.clone(),
                    Value::Array(
                        rules
                            .iter()
                            .map(|r| match r {
                                Value::Map(m) => Value::Map(
                                    m.iter()
                                        .filter(|(rk, _)| rk.as_str() != Some("require"))
                                        .cloned()
                                        .collect(),
                                ),
                                other => other.clone(),
                            })
                            .collect(),
                    ),
                )
            })
            .collect(),
    )
}

/// `$T` → the target, everywhere in a value's strings.
fn substitute(v: &Value, target: &str) -> Value {
    match v {
        Value::Text(s) => Value::text(s.replace("$T", target)),
        Value::Array(a) => Value::Array(a.iter().map(|x| substitute(x, target)).collect()),
        Value::Map(m) => Value::Map(
            m.iter()
                .map(|(k, val)| (substitute(k, target), substitute(val, target)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Every object reachable from `root`, by the refs the objects carry.
fn reachable(
    objects: &[Object],
    root: &Digest,
    hash: HashAlgo,
) -> std::collections::BTreeSet<Digest> {
    let by: std::collections::BTreeMap<Digest, &Object> =
        objects.iter().map(|o| (o.digest(hash), o)).collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = vec![*root];
    while let Some(d) = stack.pop() {
        if !seen.insert(d) {
            continue;
        }
        let Some(o) = by.get(&d) else { continue };
        if o.otype == otype::BLOB {
            continue;
        }
        if let Ok(v) = crate::cbor::decode(&o.payload) {
            crate::container::collect_refs(&v, &mut stack);
        }
    }
    seen
}

fn base_table(c: &Container) -> Res<TensorTable> {
    let manifest = c.root().map_err(|e| Error::Core(e.to_string()))?;
    let model = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(|r| crate::expr::parse_ref_value(r).ok())
        .ok_or_else(|| Error::Malformed("the base has no `model` asset".into()))?;
    let m = c
        .get_value(&model.1)
        .map_err(|e| Error::Core(e.to_string()))?;
    let t = m
        .get("tensors")
        .and_then(|r| crate::expr::parse_ref_value(r).ok())
        .ok_or_else(|| Error::Malformed("the base model has no tensor table".into()))?;
    TensorTable::from_value(&c.get_value(&t.1).map_err(|e| Error::Core(e.to_string()))?)
        .map_err(|e| Error::Core(e.to_string()))
}

// ---------------------------------------------------------------------- export --

/// What a PEFT export produced.
pub struct Exported {
    pub weights: Vec<u8>,
    pub config: Vec<u8>,
    pub factors: usize,
    /// Target module suffixes recovered from the attach rules.
    pub targets: Vec<String>,
}

impl std::fmt::Debug for Exported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Exported {{ {} bytes, {} factor tensor(s), targets {:?} }}",
            self.weights.len(),
            self.factors,
            self.targets
        )
    }
}

/// Writes an adapter container back out as a PEFT LoRA.
///
/// The two files PEFT wants are `adapter_config.json` and
/// `adapter_model.safetensors`, and both are reconstructed from the `Adapter`
/// object and its tensor table rather than from anything remembered: the rank and
/// alpha are the adapter's own fields, and the target modules come from the attach
/// rules' globs. What cannot be reconstructed is named rather than guessed.
///
/// The tensor names get PEFT's `base_model.model.` prefix put back, which is the
/// exact inverse of what the import stripped.
pub fn export(
    ctx: &Ctx<'_>,
    adapter: &crate::adapter::Adapter,
    base_name: Option<&str>,
) -> Res<Exported> {
    use crate::adapter::Method as AdapterMethod;
    if adapter.method != AdapterMethod::Lora {
        return Err(Error::Unsupported(format!(
            "this adapter is `{}`; PEFT's `adapter_model.safetensors` holds \
             `lora_A`/`lora_B` factor pairs and has no form for another method",
            adapter.method.name()
        )));
    }
    let rank = adapter.rank.ok_or_else(|| {
        Error::Malformed("the adapter declares no rank, and PEFT's `r` is required".into())
    })?;
    let alpha = adapter.alpha.unwrap_or(rank as f64);

    let table = TensorTable::load(ctx, &adapter.tensors).map_err(|e| Error::Core(e.to_string()))?;
    let names: Vec<&String> = if table.order.len() == table.tensors.len() {
        table.order.iter().collect()
    } else {
        table.tensors.keys().collect()
    };
    let mut entries: Vec<(String, crate::dtype::DType, Vec<u64>, Vec<u8>)> = Vec::new();
    for name in names {
        let r = table
            .tensors
            .get(name)
            .ok_or_else(|| Error::Core(format!("`{name}` is in the order and not the table")))?;
        let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
        let Expr::Literal { chunks, dtype, .. } = &desc.value else {
            return Err(Error::Unsupported(format!(
                "`{name}` is a `{}` expression; PEFT holds materialized factors, so \
                 a derived one would have to be evaluated into bytes the source \
                 never had",
                desc.value.op()
            )));
        };
        let shape = desc
            .sizes()
            .ok_or_else(|| Error::Core(format!("`{name}` has a symbolic shape")))?;
        let bytes = ctx
            .chunk_bytes(chunks)
            .map_err(|e| Error::Core(e.to_string()))?;
        entries.push((format!("{PREFIX}{name}"), dtype.clone(), shape, bytes));
    }
    if entries.is_empty() {
        return Err(Error::Malformed(
            "the adapter's tensor table is empty; there are no factors to write".into(),
        ));
    }

    // The target modules, recovered from the attach globs. The import wrote
    // `**.<target>.weight`, so this is the inverse — and a glob that does not
    // have that shape is reported rather than turned into a plausible name.
    let mut targets: Vec<String> = Vec::new();
    for a in &adapter.attach {
        {
            let crate::adapter::Select::Glob(g) = &a.select else {
                continue;
            };
            let core = g
                .strip_prefix("**.")
                .and_then(|r| r.strip_suffix(".weight"))
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "the attach rule selects `{g}`, and PEFT's `target_modules` \
                         is a list of module suffixes — this glob is not one, so \
                         writing it as a suffix would change which modules the \
                         adapter claims"
                    ))
                })?;
            if !targets.iter().any(|t| t == core) {
                targets.push(core.to_string());
            }
        }
    }
    targets.sort();

    let weights = write_safetensors(&entries)?;
    let config = write_config(rank, alpha, adapter.dropout, &targets, adapter, base_name);
    Ok(Exported {
        weights,
        config,
        factors: entries.len(),
        targets,
    })
}

fn write_config(
    rank: u64,
    alpha: f64,
    dropout: Option<f64>,
    targets: &[String],
    adapter: &crate::adapter::Adapter,
    base_name: Option<&str>,
) -> Vec<u8> {
    use crate::json;
    let mut p = vec![
        ("peft_type", json::string("LORA")),
        ("r", json::Value::U(rank)),
        (
            "lora_alpha",
            if alpha.fract() == 0.0 && alpha >= 0.0 {
                json::Value::U(alpha as u64)
            } else {
                json::Value::F(alpha)
            },
        ),
        (
            "target_modules",
            json::Value::Array(targets.iter().map(|t| json::string(t.clone())).collect()),
        ),
        // Written explicitly rather than left out: PEFT's defaults for these are
        // what make an adapter mean what it means, and a config that omits them
        // is one a future PEFT could reinterpret.
        ("bias", json::string("none")),
        ("fan_in_fan_out", json::Value::Bool(false)),
        ("use_dora", json::Value::Bool(false)),
        ("use_rslora", json::Value::Bool(false)),
    ];
    p.push((
        "lora_dropout",
        match dropout {
            Some(d) => json::Value::F(d),
            None => json::Value::F(0.0),
        },
    ));
    // §08.1 pins the base by digest. PEFT's field is a *name*, so the digest goes
    // in as a comment-shaped key rather than being dropped: it is the guarantee
    // OMNI added, and losing it silently would undo the one thing the import
    // insisted on.
    // §08.1 pins the base by digest and PEFT's field is a *name*, so the name is
    // whatever the container's `parents[]` recorded — which is where the import
    // put it, precisely because a name is not an identity.
    p.push((
        "base_model_name_or_path",
        match base_name {
            Some(n) => json::string(n.to_string()),
            None => json::Value::Null,
        },
    ));
    p.push((
        "omni_base_digest",
        json::string(crate::sha256::hex(&adapter.base.1)),
    ));
    json::object(p).encode().into_bytes()
}

fn write_safetensors(entries: &[(String, crate::dtype::DType, Vec<u64>, Vec<u8>)]) -> Res<Vec<u8>> {
    use crate::json;
    let mut header = Vec::new();
    let mut at = 0u64;
    for (name, dtype, shape, bytes) in entries {
        let st = safetensors::name_of(dtype).ok_or_else(|| {
            Error::Unsupported(format!(
                "`{name}` is {}, which safetensors has no name for",
                dtype.label()
            ))
        })?;
        header.push((
            name.as_str(),
            json::object(vec![
                ("dtype", json::string(st)),
                (
                    "shape",
                    json::Value::Array(shape.iter().map(|d| json::Value::U(*d)).collect()),
                ),
                (
                    "data_offsets",
                    json::Value::Array(vec![
                        json::Value::U(at),
                        json::Value::U(at + bytes.len() as u64),
                    ]),
                ),
            ]),
        ));
        at += bytes.len() as u64;
    }
    let head = json::object(header).encode().into_bytes();
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    for (_, _, _, bytes) in entries {
        out.extend_from_slice(bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{pack, PackOptions};
    use crate::dtype::DType;
    use crate::model::TensorSpec;

    /// A base whose tensor names follow the convention PEFT assumes.
    fn base() -> Container {
        let mut b = ModelBuilder::new("test/base").arch("transformer.decoder", vec![]);
        for layer in 0..2 {
            for proj in ["q_proj", "v_proj", "k_proj"] {
                b = b.tensor(TensorSpec {
                    name: format!("model.layers.{layer}.self_attn.{proj}.weight"),
                    shape: vec![8, 8],
                    dtype: DType::BF16,
                    axes: Some(vec!["out".into(), "in".into()]),
                    semantic: "weight".into(),
                    data: (0..128u32).map(|i| (i % 251) as u8).collect(),
                    layout: None,
                });
            }
        }
        let (objs, root) = b.build();
        Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
    }

    fn config(extra: &str) -> Vec<u8> {
        format!(
            r#"{{"peft_type":"LORA","r":4,"lora_alpha":8,"lora_dropout":0.05,
               "base_model_name_or_path":"acme/base","task_type":"CAUSAL_LM",
               "target_modules":["q_proj","v_proj"]{extra}}}"#
        )
        .into_bytes()
    }

    /// The weights file PEFT writes: `base_model.model.<module>.lora_{A,B}.weight`.
    fn weights(r: u64, targets: &[&str]) -> Vec<u8> {
        let mut header = std::collections::BTreeMap::new();
        let mut data = Vec::new();
        let add = |name: String,
                   shape: Vec<u64>,
                   header: &mut std::collections::BTreeMap<String, json::Value>,
                   data: &mut Vec<u8>| {
            let n: u64 = shape.iter().product();
            let bytes = DType::BF16.packed_bytes(n) as usize;
            let start = data.len();
            for i in 0..bytes {
                data.push(((start + i) % 199) as u8);
            }
            header.insert(
                name,
                json::object(vec![
                    ("dtype", json::string("BF16")),
                    (
                        "shape",
                        json::Value::Array(shape.into_iter().map(json::Value::U).collect()),
                    ),
                    (
                        "data_offsets",
                        json::Value::Array(vec![
                            json::Value::U(start as u64),
                            json::Value::U((start + bytes) as u64),
                        ]),
                    ),
                ]),
            );
        };
        for layer in 0..2 {
            for t in targets {
                let m = format!("{PREFIX}model.layers.{layer}.self_attn.{t}");
                add(
                    format!("{m}.lora_A.weight"),
                    vec![r, 8],
                    &mut header,
                    &mut data,
                );
                add(
                    format!("{m}.lora_B.weight"),
                    vec![8, r],
                    &mut header,
                    &mut data,
                );
            }
        }
        let h = json::Value::Object(header).encode().into_bytes();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn a_peft_lora_imports_as_an_adapter_pinned_to_its_base() {
        let b = base();
        let imported = import(
            &config(""),
            &weights(4, &["q_proj", "v_proj"]),
            &b,
            &ImportOpts {
                config_path: "adapter_config.json".into(),
                ..Default::default()
            },
        )
        .unwrap();

        // Two layers × two targets = four factors, eight tensors, all verified.
        assert_eq!(imported.factors.len(), 4);
        assert_eq!(imported.report.verified_tensors, 8);
        assert!(imported.report.lossless);

        let c = Container::open(
            pack(&imported.objects, &imported.root, &PackOptions::default()).unwrap(),
        )
        .unwrap();
        let r = crate::container::verify(&c).unwrap();
        assert!(r.mistyped.is_empty(), "{:?}", r.mistyped);
        // The base is a declared, non-required parent: absent, and legitimately so.
        assert_eq!(r.dangling.len(), 1);
        assert_eq!(r.dangling[0], b.header.root_digest);

        // The parent has to parse back through the reader that consumes it —
        // inventing the key names is how a non-required parent gets reported as a
        // missing one.
        let parents = crate::delta::parents(&c.root().unwrap()).unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].reference.1, b.header.root_digest);
        assert!(!parents[0].required, "the base is declared optional");
        assert_eq!(parents[0].name.as_deref(), Some("acme/base"));

        // §08.1: the adapter pins its base by digest, not by the name PEFT gave.
        let ad = c.index.iter().find(|e| e.otype == otype::ADAPTER).unwrap();
        let a = crate::adapter::Adapter::from_value(&c.get_value(&ad.digest).unwrap()).unwrap();
        assert_eq!(a.base.1, b.header.root_digest);
        assert_eq!(a.rank, Some(4));
        assert_eq!(a.alpha, Some(8.0));
        assert_eq!(a.attach.len(), 2, "one rule per target module");

        // I1: the base *name* is recorded as a name, and the digest as the pin.
        let note = imported
            .report
            .assumptions
            .iter()
            .find(|n| n.item == "base")
            .unwrap();
        assert!(note.reason.contains("acme/base"), "{}", note.reason);
        assert!(note.reason.contains("not a digest"), "{}", note.reason);

        // And the report is in the container, saying it was PEFT.
        let p = c
            .index
            .iter()
            .find(|e| e.otype == otype::PROVENANCE)
            .unwrap();
        let pv = c.get_value(&p.digest).unwrap();
        assert_eq!(
            pv.get("source")
                .and_then(|s| s.get("format"))
                .and_then(|f| f.as_str()),
            Some("peft")
        );
        assert_eq!(
            pv.get("verification")
                .and_then(|v| v.get("tensors_checked"))
                .and_then(|v| v.as_u64()),
            Some(8),
            "the report in the file carries the counts, not zeros"
        );
    }

    /// §08.3: the adapter has to attach to a base it has never seen, and the
    /// bindings have to resolve to that base's own tensor names.
    #[test]
    fn the_imported_adapter_attaches_to_the_base_it_was_built_over() {
        let b = base();
        let imported = import(
            &config(""),
            &weights(4, &["q_proj", "v_proj"]),
            &b,
            &ImportOpts::default(),
        )
        .unwrap();
        let c = Container::open(
            pack(&imported.objects, &imported.root, &PackOptions::default()).unwrap(),
        )
        .unwrap();
        let ad = c.index.iter().find(|e| e.otype == otype::ADAPTER).unwrap();
        let a = crate::adapter::Adapter::from_value(&c.get_value(&ad.digest).unwrap()).unwrap();

        // §08.3's contract: one context that resolves the base's objects *and*
        // the adapter's, which is what a runtime holding both actually has.
        let bs = crate::store::Borrowed(&b);
        let cs = crate::store::Borrowed(&c);
        let layered = crate::store::Layered::new(vec![&cs, &bs]).unwrap();
        let ctx = Ctx::new(&layered);
        let btable = base_table(&b).unwrap();
        let report = a.attach(&ctx, &btable).unwrap();
        assert_eq!(report.bindings.len(), 4, "{:?}", report.unmatched);
        assert!(report.unmatched.is_empty(), "{:?}", report.unmatched);
        assert!(
            report
                .findings
                .iter()
                .all(|f| f.severity != crate::tensor::Severity::Invalid),
            "{:?}",
            report.findings
        );
        // k_proj was not a target, so it is untouched — an adapter must not update
        // what it did not train.
        assert!(!report.bindings.iter().any(|x| x.tensor.contains("k_proj")));
    }

    /// Each of these changes what the update is, so each has to be refused by
    /// name rather than approximated.
    #[test]
    fn what_this_importer_will_not_approximate_is_refused_by_name() {
        let b = base();
        let w = weights(4, &["q_proj", "v_proj"]);
        for (extra, needle) in [
            (r#","use_dora":true"#, "use_dora"),
            (r#","fan_in_fan_out":true"#, "fan_in_fan_out"),
            (r#","use_rslora":true"#, "use_rslora"),
            (r#","rank_pattern":{"q_proj":8}"#, "rank_pattern"),
            (r#","alpha_pattern":{"q_proj":16}"#, "alpha_pattern"),
            (r#","modules_to_save":["score"]"#, "modules_to_save"),
        ] {
            match import(&config(extra), &w, &b, &ImportOpts::default()) {
                Err(Error::Unsupported(m)) => assert!(m.contains(needle), "{m}"),
                other => panic!("{needle} was accepted: {other:?}"),
            }
        }
        // A different PEFT method is not a LoRA.
        let ia3 = br#"{"peft_type":"IA3","r":4,"lora_alpha":8,"target_modules":["q_proj"]}"#;
        assert!(matches!(Config::parse(ia3), Err(Error::Unsupported(_))));
        // And a regex target list is a different matcher.
        let re = br#"{"peft_type":"LORA","r":4,"lora_alpha":8,"target_modules":".*proj"}"#;
        assert!(matches!(Config::parse(re), Err(Error::Unsupported(_))));
    }

    /// A factor whose shape disagrees with the base is the failure that would
    /// otherwise produce a silently wrong model.
    #[test]
    fn factors_are_checked_against_the_base_they_claim_to_update() {
        let b = base();
        // r in the config disagrees with the tensors.
        let cfg = br#"{"peft_type":"LORA","r":8,"lora_alpha":8,"target_modules":["q_proj"]}"#;
        match import(cfg, &weights(4, &["q_proj"]), &b, &ImportOpts::default()) {
            Err(Error::Malformed(m)) => assert!(m.contains("expected [r=8"), "{m}"),
            other => panic!("a rank mismatch was accepted: {other:?}"),
        }
        // A target the base does not have.
        let cfg = br#"{"peft_type":"LORA","r":4,"lora_alpha":8,"target_modules":["gate_proj"]}"#;
        let w = {
            let mut header = std::collections::BTreeMap::new();
            let mut data = Vec::new();
            for (n, shape) in [
                (
                    format!("{PREFIX}model.layers.0.mlp.gate_proj.lora_A.weight"),
                    vec![4u64, 8],
                ),
                (
                    format!("{PREFIX}model.layers.0.mlp.gate_proj.lora_B.weight"),
                    vec![8, 4],
                ),
            ] {
                let bytes = DType::BF16.packed_bytes(shape.iter().product()) as usize;
                let start = data.len();
                data.extend(std::iter::repeat_n(0u8, bytes));
                header.insert(
                    n,
                    json::object(vec![
                        ("dtype", json::string("BF16")),
                        (
                            "shape",
                            json::Value::Array(shape.into_iter().map(json::Value::U).collect()),
                        ),
                        (
                            "data_offsets",
                            json::Value::Array(vec![
                                json::Value::U(start as u64),
                                json::Value::U((start + bytes) as u64),
                            ]),
                        ),
                    ]),
                );
            }
            let h = json::Value::Object(header).encode().into_bytes();
            let mut out = (h.len() as u64).to_le_bytes().to_vec();
            out.extend_from_slice(&h);
            out.extend_from_slice(&data);
            out
        };
        match import(cfg, &w, &b, &ImportOpts::default()) {
            Err(Error::Malformed(m)) => assert!(m.contains("not a tensor in the base"), "{m}"),
            other => panic!("a factor for a tensor the base lacks was accepted: {other:?}"),
        }
        // A lone factor: PEFT always writes pairs, and half a pair is a broken file.
        let full = weights(4, &["q_proj"]);
        let f = safetensors::File::parse(&full).unwrap();
        let drop = f
            .entries
            .iter()
            .find(|e| e.name.ends_with("lora_B.weight"))
            .unwrap();
        let mut header = std::collections::BTreeMap::new();
        let mut data = Vec::new();
        for e in f.entries.iter().filter(|e| e.name != drop.name) {
            let start = data.len();
            data.extend_from_slice(f.tensor(e));
            header.insert(
                e.name.clone(),
                json::object(vec![
                    ("dtype", json::string(e.st_dtype.clone())),
                    (
                        "shape",
                        json::Value::Array(e.shape.iter().map(|d| json::Value::U(*d)).collect()),
                    ),
                    (
                        "data_offsets",
                        json::Value::Array(vec![
                            json::Value::U(start as u64),
                            json::Value::U(data.len() as u64),
                        ]),
                    ),
                ]),
            );
        }
        let h = json::Value::Object(header).encode().into_bytes();
        let mut lone = (h.len() as u64).to_le_bytes().to_vec();
        lone.extend_from_slice(&h);
        lone.extend_from_slice(&data);
        let cfg = br#"{"peft_type":"LORA","r":4,"lora_alpha":8,"target_modules":["q_proj"]}"#;
        match import(cfg, &lone, &b, &ImportOpts::default()) {
            Err(Error::Malformed(m)) => assert!(m.contains("only one of its two"), "{m}"),
            other => panic!("half a factor pair was accepted: {other:?}"),
        }
    }

    /// A tensor in the file that is not a factor for a declared target is reported
    /// rather than dropped in silence — the difference between lossless and not.
    #[test]
    fn a_tensor_nobody_asked_for_makes_the_import_lossy_and_says_so() {
        let b = base();
        let full = weights(4, &["q_proj", "v_proj", "k_proj"]);
        // The config names two targets; the file carries three.
        let imported = import(&config(""), &full, &b, &ImportOpts::default()).unwrap();
        assert!(!imported.report.lossless);
        assert_eq!(imported.factors.len(), 4);
        assert!(imported
            .report
            .unrepresented
            .iter()
            .any(|n| n.item.contains("k_proj")));
    }

    #[test]
    fn the_naming_convention_is_read_rather_than_assumed() {
        let t: Vec<String> = ["q_proj", "v_proj"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            base_tensor_of(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                &t
            ),
            Some((
                "model.layers.0.self_attn.q_proj.weight".into(),
                "q_proj".into()
            ))
        );
        // Not a factor, not a target, not PEFT's prefix: each is a miss rather
        // than a guess.
        assert!(base_tensor_of("model.layers.0.self_attn.q_proj.lora_A.weight", &t).is_none());
        assert!(base_tensor_of("base_model.model.x.o_proj.lora_A.weight", &t).is_none());
        assert!(base_tensor_of("base_model.model.x.q_proj.weight", &t).is_none());
        // A target that is the whole module path, which happens for a flat model.
        assert_eq!(
            base_tensor_of("base_model.model.q_proj.lora_B.weight", &t),
            Some(("q_proj.weight".into(), "q_proj".into()))
        );
    }

    #[test]
    fn a_peft_adapter_round_trips_back_to_peft() {
        // Import a LoRA over a base, export it, and check the factors come back
        // byte for byte under the names PEFT uses — prefix and all.
        let base = base();
        let (cfg, weights) = (config(""), weights(4, &["q_proj", "v_proj"]));
        let im = import(&cfg, &weights, &base, &ImportOpts::default()).unwrap();
        let hash = base.header.hash;
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &im.objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let av = im
            .objects
            .iter()
            .find(|o| o.otype == otype::ADAPTER)
            .map(|o| crate::cbor::decode(&o.payload).unwrap())
            .expect("an adapter object");
        let adapter = crate::adapter::Adapter::from_value(&av).unwrap();

        let ex = export(&ctx, &adapter, Some("acme/base")).unwrap();
        let src = safetensors::File::parse(&weights).unwrap();
        let back = safetensors::File::parse(&ex.weights).unwrap();
        assert_eq!(
            back.entries.len(),
            src.entries.len(),
            "factor count changed"
        );
        for e in &src.entries {
            let got = back
                .get(&e.name)
                .unwrap_or_else(|| panic!("`{}` did not come back", e.name));
            assert_eq!(got.shape, e.shape, "{}", e.name);
            assert_eq!(back.tensor(got), src.tensor(e), "`{}` differs", e.name);
        }

        // The config is rebuilt from the adapter object, not remembered.
        let c = crate::json::parse(&ex.config).unwrap();
        assert_eq!(c.get("peft_type").unwrap().as_str(), Some("LORA"));
        assert_eq!(c.get("r").unwrap().as_u64(), Some(4));
        let targets: Vec<&str> = c
            .get("target_modules")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(targets, vec!["q_proj", "v_proj"]);
        // The name PEFT gave the base is a name; the digest OMNI pinned it with
        // survives beside it rather than being dropped on the way out.
        assert_eq!(
            c.get("base_model_name_or_path").unwrap().as_str(),
            Some("acme/base")
        );
        assert_eq!(
            c.get("omni_base_digest").unwrap().as_str(),
            Some(crate::sha256::hex(&base.header.root_digest).as_str())
        );
        // And the fields whose defaults define what a LoRA is are written out
        // rather than left for a future PEFT to reinterpret.
        for k in ["bias", "fan_in_fan_out", "use_dora", "use_rslora"] {
            assert!(c.get(k).is_some(), "`{k}` should be explicit");
        }
    }
}
