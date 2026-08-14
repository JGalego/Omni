//! GPTQ and AWQ import (§05.2.2, §05.2.3; `docs/design/import-export.md` §3).
//!
//! Both formats are the same shape of thing: a `safetensors` file whose linear
//! weights have been replaced by four tensors — `qweight`, `qzeros`, `scales`,
//! and (GPTQ only) `g_idx` — plus a JSON config saying how to read them. §05's
//! claim is that this needs no new mechanism, because a quantized weight is not
//! a file type but an expression:
//!
//! ```text
//! weight = permute(dequantize(reshape(permute(qweight)), {affine-sub, …}))
//! ```
//!
//! so that is what the importer writes. Nothing here is a special case in the
//! evaluator: the int4-in-int32 packing is [`Layout::Packed`], AWQ's GEMM
//! interleave is a `gather`, GPTQ's act-order is a `gather` too, and the
//! arithmetic is one `dequantize` node whose `formula` is drawn from §05.1's
//! closed set. The bytes go in unchanged and are read back out by the algebra.
//!
//! ## Why the formula matters
//!
//! §05.1 gives a specific reason for making `formula` a closed enumeration:
//! whether the zero point is subtracted before or after scaling is *a recurring
//! source of silent corruption when converting between GPTQ, AWQ and GGUF*.
//! This importer is where that bites. AutoGPTQ's original checkpoint format
//! stores each zero point **one less** than its true value and adds one back in
//! `QuantLinear.forward`; `checkpoint_format: "gptq_v2"` dropped that. The two
//! conventions differ by exactly one quantization step in every weight, and
//! nothing in the tensors distinguishes them. So the offset is read from the
//! config, applied as an explicit `+1` node in the expression rather than folded
//! into a constant, and named in the fidelity report. A reader can see which
//! convention was assumed instead of having to diff dequantized weights.
//!
//! ## What is refused rather than approximated
//!
//! 3-bit GPTQ, whose values straddle 32-bit word boundaries and are therefore
//! not a [`Layout::Packed`] at all; AWQ's `gemv` and `marlin` versions, which
//! interleave differently from `gemm`; any `checkpoint_format` this build does
//! not know, because an unknown format is exactly the case where guessing the
//! zero-point convention would corrupt every weight. Each is refused by name.
//!
//! ## What is verified
//!
//! Two independent checks, both reported (I4):
//!
//! 1. **Byte identity.** Every source tensor's bytes are read back out of the
//!    object graph and compared. The packed tensors are stored verbatim inside
//!    the expression, so this is a real comparison and not a tautology.
//! 2. **Sample dequantization.** Each quantized weight is *evaluated* through
//!    the expression graph and compared against a dequantization computed here
//!    by direct scalar code that shares nothing with the evaluator. This is the
//!    check that catches a wrong interleave, a transposed axis, or a zero-point
//!    convention applied backwards — the failures byte identity cannot see.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::{DType, Round};
use crate::expr::{dims, BinOp, Ctx, Expr, Scalar};
use crate::json;
use crate::layout::{BitOrder, Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{self, Fidelity, Note};
use crate::tensor::{Materialize, TensorDesc, TensorTable};

/// AutoAWQ's `REVERSE_AWQ_PACK_ORDER`: the slot each output column comes from,
/// within one 32-bit word of eight 4-bit values.
///
/// AWQ's GEMM kernel wants the columns of a word in the order `0 2 4 6 1 3 5 7`
/// so that a warp reads two halves at once. Unpacking sequentially therefore
/// yields the columns permuted, and this is the permutation that undoes it — as
/// a `gather`, which is what §05.2.2 says a stored permutation is.
pub const AWQ_REVERSE_ORDER: [u64; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

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
            Error::Malformed(m) => write!(f, "malformed checkpoint: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// --------------------------------------------------------------------- method --

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Gptq,
    Awq,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Method::Gptq => "gptq",
            Method::Awq => "awq",
        }
    }

    pub fn importer(self) -> &'static str {
        match self {
            Method::Gptq => "omni-import-gptq",
            Method::Awq => "omni-import-awq",
        }
    }

    pub fn parse(s: &str) -> Option<Method> {
        Some(match s {
            "gptq" => Method::Gptq,
            "awq" => Method::Awq,
            _ => return None,
        })
    }
}

// --------------------------------------------------------------------- config --

/// The quantization config, from either `quantize_config.json` or the
/// `quantization_config` member of a Hugging Face `config.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub method: Method,
    pub bits: u32,
    /// `-1` means one group per column, which is `group_size = in_features`.
    pub group_size: i64,
    /// GPTQ act-order. Advisory: what `g_idx` actually contains decides.
    pub desc_act: bool,
    /// GPTQ symmetric quantization. The checkpoint still carries `qzeros`, so
    /// this is metadata rather than a change of formula.
    pub sym: bool,
    /// AWQ: whether zero points were used at all.
    pub zero_point: bool,
    /// AWQ kernel version.
    pub version: Option<String>,
    pub checkpoint_format: Option<String>,
    /// Whether the stored zero points are the true ones. False for AutoGPTQ's
    /// original format, which stores them one lower.
    pub zeros_verbatim: bool,
    pub model_name: Option<String>,
}

impl Config {
    /// Reads a config, checking it against the method the caller asked for.
    ///
    /// `want` is authoritative because it comes from the command line, but a
    /// config that names a *different* method is a mistake worth reporting
    /// rather than overriding: the two formats' `qweight` differ by a transpose,
    /// so reading one as the other produces a plausible-looking wrong answer.
    pub fn parse(bytes: &[u8], want: Method) -> Res<Config> {
        let top = json::parse(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        let v = match top.get("quantization_config") {
            Some(inner) => inner,
            None => &top,
        };
        if let Some(m) = v.get("quant_method").and_then(|x| x.as_str()) {
            match Method::parse(m) {
                Some(got) if got == want => {}
                _ => {
                    return Err(Error::Malformed(format!(
                        "the config says `quant_method` is `{m}` but this is the {} \
                         importer; the two formats' `qweight` differ by a transpose, so \
                         reading one as the other would not fail, it would be wrong",
                        want.name()
                    )))
                }
            }
        }
        // AutoAWQ's older configs spell these `w_bit` and `q_group_size`.
        let num = |a: &str, b: &str| -> Option<i64> {
            v.get(a).or_else(|| v.get(b)).and_then(|x| x.as_i64())
        };
        let bits = num("bits", "w_bit")
            .ok_or_else(|| Error::Malformed("no `bits`".into()))?
            .try_into()
            .map_err(|_| Error::Malformed("`bits` is negative".into()))?;
        let group_size = num("group_size", "q_group_size")
            .ok_or_else(|| Error::Malformed("no `group_size`".into()))?;
        if group_size == 0 || group_size < -1 {
            return Err(Error::Malformed(format!("`group_size` is {group_size}")));
        }
        let flag = |k: &str, default: bool| v.get(k).and_then(|x| x.as_bool()).unwrap_or(default);
        let checkpoint_format = v
            .get("checkpoint_format")
            .and_then(|x| x.as_str())
            .map(str::to_string);
        let version = v
            .get("version")
            .and_then(|x| x.as_str())
            .map(str::to_string);

        // The zero-point convention, which is the one thing that must not be
        // guessed. See the module documentation.
        let zeros_verbatim = match (want, checkpoint_format.as_deref()) {
            (Method::Awq, _) => true,
            (Method::Gptq, None | Some("gptq")) => false,
            (Method::Gptq, Some("gptq_v2")) => true,
            (Method::Gptq, Some(other)) => {
                return Err(Error::Unsupported(format!(
                    "`checkpoint_format` is `{other}`; this build knows `gptq` and \
                     `gptq_v2`, which differ in whether the stored zero point is the \
                     true one. Guessing would shift every weight by one quantization \
                     step, so an unknown format is refused instead"
                )))
            }
        };

        match (want, bits) {
            (Method::Gptq, 2 | 4 | 8) => {}
            (Method::Gptq, 3) => {
                return Err(Error::Unsupported(
                    "3-bit GPTQ packs ten and two-thirds values per 32-bit word, so a \
                     value straddles the word boundary; that is not a `packed` layout \
                     (§04.4) and this importer will not pretend it is"
                        .into(),
                ))
            }
            (Method::Gptq, b) => {
                return Err(Error::Unsupported(format!(
                    "{b}-bit GPTQ: this build reads 2, 4 and 8 bits, the widths that \
                     divide a 32-bit word"
                )))
            }
            (Method::Awq, 4) => {}
            (Method::Awq, b) => {
                return Err(Error::Unsupported(format!(
                    "{b}-bit AWQ: the GEMM interleave is defined for eight 4-bit values \
                     per word, and this build does not invent one for other widths"
                )))
            }
        }
        if want == Method::Awq {
            match version.as_deref() {
                None | Some("gemm") => {}
                Some(other) => {
                    return Err(Error::Unsupported(format!(
                        "AWQ `version` is `{other}`; `gemv` and `marlin` interleave \
                         their words differently from `gemm`, and reading one as the \
                         other permutes the output columns"
                    )))
                }
            }
        }

        Ok(Config {
            method: want,
            bits,
            group_size,
            desc_act: flag("desc_act", false),
            sym: flag("sym", true),
            zero_point: flag("zero_point", true),
            version,
            checkpoint_format,
            zeros_verbatim,
            model_name: top
                .get("_name_or_path")
                .and_then(|x| x.as_str())
                .map(str::to_string),
        })
    }

    /// Values per 32-bit word.
    pub fn elems_per_word(&self) -> u64 {
        32 / self.bits as u64
    }

    /// The dtype of one unpacked value.
    pub fn qdtype(&self) -> DType {
        DType::Int {
            w: self.bits as u16,
            signed: false,
        }
    }
}

// ---------------------------------------------------------------------- layers --

/// One quantized linear layer, as it was read.
#[derive(Clone, Debug)]
pub struct Layer {
    /// The tensor name prefix, e.g. `model.layers.0.self_attn.q_proj`.
    pub prefix: String,
    pub in_features: u64,
    pub out_features: u64,
    pub group_size: u64,
    pub groups: u64,
    /// Whether `g_idx` was used as a gather because it is not the ascending
    /// grouping. Independent of what `desc_act` claims.
    pub act_order: bool,
    pub has_zeros: bool,
    /// Elements compared against an independent dequantization.
    pub checked: u64,
}

/// The four tensors a quantized layer is made of.
struct Parts<'a> {
    qweight: &'a safetensors::Entry,
    qzeros: Option<&'a safetensors::Entry>,
    scales: &'a safetensors::Entry,
    g_idx: Option<&'a safetensors::Entry>,
}

/// The suffixes that belong to a quantized layer rather than to the model.
///
/// `bias` is deliberately absent: GPTQ writes one next to `qweight`, but it is an
/// ordinary dense tensor and is imported as one.
const PARTS: [&str; 4] = ["qweight", "qzeros", "scales", "g_idx"];

pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
    pub layers: Vec<Layer>,
    /// Tensors imported unchanged: embeddings, norms, biases, the head.
    pub plain: usize,
}

impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Imported {{ {} objects, root {}, {} quantized layer(s), {} plain, \
             lossless {} }}",
            self.objects.len(),
            crate::sha256::hex(&self.root[..6]),
            self.layers.len(),
            self.plain,
            self.report.lossless
        )
    }
}

#[derive(Clone, Debug)]
pub struct ImportOpts {
    pub name: String,
    pub config_path: String,
    pub weights_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    pub arch: Option<String>,
    /// The largest weight to check against an independent dequantization, in
    /// elements. Evaluating the expression materializes the whole tensor, so the
    /// check is bounded — and what it skipped is reported rather than implied.
    pub max_verify_elems: u64,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/gptq".into(),
            config_path: String::new(),
            weights_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
            max_verify_elems: 1 << 22,
        }
    }
}

// ---------------------------------------------------------------------- import --

/// Imports a GPTQ or AWQ checkpoint into an OMNI object graph.
///
/// `method` comes from the caller rather than from the config, because the caller
/// asked for a named format and being told "that is not what this is" is more
/// useful than being handed the other one.
pub fn import(
    config_bytes: &[u8],
    weights: &[u8],
    method: Method,
    opts: &ImportOpts,
) -> Res<Imported> {
    let cfg = Config::parse(config_bytes, method)?;
    let f = safetensors::File::parse(weights).map_err(|e| Error::Malformed(e.to_string()))?;
    let hash = opts.hash;

    let mut report = Fidelity {
        format: match method {
            Method::Gptq => "gptq",
            Method::Awq => "awq",
        },
        importer: method.importer(),
        source_path: opts.config_path.clone(),
        // Both halves, hashed together in a fixed order, so "which files did
        // this come from?" has one answer (I6).
        source_digest: hash.digest(&[config_bytes, weights].concat()),
        source_size: (config_bytes.len() + weights.len()) as u64,
        lossless: true,
        represented: vec![
            "tensors".into(),
            "dtypes".into(),
            "shapes".into(),
            "quantization".into(),
        ],
        verify_method: "byte-identity + sample-dequant",
        ..Default::default()
    };

    // Which prefixes are quantized layers. A prefix with a `qweight` and no
    // `scales` is malformed rather than plain: something replaced the weight and
    // did not say how to read it back.
    let mut quantized: std::collections::BTreeMap<String, Parts<'_>> = Default::default();
    let mut incomplete = Vec::new();
    for e in &f.entries {
        let Some(prefix) = e.name.strip_suffix(".qweight") else {
            continue;
        };
        let get = |suffix: &str| f.get(&format!("{prefix}.{suffix}"));
        let Some(scales) = get("scales") else {
            incomplete.push(prefix.to_string());
            continue;
        };
        quantized.insert(
            prefix.to_string(),
            Parts {
                qweight: e,
                qzeros: get("qzeros"),
                scales,
                g_idx: get("g_idx"),
            },
        );
    }
    if let Some(p) = incomplete.first() {
        return Err(Error::Malformed(format!(
            "`{p}.qweight` has no `{p}.scales`: the weight was replaced by \
             something this file does not say how to read"
        )));
    }
    if quantized.is_empty() {
        return Err(Error::Malformed(format!(
            "no `qweight` among {} tensor(s); this is not a {} checkpoint",
            f.entries.len(),
            method.name()
        )));
    }

    let mut b = ModelBuilder::new(opts.name.clone())
        .hash(hash)
        .chunk_size(opts.chunk_size);
    if let Some(spdx) = &opts.license {
        b = b.license(spdx.clone());
    }
    if let Some(family) = &opts.arch {
        b = b.arch(family.clone(), Vec::new());
    }

    // The tensors no layer claims, imported exactly as the safetensors importer
    // would: in file order, dense, with nothing invented about what they mean.
    let claimed = |name: &str| -> bool {
        PARTS.iter().any(|s| {
            name.strip_suffix(&format!(".{s}"))
                .is_some_and(|p| quantized.contains_key(p))
        })
    };
    let mut plain = 0usize;
    // Which source bytes must be found again in the graph, and where.
    let mut byte_checks: Vec<(&safetensors::Entry, ByteWhere)> = Vec::new();
    for e in &f.entries {
        if claimed(&e.name) {
            continue;
        }
        b = b.tensor(TensorSpec {
            name: e.name.clone(),
            shape: e.shape.clone(),
            dtype: e.dtype.clone(),
            axes: None,
            semantic: "".into(),
            data: f.tensor(e).to_vec(),
            layout: Some(safetensors::layout_of(&e.dtype)),
        });
        plain += 1;
        byte_checks.push((e, ByteWhere::Tensor));
    }

    // AWQ's de-interleave index, built once: the same eight values for every
    // layer, so the chunk is stored once and the expression is shared.
    let reverse_order = (method == Method::Awq).then(|| {
        let bytes: Vec<u8> = AWQ_REVERSE_ORDER
            .iter()
            .flat_map(|i| (*i as u32).to_le_bytes())
            .collect();
        b.literal(
            &bytes,
            DType::U32,
            &[AWQ_REVERSE_ORDER.len() as u64],
            Layout::default(),
        )
    });

    let mut layers = Vec::new();
    for (prefix, parts) in &quantized {
        let shapes = Shapes::of(&cfg, prefix, parts)?;
        // g_idx decides, not `desc_act`: a config that says one thing and a
        // tensor that says another is a disagreement worth reporting, and the
        // tensor is the one the weights were quantized with.
        let g_idx = match parts.g_idx {
            Some(e) => Some(read_g_idx(&f, e, &shapes, prefix)?),
            None => None,
        };
        let act_order = g_idx.as_ref().is_some_and(|g| !is_ascending(g, &shapes));
        if act_order != cfg.desc_act {
            report.warnings.push(format!(
                "{prefix}: `desc_act` is {} but `g_idx` {}; the tensor decides",
                cfg.desc_act,
                if act_order {
                    "is a non-ascending grouping"
                } else {
                    "is the plain ascending grouping"
                }
            ));
        }

        // ---- the packed values ----------------------------------------------
        let qw_bytes = f.tensor(parts.qweight);
        let (qw_lit, qw_ref) = literal(&mut b, &cfg, qw_bytes, &shapes.qweight_words);
        let qweight = match method {
            // GPTQ words run down the input axis: `[in/epw, out, epw]` puts the
            // slot last, where the packing wants it, so undoing it is a
            // transpose of the last two axes and a merge of the first two.
            Method::Gptq => Expr::Reshape {
                x: Box::new(Expr::Permute {
                    x: Box::new(qw_lit),
                    perm: vec![0, 2, 1],
                }),
                shape: dims(&[shapes.in_features, shapes.out_features]),
            },
            // AWQ words run along the output axis, so the slot is already last;
            // what needs undoing is the GEMM interleave within the word.
            Method::Awq => Expr::Reshape {
                x: Box::new(Expr::Gather {
                    x: Box::new(qw_lit),
                    idx: Box::new(reverse_order.clone().expect("built for AWQ")),
                    axis: 2,
                }),
                shape: dims(&[shapes.in_features, shapes.out_features]),
            },
        };
        byte_checks.push((parts.qweight, ByteWhere::Chunks(qw_ref)));

        // ---- the scales -----------------------------------------------------
        let sc_bytes = f.tensor(parts.scales);
        let sc_dtype = parts.scales.dtype.clone();
        let scales_ref = b.chunk_list(sc_bytes);
        let scales_lit = Expr::Literal {
            chunks: scales_ref,
            dtype: sc_dtype.clone(),
            shape: dims(&[shapes.groups, shapes.out_features]),
            layout: safetensors::layout_of(&sc_dtype),
        };
        byte_checks.push((parts.scales, ByteWhere::Chunks(scales_ref)));

        // ---- the zero points ------------------------------------------------
        let zeros = match parts.qzeros {
            Some(e) => {
                let (lit, r) = literal(&mut b, &cfg, f.tensor(e), &shapes.qzeros_words);
                byte_checks.push((e, ByteWhere::Chunks(r)));
                // Both formats pack zero points along the output axis, so the
                // slot is last in both and only AWQ's interleave differs.
                let unpacked = match method {
                    Method::Gptq => lit,
                    Method::Awq => Expr::Gather {
                        x: Box::new(lit),
                        idx: Box::new(reverse_order.clone().expect("built for AWQ")),
                        axis: 2,
                    },
                };
                let flat = Expr::Reshape {
                    x: Box::new(unpacked),
                    shape: dims(&[shapes.groups, shapes.out_features]),
                };
                Some(if cfg.zeros_verbatim {
                    flat
                } else {
                    // The `+1` of AutoGPTQ's original format, written as a node
                    // so that it is visible in the container rather than folded
                    // into a number nobody can question.
                    Expr::Bin {
                        op: BinOp::Add,
                        a: Box::new(flat),
                        b: Box::new(Expr::Full {
                            value: Scalar::Int(1),
                            dtype: DType::U8,
                            shape: dims(&[1, 1]),
                        }),
                    }
                })
            }
            None => None,
        };
        if zeros.is_none() && cfg.method == Method::Awq && cfg.zero_point {
            return Err(Error::Malformed(format!(
                "{prefix}: the config says `zero_point` is true but there is no \
                 `qzeros`"
            )));
        }

        // ---- the scheme -----------------------------------------------------
        // Act-order is a per-row group index rather than a uniform grouping, so
        // the scale and zero tensors are gathered by it and the block becomes
        // per-element. Both forms are the same arithmetic; the compact one keeps
        // the grouping visible, and is what §05.2.2 writes.
        let (block, scale_expr, zero_expr) = if act_order {
            let g = g_idx.as_ref().expect("act_order implies g_idx");
            let idx_ref = b.chunk_list(&g.bytes);
            byte_checks.push((
                parts.g_idx.expect("act_order implies g_idx"),
                ByteWhere::Chunks(idx_ref),
            ));
            let idx = Expr::Literal {
                chunks: idx_ref,
                dtype: g.dtype.clone(),
                shape: dims(&[shapes.in_features]),
                layout: safetensors::layout_of(&g.dtype),
            };
            let gather = |x: Expr| Expr::Gather {
                x: Box::new(x),
                idx: Box::new(idx.clone()),
                axis: 0,
            };
            (vec![1u64, 1], gather(scales_lit), zeros.map(gather))
        } else {
            (vec![shapes.group_size, 1], scales_lit, zeros)
        };

        let sym = zero_expr.is_none();
        let mut scheme = vec![
            ("scheme", Value::text(if sym { "sym" } else { "affine" })),
            (
                "formula",
                Value::text(if sym { "sym" } else { "affine-sub" }),
            ),
            ("out", sc_dtype.to_value()),
            ("axis", Value::U(0)),
            (
                "block",
                Value::Array(block.iter().map(|d| Value::U(*d)).collect()),
            ),
            ("scale", scale_expr.to_value()),
        ];
        if let Some(z) = &zero_expr {
            scheme.push(("zero", z.to_value()));
        } else {
            scheme.push(("sym", Value::Bool(true)));
        }
        let deq = Expr::Dequantize {
            x: Box::new(qweight),
            scheme: Value::map(scheme),
        };
        // Both formats store the transpose of the layer's weight matrix, so the
        // imported tensor is transposed back and named `<layer>.weight` — the
        // name the unquantized tensors in the same file already use.
        let value = Expr::Permute {
            x: Box::new(deq),
            perm: vec![1, 0],
        };
        b = b.derived(
            format!("{prefix}.weight"),
            TensorDesc {
                shape: dims(&[shapes.out_features, shapes.in_features]),
                dtype: sc_dtype.clone(),
                layout: Layout::default(),
                value,
                semantic: Some("weight".into()),
                role: Some("quantized".into()),
                axes: Some(vec!["out".into(), "in".into()]),
                device_hint: None,
                materialize: Materialize::Lazy,
                stats: None,
                digest_materialized: None,
            },
        );
        layers.push(Layer {
            prefix: prefix.clone(),
            in_features: shapes.in_features,
            out_features: shapes.out_features,
            group_size: shapes.group_size,
            groups: shapes.groups,
            act_order,
            has_zeros: parts.qzeros.is_some(),
            checked: 0,
        });
    }

    // I1: what the file does not state stays absent.
    report.assumptions.push(Note {
        item: "zero point".into(),
        reason: match (method, cfg.checkpoint_format.as_deref()) {
            (Method::Awq, _) => "AWQ stores zero points verbatim".into(),
            (Method::Gptq, Some(fmt)) => {
                format!("`checkpoint_format` is `{fmt}`")
            }
            (Method::Gptq, None) => {
                "no `checkpoint_format`, which AutoGPTQ writes for its original \
                 format"
                    .into()
            }
        },
        action: if cfg.zeros_verbatim {
            "used as stored".into()
        } else {
            "one added, as AutoGPTQ's QuantLinear does; written as an explicit \
             `+1` node so the convention is visible (§05.1)"
                .into()
        },
    });
    report.assumptions.push(Note {
        item: "arch.family".into(),
        reason: "the quantization config does not name an architecture".into(),
        action: match &opts.arch {
            Some(family) => format!("supplied by the caller as `{family}`"),
            None => "field omitted".into(),
        },
    });
    report.assumptions.push(Note {
        item: "license".into(),
        reason: "the config declares none".into(),
        action: match &opts.license {
            Some(spdx) => format!("supplied by the caller as `{spdx}`"),
            None => "field omitted".into(),
        },
    });
    if cfg.method == Method::Gptq && cfg.sym {
        report.assumptions.push(Note {
            item: "sym".into(),
            reason: "GPTQ writes `qzeros` whether or not it quantized symmetrically".into(),
            action: "the stored zero points are used; `sym` is recorded as metadata".into(),
        });
    }
    for l in &layers {
        if !l.act_order && quantized[&l.prefix].g_idx.is_some() {
            report.assumptions.push(Note {
                item: format!("{}.g_idx", l.prefix),
                reason: "the ascending grouping, which `group_size` already states".into(),
                action: format!(
                    "checked to equal i/{} for all {} rows; the compact block form \
                     is used instead of a gather",
                    l.group_size, l.in_features
                ),
            });
            break;
        }
    }
    report
        .represented
        .push(format!("{}-bit {}", cfg.bits, method.name()));

    // I4, both checks, before the report is written: the counts belong in the
    // object that goes into the container.
    let (probe, _) = b.build();
    let mut mem = crate::store::MemoryStore::new(hash);
    for o in &probe {
        let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
    }
    let ctx = Ctx::new(&mem);

    // 1. Byte identity, for every source tensor.
    for (e, where_) in &byte_checks {
        let got = match where_ {
            ByteWhere::Chunks(r) => ctx
                .chunk_bytes(r)
                .map_err(|err| Error::Core(err.to_string()))?,
            ByteWhere::Tensor => continue,
        };
        if got != f.tensor(e) {
            return Err(Error::Core(format!(
                "I4: `{}` did not survive import byte for byte",
                e.name
            )));
        }
        report.verified_tensors += 1;
        report.verified_bytes += e.len();
    }
    let tensors_ref = probe
        .iter()
        .find(|o| o.otype == otype::TENSOR_TABLE)
        .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
        .ok_or_else(|| Error::Core("the builder produced no tensor table".into()))?;
    let table = TensorTable::load(&ctx, &tensors_ref).map_err(|e| Error::Core(e.to_string()))?;
    for (e, where_) in &byte_checks {
        if !matches!(where_, ByteWhere::Tensor) {
            continue;
        }
        let r = table
            .get(&e.name)
            .ok_or_else(|| Error::Core(format!("`{}` did not reach the table", e.name)))?;
        let d = TensorDesc::load(&ctx, r).map_err(|err| Error::Core(err.to_string()))?;
        let Expr::Literal { chunks, .. } = &d.value else {
            return Err(Error::Core(format!("`{}` is not a literal", e.name)));
        };
        let got = ctx
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

    // 2. Sample dequantization: the expression graph against scalar code that
    // shares nothing with it.
    let mut skipped = Vec::new();
    for l in &mut layers {
        let n = l.in_features * l.out_features;
        if n > opts.max_verify_elems {
            skipped.push(l.prefix.clone());
            continue;
        }
        let parts = &quantized[&l.prefix];
        let want = reference_weight(&cfg, &f, parts, l)?;
        let r = table
            .get(&format!("{}.weight", l.prefix))
            .ok_or_else(|| Error::Core(format!("`{}.weight` is not in the table", l.prefix)))?;
        let d = TensorDesc::load(&ctx, r).map_err(|err| Error::Core(err.to_string()))?;
        let got = d
            .value
            .eval(&ctx)
            .map_err(|err| Error::Core(format!("{}.weight: {err}", l.prefix)))?;
        if got.data.len() != want.len() {
            return Err(Error::Core(format!(
                "I4: `{}.weight` has {} elements, the source dequantizes to {}",
                l.prefix,
                got.data.len(),
                want.len()
            )));
        }
        for (i, (a, w)) in got.data.iter().zip(&want).enumerate() {
            if a != w && !(a.is_nan() && w.is_nan()) {
                return Err(Error::Core(format!(
                    "I4: `{}.weight`[{i}] dequantizes to {a} through the expression \
                     and {w} from the source bytes; the layout or the zero-point \
                     convention is wrong",
                    l.prefix
                )));
            }
        }
        l.checked = n;
        report.dequant_checked += n;
    }
    for p in &skipped {
        let l = layers
            .iter()
            .find(|l| &l.prefix == p)
            .expect("skipped layers come from `layers`");
        report.warnings.push(format!(
            "{p}: too large to dequantize whole ({} elements > {}); its bytes are \
             still verified",
            l.in_features * l.out_features,
            opts.max_verify_elems
        ));
    }

    // Rebuild with the finished report attached, so the object in the container
    // and the value the caller is handed say the same thing. I3: the report goes
    // into the container, not just into the return value.
    let (objects, root) = b
        .asset("provenance", otype::PROVENANCE, report.to_value())
        .build();
    Ok(Imported {
        objects,
        root,
        report,
        layers,
        plain,
    })
}

/// Where a source tensor's bytes ended up.
enum ByteWhere {
    /// As a `TensorSpec`, reachable by name through the table.
    Tensor,
    /// As a `ChunkList` inside a derived expression.
    Chunks(crate::expr::Ref),
}

// ---------------------------------------------------------------------- shapes --

/// The shapes a layer's four tensors must have, derived from `qweight` and
/// checked against the rest. Nothing here is assumed: every extent is either
/// read or cross-checked, because a transposed `qweight` read as the other
/// format's would produce a plausible wrong answer rather than an error.
struct Shapes {
    in_features: u64,
    out_features: u64,
    group_size: u64,
    groups: u64,
    /// The `[.., .., epw]` shape the packed `qweight` is declared with.
    qweight_words: Vec<u64>,
    qzeros_words: Vec<u64>,
}

impl Shapes {
    fn of(cfg: &Config, prefix: &str, parts: &Parts<'_>) -> Res<Shapes> {
        let epw = cfg.elems_per_word();
        let bad = |m: String| Error::Malformed(format!("{prefix}: {m}"));
        for (what, e) in [("qweight", Some(parts.qweight)), ("qzeros", parts.qzeros)] {
            let Some(e) = e else { continue };
            if e.dtype.bits() != 32 || !matches!(e.dtype, DType::Int { .. }) {
                return Err(bad(format!(
                    "{what} is {}, and a packed word is a 32-bit integer",
                    e.dtype.label()
                )));
            }
            if e.shape.len() != 2 {
                return Err(bad(format!("{what} is {}-dimensional", e.shape.len())));
            }
        }
        if parts.scales.shape.len() != 2 {
            return Err(bad(format!(
                "scales is {}-dimensional",
                parts.scales.shape.len()
            )));
        }
        let qw = &parts.qweight.shape;
        let (in_features, out_features) = match cfg.method {
            Method::Gptq => (qw[0] * epw, qw[1]),
            Method::Awq => (qw[0], qw[1] * epw),
        };
        let group_size = if cfg.group_size < 0 {
            in_features
        } else {
            cfg.group_size as u64
        };
        if group_size == 0 || in_features % group_size != 0 {
            return Err(bad(format!(
                "{in_features} input features do not divide into groups of {group_size}"
            )));
        }
        let groups = in_features / group_size;
        if out_features % epw != 0 {
            return Err(bad(format!(
                "{out_features} output features do not divide into words of {epw}"
            )));
        }
        if parts.scales.shape != vec![groups, out_features] {
            return Err(bad(format!(
                "scales is {:?}, expected [groups={groups}, out={out_features}]",
                parts.scales.shape
            )));
        }
        if let Some(z) = parts.qzeros {
            if z.shape != vec![groups, out_features / epw] {
                return Err(bad(format!(
                    "qzeros is {:?}, expected [groups={groups}, out/{epw}={}]",
                    z.shape,
                    out_features / epw
                )));
            }
        }
        Ok(Shapes {
            in_features,
            out_features,
            group_size,
            groups,
            qweight_words: match cfg.method {
                Method::Gptq => vec![in_features / epw, out_features, epw],
                Method::Awq => vec![in_features, out_features / epw, epw],
            },
            qzeros_words: vec![groups, out_features / epw, epw],
        })
    }
}

/// A packed literal: `[.., .., epw]` so the slot within the word is the fastest
/// axis, which is what §04.4's `packed` layout places.
fn literal(
    b: &mut ModelBuilder,
    cfg: &Config,
    bytes: &[u8],
    shape: &[u64],
) -> (Expr, crate::expr::Ref) {
    let layout = Layout::Packed {
        elems_per_word: cfg.elems_per_word() as u32,
        word_bits: 32,
        bit_order: BitOrder::LsbFirst,
        order: Order::RowMajor,
    };
    let r = b.chunk_list(bytes);
    (
        Expr::Literal {
            chunks: r,
            dtype: cfg.qdtype(),
            shape: dims(shape),
            layout,
        },
        r,
    )
}

/// A layer's `g_idx`, read as group numbers.
struct GIdx {
    values: Vec<u64>,
    bytes: Vec<u8>,
    dtype: DType,
}

fn read_g_idx(
    f: &safetensors::File<'_>,
    e: &safetensors::Entry,
    shapes: &Shapes,
    prefix: &str,
) -> Res<GIdx> {
    if e.shape != vec![shapes.in_features] {
        return Err(Error::Malformed(format!(
            "{prefix}: g_idx is {:?}, expected [in={}]",
            e.shape, shapes.in_features
        )));
    }
    let bytes = f.tensor(e);
    let mut values = Vec::with_capacity(shapes.in_features as usize);
    for i in 0..shapes.in_features {
        let v = e
            .dtype
            .decode(bytes, i)
            .ok_or_else(|| Error::Malformed(format!("{prefix}: g_idx[{i}] is unreadable")))?;
        if v < 0.0 || v >= shapes.groups as f64 {
            return Err(Error::Malformed(format!(
                "{prefix}: g_idx[{i}] is {v}, and there are {} groups",
                shapes.groups
            )));
        }
        values.push(v as u64);
    }
    Ok(GIdx {
        values,
        bytes: bytes.to_vec(),
        dtype: e.dtype.clone(),
    })
}

/// Whether `g_idx` is just `i / group_size` — the grouping `group_size` already
/// states, and therefore nothing a gather has to express.
fn is_ascending(g: &GIdx, shapes: &Shapes) -> bool {
    g.values
        .iter()
        .enumerate()
        .all(|(i, v)| *v == i as u64 / shapes.group_size)
}

// ------------------------------------------------------------- the other route --

/// Dequantizes a layer straight from the source bytes, by scalar code that goes
/// nowhere near the expression evaluator.
///
/// This is the half of I4 that matters. Byte identity proves the packed words
/// were copied; only an independent dequantization proves they are being *read*
/// the way the format writes them. Getting AWQ's interleave, GPTQ's transpose or
/// the zero-point offset wrong all produce well-formed containers full of wrong
/// numbers, and this is what refuses to claim those.
///
/// The result is in `[out, in]` order, matching the imported tensor.
fn reference_weight(
    cfg: &Config,
    f: &safetensors::File<'_>,
    parts: &Parts<'_>,
    l: &Layer,
) -> Res<Vec<f64>> {
    let epw = cfg.elems_per_word();
    let bits = cfg.bits;
    let mask = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let words = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let qw = words(f.tensor(parts.qweight));
    let qz = parts.qzeros.map(|e| words(f.tensor(e)));
    let sc_bytes = f.tensor(parts.scales);
    let sc_dtype = &parts.scales.dtype;
    let (rows, cols) = (l.in_features, l.out_features);
    let wpr = cols / epw; // packed words per row, for the output axis

    // The group each row belongs to.
    let group: Vec<u64> = match parts.g_idx {
        Some(e) => {
            let bytes = f.tensor(e);
            (0..rows)
                .map(|i| e.dtype.decode(bytes, i).unwrap_or(0.0) as u64)
                .collect()
        }
        None => (0..rows).map(|i| i / l.group_size).collect(),
    };

    let nibble = |w: u32, slot: u64| -> f64 { ((w >> (slot as u32 * bits)) & mask) as f64 };
    // Which slot of a word holds output column `c`, for a 4-bit AWQ word.
    let slot_of = |c: u64| -> u64 {
        match cfg.method {
            Method::Gptq => c % epw,
            Method::Awq => AWQ_REVERSE_ORDER[(c % epw) as usize],
        }
    };

    let mut out = vec![0.0f64; (rows * cols) as usize];
    for i in 0..rows {
        let g = group[i as usize];
        for j in 0..cols {
            let q = match cfg.method {
                // qweight is `[in/epw, out]`, packed down the input axis.
                Method::Gptq => nibble(qw[((i / epw) * cols + j) as usize], i % epw),
                // qweight is `[in, out/epw]`, packed along the output axis and
                // interleaved for the GEMM kernel.
                Method::Awq => nibble(qw[(i * wpr + j / epw) as usize], slot_of(j)),
            };
            let z = match &qz {
                Some(qz) => {
                    let raw = nibble(qz[(g * wpr + j / epw) as usize], slot_of(j));
                    if cfg.zeros_verbatim {
                        raw
                    } else {
                        raw + 1.0
                    }
                }
                None => 0.0,
            };
            let s = sc_dtype.decode(sc_bytes, g * cols + j).ok_or_else(|| {
                Error::Malformed(format!("{}: scales[{g},{j}] is unreadable", l.prefix))
            })?;
            // The evaluator rounds a dequantized value through the declared
            // output dtype (§04.7.2), so this has to as well or the comparison
            // would fail on precision rather than on correctness.
            out[(j * rows + i) as usize] = round_through((q - z) * s, sc_dtype);
        }
    }
    Ok(out)
}

fn round_through(x: f64, d: &DType) -> f64 {
    let mut buf = vec![0u8; d.packed_bytes(1).max(1) as usize];
    if d.encode(&mut buf, 0, x, Round::Rne) {
        d.decode(&buf, 0).unwrap_or(f64::NAN)
    } else {
        x
    }
}

// ---------------------------------------------------------------------- export --

/// One quantized layer, recovered from the container.
#[derive(Debug)]
pub struct Recovered {
    pub prefix: String,
    pub in_features: u64,
    pub out_features: u64,
    pub group_size: u64,
    pub groups: u64,
    pub bits: u32,
    pub act_order: bool,
    /// Whether the expression adds one to the stored zero point, which is
    /// AutoGPTQ's original checkpoint format.
    pub zeros_verbatim: bool,
}

/// What an export produced: the safetensors bytes and the config beside them.
pub struct Exported {
    pub weights: Vec<u8>,
    pub config: Vec<u8>,
    pub config_name: &'static str,
    pub layers: Vec<Recovered>,
    /// Tensors written back unchanged.
    pub plain: usize,
}

impl std::fmt::Debug for Exported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Exported {{ {} bytes, {} quantized layer(s), {} plain }}",
            self.weights.len(),
            self.layers.len(),
            self.plain
        )
    }
}

/// Writes a container back out as a GPTQ or AWQ checkpoint.
///
/// This is the other half of `import`, and the reason it can be byte-exact is
/// structural rather than lucky: the import never converted anything. The packed
/// 32-bit words went into the container as literals and the dequantization is an
/// expression over them, so exporting is a matter of finding those literals again
/// and writing them back under the names the format uses. §5.3's "lossless
/// round-trip" for these two rows is a consequence of that decision, not an extra
/// effort.
///
/// The config is **reconstructed from the container**, not remembered: the bit
/// width comes from the packed dtype, the group size from the scale grid, the
/// act-order flag from whether the scale is gathered, and the checkpoint format
/// from whether the expression carries the `+1` node. A container that has been
/// through `omni delta` or had its provenance stripped still exports correctly,
/// because none of that is where the information lives.
pub fn export(ctx: &Ctx<'_>, table: &TensorTable, method: Method) -> Res<Exported> {
    // Every literal a tensor's expression reads, with its declared shape, dtype
    // and layout — which is all that is needed to tell qweight from scales.
    let mut layers: Vec<Recovered> = Vec::new();
    let mut entries: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut plain = 0usize;

    // In the table's own load order (§04.2), not alphabetically. That order came
    // from the source file and is information the container kept; sorting here
    // would quietly discard it, and a re-import would then build a different
    // table from the same tensors.
    let names: Vec<&String> = if table.order.len() == table.tensors.len() {
        table.order.iter().collect()
    } else {
        table.tensors.keys().collect()
    };
    for name in names {
        let r = table
            .tensors
            .get(name)
            .ok_or_else(|| Error::Core(format!("`{name}` is in the order and not the table")))?;
        let desc = TensorDesc::load(ctx, r).map_err(|e| Error::Core(e.to_string()))?;
        let is_quant = desc.role.as_deref() == Some("quantized");
        if !is_quant {
            // An ordinary tensor: written back as its own bytes.
            let Expr::Literal { chunks, dtype, .. } = &desc.value else {
                return Err(Error::Unsupported(format!(
                    "`{name}` is a `{}` expression and not a quantized layer; this \
                     exporter writes packed layers and plain literals, and a third \
                     kind would need materializing into bytes the source format \
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
            entries.push((name.clone(), dtype.clone(), shape, bytes));
            plain += 1;
            continue;
        }

        let prefix = name
            .strip_suffix(".weight")
            .ok_or_else(|| {
                Error::Core(format!(
                    "`{name}` is a quantized layer but is not named `<layer>.weight`"
                ))
            })?
            .to_string();
        let rec = recover(ctx, &desc, &prefix, method, &mut entries)?;
        layers.push(rec);
    }

    if layers.is_empty() {
        return Err(Error::Malformed(format!(
            "no quantized layer in this container; there is nothing to write as a \
             {} checkpoint",
            method.name()
        )));
    }
    // One config for the file, so every layer has to agree about the things a
    // config states once. A container mixing bit widths is representable in OMNI
    // and not in these formats, and saying so beats writing a config that is
    // right for the first layer.
    let first = &layers[0];
    for l in &layers[1..] {
        for (what, a, b) in [
            ("bits", first.bits as u64, l.bits as u64),
            ("group_size", first.group_size, l.group_size),
            (
                "desc_act",
                u64::from(first.act_order),
                u64::from(l.act_order),
            ),
            (
                "checkpoint_format",
                u64::from(first.zeros_verbatim),
                u64::from(l.zeros_verbatim),
            ),
        ] {
            if a != b {
                return Err(Error::Unsupported(format!(
                    "`{}` and `{}` disagree about `{what}` ({a} and {b}); a {} \
                     config states it once for the whole file, so this container \
                     has no faithful {} form",
                    first.prefix,
                    l.prefix,
                    method.name(),
                    method.name()
                )));
            }
        }
    }

    let weights = write_safetensors(&entries)?;
    let config = write_config(method, first);
    Ok(Exported {
        weights,
        config,
        config_name: match method {
            Method::Gptq => "quantize_config.json",
            Method::Awq => "config.json",
        },
        layers,
        plain,
    })
}

/// Pulls one quantized layer's source tensors back out of its expression.
///
/// The literals are identified by their declared shape and layout rather than by
/// walking the expression's exact shape: a `packed` three-dimensional literal is
/// `qweight` or `qzeros` and the two differ in their first extent, a
/// two-dimensional float is `scales`, and a one-dimensional integer is `g_idx`.
/// Matching on structure would break the moment a rewrite reassociated the tree;
/// matching on what the tensors *are* does not.
fn recover(
    ctx: &Ctx<'_>,
    desc: &TensorDesc,
    prefix: &str,
    method: Method,
    entries: &mut Vec<(String, DType, Vec<u64>, Vec<u8>)>,
) -> Res<Recovered> {
    let shape = desc
        .sizes()
        .ok_or_else(|| Error::Core(format!("`{prefix}` has a symbolic shape")))?;
    if shape.len() != 2 {
        return Err(Error::Core(format!(
            "`{prefix}` is {shape:?}; a quantized linear weight is [out, in]"
        )));
    }
    let (out_features, in_features) = (shape[0], shape[1]);

    // The literals are located through the `dequantize` node rather than by
    // guessing from shapes: `qweight` is what it dequantizes and `qzeros` is
    // inside its scheme, and at a group size of one those two have identical
    // shapes — so any rule based on shape alone is ambiguous exactly where it
    // matters. Within each subtree there is one literal of each kind, which is
    // robust to a rewrite reassociating the tree around it.
    let deq = find_dequantize(&desc.value).ok_or_else(|| {
        Error::Core(format!(
            "`{prefix}` is marked quantized but its value has no `dequantize`"
        ))
    })?;
    let Expr::Dequantize { x, scheme } = deq else {
        unreachable!("find_dequantize returns a dequantize")
    };

    let packed_in = |e: &Expr| -> Res<Option<Packed>> {
        let mut found = None;
        let mut all = Vec::new();
        collect_literals(e, &mut all)?;
        for leaf in &all {
            if let Expr::Literal {
                chunks,
                shape,
                layout: Layout::Packed { elems_per_word, .. },
                ..
            } = leaf
            {
                let sizes = crate::expr::concrete(shape).ok_or_else(|| {
                    Error::Core(format!("`{prefix}` reads a symbolic packed literal"))
                })?;
                let bytes = ctx
                    .chunk_bytes(chunks)
                    .map_err(|e| Error::Core(e.to_string()))?;
                found = Some((sizes, bytes, 32 / *elems_per_word));
            }
        }
        Ok(found)
    };
    let dense_in = |e: &Expr, want_rank: usize| -> Res<Option<Dense>> {
        let mut found = None;
        let mut all = Vec::new();
        collect_literals(e, &mut all)?;
        for leaf in &all {
            if let Expr::Literal {
                chunks,
                dtype,
                shape,
                layout,
            } = leaf
            {
                if matches!(layout, Layout::Packed { .. }) {
                    continue;
                }
                let sizes = crate::expr::concrete(shape)
                    .ok_or_else(|| Error::Core(format!("`{prefix}` reads a symbolic literal")))?;
                if sizes.len() != want_rank {
                    continue;
                }
                let bytes = ctx
                    .chunk_bytes(chunks)
                    .map_err(|e| Error::Core(e.to_string()))?;
                found = Some((dtype.clone(), sizes, bytes));
            }
        }
        Ok(found)
    };

    let qweight = packed_in(x)?;
    let scale_expr = match scheme.get("scale") {
        Some(v) => Some(Expr::from_value(v).map_err(|e| Error::Core(e.to_string()))?),
        None => None,
    };
    let zero_expr = match scheme.get("zero") {
        Some(v) => Some(Expr::from_value(v).map_err(|e| Error::Core(e.to_string()))?),
        None => None,
    };
    let qzeros = match &zero_expr {
        Some(z) => packed_in(z)?.map(|(sizes, bytes, _)| (sizes, bytes)),
        None => None,
    };
    let scales = match &scale_expr {
        Some(sc) => dense_in(sc, 2)?,
        None => None,
    };
    // The act-order permutation is the gather index inside the scale, and it is
    // the one one-dimensional literal there.
    let g_idx = match &scale_expr {
        Some(sc) => dense_in(sc, 1)?,
        None => None,
    };

    let (qw_shape, qw_bytes, bits) = qweight.ok_or_else(|| {
        Error::Core(format!(
            "`{prefix}` has no packed weight among its literals"
        ))
    })?;
    let (sc_dtype, sc_shape, sc_bytes) = scales
        .ok_or_else(|| Error::Core(format!("`{prefix}` has no scales among its literals")))?;
    let groups = sc_shape[0];
    if groups == 0 || in_features % groups != 0 {
        return Err(Error::Core(format!(
            "`{prefix}`: {groups} group(s) do not divide {in_features} input features"
        )));
    }
    let group_size = in_features / groups;
    let epw = 32 / bits as u64;

    // `+1` in the tree means the stored zero point is one below the true one,
    // which is what AutoGPTQ's original format does. Its presence is the only
    // record of that convention, and it is enough.
    let zeros_verbatim = !has_plus_one(&desc.value);
    // A gathered scale is act-order: the group of a row comes from `g_idx`
    // rather than from dividing the row index.
    let act_order = g_idx.is_some();

    // Write the source tensors back under the names the format uses.
    let i32t = DType::I32;
    entries.push((
        format!("{prefix}.qweight"),
        i32t.clone(),
        match method {
            Method::Gptq => vec![in_features / epw, out_features],
            Method::Awq => vec![in_features, out_features / epw],
        },
        qw_bytes,
    ));
    let _ = qw_shape;
    if let Some((_, bytes)) = qzeros {
        entries.push((
            format!("{prefix}.qzeros"),
            i32t.clone(),
            vec![groups, out_features / epw],
            bytes,
        ));
    }
    entries.push((
        format!("{prefix}.scales"),
        sc_dtype,
        vec![groups, out_features],
        sc_bytes,
    ));
    match g_idx {
        Some((dtype, sizes, bytes)) => {
            entries.push((format!("{prefix}.g_idx"), dtype, sizes, bytes));
        }
        // An ascending `g_idx` was checked and not stored on import, because
        // `group_size` already says it. Recomputing it is not inventing a tensor:
        // it is writing down, in the form the source format wants, a fact the
        // container kept in a smaller one. AWQ has no `g_idx` at all.
        None if method == Method::Gptq => {
            let mut bytes = Vec::with_capacity(in_features as usize * 4);
            for i in 0..in_features {
                bytes.extend_from_slice(&((i / group_size) as i32).to_le_bytes());
            }
            entries.push((format!("{prefix}.g_idx"), i32t, vec![in_features], bytes));
        }
        None => {}
    }

    Ok(Recovered {
        prefix: prefix.to_string(),
        in_features,
        out_features,
        group_size,
        groups,
        bits,
        act_order,
        zeros_verbatim,
    })
}

/// A packed literal recovered from an expression: its shape, its bytes, and the
/// bit width its `elems_per_word` implies.
type Packed = (Vec<u64>, Vec<u8>, u32);
/// A dense literal recovered from an expression: dtype, shape, bytes.
type Dense = (DType, Vec<u64>, Vec<u8>);

/// The first `dequantize` in a tree, which is where a quantized layer's operands
/// live.
fn find_dequantize(e: &Expr) -> Option<&Expr> {
    if matches!(e, Expr::Dequantize { .. }) {
        return Some(e);
    }
    e.children().into_iter().find_map(find_dequantize)
}

/// Every `literal` an expression reads, including the ones inside a
/// `dequantize` scheme.
///
/// §05.1 puts `scale`, `zero` and `order` in the scheme as *expressions*, and
/// the scheme is a CBOR value rather than a child node — so a walk over child
/// expressions alone silently misses them.
fn collect_literals(e: &Expr, out: &mut Vec<Expr>) -> Res<()> {
    if matches!(e, Expr::Literal { .. }) {
        out.push(e.clone());
    }
    if let Expr::Dequantize { scheme, .. } | Expr::Quantize { scheme, .. } = e {
        for key in ["scale", "zero", "order", "scale_zero", "scale_scale"] {
            if let Some(v) = scheme.get(key) {
                let sub = Expr::from_value(v).map_err(|err| {
                    Error::Core(format!("a scheme's `{key}` is not an expression: {err}"))
                })?;
                collect_literals(&sub, out)?;
            }
        }
    }
    for c in e.children() {
        collect_literals(c, out)?;
    }
    Ok(())
}

/// Whether the tree adds a constant one anywhere — AutoGPTQ's zero-point offset.
///
/// The node is inside the scheme's `zero`, not among the expression's children,
/// so this looks there too. Missing it would report every v1 checkpoint as v2 and
/// shift every weight by one quantization step on the way back out.
fn has_plus_one(e: &Expr) -> bool {
    if let Expr::Bin {
        op: crate::expr::BinOp::Add,
        b,
        ..
    } = e
    {
        if let Expr::Full { value, .. } = b.as_ref() {
            if value.as_f64() == 1.0 {
                return true;
            }
        }
    }
    if let Expr::Dequantize { scheme, .. } | Expr::Quantize { scheme, .. } = e {
        for key in ["zero", "scale"] {
            if let Some(sub) = scheme.get(key).and_then(|v| Expr::from_value(v).ok()) {
                if has_plus_one(&sub) {
                    return true;
                }
            }
        }
    }
    e.children().iter().any(|c| has_plus_one(c))
}

fn write_config(method: Method, l: &Recovered) -> Vec<u8> {
    use crate::json;
    let inner = vec![
        ("bits", json::Value::U(l.bits as u64)),
        ("group_size", json::Value::U(l.group_size)),
        ("quant_method", json::string(method.name())),
    ];
    let v = match method {
        Method::Gptq => {
            let mut p = inner;
            p.push(("desc_act", json::Value::Bool(l.act_order)));
            p.push(("sym", json::Value::Bool(true)));
            p.push((
                "checkpoint_format",
                json::string(if l.zeros_verbatim { "gptq_v2" } else { "gptq" }),
            ));
            json::object(p)
        }
        Method::Awq => {
            let mut p = inner;
            p.push(("zero_point", json::Value::Bool(true)));
            p.push(("version", json::string("gemm")));
            json::object(vec![("quantization_config", json::object(p))])
        }
    };
    v.encode().into_bytes()
}

/// Writes safetensors from named byte buffers.
fn write_safetensors(entries: &[(String, DType, Vec<u64>, Vec<u8>)]) -> Res<Vec<u8>> {
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
    use crate::tensor::Severity;

    // ------------------------------------------------------------- fixtures --

    /// A safetensors file built by hand, so the tests are against the format and
    /// not against this module's reading of it.
    fn safetensors_file(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut header = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = data.len() as u64;
            data.extend_from_slice(bytes);
            header.push((
                *name,
                json::object(vec![
                    ("dtype", json::string(*dtype)),
                    (
                        "shape",
                        json::Value::Array(shape.iter().map(|d| json::Value::U(*d)).collect()),
                    ),
                    (
                        "data_offsets",
                        json::Value::Array(vec![
                            json::Value::U(start),
                            json::Value::U(data.len() as u64),
                        ]),
                    ),
                ]),
            ));
        }
        let head = json::object(header).encode().into_bytes();
        let mut out = (head.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&head);
        out.extend_from_slice(&data);
        out
    }

    /// Packs values down the *input* axis, eight to a 32-bit word: AutoGPTQ's
    /// `qweight`, written the way `pack()` writes it.
    // The loops mirror the format's own nesting — word row, column, slot — and
    // that is the point: an idiomatic rewrite would obscure which axis is packed.
    #[allow(clippy::needless_range_loop)]
    fn pack_down_in(q: &[Vec<u32>], bits: u32) -> Vec<u8> {
        let epw = 32 / bits as usize;
        let (rows, cols) = (q.len(), q[0].len());
        let mut out = Vec::new();
        for r in 0..rows / epw {
            for j in 0..cols {
                let mut w = 0u32;
                for s in 0..epw {
                    w |= q[r * epw + s][j] << (bits as usize * s);
                }
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Packs values along the *output* axis. `interleave` is AWQ's GEMM order:
    /// column `c` of a word goes into slot `AWQ_REVERSE_ORDER[c]`, which is what
    /// makes a sequential unpack come out permuted.
    fn pack_along_out(q: &[Vec<u32>], bits: u32, interleave: bool) -> Vec<u8> {
        let epw = 32 / bits as usize;
        let (rows, cols) = (q.len(), q[0].len());
        let mut out = Vec::new();
        for row in q.iter().take(rows) {
            for g in 0..cols / epw {
                let mut w = 0u32;
                for c in 0..epw {
                    let slot = if interleave {
                        AWQ_REVERSE_ORDER[c] as usize
                    } else {
                        c
                    };
                    w |= row[g * epw + c] << (bits as usize * slot);
                }
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    fn f16s(v: &[f64]) -> Vec<u8> {
        let mut out = vec![0u8; v.len() * 2];
        for (i, x) in v.iter().enumerate() {
            assert!(DType::F16.encode(&mut out, i as u64, *x, Round::Rne));
        }
        out
    }

    fn i32s(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// A deterministic pseudo-random 4-bit value, so the fixtures exercise every
    /// slot of a word rather than a tidy pattern that hides a swap.
    fn nib(i: u64, j: u64) -> u32 {
        ((i * 7 + j * 11 + (i * j) % 5) % 16) as u32
    }

    struct Fixture {
        weights: Vec<u8>,
        /// `[in][out]` unpacked quantized values.
        q: Vec<Vec<u32>>,
        /// `[groups][out]` stored zero points.
        z: Vec<Vec<u32>>,
        /// `[groups][out]` scales.
        s: Vec<Vec<f64>>,
        in_features: u64,
        out_features: u64,
        group_size: u64,
    }

    /// One quantized layer plus one ordinary tensor, in a named format.
    fn fixture(
        method: Method,
        rows: u64,
        cols: u64,
        group_size: u64,
        g_idx: Option<Vec<u32>>,
    ) -> Fixture {
        let groups = rows / group_size;
        let q: Vec<Vec<u32>> = (0..rows)
            .map(|i| (0..cols).map(|j| nib(i, j)).collect())
            .collect();
        let z: Vec<Vec<u32>> = (0..groups)
            .map(|g| (0..cols).map(|j| 5 + ((g + j) % 3) as u32).collect())
            .collect();
        // Powers of two, so every product is exact in f16 and a mismatch is a
        // real mismatch rather than rounding.
        let s: Vec<Vec<f64>> = (0..groups)
            .map(|g| {
                (0..cols)
                    .map(|j| 0.25 * f64::from(1 << ((g + j) % 3)))
                    .collect()
            })
            .collect();

        let qweight = match method {
            Method::Gptq => pack_down_in(&q, 4),
            Method::Awq => pack_along_out(&q, 4, true),
        };
        let qzeros = pack_along_out(&z, 4, method == Method::Awq);
        let scales = f16s(&s.iter().flatten().copied().collect::<Vec<f64>>());
        let epw = 8u64;
        let mut entries: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = vec![
            (
                "l.qweight",
                "I32",
                match method {
                    Method::Gptq => vec![rows / epw, cols],
                    Method::Awq => vec![rows, cols / epw],
                },
                qweight,
            ),
            ("l.qzeros", "I32", vec![groups, cols / epw], qzeros),
            ("l.scales", "F16", vec![groups, cols], scales),
            // An ordinary tensor beside the quantized one: this is what a real
            // checkpoint's norms and embeddings look like, and they must survive
            // untouched.
            ("norm.weight", "F16", vec![4], f16s(&[1.0, 0.5, 0.25, 2.0])),
        ];
        if let Some(g) = &g_idx {
            entries.insert(3, ("l.g_idx", "I32", vec![rows], i32s(g)));
        }
        Fixture {
            weights: safetensors_file(&entries),
            q,
            z,
            s,
            in_features: rows,
            out_features: cols,
            group_size,
        }
    }

    fn config(pairs: Vec<(&str, json::Value)>) -> Vec<u8> {
        json::object(pairs).encode().into_bytes()
    }

    fn gptq_config(bits: u64, group_size: i64, extra: Vec<(&str, json::Value)>) -> Vec<u8> {
        let mut p = vec![
            ("quant_method", json::string("gptq")),
            ("bits", json::Value::U(bits)),
            (
                "group_size",
                if group_size < 0 {
                    json::Value::I(group_size)
                } else {
                    json::Value::U(group_size as u64)
                },
            ),
        ];
        p.extend(extra);
        config(p)
    }

    fn awq_config(bits: u64, group_size: u64, extra: Vec<(&str, json::Value)>) -> Vec<u8> {
        let mut p = vec![
            ("quant_method", json::string("awq")),
            ("bits", json::Value::U(bits)),
            ("group_size", json::Value::U(group_size)),
            ("zero_point", json::Value::Bool(true)),
        ];
        p.extend(extra);
        config(p)
    }

    /// What the imported `l.weight` should be, worked out from the fixture's own
    /// numbers rather than from either dequantizer.
    fn expected(f: &Fixture, group_of: &dyn Fn(u64) -> u64, plus_one: bool) -> Vec<f64> {
        let mut out = Vec::new();
        for j in 0..f.out_features {
            for i in 0..f.in_features {
                let g = group_of(i) as usize;
                let z = f.z[g][j as usize] as f64 + f64::from(u8::from(plus_one));
                out.push((f.q[i as usize][j as usize] as f64 - z) * f.s[g][j as usize]);
            }
        }
        out
    }

    /// The dequantized `l.weight`, read back out of the object graph.
    fn read_weight(im: &Imported, hash: HashAlgo, name: &str) -> crate::expr::Tensor {
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &im.objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let tref = im
            .objects
            .iter()
            .find(|o| o.otype == otype::TENSOR_TABLE)
            .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
            .unwrap();
        let table = TensorTable::load(&ctx, &tref).unwrap();
        let d = TensorDesc::load(&ctx, table.get(name).unwrap()).unwrap();
        d.value.eval(&ctx).unwrap()
    }

    // ---------------------------------------------------------------- tests --

    #[test]
    fn gptq_int4_is_a_packed_literal_a_transpose_and_one_dequantize() {
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        assert_eq!(im.layers.len(), 1);
        assert_eq!(im.plain, 1, "the norm is imported as an ordinary tensor");
        let l = &im.layers[0];
        assert_eq!((l.in_features, l.out_features), (16, 8));
        assert_eq!((l.groups, l.group_size), (2, 8));
        assert!(!l.act_order);

        // The values, against arithmetic done in the test rather than by either
        // dequantizer. AutoGPTQ's original format stores the zero point one low,
        // so the expectation adds it back.
        let got = read_weight(&im, HashAlgo::default(), "l.weight");
        assert_eq!(got.shape, vec![8, 16], "the layer weight is [out, in]");
        let want = expected(&f, &|i| i / 8, true);
        assert_eq!(got.data, want);

        // I4 reports a measurement, not an intention.
        assert_eq!(im.report.dequant_checked, 16 * 8);
        assert_eq!(im.report.verify_method, "byte-identity + sample-dequant");
        assert!(im.report.lossless);
        assert!(im.report.warnings.is_empty(), "{:?}", im.report.warnings);
    }

    #[test]
    fn awq_undoes_the_gemm_interleave_and_gptq_would_not() {
        let f = fixture(Method::Awq, 8, 8, 8, None);
        let im = import(
            &awq_config(4, 8, vec![]),
            &f.weights,
            Method::Awq,
            &ImportOpts::default(),
        )
        .unwrap();
        let got = read_weight(&im, HashAlgo::default(), "l.weight");
        assert_eq!(got.shape, vec![8, 8]);
        // AWQ stores the zero point verbatim.
        assert_eq!(got.data, expected(&f, &|_| 0, false));

        // And the interleave is load-bearing: reading the same bytes with the
        // GPTQ layout gives a different answer, so a test that passed either way
        // would not be testing anything.
        let same_bytes = import(
            &gptq_config(4, 8, vec![("checkpoint_format", json::string("gptq_v2"))]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        );
        // Refusing outright is just as good an answer as a different one.
        if let Ok(other) = same_bytes {
            assert_ne!(
                read_weight(&other, HashAlgo::default(), "l.weight").data,
                got.data,
                "the two layouts must not agree, or the interleave is untested"
            );
        }
    }

    #[test]
    fn act_order_is_a_gather_and_g_idx_decides_not_desc_act() {
        // A grouping no `group_size` describes: the second half of the rows is in
        // group 0 and the first half in group 1.
        let g_idx: Vec<u32> = (0..16).map(|i| if i < 8 { 1 } else { 0 }).collect();
        let f = fixture(Method::Gptq, 16, 8, 8, Some(g_idx.clone()));
        let im = import(
            // `desc_act` says false; the tensor says otherwise, and the tensor is
            // what the weights were quantized with.
            &gptq_config(4, 8, vec![("desc_act", json::Value::Bool(false))]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        assert!(im.layers[0].act_order);
        assert!(
            im.report.warnings.iter().any(|w| w.contains("desc_act")),
            "the disagreement is reported: {:?}",
            im.report.warnings
        );
        let got = read_weight(&im, HashAlgo::default(), "l.weight");
        assert_eq!(
            got.data,
            expected(&f, &|i| u64::from(g_idx[i as usize]), true)
        );
    }

    #[test]
    fn an_ascending_g_idx_keeps_the_compact_block_form() {
        let g_idx: Vec<u32> = (0..16).map(|i| i / 8).collect();
        let f = fixture(Method::Gptq, 16, 8, 8, Some(g_idx));
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        assert!(!im.layers[0].act_order);
        assert!(im.report.warnings.is_empty(), "{:?}", im.report.warnings);
        // And it says so, rather than leaving the reader to wonder where g_idx
        // went: it was checked, not ignored.
        assert!(im
            .report
            .assumptions
            .iter()
            .any(|n| n.item == "l.g_idx" && n.action.contains("checked to equal")));
        assert_eq!(
            read_weight(&im, HashAlgo::default(), "l.weight").data,
            expected(&f, &|i| i / 8, true)
        );
    }

    #[test]
    fn the_two_gptq_checkpoint_formats_differ_by_exactly_one_step() {
        // §05.1's whole reason for a closed `formula` set. The same bytes, two
        // named conventions, and every weight differs by one scale.
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let v1 = import(
            &gptq_config(4, 8, vec![("checkpoint_format", json::string("gptq"))]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let v2 = import(
            &gptq_config(4, 8, vec![("checkpoint_format", json::string("gptq_v2"))]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let a = read_weight(&v1, HashAlgo::default(), "l.weight");
        let b = read_weight(&v2, HashAlgo::default(), "l.weight");
        for j in 0..f.out_features {
            for i in 0..f.in_features {
                let k = (j * f.in_features + i) as usize;
                let s = f.s[(i / f.group_size) as usize][j as usize];
                assert_eq!(b.data[k] - a.data[k], s, "at [{j},{i}]");
            }
        }
        // And which one was assumed is in the report, not in someone's head.
        let note = v1
            .report
            .assumptions
            .iter()
            .find(|n| n.item == "zero point")
            .unwrap();
        assert!(note.action.contains("one added"), "{}", note.action);
        assert!(v2
            .report
            .assumptions
            .iter()
            .any(|n| n.item == "zero point" && n.action.contains("as stored")));
    }

    #[test]
    fn per_column_grouping_is_group_size_minus_one() {
        let f = fixture(Method::Gptq, 8, 8, 8, None);
        let im = import(
            &gptq_config(4, -1, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        assert_eq!((im.layers[0].groups, im.layers[0].group_size), (1, 8));
        assert_eq!(
            read_weight(&im, HashAlgo::default(), "l.weight").data,
            expected(&f, &|_| 0, true)
        );
    }

    #[test]
    fn the_imported_graph_satisfies_the_tensor_rules() {
        // R-T01 through R-T04 over expressions this module wrote: the declared
        // shape against the inferred one, the ChunkList sizes against the packed
        // layout, and the scale tensor against the block grid.
        let f = fixture(
            Method::Gptq,
            16,
            8,
            8,
            Some((0..16).map(|i| i % 2).collect()),
        );
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let hash = HashAlgo::default();
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &im.objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let tref = im
            .objects
            .iter()
            .find(|o| o.otype == otype::TENSOR_TABLE)
            .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
            .unwrap();
        let table = TensorTable::load(&ctx, &tref).unwrap();
        let mut seen = 0;
        for (name, r) in &table.tensors {
            let d = TensorDesc::load(&ctx, r).unwrap();
            let findings = d.check(&ctx, name);
            assert!(
                !findings.iter().any(|x| x.severity == Severity::Invalid),
                "{name}: {findings:?}"
            );
            seen += 1;
        }
        assert_eq!(seen, 2, "the layer weight and the norm");
    }

    #[test]
    fn the_packed_bytes_are_the_source_bytes() {
        // Byte identity is the other half of I4, and it covers the tensors that
        // only exist inside an expression: qweight, qzeros and scales are never
        // named in the table, only read by it.
        let g_idx: Vec<u32> = (0..16).map(|i| u32::from(i < 8)).collect();
        let f = fixture(Method::Gptq, 16, 8, 8, Some(g_idx));
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let src = safetensors::File::parse(&f.weights).unwrap();
        // qweight, qzeros, scales, g_idx, norm.weight.
        assert_eq!(im.report.verified_tensors, 5);
        assert_eq!(
            im.report.verified_bytes,
            src.entries.iter().map(|e| e.len()).sum::<u64>()
        );
    }

    #[test]
    fn an_ascending_g_idx_is_the_one_tensor_not_stored() {
        // It is `i / group_size`, which `group_size` already says, so storing it
        // would be storing the same fact twice. The report names it rather than
        // letting the tensor count quietly come up one short.
        let f = fixture(
            Method::Gptq,
            16,
            8,
            8,
            Some((0..16).map(|i| i / 8).collect()),
        );
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let src = safetensors::File::parse(&f.weights).unwrap();
        assert_eq!(im.report.verified_tensors, 4, "g_idx is not among them");
        let g = src.get("l.g_idx").unwrap();
        assert_eq!(
            im.report.verified_bytes,
            src.entries.iter().map(|e| e.len()).sum::<u64>() - g.len()
        );
        assert!(im
            .report
            .assumptions
            .iter()
            .any(|n| n.item == "l.g_idx" && n.action.contains("checked to equal")));
    }

    #[test]
    fn what_is_refused_is_refused_by_name() {
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let cases: Vec<(Vec<u8>, Method, &str)> = vec![
            // 3-bit GPTQ straddles the word boundary.
            (gptq_config(3, 8, vec![]), Method::Gptq, "straddle"),
            (gptq_config(5, 8, vec![]), Method::Gptq, "5-bit"),
            // An unknown checkpoint format is the one case where guessing the
            // zero-point convention corrupts every weight.
            (
                gptq_config(4, 8, vec![("checkpoint_format", json::string("gptq_v3"))]),
                Method::Gptq,
                "gptq_v3",
            ),
            (
                awq_config(4, 8, vec![("version", json::string("gemv"))]),
                Method::Awq,
                "gemv",
            ),
            (awq_config(8, 8, vec![]), Method::Awq, "8-bit AWQ"),
            // A config that names the other method.
            (awq_config(4, 8, vec![]), Method::Gptq, "quant_method"),
            (gptq_config(4, 8, vec![]), Method::Awq, "quant_method"),
        ];
        for (cfg, method, needle) in cases {
            let err = import(&cfg, &f.weights, method, &ImportOpts::default())
                .expect_err("should be refused")
                .to_string();
            assert!(err.contains(needle), "{err} should mention {needle}");
        }
    }

    #[test]
    fn a_shape_that_does_not_add_up_is_reported() {
        // The scales grid must match the grouping; a mismatch is exactly the case
        // where a wrong `group_size` would otherwise dequantize plausibly.
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let err = import(
            &gptq_config(4, 4, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .expect_err("4 groups declared, 2 stored")
        .to_string();
        assert!(err.contains("scales is [2, 8]"), "{err}");

        // And a `qweight` with no `scales` is malformed rather than plain.
        let bytes = safetensors_file(&[("l.qweight", "I32", vec![2, 8], i32s(&[0u32; 16]))]);
        let err = import(
            &gptq_config(4, 8, vec![]),
            &bytes,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .expect_err("no scales")
        .to_string();
        assert!(err.contains("no `l.scales`"), "{err}");
    }

    #[test]
    fn a_file_with_nothing_quantized_is_not_a_gptq_checkpoint() {
        let bytes = safetensors_file(&[("w", "F16", vec![2], f16s(&[1.0, 2.0]))]);
        let err = import(
            &gptq_config(4, 8, vec![]),
            &bytes,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .expect_err("no qweight")
        .to_string();
        assert!(err.contains("not a gptq checkpoint"), "{err}");
    }

    #[test]
    fn a_layer_too_large_to_dequantize_says_so() {
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let im = import(
            &gptq_config(4, 8, vec![]),
            &f.weights,
            Method::Gptq,
            &ImportOpts {
                max_verify_elems: 4,
                ..Default::default()
            },
        )
        .unwrap();
        // Silence would read as "checked and fine"; the report says what it did
        // not do.
        assert_eq!(im.report.dequant_checked, 0);
        assert!(
            im.report.warnings.iter().any(|w| w.contains("too large")),
            "{:?}",
            im.report.warnings
        );
        assert!(
            im.report.verified_tensors > 0,
            "the bytes are still checked"
        );
    }
    // ----------------------------------------------------- the round trip --

    /// Imports, exports, and compares — the whole of §5.3's claim for these two
    /// rows, checked rather than asserted.
    fn round_trip(method: Method, cfg: Vec<u8>, f: &Fixture) -> (Vec<u8>, Exported) {
        let im = import(&cfg, &f.weights, method, &ImportOpts::default()).unwrap();
        let hash = HashAlgo::default();
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &im.objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let tref = im
            .objects
            .iter()
            .find(|o| o.otype == otype::TENSOR_TABLE)
            .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
            .unwrap();
        let table = TensorTable::load(&ctx, &tref).unwrap();
        let ex = export(&ctx, &table, method)
            .unwrap_or_else(|e| panic!("{} export failed: {e}", method.name()));
        (f.weights.clone(), ex)
    }

    #[test]
    fn a_gptq_checkpoint_round_trips_byte_for_byte() {
        let f = fixture(
            Method::Gptq,
            16,
            8,
            8,
            Some((0..16).map(|i| i / 8).collect()),
        );
        let (src, ex) = round_trip(Method::Gptq, gptq_config(4, 8, vec![]), &f);

        // Every tensor the source had, with the same bytes. Compared tensor by
        // tensor rather than file by file, because safetensors' header is a JSON
        // map and two writers can order it differently while meaning the same
        // thing — that difference is not a loss, and calling it one would be as
        // wrong as missing a real one.
        let a = safetensors::File::parse(&src).unwrap();
        let b = safetensors::File::parse(&ex.weights).unwrap();
        let names_a: Vec<&str> = a.entries.iter().map(|e| e.name.as_str()).collect();
        for e in &a.entries {
            let got = b
                .get(&e.name)
                .unwrap_or_else(|| panic!("`{}` did not survive the round trip", e.name));
            assert_eq!(got.shape, e.shape, "{}", e.name);
            assert_eq!(got.dtype, e.dtype, "{}", e.name);
            assert_eq!(b.tensor(got), a.tensor(e), "`{}` differs", e.name);
        }
        // And nothing invented: the same names out as in.
        let names_b: Vec<&str> = b.entries.iter().map(|e| e.name.as_str()).collect();
        let mut sa: Vec<&str> = names_a.clone();
        let mut sb: Vec<&str> = names_b.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        assert_eq!(sa, sb, "the tensor sets differ");

        // The config is reconstructed from the container rather than remembered,
        // so it has to come back saying the same things.
        let cfg = crate::json::parse(&ex.config).unwrap();
        assert_eq!(cfg.get("bits").unwrap().as_u64(), Some(4));
        assert_eq!(cfg.get("group_size").unwrap().as_u64(), Some(8));
        assert_eq!(cfg.get("quant_method").unwrap().as_str(), Some("gptq"));
        assert_eq!(cfg.get("desc_act").unwrap().as_bool(), Some(false));
        // The `+1` node in the expression is the only record of AutoGPTQ's
        // original zero-point convention, and it survives.
        assert_eq!(
            cfg.get("checkpoint_format").unwrap().as_str(),
            Some("gptq"),
            "the v1 offset should be recovered from the expression"
        );
        assert_eq!(ex.layers.len(), 1);
        assert_eq!(ex.plain, 1, "the norm tensor came back too");
    }

    #[test]
    fn an_awq_checkpoint_round_trips_and_keeps_its_interleave() {
        let f = fixture(Method::Awq, 8, 8, 8, None);
        let (src, ex) = round_trip(Method::Awq, awq_config(4, 8, vec![]), &f);
        let a = safetensors::File::parse(&src).unwrap();
        let b = safetensors::File::parse(&ex.weights).unwrap();
        for e in &a.entries {
            let got = b
                .get(&e.name)
                .unwrap_or_else(|| panic!("`{}` missing", e.name));
            assert_eq!(b.tensor(got), a.tensor(e), "`{}` differs", e.name);
        }
        // AWQ has no g_idx, and the exporter must not invent one.
        assert!(
            b.get("l.g_idx").is_none(),
            "AWQ has no g_idx and one was written anyway"
        );
        let cfg = crate::json::parse(&ex.config).unwrap();
        let q = cfg.get("quantization_config").unwrap();
        assert_eq!(q.get("quant_method").unwrap().as_str(), Some("awq"));
        assert_eq!(q.get("version").unwrap().as_str(), Some("gemm"));
    }

    #[test]
    fn act_order_survives_the_round_trip_as_act_order() {
        // A non-ascending g_idx is stored, used as a gather, and has to come back
        // both as the tensor and as `desc_act: true`.
        let g_idx: Vec<u32> = (0..16).map(|i| u32::from(i < 8)).collect();
        let f = fixture(Method::Gptq, 16, 8, 8, Some(g_idx.clone()));
        let (src, ex) = round_trip(Method::Gptq, gptq_config(4, 8, vec![]), &f);
        let a = safetensors::File::parse(&src).unwrap();
        let b = safetensors::File::parse(&ex.weights).unwrap();
        assert_eq!(
            b.tensor(b.get("l.g_idx").expect("g_idx should be written")),
            a.tensor(a.get("l.g_idx").unwrap()),
            "the act-order permutation changed"
        );
        let cfg = crate::json::parse(&ex.config).unwrap();
        assert_eq!(cfg.get("desc_act").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn the_v2_convention_comes_back_as_v2() {
        // The two checkpoint formats differ by one node in the expression, and
        // that node is what the exporter reads. Getting this backwards would
        // shift every weight by one quantization step on the way out.
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let (_, ex) = round_trip(
            Method::Gptq,
            gptq_config(4, 8, vec![("checkpoint_format", json::string("gptq_v2"))]),
            &f,
        );
        let cfg = crate::json::parse(&ex.config).unwrap();
        assert_eq!(
            cfg.get("checkpoint_format").unwrap().as_str(),
            Some("gptq_v2")
        );
    }

    #[test]
    fn a_container_a_config_cannot_describe_is_refused() {
        // Two layers quantized differently is representable in OMNI and not in a
        // format whose config states the bit width once. Writing a config that is
        // right for the first layer would be the quietest possible corruption.
        let mut entries: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
        let _ = &mut entries;
        let a = fixture(Method::Gptq, 16, 8, 8, None);
        let b = fixture(Method::Gptq, 16, 8, 16, None);
        // Import each separately, then put both tables in one store: the check is
        // on the exporter, so the container is built to trip it.
        let hash = HashAlgo::default();
        let ia = import(
            &gptq_config(4, 8, vec![]),
            &a.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let ib = import(
            &gptq_config(4, 16, vec![]),
            &b.weights,
            Method::Gptq,
            &ImportOpts::default(),
        )
        .unwrap();
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in ia.objects.iter().chain(ib.objects.iter()) {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let tref = |im: &Imported| {
            im.objects
                .iter()
                .find(|o| o.otype == otype::TENSOR_TABLE)
                .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
                .unwrap()
        };
        let ta = TensorTable::load(&ctx, &tref(&ia)).unwrap();
        let tb = TensorTable::load(&ctx, &tref(&ib)).unwrap();
        let mut merged = ta.clone();
        for (name, r) in &tb.tensors {
            merged.tensors.insert(format!("second.{name}"), *r);
        }
        let e = export(&ctx, &merged, Method::Gptq).expect_err("mixed group sizes");
        assert!(e.to_string().contains("group_size"), "{e}");
        assert!(e.to_string().contains("states it once"), "{e}");
    }
    #[test]
    fn a_re_import_builds_the_identical_tensor_table() {
        // E4 is stronger than byte equality of the tensors: import, export and
        // import again has to produce the *same object graph*, or the round trip
        // is only nearly closed. Load order is what makes the difference —
        // sorting the tensors on the way out would keep every byte and still
        // build a different table.
        let f = fixture(Method::Gptq, 16, 8, 8, None);
        let cfg = gptq_config(4, 8, vec![]);
        let hash = HashAlgo::default();
        let table_of = |weights: &[u8]| -> Digest {
            let im = import(&cfg, weights, Method::Gptq, &ImportOpts::default()).unwrap();
            im.objects
                .iter()
                .find(|o| o.otype == otype::TENSOR_TABLE)
                .map(|o| o.digest(hash))
                .unwrap()
        };
        let first = table_of(&f.weights);
        let (_, ex) = round_trip(Method::Gptq, cfg.clone(), &f);
        assert_eq!(
            table_of(&ex.weights),
            first,
            "the round trip kept every byte and still built a different table"
        );
    }
}
