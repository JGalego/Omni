//! bitsandbytes NF4, FP4 and INT8 (row 19 of `docs/design/import-export.md` §3).
//!
//! This is the format QLoRA produces, and the one most quantized fine-tunes in
//! circulation are stored in. It is also the format that fits §05 best, which is
//! why it is worth importing as *expressions* rather than converting: NF4 is a
//! sixteen-entry codebook with a per-block scale, and that scale is itself
//! quantized against a second codebook. §05.4 has codebooks and §05.2 has
//! per-block scales, and because a scheme's `scale` is an **expression** rather
//! than a tensor, double quantization needs no new formula — the outer
//! dequantize's scale is simply an inner dequantize plus an offset.
//!
//! What a checkpoint carries, per quantized weight `W`:
//!
//! | Tensor | Meaning |
//! |---|---|
//! | `W` | the weights, two 4-bit indices per byte, high nibble first |
//! | `W.absmax` | one scale per block: `f32`, or `u8` under double quantization |
//! | `W.quant_map` | the 16-entry codebook — NF4's values, or FP4's |
//! | `W.nested_absmax` | the scale of the scales (double quantization only) |
//! | `W.nested_quant_map` | the 256-entry codebook for `absmax` (likewise) |
//! | `W.quant_state.bitsandbytes__nf4` | JSON: blocksize, shape, dtype, offset |
//!
//! The quant state is plain JSON inside a `u8` tensor, which is why this importer
//! is short: §03's JSON codec already reads it.
//!
//! Blocking is over the **flattened** tensor, in groups of `blocksize`, and that
//! does not in general correspond to any axis-aligned block shape — a 128-wide
//! row with blocksize 64 is two blocks, a 96-wide row is one and a half. So the
//! dequantize is built over a one-dimensional view of `numel` elements and then
//! reshaped, which is exactly right and not a workaround: the file's blocking is
//! flat, and saying so is more faithful than finding a block shape that happens
//! to divide this particular tensor.
//!
//! The codebook is stored as the **values the file contains**, not as §05.4's
//! `normal-float` recipe, even though the recipe reproduces NF4 to about one
//! float32 ULP (there is a test in `quant.rs` that says so, and says by how
//! much). One ULP is one ULP: §01.1 says nothing is invented, and a lossless
//! importer does not get to substitute its own arithmetic for the table it was
//! handed.
//!
//! ## How closely this agrees with bitsandbytes
//!
//! Measured against the library itself, in CI, for every variant:
//!
//! | Variant | Worst difference |
//! |---|---|
//! | NF4/FP4, single-quantized scales | **0** — bit-identical |
//! | LLM.int8 | **0** — bit-identical |
//! | NF4/FP4, double-quantized scales | 2.4e-7, one float32 ULP |
//!
//! The last row is understood rather than tolerated. Reconstructing a block scale
//! under double quantization ends in `+ nested_offset`, and bitsandbytes performs
//! that addition on an f32 value while this evaluator works in f64 and carries the
//! extra precision into the outer product. Emulating the library's rounding in
//! Python reproduces its output *exactly*, which is what identifies the cause; and
//! the two rows that involve no such addition are bit-identical, which is what
//! confirms it. OMNI is the more precise of the two here, which is a strange thing
//! to have to say about a fidelity check and is the honest description.

use crate::cbor::Value;
use crate::container::{Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::expr::{dims, Expr, Scalar};
use crate::layout::{BitOrder, Layout, Order};
use crate::model::{ModelBuilder, TensorSpec};
use crate::safetensors::{self, Fidelity, Note};
use crate::tensor::{Materialize, TensorDesc};

type Res<T> = Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    Unsupported(String),
    Core(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed bitsandbytes checkpoint: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Core(m) => write!(f, "{m}"),
        }
    }
}

/// Which of bitsandbytes' schemes a weight uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// 4-bit NormalFloat: the QLoRA codebook, quantiles of a normal.
    Nf4,
    /// 4-bit float: the same machinery with a different sixteen values.
    Fp4,
    /// LLM.int8's row-wise 8-bit, with a per-row absmax in `SCB`.
    Int8,
}

impl Quant {
    pub fn name(self) -> &'static str {
        match self {
            Quant::Nf4 => "nf4",
            Quant::Fp4 => "fp4",
            Quant::Int8 => "int8",
        }
    }
}

/// One imported quantized weight.
#[derive(Clone, Debug)]
pub struct Layer {
    pub name: String,
    pub quant: Quant,
    pub shape: Vec<u64>,
    /// Elements per scale. For `Int8` this is the row length.
    pub blocksize: u64,
    /// Whether `absmax` was itself quantized.
    pub double: bool,
}

pub struct Imported {
    pub objects: Vec<Object>,
    pub root: Digest,
    pub report: Fidelity,
    pub layers: Vec<Layer>,
    /// Tensors no quantized weight claimed, imported unchanged.
    pub plain: usize,
}

impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Imported {{ {} objects, {} quantized, {} plain, lossless {} }}",
            self.objects.len(),
            self.layers.len(),
            self.plain,
            self.report.lossless
        )
    }
}

#[derive(Clone, Debug)]
pub struct ImportOpts {
    pub name: String,
    pub weights_path: String,
    pub hash: HashAlgo,
    pub chunk_size: usize,
    pub license: Option<String>,
    pub arch: Option<String>,
}

impl Default for ImportOpts {
    fn default() -> Self {
        ImportOpts {
            name: "imported/bnb".into(),
            weights_path: String::new(),
            hash: HashAlgo::default(),
            chunk_size: 1 << 20,
            license: None,
            arch: None,
        }
    }
}

/// The suffixes a quantized weight's companions use.
const PARTS: [&str; 5] = [
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
    "bitsandbytes__nf4",
];

/// What the JSON quant state says.
#[derive(Debug)]
struct State {
    quant: Quant,
    blocksize: u64,
    shape: Vec<u64>,
    out: DType,
    nested_blocksize: u64,
    nested_offset: f64,
}

fn parse_state(raw: &[u8], suffix: &str) -> Res<State> {
    let text = std::str::from_utf8(raw)
        .map_err(|_| Error::Malformed("the quant state is not UTF-8".into()))?;
    let v = crate::json::parse(text.as_bytes())
        .map_err(|e| Error::Malformed(format!("the quant state is not JSON: {e}")))?;
    use crate::json::Value as J;
    let s = |k: &str| v.get(k).and_then(|x| x.as_str().map(str::to_string));
    let n = |k: &str| -> Option<f64> {
        v.get(k).and_then(|x| match x {
            J::F(f) => Some(*f),
            J::U(u) => Some(*u as f64),
            J::I(i) => Some(*i as f64),
            _ => None,
        })
    };
    let quant_type = s("quant_type").unwrap_or_else(|| suffix.to_string());
    let quant = match quant_type.as_str() {
        "nf4" => Quant::Nf4,
        "fp4" => Quant::Fp4,
        other => {
            return Err(Error::Unsupported(format!(
                "bitsandbytes quant_type `{other}`; this build reads nf4, fp4 and int8"
            )))
        }
    };
    let blocksize = n("blocksize")
        .ok_or_else(|| Error::Malformed("the quant state has no `blocksize`".into()))?
        as u64;
    if blocksize == 0 {
        return Err(Error::Malformed("a blocksize of zero".into()));
    }
    let shape = match v.get("shape") {
        Some(J::Array(a)) => a
            .iter()
            .map(|d| match d {
                J::U(u) => Ok(*u),
                J::I(i) if *i >= 0 => Ok(*i as u64),
                J::F(f) if *f >= 0.0 => Ok(*f as u64),
                _ => Err(Error::Malformed("a non-integer dimension".into())),
            })
            .collect::<Res<Vec<u64>>>()?,
        _ => return Err(Error::Malformed("the quant state has no `shape`".into())),
    };
    // The declared output dtype. bitsandbytes writes `float16`, `bfloat16` or
    // `float32`; the stored words are the same either way, and this is the type
    // the dequantized value is *declared* to have, which §05 requires be stated.
    let out = match s("dtype").as_deref() {
        Some("float32") | Some("torch.float32") => DType::F32,
        Some("float16") | Some("torch.float16") => DType::F16,
        Some("bfloat16") | Some("torch.bfloat16") => DType::BF16,
        None => DType::F32,
        Some(other) => {
            return Err(Error::Unsupported(format!(
                "bitsandbytes compute dtype `{other}`"
            )))
        }
    };
    Ok(State {
        quant,
        blocksize,
        shape,
        out,
        nested_blocksize: n("nested_blocksize").unwrap_or(0.0) as u64,
        nested_offset: n("nested_offset").unwrap_or(0.0),
    })
}

/// A `Codebook` object holding the values a checkpoint shipped.
fn codebook(b: &mut ModelBuilder, values: &[f64], dtype: &DType, note: &str) -> crate::expr::Ref {
    let mut bytes = vec![0u8; dtype.packed_bytes(values.len() as u64) as usize];
    for (i, v) in values.iter().enumerate() {
        dtype.encode(&mut bytes, i as u64, *v, crate::dtype::Round::Rne);
    }
    let lit = b.literal(
        &bytes,
        dtype.clone(),
        &[values.len() as u64],
        Layout::default(),
    );
    let obj = Object::structure(
        crate::container::otype::CODEBOOK,
        &Value::map(vec![
            ("t", Value::text("omni.tensor/codebook")),
            ("v", Value::U(1)),
            ("dtype", dtype.to_value()),
            ("entries", Value::U(values.len() as u64)),
            ("dim", Value::U(1)),
            ("values", lit.to_value()),
            ("note", Value::text(note)),
        ]),
    );
    b.object(obj)
}

/// Reads a whole tensor's elements as `f64`.
fn read_all(f: &safetensors::File<'_>, e: &safetensors::Entry) -> Res<Vec<f64>> {
    let bytes = f.tensor(e);
    let n: u64 = e.shape.iter().product();
    (0..n)
        .map(|i| {
            e.dtype
                .decode(bytes, i)
                .ok_or_else(|| Error::Malformed(format!("{}: element {i} is unreadable", e.name)))
        })
        .collect()
}

/// Imports a bitsandbytes checkpoint into an OMNI object graph.
pub fn import(weights: &[u8], opts: &ImportOpts) -> Res<Imported> {
    let f = safetensors::File::parse(weights).map_err(|e| Error::Malformed(e.to_string()))?;
    let hash = opts.hash;

    let mut report = Fidelity {
        format: "bitsandbytes",
        importer: "omni-import-bnb",
        source_path: opts.weights_path.clone(),
        source_digest: hash.digest(weights),
        source_size: weights.len() as u64,
        ..Default::default()
    };

    let find = |name: &str| f.entries.iter().find(|e| e.name == name);

    // A 4-bit weight announces itself by its quant state; an int8 one by an `SCB`
    // beside it. Both are found by looking for the companion rather than by
    // guessing from a dtype, because a `u8` tensor is not evidence of anything.
    let mut four_bit: Vec<(String, String)> = Vec::new();
    let mut int8: Vec<String> = Vec::new();
    for e in &f.entries {
        for suffix in ["nf4", "fp4"] {
            if let Some(stem) = e
                .name
                .strip_suffix(&format!(".quant_state.bitsandbytes__{suffix}"))
            {
                four_bit.push((stem.to_string(), suffix.to_string()));
            }
        }
        if let Some(module) = e.name.strip_suffix(".SCB") {
            let w = format!("{module}.weight");
            if find(&w).is_some_and(|t| t.dtype == DType::I8) {
                int8.push(w);
            }
        }
    }
    four_bit.sort();
    int8.sort();

    let mut b = ModelBuilder::new(opts.name.clone())
        .hash(hash)
        .chunk_size(opts.chunk_size);
    if let Some(l) = &opts.license {
        b = b.license(l.clone());
    }
    if let Some(family) = &opts.arch {
        b = b.arch(family.clone(), Vec::new());
    }

    // Which names a quantized weight has claimed, so the rest import unchanged.
    let mut claimed: Vec<String> = Vec::new();
    for (stem, suffix) in &four_bit {
        claimed.push(stem.clone());
        claimed.push(format!("{stem}.quant_state.bitsandbytes__{suffix}"));
        for p in PARTS.iter().take(4) {
            claimed.push(format!("{stem}.{p}"));
        }
    }
    for w in &int8 {
        claimed.push(w.clone());
        if let Some(m) = w.strip_suffix(".weight") {
            claimed.push(format!("{m}.SCB"));
            claimed.push(format!("{m}.weight_format"));
        }
    }

    let mut plain = 0usize;
    for e in &f.entries {
        if claimed.iter().any(|c| c == &e.name) {
            continue;
        }
        b = b.tensor(TensorSpec {
            name: e.name.clone(),
            shape: e.shape.clone(),
            dtype: e.dtype.clone(),
            axes: None,
            semantic: String::new(),
            data: f.tensor(e).to_vec(),
            layout: Some(safetensors::layout_of(&e.dtype)),
        });
        plain += 1;
    }

    let mut layers = Vec::new();

    for (stem, suffix) in &four_bit {
        let state_name = format!("{stem}.quant_state.bitsandbytes__{suffix}");
        let se = find(&state_name).expect("found by suffix");
        let st = parse_state(f.tensor(se), suffix)?;

        let we = find(stem)
            .ok_or_else(|| Error::Malformed(format!("{state_name} has no `{stem}` beside it")))?;
        let numel: u64 = st.shape.iter().product();
        let packed_bytes = f.tensor(we);
        if packed_bytes.len() as u64 != numel.div_ceil(2) {
            return Err(Error::Malformed(format!(
                "{stem}: {} packed bytes for {numel} 4-bit values",
                packed_bytes.len()
            )));
        }

        // Two nibbles per byte, high first — `MsbFirst` in §04.4's terms. GPTQ's
        // packing is the other way round in 32-bit words, which is why the layout
        // says which rather than assuming.
        let qweight = Expr::Literal {
            chunks: b.chunk_list(packed_bytes),
            dtype: DType::U4,
            shape: dims(&[numel]),
            layout: Layout::Packed {
                elems_per_word: 2,
                word_bits: 8,
                bit_order: BitOrder::MsbFirst,
                order: Order::RowMajor,
            },
        };

        let map_e = find(&format!("{stem}.quant_map"))
            .ok_or_else(|| Error::Malformed(format!("{stem}: no `quant_map`")))?;
        let map = read_all(&f, map_e)?;
        if map.len() != 16 {
            return Err(Error::Malformed(format!(
                "{stem}: quant_map has {} entries, and 4-bit needs 16",
                map.len()
            )));
        }
        let book = codebook(
            &mut b,
            &map,
            &map_e.dtype,
            "the sixteen values this checkpoint shipped, not §05.4's recipe",
        );

        let am_e = find(&format!("{stem}.absmax"))
            .ok_or_else(|| Error::Malformed(format!("{stem}: no `absmax`")))?;
        let blocks = numel.div_ceil(st.blocksize);
        let am_len: u64 = am_e.shape.iter().product();
        if am_len != blocks {
            return Err(Error::Malformed(format!(
                "{stem}: {am_len} scales for {blocks} block(s) of {}",
                st.blocksize
            )));
        }

        let double = am_e.dtype == DType::U8;
        let scale_expr = if double {
            // absmax is itself codebook-quantized, with a second-level scale and
            // a constant offset. Composing that as the outer scheme's `scale` is
            // what makes double quantization need no new formula.
            let nq_e = find(&format!("{stem}.nested_quant_map")).ok_or_else(|| {
                Error::Malformed(format!(
                    "{stem}: absmax is u8, so it is double-quantized, but there is \
                     no `nested_quant_map`"
                ))
            })?;
            let na_e = find(&format!("{stem}.nested_absmax"))
                .ok_or_else(|| Error::Malformed(format!("{stem}: no `nested_absmax`")))?;
            let nmap = read_all(&f, nq_e)?;
            if nmap.len() != 256 {
                return Err(Error::Malformed(format!(
                    "{stem}: nested_quant_map has {} entries, and 8-bit needs 256",
                    nmap.len()
                )));
            }
            let nbook = codebook(&mut b, &nmap, &nq_e.dtype, "the absmax codebook");
            let nested_blocksize = if st.nested_blocksize == 0 {
                return Err(Error::Malformed(format!(
                    "{stem}: double-quantized without a `nested_blocksize`"
                )));
            } else {
                st.nested_blocksize
            };

            let am_q = Expr::Literal {
                chunks: b.chunk_list(f.tensor(am_e)),
                dtype: DType::U8,
                shape: dims(&[blocks]),
                layout: safetensors::layout_of(&DType::U8),
            };
            let na = Expr::Literal {
                chunks: b.chunk_list(f.tensor(na_e)),
                dtype: na_e.dtype.clone(),
                shape: dims(&[na_e.shape.iter().product::<u64>()]),
                layout: safetensors::layout_of(&na_e.dtype),
            };
            let inner = Expr::Dequantize {
                x: Box::new(am_q),
                scheme: Value::map(vec![
                    ("scheme", Value::text("codebook")),
                    ("formula", Value::text("codebook")),
                    ("out", DType::F32.to_value()),
                    ("axis", Value::U(0)),
                    ("block", Value::Array(vec![Value::U(nested_blocksize)])),
                    (
                        "book",
                        Value::Array(vec![
                            Value::U(nbook.0 as u64),
                            Value::Bytes(nbook.1.to_vec()),
                        ]),
                    ),
                    ("scale", na.to_value()),
                ]),
            };
            // `+ nested_offset`: bitsandbytes subtracts the mean of the scales
            // before quantizing them, so dequantizing has to put it back.
            //
            // This addition happens in the evaluator's f64, while bitsandbytes
            // holds the reconstructed absmax as f32. The consequence is measured
            // rather than guessed at: the imported values agree with
            // `bitsandbytes.dequantize_4bit` to within one float32 ULP, and the
            // difference is this evaluator being *more* precise, not less. See
            // `tools/bnb-fixture.py` for the comparison and the tolerance it uses.
            Expr::Bin {
                op: crate::expr::BinOp::Add,
                a: Box::new(inner),
                b: Box::new(Expr::Full {
                    value: Scalar::Float(st.nested_offset),
                    dtype: DType::F32,
                    shape: dims(&[1]),
                }),
            }
        } else {
            Expr::Literal {
                chunks: b.chunk_list(f.tensor(am_e)),
                dtype: am_e.dtype.clone(),
                shape: dims(&[blocks]),
                layout: safetensors::layout_of(&am_e.dtype),
            }
        };

        let deq = Expr::Dequantize {
            x: Box::new(qweight),
            scheme: Value::map(vec![
                ("scheme", Value::text("codebook")),
                ("formula", Value::text("codebook")),
                ("out", st.out.to_value()),
                ("axis", Value::U(0)),
                ("block", Value::Array(vec![Value::U(st.blocksize)])),
                (
                    "book",
                    Value::Array(vec![Value::U(book.0 as u64), Value::Bytes(book.1.to_vec())]),
                ),
                ("scale", scale_expr.to_value()),
            ]),
        };
        // Flat blocking, then the shape: see the module comment.
        let value = Expr::Reshape {
            x: Box::new(deq),
            shape: dims(&st.shape),
        };

        b = b.derived(
            stem.clone(),
            TensorDesc {
                shape: dims(&st.shape),
                dtype: st.out.clone(),
                layout: Layout::default(),
                value,
                semantic: Some("weight".into()),
                role: Some("quantized".into()),
                axes: None,
                device_hint: None,
                materialize: Materialize::Lazy,
                stats: None,
                digest_materialized: None,
            },
        );
        report.represented.push(format!(
            "{stem} ({}, blocksize {}{})",
            st.quant.name(),
            st.blocksize,
            if double { ", double-quantized" } else { "" }
        ));
        layers.push(Layer {
            name: stem.clone(),
            quant: st.quant,
            shape: st.shape.clone(),
            blocksize: st.blocksize,
            double,
        });
    }

    for w in &int8 {
        let module = w.strip_suffix(".weight").expect("named by construction");
        let we = find(w).expect("found by construction");
        let scb_e = find(&format!("{module}.SCB")).expect("found by construction");
        if we.shape.len() != 2 {
            return Err(Error::Unsupported(format!(
                "{w}: LLM.int8 weights are matrices, and this is {:?}",
                we.shape
            )));
        }
        let rows = we.shape[0];
        let cols = we.shape[1];
        if scb_e.shape.iter().product::<u64>() != rows {
            return Err(Error::Malformed(format!(
                "{w}: SCB has {} entries for {rows} row(s)",
                scb_e.shape.iter().product::<u64>()
            )));
        }
        let q = Expr::Literal {
            chunks: b.chunk_list(f.tensor(we)),
            dtype: DType::I8,
            shape: dims(&[rows, cols]),
            layout: safetensors::layout_of(&DType::I8),
        };
        // `w = q · SCB / 127`. The division is folded into the scale rather than
        // applied afterwards, so the scheme is §05's plain `sym` and the stored
        // words keep their meaning.
        let scb = Expr::Scale {
            x: Box::new(Expr::Reshape {
                x: Box::new(Expr::Literal {
                    chunks: b.chunk_list(f.tensor(scb_e)),
                    dtype: scb_e.dtype.clone(),
                    shape: dims(&[rows]),
                    layout: safetensors::layout_of(&scb_e.dtype),
                }),
                shape: dims(&[rows, 1]),
            }),
            k: Scalar::Float(1.0 / 127.0),
        };
        let value = Expr::Dequantize {
            x: Box::new(q),
            scheme: Value::map(vec![
                ("scheme", Value::text("sym")),
                ("formula", Value::text("sym")),
                ("out", DType::F32.to_value()),
                ("axis", Value::U(0)),
                ("block", Value::Array(vec![Value::U(1), Value::U(cols)])),
                ("scale", scb.to_value()),
                ("sym", Value::Bool(true)),
            ]),
        };
        b = b.derived(
            w.clone(),
            TensorDesc {
                shape: dims(&[rows, cols]),
                dtype: DType::F32,
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
        report.represented.push(format!("{w} (int8, per-row SCB)"));
        layers.push(Layer {
            name: w.clone(),
            quant: Quant::Int8,
            shape: vec![rows, cols],
            blocksize: cols,
            double: false,
        });
    }

    if layers.is_empty() {
        return Err(Error::Malformed(
            "no bitsandbytes weights here: a 4-bit one is named by a \
             `quant_state.bitsandbytes__nf4` or `__fp4` tensor, and an int8 one by \
             an `SCB` beside an `i8` weight"
                .into(),
        ));
    }

    // §12's honesty rule applied to what bitsandbytes does not record. The
    // checkpoint says how the weights were quantized and nothing about what they
    // were quantized *from*, so the original is not recoverable and the report
    // says so rather than implying a round trip exists.
    report.assumptions.push(Note {
        item: "the unquantized weights".into(),
        reason: "a bitsandbytes checkpoint holds only the quantized words; the \
                 f16/bf16 originals are not in the file"
            .into(),
        action: "imported as a lossless representation of the quantized values, \
                 which is what the file contains"
            .into(),
    });
    if int8.iter().any(|w| {
        w.strip_suffix(".weight")
            .is_some_and(|m| find(&format!("{m}.weight_format")).is_some())
    }) {
        report.assumptions.push(Note {
            item: "`weight_format`".into(),
            reason: "a bitsandbytes marker for the kernel layout a GPU expects, \
                     not part of the tensor's value"
                .into(),
            action: "dropped; the imported weight is in row-major order".into(),
        });
    }
    // Lossless in the sense §01.1 means: every stored word — the packed nibbles,
    // the block scales, both codebooks — is in the graph verbatim, so the file is
    // reproducible from it. It does not mean the *evaluated* values are bit-equal
    // to bitsandbytes': this evaluator works in f64 where the library works in
    // f32, so the dequantized numbers agree to within one float32 ULP and are the
    // more precise of the two. That is measured, in CI, against the library.
    report.lossless = true;
    report.represented.push(format!("{plain} plain tensor(s)"));

    let (objects, root) = b.build();
    Ok(Imported {
        objects,
        root,
        report,
        layers,
        plain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::otype;
    use crate::expr::Ctx;
    use crate::tensor::TensorTable;

    /// A named tensor, evaluated back out of the imported object graph — the same
    /// path a reader takes, rather than the expression this module just built.
    fn read(imp: &Imported, name: &str) -> crate::expr::Tensor {
        let hash = HashAlgo::default();
        let mut mem = crate::store::MemoryStore::new(hash);
        for o in &imp.objects {
            let _ = crate::store::WritableStore::put(&mut mem, &o.payload);
        }
        let ctx = Ctx::new(&mem);
        let tref = imp
            .objects
            .iter()
            .find(|o| o.otype == otype::TENSOR_TABLE)
            .map(|o| (otype::TENSOR_TABLE, o.digest(hash)))
            .expect("a tensor table");
        let table = TensorTable::load(&ctx, &tref).expect("loads");
        let d = TensorDesc::load(&ctx, table.get(name).expect(name)).expect("desc");
        d.value.eval(&ctx).expect("evaluates")
    }

    /// Builds a safetensors file in memory.
    fn st(entries: &[(&str, DType, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut at = 0usize;
        for (i, (name, dtype, shape, data)) in entries.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            // An empty shape means "one dimension, as long as the data" — which
            // is how the quant-state blob is written, and stops each test from
            // having to count the JSON's bytes.
            let owned;
            let shape: &Vec<u64> = if shape.is_empty() {
                owned = vec![data.len() as u64];
                &owned
            } else {
                shape
            };
            let dt = match dtype {
                d if *d == DType::U8 => "U8",
                d if *d == DType::I8 => "I8",
                d if *d == DType::F32 => "F32",
                d if *d == DType::F16 => "F16",
                _ => panic!("unhandled dtype in the test builder"),
            };
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dt}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                dims.join(","),
                at,
                at + data.len()
            ));
            at += data.len();
        }
        header.push('}');
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        for (_, _, _, d) in entries {
            out.extend_from_slice(d);
        }
        out
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The sixteen NF4 values, as bitsandbytes ships them.
    const NF4: [f32; 16] = [
        -1.0,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_04,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ];

    /// Packs 4-bit indices two per byte, high nibble first.
    fn pack(idx: &[u8]) -> Vec<u8> {
        idx.chunks(2)
            .map(|c| (c[0] << 4) | c.get(1).copied().unwrap_or(0))
            .collect()
    }

    #[test]
    fn nf4_without_double_quantization_dequantizes_to_book_times_absmax() {
        // Four elements, one block of 4, so the arithmetic is checkable by hand.
        let idx = [0u8, 7, 15, 8];
        let absmax = [2.0f32];
        let file = st(&[
            ("w", DType::U8, vec![2, 1], pack(&idx)),
            ("w.absmax", DType::F32, vec![1], f32s(&absmax)),
            ("w.quant_map", DType::F32, vec![16], f32s(&NF4)),
            (
                "w.quant_state.bitsandbytes__nf4",
                DType::U8,
                vec![],
                br#"{"quant_type":"nf4","blocksize":4,"dtype":"float32","shape":[2,2]}"#.to_vec(),
            ),
        ]);
        let imp = import(&file, &ImportOpts::default()).expect("imports");
        assert_eq!(imp.layers.len(), 1);
        assert_eq!(imp.layers[0].quant, Quant::Nf4);
        assert!(!imp.layers[0].double);

        let t = read(&imp, "w");
        let want: Vec<f64> = idx.iter().map(|i| NF4[*i as usize] as f64 * 2.0).collect();
        assert_eq!(t.shape, vec![2, 2]);
        for (got, w) in t.data.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
    }

    #[test]
    fn double_quantization_composes_as_a_scale_expression() {
        // absmax is u8 through a 256-entry book with its own scale and an offset,
        // which is the case that would need a new formula if `scale` were a
        // tensor rather than an expression.
        let idx = [15u8, 0, 8, 7];
        let mut nmap = [0.0f32; 256];
        for (i, v) in nmap.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        let file = st(&[
            ("w", DType::U8, vec![2, 1], pack(&idx)),
            ("w.absmax", DType::U8, vec![1], vec![128]),
            ("w.quant_map", DType::F32, vec![16], f32s(&NF4)),
            ("w.nested_quant_map", DType::F32, vec![256], f32s(&nmap)),
            ("w.nested_absmax", DType::F32, vec![1], f32s(&[4.0])),
            (
                "w.quant_state.bitsandbytes__nf4",
                DType::U8,
                vec![],
                br#"{"quant_type":"nf4","blocksize":4,"dtype":"float32","shape":[4],
                     "nested_blocksize":256,"nested_dtype":"float32",
                     "nested_offset":0.25}"#
                    .to_vec(),
            ),
        ]);
        let imp = import(&file, &ImportOpts::default()).expect("imports");
        assert!(imp.layers[0].double);

        let t = read(&imp, "w");
        let absmax = nmap[128] as f64 * 4.0 + 0.25;
        for (got, i) in t.data.iter().zip(idx.iter()) {
            let want = NF4[*i as usize] as f64 * absmax;
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn int8_dequantizes_by_its_row_scale() {
        let q: Vec<u8> = vec![127i8 as u8, (-64i8) as u8, 0, 32];
        let file = st(&[
            ("m.weight", DType::I8, vec![2, 2], q),
            ("m.SCB", DType::F32, vec![2], f32s(&[1.0, 2.0])),
        ]);
        let imp = import(&file, &ImportOpts::default()).expect("imports");
        assert_eq!(imp.layers[0].quant, Quant::Int8);

        let t = read(&imp, "m.weight");
        let want = [127.0 / 127.0, -64.0 / 127.0, 0.0, 32.0 * 2.0 / 127.0];
        for (got, w) in t.data.iter().zip(want.iter()) {
            assert!((got - w).abs() < 1e-6, "got {got}, want {w}");
        }
    }

    #[test]
    fn tensors_no_quantized_weight_claims_are_imported_unchanged() {
        let file = st(&[
            ("w", DType::U8, vec![1, 1], pack(&[0, 15])),
            ("w.absmax", DType::F32, vec![1], f32s(&[1.0])),
            ("w.quant_map", DType::F32, vec![16], f32s(&NF4)),
            (
                "w.quant_state.bitsandbytes__nf4",
                DType::U8,
                vec![],
                br#"{"quant_type":"nf4","blocksize":2,"dtype":"float32","shape":[2]}"#.to_vec(),
            ),
            ("norm.weight", DType::F32, vec![2], f32s(&[1.5, 2.5])),
        ]);
        let imp = import(&file, &ImportOpts::default()).expect("imports");
        assert_eq!(imp.plain, 1, "the norm is plain and the companions are not");
        assert!(imp.report.lossless);
    }

    #[test]
    fn a_file_with_nothing_quantized_is_refused_by_name() {
        let file = st(&[("norm.weight", DType::F32, vec![1], f32s(&[1.0]))]);
        let e = import(&file, &ImportOpts::default()).expect_err("no bnb weights");
        assert!(
            format!("{e}").contains("quant_state.bitsandbytes__nf4"),
            "{e}"
        );
    }

    #[test]
    fn an_unknown_quant_type_is_unsupported_rather_than_guessed() {
        let file = st(&[
            ("w", DType::U8, vec![1, 1], pack(&[0, 1])),
            ("w.absmax", DType::F32, vec![1], f32s(&[1.0])),
            ("w.quant_map", DType::F32, vec![16], f32s(&NF4)),
            (
                "w.quant_state.bitsandbytes__nf4",
                DType::U8,
                vec![],
                br#"{"quant_type":"nf8","blocksize":2,"dtype":"float32","shape":[2]}"#.to_vec(),
            ),
        ]);
        let e = import(&file, &ImportOpts::default()).expect_err("nf8");
        assert!(matches!(e, Error::Unsupported(_)), "{e}");
        assert!(format!("{e}").contains("nf8"), "{e}");
    }
}
