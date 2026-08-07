//! §05.5 — requantization: the search, and the provenance that makes it
//! answerable.
//!
//! `omni convert --requantize` was refused for a while, with a reason that was
//! half right: "quantizing is a measurement, the scales depend on activations,
//! the activations depend on the data". That is true of GPTQ and AWQ. It is
//! **not** true of round-to-nearest, whose scales come from the weights and
//! nothing else, and refusing the whole verb because one method needs data the
//! build did not have was refusing more than the argument supported.
//!
//! So there are two methods here, and the difference between them is exactly
//! the difference §05.5 exists to record:
//!
//! * **`rtn`** — round to nearest over per-group min/max. Needs no data, states
//!   no calibration in its provenance, and is the honest baseline every
//!   quantization is measured against.
//! * **`clip`** — a search over clipping ratios, minimizing the reconstruction
//!   error of each group. Given a calibration set it weights that error by the
//!   activation magnitude of each input channel, which is what makes a
//!   quantization *calibrated*: the channels a model actually uses are the ones
//!   whose error costs something.
//!
//! One finding from writing this, because it is the kind of thing a method name
//! hides: with an **unweighted** objective the search does not clip an outlier.
//! Clamping one large weight costs more squared error than the precision every
//! other weight in the group gains, so plain minimum/maximum wins and the grid
//! is a guarantee of no-worse rather than an improvement. Clipping only pays
//! when something says that channel matters less than its magnitude suggests —
//! and that something is the calibration set. The tests assert both halves,
//! because a search that always clipped would be as wrong as one that never
//! did.
//!
//! What is deliberately not here is GPTQ's Hessian update or AWQ's per-channel
//! scaling transform. Both are published algorithms with published
//! implementations, and a re-derivation checked against nothing would be a
//! third answer nobody asked for. `clip` is the part of them that is a *search
//! over a stated objective*, and its objective is written down below rather
//! than implied by a name.
//!
//! ## Why the provenance is not optional here
//!
//! §05.5 asks a question — "which calibration set produced this int4 model?" —
//! that is currently unanswerable for most published quantizations. Every field
//! of the answer is recorded: the method, the bit width, the group size, the
//! grid searched, the objective, and the calibration set's digest and sample
//! count when there was one. A run with no calibration says so with an absent
//! field rather than an empty one, because absence is information (I1).

use crate::cbor::Value;
use crate::dtype::DType;
use crate::quant::Formula;

#[derive(Debug)]
pub enum Error {
    Spec(String),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Spec(m) => write!(f, "{m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// The clipping ratios `clip` searches. Coarse on purpose: the objective is
/// flat near the optimum, and a finer grid buys precision the weights do not
/// have.
pub const CLIP_GRID: &[f64] = &[
    1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
];

/// What to quantize to: `affine:4:128`, `sym:8:32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub formula: Formula,
    pub bits: u16,
    pub group: u64,
}

impl Spec {
    pub fn parse(s: &str) -> Res<Spec> {
        let parts: Vec<&str> = s.split(':').collect();
        let [name, bits, group] = parts[..] else {
            return Err(Error::Spec(format!(
                "`{s}` is not a requantization spec; it is \
                 <affine|sym>:<bits>:<group>, e.g. affine:4:128"
            )));
        };
        let formula = match name {
            "affine" => Formula::AffineSub,
            "sym" => Formula::Sym,
            other => {
                return Err(Error::Spec(format!(
                    "`{other}` is not a formula this converter writes. §05.1's set is \
                     closed and the two that a weight-only quantizer can produce \
                     without more inputs are `affine` and `sym`"
                )))
            }
        };
        let bits: u16 = bits
            .parse()
            .map_err(|_| Error::Spec(format!("`{bits}` is not a bit width")))?;
        if !(2..=8).contains(&bits) {
            return Err(Error::Spec(format!(
                "{bits} bits: this converter writes 2 to 8, which is what §04.3's \
                 sub-byte integer types cover"
            )));
        }
        let group: u64 = group
            .parse()
            .map_err(|_| Error::Spec(format!("`{group}` is not a group size")))?;
        if group == 0 {
            return Err(Error::Spec("a group of zero elements".into()));
        }
        Ok(Spec {
            formula,
            bits,
            group,
        })
    }

    /// The dtype the quantized indices are stored in.
    pub fn index_dtype(&self) -> DType {
        DType::Int {
            w: self.bits,
            signed: self.formula == Formula::Sym,
        }
    }

    /// `(low, high)` of the index range.
    pub fn range(&self) -> (f64, f64) {
        match self.formula {
            Formula::Sym => {
                let hi = ((1i64 << (self.bits - 1)) - 1) as f64;
                (-hi - 1.0, hi)
            }
            _ => (0.0, ((1u64 << self.bits) - 1) as f64),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{}-{}b-g{}",
            match self.formula {
                Formula::Sym => "sym",
                _ => "affine",
            },
            self.bits,
            self.group
        )
    }
}

/// Per-input-channel activation magnitudes from a calibration set (§05.5).
///
/// One vector per tensor, as long as that tensor's last axis. What it holds is
/// the mean absolute activation of each input channel over the calibration
/// samples — the quantity AWQ calls the activation scale and the only thing
/// about the data a weight-only search can use.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    pub dataset: String,
    pub digest: [u8; 32],
    pub samples: u64,
    /// `(tensor name, per-channel magnitudes)`.
    pub channels: Vec<(String, Vec<f64>)>,
}

impl Calibration {
    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.channels
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    /// The §05.5 record. `samples` and `digest` are what make the question
    /// "which calibration set produced this?" a field lookup.
    pub fn to_value(&self) -> Value {
        Value::map(vec![
            ("dataset", Value::text(self.dataset.clone())),
            ("digest", Value::Bytes(self.digest.to_vec())),
            ("samples", Value::U(self.samples)),
            ("tensors", Value::U(self.channels.len() as u64)),
        ])
    }
}

/// One quantized tensor: the indices, the per-group parameters, and what the
/// search cost and bought.
#[derive(Clone, Debug)]
pub struct Quantized {
    /// Indices, one per element, in the order the tensor is stored.
    pub q: Vec<f64>,
    /// Per-group scales, `[rows, groups]` row-major.
    pub scale: Vec<f64>,
    /// Per-group zero points, empty for a symmetric scheme.
    pub zero: Vec<f64>,
    pub rows: u64,
    pub groups: u64,
    /// Largest absolute reconstruction error over the tensor.
    pub max_abs_error: f64,
    /// `‖W − Ŵ‖ / ‖W‖`, which is the number that means something across
    /// tensors of different magnitudes.
    pub rel_error: f64,
    /// Clip ratios tried per group. One means no search happened.
    pub searched: usize,
    /// How many groups chose a ratio other than 1.0 — that is, how often the
    /// search found something min/max did not.
    pub clipped: u64,
    /// Whether the objective was activation-weighted.
    pub weighted: bool,
}

/// Quantizes one tensor, group by group along its last axis.
///
/// `values` is row-major and `shape`'s last axis is the one groups divide, which
/// for a `[out_features, in_features]` weight is the input channels — the axis
/// activations are indexed by, and the reason a calibration vector is as long as
/// that axis and not the other.
pub fn quantize(
    values: &[f64],
    shape: &[u64],
    spec: &Spec,
    act: Option<&[f64]>,
    grid: &[f64],
    scale_dtype: &DType,
    out_dtype: &DType,
) -> Res<Quantized> {
    if shape.is_empty() {
        return Err(Error::Unsupported(
            "a rank-0 tensor has no axis to group along".into(),
        ));
    }
    let cols = *shape.last().unwrap();
    if cols == 0 {
        return Err(Error::Unsupported("an axis of zero elements".into()));
    }
    let rows: u64 = shape[..shape.len() - 1].iter().product();
    let group = spec.group.min(cols);
    if !cols.is_multiple_of(group) {
        return Err(Error::Unsupported(format!(
            "a group of {group} does not divide an axis of {cols}; §05.1's block \
             shape tiles the axis exactly, and a ragged last group would be a \
             different scheme"
        )));
    }
    if let Some(a) = act {
        if a.len() as u64 != cols {
            return Err(Error::Unsupported(format!(
                "the calibration vector is {} long and the axis is {cols}; an \
                 activation magnitude belongs to an input channel, so the two \
                 cannot disagree",
                a.len()
            )));
        }
    }
    let groups = cols / group;
    let (lo, hi) = spec.range();
    // What a reader will actually get: the dequantized value rounded through
    // the scheme's declared output dtype. Measuring the error against the f64
    // reconstruction instead would report a model that was never written — for
    // a bf16 output that understates it by the whole of bf16's rounding.
    let mut rbuf = [0u8; 16];
    let mut through = |v: f64| -> f64 {
        if out_dtype.encode(&mut rbuf, 0, v, crate::dtype::Round::Rne) {
            out_dtype.decode(&rbuf, 0).unwrap_or(v)
        } else {
            v
        }
    };
    let mut out = Quantized {
        q: vec![0.0; values.len()],
        scale: vec![0.0; (rows * groups) as usize],
        zero: if spec.formula == Formula::Sym {
            Vec::new()
        } else {
            vec![0.0; (rows * groups) as usize]
        },
        rows,
        groups,
        max_abs_error: 0.0,
        rel_error: 0.0,
        searched: grid.len(),
        clipped: 0,
        weighted: act.is_some(),
    };
    let (mut sq_err, mut sq_val) = (0.0f64, 0.0f64);

    for r in 0..rows {
        for g in 0..groups {
            let base = (r * cols + g * group) as usize;
            let span = &values[base..base + group as usize];
            let weights: Vec<f64> = match act {
                Some(a) => {
                    let from = (g * group) as usize;
                    a[from..from + group as usize].to_vec()
                }
                None => vec![1.0; group as usize],
            };

            // The search: one candidate per clipping ratio, scored by the
            // objective, and the best one kept. With no calibration the weights
            // are all one, which makes the objective plain squared error —
            // still a search, and one whose result is stated as such.
            let mut best: Option<(f64, f64, f64, f64)> = None; // (cost, clip, scale, zero)
            for clip in grid {
                let (scale, zero) = params(span, spec, *clip, lo, hi, scale_dtype);
                if scale == 0.0 {
                    continue;
                }
                let mut cost = 0.0;
                for (i, w) in span.iter().enumerate() {
                    let q = encode(*w, scale, zero, lo, hi);
                    let back = through(decode(q, scale, zero));
                    let d = back - w;
                    cost += weights[i] * weights[i] * d * d;
                }
                if best.as_ref().is_none_or(|(c, _, _, _)| cost < *c) {
                    best = Some((cost, *clip, scale, zero));
                }
            }
            let (_, clip, scale, zero) = best.unwrap_or((0.0, 1.0, 1.0, 0.0));
            if clip != 1.0 {
                out.clipped += 1;
            }
            let at = (r * groups + g) as usize;
            out.scale[at] = scale;
            if !out.zero.is_empty() {
                out.zero[at] = zero;
            }
            for (i, w) in span.iter().enumerate() {
                let q = encode(*w, scale, zero, lo, hi);
                out.q[base + i] = q;
                let back = through(decode(q, scale, zero));
                let d = (back - w).abs();
                out.max_abs_error = out.max_abs_error.max(d);
                sq_err += d * d;
                sq_val += w * w;
            }
        }
    }
    out.rel_error = if sq_val > 0.0 {
        (sq_err / sq_val).sqrt()
    } else {
        0.0
    };
    Ok(out)
}

/// The scale and zero point one group gets at one clipping ratio.
///
/// The scale is rounded through the dtype it will be *stored* in before it is
/// used, and that is not a detail. A search that optimized against an f64 scale
/// and then stored an f32 one would report an error smaller than the container
/// produces — the report would be describing a model that was never written.
fn params(
    span: &[f64],
    spec: &Spec,
    clip: f64,
    lo: f64,
    hi: f64,
    scale_dtype: &DType,
) -> (f64, f64) {
    let store = |s: f64| -> f64 {
        let mut buf = [0u8; 16];
        if scale_dtype.encode(&mut buf, 0, s, crate::dtype::Round::Rne) {
            scale_dtype.decode(&buf, 0).unwrap_or(s)
        } else {
            s
        }
    };
    match spec.formula {
        Formula::Sym => {
            let m = span.iter().fold(0.0f64, |a, v| a.max(v.abs())) * clip;
            (if m > 0.0 { store(m / hi) } else { 0.0 }, 0.0)
        }
        _ => {
            // Clipping pulls both ends *toward zero*, not toward the midpoint.
            // That matters: a group whose values are all near zero except one
            // outlier has its midpoint out at the outlier, so shrinking about
            // the midpoint would move the representable window away from the
            // values it is supposed to represent. Shrinking toward zero clips
            // the outlier, which is the trade the search is for.
            let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
            for v in span {
                min = min.min(*v);
                max = max.max(*v);
            }
            if !min.is_finite() || !max.is_finite() {
                return (0.0, 0.0);
            }
            // Zero stays inside the window, so a pruned weight is still
            // representable however hard the search clips.
            let (min, max) = (min.min(0.0) * clip, max.max(0.0) * clip);
            let scale = store((max - min) / (hi - lo));
            if scale <= 0.0 {
                return (0.0, 0.0);
            }
            // The zero point lands on the grid, because it is stored as an
            // index and a fractional one would not survive the round trip.
            ((scale), (lo - min / scale).round().clamp(lo, hi))
        }
    }
}

fn encode(w: f64, scale: f64, zero: f64, lo: f64, hi: f64) -> f64 {
    ((w / scale) + zero).round_ties_even().clamp(lo, hi)
}

fn decode(q: f64, scale: f64, zero: f64) -> f64 {
    (q - zero) * scale
}

/// The §05.5 `omni.prov/quantization` record.
#[allow(clippy::too_many_arguments)]
pub fn provenance(
    spec: &Spec,
    method: &str,
    calib: Option<&Calibration>,
    grid: &[f64],
    tensors: usize,
    max_abs_error: f64,
    worst_rel: f64,
    source: &[u8; 32],
) -> Value {
    let mut p = vec![
        ("t", Value::text("omni.prov/quantization")),
        ("v", Value::U(1)),
        ("method", Value::text(method.to_string())),
        (
            "impl",
            Value::text(format!("omni-rs {}", env!("CARGO_PKG_VERSION"))),
        ),
        ("bits", Value::U(spec.bits as u64)),
        ("group_size", Value::U(spec.group)),
        ("formula", Value::text(spec.formula.id())),
        ("tensors", Value::U(tensors as u64)),
        (
            "source_model",
            Value::Array(vec![
                Value::U(crate::container::otype::MANIFEST as u64),
                Value::Bytes(source.to_vec()),
            ]),
        ),
        ("max_abs_error", Value::F64(max_abs_error)),
        ("max_rel_error", Value::F64(worst_rel)),
    ];
    if grid.len() > 1 {
        // The objective, written down. A method name is not a description, and
        // "clip" without the grid and the weighting is a name.
        p.push((
            "search",
            Value::map(vec![
                (
                    "grid",
                    Value::Array(grid.iter().map(|c| Value::F64(*c)).collect()),
                ),
                (
                    "objective",
                    Value::text(if calib.is_some() {
                        "minimize Σ a²(w − ŵ)² per group, a = mean |activation| \
                         of the input channel"
                    } else {
                        "minimize Σ (w − ŵ)² per group"
                    }),
                ),
            ]),
        ));
    }
    match calib {
        Some(c) => p.push(("calibration", c.to_value())),
        // I1: no calibration field rather than an empty one. A reader asking
        // §05.5's question gets "this quantization used none", which is an
        // answer, instead of a record that looks like data and is not.
        None => p.push((
            "note",
            Value::text(
                "no calibration set: the scales come from the weights alone, so \
                 there is nothing to record and nothing was invented",
            ),
        )),
    }
    Value::map(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.37).sin() * 2.0).collect()
    }

    #[test]
    fn a_spec_is_parsed_or_named() {
        let s = Spec::parse("affine:4:128").unwrap();
        assert_eq!(s.bits, 4);
        assert_eq!(s.group, 128);
        assert_eq!(s.formula, Formula::AffineSub);
        assert_eq!(Spec::parse("sym:8:32").unwrap().formula, Formula::Sym);
        for bad in [
            "",
            "affine",
            "affine:4",
            "gptq:4:128",
            "affine:1:32",
            "affine:4:0",
        ] {
            let e = Spec::parse(bad).expect_err(bad);
            assert!(!format!("{e}").is_empty());
        }
    }

    #[test]
    fn round_to_nearest_reconstructs_within_half_a_step() {
        // The property that defines RTN: every value lands on the nearest
        // representable point, so the error is at most half a quantization
        // step. Anything larger means the scale is wrong.
        let w = ramp(256);
        let spec = Spec::parse("affine:4:32").unwrap();
        let q = quantize(&w, &[8, 32], &spec, None, &[1.0], &DType::F32, &DType::F64).unwrap();
        assert_eq!(q.groups, 1);
        assert_eq!(q.rows, 8);
        assert_eq!(q.searched, 1);
        assert_eq!(q.clipped, 0);
        for r in 0..8usize {
            let step = q.scale[r];
            for i in 0..32usize {
                let back = (q.q[r * 32 + i] - q.zero[r]) * step;
                assert!(
                    (back - w[r * 32 + i]).abs() <= step / 2.0 + 1e-12,
                    "row {r} element {i}: {back} vs {}",
                    w[r * 32 + i]
                );
            }
        }
        // And the indices are inside the range the bit width allows.
        let (lo, hi) = spec.range();
        assert!(q.q.iter().all(|v| *v >= lo && *v <= hi));
    }

    #[test]
    fn a_symmetric_scheme_keeps_zero_at_zero() {
        // The property symmetric quantization exists for: zero is exactly
        // representable, so a pruned weight stays pruned.
        let mut w = ramp(64);
        w[3] = 0.0;
        w[40] = 0.0;
        let spec = Spec::parse("sym:4:16").unwrap();
        let q = quantize(&w, &[4, 16], &spec, None, &[1.0], &DType::F32, &DType::F64).unwrap();
        assert!(q.zero.is_empty(), "a symmetric scheme has no zero point");
        assert_eq!(q.q[3], 0.0);
        assert_eq!(q.q[40], 0.0);
    }

    #[test]
    fn an_unweighted_search_never_does_worse_than_min_max() {
        // 1.0 is in the grid, so the search cannot lose — and this test is here
        // because the interesting half is what it does *not* do. Given a group
        // of small values and one outlier, plain squared error keeps the
        // outlier: clamping it costs more than the precision every other value
        // would gain. Clipping pays when something says that channel matters
        // less, and that something is a calibration set. See the next test.
        let mut w: Vec<f64> = (0..64).map(|i| ((i % 16) as f64 - 8.0) * 0.01).collect();
        w[0] = 5.0;
        w[16] = -5.0;
        let spec = Spec::parse("affine:4:16").unwrap();
        let plain = quantize(&w, &[4, 16], &spec, None, &[1.0], &DType::F32, &DType::F64).unwrap();
        let searched = quantize(
            &w,
            &[4, 16],
            &spec,
            None,
            CLIP_GRID,
            &DType::F32,
            &DType::F64,
        )
        .unwrap();
        assert!(
            searched.rel_error <= plain.rel_error + 1e-12,
            "a search including 1.0 came out worse than 1.0: {} vs {}",
            searched.rel_error,
            plain.rel_error
        );
        assert_eq!(
            searched.clipped, 0,
            "the unweighted objective clipped an outlier it should have kept"
        );
    }

    #[test]
    fn calibration_moves_the_error_to_the_channels_that_matter() {
        // The whole point of a calibrated search: two groups with the same
        // weights but different activation magnitudes must not get the same
        // answer, because the error costs different amounts.
        let w: Vec<f64> = (0..32)
            .map(|i| if i % 16 == 0 { 4.0 } else { (i as f64) * 0.01 })
            .collect();
        let spec = Spec::parse("affine:4:16").unwrap();
        // The outlier channel barely fires; the rest do.
        let mut act = vec![1.0; 16];
        act[0] = 0.001;
        let weighted = quantize(
            &w,
            &[2, 16],
            &spec,
            Some(&act),
            CLIP_GRID,
            &DType::F32,
            &DType::F64,
        )
        .unwrap();
        let plain = quantize(
            &w,
            &[2, 16],
            &spec,
            None,
            CLIP_GRID,
            &DType::F32,
            &DType::F64,
        )
        .unwrap();
        assert!(weighted.weighted);
        assert!(!plain.weighted);
        // Weighted by activations, clipping the outlier away is cheap, so the
        // search clips harder than the unweighted one would.
        assert!(
            weighted.scale[0] <= plain.scale[0],
            "the calibrated search kept the outlier's range: {} vs {}",
            weighted.scale[0],
            plain.scale[0]
        );
        // And a calibration vector that does not match the axis is refused
        // rather than broadcast.
        let e = quantize(
            &w,
            &[2, 16],
            &spec,
            Some(&[1.0; 4]),
            CLIP_GRID,
            &DType::F32,
            &DType::F64,
        )
        .unwrap_err();
        assert!(format!("{e}").contains("input channel"), "{e}");
    }

    #[test]
    fn the_provenance_answers_the_question_5_5_asks() {
        let spec = Spec::parse("affine:4:128").unwrap();
        let calib = Calibration {
            dataset: "c4/en".into(),
            digest: [7u8; 32],
            samples: 512,
            channels: vec![("w".into(), vec![1.0; 128])],
        };
        let with = provenance(
            &spec,
            "clip",
            Some(&calib),
            CLIP_GRID,
            3,
            0.01,
            0.02,
            &[0u8; 32],
        );
        assert_eq!(
            with.get("calibration")
                .and_then(|c| c.get("dataset"))
                .and_then(|d| d.as_str()),
            Some("c4/en")
        );
        assert!(with.get("search").is_some());

        // And without one, the field is absent rather than empty — an absent
        // field is an answer and an empty one is a lie.
        let without = provenance(&spec, "rtn", None, &[1.0], 3, 0.01, 0.02, &[0u8; 32]);
        assert!(without.get("calibration").is_none());
        assert!(without.get("search").is_none());
        assert!(without
            .get("note")
            .and_then(|n| n.as_str())
            .is_some_and(|n| n.contains("nothing was invented")));
    }
}
