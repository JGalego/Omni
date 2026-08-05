//! §04.6 — sparsity schemes.
//!
//! Sparse tensors are not a separate tensor kind: they are values produced by a
//! `sparse` expression node, which is why a pruned fine-tune of a dense base is
//! just `add(base, sparse(…))` and the delta costs only its non-zeros (§08.6).
//!
//! This module densifies each scheme in the table of §04.6. Every one of them
//! validates its own structure — an index out of range, an `indptr` that is not
//! monotone, a values array whose length disagrees with the mask — because a
//! sparse encoding that is read optimistically is a sparse encoding that
//! produces silently wrong weights.

use crate::cbor::Value;
use crate::dtype::DType;
use crate::expr::{Error, Tensor};
use crate::layout::numel;

type Res<T> = Result<T, Error>;

/// The schemes of §04.6.
pub const SCHEMES: &[&str] = &[
    "coo",
    "csr",
    "csc",
    "bsr",
    "nm",
    "bitmask",
    "ragged",
    "blocklist",
];

/// Materializes a sparse value into a dense tensor.
///
/// `parts` holds the already-evaluated component tensors under the names the
/// node used (`values`, `indices`, `indptr`, `mask`, `offsets`, `blocks`,
/// `index`); `attrs` holds the scheme's scalar parameters (`n`, `m`, `block`).
pub fn densify(
    scheme: &str,
    parts: &[(&str, Tensor)],
    attrs: &Value,
    shape: &[u64],
    dtype: &DType,
    fill: f64,
) -> Res<Tensor> {
    let n = numel(shape);
    let get = |name: &str| -> Res<&Tensor> {
        parts
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, t)| t)
            .ok_or_else(|| Error::Type(format!("sparse `{scheme}` needs `{name}`")))
    };
    let attr_u = |name: &str| attrs.get(name).and_then(|x| x.as_u64());
    let attr_uvec = |name: &str| -> Option<Vec<u64>> {
        attrs
            .get(name)?
            .as_array()?
            .iter()
            .map(|x| x.as_u64())
            .collect()
    };
    let mut out = vec![fill; n as usize];
    let strides = crate::layout::Order::RowMajor.strides(shape);

    match scheme {
        // General: one index per dimension per non-zero.
        "coo" => {
            let idx = get("indices")?;
            let vals = get("values")?;
            let nnz = vals.numel();
            let rank = shape.len() as u64;
            if idx.numel() != nnz * rank {
                return Err(Error::Type(format!(
                    "coo: `indices` holds {} entries; {nnz} non-zeros of a rank-{rank} tensor \
                     need {}",
                    idx.numel(),
                    nnz * rank
                )));
            }
            for k in 0..nnz {
                let mut lin = 0u64;
                for d in 0..rank {
                    // Indices are stored as (n_dims x nnz), so dimension `d`'s
                    // index for non-zero `k` is at row d, column k.
                    let v = idx.data[(d * nnz + k) as usize];
                    let i = check_index(v, shape[d as usize], "coo index")?;
                    lin += i * strides[d as usize];
                }
                out[lin as usize] = vals.data[k as usize];
            }
        }
        // Classic compressed rows/columns.
        "csr" | "csc" => {
            require_rank2(scheme, shape)?;
            let indptr = get("indptr")?;
            let indices = get("indices")?;
            let vals = get("values")?;
            let (major, minor) = if scheme == "csr" {
                (shape[0], shape[1])
            } else {
                (shape[1], shape[0])
            };
            if indptr.numel() != major + 1 {
                return Err(Error::Type(format!(
                    "{scheme}: `indptr` has {} entries, expected {}",
                    indptr.numel(),
                    major + 1
                )));
            }
            if indices.numel() != vals.numel() {
                return Err(Error::Type(format!(
                    "{scheme}: {} indices for {} values",
                    indices.numel(),
                    vals.numel()
                )));
            }
            let mut prev = 0u64;
            for a in 0..major {
                let lo = check_ptr(indptr.data[a as usize], vals.numel(), scheme)?;
                let hi = check_ptr(indptr.data[(a + 1) as usize], vals.numel(), scheme)?;
                if lo < prev || hi < lo {
                    return Err(Error::Type(format!(
                        "{scheme}: `indptr` is not monotone at {a}"
                    )));
                }
                prev = hi;
                for p in lo..hi {
                    let b = check_index(indices.data[p as usize], minor, "index")?;
                    let lin = if scheme == "csr" {
                        a * strides[0] + b * strides[1]
                    } else {
                        b * strides[0] + a * strides[1]
                    };
                    out[lin as usize] = vals.data[p as usize];
                }
            }
        }
        // Block CSR: the block-sparse attention and pruned-MLP case.
        "bsr" => {
            require_rank2(scheme, shape)?;
            let block = attr_uvec("block").unwrap_or_else(|| vec![1, 1]);
            if block.len() != 2 || block[0] == 0 || block[1] == 0 {
                return Err(Error::Type("bsr: `block` must be [rows, cols]".into()));
            }
            let (br, bc) = (block[0], block[1]);
            let indptr = get("indptr")?;
            let indices = get("indices")?;
            let vals = get("values")?;
            let brows = shape[0].div_ceil(br);
            let bcols = shape[1].div_ceil(bc);
            if indptr.numel() != brows + 1 {
                return Err(Error::Type(format!(
                    "bsr: `indptr` has {} entries, expected {}",
                    indptr.numel(),
                    brows + 1
                )));
            }
            let per_block = br * bc;
            if vals.numel() != indices.numel() * per_block {
                return Err(Error::Type(format!(
                    "bsr: {} values for {} blocks of {br}x{bc}",
                    vals.numel(),
                    indices.numel()
                )));
            }
            for a in 0..brows {
                let lo = check_ptr(indptr.data[a as usize], indices.numel(), "bsr")?;
                let hi = check_ptr(indptr.data[(a + 1) as usize], indices.numel(), "bsr")?;
                if hi < lo {
                    return Err(Error::Type(format!("bsr: `indptr` is not monotone at {a}")));
                }
                for p in lo..hi {
                    let bcol = check_index(indices.data[p as usize], bcols, "bsr block column")?;
                    for i in 0..br {
                        for j in 0..bc {
                            let (r, c) = (a * br + i, bcol * bc + j);
                            if r >= shape[0] || c >= shape[1] {
                                continue; // edge block, partially outside
                            }
                            out[(r * strides[0] + c * strides[1]) as usize] =
                                vals.data[(p * per_block + i * bc + j) as usize];
                        }
                    }
                }
            }
        }
        // n:m structured sparsity — NVIDIA sparse tensor cores.
        "nm" => {
            let nn = attr_u("n").ok_or_else(|| Error::Type("nm: needs `n`".into()))?;
            let mm = attr_u("m").ok_or_else(|| Error::Type("nm: needs `m`".into()))?;
            if nn == 0 || mm == 0 || nn > mm {
                return Err(Error::Type(format!("nm: {nn}:{mm} is not a valid ratio")));
            }
            let mask = get("mask")?;
            let vals = get("values")?;
            if mask.numel() != n {
                return Err(Error::Type(format!(
                    "nm: `mask` has {} entries for a {n}-element tensor",
                    mask.numel()
                )));
            }
            if !n.is_multiple_of(mm) {
                return Err(Error::Type(format!(
                    "nm: {n} elements is not a whole number of {mm}-element groups"
                )));
            }
            // Exactly n set bits per group of m: the constraint the hardware
            // relies on, so it is checked rather than assumed.
            let mut taken = 0u64;
            for g in 0..(n / mm) {
                let set = (0..mm)
                    .filter(|k| mask.data[(g * mm + k) as usize] != 0.0)
                    .count() as u64;
                if set > nn {
                    return Err(Error::Type(format!(
                        "nm: group {g} has {set} non-zeros, more than {nn} of every {mm}"
                    )));
                }
                for k in 0..mm {
                    let i = g * mm + k;
                    if mask.data[i as usize] != 0.0 {
                        out[i as usize] = *vals
                            .data
                            .get(taken as usize)
                            .ok_or_else(|| Error::Bounds("nm: ran out of values".into()))?;
                        taken += 1;
                    }
                }
            }
            if taken != vals.numel() {
                return Err(Error::Type(format!(
                    "nm: `mask` selects {taken} positions but {} values were supplied",
                    vals.numel()
                )));
            }
        }
        // Unstructured pruning: a dense bitmap plus packed values.
        "bitmask" => {
            let mask = get("mask")?;
            let vals = get("values")?;
            if mask.numel() != n {
                return Err(Error::Type(format!(
                    "bitmask: `mask` has {} entries for a {n}-element tensor",
                    mask.numel()
                )));
            }
            let mut taken = 0usize;
            for (i, slot) in out.iter_mut().enumerate() {
                if mask.data[i] != 0.0 {
                    *slot = *vals.data.get(taken).ok_or_else(|| {
                        Error::Bounds(format!(
                            "bitmask: mask sets more positions than the {} values supplied",
                            vals.numel()
                        ))
                    })?;
                    taken += 1;
                }
            }
            if taken != vals.data.len() {
                return Err(Error::Type(format!(
                    "bitmask: `mask` selects {taken} positions but {} values were supplied",
                    vals.data.len()
                )));
            }
        }
        // Variable-length rows: sequences, MoE token routing.
        "ragged" => {
            require_rank2(scheme, shape)?;
            let offsets = get("offsets")?;
            let vals = get("values")?;
            if offsets.numel() != shape[0] + 1 {
                return Err(Error::Type(format!(
                    "ragged: `offsets` has {} entries, expected {}",
                    offsets.numel(),
                    shape[0] + 1
                )));
            }
            for r in 0..shape[0] {
                let lo = check_ptr(offsets.data[r as usize], vals.numel(), "ragged")?;
                let hi = check_ptr(offsets.data[(r + 1) as usize], vals.numel(), "ragged")?;
                if hi < lo {
                    return Err(Error::Type(format!(
                        "ragged: `offsets` is not monotone at {r}"
                    )));
                }
                if hi - lo > shape[1] {
                    return Err(Error::Type(format!(
                        "ragged: row {r} has {} entries but the padded width is {}",
                        hi - lo,
                        shape[1]
                    )));
                }
                for (j, p) in (lo..hi).enumerate() {
                    out[(r * strides[0] + j as u64 * strides[1]) as usize] = vals.data[p as usize];
                }
            }
        }
        // A list of (block index, dense block): MoE experts, sparse deltas.
        "blocklist" => {
            let block = attr_uvec("block").ok_or_else(|| {
                Error::Type("blocklist: needs a `block` shape in its attributes".into())
            })?;
            if block.len() != shape.len() || block.contains(&0) {
                return Err(Error::Type(format!(
                    "blocklist: block {block:?} does not fit a rank-{} tensor",
                    shape.len()
                )));
            }
            let index = get("index")?;
            let blocks = get("blocks")?;
            let per_block: u64 = block.iter().product();
            if blocks.numel() != index.numel() * per_block {
                return Err(Error::Type(format!(
                    "blocklist: {} values for {} blocks of {:?}",
                    blocks.numel(),
                    index.numel(),
                    block
                )));
            }
            let grid: Vec<u64> = shape
                .iter()
                .zip(&block)
                .map(|(d, b)| d.div_ceil(*b))
                .collect();
            let gstrides = crate::layout::Order::RowMajor.strides(&grid);
            let bstrides = crate::layout::Order::RowMajor.strides(&block);
            for (k, gi) in index.data.iter().enumerate() {
                let g = check_index(*gi, numel(&grid), "blocklist block index")?;
                // Recover the block's grid coordinates.
                let mut rem = g;
                let mut base = vec![0u64; shape.len()];
                for d in 0..shape.len() {
                    base[d] = (rem / gstrides[d]) * block[d];
                    rem %= gstrides[d];
                }
                let mut inner = vec![0u64; shape.len()];
                for e in 0..per_block {
                    let mut r = e;
                    for d in 0..shape.len() {
                        inner[d] = r / bstrides[d];
                        r %= bstrides[d];
                    }
                    let mut lin = 0u64;
                    let mut outside = false;
                    for d in 0..shape.len() {
                        let i = base[d] + inner[d];
                        if i >= shape[d] {
                            outside = true;
                            break;
                        }
                        lin += i * strides[d];
                    }
                    if !outside {
                        out[lin as usize] = blocks.data[k * per_block as usize + e as usize];
                    }
                }
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "sparsity scheme `{other}` is not in the §04.6 table"
            )))
        }
    }

    Ok(Tensor::new(shape.to_vec(), dtype.clone(), out))
}

fn require_rank2(scheme: &str, shape: &[u64]) -> Res<()> {
    if shape.len() != 2 {
        return Err(Error::Type(format!(
            "sparse `{scheme}` is two-dimensional; got a rank-{} shape",
            shape.len()
        )));
    }
    Ok(())
}

fn check_index(v: f64, extent: u64, what: &str) -> Res<u64> {
    if v < 0.0 || !v.is_finite() || v as u64 >= extent {
        return Err(Error::Bounds(format!(
            "{what} {v} is out of range for extent {extent}"
        )));
    }
    Ok(v as u64)
}

fn check_ptr(v: f64, limit: u64, scheme: &str) -> Res<u64> {
    if v < 0.0 || !v.is_finite() || v as u64 > limit {
        return Err(Error::Bounds(format!(
            "{scheme}: pointer {v} is out of range for {limit} entries"
        )));
    }
    Ok(v as u64)
}

/// The number of stored values a scheme carries, for reporting how much a
/// sparse delta actually costs (§08.6).
pub fn stored_values(parts: &[(&str, Tensor)]) -> u64 {
    parts
        .iter()
        .filter(|(k, _)| matches!(*k, "values" | "blocks"))
        .map(|(_, t)| t.numel())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{otype, HashAlgo};
    use crate::dtype::Round;
    use crate::expr::{dims, BinOp, Ctx, Expr, Scalar};
    use crate::layout::Layout;
    use crate::store::{MemoryStore, WritableStore};

    fn t(shape: &[u64], data: &[f64]) -> Tensor {
        Tensor::new(shape.to_vec(), DType::F32, data.to_vec())
    }

    fn dense_lit(s: &mut MemoryStore, shape: &[u64], dtype: &DType, data: &[f64]) -> Expr {
        let t = Tensor::new(shape.to_vec(), dtype.clone(), data.to_vec());
        let bytes = t.to_bytes(dtype, &Layout::default(), Round::Rne).unwrap();
        let d = s.put(&bytes).unwrap();
        Expr::Literal {
            chunks: (otype::BLOB, d),
            dtype: dtype.clone(),
            shape: dims(shape),
            layout: Layout::default(),
        }
    }

    fn attrs(pairs: Vec<(&str, Value)>) -> Value {
        Value::map(pairs)
    }

    #[test]
    fn coo_places_every_non_zero() {
        // indices are (n_dims x nnz): rows [0, 1, 2], cols [2, 0, 1].
        let idx = t(&[2, 3], &[0.0, 1.0, 2.0, 2.0, 0.0, 1.0]);
        let vals = t(&[3], &[7.0, 8.0, 9.0]);
        let out = densify(
            "coo",
            &[("indices", idx), ("values", vals)],
            &attrs(vec![]),
            &[3, 3],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(out.data, vec![0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 9.0, 0.0]);
    }

    #[test]
    fn coo_rejects_an_index_out_of_range() {
        let idx = t(&[2, 1], &[3.0, 0.0]);
        let vals = t(&[1], &[1.0]);
        assert!(matches!(
            densify(
                "coo",
                &[("indices", idx), ("values", vals)],
                &attrs(vec![]),
                &[3, 3],
                &DType::F32,
                0.0
            ),
            Err(Error::Bounds(_))
        ));
    }

    #[test]
    fn csr_and_csc_are_transposes_of_one_another() {
        // [[0, 5], [6, 0]]
        let csr = densify(
            "csr",
            &[
                ("indptr", t(&[3], &[0.0, 1.0, 2.0])),
                ("indices", t(&[2], &[1.0, 0.0])),
                ("values", t(&[2], &[5.0, 6.0])),
            ],
            &attrs(vec![]),
            &[2, 2],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(csr.data, vec![0.0, 5.0, 6.0, 0.0]);

        // The same structure read as CSC gives the transpose.
        let csc = densify(
            "csc",
            &[
                ("indptr", t(&[3], &[0.0, 1.0, 2.0])),
                ("indices", t(&[2], &[1.0, 0.0])),
                ("values", t(&[2], &[5.0, 6.0])),
            ],
            &attrs(vec![]),
            &[2, 2],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(csc.data, vec![0.0, 6.0, 5.0, 0.0]);
    }

    #[test]
    fn csr_rejects_a_non_monotone_indptr() {
        assert!(densify(
            "csr",
            &[
                ("indptr", t(&[3], &[0.0, 2.0, 1.0])),
                ("indices", t(&[2], &[0.0, 1.0])),
                ("values", t(&[2], &[1.0, 2.0])),
            ],
            &attrs(vec![]),
            &[2, 2],
            &DType::F32,
            0.0,
        )
        .is_err());
        // And a pointer past the end of the values array.
        assert!(densify(
            "csr",
            &[
                ("indptr", t(&[3], &[0.0, 1.0, 9.0])),
                ("indices", t(&[2], &[0.0, 1.0])),
                ("values", t(&[2], &[1.0, 2.0])),
            ],
            &attrs(vec![]),
            &[2, 2],
            &DType::F32,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn bsr_places_whole_blocks() {
        // A 4x4 matrix with two 2x2 blocks: (0,0) and (1,1).
        let out = densify(
            "bsr",
            &[
                ("indptr", t(&[3], &[0.0, 1.0, 2.0])),
                ("indices", t(&[2], &[0.0, 1.0])),
                ("values", t(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])),
            ],
            &attrs(vec![(
                "block",
                Value::Array(vec![Value::U(2), Value::U(2)]),
            )]),
            &[4, 4],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(
            out.data,
            vec![
                1.0, 2.0, 0.0, 0.0, //
                3.0, 4.0, 0.0, 0.0, //
                0.0, 0.0, 5.0, 6.0, //
                0.0, 0.0, 7.0, 8.0,
            ]
        );
    }

    #[test]
    fn nm_enforces_its_ratio() {
        // 2:4 — the sparse-tensor-core case.
        let mask = t(&[8], &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let vals = t(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let out = densify(
            "nm",
            &[("mask", mask), ("values", vals)],
            &attrs(vec![("n", Value::U(2)), ("m", Value::U(4))]),
            &[8],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(out.data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);

        // Three non-zeros in a group of four violates 2:4, and the hardware
        // relies on that not happening.
        let bad = t(&[4], &[1.0, 1.0, 1.0, 0.0]);
        assert!(densify(
            "nm",
            &[("mask", bad), ("values", t(&[3], &[1.0, 2.0, 3.0]))],
            &attrs(vec![("n", Value::U(2)), ("m", Value::U(4))]),
            &[4],
            &DType::F32,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn bitmask_is_a_dense_bitmap_and_packed_values() {
        let mask = Tensor::new(vec![2, 3], DType::Bool, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let vals = t(&[3], &[10.0, 20.0, 30.0]);
        let out = densify(
            "bitmask",
            &[("mask", mask), ("values", vals)],
            &attrs(vec![]),
            &[2, 3],
            &DType::F32,
            -1.0,
        )
        .unwrap();
        assert_eq!(out.data, vec![10.0, -1.0, 20.0, -1.0, 30.0, -1.0]);

        // A values array that disagrees with the mask is refused: guessing
        // which one to trust is how a pruned model comes out subtly wrong.
        let mask = Tensor::new(vec![3], DType::Bool, vec![1.0, 1.0, 0.0]);
        assert!(densify(
            "bitmask",
            &[("mask", mask), ("values", t(&[1], &[1.0]))],
            &attrs(vec![]),
            &[3],
            &DType::F32,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn ragged_rows_are_padded_with_the_fill() {
        let out = densify(
            "ragged",
            &[
                ("offsets", t(&[4], &[0.0, 2.0, 2.0, 5.0])),
                ("values", t(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
            &attrs(vec![]),
            &[3, 3],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0]);
        // A row longer than the padded width is an error, not a silent
        // truncation.
        assert!(densify(
            "ragged",
            &[
                ("offsets", t(&[2], &[0.0, 4.0])),
                ("values", t(&[4], &[1.0, 2.0, 3.0, 4.0])),
            ],
            &attrs(vec![]),
            &[1, 3],
            &DType::F32,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn blocklist_places_dense_blocks_by_index() {
        // A 2x4 tensor in 1x2 blocks: grid is 2x2, so blocks 1 and 2 are the
        // top-right and bottom-left halves.
        let out = densify(
            "blocklist",
            &[
                ("index", t(&[2], &[1.0, 2.0])),
                ("blocks", t(&[4], &[1.0, 2.0, 3.0, 4.0])),
            ],
            &attrs(vec![(
                "block",
                Value::Array(vec![Value::U(1), Value::U(2)]),
            )]),
            &[2, 4],
            &DType::F32,
            0.0,
        )
        .unwrap();
        assert_eq!(out.data, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        assert!(densify(
            "blocklist",
            &[("index", t(&[1], &[9.0])), ("blocks", t(&[2], &[1.0, 2.0])),],
            &attrs(vec![(
                "block",
                Value::Array(vec![Value::U(1), Value::U(2)]),
            )]),
            &[2, 4],
            &DType::F32,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn an_unknown_scheme_is_refused() {
        assert!(matches!(
            densify(
                "wavelet",
                &[("values", t(&[1], &[1.0]))],
                &attrs(vec![]),
                &[1],
                &DType::F32,
                0.0
            ),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(SCHEMES.len(), 8);
    }

    #[test]
    fn a_pruned_fine_tune_is_add_of_base_and_sparse() {
        // §08.6: the delta costs only its non-zeros. This is the whole reason
        // sparsity is an expression node rather than a tensor kind.
        let mut s = MemoryStore::new(HashAlgo::default());
        let base = dense_lit(
            &mut s,
            &[2, 3],
            &DType::F32,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        let mask = dense_lit(
            &mut s,
            &[2, 3],
            &DType::Bool,
            &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let vals = dense_lit(&mut s, &[2], &DType::F32, &[0.5, -0.25]);
        let delta = Expr::Sparse {
            scheme: "bitmask".into(),
            parts: vec![("mask".into(), mask), ("values".into(), vals)],
            attrs: Value::Map(vec![]),
            shape: dims(&[2, 3]),
            dtype: DType::F32,
            fill: Scalar::Int(0),
        };
        let tuned = Expr::Bin {
            op: BinOp::Add,
            a: Box::new(base),
            b: Box::new(delta.clone()),
        };
        let out = tuned.eval(&Ctx::new(&s)).unwrap();
        assert_eq!(out.data, vec![1.0, 1.5, 1.0, 1.0, 1.0, 0.75]);
        // Two stored values for a six-element delta.
        assert_eq!(delta.deps_all().len(), 2);
    }

    #[test]
    fn the_nm_example_of_section_04_6_round_trips_through_cbor() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mask = dense_lit(&mut s, &[4], &DType::Bool, &[1.0, 0.0, 0.0, 1.0]);
        let vals = dense_lit(&mut s, &[2], &DType::BF16, &[1.0, 2.0]);
        let v = Value::map(vec![
            ("op", Value::text("sparse")),
            ("scheme", Value::text("nm")),
            ("n", Value::U(2)),
            ("m", Value::U(4)),
            ("mask", mask.to_value()),
            ("values", vals.to_value()),
            ("shape", Value::Array(vec![Value::U(4)])),
            ("dtype", DType::BF16.to_value()),
            ("fill", Value::F64(0.0)),
        ]);
        let e = Expr::from_value(&v).unwrap();
        assert_eq!(e.infer().unwrap().dtype, DType::BF16);
        assert_eq!(
            e.eval(&Ctx::new(&s)).unwrap().data,
            vec![1.0, 0.0, 0.0, 2.0]
        );
        // Through canonical CBOR, unchanged.
        let round = crate::cbor::decode(&e.to_value().encode()).unwrap();
        let again = Expr::from_value(&round).unwrap();
        assert_eq!(
            again.eval(&Ctx::new(&s)).unwrap().data,
            vec![1.0, 0.0, 0.0, 2.0]
        );
    }
}
