//! §04.4 — layouts.
//!
//! Shape says *what*; layout says *where the bits are*. The section's one hard
//! rule is that a layout MUST be sufficient to compute the byte offset and bit
//! position of element `(i₀…i_{n-1})` with no additional knowledge — so that is
//! exactly what this module computes, for every layout kind, and
//! [`Layout::sufficiency`] reports the cases where a descriptor is *not*
//! sufficient instead of guessing (R-T03).

use crate::cbor::Value;
use crate::dtype::DType;

/// Index order for the dense kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Last axis varies fastest (C order).
    RowMajor,
    /// First axis varies fastest (Fortran order).
    ColMajor,
}

impl Order {
    fn name(self) -> &'static str {
        match self {
            Order::RowMajor => "row-major",
            Order::ColMajor => "col-major",
        }
    }
    fn parse(s: &str) -> Option<Order> {
        Some(match s {
            "row-major" | "c" => Order::RowMajor,
            "col-major" | "column-major" | "f" | "fortran" => Order::ColMajor,
            _ => return None,
        })
    }

    /// Element strides for a dense array of this shape.
    pub fn strides(self, shape: &[u64]) -> Vec<u64> {
        let mut s = vec![1u64; shape.len()];
        match self {
            Order::RowMajor => {
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    s[i] = s[i + 1] * shape[i + 1].max(1);
                }
            }
            Order::ColMajor => {
                for i in 1..shape.len() {
                    s[i] = s[i - 1] * shape[i - 1].max(1);
                }
            }
        }
        s
    }
}

/// Which end of a word the first packed element occupies. Every existing
/// implementation quietly disagrees about this, which is why §04.4 makes it
/// explicit and the conformance suite tests all four combinations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitOrder {
    LsbFirst,
    MsbFirst,
}

/// Where per-block scales sit relative to their elements (§04.4
/// `blocked-scaled`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interleave {
    /// All element blocks, then all scales. The MX/MXFP4 arrangement.
    ScalesAfter,
    /// All scales, then all element blocks.
    ScalesBefore,
    /// Each block is followed by its own scale — a GGUF-style self-contained
    /// block.
    ScalesInline,
}

impl Interleave {
    fn name(self) -> &'static str {
        match self {
            Interleave::ScalesAfter => "scales-after",
            Interleave::ScalesBefore => "scales-before",
            Interleave::ScalesInline => "scales-inline",
        }
    }
    fn parse(s: &str) -> Option<Interleave> {
        Some(match s {
            "scales-after" => Interleave::ScalesAfter,
            "scales-before" => Interleave::ScalesBefore,
            "scales-inline" => Interleave::ScalesInline,
            _ => return None,
        })
    }
}

/// One field of an `interleaved` block layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub name: String,
    pub dtype: Option<DType>,
    pub count: Option<u64>,
}

/// A layout descriptor (§04.4).
#[derive(Clone, Debug, PartialEq)]
pub enum Layout {
    Strided {
        order: Order,
        /// Explicit element strides. When absent, derived from `order`.
        strides: Option<Vec<u64>>,
        /// Element offset of the first element.
        offset: u64,
    },
    Tiled {
        tile: Vec<u64>,
        outer: Order,
        inner: Order,
    },
    BlockedScaled {
        block: Vec<u64>,
        scale_dtype: DType,
        scale_order: Order,
        interleave: Interleave,
    },
    Packed {
        elems_per_word: u32,
        word_bits: u32,
        bit_order: BitOrder,
        order: Order,
    },
    Interleaved {
        groups: Vec<Vec<Field>>,
        stride_bytes: u64,
    },
    /// §09.4 sharding: the local layout depends on the shard, so this layout is
    /// resolved against a `ShardMap` before element offsets exist.
    Sharded {
        spec: Value,
    },
    Opaque {
        id: String,
    },
}

impl Default for Layout {
    fn default() -> Self {
        Layout::row_major()
    }
}

/// Why a layout cannot answer "where is element i?" — the R-T03 report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sufficiency {
    /// Element positions are computable from the descriptor alone.
    Sufficient,
    /// The descriptor is well-formed but positions need external information.
    NeedsContext(&'static str),
    /// The descriptor is internally inconsistent.
    Inconsistent(String),
}

impl Layout {
    pub fn row_major() -> Layout {
        Layout::Strided {
            order: Order::RowMajor,
            strides: None,
            offset: 0,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Layout::Strided { .. } => "strided",
            Layout::Tiled { .. } => "tiled",
            Layout::BlockedScaled { .. } => "blocked-scaled",
            Layout::Packed { .. } => "packed",
            Layout::Interleaved { .. } => "interleaved",
            Layout::Sharded { .. } => "sharded",
            Layout::Opaque { .. } => "opaque",
        }
    }

    /// The dense linear element index of `index` under this layout, ignoring
    /// bit packing. `None` when the layout does not define one.
    pub fn linear(&self, shape: &[u64], index: &[u64]) -> Option<u64> {
        if index.len() != shape.len() {
            return None;
        }
        for (i, d) in index.iter().zip(shape) {
            if i >= d {
                return None;
            }
        }
        match self {
            Layout::Strided {
                order,
                strides,
                offset,
            } => {
                let s = match strides {
                    Some(s) if s.len() == shape.len() => s.clone(),
                    Some(_) => return None,
                    None => order.strides(shape),
                };
                let mut lin = *offset;
                for (i, st) in index.iter().zip(&s) {
                    lin = lin.checked_add(i.checked_mul(*st)?)?;
                }
                Some(lin)
            }
            Layout::Tiled { tile, outer, inner } => {
                if tile.len() != shape.len() {
                    return None;
                }
                // Outer index selects the tile; inner index the element inside
                // it. Partial edge tiles are stored full, which is what makes
                // the arithmetic constant-time.
                let tiles: Vec<u64> = shape
                    .iter()
                    .zip(tile)
                    .map(|(d, t)| d.div_ceil((*t).max(1)))
                    .collect();
                let outer_idx: Vec<u64> = index
                    .iter()
                    .zip(tile)
                    .map(|(i, t)| i / (*t).max(1))
                    .collect();
                let inner_idx: Vec<u64> = index
                    .iter()
                    .zip(tile)
                    .map(|(i, t)| i % (*t).max(1))
                    .collect();
                let os = outer.strides(&tiles);
                let is = inner.strides(tile);
                let tile_elems: u64 = tile.iter().product();
                let mut o = 0u64;
                for (i, st) in outer_idx.iter().zip(&os) {
                    o = o.checked_add(i.checked_mul(*st)?)?;
                }
                let mut n = 0u64;
                for (i, st) in inner_idx.iter().zip(&is) {
                    n = n.checked_add(i.checked_mul(*st)?)?;
                }
                o.checked_mul(tile_elems)?.checked_add(n)
            }
            Layout::BlockedScaled { block, .. } => {
                // Elements are dense row-major within a block, blocks dense
                // row-major over the block grid.
                if block.len() != shape.len() {
                    return None;
                }
                let grid: Vec<u64> = shape
                    .iter()
                    .zip(block)
                    .map(|(d, b)| d.div_ceil((*b).max(1)))
                    .collect();
                let bidx: Vec<u64> = index
                    .iter()
                    .zip(block)
                    .map(|(i, b)| i / (*b).max(1))
                    .collect();
                let iidx: Vec<u64> = index
                    .iter()
                    .zip(block)
                    .map(|(i, b)| i % (*b).max(1))
                    .collect();
                let gs = Order::RowMajor.strides(&grid);
                let is = Order::RowMajor.strides(block);
                let belems: u64 = block.iter().product();
                let mut g = 0u64;
                for (i, st) in bidx.iter().zip(&gs) {
                    g = g.checked_add(i.checked_mul(*st)?)?;
                }
                let mut n = 0u64;
                for (i, st) in iidx.iter().zip(&is) {
                    n = n.checked_add(i.checked_mul(*st)?)?;
                }
                g.checked_mul(belems)?.checked_add(n)
            }
            Layout::Packed { order, .. } => {
                let s = order.strides(shape);
                let mut lin = 0u64;
                for (i, st) in index.iter().zip(&s) {
                    lin = lin.checked_add(i.checked_mul(*st)?)?;
                }
                Some(lin)
            }
            Layout::Interleaved { .. } | Layout::Sharded { .. } | Layout::Opaque { .. } => None,
        }
    }

    /// The bit position of element `index`, which is the rule §04.4 actually
    /// requires. Divide by 8 for the byte offset; the remainder is the bit
    /// position within that byte.
    pub fn bit_offset(&self, shape: &[u64], dtype: &DType, index: &[u64]) -> Option<u128> {
        let (num, den) = dtype.bits_rational();
        match self {
            Layout::Packed {
                elems_per_word,
                word_bits,
                bit_order,
                ..
            } => {
                let lin = self.linear(shape, index)?;
                let epw = (*elems_per_word).max(1) as u64;
                let word = lin / epw;
                let slot = (lin % epw) as u32;
                let w = num.div_ceil(den);
                let in_word = match bit_order {
                    BitOrder::LsbFirst => slot * w,
                    BitOrder::MsbFirst => word_bits.checked_sub((slot + 1) * w)?,
                };
                Some(word as u128 * *word_bits as u128 + in_word as u128)
            }
            Layout::BlockedScaled {
                block,
                scale_dtype,
                interleave,
                ..
            } => {
                let lin = self.linear(shape, index)?;
                let belems: u64 = block.iter().product::<u64>().max(1);
                let nblocks = numel(shape).div_ceil(belems);
                let block_bits = dtype.packed_bytes(belems) as u128 * 8;
                let scale_bits = scale_dtype.packed_bytes(1) as u128 * 8;
                let b = lin / belems;
                let within = lin % belems;
                let elem_bits = within as u128 * num as u128 / den as u128;
                Some(match interleave {
                    Interleave::ScalesAfter => b as u128 * block_bits + elem_bits,
                    Interleave::ScalesBefore => {
                        nblocks as u128 * scale_bits + b as u128 * block_bits + elem_bits
                    }
                    Interleave::ScalesInline => {
                        b as u128 * (block_bits + scale_bits) + scale_bits + elem_bits
                    }
                })
            }
            _ => {
                let lin = self.linear(shape, index)?;
                Some(lin as u128 * num as u128 / den as u128)
            }
        }
    }

    /// The bit position of the scale covering `index`, for `blocked-scaled`
    /// layouts.
    pub fn scale_bit_offset(&self, shape: &[u64], dtype: &DType, index: &[u64]) -> Option<u128> {
        let Layout::BlockedScaled {
            block,
            scale_dtype,
            interleave,
            ..
        } = self
        else {
            return None;
        };
        let lin = self.linear(shape, index)?;
        let belems: u64 = block.iter().product::<u64>().max(1);
        let nblocks = numel(shape).div_ceil(belems);
        let block_bits = dtype.packed_bytes(belems) as u128 * 8;
        let scale_bits = scale_dtype.packed_bytes(1) as u128 * 8;
        let b = (lin / belems) as u128;
        Some(match interleave {
            Interleave::ScalesAfter => nblocks as u128 * block_bits + b * scale_bits,
            Interleave::ScalesBefore => b * scale_bits,
            Interleave::ScalesInline => b * (block_bits + scale_bits),
        })
    }

    /// Bytes needed to store a tensor of `shape` and `dtype` in this layout.
    /// This is the quantity R-T02 compares against `ChunkList.total`.
    pub fn stored_bytes(&self, shape: &[u64], dtype: &DType) -> Option<u64> {
        let n = numel(shape);
        match self {
            Layout::Strided {
                strides: Some(s),
                offset,
                ..
            } => {
                if s.len() != shape.len() {
                    return None;
                }
                let mut span = *offset + 1;
                for (d, st) in shape.iter().zip(s) {
                    span = span.checked_add(d.saturating_sub(1).checked_mul(*st)?)?;
                }
                Some(dtype.packed_bytes(span))
            }
            Layout::Strided { offset, .. } => Some(dtype.packed_bytes(offset + n)),
            Layout::Tiled { tile, .. } => {
                if tile.len() != shape.len() {
                    return None;
                }
                // Edge tiles are stored whole.
                let padded: u64 = shape
                    .iter()
                    .zip(tile)
                    .map(|(d, t)| d.div_ceil((*t).max(1)) * (*t).max(1))
                    .product();
                Some(dtype.packed_bytes(padded))
            }
            Layout::BlockedScaled {
                block, scale_dtype, ..
            } => {
                let belems: u64 = block.iter().product::<u64>().max(1);
                let nblocks = n.div_ceil(belems);
                Some(nblocks * (dtype.packed_bytes(belems) + scale_dtype.packed_bytes(1)))
            }
            Layout::Packed {
                elems_per_word,
                word_bits,
                ..
            } => {
                let epw = (*elems_per_word).max(1) as u64;
                let words = n.div_ceil(epw);
                Some(words * (*word_bits as u64).div_ceil(8))
            }
            Layout::Interleaved {
                groups,
                stride_bytes,
            } => {
                let per_block: u64 = groups
                    .iter()
                    .map(|g| g.iter().filter_map(|f| f.count).sum::<u64>())
                    .sum();
                if per_block == 0 {
                    return None;
                }
                Some(n.div_ceil(per_block) * *stride_bytes)
            }
            Layout::Sharded { .. } | Layout::Opaque { .. } => None,
        }
    }

    /// R-T03: is this descriptor sufficient to place every element?
    pub fn sufficiency(&self, shape: &[u64], dtype: &DType) -> Sufficiency {
        match self {
            Layout::Strided {
                strides: Some(s), ..
            } if s.len() != shape.len() => Sufficiency::Inconsistent(format!(
                "strided: {} strides for a {}-dimensional shape",
                s.len(),
                shape.len()
            )),
            Layout::Strided {
                strides: Some(s), ..
            } if s.contains(&0) && numel(shape) > 1 => Sufficiency::Inconsistent(
                "strided: a zero stride aliases elements onto each other".into(),
            ),
            Layout::Tiled { tile, .. } if tile.len() != shape.len() => {
                Sufficiency::Inconsistent(format!(
                    "tiled: {}-dimensional tile for a {}-dimensional shape",
                    tile.len(),
                    shape.len()
                ))
            }
            Layout::Tiled { tile, .. } if tile.contains(&0) => {
                Sufficiency::Inconsistent("tiled: zero tile extent".into())
            }
            Layout::BlockedScaled { block, .. } if block.len() != shape.len() => {
                Sufficiency::Inconsistent(format!(
                    "blocked-scaled: {}-dimensional block for a {}-dimensional shape",
                    block.len(),
                    shape.len()
                ))
            }
            Layout::BlockedScaled { block, .. } if block.contains(&0) => {
                Sufficiency::Inconsistent("blocked-scaled: zero block extent".into())
            }
            Layout::Packed {
                elems_per_word,
                word_bits,
                ..
            } => {
                let w = dtype.bits();
                if *elems_per_word == 0 || *word_bits == 0 {
                    Sufficiency::Inconsistent("packed: zero elems_per_word or word_bits".into())
                } else if elems_per_word * w > *word_bits {
                    Sufficiency::Inconsistent(format!(
                        "packed: {elems_per_word} x {w} bits does not fit in a {word_bits}-bit word"
                    ))
                } else {
                    Sufficiency::Sufficient
                }
            }
            Layout::Interleaved { groups, .. } => {
                if groups
                    .iter()
                    .flatten()
                    .any(|f| f.dtype.is_none() || f.count.is_none())
                {
                    // The abbreviated `groups:[["w","s","z"]]` form names the
                    // fields without sizing them, and §04.4's own rule then
                    // cannot be satisfied. Say so rather than inventing widths.
                    Sufficiency::NeedsContext(
                        "interleaved: fields need a dtype and a count to place elements",
                    )
                } else {
                    Sufficiency::Sufficient
                }
            }
            Layout::Sharded { .. } => {
                Sufficiency::NeedsContext("sharded: resolve against the ShardMap first (§09.4)")
            }
            Layout::Opaque { .. } => {
                Sufficiency::NeedsContext("opaque: element positions are defined by the foreign id")
            }
            _ => Sufficiency::Sufficient,
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            Layout::Strided {
                order,
                strides,
                offset,
            } => {
                let mut p = vec![("k", Value::text("strided"))];
                match strides {
                    Some(s) => p.push((
                        "strides",
                        Value::Array(s.iter().map(|x| Value::U(*x)).collect()),
                    )),
                    None => p.push(("order", Value::text(order.name()))),
                }
                if *offset != 0 {
                    p.push(("offset", Value::U(*offset)));
                }
                Value::map(p)
            }
            Layout::Tiled { tile, outer, inner } => Value::map(vec![
                ("k", Value::text("tiled")),
                (
                    "tile",
                    Value::Array(tile.iter().map(|x| Value::U(*x)).collect()),
                ),
                ("outer", Value::text(outer.name())),
                ("inner", Value::text(inner.name())),
            ]),
            Layout::BlockedScaled {
                block,
                scale_dtype,
                scale_order,
                interleave,
            } => Value::map(vec![
                ("k", Value::text("blocked-scaled")),
                (
                    "block",
                    Value::Array(block.iter().map(|x| Value::U(*x)).collect()),
                ),
                ("scale_dtype", scale_dtype.to_value()),
                (
                    "scale_layout",
                    Value::map(vec![
                        ("k", Value::text("strided")),
                        ("order", Value::text(scale_order.name())),
                    ]),
                ),
                ("interleave", Value::text(interleave.name())),
            ]),
            Layout::Packed {
                elems_per_word,
                word_bits,
                bit_order,
                order,
            } => Value::map(vec![
                ("k", Value::text("packed")),
                ("elems_per_word", Value::U(*elems_per_word as u64)),
                ("word_bits", Value::U(*word_bits as u64)),
                (
                    "bit_order",
                    Value::text(match bit_order {
                        BitOrder::LsbFirst => "lsb-first",
                        BitOrder::MsbFirst => "msb-first",
                    }),
                ),
                ("order", Value::text(order.name())),
            ]),
            Layout::Interleaved {
                groups,
                stride_bytes,
            } => Value::map(vec![
                ("k", Value::text("interleaved")),
                (
                    "groups",
                    Value::Array(
                        groups
                            .iter()
                            .map(|g| {
                                Value::Array(
                                    g.iter()
                                        .map(|f| {
                                            let mut p = vec![("n", Value::text(f.name.clone()))];
                                            if let Some(d) = &f.dtype {
                                                p.push(("dtype", d.to_value()));
                                            }
                                            if let Some(c) = f.count {
                                                p.push(("count", Value::U(c)));
                                            }
                                            Value::map(p)
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    ),
                ),
                ("stride_bytes", Value::U(*stride_bytes)),
            ]),
            Layout::Sharded { spec } => {
                Value::map(vec![("k", Value::text("sharded")), ("spec", spec.clone())])
            }
            Layout::Opaque { id } => Value::map(vec![
                ("k", Value::text("opaque")),
                ("id", Value::text(id.clone())),
            ]),
        }
    }

    pub fn from_value(v: &Value) -> Result<Layout, String> {
        let k = v
            .get("k")
            .and_then(|x| x.as_str())
            .ok_or_else(|| "layout: `k` missing".to_string())?;
        let u = |key: &str| v.get(key).and_then(|x| x.as_u64());
        let uvec = |key: &str| -> Option<Vec<u64>> {
            v.get(key)?
                .as_array()?
                .iter()
                .map(|x| x.as_u64())
                .collect::<Option<Vec<u64>>>()
        };
        let order = |key: &str, default: Order| -> Result<Order, String> {
            match v.get(key).and_then(|x| x.as_str()) {
                Some(s) => Order::parse(s).ok_or_else(|| format!("layout: unknown order `{s}`")),
                None => Ok(default),
            }
        };
        Ok(match k {
            "strided" => Layout::Strided {
                order: order("order", Order::RowMajor)?,
                strides: uvec("strides"),
                offset: u("offset").unwrap_or(0),
            },
            "tiled" => Layout::Tiled {
                tile: uvec("tile").ok_or_else(|| "layout: tiled needs `tile`".to_string())?,
                outer: order("outer", Order::RowMajor)?,
                inner: order("inner", Order::RowMajor)?,
            },
            "blocked-scaled" => Layout::BlockedScaled {
                block: uvec("block")
                    .ok_or_else(|| "layout: blocked-scaled needs `block`".to_string())?,
                scale_dtype: match v.get("scale_dtype") {
                    Some(d) => DType::from_value(d)?,
                    None => DType::E8M0,
                },
                scale_order: match v.get("scale_layout") {
                    Some(l) => match Layout::from_value(l)? {
                        Layout::Strided { order, .. } => order,
                        other => {
                            return Err(format!(
                                "layout: scale_layout must be strided, got {}",
                                other.kind()
                            ))
                        }
                    },
                    None => Order::RowMajor,
                },
                interleave: match v.get("interleave").and_then(|x| x.as_str()) {
                    Some(s) => Interleave::parse(s)
                        .ok_or_else(|| format!("layout: unknown interleave `{s}`"))?,
                    None => Interleave::ScalesAfter,
                },
            },
            "packed" => Layout::Packed {
                elems_per_word: u("elems_per_word")
                    .ok_or_else(|| "layout: packed needs `elems_per_word`".to_string())?
                    as u32,
                word_bits: u("word_bits")
                    .ok_or_else(|| "layout: packed needs `word_bits`".to_string())?
                    as u32,
                bit_order: match v.get("bit_order").and_then(|x| x.as_str()) {
                    Some("lsb-first") | None => BitOrder::LsbFirst,
                    Some("msb-first") => BitOrder::MsbFirst,
                    Some(o) => return Err(format!("layout: unknown bit_order `{o}`")),
                },
                order: order("order", Order::RowMajor)?,
            },
            "interleaved" => {
                let mut groups = Vec::new();
                for g in v.get("groups").and_then(|x| x.as_array()).unwrap_or(&[]) {
                    let mut fields = Vec::new();
                    for f in g.as_array().unwrap_or(&[]) {
                        // Both the abbreviated name-only form of §04.4 and the
                        // sized form are accepted; `sufficiency` is where the
                        // difference is reported.
                        match f {
                            Value::Text(name) => fields.push(Field {
                                name: name.clone(),
                                dtype: None,
                                count: None,
                            }),
                            other => fields.push(Field {
                                name: other
                                    .get("n")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                dtype: match other.get("dtype") {
                                    Some(d) => Some(DType::from_value(d)?),
                                    None => None,
                                },
                                count: other.get("count").and_then(|x| x.as_u64()),
                            }),
                        }
                    }
                    groups.push(fields);
                }
                Layout::Interleaved {
                    groups,
                    stride_bytes: u("stride_bytes").unwrap_or(0),
                }
            }
            "sharded" => Layout::Sharded {
                spec: v.get("spec").cloned().unwrap_or(Value::Null),
            },
            "opaque" => Layout::Opaque {
                id: v
                    .get("id")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| "layout: opaque needs `id`".to_string())?
                    .to_string(),
            },
            other => return Err(format!("layout: unknown kind `{other}`")),
        })
    }
}

pub fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_and_column_major_are_transposes() {
        let shape = [3u64, 4];
        let rm = Layout::row_major();
        let cm = Layout::Strided {
            order: Order::ColMajor,
            strides: None,
            offset: 0,
        };
        assert_eq!(rm.linear(&shape, &[0, 0]), Some(0));
        assert_eq!(rm.linear(&shape, &[1, 0]), Some(4));
        assert_eq!(rm.linear(&shape, &[0, 1]), Some(1));
        assert_eq!(cm.linear(&shape, &[1, 0]), Some(1));
        assert_eq!(cm.linear(&shape, &[0, 1]), Some(3));
        // Out of range is None rather than a wrong answer.
        assert_eq!(rm.linear(&shape, &[3, 0]), None);
        assert_eq!(rm.linear(&shape, &[0]), None);
    }

    #[test]
    fn explicit_strides_express_column_major() {
        // The example from §04.4.
        let l = Layout::Strided {
            order: Order::RowMajor,
            strides: Some(vec![1, 4096]),
            offset: 0,
        };
        let shape = [4096u64, 4096];
        assert_eq!(l.linear(&shape, &[2, 3]), Some(2 + 3 * 4096));
        assert_eq!(l.stored_bytes(&shape, &DType::BF16), Some(4096 * 4096 * 2));
    }

    #[test]
    fn tiled_places_a_tile_contiguously() {
        let l = Layout::Tiled {
            tile: vec![2, 2],
            outer: Order::RowMajor,
            inner: Order::RowMajor,
        };
        let shape = [4u64, 4];
        // The first 2x2 tile occupies linear 0..4.
        assert_eq!(l.linear(&shape, &[0, 0]), Some(0));
        assert_eq!(l.linear(&shape, &[0, 1]), Some(1));
        assert_eq!(l.linear(&shape, &[1, 0]), Some(2));
        assert_eq!(l.linear(&shape, &[1, 1]), Some(3));
        // The next tile along the row starts at 4.
        assert_eq!(l.linear(&shape, &[0, 2]), Some(4));
        // Every element gets a distinct slot.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..4 {
            for j in 0..4 {
                assert!(seen.insert(l.linear(&shape, &[i, j]).unwrap()));
            }
        }
        assert_eq!(seen.len(), 16);
        // Edge tiles are stored whole: a 3x3 tensor in 2x2 tiles costs 16.
        assert_eq!(l.stored_bytes(&[3, 3], &DType::U8), Some(16));
    }

    #[test]
    fn packed_layouts_disagree_in_all_four_ways() {
        let shape = [8u64];
        let mk = |bo| Layout::Packed {
            elems_per_word: 8,
            word_bits: 32,
            bit_order: bo,
            order: Order::RowMajor,
        };
        let lsb = mk(BitOrder::LsbFirst);
        let msb = mk(BitOrder::MsbFirst);
        // int4, 8 per 32-bit word: this is exactly GPTQ's qweight packing.
        assert_eq!(lsb.bit_offset(&shape, &DType::U4, &[0]), Some(0));
        assert_eq!(lsb.bit_offset(&shape, &DType::U4, &[1]), Some(4));
        assert_eq!(lsb.bit_offset(&shape, &DType::U4, &[7]), Some(28));
        assert_eq!(msb.bit_offset(&shape, &DType::U4, &[0]), Some(28));
        assert_eq!(msb.bit_offset(&shape, &DType::U4, &[7]), Some(0));
        assert_eq!(lsb.stored_bytes(&shape, &DType::U4), Some(4));
        // Two elements per 32-bit word wastes 24 bits per word, and the sizing
        // says so rather than silently densifying.
        let sparse = Layout::Packed {
            elems_per_word: 2,
            word_bits: 32,
            bit_order: BitOrder::LsbFirst,
            order: Order::RowMajor,
        };
        assert_eq!(sparse.stored_bytes(&shape, &DType::U4), Some(16));
    }

    #[test]
    fn packed_rejects_a_word_too_small_for_its_elements() {
        let l = Layout::Packed {
            elems_per_word: 16,
            word_bits: 32,
            bit_order: BitOrder::LsbFirst,
            order: Order::RowMajor,
        };
        assert!(matches!(
            l.sufficiency(&[16], &DType::U4),
            Sufficiency::Inconsistent(_)
        ));
    }

    #[test]
    fn blocked_scaled_is_the_mx_arrangement() {
        // MXFP4: f4e2m1 elements, blocks of 32, e8m0 scales after the data.
        let l = Layout::BlockedScaled {
            block: vec![1, 32],
            scale_dtype: DType::E8M0,
            scale_order: Order::RowMajor,
            interleave: Interleave::ScalesAfter,
        };
        let shape = [2u64, 64];
        // 4 blocks of 16 bytes plus 4 scale bytes.
        assert_eq!(l.stored_bytes(&shape, &DType::F4E2M1), Some(4 * 16 + 4));
        assert_eq!(l.bit_offset(&shape, &DType::F4E2M1, &[0, 0]), Some(0));
        assert_eq!(l.bit_offset(&shape, &DType::F4E2M1, &[0, 1]), Some(4));
        // Second block starts one 16-byte block in.
        assert_eq!(l.bit_offset(&shape, &DType::F4E2M1, &[0, 32]), Some(128));
        // Scales live after all four blocks.
        assert_eq!(
            l.scale_bit_offset(&shape, &DType::F4E2M1, &[0, 0]),
            Some(4 * 128)
        );
        assert_eq!(
            l.scale_bit_offset(&shape, &DType::F4E2M1, &[1, 33]),
            Some(4 * 128 + 3 * 8)
        );

        // The GGUF-style arrangement puts each scale next to its own block.
        let inline = Layout::BlockedScaled {
            block: vec![1, 32],
            scale_dtype: DType::F16,
            scale_order: Order::RowMajor,
            interleave: Interleave::ScalesInline,
        };
        // Q4_0: 2 bytes of scale + 16 bytes of nibbles = 18 bytes per 32.
        assert_eq!(inline.stored_bytes(&shape, &DType::U4), Some(4 * 18));
        assert_eq!(inline.bit_offset(&shape, &DType::U4, &[0, 0]), Some(16));
        assert_eq!(
            inline.scale_bit_offset(&shape, &DType::U4, &[0, 0]),
            Some(0)
        );
        assert_eq!(
            inline.scale_bit_offset(&shape, &DType::U4, &[0, 32]),
            Some(18 * 8)
        );
    }

    #[test]
    fn insufficient_descriptors_say_so() {
        let abbreviated = Layout::from_value(&Value::map(vec![
            ("k", Value::text("interleaved")),
            (
                "groups",
                Value::Array(vec![Value::Array(vec![
                    Value::text("w"),
                    Value::text("s"),
                    Value::text("z"),
                ])]),
            ),
            ("stride_bytes", Value::U(144)),
        ]))
        .unwrap();
        assert!(matches!(
            abbreviated.sufficiency(&[256], &DType::U4),
            Sufficiency::NeedsContext(_)
        ));
        assert_eq!(abbreviated.bit_offset(&[256], &DType::U4, &[0]), None);

        let sized = Layout::Interleaved {
            groups: vec![vec![
                Field {
                    name: "w".into(),
                    dtype: Some(DType::U4),
                    count: Some(256),
                },
                Field {
                    name: "s".into(),
                    dtype: Some(DType::F16),
                    count: Some(16),
                },
            ]],
            stride_bytes: 144,
        };
        assert_eq!(
            sized.sufficiency(&[256], &DType::U4),
            Sufficiency::Sufficient
        );
        assert_eq!(sized.stored_bytes(&[272], &DType::U4), Some(144));

        assert!(matches!(
            Layout::Opaque {
                id: "org.nvidia/tensorrt-weights.v10".into()
            }
            .sufficiency(&[4], &DType::U8),
            Sufficiency::NeedsContext(_)
        ));
        assert!(matches!(
            Layout::Strided {
                order: Order::RowMajor,
                strides: Some(vec![1]),
                offset: 0
            }
            .sufficiency(&[4, 4], &DType::U8),
            Sufficiency::Inconsistent(_)
        ));
    }

    #[test]
    fn descriptors_round_trip() {
        let cases = vec![
            Layout::row_major(),
            Layout::Strided {
                order: Order::RowMajor,
                strides: Some(vec![1, 4096]),
                offset: 7,
            },
            Layout::Tiled {
                tile: vec![128, 64],
                outer: Order::RowMajor,
                inner: Order::ColMajor,
            },
            Layout::BlockedScaled {
                block: vec![1, 32],
                scale_dtype: DType::E8M0,
                scale_order: Order::RowMajor,
                interleave: Interleave::ScalesAfter,
            },
            Layout::Packed {
                elems_per_word: 8,
                word_bits: 32,
                bit_order: BitOrder::LsbFirst,
                order: Order::RowMajor,
            },
            Layout::Interleaved {
                groups: vec![vec![Field {
                    name: "w".into(),
                    dtype: Some(DType::U4),
                    count: Some(256),
                }]],
                stride_bytes: 144,
            },
            Layout::Opaque {
                id: "org.ggml/q4_K".into(),
            },
        ];
        for l in cases {
            let v = l.to_value();
            assert_eq!(Layout::from_value(&v).unwrap(), l, "{}", l.kind());
            // And through canonical CBOR, which is what gets hashed.
            let round = crate::cbor::decode(&v.encode()).unwrap();
            assert_eq!(Layout::from_value(&round).unwrap(), l, "{}", l.kind());
        }
    }

    #[test]
    fn the_default_layout_is_what_existing_containers_carry() {
        let v = Value::map(vec![
            ("k", Value::text("strided")),
            ("order", Value::text("row-major")),
        ]);
        assert_eq!(Layout::from_value(&v).unwrap(), Layout::default());
        assert_eq!(Layout::default().to_value(), v);
    }
}
