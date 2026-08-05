//! §04.3 — the numeric type algebra.
//!
//! A dtype is not an enum of blessed formats; it is a structured descriptor from
//! which the four invariants of §04.3.5 are computed:
//!
//! 1. [`DType::bits_rational`] — bits per element, possibly fractional.
//! 2. [`DType::packed_bytes`] — bytes for `n` densely packed elements.
//! 3. [`DType::decode`] — bit-exact element semantics.
//! 4. [`DType::encode`] — element encoding under an explicit rounding mode.
//!
//! The alias table of §04.3.1 is *derived* data: every alias expands to a
//! descriptor, and a reader that has never heard of the alias is unaffected
//! because writers emit both ([`DType::to_value`]).
//!
//! Floats are handled generically for any `(w, e, m)` with `m <= 52`, which is
//! what `f64` can hold exactly; `f64` itself is the widest supported mantissa.
//! That covers every alias in the registry and every plausible successor.

use crate::cbor::Value;
use crate::container::Digest;

// ------------------------------------------------------------------ rounding --

/// Rounding modes of §04.3.5 clause 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// Round to nearest, ties to even — the IEEE 754 default.
    Rne,
    /// Toward zero.
    Rtz,
    /// Toward +∞.
    Rup,
    /// Toward −∞.
    Rdown,
    /// Reproducible stochastic rounding: the fractional part is the probability
    /// of rounding away from zero, drawn from a counter-based PRNG keyed by
    /// `seed` and the element index, so two implementations agree bit-for-bit.
    Stochastic { seed: u64, index: u64 },
}

impl Round {
    pub fn name(self) -> &'static str {
        match self {
            Round::Rne => "rne",
            Round::Rtz => "rtz",
            Round::Rup => "rup",
            Round::Rdown => "rdown",
            Round::Stochastic { .. } => "stochastic",
        }
    }

    pub fn parse(s: &str) -> Option<Round> {
        Some(match s {
            "rne" => Round::Rne,
            "rtz" => Round::Rtz,
            "rup" => Round::Rup,
            "rdown" => Round::Rdown,
            "stochastic" => Round::Stochastic { seed: 0, index: 0 },
            _ => return None,
        })
    }

    /// Rounds a non-negative real to an integer, given the sign of the value it
    /// came from (the directed modes are directed on the *signed* value, not on
    /// the magnitude).
    fn apply(self, mag: f64, negative: bool) -> f64 {
        let fract = mag - mag.floor();
        match self {
            Round::Rne => {
                let lo = mag.floor();
                // Ties go to the even neighbour; everything else to the
                // nearest.
                let up = fract > 0.5 || (fract == 0.5 && (lo as i64) % 2 != 0);
                if up {
                    lo + 1.0
                } else {
                    lo
                }
            }
            Round::Rtz => mag.floor(),
            Round::Rup => {
                if negative {
                    mag.floor()
                } else {
                    mag.ceil()
                }
            }
            Round::Rdown => {
                if negative {
                    mag.ceil()
                } else {
                    mag.floor()
                }
            }
            Round::Stochastic { seed, index } => {
                if fract == 0.0 {
                    mag
                } else {
                    // A counter-based PRNG: one keyed mixing step over
                    // (seed, index). Deterministic across implementations,
                    // which is the whole point of declaring the seed.
                    let mut h = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(index.wrapping_mul(0xbf58_476d_1ce4_e5b9));
                    h ^= h >> 30;
                    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    h ^= h >> 27;
                    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
                    h ^= h >> 31;
                    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
                    if u < fract {
                        mag.floor() + 1.0
                    } else {
                        mag.floor()
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------- float --

/// NaN encoding conventions (§04.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nan {
    /// All-ones exponent with non-zero mantissa, as IEEE 754.
    Ieee,
    /// Finite-only: exactly one NaN pattern (all-ones exponent *and* mantissa).
    Fn,
    /// No NaN is encodable.
    None,
}

/// Whether the most significant bit is a sign bit (§04.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Lead,
    None,
}

/// A float format descriptor. `bias`, `sub`, `inf`, `nan` and `sign` default to
/// the IEEE 754 derivation; the registry's oddities (`f8e4m3`, `e8m0`) differ
/// and say so explicitly rather than being special cases in code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatFmt {
    pub w: u16,
    pub e: u16,
    pub m: u16,
    pub bias: i32,
    pub sub: bool,
    pub inf: bool,
    pub nan: Nan,
    pub sign: Sign,
}

impl FloatFmt {
    /// The IEEE 754 derivation: bias `2^(e-1) - 1`, subnormals, infinities and
    /// IEEE NaNs, sign in the leading bit.
    pub const fn ieee(w: u16, e: u16, m: u16) -> FloatFmt {
        FloatFmt {
            w,
            e,
            m,
            bias: (1i32 << (e - 1)) - 1,
            sub: true,
            inf: true,
            nan: Nan::Ieee,
            sign: Sign::Lead,
        }
    }

    /// A finite-only format: no infinities, one NaN pattern (OCP FP8 E4M3).
    pub const fn finite(w: u16, e: u16, m: u16) -> FloatFmt {
        FloatFmt {
            inf: false,
            nan: Nan::Fn,
            ..FloatFmt::ieee(w, e, m)
        }
    }

    /// A format in which every bit pattern is a number: no infinities and no
    /// NaN. The OCP MX element types below 8 bits are like this — all 16
    /// patterns of `f4e2m1` are values, which is why its maximum is 6.0 and
    /// not 4.0.
    pub const fn total(w: u16, e: u16, m: u16) -> FloatFmt {
        FloatFmt {
            inf: false,
            nan: Nan::None,
            ..FloatFmt::ieee(w, e, m)
        }
    }

    /// True when this format is a pure power-of-two exponent type: no mantissa,
    /// no zero, no subnormals, no infinities, one NaN. This is `e8m0`, the MX
    /// scale type, and it is why dequantizing an MX block is exact (§05.2.8).
    fn is_exponent_only(&self) -> bool {
        self.m == 0 && !self.sub && !self.inf && self.nan == Nan::Fn
    }

    fn max_exp(&self) -> i32 {
        // The largest biased exponent that encodes a finite number.
        let all_ones = (1i32 << self.e) - 1;
        let top = if self.nan == Nan::Ieee || self.inf {
            all_ones - 1
        } else {
            all_ones
        };
        top - self.bias
    }

    fn min_normal_exp(&self) -> i32 {
        1 - self.bias
    }

    fn mask(&self, bits: u16) -> u64 {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    /// The largest finite magnitude.
    fn max_finite(&self) -> f64 {
        if self.is_exponent_only() {
            return ldexp(1.0, self.max_exp());
        }
        let significand = if self.nan == Nan::Fn && !self.inf {
            // The all-ones mantissa at the top exponent is the NaN pattern.
            (self.mask(self.m) - 1) as f64
        } else {
            self.mask(self.m) as f64
        };
        ldexp(
            1.0 + significand / ldexp(1.0, self.m as i32),
            self.max_exp(),
        )
    }

    /// Decodes one element from its raw bits.
    pub fn decode(&self, bits: u64) -> f64 {
        let mbits = self.m;
        let ebits = self.e;
        let mant = bits & self.mask(mbits);
        let exp = ((bits >> mbits) & self.mask(ebits)) as i32;
        let negative = self.sign == Sign::Lead && (bits >> (mbits + ebits)) & 1 == 1;
        let all_ones = (1i32 << ebits) - 1;

        let mag = if self.is_exponent_only() {
            if exp == all_ones {
                f64::NAN
            } else {
                ldexp(1.0, exp - self.bias)
            }
        } else if exp == all_ones && (self.inf || self.nan != Nan::None) {
            match self.nan {
                Nan::Ieee if mant != 0 => f64::NAN,
                Nan::Ieee => f64::INFINITY,
                Nan::Fn if mant == self.mask(mbits) => f64::NAN,
                Nan::Fn => {
                    // Still a finite number: the exponent is usable except for
                    // the single reserved pattern.
                    return finish(
                        ldexp(
                            1.0 + mant as f64 / ldexp(1.0, mbits as i32),
                            exp - self.bias,
                        ),
                        negative,
                    );
                }
                Nan::None if self.inf && mant == 0 => f64::INFINITY,
                Nan::None => {
                    return finish(
                        ldexp(
                            1.0 + mant as f64 / ldexp(1.0, mbits as i32),
                            exp - self.bias,
                        ),
                        negative,
                    )
                }
            }
        } else if exp == 0 {
            if mant == 0 {
                0.0
            } else if self.sub {
                ldexp(mant as f64, self.min_normal_exp() - mbits as i32)
            } else {
                // Subnormals not supported: the encoding is reserved, and a
                // reader that silently treated it as a number would disagree
                // with one that did not. Treat as zero, which is what every
                // flush-to-zero implementation does.
                0.0
            }
        } else {
            ldexp(
                1.0 + mant as f64 / ldexp(1.0, mbits as i32),
                exp - self.bias,
            )
        };
        finish(mag, negative)
    }

    /// Encodes one element, returning its raw bits.
    pub fn encode(&self, x: f64, round: Round) -> u64 {
        let negative = x.is_sign_negative();
        let sign_bit = if self.sign == Sign::Lead && negative {
            1u64 << (self.m + self.e)
        } else {
            0
        };
        let all_ones = self.mask(self.e);

        if x.is_nan() {
            return match self.nan {
                Nan::Ieee => sign_bit | (all_ones << self.m) | 1,
                Nan::Fn => sign_bit | (all_ones << self.m) | self.mask(self.m),
                // Nothing can represent it; the largest finite value is the
                // least-wrong answer and `stats.nan` in the descriptor is where
                // a publisher declares that this happened.
                Nan::None => sign_bit | self.encode_finite(self.max_finite(), Round::Rtz),
            };
        }
        if self.is_exponent_only() {
            let mag = x.abs();
            if mag == 0.0 || !mag.is_finite() {
                return if mag == 0.0 {
                    0
                } else {
                    (all_ones - 1) << self.m
                };
            }
            let e = ilogb(mag);
            let e = e.clamp(-self.bias, self.max_exp());
            return ((e + self.bias) as u64) << self.m;
        }
        if x.is_infinite() {
            if self.inf {
                return sign_bit | (all_ones << self.m);
            }
            return sign_bit | self.encode_finite(self.max_finite(), Round::Rtz);
        }
        sign_bit | self.encode_finite(x, round)
    }

    /// Encodes the magnitude of a finite value; the sign is applied by the
    /// caller so that directed rounding sees the original signedness.
    fn encode_finite(&self, x: f64, round: Round) -> u64 {
        let negative = x.is_sign_negative();
        let mag = x.abs();
        if mag == 0.0 {
            return 0;
        }
        let mbits = self.m as i32;
        let mut e = ilogb(mag);

        // Subnormal (or zero) range.
        if e < self.min_normal_exp() {
            let scale = self.min_normal_exp() - mbits;
            let q = round.apply(ldexp(mag, -scale), negative);
            let qi = q as u64;
            if qi >= (1u64 << mbits) {
                // Rounded up into the smallest normal.
                return 1u64 << mbits;
            }
            return qi;
        }

        let mut q = round.apply(ldexp(mag, mbits - e), negative) as u64;
        if q >= (1u64 << (mbits + 1)) {
            e += 1;
            q >>= 1;
        }
        if e > self.max_exp() {
            if self.inf {
                return self.mask(self.e) << self.m;
            }
            // Saturate: the max finite value. OCP calls this saturating
            // conversion; the alternative silently turns big weights into NaN.
            let mag = self.max_finite();
            let e2 = ilogb(mag);
            let q2 = ldexp(mag, mbits - e2) as u64;
            return (((e2 + self.bias) as u64) << self.m) | (q2 & self.mask(self.m));
        }
        (((e + self.bias) as u64) << self.m) | (q & self.mask(self.m))
    }
}

fn finish(mag: f64, negative: bool) -> f64 {
    if negative {
        -mag
    } else {
        mag
    }
}

/// `x * 2^n`, exact and overflow-safe for the ranges the dtype code uses.
fn ldexp(x: f64, n: i32) -> f64 {
    let mut r = x;
    let mut n = n;
    while n > 512 {
        r *= f64::from_bits(0x7fd0_0000_0000_0000); // 2^1022
        n -= 1022;
    }
    while n < -512 {
        r *= f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
        n += 1022;
    }
    r * (2f64).powi(n)
}

/// The unbiased binary exponent of a non-zero finite magnitude: `floor(log2 x)`,
/// computed exactly from the bit pattern.
fn ilogb(x: f64) -> i32 {
    let b = x.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    if e != 0 {
        return e - 1023;
    }
    // Subnormal f64: normalize.
    let m = b & ((1u64 << 52) - 1);
    -1022 - (m.leading_zeros() as i32 - 11)
}

// ------------------------------------------------------------------ sub-byte --

/// Ternary packing schemes (§04.3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TernPack {
    /// Two bits per trit; one value wasted.
    Naive,
    /// Base-3 encoding of five trits in one byte: 1.6 bits per value, which is
    /// what BitNet-class models actually need.
    B3x5,
}

/// How a codebook is shared across a tensor (§04.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shared {
    PerTensor,
    PerBlock,
    PerRow,
}

impl Shared {
    fn name(self) -> &'static str {
        match self {
            Shared::PerTensor => "per-tensor",
            Shared::PerBlock => "per-block",
            Shared::PerRow => "per-row",
        }
    }
    fn parse(s: &str) -> Option<Shared> {
        Some(match s {
            "per-tensor" => Shared::PerTensor,
            "per-block" => Shared::PerBlock,
            "per-row" => Shared::PerRow,
            _ => return None,
        })
    }
}

// --------------------------------------------------------------------- dtype --

/// A structured dtype descriptor (§04.3).
#[derive(Clone, Debug, PartialEq)]
pub enum DType {
    Float(FloatFmt),
    Int {
        w: u16,
        signed: bool,
    },
    /// Fixed point with `frac` fractional bits, e.g. Q8.8 is `w:16, frac:8`.
    Fixed {
        w: u16,
        signed: bool,
        frac: u16,
    },
    Bool,
    Ternary {
        pack: TernPack,
    },
    /// One-bit sign weights, values `{-1, +1}`.
    Binary,
    /// `w`-bit indices into a `Codebook` object (§05.4).
    Codebook {
        w: u16,
        book: Digest,
        dim: u32,
        shared: Shared,
    },
    Complex {
        re: Box<DType>,
    },
    Struct {
        fields: Vec<(String, DType)>,
        packed: bool,
    },
    /// A foreign block format preserved bit-exactly. Only sizing is known;
    /// §04.3.5 says any operation other than `literal`/`slice`/`cast-to-opaque`
    /// is undefined, and this implementation enforces that.
    Opaque {
        id: String,
        block_elems: u64,
        block_bytes: u64,
    },
    Posit {
        w: u16,
        es: u16,
    },
    /// Log-number-system accelerator format.
    LogDom {
        w: u16,
        base: u16,
        frac: u16,
    },
    Str,
}

impl DType {
    pub const F64: DType = DType::Float(FloatFmt::ieee(64, 11, 52));
    pub const F32: DType = DType::Float(FloatFmt::ieee(32, 8, 23));
    pub const F16: DType = DType::Float(FloatFmt::ieee(16, 5, 10));
    pub const BF16: DType = DType::Float(FloatFmt::ieee(16, 8, 7));
    pub const TF32: DType = DType::Float(FloatFmt::ieee(19, 8, 10));
    pub const F8E4M3: DType = DType::Float(FloatFmt::finite(8, 4, 3));
    pub const F8E5M2: DType = DType::Float(FloatFmt::ieee(8, 5, 2));
    pub const F6E3M2: DType = DType::Float(FloatFmt::total(6, 3, 2));
    pub const F6E2M3: DType = DType::Float(FloatFmt::total(6, 2, 3));
    pub const F4E2M1: DType = DType::Float(FloatFmt::total(4, 2, 1));
    /// The MX scale type: a power-of-two exponent, no zero, no infinities.
    pub const E8M0: DType = DType::Float(FloatFmt {
        w: 8,
        e: 8,
        m: 0,
        bias: 127,
        sub: false,
        inf: false,
        nan: Nan::Fn,
        sign: Sign::None,
    });
    pub const I8: DType = DType::Int { w: 8, signed: true };
    pub const I16: DType = DType::Int {
        w: 16,
        signed: true,
    };
    pub const I32: DType = DType::Int {
        w: 32,
        signed: true,
    };
    pub const I64: DType = DType::Int {
        w: 64,
        signed: true,
    };
    pub const U8: DType = DType::Int {
        w: 8,
        signed: false,
    };
    pub const U32: DType = DType::Int {
        w: 32,
        signed: false,
    };
    pub const I4: DType = DType::Int { w: 4, signed: true };
    pub const U4: DType = DType::Int {
        w: 4,
        signed: false,
    };
    pub const I2: DType = DType::Int { w: 2, signed: true };

    /// Bits per element as an exact rational (§04.3.5 clause 1). `b3x5`
    /// ternary is 8/5 — the reason this returns a rational at all.
    pub fn bits_rational(&self) -> (u32, u32) {
        match self {
            DType::Float(f) => (f.w as u32, 1),
            DType::Int { w, .. } | DType::Fixed { w, .. } => (*w as u32, 1),
            DType::Bool | DType::Binary => (1, 1),
            DType::Ternary { pack } => match pack {
                TernPack::Naive => (2, 1),
                TernPack::B3x5 => (8, 5),
            },
            DType::Codebook { w, dim, .. } => (*w as u32, (*dim).max(1)),
            DType::Complex { re } => {
                let (n, d) = re.bits_rational();
                (n * 2, d)
            }
            DType::Struct { fields, packed } => {
                let mut num = 0u32;
                for (_, f) in fields {
                    let (n, d) = f.bits_rational();
                    // Unpacked structs pad each field to a byte boundary.
                    num += if *packed {
                        n.div_ceil(d)
                    } else {
                        n.div_ceil(d).div_ceil(8) * 8
                    };
                }
                (num, 1)
            }
            DType::Opaque {
                block_elems,
                block_bytes,
                ..
            } => {
                let e = (*block_elems).max(1) as u32;
                ((*block_bytes as u32) * 8, e)
            }
            DType::Posit { w, .. } | DType::LogDom { w, .. } => (*w as u32, 1),
            // A variable-length type has no fixed width; it is sized by its
            // container, not by this function.
            DType::Str => (0, 1),
        }
    }

    /// Bits per element, rounded up. Prefer [`DType::bits_rational`] for
    /// sizing; this exists for reporting.
    pub fn bits(&self) -> u32 {
        let (n, d) = self.bits_rational();
        n.div_ceil(d)
    }

    /// Bytes required for `n` densely packed elements (§04.3.5 clause 2).
    pub fn packed_bytes(&self, n: u64) -> u64 {
        let (num, den) = self.bits_rational();
        match self {
            // Base-3 packing groups five trits into one byte; the last partial
            // group still costs a whole byte.
            DType::Ternary {
                pack: TernPack::B3x5,
            } => n.div_ceil(5),
            DType::Opaque { block_elems, .. } => {
                let be = (*block_elems).max(1);
                let blocks = n.div_ceil(be);
                blocks * (num as u64 * be) / (8 * den as u64)
            }
            _ => (n * num as u64).div_ceil(8 * den as u64),
        }
    }

    /// Whether this type's element semantics are known. `opaque` and `string`
    /// carry bytes whose meaning this implementation does not model, and
    /// §04.3.5 says so.
    pub fn is_numeric(&self) -> bool {
        !matches!(
            self,
            DType::Opaque { .. } | DType::Str | DType::Struct { .. }
        )
    }

    /// Reads element `i` from a dense little-endian, LSB-first bit stream
    /// (§04.3.5 clause 3). Returns `None` for types without defined element
    /// semantics, or when the buffer is too short.
    pub fn decode(&self, bytes: &[u8], i: u64) -> Option<f64> {
        match self {
            DType::Ternary { pack } => {
                let raw = match pack {
                    TernPack::Naive => read_bits(bytes, i * 2, 2)?,
                    TernPack::B3x5 => {
                        let byte = *bytes.get((i / 5) as usize)? as u64;
                        let mut v = byte;
                        for _ in 0..(i % 5) {
                            v /= 3;
                        }
                        v % 3
                    }
                };
                // 0,1,2 -> -1,0,+1
                Some(match raw {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                })
            }
            DType::Binary => {
                let b = read_bits(bytes, i, 1)?;
                Some(if b == 0 { -1.0 } else { 1.0 })
            }
            DType::Bool => Some(if read_bits(bytes, i, 1)? == 0 {
                0.0
            } else {
                1.0
            }),
            DType::Float(f) => Some(f.decode(read_bits(bytes, i * f.w as u64, f.w)?)),
            DType::Int { w, signed } => {
                let raw = read_bits(bytes, i * *w as u64, *w)?;
                Some(if *signed {
                    sign_extend(raw, *w) as f64
                } else {
                    raw as f64
                })
            }
            DType::Fixed { w, signed, frac } => {
                let raw = read_bits(bytes, i * *w as u64, *w)?;
                let v = if *signed {
                    sign_extend(raw, *w) as f64
                } else {
                    raw as f64
                };
                Some(ldexp(v, -(*frac as i32)))
            }
            DType::Posit { w, es } => {
                let raw = read_bits(bytes, i * *w as u64, *w)?;
                Some(decode_posit(raw, *w, *es))
            }
            DType::LogDom { w, base, frac } => {
                let raw = read_bits(bytes, i * *w as u64, *w)?;
                let l = ldexp(sign_extend(raw, *w) as f64, -(*frac as i32));
                Some((*base as f64).powf(l))
            }
            // Codebook indices are integers here; resolving them against the
            // book is the `dequantize` node's job (§05.4).
            DType::Codebook { w, .. } => Some(read_bits(bytes, i * *w as u64, *w)? as f64),
            DType::Complex { re } => re.decode(bytes, i * 2),
            DType::Opaque { .. } | DType::Str | DType::Struct { .. } => None,
        }
    }

    /// Reads an element from an arbitrary *bit* position rather than an element
    /// index. A layout (§04.4) computes bit positions that need not be a
    /// multiple of the element width — a `blocked-scaled` layout interleaves
    /// scales between blocks — so this is the accessor layout-aware code needs.
    ///
    /// Only defined for types whose width is a whole number of bits;
    /// fractional-width packings (base-3 ternary) are addressed by element
    /// index because their elements do not start on bit boundaries.
    pub fn decode_bits(&self, bytes: &[u8], bit: u64) -> Option<f64> {
        let (num, den) = self.bits_rational();
        if den != 1 {
            return None;
        }
        let w = num as u16;
        match self {
            DType::Binary => Some(if read_bits(bytes, bit, 1)? == 0 {
                -1.0
            } else {
                1.0
            }),
            DType::Bool => Some(if read_bits(bytes, bit, 1)? == 0 {
                0.0
            } else {
                1.0
            }),
            DType::Ternary {
                pack: TernPack::Naive,
            } => Some(match read_bits(bytes, bit, 2)? {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            }),
            DType::Float(f) => Some(f.decode(read_bits(bytes, bit, w)?)),
            DType::Int { signed, .. } => {
                let raw = read_bits(bytes, bit, w)?;
                Some(if *signed {
                    sign_extend(raw, w) as f64
                } else {
                    raw as f64
                })
            }
            DType::Fixed { signed, frac, .. } => {
                let raw = read_bits(bytes, bit, w)?;
                let v = if *signed {
                    sign_extend(raw, w) as f64
                } else {
                    raw as f64
                };
                Some(ldexp(v, -(*frac as i32)))
            }
            DType::Posit { w: pw, es } => Some(decode_posit(read_bits(bytes, bit, *pw)?, *pw, *es)),
            DType::LogDom { base, frac, .. } => {
                let raw = read_bits(bytes, bit, w)?;
                let l = ldexp(sign_extend(raw, w) as f64, -(*frac as i32));
                Some((*base as f64).powf(l))
            }
            DType::Codebook { w: cw, .. } => Some(read_bits(bytes, bit, *cw)? as f64),
            DType::Complex { .. }
            | DType::Ternary { .. }
            | DType::Opaque { .. }
            | DType::Str
            | DType::Struct { .. } => None,
        }
    }

    /// The inverse of [`DType::decode_bits`].
    pub fn encode_bits(&self, bytes: &mut [u8], bit: u64, x: f64, round: Round) -> bool {
        let (num, den) = self.bits_rational();
        if den != 1 {
            return false;
        }
        let w = num as u16;
        match self {
            DType::Binary => write_bits(bytes, bit, 1, if x < 0.0 { 0 } else { 1 }),
            DType::Bool => write_bits(bytes, bit, 1, if x == 0.0 { 0 } else { 1 }),
            DType::Ternary {
                pack: TernPack::Naive,
            } => write_bits(
                bytes,
                bit,
                2,
                if x < -0.5 {
                    0
                } else if x > 0.5 {
                    2
                } else {
                    1
                },
            ),
            DType::Float(f) => write_bits(bytes, bit, w, f.encode(x, round)),
            DType::Int { signed, .. } => {
                let (lo, hi) = int_range(w, *signed);
                let r = round.apply(x.abs(), x.is_sign_negative());
                let v = if x.is_sign_negative() { -r } else { r };
                write_bits(bytes, bit, w, (v.clamp(lo, hi) as i64) as u64 & mask64(w))
            }
            DType::Fixed { signed, frac, .. } => {
                let scaled = ldexp(x, *frac as i32);
                let (lo, hi) = int_range(w, *signed);
                let r = round.apply(scaled.abs(), scaled.is_sign_negative());
                let v = if scaled.is_sign_negative() { -r } else { r };
                write_bits(bytes, bit, w, (v.clamp(lo, hi) as i64) as u64 & mask64(w))
            }
            DType::Codebook { w: cw, .. } => {
                write_bits(bytes, bit, *cw, (x.max(0.0) as u64) & mask64(*cw))
            }
            DType::Posit { .. }
            | DType::LogDom { .. }
            | DType::Complex { .. }
            | DType::Ternary { .. }
            | DType::Opaque { .. }
            | DType::Str
            | DType::Struct { .. } => false,
        }
    }

    /// Writes element `i` into a dense bit stream. Returns false for types
    /// without defined element semantics.
    pub fn encode(&self, bytes: &mut [u8], i: u64, x: f64, round: Round) -> bool {
        match self {
            DType::Ternary { pack } => {
                let raw = if x < -0.5 {
                    0u64
                } else if x > 0.5 {
                    2
                } else {
                    1
                };
                match pack {
                    TernPack::Naive => write_bits(bytes, i * 2, 2, raw),
                    TernPack::B3x5 => {
                        let Some(byte) = bytes.get_mut((i / 5) as usize) else {
                            return false;
                        };
                        let mut place = 1u64;
                        for _ in 0..(i % 5) {
                            place *= 3;
                        }
                        let cur = *byte as u64;
                        let old = (cur / place) % 3;
                        *byte = (cur + (raw.wrapping_sub(old)).wrapping_mul(place)) as u8;
                        true
                    }
                }
            }
            DType::Binary => write_bits(bytes, i, 1, if x < 0.0 { 0 } else { 1 }),
            DType::Bool => write_bits(bytes, i, 1, if x == 0.0 { 0 } else { 1 }),
            DType::Float(f) => write_bits(bytes, i * f.w as u64, f.w, f.encode(x, round)),
            DType::Int { w, signed } => {
                let (lo, hi) = int_range(*w, *signed);
                let r = round.apply(x.abs(), x.is_sign_negative());
                let v = if x.is_sign_negative() { -r } else { r };
                let v = v.clamp(lo, hi);
                write_bits(bytes, i * *w as u64, *w, (v as i64) as u64 & mask64(*w))
            }
            DType::Fixed { w, signed, frac } => {
                let scaled = ldexp(x, *frac as i32);
                let (lo, hi) = int_range(*w, *signed);
                let r = round.apply(scaled.abs(), scaled.is_sign_negative());
                let v = if scaled.is_sign_negative() { -r } else { r };
                let v = v.clamp(lo, hi);
                write_bits(bytes, i * *w as u64, *w, (v as i64) as u64 & mask64(*w))
            }
            DType::Codebook { w, .. } => {
                let v = x.max(0.0) as u64;
                write_bits(bytes, i * *w as u64, *w, v & mask64(*w))
            }
            DType::Complex { re } => re.encode(bytes, i * 2, x, round),
            DType::Posit { .. }
            | DType::LogDom { .. }
            | DType::Opaque { .. }
            | DType::Str
            | DType::Struct { .. } => false,
        }
    }

    // ------------------------------------------------------------- registry --

    /// The registered alias for this descriptor, if it has one (§04.3.6).
    pub fn alias(&self) -> Option<&'static str> {
        Some(match self {
            DType::Float(f) => match (f.w, f.e, f.m) {
                (64, 11, 52) => "f64",
                (32, 8, 23) => "f32",
                (16, 5, 10) => "f16",
                (16, 8, 7) => "bf16",
                (19, 8, 10) => "tf32",
                (8, 4, 3) => "f8e4m3",
                (8, 5, 2) => "f8e5m2",
                (6, 3, 2) => "f6e3m2",
                (6, 2, 3) => "f6e2m3",
                (4, 2, 1) => "f4e2m1",
                (8, 8, 0) => "e8m0",
                _ => return None,
            },
            DType::Int { w, signed } => match (w, signed) {
                (8, true) => "i8",
                (16, true) => "i16",
                (32, true) => "i32",
                (64, true) => "i64",
                (8, false) => "u8",
                (16, false) => "u16",
                (32, false) => "u32",
                (64, false) => "u64",
                (4, true) => "i4",
                (4, false) => "u4",
                (2, true) => "i2",
                (2, false) => "u2",
                _ => return None,
            },
            DType::Bool => "bool",
            DType::Binary => "binary",
            DType::Ternary {
                pack: TernPack::B3x5,
            } => "ternary-b3x5",
            _ => return None,
        })
    }

    /// Expands a registered alias. Readers must accept the expanded form even
    /// for known aliases (§04.3.6), so this is a convenience, not the parser.
    pub fn from_alias(name: &str) -> Option<DType> {
        Some(match name {
            "f64" => DType::F64,
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "tf32" => DType::TF32,
            "f8e4m3" => DType::F8E4M3,
            "f8e5m2" => DType::F8E5M2,
            "f6e3m2" => DType::F6E3M2,
            "f6e2m3" => DType::F6E2M3,
            "f4e2m1" => DType::F4E2M1,
            "e8m0" => DType::E8M0,
            "i8" => DType::I8,
            "i16" => DType::I16,
            "i32" => DType::I32,
            "i64" => DType::I64,
            "u8" => DType::U8,
            "u16" => DType::Int {
                w: 16,
                signed: false,
            },
            "u32" => DType::U32,
            "u64" => DType::Int {
                w: 64,
                signed: false,
            },
            "i4" => DType::I4,
            "u4" => DType::U4,
            "i2" => DType::I2,
            "u2" => DType::Int {
                w: 2,
                signed: false,
            },
            "bool" => DType::Bool,
            "binary" => DType::Binary,
            "ternary-b3x5" => DType::Ternary {
                pack: TernPack::B3x5,
            },
            _ => return None,
        })
    }

    /// The structural descriptor of §04.3. Writers emit the alias *and* the
    /// expansion; fields equal to the IEEE derivation are omitted, so a reader
    /// that applies the documented defaults reconstructs this exactly.
    pub fn to_value(&self) -> Value {
        let mut pairs: Vec<(&str, Value)> = Vec::new();
        if let Some(a) = self.alias() {
            pairs.push(("alias", Value::text(a)));
        }
        match self {
            DType::Float(f) => {
                pairs.push(("k", Value::text("float")));
                pairs.push(("w", Value::U(f.w as u64)));
                pairs.push(("e", Value::U(f.e as u64)));
                pairs.push(("m", Value::U(f.m as u64)));
                let d = FloatFmt::ieee(f.w, f.e, f.m);
                if f.bias != d.bias {
                    pairs.push(("bias", int_value(f.bias as i64)));
                }
                if f.sub != d.sub {
                    pairs.push(("sub", Value::Bool(f.sub)));
                }
                if f.inf != d.inf {
                    pairs.push(("inf", Value::Bool(f.inf)));
                }
                if f.nan != d.nan {
                    pairs.push((
                        "nan",
                        Value::text(match f.nan {
                            Nan::Ieee => "ieee",
                            Nan::Fn => "fn",
                            Nan::None => "none",
                        }),
                    ));
                }
                if f.sign != d.sign {
                    pairs.push((
                        "sign",
                        Value::text(match f.sign {
                            Sign::Lead => "lead",
                            Sign::None => "none",
                        }),
                    ));
                }
            }
            DType::Int { w, signed } => {
                pairs.push(("k", Value::text("int")));
                pairs.push(("w", Value::U(*w as u64)));
                pairs.push(("signed", Value::Bool(*signed)));
            }
            DType::Fixed { w, signed, frac } => {
                pairs.push(("k", Value::text("fixed")));
                pairs.push(("w", Value::U(*w as u64)));
                pairs.push(("signed", Value::Bool(*signed)));
                pairs.push(("frac", Value::U(*frac as u64)));
            }
            DType::Bool => {
                pairs.push(("k", Value::text("bool")));
                pairs.push(("w", Value::U(1)));
            }
            DType::Ternary { pack } => {
                pairs.push(("k", Value::text("ternary")));
                pairs.push((
                    "vals",
                    Value::Array(vec![Value::I(-1), Value::U(0), Value::U(1)]),
                ));
                pairs.push((
                    "pack",
                    Value::text(match pack {
                        TernPack::Naive => "naive",
                        TernPack::B3x5 => "b3x5",
                    }),
                ));
            }
            DType::Binary => {
                pairs.push(("k", Value::text("binary")));
                pairs.push(("vals", Value::Array(vec![Value::I(-1), Value::U(1)])));
            }
            DType::Codebook {
                w,
                book,
                dim,
                shared,
            } => {
                pairs.push(("k", Value::text("codebook")));
                pairs.push(("w", Value::U(*w as u64)));
                pairs.push((
                    "book",
                    Value::Array(vec![
                        Value::U(crate::container::otype::CODEBOOK as u64),
                        Value::Bytes(book.to_vec()),
                    ]),
                ));
                pairs.push(("dim", Value::U(*dim as u64)));
                pairs.push(("shared", Value::text(shared.name())));
            }
            DType::Complex { re } => {
                pairs.push(("k", Value::text("complex")));
                pairs.push(("re", re.to_value()));
            }
            DType::Struct { fields, packed } => {
                pairs.push(("k", Value::text("struct")));
                pairs.push((
                    "fields",
                    Value::Array(
                        fields
                            .iter()
                            .map(|(n, t)| Value::Array(vec![Value::text(n.clone()), t.to_value()]))
                            .collect(),
                    ),
                ));
                pairs.push(("packed", Value::Bool(*packed)));
            }
            DType::Opaque {
                id,
                block_elems,
                block_bytes,
            } => {
                pairs.push(("k", Value::text("opaque")));
                pairs.push(("id", Value::text(id.clone())));
                pairs.push(("block_elems", Value::U(*block_elems)));
                pairs.push(("block_bytes", Value::U(*block_bytes)));
            }
            DType::Posit { w, es } => {
                pairs.push(("k", Value::text("posit")));
                pairs.push(("w", Value::U(*w as u64)));
                pairs.push(("es", Value::U(*es as u64)));
            }
            DType::LogDom { w, base, frac } => {
                pairs.push(("k", Value::text("logdom")));
                pairs.push(("w", Value::U(*w as u64)));
                pairs.push(("base", Value::U(*base as u64)));
                pairs.push(("frac", Value::U(*frac as u64)));
            }
            DType::Str => {
                pairs.push(("k", Value::text("string")));
                pairs.push(("enc", Value::text("utf8")));
            }
        }
        Value::map(pairs)
    }

    /// Parses a descriptor. An inline `k` always wins over `alias`; an unknown
    /// alias with an inline descriptor is fully understood (§04.3.6), and an
    /// unknown alias *without* one is an error rather than a guess.
    pub fn from_value(v: &Value) -> Result<DType, String> {
        let v = match v {
            Value::Tag(crate::cbor::TAG_DTYPE, inner) => inner.as_ref(),
            other => other,
        };
        let k = v.get("k").and_then(|x| x.as_str());
        if k.is_none() {
            let alias = v
                .get("alias")
                .and_then(|x| x.as_str())
                .or_else(|| v.as_str())
                .ok_or_else(|| "dtype: neither `k` nor `alias` present".to_string())?;
            return DType::from_alias(alias)
                .ok_or_else(|| format!("dtype: unknown alias `{alias}` with no descriptor"));
        }
        let u = |key: &str| v.get(key).and_then(|x| x.as_u64());
        let need = |key: &'static str| -> Result<u64, String> {
            u(key).ok_or_else(|| format!("dtype: `{key}` missing or not an integer"))
        };
        let b = |key: &str, default: bool| match v.get(key) {
            Some(Value::Bool(x)) => *x,
            _ => default,
        };
        Ok(match k.unwrap() {
            "float" => {
                let w = need("w")? as u16;
                let e = need("e")? as u16;
                let m = need("m")? as u16;
                if e == 0 || e > 32 || m > 52 || w > 64 || (e + m + 1) > w.max(1) + 1 {
                    return Err(format!("dtype: float w={w} e={e} m={m} is not encodable"));
                }
                let d = FloatFmt::ieee(w, e, m);
                let bias = match v.get("bias") {
                    Some(Value::U(n)) => *n as i32,
                    Some(Value::I(n)) => *n as i32,
                    _ => d.bias,
                };
                let nan = match v.get("nan").and_then(|x| x.as_str()) {
                    Some("ieee") => Nan::Ieee,
                    Some("fn") => Nan::Fn,
                    Some("none") => Nan::None,
                    Some(other) => return Err(format!("dtype: unknown nan mode `{other}`")),
                    None => d.nan,
                };
                let sign = match v.get("sign").and_then(|x| x.as_str()) {
                    Some("lead") => Sign::Lead,
                    Some("none") => Sign::None,
                    Some(other) => return Err(format!("dtype: unknown sign mode `{other}`")),
                    None => d.sign,
                };
                DType::Float(FloatFmt {
                    w,
                    e,
                    m,
                    bias,
                    sub: b("sub", d.sub),
                    inf: b("inf", d.inf),
                    nan,
                    sign,
                })
            }
            "int" => DType::Int {
                w: need("w")? as u16,
                signed: b("signed", true),
            },
            "fixed" => DType::Fixed {
                w: need("w")? as u16,
                signed: b("signed", true),
                frac: need("frac")? as u16,
            },
            "bool" => DType::Bool,
            "binary" => DType::Binary,
            "ternary" => DType::Ternary {
                pack: match v.get("pack").and_then(|x| x.as_str()) {
                    Some("b3x5") => TernPack::B3x5,
                    Some("naive") | None => TernPack::Naive,
                    Some(other) => return Err(format!("dtype: unknown ternary pack `{other}`")),
                },
            },
            "codebook" => {
                let r = v
                    .get("book")
                    .and_then(|x| x.as_array())
                    .ok_or_else(|| "dtype: codebook needs `book`".to_string())?;
                let d: Digest = r
                    .get(1)
                    .and_then(|x| x.as_bytes())
                    .and_then(|b| b.try_into().ok())
                    .ok_or_else(|| "dtype: codebook `book` is not a ref".to_string())?;
                DType::Codebook {
                    w: need("w")? as u16,
                    book: d,
                    dim: u("dim").unwrap_or(1) as u32,
                    shared: v
                        .get("shared")
                        .and_then(|x| x.as_str())
                        .and_then(Shared::parse)
                        .unwrap_or(Shared::PerTensor),
                }
            }
            "complex" => DType::Complex {
                re: Box::new(DType::from_value(
                    v.get("re")
                        .ok_or_else(|| "dtype: complex needs `re`".to_string())?,
                )?),
            },
            "struct" => {
                let mut fields = Vec::new();
                for f in v.get("fields").and_then(|x| x.as_array()).unwrap_or(&[]) {
                    let a = f
                        .as_array()
                        .ok_or_else(|| "dtype: struct field must be a pair".to_string())?;
                    let name = a
                        .first()
                        .and_then(|x| x.as_str())
                        .ok_or_else(|| "dtype: struct field name".to_string())?;
                    let t = DType::from_value(
                        a.get(1)
                            .ok_or_else(|| "dtype: struct field type".to_string())?,
                    )?;
                    fields.push((name.to_string(), t));
                }
                DType::Struct {
                    fields,
                    packed: b("packed", false),
                }
            }
            "opaque" => DType::Opaque {
                id: v
                    .get("id")
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| "dtype: opaque needs `id`".to_string())?
                    .to_string(),
                block_elems: need("block_elems")?,
                block_bytes: need("block_bytes")?,
            },
            "posit" => DType::Posit {
                w: need("w")? as u16,
                es: need("es")? as u16,
            },
            "logdom" => DType::LogDom {
                w: need("w")? as u16,
                base: u("base").unwrap_or(2) as u16,
                frac: need("frac")? as u16,
            },
            "string" => DType::Str,
            other => return Err(format!("dtype: unknown kind `{other}`")),
        })
    }

    /// A short human label for reports: the alias when there is one, else the
    /// structural form.
    pub fn label(&self) -> String {
        if let Some(a) = self.alias() {
            return a.to_string();
        }
        match self {
            DType::Float(f) => format!("float{}e{}m{}", f.w, f.e, f.m),
            DType::Int { w, signed } => format!("{}{}", if *signed { "i" } else { "u" }, w),
            DType::Fixed { w, frac, .. } => format!("q{}.{}", w - frac, frac),
            DType::Codebook { w, dim, .. } => format!("codebook{w}x{dim}"),
            DType::Complex { re } => format!("complex<{}>", re.label()),
            DType::Struct { fields, .. } => format!("struct<{} fields>", fields.len()),
            DType::Opaque { id, .. } => format!("opaque:{id}"),
            DType::Posit { w, es } => format!("posit{w}es{es}"),
            DType::LogDom { w, base, .. } => format!("logdom{w}base{base}"),
            DType::Str => "string".into(),
            DType::Ternary { .. } => "ternary".into(),
            DType::Binary => "binary".into(),
            DType::Bool => "bool".into(),
        }
    }
}

// ------------------------------------------------------------------- helpers --

fn int_value(n: i64) -> Value {
    if n < 0 {
        Value::I(n)
    } else {
        Value::U(n as u64)
    }
}

fn mask64(w: u16) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

fn int_range(w: u16, signed: bool) -> (f64, f64) {
    if signed {
        let hi = ldexp(1.0, w as i32 - 1) - 1.0;
        (-ldexp(1.0, w as i32 - 1), hi)
    } else {
        (0.0, ldexp(1.0, w as i32) - 1.0)
    }
}

fn sign_extend(raw: u64, w: u16) -> i64 {
    if w >= 64 {
        return raw as i64;
    }
    let sign = 1u64 << (w - 1);
    if raw & sign != 0 {
        (raw | !mask64(w)) as i64
    } else {
        raw as i64
    }
}

/// Reads `w` bits starting at bit `at` from a little-endian, LSB-first bit
/// stream — the canonical dense packing for sub-byte types (§04.4 `packed`
/// with `bit_order:"lsb-first"`).
pub fn read_bits(bytes: &[u8], at: u64, w: u16) -> Option<u64> {
    if w == 0 || w > 64 {
        return None;
    }
    let end = at + w as u64;
    if end > (bytes.len() as u64) * 8 {
        return None;
    }
    let mut out = 0u64;
    let mut got = 0u16;
    let mut pos = at;
    while got < w {
        let byte = bytes[(pos / 8) as usize] as u64;
        let bit_in_byte = (pos % 8) as u16;
        let take = (8 - bit_in_byte).min(w - got);
        let chunk = (byte >> bit_in_byte) & mask64(take);
        out |= chunk << got;
        got += take;
        pos += take as u64;
    }
    Some(out)
}

/// The inverse of [`read_bits`].
pub fn write_bits(bytes: &mut [u8], at: u64, w: u16, value: u64) -> bool {
    if w == 0 || w > 64 {
        return false;
    }
    if at + w as u64 > (bytes.len() as u64) * 8 {
        return false;
    }
    let mut done = 0u16;
    let mut pos = at;
    while done < w {
        let idx = (pos / 8) as usize;
        let bit_in_byte = (pos % 8) as u16;
        let take = (8 - bit_in_byte).min(w - done);
        let m = mask64(take) << bit_in_byte;
        let chunk = ((value >> done) & mask64(take)) << bit_in_byte;
        bytes[idx] = ((bytes[idx] as u64 & !m) | chunk) as u8;
        done += take;
        pos += take as u64;
    }
    true
}

/// Type III posit decoding: regime run-length, `es` exponent bits, then the
/// remaining bits are fraction. Included because §04.3.4 lists it and a
/// descriptor whose semantics nobody implements is a descriptor nobody can
/// trust.
fn decode_posit(raw: u64, w: u16, es: u16) -> f64 {
    if raw == 0 {
        return 0.0;
    }
    let sign_bit = 1u64 << (w - 1);
    if raw == sign_bit {
        return f64::NAN; // NaR
    }
    let negative = raw & sign_bit != 0;
    // Two's complement for negatives, then decode as positive.
    let bits = if negative {
        (raw ^ mask64(w)).wrapping_add(1) & mask64(w)
    } else {
        raw
    };
    let mut pos = w as i32 - 2; // first regime bit
    let first = (bits >> pos) & 1;
    let mut run = 0i32;
    while pos >= 0 && (bits >> pos) & 1 == first {
        run += 1;
        pos -= 1;
    }
    pos -= 1; // the terminating opposite bit
    let k = if first == 1 { run - 1 } else { -run };
    let mut exp = 0i64;
    for _ in 0..es {
        exp <<= 1;
        if pos >= 0 {
            exp |= ((bits >> pos) & 1) as i64;
            pos -= 1;
        }
    }
    let fbits = (pos + 1).max(0);
    let frac = if fbits > 0 {
        (bits & mask64(fbits as u16)) as f64 / ldexp(1.0, fbits)
    } else {
        0.0
    };
    let useed_exp = (1i64 << es) * k as i64 + exp;
    let mag = ldexp(1.0 + frac, useed_exp as i32);
    if negative {
        -mag
    } else {
        mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(t: &DType, xs: &[f64]) {
        let n = xs.len() as u64;
        let mut buf = vec![0u8; t.packed_bytes(n) as usize];
        for (i, x) in xs.iter().enumerate() {
            assert!(t.encode(&mut buf, i as u64, *x, Round::Rne), "{:?}", t);
        }
        for (i, x) in xs.iter().enumerate() {
            let got = t.decode(&buf, i as u64).unwrap();
            assert_eq!(got, *x, "{} element {i} of {:?}", t.label(), t);
        }
    }

    #[test]
    fn ieee_formats_round_trip_exactly_representable_values() {
        roundtrip(&DType::F64, &[0.0, 1.0, -1.0, 0.5, 1e300, -1e-300]);
        roundtrip(
            &DType::F32,
            &[0.0, 1.0, -2.5, 3.25, 1e30f32 as f64, -1e-30f32 as f64],
        );
        roundtrip(&DType::F16, &[0.0, 1.0, -2.5, 65504.0, 6.103515625e-5]);
        roundtrip(&DType::BF16, &[0.0, 1.0, -2.5, 3.3895313892515355e38]);
        roundtrip(&DType::F8E5M2, &[0.0, 1.0, -2.0, 57344.0]);
        roundtrip(&DType::F8E4M3, &[0.0, 1.0, -2.0, 448.0, 0.5]);
        roundtrip(&DType::F4E2M1, &[0.0, 0.5, 1.0, -6.0, 1.5, 3.0]);
        roundtrip(&DType::F6E2M3, &[0.0, 0.125, 1.0, -7.5]);
        roundtrip(&DType::F6E3M2, &[0.0, 1.0, -28.0, 0.0625]);
    }

    #[test]
    fn float_bit_patterns_match_ieee_754() {
        // f32 and f64 must agree with the host's own encoding, which is the
        // only independent implementation available inside a no-dependency
        // crate.
        for x in [
            1.0f32,
            -0.0,
            core::f32::consts::PI,
            1e-40,
            f32::MIN_POSITIVE,
        ] {
            let DType::Float(f) = DType::F32 else {
                unreachable!()
            };
            assert_eq!(
                f.encode(x as f64, Round::Rne) as u32,
                x.to_bits(),
                "encode {x}"
            );
            assert_eq!(f.decode(x.to_bits() as u64), x as f64, "decode {x}");
        }
        for x in [1.0f64, -2.5, 1e300, 5e-324] {
            let DType::Float(f) = DType::F64 else {
                unreachable!()
            };
            assert_eq!(f.encode(x, Round::Rne), x.to_bits(), "encode {x}");
            assert_eq!(f.decode(x.to_bits()), x, "decode {x}");
        }
        // f16 against known bit patterns.
        let DType::Float(h) = DType::F16 else {
            unreachable!()
        };
        assert_eq!(h.encode(1.0, Round::Rne), 0x3c00);
        assert_eq!(h.encode(-2.0, Round::Rne), 0xc000);
        assert_eq!(h.encode(65504.0, Round::Rne), 0x7bff);
        assert_eq!(h.decode(0x0001), 5.960464477539063e-8); // smallest subnormal
        assert!(h.decode(0x7e00).is_nan());
        assert_eq!(h.decode(0x7c00), f64::INFINITY);
        // bf16 is the top half of an f32.
        let DType::Float(b) = DType::BF16 else {
            unreachable!()
        };
        assert_eq!(b.encode(1.0, Round::Rne), 0x3f80);
        assert_eq!(b.decode(0x3f80), 1.0);
    }

    #[test]
    fn f8e4m3_has_no_infinity_and_saturates() {
        let DType::Float(f) = DType::F8E4M3 else {
            unreachable!()
        };
        // 0x7f is the NaN pattern; 0x7e is the largest finite value, 448.
        assert!(f.decode(0x7f).is_nan());
        assert_eq!(f.decode(0x7e), 448.0);
        assert_eq!(f.encode(f64::INFINITY, Round::Rne), 0x7e);
        assert_eq!(f.encode(1e9, Round::Rne), 0x7e);
        assert_eq!(f.encode(f64::NAN, Round::Rne), 0x7f);
    }

    #[test]
    fn e8m0_is_a_power_of_two_scale_type() {
        let DType::Float(f) = DType::E8M0 else {
            unreachable!()
        };
        assert_eq!(f.decode(127), 1.0);
        assert_eq!(f.decode(128), 2.0);
        assert_eq!(f.decode(126), 0.5);
        assert!(f.decode(255).is_nan());
        assert_eq!(f.encode(4.0, Round::Rne), 129);
        // Every decode is an exact power of two, which is what makes MX
        // dequantization bit-reproducible (§05.2.8).
        for b in 0..255u64 {
            let v = f.decode(b);
            assert_eq!(v, ldexp(1.0, b as i32 - 127));
        }
    }

    #[test]
    fn rounding_modes_are_distinguishable() {
        let DType::Float(f) = DType::F16 else {
            unreachable!()
        };
        // 1.0009765625 sits exactly halfway between two f16 values.
        let half = 1.0 + 1.0 / 2048.0;
        assert_eq!(f.encode(half, Round::Rne), 0x3c00); // ties to even
        assert_eq!(f.encode(half, Round::Rtz), 0x3c00);
        assert_eq!(f.encode(half, Round::Rup), 0x3c01);
        assert_eq!(f.encode(half, Round::Rdown), 0x3c00);
        assert_eq!(f.encode(-half, Round::Rup), 0xbc00);
        assert_eq!(f.encode(-half, Round::Rdown), 0xbc01);
        assert_eq!(f.encode(-half, Round::Rtz), 0xbc00);
        // Stochastic rounding is reproducible from its seed.
        let a = f.encode(half, Round::Stochastic { seed: 7, index: 3 });
        let b = f.encode(half, Round::Stochastic { seed: 7, index: 3 });
        assert_eq!(a, b);
        assert!(a == 0x3c00 || a == 0x3c01);
    }

    #[test]
    fn integer_and_fixed_point() {
        roundtrip(&DType::I8, &[0.0, 1.0, -1.0, 127.0, -128.0]);
        roundtrip(&DType::U4, &[0.0, 1.0, 15.0, 7.0]);
        roundtrip(&DType::I4, &[0.0, -8.0, 7.0, -1.0]);
        roundtrip(&DType::I2, &[-2.0, -1.0, 0.0, 1.0]);
        let q88 = DType::Fixed {
            w: 16,
            signed: true,
            frac: 8,
        };
        roundtrip(&q88, &[0.0, 1.0, -1.0, 0.5, 127.99609375, -128.0]);
        // Saturation rather than wraparound.
        let mut b = [0u8; 1];
        DType::I8.encode(&mut b, 0, 1e9, Round::Rne);
        assert_eq!(DType::I8.decode(&b, 0), Some(127.0));
    }

    #[test]
    fn sub_byte_packing_is_lsb_first_and_dense() {
        // Two u4 values in one byte: element 0 in the low nibble.
        let mut b = [0u8; 1];
        DType::U4.encode(&mut b, 0, 3.0, Round::Rne);
        DType::U4.encode(&mut b, 1, 10.0, Round::Rne);
        assert_eq!(b[0], 0xa3);
        assert_eq!(DType::U4.decode(&b, 0), Some(3.0));
        assert_eq!(DType::U4.decode(&b, 1), Some(10.0));
        assert_eq!(DType::U4.packed_bytes(9), 5);
        assert_eq!(DType::Bool.packed_bytes(9), 2);
    }

    #[test]
    fn ternary_b3x5_is_one_point_six_bits_per_value() {
        let t = DType::Ternary {
            pack: TernPack::B3x5,
        };
        assert_eq!(t.bits_rational(), (8, 5));
        assert_eq!(t.packed_bytes(5), 1);
        assert_eq!(t.packed_bytes(6), 2);
        assert_eq!(t.packed_bytes(1000), 200); // vs 250 for 2-bit packing
        roundtrip(&t, &[-1.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0]);
        // Base-3: [1,0,-1,0,0] -> digits 2,1,0,1,1 little-endian base 3.
        let mut b = vec![0u8; 1];
        for (i, x) in [1.0, 0.0, -1.0, 0.0, 0.0].iter().enumerate() {
            t.encode(&mut b, i as u64, *x, Round::Rne);
        }
        // digits (little-endian base 3): 2, 1, 0, 1, 1
        assert_eq!(b[0] as u64, 2 + 3 + 27 + 81);
    }

    #[test]
    fn binary_and_bool_are_distinct() {
        roundtrip(&DType::Binary, &[-1.0, 1.0, 1.0, -1.0]);
        roundtrip(&DType::Bool, &[0.0, 1.0, 1.0, 0.0]);
        assert_eq!(DType::Binary.bits(), 1);
    }

    #[test]
    fn opaque_sizes_by_block_and_refuses_element_access() {
        let t = DType::Opaque {
            id: "org.ggml/q4_K".into(),
            block_elems: 256,
            block_bytes: 144,
        };
        assert_eq!(t.packed_bytes(256), 144);
        assert_eq!(t.packed_bytes(512), 288);
        assert_eq!(t.packed_bytes(1), 144); // a partial block still costs one
        assert_eq!(t.decode(&[0u8; 144], 0), None);
        assert!(!t.is_numeric());
    }

    #[test]
    fn descriptors_round_trip_through_cbor() {
        let cases = vec![
            DType::F32,
            DType::BF16,
            DType::F8E4M3,
            DType::E8M0,
            DType::I4,
            DType::Bool,
            DType::Binary,
            DType::Ternary {
                pack: TernPack::B3x5,
            },
            DType::Fixed {
                w: 16,
                signed: true,
                frac: 8,
            },
            DType::Codebook {
                w: 4,
                book: [7u8; 32],
                dim: 1,
                shared: Shared::PerTensor,
            },
            DType::Complex {
                re: Box::new(DType::F32),
            },
            DType::Struct {
                fields: vec![("r".into(), DType::F32), ("i".into(), DType::F32)],
                packed: true,
            },
            DType::Opaque {
                id: "org.ggml/q4_K".into(),
                block_elems: 256,
                block_bytes: 144,
            },
            DType::Posit { w: 16, es: 2 },
            DType::LogDom {
                w: 8,
                base: 2,
                frac: 4,
            },
            DType::Str,
        ];
        for t in cases {
            let v = t.to_value();
            // The encoding must be canonical, which is what makes the digest
            // of a TensorDesc stable (§03.2).
            let bytes = v.encode();
            let decoded = crate::cbor::decode(&bytes).unwrap();
            assert_eq!(decoded.encode(), bytes, "{} re-encodes", t.label());
            assert_eq!(DType::from_value(&v).unwrap(), t, "{}", t.label());
            assert_eq!(
                DType::from_value(&decoded).unwrap(),
                t,
                "{} via cbor",
                t.label()
            );
        }
    }

    #[test]
    fn common_aliases_emit_no_redundant_fields() {
        // bf16 is the IEEE derivation for (16, 8, 7), so nothing beyond
        // alias/k/w/e/m is written. This keeps descriptors small and is why
        // the committed example container is stable.
        let v = DType::BF16.to_value();
        let keys: Vec<&str> = v
            .as_map()
            .unwrap()
            .iter()
            .filter_map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, vec!["alias", "k", "w", "e", "m"]);
        // f8e4m3 is not, and says so.
        let v = DType::F8E4M3.to_value();
        assert_eq!(v.get("inf"), Some(&Value::Bool(false)));
        assert_eq!(v.get("nan").and_then(|x| x.as_str()), Some("fn"));
    }

    #[test]
    fn unknown_alias_needs_a_descriptor() {
        let bare = Value::map(vec![("alias", Value::text("f13e4m8"))]);
        assert!(DType::from_value(&bare).is_err());
        let with_desc = Value::map(vec![
            ("alias", Value::text("f13e4m8")),
            ("k", Value::text("float")),
            ("w", Value::U(13)),
            ("e", Value::U(4)),
            ("m", Value::U(8)),
        ]);
        let t = DType::from_value(&with_desc).unwrap();
        assert_eq!(t.bits(), 13);
        // Fully understood: it decodes and encodes like any other float.
        let DType::Float(f) = t else { unreachable!() };
        assert_eq!(f.decode(f.encode(1.5, Round::Rne)), 1.5);
    }

    #[test]
    fn a_dtype_tagged_descriptor_is_accepted() {
        let tagged = Value::Tag(crate::cbor::TAG_DTYPE, Box::new(DType::F16.to_value()));
        assert_eq!(DType::from_value(&tagged).unwrap(), DType::F16);
    }

    #[test]
    fn posits_decode() {
        // 16-bit posit with es=2: 0x4000 is 1.0, and the type is symmetric.
        let t = DType::Posit { w: 16, es: 2 };
        let one = t.decode(&[0x00, 0x40], 0).unwrap();
        assert_eq!(one, 1.0);
        let neg = t.decode(&[0x00, 0xc0], 0).unwrap();
        assert_eq!(neg, -1.0);
        assert_eq!(t.decode(&[0x00, 0x00], 0), Some(0.0));
    }
}
