//! zfp decompression — §03.7.1's `zfp`, the first of the two lossy codecs.
//!
//! zfp is a lossy compressor for floating-point arrays, and it is here for the
//! reason §03.7.3 exists: a container may legitimately hold weights that were
//! compressed with loss, and a reader that cannot decode them cannot read the
//! model. What makes it safe to *hold* such an object is the `LOSSY` flag and a
//! declared error bound; what makes it useful is a decoder, and this is that.
//!
//! It is the whole lossy pipeline of the format, run backwards: the header's
//! field metadata and compression mode, the per-block common exponent, the
//! embedded bit-plane coder with its unary group tests, the negabinary
//! conversion, the coefficient ordering by total sequency, and the inverse
//! decorrelating lifting transform — for 1D, 2D and 3D fields of `f32` and
//! `f64`, in every one of the fixed-rate, fixed-precision and fixed-accuracy
//! modes, which all reduce to the same four stream parameters.
//!
//! What is *not* here is stated where it is claimed and refused by name:
//! **reversible** mode is a different codec sharing the header (it has its own
//! block path, not a parameter setting), 4D fields need a fourth transform and a
//! 256-entry ordering, and the integer scalar types skip the block-floating-point
//! step entirely. Each is reported rather than approximated.
//!
//! There is no **encoder** here, for the same reason as brotli: this build has
//! nothing to gain by *producing* a lossy stream — §03.7.3 would make it declare
//! an error bound it did not choose — and the codec exists so that a container
//! somebody else wrote can be read and `repack`ed to a lossless codec.
//!
//! Like brotli, zfp is one library rather than a spec with a field of
//! implementations, so the check that means anything is differential:
//! `tools/zfp-fixture.py` compresses corpora with `zfpy` across dimensions,
//! types and modes, and this decoder must reproduce what that library's own
//! decompressor produces, value for value. The tables and rules below come from
//! LLNL/zfp's `src/template/` (BSD 3-Clause) — see the NOTICE beside the vectors.

/// zfp's codec version, the fourth byte of the magic. A stream from a future
/// codec revision is refused rather than guessed at.
const CODEC: u8 = 5;

const MAX_BITS: u32 = 16658;
const MAX_PREC: u32 = 64;
const MIN_EXP: i32 = -1074;

type Res<T> = Result<T, Error>;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Corrupt(&'static str),
    Unsupported(String),
    TooLarge,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Truncated => write!(f, "zfp stream ends mid-block"),
            Error::Corrupt(w) => write!(f, "corrupt zfp stream: {w}"),
            Error::Unsupported(w) => write!(f, "zfp: {w}"),
            Error::TooLarge => write!(f, "zfp output exceeds the declared bound"),
        }
    }
}

impl std::error::Error for Error {}

/// The scalar type a zfp field holds. The integer types exist in the format and
/// are refused here, so this enum names only what can be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    F64,
}

impl Scalar {
    fn bytes(self) -> usize {
        match self {
            Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }
}

/// A decoded zfp header: what the stream says it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub scalar: Scalar,
    /// 1, 2 or 3. A 4D field is refused.
    pub dims: usize,
    /// Extent along each axis; unused axes are 1.
    pub shape: [usize; 3],
    minbits: u32,
    maxbits: u32,
    maxprec: u32,
    minexp: i32,
}

impl Field {
    pub fn elements(&self) -> usize {
        self.shape[..self.dims].iter().product()
    }

    /// The logical size of the decoded array, which is what §03.7.4's bound is
    /// checked against before a single block is decoded.
    pub fn logical_len(&self) -> usize {
        self.elements() * self.scalar.bytes()
    }
}

// ---------------------------------------------------------------------------
// Bit reader. zfp's stream is a sequence of little-endian 64-bit words read
// LSB-first, which is byte-for-byte the same as reading the bytes LSB-first —
// the magic decodes as 'z','f','p' only under that reading, so it is checked
// rather than assumed.
// ---------------------------------------------------------------------------

struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Bits<'a> {
        Bits { data, at: 0 }
    }

    fn read(&mut self, n: u32) -> Res<u64> {
        debug_assert!(n <= 64);
        let mut v = 0u64;
        for i in 0..n as usize {
            let bit = self.at + i;
            if bit / 8 >= self.data.len() {
                return Err(Error::Truncated);
            }
            let b = (self.data[bit / 8] >> (bit % 8)) & 1;
            v |= (b as u64) << i;
        }
        self.at += n as usize;
        Ok(v)
    }

    fn bit(&mut self) -> Res<u64> {
        self.read(1)
    }

    /// `stream_skip`: advance without reading, which is what a block shorter
    /// than `minbits` costs.
    fn skip(&mut self, n: u32) -> Res<()> {
        let end = self.at + n as usize;
        if end.div_ceil(8) > self.data.len() {
            return Err(Error::Truncated);
        }
        self.at = end;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Header (zfp.h: magic 32 bits, field metadata 52, mode 12 or 64).
// ---------------------------------------------------------------------------

/// Parses the stream header, returning the field and the bit offset the first
/// block starts at.
pub fn header(data: &[u8]) -> Res<(Field, usize)> {
    let mut br = Bits::new(data);
    for want in *b"zfp" {
        if br.read(8)? != want as u64 {
            return Err(Error::Corrupt("not a zfp stream"));
        }
    }
    let version = br.read(8)? as u8;
    if version != CODEC {
        return Err(Error::Unsupported(format!(
            "codec version {version}, and this decoder implements {CODEC}"
        )));
    }

    // Field metadata: two bits of type, two of dimensionality, then the extents,
    // each stored as one less than the true value.
    let mut meta = br.read(52)?;
    let scalar = match meta & 3 {
        0 => {
            return Err(Error::Unsupported(
                "an int32 field (no block-float step)".into(),
            ))
        }
        1 => {
            return Err(Error::Unsupported(
                "an int64 field (no block-float step)".into(),
            ))
        }
        2 => Scalar::F32,
        _ => Scalar::F64,
    };
    meta >>= 2;
    let dims = (meta & 3) as usize + 1;
    meta >>= 2;
    let mut shape = [1usize; 3];
    match dims {
        1 => shape[0] = (meta & 0x0000_ffff_ffff) as usize + 1,
        2 => {
            shape[0] = (meta & 0xff_ffff) as usize + 1;
            shape[1] = ((meta >> 24) & 0xff_ffff) as usize + 1;
        }
        3 => {
            shape[0] = (meta & 0xffff) as usize + 1;
            shape[1] = ((meta >> 16) & 0xffff) as usize + 1;
            shape[2] = ((meta >> 32) & 0xffff) as usize + 1;
        }
        _ => {
            return Err(Error::Unsupported(
                "a 4D field, which needs a fourth transform and a 256-entry ordering".into(),
            ))
        }
    }

    // Compression mode: a 12-bit form covering the four named modes, or a 64-bit
    // form carrying the four parameters directly. Both land in the same place,
    // which is why one decoder serves every mode.
    let short = br.read(12)?;
    let mode = if short > (1 << 12) - 2 {
        short + (br.read(52)? << 12)
    } else {
        short
    };
    let (minbits, maxbits, maxprec, minexp) = if mode <= (1 << 12) - 2 {
        if mode < 2048 {
            // fixed rate
            let n = mode as u32 + 1;
            (n, n, MAX_PREC, MIN_EXP)
        } else if mode < 2048 + 128 {
            // fixed precision
            (1, MAX_BITS, mode as u32 + 1 - 2048, MIN_EXP)
        } else if mode == 2048 + 128 {
            return Err(Error::Unsupported(
                "reversible mode, which is a different block codec sharing this header".into(),
            ));
        } else {
            // fixed accuracy
            (
                1,
                MAX_BITS,
                MAX_PREC,
                mode as i32 + MIN_EXP - (2048 + 128 + 1),
            )
        }
    } else {
        let m = mode >> 12;
        let minbits = (m & 0x7fff) as u32 + 1;
        let maxbits = ((m >> 15) & 0x7fff) as u32 + 1;
        let maxprec = ((m >> 30) & 0x7f) as u32 + 1;
        let minexp = ((m >> 37) & 0x7fff) as i32 - 16495;
        (minbits, maxbits, maxprec, minexp)
    };
    if minexp < MIN_EXP {
        return Err(Error::Unsupported(
            "reversible mode, which is a different block codec sharing this header".into(),
        ));
    }
    if minbits > maxbits || maxprec == 0 || maxprec > MAX_PREC {
        return Err(Error::Corrupt("implausible compression parameters"));
    }
    Ok((
        Field {
            scalar,
            dims,
            shape,
            minbits,
            maxbits,
            maxprec,
            minexp,
        },
        br.at,
    ))
}

// ---------------------------------------------------------------------------
// Coefficient ordering (src/template/codec{1,2,3}.c): by total sequency, then
// by the sum of squares. Copied rather than derived, for the same reason as
// brotli's dictionary — the format is defined by these numbers.
// ---------------------------------------------------------------------------

const PERM_1: [u8; 4] = [0, 1, 2, 3];
const PERM_2: [u8; 16] = [0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15];
const PERM_3: [u8; 64] = [
    0, 1, 4, 16, 20, 17, 5, 2, 8, 32, 21, 6, 18, 24, 9, 33, 36, 3, 12, 48, 22, 25, 37, 40, 34, 10,
    7, 19, 28, 13, 49, 52, 41, 38, 26, 23, 29, 53, 11, 35, 44, 14, 50, 56, 42, 27, 39, 45, 30, 54,
    57, 60, 51, 15, 43, 46, 58, 61, 55, 31, 62, 59, 47, 63,
];

fn perm(dims: usize) -> &'static [u8] {
    match dims {
        1 => &PERM_1,
        2 => &PERM_2,
        _ => &PERM_3,
    }
}

/// `precision()`: how many bit planes a block may spend, given its exponent.
fn precision(maxexp: i32, maxprec: u32, minexp: i32, dims: usize) -> u32 {
    maxprec.min((maxexp - minexp + 2 * dims as i32 + 2).max(0) as u32)
}

/// `with_maxbits()`: whether the rate constraint actually binds. It decides
/// which of the two bit-plane decoders runs, and they are not interchangeable.
fn with_maxbits(maxbits: u32, maxprec: u32, size: u32) -> bool {
    (maxprec + 1) * size - 1 > maxbits
}

// ---------------------------------------------------------------------------
// The bit-plane coder (src/template/decode.c). Planes run from the most
// significant down; within a plane the first `n` bits are read directly and the
// rest are found by unary group tests, `n` carrying across planes.
// ---------------------------------------------------------------------------

/// The rate-constrained decoder: `maxbits` is a budget and a plane may be cut
/// short by it.
fn decode_ints_bounded(
    br: &mut Bits<'_>,
    maxbits: u32,
    maxprec: u32,
    intprec: u32,
    size: usize,
) -> Res<Vec<u64>> {
    let mut data = vec![0u64; size];
    let kmin = intprec.saturating_sub(maxprec);
    let mut bits = maxbits;
    let mut n = 0usize;
    let mut k = intprec;
    while bits > 0 && k > kmin {
        k -= 1;
        // Step 1: the first `n` bits of this plane, directly.
        let m = (n as u32).min(bits);
        bits -= m;
        let mut x = br.read(m)?;
        // Step 2: unary group tests for the rest of the plane.
        while bits > 0 && n < size {
            bits -= 1;
            if br.bit()? == 1 {
                // A one somewhere ahead: scan for it.
                while bits > 0 && n < size - 1 {
                    bits -= 1;
                    if br.bit()? == 1 {
                        break;
                    }
                    n += 1;
                }
                x += 1u64 << n;
                n += 1;
            } else {
                // Nothing more in this plane.
                break;
            }
        }
        // Step 3: deposit the plane.
        let mut i = 0usize;
        while x != 0 {
            data[i] += (x & 1) << k;
            x >>= 1;
            i += 1;
        }
    }
    Ok(data)
}

/// The variable-rate decoder: whole planes, no budget.
fn decode_ints_prec(br: &mut Bits<'_>, maxprec: u32, intprec: u32, size: usize) -> Res<Vec<u64>> {
    let mut data = vec![0u64; size];
    let kmin = intprec.saturating_sub(maxprec);
    let mut n = 0usize;
    let mut k = intprec;
    while k > kmin {
        k -= 1;
        let mut x = br.read(n as u32)?;
        while n < size && br.bit()? == 1 {
            while n < size - 1 && br.bit()? == 0 {
                n += 1;
            }
            x += 1u64 << n;
            n += 1;
        }
        let mut i = 0usize;
        while x != 0 {
            data[i] += (x & 1) << k;
            x >>= 1;
            i += 1;
        }
    }
    Ok(data)
}

/// The inverse lifting transform of a 4-vector, in the integer width the scalar
/// implies. The overflow is part of the transform — the original is written with
/// shifts that wrap — so the arithmetic has to happen at the right width.
macro_rules! inv_lift {
    ($p:expr, $off:expr, $s:expr, $t:ty) => {{
        let (p, off, s) = (&mut *$p, $off, $s);
        let mut x = p[off] as $t;
        let mut y = p[off + s] as $t;
        let mut z = p[off + 2 * s] as $t;
        let mut w = p[off + 3 * s] as $t;
        y = y.wrapping_add(w >> 1);
        w = w.wrapping_sub(y >> 1);
        y = y.wrapping_add(w);
        w = w.wrapping_sub(y.wrapping_sub(w));
        z = z.wrapping_add(x);
        x = x.wrapping_sub(z.wrapping_sub(x));
        y = y.wrapping_add(z);
        z = z.wrapping_sub(y.wrapping_sub(z));
        w = w.wrapping_add(x);
        x = x.wrapping_sub(w.wrapping_sub(x));
        p[off] = x as i64;
        p[off + s] = y as i64;
        p[off + 2 * s] = z as i64;
        p[off + 3 * s] = w as i64;
    }};
}

macro_rules! inv_xform {
    ($p:expr, $dims:expr, $t:ty) => {{
        let p = &mut *$p;
        match $dims {
            1 => inv_lift!(p, 0, 1, $t),
            2 => {
                for x in 0..4 {
                    inv_lift!(p, x, 4, $t);
                }
                for y in 0..4 {
                    inv_lift!(p, 4 * y, 1, $t);
                }
            }
            _ => {
                for y in 0..4 {
                    for x in 0..4 {
                        inv_lift!(p, x + 4 * y, 16, $t);
                    }
                }
                for x in 0..4 {
                    for z in 0..4 {
                        inv_lift!(p, 16 * z + x, 4, $t);
                    }
                }
                for z in 0..4 {
                    for y in 0..4 {
                        inv_lift!(p, 4 * y + 16 * z, 1, $t);
                    }
                }
            }
        }
    }};
}

/// Decodes one block into `out` (a `4^dims` buffer of scalars).
fn decode_block(br: &mut Bits<'_>, f: &Field, out: &mut [f64]) -> Res<()> {
    let size = 1usize << (2 * f.dims);
    let (ebits, ebias, intprec) = match f.scalar {
        Scalar::F32 => (8u32, 127i32, 32u32),
        Scalar::F64 => (11, 1023, 64),
    };
    let mut used = 1u32;
    if br.bit()? == 0 {
        // An all-zero block, which costs one bit plus any padding to `minbits`.
        out[..size].fill(0.0);
        if f.minbits > used {
            br.skip(f.minbits - used)?;
        }
        return Ok(());
    }
    used += ebits;
    let emax = br.read(ebits)? as i32 - ebias;
    let maxprec = precision(emax, f.maxprec, f.minexp, f.dims);

    let maxbits = f.maxbits.saturating_sub(used);
    let before = br.at;
    let ublock = if with_maxbits(maxbits, maxprec, size as u32) {
        decode_ints_bounded(br, maxbits, maxprec, intprec, size)?
    } else {
        decode_ints_prec(br, maxprec, intprec, size)?
    };
    let spent = (br.at - before) as u32;
    let minbits = f.minbits.saturating_sub(used.min(f.minbits));
    if spent < minbits {
        br.skip(minbits - spent)?;
    }

    // Negabinary to two's complement, through the sequency ordering.
    let mut iblock = vec![0i64; size];
    let pm = perm(f.dims);
    match f.scalar {
        Scalar::F32 => {
            const NB: u32 = 0xaaaa_aaaa;
            for (i, &u) in ublock.iter().enumerate() {
                let x = u as u32;
                iblock[pm[i] as usize] = ((x ^ NB).wrapping_sub(NB)) as i32 as i64;
            }
            inv_xform!(&mut iblock, f.dims, i32);
        }
        Scalar::F64 => {
            const NB: u64 = 0xaaaa_aaaa_aaaa_aaaa;
            for (i, &u) in ublock.iter().enumerate() {
                iblock[pm[i] as usize] = ((u ^ NB).wrapping_sub(NB)) as i64;
            }
            inv_xform!(&mut iblock, f.dims, i64);
        }
    }

    // Inverse block-floating-point: one power-of-two scale for the whole block.
    let shift = emax - (intprec as i32 - 2);
    for (o, &v) in out[..size].iter_mut().zip(iblock.iter()) {
        // `ldexp` by hand: the scale is exact, so the product rounds once.
        *o = match f.scalar {
            Scalar::F32 => (v as f32 * pow2_f32(shift)) as f64,
            Scalar::F64 => v as f64 * pow2_f64(shift),
        };
    }
    Ok(())
}

/// `2^e` as an exact `f32`, including the subnormal range, which a naive
/// `powi` gets wrong at the ends.
fn pow2_f32(e: i32) -> f32 {
    if e > 127 {
        f32::INFINITY
    } else if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e >= -149 {
        f32::from_bits(1u32 << (e + 149))
    } else {
        0.0
    }
}

fn pow2_f64(e: i32) -> f64 {
    if e > 1023 {
        f64::INFINITY
    } else if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else if e >= -1074 {
        f64::from_bits(1u64 << (e + 1074))
    } else {
        0.0
    }
}

/// Decompresses a zfp stream to the field's raw little-endian bytes, bounded by
/// `cap`.
///
/// The bound is checked from the *header* before any block is decoded, so a
/// stream claiming a field larger than the caller will accept costs nothing to
/// refuse — which is §03.7.4 applied one step earlier than it could be.
pub fn decompress(data: &[u8], cap: usize) -> Res<Vec<u8>> {
    let (f, start) = header(data)?;
    let n = f.elements();
    if f.logical_len() > cap {
        return Err(Error::TooLarge);
    }
    let mut out = vec![0f64; n];
    let mut br = Bits::new(data);
    br.at = start;

    let bsize = 1usize << (2 * f.dims);
    let mut block = vec![0f64; bsize];
    let [nx, ny, nz] = f.shape;

    // Blocks in the format's own order: x fastest, then y, then z, with the
    // partial blocks at the far edges decoded whole and scattered in part.
    let mut z = 0;
    while z < nz.max(1) {
        let mut y = 0;
        while y < ny.max(1) {
            let mut x = 0;
            while x < nx {
                decode_block(&mut br, &f, &mut block)?;
                let (bx, by, bz) = (
                    (nx - x).min(4),
                    if f.dims >= 2 { (ny - y).min(4) } else { 1 },
                    if f.dims >= 3 { (nz - z).min(4) } else { 1 },
                );
                for k in 0..bz {
                    for j in 0..by {
                        for i in 0..bx {
                            let src = i + 4 * j + 16 * k;
                            let dst = (x + i) + nx * ((y + j) + ny * (z + k));
                            out[dst] = block[src];
                        }
                    }
                }
                x += 4;
            }
            y += 4;
            if f.dims < 2 {
                break;
            }
        }
        z += 4;
        if f.dims < 3 {
            break;
        }
    }

    let mut bytes = Vec::with_capacity(f.logical_len());
    match f.scalar {
        Scalar::F32 => {
            for v in &out {
                bytes.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
        Scalar::F64 => {
            for v in &out {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}
