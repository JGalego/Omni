//! zfp against zfpy.
//!
//! zfp is a single library, not a specification with a field of implementations,
//! so there is nothing to write a second version *from* — the check that means
//! anything is whether this decoder reproduces what the reference implementation
//! produces. And because zfp is lossy, the thing to reproduce is what the
//! *stream* holds, which is what `zfpy`'s own decompressor returns, not the array
//! the stream was made from. Comparing against the original would be measuring
//! zfp's error rather than this decoder's correctness.
//!
//! The corpus spans what changes the bitstream rather than what changes its size:
//! one, two and three dimensions, `f32` and `f64`, all three lossy modes, extents
//! that are not multiples of four so the partial-block path runs, arrays of zeros
//! so the one-bit block form runs, and a block whose exponent range is extreme.
//! `tools/zfp-fixture.py` builds it.

use std::collections::BTreeMap;

fn dir() -> String {
    format!("{}/tests/vectors/zfp", env!("CARGO_MANIFEST_DIR"))
}

struct Case {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    raw_len: usize,
}

fn manifest() -> Vec<Case> {
    let text =
        std::fs::read_to_string(format!("{}/manifest.txt", dir())).expect("the zfp manifest");
    text.lines()
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert!(f.len() >= 4, "manifest line: {l}");
            Case {
                name: f[0].to_string(),
                dtype: f[1].to_string(),
                shape: f[2].split('x').map(|n| n.parse().unwrap()).collect(),
                raw_len: f[3].parse().unwrap(),
            }
        })
        .collect()
}

#[test]
fn every_zfpy_stream_decodes_to_what_zfpy_makes_of_it() {
    let cases = manifest();
    let mut by_dims: BTreeMap<usize, usize> = BTreeMap::new();
    let mut by_mode: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = Vec::new();

    for c in &cases {
        let comp = std::fs::read(format!("{}/{}.zfp", dir(), c.name)).expect("a .zfp");
        let want = std::fs::read(format!("{}/{}.raw", dir(), c.name)).expect("a .raw");
        assert_eq!(
            want.len(),
            c.raw_len,
            "{}: manifest length disagrees",
            c.name
        );

        // The header alone has to describe the field the fixture says it is,
        // before any block is decoded — a reader plans allocations from it.
        match omni_core::zfp::header(&comp) {
            Ok((f, _)) => {
                if f.dims != c.shape.len() {
                    failures.push(format!(
                        "{}: header says {} dims, fixture is {}",
                        c.name,
                        f.dims,
                        c.shape.len()
                    ));
                    continue;
                }
                // zfp's axes run fastest-first, and numpy's shape is written
                // slowest-first, so the fixture's shape reversed is the field's.
                let want_shape: Vec<usize> = c.shape.iter().rev().copied().collect();
                if f.shape[..f.dims] != want_shape[..] {
                    failures.push(format!(
                        "{}: header shape {:?}, fixture {:?}",
                        c.name,
                        &f.shape[..f.dims],
                        want_shape
                    ));
                    continue;
                }
                if f.logical_len() != c.raw_len {
                    failures.push(format!(
                        "{}: header implies {} bytes, fixture has {}",
                        c.name,
                        f.logical_len(),
                        c.raw_len
                    ));
                    continue;
                }
            }
            Err(e) => {
                failures.push(format!("{}: header: {e}", c.name));
                continue;
            }
        }

        // The cap is the true length, so the decoder must not need slack.
        match omni_core::zfp::decompress(&comp, want.len()) {
            Ok(got) if got == want => {}
            Ok(got) => {
                let n = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
                failures.push(format!(
                    "{}: {} of {} bytes differ, first at {n}",
                    c.name,
                    want.len() - n,
                    want.len()
                ));
            }
            Err(e) => failures.push(format!("{}: {e}", c.name)),
        }

        *by_dims.entry(c.shape.len()).or_default() += 1;
        // The mode tag is everything after the first hyphen — `a1e-3` has one
        // of its own, so splitting from the right would find `3`.
        let mode = c.name.split_once('-').map_or("?", |x| x.1).to_string();
        *by_mode.entry(mode).or_default() += 1;
        *by_type.entry(c.dtype.clone()).or_default() += 1;
    }

    assert!(
        failures.is_empty(),
        "{} of {} zfp streams disagree with zfpy:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    // Every axis of the corpus present, so a truncated fixture set cannot pass
    // by having quietly lost the hard cases.
    for d in [1usize, 2, 3] {
        assert!(by_dims.contains_key(&d), "no {d}D cases");
    }
    for m in ["r8", "r16", "p14", "p22", "a1e-3", "a1e-6"] {
        assert!(by_mode.contains_key(m), "no {m} cases");
    }
    for t in ["float32", "float64"] {
        assert!(by_type.contains_key(t), "no {t} cases");
    }
    assert!(cases.len() >= 90, "expected at least 90 streams");
    println!(
        "zfp: {} streams match zfpy, dims {by_dims:?}, modes {by_mode:?}, types {by_type:?}",
        cases.len()
    );
}

/// §03.7.4's bound, applied from the header rather than after decoding: a stream
/// that declares a field larger than the caller will take is refused before a
/// single block is read.
#[test]
fn a_field_larger_than_the_cap_is_refused_before_decoding() {
    let cases = manifest();
    let c = cases
        .iter()
        .find(|c| c.raw_len > 1000)
        .expect("a fixture big enough to bound");
    let comp = std::fs::read(format!("{}/{}.zfp", dir(), c.name)).expect("a .zfp");
    assert!(omni_core::zfp::decompress(&comp, c.raw_len).is_ok());
    let e = omni_core::zfp::decompress(&comp, c.raw_len - 1);
    assert!(e.is_err(), "a too-small cap should refuse");
}

/// The parts of the format this build does not implement are refused by name
/// rather than approximated. Reversible mode is the one that matters: it shares
/// the header and would otherwise look like an ordinary stream.
#[test]
fn what_is_not_implemented_is_refused_by_name() {
    // A valid header, edited. The mode field's short form is at bit 84; 2176
    // (2048 + 128) selects reversible.
    let cases = manifest();
    let c = &cases[0];
    let comp = std::fs::read(format!("{}/{}.zfp", dir(), c.name)).expect("a .zfp");

    let mut rev = comp.clone();
    write_bits(&mut rev, 84, 12, 2048 + 128);
    match omni_core::zfp::decompress(&rev, 1 << 20) {
        Err(omni_core::zfp::Error::Unsupported(m)) => {
            assert!(m.contains("reversible"), "{m}");
        }
        other => panic!("reversible mode was not refused by name: {other:?}"),
    }

    // A 4D field: the dimensionality is two bits at offset 34 (32 magic + 2 type).
    let mut four = comp.clone();
    write_bits(&mut four, 34, 2, 3);
    match omni_core::zfp::decompress(&four, 1 << 20) {
        Err(omni_core::zfp::Error::Unsupported(m)) => assert!(m.contains("4D"), "{m}"),
        other => panic!("a 4D field was not refused by name: {other:?}"),
    }

    // An integer field: type 0 is int32.
    let mut ints = comp.clone();
    write_bits(&mut ints, 32, 2, 0);
    match omni_core::zfp::decompress(&ints, 1 << 20) {
        Err(omni_core::zfp::Error::Unsupported(m)) => assert!(m.contains("int32"), "{m}"),
        other => panic!("an int32 field was not refused by name: {other:?}"),
    }

    // And a stream from a codec revision this decoder does not implement.
    let mut future = comp.clone();
    future[3] = 9;
    match omni_core::zfp::decompress(&future, 1 << 20) {
        Err(omni_core::zfp::Error::Unsupported(m)) => {
            assert!(m.contains("codec version"), "{m}")
        }
        other => panic!("a future codec version was not refused: {other:?}"),
    }

    // A truncated stream is an error rather than a panic, at every length.
    for cut in 1..comp.len().min(400) {
        let _ = omni_core::zfp::decompress(&comp[..cut], 1 << 20);
    }
}

/// Writes `n` bits of `v` at bit offset `at`, LSB-first — the same order the
/// decoder reads, so a test that edits a header edits what the header says.
fn write_bits(buf: &mut [u8], at: usize, n: u32, v: u64) {
    for i in 0..n as usize {
        let bit = at + i;
        let mask = 1u8 << (bit % 8);
        if (v >> i) & 1 == 1 {
            buf[bit / 8] |= mask;
        } else {
            buf[bit / 8] &= !mask;
        }
    }
}
