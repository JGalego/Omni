//! WebAssembly SIMD against a real engine.
//!
//! §11.6 permits the fixed-width, deterministic SIMD subset, and the plugin host
//! implements it: some 230 instructions, each with exact lane semantics. A table
//! that size is not checkable by reading it. The saturating adds, the two
//! different NaN rules that separate `min` from `pmin`, the rounding Q15
//! multiply, the narrowing saturations, the `trunc_sat` clamps and the lane
//! order of every `extend`/`extmul` half are each a place where a plausible
//! implementation is wrong and its own tests agree with it.
//!
//! So the check is differential. `tools/wasm-simd-fixture.py` builds one module
//! per case, runs it under `wasmtime`, and records the sixteen result bytes;
//! this host runs the same modules and must produce the same bytes. Determinism
//! is the whole requirement §11.6 makes of a plugin, and two independent engines
//! agreeing byte for byte is what it looks like when it holds.

use std::collections::BTreeMap;

use omni_core::wasm::{Env, Instance, Limits, Module};

fn dir() -> String {
    format!("{}/tests/vectors/wasmsimd", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn every_simd_case_agrees_with_wasmtime_byte_for_byte() {
    let manifest =
        std::fs::read_to_string(format!("{}/manifest.txt", dir())).expect("the SIMD manifest");
    let env = Env::default();
    let mut checked = 0usize;
    let mut families: BTreeMap<&str, usize> = BTreeMap::new();
    let mut failures = Vec::new();

    for line in manifest.lines() {
        let (name, want_hex) = line.split_once('\t').expect("name<TAB>hex");
        let want = unhex(want_hex);
        let bytes = std::fs::read(format!("{}/{name}.wasm", dir())).expect("a .wasm");

        let m = match Module::load(&bytes) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: load: {e}"));
                continue;
            }
        };
        let mut inst = match Instance::new(&m, &env, Limits::default()) {
            Ok(i) => i,
            Err(e) => {
                failures.push(format!("{name}: instantiate: {e}"));
                continue;
            }
        };
        if let Err(e) = inst.call("run", &[]) {
            failures.push(format!("{name}: call: {e}"));
            continue;
        }
        let got = match inst.read(0, 16) {
            Ok(g) => g,
            Err(e) => {
                failures.push(format!("{name}: read: {e}"));
                continue;
            }
        };
        if got != want && !nan_equivalent(name, &got, &want) {
            failures.push(format!("{name}: got {} want {}", hexs(&got), hexs(&want)));
            continue;
        }
        // The family is the lane shape, so a set of fixtures that lost a whole
        // shape is visible rather than merely smaller.
        let family = name.split('_').next().unwrap_or(name);
        *families
            .entry(match family {
                "i8x16" | "i16x8" | "i32x4" | "i64x2" | "f32x4" | "f64x2" | "v128" => family,
                _ => "other",
            })
            .or_default() += 1;
        checked += 1;
    }

    assert!(
        failures.is_empty(),
        "{} of {} SIMD cases disagree with wasmtime:\n{}",
        failures.len(),
        failures.len() + checked,
        failures.join("\n")
    );
    // Every lane shape present, so a truncated fixture set cannot pass by having
    // quietly lost the hard ones.
    for f in ["i8x16", "i16x8", "i32x4", "i64x2", "f32x4", "f64x2", "v128"] {
        assert!(families.contains_key(f), "the {f} cases are missing");
    }
    assert!(
        checked >= 200,
        "expected at least 200 SIMD cases, checked {checked}"
    );
    println!("wasm SIMD: {checked} cases agree with wasmtime, {families:?}");
}

/// Relaxed-SIMD shares the `0xfd` prefix and is *forbidden* rather than merely
/// absent: its results are permitted to differ between hosts, and a plugin whose
/// output depends on the engine is the one thing §11.6 rules out. The check is
/// that it is refused at load, before anything runs.
#[test]
fn a_relaxed_simd_instruction_is_forbidden_at_load() {
    // `0xfd 0x100` is `i8x16.relaxed_swizzle`, LEB-encoded as 0x80 0x02.
    let mut b = Vec::new();
    b.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
    // type: () -> ()
    b.extend_from_slice(&[0x01, 0x04, 0x01, 0x60, 0x00, 0x00]);
    // function: one, type 0
    b.extend_from_slice(&[0x03, 0x02, 0x01, 0x00]);
    // code: one body containing the relaxed instruction
    let body = [0x00u8, 0xfd, 0x80, 0x02, 0x0b];
    b.push(0x0a);
    // Section payload: the function count, the body's length, and the body.
    b.push((body.len() + 2) as u8);
    b.push(0x01);
    b.push(body.len() as u8);
    b.extend_from_slice(&body);

    match Module::load(&b) {
        Err(e) => {
            let m = format!("{e}");
            assert!(
                m.contains("relaxed") && m.contains("nondeterministic"),
                "refused, but not for being nondeterministic: {m}"
            );
        }
        Ok(_) => panic!("a relaxed-SIMD module loaded"),
    }
}

/// Whether two results differ only in the bits of a NaN that the specification
/// leaves unspecified.
///
/// This is the one place the comparison is deliberately looser than byte
/// equality, and the reason is in the specification rather than in convenience.
/// When an arithmetic operation returns NaN, WebAssembly permits *any* NaN: the
/// payload and the sign are both unconstrained, so two conforming engines may
/// legitimately differ, and asserting bit equality there would be testing
/// wasmtime rather than the specification. This host canonicalises instead,
/// which is what §11.6 asks of it — the same input always gives the same output.
///
/// The looseness is bounded three ways: it applies only to cases whose result is
/// a float vector, only to lanes where *both* engines produced a NaN, and never
/// to `abs`, `neg`, or `copysign`, whose behaviour on a NaN is specified exactly
/// as a bit operation. A lane where one side is NaN and the other is a number is
/// still a failure.
fn nan_equivalent(name: &str, got: &[u8], want: &[u8]) -> bool {
    if name.contains("_abs") || name.contains("_neg") || name.contains("copysign") {
        return false;
    }
    if name.starts_with("f32x4_") {
        let (g, w) = (lanes32(got), lanes32(want));
        (0..4).all(|i| g[i] == w[i] || (g[i].is_nan() && w[i].is_nan()))
    } else if name.starts_with("f64x2_") {
        let (g, w) = (lanes64(got), lanes64(want));
        (0..2).all(|i| g[i] == w[i] || (g[i].is_nan() && w[i].is_nan()))
    } else {
        false
    }
}

fn lanes32(b: &[u8]) -> [f32; 4] {
    std::array::from_fn(|i| f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

fn lanes64(b: &[u8]) -> [f64; 2] {
    std::array::from_fn(|i| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn hexs(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
