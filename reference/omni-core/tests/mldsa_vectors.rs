//! ML-DSA against NIST's own known-answer vectors.
//!
//! This is the test that decides whether `mldsa.rs` implements FIPS 204 or
//! merely implements itself. The unit tests in that module check internal
//! consistency — the NTT inverts, `Decompose` reconstructs, a signature this
//! code produced verifies under this code — and every one of them would pass for
//! an implementation that had, say, the two nibbles of `RejBoundedPoly` the wrong
//! way round, or `ExpandA`'s row and column indices swapped. Both produce
//! perfectly uniform-looking polynomials, valid-looking keys, and signatures that
//! verify. Against NIST's vectors, both fail on the first byte.
//!
//! Fixtures come from `tools/acvp-vectors.py`; see that file for provenance.

use std::collections::BTreeMap;

use omni_core::mldsa::{self, Params};

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// The fixtures are blank-line-separated records of `key value` lines. A parser
/// this simple is the point: a fixture format that needs a library is a fixture
/// format whose reader can be wrong.
fn records(name: &str) -> Vec<BTreeMap<String, String>> {
    let path = format!("{}/tests/vectors/mldsa/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = Vec::new();
    let mut cur = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.starts_with('#') {
            continue;
        }
        if line.is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        cur.insert(k.to_string(), v.to_string());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    assert!(!out.is_empty(), "{path} parsed to nothing");
    out
}

fn params_of(r: &BTreeMap<String, String>) -> Params {
    let name = r.get("set").expect("every record names its parameter set");
    Params::by_name(name).unwrap_or_else(|| panic!("unknown parameter set {name}"))
}

#[test]
fn key_generation_matches_nists_vectors() {
    let recs = records("keygen.txt");
    let mut seen = BTreeMap::new();
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        let seed = unhex(&r["seed"]);
        let seed: [u8; 32] = seed.try_into().expect("a 32-byte seed");
        let kp = mldsa::keygen(&p, &seed);
        assert_eq!(
            hexs(&kp.public),
            r["pk"].to_lowercase(),
            "{} vector {i}: public key",
            p.name
        );
        assert_eq!(
            hexs(&kp.secret),
            r["sk"].to_lowercase(),
            "{} vector {i}: secret key",
            p.name
        );
        *seen.entry(p.name).or_insert(0) += 1;
    }
    // Every parameter set must actually have been exercised; a fixture file that
    // silently lost a set would otherwise look like a pass.
    assert_eq!(
        seen.len(),
        3,
        "expected all three parameter sets, saw {seen:?}"
    );
    println!("keyGen: {} vectors, {seen:?}", recs.len());
}

#[test]
fn signing_matches_nists_vectors_byte_for_byte() {
    let recs = records("siggen.txt");
    let mut seen = BTreeMap::new();
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        let kp = mldsa::KeyPair {
            params: p,
            // Signing needs only the secret key; the public key is recomputed
            // from `pk` in the verifier, so an empty one here would be a bug in
            // this test rather than a shortcut.
            public: vec![0u8; p.public_key_len()],
            secret: unhex(&r["sk"]),
        };
        let msg = unhex(&r["msg"]);
        let ctx = unhex(r.get("ctx").map(String::as_str).unwrap_or(""));
        let sig = mldsa::sign(&kp, &msg, &ctx).expect("signing must succeed");
        assert_eq!(
            hexs(&sig),
            r["sig"].to_lowercase(),
            "{} vector {i}: signature ({} byte message, {} byte context)",
            p.name,
            msg.len(),
            ctx.len()
        );
        *seen.entry(p.name).or_insert(0) += 1;
    }
    assert_eq!(
        seen.len(),
        3,
        "expected all three parameter sets, saw {seen:?}"
    );
    println!("sigGen: {} vectors, {seen:?}", recs.len());
}

#[test]
fn verification_accepts_and_rejects_exactly_what_nist_says() {
    let recs = records("sigver.txt");
    let mut passes = 0;
    let mut fails = 0;
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        let pk = unhex(&r["pk"]);
        let msg = unhex(&r["msg"]);
        let ctx = unhex(r.get("ctx").map(String::as_str).unwrap_or(""));
        let sig = unhex(&r["sig"]);
        let want = r["expect"] == "pass";
        let got = mldsa::verify(&p, &pk, &msg, &ctx, &sig);
        assert_eq!(
            got,
            want,
            "{} vector {i}: expected {} ({}), got {got}",
            p.name,
            r["expect"],
            r.get("reason").map(String::as_str).unwrap_or("-")
        );
        if want {
            passes += 1;
        } else {
            fails += 1;
        }
    }
    // Both outcomes have to be present. A vector file of nothing but rejections
    // would be passed by a `verify` that always returns false, which is the one
    // wrong implementation easiest to write by accident.
    assert!(passes > 0, "no valid signatures in the fixtures");
    assert!(fails > 0, "no invalid signatures in the fixtures");
    println!("sigVer: {passes} accepted, {fails} rejected");
}

fn hexs(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
