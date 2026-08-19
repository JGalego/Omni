//! SLH-DSA against NIST's own known-answer vectors.
//!
//! The same argument as `mldsa_vectors.rs`, and if anything a stronger one.
//! SLH-DSA is built from one hash function called through six wrappers that
//! differ only in what they are handed, and every hash is domain-separated by a
//! 32-byte address whose every field is part of the input. An implementation that
//! sets the wrong address field, clears it at the wrong moment, or concatenates
//! the two children of a Merkle node in the wrong order is internally consistent:
//! it generates keys, signs, and verifies its own signatures perfectly. It simply
//! is not SLH-DSA, and nothing short of somebody else's answers can tell.
//!
//! Fixtures come from `tools/acvp-vectors.py`; see that file for provenance and
//! for which cases were kept.

use std::collections::BTreeMap;

use omni_core::slhdsa::{self, Params};

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn records(name: &str) -> Vec<BTreeMap<String, String>> {
    let path = format!("{}/tests/vectors/slhdsa/{name}", env!("CARGO_MANIFEST_DIR"));
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

fn hexs(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Key generation across all six SHAKE parameter sets.
///
/// This is the expensive test and the broadest one. For an `s` set it builds a
/// top XMSS tree of 2^9 WOTS+ key pairs — a few hundred thousand SHAKE calls —
/// and it is worth it, because the root it produces is the single value that
/// every other part of the scheme is anchored to. If `wots_pk_gen`, `xmss_node`,
/// the address layout or the layer arithmetic is wrong anywhere, this root is
/// wrong.
#[test]
fn key_generation_matches_nists_vectors() {
    let recs = records("keygen.txt");
    let mut seen = BTreeMap::new();
    let mut skipped = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        // An `s` set's top tree is 512 WOTS+ key pairs, about a million SHAKE
        // calls, which is a quarter-minute each without optimisation. Skipping
        // them in a debug build keeps `cargo test` usable; CI runs this same test
        // in release, where nothing is skipped, and the skip is *named* here
        // rather than left to be inferred from a count.
        if cfg!(debug_assertions) && p.name.ends_with('s') {
            skipped.push(p.name);
            continue;
        }
        let kp = slhdsa::keygen(
            &p,
            &unhex(&r["skSeed"]),
            &unhex(&r["skPrf"]),
            &unhex(&r["pkSeed"]),
        );
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
    let want = if cfg!(debug_assertions) { 3 } else { 6 };
    assert_eq!(
        seen.len(),
        want,
        "expected {want} parameter sets, saw {seen:?}"
    );
    println!("keyGen: {} of {} vectors, {seen:?}", seen.len(), recs.len());
    if !skipped.is_empty() {
        println!(
            "keyGen: skipped {skipped:?} — slow without optimisation; \
             run `cargo test --release` to include them"
        );
    }
}

/// Signing, byte for byte. Only the fast parameter set: signing an `s` set walks
/// seven trees of 512 WOTS+ key pairs and takes seconds, and what it would
/// exercise beyond this is the tree *height*, which key generation already
/// covers for every set.
#[test]
fn signing_matches_nists_vectors_byte_for_byte() {
    let recs = records("siggen.txt");
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        let secret = unhex(&r["sk"]);
        let kp = slhdsa::KeyPair {
            params: p,
            public: secret[2 * p.n..].to_vec(),
            secret,
        };
        let msg = unhex(&r["msg"]);
        let ctx = unhex(r.get("ctx").map(String::as_str).unwrap_or(""));
        let sig = slhdsa::sign(&kp, &msg, &ctx).expect("signing must succeed");
        assert_eq!(
            hexs(&sig),
            r["sig"].to_lowercase(),
            "{} vector {i}: signature ({} byte message, {} byte context)",
            p.name,
            msg.len(),
            ctx.len()
        );
    }
    println!("sigGen: {} vectors", recs.len());
}

#[test]
fn verification_accepts_and_rejects_exactly_what_nist_says() {
    let recs = records("sigver.txt");
    let mut passes = 0;
    let mut fails = 0;
    for (i, r) in recs.iter().enumerate() {
        let p = params_of(r);
        let want = r["expect"] == "pass";
        let got = slhdsa::verify(
            &p,
            &unhex(&r["pk"]),
            &unhex(&r["msg"]),
            &unhex(r.get("ctx").map(String::as_str).unwrap_or("")),
            &unhex(&r["sig"]),
        );
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
    // Both outcomes, for the same reason as ML-DSA: a fixture set of nothing but
    // rejections is passed by a `verify` that always returns false.
    assert!(passes > 0, "no valid signatures in the fixtures");
    assert!(fails > 0, "no invalid signatures in the fixtures");
    println!("sigVer: {passes} accepted, {fails} rejected");
}
