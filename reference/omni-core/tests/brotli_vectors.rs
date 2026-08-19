//! Brotli against libbrotli.
//!
//! Brotli is a single library, not a spec with a field of implementations, so
//! there is nothing to write a second version *from* — the check that means
//! anything is whether this decoder reproduces what the reference encoder
//! produced. `tools/brotli-fixture.py` compresses a corpus with libbrotli at
//! three quality levels each, and every stream here must decode to its original
//! byte for byte.
//!
//! The corpus is chosen for the parts a decoder gets wrong: English prose and
//! HTML force static-dictionary references and the §8 transforms, which is the
//! machinery this crate declined to ship until the dictionary was available.

use std::collections::BTreeMap;

fn dir() -> String {
    format!("{}/tests/vectors/brotli", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn every_libbrotli_stream_decodes_byte_for_byte() {
    let manifest =
        std::fs::read_to_string(format!("{}/manifest.txt", dir())).expect("the brotli manifest");
    let mut by_case: BTreeMap<String, usize> = BTreeMap::new();
    let mut checked = 0usize;
    let mut used_dictionary = 0usize;

    for line in manifest.lines() {
        let (name, rest) = line.split_once('\t').expect("name<TAB>raw<TAB>comp");
        let raw_len: usize = rest.split('\t').next().unwrap().parse().unwrap();

        let comp = std::fs::read(format!("{}/{name}.br", dir())).expect("a .br");
        let want = std::fs::read(format!("{}/{name}.raw", dir())).expect("a .raw");
        assert_eq!(want.len(), raw_len, "{name}: manifest length disagrees");

        // The cap is the true length; the decoder must not need slack.
        let got = omni_core::brotli::decompress(&comp, want.len())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, want, "{name}: decoded bytes differ from the original");

        // Record which base cases were exercised, so a silently-truncated
        // fixture set is visible.
        let base = name.split('-').next().unwrap().to_string();
        *by_case.entry(base).or_default() += 1;
        checked += 1;

        // The prose and markup cases are the ones that force dictionary use;
        // note if the compressed form is much smaller than any literal-only or
        // backward-reference-only encoding could manage on this little text.
        if (name.starts_with("prose") || name.starts_with("html") || name.starts_with("mixed"))
            && comp.len() * 3 < want.len()
        {
            used_dictionary += 1;
        }
    }

    // Every base case present, both textual and binary, so the run cannot pass
    // by having quietly lost the hard ones.
    for base in [
        "prose", "html", "json", "repeat", "random", "floats", "mixed", "empty", "onebyte",
    ] {
        assert!(by_case.contains_key(base), "the {base} case is missing");
    }
    assert!(
        checked >= 27,
        "expected at least 27 streams, checked {checked}"
    );
    assert!(
        used_dictionary > 0,
        "no case compressed tightly enough to have exercised the dictionary"
    );
    println!("brotli: {checked} libbrotli streams decoded byte for byte, {by_case:?}");
}

#[test]
fn a_bounded_decode_refuses_to_exceed_its_cap() {
    // The floats case is 16 000 bytes; decoding it with a smaller cap must be an
    // error, not an over-allocation. This is §03.7.4's bound applied to brotli.
    let comp = std::fs::read(format!("{}/floats-q11.br", dir())).expect("a .br");
    let e = omni_core::brotli::decompress(&comp, 100);
    assert!(e.is_err(), "a too-small cap should refuse rather than grow");
}
