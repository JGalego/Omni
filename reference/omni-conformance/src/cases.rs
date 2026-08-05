//! The corpus: every case, the rule it exercises, and the behaviour a
//! conforming implementation must show.
//!
//! Cases are *generated* rather than committed as opaque blobs, so that each
//! one has a stated reason and a rule ID next to the mutation that produces
//! it. A corpus of hand-made bad files nobody can regenerate rots; this one
//! rebuilds from source and CI fails if it drifts.

use omni_core::cbor::Value;
use omni_core::container::{
    oflags, otype, pack, HashAlgo, Object, PackOptions, HEADER_SIZE, MAGIC, SEG_HEADER_SIZE,
    TRAILER_SIZE,
};
use omni_core::crc32c::crc32c;
use omni_core::{Container, DType, Digest, ModelBuilder, TensorSpec};

/// What a conforming implementation must do with a case.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Expect {
    /// Load and validate cleanly. Exit 0.
    Accept,
    /// Refuse the file. Exit 1, citing the rule.
    Reject,
    /// Load structurally, but decline to act on what it does not understand.
    /// Exit 0 or 3 — never 1. Rejecting these outright is a conformance
    /// failure, because it is how a 2026 reader stays useful in 2071.
    Degrade,
}

impl Expect {
    pub fn name(self) -> &'static str {
        match self {
            Expect::Accept => "accept",
            Expect::Reject => "reject",
            Expect::Degrade => "degrade",
        }
    }
}

pub struct Case {
    pub category: &'static str,
    pub name: &'static str,
    pub expect: Expect,
    /// The normative rule this case exercises, where one applies.
    pub rule: Option<&'static str>,
    pub why: &'static str,
    pub bytes: Vec<u8>,
}

// ------------------------------------------------------------- foundations --

/// The smallest thing that is still a model: a manifest, metadata, one model
/// object, one tensor. Small enough that the whole corpus stays under a
/// megabyte in git.
fn base(hash: HashAlgo, align: u8) -> (Vec<u8>, Vec<Object>, Digest) {
    let (objs, root) = ModelBuilder::new("omni/conformance")
        .hash(hash)
        .chunk_size(256)
        .arch("test", vec![("hidden_size", Value::U(8))])
        .tensor(TensorSpec {
            name: "w".into(),
            shape: vec![8, 8],
            dtype: DType::F32,
            axes: None,
            semantic: "weight",
            data: (0..8 * 8 * 4).map(|i| (i % 251) as u8).collect(),
            layout: None,
        })
        .build();
    let opts = PackOptions {
        hash,
        log2_align: align,
        creator: "omni-conformance".into(),
        reproducible: true,
        ..Default::default()
    };
    (pack(&objs, &root, &opts).unwrap(), objs, root)
}

fn fix_header_crc(b: &mut [u8]) {
    let c = crc32c(&b[0..124]);
    b[124..128].copy_from_slice(&c.to_le_bytes());
}

fn fix_trailer_crc(b: &mut [u8]) {
    let t = b.len() - TRAILER_SIZE;
    let c = crc32c(&b[t..t + 52]);
    b[t + 52..t + 56].copy_from_slice(&c.to_le_bytes());
}

fn fix_segment_crcs(b: &mut [u8], hdr: usize, plen: usize) {
    let p = hdr + SEG_HEADER_SIZE;
    let pc = crc32c(&b[p..p + plen]);
    b[hdr + 24..hdr + 28].copy_from_slice(&pc.to_le_bytes());
    let hc = crc32c(&b[hdr..hdr + 28]);
    b[hdr + 28..hdr + 32].copy_from_slice(&hc.to_le_bytes());
}

/// Segments of a container, as `(header_offset, kind, payload_len)`.
fn segments(bytes: &[u8]) -> Vec<(usize, u16, u64)> {
    omni_core::container::scan_segments(bytes).unwrap()
}

fn case(
    category: &'static str,
    name: &'static str,
    expect: Expect,
    rule: Option<&'static str>,
    why: &'static str,
    bytes: Vec<u8>,
) -> Case {
    Case {
        category,
        name,
        expect,
        rule,
        why,
        bytes,
    }
}

/// Builds the whole corpus.
pub fn corpus() -> Vec<Case> {
    let mut out = Vec::new();
    out.extend(valid_minimal());
    out.extend(invalid_framing());
    out.extend(invalid_encoding());
    out.extend(forward());
    out
}

// ---------------------------------------------------------- valid/minimal --

fn valid_minimal() -> Vec<Case> {
    let mut out = Vec::new();

    for (hash, name) in [
        (HashAlgo::Blake3_256, "blake3-256"),
        (HashAlgo::Sha256, "sha2-256"),
    ] {
        let (bytes, _, _) = base(hash, 12);
        out.push(case(
            "valid/minimal",
            match hash {
                HashAlgo::Blake3_256 => "hash-blake3",
                HashAlgo::Sha256 => "hash-sha256",
            },
            Expect::Accept,
            None,
            match hash {
                HashAlgo::Blake3_256 => "The default digest algorithm (§03.5.1).",
                HashAlgo::Sha256 => "The mandatory interoperability algorithm (§03.5.1).",
            },
            bytes,
        ));
        let _ = name;
    }

    // Alignment is configurable (§02.9), and a reader that hardcodes 4096
    // passes every test until it meets one of these. 64 bytes is one
    // cacheline; 64 KiB is a plausible huge-page-adjacent choice.
    //
    // Larger alignments are legal up to 2^30 and the generator will produce
    // them, but they are not committed: at 1 MiB this model pads out to 2.1 MB,
    // which is the honest cost §02.9 describes and not worth carrying in git.
    for (log2, name) in [(6u8, "align-64"), (16, "align-64k")] {
        let (bytes, _, _) = base(HashAlgo::Blake3_256, log2);
        out.push(case(
            "valid/minimal",
            name,
            Expect::Accept,
            Some("R-C04"),
            "Alignment is a container-wide choice; readers must honour it, not assume 4096.",
            bytes,
        ));
    }

    // A manifest and nothing else. No tensors, no data objects, hence no BLOB
    // segment at all — a reader that assumes one exists fails here.
    let manifest = Object::structure(
        otype::MANIFEST,
        &Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("model")),
            ("created", Value::U(0)),
        ]),
    );
    let root = manifest.digest(HashAlgo::Blake3_256);
    let bytes = pack(
        &[manifest],
        &root,
        &PackOptions {
            creator: "omni-conformance".into(),
            ..Default::default()
        },
    )
    .unwrap();
    out.push(case(
        "valid/minimal",
        "manifest-only",
        Expect::Accept,
        None,
        "The smallest legal container: one object, no data segment.",
        bytes,
    ));

    out
}

// -------------------------------------------------------- invalid/framing --

fn invalid_framing() -> Vec<Case> {
    let mut out = Vec::new();
    let (good, _, _) = base(HashAlgo::Blake3_256, 12);

    // R-C01 magic.
    let mut b = good.clone();
    b[3] = b'X';
    fix_header_crc(&mut b);
    out.push(case(
        "invalid/framing",
        "magic-wrong",
        Expect::Reject,
        Some("R-C01"),
        "The magic is how a file is identified before anything else is trusted.",
        b,
    ));

    // The magic's own robustness features, each defeated.
    let mut b = good.clone();
    b[0] = 0x0a; // the high bit that catches 7-bit-stripping transports
    fix_header_crc(&mut b);
    out.push(case(
        "invalid/framing",
        "magic-high-bit-stripped",
        Expect::Reject,
        Some("R-C01"),
        "0x89 detects transports that strip the eighth bit; a stripped file is not valid.",
        b,
    ));

    // R-C02 header CRC.
    let mut b = good.clone();
    b[96] = b'X'; // creator, a field nothing depends on
    out.push(case(
        "invalid/framing",
        "header-crc-wrong",
        Expect::Reject,
        Some("R-C02"),
        "Any header edit without a CRC fixup must be caught, even to an inert field.",
        b,
    ));

    // R-C03 header_size.
    for (v, name) in [
        (64u16, "header-size-too-small"),
        (8192, "header-size-too-big"),
    ] {
        let mut b = good.clone();
        b[14..16].copy_from_slice(&v.to_le_bytes());
        fix_header_crc(&mut b);
        out.push(case(
            "invalid/framing",
            name,
            Expect::Reject,
            Some("R-C03"),
            "header_size outside [128, 4096] cannot be honoured by a forward-compatible reader.",
            b,
        ));
    }

    // R-C04 log2_align.
    for (v, name) in [(5u8, "align-too-small"), (31, "align-too-big")] {
        let mut b = good.clone();
        b[13] = v;
        fix_header_crc(&mut b);
        out.push(case(
            "invalid/framing",
            name,
            Expect::Reject,
            Some("R-C04"),
            "log2_align outside [6, 30] is not a supportable alignment.",
            b,
        ));
    }

    // R-C05 unknown hash algorithm. Not skippable: every digest becomes
    // uninterpretable, including the root.
    let mut b = good.clone();
    b[32] = 0x99;
    fix_header_crc(&mut b);
    out.push(case(
        "invalid/framing",
        "hash-algorithm-unknown",
        Expect::Reject,
        Some("R-C05"),
        "An unknown digest algorithm makes every identity in the file unverifiable.",
        b,
    ));

    // R-C05 segment payload CRC.
    let segs = segments(&good);
    let (obj_hdr, _, obj_len) = *segs
        .iter()
        .find(|(_, k, _)| *k == omni_core::seg::OBJ)
        .unwrap();
    let mut b = good.clone();
    b[obj_hdr + SEG_HEADER_SIZE + 4] ^= 0xff;
    out.push(case(
        "invalid/framing",
        "segment-payload-crc-wrong",
        Expect::Reject,
        Some("R-C05"),
        "A segment whose payload CRC fails is damaged, whatever its objects hash to.",
        b,
    ));

    // R-C07 padding must be zero.
    if let Some(&(pad_hdr, _, pad_len)) = segs.iter().find(|(_, k, _)| *k == omni_core::seg::PAD) {
        if pad_len > 0 {
            let mut b = good.clone();
            b[pad_hdr + SEG_HEADER_SIZE] = 0x41;
            fix_segment_crcs(&mut b, pad_hdr, pad_len as usize);
            out.push(case(
                "invalid/framing",
                "padding-nonzero",
                Expect::Reject,
                Some("R-C07"),
                "Non-zero padding hides data in space that is declared to hold none.",
                b,
            ));
        }
    }

    // R-C09 trailer.
    let mut b = good.clone();
    let t = b.len() - 1;
    b[t] = 0x00;
    fix_trailer_crc(&mut b);
    out.push(case(
        "invalid/framing",
        "trailer-magic-wrong",
        Expect::Reject,
        Some("R-C09"),
        "The end magic is how a reader confirms it has the whole file.",
        b,
    ));

    let mut b = good.clone();
    let t = b.len() - TRAILER_SIZE;
    b[t + 16] ^= 0xff; // superblock digest in the trailer
    fix_trailer_crc(&mut b);
    out.push(case(
        "invalid/framing",
        "trailer-superblock-digest-wrong",
        Expect::Reject,
        Some("R-C09"),
        "The trailer's digest is what makes the superblock trustworthy after one seek.",
        b,
    ));

    // R-C10 the two superblocks must agree.
    let supers: Vec<_> = segs
        .iter()
        .filter(|(_, k, _)| *k == omni_core::seg::SUPER)
        .collect();
    if supers.len() == 2 {
        let (front_hdr, _, front_len) = *supers[0];
        let mut b = good.clone();
        b[front_hdr + SEG_HEADER_SIZE + 3] ^= 0x01;
        fix_segment_crcs(&mut b, front_hdr, front_len as usize);
        out.push(case(
            "invalid/framing",
            "superblocks-disagree",
            Expect::Reject,
            Some("R-C10"),
            "Two superblocks exist for redundancy; if they differ, neither can be trusted.",
            b,
        ));
    }

    // R-C12 an index entry pointing outside the file.
    let mut b = good.clone();
    let (idx_hdr, _, _) = *segs
        .iter()
        .find(|(_, k, _)| *k == omni_core::seg::INDEX)
        .unwrap();
    let e0 = idx_hdr + SEG_HEADER_SIZE + 64; // first entry
    b[e0 + 32..e0 + 40].copy_from_slice(&u64::MAX.to_le_bytes());
    let ilen = u64::from_le_bytes(b[idx_hdr + 8..idx_hdr + 16].try_into().unwrap()) as usize;
    fix_segment_crcs(&mut b, idx_hdr, ilen);
    out.push(case(
        "invalid/framing",
        "index-offset-out-of-range",
        Expect::Reject,
        Some("R-C12"),
        "An offset past the end of the file must be caught by bounds checking, not by faulting.",
        b,
    ));

    // Truncation at each segment boundary. A reader must fail cleanly at every
    // one of them rather than at only the convenient ones.
    for (i, &(hdr, kind, _)) in segs.iter().enumerate() {
        let name: &'static str = Box::leak(
            format!(
                "truncated-before-{}-{i}",
                omni_core::seg::name(kind).to_lowercase()
            )
            .into_boxed_str(),
        );
        // Cutting before the first segment leaves nothing but a header, which
        // a reader should diagnose as "too small to be a container" rather
        // than as a missing trailer.
        let (rule, why) = if hdr <= HEADER_SIZE {
            (
                "R-C01",
                "A file with a header and nothing else is too small to be a container.",
            )
        } else {
            (
                "R-C09",
                "A truncated container must be refused cleanly, not read past its end.",
            )
        };
        out.push(case(
            "invalid/framing",
            name,
            Expect::Reject,
            Some(rule),
            why,
            good[..hdr].to_vec(),
        ));
    }

    // A file too short to hold a header at all.
    out.push(case(
        "invalid/framing",
        "shorter-than-header",
        Expect::Reject,
        Some("R-C01"),
        "The smallest legal container is larger than this; length must be checked first.",
        good[..HEADER_SIZE / 2].to_vec(),
    ));

    // Nothing but the magic.
    out.push(case(
        "invalid/framing",
        "magic-only",
        Expect::Reject,
        Some("R-C01"),
        "Recognising the magic is not the same as having a file.",
        MAGIC.to_vec(),
    ));

    let _ = obj_len;
    out
}

// ------------------------------------------------------- invalid/encoding --

/// Builds a container whose root object payload is `payload` verbatim, without
/// re-encoding it, so non-canonical bytes survive into the file.
fn container_with_raw_root(payload: Vec<u8>) -> Vec<u8> {
    let hash = HashAlgo::Blake3_256;
    let obj = Object {
        otype: otype::MANIFEST,
        payload,
        oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
        stored: None,
    };
    let root = obj.digest(hash);
    pack(
        &[obj],
        &root,
        &PackOptions {
            hash,
            creator: "omni-conformance".into(),
            ..Default::default()
        },
    )
    .unwrap()
}

fn invalid_encoding() -> Vec<Case> {
    // Each payload is a hand-built CBOR map that decodes to something
    // reasonable but violates one canonicalisation rule from §03.2. The digest
    // is computed over these exact bytes, so the object *verifies* — which is
    // the point: integrity and canonicality are separate properties, and a
    // reader that checks only the first accepts files no two writers would
    // ever agree on.
    let cases: &[(&str, &str, &str, Vec<u8>)] = &[
        (
            "indefinite-length-map",
            "D2",
            "Indefinite-length items have two encodings for one value, so digests would not be stable.",
            {
                let mut v = vec![0xbf]; // map(*)
                v.extend_from_slice(&[0x61, b't']);
                v.extend_from_slice(&[0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v.extend_from_slice(&[0x61, b'v', 0x01]);
                v.push(0xff); // break
                v
            },
        ),
        (
            "non-shortest-integer",
            "D1",
            "Encoding 1 as a two-byte integer is a second spelling of the same value.",
            {
                let mut v = vec![0xa2];
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v.extend_from_slice(&[0x61, b'v', 0x18, 0x01]); // uint8 1, not 0x01
                v
            },
        ),
        (
            "map-keys-unsorted",
            "D3",
            "Key order is part of the encoding, so an unsorted map hashes differently from its canonical form.",
            {
                let mut v = vec![0xa2];
                v.extend_from_slice(&[0x61, b'v', 0x01]); // "v" before "t"
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v
            },
        ),
        (
            "duplicate-map-keys",
            "D4",
            "A duplicated key has no defined meaning and lets two readers disagree about one file.",
            {
                let mut v = vec![0xa3];
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v.extend_from_slice(&[0x61, b'v', 0x01]);
                v.extend_from_slice(&[0x61, b'v', 0x02]); // again
                v
            },
        ),
        (
            "invalid-utf8",
            "R-E03",
            "Text that is not UTF-8 cannot be compared, normalised or displayed safely.",
            {
                let mut v = vec![0xa2];
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v.extend_from_slice(&[0x61, b'v', 0x62, 0xff, 0xfe]); // 2-byte text, invalid
                v
            },
        ),
        (
            "non-shortest-float",
            "D5",
            "A double that fits in a half must be encoded as a half, or one value has three spellings.",
            {
                let mut v = vec![0xa2];
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                // "v": 1.0 as f64 rather than the shortest f16
                v.extend_from_slice(&[0x61, b'v', 0xfb]);
                v.extend_from_slice(&1.0f64.to_bits().to_be_bytes());
                v
            },
        ),
        (
            "trailing-bytes",
            "D8",
            "Bytes after the value are unreachable data inside an object's own extent.",
            {
                let mut v = vec![0xa2];
                v.extend_from_slice(&[0x61, b't', 0x6a]);
                v.extend_from_slice(b"omni.x/bad");
                v.extend_from_slice(&[0x61, b'v', 0x01]);
                v.push(0x00); // one byte too many
                v
            },
        ),
    ];

    cases
        .iter()
        .map(|(name, rule, why, payload)| {
            Case {
                category: "invalid/encoding",
                name,
                expect: Expect::Reject,
                // Rule IDs are `&'static str` in the spec; these are too.
                rule: Some(rule),
                why,
                bytes: container_with_raw_root(payload.clone()),
            }
        })
        .collect()
}

// ------------------------------------------------------------- forward/ ----

/// Files using things this version does not define. §11's criticality bits and
/// §14's versioning exist so these degrade rather than fail, and an
/// implementation that rejects them outright is not conformant — it is the
/// behaviour that makes a format unable to evolve.
fn forward() -> Vec<Case> {
    let mut out = Vec::new();
    let hash = HashAlgo::Blake3_256;

    // An object type from the future, referenced but not critical.
    let future = Object {
        otype: 0x8001, // plugin/extension space
        payload: Value::map(vec![
            ("t", Value::text("com.example/thing-from-2071")),
            ("v", Value::U(3)),
            ("wavelength", Value::U(42)),
        ])
        .encode(),
        oflags: oflags::SAFE_TO_COPY, // not CRITICAL
        stored: None,
    };
    let fd = future.digest(hash);
    let manifest = Object::structure(
        otype::MANIFEST,
        &Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("model")),
            ("created", Value::U(0)),
            (
                "ext",
                Value::map(vec![(
                    "com.example/thing",
                    Value::Array(vec![Value::U(0x8001), Value::Bytes(fd.to_vec())]),
                )]),
            ),
        ]),
    );
    let root = manifest.digest(hash);
    out.push(case(
        "forward",
        "unknown-otype-non-critical",
        Expect::Degrade,
        Some("R-P01"),
        "An unknown, non-critical object must be copied and ignored, not fatal.",
        pack(
            &[manifest, future],
            &root,
            &PackOptions {
                hash,
                creator: "omni-conformance".into(),
                ..Default::default()
            },
        )
        .unwrap(),
    ));

    // A future minor version. Minor versions are additive by §14.
    let (mut b, _, _) = base(hash, 12);
    b[10..12].copy_from_slice(&99u16.to_le_bytes());
    fix_header_crc(&mut b);
    out.push(case(
        "forward",
        "future-container-minor",
        Expect::Degrade,
        Some("R-V01"),
        "Minor versions are additive; a reader must not refuse a file for having a higher one.",
        b,
    ));

    // An unknown segment kind. Segments are self-framing precisely so that
    // unknown ones can be stepped over.
    let (good, _, _) = base(hash, 12);
    let mut b = good.clone();
    let segs = segments(&good);
    let (pad_hdr, _, pad_len) = *segs
        .iter()
        .find(|(_, k, _)| *k == omni_core::seg::PAD)
        .expect("the base container has padding");
    b[pad_hdr + 4..pad_hdr + 6].copy_from_slice(&0x7f00u16.to_le_bytes());
    fix_segment_crcs(&mut b, pad_hdr, pad_len as usize);
    out.push(case(
        "forward",
        "unknown-segment-kind",
        Expect::Degrade,
        Some("R-C05"),
        "Segment framing carries its own length so unknown kinds can be skipped.",
        b,
    ));

    // A required feature nobody has heard of. This one is *not* degradable in
    // the sense of executing the model, but the file must still open, be
    // listed, be copied and be verified — hence exit 3, not 1.
    let manifest = Object::structure(
        otype::MANIFEST,
        &Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("model")),
            ("created", Value::U(0)),
            (
                "features",
                Value::map(vec![
                    (
                        "required",
                        Value::Array(vec![Value::text("com.example/hyperquant.7")]),
                    ),
                    ("optional", Value::Array(vec![])),
                ]),
            ),
        ]),
    );
    let root = manifest.digest(hash);
    out.push(case(
        "forward",
        "unknown-required-feature",
        Expect::Degrade,
        Some("R-V02"),
        "An unsupported required feature blocks execution, not inspection: indeterminate, not invalid.",
        pack(
            &[manifest],
            &root,
            &PackOptions {
                hash,
                creator: "omni-conformance".into(),
                ..Default::default()
            },
        )
        .unwrap(),
    ));

    out
}

/// Sanity-checks the corpus against this implementation's own reader, so a
/// generator bug cannot ship a case that does not test what it claims.
pub fn self_check(cases: &[Case]) -> Vec<String> {
    let mut problems = Vec::new();
    for c in cases {
        let opened = Container::open(c.bytes.clone());
        let verdict = match &opened {
            Err(_) => Expect::Reject,
            Ok(container) => match omni_core::verify(container) {
                Err(_) => Expect::Reject,
                // A report can condemn a file without the call failing:
                // R-C07 and R-C08 are reported as flags, and the CLI exits 1
                // on them. The self-check has to agree with the CLI, or it
                // would bless cases no implementation actually passes.
                Ok(r) if !r.padding_ok || !r.alignment_ok => Expect::Reject,
                Ok(_) => Expect::Accept,
            },
        };
        let ok = match c.expect {
            Expect::Accept => verdict == Expect::Accept,
            Expect::Reject => verdict == Expect::Reject,
            // Degrade cases must at least open; whether this implementation
            // then refuses to *execute* is beyond what it implements.
            Expect::Degrade => opened.is_ok(),
        };
        if !ok {
            problems.push(format!(
                "{}/{}: expected {}, this reader said {}",
                c.category,
                c.name,
                c.expect.name(),
                verdict.name()
            ));
        }
    }
    problems
}
