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
    /// Extra arguments the runner passes to the implementation for this case.
    ///
    /// Empty for every structural case, because a file that is malformed is
    /// malformed at any depth. The `numeric/` suite is the exception: its
    /// cases are structurally perfect and wrong only in their *values*, so
    /// they ask for the validation level that evaluates. An implementation
    /// whose default already evaluates may ignore the argument.
    pub args: &'static [&'static str],
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
        // The `numeric/` suite cannot be judged from the framing, so it is run
        // at the level that evaluates. Everything else is judged from the bytes.
        args: if category == "numeric" {
            &["--level", "6"]
        } else {
            &[]
        },
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
    out.extend(numeric());
    out.extend(valid_features());
    out
}

// ---------------------------------------------------------- valid/features --

/// `valid/features` — containers that are valid *and* use something optional.
///
/// The `degrade` suite tests what a reader does with a feature from the future.
/// This one tests the opposite failure: a reader that refuses a feature the
/// format defines today, because its author only ever tried the minimal case.
/// Every file here must be accepted outright — not degraded — and each uses one
/// mechanism the minimal corpus never exercises.
fn valid_features() -> Vec<Case> {
    let mut out = Vec::new();
    let hash = HashAlgo::Blake3_256;
    let opts = |codec| PackOptions {
        hash,
        log2_align: 12,
        creator: "omni-conformance".into(),
        reproducible: true,
        codec,
    };

    // 1. A compressed container. §03.7 makes `zstd` the one codec a reader MUST
    //    support, and compression is a property of the stored copy: every
    //    object digest here is the digest of the *logical* bytes, so a reader
    //    that verified the compressed bytes instead fails.
    let (objs, root) = base_objects(hash, 256);
    out.push(case(
        "valid/features",
        "codec-zstd",
        Expect::Accept,
        Some("R-C05"),
        "segments compressed with the codec §03.7 marks MUST; identities are \
         over the logical bytes, so decompression is required to check them",
        pack(
            &objs,
            &root,
            &opts(omni_core::codec::Codec::Zstd { level: 3 }),
        )
        .unwrap(),
    ));

    // 2. A tensor split across several chunks. A 40 GB tensor is never one
    //    object, and a reader that assumes a literal is one blob works
    //    perfectly until the first real model.
    let (objs, root) = ModelBuilder::new("omni/conformance/features")
        .hash(hash)
        .chunk_size(64)
        .tensor(TensorSpec {
            name: "w".into(),
            shape: vec![64, 4],
            dtype: DType::F32,
            axes: None,
            semantic: "weight",
            data: (0..64 * 4 * 4).map(|i| (i % 251) as u8).collect(),
            layout: None,
        })
        .build();
    out.push(case(
        "valid/features",
        "chunked-tensor",
        Expect::Accept,
        Some("R-T02"),
        "one tensor over sixteen chunks: the chunk list's total must equal the \
         sum of its parts and the tensor's stored size",
        pack(&objs, &root, &opts(omni_core::codec::Codec::Raw)).unwrap(),
    ));

    // 3. A quantized weight: the expression form of §05, with the packed words
    //    stored verbatim and the arithmetic in a `dequantize` node. A reader
    //    that only understands `literal` must still open, list and verify this
    //    file — it is the shape of every quantized model in the wild.
    let mut b = ModelBuilder::new("omni/conformance/features")
        .hash(hash)
        .chunk_size(256);
    // 32 int4 values, two per byte, with an f16 scale per block of 8.
    let packed: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();
    let q = b.literal(
        &packed,
        DType::U4,
        &[4, 8],
        omni_core::layout::Layout::Packed {
            elems_per_word: 2,
            word_bits: 8,
            bit_order: omni_core::layout::BitOrder::LsbFirst,
            order: omni_core::layout::Order::RowMajor,
        },
    );
    let mut scales = vec![0u8; 8];
    for (i, v) in [1.0f64, 0.5, 0.25, 2.0].iter().enumerate() {
        DType::F16.encode(&mut scales, i as u64, *v, omni_core::dtype::Round::Rne);
    }
    let s = b.literal(
        &scales,
        DType::F16,
        &[4, 1],
        omni_core::layout::Layout::default(),
    );
    let value = omni_core::expr::Expr::Dequantize {
        x: Box::new(q),
        scheme: Value::map(vec![
            ("scheme", Value::text("affine")),
            ("formula", Value::text("affine-sub")),
            ("out", DType::F32.to_value()),
            ("axis", Value::U(1)),
            ("block", Value::Array(vec![Value::U(1), Value::U(8)])),
            ("scale", s.to_value()),
            (
                "zero",
                omni_core::expr::Expr::Full {
                    value: omni_core::expr::Scalar::Int(8),
                    dtype: DType::U8,
                    shape: omni_core::expr::dims(&[1, 1]),
                }
                .to_value(),
            ),
        ]),
    };
    let (objs, root) = b
        .derived(
            "w",
            omni_core::tensor::TensorDesc {
                shape: omni_core::expr::dims(&[4, 8]),
                dtype: DType::F32,
                layout: omni_core::layout::Layout::default(),
                value,
                semantic: Some("weight".into()),
                role: Some("quantized".into()),
                axes: None,
                device_hint: None,
                materialize: omni_core::tensor::Materialize::Lazy,
                stats: None,
                digest_materialized: None,
            },
        )
        .build();
    out.push(case(
        "valid/features",
        "quantized-expression",
        Expect::Accept,
        Some("R-T04"),
        "a weight that is an expression rather than a buffer (§05.1): int4 \
         packed two per byte, one f16 scale per block of eight, and the \
         zero-point subtracted before scaling",
        pack(&objs, &root, &opts(omni_core::codec::Codec::Raw)).unwrap(),
    ));

    // 4. A model that carries its own graph (§07.5). The tensor names in it are
    //    checked against the tensor table by R-I10, so this file is also a test
    //    that a reader can tell a graph that matches its weights from one that
    //    does not.
    let sizes = [4u64, 3, 2];
    let mut b = ModelBuilder::new("omni/conformance/features")
        .hash(hash)
        .chunk_size(256)
        .arch(
            "mlp",
            vec![(
                "hidden_sizes",
                Value::Array(sizes.iter().map(|n| Value::U(*n)).collect()),
            )],
        );
    let mut names = Vec::new();
    for i in 0..sizes.len() - 1 {
        let (fan_in, fan_out) = (sizes[i], sizes[i + 1]);
        let name = format!("mlp.layers.{i}.weight");
        b = b.tensor(TensorSpec {
            name: name.clone(),
            shape: vec![fan_out, fan_in],
            dtype: DType::F32,
            axes: None,
            semantic: "weight",
            data: (0..fan_in * fan_out * 4).map(|k| (k % 251) as u8).collect(),
            layout: None,
        });
        names.push(name);
    }
    let params = Value::map(vec![(
        "hidden_sizes",
        Value::Array(sizes.iter().map(|n| Value::U(*n)).collect()),
    )]);
    let module = omni_core::ir::synthesize("mlp", &params, &names).expect("synthesizes");
    let (objs, root) = b.graph(module, Vec::new()).build();
    out.push(case(
        "valid/features",
        "graph-semantic",
        Expect::Accept,
        Some("R-I10"),
        "a model that describes its own computation (§07): every `constant` in \
         the graph names a tensor the table has, at the shape the graph expects",
        pack(&objs, &root, &opts(omni_core::codec::Codec::Raw)).unwrap(),
    ));
    out
}

/// The minimal graph's objects, for cases that only vary the packing.
fn base_objects(hash: HashAlgo, chunk: usize) -> (Vec<Object>, Digest) {
    ModelBuilder::new("omni/conformance")
        .hash(hash)
        .chunk_size(chunk)
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
        .build()
}

// ---------------------------------------------------------------- numeric --

/// `numeric/` — cases whose point is arithmetic rather than framing.
///
/// Every other suite can be judged from the bytes alone: a file is well-formed
/// or it is not. These cannot. A reader that unpacks an `int4` in the wrong
/// nibble order, rounds `bf16` away from even, or reads an MX block's scale as
/// a float instead of an exponent produces a *valid container* full of wrong
/// numbers, and no amount of structural checking notices.
///
/// So each case carries the publisher's own digest of what its tensors
/// evaluate to (§04.3's `digest_materialized`), over subtrees that §04.7.6
/// makes normative — no reductions, no plugins, nothing whose order is a
/// choice. An implementation that computes something else fails its own file,
/// which is the only way a corpus can test arithmetic through an exit code.
fn numeric() -> Vec<Case> {
    let mut out = Vec::new();
    let hash = HashAlgo::Blake3_256;

    // (name, dtype, values, why) — the values are chosen so that a wrong
    // reading is a *different* number rather than a slightly different one.
    let cases: Vec<(&'static str, DType, Vec<f64>, &'static str)> = vec![
        (
            "f32-f16-bf16",
            DType::BF16,
            vec![1.0, -2.5, 3.4028235e38, 1.0e-8, 0.0, -0.0],
            "bf16 keeps f32's exponent range and eight bits of its mantissa; a \
             reader that rounds through f16 loses the large value to infinity \
             and the small one to zero",
        ),
        (
            "round-half-to-even",
            DType::F16,
            // Exactly halfway between two f16 values, in both directions.
            vec![2049.0, 2051.0, 2053.0, -2049.0],
            "§04.3 says round-nearest-ties-to-even, and every one of these is a \
             tie: rounding half away from zero gives four different numbers",
        ),
        (
            "f16-subnormals",
            DType::F16,
            vec![6.0e-8, 5.96e-8, 1.0e-7, -6.0e-8],
            "the subnormal range below 2⁻¹⁴, where flush-to-zero is a common \
             shortcut and a detectable one",
        ),
        (
            "f8e4m3-saturation",
            DType::F8E4M3,
            vec![448.0, 500.0, -448.0, 0.001953125],
            "f8e4m3 has no infinity: §04.3 saturates to ±448 rather than \
             producing a NaN, and the two readings differ on every overflow",
        ),
        (
            "int4-packing",
            DType::I4,
            (0..16).map(|i| (i as f64) - 8.0).collect(),
            "two signed 4-bit values per byte, low nibble first (§04.4): a \
             reader that swaps them reads the tensor transposed within every \
             byte",
        ),
        (
            "e8m0-exponents",
            DType::E8M0,
            vec![1.0, 2.0, 0.5, 256.0, 0.00390625],
            "the MX scale type is a bare power-of-two exponent (§05.2.8), not a \
             float: read as one, every scale is wrong by orders of magnitude",
        ),
    ];

    for (name, dtype, values, why) in cases {
        let n = values.len() as u64;
        let mut data = vec![0u8; dtype.packed_bytes(n) as usize];
        for (i, v) in values.iter().enumerate() {
            dtype.encode(&mut data, i as u64, *v, omni_core::dtype::Round::Rne);
        }
        // The digest is of the *encoded* bytes, which is what a reader that
        // evaluated the expression and re-encoded it in the declared dtype
        // must arrive at.
        let want = hash.digest(&data);
        let (objs, _) = ModelBuilder::new("omni/conformance/numeric")
            .hash(hash)
            .chunk_size(256)
            .tensor(TensorSpec {
                name: "values".into(),
                shape: vec![n],
                dtype: dtype.clone(),
                axes: None,
                semantic: "",
                data: data.clone(),
                layout: None,
            })
            .build();
        // The declared digest goes onto the descriptor after the fact: the
        // builder writes tensors, and this is the publisher's claim about what
        // reading one produces.
        let (objs, root) = with_materialized_digest(objs, hash, want);
        let bytes = pack(
            &objs,
            &root,
            &PackOptions {
                hash,
                log2_align: 12,
                creator: "omni-conformance".into(),
                reproducible: true,
                ..Default::default()
            },
        )
        .unwrap();
        out.push(case(
            "numeric",
            name,
            Expect::Accept,
            Some("R-T08"),
            why,
            bytes,
        ));
    }

    // And the negative: the same mechanism, with a digest that does not match
    // what the tensor evaluates to. A reader that ignores the field passes this
    // file and should not.
    let dtype = DType::F32;
    let data: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let (objs, _) = ModelBuilder::new("omni/conformance/numeric")
        .hash(hash)
        .chunk_size(256)
        .tensor(TensorSpec {
            name: "values".into(),
            shape: vec![16],
            dtype,
            axes: None,
            semantic: "",
            data,
            layout: None,
        })
        .build();
    let (objs, root) = with_materialized_digest(objs, hash, [0x5au8; 32]);
    let bytes = pack(
        &objs,
        &root,
        &PackOptions {
            hash,
            log2_align: 12,
            creator: "omni-conformance".into(),
            reproducible: true,
            ..Default::default()
        },
    )
    .unwrap();
    out.push(case(
        "numeric",
        "materialized-digest-wrong",
        Expect::Reject,
        Some("R-T08"),
        "the tensor declares a `digest_materialized` its own values do not \
         produce. The container is structurally perfect, so this is the one \
         case in the corpus that can only be caught by *evaluating*",
        bytes,
    ));
    out
}

/// Rewrites every `TensorDesc` in a just-built graph to declare `digest`, and
/// fixes up every object that pointed at one.
///
/// The builder does not take the field because a publisher computes it *after*
/// deciding what the tensor is, which is the order this does it in — and
/// changing a content-addressed object means every ref to it moves, so the
/// table, the model and the manifest are rewritten in turn until nothing
/// changes. Doing it by digest substitution rather than by rebuilding each
/// object type is what keeps this honest: nothing here needs to know which
/// fields hold refs.
fn with_materialized_digest(
    objs: Vec<Object>,
    hash: HashAlgo,
    digest: Digest,
) -> (Vec<Object>, Digest) {
    let mut map: Vec<(Digest, Digest)> = Vec::new();
    let mut out: Vec<Object> = objs
        .into_iter()
        .map(|o| {
            if o.otype != otype::TENSOR_DESC {
                return o;
            }
            let before = o.digest(hash);
            let mut v = omni_core::cbor::decode(&o.payload).expect("a just-built descriptor");
            if let Value::Map(pairs) = &mut v {
                pairs.retain(|(k, _)| k.as_str() != Some("digest_materialized"));
                pairs.push((
                    Value::text("digest_materialized"),
                    Value::Bytes(digest.to_vec()),
                ));
                pairs.sort_by(|a, b| a.0.encode().cmp(&b.0.encode()));
            }
            let after = Object::structure(otype::TENSOR_DESC, &v);
            map.push((before, after.digest(hash)));
            after
        })
        .collect();

    // Propagate: every object that named a moved digest moves too. The graph is
    // four deep (desc → table → model → manifest), and the loop runs until a
    // pass changes nothing rather than a fixed number of times.
    while !map.is_empty() {
        let mut next_map: Vec<(Digest, Digest)> = Vec::new();
        for o in out.iter_mut() {
            if o.otype == otype::BLOB {
                continue;
            }
            let Ok(mut v) = omni_core::cbor::decode(&o.payload) else {
                continue;
            };
            if substitute_refs(&mut v, &map) {
                let before = o.digest(hash);
                *o = Object::structure(o.otype, &v);
                next_map.push((before, o.digest(hash)));
            }
        }
        map = next_map;
    }
    let root = out
        .iter()
        .find(|o| o.otype == otype::MANIFEST)
        .expect("a manifest")
        .digest(hash);
    (out, root)
}

/// Replaces any 32-byte digest in a value with its replacement.
fn substitute_refs(v: &mut Value, map: &[(Digest, Digest)]) -> bool {
    match v {
        Value::Bytes(b) if b.len() == 32 => {
            for (from, to) in map {
                if b.as_slice() == from.as_slice() {
                    *b = to.to_vec();
                    return true;
                }
            }
            false
        }
        Value::Array(xs) => {
            let mut any = false;
            for x in xs {
                any |= substitute_refs(x, map);
            }
            any
        }
        Value::Map(pairs) => {
            let mut any = false;
            for (_, x) in pairs {
                any |= substitute_refs(x, map);
            }
            any
        }
        Value::Tag(_, inner) => substitute_refs(inner, map),
        _ => false,
    }
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
                args: &[],
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
/// Whether every tensor that declares `digest_materialized` produces it.
///
/// Deterministic subtrees only, which is §04.7.6's own condition: an unpinned
/// reduction order may legitimately differ, and treating that as invalid would
/// make the field unusable on exactly the models it matters for.
fn materialized_ok(c: &Container) -> bool {
    use omni_core::tensor::{Materialized, TensorDesc, TensorTable};
    struct Borrowed<'a>(&'a Container);
    impl omni_core::store::Store for Borrowed<'_> {
        fn hash(&self) -> HashAlgo {
            self.0.header.hash
        }
        fn resolve(&self, d: &Digest) -> Result<Option<Vec<u8>>, omni_core::store::Error> {
            match self.0.read(d) {
                Ok(b) => Ok(Some(b)),
                Err(omni_core::container::Error::NotFound(_)) => Ok(None),
                Err(e) => Err(omni_core::store::Error::Corrupt(e.to_string())),
            }
        }
    }
    let store = Borrowed(c);
    let ctx = omni_core::expr::Ctx::new(&store);
    let Ok(manifest) = c.root() else { return true };
    let Some(model) = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(|r| omni_core::expr::parse_ref_value(r).ok())
    else {
        return true;
    };
    let Ok(mv) = ctx.value(&model.1) else {
        return true;
    };
    let Some(tref) = mv
        .get("tensors")
        .and_then(|r| omni_core::expr::parse_ref_value(r).ok())
    else {
        return true;
    };
    let Ok(table) = TensorTable::load(&ctx, &tref) else {
        return true;
    };
    for r in table.tensors.values() {
        let Ok(desc) = TensorDesc::load(&ctx, r) else {
            continue;
        };
        if matches!(
            desc.check_materialized(&ctx, c.header.hash),
            Materialized::Mismatch { .. }
        ) {
            return false;
        }
    }
    true
}

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
                // The `numeric/` suite is the one that cannot be judged from
                // the framing, so the self-check evaluates too: a tensor that
                // does not produce the digest it declares makes the file
                // invalid (§04.3), and a reader that skips the field would
                // otherwise bless a container full of wrong numbers.
                Ok(_) if !materialized_ok(container) => Expect::Reject,
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
