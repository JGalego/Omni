//! Model-level helpers: a minimal builder that produces the object graph of
//! §00.4, and the small slice of the tensor/dtype algebra needed to describe
//! and size a `literal` tensor.
//!
//! This is deliberately not a complete implementation of §04–§08. It is enough
//! to build real containers, prove the binary format, and generate the worked
//! examples in `examples/`.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};

// ------------------------------------------------------------------ dtypes --

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    /// (total bits, exponent bits, mantissa bits)
    Float {
        w: u16,
        e: u16,
        m: u16,
    },
    Int {
        w: u16,
        signed: bool,
    },
    Bool,
}

impl DType {
    pub const F32: DType = DType::Float { w: 32, e: 8, m: 23 };
    pub const F16: DType = DType::Float { w: 16, e: 5, m: 10 };
    pub const BF16: DType = DType::Float { w: 16, e: 8, m: 7 };
    pub const F8E4M3: DType = DType::Float { w: 8, e: 4, m: 3 };
    pub const I8: DType = DType::Int { w: 8, signed: true };
    pub const U4: DType = DType::Int {
        w: 4,
        signed: false,
    };

    /// Bits per element. Fractional widths (e.g. base-3 ternary packing) would
    /// return a rational here; the registered types in this subset are all
    /// integral.
    pub fn bits(&self) -> u32 {
        match self {
            DType::Float { w, .. } => *w as u32,
            DType::Int { w, .. } => *w as u32,
            DType::Bool => 1,
        }
    }

    /// Bytes required for `n` densely packed elements (§04.3.5 clause 2).
    pub fn packed_bytes(&self, n: u64) -> u64 {
        let bits = self.bits() as u64 * n;
        bits.div_ceil(8)
    }

    pub fn alias(&self) -> Option<&'static str> {
        Some(match *self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8E4M3 => "f8e4m3",
            DType::I8 => "i8",
            DType::U4 => "u4",
            DType::Float {
                w: 64,
                e: 11,
                m: 52,
            } => "f64",
            DType::Bool => "bool",
            _ => return None,
        })
    }

    /// The structural descriptor of §04.3. Writers emit the alias *and* the
    /// expansion, so a reader that has never heard of the alias is unaffected.
    pub fn to_value(&self) -> Value {
        let mut pairs: Vec<(&str, Value)> = Vec::new();
        if let Some(a) = self.alias() {
            pairs.push(("alias", Value::text(a)));
        }
        match *self {
            DType::Float { w, e, m } => {
                pairs.push(("k", Value::text("float")));
                pairs.push(("w", Value::U(w as u64)));
                pairs.push(("e", Value::U(e as u64)));
                pairs.push(("m", Value::U(m as u64)));
            }
            DType::Int { w, signed } => {
                pairs.push(("k", Value::text("int")));
                pairs.push(("w", Value::U(w as u64)));
                pairs.push(("signed", Value::Bool(signed)));
            }
            DType::Bool => {
                pairs.push(("k", Value::text("bool")));
                pairs.push(("w", Value::U(1)));
            }
        }
        Value::map(pairs)
    }
}

// ------------------------------------------------------------------ builder --

pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub axes: Option<Vec<String>>,
    pub semantic: &'static str,
    pub data: Vec<u8>,
}

pub struct ModelBuilder {
    pub name: String,
    pub license_spdx: Option<String>,
    pub arch_family: Option<String>,
    pub arch_params: Vec<(String, Value)>,
    pub tensors: Vec<TensorSpec>,
    pub chunk_size: usize,
    pub extra: Vec<(String, Value)>,
    /// The digest algorithm the resulting container will use. Object
    /// identities depend on it, so it has to be fixed before the graph is
    /// built rather than at pack time.
    pub hash: HashAlgo,
}

impl ModelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        ModelBuilder {
            name: name.into(),
            license_spdx: None,
            arch_family: None,
            arch_params: Vec::new(),
            tensors: Vec::new(),
            chunk_size: 4 << 20,
            extra: Vec::new(),
            hash: HashAlgo::default(),
        }
    }

    pub fn hash(mut self, algo: HashAlgo) -> Self {
        self.hash = algo;
        self
    }

    pub fn license(mut self, spdx: impl Into<String>) -> Self {
        self.license_spdx = Some(spdx.into());
        self
    }

    pub fn arch(mut self, family: impl Into<String>, params: Vec<(&str, Value)>) -> Self {
        self.arch_family = Some(family.into());
        self.arch_params = params
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        self
    }

    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    pub fn tensor(mut self, t: TensorSpec) -> Self {
        self.tensors.push(t);
        self
    }

    /// Builds the object graph. Returns every object plus the root manifest's
    /// digest.
    pub fn build(&self) -> (Vec<Object>, Digest) {
        let mut objects: Vec<Object> = Vec::new();
        let mut table_entries: Vec<(Value, Value)> = Vec::new();
        let mut order: Vec<Value> = Vec::new();
        let mut params_total: u64 = 0;

        for t in &self.tensors {
            let numel: u64 = t.shape.iter().product();
            params_total += numel;
            let expected = t.dtype.packed_bytes(numel);
            assert_eq!(
                expected,
                t.data.len() as u64,
                "R-T02: tensor `{}` declares {} bytes but carries {}",
                t.name,
                expected,
                t.data.len()
            );

            // Chunk the payload (§03.6 `fixed`).
            let mut chunk_refs = Vec::new();
            for chunk in t.data.chunks(self.chunk_size) {
                let blob = Object::blob(chunk.to_vec());
                let d = blob.digest(self.hash);
                objects.push(blob);
                chunk_refs.push(Value::map(vec![
                    (
                        "r",
                        Value::Array(vec![Value::U(0), Value::Bytes(d.to_vec())]),
                    ),
                    ("n", Value::U(chunk.len() as u64)),
                ]));
            }

            let chunklist = Object::structure(
                otype::CHUNK_LIST,
                &Value::map(vec![
                    ("t", Value::text("omni.tensor/chunklist")),
                    ("v", Value::U(1)),
                    ("total", Value::U(t.data.len() as u64)),
                    (
                        "chunker",
                        Value::map(vec![
                            ("k", Value::text("fixed")),
                            ("size", Value::U(self.chunk_size as u64)),
                        ]),
                    ),
                    ("chunks", Value::Array(chunk_refs)),
                ]),
            );
            let cl_digest = chunklist.digest(self.hash);
            objects.push(chunklist);

            let mut desc_pairs: Vec<(&str, Value)> = vec![
                ("t", Value::text("omni.tensor/desc")),
                ("v", Value::U(1)),
                (
                    "shape",
                    Value::Array(t.shape.iter().map(|d| Value::U(*d)).collect()),
                ),
                ("dtype", t.dtype.to_value()),
                (
                    "layout",
                    Value::map(vec![
                        ("k", Value::text("strided")),
                        ("order", Value::text("row-major")),
                    ]),
                ),
                ("semantic", Value::text(t.semantic)),
                (
                    "value",
                    Value::map(vec![
                        ("op", Value::text("literal")),
                        (
                            "chunks",
                            Value::Array(vec![
                                Value::U(otype::CHUNK_LIST as u64),
                                Value::Bytes(cl_digest.to_vec()),
                            ]),
                        ),
                    ]),
                ),
                ("materialize", Value::text("lazy")),
            ];
            if let Some(axes) = &t.axes {
                desc_pairs.push((
                    "axes",
                    Value::Array(axes.iter().map(|a| Value::text(a.clone())).collect()),
                ));
            }
            let desc = Object::structure(otype::TENSOR_DESC, &Value::map(desc_pairs));
            let d_digest = desc.digest(self.hash);
            objects.push(desc);

            table_entries.push((
                Value::text(t.name.clone()),
                Value::Array(vec![
                    Value::U(otype::TENSOR_DESC as u64),
                    Value::Bytes(d_digest.to_vec()),
                ]),
            ));
            order.push(Value::text(t.name.clone()));
        }

        let table = Object::structure(
            otype::TENSOR_TABLE,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/table")),
                ("v", Value::U(1)),
                ("tensors", Value::Map(table_entries)),
                ("order", Value::Array(order)),
            ]),
        );
        let table_digest = table.digest(self.hash);
        objects.push(table);

        let model = Object::structure(
            otype::MODEL,
            &Value::map(vec![
                ("t", Value::text("omni.core/model")),
                ("v", Value::U(1)),
                (
                    "tensors",
                    Value::Array(vec![
                        Value::U(otype::TENSOR_TABLE as u64),
                        Value::Bytes(table_digest.to_vec()),
                    ]),
                ),
            ]),
        );
        let model_digest = model.digest(self.hash);
        objects.push(model);

        // Metadata. Note what is *absent*: no license unless one was supplied,
        // no fabricated defaults (importer rule I1).
        let mut meta_pairs: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.meta/model")),
            ("v", Value::U(1)),
            ("name", Value::text(self.name.clone())),
            ("params_total", Value::U(params_total)),
        ];
        if let Some(fam) = &self.arch_family {
            let mut arch = vec![
                ("family", Value::text(fam.clone())),
                (
                    "dialects",
                    Value::Array(vec![Value::map(vec![
                        ("ns", Value::text("omni.nn")),
                        ("v", Value::U(1)),
                    ])]),
                ),
            ];
            if !self.arch_params.is_empty() {
                arch.push((
                    "params",
                    Value::Map(
                        self.arch_params
                            .iter()
                            .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                            .collect(),
                    ),
                ));
            }
            meta_pairs.push(("arch", Value::map(arch)));
        }
        if let Some(l) = &self.license_spdx {
            meta_pairs.push((
                "license",
                Value::map(vec![("spdx", Value::text(l.clone()))]),
            ));
        }
        for (k, v) in &self.extra {
            meta_pairs.push((Box::leak(k.clone().into_boxed_str()), v.clone()));
        }
        let meta = Object::structure(otype::METADATA, &Value::map(meta_pairs));
        let meta_digest = meta.digest(self.hash);
        objects.push(meta);

        let manifest = Object::structure(
            otype::MANIFEST,
            &Value::map(vec![
                ("t", Value::text("omni.core/manifest")),
                ("v", Value::U(1)),
                ("kind", Value::text("model")),
                ("created", Value::U(0)),
                (
                    "meta",
                    Value::Array(vec![
                        Value::U(otype::METADATA as u64),
                        Value::Bytes(meta_digest.to_vec()),
                    ]),
                ),
                (
                    "assets",
                    Value::map(vec![(
                        "model",
                        Value::Array(vec![
                            Value::U(otype::MODEL as u64),
                            Value::Bytes(model_digest.to_vec()),
                        ]),
                    )]),
                ),
                ("entry", Value::text("model")),
                (
                    "features",
                    Value::map(vec![
                        (
                            "required",
                            Value::Array(vec![
                                Value::text("omni.core/1.0"),
                                Value::text("omni.tensor/expr.1"),
                            ]),
                        ),
                        ("optional", Value::Array(vec![])),
                    ]),
                ),
            ]),
        );
        let root = manifest.digest(self.hash);
        objects.push(manifest);

        (objects, root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{pack, verify, Container, PackOptions};

    #[test]
    fn dtype_sizing() {
        assert_eq!(DType::BF16.packed_bytes(10), 20);
        assert_eq!(DType::U4.packed_bytes(10), 5);
        assert_eq!(DType::U4.packed_bytes(9), 5); // rounds up
        assert_eq!(DType::Bool.packed_bytes(9), 2);
        assert_eq!(DType::F8E4M3.packed_bytes(1000), 1000);
    }

    #[test]
    fn dtype_descriptor_carries_alias_and_expansion() {
        let v = DType::BF16.to_value();
        assert_eq!(v.get("alias").and_then(|x| x.as_str()), Some("bf16"));
        assert_eq!(v.get("w").and_then(|x| x.as_u64()), Some(16));
        assert_eq!(v.get("e").and_then(|x| x.as_u64()), Some(8));
        assert_eq!(v.get("m").and_then(|x| x.as_u64()), Some(7));
    }

    #[test]
    fn end_to_end_build_pack_verify() {
        let shape = vec![32u64, 16];
        let numel: u64 = shape.iter().product();
        let data = vec![0x3cu8; DType::BF16.packed_bytes(numel) as usize];

        let (objs, root) = ModelBuilder::new("test/tiny")
            .license("Apache-2.0")
            .arch("transformer.decoder", vec![("n_layers", Value::U(1))])
            .chunk_size(512)
            .tensor(TensorSpec {
                name: "model.layers.0.attn.q_proj.weight".into(),
                shape,
                dtype: DType::BF16,
                axes: Some(vec!["out_features".into(), "in_features".into()]),
                semantic: "weight",
                data,
            })
            .build();

        let bytes = pack(&objs, &root, &PackOptions::default()).unwrap();
        let c = Container::open(bytes).unwrap();
        let r = verify(&c).unwrap();
        assert!(r.dangling.is_empty());
        assert!(r.padding_ok && r.alignment_ok);

        // Metadata is reachable without touching any tensor payload.
        let manifest = c.root().unwrap();
        let meta_ref = manifest.get("meta").unwrap().as_array().unwrap();
        let mut d = [0u8; 32];
        d.copy_from_slice(meta_ref[1].as_bytes().unwrap());
        let meta = c.get_value(&d).unwrap();
        assert_eq!(meta.get("params_total").and_then(|v| v.as_u64()), Some(512));
        assert_eq!(meta.get("name").and_then(|v| v.as_str()), Some("test/tiny"));
    }
}
