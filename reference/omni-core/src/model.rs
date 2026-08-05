//! Model-level helpers: a minimal builder that produces the object graph of
//! §00.4 for a model whose tensors are bare `literal`s.
//!
//! The dtype algebra it uses lives in [`crate::dtype`]; expressions over those
//! tensors live in [`crate::expr`]. This module is the convenience layer that
//! turns "here are some named byte buffers" into a valid object graph, and it
//! generates the worked examples in `examples/`.

use crate::cbor::Value;
use crate::container::{otype, Digest, HashAlgo, Object};
use crate::dtype::DType;
use crate::expr::{Expr, Ref};
use crate::layout::Layout;
use crate::tensor::TensorDesc;

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
    /// Tensors whose value is an arbitrary expression rather than a bare
    /// literal: a dequantized weight, a LoRA-merged one, a cast realization.
    /// These cost no storage of their own (§04.1).
    pub derived: Vec<(String, TensorDesc)>,
    /// Objects the caller stored directly — the chunks and chunk lists behind
    /// the literals inside `derived` expressions.
    pub extra_objects: Vec<Object>,
    /// Extra manifest assets beyond `model`, as (slot name, object type,
    /// value): a tokenizer, a chat template. Each becomes an object plus an
    /// `assets` entry, which is how a reader finds it without a full walk
    /// (§03.4).
    pub assets: Vec<(String, u16, Value)>,
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
            derived: Vec::new(),
            extra_objects: Vec::new(),
            assets: Vec::new(),
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

    /// Stores `data` as chunks plus a `ChunkList` and returns the `literal`
    /// expression that reads it back.
    ///
    /// This is the writer half of §04.5: the caller gets a value it can build
    /// expressions over, and the bytes exist exactly once no matter how many
    /// expressions mention them.
    pub fn literal(&mut self, data: &[u8], dtype: DType, shape: &[u64], layout: Layout) -> Expr {
        let cl = self.chunk_list(data);
        Expr::Literal {
            chunks: cl,
            dtype,
            shape: crate::expr::dims(shape),
            layout,
        }
    }

    /// Stores bytes as a `ChunkList` and returns the ref.
    pub fn chunk_list(&mut self, data: &[u8]) -> Ref {
        let mut chunk_refs = Vec::new();
        for chunk in data.chunks(self.chunk_size) {
            let blob = Object::blob(chunk.to_vec());
            let d = blob.digest(self.hash);
            self.extra_objects.push(blob);
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
                ("total", Value::U(data.len() as u64)),
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
        let d = chunklist.digest(self.hash);
        self.extra_objects.push(chunklist);
        (otype::CHUNK_LIST, d)
    }

    /// Adds a manifest asset: an object reachable from `assets` under `slot`.
    pub fn asset(mut self, slot: impl Into<String>, otype: u16, value: Value) -> Self {
        self.assets.push((slot.into(), otype, value));
        self
    }

    /// Adds a tensor whose value is an expression. The descriptor is stored as
    /// given, so the caller decides what the tensor claims — and
    /// [`TensorDesc::check`] is what verifies the claim.
    pub fn derived(mut self, name: impl Into<String>, desc: TensorDesc) -> Self {
        self.derived.push((name.into(), desc));
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

        // Derived tensors: descriptors only, no new payload. A tensor whose
        // value is a bare literal does carry parameters and is counted; one
        // whose value is an expression is a *view* of parameters that already
        // exist, and counting it would report the same weights twice (R-M01).
        // Quantization scales and zero points are not parameters either, and are
        // recognised the same way a reader recognises them: by use.
        let mut machinery = std::collections::BTreeSet::new();
        for (_, desc) in &self.derived {
            crate::tensor::scheme_leaves(&desc.value, &mut machinery);
        }
        for (name, desc) in &self.derived {
            if let Expr::Literal { chunks, .. } = &desc.value {
                if desc.is_weight() && !machinery.contains(chunks) {
                    params_total += desc.numel().unwrap_or(0);
                }
            }
            let obj = Object::structure(otype::TENSOR_DESC, &desc.to_value());
            let d = obj.digest(self.hash);
            objects.push(obj);
            table_entries.push((
                Value::text(name.clone()),
                Value::Array(vec![
                    Value::U(otype::TENSOR_DESC as u64),
                    Value::Bytes(d.to_vec()),
                ]),
            ));
            order.push(Value::text(name.clone()));
        }
        objects.extend(self.extra_objects.iter().cloned());

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

        // Assets: the model, plus whatever else the caller attached.
        let mut asset_entries: Vec<(Value, Value)> = vec![(
            Value::text("model"),
            Value::Array(vec![
                Value::U(otype::MODEL as u64),
                Value::Bytes(model_digest.to_vec()),
            ]),
        )];
        for (slot, ot, value) in &self.assets {
            let obj = Object::structure(*ot, value);
            let d = obj.digest(self.hash);
            objects.push(obj);
            asset_entries.push((
                Value::text(slot.clone()),
                Value::Array(vec![Value::U(*ot as u64), Value::Bytes(d.to_vec())]),
            ));
        }

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
                ("assets", Value::Map(asset_entries)),
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
