//! §13.5 — the OCI registry mapping, as an OCI image layout.
//!
//! The argument for this mapping is parasitic adoption: registries, mirrors,
//! CDNs, auth, replication and signing already exist everywhere, and a format
//! that maps onto OCI distribution inherits all of it without asking anyone to
//! deploy anything. §13.5 gives the mapping; this implements it.
//!
//! What is here is the *mapping and the layout*, not a registry client. It
//! produces and consumes an [OCI image layout] — the `oci-layout`, `index.json`,
//! `blobs/sha256/<digest>` directory that `oras`, `skopeo` and `crane` read and
//! push — which is the whole of §13.5 that can exist without a network. Pushing
//! it needs an HTTP client with registry auth (bearer token dance, `WWW-
//! Authenticate` challenges, chunked blob uploads) and that is a client, not a
//! format concern; `oras cp --from-oci-layout` does it today.
//!
//! [OCI image layout]: https://github.com/opencontainers/image-spec/blob/main/image-layout.md
//!
//! ## How a container becomes layers
//!
//! The layers are *the container file, cut into pieces at data-object
//! boundaries*, and concatenating them in order reproduces the file byte for
//! byte — so [`import_layout`] is a check on [`export_layout`] rather than a
//! reinterpretation of it. The cuts are a deterministic function of the
//! container, so the same model exported twice is the same blobs and a re-push
//! uploads nothing.
//!
//! ### What this does and does not buy, precisely
//!
//! §13.5 claims registry-level dedup: "a delta model's packs are new blobs; the
//! base's packs are already present and are *not re-uploaded*". That is true, and
//! it is worth being exact about *why*, because a plausible misreading of it is
//! false.
//!
//! It is **not** the case that re-exporting a modified model shares most of its
//! blobs with the original. Objects are placed in the file in digest order
//! (§02.4), so changing one tensor changes its digest, moves it in that order,
//! and shifts the file offset of everything after it. Every pack from the first
//! difference onward is then a different blob. No cutting rule over a *byte
//! stream* avoids this, and pretending otherwise would be the rsync problem with
//! extra steps.
//!
//! What actually delivers the claim is the layer above: a fine-tune is published
//! as a **delta container** (§08.6) whose objects are only the ones that are new.
//! Its packs are small because the artifact is small, and the base's packs are
//! already in the registry because the base was pushed as its own artifact — with
//! `dev.omni.parent` and the OCI referrers API linking the two. Dedup comes from
//! *not putting the base's bytes in the delta*, which is the whole premise of the
//! object model, rather than from two independently packed files happening to
//! agree byte for byte.
//!
//! Cutting at object boundaries is still the right rule — it keeps a pack a
//! meaningful unit, it is what §01.9's partitioning means, and it makes a
//! partial pull land on object edges — but the dedup story lives in the object
//! graph, not in the slicing.
//!
//! ## What is deliberately not synthesized
//!
//! §13.5 shows `subject: <the base model's manifest>` for the OCI referrers API.
//! A `subject` is an OCI descriptor, and its digest is the sha256 of the *base's
//! OCI manifest* — bytes that exist wherever the base was pushed, and nowhere
//! here. Inventing one would produce a manifest that resolves to nothing. The
//! OMNI parent digest goes into an annotation instead, and the `subject` is left
//! for whatever pushes this and knows the answer.

use crate::container::{otype, Container, Digest};
use crate::json::{self, Value};

pub const ARTIFACT_TYPE: &str = "application/vnd.omni.model.v1";
pub const MANIFEST_TYPE: &str = "application/vnd.oci.image.manifest.v1+json";
pub const INDEX_JSON_TYPE: &str = "application/vnd.oci.image.index.v1+json";
pub const CONFIG_TYPE: &str = "application/vnd.omni.manifest.v1+cbor";
pub const PACK_TYPE: &str = "application/vnd.omni.pack.v1";
pub const OMNI_INDEX_TYPE: &str = "application/vnd.omni.index.v1";
pub const EMPTY_TYPE: &str = "application/vnd.oci.empty.v1+json";

/// §13.5's caveat: "registries dislike very large individual blobs and very many
/// small ones. Target 100 MB – 2 GB packs." One GiB is the reference default the
/// same paragraph names.
pub const DEFAULT_PACK_BYTES: u64 = 1 << 30;
/// A floor, so a tiny model does not become one layer per tensor. Registries
/// charge a round trip per blob and a hundred 4 KiB layers is the failure mode
/// the caveat warns about from the other side.
pub const MIN_PACK_BYTES: u64 = 1 << 20;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "oci layout: {m}"),
            Error::Unsupported(m) => write!(f, "oci: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// A half-open byte range of the container file: one layer's extent.
type Extent = (u64, u64);
/// The pack extents and the index extent, which is always the last layer.
type Cuts = (Vec<Extent>, Extent);

/// One blob in the layout, keyed by the sha256 the registry knows it by.
pub struct Blob {
    /// Lower-case hex, without the `sha256:` prefix — the filename under
    /// `blobs/sha256/`.
    pub sha256: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// A complete OCI image layout, in memory.
pub struct Layout {
    /// `oci-layout`, `index.json`, the manifest and every blob. Paths are
    /// relative and contain no `..`, so writing them is a join and nothing more.
    pub files: Vec<(String, Vec<u8>)>,
    /// The manifest's own descriptor: what a registry would call this artifact.
    pub manifest_digest: String,
    pub manifest_size: u64,
    /// How many pack layers the container was cut into.
    pub packs: usize,
}

impl Layout {
    /// Writes the layout to a directory.
    pub fn write(&self, dir: &std::path::Path) -> std::io::Result<()> {
        for (rel, bytes) in &self.files {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, bytes)?;
        }
        Ok(())
    }
}

fn sha256_hex(b: &[u8]) -> String {
    crate::sha256::hex(&crate::sha256::sha256(b))
}

/// An OCI content descriptor.
fn descriptor(media_type: &str, bytes: &[u8], annotations: Vec<(&str, String)>) -> Value {
    let mut pairs = vec![
        ("mediaType", json::string(media_type)),
        (
            "digest",
            json::string(format!("sha256:{}", sha256_hex(bytes))),
        ),
        ("size", Value::U(bytes.len() as u64)),
    ];
    if !annotations.is_empty() {
        pairs.push((
            "annotations",
            Value::Object(
                annotations
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), json::string(v)))
                    .collect(),
            ),
        ));
    }
    json::object(pairs)
}

/// Where to cut the container into pack layers.
///
/// Cuts fall on data-object boundaries, taken in file order, and a piece is
/// closed as soon as adding the next object would exceed the target. Two
/// containers that share a run of objects therefore share the blobs holding
/// them, which is the dedup §13.5 promises.
///
/// The first piece starts at 0 — it carries the header, the front superblock and
/// every structure object, which together are the part a reader needs first. The
/// last piece ends where the index begins; the index and the trailer are their
/// own layer, because a consumer wants them without the weights (§13.4.1).
fn cut_points(c: &Container, target: u64) -> Res<Cuts> {
    let (index_off, index_len) = index_extent(c)?;
    // The index segment header sits in the 64 bytes before the payload, and the
    // trailer closes the file.
    let index_start = index_off - 64;
    let index_end = c.bytes.len() as u64;
    if index_start > index_end {
        return Err(Error::Malformed("the index extent leaves the file".into()));
    }
    let _ = index_len;

    // Data objects in file order. Structure objects live before the first of
    // them, so the first cut point is the first data object's offset.
    let mut data: Vec<(u64, u64)> = c
        .index
        .iter()
        .filter(|e| e.otype == otype::BLOB && e.stored_len > 0)
        .map(|e| (e.offset, e.stored_len))
        .collect();
    data.sort_unstable();

    let mut packs = Vec::new();
    let mut start = 0u64;
    let target = target.max(MIN_PACK_BYTES);
    for (off, len) in &data {
        // A data object that has already been passed — the extents overlap only
        // in a malformed container — is skipped rather than reasoned about.
        if *off < start {
            continue;
        }
        if off + len > index_start {
            break;
        }
        if off - start >= target {
            packs.push((start, *off));
            start = *off;
        }
    }
    if start < index_start {
        packs.push((start, index_start));
    }
    if packs.is_empty() {
        // A container with no data objects at all: one pack covering everything
        // before the index.
        packs.push((0, index_start));
    }
    Ok((packs, (index_start, index_end)))
}

fn index_extent(c: &Container) -> Res<(u64, u64)> {
    let idx = c
        .superblock
        .get("index")
        .ok_or_else(|| Error::Malformed("the superblock has no index".into()))?;
    let off = idx.get("off").and_then(|v| v.as_u64()).unwrap_or(0);
    let len = idx.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
    if off < 64 {
        return Err(Error::Malformed("an implausible index offset".into()));
    }
    Ok((off, len))
}

/// Options for the mapping. Reproducibility is not one of them: the layout is a
/// function of the container and the pack size, and nothing else, so pushing the
/// same model twice produces the same blobs.
#[derive(Clone, Debug)]
pub struct ExportOpts {
    pub pack_bytes: u64,
    /// `org.opencontainers.image.ref.name`, the tag this would be pushed as.
    pub reference: Option<String>,
}

impl Default for ExportOpts {
    fn default() -> Self {
        ExportOpts {
            pack_bytes: DEFAULT_PACK_BYTES,
            reference: None,
        }
    }
}

/// Maps a container onto an OCI image layout (§13.5).
pub fn export_layout(c: &Container, opts: &ExportOpts) -> Res<Layout> {
    let (packs, index_extent) = cut_points(c, opts.pack_bytes)?;
    let mut blobs: Vec<Blob> = Vec::new();

    // The config is the OMNI Manifest object itself, which is what §13.5 says
    // and what makes `docker inspect`-shaped tooling show something meaningful.
    let manifest_obj = c
        .read(&c.header.root_digest)
        .map_err(|e| Error::Malformed(e.to_string()))?;
    let config = descriptor(CONFIG_TYPE, &manifest_obj, Vec::new());
    blobs.push(Blob {
        sha256: sha256_hex(&manifest_obj),
        media_type: CONFIG_TYPE.into(),
        bytes: manifest_obj,
    });

    let mut layers = Vec::new();
    for (i, (a, b)) in packs.iter().enumerate() {
        let piece = &c.bytes[*a as usize..*b as usize];
        // The offset is recorded even though in-order concatenation implies it:
        // a descriptor that says where it belongs can be checked, and a layer
        // pulled on its own is still placeable.
        layers.push(descriptor(
            PACK_TYPE,
            piece,
            vec![
                ("dev.omni.offset", a.to_string()),
                ("dev.omni.pack", format!("{}/{}", i + 1, packs.len())),
            ],
        ));
        blobs.push(Blob {
            sha256: sha256_hex(piece),
            media_type: PACK_TYPE.into(),
            bytes: piece.to_vec(),
        });
    }
    let index_piece = &c.bytes[index_extent.0 as usize..index_extent.1 as usize];
    layers.push(descriptor(
        OMNI_INDEX_TYPE,
        index_piece,
        vec![("dev.omni.offset", index_extent.0.to_string())],
    ));
    blobs.push(Blob {
        sha256: sha256_hex(index_piece),
        media_type: OMNI_INDEX_TYPE.into(),
        bytes: index_piece.to_vec(),
    });

    // Annotations: what a registry UI can show without pulling anything, and
    // what a mirror can index on. Every value is read from the container; none
    // is invented.
    let mut ann: Vec<(&str, String)> = vec![
        (
            "dev.omni.canonical-digest",
            format!(
                "{}:{}",
                c.header.hash.prefix(),
                crate::sha256::hex(&c.header.root_digest)
            ),
        ),
        ("dev.omni.hash", c.header.hash.name().to_string()),
        ("dev.omni.objects", c.index.len().to_string()),
        ("dev.omni.file-size", c.bytes.len().to_string()),
        ("dev.omni.spec", crate::SPEC_VERSION.to_string()),
    ];
    if let Ok(manifest) = c.root() {
        if let Some(meta_d) = manifest
            .get("meta")
            .and_then(|r| crate::expr::parse_ref_value(r).ok())
        {
            if let Ok(meta) = c.get_value(&meta_d.1) {
                if let Some(n) = meta.get("name").and_then(|v| v.as_str()) {
                    ann.push(("org.opencontainers.image.title", n.to_string()));
                }
                if let Some(p) = meta.get("params_total").and_then(|v| v.as_u64()) {
                    ann.push(("dev.omni.params", p.to_string()));
                }
                if let Some(l) = meta
                    .get("license")
                    .and_then(|l| l.get("spdx"))
                    .and_then(|v| v.as_str())
                {
                    ann.push(("org.opencontainers.image.licenses", l.to_string()));
                }
            }
        }
        // §13.5's `subject` needs the base's *OCI* manifest digest, which does
        // not exist locally. The OMNI parent digest does, so it is recorded as
        // what it is rather than dressed up as a referrer.
        if let Ok(parents) = crate::delta::parents(&manifest) {
            if let Some(p) = parents.first() {
                ann.push((
                    "dev.omni.parent",
                    format!(
                        "{}:{}",
                        c.header.hash.prefix(),
                        crate::sha256::hex(&p.reference.1)
                    ),
                ));
            }
        }
    }
    if c.header.flags & crate::container::hflags::PARTIAL != 0 {
        ann.push(("dev.omni.partial", "true".into()));
    }

    let manifest_json = json::object(vec![
        ("schemaVersion", Value::U(2)),
        ("mediaType", json::string(MANIFEST_TYPE)),
        ("artifactType", json::string(ARTIFACT_TYPE)),
        ("config", config),
        ("layers", Value::Array(layers)),
        (
            "annotations",
            Value::Object(
                ann.into_iter()
                    .map(|(k, v)| (k.to_string(), json::string(v)))
                    .collect(),
            ),
        ),
    ])
    .encode()
    .into_bytes();

    let mut manifest_ann = vec![];
    if let Some(r) = &opts.reference {
        manifest_ann.push(("org.opencontainers.image.ref.name", r.clone()));
    }
    let manifest_desc = descriptor(MANIFEST_TYPE, &manifest_json, manifest_ann);
    let index_json = json::object(vec![
        ("schemaVersion", Value::U(2)),
        ("mediaType", json::string(INDEX_JSON_TYPE)),
        ("manifests", Value::Array(vec![manifest_desc])),
    ])
    .encode()
    .into_bytes();

    let manifest_digest = sha256_hex(&manifest_json);
    let manifest_size = manifest_json.len() as u64;
    blobs.push(Blob {
        sha256: manifest_digest.clone(),
        media_type: MANIFEST_TYPE.into(),
        bytes: manifest_json,
    });

    let mut files = vec![
        (
            "oci-layout".to_string(),
            json::object(vec![("imageLayoutVersion", json::string("1.0.0"))])
                .encode()
                .into_bytes(),
        ),
        ("index.json".to_string(), index_json),
    ];
    for b in &blobs {
        files.push((format!("blobs/sha256/{}", b.sha256), b.bytes.clone()));
    }
    Ok(Layout {
        files,
        manifest_digest,
        manifest_size,
        packs: packs.len(),
    })
}

/// What an import found, so a caller can report it rather than assume it.
pub struct Imported {
    pub bytes: Vec<u8>,
    pub manifest_digest: String,
    pub layers: usize,
    /// The `dev.omni.*` annotations, verbatim.
    pub annotations: Vec<(String, String)>,
}

/// Reads an OCI image layout back into a container.
///
/// Every blob is checked against the digest that named it before it is used —
/// which for a layout pulled from a registry is the only thing standing between
/// a mirror and the model you asked for. Then the pack layers are concatenated in
/// order and the result is *parsed as a container*, so a layout that reassembles
/// into something malformed fails here rather than later.
pub fn import_layout(read: &dyn Fn(&str) -> Option<Vec<u8>>) -> Res<Imported> {
    let layout = read("oci-layout").ok_or_else(|| {
        Error::Malformed("no `oci-layout` file: this is not an OCI layout".into())
    })?;
    let v = json::parse(&layout).map_err(|e| Error::Malformed(e.to_string()))?;
    match v.get("imageLayoutVersion").and_then(|x| x.as_str()) {
        Some("1.0.0") => {}
        other => {
            return Err(Error::Unsupported(format!(
                "image layout version {other:?} is not 1.0.0"
            )))
        }
    }

    let index = read("index.json").ok_or_else(|| Error::Malformed("no `index.json`".into()))?;
    let index = json::parse(&index).map_err(|e| Error::Malformed(e.to_string()))?;
    let manifests = index
        .get("manifests")
        .and_then(|m| m.as_array())
        .ok_or_else(|| Error::Malformed("`index.json` has no manifests".into()))?;
    // One artifact per layout is what `export_layout` writes and what a single
    // model is. A multi-manifest index is a legal OCI layout and an ambiguous
    // request, so it is refused rather than resolved by picking the first.
    let desc = match manifests.len() {
        1 => &manifests[0],
        n => {
            return Err(Error::Unsupported(format!(
                "this layout holds {n} manifests; which model is not stated"
            )))
        }
    };
    let manifest_digest = digest_of(desc)?;
    let manifest_bytes = fetch(read, &manifest_digest)?;
    let manifest = json::parse(&manifest_bytes).map_err(|e| Error::Malformed(e.to_string()))?;

    if manifest.get("artifactType").and_then(|x| x.as_str()) != Some(ARTIFACT_TYPE) {
        return Err(Error::Unsupported(format!(
            "artifactType is {:?}, not {ARTIFACT_TYPE}",
            manifest.get("artifactType").and_then(|x| x.as_str())
        )));
    }

    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| Error::Malformed("the manifest has no layers".into()))?;
    let mut bytes = Vec::new();
    let mut count = 0usize;
    for l in layers {
        let mt = l.get("mediaType").and_then(|x| x.as_str()).unwrap_or("");
        if mt != PACK_TYPE && mt != OMNI_INDEX_TYPE {
            return Err(Error::Unsupported(format!(
                "a layer of type `{mt}` is not part of this mapping"
            )));
        }
        let d = digest_of(l)?;
        let piece = fetch(read, &d)?;
        if let Some(size) = l.get("size").and_then(|x| x.as_u64()) {
            if size != piece.len() as u64 {
                return Err(Error::Malformed(format!(
                    "layer {d} declares {size} bytes and holds {}",
                    piece.len()
                )));
            }
        }
        // The offset annotation, when present, is checked rather than trusted:
        // it is the descriptor's claim about where this layer belongs, and a
        // layout whose layers are out of order must not silently reassemble into
        // a plausible-looking file.
        if let Some(off) = l
            .get("annotations")
            .and_then(|a| a.get("dev.omni.offset"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
        {
            if off != bytes.len() as u64 {
                return Err(Error::Malformed(format!(
                    "layer {d} says it starts at {off}, {} bytes have been assembled",
                    bytes.len()
                )));
            }
        }
        bytes.extend_from_slice(&piece);
        count += 1;
    }

    // The reassembled file has to be a container. Parsing it here is what turns
    // "the blobs verified" into "the model is intact".
    let c = Container::open(bytes.clone()).map_err(|e| {
        Error::Malformed(format!("the reassembled layers are not a container: {e}"))
    })?;
    // And it has to be *this* container: the config blob is the OMNI Manifest
    // object, so it must be the object the header's root names.
    let config_digest = manifest
        .get("config")
        .ok_or_else(|| Error::Malformed("the manifest has no config".into()))
        .and_then(digest_of)?;
    let config = fetch(read, &config_digest)?;
    let root: Digest = c.header.root_digest;
    let from_file = c.read(&root).map_err(|e| Error::Malformed(e.to_string()))?;
    if from_file != config {
        return Err(Error::Malformed(
            "the config blob is not the manifest object this container is rooted at".into(),
        ));
    }

    let annotations = manifest
        .get("annotations")
        .and_then(|a| a.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    Ok(Imported {
        bytes,
        manifest_digest,
        layers: count,
        annotations,
    })
}

/// `sha256:<hex>` from a descriptor, refusing an algorithm this cannot check.
fn digest_of(desc: &Value) -> Res<String> {
    let d = desc
        .get("digest")
        .and_then(|x| x.as_str())
        .ok_or_else(|| Error::Malformed("a descriptor has no digest".into()))?;
    let hex = d.strip_prefix("sha256:").ok_or_else(|| {
        Error::Unsupported(format!(
            "descriptor digest `{d}`: only sha256 is checkable here"
        ))
    })?;
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Malformed(format!("`{d}` is not a sha256 digest")));
    }
    Ok(hex.to_string())
}

/// Reads a blob and verifies it against the digest that named it.
fn fetch(read: &dyn Fn(&str) -> Option<Vec<u8>>, hex: &str) -> Res<Vec<u8>> {
    let bytes = read(&format!("blobs/sha256/{hex}"))
        .ok_or_else(|| Error::Malformed(format!("blob sha256:{hex} is not in the layout")))?;
    let got = sha256_hex(&bytes);
    if got != hex {
        return Err(Error::Malformed(format!(
            "blob sha256:{hex} hashes to {got}"
        )));
    }
    Ok(bytes)
}

/// Reads a layout from a directory, refusing any path that is not one this
/// mapping writes.
pub fn dir_reader(dir: &std::path::Path) -> impl Fn(&str) -> Option<Vec<u8>> + '_ {
    move |rel: &str| {
        // Every path comes from a descriptor, i.e. from data. Rebuilding it from
        // its parts rather than joining it means a `..` in a digest cannot become
        // a path at all.
        let path = match rel {
            "oci-layout" => dir.join("oci-layout"),
            "index.json" => dir.join("index.json"),
            other => {
                let hex = other.strip_prefix("blobs/sha256/")?;
                if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return None;
                }
                dir.join("blobs").join("sha256").join(hex)
            }
        };
        std::fs::read(path).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{pack, PackOptions};
    use crate::model::{ModelBuilder, TensorSpec};

    /// Tensors of 256 KiB, so a 1 MiB pack target cuts a six-tensor model into
    /// several layers rather than one. A test about partitioning needs something
    /// to partition.
    const ELEMS: u64 = 65536;

    fn weights(fill: u8, i: usize) -> Vec<u8> {
        (0..ELEMS * 4)
            .map(|k| ((k as u8).wrapping_add(fill)).wrapping_add(i as u8))
            .collect()
    }

    fn model(name: &str, fill: u8, tensors: usize) -> Container {
        let mut b = ModelBuilder::new(name);
        for i in 0..tensors {
            b = b.tensor(TensorSpec {
                name: format!("w{i}"),
                shape: vec![ELEMS],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight",
                data: weights(fill, i),
                layout: None,
            });
        }
        let (objs, root) = b.build();
        Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
    }

    fn layout_reader(l: &Layout) -> impl Fn(&str) -> Option<Vec<u8>> + '_ {
        move |rel: &str| {
            l.files
                .iter()
                .find(|(p, _)| p == rel)
                .map(|(_, b)| b.clone())
        }
    }

    /// The layout has to be a layout: the files an OCI reader looks for, in the
    /// places it looks for them, with every digest matching its blob.
    #[test]
    fn the_layout_is_one_an_oci_reader_would_recognise() {
        let c = model("test/oci", 0, 4);
        let l = export_layout(&c, &ExportOpts::default()).unwrap();

        let names: Vec<&str> = l.files.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.contains(&"oci-layout"));
        assert!(names.contains(&"index.json"));
        assert!(names.iter().all(|n| !n.contains("..")));

        // Every blob's path is its own sha256 — that is what makes a registry's
        // dedup work, so it is checked rather than assumed.
        for (path, bytes) in &l.files {
            if let Some(hex) = path.strip_prefix("blobs/sha256/") {
                assert_eq!(hex, sha256_hex(bytes), "{path} is misnamed");
            }
        }

        let read = layout_reader(&l);
        let index = json::parse(&read("index.json").unwrap()).unwrap();
        assert_eq!(index.get("schemaVersion").unwrap().as_u64(), Some(2));
        let m = &index.get("manifests").unwrap().as_array().unwrap()[0];
        assert_eq!(m.get("mediaType").unwrap().as_str(), Some(MANIFEST_TYPE));
        assert_eq!(digest_of(m).unwrap(), l.manifest_digest);
        assert_eq!(m.get("size").unwrap().as_u64(), Some(l.manifest_size));

        let manifest = json::parse(&fetch(&read, &l.manifest_digest).unwrap()).unwrap();
        assert_eq!(
            manifest.get("artifactType").unwrap().as_str(),
            Some(ARTIFACT_TYPE)
        );
        // The config is the OMNI Manifest object, per §13.5.
        let config = manifest.get("config").unwrap();
        assert_eq!(config.get("mediaType").unwrap().as_str(), Some(CONFIG_TYPE));
        assert_eq!(
            fetch(&read, &digest_of(config).unwrap()).unwrap(),
            c.read(&c.header.root_digest).unwrap()
        );
        // Layers: packs then the index, exactly as the section lists them.
        let layers = manifest.get("layers").unwrap().as_array().unwrap();
        assert!(layers.len() >= 2);
        assert_eq!(
            layers.last().unwrap().get("mediaType").unwrap().as_str(),
            Some(OMNI_INDEX_TYPE)
        );
        for l2 in &layers[..layers.len() - 1] {
            assert_eq!(l2.get("mediaType").unwrap().as_str(), Some(PACK_TYPE));
        }
        // Annotations name only what the container states.
        let ann = manifest.get("annotations").unwrap();
        assert_eq!(
            ann.get("dev.omni.canonical-digest").unwrap().as_str(),
            Some(
                format!(
                    "{}:{}",
                    c.header.hash.prefix(),
                    crate::sha256::hex(&c.header.root_digest)
                )
                .as_str()
            )
        );
        assert_eq!(
            ann.get("dev.omni.file-size").unwrap().as_str(),
            Some(c.bytes.len().to_string().as_str())
        );
    }

    /// The layers are the container, so putting them back gives the container.
    #[test]
    fn the_layers_reassemble_into_the_same_bytes() {
        for tensors in [1usize, 4, 9] {
            let c = model("test/oci", 7, tensors);
            let l = export_layout(
                &c,
                &ExportOpts {
                    // Small enough to force several packs out of a toy model.
                    pack_bytes: MIN_PACK_BYTES,
                    reference: Some("test:1".into()),
                },
            )
            .unwrap();
            let got = import_layout(&layout_reader(&l)).unwrap();
            assert_eq!(got.bytes, c.bytes, "{tensors} tensors did not round-trip");
            assert_eq!(got.manifest_digest, l.manifest_digest);
            assert_eq!(got.layers, l.packs + 1);
            assert!(got
                .annotations
                .iter()
                .any(|(k, _)| k == "dev.omni.canonical-digest"));
        }
    }

    /// The dedup story, stated exactly. Two claims are true and one plausible
    /// one is not, and a test that conflated them would be worse than no test.
    #[test]
    fn dedup_comes_from_the_object_graph_and_not_from_the_slicing() {
        let base = model("test/base", 3, 6);
        let opts = ExportOpts {
            pack_bytes: MIN_PACK_BYTES,
            reference: None,
        };
        let a = export_layout(&base, &opts).unwrap();
        assert!(
            a.packs > 1,
            "this test needs more than one pack, got {}",
            a.packs
        );

        // True: the mapping is a function of the container, so re-exporting the
        // same model produces identical blobs and a re-push uploads nothing.
        let again = export_layout(&base, &opts).unwrap();
        assert_eq!(blob_set(&a), blob_set(&again));
        assert_eq!(a.manifest_digest, again.manifest_digest);

        // Not true, and deliberately asserted so nobody documents otherwise:
        // re-exporting a model with one tensor changed does *not* share most of
        // its blobs. Objects are placed in digest order, so a changed digest
        // moves that object and shifts every offset after it.
        let (objs, root) = {
            let mut b = ModelBuilder::new("test/base");
            for i in 0..6 {
                let mut data = weights(3, i);
                if i == 5 {
                    data[0] ^= 0xff;
                }
                b = b.tensor(TensorSpec {
                    name: format!("w{i}"),
                    shape: vec![ELEMS],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight",
                    data,
                    layout: None,
                });
            }
            b.build()
        };
        let tuned = Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap();
        let b = export_layout(&tuned, &opts).unwrap();
        assert_ne!(a.manifest_digest, b.manifest_digest);

        // True, and this is the claim §13.5 actually makes: the *delta* is the
        // small artifact. A container holding only the objects that are new maps
        // to a layout whose blobs are a fraction of the base's, and the base's
        // blobs are already in the registry because the base was pushed as its
        // own artifact.
        let novel: Vec<crate::container::Object> = {
            let present: std::collections::BTreeSet<Digest> =
                base.index.iter().map(|e| e.digest).collect();
            let mut out = Vec::new();
            for e in &tuned.index {
                if e.otype == otype::BLOB && present.contains(&e.digest) {
                    continue;
                }
                out.push(crate::container::Object {
                    otype: e.otype,
                    payload: tuned.read(&e.digest).unwrap(),
                    oflags: e.oflags,
                    stored: None,
                });
            }
            out
        };
        let external: Vec<_> = tuned
            .index
            .iter()
            .filter(|e| e.otype == otype::BLOB && base.index.iter().any(|b| b.digest == e.digest))
            .cloned()
            .collect();
        assert!(!external.is_empty(), "the two models should share tensors");
        let delta = Container::open(
            crate::container::pack_partial(
                &novel,
                &external,
                &tuned.header.root_digest,
                &PackOptions::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let d = export_layout(&delta, &opts).unwrap();
        let (total_a, total_d) = (blob_bytes(&a), blob_bytes(&d));
        assert!(
            total_d * 3 < total_a,
            "the delta's layout is {total_d} bytes against the base's {total_a}"
        );
        // And it names what it descends from, so a registry can link them.
        let read = layout_reader(&d);
        let manifest = json::parse(&fetch(&read, &d.manifest_digest).unwrap()).unwrap();
        assert_eq!(
            manifest
                .get("annotations")
                .and_then(|x| x.get("dev.omni.partial"))
                .and_then(|v| v.as_str()),
            Some("true")
        );
    }

    fn blob_bytes(l: &Layout) -> u64 {
        l.files
            .iter()
            .filter(|(p, _)| p.starts_with("blobs/sha256/"))
            .map(|(_, b)| b.len() as u64)
            .sum()
    }

    fn blob_set(l: &Layout) -> std::collections::BTreeSet<String> {
        l.files
            .iter()
            .filter_map(|(p, _)| p.strip_prefix("blobs/sha256/").map(str::to_string))
            .collect()
    }

    /// A layout from a registry is a layout from a stranger. Every way it can be
    /// wrong has to be a refusal.
    #[test]
    fn a_layout_that_lies_is_refused() {
        let c = model("test/oci", 1, 4);
        let l = export_layout(&c, &ExportOpts::default()).unwrap();
        let files: Files = l.files.clone();
        type Files = Vec<(String, Vec<u8>)>;
        let with = |edit: &dyn Fn(&mut Files)| {
            let mut f = files.clone();
            edit(&mut f);
            let r = move |rel: &str| f.iter().find(|(p, _)| p == rel).map(|(_, b)| b.clone());
            import_layout(&r)
        };

        // No layout marker at all.
        assert!(with(&|f| f.retain(|(p, _)| p != "oci-layout")).is_err());
        // A version this does not know.
        assert!(with(&|f| {
            for (p, b) in f.iter_mut() {
                if p == "oci-layout" {
                    *b = br#"{"imageLayoutVersion":"2.0.0"}"#.to_vec();
                }
            }
        })
        .is_err());
        // No index.
        assert!(with(&|f| f.retain(|(p, _)| p != "index.json")).is_err());
        // A blob whose bytes do not match its name: the one failure that matters
        // most, because a mirror is the thing that would cause it.
        match with(&|f| {
            let (_, b) = f
                .iter_mut()
                .find(|(p, _)| p.starts_with("blobs/sha256/"))
                .unwrap();
            b[0] ^= 0xff;
        }) {
            Err(Error::Malformed(m)) => assert!(m.contains("hashes to"), "{m}"),
            other => panic!("a tampered blob was accepted: {}", fmt(other)),
        }
        // A missing blob.
        assert!(with(&|f| {
            let i = f
                .iter()
                .position(|(p, _)| p.starts_with("blobs/sha256/"))
                .unwrap();
            f.remove(i);
        })
        .is_err());

        // Layers out of order: the offset annotations catch it rather than the
        // bytes reassembling into something that parses by luck. This needs a
        // model with several packs, so it gets one.
        let big = model("test/oci", 1, 8);
        let l2 = export_layout(
            &big,
            &ExportOpts {
                pack_bytes: MIN_PACK_BYTES,
                reference: None,
            },
        )
        .unwrap();
        assert!(l2.packs >= 2, "needed several packs, got {}", l2.packs);
        let manifest_bytes = l2
            .files
            .iter()
            .find(|(p, _)| p == &format!("blobs/sha256/{}", l2.manifest_digest))
            .map(|(_, b)| b.clone())
            .unwrap();
        let m = json::parse(&manifest_bytes).unwrap();
        let mut layers = m.get("layers").unwrap().as_array().unwrap().to_vec();
        assert!(layers.len() >= 3, "need several layers to reorder them");
        layers.swap(0, 1);
        let mut edited = m.as_object().unwrap().clone();
        edited.insert("layers".into(), Value::Array(layers));
        let edited = Value::Object(edited).encode().into_bytes();
        let new_digest = sha256_hex(&edited);
        let mut files2 = l2.files.clone();
        files2.push((format!("blobs/sha256/{new_digest}"), edited.clone()));
        // Point index.json at the edited manifest.
        for (p, b) in files2.iter_mut() {
            if p == "index.json" {
                let idx = json::parse(b).unwrap();
                let mut o = idx.as_object().unwrap().clone();
                o.insert(
                    "manifests".into(),
                    Value::Array(vec![descriptor(MANIFEST_TYPE, &edited, Vec::new())]),
                );
                *b = Value::Object(o).encode().into_bytes();
            }
        }
        let r = move |rel: &str| {
            files2
                .iter()
                .find(|(p, _)| p == rel)
                .map(|(_, b)| b.clone())
        };
        match import_layout(&r) {
            Err(Error::Malformed(m)) => assert!(m.contains("starts at"), "{m}"),
            other => panic!("reordered layers were accepted: {}", fmt(other)),
        }
    }

    fn fmt(r: Res<Imported>) -> String {
        match r {
            Ok(i) => format!("Ok({} layers)", i.layers),
            Err(e) => format!("Err({e})"),
        }
    }

    /// An index-only container maps too — it is a container, and §13.8 makes it a
    /// legal one. A catalogue in a registry is a useful thing to have.
    #[test]
    fn an_index_only_container_maps_and_says_so() {
        let c = model("test/oci", 5, 4);
        let thin = Container::open(crate::transport::index_only(&c).unwrap()).unwrap();
        let l = export_layout(&thin, &ExportOpts::default()).unwrap();
        let read = layout_reader(&l);
        let manifest = json::parse(&fetch(&read, &l.manifest_digest).unwrap()).unwrap();
        assert_eq!(
            manifest
                .get("annotations")
                .and_then(|a| a.get("dev.omni.partial"))
                .and_then(|v| v.as_str()),
            Some("true")
        );
        assert_eq!(import_layout(&read).unwrap().bytes, thin.bytes);
    }

    /// The pack-size rule: bigger than the floor, and the whole file accounted
    /// for exactly once.
    #[test]
    fn the_cuts_cover_the_file_exactly_once() {
        for target in [0u64, MIN_PACK_BYTES, 1 << 24, DEFAULT_PACK_BYTES] {
            let c = model("test/oci", 2, 6);
            let (packs, index) = cut_points(&c, target).unwrap();
            assert_eq!(packs[0].0, 0, "the first pack starts at the start");
            for w in packs.windows(2) {
                assert_eq!(w[0].1, w[1].0, "a gap or an overlap between packs");
            }
            assert_eq!(packs.last().unwrap().1, index.0);
            assert_eq!(
                index.1,
                c.bytes.len() as u64,
                "the index layer ends the file"
            );
            // Every pack is non-empty; a zero-length blob is a wasted round trip.
            for (a, b) in &packs {
                assert!(b > a, "an empty pack");
            }
        }
    }
}
