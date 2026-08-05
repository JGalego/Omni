//! Object stores (§01.8) — the same object graph, in different places.
//!
//! §01 axiom A5 says nothing in an object refers to a file, an offset or a
//! URL. Everything is a digest. The payoff is this module: a container, a
//! directory and an in-process map are interchangeable backings for the same
//! graph, and moving a model between them is a copy loop rather than a
//! conversion.
//!
//! Three stores are implemented here — [`MemoryStore`], [`DirStore`] and
//! [`ContainerStore`] — plus [`Layered`], which resolves through a stack of
//! them. Layering needs no invalidation logic at all: because identity is a
//! hash, an entry is either present and correct or absent. That is the whole
//! cache-coherence design, and it is a consequence of content addressing
//! rather than a feature anyone had to build.
//!
//! ## Types live in refs, not in stores
//!
//! A store maps digest to bytes and nothing else — no otype column, no
//! metadata sidecar. Object types are carried by the `[otype, digest]` refs
//! that point at objects, so the type of every object is recovered by walking
//! the graph from a root whose type is known. [`walk`] does exactly that, and
//! it is how [`DirStore`] can be a plain directory of files and still round
//! trip through a container that has a typed index.

use crate::cbor::{self, Value};
use crate::container::{collect_typed_refs, otype, Container, Digest, HashAlgo, Object};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Cbor(cbor::Error),
    /// The bytes under a digest do not hash to it. A store that returns this
    /// is corrupt; a store that does not check cannot say so.
    Corrupt(String),
    /// Not an OMNI directory store, or one this build cannot read.
    BadStore(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Cbor(e) => write!(f, "cbor: {e}"),
            Error::Corrupt(m) => write!(f, "corrupt store: {m}"),
            Error::BadStore(m) => write!(f, "not a usable store: {m}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<cbor::Error> for Error {
    fn from(e: cbor::Error) -> Self {
        Error::Cbor(e)
    }
}

type Res<T> = Result<T, Error>;

/// Read access to a set of objects (§01.8).
pub trait Store {
    /// The digest algorithm this store's identities are computed under. Two
    /// stores with different algorithms hold unrelated namespaces even if they
    /// hold the same models.
    fn hash(&self) -> HashAlgo;

    /// `None` means absent, which is not an error: a partial store is legal
    /// (§01.4) and a layered lookup expects misses.
    fn resolve(&self, d: &Digest) -> Res<Option<Vec<u8>>>;

    /// Byte range of an object.
    ///
    /// The default reads the whole object and slices, which is correct but
    /// forfeits the point. Note that a range read cannot be verified on its
    /// own: the object digest covers all the bytes. Checking a range in
    /// isolation needs a Bao tree ([`crate::bao`], §13.3), and returning
    /// unverified bytes merely because the caller asked for fewer of them
    /// would be a strange bargain.
    fn resolve_range(&self, d: &Digest, off: u64, n: u64) -> Res<Option<Vec<u8>>> {
        Ok(self.resolve(d)?.map(|b| {
            let s = (off as usize).min(b.len());
            let e = s.saturating_add(n as usize).min(b.len());
            b[s..e].to_vec()
        }))
    }

    fn has(&self, d: &Digest) -> Res<bool> {
        Ok(self.resolve(d)?.is_some())
    }
}

/// A store that can be written to.
pub trait WritableStore: Store {
    /// Stores `bytes` and returns their digest. Writing an object that is
    /// already present is a no-op, not an error — that is deduplication, and
    /// it needs no separate code path.
    fn put(&mut self, bytes: &[u8]) -> Res<Digest>;
}

/// A store whose contents can be listed.
pub trait EnumerableStore: Store {
    fn iter(&self) -> Res<Vec<Digest>>;
}

// ------------------------------------------------------------------ memory --

/// In-process store. The layer a runtime puts in front of everything else.
#[derive(Default, Clone)]
pub struct MemoryStore {
    hash: HashAlgo,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl MemoryStore {
    pub fn new(hash: HashAlgo) -> Self {
        MemoryStore {
            hash,
            objects: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// The objects, ordered by digest, ready for [`crate::pack`].
    pub fn objects(&self, types: &BTreeMap<Digest, u16>) -> Vec<Object> {
        self.objects
            .iter()
            .map(|(d, payload)| Object {
                otype: types.get(d).copied().unwrap_or(otype::BLOB),
                payload: payload.clone(),
                oflags: crate::container::oflags::CRITICAL | crate::container::oflags::SAFE_TO_COPY,
                stored: None,
            })
            .collect()
    }
}

impl Store for MemoryStore {
    fn hash(&self) -> HashAlgo {
        self.hash
    }
    fn resolve(&self, d: &Digest) -> Res<Option<Vec<u8>>> {
        Ok(self.objects.get(d).cloned())
    }
    fn has(&self, d: &Digest) -> Res<bool> {
        Ok(self.objects.contains_key(d))
    }
}

impl WritableStore for MemoryStore {
    fn put(&mut self, bytes: &[u8]) -> Res<Digest> {
        let d = self.hash.digest(bytes);
        self.objects.entry(d).or_insert_with(|| bytes.to_vec());
        Ok(d)
    }
}

impl EnumerableStore for MemoryStore {
    fn iter(&self) -> Res<Vec<Digest>> {
        Ok(self.objects.keys().copied().collect())
    }
}

// --------------------------------------------------------------- container --

/// A sealed `.omni` file, read-only (§01.8: append-only in general; this
/// implementation does not append).
pub struct ContainerStore {
    inner: Container,
}

impl ContainerStore {
    pub fn new(inner: Container) -> Self {
        ContainerStore { inner }
    }

    pub fn container(&self) -> &Container {
        &self.inner
    }

    /// The root object's ref, `[otype, digest]`. A container names its root in
    /// the header, so unlike a bare directory it needs no side channel.
    pub fn root(&self) -> (u16, Digest) {
        (otype::MANIFEST, self.inner.header.root_digest)
    }

    /// The otype the container's index records for an object. A directory
    /// store cannot answer this; a container can, because its index is typed.
    pub fn otype_of(&self, d: &Digest) -> Option<u16> {
        self.inner.find(d).map(|e| e.otype)
    }
}

impl Store for ContainerStore {
    fn hash(&self) -> HashAlgo {
        self.inner.header.hash
    }

    fn resolve(&self, d: &Digest) -> Res<Option<Vec<u8>>> {
        // `read` rather than `get`: a compressed copy is still the same object,
        // and a store's callers should not have to know how it was packed.
        match self.inner.read(d) {
            Ok(b) => Ok(Some(b)),
            // A digest that is not in the index is absent, not an error; a
            // digest that is in the index but fails to verify is corruption
            // and must not be quietly reported as absence.
            Err(crate::container::Error::NotFound(_)) => Ok(None),
            Err(e) => Err(Error::Corrupt(e.to_string())),
        }
    }

    fn has(&self, d: &Digest) -> Res<bool> {
        Ok(self.inner.find(d).is_some())
    }
}

impl EnumerableStore for ContainerStore {
    fn iter(&self) -> Res<Vec<Digest>> {
        Ok(self.inner.index.iter().map(|e| e.digest).collect())
    }
}

// --------------------------------------------------------------- directory --

const CONFIG_TYPE: &str = "omni.store/config";
const ROOT_TYPE: &str = "omni.store/root";

/// A `.omnid/` directory store (§01.8).
///
/// ```text
/// .omnid/
///   config              canonical CBOR: the digest algorithm
///   root                canonical CBOR: [otype, digest] of the graph root
///   objects/ab/cdef…    one file per object, raw payload bytes
/// ```
///
/// The two-character fan-out is Git's, for the same reason: directories with
/// a million entries are slow on most filesystems and unpleasant on all of
/// them.
pub struct DirStore {
    path: PathBuf,
    hash: HashAlgo,
}

impl DirStore {
    /// Creates a store, or opens one that already exists with the same
    /// algorithm.
    pub fn create(path: impl AsRef<Path>, hash: HashAlgo) -> Res<DirStore> {
        let path = path.as_ref().to_path_buf();
        if path.join("config").exists() {
            let s = DirStore::open(&path)?;
            if s.hash != hash {
                return Err(Error::BadStore(format!(
                    "existing store uses {}, not {}",
                    s.hash.name(),
                    hash.name()
                )));
            }
            return Ok(s);
        }
        std::fs::create_dir_all(path.join("objects"))?;
        let config = Value::map(vec![
            ("t", Value::text(CONFIG_TYPE)),
            ("v", Value::U(1)),
            ("hash", Value::text(hash.name())),
        ]);
        std::fs::write(path.join("config"), config.encode())?;
        Ok(DirStore { path, hash })
    }

    pub fn open(path: impl AsRef<Path>) -> Res<DirStore> {
        let path = path.as_ref().to_path_buf();
        let raw = std::fs::read(path.join("config"))
            .map_err(|e| Error::BadStore(format!("{}/config: {e}", path.display())))?;
        let config = cbor::decode(&raw)?;
        if config.get("t").and_then(|v| v.as_str()) != Some(CONFIG_TYPE) {
            return Err(Error::BadStore("config is not omni.store/config".into()));
        }
        let name = config
            .get("hash")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::BadStore("config has no hash".into()))?;
        let hash = HashAlgo::parse(name)
            .ok_or_else(|| Error::BadStore(format!("unsupported hash algorithm `{name}`")))?;
        Ok(DirStore { path, hash })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn object_path(&self, d: &Digest) -> PathBuf {
        let h = crate::hex(d);
        self.path.join("objects").join(&h[..2]).join(&h[2..])
    }

    /// Records the graph root. A directory has no header to put it in, so it
    /// goes in a file — the equivalent of Git's `HEAD`.
    pub fn set_root(&self, otype: u16, d: &Digest) -> Res<()> {
        let v = Value::map(vec![
            ("t", Value::text(ROOT_TYPE)),
            ("v", Value::U(1)),
            (
                "root",
                Value::Array(vec![Value::U(otype as u64), Value::Bytes(d.to_vec())]),
            ),
        ]);
        std::fs::write(self.path.join("root"), v.encode())?;
        Ok(())
    }

    pub fn root(&self) -> Res<Option<(u16, Digest)>> {
        let p = self.path.join("root");
        if !p.exists() {
            return Ok(None);
        }
        let v = cbor::decode(&std::fs::read(p)?)?;
        if v.get("t").and_then(|x| x.as_str()) != Some(ROOT_TYPE) {
            return Err(Error::BadStore("root is not omni.store/root".into()));
        }
        let r = v
            .get("root")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::BadStore("root has no ref".into()))?;
        if r.len() != 2 {
            return Err(Error::BadStore("root ref is malformed".into()));
        }
        let t = r[0]
            .as_u64()
            .ok_or_else(|| Error::BadStore("root otype is not an integer".into()))?;
        let b = r[1]
            .as_bytes()
            .filter(|b| b.len() == 32)
            .ok_or_else(|| Error::BadStore("root digest is malformed".into()))?;
        let mut d = [0u8; 32];
        d.copy_from_slice(b);
        Ok(Some((t as u16, d)))
    }
}

impl Store for DirStore {
    fn hash(&self) -> HashAlgo {
        self.hash
    }

    fn resolve(&self, d: &Digest) -> Res<Option<Vec<u8>>> {
        match std::fs::read(self.object_path(d)) {
            Ok(b) => {
                // The filename is a claim; the bytes are the fact. Checking
                // costs one hash and catches bit rot, a bad rsync and a
                // hand-edited file alike.
                if self.hash.digest(&b) != *d {
                    return Err(Error::Corrupt(format!(
                        "{} does not hash to its name",
                        self.object_path(d).display()
                    )));
                }
                Ok(Some(b))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }

    fn has(&self, d: &Digest) -> Res<bool> {
        Ok(self.object_path(d).exists())
    }
}

impl WritableStore for DirStore {
    fn put(&mut self, bytes: &[u8]) -> Res<Digest> {
        let d = self.hash.digest(bytes);
        let p = self.object_path(&d);
        if p.exists() {
            return Ok(d);
        }
        std::fs::create_dir_all(p.parent().expect("object path has a parent"))?;
        // Write-then-rename: a reader must never see a half-written object
        // under a name that promises its digest.
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &p)?;
        Ok(d)
    }
}

impl EnumerableStore for DirStore {
    fn iter(&self) -> Res<Vec<Digest>> {
        let mut out = Vec::new();
        let objects = self.path.join("objects");
        if !objects.exists() {
            return Ok(out);
        }
        for lvl1 in std::fs::read_dir(&objects)? {
            let lvl1 = lvl1?;
            if !lvl1.file_type()?.is_dir() {
                continue;
            }
            let prefix = lvl1.file_name().to_string_lossy().to_string();
            for f in std::fs::read_dir(lvl1.path())? {
                let f = f?;
                let name = format!("{prefix}{}", f.file_name().to_string_lossy());
                if let Some(d) = parse_hex_digest(&name) {
                    out.push(d);
                }
            }
        }
        out.sort_unstable();
        Ok(out)
    }
}

fn parse_hex_digest(s: &str) -> Option<Digest> {
    if s.len() != 64 {
        return None;
    }
    let mut d = [0u8; 32];
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(d)
}

// ----------------------------------------------------------------- layered --

/// Resolves through a stack of stores, nearest first (§01.8).
///
/// The canonical arrangement is `Memory → Directory → Container → HTTP`. There
/// is no invalidation, no TTL and no coherence protocol, because a hit is
/// correct by construction.
pub struct Layered<'a> {
    layers: Vec<&'a dyn Store>,
}

impl<'a> Layered<'a> {
    pub fn new(layers: Vec<&'a dyn Store>) -> Res<Layered<'a>> {
        match layers.first() {
            None => Err(Error::BadStore("a layered store needs a layer".into())),
            Some(first) => {
                let h = first.hash();
                if layers.iter().any(|l| l.hash() != h) {
                    return Err(Error::BadStore(
                        "layers disagree about the digest algorithm".into(),
                    ));
                }
                Ok(Layered { layers })
            }
        }
    }

    /// Index of the layer that answered, for cache-hit accounting.
    pub fn resolve_from(&self, d: &Digest) -> Res<Option<(usize, Vec<u8>)>> {
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(b) = l.resolve(d)? {
                return Ok(Some((i, b)));
            }
        }
        Ok(None)
    }
}

impl Store for Layered<'_> {
    fn hash(&self) -> HashAlgo {
        self.layers[0].hash()
    }
    fn resolve(&self, d: &Digest) -> Res<Option<Vec<u8>>> {
        Ok(self.resolve_from(d)?.map(|(_, b)| b))
    }
    fn has(&self, d: &Digest) -> Res<bool> {
        for l in &self.layers {
            if l.has(d)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// -------------------------------------------------------------- traversal --

/// The result of walking a graph: every object reached, with the otype the
/// refs gave it, plus the refs that pointed nowhere.
#[derive(Debug, Default)]
pub struct Walk {
    /// Reachable objects, digest to otype, in digest order.
    pub objects: BTreeMap<Digest, u16>,
    /// Refs to objects the store does not have (§01.4: incomplete, not
    /// invalid).
    pub dangling: BTreeSet<Digest>,
}

/// Walks the object graph from a typed root, recovering every object's type
/// from the refs that point at it.
///
/// This is what lets a store be a bare digest-to-bytes map: types are in the
/// graph, not in the storage layer. Only structure objects are parsed as CBOR;
/// blobs are opaque and are not descended into.
pub fn walk(store: &dyn Store, root_otype: u16, root: &Digest) -> Res<Walk> {
    let mut w = Walk::default();
    let mut stack = vec![(root_otype, *root)];
    while let Some((t, d)) = stack.pop() {
        if w.objects.contains_key(&d) {
            continue;
        }
        let bytes = match store.resolve(&d)? {
            Some(b) => b,
            None => {
                w.dangling.insert(d);
                continue;
            }
        };
        w.objects.insert(d, t);
        if t == otype::BLOB {
            continue;
        }
        // A structure object that will not decode is corruption, not an
        // unknown extension: the digest already matched, so the bytes are the
        // ones the author wrote.
        let v = cbor::decode(&bytes)?;
        let mut refs = Vec::new();
        collect_typed_refs(&v, &mut refs);
        for (rt, rd) in refs {
            if !w.objects.contains_key(&rd) {
                stack.push((rt, rd));
            }
        }
    }
    Ok(w)
}

/// Copies every object reachable from `root` into `dst`.
///
/// Returns the number of objects copied and the number already present.
/// Objects that were already there are the interesting figure: it is dedup
/// across models, measured, with no dedup code anywhere.
pub fn copy_reachable(
    src: &dyn Store,
    dst: &mut dyn WritableStore,
    root_otype: u16,
    root: &Digest,
) -> Res<(usize, usize)> {
    if src.hash() != dst.hash() {
        return Err(Error::BadStore(format!(
            "cannot copy between stores using {} and {}",
            src.hash().name(),
            dst.hash().name()
        )));
    }
    let w = walk(src, root_otype, root)?;
    let (mut copied, mut present) = (0, 0);
    for d in w.objects.keys() {
        if dst.has(d)? {
            present += 1;
            continue;
        }
        let bytes = src
            .resolve(d)?
            .ok_or_else(|| Error::Corrupt(format!("{} vanished mid-copy", crate::hex(d))))?;
        let got = dst.put(&bytes)?;
        debug_assert_eq!(got, *d);
        copied += 1;
    }
    Ok((copied, present))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::oflags;
    use crate::{pack, ModelBuilder, PackOptions, TensorSpec};

    fn tiny(hash: HashAlgo) -> (Vec<Object>, Digest) {
        ModelBuilder::new("omni/store-test")
            .hash(hash)
            .chunk_size(4096)
            .arch("test", vec![])
            .tensor(TensorSpec {
                name: "w".into(),
                shape: vec![64, 64],
                dtype: crate::DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..64 * 64 * 4).map(|i| (i % 251) as u8).collect(),
            })
            .build()
    }

    fn container(hash: HashAlgo) -> (ContainerStore, Digest) {
        let (objs, root) = tiny(hash);
        let bytes = pack(
            &objs,
            &root,
            &PackOptions {
                hash,
                ..Default::default()
            },
        )
        .unwrap();
        (ContainerStore::new(Container::open(bytes).unwrap()), root)
    }

    #[test]
    fn memory_store_round_trips_and_dedups() {
        let mut m = MemoryStore::new(HashAlgo::default());
        let a = m.put(b"hello").unwrap();
        let b = m.put(b"hello").unwrap();
        assert_eq!(a, b, "identical bytes are one object");
        assert_eq!(m.len(), 1);
        assert_eq!(m.resolve(&a).unwrap().unwrap(), b"hello");
        assert!(m.has(&a).unwrap());
        assert_eq!(
            m.resolve(&[0u8; 32]).unwrap(),
            None,
            "absence is not an error"
        );
        assert_eq!(m.resolve_range(&a, 1, 3).unwrap().unwrap(), b"ell");
    }

    #[test]
    fn directory_store_round_trips() {
        let dir = tempdir("dir-round-trip");
        let mut s = DirStore::create(&dir, HashAlgo::default()).unwrap();
        let d = s.put(b"weights").unwrap();
        assert_eq!(s.put(b"weights").unwrap(), d, "put is idempotent");
        assert_eq!(s.resolve(&d).unwrap().unwrap(), b"weights");
        assert_eq!(s.iter().unwrap(), vec![d]);

        // Reopening must recover the algorithm rather than assume it.
        let re = DirStore::open(&dir).unwrap();
        assert_eq!(re.hash(), HashAlgo::default());
        assert_eq!(re.resolve(&d).unwrap().unwrap(), b"weights");

        s.set_root(otype::MANIFEST, &d).unwrap();
        assert_eq!(
            DirStore::open(&dir).unwrap().root().unwrap(),
            Some((otype::MANIFEST, d))
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The filename is a claim about the contents. A store that does not check
    /// it will serve whatever an attacker, a bad disk or a careless `cp` put
    /// there.
    #[test]
    fn directory_store_detects_a_lying_filename() {
        let dir = tempdir("dir-lying-filename");
        let mut s = DirStore::create(&dir, HashAlgo::default()).unwrap();
        let d = s.put(b"genuine").unwrap();
        let p = s.object_path(&d);
        std::fs::write(&p, b"forged").unwrap();
        match s.resolve(&d) {
            Err(Error::Corrupt(_)) => {}
            other => panic!("expected corruption, got {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_container_is_a_store() {
        let (c, root) = container(HashAlgo::default());
        assert!(c.has(&root).unwrap());
        assert_eq!(c.root().0, otype::MANIFEST);
        assert_eq!(c.iter().unwrap().len(), c.container().index.len());
        assert_eq!(c.resolve(&[0u8; 32]).unwrap(), None);
    }

    /// The point of A5: the same graph moves between backings without
    /// conversion, and every digest survives the trip.
    #[test]
    fn a_graph_round_trips_container_to_directory_to_container() {
        let hash = HashAlgo::default();
        let (src, root) = container(hash);
        let dir = tempdir("graph-round-trip");
        let mut d = DirStore::create(&dir, hash).unwrap();

        let (copied, present) = copy_reachable(&src, &mut d, otype::MANIFEST, &root).unwrap();
        assert_eq!(present, 0);
        assert_eq!(copied, src.container().index.len());
        d.set_root(otype::MANIFEST, &root).unwrap();

        // Copying again moves nothing: every object is already there, by
        // digest, with no dedup logic involved.
        let (copied2, present2) = copy_reachable(&src, &mut d, otype::MANIFEST, &root).unwrap();
        assert_eq!((copied2, present2), (0, copied));

        // Rebuild a container from the directory and check it is the same file.
        let w = walk(&d, otype::MANIFEST, &root).unwrap();
        assert!(w.dangling.is_empty());
        let objects: Vec<Object> = w
            .objects
            .iter()
            .map(|(dig, t)| Object {
                otype: *t,
                payload: d.resolve(dig).unwrap().unwrap(),
                oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
                stored: None,
            })
            .collect();
        let rebuilt = pack(
            &objects,
            &root,
            &PackOptions {
                hash,
                ..Default::default()
            },
        )
        .unwrap();
        let original = pack(
            &src.container()
                .index
                .iter()
                .map(|e| Object {
                    otype: e.otype,
                    payload: src.resolve(&e.digest).unwrap().unwrap(),
                    oflags: e.oflags,
                    stored: None,
                })
                .collect::<Vec<_>>(),
            &root,
            &PackOptions {
                hash,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(rebuilt, original, "the round trip must be byte-exact");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Types are recovered from refs, so a store that knows nothing about
    /// otypes still yields a correctly typed graph.
    #[test]
    fn walking_recovers_every_object_type_from_refs() {
        let (src, root) = container(HashAlgo::default());
        let w = walk(&src, otype::MANIFEST, &root).unwrap();
        assert_eq!(w.objects.len(), src.container().index.len());
        for (d, t) in &w.objects {
            assert_eq!(
                Some(*t),
                src.otype_of(d),
                "walked type must match the container index for {}",
                crate::hex(d)
            );
        }
    }

    #[test]
    fn a_missing_object_is_dangling_not_fatal() {
        let (src, root) = container(HashAlgo::default());
        let mut m = MemoryStore::new(HashAlgo::default());
        // Copy everything except the blobs.
        for e in &src.container().index {
            if e.otype != otype::BLOB {
                m.put(&src.resolve(&e.digest).unwrap().unwrap()).unwrap();
            }
        }
        let w = walk(&m, otype::MANIFEST, &root).unwrap();
        assert!(!w.dangling.is_empty());
        assert!(w.objects.contains_key(&root));
    }

    #[test]
    fn layers_resolve_nearest_first() {
        let hash = HashAlgo::default();
        let (far, root) = container(hash);
        let mut near = MemoryStore::new(hash);
        let manifest = far.resolve(&root).unwrap().unwrap();
        near.put(&manifest).unwrap();

        let l = Layered::new(vec![&near, &far]).unwrap();
        assert_eq!(
            l.resolve_from(&root).unwrap().unwrap().0,
            0,
            "hit the cache"
        );

        // Something only the container has must come from the container.
        let blob = far
            .container()
            .index
            .iter()
            .find(|e| e.otype == otype::BLOB)
            .unwrap()
            .digest;
        assert_eq!(l.resolve_from(&blob).unwrap().unwrap().0, 1);
        assert_eq!(l.resolve(&[0u8; 32]).unwrap(), None);
    }

    /// Two stores under different algorithms hold unrelated namespaces, so
    /// layering or copying between them is a mistake worth refusing.
    #[test]
    fn mismatched_algorithms_are_refused() {
        let b3 = MemoryStore::new(HashAlgo::Blake3_256);
        let sha = MemoryStore::new(HashAlgo::Sha256);
        assert!(Layered::new(vec![&b3, &sha]).is_err());

        let (src, root) = container(HashAlgo::Blake3_256);
        let mut dst = MemoryStore::new(HashAlgo::Sha256);
        assert!(copy_reachable(&src, &mut dst, otype::MANIFEST, &root).is_err());
    }

    fn tempdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("omni-store-test-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&p).ok();
        p
    }
}
