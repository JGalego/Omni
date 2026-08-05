//! §09 — training state.
//!
//! Training state is optional and **separable**, and §09.1 makes separability a
//! normative rule rather than a nicety: a 70 B checkpoint with Adam moments is
//! ~1.7 TB and its inference artifact is 140 GB, and those must never be the
//! same download. In OMNI they are the same object graph with a different root,
//! which is only true if stripping the training root really does leave the
//! weights — every one of them, byte for byte — untouched. [`separate`] is how a
//! reader establishes that: it walks both sides of the graph and reports which
//! objects only training reaches (what `omni strip --training` removes), which
//! bytes each side costs, and any inference object that references a training
//! one, which would make the split impossible.
//!
//! What is here:
//!
//! * [`TrainingState`]: the §09.2 object — framework, step counters, the
//!   optimizer with its hyperparameters and schedule, gradients, EMA, the
//!   gradient scaler, RNG streams, the shard map, dataloader position and loss
//!   history. Optimizer moments are ordinary `TensorTable`s of ordinary tensors,
//!   which is what makes them chunked, deduplicated and delta-able against the
//!   previous checkpoint.
//! * [`RngStream`]: §09.3, including the distinction that matters — a
//!   counter-based generator (Philox, Threefry, ChaCha) can be reproduced across
//!   implementations, and a stateful CPU generator cannot. The second kind is
//!   stored honestly, as an opaque blob with its implementation named, and
//!   reported as non-portable rather than quietly accepted.
//! * [`ShardMap`]: §09.4's answer to the hard requirement — a checkpoint written
//!   by 512 ranks under FSDP read by 8 under tensor parallelism, with no
//!   conversion script. Logical tensors are stored; sharding is layout
//!   metadata; FSDP's flat parameter buffers are described by their
//!   `(param, offset, numel, shape)` table so each parameter is a slice of a
//!   literal.
//! * [`reshard`]: §09.4.2 — rewriting only the `ShardMap` when the chunking
//!   permits, and saying which tensors would need bytes moved when it does not.
//!
//! What is not: an opinion about optimizer *algorithms*. §09.8 is explicit that
//! `kind: "adamw"` plus `hyper` is a label and a parameter bag, and that
//! specifying optimizer semantics would be a research-tracking treadmill with no
//! interoperability benefit.

use crate::cbor::Value;
use crate::container::{otype, Digest};
use crate::dtype::DType;
use crate::expr::Ref;

/// Resolves an object by digest to its type and bytes — what a walk over the
/// object graph needs, and all it needs.
pub type Resolver<'a> = &'a dyn Fn(&Digest) -> Option<(u16, Vec<u8>)>;

pub const SCHEMA: &str = "omni.train/state";
pub const SHARDMAP_SCHEMA: &str = "omni.train/shardmap";

#[derive(Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "training state: {}", self.0)
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

fn err<T>(msg: impl Into<String>) -> Res<T> {
    Err(Error(msg.into()))
}

fn ref_value(r: &Ref) -> Value {
    Value::Array(vec![Value::U(r.0 as u64), Value::Bytes(r.1.to_vec())])
}

fn parse_ref(v: &Value) -> Option<Ref> {
    crate::expr::parse_ref_value(v).ok()
}

fn f64_of(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::U(n) => Some(*n as f64),
        Value::I(n) => Some(*n as f64),
        _ => None,
    }
}

// ---------------------------------------------------------------------- rng --

/// How a random number generator carries its state (§09.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngKind {
    /// Philox, Threefry, ChaCha: state is `(key, counter)`, so any
    /// implementation of the same function reproduces the same stream. These are
    /// the only generators OMNI can promise cross-implementation
    /// reproducibility for.
    Counter,
    /// A stateful CPU generator (Mersenne Twister, PCG in library form). Stored
    /// as an opaque blob with its implementation named: honest, and not
    /// portable.
    Opaque,
}

/// One captured RNG stream.
#[derive(Clone, Debug, PartialEq)]
pub struct RngStream {
    /// `global`, `cuda`, `dataloader`, `dropout`, `jax`, …
    pub scope: String,
    /// The implementation, e.g. `philox`, `pytorch-cpu`, `numpy-pcg64`.
    pub implementation: String,
    pub kind: RngKind,
    pub device: Option<u64>,
    pub worker: Option<u64>,
    /// Counter-based: the key.
    pub key: Vec<u64>,
    pub seed: Option<u64>,
    pub counter: Option<u64>,
    pub offset: Option<u64>,
    /// Opaque: the state blob.
    pub state: Option<Ref>,
}

impl RngStream {
    /// Whether this stream can be reproduced by a different implementation of
    /// the same generator (§09.3).
    pub fn portable(&self) -> bool {
        self.kind == RngKind::Counter
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("scope", Value::text(self.scope.clone())),
            ("impl", Value::text(self.implementation.clone())),
        ];
        if let Some(d) = self.device {
            p.push(("device", Value::U(d)));
        }
        if let Some(w) = self.worker {
            p.push(("worker", Value::U(w)));
        }
        if !self.key.is_empty() {
            p.push((
                "key",
                Value::Array(self.key.iter().map(|k| Value::U(*k)).collect()),
            ));
        }
        if let Some(s) = self.seed {
            p.push(("seed", Value::U(s)));
        }
        if let Some(c) = self.counter {
            p.push(("counter", Value::U(c)));
        }
        if let Some(o) = self.offset {
            p.push(("offset", Value::U(o)));
        }
        if let Some(s) = &self.state {
            p.push(("state", ref_value(s)));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<RngStream> {
        let scope = v
            .get("scope")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("an RNG stream has no `scope`".into()))?
            .to_string();
        let implementation = v
            .get("impl")
            .and_then(|x| x.as_str())
            .ok_or_else(|| Error("an RNG stream has no `impl`".into()))?
            .to_string();
        let key: Vec<u64> = match v.get("key") {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).collect(),
            _ => Vec::new(),
        };
        let counter = v.get("counter").and_then(|x| x.as_u64());
        let offset = v.get("offset").and_then(|x| x.as_u64());
        let state = v.get("state").and_then(parse_ref);
        // §09.3 names the counter-based generators it trusts. Anything with an
        // opaque state blob is the other kind by construction.
        let counter_based = matches!(
            implementation.as_str(),
            "philox" | "threefry" | "chacha" | "counter" | "counter-based"
        ) || (state.is_none() && (counter.is_some() || offset.is_some()));
        Ok(RngStream {
            scope,
            implementation,
            kind: if counter_based {
                RngKind::Counter
            } else {
                RngKind::Opaque
            },
            device: v.get("device").and_then(|x| x.as_u64()),
            worker: v.get("worker").and_then(|x| x.as_u64()),
            key,
            seed: v.get("seed").and_then(|x| x.as_u64()),
            counter,
            offset,
            state,
        })
    }
}

// ---------------------------------------------------------------- shard map --

/// A device mesh: named dimensions with extents (§09.4).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Mesh {
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
}

impl Mesh {
    pub fn size(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Parses `dp=8,tp=8` — the form `omni reshard --mesh` takes.
    pub fn parse(spec: &str) -> Res<Mesh> {
        let mut m = Mesh::default();
        for part in spec.split(',').filter(|p| !p.is_empty()) {
            let Some((name, n)) = part.split_once('=') else {
                return err(format!("`{part}` is not `dim=extent`"));
            };
            let Ok(n) = n.parse::<u64>() else {
                return err(format!("`{n}` is not an extent"));
            };
            if n == 0 {
                return err("a mesh dimension cannot be zero");
            }
            if m.dims.iter().any(|d| d == name) {
                return err(format!("mesh dimension `{name}` appears twice"));
            }
            m.dims.push(name.to_string());
            m.shape.push(n);
        }
        if m.dims.is_empty() {
            return err("a mesh needs at least one dimension");
        }
        Ok(m)
    }

    pub fn extent(&self, dim: &str) -> Option<u64> {
        self.dims
            .iter()
            .position(|d| d == dim)
            .map(|i| self.shape[i])
    }

    fn to_value(&self) -> Value {
        Value::map(vec![
            (
                "dims",
                Value::Array(self.dims.iter().map(|d| Value::text(d.clone())).collect()),
            ),
            (
                "shape",
                Value::Array(self.shape.iter().map(|n| Value::U(*n)).collect()),
            ),
        ])
    }

    fn from_value(v: &Value) -> Res<Mesh> {
        let dims: Vec<String> = match v.get("dims") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            _ => return err("a mesh has no `dims`"),
        };
        let shape: Vec<u64> = match v.get("shape") {
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).collect(),
            _ => return err("a mesh has no `shape`"),
        };
        if dims.len() != shape.len() {
            return err("a mesh's `dims` and `shape` disagree in length");
        }
        Ok(Mesh { dims, shape })
    }

    pub fn describe(&self) -> String {
        self.dims
            .iter()
            .zip(&self.shape)
            .map(|(d, n)| format!("{d}={n}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// How one axis of a tensor is split across one mesh dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sharding {
    pub axis: usize,
    pub mesh_dim: String,
    pub parts: u64,
}

/// One shard: where it lives, what range of the logical tensor it holds, and the
/// expression that reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct Shard {
    /// Mesh coordinates, e.g. `{"tp": 1}`.
    pub coord: Vec<(String, u64)>,
    /// Half-open ranges per axis.
    pub range: Vec<(u64, u64)>,
    pub value: Option<Ref>,
}

/// The placement of one logical tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub logical_shape: Vec<u64>,
    pub sharding: Vec<Sharding>,
    pub shards: Vec<Shard>,
}

impl Placement {
    /// Whether the shards tile the logical tensor exactly: no gaps, no overlaps.
    ///
    /// This is the property that makes resharding an expression rewrite instead
    /// of a conversion script, so it is checked rather than assumed.
    pub fn tiles(&self) -> Result<(), String> {
        let n: u64 = self.logical_shape.iter().product();
        let mut covered = 0u64;
        for s in &self.shards {
            if s.range.len() != self.logical_shape.len() {
                return Err(format!(
                    "a shard declares {} range(s) for a rank-{} tensor",
                    s.range.len(),
                    self.logical_shape.len()
                ));
            }
            let mut count = 1u64;
            for (i, (lo, hi)) in s.range.iter().enumerate() {
                if hi <= lo || *hi > self.logical_shape[i] {
                    return Err(format!(
                        "a shard's range [{lo}, {hi}) is outside axis {i} of length {}",
                        self.logical_shape[i]
                    ));
                }
                count *= hi - lo;
            }
            covered += count;
        }
        if covered != n {
            return Err(format!(
                "the shards cover {covered} of {n} elements; they must tile the tensor exactly"
            ));
        }
        Ok(())
    }
}

/// FSDP's flat-parameter buffer description (§09.4).
///
/// The thing that makes FSDP checkpoints notoriously non-portable is that
/// parameters are concatenated into one opaque buffer. Recording the table makes
/// each parameter a `reshape(slice(flat, …))`: zero copy, zero conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatParam {
    pub name: String,
    pub offset: u64,
    pub numel: u64,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShardMap {
    pub world_size: u64,
    pub mesh: Mesh,
    /// `fsdp | zero1 | zero2 | zero3 | tp | pp | ep | hybrid | megatron`
    pub strategy: String,
    pub placements: Vec<(String, Placement)>,
    pub flat_params: Vec<FlatParam>,
}

impl ShardMap {
    pub fn to_value(&self) -> Value {
        let mut p = vec![
            ("t", Value::text(SHARDMAP_SCHEMA)),
            ("v", Value::U(1)),
            (
                "world",
                Value::map(vec![
                    ("size", Value::U(self.world_size)),
                    ("mesh", self.mesh.to_value()),
                ]),
            ),
            ("strategy", Value::text(self.strategy.clone())),
            (
                "placements",
                Value::Map(
                    self.placements
                        .iter()
                        .map(|(name, pl)| {
                            (
                                Value::text(name.clone()),
                                Value::map(vec![
                                    (
                                        "logical_shape",
                                        Value::Array(
                                            pl.logical_shape.iter().map(|n| Value::U(*n)).collect(),
                                        ),
                                    ),
                                    (
                                        "sharding",
                                        Value::Array(
                                            pl.sharding
                                                .iter()
                                                .map(|s| {
                                                    Value::map(vec![
                                                        ("axis", Value::U(s.axis as u64)),
                                                        (
                                                            "mesh_dim",
                                                            Value::text(s.mesh_dim.clone()),
                                                        ),
                                                        ("parts", Value::U(s.parts)),
                                                    ])
                                                })
                                                .collect(),
                                        ),
                                    ),
                                    (
                                        "shards",
                                        Value::Array(
                                            pl.shards
                                                .iter()
                                                .map(|s| {
                                                    let mut q = vec![
                                                        (
                                                            "coord",
                                                            Value::Map(
                                                                s.coord
                                                                    .iter()
                                                                    .map(|(d, i)| {
                                                                        (
                                                                            Value::text(d.clone()),
                                                                            Value::U(*i),
                                                                        )
                                                                    })
                                                                    .collect(),
                                                            ),
                                                        ),
                                                        (
                                                            "range",
                                                            Value::Array(
                                                                s.range
                                                                    .iter()
                                                                    .map(|(lo, hi)| {
                                                                        Value::Array(vec![
                                                                            Value::U(*lo),
                                                                            Value::U(*hi),
                                                                        ])
                                                                    })
                                                                    .collect(),
                                                            ),
                                                        ),
                                                    ];
                                                    if let Some(r) = &s.value {
                                                        q.push(("value", ref_value(r)));
                                                    }
                                                    Value::map(q)
                                                })
                                                .collect(),
                                        ),
                                    ),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
        ];
        if !self.flat_params.is_empty() {
            p.push((
                "flat_params",
                Value::Array(
                    self.flat_params
                        .iter()
                        .map(|f| {
                            Value::map(vec![
                                ("param", Value::text(f.name.clone())),
                                ("offset", Value::U(f.offset)),
                                ("numel", Value::U(f.numel)),
                                (
                                    "orig_shape",
                                    Value::Array(f.shape.iter().map(|n| Value::U(*n)).collect()),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<ShardMap> {
        if v.get("t").and_then(|x| x.as_str()) != Some(SHARDMAP_SCHEMA) {
            return err("not an omni.train/shardmap object");
        }
        let world = v
            .get("world")
            .ok_or_else(|| Error("a shard map has no `world`".into()))?;
        let mesh = match world.get("mesh") {
            Some(m) => Mesh::from_value(m)?,
            None => return err("a shard map has no mesh"),
        };
        let world_size = world.get("size").and_then(|x| x.as_u64()).unwrap_or(0);
        if world_size != 0 && world_size != mesh.size() {
            return err(format!(
                "world size {world_size} does not equal the mesh's {} devices",
                mesh.size()
            ));
        }
        let mut placements = Vec::new();
        if let Some(Value::Map(m)) = v.get("placements") {
            for (name, pl) in m {
                let name = name
                    .as_str()
                    .ok_or_else(|| Error("a placement name is not text".into()))?
                    .to_string();
                let logical_shape: Vec<u64> = match pl.get("logical_shape") {
                    Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).collect(),
                    _ => return err(format!("`{name}` has no `logical_shape`")),
                };
                let mut sharding = Vec::new();
                if let Some(Value::Array(a)) = pl.get("sharding") {
                    for s in a {
                        sharding.push(Sharding {
                            axis: s.get("axis").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                            mesh_dim: s
                                .get("mesh_dim")
                                .and_then(|x| x.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            parts: s.get("parts").and_then(|x| x.as_u64()).unwrap_or(1),
                        });
                    }
                }
                let mut shards = Vec::new();
                if let Some(Value::Array(a)) = pl.get("shards") {
                    for s in a {
                        let coord = match s.get("coord") {
                            Some(Value::Map(m)) => m
                                .iter()
                                .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.as_u64()?)))
                                .collect(),
                            _ => Vec::new(),
                        };
                        let range = match s.get("range") {
                            Some(Value::Array(a)) => a
                                .iter()
                                .filter_map(|r| match r {
                                    Value::Array(p) if p.len() == 2 => {
                                        Some((p[0].as_u64()?, p[1].as_u64()?))
                                    }
                                    _ => None,
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        shards.push(Shard {
                            coord,
                            range,
                            value: s.get("value").and_then(parse_ref),
                        });
                    }
                }
                placements.push((
                    name,
                    Placement {
                        logical_shape,
                        sharding,
                        shards,
                    },
                ));
            }
        }
        let mut flat_params = Vec::new();
        if let Some(Value::Array(a)) = v.get("flat_params") {
            for f in a {
                flat_params.push(FlatParam {
                    name: f
                        .get("param")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    offset: f.get("offset").and_then(|x| x.as_u64()).unwrap_or(0),
                    numel: f.get("numel").and_then(|x| x.as_u64()).unwrap_or(0),
                    shape: match f.get("orig_shape") {
                        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).collect(),
                        _ => Vec::new(),
                    },
                });
            }
        }
        Ok(ShardMap {
            world_size: if world_size == 0 {
                mesh.size()
            } else {
                world_size
            },
            mesh,
            strategy: v
                .get("strategy")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string(),
            placements,
            flat_params,
        })
    }

    /// Structural checks a reader can make without touching a weight.
    pub fn check(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, pl) in &self.placements {
            if let Err(e) = pl.tiles() {
                out.push(format!("R-N04 `{name}`: {e}"));
            }
            for s in &pl.sharding {
                if s.axis >= pl.logical_shape.len() {
                    out.push(format!(
                        "R-N05 `{name}`: sharded on axis {} of a rank-{} tensor",
                        s.axis,
                        pl.logical_shape.len()
                    ));
                }
                match self.mesh.extent(&s.mesh_dim) {
                    None => out.push(format!(
                        "R-N05 `{name}`: sharded over mesh dimension `{}`, which the mesh does \
                         not have",
                        s.mesh_dim
                    )),
                    Some(n) if n != s.parts => out.push(format!(
                        "R-N05 `{name}`: split into {} parts over a mesh dimension of {n}",
                        s.parts
                    )),
                    Some(_) => {}
                }
            }
            for s in &pl.shards {
                for (d, i) in &s.coord {
                    match self.mesh.extent(d) {
                        None => out.push(format!(
                            "R-N05 `{name}`: a shard sits at mesh dimension `{d}`, which does \
                             not exist"
                        )),
                        Some(n) if *i >= n => out.push(format!(
                            "R-N05 `{name}`: a shard sits at {d}={i} in a mesh of {n}"
                        )),
                        Some(_) => {}
                    }
                }
            }
        }
        for f in &self.flat_params {
            let n: u64 = f.shape.iter().product();
            if n != f.numel {
                out.push(format!(
                    "R-N06 flat parameter `{}` declares {} elements but a shape of {n}",
                    f.name, f.numel
                ));
            }
        }
        out
    }
}

/// The outcome of a reshard (§09.4.2).
#[derive(Clone, Debug, Default)]
pub struct Reshard {
    pub map: Option<ShardMap>,
    /// Tensors whose new ranges are covered by the existing shards' boundaries,
    /// so no bytes move.
    pub metadata_only: Vec<String>,
    /// Tensors whose new ranges cut across existing shard boundaries: the
    /// affected chunks have to be rewritten.
    pub needs_copy: Vec<String>,
}

/// Re-expresses a shard map over a different mesh.
///
/// §09.4.2's claim is that this rewrites *only* the `ShardMap` when the
/// underlying chunking permits the new ranges, and that when bytes must move,
/// only the affected ones do. Both halves are reported: a caller gets the new
/// map plus the list of tensors that cannot be resharded for free, which is the
/// difference between "no tensor bytes move" as a design goal and as a claim
/// about a specific checkpoint.
pub fn reshard(map: &ShardMap, target: &Mesh) -> Res<Reshard> {
    let mut out = Reshard::default();
    let mut placements = Vec::with_capacity(map.placements.len());
    for (name, pl) in &map.placements {
        // The axis this tensor is split on, if any. A replicated tensor is
        // trivially resharded: every rank holds all of it.
        let Some(sh) = pl.sharding.first() else {
            placements.push((name.clone(), pl.clone()));
            out.metadata_only.push(name.clone());
            continue;
        };
        let dim = &sh.mesh_dim;
        let Some(parts) = target.extent(dim) else {
            return err(format!(
                "the target mesh has no `{dim}` dimension, which `{name}` is sharded over"
            ));
        };
        if sh.axis >= pl.logical_shape.len() {
            return err(format!("`{name}` is sharded on an axis it does not have"));
        }
        let len = pl.logical_shape[sh.axis];
        if len % parts != 0 {
            return err(format!(
                "`{name}` has {len} rows on axis {}, which {parts} ranks cannot divide evenly",
                sh.axis
            ));
        }
        let step = len / parts;
        // Old boundaries. A new range that starts and ends on one of them can be
        // read from the existing shards without rewriting a byte.
        let mut boundaries: Vec<u64> = pl
            .shards
            .iter()
            .flat_map(|s| {
                s.range
                    .get(sh.axis)
                    .map(|(lo, hi)| [*lo, *hi])
                    .unwrap_or([0, 0])
            })
            .collect();
        boundaries.sort_unstable();
        boundaries.dedup();

        let mut shards = Vec::with_capacity(parts as usize);
        let mut free = true;
        for i in 0..parts {
            let lo = i * step;
            let hi = lo + step;
            if !boundaries.contains(&lo) || !boundaries.contains(&hi) {
                free = false;
            }
            // The value expression comes from whichever old shard contains this
            // range; when the new range is a subset of one old shard, reading it
            // is a range read (§04.7.4) rather than a copy.
            let value = pl
                .shards
                .iter()
                .find(|s| {
                    s.range
                        .get(sh.axis)
                        .is_some_and(|(a, b)| *a <= lo && hi <= *b)
                })
                .and_then(|s| s.value);
            let mut range = pl
                .logical_shape
                .iter()
                .map(|n| (0u64, *n))
                .collect::<Vec<_>>();
            range[sh.axis] = (lo, hi);
            shards.push(Shard {
                coord: vec![(dim.clone(), i)],
                range,
                value,
            });
        }
        if free {
            out.metadata_only.push(name.clone());
        } else {
            out.needs_copy.push(name.clone());
        }
        placements.push((
            name.clone(),
            Placement {
                logical_shape: pl.logical_shape.clone(),
                sharding: vec![Sharding {
                    axis: sh.axis,
                    mesh_dim: dim.clone(),
                    parts,
                }],
                shards,
            },
        ));
    }
    out.map = Some(ShardMap {
        world_size: target.size(),
        mesh: target.clone(),
        strategy: map.strategy.clone(),
        placements,
        flat_params: map.flat_params.clone(),
    });
    Ok(out)
}

// ----------------------------------------------------------------- optimizer --

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Optimizer {
    pub kind: String,
    /// A parameter bag, interpreted by the framework (§09.8).
    pub hyper: Vec<(String, Value)>,
    pub schedule: Vec<(String, Value)>,
    /// The moments, as an ordinary `TensorTable`.
    pub states: Option<Ref>,
    /// The fp32 master copy, when the run keeps one.
    pub master_weights: Option<Ref>,
    pub state_dtype: Option<DType>,
}

impl Optimizer {
    fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![("kind", Value::text(self.kind.clone()))];
        if !self.hyper.is_empty() {
            p.push((
                "hyper",
                Value::Map(
                    self.hyper
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if !self.schedule.is_empty() {
            p.push((
                "schedule",
                Value::Map(
                    self.schedule
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if let Some(r) = &self.states {
            p.push(("states", ref_value(r)));
        }
        if let Some(r) = &self.master_weights {
            p.push(("master_weights", ref_value(r)));
        }
        if let Some(d) = &self.state_dtype {
            p.push(("state_dtype", d.to_value()));
        }
        Value::map(p)
    }

    fn from_value(v: &Value) -> Res<Optimizer> {
        let pairs = |key: &str| -> Vec<(String, Value)> {
            match v.get(key) {
                Some(Value::Map(m)) => m
                    .iter()
                    .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                    .collect(),
                _ => Vec::new(),
            }
        };
        Ok(Optimizer {
            kind: v
                .get("kind")
                .and_then(|x| x.as_str())
                .ok_or_else(|| Error("the optimizer has no `kind`".into()))?
                .to_string(),
            hyper: pairs("hyper"),
            schedule: pairs("schedule"),
            states: v.get("states").and_then(parse_ref),
            master_weights: v.get("master_weights").and_then(parse_ref),
            state_dtype: match v.get("state_dtype") {
                Some(d) => Some(DType::from_value(d).map_err(Error)?),
                None => None,
            },
        })
    }

    pub fn learning_rate(&self) -> Option<f64> {
        self.hyper
            .iter()
            .find(|(k, _)| k == "lr")
            .and_then(|(_, v)| f64_of(v))
    }
}

/// Where a streaming dataloader had got to (§09.5).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Dataloader {
    pub kind: String,
    pub shard: Option<u64>,
    pub offset: Option<u64>,
    pub seed: Option<u64>,
    pub shuffle_buffer: Option<u64>,
    pub epoch: Option<u64>,
    pub consumed_digest: Option<Vec<u8>>,
    pub sample_bitmap: Option<Ref>,
}

impl Dataloader {
    /// Whether the position is exact enough to resume the same statistical run
    /// (§09.5), rather than merely restarting.
    pub fn exact(&self) -> bool {
        self.sample_bitmap.is_some() || (self.shard.is_some() && self.offset.is_some())
    }

    fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![("kind", Value::text(self.kind.clone()))];
        if self.shard.is_some() || self.offset.is_some() {
            let mut pos = Vec::new();
            if let Some(s) = self.shard {
                pos.push(("shard", Value::U(s)));
            }
            if let Some(o) = self.offset {
                pos.push(("offset", Value::U(o)));
            }
            p.push(("position", Value::map(pos)));
        }
        for (k, v) in [
            ("seed", self.seed),
            ("shuffle_buffer", self.shuffle_buffer),
            ("epoch", self.epoch),
        ] {
            if let Some(n) = v {
                p.push((k, Value::U(n)));
            }
        }
        if let Some(d) = &self.consumed_digest {
            p.push(("consumed_digest", Value::Bytes(d.clone())));
        }
        if let Some(r) = &self.sample_bitmap {
            p.push(("sample_bitmap", ref_value(r)));
        }
        Value::map(p)
    }

    fn from_value(v: &Value) -> Dataloader {
        Dataloader {
            kind: v
                .get("kind")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown")
                .to_string(),
            shard: v.get("position").and_then(|p| p.get("shard")?.as_u64()),
            offset: v.get("position").and_then(|p| p.get("offset")?.as_u64()),
            seed: v.get("seed").and_then(|x| x.as_u64()),
            shuffle_buffer: v.get("shuffle_buffer").and_then(|x| x.as_u64()),
            epoch: v.get("epoch").and_then(|x| x.as_u64()),
            consumed_digest: match v.get("consumed_digest") {
                Some(Value::Bytes(b)) => Some(b.clone()),
                _ => None,
            },
            sample_bitmap: v.get("sample_bitmap").and_then(parse_ref),
        }
    }
}

// ------------------------------------------------------------ training state --

#[derive(Clone, Debug, PartialEq, Default)]
pub struct TrainingState {
    pub framework: Vec<(String, Value)>,
    pub step: u64,
    pub epoch: Option<u64>,
    pub samples_seen: Option<u64>,
    pub tokens_seen: Option<u64>,
    pub wall_clock_s: Option<u64>,
    pub optimizer: Optimizer,
    /// §09.7: gradients are stored only when asked for, and their presence is
    /// reported prominently because it is almost never what anyone wanted.
    pub gradients: Option<Ref>,
    pub ema: Vec<(f64, Ref)>,
    pub grad_scaler: Vec<(String, Value)>,
    pub rng: Vec<RngStream>,
    pub shards: Option<Ref>,
    pub dataloader: Option<Dataloader>,
    pub loss_history: Option<Ref>,
    pub config: Option<Ref>,
}

impl TrainingState {
    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text(SCHEMA)),
            ("v", Value::U(1)),
            ("step", Value::U(self.step)),
            ("optimizer", self.optimizer.to_value()),
        ];
        if !self.framework.is_empty() {
            p.push((
                "framework",
                Value::Map(
                    self.framework
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        for (k, v) in [
            ("epoch", self.epoch),
            ("samples_seen", self.samples_seen),
            ("tokens_seen", self.tokens_seen),
            ("wall_clock_s", self.wall_clock_s),
        ] {
            if let Some(n) = v {
                p.push((k, Value::U(n)));
            }
        }
        if let Some(r) = &self.gradients {
            p.push(("gradients", ref_value(r)));
        }
        if !self.ema.is_empty() {
            p.push((
                "ema",
                Value::Array(
                    self.ema
                        .iter()
                        .map(|(decay, r)| {
                            Value::map(vec![
                                ("decay", Value::F64(*decay)),
                                ("tensors", ref_value(r)),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        if !self.grad_scaler.is_empty() {
            p.push((
                "grad_scaler",
                Value::Map(
                    self.grad_scaler
                        .iter()
                        .map(|(k, v)| (Value::text(k.clone()), v.clone()))
                        .collect(),
                ),
            ));
        }
        if !self.rng.is_empty() {
            p.push((
                "rng",
                Value::Array(self.rng.iter().map(RngStream::to_value).collect()),
            ));
        }
        if let Some(r) = &self.shards {
            p.push(("shards", ref_value(r)));
        }
        if let Some(d) = &self.dataloader {
            p.push(("dataloader", d.to_value()));
        }
        if let Some(r) = &self.loss_history {
            p.push(("loss_history", ref_value(r)));
        }
        if let Some(r) = &self.config {
            p.push(("config", ref_value(r)));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<TrainingState> {
        if v.get("t").and_then(|x| x.as_str()) != Some(SCHEMA) {
            return err("not an omni.train/state object");
        }
        Ok(TrainingState {
            framework: match v.get("framework") {
                Some(Value::Map(m)) => m
                    .iter()
                    .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                    .collect(),
                _ => Vec::new(),
            },
            step: v
                .get("step")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| Error("a training state has no `step`".into()))?,
            epoch: v.get("epoch").and_then(|x| x.as_u64()),
            samples_seen: v.get("samples_seen").and_then(|x| x.as_u64()),
            tokens_seen: v.get("tokens_seen").and_then(|x| x.as_u64()),
            wall_clock_s: v.get("wall_clock_s").and_then(|x| x.as_u64()),
            optimizer: match v.get("optimizer") {
                Some(o) => Optimizer::from_value(o)?,
                None => return err("a training state has no `optimizer`"),
            },
            gradients: v.get("gradients").and_then(parse_ref),
            ema: match v.get("ema") {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|e| Some((f64_of(e.get("decay")?)?, parse_ref(e.get("tensors")?)?)))
                    .collect(),
                _ => Vec::new(),
            },
            grad_scaler: match v.get("grad_scaler") {
                Some(Value::Map(m)) => m
                    .iter()
                    .filter_map(|(k, v)| k.as_str().map(|k| (k.to_string(), v.clone())))
                    .collect(),
                _ => Vec::new(),
            },
            rng: match v.get("rng") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(RngStream::from_value)
                    .collect::<Res<Vec<_>>>()?,
                _ => Vec::new(),
            },
            shards: v.get("shards").and_then(parse_ref),
            dataloader: v.get("dataloader").map(Dataloader::from_value),
            loss_history: v.get("loss_history").and_then(parse_ref),
            config: v.get("config").and_then(parse_ref),
        })
    }

    /// The RNG streams that cannot be reproduced by another implementation
    /// (§09.3). `omni verify --reproducible` reports these; a run with any of
    /// them is resumable but not portably so.
    pub fn non_portable_rng(&self) -> Vec<&RngStream> {
        self.rng.iter().filter(|r| !r.portable()).collect()
    }

    /// Whether every stream needed to replay the run is captured.
    ///
    /// "Reproducible" is not a property of the format; it is a property of what
    /// the writer chose to store. A checkpoint with no dropout stream cannot
    /// reproduce its own dropout, and saying so is more useful than a flag.
    pub fn reproducibility(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.rng.is_empty() {
            out.push("no RNG streams are captured, so the run cannot be replayed".into());
        }
        for r in self.non_portable_rng() {
            out.push(format!(
                "the `{}` stream is `{}`, a stateful generator: resumable in that \
                 implementation, not across implementations (§09.3)",
                r.scope, r.implementation
            ));
        }
        match &self.dataloader {
            None => out.push("no dataloader position: the data order is not resumable".into()),
            Some(d) if !d.exact() => out.push(format!(
                "the `{}` dataloader records no exact position, so resumption restarts the \
                 stream rather than continuing it (§09.5)",
                d.kind
            )),
            Some(_) => {}
        }
        out
    }
}

// --------------------------------------------------------- separability rules --

/// Which objects a container's training state reaches, and which its inference
/// graph does (§09.1). See [`separate`].
#[derive(Clone, Debug, Default)]
pub struct Separation {
    /// Reachable from the manifest without going through the training state.
    pub inference: std::collections::BTreeSet<Digest>,
    /// Reachable *only* through the training state — what `strip --training`
    /// removes.
    pub training_only: std::collections::BTreeSet<Digest>,
    /// R-N02 violations: an inference object that references a training object.
    pub violations: Vec<String>,
    pub inference_bytes: u64,
    pub training_bytes: u64,
}

/// The `Model` object with its `training` ref removed — the one edge
/// `omni strip --training` is allowed to cut (§09.1).
///
/// Removing it changes the model's digest, and so the manifest's, which is
/// correct and intended: the result is a different (derived) artifact, and
/// §11.4's subtlety applies. What it must *not* change is a single tensor
/// digest.
pub fn without_training(model: &Value) -> Value {
    match model {
        Value::Map(pairs) => Value::Map(
            pairs
                .iter()
                .filter(|(k, _)| k.as_str() != Some("training"))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Splits a container's object graph into the inference part and the part only
/// training needs.
///
/// The rule being checked is §09.1's: every object reachable only through
/// `TrainingState` must be removable *by reachability alone*, and no
/// inference-relevant object may point at a training one. Both directions are
/// walked, because the second is the one that quietly stops being true.
///
/// `training` is the digest of the `TrainingState` object, which lives at
/// `Model.training` (§00.4). The inference walk starts at the manifest and
/// treats that one ref as a wall.
pub fn separate(root: &Digest, training: Option<Digest>, resolve: Resolver<'_>) -> Separation {
    let mut s = Separation::default();
    s.inference = walk(&[*root], training.as_ref(), resolve, &mut s.inference_bytes);

    let Some(t) = training else {
        return s;
    };
    let mut training_bytes = 0u64;
    let all = walk(&[t], None, resolve, &mut training_bytes);
    let mut shared = 0u64;
    for d in &all {
        if s.inference.contains(d) {
            // Optimizer moments delta against the weights and share chunks with
            // them; counting those bytes twice would make "training adds N GB"
            // a fiction.
            if let Some((_, bytes)) = resolve(d) {
                shared += bytes.len() as u64;
            }
        } else {
            s.training_only.insert(*d);
        }
    }
    s.training_bytes = training_bytes.saturating_sub(shared);

    // R-N02: nothing on the inference path may reference the training graph. The
    // `Model.training` ref itself is the one legitimate edge, and it is the wall
    // the walk above stopped at rather than a violation.
    for d in &s.inference {
        let Some((ot, bytes)) = resolve(d) else {
            continue;
        };
        if ot == otype::BLOB {
            continue;
        }
        let Ok(v) = crate::cbor::decode(&bytes) else {
            continue;
        };
        let mut refs = Vec::new();
        collect_refs(&v, &mut refs);
        for r in refs {
            // The `Model.training` edge is the wall the inference walk stopped
            // at, not a violation: it is what makes the split possible.
            if r == t {
                continue;
            }
            if s.training_only.contains(&r) {
                s.violations.push(format!(
                    "R-N02 {} references the training-only object {}",
                    crate::sha256::hex(d),
                    crate::sha256::hex(&r)
                ));
            }
        }
    }
    s
}

fn walk(
    roots: &[Digest],
    skip: Option<&Digest>,
    resolve: Resolver<'_>,
    bytes_out: &mut u64,
) -> std::collections::BTreeSet<Digest> {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = roots.to_vec();
    while let Some(d) = stack.pop() {
        if skip == Some(&d) || !seen.insert(d) {
            continue;
        }
        let Some((t, bytes)) = resolve(&d) else {
            continue;
        };
        *bytes_out += bytes.len() as u64;
        if t == otype::BLOB {
            continue;
        }
        if let Ok(v) = crate::cbor::decode(&bytes) {
            collect_refs(&v, &mut stack);
        }
    }
    seen
}

fn collect_refs(v: &Value, out: &mut Vec<Digest>) {
    if let Some(r) = parse_ref(v) {
        out.push(r.1);
        return;
    }
    match v {
        Value::Array(a) => a.iter().for_each(|x| collect_refs(x, out)),
        Value::Map(m) => m.iter().for_each(|(_, x)| collect_refs(x, out)),
        Value::Tag(_, inner) => collect_refs(inner, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(n: u8) -> Digest {
        [n; 32]
    }

    fn sample_shardmap() -> ShardMap {
        // 4096×4096 split four ways along rows, as tensor parallelism does.
        let shards = (0..4u64)
            .map(|i| Shard {
                coord: vec![("tp".into(), i)],
                range: vec![(i * 1024, (i + 1) * 1024), (0, 4096)],
                value: Some((otype::TENSOR_DESC, digest(i as u8 + 1))),
            })
            .collect();
        ShardMap {
            world_size: 8,
            mesh: Mesh {
                dims: vec!["dp".into(), "tp".into()],
                shape: vec![2, 4],
            },
            strategy: "megatron".into(),
            placements: vec![(
                "model.layers.0.attn.q_proj.weight".into(),
                Placement {
                    logical_shape: vec![4096, 4096],
                    sharding: vec![Sharding {
                        axis: 0,
                        mesh_dim: "tp".into(),
                        parts: 4,
                    }],
                    shards,
                },
            )],
            flat_params: vec![FlatParam {
                name: "model.layers.0.attn.q_proj.weight".into(),
                offset: 0,
                numel: 4096 * 4096,
                shape: vec![4096, 4096],
            }],
        }
    }

    fn sample_state() -> TrainingState {
        TrainingState {
            framework: vec![
                ("name".into(), Value::text("pytorch")),
                ("version".into(), Value::text("2.9.0")),
            ],
            step: 128_000,
            epoch: Some(2),
            samples_seen: Some(4_194_304_000),
            tokens_seen: Some(8_589_934_592_000),
            wall_clock_s: Some(1_382_400),
            optimizer: Optimizer {
                kind: "adamw".into(),
                hyper: vec![
                    ("lr".into(), Value::F64(3e-4)),
                    (
                        "betas".into(),
                        Value::Array(vec![Value::F64(0.9), Value::F64(0.95)]),
                    ),
                    ("weight_decay".into(), Value::F64(0.1)),
                ],
                schedule: vec![
                    ("kind".into(), Value::text("cosine")),
                    ("warmup".into(), Value::U(2000)),
                    ("total".into(), Value::U(500_000)),
                ],
                states: Some((otype::TENSOR_TABLE, digest(9))),
                master_weights: None,
                state_dtype: Some(DType::F32),
            },
            gradients: None,
            ema: vec![(0.9999, (otype::TENSOR_TABLE, digest(10)))],
            grad_scaler: vec![
                ("kind".into(), Value::text("dynamic")),
                ("scale".into(), Value::F64(65536.0)),
            ],
            rng: vec![
                RngStream {
                    scope: "cuda".into(),
                    implementation: "philox".into(),
                    kind: RngKind::Counter,
                    device: Some(0),
                    worker: None,
                    key: vec![],
                    seed: Some(1234),
                    counter: Some(98304),
                    offset: None,
                    state: None,
                },
                RngStream {
                    scope: "global".into(),
                    implementation: "pytorch-cpu".into(),
                    kind: RngKind::Opaque,
                    device: None,
                    worker: None,
                    key: vec![],
                    seed: None,
                    counter: None,
                    offset: None,
                    state: Some((otype::BLOB, digest(11))),
                },
            ],
            shards: Some((otype::SHARD_MAP, digest(12))),
            dataloader: Some(Dataloader {
                kind: "streaming".into(),
                shard: Some(41),
                offset: Some(9_182_734),
                seed: Some(1234),
                shuffle_buffer: Some(10_000),
                epoch: Some(2),
                consumed_digest: None,
                sample_bitmap: None,
            }),
            loss_history: None,
            config: Some((otype::BLOB, digest(13))),
        }
    }

    #[test]
    fn a_training_state_round_trips() {
        let s = sample_state();
        let bytes = s.to_value().encode();
        let back = TrainingState::from_value(&crate::cbor::decode(&bytes).unwrap()).unwrap();
        // The parameter bags of §09.8 are maps, and canonical CBOR sorts map keys
        // (D3), so the encoding is what round-trips — not the insertion order a
        // framework happened to write them in.
        assert_eq!(back.to_value().encode(), bytes);
        assert_eq!(back.step, s.step);
        assert_eq!(back.tokens_seen, s.tokens_seen);
        assert_eq!(back.rng, s.rng);
        assert_eq!(back.dataloader, s.dataloader);
        assert_eq!(back.shards, s.shards);
        assert_eq!(back.ema, s.ema);
        assert_eq!(back.optimizer.learning_rate(), Some(3e-4));
        assert_eq!(back.optimizer.state_dtype, Some(DType::F32));
    }

    #[test]
    fn a_shard_map_round_trips_and_checks_itself() {
        let m = sample_shardmap();
        let bytes = m.to_value().encode();
        let back = ShardMap::from_value(&crate::cbor::decode(&bytes).unwrap()).unwrap();
        assert_eq!(back, m);
        assert!(m.check().is_empty(), "{:?}", m.check());

        // Shards that do not tile the tensor: the property everything else
        // depends on, so it fails loudly.
        let mut gap = m.clone();
        gap.placements[0].1.shards.pop();
        assert!(gap.check().iter().any(|f| f.starts_with("R-N04")));

        let mut overlap = m.clone();
        overlap.placements[0].1.shards[1].range[0] = (0, 2048);
        assert!(overlap.check().iter().any(|f| f.starts_with("R-N04")));

        // A mesh dimension that does not exist, and a split that disagrees with
        // the mesh extent.
        let mut bad_dim = m.clone();
        bad_dim.placements[0].1.sharding[0].mesh_dim = "ep".into();
        assert!(bad_dim.check().iter().any(|f| f.starts_with("R-N05")));
        let mut bad_parts = m.clone();
        bad_parts.placements[0].1.sharding[0].parts = 3;
        assert!(bad_parts.check().iter().any(|f| f.starts_with("R-N05")));

        // A flat parameter whose shape and element count disagree.
        let mut bad_flat = m.clone();
        bad_flat.flat_params[0].numel = 7;
        assert!(bad_flat.check().iter().any(|f| f.starts_with("R-N06")));

        // A world size that is not the mesh's device count is refused at parse
        // time: it means the checkpoint and its topology disagree.
        let mut v = match m.to_value() {
            Value::Map(pairs) => pairs,
            _ => unreachable!(),
        };
        for (k, val) in v.iter_mut() {
            if k.as_str() == Some("world") {
                *val = Value::map(vec![("size", Value::U(9)), ("mesh", m.mesh.to_value())]);
            }
        }
        assert!(ShardMap::from_value(&Value::Map(v)).is_err());
    }

    #[test]
    fn resharding_a_divisor_friendly_split_moves_no_bytes() {
        // §09.4.2: 4 ranks to 2 is free, because every new boundary is an old
        // boundary. 4 to 8 is not, and the report says which tensors.
        let m = sample_shardmap();
        let target = Mesh::parse("dp=1,tp=2").unwrap();
        let r = reshard(&m, &target).unwrap();
        let new = r.map.unwrap();
        assert_eq!(new.placements[0].1.shards.len(), 2);
        assert_eq!(new.placements[0].1.shards[0].range[0], (0, 2048));
        assert!(new.check().is_empty(), "{:?}", new.check());
        assert_eq!(r.metadata_only, vec!["model.layers.0.attn.q_proj.weight"]);
        assert!(r.needs_copy.is_empty());

        let target = Mesh::parse("tp=8").unwrap();
        let r = reshard(&m, &target).unwrap();
        assert_eq!(r.needs_copy, vec!["model.layers.0.attn.q_proj.weight"]);
        let new = r.map.unwrap();
        assert_eq!(new.placements[0].1.shards.len(), 8);
        assert!(new.check().is_empty(), "{:?}", new.check());

        // A mesh that cannot divide the tensor is refused, with the numbers.
        let target = Mesh::parse("tp=3").unwrap();
        let e = reshard(&m, &target).unwrap_err();
        assert!(e.to_string().contains("4096"), "{e}");
        // As is one that lacks the dimension the tensor is sharded over.
        assert!(reshard(&m, &Mesh::parse("dp=4").unwrap()).is_err());
        // And a malformed mesh spec.
        assert!(Mesh::parse("dp=0").is_err());
        assert!(Mesh::parse("dp").is_err());
        assert!(Mesh::parse("dp=2,dp=2").is_err());
    }

    #[test]
    fn reproducibility_is_reported_rather_than_claimed() {
        let s = sample_state();
        // One counter-based stream and one stateful one: resumable, and not
        // portably so (§09.3).
        assert_eq!(s.non_portable_rng().len(), 1);
        let notes = s.reproducibility();
        assert!(notes.iter().any(|n| n.contains("pytorch-cpu")), "{notes:?}");
        assert!(!notes.iter().any(|n| n.contains("dataloader position")));

        // A state with only counter-based streams and an exact dataloader
        // position has nothing to report.
        let mut clean = s.clone();
        clean.rng.retain(|r| r.portable());
        assert!(
            clean.reproducibility().is_empty(),
            "{:?}",
            clean.reproducibility()
        );

        // No streams at all is worse than a non-portable one, and says so.
        let mut none = s.clone();
        none.rng.clear();
        assert!(none
            .reproducibility()
            .iter()
            .any(|n| n.contains("cannot be replayed")));

        // A dataloader with no position: resumption restarts the stream.
        let mut vague = s.clone();
        vague.dataloader = Some(Dataloader {
            kind: "streaming".into(),
            ..Default::default()
        });
        assert!(vague
            .reproducibility()
            .iter()
            .any(|n| n.contains("restarts")));
    }

    /// A checkpoint container: weights, optimizer moments over them, and a
    /// training state that reaches the moments and nothing else reaches.
    fn checkpoint() -> (Vec<crate::container::Object>, Digest, Vec<Digest>) {
        use crate::container::Object;
        use crate::model::{ModelBuilder, TensorSpec};
        let hash = crate::container::HashAlgo::default();
        let weight: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
        let mut b = ModelBuilder::new("test/checkpoint").chunk_size(1 << 20);
        b = b.tensor(TensorSpec {
            name: "w".into(),
            shape: vec![32, 32],
            dtype: DType::BF16,
            axes: None,
            semantic: "weight",
            data: weight,
        });
        // The moment lives in its own table, reachable only from the training
        // state — which is what makes it strippable.
        let moment: Vec<u8> = (0..4096u32).map(|i| (i % 97) as u8).collect();
        let expr = b.literal(
            &moment,
            DType::F32,
            &[32, 32],
            crate::layout::Layout::default(),
        );
        let desc = crate::tensor::TensorDesc {
            shape: crate::expr::dims(&[32, 32]),
            dtype: DType::F32,
            layout: crate::layout::Layout::default(),
            value: expr,
            semantic: Some("optimizer".into()),
            role: None,
            axes: None,
            device_hint: None,
            materialize: crate::tensor::Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        };
        let desc_obj = Object::structure(otype::TENSOR_DESC, &desc.to_value());
        let desc_d = desc_obj.digest(hash);
        let table = Object::structure(
            otype::TENSOR_TABLE,
            &Value::map(vec![
                ("t", Value::text("omni.tensor/table")),
                ("v", Value::U(1)),
                (
                    "tensors",
                    Value::Map(vec![(
                        Value::text("w.exp_avg"),
                        Value::Array(vec![
                            Value::U(otype::TENSOR_DESC as u64),
                            Value::Bytes(desc_d.to_vec()),
                        ]),
                    )]),
                ),
                ("order", Value::Array(vec![Value::text("w.exp_avg")])),
            ]),
        );
        let states_d = table.digest(hash);
        b.extra_objects.push(desc_obj);
        b.extra_objects.push(table);
        let mut state = sample_state();
        state.optimizer.states = Some((otype::TENSOR_TABLE, states_d));
        state.optimizer.master_weights = None;
        state.ema.clear();
        state.shards = None;
        state.loss_history = None;
        state.config = None;
        state.rng.retain(|r| r.portable());
        let b = b.training(state);
        let (objects, root) = b.build();
        // The weight's data objects, which must survive a strip untouched.
        let weights: Vec<Digest> = objects
            .iter()
            .filter(|o| o.otype == otype::BLOB)
            .map(|o| o.digest(hash))
            .collect();
        (objects, root, weights)
    }

    #[test]
    fn training_state_is_separable_and_the_weights_survive_it() {
        use crate::container::{pack, Container, PackOptions};
        let (objects, root, weights) = checkpoint();
        let bytes = pack(&objects, &root, &PackOptions::default()).unwrap();
        let c = Container::open(bytes).unwrap();

        // Find the training root the way a reader does: Model.training.
        let manifest = c.root().unwrap();
        let model_d = manifest
            .get("assets")
            .and_then(|a| a.get("model"))
            .and_then(parse_ref)
            .unwrap()
            .1;
        let model = c.get_value(&model_d).unwrap();
        let training = model.get("training").and_then(parse_ref).unwrap().1;

        let resolve = |d: &Digest| -> Option<(u16, Vec<u8>)> {
            let e = c.find(d)?;
            Some((e.otype, c.read(d).ok()?))
        };
        let sep = separate(&c.header.root_digest, Some(training), &resolve);

        // R-N02: no inference object may reach the training graph. The
        // `Model.training` ref is the wall, not a violation.
        assert!(sep.violations.is_empty(), "{:?}", sep.violations);
        // The optimizer moment is training-only; every weight chunk is not.
        assert!(!sep.training_only.is_empty());
        for w in &weights {
            // Some blobs belong to the moment, some to the weight; the weight's
            // are reachable without the training state.
            if sep.inference.contains(w) {
                assert!(!sep.training_only.contains(w));
            }
        }
        // R-N01: strip by reachability alone, and every tensor digest the
        // inference model needs is still there.
        let new_model =
            crate::container::Object::structure(otype::MODEL, &without_training(&model));
        assert_ne!(
            new_model.digest(c.header.hash),
            model_d,
            "the model's identity must change"
        );
        let kept: Vec<&Digest> = sep
            .inference
            .iter()
            .filter(|d| c.find(d).is_some_and(|e| e.otype == otype::BLOB))
            .collect();
        assert!(!kept.is_empty());
        for d in kept {
            assert_eq!(
                c.header.hash.digest(&c.read(d).unwrap()),
                *d,
                "a kept weight changed"
            );
        }
        // And the training side is the bigger of the two, as it is in life: f32
        // moments over bf16 weights.
        assert!(
            sep.training_bytes > sep.inference_bytes / 2,
            "{} vs {}",
            sep.training_bytes,
            sep.inference_bytes
        );
    }

    #[test]
    fn rng_kinds_are_derived_from_what_is_stored() {
        // A stream with a state blob is opaque no matter what it is called; one
        // with a counter and no blob is counter-based.
        let v = Value::map(vec![
            ("scope", Value::text("dropout")),
            ("impl", Value::text("counter")),
            ("key", Value::Array(vec![Value::U(1234), Value::U(0)])),
            ("counter", Value::U(8_812_345)),
        ]);
        let r = RngStream::from_value(&v).unwrap();
        assert!(r.portable());
        assert_eq!(r.key, vec![1234, 0]);

        let v = Value::map(vec![
            ("scope", Value::text("dataloader")),
            ("impl", Value::text("numpy-pcg64")),
            ("worker", Value::U(3)),
            (
                "state",
                Value::Array(vec![Value::U(0), Value::Bytes(vec![7; 32])]),
            ),
        ]);
        let r = RngStream::from_value(&v).unwrap();
        assert!(!r.portable());
        assert_eq!(r.worker, Some(3));
    }
}
