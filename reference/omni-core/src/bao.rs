//! Bao outboard trees — verified streaming and partial reads (§13.3).
//!
//! This is the primitive that makes "start executing before the download
//! finishes" mean something other than "execute unverified bytes". BLAKE3 is
//! already a Merkle tree over 1 KiB chunks; Bao stores that tree's interior
//! nodes *outboard*, separate from the data, so any byte range can be checked
//! against the object's root digest using a proof of about log₂(n) nodes and
//! only the bytes actually received.
//!
//! ## Granularity is the whole tradeoff
//!
//! A tree down to 1 KiB chunks costs 64 bytes per chunk — 6.25 % overhead — and
//! localises corruption to 1 KiB. Pruning the tree so that a subtree of
//! `granularity` bytes is treated as a single leaf drops that to 64 bytes per
//! group: 0.098 % at 64 KiB, at the cost of having to fetch and hash a whole
//! 64 KiB group to verify one byte in it.
//!
//! | Granularity | Overhead | Bytes to verify one byte |
//! |---|---|---|
//! | 1 KiB | 6.25 % | 1 KiB |
//! | 16 KiB | 0.39 % | 16 KiB |
//! | 64 KiB | 0.098 % | 64 KiB |
//! | 1 MiB | 0.006 % | 1 MiB |
//!
//! 64 KiB is the default: it is a small multiple of a page, it keeps the tree
//! for a 140 GB model down to ~140 MB, and it matches the granularity at which
//! HTTP range requests and NVMe reads are efficient anyway.
//!
//! Pruning does not change the root: it is the same BLAKE3 tree, with the
//! bottom levels recomputed from data instead of stored. A tree built at one
//! granularity verifies against a root computed at any other.
//!
//! ## Why the tree is a cache object
//!
//! It is recomputable from the data it describes, so §11's `CACHEABLE` bit
//! applies: a reader may always drop it and a writer may always omit it. What
//! it buys is the ability to verify *before* having the whole object, which is
//! not something the flat object digest can offer at any price.

use crate::blake3::{chunk_chaining_value, chunk_root, parent_chaining_value, parent_root};
use crate::cbor::Value;
use crate::container::{otype, Digest};

/// BLAKE3's leaf size. Not configurable — it is part of the hash.
const CHUNK: u64 = 1024;

/// Default verification granularity (§13.3).
pub const DEFAULT_GRANULARITY: u32 = 64 * 1024;

/// Bytes per stored interior node: two 32-byte chaining values.
pub const NODE_LEN: usize = 64;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Granularity must be a power of two of at least one chunk.
    BadGranularity(u32),
    /// The requested range is not aligned to the verification granularity, so
    /// it cannot be checked without data the caller has not supplied.
    Unaligned(String),
    /// The range extends past the end of the object.
    OutOfRange(String),
    /// The outboard tree is the wrong size for the object it claims to cover.
    MalformedTree(String),
    /// A node or leaf did not hash to what its parent said it would. This is
    /// the interesting one: it means the data is corrupt or forged.
    Mismatch { chunk_start: u64, chunks: u64 },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadGranularity(g) => {
                write!(f, "granularity {g} must be a power of two ≥ 1024")
            }
            Error::Unaligned(m) => write!(f, "unaligned range: {m}"),
            Error::OutOfRange(m) => write!(f, "out of range: {m}"),
            Error::MalformedTree(m) => write!(f, "malformed outboard tree: {m}"),
            Error::Mismatch {
                chunk_start,
                chunks,
            } => write!(
                f,
                "verification failed for chunks {chunk_start}..{}",
                chunk_start + chunks
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Number of BLAKE3 chunks in an object. An empty object is one empty chunk.
fn total_chunks(size: u64) -> u64 {
    if size == 0 {
        1
    } else {
        size.div_ceil(CHUNK)
    }
}

/// BLAKE3's split rule: the left subtree of an `n`-chunk node covers the
/// largest power of two strictly less than `n`. This is what makes the tree
/// left-complete, so appending data never reshapes the existing left side.
fn left_chunks(n: u64) -> u64 {
    debug_assert!(n > 1);
    n.next_power_of_two() / 2
}

/// Interior nodes in a subtree of `n` chunks pruned to `g`-chunk leaves. Any
/// binary tree with k leaves has k−1 interior nodes.
fn interior_nodes(n: u64, g: u64) -> usize {
    (n.div_ceil(g) - 1) as usize
}

/// An outboard BLAKE3 tree: interior nodes only, in pre-order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaoTree {
    /// Size in bytes of the object this tree covers.
    pub size: u64,
    /// Smallest independently verifiable span, in bytes.
    pub granularity: u32,
    /// Interior nodes, pre-order (parent before left subtree before right).
    /// Pre-order is what lets a verifier skip a subtree it does not need by
    /// advancing a cursor, without an index.
    nodes: Vec<[u8; NODE_LEN]>,
}

impl BaoTree {
    /// Builds the outboard tree for `data` and returns it with the root hash.
    ///
    /// The root is BLAKE3 of the same bytes — the tree is derived, never
    /// authoritative.
    pub fn encode(data: &[u8], granularity: u32) -> Result<(Digest, BaoTree), Error> {
        let g = granularity as u64;
        if g < CHUNK || !g.is_power_of_two() {
            return Err(Error::BadGranularity(granularity));
        }
        let size = data.len() as u64;
        let n = total_chunks(size);
        let mut nodes = Vec::with_capacity(interior_nodes(n, g / CHUNK));
        let root = build(data, 0, n, g / CHUNK, &mut nodes, true);
        Ok((
            root,
            BaoTree {
                size,
                granularity,
                nodes,
            },
        ))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Serialised size of the tree, for the overhead calculations above.
    pub fn byte_len(&self) -> usize {
        self.nodes.len() * NODE_LEN
    }

    /// The tree as a data object payload: nodes concatenated, pre-order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        for n in &self.nodes {
            out.extend_from_slice(n);
        }
        out
    }

    /// Parses a tree blob. `size` and `granularity` come from the descriptor,
    /// so the expected node count is known before reading and a blob of the
    /// wrong length is rejected rather than interpreted.
    pub fn from_bytes(size: u64, granularity: u32, bytes: &[u8]) -> Result<BaoTree, Error> {
        let g = granularity as u64;
        if g < CHUNK || !g.is_power_of_two() {
            return Err(Error::BadGranularity(granularity));
        }
        let want = interior_nodes(total_chunks(size), g / CHUNK);
        if bytes.len() != want * NODE_LEN {
            return Err(Error::MalformedTree(format!(
                "{} bytes, expected {} ({} nodes)",
                bytes.len(),
                want * NODE_LEN,
                want
            )));
        }
        let nodes = bytes
            .chunks_exact(NODE_LEN)
            .map(|c| {
                let mut n = [0u8; NODE_LEN];
                n.copy_from_slice(c);
                n
            })
            .collect();
        Ok(BaoTree {
            size,
            granularity,
            nodes,
        })
    }

    /// The §13.3 descriptor for this tree.
    ///
    /// `target` is the object the tree verifies; `tree_blob` is the digest of
    /// [`to_bytes`](Self::to_bytes) stored as a data object.
    pub fn descriptor(&self, target: &Digest, tree_blob: &Digest) -> Value {
        Value::map(vec![
            ("t", Value::text("omni.stream/bao")),
            ("v", Value::U(1)),
            ("target", Value::Bytes(target.to_vec())),
            ("granularity", Value::U(self.granularity as u64)),
            ("size", Value::U(self.size)),
            (
                "tree",
                Value::Array(vec![
                    Value::U(otype::BLOB as u64),
                    Value::Bytes(tree_blob.to_vec()),
                ]),
            ),
        ])
    }

    /// Verifies that `data` really is the bytes of the object with digest
    /// `root` at byte `offset`.
    ///
    /// `offset` must be granularity-aligned, and `data` must end on a
    /// granularity boundary or at the end of the object — otherwise the final
    /// group cannot be hashed and there is nothing honest to report.
    ///
    /// Only the nodes on the path to the requested range are touched; the rest
    /// of the tree is skipped. See [`proof_nodes`](Self::proof_nodes).
    pub fn verify_range(&self, root: &Digest, offset: u64, data: &[u8]) -> Result<(), Error> {
        let g = self.granularity as u64;
        let end = offset
            .checked_add(data.len() as u64)
            .ok_or_else(|| Error::OutOfRange("offset + len overflows".into()))?;
        if end > self.size {
            return Err(Error::OutOfRange(format!(
                "{offset}..{end} exceeds object size {}",
                self.size
            )));
        }
        if !offset.is_multiple_of(g) {
            return Err(Error::Unaligned(format!(
                "offset {offset} is not a multiple of granularity {g}"
            )));
        }
        if !end.is_multiple_of(g) && end != self.size {
            return Err(Error::Unaligned(format!(
                "range ends at {end}, neither a multiple of {g} nor the end of the object"
            )));
        }

        let n = total_chunks(self.size);
        let gc = g / CHUNK;
        if self.nodes.len() != interior_nodes(n, gc) {
            return Err(Error::MalformedTree(format!(
                "{} nodes for {} chunks at granularity {}",
                self.nodes.len(),
                n,
                self.granularity
            )));
        }
        // Half-open chunk range covering the request. An empty request verifies
        // nothing and succeeds, which is the honest answer.
        if data.is_empty() {
            return Ok(());
        }
        let want = (offset / CHUNK)..end.div_ceil(CHUNK);

        let mut cursor = 0usize;
        self.walk(root, true, 0, n, gc, &want, offset, data, &mut cursor)
    }

    /// Number of interior nodes a verifier must consult for this range — the
    /// proof size. Logarithmic in the object size for a contiguous range.
    pub fn proof_nodes(&self, offset: u64, len: u64) -> usize {
        let g = self.granularity as u64;
        if len == 0 || self.size == 0 {
            return 0;
        }
        let end = (offset + len).min(self.size);
        let want = (offset / CHUNK)..end.div_ceil(CHUNK);
        count_path(0, total_chunks(self.size), g / CHUNK, &want)
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        expect: &Digest,
        is_root: bool,
        start: u64,
        n: u64,
        gc: u64,
        want: &std::ops::Range<u64>,
        offset: u64,
        data: &[u8],
        cursor: &mut usize,
    ) -> Result<(), Error> {
        if n <= gc {
            // Leaf group: recompute it from the caller's bytes. This is the
            // only place data is hashed, and it is why a wrong byte anywhere in
            // the group fails rather than passing silently.
            let from = start * CHUNK;
            let to = ((start + n) * CHUNK).min(self.size);
            let lo = (from - offset) as usize;
            let hi = (to - offset) as usize;
            let got = group_cv(&data[lo..hi], start, n, is_root);
            return if got == *expect {
                Ok(())
            } else {
                Err(Error::Mismatch {
                    chunk_start: start,
                    chunks: n,
                })
            };
        }

        let node = self
            .nodes
            .get(*cursor)
            .ok_or_else(|| Error::MalformedTree("ran out of nodes".into()))?;
        *cursor += 1;
        let mut l = [0u8; 32];
        let mut r = [0u8; 32];
        l.copy_from_slice(&node[..32]);
        r.copy_from_slice(&node[32..]);

        let got = if is_root {
            parent_root(&l, &r)
        } else {
            parent_chaining_value(&l, &r)
        };
        if got != *expect {
            return Err(Error::Mismatch {
                chunk_start: start,
                chunks: n,
            });
        }

        let lc = left_chunks(n);
        let rc = n - lc;

        if overlaps(start, lc, want) {
            self.walk(&l, false, start, lc, gc, want, offset, data, cursor)?;
        } else {
            *cursor += interior_nodes(lc, gc);
        }
        if overlaps(start + lc, rc, want) {
            self.walk(&r, false, start + lc, rc, gc, want, offset, data, cursor)?;
        } else {
            *cursor += interior_nodes(rc, gc);
        }
        Ok(())
    }
}

fn overlaps(start: u64, n: u64, want: &std::ops::Range<u64>) -> bool {
    start < want.end && want.start < start + n
}

fn count_path(start: u64, n: u64, gc: u64, want: &std::ops::Range<u64>) -> usize {
    if n <= gc || !overlaps(start, n, want) {
        return 0;
    }
    let lc = left_chunks(n);
    1 + count_path(start, lc, gc, want) + count_path(start + lc, n - lc, gc, want)
}

/// Builds the tree, emitting interior nodes in pre-order, and returns the
/// chaining value (or root hash) of the subtree.
fn build(
    data: &[u8],
    start: u64,
    n: u64,
    gc: u64,
    nodes: &mut Vec<[u8; NODE_LEN]>,
    is_root: bool,
) -> Digest {
    if n <= gc {
        let from = (start * CHUNK) as usize;
        let to = (((start + n) * CHUNK) as usize).min(data.len());
        return group_cv(&data[from..to], start, n, is_root);
    }
    // Reserve this node's slot before recursing so children land after it.
    let slot = nodes.len();
    nodes.push([0u8; NODE_LEN]);

    let lc = left_chunks(n);
    let l = build(data, start, lc, gc, nodes, false);
    let r = build(data, start + lc, n - lc, gc, nodes, false);

    nodes[slot][..32].copy_from_slice(&l);
    nodes[slot][32..].copy_from_slice(&r);

    if is_root {
        parent_root(&l, &r)
    } else {
        parent_chaining_value(&l, &r)
    }
}

/// Chaining value of a pruned leaf: the BLAKE3 subtree over `n` chunks
/// beginning at chunk `start`, computed from data rather than stored.
fn group_cv(data: &[u8], start: u64, n: u64, is_root: bool) -> Digest {
    if n == 1 {
        return if is_root {
            chunk_root(data, start)
        } else {
            chunk_chaining_value(data, start)
        };
    }
    let lc = left_chunks(n);
    let split = (lc * CHUNK) as usize;
    let l = group_cv(&data[..split], start, lc, false);
    let r = group_cv(&data[split..], start + lc, n - lc, false);
    if is_root {
        parent_root(&l, &r)
    } else {
        parent_chaining_value(&l, &r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blake3::blake3;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    const SIZES: &[usize] = &[
        0, 1, 1023, 1024, 1025, 2048, 4096, 65536, 65537, 100_000, 262_144, 300_000,
    ];
    const GRANS: &[u32] = &[1024, 4096, 65536];

    /// The tree is derived, so its root must be the object's ordinary digest.
    /// If this ever diverges, every verified read is checking the wrong thing.
    #[test]
    fn root_is_the_object_digest() {
        for &size in SIZES {
            let d = data(size);
            for &g in GRANS {
                let (root, _) = BaoTree::encode(&d, g).unwrap();
                assert_eq!(root, blake3(&d), "size {size}, granularity {g}");
            }
        }
    }

    /// Every granularity describes the same tree, so a reader that wants finer
    /// granularity than the publisher chose can rebuild rather than refetch.
    #[test]
    fn granularity_does_not_change_the_root() {
        let d = data(300_000);
        let roots: Vec<Digest> = GRANS
            .iter()
            .map(|&g| BaoTree::encode(&d, g).unwrap().0)
            .collect();
        assert!(roots.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn every_group_verifies_on_its_own() {
        for &size in SIZES {
            let d = data(size);
            for &g in GRANS {
                let (root, tree) = BaoTree::encode(&d, g).unwrap();
                let mut off = 0u64;
                while off < size as u64 {
                    let end = ((off + g as u64) as usize).min(size);
                    tree.verify_range(&root, off, &d[off as usize..end])
                        .unwrap_or_else(|e| {
                            panic!("size {size}, granularity {g}, offset {off}: {e}")
                        });
                    off += g as u64;
                }
            }
        }
    }

    #[test]
    fn multi_group_and_whole_object_ranges_verify() {
        let size = 300_000;
        let d = data(size);
        let g = 65536u32;
        let (root, tree) = BaoTree::encode(&d, g).unwrap();
        tree.verify_range(&root, 0, &d).unwrap();
        tree.verify_range(&root, 0, &d[..2 * g as usize]).unwrap();
        tree.verify_range(&root, g as u64, &d[g as usize..3 * g as usize])
            .unwrap();
        // The tail group is short; ending at the object's end is legal.
        tree.verify_range(&root, 4 * g as u64, &d[4 * g as usize..])
            .unwrap();
    }

    /// The point of the whole exercise: a flipped bit in a range you fetched is
    /// caught, using only that range.
    #[test]
    fn corruption_is_caught_within_the_fetched_range() {
        let size = 300_000;
        let d = data(size);
        let g = 65536u32;
        let (root, tree) = BaoTree::encode(&d, g).unwrap();

        for &pos in &[0usize, 1, 65535, 65536, 200_000, size - 1] {
            let mut bad = d.clone();
            bad[pos] ^= 1;
            let group = (pos / g as usize) * g as usize;
            let end = (group + g as usize).min(size);
            let err = tree
                .verify_range(&root, group as u64, &bad[group..end])
                .expect_err("a flipped bit must not verify");
            assert!(matches!(err, Error::Mismatch { .. }), "got {err:?}");

            // Corruption is localised: the other groups still verify.
            let other = if group == 0 { g as usize } else { 0 };
            let oend = (other + g as usize).min(size);
            tree.verify_range(&root, other as u64, &bad[other..oend])
                .expect("an untouched group must still verify");
        }
    }

    /// A forged tree cannot make forged data pass: the interior nodes are
    /// themselves checked against the root on the way down.
    #[test]
    fn a_tampered_tree_is_rejected() {
        let d = data(300_000);
        let (root, tree) = BaoTree::encode(&d, 65536).unwrap();
        let mut bytes = tree.to_bytes();
        bytes[0] ^= 1;
        let forged = BaoTree::from_bytes(tree.size, tree.granularity, &bytes).unwrap();
        let err = forged
            .verify_range(&root, 0, &d[..65536])
            .expect_err("a tampered node must not verify");
        assert!(matches!(err, Error::Mismatch { .. }), "got {err:?}");
    }

    /// A peer that serves the wrong range — genuine bytes from elsewhere in the
    /// same object — must be caught. Without this, a CDN could satisfy a range
    /// request with any block it happened to have.
    #[test]
    fn misdelivered_ranges_are_rejected() {
        let g = 65536usize;
        let d = data(4 * g);
        let (root, tree) = BaoTree::encode(&d, g as u32).unwrap();

        tree.verify_range(&root, 0, &d[..g]).unwrap();
        // Group 0's real bytes, offered as group 1.
        assert!(matches!(
            tree.verify_range(&root, g as u64, &d[..g]),
            Err(Error::Mismatch { .. })
        ));
    }

    /// Position is bound into the hash, so even two groups with identical
    /// contents get distinct chaining values. This is what makes the previous
    /// test's guarantee hold for objects with repeated blocks — a zero-filled
    /// region, say — rather than only for objects whose blocks happen to
    /// differ.
    #[test]
    fn identical_groups_still_get_distinct_chaining_values() {
        let g = 65536usize;
        let mut d = data(2 * g);
        let (first, second) = d.split_at_mut(g);
        second.copy_from_slice(first);
        let (_, tree) = BaoTree::encode(&d, g as u32).unwrap();

        assert_eq!(tree.node_count(), 1);
        let node = tree.to_bytes();
        assert_eq!(&d[..g], &d[g..], "the two groups really are identical");
        assert_ne!(
            &node[..32],
            &node[32..],
            "but their chaining values are not"
        );
    }

    /// Ranges that cannot be checked are refused rather than half-checked.
    #[test]
    fn unverifiable_ranges_are_refused_not_guessed() {
        let d = data(300_000);
        let (root, tree) = BaoTree::encode(&d, 65536).unwrap();
        assert!(matches!(
            tree.verify_range(&root, 100, &d[100..65636]),
            Err(Error::Unaligned(_))
        ));
        assert!(matches!(
            tree.verify_range(&root, 0, &d[..1000]),
            Err(Error::Unaligned(_))
        ));
        assert!(matches!(
            tree.verify_range(&root, 262_144, &d[..65536]),
            Err(Error::OutOfRange(_))
        ));
        assert!(matches!(
            BaoTree::encode(&d, 3000),
            Err(Error::BadGranularity(_))
        ));
        assert!(matches!(
            BaoTree::encode(&d, 512),
            Err(Error::BadGranularity(_))
        ));
    }

    #[test]
    fn serialisation_round_trips_and_rejects_wrong_lengths() {
        let d = data(300_000);
        let (_, tree) = BaoTree::encode(&d, 65536).unwrap();
        let bytes = tree.to_bytes();
        assert_eq!(bytes.len(), tree.byte_len());
        assert_eq!(
            BaoTree::from_bytes(tree.size, tree.granularity, &bytes).unwrap(),
            tree
        );
        assert!(matches!(
            BaoTree::from_bytes(tree.size, tree.granularity, &bytes[..bytes.len() - 1]),
            Err(Error::MalformedTree(_))
        ));
    }

    /// The overhead figures quoted in §13.3 and in this module's documentation
    /// are arithmetic, so they can be asserted rather than believed.
    #[test]
    fn overhead_matches_the_documented_figures() {
        let size = 1 << 20; // 1 MiB
        let d = data(size);
        for &(g, pct) in &[(1024u32, 6.25f64), (16384, 0.39), (65536, 0.098)] {
            let (_, tree) = BaoTree::encode(&d, g).unwrap();
            let groups = (size as u64).div_ceil(g as u64);
            assert_eq!(tree.node_count() as u64, groups - 1);
            let got = 100.0 * tree.byte_len() as f64 / size as f64;
            assert!(
                (got - pct).abs() < 0.02,
                "granularity {g}: {got:.3}% vs {pct}%"
            );
        }
    }

    /// Proof size is what makes verified streaming affordable: verifying one
    /// 64 KiB group of a 1 GiB object must not require walking the tree.
    #[test]
    fn proof_size_is_logarithmic() {
        let size = 1u64 << 24; // 16 MiB, 256 groups at 64 KiB
        let d = data(size as usize);
        let g = 65536u32;
        let (root, tree) = BaoTree::encode(&d, g).unwrap();
        let groups = size / g as u64;
        assert_eq!(tree.node_count() as u64, groups - 1);

        // A single group needs ~log2(groups) nodes, not groups-1 of them.
        let depth = (groups as f64).log2().ceil() as usize;
        for off in [0u64, g as u64, size / 2, size - g as u64] {
            let n = tree.proof_nodes(off, g as u64);
            assert!(
                n <= depth + 1,
                "offset {off}: {n} nodes, expected at most {}",
                depth + 1
            );
            tree.verify_range(&root, off, &d[off as usize..off as usize + g as usize])
                .unwrap();
        }
        assert_eq!(tree.proof_nodes(0, size), tree.node_count());
    }

    #[test]
    fn descriptor_matches_the_specified_shape() {
        let d = data(4096);
        let (root, tree) = BaoTree::encode(&d, 1024).unwrap();
        let blob = blake3(&tree.to_bytes());
        let v = tree.descriptor(&root, &blob);
        assert_eq!(v.get("t").unwrap().as_str(), Some("omni.stream/bao"));
        assert_eq!(v.get("granularity").unwrap().as_u64(), Some(1024));
        assert_eq!(v.get("size").unwrap().as_u64(), Some(4096));
        assert_eq!(v.get("target").unwrap().as_bytes(), Some(&root[..]));
        // Canonical encoding must round-trip, since this object gets hashed.
        let e = v.encode();
        assert_eq!(crate::cbor::decode(&e).unwrap().encode(), e);
    }
}
