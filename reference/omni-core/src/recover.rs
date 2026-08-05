//! Recovery by segment scan (§02.8) — `omni fsck --rebuild`.
//!
//! A container whose trailer, superblock or index is destroyed is not lost.
//! The object payloads are still there, self-framing enough to be found, and
//! every one of them is content-addressed. This module reassembles a container
//! from the payloads alone.
//!
//! ## Why this is safe in a way that recovering other formats is not
//!
//! Recovery normally means guessing, and a guess that looks plausible is worse
//! than a failure. Here it cannot be: an object is accepted only if its bytes
//! hash to a digest something else in the graph already referred to. A
//! mis-assembled object does not become a subtly wrong model — it fails to
//! match anything and is reported missing. §02.8's claim that "recovery cannot
//! silently produce a *wrong* model" is a consequence of content addressing,
//! and this module is where it either holds or does not.
//!
//! ## How the two object kinds are found
//!
//! Structure objects are canonical CBOR, which is self-delimiting: decoding
//! one tells you where the next begins. They are recovered by decoding the
//! `OBJ` segments straight through.
//!
//! Data objects have no framing at all — the index is the only thing that ever
//! knew their lengths. They are recovered the way §02.8 describes: the
//! structure objects name them, `ChunkList` entries give their lengths, and
//! every data object is alignment-aligned (R-C08), so the search space is
//! (aligned offsets × known lengths) and each candidate is confirmed by
//! hashing. This is the part that would be impossible without the alignment
//! rule, which is worth knowing when someone proposes making alignment
//! optional to save padding.

use crate::cbor::{self, Value};
use crate::container::{
    collect_typed_refs, oflags, otype, parse_header, scan_segments, seg, Digest, Error, Header,
    Object,
};
use std::collections::{BTreeMap, BTreeSet};

/// What a recovery pass found.
pub struct Recovery {
    pub header: Header,
    /// Segments located by scanning, independent of the superblock.
    pub segments: Vec<(usize, u16, u64)>,
    /// Structure objects decoded out of `OBJ` segments.
    pub structures: usize,
    /// Data objects located and confirmed by hashing.
    pub blobs: usize,
    /// Referenced objects that could not be found or confirmed. A non-empty
    /// list means the recovered container is incomplete (§01.4) — which is a
    /// legal state to be in, and an honest one to report.
    pub missing: Vec<Digest>,
    /// Non-zero bytes in `BLOB` segments that no recovered object accounts
    /// for. Alignment padding is zero fill (R-C07) and is excluded, so a
    /// non-zero value here means real data that could not be identified.
    pub unaccounted_blob_bytes: u64,
    /// The recovered object set, ready to repack.
    pub objects: Vec<Object>,
    pub root: Digest,
}

impl Recovery {
    pub fn complete(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Rebuilds an object set from a container's payload bytes alone.
///
/// Only the file header is trusted, and only after its CRC checks out: it
/// carries the digest algorithm, the alignment and the root digest, none of
/// which are recoverable from the payloads. Everything else — the trailer, the
/// superblock, the index — is ignored, so a file whose tail is destroyed
/// recovers exactly as well as one whose tail is intact.
pub fn recover(bytes: &[u8]) -> Result<Recovery, Error> {
    let header = parse_header(bytes)?;
    let algo = header.hash;
    let align = 1usize << header.log2_align;
    let segments = scan_segments(bytes)?;

    // --- structure objects: CBOR is self-delimiting ------------------------
    let mut payloads: BTreeMap<Digest, Vec<u8>> = BTreeMap::new();
    let mut refs: Vec<(u16, Digest)> = Vec::new();
    let mut chunk_lens: BTreeMap<Digest, usize> = BTreeMap::new();
    let mut structures = 0usize;

    for &(hdr, kind, plen) in &segments {
        if kind != seg::OBJ {
            continue;
        }
        let start = hdr + crate::container::SEG_HEADER_SIZE;
        let end = start + plen as usize;
        let mut off = start;
        while off < end {
            // Structure objects are 8-byte aligned within the segment, and the
            // gaps are zero fill (R-C07), so a zero byte here means padding
            // rather than an object.
            if bytes[off] == 0 {
                off += 1;
                continue;
            }
            let (v, used) = match cbor::decode_prefix(&bytes[off..end]) {
                Ok(x) => x,
                // A payload that will not decode is damage. Skip a byte and
                // keep looking rather than abandoning the rest of the segment.
                Err(_) => {
                    off += 1;
                    continue;
                }
            };
            let payload = &bytes[off..off + used];
            let d = algo.digest(payload);
            payloads.insert(d, payload.to_vec());
            structures += 1;
            collect_typed_refs(&v, &mut refs);
            collect_chunk_lengths(&v, &mut chunk_lens);
            off += used;
            off = (off + 7) & !7;
        }
    }

    // --- data objects: located by alignment, confirmed by hashing ----------
    let wanted: BTreeSet<Digest> = refs
        .iter()
        .filter(|(t, _)| *t == otype::BLOB)
        .map(|(_, d)| *d)
        .collect();
    let mut lengths: BTreeSet<usize> = chunk_lens.values().copied().collect();
    // A referenced blob with no recorded length still has to be findable, so
    // fall back to every length any chunk list mentioned.
    if lengths.is_empty() {
        lengths.insert(0);
    }

    let mut blobs = 0usize;
    let mut unaccounted = 0u64;

    for &(hdr, kind, plen) in &segments {
        if kind != seg::BLOB {
            continue;
        }
        let start = hdr + crate::container::SEG_HEADER_SIZE;
        let end = start + plen as usize;
        // Alignment padding is zero fill (R-C07) and is not a loss. Track what
        // objects claim so that "unaccounted" can mean actual unexplained
        // data rather than the gaps R-C08 requires.
        let mut claimed = vec![false; plen as usize];
        let mut off = start;
        while off < end {
            let mut hit = None;
            // Prefer the length recorded for a digest we expect at all; then
            // try every other known length. Hashing is the confirmation, so a
            // wrong guess costs time and nothing else.
            for &len in &lengths {
                if len == 0 || off + len > end {
                    continue;
                }
                let d = algo.digest(&bytes[off..off + len]);
                if wanted.contains(&d) && chunk_lens.get(&d).copied().unwrap_or(len) == len {
                    hit = Some((d, len));
                    break;
                }
            }
            match hit {
                Some((d, len)) => {
                    if payloads.insert(d, bytes[off..off + len].to_vec()).is_none() {
                        blobs += 1;
                    }
                    claimed[off - start..off - start + len].fill(true);
                    off += len;
                    off = round_up(off, align);
                }
                // Nothing starts here; the next data object can only be at the
                // next alignment boundary (R-C08).
                None => off = round_up(off + 1, align),
            }
        }
        unaccounted += claimed
            .iter()
            .enumerate()
            .filter(|(i, c)| !**c && bytes[start + i] != 0)
            .count() as u64;
    }

    // --- assemble the graph from the root ---------------------------------
    let mut objects = Vec::new();
    let mut missing = Vec::new();
    let mut seen = BTreeSet::new();
    let mut stack = vec![(otype::MANIFEST, header.root_digest)];
    while let Some((t, d)) = stack.pop() {
        if !seen.insert(d) {
            continue;
        }
        let payload = match payloads.get(&d) {
            Some(p) => p.clone(),
            None => {
                missing.push(d);
                continue;
            }
        };
        if t != otype::BLOB {
            if let Ok(v) = cbor::decode(&payload) {
                let mut r = Vec::new();
                collect_typed_refs(&v, &mut r);
                stack.extend(r);
            }
        }
        objects.push(Object {
            otype: t,
            payload,
            oflags: oflags::CRITICAL | oflags::SAFE_TO_COPY,
            stored: None,
        });
    }
    missing.sort_unstable();

    let root = header.root_digest;
    Ok(Recovery {
        header,
        segments,
        structures,
        blobs,
        missing,
        unaccounted_blob_bytes: unaccounted,
        objects,
        root,
    })
}

fn round_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

/// Pulls `{ "r": [0, h'…'], "n": len }` chunk entries (§04.5) out of a decoded
/// object, which is where data objects' lengths are recorded.
fn collect_chunk_lengths(v: &Value, out: &mut BTreeMap<Digest, usize>) {
    match v {
        Value::Map(m) => {
            let mut digest = None;
            let mut len = None;
            for (k, val) in m {
                match k.as_str() {
                    Some("r") => {
                        let mut r = Vec::new();
                        collect_typed_refs(val, &mut r);
                        if let Some((t, d)) = r.first() {
                            if *t == otype::BLOB {
                                digest = Some(*d);
                            }
                        }
                    }
                    Some("n") => len = val.as_u64(),
                    _ => {}
                }
                collect_chunk_lengths(val, out);
            }
            if let (Some(d), Some(n)) = (digest, len) {
                out.insert(d, n as usize);
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_chunk_lengths(x, out);
            }
        }
        Value::Tag(_, inner) => collect_chunk_lengths(inner, out),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{seg, Container, PackOptions};
    use crate::{pack, verify, DType, HashAlgo, ModelBuilder, TensorSpec};

    fn built(hash: HashAlgo) -> (Vec<u8>, Digest) {
        let (objs, root) = ModelBuilder::new("omni/recover-test")
            .hash(hash)
            .chunk_size(4096)
            .arch("test", vec![])
            .tensor(TensorSpec {
                name: "w".into(),
                shape: vec![128, 64],
                dtype: DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..128 * 64 * 4).map(|i| (i % 251) as u8).collect(),
            })
            .tensor(TensorSpec {
                name: "b".into(),
                shape: vec![64],
                dtype: DType::F32,
                axes: None,
                semantic: "bias",
                data: (0..64 * 4).map(|i| (i % 97) as u8).collect(),
            })
            .build();
        let opts = PackOptions {
            hash,
            ..Default::default()
        };
        (pack(&objs, &root, &opts).unwrap(), root)
    }

    /// Destroy everything a reader normally uses to find objects — trailer,
    /// back superblock, index — and rebuild. The recovered container must be
    /// byte-identical to the original, because packing is deterministic and
    /// the recovered object set is exactly the original one.
    #[test]
    fn a_container_with_no_index_or_trailer_rebuilds_byte_identically() {
        for hash in [HashAlgo::Blake3_256, HashAlgo::Sha256] {
            let (original, root) = built(hash);
            let mut damaged = original.clone();

            // Wipe the index segment, the back superblock and the trailer.
            let c = Container::open(original.clone()).unwrap();
            for (off, kind, plen) in c.segments().unwrap() {
                if kind == seg::INDEX || (kind == seg::SUPER && off > c.bytes.len() / 2) {
                    let end = off + crate::container::SEG_HEADER_SIZE + plen as usize;
                    damaged[off..end].fill(0);
                }
            }
            let n = damaged.len();
            damaged[n - crate::container::TRAILER_SIZE..].fill(0);
            assert!(
                Container::open(damaged.clone()).is_err(),
                "the damaged file must not open normally, or this proves nothing"
            );

            let r = recover(&damaged).unwrap();
            assert!(r.complete(), "missing: {:?}", r.missing.len());
            assert_eq!(r.root, root);
            assert_eq!(r.objects.len(), c.index.len());

            let rebuilt = pack(
                &r.objects,
                &r.root,
                &PackOptions {
                    hash,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(rebuilt, original, "recovery must reproduce the original");
            verify(&Container::open(rebuilt).unwrap()).unwrap();
        }
    }

    /// The tail can be gone entirely rather than zeroed.
    #[test]
    fn a_truncated_tail_still_recovers_what_survives() {
        let (original, root) = built(HashAlgo::default());
        let c = Container::open(original.clone()).unwrap();
        let blob_end = c
            .segments()
            .unwrap()
            .iter()
            .filter(|(_, k, _)| *k == seg::BLOB)
            .map(|(off, _, plen)| off + crate::container::SEG_HEADER_SIZE + *plen as usize)
            .max()
            .unwrap();
        let truncated = original[..blob_end].to_vec();

        let r = recover(&truncated).unwrap();
        assert!(r.complete(), "everything before the index is intact");
        assert_eq!(r.root, root);
    }

    /// The safety property §02.8 claims: a corrupted data object cannot be
    /// mistaken for a good one. It is reported missing, and the recovered
    /// container is honestly incomplete rather than quietly wrong.
    #[test]
    fn corrupted_data_is_reported_missing_not_accepted() {
        let (original, _) = built(HashAlgo::default());
        let c = Container::open(original.clone()).unwrap();
        let blob = c.index.iter().find(|e| e.otype == otype::BLOB).unwrap();
        let mut damaged = original.clone();
        damaged[blob.offset as usize + 7] ^= 0xff;

        let r = recover(&damaged).unwrap();
        assert!(!r.complete(), "a corrupted object must not be accepted");
        assert!(r.missing.contains(&blob.digest));
        assert!(r.unaccounted_blob_bytes > 0);

        // What survived is still usable: the graph packs, and verification
        // reports the loss as a dangling ref rather than as invalidity.
        let rebuilt = pack(&r.objects, &r.root, &PackOptions::default()).unwrap();
        let rc = Container::open(rebuilt).unwrap();
        let report = verify(&rc).unwrap();
        assert_eq!(report.dangling.len(), 1);
    }

    /// Recovery must not depend on the header's own layout fields being
    /// believable beyond what the CRC covers.
    #[test]
    fn a_damaged_header_is_refused_rather_than_guessed() {
        let (mut damaged, _) = built(HashAlgo::default());
        damaged[13] = 31; // absurd log2_align, CRC now wrong
        assert!(recover(&damaged).is_err());
    }
}
