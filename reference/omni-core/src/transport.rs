//! §13 — streaming, transport and distribution.
//!
//! "The unit of transfer is the **object**, not the file. Everything in this
//! section follows from that." What is here follows from it too:
//!
//! * **The `.omni.idx` sidecar** (§13.4.1). Opening a remote container costs
//!   three round trips — the header, the superblock it points at, the index —
//!   and four if there is no front superblock, because then the superblock's
//!   extent is only in the trailer. The sidecar collapses all of them into one
//!   immutable object: header, superblock and index, with framing of its own so
//!   a truncated one is detected rather than mis-parsed.
//! * **Index-only containers** (§13.8). A few megabytes that fully describe a
//!   700 GB model: every object in the index with its digest, type and size,
//!   the data ones marked `EXTERNAL` and their bytes left behind. Inspectable,
//!   verifiable against a signature, plannable — and a store that answers
//!   absence for the weights, which the object model already handles because
//!   §01.4 makes a partial graph incomplete rather than invalid.
//! * **An HTTP range store** (§13.4.2), speaking HTTP/1.1 over a TCP socket with
//!   range coalescing and resumption. Plain HTTP only: TLS needs a
//!   cryptographic transport stack, this crate has zero dependencies, and
//!   implementing TLS 1.3 from scratch to fetch a model would be a worse idea
//!   than saying so. An `https://` URL is refused with that reason rather than
//!   silently downgraded.
//!
//! What is not here: the OCI mapping of §13.5, `omni mount` (§13.9) and
//! `omni serve`. Each needs something outside this crate's reach — a JSON
//! codec and a registry client, FUSE, a server — and none of them is pretended
//! to exist.

use crate::cbor::Value;
use crate::container::{oflags, Container, Digest, HashAlgo, IndexEntry};
use crate::store::{Error as StoreError, Store};

/// `.omni.idx` magic: the container magic with the last byte changed, so a
/// sidecar handed to a container reader is rejected by the reader rather than
/// half-parsed.
pub const IDX_MAGIC: [u8; 8] = [0x89, b'O', b'M', b'N', b'I', b'X', 0x0d, 0x0a];
pub const IDX_VERSION: u16 = 1;

/// The sidecar header is 128 bytes with its CRC in the last four, exactly like
/// the container header of §02.3. Not for elegance: a reader that already knows
/// where a container keeps its CRC does not have to learn a second convention.
const IDX_HEADER: usize = 128;
/// Where the three (offset, length) pairs start.
const IDX_PARTS: usize = 24;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    Io(String),
    /// A transport this build does not speak.
    Unsupported(String),
    /// The server said something other than what was asked for.
    Protocol(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "malformed sidecar: {m}"),
            Error::Io(m) => write!(f, "transport i/o: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported transport: {m}"),
            Error::Protocol(m) => write!(f, "transport: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// ------------------------------------------------------------------- sidecar --

/// Builds the `.omni.idx` sidecar of §13.4.1 for a container.
///
/// Layout — a 128-byte header, then three parts:
///
/// | Bytes | Field |
/// |---|---|
/// | 0..8 | magic |
/// | 8..10 | version |
/// | 10 | hash algorithm code |
/// | 11..16 | reserved, zero |
/// | 16..24 | the described container's `file_size` |
/// | 24..72 | three (offset, length) pairs: header, superblock, index |
/// | 72..104 | superblock digest |
/// | 104..124 | reserved, zero |
/// | 124..128 | CRC-32C of bytes 0..124 |
///
/// The parts are the container's file header verbatim, its superblock, and its
/// object index with the index segment header in front — the same bytes a reader
/// would have fetched from the file, so the parsing code is shared rather than
/// re-derived. Everything needed to plan a load, in one immutable, infinitely
/// cacheable object.
pub fn sidecar(c: &Container) -> Vec<u8> {
    let header = &c.bytes[..c.header.header_size as usize];
    let sb = c.superblock.encode();
    let (index_off, index_len) = match index_extent(&c.superblock) {
        Ok((o, l)) => (o as usize, l as usize),
        // A container without an index does not parse in the first place, so
        // this is not a case that can arrive from a file.
        Err(_) => (64, 0),
    };
    let index = &c.bytes[index_off - 64..index_off + index_len];

    let mut out = vec![0u8; IDX_HEADER];
    out[0..8].copy_from_slice(&IDX_MAGIC);
    out[8..10].copy_from_slice(&IDX_VERSION.to_le_bytes());
    out[10] = c.header.hash.code();
    // The container's own size, so a reader can tell that the sidecar belongs to
    // the file it is about to range-read before it reads any of it.
    out[16..24].copy_from_slice(&(c.bytes.len() as u64).to_le_bytes());
    let mut at = IDX_HEADER;
    for (i, part) in [header, &sb[..], index].into_iter().enumerate() {
        let base = IDX_PARTS + i * 16;
        out[base..base + 8].copy_from_slice(&(at as u64).to_le_bytes());
        out[base + 8..base + 16].copy_from_slice(&(part.len() as u64).to_le_bytes());
        at += part.len();
    }
    out[72..104].copy_from_slice(&c.header.hash.digest(&sb));
    out.extend_from_slice(header);
    out.extend_from_slice(&sb);
    out.extend_from_slice(index);
    let crc = crate::crc32c::crc32c(&out[..IDX_HEADER - 4]);
    out[IDX_HEADER - 4..IDX_HEADER].copy_from_slice(&crc.to_le_bytes());
    out
}

/// A container's framing, read from a sidecar instead of from the container.
#[derive(Debug)]
pub struct Sidecar {
    pub hash: HashAlgo,
    pub superblock: Value,
    pub index: Vec<IndexEntry>,
    pub root: Digest,
    /// The size of the container this sidecar describes.
    pub file_size: u64,
}

impl Sidecar {
    pub fn parse(d: &[u8]) -> Res<Sidecar> {
        if d.len() < IDX_HEADER || d[0..8] != IDX_MAGIC {
            return Err(Error::Malformed("bad magic".into()));
        }
        let version = u16::from_le_bytes([d[8], d[9]]);
        if version != IDX_VERSION {
            return Err(Error::Malformed(format!("version {version} is not 1")));
        }
        let want = u32::from_le_bytes([
            d[IDX_HEADER - 4],
            d[IDX_HEADER - 3],
            d[IDX_HEADER - 2],
            d[IDX_HEADER - 1],
        ]);
        if crate::crc32c::crc32c(&d[..IDX_HEADER - 4]) != want {
            return Err(Error::Malformed("R-X01: header CRC mismatch".into()));
        }
        let hash = HashAlgo::from_code(d[10]).ok_or_else(|| {
            Error::Malformed(format!(
                "hash algorithm {:#04x} is not one of the two §03.5.1 requires",
                d[10]
            ))
        })?;
        let file_size = u64::from_le_bytes(d[16..24].try_into().unwrap());
        let mut parts = Vec::new();
        for i in 0..3 {
            let base = IDX_PARTS + i * 16;
            let off = u64::from_le_bytes(d[base..base + 8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(d[base + 8..base + 16].try_into().unwrap()) as usize;
            let part = off
                .checked_add(len)
                .and_then(|end| d.get(off..end))
                .ok_or_else(|| Error::Malformed(format!("part {i} is out of range")))?;
            parts.push(part);
        }
        // R-X02: the header this sidecar carries has to declare the same size the
        // sidecar does. `parse_header_bytes` is given the sidecar's figure and
        // rejects a header that disagrees, so the two cannot drift.
        let header = crate::container::parse_header_bytes(parts[0], file_size)
            .map_err(|e| Error::Malformed(format!("R-X02: {e}")))?;
        if header.hash != hash {
            return Err(Error::Malformed(
                "R-X01: the sidecar and the header it carries disagree about the hash".into(),
            ));
        }
        // The superblock is checked against the digest the sidecar carries, and
        // that digest is under the header CRC. A sidecar with a rewritten
        // superblock therefore fails here rather than being believed.
        if hash.digest(parts[1]) != d[72..104] {
            return Err(Error::Malformed("R-X01: superblock digest mismatch".into()));
        }
        let superblock =
            crate::cbor::decode(parts[1]).map_err(|e| Error::Malformed(e.to_string()))?;
        let (_, index_len) = index_extent(&superblock).map_err(|e| {
            Error::Malformed(match e {
                Error::Protocol(m) => m,
                other => other.to_string(),
            })
        })?;
        let index = crate::container::parse_index_bytes(parts[2], 64, index_len as usize)
            .map_err(|e| Error::Malformed(e.to_string()))?;
        Ok(Sidecar {
            hash,
            superblock,
            index,
            root: header.root_digest,
            file_size,
        })
    }

    /// Total logical bytes the described container holds.
    pub fn logical_bytes(&self) -> u64 {
        self.index.iter().map(|e| e.logical_len).sum()
    }
}

// ------------------------------------------------------ index-only container --

/// Turns a container into the index-only form of §13.8: every object described,
/// the weights left behind.
///
/// The result is a *store that answers `NotFound`*, which the object model
/// already handles — §01.4 makes a partial graph incomplete rather than invalid,
/// and §13.8 says nothing about this is exceptional. What it buys is a catalogue:
/// a few megabytes that can be inspected, verified against a signature and
/// planned against a runtime, with the weights fetched later from wherever they
/// live.
/// Structure objects stay — they are what makes the file useful, and together
/// they are kilobytes. The data objects become descriptions: digest, type and
/// logical length in the index, `EXTERNAL` set, no bytes. Nothing is rewritten
/// in place; the container is re-packed, because a file that still *contains* the
/// weights it claims not to have would be an index-only container in name only.
pub fn index_only(c: &Container) -> Res<Vec<u8>> {
    let mut objects = Vec::new();
    let mut absent = Vec::new();
    for e in &c.index {
        if e.otype == crate::container::otype::BLOB || e.oflags & oflags::EXTERNAL != 0 {
            absent.push(e.clone());
            continue;
        }
        let payload = c
            .read(&e.digest)
            .map_err(|err| Error::Malformed(err.to_string()))?;
        objects.push(crate::container::Object {
            otype: e.otype,
            payload,
            oflags: e.oflags,
            stored: None,
        });
    }
    let opts = crate::container::PackOptions {
        log2_align: c.header.log2_align,
        creator: c.header.creator.clone(),
        reproducible: true,
        hash: c.header.hash,
        codec: crate::codec::Codec::Raw,
    };
    crate::container::pack_partial(&objects, &absent, &c.header.root_digest, &opts)
        .map_err(|err| Error::Malformed(err.to_string()))
}

/// How complete a container is (§13.8's `complete: no (12 %, …)`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Completeness {
    pub described: usize,
    pub local: usize,
    pub described_bytes: u64,
    pub local_bytes: u64,
}

impl Completeness {
    pub fn of(c: &Container) -> Completeness {
        let mut out = Completeness::default();
        for e in &c.index {
            out.described += 1;
            out.described_bytes += e.logical_len;
            if e.oflags & oflags::EXTERNAL == 0 {
                out.local += 1;
                out.local_bytes += e.logical_len;
            }
        }
        out
    }

    pub fn is_complete(&self) -> bool {
        self.described == self.local
    }

    pub fn percent(&self) -> f64 {
        if self.described_bytes == 0 {
            return 100.0;
        }
        100.0 * self.local_bytes as f64 / self.described_bytes as f64
    }
}

// ---------------------------------------------------------------- HTTP store --

/// A parsed `http://host[:port]/path` URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Url {
    pub host: String,
    pub port: u16,
    pub path: String,
}

impl Url {
    pub fn parse(s: &str) -> Res<Url> {
        if let Some(rest) = s.strip_prefix("https://") {
            let _ = rest;
            return Err(Error::Unsupported(
                "https needs a TLS stack, and this crate has no dependencies to \
                 provide one; fetch over http, or hand it a file"
                    .into(),
            ));
        }
        let rest = s
            .strip_prefix("http://")
            .ok_or_else(|| Error::Unsupported(format!("`{s}` is not an http:// URL")))?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>()
                    .map_err(|_| Error::Unsupported(format!("`{p}` is not a port")))?,
            ),
            None => (authority.to_string(), 80),
        };
        if host.is_empty() {
            return Err(Error::Unsupported("no host".into()));
        }
        Ok(Url {
            host,
            port,
            path: path.to_string(),
        })
    }
}

/// One range to fetch, in file coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub off: u64,
    pub len: u64,
}

/// Merges ranges that are adjacent or within `gap` bytes of each other
/// (§13.4.2: "coalesce adjacent chunk ranges into one request; the index makes
/// this a sort and a merge").
///
/// The gap is what makes this worth doing: two chunks 4 KiB apart are one
/// request, and the 4 KiB of slack costs less than a round trip. Nothing here
/// decides *how much* slack is worth it — that is a deployment question, and
/// §13.6's numbers are the input to it.
pub fn coalesce(mut ranges: Vec<Range>, gap: u64, max: u64) -> Vec<Range> {
    ranges.sort_by_key(|r| (r.off, r.len));
    let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
    for r in ranges {
        if r.len == 0 {
            continue;
        }
        match out.last_mut() {
            Some(last)
                if r.off <= last.off + last.len + gap
                    && (r.off + r.len).saturating_sub(last.off) <= max =>
            {
                let end = (last.off + last.len).max(r.off + r.len);
                last.len = end - last.off;
            }
            _ => out.push(r),
        }
    }
    out
}

/// An HTTP/1.1 client that speaks exactly what §13.4 needs: `GET` with a
/// `Range` header, on a connection it reuses.
#[derive(Debug)]
pub struct Http {
    url: Url,
    /// The reader is kept across requests, not rebuilt per request. A fresh
    /// `BufReader` per response would discard whatever it had buffered past the
    /// body, which on a reused connection is the start of the next response.
    stream: std::cell::RefCell<Option<std::io::BufReader<std::net::TcpStream>>>,
    /// Requests issued and bytes received, for the same reason
    /// [`crate::store::FileStore`] counts them: a claim about round trips is
    /// only a claim if something counts them.
    pub requests: std::cell::Cell<u64>,
    pub bytes: std::cell::Cell<u64>,
    pub retries: std::cell::Cell<u64>,
}

impl Http {
    pub fn new(url: &str) -> Res<Http> {
        Ok(Http {
            url: Url::parse(url)?,
            stream: std::cell::RefCell::new(None),
            requests: std::cell::Cell::new(0),
            bytes: std::cell::Cell::new(0),
            retries: std::cell::Cell::new(0),
        })
    }

    fn connect(&self) -> Res<()> {
        if self.stream.borrow().is_some() {
            return Ok(());
        }
        let s = std::net::TcpStream::connect((self.url.host.as_str(), self.url.port))
            .map_err(|e| Error::Io(e.to_string()))?;
        s.set_nodelay(true).ok();
        *self.stream.borrow_mut() = Some(std::io::BufReader::new(s));
        Ok(())
    }

    /// A ranged GET. Retries once on a dropped connection, because §13.4.2's
    /// resumption story is that a partially received request costs its range and
    /// not the download.
    pub fn get_range(&self, off: u64, len: u64) -> Res<Vec<u8>> {
        match self.request(off, len) {
            Ok(v) => Ok(v),
            Err(Error::Io(_)) | Err(Error::Protocol(_)) => {
                // The connection may simply have been closed between requests.
                *self.stream.borrow_mut() = None;
                self.retries.set(self.retries.get() + 1);
                self.request(off, len)
            }
            Err(e) => Err(e),
        }
    }

    /// The whole object, for a server that has no ranges.
    pub fn get(&self) -> Res<Vec<u8>> {
        self.request_raw(None)
    }

    fn request(&self, off: u64, len: u64) -> Res<Vec<u8>> {
        let end = off + len.max(1) - 1;
        let body = self.request_raw(Some((off, end)))?;
        if body.len() as u64 != len {
            return Err(Error::Protocol(format!(
                "asked for {len} bytes at {off}, received {}",
                body.len()
            )));
        }
        Ok(body)
    }

    fn request_raw(&self, range: Option<(u64, u64)>) -> Res<Vec<u8>> {
        use std::io::{BufRead, Read, Write};
        self.connect()?;
        let mut guard = self.stream.borrow_mut();
        let reader = guard.as_mut().expect("connected");
        let mut req = format!(
            "GET {} HTTP/1.1\r\nHost: {}:{}\r\nUser-Agent: omni-rs/{}\r\n",
            self.url.path,
            self.url.host,
            self.url.port,
            env!("CARGO_PKG_VERSION")
        );
        if let Some((a, b)) = range {
            req.push_str(&format!("Range: bytes={a}-{b}\r\n"));
        }
        req.push_str("Connection: keep-alive\r\n\r\n");
        reader
            .get_mut()
            .write_all(req.as_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
        reader
            .get_mut()
            .flush()
            .map_err(|e| Error::Io(e.to_string()))?;
        self.requests.set(self.requests.get() + 1);

        let mut status = String::new();
        reader
            .read_line(&mut status)
            .map_err(|e| Error::Io(e.to_string()))?;
        let code: u16 = status
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .ok_or_else(|| Error::Protocol(format!("bad status line `{}`", status.trim())))?;
        let mut length: Option<usize> = None;
        let mut chunked = false;
        loop {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| Error::Io(e.to_string()))?;
            if n == 0 {
                return Err(Error::Protocol("headers ended without a blank line".into()));
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                length = v.trim().parse().ok();
            }
            if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
                chunked = true;
            }
        }
        // 206 for a range, 200 for the whole object. A 200 in response to a
        // Range header means the server ignored it, which is a different bug
        // from a failure and is reported as one.
        if range.is_some() && code == 200 {
            return Err(Error::Protocol(
                "the server ignored the Range header and sent the whole object".into(),
            ));
        }
        if !(code == 200 || code == 206) {
            return Err(Error::Protocol(format!("HTTP {code}")));
        }
        let mut body = Vec::new();
        if chunked {
            loop {
                let mut size_line = String::new();
                reader
                    .read_line(&mut size_line)
                    .map_err(|e| Error::Io(e.to_string()))?;
                let n = usize::from_str_radix(size_line.trim(), 16)
                    .map_err(|_| Error::Protocol("bad chunk size".into()))?;
                if n == 0 {
                    let mut trailer = String::new();
                    reader.read_line(&mut trailer).ok();
                    break;
                }
                let mut chunk = vec![0u8; n];
                reader
                    .read_exact(&mut chunk)
                    .map_err(|e| Error::Io(e.to_string()))?;
                body.extend_from_slice(&chunk);
                let mut crlf = [0u8; 2];
                reader
                    .read_exact(&mut crlf)
                    .map_err(|e| Error::Io(e.to_string()))?;
            }
        } else {
            let n = length.ok_or_else(|| {
                Error::Protocol("no Content-Length and not chunked; nothing to read".into())
            })?;
            if n > 1 << 30 {
                return Err(Error::Protocol("a response over 1 GiB".into()));
            }
            body = vec![0u8; n];
            reader
                .read_exact(&mut body)
                .map_err(|e| Error::Io(e.to_string()))?;
        }
        self.bytes.set(self.bytes.get() + body.len() as u64);
        Ok(body)
    }
}

/// A container served over HTTP, opened the way §13.4.1 describes.
///
/// With a sidecar: one request. Without: three — the trailer by suffix range,
/// the superblock, the index. Both paths are here because a CDN that handles
/// suffix ranges badly is the reason §13.4.1 offers the sidecar at all.
#[derive(Debug)]
pub struct HttpStore {
    http: Http,
    hash: HashAlgo,
    index: Vec<IndexEntry>,
    pub superblock: Value,
    pub root: Digest,
    /// The size of the container this store believes it is reading.
    pub file_size: u64,
}

impl HttpStore {
    /// Opens from the container itself, following §02.7's two-read open as
    /// closely as a stateless transport allows.
    ///
    /// Three requests when the container carries a front superblock — which the
    /// writer here always does, and §02.7 exists to encourage: the header, the
    /// front superblock it points at, the index. Four when it does not, because
    /// then the superblock's extent is only in the trailer and the trailer costs
    /// a request of its own. The count is observable through [`HttpStore::io`],
    /// so a claim about it is checkable rather than asserted.
    pub fn open(url: &str) -> Res<HttpStore> {
        let http = Http::new(url)?;
        let head = http.get_range(0, 128)?;
        let file_size = u64::from_le_bytes(head[56..64].try_into().unwrap());
        if file_size < 192 {
            return Err(Error::Protocol(format!(
                "the header declares {file_size} bytes, too few to be a container"
            )));
        }
        let header = crate::container::parse_header_bytes(&head, file_size)
            .map_err(|e| Error::Protocol(e.to_string()))?;

        // Where the superblock is, and what it must hash to. The front copy is
        // R-C10-identical to the back one, so either answers; the front one is
        // reachable without a second round trip to the end of the file.
        let (sb_off, sb_len, sb_digest) = if header.flags & crate::container::hflags::FRONT_SB != 0
        {
            (header.front_sb_off, header.front_sb_len, None)
        } else {
            let trailer = http.get_range(file_size - 64, 64)?;
            if trailer[56..64] != crate::container::MAGIC_END {
                return Err(Error::Protocol("trailer magic mismatch".into()));
            }
            let d: Digest = trailer[16..48].try_into().unwrap();
            (
                u64::from_le_bytes(trailer[0..8].try_into().unwrap()),
                u64::from_le_bytes(trailer[8..16].try_into().unwrap()),
                Some(d),
            )
        };
        if sb_len == 0 || sb_len > 1 << 24 || sb_off + sb_len > file_size {
            return Err(Error::Protocol(format!(
                "a superblock of {sb_len} bytes at {sb_off} is not plausible"
            )));
        }
        let sb = http.get_range(sb_off, sb_len)?;
        if let Some(want) = sb_digest {
            if header.hash.digest(&sb) != want {
                return Err(Error::Protocol("superblock digest mismatch".into()));
            }
        }
        let superblock = crate::cbor::decode(&sb).map_err(|e| Error::Protocol(e.to_string()))?;
        let (ioff, ilen) = index_extent(&superblock)?;
        if ioff + ilen > file_size {
            return Err(Error::Protocol("the index extent leaves the file".into()));
        }
        // The segment header sits in the 64 bytes before the index body, and
        // `parse_index` wants both, so one request covers them.
        let raw = http.get_range(ioff - 64, ilen + 64)?;
        let index = crate::container::parse_index_bytes(&raw, 64, ilen as usize)
            .map_err(|e| Error::Protocol(e.to_string()))?;
        Ok(HttpStore {
            http,
            hash: header.hash,
            index,
            superblock,
            root: header.root_digest,
            file_size,
        })
    }

    /// Opens from a `.omni.idx` sidecar and a data URL: §13.4.1's one round trip.
    ///
    /// Zero requests. Nothing about the served file is checked here because
    /// nothing about it has been read; R-X02's other half — that the container
    /// actually served is the size the sidecar claims — is checked by
    /// [`HttpStore::confirm_target`], which costs a request and is therefore the
    /// caller's decision.
    pub fn open_with_sidecar(data_url: &str, sidecar_bytes: &[u8]) -> Res<HttpStore> {
        let s = Sidecar::parse(sidecar_bytes)?;
        Ok(HttpStore {
            http: Http::new(data_url)?,
            hash: s.hash,
            index: s.index,
            superblock: s.superblock,
            root: s.root,
            file_size: s.file_size,
        })
    }

    /// R-X02: one request that confirms the served container is the one the
    /// sidecar describes, before any offset from the sidecar is trusted.
    ///
    /// The container's own 128-byte header answers it: it carries `file_size` and
    /// the root digest, and it is covered by a CRC, so a truncated or swapped file
    /// cannot pass by accident. Both are checked — size alone would let two
    /// containers of equal length be confused for each other, and two builds of
    /// the same model differing only in weights are exactly that.
    pub fn confirm_target(&self) -> Res<()> {
        let head = self.http.get_range(0, 128)?;
        let served_size = u64::from_le_bytes(head[56..64].try_into().unwrap());
        let header = crate::container::parse_header_bytes(&head, served_size)
            .map_err(|e| Error::Protocol(format!("R-X02: {e}")))?;
        if served_size != self.file_size {
            return Err(Error::Protocol(format!(
                "R-X02: the sidecar describes a container of {} bytes, the server \
                 is serving {served_size}",
                self.file_size
            )));
        }
        if header.root_digest != self.root {
            return Err(Error::Protocol(format!(
                "R-X02: the sidecar describes the container rooted at {}, the \
                 server is serving {}",
                crate::sha256::hex(&self.root),
                crate::sha256::hex(&header.root_digest)
            )));
        }
        Ok(())
    }

    pub fn index(&self) -> &[IndexEntry] {
        &self.index
    }

    pub fn io(&self) -> (u64, u64, u64) {
        (
            self.http.requests.get(),
            self.http.bytes.get(),
            self.http.retries.get(),
        )
    }

    pub fn find(&self, d: &Digest) -> Option<&IndexEntry> {
        let i = self.index.binary_search_by(|e| e.digest.cmp(d)).ok()?;
        self.index.get(i)
    }

    /// Fetches several objects in as few requests as §13.4.2's coalescing allows.
    ///
    /// Returns them in the order asked for. An absent object is `None`; a present
    /// one is checked against its digest, because bytes from a CDN edge are bytes
    /// from a stranger.
    pub fn fetch_many(&self, want: &[Digest], gap: u64, max: u64) -> Res<Vec<Option<Vec<u8>>>> {
        let mut ranges = Vec::new();
        for d in want {
            if let Some(e) = self.find(d) {
                if e.oflags & oflags::EXTERNAL == 0 && e.stored_len > 0 {
                    ranges.push(Range {
                        off: e.offset,
                        len: e.stored_len,
                    });
                }
            }
        }
        let merged = coalesce(ranges, gap, max);
        let mut buffers: Vec<(Range, Vec<u8>)> = Vec::with_capacity(merged.len());
        for r in merged {
            buffers.push((r, self.http.get_range(r.off, r.len)?));
        }
        let mut out = Vec::with_capacity(want.len());
        for d in want {
            let Some(e) = self.find(d) else {
                out.push(None);
                continue;
            };
            if e.oflags & oflags::EXTERNAL != 0 || e.stored_len == 0 {
                out.push(None);
                continue;
            }
            let Some((r, buf)) = buffers
                .iter()
                .find(|(r, _)| r.off <= e.offset && e.offset + e.stored_len <= r.off + r.len)
            else {
                out.push(None);
                continue;
            };
            let at = (e.offset - r.off) as usize;
            let stored = &buf[at..at + e.stored_len as usize];
            let logical = decode_object(stored, e, self.hash, &self.superblock)?;
            out.push(Some(logical));
        }
        Ok(out)
    }
}

fn index_extent(sb: &Value) -> Res<(u64, u64)> {
    let idx = sb
        .get("index")
        .ok_or_else(|| Error::Protocol("superblock has no index".into()))?;
    let off = idx.get("off").and_then(|v| v.as_u64()).unwrap_or(0);
    let len = idx.get("len").and_then(|v| v.as_u64()).unwrap_or(0);
    if off < 64 || len > 1 << 32 {
        return Err(Error::Protocol("an implausible index extent".into()));
    }
    Ok((off, len))
}

/// The codec an index entry names, with its parameters taken from the
/// superblock's descriptor rather than guessed.
///
/// The index has room for a codec *id* and nothing else (§02.6), so
/// `bitshuffle+zstd`'s element size lives in the superblock. A reader that
/// skips this step decodes with the wrong element size and gets plausible
/// garbage, which is the worst kind.
fn codec_for(sb: &Value, id: u8) -> crate::codec::Codec {
    if id == crate::codec::id::RAW {
        return crate::codec::Codec::Raw;
    }
    if let Some(Value::Array(list)) = sb.get("codecs") {
        for c in list {
            let declared = crate::codec::Codec::from_value(c);
            if declared.id() == id {
                return declared;
            }
        }
    }
    crate::codec::Codec::from_id(id)
}

fn decode_object(stored: &[u8], e: &IndexEntry, hash: HashAlgo, sb: &Value) -> Res<Vec<u8>> {
    let codec = codec_for(sb, e.codec);
    let logical = match codec {
        crate::codec::Codec::Raw => stored.to_vec(),
        other => other
            .decode(stored, e.logical_len, false)
            .map_err(|err| Error::Protocol(err.to_string()))?,
    };
    if hash.digest(&logical) != e.digest {
        return Err(Error::Protocol(format!(
            "R-O01: the bytes served for {} do not hash to it",
            crate::sha256::hex(&e.digest)
        )));
    }
    Ok(logical)
}

impl Store for HttpStore {
    fn hash(&self) -> HashAlgo {
        self.hash
    }

    fn resolve(&self, d: &Digest) -> Result<Option<Vec<u8>>, StoreError> {
        let Some(e) = self.find(d) else {
            return Ok(None);
        };
        if e.oflags & oflags::EXTERNAL != 0 || e.stored_len == 0 {
            return Ok(None);
        }
        let stored = self
            .http
            .get_range(e.offset, e.stored_len)
            .map_err(|err| StoreError::Corrupt(err.to_string()))?;
        decode_object(&stored, e, self.hash, &self.superblock)
            .map(Some)
            .map_err(|err| StoreError::Corrupt(err.to_string()))
    }

    /// §13.4.2: a range of an object is a range of a request. Only for
    /// uncompressed objects, for the same reason the file store says so.
    fn resolve_range(&self, d: &Digest, off: u64, n: u64) -> Result<Option<Vec<u8>>, StoreError> {
        let Some(e) = self.find(d) else {
            return Ok(None);
        };
        if e.codec != crate::codec::id::RAW || e.oflags & oflags::EXTERNAL != 0 {
            return Ok(self.resolve(d)?.map(|b| {
                let s = (off as usize).min(b.len());
                let end = s.saturating_add(n as usize).min(b.len());
                b[s..end].to_vec()
            }));
        }
        if off >= e.logical_len {
            return Ok(Some(Vec::new()));
        }
        let take = n.min(e.logical_len - off);
        self.http
            .get_range(e.offset + off, take)
            .map(Some)
            .map_err(|err| StoreError::Corrupt(err.to_string()))
    }

    fn has(&self, d: &Digest) -> Result<bool, StoreError> {
        Ok(self
            .find(d)
            .is_some_and(|e| e.oflags & oflags::EXTERNAL == 0 && e.stored_len > 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{pack, PackOptions};
    use crate::model::{ModelBuilder, TensorSpec};
    use std::io::Write;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    fn toy() -> Container {
        let (objs, root) = ModelBuilder::new("test/transport")
            .tensor(TensorSpec {
                name: "a".into(),
                shape: vec![64],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..256u32).map(|i| (i % 251) as u8).collect(),
                layout: None,
            })
            .tensor(TensorSpec {
                name: "b".into(),
                shape: vec![32],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..128u32).map(|i| (i % 97) as u8).collect(),
                layout: None,
            })
            .build();
        Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
    }

    // ------------------------------------------------------------ test server --

    /// How the server should misbehave, because §13.4's interesting cases are
    /// all failures: a CDN that drops the connection, one that ignores `Range`,
    /// one that answers in chunks.
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Ranges,
        IgnoreRanges,
        Chunked,
        /// Accept the first request and close without answering it.
        DropFirst,
    }

    /// An HTTP/1.1 server that serves ranges of one byte slice, and nothing
    /// else. It exists so the range store is tested rather than asserted; a test
    /// that needs the network is a test that does not run.
    struct Server {
        port: u16,
        served: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
    }

    impl Server {
        fn start(body: Vec<u8>, mode: Mode) -> Server {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let served = Arc::new(AtomicU64::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let (s2, st2) = (served.clone(), stop.clone());
            let body = Arc::new(body);
            std::thread::spawn(move || {
                let mut first = true;
                for conn in listener.incoming() {
                    if st2.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut conn) = conn else { break };
                    if mode == Mode::DropFirst && first {
                        first = false;
                        // Read the request, then hang up without answering.
                        let mut r = std::io::BufReader::new(&conn);
                        read_request(&mut r);
                        continue;
                    }
                    // One thread per connection. Serving connections in sequence
                    // instead would deadlock the moment a test holds two clients
                    // open at once — and holding two open is exactly how you check
                    // that a store planning against the wrong sidecar is caught.
                    let (body, s2) = (body.clone(), s2.clone());
                    std::thread::spawn(move || {
                        let mut r = std::io::BufReader::new(conn.try_clone().unwrap());
                        while let Some(range) = read_request(&mut r) {
                            s2.fetch_add(1, Ordering::SeqCst);
                            let (start, end) = match range {
                                Some((a, b)) => (a as usize, (b as usize + 1).min(body.len())),
                                None => (0, body.len()),
                            };
                            if start >= body.len() {
                                let _ = conn.write_all(
                                    b"HTTP/1.1 416 Range Not Satisfiable\r\n\
                                      Content-Length: 0\r\n\r\n",
                                );
                                continue;
                            }
                            let part: &[u8] = if mode == Mode::IgnoreRanges {
                                &body
                            } else {
                                &body[start..end]
                            };
                            let code = if range.is_some() && mode != Mode::IgnoreRanges {
                                "206 Partial Content"
                            } else {
                                "200 OK"
                            };
                            if mode == Mode::Chunked {
                                let mut head = format!(
                                    "HTTP/1.1 {code}\r\nTransfer-Encoding: chunked\r\n\
                                     Connection: keep-alive\r\n\r\n"
                                );
                                // Two chunks, so the dechunker has to concatenate.
                                let mid = part.len() / 2;
                                for c in [&part[..mid], &part[mid..]] {
                                    head.push_str(&format!("{:x}\r\n", c.len()));
                                    let mut out = head.into_bytes();
                                    out.extend_from_slice(c);
                                    out.extend_from_slice(b"\r\n");
                                    let _ = conn.write_all(&out);
                                    head = String::new();
                                }
                                let _ = conn.write_all(b"0\r\n\r\n");
                            } else {
                                let head = format!(
                                    "HTTP/1.1 {code}\r\nContent-Length: {}\r\n\
                                     Accept-Ranges: bytes\r\nConnection: keep-alive\r\n\r\n",
                                    part.len()
                                );
                                let mut out = head.into_bytes();
                                out.extend_from_slice(part);
                                let _ = conn.write_all(&out);
                            }
                            let _ = conn.flush();
                        }
                    });
                }
            });
            Server { port, served, stop }
        }

        fn url(&self) -> String {
            format!("http://127.0.0.1:{}/model.omni", self.port)
        }

        fn requests(&self) -> u64 {
            self.served.load(Ordering::SeqCst)
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            // Wake the blocking `accept` so the thread notices and exits.
            let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
        }
    }

    /// Reads one request; returns its byte range, or `None` at end of stream.
    #[allow(clippy::type_complexity)]
    fn read_request<R: std::io::BufRead>(r: &mut R) -> Option<Option<(u64, u64)>> {
        let mut range = None;
        let mut any = false;
        loop {
            let mut line = String::new();
            if r.read_line(&mut line).ok()? == 0 {
                return if any { Some(range) } else { None };
            }
            any = true;
            let line = line.trim_end();
            if line.is_empty() {
                return Some(range);
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                let (a, b) = v.split_once('-')?;
                range = Some((a.parse().ok()?, b.parse().ok()?));
            }
        }
    }

    // ---------------------------------------------------------------- sidecar --

    /// §13.4.1: the sidecar has to say everything the file's framing says, or a
    /// reader that trusts it plans against a fiction.
    #[test]
    fn a_sidecar_carries_the_framing_of_the_container_it_describes() {
        let c = toy();
        let s = Sidecar::parse(&sidecar(&c)).unwrap();
        assert_eq!(s.hash, c.header.hash);
        assert_eq!(s.root, c.header.root_digest);
        assert_eq!(s.file_size, c.bytes.len() as u64);
        assert_eq!(s.superblock.encode(), c.superblock.encode());
        assert_eq!(s.index.len(), c.index.len());
        for (a, b) in s.index.iter().zip(c.index.iter()) {
            assert_eq!(a.digest, b.digest);
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.stored_len, b.stored_len);
            assert_eq!(a.logical_len, b.logical_len);
            assert_eq!(a.otype, b.otype);
        }
        // And it is a fraction of the file, which is the entire point.
        assert!(sidecar(&c).len() < c.bytes.len());
    }

    /// A sidecar is not a container, and a container is not a sidecar. Each has
    /// to refuse the other rather than parse three fields of it and continue.
    #[test]
    fn the_two_magics_do_not_collide() {
        let c = toy();
        assert!(Sidecar::parse(&c.bytes).is_err());
        assert!(Container::open(sidecar(&c)).is_err());
    }

    #[test]
    fn a_damaged_sidecar_is_detected_rather_than_believed() {
        let c = toy();
        let good = sidecar(&c);

        // Truncation, at every length. None of these may panic, and none may
        // parse.
        for n in 0..good.len() {
            assert!(Sidecar::parse(&good[..n]).is_err(), "prefix of {n} parsed");
        }

        // A single flipped bit anywhere in the header.
        for i in 0..IDX_HEADER {
            let mut bad = good.clone();
            bad[i] ^= 0x01;
            assert!(Sidecar::parse(&bad).is_err(), "byte {i} of the header");
        }

        // A rewritten superblock: the CRC covers the digest, the digest covers
        // the superblock, so editing the superblock alone is caught.
        let sb_off =
            u64::from_le_bytes(good[IDX_PARTS + 16..IDX_PARTS + 24].try_into().unwrap()) as usize;
        let mut bad = good.clone();
        bad[sb_off + 4] ^= 0x20;
        match Sidecar::parse(&bad) {
            Err(Error::Malformed(m)) => assert!(m.contains("digest"), "{m}"),
            other => panic!("a rewritten superblock parsed: {other:?}"),
        }
    }

    // ------------------------------------------------------------ index-only --

    /// §13.8: a catalogue that describes everything and holds no weights. The
    /// numbers are the claim — it must be far smaller, and it must still say how
    /// large the model is.
    #[test]
    fn an_index_only_container_describes_everything_and_holds_no_weights() {
        // Big enough for the claim to be a claim: §13.8's point is a catalogue
        // far smaller than the model it describes, and at 384 bytes of weights
        // there is nothing to be smaller than.
        let (objs, root) = ModelBuilder::new("test/index-only")
            .tensor(TensorSpec {
                name: "w".into(),
                shape: vec![65536],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..262144u32).map(|i| (i % 251) as u8).collect(),
                layout: None,
            })
            .build();
        let c = Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap();
        let thin = Container::open(index_only(&c).unwrap()).unwrap();

        assert!(
            thin.bytes.len() < c.bytes.len(),
            "index-only is {} bytes against {}",
            thin.bytes.len(),
            c.bytes.len()
        );
        assert_eq!(thin.index.len(), c.index.len(), "every object described");
        assert_eq!(thin.header.root_digest, c.header.root_digest);
        assert_ne!(thin.header.flags & crate::container::hflags::PARTIAL, 0);

        // Every blob is described and absent; every structure object is present.
        let mut absent = 0;
        for e in &thin.index {
            if e.otype == crate::container::otype::BLOB {
                assert_ne!(e.oflags & oflags::EXTERNAL, 0);
                assert_eq!(e.stored_len, 0);
                assert!(e.logical_len > 0, "the size is still described");
                absent += 1;
            } else {
                assert_eq!(e.oflags & oflags::EXTERNAL, 0);
                assert_eq!(thin.read(&e.digest).unwrap(), c.read(&e.digest).unwrap());
            }
        }
        assert!(absent > 0);

        // It still validates: an incomplete container is incomplete, not invalid
        // (§01.4).
        let r = crate::container::verify(&thin).unwrap();
        assert!(r.padding_ok, "R-C07");
        assert!(r.alignment_ok, "R-C08");
        assert!(r.mistyped.is_empty(), "R-O02");
        assert!(r.dangling.is_empty(), "the graph is whole even so");

        // And it reports itself as what it is.
        let comp = Completeness::of(&thin);
        assert_eq!(comp.described, thin.index.len());
        assert!(!comp.is_complete());
        assert!(comp.percent() < 100.0);
        assert!(Completeness::of(&c).is_complete());
        assert_eq!(Completeness::of(&c).percent(), 100.0);
        // The superblock keeps the model's real size while the file does not.
        let logical = thin
            .superblock
            .get("stats")
            .and_then(|s| s.get("bytes_logical"))
            .and_then(|v| v.as_u64())
            .unwrap();
        assert_eq!(logical, comp.described_bytes);
        assert!(logical > thin.bytes.len() as u64);
    }

    // ----------------------------------------------------------------- ranges --

    #[test]
    fn coalescing_merges_what_is_worth_merging() {
        let r = |off, len| Range { off, len };
        // Adjacent and overlapping ranges become one; a gap larger than the
        // slack stays two requests.
        assert_eq!(
            coalesce(vec![r(0, 10), r(10, 10)], 0, 1 << 30),
            vec![r(0, 20)]
        );
        assert_eq!(
            coalesce(vec![r(0, 10), r(5, 10)], 0, 1 << 30),
            vec![r(0, 15)]
        );
        assert_eq!(
            coalesce(vec![r(0, 10), r(4106, 10)], 4096, 1 << 30),
            vec![r(0, 4116)]
        );
        assert_eq!(
            coalesce(vec![r(0, 10), r(4107, 10)], 4096, 1 << 30),
            vec![r(0, 10), r(4107, 10)]
        );
        // Order does not matter, and empty ranges are not requests.
        assert_eq!(
            coalesce(vec![r(20, 10), r(0, 10), r(5, 0)], 0, 1 << 30),
            vec![r(0, 10), r(20, 10)]
        );
        // The cap is honoured: a merge that would exceed it does not happen.
        assert_eq!(
            coalesce(vec![r(0, 10), r(10, 10)], 0, 15),
            vec![r(0, 10), r(10, 10)]
        );
    }

    #[test]
    fn a_url_is_parsed_and_https_is_refused_with_its_reason() {
        assert_eq!(
            Url::parse("http://example.org/a/b.omni").unwrap(),
            Url {
                host: "example.org".into(),
                port: 80,
                path: "/a/b.omni".into()
            }
        );
        assert_eq!(Url::parse("http://h:8080").unwrap().port, 8080);
        assert_eq!(Url::parse("http://h:8080").unwrap().path, "/");
        match Url::parse("https://example.org/a.omni") {
            Err(Error::Unsupported(m)) => {
                assert!(m.contains("TLS"), "{m}");
                assert!(m.contains("no dependencies"), "{m}");
            }
            other => panic!("https was not refused: {other:?}"),
        }
        assert!(Url::parse("ftp://example.org/a").is_err());
        assert!(Url::parse("http:///a").is_err());
    }

    // ------------------------------------------------------------ http store --

    /// §13.4.1: opening a remote container costs a bounded number of round trips
    /// — and the bound is checkable, so here it is checked.
    #[test]
    fn a_container_opens_over_http_in_three_requests() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open(&srv.url()).unwrap();

        assert_eq!(store.root, c.header.root_digest);
        assert_eq!(store.index().len(), c.index.len());
        let (requests, bytes, retries) = store.io();
        assert_eq!(requests, 3, "header, front superblock, index");
        assert_eq!(retries, 0);
        assert!(
            bytes < c.bytes.len() as u64 / 2,
            "opening read {bytes} of {} bytes",
            c.bytes.len()
        );
        assert_eq!(srv.requests(), 3);
    }

    /// The sidecar's whole claim: one round trip instead of three.
    #[test]
    fn a_sidecar_opens_a_remote_container_without_touching_it() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(&c)).unwrap();
        assert_eq!(store.io().0, 0, "the sidecar was already in hand");
        assert_eq!(srv.requests(), 0);
        assert_eq!(store.root, c.header.root_digest);

        // And it is a working store: every object resolves, byte for byte.
        for e in &c.index {
            let got = store.resolve(&e.digest).unwrap().expect("present");
            assert_eq!(got, c.read(&e.digest).unwrap());
        }
        assert_eq!(store.io().0 as usize, c.index.len());
    }

    /// R-X02: a sidecar is an assertion about one specific container. Planning
    /// ranges from it against a different file reads the wrong bytes of the right
    /// URL, which is the failure mode that looks like corruption.
    #[test]
    fn a_sidecar_is_checked_against_the_container_actually_served() {
        let c = toy();
        let other = {
            let (objs, root) = ModelBuilder::new("test/a-different-model")
                .tensor(TensorSpec {
                    name: "w".into(),
                    shape: vec![16],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight",
                    data: vec![9u8; 64],
                    layout: None,
                })
                .build();
            Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
        };
        assert_ne!(other.bytes.len(), c.bytes.len());
        // And a container of exactly the same size holding different weights:
        // the case a size comparison alone would wave through.
        let twin = {
            let (objs, root) = ModelBuilder::new("test/transport")
                .tensor(TensorSpec {
                    name: "a".into(),
                    shape: vec![64],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight",
                    data: (0..256u32).map(|i| (i % 241) as u8).collect(),
                    layout: None,
                })
                .tensor(TensorSpec {
                    name: "b".into(),
                    shape: vec![32],
                    dtype: crate::dtype::DType::F32,
                    axes: None,
                    semantic: "weight",
                    data: (0..128u32).map(|i| (i % 89) as u8).collect(),
                    layout: None,
                })
                .build();
            Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
        };
        assert_eq!(twin.bytes.len(), c.bytes.len());
        assert_ne!(twin.header.root_digest, c.header.root_digest);

        // The right sidecar for the served file: one request, and it agrees.
        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(&c)).unwrap();
        store.confirm_target().unwrap();
        assert_eq!(store.io().0, 1, "confirming costs exactly one request");

        // The wrong ones: each caught before an offset from it is used, whether
        // the size gives it away or only the root digest does.
        for wrong in [&other, &twin] {
            let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(wrong)).unwrap();
            match store.confirm_target() {
                Err(Error::Protocol(m)) => assert!(m.contains("R-X02"), "{m}"),
                got => panic!("a sidecar for another container was accepted: {got:?}"),
            }
        }
    }

    /// Bytes from a CDN edge are bytes from a stranger (§13.7). A store that
    /// does not check them is a store that serves whatever it was handed.
    #[test]
    fn served_bytes_are_checked_against_their_digest() {
        let c = toy();
        let blob = c
            .index
            .iter()
            .find(|e| e.otype == crate::container::otype::BLOB && e.stored_len > 8)
            .unwrap()
            .clone();

        let mut tampered = c.bytes.clone();
        tampered[blob.offset as usize + 3] ^= 0xff;
        let srv = Server::start(tampered, Mode::Ranges);
        let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(&c)).unwrap();

        match store.resolve(&blob.digest) {
            Err(crate::store::Error::Corrupt(m)) => assert!(m.contains("R-O01"), "{m}"),
            other => panic!("a tampered object was accepted: {other:?}"),
        }
        // An untouched object from the same file still resolves: one bad object
        // is one bad object.
        let ok = c
            .index
            .iter()
            .find(|e| e.otype != crate::container::otype::BLOB)
            .unwrap();
        assert!(store.resolve(&ok.digest).unwrap().is_some());
    }

    /// §13.4.2: many objects, few requests. The count is the whole claim.
    #[test]
    fn fetching_many_objects_coalesces_into_fewer_requests() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(&c)).unwrap();

        let want: Vec<Digest> = c.index.iter().map(|e| e.digest).collect();
        let got = store.fetch_many(&want, 1 << 20, 1 << 30).unwrap();
        assert_eq!(got.len(), want.len());
        for (d, g) in want.iter().zip(got.iter()) {
            assert_eq!(g.as_deref(), Some(&c.read(d).unwrap()[..]), "object {d:?}");
        }
        let requests = store.io().0;
        assert!(
            requests < want.len() as u64,
            "{requests} requests for {} objects",
            want.len()
        );

        // A digest the container never held is absent, not an error.
        assert_eq!(
            store.fetch_many(&[[7u8; 32]], 0, 1 << 30).unwrap(),
            vec![None]
        );
    }

    /// A range of an object is a range of a request, not a whole download.
    #[test]
    fn a_range_of_an_object_is_a_range_of_a_request() {
        let c = toy();
        let blob = c
            .index
            .iter()
            .find(|e| e.otype == crate::container::otype::BLOB && e.logical_len >= 64)
            .unwrap()
            .clone();
        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open_with_sidecar(&srv.url(), &sidecar(&c)).unwrap();

        let whole = c.read(&blob.digest).unwrap();
        let part = store.resolve_range(&blob.digest, 16, 32).unwrap().unwrap();
        assert_eq!(part, &whole[16..48]);
        assert!(
            store.io().1 < whole.len() as u64,
            "it read less than all of it"
        );
        // Past the end is empty, not an error.
        assert_eq!(
            store
                .resolve_range(&blob.digest, blob.logical_len + 8, 16)
                .unwrap(),
            Some(Vec::new())
        );
    }

    /// A dropped connection costs its range, not the download (§13.4.2).
    #[test]
    fn a_dropped_connection_is_retried() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::DropFirst);
        let store = HttpStore::open(&srv.url()).unwrap();
        assert_eq!(store.io().2, 1, "one retry, then it worked");
        assert_eq!(store.root, c.header.root_digest);
    }

    #[test]
    fn a_chunked_response_is_dechunked() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::Chunked);
        let store = HttpStore::open(&srv.url()).unwrap();
        assert_eq!(store.index().len(), c.index.len());
    }

    /// A server that ignores `Range` is a different failure from a server that
    /// errors, and conflating them means silently downloading 700 GB to read a
    /// header.
    #[test]
    fn a_server_that_ignores_ranges_is_told_apart_from_one_that_fails() {
        let c = toy();
        let srv = Server::start(c.bytes.clone(), Mode::IgnoreRanges);
        match HttpStore::open(&srv.url()) {
            Err(Error::Protocol(m)) => assert!(m.contains("ignored the Range header"), "{m}"),
            other => panic!("expected a protocol error, got {other:?}"),
        }
    }

    /// An index-only container served over HTTP: the descriptions resolve to
    /// `None`, which is absence rather than failure.
    #[test]
    fn an_absent_object_is_absent_rather_than_an_error() {
        let c = toy();
        let thin = Container::open(index_only(&c).unwrap()).unwrap();
        let srv = Server::start(thin.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open(&srv.url()).unwrap();
        for e in &thin.index {
            let external = e.oflags & oflags::EXTERNAL != 0;
            assert_eq!(store.has(&e.digest).unwrap(), !external);
            assert_eq!(store.resolve(&e.digest).unwrap().is_none(), external);
        }
    }

    /// A compressed container over HTTP: the index carries a codec id and the
    /// superblock carries its parameters, so a reader that skips the superblock
    /// decodes with the wrong ones.
    #[test]
    fn a_compressed_container_decodes_over_http_with_the_declared_parameters() {
        let (objs, root) = ModelBuilder::new("test/transport-zstd")
            .tensor(TensorSpec {
                name: "w".into(),
                shape: vec![512],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight",
                data: (0..2048u32).map(|i| (i / 64) as u8).collect(),
                layout: None,
            })
            .build();
        let c = Container::open(
            pack(
                &objs,
                &root,
                &PackOptions {
                    codec: crate::codec::Codec::BitshuffleZstd {
                        elem_size: 4,
                        level: 3,
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert!(c.index.iter().any(|e| e.codec != crate::codec::id::RAW));

        let srv = Server::start(c.bytes.clone(), Mode::Ranges);
        let store = HttpStore::open(&srv.url()).unwrap();
        for e in &c.index {
            assert_eq!(
                store.resolve(&e.digest).unwrap().as_deref(),
                Some(&c.read(&e.digest).unwrap()[..]),
                "object with codec {}",
                e.codec
            );
        }
    }
}
