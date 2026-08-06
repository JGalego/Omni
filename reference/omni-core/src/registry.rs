//! §13.5 — the registry *client*: push and pull over the OCI distribution API.
//!
//! [`crate::oci`] maps a container onto an OCI image layout, which is the half
//! of §13.5 that needs no network. This is the other half, and the roadmap has
//! carried it as the missing piece of Phase 3 since the mapping was written: a
//! layout that has never been pushed anywhere is a claim about distribution
//! rather than a demonstration of it.
//!
//! ## What the protocol actually is
//!
//! Four requests per blob and one for the manifest:
//!
//! ```text
//! HEAD /v2/<name>/blobs/<digest>          → 200 present, 404 absent
//! POST /v2/<name>/blobs/uploads/          → 202 + Location
//! PUT  <location>?digest=<digest>         → 201
//! PUT  /v2/<name>/manifests/<reference>   → 201
//! ```
//!
//! The `HEAD` is not an optimization. It is where §13.5's dedup claim becomes a
//! *measurement*: pushing a delta container after its base uploads only the
//! blobs the registry does not already have, and [`Push::skipped`] counts them.
//! A number that comes from a registry answering 200 is worth more than the same
//! number computed locally, because it is the registry's own opinion about what
//! it already stores.
//!
//! Pulling is the same in reverse, with one rule that is not optional: **every
//! blob is verified against the digest that named it** before it becomes part of
//! a container. A registry is a mirror, a CDN edge is a stranger, and a pull that
//! trusts the bytes it received has verified nothing. That check is
//! [`crate::oci::import_layout`]'s already — so `pull` fetches into the same
//! reader interface a directory layout uses, and the verification code is shared
//! rather than reimplemented for the network.
//!
//! ## What this cannot do, and why
//!
//! **`https://`.** [`crate::transport::Url`] refuses it: TLS needs a
//! cryptographic transport stack and this crate has no dependencies to provide
//! one. Every registry on the public internet is HTTPS, so what works here is a
//! registry reachable over plaintext — a local one, a mirror inside a cluster, or
//! anything behind a TLS terminator, which is what a registry looks like from
//! inside most deployments. CI runs a real `registry:2` and pushes to it.
//!
//! **The bearer-token dance.** A registry that answers `401` with a
//! `WWW-Authenticate: Bearer realm=…` challenge is asking for a token from a
//! realm that is, in practice, always `https://`. The challenge is parsed and
//! reported with the URL it would have fetched, because "401" alone tells a user
//! nothing about which credential is missing. What is *not* done is a guess:
//! there is no anonymous-token retry, since a token endpoint this build cannot
//! reach is not made reachable by trying it twice.
//!
//! **Chunked blob uploads.** The monolithic `PUT` is what this does, and it is
//! what the specification allows for any blob size. Chunking exists so a 5 GB
//! layer can resume after a broken connection, which matters for a real mirror
//! and is a resumption story rather than a protocol requirement — it is named
//! here rather than half-implemented.

use crate::container::Container;
use crate::json;
use crate::oci::{self, Layout};
use crate::transport::{Http, Url};

pub const MANIFEST_TYPE: &str = oci::MANIFEST_TYPE;

#[derive(Debug)]
pub enum Error {
    /// The reference does not name a repository and a tag.
    Reference(String),
    /// The registry said something this client cannot act on.
    Registry(String),
    /// The registry wants a credential this build cannot obtain.
    Auth(String),
    Transport(String),
    Oci(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Reference(m) => write!(f, "{m}"),
            Error::Registry(m) => write!(f, "the registry: {m}"),
            Error::Auth(m) => write!(f, "{m}"),
            Error::Transport(m) => write!(f, "{m}"),
            Error::Oci(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// A parsed `host[:port]/name/space/repo:tag` reference.
///
/// Deliberately not Docker's shorthand. `alpine` meaning
/// `registry-1.docker.io/library/alpine:latest` is a convenience with a default
/// registry baked into it, and a tool that silently contacts a host nobody named
/// is doing something the user did not ask for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    pub host: String,
    pub port: u16,
    /// The repository path, without a leading slash: `models/llama-3-8b`.
    pub name: String,
    /// A tag, or a `sha256:…` digest.
    pub reference: String,
}

impl Reference {
    /// Parses `host[:port]/name[:tag|@sha256:…]`.
    pub fn parse(s: &str) -> Res<Reference> {
        let s = s.strip_prefix("http://").unwrap_or(s);
        if s.starts_with("https://") {
            return Err(Error::Reference(
                "https needs a TLS stack, and this crate has no dependencies to \
                 provide one. A registry behind a TLS terminator, or a local one, \
                 is reachable over http"
                    .into(),
            ));
        }
        let (authority, rest) = s.split_once('/').ok_or_else(|| {
            Error::Reference(format!(
                "`{s}` names no repository: a reference is host[:port]/name[:tag]"
            ))
        })?;
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>()
                    .map_err(|_| Error::Reference(format!("`{p}` is not a port")))?,
            ),
            None => (authority.to_string(), 80),
        };
        if host.is_empty() {
            return Err(Error::Reference("no host".into()));
        }
        // A digest reference wins over a tag: `@` cannot appear in either.
        let (name, reference) = match rest.split_once('@') {
            Some((n, d)) => (n.to_string(), d.to_string()),
            None => match rest.rsplit_once(':') {
                Some((n, t)) => (n.to_string(), t.to_string()),
                None => (rest.to_string(), "latest".to_string()),
            },
        };
        if name.is_empty() {
            return Err(Error::Reference("no repository name".into()));
        }
        Ok(Reference {
            host,
            port,
            name,
            reference,
        })
    }

    fn base_url(&self) -> String {
        format!("http://{}:{}/", self.host, self.port)
    }
}

/// What a push did, counted rather than described.
#[derive(Clone, Debug, Default)]
pub struct Push {
    /// Blobs uploaded, and their total size.
    pub uploaded: usize,
    pub uploaded_bytes: u64,
    /// Blobs the registry already had — §13.5's dedup, as the registry sees it.
    pub skipped: usize,
    pub skipped_bytes: u64,
    /// The manifest digest the artifact is now addressable by.
    pub manifest_digest: String,
    pub requests: u64,
}

impl Push {
    pub fn blobs(&self) -> usize {
        self.uploaded + self.skipped
    }

    pub fn total_bytes(&self) -> u64 {
        self.uploaded_bytes + self.skipped_bytes
    }
}

/// What a pull found.
#[derive(Debug)]
pub struct Pull {
    pub bytes: Vec<u8>,
    pub manifest_digest: String,
    pub layers: usize,
    pub annotations: Vec<(String, String)>,
    pub requests: u64,
    pub fetched_bytes: u64,
}

fn client(r: &Reference) -> Res<Http> {
    let _ = Url::parse(&r.base_url()).map_err(|e| Error::Transport(e.to_string()))?;
    Http::new(&r.base_url()).map_err(|e| Error::Transport(e.to_string()))
}

/// Turns a `401` into an error that says which credential is missing.
fn challenge(resp: &crate::transport::Response, what: &str) -> Error {
    match resp.header("www-authenticate") {
        Some(c) => {
            let realm = c
                .split(',')
                .find_map(|p| p.trim().strip_prefix("realm=").map(|r| r.trim_matches('"')))
                .unwrap_or("(unstated)");
            Error::Auth(format!(
                "{what}: the registry wants a bearer token from `{realm}`. That \
                 endpoint is https on every registry this could reach, and this \
                 build has no TLS — so the token cannot be fetched here. Push to a \
                 registry that does not require one, or through something that \
                 terminates TLS"
            ))
        }
        None => Error::Auth(format!(
            "{what}: the registry refused with 401 and stated no challenge"
        )),
    }
}

fn status_error(what: &str, resp: &crate::transport::Response) -> Error {
    // A registry states its errors as JSON, and quoting the code it chose is
    // more use than the status line.
    let detail = json::parse(&resp.body)
        .ok()
        .and_then(|v| {
            v.get("errors")
                .and_then(|e| e.as_array())
                .and_then(|a| a.first())
                .and_then(|e| e.get("code").and_then(|c| c.as_str()).map(str::to_string))
        })
        .map(|c| format!(" ({c})"))
        .unwrap_or_default();
    Error::Registry(format!("{what}: HTTP {}{detail}", resp.status))
}

/// Pushes an OCI layout to a registry.
///
/// The layout is the one [`crate::oci::export_layout`] produces, so what reaches
/// the registry is exactly what `oras cp --from-oci-layout` would have pushed
/// from disk — the same blobs, the same manifest, the same digests.
pub fn push(layout: &Layout, r: &Reference) -> Res<Push> {
    let http = client(r)?;
    let mut out = Push {
        manifest_digest: layout.manifest_digest.clone(),
        ..Default::default()
    };

    // §13.5 says the registry API is reachable before anything is uploaded, and
    // a client that discovers otherwise halfway through a 4 GB push has wasted
    // the push. `GET /v2/` is the specification's own ping.
    let ping = http
        .send("GET", "/v2/", &[], None)
        .map_err(|e| Error::Transport(e.to_string()))?;
    match ping.status {
        200 => {}
        401 => return Err(challenge(&ping, "/v2/")),
        _ => return Err(status_error("/v2/", &ping)),
    }

    let manifest_hex = layout.manifest_digest.trim_start_matches("sha256:");
    for (path, bytes) in &layout.files {
        let Some(hex) = path.strip_prefix("blobs/sha256/") else {
            // `oci-layout` and `index.json` are the layout's own bookkeeping;
            // a registry keeps that state itself.
            continue;
        };
        // A layout stores the manifest among the blobs because a directory has
        // nowhere else to put it. A registry does not: it has a manifest
        // endpoint, and uploading the same bytes to both would leave a blob
        // nothing refers to.
        if hex == manifest_hex {
            continue;
        }
        let digest = format!("sha256:{hex}");
        let head = http
            .send("HEAD", &format!("/v2/{}/blobs/{digest}", r.name), &[], None)
            .map_err(|e| Error::Transport(e.to_string()))?;
        match head.status {
            200 => {
                // The registry already has these bytes. This is the dedup claim,
                // answered by the party that would know.
                out.skipped += 1;
                out.skipped_bytes += bytes.len() as u64;
                continue;
            }
            404 => {}
            401 => return Err(challenge(&head, "blob HEAD")),
            _ => return Err(status_error("blob HEAD", &head)),
        }

        let start = http
            .send("POST", &format!("/v2/{}/blobs/uploads/", r.name), &[], None)
            .map_err(|e| Error::Transport(e.to_string()))?;
        if start.status != 202 {
            return Err(status_error("starting a blob upload", &start));
        }
        let location = start.header("location").ok_or_else(|| {
            Error::Registry("a 202 with no Location: there is nowhere to upload to".into())
        })?;
        let location = upload_path(location, &digest)?;
        let put = http
            .send(
                "PUT",
                &location,
                &[(
                    "Content-Type".into(),
                    "application/octet-stream".to_string(),
                )],
                Some(bytes),
            )
            .map_err(|e| Error::Transport(e.to_string()))?;
        if put.status != 201 {
            return Err(status_error(&format!("uploading {digest}"), &put));
        }
        out.uploaded += 1;
        out.uploaded_bytes += bytes.len() as u64;
    }

    let manifest = layout
        .files
        .iter()
        .find(|(p, _)| p.ends_with(layout.manifest_digest.trim_start_matches("sha256:")))
        .map(|(_, b)| b.clone())
        .ok_or_else(|| Error::Oci("the layout holds no manifest blob".into()))?;
    let put = http
        .send(
            "PUT",
            &format!("/v2/{}/manifests/{}", r.name, r.reference),
            &[("Content-Type".into(), MANIFEST_TYPE.to_string())],
            Some(&manifest),
        )
        .map_err(|e| Error::Transport(e.to_string()))?;
    if put.status != 201 {
        return Err(status_error("putting the manifest", &put));
    }
    out.requests = http.requests.get();
    Ok(out)
}

/// The path to `PUT` a blob to, with the digest the registry needs to see.
///
/// A `Location` may be absolute or relative and may already carry a query, and
/// getting either wrong produces a 400 that says nothing useful.
fn upload_path(location: &str, digest: &str) -> Res<String> {
    let path = if let Some(rest) = location.strip_prefix("http://") {
        match rest.find('/') {
            Some(i) => rest[i..].to_string(),
            None => "/".to_string(),
        }
    } else if location.starts_with("https://") {
        return Err(Error::Transport(
            "the registry redirected the upload to https, which needs a TLS stack \
             this crate does not have"
                .into(),
        ));
    } else {
        location.to_string()
    };
    let sep = if path.contains('?') { '&' } else { '?' };
    Ok(format!("{path}{sep}digest={digest}"))
}

/// Pulls an artifact from a registry and reassembles the container.
///
/// The fetched blobs are handed to [`crate::oci::import_layout`] through the
/// same reader interface a directory layout uses, so every blob is checked
/// against the digest that named it by the code that already does that — and a
/// registry that returns the wrong bytes fails here rather than becoming a
/// model.
pub fn pull(r: &Reference) -> Res<Pull> {
    let http = client(r)?;
    let accept = (
        "Accept".to_string(),
        format!("{MANIFEST_TYPE}, {}", oci::INDEX_JSON_TYPE),
    );
    let m = http
        .send(
            "GET",
            &format!("/v2/{}/manifests/{}", r.name, r.reference),
            std::slice::from_ref(&accept),
            None,
        )
        .map_err(|e| Error::Transport(e.to_string()))?;
    match m.status {
        200 => {}
        401 => return Err(challenge(&m, "manifest GET")),
        404 => {
            return Err(Error::Registry(format!(
                "no `{}` in `{}` on {}",
                r.reference, r.name, r.host
            )))
        }
        _ => return Err(status_error("manifest GET", &m)),
    }
    let manifest_bytes = m.body.clone();
    let manifest_digest = format!(
        "sha256:{}",
        crate::sha256::hex(&crate::sha256::sha256(&manifest_bytes))
    );
    // The digest a registry reports and the digest of the bytes it sent are two
    // different things, and only the second one is evidence.
    if let Some(claimed) = m.header("docker-content-digest") {
        if claimed != manifest_digest {
            return Err(Error::Registry(format!(
                "the registry called this manifest {claimed} and sent bytes that \
                 hash to {manifest_digest}"
            )));
        }
    }
    if r.reference.starts_with("sha256:") && r.reference != manifest_digest {
        return Err(Error::Registry(format!(
            "asked for {} and received {manifest_digest}",
            r.reference
        )));
    }

    // Fetch every blob the manifest names, then let the layout importer verify
    // and reassemble them.
    let parsed = json::parse(&manifest_bytes).map_err(|e| Error::Oci(e.to_string()))?;
    let mut digests: Vec<String> = Vec::new();
    if let Some(c) = parsed
        .get("config")
        .and_then(|c| c.get("digest"))
        .and_then(|d| d.as_str())
    {
        digests.push(c.to_string());
    }
    for l in parsed
        .get("layers")
        .and_then(|l| l.as_array())
        .unwrap_or(&[])
    {
        if let Some(d) = l.get("digest").and_then(|d| d.as_str()) {
            digests.push(d.to_string());
        }
    }
    let mut blobs: Vec<(String, Vec<u8>)> = Vec::new();
    for d in &digests {
        let hex = d.strip_prefix("sha256:").ok_or_else(|| {
            Error::Oci(format!("`{d}` is not a sha256 digest; §13.5 uses sha256"))
        })?;
        let b = http
            .send("GET", &format!("/v2/{}/blobs/{d}", r.name), &[], None)
            .map_err(|e| Error::Transport(e.to_string()))?;
        if b.status != 200 {
            return Err(status_error(&format!("fetching {d}"), &b));
        }
        blobs.push((format!("blobs/sha256/{hex}"), b.body));
    }
    blobs.push((
        format!(
            "blobs/sha256/{}",
            manifest_digest.trim_start_matches("sha256:")
        ),
        manifest_bytes.clone(),
    ));
    // The two files a layout has and a registry does not: the registry keeps
    // that state as the repository and the tag, so they are written here from
    // what was asked for rather than fetched.
    blobs.push((
        "oci-layout".to_string(),
        json::object(vec![("imageLayoutVersion", json::string("1.0.0"))])
            .encode()
            .into_bytes(),
    ));
    blobs.push((
        "index.json".to_string(),
        index_json(&manifest_digest, manifest_bytes.len() as u64, &r.reference)
            .encode()
            .into_bytes(),
    ));

    let read = |path: &str| -> Option<Vec<u8>> {
        blobs
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, b)| b.clone())
    };
    let imported = oci::import_layout(&read).map_err(|e| Error::Oci(e.to_string()))?;
    Ok(Pull {
        bytes: imported.bytes,
        manifest_digest,
        layers: imported.layers,
        annotations: imported.annotations,
        requests: http.requests.get(),
        fetched_bytes: http.bytes.get(),
    })
}

fn index_json(digest: &str, size: u64, reference: &str) -> json::Value {
    let mut desc = vec![
        ("mediaType", json::string(MANIFEST_TYPE)),
        ("digest", json::string(digest.to_string())),
        ("size", json::Value::U(size)),
    ];
    if !reference.starts_with("sha256:") {
        desc.push((
            "annotations",
            json::object(vec![(
                "org.opencontainers.image.ref.name",
                json::string(reference.to_string()),
            )]),
        ));
    }
    json::object(vec![
        ("schemaVersion", json::Value::U(2)),
        ("mediaType", json::string(oci::INDEX_JSON_TYPE)),
        ("manifests", json::Value::Array(vec![json::object(desc)])),
    ])
}

/// Pushes a container: the mapping and the transfer in one call.
pub fn push_container(c: &Container, r: &Reference, opts: &oci::ExportOpts) -> Res<Push> {
    let mut opts = opts.clone();
    if opts.reference.is_none() && !r.reference.starts_with("sha256:") {
        opts.reference = Some(r.reference.clone());
    }
    let layout = oci::export_layout(c, &opts).map_err(|e| Error::Oci(e.to_string()))?;
    push(&layout, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_parse_the_way_they_are_written() {
        let r = Reference::parse("localhost:5000/models/llama:v1").unwrap();
        assert_eq!(r.host, "localhost");
        assert_eq!(r.port, 5000);
        assert_eq!(r.name, "models/llama");
        assert_eq!(r.reference, "v1");

        // No tag is `latest`, which is the registry's own default rather than
        // this client inventing one.
        assert_eq!(
            Reference::parse("reg.internal/m").unwrap(),
            Reference {
                host: "reg.internal".into(),
                port: 80,
                name: "m".into(),
                reference: "latest".into(),
            }
        );
        // A digest reference, where the `:` inside it is not a tag separator.
        let d = Reference::parse("h:5000/m@sha256:abc").unwrap();
        assert_eq!(d.reference, "sha256:abc");
        assert_eq!(d.name, "m");
    }

    #[test]
    fn a_reference_with_no_repository_is_refused() {
        for bad in ["localhost:5000", "", "https://reg/m:1"] {
            assert!(Reference::parse(bad).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn an_upload_location_keeps_its_query() {
        assert_eq!(
            upload_path("/v2/m/blobs/uploads/abc?_state=xyz", "sha256:d").unwrap(),
            "/v2/m/blobs/uploads/abc?_state=xyz&digest=sha256:d"
        );
        assert_eq!(
            upload_path("http://h:5000/v2/m/blobs/uploads/abc", "sha256:d").unwrap(),
            "/v2/m/blobs/uploads/abc?digest=sha256:d"
        );
        assert!(upload_path("https://h/v2/x", "sha256:d").is_err());
    }
}
