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
//! **Fetching a bearer token.** [`Credentials`] carries one the caller already
//! has — `Bearer` for a token, `Basic` for a username and password — and it is
//! sent on every request rather than only after a challenge, which is what a
//! client with explicit credentials should do. What is still not done is the
//! *token dance*: a registry that answers `401 WWW-Authenticate: Bearer
//! realm=…` is pointing at a token endpoint that is https on every registry
//! this could reach, so the challenge is parsed and reported with the realm it
//! named instead of being followed. "401" alone tells a user nothing about
//! which credential is missing; the realm does.

use crate::container::Container;
use crate::json;
use crate::oci::{self, Layout};
use crate::transport::{Http, Url};

/// The credential to send, if any.
///
/// Sent on every request rather than only in answer to a challenge. A client
/// that waits to be asked spends an extra round trip per blob and, worse, gives
/// the registry a chance to answer 404-for-unauthorized — which some do, and
/// which turns a permissions problem into "your model is missing".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Credentials {
    #[default]
    None,
    /// A username and password, sent as `Authorization: Basic`.
    Basic { user: String, password: String },
    /// A token the caller obtained elsewhere.
    Bearer(String),
}

impl Credentials {
    /// Parses `user:password`, which is how every registry client spells it.
    pub fn basic(spec: &str) -> Res<Credentials> {
        let (user, password) = spec.split_once(':').ok_or_else(|| {
            Error::Auth("credentials are `user:password`; there is no colon in this one".into())
        })?;
        Ok(Credentials::Basic {
            user: user.to_string(),
            password: password.to_string(),
        })
    }

    /// The `Authorization` header this credential produces, if any.
    fn header(&self) -> Option<(String, String)> {
        match self {
            Credentials::None => None,
            Credentials::Basic { user, password } => Some((
                "Authorization".into(),
                format!("Basic {}", base64(format!("{user}:{password}").as_bytes())),
            )),
            Credentials::Bearer(t) => Some(("Authorization".into(), format!("Bearer {t}"))),
        }
    }
}

/// Standard base64, for the one place HTTP needs it.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// How a push should behave beyond the bytes it sends.
#[derive(Clone, Debug, Default)]
pub struct PushOpts {
    pub creds: Credentials,
    /// Upload blobs larger than this in `PATCH` chunks of this size, instead of
    /// one `PUT`.
    ///
    /// The monolithic `PUT` the specification allows at any size is the default
    /// and is what a healthy link wants. Chunking exists so a 5 GB layer over a
    /// bad link does not start again from zero, and so a registry with a request
    /// size limit — most managed ones have one — can be pushed to at all.
    pub chunk_size: Option<usize>,
    /// Make this artifact a *referrer* of an existing manifest (§13.5's
    /// referrers API): the descriptor a registry answers `GET
    /// /v2/<name>/referrers/<digest>` with.
    pub subject: Option<String>,
    /// The `artifactType` a referring manifest declares, so a client can filter
    /// signatures from adapters without pulling either.
    pub artifact_type: Option<String>,
}

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
    /// `PATCH` requests made, when the upload was chunked.
    pub chunks: u64,
    /// The manifest this artifact was linked to, if any (§13.5's referrers).
    pub subject: Option<String>,
    /// Whether the registry acknowledged the link with `OCI-Subject`. When it
    /// did not, the fallback tag was written instead.
    pub subject_accepted: bool,
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

/// A request with the credential attached, if there is one.
fn send(
    http: &Http,
    creds: &Credentials,
    method: &str,
    path: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
) -> Res<crate::transport::Response> {
    let mut all: Vec<(String, String)> = headers.to_vec();
    if let Some(h) = creds.header() {
        all.push(h);
    }
    http.send(method, path, &all, body)
        .map_err(|e| Error::Transport(e.to_string()))
}

/// Turns a `401` into an error that says which credential is missing, and
/// whether one was even offered.
///
/// The distinction matters: "you sent nothing" and "what you sent was refused"
/// are different problems with different fixes, and a client that reports both
/// as `401` makes the user guess.
fn challenge(resp: &crate::transport::Response, what: &str, creds: &Credentials) -> Error {
    let Some(c) = resp.header("www-authenticate") else {
        return Error::Auth(format!(
            "{what}: the registry refused with 401 and stated no challenge"
        ));
    };
    // `Bearer realm="…",service="…"` — the scheme is the first token and the
    // parameters follow it, so `realm=` is not necessarily at the start of a
    // comma-separated part.
    let scheme = c.split_whitespace().next().unwrap_or("").to_string();
    let realm = c
        .split(',')
        .find_map(|p| {
            let p = p.trim();
            let p = p.strip_prefix(&scheme).map(str::trim_start).unwrap_or(p);
            p.strip_prefix("realm=").map(|r| r.trim_matches('"'))
        })
        .unwrap_or("(unstated)");
    let offered = match creds {
        Credentials::None => "no credential was sent",
        Credentials::Basic { .. } => "the username and password sent were refused",
        Credentials::Bearer(_) => "the token sent was refused",
    };
    if scheme.eq_ignore_ascii_case("basic") {
        return Error::Auth(format!(
            "{what}: the registry wants a username and password for realm \
             `{realm}`, and {offered}. Pass `--user <user>:<password>`"
        ));
    }
    Error::Auth(format!(
        "{what}: the registry wants a bearer token from `{realm}`, and {offered}. \
         That endpoint is https on every registry this could reach and this build \
         has no TLS, so the token cannot be fetched here — pass one with `--token`, \
         or push through something that terminates TLS"
    ))
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
    push_with(layout, r, &PushOpts::default())
}

/// [`push`], with credentials, chunking and a referrers `subject`.
pub fn push_with(layout: &Layout, r: &Reference, opts: &PushOpts) -> Res<Push> {
    let http = client(r)?;
    let creds = &opts.creds;
    let mut out = Push {
        manifest_digest: layout.manifest_digest.clone(),
        ..Default::default()
    };

    // §13.5 says the registry API is reachable before anything is uploaded, and
    // a client that discovers otherwise halfway through a 4 GB push has wasted
    // the push. `GET /v2/` is the specification's own ping.
    let ping = send(&http, creds, "GET", "/v2/", &[], None)?;
    match ping.status {
        200 => {}
        401 => return Err(challenge(&ping, "/v2/", creds)),
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
        let head = send(
            &http,
            creds,
            "HEAD",
            &format!("/v2/{}/blobs/{digest}", r.name),
            &[],
            None,
        )?;
        match head.status {
            200 => {
                // The registry already has these bytes. This is the dedup claim,
                // answered by the party that would know.
                out.skipped += 1;
                out.skipped_bytes += bytes.len() as u64;
                continue;
            }
            404 => {}
            401 => return Err(challenge(&head, "blob HEAD", creds)),
            _ => return Err(status_error("blob HEAD", &head)),
        }

        let start = send(
            &http,
            creds,
            "POST",
            &format!("/v2/{}/blobs/uploads/", r.name),
            &[],
            None,
        )?;
        if start.status != 202 {
            return Err(status_error("starting a blob upload", &start));
        }
        let mut location = start
            .header("location")
            .ok_or_else(|| {
                Error::Registry("a 202 with no Location: there is nowhere to upload to".into())
            })?
            .to_string();

        // Chunked, when the caller asked for it and the blob is worth it. Each
        // `PATCH` states the byte range it carries and the registry answers with
        // the `Location` to continue at — which may move, so it is read from the
        // response rather than assumed.
        let mut tail: &[u8] = bytes;
        if let Some(chunk) = opts.chunk_size.filter(|c| *c > 0 && bytes.len() > *c) {
            let mut at = 0usize;
            while at + chunk < bytes.len() {
                let end = at + chunk;
                let patch = send(
                    &http,
                    creds,
                    "PATCH",
                    &plain_path(&location)?,
                    &[
                        (
                            "Content-Type".into(),
                            "application/octet-stream".to_string(),
                        ),
                        ("Content-Range".into(), format!("{at}-{}", end - 1)),
                    ],
                    Some(&bytes[at..end]),
                )?;
                if patch.status != 202 {
                    return Err(status_error(
                        &format!("uploading bytes {at}..{end} of {digest}"),
                        &patch,
                    ));
                }
                if let Some(next) = patch.header("location") {
                    location = next.to_string();
                }
                out.chunks += 1;
                at = end;
            }
            tail = &bytes[at..];
        }

        // The closing `PUT` carries whatever is left — the whole blob when the
        // upload was monolithic, the last chunk when it was not.
        let put = send(
            &http,
            creds,
            "PUT",
            &upload_path(&location, &digest)?,
            &[(
                "Content-Type".into(),
                "application/octet-stream".to_string(),
            )],
            Some(tail),
        )?;
        if put.status != 201 {
            return Err(status_error(&format!("uploading {digest}"), &put));
        }
        out.uploaded += 1;
        out.uploaded_bytes += bytes.len() as u64;
    }

    let mut manifest = layout
        .files
        .iter()
        .find(|(p, _)| p.ends_with(layout.manifest_digest.trim_start_matches("sha256:")))
        .map(|(_, b)| b.clone())
        .ok_or_else(|| Error::Oci("the layout holds no manifest blob".into()))?;

    // §13.5's referrers API: an artifact that names a `subject` is *linked* to
    // it by the registry, which is how a signature, an adapter or an evaluation
    // is found from the model rather than by a naming convention two tools have
    // to agree on out of band.
    if let Some(subject) = &opts.subject {
        manifest = with_subject(
            &manifest,
            subject,
            opts.artifact_type.as_deref(),
            &http,
            r,
            creds,
        )?;
        // The layout's digest names the manifest *before* the subject was added,
        // and the artifact is addressable by what was actually written. Bare
        // hex, like the layout's, because the caller prefixes it.
        out.manifest_digest = crate::sha256::hex(&crate::sha256::sha256(&manifest));
        out.subject = Some(subject.clone());
    }

    let put = send(
        &http,
        creds,
        "PUT",
        &format!("/v2/{}/manifests/{}", r.name, r.reference),
        &[("Content-Type".into(), MANIFEST_TYPE.to_string())],
        Some(&manifest),
    )?;
    if put.status != 201 {
        return Err(status_error("putting the manifest", &put));
    }
    // A registry that understands referrers echoes the subject it recorded. One
    // that does not says nothing, and the fallback tag below is what makes the
    // link findable there — the specification's own compatibility path, not a
    // convention invented here.
    if opts.subject.is_some() {
        out.subject_accepted = put.header("oci-subject").is_some();
        if !out.subject_accepted {
            let tag = fallback_tag(opts.subject.as_deref().unwrap_or_default())?;
            let index = referrers_index(&[(
                format!("sha256:{}", out.manifest_digest),
                manifest.len() as u64,
                opts.artifact_type.clone().unwrap_or_default(),
            )]);
            let bytes = index.encode().into_bytes();
            let put = send(
                &http,
                creds,
                "PUT",
                &format!("/v2/{}/manifests/{tag}", r.name),
                &[("Content-Type".into(), oci::INDEX_JSON_TYPE.to_string())],
                Some(&bytes),
            )?;
            if put.status != 201 {
                return Err(status_error("putting the referrers fallback tag", &put));
            }
        }
    }
    out.requests = http.requests.get();
    Ok(out)
}

/// Adds `subject` (and an `artifactType`) to a manifest, checking first that the
/// subject is actually there.
///
/// Pointing at a manifest the registry does not have produces a dangling link
/// that only shows up when somebody follows it, so it is checked before the
/// artifact is written rather than after.
fn with_subject(
    manifest: &[u8],
    subject: &str,
    artifact_type: Option<&str>,
    http: &Http,
    r: &Reference,
    creds: &Credentials,
) -> Res<Vec<u8>> {
    let head = send(
        http,
        creds,
        "HEAD",
        &format!("/v2/{}/manifests/{subject}", r.name),
        &[(
            "Accept".into(),
            format!("{MANIFEST_TYPE}, {}", oci::INDEX_JSON_TYPE),
        )],
        None,
    )?;
    if head.status == 404 {
        return Err(Error::Registry(format!(
            "the subject {subject} is not in `{}`; a referrer pointing at a \
             manifest the registry does not have is a link nobody can follow",
            r.name
        )));
    }
    if head.status != 200 {
        return Err(status_error("checking the subject", &head));
    }
    let size: u64 = head
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let media = head
        .header("content-type")
        .unwrap_or(MANIFEST_TYPE)
        .to_string();

    let mut doc = json::parse(manifest).map_err(|e| Error::Oci(e.to_string()))?;
    let json::Value::Object(map) = &mut doc else {
        return Err(Error::Oci("the manifest is not a JSON object".into()));
    };
    map.insert(
        "subject".to_string(),
        json::object(vec![
            ("mediaType", json::string(media)),
            ("digest", json::string(subject.to_string())),
            ("size", json::Value::U(size)),
        ]),
    );
    if let Some(t) = artifact_type {
        map.insert("artifactType".to_string(), json::string(t.to_string()));
    }
    Ok(doc.encode().into_bytes())
}

/// The referrers fallback tag of the distribution specification: the subject's
/// digest with its separator replaced, because a tag cannot contain a colon.
fn fallback_tag(subject: &str) -> Res<String> {
    let (algo, hex) = subject.split_once(':').ok_or_else(|| {
        Error::Registry(format!("`{subject}` is not an algorithm-prefixed digest"))
    })?;
    Ok(format!("{algo}-{hex}"))
}

/// An OCI image index of referring descriptors.
fn referrers_index(entries: &[(String, u64, String)]) -> json::Value {
    json::object(vec![
        ("schemaVersion", json::Value::U(2)),
        ("mediaType", json::string(oci::INDEX_JSON_TYPE)),
        (
            "manifests",
            json::Value::Array(
                entries
                    .iter()
                    .map(|(digest, size, artifact)| {
                        let mut d = vec![
                            ("mediaType", json::string(MANIFEST_TYPE)),
                            ("digest", json::string(digest.clone())),
                            ("size", json::Value::U(*size)),
                        ];
                        if !artifact.is_empty() {
                            d.push(("artifactType", json::string(artifact.clone())));
                        }
                        json::object(d)
                    })
                    .collect(),
            ),
        ),
    ])
}

/// One artifact that refers to another.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Referrer {
    pub digest: String,
    pub size: u64,
    pub artifact_type: String,
}

/// Lists what refers to a manifest (§13.5).
///
/// The referrers endpoint first, then the fallback tag for a registry that does
/// not implement it — which the distribution specification defines precisely so
/// that a client does not have to choose between the two ecosystems.
pub fn referrers(r: &Reference, subject: &str, creds: &Credentials) -> Res<Vec<Referrer>> {
    let http = client(r)?;
    let resp = send(
        &http,
        creds,
        "GET",
        &format!("/v2/{}/referrers/{subject}", r.name),
        &[],
        None,
    )?;
    let body = match resp.status {
        200 => resp.body,
        401 => return Err(challenge(&resp, "referrers", creds)),
        404 | 405 | 501 => {
            // The fallback. An absent tag means no referrers, which is a
            // different answer from "this registry cannot tell you".
            let tag = fallback_tag(subject)?;
            let f = send(
                &http,
                creds,
                "GET",
                &format!("/v2/{}/manifests/{tag}", r.name),
                &[("Accept".into(), oci::INDEX_JSON_TYPE.to_string())],
                None,
            )?;
            match f.status {
                200 => f.body,
                404 => return Ok(Vec::new()),
                _ => return Err(status_error("the referrers fallback tag", &f)),
            }
        }
        _ => return Err(status_error("referrers", &resp)),
    };
    let doc = json::parse(&body).map_err(|e| Error::Oci(e.to_string()))?;
    let mut out = Vec::new();
    for m in doc
        .get("manifests")
        .and_then(|m| m.as_array())
        .unwrap_or(&[])
    {
        let Some(digest) = m.get("digest").and_then(|d| d.as_str()) else {
            continue;
        };
        out.push(Referrer {
            digest: digest.to_string(),
            size: m.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
            artifact_type: m
                .get("artifactType")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    Ok(out)
}

/// The path to `PUT` a blob to, with the digest the registry needs to see.
///
/// A `Location` may be absolute or relative and may already carry a query, and
/// getting either wrong produces a 400 that says nothing useful.
fn upload_path(location: &str, digest: &str) -> Res<String> {
    let path = plain_path(location)?;
    let sep = if path.contains('?') { '&' } else { '?' };
    Ok(format!("{path}{sep}digest={digest}"))
}

/// The path part of a `Location`, absolute or relative.
fn plain_path(location: &str) -> Res<String> {
    if let Some(rest) = location.strip_prefix("http://") {
        return Ok(match rest.find('/') {
            Some(i) => rest[i..].to_string(),
            None => "/".to_string(),
        });
    }
    if location.starts_with("https://") {
        return Err(Error::Transport(
            "the registry redirected the upload to https, which needs a TLS stack \
             this crate does not have"
                .into(),
        ));
    }
    Ok(location.to_string())
}

/// Pulls an artifact from a registry and reassembles the container.
///
/// The fetched blobs are handed to [`crate::oci::import_layout`] through the
/// same reader interface a directory layout uses, so every blob is checked
/// against the digest that named it by the code that already does that — and a
/// registry that returns the wrong bytes fails here rather than becoming a
/// model.
pub fn pull(r: &Reference) -> Res<Pull> {
    pull_with(r, &Credentials::None)
}

/// [`pull`], with a credential.
pub fn pull_with(r: &Reference, creds: &Credentials) -> Res<Pull> {
    let http = client(r)?;
    let accept = (
        "Accept".to_string(),
        format!("{MANIFEST_TYPE}, {}", oci::INDEX_JSON_TYPE),
    );
    let m = send(
        &http,
        creds,
        "GET",
        &format!("/v2/{}/manifests/{}", r.name, r.reference),
        std::slice::from_ref(&accept),
        None,
    )?;
    match m.status {
        200 => {}
        401 => return Err(challenge(&m, "manifest GET", creds)),
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
        let b = send(
            &http,
            creds,
            "GET",
            &format!("/v2/{}/blobs/{d}", r.name),
            &[],
            None,
        )?;
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
    push_container_with(c, r, opts, &PushOpts::default())
}

/// [`push_container`], with credentials, chunking and a referrers `subject`.
pub fn push_container_with(
    c: &Container,
    r: &Reference,
    opts: &oci::ExportOpts,
    push_opts: &PushOpts,
) -> Res<Push> {
    let mut opts = opts.clone();
    if opts.reference.is_none() && !r.reference.starts_with("sha256:") {
        opts.reference = Some(r.reference.clone());
    }
    let layout = oci::export_layout(c, &opts).map_err(|e| Error::Oci(e.to_string()))?;
    push_with(&layout, r, push_opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_credentials_are_the_header_http_defines() {
        // RFC 7617's example, so the encoder is checked against something
        // outside this crate rather than against itself.
        let c = Credentials::basic("Aladdin:open sesame").unwrap();
        let (k, v) = c.header().unwrap();
        assert_eq!(k, "Authorization");
        assert_eq!(v, "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==");
        // Every padding case, since that is where a hand-written base64 goes
        // wrong.
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");

        assert_eq!(
            Credentials::Bearer("t0k".into()).header().unwrap().1,
            "Bearer t0k"
        );
        assert!(Credentials::None.header().is_none());
        // A credential with no colon is a usage error rather than a username
        // with an empty password.
        assert!(Credentials::basic("nocolon").is_err());
        // A password may contain colons; only the first one separates.
        let c = Credentials::basic("u:a:b").unwrap();
        assert!(matches!(c, Credentials::Basic { ref password, .. } if password == "a:b"));
    }

    #[test]
    fn the_referrers_fallback_tag_is_the_one_the_specification_names() {
        // A tag cannot contain a colon, so the digest's separator becomes a
        // dash. Getting this wrong means writing a link no other client finds.
        assert_eq!(fallback_tag("sha256:abc123").unwrap(), "sha256-abc123");
        assert!(fallback_tag("abc123").is_err());
    }

    #[test]
    fn an_upload_location_is_followed_wherever_it_points() {
        // Relative, absolute, and one that already carries a query — all three
        // are what registries actually send, and getting any of them wrong
        // produces a 400 that says nothing useful.
        assert_eq!(
            upload_path("/v2/m/blobs/uploads/abc", "sha256:d").unwrap(),
            "/v2/m/blobs/uploads/abc?digest=sha256:d"
        );
        assert_eq!(
            upload_path("http://reg:5000/v2/m/blobs/uploads/abc", "sha256:d").unwrap(),
            "/v2/m/blobs/uploads/abc?digest=sha256:d"
        );
        assert_eq!(
            upload_path("/v2/m/blobs/uploads/abc?_state=xyz", "sha256:d").unwrap(),
            "/v2/m/blobs/uploads/abc?_state=xyz&digest=sha256:d"
        );
        assert_eq!(plain_path("/a/b?c=d").unwrap(), "/a/b?c=d");
        // https is refused rather than downgraded, here as everywhere.
        assert!(upload_path("https://reg/v2/m/blobs/uploads/abc", "sha256:d").is_err());
    }

    #[test]
    fn a_401_says_which_credential_is_missing_and_whether_one_was_offered() {
        let resp = |c: &str| crate::transport::Response {
            status: 401,
            headers: vec![("www-authenticate".into(), c.into())],
            body: Vec::new(),
        };
        let basic = resp(r#"Basic realm="https://auth.example/token",service="reg""#);
        let e = challenge(&basic, "/v2/", &Credentials::None).to_string();
        assert!(e.contains("username and password"), "{e}");
        assert!(e.contains("https://auth.example/token"), "{e}");
        assert!(e.contains("no credential was sent"), "{e}");

        let e = challenge(
            &basic,
            "/v2/",
            &Credentials::Basic {
                user: "a".into(),
                password: "b".into(),
            },
        )
        .to_string();
        assert!(e.contains("were refused"), "{e}");

        let bearer = resp(r#"Bearer realm="https://auth.example/token",scope="repo:m:pull""#);
        let e = challenge(&bearer, "manifest GET", &Credentials::None).to_string();
        assert!(e.contains("bearer token"), "{e}");
        assert!(e.contains("https://auth.example/token"), "{e}");

        // A 401 with no challenge at all is still an answer, and a different one.
        let bare = crate::transport::Response {
            status: 401,
            headers: Vec::new(),
            body: Vec::new(),
        };
        let e = challenge(&bare, "/v2/", &Credentials::None).to_string();
        assert!(e.contains("stated no challenge"), "{e}");
    }

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
