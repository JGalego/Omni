//! §13.4.3 — an object server, read-only.
//!
//! §13.4 describes two ways to serve a model, and says a hub SHOULD do both:
//! the **pack**, range-read by a client that knows the index, and **per-object
//! URLs**, one immutable object per path, which is what makes a CDN's cache hit
//! across every model that shares a tokenizer or a base layer. This serves both,
//! plus the `.omni.idx` sidecar of §13.4.1, from one container.
//!
//! It is deliberately small, and the reasons are the same ones that shape the
//! rest of this crate:
//!
//! * **Read-only.** There is no path that writes, so there is no path that can
//!   be tricked into writing. Content addressing makes that costless: an
//!   immutable object cannot be updated in place, only added.
//! * **No path traversal, because there are no paths.** A request either names a
//!   digest, or it names one of three fixed routes. Nothing is joined to a
//!   filesystem path, so there is nothing to escape from.
//! * **Bounded.** Request lines and headers are capped, and a body is never
//!   read — this answers `GET` and `HEAD` and refuses every other method.
//!
//! What it is not: a production server. There is no TLS (see [`crate::transport`]
//! for why), no authentication, no compression negotiation, a thread per
//! connection rather than an event loop, and no attempt at HTTP/2. It exists so the
//! transport claims in §13.4 can be exercised against something that speaks the
//! protocol rather than against a mock — and because a format whose whole
//! premise is content-addressed objects should be able to hand you one.

use crate::container::{oflags, otype, Container, Digest};

/// The media types of §14.7, which are what a client uses to decide how to read
/// what it just fetched.
pub const CONTAINER_TYPE: &str = "application/vnd.omni.container.v1";
pub const OBJECT_TYPE: &str = "application/vnd.omni.object.v1+cbor";
pub const CHUNK_TYPE: &str = "application/vnd.omni.chunk.v1";
pub const INDEX_TYPE: &str = "application/vnd.omni.index.v1";

/// Longest request line accepted. A digest path is under 80 bytes; 8 KiB is
/// generous and finite, which is the property that matters.
const MAX_LINE: usize = 8 << 10;
/// Most header lines read before giving up on a request.
const MAX_HEADERS: usize = 64;

#[derive(Debug)]
pub enum Error {
    Io(String),
    Bind(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(m) => write!(f, "serve: {m}"),
            Error::Bind(m) => write!(f, "serve: cannot listen: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

/// What a request asked for, once parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// `/` — a short listing, so a human who opens the URL learns what is here.
    Root,
    /// `/model.omni` — the pack itself, range-readable.
    Pack,
    /// `/model.omni.idx` — the detached index of §13.4.1.
    Sidecar,
    /// `/objects/<prefix>-<hex>` — one immutable object (§13.4.3).
    Object(Digest),
    /// A well-formed request for something that is not here.
    NotFound,
    /// A method other than GET or HEAD.
    BadMethod,
    /// Not a request this server can parse at all.
    Bad,
}

/// The counters that make a claim about a server checkable.
#[derive(Debug, Default)]
pub struct Stats {
    pub requests: std::sync::atomic::AtomicU64,
    pub bytes: std::sync::atomic::AtomicU64,
    pub not_found: std::sync::atomic::AtomicU64,
}

impl Stats {
    /// Counted when the request is *parsed*, before its response is written.
    ///
    /// The order is what makes these numbers usable. A counter bumped after the
    /// write can be read by a client that already holds the response, so "the
    /// server saw my request" would be unobservable and any assertion about the
    /// count would be a race. Counting first makes them a sound lower bound on
    /// what has been answered.
    fn count_request(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.requests.fetch_add(1, Relaxed);
    }

    fn count_bytes(&self, bytes: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.bytes.fetch_add(bytes, Relaxed);
    }

    pub fn read(&self) -> (u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.requests.load(Relaxed),
            self.bytes.load(Relaxed),
            self.not_found.load(Relaxed),
        )
    }
}

/// A container, served.
pub struct Server {
    container: Container,
    sidecar: Vec<u8>,
    /// The path the pack is served at, which is also what the sidecar's path is
    /// derived from.
    name: String,
    listener: std::net::TcpListener,
    /// Bytes per second this server will write, or `0` for as fast as the
    /// socket takes them.
    ///
    /// A throttle is not a feature of a model server; it is a *measuring
    /// instrument*. §13's claim is that a container's layout makes the first
    /// tensor cheap to reach, and on a loopback socket every layout is
    /// instantaneous — so the difference between reading three ranges and
    /// reading a whole file is invisible exactly where it is supposed to
    /// matter. Rate-limiting the writes restores the only variable the claim is
    /// about: bytes.
    throttle: u64,
    pub stats: Stats,
}

impl Server {
    /// Binds to `addr` (e.g. `127.0.0.1:0` for an ephemeral port) and prepares
    /// the sidecar up front: it is derived from the container and immutable, so
    /// computing it per request would be work with no purpose.
    pub fn bind(addr: &str, container: Container, name: &str) -> Res<Server> {
        let listener = std::net::TcpListener::bind(addr).map_err(|e| Error::Bind(e.to_string()))?;
        Ok(Server {
            sidecar: crate::transport::sidecar(&container),
            container,
            name: name.to_string(),
            listener,
            throttle: 0,
            stats: Stats::default(),
        })
    }

    /// Limits the server to `bytes_per_second`, for the measurement described on
    /// [`Server::throttle`]. Zero removes the limit.
    pub fn throttled(mut self, bytes_per_second: u64) -> Server {
        self.throttle = bytes_per_second;
        self
    }

    pub fn rate(&self) -> u64 {
        self.throttle
    }

    pub fn port(&self) -> u16 {
        self.listener.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/{}", self.port(), self.name)
    }

    /// Serves until `stop` returns true between accepts.
    ///
    /// One thread per connection. Serving them in sequence instead would be
    /// smaller code and a worse server: a client that keeps a connection open
    /// while opening a second one — which is what any client fetching several
    /// objects does — would wait for itself forever.
    ///
    /// The predicate is checked between accepts rather than during one, so a
    /// caller that wants to stop a blocked accept connects to itself once. That
    /// is the whole shutdown story, and it is enough for a tool that runs in the
    /// foreground until interrupted.
    pub fn serve_while(self: &std::sync::Arc<Self>, stop: &dyn Fn() -> bool) -> Res<()> {
        for conn in self.listener.incoming() {
            if stop() {
                return Ok(());
            }
            let Ok(conn) = conn else { continue };
            let me = self.clone();
            // One misbehaving client must not take the server down with it, so a
            // connection error ends that connection and nothing else.
            std::thread::spawn(move || {
                let _ = me.serve_connection(conn);
            });
        }
        Ok(())
    }

    /// Handles one connection, keeping it alive for as many requests as the
    /// client sends. Returns how many it answered.
    fn serve_connection(&self, conn: std::net::TcpStream) -> Res<usize> {
        use std::io::BufReader;
        conn.set_nodelay(true).ok();
        let mut reader = BufReader::new(conn.try_clone().map_err(|e| Error::Io(e.to_string()))?);
        let mut conn = conn;
        let mut answered = 0usize;
        loop {
            let mut line = String::new();
            // A request line longer than the cap ends the connection rather than
            // growing a buffer for it.
            let n = read_line_bounded(&mut reader, &mut line, MAX_LINE)?;
            if n == 0 {
                return Ok(answered);
            }
            let (route, head_only) = parse_request_line(&line, &self.name);
            self.stats.count_request();
            let mut range = None;
            let mut headers = 0;
            loop {
                let mut h = String::new();
                if read_line_bounded(&mut reader, &mut h, MAX_LINE)? == 0 {
                    return Ok(answered);
                }
                let h = h.trim_end();
                if h.is_empty() {
                    break;
                }
                headers += 1;
                if headers > MAX_HEADERS {
                    self.write_status(&mut conn, 431, "Request Header Fields Too Large")?;
                    return Ok(answered);
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("range:") {
                    range = parse_range(v.trim());
                }
                // A request with a body is not something this server answers; a
                // GET with Content-Length would leave unread bytes on the wire
                // and desynchronise the connection, so it ends here.
                if h.to_ascii_lowercase().starts_with("content-length:")
                    && h.trim_end().rsplit(':').next().map(str::trim) != Some("0")
                {
                    self.write_status(&mut conn, 400, "Bad Request")?;
                    return Ok(answered);
                }
            }
            self.respond(&mut conn, route, range, head_only)?;
            answered += 1;
            // The pack and the objects are immutable, so keep-alive costs nothing
            // and saves a round trip per object.
            if answered > 10_000 {
                return Ok(answered);
            }
        }
    }

    fn respond(
        &self,
        out: &mut std::net::TcpStream,
        route: Route,
        range: Option<(u64, Option<u64>)>,
        head_only: bool,
    ) -> Res<()> {
        use std::sync::atomic::Ordering::Relaxed;
        match route {
            Route::Bad => self.write_status(out, 400, "Bad Request"),
            Route::BadMethod => self.write_status(out, 405, "Method Not Allowed"),
            Route::NotFound => {
                self.stats.not_found.fetch_add(1, Relaxed);
                self.write_status(out, 404, "Not Found")
            }
            Route::Root => {
                let body = self.listing().into_bytes();
                self.write_body(out, &body, "text/plain; charset=utf-8", None, head_only)
            }
            Route::Sidecar => self.write_body(out, &self.sidecar, INDEX_TYPE, range, head_only),
            Route::Pack => {
                self.write_body(out, &self.container.bytes, CONTAINER_TYPE, range, head_only)
            }
            Route::Object(d) => {
                // Absent and external are both "not here": §01.4 makes a partial
                // store legal, and a 404 is how that reads over HTTP.
                let Some(e) = self.container.find(&d) else {
                    self.stats.not_found.fetch_add(1, Relaxed);
                    return self.write_status(out, 404, "Not Found");
                };
                if e.oflags & oflags::EXTERNAL != 0 {
                    self.stats.not_found.fetch_add(1, Relaxed);
                    return self.write_status(out, 404, "Not Found");
                }
                // The *logical* bytes: a per-object URL serves the object, and
                // the object is what its digest covers (§03.5.2). Serving the
                // stored form would hand back bytes that hash to nothing.
                let Ok(body) = self.container.read(&d) else {
                    return self.write_status(out, 500, "Internal Server Error");
                };
                let ct = if e.otype == otype::BLOB {
                    CHUNK_TYPE
                } else {
                    OBJECT_TYPE
                };
                self.write_body(out, &body, ct, range, head_only)
            }
        }
    }

    fn listing(&self) -> String {
        let c = &self.container;
        let comp = crate::transport::Completeness::of(c);
        format!(
            "OMNI object server\n\
             \n\
             GET /{name}                    the pack, {size} bytes, range-readable\n\
             GET /{name}.idx                the detached object index (§13.4.1)\n\
             GET /objects/<digest>          one immutable object (§13.4.3)\n\
             \n\
             root      {prefix}:{root}\n\
             hash      {hash}\n\
             objects   {objects}{partial}\n",
            name = self.name,
            size = c.bytes.len(),
            prefix = c.header.hash.prefix(),
            root = crate::sha256::hex(&c.header.root_digest),
            hash = c.header.hash.name(),
            objects = c.index.len(),
            partial = if comp.is_complete() {
                String::new()
            } else {
                format!(
                    " ({} present, {:.1} % complete)",
                    comp.local,
                    comp.percent()
                )
            }
        )
    }

    fn write_status(&self, out: &mut std::net::TcpStream, code: u16, reason: &str) -> Res<()> {
        use std::io::Write;
        let head = format!(
            "HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\n\
             Connection: keep-alive\r\n\r\n"
        );
        out.write_all(head.as_bytes())
            .map_err(|e| Error::Io(e.to_string()))
    }

    fn write_body(
        &self,
        out: &mut std::net::TcpStream,
        body: &[u8],
        content_type: &str,
        range: Option<(u64, Option<u64>)>,
        head_only: bool,
    ) -> Res<()> {
        use std::io::Write;
        let total = body.len() as u64;
        let (slice, code, content_range) = match range {
            None => (body, 200, None),
            Some((start, end)) => {
                if start >= total {
                    // RFC 9110: an unsatisfiable range is 416, and the response
                    // says how large the thing actually is.
                    let head = format!(
                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{total}\r\n\
                         Content-Length: 0\r\nConnection: keep-alive\r\n\r\n"
                    );
                    return out
                        .write_all(head.as_bytes())
                        .map_err(|e| Error::Io(e.to_string()));
                }
                let last = end.unwrap_or(total - 1).min(total - 1);
                let s = &body[start as usize..=last as usize];
                (s, 206, Some(format!("bytes {start}-{last}/{total}")))
            }
        };
        let mut head = format!(
            "HTTP/1.1 {code} {}\r\nContent-Type: {content_type}\r\n\
             Content-Length: {}\r\nAccept-Ranges: bytes\r\n",
            if code == 206 { "Partial Content" } else { "OK" },
            slice.len()
        );
        if let Some(cr) = content_range {
            head.push_str(&format!("Content-Range: {cr}\r\n"));
        }
        // Every URL this server answers is derived from a digest or from an
        // immutable pack, so §13.4.2's cache advice is unconditionally correct.
        head.push_str("Cache-Control: public, max-age=31536000, immutable\r\n");
        head.push_str("Connection: keep-alive\r\n\r\n");
        // Before the write, for the same reason the request is counted before
        // the response: a number a client cannot yet observe is not a number.
        self.stats
            .count_bytes(if head_only { 0 } else { slice.len() as u64 });
        out.write_all(head.as_bytes())
            .map_err(|e| Error::Io(e.to_string()))?;
        if !head_only {
            self.write_paced(out, slice)?;
        }
        out.flush().map_err(|e| Error::Io(e.to_string()))?;
        Ok(())
    }

    /// Writes a response body, at the configured rate.
    ///
    /// The pacing is deliberately crude — a fixed slice per tick, and a sleep
    /// for whatever is left of the tick — because a precise shaper would be
    /// measuring its own scheduler. What it has to be is *the same* for every
    /// response, so that two ways of reading the same container differ only in
    /// how many bytes they ask for.
    fn write_paced(&self, out: &mut dyn std::io::Write, body: &[u8]) -> Res<()> {
        if self.throttle == 0 {
            return out.write_all(body).map_err(|e| Error::Io(e.to_string()));
        }
        // A twentieth of a second's worth per tick: small enough that a short
        // response is still paced, large enough that a long one is not a syscall
        // storm.
        let slice = (self.throttle / 20).max(1) as usize;
        for part in body.chunks(slice) {
            let t0 = std::time::Instant::now();
            out.write_all(part).map_err(|e| Error::Io(e.to_string()))?;
            out.flush().map_err(|e| Error::Io(e.to_string()))?;
            let owed = std::time::Duration::from_secs_f64(part.len() as f64 / self.throttle as f64);
            if let Some(left) = owed.checked_sub(t0.elapsed()) {
                std::thread::sleep(left);
            }
        }
        Ok(())
    }
}

/// Reads a line, refusing to grow past `max`. Returns bytes consumed, 0 at end
/// of stream.
fn read_line_bounded<R: std::io::BufRead>(r: &mut R, out: &mut String, max: usize) -> Res<usize> {
    let mut total = 0;
    loop {
        let mut byte = [0u8; 1];
        match std::io::Read::read(r, &mut byte) {
            Ok(0) => return Ok(total),
            Ok(_) => {
                total += 1;
                if total > max {
                    return Err(Error::Io("a request line past the bound".into()));
                }
                out.push(byte[0] as char);
                if byte[0] == b'\n' {
                    return Ok(total);
                }
            }
            Err(e) => return Err(Error::Io(e.to_string())),
        }
    }
}

/// `GET /path HTTP/1.1` → the route it names, and whether the body is wanted.
fn parse_request_line(line: &str, name: &str) -> (Route, bool) {
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return (Route::Bad, false);
    };
    let head_only = match method {
        "GET" => false,
        "HEAD" => true,
        _ => return (Route::BadMethod, false),
    };
    // Query strings and fragments name nothing here: every URL is content
    // addressed, so a parameter could only ever be noise.
    let path = target.split(['?', '#']).next().unwrap_or("");
    let route = if path == "/" {
        Route::Root
    } else if path == format!("/{name}") {
        Route::Pack
    } else if path == format!("/{name}.idx") {
        Route::Sidecar
    } else if let Some(rest) = path.strip_prefix("/objects/") {
        match parse_digest(rest) {
            Some(d) => Route::Object(d),
            // A path under /objects/ that is not a digest cannot name an object:
            // there is nothing else in that namespace.
            None => Route::NotFound,
        }
    } else {
        Route::NotFound
    };
    (route, head_only)
}

/// `b3-<64 hex>` or `b3:<64 hex>`, or bare hex. The separator varies across
/// tools; the digest does not.
pub fn parse_digest(s: &str) -> Option<Digest> {
    let hex = match s.split_once(['-', ':']) {
        Some((prefix, rest)) => {
            if !matches!(prefix, "b3" | "blake3" | "sha2" | "sha256") {
                return None;
            }
            rest
        }
        None => s,
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// `bytes=a-b`, `bytes=a-`, or `bytes=-n` (a suffix range).
fn parse_range(v: &str) -> Option<(u64, Option<u64>)> {
    let spec = v.strip_prefix("bytes=")?;
    // Multiple ranges are legal HTTP and would need a multipart body; a single
    // range covers what §13.4 asks for, and refusing the rest is better than
    // answering the first and calling it the answer.
    if spec.contains(',') {
        return None;
    }
    let (a, b) = spec.split_once('-')?;
    match (a.trim(), b.trim()) {
        ("", n) => {
            // A suffix range needs the total, which the caller has; encode it as
            // "from here to the end" is wrong, so it is refused and the client
            // falls back to an absolute range. `transport::Http` never sends one.
            let _ = n;
            None
        }
        (start, "") => Some((start.parse().ok()?, None)),
        (start, end) => Some((start.parse().ok()?, Some(end.parse().ok()?))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{pack, PackOptions};
    use crate::model::{ModelBuilder, TensorSpec};
    use crate::store::Store;
    use crate::transport::HttpStore;
    use std::sync::Arc;

    fn toy() -> Container {
        let (objs, root) = ModelBuilder::new("test/serve")
            .tensor(TensorSpec {
                name: "w".into(),
                shape: vec![256],
                dtype: crate::dtype::DType::F32,
                axes: None,
                semantic: "weight".into(),
                data: (0..1024u32).map(|i| (i % 251) as u8).collect(),
                layout: None,
            })
            .build();
        Container::open(pack(&objs, &root, &PackOptions::default()).unwrap()).unwrap()
    }

    /// Starts a server on its own thread and hands back the URL. The `Arc` is so
    /// the test can read the counters the server is updating.
    fn running(c: Container) -> (Arc<Server>, String) {
        let s = Arc::new(Server::bind("127.0.0.1:0", c, "model.omni").unwrap());
        let url = s.url();
        let t = s.clone();
        std::thread::spawn(move || {
            let _ = t.serve_while(&|| false);
        });
        (s, url)
    }

    #[test]
    fn a_request_line_names_a_route_or_nothing() {
        let r = |line: &str| parse_request_line(line, "model.omni").0;
        assert_eq!(r("GET / HTTP/1.1\r\n"), Route::Root);
        assert_eq!(r("GET /model.omni HTTP/1.1\r\n"), Route::Pack);
        assert_eq!(r("GET /model.omni.idx HTTP/1.1\r\n"), Route::Sidecar);
        assert_eq!(r("HEAD /model.omni HTTP/1.1\r\n"), Route::Pack);
        assert!(parse_request_line("HEAD / HTTP/1.1\r\n", "model.omni").1);
        assert!(!parse_request_line("GET / HTTP/1.1\r\n", "model.omni").1);
        // A query string is noise on a content-addressed URL, not a different
        // resource.
        assert_eq!(r("GET /model.omni?v=2 HTTP/1.1\r\n"), Route::Pack);

        let hex = "0".repeat(64);
        assert_eq!(
            r(&format!("GET /objects/b3-{hex} HTTP/1.1\r\n")),
            Route::Object([0u8; 32])
        );
        assert_eq!(
            r(&format!("GET /objects/{hex} HTTP/1.1\r\n")),
            Route::Object([0u8; 32])
        );

        // Everything else is not here, and nothing is joined to a path.
        for bad in [
            "GET /objects/../../etc/passwd HTTP/1.1\r\n",
            "GET /../../etc/passwd HTTP/1.1\r\n",
            "GET /objects/ HTTP/1.1\r\n",
            "GET /objects/zz HTTP/1.1\r\n",
            "GET /model.omni.idx.idx HTTP/1.1\r\n",
            "GET /other.omni HTTP/1.1\r\n",
        ] {
            assert_eq!(r(bad), Route::NotFound, "{bad}");
        }
        // Writing is not a route, so it cannot be a bug.
        for bad in [
            "PUT /model.omni HTTP/1.1\r\n",
            "DELETE / HTTP/1.1\r\n",
            "POST / HTTP/1.1\r\n",
        ] {
            assert_eq!(r(bad), Route::BadMethod, "{bad}");
        }
        assert_eq!(r("garbage\r\n"), Route::Bad);
        assert_eq!(r("\r\n"), Route::Bad);
    }

    #[test]
    fn a_digest_path_is_parsed_strictly() {
        let hex = "ab".repeat(32);
        assert!(parse_digest(&hex).is_some());
        assert!(parse_digest(&format!("b3-{hex}")).is_some());
        assert!(parse_digest(&format!("b3:{hex}")).is_some());
        assert!(parse_digest(&format!("sha2-{hex}")).is_some());
        assert!(parse_digest(&format!("md5-{hex}")).is_none());
        assert!(parse_digest(&hex[..62]).is_none());
        assert!(parse_digest(&format!("{hex}00")).is_none());
        assert!(parse_digest(&"zz".repeat(32)).is_none());
    }

    /// §13.4.3: one immutable object per URL, and it has to be the object — the
    /// bytes its digest covers, not the stored form.
    #[test]
    fn every_object_is_served_at_its_own_digest_and_hashes_to_it() {
        let c = toy();
        let hash = c.header.hash;
        let (srv, _) = running(toy());
        let http = crate::transport::Http::new(&format!(
            "http://127.0.0.1:{}/objects/{}",
            srv.port(),
            crate::sha256::hex(&c.header.root_digest)
        ))
        .unwrap();
        let got = http.get().unwrap();
        assert_eq!(hash.digest(&got), c.header.root_digest);
        assert_eq!(got, c.read(&c.header.root_digest).unwrap());

        // Every object in the index, including the compressed ones if there were
        // any: what is served is always the logical form.
        for e in &c.index {
            let http = crate::transport::Http::new(&format!(
                "http://127.0.0.1:{}/objects/{}-{}",
                srv.port(),
                hash.prefix(),
                crate::sha256::hex(&e.digest)
            ))
            .unwrap();
            let got = http.get().unwrap();
            assert_eq!(
                hash.digest(&got),
                e.digest,
                "the URL did not serve its object"
            );
        }
    }

    /// A digest nobody has is 404, not an error and not someone else's object.
    #[test]
    fn an_unknown_digest_is_not_found() {
        let (srv, _) = running(toy());
        let http = crate::transport::Http::new(&format!(
            "http://127.0.0.1:{}/objects/{}",
            srv.port(),
            "ff".repeat(32)
        ))
        .unwrap();
        match http.get() {
            Err(crate::transport::Error::Protocol(m)) => assert!(m.contains("404"), "{m}"),
            other => panic!("expected a 404, got {other:?}"),
        }
        assert_eq!(srv.stats.read().2, 1, "the miss was counted");
    }

    /// The other half of §13.4: the pack, range-read by a client that reads the
    /// index first. The client here is this crate's own, which is the point —
    /// `omni fetch` against `omni serve` exercises both sides of the protocol.
    #[test]
    fn the_pack_is_range_readable_by_the_transport_client() {
        let c = toy();
        let (srv, url) = running(toy());

        let store = HttpStore::open(&url).unwrap();
        assert_eq!(store.root, c.header.root_digest);
        assert_eq!(store.index().len(), c.index.len());
        assert_eq!(store.io().0, 3, "header, front superblock, index");
        assert!(
            store.io().1 < c.bytes.len() as u64 / 2,
            "opening read {} of {}",
            store.io().1,
            c.bytes.len()
        );

        // And every object resolves through it, digest-checked by the client.
        for e in &c.index {
            assert_eq!(
                store.resolve(&e.digest).unwrap().as_deref(),
                Some(&c.read(&e.digest).unwrap()[..])
            );
        }
        let (requests, bytes, _) = srv.stats.read();
        assert!(requests >= 3 + c.index.len() as u64);
        assert!(bytes > 0);
    }

    /// The sidecar, served: one request to open a container that is here rather
    /// than a file that has to be shipped separately.
    #[test]
    fn the_sidecar_is_served_and_opens_the_pack_it_describes() {
        let c = toy();
        let (srv, _) = running(toy());
        let http =
            crate::transport::Http::new(&format!("http://127.0.0.1:{}/model.omni.idx", srv.port()))
                .unwrap();
        let bytes = http.get().unwrap();
        let s = crate::transport::Sidecar::parse(&bytes).unwrap();
        assert_eq!(s.root, c.header.root_digest);
        assert_eq!(s.file_size, c.bytes.len() as u64);

        // R-X02 against the served pack: the sidecar and the file agree.
        let store = HttpStore::open_with_sidecar(
            &format!("http://127.0.0.1:{}/model.omni", srv.port()),
            &bytes,
        )
        .unwrap();
        store.confirm_target().unwrap();
    }

    /// A range request gets a range, and an impossible one gets 416 rather than
    /// a truncated success.
    #[test]
    fn ranges_and_unsatisfiable_ranges_are_both_answered_correctly() {
        let c = toy();
        let (srv, _) = running(toy());
        let http =
            crate::transport::Http::new(&format!("http://127.0.0.1:{}/model.omni", srv.port()))
                .unwrap();
        let part = http.get_range(128, 64).unwrap();
        assert_eq!(part, &c.bytes[128..192]);

        // Past the end: the client reports the protocol failure rather than
        // returning short data.
        assert!(http.get_range(c.bytes.len() as u64 + 10, 16).is_err());
    }

    /// A HEAD request answers with the length and no body — which is how a client
    /// learns a size without moving the bytes.
    #[test]
    fn head_answers_the_size_without_the_body() {
        use std::io::{BufRead, BufReader, Write};
        let c = toy();
        let (srv, _) = running(toy());
        let mut s = std::net::TcpStream::connect(("127.0.0.1", srv.port())).unwrap();
        s.write_all(b"HEAD /model.omni HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut r = BufReader::new(s.try_clone().unwrap());
        let mut status = String::new();
        r.read_line(&mut status).unwrap();
        assert!(status.contains("200"), "{status}");
        let mut length = None;
        loop {
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse::<u64>().ok();
            }
        }
        assert_eq!(length, Some(c.bytes.len() as u64));
        // Nothing followed the headers, so a second request on the same
        // connection is answered rather than reading a stale body.
        s.write_all(b"HEAD / HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut status2 = String::new();
        r.read_line(&mut status2).unwrap();
        assert!(status2.contains("200"), "{status2}");
    }

    /// An index-only container serves what it has and 404s what it does not, so
    /// §13.8's catalogue works over HTTP without a special code path.
    #[test]
    fn an_index_only_container_serves_its_structure_and_not_its_weights() {
        let c = toy();
        let thin = Container::open(crate::transport::index_only(&c).unwrap()).unwrap();
        let (srv, url) = running(Container::open(thin.bytes.clone()).unwrap());
        let store = HttpStore::open(&url).unwrap();
        for e in &thin.index {
            let external = e.oflags & oflags::EXTERNAL != 0;
            assert_eq!(store.resolve(&e.digest).unwrap().is_none(), external);
        }
        assert!(srv.stats.read().0 > 0);
    }

    /// Nothing about a malformed request may take the server down: the next
    /// client has to be served.
    #[test]
    fn a_malformed_request_ends_its_own_connection_and_nothing_else() {
        use std::io::Write;
        let (srv, url) = running(toy());
        for junk in [
            &b"not http at all\r\n\r\n"[..],
            b"GET\r\n\r\n",
            b"PUT /model.omni HTTP/1.1\r\n\r\n",
            b"GET /model.omni HTTP/1.1\r\nContent-Length: 99\r\n\r\n",
            // A request line past the bound.
            b"GET /",
        ] {
            let mut s = std::net::TcpStream::connect(("127.0.0.1", srv.port())).unwrap();
            let _ = s.write_all(junk);
            let _ = s.flush();
            drop(s);
        }
        // A long request line, sent without an end.
        let mut s = std::net::TcpStream::connect(("127.0.0.1", srv.port())).unwrap();
        let _ = s.write_all(&vec![b'A'; MAX_LINE * 2]);
        drop(s);

        // The server is still there.
        let store = HttpStore::open(&url).unwrap();
        assert_eq!(store.index().len(), toy().index.len());
    }
}
