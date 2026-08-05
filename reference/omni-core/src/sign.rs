//! §12.5 — signatures, trust policy, and revocation.
//!
//! Signatures are COSE_Sign1 (RFC 9052) over a `to-be-signed` structure defined
//! by §12.5.2, so the trust path is CBOR end to end and no second encoding
//! appears in it.
//!
//! Three details from the section carry most of the weight:
//!
//! * **The self-reference paradox** is resolved by hashing the manifest with
//!   `attestations` removed. A signature can therefore be listed inside the
//!   object it signs, with no second manifest and no ordering games (R-S01).
//! * **`canonical_digest`** (§12.5.3) is the identity of the model *as a model*:
//!   the manifest with every cacheable object dropped and re-serialized. Two
//!   files with the same `canonical_digest` are the same model regardless of
//!   which indexes, caches or packing they carry, which is the number that
//!   belongs in a model card or a compliance record.
//! * **`summary.executables`** is inside the signed payload, so a mirror cannot
//!   add an executable cache to a signed model without detection even though
//!   caches are droppable.
//!
//! Revocation is a signed statement rather than an absence (§12.5.6). An
//! air-gapped verifier cannot check for one, and [`Verdict`] says so plainly
//! instead of reporting "not revoked".

use crate::cbor::{self, Value};
use crate::container::{otype, Digest, HashAlgo};
use crate::ed25519;

/// COSE algorithm identifier for EdDSA (RFC 9053 §2.2).
pub const COSE_ALG_EDDSA: i64 = -8;

#[derive(Debug)]
pub enum Error {
    Malformed(String),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Malformed(m) => write!(f, "signature: {m}"),
            Error::Unsupported(m) => write!(f, "signature: unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

type Res<T> = Result<T, Error>;

// ----------------------------------------------------------------------- tbs --

/// What a signature is *for* (§12.5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    Release,
    Internal,
    Test,
    Revocation,
}

impl Purpose {
    pub fn name(self) -> &'static str {
        match self {
            Purpose::Release => "release",
            Purpose::Internal => "internal",
            Purpose::Test => "test",
            Purpose::Revocation => "revocation",
        }
    }
    pub fn parse(s: &str) -> Option<Purpose> {
        Some(match s {
            "release" => Purpose::Release,
            "internal" => Purpose::Internal,
            "test" => Purpose::Test,
            "revocation" => Purpose::Revocation,
            _ => return None,
        })
    }
}

/// The signed summary of §12.5.2.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub tensors: u64,
    pub params: u64,
    /// §12.5.3.
    pub canonical_digest: Digest,
    /// Number of objects marked executable. Signed, so a mirror cannot add one.
    pub executables: u64,
}

impl Summary {
    fn to_value(&self) -> Value {
        Value::map(vec![
            ("tensors", Value::U(self.tensors)),
            ("params", Value::U(self.params)),
            (
                "canonical_digest",
                Value::Bytes(self.canonical_digest.to_vec()),
            ),
            ("executables", Value::U(self.executables)),
        ])
    }

    fn from_value(v: &Value) -> Res<Summary> {
        Ok(Summary {
            tensors: v.get("tensors").and_then(|x| x.as_u64()).unwrap_or(0),
            params: v.get("params").and_then(|x| x.as_u64()).unwrap_or(0),
            canonical_digest: v
                .get("canonical_digest")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| {
                    Error::Malformed("summary needs a 32-byte canonical_digest".into())
                })?,
            executables: v.get("executables").and_then(|x| x.as_u64()).unwrap_or(0),
        })
    }
}

/// The to-be-signed structure of §12.5.2.
#[derive(Clone, Debug, PartialEq)]
pub struct Tbs {
    /// Digest of the manifest with `attestations` removed.
    pub root: Digest,
    pub alg: String,
    pub purpose: Purpose,
    pub subject_name: String,
    pub subject_version: Option<String>,
    pub not_before: Option<String>,
    pub not_after: Option<String>,
    pub summary: Summary,
    /// Monotonic per subject: replay and rollback defense.
    pub counter: u64,
}

impl Tbs {
    pub fn to_value(&self) -> Value {
        let mut subject: Vec<(&str, Value)> =
            vec![("name", Value::text(self.subject_name.clone()))];
        if let Some(v) = &self.subject_version {
            subject.push(("version", Value::text(v.clone())));
        }
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.sec/tbs")),
            ("v", Value::U(1)),
            ("root", Value::Bytes(self.root.to_vec())),
            ("alg", Value::text(self.alg.clone())),
            ("purpose", Value::text(self.purpose.name())),
            ("subject", Value::map(subject)),
        ];
        p.push((
            "not_before",
            match &self.not_before {
                Some(s) => Value::text(s.clone()),
                None => Value::Null,
            },
        ));
        p.push((
            "not_after",
            match &self.not_after {
                Some(s) => Value::text(s.clone()),
                None => Value::Null,
            },
        ));
        p.push(("summary", self.summary.to_value()));
        p.push(("counter", Value::U(self.counter)));
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Tbs> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.sec/tbs") {
            return Err(Error::Malformed("payload is not an omni.sec/tbs".into()));
        }
        let text = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
        let subject = v
            .get("subject")
            .ok_or_else(|| Error::Malformed("tbs has no subject".into()))?;
        Ok(Tbs {
            root: v
                .get("root")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| Error::Malformed("tbs needs a 32-byte root".into()))?,
            alg: text("alg").unwrap_or_else(|| "EdDSA".into()),
            purpose: v
                .get("purpose")
                .and_then(|x| x.as_str())
                .and_then(Purpose::parse)
                .ok_or_else(|| Error::Malformed("tbs has no known purpose".into()))?,
            subject_name: subject
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            subject_version: subject
                .get("version")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            not_before: text("not_before"),
            not_after: text("not_after"),
            summary: Summary::from_value(
                v.get("summary")
                    .ok_or_else(|| Error::Malformed("tbs has no summary".into()))?,
            )?,
            counter: v.get("counter").and_then(|x| x.as_u64()).unwrap_or(0),
        })
    }

    /// The canonical bytes that get signed.
    pub fn encode(&self) -> Vec<u8> {
        self.to_value().encode()
    }
}

// ---------------------------------------------------------------------- cose --

/// A COSE_Sign1 message (RFC 9052 §4.2), untagged: `[protected, unprotected,
/// payload, signature]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CoseSign1 {
    /// Canonical CBOR of the protected header map, as a byte string.
    pub protected: Vec<u8>,
    pub unprotected: Vec<(Value, Value)>,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

impl CoseSign1 {
    /// The `Sig_structure` of RFC 9052 §4.4 — what is actually signed.
    ///
    /// Nothing in the surrounding OMNI object contributes to it, so an
    /// attacker cannot change the meaning of a signature by editing the
    /// container around it.
    pub fn sig_structure(protected: &[u8], external_aad: &[u8], payload: &[u8]) -> Vec<u8> {
        Value::Array(vec![
            Value::text("Signature1"),
            Value::Bytes(protected.to_vec()),
            Value::Bytes(external_aad.to_vec()),
            Value::Bytes(payload.to_vec()),
        ])
        .encode()
    }

    pub fn to_value(&self) -> Value {
        Value::Array(vec![
            Value::Bytes(self.protected.clone()),
            Value::Map(self.unprotected.clone()),
            Value::Bytes(self.payload.clone()),
            Value::Bytes(self.signature.clone()),
        ])
    }

    pub fn encode(&self) -> Vec<u8> {
        self.to_value().encode()
    }

    pub fn decode(bytes: &[u8]) -> Res<CoseSign1> {
        let v = cbor::decode(bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        let a = v
            .as_array()
            .ok_or_else(|| Error::Malformed("COSE_Sign1 is a four-element array".into()))?;
        if a.len() != 4 {
            return Err(Error::Malformed(format!(
                "COSE_Sign1 has {} elements, expected 4",
                a.len()
            )));
        }
        let bstr = |i: usize| -> Res<Vec<u8>> {
            a[i].as_bytes()
                .map(|b| b.to_vec())
                .ok_or_else(|| Error::Malformed(format!("COSE_Sign1 element {i} is not a bstr")))
        };
        Ok(CoseSign1 {
            protected: bstr(0)?,
            unprotected: a[1].as_map().unwrap_or(&[]).to_vec(),
            payload: bstr(2)?,
            signature: bstr(3)?,
        })
    }

    /// The `alg` in the protected header (COSE label 1).
    pub fn alg(&self) -> Res<i64> {
        let h = cbor::decode(&self.protected).map_err(|e| Error::Malformed(e.to_string()))?;
        let m = h
            .as_map()
            .ok_or_else(|| Error::Malformed("protected header is not a map".into()))?;
        for (k, v) in m {
            if k.as_u64() == Some(1) {
                return match v {
                    Value::I(n) => Ok(*n),
                    Value::U(n) => Ok(*n as i64),
                    _ => Err(Error::Malformed("alg is not an integer".into())),
                };
            }
        }
        Err(Error::Malformed("protected header has no alg".into()))
    }

    /// The key identifier from the unprotected header (COSE label 4).
    pub fn kid(&self) -> Option<Vec<u8>> {
        self.unprotected
            .iter()
            .find(|(k, _)| k.as_u64() == Some(4))
            .and_then(|(_, v)| v.as_bytes().map(|b| b.to_vec()))
    }
}

/// Signs a TBS payload with Ed25519, producing a COSE_Sign1.
pub fn sign_cose(key: &ed25519::SecretKey, tbs: &Tbs) -> CoseSign1 {
    let protected = Value::Map(vec![(Value::U(1), Value::I(COSE_ALG_EDDSA))]).encode();
    let payload = tbs.encode();
    let to_sign = CoseSign1::sig_structure(&protected, &[], &payload);
    let sig = key.sign(&to_sign);
    CoseSign1 {
        protected,
        unprotected: vec![(Value::U(4), Value::Bytes(key.public_key().to_vec()))],
        payload,
        signature: sig.to_vec(),
    }
}

/// Verifies a COSE_Sign1 against a public key, returning the decoded TBS.
pub fn verify_cose(cose: &CoseSign1, public: &[u8; ed25519::KEY_LEN]) -> Res<Tbs> {
    let alg = cose.alg()?;
    if alg != COSE_ALG_EDDSA {
        return Err(Error::Unsupported(format!(
            "COSE algorithm {alg}; only EdDSA ({COSE_ALG_EDDSA}) is implemented, and an \
             unsupported algorithm is indeterminate rather than invalid (§15.1)"
        )));
    }
    let sig: [u8; ed25519::SIG_LEN] = cose
        .signature
        .clone()
        .try_into()
        .map_err(|_| Error::Malformed("EdDSA signatures are 64 bytes".into()))?;
    let to_sign = CoseSign1::sig_structure(&cose.protected, &[], &cose.payload);
    if !ed25519::verify(public, &to_sign, &sig) {
        return Err(Error::Malformed("signature does not verify".into()));
    }
    let payload = cbor::decode(&cose.payload).map_err(|e| Error::Malformed(e.to_string()))?;
    Tbs::from_value(&payload)
}

// --------------------------------------------------------------- signature obj --

/// A `Signature` object (otype 0x0012).
///
/// The COSE message is carried as a byte string: it is the trust path, and
/// wrapping it in a structure whose keys could be reordered or extended would
/// put OMNI's encoder inside that path for no benefit. The convenience fields
/// beside it are unsigned and exist only so a reader can pick which signatures
/// to check without parsing all of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    pub cose: Vec<u8>,
    pub alg: String,
    pub kid: Option<Vec<u8>>,
    /// Certificate chain or Sigstore bundle, when identity is not a bare key.
    pub identity: Option<Value>,
}

impl Signature {
    pub fn new(cose: &CoseSign1) -> Signature {
        Signature {
            cose: cose.encode(),
            alg: "EdDSA".into(),
            kid: cose.kid(),
            identity: None,
        }
    }

    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.sec/signature")),
            ("v", Value::U(1)),
            ("alg", Value::text(self.alg.clone())),
            ("cose", Value::Bytes(self.cose.clone())),
        ];
        if let Some(k) = &self.kid {
            p.push(("kid", Value::Bytes(k.clone())));
        }
        if let Some(i) = &self.identity {
            p.push(("identity", i.clone()));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Signature> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.sec/signature") {
            return Err(Error::Malformed(
                "R-O02: object is not an omni.sec/signature".into(),
            ));
        }
        Ok(Signature {
            cose: v
                .get("cose")
                .and_then(|x| x.as_bytes())
                .ok_or_else(|| Error::Malformed("signature has no `cose`".into()))?
                .to_vec(),
            alg: v
                .get("alg")
                .and_then(|x| x.as_str())
                .unwrap_or("EdDSA")
                .to_string(),
            kid: v.get("kid").and_then(|x| x.as_bytes()).map(|b| b.to_vec()),
            identity: v.get("identity").cloned(),
        })
    }

    pub fn message(&self) -> Res<CoseSign1> {
        CoseSign1::decode(&self.cose)
    }
}

// ------------------------------------------------------- manifest preparation --

/// The manifest with `attestations` removed — what §12.5.2 hashes, resolving the
/// self-reference paradox.
pub fn strip_attestations(manifest: &Value) -> Value {
    match manifest {
        Value::Map(m) => Value::Map(
            m.iter()
                .filter(|(k, _)| k.as_str() != Some("attestations"))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The digest §12.5.2 calls `root`.
pub fn signing_root(manifest: &Value, algo: HashAlgo) -> Digest {
    algo.digest(&strip_attestations(manifest).encode())
}

/// §12.5.3 — the manifest with `attestations` *and* every cacheable object
/// removed, re-serialized canonically.
///
/// `is_cacheable` answers for a digest; a caller with a container passes its
/// index's `CACHEABLE` flag, and a caller with a bare store passes whatever it
/// knows. Objects a reader cannot classify are kept, because dropping something
/// load-bearing would change the model's identity — the failure has to be in the
/// safe direction.
pub fn canonical_manifest(manifest: &Value, is_cacheable: &dyn Fn(&Digest) -> bool) -> Value {
    fn prune(v: &Value, is_cacheable: &dyn Fn(&Digest) -> bool) -> Option<Value> {
        // A ref is `[otype, digest]`; drop the whole entry when it points at a
        // cacheable object.
        if let Some(d) = ref_digest(v) {
            if is_cacheable(&d) {
                return None;
            }
        }
        Some(match v {
            Value::Array(a) => {
                Value::Array(a.iter().filter_map(|x| prune(x, is_cacheable)).collect())
            }
            Value::Map(m) => Value::Map(
                m.iter()
                    .filter(|(k, _)| k.as_str() != Some("attestations"))
                    .filter_map(|(k, val)| prune(val, is_cacheable).map(|p| (k.clone(), p)))
                    .collect(),
            ),
            Value::Tag(t, inner) => Value::Tag(*t, Box::new(prune(inner, is_cacheable)?)),
            other => other.clone(),
        })
    }
    prune(manifest, is_cacheable).unwrap_or(Value::Map(vec![]))
}

/// §12.5.3's number: the identity of the model as a model.
pub fn canonical_digest(
    manifest: &Value,
    algo: HashAlgo,
    is_cacheable: &dyn Fn(&Digest) -> bool,
) -> Digest {
    algo.digest(&canonical_manifest(manifest, is_cacheable).encode())
}

fn ref_digest(v: &Value) -> Option<Digest> {
    let v = match v {
        Value::Tag(cbor::TAG_REF, inner) => inner.as_ref(),
        other => other,
    };
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    a.first()?.as_u64()?;
    a.get(1)?.as_bytes()?.try_into().ok()
}

// ------------------------------------------------------------------- policy --

/// A key a verifier trusts.
#[derive(Clone, Debug, PartialEq)]
pub struct TrustedKey {
    pub kid: Vec<u8>,
    pub public: [u8; ed25519::KEY_LEN],
    pub roles: Vec<String>,
}

impl TrustedKey {
    pub fn new(public: [u8; ed25519::KEY_LEN]) -> TrustedKey {
        TrustedKey {
            kid: public.to_vec(),
            public,
            roles: Vec::new(),
        }
    }

    pub fn with_role(mut self, role: &str) -> TrustedKey {
        self.roles.push(role.to_string());
        self
    }
}

/// How many and whose signatures are required (§12.5.4).
#[derive(Clone, Debug, PartialEq)]
pub enum Requirement {
    AnyOf,
    AllOf,
    KOfN(usize),
    /// One valid signature from a key holding each named role.
    RoleBased(Vec<String>),
}

/// A verifier's trust policy.
#[derive(Clone, Debug)]
pub struct Policy {
    pub keys: Vec<TrustedKey>,
    pub requirement: Requirement,
    /// Accepted purposes. A test-purpose signature must not satisfy a release
    /// policy, which is the whole reason `purpose` is in the signed payload.
    pub purposes: Vec<Purpose>,
    /// Rollback defense: the lowest counter this verifier will accept for the
    /// subject (§12.5.2).
    pub min_counter: Option<u64>,
    /// RFC 3339 timestamp to evaluate validity windows against. `None` skips the
    /// freshness check and says so in the verdict.
    pub now: Option<String>,
}

impl Policy {
    pub fn keys(keys: Vec<TrustedKey>) -> Policy {
        Policy {
            keys,
            requirement: Requirement::AnyOf,
            purposes: vec![Purpose::Release],
            min_counter: None,
            now: None,
        }
    }

    pub fn requirement(mut self, r: Requirement) -> Policy {
        self.requirement = r;
        self
    }

    pub fn purposes(mut self, p: Vec<Purpose>) -> Policy {
        self.purposes = p;
        self
    }

    pub fn at(mut self, now: &str) -> Policy {
        self.now = Some(now.to_string());
        self
    }
}

/// One signature's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub kid: Option<Vec<u8>>,
    pub roles: Vec<String>,
    /// `Ok` with the TBS, or the reason it did not count.
    pub ok: bool,
    pub message: String,
    /// Distinguishes "this signature is bad" from "this verifier cannot say".
    pub indeterminate: bool,
    pub tbs: Option<Tbs>,
}

/// The result of a V7 check.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub outcomes: Vec<Outcome>,
    /// True when the policy's requirement is met.
    pub satisfied: bool,
    /// Reasons the verdict is not a clean yes-or-no.
    pub indeterminate: Vec<String>,
}

impl Verdict {
    pub fn valid_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.ok).count()
    }

    pub fn invalid_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| !o.ok && !o.indeterminate)
            .count()
    }
}

/// Verifies a set of signatures against a policy (V7).
///
/// `root` is the digest of the manifest with `attestations` removed, and
/// `canonical` is §12.5.3's digest; both are recomputed by the caller and
/// compared here against what each signature claims, so a signature over a
/// different model cannot be replayed onto this one (R-S01, R-S02).
pub fn verify_signatures(
    signatures: &[Signature],
    root: &Digest,
    canonical: &Digest,
    policy: &Policy,
) -> Verdict {
    let mut v = Verdict::default();
    if signatures.is_empty() {
        v.indeterminate
            .push("no signatures are present".to_string());
        return v;
    }
    for s in signatures {
        let mut out = Outcome {
            kid: s.kid.clone(),
            roles: Vec::new(),
            ok: false,
            message: String::new(),
            indeterminate: false,
            tbs: None,
        };
        let cose = match s.message() {
            Ok(c) => c,
            Err(e) => {
                out.message = e.to_string();
                v.outcomes.push(out);
                continue;
            }
        };
        // Which trusted key does this claim? An unknown key is indeterminate:
        // the signature may be perfectly good and simply not ours to trust.
        let kid = cose.kid().or_else(|| s.kid.clone());
        let key = policy.keys.iter().find(|k| match &kid {
            Some(id) => &k.kid == id,
            None => false,
        });
        let Some(key) = key else {
            out.indeterminate = true;
            out.message = "no trusted key matches this signature's kid".into();
            v.indeterminate.push(out.message.clone());
            v.outcomes.push(out);
            continue;
        };
        out.roles = key.roles.clone();
        let tbs = match verify_cose(&cose, &key.public) {
            Ok(t) => t,
            Err(Error::Unsupported(m)) => {
                out.indeterminate = true;
                out.message = m.clone();
                v.indeterminate.push(m);
                v.outcomes.push(out);
                continue;
            }
            Err(e) => {
                out.message = e.to_string();
                v.outcomes.push(out);
                continue;
            }
        };
        // R-S01: the signature must cover *this* manifest.
        if &tbs.root != root {
            out.message = format!(
                "signature covers manifest {} but this one is {}",
                crate::sha256::hex(&tbs.root[..8]),
                crate::sha256::hex(&root[..8])
            );
            v.outcomes.push(out);
            continue;
        }
        // R-S02: and this model.
        if &tbs.summary.canonical_digest != canonical {
            out.message = format!(
                "summary.canonical_digest is {} but recomputation gives {}",
                crate::sha256::hex(&tbs.summary.canonical_digest[..8]),
                crate::sha256::hex(&canonical[..8])
            );
            v.outcomes.push(out);
            continue;
        }
        if !policy.purposes.contains(&tbs.purpose) {
            out.message = format!(
                "signature purpose is `{}`, which this policy does not accept",
                tbs.purpose.name()
            );
            v.outcomes.push(out);
            continue;
        }
        if let Some(min) = policy.min_counter {
            if tbs.counter < min {
                out.message = format!(
                    "counter {} is below the policy's floor of {min}: this looks like a rollback",
                    tbs.counter
                );
                v.outcomes.push(out);
                continue;
            }
        }
        match freshness(&tbs, policy.now.as_deref()) {
            Freshness::Valid => {}
            Freshness::Expired(m) | Freshness::NotYet(m) => {
                out.message = m;
                v.outcomes.push(out);
                continue;
            }
            Freshness::Unknown(m) => {
                out.indeterminate = true;
                out.message = m.clone();
                v.indeterminate.push(m);
                v.outcomes.push(out);
                continue;
            }
        }
        out.ok = true;
        out.message = "valid".into();
        out.tbs = Some(tbs);
        v.outcomes.push(out);
    }

    let valid: Vec<&Outcome> = v.outcomes.iter().filter(|o| o.ok).collect();
    v.satisfied = match &policy.requirement {
        Requirement::AnyOf => !valid.is_empty(),
        Requirement::AllOf => {
            !policy.keys.is_empty()
                && policy.keys.iter().all(|k| {
                    valid
                        .iter()
                        .any(|o| o.kid.as_ref().is_some_and(|id| id == &k.kid))
                })
        }
        Requirement::KOfN(k) => valid.len() >= *k,
        Requirement::RoleBased(roles) => roles
            .iter()
            .all(|r| valid.iter().any(|o| o.roles.contains(r))),
    };
    v
}

enum Freshness {
    Valid,
    Expired(String),
    NotYet(String),
    Unknown(String),
}

fn freshness(tbs: &Tbs, now: Option<&str>) -> Freshness {
    let Some(now) = now else {
        if tbs.not_before.is_some() || tbs.not_after.is_some() {
            return Freshness::Unknown(
                "the signature declares a validity window but the verifier has no clock; \
                 an air-gapped check cannot decide freshness (§12.5.6)"
                    .into(),
            );
        }
        return Freshness::Valid;
    };
    let Some(now_ts) = parse_rfc3339(now) else {
        return Freshness::Unknown(format!("`{now}` is not an RFC 3339 timestamp"));
    };
    if let Some(nb) = &tbs.not_before {
        match parse_rfc3339(nb) {
            Some(t) if now_ts < t => {
                return Freshness::NotYet(format!("not valid before {nb}"));
            }
            None => return Freshness::Unknown(format!("`{nb}` is not an RFC 3339 timestamp")),
            _ => {}
        }
    }
    if let Some(na) = &tbs.not_after {
        match parse_rfc3339(na) {
            Some(t) if now_ts > t => {
                return Freshness::Expired(format!("expired at {na}"));
            }
            None => return Freshness::Unknown(format!("`{na}` is not an RFC 3339 timestamp")),
            _ => {}
        }
    }
    Freshness::Valid
}

/// Parses an RFC 3339 timestamp into seconds since the Unix epoch.
///
/// Accepts `Z` and numeric offsets, and rejects anything else rather than
/// guessing — a misparsed expiry is a security bug, not a formatting nuisance.
pub fn parse_rfc3339(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || (b[10] != b'T' && b[10] != b't') {
        return None;
    }
    let num = |from: usize, to: usize| -> Option<i64> { s.get(from..to)?.parse().ok() };
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, sec) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    // Skip fractional seconds.
    let mut i = 19;
    if b.get(i) == Some(&b'.') {
        i += 1;
        while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
    }
    let offset = match b.get(i) {
        Some(b'Z') | Some(b'z') => 0,
        Some(sign @ (b'+' | b'-')) => {
            let oh = num(i + 1, i + 3)?;
            if b.get(i + 3) != Some(&b':') {
                return None;
            }
            let om = num(i + 4, i + 6)?;
            let mag = oh * 3600 + om * 60;
            if *sign == b'+' {
                -mag
            } else {
                mag
            }
        }
        _ => return None,
    };
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec + offset)
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01 for a proleptic
/// Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

// --------------------------------------------------------------- revocation --

/// A revocation statement (§12.5.6). Revocation is a signed statement, not an
/// absence.
#[derive(Clone, Debug, PartialEq)]
pub struct Revocation {
    /// `canonical_digest` of the revoked model.
    pub target: Digest,
    pub reason: String,
    pub replacement: Option<Digest>,
    pub issued: Option<String>,
    pub issuer: Option<String>,
}

impl Revocation {
    pub fn to_value(&self) -> Value {
        let mut p: Vec<(&str, Value)> = vec![
            ("t", Value::text("omni.sec/revocation")),
            ("v", Value::U(1)),
            ("target", Value::Bytes(self.target.to_vec())),
            ("reason", Value::text(self.reason.clone())),
        ];
        if let Some(r) = &self.replacement {
            p.push(("replacement", Value::Bytes(r.to_vec())));
        }
        if let Some(i) = &self.issued {
            p.push(("issued", Value::text(i.clone())));
        }
        if let Some(i) = &self.issuer {
            p.push(("issuer", Value::text(i.clone())));
        }
        Value::map(p)
    }

    pub fn from_value(v: &Value) -> Res<Revocation> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.sec/revocation") {
            return Err(Error::Malformed("not an omni.sec/revocation".into()));
        }
        Ok(Revocation {
            target: v
                .get("target")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| Error::Malformed("revocation needs a 32-byte target".into()))?,
            reason: v
                .get("reason")
                .and_then(|x| x.as_str())
                .unwrap_or("unspecified")
                .to_string(),
            replacement: v
                .get("replacement")
                .and_then(|x| x.as_bytes())
                .and_then(|b| b.try_into().ok()),
            issued: v
                .get("issued")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
            issuer: v
                .get("issuer")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string()),
        })
    }
}

/// Whether any known revocation targets this model.
pub fn find_revocation<'a>(
    canonical: &Digest,
    revocations: &'a [Revocation],
) -> Option<&'a Revocation> {
    revocations.iter().find(|r| &r.target == canonical)
}

/// The `attestations` refs a manifest carries.
pub fn attestation_refs(manifest: &Value) -> Vec<Digest> {
    let mut out = Vec::new();
    for a in manifest
        .get("attestations")
        .and_then(|x| x.as_array())
        .unwrap_or(&[])
    {
        if let Some(d) = ref_digest(a) {
            out.push(d);
        }
    }
    out
}

/// Adds a signature ref to a manifest's `attestations`, returning the new
/// manifest. The signed digest does not change, because it is computed with
/// `attestations` removed — which is the point of §12.5.2.
pub fn attach(manifest: &Value, signature: &Digest) -> Value {
    let r = Value::Array(vec![
        Value::U(otype::SIGNATURE as u64),
        Value::Bytes(signature.to_vec()),
    ]);
    let mut pairs = match manifest {
        Value::Map(m) => m.clone(),
        other => return other.clone(),
    };
    let mut existing: Vec<Value> = manifest
        .get("attestations")
        .and_then(|x| x.as_array())
        .map(|a| a.to_vec())
        .unwrap_or_default();
    existing.push(r);
    pairs.retain(|(k, _)| k.as_str() != Some("attestations"));
    pairs.push((Value::text("attestations"), Value::Array(existing)));
    Value::Map(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Value {
        Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("model")),
            (
                "assets",
                Value::map(vec![(
                    "model",
                    Value::Array(vec![Value::U(3), Value::Bytes(vec![1u8; 32])]),
                )]),
            ),
            (
                "caches",
                Value::Array(vec![Value::Array(vec![
                    Value::U(15),
                    Value::Bytes(vec![2u8; 32]),
                ])]),
            ),
        ])
    }

    fn tbs_for(m: &Value, algo: HashAlgo, counter: u64) -> Tbs {
        Tbs {
            root: signing_root(m, algo),
            alg: "EdDSA".into(),
            purpose: Purpose::Release,
            subject_name: "acme/llm-8b".into(),
            subject_version: Some("2026.08.1".into()),
            not_before: None,
            not_after: None,
            summary: Summary {
                tensors: 291,
                params: 8_030_261_248,
                canonical_digest: canonical_digest(m, algo, &|d| d == &[2u8; 32]),
                executables: 0,
            },
            counter,
        }
    }

    fn key(seed: u8) -> ed25519::SecretKey {
        ed25519::SecretKey::from_seed(&[seed; 32])
    }

    #[test]
    fn a_signature_verifies_and_round_trips() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(1);
        let tbs = tbs_for(&m, algo, 3);
        let cose = sign_cose(&sk, &tbs);
        assert_eq!(cose.alg().unwrap(), COSE_ALG_EDDSA);
        assert_eq!(cose.kid().unwrap(), sk.public_key().to_vec());
        let got = verify_cose(&cose, &sk.public_key()).unwrap();
        assert_eq!(got, tbs);

        // Through the Signature object and canonical CBOR.
        let s = Signature::new(&cose);
        let v = s.to_value();
        let again = Signature::from_value(&cbor::decode(&v.encode()).unwrap()).unwrap();
        assert_eq!(again, s);
        assert_eq!(
            verify_cose(&again.message().unwrap(), &sk.public_key()).unwrap(),
            tbs
        );
    }

    #[test]
    fn the_signature_covers_the_manifest_with_attestations_removed() {
        // §12.5.2's self-reference resolution: attaching the signature to the
        // manifest it signs must not invalidate it.
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(2);
        let tbs = tbs_for(&m, algo, 1);
        let cose = sign_cose(&sk, &tbs);
        let sig = Signature::new(&cose);
        let sig_digest = algo.digest(&sig.to_value().encode());
        let signed_manifest = attach(&m, &sig_digest);
        // The manifest's own bytes changed...
        assert_ne!(m.encode(), signed_manifest.encode());
        // ...but the signed digest did not.
        assert_eq!(signing_root(&signed_manifest, algo), signing_root(&m, algo));
        let v = verify_signatures(
            &[sig],
            &signing_root(&signed_manifest, algo),
            &canonical_digest(&signed_manifest, algo, &|d| d == &[2u8; 32]),
            &Policy::keys(vec![TrustedKey::new(sk.public_key())]),
        );
        assert!(v.satisfied, "{:?}", v.outcomes);
        assert_eq!(attestation_refs(&signed_manifest), vec![sig_digest]);
    }

    #[test]
    fn the_canonical_digest_ignores_caches_and_packing() {
        let algo = HashAlgo::default();
        let m = manifest();
        let cacheable = |d: &Digest| d == &[2u8; 32];
        let a = canonical_digest(&m, algo, &cacheable);
        // Adding another cache does not change the model's identity.
        let mut pairs = m.as_map().unwrap().to_vec();
        pairs.retain(|(k, _)| k.as_str() != Some("caches"));
        pairs.push((
            Value::text("caches"),
            Value::Array(vec![
                Value::Array(vec![Value::U(15), Value::Bytes(vec![2u8; 32])]),
                Value::Array(vec![Value::U(21), Value::Bytes(vec![3u8; 32])]),
            ]),
        ));
        let with_more = Value::Map(pairs);
        let b = canonical_digest(&with_more, algo, &|d| d == &[2u8; 32] || d == &[3u8; 32]);
        assert_eq!(a, b);
        // But changing a non-cacheable asset does.
        let mut pairs = m.as_map().unwrap().to_vec();
        pairs.retain(|(k, _)| k.as_str() != Some("assets"));
        pairs.push((
            Value::text("assets"),
            Value::map(vec![(
                "model",
                Value::Array(vec![Value::U(3), Value::Bytes(vec![9u8; 32])]),
            )]),
        ));
        assert_ne!(a, canonical_digest(&Value::Map(pairs), algo, &cacheable));
        // An object the reader cannot classify is kept, so identity fails in the
        // safe direction.
        assert_ne!(a, canonical_digest(&m, algo, &|_| false));
    }

    #[test]
    fn a_signature_over_a_different_model_does_not_transfer() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(3);
        let tbs = tbs_for(&m, algo, 1);
        let sig = Signature::new(&sign_cose(&sk, &tbs));
        let policy = Policy::keys(vec![TrustedKey::new(sk.public_key())]);
        // R-S01: a different manifest root.
        let v = verify_signatures(
            std::slice::from_ref(&sig),
            &[7u8; 32],
            &tbs.summary.canonical_digest,
            &policy,
        );
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("covers manifest"));
        assert_eq!(v.invalid_count(), 1);
        // R-S02: a different canonical digest.
        let v = verify_signatures(&[sig], &signing_root(&m, algo), &[8u8; 32], &policy);
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("canonical_digest"));
    }

    #[test]
    fn tampering_with_the_payload_or_the_header_is_caught() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(4);
        let cose = sign_cose(&sk, &tbs_for(&m, algo, 1));
        // A changed payload.
        let mut bad = cose.clone();
        bad.payload[10] ^= 1;
        assert!(verify_cose(&bad, &sk.public_key()).is_err());
        // A changed protected header, which is inside Sig_structure.
        let mut bad = cose.clone();
        bad.protected = Value::Map(vec![(Value::U(1), Value::I(-7))]).encode();
        assert!(verify_cose(&bad, &sk.public_key()).is_err());
        // A changed signature.
        let mut bad = cose;
        bad.signature[0] ^= 1;
        assert!(verify_cose(&bad, &sk.public_key()).is_err());
    }

    #[test]
    fn an_unknown_key_is_indeterminate_not_invalid() {
        // §15.1: reporting indeterminate as invalid is itself a conformance
        // violation. A signature by someone we do not trust is not a bad
        // signature.
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(5);
        let sig = Signature::new(&sign_cose(&sk, &tbs_for(&m, algo, 1)));
        let policy = Policy::keys(vec![TrustedKey::new(key(6).public_key())]);
        let v = verify_signatures(
            &[sig],
            &signing_root(&m, algo),
            &canonical_digest(&m, algo, &|d| d == &[2u8; 32]),
            &policy,
        );
        assert!(!v.satisfied);
        assert_eq!(v.invalid_count(), 0);
        assert!(v.outcomes[0].indeterminate);
        assert_eq!(v.indeterminate.len(), 1);
    }

    #[test]
    fn an_unsupported_algorithm_is_indeterminate() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(7);
        let mut cose = sign_cose(&sk, &tbs_for(&m, algo, 1));
        // ES256: legal per §12.5.1, not implemented here.
        cose.protected = Value::Map(vec![(Value::U(1), Value::I(-7))]).encode();
        let sig = Signature::new(&cose);
        let v = verify_signatures(
            &[sig],
            &signing_root(&m, algo),
            &canonical_digest(&m, algo, &|d| d == &[2u8; 32]),
            &Policy::keys(vec![TrustedKey::new(sk.public_key())]),
        );
        assert!(!v.satisfied);
        assert_eq!(v.invalid_count(), 0);
        assert!(v.outcomes[0].indeterminate);
        assert!(v.outcomes[0].message.contains("EdDSA"));
    }

    #[test]
    fn purposes_do_not_leak_across() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(8);
        let mut tbs = tbs_for(&m, algo, 1);
        tbs.purpose = Purpose::Test;
        let sig = Signature::new(&sign_cose(&sk, &tbs));
        let root = signing_root(&m, algo);
        let canon = canonical_digest(&m, algo, &|d| d == &[2u8; 32]);
        // A test signature does not satisfy a release policy.
        let v = verify_signatures(
            std::slice::from_ref(&sig),
            &root,
            &canon,
            &Policy::keys(vec![TrustedKey::new(sk.public_key())]),
        );
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("purpose"));
        // But it does satisfy one that asks for it.
        let v = verify_signatures(
            &[sig],
            &root,
            &canon,
            &Policy::keys(vec![TrustedKey::new(sk.public_key())]).purposes(vec![Purpose::Test]),
        );
        assert!(v.satisfied);
    }

    #[test]
    fn the_counter_defends_against_rollback() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(9);
        let sig = Signature::new(&sign_cose(&sk, &tbs_for(&m, algo, 2)));
        let mut policy = Policy::keys(vec![TrustedKey::new(sk.public_key())]);
        policy.min_counter = Some(5);
        let v = verify_signatures(
            &[sig],
            &signing_root(&m, algo),
            &canonical_digest(&m, algo, &|d| d == &[2u8; 32]),
            &policy,
        );
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("rollback"));
    }

    #[test]
    fn multi_party_policies() {
        let algo = HashAlgo::default();
        let m = manifest();
        let root = signing_root(&m, algo);
        let canon = canonical_digest(&m, algo, &|d| d == &[2u8; 32]);
        let (a, b, c) = (key(11), key(12), key(13));
        let sa = Signature::new(&sign_cose(&a, &tbs_for(&m, algo, 1)));
        let sb = Signature::new(&sign_cose(&b, &tbs_for(&m, algo, 1)));
        let keys = vec![
            TrustedKey::new(a.public_key()).with_role("publisher"),
            TrustedKey::new(b.public_key()).with_role("auditor"),
            TrustedKey::new(c.public_key()).with_role("mirror"),
        ];

        // any-of: one is enough.
        let v = verify_signatures(
            std::slice::from_ref(&sa),
            &root,
            &canon,
            &Policy::keys(keys.clone()),
        );
        assert!(v.satisfied);
        // all-of: three keys, two signatures — not satisfied.
        let v = verify_signatures(
            &[sa.clone(), sb.clone()],
            &root,
            &canon,
            &Policy::keys(keys.clone()).requirement(Requirement::AllOf),
        );
        assert!(!v.satisfied);
        // 2-of-n: satisfied.
        let v = verify_signatures(
            &[sa.clone(), sb.clone()],
            &root,
            &canon,
            &Policy::keys(keys.clone()).requirement(Requirement::KOfN(2)),
        );
        assert!(v.satisfied);
        assert_eq!(v.valid_count(), 2);
        // 3-of-n: not.
        let v = verify_signatures(
            &[sa.clone(), sb.clone()],
            &root,
            &canon,
            &Policy::keys(keys.clone()).requirement(Requirement::KOfN(3)),
        );
        assert!(!v.satisfied);
        // role-based: publisher and auditor both signed.
        let v = verify_signatures(
            &[sa.clone(), sb],
            &root,
            &canon,
            &Policy::keys(keys.clone()).requirement(Requirement::RoleBased(vec![
                "publisher".into(),
                "auditor".into(),
            ])),
        );
        assert!(v.satisfied);
        // The mirror's role is missing.
        let v = verify_signatures(
            &[sa],
            &root,
            &canon,
            &Policy::keys(keys).requirement(Requirement::RoleBased(vec!["mirror".into()])),
        );
        assert!(!v.satisfied);
    }

    #[test]
    fn no_signatures_is_indeterminate() {
        let v = verify_signatures(&[], &[0u8; 32], &[0u8; 32], &Policy::keys(vec![]));
        assert!(!v.satisfied);
        assert_eq!(v.indeterminate.len(), 1);
        assert_eq!(v.invalid_count(), 0);
    }

    #[test]
    fn validity_windows_are_checked_when_there_is_a_clock() {
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(14);
        let mut tbs = tbs_for(&m, algo, 1);
        tbs.not_before = Some("2026-01-01T00:00:00Z".into());
        tbs.not_after = Some("2026-12-31T23:59:59Z".into());
        let sig = Signature::new(&sign_cose(&sk, &tbs));
        let root = signing_root(&m, algo);
        let canon = canonical_digest(&m, algo, &|d| d == &[2u8; 32]);
        let base = Policy::keys(vec![TrustedKey::new(sk.public_key())]);

        assert!(
            verify_signatures(
                std::slice::from_ref(&sig),
                &root,
                &canon,
                &base.clone().at("2026-08-05T12:00:00Z")
            )
            .satisfied
        );
        let v = verify_signatures(
            std::slice::from_ref(&sig),
            &root,
            &canon,
            &base.clone().at("2027-01-01T00:00:00Z"),
        );
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("expired"));
        let v = verify_signatures(
            std::slice::from_ref(&sig),
            &root,
            &canon,
            &base.clone().at("2025-06-01T00:00:00Z"),
        );
        assert!(!v.satisfied);
        assert!(v.outcomes[0].message.contains("not valid before"));
        // With no clock, a window makes the verdict indeterminate rather than
        // valid: an air-gapped verifier cannot decide freshness, and §12.5.6
        // says so plainly.
        let v = verify_signatures(std::slice::from_ref(&sig), &root, &canon, &base);
        assert!(!v.satisfied);
        assert!(v.outcomes[0].indeterminate);
        assert!(v.outcomes[0].message.contains("no clock"));
    }

    #[test]
    fn rfc3339_parsing_is_strict() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2026-08-05T00:00:00Z"), Some(1_785_888_000));
        // Offsets are honoured, in the right direction.
        assert_eq!(
            parse_rfc3339("2026-08-05T01:00:00+01:00"),
            parse_rfc3339("2026-08-05T00:00:00Z")
        );
        assert_eq!(
            parse_rfc3339("2026-08-04T23:00:00-01:00"),
            parse_rfc3339("2026-08-05T00:00:00Z")
        );
        // Fractional seconds are skipped, not misparsed.
        assert_eq!(
            parse_rfc3339("2026-08-05T00:00:00.123456Z"),
            parse_rfc3339("2026-08-05T00:00:00Z")
        );
        // A leap second is accepted rather than rejected.
        assert!(parse_rfc3339("2016-12-31T23:59:60Z").is_some());
        // Anything else is refused: a misparsed expiry is a security bug.
        for bad in [
            "2026-08-05",
            "2026-08-05 00:00:00Z",
            "2026-13-05T00:00:00Z",
            "2026-08-05T25:00:00Z",
            "2026-08-05T00:00:00",
            "2026-08-05T00:00:00+0100",
            "",
        ] {
            assert!(parse_rfc3339(bad).is_none(), "`{bad}` should not parse");
        }
    }

    #[test]
    fn revocation_is_a_statement_and_round_trips() {
        let r = Revocation {
            target: [4u8; 32],
            reason: "weights-compromised".into(),
            replacement: Some([5u8; 32]),
            issued: Some("2026-08-05T00:00:00Z".into()),
            issuer: Some("security@acme.com".into()),
        };
        let v = r.to_value();
        assert_eq!(
            Revocation::from_value(&cbor::decode(&v.encode()).unwrap()).unwrap(),
            r
        );
        assert!(find_revocation(&[4u8; 32], std::slice::from_ref(&r)).is_some());
        assert!(find_revocation(&[9u8; 32], std::slice::from_ref(&r)).is_none());
        // A revocation is signed like anything else: its purpose says so.
        let sk = key(15);
        let tbs = Tbs {
            root: HashAlgo::default().digest(&r.to_value().encode()),
            alg: "EdDSA".into(),
            purpose: Purpose::Revocation,
            subject_name: "acme/llm-8b".into(),
            subject_version: None,
            not_before: None,
            not_after: None,
            summary: Summary {
                canonical_digest: r.target,
                ..Default::default()
            },
            counter: 1,
        };
        let cose = sign_cose(&sk, &tbs);
        assert_eq!(
            verify_cose(&cose, &sk.public_key()).unwrap().purpose,
            Purpose::Revocation
        );
    }

    #[test]
    fn the_signed_summary_pins_the_executable_count() {
        // §12.5.2: a mirror cannot add an executable cache to a signed model
        // without detection, even though caches are droppable.
        let algo = HashAlgo::default();
        let m = manifest();
        let sk = key(16);
        let tbs = tbs_for(&m, algo, 1);
        assert_eq!(tbs.summary.executables, 0);
        let cose = sign_cose(&sk, &tbs);
        // Re-signing with a different count changes the signature, so the two
        // are distinguishable — the count is inside the signed bytes.
        let mut with_exec = tbs.clone();
        with_exec.summary.executables = 1;
        let other = sign_cose(&sk, &with_exec);
        assert_ne!(cose.signature, other.signature);
        assert_eq!(
            verify_cose(&other, &sk.public_key())
                .unwrap()
                .summary
                .executables,
            1
        );
    }

    #[test]
    fn tbs_round_trips_through_canonical_cbor() {
        let algo = HashAlgo::default();
        let tbs = tbs_for(&manifest(), algo, 7);
        let bytes = tbs.encode();
        let again = Tbs::from_value(&cbor::decode(&bytes).unwrap()).unwrap();
        assert_eq!(again, tbs);
        assert_eq!(again.encode(), bytes);
        // A payload that is not a TBS is refused.
        assert!(Tbs::from_value(&Value::map(vec![("t", Value::text("x"))])).is_err());
    }
}
