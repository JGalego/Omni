//! # omni-core — OMNI reference implementation
//!
//! A dependency-free implementation of the OMNI/1.0 container (§02), canonical
//! CBOR encoding (§03), object model (§01) and enough of the tensor layer (§04)
//! to build and inspect real `.omni` files.
//!
//! Deliberate constraints, mirroring the specification's own claims:
//!
//! * **No `unsafe`.** Untrusted binary input is parsed here; see §12.4.
//! * **No dependencies.** The C0 reader budget (`docs/design/sdk.md` §5) claims
//!   a conforming reader needs nothing beyond a hash function. This crate is
//!   the evidence.
//! * **Both mandatory hashes, from scratch.** §03.5.1 requires BLAKE3-256 and
//!   SHA-256; both are implemented here, the former including the tree
//!   internals that Bao verified streaming (§13.3) is built on.
//!
//! ## Not implemented here
//!
//! The tensor expression evaluator (§04.7), quantization schemes (§05),
//! OMNI-IR (§07), adapters (§08), capability negotiation (§10), the WASM
//! plugin host (§11), signatures (§12.5), compression codecs (§03.7) and every
//! transport beyond the local filesystem. Bao trees (§13.3) are implemented,
//! but nothing yet fetches over a network to use them.
//! See `docs/design/roadmap.md`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod bao;
pub mod blake3;
pub mod cbor;
pub mod container;
pub mod crc32c;
pub mod dtype;
pub mod layout;
pub mod model;
pub mod recover;
pub mod sha256;
pub mod store;

pub use bao::BaoTree;
pub use blake3::blake3 as blake3_256;
pub use cbor::Value;
pub use container::{
    otype, pack, seg, verify, Container, Digest, HashAlgo, Object, PackOptions, Report,
};
pub use dtype::{DType, FloatFmt, Round};
pub use layout::Layout;
pub use model::{ModelBuilder, TensorSpec};
pub use sha256::hex;
pub use store::{ContainerStore, DirStore, MemoryStore, Store, WritableStore};

/// The specification version this implementation targets.
pub const SPEC_VERSION: &str = "OMNI/1.0-draft";

/// Conformance profiles claimed by this build (§00.6).
pub const PROFILES: &[&str] = &["C0", "C3"];
