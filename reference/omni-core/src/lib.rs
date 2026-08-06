//! # omni-core — OMNI reference implementation
//!
//! A dependency-free implementation of the OMNI/1.0 container (§02), canonical
//! CBOR encoding (§03), object model (§01) and the tensor layer (§04): the
//! numeric type algebra, layouts, and the tensor expression algebra with its
//! typing, identity, evaluation and range pushdown, plus the quantization
//! sparsity and quantization scheme catalogues of §04.6 and §05. Above those:
//! OMNI-IR (§07) with a reference interpreter that executes it, a Jinja2 to
//! OMNI-CT translator (§06.9), training state (§09), a WebAssembly plugin host
//! (§11.6),
//! streaming transport (§13), and safetensors, PEFT, GPTQ and AWQ import.
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
//! `https://`: [`transport`] speaks HTTP/1.1 over a socket with ranges,
//! coalescing and per-object verification, but TLS needs a cryptographic
//! transport stack and there are no dependencies to provide one, so an
//! `https://` URL is refused with that reason rather than downgraded. The OCI
//! `omni mount` (§13.9) is unimplemented, and so is the registry client behind
//! [`oci`]'s §13.5 mapping; [`serve`] is the object server of §13.4.3.
//! Of §03.7's codecs, `zstd` (the MUST) and `deflate` are here; the MAY-level
//! ones are reported as unsupported rather than half-decoded. The WebAssembly
//! host of §11.6 runs the core instruction set but not SIMD. Of the 25 formats
//! in `docs/design/import-export.md` §3, four are implemented — [`safetensors`],
//! [`peft`], and GPTQ and AWQ in [`hfquant`] — and a request to import another is
//! refused by name. Only safetensors exports, so a GPTQ import can be
//! dequantized out but not written back as GPTQ.
//! See `docs/design/roadmap.md`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod adapter;
pub mod bao;
pub mod blake3;
pub mod cbor;
pub mod codec;
pub mod container;
pub mod crc32c;
pub mod ct;
pub mod delta;
pub mod dtype;
pub mod ed25519;
pub mod expr;
pub mod hfquant;
pub mod interp;
pub mod ir;
pub mod jinja;
pub mod json;
pub mod layout;
pub mod model;
pub mod oci;
pub mod pattern;
pub mod peft;
pub mod plan;
pub mod plugin;
pub mod quant;
pub mod recover;
pub mod safetensors;
pub mod serve;
pub mod sha256;
pub mod sha512;
pub mod sign;
pub mod sparse;
pub mod store;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod transport;
pub mod wasm;
pub mod zstd;

pub use bao::BaoTree;
pub use blake3::blake3 as blake3_256;
pub use cbor::Value;
pub use container::{
    otype, pack, seg, verify, Container, Digest, HashAlgo, Object, PackOptions, Report,
};
pub use dtype::{DType, FloatFmt, Round};
pub use expr::{Expr, Tensor};
pub use layout::Layout;
pub use model::{ModelBuilder, TensorSpec};
pub use sha256::hex;
pub use store::{ContainerStore, DirStore, FileStore, MemoryStore, Store, WritableStore};

/// The specification version this implementation targets.
pub const SPEC_VERSION: &str = "OMNI/1.0-draft";

/// Conformance profiles claimed by this build (§00.6).
pub const PROFILES: &[&str] = &["C0", "C3"];
