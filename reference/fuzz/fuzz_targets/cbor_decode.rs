//! Canonical CBOR decoding, plus the round-trip property.
//!
//! If `decode` accepts an input then the input was canonical, so re-encoding
//! must reproduce it byte for byte. A violation is not merely cosmetic: two
//! writers would give the same value two digests, and content addressing would
//! stop working.
#![no_main]

use libfuzzer_sys::fuzz_target;
use omni_core::cbor;

fuzz_target!(|data: &[u8]| {
    if let Ok(v) = cbor::decode(data) {
        let re = v.encode();
        assert_eq!(
            re, data,
            "canonical decode must be a fixed point: re-encoding changed the bytes"
        );
        // Decoding our own output must also succeed and be idempotent.
        let again = cbor::decode(&re).expect("re-encoded canonical bytes must decode");
        assert_eq!(again.encode(), re);
    }
});
