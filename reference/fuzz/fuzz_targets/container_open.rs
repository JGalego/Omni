//! The whole read path on untrusted bytes: header, trailer, superblock, index,
//! and full verification of anything that opens.
//!
//! This is the target that matters most, because it is the code a model hub's
//! users run on files a model hub's users uploaded.
#![no_main]

use libfuzzer_sys::fuzz_target;
use omni_core::{verify, Container};

fuzz_target!(|data: &[u8]| {
    if let Ok(c) = Container::open(data.to_vec()) {
        // Opening is not the end of the attack surface — verification walks
        // every segment, every index entry and every object.
        let _ = verify(&c);
        let _ = c.segments();
        let _ = c.root();
        for e in c.index.iter().take(64) {
            let _ = c.get(&e.digest);
        }
    }
});
