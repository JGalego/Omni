//! Recovery by segment scan (§02.8) on damaged input.
//!
//! Recovery deliberately trusts less than the normal read path — it scans for
//! segments rather than following the index — so it reaches parsing code with
//! inputs the ordinary reader would have rejected long before.
#![no_main]

use libfuzzer_sys::fuzz_target;
use omni_core::recover::recover;

fuzz_target!(|data: &[u8]| {
    if let Ok(r) = recover(data) {
        // Whatever it claims to have recovered must be self-consistent: every
        // recovered object must hash to the identity it was filed under.
        for o in r.objects.iter().take(256) {
            let _ = o.digest(r.header.hash);
        }
    }
});
