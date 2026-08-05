//! CRC-32C (Castagnoli), reflected, poly 0x1EDC6F41 → reversed 0x82F63B78.
//!
//! Used for container framing integrity only (§02.3.2). It is not a security
//! mechanism; digests (§03.5) are.

const POLY: u32 = 0x82F6_3B78;

fn table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

/// CRC-32C over `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    // Recomputing the table is cheap relative to everything else here and keeps
    // the crate free of lazy-static machinery.
    let t = table();
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = (crc >> 8) ^ t[((crc ^ b as u32) & 0xFF) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // Standard CRC-32C check values.
        assert_eq!(crc32c(b""), 0x0000_0000);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b"a"), 0xC1D0_4330);
    }
}
