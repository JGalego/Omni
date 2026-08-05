//! A deterministic mutation fuzzer that runs in CI (roadmap Phase 0).
//!
//! Coverage-guided fuzzing with libFuzzer lives in `reference/fuzz/` and needs
//! a nightly toolchain, so it cannot gate every push. This is the part that
//! can: a seeded, dependency-free mutator over the conformance corpus, run for
//! a bounded number of iterations on stable, with reproduction by seed.
//!
//! It is weaker than libFuzzer and does not pretend otherwise. What it buys is
//! that a parser regression which panics on malformed input fails the build on
//! the commit that introduced it, rather than months later when somebody
//! finally runs the real fuzzer.
//!
//! ## The oracles
//!
//! Finding crashes is the obvious one, but a parser can be memory-safe and
//! still wrong. Three properties are checked on every input that parses:
//!
//! 1. **No panic.** `#![forbid(unsafe_code)]` rules out memory corruption; a
//!    panic on untrusted input is the remaining failure mode, and it is a
//!    denial of service in anything that loads models from a hub.
//! 2. **Canonical decoding is a fixed point.** If `decode(x)` succeeds then
//!    `x` was canonical, so re-encoding must reproduce `x` byte for byte.
//!    Any input where it does not is either a decoder that accepted
//!    non-canonical bytes or an encoder that does not agree with it — and
//!    either one breaks content addressing, because two writers would give
//!    the same value two digests.
//! 3. **Accepting implies verifiable.** If a container opens, verification
//!    must reach a verdict rather than panicking partway through.

use omni_core::{cbor, container::scan_segments, verify, Container};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

/// xorshift64*. Not cryptographic — it needs to be reproducible from a seed
/// and identical on every platform, which rules out anything from the standard
/// library's hasher.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// Mutations chosen for what actually breaks binary parsers: length and offset
/// fields, not random bytes in payloads.
fn mutate(input: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut b = input.to_vec();
    if b.is_empty() {
        return vec![0u8; rng.below(64)];
    }
    let rounds = 1 + rng.below(8);
    for _ in 0..rounds {
        // An earlier round may have truncated the buffer to nothing; the
        // mutations below index into it.
        if b.is_empty() {
            b.push(rng.next() as u8);
        }
        match rng.below(8) {
            0 => {
                // Flip a bit.
                let i = rng.below(b.len());
                b[i] ^= 1 << rng.below(8);
            }
            1 => {
                // Replace a byte with an interesting one.
                const INTERESTING: [u8; 8] = [0x00, 0x01, 0x7f, 0x80, 0xff, 0xfe, 0x20, 0x89];
                let i = rng.below(b.len());
                b[i] = INTERESTING[rng.below(INTERESTING.len())];
            }
            2 => {
                // Overwrite an aligned 8-byte field with an extreme value —
                // the classic offset/length overflow.
                const EXTREMES: [u64; 6] =
                    [0, 1, u64::MAX, u64::MAX - 1, 1 << 63, 0x0000_0000_FFFF_FFFF];
                if b.len() >= 8 {
                    let i = rng.below(b.len() - 7) & !7;
                    let v = EXTREMES[rng.below(EXTREMES.len())];
                    b[i..i + 8].copy_from_slice(&v.to_le_bytes());
                }
            }
            3 => {
                // Truncate.
                let n = rng.below(b.len());
                b.truncate(n);
            }
            4 => {
                // Extend with garbage.
                let n = rng.below(256);
                for _ in 0..n {
                    b.push(rng.next() as u8);
                }
            }
            5 => {
                // Splice a run from elsewhere in the same input, which tends to
                // produce structurally plausible nonsense.
                if b.len() > 16 {
                    let len = 1 + rng.below(b.len() / 4);
                    let from = rng.below(b.len() - len);
                    let to = rng.below(b.len() - len);
                    let piece: Vec<u8> = b[from..from + len].to_vec();
                    b[to..to + len].copy_from_slice(&piece);
                }
            }
            6 => {
                // Zero a span, the way a bad disk does.
                let len = 1 + rng.below(b.len());
                let at = rng.below(b.len() - len + 1);
                b[at..at + len].fill(0);
            }
            _ => {
                // Deep nesting, to probe the recursion bound.
                let depth = rng.below(2048);
                let mut deep = vec![0x81u8; depth]; // array(1), repeated
                deep.push(0x00);
                b = deep;
            }
        }
    }
    b
}

struct Finding {
    what: &'static str,
    seed: u64,
    iteration: u64,
    input: Vec<u8>,
}

/// Runs `iterations` mutations seeded from the corpus. Returns findings.
pub fn run(corpus_dir: &Path, iterations: u64, seed: u64, out_dir: Option<&Path>) -> u8 {
    let seeds = load_seeds(corpus_dir);
    if seeds.is_empty() {
        eprintln!(
            "omni-conformance: no seed files under {}",
            corpus_dir.display()
        );
        return 2;
    }
    println!(
        "fuzzing {} seeds for {iterations} iterations (seed {seed})",
        seeds.len()
    );

    // Panics are the finding, so they must not also be noise on stderr.
    let previous = std::panic::take_hook();
    if std::env::var("OMNI_FUZZ_LOUD").is_err() {
        std::panic::set_hook(Box::new(|_| {}));
    }

    let mut rng = Rng(seed | 1);
    let mut findings: Vec<Finding> = Vec::new();
    let mut accepted = 0u64;

    for i in 0..iterations {
        let base = &seeds[rng.below(seeds.len())];
        let input = mutate(base, &mut rng);

        // Oracle 1 and 3: the container path.
        let r = catch_unwind(AssertUnwindSafe(|| {
            if let Ok(c) = Container::open(input.clone()) {
                let _ = verify(&c);
                return true;
            }
            let _ = scan_segments(&input);
            false
        }));
        match r {
            Err(_) => findings.push(Finding {
                what: "panic while opening or verifying a container",
                seed,
                iteration: i,
                input: input.clone(),
            }),
            Ok(true) => accepted += 1,
            Ok(false) => {}
        }

        // Oracle 1 and 2: the encoding path.
        let r = catch_unwind(AssertUnwindSafe(|| match cbor::decode(&input) {
            Ok(v) => Some(v.encode()),
            Err(_) => None,
        }));
        match r {
            Err(_) => findings.push(Finding {
                what: "panic while decoding CBOR",
                seed,
                iteration: i,
                input: input.clone(),
            }),
            Ok(Some(re)) if re != input => findings.push(Finding {
                what: "canonical decode is not a fixed point: re-encoding changed the bytes",
                seed,
                iteration: i,
                input: input.clone(),
            }),
            Ok(_) => {}
        }

        if findings.len() >= 32 {
            break;
        }
    }

    std::panic::set_hook(previous);

    if findings.is_empty() {
        println!("{iterations} iterations, {accepted} inputs opened as containers, 0 findings");
        return 0;
    }

    println!("\n{} findings:", findings.len());
    for f in &findings {
        println!(
            "  {} (seed {}, iteration {}, {} bytes)",
            f.what,
            f.seed,
            f.iteration,
            f.input.len()
        );
        if let Some(dir) = out_dir {
            std::fs::create_dir_all(dir).ok();
            let p = dir.join(format!("finding-{}-{}.bin", f.seed, f.iteration));
            if std::fs::write(&p, &f.input).is_ok() {
                println!("    written to {}", p.display());
            }
        }
    }
    println!("\nreproduce with: omni-conformance fuzz <corpus> --seed {seed}");
    1
}

fn load_seeds(dir: &Path) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(p) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&p) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "omni" || x == "cbor") {
                if let Ok(b) = std::fs::read(&path) {
                    out.push(b);
                }
            }
        }
    }
    out.sort();
    out
}
