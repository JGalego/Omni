//! `omni-conformance` — corpus generator and test runner (§15.3).
//!
//! ```console
//! $ omni-conformance generate ../../conformance
//! $ omni-conformance run ../../conformance --impl ../target/release/omni
//! ```
//!
//! The runner drives an implementation as a subprocess and judges it by its
//! exit code, using the semantic codes from `docs/design/cli.md` §10. That is
//! deliberately the lowest-common-denominator interface: an implementation
//! written in any language, by anyone, can be tested without linking against
//! this crate or agreeing with it about anything except what the specification
//! already says.

mod cases;
mod fuzz;

use cases::{corpus, Case, Expect};
use omni_core::cbor::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const SUITE_VERSION: &str = "0";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(|s| s.as_str()) {
        Some("generate") => generate(args.get(1).map(PathBuf::from)),
        Some("run") => run(&args),
        Some("list") => list(),
        Some("fuzz") => do_fuzz(&args),
        Some("codec") => do_codec(&args),
        _ => {
            eprintln!(
                "omni-conformance — OMNI conformance corpus (suite {SUITE_VERSION})\n\
                 \n\
                 USAGE:\n\
                 \x20   omni-conformance generate <dir>\n\
                 \x20   omni-conformance run <dir> --impl <command>\n\
                 \x20   omni-conformance list\n\
                 \x20   omni-conformance fuzz <dir> [--iterations N] [--seed S] [--out <dir>]\n\
                 \x20   omni-conformance codec <id> encode|decode <in> <out>\n\
                 \x20       [--level N] [--elem-size N] [--logical-len N]\n"
            );
            2
        }
    };
    ExitCode::from(code)
}

fn list() -> u8 {
    let cases = corpus();
    for c in &cases {
        println!(
            "{:<18} {:<34} {:<8} {}",
            c.category,
            c.name,
            c.expect.name(),
            c.rule.unwrap_or("-")
        );
    }
    println!("\n{} cases", cases.len());
    0
}

/// `codec <id> encode|decode <in> <out>` — the differential-testing hook.
///
/// §03.7's codecs are the one part of OMNI whose correctness is defined by
/// somebody else's bitstream specification, so the only test that means
/// anything is one that exchanges bytes with an independent implementation.
/// This subcommand is that exchange point: CI compresses with libzstd and
/// decodes here, then compresses here and decodes with libzstd.
fn do_codec(args: &[String]) -> u8 {
    let (Some(id), Some(dir), Some(input), Some(output)) =
        (args.get(1), args.get(2), args.get(3), args.get(4))
    else {
        eprintln!("omni-conformance: codec needs <id> encode|decode <in> <out>");
        return 2;
    };
    let mut level = 3u64;
    let mut elem_size = 2u64;
    let mut logical_len: Option<u64> = None;
    let mut high_ratio = false;
    let mut i = 5;
    while i < args.len() {
        if args[i] == "--level" {
            level = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(3);
            i += 2;
        } else if args[i] == "--elem-size" {
            elem_size = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(2);
            i += 2;
        } else if args[i] == "--logical-len" {
            logical_len = args.get(i + 1).and_then(|v| v.parse().ok());
            i += 2;
        } else if args[i] == "--high-ratio" {
            // §03.7.4's escape hatch, declared rather than assumed: a container
            // whose features include `omni.codec/high-ratio.1` may exceed the
            // 1000:1 bound, and a differential test over 200 KB of zeros needs
            // it.
            high_ratio = true;
            i += 1;
        } else {
            eprintln!("omni-conformance: unknown option `{}`", args[i]);
            return 2;
        }
    }
    let data = match std::fs::read(input) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("omni-conformance: {input}: {e}");
            return 2;
        }
    };
    let codec = omni_core::codec::Codec::from_value(&Value::map(vec![
        ("id", Value::text(id)),
        ("level", Value::U(level)),
        ("elem_size", Value::U(elem_size)),
    ]));
    const CAP: usize = 1 << 30;
    let out = match dir.as_str() {
        "encode" => codec.encode(&data),
        // §03.7.4's logical length is authoritative wherever an index declares
        // it, and `--logical-len` is how it gets declared here. Without it there
        // is nothing to check a ratio against, so the output is bounded and the
        // codec's own framing decides the rest.
        "decode" => match logical_len {
            Some(n) => codec.decode(&data, n, high_ratio),
            None => codec.decode_framed(&data, CAP),
        },
        _ => {
            eprintln!("omni-conformance: codec direction must be encode or decode");
            return 2;
        }
    };
    match out {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(output, &bytes) {
                eprintln!("omni-conformance: {output}: {e}");
                return 2;
            }
            println!(
                "{} {} {} B -> {} B",
                codec.name(),
                dir,
                data.len(),
                bytes.len()
            );
            0
        }
        Err(e) => {
            eprintln!("omni-conformance: {e}");
            // An unimplemented codec is indeterminate (3), a broken stream is
            // invalid (1) — the same distinction the CLI draws everywhere else.
            match e {
                omni_core::codec::Error::Unsupported(_) => 3,
                _ => 1,
            }
        }
    }
}

fn generate(dir: Option<PathBuf>) -> u8 {
    let Some(dir) = dir else {
        eprintln!("omni-conformance: generate needs an output directory");
        return 2;
    };
    let cases = corpus();

    // A generator that emits cases contradicting its own reader has a bug, and
    // shipping the corpus anyway would spread it to everyone who runs it.
    let problems = cases::self_check(&cases);
    if !problems.is_empty() {
        eprintln!("omni-conformance: the corpus disagrees with this implementation:");
        for p in &problems {
            eprintln!("  {p}");
        }
        return 1;
    }

    for c in &cases {
        let path = case_path(&dir, c);
        std::fs::create_dir_all(path.parent().unwrap()).expect("create category directory");
        std::fs::write(&path, &c.bytes).expect("write case");
    }
    let manifest = manifest(&cases);
    std::fs::write(dir.join("manifest.cbor"), manifest.encode()).expect("write manifest");
    std::fs::write(dir.join("README.md"), readme(&cases)).expect("write README");

    let bytes: usize = cases.iter().map(|c| c.bytes.len()).sum();
    println!(
        "generated {} cases ({} bytes) into {}",
        cases.len(),
        bytes,
        dir.display()
    );
    0
}

fn case_path(dir: &Path, c: &Case) -> PathBuf {
    dir.join(c.category).join(format!("{}.omni", c.name))
}

/// The machine-readable corpus description, in canonical CBOR so that it is
/// itself an OMNI-encoded artifact rather than a second format to parse.
fn manifest(cases: &[Case]) -> Value {
    Value::map(vec![
        ("t", Value::text("omni.conformance/suite")),
        ("v", Value::U(1)),
        ("suite", Value::text(SUITE_VERSION)),
        ("spec", Value::text(omni_core::SPEC_VERSION)),
        (
            "cases",
            Value::Array(
                cases
                    .iter()
                    .map(|c| {
                        Value::map(vec![
                            (
                                "path",
                                Value::text(format!("{}/{}.omni", c.category, c.name)),
                            ),
                            ("category", Value::text(c.category)),
                            ("name", Value::text(c.name)),
                            ("expect", Value::text(c.expect.name())),
                            (
                                "args",
                                Value::Array(c.args.iter().map(|a| Value::text(*a)).collect()),
                            ),
                            (
                                "rule",
                                match c.rule {
                                    Some(r) => Value::text(r),
                                    None => Value::Null,
                                },
                            ),
                            ("why", Value::text(c.why)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

fn readme(cases: &[Case]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# OMNI conformance corpus (suite {SUITE_VERSION})\n\n\
         Generated by `omni-conformance generate`. Do not edit by hand — every\n\
         case is produced from source in `reference/omni-conformance/src/cases.rs`,\n\
         next to the rule it exercises and the reason it exists.\n\n\
         ```console\n\
         $ omni-conformance run conformance/ --impl ./my-reader\n\
         ```\n\n\
         The runner judges an implementation by its exit code, so any\n\
         implementation in any language can be tested without linking against\n\
         this one.\n\n\
         | Expectation | Exit code | Meaning |\n\
         |---|---|---|\n\
         | `accept` | 0 | The file is valid and must load and validate cleanly. |\n\
         | `reject` | 1 | The file violates a normative rule and must be refused. |\n\
         | `degrade` | 0 or 3 | The file uses something this version does not define. It must still open, list, copy and verify. Rejecting it is a conformance failure. |\n\n\
         The `degrade` row is the one that matters most. It is the difference\n\
         between a format that can be extended and one that cannot, and it is\n\
         the case implementations most often get wrong — refusing a whole file\n\
         because one part of it is from the future.\n\n"
    ));

    let mut current = "";
    for c in cases {
        if c.category != current {
            current = c.category;
            s.push_str(&format!(
                "## `{current}`\n\n| Case | Expect | Rule | Why |\n|---|---|---|---|\n"
            ));
        }
        s.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            c.name,
            c.expect.name(),
            c.rule.unwrap_or("—"),
            c.why
        ));
    }
    s.push_str(&format!("\n{} cases in this suite.\n", cases.len()));
    s
}

fn run(args: &[String]) -> u8 {
    let Some(dir) = args.get(1).map(PathBuf::from) else {
        eprintln!("omni-conformance: run needs a corpus directory");
        return 2;
    };
    let Some(imp) = flag(args, "--impl") else {
        eprintln!("omni-conformance: run needs --impl <command>");
        return 2;
    };

    let cases = corpus();
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut failures = Vec::new();

    for c in &cases {
        let path = case_path(&dir, c);
        if !path.exists() {
            failures.push(format!("{}/{}: case file missing", c.category, c.name));
            fail += 1;
            continue;
        }
        let out = match Command::new(imp)
            .arg("verify")
            .arg(&path)
            .args(c.args)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                eprintln!("omni-conformance: cannot run `{imp}`: {e}");
                return 2;
            }
        };
        let code = out.status.code().unwrap_or(-1);
        let text = String::from_utf8_lossy(&out.stderr).to_string()
            + &String::from_utf8_lossy(&out.stdout);

        let ok = match c.expect {
            Expect::Accept => code == 0,
            Expect::Reject => code == 1,
            Expect::Degrade => code == 0 || code == 3,
        };
        // A rejection that cites the wrong rule is still a rejection, but it
        // is worth reporting: rule IDs are how a user finds out what to fix.
        let rule_ok = match (c.expect, c.rule) {
            (Expect::Reject, Some(r)) => text.contains(r),
            _ => true,
        };

        if ok {
            pass += 1;
            let note = if rule_ok {
                String::new()
            } else {
                format!("   (expected rule {} not cited)", c.rule.unwrap())
            };
            println!(
                "case {:<48} PASS{note}",
                format!("{}/{}", c.category, c.name)
            );
        } else {
            fail += 1;
            println!("case {:<48} FAIL", format!("{}/{}", c.category, c.name));
            println!(
                "    expected: {} (exit {})",
                c.expect.name(),
                expected_code(c.expect)
            );
            println!("    actual:   exit {code}");
            let first = text.lines().next().unwrap_or("").trim();
            if !first.is_empty() {
                println!("    said:     {first}");
            }
            failures.push(format!("{}/{}", c.category, c.name));
        }
    }

    println!("\n{pass}/{} passed", cases.len());
    if fail == 0 {
        println!("suite {SUITE_VERSION}: CONFORMANT");
        0
    } else {
        println!("suite {SUITE_VERSION}: NOT CONFORMANT ({fail} failed)");
        1
    }
}

fn do_fuzz(args: &[String]) -> u8 {
    let Some(dir) = args.get(1).map(PathBuf::from) else {
        eprintln!("omni-conformance: fuzz needs a corpus directory to seed from");
        return 2;
    };
    let iterations = flag(args, "--iterations")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    // Default to a fixed seed. A fuzzer that picks a different seed every run
    // finds more over time but makes CI flaky and failures unreproducible;
    // pass --seed to explore.
    let seed = flag(args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out = flag(args, "--out").map(PathBuf::from);
    fuzz::run(&dir, iterations, seed, out.as_deref())
}

fn expected_code(e: Expect) -> &'static str {
    match e {
        Expect::Accept => "0",
        Expect::Reject => "1",
        Expect::Degrade => "0 or 3",
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(|s| s.as_str())
}
