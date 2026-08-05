//! `omni` — reference CLI subset.
//!
//! Implements the verbs that exercise the container specification end to end:
//! `inspect`, `verify`, `ls`, `dump`, `cat`, `example`. The full verb set is in
//! `docs/design/cli.md`; everything not here is unimplemented, not silently
//! degraded.

use omni_core::cbor::Value;
use omni_core::container::{otype, seg, IndexEntry};
use omni_core::recover::recover;
use omni_core::store::{copy_reachable, walk, DirStore, EnumerableStore, Store};
use omni_core::{
    hex, pack, verify, Container, ContainerStore, DType, HashAlgo, ModelBuilder, PackOptions,
    TensorSpec,
};
use std::process::ExitCode;

/// `println!` that treats a closed pipe as success rather than panicking, so
/// `omni ls big.omni | head` behaves like every other Unix tool. Must be used
/// only inside functions returning `R`.
macro_rules! pr {
    () => { pr!("") };
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut o = std::io::stdout().lock();
        match writeln!(o, $($t)*) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => return Ok(0),
            Err(e) => return Err(Box::new(e)),
        }
    }};
}

/// As `pr!`, without a trailing newline.
macro_rules! prr {
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut o = std::io::stdout().lock();
        match write!(o, $($t)*) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => return Ok(0),
            Err(e) => return Err(Box::new(e)),
        }
    }};
}

const USAGE: &str = "\
omni — Open Model Neutral Interchange (reference implementation)

USAGE:
    omni <verb> [args]

VERBS:
    inspect <file>            Summarize a container without reading tensor payloads
    verify  <file> [--level N]  Validate (V0-V4); exit 1 invalid, 3 indeterminate
    ls      <file>            List objects in the index
    dump    <file> --header   Annotated hexdump of the 128-byte file header
    dump    <file> --object <hex>   CBOR diagnostic notation for one object
    cat     <file> --tensor <name> --hex [--limit N]
    pack    <dir.omnid> -o <file.omni> [--align N]
                              Build a container from a directory store
    unpack  <file.omni> -o <dir.omnid>
                              Explode a container into a directory store
    fsck    <file> [--rebuild -o <out.omni>]
                              Diagnose damage; rebuild by segment scan (§02.8)
    example <out.omni> [--hash blake3|sha256]
                              Build a small but complete example container

EXIT CODES (docs/design/cli.md §10):
    0 ok · 1 invalid · 2 usage · 3 indeterminate · 5 incomplete
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    }
    let r = match args[0].as_str() {
        "inspect" => run(&args, cmd_inspect),
        "verify" => run(&args, cmd_verify),
        "ls" => run(&args, cmd_ls),
        "dump" => run(&args, cmd_dump),
        "cat" => run(&args, cmd_cat),
        "pack" => cmd_pack(&args),
        "unpack" => cmd_unpack(&args),
        // fsck must work on files that do not open, so it does not go through
        // `run`, which opens the container first.
        "fsck" => cmd_fsck(&args),
        "example" => cmd_example(&args),
        "-h" | "--help" | "help" => cmd_help(),
        "--version" => cmd_version(),
        other => {
            eprintln!("unknown verb `{other}`\n");
            eprint!("{USAGE}");
            Ok(2)
        }
    };
    match r {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

type R = Result<u8, Box<dyn std::error::Error>>;

fn cmd_help() -> R {
    prr!("{USAGE}");
    Ok(0)
}

fn cmd_version() -> R {
    pr!(
        "omni {} ({}, profiles {})",
        env!("CARGO_PKG_VERSION"),
        omni_core::SPEC_VERSION,
        omni_core::PROFILES.join("+")
    );
    Ok(0)
}

fn run(args: &[String], f: impl Fn(&Container, &[String]) -> R) -> R {
    if args.len() < 2 {
        eprint!("{USAGE}");
        return Ok(2);
    }
    let bytes = std::fs::read(&args[1])?;
    let size = bytes.len();
    let c = Container::open(bytes)?;
    if c.header.file_size != 0 && c.header.file_size as usize != size {
        eprintln!(
            "warning: header file_size {} != actual {}",
            c.header.file_size, size
        );
    }
    f(&c, args)
}

fn human(n: u64) -> String {
    const U: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

fn as_ref_digest(v: &Value) -> Option<[u8; 32]> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    let b = a[1].as_bytes()?;
    if b.len() != 32 {
        return None;
    }
    let mut d = [0u8; 32];
    d.copy_from_slice(b);
    Some(d)
}

fn cmd_inspect(c: &Container, _args: &[String]) -> R {
    let h = &c.header;
    pr!(
        "{:<28} {:>16}",
        std::env::args().nth(2).unwrap_or_default(),
        human(h.file_size)
    );
    pr!(
        "  container   OMNI/{}.{}  profile={}  align={}  hash={}  {}",
        h.container_major,
        h.container_minor,
        match h.profile {
            0 => "core",
            1 => "stream",
            2 => "append",
            3 => "archive",
            4 => "cache",
            _ => "?",
        },
        1u64 << h.log2_align,
        h.hash.name(),
        if h.flags & 1 != 0 {
            "sealed"
        } else {
            "unsealed"
        }
    );
    pr!("  creator     {}", h.creator);
    pr!("  uuid        {}", hex(&h.uuid));
    pr!("  root        {}", short(h.hash, &h.root_digest));

    // Manifest → Metadata, without touching a tensor payload.
    let manifest = c.root()?;
    pr!();
    if let Some(kind) = manifest.get("kind").and_then(|v| v.as_str()) {
        pr!("manifest    kind={kind}");
    }
    if let Some(meta_d) = manifest.get("meta").and_then(as_ref_digest) {
        let meta = c.get_value(&meta_d)?;
        let name = meta
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(unnamed)");
        pr!("model  {name}");
        match meta.get("arch") {
            Some(a) => {
                pr!(
                    "  architecture  {}",
                    a.get("family")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(not stated)")
                );
                if let Some(Value::Map(p)) = a.get("params") {
                    let mut parts = Vec::new();
                    for (k, v) in p {
                        parts.push(format!("{} {}", k.as_str().unwrap_or("?"), v.diag()));
                    }
                    if !parts.is_empty() {
                        pr!("  params        {}", parts.join("  "));
                    }
                }
            }
            None => pr!("  architecture  (not stated)"),
        }
        match meta.get("params_total").and_then(|v| v.as_u64()) {
            Some(p) => pr!("  parameters    {}", commas(p)),
            None => pr!("  parameters    (not stated)"),
        }
        match meta
            .get("license")
            .and_then(|l| l.get("spdx"))
            .and_then(|v| v.as_str())
        {
            Some(l) => pr!("  license       {l}"),
            None => pr!("  license       (not stated)"),
        }
    }

    if let Some(f) = manifest.get("features") {
        let req = f
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        pr!("  features      required: {}", req.join(", "));
    }

    // Tensors, from descriptors only.
    let mut n_tensors = 0usize;
    let mut tensor_bytes = 0u64;
    let mut rows: Vec<(String, String, String, u64)> = Vec::new();
    if let Some(model_d) = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(as_ref_digest)
    {
        let model = c.get_value(&model_d)?;
        if let Some(tt) = model.get("tensors").and_then(as_ref_digest) {
            let table = c.get_value(&tt)?;
            if let Some(Value::Map(m)) = table.get("tensors") {
                n_tensors = m.len();
                for (k, v) in m {
                    let Some(d) = as_ref_digest(v) else { continue };
                    let desc = c.get_value(&d)?;
                    let shape: Vec<u64> = desc
                        .get("shape")
                        .and_then(|s| s.as_array())
                        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
                        .unwrap_or_default();
                    let dt = desc
                        .get("dtype")
                        .and_then(|d| {
                            d.get("alias")
                                .and_then(|x| x.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "(structural)".into());
                    let bytes = desc
                        .get("value")
                        .and_then(|v| v.get("chunks"))
                        .and_then(as_ref_digest)
                        .and_then(|cd| c.get_value(&cd).ok())
                        .and_then(|cl| cl.get("total").and_then(|t| t.as_u64()))
                        .unwrap_or(0);
                    tensor_bytes += bytes;
                    rows.push((
                        k.as_str().unwrap_or("?").to_string(),
                        format!(
                            "[{}]",
                            shape
                                .iter()
                                .map(|x| x.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        ),
                        dt,
                        bytes,
                    ));
                }
            }
        }
    }
    let unique_blob_bytes: u64 = c
        .index
        .iter()
        .filter(|e| e.otype == otype::BLOB)
        .map(|e| e.logical_len)
        .sum();

    pr!();
    pr!("tensors       {:<6} {:>28}", n_tensors, human(tensor_bytes));
    rows.sort_by_key(|r| std::cmp::Reverse(r.3));
    for (name, shape, dt, bytes) in rows.iter().take(8) {
        pr!(
            "  {:<44} {:<14} {:<8} {:>10}",
            name,
            shape,
            dt,
            human(*bytes)
        );
    }
    if rows.len() > 8 {
        pr!("  … {} more (use `omni ls`)", rows.len() - 8);
    }
    if unique_blob_bytes < tensor_bytes {
        let saved = tensor_bytes - unique_blob_bytes;
        pr!(
            "  dedup         {} logical → {} stored ({:.1}% saved, shared chunk objects)",
            human(tensor_bytes),
            human(unique_blob_bytes),
            100.0 * saved as f64 / tensor_bytes as f64
        );
    }

    pr!();
    pr!("graph         none (weights-only)");
    pr!("tokenizer     (not present)");
    pr!("adapters      none");
    pr!(
        "signatures    {}",
        if c.header.flags & 8 != 0 {
            "present"
        } else {
            "none"
        }
    );

    let stats = c.superblock.get("stats");
    pr!();
    pr!(
        "objects       {} in index ({} structure, {} blob)",
        c.index.len(),
        c.index.iter().filter(|e| e.otype != otype::BLOB).count(),
        c.index.iter().filter(|e| e.otype == otype::BLOB).count()
    );
    if let Some(s) = stats {
        if let Some(b) = s.get("bytes_logical").and_then(|v| v.as_u64()) {
            pr!(
                "storage       {} logical in objects · {} container overhead",
                human(b),
                human(c.header.file_size.saturating_sub(b))
            );
        }
    }

    // The point of §06.12: everything above came from a bounded read.
    let structural: u64 = c
        .index
        .iter()
        .filter(|e| e.otype != otype::BLOB)
        .map(|e| e.stored_len)
        .sum();
    let idx_len = c
        .superblock
        .get("index")
        .and_then(|i| i.get("len"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    pr!();
    pr!(
        "read          header 128 B + trailer 64 B + superblock {} B + index {} B + structure {} B",
        c.header.front_sb_len,
        idx_len,
        structural
    );
    pr!(
        "              = {} total, 0 tensor payload bytes",
        human(128 + 64 + c.header.front_sb_len + idx_len + structural)
    );
    Ok(0)
}

fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn short(algo: HashAlgo, d: &[u8]) -> String {
    format!("{}:{}…", algo.prefix(), &hex(d)[..16])
}

fn cmd_verify(c: &Container, _args: &[String]) -> R {
    let r = verify(c)?;
    pr!("V0 framing     ✓ {} segments", r.segments.len());
    for (off, kind, len) in &r.segments {
        pr!("     {:#010x}  {:<6} {:>12} B", off, seg::name(*kind), len);
    }
    pr!(
        "V0 padding     {} (R-C07 zero fill)",
        if r.padding_ok { "✓" } else { "✗" }
    );
    pr!(
        "V0 alignment   {} (R-C08 data objects on {}-byte boundaries)",
        if r.alignment_ok { "✓" } else { "✗" },
        1u64 << c.header.log2_align
    );
    pr!(
        "V1 index       ✓ {} entries, sorted, complete",
        c.index.len()
    );
    pr!(
        "V2 structure   ✓ canonical CBOR, schemas present on {} objects",
        c.index.iter().filter(|e| e.otype != otype::BLOB).count()
    );
    pr!(
        "V3 integrity   ✓ {} objects, {} verified (R-O01)",
        r.objects_verified,
        human(r.bytes_verified)
    );
    if r.dangling.is_empty() {
        pr!(
            "V4 graph       ✓ {} objects reachable from root",
            r.reachable
        );
    } else {
        pr!(
            "V4 graph       ⚠ {} reachable, {} dangling ref(s):",
            r.reachable,
            r.dangling.len()
        );
        for d in &r.dangling {
            pr!("     {}", short(c.header.hash, d));
        }
        pr!("\nincomplete: valid container, objects missing from all stores");
        return Ok(5);
    }
    if !r.padding_ok || !r.alignment_ok {
        return Ok(1);
    }
    pr!("\nvalid");
    Ok(0)
}

fn cmd_ls(c: &Container, _args: &[String]) -> R {
    pr!(
        "{:<20} {:<16} {:>12} {:>12}  {}",
        "DIGEST",
        "TYPE",
        "OFFSET",
        "BYTES",
        "FLAGS"
    );
    let mut entries: Vec<&IndexEntry> = c.index.iter().collect();
    entries.sort_by_key(|e| e.offset);
    for e in entries {
        pr!(
            "{:<20} {:<16} {:>12} {:>12}  {}",
            &hex(&e.digest)[..18],
            otype::name(e.otype),
            e.offset,
            e.stored_len,
            flags_str(e.oflags)
        );
    }
    Ok(0)
}

fn flags_str(f: u8) -> String {
    let mut s = Vec::new();
    if f & 1 != 0 {
        s.push("critical");
    }
    if f & 2 != 0 {
        s.push("cacheable");
    }
    if f & 4 != 0 {
        s.push("external");
    }
    if f & 8 != 0 {
        s.push("lossy");
    }
    if f & 64 != 0 {
        s.push("safe-to-copy");
    }
    s.join(",")
}

fn cmd_dump(c: &Container, args: &[String]) -> R {
    if args.iter().any(|a| a == "--header") {
        return dump_header(c);
    }
    if let Some(i) = args.iter().position(|a| a == "--object") {
        let want = args
            .get(i + 1)
            .ok_or("--object needs a hex digest prefix")?;
        let matches: Vec<_> = c
            .index
            .iter()
            .filter(|e| hex(&e.digest).starts_with(want))
            .collect();
        match matches.len() {
            0 => {
                eprintln!("no object matching {want}");
                return Ok(5);
            }
            1 => {}
            n => {
                eprintln!("{n} objects match {want}");
                return Ok(2);
            }
        }
        let e = matches[0];
        pr!("; {} object {}", otype::name(e.otype), hex(&e.digest));
        pr!("; {} bytes at offset {}", e.stored_len, e.offset);
        if e.otype == otype::BLOB {
            let b = c.get(&e.digest)?;
            hexdump(b, e.offset, 256)?;
        } else {
            pr!("{}", c.get_value(&e.digest)?.diag());
        }
        return Ok(0);
    }
    eprint!("{USAGE}");
    Ok(2)
}

fn dump_header(c: &Container) -> R {
    let b = &c.bytes[..128];
    let fields: &[(usize, usize, &str)] = &[
        (0, 8, "magic  \\x89 O M N I \\r \\n \\x1a"),
        (8, 2, "container_major"),
        (10, 2, "container_minor"),
        (12, 1, "byte_order (01 = little)"),
        (13, 1, "log2_align"),
        (14, 2, "header_size"),
        (16, 16, "file_uuid (UUIDv7-shaped, derived)"),
        (
            32,
            1,
            &format!(
                "hash_algo (0x{:02x} = {})",
                c.header.hash.code(),
                c.header.hash.name()
            ),
        ),
        (33, 1, "profile (0 = core)"),
        (34, 1, "digest_len"),
        (35, 1, "reserved0"),
        (36, 4, "flags"),
        (40, 8, "front_sb_off"),
        (48, 8, "front_sb_len"),
        (56, 8, "file_size"),
        (64, 32, "root_digest"),
        (96, 16, "creator"),
        (112, 8, "created_unix_ms"),
        (120, 4, "reserved1"),
        (124, 4, "header_crc32c"),
    ];
    pr!("OMNI FileHeader (§02.3)\n");
    for (off, len, name) in fields {
        let bytes = &b[*off..*off + *len];
        let hexs: Vec<String> = bytes.iter().map(|x| format!("{x:02x}")).collect();
        let shown = if hexs.len() > 16 {
            format!("{} …", hexs[..16].join(" "))
        } else {
            hexs.join(" ")
        };
        pr!("  {:>3}  {:<50}  {}", off, shown, name);
    }
    pr!("\nraw:");
    hexdump(b, 0, 128)?;
    Ok(0)
}

fn hexdump(b: &[u8], base: u64, limit: usize) -> R {
    for (i, row) in b
        .iter()
        .take(limit)
        .collect::<Vec<_>>()
        .chunks(16)
        .enumerate()
    {
        let off = base + (i * 16) as u64;
        let hexs: Vec<String> = row.iter().map(|x| format!("{x:02x}")).collect();
        let ascii: String = row
            .iter()
            .map(|&&x| {
                if (0x20..0x7f).contains(&x) {
                    x as char
                } else {
                    '.'
                }
            })
            .collect();
        let mut h = hexs.join(" ");
        if h.len() < 47 {
            h.push_str(&" ".repeat(47 - h.len()));
        }
        pr!("{off:08x}  {h}  |{ascii}|");
    }
    if b.len() > limit {
        pr!("… {} more bytes", b.len() - limit);
    }
    Ok(0)
}

fn cmd_cat(c: &Container, args: &[String]) -> R {
    let name = args
        .iter()
        .position(|a| a == "--tensor")
        .and_then(|i| args.get(i + 1))
        .ok_or("--tensor <name> required")?;
    let limit: usize = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let manifest = c.root()?;
    let model_d = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(as_ref_digest)
        .ok_or("no `model` asset")?;
    let model = c.get_value(&model_d)?;
    let tt = model
        .get("tensors")
        .and_then(as_ref_digest)
        .ok_or("model has no tensor table")?;
    let table = c.get_value(&tt)?;
    let entry = table
        .get("tensors")
        .and_then(|t| t.get(name))
        .and_then(as_ref_digest)
        .ok_or_else(|| format!("no tensor named `{name}`"))?;
    let desc = c.get_value(&entry)?;
    pr!("; {name}");
    pr!("; {}", desc.diag());

    let cl = desc
        .get("value")
        .and_then(|v| v.get("chunks"))
        .and_then(as_ref_digest)
        .ok_or("tensor value is not a literal (evaluator not implemented: profile C1)")?;
    let chunklist = c.get_value(&cl)?;
    let chunks = chunklist
        .get("chunks")
        .and_then(|v| v.as_array())
        .unwrap_or(&[]);
    let mut shown = 0usize;
    for ch in chunks {
        let Some(d) = ch.get("r").and_then(as_ref_digest) else {
            continue;
        };
        let e = c.find(&d).ok_or("chunk missing")?;
        let bytes = c.get(&d)?;
        pr!(
            "; chunk {} @ file offset {}",
            short(c.header.hash, &d),
            e.offset
        );
        hexdump(bytes, e.offset, limit.saturating_sub(shown))?;
        shown += bytes.len().min(limit);
        if shown >= limit {
            break;
        }
    }
    Ok(0)
}

/// `omni unpack <file.omni> -o <dir.omnid>` — a container into a directory
/// store. No conversion happens: the objects are the same objects (§01 A5).
fn cmd_unpack(args: &[String]) -> R {
    let (Some(input), Some(out)) = (args.get(1), flag(args, "-o").or(flag(args, "--out"))) else {
        eprint!("{USAGE}");
        return Ok(2);
    };
    let src = ContainerStore::new(Container::open(std::fs::read(input)?)?);
    let (root_type, root) = src.root();
    let mut dst = DirStore::create(out, src.hash())?;
    let (copied, present) = copy_reachable(&src, &mut dst, root_type, &root)?;
    dst.set_root(root_type, &root)?;

    pr!("unpacked {input} -> {out}");
    pr!("  hash           {}", src.hash().name());
    pr!("  root           {}", short(src.hash(), &root));
    pr!("  objects        {copied} written, {present} already present");
    let unreachable = src.container().index.len() - (copied + present);
    if unreachable > 0 {
        pr!("  unreferenced   {unreachable} objects in the index, not reachable from the root");
    }
    Ok(0)
}

/// `omni pack <dir.omnid> -o <file.omni>` — the inverse. Object types are
/// recovered by walking from the root, since a directory store has no index to
/// record them in.
fn cmd_pack(args: &[String]) -> R {
    let (Some(input), Some(out)) = (args.get(1), flag(args, "-o").or(flag(args, "--out"))) else {
        eprint!("{USAGE}");
        return Ok(2);
    };
    let src = DirStore::open(input)?;
    let Some((root_type, root)) = src.root()? else {
        prr!("omni: {input} has no recorded root; nothing to pack\n");
        return Ok(2);
    };

    let w = walk(&src, root_type, &root)?;
    if !w.dangling.is_empty() {
        prr!(
            "omni: {} referenced objects are missing from the store\n",
            w.dangling.len()
        );
        for d in w.dangling.iter().take(5) {
            prr!("  {}\n", short(src.hash(), d));
        }
        return Ok(5);
    }

    let objects: Result<Vec<_>, _> = w
        .objects
        .iter()
        .map(|(d, t)| {
            src.resolve(d).map(|b| omni_core::Object {
                otype: *t,
                payload: b.expect("walk found it a moment ago"),
                oflags: 0b0100_0001, // CRITICAL | SAFE_TO_COPY
            })
        })
        .collect();
    let objects = objects?;

    let mut opts = PackOptions {
        hash: src.hash(),
        ..Default::default()
    };
    if let Some(a) = flag(args, "--align") {
        match a.parse::<u32>() {
            Ok(n) if n.is_power_of_two() && (64..=1 << 30).contains(&n) => {
                opts.log2_align = n.trailing_zeros() as u8
            }
            _ => {
                prr!("omni: --align must be a power of two between 64 and 1Gi\n");
                return Ok(2);
            }
        }
    }
    let bytes = pack(&objects, &root, &opts)?;
    std::fs::write(out, &bytes)?;

    let c = Container::open(bytes.clone())?;
    let r = verify(&c)?;
    pr!("packed {input} -> {out}");
    pr!("  size           {}", human(bytes.len() as u64));
    pr!("  hash           {}", src.hash().name());
    pr!("  root           {}", short(src.hash(), &root));
    pr!("  objects        {}", objects.len());
    pr!(
        "  verified       {} objects, {}",
        r.objects_verified,
        human(r.bytes_verified)
    );
    let stored: usize = src.iter()?.len();
    if stored > objects.len() {
        pr!(
            "  left behind    {} objects in the store, unreachable from the root",
            stored - objects.len()
        );
    }
    Ok(0)
}

/// `omni fsck <file> [--rebuild -o <out>]` — §02.8.
///
/// Unlike `verify`, this trusts nothing but the file header, and reaches its
/// conclusions by scanning rather than by reading the index. That is the whole
/// point: it is the tool for files where the index is the damaged part.
fn cmd_fsck(args: &[String]) -> R {
    let Some(input) = args.get(1) else {
        eprint!("{USAGE}");
        return Ok(2);
    };
    let bytes = std::fs::read(input)?;

    // What a normal reader would say.
    let opens = Container::open(bytes.clone());
    match &opens {
        Ok(c) => match verify(c) {
            Ok(r) => pr!(
                "normal open   ✓ valid, {} objects, {} reachable",
                c.index.len(),
                r.reachable
            ),
            Err(e) => pr!("normal open   ✗ opens but fails validation: {e}"),
        },
        Err(e) => pr!("normal open   ✗ {e}"),
    }

    let r = recover(&bytes)?;
    pr!(
        "header        ✓ OMNI/{}.{}  hash={}  align={}",
        r.header.container_major,
        r.header.container_minor,
        r.header.hash.name(),
        1u64 << r.header.log2_align
    );
    pr!("root          {}", short(r.header.hash, &r.root));

    pr!(
        "segment scan  {} segments with valid CRCs",
        r.segments.len()
    );
    for (off, kind, plen) in &r.segments {
        pr!("     {off:#010x}  {:<6} {:>10} B", seg::name(*kind), plen);
    }
    pr!("structures    {} decoded from OBJ segments", r.structures);
    pr!(
        "data objects  {} located by alignment, confirmed by hashing",
        r.blobs
    );
    if r.unaccounted_blob_bytes > 0 {
        pr!(
            "unaccounted   {} of BLOB payload matched nothing",
            human(r.unaccounted_blob_bytes)
        );
    }
    pr!(
        "graph         {} objects reachable from the root",
        r.objects.len()
    );

    if !r.missing.is_empty() {
        pr!(
            "missing       {} referenced objects could not be recovered",
            r.missing.len()
        );
        for d in r.missing.iter().take(5) {
            pr!("     {}", short(r.header.hash, d));
        }
    }

    let code = if r.complete() {
        pr!();
        pr!("recoverable   ✓ complete");
        0
    } else {
        pr!();
        pr!("recoverable   partial — the graph is incomplete (§01.4), not wrong");
        5
    };

    if args.iter().any(|a| a == "--rebuild") {
        let Some(out) = flag(args, "-o").or(flag(args, "--out")) else {
            prr!("omni: --rebuild needs -o <out.omni>\n");
            return Ok(2);
        };
        let opts = PackOptions {
            hash: r.header.hash,
            log2_align: r.header.log2_align,
            ..Default::default()
        };
        let rebuilt = pack(&r.objects, &r.root, &opts)?;
        std::fs::write(out, &rebuilt)?;
        pr!();
        pr!("rebuilt       {out}  {}", human(rebuilt.len() as u64));
        if rebuilt == bytes {
            pr!("              byte-identical to the input");
        } else if r.complete() {
            pr!("              differs from the input, which is expected when the input was damaged");
        }
        let c = Container::open(rebuilt)?;
        match verify(&c) {
            Ok(v) => pr!(
                "              verifies: {} objects, {} reachable",
                v.objects_verified,
                v.reachable
            ),
            Err(e) => {
                prr!("omni: the rebuilt container does not verify: {e}\n");
                return Ok(1);
            }
        }
    }
    Ok(code)
}

/// Value of a `--name <value>` option, if present.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(|s| s.as_str())
}

/// Builds the example container used by `examples/` and the specification's
/// worked byte layout (§02.11).
fn cmd_example(args: &[String]) -> R {
    // Every option this verb takes has a value, so a `--name value` pair can be
    // skipped wholesale to find the positional output path.
    let mut positional = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += 2;
        } else {
            positional.push(args[i].as_str());
            i += 1;
        }
    }
    let out = positional.first().copied().unwrap_or("example.omni");

    // `--hash sha256` produces a container whose every digest can be checked
    // with `sha256sum`, which is worth having even though BLAKE3-256 is the
    // default (§03.5.1).
    let algo = match flag(args, "--hash") {
        None => HashAlgo::default(),
        Some(name) => match HashAlgo::parse(name) {
            Some(a) => a,
            None => {
                prr!("omni: unknown hash algorithm `{name}` (try blake3 or sha256)");
                return Ok(2);
            }
        },
    };

    // A two-layer toy transformer: enough structure to be realistic, small
    // enough to hexdump.
    let mut b = ModelBuilder::new("omni/example-toy")
        .license("Apache-2.0")
        .arch(
            "transformer.decoder",
            vec![
                ("hidden_size", Value::U(64)),
                ("n_layers", Value::U(2)),
                ("n_heads", Value::U(4)),
                ("n_kv_heads", Value::U(2)),
                ("ffn_hidden", Value::U(128)),
                ("activation", Value::text("silu")),
                (
                    "rope",
                    Value::map(vec![
                        ("kind", Value::text("rope")),
                        ("theta", Value::F64(10000.0)),
                        ("dims", Value::U(16)),
                        ("interleaved", Value::Bool(false)),
                    ]),
                ),
            ],
        )
        .chunk_size(4096);

    let vocab = 256u64;
    let hidden = 64u64;

    // The embedding table, and `lm_head` **tied** to it — the overwhelmingly
    // common case in decoder models. safetensors and GGUF both store these
    // bytes twice; here the two TensorDescs differ (different `semantic`) but
    // reference the same ChunkList, so the payload exists once.
    let embed = pattern(DType::BF16.packed_bytes(vocab * hidden) as usize, "embed");
    b = b.tensor(TensorSpec {
        name: "model.embed_tokens.weight".into(),
        shape: vec![vocab, hidden],
        dtype: DType::BF16,
        axes: Some(vec!["vocab".into(), "hidden".into()]),
        semantic: "embedding",
        data: embed.clone(),
    });
    b = b.tensor(TensorSpec {
        name: "lm_head.weight".into(),
        shape: vec![vocab, hidden],
        dtype: DType::BF16,
        axes: Some(vec!["vocab".into(), "hidden".into()]),
        semantic: "weight",
        data: embed,
    });

    for layer in 0..2u32 {
        for (proj, out_dim) in [
            ("q_proj", 64u64),
            ("k_proj", 32),
            ("v_proj", 32),
            ("o_proj", 64),
        ] {
            let name = format!("model.layers.{layer}.attn.{proj}.weight");
            let data = pattern(DType::BF16.packed_bytes(out_dim * hidden) as usize, &name);
            b = b.tensor(TensorSpec {
                name,
                shape: vec![out_dim, hidden],
                dtype: DType::BF16,
                axes: Some(vec!["out_features".into(), "in_features".into()]),
                semantic: "weight",
                data,
            });
        }
        let name = format!("model.layers.{layer}.norm.weight");
        let data = pattern(DType::F32.packed_bytes(hidden) as usize, &name);
        b = b.tensor(TensorSpec {
            name,
            shape: vec![hidden],
            dtype: DType::F32,
            axes: Some(vec!["hidden".into()]),
            semantic: "scale",
            data,
        });
    }

    let b = b.hash(algo);
    let (objs, root) = b.build();
    let opts = PackOptions {
        hash: algo,
        ..Default::default()
    };
    let bytes = pack(&objs, &root, &opts)?;
    std::fs::write(out, &bytes)?;

    // Reproducibility is a normative writer requirement (W1).
    let again = pack(&objs, &root, &opts)?;
    assert_eq!(bytes, again, "W1: pack must be byte-reproducible");

    let c = Container::open(bytes.clone())?;
    let r = verify(&c)?;
    pr!("wrote {out}");
    pr!("  size           {}", human(bytes.len() as u64));
    pr!("  root           {}", short(algo, &root));
    pr!("  objects        {}", c.index.len());
    pr!("  reachable      {}", r.reachable);
    pr!(
        "  verified       {} objects, {}",
        r.objects_verified,
        human(r.bytes_verified)
    );
    pr!("  reproducible   ✓ (two packs byte-identical)");
    Ok(0)
}

/// Deterministic filler so example files are reproducible and their hexdumps
/// are stable across runs. Seeded by tensor name, so distinct tensors get
/// distinct bytes and any deduplication observed is real.
fn pattern(n: usize, seed: &str) -> Vec<u8> {
    let mut x = 0x811c_9dc5u32;
    for b in seed.bytes() {
        x = (x ^ b as u32).wrapping_mul(0x0100_0193);
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push((x >> 24) as u8);
    }
    v
}
