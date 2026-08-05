//! `omni` — reference CLI subset.
//!
//! Implements the verbs that exercise the container specification end to end:
//! `inspect`, `verify`, `ls`, `dump`, `cat`, `example`. The full verb set is in
//! `docs/design/cli.md`; everything not here is unimplemented, not silently
//! degraded.

use omni_core::cbor::Value;
use omni_core::container::{otype, seg, Digest, IndexEntry};
use omni_core::expr::{Ctx, Expr};
use omni_core::layout::Layout;
use omni_core::recover::recover;
use omni_core::store::{copy_reachable, walk, DirStore, EnumerableStore, Store};
use omni_core::tensor::{Severity, TensorDesc, TensorTable};
use omni_core::{
    hex, pack, quant, verify, Container, ContainerStore, DType, HashAlgo, ModelBuilder,
    PackOptions, TensorSpec,
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
    verify  <file> [--level N] [--tokenizer] [--template]
                              Validate (V0-V6); exit 1 invalid, 3 indeterminate;
                              --tokenizer and --template also run their
                              conformance vectors
    ls      <file>            List objects in the index
    dump    <file> --header   Annotated hexdump of the 128-byte file header
    dump    <file> --object <hex>   CBOR diagnostic notation for one object
    cat     <file> --tensor <name> [--hex] [--limit N] [--raw] [--with <file>]
                              Evaluate a tensor's expression and print elements;
                              --raw hexdumps the stored bytes instead, --with
                              layers another container (a delta's parent)
    deps    <file> --tensor <name> [--range A:B]
                              What a (partial) read of that tensor must fetch
    tokenize <file> --text <string> | --ids <a,b,c>
                              Encode or decode with the container's tokenizer
    render  <file> [--message role:content]… [--var name=value]… [--inputs]
                              Render the OMNI-CT chat template (§06.9);
                              --inputs lists the variables it reads
    pack    <dir.omnid> -o <file.omni> [--align N]
                              Build a container from a directory store
    unpack  <file.omni> -o <dir.omnid>
                              Explode a container into a directory store
    fsck    <file> [--rebuild -o <out.omni>]
                              Diagnose damage; rebuild by segment scan (§02.8)
    caps    [--out caps.cbor]  Emit this build's CapabilitySet (§10.2)
    plan    <file> [--caps caps.cbor] [--objective O] [--memory N]
                              [--optimistic] [--allow-lossy]
                              Resolve a model against a runtime (§10.5)
    keygen  [--out key.hex] [--seed <hex>]
                              Make an Ed25519 signing key (§12.5.1)
    sign    <file> --key <hex> [-o <out.omni>] [--purpose P] [--counter N]
                              Sign a manifest and embed the attestation
    sign    --verify <file> --key <pubkey-hex>[,<hex>…] [--require any|all|k:N]
                              V7: authenticity against a trust policy
    delta   <base.omni> <tuned.omni> -o <delta.omni> [--max-err E]
                              Express one model as a delta over another (§08.6)
    adapter make  <base.omni> -o <lora.omni> [--rank R] [--alpha A]
                              [--targets <glob>]  Build a LoRA over that base
    adapter check <base.omni> <adapter.omni>
                              Validate attachment before loading weights (§08.3)
    example <out.omni> [--hash blake3|sha256] [--quantized] [--tune N]
                              [--tokenizer] [--chat-template]
                              Build a small but complete example container;
                              --quantized exercises the value layer (int4 +
                              per-group scales + a LoRA, all as expressions);
                              --tune N fills the weights with plausible values
                              and, for N > 0, applies a seeded rank-1 update —
                              two containers that differ by a fine-tune;
                              --tokenizer attaches a byte-level tokenizer and
                              --chat-template an OMNI-CT template, both with
                              conformance vectors

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
        "deps" => run(&args, cmd_deps),
        "tokenize" => run(&args, cmd_tokenize),
        "render" => run(&args, cmd_render),
        "pack" => cmd_pack(&args),
        "unpack" => cmd_unpack(&args),
        // fsck must work on files that do not open, so it does not go through
        // `run`, which opens the container first.
        "fsck" => cmd_fsck(&args),
        "example" => cmd_example(&args),
        "caps" => cmd_caps(&args),
        "plan" => run(&args, cmd_plan),
        "keygen" => cmd_keygen(&args),
        "sign" => cmd_sign(&args),
        "delta" => cmd_delta(&args),
        "adapter" => cmd_adapter(&args),
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

/// A [`Store`] view over a borrowed container.
///
/// [`ContainerStore`] takes ownership, which the read-only verbs cannot give it,
/// and the evaluator only ever needs `resolve`.
struct Borrowed<'a>(&'a Container);

impl Store for Borrowed<'_> {
    fn hash(&self) -> HashAlgo {
        self.0.header.hash
    }

    fn resolve(&self, d: &Digest) -> Result<Option<Vec<u8>>, omni_core::store::Error> {
        match self.0.read(d) {
            Ok(b) => Ok(Some(b)),
            Err(omni_core::container::Error::NotFound(_)) => Ok(None),
            Err(e) => Err(omni_core::store::Error::Corrupt(e.to_string())),
        }
    }

    fn has(&self, d: &Digest) -> Result<bool, omni_core::store::Error> {
        Ok(self.0.find(d).is_some())
    }
}

/// The model's tensor table, from the manifest.
fn tensor_table(c: &Container) -> Result<TensorTable, Box<dyn std::error::Error>> {
    let manifest = c.root()?;
    let model_d = manifest
        .get("assets")
        .and_then(|a| a.get("model"))
        .and_then(as_ref_digest)
        .ok_or("this container has no `model` asset")?;
    let model = c.get_value(&model_d)?;
    let tt = model
        .get("tensors")
        .and_then(as_ref_digest)
        .ok_or("the model has no tensor table")?;
    Ok(TensorTable::from_value(&c.get_value(&tt)?)?)
}

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

    // §05.3: there is no "the model's quantization", so report the histogram.
    // Everything here comes from descriptors; no tensor payload is read.
    //
    // Rows count tensors and the parameters they describe. Bytes are
    // deliberately *not* attributed per row: one set of stored int4 weights can
    // back three realizations (§04.8), and splitting its bytes between them
    // would be an invention. The honest byte number is the whole-model one
    // below.
    if let Ok(table) = tensor_table(c) {
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        let mut rows: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
        let machinery = omni_core::tensor::scheme_parameters(&ctx, &table);
        for r in table.tensors.values() {
            let Ok(d) = TensorDesc::load(&ctx, r) else {
                continue;
            };
            if let Expr::Literal { chunks, .. } = &d.value {
                // Scales, zero points and permutations are machinery, not a
                // precision choice, so they do not get a row.
                if machinery.contains(chunks) {
                    continue;
                }
            }
            let e = rows.entry(quant::describe(&d.value)).or_insert((0, 0));
            e.0 += 1;
            e.1 += d.numel().unwrap_or(0);
        }
        if rows.len() > 1 {
            pr!();
            pr!("quantization");
            for (label, (tensors, params)) in &rows {
                pr!(
                    "  {:<28} {:>4} tensors  {:>14} params",
                    label,
                    tensors,
                    commas(*params)
                );
            }
        }
        // The honest number §05.3 asks for: stored bytes over parameter count.
        if let Some(p) = c
            .root()?
            .get("meta")
            .and_then(as_ref_digest)
            .and_then(|d| c.get_value(&d).ok())
            .and_then(|m| m.get("params_total").and_then(|v| v.as_u64()))
        {
            if p > 0 {
                pr!(
                    "  effective bits/param       {:.2}   ({} stored / {} parameters)",
                    unique_blob_bytes as f64 * 8.0 / p as f64,
                    human(unique_blob_bytes),
                    commas(p)
                );
            }
        }
    }

    pr!();
    // §08.7: a delta's parents are pinned by digest, and an unresolvable
    // required parent makes the container incomplete.
    let store = Borrowed(c);
    let ctx = Ctx::new(&store);
    let parents = omni_core::delta::parents(&manifest).unwrap_or_default();
    if parents.is_empty() {
        pr!("parents       none (self-contained)");
    } else {
        let chain = omni_core::delta::resolve_chain(
            &ctx,
            &c.header.root_digest,
            omni_core::delta::MAX_CHAIN_DEPTH,
        )?;
        pr!(
            "parents       {} declared, chain depth {}{}",
            parents.len(),
            chain.links.len() - 1,
            if chain.is_complete() {
                ""
            } else {
                "  ⚠ incomplete"
            }
        );
        for p in &parents {
            pr!(
                "  {:<12} {} {}",
                p.role,
                short(c.header.hash, &p.reference.1),
                p.name.clone().unwrap_or_default()
            );
            for l in &p.locators {
                pr!("               locator (advisory): {l}");
            }
        }
        for m in &chain.missing {
            pr!(
                "  missing      {} {}",
                short(c.header.hash, &m.reference.1),
                m.name.clone().unwrap_or_default()
            );
        }
    }
    let adapters = c.index.iter().filter(|e| e.otype == otype::ADAPTER).count();
    pr!(
        "graph         {}",
        if c.index.iter().any(|e| e.otype == otype::GRAPH_MODULE) {
            "present"
        } else {
            "none (weights-only)"
        }
    );
    pr!(
        "adapters      {}",
        if adapters == 0 {
            "none".to_string()
        } else {
            format!("{adapters}")
        }
    );
    // The tokenizer, from the manifest asset rather than a graph walk (§06.12).
    {
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        match tokenizer_of(c, &ctx) {
            Ok(Some(t)) => {
                let vectors = match t.vectors {
                    Some(_) => "with conformance vectors",
                    None => "no conformance vectors (§06.7.1 SHOULD)",
                };
                pr!(
                    "tokenizer     {} · {} tokens · {} merges · {vectors}",
                    t.kind.name(),
                    commas(t.vocab_size() as u64),
                    commas(t.merges.len() as u64)
                );
                for u in t.unsupported() {
                    pr!("              ⚠ {u}");
                }
            }
            Ok(None) => pr!("tokenizer     none"),
            Err(e) => pr!("tokenizer     ⚠ present but unreadable: {e}"),
        }
    }
    match chat_template_of(c) {
        Ok(Some(t)) => pr!(
            "chat template {} · inputs: {}",
            t.lang,
            t.template
                .free_vars()
                .into_iter()
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ok(None) => pr!("chat template none"),
        Err(e) => pr!("chat template ⚠ present but unreadable: {e}"),
    }
    let sigs = c
        .index
        .iter()
        .filter(|e| e.otype == otype::SIGNATURE)
        .count();
    pr!(
        "signatures    {}",
        if sigs == 0 {
            "none".to_string()
        } else {
            format!("{sigs} (use `omni verify --level 7`)")
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

fn cmd_verify(c: &Container, args: &[String]) -> R {
    let level: u8 = flag(args, "--level")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
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
    if r.mistyped.is_empty() {
        pr!(
            "V2 structure   ✓ canonical CBOR, schemas present and agreeing on {} objects (R-O02)",
            c.index.iter().filter(|e| e.otype != otype::BLOB).count()
        );
    } else {
        pr!(
            "V2 structure   ✗ {} object(s) whose `t` contradicts the index's otype (R-O02):",
            r.mistyped.len()
        );
        for (d, ot, got) in &r.mistyped {
            pr!(
                "     {} indexed as {} but carries `{}`",
                short(c.header.hash, d),
                otype::name(*ot),
                if got.is_empty() { "(no `t`)" } else { got }
            );
        }
    }
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
    if !r.padding_ok || !r.alignment_ok || !r.mistyped.is_empty() {
        return Ok(1);
    }

    let mut invalid = 0usize;
    let mut indeterminate = 0usize;

    // V5 — semantics. The tensor rules of §15.2, decided from descriptors.
    if level >= 5 {
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        match tensor_table(c) {
            Ok(table) => {
                let mut findings = omni_core::tensor::validate_table(&ctx, &table);
                // R-M01: params_total, when declared, is recomputed.
                if let Some(meta_d) = c.root()?.get("meta").and_then(as_ref_digest) {
                    if let Some(p) = c
                        .get_value(&meta_d)
                        .ok()
                        .and_then(|m| m.get("params_total").and_then(|v| v.as_u64()))
                    {
                        findings.extend(omni_core::tensor::check_params_total(&ctx, &table, p));
                    }
                }
                let bad = findings
                    .iter()
                    .filter(|f| f.severity == Severity::Invalid)
                    .count();
                let unknown = findings.len() - bad;
                invalid += bad;
                indeterminate += unknown;
                if findings.is_empty() {
                    pr!(
                        "V5 semantics   ✓ {} tensors: shapes, dtypes, chunk sizing, layouts, \
                         schemes",
                        table.len()
                    );
                } else {
                    pr!(
                        "V5 semantics   {} {} tensor(s) checked, {bad} invalid, {unknown} \
                         indeterminate",
                        if bad > 0 { "✗" } else { "⚠" },
                        table.len()
                    );
                    for f in &findings {
                        pr!("     {f}");
                    }
                }
            }
            Err(e) => {
                pr!("V5 semantics   ⚠ {e}");
                indeterminate += 1;
            }
        }
    }

    // V6 — derived. Recompute every object a reader is allowed to throw away
    // and compare (§00.5's droppability invariant is only true if it holds).
    if level >= 6 {
        let mut checked = 0usize;
        let mut wrong = 0usize;
        // The object index is itself a derived accelerator: rebuilding it from
        // the objects it describes must reproduce it.
        let mut sorted = c.index.clone();
        sorted.sort_by_key(|e| e.digest);
        let index_ok = sorted
            .iter()
            .zip(c.index.iter())
            .all(|(a, b)| a.digest == b.digest)
            && c.index.windows(2).all(|w| w[0].digest < w[1].digest);
        checked += 1;
        if !index_ok {
            wrong += 1;
        }
        // Bao outboard trees (§13.3) are derived from their target object.
        for e in c.index.iter().filter(|e| e.otype == otype::BAO_TREE) {
            checked += 1;
            let Ok(v) = c.get_value(&e.digest) else {
                wrong += 1;
                continue;
            };
            let Some(target) = v.get("target").and_then(as_ref_digest) else {
                continue;
            };
            let Ok(bytes) = c.read(&target) else {
                indeterminate += 1;
                continue;
            };
            let granularity = v
                .get("granularity")
                .and_then(|x| x.as_u64())
                .unwrap_or(1024)
                .max(1) as u32;
            // The tree's root must be the object's own digest, which is the
            // property that makes verified streaming work at all (§13.3).
            match omni_core::BaoTree::encode(&bytes, granularity) {
                Ok((root, _)) if root == target => {}
                Ok(_) => wrong += 1,
                Err(_) => indeterminate += 1,
            }
        }
        // §06.9's `compiled` is a cached parse of `source`. A cache that
        // disagrees with its input is worse than no cache, so it is recomputed
        // here rather than trusted.
        if let Ok(Some(t)) = chat_template_of(c) {
            if t.compiled.is_some() {
                let store = Borrowed(c);
                let ctx = Ctx::new(&store);
                checked += 1;
                let bad = t
                    .check(&ctx)
                    .into_iter()
                    .filter(|f| f.contains("compiled"))
                    .count();
                wrong += bad;
            }
        }
        invalid += wrong;
        pr!(
            "V6 derived     {} {checked} derived object(s) recomputed, {wrong} mismatched",
            if wrong > 0 { "✗" } else { "✓" }
        );
    }

    // --tokenizer: run the conformance vectors of §06.7.1. A tokenizer that
    // changed during conversion is a silent quality regression everywhere else;
    // here it is a build failure.
    if args.iter().any(|a| a == "--tokenizer") {
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        match tokenizer_of(c, &ctx)? {
            None => {
                pr!("tokenizer      ⚠ this container has no `tokenizer` asset");
                indeterminate += 1;
            }
            Some(t) => {
                let un = t.unsupported();
                if un.is_empty() {
                    let report = t.check_vectors(&ctx)?;
                    if report.total == 0 {
                        pr!(
                            "tokenizer      ⚠ {} kind, {} tokens, but no conformance vectors \
                             (§06.7.1 SHOULD)",
                            t.kind.name(),
                            commas(t.vocab_size() as u64)
                        );
                        indeterminate += 1;
                    } else {
                        pr!(
                            "tokenizer      {} {report}",
                            if report.ok() { "✓" } else { "✗" }
                        );
                        for f in &report.failures {
                            pr!("     {f}");
                        }
                        if !report.ok() {
                            invalid += 1;
                        }
                    }
                } else {
                    // Encoding would produce *some* ids; they would just be the
                    // wrong ones. That is indeterminate, not valid (§15.1).
                    pr!("tokenizer      ⚠ this build cannot run it:");
                    for u in &un {
                        pr!("     {u}");
                    }
                    indeterminate += 1;
                }
                // The vocabulary and the embedding table have to agree; a
                // mismatch is the classic sign of a truncated conversion.
                if let Ok(table) = tensor_table(c) {
                    for name in ["model.embed_tokens.weight", "model.embed.weight"] {
                        let Some(entry) = table.get(name) else {
                            continue;
                        };
                        let Ok(desc) = c
                            .get_value(&entry.1)
                            .map_err(|e| e.to_string())
                            .and_then(|v| TensorDesc::from_value(&v).map_err(|e| e.to_string()))
                        else {
                            continue;
                        };
                        let row = desc
                            .axes
                            .as_ref()
                            .and_then(|a| a.iter().position(|x| x == "vocab"))
                            .unwrap_or(0);
                        if let Some(n) = desc.shape.get(row).and_then(|d| d.size()) {
                            if n != t.vocab_size() as u64 {
                                pr!(
                                    "tokenizer      ✗ vocabulary is {} tokens but `{name}` has {} \
                                     rows",
                                    commas(t.vocab_size() as u64),
                                    commas(n)
                                );
                                invalid += 1;
                            } else {
                                pr!("tokenizer      ✓ vocabulary matches `{name}` ({n} rows)");
                            }
                        }
                    }
                }
            }
        }
    }

    // --template: run the §06.9 vectors and check the template's claims about
    // itself. A chat template that drifted during conversion changes every
    // prompt the model ever sees.
    if args.iter().any(|a| a == "--template") {
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        match chat_template_of(c)? {
            None => {
                pr!("template       ⚠ this container has no `chat_template` asset");
                indeterminate += 1;
            }
            Some(t) => {
                let findings = t.check(&ctx);
                let report = t.check_vectors(&ctx)?;
                if report.total == 0 {
                    pr!("template       ⚠ no conformance vectors (§06.9)");
                    indeterminate += 1;
                } else {
                    pr!(
                        "template       {} {report}",
                        if report.ok() { "✓" } else { "✗" }
                    );
                    for f in &report.failures {
                        pr!("     {f}");
                    }
                    if !report.ok() {
                        invalid += 1;
                    }
                }
                if findings.is_empty() {
                    pr!(
                        "template       ✓ omni-ct/1, inputs: {}",
                        t.template
                            .free_vars()
                            .into_iter()
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                } else {
                    for f in &findings {
                        pr!("template       ✗ {f}");
                    }
                    invalid += findings.len();
                }
            }
        }
    }

    if level >= 7 {
        // V7 lives in `omni sign --verify`, which needs a trust policy; a
        // verifier with no keys cannot decide authenticity, and saying "valid"
        // here would be worse than saying nothing.
        pr!("V7 authenticity ⚠ no trust policy given; use `omni sign --verify <file> --key <hex>`");
        indeterminate += 1;
    }

    if invalid > 0 {
        pr!("\ninvalid: {invalid} finding(s)");
        return Ok(1);
    }
    if indeterminate > 0 {
        pr!("\nindeterminate: {indeterminate} finding(s) this build cannot decide");
        return Ok(3);
    }
    pr!("\nvalid");
    Ok(0)
}

/// Finds the `tokenizer` asset, if the manifest declares one (§06.12: the
/// manifest is where a reader looks, not a full graph walk).
fn tokenizer_of(
    c: &Container,
    ctx: &Ctx<'_>,
) -> Result<Option<omni_core::tokenizer::Tokenizer>, Box<dyn std::error::Error>> {
    let Some(r) = c
        .root()?
        .get("assets")
        .and_then(|a| a.get("tokenizer"))
        .cloned()
    else {
        return Ok(None);
    };
    let r = omni_core::expr::parse_ref_value(&r)?;
    Ok(Some(omni_core::tokenizer::Tokenizer::load(ctx, &r)?))
}

/// Finds the `chat_template` asset, if the manifest declares one.
fn chat_template_of(
    c: &Container,
) -> Result<Option<omni_core::ct::ChatTemplate>, Box<dyn std::error::Error>> {
    let Some(r) = c
        .root()?
        .get("assets")
        .and_then(|a| a.get("chat_template"))
        .cloned()
    else {
        return Ok(None);
    };
    let d = omni_core::expr::parse_ref_value(&r)?.1;
    Ok(Some(omni_core::ct::ChatTemplate::from_value(
        &c.get_value(&d)?,
    )?))
}

/// `omni render` — render the container's chat template (§06.9).
///
/// Messages come in as `--message role:content`, repeatable, and other inputs
/// as `--var name=value`. There is no JSON on the command line because the
/// template's required inputs are computable: `--inputs` prints them.
fn cmd_render(c: &Container, args: &[String]) -> R {
    let Some(t) = chat_template_of(c)? else {
        prr!("omni: this container has no `chat_template` asset\n");
        return Ok(5);
    };
    let free: Vec<String> = t.template.free_vars().into_iter().collect();
    if args.iter().any(|a| a == "--inputs") {
        pr!("; {} · inputs computed statically", t.lang);
        for v in &free {
            pr!("{v}");
        }
        return Ok(0);
    }
    let mut messages = Vec::new();
    let mut vars: Vec<(String, Value)> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--message" => {
                let Some(spec) = args.get(i + 1) else {
                    prr!("omni: --message needs `role:content`\n");
                    return Ok(2);
                };
                let Some((role, content)) = spec.split_once(':') else {
                    prr!("omni: --message wants `role:content`, got `{spec}`\n");
                    return Ok(2);
                };
                messages.push(Value::map(vec![
                    ("role", Value::text(role)),
                    ("content", Value::text(content)),
                ]));
                i += 2;
            }
            "--var" => {
                let Some(spec) = args.get(i + 1) else {
                    prr!("omni: --var needs `name=value`\n");
                    return Ok(2);
                };
                let Some((name, value)) = spec.split_once('=') else {
                    prr!("omni: --var wants `name=value`, got `{spec}`\n");
                    return Ok(2);
                };
                // `true`/`false` and integers are given their own types; the
                // value domain is small enough to infer unambiguously.
                let v = match value {
                    "true" => Value::Bool(true),
                    "false" => Value::Bool(false),
                    "null" | "none" => Value::Null,
                    other => match other.parse::<u64>() {
                        Ok(n) => Value::U(n),
                        Err(_) => Value::text(other),
                    },
                };
                vars.push((name.to_string(), v));
                i += 2;
            }
            _ => i += 1,
        }
    }
    let mut pairs: Vec<(Value, Value)> = vec![(Value::text("messages"), Value::Array(messages))];
    for (k, v) in vars {
        pairs.push((Value::text(k), v));
    }
    match t.template.render(&Value::Map(pairs)) {
        Ok(out) => {
            prr!("{out}");
            Ok(0)
        }
        Err(omni_core::ct::Error::Undefined(m)) => {
            // The inputs are knowable in advance, so say which ones.
            prr!("omni: {m}\n");
            prr!("omni: this template's inputs are: {}\n", free.join(", "));
            Ok(2)
        }
        Err(e) => {
            prr!("omni: {e}\n");
            Ok(1)
        }
    }
}

/// `omni tokenize` — encode text, or decode ids, with the container's own
/// tokenizer.
fn cmd_tokenize(c: &Container, args: &[String]) -> R {
    let store = Borrowed(c);
    let ctx = Ctx::new(&store);
    let Some(t) = tokenizer_of(c, &ctx)? else {
        prr!("omni: this container has no `tokenizer` asset\n");
        return Ok(5);
    };
    pr!(
        "; {} · {} tokens · {} merges{}",
        t.kind.name(),
        commas(t.vocab_size() as u64),
        commas(t.merges.len() as u64),
        if t.byte_fallback {
            " · byte fallback"
        } else {
            ""
        }
    );
    let un = t.unsupported();
    if !un.is_empty() {
        // Producing plausible-looking wrong ids is the failure mode this whole
        // module exists to prevent.
        prr!("omni: this build cannot run this tokenizer:\n");
        for u in &un {
            prr!("  {u}\n");
        }
        return Ok(3);
    }
    if let Some(ids) = flag(args, "--ids") {
        let ids: Result<Vec<u32>, _> = ids
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>())
            .collect();
        let text = t.decode(&ids?)?;
        pr!("{text}");
        return Ok(0);
    }
    let Some(text) = flag(args, "--text") else {
        prr!("omni: --text <string> or --ids <a,b,c> required\n");
        return Ok(2);
    };
    let ids = t.encode(text)?;
    pr!(
        "{}",
        ids.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    for id in &ids {
        pr!(
            "  {:>6}  {}",
            id,
            t.tokens
                .get(*id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<out of vocabulary>")
        );
    }
    // Round-tripping is the property a tokenizer is actually used for, so it is
    // reported rather than assumed.
    match t.decode(&ids) {
        Ok(back) if back == text => pr!("; round-trips"),
        Ok(back) => pr!("; does NOT round-trip: decoded {back:?}"),
        Err(e) => pr!("; does NOT round-trip: {e}"),
    }
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
    let raw = args.iter().any(|a| a == "--raw");

    let table = tensor_table(c)?;
    let entry = table
        .get(name)
        .ok_or_else(|| format!("no tensor named `{name}`"))?;
    let desc = TensorDesc::from_value(&c.get_value(&entry.1)?)?;
    pr!("; {name}");
    pr!(
        "; shape {:?}  dtype {}  layout {}  value {}",
        desc.shape,
        desc.dtype.label(),
        desc.layout.kind(),
        quant::describe(&desc.value)
    );

    // `--raw` shows the stored bytes of a literal, which is what a C0 reader
    // sees. Without it, the expression is evaluated — the C1 path.
    if raw {
        let Expr::Literal { chunks, .. } = &desc.value else {
            prr!(
                "omni: `--raw` needs a bare `literal` value; this tensor is `{}`\n",
                desc.value.op()
            );
            return Ok(2);
        };
        let store = Borrowed(c);
        let ctx = Ctx::new(&store);
        let bytes = ctx.chunk_bytes(chunks)?;
        pr!("; {} stored bytes", bytes.len());
        hexdump(&bytes, 0, limit)?;
        return Ok(0);
    }

    // §01.8: a delta's tensors are expressions over its parent's objects, so
    // reading one needs both containers. Layering is how a runtime holds a base
    // once and many deltas beside it.
    let store = Borrowed(c);
    let extra = match flag(args, "--with") {
        Some(path) => Some(ContainerStore::new(Container::open(std::fs::read(path)?)?)),
        None => None,
    };
    let layered = match &extra {
        Some(e) => Some(omni_core::store::Layered::new(vec![&store, e])?),
        None => None,
    };
    let backing: &dyn Store = match &layered {
        Some(l) => l,
        None => &store,
    };
    let ctx = Ctx::new(backing);
    let t = match desc.value.eval(&ctx) {
        Ok(t) => t,
        Err(e) => {
            // A tensor this build cannot materialize is indeterminate, not
            // invalid: the container may be perfectly good (§15.1).
            prr!("omni: cannot evaluate `{name}`: {e}\n");
            return Ok(3);
        }
    };
    pr!(
        "; {} elements, evaluated from {} stored object(s)",
        t.data.len(),
        desc.value.deps_all().len()
    );
    let n = t.data.len().min(limit);
    if args.iter().any(|a| a == "--hex") {
        let bytes = t.to_bytes(&desc.dtype, &desc.layout, omni_core::Round::Rne)?;
        hexdump(&bytes, 0, limit)?;
    } else {
        for chunk in t.data[..n].chunks(8) {
            let row: Vec<String> = chunk.iter().map(|x| format!("{x:>12.6}")).collect();
            pr!("  {}", row.join(" "));
        }
        if t.data.len() > n {
            pr!("  … {} more (use --limit)", t.data.len() - n);
        }
    }
    Ok(0)
}

/// `omni deps` — what a read of a tensor, or of a range of it, has to fetch.
///
/// This is §04.7.4 made visible: the point of range pushdown is that reading
/// rows 100-200 of `dequantize(literal(...))` fetches only the covering chunks,
/// and the only way to believe that is to be shown the byte ranges.
fn cmd_deps(c: &Container, args: &[String]) -> R {
    let name = args
        .iter()
        .position(|a| a == "--tensor")
        .and_then(|i| args.get(i + 1))
        .ok_or("--tensor <name> required")?;
    let table = tensor_table(c)?;
    let entry = table
        .get(name)
        .ok_or_else(|| format!("no tensor named `{name}`"))?;
    let desc = TensorDesc::from_value(&c.get_value(&entry.1)?)?;
    let total = desc.numel().unwrap_or(0);

    let range = match flag(args, "--range") {
        None => (0, total),
        Some(spec) => {
            let (a, b) = spec
                .split_once(':')
                .ok_or("--range takes A:B in logical elements")?;
            (a.parse::<u64>()?, b.parse::<u64>()?)
        }
    };
    if range.1 > total || range.0 > range.1 {
        prr!(
            "omni: range {}:{} is outside the tensor's {total} elements\n",
            range.0,
            range.1
        );
        return Ok(2);
    }
    pr!("{name}  elements {}..{} of {}", range.0, range.1, total);
    pr!("  value        {}", desc.value.op());
    pr!("  deterministic {}", desc.value.deterministic());
    let deps = desc.value.deps(range);
    if deps.is_empty() {
        pr!("  no stored bytes: this value is generated");
        return Ok(0);
    }
    let mut fetch = 0u64;
    for d in &deps {
        let (lo, hi) = d.bytes;
        fetch += hi - lo;
        match (&d.source, &d.uri) {
            (Some(r), _) => pr!(
                "  {}  bytes {}..{} ({}){}",
                short(c.header.hash, &r.1),
                lo,
                hi,
                human(hi - lo),
                if d.exact { "" } else { "  [bound]" }
            ),
            (None, Some(u)) => pr!("  extern {u}  bytes {lo}..{hi}  [never fetched implicitly]"),
            _ => {}
        }
        if let Some(why) = d.reason {
            pr!("      why not exact: {why}");
        }
    }
    let whole: u64 = desc
        .value
        .deps_all()
        .iter()
        .map(|d| d.bytes.1 - d.bytes.0)
        .sum();
    pr!(
        "  total        {} of {} ({:.1}% of a whole-tensor read)",
        human(fetch),
        human(whole),
        if whole == 0 {
            0.0
        } else {
            100.0 * fetch as f64 / whole as f64
        }
    );
    Ok(0)
}

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
                stored: None,
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
    // Options come in two shapes: valueless switches, and `--name value`
    // pairs whose value must not be mistaken for the positional output path.
    const SWITCHES: &[&str] = &["--quantized", "--tokenizer", "--chat-template"];
    let tune: Option<u64> = flag(args, "--tune").and_then(|s| s.parse().ok());
    let mut positional = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if SWITCHES.contains(&args[i].as_str()) {
            i += 1;
        } else if args[i].starts_with("--") {
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
            // Without --tune the filler is random *bytes*, which is right for
            // proving the container and useless as a weight: random bf16 bit
            // patterns span 1e±38. With it, the values are plausible and a
            // fine-tune of them is a meaningful thing to take a delta of.
            let data = match tune {
                None => pattern(DType::BF16.packed_bytes(out_dim * hidden) as usize, &name),
                Some(seed) => {
                    let mut d = floats(&DType::BF16, (out_dim * hidden) as usize, &name, 0.05);
                    if seed != 0 {
                        rank1_update(&mut d, out_dim as usize, hidden as usize, seed, 0.002);
                    }
                    d
                }
            };
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

    let mut b = b.hash(algo);
    if args.iter().any(|a| a == "--quantized") {
        b = add_quantized_layer(b);
    }
    if args.iter().any(|a| a == "--tokenizer") {
        b = add_tokenizer(b, vocab);
    }
    if args.iter().any(|a| a == "--chat-template") {
        b = add_chat_template(b);
    }
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

/// Attaches a tokenizer to the example (§06.7).
///
/// A byte-level vocabulary: token `i` is the GPT-2 printable stand-in for byte
/// `i`, so the tokenizer's vocabulary is exactly the model's `vocab` axis and
/// every input round-trips. Merges are what make BPE interesting, and there is
/// no room for a merged token in a 256-entry byte vocabulary — so this example
/// stores none, and says so rather than claiming a merge table it cannot honour.
///
/// The conformance vectors (§06.7.1) are computed from the *definition* — the
/// ids of a byte-level tokenizer are the UTF-8 bytes of the input — not by
/// running the encoder. Vectors generated by the thing they test cannot fail.
fn add_tokenizer(mut b: ModelBuilder, vocab: u64) -> ModelBuilder {
    let tokens: Vec<String> = (0..vocab)
        .map(|i| omni_core::tokenizer::byte_to_unicode(i as u8).to_string())
        .collect();
    let blob = omni_core::tokenizer::encode_vocab(&tokens);
    let vocab_expr = b.literal(&blob, DType::Str, &[vocab], Layout::default());

    let mut vectors = String::from("# text \t comma-separated ids (§06.7.1)\n");
    for text in ["hello", "Hello, world!", "  leading", "a\tb", "ol\u{e1}"] {
        let ids: Vec<String> = text.bytes().map(|x| x.to_string()).collect();
        // Tabs and newlines are the field and record separators, so the input
        // column is escaped; §06.7.1's own examples show escaped inputs.
        let escaped: String = text
            .chars()
            .map(|c| match c {
                '\t' => "\\t".to_string(),
                '\n' => "\\n".to_string(),
                '\\' => "\\\\".to_string(),
                c => c.to_string(),
            })
            .collect();
        vectors.push_str(&format!("{escaped}\t{}\n", ids.join(",")));
    }
    let vec_blob = omni_core::container::Object::blob(vectors.into_bytes());
    let vec_digest = vec_blob.digest(b.hash);
    b.extra_objects.push(vec_blob);

    let tok = Value::map(vec![
        ("t", Value::text("omni.tok/tokenizer")),
        ("v", Value::U(1)),
        ("kind", Value::text("bpe")),
        ("vocab", Value::map(vec![("tokens", vocab_expr.to_value())])),
        (
            "pretokenizers",
            Value::Array(vec![Value::map(vec![
                ("k", Value::text("byte-level")),
                ("add_prefix_space", Value::Bool(false)),
            ])]),
        ),
        (
            "decoder",
            Value::Array(vec![Value::map(vec![("k", Value::text("byte-level"))])]),
        ),
        ("byte_fallback", Value::Bool(true)),
        (
            "conformance",
            Value::map(vec![(
                "vectors",
                Value::Array(vec![Value::U(0), Value::Bytes(vec_digest.to_vec())]),
            )]),
        ),
    ]);
    b.asset("tokenizer", otype::TOKENIZER, tok)
}

/// Attaches an OMNI-CT chat template to the example (§06.9).
///
/// The same template a Jinja2 artifact would carry, in a language that cannot
/// execute anything. The `compiled` AST and the conformance vectors are both
/// stored, so `omni verify --level 6 --template` has something to recompute and
/// something to compare against.
fn add_chat_template(mut b: ModelBuilder) -> ModelBuilder {
    use omni_core::ct::{encode_vectors, Template};

    const SOURCE: &str = "\
{%- for m in messages -%}
<|{{ m.role }}|>
{{ m.content }}<|end|>
{% endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}";

    let t = Template::parse(SOURCE).expect("the example template must parse");

    // §06.9 stores the cached AST as a Blob — it is the CBOR encoding of an
    // AST, not an object with a schema of its own, and giving it the
    // ChatTemplate otype would make it contradict its own `t` (R-O02).
    let ast = omni_core::container::Object::blob(t.to_value().encode());
    let ast_digest = ast.digest(b.hash);
    b.extra_objects.push(ast);

    // Vectors: an input and the string it must render to. Written out by hand
    // rather than captured from the renderer — a vector produced by the code it
    // tests cannot fail.
    let msg = |role: &str, content: &str| {
        Value::map(vec![
            ("role", Value::text(role)),
            ("content", Value::text(content)),
        ])
    };
    let cases = vec![
        (
            Value::map(vec![
                ("messages", Value::Array(vec![msg("user", "Hi")])),
                ("add_generation_prompt", Value::Bool(false)),
            ]),
            "<|user|>\nHi<|end|>\n".to_string(),
        ),
        (
            Value::map(vec![
                (
                    "messages",
                    Value::Array(vec![msg("system", "Be brief."), msg("user", "Hi")]),
                ),
                ("add_generation_prompt", Value::Bool(true)),
            ]),
            "<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>".to_string(),
        ),
        (
            // No messages at all still renders — the loop is over a finite
            // structure, and an empty one is not a special case.
            Value::map(vec![
                ("messages", Value::Array(vec![])),
                ("add_generation_prompt", Value::Bool(true)),
            ]),
            "<|assistant|>".to_string(),
        ),
    ];
    let vec_blob = omni_core::container::Object::blob(encode_vectors(&cases));
    let vec_digest = vec_blob.digest(b.hash);
    b.extra_objects.push(vec_blob);

    let obj = Value::map(vec![
        ("t", Value::text("omni.tok/chat-template")),
        ("v", Value::U(1)),
        ("lang", Value::text(omni_core::ct::LANG)),
        ("source", Value::text(SOURCE)),
        (
            "compiled",
            Value::Array(vec![Value::U(0), Value::Bytes(ast_digest.to_vec())]),
        ),
        ("capabilities", Value::Array(vec![Value::text("system")])),
        (
            "vectors",
            Value::Array(vec![Value::U(0), Value::Bytes(vec_digest.to_vec())]),
        ),
    ]);
    b.asset("chat_template", otype::CHAT_TEMPLATE, obj)
}

/// Adds the worked example of §04.8: one set of stored bytes, four tensors.
///
/// `q` (int4), its per-group scales and zeros, and a rank-8 LoRA are the only
/// things stored. `W_bf16`, `W_lora` and `W_fp8` are *definitions* — three more
/// tensors at zero bytes, which is the whole claim of §04.1 made checkable by
/// `omni verify --level 5` and `omni cat`.
fn add_quantized_layer(mut b: ModelBuilder) -> ModelBuilder {
    use omni_core::expr::{BinOp, Scalar, Sum};
    use omni_core::tensor::Materialize;

    let (out_f, in_f, group, rank) = (32u64, 64u64, 32u64, 8u64);
    let groups = in_f / group;

    // int4 weights, eight to a 32-bit word — the GPTQ packing (§05.2.2).
    let packed = Layout::Packed {
        elems_per_word: 8,
        word_bits: 32,
        bit_order: omni_core::layout::BitOrder::LsbFirst,
        order: omni_core::layout::Order::RowMajor,
    };
    let q_bytes = pattern(
        packed
            .stored_bytes(&[out_f, in_f], &DType::U4)
            .expect("packed int4 sizes") as usize,
        "quant.q",
    );
    let q = b.literal(&q_bytes, DType::U4, &[out_f, in_f], packed);

    // Plausible magnitudes: random *bits* interpreted as bf16 span 1e±38, which
    // makes the example's output unreadable. Encode real values instead.
    let scale_bytes = floats(
        &DType::BF16,
        (out_f * groups) as usize,
        "quant.scales",
        0.01,
    );
    let scales = b.literal(
        &scale_bytes,
        DType::BF16,
        &[out_f, groups],
        Layout::default(),
    );
    let zero_bytes = vec![0x08u8; (out_f * groups / 2) as usize];
    let zeros = b.literal(&zero_bytes, DType::U4, &[out_f, groups], Layout::default());

    let a_bytes = floats(&DType::BF16, (rank * in_f) as usize, "lora.A", 0.05);
    let lora_a = b.literal(&a_bytes, DType::BF16, &[rank, in_f], Layout::default());
    let b_bytes = floats(&DType::BF16, (out_f * rank) as usize, "lora.B", 0.05);
    let lora_b = b.literal(&b_bytes, DType::BF16, &[out_f, rank], Layout::default());

    // W_bf16 = dequantize(q, {affine-sub, block [1,32], scale, zero})
    let w_bf16 = Expr::Dequantize {
        x: Box::new(q.clone()),
        scheme: Value::map(vec![
            ("scheme", Value::text("affine")),
            ("formula", Value::text("affine-sub")),
            ("out", DType::BF16.to_value()),
            ("axis", Value::U(1)),
            ("block", Value::Array(vec![Value::U(1), Value::U(group)])),
            ("scale", scales.to_value()),
            ("zero", zeros.to_value()),
        ]),
    };
    // W_lora = add(W_bf16, scale(matmul(B, A), 30/16))
    let w_lora = Expr::Bin {
        op: BinOp::Add,
        a: Box::new(w_bf16.clone()),
        b: Box::new(Expr::Scale {
            x: Box::new(Expr::MatMul {
                a: Box::new(lora_b.clone()),
                b: Box::new(lora_a.clone()),
                sum: Sum::Pairwise,
            }),
            k: Scalar::Ratio(30, 16),
        }),
    };
    // W_fp8 = cast(W_lora, f8e4m3, "rne")
    let w_fp8 = Expr::Cast {
        x: Box::new(w_lora.clone()),
        dtype: DType::F8E4M3,
        round: omni_core::Round::Rne,
    };

    let desc = |value: &Expr, dtype: DType, semantic: &str| -> TensorDesc {
        let t = value.infer().expect("the example's expressions type-check");
        TensorDesc {
            shape: t.shape,
            dtype,
            layout: Layout::default(),
            value: value.clone(),
            semantic: Some(semantic.to_string()),
            role: Some("attn.q_proj".into()),
            axes: Some(vec!["out_features".into(), "in_features".into()]),
            device_hint: None,
            materialize: Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        }
    };
    // The stored leaves, named so `omni cat --raw` can reach them.
    let stored = |value: &Expr, dtype: DType, semantic: &str, axes: Vec<&str>| -> TensorDesc {
        let t = value.infer().expect("literals type-check");
        TensorDesc {
            shape: t.shape,
            dtype,
            layout: match value {
                Expr::Literal { layout, .. } => layout.clone(),
                _ => Layout::default(),
            },
            value: value.clone(),
            semantic: Some(semantic.to_string()),
            role: None,
            axes: Some(axes.iter().map(|a| a.to_string()).collect()),
            device_hint: None,
            materialize: Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        }
    };

    b.derived(
        "model.layers.0.attn.q_proj.qweight",
        stored(&q, DType::U4, "weight", vec!["out_features", "in_features"]),
    )
    .derived(
        "model.layers.0.attn.q_proj.scales",
        stored(&scales, DType::BF16, "scale", vec!["out_features", "group"]),
    )
    .derived(
        "model.layers.0.attn.q_proj.qzeros",
        stored(&zeros, DType::U4, "zero", vec!["out_features", "group"]),
    )
    .derived(
        "lora.0.q_proj.A",
        stored(&lora_a, DType::BF16, "weight", vec!["rank", "in_features"]),
    )
    .derived(
        "lora.0.q_proj.B",
        stored(&lora_b, DType::BF16, "weight", vec!["out_features", "rank"]),
    )
    .derived(
        "model.layers.0.attn.q_proj.weight.bf16",
        desc(&w_bf16, DType::BF16, "weight"),
    )
    .derived(
        "model.layers.0.attn.q_proj.weight.lora",
        desc(&w_lora, DType::BF16, "weight"),
    )
    .derived(
        "model.layers.0.attn.q_proj.weight.fp8",
        desc(&w_fp8, DType::F8E4M3, "weight"),
    )
}

/// `omni caps` — publish what this build can do (§10.2).
fn cmd_caps(args: &[String]) -> R {
    let caps = omni_core::plan::Capabilities::reference();
    let bytes = caps.to_value().encode();
    match flag(args, "--out") {
        Some(path) => {
            std::fs::write(path, &bytes)?;
            pr!("wrote {path}  {} bytes", bytes.len());
        }
        None => pr!("{}", caps.to_value().diag()),
    }
    Ok(0)
}

/// `omni plan` — resolve a model against a runtime's capabilities (§10.5).
fn cmd_plan(c: &Container, args: &[String]) -> R {
    use omni_core::plan::{resolve, Capabilities, Objective};

    let mut caps = match flag(args, "--caps") {
        Some(path) => Capabilities::from_value(&omni_core::cbor::decode(&std::fs::read(path)?)?)?,
        None => Capabilities::reference(),
    };
    if let Some(m) = flag(args, "--memory").and_then(|s| s.parse::<u64>().ok()) {
        caps.memory_bytes = Some(m);
    }
    if args.iter().any(|a| a == "--allow-lossy") {
        caps.allow_lossy = true;
    }
    if args.iter().any(|a| a == "--no-lossy") {
        caps.allow_lossy = false;
    }
    let objective = match flag(args, "--objective") {
        None => Objective::MinMemory,
        Some(o) => match Objective::parse(o) {
            Some(o) => o,
            None => {
                prr!("omni: unknown objective `{o}`\n");
                return Ok(2);
            }
        },
    };
    let optimistic = args.iter().any(|a| a == "--optimistic");

    let store = Borrowed(c);
    let ctx = Ctx::new(&store);
    let manifest = c.root()?;
    let table = tensor_table(c)?;
    let plan = resolve(
        &ctx,
        &manifest,
        (otype::MANIFEST, c.header.root_digest),
        &table,
        &caps,
        objective,
        optimistic,
    )?;

    pr!(
        "{}  →  {} {}",
        std::env::args().nth(2).unwrap_or_default(),
        caps.runtime_name,
        caps.runtime_version
    );
    pr!("  objective     {}", objective.name());
    pr!(
        "  caps digest   {}",
        short(c.header.hash, &plan.caps_digest)
    );
    pr!(
        "  plan key      {}",
        short(c.header.hash, &plan.key(c.header.hash))
    );
    pr!();
    if plan.is_feasible() {
        pr!("FEASIBLE");
    } else {
        pr!("INFEASIBLE");
    }
    for u in &plan.unmet {
        pr!("  {u}");
    }
    if plan.is_feasible() {
        pr!(
            "  ✓ {} tensor(s): {} resident, {} read",
            plan.tensors.len(),
            human(plan.resident_bytes),
            human(plan.read_bytes)
        );
        // Group the choices, since a plan for a real model has thousands.
        let mut by: std::collections::BTreeMap<(String, &'static str), (usize, u64)> =
            Default::default();
        for t in &plan.tensors {
            let e = by
                .entry((t.dtype.label(), t.materialization.name()))
                .or_insert((0, 0));
            e.0 += 1;
            e.1 += t.resident_bytes;
        }
        for ((dtype, how), (n, bytes)) in by {
            pr!(
                "    {:<10} {:<20} {:>4} tensors  {:>10}",
                dtype,
                how,
                n,
                human(bytes)
            );
        }
    }
    for w in &plan.warnings {
        pr!("  ⚠ {w}");
    }
    if let Some(path) = flag(args, "-o").or_else(|| flag(args, "--out")) {
        std::fs::write(path, plan.to_value().encode())?;
        pr!("\nwrote {path}");
    }
    // §10.5.2: an infeasible plan is a definite answer, not an error.
    if plan.is_feasible() {
        Ok(0)
    } else {
        Ok(1)
    }
}

fn unhex(s: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if !s.len().is_multiple_of(2) {
        return Err("hex must have an even number of digits".into());
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string().into()))
        .collect()
}

/// `omni keygen` — an Ed25519 key pair (§12.5.1).
fn cmd_keygen(args: &[String]) -> R {
    let seed: [u8; 32] = match flag(args, "--seed") {
        // A declared seed makes the key reproducible, which is what tests and
        // CI need. It is not a way to make a release key: the seed *is* the
        // private key.
        Some(h) => unhex(h)?
            .try_into()
            .map_err(|_| "--seed takes 32 bytes of hex")?,
        None => {
            let bytes = std::fs::read("/dev/urandom")
                .map_err(|_| "no /dev/urandom on this platform; pass --seed <64 hex digits>")?;
            bytes
                .get(..32)
                .ok_or("short read from /dev/urandom")?
                .try_into()
                .map_err(|_| "short read from /dev/urandom")?
        }
    };
    let sk = omni_core::ed25519::SecretKey::from_seed(&seed);
    let out = flag(args, "--out");
    let line = format!(
        "# OMNI Ed25519 key. The seed is the private key: treat this file as one.\nseed {}\npublic {}\n",
        hex(&sk.seed()),
        hex(&sk.public_key())
    );
    match out {
        Some(path) => {
            std::fs::write(path, &line)?;
            pr!("wrote {path}");
            pr!("  public  {}", hex(&sk.public_key()));
        }
        None => prr!("{line}"),
    }
    Ok(0)
}

/// The objects of a container, ready to repack.
fn container_objects(c: &Container) -> Result<Vec<omni_core::Object>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(c.index.len());
    for e in &c.index {
        out.push(omni_core::Object {
            otype: e.otype,
            payload: c.read(&e.digest)?,
            oflags: e.oflags,
            stored: None,
        });
    }
    Ok(out)
}

/// `omni sign` — sign a manifest, or verify signatures against a policy.
fn cmd_sign(args: &[String]) -> R {
    use omni_core::sign::{
        canonical_digest, sign_cose, signing_root, verify_signatures, Policy, Purpose, Requirement,
        Signature, Summary, Tbs, TrustedKey,
    };

    if args.iter().any(|a| a == "--verify") {
        let path = flag(args, "--verify").ok_or("--verify <file>")?;
        let c = Container::open(std::fs::read(path)?)?;
        let manifest = c.root()?;
        let algo = c.header.hash;
        let cacheable: std::collections::BTreeSet<omni_core::Digest> = c
            .index
            .iter()
            .filter(|e| e.oflags & 0b10 != 0)
            .map(|e| e.digest)
            .collect();
        let root = signing_root(&manifest, algo);
        let canon = canonical_digest(&manifest, algo, &|d| cacheable.contains(d));
        let mut keys = Vec::new();
        for k in flag(args, "--key").unwrap_or_default().split(',') {
            if k.is_empty() {
                continue;
            }
            let b: [u8; 32] = unhex(k)?
                .try_into()
                .map_err(|_| "--key takes 32 bytes of hex per key")?;
            keys.push(TrustedKey::new(b));
        }
        let requirement = match flag(args, "--require") {
            None | Some("any") => Requirement::AnyOf,
            Some("all") => Requirement::AllOf,
            Some(spec) if spec.starts_with("k:") => {
                Requirement::KOfN(spec[2..].parse().map_err(|_| "--require k:N")?)
            }
            Some(other) => {
                prr!("omni: unknown --require `{other}` (any, all, k:N)\n");
                return Ok(2);
            }
        };
        let mut policy = Policy::keys(keys)
            .requirement(requirement)
            .purposes(vec![Purpose::Release, Purpose::Internal]);
        if let Some(now) = flag(args, "--at") {
            policy = policy.at(now);
        }
        let mut sigs = Vec::new();
        for d in omni_core::sign::attestation_refs(&manifest) {
            match c.get_value(&d) {
                Ok(v) => match Signature::from_value(&v) {
                    Ok(s) => sigs.push(s),
                    Err(e) => pr!("  ⚠ {}: {e}", short(algo, &d)),
                },
                Err(e) => pr!("  ⚠ {}: {e}", short(algo, &d)),
            }
        }
        pr!("{path}");
        pr!("  root            {}", short(algo, &root));
        pr!("  canonical       {}", short(algo, &canon));
        pr!("  attestations    {}", sigs.len());
        let v = verify_signatures(&sigs, &root, &canon, &policy);
        for o in &v.outcomes {
            pr!(
                "  {} {}  {}",
                if o.ok {
                    "✓"
                } else if o.indeterminate {
                    "?"
                } else {
                    "✗"
                },
                o.kid
                    .as_ref()
                    .map(|k| hex(&k[..8.min(k.len())]))
                    .unwrap_or_else(|| "(no kid)".into()),
                o.message
            );
            if let Some(t) = &o.tbs {
                pr!(
                    "      subject {} {}  purpose {}  counter {}",
                    t.subject_name,
                    t.subject_version.clone().unwrap_or_default(),
                    t.purpose.name(),
                    t.counter
                );
            }
        }
        if v.satisfied {
            pr!("\nV7 authenticity ✓ policy satisfied");
            return Ok(0);
        }
        if v.invalid_count() > 0 {
            pr!(
                "\ninvalid: {} signature(s) do not verify",
                v.invalid_count()
            );
            return Ok(1);
        }
        pr!("\nindeterminate: the policy is not satisfied and nothing is provably wrong");
        return Ok(3);
    }

    // --- signing ---
    let path = args.get(1).ok_or("usage: omni sign <file> --key <hex>")?;
    if path.starts_with("--") {
        return Err("usage: omni sign <file> --key <hex>".into());
    }
    let key_hex = flag(args, "--key").ok_or("--key <64 hex digits of seed>")?;
    let seed: [u8; 32] = unhex(key_hex)?
        .try_into()
        .map_err(|_| "--key takes 32 bytes of hex (the seed)")?;
    let sk = omni_core::ed25519::SecretKey::from_seed(&seed);
    let out = flag(args, "-o")
        .or_else(|| flag(args, "--out"))
        .unwrap_or(path);
    let purpose = match flag(args, "--purpose") {
        None => Purpose::Release,
        Some(p) => match Purpose::parse(p) {
            Some(p) => p,
            None => {
                prr!("omni: unknown purpose `{p}` (release, internal, test, revocation)\n");
                return Ok(2);
            }
        },
    };
    let counter: u64 = flag(args, "--counter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let c = Container::open(std::fs::read(path)?)?;
    let algo = c.header.hash;
    let manifest = c.root()?;
    let cacheable: std::collections::BTreeSet<omni_core::Digest> = c
        .index
        .iter()
        .filter(|e| e.oflags & 0b10 != 0)
        .map(|e| e.digest)
        .collect();
    let store = Borrowed(&c);
    let ctx = Ctx::new(&store);
    let (tensors, params) = match tensor_table(&c) {
        Ok(t) => {
            let n = t.len() as u64;
            let p = t
                .tensors
                .values()
                .filter_map(|r| TensorDesc::load(&ctx, r).ok())
                .filter(|d| matches!(d.value, Expr::Literal { .. }) && d.is_weight())
                .filter_map(|d| d.numel())
                .sum();
            (n, p)
        }
        Err(_) => (0, 0),
    };
    let name = manifest
        .get("meta")
        .and_then(as_ref_digest)
        .and_then(|d| c.get_value(&d).ok())
        .and_then(|m| {
            m.get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "(unnamed)".into());

    let tbs = Tbs {
        root: signing_root(&manifest, algo),
        alg: "EdDSA".into(),
        purpose,
        subject_name: name.clone(),
        subject_version: flag(args, "--version").map(|s| s.to_string()),
        not_before: flag(args, "--not-before").map(|s| s.to_string()),
        not_after: flag(args, "--not-after").map(|s| s.to_string()),
        summary: Summary {
            tensors,
            params,
            canonical_digest: canonical_digest(&manifest, algo, &|d| cacheable.contains(d)),
            // Nothing in this implementation marks an object executable, and
            // saying so is part of the signed payload (§12.5.2).
            executables: 0,
        },
        counter,
    };
    let sig = Signature::new(&sign_cose(&sk, &tbs));
    let sig_obj = omni_core::Object::structure(otype::SIGNATURE, &sig.to_value());
    let sig_digest = sig_obj.digest(algo);
    let signed_manifest = omni_core::sign::attach(&manifest, &sig_digest);
    let manifest_obj = omni_core::Object::structure(otype::MANIFEST, &signed_manifest);
    let new_root = manifest_obj.digest(algo);

    // Repack: everything except the old manifest, plus the new manifest and the
    // signature. The old manifest is dropped because nothing references it any
    // more; the signature is valid over either, which is the point of hashing
    // the manifest with `attestations` removed.
    let mut objects: Vec<omni_core::Object> = container_objects(&c)?
        .into_iter()
        .filter(|o| o.digest(algo) != c.header.root_digest)
        .collect();
    objects.push(manifest_obj);
    objects.push(sig_obj);
    let bytes = pack(
        &objects,
        &new_root,
        &PackOptions {
            hash: algo,
            log2_align: c.header.log2_align,
            ..Default::default()
        },
    )?;
    std::fs::write(out, &bytes)?;
    pr!("wrote {out}");
    pr!("  subject         {name}");
    pr!("  purpose         {}  counter {counter}", purpose.name());
    pr!("  signing root    {}", short(algo, &tbs.root));
    pr!(
        "  canonical       {}",
        short(algo, &tbs.summary.canonical_digest)
    );
    pr!("  public key      {}", hex(&sk.public_key()));
    pr!("  new file root   {}", short(algo, &new_root));
    pr!(
        "  verify with     omni sign --verify {out} --key {}",
        hex(&sk.public_key())
    );
    Ok(0)
}

/// `omni delta` — express one model as a delta over another (§08.6).
///
/// The result is deliberately *incomplete* on its own: its tensors are
/// expressions over the base's chunk objects, which live in the base container.
/// That is what makes a delta small, and `omni verify` reports it as incomplete
/// rather than invalid.
fn cmd_delta(args: &[String]) -> R {
    use omni_core::delta::{analyze, literal_of, Kind, Options, Parent, Report};
    use omni_core::tensor::Materialize;

    let base_path = args
        .get(1)
        .ok_or("usage: omni delta <base> <tuned> -o <out>")?;
    let tuned_path = args
        .get(2)
        .ok_or("usage: omni delta <base> <tuned> -o <out>")?;
    let out = flag(args, "-o")
        .or_else(|| flag(args, "--out"))
        .ok_or("-o <delta.omni> required")?;
    let max_err: f64 = flag(args, "--max-err")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-2);

    let base = Container::open(std::fs::read(base_path)?)?;
    let tuned = Container::open(std::fs::read(tuned_path)?)?;
    if base.header.hash != tuned.header.hash {
        prr!("omni: the two containers use different digest algorithms\n");
        return Ok(2);
    }
    let algo = base.header.hash;
    let bs = Borrowed(&base);
    let ts = Borrowed(&tuned);
    let bctx = Ctx::new(&bs);
    let tctx = Ctx::new(&ts);
    let btable = tensor_table(&base)?;
    let ttable = tensor_table(&tuned)?;

    let base_opts = Options {
        max_err,
        ..Default::default()
    };
    let mut report = Report {
        base_bytes: base
            .index
            .iter()
            .filter(|e| e.otype == otype::BLOB)
            .map(|e| e.logical_len)
            .sum(),
        ..Default::default()
    };
    let mut b = ModelBuilder::new(format!("delta of {tuned_path}")).hash(algo);
    let mut carried = 0usize;

    for (name, tr) in &ttable.tensors {
        let td = TensorDesc::load(&tctx, tr)?;
        let Some(br) = btable.get(name) else {
            // A tensor the base does not have is carried whole.
            carried += 1;
            continue;
        };
        let bd = TensorDesc::load(&bctx, br)?;
        let (Ok(bt), Ok(tt)) = (bd.value.eval(&bctx), td.value.eval(&tctx)) else {
            carried += 1;
            continue;
        };
        // Store the delta's own tensors in the *tuned* tensor's dtype: LoRA
        // factors are bf16 in practice, and it keeps the delta's inferred dtype
        // equal to the tensor it replaces.
        let opts = Options {
            store_dtype: td.dtype.clone(),
            ..base_opts.clone()
        };
        let plan = analyze(&bt, &tt, &opts)?;
        report.add(&plan);
        // The delta's expression is written over the *base's* value, so the
        // parent's bytes are referenced rather than copied.
        let mut stored = std::collections::BTreeMap::new();
        for (role, tensor) in &plan.tensors {
            let bytes = tensor.to_bytes(
                &tensor.dtype,
                &omni_core::layout::Layout::default(),
                omni_core::Round::Rne,
            )?;
            let cl = b.chunk_list(&bytes);
            stored.insert(
                role.clone(),
                Expr::Literal {
                    chunks: cl,
                    dtype: tensor.dtype.clone(),
                    shape: omni_core::expr::dims(&tensor.shape),
                    layout: omni_core::layout::Layout::default(),
                },
            );
        }
        let _ = literal_of; // the helper above builds the same node from a blob
        let value = plan.build(&bd.value, &stored)?;
        let inferred = value.infer()?;
        b = b.derived(
            name.clone(),
            TensorDesc {
                shape: inferred.shape,
                dtype: inferred.dtype,
                layout: td.layout.clone(),
                value,
                semantic: td.semantic.clone(),
                role: td.role.clone(),
                axes: td.axes.clone(),
                device_hint: None,
                materialize: Materialize::Lazy,
                stats: None,
                digest_materialized: None,
            },
        );
    }

    // §08.7: the parent is pinned by digest.
    b.extra.push((
        "parents".into(),
        Value::Array(vec![Parent {
            reference: (otype::MANIFEST, base.header.root_digest),
            role: "base".into(),
            name: Some(base_path.to_string()),
            locators: vec![],
            required: true,
        }
        .to_value()]),
    ));
    let (objs, root) = b.build();
    // A delta's objects are small — a few kilobytes of low-rank factors — and
    // page alignment would cost more in padding than the payload itself. The
    // 64-byte minimum of §02.3 is the right choice here, and `--align` overrides
    // it for a delta big enough to want mmap alignment back.
    let log2_align: u8 = flag(args, "--align")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let bytes = pack(
        &objs,
        &root,
        &PackOptions {
            hash: algo,
            log2_align,
            ..Default::default()
        },
    )?;
    std::fs::write(out, &bytes)?;
    pr!("{report}");
    if carried > 0 {
        pr!("  ({carried} tensor(s) had no counterpart in the base and were skipped)");
    }
    pr!();
    pr!("wrote {out}  {}", human(bytes.len() as u64));
    pr!("  parent        {}", short(algo, &base.header.root_digest));
    pr!(
        "  identical     {} tensor(s) reference the parent directly, at zero bytes",
        report
            .rows
            .get(Kind::Identical.name())
            .map(|r| r.tensors)
            .unwrap_or(0)
    );
    pr!("  incomplete    reading it needs the base: `omni cat <delta> --with <base>`");
    Ok(0)
}

/// `omni adapter check` — §08.3 attachment validation, before any weights load.
fn cmd_adapter(args: &[String]) -> R {
    match args.get(1).map(|s| s.as_str()) {
        Some("check") => cmd_adapter_check(args),
        Some("make") => cmd_adapter_make(args),
        _ => {
            prr!(
                "usage: omni adapter check <base.omni> <adapter.omni>\n       omni adapter make \
                 <base.omni> -o <lora.omni>\n"
            );
            Ok(2)
        }
    }
}

/// `omni adapter make` — a LoRA over a base, as §08.1 describes it: a manifest
/// with `kind: "adapter"` whose only asset is an `Adapter`, referencing the base
/// by digest.
fn cmd_adapter_make(args: &[String]) -> R {
    use omni_core::adapter::lora_adapter_value;
    use omni_core::tensor::Materialize;

    let base_path = args
        .get(2)
        .ok_or("usage: omni adapter make <base.omni> -o <out>")?;
    let out = flag(args, "-o")
        .or_else(|| flag(args, "--out"))
        .ok_or("-o <lora.omni> required")?;
    let rank: u64 = flag(args, "--rank")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let alpha: f64 = flag(args, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);
    let targets = flag(args, "--targets").unwrap_or("model.layers.*.attn.q_proj.weight");

    let base = Container::open(std::fs::read(base_path)?)?;
    let algo = base.header.hash;
    let bs = Borrowed(&base);
    let bctx = Ctx::new(&bs);
    let table = tensor_table(&base)?;

    // One A/B pair per matched base tensor, sized from that tensor's own shape.
    let matched: Vec<String> = table.select(targets).into_iter().cloned().collect();
    if matched.is_empty() {
        prr!("omni: `{targets}` matches no tensor in {base_path}\n");
        return Ok(2);
    }
    let mut b = ModelBuilder::new(format!("lora over {base_path}")).hash(algo);
    let mut factors: Vec<(String, TensorDesc)> = Vec::new();
    for name in &matched {
        let d = TensorDesc::load(&bctx, table.get(name).unwrap())?;
        let sizes = d.sizes().ok_or("a target tensor has a symbolic shape")?;
        if sizes.len() != 2 {
            continue;
        }
        let (out_f, in_f) = (sizes[0], sizes[1]);
        // §08.3's `{1}` capture: the adapter's tensors are named by what the
        // glob matched, so the binding survives renaming of everything else.
        let caps = omni_core::pattern::glob_captures(targets, name).unwrap_or_default();
        let key = caps.join(".");
        let a_bytes = floats(
            &DType::BF16,
            (rank * in_f) as usize,
            &format!("A{key}"),
            0.02,
        );
        let b_bytes = vec![0u8; DType::BF16.packed_bytes(out_f * rank) as usize];
        let a_expr = b.literal(
            &a_bytes,
            DType::BF16,
            &[rank, in_f],
            omni_core::layout::Layout::default(),
        );
        let b_expr = b.literal(
            &b_bytes,
            DType::BF16,
            &[out_f, rank],
            omni_core::layout::Layout::default(),
        );
        let desc = |value: Expr, axes: Vec<&str>, shape: &[u64]| TensorDesc {
            shape: omni_core::expr::dims(shape),
            dtype: DType::BF16,
            layout: omni_core::layout::Layout::default(),
            value,
            semantic: Some("weight".into()),
            role: Some("lora".into()),
            axes: Some(axes.iter().map(|a| a.to_string()).collect()),
            device_hint: None,
            materialize: Materialize::Lazy,
            stats: None,
            digest_materialized: None,
        };
        factors.push((
            format!("lora.{key}.A"),
            desc(a_expr, vec!["rank", "in"], &[rank, in_f]),
        ));
        factors.push((
            format!("lora.{key}.B"),
            desc(b_expr, vec!["out", "rank"], &[out_f, rank]),
        ));
    }
    for (name, d) in factors {
        b = b.derived(name, d);
    }
    // The adapter's tensors live in their own table, so build the model graph
    // and then point the Adapter object at that table.
    let (mut objs, _) = b.build();
    let table_ref = objs
        .iter()
        .find(|o| o.otype == otype::TENSOR_TABLE)
        .map(|o| (otype::TENSOR_TABLE, o.digest(algo)))
        .ok_or("the builder produced no tensor table")?;
    // The rank contracts over the base's *input* axis, whatever the base calls
    // it. Reading the name off the base is what lets `require` catch a mismatch
    // rather than the multiplication being quietly wrong.
    let rank_axis = TensorDesc::load(&bctx, table.get(&matched[0]).unwrap())?
        .axes
        .and_then(|a| a.last().cloned())
        .unwrap_or_else(|| "in".to_string());
    let adapter = lora_adapter_value(
        &(otype::MANIFEST, base.header.root_digest),
        &table_ref,
        rank,
        alpha,
        &[targets],
        "lora.{1}.A",
        "lora.{1}.B",
        &rank_axis,
    )?;
    let adapter_obj = omni_core::Object::structure(otype::ADAPTER, &adapter);
    let adapter_ref = adapter_obj.digest(algo);
    objs.push(adapter_obj);
    // §08.1: an adapter is a first-class publishable artifact.
    let manifest = omni_core::Object::structure(
        otype::MANIFEST,
        &Value::map(vec![
            ("t", Value::text("omni.core/manifest")),
            ("v", Value::U(1)),
            ("kind", Value::text("adapter")),
            ("created", Value::U(0)),
            (
                "assets",
                Value::map(vec![(
                    "adapter",
                    Value::Array(vec![
                        Value::U(otype::ADAPTER as u64),
                        Value::Bytes(adapter_ref.to_vec()),
                    ]),
                )]),
            ),
            ("entry", Value::text("adapter")),
            (
                "parents",
                Value::Array(vec![omni_core::delta::Parent {
                    reference: (otype::MANIFEST, base.header.root_digest),
                    role: "base".into(),
                    name: Some(base_path.to_string()),
                    locators: vec![],
                    required: true,
                }
                .to_value()]),
            ),
        ]),
    );
    let root = manifest.digest(algo);
    objs.push(manifest);
    let bytes = pack(
        &objs,
        &root,
        &PackOptions {
            hash: algo,
            log2_align: 6,
            ..Default::default()
        },
    )?;
    std::fs::write(out, &bytes)?;
    pr!("wrote {out}  {}", human(bytes.len() as u64));
    pr!("  method        lora  rank {rank}  alpha {alpha}");
    pr!("  base          {}", short(algo, &base.header.root_digest));
    pr!("  targets       {targets}  ({} matched)", matched.len());
    pr!("  check with    omni adapter check {base_path} {out}");
    Ok(0)
}

fn cmd_adapter_check(args: &[String]) -> R {
    let base_path = args
        .get(2)
        .ok_or("usage: omni adapter check <base> <adapter>")?;
    let adapter_path = args
        .get(3)
        .ok_or("usage: omni adapter check <base> <adapter>")?;
    let base = Container::open(std::fs::read(base_path)?)?;
    let adapter = Container::open(std::fs::read(adapter_path)?)?;
    let bs = ContainerStore::new(base);
    let as_ = ContainerStore::new(adapter);
    let layered = omni_core::store::Layered::new(vec![&as_, &bs])?;
    let ctx = Ctx::new(&layered);

    // The adapter object is the manifest's only asset.
    let manifest = as_.container().root()?;
    let ad = manifest
        .get("assets")
        .and_then(|a| a.get("adapter"))
        .and_then(as_ref_digest)
        .ok_or("this container has no `adapter` asset")?;
    let a = omni_core::adapter::Adapter::load(&ctx, &(otype::ADAPTER, ad))?;
    let table = tensor_table(bs.container())?;
    let r = a.check(&ctx, &table)?;

    pr!("{adapter_path} against {base_path}");
    pr!("  method        {}", a.method.name());
    pr!("  base          {}", short(as_.hash(), &a.base.1));
    if let (Some(rank), Some(alpha)) = (a.rank, a.alpha) {
        pr!("  rank/alpha    {rank} / {alpha}");
    }
    pr!("  attached      {} tensor(s)", r.bindings.len());
    for b in r.bindings.iter().take(8) {
        pr!("    {:<52} <- {}", b.tensor, b.used.join(", "));
    }
    if r.bindings.len() > 8 {
        pr!("    … {} more", r.bindings.len() - 8);
    }
    for f in &r.findings {
        pr!("  {f}");
    }
    if !r.graph_patches.is_empty() {
        pr!("  graph patches {} (applied by §07)", r.graph_patches.len());
    }
    if r.is_ok() {
        pr!("\nattachable");
        return Ok(0);
    }
    pr!("\ninvalid: the adapter cannot attach to this base");
    Ok(1)
}

/// Adds a seeded rank-1 update to a bf16 matrix in place — a fine-tune's worth
/// of change, and the case §08.6's low-rank representation exists for.
fn rank1_update(bytes: &mut [u8], rows: usize, cols: usize, seed: u64, scale: f64) {
    let u: Vec<f64> = (0..rows)
        .map(|i| omni_core::expr::uniform01(seed, i as u64) - 0.5)
        .collect();
    let v: Vec<f64> = (0..cols)
        .map(|j| omni_core::expr::uniform01(seed ^ 0xffff, j as u64) - 0.5)
        .collect();
    for (i, ui) in u.iter().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            let k = (i * cols + j) as u64;
            let Some(old) = DType::BF16.decode(bytes, k) else {
                continue;
            };
            DType::BF16.encode(bytes, k, old + scale * ui * vj, omni_core::Round::Rne);
        }
    }
}

/// Deterministic filler in a *dtype*, with values bounded by `scale`.
///
/// `pattern` fills bytes, which is right for opaque payloads and wrong for
/// floats: random bf16 bit patterns span 1e±38 and make an example's output
/// unreadable. These are real numbers, encoded through the dtype.
fn floats(dtype: &DType, n: usize, seed: &str, scale: f64) -> Vec<u8> {
    let mut out = vec![0u8; dtype.packed_bytes(n as u64) as usize];
    let mut x = 0x811c_9dc5u32;
    for byte in seed.bytes() {
        x = (x ^ byte as u32).wrapping_mul(0x0100_0193);
    }
    for i in 0..n {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (x >> 8) as f64 / (1u32 << 24) as f64;
        dtype.encode(
            &mut out,
            i as u64,
            (u - 0.5) * 2.0 * scale,
            omni_core::Round::Rne,
        );
    }
    out
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
