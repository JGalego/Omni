# OMNI/1.0 — §6 Metadata, Tokenizers, and Templates

Metadata must be **discoverable without loading tensors**, **structured enough to
query**, and **open enough to never block a publisher**. OMNI achieves the first
by putting metadata in small objects near the front of the container (§13.2), the
second with schemas (§03.4), and the third with namespaced extensions (§11).

## 6.1 Metadata object

```cbor-diag
{ "t":"omni.meta/model", "v":1,

  "name": "acme/llm-8b-instruct",
  "display_name": "Acme LLM 8B Instruct",
  "version": "2026.08.1",
  "description": "…",
  "homepage": "https://…",
  "authors": [{"name":"…","org":"Acme","orcid":"…"}],
  "created": "2026-08-04T00:00:00Z",

  "arch": {                                    ; §6.2
    "family": "transformer.decoder",
    "dialects": [ {"ns":"omni.nn","v":1}, {"ns":"org.acme/moe","v":2} ],
    "params": { … }
  },

  "params_total": 8030261248,
  "params_trainable": 8030261248,
  "params_effective": 2100000000,              ; active params/token for MoE
  "modality": {"in":["text","image"], "out":["text"]},

  "license": { … },                            ; §6.6
  "provenance": [19, h'…'],                    ; -> Provenance
  "evaluations": [ [32, h'…'] ],               ; -> Evaluation
  "citations": [ {"doi":"10.…","bibtex":"…"} ],

  "generation": { … },                         ; §6.5
  "context": { … },                            ; §6.4
  "hardware_hints": { … },                     ; §6.11
  "tags": ["instruct","multilingual","tool-use"],
  "ext": { "org.acme/internal": [31, h'…'] }
}
```

Everything here is optional except `t`/`v`. **Absent means unknown.** OMNI never
defaults a metadata field to a plausible value — an importer that cannot
determine the parameter count leaves it out rather than computing something that
might be wrong. `omni inspect` distinguishes "unknown" from "zero".

## 6.2 Architecture descriptor

Architecture is a *hint plus a dialect list*, never an enum:

```cbor-diag
"arch": {
  "family": "transformer.decoder",     ; free text, registry-suggested
  "dialects": [{"ns":"omni.nn","v":1}],
  "params": {
    "hidden_size": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 8,
    "head_dim": 128, "ffn_hidden": 14336, "activation": "silu",
    "norm": {"kind":"rmsnorm","eps":1e-5,"pre":true},
    "rope": { … },                     ; §6.3
    "moe": {"experts":64,"top_k":8,"shared":2,"router":"softmax-topk",
            "aux_loss":0.001,"capacity_factor":1.25},
    "ssm": {"state_size":16,"conv_kernel":4,"dt_rank":"auto"},
    "attention": {"kind":"gqa","window":[4096,null],"sink":4,"logit_softcap":50.0}
  }
}
```

`arch.params` is a **free-form, dialect-interpreted map**. The core specification
assigns meaning to *none* of it. A runtime that recognizes
`family = "transformer.decoder"` and the `omni.nn` dialect knows how to read it;
one that does not falls back to the graph (§07) or refuses. This is deliberate:
the alternative — the GGUF approach of a hand-maintained key list per
architecture — makes the format's maintainers a bottleneck on every new model.

**The escape hatch is the graph.** A model carrying an OMNI-IR graph needs no
`arch.params` at all; the graph *is* the architecture. `arch.params` exists so
that weights-only models (the safetensors-equivalent, C0 case) remain viable.

## 6.3 Positional encoding

Enough structure to stop the recurring "which RoPE variant is this?" failure:

```cbor-diag
"rope": {
  "kind": "rope",                 ; rope|alibi|nope|learned|sinusoidal|relative|plugin
  "theta": 500000.0,
  "dims": 128,
  "partial": 1.0,                 ; fraction of head_dim rotated
  "interleaved": false,           ; GPT-NeoX vs GPT-J pairing — the classic bug
  "scaling": {
    "kind": "yarn",               ; none|linear|ntk|dynamic-ntk|yarn|longrope|llama3
    "factor": 8.0,
    "original_context": 8192,
    "beta_fast": 32, "beta_slow": 1,
    "attn_factor": 1.0,
    "low_freq_factor": 1.0, "high_freq_factor": 4.0
  },
  "per_layer": null               ; or an array/tensor ref for per-layer schedules
}
```

`interleaved` alone has caused more silent output corruption in format
conversions than any other single field. Making it required-when-`kind=rope`
(enforced by the schema) removes the class of bug.

## 6.4 Context and sequence

```cbor-diag
"context": {
  "trained": 8192,
  "max": 131072,                  ; with the declared scaling applied
  "recommended": 32768,
  "sliding_window": 4096,
  "attention_sinks": 4,
  "chunked_prefill_ok": true,
  "kv_layout_hint": {"kind":"paged","page":16}
}
```

## 6.5 Generation defaults

```cbor-diag
"generation": {
  "bos": 128000, "eos": [128001, 128009], "pad": null, "unk": null,
  "temperature": 0.6, "top_p": 0.9, "top_k": 0,
  "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
  "max_new_tokens": 4096,
  "stop": ["<|eot_id|>"],
  "sampler_chain": ["min_p","temperature"],       ; explicit order
  "logit_bias": null,
  "seed_behavior": "unspecified"
}
```

`sampler_chain` is included because sampler *order* changes outputs and is
currently transmitted by folklore.

## 6.6 Licensing and use restrictions

```cbor-diag
"license": {
  "spdx": "Apache-2.0",                     ; or "LicenseRef-Acme-Community-1.0"
  "text": [0, h'…'],                        ; -> Blob with the full text
  "url": "https://…",
  "components": [                           ; different licenses for different assets
    {"asset":"vision","spdx":"CC-BY-NC-4.0"}
  ],
  "use_restrictions": ["no-military","no-surveillance"],   ; RAIL-style, advisory
  "attribution_required": true,
  "redistribution": {"allowed":true,"share_alike":false},
  "derived_from": [ {"manifest":[1,h'…'], "license":"Llama-3.1"} ],
  "training_data_disclosure": "partial"      ; none|partial|full
}
```

OMNI takes no position on license enforceability. It provides a **signed,
tamper-evident place** for the claim, which is strictly better than a README that
gets lost on the first conversion.

## 6.7 Tokenizer IR

Tokenizers are the second-most-common source of conversion corruption after
positional encoding. OMNI stores them **structurally**, not as an opaque
`tokenizer.json`.

```cbor-diag
{ "t":"omni.tok/tokenizer", "v":1,
  "kind": "bpe",                  ; bpe|unigram|wordpiece|wordlevel|char|byte|
                                  ; sentencepiece-bpe|tiktoken|plugin
  "normalizers": [
    {"k":"nfc"}, {"k":"replace","pattern":{"re":"\\s+"},"to":" "}
  ],
  "pretokenizers": [
    {"k":"regex-split","pattern":{"re":"(?i:'s|'t)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|…",
      "flavor":"pcre2","behavior":"isolated"}},
    {"k":"byte-level","add_prefix_space":false,"use_regex":false}
  ],
  "vocab": {                      ; large: stored as tensors, not inline
    "tokens":  <expr>,            ; string tensor  [V]
    "scores":  <expr>,            ; f32 [V]  (unigram)
    "types":   <expr>             ; u8  [V]  (normal|byte|control|user|unused)
  },
  "merges": <expr>,               ; u32 [M,2] — id pairs, not strings
  "byte_fallback": true,
  "unk": null, "fuse_unk": false,
  "added_tokens": [
    {"id":128000,"content":"<|begin_of_text|>","special":true,
     "lstrip":false,"rstrip":false,"normalized":false,"single_word":false}
  ],
  "postprocessor": {"k":"template","single":["<|bos|>","$A"],"pair":[…]},
  "decoder": [{"k":"byte-level"},{"k":"replace","pattern":"▁","to":" "}],
  "max_token_len": 128,
  "conformance": {                ; §6.7.1
     "vectors": [0, h'…'], "digest": h'…'
  },
  "ext": {}
}
```

Design points:

- **Vocabulary is a tensor, not JSON.** A 256 k-entry vocabulary is ~3 MB; as
  inline CBOR it would bloat every metadata read. As a tensor it is chunked,
  deduplicated across models with shared tokenizers (which is most of them), and
  memory-mappable.
- **Merges are id pairs.** Storing merges as strings (as
  `tokenizer.json` does) requires re-resolution and is ambiguous under
  normalization. Ids are unambiguous and 4× smaller.
- **Regex flavor is explicit.** `pcre2` vs. `re2` vs. `oniguruma` differ on
  `\p{L}` boundaries, possessive quantifiers, and lookahead. The de facto
  standard tokenizers rely on PCRE2/Oniguruma semantics; a format that does not
  say which will produce different tokenizations in different runtimes. This is
  currently a real, unacknowledged interoperability failure.

### 6.7.1 Tokenizer conformance vectors

A tokenizer object SHOULD carry a `Blob` of test vectors:

```
text \t token_ids
"Hello, world!"            \t 9906,11,1917,0
"  leading spaces"         \t 262,6522,12908
"emoji 👨‍👩‍👧‍👦 zwj"        \t …
"́combining"          \t …
```

`omni verify --tokenizer` runs them. This turns "the tokenizer changed during
conversion" from a silent quality regression into a build failure. Every
importer MUST generate vectors by running the source tokenizer if it is
available, and MUST record in the fidelity report if it could not.

### 6.7.2 Plugin tokenizers

`kind: "plugin"` with a `PluginModule` (WASM) implementing
`encode(text) -> [u32]` / `decode([u32]) -> text` is the escape hatch for
genuinely novel tokenizers (audio codecs, byte-patch schemes, learned
tokenizers). Deterministic, sandboxed, and portable — a 2050 runtime can still
tokenize a 2026 model exactly.

## 6.8 Evaluation results

```cbor-diag
{ "t":"omni.meta/evaluation", "v":1,
  "results": [
    {"benchmark":"mmlu","version":"1.0","split":"test","shots":5,
     "metric":"acc","value":[1007,[-4,6821]],      ; exact decimal 0.6821
     "n":14042, "stderr":0.004,
     "harness":{"name":"lm-eval","version":"0.4.3","commit":"…"},
     "config_digest":h'…', "date":"2026-07-30",
     "self_reported":true}
  ],
  "attested_by": [18, h'…']        ; optional signature by the evaluator
}
```

Scores use exact decimals (tag 1007), not floats, because a benchmark score is a
reported quantity, not a computed one, and `0.6821` should round-trip as
`0.6821`. `self_reported` is mandatory and defaults to `true`; third-party
evaluations carry their own signature, making "who claims this score?" a
verifiable question.

## 6.9 Chat templates

Chat templates are currently Jinja2 strings executed by the runtime — i.e.
**arbitrary code execution from a downloaded file**, which is unacceptable in a
format whose whole premise is safe loading.

OMNI defines **OMNI-CT**, a total (non-Turing-complete) template language:

```cbor-diag
{ "t":"omni.tok/chat-template", "v":1,
  "lang": "omni-ct/1",
  "source": "…",                       ; OMNI-CT text
  "compiled": <expr>,                  ; optional pre-parsed AST as a Blob
  "jinja_compat": "…",                 ; optional Jinja2 rendering of the same
                                       ; template, for legacy runtimes
  "capabilities": ["tools","system","thinking","multimodal"],
  "vectors": [0, h'…']                 ; input JSON → expected string
}
```

OMNI-CT semantics:

- Values: strings, integers, booleans, lists, maps. No arbitrary objects.
- Constructs: `{{ expr }}`, `{% if %}`, `{% for x in list %}` (bounded by the
  input's size), `{% set %}` (local, no recursion), whitespace control.
- Functions: a fixed, pure standard library (`upper`, `lower`, `trim`, `join`,
  `default`, `tojson`, `strftime` on explicit inputs). No imports, no eval, no
  attribute access into host objects, no method calls on arbitrary types.
- Guaranteed termination: loops iterate over finite input structures only; no
  `while`, no recursion. Rendering cost is O(input size × template size).

A Jinja2 subset maps mechanically onto OMNI-CT, so existing templates convert
automatically in the overwhelming majority of cases; the `jinja_compat` field
carries a Jinja2 rendering for runtimes that still want it, clearly marked as the
*derived* form.

`vectors` (message list → expected prompt string) make template regressions
detectable by `omni verify --template`.

## 6.10 Dataset descriptors

```cbor-diag
{ "t":"omni.meta/dataset", "v":1,
  "name":"acme/pretrain-mix-v3", "role":"pretraining",
  "size_tokens": 15000000000000,
  "composition":[{"name":"common-crawl-2025-13","tokens":8.1e12,"weight":0.54,
                  "license":"varied","digest":h'…'}],
  "availability":"reference-only",     ; embedded|reference-only|withheld
  "sample": [0, h'…'],                 ; optional small sample Blob
  "consent":{"opt_out_honored":["robots.txt","ai.txt","C2PA-do-not-train"],
             "as_of":"2026-01-15"},
  "pii":{"scan":"presidio-2.x","action":"redacted"},
  "contamination_check":{"benchmarks":["mmlu","gsm8k"],"method":"13-gram","found":0.0002}
}
```

`availability: "withheld"` is an honest, machine-readable answer. A format that
only allows "here is the dataset" guarantees the field goes unused.

## 6.11 Hardware hints

Advisory only; never affects semantics.

```cbor-diag
"hardware_hints": {
  "min_vram_bytes": {"bf16":17179869184,"int4":6442450944},
  "min_ram_bytes": 8589934592,
  "recommended": [{"device":"nvidia:sm_90","dtype":"f8e4m3","tp":1,"note":"…"}],
  "kv_bytes_per_token": {"bf16":131072,"int8":65536},
  "prefill_flops_per_token": 1.6e10,
  "decode_flops_per_token": 1.6e10,
  "supports": {"flash_attn":true,"paged_kv":true,"cuda_graphs":true,
               "tensor_parallel":[1,2,4,8],"pipeline_parallel":true}
}
```

`kv_bytes_per_token` and the FLOP figures let a scheduler size a deployment from
the manifest alone, without loading a byte of weights — a capability every
serving stack currently reimplements with per-architecture Python.

## 6.12 Discoverability guarantee

A conforming writer MUST place, in this order, at the front of the container:
superblock, `Manifest`, `Metadata`, `TensorTable` (or its shard directory),
`Tokenizer` header (excluding vocabulary tensors), and any `Evaluation`,
`Signature` and `Provenance` objects.

**Normative guarantee:** for a container of any size, a reader MUST be able to
answer everything `omni inspect` prints (§CLI) by reading the header, the
trailer, the superblock, the index, and the objects listed above — bounded by
`min(4 MiB, file_size)` of transfer beyond the index in the overwhelming
majority of cases, and never requiring a single tensor chunk.

**Prev:** [§05 Quantization](05-quantization.md) · **Next:** [§07 Execution Graph](07-graph.md)
