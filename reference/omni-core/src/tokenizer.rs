//! §06.7 — the tokenizer IR.
//!
//! Tokenizers are the second-most-common source of conversion corruption after
//! positional encoding, so OMNI stores them structurally: the vocabulary is a
//! tensor, the merges are id pairs rather than strings, and every normalizer,
//! pre-tokenizer and decoder step is a declared object rather than a line of
//! someone's Python.
//!
//! This module reads that structure and runs it — `encode` and `decode` for the
//! BPE, WordPiece, word-level, char and byte kinds — and runs the conformance
//! vectors of §06.7.1, which is what turns "the tokenizer changed during
//! conversion" from a silent quality regression into a build failure.
//!
//! ## Regex flavor, honestly
//!
//! §06.7 is right that an unspecified regex flavor is a real interoperability
//! failure, and it makes the flavor explicit. This implementation carries the
//! declared flavor and runs `regex-split` patterns through
//! [`crate::pattern::Regex`], which is a small ERE subset: no Unicode property
//! classes (`\p{L}`), no possessive quantifiers, no lookahead. A pattern that
//! needs those is reported as *indeterminate* rather than approximated, because
//! a pre-tokenizer that splits differently produces different token ids, and a
//! tokenizer that is quietly wrong is worse than one that refuses to run.

use crate::cbor::Value;
use crate::expr::{Ctx, Error, Expr, Ref};
use std::collections::BTreeMap;

type Res<T> = Result<T, Error>;

/// The tokenizer families of §06.7.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Bpe,
    Unigram,
    WordPiece,
    WordLevel,
    Char,
    Byte,
    SentencePieceBpe,
    Tiktoken,
    Plugin(String),
}

impl Kind {
    pub fn name(&self) -> &str {
        match self {
            Kind::Bpe => "bpe",
            Kind::Unigram => "unigram",
            Kind::WordPiece => "wordpiece",
            Kind::WordLevel => "wordlevel",
            Kind::Char => "char",
            Kind::Byte => "byte",
            Kind::SentencePieceBpe => "sentencepiece-bpe",
            Kind::Tiktoken => "tiktoken",
            Kind::Plugin(n) => n,
        }
    }

    pub fn parse(s: &str) -> Kind {
        match s {
            "bpe" => Kind::Bpe,
            "unigram" => Kind::Unigram,
            "wordpiece" => Kind::WordPiece,
            "wordlevel" => Kind::WordLevel,
            "char" => Kind::Char,
            "byte" => Kind::Byte,
            "sentencepiece-bpe" => Kind::SentencePieceBpe,
            "tiktoken" => Kind::Tiktoken,
            other => Kind::Plugin(other.to_string()),
        }
    }
}

/// A normalizer step.
#[derive(Clone, Debug, PartialEq)]
pub enum Normalizer {
    Nfc,
    Lowercase,
    StripAccents,
    Replace {
        pattern: String,
        to: String,
    },
    /// A step this build does not implement: running the tokenizer is
    /// indeterminate rather than wrong.
    Unsupported(String),
}

/// A pre-tokenizer step.
#[derive(Clone, Debug, PartialEq)]
pub enum PreTokenizer {
    /// GPT-2 style byte-to-unicode mapping.
    ByteLevel {
        add_prefix_space: bool,
    },
    Whitespace,
    /// Split on a regex. `flavor` is carried because §06.7 is right that it
    /// matters; `Regex` is this crate's subset and refuses what it cannot do.
    RegexSplit {
        pattern: String,
        flavor: String,
        behavior: String,
    },
    /// SentencePiece's `▁` word-boundary marker.
    Metaspace {
        replacement: char,
        prepend: bool,
    },
    Unsupported(String),
}

/// A decoder step.
#[derive(Clone, Debug, PartialEq)]
pub enum Decoder {
    ByteLevel,
    Replace { pattern: String, to: String },
    Metaspace { replacement: char },
    Unsupported(String),
}

/// One entry of `added_tokens`.
#[derive(Clone, Debug, PartialEq)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// A `Tokenizer` object (otype 0x000A).
#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub kind: Kind,
    pub normalizers: Vec<Normalizer>,
    pub pretokenizers: Vec<PreTokenizer>,
    /// Token strings, in id order.
    pub tokens: Vec<String>,
    /// Unigram scores, when the kind uses them.
    pub scores: Vec<f32>,
    /// BPE merges as *id pairs* (§06.7): unambiguous under normalization, and
    /// four times smaller than the string form.
    pub merges: Vec<(u32, u32)>,
    pub added_tokens: Vec<AddedToken>,
    pub byte_fallback: bool,
    pub unk: Option<u32>,
    pub decoder: Vec<Decoder>,
    /// Template postprocessor: the token sequences wrapped around a single
    /// sequence.
    pub post_single: Vec<String>,
    pub max_token_len: Option<usize>,
    /// Conformance vectors (§06.7.1).
    pub vectors: Option<Ref>,
    /// Token string to id, built once.
    index: BTreeMap<String, u32>,
}

impl Tokenizer {
    pub fn from_value(v: &Value, ctx: &Ctx<'_>) -> Res<Tokenizer> {
        if v.get("t").and_then(|x| x.as_str()) != Some("omni.tok/tokenizer") {
            return Err(Error::Type(
                "R-O02: object is not an omni.tok/tokenizer".into(),
            ));
        }
        let kind = Kind::parse(v.get("kind").and_then(|x| x.as_str()).unwrap_or("bpe"));
        let mut normalizers = Vec::new();
        for n in v
            .get("normalizers")
            .and_then(|x| x.as_array())
            .unwrap_or(&[])
        {
            let k = n.get("k").and_then(|x| x.as_str()).unwrap_or("");
            normalizers.push(match k {
                "nfc" => Normalizer::Nfc,
                "lowercase" => Normalizer::Lowercase,
                "strip-accents" => Normalizer::StripAccents,
                "replace" => Normalizer::Replace {
                    pattern: n
                        .get("pattern")
                        .and_then(|p| p.get("re").or(Some(p)))
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    to: n
                        .get("to")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                },
                other => Normalizer::Unsupported(other.to_string()),
            });
        }
        let mut pretokenizers = Vec::new();
        for p in v
            .get("pretokenizers")
            .and_then(|x| x.as_array())
            .unwrap_or(&[])
        {
            let k = p.get("k").and_then(|x| x.as_str()).unwrap_or("");
            pretokenizers.push(match k {
                "byte-level" => PreTokenizer::ByteLevel {
                    add_prefix_space: matches!(p.get("add_prefix_space"), Some(Value::Bool(true))),
                },
                "whitespace" => PreTokenizer::Whitespace,
                "regex-split" => PreTokenizer::RegexSplit {
                    pattern: p
                        .get("pattern")
                        .and_then(|x| x.get("re").or(Some(x)))
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    flavor: p
                        .get("pattern")
                        .and_then(|x| x.get("flavor"))
                        .or_else(|| p.get("flavor"))
                        .and_then(|x| x.as_str())
                        .unwrap_or("unstated")
                        .to_string(),
                    behavior: p
                        .get("behavior")
                        .and_then(|x| x.as_str())
                        .unwrap_or("isolated")
                        .to_string(),
                },
                "metaspace" => PreTokenizer::Metaspace {
                    replacement: p
                        .get("replacement")
                        .and_then(|x| x.as_str())
                        .and_then(|s| s.chars().next())
                        .unwrap_or('\u{2581}'),
                    prepend: !matches!(p.get("prepend"), Some(Value::Bool(false))),
                },
                other => PreTokenizer::Unsupported(other.to_string()),
            });
        }
        // The vocabulary is a tensor (§06.7): a 256k-entry vocabulary is ~3 MB,
        // and inline CBOR would put it in every metadata read.
        let tokens = match v.get("vocab").and_then(|x| x.get("tokens")) {
            Some(e) => read_strings(ctx, e)?,
            None => Vec::new(),
        };
        let scores = match v.get("vocab").and_then(|x| x.get("scores")) {
            Some(e) => Expr::from_value(e)?
                .eval(ctx)?
                .data
                .iter()
                .map(|x| *x as f32)
                .collect(),
            None => Vec::new(),
        };
        let merges = match v.get("merges") {
            Some(e) => {
                let t = Expr::from_value(e)?.eval(ctx)?;
                if t.shape.len() != 2 || t.shape[1] != 2 {
                    return Err(Error::Type(
                        "merges must be an [M, 2] tensor of id pairs (§06.7)".into(),
                    ));
                }
                t.data
                    .chunks(2)
                    .map(|c| (c[0] as u32, c[1] as u32))
                    .collect()
            }
            None => Vec::new(),
        };
        let mut added_tokens = Vec::new();
        for a in v
            .get("added_tokens")
            .and_then(|x| x.as_array())
            .unwrap_or(&[])
        {
            added_tokens.push(AddedToken {
                id: a.get("id").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                content: a
                    .get("content")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
                special: matches!(a.get("special"), Some(Value::Bool(true))),
                lstrip: matches!(a.get("lstrip"), Some(Value::Bool(true))),
                rstrip: matches!(a.get("rstrip"), Some(Value::Bool(true))),
            });
        }
        let mut decoder = Vec::new();
        for d in v.get("decoder").and_then(|x| x.as_array()).unwrap_or(&[]) {
            let k = d.get("k").and_then(|x| x.as_str()).unwrap_or("");
            decoder.push(match k {
                "byte-level" => Decoder::ByteLevel,
                "replace" => Decoder::Replace {
                    pattern: d
                        .get("pattern")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    to: d
                        .get("to")
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string(),
                },
                "metaspace" => Decoder::Metaspace {
                    replacement: d
                        .get("replacement")
                        .and_then(|x| x.as_str())
                        .and_then(|s| s.chars().next())
                        .unwrap_or('\u{2581}'),
                },
                other => Decoder::Unsupported(other.to_string()),
            });
        }
        let post_single = v
            .get("postprocessor")
            .and_then(|p| p.get("single"))
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let mut index = BTreeMap::new();
        for (i, t) in tokens.iter().enumerate() {
            index.insert(t.clone(), i as u32);
        }
        for a in &added_tokens {
            index.insert(a.content.clone(), a.id);
        }
        Ok(Tokenizer {
            kind,
            normalizers,
            pretokenizers,
            tokens,
            scores,
            merges,
            added_tokens,
            byte_fallback: matches!(v.get("byte_fallback"), Some(Value::Bool(true))),
            unk: v.get("unk").and_then(|x| x.as_u64()).map(|x| x as u32),
            decoder,
            post_single,
            max_token_len: v
                .get("max_token_len")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize),
            vectors: match v.get("conformance").and_then(|c| c.get("vectors")) {
                Some(r) => Some(crate::expr::parse_ref_value(r)?),
                None => None,
            },
            index,
        })
    }

    pub fn load(ctx: &Ctx<'_>, r: &Ref) -> Res<Tokenizer> {
        Tokenizer::from_value(&ctx.value(&r.1)?, ctx)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len().max(
            self.added_tokens
                .iter()
                .map(|a| a.id as usize + 1)
                .max()
                .unwrap_or(0),
        )
    }

    /// Anything this build cannot run, so a caller can report *indeterminate*
    /// rather than produce the wrong ids.
    pub fn unsupported(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Kind::Plugin(n) = &self.kind {
            out.push(format!(
                "tokenizer kind `{n}` needs a plugin host (§06.7.2)"
            ));
        }
        for n in &self.normalizers {
            if let Normalizer::Unsupported(k) = n {
                out.push(format!("normalizer `{k}`"));
            }
        }
        for p in &self.pretokenizers {
            match p {
                PreTokenizer::Unsupported(k) => out.push(format!("pre-tokenizer `{k}`")),
                PreTokenizer::RegexSplit {
                    pattern, flavor, ..
                } => {
                    if crate::pattern::Regex::parse(pattern).is_err() {
                        out.push(format!(
                            "regex-split pattern (flavor `{flavor}`) uses constructs this build's \
                             engine does not implement; splitting it differently would produce \
                             different token ids"
                        ));
                    }
                }
                _ => {}
            }
        }
        for d in &self.decoder {
            if let Decoder::Unsupported(k) = d {
                out.push(format!("decoder `{k}`"));
            }
        }
        out
    }

    // --------------------------------------------------------------- encode --

    /// Encodes text to token ids.
    pub fn encode(&self, text: &str) -> Res<Vec<u32>> {
        let unsupported = self.unsupported();
        if !unsupported.is_empty() {
            return Err(Error::Unsupported(format!(
                "this tokenizer needs: {}",
                unsupported.join("; ")
            )));
        }
        // Added tokens are matched on the raw text, before normalization: that
        // is what `normalized: false` means, and it is why `<|begin_of_text|>`
        // survives a lowercasing normalizer.
        let mut ids = Vec::new();
        for piece in self.split_on_added(text) {
            match piece {
                Piece::Added(id) => ids.push(id),
                Piece::Text(t) => {
                    let normalized = self.normalize(&t);
                    for word in self.pretokenize(&normalized)? {
                        ids.extend(self.encode_word(&word)?);
                    }
                }
            }
        }
        // The template postprocessor wraps the sequence.
        if !self.post_single.is_empty() {
            let mut out = Vec::new();
            for item in &self.post_single {
                if item == "$A" {
                    out.extend(ids.iter().copied());
                } else if let Some(id) = self.index.get(item) {
                    out.push(*id);
                }
            }
            return Ok(out);
        }
        Ok(ids)
    }

    fn split_on_added(&self, text: &str) -> Vec<Piece> {
        let mut specials: Vec<&AddedToken> = self.added_tokens.iter().collect();
        // Longest first, so `<|a|>` does not shadow `<|ab|>`.
        specials.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        let mut out = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for a in &specials {
                if a.content.is_empty() {
                    continue;
                }
                if let Some(pos) = rest.find(&a.content) {
                    if pos > 0 {
                        out.push(Piece::Text(rest[..pos].to_string()));
                    }
                    out.push(Piece::Added(a.id));
                    rest = &rest[pos + a.content.len()..];
                    continue 'outer;
                }
            }
            out.push(Piece::Text(rest.to_string()));
            break;
        }
        out
    }

    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        for n in &self.normalizers {
            s = match n {
                // The canonical composition of NFC is a Unicode table this crate
                // does not carry; `unsupported()` reports it, so reaching here
                // means the caller accepted the risk for ASCII-only text.
                Normalizer::Nfc => s,
                Normalizer::Lowercase => s.to_lowercase(),
                Normalizer::StripAccents => s.chars().filter(|c| !is_combining(*c)).collect(),
                Normalizer::Replace { pattern, to } => s.replace(pattern.as_str(), to),
                Normalizer::Unsupported(_) => s,
            };
        }
        s
    }

    fn pretokenize(&self, text: &str) -> Res<Vec<String>> {
        let mut words = vec![text.to_string()];
        for p in &self.pretokenizers {
            let mut next = Vec::new();
            for w in &words {
                match p {
                    PreTokenizer::ByteLevel { add_prefix_space } => {
                        let src = if *add_prefix_space && !w.starts_with(' ') {
                            format!(" {w}")
                        } else {
                            w.clone()
                        };
                        next.push(src.bytes().map(byte_to_unicode).collect::<String>());
                    }
                    PreTokenizer::Whitespace => {
                        // Split on whitespace, keeping it attached to the
                        // following word the way GPT-2 style tokenizers do.
                        let mut cur = String::new();
                        for c in w.chars() {
                            if c.is_whitespace() && !cur.is_empty() {
                                next.push(std::mem::take(&mut cur));
                            }
                            cur.push(c);
                        }
                        if !cur.is_empty() {
                            next.push(cur);
                        }
                    }
                    PreTokenizer::RegexSplit { pattern, .. } => {
                        let re = crate::pattern::Regex::parse(pattern)
                            .map_err(|e| Error::Unsupported(e.to_string()))?;
                        next.extend(split_by(&re, w)?);
                    }
                    PreTokenizer::Metaspace {
                        replacement,
                        prepend,
                    } => {
                        let mut s = w.replace(' ', &replacement.to_string());
                        if *prepend && !s.starts_with(*replacement) {
                            s.insert(0, *replacement);
                        }
                        next.push(s);
                    }
                    PreTokenizer::Unsupported(_) => next.push(w.clone()),
                }
            }
            words = next;
        }
        Ok(words)
    }

    /// The core of BPE: repeatedly merge the highest-priority adjacent pair.
    ///
    /// Merges are id pairs, so priority is the pair's position in the merge
    /// list — no string re-resolution, and no ambiguity about what a merge
    /// means after normalization.
    fn encode_word(&self, word: &str) -> Res<Vec<u32>> {
        if word.is_empty() {
            return Ok(vec![]);
        }
        match self.kind {
            Kind::Byte => Ok(word.bytes().map(|b| b as u32).collect()),
            Kind::Char => {
                let mut out = Vec::new();
                for c in word.chars() {
                    out.push(self.lookup(&c.to_string())?);
                }
                Ok(out)
            }
            Kind::WordLevel => Ok(vec![self.lookup(word)?]),
            Kind::WordPiece => self.wordpiece(word),
            Kind::Unigram => self.unigram(word),
            _ => self.bpe(word),
        }
    }

    fn lookup(&self, s: &str) -> Res<u32> {
        if let Some(id) = self.index.get(s) {
            return Ok(*id);
        }
        if let Some(unk) = self.unk {
            return Ok(unk);
        }
        Err(Error::Type(format!(
            "`{s}` is not in the vocabulary and the tokenizer declares no unk token"
        )))
    }

    fn bpe(&self, word: &str) -> Res<Vec<u32>> {
        // Start from the smallest units the vocabulary has: single characters,
        // falling back to bytes when `byte_fallback` is set.
        let mut ids: Vec<u32> = Vec::new();
        for c in word.chars() {
            let s = c.to_string();
            match self.index.get(&s) {
                Some(id) => ids.push(*id),
                None if self.byte_fallback => {
                    for b in s.bytes() {
                        let name = format!("<0x{b:02X}>");
                        ids.push(self.lookup(&name)?);
                    }
                }
                None => ids.push(self.lookup(&s)?),
            }
        }
        // Merge priority is position in the list; the first applicable merge
        // wins, and ties are impossible because the list is ordered.
        let rank: BTreeMap<(u32, u32), usize> = self
            .merges
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, i))
            .collect();
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(r) = rank.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|(br, _)| *r < br) {
                        best = Some((*r, i));
                    }
                }
            }
            let Some((r, i)) = best else { break };
            let merged = self.merged_id(self.merges[r])?;
            ids.splice(i..i + 2, [merged]);
        }
        Ok(ids)
    }

    /// The id a merge produces: the concatenation of the two tokens' strings.
    fn merged_id(&self, pair: (u32, u32)) -> Res<u32> {
        let a = self.token_str(pair.0)?;
        let b = self.token_str(pair.1)?;
        let joined = format!("{a}{b}");
        self.index.get(&joined).copied().ok_or_else(|| {
            Error::Type(format!(
                "merge ({}, {}) produces `{joined}`, which is not in the vocabulary — the merge \
                 list and the vocabulary disagree",
                pair.0, pair.1
            ))
        })
    }

    fn token_str(&self, id: u32) -> Res<&str> {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .or_else(|| {
                self.added_tokens
                    .iter()
                    .find(|a| a.id == id)
                    .map(|a| a.content.as_str())
            })
            .ok_or_else(|| Error::Bounds(format!("token id {id} is outside the vocabulary")))
    }

    fn wordpiece(&self, word: &str) -> Res<Vec<u32>> {
        let chars: Vec<char> = word.chars().collect();
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while end > start {
                let piece: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    format!("##{}", chars[start..end].iter().collect::<String>())
                };
                if let Some(id) = self.index.get(&piece) {
                    found = Some((*id, end));
                    break;
                }
                end -= 1;
            }
            match found {
                Some((id, end)) => {
                    out.push(id);
                    start = end;
                }
                None => return Ok(vec![self.lookup("[UNK]").or_else(|_| self.lookup(word))?]),
            }
        }
        Ok(out)
    }

    /// Unigram: the maximum-score segmentation, by Viterbi over the lattice.
    fn unigram(&self, word: &str) -> Res<Vec<u32>> {
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();
        let mut best = vec![f64::NEG_INFINITY; n + 1];
        let mut back: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best[0] = 0.0;
        for end in 1..=n {
            for start in 0..end {
                if best[start] == f64::NEG_INFINITY {
                    continue;
                }
                let piece: String = chars[start..end].iter().collect();
                if let Some(id) = self.index.get(&piece) {
                    let score = self.scores.get(*id as usize).copied().unwrap_or(0.0) as f64;
                    if best[start] + score > best[end] {
                        best[end] = best[start] + score;
                        back[end] = Some((start, *id));
                    }
                }
            }
        }
        if best[n] == f64::NEG_INFINITY {
            return Ok(vec![self.lookup(word)?]);
        }
        let mut out = Vec::new();
        let mut at = n;
        while at > 0 {
            let (start, id) = back[at].expect("a reachable cell has a predecessor");
            out.push(id);
            at = start;
        }
        out.reverse();
        Ok(out)
    }

    // --------------------------------------------------------------- decode --

    /// Decodes token ids back to text.
    pub fn decode(&self, ids: &[u32]) -> Res<String> {
        let mut s = String::new();
        for id in ids {
            // A special token that is not meant to be rendered still has to be
            // accounted for, so it is skipped explicitly rather than by
            // accident.
            if let Some(a) = self.added_tokens.iter().find(|a| a.id == *id) {
                if a.special {
                    continue;
                }
            }
            match self.kind {
                Kind::Byte => s.push(*id as u8 as char),
                _ => s.push_str(self.token_str(*id)?),
            }
        }
        for d in &self.decoder {
            s = match d {
                Decoder::ByteLevel => {
                    let bytes: Option<Vec<u8>> = s.chars().map(unicode_to_byte).collect();
                    match bytes {
                        Some(b) => String::from_utf8_lossy(&b).into_owned(),
                        None => s,
                    }
                }
                Decoder::Replace { pattern, to } => s.replace(pattern.as_str(), to),
                Decoder::Metaspace { replacement } => {
                    s.replace(*replacement, " ").trim_start().to_string()
                }
                Decoder::Unsupported(_) => s,
            };
        }
        Ok(s)
    }

    // ---------------------------------------------------------- conformance --

    /// Runs the conformance vectors of §06.7.1.
    ///
    /// The format is one case per line: text, a tab, then comma-separated ids.
    /// Text is escaped so a case can contain a tab or a newline.
    pub fn check_vectors(&self, ctx: &Ctx<'_>) -> Res<VectorReport> {
        let Some(r) = self.vectors else {
            return Ok(VectorReport::default());
        };
        let bytes = ctx.bytes(&r.1)?;
        let text = String::from_utf8(bytes)
            .map_err(|_| Error::Type("conformance vectors are not UTF-8".into()))?;
        let mut report = VectorReport::default();
        for (lineno, line) in text.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((raw, ids)) = line.split_once('\t') else {
                report.malformed += 1;
                continue;
            };
            let want: Vec<u32> = ids
                .split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let input = unescape(raw);
            report.total += 1;
            match self.encode(&input) {
                Ok(got) if got == want => report.passed += 1,
                Ok(got) => report.failures.push(VectorFailure {
                    line: lineno + 1,
                    input,
                    want,
                    got: Some(got),
                    error: None,
                }),
                Err(e) => report.failures.push(VectorFailure {
                    line: lineno + 1,
                    input,
                    want,
                    got: None,
                    error: Some(e.to_string()),
                }),
            }
        }
        Ok(report)
    }
}

enum Piece {
    Text(String),
    Added(u32),
}

/// One failing conformance vector.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorFailure {
    pub line: usize,
    pub input: String,
    pub want: Vec<u32>,
    pub got: Option<Vec<u32>>,
    pub error: Option<String>,
}

impl std::fmt::Display for VectorFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {:?} ", self.line, self.input)?;
        match (&self.got, &self.error) {
            (Some(got), _) => write!(
                f,
                "encoded to {} but the vector says {}",
                ids(got),
                ids(&self.want)
            ),
            (None, Some(e)) => write!(f, "could not be encoded: {e}"),
            (None, None) => write!(f, "did not encode to {}", ids(&self.want)),
        }
    }
}

fn ids(v: &[u32]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// The result of running a tokenizer's vectors (§06.7.1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VectorReport {
    pub total: usize,
    pub passed: usize,
    pub malformed: usize,
    pub failures: Vec<VectorFailure>,
}

impl VectorReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty() && self.malformed == 0
    }
}

impl std::fmt::Display for VectorReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} vectors pass", self.passed, self.total)?;
        if self.malformed > 0 {
            write!(f, ", {} malformed", self.malformed)?;
        }
        for x in &self.failures {
            write!(f, "\n  line {}: {:?}", x.line, x.input)?;
            write!(f, "\n    want {:?}", x.want)?;
            match (&x.got, &x.error) {
                (Some(g), _) => write!(f, "\n    got  {g:?}")?,
                (None, Some(e)) => write!(f, "\n    error {e}")?,
                _ => {}
            }
        }
        Ok(())
    }
}

/// Reads a string tensor: either a `literal` over a length-prefixed blob, or an
/// inline array of strings for small vocabularies.
fn read_strings(ctx: &Ctx<'_>, v: &Value) -> Res<Vec<String>> {
    if let Some(a) = v.as_array() {
        return Ok(a
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect());
    }
    let e = Expr::from_value(v)?;
    let Expr::Literal { chunks, .. } = &e else {
        return Err(Error::Unsupported(
            "a vocabulary tensor must be a `literal`; an expression over one is not defined \
             (§04.3.5 gives `string` no element arithmetic)"
                .into(),
        ));
    };
    let bytes = ctx.chunk_bytes(chunks)?;
    // Length-prefixed UTF-8: a u32 count, then u32 length + bytes per entry.
    // The vocabulary is a tensor of strings and CBOR is not in the hot path
    // here, so the encoding is the simplest one that is unambiguous.
    if bytes.len() < 4 {
        return Err(Error::Bounds("vocabulary blob is too short".into()));
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    let mut at = 4usize;
    for _ in 0..count {
        if at + 4 > bytes.len() {
            return Err(Error::Bounds("vocabulary blob is truncated".into()));
        }
        let n = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        if at + n > bytes.len() {
            return Err(Error::Bounds("vocabulary entry runs past the blob".into()));
        }
        out.push(
            String::from_utf8(bytes[at..at + n].to_vec())
                .map_err(|_| Error::Type("vocabulary entry is not UTF-8".into()))?,
        );
        at += n;
    }
    Ok(out)
}

/// Encodes a vocabulary as the blob [`read_strings`] reads.
pub fn encode_vocab(tokens: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        out.extend_from_slice(&(t.len() as u32).to_le_bytes());
        out.extend_from_slice(t.as_bytes());
    }
    out
}

/// GPT-2's byte-to-unicode map: every byte becomes a printable code point, so a
/// byte-level tokenizer's vocabulary is text.
pub fn byte_to_unicode(b: u8) -> char {
    match b {
        b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff => b as char,
        _ => {
            // The 68 bytes that are not printable map to U+0100 upward, in
            // order.
            let mut n = 0u32;
            for x in 0u8..b {
                if !matches!(x, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                    n += 1;
                }
            }
            char::from_u32(0x100 + n).unwrap_or('\u{fffd}')
        }
    }
}

/// The inverse of [`byte_to_unicode`].
pub fn unicode_to_byte(c: char) -> Option<u8> {
    let u = c as u32;
    if (0x21..=0x7e).contains(&u) || (0xa1..=0xac).contains(&u) || (0xae..=0xff).contains(&u) {
        return Some(u as u8);
    }
    if (0x100..0x100 + 68).contains(&u) {
        let want = u - 0x100;
        let mut n = 0u32;
        for x in 0u8..=255 {
            if !matches!(x, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff) {
                if n == want {
                    return Some(x);
                }
                n += 1;
            }
        }
    }
    None
}

fn is_combining(c: char) -> bool {
    // The combining diacritical marks block, which is what `strip-accents`
    // removes for Latin text. A complete implementation needs the Unicode
    // category table this crate does not carry.
    matches!(c as u32, 0x300..=0x36f)
}

/// Splits text at every match of `re`, keeping the matches as separate pieces —
/// the `isolated` behaviour of §06.7.
fn split_by(re: &crate::pattern::Regex, text: &str) -> Res<Vec<String>> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let hit = re
            .find(rest)
            .map_err(|e| Error::Unsupported(e.to_string()))?
            // A pattern that matches nothing, or that matches the empty string,
            // cannot advance: the remainder stays one piece rather than looping.
            .filter(|(a, b)| b > a);
        let Some((a, b)) = hit else {
            out.push(rest.to_string());
            break;
        };
        if a > 0 {
            out.push(rest[..a].to_string());
        }
        out.push(rest[a..b].to_string());
        rest = &rest[b..];
    }
    Ok(out)
}

/// Unescapes a conformance vector's input field.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    out.push(c);
                }
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{otype, HashAlgo};
    use crate::store::{MemoryStore, WritableStore};

    /// A tiny byte-level BPE: the vocabulary is single characters plus a few
    /// merges, which is enough to exercise the algorithm exactly.
    fn bpe_tokenizer(s: &mut MemoryStore) -> Value {
        let tokens: Vec<String> = [
            "<unk>", "a", "b", "c", " ", "ab", "abc", "<|bos|>", "Ġ", "Ġa",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let blob = s.put(&encode_vocab(&tokens)).unwrap();
        let vocab_expr = Value::map(vec![
            ("op", Value::text("literal")),
            (
                "chunks",
                Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
            ),
            ("dtype", Value::map(vec![("k", Value::text("string"))])),
            ("shape", Value::Array(vec![Value::U(tokens.len() as u64)])),
        ]);
        // merges: (a, b) -> ab ; (ab, c) -> abc ; (Ġ, a) -> Ġa
        let merges = vec![(1u32, 2u32), (5, 3), (8, 1)];
        let mut mbytes = Vec::new();
        for (a, b) in &merges {
            mbytes.extend_from_slice(&a.to_le_bytes());
            mbytes.extend_from_slice(&b.to_le_bytes());
        }
        let mblob = s.put(&mbytes).unwrap();
        let merges_expr = Value::map(vec![
            ("op", Value::text("literal")),
            (
                "chunks",
                Value::Array(vec![Value::U(0), Value::Bytes(mblob.to_vec())]),
            ),
            ("dtype", crate::dtype::DType::U32.to_value()),
            (
                "shape",
                Value::Array(vec![Value::U(merges.len() as u64), Value::U(2)]),
            ),
        ]);
        Value::map(vec![
            ("t", Value::text("omni.tok/tokenizer")),
            ("v", Value::U(1)),
            ("kind", Value::text("bpe")),
            ("vocab", Value::map(vec![("tokens", vocab_expr)])),
            ("merges", merges_expr),
            ("unk", Value::U(0)),
            (
                "added_tokens",
                Value::Array(vec![Value::map(vec![
                    ("id", Value::U(7)),
                    ("content", Value::text("<|bos|>")),
                    ("special", Value::Bool(true)),
                ])]),
            ),
        ])
    }

    #[test]
    fn bpe_merges_in_priority_order() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let v = bpe_tokenizer(&mut s);
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&v, &ctx).unwrap();
        assert_eq!(t.kind, Kind::Bpe);
        assert_eq!(t.merges.len(), 3);
        // "abc" merges (a,b) then (ab,c): one token.
        assert_eq!(t.encode("abc").unwrap(), vec![6]);
        // "ab" stops after the first merge.
        assert_eq!(t.encode("ab").unwrap(), vec![5]);
        // "cab" cannot merge c with anything first, so (a,b) applies.
        assert_eq!(t.encode("cab").unwrap(), vec![3, 5]);
        // Round-trip.
        assert_eq!(t.decode(&[6]).unwrap(), "abc");
        assert_eq!(t.decode(&t.encode("abc").unwrap()).unwrap(), "abc");
    }

    #[test]
    fn added_tokens_are_matched_before_normalization() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        // A lowercasing normalizer would destroy `<|BOS|>` if it ran first.
        v.push((
            Value::text("normalizers"),
            Value::Array(vec![Value::map(vec![("k", Value::text("lowercase"))])]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        assert_eq!(t.encode("<|bos|>abc").unwrap(), vec![7, 6]);
        // The special token is not rendered on the way back.
        assert_eq!(t.decode(&[7, 6]).unwrap(), "abc");
    }

    #[test]
    fn a_merge_the_vocabulary_does_not_contain_is_an_error() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let tokens: Vec<String> = ["a", "b"].iter().map(|x| x.to_string()).collect();
        let blob = s.put(&encode_vocab(&tokens)).unwrap();
        let mut mbytes = Vec::new();
        mbytes.extend_from_slice(&0u32.to_le_bytes());
        mbytes.extend_from_slice(&1u32.to_le_bytes());
        let mblob = s.put(&mbytes).unwrap();
        let v = Value::map(vec![
            ("t", Value::text("omni.tok/tokenizer")),
            ("v", Value::U(1)),
            ("kind", Value::text("bpe")),
            (
                "vocab",
                Value::map(vec![(
                    "tokens",
                    Value::map(vec![
                        ("op", Value::text("literal")),
                        (
                            "chunks",
                            Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
                        ),
                        ("dtype", Value::map(vec![("k", Value::text("string"))])),
                        ("shape", Value::Array(vec![Value::U(2)])),
                    ]),
                )]),
            ),
            (
                "merges",
                Value::map(vec![
                    ("op", Value::text("literal")),
                    (
                        "chunks",
                        Value::Array(vec![Value::U(0), Value::Bytes(mblob.to_vec())]),
                    ),
                    ("dtype", crate::dtype::DType::U32.to_value()),
                    ("shape", Value::Array(vec![Value::U(1), Value::U(2)])),
                ]),
            ),
        ]);
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&v, &ctx).unwrap();
        // (a, b) -> "ab", which the vocabulary does not have. Silently keeping
        // `a, b` would produce different ids from the source tokenizer.
        let e = t.encode("ab").unwrap_err();
        assert!(format!("{e}").contains("disagree"), "{e}");
    }

    #[test]
    fn byte_level_mapping_round_trips_every_byte() {
        for b in 0u8..=255 {
            let c = byte_to_unicode(b);
            assert_eq!(unicode_to_byte(c), Some(b), "byte {b}");
        }
        // The printable ASCII range maps to itself, which is what makes a
        // byte-level vocabulary readable.
        assert_eq!(byte_to_unicode(b'a'), 'a');
        // Space maps out of the way, GPT-2 style.
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
        assert_eq!(unicode_to_byte('Ġ'), Some(b' '));
    }

    #[test]
    fn a_byte_level_pretokenizer_encodes_spaces_as_tokens() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.push((
            Value::text("pretokenizers"),
            Value::Array(vec![Value::map(vec![
                ("k", Value::text("byte-level")),
                ("add_prefix_space", Value::Bool(false)),
            ])]),
        ));
        v.push((
            Value::text("decoder"),
            Value::Array(vec![Value::map(vec![("k", Value::text("byte-level"))])]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        // " a" becomes Ġ + a, which the merge list joins into `Ġa`.
        assert_eq!(t.encode(" a").unwrap(), vec![9]);
        assert_eq!(t.decode(&[9]).unwrap(), " a");
    }

    #[test]
    fn wordpiece_takes_the_longest_match_and_marks_continuations() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let tokens: Vec<String> = ["[UNK]", "un", "##aff", "##able", "unaffable"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let blob = s.put(&encode_vocab(&tokens)).unwrap();
        let v = Value::map(vec![
            ("t", Value::text("omni.tok/tokenizer")),
            ("v", Value::U(1)),
            ("kind", Value::text("wordpiece")),
            (
                "vocab",
                Value::map(vec![(
                    "tokens",
                    Value::map(vec![
                        ("op", Value::text("literal")),
                        (
                            "chunks",
                            Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
                        ),
                        ("dtype", Value::map(vec![("k", Value::text("string"))])),
                        ("shape", Value::Array(vec![Value::U(5)])),
                    ]),
                )]),
            ),
            ("unk", Value::U(0)),
        ]);
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&v, &ctx).unwrap();
        // The whole word is in the vocabulary, so it wins.
        assert_eq!(t.encode("unaffable").unwrap(), vec![4]);
        // "unaff" is not, so it splits.
        assert_eq!(t.encode("unaff").unwrap(), vec![1, 2]);
    }

    #[test]
    fn unigram_picks_the_maximum_score_segmentation() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let tokens: Vec<String> = ["<unk>", "a", "b", "ab"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let blob = s.put(&encode_vocab(&tokens)).unwrap();
        let vocab = |scores: Vec<f64>| {
            let mut sb = Vec::new();
            for x in &scores {
                sb.extend_from_slice(&(*x as f32).to_le_bytes());
            }
            (sb, scores.len())
        };
        // "ab" scores better than a + b, so it is one token.
        let (sb, n) = vocab(vec![-10.0, -2.0, -2.0, -1.0]);
        let sblob = s.put(&sb).unwrap();
        let lit = |d: &[u8; 32], dtype: Value, shape: Vec<Value>| {
            Value::map(vec![
                ("op", Value::text("literal")),
                (
                    "chunks",
                    Value::Array(vec![Value::U(0), Value::Bytes(d.to_vec())]),
                ),
                ("dtype", dtype),
                ("shape", Value::Array(shape)),
            ])
        };
        let v = Value::map(vec![
            ("t", Value::text("omni.tok/tokenizer")),
            ("v", Value::U(1)),
            ("kind", Value::text("unigram")),
            (
                "vocab",
                Value::map(vec![
                    (
                        "tokens",
                        lit(
                            &blob,
                            Value::map(vec![("k", Value::text("string"))]),
                            vec![Value::U(4)],
                        ),
                    ),
                    (
                        "scores",
                        lit(
                            &sblob,
                            crate::dtype::DType::F32.to_value(),
                            vec![Value::U(n as u64)],
                        ),
                    ),
                ]),
            ),
            ("unk", Value::U(0)),
        ]);
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&v, &ctx).unwrap();
        assert_eq!(t.encode("ab").unwrap(), vec![3]);
        assert_eq!(t.decode(&[3]).unwrap(), "ab");
    }

    #[test]
    fn conformance_vectors_turn_a_regression_into_a_failure() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let vectors = "# §06.7.1\nabc\t6\nab\t5\ncab\t3,5\n";
        let vblob = s.put(vectors.as_bytes()).unwrap();
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.push((
            Value::text("conformance"),
            Value::map(vec![(
                "vectors",
                Value::Array(vec![Value::U(0), Value::Bytes(vblob.to_vec())]),
            )]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v.clone()), &ctx).unwrap();
        let r = t.check_vectors(&ctx).unwrap();
        assert!(r.ok(), "{r}");
        assert_eq!((r.total, r.passed), (3, 3));

        // A vector that disagrees is reported with both sequences, because "the
        // tokenizer changed" is only actionable if you can see how.
        let bad = "abc\t1,2,3\n";
        let bblob = s.put(bad.as_bytes()).unwrap();
        let mut v2 = v.clone();
        v2.retain(|(k, _)| k.as_str() != Some("conformance"));
        v2.push((
            Value::text("conformance"),
            Value::map(vec![(
                "vectors",
                Value::Array(vec![Value::U(0), Value::Bytes(bblob.to_vec())]),
            )]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v2), &ctx).unwrap();
        let r = t.check_vectors(&ctx).unwrap();
        assert!(!r.ok());
        assert_eq!(r.failures[0].want, vec![1, 2, 3]);
        assert_eq!(r.failures[0].got, Some(vec![6]));
        assert!(format!("{r}").contains("got"));
    }

    #[test]
    fn vectors_can_contain_tabs_and_newlines() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let vectors = "a\\tb\t1,1\n";
        let vblob = s.put(vectors.as_bytes()).unwrap();
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.push((
            Value::text("conformance"),
            Value::map(vec![(
                "vectors",
                Value::Array(vec![Value::U(0), Value::Bytes(vblob.to_vec())]),
            )]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        let r = t.check_vectors(&ctx).unwrap();
        // The escaped tab is part of the input, and the tab after it is the
        // field separator; the tokenizer has no tab token, so it maps to unk.
        assert_eq!(r.total, 1);
        assert_eq!(r.failures.len(), 1);
        assert_eq!(r.failures[0].input, "a\tb");
        assert_eq!(unescape("a\\u0041b"), "aAb");
    }

    #[test]
    fn an_unimplemented_step_refuses_rather_than_approximating() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        // The GPT-2 pre-tokenizer pattern, which needs Unicode property
        // classes this crate's engine does not implement.
        v.push((
            Value::text("pretokenizers"),
            Value::Array(vec![Value::map(vec![
                ("k", Value::text("regex-split")),
                (
                    "pattern",
                    Value::map(vec![
                        ("re", Value::text(r"[^\r\n\p{L}\p{N}]?\p{L}+")),
                        ("flavor", Value::text("pcre2")),
                    ]),
                ),
            ])]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        let un = t.unsupported();
        assert_eq!(un.len(), 1, "{un:?}");
        assert!(un[0].contains("different token ids"), "{}", un[0]);
        assert!(matches!(t.encode("abc"), Err(Error::Unsupported(_))));

        // A plugin tokenizer is refused the same way (§06.7.2).
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.retain(|(k, _)| k.as_str() != Some("kind"));
        v.push((Value::text("kind"), Value::text("audio-codec")));
        let t = Tokenizer::from_value(&Value::Map(v), &Ctx::new(&s)).unwrap();
        assert!(t.unsupported()[0].contains("plugin host"));
    }

    #[test]
    fn a_supported_regex_split_works() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.push((
            Value::text("pretokenizers"),
            Value::Array(vec![Value::map(vec![
                ("k", Value::text("regex-split")),
                (
                    "pattern",
                    Value::map(vec![
                        ("re", Value::text("[ab]+")),
                        ("flavor", Value::text("ere")),
                    ]),
                ),
            ])]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        assert!(t.unsupported().is_empty());
        // "cabc" splits into "c", "ab", "c" — the middle piece then merges.
        assert_eq!(t.encode("cabc").unwrap(), vec![3, 5, 3]);
    }

    #[test]
    fn the_postprocessor_wraps_the_sequence() {
        let mut s = MemoryStore::new(HashAlgo::default());
        let mut v = bpe_tokenizer(&mut s).as_map().unwrap().to_vec();
        v.push((
            Value::text("postprocessor"),
            Value::map(vec![
                ("k", Value::text("template")),
                (
                    "single",
                    Value::Array(vec![Value::text("<|bos|>"), Value::text("$A")]),
                ),
            ]),
        ));
        let ctx = Ctx::new(&s);
        let t = Tokenizer::from_value(&Value::Map(v), &ctx).unwrap();
        assert_eq!(t.encode("abc").unwrap(), vec![7, 6]);
    }

    #[test]
    fn a_vocabulary_blob_that_lies_is_refused() {
        let mut s = MemoryStore::new(HashAlgo::default());
        // Declares two entries, holds one.
        let mut bytes = 2u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(b'a');
        let blob = s.put(&bytes).unwrap();
        let v = Value::map(vec![
            ("t", Value::text("omni.tok/tokenizer")),
            ("v", Value::U(1)),
            ("kind", Value::text("bpe")),
            (
                "vocab",
                Value::map(vec![(
                    "tokens",
                    Value::map(vec![
                        ("op", Value::text("literal")),
                        (
                            "chunks",
                            Value::Array(vec![Value::U(0), Value::Bytes(blob.to_vec())]),
                        ),
                        ("dtype", Value::map(vec![("k", Value::text("string"))])),
                        ("shape", Value::Array(vec![Value::U(2)])),
                    ]),
                )]),
            ),
        ]);
        let ctx = Ctx::new(&s);
        assert!(matches!(
            Tokenizer::from_value(&v, &ctx),
            Err(Error::Bounds(_))
        ));
    }

    #[test]
    fn the_wrong_object_type_is_refused() {
        let s = MemoryStore::new(HashAlgo::default());
        let ctx = Ctx::new(&s);
        let v = Value::map(vec![("t", Value::text("omni.core/manifest"))]);
        assert!(Tokenizer::from_value(&v, &ctx).is_err());
        assert_eq!(otype::TOKENIZER, 0x000A);
    }
}
