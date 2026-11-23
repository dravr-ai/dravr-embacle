// ABOUTME: The one log line a classified vendor failure leaves: what the vendor sent, redacted and capped
// ABOUTME: Owns the free-text redactor every provider's error path shares, HTTP or CLI
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! What the vendor sent, beside what it was classified as.
//!
//! A [`RunnerError`] carries the kind a failure was given and a shortened
//! message. Neither says what the vendor actually answered: the HTTP status,
//! the `retry-after` header, the vendor's request id and the body are gone by
//! the time the error reaches a log. [`log_classified`] writes them once, at
//! the site that classifies, so a production log line can become a test
//! fixture built on a real body.
//!
//! A vendor body can echo the request, and a request can carry a credential
//! or a person's data, so nothing reaches the line before [`redact_and_cap`]
//! has read it. The redactor errs towards removing too much.

use std::borrow::Cow;

#[cfg(any(feature = "http-api", feature = "quota-http"))]
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};
use tracing::warn;

use crate::types::RunnerError;

/// The most characters of a vendor body one log line carries.
pub(crate) const BODY_CAP_CHARS: usize = 600;

/// What stands in for anything shaped like a key or a token.
pub(crate) const SECRET_MARKER: &str = "[REDACTED]";

/// What stands in for an email address.
pub(crate) const EMAIL_MARKER: &str = "[EMAIL]";

/// The most bytes of a vendor text the redactor reads. A failed CLI can
/// hand over megabytes to keep [`BODY_CAP_CHARS`] characters of.
const SCAN_CAP_BYTES: usize = 16 * 1024;

/// An unbroken run this long is taken for a secret whatever it looks like.
const LONG_RUN_CHARS: usize = 24;

/// Three dot-joined parts each this long are a token, not a host name.
const DOTTED_PART_CHARS: usize = 16;

/// The shortest word after `Basic` that is read as credentials.
const BASIC_MIN_CHARS: usize = 8;

/// The most characters of a response header value the line carries.
#[cfg(any(feature = "http-api", feature = "quota-http"))]
const HEADER_CAP_CHARS: usize = 128;

/// How the vendors' API keys and tokens start: `OpenAI` / Anthropic /
/// `OpenRouter` (`sk-`, which covers `sk-ant-` and `sk-or-`), Google
/// (`AIza`), Groq (`gsk_`), GitHub, xAI, Hugging Face, GitLab, AWS and
/// Slack.
const KEY_PREFIXES: [&str; 15] = [
    "sk-",
    "AIza",
    "gsk_",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "xai-",
    "hf_",
    "glpat-",
    "AKIA",
    "xoxb-",
    "xoxp-",
];

/// What a field that holds a secret is called, as the end of its name:
/// `access_token`, `client_secret` and `x-api-key` all match, `max_tokens`
/// does not.
const SECRET_NAMES: [&str; 8] = [
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "api-key",
    "apikey",
    "authorization",
];

/// The headers a vendor may name its request in, in the order they are
/// tried: Anthropic's `request-id`, the common `x-request-id`, Google's and
/// GitHub's own, then two fallbacks for a response that carries none of
/// those — a trace id and the Cloudflare ray id of a proxied endpoint.
#[cfg(any(feature = "http-api", feature = "quota-http"))]
const REQUEST_ID_HEADERS: [&str; 6] = [
    "request-id",
    "x-request-id",
    "x-goog-request-id",
    "x-github-request-id",
    "x-debug-trace-id",
    "cf-ray",
];

/// A vendor text made safe for one log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoggedText {
    /// The text: one line, redacted, at most [`BODY_CAP_CHARS`] characters
    pub(crate) text: String,
    /// The length of the original, in bytes
    pub(crate) bytes: usize,
    /// Whether the line carries less than the redacted original
    pub(crate) truncated: bool,
}

/// `raw` as a log line may carry it.
///
/// - Every run of whitespace or control characters becomes one space, so the
///   value is a single line.
/// - Anything key- or token-shaped becomes [`SECRET_MARKER`]: a word that
///   holds the start of a vendor key ([`KEY_PREFIXES`]), the word after
///   `Bearer` or `Basic`, a JWT, three long dot-joined parts, the value of a
///   field named like a secret ([`SECRET_NAMES`]), and any unbroken run of
///   24 or more characters from `[A-Za-z0-9_\-+/=]`. A JSON-escaped slash
///   and a percent escape do not break a run.
/// - An email address becomes [`EMAIL_MARKER`], whatever its local part
///   holds.
/// - The result is cut at [`BODY_CAP_CHARS`] characters, after redaction, on
///   a character boundary.
///
/// Only the first [`SCAN_CAP_BYTES`] of `raw` are read. The rules take
/// identifiers with them (a UUID, a long URL path, a 24-character error
/// code, the word after `token:`): a readable word lost costs less than a
/// key kept.
pub(crate) fn redact_and_cap(raw: &str) -> LoggedText {
    let (head, cut_early) = bounded(raw);
    let (text, cut) = cap(redact(&one_line(head), LONG_RUN_CHARS));
    LoggedText {
        text,
        bytes: raw.len(),
        truncated: cut_early || cut,
    }
}

/// The part of `raw` the redactor reads, and whether that left some out.
///
/// The word the bound falls in is dropped with what follows it: the head of
/// a key cut in two can be too short to be recognised.
fn bounded(raw: &str) -> (&str, bool) {
    if raw.len() <= SCAN_CAP_BYTES {
        return (raw, false);
    }
    let cut = (0..=SCAN_CAP_BYTES)
        .rev()
        .find(|&at| raw.is_char_boundary(at))
        .unwrap_or(0);
    let head = raw[..cut].trim_end_matches(|c: char| {
        c.is_ascii() && (is_run_byte(c as u8) || c == '%' || c == '\\')
    });
    (head, true)
}

/// `text` with each run of whitespace or control characters as one space,
/// and none at either end.
fn one_line(text: &str) -> String {
    let mut line = String::with_capacity(text.len());
    let mut gap = false;
    for c in text.chars() {
        if c.is_whitespace() || c.is_control() {
            gap = true;
            continue;
        }
        if gap && !line.is_empty() {
            line.push(' ');
        }
        gap = false;
        line.push(c);
    }
    line
}

/// `text` cut at [`BODY_CAP_CHARS`] characters, and whether it was cut.
fn cap(mut text: String) -> (String, bool) {
    match text.char_indices().nth(BODY_CAP_CHARS) {
        Some((cut, _)) => {
            text.truncate(cut);
            (text, true)
        }
        None => (text, false),
    }
}

/// Whether `b` can be part of a key, a token or an email address: the
/// base64 and identifier alphabet, plus the `.` and `@` that join a JWT's
/// and an address's parts.
const fn is_run_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'+' | b'/' | b'=' | b'.' | b'@')
}

/// How many bytes at the start of `text` continue a run: one run character,
/// a JSON-escaped slash (`\/`) or a percent escape (`%2F`). Escaping must
/// not cut a secret into pieces short enough to pass.
fn run_unit(text: &str) -> usize {
    match text.as_bytes() {
        [b'\\', b'/', ..] => 2,
        [b'%', high, low, ..] if high.is_ascii_hexdigit() && low.is_ascii_hexdigit() => 3,
        [first, ..] if is_run_byte(*first) => 1,
        _ => 0,
    }
}

/// The length of the run `text` starts with, in bytes. A run is ASCII.
fn run_len(text: &str) -> usize {
    let mut len = 0;
    loop {
        match run_unit(&text[len..]) {
            0 => return len,
            unit => len += unit,
        }
    }
}

/// Whether `text` ends with `suffix`, whatever the case of its letters.
fn ends_with_ignore_case(text: &str, suffix: &str) -> bool {
    text.len() >= suffix.len()
        && text.as_bytes()[text.len() - suffix.len()..].eq_ignore_ascii_case(suffix.as_bytes())
}

/// One-line `text` with its secrets and addresses replaced. A segment of
/// `long_run` characters is a secret whatever it looks like.
fn redact(text: &str, long_run: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let len = run_len(rest);
        if len == 0 {
            let mut chars = rest.chars();
            if let Some(c) = chars.next() {
                out.push(c);
            }
            rest = chars.as_str();
            continue;
        }
        let (run, tail) = rest.split_at(len);
        rest = push_run(&mut out, run, tail, long_run);
    }
    out
}

/// Write one run, redacted, with whatever it makes a secret of in `tail`.
/// Returns what is left to read.
///
/// The dots that end a sentence are not part of the word they follow, and a
/// word that ends a sentence says nothing about the next one.
fn push_run<'a>(out: &mut String, run: &str, tail: &'a str, long_run: usize) -> &'a str {
    let word = run.trim_end_matches('.');
    let dots = &run[word.len()..];
    if let Some(rest) = push_address(out, word, if dots.is_empty() { tail } else { "" }) {
        out.push_str(dots);
        return if dots.is_empty() { rest } else { tail };
    }
    match secret_start(word) {
        Some(at) => {
            push_segments(out, &word[..at], long_run);
            out.push_str(SECRET_MARKER);
        }
        None => push_segments(out, word, long_run),
    }
    out.push_str(dots);
    if !dots.is_empty() {
        return tail;
    }
    match what_follows(word) {
        Some(Follows::Token { strict }) => skip_scheme_token(out, tail, strict),
        Some(Follows::Value { separated }) => skip_named_value(out, tail, separated),
        None => tail,
    }
}

/// Whether `c` ends an address on either side: whitespace, a quote's or a
/// bracket's neighbours, and what separates fields and URL parameters.
fn is_address_break(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '"' | '<'
                | '>'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | ','
                | ';'
                | ':'
                | '|'
                | '\\'
                | '`'
                | '='
                | '&'
                | '?'
                | '#'
                | '/'
        )
}

/// Replace `word` when it is, or is part of, an email address. Returns what
/// is left of `tail`, or `None` when `word` holds no address.
///
/// The `@` (or its escape, `%40`) anchors the address. A local part can hold
/// what a run cannot (`rené@`, `o'brien@`, `"jane doe"@`), so the part of it
/// already written is taken back, and a domain can go on past the run
/// (`@exämple.com`). A bare `@name` with nothing before it and no dot is a
/// handle or a package scope, and stays.
fn push_address<'a>(out: &mut String, word: &str, tail: &'a str) -> Option<&'a str> {
    let (local, domain) = word.split_once('@').or_else(|| word.split_once("%40"))?;
    let written = written_local_len(out);
    let more = address_tail_len(tail, domain.is_empty());
    let has_domain = !domain.is_empty() || more > 0;
    let has_local = !local.is_empty() || written > 0;
    if !has_domain || !(has_local || domain.contains('.')) {
        return None;
    }
    out.truncate(out.len() - written);
    out.push_str(EMAIL_MARKER);
    Some(&tail[more..])
}

/// How many bytes at the end of `out` belong to the local part of an address
/// whose `@` comes next.
fn written_local_len(out: &str) -> usize {
    if let Some(quoted) = quoted_local_len(out) {
        return quoted;
    }
    let mut len = 0;
    let mut chars = out.chars().rev().peekable();
    while let Some(c) = chars.next() {
        // An apostrophe is part of a name (`o'brien`), not of a quotation.
        let joins = if c == '\'' {
            chars.peek().is_some_and(|before| before.is_alphanumeric())
        } else {
            !is_address_break(c)
        };
        if !joins {
            break;
        }
        len += c.len_utf8();
    }
    len
}

/// The length of a quoted local part (`"jane doe"`) at the end of `out`,
/// with its quotes. A quote that closes a JSON key or value has structure
/// between it and the one before, and is not one.
fn quoted_local_len(out: &str) -> Option<usize> {
    let inner = out.strip_suffix('"')?;
    let open = inner.rfind('"')?;
    let content = &inner[open + 1..];
    if content.contains([':', '{', '}', '[', ']', ',']) || !content.contains(char::is_alphanumeric)
    {
        return None;
    }
    let open = if inner[..open].ends_with('\\') {
        open - 1
    } else {
        open
    };
    Some(out.len() - open)
}

/// How many bytes of `tail` continue a domain the run could not hold. When
/// the run held none of it (`must_start`), it has to start like one.
fn address_tail_len(tail: &str, must_start: bool) -> usize {
    if must_start && !tail.starts_with(char::is_alphanumeric) {
        return 0;
    }
    tail.find(|c: char| c == '\'' || is_address_break(c))
        .unwrap_or(tail.len())
}

/// Where the rest of `word` is a secret, if it holds one: at a JWT, at a
/// vendor key, after the `=` of a field named like a secret, or from its
/// start when it is three long dot-joined parts.
fn secret_start(word: &str) -> Option<usize> {
    if is_dotted_token(word) {
        return Some(0);
    }
    [jwt_start(word), key_start(word), named_value_start(word)]
        .into_iter()
        .flatten()
        .min()
}

/// Whether what starts at `at` in `word` starts a word of its own: at the
/// start, after a joining character (`key=sk-…`) or after a percent escape
/// (`key%3Dsk-…`), never after a letter or a digit: `risk-free` holds `sk-`
/// and no key.
fn follows_a_break(word: &str, at: usize) -> bool {
    let bytes = word.as_bytes();
    at == 0
        || !bytes[at - 1].is_ascii_alphanumeric()
        || (at >= 3
            && bytes[at - 3] == b'%'
            && bytes[at - 2].is_ascii_hexdigit()
            && bytes[at - 1].is_ascii_hexdigit())
}

/// Where a vendor key starts in `word`, if one does.
fn key_start(word: &str) -> Option<usize> {
    KEY_PREFIXES
        .iter()
        .flat_map(|prefix| word.match_indices(prefix))
        .map(|(at, _)| at)
        .filter(|&at| follows_a_break(word, at))
        .min()
}

/// Where a JWT starts in `word`, if one does: the base64 of a JSON object's
/// opening, with two more dot-joined parts after it. A host name or a
/// version number has the dots and not the header.
fn jwt_start(word: &str) -> Option<usize> {
    word.match_indices("eyJ")
        .map(|(at, _)| at)
        .find(|&at| follows_a_break(word, at) && word[at..].matches('.').count() >= 2)
}

/// Whether `word` is three or more dot-joined parts, each a long base64url
/// run: a signed token without a JWT's header. Host names, versions and
/// dotted identifiers have a short part.
fn is_dotted_token(word: &str) -> bool {
    word.split('.').count() >= 3
        && word.split('.').all(|part| {
            part.len() >= DOTTED_PART_CHARS
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        })
}

/// Whether `name` is what a field that holds a secret is called.
fn names_a_secret(name: &str) -> bool {
    SECRET_NAMES
        .iter()
        .any(|secret| ends_with_ignore_case(name, secret))
}

/// Whether `word` ends with the authorization scheme `scheme` as a word of
/// its own.
fn ends_with_scheme(word: &str, scheme: &str) -> bool {
    ends_with_ignore_case(word, scheme) && follows_a_break(word, word.len() - scheme.len())
}

/// Whether `value` is an authorization scheme, with the credentials still
/// to come.
fn is_scheme(value: &str) -> bool {
    value.eq_ignore_ascii_case("bearer") || value.eq_ignore_ascii_case("basic")
}

/// Where the value of a secret starts inside `word`, if it holds one: after
/// `=` or its escape when a field named like a secret or `Bearer` comes
/// before, and after an escaped space when `Bearer` does.
fn named_value_start(word: &str) -> Option<usize> {
    const SEPARATORS: [&str; 3] = ["=", "%3d", "%20"];
    if !word.contains(['=', '%']) {
        return None;
    }
    let lower = word.to_ascii_lowercase();
    SEPARATORS
        .iter()
        .flat_map(|separator| lower.match_indices(separator))
        .filter_map(|(at, separator)| {
            let (name, value) = (&lower[..at], &lower[at + separator.len()..]);
            let named =
                ends_with_scheme(name, "bearer") || (separator != "%20" && names_a_secret(name));
            (named && !value.is_empty() && !is_scheme(value)).then_some(at + separator.len())
        })
        .min()
}

/// What a word makes of the text after it.
enum Follows {
    /// The credentials of an authorization scheme. `strict` when the scheme
    /// is also an ordinary word, so only something shaped like credentials
    /// counts
    Token { strict: bool },
    /// The value of a field named like a secret, after its separator or
    /// with it still to read
    Value { separated: bool },
}

/// What `word` says about the text after it, if anything.
fn what_follows(word: &str) -> Option<Follows> {
    if ends_with_scheme(word, "bearer") {
        return Some(Follows::Token { strict: false });
    }
    if ends_with_scheme(word, "basic") {
        return Some(Follows::Token { strict: true });
    }
    if names_a_secret(word) {
        return Some(Follows::Value { separated: false });
    }
    let name = word
        .strip_suffix('=')
        .or_else(|| word.strip_suffix("%3D"))
        .or_else(|| word.strip_suffix("%3d"))?;
    names_a_secret(name).then_some(Follows::Value { separated: true })
}

/// `text` without the quote it opens with, and what closes that quote: a
/// plain one, or the escaped one of JSON carried inside a JSON string.
fn opening_quote(text: &str) -> (Option<&'static str>, &str) {
    ["\\\"", "\"", "'"]
        .into_iter()
        .find_map(|quote| text.strip_prefix(quote).map(|rest| (Some(quote), rest)))
        .unwrap_or((None, text))
}

/// Where the quote `close` ends `body`: its whole length when it never
/// does. A backslash keeps a plain double quote from closing.
fn closing_quote_at(body: &str, close: &str) -> usize {
    if close != "\"" {
        return body.find(close).unwrap_or(body.len());
    }
    let mut escaped = false;
    for (at, c) in body.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '"' => return at,
            _ => {}
        }
    }
    body.len()
}

/// Whether `token` is shaped like `Basic` credentials rather than a word:
/// base64, long enough, and not all lowercase letters.
fn is_base64_word(token: &str) -> bool {
    token.len() >= BASIC_MIN_CHARS
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'_' | b'-'))
        && token
            .bytes()
            .any(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

/// Replace the word that follows an authorization scheme, when one does.
/// Returns what is left to read.
///
/// The word may come after a colon and inside quotes. After `Bearer` it is
/// not weighed: `Bearer token expired` loses `token`, the price of never
/// keeping a short or oddly shaped credential.
fn skip_scheme_token<'a>(out: &mut String, tail: &'a str, strict: bool) -> &'a str {
    let after_gap = tail.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
    if after_gap.len() == tail.len() {
        return tail;
    }
    let (_, token_on) = opening_quote(after_gap);
    let end = token_on
        .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | '\\'))
        .unwrap_or(token_on.len());
    let token = &token_on[..end];
    if token.is_empty() || (strict && !is_base64_word(token)) {
        return tail;
    }
    out.push_str(&tail[..tail.len() - token_on.len()]);
    out.push_str(SECRET_MARKER);
    &token_on[end..]
}

/// Replace the value of a field named like a secret, when one follows its
/// name: `name: value`, `"name":"value"`, `name="value"`. Returns what is
/// left to read.
///
/// A quoted value goes whole, an unquoted one up to what ends a word. A
/// nested object or array is read on as text, and an authorization scheme
/// stays beside its replaced credentials.
fn skip_named_value<'a>(out: &mut String, tail: &'a str, separated: bool) -> &'a str {
    let mut rest = tail;
    if !separated {
        let (_, after_quote) = opening_quote(rest);
        let Some(after_separator) = after_quote.trim_start().strip_prefix([':', '=']) else {
            return tail;
        };
        rest = after_separator;
    }
    let (quote, body) = opening_quote(rest.trim_start());
    let end = quote.map_or_else(
        || {
            body.find(|c: char| {
                c.is_whitespace()
                    || matches!(c, ',' | ';' | '&' | '"' | '\'' | '}' | ']' | ')' | '\\')
            })
            .unwrap_or(body.len())
        },
        |close| closing_quote_at(body, close),
    );
    let value = &body[..end];
    if value.is_empty() || (quote.is_none() && value.starts_with(['{', '['])) {
        return tail;
    }
    out.push_str(&tail[..tail.len() - body.len()]);
    if quote.is_none() && is_scheme(value.trim_end_matches(':')) {
        let scheme = value.trim_end_matches(':');
        out.push_str(scheme);
        return skip_scheme_token(out, &body[scheme.len()..], false);
    }
    out.push_str(SECRET_MARKER);
    &body[end..]
}

/// Write `word` part by part: `.` and `@` separate the parts and are kept.
fn push_segments(out: &mut String, word: &str, long_run: usize) {
    let mut rest = word;
    while let Some(at) = rest.find(['.', '@']) {
        let (segment, tail) = rest.split_at(at);
        push_segment(out, segment, long_run);
        let mut chars = tail.chars();
        if let Some(separator) = chars.next() {
            out.push(separator);
        }
        rest = chars.as_str();
    }
    push_segment(out, rest, long_run);
}

/// Write one unbroken run from `[A-Za-z0-9_\-+/=]` and its escapes: the
/// marker when it is long enough to be a secret, itself otherwise.
fn push_segment(out: &mut String, segment: &str, long_run: usize) {
    if segment.len() >= long_run {
        out.push_str(SECRET_MARKER);
    } else {
        out.push_str(segment);
    }
}

/// A vendor text that did not read as the vendor's protocol, reduced to
/// what is safe to log of it.
///
/// A frame of a success-status stream, a 200 body or a CLI's JSON line can
/// be the model's own output, which no redactor makes safe: of a JSON
/// document only the `type` and `error` members are kept. Plain text is the
/// vendor talking (`Error: Not authenticated`, a proxy's error page) and is
/// kept for [`redact_and_cap`], up to where a JSON span opens in it: a
/// warning printed before the answer keeps the warning, and a document cut
/// short keeps nothing.
fn error_members(text: &str) -> Cow<'_, str> {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(mut document)) => {
            let kept: Map<String, Value> = ["type", "error"]
                .into_iter()
                .filter_map(|name| Some((name.to_owned(), document.remove(name)?)))
                .filter(|(name, value)| name == "error" || value.is_string())
                .collect();
            if kept.is_empty() {
                Cow::Borrowed("")
            } else {
                Cow::Owned(Value::Object(kept).to_string())
            }
        }
        Ok(_) => Cow::Borrowed(""),
        Err(_) => Cow::Borrowed(&text[..json_opens_at(text).unwrap_or(text.len())]),
    }
}

/// Where a JSON span opens in `text`, if one does: a string that starts
/// it, an object at its first key, an array at its first string, object or
/// array. A bracketed tag (`[ERROR]`) and a brace in prose open none.
fn json_opens_at(text: &str) -> Option<usize> {
    if text.trim_start().starts_with('"') {
        return Some(0);
    }
    text.char_indices().find_map(|(at, c)| {
        let inside = text[at + c.len_utf8()..].trim_start();
        let opens = match c {
            '{' => inside.starts_with('"'),
            '[' => inside.starts_with(['"', '{', '[']),
            _ => false,
        };
        opens.then_some(at)
    })
}

/// What a vendor answered when it refused, as far as the site that
/// classified it could see.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VendorFailure<'a> {
    /// The provider that was asked, as its errors name it
    pub(crate) provider: &'a str,
    /// The HTTP status the vendor sent. `None` when the failure did not
    /// arrive as one: a CLI's exit, a mid-stream error event, an RPC error
    pub(crate) status: Option<u16>,
    /// The `retry-after` header as sent
    pub(crate) retry_after: Option<&'a str>,
    /// The vendor's id for the request
    pub(crate) request_id: Option<&'a str>,
    /// The vendor's own words: a response body, a CLI's stderr, an event
    pub(crate) body: &'a str,
    /// The length of what the vendor sent, in bytes: more than `body` holds
    /// when only part of it is safe to log
    pub(crate) body_bytes: usize,
}

impl<'a> VendorFailure<'a> {
    /// A failure that arrived as text alone, with no HTTP response around it.
    pub(crate) const fn text(provider: &'a str, body: &'a str) -> Self {
        Self {
            provider,
            status: None,
            retry_after: None,
            request_id: None,
            body,
            body_bytes: body.len(),
        }
    }

    /// The same failure with the status the vendor's API answered.
    pub(crate) const fn with_status(mut self, status: Option<u16>) -> Self {
        self.status = status;
        self
    }
}

/// Log what the vendor sent beside the kind it was classified as.
///
/// One line per classified failure, at warn level, with a fixed message and
/// the same fields at every site: `provider`, `status`, `kind`,
/// `retry_after`, `request_id`, `body_bytes` (the original length),
/// `truncated` and `body` (through [`redact_and_cap`]). A field the site
/// does not have is left out of the line.
///
/// `body` is the last field and the only one written unquoted, so a text
/// log is read with everything after `body=` as the body: vendor text that
/// imitates a field (`kind=Auth`) cannot be taken for one.
pub(crate) fn log_classified(failure: &VendorFailure<'_>, error: &RunnerError) {
    let body = redact_and_cap(failure.body);
    warn!(
        provider = failure.provider,
        status = failure.status,
        kind = ?error.kind,
        retry_after = failure.retry_after,
        request_id = failure.request_id,
        body_bytes = failure.body_bytes,
        truncated = body.truncated || body.bytes < failure.body_bytes,
        body = %body.text,
        "LLM vendor error classified"
    );
}

/// [`log_classified`] for a failure that arrived as text alone, handing the
/// error back so a classifier can end on it.
pub(crate) fn classified_text(
    provider: &str,
    vendor_text: &str,
    error: RunnerError,
) -> RunnerError {
    log_classified(&VendorFailure::text(provider, vendor_text), &error);
    error
}

/// [`classified_text`] for a text that did not read as the vendor's
/// protocol and may hold the model's output: only its [`error_members`]
/// reach the line, with the length of the whole.
pub(crate) fn classified_unparsed(
    provider: &str,
    vendor_text: &str,
    error: RunnerError,
) -> RunnerError {
    let said = error_members(vendor_text);
    let mut failure = VendorFailure::text(provider, &said);
    failure.body_bytes = vendor_text.len();
    log_classified(&failure, &error);
    error
}

/// The response headers a classified failure is logged with.
#[cfg(any(feature = "http-api", feature = "quota-http"))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ResponseMeta {
    /// The `retry-after` header as sent
    retry_after: Option<String>,
    /// The first of [`REQUEST_ID_HEADERS`] the response carries
    request_id: Option<String>,
}

#[cfg(any(feature = "http-api", feature = "quota-http"))]
impl ResponseMeta {
    /// Read `retry-after` and the vendor's request id from `headers`,
    /// before the body of their response is consumed.
    ///
    /// A request id is the vendor's own identifier, and losing it would cost
    /// the one field that finds the request in the vendor's logs, so a long
    /// run is kept here. An endpoint or a proxy in front of it can still put
    /// anything in a header: a value passes the rest of the redactor (key
    /// prefixes, `Bearer`, JWTs, addresses), on one line and cut at
    /// [`HEADER_CAP_CHARS`] characters.
    pub(crate) fn read(headers: &HeaderMap) -> Self {
        let value = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(|v| {
                    redact(&one_line(v), usize::MAX)
                        .chars()
                        .take(HEADER_CAP_CHARS)
                        .collect::<String>()
                })
                .filter(|v| !v.is_empty())
        };
        Self {
            retry_after: value("retry-after"),
            request_id: REQUEST_ID_HEADERS.iter().find_map(|name| value(name)),
        }
    }

    /// This response's failure, as [`log_classified`] takes it.
    fn failure<'a>(
        &'a self,
        provider: &'a str,
        status: Option<u16>,
        body: &'a str,
    ) -> VendorFailure<'a> {
        VendorFailure {
            provider,
            status,
            retry_after: self.retry_after.as_deref(),
            request_id: self.request_id.as_deref(),
            body,
            body_bytes: body.len(),
        }
    }

    /// [`log_classified`] for a refused HTTP response, handing the error
    /// back so a call site can wrap its classifier in it.
    pub(crate) fn classified(
        &self,
        provider: &str,
        status: u16,
        body: &str,
        error: RunnerError,
    ) -> RunnerError {
        log_classified(&self.failure(provider, Some(status), body), &error);
        error
    }

    /// [`Self::classified`] for a body or a stream frame that did not read
    /// as the vendor's protocol and may hold the model's output: only its
    /// [`error_members`] reach the line, with the length of the whole. A
    /// frame has no `status` of its own.
    #[cfg(feature = "http-api")]
    pub(crate) fn classified_unparsed(
        &self,
        provider: &str,
        status: Option<u16>,
        body: &str,
        error: RunnerError,
    ) -> RunnerError {
        let said = error_members(body);
        let mut failure = self.failure(provider, status, &said);
        failure.body_bytes = body.len();
        log_classified(&failure, &error);
        error
    }

    /// [`log_classified`] for an error event the vendor sent on this
    /// response's stream: the event as the body, the response's request id,
    /// and no status, since the one on the wire was a success.
    #[cfg(feature = "http-api")]
    pub(crate) fn classified_event(
        &self,
        provider: &str,
        event: &str,
        error: RunnerError,
    ) -> RunnerError {
        log_classified(&self.failure(provider, None, event), &error);
        error
    }
}

#[cfg(test)]
pub(crate) mod capture {
    //! Reads back what a test's code logged.

    use std::io::{self, Write};
    use std::sync::{Arc, Mutex, Once, PoisonError};

    use tracing::subscriber::{set_global_default, with_default};
    use tracing::Level;
    use tracing_subscriber::fmt::MakeWriter;

    /// Installs [`keep_warnings_live`]'s subscriber once per test binary.
    static WARNINGS_LIVE: Once = Once::new();

    /// Give the process a default subscriber that takes warnings and writes
    /// them nowhere.
    ///
    /// A callsite caches whether anyone listens the first time it is hit. A
    /// test on another thread, with no subscriber, can be that first hit:
    /// the callsite is then off for the capturing test too, and its line
    /// never arrives. With a process default that wants warnings, the first
    /// hit finds a listener whichever thread makes it.
    fn keep_warnings_live() {
        WARNINGS_LIVE.call_once(|| {
            let sink = tracing_subscriber::fmt()
                .with_max_level(Level::WARN)
                .with_writer(io::sink)
                .finish();
            // Err means a default is already installed; either one listens.
            let _ = set_global_default(sink);
        });
    }

    /// Collects what a subscriber writes.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl Captured {
        fn text(&self) -> String {
            let bytes = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    impl Write for Captured {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Captured {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// What `log` writes while it runs, as one string.
    pub(crate) fn logged_by(log: impl FnOnce()) -> String {
        keep_warnings_live();
        let captured = Captured::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(captured.clone())
            .finish();
        with_default(subscriber, log);
        captured.text()
    }
}

#[cfg(test)]
mod tests {
    use super::capture::logged_by;
    use super::*;
    use crate::types::ErrorKind;

    fn redacted(raw: &str) -> String {
        redact_and_cap(raw).text
    }

    /// The line itself: one per failure, every field a fixture needs, and
    /// the key the vendor echoed gone from it.
    #[test]
    fn the_line_carries_what_the_vendor_sent_and_the_kind_it_got() {
        const BODY: &str = "{\"error\":{\"type\":\"rate_limit_error\",\n\"message\":\"slow down sk-ant-api03-abcdef\"}}";
        let failure = VendorFailure {
            provider: "acme",
            status: Some(429),
            retry_after: Some("12"),
            request_id: Some("req_011CSHoEeqs5C35K2UUqR7Fy"),
            body: BODY,
            body_bytes: BODY.len(),
        };
        let error = RunnerError::rate_limit("acme", "Rate limit reached");
        let line = logged_by(|| log_classified(&failure, &error));

        assert_eq!(line.lines().count(), 1, "{line}");
        for expected in [
            "WARN",
            "LLM vendor error classified",
            "provider=\"acme\"",
            "status=429",
            "kind=RateLimit",
            "retry_after=\"12\"",
            "request_id=\"req_011CSHoEeqs5C35K2UUqR7Fy\"",
            "rate_limit_error",
            "slow down [REDACTED]",
            &format!("body_bytes={}", failure.body.len()),
            "truncated=false",
        ] {
            assert!(line.contains(expected), "{expected} missing from {line}");
        }
        assert!(!line.contains("sk-ant"), "{line}");
    }

    /// A failure with no HTTP response around it leaves the fields it does
    /// not have out, rather than writing a status nobody sent.
    #[test]
    fn a_failure_without_a_response_logs_no_status() {
        let error = RunnerError::external_service("copilot", "copilot exited with code 1");
        assert_eq!(error.kind, ErrorKind::ExternalService);
        let line = logged_by(|| {
            let returned = classified_text("copilot", "Error: Not authenticated", error.clone());
            assert_eq!(returned.kind, error.kind);
            assert_eq!(returned.message, error.message);
        });
        assert!(line.contains("kind=ExternalService"), "{line}");
        assert!(line.contains("body=Error: Not authenticated"), "{line}");
        for absent in ["status=", "retry_after=", "request_id="] {
            assert!(!line.contains(absent), "{absent} present in {line}");
        }
    }

    /// `body` closes the line, so vendor text shaped like a field cannot be
    /// read as one: everything after the first `body=` is the body.
    #[test]
    fn the_body_is_the_last_field_of_the_line() {
        let forged = "refused kind=Auth status=200 provider=\"evil\"";
        let error = RunnerError::external_service("acme", "refused");
        let line = logged_by(|| {
            let returned = classified_text("acme", forged, error);
            assert_eq!(returned.kind, ErrorKind::ExternalService);
        });
        let (fields, body) = line.trim_end().split_once(" body=").unwrap();
        assert_eq!(body, forged);
        assert!(fields.contains("kind=ExternalService"), "{line}");
        assert!(fields.contains("truncated=false"), "{line}");
        assert!(!fields.contains("evil"), "{line}");
    }

    /// A text that did not read as the vendor's protocol may be the model's
    /// output: of JSON only the error and its type are kept, and plain text
    /// is the vendor talking.
    #[test]
    fn an_unparsed_text_keeps_only_what_names_the_error() {
        // Member order follows serde_json's map, which a feature can change.
        let kept = error_members(
            r#"{"type":"error","error":{"message":"overloaded"},"text":"model words"}"#,
        );
        assert_eq!(
            serde_json::from_str::<Value>(&kept).unwrap(),
            serde_json::json!({"type": "error", "error": {"message": "overloaded"}})
        );
        for unsafe_text in [
            r#"{"choices":[{"message":{"content":"model words"}}]}"#,
            r#"{"type":{"nested":"model words"}}"#,
            r#"{"type":"delta","text":"model wor"#,
            r#"["model words"]"#,
            r#""model words""#,
            "42",
        ] {
            let kept = error_members(unsafe_text);
            assert!(!kept.contains("model"), "{kept}");
        }
        for plain in ["Error: Not authenticated", "[ERROR] not logged in {retry}"] {
            assert_eq!(error_members(plain), plain);
        }
        // Plain text ends where a document opens after it.
        assert_eq!(
            error_members("Warning: update available\n{\"result\":\"model words\"}"),
            "Warning: update available\n"
        );
        assert_eq!(
            error_members(r#"data: {"choices":[{"delta":{"content":"model words"}}]}"#),
            "data: "
        );
        assert_eq!(error_members(r#"note [ "model words" ]"#), "note ");

        let frame = r#"{"choices":[{"message":{"content":"model words"}}]}"#;
        let error = RunnerError::external_service("acme", "Failed to parse response");
        let line = logged_by(|| {
            let returned = classified_unparsed("acme", frame, error);
            assert_eq!(returned.kind, ErrorKind::ExternalService);
        });
        assert!(
            line.contains(&format!("body_bytes={}", frame.len())),
            "{line}"
        );
        assert!(line.contains("truncated=true"), "{line}");
        assert!(line.trim_end().ends_with("body="), "{line}");
    }

    /// The body that started this: a short vendor error must stay readable,
    /// word for word, or the line is no use as a fixture.
    #[test]
    fn an_ordinary_vendor_error_body_comes_through_whole() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}"#;
        let logged = redact_and_cap(body);
        assert_eq!(logged.text, body);
        assert_eq!(logged.bytes, body.len());
        assert!(!logged.truncated);
    }

    #[test]
    fn host_names_versions_and_plain_words_are_not_secrets() {
        for text in [
            "POST https://api.anthropic.com/v1/messages failed",
            "copilot runtime 1.0.86 answered",
            "a risk-free task-list, please retry in 6.406453963s.",
            "rate_limit_error: context_length_exceeded (HTTP 429)",
        ] {
            assert_eq!(redacted(text), text);
        }
    }

    #[test]
    fn a_vendor_key_is_replaced_whatever_surrounds_it() {
        let body = r#"{"error":{"message":"Incorrect API key provided: sk-ant-api03-AbCdEf0123456789xyz. Check key=sk-short or AIzaSyD-9tSrke72PouQMnMX, ghp_16C7e42F292c6912E7, github_pat_11ABCDE, gsk_abc123."}}"#;
        let text = redacted(body);
        for secret in [
            "sk-ant",
            "AbCdEf0123456789xyz",
            "sk-short",
            "AIza",
            "ghp_",
            "github_pat_",
            "gsk_",
        ] {
            assert!(!text.contains(secret), "{secret} survived in {text}");
        }
        assert!(text.contains("Incorrect API key provided: [REDACTED]."));
        assert!(text.contains("key=[REDACTED]"));
    }

    #[test]
    fn a_bearer_token_and_a_jwt_are_replaced() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
        let text = redacted(&format!(
            "sent Authorization: Bearer opaque~token:with.odd$chars, got {jwt} back"
        ));
        assert_eq!(
            text,
            "sent Authorization: Bearer [REDACTED], got [REDACTED] back"
        );
        // The word after `Bearer` goes whatever it is, and a `Bearer` that
        // ends the text has nothing to lose.
        assert_eq!(
            redacted("Bearer abc123 was refused"),
            "Bearer [REDACTED] was refused"
        );
        assert_eq!(redacted("no bearer"), "no bearer");
    }

    #[test]
    fn an_email_address_is_replaced() {
        let text = redacted("no account for Jane.Doe+run@example.co.uk. Ask admin@dravr.ai");
        assert_eq!(text, "no account for [EMAIL]. Ask [EMAIL]");
    }

    /// What the reviewers' probes showed surviving: each input beside what
    /// must be gone from the line.
    #[test]
    fn an_address_is_replaced_whatever_its_local_part_holds() {
        for (raw, expected) in [
            ("no user rené@example.com found", "no user [EMAIL] found"),
            (
                "no account for josé@example.com here",
                "no account for [EMAIL] here",
            ),
            ("user o'brien@example.com", "user [EMAIL]"),
            ("mail a%b@example.com", "mail [EMAIL]"),
            (
                "no account for jane%doe@example.com",
                "no account for [EMAIL]",
            ),
            ("jane%40example.com", "[EMAIL]"),
            ("to jane.doe%40example.com now", "to [EMAIL] now"),
            ("\"jane doe\"@example.com", "[EMAIL]"),
            ("é@é.com", "[EMAIL]"),
            ("seen rené@exämple.com today", "seen [EMAIL] today"),
            (
                "{\"email\":\"'jane@example.com'\"}",
                "{\"email\":\"'[EMAIL]'\"}",
            ),
        ] {
            assert_eq!(redacted(raw), expected, "{raw}");
        }
    }

    #[test]
    fn a_bearer_token_is_replaced_however_it_is_joined() {
        for (raw, expected) in [
            ("Bearer 'abc123secret'", "Bearer '[REDACTED]'"),
            ("bearer \"abc\"", "bearer \"[REDACTED]\""),
            (
                "Authorization: Bearer \"abc123secret\"",
                "Authorization: Bearer \"[REDACTED]\"",
            ),
            ("Bearer: abc123secret", "Bearer: [REDACTED]"),
            ("Bearer=abc123secret", "Bearer=[REDACTED]"),
            ("Bearer%20abc123secret", "Bearer%20[REDACTED]"),
            (
                "authorization=Bearer abc123secret",
                "authorization=Bearer [REDACTED]",
            ),
            (
                "Authorization: Bearer: abc123secret",
                "Authorization: Bearer: [REDACTED]",
            ),
            (
                "Authorization: Basic dXNlcjpwYXNz",
                "Authorization: Basic [REDACTED]",
            ),
            ("sent Basic dXNlcjpwYXNz on", "sent Basic [REDACTED] on"),
        ] {
            assert_eq!(redacted(raw), expected, "{raw}");
        }
        // `Basic` is also a word: only a base64-shaped word after it goes.
        assert_eq!(
            redacted("Basic plan limit reached, basic authentication failed"),
            "Basic plan limit reached, basic authentication failed"
        );
    }

    #[test]
    fn a_jwt_is_replaced_wherever_it_starts_in_a_run() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
        assert_eq!(redacted(&format!("id_token={jwt}")), "id_token=[REDACTED]");
        assert_eq!(
            redacted(&format!("https://x.io/cb?access_token={jwt}&a=1")),
            "https://x.io/cb?access_token=[REDACTED]&a=1"
        );
        assert_eq!(redacted(&format!("at /{jwt} now")), "at /[REDACTED] now");
        // Three long dot-joined parts are a token with or without the header.
        let part = "abcdefghijklmnopqrstuvw";
        assert_eq!(
            redacted(&format!("tok {part}.{part}.{part}")),
            "tok [REDACTED]"
        );
    }

    #[test]
    fn a_key_prefix_counts_after_an_escape_and_takes_the_whole_word() {
        for (raw, secret) in [
            ("key%3Dsk-shortkey123", "shortkey123"),
            ("key%20sk-shortkey123", "shortkey123"),
            ("got ghr_16C7e42F292c6912E7 back", "16C7e42F292c6912E7"),
            ("got AKIAIOSFODNN7EXAMPLE back", "IOSFODNN7EXAMPLE"),
            ("got hf_abcDEF123 back", "abcDEF123"),
            ("got glpat-abcDEF123 back", "abcDEF123"),
            ("got xai-abcDEF123 back", "abcDEF123"),
            ("got xoxb-12345-abcde back", "12345"),
            ("key sk-ant-api03.abcdefghijk tail", "abcdefghijk"),
        ] {
            let text = redacted(raw);
            assert!(!text.contains(secret), "{secret} survived in {text}");
            assert!(text.contains(SECRET_MARKER), "{text}");
        }
        assert_eq!(
            redacted("key sk-ant-api03.abcdefghijk tail"),
            "key [REDACTED] tail"
        );
    }

    #[test]
    fn a_secret_broken_up_by_escaping_is_still_one_run() {
        for raw in [
            r#""QWxhZGRpbjpvcGVu\/IHNlc2FtZQabc\/defghijklmnopqrstuv\/xyz0123456789""#,
            "sig=QWxhZGRpbjpvcGVu%2FIHNlc2FtZQabc%2Bdefghijklmnopqrst%3D",
        ] {
            let text = redacted(raw);
            assert!(!text.contains("QWxh"), "{text}");
            assert!(!text.contains("defghij"), "{text}");
        }
    }

    #[test]
    fn the_value_of_a_secret_named_field_is_replaced_whatever_its_length() {
        for (raw, expected) in [
            (
                r#"{"password":"hunter2","api_key":"abcd1234efgh5678"}"#,
                r#"{"password":"[REDACTED]","api_key":"[REDACTED]"}"#,
            ),
            (
                "x-api-key: 0123456789abcdef0123456",
                "x-api-key: [REDACTED]",
            ),
            (
                "password=hunter2&user=jane",
                "password=[REDACTED]&user=jane",
            ),
            (
                "client_secret = \"a b c\" ok",
                "client_secret = \"[REDACTED]\" ok",
            ),
            (
                r#"{"token":"a\"b","n":1}"#,
                r#"{"token":"[REDACTED]","n":1}"#,
            ),
            ("refresh_token='abc'", "refresh_token='[REDACTED]'"),
        ] {
            assert_eq!(redacted(raw), expected, "{raw}");
        }
    }

    /// What the vendors really send must stay readable: the rules above
    /// name fields and shapes these bodies come close to.
    #[test]
    fn real_vendor_bodies_stay_readable() {
        for body in [
            r#"{"error":{"code":429,"message":"Quota exceeded for generativelanguage.googleapis.com","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED","domain":"googleapis.com"}]}}"#,
            r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#,
            r#"{"error":{"message":"max_tokens is too large: 9000 > 8192","type":"invalid_request_error","param":"max_tokens","code":"invalid_api_key"}}"#,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 205000 tokens > 200000 maximum"}}"#,
            r#"{"usage":{"input_tokens":12,"output_tokens":34},"stop_reason":"max_tokens"}"#,
            "npm package @anthropic-ai/sdk and @mention stay",
        ] {
            assert_eq!(redacted(body), body);
        }
    }

    /// A body far past the cap is not read whole, and the word the bound
    /// falls in is dropped: half a key is too short to recognise.
    #[test]
    fn a_huge_body_is_bounded_before_it_is_read() {
        let filler = "word ".repeat(SCAN_CAP_BYTES / 5 - 2);
        let body = format!("{filler}sk-ant-api03-{}", "k".repeat(400));
        assert!(body.len() > SCAN_CAP_BYTES);
        let (head, cut) = bounded(&body);
        assert!(cut);
        assert!(!head.contains("sk-"), "{}", &head[head.len() - 20..]);
        let logged = redact_and_cap(&body);
        assert!(logged.truncated);
        assert_eq!(logged.bytes, body.len());
        // A bound inside a multibyte character moves back to its start.
        let wide = format!("a{}", "é".repeat(SCAN_CAP_BYTES));
        let (head, cut) = bounded(&wide);
        assert!(cut);
        assert_eq!(head.len(), SCAN_CAP_BYTES - 1);
        assert!(!bounded(&"a ".repeat(SCAN_CAP_BYTES / 2)).1);
    }

    #[test]
    fn a_long_unbroken_run_is_replaced() {
        let base64 = "QWxhZGRpbjpvcGVuIHNlc2FtZQ+/abc012345678==";
        let text = redacted(&format!("payload {base64} rejected"));
        assert_eq!(text, "payload [REDACTED] rejected");
        // 23 characters is still a word; 24 is not.
        assert_eq!(redacted(&"a".repeat(23)), "a".repeat(23));
        assert_eq!(redacted(&"a".repeat(24)), SECRET_MARKER);
    }

    #[test]
    fn newlines_and_control_characters_collapse_to_one_line() {
        let text = redacted("  Error: first line\r\n\tsecond\u{0}line\n\n");
        assert_eq!(text, "Error: first line second line");
    }

    /// The cap counts characters and cuts between them: a byte cut inside a
    /// multibyte character would panic.
    #[test]
    fn the_cap_never_splits_a_multibyte_character() {
        // One leading ASCII byte puts byte 600 inside a character for each
        // width: two, three and four bytes.
        for unit in ["é", "€", "🏃"] {
            let body = format!("a{}", unit.repeat(BODY_CAP_CHARS));
            assert!(!body.is_char_boundary(BODY_CAP_CHARS), "{unit}");
            let logged = redact_and_cap(&body);
            assert!(logged.truncated, "{unit}");
            assert_eq!(logged.text.chars().count(), BODY_CAP_CHARS, "{unit}");
            assert!(body.starts_with(&logged.text), "{unit}");
            assert_eq!(logged.bytes, body.len(), "{unit}");
        }
        let exact = "é".repeat(BODY_CAP_CHARS);
        assert!(!redact_and_cap(&exact).truncated);
    }

    /// The scanner slices at `@`, quotes, escapes and multibyte characters:
    /// every mix of them must come through without a panic.
    #[test]
    fn no_mix_of_separators_and_multibyte_text_panics_the_redactor() {
        const PIECES: [&str; 24] = [
            "@", "%40", "%", "%3D", "%2", "\\/", "\\", "\"", "\\\"", "'", "=", ":", ".", " ", "é",
            "🏃", "Bearer", "Basic", "token", "sk-", "eyJ", "a", "password", "x.y",
        ];
        // A fixed linear congruential sequence: the same texts on every run.
        let mut state: u32 = 0x2545_F491;
        for _ in 0..4000 {
            let mut text = String::new();
            for _ in 0..12 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                text.push_str(PIECES[(state >> 16) as usize % PIECES.len()]);
            }
            let logged = redact_and_cap(&text);
            assert!(logged.text.chars().count() <= BODY_CAP_CHARS, "{text}");
        }
    }

    /// The cap applies to what is logged, so a secret cannot sit half inside
    /// it and come out as a readable prefix.
    #[test]
    fn the_cap_is_applied_after_redaction() {
        let body = format!("{} sk-ant-api03-{}", "word ".repeat(118), "k".repeat(40));
        let logged = redact_and_cap(&body);
        assert!(!logged.text.contains("sk-ant"));
        assert!(logged.text.chars().count() <= BODY_CAP_CHARS);
    }

    /// A header is whatever the endpoint or a proxy before it sent: a key or
    /// a token in one is replaced, and a long request id is still kept.
    #[cfg(any(feature = "http-api", feature = "quota-http"))]
    #[test]
    fn a_secret_in_a_response_header_is_replaced() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", "Bearer abc123secret".parse().unwrap());
        headers.insert(
            "x-debug-trace-id",
            "trace sk-ant-api03-abcdef for jane@example.com"
                .parse()
                .unwrap(),
        );
        let meta = ResponseMeta::read(&headers);
        assert_eq!(meta.retry_after.as_deref(), Some("Bearer [REDACTED]"));
        assert_eq!(
            meta.request_id.as_deref(),
            Some("trace [REDACTED] for [EMAIL]")
        );
        let mut headers = HeaderMap::new();
        let uuid = "0191f0c2-7b1e-7c3a-9d2e-5f6a7b8c9d0e-0191f0c27b1e";
        headers.insert("x-request-id", uuid.parse().unwrap());
        assert_eq!(
            ResponseMeta::read(&headers).request_id.as_deref(),
            Some(uuid)
        );
    }

    #[cfg(any(feature = "http-api", feature = "quota-http"))]
    #[test]
    fn response_headers_are_read_as_sent() {
        let mut headers = HeaderMap::new();
        assert_eq!(ResponseMeta::read(&headers), ResponseMeta::default());
        headers.insert("retry-after", "12".parse().unwrap());
        headers.insert("x-request-id", "x-second".parse().unwrap());
        headers.insert(
            "request-id",
            "req_011CSHoEeqs5C35K2UUqR7Fy".parse().unwrap(),
        );
        let meta = ResponseMeta::read(&headers);
        assert_eq!(meta.retry_after.as_deref(), Some("12"));
        // The vendor's id is kept whole: it is what finds the request again.
        assert_eq!(
            meta.request_id.as_deref(),
            Some("req_011CSHoEeqs5C35K2UUqR7Fy")
        );
    }
}
