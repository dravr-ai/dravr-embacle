// ABOUTME: Measures whether Copilot's implicit prefix cache follows OUR stable prefix or a fixed boundary
// ABOUTME: Decides whether any prompt-ordering work can move cachedReadTokens at all

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run --example acp_cache_boundary_probe --features copilot-headless
//
// WHY THIS EXISTS
//
// Copilot caches prompt prefixes, and we can measure it: `cachedReadTokens` comes
// back on every turn. What we CANNOT do is ask for it. The ACP schema defines the
// two cache counts on `Usage` and nothing else — no cache_control, no breakpoint,
// no ephemeral marker anywhere in the protocol — and Copilot CLI 1.0.81 advertises
// `loadSession`, `mcpCapabilities`, `promptCapabilities` and
// `sessionCapabilities{close,list}`, with no caching capability at all. The
// boundary is picked entirely by the vendor.
//
// That leaves one question worth money: does the vendor pick it by following the
// stable head of what we send, or does it sit at a fixed structural offset we
// cannot influence? An audit proposed reordering ~12 KB of prompt text to lengthen
// the cached prefix. If the boundary is fixed, that work buys exactly nothing, and
// no amount of reading code can tell the difference — production shows a cached
// constant of 12,709 tokens repeated byte-identically across two conversations and
// two users, which is equally consistent with both explanations.
//
// THE ARMS
//
//   floor  — a tiny prefix. Whatever caches here is vendor preamble, not ours, and
//            is the baseline every other arm must be read against. Without it,
//            "20k prefix cached 12k" cannot be told from "12k of vendor preamble
//            cached and none of ours".
//   s10 /
//   s20 /
//   s40   — identical prefixes of growing size, sent twice each. If cached reads
//            climb with prefix size, the boundary follows us and prefix work pays.
//            If all three land on the same number, it is fixed and it does not.
//   vary  — s20's prefix, with one word changed near the FRONT on the second turn.
//            Isolates whether an early-varying byte truncates everything after it,
//            which is the specific mechanism prompt-ordering work would exploit.
//
// Each arm sends the same prefix twice: turn 1 can only WRITE cache, so the read
// count there is necessarily 0. Turn 2 is the measurement.
//
// WHAT IT MEASURED (CLI 1.0.81, claude-sonnet-5, 2026-08-29)
//
//   arm     prefix~tok     t1_write      t2_read     t2_write
//   floor           32        28180        13964        14214
//   s10          10000        26274        13964        26276
//   s20          20000        38375        13964        38371
//   s40          40000        62568        13964        62564
//   vary         20000        38374        13964        38377
//
// The read is a CONSTANT. A 32-token prefix and a 40,000-token prefix are served
// exactly 13,964 cached tokens each — identical to the token, across a 1,250x
// change in what we send. The boundary does not follow our prefix; the cached
// region is Copilot's own preamble, and our text is never in it. `vary` confirms
// it from the other side: perturbing our head changes nothing, because our head
// was never being read back.
//
// Two consequences worth keeping:
//
//   1. No ordering of our prompt can move `cachedReadTokens`. A proposal to hoist
//      stable blocks earlier to lengthen the cached prefix cannot pay, and this
//      table is why — the lever does not exist, rather than being small.
//   2. `t2_write` tracks our prompt size almost exactly (14,214 / 26,276 / 38,371
//      / 62,564 against 32 / 10k / 20k / 40k prefixes). We pay the cache-WRITE
//      premium on the whole prompt every turn and are served none of it back, so
//      prompt SIZE is the only thing on our side of the boundary that costs money.
//
// Re-run this before trusting any future claim that prompt layout affects caching;
// a vendor that starts honouring our prefix would show s40 > s20 > floor.

use std::io::stderr;
use std::time::Duration;

use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::CopilotHeadlessRunner;
use tokio::time::timeout;
use tracing::Level;
use tracing_subscriber::fmt;

/// Roughly four characters per token, which is close enough: the arms only need to
/// differ from each other by a lot, not to hit an exact token count.
const CHARS_PER_TOKEN: usize = 4;

/// Deterministic filler shaped like the coaching instructions this prompt actually
/// carries, so the probe measures a realistic payload rather than a pathological one.
fn prefix_of(approx_tokens: usize, mutate: bool) -> String {
    let unit = "The coach grounds every plan in the athlete's own recorded sessions, \
                citing each by name, date and one measured field. ";
    let mut s = String::with_capacity(approx_tokens * CHARS_PER_TOKEN + unit.len());
    while s.len() < approx_tokens * CHARS_PER_TOKEN {
        s.push_str(unit);
    }
    if mutate {
        // One word, deliberately near the front — around 2,000 characters in, well
        // inside any plausible cached head. Everything after it is untouched, so a
        // drop in cached reads can only be the divergence truncating the prefix.
        let at = 2_000.min(s.len().saturating_sub(unit.len()));
        s.replace_range(at..at + 5, "XXXXX");
    }
    s
}

fn request(prefix: &str) -> ChatRequest {
    ChatRequest {
        messages: vec![
            ChatMessage::system(prefix),
            ChatMessage::user("Reply with the single word: pong."),
        ],
        model: None,
        temperature: Some(0.0),
        max_tokens: Some(16),
        stream: false,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: Vec::new(),
    }
}

#[tokio::main]
async fn main() {
    fmt()
        .with_max_level(Level::DEBUG)
        .with_target(true)
        .with_writer(stderr)
        .init();

    let runner = CopilotHeadlessRunner::from_env();

    // (label, approx prefix tokens, whether turn 2 diverges near the front)
    let arms: [(&str, usize, bool); 5] = [
        ("floor", 32, false),
        ("s10", 10_000, false),
        ("s20", 20_000, false),
        ("s40", 40_000, false),
        ("vary", 20_000, true),
    ];

    println!("=== ACP cache boundary probe ===");
    println!("Does the cached prefix follow OUR stable head, or sit at a fixed vendor offset?\n");
    println!(
        "{:<7} {:>10} {:>12} {:>12} {:>12}",
        "arm", "prefix~tok", "t1_write", "t2_read", "t2_write"
    );

    let mut results: Vec<(String, usize, u64)> = Vec::new();

    for (label, tokens, mutate) in arms {
        let first = prefix_of(tokens, false);
        // Only the SECOND turn diverges; the first must be the common prefix or
        // there is nothing for turn 2 to have hit.
        let second = prefix_of(tokens, mutate);

        let mut t1_write = 0_u64;
        let mut t2_read = 0_u64;
        let mut t2_write = 0_u64;

        for (turn, body) in [(1_u8, &first), (2_u8, &second)] {
            match timeout(Duration::from_mins(3), runner.complete(&request(body))).await {
                Ok(Ok(resp)) => {
                    let u = resp.usage;
                    let read = u.as_ref().and_then(|u| u.cached_read_tokens).unwrap_or(0);
                    let write = u.as_ref().and_then(|u| u.cached_write_tokens).unwrap_or(0);
                    if turn == 1 {
                        t1_write = u64::from(write);
                    } else {
                        t2_read = u64::from(read);
                        t2_write = u64::from(write);
                    }
                }
                Ok(Err(e)) => println!("  {label} turn {turn}: error {e}"),
                Err(_) => println!("  {label} turn {turn}: timed out"),
            }
        }

        println!("{label:<7} {tokens:>10} {t1_write:>12} {t2_read:>12} {t2_write:>12}");
        results.push((label.to_owned(), tokens, t2_read));
    }

    println!("\n--- reading the result ---");
    let floor = results.first().map_or(0, |r| r.2);
    let s20 = results.iter().find(|r| r.0 == "s20").map_or(0, |r| r.2);
    let s40 = results.iter().find(|r| r.0 == "s40").map_or(0, |r| r.2);
    let vary = results.iter().find(|r| r.0 == "vary").map_or(0, |r| r.2);

    println!("vendor floor (tiny prefix):        {floor}");
    println!(
        "s20 above floor:                   {}",
        s20.saturating_sub(floor)
    );
    println!(
        "s40 above floor:                   {}",
        s40.saturating_sub(floor)
    );
    println!(
        "vary above floor:                  {}",
        vary.saturating_sub(floor)
    );
    println!(
        "\nIf s40 > s20 > floor, the boundary FOLLOWS our prefix and prompt-ordering work pays.\n\
         If s10 == s20 == s40, it is FIXED and no reordering can move it.\n\
         If vary << s20, an early-varying byte truncates the cached head — which is the\n\
         one lever available to us, since the protocol offers no cache_control at all."
    );
}
