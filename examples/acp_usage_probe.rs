// ABOUTME: Prints the RAW ACP usage object Copilot returns, before embacle parses it
// ABOUTME: Answers empirically whether cachedReadTokens/thoughtTokens are populated or merely permitted
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run --example acp_usage_probe --features copilot-headless
//
// WHY THIS EXISTS
//
// The ACP schema (agent-client-protocol-schema, `Usage`) defines cachedReadTokens,
// cachedWriteTokens and thoughtTokens alongside inputTokens/outputTokens/totalTokens.
// All three are Option on an explicitly UNSTABLE capability, so the schema says only
// that an agent MAY send them. `extract_usage` reads three fields and drops the rest,
// which means no consumer can tell "the agent sent nothing" from "we never looked".
//
// That ambiguity is not resolvable by reading code. It cost a downstream project a
// registered limitation asserting the transport "carries no cache-read count" — a claim
// about the wire made without reading the wire. This probe reads the wire.
//
// NOTE: like every ACP call through this runner, spawning pins the routing model in
// ~/.copilot/settings.json (see `ensure_copilot_settings_model`).

use std::io::stderr;
use std::time::Duration;

use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::CopilotHeadlessRunner;
use tokio::time::timeout;
use tracing::Level;
use tracing_subscriber::fmt;

#[tokio::main]
async fn main() {
    // DEBUG so the `ACP usage payload` line in extract_usage reaches stderr; that
    // line, not the return value, is the point of this example.
    fmt()
        .with_max_level(Level::DEBUG)
        .with_target(true)
        .with_writer(stderr)
        .init();

    println!("=== ACP usage probe: what does Copilot actually report? ===");

    let runner = CopilotHeadlessRunner::from_env();

    // Deliberately trivial: the answer is in the usage object, not the content. A
    // short prompt also keeps the turn cheap and fast.
    let request = ChatRequest {
        messages: vec![ChatMessage::user("Reply with the single word: pong.")],
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
    };

    // TWO calls, deliberately. The first is cold and can only WRITE cache; the
    // read count is necessarily 0 there, which on its own would look exactly like
    // "this agent does not report cache reads". The second reuses the warm
    // subprocess and the same prefix, so a non-zero cachedReadTokens on turn 2 is
    // the proof that caching is live and simply invisible downstream.
    for turn in 1..=2 {
        println!("\n=== turn {turn} ===");
        match timeout(Duration::from_mins(2), runner.complete(&request)).await {
            Ok(Ok(resp)) => {
                println!("model:   {:?}", resp.model);
                println!("content: {}", resp.content.trim());
                println!("usage:   {:?}   <- all embacle keeps", resp.usage);
            }
            Ok(Err(e)) => println!("complete() error: {e}"),
            Err(_) => println!("timed out after 120s"),
        }
    }

    println!(
        "\nThe raw objects are on stderr above, logged as `ACP usage payload`.\n\
         Compare cachedReadTokens across the two turns."
    );
}
