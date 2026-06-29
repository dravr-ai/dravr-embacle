// ABOUTME: Live smoke test proving the copilot --acp headless runner routes to
// ABOUTME: the configured model by pinning it in ~/.copilot/settings.json.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: COPILOT_HEADLESS_MODEL=claude-sonnet-4.6 cargo run --example acp_model_smoke --features copilot-headless
// NOTE: overwrites ~/.copilot/settings.json "model" — back it up first.
//
// Expect the agent to self-identify as the requested model, e.g.
//   "I'm powered by Claude Sonnet 4.6 (model ID: claude-sonnet-4.6)."
// Before the fix this self-identified as the account default (GPT/Gemini)
// because copilot --acp ignores both the session/new model field and --model.

use std::env;
use std::time::Duration;

use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::CopilotHeadlessRunner;
use tokio::time::timeout;
use tracing::Level;
use tracing_subscriber::fmt;

#[tokio::main]
async fn main() {
    fmt().with_max_level(Level::INFO).with_target(false).init();

    let model = env::var("COPILOT_HEADLESS_MODEL").unwrap_or_default();
    println!("=== ACP model smoke: requesting model = {model:?} ===");

    let runner = CopilotHeadlessRunner::from_env();

    let request = ChatRequest {
        messages: vec![ChatMessage::user(
            "In one short line, state your exact model name. Then on a new line write PONG.",
        )],
        model: None,
        temperature: Some(0.0),
        max_tokens: Some(64),
        stream: false,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: Vec::new(),
    };

    match timeout(Duration::from_secs(90), runner.complete(&request)).await {
        Ok(Ok(resp)) => {
            println!("\n--- response.content ---\n{}", resp.content);
            println!("--- response.model: {:?} ---", resp.model);
        }
        Ok(Err(e)) => println!("complete() error: {e}"),
        Err(_) => println!("timed out after 90s"),
    }
}
