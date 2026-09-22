// ABOUTME: Live proof against the real Copilot CLI that a turn the CLI answers itself is an error, not a reply
// ABOUTME: Gated by EMBACLE_E2E_COPILOT_CLI_NOTICE=1; needs the copilot binary and neither a model nor an account
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! The Copilot CLI's ACP adapter writes the runtime's errors into the reply
//! stream as plain text and still ends the turn normally. This drives the real
//! CLI into one such error deterministically: with a provider prompt budget far
//! below its own static context, it refuses before calling any model — so the
//! test spends nothing and needs no credential — and answers "Warning: Static
//! system messages and tool definitions exceed the model's usable context
//! budget … Error: No response was returned."
//!
//! That text once reached athletes as the coach's reply. Its own binary because
//! the budget is process environment the CLI inherits; the lane sets it:
//!
//! ```bash
//! EMBACLE_E2E_COPILOT_CLI_NOTICE=1 COPILOT_OFFLINE=true \
//!   COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:9/v1 COPILOT_PROVIDER_MAX_PROMPT_TOKENS=3000 \
//!   COPILOT_HEADLESS_MODEL=any-model cargo test --features copilot-headless --test e2e_cli_notice
//! ```

#![cfg(feature = "copilot-headless")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;

use embacle::types::{ChatMessage, ChatRequest, ErrorKind, LlmProvider, RunnerError};
use embacle::CopilotHeadlessRunner;

fn enabled() -> bool {
    if env::var("EMBACLE_E2E_COPILOT_CLI_NOTICE").as_deref() == Ok("1") {
        return true;
    }
    eprintln!(
        "SKIP e2e_cli_notice (set EMBACLE_E2E_COPILOT_CLI_NOTICE=1 with \
         COPILOT_PROVIDER_MAX_PROMPT_TOKENS far below the CLI's static context)"
    );
    false
}

fn assert_is_the_refusal(err: &RunnerError, call: &str) {
    let text = err.to_string();
    assert_eq!(err.kind, ErrorKind::ExternalService, "{call}: {text}");
    assert!(
        err.kind.is_provider_fault(),
        "{call}: a fallback chain asks its next tier"
    );
    assert!(
        text.contains("No response was returned"),
        "{call}: carries the CLI's error: {text}"
    );
    assert!(
        text.contains("usable context budget"),
        "{call}: the refusal path, not some other failure: {text}"
    );
}

#[tokio::test]
async fn a_turn_the_cli_answers_itself_is_an_error_the_chain_can_act_on() {
    if !enabled() {
        return;
    }
    let runner = CopilotHeadlessRunner::from_env();
    let request = ChatRequest::new(vec![ChatMessage::user("Reply with: pong")]);

    let converse = runner
        .converse(&request)
        .await
        .expect_err("converse(): the CLI answered, not a model");
    assert_is_the_refusal(&converse, "converse");

    let complete = runner
        .complete(&request)
        .await
        .expect_err("complete(): the CLI answered, not a model");
    assert_is_the_refusal(&complete, "complete");
}
