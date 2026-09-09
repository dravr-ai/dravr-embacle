// ABOUTME: Drives the real router across real claude and copilot binaries with real tokens
// ABOUTME: Inert unless EMBACLE_E2E_ROUTER=1 and both credentials are actually present
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Router end-to-end.
//!
//! Everything here needs two live subscriptions and two installed binaries, so
//! it is off by default and stays off on CI and on any machine that does not
//! have both. The gate is deliberately two-part: the opt-in says a human meant
//! to run this, and the credential check means an opt-in on a machine without
//! tokens skips rather than fails. A test that goes red for want of a secret
//! teaches everyone to ignore it.
//!
//! What the offline suite cannot prove, and this can: that the two runners
//! really are interchangeable behind one provider, that a real quota reading
//! moves the decision, and that `name()` names whoever actually answered.

#![cfg(feature = "quota-http")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::env;
use std::time::{Duration, UNIX_EPOCH};

use embacle::config::CliRunnerType;
use embacle::discovery::resolve_binary;
use embacle::quota::LimitChecker;
use embacle::quota_http::AnthropicUsageChecker;
use embacle::router::{Backend, PreferInOrder, RouterProvider};
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::{ClaudeCodeRunner, RunnerConfig};
use tokio::time::timeout;

/// Opt-in, as the rest of the E2E suite does it.
fn opted_in() -> bool {
    env::var("EMBACLE_E2E_ALL").as_deref() == Ok("1")
        || env::var("EMBACLE_E2E_ROUTER").as_deref() == Ok("1")
}

/// The Claude OAuth token, if this machine has one.
fn claude_token() -> Option<String> {
    env::var("CLAUDE_CODE_OAUTH_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// Skip unless a human asked for this AND the machine can actually do it.
macro_rules! require_e2e {
    () => {
        if !opted_in() {
            eprintln!("skipping: set EMBACLE_E2E_ROUTER=1 to run the router E2E suite");
            return;
        }
        if claude_token().is_none() {
            eprintln!("skipping: CLAUDE_CODE_OAUTH_TOKEN is not set on this machine");
            return;
        }
    };
}

fn claude_backend() -> Option<Backend> {
    let path = resolve_binary(
        CliRunnerType::ClaudeCode.binary_name(),
        env::var(CliRunnerType::ClaudeCode.env_override_key())
            .ok()
            .as_deref(),
    )
    .ok()?;
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(path).with_model("sonnet"));
    let checker = AnthropicUsageChecker::new(claude_token()?);
    Some(Backend::metered(Box::new(runner), Box::new(checker)))
}

fn ping() -> ChatRequest {
    ChatRequest::new(vec![
        ChatMessage::system("You are a test bot. Follow instructions exactly."),
        ChatMessage::user("Respond with exactly: PONG. Nothing else."),
    ])
    .with_max_tokens(20)
}

#[tokio::test]
async fn the_real_usage_endpoint_answers_with_windows() {
    require_e2e!();
    let checker = AnthropicUsageChecker::new(claude_token().unwrap());

    let snapshots = checker.check().await.expect("usage endpoint should answer");
    assert!(
        !snapshots.is_empty(),
        "a live subscription reports at least a session window; an empty list means the \
         response shape changed and the parser is now reading nothing as headroom"
    );
    for s in &snapshots {
        assert!(
            (0.0..=100.0).contains(&s.percent),
            "{} reported {}%, which is off the 0-100 scale the endpoint documents — the \
             headers use 0-1 and confusing the two turns 77% into 0.77%",
            s.key,
            s.percent
        );
        assert!(
            s.resets_at > UNIX_EPOCH,
            "{} carried no usable reset instant; a window with no reset cannot be returned to",
            s.key
        );
    }
}

#[tokio::test]
async fn a_router_over_one_real_backend_answers() {
    require_e2e!();
    let Some(backend) = claude_backend() else {
        eprintln!("skipping: claude binary not found");
        return;
    };

    let router = RouterProvider::new(vec![backend], Box::new(PreferInOrder))
        .unwrap()
        // Never step aside: this asserts the plumbing, not the policy.
        .with_threshold(1_000.0);

    let response = timeout(Duration::from_mins(5), router.complete(&ping()))
        .await
        .expect("router turn timed out")
        .expect("router turn failed");

    assert!(
        response.content.to_uppercase().contains("PONG"),
        "expected PONG through the router, got {:?}",
        response.content
    );
    assert_eq!(
        router.name(),
        "claude-code",
        "name() must report the backend that actually answered, not a fixed label"
    );
}

#[tokio::test]
async fn a_zero_threshold_steps_aside_to_the_second_backend() {
    require_e2e!();
    let Some(primary) = claude_backend() else {
        eprintln!("skipping: claude binary not found");
        return;
    };
    let Ok(copilot_path) = resolve_binary(
        CliRunnerType::Copilot.binary_name(),
        env::var(CliRunnerType::Copilot.env_override_key())
            .ok()
            .as_deref(),
    ) else {
        eprintln!("skipping: copilot binary not found");
        return;
    };
    let secondary = Backend::unmetered(Box::new(embacle::CopilotRunner::new(RunnerConfig::new(
        copilot_path,
    ))));

    // Threshold 0 means "any consumption at all is too much", so a live
    // account with any usage steps aside. This exercises the real decision
    // against a real reading rather than a scripted one.
    let router = RouterProvider::new(vec![primary, secondary], Box::new(PreferInOrder))
        .unwrap()
        .with_threshold(0.0);

    let response = timeout(Duration::from_mins(5), router.complete(&ping()))
        .await
        .expect("router turn timed out")
        .expect("router turn failed");

    assert!(
        response.content.to_uppercase().contains("PONG"),
        "expected PONG from the secondary, got {:?}",
        response.content
    );
    assert_eq!(
        router.name(),
        "copilot",
        "with no headroom on the primary the turn must be served by, and attributed to, \
         the secondary"
    );
}
