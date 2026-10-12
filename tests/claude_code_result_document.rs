// ABOUTME: Drives the Claude Code runner against a shell script standing in for `claude`, proving
// ABOUTME: the result document is read before the exit code and the exit code speaks only otherwise
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Measured on the dravr-platform image on 2026-09-21: the CLI exits 1 with
//! the failure's reason in `result`, the last field of a ~1.8 KB document.
//! Judging the exit code first surfaced the first 500 chars (cost and token
//! counts) and never classified the failure.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, LlmProvider};
use embacle::{ClaudeCodeRunner, MetricsProvider};

// Each test execs a script it just wrote; see `support` for the ETXTBSY race
// that makes every call here retry its spawn.
mod support;
use support::retry_text_file_busy;

/// A `claude` stand-in: prints the given stdout and exits with the given code.
fn fake_claude(dir: &Path, stdout: &str, exit_code: i32) -> PathBuf {
    let path = dir.join("claude");
    let script = format!("#!/bin/sh\ncat <<'CLAUDE_EOF'\n{stdout}\nCLAUDE_EOF\nexit {exit_code}\n");
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn ping() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("ping")])
}

#[tokio::test]
async fn an_error_document_outranks_the_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let padding = "x".repeat(600);
    let doc = format!(
        r#"{{"duration_api_ms":6681,"session_id":"{padding}","usage":{{"input_tokens":8,"output_tokens":4}},"is_error":true,"subtype":"error_during_execution","api_error_status":429,"result":"Claude usage limit reached"}}"#
    );
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_claude(dir.path(), &doc, 1)));

    let request = ping();
    let err = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap_err();

    assert_eq!(err.kind, ErrorKind::RateLimit, "{}", err.message);
    assert!(
        err.message.contains("usage limit reached"),
        "{}",
        err.message
    );
    assert!(
        err.message.contains("api_error_status: 429"),
        "{}",
        err.message
    );
    assert!(!err.message.contains("duration_api_ms"), "{}", err.message);
}

#[tokio::test]
async fn a_non_document_on_a_failed_exit_keeps_the_exit_diagnostic() {
    let dir = tempfile::tempdir().unwrap();
    let binary = fake_claude(dir.path(), "Not logged in · Please run /login", 1);
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(binary));

    let request = ping();
    let err = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap_err();

    assert_eq!(err.kind, ErrorKind::ExternalService);
    assert!(
        err.message.contains("exited with code 1"),
        "{}",
        err.message
    );
    assert!(err.message.contains("Not logged in"), "{}", err.message);
}

#[tokio::test]
async fn a_success_document_on_a_failed_exit_is_still_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let binary = fake_claude(dir.path(), r#"{"is_error":false,"result":"pong"}"#, 3);
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(binary));

    let request = ping();
    let err = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap_err();

    assert!(
        err.message.contains("exited with code 3"),
        "{}",
        err.message
    );
}

/// A success document with the `modelUsage` block CLI 2.1.296 prints
/// (measured 2026-10-12 with `--model haiku`).
const HAIKU_5_5_TURN: &str = r#"{"type":"result","subtype":"success","is_error":false,"result":"ok","usage":{"input_tokens":200000,"output_tokens":100000},"modelUsage":{"claude-haiku-5-5":{"inputTokens":200000,"outputTokens":100000,"costUSD":0.07}}}"#;

#[tokio::test]
async fn a_claude_code_turn_is_priced_on_the_model_that_served_it() {
    let dir = tempfile::tempdir().unwrap();
    let binary = fake_claude(dir.path(), HAIKU_5_5_TURN, 0);
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(binary).with_model("haiku"));
    let metered = MetricsProvider::new(Box::new(runner));

    let request = ping();
    let response = retry_text_file_busy(|| metered.complete(&request))
        .await
        .unwrap();
    let report = metered.report().unwrap();

    assert_eq!(response.model, "claude-haiku-5-5");
    // Haiku 5.5's base card, $0.10/$0.50 per MTok: 0.2 * 0.10 + 0.1 * 0.50.
    assert!(
        (report.total_cost - 0.07).abs() < 1e-9,
        "a Claude Code turn must bill at Haiku 5.5's rate, not $0; got {}",
        report.total_cost
    );
}

#[tokio::test]
async fn without_model_usage_the_turn_names_the_model_sent() {
    let dir = tempfile::tempdir().unwrap();
    let binary = fake_claude(dir.path(), r#"{"is_error":false,"result":"pong"}"#, 0);
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(binary).with_model("claude-sonnet-5-5"));

    let request = ping();
    let response = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap();

    assert_eq!(response.model, "claude-sonnet-5-5");
}
