// ABOUTME: Drives each CLI runner against a shell stand-in that records its argv, proving the
// ABOUTME: model a request names reaches the CLI's model flag over the configured one
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! One runner instance serves every caller in a process, so the model has to
//! travel with the request: a request's model wins over `RunnerConfig::model`,
//! which wins over the runner default. Callers such as embacle-server set the
//! model only on the request, so this precedence is how they choose one.

#![cfg(unix)]
#![allow(clippy::unwrap_used)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::{
    ClaudeCodeRunner, ClineCliRunner, CodexCliRunner, ContinueCliRunner, CursorAgentRunner,
    GeminiCliRunner, KiloCliRunner, KiroCliRunner, OpenCodeRunner, WarpCliRunner,
};
use tokio_stream::StreamExt;

use crate::support::retry_text_file_busy;

/// The model every runner is configured with, which a request must override.
const CONFIGURED: &str = "configured-model";

/// The model every request names.
const REQUESTED: &str = "haiku";

/// A CLI stand-in that writes its argv, one argument per line, to
/// `argv.log` beside it, then prints `stdout`.
fn fake_cli(dir: &Path, stdout: &str) -> PathBuf {
    let path = dir.join("fake-cli");
    let script = format!(
        "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '{}'\ncat <<'FAKE_EOF'\n{stdout}\nFAKE_EOF\n",
        dir.join("argv.log").display()
    );
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// The value the recorded argv passed after `flag`.
fn flag_value(dir: &Path, flag: &str) -> Option<String> {
    let log = fs::read_to_string(dir.join("argv.log")).unwrap();
    let argv: Vec<&str> = log.lines().collect();
    argv.iter()
        .position(|a| *a == flag)
        .and_then(|i| argv.get(i + 1))
        .map(|v| (*v).to_owned())
}

fn configured(dir: &Path, stdout: &str) -> RunnerConfig {
    RunnerConfig::new(fake_cli(dir, stdout)).with_model(CONFIGURED)
}

fn request(model: Option<&str>) -> ChatRequest {
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);
    match model {
        Some(model) => request.with_model(model),
        None => request,
    }
}

/// Run one `complete()`; the stand-in's output need not parse, only its argv
/// is read.
async fn run(runner: &dyn LlmProvider, model: Option<&str>) {
    let request = request(model);
    drop(retry_text_file_busy(|| runner.complete(&request)).await);
}

const CLAUDE_DOC: &str = r#"{"result":"pong","is_error":false}"#;

#[tokio::test]
async fn claude_code_runs_the_requested_model_and_names_it() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(configured(dir.path(), CLAUDE_DOC));

    let request = request(Some(REQUESTED));
    let response = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap();

    assert_eq!(
        flag_value(dir.path(), "--model").as_deref(),
        Some(REQUESTED),
        "bug: claude-code ignored the requested model"
    );
    // No `modelUsage` in the document: the response names what was sent.
    assert_eq!(
        response.model, REQUESTED,
        "bug: claude-code named a model other than the one it sent"
    );
}

#[tokio::test]
async fn claude_code_stream_runs_the_requested_model() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(configured(dir.path(), CLAUDE_DOC));

    let request = request(Some(REQUESTED));
    let mut stream = retry_text_file_busy(|| runner.complete_stream(&request))
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    assert_eq!(
        flag_value(dir.path(), "--model").as_deref(),
        Some(REQUESTED),
        "bug: claude-code streaming ignored the requested model"
    );
}

#[tokio::test]
async fn claude_code_without_a_requested_model_runs_the_configured_one() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(configured(dir.path(), CLAUDE_DOC));

    run(&runner, None).await;

    assert_eq!(
        flag_value(dir.path(), "--model").as_deref(),
        Some(CONFIGURED),
        "bug: claude-code dropped the configured model"
    );
}

#[tokio::test]
async fn claude_code_without_any_model_runs_its_default() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), CLAUDE_DOC)));

    run(&runner, None).await;

    assert_eq!(
        flag_value(dir.path(), "--model").as_deref(),
        Some("opus"),
        "bug: claude-code did not fall back to its default model"
    );
}

/// Builds one runner from its config.
type BuildRunner = fn(RunnerConfig) -> Box<dyn LlmProvider>;

/// Every other CLI runner that pins a model, with the flag it pins it by and
/// what it pins when neither the request nor the config names one: its own
/// default, or nothing for the runners that leave the choice to the CLI.
#[tokio::test]
async fn every_cli_runner_passes_the_requested_model() {
    let cases: Vec<(&str, &str, Option<&str>, BuildRunner)> = vec![
        ("codex", "-m", Some("o4-mini"), |c| {
            Box::new(CodexCliRunner::new(c))
        }),
        ("cursor-agent", "--model", Some("sonnet-4"), |c| {
            Box::new(CursorAgentRunner::new(c))
        }),
        ("gemini", "-m", Some("gemini-2.5-flash"), |c| {
            Box::new(GeminiCliRunner::new(c))
        }),
        (
            "kilo",
            "--model",
            Some("anthropic/claude-sonnet-4-6"),
            |c| Box::new(KiloCliRunner::new(c)),
        ),
        (
            "opencode",
            "--model",
            Some("github-copilot/claude-sonnet-4.6"),
            |c| Box::new(OpenCodeRunner::new(c)),
        ),
        ("warp", "--model", Some("auto"), |c| {
            Box::new(WarpCliRunner::new(c))
        }),
        ("cline", "-m", None, |c| Box::new(ClineCliRunner::new(c))),
        ("continue", "--model", None, |c| {
            Box::new(ContinueCliRunner::new(c))
        }),
        ("kiro", "--model", None, |c| Box::new(KiroCliRunner::new(c))),
    ];

    for (name, flag, unnamed, build) in cases {
        let dir = tempfile::tempdir().unwrap();
        let runner = build(configured(dir.path(), "{}"));
        run(runner.as_ref(), Some(REQUESTED)).await;
        assert_eq!(
            flag_value(dir.path(), flag).as_deref(),
            Some(REQUESTED),
            "bug: {name} ignored the requested model"
        );

        let dir = tempfile::tempdir().unwrap();
        let runner = build(configured(dir.path(), "{}"));
        run(runner.as_ref(), None).await;
        assert_eq!(
            flag_value(dir.path(), flag).as_deref(),
            Some(CONFIGURED),
            "bug: {name} dropped the configured model"
        );

        let dir = tempfile::tempdir().unwrap();
        let runner = build(RunnerConfig::new(fake_cli(dir.path(), "{}")));
        run(runner.as_ref(), None).await;
        assert_eq!(
            flag_value(dir.path(), flag).as_deref(),
            unnamed,
            "bug: {name} pinned the wrong model when none was named"
        );
    }
}
