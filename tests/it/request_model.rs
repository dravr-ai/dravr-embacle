// ABOUTME: Drives each CLI runner against a shell stand-in that records its argv, proving the
// ABOUTME: model a request names reaches the CLI's model flag over the configured one
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! One runner instance serves every caller in a process, so the model has to
//! travel with the request. Until 0.40.0 the CLI runners built their model
//! flag from `RunnerConfig::model` alone: a request naming `haiku` logged
//! `haiku` and ran Claude Code on `opus`, and embacle-server, which sets the
//! model only on the request, could not choose one at all.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
        Some(REQUESTED)
    );
    // No `modelUsage` in the document: the response names what was sent.
    assert_eq!(response.model, REQUESTED);
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
        Some(REQUESTED)
    );
}

#[tokio::test]
async fn claude_code_without_a_requested_model_runs_the_configured_one() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(configured(dir.path(), CLAUDE_DOC));

    run(&runner, None).await;

    assert_eq!(
        flag_value(dir.path(), "--model").as_deref(),
        Some(CONFIGURED)
    );
}

#[tokio::test]
async fn claude_code_without_any_model_runs_its_default() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), CLAUDE_DOC)));

    run(&runner, None).await;

    assert_eq!(flag_value(dir.path(), "--model").as_deref(), Some("opus"));
}

/// Builds one runner from its config.
type BuildRunner = fn(RunnerConfig) -> Box<dyn LlmProvider>;

/// Every other CLI runner that pins a model, with the flag it pins it by.
#[tokio::test]
async fn every_cli_runner_passes_the_requested_model() {
    let cases: Vec<(&str, &str, BuildRunner)> = vec![
        ("codex", "-m", |c| Box::new(CodexCliRunner::new(c))),
        ("cursor-agent", "--model", |c| {
            Box::new(CursorAgentRunner::new(c))
        }),
        ("gemini", "-m", |c| Box::new(GeminiCliRunner::new(c))),
        ("kilo", "--model", |c| Box::new(KiloCliRunner::new(c))),
        ("opencode", "--model", |c| Box::new(OpenCodeRunner::new(c))),
        ("warp", "--model", |c| Box::new(WarpCliRunner::new(c))),
        ("cline", "-m", |c| Box::new(ClineCliRunner::new(c))),
        ("continue", "--model", |c| {
            Box::new(ContinueCliRunner::new(c))
        }),
        ("kiro", "--model", |c| Box::new(KiroCliRunner::new(c))),
    ];

    for (name, flag, build) in cases {
        let dir = tempfile::tempdir().unwrap();
        let runner = build(configured(dir.path(), "{}"));
        run(runner.as_ref(), Some(REQUESTED)).await;
        assert_eq!(
            flag_value(dir.path(), flag).as_deref(),
            Some(REQUESTED),
            "{name} ignored the requested model"
        );

        let dir = tempfile::tempdir().unwrap();
        let runner = build(configured(dir.path(), "{}"));
        run(runner.as_ref(), None).await;
        assert_eq!(
            flag_value(dir.path(), flag).as_deref(),
            Some(CONFIGURED),
            "{name} dropped the configured model"
        );
    }
}
