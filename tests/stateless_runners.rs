// ABOUTME: Drives each CLI runner twice against a shell stand-in that records its argv, proving
// ABOUTME: no call ever resumes the session a previous call on the same runner instance opened
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! One runner instance serves every caller in a process, and each request
//! carries its whole conversation. A runner that remembered the session id a
//! CLI returned and passed it back on the next call (`--resume`, `--session`,
//! `--taskId`, `--conversation`) spliced the previous caller's conversation
//! into the next caller's turn: measured on 2026-09-30, athlete B's Claude Code
//! call quoted athlete A's resting heart rate.
//!
//! Every stand-in here reports a session id the way its real CLI does, and
//! every test runs two calls with the same model on one runner. The second
//! call's argv must carry no resume flag and must never name the id the first
//! call was handed back.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::mem;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::{
    ClaudeCodeRunner, ClineCliRunner, ContinueCliRunner, CursorAgentRunner, GeminiCliRunner,
    GooseCliRunner, KiloCliRunner, KiroCliRunner, OpenCodeRunner, WarpCliRunner,
};
use tokio_stream::StreamExt;

/// The id every stand-in reports as the session it opened.
const SESSION_ID: &str = "sess-alice-7f3a";

/// Written after each invocation's arguments in the argv log.
const CALL_END: &str = "--END-OF-CALL--";

/// Flags any of the wrapped CLIs uses to continue an earlier session.
const RESUME_FLAGS: &[&str] = &[
    "--resume",
    "-r",
    "--continue",
    "-c",
    "--session",
    "--session-id",
    "--taskId",
    "--conversation",
    "--fork-session",
];

/// A CLI stand-in that appends its argv, one argument per line, to
/// `argv.log` beside it, then prints `stdout`.
fn fake_cli(dir: &Path, stdout: &str) -> PathBuf {
    let path = dir.join("fake-cli");
    let log = dir.join("argv.log");
    let script = format!(
        "#!/bin/sh\n{{ for a in \"$@\"; do printf '%s\\n' \"$a\"; done; printf '%s\\n' '{CALL_END}'; }} >> '{}'\ncat <<'FAKE_EOF'\n{stdout}\nFAKE_EOF\n",
        log.display()
    );
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// The argv of every recorded invocation, in call order.
fn recorded_calls(dir: &Path) -> Vec<Vec<String>> {
    let log = fs::read_to_string(dir.join("argv.log")).unwrap();
    let mut calls = Vec::new();
    let mut current = Vec::new();
    for line in log.lines() {
        if line == CALL_END {
            calls.push(mem::take(&mut current));
        } else {
            current.push(line.to_owned());
        }
    }
    calls
}

/// Two turns from two different athletes, on the same model.
fn requests() -> [ChatRequest; 2] {
    [
        ChatRequest::new(vec![ChatMessage::user("I am Alice, resting HR 47")]).with_model("m"),
        ChatRequest::new(vec![ChatMessage::user("I am Bob, what did Alice say?")]).with_model("m"),
    ]
}

/// Both calls ran, and neither carries a resume flag or the reported id.
fn assert_never_resumed(dir: &Path) -> Vec<Vec<String>> {
    let calls = assert_no_resume_args(dir);
    assert!(
        calls[1].iter().any(|a| a.contains("Bob")),
        "the second call carries its own prompt: {:?}",
        calls[1]
    );
    calls
}

/// [`assert_never_resumed`] for a CLI that reads its prompt from a file or
/// stdin rather than from argv.
fn assert_no_resume_args(dir: &Path) -> Vec<Vec<String>> {
    let calls = recorded_calls(dir);
    assert_eq!(calls.len(), 2, "expected two invocations, got {calls:?}");
    for (i, argv) in calls.iter().enumerate() {
        for flag in RESUME_FLAGS {
            assert!(
                !argv.iter().any(|a| a == flag),
                "call {i} passed {flag}: {argv:?}"
            );
        }
        assert!(
            !argv.iter().any(|a| a.contains(SESSION_ID)),
            "call {i} named the session a previous call opened: {argv:?}"
        );
    }
    calls
}

/// Run `complete()` for both requests and return the two answers.
async fn complete_twice(runner: &dyn LlmProvider) -> Vec<String> {
    let mut answers = Vec::new();
    for request in &requests() {
        answers.push(runner.complete(request).await.unwrap().content);
    }
    answers
}

/// Run `complete_stream()` for both requests, draining each stream, and
/// return the text each one streamed.
async fn stream_twice(runner: &dyn LlmProvider) -> Vec<String> {
    let mut answers = Vec::new();
    for request in &requests() {
        let mut stream = runner.complete_stream(request).await.unwrap();
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            if let Ok(chunk) = chunk {
                text.push_str(&chunk.delta);
            }
        }
        answers.push(text);
    }
    answers
}

#[tokio::test]
async fn claude_code_complete_never_resumes_and_never_persists() {
    let dir = tempfile::tempdir().unwrap();
    let doc = format!(r#"{{"result":"pong","is_error":false,"session_id":"{SESSION_ID}"}}"#);
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), &doc)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);

    for argv in assert_never_resumed(dir.path()) {
        assert!(
            argv.iter().any(|a| a == "--no-session-persistence"),
            "every call is an unsaved session: {argv:?}"
        );
    }
}

#[tokio::test]
async fn claude_code_stream_never_resumes_and_never_persists() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"assistant\",\"message\":{{\"content\":[{{\"type\":\"text\",\"text\":\"pong\"}}]}},\"session_id\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"result\",\"is_error\":false,\"result\":\"pong\",\"session_id\":\"{SESSION_ID}\"}}"
    );
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    assert_eq!(stream_twice(&runner).await, ["pong", "pong"]);

    for argv in assert_never_resumed(dir.path()) {
        assert!(
            argv.iter().any(|a| a == "--no-session-persistence"),
            "every call is an unsaved session: {argv:?}"
        );
    }
}

#[tokio::test]
async fn claude_code_complete_then_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = format!(
        r#"{{"type":"result","result":"pong","is_error":false,"session_id":"{SESSION_ID}"}}"#
    );
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), &doc)));
    let [alice, bob] = requests();

    assert_eq!(runner.complete(&alice).await.unwrap().content, "pong");
    let mut stream = runner.complete_stream(&bob).await.unwrap();
    while stream.next().await.is_some() {}

    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn cursor_agent_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = format!(r#"{{"result":"pong","is_error":false,"session_id":"{SESSION_ID}"}}"#);
    let runner = CursorAgentRunner::new(RunnerConfig::new(fake_cli(dir.path(), &doc)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn cursor_agent_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = format!(
        r#"{{"type":"result","result":"pong","is_error":false,"session_id":"{SESSION_ID}"}}"#
    );
    let runner = CursorAgentRunner::new(RunnerConfig::new(fake_cli(dir.path(), &doc)));

    stream_twice(&runner).await;
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn gemini_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = format!(r#"{{"session_id":"{SESSION_ID}","response":"pong"}}"#);
    let runner = GeminiCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &doc)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn gemini_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"init\",\"session_id\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"pong\",\"delta\":true}}\n\
         {{\"type\":\"result\",\"status\":\"success\"}}"
    );
    let runner = GeminiCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    stream_twice(&runner).await;
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn goose_never_resumes_and_runs_without_a_session() {
    let dir = tempfile::tempdir().unwrap();
    let doc = r#"{"messages":[{"role":"assistant","content":[{"type":"text","text":"pong"}]}]}"#;
    let runner = GooseCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), doc)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    for argv in assert_no_resume_args(dir.path()) {
        assert!(argv.iter().any(|a| a == "--no-session"), "{argv:?}");
    }
}

#[tokio::test]
async fn goose_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = r#"{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"pong"}]}}"#;
    let runner = GooseCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), doc)));

    stream_twice(&runner).await;
    for argv in assert_no_resume_args(dir.path()) {
        assert!(argv.iter().any(|a| a == "--no-session"), "{argv:?}");
    }
}

#[tokio::test]
async fn cline_never_resumes_a_task() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"task_started\",\"taskId\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"say\",\"say\":\"completion_result\",\"text\":\"pong\"}}"
    );
    let runner = ClineCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn cline_stream_never_resumes_a_task() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"task_started\",\"taskId\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"say\",\"say\":\"completion_result\",\"text\":\"pong\"}}"
    );
    let runner = ClineCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    stream_twice(&runner).await;
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn continue_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ContinueCliRunner::new(RunnerConfig::new(fake_cli(
        dir.path(),
        r#"{"response":"pong"}"#,
    )));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn continue_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ContinueCliRunner::new(RunnerConfig::new(fake_cli(
        dir.path(),
        r#"{"response":"pong"}"#,
    )));

    assert_eq!(stream_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn kiro_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let runner = KiroCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), "pong")));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn kiro_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let runner = KiroCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), "pong")));

    assert_eq!(stream_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn kilo_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"text\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"text\",\"text\":\"pong\"}}}}\n\
         {{\"type\":\"step_finish\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"step-finish\",\"reason\":\"stop\"}}}}"
    );
    let runner = KiloCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn kilo_stream_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"text\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"text\",\"text\":\"pong\"}}}}\n\
         {{\"type\":\"step_finish\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"step-finish\",\"reason\":\"stop\"}}}}"
    );
    let runner = KiloCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    stream_twice(&runner).await;
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn opencode_never_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"text\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"text\",\"text\":\"pong\"}}}}\n\
         {{\"type\":\"step_finish\",\"sessionID\":\"{SESSION_ID}\",\"part\":{{\"type\":\"step-finish\",\"reason\":\"stop\"}}}}"
    );
    let runner = OpenCodeRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}

#[tokio::test]
async fn warp_never_resumes_a_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let lines = format!(
        "{{\"type\":\"system\",\"event_type\":\"conversation_started\",\"conversation_id\":\"{SESSION_ID}\"}}\n\
         {{\"type\":\"agent\",\"text\":\"pong\"}}"
    );
    let runner = WarpCliRunner::new(RunnerConfig::new(fake_cli(dir.path(), &lines)));

    assert_eq!(complete_twice(&runner).await, ["pong", "pong"]);
    assert_never_resumed(dir.path());
}
