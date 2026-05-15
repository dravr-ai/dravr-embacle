// ABOUTME: Integration tests for the tmux_session substrate driving real tmux + bash
// ABOUTME: Exercises spawn, paste, send-enter, wait-for-marker, and kill end-to-end
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![cfg(feature = "tmux-session")]

use std::process::{Command, Stdio};
use std::time::Duration;

use embacle::tmux_session::{strip_ansi, TmuxSession};

/// Generate a unique session name so concurrent test runs do not collide.
fn unique_name(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

/// Returns true if `tmux` is available on PATH. Used to skip integration
/// tests in environments where tmux is not installed.
fn tmux_available() -> bool {
    Command::new("tmux")
        .arg("-V")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn spawn_and_kill_session_round_trip() {
    if !tmux_available() {
        eprintln!("skip: tmux not on PATH");
        return;
    }
    let name = unique_name("embacle-test-spawn");

    let session = TmuxSession::spawn(&name, "bash", &["--norc", "--noprofile"])
        .await
        .expect("spawn should succeed");

    assert!(TmuxSession::exists(&name).await.expect("has-session query"));

    session.kill().await.expect("kill should succeed");
    assert!(!TmuxSession::exists(&name)
        .await
        .expect("has-session after kill"));
}

#[tokio::test]
async fn paste_and_capture_round_trips_text() {
    if !tmux_available() {
        eprintln!("skip: tmux not on PATH");
        return;
    }
    let name = unique_name("embacle-test-paste");

    // `cat` echoes whatever is pasted into its pane to the captured screen,
    // making it a clean stand-in for an interactive prompt without needing a
    // real LLM CLI.
    let session = TmuxSession::spawn(&name, "cat", &[])
        .await
        .expect("spawn cat");

    session
        .paste("hello tmux substrate\nsecond line")
        .await
        .expect("paste");
    session.send_enter().await.expect("submit");

    // Give cat a moment to echo the buffered line back.
    let observed = session
        .wait_for_marker(
            "second line",
            Duration::from_secs(5),
            Duration::from_millis(100),
        )
        .await
        .expect("marker should appear in echo");

    assert!(observed.contains("hello tmux substrate"));
    assert!(observed.contains("second line"));

    session.kill().await.expect("kill");
}

#[tokio::test]
async fn wait_for_marker_times_out_when_absent() {
    if !tmux_available() {
        eprintln!("skip: tmux not on PATH");
        return;
    }
    let name = unique_name("embacle-test-timeout");

    let session = TmuxSession::spawn(&name, "bash", &["--norc", "--noprofile"])
        .await
        .expect("spawn bash");

    let result = session
        .wait_for_marker(
            "this-marker-will-never-appear-xyz123",
            Duration::from_millis(400),
            Duration::from_millis(50),
        )
        .await;

    assert!(result.is_err(), "expected a timeout error, got {result:?}");

    session.kill().await.expect("kill");
}

#[tokio::test]
async fn drop_kills_orphaned_session() {
    if !tmux_available() {
        eprintln!("skip: tmux not on PATH");
        return;
    }
    let name = unique_name("embacle-test-drop");

    {
        let _session = TmuxSession::spawn(&name, "bash", &["--norc", "--noprofile"])
            .await
            .expect("spawn bash");
        assert!(TmuxSession::exists(&name).await.expect("has-session live"));
    }

    // Drop ran synchronously; the session should be gone.
    assert!(!TmuxSession::exists(&name)
        .await
        .expect("has-session after drop"));
}

#[tokio::test]
async fn strip_ansi_round_trips_via_capture() {
    if !tmux_available() {
        eprintln!("skip: tmux not on PATH");
        return;
    }
    let name = unique_name("embacle-test-ansi");

    // printf emits an ANSI red sequence around 'PAYLOAD'; capture-pane will
    // record the rendered cells, and our strip_ansi helper must produce
    // bare text containing 'PAYLOAD'.
    let session = TmuxSession::spawn(
        &name,
        "bash",
        &[
            "--norc",
            "--noprofile",
            "-c",
            "printf '\\033[31mPAYLOAD\\033[0m\\n'; sleep 30",
        ],
    )
    .await
    .expect("spawn bash");

    let observed = session
        .wait_for_marker(
            "PAYLOAD",
            Duration::from_secs(3),
            Duration::from_millis(100),
        )
        .await
        .expect("marker visible");

    assert!(observed.contains("PAYLOAD"));
    assert!(!observed.contains("\x1b["), "ANSI not stripped");

    // Also verify the helper handles a synthetic ESC sequence directly.
    assert_eq!(strip_ansi("\x1b[1;31mRED\x1b[0m"), "RED");

    session.kill().await.expect("kill");
}
