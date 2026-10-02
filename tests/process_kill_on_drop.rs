// ABOUTME: Proves a CLI child is killed when the caller drops `run_cli_command` mid-flight,
// ABOUTME: so an abandoned call does not leave its subprocess running to completion
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A caller abandons a call by dropping its future: a stopped chat turn, an
//! outer deadline, a cancelled request. The child it spawned must die with
//! it. The stand-in below records its own pid and then sleeps far longer than
//! the test waits, so a child that survives the drop is still alive when the
//! test looks.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::Stdio;
use std::time::Duration;

use embacle::process::run_cli_command;
use tokio::fs;
use tokio::process::Command;
use tokio::time::{sleep, timeout};

/// How long the stand-in child sleeps; far beyond anything the test waits.
const CHILD_SLEEP_SECS: &str = "120";
/// How long to wait for a signalled child to be reaped.
const REAP_DEADLINE: Duration = Duration::from_secs(10);

/// Whether a process with this pid still exists, asked of `kill -0`.
async fn is_alive(pid: &str) -> bool {
    Command::new("kill")
        .args(["-0", pid])
        .stderr(Stdio::null())
        .status()
        .await
        .unwrap()
        .success()
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_the_call_kills_the_child() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("child.pid");

    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(format!(
        "echo $$ > '{}'; exec sleep {CHILD_SLEEP_SECS}",
        pid_file.display()
    ));

    // The call's own timeout is far away; the outer one drops the future.
    let dropped = timeout(
        Duration::from_secs(2),
        run_cli_command(&mut cmd, Duration::from_secs(300), 0),
    )
    .await;
    assert!(dropped.is_err(), "the child should still be sleeping");

    let pid = fs::read_to_string(&pid_file).await.unwrap();
    let pid = pid.trim();
    assert!(!pid.is_empty(), "the child never recorded its pid");

    let waited = timeout(REAP_DEADLINE, async {
        while is_alive(pid).await {
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await;
    assert!(
        waited.is_ok(),
        "child {pid} outlived the dropped call by {REAP_DEADLINE:?}"
    );
}
