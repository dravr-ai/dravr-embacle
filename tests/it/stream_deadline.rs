// ABOUTME: Drives a CLI runner's streaming path against a stand-in that stalls after its first line,
// ABOUTME: proving RunnerConfig::timeout bounds a stream as it bounds complete() and kills the child
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! `complete()` has always run under `RunnerConfig::timeout`; the streaming
//! paths read their child's stdout until it closed, so a CLI that stalled
//! after spawning held its subprocess and its runner seat for as long as the
//! caller kept the stream.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, LlmProvider};
use embacle::ClaudeCodeRunner;
#[cfg(target_os = "linux")]
use tokio::time::sleep;
use tokio::time::timeout;
use tokio_stream::StreamExt;

// Each test execs a script it just wrote; see `support` for the ETXTBSY race
// that makes every call here retry its spawn.
use crate::support::{retry_text_file_busy, warm_up};

/// Prints one stream-json event, then sleeps in place of answering, as a
/// CLI hung on its upstream does. Called without arguments it exits at once,
/// which is how [`warm_up`] runs it.
const STALLING_CLAUDE: &str = r#"#!/bin/sh
[ $# -eq 0 ] && exit 0
echo $$ > "$(dirname "$0")/pid"
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}'
exec sleep 30
"#;

/// Whether process `pid` has exited (gone, or a zombie awaiting its reaper).
#[cfg(target_os = "linux")]
fn has_exited(pid: &str) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
        stat.rsplit(") ")
            .next()
            .is_some_and(|rest| rest.starts_with('Z'))
    })
}

#[tokio::test]
async fn a_stalled_stream_times_out_and_its_child_is_killed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claude");
    fs::write(&path, STALLING_CLAUDE).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    let mut config = RunnerConfig::new(path);
    config.timeout = Duration::from_millis(300);
    let runner = ClaudeCodeRunner::new(config);
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);
    // The 300 ms deadline covers the stand-in's start, so its first exec,
    // which the OS may hold for an assessment, happens before the clock runs.
    assert!(warm_up(dir.path().join("claude").as_path())
        .await
        .unwrap()
        .success());

    let started = Instant::now();
    let mut stream = retry_text_file_busy(|| runner.complete_stream(&request))
        .await
        .unwrap();
    let mut items = Vec::new();
    while let Some(item) = timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("the stream ends on its own once the timeout fires")
    {
        items.push(item);
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "a 300 ms timeout ended the stream after {elapsed:?}"
    );
    assert_eq!(items.len(), 2, "the partial chunk, then the timeout");
    assert_eq!(items[0].as_ref().unwrap().delta, "partial");
    let err = items[1].as_ref().unwrap_err();
    assert_eq!(err.kind, ErrorKind::Timeout, "{}", err.message);

    #[cfg(target_os = "linux")]
    {
        let pid = fs::read_to_string(dir.path().join("pid")).unwrap();
        let pid = pid.trim();
        let mut exited = has_exited(pid);
        for _ in 0..40 {
            if exited {
                break;
            }
            sleep(Duration::from_millis(50)).await;
            exited = has_exited(pid);
        }
        assert!(exited, "the timed-out child {pid} was killed");
    }
    drop(stream);
}
