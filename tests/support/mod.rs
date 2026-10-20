// ABOUTME: Shared helper for the tests that exec a shell stand-in they just wrote: retries a
// ABOUTME: runner call whose spawn failed with ETXTBSY, the race a concurrent test's fork opens
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A test that writes an executable script and then execs it races every
//! other test in the same binary. While one test holds the script's write
//! descriptor open, another test's runner may fork; the child inherits that
//! descriptor and keeps it until its own exec closes it (close-on-exec). An
//! exec of the script in that window fails with `ETXTBSY` ("Text file busy",
//! os error 26) on Linux. Production never execs a file it has just written,
//! so the race exists only here, and only the spawn of such a call is retried:
//! every other error, including every other spawn failure, is returned as is.

use std::future::Future;
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;

use embacle::types::{ErrorKind, RunnerError};
use tokio::process::Command;
use tokio::time::sleep;

/// How many times a call is made before its `ETXTBSY` is returned.
const ATTEMPTS: u32 = 10;

/// Pause between attempts: long enough for a sibling's fork to reach exec.
const PAUSE: Duration = Duration::from_millis(50);

/// The errno an exec of a file still open for writing fails with on Linux.
const ETXTBSY: i32 = 26;

/// True for exactly the runner error a spawn that hit `ETXTBSY` produces:
/// runners report a spawn failure as an internal `Failed to spawn …: {io error}`,
/// and the io error displays its errno last.
fn is_text_file_busy(error: &RunnerError) -> bool {
    error.kind == ErrorKind::Internal
        && error.message.contains("Failed to spawn ")
        && error.message.ends_with("(os error 26)")
}

/// Run `call`, retrying it while its spawn fails with `ETXTBSY`, up to
/// [`ATTEMPTS`] times; any other outcome is returned on the first attempt.
pub async fn retry_text_file_busy<T, F, Fut>(mut call: F) -> Result<T, RunnerError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, RunnerError>>,
{
    let mut attempt = 1;
    loop {
        match call().await {
            Err(error) if attempt < ATTEMPTS && is_text_file_busy(&error) => {
                attempt += 1;
                sleep(PAUSE).await;
            }
            outcome => return outcome,
        }
    }
}

/// Run the stand-in at `path` once with no arguments, retrying its spawn while
/// it fails with `ETXTBSY` as [`retry_text_file_busy`] does.
///
/// The first exec of a freshly written file can be held while the OS assesses
/// it (macOS does), and with sibling tests writing and running stand-ins of
/// their own that wait can run to seconds; later execs of the same file start
/// at once. A test whose deadline covers the stand-in's start runs it here
/// first, so the deadline measures the runner rather than that assessment.
/// The stand-in must exit promptly when called without arguments.
pub async fn warm_up(path: &Path) -> io::Result<ExitStatus> {
    let mut attempt = 1;
    loop {
        match Command::new(path).status().await {
            Err(error) if attempt < ATTEMPTS && error.raw_os_error() == Some(ETXTBSY) => {
                attempt += 1;
                sleep(PAUSE).await;
            }
            outcome => return outcome,
        }
    }
}
