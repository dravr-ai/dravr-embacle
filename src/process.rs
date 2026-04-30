// ABOUTME: Subprocess spawning with timeout and output-size safety limits
// ABOUTME: Wraps tokio::process::Command with structured output and error handling
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::process::Stdio;
use std::time::{Duration, Instant};

use crate::types::RunnerError;
use tokio::io::AsyncReadExt;
use tokio::process::{ChildStderr, ChildStdout, Command};
use tokio::time::timeout as tokio_timeout;
use tracing::{info, trace, warn};

/// Default maximum output size (10 MiB)
const DEFAULT_MAX_OUTPUT_BYTES: usize = 10 * 1024 * 1024;

/// Structured output from a CLI command execution
#[derive(Debug, Clone)]
pub struct CliOutput {
    /// Captured standard output bytes
    pub stdout: Vec<u8>,
    /// Captured standard error bytes
    pub stderr: Vec<u8>,
    /// Process exit code (-1 if the process was killed)
    pub exit_code: i32,
    /// Wall-clock duration of the command
    pub duration: Duration,
}

/// Read up to `limit` bytes from a `ChildStdout`, returning collected bytes
async fn read_stdout_capped(stream: Option<ChildStdout>, limit: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(mut reader) = stream {
        let mut tmp = [0u8; 8192];
        loop {
            match reader.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let remaining = limit.saturating_sub(buf.len());
                    buf.extend_from_slice(&tmp[..n.min(remaining)]);
                    if buf.len() >= limit {
                        warn!(limit_bytes = limit, "stdout output truncated at cap");
                        break;
                    }
                }
            }
        }
    }
    buf
}

/// Read up to `limit` bytes from a `ChildStderr`, returning collected bytes
pub(crate) async fn read_stderr_capped(stream: Option<ChildStderr>, limit: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(mut reader) = stream {
        let mut tmp = [0u8; 8192];
        loop {
            match reader.read(&mut tmp).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let remaining = limit.saturating_sub(buf.len());
                    buf.extend_from_slice(&tmp[..n.min(remaining)]);
                    if buf.len() >= limit {
                        break;
                    }
                }
            }
        }
    }
    buf
}

/// Run a CLI command with timeout and output-size limits
///
/// The command is spawned as a child process. If it does not exit within
/// `timeout`, it is killed and an error is returned. Output is capped at
/// `max_output_bytes` to prevent unbounded memory consumption.
///
/// # Errors
///
/// Returns `RunnerError` if:
/// - The process cannot be spawned
/// - The process exceeds the timeout (killed and reported)
///
/// Note: a non-zero exit code is returned inside `CliOutput::exit_code`
/// as data, not as an error. Callers decide how to handle it.
pub async fn run_cli_command(
    cmd: &mut Command,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<CliOutput, RunnerError> {
    let effective_max = if max_output_bytes == 0 {
        DEFAULT_MAX_OUTPUT_BYTES
    } else {
        max_output_bytes
    };

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Trace-level dump of the resolved program + argv so an operator with
    // `RUST_LOG=embacle::process=trace` can see exactly which CLI runner
    // got invoked and with which arguments. Stdin (the prompt body) does
    // not pass through `Command` — runners that pipe prompt to stdin log
    // it themselves; runners that pass it as an argv argument get it for
    // free here.
    if tracing::enabled!(tracing::Level::TRACE) {
        let program = cmd.as_std().get_program().to_string_lossy().to_string();
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        trace!(program = %program, args = ?args, "Spawning CLI process");
    }

    let start = Instant::now();

    let mut child = cmd
        .spawn()
        .map_err(|e| RunnerError::internal(format!("Failed to spawn CLI process: {e}")))?;

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    let stdout_task = tokio::spawn(read_stdout_capped(stdout_handle, effective_max));
    let stderr_task = tokio::spawn(read_stderr_capped(stderr_handle, effective_max));

    let wait_result = tokio_timeout(timeout, child.wait()).await;

    let duration = start.elapsed();

    match wait_result {
        Ok(Ok(status)) => {
            let exit_code = status.code().unwrap_or(-1);
            let stdout = stdout_task.await.unwrap_or_default();
            let stderr = stderr_task.await.unwrap_or_default();

            // Operator-facing summary: byte counts, exit code, duration. Sized
            // metadata only — content goes to the trace event below so an
            // info-level RUST_LOG never dumps user prompts or LLM responses.
            info!(
                exit_code,
                stdout_len = stdout.len(),
                stderr_len = stderr.len(),
                latency_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
                "CLI command completed"
            );

            // Full stdout/stderr at trace level so an operator running
            // `RUST_LOG=embacle::process=trace` can follow what each CLI
            // runner emitted without per-runner instrumentation.
            if tracing::enabled!(tracing::Level::TRACE) {
                trace!(
                    stdout = %String::from_utf8_lossy(&stdout),
                    "CLI command stdout"
                );
                if !stderr.is_empty() {
                    trace!(
                        stderr = %String::from_utf8_lossy(&stderr),
                        "CLI command stderr"
                    );
                }
            }

            Ok(CliOutput {
                stdout,
                stderr,
                exit_code,
                duration,
            })
        }
        Ok(Err(e)) => Err(RunnerError::internal(format!(
            "Failed to wait for CLI process: {e}"
        ))),
        Err(_) => {
            warn!(?timeout, "CLI command timed out, killing process");
            let _ = child.kill().await;
            Err(RunnerError::timeout(format!(
                "CLI command timed out after {timeout:?}"
            )))
        }
    }
}
