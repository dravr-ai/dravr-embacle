// ABOUTME: Subprocess spawning with timeout and output-size safety limits
// ABOUTME: Wraps tokio::process::Command with structured output and error handling
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::ffi::OsStr;
use std::io::Write;
use std::process::Stdio;
use std::time::{Duration, Instant};

use crate::types::RunnerError;
use tempfile::{Builder as TempFileBuilder, TempPath};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::time::timeout as tokio_timeout;
use tracing::{debug, info, trace, warn};

/// Default maximum output size (10 MiB)
const DEFAULT_MAX_OUTPUT_BYTES: usize = 10 * 1024 * 1024;

/// Flags whose value is a prompt or a configuration that can carry a
/// credential (an MCP server's `Authorization` header), never traced.
const REDACTED_VALUE_FLAGS: &[&str] = &[
    "-p",
    "--prompt",
    "--system-prompt",
    "--append-system-prompt",
    "--mcp-config",
    "--additional-mcp-config",
];

/// Longest argument traced as it is. Anything longer — a prompt passed as a
/// positional argument, an inline JSON document — is free text, not a flag,
/// and is traced by its size only.
const MAX_TRACED_ARG_BYTES: usize = 128;

/// `args` as the spawn trace may show them: every flag and short value as
/// is, and in place of each prompt, system prompt, MCP configuration or other
/// free text, `<redacted N bytes>`.
///
/// The value after a [`REDACTED_VALUE_FLAGS`] flag is redacted, as is the
/// value of its `--flag=value` form, unless the next argument is itself a
/// `--` flag (the flag took no value, as `continue -p --format json` does).
/// Any other argument longer than [`MAX_TRACED_ARG_BYTES`] is redacted too.
pub(crate) fn redacted_argv<'a>(args: impl IntoIterator<Item = &'a OsStr>) -> Vec<String> {
    let mut traced = Vec::new();
    let mut value_follows = false;
    for arg in args {
        let text = arg.to_string_lossy();
        let is_value = value_follows && !text.starts_with("--");
        value_follows = REDACTED_VALUE_FLAGS.contains(&text.as_ref());
        let flag_with_value = text
            .split_once('=')
            .filter(|(flag, _)| REDACTED_VALUE_FLAGS.contains(flag));
        traced.push(if is_value {
            redaction(arg)
        } else if let Some((flag, value)) = flag_with_value {
            format!("{flag}={}", redaction(OsStr::new(value)))
        } else if text.len() > MAX_TRACED_ARG_BYTES {
            redaction(arg)
        } else {
            text.into_owned()
        });
    }
    traced
}

/// What the trace shows in place of `value`.
fn redaction(value: &OsStr) -> String {
    format!("<redacted {} bytes>", value.len())
}

/// Write `contents` to a new file only this process's user can read (mode
/// 0600 on Unix), for a CLI to read a configuration from by path rather than
/// from its command line, where every process on the host can read it in
/// `/proc/<pid>/cmdline`.
///
/// The file is removed when the returned path is dropped, so the caller keeps
/// it until the child that reads it has exited.
///
/// # Errors
///
/// `RunnerError::internal` when the file cannot be created or written.
pub(crate) fn private_config_file(prefix: &str, contents: &str) -> Result<TempPath, RunnerError> {
    let mut file = TempFileBuilder::new()
        .prefix(prefix)
        .suffix(".json")
        .tempfile()
        .map_err(|e| RunnerError::internal(format!("Failed to create a config file: {e}")))?;
    file.write_all(contents.as_bytes())
        .and_then(|()| file.flush())
        .map_err(|e| RunnerError::internal(format!("Failed to write a config file: {e}")))?;
    Ok(file.into_temp_path())
}

/// Hand `input` to the child's stdin and close it, on a task of its own so a
/// child that writes before it has read everything cannot deadlock against
/// this write.
///
/// A child that exits without reading it all closes the pipe; that write
/// failure is the child's to report, through its own exit, and is only
/// logged here.
pub(crate) fn feed_stdin(stdin: Option<ChildStdin>, input: Vec<u8>) {
    let Some(mut stdin) = stdin else {
        warn!("CLI stdin was not piped; the input was not delivered");
        return;
    };
    tokio::spawn(async move {
        if let Err(e) = stdin.write_all(&input).await {
            debug!(error = %e, "CLI closed stdin before reading all of its input");
            return;
        }
        if let Err(e) = stdin.shutdown().await {
            debug!(error = %e, "Closing CLI stdin failed");
        }
    });
}

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
/// `max_output_bytes` to prevent unbounded memory consumption. Dropping the
/// returned future before it completes kills the child as well.
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
    run_cli_command_with_input(cmd, None, timeout, max_output_bytes).await
}

/// [`run_cli_command`], with `input` written to the child's stdin and the
/// pipe then closed.
///
/// For a CLI that reads its prompt from stdin, which keeps the prompt off
/// the command line every process on the host can read. With `input` `None`
/// the child's stdin is left as [`run_cli_command`] leaves it.
///
/// # Errors
///
/// As [`run_cli_command`].
pub async fn run_cli_command_with_input(
    cmd: &mut Command,
    input: Option<&[u8]>,
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
    // A caller that drops this future has abandoned the call: its turn was
    // stopped, or its own deadline fired. Without this the child runs to
    // completion with nobody reading it, holding a runner seat and spending
    // provider quota on an answer that is thrown away.
    cmd.kill_on_drop(true);

    if input.is_some() {
        cmd.stdin(Stdio::piped());
    }

    // Trace-level dump of the resolved program + argv so an operator with
    // `RUST_LOG=embacle::process=trace` can see exactly which CLI runner
    // got invoked and with which flags. Prompts, system prompts and MCP
    // configurations (whose headers carry bearer tokens) are redacted to
    // their size: this line is a debugging aid, and a log sink is no place
    // for a credential or an athlete's words.
    if tracing::enabled!(tracing::Level::TRACE) {
        let program = cmd.as_std().get_program().to_string_lossy().to_string();
        let args = redacted_argv(cmd.as_std().get_args());
        trace!(program = %program, args = ?args, "Spawning CLI process");
    }

    let start = Instant::now();

    let mut child = cmd
        .spawn()
        .map_err(|e| RunnerError::internal(format!("Failed to spawn CLI process: {e}")))?;
    if let Some(input) = input {
        feed_stdin(child.stdin.take(), input.to_vec());
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn traced(args: &[&str]) -> Vec<String> {
        redacted_argv(args.iter().map(OsStr::new))
    }

    #[test]
    fn prompts_and_mcp_configs_are_traced_by_size_only() {
        let config = r#"{"mcpServers":{"dravr":{"type":"http","url":"http://127.0.0.1:9/mcp","headers":{"Authorization":"Bearer tok-secret"}}}}"#;
        let args = traced(&[
            "-p",
            "my athlete's resting HR is 41",
            "--system-prompt",
            "You coach Jane Doe",
            "--mcp-config",
            config,
            "--additional-mcp-config",
            "@/tmp/embacle-mcp-x.json",
            "--model",
            "opus",
        ]);

        assert_eq!(
            args,
            [
                "-p",
                "<redacted 29 bytes>",
                "--system-prompt",
                "<redacted 18 bytes>",
                "--mcp-config",
                &format!("<redacted {} bytes>", config.len()),
                "--additional-mcp-config",
                "<redacted 24 bytes>",
                "--model",
                "opus",
            ]
        );
        let joined = args.join(" ");
        assert!(!joined.contains("Bearer tok-secret"), "{joined}");
        assert!(!joined.contains("resting HR"), "{joined}");
    }

    #[test]
    fn a_flag_followed_by_a_flag_took_no_value() {
        assert_eq!(
            traced(&["-p", "--format", "json", "short"]),
            ["-p", "--format", "json", "short"]
        );
    }

    #[test]
    fn the_equals_form_is_redacted_too() {
        assert_eq!(
            traced(&["--system-prompt=You coach Jane", "--output-format=json"]),
            [
                "--system-prompt=<redacted 14 bytes>",
                "--output-format=json"
            ]
        );
    }

    #[test]
    fn a_long_positional_argument_is_free_text_and_redacted() {
        let prompt = "x".repeat(MAX_TRACED_ARG_BYTES + 1);
        let kept = "y".repeat(MAX_TRACED_ARG_BYTES);
        assert_eq!(
            traced(&["exec", &prompt, &kept]),
            [
                "exec".to_owned(),
                format!("<redacted {} bytes>", MAX_TRACED_ARG_BYTES + 1),
                kept
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_private_config_file_is_0600_and_removed_on_drop() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let path = private_config_file("embacle-test-", "{\"k\":1}").unwrap(); // Safe: test assertion
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777; // Safe: test assertion
        assert_eq!(mode, 0o600);
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"k\":1}"); // Safe: test assertion
        let kept = path.to_path_buf();
        drop(path);
        assert!(!kept.exists(), "the file goes with its path");
    }
}
