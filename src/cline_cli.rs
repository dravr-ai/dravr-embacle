// ABOUTME: Cline CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `cline` CLI with NDJSON output parsing and session resume via task IDs
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::prepare_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Default model for Cline CLI (provider-agnostic)
const DEFAULT_MODEL: &str = "auto";

/// Fallback model list (Cline delegates to whatever backend the user configured)
const FALLBACK_MODELS: &[&str] = &["auto"];

/// Cline CLI runner
///
/// Implements `LlmProvider` by delegating to the `cline` binary with
/// `task --json --act --yolo` for automatic execution. All output is NDJSON
/// with event types `task_started`, `say` (text deltas), and `say`
/// (`completion_result` for final output).
pub struct ClineCliRunner {
    base: CliRunnerBase,
}

impl ClineCliRunner {
    /// Create a new Cline CLI runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            base: CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS),
        }
    }

    /// Store a task ID for later session resumption
    pub async fn set_session(&self, key: &str, task_id: &str) {
        self.base.set_session(key, task_id).await;
    }

    /// Build the command with all arguments
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["task", "--json", "--act", "--yolo", prompt]);

        if let Some(model) = self.base.config.model.as_deref() {
            cmd.args(["-m", model]);
        }

        for arg in &self.base.config.extra_args {
            cmd.arg(arg);
        }

        if let Ok(policy) = build_policy(
            self.base.config.working_directory.as_deref(),
            &self.base.config.allowed_env_keys,
        ) {
            apply_sandbox(&mut cmd, &policy);
        }

        cmd
    }

    /// Parse NDJSON output from `cline task --json`
    ///
    /// Scans for `task_started` to capture the task ID, and for
    /// `say:"completion_result"` to extract the final response text.
    fn parse_ndjson_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Cline CLI output is not valid UTF-8: {e}"))
        })?;

        let mut task_id: Option<String> = None;
        let mut content = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let line_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match line_type {
                "task_started" => {
                    if let Some(tid) = value.get("taskId").and_then(|v| v.as_str()) {
                        task_id = Some(tid.to_owned());
                    }
                }
                "say" => {
                    let say_type = value.get("say").and_then(|v| v.as_str()).unwrap_or("");
                    if say_type == "completion_result" {
                        if let Some(t) = value.get("text").and_then(|v| v.as_str()) {
                            t.clone_into(&mut content);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok((
            ChatResponse {
                content,
                model: "cline".to_owned(),
                usage: None,
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            },
            task_id,
        ))
    }
}

#[async_trait]
impl LlmProvider for ClineCliRunner {
    crate::delegate_provider_base!("cline", "Cline CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "cline"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if let Some(tid) = self.base.get_session(model).await {
                cmd.args(["--taskId", &tid]);
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "cline")?;

        let (response, task_id) = Self::parse_ndjson_response(&output.stdout)?;

        if let Some(tid) = task_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &tid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "cline"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if let Some(tid) = self.base.get_session(model).await {
                cmd.args(["--taskId", &tid]);
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn cline for streaming: {e}"))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture cline stdout for streaming"))?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading cline stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| RunnerError::internal(format!("Invalid JSON in cline stream: {e}")))?;

            let line_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if line_type == "say" {
                let say_type = value.get("say").and_then(|v| v.as_str()).unwrap_or("");
                match say_type {
                    "text" => {
                        let delta = value
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        Ok(StreamChunk {
                            delta,
                            is_final: false,
                            finish_reason: None,
                        })
                    }
                    "completion_result" => Ok(StreamChunk {
                        delta: value
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned(),
                        is_final: true,
                        finish_reason: Some("stop".to_owned()),
                    }),
                    _ => Ok(StreamChunk {
                        delta: String::new(),
                        is_final: false,
                        finish_reason: None,
                    }),
                }
            } else {
                Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                })
            }
        });

        Ok(Box::pin(GuardedStream::new(stream, child, stderr_task)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_ndjson_response_basic() {
        let ndjson = br#"{"type":"task_started","taskId":"abc-123"}
{"type":"say","say":"text","text":"working on it","ts":1234,"partial":false}
{"type":"say","say":"completion_result","text":"hello from cline","ts":1235}"#;

        let (resp, tid) = ClineCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "hello from cline");
        assert_eq!(tid, Some("abc-123".to_owned()));
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_ndjson_captures_task_id() {
        let ndjson = br#"{"type":"task_started","taskId":"uuid-456"}
{"type":"say","say":"completion_result","text":"done"}"#;

        let (_, tid) = ClineCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(tid, Some("uuid-456".to_owned()));
    }

    #[test]
    fn test_parse_ndjson_no_completion_returns_empty() {
        let ndjson = br#"{"type":"task_started","taskId":"uuid-789"}
{"type":"say","say":"text","text":"partial output"}"#;

        let (resp, _) = ClineCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("cline"));
        let runner = ClineCliRunner::new(config);
        assert_eq!(runner.default_model(), "auto");
    }

    #[test]
    fn test_capabilities() {
        let config = RunnerConfig::new(PathBuf::from("cline"));
        let runner = ClineCliRunner::new(config);
        assert!(runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("cline"));
        let runner = ClineCliRunner::new(config);
        assert_eq!(runner.name(), "cline");
        assert_eq!(runner.display_name(), "Cline CLI");
    }
}
