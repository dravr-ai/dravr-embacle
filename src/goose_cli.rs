// ABOUTME: Goose CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `goose` CLI with JSON/stream-JSON output parsing and session resume
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
use tempfile::NamedTempFile;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::prepare_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Default model for Goose CLI (provider-agnostic)
const DEFAULT_MODEL: &str = "auto";

/// Fallback model list (Goose delegates to whatever backend the user configured)
const FALLBACK_MODELS: &[&str] = &["auto"];

/// Goose CLI runner
///
/// Implements `LlmProvider` by delegating to the `goose` binary with
/// `--output-format json` for complete responses and `--output-format stream-json`
/// for streaming. Uses `--quiet` to suppress progress output and `--no-session`
/// for stateless invocations.
pub struct GooseCliRunner {
    base: CliRunnerBase,
}

impl GooseCliRunner {
    /// Create a new Goose CLI runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            base: CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS),
        }
    }

    /// Store a session ID for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        self.base.set_session(key, session_id).await;
    }

    /// Build the base command with common arguments (without prompt delivery)
    fn build_command_base(&self, output_format: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args([
            "run",
            "--quiet",
            "--no-session",
            "--output-format",
            output_format,
        ]);

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

    /// Parse a JSON response from `goose run --output-format json`
    ///
    /// The response contains a `messages` array; we extract the last assistant
    /// message and join its `content` text parts.
    fn parse_json_response(raw: &[u8]) -> Result<ChatResponse, RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Goose CLI output is not valid UTF-8: {e}"))
        })?;

        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| {
            RunnerError::internal(format!("Failed to parse Goose JSON response: {e}"))
        })?;

        let messages = value
            .get("messages")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RunnerError::internal("Goose response missing 'messages' array"))?;

        // Find the last assistant message and join its content text parts
        let mut content = String::new();
        for msg in messages.iter().rev() {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role == "assistant" {
                if let Some(parts) = msg.get("content").and_then(|v| v.as_array()) {
                    for part in parts {
                        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if part_type == "text" {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(t);
                            }
                        }
                    }
                }
                break;
            }
        }

        Ok(ChatResponse {
            content,
            model: "goose".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }
}

#[async_trait]
impl LlmProvider for GooseCliRunner {
    crate::delegate_provider_base!("goose", "Goose CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "goose"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        // Write prompt to a temp file since Goose reads from `-i <path>`
        let mut prompt_file = NamedTempFile::new().map_err(|e| {
            RunnerError::internal(format!("Failed to create temp file for Goose prompt: {e}"))
        })?;
        std::io::Write::write_all(&mut prompt_file, prompt.as_bytes()).map_err(|e| {
            RunnerError::internal(format!("Failed to write Goose prompt to temp file: {e}"))
        })?;

        let mut cmd = self.build_command_base("json");
        cmd.args(["-i", &prompt_file.path().display().to_string()]);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--session-id", &sid, "--resume"]);
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "goose")?;

        Self::parse_json_response(&output.stdout)
    }

    #[instrument(skip_all, fields(runner = "goose"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        let mut cmd = self.build_command_base("stream-json");
        cmd.args(["-i", "-"]);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--session-id", &sid, "--resume"]);
            }
        }

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn goose for streaming: {e}"))
        })?;

        // Write prompt to stdin then close it
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture goose stdin for streaming"))?;
        let prompt_owned = prompt.to_owned();
        tokio::spawn(async move {
            let _ = stdin.write_all(prompt_owned.as_bytes()).await;
            let _ = stdin.shutdown().await;
        });

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture goose stdout for streaming"))?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading goose stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| RunnerError::internal(format!("Invalid JSON in goose stream: {e}")))?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "message" => {
                    // Extract text from message.content[] text parts
                    let mut delta = String::new();
                    if let Some(msg) = value.get("message") {
                        if let Some(parts) = msg.get("content").and_then(|v| v.as_array()) {
                            for part in parts {
                                let pt = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                if pt == "text" {
                                    if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                        delta.push_str(t);
                                    }
                                }
                            }
                        }
                    }
                    Ok(StreamChunk {
                        delta,
                        is_final: false,
                        finish_reason: None,
                    })
                }
                "complete" => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason: Some("stop".to_owned()),
                }),
                _ => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                }),
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
    fn test_parse_json_response_basic() {
        let json = br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":[{"type":"text","text":"hello"}]}],"metadata":{"total_tokens":null,"status":"completed"}}"#;
        let resp = GooseCliRunner::parse_json_response(json).unwrap();
        assert_eq!(resp.content, "hello");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_json_response_multi_content() {
        let json = br#"{"messages":[{"role":"assistant","content":[{"type":"text","text":"part1"},{"type":"text","text":"part2"}]}],"metadata":{"status":"completed"}}"#;
        let resp = GooseCliRunner::parse_json_response(json).unwrap();
        assert_eq!(resp.content, "part1part2");
    }

    #[test]
    fn test_parse_json_response_skips_user_messages() {
        let json = br#"{"messages":[{"role":"user","content":[{"type":"text","text":"ignored"}]},{"role":"assistant","content":[{"type":"text","text":"kept"}]}],"metadata":{}}"#;
        let resp = GooseCliRunner::parse_json_response(json).unwrap();
        assert_eq!(resp.content, "kept");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("goose"));
        let runner = GooseCliRunner::new(config);
        assert_eq!(runner.default_model(), "auto");
    }

    #[test]
    fn test_capabilities() {
        let config = RunnerConfig::new(PathBuf::from("goose"));
        let runner = GooseCliRunner::new(config);
        assert!(runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("goose"));
        let runner = GooseCliRunner::new(config);
        assert_eq!(runner.name(), "goose");
        assert_eq!(runner.display_name(), "Goose CLI");
    }
}
