// ABOUTME: Goose CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `goose` CLI with JSON/stream-JSON output parsing and session resume
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::any::Any;
use std::collections::HashMap;
use std::io;
use std::process::Stdio;
use std::str;
use std::sync::Arc;
use std::time::Duration;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
use tempfile::NamedTempFile;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::build_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Maximum output size for a single Goose CLI invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

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
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl GooseCliRunner {
    /// Create a new Goose CLI runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        let default_model = config
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MODEL.to_owned());
        let available_models = FALLBACK_MODELS.iter().map(|s| (*s).to_owned()).collect();
        Self {
            config,
            default_model,
            available_models,
            session_ids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Store a session ID for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        let mut sessions = self.session_ids.lock().await;
        sessions.insert(key.to_owned(), session_id.to_owned());
    }

    /// Build the base command with common arguments (without prompt delivery)
    fn build_command_base(&self, output_format: &str) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.args([
            "run",
            "--quiet",
            "--no-session",
            "--output-format",
            output_format,
        ]);

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        if let Ok(policy) = build_policy(
            self.config.working_directory.as_deref(),
            &self.config.allowed_env_keys,
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
        })
    }
}

#[async_trait]
impl LlmProvider for GooseCliRunner {
    fn name(&self) -> &'static str {
        "goose"
    }

    fn display_name(&self) -> &'static str {
        "Goose CLI"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip_all, fields(runner = "goose"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_user_prompt(&request.messages);

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
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--session-id", sid, "--resume"]);
            }
        }

        let output = run_cli_command(&mut cmd, self.config.timeout, MAX_OUTPUT_BYTES).await?;

        if output.exit_code != 0 {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            warn!(
                exit_code = output.exit_code,
                stdout_len = output.stdout.len(),
                stderr_len = output.stderr.len(),
                stdout_preview = %stdout.chars().take(500).collect::<String>(),
                stderr_preview = %stderr.chars().take(500).collect::<String>(),
                "Goose CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "goose",
                format!("goose exited with code {}: {detail}", output.exit_code),
            ));
        }

        Self::parse_json_response(&output.stdout)
    }

    #[instrument(skip_all, fields(runner = "goose"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prompt = build_user_prompt(&request.messages);

        let mut cmd = self.build_command_base("stream-json");
        cmd.args(["-i", "-"]);

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--session-id", sid, "--resume"]);
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
        tokio::spawn(async move {
            let _ = stdin.write_all(prompt.as_bytes()).await;
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

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("Goose CLI health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "Goose CLI health check failed"
            );
            Ok(false)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
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
