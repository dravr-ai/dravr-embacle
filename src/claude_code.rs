// ABOUTME: Claude Code CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `claude` CLI with JSON output parsing and session management
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
    TokenUsage,
};
use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::{build_user_prompt, extract_system_message};
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Maximum output size for a single Claude Code invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

/// Default model for Claude Code
const DEFAULT_MODEL: &str = "opus";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["sonnet", "opus", "haiku"];

/// Claude Code CLI response JSON structure
#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    result: Option<String>,
    #[serde(default)]
    is_error: bool,
    session_id: Option<String>,
    usage: Option<ClaudeUsage>,
}

/// Token usage from Claude Code CLI
#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Claude Code CLI runner
///
/// Implements `LlmProvider` by delegating to the `claude` binary with
/// `--output-format json` for structured responses and optional session
/// resumption.
pub struct ClaudeCodeRunner {
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl ClaudeCodeRunner {
    /// Create a new Claude Code runner with the given configuration
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

    /// Build the base command with common arguments
    ///
    /// When `max_tokens` is `Some`, the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env var
    /// is injected after sandbox application so the CLI limits its output length.
    fn build_command(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
        output_format: &str,
        max_tokens: Option<u32>,
    ) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.args(["-p", prompt, "--output-format", output_format]);

        // stream-json requires --verbose flag in Claude Code CLI
        if output_format == "stream-json" {
            cmd.arg("--verbose");
        }

        if let Some(sys) = system_prompt {
            cmd.args(["--system-prompt", sys]);
        }

        let model = self
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.default_model());
        cmd.args(["--model", model]);

        // Disable Claude Code's native MCP servers so it uses our text-based
        // tool catalog injected via the system prompt instead.
        cmd.args(["--strict-mcp-config", "{}"]);

        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }

        if let Ok(policy) = build_policy(
            self.config.working_directory.as_deref(),
            &self.config.allowed_env_keys,
        ) {
            apply_sandbox(&mut cmd, &policy);
            debug!(
                allowed_keys = ?policy.allowed_env_keys,
                cwd = %policy.working_directory.display(),
                "Sandbox applied to claude command"
            );
        } else {
            warn!("Failed to build sandbox policy, running with inherited env");
        }

        // Inject max output tokens after sandbox (env_clear) so the value persists
        if let Some(tokens) = max_tokens {
            cmd.env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", tokens.to_string());
        }

        cmd
    }

    /// Parse a Claude Code JSON response into a `ChatResponse`
    fn parse_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Claude Code output is not valid UTF-8: {e}"))
        })?;

        let parsed: ClaudeResponse = serde_json::from_str(text).map_err(|e| {
            RunnerError::internal(format!("Failed to parse Claude Code JSON response: {e}"))
        })?;

        if parsed.is_error {
            return Err(RunnerError::external_service(
                "claude-code",
                parsed
                    .result
                    .as_deref()
                    .unwrap_or("Unknown error from Claude Code"),
            ));
        }

        let content = parsed.result.unwrap_or_default();
        let usage = parsed.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens.unwrap_or(0),
            completion_tokens: u.output_tokens.unwrap_or(0),
            total_tokens: u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0),
        });

        let response = ChatResponse {
            content,
            model: "claude-code".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for ClaudeCodeRunner {
    fn name(&self) -> &'static str {
        "claude-code"
    }

    fn display_name(&self) -> &'static str {
        "Claude Code CLI"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::SYSTEM_MESSAGES | LlmCapabilities::STREAMING | LlmCapabilities::MAX_TOKENS
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prompt = build_user_prompt(&request.messages);

        let mut cmd = self.build_command(&prompt, system, "json", request.max_tokens);

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--resume", sid]);
            }
        }

        let model_name = request
            .model
            .as_deref()
            .unwrap_or_else(|| self.default_model());
        debug!(
            binary = %self.config.binary_path.display(),
            model = model_name,
            has_system_prompt = system.is_some(),
            prompt_len = prompt.len(),
            "Spawning claude CLI"
        );

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
                "Claude CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "claude-code",
                format!("claude exited with code {}: {detail}", output.exit_code),
            ));
        }

        let (response, session_id) = Self::parse_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prompt = build_user_prompt(&request.messages);

        let mut cmd = self.build_command(&prompt, system, "stream-json", request.max_tokens);

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--resume", sid]);
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn claude for streaming: {e}"))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            RunnerError::internal("Failed to capture claude stdout for streaming")
        })?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading claude stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                RunnerError::internal(format!("Invalid JSON in claude stream: {e}"))
            })?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "result" => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason: Some("stop".to_owned()),
                }),
                "assistant" => {
                    // Extract text from content array: message.content[].text where type == "text"
                    let text = value
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter(|item| {
                                    item.get("type").and_then(|t| t.as_str()) == Some("text")
                                })
                                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("")
                        })
                        .unwrap_or_default();
                    Ok(StreamChunk {
                        delta: text,
                        is_final: false,
                        finish_reason: None,
                    })
                }
                // system, rate_limit_event, and other event types are ignored
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
            debug!("Claude Code health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "Claude Code health check failed"
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

    #[test]
    fn test_parse_response_valid_json() {
        let json = br#"{"result":"Hello world","is_error":false,"session_id":"abc123","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap();

        assert_eq!(response.content, "Hello world");
        assert_eq!(session_id, Some("abc123".to_owned()));
        assert_eq!(response.model, "claude-code");
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_response_error_flag() {
        let json = br#"{"result":"rate limited","is_error":true}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();

        assert_eq!(err.kind, crate::types::ErrorKind::ExternalService);
        assert!(err.message.contains("rate limited"));
    }

    #[test]
    fn test_parse_response_missing_optional_fields() {
        let json = br#"{"result":"hi","is_error":false}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap();

        assert_eq!(response.content, "hi");
        assert!(session_id.is_none());
        assert!(response.usage.is_none());
    }

    #[test]
    fn test_parse_response_null_result() {
        let json = br#"{"is_error":false}"#;
        let (response, _) = ClaudeCodeRunner::parse_response(json).unwrap();
        assert_eq!(response.content, "");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let json = b"not json at all";
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();
        assert_eq!(err.kind, crate::types::ErrorKind::Internal);
    }
}
