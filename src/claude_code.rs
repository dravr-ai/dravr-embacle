// ABOUTME: Claude Code CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `claude` CLI with JSON output parsing and session management
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
    TokenUsage,
};
use async_trait::async_trait;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::{extract_system_message, prepare_user_prompt};
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

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
    base: CliRunnerBase,
}

impl ClaudeCodeRunner {
    /// Create a new Claude Code runner with the given configuration
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
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["-p", prompt, "--output-format", output_format]);

        // stream-json requires --verbose flag in Claude Code CLI
        if output_format == "stream-json" {
            cmd.arg("--verbose");
        }

        if let Some(sys) = system_prompt {
            cmd.args(["--system-prompt", sys]);
        }

        let model = self
            .base
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        cmd.args(["--model", model]);

        // Disable Claude Code's native MCP servers so it uses our text-based
        // tool catalog injected via the system prompt instead.
        cmd.args(["--strict-mcp-config", "{}"]);

        for arg in &self.base.config.extra_args {
            cmd.arg(arg);
        }

        if let Ok(policy) = build_policy(
            self.base.config.working_directory.as_deref(),
            &self.base.config.allowed_env_keys,
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
            tool_calls: None,
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for ClaudeCodeRunner {
    crate::delegate_provider_base!(
        "claude-code",
        "Claude Code CLI",
        LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS
    );

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        let mut cmd = self.build_command(prompt, system, "json", request.max_tokens);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
            }
        }

        let model_name = request
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        debug!(
            binary = %self.base.config.binary_path.display(),
            model = model_name,
            has_system_prompt = system.is_some(),
            prompt_len = prompt.len(),
            "Spawning claude CLI"
        );

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "claude-code")?;

        let (response, session_id) = Self::parse_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        let mut cmd = self.build_command(prompt, system, "stream-json", request.max_tokens);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn test_parse_response_valid_json() {
        let json = br#"{"result":"Hello world","is_error":false,"session_id":"abc123","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion

        assert_eq!(response.content, "Hello world");
        assert_eq!(session_id, Some("abc123".to_owned()));
        assert_eq!(response.model, "claude-code");
        let usage = response.usage.unwrap(); // Safe: test assertion
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_response_error_flag() {
        let json = br#"{"result":"rate limited","is_error":true}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();

        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("rate limited"));
    }

    #[test]
    fn test_parse_response_missing_optional_fields() {
        let json = br#"{"result":"hi","is_error":false}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion

        assert_eq!(response.content, "hi");
        assert!(session_id.is_none());
        assert!(response.usage.is_none());
    }

    #[test]
    fn test_parse_response_null_result() {
        let json = br#"{"is_error":false}"#;
        let (response, _) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion
        assert_eq!(response.content, "");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let json = b"not json at all";
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Internal);
    }
}
