// ABOUTME: Cursor Agent CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `cursor-agent` CLI with JSON output parsing and MCP approval
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
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::prepare_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Cursor Agent CLI response JSON structure
#[derive(Debug, Deserialize)]
struct CursorResponse {
    result: Option<String>,
    #[serde(default)]
    is_error: bool,
    session_id: Option<String>,
    usage: Option<CursorUsage>,
}

/// Token usage from Cursor Agent CLI
#[derive(Debug, Deserialize)]
struct CursorUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Default model for Cursor Agent
const DEFAULT_MODEL: &str = "sonnet-4";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["sonnet-4", "gpt-5", "gemini-2.5-pro"];

/// Cursor Agent CLI runner
///
/// Implements `LlmProvider` by delegating to the `cursor-agent` binary
/// with `--output-format json` and `--approve-mcps` for automatic MCP
/// server approval.
pub struct CursorAgentRunner {
    base: CliRunnerBase,
}

impl CursorAgentRunner {
    /// Create a new Cursor Agent runner with the given configuration
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
    fn build_command(&self, prompt: &str, output_format: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["-p", prompt, "--output-format", output_format]);

        // Cursor Agent always gets --approve-mcps
        cmd.arg("--approve-mcps");

        let model = self
            .base
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        cmd.args(["--model", model]);

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

    /// Parse a Cursor Agent JSON response into a `ChatResponse`
    fn parse_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Cursor Agent output is not valid UTF-8: {e}"))
        })?;

        let parsed: CursorResponse = serde_json::from_str(text).map_err(|e| {
            RunnerError::internal(format!("Failed to parse Cursor Agent JSON response: {e}"))
        })?;

        if parsed.is_error {
            return Err(RunnerError::external_service(
                "cursor-agent",
                parsed
                    .result
                    .as_deref()
                    .unwrap_or("Unknown error from Cursor Agent"),
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
            model: "cursor-agent".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for CursorAgentRunner {
    crate::delegate_provider_base!(
        "cursor-agent",
        "Cursor Agent CLI",
        LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS
    );

    #[instrument(skip_all, fields(runner = "cursor_agent"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt, "json");

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "cursor-agent")?;

        let (response, session_id) = Self::parse_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "cursor_agent"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt, "stream-json");

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn cursor-agent for streaming: {e}"))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            RunnerError::internal("Failed to capture cursor-agent stdout for streaming")
        })?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result.map_err(|e| {
                RunnerError::internal(format!("Error reading cursor-agent stream: {e}"))
            })?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                RunnerError::internal(format!("Invalid JSON in cursor-agent stream: {e}"))
            })?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "result" => Ok(StreamChunk {
                    delta: value
                        .get("result")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    is_final: true,
                    finish_reason: Some("stop".to_owned()),
                }),
                "content" => Ok(StreamChunk {
                    delta: value
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    is_final: false,
                    finish_reason: None,
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
