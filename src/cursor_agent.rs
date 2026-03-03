// ABOUTME: Cursor Agent CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `cursor-agent` CLI with JSON output parsing and MCP approval
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
use crate::prompt::build_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Maximum output size for a single Cursor Agent invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

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
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl CursorAgentRunner {
    /// Create a new Cursor Agent runner with the given configuration
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
    fn build_command(&self, prompt: &str, output_format: &str) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.args(["-p", prompt, "--output-format", output_format]);

        // Cursor Agent always gets --approve-mcps
        cmd.arg("--approve-mcps");

        let model = self
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.default_model());
        cmd.args(["--model", model]);

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
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for CursorAgentRunner {
    fn name(&self) -> &'static str {
        "cursor-agent"
    }

    fn display_name(&self) -> &'static str {
        "Cursor Agent CLI"
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

    #[instrument(skip_all, fields(runner = "cursor_agent"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_user_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt, "json");

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--resume", sid]);
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
                "Cursor Agent CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "cursor-agent",
                format!(
                    "cursor-agent exited with code {}: {detail}",
                    output.exit_code
                ),
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

    #[instrument(skip_all, fields(runner = "cursor_agent"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prompt = build_user_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt, "stream-json");

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--resume", sid]);
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

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("Cursor Agent health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "Cursor Agent health check failed"
            );
            Ok(false)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
