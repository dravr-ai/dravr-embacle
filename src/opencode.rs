// ABOUTME: `OpenCode` CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `opencode` CLI with JSON output parsing (no streaming support)
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::any::Any;
use std::collections::HashMap;
use std::str;
use std::sync::Arc;
use std::time::Duration;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, TokenUsage,
};
use async_trait::async_trait;
use serde::Deserialize;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::run_cli_command;
use crate::prompt::build_prompt;
use crate::sandbox::{apply_sandbox, build_policy};

/// Maximum output size for a single `OpenCode` invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

/// `OpenCode` CLI response JSON structure
#[derive(Debug, Deserialize)]
struct OpenCodeResponse {
    result: Option<String>,
    #[serde(default)]
    is_error: bool,
    session_id: Option<String>,
    usage: Option<OpenCodeUsage>,
}

/// Token usage from `OpenCode` CLI
#[derive(Debug, Deserialize)]
struct OpenCodeUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

/// Default model for `OpenCode`
const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &[
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "openai/gpt-5",
];

/// `OpenCode` CLI runner
///
/// Implements `LlmProvider` by delegating to the `opencode` binary with
/// `--format json`. Models use `provider/model` format (e.g.
/// `anthropic/claude-sonnet-4`). Streaming is not supported.
pub struct OpenCodeRunner {
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl OpenCodeRunner {
    /// Create a new `OpenCode` runner with the given configuration
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
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.args(["run", prompt, "--format", "json"]);

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

    /// Parse an `OpenCode` JSON response into a `ChatResponse`
    fn parse_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("OpenCode output is not valid UTF-8: {e}"))
        })?;

        let parsed: OpenCodeResponse = serde_json::from_str(text).map_err(|e| {
            RunnerError::internal(format!("Failed to parse OpenCode JSON response: {e}"))
        })?;

        if parsed.is_error {
            return Err(RunnerError::external_service(
                "opencode",
                parsed
                    .result
                    .as_deref()
                    .unwrap_or("Unknown error from OpenCode"),
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
            model: "opencode".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for OpenCodeRunner {
    fn name(&self) -> &'static str {
        "opencode"
    }

    fn display_name(&self) -> &'static str {
        "OpenCode CLI"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::empty()
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip_all, fields(runner = "opencode"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt);

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if let Some(sid) = sessions.get(model) {
                cmd.args(["--session", sid]);
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
                "OpenCode CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "opencode",
                format!("opencode exited with code {}: {detail}", output.exit_code),
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

    #[instrument(skip_all, fields(runner = "opencode"))]
    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal(
            "OpenCode CLI does not support streaming responses",
        ))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("OpenCode health check passed");
            Ok(true)
        } else {
            warn!(exit_code = output.exit_code, "OpenCode health check failed");
            Ok(false)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
