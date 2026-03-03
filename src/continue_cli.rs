// ABOUTME: Continue CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `cn` CLI with JSON output parsing; no streaming support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::any::Any;
use std::collections::HashMap;
use std::str;
use std::sync::Arc;
use std::time::Duration;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::run_cli_command;
use crate::prompt::build_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};

/// Maximum output size for a single Continue CLI invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

/// Default model for Continue CLI (provider-agnostic)
const DEFAULT_MODEL: &str = "auto";

/// Fallback model list (Continue delegates to whatever backend the user configured)
const FALLBACK_MODELS: &[&str] = &["auto"];

/// Continue CLI runner
///
/// Implements `LlmProvider` by delegating to the `cn` binary with
/// `-p --format json` for non-interactive JSON output. Streaming is
/// not supported; `complete_stream()` wraps `complete()` in a single-chunk
/// stream via `tokio_stream::once`.
pub struct ContinueCliRunner {
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl ContinueCliRunner {
    /// Create a new Continue CLI runner with the given configuration
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

    /// Store a session marker for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        let mut sessions = self.session_ids.lock().await;
        sessions.insert(key.to_owned(), session_id.to_owned());
    }

    /// Build the command with all arguments
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.args(["-p", "--format", "json", prompt]);

        if let Some(model) = self.config.model.as_deref() {
            cmd.args(["--model", model]);
        }

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

    /// Parse JSON output from `cn -p --format json`
    ///
    /// Continue may emit lifecycle NDJSON lines (status/info events) before the
    /// actual response object. We scan lines for the first object containing a
    /// `"response"` field.
    fn parse_json_response(raw: &[u8]) -> Result<ChatResponse, RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Continue CLI output is not valid UTF-8: {e}"))
        })?;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Skip lifecycle events (status/info lines)
            if value.get("response").is_some() {
                let content = value
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();

                return Ok(ChatResponse {
                    content,
                    model: "continue".to_owned(),
                    usage: None,
                    finish_reason: Some("stop".to_owned()),
                    warnings: None,
                });
            }
        }

        // No response object found; return empty content
        Ok(ChatResponse {
            content: String::new(),
            model: "continue".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
        })
    }
}

#[async_trait]
impl LlmProvider for ContinueCliRunner {
    fn name(&self) -> &'static str {
        "continue"
    }

    fn display_name(&self) -> &'static str {
        "Continue CLI"
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

    #[instrument(skip_all, fields(runner = "continue"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_user_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt);

        if let Some(model) = &request.model {
            let sessions = self.session_ids.lock().await;
            if sessions.contains_key(model) {
                cmd.arg("--resume");
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
                "Continue CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "continue",
                format!("cn exited with code {}: {detail}", output.exit_code),
            ));
        }

        let response = Self::parse_json_response(&output.stdout)?;

        // Mark session as active for this model key (Continue uses `--resume` flag)
        if let Some(model) = &request.model {
            self.set_session(model, "active").await;
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "continue"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        // Continue CLI does not support streaming; wrap `complete()` as a single chunk
        let response = self.complete(request).await?;
        let chunk = StreamChunk {
            delta: response.content,
            is_final: true,
            finish_reason: Some("stop".to_owned()),
        };
        Ok(Box::pin(tokio_stream::once(Ok(chunk))))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("Continue CLI health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "Continue CLI health check failed"
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
        let json = br#"{"response":"hello from continue","status":"success","note":"completed"}"#;
        let resp = ContinueCliRunner::parse_json_response(json).unwrap();
        assert_eq!(resp.content, "hello from continue");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_json_response_skips_lifecycle_events() {
        let ndjson = br#"{"status":"info","message":"Auto-compacting triggered"}
{"response":"actual answer","status":"success"}"#;
        let resp = ContinueCliRunner::parse_json_response(ndjson).unwrap();
        assert_eq!(resp.content, "actual answer");
    }

    #[test]
    fn test_parse_json_response_empty_output() {
        let empty = b"";
        let resp = ContinueCliRunner::parse_json_response(empty).unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("cn"));
        let runner = ContinueCliRunner::new(config);
        assert_eq!(runner.default_model(), "auto");
    }

    #[test]
    fn test_capabilities_no_streaming() {
        let config = RunnerConfig::new(PathBuf::from("cn"));
        let runner = ContinueCliRunner::new(config);
        assert!(!runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("cn"));
        let runner = ContinueCliRunner::new(config);
        assert_eq!(runner.name(), "continue");
        assert_eq!(runner.display_name(), "Continue CLI");
    }
}
