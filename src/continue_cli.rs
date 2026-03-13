// ABOUTME: Continue CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `cn` CLI with JSON output parsing; no streaming support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
use tokio::process::Command;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::run_cli_command;
use crate::prompt::prepare_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};

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
    base: CliRunnerBase,
}

impl ContinueCliRunner {
    /// Create a new Continue CLI runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            base: CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS),
        }
    }

    /// Store a session marker for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        self.base.set_session(key, session_id).await;
    }

    /// Build the command with all arguments
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["-p", "--format", "json", prompt]);

        if let Some(model) = self.base.config.model.as_deref() {
            cmd.args(["--model", model]);
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
                    tool_calls: None,
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
            tool_calls: None,
        })
    }
}

#[async_trait]
impl LlmProvider for ContinueCliRunner {
    crate::delegate_provider_base!("continue", "Continue CLI", LlmCapabilities::empty());

    #[instrument(skip_all, fields(runner = "continue"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if self.base.get_session(model).await.is_some() {
                cmd.arg("--resume");
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "continue")?;

        let response = Self::parse_json_response(&output.stdout)?;

        // Mark session as active for this model key (Continue uses `--resume` flag)
        if let Some(model) = &request.model {
            self.base.set_session(model, "active").await;
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
