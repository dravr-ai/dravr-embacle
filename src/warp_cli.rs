// ABOUTME: Warp terminal `oz` CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `oz` CLI with NDJSON output parsing and conversation session support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};
use async_trait::async_trait;
use tokio::process::Command;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::run_cli_command;
use crate::prompt::prepare_prompt;
use crate::sandbox::{apply_sandbox, build_policy};

/// Default model for the Warp `oz` CLI
const DEFAULT_MODEL: &str = "auto";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["auto", "gpt-4.1", "claude-sonnet-4-20250514"];

/// Warp terminal `oz` CLI runner
///
/// Implements `LlmProvider` by delegating to the `oz` binary with
/// `agent run --output-format json`. The CLI emits NDJSON lines with
/// `type` discriminators: `system`, `agent`, `agent_reasoning`,
/// `tool_call`, and `tool_result`. Streaming is not supported.
pub struct WarpCliRunner {
    base: CliRunnerBase,
}

impl WarpCliRunner {
    /// Create a new Warp CLI runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            base: CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS),
        }
    }

    /// Store a session ID (conversation ID) for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        self.base.set_session(key, session_id).await;
    }

    /// Build the base command with common arguments
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args([
            "agent",
            "run",
            "--prompt",
            prompt,
            "--output-format",
            "json",
        ]);

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

    /// Parse NDJSON output from `oz agent run --output-format json` into a `ChatResponse`.
    ///
    /// The `oz` CLI emits NDJSON lines with `type` discriminator:
    /// - `system`           — `conversation_id` for session continuation
    /// - `agent`            — final response content in `text`
    /// - `agent_reasoning`  — model reasoning in `text` (ignored for content)
    /// - `tool_call`        — tool invocation with `tool` and `command` (ignored)
    /// - `tool_result`      — tool output with `tool`, `status`, `exit_code`, `output` (ignored)
    ///
    /// The `conversation_id` from the `system` event is captured for session resumption.
    fn parse_ndjson_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Warp oz output is not valid UTF-8: {e}"))
        })?;

        let mut content_parts: Vec<String> = Vec::new();
        let mut conversation_id: Option<String> = None;

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
                "system" => {
                    if let Some(cid) = value.get("conversation_id").and_then(|v| v.as_str()) {
                        conversation_id = Some(cid.to_owned());
                    }
                }
                "agent" => {
                    if let Some(t) = value.get("text").and_then(|v| v.as_str()) {
                        content_parts.push(t.to_owned());
                    }
                }
                // agent_reasoning, tool_call, tool_result — not included in response content
                _ => {}
            }
        }

        let content = content_parts.join("");

        let response = ChatResponse {
            content,
            model: "warp-oz".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        };

        Ok((response, conversation_id))
    }
}

#[async_trait]
impl LlmProvider for WarpCliRunner {
    crate::delegate_provider_base!("warp_cli", "Warp oz CLI", LlmCapabilities::empty());

    #[instrument(skip_all, fields(runner = "warp_cli"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if let Some(cid) = self.base.get_session(model).await {
                cmd.args(["--conversation", &cid]);
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "warp_cli")?;

        let (response, conversation_id) = Self::parse_ndjson_response(&output.stdout)?;

        if let Some(cid) = conversation_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &cid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "warp_cli"))]
    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal(
            "Warp oz CLI does not support streaming responses",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_ndjson_response_basic() {
        let ndjson = br#"{"type":"system","event_type":"conversation_started","conversation_id":"conv_abc123"}
{"type":"agent_reasoning","text":"Let me respond to this."}
{"type":"agent","text":"PONG"}"#;

        let (resp, cid) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "PONG");
        assert_eq!(cid, Some("conv_abc123".to_owned()));
        assert_eq!(resp.finish_reason, Some("stop".to_owned()));
        assert_eq!(resp.model, "warp-oz");
    }

    #[test]
    fn test_parse_ndjson_response_with_tool_calls() {
        let ndjson = br#"{"type":"system","event_type":"conversation_started","conversation_id":"conv_xyz"}
{"type":"agent_reasoning","text":"I need to check files."}
{"type":"tool_call","tool":"run_command","command":"ls -la"}
{"type":"tool_result","tool":"run_command","status":"complete","exit_code":0,"output":"total 42\ndrwxr-xr-x  5 user  staff  160 Mar  9 10:00 ."}
{"type":"agent","text":"The directory contains 5 items."}"#;

        let (resp, cid) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "The directory contains 5 items.");
        assert_eq!(cid, Some("conv_xyz".to_owned()));
    }

    #[test]
    fn test_parse_ndjson_response_multiple_agent_lines() {
        let ndjson =
            br#"{"type":"system","event_type":"conversation_started","conversation_id":"conv_1"}
{"type":"agent","text":"Hello "}
{"type":"agent","text":"World"}"#;

        let (resp, _) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "Hello World");
    }

    #[test]
    fn test_parse_ndjson_response_empty_output() {
        let ndjson = b"";
        let (resp, cid) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "");
        assert!(cid.is_none());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_ndjson_response_no_conversation_id() {
        let ndjson = br#"{"type":"agent","text":"OK"}"#;

        let (resp, cid) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "OK");
        assert!(cid.is_none());
    }

    #[test]
    fn test_parse_ndjson_response_skips_invalid_json() {
        let ndjson = b"not json\n{\"type\":\"agent\",\"text\":\"OK\"}\nalso not json";

        let (resp, _) = WarpCliRunner::parse_ndjson_response(ndjson).unwrap(); // Safe: test assertion
        assert_eq!(resp.content, "OK");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("oz"));
        let runner = WarpCliRunner::new(config);
        assert_eq!(runner.default_model(), "auto");
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("oz"));
        let runner = WarpCliRunner::new(config);
        assert_eq!(runner.name(), "warp_cli");
        assert_eq!(runner.display_name(), "Warp oz CLI");
    }

    #[test]
    fn test_capabilities_no_streaming() {
        let config = RunnerConfig::new(PathBuf::from("oz"));
        let runner = WarpCliRunner::new(config);
        assert!(!runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_available_models() {
        let config = RunnerConfig::new(PathBuf::from("oz"));
        let runner = WarpCliRunner::new(config);
        let models = runner.available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"auto".to_owned()));
    }
}
