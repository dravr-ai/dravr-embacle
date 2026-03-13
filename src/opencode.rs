// ABOUTME: `OpenCode` CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `opencode` CLI with NDJSON output parsing (no streaming support)
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, TokenUsage,
};
use async_trait::async_trait;
use serde::Deserialize;
use tokio::process::Command;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::run_cli_command;
use crate::prompt::prepare_prompt;
use crate::sandbox::{apply_sandbox, build_policy};

/// Token counts from `OpenCode` NDJSON `step_finish` events.
#[derive(Debug, Deserialize)]
struct OpenCodeTokens {
    input: Option<u64>,
    output: Option<u64>,
    total: Option<u64>,
}

/// Default model for `OpenCode`
const DEFAULT_MODEL: &str = "github-copilot/claude-sonnet-4.6";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &[
    "github-copilot/claude-sonnet-4.6",
    "github-copilot/claude-opus-4.6",
    "github-copilot/gpt-5",
];

/// `OpenCode` CLI runner
///
/// Implements `LlmProvider` by delegating to the `opencode` binary with
/// `--format json`. Models use `provider/model` format (e.g.
/// `anthropic/claude-sonnet-4`). Streaming is not supported.
pub struct OpenCodeRunner {
    base: CliRunnerBase,
}

impl OpenCodeRunner {
    /// Create a new `OpenCode` runner with the given configuration
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
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["run", prompt, "--format", "json"]);

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

    /// Parse NDJSON output from `opencode run --format json` into a `ChatResponse`.
    ///
    /// `OpenCode` emits NDJSON lines with `type` discriminator:
    /// - `text`  — content in `part.text`
    /// - `step_finish` — finish reason in `part.reason`, token counts in `part.tokens`
    /// - `step_start` / other — ignored
    ///
    /// The `sessionID` from any line is captured for session resumption.
    fn parse_ndjson_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("OpenCode output is not valid UTF-8: {e}"))
        })?;

        let mut content_parts: Vec<String> = Vec::new();
        let mut usage: Option<TokenUsage> = None;
        let mut session_id: Option<String> = None;
        let mut finish_reason: Option<String> = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Capture session ID from any line
            if session_id.is_none() {
                if let Some(sid) = value.get("sessionID").and_then(|v| v.as_str()) {
                    session_id = Some(sid.to_owned());
                }
            }

            let line_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match line_type {
                "text" => {
                    if let Some(t) = value.pointer("/part/text").and_then(|v| v.as_str()) {
                        content_parts.push(t.to_owned());
                    }
                }
                "step_finish" => {
                    if let Some(reason) = value.pointer("/part/reason").and_then(|v| v.as_str()) {
                        finish_reason = Some(reason.to_owned());
                    }
                    if let Some(tokens) = value.pointer("/part/tokens") {
                        if let Ok(t) = serde_json::from_value::<OpenCodeTokens>(tokens.clone()) {
                            #[allow(clippy::cast_possible_truncation)]
                            {
                                let input = t.input.unwrap_or(0);
                                let output = t.output.unwrap_or(0);
                                let total = t.total.unwrap_or(input + output);
                                usage = Some(TokenUsage {
                                    prompt_tokens: input as u32,
                                    completion_tokens: output as u32,
                                    total_tokens: total as u32,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let content = content_parts.join("");

        let response = ChatResponse {
            content,
            model: "opencode".to_owned(),
            usage,
            finish_reason: finish_reason.or_else(|| Some("stop".to_owned())),
            warnings: None,
            tool_calls: None,
        };

        Ok((response, session_id))
    }
}

#[async_trait]
impl LlmProvider for OpenCodeRunner {
    crate::delegate_provider_base!("opencode", "OpenCode CLI", LlmCapabilities::empty());

    #[instrument(skip_all, fields(runner = "opencode"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--session", &sid]);
            }
        }

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "opencode")?;

        let (response, session_id) = Self::parse_ndjson_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &sid).await;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_ndjson_response_basic() {
        let ndjson = br#"{"type":"step_start","timestamp":1772896674815,"sessionID":"ses_abc123","part":{"type":"step-start"}}
{"type":"text","timestamp":1772896674817,"sessionID":"ses_abc123","part":{"type":"text","text":"PONG"}}
{"type":"step_finish","timestamp":1772896674834,"sessionID":"ses_abc123","part":{"type":"step-finish","reason":"stop","tokens":{"total":14976,"input":14963,"output":13,"reasoning":0}}}"#;

        let (resp, sid) = OpenCodeRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "PONG");
        assert_eq!(sid, Some("ses_abc123".to_owned()));
        assert_eq!(resp.finish_reason, Some("stop".to_owned()));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 14963);
        assert_eq!(usage.completion_tokens, 13);
        assert_eq!(usage.total_tokens, 14976);
    }

    #[test]
    fn test_parse_ndjson_response_multiple_text_parts() {
        let ndjson = br#"{"type":"text","sessionID":"ses_1","part":{"type":"text","text":"Hello "}}
{"type":"text","sessionID":"ses_1","part":{"type":"text","text":"World"}}
{"type":"step_finish","sessionID":"ses_1","part":{"type":"step-finish","reason":"stop","tokens":{"total":100,"input":80,"output":20}}}"#;

        let (resp, _) = OpenCodeRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "Hello World");
    }

    #[test]
    fn test_parse_ndjson_response_empty_output() {
        let ndjson = b"";
        let (resp, sid) = OpenCodeRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "");
        assert!(sid.is_none());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_ndjson_response_no_tokens() {
        let ndjson = br#"{"type":"text","sessionID":"ses_x","part":{"type":"text","text":"OK"}}
{"type":"step_finish","sessionID":"ses_x","part":{"type":"step-finish","reason":"stop"}}"#;

        let (resp, _) = OpenCodeRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "OK");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("opencode"));
        let runner = OpenCodeRunner::new(config);
        assert_eq!(runner.default_model(), "github-copilot/claude-sonnet-4.6");
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("opencode"));
        let runner = OpenCodeRunner::new(config);
        assert_eq!(runner.name(), "opencode");
        assert_eq!(runner.display_name(), "OpenCode CLI");
    }

    #[test]
    fn test_capabilities_no_streaming() {
        let config = RunnerConfig::new(PathBuf::from("opencode"));
        let runner = OpenCodeRunner::new(config);
        assert!(!runner.capabilities().supports_streaming());
    }
}
