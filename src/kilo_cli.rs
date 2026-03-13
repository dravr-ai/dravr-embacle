// ABOUTME: Kilo Code CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `kilo` CLI with NDJSON output parsing, token tracking, and streaming support
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
use crate::prompt::prepare_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Default model for Kilo CLI (uses Kilo Gateway routing)
const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-4-6";

/// Fallback model list (Kilo supports 500+ models via its Gateway)
const FALLBACK_MODELS: &[&str] = &[
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "openai/gpt-5.4",
];

/// Token counts from Kilo NDJSON `step_finish` events
#[derive(Debug, Deserialize)]
struct KiloTokens {
    input: Option<u64>,
    output: Option<u64>,
    reasoning: Option<u64>,
    total: Option<u64>,
}

/// Kilo Code CLI runner
///
/// Implements `LlmProvider` by delegating to the `kilo` binary with
/// `run --auto --format json`. Output is NDJSON with event types:
/// `text` (content), `step_finish` (tokens/cost), `tool_use`, `error`.
/// Session resume via `--session <id>`.
pub struct KiloCliRunner {
    base: CliRunnerBase,
}

impl KiloCliRunner {
    /// Create a new Kilo CLI runner with the given configuration
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

    /// Build the command with all arguments
    fn build_command(&self, prompt: &str) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["run", "--auto", "--format", "json", prompt]);

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

    /// Parse NDJSON output from `kilo run --format json`
    ///
    /// Kilo emits NDJSON lines with `type` discriminator:
    /// - `text` — content in `part.text`
    /// - `step_finish` — finish reason in `part.reason`, token counts in `part.tokens`, cost in `part.cost`
    /// - `error` — error info in `error.name` and `error.data.message`
    /// - `step_start`, `tool_use`, `reasoning` — ignored
    ///
    /// The `sessionID` from any line is captured for session resumption.
    fn parse_ndjson_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Kilo CLI output is not valid UTF-8: {e}"))
        })?;

        let mut content_parts: Vec<String> = Vec::new();
        let mut usage: Option<TokenUsage> = None;
        let mut session_id: Option<String> = None;
        let mut finish_reason: Option<String> = None;
        let mut error_message: Option<String> = None;

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
                        if let Ok(t) = serde_json::from_value::<KiloTokens>(tokens.clone()) {
                            #[allow(clippy::cast_possible_truncation)]
                            {
                                let input = t.input.unwrap_or(0);
                                let output = t.output.unwrap_or(0);
                                let reasoning = t.reasoning.unwrap_or(0);
                                let total = t.total.unwrap_or(input + output + reasoning);
                                usage = Some(TokenUsage {
                                    prompt_tokens: input as u32,
                                    completion_tokens: output as u32,
                                    total_tokens: total as u32,
                                });
                            }
                        }
                    }
                }
                "error" => {
                    let msg = value
                        .pointer("/error/data/message")
                        .and_then(|v| v.as_str())
                        .or_else(|| value.pointer("/error/name").and_then(|v| v.as_str()));
                    if let Some(m) = msg {
                        error_message = Some(m.to_owned());
                    }
                }
                _ => {}
            }
        }

        if let Some(err) = error_message {
            if content_parts.is_empty() {
                return Err(RunnerError::external_service("kilo", err));
            }
        }

        let content = content_parts.join("");

        let response = ChatResponse {
            content,
            model: "kilo".to_owned(),
            usage,
            finish_reason: finish_reason.or_else(|| Some("stop".to_owned())),
            warnings: None,
            tool_calls: None,
        };

        Ok((response, session_id))
    }
}

#[async_trait]
impl LlmProvider for KiloCliRunner {
    crate::delegate_provider_base!("kilo", "Kilo Code CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "kilo"))]
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
        self.base.check_exit_code(&output, "kilo")?;

        let (response, session_id) = Self::parse_ndjson_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "kilo"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let mut cmd = self.build_command(prompt);

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--session", &sid]);
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn kilo for streaming: {e}"))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture kilo stdout for streaming"))?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading kilo stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => {
                    return Ok(StreamChunk {
                        delta: String::new(),
                        is_final: false,
                        finish_reason: None,
                    });
                }
            };

            let line_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match line_type {
                "text" => {
                    let delta = value
                        .pointer("/part/text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    Ok(StreamChunk {
                        delta,
                        is_final: false,
                        finish_reason: None,
                    })
                }
                "step_finish" => {
                    let reason = value
                        .pointer("/part/reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stop")
                        .to_owned();
                    Ok(StreamChunk {
                        delta: String::new(),
                        is_final: true,
                        finish_reason: Some(reason),
                    })
                }
                "error" => {
                    let msg = value
                        .pointer("/error/data/message")
                        .and_then(|v| v.as_str())
                        .or_else(|| value.pointer("/error/name").and_then(|v| v.as_str()))
                        .unwrap_or("unknown error");
                    Err(RunnerError::external_service("kilo", msg))
                }
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
    use std::path::PathBuf;

    #[test]
    fn test_parse_ndjson_response_basic() {
        let ndjson = br#"{"type":"step_start","timestamp":1710000000000,"sessionID":"ses_kilo1","part":{"type":"step-start"}}
{"type":"text","timestamp":1710000001000,"sessionID":"ses_kilo1","part":{"type":"text","text":"Hello from Kilo!"}}
{"type":"step_finish","timestamp":1710000002000,"sessionID":"ses_kilo1","part":{"type":"step-finish","reason":"endTurn","cost":0.0042,"tokens":{"total":1500,"input":1000,"output":400,"reasoning":100}}}"#;

        let (resp, sid) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "Hello from Kilo!");
        assert_eq!(sid, Some("ses_kilo1".to_owned()));
        assert_eq!(resp.finish_reason, Some("endTurn".to_owned()));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.completion_tokens, 400);
        assert_eq!(usage.total_tokens, 1500);
    }

    #[test]
    fn test_parse_ndjson_response_multiple_text_parts() {
        let ndjson = br#"{"type":"text","sessionID":"ses_k2","part":{"type":"text","text":"Hello "}}
{"type":"text","sessionID":"ses_k2","part":{"type":"text","text":"World"}}
{"type":"step_finish","sessionID":"ses_k2","part":{"type":"step-finish","reason":"stop","tokens":{"total":100,"input":80,"output":20}}}"#;

        let (resp, _) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "Hello World");
    }

    #[test]
    fn test_parse_ndjson_response_empty_output() {
        let ndjson = b"";
        let (resp, sid) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "");
        assert!(sid.is_none());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_ndjson_response_no_tokens() {
        let ndjson = br#"{"type":"text","sessionID":"ses_k3","part":{"type":"text","text":"OK"}}
{"type":"step_finish","sessionID":"ses_k3","part":{"type":"step-finish","reason":"stop"}}"#;

        let (resp, _) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "OK");
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_ndjson_response_with_error() {
        let ndjson = br#"{"type":"error","timestamp":1710000000000,"sessionID":"ses_k4","error":{"name":"APIError","data":{"message":"Rate limit exceeded","statusCode":429,"isRetryable":true}}}"#;

        let result = KiloCliRunner::parse_ndjson_response(ndjson);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Rate limit exceeded"));
    }

    #[test]
    fn test_parse_ndjson_response_error_with_content() {
        let ndjson = br#"{"type":"text","sessionID":"ses_k5","part":{"type":"text","text":"partial response"}}
{"type":"error","sessionID":"ses_k5","error":{"name":"ContextOverflowError","data":{"message":"context too long"}}}"#;

        let (resp, _) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "partial response");
    }

    #[test]
    fn test_parse_ndjson_response_with_tool_use() {
        let ndjson = br#"{"type":"step_start","timestamp":1710000000000,"sessionID":"ses_k6","part":{"type":"step-start"}}
{"type":"tool_use","timestamp":1710000001000,"sessionID":"ses_k6","part":{"type":"tool","tool":"bash","state":{"status":"completed","input":{"command":"ls"},"output":"file.rs"}}}
{"type":"text","timestamp":1710000002000,"sessionID":"ses_k6","part":{"type":"text","text":"Listed files."}}
{"type":"step_finish","timestamp":1710000003000,"sessionID":"ses_k6","part":{"type":"step-finish","reason":"endTurn","tokens":{"input":500,"output":50}}}"#;

        let (resp, sid) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "Listed files.");
        assert_eq!(sid, Some("ses_k6".to_owned()));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 500);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 550);
    }

    #[test]
    fn test_parse_ndjson_response_with_reasoning() {
        let ndjson = br#"{"type":"text","sessionID":"ses_k7","part":{"type":"text","text":"result"}}
{"type":"step_finish","sessionID":"ses_k7","part":{"type":"step-finish","reason":"stop","tokens":{"input":200,"output":50,"reasoning":100,"total":350}}}"#;

        let (resp, _) = KiloCliRunner::parse_ndjson_response(ndjson).unwrap();
        assert_eq!(resp.content, "result");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.total_tokens, 350);
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("kilo"));
        let runner = KiloCliRunner::new(config);
        assert_eq!(runner.default_model(), "anthropic/claude-sonnet-4-6");
    }

    #[test]
    fn test_capabilities() {
        let config = RunnerConfig::new(PathBuf::from("kilo"));
        let runner = KiloCliRunner::new(config);
        assert!(runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("kilo"));
        let runner = KiloCliRunner::new(config);
        assert_eq!(runner.name(), "kilo");
        assert_eq!(runner.display_name(), "Kilo Code CLI");
    }
}
