// ABOUTME: Codex CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `codex` CLI with JSONL output parsing in non-interactive exec mode
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
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::instrument;

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::build_user_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Default model for Codex CLI
const DEFAULT_MODEL: &str = "o4-mini";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["o4-mini", "o3", "gpt-4.1"];

/// Codex CLI runner
///
/// Implements `LlmProvider` by delegating to the `codex` binary with
/// `exec` subcommand for non-interactive mode. Uses `--json` for JSONL
/// output and `--full-auto` for automatic sandbox approval.
pub struct CodexCliRunner {
    base: CliRunnerBase,
}

impl CodexCliRunner {
    /// Create a new Codex CLI runner with the given configuration
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
        cmd.args(["exec", prompt, "--json", "--full-auto"]);

        let model = self
            .base
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        cmd.args(["-m", model]);

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

    /// Parse JSONL output from Codex CLI into a `ChatResponse`
    ///
    /// Scans through JSONL lines looking for `item.completed` events with
    /// `agent_message` type for content, and `turn.completed` for usage stats.
    fn parse_jsonl_response(raw: &[u8]) -> Result<ChatResponse, RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Codex CLI output is not valid UTF-8: {e}"))
        })?;

        let mut content_parts: Vec<String> = Vec::new();
        let mut usage: Option<TokenUsage> = None;

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
                "item.completed" => {
                    if let Some(item) = value.get("item") {
                        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if item_type == "agent_message" {
                            if let Some(text_content) = item.get("text").and_then(|v| v.as_str()) {
                                content_parts.push(text_content.to_owned());
                            }
                        }
                    }
                }
                "turn.completed" => {
                    if let Some(u) = value.get("usage") {
                        let input = u
                            .get("input_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let output = u
                            .get("output_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            usage = Some(TokenUsage {
                                prompt_tokens: input as u32,
                                completion_tokens: output as u32,
                                total_tokens: (input + output) as u32,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        let content = content_parts.join("");

        Ok(ChatResponse {
            content,
            model: "codex".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }
}

#[async_trait]
impl LlmProvider for CodexCliRunner {
    crate::delegate_provider_base!("codex", "Codex CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "codex"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_user_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt);

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "codex")?;

        Self::parse_jsonl_response(&output.stdout)
    }

    #[instrument(skip_all, fields(runner = "codex"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prompt = build_user_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn codex for streaming: {e}"))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture codex stdout for streaming"))?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading codex stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| RunnerError::internal(format!("Invalid JSON in codex stream: {e}")))?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "item.completed" => {
                    let item_type = value
                        .get("item")
                        .and_then(|v| v.get("type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if item_type == "agent_message" {
                        Ok(StreamChunk {
                            delta: value
                                .get("item")
                                .and_then(|v| v.get("text"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_owned(),
                            is_final: false,
                            finish_reason: None,
                        })
                    } else {
                        Ok(StreamChunk {
                            delta: String::new(),
                            is_final: false,
                            finish_reason: None,
                        })
                    }
                }
                "turn.completed" => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason: Some("stop".to_owned()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_jsonl_response_basic() {
        let jsonl = br#"{"type":"thread.started","thread_id":"t-123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"msg-1","type":"agent_message","text":"hello from codex"}}
{"type":"turn.completed","usage":{"input_tokens":11764,"output_tokens":22}}"#;

        let resp = CodexCliRunner::parse_jsonl_response(jsonl).unwrap();
        assert_eq!(resp.content, "hello from codex");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 11764);
        assert_eq!(usage.completion_tokens, 22);
        assert_eq!(usage.total_tokens, 11786);
    }

    #[test]
    fn test_parse_jsonl_response_empty() {
        let jsonl = br#"{"type":"thread.started","thread_id":"t-123"}
{"type":"turn.started"}
{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":0}}"#;

        let resp = CodexCliRunner::parse_jsonl_response(jsonl).unwrap();
        assert_eq!(resp.content, "");
        assert!(resp.usage.is_some());
    }

    #[test]
    fn test_parse_jsonl_response_multiple_messages() {
        let jsonl =
            br#"{"type":"item.completed","item":{"id":"1","type":"agent_message","text":"part1"}}
{"type":"item.completed","item":{"id":"2","type":"agent_message","text":"part2"}}
{"type":"turn.completed","usage":{"input_tokens":50,"output_tokens":10}}"#;

        let resp = CodexCliRunner::parse_jsonl_response(jsonl).unwrap();
        assert_eq!(resp.content, "part1part2");
    }

    #[test]
    fn test_parse_jsonl_skips_non_agent_items() {
        let jsonl =
            br#"{"type":"item.completed","item":{"id":"1","type":"tool_call","text":"ignored"}}
{"type":"item.completed","item":{"id":"2","type":"agent_message","text":"kept"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}"#;

        let resp = CodexCliRunner::parse_jsonl_response(jsonl).unwrap();
        assert_eq!(resp.content, "kept");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("codex"));
        let runner = CodexCliRunner::new(config);
        assert_eq!(runner.default_model(), "o4-mini");
    }

    #[test]
    fn test_custom_model() {
        let config = RunnerConfig::new(PathBuf::from("codex")).with_model("o3");
        let runner = CodexCliRunner::new(config);
        assert_eq!(runner.default_model(), "o3");
    }

    #[test]
    fn test_available_models() {
        let config = RunnerConfig::new(PathBuf::from("codex"));
        let runner = CodexCliRunner::new(config);
        let models = runner.available_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"o4-mini".to_owned()));
        assert!(models.contains(&"o3".to_owned()));
        assert!(models.contains(&"gpt-4.1".to_owned()));
    }

    #[test]
    fn test_capabilities() {
        let config = RunnerConfig::new(PathBuf::from("codex"));
        let runner = CodexCliRunner::new(config);
        assert!(runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("codex"));
        let runner = CodexCliRunner::new(config);
        assert_eq!(runner.name(), "codex");
        assert_eq!(runner.display_name(), "Codex CLI");
    }
}
