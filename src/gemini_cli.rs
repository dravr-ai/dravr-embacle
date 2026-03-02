// ABOUTME: Gemini CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `gemini` CLI with JSON/stream-JSON output parsing and session resume
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

/// Maximum output size for a single Gemini CLI invocation (50 MiB)
const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

/// Gemini CLI JSON response structure (`-o json`)
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    response: Option<String>,
    session_id: Option<String>,
    #[serde(default)]
    stats: Option<GeminiStats>,
}

/// Aggregated stats from Gemini CLI output (field names match external JSON schema)
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)]
struct GeminiStats {
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

/// Default model for Gemini CLI
const DEFAULT_MODEL: &str = "gemini-2.5-flash";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"];

/// Gemini CLI runner
///
/// Implements `LlmProvider` by delegating to the `gemini` binary
/// with `-o json` for complete responses and `-o stream-json` for
/// streaming. Uses `-y` (yolo mode) to auto-approve tool usage.
pub struct GeminiCliRunner {
    config: RunnerConfig,
    default_model: String,
    available_models: Vec<String>,
    session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl GeminiCliRunner {
    /// Create a new Gemini CLI runner with the given configuration
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
        cmd.args(["-p", prompt, "-o", output_format]);

        // -y (yolo mode) auto-approves tool usage
        cmd.arg("-y");

        let model = self
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.default_model());
        cmd.args(["-m", model]);

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

    /// Parse JSONL output from Gemini CLI for non-streaming complete mode.
    ///
    /// Scans through JSONL lines looking for assistant messages and result stats.
    fn parse_jsonl_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Gemini CLI output is not valid UTF-8: {e}"))
        })?;

        // Try single JSON object first
        if let Ok(parsed) = serde_json::from_str::<GeminiResponse>(text) {
            let content = parsed.response.unwrap_or_default();
            let usage = parsed.stats.map(|s| {
                let input = s.input_tokens.unwrap_or(0);
                let output = s.output_tokens.unwrap_or(0);
                let total = s.total_tokens.unwrap_or(input + output);
                TokenUsage {
                    prompt_tokens: input,
                    completion_tokens: output,
                    total_tokens: total,
                }
            });
            return Ok((
                ChatResponse {
                    content,
                    model: "gemini".to_owned(),
                    usage,
                    finish_reason: Some("stop".to_owned()),
                },
                parsed.session_id,
            ));
        }

        // Fall back to JSONL parsing
        let mut content_parts: Vec<String> = Vec::new();
        let mut session_id: Option<String> = None;
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
                "init" => {
                    if let Some(sid) = value.get("session_id").and_then(|v| v.as_str()) {
                        session_id = Some(sid.to_owned());
                    }
                }
                "message" => {
                    let role = value.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    if role == "assistant" {
                        if let Some(c) = value.get("content").and_then(|v| v.as_str()) {
                            content_parts.push(c.to_owned());
                        }
                    }
                }
                "result" => {
                    if let Some(stats) = value.get("stats") {
                        let input = stats
                            .get("input_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let output = stats
                            .get("output_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let total = stats
                            .get("total_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(input + output);
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            usage = Some(TokenUsage {
                                prompt_tokens: input as u32,
                                completion_tokens: output as u32,
                                total_tokens: total as u32,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        let content = content_parts.join("");
        Ok((
            ChatResponse {
                content,
                model: "gemini".to_owned(),
                usage,
                finish_reason: Some("stop".to_owned()),
            },
            session_id,
        ))
    }
}

#[async_trait]
impl LlmProvider for GeminiCliRunner {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn display_name(&self) -> &'static str {
        "Gemini CLI"
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

    #[instrument(skip_all, fields(runner = "gemini"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        if request.temperature.is_some() || request.max_tokens.is_some() {
            debug!(
                temperature = ?request.temperature,
                max_tokens = ?request.max_tokens,
                "Gemini CLI does not support temperature or max_tokens; ignoring",
            );
        }

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
                "Gemini CLI failed"
            );
            let detail = if stderr.is_empty() { &stdout } else { &stderr };
            return Err(RunnerError::external_service(
                "gemini",
                format!("gemini exited with code {}: {detail}", output.exit_code),
            ));
        }

        let (response, session_id) = Self::parse_jsonl_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "gemini"))]
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
            RunnerError::internal(format!("Failed to spawn gemini for streaming: {e}"))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            RunnerError::internal("Failed to capture gemini stdout for streaming")
        })?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading gemini stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                RunnerError::internal(format!("Invalid JSON in gemini stream: {e}"))
            })?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "message" => {
                    let role = value.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    if role == "assistant" {
                        Ok(StreamChunk {
                            delta: value
                                .get("content")
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
                "result" => Ok(StreamChunk {
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

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("Gemini CLI health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "Gemini CLI health check failed"
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
    fn test_parse_single_json_response() {
        let json = br#"{"session_id":"abc123","response":"hello from gemini","stats":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#;
        let (resp, sid) = GeminiCliRunner::parse_jsonl_response(json).unwrap();
        assert_eq!(resp.content, "hello from gemini");
        assert_eq!(sid, Some("abc123".to_owned()));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_single_json_response_no_stats() {
        let json = br#"{"response":"hello"}"#;
        let (resp, sid) = GeminiCliRunner::parse_jsonl_response(json).unwrap();
        assert_eq!(resp.content, "hello");
        assert!(sid.is_none());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn test_parse_jsonl_response() {
        let jsonl = b"
{\"type\":\"init\",\"session_id\":\"sess-42\",\"model\":\"auto-gemini-3\"}
{\"type\":\"message\",\"role\":\"user\",\"content\":\"hi\"}
{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"hello from gemini\",\"delta\":true}
{\"type\":\"result\",\"status\":\"success\",\"stats\":{\"total_tokens\":8628,\"input_tokens\":100,\"output_tokens\":50}}
";
        let (resp, sid) = GeminiCliRunner::parse_jsonl_response(jsonl).unwrap();
        assert_eq!(resp.content, "hello from gemini");
        assert_eq!(sid, Some("sess-42".to_owned()));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 8628);
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("gemini"));
        let runner = GeminiCliRunner::new(config);
        assert_eq!(runner.default_model(), "gemini-2.5-flash");
    }

    #[test]
    fn test_custom_model() {
        let config = RunnerConfig::new(PathBuf::from("gemini")).with_model("gemini-2.5-pro");
        let runner = GeminiCliRunner::new(config);
        assert_eq!(runner.default_model(), "gemini-2.5-pro");
    }

    #[test]
    fn test_available_models() {
        let config = RunnerConfig::new(PathBuf::from("gemini"));
        let runner = GeminiCliRunner::new(config);
        let models = runner.available_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"gemini-2.5-flash".to_owned()));
        assert!(models.contains(&"gemini-2.5-pro".to_owned()));
        assert!(models.contains(&"gemini-2.0-flash".to_owned()));
    }

    #[test]
    fn test_capabilities() {
        let config = RunnerConfig::new(PathBuf::from("gemini"));
        let runner = GeminiCliRunner::new(config);
        assert!(runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("gemini"));
        let runner = GeminiCliRunner::new(config);
        assert_eq!(runner.name(), "gemini");
        assert_eq!(runner.display_name(), "Gemini CLI");
    }
}
