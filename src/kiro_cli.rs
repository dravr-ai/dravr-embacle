// ABOUTME: Kiro CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `kiro-cli` binary with ANSI stripping and plain text parsing; no streaming support
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

/// Default model for Kiro CLI (delegates to Kiro's auto-selection)
const DEFAULT_MODEL: &str = "auto";

/// Available models exposed by the Kiro CLI
const FALLBACK_MODELS: &[&str] = &[
    "auto",
    "claude-sonnet-4.5",
    "claude-sonnet-4",
    "claude-haiku-4.5",
    "deepseek-3.2",
    "minimax-m2.1",
    "qwen3-coder-next",
];

/// Kiro CLI runner
///
/// Implements `LlmProvider` by delegating to the `kiro-cli` binary with
/// `chat --no-interactive --wrap never` for non-interactive plain text output.
/// ANSI escape codes are stripped from the response, and the `> ` prefix on
/// response lines is removed. Streaming is not supported; `complete_stream()`
/// wraps `complete()` in a single-chunk stream via `tokio_stream::once`.
pub struct KiroCliRunner {
    base: CliRunnerBase,
}

impl KiroCliRunner {
    /// Create a new Kiro CLI runner with the given configuration
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
        cmd.args([
            "chat",
            "--no-interactive",
            "--wrap",
            "never",
            "--trust-all-tools",
        ]);

        if let Some(model) = self.base.config.model.as_deref() {
            cmd.args(["--model", model]);
        }

        cmd.arg(prompt);

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

    /// Strip ANSI escape codes from text
    ///
    /// Removes all ANSI CSI sequences (e.g. `\x1b[0m`, `\x1b[31m`, `\x1b[?25h`)
    /// and OSC sequences (e.g. `\x1b]...BEL/ST`).
    fn strip_ansi(input: &str) -> String {
        let bytes = input.as_bytes();
        let mut output = Vec::with_capacity(bytes.len());
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() {
                match bytes[i + 1] {
                    // CSI sequence: \x1b[ ... <final byte>
                    b'[' => {
                        i += 2;
                        while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                            i += 1;
                        }
                        if i < bytes.len() {
                            i += 1; // skip final byte
                        }
                    }
                    // OSC sequence: \x1b] ... (terminated by BEL or ST)
                    b']' => {
                        i += 2;
                        while i < bytes.len() {
                            if bytes[i] == 0x07 {
                                i += 1;
                                break;
                            }
                            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                                i += 2;
                                break;
                            }
                            i += 1;
                        }
                    }
                    // Other escape sequences (2 bytes)
                    _ => {
                        i += 2;
                    }
                }
            } else {
                output.push(bytes[i]);
                i += 1;
            }
        }

        String::from_utf8_lossy(&output).into_owned()
    }

    /// Parse plain text output from `kiro-cli chat`
    ///
    /// Kiro CLI outputs response lines prefixed with `> ` and may contain ANSI
    /// escape codes. This method strips ANSI codes, removes the `> ` prefix from
    /// each line, and joins the cleaned lines.
    fn parse_text_response(raw: &[u8]) -> Result<ChatResponse, RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Kiro CLI output is not valid UTF-8: {e}"))
        })?;

        let cleaned = Self::strip_ansi(text);
        let content: String = cleaned
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.strip_prefix("> ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ChatResponse {
            content,
            model: "kiro".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }
}

#[async_trait]
impl LlmProvider for KiroCliRunner {
    crate::delegate_provider_base!("kiro", "Kiro CLI", LlmCapabilities::empty());

    #[instrument(skip_all, fields(runner = "kiro"))]
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
        self.base.check_exit_code(&output, "kiro")?;

        let response = Self::parse_text_response(&output.stdout)?;

        // Mark session as active for this model key (Kiro uses `--resume` flag)
        if let Some(model) = &request.model {
            self.base.set_session(model, "active").await;
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "kiro"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        // Kiro CLI does not support streaming; wrap `complete()` as a single chunk
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
    fn test_strip_ansi_color_codes() {
        let input = "\x1b[32m> Hello\x1b[0m";
        let stripped = KiroCliRunner::strip_ansi(input);
        assert_eq!(stripped, "> Hello");
    }

    #[test]
    fn test_strip_ansi_cursor_codes() {
        let input = "\x1b[?25h\x1b[?25lHello";
        let stripped = KiroCliRunner::strip_ansi(input);
        assert_eq!(stripped, "Hello");
    }

    #[test]
    fn test_strip_ansi_osc_sequences() {
        let input = "\x1b]0;title\x07Hello";
        let stripped = KiroCliRunner::strip_ansi(input);
        assert_eq!(stripped, "Hello");
    }

    #[test]
    fn test_strip_ansi_no_codes() {
        let input = "plain text";
        let stripped = KiroCliRunner::strip_ansi(input);
        assert_eq!(stripped, "plain text");
    }

    #[test]
    fn test_strip_ansi_empty() {
        assert_eq!(KiroCliRunner::strip_ansi(""), "");
    }

    #[test]
    fn test_parse_text_response_with_prefix() {
        let raw = b"> Paris is the capital of France.";
        let resp = KiroCliRunner::parse_text_response(raw).unwrap();
        assert_eq!(resp.content, "Paris is the capital of France.");
    }

    #[test]
    fn test_parse_text_response_multiline() {
        let raw = b"> Line one\n> Line two\n> Line three";
        let resp = KiroCliRunner::parse_text_response(raw).unwrap();
        assert_eq!(resp.content, "Line one\nLine two\nLine three");
    }

    #[test]
    fn test_parse_text_response_with_ansi() {
        let raw = b"\x1b[32m> Hello world\x1b[0m";
        let resp = KiroCliRunner::parse_text_response(raw).unwrap();
        assert_eq!(resp.content, "Hello world");
    }

    #[test]
    fn test_parse_text_response_empty_output() {
        let resp = KiroCliRunner::parse_text_response(b"").unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn test_parse_text_response_mixed_lines() {
        let raw = b"Some debug info\n> Actual response\n";
        let resp = KiroCliRunner::parse_text_response(raw).unwrap();
        assert_eq!(resp.content, "Some debug info\nActual response");
    }

    #[test]
    fn test_default_model() {
        let config = RunnerConfig::new(PathBuf::from("kiro-cli"));
        let runner = KiroCliRunner::new(config);
        assert_eq!(runner.default_model(), "auto");
    }

    #[test]
    fn test_available_models() {
        let config = RunnerConfig::new(PathBuf::from("kiro-cli"));
        let runner = KiroCliRunner::new(config);
        let models = runner.available_models();
        assert_eq!(models.len(), 7);
        assert!(models.contains(&"auto".to_owned()));
        assert!(models.contains(&"claude-sonnet-4".to_owned()));
        assert!(models.contains(&"deepseek-3.2".to_owned()));
    }

    #[test]
    fn test_capabilities_no_streaming() {
        let config = RunnerConfig::new(PathBuf::from("kiro-cli"));
        let runner = KiroCliRunner::new(config);
        assert!(!runner.capabilities().supports_streaming());
    }

    #[test]
    fn test_name_and_display() {
        let config = RunnerConfig::new(PathBuf::from("kiro-cli"));
        let runner = KiroCliRunner::new(config);
        assert_eq!(runner.name(), "kiro");
        assert_eq!(runner.display_name(), "Kiro CLI");
    }
}
