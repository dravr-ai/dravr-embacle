// ABOUTME: GitHub Copilot CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `copilot` CLI with plain-text output parsing and streaming support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::build_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Default model for Copilot CLI
const DEFAULT_MODEL: &str = "claude-opus-4.6-fast";

/// Fallback model list when `gh copilot models` discovery fails
const FALLBACK_MODELS: &[&str] = &[
    "claude-sonnet-4.6",
    "claude-opus-4.6",
    "claude-opus-4.6-fast",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-sonnet-4",
    "gemini-3-pro-preview",
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-4.1",
];

/// Discover available Copilot models by running `gh copilot models`.
///
/// Returns `None` if `gh` is not found, the command fails, or the output
/// cannot be parsed into a non-empty model list.
pub async fn discover_copilot_models() -> Option<Vec<String>> {
    let output = Command::new("gh")
        .args(["copilot", "models"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        debug!(
            exit_code = output.status.code().unwrap_or(-1),
            "gh copilot models failed, falling back to static list"
        );
        return None;
    }

    let stdout = str::from_utf8(&output.stdout).ok()?;
    let models: Vec<String> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    if models.is_empty() {
        debug!("gh copilot models returned empty output, falling back to static list");
        return None;
    }

    debug!(
        count = models.len(),
        "Discovered available Copilot models via gh copilot models"
    );
    Some(models)
}

/// Fallback model list when `gh copilot models` discovery fails.
///
/// Used by both `CopilotRunner` and `CopilotHeadlessRunner`.
pub fn copilot_fallback_models() -> Vec<String> {
    FALLBACK_MODELS.iter().map(|s| (*s).to_owned()).collect()
}

/// GitHub Copilot CLI runner
///
/// Implements `LlmProvider` by delegating to the `copilot` binary in
/// non-interactive mode (`-p`). Copilot CLI outputs plain text (no JSON
/// structure), so the raw stdout is captured as the response content.
/// System messages are embedded into the user prompt since Copilot CLI
/// has no `--system-prompt` flag.
pub struct CopilotRunner {
    base: CliRunnerBase,
}

impl CopilotRunner {
    /// Create a new Copilot CLI runner with the given configuration.
    ///
    /// Attempts to discover available models by running `gh copilot models`.
    /// Falls back to a static list if discovery fails.
    pub async fn new(config: RunnerConfig) -> Self {
        let mut base = CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS);
        if let Some(models) = discover_copilot_models().await {
            base.available_models = models;
        }
        Self { base }
    }

    /// Build the base command with common arguments
    fn build_command(&self, prompt: &str, silent: bool) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);

        // Non-interactive prompt mode
        cmd.args(["-p", prompt]);

        let model = self
            .base
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        cmd.args(["--model", model]);

        // Required for non-interactive mode
        cmd.arg("--allow-all-tools");

        // Disable MCP servers to force text-based tool catalog usage
        cmd.arg("--disable-builtin-mcps");

        // Prevent reading project AGENTS.md instructions
        cmd.arg("--no-custom-instructions");

        // Autonomous mode — no interactive prompts
        cmd.arg("--no-ask-user");

        // Clean text output
        cmd.arg("--no-color");

        if silent {
            // Output only the agent response (no stats footer)
            cmd.arg("-s");
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

    /// Parse plain-text output into a `ChatResponse`
    fn parse_response(raw: &[u8]) -> Result<ChatResponse, RunnerError> {
        let content = str::from_utf8(raw)
            .map_err(|e| {
                RunnerError::internal(format!("Copilot CLI output is not valid UTF-8: {e}"))
            })?
            .trim()
            .to_owned();

        Ok(ChatResponse {
            content,
            model: "copilot".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }
}

#[async_trait]
impl LlmProvider for CopilotRunner {
    // Copilot CLI has no --system-prompt flag; system messages are
    // embedded into the prompt via build_prompt(). Streaming is
    // supported by reading stdout line by line.
    crate::delegate_provider_base!("copilot", "GitHub Copilot CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "copilot"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = build_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt, true);

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "copilot")?;

        Self::parse_response(&output.stdout)
    }

    #[instrument(skip_all, fields(runner = "copilot"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prompt = build_prompt(&request.messages);
        let mut cmd = self.build_command(&prompt, true);

        // Enable streaming
        cmd.args(["--stream", "on"]);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn copilot for streaming: {e}"))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            RunnerError::internal("Failed to capture copilot stdout for streaming")
        })?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading copilot stream: {e}")))?;

            Ok(StreamChunk {
                delta: line,
                is_final: false,
                finish_reason: None,
            })
        });

        Ok(Box::pin(GuardedStream::new(stream, child, stderr_task)))
    }
}
