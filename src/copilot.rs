// ABOUTME: GitHub Copilot CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `copilot` CLI with plain-text output parsing and streaming support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::copilot_models::{
    catalog_ids, classify_model_error, default_effort_for, ReasoningEffort,
};
use crate::process::CliOutput;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};
use async_trait::async_trait;
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

/// Sentinel model id used when the caller has not pinned a specific model.
///
/// Surfaced in [`ChatResponse::model`] to signal "the Copilot CLI chose its own
/// account-entitled default" rather than a particular identifier. We pass no
/// `--model` flag in this case, so the CLI picks the highest-ranked model the
/// authenticated account is entitled to.
const AUTO_MODEL_SENTINEL: &str = "auto";

/// Every model id in the ranked Copilot catalog.
///
/// Kept for backward-compatible access to the list used as `available_models()`.
#[must_use]
pub fn copilot_fallback_models() -> Vec<String> {
    catalog_ids()
}

/// GitHub Copilot CLI runner
///
/// Implements `LlmProvider` by delegating to the `copilot` binary in
/// non-interactive mode (`-p`). Copilot CLI outputs plain text (no JSON
/// structure), so the raw stdout is captured as the response content.
/// System messages are embedded into the user prompt since Copilot CLI
/// has no `--system-prompt` flag.
///
/// # Model selection
///
/// Pins a model only when the caller supplies one — via `ChatRequest::model`
/// or [`RunnerConfig::model`]. Otherwise, the `--model` flag is omitted and
/// the Copilot CLI selects the highest-ranked model the authenticated account
/// is entitled to. This avoids surfacing hardcoded defaults (e.g. an Opus
/// variant) to accounts that aren't entitled to them.
pub struct CopilotRunner {
    base: CliRunnerBase,
}

impl CopilotRunner {
    /// Create a new Copilot CLI runner with the given configuration.
    ///
    /// When no model is configured, completions omit `--model` so the CLI
    /// uses its own account-entitled default. The advertised `default_model`
    /// is therefore the [`AUTO_MODEL_SENTINEL`], not a specific id.
    pub fn new(config: RunnerConfig) -> Self {
        let catalog = catalog_ids();
        let fallback_slice: Vec<&str> = catalog.iter().map(String::as_str).collect();
        let base = CliRunnerBase::new(config, AUTO_MODEL_SENTINEL, &fallback_slice);
        Self { base }
    }

    /// Effective reasoning effort for the pinned model, when one is known.
    ///
    /// Honors an explicit `reasoning_effort` in [`RunnerConfig`] regardless of
    /// model. When no model is pinned, falls back to `None` so the CLI can
    /// pick its own default alongside its chosen model.
    fn effective_effort(&self, model: Option<&str>) -> Option<ReasoningEffort> {
        if let Some(effort) = self.base.config.reasoning_effort {
            return Some(effort);
        }
        model.and_then(default_effort_for)
    }

    /// Build the base command with common arguments.
    ///
    /// `model` is threaded through as `--model <id>` only when `Some`; a
    /// `None` means "let the CLI pick the account default".
    fn build_command(&self, prompt: &str, model: Option<&str>, silent: bool) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);

        cmd.args(["-p", prompt]);
        if let Some(id) = model {
            cmd.args(["--model", id]);
        }

        if let Some(effort) = self.effective_effort(model) {
            cmd.args(["--reasoning-effort", effort.as_str()]);
        }

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

    /// Resolve the model to pin for the given request, if any.
    ///
    /// Priority: request-level override → config-level override → `None`
    /// (meaning: let the Copilot CLI pick the account's entitled default).
    fn resolve_model(&self, request: &ChatRequest) -> Option<String> {
        request
            .model
            .clone()
            .or_else(|| self.base.config.model.clone())
    }

    /// Map a non-zero CLI exit into a typed error, detecting model-availability failures.
    fn classify_exit(&self, output: &CliOutput) -> RunnerError {
        if output.exit_code == 0 {
            return RunnerError::internal("classify_exit called on successful output");
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if let Some(model) = classify_model_error(&stderr) {
            return RunnerError::model_unavailable(model);
        }
        // Delegate to the shared helper for the generic external-service mapping.
        match self.base.check_exit_code(output, "copilot") {
            Ok(()) => RunnerError::internal("classify_exit: exit code was zero but stderr matched"),
            Err(e) => e,
        }
    }

    /// Parse plain-text output into a `ChatResponse`
    ///
    /// `model` reflects the id we pinned via `--model`, or the
    /// [`AUTO_MODEL_SENTINEL`] when the CLI chose its own default.
    fn parse_response(raw: &[u8], model: &str) -> Result<ChatResponse, RunnerError> {
        let content = str::from_utf8(raw)
            .map_err(|e| {
                RunnerError::internal(format!("Copilot CLI output is not valid UTF-8: {e}"))
            })?
            .trim()
            .to_owned();

        Ok(ChatResponse {
            content,
            model: model.to_owned(),
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
    // embedded into the prompt via prepare_prompt(). Streaming is
    // supported by reading stdout line by line.
    crate::delegate_provider_base!("copilot", "GitHub Copilot CLI", LlmCapabilities::STREAMING);

    #[instrument(skip_all, fields(runner = "copilot"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let pinned = self.resolve_model(request);
        let mut cmd = self.build_command(&prepared.prompt, pinned.as_deref(), true);

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;

        if output.exit_code != 0 {
            return Err(self.classify_exit(&output));
        }

        let reported = pinned.as_deref().unwrap_or(AUTO_MODEL_SENTINEL);
        Self::parse_response(&output.stdout, reported)
    }

    #[instrument(skip_all, fields(runner = "copilot"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let pinned = self.resolve_model(request);
        let mut cmd = self.build_command(prompt, pinned.as_deref(), true);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copilot_models;
    use crate::types::ChatMessage;
    use std::path::PathBuf;

    fn runner() -> CopilotRunner {
        CopilotRunner::new(RunnerConfig::new(PathBuf::from("copilot")))
    }

    fn request_with_model(model: Option<&str>) -> ChatRequest {
        let mut req = ChatRequest::new(vec![ChatMessage::user("hi")]);
        req.model = model.map(str::to_owned);
        req
    }

    #[test]
    fn default_model_is_auto_sentinel() {
        assert_eq!(runner().default_model(), AUTO_MODEL_SENTINEL);
    }

    #[test]
    fn fallback_models_match_catalog() {
        let runner = runner();
        let available: Vec<&str> = runner
            .available_models()
            .iter()
            .map(String::as_str)
            .collect();
        let catalog: Vec<&str> = copilot_models::CATALOG.iter().map(|c| c.id).collect();
        assert_eq!(available, catalog);
    }

    #[test]
    fn resolve_model_prefers_request_override() {
        let req = request_with_model(Some("gpt-5.4"));
        assert_eq!(runner().resolve_model(&req).as_deref(), Some("gpt-5.4"));
    }

    #[test]
    fn resolve_model_falls_back_to_config() {
        let config =
            RunnerConfig::new(PathBuf::from("copilot")).with_model("claude-sonnet-4.6".to_owned());
        let runner = CopilotRunner::new(config);
        let req = request_with_model(None);
        assert_eq!(
            runner.resolve_model(&req).as_deref(),
            Some("claude-sonnet-4.6")
        );
    }

    #[test]
    fn resolve_model_returns_none_when_unpinned() {
        let req = request_with_model(None);
        assert!(runner().resolve_model(&req).is_none());
    }

    #[test]
    fn build_command_omits_model_flag_when_none() {
        let cmd = runner().build_command("hi", None, true);
        let args: Vec<&str> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_str().unwrap_or_default())
            .collect();
        assert!(!args.contains(&"--model"));
        assert!(!args.contains(&"--reasoning-effort"));
    }

    #[test]
    fn build_command_includes_model_flag_when_some() {
        let cmd = runner().build_command("hi", Some("gpt-5.4"), true);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let idx = args
            .iter()
            .position(|a| a == "--model")
            .expect("--model flag should be present"); // Safe: test assertion
        assert_eq!(args.get(idx + 1).map(String::as_str), Some("gpt-5.4"));
    }

    #[test]
    fn effective_effort_prefers_config_override() {
        let config = RunnerConfig::new(PathBuf::from("copilot"))
            .with_reasoning_effort(ReasoningEffort::High);
        let runner = CopilotRunner::new(config);
        assert_eq!(
            runner.effective_effort(Some("claude-opus-4.7")),
            Some(ReasoningEffort::High)
        );
    }

    #[test]
    fn effective_effort_config_override_applies_without_model() {
        let config = RunnerConfig::new(PathBuf::from("copilot"))
            .with_reasoning_effort(ReasoningEffort::Medium);
        let runner = CopilotRunner::new(config);
        assert_eq!(runner.effective_effort(None), Some(ReasoningEffort::Medium));
    }

    #[test]
    fn effective_effort_falls_back_to_family_default() {
        assert_eq!(
            runner().effective_effort(Some("claude-opus-4.7")),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(runner().effective_effort(Some("claude-haiku-4.5")), None);
    }

    #[test]
    fn effective_effort_none_when_unpinned_and_unconfigured() {
        assert_eq!(runner().effective_effort(None), None);
    }
}
