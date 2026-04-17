// ABOUTME: GitHub Copilot CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `copilot` CLI with plain-text output parsing and streaming support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;
use std::sync::Arc;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::copilot_models::{
    self, catalog_ids, classify_model_error, default_effort_for, next_preferred, preferred_default,
    ReasoningEffort,
};
use crate::process::CliOutput;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider, RunnerError,
    StreamChunk,
};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::RwLock;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::prepare_prompt;
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

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
/// The default model comes from the ranked [`copilot_models::CATALOG`].
/// When a request fails with [`ErrorKind::ModelUnavailable`] (e.g. the account
/// is no longer entitled to the preferred model), the runner advances one step
/// down the catalog and retries. The healed choice is cached so subsequent
/// requests skip the dead model.
pub struct CopilotRunner {
    base: CliRunnerBase,
    /// Runtime-healed default model id, shared across concurrent requests.
    runtime_model: Arc<RwLock<String>>,
}

impl CopilotRunner {
    /// Create a new Copilot CLI runner with the given configuration.
    ///
    /// The default model is resolved from the ranked catalog and will self-heal
    /// when the top-ranked model is unavailable for the calling account.
    pub fn new(config: RunnerConfig) -> Self {
        let catalog = catalog_ids();
        let fallback_slice: Vec<&str> = catalog.iter().map(String::as_str).collect();
        let base = CliRunnerBase::new(config, preferred_default(), &fallback_slice);
        let runtime_model = Arc::new(RwLock::new(base.default_model().to_owned()));
        Self {
            base,
            runtime_model,
        }
    }

    /// Effective reasoning effort for a given model id.
    ///
    /// Honors an explicit `reasoning_effort` in [`RunnerConfig`]; otherwise
    /// falls back to the family-aware default from [`copilot_models`].
    fn effective_effort(&self, model: &str) -> Option<ReasoningEffort> {
        self.base
            .config
            .reasoning_effort
            .or_else(|| default_effort_for(model))
    }

    /// Build the base command with common arguments for a specific model.
    fn build_command(&self, prompt: &str, model: &str, silent: bool) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);

        cmd.args(["-p", prompt]);
        cmd.args(["--model", model]);

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

    /// Resolve the model to use for the given request.
    ///
    /// Priority: request-level override → config-level override → runtime-healed default.
    async fn resolve_model(&self, request: &ChatRequest) -> String {
        if let Some(m) = request.model.as_deref() {
            return m.to_owned();
        }
        if let Some(m) = self.base.config.model.as_deref() {
            return m.to_owned();
        }
        self.runtime_model.read().await.clone()
    }

    /// Promote `model` to the runtime default if it differs from the current value.
    async fn remember_healed(&self, model: &str) {
        let mut guard = self.runtime_model.write().await;
        if *guard != model {
            debug!(
                previous = %guard,
                healed = %model,
                "copilot: caching healed default model"
            );
            model.clone_into(&mut guard);
        }
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

    /// Execute `complete` with model self-heal: on [`ErrorKind::ModelUnavailable`]
    /// rotate to the next catalog candidate, update the runtime default, and retry.
    ///
    /// The walk is bounded by the catalog length to avoid infinite loops when
    /// the account has no entitled models.
    async fn complete_with_heal(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let mut model = self.resolve_model(request).await;
        let user_specified_model = request.model.is_some() || self.base.config.model.is_some();

        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        for _ in 0..copilot_models::CATALOG.len() {
            let mut cmd = self.build_command(prompt, &model, true);
            let output =
                run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;

            if output.exit_code == 0 {
                if !user_specified_model {
                    self.remember_healed(&model).await;
                }
                return Self::parse_response(&output.stdout, &model);
            }

            let err = self.classify_exit(&output);
            if err.kind != ErrorKind::ModelUnavailable {
                return Err(err);
            }

            // User pinned a specific model → surface the availability error
            // instead of silently substituting something else.
            if user_specified_model {
                return Err(err);
            }

            match next_preferred(&model) {
                Some(next) => {
                    warn!(
                        previous = %model,
                        next = %next,
                        "copilot: model unavailable, advancing to next ranked candidate"
                    );
                    model = next.to_owned();
                }
                None => return Err(err),
            }
        }

        Err(RunnerError::model_unavailable("exhausted catalog"))
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
        self.complete_with_heal(request).await
    }

    #[instrument(skip_all, fields(runner = "copilot"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prepared = prepare_prompt(&request.messages)?;
        let prompt = &prepared.prompt;
        let model = self.resolve_model(request).await;
        let mut cmd = self.build_command(prompt, &model, true);

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
    use std::path::PathBuf;

    fn runner() -> CopilotRunner {
        CopilotRunner::new(RunnerConfig::new(PathBuf::from("copilot")))
    }

    #[test]
    fn default_model_comes_from_catalog() {
        assert_eq!(runner().default_model(), preferred_default());
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
    fn effective_effort_prefers_config_override() {
        let config = RunnerConfig::new(PathBuf::from("copilot"))
            .with_reasoning_effort(ReasoningEffort::High);
        let runner = CopilotRunner::new(config);
        assert_eq!(
            runner.effective_effort("claude-opus-4.7"),
            Some(ReasoningEffort::High)
        );
    }

    #[test]
    fn effective_effort_falls_back_to_family_default() {
        assert_eq!(
            runner().effective_effort("claude-opus-4.7"),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(runner().effective_effort("claude-haiku-4.5"), None);
    }
}
