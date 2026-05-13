// ABOUTME: Shared base struct and macro for CLI runner boilerplate reduction
// ABOUTME: Provides CliRunnerBase (fields, constructor, health check, exit code) and delegate_provider_base!
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # CLI Runner Common Infrastructure
//!
//! Shared base struct and delegation macro that eliminate boilerplate across
//! the 11 CLI runner implementations. Each runner wraps [`CliRunnerBase`] and
//! uses [`delegate_provider_base!`] to auto-generate the repetitive
//! [`LlmProvider`](crate::types::LlmProvider) trait methods.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::process::Command;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::config::RunnerConfig;
use crate::process::{run_cli_command, CliOutput};
use crate::types::RunnerError;

/// Maximum output size for a single CLI invocation (50 MiB)
pub const MAX_OUTPUT_BYTES: usize = 50 * 1024 * 1024;

/// Health check timeout (10 seconds)
pub const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// Health check output limit (4 KiB)
pub const HEALTH_CHECK_MAX_OUTPUT: usize = 4096;

/// Shared base struct for all CLI runners.
///
/// Holds the common fields (config, model info, session tracking) that every
/// CLI runner needs. Individual runners wrap this and add only their
/// command-building and response-parsing logic.
pub struct CliRunnerBase {
    /// Runner configuration (binary path, timeout, extra args, etc.)
    pub(crate) config: RunnerConfig,
    /// Resolved default model identifier
    pub(crate) default_model: String,
    /// List of available models for this provider
    pub(crate) available_models: Vec<String>,
    /// Session ID cache keyed by model name (for multi-turn sessions)
    pub(crate) session_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl CliRunnerBase {
    /// Create a new base with the given config, default model, and fallback model list.
    pub fn new(config: RunnerConfig, default_model: &str, fallback_models: &[&str]) -> Self {
        let resolved_model = config
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_owned());
        let available_models = fallback_models.iter().map(|s| (*s).to_owned()).collect();
        Self {
            config,
            default_model: resolved_model,
            available_models,
            session_ids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the default model identifier
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// Get the list of available models
    pub fn available_models(&self) -> &[String] {
        &self.available_models
    }

    /// Store a session ID for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        let mut sessions = self.session_ids.lock().await;
        sessions.insert(key.to_owned(), session_id.to_owned());
    }

    /// Get a stored session ID
    pub async fn get_session(&self, key: &str) -> Option<String> {
        let sessions = self.session_ids.lock().await;
        sessions.get(key).cloned()
    }

    /// Run a `--version` health check against the runner binary.
    ///
    /// Returns `true` if the binary exits 0, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the subprocess cannot be spawned.
    pub async fn health_check(&self, runner_name: &str) -> Result<bool, RunnerError> {
        let mut cmd = Command::new(&self.config.binary_path);
        cmd.arg("--version");

        let output =
            run_cli_command(&mut cmd, HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_MAX_OUTPUT).await?;

        if output.exit_code == 0 {
            debug!("{runner_name} health check passed");
            Ok(true)
        } else {
            warn!(
                exit_code = output.exit_code,
                "{runner_name} health check failed"
            );
            Ok(false)
        }
    }

    /// Check CLI exit code and return an error if non-zero.
    ///
    /// Logs output byte lengths (not content) and constructs an error
    /// message that surfaces the most diagnostic stream available.
    /// Stderr is preferred (where well-behaved Unix programs report
    /// errors), but some CLIs — notably `claude-code` exit-1 paths —
    /// write the error to stdout instead, so a non-empty stdout is used
    /// as a fallback rather than rendering the misleading `(no output)`
    /// the earlier implementation produced. Up to 500 chars are
    /// surfaced so multi-line diagnostics aren't truncated to one line.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError::external_service`] when `output.exit_code != 0`.
    pub fn check_exit_code(
        &self,
        output: &CliOutput,
        runner_name: &str,
    ) -> Result<(), RunnerError> {
        if output.exit_code == 0 {
            return Ok(());
        }

        warn!(
            exit_code = output.exit_code,
            stdout_len = output.stdout.len(),
            stderr_len = output.stderr.len(),
            "{runner_name} CLI failed"
        );
        let diagnostic = pick_diagnostic_stream(&output.stderr, &output.stdout);
        Err(RunnerError::external_service(
            runner_name,
            format!(
                "{runner_name} exited with code {}: {diagnostic}",
                output.exit_code
            ),
        ))
    }
}

/// Build a short, single-string diagnostic from the CLI's output streams.
///
/// Preference order: `stderr` (when non-empty after trim) → `stdout`
/// (when non-empty after trim) → a literal `(no output on stderr or
/// stdout)`. Truncated to 500 characters so multi-line errors carry
/// across without being clipped to a useless first-line, while still
/// avoiding multi-MB prompt-replay payloads in our error logs and
/// downstream Slack messages.
fn pick_diagnostic_stream(stderr: &[u8], stdout: &[u8]) -> String {
    let stderr_str = String::from_utf8_lossy(stderr);
    let stdout_str = String::from_utf8_lossy(stdout);
    let chosen = if !stderr_str.trim().is_empty() {
        stderr_str
    } else if !stdout_str.trim().is_empty() {
        stdout_str
    } else {
        return "(no output on stderr or stdout)".to_owned();
    };
    let trimmed = chosen.trim();
    if trimmed.chars().count() <= 500 {
        trimmed.to_owned()
    } else {
        let head: String = trimmed.chars().take(500).collect();
        format!("{head}… (truncated)")
    }
}

/// Generate the boilerplate [`LlmProvider`](crate::types::LlmProvider) trait methods.
///
/// Must be invoked inside an `#[async_trait] impl LlmProvider for ...` block.
/// The implementing struct must have a field named `base` of type [`CliRunnerBase`].
///
/// Generates: `name()`, `display_name()`, `capabilities()`, `default_model()`,
/// `available_models()`, and `health_check()`.
///
/// The caller still provides `complete()` and `complete_stream()`.
#[macro_export]
macro_rules! delegate_provider_base {
    ($runner_name:expr, $display_name:expr, $caps:expr) => {
        fn name(&self) -> &'static str {
            $runner_name
        }

        fn display_name(&self) -> &str {
            $display_name
        }

        fn capabilities(&self) -> $crate::types::LlmCapabilities {
            $caps
        }

        fn default_model(&self) -> &str {
            self.base.default_model()
        }

        fn available_models(&self) -> &[String] {
            self.base.available_models()
        }

        fn health_check<'life0, 'async_trait>(
            &'life0 self,
        ) -> ::core::pin::Pin<
            Box<
                dyn ::core::future::Future<Output = Result<bool, $crate::types::RunnerError>>
                    + ::core::marker::Send
                    + 'async_trait,
            >,
        >
        where
            'life0: 'async_trait,
            Self: 'async_trait,
        {
            Box::pin(async move { self.base.health_check($runner_name).await })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::pick_diagnostic_stream;

    #[test]
    fn prefers_stderr_when_both_present() {
        let out = pick_diagnostic_stream(b"err line", b"out line");
        assert_eq!(out, "err line");
    }

    #[test]
    fn falls_back_to_stdout_when_stderr_empty() {
        let out = pick_diagnostic_stream(b"", b"the actual error from claude-code");
        assert_eq!(out, "the actual error from claude-code");
    }

    #[test]
    fn falls_back_to_stdout_when_stderr_whitespace_only() {
        let out = pick_diagnostic_stream(b"   \n  ", b"stdout has the answer");
        assert_eq!(out, "stdout has the answer");
    }

    #[test]
    fn renders_no_output_when_both_empty() {
        let out = pick_diagnostic_stream(b"", b"");
        assert_eq!(out, "(no output on stderr or stdout)");
    }

    #[test]
    fn preserves_multiline_errors() {
        let out = pick_diagnostic_stream(b"", b"line1\nline2\nline3");
        assert!(out.contains("line1"));
        assert!(out.contains("line2"));
        assert!(out.contains("line3"));
    }

    #[test]
    fn truncates_oversized_streams() {
        let big: String = "a".repeat(1500);
        let out = pick_diagnostic_stream(b"", big.as_bytes());
        assert!(out.ends_with("… (truncated)"));
        assert!(out.chars().count() < 600);
    }
}
