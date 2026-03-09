// ABOUTME: Shared base struct and macro for CLI runner boilerplate reduction
// ABOUTME: Provides CliRunnerBase (fields, constructor, health check, exit code) and delegate_provider_base!
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # CLI Runner Common Infrastructure
//!
//! Shared base struct and delegation macro that eliminate boilerplate across
//! the 9 CLI runner implementations. Each runner wraps [`CliRunnerBase`] and
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
    /// Logs stderr/stdout previews and constructs a standard error message.
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

        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        warn!(
            exit_code = output.exit_code,
            stdout_len = output.stdout.len(),
            stderr_len = output.stderr.len(),
            stdout_preview = %stdout.chars().take(500).collect::<String>(),
            stderr_preview = %stderr.chars().take(500).collect::<String>(),
            "{runner_name} CLI failed"
        );
        let detail = if stderr.is_empty() { &stdout } else { &stderr };
        Err(RunnerError::external_service(
            runner_name,
            format!(
                "{runner_name} exited with code {}: {detail}",
                output.exit_code
            ),
        ))
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
