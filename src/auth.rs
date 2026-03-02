// ABOUTME: Auth readiness checking for CLI-based LLM runners
// ABOUTME: Verifies that CLI tools are installed, authenticated, and ready to use
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::fmt;
use std::path::Path;
use std::time::Duration;

use crate::types::RunnerError;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, warn};

use crate::config::CliRunnerType;
use crate::process::{run_cli_command, CliOutput};

/// Maximum time to wait for an auth-check command
const AUTH_CHECK_TIMEOUT: Duration = Duration::from_secs(15);

/// Maximum output size for auth-check commands (64 KiB)
const AUTH_CHECK_MAX_OUTPUT: usize = 64 * 1024;

/// Provider readiness status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderReadiness {
    /// CLI is installed and authenticated
    Ready,
    /// CLI is installed but not authenticated or misconfigured
    NotReady {
        /// Human-readable explanation of why the provider is not ready
        reason: String,
        /// Suggested action to fix the issue
        action: String,
    },
    /// CLI binary was not found at the expected path
    BinaryMissing {
        /// Name of the binary that was expected
        expected_binary: String,
    },
    /// Unable to determine readiness
    Unknown {
        /// Explanation of why readiness could not be determined
        reason: String,
    },
}

impl ProviderReadiness {
    /// Returns `true` when the provider is authenticated and ready to serve requests
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

impl fmt::Display for ProviderReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "ready"),
            Self::NotReady { reason, action } => {
                write!(f, "not ready: {reason} (action: {action})")
            }
            Self::BinaryMissing { expected_binary } => {
                write!(f, "binary missing: {expected_binary}")
            }
            Self::Unknown { reason } => write!(f, "unknown: {reason}"),
        }
    }
}

/// Check whether a CLI runner is ready to handle requests
///
/// Runs a lightweight probe command appropriate for the runner type and
/// interprets the result to determine authentication and availability status.
///
/// # Errors
///
/// Returns `RunnerError` only on internal failures (e.g. I/O errors when
/// spawning the probe process). A non-ready provider is reported as
/// `ProviderReadiness::NotReady`, not as an error.
pub async fn check_readiness(
    runner_type: &CliRunnerType,
    binary_path: &Path,
) -> Result<ProviderReadiness, RunnerError> {
    if !binary_path.exists() {
        return Ok(ProviderReadiness::BinaryMissing {
            expected_binary: binary_path.display().to_string(),
        });
    }

    match runner_type {
        CliRunnerType::ClaudeCode => check_claude_readiness(binary_path).await,
        CliRunnerType::Copilot => check_copilot_readiness(binary_path).await,
        CliRunnerType::CursorAgent => check_version_probe(binary_path, "cursor-agent").await,
        CliRunnerType::OpenCode => check_version_probe(binary_path, "opencode").await,
        CliRunnerType::GeminiCli => check_version_probe(binary_path, "gemini").await,
        CliRunnerType::CodexCli => check_version_probe(binary_path, "codex").await,
    }
}

/// Claude Code has an explicit `auth status` sub-command
async fn check_claude_readiness(binary_path: &Path) -> Result<ProviderReadiness, RunnerError> {
    let mut cmd = Command::new(binary_path);
    cmd.args(["auth", "status"]);

    let output = run_cli_command(&mut cmd, AUTH_CHECK_TIMEOUT, AUTH_CHECK_MAX_OUTPUT).await;

    match output {
        Ok(CliOutput { exit_code: 0, .. }) => {
            debug!("Claude Code auth status: ready");
            Ok(ProviderReadiness::Ready)
        }
        Ok(cli_output) => {
            let stderr = String::from_utf8_lossy(&cli_output.stderr);
            warn!(exit_code = cli_output.exit_code, %stderr, "Claude Code auth check failed");
            Ok(ProviderReadiness::NotReady {
                reason: format!("Auth check exited with code {}", cli_output.exit_code),
                action: "Run `claude auth login` to authenticate".to_owned(),
            })
        }
        Err(e) => Ok(ProviderReadiness::Unknown {
            reason: format!("Failed to run auth check: {e}"),
        }),
    }
}

/// Check Copilot readiness via `gh auth status` for real auth verification.
///
/// Falls back to a `copilot --version` probe if `gh` is not available.
/// A successful `--version` only confirms the binary exists; `gh auth status`
/// confirms the user is actually authenticated with GitHub.
async fn check_copilot_readiness(binary_path: &Path) -> Result<ProviderReadiness, RunnerError> {
    // Try `gh auth status` for a real authentication check
    if let Some(gh_result) = check_gh_auth_status().await {
        return Ok(gh_result);
    }

    // Fall back to version probe when `gh` is unavailable
    debug!("gh CLI not available, falling back to copilot --version probe");
    let mut cmd = Command::new(binary_path);
    cmd.arg("--version");

    let output = run_cli_command(&mut cmd, AUTH_CHECK_TIMEOUT, AUTH_CHECK_MAX_OUTPUT).await;

    match output {
        Ok(CliOutput { exit_code: 0, .. }) => {
            debug!("Copilot CLI version probe succeeded (auth not verified)");
            Ok(ProviderReadiness::Ready)
        }
        Ok(cli_output) => {
            let stderr = String::from_utf8_lossy(&cli_output.stderr);
            warn!(exit_code = cli_output.exit_code, %stderr, "Copilot CLI version probe failed");
            Ok(ProviderReadiness::NotReady {
                reason: format!(
                    "copilot --version exited with code {}",
                    cli_output.exit_code
                ),
                action: "Run `copilot` to complete GitHub authentication".to_owned(),
            })
        }
        Err(e) => Ok(ProviderReadiness::Unknown {
            reason: format!("Failed to run copilot --version: {e}"),
        }),
    }
}

/// Attempt to verify GitHub authentication via `gh auth status`.
///
/// Returns `Some(ProviderReadiness)` when `gh` is available and produces
/// a definitive result; returns `None` when `gh` cannot be found so the
/// caller can fall back to another probe.
async fn check_gh_auth_status() -> Option<ProviderReadiness> {
    let mut cmd = Command::new("gh");
    cmd.args(["auth", "status"]);

    let output = run_cli_command(&mut cmd, AUTH_CHECK_TIMEOUT, AUTH_CHECK_MAX_OUTPUT)
        .await
        .ok()?;

    if output.exit_code == 0 {
        debug!("gh auth status: authenticated");
        Some(ProviderReadiness::Ready)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!(%stderr, "gh auth status: not authenticated");
        Some(ProviderReadiness::NotReady {
            reason: "GitHub CLI reports not authenticated".to_owned(),
            action: "Run `gh auth login` to authenticate with GitHub".to_owned(),
        })
    }
}

/// Generic version probe — success means the binary is installed and executable.
///
/// This does NOT verify authentication; it only confirms the binary can
/// run `--version` without error. Runners that lack a dedicated auth
/// sub-command fall back to this heuristic.
async fn check_version_probe(
    binary_path: &Path,
    name: &str,
) -> Result<ProviderReadiness, RunnerError> {
    let mut cmd = Command::new(binary_path);
    cmd.arg("--version");

    let output = run_cli_command(&mut cmd, AUTH_CHECK_TIMEOUT, AUTH_CHECK_MAX_OUTPUT).await;

    match output {
        Ok(CliOutput { exit_code: 0, .. }) => {
            debug!(runner = name, "Version probe succeeded");
            Ok(ProviderReadiness::Ready)
        }
        Ok(cli_output) => {
            let stderr = String::from_utf8_lossy(&cli_output.stderr);
            warn!(runner = name, exit_code = cli_output.exit_code, %stderr, "Version probe failed");
            Ok(ProviderReadiness::NotReady {
                reason: format!("{name} --version exited with code {}", cli_output.exit_code),
                action: format!("Verify {name} is properly installed"),
            })
        }
        Err(e) => Ok(ProviderReadiness::Unknown {
            reason: format!("Failed to run {name} --version: {e}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_readiness_is_ready() {
        assert!(ProviderReadiness::Ready.is_ready());
    }

    #[test]
    fn test_provider_readiness_not_ready() {
        let status = ProviderReadiness::NotReady {
            reason: "not authed".to_owned(),
            action: "login".to_owned(),
        };
        assert!(!status.is_ready());
    }

    #[test]
    fn test_provider_readiness_binary_missing() {
        let status = ProviderReadiness::BinaryMissing {
            expected_binary: "claude".to_owned(),
        };
        assert!(!status.is_ready());
    }

    #[test]
    fn test_provider_readiness_unknown() {
        let status = ProviderReadiness::Unknown {
            reason: "timeout".to_owned(),
        };
        assert!(!status.is_ready());
    }

    #[test]
    fn test_provider_readiness_display() {
        assert_eq!(format!("{}", ProviderReadiness::Ready), "ready");

        let not_ready = ProviderReadiness::NotReady {
            reason: "expired token".to_owned(),
            action: "re-login".to_owned(),
        };
        assert!(format!("{not_ready}").contains("expired token"));
        assert!(format!("{not_ready}").contains("re-login"));

        let missing = ProviderReadiness::BinaryMissing {
            expected_binary: "claude".to_owned(),
        };
        assert!(format!("{missing}").contains("claude"));

        let unknown = ProviderReadiness::Unknown {
            reason: "error".to_owned(),
        };
        assert!(format!("{unknown}").contains("error"));
    }
}
