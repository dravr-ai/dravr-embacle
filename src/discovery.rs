// ABOUTME: Binary auto-detection for CLI-based LLM runners
// ABOUTME: Discovers available CLI tools on the system in priority order
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;

use crate::types::RunnerError;
use tracing::debug;

use crate::config::{CliRunnerType, RunnerConfig};

/// Priority-ordered list of CLI runners to probe during discovery
const DISCOVERY_ORDER: &[CliRunnerType] = &[
    CliRunnerType::ClaudeCode,
    CliRunnerType::Copilot,
    CliRunnerType::CursorAgent,
    CliRunnerType::OpenCode,
    CliRunnerType::GeminiCli,
    CliRunnerType::CodexCli,
];

/// Discover the first available CLI runner on the system
///
/// Probes runners in priority order: Claude Code → Copilot → Cursor Agent → `OpenCode` → Gemini → Codex.
/// Returns a `RunnerConfig` for the first binary found on `PATH` or via its
/// environment-variable override.
///
/// # Errors
///
/// Returns `RunnerError` if no runner binary can be located.
pub fn discover_runner() -> Result<(CliRunnerType, RunnerConfig), RunnerError> {
    for runner_type in DISCOVERY_ORDER {
        let env_key = runner_type.env_override_key();
        let env_override = env::var(env_key).ok();

        match resolve_binary(runner_type.binary_name(), env_override.as_deref()) {
            Ok(path) => {
                debug!(
                    runner = runner_type.binary_name(),
                    path = %path.display(),
                    "Discovered CLI runner"
                );
                return Ok((*runner_type, RunnerConfig::new(path)));
            }
            Err(_) => {
                debug!(
                    runner = runner_type.binary_name(),
                    env_key, "Runner not found, trying next"
                );
            }
        }
    }

    Err(RunnerError::internal(
        "No CLI runner found. Install one of: claude, copilot, cursor-agent, opencode, gemini, codex",
    ))
}

/// Resolve a binary path by name, optionally using an environment variable override
///
/// Resolution order:
/// 1. If `env_override` is `Some`, use that value as the path
/// 2. Otherwise, search `PATH` using `which`
///
/// # Errors
///
/// Returns `RunnerError` if the binary cannot be found.
pub fn resolve_binary(name: &str, env_override: Option<&str>) -> Result<PathBuf, RunnerError> {
    if let Some(override_path) = env_override {
        let path = PathBuf::from(override_path);
        if path.exists() {
            debug!(binary = name, path = %path.display(), "Resolved via env override");
            return Ok(path);
        }
        return Err(RunnerError::internal(format!(
            "Environment override points to non-existent path: {override_path}"
        )));
    }

    which::which(name)
        .map_err(|e| RunnerError::internal(format!("Binary '{name}' not found on PATH: {e}")))
}
