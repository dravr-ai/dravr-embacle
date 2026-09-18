// ABOUTME: Configuration for the Copilot SDK provider — github-copilot-sdk over the Rust copilot-runtime on stdio.
// ABOUTME: Reads environment variables and provides defaults; the transport is stdio and nothing else.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;
use std::process;
use std::time::Duration;

use crate::copilot_common::{resolve_github_token, PermissionPolicy, DEFAULT_MAX_HISTORY_TURNS};
use crate::copilot_models::preferred_default;

/// Default bound on one turn — prompt sent to session idle. The runtime
/// retries a failed model call on its own, so a turn can legitimately outlive
/// a single call by a wide margin.
pub const DEFAULT_PROMPT_TIMEOUT_SECS: u64 = 300;

/// Default bound on starting the client and opening a session: the runtime
/// spawn, its protocol handshake, the model catalogue, `session.create`.
pub const DEFAULT_SESSION_TIMEOUT_SECS: u64 = 60;

/// Configuration for the Copilot SDK provider.
#[derive(Debug, Clone)]
pub struct CopilotSdkConfig {
    /// Path to the `copilot-runtime` wrapper. `runtime.node` must sit next to
    /// it. `None` lets the SDK resolve `COPILOT_CLI_PATH` itself; there is no
    /// `PATH` scan.
    pub runtime_path: Option<PathBuf>,
    /// Directory the runtime treats as its home (`COPILOT_HOME`): session
    /// state, credential store, logs. Required because the client runs in
    /// [`ClientMode::Empty`](github_copilot_sdk::ClientMode::Empty), which
    /// refuses to fall back to `~/.copilot`.
    pub base_directory: PathBuf,
    /// Model used when a request names none.
    pub model: String,
    /// GitHub token the runtime authenticates with; `None` uses the runtime's
    /// stored login.
    pub github_token: Option<String>,
    /// Policy for handling permission requests from the runtime.
    pub permission_policy: PermissionPolicy,
    /// Maximum number of prior conversation messages (user + assistant) rendered
    /// into the prompt for multi-turn context. Set to 0 to disable history.
    pub max_history_turns: usize,
    /// Advertise `SDK_TOOL_CALLING` so callers route tool turns through
    /// [`HeadlessTurnProvider::converse`](crate::copilot_common::HeadlessTurnProvider::converse)
    /// with per-request MCP servers (native tool calling), instead of falling
    /// through to text-based `<tool_call>` parsing. Default: false.
    pub mcp_tool_calling: bool,
    /// Bound on one turn, prompt sent to session idle.
    pub prompt_timeout: Duration,
    /// Bound on client start and session creation.
    pub session_timeout: Duration,
}

impl Default for CopilotSdkConfig {
    fn default() -> Self {
        Self {
            runtime_path: None,
            base_directory: default_base_directory(),
            model: preferred_default().to_owned(),
            github_token: None,
            permission_policy: PermissionPolicy::default(),
            max_history_turns: DEFAULT_MAX_HISTORY_TURNS,
            mcp_tool_calling: false,
            prompt_timeout: Duration::from_secs(DEFAULT_PROMPT_TIMEOUT_SECS),
            session_timeout: Duration::from_secs(DEFAULT_SESSION_TIMEOUT_SECS),
        }
    }
}

/// A per-process home under the system temp dir, so two servers on one host
/// never share a credential store or session state.
fn default_base_directory() -> PathBuf {
    env::temp_dir().join(format!("embacle-copilot-sdk-{}", process::id()))
}

impl CopilotSdkConfig {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `COPILOT_RUNTIME_PATH` — Path to the `copilot-runtime` wrapper
    ///   (`runtime.node` adjacent); unset lets the SDK resolve `COPILOT_CLI_PATH`
    /// - `COPILOT_SDK_HOME` — The runtime's home directory (default: a
    ///   per-process directory under the system temp dir)
    /// - `COPILOT_SDK_MODEL` — Default model (defaults to the top-ranked
    ///   candidate in [`crate::copilot_models::CATALOG`])
    /// - `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` — GitHub auth token
    /// - `COPILOT_SDK_PERMISSION_POLICY` — `auto_approve` to approve the
    ///   runtime's permission prompts; anything else denies
    /// - `COPILOT_SDK_MAX_HISTORY_TURNS` — Max conversation history turns (default: 20)
    /// - `COPILOT_SDK_MCP_TOOL_CALLING` — Advertise `SDK_TOOL_CALLING` for native
    ///   MCP tool calling (default: false)
    /// - `EMBACLE_SDK_PROMPT_TIMEOUT_SECS` — Bound on one turn (default: 300)
    /// - `EMBACLE_SDK_SESSION_TIMEOUT_SECS` — Bound on client start and session
    ///   creation (default: 60)
    #[must_use]
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let runtime_path = env::var("COPILOT_RUNTIME_PATH").ok().map(PathBuf::from);

        let base_directory = env::var("COPILOT_SDK_HOME")
            .ok()
            .map_or(defaults.base_directory, PathBuf::from);

        let model = env::var("COPILOT_SDK_MODEL").unwrap_or(defaults.model);

        let permission_policy =
            PermissionPolicy::parse(&env::var("COPILOT_SDK_PERMISSION_POLICY").unwrap_or_default());

        let max_history_turns = env::var("COPILOT_SDK_MAX_HISTORY_TURNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(defaults.max_history_turns);

        let mcp_tool_calling = env::var("COPILOT_SDK_MCP_TOOL_CALLING")
            .is_ok_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"));

        let prompt_timeout = env_secs("EMBACLE_SDK_PROMPT_TIMEOUT_SECS", defaults.prompt_timeout);
        let session_timeout =
            env_secs("EMBACLE_SDK_SESSION_TIMEOUT_SECS", defaults.session_timeout);

        Self {
            runtime_path,
            base_directory,
            model,
            github_token: resolve_github_token(),
            permission_policy,
            max_history_turns,
            mcp_tool_calling,
            prompt_timeout,
            session_timeout,
        }
    }
}

/// A duration in whole seconds from an environment variable, or the default
/// when the variable is unset or not a number.
fn env_secs(key: &str, default: Duration) -> Duration {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map_or(default, Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_safe_side() {
        let config = CopilotSdkConfig::default();
        assert_eq!(config.permission_policy, PermissionPolicy::DenyAll);
        assert!(!config.mcp_tool_calling);
        assert_eq!(config.max_history_turns, DEFAULT_MAX_HISTORY_TURNS);
        assert_eq!(config.model, preferred_default());
        assert!(config.runtime_path.is_none());
        assert_eq!(config.prompt_timeout, Duration::from_mins(5));
        assert_eq!(config.session_timeout, Duration::from_mins(1));
        assert!(config
            .base_directory
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("embacle-copilot-sdk-")));
    }

    #[test]
    fn env_secs_ignores_garbage() {
        assert_eq!(
            env_secs("EMBACLE_SDK_TEST_NEVER_SET", Duration::from_secs(7)),
            Duration::from_secs(7)
        );
    }
}
