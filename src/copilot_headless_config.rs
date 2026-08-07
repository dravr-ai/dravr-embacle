// ABOUTME: Configuration for the Copilot Headless (ACP) provider.
// ABOUTME: Reads environment variables and provides defaults for the ACP client.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;

use crate::copilot_models::preferred_default;

/// Policy for handling ACP permission requests from the copilot subprocess.
///
/// Controls whether tool-execution permission prompts are auto-approved or denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionPolicy {
    /// Automatically approve permission requests by selecting the best allow option.
    #[default]
    AutoApprove,
    /// Deny all permission requests by cancelling them.
    DenyAll,
}

/// Default number of conversation history turns injected into the ACP prompt.
/// Each "turn" is one user or assistant message. Override with
/// `COPILOT_HEADLESS_MAX_HISTORY_TURNS`.
pub const DEFAULT_MAX_HISTORY_TURNS: usize = 20;

/// Configuration for the Copilot Headless (ACP) provider.
#[derive(Debug, Clone)]
pub struct CopilotHeadlessConfig {
    /// Override path to the copilot CLI binary (default: auto-detect via PATH).
    pub cli_path: Option<PathBuf>,
    /// Default model to use for completions.
    pub model: String,
    /// GitHub token for authentication (optional, uses stored OAuth by default).
    pub github_token: Option<String>,
    /// Policy for handling permission requests from the copilot subprocess.
    pub permission_policy: PermissionPolicy,
    /// Maximum number of prior conversation messages (user + assistant) to include
    /// in the ACP prompt for multi-turn context. Set to 0 to disable history injection.
    pub max_history_turns: usize,
    /// Advertise `SDK_TOOL_CALLING` so callers route tool turns through the
    /// ACP `converse()` loop with per-request MCP servers (native tool
    /// calling), instead of falling through to text-based `<tool_call>`
    /// parsing. The caller must pass `mcp_servers` on each request for tools
    /// to be reachable. Default: false.
    pub mcp_tool_calling: bool,
}

impl CopilotHeadlessConfig {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `COPILOT_CLI_PATH` — Override path to copilot binary
    /// - `COPILOT_HEADLESS_MODEL` — Default model (defaults to the top-ranked
    ///   candidate in [`crate::copilot_models::CATALOG`])
    /// - `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` — GitHub auth token
    /// - `COPILOT_HEADLESS_MAX_HISTORY_TURNS` — Max conversation history turns (default: 20)
    /// - `COPILOT_HEADLESS_MCP_TOOL_CALLING` — Advertise `SDK_TOOL_CALLING` for native ACP MCP tool calling (default: false)
    #[must_use]
    pub fn from_env() -> Self {
        let cli_path = env::var("COPILOT_CLI_PATH").ok().map(PathBuf::from);

        let model =
            env::var("COPILOT_HEADLESS_MODEL").unwrap_or_else(|_| preferred_default().to_owned());

        let github_token = env::var("COPILOT_GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .or_else(|_| env::var("GITHUB_TOKEN"))
            .ok();

        let permission_policy = match env::var("COPILOT_HEADLESS_PERMISSION_POLICY")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "deny_all" | "denyall" | "deny" => PermissionPolicy::DenyAll,
            _ => PermissionPolicy::AutoApprove,
        };

        let max_history_turns = env::var("COPILOT_HEADLESS_MAX_HISTORY_TURNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_HISTORY_TURNS);

        let mcp_tool_calling = env::var("COPILOT_HEADLESS_MCP_TOOL_CALLING")
            .is_ok_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"));

        Self {
            cli_path,
            model,
            github_token,
            permission_policy,
            max_history_turns,
            mcp_tool_calling,
        }
    }
}

impl Default for CopilotHeadlessConfig {
    fn default() -> Self {
        Self {
            cli_path: None,
            model: preferred_default().to_owned(),
            github_token: None,
            permission_policy: PermissionPolicy::default(),
            max_history_turns: DEFAULT_MAX_HISTORY_TURNS,
            mcp_tool_calling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
