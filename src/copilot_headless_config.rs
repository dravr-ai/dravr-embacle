// ABOUTME: Configuration for the Copilot Headless (ACP) provider.
// ABOUTME: Reads environment variables and provides defaults for the ACP client.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;

use crate::copilot_common::{resolve_github_token, PermissionPolicy, DEFAULT_MAX_HISTORY_TURNS};
use crate::copilot_models::preferred_default;

/// Configuration for the Copilot Headless (ACP) provider.
#[derive(Debug, Clone)]
pub struct CopilotHeadlessConfig {
    /// Override path to the copilot CLI binary (default: auto-detect via PATH).
    pub cli_path: Option<PathBuf>,
    /// Directory every `copilot --acp` subprocess runs in and the `cwd` of every
    /// session it serves. Copilot treats that directory as the project — it
    /// loads `.mcp.json`, agent files and custom instructions from it — so a
    /// host that sets this hands the model that project. A relative path is
    /// anchored to the host's cwd. Unset, the runner uses a scratch directory
    /// under the system temp dir, not the host process's working directory.
    pub working_directory: Option<PathBuf>,
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
    /// - `COPILOT_HEADLESS_WORKING_DIR` — Directory the subprocess and its sessions
    ///   run in (default: a scratch directory under the system temp dir)
    /// - `COPILOT_HEADLESS_MODEL` — Default model (defaults to the top-ranked
    ///   candidate in [`crate::copilot_models::CATALOG`])
    /// - `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` — GitHub auth token
    /// - `COPILOT_HEADLESS_MAX_HISTORY_TURNS` — Max conversation history turns (default: 20)
    /// - `COPILOT_HEADLESS_MCP_TOOL_CALLING` — Advertise `SDK_TOOL_CALLING` for native ACP MCP tool calling (default: false)
    #[must_use]
    pub fn from_env() -> Self {
        let cli_path = env::var("COPILOT_CLI_PATH").ok().map(PathBuf::from);

        // An empty value reads as unset, so `export COPILOT_HEADLESS_WORKING_DIR=`
        // in a profile does not pin the subprocess to a relative "".
        let working_directory = env::var("COPILOT_HEADLESS_WORKING_DIR")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);

        let model =
            env::var("COPILOT_HEADLESS_MODEL").unwrap_or_else(|_| preferred_default().to_owned());

        let github_token = resolve_github_token();

        let permission_policy = PermissionPolicy::parse(
            &env::var("COPILOT_HEADLESS_PERMISSION_POLICY").unwrap_or_default(),
        );

        let max_history_turns = env::var("COPILOT_HEADLESS_MAX_HISTORY_TURNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_HISTORY_TURNS);

        let mcp_tool_calling = env::var("COPILOT_HEADLESS_MCP_TOOL_CALLING")
            .is_ok_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"));

        Self {
            cli_path,
            working_directory,
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
            working_directory: None,
            model: preferred_default().to_owned(),
            github_token: None,
            permission_policy: PermissionPolicy::default(),
            max_history_turns: DEFAULT_MAX_HISTORY_TURNS,
            mcp_tool_calling: false,
        }
    }
}
