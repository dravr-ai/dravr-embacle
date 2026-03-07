// ABOUTME: Configuration for the Copilot Headless (ACP) provider.
// ABOUTME: Reads environment variables and provides defaults for the ACP client.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;

/// Configuration for the Copilot Headless (ACP) provider.
#[derive(Debug, Clone)]
pub struct CopilotHeadlessConfig {
    /// Override path to the copilot CLI binary (default: auto-detect via PATH).
    pub cli_path: Option<PathBuf>,
    /// Default model to use for completions.
    pub model: String,
    /// GitHub token for authentication (optional, uses stored OAuth by default).
    pub github_token: Option<String>,
}

impl CopilotHeadlessConfig {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `COPILOT_CLI_PATH` — Override path to copilot binary
    /// - `COPILOT_HEADLESS_MODEL` — Default model (default: `claude-opus-4.6-fast`)
    /// - `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` — GitHub auth token
    #[must_use]
    pub fn from_env() -> Self {
        let cli_path = env::var("COPILOT_CLI_PATH").ok().map(PathBuf::from);

        let model = env::var("COPILOT_HEADLESS_MODEL")
            .unwrap_or_else(|_| "claude-opus-4.6-fast".to_owned());

        let github_token = env::var("COPILOT_GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .or_else(|_| env::var("GITHUB_TOKEN"))
            .ok();

        Self {
            cli_path,
            model,
            github_token,
        }
    }
}

impl Default for CopilotHeadlessConfig {
    fn default() -> Self {
        Self {
            cli_path: None,
            model: "claude-opus-4.6-fast".to_owned(),
            github_token: None,
        }
    }
}
