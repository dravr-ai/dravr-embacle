// ABOUTME: Configuration for the Copilot Headless (ACP) provider.
// ABOUTME: Reads environment variables and provides defaults for the ACP client.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;

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
    /// Re-inject the system prompt into the prompt text wrapped in
    /// `<system-instructions>` tags, in addition to passing it via `session/new`.
    /// Default: false (rely on the ACP `systemPrompt` parameter only).
    pub inject_system_in_prompt: bool,
}

impl CopilotHeadlessConfig {
    /// Create configuration from environment variables.
    ///
    /// Environment variables:
    /// - `COPILOT_CLI_PATH` — Override path to copilot binary
    /// - `COPILOT_HEADLESS_MODEL` — Default model (default: `claude-opus-4.6-fast`)
    /// - `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_TOKEN` — GitHub auth token
    /// - `COPILOT_HEADLESS_MAX_HISTORY_TURNS` — Max conversation history turns (default: 20)
    /// - `COPILOT_HEADLESS_INJECT_SYSTEM_IN_PROMPT` — Re-inject system prompt in prompt text (default: false)
    #[must_use]
    pub fn from_env() -> Self {
        let cli_path = env::var("COPILOT_CLI_PATH").ok().map(PathBuf::from);

        let model = env::var("COPILOT_HEADLESS_MODEL")
            .unwrap_or_else(|_| "claude-opus-4.6-fast".to_owned());

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

        let inject_system_in_prompt = env::var("COPILOT_HEADLESS_INJECT_SYSTEM_IN_PROMPT")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        Self {
            cli_path,
            model,
            github_token,
            permission_policy,
            max_history_turns,
            inject_system_in_prompt,
        }
    }
}

impl Default for CopilotHeadlessConfig {
    fn default() -> Self {
        Self {
            cli_path: None,
            model: "claude-opus-4.6-fast".to_owned(),
            github_token: None,
            permission_policy: PermissionPolicy::default(),
            max_history_turns: DEFAULT_MAX_HISTORY_TURNS,
            inject_system_in_prompt: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_max_history_turns_is_20() {
        assert_eq!(DEFAULT_MAX_HISTORY_TURNS, 20);
    }

    #[test]
    fn default_config_uses_default_max_history_turns() {
        let config = CopilotHeadlessConfig::default();
        assert_eq!(config.max_history_turns, 20);
    }

    #[test]
    fn config_max_history_turns_overridable() {
        let config = CopilotHeadlessConfig {
            max_history_turns: 50,
            ..CopilotHeadlessConfig::default()
        };
        assert_eq!(config.max_history_turns, 50);
    }

    #[test]
    fn config_max_history_turns_can_be_zero() {
        let config = CopilotHeadlessConfig {
            max_history_turns: 0,
            ..CopilotHeadlessConfig::default()
        };
        assert_eq!(config.max_history_turns, 0);
    }

    #[test]
    fn default_inject_system_in_prompt_is_false() {
        let config = CopilotHeadlessConfig::default();
        assert!(!config.inject_system_in_prompt);
    }

    /// Env var tests run sequentially in a single test to avoid race conditions
    /// (env vars are process-global state shared across parallel test threads).
    #[test]
    fn from_env_max_history_turns_parsing() {
        let key = "COPILOT_HEADLESS_MAX_HISTORY_TURNS";

        // Default when env var is not set
        env::remove_var(key);
        let config = CopilotHeadlessConfig::from_env();
        assert_eq!(
            config.max_history_turns, DEFAULT_MAX_HISTORY_TURNS,
            "should use default when env var absent"
        );

        // Valid integer value
        env::set_var(key, "42");
        let config = CopilotHeadlessConfig::from_env();
        assert_eq!(config.max_history_turns, 42, "should parse valid integer");

        // Invalid value falls back to default
        env::set_var(key, "not_a_number");
        let config = CopilotHeadlessConfig::from_env();
        assert_eq!(
            config.max_history_turns, DEFAULT_MAX_HISTORY_TURNS,
            "should fall back to default on invalid input"
        );

        // Zero is a valid value (disables history)
        env::set_var(key, "0");
        let config = CopilotHeadlessConfig::from_env();
        assert_eq!(
            config.max_history_turns, 0,
            "should accept zero to disable history"
        );

        // Cleanup
        env::remove_var(key);
    }
}
