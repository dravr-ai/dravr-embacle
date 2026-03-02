// ABOUTME: Shared configuration types for CLI-based LLM runners
// ABOUTME: Defines runner types, runner configuration, and environment key parsing
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default timeout for CLI command execution (120 seconds)
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Supported CLI runner types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CliRunnerType {
    /// Claude Code CLI (`claude`)
    ClaudeCode,
    /// Cursor Agent CLI (`cursor-agent`)
    CursorAgent,
    /// `OpenCode` CLI (`opencode`)
    OpenCode,
    /// GitHub Copilot CLI (`copilot`)
    Copilot,
    /// Gemini CLI (`gemini`)
    GeminiCli,
    /// Codex CLI (`codex`)
    CodexCli,
}

impl CliRunnerType {
    /// Binary name used to locate the CLI tool on disk
    #[must_use]
    pub const fn binary_name(&self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude",
            Self::CursorAgent => "cursor-agent",
            Self::OpenCode => "opencode",
            Self::Copilot => "copilot",
            Self::GeminiCli => "gemini",
            Self::CodexCli => "codex",
        }
    }

    /// Environment variable that can override the binary path
    #[must_use]
    pub const fn env_override_key(&self) -> &'static str {
        match self {
            Self::ClaudeCode => "CLAUDE_CODE_BINARY",
            Self::CursorAgent => "CURSOR_AGENT_BINARY",
            Self::OpenCode => "OPENCODE_BINARY",
            Self::Copilot => "COPILOT_BINARY",
            Self::GeminiCli => "GEMINI_CLI_BINARY",
            Self::CodexCli => "CODEX_CLI_BINARY",
        }
    }
}

impl fmt::Display for CliRunnerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClaudeCode => write!(f, "claude_code"),
            Self::CursorAgent => write!(f, "cursor_agent"),
            Self::OpenCode => write!(f, "opencode"),
            Self::Copilot => write!(f, "copilot"),
            Self::GeminiCli => write!(f, "gemini_cli"),
            Self::CodexCli => write!(f, "codex_cli"),
        }
    }
}

/// Configuration for a CLI runner instance
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Path to the CLI binary
    pub binary_path: PathBuf,
    /// Model override (provider-specific format)
    pub model: Option<String>,
    /// Maximum time to wait for a CLI command to complete
    pub timeout: Duration,
    /// Additional CLI arguments appended to every invocation
    pub extra_args: Vec<String>,
    /// Environment variable keys passed through to the subprocess
    pub allowed_env_keys: Vec<String>,
    /// Working directory for the subprocess
    pub working_directory: Option<PathBuf>,
}

impl RunnerConfig {
    /// Create a new runner configuration with the given binary path
    #[must_use]
    pub fn new(binary_path: PathBuf) -> Self {
        Self {
            binary_path,
            model: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            extra_args: Vec::new(),
            allowed_env_keys: default_allowed_env_keys(),
            working_directory: None,
        }
    }

    /// Set the model to use
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the command timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set extra CLI arguments
    #[must_use]
    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    /// Set the environment variable keys passed through to the subprocess
    #[must_use]
    pub fn with_allowed_env_keys(mut self, keys: Vec<String>) -> Self {
        self.allowed_env_keys = keys;
        self
    }

    /// Set the working directory for the subprocess
    #[must_use]
    pub fn with_working_directory(mut self, dir: PathBuf) -> Self {
        self.working_directory = Some(dir);
        self
    }
}

/// Default set of environment variable keys safe to pass through to subprocesses
#[must_use]
pub fn default_allowed_env_keys() -> Vec<String> {
    ["HOME", "PATH", "TERM", "USER", "LANG"]
        .iter()
        .map(|k| (*k).to_owned())
        .collect()
}

/// Parse a comma-separated list of environment variable keys
#[must_use]
pub fn parse_env_keys(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

use std::num::ParseIntError;

/// Parse a timeout value from a string (in seconds)
///
/// # Errors
///
/// Returns an error if the string cannot be parsed as a `u64`.
pub fn parse_timeout(input: &str) -> Result<Duration, ParseIntError> {
    input.trim().parse::<u64>().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_config_defaults() {
        let config = RunnerConfig::new(PathBuf::from("/usr/bin/claude"));
        assert_eq!(config.binary_path, PathBuf::from("/usr/bin/claude"));
        assert!(config.model.is_none());
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert!(config.extra_args.is_empty());
        assert!(config.working_directory.is_none());
    }

    #[test]
    fn test_runner_config_builder() {
        let config = RunnerConfig::new(PathBuf::from("claude"))
            .with_model("opus")
            .with_timeout(Duration::from_secs(60))
            .with_extra_args(vec!["--verbose".to_owned()])
            .with_working_directory(PathBuf::from("/tmp"));

        assert_eq!(config.model.as_deref(), Some("opus"));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.extra_args, vec!["--verbose"]);
        assert_eq!(config.working_directory, Some(PathBuf::from("/tmp")));
    }

    #[test]
    fn test_default_allowed_env_keys() {
        let keys = default_allowed_env_keys();
        assert!(keys.contains(&"HOME".to_owned()));
        assert!(keys.contains(&"PATH".to_owned()));
        assert!(keys.contains(&"TERM".to_owned()));
        assert!(keys.contains(&"USER".to_owned()));
        assert!(keys.contains(&"LANG".to_owned()));
        assert_eq!(keys.len(), 5);
    }

    #[test]
    fn test_parse_env_keys_basic() {
        let keys = parse_env_keys("FOO,BAR,BAZ");
        assert_eq!(keys, vec!["FOO", "BAR", "BAZ"]);
    }

    #[test]
    fn test_parse_env_keys_with_whitespace() {
        let keys = parse_env_keys(" FOO , BAR , BAZ ");
        assert_eq!(keys, vec!["FOO", "BAR", "BAZ"]);
    }

    #[test]
    fn test_parse_env_keys_empty_string() {
        let keys = parse_env_keys("");
        assert!(keys.is_empty());
    }

    #[test]
    fn test_parse_env_keys_trailing_comma() {
        let keys = parse_env_keys("FOO,BAR,");
        assert_eq!(keys, vec!["FOO", "BAR"]);
    }

    #[test]
    fn test_parse_timeout_valid() {
        assert_eq!(parse_timeout("60"), Ok(Duration::from_secs(60)));
        assert_eq!(parse_timeout("  120  "), Ok(Duration::from_secs(120)));
    }

    #[test]
    fn test_parse_timeout_invalid() {
        assert!(parse_timeout("abc").is_err());
        assert!(parse_timeout("").is_err());
    }

    #[test]
    fn test_cli_runner_type_binary_names() {
        assert_eq!(CliRunnerType::ClaudeCode.binary_name(), "claude");
        assert_eq!(CliRunnerType::CursorAgent.binary_name(), "cursor-agent");
        assert_eq!(CliRunnerType::OpenCode.binary_name(), "opencode");
        assert_eq!(CliRunnerType::Copilot.binary_name(), "copilot");
        assert_eq!(CliRunnerType::GeminiCli.binary_name(), "gemini");
        assert_eq!(CliRunnerType::CodexCli.binary_name(), "codex");
    }

    #[test]
    fn test_cli_runner_type_env_keys() {
        assert_eq!(
            CliRunnerType::ClaudeCode.env_override_key(),
            "CLAUDE_CODE_BINARY"
        );
        assert_eq!(CliRunnerType::Copilot.env_override_key(), "COPILOT_BINARY");
        assert_eq!(
            CliRunnerType::GeminiCli.env_override_key(),
            "GEMINI_CLI_BINARY"
        );
        assert_eq!(
            CliRunnerType::CodexCli.env_override_key(),
            "CODEX_CLI_BINARY"
        );
    }

    #[test]
    fn test_cli_runner_type_display() {
        assert_eq!(format!("{}", CliRunnerType::ClaudeCode), "claude_code");
        assert_eq!(format!("{}", CliRunnerType::Copilot), "copilot");
        assert_eq!(format!("{}", CliRunnerType::CursorAgent), "cursor_agent");
        assert_eq!(format!("{}", CliRunnerType::OpenCode), "opencode");
        assert_eq!(format!("{}", CliRunnerType::GeminiCli), "gemini_cli");
        assert_eq!(format!("{}", CliRunnerType::CodexCli), "codex_cli");
    }
}
