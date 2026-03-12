// ABOUTME: Version compatibility and capability detection for CLI LLM runners
// ABOUTME: Probes installed CLI binaries to determine supported features and minimum version
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::path::Path;
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, warn};

use crate::config::CliRunnerType;
use crate::types::RunnerError;

/// Minimum supported version per CLI runner
const CLAUDE_CODE_MIN_VERSION: &str = "1.0.0";
const COPILOT_MIN_VERSION: &str = "0.0.1";
const CURSOR_AGENT_MIN_VERSION: &str = "0.1.0";
const OPENCODE_MIN_VERSION: &str = "0.1.0";
const GEMINI_CLI_MIN_VERSION: &str = "0.1.0";
const CODEX_CLI_MIN_VERSION: &str = "0.1.0";
const GOOSE_CLI_MIN_VERSION: &str = "1.0.0";
const CLINE_CLI_MIN_VERSION: &str = "2.0.0";
const CONTINUE_CLI_MIN_VERSION: &str = "1.0.0";
const WARP_CLI_MIN_VERSION: &str = "0.1.0";
const KIRO_CLI_MIN_VERSION: &str = "1.0.0";
const KILO_CLI_MIN_VERSION: &str = "7.0.0";

/// Detected capabilities of a CLI runner binary
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct CliCapabilities {
    /// CLI runner type
    pub runner_type: CliRunnerType,
    /// Detected version string (raw output from --version)
    pub version_string: String,
    /// Parsed semantic version components (major, minor, patch)
    pub version: Option<(u32, u32, u32)>,
    /// Whether JSON output mode is supported
    pub json_output: bool,
    /// Whether streaming JSON output is supported
    pub streaming: bool,
    /// Whether system prompt flag is supported
    pub system_prompt: bool,
    /// Whether session resume is supported
    pub session_resume: bool,
    /// Whether the binary meets the minimum version requirement
    pub meets_minimum_version: bool,
}

impl CliCapabilities {
    /// Check if this binary is fully compatible with the runner
    #[must_use]
    pub const fn is_compatible(&self) -> bool {
        self.meets_minimum_version && self.json_output
    }
}

/// Detect capabilities of an installed CLI binary
///
/// Runs the binary with `--version` to obtain the version string, then maps
/// known capabilities for the runner type. Feature flags (JSON output,
/// streaming, system prompt, session resume) are determined from a static
/// capability table keyed by runner type and version.
///
/// # Errors
///
/// Returns an error if the binary cannot be executed.
pub async fn detect_capabilities(
    runner_type: CliRunnerType,
    binary_path: &Path,
) -> Result<CliCapabilities, RunnerError> {
    let version_string = detect_version(binary_path, runner_type).await?;
    let parsed_version = parse_semver(&version_string);
    let min_version = minimum_version(runner_type);
    let meets_minimum = parsed_version.is_some_and(|v| compare_versions(v, min_version));

    let (json_output, streaming, system_prompt, session_resume) =
        capabilities_for_runner(runner_type);

    if !meets_minimum {
        warn!(
            runner = %runner_type,
            detected = %version_string,
            minimum = format!("{}.{}.{}", min_version.0, min_version.1, min_version.2),
            "CLI binary version is below minimum supported version",
        );
    }

    Ok(CliCapabilities {
        runner_type,
        version_string,
        version: parsed_version,
        json_output,
        streaming,
        system_prompt,
        session_resume,
        meets_minimum_version: meets_minimum,
    })
}

/// Run the CLI binary with `--version` and return the version string
async fn detect_version(
    binary_path: &Path,
    runner_type: CliRunnerType,
) -> Result<String, RunnerError> {
    let version_flag = match runner_type {
        CliRunnerType::OpenCode => "version",
        CliRunnerType::ClaudeCode
        | CliRunnerType::Copilot
        | CliRunnerType::CursorAgent
        | CliRunnerType::GeminiCli
        | CliRunnerType::CodexCli
        | CliRunnerType::GooseCli
        | CliRunnerType::ClineCli
        | CliRunnerType::ContinueCli
        | CliRunnerType::WarpCli
        | CliRunnerType::KiroCli
        | CliRunnerType::KiloCli => "--version",
    };

    let output = Command::new(binary_path)
        .arg(version_flag)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| {
            RunnerError::external_service(
                runner_type.binary_name(),
                format!("failed to run version check: {e}"),
            )
        })?;

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if raw.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        debug!(
            runner = %runner_type,
            stderr = %stderr,
            "Version check returned empty stdout",
        );
        return Ok(stderr);
    }
    Ok(raw)
}

/// Parse a semantic version string into (major, minor, patch)
///
/// Handles formats like "1.2.3", "v1.2.3", "claude 1.2.3", "opencode v0.3.1"
#[must_use]
pub fn parse_semver(version_str: &str) -> Option<(u32, u32, u32)> {
    let cleaned = version_str.split_whitespace().find(|word| {
        let stripped = word.strip_prefix('v').unwrap_or(word);
        let dot_count = stripped.split('.').count();
        dot_count >= 3
            && stripped.split('.').all(|part| {
                // Strip pre-release suffix (e.g., "0-rc1" → "0") before checking
                let numeric = part.split('-').next().unwrap_or(part);
                numeric.parse::<u32>().is_ok()
            })
    })?;

    let stripped = cleaned.strip_prefix('v').unwrap_or(cleaned);
    let mut parts = stripped.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_str = parts.next()?;
    // Handle pre-release suffixes like "1.0.0-rc1"
    let patch = patch_str.split('-').next()?.parse().ok()?;
    Some((major, minor, patch))
}

/// Get the minimum required version for a runner type
#[must_use]
const fn minimum_version(runner_type: CliRunnerType) -> (u32, u32, u32) {
    match runner_type {
        CliRunnerType::ClaudeCode => parse_const_version(CLAUDE_CODE_MIN_VERSION),
        CliRunnerType::Copilot => parse_const_version(COPILOT_MIN_VERSION),
        CliRunnerType::CursorAgent => parse_const_version(CURSOR_AGENT_MIN_VERSION),
        CliRunnerType::OpenCode => parse_const_version(OPENCODE_MIN_VERSION),
        CliRunnerType::GeminiCli => parse_const_version(GEMINI_CLI_MIN_VERSION),
        CliRunnerType::CodexCli => parse_const_version(CODEX_CLI_MIN_VERSION),
        CliRunnerType::GooseCli => parse_const_version(GOOSE_CLI_MIN_VERSION),
        CliRunnerType::ClineCli => parse_const_version(CLINE_CLI_MIN_VERSION),
        CliRunnerType::ContinueCli => parse_const_version(CONTINUE_CLI_MIN_VERSION),
        CliRunnerType::WarpCli => parse_const_version(WARP_CLI_MIN_VERSION),
        CliRunnerType::KiroCli => parse_const_version(KIRO_CLI_MIN_VERSION),
        CliRunnerType::KiloCli => parse_const_version(KILO_CLI_MIN_VERSION),
    }
}

/// Parse a version string at compile time (only handles simple "X.Y.Z" format)
const fn parse_const_version(s: &str) -> (u32, u32, u32) {
    let bytes = s.as_bytes();
    let mut major = 0u32;
    let mut minor = 0u32;
    let mut patch = 0u32;
    let mut dot_count = 0u8;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' {
            dot_count += 1;
        } else {
            let digit = (b - b'0') as u32;
            match dot_count {
                0 => major = major * 10 + digit,
                1 => minor = minor * 10 + digit,
                _ => patch = patch * 10 + digit,
            }
        }
        i += 1;
    }
    (major, minor, patch)
}

/// Compare two versions: returns true if `actual >= minimum`
#[must_use]
const fn compare_versions(actual: (u32, u32, u32), minimum: (u32, u32, u32)) -> bool {
    if actual.0 != minimum.0 {
        return actual.0 > minimum.0;
    }
    if actual.1 != minimum.1 {
        return actual.1 > minimum.1;
    }
    actual.2 >= minimum.2
}

/// Capability flags per runner type based on known CLI features
///
/// Returns (`json_output`, `streaming`, `system_prompt`, `session_resume`)
#[must_use]
const fn capabilities_for_runner(runner_type: CliRunnerType) -> (bool, bool, bool, bool) {
    match runner_type {
        // Claude Code: --output-format json, --output-format stream-json, --system-prompt, --continue
        CliRunnerType::ClaudeCode => (true, true, true, true),
        // Copilot: plain text output, line-by-line streaming, no --system-prompt, no session resume
        CliRunnerType::Copilot => (false, true, false, false),
        // Cursor Agent, Gemini CLI, Goose CLI, Cline CLI, Kilo CLI: JSON + streaming, no system prompt, session resume
        CliRunnerType::CursorAgent
        | CliRunnerType::GeminiCli
        | CliRunnerType::GooseCli
        | CliRunnerType::ClineCli
        | CliRunnerType::KiloCli => (true, true, false, true),
        // OpenCode, Continue CLI, Warp oz: JSON output, no streaming, session resume
        CliRunnerType::OpenCode | CliRunnerType::ContinueCli | CliRunnerType::WarpCli => {
            (true, false, false, true)
        }
        // Codex CLI: --json (JSONL), streaming via JSONL events
        CliRunnerType::CodexCli => (true, true, false, false),
        // Kiro CLI: plain text output (no JSON), no streaming, session resume via --resume
        CliRunnerType::KiroCli => (false, false, false, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_semver_simple() {
        assert_eq!(parse_semver("1.2.3"), Some((1, 2, 3)));
    }

    #[test]
    fn test_parse_semver_with_v_prefix() {
        assert_eq!(parse_semver("v1.2.3"), Some((1, 2, 3)));
    }

    #[test]
    fn test_parse_semver_with_name_prefix() {
        assert_eq!(parse_semver("claude 1.0.18"), Some((1, 0, 18)));
        assert_eq!(parse_semver("opencode v0.3.1"), Some((0, 3, 1)));
    }

    #[test]
    fn test_parse_semver_with_prerelease() {
        assert_eq!(parse_semver("1.0.0-rc1"), Some((1, 0, 0)));
    }

    #[test]
    fn test_parse_semver_invalid() {
        assert_eq!(parse_semver("not-a-version"), None);
        assert_eq!(parse_semver(""), None);
    }

    #[test]
    fn test_compare_versions_equal() {
        assert!(compare_versions((1, 0, 0), (1, 0, 0)));
    }

    #[test]
    fn test_compare_versions_newer() {
        assert!(compare_versions((2, 0, 0), (1, 0, 0)));
        assert!(compare_versions((1, 1, 0), (1, 0, 0)));
        assert!(compare_versions((1, 0, 1), (1, 0, 0)));
    }

    #[test]
    fn test_compare_versions_older() {
        assert!(!compare_versions((0, 9, 0), (1, 0, 0)));
        assert!(!compare_versions((1, 0, 0), (1, 0, 1)));
    }

    #[test]
    fn test_const_parse_version() {
        assert_eq!(parse_const_version("1.0.0"), (1, 0, 0));
        assert_eq!(parse_const_version("0.3.12"), (0, 3, 12));
    }

    #[test]
    fn test_capabilities_claude_code() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::ClaudeCode);
        assert!(json);
        assert!(stream);
        assert!(sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_cursor_agent() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::CursorAgent);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_opencode() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::OpenCode);
        assert!(json);
        assert!(!stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_kilo_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::KiloCli);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_cli_capabilities_compatible() {
        let caps = CliCapabilities {
            runner_type: CliRunnerType::ClaudeCode,
            version_string: "claude 1.0.18".to_owned(),
            version: Some((1, 0, 18)),
            json_output: true,
            streaming: true,
            system_prompt: true,
            session_resume: true,
            meets_minimum_version: true,
        };
        assert!(caps.is_compatible());
    }

    #[test]
    fn test_capabilities_gemini_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::GeminiCli);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_codex_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::CodexCli);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(!resume);
    }

    #[test]
    fn test_capabilities_goose_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::GooseCli);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_cline_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::ClineCli);
        assert!(json);
        assert!(stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_capabilities_continue_cli() {
        let (json, stream, sys, resume) = capabilities_for_runner(CliRunnerType::ContinueCli);
        assert!(json);
        assert!(!stream);
        assert!(!sys);
        assert!(resume);
    }

    #[test]
    fn test_cli_capabilities_incompatible_old_version() {
        let caps = CliCapabilities {
            runner_type: CliRunnerType::ClaudeCode,
            version_string: "claude 0.0.1".to_owned(),
            version: Some((0, 0, 1)),
            json_output: true,
            streaming: true,
            system_prompt: true,
            session_resume: true,
            meets_minimum_version: false,
        };
        assert!(!caps.is_compatible());
    }
}
