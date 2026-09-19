// ABOUTME: Factory for creating LlmProvider instances from runner type identifiers
// ABOUTME: Centralizes binary resolution and runner construction for all CLI runners
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::sync::LazyLock;

use crate::config::{CliRunnerType, RunnerConfig};
use crate::discovery::resolve_binary;
use crate::types::{LlmProvider, RunnerError};
use crate::{
    ClaudeCodeRunner, ClineCliRunner, CodexCliRunner, ContinueCliRunner, CopilotRunner,
    CursorAgentRunner, GeminiCliRunner, GooseCliRunner, KiloCliRunner, KiroCliRunner,
    OpenCodeRunner, WarpCliRunner,
};

/// Create an `LlmProvider` instance for the given runner type
///
/// Resolves the CLI binary via environment variable override or PATH lookup,
/// then constructs the appropriate runner with default configuration.
///
/// # Errors
///
/// Returns [`RunnerError`] if the CLI binary cannot be found.
pub async fn create_runner(
    runner_type: CliRunnerType,
) -> Result<Box<dyn LlmProvider>, RunnerError> {
    // CopilotSdk uses its own config (env-based), not RunnerConfig: the
    // runtime is resolved by the SDK, not discovered on PATH
    #[cfg(feature = "copilot-sdk")]
    if runner_type == CliRunnerType::CopilotSdk {
        return Ok(Box::new(crate::CopilotSdkRunner::from_env()));
    }

    // ClaudeWeb drives a browser, not a PATH binary — env-based config
    #[cfg(feature = "web-ui")]
    if runner_type == CliRunnerType::ClaudeWeb {
        return Ok(Box::new(crate::WebUiRunner::from_env()?));
    }

    let binary_name = runner_type.binary_name();
    let env_key = runner_type.env_override_key();
    let env_override = env::var(env_key).ok();

    let binary_path = resolve_binary(binary_name, env_override.as_deref())?;
    let config = RunnerConfig::new(binary_path);

    let runner: Box<dyn LlmProvider> = match runner_type {
        CliRunnerType::ClaudeCode => Box::new(ClaudeCodeRunner::new(config)),
        CliRunnerType::Copilot => Box::new(CopilotRunner::new(config)),
        CliRunnerType::CursorAgent => Box::new(CursorAgentRunner::new(config)),
        CliRunnerType::OpenCode => Box::new(OpenCodeRunner::new(config)),
        CliRunnerType::GeminiCli => Box::new(GeminiCliRunner::new(config)),
        CliRunnerType::CodexCli => Box::new(CodexCliRunner::new(config)),
        CliRunnerType::GooseCli => Box::new(GooseCliRunner::new(config)),
        CliRunnerType::ClineCli => Box::new(ClineCliRunner::new(config)),
        CliRunnerType::ContinueCli => Box::new(ContinueCliRunner::new(config)),
        CliRunnerType::WarpCli => Box::new(WarpCliRunner::new(config)),
        CliRunnerType::KiroCli => Box::new(KiroCliRunner::new(config)),
        CliRunnerType::KiloCli => Box::new(KiloCliRunner::new(config)),
        #[cfg(feature = "copilot-sdk")]
        CliRunnerType::CopilotSdk => unreachable!("handled above"),
        #[cfg(feature = "web-ui")]
        CliRunnerType::ClaudeWeb => unreachable!("handled above"),
    };

    Ok(runner)
}

/// Create an `LlmProvider` instance for the given runner type with a pre-built config.
///
/// Unlike [`create_runner()`], this function does not perform binary discovery;
/// it uses the provided `RunnerConfig` directly.
///
/// # Panics
///
/// With the `web-ui` feature, panics only if the compiled-in `claude_web`
/// provider TOML is malformed — a compile-time constant guarded by a unit test,
/// so never in practice.
pub async fn create_runner_with_config(
    runner_type: CliRunnerType,
    config: RunnerConfig,
) -> Result<Box<dyn LlmProvider>, RunnerError> {
    Ok(match runner_type {
        CliRunnerType::ClaudeCode => Box::new(ClaudeCodeRunner::new(config)),
        CliRunnerType::Copilot => Box::new(CopilotRunner::new(config)),
        CliRunnerType::CursorAgent => Box::new(CursorAgentRunner::new(config)),
        CliRunnerType::OpenCode => Box::new(OpenCodeRunner::new(config)),
        CliRunnerType::GeminiCli => Box::new(GeminiCliRunner::new(config)),
        CliRunnerType::CodexCli => Box::new(CodexCliRunner::new(config)),
        CliRunnerType::GooseCli => Box::new(GooseCliRunner::new(config)),
        CliRunnerType::ClineCli => Box::new(ClineCliRunner::new(config)),
        CliRunnerType::ContinueCli => Box::new(ContinueCliRunner::new(config)),
        CliRunnerType::WarpCli => Box::new(WarpCliRunner::new(config)),
        CliRunnerType::KiroCli => Box::new(KiroCliRunner::new(config)),
        CliRunnerType::KiloCli => Box::new(KiloCliRunner::new(config)),
        // CopilotSdk ignores RunnerConfig — uses env-based config
        #[cfg(feature = "copilot-sdk")]
        CliRunnerType::CopilotSdk => Box::new(crate::CopilotSdkRunner::from_env()),
        // ClaudeWeb ignores RunnerConfig — drives a browser via the embedded provider config
        #[cfg(feature = "web-ui")]
        CliRunnerType::ClaudeWeb => {
            let provider = crate::WebProviderConfig::claude_web_default()?;
            Box::new(crate::WebUiRunner::new(
                crate::WebUiConfig::from_env(),
                provider,
            ))
        }
    })
}

/// All provider types supported by embacle, in discovery priority order.
///
/// One list, with the feature-gated Copilot SDK runner slotted in behind the
/// Copilot CLI: a `const` slice cannot carry `#[cfg]` on its elements, and a
/// list per feature combination would drift.
pub static ALL_PROVIDERS: LazyLock<Vec<CliRunnerType>> = LazyLock::new(|| {
    let mut providers = vec![CliRunnerType::ClaudeCode, CliRunnerType::Copilot];
    #[cfg(feature = "copilot-sdk")]
    providers.push(CliRunnerType::CopilotSdk);
    providers.extend([
        CliRunnerType::CursorAgent,
        CliRunnerType::OpenCode,
        CliRunnerType::GeminiCli,
        CliRunnerType::CodexCli,
        CliRunnerType::GooseCli,
        CliRunnerType::ClineCli,
        CliRunnerType::ContinueCli,
        CliRunnerType::WarpCli,
        CliRunnerType::KiroCli,
        CliRunnerType::KiloCli,
    ]);
    providers
});

/// Parse a provider name string into a `CliRunnerType`
///
/// Accepts multiple naming conventions: `snake_case`, kebab-case, and
/// short forms for flexible input.
pub fn parse_runner_type(s: &str) -> Option<CliRunnerType> {
    match s.to_lowercase().as_str() {
        "claude_code" | "claude" | "claudecode" => Some(CliRunnerType::ClaudeCode),
        "copilot" => Some(CliRunnerType::Copilot),
        "cursor_agent" | "cursoragent" | "cursor-agent" => Some(CliRunnerType::CursorAgent),
        "opencode" | "open_code" => Some(CliRunnerType::OpenCode),
        "gemini" | "gemini_cli" | "geminicli" | "gemini-cli" => Some(CliRunnerType::GeminiCli),
        "codex" | "codex_cli" | "codexcli" | "codex-cli" => Some(CliRunnerType::CodexCli),
        "goose" | "goose_cli" | "goosecli" | "goose-cli" => Some(CliRunnerType::GooseCli),
        "cline" | "cline_cli" | "clinecli" | "cline-cli" => Some(CliRunnerType::ClineCli),
        "continue" | "continue_cli" | "continuecli" | "continue-cli" | "cn" => {
            Some(CliRunnerType::ContinueCli)
        }
        "warp" | "warp_cli" | "warpcli" | "warp-cli" | "oz" => Some(CliRunnerType::WarpCli),
        "kiro" | "kiro_cli" | "kirocli" | "kiro-cli" => Some(CliRunnerType::KiroCli),
        "kilo" | "kilo_cli" | "kilocli" | "kilo-cli" | "kilocode" => Some(CliRunnerType::KiloCli),
        #[cfg(feature = "copilot-sdk")]
        "copilot_sdk" | "copilot-sdk" | "copilotsdk" => Some(CliRunnerType::CopilotSdk),
        #[cfg(feature = "web-ui")]
        "claude_web" | "claude-web" | "claudeweb" | "web" => Some(CliRunnerType::ClaudeWeb),
        _ => None,
    }
}

/// The valid provider names for error messages, comma-separated, derived
/// from [`ALL_PROVIDERS`] so the list cannot drift from what the crate serves.
#[must_use]
pub fn valid_provider_names() -> String {
    ALL_PROVIDERS
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_snake_case_variants() {
        assert_eq!(
            parse_runner_type("claude_code"),
            Some(CliRunnerType::ClaudeCode)
        );
        assert_eq!(parse_runner_type("copilot"), Some(CliRunnerType::Copilot));
        assert_eq!(
            parse_runner_type("cursor_agent"),
            Some(CliRunnerType::CursorAgent)
        );
        assert_eq!(parse_runner_type("opencode"), Some(CliRunnerType::OpenCode));
        assert_eq!(
            parse_runner_type("gemini_cli"),
            Some(CliRunnerType::GeminiCli)
        );
        assert_eq!(
            parse_runner_type("codex_cli"),
            Some(CliRunnerType::CodexCli)
        );
        assert_eq!(
            parse_runner_type("goose_cli"),
            Some(CliRunnerType::GooseCli)
        );
        assert_eq!(
            parse_runner_type("cline_cli"),
            Some(CliRunnerType::ClineCli)
        );
        assert_eq!(
            parse_runner_type("continue_cli"),
            Some(CliRunnerType::ContinueCli)
        );
        assert_eq!(parse_runner_type("warp_cli"), Some(CliRunnerType::WarpCli));
        assert_eq!(parse_runner_type("kiro_cli"), Some(CliRunnerType::KiroCli));
        assert_eq!(parse_runner_type("kilo_cli"), Some(CliRunnerType::KiloCli));
    }

    #[test]
    fn parse_short_forms() {
        assert_eq!(parse_runner_type("claude"), Some(CliRunnerType::ClaudeCode));
        assert_eq!(
            parse_runner_type("cursor-agent"),
            Some(CliRunnerType::CursorAgent)
        );
        assert_eq!(parse_runner_type("gemini"), Some(CliRunnerType::GeminiCli));
        assert_eq!(parse_runner_type("codex"), Some(CliRunnerType::CodexCli));
        assert_eq!(parse_runner_type("goose"), Some(CliRunnerType::GooseCli));
        assert_eq!(parse_runner_type("cline"), Some(CliRunnerType::ClineCli));
        assert_eq!(
            parse_runner_type("continue"),
            Some(CliRunnerType::ContinueCli)
        );
        assert_eq!(parse_runner_type("cn"), Some(CliRunnerType::ContinueCli));
        assert_eq!(parse_runner_type("warp"), Some(CliRunnerType::WarpCli));
        assert_eq!(parse_runner_type("oz"), Some(CliRunnerType::WarpCli));
        assert_eq!(parse_runner_type("kiro"), Some(CliRunnerType::KiroCli));
        assert_eq!(parse_runner_type("kiro-cli"), Some(CliRunnerType::KiroCli));
        assert_eq!(parse_runner_type("kilo"), Some(CliRunnerType::KiloCli));
        assert_eq!(parse_runner_type("kilo-cli"), Some(CliRunnerType::KiloCli));
        assert_eq!(parse_runner_type("kilocode"), Some(CliRunnerType::KiloCli));
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!(parse_runner_type("COPILOT"), Some(CliRunnerType::Copilot));
        assert_eq!(
            parse_runner_type("Claude_Code"),
            Some(CliRunnerType::ClaudeCode)
        );
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(parse_runner_type("gpt4"), None);
        assert_eq!(parse_runner_type(""), None);
    }

    #[test]
    fn all_providers_count() {
        let expected = 12 + usize::from(cfg!(feature = "copilot-sdk"));
        assert_eq!(ALL_PROVIDERS.len(), expected);
    }

    #[test]
    fn valid_provider_names_lists_every_provider_once() {
        let names = valid_provider_names();
        let listed: Vec<&str> = names.split(", ").collect();
        for provider in ALL_PROVIDERS.iter() {
            let name = provider.to_string();
            assert_eq!(
                listed.iter().filter(|n| **n == name).count(),
                1,
                "{provider} appears once in {names}"
            );
        }
        assert!(names.starts_with("claude_code, copilot"));
    }
}
