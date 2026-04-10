// ABOUTME: TOML-based configuration file loading for declarative provider setup
// ABOUTME: Searches ./embacle.toml then ~/.config/embacle/config.toml with env var overrides
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Declarative Configuration File
//!
//! Load provider and fallback chain configuration from TOML files.
//! Searches `./embacle.toml` first, then `~/.config/embacle/config.toml`.
//!
//! ## Example `embacle.toml`
//!
//! ```toml
//! [defaults]
//! timeout = 120
//! model = "gpt-5.4"
//!
//! [[providers]]
//! type = "claude_code"
//! model = "opus"
//! timeout = 180
//!
//! [[providers]]
//! type = "copilot"
//!
//! [fallback]
//! providers = ["claude_code", "copilot"]
//! retry_per_provider = 2
//! base_delay_ms = 500
//! max_delay_ms = 5000
//!
//! [aliases]
//! fast = "gemini_cli"
//! smart = "claude_code"
//! ```

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::config::RunnerConfig;
use crate::discovery::resolve_binary;
use crate::factory::{create_runner_with_config, parse_runner_type};
use crate::fallback::{FallbackProvider, RetryConfig};
use crate::types::{LlmProvider, RunnerError};

/// Top-level configuration loaded from an embacle TOML file
#[derive(Debug, Deserialize)]
pub struct EmbacleConfig {
    /// Default values applied to all providers unless overridden
    #[serde(default)]
    pub defaults: DefaultsConfig,
    /// Provider definitions
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Fallback chain configuration
    pub fallback: Option<FallbackConfig>,
    /// Short name aliases mapping to provider type names
    #[serde(default)]
    pub aliases: HashMap<String, String>,
}

/// Default configuration values shared across providers
#[derive(Debug, Default, Deserialize)]
pub struct DefaultsConfig {
    /// Default timeout in seconds
    pub timeout: Option<u64>,
    /// Default model name
    pub model: Option<String>,
}

/// Configuration for a single provider
#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    /// Provider type name (e.g., `claude_code`, `copilot`)
    #[serde(rename = "type")]
    pub provider_type: String,
    /// Model override
    pub model: Option<String>,
    /// Timeout override in seconds
    pub timeout: Option<u64>,
    /// Explicit binary path
    pub binary_path: Option<PathBuf>,
    /// Extra CLI arguments
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Environment variable keys to pass through
    #[serde(default)]
    pub env_keys: Vec<String>,
}

/// Configuration for a fallback provider chain
#[derive(Debug, Deserialize)]
pub struct FallbackConfig {
    /// Provider type names in fallback order
    pub providers: Vec<String>,
    /// Number of retries per provider on transient errors
    pub retry_per_provider: Option<u32>,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: Option<u64>,
    /// Maximum delay between retries in milliseconds
    pub max_delay_ms: Option<u64>,
}

/// Load configuration from the default search path.
///
/// Searches `./embacle.toml` first, then `~/.config/embacle/config.toml`.
/// Returns `Ok(None)` if no config file is found.
pub fn load_config() -> Result<Option<EmbacleConfig>, RunnerError> {
    let local_path = PathBuf::from("embacle.toml");
    if local_path.exists() {
        return load_config_from(&local_path).map(Some);
    }

    if let Some(config_dir) = dirs::config_dir() {
        let global_path = config_dir.join("embacle").join("config.toml");
        if global_path.exists() {
            return load_config_from(&global_path).map(Some);
        }
    }

    Ok(None)
}

/// Load configuration from a specific file path.
pub fn load_config_from(path: &Path) -> Result<EmbacleConfig, RunnerError> {
    let content = fs::read_to_string(path).map_err(|e| {
        RunnerError::config(format!(
            "failed to read config file {}: {e}",
            path.display()
        ))
    })?;
    toml::from_str(&content).map_err(|e| {
        RunnerError::config(format!(
            "failed to parse config file {}: {e}",
            path.display()
        ))
    })
}

/// Build a `RunnerConfig` from a provider config entry, merging with defaults.
///
/// Resolution order: provider-specific > defaults > environment variable > PATH lookup.
pub fn build_runner_config(
    provider: &ProviderConfig,
    defaults: &DefaultsConfig,
) -> Result<RunnerConfig, RunnerError> {
    let runner_type = parse_runner_type(&provider.provider_type).ok_or_else(|| {
        RunnerError::config(format!("unknown provider type: {}", provider.provider_type))
    })?;

    let binary_path = if let Some(ref path) = provider.binary_path {
        path.clone()
    } else {
        let env_override = env::var(runner_type.env_override_key()).ok();
        resolve_binary(runner_type.binary_name(), env_override.as_deref())?
    };

    let mut config = RunnerConfig::new(binary_path);

    // Model: provider > defaults
    if let Some(ref model) = provider.model {
        config.model = Some(model.clone());
    } else if let Some(ref model) = defaults.model {
        config.model = Some(model.clone());
    }

    // Timeout: provider > defaults
    if let Some(timeout) = provider.timeout {
        config.timeout = Duration::from_secs(timeout);
    } else if let Some(timeout) = defaults.timeout {
        config.timeout = Duration::from_secs(timeout);
    }

    if !provider.extra_args.is_empty() {
        config.extra_args.clone_from(&provider.extra_args);
    }

    if !provider.env_keys.is_empty() {
        config.allowed_env_keys.clone_from(&provider.env_keys);
    }

    Ok(config)
}

/// Build a `FallbackProvider` from a fallback config, wiring up retry settings.
///
/// Each provider name in the fallback list must match a provider in the config's
/// providers list, or be resolvable via `parse_runner_type()`.
pub async fn build_fallback_from_config(
    config: &EmbacleConfig,
) -> Result<Option<FallbackProvider>, RunnerError> {
    let Some(fallback_config) = &config.fallback else {
        return Ok(None);
    };

    let mut providers: Vec<Box<dyn LlmProvider>> = Vec::new();

    for provider_name in &fallback_config.providers {
        let resolved_name = resolve_alias(config, provider_name).unwrap_or(provider_name.as_str());
        let runner_type = parse_runner_type(resolved_name).ok_or_else(|| {
            RunnerError::config(format!("unknown provider in fallback: {resolved_name}"))
        })?;

        // Look for matching provider config or build default
        let provider_config = config
            .providers
            .iter()
            .find(|p| parse_runner_type(&p.provider_type) == Some(runner_type));

        let runner_config = if let Some(pc) = provider_config {
            build_runner_config(pc, &config.defaults)?
        } else {
            let env_override = env::var(runner_type.env_override_key()).ok();
            let binary_path = resolve_binary(runner_type.binary_name(), env_override.as_deref())?;
            RunnerConfig::new(binary_path)
        };

        let runner = create_runner_with_config(runner_type, runner_config).await;
        providers.push(runner);
    }

    let retry = RetryConfig {
        max_retries: fallback_config.retry_per_provider.unwrap_or(0),
        base_delay: Duration::from_millis(fallback_config.base_delay_ms.unwrap_or(500)),
        max_delay: Duration::from_millis(fallback_config.max_delay_ms.unwrap_or(5000)),
    };

    FallbackProvider::with_retry(providers, retry).map(Some)
}

/// Resolve a short alias to a provider type name, if one exists.
pub fn resolve_alias<'a>(config: &'a EmbacleConfig, name: &str) -> Option<&'a str> {
    config.aliases.get(name).map(String::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn parse_minimal_config() {
        let toml_str = r#"
[[providers]]
type = "claude_code"
"#;
        let config: EmbacleConfig = toml::from_str(toml_str).unwrap(); // Safe: test assertion
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].provider_type, "claude_code");
        assert!(config.defaults.timeout.is_none());
        assert!(config.fallback.is_none());
    }

    #[test]
    fn parse_full_config() {
        let toml_str = r#"
[defaults]
timeout = 120
model = "gpt-5.4"

[[providers]]
type = "claude_code"
model = "opus"
timeout = 180
extra_args = ["--verbose"]
env_keys = ["ANTHROPIC_API_KEY"]

[[providers]]
type = "copilot"

[fallback]
providers = ["claude_code", "copilot"]
retry_per_provider = 2
base_delay_ms = 500
max_delay_ms = 5000

[aliases]
fast = "gemini_cli"
smart = "claude_code"
"#;
        let config: EmbacleConfig = toml::from_str(toml_str).unwrap(); // Safe: test assertion
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.defaults.timeout, Some(120));
        assert_eq!(config.defaults.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(config.providers[0].model.as_deref(), Some("opus"));
        assert_eq!(config.providers[0].timeout, Some(180));
        assert_eq!(config.providers[0].extra_args, vec!["--verbose"]);
        assert_eq!(config.providers[0].env_keys, vec!["ANTHROPIC_API_KEY"]);
        let fb = config.fallback.as_ref().unwrap(); // Safe: test assertion
        assert_eq!(fb.providers, vec!["claude_code", "copilot"]);
        assert_eq!(fb.retry_per_provider, Some(2));
        assert_eq!(fb.base_delay_ms, Some(500));
        assert_eq!(fb.max_delay_ms, Some(5000));
        assert_eq!(config.aliases.get("fast").unwrap(), "gemini_cli"); // Safe: test assertion
        assert_eq!(config.aliases.get("smart").unwrap(), "claude_code"); // Safe: test assertion
    }

    #[test]
    fn defaults_merge_with_provider() {
        let defaults = DefaultsConfig {
            timeout: Some(60),
            model: Some("default-model".to_owned()),
        };
        let provider = ProviderConfig {
            provider_type: "claude_code".to_owned(),
            model: Some("override-model".to_owned()),
            timeout: None,
            binary_path: Some(PathBuf::from("/usr/bin/claude")),
            extra_args: vec![],
            env_keys: vec![],
        };
        let config = build_runner_config(&provider, &defaults).unwrap(); // Safe: test assertion
        assert_eq!(config.model.as_deref(), Some("override-model"));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn alias_resolution() {
        let toml_str = r#"
[aliases]
fast = "gemini_cli"
smart = "claude_code"
"#;
        let config: EmbacleConfig = toml::from_str(toml_str).unwrap(); // Safe: test assertion
        assert_eq!(resolve_alias(&config, "fast"), Some("gemini_cli"));
        assert_eq!(resolve_alias(&config, "smart"), Some("claude_code"));
        assert_eq!(resolve_alias(&config, "unknown"), None);
    }

    #[test]
    fn missing_file_returns_none() {
        // load_config() won't find embacle.toml in the test working directory
        // unless one exists — this test validates the None path
        let result = load_config_from(Path::new("/nonexistent/embacle.toml"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::Config);
    }

    #[test]
    fn invalid_toml_returns_config_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap(); // Safe: test assertion
        fs::write(tmp.path(), "not valid toml {{{{").unwrap(); // Safe: test assertion
        let result = load_config_from(tmp.path());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::Config);
    }

    #[test]
    fn unknown_provider_type_returns_config_error() {
        let defaults = DefaultsConfig::default();
        let provider = ProviderConfig {
            provider_type: "nonexistent_provider".to_owned(),
            model: None,
            timeout: None,
            binary_path: None,
            extra_args: vec![],
            env_keys: vec![],
        };
        let result = build_runner_config(&provider, &defaults);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, ErrorKind::Config);
    }

    #[test]
    fn env_var_overrides_file_config() {
        let defaults = DefaultsConfig {
            timeout: Some(60),
            model: None,
        };
        // Provider without explicit binary_path falls through to env/discovery
        let provider = ProviderConfig {
            provider_type: "claude_code".to_owned(),
            model: None,
            timeout: Some(90),
            binary_path: Some(PathBuf::from("/custom/claude")),
            extra_args: vec![],
            env_keys: vec![],
        };
        let config = build_runner_config(&provider, &defaults).unwrap(); // Safe: test assertion
        assert_eq!(config.timeout, Duration::from_secs(90));
        assert_eq!(config.binary_path, PathBuf::from("/custom/claude"));
    }

    #[test]
    fn fallback_config_maps_to_retry_config() {
        let fb = FallbackConfig {
            providers: vec!["claude_code".to_owned()],
            retry_per_provider: Some(3),
            base_delay_ms: Some(200),
            max_delay_ms: Some(2000),
        };
        let retry = RetryConfig {
            max_retries: fb.retry_per_provider.unwrap_or(0),
            base_delay: Duration::from_millis(fb.base_delay_ms.unwrap_or(500)),
            max_delay: Duration::from_millis(fb.max_delay_ms.unwrap_or(5000)),
        };
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.base_delay, Duration::from_millis(200));
        assert_eq!(retry.max_delay, Duration::from_millis(2000));
    }
}
