// ABOUTME: Environment sandboxing and tool policy for CLI subprocesses
// ABOUTME: Clears environment, whitelists keys, and sets working directory
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::io;
use std::path::{Path, PathBuf};

use tokio::process::Command;
use tracing::debug;

use crate::config::default_allowed_env_keys;

/// Policy controlling the subprocess execution environment
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Environment variable keys to pass through from the host
    pub allowed_env_keys: Vec<String>,
    /// Working directory for the subprocess
    pub working_directory: PathBuf,
}

impl SandboxPolicy {
    /// Create a policy with defaults (standard env keys, current directory)
    #[must_use]
    pub fn new(working_directory: PathBuf) -> Self {
        Self {
            allowed_env_keys: default_allowed_env_keys(),
            working_directory,
        }
    }

    /// Create a policy with custom allowed environment keys
    #[must_use]
    pub fn with_env_keys(mut self, keys: Vec<String>) -> Self {
        self.allowed_env_keys = keys;
        self
    }
}

/// Apply sandbox policy to a command before execution
///
/// This clears the subprocess environment, then re-injects only the
/// allowed keys from the host environment. The working directory is
/// also set according to the policy.
pub fn apply_sandbox(cmd: &mut Command, policy: &SandboxPolicy) {
    cmd.env_clear();

    let mut resolved = Vec::new();
    let mut missing = Vec::new();
    for key in &policy.allowed_env_keys {
        if let Ok(value) = env::var(key) {
            cmd.env(key, &value);
            resolved.push(key.as_str());
        } else {
            missing.push(key.as_str());
        }
    }

    cmd.current_dir(&policy.working_directory);

    debug!(
        cwd = %policy.working_directory.display(),
        resolved_count = resolved.len(),
        missing_count = missing.len(),
        ?missing,
        "Applied sandbox policy"
    );
}

/// Build a `SandboxPolicy` from a working directory path, falling back to
/// the current directory if the provided path does not exist.
///
/// # Errors
///
/// Returns an `io::Error` if neither the provided path nor the
/// current directory can be resolved.
pub fn build_policy(
    working_dir: Option<&Path>,
    allowed_env_keys: &[String],
) -> io::Result<SandboxPolicy> {
    let dir = match working_dir {
        Some(p) if p.exists() => p.to_path_buf(),
        _ => env::current_dir()?,
    };
    Ok(SandboxPolicy::new(dir).with_env_keys(allowed_env_keys.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_policy_defaults() {
        let policy = SandboxPolicy::new(PathBuf::from("/tmp"));
        assert_eq!(policy.working_directory, PathBuf::from("/tmp"));
        assert!(!policy.allowed_env_keys.is_empty());
    }

    #[test]
    fn test_sandbox_policy_with_custom_keys() {
        let policy =
            SandboxPolicy::new(PathBuf::from("/tmp")).with_env_keys(vec!["CUSTOM_KEY".to_owned()]);
        assert_eq!(policy.allowed_env_keys, vec!["CUSTOM_KEY"]);
    }

    #[test]
    fn test_build_policy_fallback_to_cwd() {
        let keys = vec!["HOME".to_owned()];
        let policy = build_policy(None, &keys).unwrap();
        // With None, should fall back to current directory
        assert_eq!(policy.working_directory, env::current_dir().unwrap());
    }

    #[test]
    fn test_build_policy_nonexistent_dir_falls_back() {
        let keys = vec!["HOME".to_owned()];
        let policy = build_policy(Some(Path::new("/nonexistent/path/xyz123")), &keys).unwrap();
        // Nonexistent path should fall back to cwd
        assert_eq!(policy.working_directory, env::current_dir().unwrap());
    }

    #[test]
    fn test_build_policy_existing_dir() {
        let keys = vec!["HOME".to_owned()];
        let dir = env::temp_dir();
        let policy = build_policy(Some(&dir), &keys).unwrap();
        assert_eq!(policy.working_directory, dir);
        assert_eq!(policy.allowed_env_keys, vec!["HOME"]);
    }
}
