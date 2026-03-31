// ABOUTME: Container-based execution backend for CLI runners
// ABOUTME: Spawns CLI commands in ephemeral containers with security isolation
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::types::RunnerError;
use tokio::fs;
use tokio::process::Command;
use tracing::{debug, warn};

use crate::process::{run_cli_command, CliOutput};

/// Environment variable for the container image
const ENV_CONTAINER_IMAGE: &str = "CLI_LLM_CONTAINER_IMAGE";
/// Environment variable for the memory limit
const ENV_CONTAINER_MEMORY: &str = "CLI_LLM_CONTAINER_MEMORY";
/// Environment variable for the PIDs limit
const ENV_CONTAINER_PIDS_LIMIT: &str = "CLI_LLM_CONTAINER_PIDS_LIMIT";
/// Environment variable for the network mode
const ENV_CONTAINER_NETWORK: &str = "CLI_LLM_CONTAINER_NETWORK";

/// Network isolation mode for the container
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkMode {
    /// No network access (`--network none`)
    None,
    /// Share the host network namespace (`--network host`)
    Host,
    /// Use a named Docker network (`--network <name>`)
    Custom(String),
}

impl fmt::Display for NetworkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Host => write!(f, "host"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Bind mount specification for a container volume
#[derive(Debug, Clone)]
pub struct Mount {
    /// Host path to mount from
    pub source: PathBuf,
    /// Container path to mount to
    pub target: PathBuf,
    /// Whether the mount is read-only inside the container
    pub read_only: bool,
}

/// Configuration for container-based CLI execution
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Container image reference (e.g. `ghcr.io/org/cli-llm-runner:latest`)
    pub image: String,
    /// Memory limit for the container (e.g. `"512m"`)
    pub memory_limit: Option<String>,
    /// Maximum number of PIDs allowed inside the container
    pub pids_limit: Option<u32>,
    /// Network isolation mode
    pub network_mode: NetworkMode,
    /// Additional bind mounts passed to the container
    pub extra_mounts: Vec<Mount>,
    /// Environment variables injected into the container
    pub env_vars: Vec<(String, String)>,
}

impl ContainerConfig {
    /// Build a container configuration from environment variables
    ///
    /// Reads:
    /// - `CLI_LLM_CONTAINER_IMAGE` → image (required)
    /// - `CLI_LLM_CONTAINER_MEMORY` → memory limit
    /// - `CLI_LLM_CONTAINER_PIDS_LIMIT` → PIDs limit
    /// - `CLI_LLM_CONTAINER_NETWORK` → network mode (`none`, `host`, or custom name)
    ///
    /// # Errors
    ///
    /// Returns `RunnerError` if `CLI_LLM_CONTAINER_IMAGE` is not set or if
    /// `CLI_LLM_CONTAINER_PIDS_LIMIT` is set but not a valid `u32`.
    pub fn from_env() -> Result<Self, RunnerError> {
        let image = env::var(ENV_CONTAINER_IMAGE).map_err(|_| {
            RunnerError::internal(format!(
                "{ENV_CONTAINER_IMAGE} environment variable is required"
            ))
        })?;

        let memory_limit = env::var(ENV_CONTAINER_MEMORY).ok();

        let pids_limit = match env::var(ENV_CONTAINER_PIDS_LIMIT) {
            Ok(val) => {
                let parsed = val.trim().parse::<u32>().map_err(|e| {
                    RunnerError::internal(format!(
                        "{ENV_CONTAINER_PIDS_LIMIT} is not a valid u32: {e}"
                    ))
                })?;
                Some(parsed)
            }
            Err(_) => Option::None,
        };

        let network_mode =
            env::var(ENV_CONTAINER_NETWORK).map_or(NetworkMode::None, |val| {
                match val.trim().to_lowercase().as_str() {
                    "none" => NetworkMode::None,
                    "host" => NetworkMode::Host,
                    other => NetworkMode::Custom(other.to_owned()),
                }
            });

        Ok(Self {
            image,
            memory_limit,
            pids_limit,
            network_mode,
            extra_mounts: Vec::new(),
            env_vars: Vec::new(),
        })
    }
}

/// Executor that runs CLI commands inside ephemeral Docker containers
///
/// Each invocation creates a fresh `docker run --rm` container with
/// a read-only root filesystem, all capabilities dropped, and
/// `no-new-privileges` enforced. A writable scratch directory is
/// bind-mounted at `/scratch` for temporary files.
#[derive(Debug, Clone)]
pub struct ContainerExecutor {
    /// Container configuration controlling image, limits, and mounts
    config: ContainerConfig,
}

impl ContainerExecutor {
    /// Create a new container executor with the given configuration
    #[must_use]
    pub const fn new(config: ContainerConfig) -> Self {
        Self { config }
    }

    /// Execute a CLI command inside an ephemeral container
    ///
    /// The container is created with security hardening flags:
    /// - `--rm` to auto-remove after exit
    /// - `--read-only` root filesystem
    /// - `--cap-drop=ALL` to drop all Linux capabilities
    /// - `--security-opt=no-new-privileges` to prevent privilege escalation
    /// - A writable `/scratch` tmpfs for temporary files
    ///
    /// If `input` is provided, it is written to a temp file inside the
    /// scratch mount and piped as stdin to the command.
    ///
    /// # Errors
    ///
    /// Returns `RunnerError` if:
    /// - The scratch directory cannot be created or written to
    /// - The docker command fails to spawn or times out
    /// - The container exits with a non-zero code
    pub async fn execute(
        &self,
        binary_name: &str,
        args: &[&str],
        input: Option<&str>,
        timeout: Duration,
        max_output_bytes: usize,
    ) -> Result<CliOutput, RunnerError> {
        let scratch_dir = tempfile::tempdir().map_err(|e| {
            RunnerError::internal(format!("Failed to create scratch directory: {e}"))
        })?;

        let scratch_path = scratch_dir.path();

        // Write stdin content to a file in the scratch directory
        let stdin_file_container_path = if let Some(content) = input {
            let stdin_path = scratch_path.join("stdin.txt");
            fs::write(&stdin_path, content)
                .await
                .map_err(|e| RunnerError::internal(format!("Failed to write stdin file: {e}")))?;
            Some("/scratch/stdin.txt".to_owned())
        } else {
            Option::None
        };

        let docker_args = build_docker_args(
            &self.config,
            scratch_path,
            binary_name,
            args,
            stdin_file_container_path.as_deref(),
        );

        debug!(
            image = %self.config.image,
            binary = binary_name,
            scratch = %scratch_path.display(),
            "Launching container"
        );

        let mut cmd = Command::new("docker");
        cmd.args(&docker_args);

        let result = run_cli_command(&mut cmd, timeout, max_output_bytes).await;

        // Cleanup is handled by TempDir drop, but log if removal fails
        if let Err(e) = scratch_dir.close() {
            warn!("Failed to clean up scratch directory: {e}");
        }

        result
    }
}

/// Build the full `docker run` argument list
fn build_docker_args(
    config: &ContainerConfig,
    scratch_path: &Path,
    binary_name: &str,
    args: &[&str],
    stdin_file_container_path: Option<&str>,
) -> Vec<String> {
    let mut docker_args: Vec<String> = vec![
        "run".to_owned(),
        "--rm".to_owned(),
        "--read-only".to_owned(),
        "--cap-drop=ALL".to_owned(),
        "--security-opt=no-new-privileges".to_owned(),
    ];

    // Memory limit
    if let Some(ref mem) = config.memory_limit {
        docker_args.push(format!("--memory={mem}"));
    }

    // PIDs limit
    if let Some(pids) = config.pids_limit {
        docker_args.push(format!("--pids-limit={pids}"));
    }

    // Network mode
    docker_args.push(format!("--network={}", config.network_mode));

    // Scratch mount (writable)
    docker_args.push("-v".to_owned());
    docker_args.push(format!("{}:/scratch", scratch_path.display()));

    // Extra mounts from configuration
    for mount in &config.extra_mounts {
        docker_args.push("-v".to_owned());
        let ro_suffix = if mount.read_only { ":ro" } else { "" };
        docker_args.push(format!(
            "{}:{}{}",
            mount.source.display(),
            mount.target.display(),
            ro_suffix
        ));
    }

    // Environment variables
    for (key, value) in &config.env_vars {
        docker_args.push("-e".to_owned());
        docker_args.push(format!("{key}={value}"));
    }

    // If stdin content was provided, use shell to redirect it
    if let Some(stdin_path) = stdin_file_container_path {
        docker_args.push("-i".to_owned());
        docker_args.push(config.image.clone());
        docker_args.push("sh".to_owned());
        docker_args.push("-c".to_owned());

        let escaped_args: Vec<String> = args.iter().map(|a| shell_escape(a)).collect();
        let cmd_str = format!("{binary_name} {} < {stdin_path}", escaped_args.join(" "));
        docker_args.push(cmd_str);
    } else {
        docker_args.push(config.image.clone());
        docker_args.push(binary_name.to_owned());
        docker_args.extend(args.iter().map(|a| (*a).to_owned()));
    }

    docker_args
}

/// Escape a shell argument by wrapping in single quotes and escaping
/// embedded single quotes.
fn shell_escape(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_owned();
    }
    // Replace single quotes with '\'' (end quote, escaped quote, start quote)
    format!("'{}'", arg.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> ContainerConfig {
        ContainerConfig {
            image: "ghcr.io/test/runner:latest".to_owned(),
            memory_limit: None,
            pids_limit: None,
            network_mode: NetworkMode::None,
            extra_mounts: Vec::new(),
            env_vars: Vec::new(),
        }
    }

    #[test]
    fn test_docker_args_security_hardening() {
        let config = base_config();
        let args = build_docker_args(
            &config,
            Path::new("/tmp/scratch"),
            "claude",
            &["-p", "hi"],
            None,
        );

        assert!(args.contains(&"--rm".to_owned()));
        assert!(args.contains(&"--read-only".to_owned()));
        assert!(args.contains(&"--cap-drop=ALL".to_owned()));
        assert!(args.contains(&"--security-opt=no-new-privileges".to_owned()));
    }

    #[test]
    fn test_docker_args_memory_and_pids_limits() {
        let mut config = base_config();
        config.memory_limit = Some("512m".to_owned());
        config.pids_limit = Some(100);

        let args = build_docker_args(&config, Path::new("/tmp/scratch"), "claude", &[], None);

        assert!(args.contains(&"--memory=512m".to_owned()));
        assert!(args.contains(&"--pids-limit=100".to_owned()));
    }

    #[test]
    fn test_docker_args_network_modes() {
        let mut config = base_config();

        config.network_mode = NetworkMode::None;
        let args = build_docker_args(&config, Path::new("/tmp/s"), "claude", &[], None);
        assert!(args.contains(&"--network=none".to_owned()));

        config.network_mode = NetworkMode::Host;
        let args = build_docker_args(&config, Path::new("/tmp/s"), "claude", &[], None);
        assert!(args.contains(&"--network=host".to_owned()));

        config.network_mode = NetworkMode::Custom("my-net".to_owned());
        let args = build_docker_args(&config, Path::new("/tmp/s"), "claude", &[], None);
        assert!(args.contains(&"--network=my-net".to_owned()));
    }

    #[test]
    fn test_docker_args_extra_mounts() {
        let mut config = base_config();
        config.extra_mounts = vec![
            Mount {
                source: PathBuf::from("/host/data"),
                target: PathBuf::from("/container/data"),
                read_only: true,
            },
            Mount {
                source: PathBuf::from("/host/work"),
                target: PathBuf::from("/container/work"),
                read_only: false,
            },
        ];

        let args = build_docker_args(&config, Path::new("/tmp/scratch"), "claude", &[], None);

        assert!(args.contains(&"/host/data:/container/data:ro".to_owned()));
        assert!(args.contains(&"/host/work:/container/work".to_owned()));
    }

    #[test]
    fn test_docker_args_env_vars() {
        let mut config = base_config();
        config.env_vars = vec![("API_KEY".to_owned(), "secret123".to_owned())];

        let args = build_docker_args(&config, Path::new("/tmp/scratch"), "claude", &[], None);

        assert!(args.contains(&"-e".to_owned()));
        assert!(args.contains(&"API_KEY=secret123".to_owned()));
    }

    #[test]
    fn test_docker_args_with_stdin_redirect() {
        let config = base_config();
        let args = build_docker_args(
            &config,
            Path::new("/tmp/scratch"),
            "claude",
            &["-p", "hello"],
            Some("/scratch/stdin.txt"),
        );

        assert!(args.contains(&"-i".to_owned()));
        assert!(args.contains(&"sh".to_owned()));
        assert!(args.contains(&"-c".to_owned()));
        // The last arg should be the shell command with stdin redirect
        let last = args.last().unwrap(); // Safe: test assertion
        assert!(last.contains("< /scratch/stdin.txt"));
        assert!(last.contains("claude"));
    }

    #[test]
    fn test_docker_args_without_stdin() {
        let config = base_config();
        let args = build_docker_args(
            &config,
            Path::new("/tmp/scratch"),
            "claude",
            &["-p", "hello"],
            None,
        );

        // Should end with: image, binary, args
        assert!(args.contains(&"ghcr.io/test/runner:latest".to_owned()));
        assert!(args.contains(&"claude".to_owned()));
        assert!(args.contains(&"-p".to_owned()));
        assert!(args.contains(&"hello".to_owned()));
        // Should NOT contain shell redirect markers
        assert!(!args.contains(&"sh".to_owned()));
    }

    #[test]
    fn test_shell_escape_empty() {
        assert_eq!(shell_escape(""), "''");
    }

    #[test]
    fn test_shell_escape_simple() {
        assert_eq!(shell_escape("hello"), "'hello'");
    }

    #[test]
    fn test_shell_escape_single_quotes() {
        assert_eq!(shell_escape("it's"), "'it'\\''s'");
    }

    #[test]
    fn test_network_mode_display() {
        assert_eq!(format!("{}", NetworkMode::None), "none");
        assert_eq!(format!("{}", NetworkMode::Host), "host");
        assert_eq!(
            format!("{}", NetworkMode::Custom("my-net".to_owned())),
            "my-net"
        );
    }
}
