// ABOUTME: Applies a harness's skills, MCP servers and disabled tools to the runner a task runs on
// ABOUTME: A setting the chosen runner has no way to apply refuses the task instead of being dropped
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! What a harness's configuration adds to one task's runner invocation.
//!
//! A configured harness carries three settings beyond its model: skill
//! bundles, MCP servers, and tools withheld from the model. Each reaches the
//! underlying CLI through whatever that CLI actually takes, and each runner
//! family is listed in [`mechanism`] with exactly what that is:
//!
//! | Base | skills | mcpServers | disabledTools |
//! |---|---|---|---|
//! | `claude-code` | `--plugin-dir` | `--mcp-config` | `--disallowed-tools` |
//! | `copilot` | `--plugin-dir` | `--additional-mcp-config` | `--excluded-tools` |
//! | `copilot_headless` | — | ACP `session/new` | — |
//! | every other base | — | — | — |
//!
//! A setting a base has no way to apply refuses the task with
//! `unsupported_harness_setting`, naming the setting and the base. Running the
//! task without it would hand the client a result produced under a
//! configuration it did not ask for — a withheld tool offered anyway, a skill
//! never loaded — and nothing in the response could tell it so.
//!
//! Only what is switched on counts: a skill or MCP server with `enabled: false`
//! is neither applied nor refused.

use std::fs;
use std::path::{Path, PathBuf};

use embacle::config::CliRunnerType;
use embacle::types::{McpServerConfig, McpTransport};
use serde_json::json;

use super::error::{ErrorType, UhpFailure};
use super::files;
use super::harnesses::{Harness, McpServer, Skill};

/// The name the staged skill bundles are loaded under.
///
/// Both CLIs that load skills namespace a plugin's skills by the plugin's
/// name, so a bundle named `review` reaches the model as `harness:review`.
pub const PLUGIN_NAME: &str = "harness";

/// Where a plugin directory keeps its manifest, read by Claude Code and by the
/// Copilot CLI alike.
const MANIFEST_PATH: [&str; 2] = [".claude-plugin", "plugin.json"];

/// The folder inside a plugin directory that holds one folder per skill.
const SKILLS_DIR: &str = "skills";

/// One of the three harness settings a task run applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    /// Skill bundles.
    Skills,
    /// MCP servers the harness may reach.
    McpServers,
    /// Tools withheld from the model.
    DisabledTools,
}

impl Setting {
    /// The setting's name on the wire, which is how a refusal names it.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Skills => "skills",
            Self::McpServers => "mcpServers",
            Self::DisabledTools => "disabledTools",
        }
    }
}

/// How one runner family takes each setting, or `None` where it has no way to.
struct Mechanism {
    /// The flag that loads a plugin directory holding the skill bundles.
    skills: Option<&'static str>,
    /// Whether MCP servers handed over on the request reach the model.
    mcp_servers: bool,
    /// The flag that removes a tool from what the model is offered.
    disabled_tools: Option<&'static str>,
}

/// What `provider` can apply.
///
/// Claude Code and the Copilot CLI both load a plugin directory for one run
/// (`--plugin-dir`) and read its `skills/<name>/SKILL.md` bundles, and both
/// take MCP servers on their command line, rendered from the request by their
/// runners. Claude Code's `--disallowed-tools` and the Copilot CLI's
/// `--excluded-tools` each remove a tool from the set the model is offered,
/// rather than only denying its use once called.
///
/// Copilot over ACP builds its runner from the environment, not from a
/// per-task configuration, so no flag reaches it; MCP servers do, through the
/// `session/new` request its runner sends. The remaining runners take none of
/// the three: their CLIs read MCP servers, skills and tool policy only from
/// their own configuration files, which embacle does not write.
const fn mechanism(provider: CliRunnerType) -> Mechanism {
    match provider {
        CliRunnerType::ClaudeCode => Mechanism {
            skills: Some("--plugin-dir"),
            mcp_servers: true,
            disabled_tools: Some("--disallowed-tools"),
        },
        CliRunnerType::Copilot => Mechanism {
            skills: Some("--plugin-dir"),
            mcp_servers: true,
            disabled_tools: Some("--excluded-tools"),
        },
        CliRunnerType::CopilotHeadless => Mechanism {
            skills: None,
            mcp_servers: true,
            disabled_tools: None,
        },
        _ => Mechanism {
            skills: None,
            mcp_servers: false,
            disabled_tools: None,
        },
    }
}

/// Skill bundles written out for one task, removed from disk when dropped.
///
/// Held by the task until its runner is done with them: the CLI reads the
/// bundles while it runs, not when it starts.
#[derive(Debug)]
pub struct StagedSkills {
    dir: PathBuf,
}

impl StagedSkills {
    /// The plugin directory the runner is pointed at.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Write `skills` as one plugin directory at `dir`.
    ///
    /// The guard exists before the first write, so a bundle that fails half
    /// way leaves nothing behind.
    fn write(dir: PathBuf, skills: &[&Skill]) -> Result<Self, UhpFailure> {
        let staged = Self { dir };
        let manifest: PathBuf = MANIFEST_PATH.iter().collect();
        write_file(
            &staged.dir.join(manifest),
            json!({ "name": PLUGIN_NAME }).to_string().as_bytes(),
        )?;

        for skill in skills {
            if !skill.has_plain_name() {
                return Err(invalid_skill(
                    &skill.name,
                    "its name is not one path segment",
                ));
            }
            let root = staged.dir.join(SKILLS_DIR).join(&skill.name);
            for file in &skill.files {
                let relative = file
                    .relative_path()
                    .ok_or_else(|| invalid_skill(&skill.name, "a member path leaves the bundle"))?;
                let bytes = file.bytes().map_err(|_| {
                    invalid_skill(&skill.name, "a member's content_b64 is not base64")
                })?;
                write_file(&root.join(relative), &bytes)?;
            }
        }
        Ok(staged)
    }
}

impl Drop for StagedSkills {
    fn drop(&mut self) {
        // Nothing reads the bundle after its task, and a folder that cannot
        // be removed has no one left to report to.
        fs::remove_dir_all(&self.dir).ok();
    }
}

/// Write one staged file, creating the folders above it.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), UhpFailure> {
    path.parent()
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|()| fs::write(path, bytes))
        .map_err(|_| staging_failed())
}

/// The bundles could not be put where the runner reads them.
fn staging_failed() -> UhpFailure {
    UhpFailure::new(
        ErrorType::ServerError,
        "skill_staging_failed",
        "the harness's skills could not be prepared for this task",
    )
}

/// A skill bundle that cannot be written as it stands.
fn invalid_skill(name: &str, why: &str) -> UhpFailure {
    UhpFailure::new(
        ErrorType::InvalidRequestError,
        "invalid_input",
        format!("skill '{name}' cannot be staged for this task: {why}"),
    )
    .with_detail(json!({ "setting": Setting::Skills.wire_name(), "skill": name }))
}

/// Everything a harness's configuration adds to one task's run.
#[derive(Debug, Default)]
pub struct Applied {
    /// Arguments appended to the runner's command line.
    pub args: Vec<String>,
    /// MCP servers handed to the runner on the request.
    pub mcp_servers: Vec<McpServerConfig>,
    /// The staged skill bundles, kept on disk while the task holds this.
    pub skills: Option<StagedSkills>,
}

/// Render one enabled MCP server the way the runners take it.
///
/// The harness object names a server by URL alone, so only the transports
/// that reach a URL apply: streamable HTTP, which is also what an absent
/// transport means, and SSE.
fn mcp_config(server: &McpServer) -> Result<McpServerConfig, UhpFailure> {
    let refuse = |why: String| {
        UhpFailure::new(ErrorType::InvalidRequestError, "invalid_input", why).with_detail(
            json!({ "setting": Setting::McpServers.wire_name(), "server": server.name }),
        )
    };
    let url = server.url.clone().ok_or_else(|| {
        refuse(format!(
            "MCP server '{}' is enabled but names no url",
            server.name
        ))
    })?;
    let headers = Vec::new();
    let transport = match server.transport.as_deref() {
        None | Some("http" | "streamable-http" | "streamable_http") => {
            McpTransport::Http { url, headers }
        }
        Some("sse") => McpTransport::Sse { url, headers },
        Some(other) => {
            return Err(refuse(format!(
                "MCP server '{}' uses transport '{other}'; a harness reaches MCP servers over \
                 http or sse",
                server.name
            )))
        }
    };
    Ok(McpServerConfig {
        name: server.name.clone(),
        transport,
    })
}

/// Apply `harness`'s settings to a run of `provider`, the task `task_id`.
///
/// # Errors
///
/// Answers `400 unsupported_harness_setting` naming every switched-on
/// setting `provider` has no way to apply, `400 invalid_input` for an MCP
/// server or skill bundle that cannot be handed over as it stands, and
/// `500 skill_staging_failed` when the bundles cannot be written.
pub fn apply(
    provider: CliRunnerType,
    harness: &Harness,
    task_id: &str,
) -> Result<Applied, UhpFailure> {
    let mechanism = mechanism(provider);
    let skills: Vec<&Skill> = harness.skills.iter().filter(|s| s.enabled).collect();
    let servers: Vec<&McpServer> = harness.mcp_servers.iter().filter(|s| s.enabled).collect();

    let unsupported: Vec<&str> = [
        (
            Setting::Skills,
            !skills.is_empty() && mechanism.skills.is_none(),
        ),
        (
            Setting::McpServers,
            !servers.is_empty() && !mechanism.mcp_servers,
        ),
        (
            Setting::DisabledTools,
            !harness.disabled_tools.is_empty() && mechanism.disabled_tools.is_none(),
        ),
    ]
    .into_iter()
    .filter_map(|(setting, refused)| refused.then_some(setting.wire_name()))
    .collect();
    if !unsupported.is_empty() {
        return Err(UhpFailure::new(
            ErrorType::InvalidRequestError,
            "unsupported_harness_setting",
            format!(
                "harness '{}' sets {}, which its base '{}' has no way to apply",
                harness.name,
                unsupported.join(", "),
                harness.base
            ),
        )
        .with_detail(json!({
            "harness_id": harness.id,
            "base": harness.base,
            "unsupported": unsupported,
        })));
    }

    let mut applied = Applied {
        mcp_servers: servers
            .into_iter()
            .map(mcp_config)
            .collect::<Result<_, _>>()?,
        ..Applied::default()
    };

    if let Some(flag) = mechanism.disabled_tools {
        if !harness.disabled_tools.is_empty() {
            applied.args.push(flag.to_owned());
            applied.args.extend(harness.disabled_tools.iter().cloned());
        }
    }

    if let Some(flag) = mechanism.skills {
        if !skills.is_empty() {
            let staged = StagedSkills::write(files::staging_dir(task_id), &skills)?;
            // An argument is a string; a staging path that is not one would
            // reach the CLI mangled and load nothing, so it stops here.
            let dir = staged.dir().to_str().ok_or_else(staging_failed)?.to_owned();
            applied.args.push(flag.to_owned());
            applied.args.push(dir);
            applied.skills = Some(staged);
        }
    }

    Ok(applied)
}
