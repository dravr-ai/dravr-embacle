// ABOUTME: GET /v1/harnesses, /v1/harnesses/{id} and the UHP model catalogue endpoints
// ABOUTME: Each installed embacle runner is advertised as one harness; availability is probed
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Harnesses and their models.
//!
//! embacle already runs twelve CLI harnesses behind one trait, which is exactly
//! what UHP calls a **runner**: a server that puts existing harnesses behind the
//! contract and advertises them as its catalog. So a harness here is one
//! embacle provider whose binary is actually installed.
//!
//! `available` is computed, never asserted. The specification is blunt about
//! why: listing a model as available and then failing the task is the worst
//! outcome for a client, because a user has already chosen it.

use std::collections::BTreeMap;
use std::env;
use std::path::{Component, Path as FsPath};

use axum::extract::{Path, State};
use axum::response::IntoResponse;
use axum::Json;
use base64::engine::general_purpose::STANDARD;
use base64::{DecodeError, Engine};
use embacle::config::CliRunnerType;
use embacle::discovery::resolve_binary;
use serde::{Deserialize, Serialize};

use super::error::UhpFailure;
use super::state::UhpState;
use crate::runner::ALL_PROVIDERS;
use crate::state::SharedState;

/// One harness this server can run.
#[derive(Debug, Clone, Serialize)]
pub struct Harness {
    /// Stable id, prefixed `chrn_` as the schema requires.
    pub id: String,
    /// Always `harness`.
    pub object: &'static str,
    /// Human-readable name.
    pub name: String,
    /// Opaque harness family, e.g. `claude-code`. Deliberately not enumerated
    /// by the specification, so a client must treat it as a string.
    pub base: String,
    /// The model used when a task names none.
    #[serde(rename = "defaultModel", skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Skill bundles this harness carries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<Skill>,
    /// MCP servers this harness may reach.
    #[serde(rename = "mcpServers", default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServer>,
    /// Tools withheld from the model on this harness.
    #[serde(
        rename = "disabledTools",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub disabled_tools: Vec<String>,
}

/// One file inside a skill bundle.
///
/// A member may carry its bytes as `content` or, when it is not text, as
/// `content_b64`. Both round-trip, and anything else the client attached comes
/// back too: a bundle is the client's, and dropping a field it set would be
/// indistinguishable from the server having stored it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillFile {
    /// Path relative to the bundle root.
    pub path: String,
    /// Text contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Base64 contents, for a member that is not text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_b64: Option<String>,
    /// Everything else the client attached.
    #[serde(flatten)]
    pub rest: serde_json::Map<String, serde_json::Value>,
}

impl SkillFile {
    /// Where this member sits inside its bundle, when `path` is a plain
    /// relative path.
    ///
    /// The path is the client's and is written to disk when a task runs, so
    /// anything that could climb out of the bundle — a `..`, a root, a drive
    /// prefix, a leading `.` — is refused rather than resolved.
    #[must_use]
    pub fn relative_path(&self) -> Option<&FsPath> {
        let path = FsPath::new(&self.path);
        let mut components = path.components().peekable();
        components.peek()?;
        components
            .all(|c| matches!(c, Component::Normal(_)))
            .then_some(path)
    }

    /// The member's bytes: its text, else its decoded base64, else none at all.
    ///
    /// # Errors
    ///
    /// Returns the decode error when `content_b64` is not standard base64.
    pub fn bytes(&self) -> Result<Vec<u8>, DecodeError> {
        match (&self.content, &self.content_b64) {
            (Some(text), _) => Ok(text.as_bytes().to_vec()),
            (None, Some(encoded)) => STANDARD.decode(encoded),
            (None, None) => Ok(Vec::new()),
        }
    }
}

/// A skill bundle: a manifest plus whatever it references.
///
/// The whole folder round-trips. Storing only `SKILL.md` breaks every skill
/// that carries references, scripts or data — and breaks it at run time, long
/// after the client thought it had configured the harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Bundle name, which is how it is addressed.
    pub name: String,
    /// Whether the harness may use it.
    #[serde(default)]
    pub enabled: bool,
    /// Every file in the bundle.
    #[serde(default)]
    pub files: Vec<SkillFile>,
}

impl Skill {
    /// The manifest every bundle must carry.
    pub const MANIFEST: &'static str = "SKILL.md";

    /// Whether this bundle carries its manifest.
    ///
    /// A bundle without one would be stored and then silently ignored at run
    /// time, which is the hardest kind of failure for a user to diagnose — so
    /// it is refused at config time instead.
    #[must_use]
    pub fn has_manifest(&self) -> bool {
        self.files.iter().any(|f| f.path == Self::MANIFEST)
    }

    /// Whether the bundle's name is one plain path segment.
    ///
    /// The name becomes the folder the bundle is written to when a task runs,
    /// so a name carrying a separator or a `..` would place it elsewhere.
    #[must_use]
    pub fn has_plain_name(&self) -> bool {
        let mut components = FsPath::new(&self.name).components();
        matches!(
            (components.next(), components.next()),
            (Some(Component::Normal(_)), None)
        )
    }
}

/// An MCP server a harness may reach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServer {
    /// Server name.
    pub name: String,
    /// Where it lives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// How to reach it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    /// Whether it is contacted at all.
    ///
    /// Preserved exactly as sent: the difference between true and false decides
    /// whether a third party is contacted, so a client must be able to tell a
    /// disabled entry from an enabled one.
    #[serde(default)]
    pub enabled: bool,
}

/// The `GET /v1/harnesses` body.
#[derive(Debug, Clone, Serialize)]
pub struct HarnessList {
    /// Every harness whose binary is installed.
    pub harnesses: Vec<Harness>,
}

/// One model, with computed availability.
#[derive(Debug, Clone, Serialize)]
pub struct Model {
    /// Model id as the harness names it.
    pub id: String,
    /// Which harness serves it.
    pub backend: String,
    /// Whether this server can serve it right now.
    pub available: bool,
    /// Whether it is the harness's default.
    pub default: bool,
}

/// Models grouped by backend, which is the shape the schema requires — a flat
/// list is what the `OpenAI` surface serves, and the two must not be confused.
#[derive(Debug, Clone, Serialize)]
pub struct BackendModels {
    /// The model used when a task names none.
    pub default: String,
    /// Every model this backend accepts.
    pub models: Vec<Model>,
}

/// The `GET /v1/harnesses/{id}/models` body.
///
/// Deliberately NOT [`ModelCatalog`]: one harness answers a flat `models`
/// array plus its own defaults, while the server-wide catalogue groups by
/// backend. Serving the grouped shape here fails schema validation on a
/// missing `models` property.
#[derive(Debug, Clone, Serialize)]
pub struct HarnessModels {
    /// The harness these models belong to.
    pub harness_id: String,
    /// Its backend family.
    pub backend: String,
    /// The model used when a task names none.
    pub default: String,
    /// Every model this harness accepts.
    pub models: Vec<Model>,
}

/// The `GET /v1/models` body.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCatalog {
    /// Keyed by backend id.
    pub backends: BTreeMap<String, BackendModels>,
}

/// A harness id for a provider name, satisfying the schema's `^chrn_` pattern.
fn harness_id(provider: &str) -> String {
    let slug: String = provider
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("chrn_{slug}")
}

/// Whether this provider's CLI binary resolves on this machine.
fn installed(provider: CliRunnerType) -> bool {
    let env_override = env::var(provider.env_override_key()).ok();
    resolve_binary(provider.binary_name(), env_override.as_deref()).is_ok()
}

/// One installed provider, as everything the routes need to describe or run it.
pub struct Discovered {
    /// The embacle runner type, for constructing the runner.
    pub provider: CliRunnerType,
    /// Its provider name, which is also its UHP backend id.
    pub name: String,
    /// The harness object served over the wire.
    pub harness: Harness,
    /// The models it accepts.
    pub models: Vec<String>,
}

/// Every installed provider, as harnesses.
pub async fn discover(state: &SharedState) -> Vec<Discovered> {
    let mut found = Vec::new();
    for &provider in ALL_PROVIDERS.iter() {
        if !installed(provider) {
            continue;
        }
        let Ok(runner) = state.get_runner(provider).await else {
            continue;
        };
        let name = runner.name().to_owned();
        let models = runner.available_models().to_vec();
        let default_model = runner.default_model().to_owned();
        found.push(Discovered {
            provider,
            harness: Harness {
                id: harness_id(&name),
                object: "harness",
                name: name.clone(),
                base: name.clone(),
                default_model: Some(default_model),
                skills: Vec::new(),
                mcp_servers: Vec::new(),
                disabled_tools: Vec::new(),
            },
            name,
            models,
        });
    }
    found
}

/// The harness `id` names, with the installed base that runs it.
///
/// A configured harness is looked up first and runs on the discovered harness
/// whose base it names, carrying its own id, skills, MCP servers and disabled
/// tools; its default model falls back to that base's when it names none.
/// Anything else is looked up among the discovered harnesses. `None` when
/// the id names neither, or names a configured harness whose base is no longer
/// installed.
pub async fn resolve(state: &UhpState, id: &str) -> Option<Discovered> {
    let found = discover(&state.shared).await;
    let Some(configured) = state.configured.get(id) else {
        return found.into_iter().find(|d| d.harness.id == id);
    };
    let base = found
        .into_iter()
        .find(|d| d.harness.base == configured.base)?;
    Some(Discovered {
        provider: base.provider,
        name: base.name,
        harness: Harness {
            default_model: configured
                .default_model
                .clone()
                .or(base.harness.default_model),
            ..configured
        },
        models: base.models,
    })
}

/// Handle `GET /v1/harnesses`.
///
/// Every discovered harness, then every configured one: a harness a client
/// created is as runnable as one discovery found, so it is listed beside them.
pub async fn list(State(state): State<UhpState>) -> impl IntoResponse {
    let mut harnesses: Vec<Harness> = discover(&state.shared)
        .await
        .into_iter()
        .map(|d| d.harness)
        .collect();
    harnesses.extend(state.configured.all());
    Json(HarnessList { harnesses })
}

/// Handle `GET /v1/harnesses/{harness_id}`.
///
/// # Errors
///
/// Answers `404 harness_not_found` when the id names nothing installed.
pub async fn get(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<Harness>, UhpFailure> {
    if let Some(configured) = state.configured.get(&id) {
        return Ok(Json(configured));
    }
    discover(&state.shared)
        .await
        .into_iter()
        .find(|d| d.harness.id == id)
        .map(|d| Json(d.harness))
        .ok_or_else(|| {
            UhpFailure::not_found("harness_not_found", "no harness with that id is configured")
        })
}

/// Build the catalogue for a set of discovered harnesses.
fn catalogue(found: Vec<Discovered>) -> ModelCatalog {
    let mut backends = BTreeMap::new();
    for d in found {
        let name = d.name;
        let default = d.harness.default_model.unwrap_or_default();
        let models = d
            .models
            .into_iter()
            .map(|id| Model {
                default: id == default,
                id,
                backend: name.clone(),
                // Installed and constructible, which is what this server can
                // actually act on. A model the harness lists but cannot serve
                // would be the failure the spec warns about.
                available: true,
            })
            .collect();
        backends.insert(name, BackendModels { default, models });
    }
    ModelCatalog { backends }
}

/// Handle `GET /v1/models` on the UHP surface.
pub async fn models(State(state): State<UhpState>) -> impl IntoResponse {
    Json(catalogue(discover(&state.shared).await))
}

/// Handle `GET /v1/harnesses/{harness_id}/models`.
///
/// A configured harness answers with its base's models.
///
/// # Errors
///
/// Answers `404 harness_not_found` when the id names nothing installed.
pub async fn harness_models(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<HarnessModels>, UhpFailure> {
    let d = resolve(&state, &id).await.ok_or_else(|| {
        UhpFailure::not_found("harness_not_found", "no harness with that id is configured")
    })?;

    let name = d.name;
    let default = d.harness.default_model.unwrap_or_default();
    Ok(Json(HarnessModels {
        harness_id: d.harness.id,
        backend: name.clone(),
        models: d
            .models
            .into_iter()
            .map(|id| Model {
                default: id == default,
                id,
                backend: name.clone(),
                available: true,
            })
            .collect(),
        default,
    }))
}

/// Fallback for a path under the UHP base that names no route.
pub async fn unknown_route() -> UhpFailure {
    UhpFailure::not_found("not_found", "no such endpoint on this server")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_harness_id_satisfies_the_schema_pattern() {
        // The Harness schema pins `^chrn_`, so an id that does not start with
        // it fails validation for every harness the server advertises.
        assert!(harness_id("claude_code").starts_with("chrn_"));
        assert_eq!(harness_id("claude-code"), "chrn_claude_code");
        assert_eq!(harness_id("copilot"), "chrn_copilot");
    }

    #[test]
    fn the_catalogue_groups_by_backend_and_marks_one_default() {
        let found = vec![Discovered {
            provider: CliRunnerType::Copilot,
            name: "copilot".to_owned(),
            harness: Harness {
                id: harness_id("copilot"),
                object: "harness",
                name: "copilot".to_owned(),
                base: "copilot".to_owned(),
                default_model: Some("gpt-5".to_owned()),
                skills: Vec::new(),
                mcp_servers: Vec::new(),
                disabled_tools: Vec::new(),
            },
            models: vec!["gpt-5".to_owned(), "claude-sonnet".to_owned()],
        }];
        let cat = catalogue(found);

        let backend = cat.backends.get("copilot").expect("backend present"); // Safe: test assertion
        assert_eq!(backend.default, "gpt-5");
        assert_eq!(backend.models.len(), 2);
        assert_eq!(
            backend.models.iter().filter(|m| m.default).count(),
            1,
            "exactly one model is the default"
        );
        assert!(
            backend.models.iter().all(|m| m.available),
            "availability is a boolean on every model, never absent"
        );
    }
}
