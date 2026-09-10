// ABOUTME: Harness management — create, update and delete configured harnesses over the API
// ABOUTME: A base this server cannot run is refused at config time, not at task time
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Configured harnesses.
//!
//! The catalogue this server advertises is discovered: one harness per embacle
//! runner whose binary is installed. Management adds a second kind — a *named
//! configuration* over one of those bases, so a client can keep "the reviewer"
//! and "the summariser" as distinct harnesses that happen to run on the same
//! CLI.
//!
//! A base this server cannot run is refused when the harness is **created**,
//! not when a task is finally sent to it. The specification is explicit about
//! why: a server that accepts a base it cannot run fails at task time instead,
//! after the client has committed.

use std::collections::HashMap;
use std::sync::RwLock;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use super::error::{ErrorType, UhpFailure};
use super::harnesses::{Harness, McpServer, Skill};
use super::state::UhpState;

/// What a client sends to create or update a harness.
#[derive(Debug, Clone, Deserialize)]
pub struct HarnessConfig {
    /// Human-readable name.
    pub name: String,
    /// Which installed harness family it runs on.
    pub base: String,
    /// The model to use when a task names none.
    #[serde(default, alias = "defaultModel")]
    pub default_model: Option<String>,
    /// Skill bundles to attach.
    #[serde(default)]
    pub skills: Vec<Skill>,
    /// MCP servers this harness may reach.
    #[serde(default, alias = "mcpServers")]
    pub mcp_servers: Vec<McpServer>,
    /// Tools withheld from the model.
    #[serde(default, alias = "disabledTools")]
    pub disabled_tools: Vec<String>,
}

/// Refuse a skill bundle that carries no manifest.
fn validated_skills(skills: &[Skill]) -> Result<(), UhpFailure> {
    for skill in skills {
        if !skill.has_manifest() {
            return Err(UhpFailure::new(
                ErrorType::InvalidRequestError,
                "invalid_input",
                format!(
                    "skill '{}' carries no {} — it would be stored and then silently ignored at \
                     run time",
                    skill.name,
                    Skill::MANIFEST
                ),
            ));
        }
    }
    Ok(())
}

/// Harnesses a client configured, as opposed to the ones discovery found.
///
/// LIMITATION(registre#410): `ConfiguredHarnesses` is per-process, so a harness created
/// on one instance is invisible to another.
#[derive(Debug, Default)]
pub struct ConfiguredHarnesses(RwLock<HashMap<String, Harness>>);

impl ConfiguredHarnesses {
    /// Every configured harness.
    pub fn all(&self) -> Vec<Harness> {
        self.0
            .read()
            .map(|h| h.values().cloned().collect())
            .unwrap_or_default()
    }

    /// One configured harness by id.
    pub fn get(&self, id: &str) -> Option<Harness> {
        self.0.read().ok()?.get(id).cloned()
    }

    /// Store or replace one.
    pub fn put(&self, harness: Harness) {
        if let Ok(mut all) = self.0.write() {
            all.insert(harness.id.clone(), harness);
        }
    }

    /// Remove one, reporting whether it was there.
    pub fn remove(&self, id: &str) -> bool {
        self.0.write().is_ok_and(|mut all| all.remove(id).is_some())
    }
}

/// Refuse a base this server has no installed harness for.
async fn validated_base(state: &UhpState, base: &str) -> Result<(), UhpFailure> {
    let supported: Vec<String> = super::harnesses::discover(&state.shared)
        .await
        .into_iter()
        .map(|d| d.harness.base)
        .collect();

    if supported.iter().any(|b| b == base) {
        return Ok(());
    }
    Err(UhpFailure::new(
        ErrorType::InvalidRequestError,
        "unsupported_base",
        format!("this server has no installed harness for base '{base}'"),
    )
    // The supported list travels with the refusal so a client can correct
    // itself without a second round trip to discover what it may ask for.
    .with_detail(json!({ "supported": supported })))
}

/// Handle `POST /v1/harnesses`.
///
/// # Errors
///
/// Answers `400 unsupported_base` when no installed harness runs that base.
pub async fn create(
    State(state): State<UhpState>,
    Json(config): Json<HarnessConfig>,
) -> Result<Json<Harness>, UhpFailure> {
    validated_base(&state, &config.base).await?;
    validated_skills(&config.skills)?;

    let harness = Harness {
        id: format!("chrn_{}", Uuid::new_v4().simple()),
        object: "harness",
        name: config.name,
        base: config.base,
        default_model: config.default_model,
        skills: config.skills,
        mcp_servers: config.mcp_servers,
        disabled_tools: config.disabled_tools,
    };
    state.configured.put(harness.clone());
    Ok(Json(harness))
}

/// Handle `PUT /v1/harnesses/{harness_id}`.
///
/// The base is immutable: changing it would silently repoint every session
/// already running on this harness at a different CLI.
///
/// # Errors
///
/// Answers `404 harness_not_found` for an unknown id and `400 unsupported_base`
/// when the body names a base this server cannot run.
pub async fn update(
    State(state): State<UhpState>,
    Path(id): Path<String>,
    Json(config): Json<HarnessConfig>,
) -> Result<Json<Harness>, UhpFailure> {
    let existing = state.configured.get(&id).ok_or_else(|| {
        UhpFailure::not_found("harness_not_found", "no harness with that id is configured")
    })?;
    validated_base(&state, &config.base).await?;
    validated_skills(&config.skills)?;

    // An edit that names no skills keeps the ones already there. A client that
    // PUTs back a rename must not lose the bundle it never mentioned.
    let updated = Harness {
        id: existing.id,
        object: "harness",
        name: config.name,
        base: existing.base,
        default_model: config.default_model.or(existing.default_model),
        skills: if config.skills.is_empty() {
            existing.skills
        } else {
            config.skills
        },
        mcp_servers: if config.mcp_servers.is_empty() {
            existing.mcp_servers
        } else {
            config.mcp_servers
        },
        disabled_tools: if config.disabled_tools.is_empty() {
            existing.disabled_tools
        } else {
            config.disabled_tools
        },
    };
    state.configured.put(updated.clone());
    Ok(Json(updated))
}

/// Handle `DELETE /v1/harnesses/{harness_id}`.
///
/// # Errors
///
/// Answers `404 harness_not_found` when no configured harness has that id.
pub async fn delete(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<StatusCode, UhpFailure> {
    if state.configured.remove(&id) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(UhpFailure::not_found(
            "harness_not_found",
            "no harness with that id is configured",
        ))
    }
}

/// The files of one skill bundle.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SkillFiles {
    /// Every file in the bundle, as stored.
    pub files: Vec<super::harnesses::SkillFile>,
}

/// Handle `GET /v1/harnesses/{harness_id}/skills/{skill_name}/files`.
///
/// The whole folder comes back, not just the manifest.
///
/// # Errors
///
/// Answers `404 harness_not_found` for an unknown harness and
/// `404 skill_not_found` when it carries no bundle by that name.
pub async fn skill_files(
    State(state): State<UhpState>,
    Path((harness_id, skill_name)): Path<(String, String)>,
) -> Result<Json<SkillFiles>, UhpFailure> {
    let harness = state.configured.get(&harness_id).ok_or_else(|| {
        UhpFailure::not_found("harness_not_found", "no harness with that id is configured")
    })?;
    let skill = harness
        .skills
        .into_iter()
        .find(|s| s.name == skill_name)
        .ok_or_else(|| {
            UhpFailure::not_found(
                "skill_not_found",
                "this harness carries no skill by that name",
            )
        })?;
    Ok(Json(SkillFiles { files: skill.files }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harness(id: &str, base: &str) -> Harness {
        Harness {
            id: id.to_owned(),
            object: "harness",
            name: "n".to_owned(),
            base: base.to_owned(),
            default_model: None,
            skills: Vec::new(),
            mcp_servers: Vec::new(),
            disabled_tools: Vec::new(),
        }
    }

    #[test]
    fn a_binary_member_round_trips_by_its_base64() {
        use super::super::harnesses::SkillFile;

        // The conformance bundle carries assets/blob.bin as content_b64, not
        // content. A type that models only text drops it silently, and the
        // folder comes back missing a member the client set.
        let file = SkillFile {
            path: "assets/blob.bin".to_owned(),
            content: None,
            content_b64: Some("AAECAwQF".to_owned()),
            rest: serde_json::Map::new(),
        };
        let json = serde_json::to_string(&file).expect("serialises"); // Safe: test assertion
        assert!(
            json.contains("AAECAwQF"),
            "the binary member's bytes must survive the round trip: {json}"
        );
        assert!(
            !json.contains("\"content\":"),
            "a member with no text content must not gain an empty one"
        );
    }

    #[test]
    fn a_skill_bundle_without_a_manifest_is_refused() {
        use super::super::harnesses::SkillFile;

        let ok = Skill {
            name: "good".to_owned(),
            enabled: true,
            files: vec![SkillFile {
                path: "SKILL.md".to_owned(),
                content: Some("# skill".to_owned()),
                content_b64: None,
                rest: serde_json::Map::new(),
            }],
        };
        assert!(validated_skills(&[ok]).is_ok());

        let bad = Skill {
            name: "no-manifest".to_owned(),
            enabled: true,
            files: vec![SkillFile {
                path: "notes.md".to_owned(),
                content: Some("no manifest here".to_owned()),
                content_b64: None,
                rest: serde_json::Map::new(),
            }],
        };
        assert!(
            validated_skills(&[bad]).is_err(),
            "a bundle with no SKILL.md must be refused at config time, not ignored at run time"
        );
    }

    #[test]
    fn a_configured_harness_round_trips_and_then_is_gone() {
        let store = ConfiguredHarnesses::default();
        store.put(harness("chrn_a", "claude-code"));

        assert_eq!(
            store.get("chrn_a").map(|h| h.base),
            Some("claude-code".to_owned())
        );
        assert_eq!(store.all().len(), 1);

        assert!(
            store.remove("chrn_a"),
            "removing a present harness reports true"
        );
        assert!(
            store.get("chrn_a").is_none(),
            "a deleted harness must not still resolve"
        );
        assert!(
            !store.remove("chrn_a"),
            "removing it twice reports false, so the caller can answer 404"
        );
    }

    #[test]
    fn a_created_id_satisfies_the_schema_pattern() {
        let id = format!("chrn_{}", Uuid::new_v4().simple());
        assert!(id.starts_with("chrn_"));
        assert!(id.len() > "chrn_".len());
    }
}
