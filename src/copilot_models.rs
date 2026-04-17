// ABOUTME: Intelligent model catalog, ranking, and reasoning-effort helpers for the Copilot runner.
// ABOUTME: Replaces hardcoded defaults with version-aware selection and self-heal on ModelUnavailable.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Copilot Model Catalog
//!
//! Ranked catalog of GitHub-Copilot-served models used by the Copilot CLI runner
//! and the Copilot Headless (ACP) runner to pick a sensible default and to
//! self-heal when the requested model has been rotated out of the account's
//! entitlement.
//!
//! The catalog is the single source of truth: adding or retiring a model happens
//! here, not in consumer code. Ranking is `(family, version desc, tier)` so the
//! top entry is always the newest highest-quality model we know about.

/// Model family grouping — higher priorities are preferred defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Anthropic Claude Opus
    ClaudeOpus,
    /// Anthropic Claude Sonnet
    ClaudeSonnet,
    /// Anthropic Claude Haiku
    ClaudeHaiku,
    /// `OpenAI` GPT family
    Gpt,
    /// Google Gemini family
    Gemini,
}

impl Family {
    /// Ranking weight for family preference.
    const fn priority(self) -> u8 {
        match self {
            Self::ClaudeOpus => 10,
            Self::ClaudeSonnet => 8,
            Self::ClaudeHaiku => 6,
            Self::Gpt => 5,
            Self::Gemini => 4,
        }
    }
}

/// Speed/capacity tier within a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Full-capability model
    Full,
    /// Latency-optimized variant (e.g. `*-fast`)
    Fast,
    /// Compact variant (e.g. `*-mini`)
    Mini,
}

impl Tier {
    const fn priority(self) -> u8 {
        match self {
            Self::Full => 2,
            Self::Fast => 1,
            Self::Mini => 0,
        }
    }
}

/// Reasoning effort level forwarded via `--reasoning-effort` to Copilot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// Low reasoning effort — fastest, least deliberation
    Low,
    /// Medium reasoning effort — balanced default
    Medium,
    /// High reasoning effort — more deliberation
    High,
    /// Extra-high reasoning effort — maximum deliberation
    XHigh,
}

impl ReasoningEffort {
    /// Flag value accepted by `copilot --reasoning-effort`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

/// One entry in the ranked Copilot model catalog.
#[derive(Debug, Clone, Copy)]
pub struct ModelCandidate {
    /// Model identifier as recognized by `copilot --model`.
    pub id: &'static str,
    /// Family grouping used for ranking and default-effort decisions.
    pub family: Family,
    /// Major/minor version pair (e.g. `(4, 7)` for `claude-opus-4.7`).
    pub version: (u8, u8),
    /// Speed/capacity tier within the family.
    pub tier: Tier,
}

/// Canonical ranked catalog of GitHub-Copilot-served models.
///
/// Ordered by `(family priority, version desc, tier priority)` so callers can
/// iterate from the most-preferred default downward.
///
/// Keep this list in sync with `copilot --model` entitlement changes.
pub const CATALOG: &[ModelCandidate] = &[
    // Claude Opus — newest first
    ModelCandidate {
        id: "claude-opus-4.7",
        family: Family::ClaudeOpus,
        version: (4, 7),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "claude-opus-4.7-fast",
        family: Family::ClaudeOpus,
        version: (4, 7),
        tier: Tier::Fast,
    },
    ModelCandidate {
        id: "claude-opus-4.6",
        family: Family::ClaudeOpus,
        version: (4, 6),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "claude-opus-4.6-fast",
        family: Family::ClaudeOpus,
        version: (4, 6),
        tier: Tier::Fast,
    },
    ModelCandidate {
        id: "claude-opus-4.5",
        family: Family::ClaudeOpus,
        version: (4, 5),
        tier: Tier::Full,
    },
    // Claude Sonnet
    ModelCandidate {
        id: "claude-sonnet-4.6",
        family: Family::ClaudeSonnet,
        version: (4, 6),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "claude-sonnet-4.5",
        family: Family::ClaudeSonnet,
        version: (4, 5),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "claude-sonnet-4",
        family: Family::ClaudeSonnet,
        version: (4, 0),
        tier: Tier::Full,
    },
    // Claude Haiku
    ModelCandidate {
        id: "claude-haiku-4.5",
        family: Family::ClaudeHaiku,
        version: (4, 5),
        tier: Tier::Full,
    },
    // GPT — newest first
    ModelCandidate {
        id: "gpt-5.4",
        family: Family::Gpt,
        version: (5, 4),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.3-codex",
        family: Family::Gpt,
        version: (5, 3),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.2-codex",
        family: Family::Gpt,
        version: (5, 2),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.2",
        family: Family::Gpt,
        version: (5, 2),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.1-codex-max",
        family: Family::Gpt,
        version: (5, 1),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.1-codex",
        family: Family::Gpt,
        version: (5, 1),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.1",
        family: Family::Gpt,
        version: (5, 1),
        tier: Tier::Full,
    },
    ModelCandidate {
        id: "gpt-5.1-codex-mini",
        family: Family::Gpt,
        version: (5, 1),
        tier: Tier::Mini,
    },
    ModelCandidate {
        id: "gpt-5-mini",
        family: Family::Gpt,
        version: (5, 0),
        tier: Tier::Mini,
    },
    ModelCandidate {
        id: "gpt-4.1",
        family: Family::Gpt,
        version: (4, 1),
        tier: Tier::Full,
    },
    // Gemini
    ModelCandidate {
        id: "gemini-3-pro-preview",
        family: Family::Gemini,
        version: (3, 0),
        tier: Tier::Full,
    },
];

/// Sort a slice of candidates by `(family priority, version desc, tier priority)`.
///
/// Exported so callers that build subset catalogs (e.g. after entitlement
/// filtering) can reuse the exact ranking rule.
pub fn rank(candidates: &mut [ModelCandidate]) {
    candidates.sort_by(|a, b| {
        b.family
            .priority()
            .cmp(&a.family.priority())
            .then_with(|| b.version.cmp(&a.version))
            .then_with(|| b.tier.priority().cmp(&a.tier.priority()))
    });
}

/// Identifier of the top-ranked candidate in [`CATALOG`].
///
/// Returns `"claude-opus-4.7"` today — the catalog is compiled in, so the
/// value is statically known and the indexing below cannot panic.
#[must_use]
pub fn preferred_default() -> &'static str {
    CATALOG[0].id
}

/// Every model id in [`CATALOG`] as owned `String`s.
///
/// Shape matches `LlmProvider::available_models()` which returns `&[String]`.
#[must_use]
pub fn catalog_ids() -> Vec<String> {
    CATALOG.iter().map(|c| c.id.to_owned()).collect()
}

/// Next candidate id after `current` in catalog order.
///
/// Used by the runner's self-heal loop after a [`ModelUnavailable`] error.
/// Returns `None` when `current` is the last entry or is unknown.
///
/// [`ModelUnavailable`]: crate::types::ErrorKind::ModelUnavailable
#[must_use]
pub fn next_preferred(current: &str) -> Option<&'static str> {
    let idx = CATALOG.iter().position(|c| c.id == current)?;
    CATALOG.get(idx + 1).map(|c| c.id)
}

/// Lookup a candidate by exact id.
#[must_use]
pub fn find(id: &str) -> Option<&'static ModelCandidate> {
    CATALOG.iter().find(|c| c.id == id)
}

/// Heuristic default reasoning effort for a model id.
///
/// Returns `Medium` for Opus/Sonnet/GPT-5.x (models that benefit from deliberation)
/// and `None` for Haiku/Gemini/older GPT (flag is ignored or unsupported).
#[must_use]
pub fn default_effort_for(model_id: &str) -> Option<ReasoningEffort> {
    let candidate = find(model_id)?;
    match candidate.family {
        Family::ClaudeOpus | Family::ClaudeSonnet => Some(ReasoningEffort::Medium),
        Family::Gpt if candidate.version.0 >= 5 => Some(ReasoningEffort::Medium),
        _ => None,
    }
}

/// Detect Copilot's "model not available" failure in captured stderr.
///
/// The copilot CLI emits a well-known line when `--model X` is rejected:
///
/// ```text
/// Error: Model "claude-opus-4.6-fast" from --model flag is not available.
/// ```
///
/// Returns `Some(requested_model)` on match, `None` otherwise. The returned
/// string is owned to decouple the caller from the stderr buffer's lifetime.
#[must_use]
pub fn classify_model_error(stderr: &str) -> Option<String> {
    let needle = "from --model flag is not available";
    let line = stderr.lines().find(|l| l.contains(needle))?;
    let after_first_quote = line.split_once('"')?.1;
    let (model, _) = after_first_quote.split_once('"')?;
    Some(model.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_pre_ranked() {
        let mut cloned: Vec<ModelCandidate> = CATALOG.to_vec();
        rank(&mut cloned);
        let ordered_ids: Vec<&str> = cloned.iter().map(|c| c.id).collect();
        let catalog_ids: Vec<&str> = CATALOG.iter().map(|c| c.id).collect();
        assert_eq!(
            ordered_ids, catalog_ids,
            "CATALOG entries should already be sorted by rank()"
        );
    }

    #[test]
    fn preferred_default_is_newest_opus() {
        assert_eq!(preferred_default(), "claude-opus-4.7");
    }

    #[test]
    fn next_preferred_walks_catalog() {
        assert_eq!(
            next_preferred("claude-opus-4.7"),
            Some("claude-opus-4.7-fast")
        );
        assert_eq!(
            next_preferred("claude-opus-4.6-fast"),
            Some("claude-opus-4.5")
        );
    }

    #[test]
    fn next_preferred_unknown_id_returns_none() {
        assert_eq!(next_preferred("model-that-does-not-exist"), None);
    }

    #[test]
    fn next_preferred_last_entry_returns_none() {
        let last = CATALOG[CATALOG.len() - 1].id;
        assert_eq!(next_preferred(last), None);
    }

    #[test]
    fn classify_model_error_parses_copilot_message() {
        let stderr = "Error: Model \"claude-opus-4.6-fast\" from --model flag is not available.";
        assert_eq!(
            classify_model_error(stderr),
            Some("claude-opus-4.6-fast".to_owned())
        );
    }

    #[test]
    fn classify_model_error_ignores_unrelated_stderr() {
        assert_eq!(classify_model_error(""), None);
        assert_eq!(classify_model_error("Error: network unreachable"), None);
        assert_eq!(classify_model_error("Model \"x\" is invalid"), None);
    }

    #[test]
    fn default_effort_picks_medium_for_opus_sonnet_gpt5() {
        assert_eq!(
            default_effort_for("claude-opus-4.7"),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(
            default_effort_for("claude-sonnet-4.6"),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(default_effort_for("gpt-5.4"), Some(ReasoningEffort::Medium));
    }

    #[test]
    fn default_effort_is_none_for_haiku_and_gemini() {
        assert_eq!(default_effort_for("claude-haiku-4.5"), None);
        assert_eq!(default_effort_for("gemini-3-pro-preview"), None);
        assert_eq!(default_effort_for("gpt-4.1"), None);
    }

    #[test]
    fn default_effort_is_none_for_unknown_id() {
        assert_eq!(default_effort_for("unknown-model"), None);
    }

    #[test]
    fn catalog_ids_matches_catalog_len() {
        assert_eq!(catalog_ids().len(), CATALOG.len());
    }

    #[test]
    fn reasoning_effort_as_str() {
        assert_eq!(ReasoningEffort::Low.as_str(), "low");
        assert_eq!(ReasoningEffort::Medium.as_str(), "medium");
        assert_eq!(ReasoningEffort::High.as_str(), "high");
        assert_eq!(ReasoningEffort::XHigh.as_str(), "xhigh");
    }
}
