// ABOUTME: Conversation-turn correlation identifier threaded through one user utterance
// ABOUTME: Wraps a Uuid; wire format is a plain UUID string so it is cross-repo compatible
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Conversation Turn Identifier
//!
//! A conversation *turn* is a single user utterance plus the full chain of
//! LLM calls, tool invocations, and the resulting reply. Every call that
//! participates in that chain carries the same [`ConversationTurnId`], which
//! lets downstream observers correlate cost, latency, and tool usage for
//! the turn.
//!
//! The identifier is generated **once** at the inbound boundary (a webhook,
//! a chat endpoint, a CLI entry-point) and propagated — never regenerated —
//! through every subsequent call.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identifier for a single conversation turn.
///
/// The wire format is a standard UUID string, which keeps this type
/// compatible with identically-shaped newtypes defined in sibling crates
/// (`dravr-canot`, `pierre-core`) without forcing a shared dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConversationTurnId(pub Uuid);

impl ConversationTurnId {
    /// Generate a new random turn identifier.
    ///
    /// Only inbound boundaries (webhook handlers, chat entry points) should
    /// call this. Downstream callers must propagate the identifier they
    /// received.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID as a turn identifier.
    #[must_use]
    pub const fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }

    /// Return the underlying UUID.
    #[must_use]
    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for ConversationTurnId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConversationTurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Uuid> for ConversationTurnId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

impl From<ConversationTurnId> for Uuid {
    fn from(value: ConversationTurnId) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::ConversationTurnId;

    #[test]
    fn serde_round_trip_is_plain_uuid_string() -> serde_json::Result<()> {
        let id = ConversationTurnId::new();
        let json = serde_json::to_string(&id)?;
        assert!(json.starts_with('"') && json.ends_with('"'));
        let parsed: ConversationTurnId = serde_json::from_str(&json)?;
        assert_eq!(id, parsed);
        Ok(())
    }

    #[test]
    fn new_ids_are_unique() {
        let a = ConversationTurnId::new();
        let b = ConversationTurnId::new();
        assert_ne!(a, b);
    }
}
