// ABOUTME: Session sharing — mint, read back and revoke a public read-only view of a session
// ABOUTME: A share id opens the view and nothing else; it is never a credential for the API
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Sharing a session.
//!
//! Sharing publishes a **view**, not access. Three properties make that true,
//! and each is a separate way to get this wrong:
//!
//! The link alone opens it. A share only its minter can open has not been
//! published to anyone, so the view sits outside the auth layer.
//!
//! The share id is not a credential. Presenting it as a bearer token against
//! the API must fail exactly as any other unknown token does — a link handed to
//! a reader must never become a key to the account that minted it.
//!
//! The view is read-only. It answers `GET` and refuses every method that would
//! change anything, because a reader with a link is not a principal.

use std::collections::HashMap;
use std::sync::RwLock;

use axum::extract::{Path, State};
use axum::Json;
use serde::Serialize;
use uuid::Uuid;

use super::error::UhpFailure;
use super::sessions::Turn;
use super::state::UhpState;

/// Shares this server has minted.
///
/// LIMITATION(registre#410): `Shares` is per-process, so a link minted on one instance
/// does not open on another.
#[derive(Debug, Default)]
pub struct Shares {
    by_share: RwLock<HashMap<String, String>>,
    by_session: RwLock<HashMap<String, String>>,
}

impl Shares {
    /// Mint a share for a session, or return the one it already has.
    ///
    /// Minting twice returns the same link rather than a second one, so a
    /// client retrying after a dropped connection does not scatter live links
    /// it cannot then revoke.
    pub fn mint(&self, session_id: &str) -> String {
        if let Some(existing) = self.for_session(session_id) {
            return existing;
        }
        let share_id = format!("share_{}", Uuid::new_v4().simple());
        if let Ok(mut by_share) = self.by_share.write() {
            by_share.insert(share_id.clone(), session_id.to_owned());
        }
        if let Ok(mut by_session) = self.by_session.write() {
            by_session.insert(session_id.to_owned(), share_id.clone());
        }
        share_id
    }

    /// The share a session already has, if any.
    pub fn for_session(&self, session_id: &str) -> Option<String> {
        self.by_session.read().ok()?.get(session_id).cloned()
    }

    /// The session a share opens.
    pub fn session_of(&self, share_id: &str) -> Option<String> {
        self.by_share.read().ok()?.get(share_id).cloned()
    }

    /// Revoke a session's share, reporting whether there was one.
    ///
    /// Revocation kills the link itself, not one holder's copy of it: the id is
    /// removed, so every link ever minted for this session stops opening.
    pub fn revoke(&self, session_id: &str) -> bool {
        let share_id = self
            .by_session
            .write()
            .map_or(None, |mut by_session| by_session.remove(session_id));
        let Some(share_id) = share_id else {
            return false;
        };
        if let Ok(mut by_share) = self.by_share.write() {
            by_share.remove(&share_id);
        }
        true
    }
}

/// A minted share.
#[derive(Debug, Clone, Serialize)]
pub struct Share {
    /// The share id, which appears in the link.
    pub id: String,
    /// Always `session.share`.
    pub object: &'static str,
    /// The session it opens.
    pub session_id: String,
    /// Where to open it, relative to this server's UHP base.
    pub url: String,
}

/// The public path a share id opens at.
fn share_path(share_id: &str) -> String {
    format!("/share/{share_id}")
}

/// Handle `POST /v1/sessions/{session_id}/share`.
///
/// Takes no body, as the specification describes.
///
/// # Errors
///
/// Answers `404 session_not_found` when this server holds no such session.
pub async fn create(
    State(state): State<UhpState>,
    Path(session_id): Path<String>,
) -> Result<Json<Share>, UhpFailure> {
    if state.store.session_turns(&session_id).is_none() {
        return Err(UhpFailure::not_found(
            "session_not_found",
            "no session with that id exists",
        ));
    }
    let id = state.shares.mint(&session_id);
    Ok(Json(Share {
        url: share_path(&id),
        id,
        object: "session.share",
        session_id,
    }))
}

/// Handle `GET /v1/sessions/{session_id}/share`.
///
/// # Errors
///
/// Answers `404 share_not_found` when the session has no live share.
pub async fn get(
    State(state): State<UhpState>,
    Path(session_id): Path<String>,
) -> Result<Json<Share>, UhpFailure> {
    let id = state.shares.for_session(&session_id).ok_or_else(|| {
        UhpFailure::not_found("share_not_found", "this session has no shared view")
    })?;
    Ok(Json(Share {
        url: share_path(&id),
        id,
        object: "session.share",
        session_id,
    }))
}

/// Handle `DELETE /v1/sessions/{session_id}/share`.
///
/// # Errors
///
/// Answers `404 share_not_found` when the session has no live share.
pub async fn revoke(
    State(state): State<UhpState>,
    Path(session_id): Path<String>,
) -> Result<Json<super::sessions::Deleted>, UhpFailure> {
    if state.shares.revoke(&session_id) {
        Ok(Json(super::sessions::Deleted {
            id: session_id,
            object: "session.share.deleted",
            deleted: true,
        }))
    } else {
        Err(UhpFailure::not_found(
            "share_not_found",
            "this session has no shared view",
        ))
    }
}

/// What a reader sees when they open a link.
///
/// Turns only. No harness configuration, no MCP server URLs, no bearer token,
/// nothing about the principal who minted it — a reader with a link is not a
/// principal, and the view must not become a way to learn what one holds.
#[derive(Debug, Clone, Serialize)]
pub struct SharedView {
    /// Always `session.shared_view`.
    pub object: &'static str,
    /// The session being read.
    pub session_id: String,
    /// Its turns, oldest first.
    pub turns: Vec<Turn>,
    /// Stated so a reader knows the view will refuse anything else.
    pub read_only: bool,
}

/// Handle `GET /share/{share_id}` — served without a credential.
///
/// # Errors
///
/// Answers `404 share_not_found` for a revoked or unknown link, which is the
/// same answer either way: a revoked link must not be distinguishable from one
/// that never existed.
pub async fn view(
    State(state): State<UhpState>,
    Path(share_id): Path<String>,
) -> Result<Json<SharedView>, UhpFailure> {
    let session_id = state.shares.session_of(&share_id).ok_or_else(|| {
        UhpFailure::not_found("share_not_found", "this link does not open anything")
    })?;
    let turns = state.store.session_turns(&session_id).ok_or_else(|| {
        UhpFailure::not_found("share_not_found", "this link does not open anything")
    })?;
    Ok(Json(SharedView {
        object: "session.shared_view",
        session_id,
        turns: turns.iter().map(Turn::from).collect(),
        read_only: true,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minting_twice_returns_one_link() {
        // Otherwise a client retrying after a dropped connection scatters live
        // links it can no longer revoke.
        let shares = Shares::default();
        let first = shares.mint("sess_1");
        let second = shares.mint("sess_1");
        assert_eq!(first, second);
        assert!(first.starts_with("share_"));
    }

    #[test]
    fn revocation_kills_the_link_itself() {
        let shares = Shares::default();
        let id = shares.mint("sess_1");
        assert_eq!(shares.session_of(&id).as_deref(), Some("sess_1"));

        assert!(shares.revoke("sess_1"));
        assert!(
            shares.session_of(&id).is_none(),
            "a revoked link must stop opening for every holder, not just the minter"
        );
        assert!(shares.for_session("sess_1").is_none());
        assert!(!shares.revoke("sess_1"), "revoking twice reports false");
    }

    #[test]
    fn a_share_id_is_not_a_session_id() {
        // R-03 presents the share id as a bearer token. It must be worthless
        // there, and it is: it names nothing the API looks up.
        let shares = Shares::default();
        let id = shares.mint("sess_1");
        assert_ne!(id, "sess_1");
        assert!(id.starts_with("share_"), "distinct namespace from sess_");
    }
}
