// ABOUTME: Session listing, inspection, turn history and deletion over the UHP contract
// ABOUTME: Pagination reports its end explicitly rather than leaving a client to infer it
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Sessions.
//!
//! A session is the thread a task continues. The store already records which
//! responses belong to which session, so these endpoints read it back rather
//! than keeping a second account of the same thing.
//!
//! The listing reports the end of pagination **explicitly**, with `has_more`
//! and a `next_cursor`. Leaving a client to infer the end from a short page is
//! a heuristic that is wrong exactly when a page comes back full.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::{Deserialize, Serialize};

use super::error::UhpFailure;
use super::state::UhpState;
use super::tasks::Response;

/// How many sessions a listing returns when the client names no limit.
const DEFAULT_LIMIT: usize = 20;

/// Paging controls on a session listing.
#[derive(Debug, Clone, Deserialize)]
pub struct Paging {
    /// Maximum sessions to return.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Opaque marker naming where the previous page stopped.
    #[serde(default)]
    pub cursor: Option<String>,
}

/// One session, as the listing describes it.
#[derive(Debug, Clone, Serialize)]
pub struct Session {
    /// `sess_`-prefixed id.
    pub id: String,
    /// Always `session`.
    pub object: &'static str,
    /// How many turns it holds.
    pub turn_count: usize,
    /// When its first turn was created, in Unix seconds.
    pub created_at: u64,
}

/// A page of sessions.
#[derive(Debug, Clone, Serialize)]
pub struct SessionList {
    /// This page.
    pub sessions: Vec<Session>,
    /// Whether another page follows.
    pub has_more: bool,
    /// Where to resume, or `null` at the end.
    pub next_cursor: Option<String>,
}

/// One turn of a session.
#[derive(Debug, Clone, Serialize)]
pub struct Turn {
    /// The response id of this turn.
    pub id: String,
    /// Its terminal status.
    pub status: String,
    /// When it ran, in Unix seconds.
    pub created_at: u64,
    /// The model that answered.
    pub model: String,
}

/// A session's turns, oldest first.
#[derive(Debug, Clone, Serialize)]
pub struct TurnList {
    /// The session these belong to.
    pub session_id: String,
    /// Its turns.
    pub turns: Vec<Turn>,
}

impl From<&Response> for Turn {
    fn from(r: &Response) -> Self {
        Self {
            id: r.id.clone(),
            status: r.status.to_owned(),
            created_at: r.created_at,
            model: r.model.clone(),
        }
    }
}

/// Build one session summary from its turns.
fn summarise(id: String, turns: &[Response]) -> Session {
    Session {
        object: "session",
        turn_count: turns.len(),
        created_at: turns.first().map_or(0, |t| t.created_at),
        id,
    }
}

/// Handle `GET /v1/sessions`.
pub async fn list(
    State(state): State<UhpState>,
    Query(paging): Query<Paging>,
) -> Json<SessionList> {
    let limit = paging.limit.unwrap_or(DEFAULT_LIMIT).max(1);

    let mut ids = state.store.session_ids();
    ids.sort();

    // The cursor is the last id of the previous page, so resuming is a scan to
    // just past it. Opaque to the client, which is all the contract requires.
    let start = paging
        .cursor
        .as_ref()
        .map_or(0, |c| ids.iter().position(|i| i == c).map_or(0, |p| p + 1));

    let page: Vec<Session> = ids
        .iter()
        .skip(start)
        .take(limit)
        .map(|id| {
            let turns = state.store.session_turns(id).unwrap_or_default();
            summarise(id.clone(), &turns)
        })
        .collect();

    let has_more = start + page.len() < ids.len();
    Json(SessionList {
        next_cursor: has_more
            .then(|| page.last().map(|s| s.id.clone()))
            .flatten(),
        has_more,
        sessions: page,
    })
}

/// Handle `GET /v1/sessions/{session_id}`.
///
/// # Errors
///
/// Answers `404 session_not_found` when this server holds no such session.
pub async fn get(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<Session>, UhpFailure> {
    let turns = state.store.session_turns(&id).ok_or_else(|| {
        UhpFailure::not_found("session_not_found", "no session with that id exists")
    })?;
    Ok(Json(summarise(id, &turns)))
}

/// Handle `GET /v1/sessions/{session_id}/turns`.
///
/// # Errors
///
/// Answers `404 session_not_found` when this server holds no such session.
pub async fn turns(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<TurnList>, UhpFailure> {
    let responses = state.store.session_turns(&id).ok_or_else(|| {
        UhpFailure::not_found("session_not_found", "no session with that id exists")
    })?;
    Ok(Json(TurnList {
        session_id: id,
        turns: responses.iter().map(Turn::from).collect(),
    }))
}

/// Handle `DELETE /v1/sessions/{session_id}`, and its legacy alias.
///
/// `DELETE /v1/traces/{session_id}` is the older path for the same operation,
/// which the specification says a server MAY keep serving as the same handler.
/// It is routed here rather than reimplemented, so the two cannot diverge.
///
/// # Errors
///
/// Answers `404 session_not_found` when this server holds no such session.
pub async fn delete(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<Deleted>, UhpFailure> {
    // Cancel first, delete second. Deletion is the one place the specification
    // couples the two deliberately: the alternative is a running task writing
    // into storage that no longer has an owner.
    for running in state.store.running_in(&id) {
        state.store.cancel(&running);
    }

    if !state.store.delete_session(&id) {
        return Err(UhpFailure::not_found(
            "session_not_found",
            "no session with that id exists",
        ));
    }
    // A shared view of a deleted session would outlive the thing it shows, so
    // deleting the session takes its link with it.
    state.shares.revoke(&id);
    // §6: deletion removes the working folder the session's tasks wrote, so an
    // artifact cannot outlive the session that produced it.
    super::files::remove_workdir(&id);
    Ok(Json(Deleted {
        id,
        object: "session.deleted",
        deleted: true,
    }))
}

/// What a delete answers with.
#[derive(Debug, Clone, Serialize)]
pub struct Deleted {
    /// The id that was removed.
    pub id: String,
    /// The kind of thing removed.
    pub object: &'static str,
    /// Always true; a failure is an error envelope instead.
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn response(id: &str, created_at: u64) -> Response {
        Response {
            id: id.to_owned(),
            object: "response",
            created_at,
            status: "completed",
            error: None,
            previous_response_id: None,
            model: "m".to_owned(),
            output: Vec::new(),
            store: true,
            usage: None,
            metadata: Map::new(),
        }
    }

    #[test]
    fn a_session_summarises_its_turns_oldest_first() {
        let turns = [response("resp_a", 10), response("resp_b", 20)];
        let s = summarise("sess_1".to_owned(), &turns);
        assert_eq!(s.turn_count, 2);
        assert_eq!(
            s.created_at, 10,
            "a session is as old as its first turn, not its last"
        );
    }

    #[test]
    fn an_empty_session_still_summarises() {
        let s = summarise("sess_empty".to_owned(), &[]);
        assert_eq!(s.turn_count, 0);
        assert_eq!(s.created_at, 0);
    }

    #[test]
    fn a_turn_carries_the_shape_the_contract_names() {
        let t = Turn::from(&response("resp_a", 7));
        assert_eq!(t.id, "resp_a");
        assert_eq!(t.status, "completed", "at least id and status are required");
        assert_eq!(t.created_at, 7);
    }
}
