// ABOUTME: The Unified Harness Protocol surface, mounted under its own base path
// ABOUTME: Additive — the OpenAI-compatible API at /v1 is untouched and keeps its shapes
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Unified Harness Protocol (`2026-08-11`).
//!
//! embacle already puts twelve CLI harnesses behind one trait, which is what
//! UHP calls a *runner*. This module serves that catalogue over the protocol's
//! HTTP contract.
//!
//! ## Why it mounts under a prefix
//!
//! Both surfaces define `GET /v1/models`, and the bodies are incompatible:
//! the `OpenAI` API answers a flat `{"object":"list","data":[…]}`, UHP requires
//! `{"backends":{"<id>":{"default":…,"models":[…]}}}`. Serving both on one path
//! would mean content negotiation or breaking every existing caller.
//!
//! The protocol anticipates this — its published `servers:` block lists
//! `http://127.0.0.1:3000/api/harness` — so UHP is designed to sit under an
//! arbitrary base path, and the conformance suite takes a `--base-url`. The
//! `OpenAI` surface keeps `/v1` exactly as it was.
//!
//! ## What is implemented
//!
//! Discovery, harnesses and the model catalogue, with version negotiation and
//! the structured error envelope beneath them. Tasks, streaming, sessions and
//! files are not yet served; [`discovery::Capabilities::current`] reports that
//! honestly rather than advertising surfaces that would fail on first use.

pub mod auth;
pub mod discovery;
pub mod error;
pub mod files;
pub mod harnesses;
pub mod manage;
pub mod sessions;
pub mod share;
pub mod state;
pub mod streaming;
pub mod tasks;
pub mod version;

use std::env;

use axum::middleware;
use axum::routing::{delete, get, post};
use axum::Router;

use crate::state::AppState;

/// Environment variable naming the base path the protocol is served under.
pub const BASE_PATH_ENV: &str = "UHP_BASE_PATH";

/// Where the protocol is served when the environment names nothing.
pub const DEFAULT_BASE_PATH: &str = "/uhp";

/// The base path this server serves UHP under, normalised to a leading slash
/// and no trailing one so it can be concatenated with the protocol's own paths.
pub fn base_path() -> String {
    let raw = env::var(BASE_PATH_ENV).unwrap_or_else(|_| DEFAULT_BASE_PATH.to_owned());
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else if trimmed.starts_with('/') {
        trimmed.to_owned()
    } else {
        format!("/{trimmed}")
    }
}

/// Build the UHP router, to be nested under [`base_path`].
///
/// Discovery sits outside the auth layer deliberately: a client must be able to
/// learn whether this is a UHP server, and which versions it speaks, before
/// deciding what credential to present. Everything else requires one when
/// `EMBACLE_API_KEY` is configured.
///
/// Version negotiation wraps both, so even a refusal carries `UHP-Version`.
pub fn router(state: AppState) -> Router {
    let uhp_state = state::UhpState::new(state.shared);
    let public_state = uhp_state.clone();

    let authenticated = Router::new()
        .route("/v1/harnesses", get(harnesses::list).post(manage::create))
        .route(
            "/v1/harnesses/{harness_id}",
            get(harnesses::get)
                .put(manage::update)
                .delete(manage::delete),
        )
        .route(
            "/v1/harnesses/{harness_id}/models",
            get(harnesses::harness_models),
        )
        .route(
            "/v1/harnesses/{harness_id}/skills/{skill_name}/files",
            get(manage::skill_files),
        )
        .route("/v1/models", get(harnesses::models))
        .route("/v1/responses", post(tasks::create))
        .route("/v1/responses/{response_id}", get(tasks::get))
        .route("/v1/responses/{response_id}/cancel", post(tasks::cancel))
        .route("/v1/sessions", get(sessions::list))
        .route(
            "/v1/sessions/{session_id}",
            get(sessions::get).delete(sessions::delete),
        )
        // The protocol names /v1/sessions/{id}; /v1/traces/{id} is the older
        // path for the same operation, which a server MAY keep serving. Same
        // handler, so the two cannot drift.
        .route("/v1/traces/{session_id}", delete(sessions::delete))
        .route("/v1/sessions/{session_id}/turns", get(sessions::turns))
        .route(
            "/v1/sessions/{session_id}/share",
            get(share::get).post(share::create).delete(share::revoke),
        )
        .route("/v1/sessions/{session_id}/files", get(files::session_files))
        .route(
            "/v1/containers/{container_id}/files/{file_id}/content",
            get(files::container_file),
        )
        .with_state(uhp_state)
        .layer(middleware::from_fn(auth::require_auth));

    // The shared view sits beside discovery, outside the auth layer: a share
    // only its minter can open has not been published to anyone.
    let public = Router::new()
        .route("/share/{share_id}", get(share::view))
        .with_state(public_state);

    Router::new()
        .route("/v1/uhp", get(discovery::handle))
        .merge(public)
        .merge(authenticated)
        .fallback(harnesses::unknown_route)
        .layer(middleware::from_fn(version::negotiate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_base_path_is_normalised_and_can_be_emptied() {
        // One test, not two: UHP_BASE_PATH is process-global, so two tests
        // mutating it race each other and fail intermittently.
        //
        // The path is concatenated with the protocol's own "/v1/..." paths, so
        // a trailing slash would produce "//v1/uhp" and route nowhere.
        env::set_var(BASE_PATH_ENV, "/api/harness/");
        assert_eq!(base_path(), "/api/harness");

        env::set_var(BASE_PATH_ENV, "harness");
        assert_eq!(base_path(), "/harness");

        // Deliberate: an operator who does not run the OpenAI surface can put
        // UHP at the root, which is what most UHP clients expect.
        env::set_var(BASE_PATH_ENV, "/");
        assert_eq!(base_path(), "");

        env::remove_var(BASE_PATH_ENV);
        assert_eq!(base_path(), DEFAULT_BASE_PATH);
    }
}
