// ABOUTME: State shared by the UHP routes — provider runners plus the response/session store
// ABOUTME: Kept separate from AppState so the OpenAI surface carries no protocol state
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! What the UHP routes share.
//!
//! The protocol needs something the `OpenAI` surface never did: a task, once
//! run, must be readable back by id and must report the session it belongs to.
//! That store lives here rather than in `AppState` so the surface beside it
//! carries no protocol state it does not use.

use std::sync::Arc;

use axum::extract::FromRef;

use super::manage::ConfiguredHarnesses;
use super::share::Shares;
use super::tasks::ResponseStore;
use crate::state::SharedState;

/// Provider runners plus retained responses.
#[derive(Clone)]
pub struct UhpState {
    /// The runners this server can dispatch a task to.
    pub shared: SharedState,
    /// Responses this server has run and the sessions they belong to.
    pub store: Arc<ResponseStore>,
    /// Harnesses a client configured over the API, as opposed to the ones
    /// discovery found by probing installed binaries.
    pub configured: Arc<ConfiguredHarnesses>,
    /// Public read-only views this server has published.
    pub shares: Arc<Shares>,
}

impl UhpState {
    /// Build state around the shared runner state, with an empty store.
    pub fn new(shared: SharedState) -> Self {
        Self {
            shared,
            store: Arc::new(ResponseStore::default()),
            configured: Arc::new(ConfiguredHarnesses::default()),
            shares: Arc::new(Shares::default()),
        }
    }
}

impl FromRef<UhpState> for SharedState {
    fn from_ref(state: &UhpState) -> Self {
        state.shared.clone()
    }
}
