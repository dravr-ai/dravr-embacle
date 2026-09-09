// ABOUTME: Where routing state lives, behind a trait so the process is not the only answer
// ABOUTME: In-memory is the shipped implementation; a shared one is a plugin, not a rewrite
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Routing state storage.
//!
//! The router needs to remember three things per backend: the last quota
//! reading, when it was taken, and any penalty in force. Where that lives is a
//! deployment question, not a routing one — a single process is happy with a
//! lock, and several processes sharing one account are not.
//!
//! So it sits behind [`QuotaStore`]. [`InMemoryQuotaStore`] is what ships and
//! what a single process wants; a Redis or Postgres implementation is a new
//! type in a new file, with no change to the router or to the strategy. The
//! trait is async because a shared store performs I/O even though the
//! in-memory one never blocks.

use std::sync::RwLock;
use std::time::SystemTime;

use async_trait::async_trait;

use crate::quota::QuotaSnapshot;

/// What the router remembers about one backend between turns.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct BackendState {
    /// Latest windows, empty when nothing has been read.
    pub snapshots: Vec<QuotaSnapshot>,
    /// When that reading was taken, `None` when there has never been one.
    pub last_checked: Option<SystemTime>,
    /// Set while this backend sits out after refusing for quota.
    pub penalty_until: Option<SystemTime>,
}

/// Holds routing state for every backend.
///
/// Implementations must tolerate a `len` that does not match their stored
/// width — a router can be rebuilt with a different backend list against a
/// store that outlives it — by returning exactly `len` entries, padding with
/// defaults. A short vec would panic the router's indexing; that is the
/// contract's whole job.
#[async_trait]
pub trait QuotaStore: Send + Sync {
    /// Identifier for logs.
    fn name(&self) -> &str;

    /// Every backend's state, exactly `len` entries.
    async fn load(&self, len: usize) -> Vec<BackendState>;

    /// Record a fresh reading for one backend.
    async fn record_reading(
        &self,
        index: usize,
        snapshots: Vec<QuotaSnapshot>,
        observed_at: SystemTime,
    );

    /// Set or clear a backend's penalty.
    async fn set_penalty(&self, index: usize, until: Option<SystemTime>);
}

/// Routing state held in this process, and nowhere else.
///
/// Correct for a single process and the default for that reason. Under
/// several processes sharing one account, each keeps its own view of a budget
/// they jointly spend — see the `LIMITATION` on `RouterProvider`.
#[derive(Debug, Default)]
pub struct InMemoryQuotaStore {
    state: RwLock<Vec<BackendState>>,
}

impl InMemoryQuotaStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grow the backing vec so `index` is addressable.
    fn ensure(state: &mut Vec<BackendState>, index: usize) {
        if state.len() <= index {
            state.resize(index + 1, BackendState::default());
        }
    }
}

#[async_trait]
impl QuotaStore for InMemoryQuotaStore {
    fn name(&self) -> &str {
        "in-memory"
    }

    async fn load(&self, len: usize) -> Vec<BackendState> {
        let mut out = self.state.read().map_or_else(|_| Vec::new(), |s| s.clone());
        out.resize(len, BackendState::default());
        out
    }

    async fn record_reading(
        &self,
        index: usize,
        snapshots: Vec<QuotaSnapshot>,
        observed_at: SystemTime,
    ) {
        if let Ok(mut state) = self.state.write() {
            Self::ensure(&mut state, index);
            if let Some(b) = state.get_mut(index) {
                b.snapshots = snapshots;
                b.last_checked = Some(observed_at);
                // A reading that came back means the provider is answering
                // again, so a penalty whose instant has passed is stale.
                if b.penalty_until.is_some_and(|u| observed_at >= u) {
                    b.penalty_until = None;
                }
            }
        }
    }

    async fn set_penalty(&self, index: usize, until: Option<SystemTime>) {
        if let Ok(mut state) = self.state.write() {
            Self::ensure(&mut state, index);
            if let Some(b) = state.get_mut(index) {
                b.penalty_until = until;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    fn at(epoch: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(epoch)
    }

    fn snap(percent: f32) -> QuotaSnapshot {
        QuotaSnapshot {
            key: "weekly_all".to_owned(),
            label: "Weekly".to_owned(),
            percent,
            resets_at: at(9_999),
            observed_at: at(0),
        }
    }

    #[tokio::test]
    async fn an_empty_store_still_answers_for_every_backend() {
        // The router indexes straight into this. A short vec panics it, which
        // is why the width is the contract rather than the caller's problem.
        let store = InMemoryQuotaStore::new();
        assert_eq!(store.load(3).await.len(), 3);
    }

    #[tokio::test]
    async fn a_reading_survives_and_is_addressable_out_of_order() {
        let store = InMemoryQuotaStore::new();
        store.record_reading(2, vec![snap(42.0)], at(100)).await;
        let loaded = store.load(3).await;
        assert_eq!(loaded[2].snapshots.len(), 1);
        assert!((loaded[2].snapshots[0].percent - 42.0).abs() < f32::EPSILON);
        assert_eq!(loaded[2].last_checked, Some(at(100)));
        assert_eq!(
            loaded[0],
            BackendState::default(),
            "untouched slots stay empty"
        );
    }

    #[tokio::test]
    async fn a_reading_clears_a_penalty_whose_instant_has_passed() {
        let store = InMemoryQuotaStore::new();
        store.set_penalty(0, Some(at(500))).await;
        store.record_reading(0, vec![snap(1.0)], at(600)).await;
        assert_eq!(store.load(1).await[0].penalty_until, None);
    }

    #[tokio::test]
    async fn a_reading_leaves_a_penalty_that_is_still_live() {
        let store = InMemoryQuotaStore::new();
        store.set_penalty(0, Some(at(500))).await;
        store.record_reading(0, vec![snap(1.0)], at(400)).await;
        assert_eq!(
            store.load(1).await[0].penalty_until,
            Some(at(500)),
            "a reading is not permission to serve; the window has not reset yet"
        );
    }

    #[tokio::test]
    async fn a_narrower_load_does_not_lose_what_was_stored() {
        let store = InMemoryQuotaStore::new();
        store.record_reading(1, vec![snap(9.0)], at(1)).await;
        assert_eq!(
            store.load(1).await.len(),
            1,
            "truncated to what was asked for"
        );
        assert_eq!(store.load(2).await[1].snapshots.len(), 1, "and still there");
    }
}
