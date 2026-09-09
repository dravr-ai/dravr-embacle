// ABOUTME: Routes a turn to whichever backend still has budget, and brings the first one back
// ABOUTME: The strategy is a pure decision; the router owns every bit of the I/O
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Budget-driven routing.
//!
//! [`FallbackProvider`](crate::fallback::FallbackProvider) answers a different
//! question. It is reactive and stateless: try one, and when it *fails*, try
//! the next. That is right for a service that breaks, and wrong for a
//! subscription that runs out — by the time the failure arrives, an athlete has
//! already waited for a turn that was never going to work, and nothing
//! remembers to go back afterwards.
//!
//! This module is the other half: proactive, stateful, and able to return.
//! It steps aside *before* a window is exhausted and steps back once the
//! provider says the window has reset. Both live in the crate on purpose; they
//! are different policies, not two attempts at one.
//!
//! ## The seam
//!
//! [`RoutingStrategy`] is a **pure** decision over facts the router has already
//! gathered — no I/O, no clock of its own, no async. Everything expensive
//! (reading quota, honouring cooldowns, retrying) belongs to the router. That
//! makes a strategy trivial to test and makes round-robin, or weighting, or
//! anything else, a small self-contained type rather than a rewrite.
//!
//! ## Limitations
//!
//! Routing state is per-process — see the marker on `RouterProvider.state`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use tracing::{info, warn};

use crate::quota::{LimitChecker, QuotaSnapshot};
use crate::quota_store::{BackendState, InMemoryQuotaStore, QuotaStore};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider, RunnerError,
};

/// How long a reading is reused before the router refreshes it.
///
/// A quota check costs an HTTP round trip, so doing one per turn would put the
/// checker's latency on every athlete's reply. Budgets move slowly; a minute
/// of staleness cannot hide a window emptying.
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_mins(1);

/// Share of a window that counts as "close enough to the wall to step aside".
const DEFAULT_THRESHOLD_PERCENT: f32 = 80.0;

/// How long a backend that refused for quota sits out when it named no reset.
const DEFAULT_PENALTY: Duration = Duration::from_mins(15);

/// One routable provider and the checker that knows its budget.
pub struct Backend {
    /// The provider itself.
    pub provider: Box<dyn LlmProvider>,
    /// Reads this provider's remaining budget. `None` means unmetered: the
    /// router will never step aside from it proactively, only on a refusal.
    pub checker: Option<Box<dyn LimitChecker>>,
}

impl Backend {
    /// A backend whose budget can be read.
    #[must_use]
    pub fn metered(provider: Box<dyn LlmProvider>, checker: Box<dyn LimitChecker>) -> Self {
        Self {
            provider,
            checker: Some(checker),
        }
    }

    /// A backend with no quota signal, used only when others step aside.
    #[must_use]
    pub fn unmetered(provider: Box<dyn LlmProvider>) -> Self {
        Self {
            provider,
            checker: None,
        }
    }
}

/// What the router knows about one backend when it makes a decision.
#[derive(Debug, Clone)]
pub struct BackendView<'a> {
    /// Provider name, for logs and for a strategy that pins by name.
    pub name: &'a str,
    /// Latest windows, empty when nothing has been read.
    pub snapshots: &'a [QuotaSnapshot],
    /// Set while this backend is sitting out after refusing.
    pub penalty_until: Option<SystemTime>,
}

impl BackendView<'_> {
    /// The fullest window, which is the one that will run out first.
    #[must_use]
    pub fn worst_percent(&self) -> Option<f32> {
        self.snapshots
            .iter()
            .map(|s| s.percent)
            .fold(None, |acc: Option<f32>, p| {
                Some(acc.map_or(p, |a| a.max(p)))
            })
    }

    /// Whether this backend may serve a turn right now.
    #[must_use]
    pub fn is_available(&self, now: SystemTime, threshold: f32) -> bool {
        if self.penalty_until.is_some_and(|until| now < until) {
            return false;
        }
        // No reading is not the same as no budget. A checker that cannot answer
        // must not quietly retire a provider that is working perfectly well.
        self.worst_percent().is_none_or(|p| p < threshold)
    }
}

/// Chooses which backend serves the next turn.
///
/// Pure by design: no I/O, no async, and the clock arrives as an argument.
/// Everything a decision needs is in the [`BackendView`]s.
pub trait RoutingStrategy: Send + Sync {
    /// Identifier for logs.
    fn name(&self) -> &str;

    /// Index of the backend that should serve the next turn.
    ///
    /// Returning an index outside `views` is treated as index 0, so a strategy
    /// cannot take the router down by miscounting.
    fn select(&self, views: &[BackendView<'_>], now: SystemTime, threshold: f32) -> usize;
}

/// Prefer the first backend that still has room; fall to the next when it does not.
///
/// Order is priority: index 0 is preferred whenever it is available, so the
/// router returns to it on its own the moment its window resets. There is no
/// separate coordinator, and deliberately no timer — the provider states when
/// the window resets, so coming back is a comparison rather than a schedule.
pub struct PreferInOrder;

impl RoutingStrategy for PreferInOrder {
    fn name(&self) -> &str {
        "prefer-in-order"
    }

    fn select(&self, views: &[BackendView<'_>], now: SystemTime, threshold: f32) -> usize {
        views
            .iter()
            .position(|v| v.is_available(now, threshold))
            // Everything is out of budget. Take the first anyway: refusing to
            // route would fail the turn outright, and a provider that is merely
            // over a soft threshold will very often still answer.
            .unwrap_or(0)
    }
}

/// Routes each turn to a backend that still has budget.
pub struct RouterProvider {
    backends: Vec<Backend>,
    strategy: Box<dyn RoutingStrategy>,
    /// LIMITATION(registre#410): `RouterProvider.store` defaults to
    /// `InMemoryQuotaStore`, which is per-process, so N instances serving one
    /// account each keep their own view of a budget they jointly spend. The
    /// trait is the seam a shared store plugs into without touching the router
    /// or the strategy; nothing here needs to change to close it.
    store: Box<dyn QuotaStore>,
    /// Which backend answered last, so `name()` reports the truth.
    ///
    /// An atomic rather than part of the lock: `name()` returns `&'static str`
    /// and cannot hold a guard across the return.
    active: AtomicUsize,
    threshold: f32,
    refresh_interval: Duration,
    display_name: String,
    combined_models: Vec<String>,
}

impl RouterProvider {
    /// Build a router over at least one backend.
    ///
    /// # Errors
    ///
    /// Returns an error when no backends are given — a router over nothing has
    /// no honest behaviour, and failing at construction beats failing on the
    /// first athlete's turn.
    pub fn new(
        backends: Vec<Backend>,
        strategy: Box<dyn RoutingStrategy>,
    ) -> Result<Self, RunnerError> {
        if backends.is_empty() {
            return Err(RunnerError::config("router requires at least one backend"));
        }
        let display_name = format!(
            "Router ({})",
            backends
                .iter()
                .map(|b| b.provider.display_name())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut combined_models: Vec<String> = Vec::new();
        for b in &backends {
            for m in b.provider.available_models() {
                if !combined_models.contains(m) {
                    combined_models.push(m.clone());
                }
            }
        }
        Ok(Self {
            backends,
            strategy,
            store: Box::new(InMemoryQuotaStore::new()),
            active: AtomicUsize::new(0),
            threshold: DEFAULT_THRESHOLD_PERCENT,
            refresh_interval: DEFAULT_REFRESH_INTERVAL,
            display_name,
            combined_models,
        })
    }

    /// Swap the state store — the seam a shared implementation plugs into.
    ///
    /// The default is [`InMemoryQuotaStore`], correct for one process. A Redis
    /// or Postgres store is a different type passed here, with no change to
    /// routing or to the strategy.
    #[must_use]
    pub fn with_store(mut self, store: Box<dyn QuotaStore>) -> Self {
        self.store = store;
        self
    }

    /// Set the share of a window at which the router steps aside.
    #[must_use]
    pub const fn with_threshold(mut self, percent: f32) -> Self {
        self.threshold = percent;
        self
    }

    /// Set how long a quota reading is reused before refreshing.
    #[must_use]
    pub const fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }

    /// Index of the backend currently serving.
    #[must_use]
    pub fn active_index(&self) -> usize {
        self.active
            .load(Ordering::Relaxed)
            .min(self.backends.len() - 1)
    }

    /// Refresh any reading old enough to be worth re-reading.
    ///
    /// Takes the loaded state so staleness is judged without a second read,
    /// and writes through the store so a shared one sees the reading too.
    async fn refresh_stale(&self, now: SystemTime, state: &[BackendState]) {
        for (i, backend) in self.backends.iter().enumerate() {
            let Some(checker) = backend.checker.as_ref() else {
                continue;
            };
            let due = state.get(i).is_none_or(|b| {
                b.last_checked.is_none_or(|t| {
                    now.duration_since(t)
                        .is_ok_and(|age| age >= self.refresh_interval)
                })
            });
            if !due {
                continue;
            }
            match checker.check().await {
                Ok(snapshots) => {
                    self.store.record_reading(i, snapshots, now).await;
                }
                Err(e) => {
                    // An unreadable quota is not an exhausted one. Log and keep
                    // whatever was last known rather than retiring a provider
                    // that may be perfectly healthy.
                    warn!(
                        checker = checker.name(),
                        error = %e,
                        "quota check failed; keeping the previous reading"
                    );
                }
            }
        }
    }

    /// Decide who serves this turn.
    async fn choose(&self, now: SystemTime) -> usize {
        let state = self.store.load(self.backends.len()).await;
        self.refresh_stale(now, &state).await;
        // Re-load: refresh_stale may have written through the store, and a
        // shared store can also have moved under us between the two calls.
        let state = self.store.load(self.backends.len()).await;
        let views: Vec<BackendView<'_>> = self
            .backends
            .iter()
            .zip(state.iter())
            .map(|(b, s)| BackendView {
                name: b.provider.name(),
                snapshots: &s.snapshots,
                penalty_until: s.penalty_until,
            })
            .collect();
        let idx = self
            .strategy
            .select(&views, now, self.threshold)
            .min(self.backends.len() - 1);
        self.active.store(idx, Ordering::Relaxed);
        idx
    }

    /// Record that a backend refused for quota, and for how long to stay away.
    async fn penalise(&self, index: usize, now: SystemTime) {
        let state = self.store.load(self.backends.len()).await;
        let until = state
            .get(index)
            .and_then(|b| {
                b.snapshots
                    .iter()
                    .filter(|s| s.resets_at > now)
                    .map(|s| s.resets_at)
                    .min()
            })
            // The provider refused without our knowing when it resets. Sit out
            // a bounded stretch rather than forever: a guessed reset would be
            // fiction, but never coming back would be worse.
            .unwrap_or(now + DEFAULT_PENALTY);
        self.store.set_penalty(index, Some(until)).await;
        warn!(
            backend = self.backends[index].provider.name(),
            "backend refused for quota; stepping aside until its window resets"
        );
    }

    /// Index of the next backend that is not `avoid`, if there is one.
    async fn next_other(&self, avoid: usize, now: SystemTime) -> Option<usize> {
        let state = self.store.load(self.backends.len()).await;
        (0..self.backends.len()).find(|&i| {
            i != avoid
                && state
                    .get(i)
                    .is_none_or(|b| b.penalty_until.is_none_or(|u| now >= u))
        })
    }
}

#[async_trait]
impl LlmProvider for RouterProvider {
    fn name(&self) -> &'static str {
        // The backend that actually answered, not the head of a list. Every
        // provider name is a 'static literal, so this stays honest without
        // synthesising a string — and it keeps usage rows attributed to
        // whoever really served the turn.
        self.backends[self.active_index()].provider.name()
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.backends[self.active_index()].provider.capabilities()
    }

    fn default_model(&self) -> &str {
        self.backends[self.active_index()].provider.default_model()
    }

    fn available_models(&self) -> &[String] {
        &self.combined_models
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let now = SystemTime::now();
        let idx = self.choose(now).await;
        match self.backends[idx].provider.complete(request).await {
            Err(e) if e.kind == ErrorKind::RateLimit => {
                self.penalise(idx, now).await;
                let Some(other) = self.next_other(idx, now).await else {
                    return Err(e);
                };
                info!(
                    from = self.backends[idx].provider.name(),
                    to = self.backends[other].provider.name(),
                    "rerouting after a quota refusal"
                );
                self.active.store(other, Ordering::Relaxed);
                self.backends[other].provider.complete(request).await
            }
            other => other,
        }
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let now = SystemTime::now();
        let idx = self.choose(now).await;
        match self.backends[idx].provider.complete_stream(request).await {
            Err(e) if e.kind == ErrorKind::RateLimit => {
                self.penalise(idx, now).await;
                let Some(other) = self.next_other(idx, now).await else {
                    return Err(e);
                };
                self.active.store(other, Ordering::Relaxed);
                self.backends[other].provider.complete_stream(request).await
            }
            other => other,
        }
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        // Healthy when any backend is, matching FallbackProvider: the router's
        // whole purpose is that one being out does not stop the service.
        for b in &self.backends {
            if b.provider.health_check().await.unwrap_or(false) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, TokenUsage};
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Mutex};
    use std::time::UNIX_EPOCH;

    fn at(epoch: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(epoch)
    }

    /// A provider that answers with its own name, and counts how often it is asked.
    struct TestProvider {
        provider_name: &'static str,
        models: Vec<String>,
        scripted: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        calls: Arc<AtomicU32>,
    }

    impl TestProvider {
        fn ok(name: &'static str) -> Self {
            Self::counted(name, Arc::new(AtomicU32::new(0)))
        }

        /// Share the call counter with the test, so an assertion can say
        /// "this provider was never asked" — which is the actual claim when
        /// the router steps aside, and is stronger than reading the reply.
        fn counted(name: &'static str, calls: Arc<AtomicU32>) -> Self {
            Self {
                provider_name: name,
                models: vec![format!("{name}-model")],
                scripted: Mutex::new(Vec::new()),
                calls,
            }
        }

        fn scripted(name: &'static str, script: Vec<Result<ChatResponse, RunnerError>>) -> Self {
            Self {
                provider_name: name,
                models: vec![format!("{name}-model")],
                scripted: Mutex::new(script),
                calls: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    fn answer(from: &str) -> ChatResponse {
        ChatResponse {
            content: format!("answered by {from}"),
            model: format!("{from}-model"),
            usage: Some(TokenUsage::new(1, 1, 2)),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }
        fn display_name(&self) -> &str {
            self.provider_name
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::STREAMING
        }
        fn default_model(&self) -> &str {
            &self.models[0]
        }
        fn available_models(&self) -> &[String] {
            &self.models
        }
        async fn complete(&self, _r: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let scripted = self.scripted.lock().ok().and_then(|mut s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s.remove(0))
                }
            });
            scripted.unwrap_or_else(|| Ok(answer(self.provider_name)))
        }
        async fn complete_stream(&self, _r: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("not used in these tests"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
        }
    }

    /// A checker that reports whatever the test says, or fails on demand.
    struct FakeChecker {
        checker_name: &'static str,
        result: Mutex<Result<Vec<QuotaSnapshot>, RunnerError>>,
    }

    impl FakeChecker {
        fn at_percent(name: &'static str, percent: f32, resets_at: u64) -> Self {
            Self {
                checker_name: name,
                result: Mutex::new(Ok(vec![QuotaSnapshot {
                    key: "weekly_all".to_owned(),
                    label: "Weekly".to_owned(),
                    percent,
                    resets_at: at(resets_at),
                    observed_at: at(0),
                }])),
            }
        }

        fn failing(name: &'static str) -> Self {
            Self {
                checker_name: name,
                result: Mutex::new(Err(RunnerError::external_service(name, "unreachable"))),
            }
        }
    }

    #[async_trait]
    impl LimitChecker for FakeChecker {
        fn name(&self) -> &str {
            self.checker_name
        }
        async fn check(&self) -> Result<Vec<QuotaSnapshot>, RunnerError> {
            self.result
                .lock()
                .map_or_else(|_| Err(RunnerError::internal("poisoned")), |g| g.clone())
        }
    }

    fn request() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello")])
    }

    // ---- the pure strategy ----

    fn view<'a>(
        name: &'a str,
        snaps: &'a [QuotaSnapshot],
        penalty: Option<SystemTime>,
    ) -> BackendView<'a> {
        BackendView {
            name,
            snapshots: snaps,
            penalty_until: penalty,
        }
    }

    fn snaps(percent: f32, resets_at: u64) -> Vec<QuotaSnapshot> {
        vec![QuotaSnapshot {
            key: "weekly_all".to_owned(),
            label: "Weekly".to_owned(),
            percent,
            resets_at: at(resets_at),
            observed_at: at(0),
        }]
    }

    #[test]
    fn the_first_backend_wins_while_it_has_room() {
        let a = snaps(10.0, 9_999);
        let b = snaps(0.0, 9_999);
        let views = [view("a", &a, None), view("b", &b, None)];
        assert_eq!(PreferInOrder.select(&views, at(100), 80.0), 0);
    }

    #[test]
    fn crossing_the_threshold_moves_the_selection() {
        let a = snaps(80.0, 9_999);
        let b = snaps(0.0, 9_999);
        let views = [view("a", &a, None), view("b", &b, None)];
        assert_eq!(
            PreferInOrder.select(&views, at(100), 80.0),
            1,
            "exactly at the threshold counts as reached; waiting for 81 spends the last of \
             the window on turns that could have gone elsewhere"
        );
    }

    #[test]
    fn the_fullest_window_decides_not_the_average() {
        // Observed live: weekly-all at 85% while the scoped bucket was at 100%.
        // Averaging would call that 92 and keep routing at a dead model.
        let mut a = snaps(85.0, 9_999);
        a.push(QuotaSnapshot {
            key: "weekly_scoped:Fable".to_owned(),
            label: "Weekly (Fable)".to_owned(),
            percent: 100.0,
            resets_at: at(9_999),
            observed_at: at(0),
        });
        let b = snaps(0.0, 9_999);
        let views = [view("a", &a, None), view("b", &b, None)];
        assert_eq!(views[0].worst_percent(), Some(100.0));
        assert_eq!(PreferInOrder.select(&views, at(100), 90.0), 1);
    }

    #[test]
    fn a_backend_with_no_reading_is_not_retired() {
        // An unreadable quota is not an exhausted one. Treating "I don't know"
        // as "it's full" would park every turn on the secondary the moment the
        // usage endpoint had a bad minute.
        let empty: Vec<QuotaSnapshot> = Vec::new();
        let b = snaps(0.0, 9_999);
        let views = [view("a", &empty, None), view("b", &b, None)];
        assert_eq!(PreferInOrder.select(&views, at(100), 80.0), 0);
    }

    #[test]
    fn a_penalty_holds_only_until_its_instant() {
        let a = snaps(0.0, 9_999);
        let b = snaps(0.0, 9_999);
        let views = [view("a", &a, Some(at(500))), view("b", &b, None)];
        assert_eq!(
            PreferInOrder.select(&views, at(499), 80.0),
            1,
            "still penalised"
        );
        assert_eq!(
            PreferInOrder.select(&views, at(500), 80.0),
            0,
            "the stated reset instant returns the preferred backend, with no timer anywhere"
        );
    }

    #[test]
    fn everything_exhausted_still_routes_somewhere() {
        // Refusing to route would fail the turn outright, and a provider merely
        // over a soft threshold will often still answer.
        let a = snaps(100.0, 9_999);
        let b = snaps(100.0, 9_999);
        let views = [view("a", &a, None), view("b", &b, None)];
        assert_eq!(PreferInOrder.select(&views, at(100), 80.0), 0);
    }

    // ---- the router ----

    #[tokio::test]
    async fn a_router_needs_a_backend() {
        assert!(RouterProvider::new(Vec::new(), Box::new(PreferInOrder)).is_err());
    }

    #[tokio::test]
    async fn the_preferred_backend_serves_while_it_has_room() {
        let router = RouterProvider::new(
            vec![
                Backend::metered(
                    Box::new(TestProvider::ok("claude-code")),
                    Box::new(FakeChecker::at_percent("c", 10.0, 9_999_999_999)),
                ),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion

        let r = router.complete(&request()).await.unwrap(); // Safe: test assertion
        assert_eq!(r.content, "answered by claude-code");
        assert_eq!(
            router.name(),
            "claude-code",
            "name() must report who answered"
        );
    }

    #[tokio::test]
    async fn a_full_window_steps_aside_before_a_turn_is_spent() {
        let primary_calls = Arc::new(AtomicU32::new(0));
        let router = RouterProvider::new(
            vec![
                Backend::metered(
                    Box::new(TestProvider::counted(
                        "claude-code",
                        Arc::clone(&primary_calls),
                    )),
                    Box::new(FakeChecker::at_percent("c", 95.0, 9_999_999_999)),
                ),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion

        let r = router.complete(&request()).await.unwrap(); // Safe: test assertion
        assert_eq!(r.content, "answered by copilot_headless");
        assert_eq!(
            primary_calls.load(Ordering::Relaxed),
            0,
            "the exhausted provider must never be asked at all — stepping aside AFTER a \
             failed turn is what the reactive path already did, and it costs an athlete a wait"
        );
        assert_eq!(router.name(), "copilot_headless");
    }

    #[tokio::test]
    async fn a_quota_refusal_reroutes_within_the_same_turn() {
        // The reactive half: the reading said there was room, the provider
        // disagreed. The athlete must still get an answer.
        let primary = TestProvider::scripted(
            "claude-code",
            vec![Err(RunnerError::rate_limit(
                "claude-code",
                "usage limit reached",
            ))],
        );
        let router = RouterProvider::new(
            vec![
                Backend::metered(
                    Box::new(primary),
                    Box::new(FakeChecker::at_percent("c", 5.0, 9_999_999_999)),
                ),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion

        let r = router.complete(&request()).await.unwrap(); // Safe: test assertion
        assert_eq!(r.content, "answered by copilot_headless");
        assert_eq!(router.name(), "copilot_headless");
    }

    #[tokio::test]
    async fn an_ordinary_failure_is_not_rerouted() {
        // Only a quota refusal reroutes. A genuine fault must surface, or a
        // broken primary silently becomes a permanently paid secondary.
        let primary = TestProvider::scripted(
            "claude-code",
            vec![Err(RunnerError::external_service(
                "claude-code",
                "stream closed",
            ))],
        );
        let secondary = TestProvider::ok("copilot_headless");
        let router = RouterProvider::new(
            vec![
                Backend::unmetered(Box::new(primary)),
                Backend::unmetered(Box::new(secondary)),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion

        let err = router.complete(&request()).await.unwrap_err(); // Safe: test assertion
        assert_eq!(err.kind, ErrorKind::ExternalService);
    }

    #[tokio::test]
    async fn a_failing_checker_does_not_park_us_on_the_secondary() {
        let router = RouterProvider::new(
            vec![
                Backend::metered(
                    Box::new(TestProvider::ok("claude-code")),
                    Box::new(FakeChecker::failing("c")),
                ),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion

        let r = router.complete(&request()).await.unwrap(); // Safe: test assertion
        assert_eq!(
            r.content, "answered by claude-code",
            "a checker outage must not quietly retire a provider that is working"
        );
    }

    #[tokio::test]
    async fn a_reading_is_reused_rather_than_refetched_every_turn() {
        // A check per turn would put the checker's HTTP latency on every reply.
        let checker = FakeChecker::at_percent("c", 10.0, 9_999_999_999);
        let router = RouterProvider::new(
            vec![
                Backend::metered(Box::new(TestProvider::ok("claude-code")), Box::new(checker)),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap() // Safe: test assertion
        .with_refresh_interval(Duration::from_hours(1));

        for _ in 0..3 {
            router.complete(&request()).await.unwrap(); // Safe: test assertion
        }
        assert_eq!(router.active_index(), 0);
    }

    /// A store that is not the shipped one, to prove the seam is usable from
    /// outside rather than being a trait with a single blessed implementation.
    /// A Redis or Postgres store is this shape.
    struct SeededStore {
        seeded: Vec<BackendState>,
    }

    #[async_trait]
    impl QuotaStore for SeededStore {
        fn name(&self) -> &str {
            "seeded"
        }
        async fn load(&self, len: usize) -> Vec<BackendState> {
            let mut out = self.seeded.clone();
            out.resize(len, BackendState::default());
            out
        }
        async fn record_reading(&self, _i: usize, _s: Vec<QuotaSnapshot>, _at: SystemTime) {}
        async fn set_penalty(&self, _i: usize, _until: Option<SystemTime>) {}
    }

    #[tokio::test]
    async fn a_foreign_store_drives_the_decision() {
        // State the router never gathered itself — which is exactly what a
        // shared store contributes: one instance's reading steering another's
        // turn. The primary carries no checker, so nothing but the store can
        // explain the step-aside.
        let store = SeededStore {
            seeded: vec![
                BackendState {
                    snapshots: vec![QuotaSnapshot {
                        key: "weekly_all".to_owned(),
                        label: "Weekly".to_owned(),
                        percent: 99.0,
                        resets_at: at(9_999_999_999),
                        observed_at: at(0),
                    }],
                    last_checked: Some(at(9_999_999_999)),
                    penalty_until: None,
                },
                BackendState::default(),
            ],
        };
        let router = RouterProvider::new(
            vec![
                Backend::unmetered(Box::new(TestProvider::ok("claude-code"))),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap() // Safe: test assertion
        .with_store(Box::new(store));

        let r = router.complete(&request()).await.unwrap(); // Safe: test assertion
        assert_eq!(
            r.content, "answered by copilot_headless",
            "the router must act on state it did not gather, or a shared store buys nothing"
        );
    }

    #[tokio::test]
    async fn the_model_list_is_the_union_of_every_backend() {
        let router = RouterProvider::new(
            vec![
                Backend::unmetered(Box::new(TestProvider::ok("claude-code"))),
                Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
            ],
            Box::new(PreferInOrder),
        )
        .unwrap(); // Safe: test assertion
        assert_eq!(
            router.available_models(),
            ["claude-code-model", "copilot_headless-model"]
        );
    }
}
