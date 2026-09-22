// ABOUTME: Provider fallback chains that try multiple LlmProviders in order
// ABOUTME: Retry in place, fall through on the provider's fault, reject an empty answer, observe every hop
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Provider Fallback Chains
//!
//! [`FallbackProvider`] wraps multiple `Box<dyn LlmProvider>` instances and
//! asks them in order for each request. The first accepted response is
//! returned; if every tier fails, the last error is propagated.
//!
//! Three policies decide what happens between one tier and the next:
//!
//! - **Retry in place** — [`RetryConfig`] via [`FallbackProvider::with_retry()`]
//!   re-asks the *same* tier after exponential backoff while the error is
//!   transient (see [`ErrorKind::is_transient()`]).
//! - **Fall through** — [`ResponsePolicy`] via
//!   [`FallbackProvider::with_fallthrough()`] decides which errors move the
//!   request to the *next* tier. The default, [`FallThrough::Always`], moves on
//!   every error; [`FallThrough::ProviderFault`] moves only when the provider,
//!   not the request, failed (see [`ErrorKind::is_provider_fault()`]), so a
//!   malformed request surfaces its own diagnostic instead of the next
//!   provider's version of the same rejection.
//! - **Reject** — the same policy can treat an `Ok` that carries nothing
//!   deliverable (blank content, no tool call) as a failed tier, and can clear
//!   `request.model` before the next tier so each provider resolves its own
//!   configured model instead of the first tier's namespace.
//!
//! A [`FallbackObserver`] attached with [`FallbackProvider::with_observer()`]
//! sees every hop — and can veto a tier before it is asked, which is how a
//! circuit breaker that lives outside the chain steers it. Every callback runs
//! synchronously on the calling task, so the caller's tracing span is live
//! inside each one.
//!
//! Health checks pass if ANY provider is healthy; otherwise the last tier's
//! outcome is returned verbatim, so an `Err` from the final tier reaches the
//! caller with its diagnostic intact. Capabilities are the bitwise OR of all
//! inner providers.
//!
//! [`RouterProvider`](crate::router::RouterProvider) is the deliberate
//! sibling: it routes *proactively* on quota readings, where this chain reacts
//! to outcomes. The two compose.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::time;
use tracing::warn;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider, RunnerError,
};

/// Configuration for per-provider retry with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts per provider (0 = no retries, original behavior)
    pub max_retries: u32,
    /// Base delay between retries (doubled on each attempt)
    pub base_delay: Duration,
    /// Upper bound on the delay between retries
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 0,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(5),
        }
    }
}

/// Which errors move a request to the next tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallThrough {
    /// Every error moves on. The default.
    #[default]
    Always,
    /// Only a provider fault moves on ([`ErrorKind::is_provider_fault`]); a
    /// deterministic rejection propagates so the caller sees the real
    /// diagnostic.
    ProviderFault,
}

impl FallThrough {
    /// Whether an error of this kind moves the request to the next tier.
    #[must_use]
    pub const fn allows(self, kind: ErrorKind) -> bool {
        match self {
            Self::Always => true,
            Self::ProviderFault => kind.is_provider_fault(),
        }
    }
}

/// What the chain does with one tier's outcome before asking the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResponsePolicy {
    /// Which errors move the request on.
    pub fall_through: FallThrough,
    /// An `Ok` whose content is blank and carries no tool call is treated as
    /// a failed tier ([`FallthroughReason::EmptyCompletion`]). An empty answer
    /// from the last tier is still returned as `Ok` — there is nobody left to
    /// ask.
    pub reject_empty_completion: bool,
    /// Clear `request.model` before the next tier so it resolves its own
    /// configured model instead of the first tier's namespace.
    pub own_model_per_tier: bool,
}

impl ResponsePolicy {
    /// Fall through on every error, accept every `Ok`, forward the request
    /// untouched. Equal to `Default`.
    #[must_use]
    pub const fn permissive() -> Self {
        Self {
            fall_through: FallThrough::Always,
            reject_empty_completion: false,
            own_model_per_tier: false,
        }
    }

    /// Fall through only on a provider fault, reject an empty completion, and
    /// give every tier its own model.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            fall_through: FallThrough::ProviderFault,
            reject_empty_completion: true,
            own_model_per_tier: true,
        }
    }
}

/// `true` when a completion carries nothing the caller can deliver.
///
/// Blank means `content.trim().is_empty()` — every surface trims before
/// rendering, so three spaces reach the reader as nothing at all — and no
/// non-empty `tool_calls`. `Some(vec![])` reads as none: providers disagree on
/// which they send, and a model that calls a tool instead of speaking returns
/// empty content by design, which must NOT count as empty.
#[must_use]
pub fn is_empty_completion(response: &ChatResponse) -> bool {
    let has_tool_calls = response
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty());
    response.content.trim().is_empty() && !has_tool_calls
}

/// One tier of the chain, as the observer sees it.
#[derive(Clone, Copy)]
pub struct Tier<'a> {
    /// Zero-based position in the chain; `0` is the primary.
    pub position: usize,
    /// The provider at that position.
    pub provider: &'a dyn LlmProvider,
}

/// Why a tier was passed over.
pub enum FallthroughReason<'a> {
    /// [`FallbackObserver::before_attempt`] vetoed the tier; the observer's
    /// own reason string.
    Skipped(&'static str),
    /// An `Ok` that carried nothing deliverable.
    EmptyCompletion,
    /// An error the [`FallThrough`] policy classified as the provider's fault.
    Error(&'a RunnerError),
}

/// The observer's answer to "may this tier be asked?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attempt {
    /// Ask it.
    Try,
    /// Pass it over, for this reason.
    Skip(&'static str),
}

/// Per-attempt visibility, plus the one veto a breaker that lives outside
/// the chain needs.
///
/// Every method is called synchronously on the calling task, never spawned,
/// so the caller's tracing span is live inside each callback. Each has a
/// no-op default, so an observer implements only the hops it cares about.
pub trait FallbackObserver: Send + Sync {
    /// Consulted only when a later tier exists — the last tier is always
    /// asked.
    fn before_attempt(&self, tier: Tier<'_>) -> Attempt {
        let _ = tier;
        Attempt::Try
    }

    /// `from` was passed over for `reason`; `to` is the tier asked next.
    fn on_fallthrough(&self, from: Tier<'_>, to: Tier<'_>, reason: FallthroughReason<'_>) {
        let _ = (from, to, reason);
    }

    /// A tier's `complete()` returned an accepted response, or its stream
    /// opened.
    fn on_success(&self, tier: Tier<'_>) {
        let _ = tier;
    }

    /// The last tier failed too; `error` is what `complete()` returns.
    fn on_exhausted(&self, last: Tier<'_>, error: &RunnerError) {
        let _ = (last, error);
    }
}

/// One tier's call, boxed so the driver can run `complete` and
/// `complete_stream` through the same loop.
type TierCall<'r, T> = Pin<Box<dyn Future<Output = Result<T, RunnerError>> + Send + 'r>>;

/// A provider method the driver can call on any tier with any request
/// borrow: `complete` or `complete_stream`, boxed.
type TierMethod<'m, T> =
    &'m (dyn for<'r> Fn(&'r dyn LlmProvider, &'r ChatRequest) -> TierCall<'r, T> + Sync);

/// Provider that tries multiple inner providers in order, returning the first success.
///
/// # Construction
///
/// Use [`FallbackProvider::new()`] with a non-empty `Vec` of providers.
/// An empty vec is rejected with a config error.
///
/// Use [`FallbackProvider::with_retry()`] to enable per-provider retry
/// with exponential backoff on transient errors,
/// [`FallbackProvider::with_fallthrough()`] to choose what moves a request
/// on, and [`FallbackProvider::with_observer()`] to watch every hop.
pub struct FallbackProvider {
    providers: Vec<Box<dyn LlmProvider>>,
    display_name: String,
    combined_models: Vec<String>,
    retry_config: RetryConfig,
    policy: ResponsePolicy,
    /// Shared with whoever else holds the breaker or the metrics the observer
    /// writes to; the chain only reads through it.
    observer: Option<Arc<dyn FallbackObserver>>,
    /// How long a tier that refused on quota is passed over before it is
    /// asked again. See [`FallbackProvider::with_quota_cooldown`].
    quota_cooldown: Duration,
    /// Per tier: the instant until which it is passed over, set when it
    /// refused on quota. A `std` mutex: held for a read or a write of one
    /// slot, never across an await.
    cooldown_until: Mutex<Vec<Option<Instant>>>,
}

/// A refused account is asked again after this long, unless the chain says
/// otherwise. Long enough that a burst of turns does not pay a refusal each,
/// short enough that a window that reset is noticed within the quarter hour.
const DEFAULT_QUOTA_COOLDOWN: Duration = Duration::from_mins(15);

impl FallbackProvider {
    /// Create a fallback chain from a non-empty list of providers.
    ///
    /// No retries are attempted (equivalent to `max_retries = 0`), every
    /// error falls through, every `Ok` is accepted.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` if `providers` is empty.
    pub fn new(providers: Vec<Box<dyn LlmProvider>>) -> Result<Self, RunnerError> {
        Self::with_retry(providers, RetryConfig::default())
    }

    /// Create a fallback chain with per-provider retry configuration.
    ///
    /// When a provider returns a transient error, the request is retried up to
    /// `retry_config.max_retries` times with exponential backoff before moving
    /// to the next provider. Permanent errors skip retries immediately.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` if `providers` is empty.
    pub fn with_retry(
        providers: Vec<Box<dyn LlmProvider>>,
        retry_config: RetryConfig,
    ) -> Result<Self, RunnerError> {
        if providers.is_empty() {
            return Err(RunnerError::config(
                "FallbackProvider requires at least one provider",
            ));
        }

        let names: Vec<&str> = providers.iter().map(|p| p.name()).collect();
        let display_name = format!("Fallback ({})", names.join(", "));

        // Deduplicated union of all available models
        let mut combined_models = Vec::new();
        for provider in &providers {
            for model in provider.available_models() {
                if !combined_models.contains(model) {
                    combined_models.push(model.clone());
                }
            }
        }

        let tiers = providers.len();
        Ok(Self {
            providers,
            display_name,
            combined_models,
            retry_config,
            policy: ResponsePolicy::permissive(),
            observer: None,
            quota_cooldown: DEFAULT_QUOTA_COOLDOWN,
            cooldown_until: Mutex::new(vec![None; tiers]),
        })
    }

    /// How long a tier that answered with a quota refusal
    /// ([`ErrorKind::RateLimit`]) is passed over before it is asked again.
    ///
    /// A spent account refuses every turn until its window resets, and every
    /// refusal is a round-trip paid before the next tier is reached. With a
    /// cooldown the chain goes straight to the next tier for that long, then
    /// asks the refused one again — a reset is discovered by asking, which
    /// costs one refusal per cooldown rather than one per turn, and needs no
    /// parsing of a reset time out of vendor prose. The last tier is never
    /// passed over: a chain with nowhere else to go asks it and returns its
    /// answer. Default: 15 minutes.
    #[must_use]
    pub const fn with_quota_cooldown(mut self, cooldown: Duration) -> Self {
        self.quota_cooldown = cooldown;
        self
    }

    /// Whether `position` is being passed over after a quota refusal.
    fn in_cooldown(&self, position: usize) -> bool {
        let slots = self
            .cooldown_until
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        slots
            .get(position)
            .copied()
            .flatten()
            .is_some_and(|until| Instant::now() < until)
    }

    /// Note a quota refusal from `position`: pass it over for the cooldown.
    fn start_cooldown(&self, position: usize) {
        let mut slots = self
            .cooldown_until
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.get_mut(position) {
            *slot = Some(Instant::now() + self.quota_cooldown);
        }
    }

    /// A tier answered: whatever cooldown it carried is over.
    fn end_cooldown(&self, position: usize) {
        let mut slots = self
            .cooldown_until
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.get_mut(position) {
            *slot = None;
        }
    }

    /// Choose what moves a request from one tier to the next.
    #[must_use]
    pub const fn with_fallthrough(mut self, policy: ResponsePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Watch every hop, and veto a tier before it is asked.
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<dyn FallbackObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Compute the backoff delay for a given attempt (0-indexed)
    fn backoff_delay(&self, attempt: u32) -> Duration {
        let delay = self
            .retry_config
            .base_delay
            .saturating_mul(2u32.saturating_pow(attempt));
        delay.min(self.retry_config.max_delay)
    }

    fn tier(&self, position: usize) -> Tier<'_> {
        Tier {
            position,
            provider: self.providers[position].as_ref(),
        }
    }

    /// Ask the observer whether this tier may be asked; `Try` when there is
    /// no observer.
    fn before_attempt(&self, tier: Tier<'_>) -> Attempt {
        self.observer
            .as_ref()
            .map_or(Attempt::Try, |o| o.before_attempt(tier))
    }

    fn on_fallthrough(&self, from: Tier<'_>, to: Tier<'_>, reason: FallthroughReason<'_>) {
        if let Some(observer) = &self.observer {
            observer.on_fallthrough(from, to, reason);
        }
    }

    fn on_success(&self, tier: Tier<'_>) {
        if let Some(observer) = &self.observer {
            observer.on_success(tier);
        }
    }

    fn on_exhausted(&self, last: Tier<'_>, error: &RunnerError) {
        if let Some(observer) = &self.observer {
            observer.on_exhausted(last, error);
        }
    }

    /// The request the next tier sees: the original, or a copy with `model`
    /// cleared once the policy asks for it. Made at the first hop and reused.
    fn forwarded<'r>(&self, original: &'r ChatRequest, held: &'r mut Option<ChatRequest>) {
        if self.policy.own_model_per_tier && held.is_none() {
            let mut cleared = original.clone();
            cleared.model = None;
            *held = Some(cleared);
        }
    }

    /// Ask one tier, retrying in place while the error is transient and the
    /// retry budget allows.
    async fn attempt<T>(
        &self,
        tier: Tier<'_>,
        request: &ChatRequest,
        call: TierMethod<'_, T>,
    ) -> Result<T, RunnerError> {
        let mut attempt = 0;
        loop {
            match call(tier.provider, request).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    let retryable =
                        err.kind.is_transient() && attempt < self.retry_config.max_retries;
                    if !retryable {
                        return Err(err);
                    }
                    let delay = self.backoff_delay(attempt);
                    #[allow(clippy::cast_possible_truncation)]
                    let delay_ms = delay.as_millis() as u64;
                    warn!(
                        provider = tier.provider.name(),
                        attempt,
                        error = %err,
                        delay_ms,
                        "fallback: transient error, retrying after backoff"
                    );
                    time::sleep(delay).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Walk the tiers. `reject` says whether an `Ok` counts as a failed tier
    /// (only consulted while a later tier exists); `call` is the provider
    /// method being driven.
    async fn drive<T>(
        &self,
        request: &ChatRequest,
        reject: impl Fn(&T) -> bool + Send + Sync,
        call: TierMethod<'_, T>,
    ) -> Result<T, RunnerError> {
        let last = self.providers.len().saturating_sub(1);
        let mut held: Option<ChatRequest> = None;

        for position in 0..self.providers.len() {
            let tier = self.tier(position);
            let has_successor = position < last;

            if has_successor {
                let verdict = if self.in_cooldown(position) {
                    Attempt::Skip("quota cooldown")
                } else {
                    self.before_attempt(tier)
                };
                if let Attempt::Skip(reason) = verdict {
                    self.on_fallthrough(
                        tier,
                        self.tier(position + 1),
                        FallthroughReason::Skipped(reason),
                    );
                    self.forwarded(request, &mut held);
                    continue;
                }
            }

            let current: &ChatRequest = held.as_ref().unwrap_or(request);
            // The borrow of `held` ends with the call; a later hop may rewrite it.
            let outcome = self.attempt(tier, current, call).await;

            match outcome {
                Ok(value) if has_successor && reject(&value) => {
                    warn!(
                        provider = tier.provider.name(),
                        "fallback: provider answered with nothing deliverable, trying next"
                    );
                    self.on_fallthrough(
                        tier,
                        self.tier(position + 1),
                        FallthroughReason::EmptyCompletion,
                    );
                    self.forwarded(request, &mut held);
                }
                Ok(value) => {
                    self.end_cooldown(position);
                    self.on_success(tier);
                    return Ok(value);
                }
                Err(err) if !self.policy.fall_through.allows(err.kind) => {
                    return Err(err);
                }
                Err(err) if has_successor => {
                    if err.kind == ErrorKind::RateLimit {
                        self.start_cooldown(position);
                    }
                    warn!(
                        provider = tier.provider.name(),
                        error = %err,
                        "fallback: provider failed, trying next"
                    );
                    self.on_fallthrough(
                        tier,
                        self.tier(position + 1),
                        FallthroughReason::Error(&err),
                    );
                    self.forwarded(request, &mut held);
                }
                Err(err) => {
                    if err.kind == ErrorKind::RateLimit {
                        self.start_cooldown(position);
                    }
                    warn!(
                        provider = tier.provider.name(),
                        error = %err,
                        "fallback: last provider failed, chain exhausted"
                    );
                    self.on_exhausted(tier, &err);
                    return Err(err);
                }
            }
        }

        Err(RunnerError::internal("no providers configured"))
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        "fallback"
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.providers
            .iter()
            .fold(LlmCapabilities::empty(), |acc, p| acc | p.capabilities())
    }

    fn default_model(&self) -> &str {
        self.providers[0].default_model()
    }

    fn available_models(&self) -> &[String] {
        &self.combined_models
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let reject_empty = self.policy.reject_empty_completion;
        self.drive(
            request,
            move |response: &ChatResponse| reject_empty && is_empty_completion(response),
            &|provider, request| Box::pin(provider.complete(request)),
        )
        .await
    }

    /// An opened stream is a success; its bytes are not inspected for
    /// emptiness, so the empty-completion rule does not apply here.
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.drive(request, |_: &ChatStream| false, &|provider, request| {
            Box::pin(provider.complete_stream(request))
        })
        .await
    }

    /// `Ok(true)` as soon as any tier is healthy; otherwise the last tier's
    /// outcome, verbatim — an `Err` there keeps its diagnostic.
    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut last = Ok(false);
        for provider in &self.providers {
            let outcome = provider.health_check().await;
            if matches!(outcome, Ok(true)) {
                return Ok(true);
            }
            last = outcome;
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities,
        LlmProvider, RunnerError,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    struct TestProvider {
        provider_name: &'static str,
        display: &'static str,
        caps: LlmCapabilities,
        models: Vec<String>,
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
        health: Result<bool, RunnerError>,
    }

    impl TestProvider {
        fn ok(name: &'static str, content: &str) -> Self {
            Self {
                provider_name: name,
                display: name,
                caps: LlmCapabilities::text_only(),
                models: vec![format!("{name}-model")],
                responses: Mutex::new(vec![Ok(ChatResponse {
                    content: content.to_owned(),
                    model: format!("{name}-model"),
                    usage: None,
                    finish_reason: Some("stop".to_owned()),
                    warnings: None,
                    tool_calls: None,
                })]),
                call_count: AtomicU32::new(0),
                health: Ok(true),
            }
        }

        fn failing(name: &'static str) -> Self {
            Self::failing_with_kind(name, ErrorKind::ExternalService)
        }

        fn failing_with_kind(name: &'static str, kind: ErrorKind) -> Self {
            let err = RunnerError {
                kind,
                message: format!("{name}: down"),
            };
            Self {
                provider_name: name,
                display: name,
                caps: LlmCapabilities::FUNCTION_CALLING,
                models: vec![format!("{name}-model")],
                responses: Mutex::new(vec![Err(err)]),
                call_count: AtomicU32::new(0),
                health: Ok(false),
            }
        }

        /// A tier whose health probe itself errors — a rejected credential,
        /// a dead endpoint — rather than answering `false`.
        fn health_erroring(name: &'static str, err: RunnerError) -> Self {
            Self {
                provider_name: name,
                display: name,
                caps: LlmCapabilities::text_only(),
                models: vec![format!("{name}-model")],
                responses: Mutex::new(vec![]),
                call_count: AtomicU32::new(0),
                health: Err(err),
            }
        }

        fn with_responses(
            name: &'static str,
            responses: Vec<Result<ChatResponse, RunnerError>>,
        ) -> Self {
            Self {
                provider_name: name,
                display: name,
                caps: LlmCapabilities::text_only(),
                models: vec![format!("{name}-model")],
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
                health: Ok(true),
            }
        }
    }

    /// A tier that has spent its quota cannot serve this request; the next
    /// tier, on its own account, can. Under the strict policy the chain moves
    /// on instead of handing the athlete the first tier's quota refusal.
    #[tokio::test]
    async fn strict_policy_falls_through_on_a_tiers_rate_limit() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing_with_kind(
                "quota",
                ErrorKind::RateLimit,
            )),
            Box::new(TestProvider::ok("second", "served by second")),
        ];
        let fallback = FallbackProvider::new(providers)
            .expect("non-empty") // Safe: test assertion
            .with_fallthrough(ResponsePolicy::strict());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        let response = fallback
            .complete(&request)
            .await
            .expect("second tier answers"); // Safe: test assertion
        assert_eq!(response.content, "served by second");
    }

    /// The request's own fault still propagates under the strict policy: the
    /// next tier would only produce its own version of the same rejection.
    #[tokio::test]
    async fn strict_policy_propagates_an_invalid_request() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing_with_kind(
                "first",
                ErrorKind::InvalidRequest,
            )),
            Box::new(TestProvider::ok("second", "never asked")),
        ];
        let fallback = FallbackProvider::new(providers)
            .expect("non-empty") // Safe: test assertion
            .with_fallthrough(ResponsePolicy::strict());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        let err = fallback.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert_eq!(err.message, "first: down");
    }

    fn make_response(content: &str) -> ChatResponse {
        ChatResponse {
            content: content.to_owned(),
            model: "test-model".to_owned(),
            usage: None,
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
            self.display
        }
        fn capabilities(&self) -> LlmCapabilities {
            self.caps
        }
        fn default_model(&self) -> &str {
            &self.models[0]
        }
        fn available_models(&self) -> &[String] {
            &self.models
        }
        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock"); // Safe: test assertion
            if responses.is_empty() {
                Err(RunnerError::internal("no more responses"))
            } else {
                responses.remove(0)
            }
        }
        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock"); // Safe: test assertion
            if responses.is_empty() {
                Err(RunnerError::internal("no more stream responses"))
            } else {
                match responses.remove(0) {
                    Ok(_) => Err(RunnerError::internal(
                        "use complete() for ok responses in test",
                    )),
                    Err(e) => Err(e),
                }
            }
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            self.health.clone()
        }
    }

    // ========================================================================
    // Original tests (unchanged behavior with max_retries=0)
    // ========================================================================

    #[tokio::test]
    async fn single_provider_passthrough() {
        let providers: Vec<Box<dyn LlmProvider>> =
            vec![Box::new(TestProvider::ok("claude", "hello"))];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback.complete(&request).await.expect("should succeed"); // Safe: test assertion
        assert_eq!(response.content, "hello");
    }

    #[tokio::test]
    async fn first_fails_second_succeeds() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("primary")),
            Box::new(TestProvider::ok("secondary", "fallback response")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback
            .complete(&request)
            .await
            .expect("second should work"); // Safe: test assertion
        assert_eq!(response.content, "fallback response");
    }

    #[tokio::test]
    async fn all_fail_returns_last_error() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("first")),
            Box::new(TestProvider::failing("second")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let err = fallback.complete(&request).await.unwrap_err();
        assert!(err.message.contains("second"));
    }

    #[tokio::test]
    async fn health_or_logic() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("unhealthy")), // healthy=false
            Box::new(TestProvider::ok("healthy", "ok")),  // healthy=true
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let healthy = fallback.health_check().await.expect("health check"); // Safe: test assertion
        assert!(healthy);
    }

    #[tokio::test]
    async fn health_all_down() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("a")),
            Box::new(TestProvider::failing("b")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let healthy = fallback.health_check().await.expect("health check"); // Safe: test assertion
        assert!(!healthy);
    }

    /// When no tier is healthy, the last tier's own outcome comes back
    /// verbatim — an `Err` keeps the diagnostic a caller's health probe logs,
    /// instead of collapsing into an `Ok(false)` that says nothing.
    #[tokio::test]
    async fn health_check_returns_the_last_tiers_error_when_none_is_healthy() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("a")),
            Box::new(TestProvider::health_erroring(
                "b",
                RunnerError::auth_failure("b: invalid api token"),
            )),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let err = fallback
            .health_check()
            .await
            .expect_err("the last tier's probe error must propagate");
        assert_eq!(err.kind, ErrorKind::AuthFailure);
        assert!(err.message.contains("invalid api token"), "{err}");
    }

    /// The rule is "the last tier's outcome", not "any error": an earlier
    /// tier's probe error is superseded by a later `Ok(false)`.
    #[tokio::test]
    async fn health_check_last_tier_unhealthy_outranks_an_earlier_error() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::health_erroring(
                "a",
                RunnerError::timeout("a: probe timed out"),
            )),
            Box::new(TestProvider::failing("b")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let healthy = fallback
            .health_check()
            .await
            .expect("Ok(false), not the earlier error"); // Safe: test assertion
        assert!(!healthy);
    }

    /// A healthy tier anywhere short-circuits, even after an erroring probe.
    #[tokio::test]
    async fn health_check_any_healthy_tier_wins_over_an_error() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::health_erroring(
                "a",
                RunnerError::auth_failure("a: down"),
            )),
            Box::new(TestProvider::ok("b", "ok")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        assert!(fallback.health_check().await.expect("healthy")); // Safe: test assertion
    }

    #[test]
    fn capabilities_union() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::ok("a", "ok")), // text_only = STREAMING | SYSTEM_MESSAGES
            Box::new(TestProvider::failing("b")),  // FUNCTION_CALLING
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let caps = fallback.capabilities();
        assert!(caps.supports_streaming());
        assert!(caps.supports_system_messages());
        assert!(caps.supports_function_calling());
    }

    #[test]
    fn empty_vec_rejected() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![];
        let result = FallbackProvider::new(providers);
        assert!(result.is_err());
    }

    #[test]
    fn available_models_deduplicated() {
        // Both providers share "shared-model"
        let a = TestProvider {
            provider_name: "a",
            display: "A",
            caps: LlmCapabilities::text_only(),
            models: vec!["shared-model".to_owned(), "a-only".to_owned()],
            responses: Mutex::new(vec![]),
            call_count: AtomicU32::new(0),
            health: Ok(true),
        };
        let b = TestProvider {
            provider_name: "b",
            display: "B",
            caps: LlmCapabilities::text_only(),
            models: vec!["shared-model".to_owned(), "b-only".to_owned()],
            responses: Mutex::new(vec![]),
            call_count: AtomicU32::new(0),
            health: Ok(true),
        };

        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(a), Box::new(b)];
        let fallback = FallbackProvider::new(providers).expect("non-empty"); // Safe: test assertion

        let models = fallback.available_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"shared-model".to_owned()));
        assert!(models.contains(&"a-only".to_owned()));
        assert!(models.contains(&"b-only".to_owned()));
    }

    // ========================================================================
    // Retry tests
    // ========================================================================

    #[tokio::test]
    async fn retry_on_transient_then_succeeds() {
        let provider = TestProvider::with_responses(
            "alpha",
            vec![
                Err(RunnerError::timeout("timed out")),
                Ok(make_response("recovered")),
            ],
        );
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(provider)];
        let retry = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback.complete(&request).await.expect("should recover"); // Safe: test assertion
        assert_eq!(response.content, "recovered");
    }

    #[tokio::test]
    async fn no_retry_on_permanent_error() {
        let provider = TestProvider::with_responses(
            "alpha",
            vec![
                Err(RunnerError::config("bad config")),
                Ok(make_response("should not reach")),
            ],
        );
        let backup = TestProvider::ok("beta", "from backup");
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(provider), Box::new(backup)];
        let retry = RetryConfig {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback
            .complete(&request)
            .await
            .expect("backup should work"); // Safe: test assertion
        assert_eq!(response.content, "from backup");
    }

    #[tokio::test]
    async fn retry_exhausts_then_next_provider() {
        let primary = TestProvider::with_responses(
            "primary",
            vec![
                Err(RunnerError::timeout("t1")),
                Err(RunnerError::timeout("t2")),
                Err(RunnerError::timeout("t3")),
            ],
        );
        let secondary = TestProvider::ok("secondary", "secondary response");
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(primary), Box::new(secondary)];
        let retry = RetryConfig {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback
            .complete(&request)
            .await
            .expect("secondary should work"); // Safe: test assertion
        assert_eq!(response.content, "secondary response");
    }

    #[tokio::test]
    async fn zero_retries_matches_original_behavior() {
        let provider = TestProvider::with_responses(
            "alpha",
            vec![
                Err(RunnerError::timeout("t1")),
                Ok(make_response("should not reach")),
            ],
        );
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(provider)];
        let retry = RetryConfig {
            max_retries: 0,
            ..RetryConfig::default()
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let err = fallback.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Timeout);
    }

    #[test]
    fn backoff_respects_max_delay() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(TestProvider::ok("a", "ok"))];
        let retry = RetryConfig {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion

        // attempt 0: 100 * 2^0 = 100ms
        assert_eq!(fallback.backoff_delay(0), Duration::from_millis(100));
        // attempt 1: 100 * 2^1 = 200ms
        assert_eq!(fallback.backoff_delay(1), Duration::from_millis(200));
        // attempt 2: 100 * 2^2 = 400ms
        assert_eq!(fallback.backoff_delay(2), Duration::from_millis(400));
        // attempt 3: 100 * 2^3 = 800ms -> capped to 500ms
        assert_eq!(fallback.backoff_delay(3), Duration::from_millis(500));
    }

    #[tokio::test]
    async fn stream_retry_on_transient() {
        let provider = TestProvider::with_responses(
            "alpha",
            vec![
                Err(RunnerError::external_service("alpha", "503")),
                Err(RunnerError::external_service("alpha", "503 again")),
            ],
        );
        let backup =
            TestProvider::with_responses("beta", vec![Err(RunnerError::config("bad config"))]);
        let providers: Vec<Box<dyn LlmProvider>> = vec![Box::new(provider), Box::new(backup)];
        let retry = RetryConfig {
            max_retries: 1,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
        };
        let fallback = FallbackProvider::with_retry(providers, retry).expect("non-empty"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        // alpha: attempt 0 fails (transient), attempt 1 fails (transient, exhausted) -> next
        // beta: attempt 0 fails (permanent, no retry) -> error
        match fallback.complete_stream(&request).await {
            Err(err) => assert_eq!(err.kind, ErrorKind::Config),
            Ok(_) => unreachable!("expected error"), // Safe: test assertion
        }
    }
}
