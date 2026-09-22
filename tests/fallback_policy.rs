// ABOUTME: ResponsePolicy on FallbackProvider — what moves a request to the next tier, and what does not
// ABOUTME: An empty completion and a provider fault fall through; a malformed request and a tool call do not
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A chain that matches `Ok(response)` unconditionally reports an answer that
//! carries nothing — a blank turn with no tool call — as a success, and the
//! working tier behind it is never asked. A chain that falls through on every
//! error reroutes a malformed request too, so the caller sees the second
//! provider's version of the same rejection instead of the real diagnostic.
//! [`ResponsePolicy::strict`] fixes both; [`ResponsePolicy::permissive`] keeps
//! the original behaviour, and both are pinned here so neither drifts.
//!
//! The negative cases are load-bearing: an empty answer that carries a tool
//! call is an ordinary mid-loop turn, and rerouting it would spend a second
//! completion on every tool-using turn in every conversation.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;

use async_trait::async_trait;
use serde_json::json;

use embacle::fallback::{is_empty_completion, FallbackProvider, ResponsePolicy};
use embacle::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    RunnerError, ToolCallRequest,
};

/// What a scripted tier does when asked to complete.
enum Scripted {
    /// Answer with this text and no tool calls.
    Says(&'static str),
    /// Answer with no content but a `get_activities` call — an ordinary
    /// mid-loop turn.
    CallsATool,
    /// Answer `Ok` with nothing at all: no content, no tool calls.
    Empty,
    /// Fail with an error the policy classifies as the provider's fault.
    ProviderFault,
    /// Fail with a deterministic rejection the policy refuses to reroute.
    InvalidRequest,
    /// Refuse on quota: the account is spent until its window resets.
    QuotaRefusal,
}

/// A tier whose every call is decided in advance and counted.
struct Fake {
    label: &'static str,
    script: Scripted,
    calls: Arc<AtomicUsize>,
    models: Vec<String>,
}

impl Fake {
    fn scripted(label: &'static str, script: Scripted) -> (Box<dyn LlmProvider>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = Box::new(Self {
            label,
            script,
            calls: Arc::clone(&calls),
            models: vec!["test-model".to_owned()],
        });
        (provider, calls)
    }
}

fn reply(content: &str, tool_calls: Option<Vec<ToolCallRequest>>) -> ChatResponse {
    ChatResponse {
        content: content.to_owned(),
        model: "test-model".to_owned(),
        usage: None,
        finish_reason: Some("stop".to_owned()),
        warnings: None,
        tool_calls,
    }
}

fn a_tool_call() -> ToolCallRequest {
    ToolCallRequest {
        id: "call_1".to_owned(),
        function_name: "get_activities".to_owned(),
        arguments: json!({"limit": 10}),
    }
}

#[async_trait]
impl LlmProvider for Fake {
    fn name(&self) -> &'static str {
        self.label
    }

    fn display_name(&self) -> &str {
        self.label
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::FUNCTION_CALLING
    }

    fn default_model(&self) -> &str {
        "test-model"
    }

    fn available_models(&self) -> &[String] {
        &self.models
    }

    async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.script {
            Scripted::Says(text) => Ok(reply(text, None)),
            Scripted::CallsATool => Ok(reply("", Some(vec![a_tool_call()]))),
            Scripted::Empty => Ok(reply("", None)),
            Scripted::ProviderFault => Err(RunnerError::external_service(
                self.label,
                "upstream returned 503",
            )),
            Scripted::InvalidRequest => Err(RunnerError::invalid_request(
                "the request body was malformed",
            )),
            Scripted::QuotaRefusal => {
                Err(RunnerError::rate_limit(self.label, "usage limit reached"))
            }
        }
    }

    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::invalid_request(
            "the scripted tier answers only complete()",
        ))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

fn strict_chain(
    primary: Box<dyn LlmProvider>,
    secondary: Box<dyn LlmProvider>,
) -> FallbackProvider {
    FallbackProvider::new(vec![primary, secondary])
        .expect("two tiers")
        .with_fallthrough(ResponsePolicy::strict())
}

fn a_turn() -> ChatRequest {
    ChatRequest::new(vec![
        ChatMessage::system("coach persona"),
        ChatMessage::user("how did my week go?"),
    ])
    .with_model("primary-only-model")
    .with_tools(vec![])
}

// ============================================================================
// ResponsePolicy::strict() — the five chain decisions
// ============================================================================

/// A provider fault is answered by the next tier.
#[tokio::test]
async fn a_provider_fault_falls_through_to_the_secondary() {
    let (primary, primary_calls) = Fake::scripted("primary", Scripted::ProviderFault);
    let (secondary, secondary_calls) =
        Fake::scripted("secondary", Scripted::Says("42 km this week."));

    let response = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect("a provider fault must be answered by the secondary");

    assert_eq!(response.content, "42 km this week.");
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        secondary_calls.load(Ordering::SeqCst),
        1,
        "the secondary is consulted exactly once"
    );
}

/// A 200 carrying neither content nor a tool call is a lost turn, not an
/// answer; the next tier is asked.
#[tokio::test]
async fn an_empty_completion_falls_through_to_the_secondary() {
    let (primary, primary_calls) = Fake::scripted("primary", Scripted::Empty);
    let (secondary, secondary_calls) =
        Fake::scripted("secondary", Scripted::Says("Tu as couru 42 km ce mois-ci."));

    let response = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect("an empty primary completion must be reissued against the secondary");

    assert_eq!(response.content, "Tu as couru 42 km ce mois-ci.");
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 1);
}

/// The critical negative case: empty content WITH a tool call is how a model
/// asks for a tool.
#[tokio::test]
async fn an_empty_answer_carrying_a_tool_call_stays_with_the_primary() {
    let (primary, primary_calls) = Fake::scripted("primary", Scripted::CallsATool);
    let (secondary, secondary_calls) =
        Fake::scripted("secondary", Scripted::Says("should not be used"));

    let response = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect("a tool-call turn is a success");

    let calls = response
        .tool_calls
        .expect("the primary's tool calls must survive the chain");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function_name, "get_activities");
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        secondary_calls.load(Ordering::SeqCst),
        0,
        "falling back on a tool-call turn would spend a completion on the common case"
    );
}

/// A deterministic failure surfaces as itself; the secondary is never woken.
#[tokio::test]
async fn an_invalid_request_is_not_rerouted() {
    let (primary, primary_calls) = Fake::scripted("primary", Scripted::InvalidRequest);
    let (secondary, secondary_calls) =
        Fake::scripted("secondary", Scripted::Says("should not be used"));

    let error = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect_err("an invalid-request error does not fall through");

    assert_eq!(error.kind, ErrorKind::InvalidRequest);
    assert!(
        error.message.contains("malformed"),
        "the caller must see the primary's own diagnostic, got: {error}"
    );
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 0);
}

/// A healthy primary answers, and the secondary is never woken.
#[tokio::test]
async fn a_primary_that_answers_is_the_answer() {
    let (primary, primary_calls) = Fake::scripted("primary", Scripted::Says("Rest today."));
    let (secondary, secondary_calls) =
        Fake::scripted("secondary", Scripted::Says("should not be used"));

    let response = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect("a healthy primary answers");

    assert_eq!(response.content, "Rest today.");
    assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 0);
}

/// Every request-side kind propagates under `ProviderFault`; every provider
/// fault moves on. Pinned kind by kind so the classification cannot drift
/// from the predicate it is built on.
#[tokio::test]
async fn strict_fall_through_follows_is_provider_fault_for_every_kind() {
    let kinds = [
        ErrorKind::Internal,
        ErrorKind::ExternalService,
        ErrorKind::Timeout,
        ErrorKind::BinaryNotFound,
        ErrorKind::AuthFailure,
        ErrorKind::Config,
        ErrorKind::Guardrail,
        ErrorKind::ContextLength,
        ErrorKind::ModelUnavailable,
        ErrorKind::RateLimit,
        ErrorKind::InvalidRequest,
    ];
    for kind in kinds {
        let primary = Box::new(Erring { kind });
        let (secondary, secondary_calls) = Fake::scripted("secondary", Scripted::Says("answered"));
        let outcome = strict_chain(primary, secondary).complete(&a_turn()).await;

        if kind.is_provider_fault() {
            assert_eq!(
                outcome
                    .expect("a provider fault reaches the secondary")
                    .content,
                "answered",
                "{kind:?}"
            );
            assert_eq!(secondary_calls.load(Ordering::SeqCst), 1, "{kind:?}");
        } else {
            let err = outcome.expect_err("a request-side error propagates");
            assert_eq!(err.kind, kind);
            assert_eq!(
                secondary_calls.load(Ordering::SeqCst),
                0,
                "{kind:?} must not spend a second tier"
            );
        }
    }
}

/// A tier that fails with exactly one kind.
struct Erring {
    kind: ErrorKind,
}

#[async_trait]
impl LlmProvider for Erring {
    fn name(&self) -> &'static str {
        "erring"
    }
    fn display_name(&self) -> &str {
        "erring"
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::text_only()
    }
    fn default_model(&self) -> &str {
        "test-model"
    }
    fn available_models(&self) -> &[String] {
        &[]
    }
    async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        Err(RunnerError {
            kind: self.kind,
            message: format!("{:?}", self.kind),
        })
    }
    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError {
            kind: self.kind,
            message: format!("{:?}", self.kind),
        })
    }
    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

// ============================================================================
// ResponsePolicy::permissive() — the original chain, pinned as the contrast
// ============================================================================

/// Without the policy, an empty `Ok` is returned as the answer and the
/// secondary is never asked. This is the behaviour `strict()` exists to
/// replace; it stays available, and it stays pinned.
#[tokio::test]
async fn permissive_returns_an_empty_completion_as_success() {
    let (primary, _) = Fake::scripted("primary", Scripted::Empty);
    let (secondary, secondary_calls) = Fake::scripted("secondary", Scripted::Says("never asked"));

    let chain = FallbackProvider::new(vec![primary, secondary])
        .expect("two tiers")
        .with_fallthrough(ResponsePolicy::permissive());
    let response = chain.complete(&a_turn()).await.expect("Ok, however empty");

    assert!(response.content.is_empty());
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 0);
}

/// Without the policy, every error moves on — a malformed request included.
#[tokio::test]
async fn permissive_falls_through_on_every_error() {
    let (primary, _) = Fake::scripted("primary", Scripted::InvalidRequest);
    let (secondary, secondary_calls) = Fake::scripted("secondary", Scripted::Says("rerouted"));

    let chain = FallbackProvider::new(vec![primary, secondary])
        .expect("two tiers")
        .with_fallthrough(ResponsePolicy::permissive());
    let response = chain
        .complete(&a_turn())
        .await
        .expect("the secondary answers");

    assert_eq!(response.content, "rerouted");
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 1);
}

/// `Default` is `permissive()`: a chain built without `with_fallthrough`
/// behaves exactly as one built with it.
#[test]
fn the_default_policy_is_permissive() {
    assert_eq!(ResponsePolicy::default(), ResponsePolicy::permissive());
    assert_ne!(ResponsePolicy::default(), ResponsePolicy::strict());
}

/// An empty answer from the LAST tier is still returned as `Ok`: there is
/// nobody left to ask, and an empty string is what the caller can act on.
#[tokio::test]
async fn an_empty_completion_from_the_last_tier_is_returned() {
    let (primary, _) = Fake::scripted("primary", Scripted::ProviderFault);
    let (secondary, secondary_calls) = Fake::scripted("secondary", Scripted::Empty);

    let response = strict_chain(primary, secondary)
        .complete(&a_turn())
        .await
        .expect("the last tier's empty answer is still an answer");

    assert!(response.content.is_empty());
    assert_eq!(secondary_calls.load(Ordering::SeqCst), 1);
}

// ============================================================================
// is_empty_completion — the classification, case by case
// ============================================================================

/// The live shape: `finish_reason: "stop"`, no content, no tool calls.
#[test]
fn a_bare_empty_turn_is_empty() {
    assert!(
        is_empty_completion(&reply("", None)),
        "an empty turn with no tool calls is the failure this exists to catch"
    );
}

/// Whitespace is not content. Every surface trims before rendering, so a
/// reply of three spaces reaches the reader as nothing at all.
#[test]
fn whitespace_only_content_is_empty() {
    assert!(is_empty_completion(&reply("   \n\t  ", None)));
}

/// An empty `tool_calls` vec is the same as none. Providers differ on which
/// they send, and a `Some(vec![])` reading as "has tool calls" would restore
/// the original bug for whichever provider serialises it that way.
#[test]
fn an_empty_tool_call_vec_is_still_empty() {
    assert!(is_empty_completion(&reply("", Some(vec![]))));
}

/// A model that calls a tool instead of speaking returns empty content by
/// design; treating that as a failure would fall back on every tool-using turn.
#[test]
fn empty_content_with_tool_calls_is_not_empty() {
    assert!(
        !is_empty_completion(&reply("", Some(vec![a_tool_call()]))),
        "a tool-call turn is an ordinary mid-loop turn, not a lost one"
    );
}

/// Whitespace content alongside a tool call is still a tool-call turn.
#[test]
fn whitespace_content_with_tool_calls_is_not_empty() {
    assert!(!is_empty_completion(&reply(
        "  ",
        Some(vec![a_tool_call()])
    )));
}

/// An ordinary answer is never empty — guards against a classifier that
/// returns true for everything, which would route all traffic to the secondary.
#[test]
fn a_real_reply_is_not_empty() {
    assert!(!is_empty_completion(&reply(
        "Tu as couru 42 km ce mois-ci.",
        None
    )));
}

// ---------------------------------------------------------------------------
// Quota cooldown: a refused account is passed over, then asked again
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_tier_that_refused_on_quota_is_passed_over_until_its_cooldown_ends() {
    let (spent, spent_calls) = Fake::scripted("claude-code", Scripted::QuotaRefusal);
    let (second, second_calls) = Fake::scripted("claude-code#2", Scripted::Says("pong"));
    let chain = FallbackProvider::new(vec![spent, second])
        .unwrap() // Safe: test setup
        .with_fallthrough(ResponsePolicy::strict())
        .with_quota_cooldown(Duration::from_millis(200));
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);

    // Turn 1: the spent account refuses, the second answers.
    let first = chain.complete(&request).await.unwrap(); // Safe: test assertion
    assert_eq!(first.content, "pong");
    assert_eq!(spent_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);

    // Turns 2 and 3, inside the cooldown: the spent account is not asked.
    for _ in 0..2 {
        chain.complete(&request).await.unwrap(); // Safe: test assertion
    }
    assert_eq!(
        spent_calls.load(Ordering::SeqCst),
        1,
        "a refused account is not asked again on every turn"
    );
    assert_eq!(second_calls.load(Ordering::SeqCst), 3);

    // After the cooldown the spent account is asked again — that is how a
    // reset window is discovered.
    sleep(Duration::from_millis(250)).await;
    chain.complete(&request).await.unwrap(); // Safe: test assertion
    assert_eq!(spent_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn the_last_tier_is_asked_even_in_cooldown() {
    let (only, calls) = Fake::scripted("claude-code", Scripted::QuotaRefusal);
    let chain = FallbackProvider::new(vec![only])
        .unwrap() // Safe: test setup
        .with_fallthrough(ResponsePolicy::strict())
        .with_quota_cooldown(Duration::from_hours(1));
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);

    for _ in 0..2 {
        let err = chain.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "nowhere else to go: it is asked"
    );
}

#[tokio::test]
async fn a_provider_fault_is_not_a_cooldown() {
    let (flaky, flaky_calls) = Fake::scripted("claude-code", Scripted::ProviderFault);
    let (second, _) = Fake::scripted("gemini", Scripted::Says("pong"));
    let chain = FallbackProvider::new(vec![flaky, second])
        .unwrap() // Safe: test setup
        .with_fallthrough(ResponsePolicy::strict())
        .with_quota_cooldown(Duration::from_hours(1));
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);

    for _ in 0..3 {
        chain.complete(&request).await.unwrap(); // Safe: test assertion
    }
    assert_eq!(
        flaky_calls.load(Ordering::SeqCst),
        3,
        "a fault may clear on the next turn; only a quota refusal is held off"
    );
}
