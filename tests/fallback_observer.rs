// ABOUTME: FallbackObserver on FallbackProvider — every hop is reported, and a tier can be vetoed before it is asked
// ABOUTME: Pins the callback set, the position-0 veto, the per-tier model reset, and same-task invocation
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A circuit breaker that lives outside the chain needs two things from it:
//! to be told what happened at each tier, and to say "not this one" before
//! the primary is asked. Both are the [`FallbackObserver`] contract, and the
//! contract includes *where* the callbacks run — synchronously on the calling
//! task, so whatever span the caller opened is live inside them.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use async_trait::async_trait;
use tokio_stream::{self as stream, StreamExt};

use embacle::fallback::{
    Attempt, FallbackObserver, FallbackProvider, FallthroughReason, ResponsePolicy, Tier,
};
use embacle::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    RunnerError, StreamChunk,
};

/// One recorded callback, flattened to owned data.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    BeforeAttempt {
        position: usize,
    },
    Fallthrough {
        from: usize,
        to: usize,
        reason: String,
    },
    Success {
        position: usize,
    },
    Exhausted {
        position: usize,
        kind: ErrorKind,
    },
}

/// Records every callback, the thread it ran on, and vetoes the positions it
/// was told to.
struct Recording {
    events: Mutex<Vec<Event>>,
    threads: Mutex<Vec<ThreadId>>,
    veto: Vec<usize>,
}

impl Recording {
    fn new(veto: Vec<usize>) -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
            threads: Mutex::new(Vec::new()),
            veto,
        })
    }

    fn record(&self, event: Event) {
        self.events.lock().unwrap().push(event);
        self.threads.lock().unwrap().push(thread::current().id());
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }
}

impl FallbackObserver for Recording {
    fn before_attempt(&self, tier: Tier<'_>) -> Attempt {
        self.record(Event::BeforeAttempt {
            position: tier.position,
        });
        if self.veto.contains(&tier.position) {
            Attempt::Skip("preemptive_guard")
        } else {
            Attempt::Try
        }
    }

    fn on_fallthrough(&self, from: Tier<'_>, to: Tier<'_>, reason: FallthroughReason<'_>) {
        let reason = match reason {
            FallthroughReason::Skipped(r) => format!("skipped:{r}"),
            FallthroughReason::EmptyCompletion => "empty_completion".to_owned(),
            FallthroughReason::Error(e) => format!("error:{:?}", e.kind),
        };
        self.record(Event::Fallthrough {
            from: from.position,
            to: to.position,
            reason,
        });
    }

    fn on_success(&self, tier: Tier<'_>) {
        self.record(Event::Success {
            position: tier.position,
        });
    }

    fn on_exhausted(&self, last: Tier<'_>, error: &RunnerError) {
        self.record(Event::Exhausted {
            position: last.position,
            kind: error.kind,
        });
    }
}

/// What a scripted tier does when asked.
#[derive(Clone, Copy)]
enum Script {
    Says(&'static str),
    Empty,
    Fails(ErrorKind),
}

/// A tier that answers per its script and remembers the model each request
/// carried.
struct Fake {
    label: &'static str,
    script: Script,
    seen_models: Mutex<Vec<Option<String>>>,
    calls: Mutex<usize>,
}

impl Fake {
    fn new(label: &'static str, script: Script) -> Arc<Self> {
        Arc::new(Self {
            label,
            script,
            seen_models: Mutex::new(Vec::new()),
            calls: Mutex::new(0),
        })
    }

    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }

    fn seen_models(&self) -> Vec<Option<String>> {
        self.seen_models.lock().unwrap().clone()
    }

    fn note(&self, request: &ChatRequest) {
        *self.calls.lock().unwrap() += 1;
        self.seen_models.lock().unwrap().push(request.model.clone());
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
        LlmCapabilities::text_only()
    }
    fn default_model(&self) -> &str {
        "test-model"
    }
    fn available_models(&self) -> &[String] {
        &[]
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        self.note(request);
        match self.script {
            Script::Says(text) => Ok(ChatResponse {
                content: text.to_owned(),
                model: "test-model".to_owned(),
                usage: None,
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
            Script::Empty => Ok(ChatResponse {
                content: String::new(),
                model: "test-model".to_owned(),
                usage: None,
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
            Script::Fails(kind) => Err(RunnerError {
                kind,
                message: format!("{}: {kind:?}", self.label),
            }),
        }
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.note(request);
        match self.script {
            Script::Says(text) => Ok(Box::pin(stream::iter(vec![Ok(StreamChunk {
                delta: text.to_owned(),
                is_final: true,
                finish_reason: Some("stop".to_owned()),
            })]))),
            Script::Empty => Ok(Box::pin(stream::empty())),
            Script::Fails(kind) => Err(RunnerError {
                kind,
                message: format!("{}: {kind:?}", self.label),
            }),
        }
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

/// A strict chain over shared tiers, so the test keeps a handle on each.
fn chain(tiers: &[Arc<Fake>], observer: Arc<Recording>) -> FallbackProvider {
    let boxed: Vec<Box<dyn LlmProvider>> = tiers
        .iter()
        .map(|t| Box::new(Arc::clone(t)) as Box<dyn LlmProvider>)
        .collect();
    FallbackProvider::new(boxed)
        .expect("tiers")
        .with_fallthrough(ResponsePolicy::strict())
        .with_observer(observer)
}

fn a_turn() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi")]).with_model("primary-only-model")
}

/// The veto: tier 0 is never called, the skip is reported as a fallthrough
/// from 0 to 1, and the forwarded request has its model cleared.
#[tokio::test]
async fn a_vetoed_primary_is_never_called_and_the_skip_is_reported() {
    let primary = Fake::new("primary", Script::Says("should not be asked"));
    let secondary = Fake::new("secondary", Script::Says("from the secondary"));
    let observer = Recording::new(vec![0]);

    let response = chain(
        &[Arc::clone(&primary), Arc::clone(&secondary)],
        Arc::clone(&observer),
    )
    .complete(&a_turn())
    .await
    .expect("the secondary answers");

    assert_eq!(response.content, "from the secondary");
    assert_eq!(primary.calls(), 0, "a vetoed tier is not asked");
    assert_eq!(
        observer.events(),
        vec![
            Event::BeforeAttempt { position: 0 },
            Event::Fallthrough {
                from: 0,
                to: 1,
                reason: "skipped:preemptive_guard".to_owned()
            },
            Event::Success { position: 1 },
        ]
    );
    assert_eq!(
        secondary.seen_models(),
        vec![None],
        "own_model_per_tier clears the model on the forwarded request"
    );
}

/// After a provider fault at tier 0, `on_success` names tier 1.
#[tokio::test]
async fn success_after_a_fault_names_the_tier_that_answered() {
    let primary = Fake::new("primary", Script::Fails(ErrorKind::ExternalService));
    let secondary = Fake::new("secondary", Script::Says("answered"));
    let observer = Recording::new(vec![]);

    chain(&[primary, secondary], Arc::clone(&observer))
        .complete(&a_turn())
        .await
        .expect("answered");

    assert_eq!(
        observer.events(),
        vec![
            Event::BeforeAttempt { position: 0 },
            Event::Fallthrough {
                from: 0,
                to: 1,
                reason: "error:ExternalService".to_owned()
            },
            Event::Success { position: 1 },
        ]
    );
}

/// An empty completion is its own reason, distinct from an error.
#[tokio::test]
async fn an_empty_completion_is_reported_as_such() {
    let primary = Fake::new("primary", Script::Empty);
    let secondary = Fake::new("secondary", Script::Says("answered"));
    let observer = Recording::new(vec![]);

    chain(&[primary, secondary], Arc::clone(&observer))
        .complete(&a_turn())
        .await
        .expect("answered");

    assert_eq!(
        observer.events()[1],
        Event::Fallthrough {
            from: 0,
            to: 1,
            reason: "empty_completion".to_owned()
        }
    );
}

/// When the last tier fails too, `on_exhausted` carries that error, and the
/// same error is what `complete()` returns.
#[tokio::test]
async fn exhaustion_reports_the_last_error() {
    let primary = Fake::new("primary", Script::Fails(ErrorKind::Timeout));
    let secondary = Fake::new("secondary", Script::Fails(ErrorKind::AuthFailure));
    let observer = Recording::new(vec![]);

    let err = chain(&[primary, secondary], Arc::clone(&observer))
        .complete(&a_turn())
        .await
        .expect_err("both tiers failed");

    assert_eq!(err.kind, ErrorKind::AuthFailure);
    assert!(err.message.starts_with("secondary:"), "{err}");
    assert_eq!(
        observer.events().last(),
        Some(&Event::Exhausted {
            position: 1,
            kind: ErrorKind::AuthFailure
        })
    );
}

/// A deterministic error is not the chain's business: it propagates with no
/// fallthrough and no exhaustion callback.
#[tokio::test]
async fn a_deterministic_error_fires_no_hop_callback_and_propagates() {
    let primary = Fake::new("primary", Script::Fails(ErrorKind::InvalidRequest));
    let secondary = Fake::new("secondary", Script::Says("should not be asked"));
    let observer = Recording::new(vec![]);

    let err = chain(&[primary, Arc::clone(&secondary)], Arc::clone(&observer))
        .complete(&a_turn())
        .await
        .expect_err("propagates");

    assert_eq!(err.kind, ErrorKind::InvalidRequest);
    assert_eq!(secondary.calls(), 0);
    assert_eq!(
        observer.events(),
        vec![Event::BeforeAttempt { position: 0 }],
        "only the pre-attempt consultation happened"
    );
}

/// An opened stream is a success; `on_success` fires as soon as it opens.
#[tokio::test]
async fn an_opened_stream_reports_success() {
    let primary = Fake::new("primary", Script::Fails(ErrorKind::ExternalService));
    let secondary = Fake::new("secondary", Script::Says("streamed"));
    let observer = Recording::new(vec![]);

    let mut stream = chain(&[primary, secondary], Arc::clone(&observer))
        .complete_stream(&a_turn())
        .await
        .expect("the secondary's stream opens");

    assert_eq!(
        observer.events().last(),
        Some(&Event::Success { position: 1 }),
        "success is reported on open, before a byte is read"
    );
    let first = stream.next().await.expect("one chunk").expect("ok");
    assert_eq!(first.delta, "streamed");
}

/// Three tiers, flat: the veto is asked for every tier that has a successor
/// and never for the last, and every hop is reported with its positions.
#[tokio::test]
async fn three_tiers_are_walked_flat_and_the_last_is_always_asked() {
    let first = Fake::new("first", Script::Fails(ErrorKind::ExternalService));
    let second = Fake::new("second", Script::Fails(ErrorKind::Timeout));
    let third = Fake::new("third", Script::Fails(ErrorKind::Internal));
    let observer = Recording::new(vec![2]);

    let err = chain(
        &[Arc::clone(&first), Arc::clone(&second), Arc::clone(&third)],
        Arc::clone(&observer),
    )
    .complete(&a_turn())
    .await
    .expect_err("all three failed");

    assert_eq!(err.kind, ErrorKind::Internal, "the last error wins");
    assert_eq!(
        third.calls(),
        1,
        "a veto on the last tier is never consulted"
    );
    assert_eq!(
        observer.events(),
        vec![
            Event::BeforeAttempt { position: 0 },
            Event::Fallthrough {
                from: 0,
                to: 1,
                reason: "error:ExternalService".to_owned()
            },
            Event::BeforeAttempt { position: 1 },
            Event::Fallthrough {
                from: 1,
                to: 2,
                reason: "error:Timeout".to_owned()
            },
            Event::Exhausted {
                position: 2,
                kind: ErrorKind::Internal
            },
        ]
    );
}

/// With the veto on the primary only, tier 1 is asked and tier 2 is not.
#[tokio::test]
async fn a_position_zero_veto_lands_on_the_second_tier_not_the_third() {
    let first = Fake::new("first", Script::Says("vetoed"));
    let second = Fake::new("second", Script::Says("answered by the second"));
    let third = Fake::new("third", Script::Says("never reached"));
    let observer = Recording::new(vec![0]);

    let response = chain(
        &[Arc::clone(&first), Arc::clone(&second), Arc::clone(&third)],
        observer,
    )
    .complete(&a_turn())
    .await
    .expect("answered");

    assert_eq!(response.content, "answered by the second");
    assert_eq!((first.calls(), second.calls(), third.calls()), (0, 1, 0));
}

/// The model reset happens once, at the first hop, and every later tier sees
/// the cleared request; the primary sees the original.
#[tokio::test]
async fn the_model_is_cleared_once_and_every_later_tier_sees_it_cleared() {
    let first = Fake::new("first", Script::Fails(ErrorKind::ExternalService));
    let second = Fake::new("second", Script::Fails(ErrorKind::Timeout));
    let third = Fake::new("third", Script::Says("answered"));

    chain(
        &[Arc::clone(&first), Arc::clone(&second), Arc::clone(&third)],
        Recording::new(vec![]),
    )
    .complete(&a_turn())
    .await
    .expect("answered");

    assert_eq!(
        first.seen_models(),
        vec![Some("primary-only-model".to_owned())]
    );
    assert_eq!(second.seen_models(), vec![None]);
    assert_eq!(third.seen_models(), vec![None]);
}

/// Under the permissive policy the request is forwarded untouched: the
/// secondary sees the primary's model.
#[tokio::test]
async fn the_permissive_policy_forwards_the_model_unchanged() {
    let first = Fake::new("first", Script::Fails(ErrorKind::ExternalService));
    let second = Fake::new("second", Script::Says("answered"));
    let boxed: Vec<Box<dyn LlmProvider>> =
        vec![Box::new(Arc::clone(&first)), Box::new(Arc::clone(&second))];

    FallbackProvider::new(boxed)
        .expect("tiers")
        .with_fallthrough(ResponsePolicy::permissive())
        .complete(&a_turn())
        .await
        .expect("answered");

    assert_eq!(
        second.seen_models(),
        vec![Some("primary-only-model".to_owned())]
    );
}

/// Every callback runs on the task that called `complete()` — the observer
/// sees the same thread on a current-thread runtime, so the caller's span is
/// live inside it.
#[tokio::test(flavor = "current_thread")]
async fn callbacks_run_on_the_calling_task() {
    let primary = Fake::new("primary", Script::Fails(ErrorKind::ExternalService));
    let secondary = Fake::new("secondary", Script::Fails(ErrorKind::Timeout));
    let observer = Recording::new(vec![]);

    let caller = thread::current().id();
    let _ = chain(&[primary, secondary], Arc::clone(&observer))
        .complete(&a_turn())
        .await;

    let threads = observer.threads.lock().unwrap().clone();
    assert_eq!(
        threads.len(),
        3,
        "before_attempt, on_fallthrough, on_exhausted"
    );
    assert!(
        threads.iter().all(|t| *t == caller),
        "a callback ran on another thread — the caller's span would be lost"
    );
}
