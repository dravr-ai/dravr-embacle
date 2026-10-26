// ABOUTME: complete_stream_with_tools: the default is complete() as one Done, and every decorator forwards the stream
// ABOUTME: Pins that a wrapper never silently degrades a streaming tier to its own complete() call
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio_stream::StreamExt;

use embacle::cache::{CacheConfig, CacheProvider};
use embacle::fallback::FallbackProvider;
use embacle::guardrail::GuardrailProvider;
use embacle::metrics::MetricsProvider;
use embacle::pool::provider_pool;
use embacle::quality_gate::{QualityGateProvider, QualityPolicy};
use embacle::router::{Backend, PreferInOrder, RouterProvider};
use embacle::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    RunnerError, TokenUsage, ToolCallRequest, ToolStream, ToolStreamEvent,
};

/// What every provider below answers: text, one tool call and usage
fn answer() -> ChatResponse {
    ChatResponse {
        content: "Let me look".to_owned(),
        model: "test-model".to_owned(),
        usage: Some(TokenUsage::new(12, 4, 16).with_cache(Some(8), Some(0))),
        finish_reason: Some("tool_calls".to_owned()),
        warnings: None,
        tool_calls: Some(vec![ToolCallRequest {
            id: "call_1".to_owned(),
            function_name: "get_activities".to_owned(),
            arguments: json!({"limit": 5}),
        }]),
    }
}

/// How a streaming test provider behaves
#[derive(Clone, Copy)]
enum Behaviour {
    /// Streams two deltas and the answer; `complete()` fails, so only a
    /// forwarded stream succeeds
    StreamsNatively,
    /// The stream refuses to open on quota
    StreamRateLimited,
}

/// A provider that overrides `complete_stream_with_tools`
struct StreamingProvider {
    name: &'static str,
    behaviour: Behaviour,
}

impl StreamingProvider {
    fn boxed(name: &'static str, behaviour: Behaviour) -> Box<dyn LlmProvider> {
        Box::new(Self { name, behaviour })
    }
}

#[async_trait]
impl LlmProvider for StreamingProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn display_name(&self) -> &str {
        self.name
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING | LlmCapabilities::FUNCTION_CALLING
    }

    fn default_model(&self) -> &str {
        "test-model"
    }

    fn available_models(&self) -> &[String] {
        &[]
    }

    async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        Err(RunnerError::external_service(
            self.name,
            "complete() was called instead of the stream",
        ))
    }

    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal("the text stream is not under test"))
    }

    async fn complete_stream_with_tools(
        &self,
        _request: &ChatRequest,
    ) -> Result<ToolStream, RunnerError> {
        match self.behaviour {
            Behaviour::StreamsNatively => Ok(Box::pin(tokio_stream::iter(vec![
                Ok(ToolStreamEvent::TextDelta("Let me".to_owned())),
                Ok(ToolStreamEvent::TextDelta(" look".to_owned())),
                Ok(ToolStreamEvent::Done(answer())),
            ]))),
            Behaviour::StreamRateLimited => Err(RunnerError::rate_limit(self.name, "quota spent")),
        }
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

/// A provider that keeps the trait's default `complete_stream_with_tools`;
/// its `complete()` answers when `answers` is set and fails otherwise
struct DefaultOnly {
    answers: bool,
}

#[async_trait]
impl LlmProvider for DefaultOnly {
    fn name(&self) -> &'static str {
        "default-only"
    }

    fn display_name(&self) -> &str {
        "default-only"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::FUNCTION_CALLING
    }

    fn default_model(&self) -> &str {
        "test-model"
    }

    fn available_models(&self) -> &[String] {
        &[]
    }

    async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        if self.answers {
            Ok(answer())
        } else {
            Err(RunnerError::external_service("default-only", "unreachable"))
        }
    }

    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal("the text stream is not under test"))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

fn request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("what did I ride this week?")])
}

/// Every item of a stream, in order
async fn drain(stream: ToolStream) -> Vec<Result<ToolStreamEvent, RunnerError>> {
    stream.collect().await
}

/// The deltas and the final response of a stream that succeeded
async fn deltas_and_done(stream: ToolStream) -> (Vec<String>, ChatResponse) {
    let mut deltas = Vec::new();
    let mut done = None;
    for item in drain(stream).await {
        match item.expect("no error item") {
            ToolStreamEvent::TextDelta(text) => deltas.push(text),
            ToolStreamEvent::Done(response) => {
                assert!(done.is_none(), "exactly one Done");
                done = Some(response);
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
    (deltas, done.expect("a Done closes the stream"))
}

#[tokio::test]
async fn the_default_yields_what_complete_returns_as_one_done() {
    let provider = DefaultOnly { answers: true };
    let expected = provider
        .complete(&request())
        .await
        .expect("complete answers");
    let (deltas, done) = deltas_and_done(
        provider
            .complete_stream_with_tools(&request())
            .await
            .expect("the stream opens"),
    )
    .await;
    assert!(deltas.is_empty(), "the default streams no text");
    assert_eq!(
        serde_json::to_value(&done).expect("serializes"),
        serde_json::to_value(&expected).expect("serializes"),
        "the Done is complete()'s response, tool calls and usage included"
    );
}

#[tokio::test]
async fn the_default_surfaces_a_complete_failure_as_the_call_error() {
    let Err(err) = DefaultOnly { answers: false }
        .complete_stream_with_tools(&request())
        .await
    else {
        panic!("a failing complete() must fail the call, not open a stream");
    };
    assert_eq!(err.kind, ErrorKind::ExternalService);
}

/// Each decorator over a natively streaming provider, by name
fn wrapped(inner: fn() -> Box<dyn LlmProvider>) -> Vec<(&'static str, Box<dyn LlmProvider>)> {
    vec![
        ("arc", Box::new(Arc::<dyn LlmProvider>::from(inner()))),
        (
            "fallback",
            Box::new(FallbackProvider::new(vec![inner()]).expect("chain")),
        ),
        (
            "pool",
            provider_pool(vec![inner()])
                .expect("pool")
                .pop()
                .expect("one tier"),
        ),
        ("metrics", Box::new(MetricsProvider::new(inner()))),
        (
            "guardrail",
            Box::new(GuardrailProvider::new(inner(), Vec::new())),
        ),
        (
            "quality_gate",
            Box::new(QualityGateProvider::new(inner(), QualityPolicy::default())),
        ),
        (
            "cache",
            Box::new(CacheProvider::new(inner(), CacheConfig::default())),
        ),
        (
            "router",
            Box::new(
                RouterProvider::new(vec![Backend::unmetered(inner())], Box::new(PreferInOrder))
                    .expect("router"),
            ),
        ),
    ]
}

#[tokio::test]
async fn every_decorator_forwards_the_stream_instead_of_calling_complete() {
    for (name, provider) in
        wrapped(|| StreamingProvider::boxed("streams", Behaviour::StreamsNatively))
    {
        let stream = provider
            .complete_stream_with_tools(&request())
            .await
            .unwrap_or_else(|e| panic!("{name}: the stream opens, got {e}"));
        let (deltas, done) = deltas_and_done(stream).await;
        assert_eq!(deltas, ["Let me", " look"], "{name}");
        assert_eq!(
            done.tool_calls.expect("tool calls")[0].function_name,
            "get_activities",
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_chain_hands_a_stream_refused_on_quota_to_the_next_tier() {
    let chain = FallbackProvider::new(vec![
        StreamingProvider::boxed("spent", Behaviour::StreamRateLimited),
        StreamingProvider::boxed("streams", Behaviour::StreamsNatively),
    ])
    .expect("chain");
    let (deltas, done) = deltas_and_done(
        chain
            .complete_stream_with_tools(&request())
            .await
            .expect("the second tier streams"),
    )
    .await;
    assert_eq!(deltas.concat(), done.content);
    assert_eq!(done.finish_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn a_pool_account_names_itself_in_a_stream_refusal() {
    let tiers = provider_pool(vec![
        StreamingProvider::boxed("streams", Behaviour::StreamsNatively),
        StreamingProvider::boxed("streams", Behaviour::StreamRateLimited),
    ])
    .expect("pool");
    let Err(err) = tiers[1].complete_stream_with_tools(&request()).await else {
        panic!("account 2 refuses");
    };
    assert_eq!(err.kind, ErrorKind::RateLimit);
    assert_eq!(err.message, "streams#2: quota spent");
}
