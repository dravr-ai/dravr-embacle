// ABOUTME: Cross-decorator scenario tests exercising FallbackProvider, MetricsProvider,
// ABOUTME: QualityGateProvider, and GuardrailProvider together with ScriptedProvider
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use embacle::fallback::{FallbackProvider, RetryConfig};
use embacle::guardrail::{GuardrailProvider, TopicFilterGuardrail};
use embacle::metrics::MetricsProvider;
use embacle::quality_gate::{QualityGateProvider, QualityPolicy};
use embacle::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    RunnerError,
};

// ============================================================================
// ScriptedProvider — programmable test double
// ============================================================================

struct ScriptedProvider {
    provider_name: &'static str,
    caps: LlmCapabilities,
    models: Vec<String>,
    responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
    call_count: AtomicU32,
    healthy: bool,
}

impl ScriptedProvider {
    fn builder(name: &'static str) -> ScriptedProviderBuilder {
        ScriptedProviderBuilder {
            name,
            caps: LlmCapabilities::text_only(),
            models: vec![format!("{name}-model")],
            responses: vec![],
            healthy: true,
        }
    }
}

struct ScriptedProviderBuilder {
    name: &'static str,
    caps: LlmCapabilities,
    models: Vec<String>,
    responses: Vec<Result<ChatResponse, RunnerError>>,
    healthy: bool,
}

impl ScriptedProviderBuilder {
    const fn with_capabilities(mut self, caps: LlmCapabilities) -> Self {
        self.caps = caps;
        self
    }

    fn with_response(mut self, response: Result<ChatResponse, RunnerError>) -> Self {
        self.responses.push(response);
        self
    }

    const fn with_health(mut self, healthy: bool) -> Self {
        self.healthy = healthy;
        self
    }

    fn build(self) -> ScriptedProvider {
        ScriptedProvider {
            provider_name: self.name,
            caps: self.caps,
            models: self.models,
            responses: Mutex::new(self.responses),
            call_count: AtomicU32::new(0),
            healthy: self.healthy,
        }
    }
}

fn make_response(content: &str, model: &str) -> ChatResponse {
    ChatResponse {
        content: content.to_owned(),
        model: model.to_owned(),
        usage: None,
        finish_reason: Some("stop".to_owned()),
        warnings: None,
        tool_calls: None,
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    fn name(&self) -> &'static str {
        self.provider_name
    }
    fn display_name(&self) -> &str {
        self.provider_name
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
        let mut responses = self.responses.lock().expect("test lock");
        if responses.is_empty() {
            Err(RunnerError::internal("no more scripted responses"))
        } else {
            responses.remove(0)
        }
    }
    async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal(
            "streaming not supported in scripted provider",
        ))
    }
    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(self.healthy)
    }
}

// ============================================================================
// Scenario 1: Fallback(Metrics(A), Metrics(B)) — A fails, B succeeds
// ============================================================================

#[tokio::test]
async fn fallback_with_metrics_a_fails_b_succeeds() {
    let a = ScriptedProvider::builder("provider_a")
        .with_response(Err(RunnerError::external_service("a", "service down")))
        .build();
    let b = ScriptedProvider::builder("provider_b")
        .with_response(Ok(make_response("from B", "b-model")))
        .build();

    let metered_a = MetricsProvider::new(Box::new(a));
    let metered_b = MetricsProvider::new(Box::new(b));

    let fallback =
        FallbackProvider::new(vec![Box::new(metered_a), Box::new(metered_b)]).expect("non-empty");

    let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
    let response = fallback.complete(&request).await.expect("B should succeed");
    assert_eq!(response.content, "from B");
}

// ============================================================================
// Scenario 2: Metrics(QualityGate(provider)) — refusal then success
// ============================================================================

#[tokio::test]
async fn metrics_wrapping_quality_gate_refusal_then_success() {
    let provider = ScriptedProvider::builder("inner")
        .with_response(Ok(make_response("I cannot help with that", "m")))
        .with_response(Ok(make_response("Here is a helpful answer", "m")))
        .build();

    let policy = QualityPolicy {
        max_retries: 1,
        ..QualityPolicy::default()
    };
    let gated = QualityGateProvider::new(Box::new(provider), policy);
    let metered = MetricsProvider::new(Box::new(gated));

    let request = ChatRequest::new(vec![ChatMessage::user("help me")]);
    let response = metered
        .complete(&request)
        .await
        .expect("should succeed on retry");
    assert_eq!(response.content, "Here is a helpful answer");

    let report = metered.report();
    assert_eq!(report.call_count, 1);
    assert_eq!(report.errors_count, 0);
}

// ============================================================================
// Scenario 3: Fallback with mixed capabilities
// ============================================================================

#[tokio::test]
async fn fallback_with_mixed_capabilities() {
    let a = ScriptedProvider::builder("full")
        .with_capabilities(LlmCapabilities::full_featured())
        .with_response(Ok(make_response("full featured", "full-model")))
        .build();
    let b = ScriptedProvider::builder("text_only")
        .with_capabilities(LlmCapabilities::text_only())
        .with_response(Ok(make_response("text only", "text-model")))
        .build();

    let fallback = FallbackProvider::new(vec![Box::new(a), Box::new(b)]).expect("non-empty");

    // Combined capabilities should include function calling from A
    let caps = fallback.capabilities();
    assert!(caps.supports_function_calling());
    assert!(caps.supports_streaming());
    assert!(caps.supports_vision());

    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
    let response = fallback.complete(&request).await.expect("A should work");
    assert_eq!(response.content, "full featured");
}

// ============================================================================
// Scenario 4: Fallback with retry — A times out, B returns refusal then passes quality gate
// ============================================================================

#[tokio::test]
async fn fallback_retry_timeout_then_quality_gate_success() {
    let a = ScriptedProvider::builder("timeout_provider")
        .with_response(Err(RunnerError::timeout("t1")))
        .with_response(Err(RunnerError::timeout("t2")))
        .with_response(Err(RunnerError::timeout("t3")))
        .build();
    let b = ScriptedProvider::builder("refusal_then_ok")
        .with_response(Ok(make_response("I cannot help", "b-model")))
        .with_response(Ok(make_response("Here is the answer", "b-model")))
        .build();

    let gated_b = QualityGateProvider::new(
        Box::new(b),
        QualityPolicy {
            max_retries: 1,
            ..QualityPolicy::default()
        },
    );

    let retry = RetryConfig {
        max_retries: 2,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(10),
    };
    let fallback = FallbackProvider::with_retry(vec![Box::new(a), Box::new(gated_b)], retry)
        .expect("non-empty");

    let request = ChatRequest::new(vec![ChatMessage::user("help")]);
    let response = fallback
        .complete(&request)
        .await
        .expect("B should pass quality gate");
    assert_eq!(response.content, "Here is the answer");
}

// ============================================================================
// Scenario 5: All providers return refusal — quality gate exhaustion propagates
// ============================================================================

#[tokio::test]
async fn all_providers_refuse_quality_gate_exhaustion() {
    let a = ScriptedProvider::builder("refuser_a")
        .with_response(Ok(make_response("I cannot do that", "a-model")))
        .with_response(Ok(make_response("I can't help", "a-model")))
        .with_response(Ok(make_response("As an AI model", "a-model")))
        .build();
    let b = ScriptedProvider::builder("refuser_b")
        .with_response(Ok(make_response("I cannot assist", "b-model")))
        .with_response(Ok(make_response("I can't do that", "b-model")))
        .with_response(Ok(make_response("I cannot comply", "b-model")))
        .build();

    let policy = QualityPolicy {
        max_retries: 2,
        ..QualityPolicy::default()
    };
    let gated_a = QualityGateProvider::new(Box::new(a), policy.clone());
    let gated_b = QualityGateProvider::new(Box::new(b), policy);

    let fallback =
        FallbackProvider::new(vec![Box::new(gated_a), Box::new(gated_b)]).expect("non-empty");

    let request = ChatRequest::new(vec![ChatMessage::user("help me")]);
    let response = fallback
        .complete(&request)
        .await
        .expect("returns last response");
    // quality_gate_exhausted means the quality gate gave up but still returned a response
    assert_eq!(
        response.finish_reason,
        Some("quality_gate_exhausted".to_owned())
    );
}

// ============================================================================
// Scenario 6: Health check composition — mixed healthy/unhealthy
// ============================================================================

#[tokio::test]
async fn health_check_composition() {
    let a = ScriptedProvider::builder("unhealthy_a")
        .with_health(false)
        .build();
    let b = ScriptedProvider::builder("healthy_b")
        .with_health(true)
        .build();
    let c = ScriptedProvider::builder("unhealthy_c")
        .with_health(false)
        .build();

    let fallback =
        FallbackProvider::new(vec![Box::new(a), Box::new(b), Box::new(c)]).expect("non-empty");

    // At least one provider is healthy
    assert!(fallback.health_check().await.expect("health check"));

    // All unhealthy
    let all_down = FallbackProvider::new(vec![
        Box::new(ScriptedProvider::builder("x").with_health(false).build()),
        Box::new(ScriptedProvider::builder("y").with_health(false).build()),
    ])
    .expect("non-empty");

    assert!(!all_down.health_check().await.expect("health check"));
}

// ============================================================================
// Scenario 7: Guardrail(Fallback) — pre-request rejection, provider never called
// ============================================================================

#[tokio::test]
async fn guardrail_blocks_before_fallback() {
    let a = ScriptedProvider::builder("provider_a")
        .with_response(Ok(make_response("should not reach", "a-model")))
        .build();
    let b = ScriptedProvider::builder("provider_b")
        .with_response(Ok(make_response("should not reach", "b-model")))
        .build();

    let fallback = FallbackProvider::new(vec![Box::new(a), Box::new(b)]).expect("non-empty");

    let guard = TopicFilterGuardrail {
        blocked_patterns: vec!["prohibited".to_owned()],
    };
    let guarded = GuardrailProvider::new(Box::new(fallback), vec![Box::new(guard)]);

    let request = ChatRequest::new(vec![ChatMessage::user("tell me about prohibited topics")]);
    let err = guarded.complete(&request).await.unwrap_err();

    assert_eq!(err.kind, ErrorKind::Guardrail);
    assert!(err.message.contains("prohibited"));
}
