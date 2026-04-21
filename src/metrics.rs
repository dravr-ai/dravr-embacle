// ABOUTME: Decorator wrapping any LlmProvider to measure latency, token usage, cost, and call counts
// ABOUTME: Provides MetricsReport snapshots for cost and performance normalization, optional OTel export
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Cost/Latency Normalization
//!
//! [`MetricsProvider`] is a decorator that wraps any `Box<dyn LlmProvider>`,
//! measuring per-call latency and token usage. Callers can retrieve a
//! [`MetricsReport`] snapshot at any time via [`MetricsProvider::report()`].
//!
//! ## Cost Tracking
//!
//! Attach a [`PricingTable`] via [`MetricsProvider::with_pricing()`] or use
//! [`MetricsProvider::with_default_pricing()`] for built-in model prices.
//! Costs are computed from token counts and accumulated in the report.
//!
//! ## OpenTelemetry (feature: `otel`)
//!
//! When built with `--features otel`, instruments are created via the
//! global meter `embacle` and recorded on each `complete()` call.
//!
//! Token estimation: when `TokenUsage` is not provided by the inner provider,
//! tokens are estimated at ~4 characters per token.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "otel")]
use opentelemetry::global;
#[cfg(feature = "otel")]
use opentelemetry::metrics::{Counter, Histogram};

use async_trait::async_trait;
use tracing::info;

use crate::turn::ConversationTurnId;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// Characters-per-token estimate used when the provider does not report usage
const CHARS_PER_TOKEN_ESTIMATE: u32 = 4;

/// Per-model token pricing (cost per 1000 tokens)
#[derive(Debug, Clone)]
pub struct TokenPricing {
    /// Cost per 1000 prompt (input) tokens
    pub prompt_price_per_1k: f64,
    /// Cost per 1000 completion (output) tokens
    pub completion_price_per_1k: f64,
}

/// Mapping from model name to pricing
pub type PricingTable = HashMap<String, TokenPricing>;

/// Built-in pricing table with known model prices (approximate, USD)
pub fn default_pricing_table() -> PricingTable {
    let mut table = PricingTable::new();
    // Claude models
    table.insert(
        "opus".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.015,
            completion_price_per_1k: 0.075,
        },
    );
    table.insert(
        "sonnet".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.003,
            completion_price_per_1k: 0.015,
        },
    );
    table.insert(
        "haiku".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.00025,
            completion_price_per_1k: 0.00125,
        },
    );
    // GPT models
    table.insert(
        "gpt-5.4".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.005,
            completion_price_per_1k: 0.015,
        },
    );
    table.insert(
        "gpt-4o".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.005,
            completion_price_per_1k: 0.015,
        },
    );
    // Gemini models
    table.insert(
        "gemini-2.5-pro".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.00125,
            completion_price_per_1k: 0.005,
        },
    );
    table.insert(
        "gemini-2.5-flash".to_owned(),
        TokenPricing {
            prompt_price_per_1k: 0.000_075,
            completion_price_per_1k: 0.0003,
        },
    );
    table
}

/// Accumulated metrics state protected by a mutex
#[derive(Debug, Default)]
struct MetricsState {
    call_count: u64,
    total_latency_ms: u64,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    total_tokens: u64,
    errors_count: u64,
    total_cost: f64,
}

/// Record describing one `complete()` call attributed to a conversation turn.
///
/// Emitted by [`MetricsProvider`] via a user-supplied [`PerCallMetricsSink`]
/// whenever the incoming [`ChatRequest`](crate::types::ChatRequest) carries a
/// [`ConversationTurnId`]. The aggregate [`MetricsReport`] path is unaffected.
#[derive(Debug, Clone)]
pub struct PerCallMetric {
    /// Conversation turn the call belongs to
    pub turn_id: ConversationTurnId,
    /// Name of the underlying provider (e.g. `claude_code`, `openai_api`)
    pub provider: String,
    /// Model identifier reported by the provider response
    pub model: String,
    /// Measured wall-clock latency for this call
    pub latency_ms: u64,
    /// Cost in USD computed from the configured pricing table (0.0 if none)
    pub cost_usd: f64,
    /// Prompt tokens reported by the provider, or estimated when absent
    pub prompt_tokens: u64,
    /// Completion tokens reported by the provider, or estimated when absent
    pub completion_tokens: u64,
    /// Whether the call succeeded (false means the provider returned an error)
    pub success: bool,
    /// Wall-clock time the call completed, as milliseconds since UNIX epoch
    pub timestamp_ms: u64,
}

/// Sink that receives per-call metrics emitted by [`MetricsProvider`].
///
/// Implementations are expected to enqueue or persist the metric without
/// blocking. The sink is invoked synchronously from within `complete()`, so
/// expensive work (for example, a database write) should be dispatched to a
/// background task.
pub trait PerCallMetricsSink: Send + Sync {
    /// Record a single per-call metric.
    fn record(&self, metric: PerCallMetric);
}

impl<F> PerCallMetricsSink for F
where
    F: Fn(PerCallMetric) + Send + Sync,
{
    fn record(&self, metric: PerCallMetric) {
        self(metric);
    }
}

/// Snapshot of accumulated metrics for a provider
#[derive(Debug, Clone)]
pub struct MetricsReport {
    /// Name of the wrapped provider
    pub provider_name: String,
    /// Total number of `complete()` calls
    pub call_count: u64,
    /// Total latency across all calls (milliseconds)
    pub total_latency_ms: u64,
    /// Average latency per call (milliseconds)
    pub avg_latency_ms: u64,
    /// Total prompt tokens consumed
    pub total_prompt_tokens: u64,
    /// Total completion tokens generated
    pub total_completion_tokens: u64,
    /// Total tokens (prompt + completion)
    pub total_tokens: u64,
    /// Number of calls that returned an error
    pub errors_count: u64,
    /// Accumulated cost in USD (0.0 if no pricing table configured)
    pub total_cost: f64,
}

/// OpenTelemetry instruments for metrics export
#[cfg(feature = "otel")]
struct OtelInstruments {
    requests_total: Counter<u64>,
    requests_duration_ms: Histogram<f64>,
    tokens_prompt: Counter<u64>,
    tokens_completion: Counter<u64>,
    errors_total: Counter<u64>,
    cost_total: Counter<f64>,
}

#[cfg(feature = "otel")]
impl OtelInstruments {
    fn new() -> Self {
        let meter = global::meter("embacle");
        Self {
            requests_total: meter
                .u64_counter("embacle.requests.total")
                .with_description("Total LLM requests")
                .build(),
            requests_duration_ms: meter
                .f64_histogram("embacle.requests.duration_ms")
                .with_description("Request duration in milliseconds")
                .build(),
            tokens_prompt: meter
                .u64_counter("embacle.tokens.prompt")
                .with_description("Total prompt tokens consumed")
                .build(),
            tokens_completion: meter
                .u64_counter("embacle.tokens.completion")
                .with_description("Total completion tokens generated")
                .build(),
            errors_total: meter
                .u64_counter("embacle.errors.total")
                .with_description("Total error count")
                .build(),
            cost_total: meter
                .f64_counter("embacle.cost.total")
                .with_description("Total cost in USD")
                .build(),
        }
    }
}

/// Decorator wrapping any `Box<dyn LlmProvider>` to collect latency, token, and cost metrics.
///
/// # Usage
///
/// ```rust,no_run
/// # use embacle::metrics::MetricsProvider;
/// # use embacle::types::LlmProvider;
/// # fn example(provider: Box<dyn LlmProvider>) {
/// let metered = MetricsProvider::new(provider);
/// // ... use metered as LlmProvider ...
/// let report = metered.report().unwrap(); // Safe: test assertion
/// println!("calls={} avg_latency={}ms", report.call_count, report.avg_latency_ms);
/// # }
/// ```
pub struct MetricsProvider {
    inner: Box<dyn LlmProvider>,
    state: Arc<Mutex<MetricsState>>,
    pricing: Option<PricingTable>,
    per_call_sink: Option<Arc<dyn PerCallMetricsSink>>,
    #[cfg(feature = "otel")]
    otel: OtelInstruments,
}

impl MetricsProvider {
    /// Wrap a provider with metrics collection
    pub fn new(inner: Box<dyn LlmProvider>) -> Self {
        Self {
            inner,
            state: Arc::new(Mutex::new(MetricsState::default())),
            pricing: None,
            per_call_sink: None,
            #[cfg(feature = "otel")]
            otel: OtelInstruments::new(),
        }
    }

    /// Attach a custom pricing table for cost tracking
    pub fn with_pricing(mut self, pricing: PricingTable) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Attach the built-in pricing table for known models
    pub fn with_default_pricing(self) -> Self {
        self.with_pricing(default_pricing_table())
    }

    /// Attach a per-call metrics sink.
    ///
    /// When set, every `complete()` invocation whose request carries a
    /// [`ConversationTurnId`] produces a [`PerCallMetric`] record that is
    /// handed to `sink`. Requests without a turn identifier are recorded
    /// only in the aggregate state.
    pub fn with_per_call_sink(mut self, sink: Arc<dyn PerCallMetricsSink>) -> Self {
        self.per_call_sink = Some(sink);
        self
    }

    /// Return a snapshot of the current metrics
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the internal mutex is poisoned.
    pub fn report(&self) -> Result<MetricsReport, RunnerError> {
        let state = self
            .state
            .lock()
            .map_err(|_| RunnerError::internal("metrics lock poisoned"))?;
        let divisor = state.call_count.max(1);
        Ok(MetricsReport {
            provider_name: self.inner.name().to_owned(),
            call_count: state.call_count,
            total_latency_ms: state.total_latency_ms,
            avg_latency_ms: state.total_latency_ms / divisor,
            total_prompt_tokens: state.total_prompt_tokens,
            total_completion_tokens: state.total_completion_tokens,
            total_tokens: state.total_tokens,
            errors_count: state.errors_count,
            total_cost: state.total_cost,
        })
    }

    /// Reset all counters to zero
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the internal mutex is poisoned.
    pub fn reset(&self) -> Result<(), RunnerError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| RunnerError::internal("metrics lock poisoned"))?;
        *state = MetricsState::default();
        Ok(())
    }

    /// Compute cost for a single call based on token counts and model name
    fn compute_cost(&self, model: &str, prompt_tokens: u64, completion_tokens: u64) -> f64 {
        let Some(table) = &self.pricing else {
            return 0.0;
        };
        // Try exact match first, then try substring matching for partial model names
        let pricing = table.get(model).or_else(|| {
            table
                .iter()
                .find(|(key, _)| model.contains(key.as_str()))
                .map(|(_, v)| v)
        });
        let Some(pricing) = pricing else {
            return 0.0;
        };
        #[allow(clippy::cast_precision_loss)]
        let cost = (prompt_tokens as f64 * pricing.prompt_price_per_1k / 1000.0)
            + (completion_tokens as f64 * pricing.completion_price_per_1k / 1000.0);
        cost
    }
}

/// Estimate token count from character length (~4 chars per token)
fn estimate_tokens(text: &str) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let len = text.len() as u32;
    len / CHARS_PER_TOKEN_ESTIMATE.max(1)
}

/// Wall-clock time as milliseconds since the UNIX epoch (saturates at zero on clock skew).
fn unix_epoch_millis() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis()) as u64;
    millis
}

#[async_trait]
impl LlmProvider for MetricsProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn display_name(&self) -> &str {
        self.inner.display_name()
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.inner.capabilities()
    }

    fn default_model(&self) -> &str {
        self.inner.default_model()
    }

    fn available_models(&self) -> &[String] {
        self.inner.available_models()
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let start = Instant::now();
        let result = self.inner.complete(request).await;
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = start.elapsed().as_millis() as u64;

        let mut state = self
            .state
            .lock()
            .map_err(|_| RunnerError::internal("metrics lock poisoned"))?;
        state.call_count += 1;
        state.total_latency_ms += elapsed_ms;

        #[cfg(feature = "otel")]
        let otel_attrs: Vec<opentelemetry::KeyValue> = {
            let mut attrs = vec![opentelemetry::KeyValue::new("provider", self.inner.name())];
            if let Some(turn_id) = request.turn_id {
                attrs.push(opentelemetry::KeyValue::new("turn_id", turn_id.to_string()));
            }
            attrs
        };

        let per_call = if let Ok(response) = &result {
            let usage = response.usage.as_ref();
            let prompt_tokens = u64::from(
                usage.map_or_else(|| estimate_prompt_tokens(request), |u| u.prompt_tokens),
            );
            let completion_tokens = u64::from(usage.map_or_else(
                || estimate_tokens(&response.content),
                |u| u.completion_tokens,
            ));
            let total = prompt_tokens + completion_tokens;

            state.total_prompt_tokens += prompt_tokens;
            state.total_completion_tokens += completion_tokens;
            state.total_tokens += total;

            let cost = self.compute_cost(&response.model, prompt_tokens, completion_tokens);
            state.total_cost += cost;

            info!(
                provider = self.inner.name(),
                elapsed_ms, prompt_tokens, completion_tokens, cost, "metrics: complete() succeeded"
            );

            #[cfg(feature = "otel")]
            {
                let attrs = otel_attrs.as_slice();
                self.otel.requests_total.add(1, attrs);
                #[allow(clippy::cast_precision_loss)]
                self.otel
                    .requests_duration_ms
                    .record(elapsed_ms as f64, attrs);
                self.otel.tokens_prompt.add(prompt_tokens, attrs);
                self.otel.tokens_completion.add(completion_tokens, attrs);
                if cost > 0.0 {
                    self.otel.cost_total.add(cost, attrs);
                }
            }

            request.turn_id.map(|turn_id| PerCallMetric {
                turn_id,
                provider: self.inner.name().to_owned(),
                model: response.model.clone(),
                latency_ms: elapsed_ms,
                cost_usd: cost,
                prompt_tokens,
                completion_tokens,
                success: true,
                timestamp_ms: unix_epoch_millis(),
            })
        } else {
            state.errors_count += 1;
            info!(
                provider = self.inner.name(),
                elapsed_ms, "metrics: complete() failed"
            );

            #[cfg(feature = "otel")]
            {
                self.otel.errors_total.add(1, otel_attrs.as_slice());
            }

            request.turn_id.map(|turn_id| PerCallMetric {
                turn_id,
                provider: self.inner.name().to_owned(),
                model: request.model.clone().unwrap_or_default(),
                latency_ms: elapsed_ms,
                cost_usd: 0.0,
                prompt_tokens: 0,
                completion_tokens: 0,
                success: false,
                timestamp_ms: unix_epoch_millis(),
            })
        };

        drop(state);

        if let (Some(metric), Some(sink)) = (per_call, self.per_call_sink.as_ref()) {
            sink.record(metric);
        }

        result
    }

    /// Delegate streaming directly; only measures stream setup time (documented limitation)
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.inner.complete_stream(request).await
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.inner.health_check().await
    }
}

/// Estimate prompt tokens from request messages
fn estimate_prompt_tokens(request: &ChatRequest) -> u32 {
    let total_chars: usize = request.messages.iter().map(|m| m.content.len()).sum();
    #[allow(clippy::cast_possible_truncation)]
    let len = total_chars as u32;
    len / CHARS_PER_TOKEN_ESTIMATE.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
        RunnerError, TokenUsage,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TestProvider {
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
    }

    impl TestProvider {
        fn new(responses: Vec<Result<ChatResponse, RunnerError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            "test"
        }
        fn display_name(&self) -> &str {
            "Test Provider"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::text_only()
        }
        fn default_model(&self) -> &'static str {
            "test-model"
        }
        fn available_models(&self) -> &[String] {
            &[]
        }

        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock"); // Safe: test assertion
            if responses.is_empty() {
                Ok(ChatResponse {
                    content: "default".to_owned(),
                    model: "test-model".to_owned(),
                    usage: None,
                    finish_reason: Some("stop".to_owned()),
                    warnings: None,
                    tool_calls: None,
                })
            } else {
                responses.remove(0)
            }
        }

        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("streaming not supported in test"))
        }

        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
        }
    }

    #[test]
    fn fresh_report_is_zeroed() {
        let provider = TestProvider::new(vec![]);
        let metered = MetricsProvider::new(Box::new(provider));
        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.call_count, 0);
        assert_eq!(report.total_latency_ms, 0);
        assert_eq!(report.avg_latency_ms, 0);
        assert_eq!(report.total_prompt_tokens, 0);
        assert_eq!(report.total_completion_tokens, 0);
        assert_eq!(report.total_tokens, 0);
        assert_eq!(report.errors_count, 0);
        assert!(report.total_cost == 0.0);
        assert_eq!(report.provider_name, "test");
    }

    #[tokio::test]
    async fn call_count_increments() {
        let provider = TestProvider::new(vec![
            Ok(ChatResponse {
                content: "hello world".to_owned(),
                model: "test-model".to_owned(),
                usage: Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                }),
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
            Ok(ChatResponse {
                content: "second".to_owned(),
                model: "test-model".to_owned(),
                usage: Some(TokenUsage {
                    prompt_tokens: 8,
                    completion_tokens: 3,
                    total_tokens: 11,
                }),
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
        ]);
        let metered = MetricsProvider::new(Box::new(provider));
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        metered.complete(&request).await.expect("first call"); // Safe: test assertion
        metered.complete(&request).await.expect("second call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.call_count, 2);
        assert_eq!(report.total_prompt_tokens, 18);
        assert_eq!(report.total_completion_tokens, 8);
        assert_eq!(report.total_tokens, 26);
        assert_eq!(report.errors_count, 0);
    }

    #[tokio::test]
    async fn errors_count_on_failure() {
        let provider = TestProvider::new(vec![Err(RunnerError::external_service("test", "boom"))]);
        let metered = MetricsProvider::new(Box::new(provider));
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let result = metered.complete(&request).await;
        assert!(result.is_err());

        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.call_count, 1);
        assert_eq!(report.errors_count, 1);
    }

    #[tokio::test]
    async fn token_estimation_when_no_usage() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "abcdefghijklmnop".to_owned(), // 16 chars => 4 tokens
            model: "test-model".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider));
        let request = ChatRequest::new(vec![ChatMessage::user("12345678")]); // 8 chars => 2 tokens

        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.total_prompt_tokens, 2);
        assert_eq!(report.total_completion_tokens, 4);
        assert_eq!(report.total_tokens, 6);
    }

    #[test]
    fn div_by_zero_guard_on_avg_latency() {
        let provider = TestProvider::new(vec![]);
        let metered = MetricsProvider::new(Box::new(provider));
        // No calls made — avg should be 0/max(0,1) = 0, not panic
        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.avg_latency_ms, 0);
    }

    #[tokio::test]
    async fn reset_zeroes_counters() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "hello".to_owned(),
            model: "test-model".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 2,
                total_tokens: 7,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider));
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        metered.complete(&request).await.expect("call"); // Safe: test assertion
        assert_eq!(metered.report().unwrap().call_count, 1); // Safe: test assertion

        metered.reset().unwrap(); // Safe: test assertion
        let report = metered.report().unwrap(); // Safe: test assertion
        assert_eq!(report.call_count, 0);
        assert_eq!(report.total_tokens, 0);
        assert_eq!(report.errors_count, 0);
        assert!(report.total_cost == 0.0);
    }

    // ========================================================================
    // Cost tracking tests
    // ========================================================================

    #[tokio::test]
    async fn cost_with_known_model() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "response".to_owned(),
            model: "opus".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 500,
                total_tokens: 1500,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider)).with_default_pricing();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
                                                // opus: 1000 prompt * 0.015/1000 + 500 completion * 0.075/1000
                                                // = 0.015 + 0.0375 = 0.0525
        assert!((report.total_cost - 0.0525).abs() < 1e-10);
    }

    #[tokio::test]
    async fn cost_with_unknown_model() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "response".to_owned(),
            model: "some-unknown-model".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 500,
                total_tokens: 1500,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider)).with_default_pricing();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
        assert!(report.total_cost == 0.0);
    }

    #[tokio::test]
    async fn cost_accumulates() {
        let provider = TestProvider::new(vec![
            Ok(ChatResponse {
                content: "r1".to_owned(),
                model: "opus".to_owned(),
                usage: Some(TokenUsage {
                    prompt_tokens: 1000,
                    completion_tokens: 500,
                    total_tokens: 1500,
                }),
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
            Ok(ChatResponse {
                content: "r2".to_owned(),
                model: "opus".to_owned(),
                usage: Some(TokenUsage {
                    prompt_tokens: 2000,
                    completion_tokens: 1000,
                    total_tokens: 3000,
                }),
                finish_reason: Some("stop".to_owned()),
                warnings: None,
                tool_calls: None,
            }),
        ]);
        let metered = MetricsProvider::new(Box::new(provider)).with_default_pricing();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        metered.complete(&request).await.expect("call 1"); // Safe: test assertion
        metered.complete(&request).await.expect("call 2"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
                                                // call1: 0.015 + 0.0375 = 0.0525
                                                // call2: 0.030 + 0.075 = 0.105
                                                // total: 0.1575
        assert!((report.total_cost - 0.1575).abs() < 1e-10);
    }

    #[tokio::test]
    async fn cost_without_pricing() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "response".to_owned(),
            model: "opus".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 500,
                total_tokens: 1500,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider));
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
        assert!(report.total_cost == 0.0);
    }

    #[tokio::test]
    async fn cost_with_estimated_tokens() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "abcdefghijklmnop".to_owned(), // 16 chars => 4 tokens
            model: "opus".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider)).with_default_pricing();
        let request = ChatRequest::new(vec![ChatMessage::user("12345678")]); // 8 chars => 2 tokens
        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let report = metered.report().unwrap(); // Safe: test assertion
                                                // 2 prompt tokens, 4 completion tokens via estimation
                                                // 2 * 0.015/1000 + 4 * 0.075/1000 = 0.00003 + 0.0003 = 0.00033
        assert!(report.total_cost > 0.0);
        assert!((report.total_cost - 0.00033).abs() < 1e-10);
    }

    #[test]
    fn default_pricing_populated() {
        let table = default_pricing_table();
        assert!(table.contains_key("opus"));
        assert!(table.contains_key("sonnet"));
        assert!(table.contains_key("haiku"));
        assert!(table.contains_key("gpt-5.4"));
        assert!(table.contains_key("gemini-2.5-pro"));
        assert!(table.contains_key("gemini-2.5-flash"));
        assert!(table.len() >= 7);
    }

    #[tokio::test]
    async fn reset_zeroes_cost() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "response".to_owned(),
            model: "opus".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 1000,
                completion_tokens: 500,
                total_tokens: 1500,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let metered = MetricsProvider::new(Box::new(provider)).with_default_pricing();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        metered.complete(&request).await.expect("call"); // Safe: test assertion
        assert!(metered.report().unwrap().total_cost > 0.0); // Safe: test assertion

        metered.reset().unwrap(); // Safe: test assertion
        assert!(metered.report().unwrap().total_cost == 0.0); // Safe: test assertion
    }

    // ========================================================================
    // Per-call sink tests
    // ========================================================================

    #[tokio::test]
    async fn per_call_sink_emits_when_turn_id_present() {
        use crate::turn::ConversationTurnId;
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "hello".to_owned(),
            model: "opus".to_owned(),
            usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let captured: Arc<Mutex<Vec<PerCallMetric>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let metered = MetricsProvider::new(Box::new(provider))
            .with_default_pricing()
            .with_per_call_sink(Arc::new(move |m: PerCallMetric| {
                sink.lock().expect("test lock").push(m); // Safe: test assertion
            }));
        let turn_id = ConversationTurnId::new();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]).with_turn_id(turn_id);

        metered.complete(&request).await.expect("call"); // Safe: test assertion

        let captured = captured.lock().expect("test lock"); // Safe: test assertion
        assert_eq!(captured.len(), 1);
        let metric = &captured[0];
        assert_eq!(metric.turn_id, turn_id);
        assert_eq!(metric.provider, "test");
        assert_eq!(metric.model, "opus");
        assert_eq!(metric.prompt_tokens, 10);
        assert_eq!(metric.completion_tokens, 5);
        assert!(metric.success);
        assert!(metric.cost_usd > 0.0);
    }

    #[tokio::test]
    async fn per_call_sink_silent_without_turn_id() {
        let provider = TestProvider::new(vec![Ok(ChatResponse {
            content: "hello".to_owned(),
            model: "opus".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })]);
        let captured: Arc<Mutex<Vec<PerCallMetric>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let metered = MetricsProvider::new(Box::new(provider)).with_per_call_sink(Arc::new(
            move |m: PerCallMetric| {
                sink.lock().expect("test lock").push(m); // Safe: test assertion
            },
        ));
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        metered.complete(&request).await.expect("call"); // Safe: test assertion

        assert!(captured.lock().expect("test lock").is_empty()); // Safe: test assertion
    }

    #[tokio::test]
    async fn per_call_sink_emits_on_error_with_turn_id() {
        use crate::turn::ConversationTurnId;
        let provider = TestProvider::new(vec![Err(RunnerError::external_service("test", "boom"))]);
        let captured: Arc<Mutex<Vec<PerCallMetric>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let metered = MetricsProvider::new(Box::new(provider)).with_per_call_sink(Arc::new(
            move |m: PerCallMetric| {
                sink.lock().expect("test lock").push(m); // Safe: test assertion
            },
        ));
        let turn_id = ConversationTurnId::new();
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]).with_turn_id(turn_id);

        assert!(metered.complete(&request).await.is_err());

        let captured = captured.lock().expect("test lock"); // Safe: test assertion
        assert_eq!(captured.len(), 1);
        assert!(!captured[0].success);
        assert_eq!(captured[0].turn_id, turn_id);
    }
}
