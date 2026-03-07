// ABOUTME: Decorator wrapping any LlmProvider to measure latency, token usage, and call counts
// ABOUTME: Provides MetricsReport snapshots for cost and performance normalization
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Cost/Latency Normalization
//!
//! [`MetricsProvider`] is a decorator that wraps any `Box<dyn LlmProvider>`,
//! measuring per-call latency and token usage. Callers can retrieve a
//! [`MetricsReport`] snapshot at any time via [`MetricsProvider::report()`].
//!
//! Token estimation: when `TokenUsage` is not provided by the inner provider,
//! tokens are estimated at ~4 characters per token.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use tracing::info;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// Characters-per-token estimate used when the provider does not report usage
const CHARS_PER_TOKEN_ESTIMATE: u32 = 4;

/// Accumulated metrics state protected by a mutex
#[derive(Debug, Default)]
struct MetricsState {
    call_count: u64,
    total_latency_ms: u64,
    total_prompt_tokens: u64,
    total_completion_tokens: u64,
    total_tokens: u64,
    errors_count: u64,
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
}

/// Decorator wrapping any `Box<dyn LlmProvider>` to collect latency and token metrics.
///
/// # Usage
///
/// ```rust,no_run
/// # use embacle::metrics::MetricsProvider;
/// # use embacle::types::LlmProvider;
/// # fn example(provider: Box<dyn LlmProvider>) {
/// let metered = MetricsProvider::new(provider);
/// // ... use metered as LlmProvider ...
/// let report = metered.report();
/// println!("calls={} avg_latency={}ms", report.call_count, report.avg_latency_ms);
/// # }
/// ```
pub struct MetricsProvider {
    inner: Box<dyn LlmProvider>,
    state: Arc<Mutex<MetricsState>>,
}

impl MetricsProvider {
    /// Wrap a provider with metrics collection
    pub fn new(inner: Box<dyn LlmProvider>) -> Self {
        Self {
            inner,
            state: Arc::new(Mutex::new(MetricsState::default())),
        }
    }

    /// Return a snapshot of the current metrics
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn report(&self) -> MetricsReport {
        let state = self.state.lock().expect("metrics lock poisoned");
        let divisor = state.call_count.max(1);
        MetricsReport {
            provider_name: self.inner.name().to_owned(),
            call_count: state.call_count,
            total_latency_ms: state.total_latency_ms,
            avg_latency_ms: state.total_latency_ms / divisor,
            total_prompt_tokens: state.total_prompt_tokens,
            total_completion_tokens: state.total_completion_tokens,
            total_tokens: state.total_tokens,
            errors_count: state.errors_count,
        }
    }

    /// Reset all counters to zero
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn reset(&self) {
        let mut state = self.state.lock().expect("metrics lock poisoned");
        *state = MetricsState::default();
    }
}

/// Estimate token count from character length (~4 chars per token)
fn estimate_tokens(text: &str) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let len = text.len() as u32;
    len / CHARS_PER_TOKEN_ESTIMATE.max(1)
}

#[async_trait]
impl LlmProvider for MetricsProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn display_name(&self) -> &'static str {
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

        let mut state = self.state.lock().expect("metrics lock poisoned");
        state.call_count += 1;
        state.total_latency_ms += elapsed_ms;

        if let Ok(response) = &result {
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

            info!(
                provider = self.inner.name(),
                elapsed_ms, prompt_tokens, completion_tokens, "metrics: complete() succeeded"
            );
        } else {
            state.errors_count += 1;
            info!(
                provider = self.inner.name(),
                elapsed_ms, "metrics: complete() failed"
            );
        }

        drop(state);
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
        fn display_name(&self) -> &'static str {
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
            let mut responses = self.responses.lock().expect("test lock");
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
        let report = metered.report();
        assert_eq!(report.call_count, 0);
        assert_eq!(report.total_latency_ms, 0);
        assert_eq!(report.avg_latency_ms, 0);
        assert_eq!(report.total_prompt_tokens, 0);
        assert_eq!(report.total_completion_tokens, 0);
        assert_eq!(report.total_tokens, 0);
        assert_eq!(report.errors_count, 0);
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

        metered.complete(&request).await.expect("first call");
        metered.complete(&request).await.expect("second call");

        let report = metered.report();
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

        let report = metered.report();
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

        metered.complete(&request).await.expect("call");

        let report = metered.report();
        assert_eq!(report.total_prompt_tokens, 2);
        assert_eq!(report.total_completion_tokens, 4);
        assert_eq!(report.total_tokens, 6);
    }

    #[test]
    fn div_by_zero_guard_on_avg_latency() {
        let provider = TestProvider::new(vec![]);
        let metered = MetricsProvider::new(Box::new(provider));
        // No calls made — avg should be 0/max(0,1) = 0, not panic
        let report = metered.report();
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

        metered.complete(&request).await.expect("call");
        assert_eq!(metered.report().call_count, 1);

        metered.reset();
        let report = metered.report();
        assert_eq!(report.call_count, 0);
        assert_eq!(report.total_tokens, 0);
        assert_eq!(report.errors_count, 0);
    }
}
