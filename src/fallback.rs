// ABOUTME: Provider fallback chains that try multiple LlmProviders in order
// ABOUTME: Returns the first successful response or the last error encountered
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Provider Fallback Chains
//!
//! [`FallbackProvider`] wraps multiple `Box<dyn LlmProvider>` instances and
//! tries them in order for each request. The first successful response is
//! returned; if all providers fail, the last error is propagated.
//!
//! Optional per-provider retry with exponential backoff can be configured
//! via [`RetryConfig`] and [`FallbackProvider::with_retry()`]. Retries
//! are only attempted for transient errors (see [`ErrorKind::is_transient()`]).
//!
//! Health checks pass if ANY provider is healthy. Capabilities are the
//! bitwise OR of all inner providers.

use std::time::Duration;

use async_trait::async_trait;
use tokio::time;
use tracing::warn;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
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

/// Provider that tries multiple inner providers in order, returning the first success.
///
/// # Construction
///
/// Use [`FallbackProvider::new()`] with a non-empty `Vec` of providers.
/// An empty vec is rejected with a config error.
///
/// Use [`FallbackProvider::with_retry()`] to enable per-provider retry
/// with exponential backoff on transient errors.
pub struct FallbackProvider {
    providers: Vec<Box<dyn LlmProvider>>,
    display_name: String,
    combined_models: Vec<String>,
    retry_config: RetryConfig,
}

impl FallbackProvider {
    /// Create a fallback chain from a non-empty list of providers.
    ///
    /// No retries are attempted (equivalent to `max_retries = 0`).
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

        Ok(Self {
            providers,
            display_name,
            combined_models,
            retry_config,
        })
    }

    /// Compute the backoff delay for a given attempt (0-indexed)
    fn backoff_delay(&self, attempt: u32) -> Duration {
        let delay = self
            .retry_config
            .base_delay
            .saturating_mul(2u32.saturating_pow(attempt));
        delay.min(self.retry_config.max_delay)
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
        let mut last_error = RunnerError::internal("no providers configured");

        for provider in &self.providers {
            for attempt in 0..=self.retry_config.max_retries {
                match provider.complete(request).await {
                    Ok(response) => return Ok(response),
                    Err(err) => {
                        let is_retryable =
                            err.kind.is_transient() && attempt < self.retry_config.max_retries;
                        if is_retryable {
                            let delay = self.backoff_delay(attempt);
                            #[allow(clippy::cast_possible_truncation)]
                            let delay_ms = delay.as_millis() as u64;
                            warn!(
                                provider = provider.name(),
                                attempt,
                                error = %err,
                                delay_ms,
                                "fallback: transient error, retrying after backoff"
                            );
                            time::sleep(delay).await;
                        } else {
                            warn!(
                                provider = provider.name(),
                                error = %err,
                                "fallback: provider failed, trying next"
                            );
                            last_error = err;
                            break;
                        }
                    }
                }
            }
        }

        Err(last_error)
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let mut last_error = RunnerError::internal("no providers configured");

        for provider in &self.providers {
            for attempt in 0..=self.retry_config.max_retries {
                match provider.complete_stream(request).await {
                    Ok(stream) => return Ok(stream),
                    Err(err) => {
                        let is_retryable =
                            err.kind.is_transient() && attempt < self.retry_config.max_retries;
                        if is_retryable {
                            let delay = self.backoff_delay(attempt);
                            #[allow(clippy::cast_possible_truncation)]
                            let delay_ms = delay.as_millis() as u64;
                            warn!(
                                provider = provider.name(),
                                attempt,
                                error = %err,
                                delay_ms,
                                "fallback: transient stream error, retrying after backoff"
                            );
                            time::sleep(delay).await;
                        } else {
                            warn!(
                                provider = provider.name(),
                                error = %err,
                                "fallback: provider stream failed, trying next"
                            );
                            last_error = err;
                            break;
                        }
                    }
                }
            }
        }

        Err(last_error)
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        for provider in &self.providers {
            if matches!(provider.health_check().await, Ok(true)) {
                return Ok(true);
            }
        }
        Ok(false)
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
        healthy: bool,
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
                healthy: true,
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
                healthy: false,
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
                healthy: true,
            }
        }
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
            Ok(self.healthy)
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
            healthy: true,
        };
        let b = TestProvider {
            provider_name: "b",
            display: "B",
            caps: LlmCapabilities::text_only(),
            models: vec!["shared-model".to_owned(), "b-only".to_owned()],
            responses: Mutex::new(vec![]),
            call_count: AtomicU32::new(0),
            healthy: true,
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
