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
//! Health checks pass if ANY provider is healthy. Capabilities are the
//! bitwise OR of all inner providers.

use std::any::Any;

use async_trait::async_trait;
use tracing::warn;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// Provider that tries multiple inner providers in order, returning the first success.
///
/// # Construction
///
/// Use [`FallbackProvider::new()`] with a non-empty `Vec` of providers.
/// An empty vec is rejected with a config error.
pub struct FallbackProvider {
    providers: Vec<Box<dyn LlmProvider>>,
    display_name: &'static str,
    combined_models: Vec<String>,
}

impl FallbackProvider {
    /// Create a fallback chain from a non-empty list of providers.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` if `providers` is empty.
    pub fn new(providers: Vec<Box<dyn LlmProvider>>) -> Result<Self, RunnerError> {
        if providers.is_empty() {
            return Err(RunnerError::config(
                "FallbackProvider requires at least one provider",
            ));
        }

        let names: Vec<&str> = providers.iter().map(|p| p.name()).collect();
        // Leak once at construction so display_name() can return &'static str
        // without allocating per call. FallbackProvider is typically long-lived.
        let display_name: &'static str =
            Box::leak(format!("Fallback ({})", names.join(", ")).into_boxed_str());

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
        })
    }
}

#[async_trait]
impl LlmProvider for FallbackProvider {
    fn name(&self) -> &'static str {
        "fallback"
    }

    fn display_name(&self) -> &'static str {
        self.display_name
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
            match provider.complete(request).await {
                Ok(response) => return Ok(response),
                Err(err) => {
                    warn!(
                        provider = provider.name(),
                        error = %err,
                        "fallback: provider failed, trying next"
                    );
                    last_error = err;
                }
            }
        }

        Err(last_error)
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let mut last_error = RunnerError::internal("no providers configured");

        for provider in &self.providers {
            match provider.complete_stream(request).await {
                Ok(stream) => return Ok(stream),
                Err(err) => {
                    warn!(
                        provider = provider.name(),
                        error = %err,
                        "fallback: provider stream failed, trying next"
                    );
                    last_error = err;
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
        RunnerError,
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
                })]),
                call_count: AtomicU32::new(0),
                healthy: true,
            }
        }

        fn failing(name: &'static str) -> Self {
            Self {
                provider_name: name,
                display: name,
                caps: LlmCapabilities::FUNCTION_CALLING,
                models: vec![format!("{name}-model")],
                responses: Mutex::new(vec![Err(RunnerError::external_service(name, "down"))]),
                call_count: AtomicU32::new(0),
                healthy: false,
            }
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            self.provider_name
        }
        fn display_name(&self) -> &'static str {
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
            let mut responses = self.responses.lock().expect("test lock");
            if responses.is_empty() {
                Err(RunnerError::internal("no more responses"))
            } else {
                responses.remove(0)
            }
        }
        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("streaming not supported in test"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(self.healthy)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[tokio::test]
    async fn single_provider_passthrough() {
        let providers: Vec<Box<dyn LlmProvider>> =
            vec![Box::new(TestProvider::ok("claude", "hello"))];
        let fallback = FallbackProvider::new(providers).expect("non-empty");
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback.complete(&request).await.expect("should succeed");
        assert_eq!(response.content, "hello");
    }

    #[tokio::test]
    async fn first_fails_second_succeeds() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("primary")),
            Box::new(TestProvider::ok("secondary", "fallback response")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty");
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = fallback
            .complete(&request)
            .await
            .expect("second should work");
        assert_eq!(response.content, "fallback response");
    }

    #[tokio::test]
    async fn all_fail_returns_last_error() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("first")),
            Box::new(TestProvider::failing("second")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty");
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
        let fallback = FallbackProvider::new(providers).expect("non-empty");

        let healthy = fallback.health_check().await.expect("health check");
        assert!(healthy);
    }

    #[tokio::test]
    async fn health_all_down() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::failing("a")),
            Box::new(TestProvider::failing("b")),
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty");

        let healthy = fallback.health_check().await.expect("health check");
        assert!(!healthy);
    }

    #[test]
    fn capabilities_union() {
        let providers: Vec<Box<dyn LlmProvider>> = vec![
            Box::new(TestProvider::ok("a", "ok")), // text_only = STREAMING | SYSTEM_MESSAGES
            Box::new(TestProvider::failing("b")),  // FUNCTION_CALLING
        ];
        let fallback = FallbackProvider::new(providers).expect("non-empty");

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
        let fallback = FallbackProvider::new(providers).expect("non-empty");

        let models = fallback.available_models();
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"shared-model".to_owned()));
        assert!(models.contains(&"a-only".to_owned()));
        assert!(models.contains(&"b-only".to_owned()));
    }
}
