// ABOUTME: Pluggable guardrail middleware for pre-request and post-response validation
// ABOUTME: Includes content length, topic filter, and basic PII detection guardrails
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Guardrail Middleware
//!
//! [`GuardrailProvider`] wraps an inner `Box<dyn LlmProvider>` and runs
//! pre-request and post-response checks via pluggable [`Guardrail`] trait
//! implementations. If any guardrail rejects the request or response, a
//! [`RunnerError`] with `ErrorKind::Guardrail` is returned.
//!
//! ## Built-in Guardrails
//!
//! - [`ContentLengthGuardrail`] — rejects oversized messages
//! - [`TopicFilterGuardrail`] — blocks messages containing specified patterns
//! - [`PiiScrubGuardrail`] — detects basic email/phone patterns (not production-grade)
//!
//! ## Limitations
//!
//! - `complete_stream()` runs pre-request checks but skips post-response checks,
//!   since the full content is not available upfront.

use async_trait::async_trait;
use tracing::warn;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// A guardrail violation with details about which guardrail and why
#[derive(Debug, Clone)]
pub struct GuardrailViolation {
    /// Name of the guardrail that triggered
    pub guardrail_name: String,
    /// Explanation of why the check failed
    pub reason: String,
}

impl std::fmt::Display for GuardrailViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.guardrail_name, self.reason)
    }
}

impl From<GuardrailViolation> for RunnerError {
    fn from(violation: GuardrailViolation) -> Self {
        Self::guardrail(violation.to_string())
    }
}

/// Trait for synchronous pre/post request validation.
///
/// Guardrails run fast, in-memory checks. They must be `Send + Sync`
/// to work with async providers, but the checks themselves are synchronous.
pub trait Guardrail: Send + Sync {
    /// Human-readable name for this guardrail
    fn name(&self) -> &str;

    /// Validate a request before it reaches the provider.
    ///
    /// Return `Ok(())` to allow the request, or `Err(GuardrailViolation)` to reject it.
    fn check_request(&self, request: &ChatRequest) -> Result<(), GuardrailViolation>;

    /// Validate a response after it comes back from the provider.
    ///
    /// Return `Ok(())` to allow the response, or `Err(GuardrailViolation)` to reject it.
    fn check_response(
        &self,
        request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<(), GuardrailViolation>;
}

/// Provider wrapper that applies guardrail checks before and after LLM calls.
pub struct GuardrailProvider {
    inner: Box<dyn LlmProvider>,
    guardrails: Vec<Box<dyn Guardrail>>,
}

impl GuardrailProvider {
    /// Wrap a provider with guardrail validation
    pub fn new(inner: Box<dyn LlmProvider>, guardrails: Vec<Box<dyn Guardrail>>) -> Self {
        Self { inner, guardrails }
    }

    /// Run all pre-request guardrail checks
    fn check_all_requests(&self, request: &ChatRequest) -> Result<(), RunnerError> {
        for guardrail in &self.guardrails {
            guardrail.check_request(request).map_err(|violation| {
                warn!(
                    guardrail = guardrail.name(),
                    reason = %violation.reason,
                    "guardrail: pre-request check failed"
                );
                RunnerError::from(violation)
            })?;
        }
        Ok(())
    }

    /// Run all post-response guardrail checks
    fn check_all_responses(
        &self,
        request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<(), RunnerError> {
        for guardrail in &self.guardrails {
            guardrail
                .check_response(request, response)
                .map_err(|violation| {
                    warn!(
                        guardrail = guardrail.name(),
                        reason = %violation.reason,
                        "guardrail: post-response check failed"
                    );
                    RunnerError::from(violation)
                })?;
        }
        Ok(())
    }
}

#[async_trait]
impl LlmProvider for GuardrailProvider {
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
        self.check_all_requests(request)?;
        let response = self.inner.complete(request).await?;
        self.check_all_responses(request, &response)?;
        Ok(response)
    }

    /// Pre-request guardrails run; post-response checks are skipped (documented limitation)
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.check_all_requests(request)?;
        self.inner.complete_stream(request).await
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.inner.health_check().await
    }
}

// ============================================================================
// Built-in Guardrails
// ============================================================================

/// Rejects messages that exceed configured character limits.
pub struct ContentLengthGuardrail {
    /// Maximum characters allowed in a single message
    pub max_message_chars: usize,
    /// Maximum total characters across all messages in a request
    pub max_total_chars: usize,
}

impl Guardrail for ContentLengthGuardrail {
    fn name(&self) -> &str {
        "content_length"
    }

    fn check_request(&self, request: &ChatRequest) -> Result<(), GuardrailViolation> {
        let mut total = 0;
        for msg in &request.messages {
            if msg.content.len() > self.max_message_chars {
                return Err(GuardrailViolation {
                    guardrail_name: self.name().to_owned(),
                    reason: format!(
                        "message exceeds max length ({} > {} chars)",
                        msg.content.len(),
                        self.max_message_chars
                    ),
                });
            }
            total += msg.content.len();
        }
        if total > self.max_total_chars {
            return Err(GuardrailViolation {
                guardrail_name: self.name().to_owned(),
                reason: format!(
                    "total content exceeds max length ({total} > {} chars)",
                    self.max_total_chars
                ),
            });
        }
        Ok(())
    }

    fn check_response(
        &self,
        _request: &ChatRequest,
        _response: &ChatResponse,
    ) -> Result<(), GuardrailViolation> {
        Ok(())
    }
}

/// Blocks messages containing specified topic patterns (case-insensitive substring match).
pub struct TopicFilterGuardrail {
    /// Patterns to block (matched case-insensitively as substrings)
    pub blocked_patterns: Vec<String>,
}

impl Guardrail for TopicFilterGuardrail {
    fn name(&self) -> &str {
        "topic_filter"
    }

    fn check_request(&self, request: &ChatRequest) -> Result<(), GuardrailViolation> {
        for msg in &request.messages {
            let content_lower = msg.content.to_lowercase();
            for pattern in &self.blocked_patterns {
                if content_lower.contains(&pattern.to_lowercase()) {
                    return Err(GuardrailViolation {
                        guardrail_name: self.name().to_owned(),
                        reason: format!("blocked topic detected: \"{pattern}\""),
                    });
                }
            }
        }
        Ok(())
    }

    fn check_response(
        &self,
        _request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<(), GuardrailViolation> {
        let content_lower = response.content.to_lowercase();
        for pattern in &self.blocked_patterns {
            if content_lower.contains(&pattern.to_lowercase()) {
                return Err(GuardrailViolation {
                    guardrail_name: self.name().to_owned(),
                    reason: format!("blocked topic detected in response: \"{pattern}\""),
                });
            }
        }
        Ok(())
    }
}

/// Detects basic email and phone number patterns in messages.
///
/// This uses hand-rolled character scanning, not regex. It catches common
/// patterns but is not intended for production-grade PII detection.
pub struct PiiScrubGuardrail {
    /// Whether to check for email-like patterns (contains `@` with surrounding word chars)
    pub check_email: bool,
    /// Whether to check for phone-like patterns (sequences of 7+ digits with optional separators)
    pub check_phone: bool,
}

impl PiiScrubGuardrail {
    /// Scan text for email-like patterns: word chars, then @, then word chars with dots
    fn contains_email(text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            if c == '@' {
                // Check for word chars before @
                let has_before = i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '.');
                // Check for word chars after @
                let has_after =
                    i + 1 < chars.len() && (chars[i + 1].is_alphanumeric() || chars[i + 1] == '.');
                if has_before && has_after {
                    return true;
                }
            }
        }
        false
    }

    /// Scan text for phone-like patterns: 7 or more digits, possibly separated by
    /// dashes, spaces, dots, or parentheses
    fn contains_phone(text: &str) -> bool {
        let mut digit_count = 0;
        let mut in_sequence = false;
        for c in text.chars() {
            if c.is_ascii_digit() {
                digit_count += 1;
                in_sequence = true;
            } else if in_sequence && (c == '-' || c == ' ' || c == '.' || c == '(' || c == ')') {
                // Allow separators within a phone sequence
            } else {
                if digit_count >= 7 {
                    return true;
                }
                digit_count = 0;
                in_sequence = false;
            }
        }
        digit_count >= 7
    }
}

impl Guardrail for PiiScrubGuardrail {
    fn name(&self) -> &str {
        "pii_scrub"
    }

    fn check_request(&self, request: &ChatRequest) -> Result<(), GuardrailViolation> {
        for msg in &request.messages {
            if self.check_email && Self::contains_email(&msg.content) {
                return Err(GuardrailViolation {
                    guardrail_name: self.name().to_owned(),
                    reason: "email address detected in request".to_owned(),
                });
            }
            if self.check_phone && Self::contains_phone(&msg.content) {
                return Err(GuardrailViolation {
                    guardrail_name: self.name().to_owned(),
                    reason: "phone number detected in request".to_owned(),
                });
            }
        }
        Ok(())
    }

    fn check_response(
        &self,
        _request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<(), GuardrailViolation> {
        if self.check_email && Self::contains_email(&response.content) {
            return Err(GuardrailViolation {
                guardrail_name: self.name().to_owned(),
                reason: "email address detected in response".to_owned(),
            });
        }
        if self.check_phone && Self::contains_phone(&response.content) {
            return Err(GuardrailViolation {
                guardrail_name: self.name().to_owned(),
                reason: "phone number detected in response".to_owned(),
            });
        }
        Ok(())
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
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
    }

    impl TestProvider {
        fn ok(content: &str) -> Self {
            Self {
                responses: Mutex::new(vec![Ok(ChatResponse {
                    content: content.to_owned(),
                    model: "test-model".to_owned(),
                    usage: None,
                    finish_reason: Some("stop".to_owned()),
                    warnings: None,
                    tool_calls: None,
                })]),
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

    // ========================================================================
    // ContentLengthGuardrail tests
    // ========================================================================

    #[test]
    fn content_length_rejects_long_message() {
        let guard = ContentLengthGuardrail {
            max_message_chars: 10,
            max_total_chars: 1000,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("a".repeat(11))]);
        let result = guard.check_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("max length"));
    }

    #[test]
    fn content_length_rejects_total_overflow() {
        let guard = ContentLengthGuardrail {
            max_message_chars: 100,
            max_total_chars: 15,
        };
        let request = ChatRequest::new(vec![
            ChatMessage::user("abcdefgh"), // 8
            ChatMessage::user("ijklmnop"), // 8 -> total 16 > 15
        ]);
        let result = guard.check_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("total content"));
    }

    #[test]
    fn content_length_accepts_within_limits() {
        let guard = ContentLengthGuardrail {
            max_message_chars: 100,
            max_total_chars: 1000,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        assert!(guard.check_request(&request).is_ok());
    }

    #[test]
    fn content_length_response_passthrough() {
        let guard = ContentLengthGuardrail {
            max_message_chars: 5,
            max_total_chars: 5,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        let response = ChatResponse {
            content: "a very long response that exceeds limits".to_owned(),
            model: "m".to_owned(),
            usage: None,
            finish_reason: None,
            warnings: None,
            tool_calls: None,
        };
        // ContentLengthGuardrail does not check responses
        assert!(guard.check_response(&request, &response).is_ok());
    }

    // ========================================================================
    // TopicFilterGuardrail tests
    // ========================================================================

    #[test]
    fn topic_filter_blocks_request() {
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["forbidden".to_owned()],
        };
        let request = ChatRequest::new(vec![ChatMessage::user("this is forbidden content")]);
        let result = guard.check_request(&request);
        assert!(result.is_err());
        assert!(result.unwrap_err().reason.contains("forbidden"));
    }

    #[test]
    fn topic_filter_case_insensitive() {
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["SECRET".to_owned()],
        };
        let request = ChatRequest::new(vec![ChatMessage::user("tell me the secret")]);
        assert!(guard.check_request(&request).is_err());
    }

    #[test]
    fn topic_filter_allows_clean_content() {
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["forbidden".to_owned()],
        };
        let request = ChatRequest::new(vec![ChatMessage::user("hello world")]);
        assert!(guard.check_request(&request).is_ok());
    }

    #[test]
    fn topic_filter_blocks_response() {
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["classified".to_owned()],
        };
        let request = ChatRequest::new(vec![ChatMessage::user("tell me")]);
        let response = ChatResponse {
            content: "this information is classified".to_owned(),
            model: "m".to_owned(),
            usage: None,
            finish_reason: None,
            warnings: None,
            tool_calls: None,
        };
        assert!(guard.check_response(&request, &response).is_err());
    }

    #[test]
    fn topic_filter_multiple_patterns() {
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["alpha".to_owned(), "beta".to_owned()],
        };
        let request = ChatRequest::new(vec![ChatMessage::user("the beta release")]);
        assert!(guard.check_request(&request).is_err());
    }

    // ========================================================================
    // PiiScrubGuardrail tests
    // ========================================================================

    #[test]
    fn pii_detects_email() {
        let guard = PiiScrubGuardrail {
            check_email: true,
            check_phone: false,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("contact me at user@example.com")]);
        assert!(guard.check_request(&request).is_err());
    }

    #[test]
    fn pii_detects_phone() {
        let guard = PiiScrubGuardrail {
            check_email: false,
            check_phone: true,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("call me at 555-123-4567 please")]);
        assert!(guard.check_request(&request).is_err());
    }

    #[test]
    fn pii_allows_clean_content() {
        let guard = PiiScrubGuardrail {
            check_email: true,
            check_phone: true,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("hello world 42")]);
        assert!(guard.check_request(&request).is_ok());
    }

    #[test]
    fn pii_detects_email_in_response() {
        let guard = PiiScrubGuardrail {
            check_email: true,
            check_phone: false,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);
        let response = ChatResponse {
            content: "reach out to admin@corp.io".to_owned(),
            model: "m".to_owned(),
            usage: None,
            finish_reason: None,
            warnings: None,
            tool_calls: None,
        };
        assert!(guard.check_response(&request, &response).is_err());
    }

    #[test]
    fn pii_phone_without_separators() {
        let guard = PiiScrubGuardrail {
            check_email: false,
            check_phone: true,
        };
        let request = ChatRequest::new(vec![ChatMessage::user("number is 5551234567")]);
        assert!(guard.check_request(&request).is_err());
    }

    // ========================================================================
    // GuardrailProvider integration tests
    // ========================================================================

    #[tokio::test]
    async fn pre_request_rejection_prevents_inner_call() {
        let provider = TestProvider::ok("should not reach");
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["blocked".to_owned()],
        };
        let guarded = GuardrailProvider::new(Box::new(provider), vec![Box::new(guard)]);
        let request = ChatRequest::new(vec![ChatMessage::user("this is blocked")]);

        let err = guarded.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Guardrail);
        assert!(err.message.contains("blocked"));
    }

    #[tokio::test]
    async fn post_response_rejection() {
        let provider = TestProvider::ok("this contains classified info");
        let guard = TopicFilterGuardrail {
            blocked_patterns: vec!["classified".to_owned()],
        };
        let guarded = GuardrailProvider::new(Box::new(provider), vec![Box::new(guard)]);
        let request = ChatRequest::new(vec![ChatMessage::user("tell me something")]);

        let err = guarded.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Guardrail);
        assert!(err.message.contains("classified"));
    }

    #[tokio::test]
    async fn multiple_guardrails() {
        let provider = TestProvider::ok("safe response");
        let length_guard = ContentLengthGuardrail {
            max_message_chars: 5,
            max_total_chars: 100,
        };
        let topic_guard = TopicFilterGuardrail {
            blocked_patterns: vec!["forbidden".to_owned()],
        };
        let guarded = GuardrailProvider::new(
            Box::new(provider),
            vec![Box::new(length_guard), Box::new(topic_guard)],
        );

        // Length guardrail triggers first
        let request = ChatRequest::new(vec![ChatMessage::user("this is too long")]);
        let err = guarded.complete(&request).await.unwrap_err();
        assert_eq!(err.kind, ErrorKind::Guardrail);
        assert!(err.message.contains("content_length"));
    }

    #[tokio::test]
    async fn empty_guardrails_passthrough() {
        let provider = TestProvider::ok("hello");
        let guarded = GuardrailProvider::new(Box::new(provider), vec![]);
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let response = guarded.complete(&request).await.expect("should pass");
        assert_eq!(response.content, "hello");
    }

    #[tokio::test]
    async fn delegates_trait_methods() {
        let provider = TestProvider::ok("hello");
        let guarded = GuardrailProvider::new(Box::new(provider), vec![]);

        assert_eq!(guarded.name(), "test");
        assert_eq!(guarded.display_name(), "Test Provider");
        assert_eq!(guarded.default_model(), "test-model");
        assert!(guarded.capabilities().supports_streaming());

        let healthy = guarded.health_check().await.expect("health check");
        assert!(healthy);
    }
}
