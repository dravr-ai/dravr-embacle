// ABOUTME: Response quality gate that validates LLM output and retries on failure
// ABOUTME: Checks for empty responses, minimum length, and refusal patterns
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Response Quality Gate
//!
//! [`QualityGateProvider`] wraps an inner `Box<dyn LlmProvider>` and validates
//! each response against a configurable [`QualityPolicy`]. If validation fails,
//! the request is retried with feedback appended to help the provider correct
//! its output. After exhausting retries, the last response is returned with
//! `finish_reason` set to `"quality_gate_exhausted"`.
//!
//! ## Limitations
//!
//! - Streaming responses (`complete_stream`) are passed through without quality
//!   checking, since the full content is not available upfront.

use std::fmt;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// Default refusal patterns matched case-insensitively via substring
const DEFAULT_REFUSAL_PATTERNS: &[&str] = &["I cannot", "I can't", "As an AI"];

/// Configurable policy for response quality validation
#[derive(Debug, Clone)]
pub struct QualityPolicy {
    /// Maximum number of retry attempts after quality failure
    pub max_retries: u32,
    /// Substring patterns indicating a refusal (matched case-insensitively)
    pub refusal_patterns: Vec<String>,
    /// Minimum acceptable content length in characters
    pub min_content_length: usize,
    /// Whether empty responses are rejected
    pub require_non_empty: bool,
}

impl Default for QualityPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            refusal_patterns: DEFAULT_REFUSAL_PATTERNS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            min_content_length: 1,
            require_non_empty: true,
        }
    }
}

/// Reason a response failed quality validation
#[derive(Debug, Clone)]
pub enum QualityFailure {
    /// Response content was empty
    Empty,
    /// Response was shorter than the policy minimum
    TooShort {
        /// Actual content length
        length: usize,
        /// Required minimum
        minimum: usize,
    },
    /// Response matched a refusal pattern
    RefusalDetected {
        /// The pattern that matched
        pattern: String,
    },
}

impl fmt::Display for QualityFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "response was empty"),
            Self::TooShort { length, minimum } => {
                write!(f, "response too short ({length} < {minimum} chars)")
            }
            Self::RefusalDetected { pattern } => {
                write!(f, "refusal detected (matched \"{pattern}\")")
            }
        }
    }
}

/// Validate response content against a quality policy.
///
/// Returns `None` if the response passes all checks, or `Some(QualityFailure)`
/// describing the first failing check.
fn validate_response(content: &str, policy: &QualityPolicy) -> Option<QualityFailure> {
    if policy.require_non_empty && content.is_empty() {
        return Some(QualityFailure::Empty);
    }

    if content.len() < policy.min_content_length {
        return Some(QualityFailure::TooShort {
            length: content.len(),
            minimum: policy.min_content_length,
        });
    }

    let content_lower = content.to_lowercase();
    for pattern in &policy.refusal_patterns {
        if content_lower.contains(&pattern.to_lowercase()) {
            return Some(QualityFailure::RefusalDetected {
                pattern: pattern.clone(),
            });
        }
    }

    None
}

/// Wrapper implementing `LlmProvider` that validates responses and retries on failure.
///
/// # Usage
///
/// ```rust,no_run
/// # use embacle::quality_gate::{QualityGateProvider, QualityPolicy};
/// # use embacle::types::LlmProvider;
/// # fn example(provider: Box<dyn LlmProvider>) {
/// let policy = QualityPolicy::default();
/// let guarded = QualityGateProvider::new(provider, policy);
/// // guarded.complete() will retry on empty/short/refused responses
/// # }
/// ```
pub struct QualityGateProvider {
    inner: Box<dyn LlmProvider>,
    policy: QualityPolicy,
}

impl QualityGateProvider {
    /// Wrap a provider with quality gate validation
    pub fn new(inner: Box<dyn LlmProvider>, policy: QualityPolicy) -> Self {
        Self { inner, policy }
    }
}

#[async_trait]
impl LlmProvider for QualityGateProvider {
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
        let mut messages = request.messages.clone();
        let mut last_response = self.inner.complete(request).await?;

        for retry in 0..self.policy.max_retries {
            match validate_response(&last_response.content, &self.policy) {
                None => return Ok(last_response),
                Some(failure) => {
                    warn!(
                        provider = self.inner.name(),
                        retry,
                        failure = %failure,
                        "quality gate: validation failed, retrying"
                    );

                    // Append feedback so the provider can correct its output
                    messages.push(ChatMessage::assistant(last_response.content.clone()));
                    messages.push(ChatMessage::user(format!(
                        "Your previous response did not meet quality requirements: {failure}. \
                         Please provide a substantive, helpful response."
                    )));

                    let retry_request = ChatRequest {
                        messages: messages.clone(),
                        model: request.model.clone(),
                        temperature: request.temperature,
                        max_tokens: request.max_tokens,
                        stream: false,
                        tools: request.tools.clone(),
                        tool_choice: request.tool_choice.clone(),
                        top_p: request.top_p,
                        stop: request.stop.clone(),
                        response_format: request.response_format.clone(),
                        turn_id: request.turn_id,
                    };

                    last_response = self.inner.complete(&retry_request).await?;
                }
            }
        }

        // Final validation after exhausting retries
        if let Some(failure) = validate_response(&last_response.content, &self.policy) {
            info!(
                provider = self.inner.name(),
                failure = %failure,
                "quality gate: exhausted retries, returning last response"
            );
            last_response.finish_reason = Some("quality_gate_exhausted".to_owned());
        }

        Ok(last_response)
    }

    /// Delegate streaming directly (no quality check on streams — documented limitation)
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.inner.complete_stream(request).await
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.inner.health_check().await
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
                    content: "valid response".to_owned(),
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
            Err(RunnerError::internal("not supported"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
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

    #[test]
    fn validate_empty_fails() {
        let policy = QualityPolicy::default();
        let failure = validate_response("", &policy);
        assert!(matches!(failure, Some(QualityFailure::Empty)));
    }

    #[test]
    fn validate_too_short_fails() {
        let policy = QualityPolicy {
            min_content_length: 10,
            ..QualityPolicy::default()
        };
        let failure = validate_response("short", &policy);
        assert!(matches!(
            failure,
            Some(QualityFailure::TooShort {
                length: 5,
                minimum: 10
            })
        ));
    }

    #[test]
    fn validate_refusal_detected_case_insensitive() {
        let policy = QualityPolicy::default();
        let failure = validate_response("i cannot do that for you", &policy);
        assert!(matches!(
            failure,
            Some(QualityFailure::RefusalDetected { .. })
        ));

        let failure2 = validate_response("as an ai language model, I should clarify", &policy);
        assert!(matches!(
            failure2,
            Some(QualityFailure::RefusalDetected { .. })
        ));
    }

    #[test]
    fn validate_valid_passes() {
        let policy = QualityPolicy::default();
        let failure = validate_response("This is a perfectly valid and helpful response.", &policy);
        assert!(failure.is_none());
    }

    #[tokio::test]
    async fn retry_loop_with_eventual_success() {
        let provider = TestProvider::new(vec![
            Ok(make_response("")),                             // empty — fail
            Ok(make_response("")),                             // empty — fail
            Ok(make_response("valid helpful response here!")), // pass
        ]);
        let policy = QualityPolicy {
            max_retries: 2,
            ..QualityPolicy::default()
        };
        let guarded = QualityGateProvider::new(Box::new(provider), policy);
        let request = ChatRequest::new(vec![ChatMessage::user("help me")]);

        let response = guarded.complete(&request).await.expect("should succeed"); // Safe: test assertion
        assert_eq!(response.content, "valid helpful response here!");
        assert_eq!(response.finish_reason, Some("stop".to_owned()));
    }

    #[tokio::test]
    async fn exhaustion_sets_finish_reason() {
        let provider = TestProvider::new(vec![
            Ok(make_response("")), // empty — fail
            Ok(make_response("")), // retry 1 — fail
            Ok(make_response("")), // retry 2 — fail
        ]);
        let policy = QualityPolicy {
            max_retries: 2,
            ..QualityPolicy::default()
        };
        let guarded = QualityGateProvider::new(Box::new(provider), policy);
        let request = ChatRequest::new(vec![ChatMessage::user("help me")]);

        let response = guarded
            .complete(&request)
            .await
            .expect("should return last"); // Safe: test assertion
        assert_eq!(
            response.finish_reason,
            Some("quality_gate_exhausted".to_owned())
        );
    }
}
