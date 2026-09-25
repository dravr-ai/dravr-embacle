// ABOUTME: Groq provider — OpenAI-shaped chat completions on Groq's LPU inference, with streaming and tool calling
// ABOUTME: Retries 429/502/503 and transport errors in place; a 429 carries Groq's "try again in Ns" wait
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Groq Provider
//!
//! [`LlmProvider`] over Groq's LPU-accelerated inference, through its
//! OpenAI-compatible chat completions API.
//!
//! ## Configuration
//!
//! [`GroqConfig::from_env`] reads:
//!
//! - `GROQ_API_KEY`: required, from the Groq console
//! - `GROQ_DEFAULT_MODEL`: default `llama-3.3-70b-versatile`
//! - `GROQ_MAX_RETRIES`, `GROQ_INITIAL_RETRY_DELAY_MS`,
//!   `GROQ_MAX_RETRY_DELAY_MS`: retry tuning (defaults 3 / 500 / 5000)
//!
//! ## Rate limits
//!
//! The free tier has a 12,000 tokens-per-minute limit. A 429 is reported as
//! [`ErrorKind::RateLimit`](crate::types::ErrorKind::RateLimit) carrying
//! Groq's `Please try again in Ns` wait, and is retried in place with
//! backoff before it propagates.
//!
//! ## Supported Models
//!
//! - `llama-3.3-70b-versatile` (default): High-quality general purpose
//! - `llama-3.1-8b-instant`: Fast responses for simple tasks
//! - `mixtral-8x7b-32768`: Long context window (32K tokens)

use std::env;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{RequestBuilder, StatusCode};
use tracing::{debug, info, instrument, warn};

use super::chat_completions::{self, CompletionRequest, Usage};
use super::client::{self, with_retries, AttemptError, HttpRetryConfig, DEFAULT_TIMEOUT_SECS};
use super::sse::create_sse_stream;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// The name this provider reports, and the price-table key its usage bills under
const PROVIDER_NAME: &str = "groq";

/// Environment variable for the API key
const GROQ_API_KEY_ENV: &str = "GROQ_API_KEY";

/// Environment variable for the default model
const GROQ_DEFAULT_MODEL_ENV: &str = "GROQ_DEFAULT_MODEL";

/// Prefix of the retry-tuning environment variables
const GROQ_ENV_PREFIX: &str = "GROQ";

/// Default model when neither the config nor the environment names one
const DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";

/// Available Groq models
const AVAILABLE_MODELS: &[&str] = &[
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
];

/// Base URL for the Groq API (OpenAI-compatible)
const API_BASE_URL: &str = "https://api.groq.com/openai/v1";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Groq provider.
#[derive(Clone)]
pub struct GroqConfig {
    /// API key from the Groq console. Redacted in `Debug`.
    pub api_key: String,
    /// Model used when the request names none
    pub model: String,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
    /// Retry policy for transient failures (429, 502, 503, network errors)
    pub retry: HttpRetryConfig,
}

impl GroqConfig {
    /// A configuration with the given key and every other field at its default.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            retry: HttpRetryConfig::default(),
        }
    }

    /// Read the configuration from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when `GROQ_API_KEY` is
    /// not set.
    pub fn from_env() -> Result<Self, RunnerError> {
        let api_key = env::var(GROQ_API_KEY_ENV).map_err(|_| {
            RunnerError::config(format!(
                "Missing {GROQ_API_KEY_ENV} environment variable. Get your API key from https://console.groq.com/keys"
            ))
        })?;
        let model = env::var(GROQ_DEFAULT_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        Ok(Self {
            model,
            retry: HttpRetryConfig::from_env(GROQ_ENV_PREFIX),
            ..Self::new(api_key)
        })
    }

    /// Set the default model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the retry policy
    #[must_use]
    pub fn with_retry(mut self, retry: HttpRetryConfig) -> Self {
        self.retry = retry;
        self
    }
}

impl Debug for GroqConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("GroqConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .finish()
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

/// Groq LLM provider using LPU-accelerated inference
///
/// Provides access to open-source models (Llama, Mixtral) with
/// extremely fast inference speeds via Groq's Language Processing Units.
pub struct GroqProvider {
    config: GroqConfig,
    client: reqwest::Client,
    available_models: Vec<String>,
}

impl GroqProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: GroqConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: GroqConfig, client: reqwest::Client) -> Self {
        info!(
            default_model = %config.model,
            max_retries = config.retry.max_retries,
            initial_delay_ms = config.retry.initial_delay_ms,
            "Groq provider initialized"
        );
        Self {
            config,
            client,
            available_models: AVAILABLE_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Override the default model after construction.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Build the API URL for a given endpoint
    fn api_url(endpoint: &str) -> String {
        format!("{API_BASE_URL}/{endpoint}")
    }

    /// Parse an error response from the Groq API: the vendor's message when
    /// the body is its JSON envelope, `HTTP <status>` otherwise; the status
    /// decides the kind.
    fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
        let message = chat_completions::describe_error(PROVIDER_NAME, status, body);
        client::map_http_error(PROVIDER_NAME, status, &message)
    }

    /// Build an authenticated HTTP request to the Groq API
    fn build_request(&self, body: &CompletionRequest) -> RequestBuilder {
        self.client
            .post(Self::api_url("chat/completions"))
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(body)
    }

    /// Build the request body for a `ChatRequest`
    fn build_body(&self, request: &ChatRequest, stream: bool) -> CompletionRequest {
        CompletionRequest::new(request, &self.config.model, stream, self.capabilities())
    }

    /// One non-streaming attempt
    async fn attempt_complete(
        &self,
        body: &CompletionRequest,
    ) -> Result<ChatResponse, AttemptError> {
        let response = self
            .build_request(body)
            .send()
            .await
            .map_err(|e| AttemptError::from_send(PROVIDER_NAME, e))?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            AttemptError::transient(RunnerError::external_service(
                PROVIDER_NAME,
                format!("Failed to read response: {e}"),
            ))
        })?;

        if !status.is_success() {
            return Err(AttemptError::from_status(
                status,
                Self::parse_error_response(status, &text),
            ));
        }

        Self::read_completion(&text).map_err(AttemptError::permanent)
    }

    /// Read a completion body. Groq bills the plain counts: the usage
    /// breakdown is not read.
    fn read_completion(body: &str) -> Result<ChatResponse, RunnerError> {
        chat_completions::parse_response(PROVIDER_NAME, body, Usage::counts_only)
    }

    /// One attempt at opening the stream
    async fn attempt_stream(&self, body: &CompletionRequest) -> Result<ChatStream, AttemptError> {
        let response = self
            .build_request(body)
            .send()
            .await
            .map_err(|e| AttemptError::from_send(PROVIDER_NAME, e))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(AttemptError::from_status(
                status,
                Self::parse_error_response(status, &text),
            ));
        }

        Ok(create_sse_stream(
            response.bytes_stream(),
            |json| chat_completions::parse_stream_frame(PROVIDER_NAME, json),
            PROVIDER_NAME,
        ))
    }
}

#[async_trait]
impl LlmProvider for GroqProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> &str {
        "Groq (Llama/Mixtral)"
    }

    fn capabilities(&self) -> LlmCapabilities {
        // Groq supports streaming, function calling, and system messages
        // but does not support vision
        LlmCapabilities::STREAMING
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::JSON_MODE
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let body = self.build_body(request, false);
        let body = &body;
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Groq request",
            |attempt| async move {
                debug!(attempt, "Sending chat completion request to Groq");
                self.attempt_complete(body).await
            },
        )
        .await
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let body = self.build_body(request, true);
        let body = &body;
        // Retry the initial HTTP request (not the stream itself)
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Groq streaming request",
            |attempt| async move {
                debug!(attempt, "Sending streaming chat completion request to Groq");
                self.attempt_stream(body).await
            },
        )
        .await
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        debug!("Performing Groq API health check");
        // The models endpoint is a lightweight authenticated probe
        let response = self
            .client
            .get(Self::api_url("models"))
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .map_err(|e| client::map_send_error(PROVIDER_NAME, e))?;

        let healthy = response.status().is_success();
        if healthy {
            debug!("Groq API health check passed");
        } else {
            warn!(
                status = response.status().as_u16(),
                "Groq API health check failed"
            );
        }
        Ok(healthy)
    }
}

impl Debug for GroqProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("GroqProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ErrorKind, ImagePart, ResponseFormat, ToolCallRequest, ToolChoice,
        ToolDefinition,
    };
    use serde_json::json;

    #[test]
    fn a_429_is_a_rate_limit_carrying_the_wait() {
        let body = r#"{"error":{"message":"Rate limit reached for model x in organization y on tokens per minute (TPM): Limit 12000, Used 11000, Requested 2000. Please try again in 2.5s.","type":"tokens"}}"#;
        let err = GroqProvider::parse_error_response(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("3 seconds"), "{}", err.message);
    }

    #[test]
    fn a_400_is_an_invalid_request_a_401_an_auth_failure() {
        let body = r#"{"error":{"message":"'messages' must not be empty","type":"invalid_request_error"}}"#;
        let err = GroqProvider::parse_error_response(StatusCode::BAD_REQUEST, body);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(err.message.contains("must not be empty"));
        let err = GroqProvider::parse_error_response(StatusCode::UNAUTHORIZED, "{}");
        assert_eq!(err.kind, ErrorKind::AuthFailure);
    }

    /// Tool history and `tool_choice` travel; `top_p`, `stop`,
    /// `response_format` and image parts do not, because Groq advertises
    /// none of `TOP_P`, `STOP_SEQUENCES`, `RESPONSE_FORMAT` or `VISION`.
    #[test]
    fn the_body_is_the_openai_shape_without_what_groq_does_not_advertise() {
        let provider = GroqProvider::with_client(GroqConfig::new("k"), reqwest::Client::new());
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![ToolCallRequest {
            id: "call_1".to_owned(),
            function_name: "get_weather".to_owned(),
            arguments: json!({"city": "Paris"}),
        }]);
        let image = ImagePart::new("aGVsbG8=", "image/png").expect("valid mime"); // Safe: test assertion
        let request = ChatRequest::new(vec![
            ChatMessage::system("Be brief."),
            ChatMessage::user_with_images("Describe", vec![image]),
            assistant,
            ChatMessage::tool("get_weather", "call_1", r#"{"temp":21}"#),
        ])
        .with_temperature(0.5)
        .with_max_tokens(64)
        .with_top_p(0.9)
        .with_stop(vec!["END".to_owned()])
        .with_response_format(ResponseFormat::JsonObject)
        .with_tools(vec![ToolDefinition {
            name: "get_weather".to_owned(),
            description: "Weather".to_owned(),
            parameters: Some(json!({"type": "object"})),
        }])
        .with_tool_choice(ToolChoice::Required);

        let body = serde_json::to_value(provider.build_body(&request, false)).expect("serialises"); // Safe: test assertion
        assert_eq!(
            body,
            json!({
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Describe"},
                    {"role": "assistant", "content": null, "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]},
                    {"role": "tool", "content": "{\"temp\":21}", "tool_call_id": "call_1"}
                ],
                "temperature": 0.5,
                "max_tokens": 64,
                "stream": false,
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Weather", "parameters": {"type": "object"}
                }}],
                "tool_choice": "required"
            })
        );
    }

    #[test]
    fn tools_without_a_choice_ask_for_auto() {
        let provider = GroqProvider::with_client(GroqConfig::new("k"), reqwest::Client::new());
        let request =
            ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition {
                name: "get_weather".to_owned(),
                description: "Weather".to_owned(),
                parameters: None,
            }]);
        let body = serde_json::to_value(provider.build_body(&request, true)).expect("serialises"); // Safe: test assertion
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["tools"],
            json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather"}}])
        );
    }

    /// A recorded Groq answer: the tool call's arguments are decoded from
    /// their JSON string, and usage is the plain counts even when Groq
    /// reports a cached share.
    #[test]
    fn a_recorded_response_yields_its_tool_calls_and_the_plain_counts() {
        let recorded = r#"{
            "id": "chatcmpl-1", "object": "chat.completion", "model": "llama-3.3-70b-versatile",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "tool_calls": [{"id": "call_9", "type": "function",
                    "function": {"name": "get_activities", "arguments": "{\"limit\":5}"}}]
            }}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102,
                "prompt_tokens_details": {"cached_tokens": 64}},
            "x_groq": {"id": "req_1"}
        }"#;
        let response = GroqProvider::read_completion(recorded).expect("parses"); // Safe: test assertion
        assert_eq!(response.model, "llama-3.3-70b-versatile");
        assert_eq!(response.content, "");
        assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
        let calls = response.tool_calls.expect("tool calls"); // Safe: test assertion
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_9");
        assert_eq!(calls[0].function_name, "get_activities");
        assert_eq!(calls[0].arguments, json!({"limit": 5}));
        let usage = response.usage.expect("usage"); // Safe: test assertion
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (90, 12, 102)
        );
        assert_eq!(usage.cached_read_tokens, None);
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let output = format!("{:?}", GroqConfig::new("super-secret"));
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
