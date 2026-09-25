// ABOUTME: OpenRouter provider — one key, 200+ upstream models behind an OpenAI-compatible gateway
// ABOUTME: Sends the HTTP-Referer / X-Title ranking headers and reads cached and reasoning token details from usage
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # `OpenRouter` Provider
//!
//! [`LlmProvider`] over `OpenRouter`, a unified gateway exposing 200+
//! frontier and open-source models behind a single OpenAI-compatible
//! endpoint. Model selection is per request via the `provider/model` slug.
//!
//! ## Configuration
//!
//! [`OpenRouterConfig::from_env`] reads:
//!
//! - `OPENROUTER_API_KEY`: required, from <https://openrouter.ai/keys>
//! - `OPENROUTER_DEFAULT_MODEL`: default `meta-llama/llama-3.3-70b-instruct`
//! - `OPENROUTER_SITE_URL`: sent as `HTTP-Referer` so `OpenRouter` can rank traffic
//! - `OPENROUTER_APP_TITLE`: sent as `X-Title` for the same reason
//! - `OPENROUTER_MAX_RETRIES`, `OPENROUTER_INITIAL_RETRY_DELAY_MS`,
//!   `OPENROUTER_MAX_RETRY_DELAY_MS`: retry tuning (defaults 3 / 500 / 5000)
//!
//! ## Usage accounting
//!
//! `prompt_tokens_details.cached_tokens` and
//! `completion_tokens_details.reasoning_tokens` are read into
//! [`TokenUsage`], so a cached prefix bills at the cache rate and a
//! reasoning model's thought tokens are charged rather than dropped.

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
const PROVIDER_NAME: &str = "openrouter";

/// Environment variable for the API key
const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Environment variable for the default model
const OPENROUTER_DEFAULT_MODEL_ENV: &str = "OPENROUTER_DEFAULT_MODEL";

/// Environment variable for the `HTTP-Referer` ranking header
const OPENROUTER_SITE_URL_ENV: &str = "OPENROUTER_SITE_URL";

/// Environment variable for the `X-Title` ranking header
const OPENROUTER_APP_TITLE_ENV: &str = "OPENROUTER_APP_TITLE";

/// Prefix of the retry-tuning environment variables
const OPENROUTER_ENV_PREFIX: &str = "OPENROUTER";

/// Default model — one of the cheapest tool-calling capable models on the
/// network, mirroring the Groq Llama default.
const DEFAULT_MODEL: &str = "meta-llama/llama-3.3-70b-instruct";

/// A curated subset of `OpenRouter` model slugs.
///
/// `OpenRouter` exposes 200+ models. This list is intentionally short and
/// **not** authoritative: any slug from <https://openrouter.ai/models> works
/// as the configured or requested model.
const AVAILABLE_MODELS: &[&str] = &[
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "google/gemini-pro-1.5",
    "mistralai/mistral-large",
    "mistralai/mistral-nemo",
    "qwen/qwen-2.5-72b-instruct",
];

/// Base URL for the `OpenRouter` API (OpenAI-compatible)
const API_BASE_URL: &str = "https://openrouter.ai/api/v1";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the `OpenRouter` provider.
#[derive(Clone)]
pub struct OpenRouterConfig {
    /// API key from <https://openrouter.ai/keys>. Redacted in `Debug`.
    pub api_key: String,
    /// Model slug used when the request names none
    pub model: String,
    /// Value for the `HTTP-Referer` header (`OpenRouter` traffic ranking)
    pub site_url: Option<String>,
    /// Value for the `X-Title` header (`OpenRouter` traffic ranking)
    pub app_title: Option<String>,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
    /// Retry policy for transient failures (429, 502, 503, network errors)
    pub retry: HttpRetryConfig,
}

impl OpenRouterConfig {
    /// A configuration with the given key and every other field at its default.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            site_url: None,
            app_title: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            retry: HttpRetryConfig::default(),
        }
    }

    /// Read the configuration from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when
    /// `OPENROUTER_API_KEY` is not set.
    pub fn from_env() -> Result<Self, RunnerError> {
        let api_key = env::var(OPENROUTER_API_KEY_ENV).map_err(|_| {
            RunnerError::config(format!(
                "Missing {OPENROUTER_API_KEY_ENV} environment variable. Get your API key from https://openrouter.ai/keys"
            ))
        })?;
        let model =
            env::var(OPENROUTER_DEFAULT_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let non_empty = |name: &str| env::var(name).ok().filter(|s| !s.is_empty());
        Ok(Self {
            model,
            site_url: non_empty(OPENROUTER_SITE_URL_ENV),
            app_title: non_empty(OPENROUTER_APP_TITLE_ENV),
            retry: HttpRetryConfig::from_env(OPENROUTER_ENV_PREFIX),
            ..Self::new(api_key)
        })
    }

    /// Set the default model slug
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

impl Debug for OpenRouterConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenRouterConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("site_url", &self.site_url)
            .field("app_title", &self.app_title)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .finish()
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

/// `OpenRouter` LLM provider — unified access to 200+ models
///
/// Routes requests through <https://openrouter.ai/api/v1>, an
/// OpenAI-compatible gateway. Model selection happens per-request via
/// the `model` slug (e.g. `anthropic/claude-3.5-sonnet`).
pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: reqwest::Client,
    available_models: Vec<String>,
}

impl OpenRouterProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: OpenRouterConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: OpenRouterConfig, client: reqwest::Client) -> Self {
        info!(
            default_model = %config.model,
            site_url_set = config.site_url.is_some(),
            app_title_set = config.app_title.is_some(),
            max_retries = config.retry.max_retries,
            initial_delay_ms = config.retry.initial_delay_ms,
            "OpenRouter provider initialized"
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

    /// Parse an error response from the `OpenRouter` API: the vendor's
    /// message when the body is its JSON envelope, `HTTP <status>` otherwise;
    /// the status decides the kind. A 402 (credit balance exhausted) is an
    /// external-service error like any other non-4xx-class refusal.
    fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
        let message = match chat_completions::error_detail(body) {
            Some(detail) if status == StatusCode::PAYMENT_REQUIRED => {
                format!("credit balance exhausted: {}", detail.message)
            }
            _ => chat_completions::describe_error(PROVIDER_NAME, status, body),
        };
        client::map_http_error(PROVIDER_NAME, status, &message)
    }

    /// Apply the ranking headers when configured
    fn with_ranking_headers(&self, mut builder: RequestBuilder) -> RequestBuilder {
        if let Some(ref site) = self.config.site_url {
            builder = builder.header("HTTP-Referer", site);
        }
        if let Some(ref title) = self.config.app_title {
            builder = builder.header("X-Title", title);
        }
        builder
    }

    /// Build an authenticated HTTP request to the `OpenRouter` API
    fn build_request(&self, body: &CompletionRequest) -> RequestBuilder {
        let builder = self
            .client
            .post(Self::api_url("chat/completions"))
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json");
        self.with_ranking_headers(builder).json(body)
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

    /// Read a completion body, with the cached and reasoning shares of its
    /// usage so a cached prefix bills at the cache rate and thought tokens
    /// are charged.
    fn read_completion(body: &str) -> Result<ChatResponse, RunnerError> {
        chat_completions::parse_response(PROVIDER_NAME, body, Usage::with_details)
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
impl LlmProvider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> &str {
        "OpenRouter"
    }

    fn capabilities(&self) -> LlmCapabilities {
        // Capabilities here reflect the gateway, not any single model.
        // Vision is supported by many OpenRouter models but not all; the
        // shared OpenAI-compatible surface is advertised and model selection
        // determines effective per-request capabilities.
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
            "OpenRouter request",
            |attempt| async move {
                debug!(attempt, "Sending chat completion request to OpenRouter");
                self.attempt_complete(body).await
            },
        )
        .await
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let body = self.build_body(request, true);
        let body = &body;
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "OpenRouter streaming request",
            |attempt| async move {
                debug!(
                    attempt,
                    "Sending streaming chat completion request to OpenRouter"
                );
                self.attempt_stream(body).await
            },
        )
        .await
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        debug!("Performing OpenRouter API health check");
        let builder = self
            .client
            .get(Self::api_url("models"))
            .bearer_auth(&self.config.api_key);
        let response = self
            .with_ranking_headers(builder)
            .send()
            .await
            .map_err(|e| client::map_send_error(PROVIDER_NAME, e))?;

        let healthy = response.status().is_success();
        if healthy {
            debug!("OpenRouter API health check passed");
        } else {
            warn!(
                status = response.status().as_u16(),
                "OpenRouter API health check failed"
            );
        }
        Ok(healthy)
    }
}

impl Debug for OpenRouterProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenRouterProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ErrorKind, ResponseFormat, ToolChoice, ToolDefinition};
    use serde_json::json;

    /// A recorded `OpenRouter` answer: tool calls decoded, and the cached
    /// and reasoning shares read into usage.
    #[test]
    fn a_recorded_response_yields_its_tool_calls_and_usage_details() {
        let recorded = r#"{
            "id": "gen-1", "provider": "Together", "model": "meta-llama/llama-3.3-70b-instruct",
            "choices": [{"finish_reason": "tool_calls", "native_finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "call_7", "type": "function", "index": 0,
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Lyon\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 15}}
        }"#;
        let response = OpenRouterProvider::read_completion(recorded).expect("parses"); // Safe: test assertion
        assert_eq!(response.model, "meta-llama/llama-3.3-70b-instruct");
        let calls = response.tool_calls.expect("tool calls"); // Safe: test assertion
        assert_eq!(calls[0].id, "call_7");
        assert_eq!(calls[0].function_name, "get_weather");
        assert_eq!(calls[0].arguments, json!({"city": "Lyon"}));
        let usage = response.usage.expect("usage"); // Safe: test assertion
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cached_read_tokens, Some(60));
        assert_eq!(usage.cached_write_tokens, None);
        assert_eq!(usage.reasoning_tokens, Some(15));
    }

    /// `OpenRouter` advertises none of `TOP_P`, `STOP_SEQUENCES`,
    /// `RESPONSE_FORMAT` or `VISION`, so those never reach the body.
    #[test]
    fn the_body_is_the_openai_shape_without_what_the_gateway_does_not_advertise() {
        let provider =
            OpenRouterProvider::with_client(OpenRouterConfig::new("k"), reqwest::Client::new());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_model("openai/gpt-4o")
            .with_max_tokens(32)
            .with_top_p(0.9)
            .with_stop(vec!["END".to_owned()])
            .with_response_format(ResponseFormat::JsonObject)
            .with_tools(vec![ToolDefinition {
                name: "lookup".to_owned(),
                description: "Lookup".to_owned(),
                parameters: None,
            }])
            .with_tool_choice(ToolChoice::Specific {
                name: "lookup".to_owned(),
            });
        let body = serde_json::to_value(provider.build_body(&request, true)).expect("serialises"); // Safe: test assertion
        assert_eq!(
            body,
            json!({
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 32,
                "stream": true,
                "tools": [{"type": "function", "function": {"name": "lookup", "description": "Lookup"}}],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}}
            })
        );
    }

    #[test]
    fn a_402_is_an_external_service_error_a_429_a_rate_limit() {
        let body = r#"{"error":{"message":"Insufficient credits","type":"payment"}}"#;
        let err = OpenRouterProvider::parse_error_response(StatusCode::PAYMENT_REQUIRED, body);
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("credit balance exhausted"));
        let body = r#"{"error":{"message":"Rate limit exceeded: free-models-per-minute"}}"#;
        let err = OpenRouterProvider::parse_error_response(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("free-models-per-minute"));
    }

    #[test]
    fn ranking_headers_are_sent_when_configured() {
        let config = OpenRouterConfig {
            site_url: Some("https://example.test".to_owned()),
            app_title: Some("Example".to_owned()),
            ..OpenRouterConfig::new("k")
        };
        let provider = OpenRouterProvider::with_client(config, reqwest::Client::new());
        let request = provider
            .build_request(&provider.build_body(&ChatRequest::new(vec![]), false))
            .build()
            .expect("builds"); // Safe: test assertion
        assert_eq!(
            request
                .headers()
                .get("HTTP-Referer")
                .map(|v| v.to_str().ok()),
            Some(Some("https://example.test"))
        );
        assert_eq!(
            request.headers().get("X-Title").map(|v| v.to_str().ok()),
            Some(Some("Example"))
        );
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let output = format!("{:?}", OpenRouterConfig::new("super-secret"));
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
