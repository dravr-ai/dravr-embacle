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
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use super::client::{self, with_retries, AttemptError, HttpRetryConfig, DEFAULT_TIMEOUT_SECS};
use super::sse::create_sse_stream;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
    StreamChunk, TokenUsage, ToolCallRequest, ToolDefinition,
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
// API Request/Response Types (OpenAI-compatible format)
// ============================================================================

/// Groq API request structure (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GroqTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

/// Tool definition for Groq API (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize)]
struct GroqTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: GroqFunction,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize)]
struct GroqFunction {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

/// Message structure for Groq API (OpenAI-compatible): role and text only.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GroqMessage {
    role: String,
    content: String,
}

impl From<&ChatMessage> for GroqMessage {
    fn from(msg: &ChatMessage) -> Self {
        Self {
            role: msg.role.as_str().to_owned(),
            content: msg.content.clone(),
        }
    }
}

/// Groq API response structure (OpenAI-compatible)
#[derive(Debug, Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
    #[serde(default)]
    usage: Option<GroqUsage>,
    model: String,
}

/// Choice in Groq response
#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqResponseMessage,
    finish_reason: Option<String>,
}

/// Message in Groq response
#[derive(Debug, Deserialize)]
struct GroqResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<GroqToolCall>>,
}

/// Tool call in Groq response (OpenAI-compatible)
#[derive(Debug, Clone, Deserialize)]
struct GroqToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: GroqFunctionCall,
}

/// Function call details in Groq response
#[derive(Debug, Clone, Deserialize)]
struct GroqFunctionCall {
    name: String,
    arguments: String,
}

/// Usage statistics in Groq response
#[derive(Debug, Deserialize)]
struct GroqUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
}

/// Streaming chunk structure (OpenAI-compatible)
#[derive(Debug, Deserialize)]
struct GroqStreamChunk {
    choices: Vec<GroqStreamChoice>,
}

/// Choice in streaming chunk
#[derive(Debug, Deserialize)]
struct GroqStreamChoice {
    delta: GroqDelta,
    finish_reason: Option<String>,
}

/// Delta content in streaming chunk
#[derive(Debug, Deserialize)]
struct GroqDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Groq API error response
#[derive(Debug, Deserialize)]
struct GroqErrorResponse {
    error: GroqErrorDetail,
}

/// Error detail structure
#[derive(Debug, Deserialize)]
struct GroqErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

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

    /// Convert internal messages to Groq format
    fn convert_messages(messages: &[ChatMessage]) -> Vec<GroqMessage> {
        messages.iter().map(GroqMessage::from).collect()
    }

    /// Parse an error response from the Groq API: the vendor's message when
    /// the body is its JSON envelope, `HTTP <status>` otherwise; the status
    /// decides the kind.
    fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
        let message = serde_json::from_str::<GroqErrorResponse>(body).map_or_else(
            |_| {
                debug!(
                    status = status.as_u16(),
                    body_preview = %body.chars().take(200).collect::<String>(),
                    "Groq API returned non-JSON error response"
                );
                format!("HTTP {status}")
            },
            |e| {
                let error_type = e.error.error_type.unwrap_or_else(|| "unknown".to_owned());
                format!("{error_type} - {}", e.error.message)
            },
        );
        client::map_http_error(PROVIDER_NAME, status, &message)
    }

    /// Build an authenticated HTTP request to the Groq API
    fn build_request(&self, groq_request: &GroqRequest) -> RequestBuilder {
        self.client
            .post(Self::api_url("chat/completions"))
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(groq_request)
    }

    /// Build the request body for a `ChatRequest`
    fn build_body(&self, request: &ChatRequest, stream: bool) -> GroqRequest {
        let model = request.model.as_deref().unwrap_or(&self.config.model);
        let tools = request
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(Self::convert_tools);
        GroqRequest {
            model: model.to_owned(),
            messages: Self::convert_messages(&request.messages),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(stream),
            tool_choice: tools.as_ref().map(|_| "auto".to_owned()),
            tools,
        }
    }

    /// Parse a Groq SSE data payload into a `StreamChunk`
    fn parse_stream_data(json_str: &str) -> Option<Result<StreamChunk, RunnerError>> {
        match serde_json::from_str::<GroqStreamChunk>(json_str) {
            Ok(chunk) => {
                let choice = chunk.choices.into_iter().next()?;
                let delta = choice.delta.content.unwrap_or_default();
                let is_final = choice.finish_reason.is_some();
                Some(Ok(StreamChunk {
                    delta,
                    is_final,
                    finish_reason: choice.finish_reason,
                }))
            }
            Err(e) => {
                warn!("Failed to parse Groq stream chunk: {e}");
                None
            }
        }
    }

    /// Convert tool definitions to Groq's OpenAI-compatible format
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<GroqTool> {
        tools
            .iter()
            .map(|func| GroqTool {
                tool_type: "function".to_owned(),
                function: GroqFunction {
                    name: func.name.clone(),
                    description: func.description.clone(),
                    parameters: func.parameters.clone(),
                },
            })
            .collect()
    }

    /// Convert Groq tool calls to tool-call requests. Arguments arrive as a
    /// JSON-encoded string; one that does not parse becomes `null`.
    fn convert_tool_calls(tool_calls: &[GroqToolCall]) -> Vec<ToolCallRequest> {
        tool_calls
            .iter()
            .map(|call| {
                debug!(
                    tool_call_id = %call.id,
                    tool_call_type = %call.call_type,
                    function_name = %call.function.name,
                    "Converting Groq tool call"
                );
                ToolCallRequest {
                    id: call.id.clone(),
                    function_name: call.function.name.clone(),
                    arguments: serde_json::from_str(&call.function.arguments).unwrap_or_default(),
                }
            })
            .collect()
    }

    /// One non-streaming attempt
    async fn attempt_complete(&self, body: &GroqRequest) -> Result<ChatResponse, AttemptError> {
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

        let groq_response: GroqResponse = serde_json::from_str(&text).map_err(|e| {
            AttemptError::permanent(RunnerError::external_service(
                PROVIDER_NAME,
                format!("Failed to parse response: {e}"),
            ))
        })?;

        let choice = groq_response.choices.into_iter().next().ok_or_else(|| {
            AttemptError::permanent(RunnerError::external_service(
                PROVIDER_NAME,
                "API returned no choices",
            ))
        })?;

        let content = choice.message.content.unwrap_or_default();
        let tool_calls = choice.message.tool_calls.map(|calls| {
            info!("Groq returned {} tool calls", calls.len());
            Self::convert_tool_calls(&calls)
        });

        debug!(
            content_len = content.len(),
            tool_calls = tool_calls.as_ref().map(Vec::len),
            finish_reason = ?choice.finish_reason,
            "Received response from Groq"
        );

        Ok(ChatResponse {
            content,
            model: groq_response.model,
            usage: groq_response
                .usage
                .map(|u| TokenUsage::new(u.prompt, u.completion, u.total)),
            finish_reason: choice.finish_reason,
            warnings: None,
            tool_calls,
        })
    }

    /// One attempt at opening the stream
    async fn attempt_stream(&self, body: &GroqRequest) -> Result<ChatStream, AttemptError> {
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
            Self::parse_stream_data,
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
    use crate::types::ErrorKind;

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

    #[test]
    fn tools_are_sent_in_the_openai_shape_with_auto_choice() {
        let provider = GroqProvider::with_client(GroqConfig::new("k"), reqwest::Client::new());
        let request =
            ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition {
                name: "get_weather".to_owned(),
                description: "Weather".to_owned(),
                parameters: None,
            }]);
        let body = provider.build_body(&request, false);
        let json = serde_json::to_value(&body).expect("serialises"); // Safe: test assertion
        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["tool_choice"], "auto");
        assert_eq!(json["model"], DEFAULT_MODEL);
    }

    #[test]
    fn tool_call_arguments_are_decoded_from_their_json_string() {
        let calls = vec![GroqToolCall {
            id: "call_1".to_owned(),
            call_type: "function".to_owned(),
            function: GroqFunctionCall {
                name: "get_activities".to_owned(),
                arguments: r#"{"limit":5}"#.to_owned(),
            },
        }];
        let converted = GroqProvider::convert_tool_calls(&calls);
        assert_eq!(converted[0].id, "call_1");
        assert_eq!(converted[0].arguments["limit"], 5);
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let output = format!("{:?}", GroqConfig::new("super-secret"));
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
