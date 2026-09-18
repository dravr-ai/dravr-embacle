// ABOUTME: Generic OpenAI-compatible provider for local and self-hosted endpoints: Ollama, vLLM, LocalAI, and others
// ABOUTME: Names itself after the endpoint it targets so a self-hosted model bills at $0 by design, not by omission
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # `OpenAI`-Compatible Provider
//!
//! [`LlmProvider`] over any endpoint that implements the `OpenAI` chat
//! completions API — local LLM servers like Ollama, vLLM and `LocalAI`
//! first of all.
//!
//! ## Configuration
//!
//! [`OpenAiCompatibleProvider::from_env`] reads:
//!
//! - `LOCAL_LLM_BASE_URL`: default <http://localhost:11434/v1> (Ollama)
//! - `LOCAL_LLM_MODEL`: default `qwen2.5:14b-instruct`
//! - `LOCAL_LLM_API_KEY`: optional, empty for local servers
//!
//! The provider's `name()` is `ollama`, `vllm` or `localai` when the base
//! URL's port identifies one of those, `local` otherwise — the names the
//! price table lists as not per-token metered.
//!
//! ## Supported Backends
//!
//! - **Ollama**: <http://localhost:11434/v1>
//! - **vLLM**: <http://localhost:8000/v1>
//! - **`LocalAI`**: <http://localhost:8080/v1>
//! - **Any `OpenAI`-compatible endpoint**

use std::env;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use super::client::{self, map_send_error, DEFAULT_TIMEOUT_SECS};
use super::sse::create_sse_stream;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
    StreamChunk, TokenUsage, ToolCallRequest, ToolDefinition,
};

// ============================================================================
// Configuration Constants
// ============================================================================

/// Environment variable for local LLM base URL
const LOCAL_LLM_BASE_URL_ENV: &str = "LOCAL_LLM_BASE_URL";

/// Environment variable for local LLM default model
const LOCAL_LLM_MODEL_ENV: &str = "LOCAL_LLM_MODEL";

/// Environment variable for local LLM API key (optional)
const LOCAL_LLM_API_KEY_ENV: &str = "LOCAL_LLM_API_KEY";

/// Default base URL (Ollama)
const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// Default model for local inference
const DEFAULT_MODEL: &str = "qwen2.5:14b-instruct";

/// The name error messages and logs use for whichever endpoint is configured
const LOG_LABEL: &str = "LocalLLM";

/// Models commonly served by the local runtimes this provider targets.
const AVAILABLE_MODELS: &[&str] = &[
    "qwen2.5:14b-instruct",
    "qwen2.5:7b-instruct",
    "qwen2.5:32b-instruct",
    "llama3.1:8b-instruct",
    "llama3.1:70b-instruct",
    "llama3.3:70b-instruct",
    "mistral:7b-instruct",
    "hermes2pro:latest",
];

// ============================================================================
// API Request/Response Types (OpenAI-compatible format)
// ============================================================================

/// OpenAI-compatible API request structure
#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

/// Tool definition for OpenAI-compatible API
#[derive(Debug, Clone, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunction,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize)]
struct OpenAiFunction {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

/// Message structure for OpenAI-compatible API: role and text only.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

impl From<&ChatMessage> for OpenAiMessage {
    fn from(msg: &ChatMessage) -> Self {
        Self {
            role: msg.role.as_str().to_owned(),
            content: msg.content.clone(),
        }
    }
}

/// OpenAI-compatible API response structure
#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    model: String,
}

/// Choice in response
#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

/// Message in response
#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

/// Tool call in response
#[derive(Debug, Clone, Deserialize)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAiFunctionCall,
}

/// Function call details in response
#[derive(Debug, Clone, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

/// Usage statistics in response
#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
    /// Breakdown of `prompt_tokens`, carrying the cache-read share.
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiUsageDetails>,
    /// Breakdown of `completion_tokens`, carrying the reasoning-token share.
    #[serde(default)]
    completion_tokens_details: Option<OpenAiCompletionDetails>,
}

/// The `completion_tokens_details` sub-object, present on reasoning models.
#[derive(Debug, Deserialize)]
struct OpenAiCompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

/// The `prompt_tokens_details` sub-object of an OpenAI-compatible usage block.
#[derive(Debug, Deserialize)]
struct OpenAiUsageDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

impl OpenAiUsage {
    /// No cache-WRITE count in this API shape: writes are implicit and
    /// unbilled, so `None` is accurate, not unknown.
    fn into_token_usage(self) -> TokenUsage {
        TokenUsage::new(self.prompt, self.completion, self.total)
            .with_cache(
                self.prompt_tokens_details.and_then(|d| d.cached_tokens),
                None,
            )
            .with_reasoning(
                self.completion_tokens_details
                    .and_then(|d| d.reasoning_tokens),
            )
    }
}

/// Streaming chunk structure
#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

/// Choice in streaming chunk
#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiDelta,
    finish_reason: Option<String>,
}

/// Delta content in streaming chunk
#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Error response structure
#[derive(Debug, Deserialize)]
struct OpenAiErrorResponse {
    error: OpenAiErrorDetail,
}

/// Error detail structure
#[derive(Debug, Deserialize)]
struct OpenAiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for the `OpenAI`-compatible provider
#[derive(Clone)]
pub struct OpenAiCompatibleConfig {
    /// Base URL for the API (e.g., <http://localhost:11434/v1>)
    pub base_url: String,
    /// API key (optional for local servers). Redacted in `Debug`.
    pub api_key: Option<String>,
    /// Default model to use
    pub default_model: String,
    /// Provider name for logging; decides what `name()` reports
    pub provider_name: String,
    /// Provider display name
    pub display_name: String,
    /// Capabilities of this provider
    pub capabilities: LlmCapabilities,
}

impl OpenAiCompatibleConfig {
    /// Create configuration for a local Ollama instance
    #[must_use]
    pub fn ollama(model: &str) -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "ollama".to_owned(),
            display_name: "Ollama (Local)".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
        }
    }

    /// Create configuration for a local vLLM instance
    #[must_use]
    pub fn vllm(model: &str) -> Self {
        Self {
            base_url: "http://localhost:8000/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "vllm".to_owned(),
            display_name: "vLLM (Local)".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES
                | LlmCapabilities::JSON_MODE,
        }
    }

    /// Create configuration for `LocalAI`
    #[must_use]
    pub fn local_ai(model: &str) -> Self {
        Self {
            base_url: "http://localhost:8080/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "localai".to_owned(),
            display_name: "LocalAI".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
        }
    }

    /// Read the configuration from the environment; the endpoint's port
    /// decides the provider name (`11434` Ollama, `8000` vLLM, `8080`
    /// `LocalAI`, anything else `local`).
    #[must_use]
    pub fn from_env() -> Self {
        let base_url =
            env::var(LOCAL_LLM_BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let default_model =
            env::var(LOCAL_LLM_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let api_key = env::var(LOCAL_LLM_API_KEY_ENV)
            .ok()
            .filter(|k| !k.is_empty());

        let (provider_name, display_name) = if base_url.contains(":11434") {
            ("ollama", "Ollama (Local)")
        } else if base_url.contains(":8000") {
            ("vllm", "vLLM (Local)")
        } else if base_url.contains(":8080") {
            ("localai", "LocalAI")
        } else {
            ("local", "Local LLM")
        };

        Self {
            base_url,
            api_key,
            default_model,
            provider_name: provider_name.to_owned(),
            display_name: display_name.to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
        }
    }
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            api_key: None,
            default_model: DEFAULT_MODEL.to_owned(),
            provider_name: "local".to_owned(),
            display_name: "Local LLM".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
        }
    }
}

impl Debug for OpenAiCompatibleConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenAiCompatibleConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("default_model", &self.default_model)
            .field("provider_name", &self.provider_name)
            .field("display_name", &self.display_name)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

/// LLM provider for OpenAI-compatible APIs (Ollama, vLLM, LM Studio, etc.)
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    config: OpenAiCompatibleConfig,
    available_models: Vec<String>,
}

impl OpenAiCompatibleProvider {
    /// Create a provider that builds its own HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(Duration::from_secs(DEFAULT_TIMEOUT_SECS))?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: OpenAiCompatibleConfig, client: reqwest::Client) -> Self {
        Self {
            client,
            config,
            available_models: AVAILABLE_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Override the default model after construction.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Create a provider from environment variables (see
    /// [`OpenAiCompatibleConfig::from_env`]).
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn from_env() -> Result<Self, RunnerError> {
        let config = OpenAiCompatibleConfig::from_env();
        info!(
            provider = %config.display_name,
            base_url = %config.base_url,
            default_model = %config.default_model,
            "Initializing OpenAI-compatible provider"
        );
        Self::new(config)
    }

    /// Build the API URL for a given endpoint
    fn api_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint
        )
    }

    /// Convert internal messages to `OpenAI` format
    fn convert_messages(messages: &[ChatMessage]) -> Vec<OpenAiMessage> {
        messages.iter().map(OpenAiMessage::from).collect()
    }

    /// Log message details for debugging LLM interactions
    fn log_messages_debug(messages: &[OpenAiMessage], provider_name: &str, has_tools: bool) {
        for (i, msg) in messages.iter().enumerate() {
            debug!(
                "Message[{i}] role={}, content_len={}",
                msg.role,
                msg.content.len()
            );
            if msg.role == "system" {
                debug!("System prompt present (content redacted for security)");
            }
        }
        debug!(
            "Sending chat completion request to {provider_name} with {} messages and tools={has_tools:?}",
            messages.len()
        );
    }

    /// Parse an error response from the endpoint.
    ///
    /// Local servers answer with HTML or an empty body more often than a
    /// JSON envelope, so a 502–504 without JSON reads as "the local server is
    /// not responding". A 404 is an unavailable model (Ollama's answer to a
    /// model that was never pulled), a 503 says the server is starting.
    fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
        let Ok(error_response) = serde_json::from_str::<OpenAiErrorResponse>(body) else {
            if (502..=504).contains(&status.as_u16()) {
                return RunnerError::external_service(
                    LOG_LABEL,
                    "Local LLM server is not responding. Is Ollama/vLLM running?",
                );
            }
            debug!(
                status = status.as_u16(),
                body_preview = %body.chars().take(200).collect::<String>(),
                "Non-JSON error response body"
            );
            return client::map_http_error(LOG_LABEL, status, &format!("HTTP {status}"));
        };

        let message = error_response.error.message;
        match status.as_u16() {
            404 => RunnerError::model_unavailable(format!("endpoint or model: {message}")),
            503 => RunnerError::external_service(
                LOG_LABEL,
                format!("Service unavailable (is the local server running?): {message}"),
            ),
            _ => {
                let error_type = error_response
                    .error
                    .error_type
                    .unwrap_or_else(|| "unknown".to_owned());
                client::map_http_error(LOG_LABEL, status, &format!("{error_type} - {message}"))
            }
        }
    }

    /// Convert tool definitions to OpenAI-compatible format
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<OpenAiTool> {
        tools
            .iter()
            .map(|func| OpenAiTool {
                tool_type: "function".to_owned(),
                function: OpenAiFunction {
                    name: func.name.clone(),
                    description: func.description.clone(),
                    parameters: func.parameters.clone(),
                },
            })
            .collect()
    }

    /// Convert tool calls to tool-call requests. Arguments arrive as a
    /// JSON-encoded string; one that does not parse becomes `null`.
    fn convert_tool_calls(tool_calls: &[OpenAiToolCall]) -> Vec<ToolCallRequest> {
        tool_calls
            .iter()
            .map(|call| {
                debug!(
                    tool_call_id = %call.id,
                    tool_call_type = %call.call_type,
                    function_name = %call.function.name,
                    "Converting tool call"
                );
                ToolCallRequest {
                    id: call.id.clone(),
                    function_name: call.function.name.clone(),
                    arguments: serde_json::from_str(&call.function.arguments).unwrap_or_default(),
                }
            })
            .collect()
    }

    /// Map a transport failure, naming the endpoint when it refused the
    /// connection so the operator knows which server to start.
    fn map_send(&self, error: reqwest::Error) -> RunnerError {
        if error.is_connect() {
            RunnerError::external_service(
                LOG_LABEL,
                format!(
                    "Cannot connect to {}. Is the server running at {}?",
                    self.config.display_name, self.config.base_url
                ),
            )
        } else {
            map_send_error(LOG_LABEL, error)
        }
    }

    /// Add authorization header if API key is configured
    fn add_auth_header(&self, request: RequestBuilder) -> RequestBuilder {
        match &self.config.api_key {
            Some(api_key) => request.bearer_auth(api_key),
            None => request,
        }
    }

    /// Build the request body for a `ChatRequest`
    fn build_body(&self, request: &ChatRequest, stream: bool) -> OpenAiRequest {
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let messages = Self::convert_messages(&request.messages);
        let tools = request
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(Self::convert_tools);
        Self::log_messages_debug(&messages, &self.config.provider_name, tools.is_some());
        OpenAiRequest {
            model: model.to_owned(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(stream),
            tool_choice: tools.as_ref().map(|_| "auto".to_owned()),
            tools,
        }
    }

    /// Send a chat-completions request
    async fn send(&self, body: &OpenAiRequest) -> Result<reqwest::Response, RunnerError> {
        let http_request = self
            .client
            .post(self.api_url("chat/completions"))
            .header("Content-Type", "application/json")
            .json(body);
        self.add_auth_header(http_request)
            .send()
            .await
            .map_err(|e| self.map_send(e))
    }

    /// Parse an OpenAI-compatible SSE data payload into a `StreamChunk`
    fn parse_stream_data(json_str: &str) -> Option<Result<StreamChunk, RunnerError>> {
        match serde_json::from_str::<OpenAiStreamChunk>(json_str) {
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
                warn!("Failed to parse stream chunk: {e}");
                None
            }
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &'static str {
        // The trait wants a static string; these four are the names the
        // price table knows as self-hosted.
        match self.config.provider_name.as_str() {
            "ollama" => "ollama",
            "vllm" => "vllm",
            "localai" => "localai",
            _ => "local",
        }
    }

    fn display_name(&self) -> &str {
        &self.config.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.config.capabilities
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.default_model)))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let body = self.build_body(request, false);
        let response = self.send(&body).await?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            RunnerError::external_service(LOG_LABEL, format!("Failed to read response: {e}"))
        })?;

        if !status.is_success() {
            return Err(Self::parse_error_response(status, &text));
        }

        let openai_response: OpenAiResponse = serde_json::from_str(&text).map_err(|e| {
            RunnerError::external_service(LOG_LABEL, format!("Failed to parse response: {e}"))
        })?;

        let choice =
            openai_response.choices.into_iter().next().ok_or_else(|| {
                RunnerError::external_service(LOG_LABEL, "API returned no choices")
            })?;

        let content = choice.message.content.unwrap_or_default();
        let tool_calls = choice.message.tool_calls.map(|calls| {
            info!(
                "{} returned {} tool calls",
                self.config.provider_name,
                calls.len()
            );
            Self::convert_tool_calls(&calls)
        });

        debug!(
            provider = %self.config.provider_name,
            content_len = content.len(),
            tool_calls = tool_calls.as_ref().map(Vec::len),
            finish_reason = ?choice.finish_reason,
            "Received response"
        );

        Ok(ChatResponse {
            content,
            model: openai_response.model,
            usage: openai_response.usage.map(OpenAiUsage::into_token_usage),
            finish_reason: choice.finish_reason,
            warnings: None,
            tool_calls,
        })
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.default_model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let body = self.build_body(request, true);
        let response = self.send(&body).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Self::parse_error_response(status, &text));
        }

        Ok(create_sse_stream(
            response.bytes_stream(),
            Self::parse_stream_data,
            LOG_LABEL,
        ))
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        debug!(
            provider = %self.config.provider_name,
            base_url = %self.config.base_url,
            "Performing health check"
        );

        // The models endpoint is a lightweight health check
        let http_request = self.client.get(self.api_url("models"));
        let response = self
            .add_auth_header(http_request)
            .send()
            .await
            .map_err(|e| self.map_send(e))?;

        let healthy = response.status().is_success();
        if healthy {
            debug!("{} health check passed", self.config.provider_name);
        } else {
            warn!(
                provider = %self.config.provider_name,
                status = response.status().as_u16(),
                "health check failed"
            );
        }
        Ok(healthy)
    }
}

impl Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn a_non_json_gateway_error_says_the_server_is_not_responding() {
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::BAD_GATEWAY, "<html>");
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("not responding"));
    }

    #[test]
    fn a_404_is_an_unavailable_model_a_400_an_invalid_request() {
        let body = r#"{"error":{"message":"model 'qwen3' not found","type":"api_error"}}"#;
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::NOT_FOUND, body);
        assert_eq!(err.kind, ErrorKind::ModelUnavailable);
        let body = r#"{"error":{"message":"invalid messages","type":"invalid_request_error"}}"#;
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::BAD_REQUEST, body);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
    }

    #[test]
    fn a_429_is_a_rate_limit_and_a_503_external_service() {
        let body = r#"{"error":{"message":"Rate limit reached. Please try again in 20s."}}"#;
        let err =
            OpenAiCompatibleProvider::parse_error_response(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("20 seconds"));
        let body = r#"{"error":{"message":"loading model"}}"#;
        let err =
            OpenAiCompatibleProvider::parse_error_response(StatusCode::SERVICE_UNAVAILABLE, body);
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("is the local server running"));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let config = OpenAiCompatibleConfig {
            api_key: Some("super-secret".to_owned()),
            ..OpenAiCompatibleConfig::default()
        };
        let output = format!("{config:?}");
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
