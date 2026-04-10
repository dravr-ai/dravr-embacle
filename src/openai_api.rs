// ABOUTME: OpenAI-compatible HTTP API client implementing the LlmProvider trait
// ABOUTME: Connects to any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, vLLM, etc.)
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # OpenAI-Compatible API Runner
//!
//! HTTP client that connects to any endpoint implementing the OpenAI chat
//! completions API. Supports streaming via SSE, native tool calling,
//! model discovery, and all standard completion parameters.
//!
//! ## Configuration
//!
//! ```rust,no_run
//! use embacle::openai_api::{OpenAiApiConfig, OpenAiApiRunner};
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//!
//! # async fn example() -> Result<(), embacle::types::RunnerError> {
//! let config = OpenAiApiConfig::new("https://api.openai.com")
//!     .with_api_key("sk-...")
//!     .with_model("gpt-4o");
//! let runner = OpenAiApiRunner::new(config).await;
//! let request = ChatRequest::new(vec![ChatMessage::user("Hello!")]);
//! let response = runner.complete(&request).await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```

use std::env;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
    ResponseFormat, RunnerError, StreamChunk, TokenUsage, ToolCallRequest, ToolChoice,
    ToolDefinition,
};

// ============================================================================
// Constants
// ============================================================================

/// Default base URL for the `OpenAI` API
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Default model identifier
const DEFAULT_MODEL: &str = "gpt-5.4";

/// Default HTTP request timeout in seconds
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Timeout for model discovery requests (seconds)
const DISCOVERY_TIMEOUT_SECS: u64 = 5;

/// Chat completions API path
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Models list API path
const MODELS_PATH: &str = "/v1/models";

/// SSE stream channel buffer capacity
const STREAM_CHANNEL_CAPACITY: usize = 128;

/// Environment variable for the API base URL
const ENV_BASE_URL: &str = "OPENAI_API_BASE_URL";

/// Environment variable for the API key
const ENV_API_KEY: &str = "OPENAI_API_KEY";

/// Environment variable for the default model
const ENV_MODEL: &str = "OPENAI_API_MODEL";

/// Environment variable for the request timeout (in seconds)
const ENV_TIMEOUT_SECS: &str = "OPENAI_API_TIMEOUT_SECS";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the OpenAI-compatible API runner
///
/// Supports any endpoint implementing the `OpenAI` chat completions protocol:
/// `OpenAI`, Groq, Ollama, vLLM, Together AI, Azure `OpenAI`, etc.
///
/// Configuration can be set programmatically or via environment variables:
/// - `OPENAI_API_BASE_URL` — API base URL (default: `https://api.openai.com`)
/// - `OPENAI_API_KEY` — Bearer token for authentication
/// - `OPENAI_API_MODEL` — Default model to use
/// - `OPENAI_API_TIMEOUT_SECS` — Request timeout in seconds (default: 120)
#[derive(Debug, Clone)]
pub struct OpenAiApiConfig {
    /// Base URL for the API (e.g., `https://api.openai.com`)
    pub base_url: String,
    /// Bearer token for authentication (optional for local endpoints like Ollama)
    pub api_key: Option<String>,
    /// Default model to use when not specified in the request
    pub model: String,
    /// HTTP request timeout
    pub timeout: Duration,
}

impl OpenAiApiConfig {
    /// Create a new configuration with the given base URL
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: normalize_base_url(&base_url.into()),
            api_key: None,
            model: DEFAULT_MODEL.to_owned(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Create a configuration from environment variables
    ///
    /// Falls back to defaults for any unset variable.
    #[must_use]
    pub fn from_env() -> Self {
        let base_url = env::var(ENV_BASE_URL).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let api_key = env::var(ENV_API_KEY).ok();
        let model = env::var(ENV_MODEL).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let timeout_secs: u64 = env::var(ENV_TIMEOUT_SECS)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        Self {
            base_url: normalize_base_url(&base_url),
            api_key,
            model,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Set the API key for bearer authentication
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the default model identifier
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the HTTP request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for OpenAiApiConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Strip trailing slash from a base URL
fn normalize_base_url(url: &str) -> String {
    url.trim_end_matches('/').to_owned()
}

// ============================================================================
// Private Wire Types — Request
// ============================================================================

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCallRef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Tool call reference in an assistant message (request direction)
#[derive(Serialize)]
struct ApiToolCallRef {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ApiToolCallFunction,
}

#[derive(Serialize)]
struct ApiToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ApiTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: ApiFunctionDef,
}

#[derive(Serialize)]
struct ApiFunctionDef {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

// ============================================================================
// Private Wire Types — Response
// ============================================================================

#[derive(Deserialize)]
struct ApiResponse {
    model: String,
    choices: Vec<ApiChoice>,
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiResponseToolCall>>,
}

#[derive(Deserialize)]
struct ApiResponseToolCall {
    id: String,
    function: ApiResponseFunction,
}

#[derive(Deserialize)]
struct ApiResponseFunction {
    name: String,
    arguments: String,
}

/// Wire type — field names mandated by the `OpenAI` API JSON schema
#[derive(Deserialize)]
struct ApiUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
}

// ============================================================================
// Private Wire Types — Streaming
// ============================================================================

#[derive(Deserialize)]
struct ApiStreamResponse {
    choices: Vec<ApiStreamChoice>,
}

#[derive(Deserialize)]
struct ApiStreamChoice {
    delta: ApiStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiStreamDelta {
    content: Option<String>,
}

// ============================================================================
// Private Wire Types — Error & Models
// ============================================================================

#[derive(Deserialize)]
struct ApiErrorResponse {
    error: ApiError,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
}

#[derive(Deserialize)]
struct ApiModelsResponse {
    data: Vec<ApiModel>,
}

#[derive(Deserialize)]
struct ApiModel {
    id: String,
}

// ============================================================================
// Runner
// ============================================================================

/// OpenAI-compatible API client implementing the [`LlmProvider`] trait
///
/// Connects to any endpoint that speaks the `OpenAI` chat completions protocol.
/// Supports streaming via SSE, native tool calling, model discovery via
/// `/v1/models`, and all standard completion parameters (temperature,
/// `max_tokens`, `top_p`, stop sequences, response format).
pub struct OpenAiApiRunner {
    config: OpenAiApiConfig,
    client: reqwest::Client,
    models: Vec<String>,
}

impl OpenAiApiRunner {
    /// Create a new runner, optionally discovering available models
    ///
    /// Builds an internal HTTP client using the configured timeout.
    /// To inject an externally-managed HTTP client (e.g. a shared connection pool),
    /// use [`with_client`](Self::with_client) instead.
    ///
    /// Attempts to fetch the model list from the `/v1/models` endpoint.
    /// Falls back to the configured default model on failure.
    pub async fn new(config: OpenAiApiConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_else(|e| {
                warn!("HTTP client builder failed ({e}), using defaults");
                reqwest::Client::new()
            });

        Self::with_client(config, client).await
    }

    /// Create a new runner with an externally-provided HTTP client
    ///
    /// Use this when the caller owns a shared `reqwest::Client` (e.g. a pooled
    /// singleton with centralized timeout/TLS configuration). The runner will
    /// use the provided client for all HTTP requests instead of creating its own.
    ///
    /// Attempts to fetch the model list from the `/v1/models` endpoint.
    /// Falls back to the configured default model on failure.
    pub async fn with_client(config: OpenAiApiConfig, client: reqwest::Client) -> Self {
        let models = discover_models(&client, &config).await;
        let models = if models.is_empty() {
            vec![config.model.clone()]
        } else {
            models
        };

        debug!(
            base_url = %config.base_url,
            model = %config.model,
            discovered_models = models.len(),
            "OpenAI API runner initialized"
        );

        Self {
            config,
            client,
            models,
        }
    }

    /// Build an API request body from a `ChatRequest`
    fn build_api_request(&self, request: &ChatRequest, stream: bool) -> ApiRequest {
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.model)
            .to_owned();

        let messages = request.messages.iter().map(map_message).collect();

        let tools = request
            .tools
            .as_ref()
            .map(|defs| defs.iter().map(map_tool_definition).collect());

        let tool_choice = request.tool_choice.as_ref().map(map_tool_choice);
        let response_format = request.response_format.as_ref().map(map_response_format);

        ApiRequest {
            model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop: request.stop.clone(),
            stream,
            tools,
            tool_choice,
            response_format,
        }
    }

    /// Send an API request and return the raw HTTP response
    async fn send_request(
        &self,
        api_request: &ApiRequest,
    ) -> Result<reqwest::Response, RunnerError> {
        let url = format!("{}{CHAT_COMPLETIONS_PATH}", self.config.base_url);

        let mut req = self.client.post(&url).json(api_request);
        if let Some(ref key) = self.config.api_key {
            req = req.bearer_auth(key);
        }

        let response = req.send().await.map_err(|e| {
            if e.is_timeout() {
                RunnerError::timeout(format!("Request timed out: {e}"))
            } else if e.is_connect() {
                RunnerError::external_service("openai_api", format!("Connection failed: {e}"))
            } else {
                RunnerError::external_service("openai_api", e.to_string())
            }
        })?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        // Attempt to parse the error body for a better message
        let body = response.text().await.unwrap_or_default();
        Err(map_http_error(status, &body))
    }
}

// ============================================================================
// LlmProvider Implementation
// ============================================================================

#[async_trait]
impl LlmProvider for OpenAiApiRunner {
    fn name(&self) -> &'static str {
        "openai_api"
    }

    fn display_name(&self) -> &str {
        "OpenAI API"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::VISION
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::TEMPERATURE
            | LlmCapabilities::MAX_TOKENS
            | LlmCapabilities::TOP_P
            | LlmCapabilities::STOP_SEQUENCES
            | LlmCapabilities::RESPONSE_FORMAT
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.models
    }

    #[instrument(skip(self, request), fields(model))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let api_request = self.build_api_request(request, false);
        let response = self.send_request(&api_request).await?;

        let body = response.text().await.map_err(|e| {
            RunnerError::external_service("openai_api", format!("Failed to read response: {e}"))
        })?;

        let api_response: ApiResponse = serde_json::from_str(&body).map_err(|e| {
            RunnerError::external_service("openai_api", format!("Invalid response JSON: {e}"))
        })?;

        let choice =
            api_response.choices.into_iter().next().ok_or_else(|| {
                RunnerError::external_service("openai_api", "No choices in response")
            })?;

        let tool_calls = choice.message.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| ToolCallRequest {
                    id: tc.id,
                    function_name: tc.function.name,
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::String(tc.function.arguments)),
                })
                .collect()
        });

        let usage = api_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt,
            completion_tokens: u.completion,
            total_tokens: u.total,
        });

        Ok(ChatResponse {
            content: choice.message.content.unwrap_or_default(),
            model: api_response.model,
            usage,
            finish_reason: choice.finish_reason,
            warnings: None,
            tool_calls,
        })
    }

    #[instrument(skip(self, request), fields(model))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let api_request = self.build_api_request(request, true);
        let response = self.send_request(&api_request).await?;

        let (tx, rx) = mpsc::channel::<Result<StreamChunk, RunnerError>>(STREAM_CHANNEL_CAPACITY);
        let byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut stream = byte_stream;
            let mut buffer = String::new();

            loop {
                let chunk = stream.next().await;
                match chunk {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        for event_data in extract_sse_events(&mut buffer) {
                            if event_data == "[DONE]" {
                                let _ = tx
                                    .send(Ok(StreamChunk {
                                        delta: String::new(),
                                        is_final: true,
                                        finish_reason: Some("stop".to_owned()),
                                    }))
                                    .await;
                                return;
                            }

                            match serde_json::from_str::<ApiStreamResponse>(&event_data) {
                                Ok(resp) => {
                                    for choice in resp.choices {
                                        let delta = choice.delta.content.unwrap_or_default();
                                        let is_final = choice.finish_reason.is_some();

                                        if !delta.is_empty() || is_final {
                                            let chunk = StreamChunk {
                                                delta,
                                                is_final,
                                                finish_reason: choice.finish_reason,
                                            };
                                            if tx.send(Ok(chunk)).await.is_err() {
                                                return;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = tx
                                        .send(Err(RunnerError::external_service(
                                            "openai_api",
                                            format!("SSE parse error: {e}"),
                                        )))
                                        .await;
                                    return;
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        let _ = tx
                            .send(Err(RunnerError::external_service(
                                "openai_api",
                                e.to_string(),
                            )))
                            .await;
                        return;
                    }
                    None => {
                        // Byte stream ended without [DONE]; emit final chunk
                        let _ = tx
                            .send(Ok(StreamChunk {
                                delta: String::new(),
                                is_final: true,
                                finish_reason: Some("stop".to_owned()),
                            }))
                            .await;
                        return;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let url = format!("{}{MODELS_PATH}", self.config.base_url);

        let mut req = self
            .client
            .get(&url)
            .timeout(Duration::from_secs(DISCOVERY_TIMEOUT_SECS));
        if let Some(ref key) = self.config.api_key {
            req = req.bearer_auth(key);
        }

        Ok(req
            .send()
            .await
            .is_ok_and(|resp| resp.status().is_success()))
    }
}

// ============================================================================
// Mapping Helpers
// ============================================================================

/// Convert a `ChatMessage` into the API wire format
///
/// When the message has attached images, emits a multipart content array
/// per the `OpenAI` vision API: `[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:..."}}]`.
/// Otherwise, emits a simple string content value.
fn map_message(msg: &ChatMessage) -> ApiMessage {
    let has_images = msg.images.as_ref().is_some_and(|imgs| !imgs.is_empty());

    let content = if msg.content.is_empty() && msg.tool_calls.is_some() {
        serde_json::Value::Null
    } else if has_images {
        let mut parts = vec![serde_json::json!({
            "type": "text",
            "text": msg.content,
        })];
        if let Some(ref images) = msg.images {
            for img in images {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", img.mime_type, img.data),
                    },
                }));
            }
        }
        serde_json::Value::Array(parts)
    } else {
        serde_json::Value::String(msg.content.clone())
    };

    let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
        tcs.iter()
            .map(|tc| ApiToolCallRef {
                id: tc.id.clone(),
                call_type: "function".to_owned(),
                function: ApiToolCallFunction {
                    name: tc.function_name.clone(),
                    arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                },
            })
            .collect()
    });

    ApiMessage {
        role: msg.role.as_str().to_owned(),
        content,
        tool_calls,
        tool_call_id: msg.tool_call_id.clone(),
    }
}

/// Convert a `ToolDefinition` into the API wire format
fn map_tool_definition(def: &ToolDefinition) -> ApiTool {
    ApiTool {
        tool_type: "function".to_owned(),
        function: ApiFunctionDef {
            name: def.name.clone(),
            description: def.description.clone(),
            parameters: def.parameters.clone(),
        },
    }
}

/// Convert a `ToolChoice` into the `OpenAI` JSON wire format
fn map_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::Value::String("auto".to_owned()),
        ToolChoice::None => serde_json::Value::String("none".to_owned()),
        ToolChoice::Required => serde_json::Value::String("required".to_owned()),
        ToolChoice::Specific { name } => {
            serde_json::json!({
                "type": "function",
                "function": { "name": name }
            })
        }
    }
}

/// Convert a `ResponseFormat` into the `OpenAI` JSON wire format
fn map_response_format(format: &ResponseFormat) -> serde_json::Value {
    match format {
        ResponseFormat::Text => serde_json::json!({"type": "text"}),
        ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
        ResponseFormat::JsonSchema { name, schema } => {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": name, "schema": schema }
            })
        }
    }
}

/// Map an HTTP error status to a `RunnerError`
fn map_http_error(status: StatusCode, body: &str) -> RunnerError {
    let api_message = serde_json::from_str::<ApiErrorResponse>(body)
        .map_or_else(|_| body.to_owned(), |e| e.error.message);

    let detail = format!("HTTP {status}: {api_message}");

    match status.as_u16() {
        401 | 403 => RunnerError::auth_failure(detail),
        408 | 504 => RunnerError::timeout(detail),
        _ => RunnerError::external_service("openai_api", detail),
    }
}

// ============================================================================
// SSE Parsing
// ============================================================================

/// Extract complete SSE event data payloads from a buffer
///
/// Consumes complete events (delimited by `\n\n` or `\r\n\r\n`) from the
/// buffer and returns their `data:` field values. Partial events remain
/// in the buffer for the next call.
fn extract_sse_events(buffer: &mut String) -> Vec<String> {
    let mut events = Vec::new();

    loop {
        let boundary = buffer
            .find("\n\n")
            .map(|pos| (pos, 2))
            .or_else(|| buffer.find("\r\n\r\n").map(|pos| (pos, 4)));

        let Some((pos, skip)) = boundary else {
            break;
        };

        let event_block: String = buffer.drain(..pos + skip).collect();

        for line in event_block.lines() {
            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"));
            if let Some(data) = data {
                if !data.is_empty() {
                    events.push(data.to_owned());
                }
            }
        }
    }

    events
}

// ============================================================================
// Model Discovery
// ============================================================================

/// Fetch available models from the `/v1/models` endpoint
async fn discover_models(client: &reqwest::Client, config: &OpenAiApiConfig) -> Vec<String> {
    let url = format!("{}{MODELS_PATH}", config.base_url);

    let mut req = client
        .get(&url)
        .timeout(Duration::from_secs(DISCOVERY_TIMEOUT_SECS));
    if let Some(ref key) = config.api_key {
        req = req.bearer_auth(key);
    }

    let response = match req.send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            debug!(status = %r.status(), "Model discovery returned non-200");
            return Vec::new();
        }
        Err(e) => {
            debug!("Model discovery failed: {e}");
            return Vec::new();
        }
    };

    let body = match response.text().await {
        Ok(b) => b,
        Err(e) => {
            debug!("Model discovery body read failed: {e}");
            return Vec::new();
        }
    };

    match serde_json::from_str::<ApiModelsResponse>(&body) {
        Ok(resp) => {
            let mut ids: Vec<String> = resp.data.into_iter().map(|m| m.id).collect();
            ids.sort();
            ids
        }
        Err(e) => {
            debug!("Model discovery parse failed: {e}");
            Vec::new()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn config_new_sets_defaults() {
        let config = OpenAiApiConfig::new("https://example.com");
        assert_eq!(config.base_url, "https://example.com");
        assert!(config.api_key.is_none());
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }

    #[test]
    fn config_builder_methods() {
        let config = OpenAiApiConfig::new("https://example.com")
            .with_api_key("sk-test")
            .with_model("gpt-3.5-turbo")
            .with_timeout(Duration::from_secs(30));

        assert_eq!(config.api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn config_normalizes_trailing_slash() {
        let config = OpenAiApiConfig::new("https://example.com/");
        assert_eq!(config.base_url, "https://example.com");

        let config = OpenAiApiConfig::new("https://example.com///");
        assert_eq!(config.base_url, "https://example.com");
    }

    #[test]
    fn extract_sse_single_event() {
        let mut buffer = "data: {\"choices\":[]}\n\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["{\"choices\":[]}"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn extract_sse_multiple_events() {
        let mut buffer = "data: first\n\ndata: second\n\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["first", "second"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn extract_sse_done_signal() {
        let mut buffer = "data: [DONE]\n\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["[DONE]"]);
    }

    #[test]
    fn extract_sse_partial_event_stays_in_buffer() {
        let mut buffer = "data: partial".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert!(events.is_empty());
        assert_eq!(buffer, "data: partial");
    }

    #[test]
    fn extract_sse_crlf_boundary() {
        let mut buffer = "data: content\r\n\r\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["content"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn extract_sse_ignores_comments() {
        let mut buffer = ": keepalive\n\ndata: real\n\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["real"]);
    }

    #[test]
    fn extract_sse_no_space_after_data_colon() {
        let mut buffer = "data:{\"ok\":true}\n\n".to_owned();
        let events = extract_sse_events(&mut buffer);
        assert_eq!(events, vec!["{\"ok\":true}"]);
    }

    #[test]
    fn map_message_user() {
        let msg = ChatMessage::user("Hello");
        let api_msg = map_message(&msg);
        assert_eq!(api_msg.role, "user");
        assert_eq!(
            api_msg.content,
            serde_json::Value::String("Hello".to_owned())
        );
        assert!(api_msg.tool_calls.is_none());
        assert!(api_msg.tool_call_id.is_none());
    }

    #[test]
    fn map_message_assistant_with_tool_calls() {
        let mut msg = ChatMessage::assistant("");
        msg.tool_calls = Some(vec![ToolCallRequest {
            id: "call_1".to_owned(),
            function_name: "get_weather".to_owned(),
            arguments: serde_json::json!({"city": "Paris"}),
        }]);

        let api_msg = map_message(&msg);
        assert_eq!(api_msg.content, serde_json::Value::Null);
        assert!(api_msg.tool_calls.is_some());
        let tcs = api_msg.tool_calls.as_ref().unwrap(); // Safe: test assertion
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].call_type, "function");
        assert_eq!(tcs[0].function.name, "get_weather");
    }

    #[test]
    fn map_message_tool_result() {
        let msg = ChatMessage::tool("get_weather", "call_1", r#"{"temp": 72}"#);
        let api_msg = map_message(&msg);
        assert_eq!(api_msg.role, "tool");
        assert_eq!(
            api_msg.content,
            serde_json::Value::String(r#"{"temp": 72}"#.to_owned())
        );
        assert_eq!(api_msg.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn map_tool_choice_variants() {
        assert_eq!(
            map_tool_choice(&ToolChoice::Auto),
            serde_json::Value::String("auto".to_owned())
        );
        assert_eq!(
            map_tool_choice(&ToolChoice::None),
            serde_json::Value::String("none".to_owned())
        );
        assert_eq!(
            map_tool_choice(&ToolChoice::Required),
            serde_json::Value::String("required".to_owned())
        );

        let specific = map_tool_choice(&ToolChoice::Specific {
            name: "get_weather".to_owned(),
        });
        assert_eq!(specific["type"], "function");
        assert_eq!(specific["function"]["name"], "get_weather");
    }

    #[test]
    fn map_response_format_variants() {
        let text = map_response_format(&ResponseFormat::Text);
        assert_eq!(text["type"], "text");

        let json_obj = map_response_format(&ResponseFormat::JsonObject);
        assert_eq!(json_obj["type"], "json_object");

        let json_schema = map_response_format(&ResponseFormat::JsonSchema {
            name: "person".to_owned(),
            schema: serde_json::json!({"type": "object"}),
        });
        assert_eq!(json_schema["type"], "json_schema");
        assert_eq!(json_schema["json_schema"]["name"], "person");
    }

    #[test]
    fn map_tool_definition_format() {
        let def = ToolDefinition {
            name: "search".to_owned(),
            description: "Search the web".to_owned(),
            parameters: Some(serde_json::json!({"type": "object"})),
        };
        let api_tool = map_tool_definition(&def);
        assert_eq!(api_tool.tool_type, "function");
        assert_eq!(api_tool.function.name, "search");
        assert_eq!(api_tool.function.description, "Search the web");
    }

    #[test]
    fn map_http_error_auth() {
        let err = map_http_error(
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"bad key"}}"#,
        );
        assert_eq!(err.kind, ErrorKind::AuthFailure);
        assert!(err.message.contains("bad key"));
    }

    #[test]
    fn map_http_error_timeout() {
        let err = map_http_error(StatusCode::GATEWAY_TIMEOUT, "timeout");
        assert_eq!(err.kind, ErrorKind::Timeout);
    }

    #[test]
    fn map_http_error_server() {
        let err = map_http_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":{"message":"overloaded"}}"#,
        );
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("overloaded"));
    }

    #[test]
    fn map_http_error_unparseable_body() {
        let err = map_http_error(StatusCode::BAD_REQUEST, "not json");
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("not json"));
    }

    #[test]
    fn api_request_serialization() {
        let config = OpenAiApiConfig::new("https://example.com").with_model("gpt-4o");
        let runner = OpenAiApiRunner {
            config,
            client: reqwest::Client::new(),
            models: vec!["gpt-4o".to_owned()],
        };

        let request = ChatRequest::new(vec![ChatMessage::user("test")])
            .with_temperature(0.7)
            .with_max_tokens(100)
            .with_top_p(0.9)
            .with_stop(vec!["END".to_owned()])
            .with_response_format(ResponseFormat::JsonObject);

        let api_req = runner.build_api_request(&request, false);
        let json = serde_json::to_value(&api_req).unwrap(); // Safe: test assertion

        assert_eq!(json["model"], "gpt-4o");
        assert!(json["temperature"]
            .as_f64()
            .is_some_and(|v| (v - 0.7).abs() < 0.01));
        assert_eq!(json["max_tokens"], 100);
        assert!(json["top_p"]
            .as_f64()
            .is_some_and(|v| (v - 0.9).abs() < 0.01));
        assert_eq!(json["stop"], serde_json::json!(["END"]));
        assert!(!json["stream"].as_bool().unwrap()); // Safe: test assertion
        assert_eq!(json["response_format"]["type"], "json_object");
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
    }

    #[test]
    fn api_request_with_tools() {
        let config = OpenAiApiConfig::new("https://example.com");
        let runner = OpenAiApiRunner {
            config,
            client: reqwest::Client::new(),
            models: vec![],
        };

        let request = ChatRequest::new(vec![ChatMessage::user("test")])
            .with_tools(vec![ToolDefinition {
                name: "get_weather".to_owned(),
                description: "Get weather".to_owned(),
                parameters: Some(serde_json::json!({"type": "object"})),
            }])
            .with_tool_choice(ToolChoice::Required);

        let api_req = runner.build_api_request(&request, false);
        let json = serde_json::to_value(&api_req).unwrap(); // Safe: test assertion

        assert_eq!(json["tools"][0]["type"], "function");
        assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(json["tool_choice"], "required");
    }

    #[test]
    fn api_response_parsing() {
        let json = r#"{
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": "Hello!",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let resp: ApiResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.as_ref().unwrap().total, 15); // Safe: test assertion
    }

    #[test]
    fn api_response_with_tool_calls() {
        let json = r#"{
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": null
        }"#;

        let resp: ApiResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        let tc = &resp.choices[0].message.tool_calls.as_ref().unwrap()[0]; // Safe: test assertion
        assert_eq!(tc.id, "call_123");
        assert_eq!(tc.function.name, "get_weather");
        assert_eq!(tc.function.arguments, "{\"city\":\"Paris\"}");
    }

    #[test]
    fn stream_chunk_parsing() {
        let json = r#"{
            "choices": [{
                "delta": { "content": "Hello" },
                "finish_reason": null
            }]
        }"#;

        let resp: ApiStreamResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        assert_eq!(resp.choices[0].delta.content.as_deref(), Some("Hello"));
        assert!(resp.choices[0].finish_reason.is_none());
    }

    #[test]
    fn stream_chunk_final() {
        let json = r#"{
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        }"#;

        let resp: ApiStreamResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        assert!(resp.choices[0].delta.content.is_none());
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn normalize_base_url_strips_trailing_slashes() {
        assert_eq!(
            normalize_base_url("https://api.openai.com/"),
            "https://api.openai.com"
        );
        assert_eq!(
            normalize_base_url("https://api.openai.com"),
            "https://api.openai.com"
        );
        assert_eq!(
            normalize_base_url("http://localhost:11434///"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn capabilities_has_expected_flags() {
        let config = OpenAiApiConfig::new("https://example.com");
        let runner = OpenAiApiRunner {
            config,
            client: reqwest::Client::new(),
            models: vec![],
        };

        let caps = runner.capabilities();
        assert!(caps.supports_streaming());
        assert!(caps.supports_function_calling());
        assert!(caps.supports_system_messages());
        assert!(caps.supports_temperature());
        assert!(caps.supports_max_tokens());
        assert!(caps.supports_top_p());
        assert!(caps.supports_stop_sequences());
        assert!(caps.supports_response_format());
        assert!(caps.supports_vision());
        assert!(!caps.supports_sdk_tool_calling());
    }

    #[test]
    fn map_message_user_without_images() {
        let msg = ChatMessage::user("Hello");
        let api_msg = map_message(&msg);
        assert_eq!(
            api_msg.content,
            serde_json::Value::String("Hello".to_owned())
        );
    }

    #[test]
    fn map_message_user_with_images() {
        use crate::types::ImagePart;

        let img = ImagePart::new("aGVsbG8=", "image/png").unwrap(); // Safe: test assertion
        let msg = ChatMessage::user_with_images("Describe this", vec![img]);
        let api_msg = map_message(&msg);

        let content = api_msg.content.as_array().expect("should be array"); // Safe: test assertion
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Describe this");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/png;base64,aGVsbG8="
        );
    }

    #[test]
    fn map_message_user_with_empty_images_stays_string() {
        let msg = ChatMessage::user_with_images("Hello", vec![]);
        let api_msg = map_message(&msg);
        assert_eq!(
            api_msg.content,
            serde_json::Value::String("Hello".to_owned())
        );
    }

    #[test]
    fn map_message_with_images_full_serialization() {
        use crate::types::ImagePart;

        let img = ImagePart::new("AAAA", "image/jpeg").unwrap(); // Safe: test assertion
        let msg = ChatMessage::user_with_images("What is this?", vec![img]);
        let api_msg = map_message(&msg);
        let json = serde_json::to_value(&api_msg).unwrap(); // Safe: test assertion

        assert!(json["content"].is_array());
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(
            json["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,AAAA"
        );
    }

    #[test]
    fn error_response_parsing() {
        let json = r#"{"error":{"message":"Invalid API key","type":"invalid_request_error"}}"#;
        let resp: ApiErrorResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        assert_eq!(resp.error.message, "Invalid API key");
    }

    #[test]
    fn models_response_parsing() {
        let json = r#"{"data":[{"id":"gpt-4o","object":"model"},{"id":"gpt-3.5-turbo","object":"model"}]}"#;
        let resp: ApiModelsResponse = serde_json::from_str(json).unwrap(); // Safe: test assertion
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
    }
}
