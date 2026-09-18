// ABOUTME: Cohere provider backed by the v2/chat API — Command A and the Command R family, streaming and tool calling
// ABOUTME: Drops content-less tool-call-less turns before the wire (v2 400s the whole request) and decodes the typed event envelope
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Cohere Provider
//!
//! [`LlmProvider`] over Cohere's v2 chat API.
//!
//! ## Features
//!
//! - **Tool/Function Calling**: OpenAI-shaped `tools` + `tool_calls` payloads
//! - **Streaming**: Server-sent events with Cohere's typed event envelope
//!   (`message-start`, `content-delta`, `message-end`, ...). The shared SSE
//!   line parser handles framing; this module decodes the typed payloads.
//! - **Native multi-turn**: Cohere's `messages` array uses `system`/`user`/
//!   `assistant`/`tool` roles — identical to the canonical [`MessageRole`]
//!   variants so no role translation is needed.
//!
//! ## Configuration
//!
//! [`CohereConfig::from_env`] reads:
//!
//! - `COHERE_API_KEY`: required, from the Cohere dashboard
//! - `COHERE_DEFAULT_MODEL`: default `command-a-03-2025`
//! - `COHERE_MAX_RETRIES`, `COHERE_INITIAL_RETRY_DELAY_MS`,
//!   `COHERE_MAX_RETRY_DELAY_MS`: retry tuning (defaults 3 / 500 / 5000)
//!
//! ## Vendor quirks kept here
//!
//! - v2 rejects the WHOLE request when any message has neither non-empty
//!   content nor tool calls. Empty content is omitted from the wire; a
//!   content-less, tool-call-less turn is dropped; an empty tool result keeps
//!   its place with a `(no result)` body so the call/result pairing holds.
//! - Outbound tool-call arguments are a JSON-encoded string, not an object.
//! - `billed_units` is preferred over `tokens` for usage: it is what is
//!   invoiced.
//! - A 400/422 that says the model produced nothing is a provider fault, not
//!   an invalid request (see `cohere_errors`).
//!
//! ## Supported Models
//!
//! - `command-a-03-2025` (default): Command A — 256K context, agentic
//! - `command-a-reasoning-08-2025`: Reasoning-tuned Command A
//! - `command-a-vision-07-2025`: Vision-enabled Command A
//! - `command-r-plus-08-2024`: Command R+
//! - `command-r-08-2024`: Command R
//! - `command-r7b-12-2024`: Command R7B
//!
//! [`MessageRole`]: crate::types::MessageRole

use std::env;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use super::client::{self, with_retries, AttemptError, HttpRetryConfig, DEFAULT_TIMEOUT_SECS};
use super::cohere_errors::{parse_error_response, PROVIDER_NAME};
use super::sse::create_sse_stream;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
    StreamChunk, TokenUsage, ToolCallRequest, ToolDefinition,
};

/// Environment variable for the API key
const COHERE_API_KEY_ENV: &str = "COHERE_API_KEY";

/// Environment variable for the default model
const COHERE_DEFAULT_MODEL_ENV: &str = "COHERE_DEFAULT_MODEL";

/// Prefix of the retry-tuning environment variables
const COHERE_ENV_PREFIX: &str = "COHERE";

/// Default model — Command A is Cohere's flagship general-purpose model.
const DEFAULT_MODEL: &str = "command-a-03-2025";

/// Available Cohere models, longest-prefix-first so a prefix matcher resolves
/// the most specific entry.
const AVAILABLE_MODELS: &[&str] = &[
    "command-a-reasoning-08-2025",
    "command-a-vision-07-2025",
    "command-a-03-2025",
    "command-r-plus-08-2024",
    "command-r-08-2024",
    "command-r7b-12-2024",
];

/// Base URL for the Cohere v2 chat API
const API_BASE_URL: &str = "https://api.cohere.com/v2";

/// Body substituted for a tool-result message whose tool produced no output,
/// so it satisfies Cohere v2's "non-empty content or tool calls" rule without
/// being dropped (dropping it would orphan its matching tool call).
const EMPTY_TOOL_RESULT_PLACEHOLDER: &str = "(no result)";

// ============================================================================
// API Request/Response Types (Cohere v2 /chat)
// ============================================================================

/// Cohere v2 chat request
#[derive(Debug, Serialize)]
struct CohereRequest {
    model: String,
    messages: Vec<CohereMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CohereTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

/// Tool definition (OpenAI-compatible shape; Cohere v2 accepts it verbatim)
#[derive(Debug, Clone, Serialize)]
struct CohereTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: CohereFunction,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize)]
struct CohereFunction {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

/// Message structure for Cohere v2 chat — roles match the canonical
/// `system`/`user`/`assistant`/`tool` values produced by `MessageRole`.
///
/// Assistant tool-call turns arrive with empty `content` and the payload in
/// `tool_calls`; tool-result turns carry `tool_call_id`. Cohere v2 rejects any
/// message that has neither non-empty content nor tool calls, so empty content
/// is omitted from the wire and the tool fields are carried through verbatim.
#[derive(Debug, Clone, Serialize)]
struct CohereMessage {
    role: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereOutboundToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Assistant tool-call request serialized into a Cohere v2 chat message.
/// Cohere accepts the OpenAI-compatible shape verbatim, mirroring [`CohereTool`].
#[derive(Debug, Clone, Serialize)]
struct CohereOutboundToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: CohereOutboundFunction,
}

/// Function name plus JSON-encoded arguments inside an outbound tool call.
#[derive(Debug, Clone, Serialize)]
struct CohereOutboundFunction {
    name: String,
    /// Cohere v2 expects arguments as a JSON-encoded string, not an object.
    arguments: String,
}

impl From<&ChatMessage> for CohereMessage {
    fn from(msg: &ChatMessage) -> Self {
        let tool_calls = msg.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|call| CohereOutboundToolCall {
                    id: call.id.clone(),
                    call_type: "function".to_owned(),
                    function: CohereOutboundFunction {
                        name: call.function_name.clone(),
                        arguments: call.arguments.to_string(),
                    },
                })
                .collect()
        });

        Self {
            role: msg.role.as_str().to_owned(),
            content: msg.content.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

/// Cohere v2 chat response envelope
#[derive(Debug, Deserialize)]
struct CohereResponse {
    #[serde(default)]
    id: Option<String>,
    message: CohereResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    usage: Option<CohereUsage>,
}

/// Assistant message inside a Cohere v2 chat response
#[derive(Debug, Deserialize)]
struct CohereResponseMessage {
    /// Content is returned as an array of typed blocks; the concatenated
    /// `text` payloads become the flat `content` string. Tool-only responses
    /// arrive with an empty/missing content array and the model output in
    /// `tool_calls`.
    #[serde(default)]
    content: Vec<CohereContentBlock>,
    #[serde(default)]
    tool_calls: Option<Vec<CohereToolCall>>,
}

/// Single content block in a Cohere v2 message — only `text` blocks are
/// surfaced. Unknown variants are skipped via the `#[serde(other)]`
/// catch-all so new block kinds do not break parsing.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum CohereContentBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

/// Tool call payload returned by Cohere v2 (OpenAI-compatible shape)
#[derive(Debug, Clone, Deserialize)]
struct CohereToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: CohereFunctionCall,
}

/// Function call details inside a Cohere tool call
#[derive(Debug, Clone, Deserialize)]
struct CohereFunctionCall {
    name: String,
    /// Arguments are returned as a JSON-encoded string by Cohere v2.
    arguments: String,
}

/// Usage statistics. Cohere v2 reports both `billed_units` and `tokens`;
/// `billed_units` is what gets invoiced so that is what is surfaced.
#[derive(Debug, Deserialize)]
struct CohereUsage {
    #[serde(default)]
    billed_units: Option<CohereTokenCounts>,
    #[serde(default)]
    tokens: Option<CohereTokenCounts>,
}

/// Per-side token counts inside a Cohere usage payload
#[derive(Debug, Default, Deserialize)]
struct CohereTokenCounts {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

/// Cohere v2 streaming event envelope. Cohere uses typed events instead of
/// the OpenAI-style `delta` field, so the variant tag dictates how each
/// frame maps onto a [`StreamChunk`].
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum CohereStreamEvent {
    /// First event in a stream — carries the response id; nothing to emit.
    #[serde(rename = "message-start")]
    MessageStart {},
    /// Marks the start of a content block; nothing user-visible yet.
    #[serde(rename = "content-start")]
    ContentStart {},
    /// Incremental text payload — the only event forwarded as delta text.
    #[serde(rename = "content-delta")]
    ContentDelta {
        #[serde(default)]
        delta: Option<CohereContentDeltaPayload>,
    },
    /// Marks the end of a content block; nothing to emit.
    #[serde(rename = "content-end")]
    ContentEnd {},
    /// Terminal event — carries the finish reason and final usage payload.
    #[serde(rename = "message-end")]
    MessageEnd {
        #[serde(default)]
        delta: Option<CohereMessageEndDelta>,
    },
    /// Tool-call frames are accepted for streaming completeness; they are
    /// emitted as empty deltas because tool calls go through the
    /// non-streaming `complete()` path.
    #[serde(other)]
    Other,
}

/// Inner payload of a `content-delta` event
#[derive(Debug, Deserialize)]
struct CohereContentDeltaPayload {
    #[serde(default)]
    message: Option<CohereContentDeltaMessage>,
}

/// `message` wrapper inside a `content-delta` event
#[derive(Debug, Deserialize)]
struct CohereContentDeltaMessage {
    #[serde(default)]
    content: Option<CohereContentDeltaContent>,
}

/// Final text carrier inside a `content-delta` event
#[derive(Debug, Deserialize)]
struct CohereContentDeltaContent {
    #[serde(default)]
    text: Option<String>,
}

/// `delta` field of a `message-end` event
#[derive(Debug, Deserialize)]
struct CohereMessageEndDelta {
    #[serde(default)]
    finish_reason: Option<String>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Cohere provider.
#[derive(Clone)]
pub struct CohereConfig {
    /// API key from the Cohere dashboard. Redacted in `Debug`.
    pub api_key: String,
    /// Model used when the request names none
    pub model: String,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
    /// Retry policy for transient failures (429, 502, 503, network errors)
    pub retry: HttpRetryConfig,
}

impl CohereConfig {
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
    /// Returns [`RunnerError`] with `ErrorKind::Config` when `COHERE_API_KEY`
    /// is not set.
    pub fn from_env() -> Result<Self, RunnerError> {
        let api_key = env::var(COHERE_API_KEY_ENV).map_err(|_| {
            RunnerError::config(format!(
                "Missing {COHERE_API_KEY_ENV} environment variable. Get your API key from https://dashboard.cohere.com/api-keys"
            ))
        })?;
        let model = env::var(COHERE_DEFAULT_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        Ok(Self {
            model,
            retry: HttpRetryConfig::from_env(COHERE_ENV_PREFIX),
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

impl Debug for CohereConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("CohereConfig")
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

/// Cohere LLM provider using the v2 chat API
pub struct CohereProvider {
    config: CohereConfig,
    client: reqwest::Client,
    available_models: Vec<String>,
}

impl CohereProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: CohereConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: CohereConfig, client: reqwest::Client) -> Self {
        info!(
            default_model = %config.model,
            max_retries = config.retry.max_retries,
            initial_delay_ms = config.retry.initial_delay_ms,
            "Cohere provider initialized"
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

    /// Convert internal messages to Cohere v2 format.
    ///
    /// Cohere v2 rejects any message with neither non-empty content nor
    /// `tool_calls` (empty content is skipped on the wire, leaving a bare
    /// `{"role": ...}` that 400s the WHOLE request — "invalid message provided
    /// at index N: must have non-empty content or tool calls"). A content-less,
    /// tool-call-less turn — an empty or failed assistant turn left in a long
    /// history — conveys nothing and is dropped; an empty tool-result (carries
    /// `tool_call_id`) is kept with a placeholder body so the call/result
    /// pairing Cohere requires stays intact.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<CohereMessage> {
        messages
            .iter()
            .map(CohereMessage::from)
            .filter_map(|mut m| {
                if !m.content.is_empty() || m.tool_calls.is_some() {
                    Some(m)
                } else if m.tool_call_id.is_some() {
                    EMPTY_TOOL_RESULT_PLACEHOLDER.clone_into(&mut m.content);
                    Some(m)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Concatenate all text content blocks in a response message
    fn collect_message_text(content: &[CohereContentBlock]) -> String {
        let mut out = String::new();
        for block in content {
            if let CohereContentBlock::Text { text } = block {
                out.push_str(text);
            }
        }
        out
    }

    /// Resolve token usage from a Cohere usage payload, preferring billed
    /// units (what gets invoiced) over the raw token count.
    fn resolve_usage(usage: Option<CohereUsage>) -> Option<TokenUsage> {
        let usage = usage?;
        let counts = usage.billed_units.or(usage.tokens)?;
        Some(TokenUsage::new(
            counts.input_tokens,
            counts.output_tokens,
            counts.input_tokens.saturating_add(counts.output_tokens),
        ))
    }

    /// Build an authenticated HTTP request to the Cohere chat endpoint
    fn build_request(&self, cohere_request: &CohereRequest) -> RequestBuilder {
        self.client
            .post(Self::api_url("chat"))
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(cohere_request)
    }

    /// Build the request body for a `ChatRequest`
    fn build_body(&self, request: &ChatRequest, stream: bool) -> CohereRequest {
        let model = request.model.as_deref().unwrap_or(&self.config.model);
        let tools = request
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(Self::convert_tools);
        CohereRequest {
            model: model.to_owned(),
            messages: Self::convert_messages(&request.messages),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: Some(stream),
            tool_choice: tools.as_ref().map(|_| "auto".to_owned()),
            tools,
        }
    }

    /// Parse a Cohere v2 SSE data payload into a `StreamChunk`.
    ///
    /// Returns `None` for events that produce no delta (envelope frames
    /// like `message-start`, `content-start`, `content-end`, unknown
    /// variants). `message-end` produces a final empty chunk carrying the
    /// finish reason so consumers know the response has terminated.
    fn parse_stream_data(json_str: &str) -> Option<Result<StreamChunk, RunnerError>> {
        match serde_json::from_str::<CohereStreamEvent>(json_str) {
            Ok(CohereStreamEvent::ContentDelta { delta }) => {
                let text = delta
                    .and_then(|d| d.message)
                    .and_then(|m| m.content)
                    .and_then(|c| c.text)
                    .unwrap_or_default();
                if text.is_empty() {
                    None
                } else {
                    Some(Ok(StreamChunk {
                        delta: text,
                        is_final: false,
                        finish_reason: None,
                    }))
                }
            }
            Ok(CohereStreamEvent::MessageEnd { delta }) => {
                let finish_reason = delta
                    .and_then(|d| d.finish_reason)
                    .or_else(|| Some("stop".to_owned()));
                Some(Ok(StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason,
                }))
            }
            Ok(_) => None,
            Err(e) => {
                warn!("Failed to parse Cohere stream chunk: {e}");
                None
            }
        }
    }

    /// Convert tool definitions to Cohere v2's OpenAI-compatible format
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<CohereTool> {
        tools
            .iter()
            .map(|func| CohereTool {
                tool_type: "function".to_owned(),
                function: CohereFunction {
                    name: func.name.clone(),
                    description: func.description.clone(),
                    parameters: func.parameters.clone(),
                },
            })
            .collect()
    }

    /// Convert Cohere tool calls to tool-call requests. Arguments arrive as
    /// a JSON-encoded string; one that does not parse becomes `null`.
    fn convert_tool_calls(tool_calls: &[CohereToolCall]) -> Vec<ToolCallRequest> {
        tool_calls
            .iter()
            .map(|call| {
                debug!(
                    tool_call_id = %call.id,
                    tool_call_type = %call.call_type,
                    function_name = %call.function.name,
                    "Converting Cohere tool call"
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
    async fn attempt_complete(&self, body: &CohereRequest) -> Result<ChatResponse, AttemptError> {
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
                parse_error_response(status, &text),
            ));
        }

        let cohere_response: CohereResponse = serde_json::from_str(&text).map_err(|e| {
            AttemptError::permanent(RunnerError::external_service(
                PROVIDER_NAME,
                format!("Failed to parse response: {e}"),
            ))
        })?;

        let content = Self::collect_message_text(&cohere_response.message.content);
        let tool_calls = cohere_response.message.tool_calls.map(|calls| {
            info!("Cohere returned {} tool calls", calls.len());
            Self::convert_tool_calls(&calls)
        });

        debug!(
            id = ?cohere_response.id,
            content_len = content.len(),
            tool_calls = tool_calls.as_ref().map(Vec::len),
            finish_reason = ?cohere_response.finish_reason,
            "Received response from Cohere"
        );

        Ok(ChatResponse {
            content,
            model: body.model.clone(),
            usage: Self::resolve_usage(cohere_response.usage),
            finish_reason: cohere_response.finish_reason,
            warnings: None,
            tool_calls,
        })
    }

    /// One attempt at opening the stream
    async fn attempt_stream(&self, body: &CohereRequest) -> Result<ChatStream, AttemptError> {
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
                parse_error_response(status, &text),
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
impl LlmProvider for CohereProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> &str {
        "Cohere (Command)"
    }

    fn capabilities(&self) -> LlmCapabilities {
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
            "Cohere request",
            |attempt| async move {
                debug!(attempt, "Sending chat completion request to Cohere");
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
            "Cohere streaming request",
            |attempt| async move {
                debug!(
                    attempt,
                    "Sending streaming chat completion request to Cohere"
                );
                self.attempt_stream(body).await
            },
        )
        .await
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        debug!("Performing Cohere API health check");
        // The /v2/models endpoint is the lightest authenticated probe Cohere exposes
        let response = self
            .client
            .get(Self::api_url("models"))
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .map_err(|e| client::map_send_error(PROVIDER_NAME, e))?;

        let healthy = response.status().is_success();
        if healthy {
            debug!("Cohere API health check passed");
        } else {
            warn!(
                status = response.status().as_u16(),
                "Cohere API health check failed"
            );
        }
        Ok(healthy)
    }
}

impl Debug for CohereProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("CohereProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_mapping_matches_cohere_v2_schema() {
        let messages = vec![
            ChatMessage::system("be terse"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let converted = CohereProvider::convert_messages(&messages);
        let roles: Vec<&str> = converted.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["system", "user", "assistant"]);
    }

    #[test]
    fn assistant_tool_call_message_omits_empty_content_and_keeps_tool_calls() {
        // A prior assistant turn that was a pure tool call: empty text, payload
        // in `tool_calls`. Cohere v2 rejects a message with neither content nor
        // tool calls, so the wire form must omit `content` and carry the call.
        let mut msg = ChatMessage::assistant("");
        msg.tool_calls = Some(vec![ToolCallRequest {
            id: "call_1".to_owned(),
            function_name: "get_activities".to_owned(),
            arguments: serde_json::json!({ "limit": 5 }),
        }]);

        let converted = CohereProvider::convert_messages(&[msg]);
        let json = serde_json::to_value(&converted[0]).expect("serialises"); // Safe: test assertion

        assert!(
            json.get("content").is_none(),
            "empty content must be omitted so Cohere does not reject the message"
        );
        let calls = json["tool_calls"].as_array().expect("tool_calls present"); // Safe: test assertion
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_activities");
        assert_eq!(calls[0]["function"]["arguments"], "{\"limit\":5}");
    }

    #[test]
    fn empty_content_no_tool_calls_message_is_dropped() {
        // The "invalid message at index N" 400: a content-less assistant turn
        // with no tool_calls serializes to a bare {"role": ...}, which Cohere
        // rejects and sinks the WHOLE request. It must be dropped so the turn
        // still goes through.
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant(""),
            ChatMessage::assistant("real answer"),
        ];
        let converted = CohereProvider::convert_messages(&messages);
        let roles: Vec<&str> = converted.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant"]);
        assert_eq!(converted[1].content, "real answer");
    }

    #[test]
    fn empty_tool_result_kept_with_placeholder_not_dropped() {
        // A tool that returned nothing yields an empty tool-result. Dropping it
        // would orphan its matching tool call (Cohere requires call/result
        // pairing), so it is kept with a placeholder body.
        let converted =
            CohereProvider::convert_messages(&[ChatMessage::tool("get_activities", "call_1", "")]);
        assert_eq!(converted.len(), 1, "empty tool-result must be kept");
        assert_eq!(converted[0].content, EMPTY_TOOL_RESULT_PLACEHOLDER);
        assert_eq!(converted[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn tool_result_message_carries_tool_call_id_and_content() {
        let msg = ChatMessage::tool("get_activities", "call_1", "5 runs this week");
        let converted = CohereProvider::convert_messages(&[msg]);
        let json = serde_json::to_value(&converted[0]).expect("serialises"); // Safe: test assertion
        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_1");
        assert_eq!(json["content"], "5 runs this week");
    }

    #[test]
    fn plain_text_messages_serialize_without_tool_fields() {
        let converted = CohereProvider::convert_messages(&[ChatMessage::user("hi")]);
        let json = serde_json::to_value(&converted[0]).expect("serialises"); // Safe: test assertion
        assert_eq!(json["content"], "hi");
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_call_id").is_none());
    }

    #[test]
    fn collect_message_text_concatenates_text_blocks_and_skips_unknown() {
        let blocks = vec![
            CohereContentBlock::Text {
                text: "Hello, ".to_owned(),
            },
            CohereContentBlock::Unknown,
            CohereContentBlock::Text {
                text: "world".to_owned(),
            },
        ];
        assert_eq!(
            CohereProvider::collect_message_text(&blocks),
            "Hello, world"
        );
    }

    #[test]
    fn resolve_usage_prefers_billed_units_over_raw_tokens() {
        let usage = CohereUsage {
            billed_units: Some(CohereTokenCounts {
                input_tokens: 10,
                output_tokens: 20,
            }),
            tokens: Some(CohereTokenCounts {
                input_tokens: 99,
                output_tokens: 99,
            }),
        };
        let resolved = CohereProvider::resolve_usage(Some(usage)).expect("usage resolves"); // Safe: test assertion
        assert_eq!(resolved.prompt_tokens, 10);
        assert_eq!(resolved.completion_tokens, 20);
        assert_eq!(resolved.total_tokens, 30);
    }

    #[test]
    fn resolve_usage_falls_back_to_raw_tokens() {
        let usage = CohereUsage {
            billed_units: None,
            tokens: Some(CohereTokenCounts {
                input_tokens: 5,
                output_tokens: 7,
            }),
        };
        let resolved = CohereProvider::resolve_usage(Some(usage)).expect("usage resolves"); // Safe: test assertion
        assert_eq!(resolved.prompt_tokens, 5);
        assert_eq!(resolved.completion_tokens, 7);
        assert_eq!(resolved.total_tokens, 12);
    }

    #[test]
    fn parse_stream_data_extracts_content_delta_text() {
        let frame =
            r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hi"}}}}"#;
        let chunk = CohereProvider::parse_stream_data(frame)
            .expect("delta produces an event") // Safe: test assertion
            .expect("ok"); // Safe: test assertion
        assert_eq!(chunk.delta, "Hi");
        assert!(!chunk.is_final);
    }

    #[test]
    fn parse_stream_data_emits_final_chunk_on_message_end() {
        let frame = r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#;
        let chunk = CohereProvider::parse_stream_data(frame)
            .expect("message-end produces an event") // Safe: test assertion
            .expect("ok"); // Safe: test assertion
        assert!(chunk.is_final);
        assert_eq!(chunk.finish_reason.as_deref(), Some("COMPLETE"));
    }

    #[test]
    fn parse_stream_data_skips_envelope_frames() {
        let frames = [
            r#"{"type":"message-start","id":"123"}"#,
            r#"{"type":"content-start","index":0}"#,
            r#"{"type":"content-end","index":0}"#,
        ];
        for frame in frames {
            assert!(
                CohereProvider::parse_stream_data(frame).is_none(),
                "envelope frame `{frame}` should not produce a delta"
            );
        }
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let output = format!("{:?}", CohereConfig::new("super-secret"));
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
