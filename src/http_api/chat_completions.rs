// ABOUTME: The OpenAI chat-completions wire format, once: request body, response, usage, stream frames, error envelope
// ABOUTME: Groq, OpenRouter and the OpenAI-compatible provider build and read it here and keep only their vendor deltas
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # `OpenAI` chat-completions wire format
//!
//! Every provider that speaks `POST {base}/chat/completions` sends the body
//! [`CompletionRequest::new`] builds and reads the answer through
//! [`parse_response`] and [`parse_stream_frame`]. What differs between them
//! stays in their own modules: the base URL and authentication, `OpenRouter`'s
//! ranking headers and its 402 wording, Groq billing plain counts, the local
//! endpoints' "is the server running?" hints.
//!
//! The request carries what the provider's [`LlmCapabilities`] say it takes.
//! `temperature`, `max_tokens`, tools, `tool_choice` and the tool-call history
//! (an assistant turn's `tool_calls`, a tool result's `tool_call_id`) go to
//! every endpoint. `top_p`, `stop`, `response_format` and image parts go only
//! to a provider that advertises `TOP_P`, `STOP_SEQUENCES`, `RESPONSE_FORMAT`
//! and `VISION`: for any other, the capability guard's contract is that the
//! value is ignored, so it is never put on the wire.

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, info};

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, LlmCapabilities, ResponseFormat, RunnerError,
    StreamChunk, TokenUsage, ToolCallRequest, ToolChoice, ToolDefinition,
};

/// The only tool type the chat-completions API defines
const FUNCTION_TYPE: &str = "function";

// ============================================================================
// Request
// ============================================================================

/// A chat-completions request body
#[derive(Debug, Serialize)]
pub(super) struct CompletionRequest {
    /// Model the request targets: the request's own, else the provider default
    pub(super) model: String,
    /// Conversation in wire shape
    pub(super) messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    /// Whether the endpoint answers with an SSE stream
    pub(super) stream: bool,
    /// Tool definitions; absent when the request carries none
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
}

impl CompletionRequest {
    /// Build the body for `request`, filling only what `capabilities` covers.
    ///
    /// An empty tool list sends no `tools` (the API rejects an empty array),
    /// and `tool_choice` travels only beside tools: the request's own choice
    /// when it names one, `auto` otherwise.
    pub(super) fn new(
        request: &ChatRequest,
        default_model: &str,
        stream: bool,
        capabilities: LlmCapabilities,
    ) -> Self {
        let vision = capabilities.supports_vision();
        let tools: Option<Vec<WireTool>> = request
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(|defs| defs.iter().map(WireTool::from).collect());
        let tool_choice = tools.as_ref().map(|_| {
            request
                .tool_choice
                .as_ref()
                .map_or_else(|| Value::String("auto".to_owned()), tool_choice_value)
        });
        Self {
            model: request.model.as_deref().unwrap_or(default_model).to_owned(),
            messages: request
                .messages
                .iter()
                .map(|msg| WireMessage::new(msg, vision))
                .collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p.filter(|_| capabilities.supports_top_p()),
            stop: request
                .stop
                .clone()
                .filter(|_| capabilities.supports_stop_sequences()),
            stream,
            tools,
            tool_choice,
            response_format: request
                .response_format
                .as_ref()
                .filter(|_| capabilities.supports_response_format())
                .map(response_format_value),
        }
    }
}

/// One message in wire shape
#[derive(Debug, Serialize)]
pub(super) struct WireMessage {
    /// `system`, `user`, `assistant` or `tool`
    pub(super) role: &'static str,
    /// A string, a content-part array when images travel, or `null` for an
    /// assistant turn that only calls tools
    content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl WireMessage {
    /// Convert a message; its images become `image_url` parts when `vision`
    /// holds and are left out otherwise.
    fn new(msg: &ChatMessage, vision: bool) -> Self {
        let images = msg
            .images
            .as_deref()
            .filter(|imgs| vision && !imgs.is_empty());
        let content = if msg.content.is_empty() && msg.tool_calls.is_some() {
            Value::Null
        } else if let Some(images) = images {
            let mut parts = vec![json!({ "type": "text", "text": msg.content })];
            parts.extend(images.iter().map(|img| {
                json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{};base64,{}", img.mime_type, img.data) },
                })
            }));
            Value::Array(parts)
        } else {
            Value::String(msg.content.clone())
        };
        Self {
            role: msg.role.as_str(),
            content,
            tool_calls: msg
                .tool_calls
                .as_ref()
                .map(|calls| calls.iter().map(WireToolCall::from).collect()),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

/// A tool call the assistant made, replayed in the conversation history
#[derive(Debug, Serialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: WireFunctionCall,
}

impl From<&ToolCallRequest> for WireToolCall {
    fn from(call: &ToolCallRequest) -> Self {
        Self {
            id: call.id.clone(),
            call_type: FUNCTION_TYPE,
            function: WireFunctionCall {
                name: call.function_name.clone(),
                arguments: serde_json::to_string(&call.arguments).unwrap_or_default(),
            },
        }
    }
}

/// Function name and JSON-encoded arguments of a replayed tool call
#[derive(Debug, Serialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

/// A tool the model may call
#[derive(Debug, Serialize)]
pub(super) struct WireTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: WireFunctionDef,
}

impl From<&ToolDefinition> for WireTool {
    fn from(def: &ToolDefinition) -> Self {
        Self {
            tool_type: FUNCTION_TYPE,
            function: WireFunctionDef {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.parameters.clone(),
            },
        }
    }
}

/// Name, description and JSON Schema of a callable function
#[derive(Debug, Serialize)]
struct WireFunctionDef {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

/// `tool_choice` in wire shape
fn tool_choice_value(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".to_owned()),
        ToolChoice::None => Value::String("none".to_owned()),
        ToolChoice::Required => Value::String("required".to_owned()),
        ToolChoice::Specific { name } => {
            json!({ "type": FUNCTION_TYPE, "function": { "name": name } })
        }
    }
}

/// `response_format` in wire shape
fn response_format_value(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema { name, schema } => json!({
            "type": "json_schema",
            "json_schema": { "name": name, "schema": schema },
        }),
    }
}

// ============================================================================
// Response
// ============================================================================

/// A non-streaming chat-completions response
#[derive(Debug, Deserialize)]
struct CompletionResponse {
    model: String,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

/// One completion choice
#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

/// The assistant message of a choice
#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

/// A tool call the model asks for
#[derive(Debug, Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type", default)]
    call_type: Option<String>,
    function: FunctionCall,
}

/// The function and JSON-encoded arguments of a requested tool call
#[derive(Debug, Deserialize)]
struct FunctionCall {
    name: String,
    arguments: String,
}

/// The usage block of a response
#[derive(Debug, Deserialize)]
pub(super) struct Usage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
    /// Breakdown of `prompt_tokens`, carrying the cache-read share.
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    /// Breakdown of `completion_tokens`, carrying the reasoning-token share
    /// that reasoning models bill as output but exclude from the count.
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

/// The `prompt_tokens_details` sub-object of a usage block
#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

/// The `completion_tokens_details` sub-object, present on reasoning models
#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

impl Usage {
    /// The counts with their cached and reasoning shares. No cache-WRITE
    /// count exists in this API shape: writes are implicit and unbilled, so
    /// `None` is accurate, not unknown.
    pub(super) fn with_details(self) -> TokenUsage {
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

    /// The three counts alone, for a vendor whose breakdown is not billed.
    pub(super) fn counts_only(self) -> TokenUsage {
        TokenUsage::new(self.prompt, self.completion, self.total)
    }
}

/// Parse a non-streaming response body into a [`ChatResponse`], reading
/// usage through `usage`.
///
/// # Errors
///
/// `ExternalService` under `provider` when the body is not a completion or
/// carries no choice.
pub(super) fn parse_response(
    provider: &'static str,
    body: &str,
    usage: fn(Usage) -> TokenUsage,
) -> Result<ChatResponse, RunnerError> {
    let response: CompletionResponse = serde_json::from_str(body).map_err(|e| {
        RunnerError::external_service(provider, format!("Failed to parse response: {e}"))
    })?;
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| RunnerError::external_service(provider, "API returned no choices"))?;

    let content = choice.message.content.unwrap_or_default();
    let tool_calls = choice.message.tool_calls.map(|calls| {
        info!(provider, count = calls.len(), "response carries tool calls");
        tool_call_requests(provider, &calls)
    });
    debug!(
        provider,
        content_len = content.len(),
        tool_calls = tool_calls.as_ref().map(Vec::len),
        finish_reason = ?choice.finish_reason,
        "Received response"
    );

    Ok(ChatResponse {
        content,
        model: response.model,
        usage: response.usage.map(usage),
        finish_reason: choice.finish_reason,
        warnings: None,
        tool_calls,
    })
}

/// Convert requested tool calls. Arguments arrive as a JSON-encoded string;
/// one that does not parse becomes `null`.
fn tool_call_requests(provider: &'static str, calls: &[ToolCall]) -> Vec<ToolCallRequest> {
    calls
        .iter()
        .map(|call| {
            debug!(
                provider,
                tool_call_id = %call.id,
                tool_call_type = ?call.call_type,
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

// ============================================================================
// Streaming
// ============================================================================

/// One SSE `data:` frame of a streamed completion
#[derive(Debug, Deserialize)]
struct StreamFrame {
    choices: Vec<StreamChoice>,
}

/// One choice of a stream frame. A frame that carries only annotations
/// (Azure's content-filter results) has no `delta` and reads as empty.
#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
    finish_reason: Option<String>,
}

/// The incremental content of a stream frame
#[derive(Debug, Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

/// Parse one SSE `data:` payload into a [`StreamChunk`].
///
/// A frame with no choice (a usage-only trailer) yields nothing, and the SSE
/// layer drops empty non-final deltas. A payload that is not a completion
/// frame — a vendor's mid-stream error envelope among them — ends the stream
/// with an error rather than vanishing into a truncated answer.
pub(super) fn parse_stream_frame(
    provider: &'static str,
    json: &str,
) -> Option<Result<StreamChunk, RunnerError>> {
    match serde_json::from_str::<StreamFrame>(json) {
        Ok(frame) => {
            let choice = frame.choices.into_iter().next()?;
            Some(Ok(StreamChunk {
                delta: choice.delta.content.unwrap_or_default(),
                is_final: choice.finish_reason.is_some(),
                finish_reason: choice.finish_reason,
            }))
        }
        Err(e) => Some(Err(RunnerError::external_service(
            provider,
            format!("SSE parse error: {e}"),
        ))),
    }
}

// ============================================================================
// Errors
// ============================================================================

/// The `{"error": {...}}` envelope a non-success response carries
#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

/// The vendor's own description of a failure
#[derive(Debug, Deserialize)]
pub(super) struct ErrorDetail {
    /// Human-readable message
    pub(super) message: String,
    /// Error class, when the vendor names one
    #[serde(rename = "type")]
    error_type: Option<String>,
}

impl ErrorDetail {
    /// `<type> - <message>`, with `unknown` for an untyped error
    pub(super) fn typed_message(&self) -> String {
        format!(
            "{} - {}",
            self.error_type.as_deref().unwrap_or("unknown"),
            self.message
        )
    }
}

/// The error envelope of `body`, when it is one
pub(super) fn error_detail(body: &str) -> Option<ErrorDetail> {
    serde_json::from_str::<ErrorEnvelope>(body)
        .ok()
        .map(|envelope| envelope.error)
}

/// The message a hosted vendor's non-success response carries: its typed
/// message when the body is the error envelope, `HTTP <status>` otherwise.
pub(super) fn describe_error(provider: &'static str, status: StatusCode, body: &str) -> String {
    error_detail(body).map_or_else(
        || {
            debug!(
                provider,
                status = status.as_u16(),
                body_preview = %body.chars().take(200).collect::<String>(),
                "non-JSON error response"
            );
            format!("HTTP {status}")
        },
        |detail| detail.typed_message(),
    )
}
