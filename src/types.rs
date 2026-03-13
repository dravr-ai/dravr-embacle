// ABOUTME: Core types for CLI LLM runners — standalone definitions independent of pierre-core
// ABOUTME: Provides LlmProvider trait, ChatRequest/Response, error types, and capability flags
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Core Types
//!
//! Self-contained type definitions for the CLI LLM runners library.
//! These types mirror the LLM provider contract without requiring
//! any external platform dependency.

use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

// ============================================================================
// Error Type
// ============================================================================

/// Error type for CLI LLM runner operations
#[derive(Debug, Clone)]
#[must_use]
pub struct RunnerError {
    /// Error category
    pub kind: ErrorKind,
    /// Human-readable error message
    pub message: String,
}

/// Categories of errors produced by CLI runners
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Internal runner error (bug, unexpected state)
    Internal,
    /// External service error (CLI tool failure, bad response)
    ExternalService,
    /// CLI command exceeded its configured timeout
    Timeout,
    /// Binary not found or not executable
    BinaryNotFound,
    /// Authentication or authorization failure
    AuthFailure,
    /// Configuration error
    Config,
    /// Guardrail policy violation (request or response rejected)
    Guardrail,
}

impl ErrorKind {
    /// Whether this error category represents a transient failure worth retrying.
    ///
    /// Transient errors (timeouts, external service issues) may succeed on a
    /// subsequent attempt. Permanent errors (config, auth, missing binary) will
    /// not benefit from retries.
    #[must_use]
    pub const fn is_transient(self) -> bool {
        matches!(self, Self::Timeout | Self::ExternalService)
    }
}

impl RunnerError {
    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }

    /// Create an external service error
    pub fn external_service(service: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::ExternalService,
            message: format!("{}: {}", service.into(), message.into()),
        }
    }

    /// Create a binary-not-found error
    pub fn binary_not_found(binary: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::BinaryNotFound,
            message: format!("Binary not found: {}", binary.into()),
        }
    }

    /// Create an auth failure error
    pub fn auth_failure(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::AuthFailure,
            message: message.into(),
        }
    }

    /// Create a config error
    pub fn config(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Config,
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Timeout,
            message: message.into(),
        }
    }

    /// Create a guardrail violation error
    pub fn guardrail(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Guardrail,
            message: message.into(),
        }
    }
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for RunnerError {}

// ============================================================================
// Capability Flags
// ============================================================================

bitflags::bitflags! {
    /// LLM provider capability flags using bitflags for efficient storage
    ///
    /// Indicates which features a provider supports. Used by the system to
    /// select appropriate providers and configure request handling.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
    pub struct LlmCapabilities: u16 {
        /// Provider supports streaming responses
        const STREAMING         = 0b0000_0000_0001;
        /// Provider supports function/tool calling
        const FUNCTION_CALLING  = 0b0000_0000_0010;
        /// Provider supports vision/image input
        const VISION            = 0b0000_0000_0100;
        /// Provider supports JSON mode output
        const JSON_MODE         = 0b0000_0000_1000;
        /// Provider supports system messages
        const SYSTEM_MESSAGES   = 0b0000_0001_0000;
        /// Provider supports SDK-managed tool calling (tool loop handled by SDK, not by caller)
        const SDK_TOOL_CALLING  = 0b0000_0010_0000;
        /// Provider supports temperature parameter
        const TEMPERATURE       = 0b0000_0100_0000;
        /// Provider supports max_tokens parameter
        const MAX_TOKENS        = 0b0000_1000_0000;
        /// Provider supports top_p (nucleus sampling) parameter
        const TOP_P             = 0b0001_0000_0000;
        /// Provider supports stop sequences parameter
        const STOP_SEQUENCES    = 0b0010_0000_0000;
        /// Provider supports response format control (JSON mode, JSON Schema)
        const RESPONSE_FORMAT   = 0b0100_0000_0000;
    }
}

impl LlmCapabilities {
    /// Create capabilities for a basic text-only provider
    #[must_use]
    pub const fn text_only() -> Self {
        Self::STREAMING.union(Self::SYSTEM_MESSAGES)
    }

    /// Create capabilities for a full-featured provider (like Gemini Pro)
    #[must_use]
    pub const fn full_featured() -> Self {
        Self::STREAMING
            .union(Self::FUNCTION_CALLING)
            .union(Self::VISION)
            .union(Self::JSON_MODE)
            .union(Self::SYSTEM_MESSAGES)
    }

    /// Check if streaming is supported
    #[must_use]
    pub const fn supports_streaming(&self) -> bool {
        self.contains(Self::STREAMING)
    }

    /// Check if function calling is supported
    #[must_use]
    pub const fn supports_function_calling(&self) -> bool {
        self.contains(Self::FUNCTION_CALLING)
    }

    /// Check if vision is supported
    #[must_use]
    pub const fn supports_vision(&self) -> bool {
        self.contains(Self::VISION)
    }

    /// Check if JSON mode is supported
    #[must_use]
    pub const fn supports_json_mode(&self) -> bool {
        self.contains(Self::JSON_MODE)
    }

    /// Check if system messages are supported
    #[must_use]
    pub const fn supports_system_messages(&self) -> bool {
        self.contains(Self::SYSTEM_MESSAGES)
    }

    /// Check if SDK-managed tool calling is supported
    #[must_use]
    pub const fn supports_sdk_tool_calling(&self) -> bool {
        self.contains(Self::SDK_TOOL_CALLING)
    }

    /// Check if temperature parameter is supported
    #[must_use]
    pub const fn supports_temperature(&self) -> bool {
        self.contains(Self::TEMPERATURE)
    }

    /// Check if `max_tokens` parameter is supported
    #[must_use]
    pub const fn supports_max_tokens(&self) -> bool {
        self.contains(Self::MAX_TOKENS)
    }

    /// Check if `top_p` parameter is supported
    #[must_use]
    pub const fn supports_top_p(&self) -> bool {
        self.contains(Self::TOP_P)
    }

    /// Check if stop sequences parameter is supported
    #[must_use]
    pub const fn supports_stop_sequences(&self) -> bool {
        self.contains(Self::STOP_SEQUENCES)
    }

    /// Check if response format control is supported
    #[must_use]
    pub const fn supports_response_format(&self) -> bool {
        self.contains(Self::RESPONSE_FORMAT)
    }
}

// ============================================================================
// Message Types
// ============================================================================

/// Role of a message in the conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System instruction message
    System,
    /// User input message
    User,
    /// Assistant response message
    Assistant,
    /// Tool result message
    Tool,
}

impl MessageRole {
    /// Convert to string representation for API calls
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Supported MIME types for image content
const VALID_IMAGE_MIME_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];

/// An image attached to a chat message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImagePart {
    /// Base64-encoded image data
    pub data: String,
    /// MIME type (e.g., "image/png", "image/jpeg")
    pub mime_type: String,
}

impl ImagePart {
    /// Create a new image part, validating the MIME type.
    ///
    /// Accepted MIME types: `image/png`, `image/jpeg`, `image/webp`, `image/gif`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the MIME type is not supported.
    pub fn new(data: impl Into<String>, mime_type: impl Into<String>) -> Result<Self, RunnerError> {
        let mime_type = mime_type.into();
        if !VALID_IMAGE_MIME_TYPES.contains(&mime_type.as_str()) {
            return Err(RunnerError::config(format!(
                "Unsupported image MIME type '{mime_type}'; expected one of: {}",
                VALID_IMAGE_MIME_TYPES.join(", ")
            )));
        }
        Ok(Self {
            data: data.into(),
            mime_type,
        })
    }
}

/// A single message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
    /// Images attached to the message (only meaningful for `User` role)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImagePart>>,
    /// Tool calls requested by the assistant (only for `Assistant` role)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    /// ID of the tool call this message responds to (only for `Tool` role)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Function name for tool result messages
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Create a new chat message
    #[must_use]
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a system message
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    /// Create a user message
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    /// Create a user message with attached images
    #[must_use]
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImagePart>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            images: Some(images),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an assistant message
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }

    /// Create a tool result message
    #[must_use]
    pub fn tool(
        name: impl Into<String>,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            images: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

// ============================================================================
// Tool Calling Types
// ============================================================================

/// A tool call requested by the assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Unique identifier for this tool call
    pub id: String,
    /// Name of the function to call
    pub function_name: String,
    /// JSON-encoded arguments for the function
    pub arguments: serde_json::Value,
}

/// Definition of a tool that can be called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// JSON Schema describing the function parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Controls which tools the model may call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    /// Model decides whether to call tools
    Auto,
    /// Model will not call any tools
    None,
    /// Model must call at least one tool
    Required,
    /// Model must call the specified function
    Specific {
        /// Name of the function to call
        name: String,
    },
}

/// Controls the response format from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    /// Default text response
    Text,
    /// Force JSON object output
    JsonObject,
    /// Force JSON output conforming to a specific schema
    JsonSchema {
        /// Schema name for identification
        name: String,
        /// JSON Schema the response must conform to
        schema: serde_json::Value,
    },
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Configuration for a chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Conversation messages
    pub messages: Vec<ChatMessage>,
    /// Model identifier (provider-specific)
    pub model: Option<String>,
    /// Temperature for response randomness (0.0 - 2.0).
    ///
    /// Support depends on each provider's [`LlmCapabilities::TEMPERATURE`] flag.
    /// Use [`validate_capabilities`](crate::validate_capabilities) to check
    /// before dispatch.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    ///
    /// Support depends on each provider's [`LlmCapabilities::MAX_TOKENS`] flag.
    /// Use [`validate_capabilities`](crate::validate_capabilities) to check
    /// before dispatch.
    pub max_tokens: Option<u32>,
    /// Whether to stream the response
    pub stream: bool,
    /// Tool definitions available for the model to call
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls which tools the model may call
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Nucleus sampling parameter (0.0 - 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences that halt generation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Control over the response format (text, JSON, or schema-validated JSON)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

impl ChatRequest {
    /// Create a new chat request with messages
    #[must_use]
    pub const fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            model: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            stop: None,
            response_format: None,
        }
    }

    /// Set the model to use
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the temperature
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Enable streaming
    #[must_use]
    pub const fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    /// Set the tool definitions
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool choice
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the `top_p` (nucleus sampling) parameter
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    #[must_use]
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set the response format
    #[must_use]
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Check whether any message in this request contains images
    #[must_use]
    pub fn has_images(&self) -> bool {
        self.messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
    }
}

/// Response from a chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Generated message content
    pub content: String,
    /// Model used for generation
    pub model: String,
    /// Token usage statistics
    pub usage: Option<TokenUsage>,
    /// Finish reason (stop, length, etc.)
    pub finish_reason: Option<String>,
    /// Warnings about unsupported request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
    /// Tool calls requested by the model (populated by providers with native function calling)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total tokens used
    pub total_tokens: u32,
}

/// A chunk of a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Content delta for this chunk
    pub delta: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Finish reason if final
    pub finish_reason: Option<String>,
}

/// Stream type for chat completion responses
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, RunnerError>> + Send>>;

// ============================================================================
// Provider Trait
// ============================================================================

/// LLM provider trait for chat completion
///
/// Implement this trait to add a new LLM runner. Each runner wraps
/// a CLI tool and translates between the chat protocol and the
/// tool's native interface.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Unique provider identifier (e.g., `claude_code`, `copilot`)
    fn name(&self) -> &'static str;

    /// Human-readable display name for the provider
    fn display_name(&self) -> &str;

    /// Provider capabilities (streaming, function calling, etc.)
    fn capabilities(&self) -> LlmCapabilities;

    /// Default model to use if not specified in request
    fn default_model(&self) -> &str;

    /// Available models for this provider
    fn available_models(&self) -> &[String];

    /// Perform a chat completion (non-streaming)
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError>;

    /// Perform a streaming chat completion
    ///
    /// Returns a stream of chunks that can be consumed incrementally.
    /// Falls back to non-streaming if not supported.
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError>;

    /// Check if the provider is healthy and ready to serve requests
    async fn health_check(&self) -> Result<bool, RunnerError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn is_transient_classification() {
        assert!(ErrorKind::Timeout.is_transient());
        assert!(ErrorKind::ExternalService.is_transient());
        assert!(!ErrorKind::Internal.is_transient());
        assert!(!ErrorKind::BinaryNotFound.is_transient());
        assert!(!ErrorKind::AuthFailure.is_transient());
        assert!(!ErrorKind::Config.is_transient());
        assert!(!ErrorKind::Guardrail.is_transient());
    }

    #[test]
    fn tool_call_request_serde_round_trip() {
        let tc = ToolCallRequest {
            id: "call_1".to_owned(),
            function_name: "get_weather".to_owned(),
            arguments: json!({"city": "Paris"}),
        };
        let json = serde_json::to_string(&tc).unwrap();
        let deserialized: ToolCallRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "call_1");
        assert_eq!(deserialized.function_name, "get_weather");
        assert_eq!(deserialized.arguments["city"], "Paris");
    }

    #[test]
    fn tool_definition_serde_round_trip() {
        let td = ToolDefinition {
            name: "search".to_owned(),
            description: "Search the web".to_owned(),
            parameters: Some(json!({"type": "object", "properties": {"q": {"type": "string"}}})),
        };
        let json = serde_json::to_string(&td).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "search");
        assert!(deserialized.parameters.is_some());
    }

    #[test]
    fn tool_definition_without_parameters() {
        let td = ToolDefinition {
            name: "ping".to_owned(),
            description: "Check connectivity".to_owned(),
            parameters: None,
        };
        let json = serde_json::to_string(&td).unwrap();
        assert!(!json.contains("parameters"));
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert!(deserialized.parameters.is_none());
    }

    #[test]
    fn tool_choice_serde_variants() {
        let auto = ToolChoice::Auto;
        let json = serde_json::to_string(&auto).unwrap();
        let deserialized: ToolChoice = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ToolChoice::Auto));

        let none = ToolChoice::None;
        let json = serde_json::to_string(&none).unwrap();
        let deserialized: ToolChoice = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ToolChoice::None));

        let required = ToolChoice::Required;
        let json = serde_json::to_string(&required).unwrap();
        let deserialized: ToolChoice = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ToolChoice::Required));

        let specific = ToolChoice::Specific {
            name: "get_weather".to_owned(),
        };
        let json = serde_json::to_string(&specific).unwrap();
        let deserialized: ToolChoice = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ToolChoice::Specific { name } if name == "get_weather"));
    }

    #[test]
    fn response_format_serde_variants() {
        let text = ResponseFormat::Text;
        let json = serde_json::to_string(&text).unwrap();
        let deserialized: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ResponseFormat::Text));

        let json_obj = ResponseFormat::JsonObject;
        let json = serde_json::to_string(&json_obj).unwrap();
        let deserialized: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, ResponseFormat::JsonObject));

        let json_schema = ResponseFormat::JsonSchema {
            name: "person".to_owned(),
            schema: json!({"type": "object", "properties": {"name": {"type": "string"}}}),
        };
        let json = serde_json::to_string(&json_schema).unwrap();
        let deserialized: ResponseFormat = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(deserialized, ResponseFormat::JsonSchema { name, .. } if name == "person")
        );
    }

    #[test]
    fn chat_message_tool_constructor() {
        let msg = ChatMessage::tool("get_weather", "call_1", r#"{"temp": 72}"#);
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.content, r#"{"temp": 72}"#);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msg.name.as_deref(), Some("get_weather"));
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn chat_message_regular_constructors_have_none_tool_fields() {
        let user = ChatMessage::user("hello");
        assert!(user.tool_calls.is_none());
        assert!(user.tool_call_id.is_none());
        assert!(user.name.is_none());
        assert!(user.images.is_none());
    }

    #[test]
    fn image_part_valid_mime_types() {
        for mime in &["image/png", "image/jpeg", "image/webp", "image/gif"] {
            let part = ImagePart::new("base64data", *mime);
            assert!(part.is_ok(), "Expected {mime} to be valid");
        }
    }

    #[test]
    fn image_part_invalid_mime_type() {
        let err = ImagePart::new("data", "image/bmp").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("image/bmp"));
    }

    #[test]
    fn user_with_images_constructor() {
        let img = ImagePart::new("aGVsbG8=", "image/png").unwrap();
        let msg = ChatMessage::user_with_images("describe this", vec![img]);
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "describe this");
        let images = msg.images.as_ref().unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/png");
    }

    #[test]
    fn chat_request_has_images() {
        let img = ImagePart::new("data", "image/jpeg").unwrap();
        let with = ChatRequest::new(vec![ChatMessage::user_with_images("x", vec![img])]);
        assert!(with.has_images());

        let without = ChatRequest::new(vec![ChatMessage::user("text only")]);
        assert!(!without.has_images());
    }

    #[test]
    fn chat_request_has_images_empty_vec() {
        let msg = ChatMessage::user_with_images("x", vec![]);
        let req = ChatRequest::new(vec![msg]);
        assert!(!req.has_images());
    }

    #[test]
    fn image_part_serde_round_trip() {
        let img = ImagePart::new("aGVsbG8=", "image/png").unwrap();
        let json = serde_json::to_string(&img).unwrap();
        let deserialized: ImagePart = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, img);
    }

    #[test]
    fn chat_message_with_images_serde_round_trip() {
        let img = ImagePart::new("data", "image/jpeg").unwrap();
        let msg = ChatMessage::user_with_images("describe", vec![img]);
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.images.as_ref().unwrap().len(), 1);
        assert_eq!(deserialized.images.unwrap()[0].mime_type, "image/jpeg");
    }

    #[test]
    fn chat_message_without_images_backward_compat() {
        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.images.is_none());
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn chat_message_images_not_serialized_when_none() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("images"));
    }

    #[test]
    fn chat_request_builder_methods() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_tools(vec![ToolDefinition {
                name: "test".to_owned(),
                description: "test fn".to_owned(),
                parameters: None,
            }])
            .with_tool_choice(ToolChoice::Required)
            .with_top_p(0.9)
            .with_stop(vec!["END".to_owned()])
            .with_response_format(ResponseFormat::JsonObject);

        assert!(req.tools.is_some());
        assert!(matches!(req.tool_choice, Some(ToolChoice::Required)));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.stop.as_ref().unwrap()[0], "END");
        assert!(matches!(
            req.response_format,
            Some(ResponseFormat::JsonObject)
        ));
    }

    #[test]
    fn message_role_tool_as_str() {
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn capability_flags_new_fields() {
        let caps = LlmCapabilities::TOP_P
            | LlmCapabilities::STOP_SEQUENCES
            | LlmCapabilities::RESPONSE_FORMAT;
        assert!(caps.supports_top_p());
        assert!(caps.supports_stop_sequences());
        assert!(caps.supports_response_format());

        let empty = LlmCapabilities::empty();
        assert!(!empty.supports_top_p());
        assert!(!empty.supports_stop_sequences());
        assert!(!empty.supports_response_format());
    }
}
