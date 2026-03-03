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

use std::any::Any;
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
}

impl MessageRole {
    /// Convert to string representation for API calls
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// A single message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message
    #[must_use]
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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

    /// Create an assistant message
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }
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
    fn display_name(&self) -> &'static str;

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

    /// Downcast to a concrete type for provider-specific operations
    fn as_any(&self) -> &dyn Any;
}
