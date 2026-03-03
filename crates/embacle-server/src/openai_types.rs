// ABOUTME: OpenAI-compatible request/response envelope types for the REST API
// ABOUTME: Maps between OpenAI chat completion format and embacle ChatRequest/ChatResponse
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use serde::{Deserialize, Serialize};

// ============================================================================
// Request Types
// ============================================================================

/// OpenAI-compatible chat completion request
///
/// Accepts either a single model string or an array of model strings
/// for multiplex mode. Each model string may contain a provider prefix
/// (e.g., "copilot:gpt-4o") parsed by the provider resolver.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier(s) — single string or array for multiplex
    pub model: ModelField,
    /// Conversation messages
    pub messages: Vec<ChatCompletionMessage>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
    /// Temperature for response randomness (0.0 - 2.0)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Enable strict capability checking (reject unsupported parameters)
    #[serde(default)]
    pub strict_capabilities: Option<bool>,
}

/// A model field that can be either a single string or an array of strings
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ModelField {
    /// Single model string (standard `OpenAI`)
    Single(String),
    /// Array of model strings (multiplex extension)
    Multiple(Vec<String>),
}

/// OpenAI-compatible message in a chat completion request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

// ============================================================================
// Response Types (non-streaming)
// ============================================================================

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique response identifier
    pub id: String,
    /// Object type (always "chat.completion")
    pub object: &'static str,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for generation
    pub model: String,
    /// Response choices (always one for embacle)
    pub choices: Vec<Choice>,
    /// Token usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Warnings about unsupported request parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

/// A single choice in a chat completion response
#[derive(Debug, Serialize)]
pub struct Choice {
    /// Choice index (always 0)
    pub index: u32,
    /// Generated message
    pub message: ResponseMessage,
    /// Reason the generation stopped
    pub finish_reason: Option<String>,
}

/// Message in a chat completion response
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    /// Role (always "assistant")
    pub role: &'static str,
    /// Generated content
    pub content: String,
}

/// Token usage statistics
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Tokens in the prompt
    #[serde(rename = "prompt_tokens")]
    pub prompt: u32,
    /// Tokens in the completion
    #[serde(rename = "completion_tokens")]
    pub completion: u32,
    /// Total tokens
    #[serde(rename = "total_tokens")]
    pub total: u32,
}

// ============================================================================
// Streaming Response Types
// ============================================================================

/// OpenAI-compatible streaming chunk
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// Unique response identifier (same across all chunks)
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: &'static str,
    /// Unix timestamp of creation
    pub created: u64,
    /// Model used for generation
    pub model: String,
    /// Streaming choices
    pub choices: Vec<ChunkChoice>,
}

/// A single choice in a streaming chunk
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    /// Choice index (always 0)
    pub index: u32,
    /// Content delta
    pub delta: Delta,
    /// Reason the generation stopped (only on final chunk)
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk
#[derive(Debug, Serialize)]
pub struct Delta {
    /// Role (only present on first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Content token (empty string on role-only or final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ============================================================================
// Multiplex Response (non-standard extension)
// ============================================================================

/// Response for multiplex requests (multiple providers)
#[derive(Debug, Serialize)]
pub struct MultiplexResponse {
    /// Unique response identifier
    pub id: String,
    /// Object type (always "chat.completion.multiplex")
    pub object: &'static str,
    /// Unix timestamp of creation
    pub created: u64,
    /// Per-provider results
    pub results: Vec<MultiplexProviderResult>,
    /// Human-readable summary
    pub summary: String,
}

/// Result from a single provider in a multiplex request
#[derive(Debug, Serialize)]
pub struct MultiplexProviderResult {
    /// Provider identifier
    pub provider: String,
    /// Model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Response content (None on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Error message (None on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock time in milliseconds
    pub duration_ms: u64,
}

// ============================================================================
// Models Endpoint
// ============================================================================

/// Response for GET /v1/models
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Object type (always "list")
    pub object: &'static str,
    /// Available models
    pub data: Vec<ModelObject>,
}

/// A single model entry in the models list
#[derive(Debug, Serialize)]
pub struct ModelObject {
    /// Model identifier (e.g., "copilot:gpt-4o")
    pub id: String,
    /// Object type (always "model")
    pub object: &'static str,
    /// Owner/provider name
    pub owned_by: String,
}

// ============================================================================
// Health Endpoint
// ============================================================================

/// Response for GET /health
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Overall status
    pub status: &'static str,
    /// Per-provider readiness
    pub providers: std::collections::HashMap<String, String>,
}

// ============================================================================
// Error Response
// ============================================================================

/// OpenAI-compatible error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

/// Error detail within an `OpenAI` error response
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Parameter that caused the error (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Build an error response with the given type and message
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param: None,
                code: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_single_model() {
        let json = r#"{"model":"copilot:gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.model {
            ModelField::Single(m) => assert_eq!(m, "copilot:gpt-4o"),
            ModelField::Multiple(_) => panic!("expected single"),
        }
        assert!(!req.stream);
    }

    #[test]
    fn deserialize_multiple_models() {
        let json = r#"{"model":["copilot:gpt-4o","claude:opus"],"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.model {
            ModelField::Multiple(models) => {
                assert_eq!(models.len(), 2);
                assert_eq!(models[0], "copilot:gpt-4o");
                assert_eq!(models[1], "claude:opus");
            }
            ModelField::Single(_) => panic!("expected multiple"),
        }
    }

    #[test]
    fn deserialize_with_stream_flag() {
        let json =
            r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"stream":true}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert!(req.stream);
    }

    #[test]
    fn serialize_completion_response() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_owned(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "copilot:gpt-4o".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: "Hello!".to_owned(),
                },
                finish_reason: Some("stop".to_owned()),
            }],
            usage: None,
            warnings: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn serialize_error_response() {
        let resp = ErrorResponse::new("invalid_request_error", "Unknown model");
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("invalid_request_error"));
        assert!(json.contains("Unknown model"));
    }

    #[test]
    fn serialize_chunk_response() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-test".to_owned(),
            object: "chat.completion.chunk",
            created: 1_700_000_000,
            model: "copilot".to_owned(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some("token".to_owned()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("token"));
    }

    #[test]
    fn serialize_models_response() {
        let resp = ModelsResponse {
            object: "list",
            data: vec![ModelObject {
                id: "copilot:gpt-4o".to_owned(),
                object: "model",
                owned_by: "copilot".to_owned(),
            }],
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("copilot:gpt-4o"));
    }
}
