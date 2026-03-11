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
    /// Tool definitions for function calling
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls which tools the model may call
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Controls the response format (text, `json_object`, or `json_schema`)
    #[serde(default)]
    pub response_format: Option<ResponseFormatRequest>,
    /// Nucleus sampling parameter (0.0 - 1.0)
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Stop sequence(s) that halt generation — single string or array
    #[serde(default)]
    pub stop: Option<StopField>,
}

/// Stop field that accepts either a single string or an array of strings
///
/// Per `OpenAI` spec, the `stop` parameter can be a string or an array of
/// up to 4 strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    /// Single stop sequence
    Single(String),
    /// Multiple stop sequences
    Multiple(Vec<String>),
}

/// OpenAI-specified maximum number of stop sequences
const MAX_STOP_SEQUENCES: usize = 4;

impl StopField {
    /// Number of stop sequences in this field
    pub const fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Multiple(v) => v.len(),
        }
    }

    /// Returns true if no stop sequences are present
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Multiple(v) if v.is_empty())
    }

    /// Convert to a Vec of strings regardless of the variant,
    /// truncating to the `OpenAI`-specified maximum of 4 sequences
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v.into_iter().take(MAX_STOP_SEQUENCES).collect(),
        }
    }

    /// Clone only the bounded subset (up to 4 sequences) without copying
    /// the entire input — safe for use with user-controlled data
    pub fn to_bounded_vec(&self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s.clone()],
            Self::Multiple(v) => v.iter().take(MAX_STOP_SEQUENCES).cloned().collect(),
        }
    }
}

/// OpenAI-compatible response format request
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormatRequest {
    /// Default text response
    #[serde(rename = "text")]
    Text,
    /// Force JSON object output
    #[serde(rename = "json_object")]
    JsonObject,
    /// Force JSON output conforming to a specific schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The JSON schema specification
        json_schema: JsonSchemaSpec,
    },
}

/// JSON schema specification within a response format request
#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    /// Schema name for identification
    pub name: String,
    /// The JSON Schema definition
    pub schema: serde_json::Value,
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
    /// Role: "system", "user", "assistant", or "tool"
    pub role: String,
    /// Message content (None for tool-call-only assistant messages)
    pub content: Option<String>,
    /// Tool calls requested by the assistant
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message responds to (role="tool")
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Function name for tool result messages
    #[serde(default)]
    pub name: Option<String>,
}

// ============================================================================
// Tool Calling Types
// ============================================================================

/// A tool definition in the `OpenAI` format
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDefinition {
    /// Tool type (always "function" currently)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionObject,
}

/// A function definition within a tool
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionObject {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the function parameters
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// Controls which tools the model may call
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String variant: "none", "auto", or "required"
    Mode(String),
    /// Specific function variant: {"type": "function", "function": {"name": "..."}}
    Specific(ToolChoiceSpecific),
}

/// A specific tool choice forcing a particular function
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceSpecific {
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function to force
    pub function: ToolChoiceFunction,
}

/// Function name within a specific tool choice
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    /// Name of the function to call
    pub name: String,
}

/// A tool call issued by the assistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Position index of this tool call in the array (required by `OpenAI` spec)
    #[serde(default)]
    pub index: usize,
    /// Unique identifier for this tool call
    pub id: String,
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call details
    pub function: ToolCallFunction,
}

/// Function call details within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Name of the function to call
    pub name: String,
    /// JSON-encoded arguments
    pub arguments: String,
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
    /// Generated content (None when `tool_calls` are present)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls requested by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
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
    /// Tool calls (reserved for future streaming tool call support)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
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
    fn deserialize_message_with_null_content() {
        let json = r#"{"model":"copilot","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert!(req.messages[0].content.is_none());
        assert!(req.messages[0].tool_calls.is_some());
    }

    #[test]
    fn deserialize_message_without_content_field() {
        let json = r#"{"model":"copilot","messages":[{"role":"tool","tool_call_id":"call_1","name":"search","content":"{\"result\":\"found\"}"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.messages[0].role, "tool");
        assert_eq!(req.messages[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(req.messages[0].name.as_deref(), Some("search"));
    }

    #[test]
    fn deserialize_tool_definitions() {
        let json = r#"{
            "model": "copilot",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                }
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        let tools = req.tools.expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(tools[0].function.parameters.is_some());
    }

    #[test]
    fn deserialize_tool_choice_auto() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"tool_choice":"auto"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.tool_choice.expect("tool_choice present") {
            ToolChoice::Mode(m) => assert_eq!(m, "auto"),
            ToolChoice::Specific(_) => panic!("expected mode"),
        }
    }

    #[test]
    fn deserialize_tool_choice_specific() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"function","function":{"name":"get_weather"}}}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.tool_choice.expect("tool_choice present") {
            ToolChoice::Specific(s) => assert_eq!(s.function.name, "get_weather"),
            ToolChoice::Mode(_) => panic!("expected specific"),
        }
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
                    content: Some("Hello!".to_owned()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_owned()),
            }],
            usage: None,
            warnings: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
        assert!(!json.contains("tool_calls"));
    }

    #[test]
    fn serialize_response_with_tool_calls() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_owned(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "copilot:gpt-4o".to_owned(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant",
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        index: 0,
                        id: "call_abc123".to_owned(),
                        tool_type: "function".to_owned(),
                        function: ToolCallFunction {
                            name: "get_weather".to_owned(),
                            arguments: r#"{"city":"Paris"}"#.to_owned(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_owned()),
            }],
            usage: None,
            warnings: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("tool_calls"));
        assert!(json.contains("call_abc123"));
        assert!(json.contains("get_weather"));
        assert!(!json.contains(r#""content""#));
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
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).expect("serialize");
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("token"));
        assert!(!json.contains("tool_calls"));
    }

    #[test]
    fn deserialize_tool_choice_none() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"tool_choice":"none"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.tool_choice.expect("tool_choice present") {
            ToolChoice::Mode(m) => assert_eq!(m, "none"),
            ToolChoice::Specific(_) => panic!("expected mode"),
        }
    }

    #[test]
    fn deserialize_tool_choice_required() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.tool_choice.expect("tool_choice present") {
            ToolChoice::Mode(m) => assert_eq!(m, "required"),
            ToolChoice::Specific(_) => panic!("expected mode"),
        }
    }

    #[test]
    fn deserialize_response_format_text() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert!(matches!(
            req.response_format,
            Some(ResponseFormatRequest::Text)
        ));
    }

    #[test]
    fn deserialize_response_format_json_object() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert!(matches!(
            req.response_format,
            Some(ResponseFormatRequest::JsonObject)
        ));
    }

    #[test]
    fn deserialize_response_format_json_schema() {
        let json = r#"{
            "model": "copilot",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": {"type": "object", "properties": {"temp": {"type": "number"}}}
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        match req.response_format {
            Some(ResponseFormatRequest::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "weather");
                assert!(json_schema.schema["properties"]["temp"].is_object());
            }
            other => panic!("expected JsonSchema, got: {other:?}"),
        }
    }

    #[test]
    fn deserialize_top_p() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"top_p":0.9}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.top_p, Some(0.9));
    }

    #[test]
    fn deserialize_stop_single() {
        let json =
            r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"stop":"END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        let stop = req.stop.expect("stop present");
        assert_eq!(stop.into_vec(), vec!["END"]);
    }

    #[test]
    fn deserialize_stop_array() {
        let json = r#"{"model":"copilot","messages":[{"role":"user","content":"hi"}],"stop":["END","STOP"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        let stop = req.stop.expect("stop present");
        assert_eq!(stop.into_vec(), vec!["END", "STOP"]);
    }

    #[test]
    fn stop_field_len() {
        let single = StopField::Single("END".to_owned());
        assert_eq!(single.len(), 1);
        let multiple = StopField::Multiple(vec!["A".to_owned(), "B".to_owned(), "C".to_owned()]);
        assert_eq!(multiple.len(), 3);
    }

    #[test]
    fn stop_field_into_vec_truncates_at_four() {
        let oversized = StopField::Multiple((0..10).map(|i| format!("stop_{i}")).collect());
        let result = oversized.into_vec();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "stop_0");
        assert_eq!(result[3], "stop_3");
    }

    #[test]
    fn deserialize_all_optional_fields() {
        let json = r#"{
            "model": "copilot",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.95,
            "stop": ["END"],
            "stream": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.top_p, Some(0.95));
        assert!(req.stop.is_some());
        assert!(req.stream);
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
