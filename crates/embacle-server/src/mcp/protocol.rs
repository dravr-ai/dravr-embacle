// ABOUTME: MCP JSON-RPC protocol types for request/response handling
// ABOUTME: Defines wire format for initialize, tools/list, tools/call, and error responses
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! JSON-RPC 2.0 wire types for the MCP protocol.
//!
//! Defines request/response structures, error codes, and MCP-specific types
//! (`InitializeResult`, `ToolDefinition`, `CallToolResult`) used by the
//! server's MCP Streamable HTTP transport.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol version supported by this server
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Server name reported during MCP handshake
pub const SERVER_NAME: &str = "embacle-server";

/// Server version reported during MCP handshake
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// ============================================================================
// JSON-RPC Error Codes
// ============================================================================

/// JSON-RPC parse error: invalid JSON received
pub const PARSE_ERROR: i32 = -32_700;

/// JSON-RPC invalid request (e.g. wrong protocol version)
pub const INVALID_REQUEST: i32 = -32_600;

/// JSON-RPC method not found
pub const METHOD_NOT_FOUND: i32 = -32_601;

/// JSON-RPC invalid parameters
pub const INVALID_PARAMS: i32 = -32_602;

/// JSON-RPC internal error
pub const INTERNAL_ERROR: i32 = -32_603;

// ============================================================================
// JSON-RPC Messages
// ============================================================================

/// Incoming JSON-RPC request from MCP client
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version marker (always "2.0", validated by JSON-RPC clients)
    pub jsonrpc: String,
    /// Request identifier (None for notifications)
    pub id: Option<Value>,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(default)]
    pub params: Option<Value>,
}

/// Outgoing JSON-RPC response to MCP client
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    /// Always "2.0"
    pub jsonrpc: String,
    /// Matching request identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    /// Success payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error object
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code
    pub code: i32,
    /// Human-readable error message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    /// Build a success response with the given result
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_owned(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Build an error response with the given code and message
    pub fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_owned(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ============================================================================
// MCP Initialize
// ============================================================================

/// Parameters for the `initialize` request
#[derive(Debug, Deserialize)]
pub struct InitializeParams {
    /// Protocol version requested by the client
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client capabilities
    #[serde(default)]
    pub capabilities: Value,
    /// Client identification
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Client identification sent during initialization
#[derive(Debug, Deserialize)]
pub struct ClientInfo {
    /// Client name
    pub name: String,
    /// Client version
    #[serde(default)]
    pub version: Option<String>,
}

/// Result of a successful `initialize` response
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    /// Protocol version the server supports
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Server identification
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// Server identification
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
}

/// Server capability declarations
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    /// Tool support (presence signals tools are available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
}

/// Marker type indicating the server supports MCP tools
#[derive(Debug, Serialize)]
pub struct ToolsCapability {}

// ============================================================================
// MCP Tools
// ============================================================================

/// Tool definition exposed via `tools/list`
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,
    /// Human-readable tool description
    pub description: String,
    /// JSON Schema describing the tool's input
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Result of a `tools/list` call
#[derive(Debug, Serialize)]
pub struct ToolsListResult {
    /// Available tool definitions
    pub tools: Vec<ToolDefinition>,
}

/// Parameters for a `tools/call` request
#[derive(Debug, Deserialize)]
pub struct CallToolParams {
    /// Name of the tool to invoke
    pub name: String,
    /// Tool arguments
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// Result of a `tools/call` invocation
#[derive(Debug, Serialize)]
pub struct CallToolResult {
    /// Response content parts
    pub content: Vec<ContentPart>,
    /// Whether this result represents an error
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// A content part within a tool result
#[derive(Debug, Serialize)]
pub struct ContentPart {
    /// Content type (always "text" for now)
    #[serde(rename = "type")]
    pub content_type: String,
    /// Text content
    pub text: String,
}

impl CallToolResult {
    /// Build a successful text result
    pub fn text(content: String) -> Self {
        Self {
            content: vec![ContentPart {
                content_type: "text".to_owned(),
                text: content,
            }],
            is_error: None,
        }
    }

    /// Build an error result with the given message
    pub fn error(message: String) -> Self {
        Self {
            content: vec![ContentPart {
                content_type: "text".to_owned(),
                text: message,
            }],
            is_error: Some(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_success_response() {
        let resp = JsonRpcResponse::success(Some(Value::from(1)), serde_json::json!({"ok": true}));
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn serialize_error_response() {
        let resp = JsonRpcResponse::error(Some(Value::from(1)), PARSE_ERROR, "bad json".to_owned());
        let json = serde_json::to_string(&resp).expect("serialize");
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32700"));
        assert!(!json.contains("\"result\""));
    }

    #[test]
    fn deserialize_request() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).expect("deserialize");
        assert_eq!(req.method, "tools/list");
        assert!(req.params.is_none());
    }

    #[test]
    fn call_tool_result_text() {
        let result = CallToolResult::text("hello".to_owned());
        assert!(result.is_error.is_none());
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text, "hello");
    }

    #[test]
    fn call_tool_result_error() {
        let result = CallToolResult::error("oops".to_owned());
        assert_eq!(result.is_error, Some(true));
        assert_eq!(result.content[0].text, "oops");
    }
}
