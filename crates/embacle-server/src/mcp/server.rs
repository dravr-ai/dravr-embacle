// ABOUTME: MCP server core that routes JSON-RPC requests to protocol handlers and tools
// ABOUTME: Implements initialize, tools/list, tools/call, and ping MCP methods
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use serde_json::Value;
use tracing::debug;

use super::protocol::{
    CallToolParams, InitializeParams, InitializeResult, JsonRpcRequest, JsonRpcResponse,
    ServerCapabilities, ServerInfo, ToolsCapability, ToolsListResult, INTERNAL_ERROR,
    INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PROTOCOL_VERSION, SERVER_NAME,
    SERVER_VERSION,
};
use super::tools::ToolRegistry;
use crate::state::SharedState;

/// MCP server that dispatches JSON-RPC requests to the appropriate handler
///
/// Uses the same `SharedState` as the OpenAI-compatible endpoints.
/// Provider routing is stateless — determined per-request via tool arguments.
pub struct McpServer {
    state: SharedState,
    tools: ToolRegistry,
}

impl McpServer {
    /// Create a server with the given shared state and tool registry
    pub const fn new(state: SharedState, tools: ToolRegistry) -> Self {
        Self { state, tools }
    }

    /// Route a JSON-RPC request to the appropriate MCP handler
    ///
    /// Returns `None` for notifications (requests without an id).
    pub async fn handle_request(&self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        if request.jsonrpc != "2.0" {
            return Some(JsonRpcResponse::error(
                request.id,
                INVALID_REQUEST,
                format!("Unsupported JSON-RPC version: {}", request.jsonrpc),
            ));
        }

        // Notifications have no id and expect no response
        if request.id.is_none() {
            debug!(method = %request.method, "Received notification, no response");
            return None;
        }

        let response = match request.method.as_str() {
            "initialize" => Self::handle_initialize(request.id, request.params),
            "tools/list" => self.handle_tools_list(request.id),
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            "ping" => JsonRpcResponse::success(request.id, Value::Object(serde_json::Map::new())),
            method => {
                debug!(method, "Unknown MCP method");
                JsonRpcResponse::error(
                    request.id,
                    METHOD_NOT_FOUND,
                    format!("Method not found: {method}"),
                )
            }
        };

        Some(response)
    }

    /// Handle `initialize` — parse client info and return server capabilities
    fn handle_initialize(id: Option<Value>, params: Option<Value>) -> JsonRpcResponse {
        if let Some(params) = params {
            if let Ok(init) = serde_json::from_value::<InitializeParams>(params) {
                debug!(
                    client = %init.client_info.name,
                    version = ?init.client_info.version,
                    protocol = %init.protocol_version,
                    "MCP client connected"
                );
            }
        }

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_owned(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {}),
            },
            server_info: ServerInfo {
                name: SERVER_NAME.to_owned(),
                version: SERVER_VERSION.to_owned(),
            },
        };

        match serde_json::to_value(result) {
            Ok(val) => JsonRpcResponse::success(id, val),
            Err(e) => {
                JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Serialization error: {e}"))
            }
        }
    }

    /// Handle `tools/list` — return all registered tool definitions
    fn handle_tools_list(&self, id: Option<Value>) -> JsonRpcResponse {
        let result = ToolsListResult {
            tools: self.tools.list_definitions(),
        };

        match serde_json::to_value(result) {
            Ok(val) => JsonRpcResponse::success(id, val),
            Err(e) => {
                JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Serialization error: {e}"))
            }
        }
    }

    /// Handle `tools/call` — dispatch to the named tool handler
    async fn handle_tools_call(&self, id: Option<Value>, params: Option<Value>) -> JsonRpcResponse {
        let call_params: CallToolParams = match params {
            Some(p) => match serde_json::from_value(p) {
                Ok(cp) => cp,
                Err(e) => {
                    return JsonRpcResponse::error(
                        id,
                        INVALID_PARAMS,
                        format!("Invalid params: {e}"),
                    );
                }
            },
            None => {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Missing params for tools/call".to_owned(),
                );
            }
        };

        let arguments = call_params
            .arguments
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

        let result = self
            .tools
            .execute(&call_params.name, &self.state, arguments)
            .await;

        match serde_json::to_value(result) {
            Ok(val) => JsonRpcResponse::success(id, val),
            Err(e) => JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Result serialization error: {e}"),
            ),
        }
    }
}
