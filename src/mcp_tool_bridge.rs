// ABOUTME: Glue layer converting MCP tool definitions to embacle text-based tool simulation
// ABOUTME: Bridges async McpToolExecutor to synchronous TextToolHandler via block_in_place
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # MCP Tool Bridge
//!
//! Converts MCP (Model Context Protocol) tool definitions into embacle's
//! text-based tool simulation types ([`FunctionDeclaration`], [`TextToolHandler`]).
//!
//! This enables using MCP-compatible tool servers with any embacle CLI runner
//! via the text-based tool loop.
//!
//! ## Async bridging
//!
//! [`create_mcp_tool_handler()`](crate::mcp_tool_bridge::create_mcp_tool_handler) bridges the async [`McpToolExecutor`] trait to
//! the synchronous [`TextToolHandler`] callback via `tokio::task::block_in_place`.
//! This requires a **multi-threaded tokio runtime** (`rt-multi-thread` feature).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::task;
use tracing::warn;

use crate::tool_simulation::{FunctionDeclaration, FunctionResponse, TextToolHandler};
use crate::types::RunnerError;

/// An MCP tool definition describing a callable tool
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpToolDefinition {
    /// Unique name of the tool
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for the tool's input parameters
    pub input_schema: Value,
}

/// Trait for executing MCP tool calls asynchronously
#[async_trait]
pub trait McpToolExecutor: Send + Sync {
    /// Execute a tool call and return the result as JSON
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the tool execution fails.
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError>;
}

/// Convert MCP tool definitions to embacle `FunctionDeclaration` values.
///
/// Direct mapping: `input_schema` becomes `parameters`.
pub fn mcp_tools_to_declarations(tools: &[McpToolDefinition]) -> Vec<FunctionDeclaration> {
    tools
        .iter()
        .map(|tool| FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: Some(tool.input_schema.clone()),
        })
        .collect()
}

/// Create a [`TextToolHandler`] that delegates to an async [`McpToolExecutor`].
///
/// Uses `tokio::task::block_in_place` + `handle.block_on()` to bridge async to
/// sync. This requires a multi-threaded tokio runtime (`rt-multi-thread`).
///
/// On executor error, returns a `FunctionResponse` with `{"error": "..."}`.
pub fn create_mcp_tool_handler(executor: Arc<dyn McpToolExecutor>) -> TextToolHandler {
    Arc::new(move |tool_name: &str, arguments: &Value| {
        let executor = Arc::clone(&executor);
        let tool_name_owned = tool_name.to_owned();
        let arguments_owned = arguments.clone();

        let result = task::block_in_place(|| {
            let handle = Handle::current();
            handle.block_on(executor.execute(&tool_name_owned, &arguments_owned))
        });

        match result {
            Ok(value) => FunctionResponse {
                name: tool_name_owned,
                response: value,
            },
            Err(err) => {
                warn!(
                    tool_name = tool_name_owned,
                    error = %err,
                    "MCP tool execution failed"
                );
                FunctionResponse {
                    name: tool_name_owned,
                    response: serde_json::json!({"error": err.message}),
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn conversion_correctness() {
        let tools = vec![
            McpToolDefinition {
                name: "read_file".to_owned(),
                description: "Read a file from disk".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }),
            },
            McpToolDefinition {
                name: "list_dir".to_owned(),
                description: "List directory contents".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"}
                    }
                }),
            },
        ];

        let declarations = mcp_tools_to_declarations(&tools);
        assert_eq!(declarations.len(), 2);

        assert_eq!(declarations[0].name, "read_file");
        assert_eq!(declarations[0].description, "Read a file from disk");
        assert_eq!(
            declarations[0].parameters,
            Some(json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }))
        );

        assert_eq!(declarations[1].name, "list_dir");
    }

    #[test]
    fn empty_list_conversion() {
        let declarations = mcp_tools_to_declarations(&[]);
        assert!(declarations.is_empty());
    }

    struct MockExecutor {
        result: Result<Value, RunnerError>,
    }

    #[async_trait]
    impl McpToolExecutor for MockExecutor {
        async fn execute(
            &self,
            _tool_name: &str,
            _arguments: &Value,
        ) -> Result<Value, RunnerError> {
            self.result.clone()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handler_with_test_executor() {
        let executor = Arc::new(MockExecutor {
            result: Ok(json!({"status": "ok", "data": [1, 2, 3]})),
        });
        let handler = create_mcp_tool_handler(executor);

        let response = handler("test_tool", &json!({"key": "value"}));
        assert_eq!(response.name, "test_tool");
        assert_eq!(response.response["status"], "ok");
        assert_eq!(response.response["data"], json!([1, 2, 3]));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn handler_error_path() {
        let executor = Arc::new(MockExecutor {
            result: Err(RunnerError::external_service("mcp", "connection refused")),
        });
        let handler = create_mcp_tool_handler(executor);

        let response = handler("broken_tool", &json!({}));
        assert_eq!(response.name, "broken_tool");
        assert!(response.response["error"]
            .as_str()
            .expect("error field") // Safe: test assertion
            .contains("connection refused"));
    }
}
