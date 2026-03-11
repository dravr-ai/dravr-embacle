// ABOUTME: MCP tool that lists available providers and their models
// ABOUTME: Mirrors the /v1/models endpoint functionality via MCP protocol
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use async_trait::async_trait;
use embacle::ALL_PROVIDERS;
use serde_json::{json, Value};

use crate::mcp::protocol::{CallToolResult, ToolDefinition};
use crate::mcp::tools::McpTool;
use crate::state::SharedState;

/// Lists available LLM providers and the server's default
pub struct ListModels;

#[async_trait]
impl McpTool for ListModels {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_models".to_owned(),
            description: "List available LLM providers and the server's default provider"
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, state: &SharedState, _arguments: Value) -> CallToolResult {
        let default = state.default_provider();
        let providers: Vec<String> = ALL_PROVIDERS.iter().map(ToString::to_string).collect();

        let result = json!({
            "default_provider": default.to_string(),
            "available_providers": providers,
            "hint": "Use 'provider:model' in the prompt tool's model field to route to a specific provider"
        });

        match serde_json::to_string_pretty(&result) {
            Ok(json) => CallToolResult::text(json),
            Err(e) => CallToolResult::error(format!("Serialization failed: {e}")),
        }
    }
}
