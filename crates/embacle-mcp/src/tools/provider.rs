// ABOUTME: MCP tools for getting and setting the active LLM provider
// ABOUTME: Maps provider names to embacle CliRunnerType for runtime provider switching
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use async_trait::async_trait;
use serde_json::{json, Value};

use dravr_tronc::mcp::protocol::{CallToolResult, ToolDefinition};
use dravr_tronc::McpTool;

use crate::runner::{parse_runner_type, valid_provider_names, ALL_PROVIDERS};
use crate::state::{ServerState, SharedState};

/// Returns the currently active LLM provider and available providers
pub struct GetProvider;

#[async_trait]
impl McpTool<ServerState> for GetProvider {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_provider".to_owned(),
            description: "Get the active LLM provider and list all available providers".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn execute(&self, state: &SharedState, _arguments: Value) -> CallToolResult {
        let active = state.read().await.active_provider();
        let all: Vec<String> = ALL_PROVIDERS.iter().map(ToString::to_string).collect();

        CallToolResult::text(
            json!({
                "active_provider": active.to_string(),
                "available_providers": all
            })
            .to_string(),
        )
    }
}

/// Switches the active LLM provider (resets the model selection)
pub struct SetProvider;

#[async_trait]
impl McpTool<ServerState> for SetProvider {
    fn definition(&self) -> ToolDefinition {
        let provider_names: Vec<String> = ALL_PROVIDERS.iter().map(ToString::to_string).collect();

        ToolDefinition {
            name: "set_provider".to_owned(),
            description: "Set the active LLM provider for prompt dispatch".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                        "enum": provider_names
                    }
                },
                "required": ["provider"]
            }),
        }
    }

    async fn execute(&self, state: &SharedState, arguments: Value) -> CallToolResult {
        let Some(provider_str) = arguments.get("provider").and_then(Value::as_str) else {
            return CallToolResult::error("Missing 'provider' argument".to_owned());
        };

        let Some(provider) = parse_runner_type(provider_str) else {
            return CallToolResult::error(format!(
                "Unknown provider: {provider_str}. Valid: {}",
                valid_provider_names()
            ));
        };

        state.write().await.set_active_provider(provider);

        CallToolResult::text(
            json!({
                "active_provider": provider.to_string(),
                "status": "active"
            })
            .to_string(),
        )
    }
}
