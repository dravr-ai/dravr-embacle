// ABOUTME: MCP tools for getting and setting the active model for the current provider
// ABOUTME: Returns available models from the runner and accepts model override strings
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use async_trait::async_trait;
use serde_json::{json, Value};

use dravr_tronc::mcp::protocol::{CallToolResult, ToolDefinition};
use dravr_tronc::McpTool;

use crate::state::SharedState;

/// Returns the current model, default model, and available models for the active provider
pub struct GetModel;

#[async_trait]
impl McpTool<crate::state::ServerState> for GetModel {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_model".to_owned(),
            description: "Get the current model and list available models for the active provider"
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn execute(&self, state: &SharedState, _arguments: Value) -> CallToolResult {
        let state_guard = state.read().await;
        let provider = state_guard.active_provider();
        let current_model = state_guard.active_model().map(ToOwned::to_owned);
        let runner_result = state_guard.get_runner(provider).await;
        drop(state_guard);

        let (default_model, available_models) = match runner_result {
            Ok(runner) => (
                runner.default_model().to_owned(),
                runner.available_models().to_vec(),
            ),
            Err(e) => {
                return CallToolResult::text(
                    json!({
                        "provider": provider.to_string(),
                        "current_model": current_model,
                        "error": format!("Could not load runner: {e}")
                    })
                    .to_string(),
                );
            }
        };

        CallToolResult::text(
            json!({
                "provider": provider.to_string(),
                "current_model": current_model,
                "default_model": default_model,
                "available_models": available_models
            })
            .to_string(),
        )
    }
}

/// Sets the model for subsequent prompt dispatch requests
pub struct SetModel;

#[async_trait]
impl McpTool<crate::state::ServerState> for SetModel {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "set_model".to_owned(),
            description: "Set the model for the active provider (pass null to reset to default)"
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model identifier (e.g. claude-opus-4-20250514, gpt-4o). Pass null to reset."
                    }
                },
                "required": ["model"]
            }),
        }
    }

    async fn execute(&self, state: &SharedState, arguments: Value) -> CallToolResult {
        let model = arguments
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        let mut state_guard = state.write().await;
        state_guard.set_active_model(model.clone());
        let provider = state_guard.active_provider();
        drop(state_guard);

        CallToolResult::text(
            json!({
                "provider": provider.to_string(),
                "current_model": model,
                "status": "updated"
            })
            .to_string(),
        )
    }
}
