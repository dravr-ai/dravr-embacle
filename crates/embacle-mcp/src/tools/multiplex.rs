// ABOUTME: MCP tools for configuring multiplex providers that receive fan-out prompts
// ABOUTME: Manages the list of providers used when prompt dispatch runs in multiplex mode
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use async_trait::async_trait;
use serde_json::{json, Value};

use dravr_tronc::mcp::schema::{Tool, ToolResponse};
use dravr_tronc::{McpTool, ToolContext};

use crate::runner::{parse_runner_type, valid_provider_names, ALL_PROVIDERS};
use crate::state::{ServerState, SharedState};

/// Returns the list of providers configured for multiplex dispatch
pub struct GetMultiplexProvider;

#[async_trait]
impl McpTool<ServerState> for GetMultiplexProvider {
    fn definition(&self) -> Tool {
        Tool {
            name: "get_multiplex_provider".to_owned(),
            description: "Get the list of providers configured for multiplex prompt dispatch"
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
            annotations: None,
        }
    }

    async fn execute(
        &self,
        state: &SharedState,
        _ctx: &ToolContext,
        _arguments: Value,
    ) -> ToolResponse {
        let providers: Vec<String> = state
            .multiplex_providers()
            .await
            .iter()
            .map(ToString::to_string)
            .collect();
        let all: Vec<String> = ALL_PROVIDERS.iter().map(ToString::to_string).collect();

        ToolResponse::text(
            json!({
                "multiplex_providers": providers,
                "available_providers": all
            })
            .to_string(),
        )
    }
}

/// Sets the list of providers used when multiplexing prompts
pub struct SetMultiplexProvider;

#[async_trait]
impl McpTool<ServerState> for SetMultiplexProvider {
    fn definition(&self) -> Tool {
        let provider_names: Vec<String> = ALL_PROVIDERS.iter().map(ToString::to_string).collect();

        Tool {
            name: "set_multiplex_provider".to_owned(),
            description:
                "Set providers for multiplex mode — prompts will fan out to all listed providers"
                    .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "providers": {
                        "type": "array",
                        "description": "List of provider names to multiplex to",
                        "items": {
                            "type": "string",
                            "enum": provider_names
                        }
                    }
                },
                "required": ["providers"]
            }),
            annotations: None,
        }
    }

    async fn execute(
        &self,
        state: &SharedState,
        _ctx: &ToolContext,
        arguments: Value,
    ) -> ToolResponse {
        let Some(provider_strs) = arguments.get("providers").and_then(Value::as_array) else {
            return ToolResponse::error("Missing 'providers' array argument".to_owned());
        };

        let mut providers = Vec::with_capacity(provider_strs.len());
        for val in provider_strs {
            let Some(name) = val.as_str() else {
                return ToolResponse::error(format!("Provider must be a string, got: {val}"));
            };
            match parse_runner_type(name) {
                Some(p) => providers.push(p),
                None => {
                    return ToolResponse::error(format!(
                        "Unknown provider: {name}. Valid: {}",
                        valid_provider_names()
                    ));
                }
            }
        }

        let result_names: Vec<String> = providers.iter().map(ToString::to_string).collect();
        state.set_multiplex_providers(providers).await;

        ToolResponse::text(
            json!({
                "multiplex_providers": result_names,
                "status": "configured"
            })
            .to_string(),
        )
    }
}
