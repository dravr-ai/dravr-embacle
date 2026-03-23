// ABOUTME: Tool registry builder mapping MCP tool names to handler implementations
// ABOUTME: Delegates McpTool trait and ToolRegistry to dravr-tronc, registers embacle-specific tools
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

pub mod model;
pub mod multiplex;
pub mod prompt;
pub mod provider;

use dravr_tronc::mcp::tool::ToolRegistry;

use crate::state::ServerState;

/// Build the default tool registry with all embacle MCP tools
pub fn build_tool_registry() -> ToolRegistry<ServerState> {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(provider::GetProvider));
    registry.register(Box::new(provider::SetProvider));
    registry.register(Box::new(model::GetModel));
    registry.register(Box::new(model::SetModel));
    registry.register(Box::new(multiplex::GetMultiplexProvider));
    registry.register(Box::new(multiplex::SetMultiplexProvider));
    registry.register(Box::new(prompt::Prompt));
    registry
}
