// ABOUTME: Tool registry that maps MCP tool names to handler implementations
// ABOUTME: Provides the McpTool trait and ToolRegistry for tool discovery and dispatch
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

pub mod model;
pub mod multiplex;
pub mod prompt;
pub mod provider;

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::protocol::{CallToolResult, ToolDefinition};
use crate::state::SharedState;

/// Trait implemented by each MCP tool exposed by this server
#[async_trait]
pub trait McpTool: Send + Sync {
    /// Return the tool's MCP definition (name, description, input schema)
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given arguments against the shared server state
    async fn execute(&self, state: &SharedState, arguments: Value) -> CallToolResult;
}

/// Registry mapping tool names to their handler implementations
///
/// Tools are registered at server startup and looked up by name
/// when `tools/call` requests arrive from the MCP client.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn McpTool>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool handler, keyed by its definition name
    pub fn register(&mut self, tool: Box<dyn McpTool>) {
        let name = tool.definition().name;
        self.tools.insert(name, tool);
    }

    /// List all registered tool definitions for `tools/list` responses
    pub fn list_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Dispatch a `tools/call` to the named tool handler
    pub async fn execute(
        &self,
        name: &str,
        state: &SharedState,
        arguments: Value,
    ) -> CallToolResult {
        match self.tools.get(name) {
            Some(tool) => tool.execute(state, arguments).await,
            None => CallToolResult::error(format!("Unknown tool: {name}")),
        }
    }
}

/// Build the default tool registry with all embacle MCP tools
pub fn build_tool_registry() -> ToolRegistry {
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
