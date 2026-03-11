// ABOUTME: Tool registry mapping MCP tool names to stateless handler implementations
// ABOUTME: Provides McpTool trait and ToolRegistry for discovery and dispatch via provider routing
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

pub mod list_models;
pub mod prompt;

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use super::protocol::{CallToolResult, ToolDefinition};
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
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn McpTool>>,
}

impl ToolRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
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

/// Build the tool registry with stateless tools for the unified server
pub fn build_tool_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(prompt::Prompt));
    registry.register(Box::new(list_models::ListModels));
    registry
}
