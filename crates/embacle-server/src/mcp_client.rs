// ABOUTME: MCP stdio client pool that connects to downstream MCP tool servers
// ABOUTME: Discovers their tools and routes tool calls via rmcp for server-side agent execution
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # MCP Client Pool
//!
//! Connects to one or more downstream MCP tool servers (configured via
//! `[[mcp_servers]]`) as an MCP **client** over stdio, using the official
//! [`rmcp`] SDK. On connection it discovers each server's tools and builds a
//! routing table from tool name to owning server.
//!
//! The pool implements [`embacle::McpToolExecutor`], so it can be handed to the
//! text-based tool loop ([`embacle::mcp_tool_bridge`]) / [`embacle::agent::AgentExecutor`]
//! to power server-side tool execution on `/v1/chat/completions`.
//!
//! Tool name collisions across servers are resolved first-wins, with a warning
//! logged for every dropped tool so coverage is never silently reduced.

use std::collections::HashMap;

use async_trait::async_trait;
use embacle::types::RunnerError;
use embacle::{FunctionDeclaration, McpServerConfig, McpToolExecutor};
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::ServiceExt;
use serde_json::Value;
use tokio::process::Command;
use tracing::{info, warn};

/// A live connection to a single downstream MCP server.
struct ConnectedServer {
    /// Logical name from configuration (for diagnostics)
    name: String,
    /// Running rmcp client session bound to the spawned subprocess
    service: RunningService<RoleClient, ()>,
}

/// A pool of MCP stdio clients exposing their union of tools as a single executor.
pub struct McpClientPool {
    /// Connected downstream servers, indexed by position
    servers: Vec<ConnectedServer>,
    /// Tool name -> index into `servers`
    routing: HashMap<String, usize>,
    /// Tool declarations discovered across all servers (for catalog injection)
    declarations: Vec<FunctionDeclaration>,
}

impl McpClientPool {
    /// Connect to every configured MCP server, discover their tools, and build
    /// the routing table.
    ///
    /// Connection failures for an individual server are fatal: a misconfigured
    /// tool server should surface loudly rather than silently degrade the agent.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError::external_service`] if spawning, initializing, or
    /// listing tools for any configured server fails.
    pub async fn connect(configs: &[McpServerConfig]) -> Result<Self, RunnerError> {
        let mut servers = Vec::with_capacity(configs.len());
        let mut routing = HashMap::new();
        let mut declarations = Vec::new();

        for cfg in configs {
            let mut command = Command::new(&cfg.command);
            command.args(&cfg.args);
            for (key, value) in &cfg.env {
                command.env(key, value);
            }

            let transport = TokioChildProcess::new(command).map_err(|e| {
                RunnerError::external_service(
                    "mcp",
                    format!("failed to spawn MCP server '{}': {e}", cfg.name),
                )
            })?;

            let service = ().serve(transport).await.map_err(|e| {
                RunnerError::external_service(
                    "mcp",
                    format!("failed to initialize MCP server '{}': {e}", cfg.name),
                )
            })?;

            let tools = service.list_all_tools().await.map_err(|e| {
                RunnerError::external_service(
                    "mcp",
                    format!("failed to list tools for MCP server '{}': {e}", cfg.name),
                )
            })?;

            let server_idx = servers.len();
            let mut registered = 0_usize;
            for tool in tools {
                let name = tool.name.to_string();
                if routing.contains_key(&name) {
                    warn!(
                        tool = %name,
                        server = %cfg.name,
                        "Duplicate MCP tool name across servers; keeping first registration, dropping this one"
                    );
                    continue;
                }
                declarations.push(FunctionDeclaration {
                    name: name.clone(),
                    description: tool.description.map(|d| d.to_string()).unwrap_or_default(),
                    parameters: Some(Value::Object((*tool.input_schema).clone())),
                });
                routing.insert(name, server_idx);
                registered += 1;
            }

            info!(
                server = %cfg.name,
                command = %cfg.command,
                tools = registered,
                "Connected to MCP tool server"
            );

            servers.push(ConnectedServer {
                name: cfg.name.clone(),
                service,
            });
        }

        info!(
            servers = servers.len(),
            tools = declarations.len(),
            "MCP client pool ready"
        );

        Ok(Self {
            servers,
            routing,
            declarations,
        })
    }

    /// Tool declarations discovered across all connected servers.
    pub fn declarations(&self) -> &[FunctionDeclaration] {
        &self.declarations
    }

    /// Returns true if no servers are connected (no tools available).
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Number of tools available across all connected servers.
    pub fn tool_count(&self) -> usize {
        self.declarations.len()
    }
}

#[async_trait]
impl McpToolExecutor for McpClientPool {
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError> {
        let server_idx = *self.routing.get(tool_name).ok_or_else(|| {
            RunnerError::internal(format!(
                "no connected MCP server provides tool '{tool_name}'"
            ))
        })?;
        let server = &self.servers[server_idx];

        let mut params = CallToolRequestParams::new(tool_name.to_owned());
        if let Some(object) = arguments.as_object() {
            params = params.with_arguments(object.clone());
        }

        let result = server.service.call_tool(params).await.map_err(|e| {
            RunnerError::external_service(
                "mcp",
                format!("tool '{tool_name}' on server '{}' failed: {e}", server.name),
            )
        })?;

        Ok(call_result_to_json(result))
    }
}

/// Convert an rmcp `CallToolResult` into a single JSON value for the tool loop.
///
/// Prefers `structured_content` when present. Otherwise concatenates text
/// content blocks, attempting to parse the joined text as JSON before falling
/// back to a string. Tool-reported errors are wrapped as `{"error": ...}` so the
/// model can observe and recover rather than aborting the whole request.
fn call_result_to_json(result: CallToolResult) -> Value {
    if let Some(structured) = result.structured_content {
        return structured;
    }

    let text = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n");

    if result.is_error == Some(true) {
        return serde_json::json!({ "error": text });
    }

    serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text))
}

#[cfg(test)]
mod tests {
    use rmcp::model::Content;

    use super::*;

    #[test]
    fn call_result_prefers_structured_content() {
        let result = CallToolResult::structured(serde_json::json!({"temp": 72}));
        let value = call_result_to_json(result);
        assert_eq!(value["temp"], 72);
    }

    #[test]
    fn call_result_parses_json_text() {
        let result = CallToolResult::success(vec![Content::text(r#"{"ok":true}"#)]);
        let value = call_result_to_json(result);
        assert_eq!(value["ok"], true);
    }

    #[test]
    fn call_result_falls_back_to_string() {
        let result = CallToolResult::success(vec![Content::text("plain text answer")]);
        let value = call_result_to_json(result);
        assert_eq!(value, Value::String("plain text answer".to_owned()));
    }

    #[test]
    fn call_result_wraps_errors() {
        let result = CallToolResult::error(vec![Content::text("boom")]);
        let value = call_result_to_json(result);
        assert_eq!(value["error"], "boom");
    }
}
