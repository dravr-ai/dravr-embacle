// ABOUTME: Shared server state plus the axum AppState carrying the optional MCP tool executor
// ABOUTME: SharedState (Arc<ServerState>) is reused from embacle-mcp; AppState adds tools
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use axum::extract::FromRef;
use embacle::{FunctionDeclaration, McpToolExecutor};

pub use embacle_mcp::state::{ServerState, SharedState};

/// Server-side tools available for autonomous execution.
///
/// Bundles the [`McpToolExecutor`] that runs tool calls with the tool
/// declarations used to build the catalog injected into the conversation.
#[derive(Clone)]
pub struct ServerTools {
    /// Executor that dispatches tool calls (e.g. an MCP client pool)
    pub executor: Arc<dyn McpToolExecutor>,
    /// Declarations of the tools the executor can run
    pub declarations: Vec<FunctionDeclaration>,
}

/// Application state shared across all axum handlers.
///
/// Wraps the provider [`SharedState`] together with optional server-side tools.
/// They are present only when the server was started with configured
/// `[[mcp_servers]]` and the `mcp-tools` feature; handlers that do not need them
/// continue to extract [`SharedState`] directly via [`FromRef`].
#[derive(Clone)]
pub struct AppState {
    /// Provider/runner state shared with the MCP endpoints
    pub shared: SharedState,
    /// Server-side tools backing autonomous tool execution, if configured
    pub server_tools: Option<ServerTools>,
}

impl AppState {
    /// Build application state with no server-side tools.
    pub fn new(shared: SharedState) -> Self {
        Self {
            shared,
            server_tools: None,
        }
    }

    /// Attach server-side tools (e.g. from an MCP client pool).
    pub fn with_server_tools(mut self, tools: Option<ServerTools>) -> Self {
        self.server_tools = tools;
        self
    }
}

impl FromRef<AppState> for SharedState {
    fn from_ref(app: &AppState) -> Self {
        app.shared.clone()
    }
}
