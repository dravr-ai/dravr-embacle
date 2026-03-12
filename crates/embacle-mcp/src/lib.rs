// ABOUTME: Library root re-exporting MCP server modules for use by embacle-server
// ABOUTME: Exposes protocol, server, state, tools, transport, and runner as public modules
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # embacle-mcp
//!
//! MCP server library exposing embacle LLM runners via Model Context Protocol.
//! Supports stdio and HTTP/SSE transports with provider, model, and multiplex tools.
//!
//! ## Re-exports
//!
//! - [`McpServer`] — JSON-RPC request dispatcher
//! - [`ServerState`] / [`SharedState`] — shared server state with provider and runner cache
//! - [`build_tool_registry`] — default tool registry with all 7 MCP tools
//! - [`McpTransport`] — transport trait for stdio/HTTP backends

pub mod protocol;
pub mod runner;
pub mod server;
pub mod state;
pub mod tools;
pub mod transport;

pub use server::McpServer;
pub use state::{ServerState, SharedState};
pub use tools::build_tool_registry;
pub use transport::McpTransport;
