// ABOUTME: Library root re-exporting MCP server modules for use by embacle-server
// ABOUTME: Delegates protocol, server, and transport to dravr-tronc; provides tools, state, and runner
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
//! - [`McpServer`] — JSON-RPC request dispatcher (from dravr-tronc)
//! - [`ServerState`] / [`SharedState`] — shared server state with provider and runner cache
//! - [`build_tool_registry`] — default tool registry with all 7 MCP tools

pub mod runner;
pub mod state;
pub mod tools;

pub use dravr_tronc::McpServer;
pub use state::{ServerState, SharedState};
pub use tools::build_tool_registry;
