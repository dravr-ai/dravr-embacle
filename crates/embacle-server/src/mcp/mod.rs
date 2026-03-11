// ABOUTME: MCP protocol support for the unified embacle server
// ABOUTME: Exposes JSON-RPC 2.0 tools via POST /mcp alongside OpenAI-compatible routes
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! MCP Streamable HTTP transport for the embacle server.
//!
//! Implements the [Model Context Protocol](https://modelcontextprotocol.io/) over HTTP
//! at `POST /mcp`. Clients send JSON-RPC 2.0 requests and receive responses as JSON
//! or Server-Sent Events (based on the `Accept` header).
//!
//! ## Supported Methods
//!
//! - `initialize` — handshake with protocol version and capability negotiation
//! - `tools/list` — enumerate available MCP tools (`prompt`, `list_models`)
//! - `tools/call` — invoke a tool by name with JSON arguments
//! - `ping` — liveness check
//!
//! ## Architecture
//!
//! Provider routing is stateless — the `prompt` tool resolves providers per-request
//! using the same `provider:model` routing as the OpenAI-compatible endpoints.
//! This contrasts with `embacle-mcp` where the active provider is mutable server state.

pub mod handler;
pub mod protocol;
pub mod server;
pub mod tools;
