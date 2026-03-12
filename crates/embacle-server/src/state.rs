// ABOUTME: Re-exports shared server state from embacle-mcp for unified state management
// ABOUTME: Single SharedState (Arc<RwLock<ServerState>>) used by both OpenAI and MCP endpoints
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

pub use embacle_mcp::state::{ServerState, SharedState};
