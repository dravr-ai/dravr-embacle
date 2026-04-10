// ABOUTME: Axum router wiring OpenAI-compatible and MCP endpoints
// ABOUTME: Mounts completions, models, health, and MCP routes with optional auth middleware
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use dravr_tronc::mcp::transport::http::mcp_router as build_mcp_router;
use dravr_tronc::McpServer;

use crate::auth;
use crate::completions;
use crate::health;
use crate::models;
use crate::state::SharedState;

/// Build the application router with all endpoints
///
/// Routes:
/// - `POST /v1/chat/completions` — Chat completion (streaming and non-streaming)
/// - `GET /v1/models` — List available models
/// - `GET /health` — Provider health check
/// - `POST /mcp` — MCP Streamable HTTP (JSON-RPC 2.0, via dravr-tronc)
///
/// The auth middleware is applied to all routes. It only enforces
/// authentication when `EMBACLE_API_KEY` is set.
pub fn build(state: SharedState) -> Router {
    let mcp_server = Arc::new(McpServer::new(
        "embacle-mcp",
        env!("CARGO_PKG_VERSION"),
        embacle_mcp::build_tool_registry(),
        Arc::clone(&state),
    ));

    let mcp_router = build_mcp_router(mcp_server);

    Router::new()
        .route("/v1/chat/completions", post(completions::handle))
        .route("/v1/models", get(models::handle))
        .route("/health", get(health::handle))
        .with_state(state)
        .merge(mcp_router)
        .layer(middleware::from_fn(auth::require_auth))
}
