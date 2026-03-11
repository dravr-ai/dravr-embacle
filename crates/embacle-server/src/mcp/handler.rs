// ABOUTME: HTTP handler for MCP Streamable HTTP transport at POST /mcp
// ABOUTME: Accepts JSON-RPC requests and responds via JSON or SSE based on Accept header
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Axum handler for MCP Streamable HTTP at `POST /mcp`.
//!
//! Parses the request body as JSON-RPC 2.0, dispatches to [`McpServer`],
//! and returns the response as either plain JSON or a single SSE event
//! (when the client sends `Accept: text/event-stream`).

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream;
use tracing::{debug, error};

use super::protocol::{JsonRpcRequest, JsonRpcResponse, PARSE_ERROR};
use super::server::McpServer;

/// Handle an incoming MCP POST request
///
/// Parses the body as JSON-RPC, dispatches to the MCP server, and returns
/// the response as JSON or SSE depending on the Accept header.
pub async fn handle_mcp_post(
    State(server): State<Arc<McpServer>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let request: JsonRpcRequest = match serde_json::from_str(&body) {
        Ok(req) => req,
        Err(e) => {
            error!(error = %e, "Failed to parse MCP JSON-RPC body");
            let resp = JsonRpcResponse::error(None, PARSE_ERROR, format!("Parse error: {e}"));
            return Json(resp).into_response();
        }
    };

    debug!(method = %request.method, "Handling MCP request");

    let Some(response) = server.handle_request(request).await else {
        // Notification — no response needed
        return axum::http::StatusCode::NO_CONTENT.into_response();
    };

    let wants_sse = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|accept| accept.contains("text/event-stream"));

    if wants_sse {
        respond_sse(&response)
    } else {
        Json(response).into_response()
    }
}

/// Wrap a JSON-RPC response in a single SSE event
fn respond_sse(response: &JsonRpcResponse) -> Response {
    let data = serde_json::to_string(&response).unwrap_or_else(|e| {
        format!(
            r#"{{"jsonrpc":"2.0","error":{{"code":-32603,"message":"Serialization failed: {e}"}}}}"#
        )
    });

    let event = Event::default().data(data);
    let event_stream = stream::once(async { Ok::<_, Infallible>(event) });

    Sse::new(event_stream).into_response()
}
