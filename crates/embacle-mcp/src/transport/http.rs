// ABOUTME: HTTP transport implementing MCP Streamable HTTP with JSON and SSE responses
// ABOUTME: Serves a POST endpoint that accepts JSON-RPC and responds via JSON or event stream
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::convert::Infallible;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use embacle::types::RunnerError;
use futures::stream;
use tracing::{debug, error, info};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse, PARSE_ERROR};
use crate::server::McpServer;
use crate::transport::McpTransport;

/// MCP transport over HTTP using axum
///
/// Exposes a single POST endpoint at `/mcp` that accepts JSON-RPC messages.
/// Supports both `application/json` and `text/event-stream` response formats
/// based on the client's `Accept` header (Streamable HTTP transport).
pub struct HttpTransport {
    host: String,
    port: u16,
}

impl HttpTransport {
    /// Create an HTTP transport bound to the given host and port
    pub const fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn serve(self, server: Arc<McpServer>) -> Result<(), RunnerError> {
        let app = Router::new()
            .route("/mcp", post(handle_mcp_post))
            .with_state(server);

        let addr = format!("{}:{}", self.host, self.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| RunnerError::internal(format!("Failed to bind {addr}: {e}")))?;

        info!(address = %addr, "HTTP transport listening");

        axum::serve(listener, app)
            .await
            .map_err(|e| RunnerError::internal(format!("HTTP server error: {e}")))?;

        Ok(())
    }
}

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
            error!(error = %e, "Failed to parse HTTP JSON-RPC body");
            let resp = JsonRpcResponse::error(None, PARSE_ERROR, format!("Parse error: {e}"));
            return Json(resp).into_response();
        }
    };

    debug!(method = %request.method, "Handling HTTP MCP request");

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
