// ABOUTME: Integration tests for the embacle-server REST API and MCP endpoints
// ABOUTME: Exercises router, auth middleware, health, models, completions, and MCP via axum test client
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use embacle::config::CliRunnerType;
use embacle_mcp::ServerState;
use http_body_util::BodyExt;
use tokio::sync::{Mutex, RwLock};
use tower::ServiceExt;

use embacle_server::router;

/// Guards access to `EMBACLE_API_KEY` env var across parallel tests.
/// All tests that read or mutate this env var must hold this lock.
static ENV_MUTEX: Mutex<()> = Mutex::const_new(());

/// Build a POST /v1/chat/completions request from a JSON body
fn post_completions(body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("serialize")))
        .expect("build request")
}

/// Build a test app with the given default provider
fn test_app() -> axum::Router {
    let state = Arc::new(RwLock::new(ServerState::new(CliRunnerType::Copilot)));
    router::build(state)
}

/// Send a request and parse the response body as JSON
async fn send_and_parse(
    app: axum::Router,
    request: Request<Body>,
) -> (StatusCode, serde_json::Value) {
    let response = app.oneshot(request).await.expect("send request");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

// ============================================================================
// Health Endpoint
// ============================================================================

#[tokio::test]
async fn health_returns_json_with_status_field() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    let body = response.into_body();
    let bytes = body.collect().await.expect("collect body").to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    assert!(json.get("status").is_some(), "missing status field");
    assert!(json.get("providers").is_some(), "missing providers field");

    let status = json["status"].as_str().expect("status is string");
    assert!(
        status == "ok" || status == "degraded",
        "unexpected status: {status}"
    );
}

#[tokio::test]
async fn health_providers_contains_all_eleven() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    let providers = json["providers"].as_object().expect("providers is object");
    assert_eq!(providers.len(), 11, "expected 11 providers");

    // Each provider should have a status string
    for (name, value) in providers {
        assert!(value.is_string(), "provider {name} status is not a string");
    }
}

// ============================================================================
// Models Endpoint
// ============================================================================

#[tokio::test]
async fn models_returns_list_object() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array(), "data should be an array");
}

// ============================================================================
// Auth Middleware
//
// All auth tests run in a single function to avoid env var races.
// `EMBACLE_API_KEY` is global process state; parallel tests would
// see each other's set_var/remove_var calls.
// ============================================================================

#[tokio::test]
async fn auth_middleware_scenarios() {
    let _guard = ENV_MUTEX.lock().await;

    let health = |hdr: Option<&'static str>| {
        let mut b = Request::builder().uri("/health");
        if let Some(h) = hdr {
            b = b.header("Authorization", h);
        }
        b.body(Body::empty()).expect("build request")
    };

    // Scenario 1: No key set → pass through
    std::env::remove_var("EMBACLE_API_KEY");
    let (status, _) = send_and_parse(test_app(), health(None)).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED, "should allow without key");

    // Scenario 2: Key set, no header → 401 "Missing"
    std::env::set_var("EMBACLE_API_KEY", "test-auth-key");
    let (status, json) = send_and_parse(test_app(), health(None)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(json["error"]["type"], "authentication_error");
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("Missing"));

    // Scenario 3: Wrong bearer → 401 "Invalid"
    let (status, json) = send_and_parse(test_app(), health(Some("Bearer wrong-key"))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("Invalid"));

    // Scenario 4: Correct bearer → passes
    let (status, _) = send_and_parse(test_app(), health(Some("Bearer test-auth-key"))).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED, "correct key should pass");

    // Scenario 5: Non-Bearer scheme → 401 "Bearer"
    let (status, json) = send_and_parse(test_app(), health(Some("Basic c2VjcmV0"))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("Bearer"));

    std::env::remove_var("EMBACLE_API_KEY");
}

// ============================================================================
// Completions Endpoint
// ============================================================================

#[tokio::test]
async fn completions_rejects_empty_model_array() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let body = serde_json::json!({
        "model": [],
        "messages": [{"role": "user", "content": "hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    assert!(json["error"]["message"]
        .as_str()
        .expect("message")
        .contains("empty"));
}

#[tokio::test]
async fn completions_rejects_multiplex_with_stream() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let body = serde_json::json!({
        "model": ["copilot:gpt-4o", "claude:opus"],
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse json");

    assert!(json["error"]["message"]
        .as_str()
        .expect("message")
        .contains("Streaming"));
}

#[tokio::test]
async fn completions_returns_error_for_invalid_json() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(b"not valid json".to_vec()))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Axum returns 422 for JSON deserialization failures
    assert!(
        response.status().is_client_error(),
        "expected client error, got {}",
        response.status()
    );
}

// ============================================================================
// Input Validation
// ============================================================================

#[tokio::test]
async fn completions_rejects_temperature_above_max() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 2.5
    });
    let (status, json) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("temperature"));
}

#[tokio::test]
async fn completions_rejects_negative_temperature() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": -0.1
    });
    let (status, json) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("temperature"));
}

#[tokio::test]
async fn completions_rejects_zero_max_tokens() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 0
    });
    let (status, json) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["error"]["message"]
        .as_str()
        .expect("msg")
        .contains("max_tokens"));
}

#[tokio::test]
async fn completions_accepts_temperature_at_boundaries() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    // temperature=0.0 should pass validation (fail at provider, not 400)
    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "temperature 0.0 should pass validation"
    );

    // temperature=2.0 should pass validation
    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 2.0
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "temperature 2.0 should pass validation"
    );
}

#[tokio::test]
async fn completions_accepts_valid_max_tokens() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "max_tokens 1 should pass validation"
    );
}

#[tokio::test]
async fn multiplex_forwards_temperature_past_validation() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    // Multiplex with valid temperature should pass validation and reach
    // the provider layer (5xx for missing binary, not 400 for validation)
    let body = serde_json::json!({
        "model": ["copilot", "claude"],
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 100
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "multiplex with valid params should pass validation"
    );
}

// ============================================================================
// Tool Calling
// ============================================================================

#[tokio::test]
async fn completions_accepts_tools_with_tool_choice_none() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    // tool_choice=none should skip tool injection even when tools are provided
    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": "none"
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    // Should pass validation (not 400) — the provider layer handles execution
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "tool_choice=none should pass validation"
    );
}

#[tokio::test]
async fn completions_accepts_tool_choice_specific() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "specific tool_choice should pass validation"
    );
}

#[tokio::test]
async fn completions_accepts_response_format() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"}
    });
    let (status, _) = send_and_parse(test_app(), post_completions(&body)).await;
    // json_object format should be accepted by the parser (provider may or may not support it)
    assert_ne!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "response_format should deserialize"
    );
}

// ============================================================================
// Router
// ============================================================================

#[tokio::test]
async fn unknown_route_returns_404() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn completions_rejects_get_method() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/chat/completions")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn models_rejects_post_method() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ============================================================================
// Streaming Format
// ============================================================================

#[tokio::test]
async fn streaming_generate_id_format() {
    let id = embacle_server::completions::generate_id();
    assert!(
        id.starts_with("chatcmpl-"),
        "ID should start with chatcmpl-, got: {id}"
    );
    assert!(
        id.len() > "chatcmpl-".len(),
        "ID should have content after prefix"
    );
}

#[tokio::test]
async fn generate_id_produces_unique_ids() {
    let ids: Vec<String> = (0..100)
        .map(|_| embacle_server::completions::generate_id())
        .collect();
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(
        ids.len(),
        unique.len(),
        "Expected all IDs to be unique within the same second"
    );
}

#[tokio::test]
async fn generate_id_unique_across_concurrent_tasks() {
    let mut handles = Vec::with_capacity(50);
    for _ in 0..50 {
        handles.push(tokio::spawn(async {
            embacle_server::completions::generate_id()
        }));
    }

    let mut ids = Vec::with_capacity(50);
    for handle in handles {
        ids.push(handle.await.expect("task join"));
    }

    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(
        ids.len(),
        unique.len(),
        "Expected all IDs to be unique across concurrent tasks"
    );
}

// ============================================================================
// Provider Resolver (integration-level)
// ============================================================================

#[tokio::test]
async fn provider_resolver_round_trip_via_completions() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let app = test_app();

    // A single-model request with a provider prefix
    let body = serde_json::json!({
        "model": "copilot:gpt-4o",
        "messages": [{"role": "user", "content": "test"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
                .expect("build request"),
        )
        .await
        .expect("send request");

    // Verify correct routing: provider prefix is parsed and forwarded to the
    // runner layer. The result depends on binary availability — 5xx (binary
    // missing or invocation failure) or 2xx (actual completion). Either way it
    // must NOT be a 4xx, which would indicate a routing/parsing error.
    let status = response.status();
    assert!(
        !status.is_client_error(),
        "expected non-4xx for provider-prefixed model, got {status}"
    );
}

// ============================================================================
// State
// ============================================================================

#[tokio::test]
async fn state_active_provider_accessible() {
    let state = ServerState::new(CliRunnerType::ClaudeCode);
    assert_eq!(state.active_provider(), CliRunnerType::ClaudeCode);
}

#[tokio::test]
async fn state_runner_creation_reflects_binary_availability() {
    let state = ServerState::new(CliRunnerType::OpenCode);
    let result = state.get_runner(CliRunnerType::OpenCode).await;

    // Verify get_runner outcome matches actual binary availability
    let binary_on_path = which::which("opencode").is_ok();
    if binary_on_path {
        assert!(
            result.is_ok(),
            "opencode binary found on PATH but get_runner failed"
        );
    } else {
        assert!(
            result.is_err(),
            "opencode binary not on PATH but get_runner succeeded"
        );
    }
}

#[test]
fn resolve_binary_fails_for_missing_binary() {
    let result = embacle::discovery::resolve_binary("embacle_nonexistent_test_binary_xyz", None);
    assert!(result.is_err(), "expected error for missing binary");
}

// ============================================================================
// MCP Endpoint (via embacle-mcp)
// ============================================================================

/// Build a POST /mcp request with a JSON-RPC body
fn post_mcp(body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("serialize")))
        .expect("build request")
}

#[tokio::test]
async fn mcp_initialize_returns_protocol_version() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "test-client", "version": "1.0" }
        }
    });

    let (status, json) = send_and_parse(test_app(), post_mcp(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["jsonrpc"], "2.0");
    assert_eq!(json["id"], 1);
    assert_eq!(json["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(json["result"]["serverInfo"]["name"], "embacle-mcp");
    assert!(json["result"]["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn mcp_tools_list_returns_registered_tools() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    });

    let (status, json) = send_and_parse(test_app(), post_mcp(&body)).await;
    assert_eq!(status, StatusCode::OK);

    let tools = json["result"]["tools"].as_array().expect("tools is array");
    assert_eq!(tools.len(), 7, "expected 7 MCP tools from embacle-mcp");

    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"prompt"), "missing prompt tool");
    assert!(names.contains(&"get_provider"), "missing get_provider tool");
    assert!(names.contains(&"set_provider"), "missing set_provider tool");
    assert!(names.contains(&"get_model"), "missing get_model tool");
    assert!(names.contains(&"set_model"), "missing set_model tool");
    assert!(
        names.contains(&"get_multiplex_provider"),
        "missing get_multiplex_provider tool"
    );
    assert!(
        names.contains(&"set_multiplex_provider"),
        "missing set_multiplex_provider tool"
    );
}

#[tokio::test]
async fn mcp_ping_returns_empty_object() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "ping"
    });

    let (status, json) = send_and_parse(test_app(), post_mcp(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["result"], serde_json::json!({}));
}

#[tokio::test]
async fn mcp_unknown_method_returns_error() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "nonexistent/method"
    });

    let (status, json) = send_and_parse(test_app(), post_mcp(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["error"]["code"], -32601);
}

#[tokio::test]
async fn mcp_notification_returns_no_content() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });

    let app = test_app();
    let response = app.oneshot(post_mcp(&body)).await.expect("send request");

    assert_eq!(response.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn mcp_invalid_json_returns_parse_error() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::remove_var("EMBACLE_API_KEY");

    let request = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .body(Body::from("not valid json"))
        .expect("build request");

    let (status, json) = send_and_parse(test_app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["error"]["code"], -32700);
}

#[tokio::test]
async fn mcp_auth_enforced_when_key_set() {
    let _guard = ENV_MUTEX.lock().await;
    std::env::set_var("EMBACLE_API_KEY", "mcp-test-key");

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "ping"
    });

    // No auth header → 401
    let (status, _) = send_and_parse(test_app(), post_mcp(&body)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // With auth header → 200
    let request = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("content-type", "application/json")
        .header("Authorization", "Bearer mcp-test-key")
        .body(Body::from(serde_json::to_vec(&body).expect("serialize")))
        .expect("build request");

    let (status, json) = send_and_parse(test_app(), request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["result"], serde_json::json!({}));

    std::env::remove_var("EMBACLE_API_KEY");
}
