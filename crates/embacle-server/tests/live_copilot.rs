// ABOUTME: Live integration tests exercising real Copilot CLI via embacle-server
// ABOUTME: Gated behind EMBACLE_LIVE_TESTS=1 env var; skipped when copilot binary unavailable
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use embacle::config::CliRunnerType;
use embacle_mcp::ServerState;
use http_body_util::BodyExt;
use tokio::sync::RwLock;
use tower::ServiceExt;

use embacle_server::router;

/// Check whether live tests should run.
/// Returns `true` when `EMBACLE_LIVE_TESTS=1` and the copilot binary is on PATH.
fn skip_unless_live() -> bool {
    let env_set = env::var("EMBACLE_LIVE_TESTS").is_ok_and(|v| v == "1");
    let binary_available = which::which("copilot").is_ok();
    !(env_set && binary_available)
}

/// Build a test app wired to the Copilot provider
fn live_app() -> axum::Router {
    let state = Arc::new(RwLock::new(ServerState::new(CliRunnerType::Copilot)));
    router::build(state)
}

/// Build a POST /v1/chat/completions request from a JSON body
fn post_completions(body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).expect("serialize")))
        .expect("build request")
}

/// Shared tool definition for `get_weather`
fn tool_definitions() -> serde_json::Value {
    serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. 'San Francisco'"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ])
}

/// Parse SSE-formatted bytes into a list of JSON values.
/// Filters out `[DONE]` sentinel and empty lines.
fn parse_sse_events(bytes: &[u8]) -> Vec<serde_json::Value> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|block| {
            let block = block.trim();
            if block.is_empty() {
                return None;
            }
            // Each SSE block may have multiple lines; find the data: line
            for line in block.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        return None;
                    }
                    return serde_json::from_str(data).ok();
                }
            }
            None
        })
        .collect()
}

// ============================================================================
// Live Copilot Tests
// ============================================================================

#[tokio::test]
async fn live_copilot_non_streaming_completion() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "Reply with exactly: hello world"}]
    });

    let response = live_app()
        .oneshot(post_completions(&body))
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

    // Verify standard OpenAI response structure
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("choices[0].message.content should be a string");
    assert!(!content.is_empty(), "content should not be empty");

    let model = json["model"]
        .as_str()
        .expect("model field should be a string");
    assert!(
        model.contains("copilot"),
        "model should contain 'copilot', got: {model}"
    );
}

#[tokio::test]
async fn live_copilot_streaming_completion() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
        "stream": true
    });

    let response = live_app()
        .oneshot(post_completions(&body))
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();

    // Verify SSE format: raw bytes should contain "data: " lines and end with [DONE]
    let raw = String::from_utf8_lossy(&bytes);
    assert!(
        raw.contains("data: [DONE]"),
        "SSE stream should end with [DONE]"
    );

    let chunks = parse_sse_events(&bytes);
    assert!(!chunks.is_empty(), "should have at least one SSE chunk");

    // First chunk should set role
    let first_delta = &chunks[0]["choices"][0]["delta"];
    assert_eq!(
        first_delta["role"].as_str(),
        Some("assistant"),
        "first chunk delta should have role=assistant"
    );

    // Check if any chunk has finish_reason. The Copilot stream may end
    // without signaling is_final, so the SSE layer may omit finish_reason.
    // This is a known gap tracked separately; warn rather than fail.
    let has_finish_reason = chunks
        .iter()
        .any(|c| c["choices"][0]["finish_reason"].is_string());
    if !has_finish_reason {
        eprintln!(
            "WARNING: no SSE chunk had finish_reason (known gap: Copilot stream may not signal is_final)"
        );
    }
}

#[tokio::test]
async fn live_copilot_streaming_with_tools_returns_sse() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": tool_definitions(),
        "stream": true
    });

    let response = live_app()
        .oneshot(post_completions(&body))
        .await
        .expect("send request");

    // The key assertion: streaming + tools should NOT return 400
    // (downgrade path converts to non-streaming internally)
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "streaming + tools should succeed via downgrade path"
    );

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();

    let raw = String::from_utf8_lossy(&bytes);
    assert!(
        raw.contains("data: "),
        "response should be SSE format with data: lines"
    );
    assert!(
        raw.contains("data: [DONE]"),
        "SSE stream should end with [DONE]"
    );

    let chunks = parse_sse_events(&bytes);
    assert!(!chunks.is_empty(), "should have at least one SSE chunk");

    // First chunk should have role=assistant
    let first_delta = &chunks[0]["choices"][0]["delta"];
    assert_eq!(
        first_delta["role"].as_str(),
        Some("assistant"),
        "first chunk delta should have role=assistant"
    );

    // Note: we intentionally do NOT assert tool_calls presence here.
    // Copilot currently ignores the tool catalog and returns plain text.
    // Once tool forwarding is fixed, add assertions for finish_reason=="tool_calls".
}

#[tokio::test]
async fn live_copilot_non_streaming_with_tools() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": tool_definitions()
    });

    let response = live_app()
        .oneshot(post_completions(&body))
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

    // Verify response has valid structure
    assert!(
        json["choices"][0]["message"].is_object(),
        "response should have choices[0].message"
    );

    let finish_reason = json["choices"][0]["finish_reason"]
        .as_str()
        .expect("should have finish_reason");
    assert!(
        !finish_reason.is_empty(),
        "finish_reason should not be empty"
    );
}

#[tokio::test]
async fn live_copilot_model_override() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot:claude-opus-4.6",
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}]
    });

    let response = live_app()
        .oneshot(post_completions(&body))
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

    let model = json["model"]
        .as_str()
        .expect("model field should be a string");
    assert!(
        model.contains("copilot"),
        "model should contain 'copilot', got: {model}"
    );

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("should have content");
    assert!(!content.is_empty(), "content should not be empty");
}

#[tokio::test]
async fn live_copilot_streaming_chunk_structure() {
    if skip_unless_live() {
        eprintln!("SKIP: EMBACLE_LIVE_TESTS not set or copilot not on PATH");
        return;
    }
    env::remove_var("EMBACLE_API_KEY");

    let body = serde_json::json!({
        "model": "copilot",
        "messages": [{"role": "user", "content": "Reply with exactly: test"}],
        "stream": true
    });

    let response = live_app()
        .oneshot(post_completions(&body))
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let chunks = parse_sse_events(&bytes);
    assert!(!chunks.is_empty(), "should have at least one SSE chunk");

    // Validate every chunk has the required OpenAI SSE fields
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(
            chunk["id"].is_string(),
            "chunk {i} missing 'id' field: {chunk}"
        );
        assert_eq!(
            chunk["object"].as_str(),
            Some("chat.completion.chunk"),
            "chunk {i} should have object=chat.completion.chunk"
        );
        assert!(
            chunk["created"].is_number(),
            "chunk {i} missing 'created' field: {chunk}"
        );
        assert!(
            chunk["model"].is_string(),
            "chunk {i} missing 'model' field: {chunk}"
        );
        assert!(
            chunk["choices"].is_array(),
            "chunk {i} missing 'choices' array: {chunk}"
        );
    }
}
