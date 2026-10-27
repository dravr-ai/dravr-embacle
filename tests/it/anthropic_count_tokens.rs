// ABOUTME: AnthropicProvider::count_tokens against a local listener: path, headers, body and the count it returns
// ABOUTME: Pins that the count sends the request's input (tools included) and maps refusals like a completion does
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Each test serves canned HTTP answers from a listener on `127.0.0.1`, so no
//! request leaves the machine.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "http-api")]

use serde_json::{json, Value};

use embacle::http_api::client::HttpRetryConfig;
use embacle::http_api::{AnthropicConfig, AnthropicProvider};
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, ToolDefinition};

use crate::http_listener::{http, serve};

/// A provider on `base` that may retry in place, so a refusal that must not be
/// retried shows up as a second recorded request if it is
fn provider(base: &str) -> AnthropicProvider {
    let config = AnthropicConfig::new("key-1")
        .with_model("claude-haiku-5-5")
        .with_base_url(base)
        .with_retry(HttpRetryConfig {
            max_retries: 2,
            initial_delay_ms: 1,
            max_delay_ms: 1,
        });
    AnthropicProvider::with_client(config, reqwest::Client::new())
}

fn request_with_a_tool() -> ChatRequest {
    ChatRequest::new(vec![
        ChatMessage::system("You are a coach."),
        ChatMessage::user("What is my plan?"),
    ])
    .with_tools(vec![ToolDefinition {
        name: "get_training_plan".to_owned(),
        description: "Fetch the athlete's active training plan.".to_owned(),
        parameters: Some(json!({"type": "object", "properties": {}})),
    }])
}

#[tokio::test]
async fn a_count_posts_the_request_input_and_returns_input_tokens() {
    let (base, server) = serve(vec![http(200, r#"{"input_tokens":1234}"#)]).await;

    let counted = provider(&base)
        .count_tokens(&request_with_a_tool())
        .await
        .expect("the count answers");

    assert_eq!(counted, 1234);
    let recorded = server.await.expect("server");
    assert_eq!(recorded.len(), 1);
    let sent = &recorded[0];
    assert_eq!(sent.path, "/v1/messages/count_tokens");
    assert_eq!(sent.header("x-api-key"), Some("key-1"));
    assert_eq!(sent.header("anthropic-version"), Some("2023-06-01"));
    let body: Value = serde_json::from_str(&sent.body).expect("a JSON body");
    assert_eq!(body["model"], "claude-haiku-5-5");
    assert_eq!(body["tools"][0]["name"], "get_training_plan");
    assert_eq!(body["system"][0]["text"], "You are a coach.");
    assert_eq!(body["messages"][0]["role"], "user");
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("stream").is_none());
}

#[tokio::test]
async fn an_unknown_model_refuses_as_unavailable_without_a_retry() {
    let not_found =
        r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-nope"}}"#;
    let (base, server) = serve(vec![http(404, not_found)]).await;

    let err = provider(&base)
        .count_tokens(&request_with_a_tool())
        .await
        .expect_err("an unknown model refuses");

    assert_eq!(err.kind, ErrorKind::ModelUnavailable);
    assert_eq!(server.await.expect("server").len(), 1);
}

#[tokio::test]
async fn a_body_that_is_not_a_count_is_an_error() {
    let (base, _server) = serve(vec![http(200, r#"{"tokens":12}"#)]).await;

    let err = provider(&base)
        .count_tokens(&request_with_a_tool())
        .await
        .expect_err("a body without input_tokens is refused");

    assert_eq!(err.kind, ErrorKind::ExternalService);
}
