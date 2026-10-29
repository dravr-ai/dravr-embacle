// ABOUTME: The Anthropic provider as pool accounts, against a local listener: names, headers, a spent credit, a fallback
// ABOUTME: Pins that a 402 on account 1 reaches the chain as a quota refusal naming that account, and account 2 answers or streams
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Each test serves canned HTTP answers from a listener on `127.0.0.1`, so no
//! request leaves the machine.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
#![cfg(feature = "http-api")]

use std::sync::Arc;

use serde_json::Value;
use tokio_stream::StreamExt;

use embacle::fallback::{FallbackProvider, ResponsePolicy};
use embacle::http_api::client::HttpRetryConfig;
use embacle::http_api::{AnthropicConfig, AnthropicProvider};
use embacle::metrics::MetricsProvider;
use embacle::pool::provider_pool;
use embacle::types::{
    ChatMessage, ChatRequest, ErrorKind, LlmProvider, ToolDefinition, ToolStreamEvent,
};

use crate::http_listener::{http, serve};

/// An account that may retry in place: a status that must not be retried
/// shows up as a second recorded request if it is.
fn account(key: &str, base: &str) -> Box<dyn LlmProvider> {
    let config = AnthropicConfig::new(key)
        .with_base_url(base)
        .with_retry(HttpRetryConfig {
            max_retries: 2,
            initial_delay_ms: 1,
            max_delay_ms: 1,
        });
    Box::new(AnthropicProvider::with_client(
        config,
        reqwest::Client::new(),
    ))
}

/// A 429 with the `retry-after` header Anthropic sends
fn rate_limited() -> String {
    let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your per-minute rate limit"}}"#;
    format!(
        "HTTP/1.1 429 Canned\r\nContent-Type: application/json\r\nretry-after: 30\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

const ANSWER: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-5-5",
    "content":[{"type":"text","text":"Bonjour"}],"stop_reason":"end_turn",
    "usage":{"input_tokens":5,"output_tokens":2,"cache_read_input_tokens":600,"cache_creation_input_tokens":0}}"#;

const SPENT: &str = r#"{"type":"error","error":{"type":"billing_error","message":"Your credit balance is too low to access the Anthropic API."}}"#;

#[test]
fn accounts_are_named_in_order_and_the_first_keeps_the_provider_name() {
    let tiers = provider_pool(vec![
        account("key-1", "http://127.0.0.1:9"),
        account("key-2", "http://127.0.0.1:9"),
        account("key-3", "http://127.0.0.1:9"),
    ])
    .expect("a pool of three accounts builds");
    let names: Vec<&str> = tiers.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["anthropic", "anthropic#2", "anthropic#3"]);
    assert_eq!(tiers[0].display_name(), "Anthropic API");
    assert_eq!(tiers[1].display_name(), "Anthropic API (account 2)");
    assert_eq!(tiers[2].default_model(), "claude-haiku-5-5");
}

#[test]
fn an_empty_provider_pool_is_a_config_error() {
    let Err(err) = provider_pool(Vec::new()) else {
        panic!("an empty pool must be refused");
    };
    assert_eq!(err.kind, ErrorKind::Config);
}

#[tokio::test]
async fn a_spent_account_refuses_as_a_quota_error_naming_the_account() {
    let (base, server) = serve(vec![http(402, SPENT)]).await;
    let tiers = provider_pool(vec![
        account("key-1", "http://127.0.0.1:9"),
        account("key-2", &base),
    ])
    .expect("pool");
    let err = tiers[1]
        .complete(&ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .expect_err("a spent credit refuses");
    assert_eq!(err.kind, ErrorKind::RateLimit);
    assert!(
        err.message
            .starts_with("anthropic#2: credit balance exhausted"),
        "{}",
        err.message
    );
    assert!(!err.message.contains("key-2"));
    let recorded = server.await.expect("server");
    assert_eq!(
        recorded.len(),
        1,
        "a spent credit is never retried in place"
    );
}

#[tokio::test]
async fn the_chain_moves_from_a_spent_account_to_the_next_one() {
    let (base, server) = serve(vec![http(402, SPENT), http(200, ANSWER)]).await;
    let tiers =
        provider_pool(vec![account("key-1", &base), account("key-2", &base)]).expect("pool");
    let chain = FallbackProvider::new(tiers).expect("chain");
    let request = ChatRequest::new(vec![
        ChatMessage::system("coach"),
        ChatMessage::user("salut"),
    ]);
    let response = chain.complete(&request).await.expect("account 2 answers");
    assert_eq!(response.content, "Bonjour");
    assert_eq!(
        response.model, "claude-haiku-5-5",
        "the response keeps the model the account ran; the tier name says which account"
    );
    let usage = response.usage.expect("usage");
    assert_eq!(usage.prompt_tokens, 605);
    assert_eq!(usage.cached_read_tokens, Some(600));

    let recorded = server.await.expect("server");
    assert_eq!(recorded.len(), 2);
    let keys: Vec<Option<&str>> = recorded.iter().map(|r| r.header("x-api-key")).collect();
    assert_eq!(keys, [Some("key-1"), Some("key-2")]);
    for request in &recorded {
        assert_eq!(request.path, "/v1/messages");
        assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
        let body: Value = serde_json::from_str(&request.body).expect("JSON body");
        assert_eq!(body["model"], "claude-haiku-5-5");
        assert_eq!(body["system"][0]["text"], "coach");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }
    assert_eq!(
        recorded[0].body, recorded[1].body,
        "both accounts receive the same bytes"
    );
}

#[tokio::test]
async fn a_rate_limited_account_hands_the_turn_on_without_retrying_in_place() {
    let (base, server) = serve(vec![rate_limited(), http(200, ANSWER)]).await;
    let tiers =
        provider_pool(vec![account("key-1", &base), account("key-2", &base)]).expect("pool");
    let chain = FallbackProvider::new(tiers).expect("chain");
    let response = chain
        .complete(&ChatRequest::new(vec![ChatMessage::user("salut")]))
        .await
        .expect("account 2 answers");
    assert_eq!(response.content, "Bonjour");

    let recorded = server.await.expect("server");
    let keys: Vec<Option<&str>> = recorded.iter().map(|r| r.header("x-api-key")).collect();
    assert_eq!(
        keys,
        [Some("key-1"), Some("key-2")],
        "account 1 is asked once, then account 2"
    );
}

#[tokio::test]
async fn an_overload_is_retried_in_place_on_the_same_account() {
    let overloaded =
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let (base, server) = serve(vec![http(529, overloaded), http(200, ANSWER)]).await;
    let response = account("key-1", &base)
        .complete(&ChatRequest::new(vec![ChatMessage::user("salut")]))
        .await
        .expect("the retry answers");
    assert_eq!(response.content, "Bonjour");
    let recorded = server.await.expect("server");
    assert_eq!(recorded.len(), 2);
    assert!(recorded
        .iter()
        .all(|r| r.header("x-api-key") == Some("key-1")));
}

#[tokio::test]
async fn every_pool_account_prices_its_usage() {
    let (base, server) = serve(vec![http(200, ANSWER), http(200, ANSWER)]).await;
    let tiers =
        provider_pool(vec![account("key-1", &base), account("key-2", &base)]).expect("pool");
    let request = ChatRequest::new(vec![ChatMessage::user("salut")]);
    for tier in tiers {
        let name = tier.name();
        let metered = MetricsProvider::new(tier);
        let response = metered.complete(&request).await.expect("answers");
        assert_eq!(response.model, "claude-haiku-5-5", "{name}");
        let report = metered.report().expect("report");
        // On Haiku 5.5: 5 fresh × $0.10 + 600 cache reads × $0.01 + 2 output
        // × $0.50, per million tokens
        let expected = 7.5e-6;
        assert!(
            (report.total_cost - expected).abs() < 1e-15,
            "{name}: {} vs {expected}",
            report.total_cost
        );
    }
    server.await.expect("server");
}

#[tokio::test]
async fn a_strict_chain_moves_past_a_refusal_that_followed_partial_output() {
    let refused = r#"{"id":"msg_r","type":"message","role":"assistant","model":"claude-haiku-5-5",
        "content":[{"type":"text","text":"Sure, the first step is"}],
        "stop_reason":"refusal","stop_details":{"category":"general_harms"},
        "usage":{"input_tokens":5,"output_tokens":6}}"#;
    let (base, server) = serve(vec![http(200, refused), http(200, ANSWER)]).await;
    let tiers =
        provider_pool(vec![account("key-1", &base), account("key-2", &base)]).expect("pool");
    let chain = FallbackProvider::new(tiers)
        .expect("chain")
        .with_fallthrough(ResponsePolicy::strict());
    let response = chain
        .complete(&ChatRequest::new(vec![ChatMessage::user("salut")]))
        .await
        .expect("the next tier answers");
    assert_eq!(response.content, "Bonjour");
    assert_eq!(server.await.expect("server").len(), 2);
}

/// A `text/event-stream` answer carrying `frames` as `data:` lines
fn sse(frames: &[&str]) -> String {
    let body = frames.join("\n\ndata: ");
    let body = format!("data: {body}\n\n");
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn a_tool_stream_leaves_a_spent_account_and_streams_the_reply_then_the_call() {
    let streamed = sse(&[
        r#"{"type":"message_start","message":{"usage":{"input_tokens":5,"cache_read_input_tokens":600,"cache_creation_input_tokens":40,"output_tokens":1}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Je regarde"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" tes sorties."}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_activities","input":{}}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"limit\":"}}"#,
        r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" 10}"}}"#,
        r#"{"type":"content_block_stop","index":1}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":33}}"#,
        r#"{"type":"message_stop"}"#,
    ]);
    let (base, server) = serve(vec![http(402, SPENT), streamed]).await;
    let tiers =
        provider_pool(vec![account("key-1", &base), account("key-2", &base)]).expect("pool");
    // The shape a consumer holds: the chain shared behind an `Arc`, boxed again
    let chain: Box<dyn LlmProvider> =
        Box::new(Arc::new(FallbackProvider::new(tiers).expect("chain")));
    let request = ChatRequest::new(vec![ChatMessage::user("mes sorties?")]).with_tools(vec![
        ToolDefinition::new(
            "get_activities".to_owned(),
            "Recent activities".to_owned(),
            None,
        ),
    ]);
    let stream = chain
        .complete_stream_with_tools(&request)
        .await
        .expect("account 2 opens the stream");
    let items: Vec<_> = stream.collect().await;

    let mut deltas = Vec::new();
    let mut done = None;
    for item in items {
        match item.expect("no error item") {
            ToolStreamEvent::TextDelta(text) => deltas.push(text),
            ToolStreamEvent::Done(response) => done = Some(response),
            other => panic!("unexpected event {other:?}"),
        }
    }
    assert_eq!(deltas, ["Je regarde", " tes sorties."]);
    let done = done.expect("a Done");
    assert_eq!(done.content, "Je regarde tes sorties.");
    assert_eq!(done.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(done.model, "claude-haiku-5-5");
    let call = &done.tool_calls.expect("a call")[0];
    assert_eq!(call.id, "toolu_1");
    assert_eq!(call.arguments, serde_json::json!({"limit": 10}));
    let usage = done.usage.expect("usage");
    assert_eq!(usage.prompt_tokens, 645);
    assert_eq!(usage.completion_tokens, 33);
    assert_eq!(usage.cached_read_tokens, Some(600));
    assert_eq!(usage.cached_write_tokens, Some(40));

    let recorded = server.await.expect("server");
    let keys: Vec<Option<&str>> = recorded.iter().map(|r| r.header("x-api-key")).collect();
    assert_eq!(keys, [Some("key-1"), Some("key-2")]);
    let body: Value = serde_json::from_str(&recorded[1].body).expect("JSON body");
    assert_eq!(body["stream"], true);
    assert_eq!(body["tools"][0]["name"], "get_activities");
}
