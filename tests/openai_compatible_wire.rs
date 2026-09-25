// ABOUTME: What the OpenAI-compatible provider puts on the wire and reads back, against a local listener
// ABOUTME: Pins the OPENAI_API_* configuration's requests, model discovery, errors, health and streaming
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Each test serves canned HTTP answers from a listener on `127.0.0.1`,
//! records every request the provider sends, and asserts both halves: the
//! exact JSON body and headers that left, and what the provider made of the
//! recorded answer.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
#![cfg(feature = "http-api")]

use std::env;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;

use embacle::http_api::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use embacle::types::{
    ChatMessage, ChatRequest, ErrorKind, ImagePart, LlmCapabilities, LlmProvider, ResponseFormat,
    ToolCallRequest, ToolChoice, ToolDefinition,
};

// ============================================================================
// Harness
// ============================================================================

/// One request as the listener received it
#[derive(Debug)]
struct Recorded {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the request body is JSON")
    }
}

/// A complete HTTP/1.1 answer that closes its connection
fn http(status: u16, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} Canned\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Read one request: headers up to the blank line, then `Content-Length` bytes.
async fn read_request(stream: &mut TcpStream) -> Recorded {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut buf).await.expect("read request");
        assert!(n > 0, "connection closed before the headers ended");
        raw.extend_from_slice(&buf[..n]);
        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().expect("request line").split(' ');
    let method = request_line.next().expect("method").to_owned();
    let path = request_line.next().expect("path").to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(": "))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .map_or(0, |(_, v)| v.parse().expect("numeric content-length"));
    let mut body = raw[header_end..].to_vec();
    while body.len() < length {
        let n = stream.read(&mut buf).await.expect("read body");
        assert!(n > 0, "connection closed before the body ended");
        body.extend_from_slice(&buf[..n]);
    }
    Recorded {
        method,
        path,
        headers,
        body: String::from_utf8(body).expect("UTF-8 body"),
    }
}

/// Serve `answers` in order, one connection each, and hand back what was asked.
async fn serve(answers: Vec<String>) -> (String, JoinHandle<Vec<Recorded>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    let handle = tokio::spawn(async move {
        let mut recorded = Vec::new();
        for answer in answers {
            let (mut stream, _) = listener.accept().await.expect("accept");
            recorded.push(read_request(&mut stream).await);
            stream
                .write_all(answer.as_bytes())
                .await
                .expect("write answer");
            stream.shutdown().await.expect("shutdown");
        }
        recorded
    });
    (base, handle)
}

/// A base URL nothing listens on
async fn closed_base() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    drop(listener);
    base
}

/// The `OpenAI` API configuration against `base`
fn openai_api(base: &str) -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::with_client(
        OpenAiCompatibleConfig {
            base_url: format!("{base}/v1"),
            api_key: Some("sk-test".to_owned()),
            ..OpenAiCompatibleConfig::openai_api("gpt-4o")
        },
        reqwest::Client::new(),
    )
}

/// Ollama's configuration against `base`
fn ollama(base: &str) -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::with_client(
        OpenAiCompatibleConfig {
            base_url: format!("{base}/v1"),
            ..OpenAiCompatibleConfig::ollama("qwen2.5:14b-instruct")
        },
        reqwest::Client::new(),
    )
}

/// A conversation that uses every optional field the wire format has
fn full_request() -> ChatRequest {
    let mut assistant = ChatMessage::assistant("");
    assistant.tool_calls = Some(vec![ToolCallRequest {
        id: "call_1".to_owned(),
        function_name: "get_weather".to_owned(),
        arguments: json!({"city": "Paris"}),
    }]);
    let image = ImagePart::new("aGVsbG8=", "image/png").expect("valid mime");
    ChatRequest::new(vec![
        ChatMessage::system("Be brief."),
        ChatMessage::user_with_images("Describe", vec![image]),
        assistant,
        ChatMessage::tool("get_weather", "call_1", r#"{"temp":21}"#),
    ])
    .with_temperature(0.5)
    .with_max_tokens(64)
    .with_top_p(0.5)
    .with_stop(vec!["END".to_owned()])
    .with_response_format(ResponseFormat::JsonSchema {
        name: "forecast".to_owned(),
        schema: json!({"type": "object"}),
    })
    .with_tools(vec![ToolDefinition {
        name: "get_weather".to_owned(),
        description: "Weather for a city".to_owned(),
        parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
    }])
    .with_tool_choice(ToolChoice::Specific {
        name: "get_weather".to_owned(),
    })
}

/// The messages of [`full_request`] as a provider without vision sends them
fn text_messages() -> Value {
    json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Describe"},
        {"role": "assistant", "content": null, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
        }]},
        {"role": "tool", "content": "{\"temp\":21}", "tool_call_id": "call_1"}
    ])
}

/// The tools of [`full_request`] on the wire
fn wire_tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }}])
}

/// A recorded `OpenAI` answer that calls a tool
const TOOL_CALL_ANSWER: &str = r#"{
    "id": "chatcmpl-9", "object": "chat.completion", "created": 1, "model": "gpt-4o-2024-08-06",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": null,
        "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lyon\"}"}},
            {"id": "call_def", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":"}}
        ]
    }}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 7}}
}"#;

// ============================================================================
// Requests and answers
// ============================================================================

/// The `OpenAI` API configuration advertises vision, `top_p`, stop sequences
/// and `response_format`, so every one of them reaches the body, beside the
/// tool history and the requested tool choice; the answer's tool calls and
/// usage breakdown come back decoded.
#[tokio::test]
async fn openai_api_sends_the_full_request_and_reads_tool_calls_and_usage() {
    let (base, server) = serve(vec![http(200, "application/json", TOOL_CALL_ANSWER)]).await;
    let provider = openai_api(&base);

    let response = provider.complete(&full_request()).await.expect("completes");
    let recorded = server.await.expect("server");

    assert_eq!(recorded.len(), 1);
    let sent = &recorded[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/v1/chat/completions");
    assert_eq!(sent.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(sent.header("content-type"), Some("application/json"));
    assert_eq!(
        sent.json(),
        json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]},
                {"role": "tool", "content": "{\"temp\":21}", "tool_call_id": "call_1"}
            ],
            "temperature": 0.5,
            "max_tokens": 64,
            "top_p": 0.5,
            "stop": ["END"],
            "stream": false,
            "tools": wire_tools(),
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "forecast", "schema": {"type": "object"}
            }}
        })
    );

    assert_eq!(provider.name(), "openai_api");
    assert_eq!(response.model, "gpt-4o-2024-08-06");
    assert_eq!(response.content, "");
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    let calls = response.tool_calls.expect("tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_abc");
    assert_eq!(calls[0].function_name, "get_weather");
    assert_eq!(calls[0].arguments, json!({"city": "Lyon"}));
    // Arguments that are not JSON become null rather than a string that
    // only looks like them.
    assert_eq!(calls[1].id, "call_def");
    assert_eq!(calls[1].arguments, Value::Null);
    let usage = response.usage.expect("usage");
    assert_eq!(
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        ),
        (120, 30, 150)
    );
    assert_eq!(usage.cached_read_tokens, Some(100));
    assert_eq!(usage.cached_write_tokens, None);
    assert_eq!(usage.reasoning_tokens, Some(7));
}

/// A self-hosted preset advertises none of those four: the same request
/// reaches Ollama without them and without the image, and the provider
/// still bills under the endpoint's own name.
#[tokio::test]
async fn a_self_hosted_endpoint_gets_only_what_it_advertises() {
    let (base, server) = serve(vec![http(200, "application/json", TOOL_CALL_ANSWER)]).await;
    let provider = ollama(&base);

    let response = provider.complete(&full_request()).await.expect("completes");
    let sent = &server.await.expect("server")[0];

    assert_eq!(sent.path, "/v1/chat/completions");
    assert_eq!(sent.header("authorization"), None);
    assert_eq!(
        sent.json(),
        json!({
            "model": "qwen2.5:14b-instruct",
            "messages": text_messages(),
            "temperature": 0.5,
            "max_tokens": 64,
            "stream": false,
            "tools": wire_tools(),
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        })
    );
    assert_eq!(provider.name(), "ollama");
    assert_eq!(response.tool_calls.expect("tool calls")[0].id, "call_abc");
}

/// Tools without a choice ask for `auto`; an empty tool list sends neither
/// `tools` (the API rejects an empty array) nor a `tool_choice`.
#[tokio::test]
async fn tool_choice_travels_only_beside_tools() {
    let answer =
        r#"{"model":"gpt-4o","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
    let (base, server) = serve(vec![
        http(200, "application/json", answer),
        http(200, "application/json", answer),
    ])
    .await;
    let provider = openai_api(&base);

    let with_tools =
        ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition {
            name: "lookup".to_owned(),
            description: "Lookup".to_owned(),
            parameters: None,
        }]);
    let empty_tools = ChatRequest::new(vec![ChatMessage::user("hi")])
        .with_tools(vec![])
        .with_tool_choice(ToolChoice::Required);
    let first = provider.complete(&with_tools).await.expect("completes");
    provider.complete(&empty_tools).await.expect("completes");
    let recorded = server.await.expect("server");

    assert_eq!(first.content, "ok");
    assert_eq!(
        recorded[0].json(),
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "Lookup"}}],
            "tool_choice": "auto"
        })
    );
    assert_eq!(
        recorded[1].json(),
        json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        })
    );
}

// ============================================================================
// Streaming
// ============================================================================

/// A stream's deltas arrive in order and end on the vendor's finish reason;
/// a frame with no `delta` (Azure's content-filter annotation) and a
/// usage-only trailer with no choice add nothing.
#[tokio::test]
async fn a_stream_yields_its_deltas_and_the_finish_reason() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Bon\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"content_filter_results\":{},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"jour\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base, server) = serve(vec![http(200, "text/event-stream", sse)]).await;
    let provider = openai_api(&base);

    let stream = provider
        .complete_stream(&ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .expect("stream opens");
    let chunks: Vec<_> = stream.collect::<Vec<_>>().await;
    let sent = &server.await.expect("server")[0];

    assert_eq!(sent.json()["stream"], true);
    let chunks: Vec<_> = chunks.into_iter().map(|c| c.expect("chunk")).collect();
    let deltas: Vec<&str> = chunks.iter().map(|c| c.delta.as_str()).collect();
    assert_eq!(deltas, vec!["Bon", "jour", ""]);
    let last = chunks.last().expect("a final chunk");
    assert!(last.is_final);
    assert_eq!(last.finish_reason.as_deref(), Some("stop"));
    assert_eq!(chunks.iter().filter(|c| c.is_final).count(), 1);
}

/// A frame that is not a completion chunk — a vendor's mid-stream error
/// envelope — ends the stream with an error instead of a silently
/// truncated answer.
#[tokio::test]
async fn a_mid_stream_error_frame_is_an_error() {
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Par\"},\"finish_reason\":null}]}\n\n",
        "data: {\"error\":{\"message\":\"upstream overloaded\",\"type\":\"server_error\"}}\n\n",
    );
    let (base, server) = serve(vec![http(200, "text/event-stream", sse)]).await;
    let provider = ollama(&base);

    let stream = provider
        .complete_stream(&ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .expect("stream opens");
    let items: Vec<_> = stream.collect::<Vec<_>>().await;
    server.await.expect("server");

    assert_eq!(items[0].as_ref().expect("first delta").delta, "Par");
    let err = items[1].as_ref().expect_err("the error frame");
    assert_eq!(err.kind, ErrorKind::ExternalService);
    assert!(err.message.contains("SSE parse error"), "{}", err.message);
}

// ============================================================================
// Errors and health
// ============================================================================

/// The `OpenAI` API's failures read as its own message under the shared
/// status mapping: a 404 is an external-service error carrying the vendor's
/// text, a 429 a rate limit carrying the wait, and a body that is not the
/// envelope is quoted as sent.
#[tokio::test]
async fn openai_api_errors_carry_the_vendor_message() {
    let (base, server) = serve(vec![
        http(
            404,
            "application/json",
            r#"{"error":{"message":"The model `gpt-9` does not exist","type":"invalid_request_error"}}"#,
        ),
        http(
            429,
            "application/json",
            r#"{"error":{"message":"Rate limit reached. Please try again in 2s.","type":"requests"}}"#,
        ),
        http(502, "text/html", "<html>bad gateway</html>"),
    ])
    .await;
    let provider = openai_api(&base);
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

    let not_found = provider.complete(&request).await.expect_err("404");
    let limited = provider.complete(&request).await.expect_err("429");
    let gateway = provider.complete_stream(&request).await.err().expect("502");
    server.await.expect("server");

    assert_eq!(not_found.kind, ErrorKind::ExternalService);
    assert!(
        not_found
            .message
            .contains("The model `gpt-9` does not exist"),
        "{}",
        not_found.message
    );
    assert!(
        not_found.message.starts_with("openai_api"),
        "{}",
        not_found.message
    );
    assert_eq!(limited.kind, ErrorKind::RateLimit);
    assert!(limited.message.contains("2 seconds"), "{}", limited.message);
    assert_eq!(gateway.kind, ErrorKind::ExternalService);
    assert!(
        gateway.message.contains("<html>bad gateway</html>"),
        "{}",
        gateway.message
    );
}

/// A self-hosted endpoint keeps its hints: a gateway page means the server
/// is not responding, a 404 envelope is an unpulled model.
#[tokio::test]
async fn a_self_hosted_endpoint_errors_with_its_hints() {
    let (base, server) = serve(vec![
        http(502, "text/html", "<html>bad gateway</html>"),
        http(
            404,
            "application/json",
            r#"{"error":{"message":"model 'qwen3' not found","type":"api_error"}}"#,
        ),
    ])
    .await;
    let provider = ollama(&base);
    let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

    let gateway = provider.complete(&request).await.expect_err("502");
    let missing = provider.complete(&request).await.expect_err("404");
    server.await.expect("server");

    assert_eq!(gateway.kind, ErrorKind::ExternalService);
    assert!(
        gateway.message.contains("not responding"),
        "{}",
        gateway.message
    );
    assert_eq!(missing.kind, ErrorKind::ModelUnavailable);
    assert!(missing.message.contains("qwen3"), "{}", missing.message);
}

/// The `OpenAI` API that cannot be reached is unhealthy, not an error; a
/// self-hosted server that cannot be reached is an error naming it.
#[tokio::test]
async fn an_unreachable_endpoint_is_unhealthy_or_named() {
    let base = closed_base().await;

    assert!(!openai_api(&base).health_check().await.expect("no error"));
    let err = ollama(&base).health_check().await.expect_err("an error");
    assert_eq!(err.kind, ErrorKind::ExternalService);
    assert!(
        err.message.contains("Is the server running"),
        "{}",
        err.message
    );
}

/// A reachable endpoint answers the health probe on `GET /v1/models` with the
/// configured credential.
#[tokio::test]
async fn the_health_probe_asks_for_the_models_list() {
    let (base, server) = serve(vec![http(200, "application/json", r#"{"data":[]}"#)]).await;

    assert!(openai_api(&base).health_check().await.expect("healthy"));
    let sent = &server.await.expect("server")[0];
    assert_eq!(sent.method, "GET");
    assert_eq!(sent.path, "/v1/models");
    assert_eq!(sent.header("authorization"), Some("Bearer sk-test"));
}

// ============================================================================
// Model discovery
// ============================================================================

/// Discovery publishes the endpoint's own list, sorted; a failed discovery
/// keeps the configured model.
#[tokio::test]
async fn discovery_publishes_the_endpoint_models_or_keeps_the_default() {
    let (base, server) = serve(vec![http(
        200,
        "application/json",
        r#"{"object":"list","data":[{"id":"gpt-4o-mini","object":"model"},{"id":"gpt-4o","object":"model"}]}"#,
    )])
    .await;

    let discovered = openai_api(&base).with_discovered_models().await;
    let sent = &server.await.expect("server")[0];
    assert_eq!(sent.path, "/v1/models");
    assert_eq!(sent.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(discovered.available_models(), ["gpt-4o", "gpt-4o-mini"]);

    let unreachable = openai_api(&closed_base().await)
        .with_discovered_models()
        .await;
    assert_eq!(unreachable.available_models(), ["gpt-4o"]);
}

// ============================================================================
// Configuration from OPENAI_API_*
// ============================================================================

/// Serialises the tests that write `OPENAI_API_*`.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

const OPENAI_API_VARS: [&str; 4] = [
    "OPENAI_API_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_API_MODEL",
    "OPENAI_API_TIMEOUT_SECS",
];

fn with_openai_env<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let guard = ENV_MUTEX.lock().unwrap_or_else(PoisonError::into_inner);
    let saved: Vec<(&str, Option<String>)> = OPENAI_API_VARS
        .iter()
        .map(|name| (*name, env::var(name).ok()))
        .collect();
    for name in OPENAI_API_VARS {
        env::remove_var(name);
    }
    for (name, value) in vars {
        env::set_var(name, value);
    }
    let result = f();
    for (name, value) in saved {
        match value {
            Some(v) => env::set_var(name, v),
            None => env::remove_var(name),
        }
    }
    drop(guard);
    result
}

/// Unset, the variables name the `OpenAI` API itself on `gpt-5.4`.
#[test]
fn openai_api_from_env_defaults_to_the_openai_api() {
    let config = with_openai_env(&[], OpenAiCompatibleConfig::openai_api_from_env);
    assert_eq!(config.base_url, "https://api.openai.com/v1");
    assert_eq!(config.api_key, None);
    assert_eq!(config.default_model, "gpt-5.4");
    assert_eq!(config.provider_name, "openai_api");
    assert_eq!(config.display_name, "OpenAI API");
    assert_eq!(config.timeout, Duration::from_secs(120));
    assert_eq!(
        config.capabilities,
        LlmCapabilities::STREAMING
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::VISION
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::TEMPERATURE
            | LlmCapabilities::MAX_TOKENS
            | LlmCapabilities::TOP_P
            | LlmCapabilities::STOP_SEQUENCES
            | LlmCapabilities::RESPONSE_FORMAT
    );
}

/// `OPENAI_API_BASE_URL` names the host: requests go to its `/v1`, whatever
/// trailing slashes it carries.
#[test]
fn openai_api_from_env_reads_every_variable() {
    let config = with_openai_env(
        &[
            ("OPENAI_API_BASE_URL", "https://api.groq.com/openai//"),
            ("OPENAI_API_KEY", "gsk-test"),
            ("OPENAI_API_MODEL", "llama-3.3-70b-versatile"),
            ("OPENAI_API_TIMEOUT_SECS", "30"),
        ],
        OpenAiCompatibleConfig::openai_api_from_env,
    );
    assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
    assert_eq!(config.api_key.as_deref(), Some("gsk-test"));
    assert_eq!(config.default_model, "llama-3.3-70b-versatile");
    assert_eq!(config.timeout, Duration::from_secs(30));

    let unparsed = with_openai_env(
        &[("OPENAI_API_TIMEOUT_SECS", "soon"), ("OPENAI_API_KEY", "")],
        OpenAiCompatibleConfig::openai_api_from_env,
    );
    assert_eq!(unparsed.timeout, Duration::from_secs(120));
    assert_eq!(unparsed.api_key, None);
}

/// A provider configured from `OPENAI_API_*` sends to the configured host's
/// `/v1/chat/completions` with the configured key and model.
#[tokio::test]
async fn openai_api_from_env_reaches_the_configured_endpoint() {
    let answer = r#"{"model":"m","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}"#;
    let (base, server) = serve(vec![http(200, "application/json", answer)]).await;
    let config = with_openai_env(
        &[
            ("OPENAI_API_BASE_URL", base.as_str()),
            ("OPENAI_API_KEY", "sk-env"),
            ("OPENAI_API_MODEL", "gpt-env"),
        ],
        OpenAiCompatibleConfig::openai_api_from_env,
    );
    let provider = OpenAiCompatibleProvider::new(config).expect("builds");

    let response = provider
        .complete(&ChatRequest::new(vec![ChatMessage::user("hello")]))
        .await
        .expect("completes");
    let sent = &server.await.expect("server")[0];

    assert_eq!(response.content, "hi");
    assert_eq!(sent.path, "/v1/chat/completions");
    assert_eq!(sent.header("authorization"), Some("Bearer sk-env"));
    assert_eq!(sent.json()["model"], "gpt-env");
    assert_eq!(provider.available_models(), ["gpt-env"]);
}
