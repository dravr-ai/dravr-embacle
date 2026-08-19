// ABOUTME: Live proof that a real copilot --acp calls a caller's tools via embacle-tool-host
// ABOUTME: The unit tests use a stand-in HTTP client; this uses the actual agent
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run --example tool_host_live --features copilot-headless
//
// Requires a logged-in `copilot` CLI. Proves the whole chain end to end:
// ToolHost binds -> session published in session/new -> Copilot speaks MCP to it
// -> our executor runs the tool BY NAME -> the answer reflects the tool's data.
//
// The unit tests cannot prove this: they drive the listener with reqwest, which
// shows the server is correct but says nothing about whether Copilot's MCP
// client accepts what it serves.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, RunnerError};
use embacle::{CopilotHeadlessRunner, McpToolDefinition, McpToolExecutor};
use embacle_tool_host::{StaticSurface, ToolHost, ToolHostConfig};
use serde_json::{json, Value};
use tokio::time::timeout;
use tracing::Level;
use tracing_subscriber::fmt;

/// A tool whose answer could only come from calling it — a made-up number the
/// model cannot possibly know or guess.
struct SecretNumberTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl McpToolExecutor for SecretNumberTool {
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        println!(">>> EXECUTOR CALLED: tool={tool_name} args={arguments}");
        Ok(json!({ "secret_number": 8_675_309 }))
    }
}

#[tokio::main]
async fn main() {
    fmt().with_max_level(Level::INFO).with_target(false).init();

    let calls = Arc::new(AtomicUsize::new(0));
    let host = match ToolHost::bind(ToolHostConfig {
        server_name: "dravr".to_owned(),
        ..ToolHostConfig::default()
    })
    .await
    {
        Ok(h) => h,
        Err(e) => {
            println!("FAIL: could not bind tool host: {e}");
            return;
        }
    };
    println!("tool host listening on {}", host.local_addr());

    let session = host.open_session(Arc::new(StaticSurface::new(
        vec![McpToolDefinition {
            name: "get_secret_number".to_owned(),
            description: "Returns the secret number. The ONLY way to learn it.".to_owned(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }],
        Arc::new(SecretNumberTool {
            calls: Arc::clone(&calls),
        }),
    )));
    println!("session {} published", session.session_id());

    let runner = CopilotHeadlessRunner::from_env();
    let request = ChatRequest {
        messages: vec![ChatMessage::user(
            "Call the get_secret_number tool and tell me the number it returns. \
             Reply with just the number.",
        )],
        model: None,
        temperature: Some(0.0),
        max_tokens: Some(256),
        stream: false,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: session.mcp_servers(),
    };

    match timeout(Duration::from_mins(3), runner.converse(&request)).await {
        Ok(Ok(resp)) => {
            println!("\n--- content ---\n{}", resp.content);
            println!("--- observed tool calls: {} ---", resp.tool_calls.len());
            for tc in &resp.tool_calls {
                println!("    title={:?} status={:?}", tc.title, tc.status);
            }
        }
        Ok(Err(e)) => println!("converse() error: {e}"),
        Err(_) => println!("timed out after 180s"),
    }

    let served = session.calls_served();
    let executed = calls.load(Ordering::SeqCst);
    println!("\n=== VERDICT ===");
    println!("calls_served (host)      : {served}");
    println!("executor invocations     : {executed}");
    if executed >= 1 {
        println!("PASS: a real copilot --acp reached the caller's executor by name.");
    } else {
        println!("FAIL: the agent never called the tool — the MCP handshake did not work.");
    }
}
