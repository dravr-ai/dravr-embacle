// ABOUTME: Live proof that plain `copilot` (NO ACP) calls a caller's tools via MCP
// ABOUTME: The tool becomes part of Copilot's REAL toolset, not prose in a prompt
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run -p embacle-tool-host --example copilot_cli_mcp_live
//
// Text-simulated tools do not work with Copilot: told about a tool in the prompt it
// answers "that tool isn't part of my real toolset". This registers the host as an
// MCP server via `--additional-mcp-config`, so the tool IS part of that toolset.
//
// Same secret-number probe as the ACP test: a value the model cannot know unless it
// actually called the tool.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, RunnerError};
use embacle::{CopilotRunner, McpToolDefinition, McpToolExecutor, RunnerConfig};
use embacle_tool_host::{StaticSurface, ToolHost, ToolHostConfig};
use serde_json::{json, Value};
use tokio::time::timeout;

const SECRET: &str = "8675309";

struct SecretNumberTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl McpToolExecutor for SecretNumberTool {
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        println!(">>> EXECUTOR CALLED: tool={tool_name} args={arguments}");
        Ok(json!({ "secret_number": SECRET }))
    }
}

#[tokio::main]
async fn main() {
    let calls = Arc::new(AtomicUsize::new(0));

    let host = match ToolHost::bind(ToolHostConfig {
        server_name: "dravr".to_owned(),
        ..ToolHostConfig::default()
    })
    .await
    {
        Ok(h) => h,
        Err(e) => {
            println!("FAIL: bind: {e}");
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

    // Straight through the runner now: no hand-rolled CLI invocation, no
    // config translation here. `CopilotRunner` carries `mcp_servers` itself.
    let runner = CopilotRunner::new(RunnerConfig::new(PathBuf::from("copilot")));
    println!("provider     : {}", runner.name());
    println!("capabilities : {:?}", runner.capabilities());

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

    match timeout(Duration::from_mins(4), runner.complete(&request)).await {
        Ok(Ok(resp)) => {
            println!("\n--- content ---\n{}", resp.content);
            let executed = calls.load(Ordering::SeqCst);
            println!("\n=== VERDICT ===");
            println!("executor invocations : {executed}");
            println!("calls_served (host)  : {}", session.calls_served());
            println!("secret in the answer : {}", resp.content.contains(SECRET));
            if executed >= 1 && resp.content.contains(SECRET) {
                println!("PASS: CopilotRunner (no ACP) called the caller's tool over MCP.");
            } else {
                println!("FAIL: the tool was not reached.");
            }
        }
        Ok(Err(e)) => println!("FAIL: complete() error: {e}"),
        Err(_) => println!("FAIL: timed out after 240s"),
    }
}
