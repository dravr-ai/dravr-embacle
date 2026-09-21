// ABOUTME: Live proof that a real `claude -p` calls a caller's tools via embacle-tool-host
// ABOUTME: Exits non-zero when the hosted tool was never reached, for complete and streaming
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run -p embacle-tool-host --example claude_code_mcp_live
//
// Requires a logged-in `claude` CLI. `ClaudeCodeRunner` hands the session to the
// CLI as `--mcp-config … --strict-mcp-config --allowed-tools mcp__<server>`, and
// Claude Code then speaks MCP to the host on its own.
//
// Claude Code negotiates the newest revision the host advertises, so this is the
// one runner that exercises the host's 2026-07-28 surface. A host that answers
// `tools/list` without the `ttlMs` and `cacheScope` that revision requires is
// reported `connected` with zero tools: no error on either side, a model with
// nothing to call, and `calls_served` stuck at 0. The unit tests drive the
// listener with reqwest and cannot see that; only the real client can.
//
// Same secret-number probe as the Copilot proofs: a value the model cannot know
// unless it actually called the tool.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, RunnerError};
use embacle::{ClaudeCodeRunner, McpToolDefinition, McpToolExecutor, RunnerConfig};
use embacle_tool_host::{StaticSurface, ToolHost, ToolHostConfig, ToolSession};
use futures_util::StreamExt;
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

fn probe(session: &ToolSession, stream: bool) -> ChatRequest {
    ChatRequest {
        messages: vec![ChatMessage::user(
            "Call the get_secret_number tool and tell me the number it returns. \
             Reply with just the number.",
        )],
        model: None,
        temperature: None,
        max_tokens: None,
        stream,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: session.mcp_servers(),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
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
            return ExitCode::FAILURE;
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

    let runner = ClaudeCodeRunner::new(RunnerConfig::new(PathBuf::from("claude")));
    println!("provider     : {}", runner.name());

    let completed = complete_the_turn(&runner, &session, &calls).await;
    let streamed = stream_the_same_turn(&runner, &session, &calls).await;

    if completed && streamed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// One blocking turn. Passes only when the executor ran AND its value came back.
async fn complete_the_turn(
    runner: &ClaudeCodeRunner,
    session: &ToolSession,
    calls: &Arc<AtomicUsize>,
) -> bool {
    let content = match timeout(
        Duration::from_mins(4),
        runner.complete(&probe(session, false)),
    )
    .await
    {
        Ok(Ok(resp)) => resp.content,
        Ok(Err(e)) => {
            println!("FAIL: complete() error: {e}");
            return false;
        }
        Err(_) => {
            println!("FAIL: complete() timed out");
            return false;
        }
    };
    println!("\n--- content ---\n{content}");
    verdict("complete", calls.load(Ordering::SeqCst), session, &content)
}

/// The same turn again, streamed. `complete_stream` builds its own command, so
/// wiring `mcp_servers` into `complete` alone would leave streaming toolless.
async fn stream_the_same_turn(
    runner: &ClaudeCodeRunner,
    session: &ToolSession,
    calls: &Arc<AtomicUsize>,
) -> bool {
    println!("\n=== streaming ===");
    let before = calls.load(Ordering::SeqCst);
    let mut stream = match runner.complete_stream(&probe(session, true)).await {
        Ok(s) => s,
        Err(e) => {
            println!("FAIL: complete_stream() error: {e}");
            return false;
        }
    };
    let mut content = String::new();
    let drained = timeout(Duration::from_mins(4), async {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(c) => content.push_str(&c.delta),
                Err(e) => {
                    println!("FAIL: stream error: {e}");
                    return false;
                }
            }
        }
        true
    })
    .await;
    if !matches!(drained, Ok(true)) {
        println!("FAIL: the stream did not drain");
        return false;
    }
    println!("--- streamed content ---\n{content}");
    verdict(
        "stream",
        calls.load(Ordering::SeqCst) - before,
        session,
        &content,
    )
}

fn verdict(mode: &str, executed: usize, session: &ToolSession, content: &str) -> bool {
    let answered = content.contains(SECRET);
    println!("=== VERDICT ({mode}) ===");
    println!("executor invocations : {executed}");
    println!("calls_served (host)  : {}", session.calls_served());
    println!("secret in the answer : {answered}");
    if executed >= 1 && answered {
        println!("PASS: a real claude -p called the caller's tool over MCP.");
        true
    } else {
        println!("FAIL: the tool was not reached.");
        false
    }
}
