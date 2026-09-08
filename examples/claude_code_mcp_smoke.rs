// ABOUTME: Live proof that ClaudeCodeRunner reaches MCP tools through `claude -p --mcp-config`
// ABOUTME: The answer is only obtainable from the stdio server, so a text-only turn cannot fake it
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: MINI_MCP=/abs/path/mini_mcp.py cargo run --example claude_code_mcp_smoke
//
// Before this change the runner passed `--strict-mcp-config {}` and dropped
// `request.mcp_servers` on the floor, so every Claude Code turn was tool-less and
// the platform fell back to a prose tool catalog. This asks a question whose
// answer lives only behind a tool call.

use std::env;
use std::path::PathBuf;
use std::process::Command as StdCommand;

use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, McpServerConfig, McpTransport};
use embacle::ClaudeCodeRunner;
use tracing::Level;
use tracing_subscriber::fmt;

/// Resolve the `claude` binary the same way a shell would.
fn which_claude() -> Option<PathBuf> {
    let out = StdCommand::new("sh")
        .args(["-lc", "command -v claude"])
        .output()
        .ok()?;
    let p = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    (!p.is_empty()).then(|| PathBuf::from(p))
}

#[tokio::main]
async fn main() {
    fmt().with_max_level(Level::INFO).with_target(false).init();

    let Ok(script) = env::var("MINI_MCP") else {
        println!("set MINI_MCP to the mini MCP server path");
        return;
    };
    let server = McpServerConfig {
        name: "dravr".to_owned(),
        transport: McpTransport::Stdio {
            command: "python3".to_owned(),
            args: vec![script],
            env: Vec::new(),
        },
    };

    let Some(claude) = which_claude() else {
        println!("claude is not on PATH");
        return;
    };
    let config = RunnerConfig::new(claude).with_model("sonnet");
    let runner = ClaudeCodeRunner::new(config);

    let mut request = ChatRequest::new(vec![ChatMessage::user(
        "What is my FTP? Use the available tool. Answer with the number only.",
    )]);
    request.mcp_servers = vec![server];

    println!("=== claude -p through embacle, with an MCP server on the request ===");
    match runner.complete(&request).await {
        Ok(resp) => {
            println!("\n--- response.content ---\n{}", resp.content);
            let hit = resp.content.contains("287");
            println!("\nTOOL REACHED: {hit}");
            if !hit {
                println!("(287 lives only in the MCP server; without it the model cannot know)");
            }
        }
        Err(e) => println!("complete() error: {e}"),
    }
}
