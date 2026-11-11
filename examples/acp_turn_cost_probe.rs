// ABOUTME: Measures what one Copilot ACP turn costs through the real runner: prompt, output, reasoning, requests
// ABOUTME: Runs complete, complete_stream and an Autopilot converse against an MCP server the caller names
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run --example acp_turn_cost_probe --features copilot-headless
//
// Optional: PROBE_MCP_URL=http://127.0.0.1:8765/mcp adds a converse() turn that
// calls that server's tools under Autopilot. COPILOT_HEADLESS_REASONING_EFFORT
// and COPILOT_HEADLESS_BUILTIN_TOOLS change the launch, as they do in a host.
//
// WHY THIS EXISTS
//
// A sweep of 2,976 snippet calls read a median completion of 21x the visible
// answer and ~22,800 prompt tokens of Copilot's own per call. Each of those
// numbers has a cause that is only visible on the wire (reasoning counted twice,
// built-in tool schemas, the CLI's effortLevel, Autopilot continuations), and
// each fix is only proven by measuring the turn again. This is that measurement.
//
// NOTE: like every ACP call through this runner, spawning pins the routing model
// in ~/.copilot/settings.json (see `ensure_copilot_settings_model`).

use std::env;
use std::io::stderr;
use std::time::{Duration, Instant};

use embacle::types::{ChatMessage, ChatRequest, LlmProvider, McpServerConfig, McpTransport};
use embacle::CopilotHeadlessRunner;
use tokio::time::timeout;
use tokio_stream::StreamExt;
use tracing::Level;
use tracing_subscriber::fmt;

const SNIPPET: &str = "Convert this JMeter HTTP sampler into a Gatling Scala DSL snippet. \
Answer with only the code block.\n\n<HTTPSamplerProxy testname=\"Login\">\
<stringProp name=\"HTTPSampler.domain\">api.example.com</stringProp>\
<stringProp name=\"HTTPSampler.path\">/v1/login</stringProp>\
<stringProp name=\"HTTPSampler.method\">POST</stringProp></HTTPSamplerProxy>";

fn request(prompt: &str, mcp_servers: Vec<McpServerConfig>) -> ChatRequest {
    let mut request =
        ChatRequest::new(vec![ChatMessage::user(prompt)]).with_mcp_servers(mcp_servers);
    request.max_tokens = Some(2048);
    request
}

#[tokio::main]
async fn main() {
    fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_writer(stderr)
        .init();

    let runner = CopilotHeadlessRunner::from_env();

    println!("=== complete() ===");
    let started = Instant::now();
    match timeout(
        Duration::from_mins(2),
        runner.complete(&request(SNIPPET, Vec::new())),
    )
    .await
    {
        Ok(Ok(resp)) => {
            println!("secs:    {:.1}", started.elapsed().as_secs_f64());
            println!("finish:  {:?}", resp.finish_reason);
            println!("visible: {} chars", resp.content.len());
            println!("leaked notice: {}", resp.content.contains("Info:"));
            println!("usage:   {:?}", resp.usage);
        }
        Ok(Err(e)) => println!("complete() error: {e}"),
        Err(_) => println!("timed out after 120s"),
    }

    println!("\n=== complete_stream() ===");
    let started = Instant::now();
    match runner.complete_stream(&request(SNIPPET, Vec::new())).await {
        Ok(mut stream) => {
            let mut text = String::new();
            let mut finish = None;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => {
                        text.push_str(&chunk.delta);
                        if chunk.is_final {
                            finish = chunk.finish_reason;
                        }
                    }
                    Err(e) => println!("stream error: {e}"),
                }
            }
            println!("secs:    {:.1}", started.elapsed().as_secs_f64());
            println!("finish:  {finish:?}");
            println!("visible: {} chars", text.len());
            println!("leaked notice: {}", text.contains("Info:"));
        }
        Err(e) => println!("complete_stream() error: {e}"),
    }

    let Ok(url) = env::var("PROBE_MCP_URL") else {
        println!("\n(PROBE_MCP_URL unset: no converse() turn)");
        return;
    };
    println!("\n=== converse() with an MCP server, Autopilot ===");
    let server = McpServerConfig {
        name: "probe".to_owned(),
        transport: McpTransport::Http {
            url,
            headers: Vec::new(),
        },
    };
    let started = Instant::now();
    let ask = "Call the secret_word tool and tell me the word it returns.";
    match timeout(
        Duration::from_mins(2),
        runner.converse(&request(ask, vec![server])),
    )
    .await
    {
        Ok(Ok(resp)) => {
            println!("secs:    {:.1}", started.elapsed().as_secs_f64());
            println!("finish:  {:?}", resp.finish_reason);
            println!("content: {}", resp.content.trim());
            println!(
                "tools:   {:?}",
                resp.tool_calls.iter().map(|t| &t.title).collect::<Vec<_>>()
            );
            println!("usage:   {:?}", resp.usage);
        }
        Ok(Err(e)) => println!("converse() error: {e}"),
        Err(_) => println!("timed out after 120s"),
    }
}
