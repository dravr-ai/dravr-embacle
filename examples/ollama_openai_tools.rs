// ABOUTME: Live check that embacle's OpenAI-compatible runner still does structured tools
// ABOUTME: Points at Ollama, so it needs no cloud key and pins the contract locally
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: ollama serve &
//      cargo run --example ollama_openai_tools --features openai-api
//
// This is the OTHER half of the provider contract. `OpenAiApiRunner` advertises
// FUNCTION_CALLING, so a host routes it to the structured loop and expects
// `ChatResponse.tool_calls` to come back populated — no MCP, no prompt catalog.
// Ollama speaks the same wire format, which makes it a free local conformance rig.

use std::env;
use std::time::Duration;

use embacle::types::{ChatMessage, ChatRequest, LlmProvider, ToolDefinition};
use embacle::{OpenAiApiConfig, OpenAiApiRunner};
use serde_json::json;
use tokio::time::timeout;

#[tokio::main]
async fn main() {
    env::set_var("OPENAI_API_BASE_URL", "http://localhost:11434");
    env::set_var("OPENAI_API_KEY", "ollama");
    let model =
        env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b-instruct-q4_K_M".to_owned());
    env::set_var("OPENAI_API_MODEL", &model);

    let runner = OpenAiApiRunner::new(OpenAiApiConfig::from_env()).await;
    println!("provider     : {}", runner.name());
    println!("capabilities : {:?}", runner.capabilities());
    println!("model        : {model}\n");

    let request = ChatRequest {
        messages: vec![ChatMessage::user(
            "What is the secret number? Use the get_secret_number tool.",
        )],
        model: Some(model.clone()),
        temperature: Some(0.0),
        max_tokens: Some(256),
        stream: false,
        tools: Some(vec![ToolDefinition {
            name: "get_secret_number".to_owned(),
            description: "Returns the secret number. The ONLY way to learn it.".to_owned(),
            parameters: Some(json!({ "type": "object", "properties": {} })),
        }]),
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: Vec::new(),
    };

    match timeout(Duration::from_mins(3), runner.complete(&request)).await {
        Ok(Ok(resp)) => {
            println!("--- content ---\n{}", resp.content);
            let calls = resp.tool_calls.unwrap_or_default();
            println!("\n--- structured tool_calls: {} ---", calls.len());
            for c in &calls {
                println!(
                    "    function_name={} arguments={}",
                    c.function_name, c.arguments
                );
            }
            println!("\n=== VERDICT ===");
            if calls.iter().any(|c| c.function_name == "get_secret_number") {
                println!(
                    "PASS: the OpenAI-compatible runner returned a STRUCTURED tool call, \
                     which is exactly what ChatProvider::complete_with_tools now forwards."
                );
            } else {
                println!("FAIL: no structured tool call came back — the contract is not met here.");
            }
        }
        Ok(Err(e)) => println!("FAIL: complete() error: {e}"),
        Err(_) => println!("FAIL: timed out"),
    }
}
