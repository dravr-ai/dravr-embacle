// ABOUTME: End-to-end test binary for ClaudeCodeRunner against the real claude CLI
// ABOUTME: Validates subprocess spawning, JSON parsing, streaming, auth, and compat detection
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Usage: cargo run -p cli-llm-runners --example test_claude_e2e

//! End-to-end test for Claude Code CLI runner.

use std::process;
use std::time::Duration;

use embacle::compat::detect_capabilities;
use embacle::config::CliRunnerType;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::{auth::check_readiness, discovery::resolve_binary, ClaudeCodeRunner, RunnerConfig};
use tokio_stream::StreamExt;

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() {
    // ── 1. Binary discovery ──
    println!("━━━ 1. Binary Discovery ━━━");
    let binary_path = match resolve_binary("claude", None) {
        Ok(p) => {
            println!("  ✅ Found claude at: {}", p.display());
            p
        }
        Err(e) => {
            eprintln!("  ❌ claude binary not found: {e}");
            process::exit(1);
        }
    };

    // ── 2. Version / compatibility ──
    println!("\n━━━ 2. Version & Compatibility ━━━");
    match detect_capabilities(CliRunnerType::ClaudeCode, &binary_path).await {
        Ok(caps) => {
            println!("  Version:    {}", caps.version_string);
            println!(
                "  Parsed:     {}",
                caps.version.map_or_else(
                    || "unparseable".to_owned(),
                    |(ma, mi, p)| format!("{ma}.{mi}.{p}")
                )
            );
            println!("  Min met:    {}", caps.meets_minimum_version);
            println!("  JSON:       {}", caps.json_output);
            println!("  Streaming:  {}", caps.streaming);
            println!("  Sys prompt: {}", caps.system_prompt);
            println!("  Resume:     {}", caps.session_resume);
            println!(
                "  Compatible: {}",
                if caps.is_compatible() { "✅" } else { "❌" }
            );
        }
        Err(e) => {
            eprintln!("  ⚠️  Capability detection failed: {e}");
        }
    }

    // ── 3. Auth readiness ──
    println!("\n━━━ 3. Auth Readiness ━━━");
    match check_readiness(&CliRunnerType::ClaudeCode, &binary_path).await {
        Ok(readiness) => {
            println!("  Readiness: {readiness}");
            if !readiness.is_ready() {
                eprintln!("  ❌ Claude Code is not authenticated. Run: claude auth login");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("  ⚠️  Readiness check failed: {e}");
            eprintln!("  Continuing anyway (check may not be implemented for this version)...");
        }
    }

    // ── 4. Build runner ──
    println!("\n━━━ 4. Build Runner ━━━");
    let config = RunnerConfig::new(binary_path).with_timeout(Duration::from_secs(60));
    let runner = ClaudeCodeRunner::new(config);
    println!("  Name:          {}", runner.name());
    println!("  Display name:  {}", runner.display_name());
    println!("  Default model: {}", runner.default_model());
    println!("  Capabilities:  {:?}", runner.capabilities());
    println!("  Models:        {:?}", runner.available_models());

    // ── 5. Simple completion ──
    println!("\n━━━ 5. Simple Completion ━━━");
    let request = ChatRequest {
        messages: vec![
            ChatMessage::system("You are a test bot. Follow instructions exactly."),
            ChatMessage::user("Respond with exactly: PING_OK. Nothing else."),
        ],
        model: None,
        temperature: None,
        max_tokens: Some(20),
        stream: false,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
    };

    match runner.complete(&request).await {
        Ok(response) => {
            println!("  Content:  {:?}", response.content);
            println!("  Model:    {:?}", response.model);
            println!("  Tokens:   {:?}", response.usage);
            if response.content.contains("PING_OK") {
                println!("  ✅ Completion works!");
            } else {
                println!("  ⚠️  Unexpected content (but parsing succeeded)");
            }
        }
        Err(e) => {
            eprintln!("  ❌ Completion failed: {e}");
        }
    }

    // ── 6. Streaming completion ──
    println!("\n━━━ 6. Streaming Completion ━━━");
    let stream_request = ChatRequest {
        messages: vec![ChatMessage::user(
            "Count from 1 to 5, each number on a new line.",
        )],
        model: None,
        temperature: None,
        max_tokens: Some(50),
        stream: true,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
    };

    match runner.complete_stream(&stream_request).await {
        Ok(mut stream) => {
            let mut chunk_count = 0u32;
            let mut full_content = String::new();
            print!("  Chunks: ");
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        chunk_count += 1;
                        if !chunk.delta.is_empty() {
                            full_content.push_str(&chunk.delta);
                            print!(".");
                        }
                    }
                    Err(e) => {
                        eprintln!("\n  ❌ Stream error: {e}");
                        break;
                    }
                }
            }
            println!();
            println!("  Total chunks: {chunk_count}");
            println!("  Full content: {full_content:?}");
            if chunk_count > 0 {
                println!("  ✅ Streaming works!");
            } else {
                println!("  ⚠️  Zero chunks received");
            }
        }
        Err(e) => {
            eprintln!("  ❌ Stream setup failed: {e}");
        }
    }

    // ── 7. Health check ──
    println!("\n━━━ 7. Health Check ━━━");
    match runner.health_check().await {
        Ok(healthy) => println!(
            "  {}",
            if healthy {
                "✅ Healthy"
            } else {
                "❌ Unhealthy"
            }
        ),
        Err(e) => eprintln!("  ❌ Health check failed: {e}"),
    }

    println!("\n━━━ Done ━━━");
}
