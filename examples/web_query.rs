// ABOUTME: Send a prompt to the Claude.ai web UI via the browser runner and stream the reply
// ABOUTME: Drives WebUiRunner::complete_stream and prints deltas as they arrive
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::env;
use std::error::Error;
use std::io::{self, Write};

use embacle::types::{ChatMessage, ChatRequest, LlmProvider, MessageRole};
use embacle::WebUiRunner;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Show progress: default to info-level logs from the runner + browser layer.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("embacle::web_ui=info,dravr_browser=info")
            }),
        )
        .with_target(false)
        .init();

    let prompt = env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in one short sentence.".to_owned());

    eprintln!(
        "→ starting Claude.ai web query (set EMBACLE_WEB_HEADLESS=false to watch the browser)"
    );
    let runner = WebUiRunner::from_env()?;

    let request = ChatRequest {
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: prompt,
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
        model: None,
        temperature: None,
        max_tokens: None,
        stream: true,
        tools: None,
        tool_choice: None,
        top_p: None,
        stop: None,
        response_format: None,
        turn_id: None,
        mcp_servers: Vec::new(),
    };

    let mut stream = runner.complete_stream(&request).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{}", chunk.delta);
        io::stdout().flush().ok();
        if chunk.is_final {
            println!();
        }
    }
    Ok(())
}
