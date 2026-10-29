// ABOUTME: Live proof that ClaudeCodeRunner hands an image's bytes to the model through `claude -p`
// ABOUTME: The answer is only readable from the image, so a text-only turn cannot fake it
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: VISION_IMAGE=/abs/path/label.png VISION_EXPECT="CARBS 42 g" \
//      cargo run --example claude_code_vision_smoke
//
// A user message with images goes to the CLI as one `--input-format
// stream-json` message whose content carries base64 image blocks. This asks
// what the image says, through both `complete` and `complete_stream`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command as StdCommand;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use embacle::config::RunnerConfig;
use embacle::types::{ChatMessage, ChatRequest, ImagePart, LlmProvider};
use embacle::ClaudeCodeRunner;
use tokio_stream::StreamExt;

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
    let (Ok(path), Ok(expect)) = (env::var("VISION_IMAGE"), env::var("VISION_EXPECT")) else {
        println!("set VISION_IMAGE to a PNG path and VISION_EXPECT to the text it shows");
        return;
    };
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("cannot read {path}: {e}");
            return;
        }
    };
    let image = match ImagePart::new(BASE64.encode(bytes), "image/png") {
        Ok(image) => image,
        Err(e) => {
            println!("{e}");
            return;
        }
    };
    let Some(claude) = which_claude() else {
        println!("claude is not on PATH");
        return;
    };
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(claude).with_model("haiku"));
    let request = ChatRequest::new(vec![
        ChatMessage::system("You transcribe text from images exactly."),
        ChatMessage::user_with_images(
            "What exact text is written in the image? Reply with only that text.",
            vec![image],
        ),
    ]);

    println!(
        "vision capability: {}",
        runner.capabilities().supports_vision()
    );

    match runner.complete(&request).await {
        Ok(resp) => println!(
            "complete: {:?} model={} READ: {}",
            resp.content,
            resp.model,
            resp.content.contains(&expect)
        ),
        Err(e) => println!("complete() error: {e}"),
    }

    match runner.complete_stream(&request).await {
        Ok(mut stream) => {
            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => text.push_str(&chunk.delta),
                    Err(e) => {
                        println!("stream error: {e}");
                        return;
                    }
                }
            }
            println!("stream: {text:?} READ: {}", text.contains(&expect));
        }
        Err(e) => println!("complete_stream() error: {e}"),
    }
}
