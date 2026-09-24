// ABOUTME: Proves concurrent `copilot --acp` spawns each run on the model they asked for,
// ABOUTME: against a stand-in CLI that reads settings.json after exec the way the real one does
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! `copilot --acp` takes its served model from `$HOME/.copilot/settings.json`,
//! which the runner writes just before each spawn and the child reads some time
//! AFTER exec. Two spawns for different models can therefore interleave — A
//! writes X, B writes Y, A's child reads Y — and A's turn runs on the wrong
//! model with nothing downstream able to tell. `MODEL_ROUTING_GATE` closes that
//! by serializing write → spawn → `initialize` for every child; a spawn path
//! that skips it reopens the race for every turn it serves.
//!
//! The stand-in CLI below speaks just enough ACP to serve a turn, waits before
//! reading settings.json (the real CLI's post-exec read, made wide enough to
//! land every time), and answers each prompt with the model it read. A turn
//! whose reply is not the model it requested ran on another turn's model.
//!
//! One test in its own binary on purpose: it points `HOME` at a scratch
//! directory, which is process-global, so nothing else may run beside it.

#![cfg(all(feature = "copilot-headless", unix))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process;

use embacle::types::{ChatMessage, ChatRequest};
use embacle::{CopilotHeadlessConfig, CopilotHeadlessRunner, HeadlessStreamEvent};
use tokio_stream::StreamExt;

/// A minimal `copilot --acp`: answers `initialize`, `session/new` and
/// `session/prompt`, and replies to the prompt with the model settings.json
/// held when the child read it. The pause before the read stands in for the
/// real CLI's startup before it loads its configuration.
const STAND_IN_CLI: &str = r#"#!/bin/bash
model="unread"
session="s-$$"
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            sleep 0.4
            model=$(sed -n 's/.*"model"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$HOME/.copilot/settings.json")
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":1}}\n' "$id" ;;
        *'"method":"session/new"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"sessionId":"%s"}}\n' "$id" "$session" ;;
        *'"method":"session/prompt"'*)
            printf '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"%s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"%s"}}}}\n' "$session" "$model"
            printf '{"jsonrpc":"2.0","id":%s,"result":{"stopReason":"end_turn"}}\n' "$id" ;;
    esac
done
"#;

fn scratch_home() -> PathBuf {
    let home = env::temp_dir().join(format!("embacle-acp-routing-{}", process::id()));
    let _ = fs::remove_dir_all(&home);
    fs::create_dir_all(&home).unwrap();
    home
}

fn install_stand_in(dir: &Path) -> PathBuf {
    let cli = dir.join("copilot");
    fs::write(&cli, STAND_IN_CLI).unwrap();
    fs::set_permissions(&cli, fs::Permissions::from_mode(0o755)).unwrap();
    cli
}

fn ask(model: &str) -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("Which model are you?")]).with_model(model)
}

async fn streamed_reply(runner: &CopilotHeadlessRunner, model: &str) -> String {
    let mut stream = runner.converse_stream(&ask(model)).await.unwrap();
    while let Some(event) = stream.next().await {
        if let HeadlessStreamEvent::Done(response) = event.unwrap() {
            return response.content;
        }
    }
    panic!("the {model} stream closed without a Done event");
}

async fn pooled_reply(runner: &CopilotHeadlessRunner, model: &str) -> String {
    runner.converse(&ask(model)).await.unwrap().content
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_spawns_each_run_on_the_model_they_asked_for() {
    let home = scratch_home();
    // This binary holds only this test, so no other thread reads HOME.
    env::set_var("HOME", &home);
    let runner = CopilotHeadlessRunner::with_config(CopilotHeadlessConfig {
        cli_path: Some(install_stand_in(&home)),
        working_directory: Some(home.join("cwd")),
        model: "default-model".to_owned(),
        github_token: None,
        ..CopilotHeadlessConfig::default()
    });

    // Two streamed turns: each spawns its own child, every time.
    let (alpha, beta) = tokio::join!(
        streamed_reply(&runner, "model-alpha"),
        streamed_reply(&runner, "model-beta"),
    );
    assert_eq!(
        alpha, "model-alpha",
        "the first stream ran on another turn's model"
    );
    assert_eq!(
        beta, "model-beta",
        "the second stream ran on another turn's model"
    );

    // A pooled turn and a streamed turn: the two spawn paths share one file.
    let (pooled, streamed) = tokio::join!(
        pooled_reply(&runner, "model-gamma"),
        streamed_reply(&runner, "model-delta"),
    );
    assert_eq!(
        pooled, "model-gamma",
        "the pooled turn ran on the stream's model"
    );
    assert_eq!(
        streamed, "model-delta",
        "the stream ran on the pooled turn's model"
    );

    let _ = fs::remove_dir_all(&home);
}
