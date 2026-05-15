// ABOUTME: Smoke test exercising TmuxRunner against the real Claude Code or Copilot CLI
// ABOUTME: Spawns a tmux-driven session, sends one prompt, prints the response, tears down
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Usage:
//   cargo run --features tmux-session --example tmux_smoke -- claude "Say hi in one sentence."
//   cargo run --features tmux-session --example tmux_smoke -- copilot "Explain rustc -O in 20 words."
//
// Pre-flight:
//   1. Authenticate the CLI once outside embacle (e.g. run `copilot` in a normal
//      terminal, complete the device-code OAuth flow, exit). The persisted
//      credentials are picked up by every subsequent embacle spawn.
//   2. Set CLAUDE_CODE_BINARY / COPILOT_BINARY if the binaries are not on PATH.
//
// If a turn appears to hang, attach to the live session for diagnosis:
//   tmux attach -t <session-name>   (the program prints this on startup)
// Detach with Ctrl-b d — the embacle runner keeps owning the session.

//! End-to-end smoke test for [`embacle::TmuxRunner`].

use std::env;
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};

use embacle::discovery::resolve_binary;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, MessageRole};
use embacle::{TmuxBackend, TmuxRunner, TmuxRunnerConfig};

#[tokio::main]
async fn main() {
    let mut args = env::args().skip(1);
    let backend_arg = args.next().unwrap_or_else(|| {
        eprintln!("usage: tmux_smoke <claude|copilot> <prompt> [conversation-key]");
        process::exit(2);
    });
    let prompt = args.next().unwrap_or_else(|| {
        eprintln!("error: missing prompt argument");
        process::exit(2);
    });
    let conversation_key = args.next().unwrap_or_else(|| "smoke".to_owned());

    let (backend, binary_name, env_override) = match backend_arg.as_str() {
        "claude" | "claude_code" | "claude-code" => {
            (TmuxBackend::ClaudeCode, "claude", "CLAUDE_CODE_BINARY")
        }
        "copilot" => (TmuxBackend::Copilot, "copilot", "COPILOT_BINARY"),
        other => {
            eprintln!("error: unknown backend '{other}' (expected 'claude' or 'copilot')");
            process::exit(2);
        }
    };

    let env_path = env::var(env_override).ok();
    let binary_path: PathBuf = match resolve_binary(binary_name, env_path.as_deref()) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("error: cannot find {binary_name} binary: {err}");
            eprintln!("hint: install it, put it on PATH, or set {env_override}");
            process::exit(1);
        }
    };
    println!("─ Backend:       {backend:?}");
    println!("─ Binary:        {}", binary_path.display());
    println!("─ Conversation:  {conversation_key}");

    let mut config = match backend {
        TmuxBackend::ClaudeCode => TmuxRunnerConfig::claude_code(binary_path),
        TmuxBackend::Copilot => TmuxRunnerConfig::copilot(binary_path),
    };
    // Real LLM TUIs render banners and may stream responses across many
    // seconds. Loosen the defaults here so an interactive demo does not
    // race the agent's startup or its first reasoning pause.
    config.spawn_max = Duration::from_secs(60);
    config.turn_timeout = Duration::from_secs(180);

    let runner = TmuxRunner::new(config);

    // Print the session name now so the operator can `tmux attach` if the
    // first turn hangs (most commonly: the binary is showing an auth prompt
    // because pre-auth was skipped, and `paste()` is queuing text into the
    // device-code field instead of the chat input).
    let predicted_session = predict_session_name(backend, &conversation_key);
    println!("─ Tmux session:  {predicted_session}");
    println!("                 (attach with: tmux attach -t {predicted_session})");
    println!();

    let request = ChatRequest::new(vec![ChatMessage::new(MessageRole::User, prompt.clone())])
        .with_model(conversation_key.clone());

    println!("▶ Prompt: {prompt}");
    println!(
        "… waiting for response (turn_timeout = {:?})",
        Duration::from_secs(180)
    );

    let started = Instant::now();
    match runner.complete(&request).await {
        Ok(response) => {
            let elapsed = started.elapsed();
            println!();
            println!("◀ Response ({elapsed:?}):");
            println!("{}", response.content);
            println!();

            // Tear down so the operator does not leak a tmux session across
            // smoke-test runs. Comment out to keep the session attachable
            // for inspection.
            if let Err(err) = runner.close_session(&conversation_key).await {
                eprintln!("warning: failed to close tmux session: {err}");
            }
        }
        Err(err) => {
            let elapsed = started.elapsed();
            eprintln!();
            eprintln!("✗ Failed after {elapsed:?}: {err}");
            eprintln!();
            eprintln!("Common causes:");
            eprintln!("  • CLI not authenticated — run '{binary_name}' once manually first");
            eprintln!("  • Auth prompt blocking — attach to the tmux session and complete it");
            eprintln!("  • Marker not echoed — the agent put the done token in a code block");
            eprintln!();
            eprintln!("Attach to inspect: tmux attach -t {predicted_session}");
            // Leave the session alive so the operator can attach and look.
            process::exit(1);
        }
    }
}

/// Replicate the tmux runner's session-naming rule so the smoke test can
/// print the name *before* the first `complete()` call.
fn predict_session_name(backend: TmuxBackend, key: &str) -> String {
    let prefix = match backend {
        TmuxBackend::ClaudeCode => "embacle-claude",
        TmuxBackend::Copilot => "embacle-copilot",
    };
    let sanitised: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{prefix}-{sanitised}")
}
