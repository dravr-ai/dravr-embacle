// ABOUTME: End-to-end integration tests against real CLI binaries
// ABOUTME: Gated by EMBACLE_E2E_<RUNNER>=1 env vars; EMBACLE_E2E_ALL=1 runs everything
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::path::PathBuf;
use std::time::Duration;

use embacle::config::CliRunnerType;
use embacle::discovery::resolve_binary;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
use embacle::RunnerConfig;
use tokio_stream::StreamExt;

/// Check whether a specific E2E runner test is enabled.
///
/// Returns `true` if `EMBACLE_E2E_ALL=1` or `EMBACLE_E2E_<TAG>=1`.
fn runner_enabled(tag: &str) -> bool {
    if std::env::var("EMBACLE_E2E_ALL").as_deref() == Ok("1") {
        return true;
    }
    let key = format!("EMBACLE_E2E_{}", tag.to_uppercase());
    std::env::var(&key).as_deref() == Ok("1")
}

/// Build a simple ping request that any LLM should handle.
fn ping_request() -> ChatRequest {
    ChatRequest::new(vec![
        ChatMessage::system("You are a test bot. Follow instructions exactly."),
        ChatMessage::user("Respond with exactly: PONG. Nothing else."),
    ])
    .with_max_tokens(20)
}

/// Build a streaming request.
fn stream_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user(
        "Count from 1 to 3, each number on its own line. Nothing else.",
    )])
    .with_max_tokens(30)
}

/// Standard timeout for E2E tests (CLI tools can be slow on first invocation).
const E2E_TIMEOUT: Duration = Duration::from_secs(300);

/// Resolve a binary or skip.
fn resolve_or_skip(runner_type: CliRunnerType) -> PathBuf {
    let env_override = std::env::var(runner_type.env_override_key()).ok();
    match resolve_binary(runner_type.binary_name(), env_override.as_deref()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "  SKIP {}: binary not found: {e}",
                runner_type.binary_name()
            );
            // Return a dummy path — test will be skipped by the caller
            PathBuf::from("__not_found__")
        }
    }
}

// ============================================================================
// Shared test harness
// ============================================================================

/// Run the standard battery of tests against any `LlmProvider`.
async fn test_provider_complete(runner: &dyn LlmProvider) {
    let name = runner.name();

    // -- metadata --
    assert!(
        !runner.display_name().is_empty(),
        "{name}: display_name is empty"
    );
    assert!(
        !runner.default_model().is_empty(),
        "{name}: default_model is empty"
    );
    assert!(
        !runner.available_models().is_empty(),
        "{name}: available_models is empty"
    );

    // -- health check --
    let healthy = runner
        .health_check()
        .await
        .unwrap_or_else(|e| panic!("{name}: health_check failed: {e}"));
    assert!(healthy, "{name}: health_check returned false");

    // -- simple completion --
    let request = ping_request();
    let response = runner
        .complete(&request)
        .await
        .unwrap_or_else(|e| panic!("{name}: complete() failed: {e}"));

    assert!(
        !response.content.is_empty(),
        "{name}: complete() returned empty content"
    );
    eprintln!("  {name} complete: {:?}", response.content);
    eprintln!("  {name} model:    {:?}", response.model);
    eprintln!("  {name} usage:    {:?}", response.usage);
}

/// Run streaming tests against a provider that supports it.
async fn test_provider_stream(runner: &dyn LlmProvider) {
    let name = runner.name();

    if !runner.capabilities().supports_streaming() {
        eprintln!("  {name}: streaming not supported, skipping stream test");
        return;
    }

    let request = stream_request();
    let mut stream = runner
        .complete_stream(&request)
        .await
        .unwrap_or_else(|e| panic!("{name}: complete_stream() failed: {e}"));

    let mut chunk_count: u32 = 0;
    let mut full_content = String::new();
    while let Some(result) = stream.next().await {
        let chunk = result.unwrap_or_else(|e| panic!("{name}: stream chunk error: {e}"));
        if !chunk.delta.is_empty() {
            full_content.push_str(&chunk.delta);
            chunk_count += 1;
        }
    }

    assert!(
        chunk_count > 0,
        "{name}: streaming produced 0 non-empty chunks"
    );
    assert!(
        !full_content.is_empty(),
        "{name}: streaming produced empty content"
    );
    eprintln!("  {name} stream: {chunk_count} chunks, content: {full_content:?}");
}

// ============================================================================
// CLI Runner tests
// ============================================================================

#[tokio::test]
async fn e2e_claude_code() {
    if !runner_enabled("claude_code") {
        eprintln!("SKIP e2e_claude_code (set EMBACLE_E2E_CLAUDE_CODE=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::ClaudeCode);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::ClaudeCodeRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_copilot() {
    if !runner_enabled("copilot") {
        eprintln!("SKIP e2e_copilot (set EMBACLE_E2E_COPILOT=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::Copilot);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::CopilotRunner::new(config).await;
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_cursor_agent() {
    if !runner_enabled("cursor_agent") {
        eprintln!("SKIP e2e_cursor_agent (set EMBACLE_E2E_CURSOR_AGENT=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::CursorAgent);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::CursorAgentRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_opencode() {
    if !runner_enabled("opencode") {
        eprintln!("SKIP e2e_opencode (set EMBACLE_E2E_OPENCODE=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::OpenCode);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::OpenCodeRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_gemini_cli() {
    if !runner_enabled("gemini_cli") {
        eprintln!("SKIP e2e_gemini_cli (set EMBACLE_E2E_GEMINI_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::GeminiCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::GeminiCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_codex_cli() {
    if !runner_enabled("codex_cli") {
        eprintln!("SKIP e2e_codex_cli (set EMBACLE_E2E_CODEX_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::CodexCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::CodexCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_goose_cli() {
    if !runner_enabled("goose_cli") {
        eprintln!("SKIP e2e_goose_cli (set EMBACLE_E2E_GOOSE_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::GooseCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::GooseCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_cline_cli() {
    if !runner_enabled("cline_cli") {
        eprintln!("SKIP e2e_cline_cli (set EMBACLE_E2E_CLINE_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::ClineCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::ClineCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_continue_cli() {
    if !runner_enabled("continue_cli") {
        eprintln!("SKIP e2e_continue_cli (set EMBACLE_E2E_CONTINUE_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::ContinueCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::ContinueCliRunner::new(config);
    test_provider_complete(&runner).await;
    // Continue does not support streaming natively; complete_stream wraps complete
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_warp_cli() {
    if !runner_enabled("warp_cli") {
        eprintln!("SKIP e2e_warp_cli (set EMBACLE_E2E_WARP_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::WarpCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::WarpCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_kiro_cli() {
    if !runner_enabled("kiro_cli") {
        eprintln!("SKIP e2e_kiro_cli (set EMBACLE_E2E_KIRO_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::KiroCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::KiroCliRunner::new(config);
    test_provider_complete(&runner).await;
    // Kiro does not support streaming natively; complete_stream wraps complete
    test_provider_stream(&runner).await;
}

#[tokio::test]
async fn e2e_kilo_cli() {
    if !runner_enabled("kilo_cli") {
        eprintln!("SKIP e2e_kilo_cli (set EMBACLE_E2E_KILO_CLI=1)");
        return;
    }
    let path = resolve_or_skip(CliRunnerType::KiloCli);
    if !path.exists() {
        return;
    }
    let config = RunnerConfig::new(path).with_timeout(E2E_TIMEOUT);
    let runner = embacle::KiloCliRunner::new(config);
    test_provider_complete(&runner).await;
    test_provider_stream(&runner).await;
}

// ============================================================================
// Copilot Headless (ACP) tests — requires copilot-headless feature
// ============================================================================

#[cfg(feature = "copilot-headless")]
mod headless {
    use super::*;
    use embacle::CopilotHeadlessRunner;

    #[tokio::test]
    async fn e2e_copilot_headless_complete() {
        if !runner_enabled("copilot_headless") {
            eprintln!("SKIP e2e_copilot_headless_complete (set EMBACLE_E2E_COPILOT_HEADLESS=1)");
            return;
        }
        let runner = CopilotHeadlessRunner::from_env().await;
        test_provider_complete(&runner).await;
    }

    #[tokio::test]
    async fn e2e_copilot_headless_stream() {
        if !runner_enabled("copilot_headless") {
            eprintln!("SKIP e2e_copilot_headless_stream (set EMBACLE_E2E_COPILOT_HEADLESS=1)");
            return;
        }
        let runner = CopilotHeadlessRunner::from_env().await;
        test_provider_stream(&runner).await;
    }

    #[tokio::test]
    async fn e2e_copilot_headless_converse() {
        if !runner_enabled("copilot_headless") {
            eprintln!("SKIP e2e_copilot_headless_converse (set EMBACLE_E2E_COPILOT_HEADLESS=1)");
            return;
        }
        let runner = CopilotHeadlessRunner::from_env().await;

        let request = ChatRequest::new(vec![
            ChatMessage::system("You are a test bot. Follow instructions exactly."),
            ChatMessage::user("Respond with exactly: CONVERSE_OK. Nothing else."),
        ])
        .with_max_tokens(20);

        let response = runner.converse(&request).await.expect("converse() failed");

        assert!(
            !response.content.is_empty(),
            "copilot_headless converse: empty content"
        );
        eprintln!("  headless converse content: {:?}", response.content);
        eprintln!("  headless converse model:   {:?}", response.model);
        eprintln!("  headless converse usage:   {:?}", response.usage);
        eprintln!(
            "  headless converse tools:   {} observed",
            response.tool_calls.len()
        );
        for tc in &response.tool_calls {
            eprintln!("    tool: {} ({})", tc.title, tc.status);
        }
        eprintln!("  headless converse finish:  {:?}", response.finish_reason);
    }

    #[tokio::test]
    async fn e2e_copilot_headless_converse_with_tools() {
        if !runner_enabled("copilot_headless") {
            eprintln!(
                "SKIP e2e_copilot_headless_converse_with_tools (set EMBACLE_E2E_COPILOT_HEADLESS=1)"
            );
            return;
        }
        let runner = CopilotHeadlessRunner::from_env().await;

        // Ask something that should trigger tool use (file read)
        let request = ChatRequest::new(vec![ChatMessage::user(
            "Read the file Cargo.toml in the current directory and tell me the package name.",
        )])
        .with_max_tokens(100);

        let response = runner
            .converse(&request)
            .await
            .expect("converse() with tools failed");

        assert!(
            !response.content.is_empty(),
            "copilot_headless converse_with_tools: empty content"
        );
        eprintln!(
            "  headless tools content: {:?}",
            &response.content[..response.content.len().min(200)]
        );
        eprintln!(
            "  headless tools observed: {} tool calls",
            response.tool_calls.len()
        );
        for tc in &response.tool_calls {
            eprintln!("    tool: {} [{}] ({})", tc.title, tc.id, tc.status);
        }
        // We expect at least one tool call for reading Cargo.toml
        if response.tool_calls.is_empty() {
            eprintln!("  WARNING: expected tool calls but got none");
        }
    }
}

// ============================================================================
// OpenAI API Runner (requires `openai-api` feature + live API key)
// ============================================================================

/// E2E test for `OpenAiApiRunner` against a live `OpenAI`-compatible endpoint.
///
/// Enable with: `EMBACLE_E2E_OPENAI_API=1`
///
/// Required env vars (example for Groq):
///   `OPENAI_API_BASE_URL=https://api.groq.com/openai/v1`
///   `OPENAI_API_KEY=gsk_...`
///   `OPENAI_API_MODEL=llama-3.3-70b-versatile`
#[cfg(feature = "openai-api")]
mod openai_api_e2e {
    use super::*;
    use embacle::{OpenAiApiConfig, OpenAiApiRunner};

    #[tokio::test]
    async fn openai_api_complete_and_stream() {
        if !runner_enabled("openai_api") {
            eprintln!("  SKIP openai_api: set EMBACLE_E2E_OPENAI_API=1 to enable");
            return;
        }

        let config = OpenAiApiConfig::from_env();
        eprintln!(
            "  openai_api: base_url={}, model={}",
            config.base_url, config.model
        );

        let runner = OpenAiApiRunner::new(config).await;
        test_provider_complete(&runner).await;
    }
}
