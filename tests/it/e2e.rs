// ABOUTME: End-to-end integration tests against real CLI binaries
// ABOUTME: Gated by EMBACLE_E2E_<RUNNER>=1 env vars; EMBACLE_E2E_ALL=1 runs everything
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
    if env::var("EMBACLE_E2E_ALL").as_deref() == Ok("1") {
        return true;
    }
    let key = format!("EMBACLE_E2E_{}", tag.to_uppercase());
    env::var(&key).as_deref() == Ok("1")
}

/// Build a simple ping request that any LLM should handle.
fn ping_request() -> ChatRequest {
    ChatRequest::new(vec![
        ChatMessage::system("You are a test bot. Follow instructions exactly."),
        ChatMessage::user("Respond with exactly: PONG. Nothing else."),
    ])
    .with_max_tokens(20)
}

/// The streamed turn's instruction, worded as a verbatim reply.
///
/// Not "Count from 1 to 3": the Copilot CLI lists its skills in the system
/// prompt, and a 3B model once read "Count" as one and answered "I can't find
/// the \"counting\" skill". Every "Respond with exactly" prompt here held on
/// the same runners.
const STREAM_PROMPT: &str = "Respond with exactly these three lines and nothing else:\n1\n2\n3";

/// Build a streaming request.
fn stream_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user(STREAM_PROMPT)]).with_max_tokens(30)
}

/// Standard timeout for E2E tests (CLI tools can be slow on first invocation).
const E2E_TIMEOUT: Duration = Duration::from_mins(5);

/// Resolve a binary or skip.
fn resolve_or_skip(runner_type: CliRunnerType) -> PathBuf {
    let env_override = env::var(runner_type.env_override_key()).ok();
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

/// Whether `haystack` carries every needle, in the order given.
fn contains_in_order(haystack: &str, needles: &[&str]) -> bool {
    let mut rest = haystack;
    for needle in needles {
        match rest.find(needle) {
            Some(at) => rest = &rest[at + needle.len()..],
            None => return false,
        }
    }
    true
}

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

    eprintln!("  {name} complete: {:?}", response.content);
    assert!(
        response.content.to_uppercase().contains("PONG"),
        "{name}: the model answered the ping: {:?}",
        response.content
    );
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
    eprintln!("  {name} stream: {chunk_count} chunks, content: {full_content:?}");
    assert!(
        contains_in_order(&full_content, &["1", "2", "3"]),
        "{name}: the stream spelled the count: {full_content:?}"
    );
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
    let runner = embacle::CopilotRunner::new(config);
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
        let runner = CopilotHeadlessRunner::from_env();
        test_provider_complete(&runner).await;
    }

    #[tokio::test]
    async fn e2e_copilot_headless_stream() {
        if !runner_enabled("copilot_headless") {
            eprintln!("SKIP e2e_copilot_headless_stream (set EMBACLE_E2E_COPILOT_HEADLESS=1)");
            return;
        }
        let runner = CopilotHeadlessRunner::from_env();
        test_provider_stream(&runner).await;
    }

    #[tokio::test]
    async fn e2e_copilot_headless_converse() {
        if !runner_enabled("copilot_headless") {
            eprintln!("SKIP e2e_copilot_headless_converse (set EMBACLE_E2E_COPILOT_HEADLESS=1)");
            return;
        }
        let runner = CopilotHeadlessRunner::from_env();

        let request = ChatRequest::new(vec![
            ChatMessage::system("You are a test bot. Follow instructions exactly."),
            ChatMessage::user("Respond with exactly: CONVERSE_OK. Nothing else."),
        ])
        .with_max_tokens(20);

        let response = runner
            .converse(&request)
            .await
            .unwrap_or_else(|e| panic!("converse() failed: {e}"));

        eprintln!("  headless converse content: {:?}", response.content);
        assert!(
            response.content.contains("CONVERSE_OK"),
            "copilot_headless converse: the model answered as told: {:?}",
            response.content
        );
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
        let runner = CopilotHeadlessRunner::from_env();

        // Ask something that should trigger tool use (file read)
        let request = ChatRequest::new(vec![ChatMessage::user(
            "Read the file Cargo.toml in the current directory and tell me the package name.",
        )])
        .with_max_tokens(100);

        let response = runner
            .converse(&request)
            .await
            .unwrap_or_else(|e| panic!("converse() with tools failed: {e}"));

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
        assert!(
            !response.tool_calls.is_empty(),
            "reading a file is a tool call, and it is observed: {:?}",
            response.content
        );
    }
}

// ============================================================================
// OpenAI API (requires `http-api` feature + live API key)
// ============================================================================

/// E2E test for the `OpenAI` API configuration of `OpenAiCompatibleProvider`
/// against a live `OpenAI`-compatible endpoint.
///
/// Enable with: `EMBACLE_E2E_OPENAI_API=1`
///
/// Required env vars (example for Groq; the requests go to the base URL's `/v1`):
///   `OPENAI_API_BASE_URL=https://api.groq.com/openai`
///   `OPENAI_API_KEY=gsk_...`
///   `OPENAI_API_MODEL=llama-3.3-70b-versatile`
#[cfg(feature = "http-api")]
mod openai_api_e2e {
    use super::*;
    use embacle::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};

    #[tokio::test]
    async fn openai_api_complete_and_stream() {
        if !runner_enabled("openai_api") {
            eprintln!("  SKIP openai_api: set EMBACLE_E2E_OPENAI_API=1 to enable");
            return;
        }

        let config = OpenAiCompatibleConfig::openai_api_from_env();
        eprintln!(
            "  openai_api: base_url={}, model={}",
            config.base_url, config.default_model
        );

        let provider = OpenAiCompatibleProvider::new(config)
            .expect("the HTTP client builds")
            .with_discovered_models()
            .await;
        test_provider_complete(&provider).await;
    }
}

// ============================================================================
// Copilot SDK tests — requires copilot-sdk feature, a runtime pair on disk
// (COPILOT_RUNTIME_PATH) and Copilot auth (COPILOT_GITHUB_TOKEN)
// ============================================================================

#[cfg(feature = "copilot-sdk")]
mod sdk {
    use super::*;
    use embacle::types::ErrorKind;
    use embacle::{CopilotSdkRunner, HeadlessStreamEvent, HeadlessTurnProvider};

    const CODEWORD: &str = "DRAVR-COACH-7741";

    /// The model the live turns run on: `COPILOT_SDK_MODEL`, which a lane
    /// routing to its own provider sets to a model that provider serves, or a
    /// cheap catalogued Copilot model.
    fn model() -> String {
        env::var("COPILOT_SDK_MODEL").unwrap_or_else(|_| "claude-haiku-4.5".to_owned())
    }

    /// An id no catalogue and no provider serves.
    const UNKNOWN_MODEL: &str = "embacle-e2e-no-such-model";

    fn skipped(test: &str) -> bool {
        if runner_enabled("copilot_sdk") {
            return false;
        }
        eprintln!("SKIP {test} (set EMBACLE_E2E_COPILOT_SDK=1 and COPILOT_RUNTIME_PATH)");
        true
    }

    fn request(system: &str, user: &str) -> ChatRequest {
        let mut request =
            ChatRequest::new(vec![ChatMessage::system(system), ChatMessage::user(user)])
                .with_max_tokens(60);
        request.model = Some(model());
        request
    }

    #[tokio::test]
    async fn e2e_copilot_sdk_complete() {
        if skipped("e2e_copilot_sdk_complete") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();
        test_provider_complete(&runner).await;
    }

    #[tokio::test]
    async fn e2e_copilot_sdk_stream() {
        if skipped("e2e_copilot_sdk_stream") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();
        test_provider_stream(&runner).await;
    }

    /// The system prompt travels in the runtime's own system slot and
    /// replaces the Copilot CLI persona: the model answers from these
    /// instructions, and when asked to quote them it quotes ours.
    #[tokio::test]
    async fn e2e_copilot_sdk_system_prompt_replaces_the_copilot_persona() {
        if skipped("e2e_copilot_sdk_system_prompt_replaces_the_copilot_persona") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();

        let codeword_turn = request(
            &format!(
                "You are Dravr's coach. Your codeword is {CODEWORD}. When asked for \
                 your codeword, reply with exactly the codeword and nothing else."
            ),
            "What is your codeword?",
        );
        let response = runner
            .converse(&codeword_turn)
            .await
            .unwrap_or_else(|e| panic!("converse() failed: {e}"));
        assert!(
            response.content.contains(CODEWORD),
            "the system prompt reached the model: {:?}",
            response.content
        );

        let quote_turn = request(
            "You are Dravr's coach and nothing else. When asked to quote your \
             instructions, reply with their first sentence, verbatim.",
            "Quote the first sentence of your instructions.",
        );
        let response = runner
            .converse(&quote_turn)
            .await
            .unwrap_or_else(|e| panic!("converse() failed: {e}"));
        let quoted = response.content.to_lowercase();
        assert!(
            quoted.contains("dravr"),
            "the model quotes our instructions: {:?}",
            response.content
        );
        assert!(
            !quoted.contains("github copilot"),
            "the Copilot CLI persona was displaced: {:?}",
            response.content
        );
    }

    /// The served model and the runtime's cache counts come back on every turn.
    #[tokio::test]
    async fn e2e_copilot_sdk_reports_the_served_model_and_cache_counts() {
        if skipped("e2e_copilot_sdk_reports_the_served_model_and_cache_counts") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();
        let turn = request(
            "You are a test bot. Follow instructions exactly.",
            "Respond with exactly: PONG. Nothing else.",
        );
        let response = runner
            .converse(&turn)
            .await
            .unwrap_or_else(|e| panic!("converse() failed: {e}"));
        assert_eq!(
            response.model,
            model(),
            "the model that served is the one asked for"
        );
        let usage = response.usage.expect("assistant.usage was reported");
        assert!(usage.prompt_tokens > 0, "prompt tokens counted: {usage:?}");
        assert!(
            usage.completion_tokens > 0,
            "completion tokens counted: {usage:?}"
        );
        assert!(
            usage.cached_read_tokens.is_some() && usage.cached_write_tokens.is_some(),
            "cache counts are carried, not dropped: {usage:?}"
        );
        assert!(
            response.tool_calls.is_empty(),
            "no tools were offered, none ran"
        );
    }

    /// A model nothing serves fails loudly instead of being served by
    /// whatever the runtime substitutes — refused by the catalogue check under
    /// Copilot's routing, and by the endpoint itself under a provider.
    #[tokio::test]
    async fn e2e_copilot_sdk_unknown_model_fails_loudly() {
        if skipped("e2e_copilot_sdk_unknown_model_fails_loudly") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();
        let mut turn = request("You are a test bot.", "Respond with exactly: PONG.");
        turn.model = Some(UNKNOWN_MODEL.to_owned());
        let err = runner
            .converse(&turn)
            .await
            .expect_err("an unknown model id is refused");
        assert_eq!(err.kind, ErrorKind::ModelUnavailable, "{err}");
        assert!(err.to_string().contains(UNKNOWN_MODEL), "{err}");
    }

    /// Streaming delivers text deltas and ends with the aggregated response.
    #[tokio::test]
    async fn e2e_copilot_sdk_converse_stream_ends_with_done() {
        if skipped("e2e_copilot_sdk_converse_stream_ends_with_done") {
            return;
        }
        let runner = CopilotSdkRunner::from_env();
        let turn = request(
            "You are a test bot. Follow instructions exactly.",
            STREAM_PROMPT,
        );
        let mut stream = runner
            .converse_stream(&turn)
            .await
            .unwrap_or_else(|e| panic!("converse_stream() failed: {e}"));

        let mut deltas = 0u32;
        let mut streamed = String::new();
        let mut done = None;
        while let Some(event) = stream.next().await {
            match event.unwrap_or_else(|e| panic!("stream error: {e}")) {
                HeadlessStreamEvent::TextDelta(delta) => {
                    deltas += 1;
                    streamed.push_str(&delta);
                }
                HeadlessStreamEvent::ToolCall(call) => panic!("no tool was offered: {call:?}"),
                HeadlessStreamEvent::Done(response) => done = Some(response),
            }
        }
        let done = done.expect("the stream ends with Done");
        assert!(deltas > 0, "text arrived incrementally");
        assert_eq!(
            done.content, streamed,
            "Done carries the text the deltas spelled"
        );
        assert_eq!(done.model, model());
        assert!(done.usage.is_some(), "usage rides on Done");
    }
}
