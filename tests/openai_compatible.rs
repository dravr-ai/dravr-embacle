// ABOUTME: The OpenAI-compatible provider's config shapes, env parsing and name reporting
// ABOUTME: Live cases against a local Ollama run only when RUN_LOCAL_LLM_TESTS=1 is set
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
#![cfg(feature = "http-api")]

use std::env;
use std::sync::Mutex;

use tokio_stream::StreamExt;

use embacle::http_api::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use embacle::types::{ChatMessage, ChatRequest, LlmCapabilities, LlmProvider, ToolDefinition};

/// Live-Ollama tests gate on `RUN_LOCAL_LLM_TESTS=1` so they skip on machines
/// without a local Ollama server.
macro_rules! require_local_llm {
    () => {
        if env::var("RUN_LOCAL_LLM_TESTS").is_err() {
            eprintln!("skipping: set RUN_LOCAL_LLM_TESTS=1 (and run `ollama serve`) to enable live Ollama integration tests");
            return;
        }
    };
}

// =============================================================================
// OpenAiCompatibleConfig
// =============================================================================

#[test]
fn config_ollama() {
    let config = OpenAiCompatibleConfig::ollama("qwen2.5:7b");

    assert_eq!(config.base_url, "http://localhost:11434/v1");
    assert!(config.api_key.is_none());
    assert_eq!(config.default_model, "qwen2.5:7b");
    assert_eq!(config.provider_name, "ollama");
    assert_eq!(config.display_name, "Ollama (Local)");
    assert!(config.capabilities.supports_streaming());
    assert!(config.capabilities.supports_function_calling());
    assert!(config.capabilities.supports_system_messages());
    assert!(!config.capabilities.supports_vision());
}

#[test]
fn config_vllm() {
    let config = OpenAiCompatibleConfig::vllm("meta-llama/Llama-3.1-8B");

    assert_eq!(config.base_url, "http://localhost:8000/v1");
    assert!(config.api_key.is_none());
    assert_eq!(config.default_model, "meta-llama/Llama-3.1-8B");
    assert_eq!(config.provider_name, "vllm");
    assert_eq!(config.display_name, "vLLM (Local)");
    assert!(config.capabilities.supports_json_mode());
    assert!(!config.capabilities.supports_vision());
}

#[test]
fn config_local_ai() {
    let config = OpenAiCompatibleConfig::local_ai("mistral-7b");

    assert_eq!(config.base_url, "http://localhost:8080/v1");
    assert!(config.api_key.is_none());
    assert_eq!(config.default_model, "mistral-7b");
    assert_eq!(config.provider_name, "localai");
    assert_eq!(config.display_name, "LocalAI");
    assert!(config.capabilities.supports_function_calling());
}

#[test]
fn config_default() {
    let config = OpenAiCompatibleConfig::default();

    assert_eq!(config.base_url, "http://localhost:11434/v1");
    assert!(config.api_key.is_none());
    assert_eq!(config.default_model, "qwen2.5:14b-instruct");
    assert_eq!(config.provider_name, "local");
    assert_eq!(config.display_name, "Local LLM");
}

#[test]
fn config_capabilities_differ() {
    let ollama = OpenAiCompatibleConfig::ollama("test");
    let vllm = OpenAiCompatibleConfig::vllm("test");

    // vLLM has JSON mode, Ollama does not
    assert!(!ollama.capabilities.supports_json_mode());
    assert!(vllm.capabilities.supports_json_mode());
    assert_eq!(
        ollama.capabilities,
        LlmCapabilities::STREAMING
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::SYSTEM_MESSAGES
    );
    assert_eq!(
        vllm.capabilities,
        LlmCapabilities::STREAMING
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::JSON_MODE
    );
}

// =============================================================================
// name() per config
// =============================================================================

fn named(provider_name: &str) -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::with_client(
        OpenAiCompatibleConfig {
            provider_name: provider_name.to_owned(),
            display_name: provider_name.to_owned(),
            ..OpenAiCompatibleConfig::default()
        },
        reqwest::Client::new(),
    )
}

#[test]
fn name_follows_the_configured_provider_name() {
    assert_eq!(named("ollama").name(), "ollama");
    assert_eq!(named("vllm").name(), "vllm");
    assert_eq!(named("localai").name(), "localai");
    // Anything the config does not name explicitly reports as "local".
    assert_eq!(named("some-self-hosted-endpoint").name(), "local");
    assert_eq!(
        named("ollama").capabilities(),
        OpenAiCompatibleConfig::default().capabilities
    );
}

#[test]
fn provider_new_builds_for_every_preset() {
    assert!(OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::ollama("qwen2.5:7b")).is_ok());
    assert!(OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::vllm("llama3.1:8b")).is_ok());
    assert!(OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::local_ai("mistral-7b")).is_ok());
}

// =============================================================================
// Environment parsing
// =============================================================================

mod env_tests {
    use super::*;

    // Mutex to ensure env var tests don't interfere with each other
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn with_clean_env<F, T>(f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let _guard = ENV_MUTEX.lock().unwrap();

        let saved: Vec<(&str, Option<String>)> =
            ["LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY"]
                .into_iter()
                .map(|name| (name, env::var(name).ok()))
                .collect();
        for (name, _) in &saved {
            env::remove_var(name);
        }

        let result = f();

        for (name, value) in saved {
            match value {
                Some(v) => env::set_var(name, v),
                None => env::remove_var(name),
            }
        }
        result
    }

    #[test]
    fn from_env_defaults() {
        with_clean_env(|| {
            let config = OpenAiCompatibleConfig::from_env();
            assert_eq!(config.base_url, "http://localhost:11434/v1");
            assert_eq!(config.default_model, "qwen2.5:14b-instruct");
            assert_eq!(config.provider_name, "ollama");
            assert!(config.api_key.is_none());
        });
    }

    #[test]
    fn from_env_custom_url_is_local() {
        with_clean_env(|| {
            env::set_var("LOCAL_LLM_BASE_URL", "http://custom-host:9999/v1");
            env::set_var("LOCAL_LLM_MODEL", "custom-model");

            let config = OpenAiCompatibleConfig::from_env();
            assert_eq!(config.base_url, "http://custom-host:9999/v1");
            assert_eq!(config.default_model, "custom-model");
            assert_eq!(config.provider_name, "local");
            assert_eq!(config.display_name, "Local LLM");
        });
    }

    #[test]
    fn from_env_with_api_key() {
        with_clean_env(|| {
            env::set_var("LOCAL_LLM_API_KEY", "test-api-key-12345");
            let config = OpenAiCompatibleConfig::from_env();
            assert_eq!(config.api_key.as_deref(), Some("test-api-key-12345"));
        });
    }

    #[test]
    fn from_env_empty_api_key_ignored() {
        with_clean_env(|| {
            env::set_var("LOCAL_LLM_API_KEY", "");
            let config = OpenAiCompatibleConfig::from_env();
            assert!(config.api_key.is_none());
        });
    }

    #[test]
    fn from_env_port_detection() {
        with_clean_env(|| {
            for (url, name) in [
                ("http://localhost:11434/v1", "ollama"),
                ("http://localhost:8000/v1", "vllm"),
                ("http://localhost:8080/v1", "localai"),
            ] {
                env::set_var("LOCAL_LLM_BASE_URL", url);
                assert_eq!(
                    OpenAiCompatibleConfig::from_env().provider_name,
                    name,
                    "{url}"
                );
            }
        });
    }
}

// =============================================================================
// Live cases (require a running local server)
// =============================================================================

#[tokio::test]
async fn live_health_check() {
    require_local_llm!();

    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::ollama("qwen2.5:7b")).unwrap();
    let result = provider.health_check().await;
    assert!(result.is_ok(), "Health check should pass: {result:?}");
    assert!(result.unwrap());
}

#[tokio::test]
async fn live_complete() {
    require_local_llm!();

    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::ollama("qwen2.5:7b")).unwrap();
    let request = ChatRequest::new(vec![ChatMessage::user("Say hello in exactly 3 words.")]);

    let response = provider.complete(&request).await;
    assert!(response.is_ok(), "Completion should succeed: {response:?}");
    assert!(!response.unwrap().content.is_empty());
}

#[tokio::test]
async fn live_complete_with_tools() {
    require_local_llm!();

    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::ollama("qwen2.5:14b-instruct"))
            .unwrap();
    let request = ChatRequest::new(vec![ChatMessage::user(
        "What's the weather like in Paris? Use the get_weather tool.",
    )])
    .with_tools(vec![ToolDefinition {
        name: "get_weather".to_owned(),
        description: "Get current weather for a location".to_owned(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": { "location": { "type": "string", "description": "City name" } },
            "required": ["location"]
        })),
    }]);

    let response = provider.complete(&request).await;
    assert!(
        response.is_ok(),
        "Tool completion should succeed: {response:?}"
    );
    let response = response.unwrap();
    // Either a tool call or a text response explaining tools aren't available
    assert!(response.tool_calls.is_some() || !response.content.is_empty());
}

#[tokio::test]
async fn live_streaming() {
    require_local_llm!();

    let provider =
        OpenAiCompatibleProvider::new(OpenAiCompatibleConfig::ollama("qwen2.5:7b")).unwrap();
    let request = ChatRequest::new(vec![ChatMessage::user("Count from 1 to 5.")]);

    let mut stream = provider
        .complete_stream(&request)
        .await
        .expect("Stream should start successfully");
    let mut chunks_received = 0;
    let mut full_content = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.expect("Chunk should be valid");
        full_content.push_str(&chunk.delta);
        chunks_received += 1;
    }

    assert!(chunks_received > 0, "Should receive at least one chunk");
    assert!(!full_content.is_empty(), "Should have content");
}
