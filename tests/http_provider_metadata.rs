// ABOUTME: What each HTTP provider reports about itself: name, display name, capabilities, models, redaction
// ABOUTME: Runs under the http-api feature without reaching any network
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

use embacle::http_api::{
    CohereConfig, CohereProvider, GeminiConfig, GeminiProvider, GroqConfig, GroqProvider,
    OpenRouterConfig, OpenRouterProvider,
};
use embacle::types::LlmProvider;

fn gemini() -> GeminiProvider {
    GeminiProvider::with_client(GeminiConfig::new("test-key"), reqwest::Client::new())
}

fn groq() -> GroqProvider {
    GroqProvider::with_client(GroqConfig::new("test-key"), reqwest::Client::new())
}

// ============================================================================
// GeminiProvider
// ============================================================================

#[test]
fn gemini_provider_metadata() {
    let provider = gemini();
    assert_eq!(provider.name(), "gemini");
    assert_eq!(provider.display_name(), "Google Gemini");
    assert_eq!(provider.default_model(), "gemini-flash-lite-latest");
    assert!(!provider.available_models().is_empty());
}

#[test]
fn gemini_capabilities() {
    let caps = gemini().capabilities();
    assert!(caps.supports_streaming());
    assert!(caps.supports_function_calling());
    assert!(caps.supports_vision());
    assert!(caps.supports_system_messages());
}

#[test]
fn gemini_debug_redacts_api_key() {
    let provider = GeminiProvider::with_client(
        GeminiConfig::new("super-secret-key"),
        reqwest::Client::new(),
    );
    let debug_output = format!("{provider:?}");
    assert!(!debug_output.contains("super-secret-key"));
    assert!(debug_output.contains("[REDACTED]"));
}

#[test]
fn gemini_with_custom_model() {
    let provider = GeminiProvider::with_client(
        GeminiConfig::new("key").with_model("gemini-2.5-pro"),
        reqwest::Client::new(),
    );
    assert_eq!(provider.default_model(), "gemini-2.5-pro");
    let debug_output = format!("{provider:?}");
    assert!(debug_output.contains("gemini-2.5-pro"));
}

// ============================================================================
// GroqProvider
// ============================================================================

#[test]
fn groq_provider_metadata() {
    let provider = groq();
    assert_eq!(provider.name(), "groq");
    assert_eq!(provider.display_name(), "Groq (Llama/Mixtral)");
    assert_eq!(provider.default_model(), "llama-3.3-70b-versatile");
    assert!(!provider.available_models().is_empty());
}

#[test]
fn groq_capabilities() {
    let caps = groq().capabilities();
    assert!(caps.supports_streaming());
    assert!(caps.supports_function_calling());
    assert!(caps.supports_system_messages());
    assert!(!caps.supports_vision());
}

// ============================================================================
// CohereProvider / OpenRouterProvider
// ============================================================================

#[test]
fn cohere_provider_metadata() {
    let provider =
        CohereProvider::with_client(CohereConfig::new("test-key"), reqwest::Client::new());
    assert_eq!(provider.name(), "cohere");
    assert_eq!(provider.display_name(), "Cohere (Command)");
    assert_eq!(provider.default_model(), "command-a-03-2025");
    assert!(provider.capabilities().supports_function_calling());
    assert!(!provider.capabilities().supports_vision());
}

#[test]
fn openrouter_provider_metadata() {
    let provider =
        OpenRouterProvider::with_client(OpenRouterConfig::new("test-key"), reqwest::Client::new());
    assert_eq!(provider.name(), "openrouter");
    assert_eq!(provider.display_name(), "OpenRouter");
    assert_eq!(
        provider.default_model(),
        "meta-llama/llama-3.3-70b-instruct"
    );
    assert!(provider.capabilities().supports_function_calling());
    assert!(provider
        .available_models()
        .contains(&"anthropic/claude-3.5-sonnet".to_owned()));
}
