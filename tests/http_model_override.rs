// ABOUTME: with_default_model replaces the configured model on Gemini, Groq and the local provider
// ABOUTME: A chain's second tier must resolve its own model name, not the first tier's, or its API 404s
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A fallback tier constructed with the primary's model name issues requests
//! against a model its API has never heard of — a Gemini secondary asked for
//! a Claude model 404s on every call, and the chain collapses whenever the
//! primary is unavailable. `with_default_model` is the override that gives
//! the tier its own model; these pin that it actually rewrites the field.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
#![cfg(feature = "http-api")]

use embacle::http_api::{
    GeminiConfig, GeminiProvider, GroqConfig, GroqProvider, OpenAiCompatibleConfig,
    OpenAiCompatibleProvider,
};
use embacle::types::LlmProvider;

#[test]
fn gemini_with_default_model_replaces_inherited_model() {
    let provider = GeminiProvider::with_client(
        GeminiConfig::new("test-api-key").with_model("claude-opus-4.7"),
        reqwest::Client::new(),
    );
    assert_eq!(
        provider.default_model(),
        "claude-opus-4.7",
        "baseline: the provider starts with the inherited model name"
    );

    let provider = provider.with_default_model("gemini-flash-lite-latest");
    assert_eq!(
        provider.default_model(),
        "gemini-flash-lite-latest",
        "after override, the provider must target a model name that exists on Google's API"
    );
}

#[test]
fn groq_with_default_model_replaces_inherited_model() {
    let provider =
        GroqProvider::with_client(GroqConfig::new("test-api-key"), reqwest::Client::new());
    let baseline = provider.default_model().to_owned();

    let provider = provider.with_default_model("llama-3.1-8b-instant");
    assert_eq!(
        provider.default_model(),
        "llama-3.1-8b-instant",
        "after override, the provider must surface its own model regardless of the baseline ({baseline})"
    );
}

#[test]
fn local_with_default_model_replaces_inherited_model() {
    let config = OpenAiCompatibleConfig::ollama("qwen2.5:14b-instruct");
    let provider = OpenAiCompatibleProvider::with_client(config, reqwest::Client::new());
    assert_eq!(
        provider.default_model(),
        "qwen2.5:14b-instruct",
        "baseline: ollama config starts with the seeded model"
    );

    let provider = provider.with_default_model("qwen2.5:32b-instruct");
    assert_eq!(
        provider.default_model(),
        "qwen2.5:32b-instruct",
        "after override, the local provider must use its own model"
    );
}
