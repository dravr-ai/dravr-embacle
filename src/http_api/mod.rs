// ABOUTME: HTTP API providers — Gemini, Cohere, Groq, OpenRouter, OpenAI-compatible — and what they share
// ABOUTME: One HTTP error mapping, one SSE parser, one retry policy, one OpenAI wire format; vendor quirks stay per provider
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # HTTP API providers (feature `http-api`)
//!
//! Providers that speak to a vendor's HTTP API directly, each implementing
//! [`LlmProvider`](crate::types::LlmProvider):
//!
//! | Provider | Type | `name()` | Env (`from_env`) |
//! |---|---|---|---|
//! | Google Gemini | [`GeminiProvider`](gemini::GeminiProvider) | `gemini` | `GEMINI_API_KEY`, `GEMINI_DEFAULT_MODEL`, `GEMINI_MAX_RETRIES`, `GEMINI_INITIAL_RETRY_DELAY_MS`, `GEMINI_MAX_RETRY_DELAY_MS` |
//! | Cohere | [`CohereProvider`](cohere::CohereProvider) | `cohere` | `COHERE_API_KEY`, `COHERE_DEFAULT_MODEL`, `COHERE_MAX_RETRIES`, `COHERE_INITIAL_RETRY_DELAY_MS`, `COHERE_MAX_RETRY_DELAY_MS` |
//! | Groq | [`GroqProvider`](groq::GroqProvider) | `groq` | `GROQ_API_KEY`, `GROQ_DEFAULT_MODEL`, `GROQ_MAX_RETRIES`, `GROQ_INITIAL_RETRY_DELAY_MS`, `GROQ_MAX_RETRY_DELAY_MS` |
//! | `OpenRouter` | [`OpenRouterProvider`](openrouter::OpenRouterProvider) | `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_DEFAULT_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_TITLE`, `OPENROUTER_MAX_RETRIES`, `OPENROUTER_INITIAL_RETRY_DELAY_MS`, `OPENROUTER_MAX_RETRY_DELAY_MS` |
//! | OpenAI-compatible (Ollama, vLLM, `LocalAI`, …) | [`OpenAiCompatibleProvider`](openai_compatible::OpenAiCompatibleProvider) | `ollama` / `vllm` / `localai` / `local` | `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, `LOCAL_LLM_API_KEY` |
//! | The `OpenAI` API (or any endpoint serving it under `/v1`) | [`OpenAiCompatibleProvider`](openai_compatible::OpenAiCompatibleProvider) over [`OpenAiCompatibleConfig::openai_api_from_env`](openai_compatible::OpenAiCompatibleConfig::openai_api_from_env) | `openai_api` | `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_API_MODEL`, `OPENAI_API_TIMEOUT_SECS` |
//!
//! What they share lives in [`client`] (HTTP client construction, the one
//! status-to-[`RunnerError`](crate::types::RunnerError) mapping, the retry
//! policy), [`sse`] (the line-buffering Server-Sent Events parser) and, for
//! Groq, `OpenRouter` and the OpenAI-compatible provider, one `OpenAI`
//! chat-completions wire format (request body, response, usage, stream
//! frames, error envelope). What they do not share — Gemini's
//! `system_instruction` hoisting and thinking-only retries, Cohere's typed
//! event envelope and its rule that a content-less message sinks the whole
//! request, `OpenRouter`'s ranking headers and 402 wording, Groq billing the
//! plain token counts, the self-hosted endpoints' "is the server running?"
//! hints — stays in each provider's own module.
//!
//! Every provider accepts a caller-owned [`reqwest::Client`] through
//! `with_client`, so a host with one connection pool can hand it to all of
//! them. Tool calling rides on [`ChatRequest::tools`](crate::types::ChatRequest::tools)
//! and comes back on [`ChatResponse::tool_calls`](crate::types::ChatResponse::tool_calls).
//!
//! No request URL is ever written to a log by this module: Gemini's API key
//! travels in the query string, so a URL in a log line is a key in a log line.

/// The `OpenAI` chat-completions wire format shared by Groq, `OpenRouter` and
/// the OpenAI-compatible provider.
mod chat_completions;
/// HTTP client construction, status-code mapping and the retry policy.
pub mod client;
/// Cohere v2 chat provider.
pub mod cohere;
mod cohere_errors;
/// Google Gemini provider.
pub mod gemini;
/// Groq provider (OpenAI-shaped API on LPU inference).
pub mod groq;
/// Any OpenAI-compatible endpoint: the `OpenAI` API, Ollama, vLLM, `LocalAI`, and others.
pub mod openai_compatible;
/// `OpenRouter` provider (one key, many upstream models).
pub mod openrouter;
/// Line-buffering Server-Sent Events parser shared by every streaming provider.
pub mod sse;

pub use cohere::{CohereConfig, CohereProvider};
pub use gemini::{GeminiConfig, GeminiProvider};
pub use groq::{GroqConfig, GroqProvider};
pub use openai_compatible::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
pub use openrouter::{OpenRouterConfig, OpenRouterProvider};
