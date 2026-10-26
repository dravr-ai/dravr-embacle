// ABOUTME: Anthropic provider over the Messages API — Claude Haiku/Sonnet/Opus 5.5, streaming, tool use, prompt caching
// ABOUTME: Hoists system to cached top-level blocks, pairs tool_use with merged tool_result turns, maps a spent credit to a quota refusal
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Anthropic Provider
//!
//! [`LlmProvider`] over Anthropic's Messages API (`POST /v1/messages`).
//!
//! ## Configuration
//!
//! [`AnthropicConfig::from_env`] reads:
//!
//! - `ANTHROPIC_API_KEY`: required, sent as `x-api-key`; redacted in `Debug`
//!   and never part of a log line or an error message
//! - `ANTHROPIC_DEFAULT_MODEL`: default `claude-haiku-5-5`
//! - `ANTHROPIC_BASE_URL`: default `https://api.anthropic.com`
//! - `ANTHROPIC_MAX_TOKENS`: default `16000`, used when the request names none
//! - `ANTHROPIC_EFFORT`: `low` | `medium` | `high` | `xhigh` | `max`, sent as
//!   `output_config.effort`; unset leaves the model's default
//! - `ANTHROPIC_THINKING`: `adaptive` | `disabled`; unset sends no `thinking`
//!   field (see [Thinking](#thinking))
//! - `ANTHROPIC_MAX_RETRIES`, `ANTHROPIC_INITIAL_RETRY_DELAY_MS`,
//!   `ANTHROPIC_MAX_RETRY_DELAY_MS`: retry tuning (defaults 3 / 500 / 5000)
//!
//! [`AnthropicConfig::pool_from_env`] reads the numbered keys
//! `ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY_2`, `ANTHROPIC_API_KEY_3`, … up to
//! the first unset or blank one, and returns one configuration per key — one
//! per Anthropic organization, each with its own prepaid credit — for
//! [`provider_pool`](crate::pool::provider_pool).
//!
//! ## Request mapping
//!
//! - Every `System` message becomes one block of the top-level `system`
//!   array, in order.
//! - An assistant turn's tool calls become `tool_use` blocks after its text;
//!   a `Tool` message becomes a `tool_result` block in a `user` turn.
//!   Consecutive messages of one role merge into one turn, so parallel tool
//!   results travel in one `user` message, `tool_result` blocks first.
//! - The API rejects a `tool_use` without a `tool_result` in the next user
//!   turn, and a `tool_result` without its `tool_use` in the assistant turn
//!   before it. Once the turns are built, every such unpaired block — on
//!   either side — travels as plain text instead, so a lost or mismatched
//!   result never leaves the other half dangling. Leading assistant turns are
//!   dropped: the first message must be the user's. A trailing assistant
//!   turn is sent as given: it is a prefill, which Haiku 5.5 and the other
//!   current models refuse with a 400 (`InvalidRequest`, the request's fault).
//! - The API rejects empty text blocks, so empty content is never sent.
//! - A forced `tool_choice` (`Required`, `Specific`) is sent as `any` /
//!   `tool`, except to the models that reject it (Opus 5.5, Sonnet 5.5,
//!   Fable 5.1, Mythos 5.1), which get `auto`.
//!
//! ## Sampling parameters
//!
//! Every current model (Haiku 5.5, Sonnet 5 / 5.5, Opus 4.7 onward, Fable,
//! Mythos) answers a non-default `temperature`, `top_p` or `top_k` with a
//! 400 — Haiku 5.5 even refuses both sent together at their defaults. They
//! are therefore sent only to the legacy models known to accept them
//! (Claude 3, Haiku 4.x, Sonnet 4.x, Opus 4.0 through 4.6), and only one of
//! the two: Claude 4 models refuse `temperature` and `top_p` together. For
//! any other model they are left off the wire, and [`LlmCapabilities`] does
//! not advertise them.
//!
//! ## Thinking
//!
//! Haiku 5.5, Sonnet 5.5 and Opus 5.5 think adaptively when the request sets
//! no `thinking`. A response can then open with `thinking` blocks, which
//! this provider reads past (content is read by block type, never by
//! position).
//!
//! The crate's conversation model ([`ChatMessage`]) is provider-neutral text
//! plus tool calls, so a tool continuation replays each assistant turn as its
//! `text` and `tool_use` blocks only, without the signed `thinking` blocks
//! that preceded them. This departs from Anthropic's documented contract,
//! which says to pass thinking blocks back unmodified with tool results. The
//! same documentation lists that rule among the changes that fail no request,
//! and its recovery for a broken history is exactly this shape ("strip every
//! `thinking` and `redacted_thinking` block from the history (each turn's
//! `text` and `tool_use` blocks stay)"), so the request is accepted — and a
//! history that replays no thinking block can never fail the preserved-thinking
//! check. Anthropic calls that recovery a one-time measure, not a steady state,
//! because the model answers without the reasoning those blocks carried; this
//! provider makes that trade on every hop of a tool loop, so the model
//! re-derives its reasoning after each tool result. The replayed bytes are the
//! same on every hop, so the prompt cache is unaffected.
//!
//! The default therefore leaves `thinking` unset, so the model's own default
//! applies. `disabled` is not a safe family-wide default: Opus 5.5 and
//! Sonnet 5.5 reject it at every effort, Haiku 5.5 at `xhigh` and `max`.
//! Lower `ANTHROPIC_EFFORT` to think less — Anthropic's guidance for Haiku
//! 5.5 is "use effort as the lever, not `disabled`". Thinking counts toward
//! `max_tokens`, which is why the default cap is generous.
//!
//! A request can name any model, so the configured `thinking` and `effort`
//! are gated per model on the wire, as the sampling parameters are: a mode
//! the model refuses is left off (`disabled` on Opus 5.5, Sonnet 5.5 and
//! Fable 5.x, and on Haiku 5.5 / Opus 5 at `xhigh` or `max`; `adaptive` and
//! `effort` on the Claude 3, Haiku 4.x, Sonnet 4.0–4.5 and Opus 4.0–4.1
//! models that predate them, `adaptive` on Opus 4.5), so the model's own
//! default applies instead of a 400.
//!
//! ## Prompt caching
//!
//! Three of the four cache breakpoints are used, in render order: the last
//! tool definition (the tools prefix), the last system block (tools +
//! system), and the request's top-level `cache_control`, which places an
//! automatic breakpoint on the last cacheable block and moves it forward as
//! a conversation or a tool loop grows. Anthropic documents explicit and
//! automatic breakpoints composing in one request, and this pairing as the
//! robust one for agent loops. The body is byte-deterministic: tools,
//! system blocks and messages are serialized in the caller's order from
//! structs, never from a map with random order. A prefix under the model's
//! minimum (512 tokens on the 5.5 family) is simply not cached.
//!
//! ## Responses
//!
//! - Text blocks are concatenated into `content`; `tool_use` blocks become
//!   [`ChatResponse::tool_calls`].
//! - `stop_reason` maps onto the OpenAI vocabulary the rest of the crate
//!   speaks: `end_turn` / `stop_sequence` → `stop`, `max_tokens` /
//!   `model_context_window_exceeded` → `length`, `tool_use` → `tool_calls`,
//!   `refusal` → `refusal`, anything else passes through.
//! - A safety-classifier decline is an HTTP 200 with `stop_reason: "refusal"`.
//!   It can fire before any output or after partial output, and Anthropic's
//!   guidance is to discard the partial output rather than treat it as
//!   complete. The response is therefore returned empty — no text, no tool
//!   calls — with `finish_reason: "refusal"` and its usage, and logged at
//!   `warn` with the length discarded. It is an empty completion, which a
//!   [`FallbackProvider`](crate::fallback::FallbackProvider) under
//!   [`ResponsePolicy::strict`](crate::fallback::ResponsePolicy::strict)
//!   moves past to the next tier.
//! - Usage is normalised to the gross prompt convention: Anthropic's
//!   `input_tokens` is the fresh remainder only, so the prompt count is
//!   `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`,
//!   with the two cache counts carried on [`TokenUsage`] for pricing.
//!
//! ## Streaming
//!
//! `complete_stream` forwards `text_delta` events as chunks and closes with
//! the mapped `stop_reason` of `message_delta`. Thinking and
//! `input_json_delta` events carry nothing a [`StreamChunk`] can hold.
//!
//! `complete_stream_with_tools` streams a turn that may call tools: it
//! forwards `text_delta` events as [`ToolStreamEvent::TextDelta`] and, at
//! `message_stop`, yields one [`ToolStreamEvent::Done`] with the response
//! `complete()` returns for the same request. The `tool_use` blocks are
//! assembled from `content_block_start` (id, name) and their
//! `input_json_delta` fragments, and the usage from `message_start` (prompt
//! and cache counts) overlaid with `message_delta` (output), then mapped by
//! the same code as a non-streaming body — refusal discard and gross usage
//! included. A stream that closes before `message_stop`, or whose tool input
//! is not valid JSON, ends with an error instead of a `Done`, as a truncated
//! non-streaming body would.
//!
//! On either stream a mid-stream `error` event ends the stream with an
//! error. A stream cannot take back the deltas it already delivered, so a
//! mid-stream refusal closes with `finish_reason: "refusal"`: a streaming
//! consumer must check the final reason and discard what it showed when it
//! reads `refusal`.
//!
//! ## Errors
//!
//! - 402 / `billing_error` (prepaid credit exhausted) → `RateLimit`: a quota
//!   refusal, never retried in place; a fallback chain cools the tier down
//!   and the next account or provider answers.
//! - 429 / `rate_limit_error` → `RateLimit`, carrying the `retry-after`
//!   header's wait in its message; like a spent credit, never retried in
//!   place, so a pool moves to its next account on the first refusal.
//! - 529 `overloaded_error` → `ExternalService` (transient), retried in place
//!   with the configured backoff. Overload is Anthropic-wide, not per
//!   organization, so in a pool every account repeats that cycle before the
//!   chain leaves the provider: the worst case is N accounts ×
//!   (`max_retries` + 1) requests and N × the backoff sum. Lower
//!   `ANTHROPIC_MAX_RETRIES` for a large pool.
//! - 413 / `request_too_large`, or a 400 saying the prompt is too long →
//!   `ContextLength`.
//! - 404 `not_found_error` → `ModelUnavailable`.
//! - Everything else through [`client::map_http_error`].
//!
//! ## Example
//!
//! ```rust,no_run
//! use embacle::http_api::anthropic::{AnthropicConfig, AnthropicProvider};
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//!
//! # async fn example() -> Result<(), embacle::types::RunnerError> {
//! let provider = AnthropicProvider::new(AnthropicConfig::from_env()?)?;
//! let request = ChatRequest::new(vec![ChatMessage::user("What is machine learning?")]);
//! let response = provider.complete(&request).await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, RETRY_AFTER};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_stream::Stream;
use tracing::{debug, instrument, warn};

use super::client::{self, with_retries, AttemptError, HttpRetryConfig, DEFAULT_TIMEOUT_SECS};
use super::sse::{create_sse_stream, SseEvent, SseParser};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    MessageRole, RunnerError, StreamChunk, TokenUsage, ToolCallRequest, ToolChoice, ToolDefinition,
    ToolStream, ToolStreamEvent,
};

/// The name this provider reports, and the price-table key its usage bills under
const PROVIDER_NAME: &str = "anthropic";

/// Environment variable for the API key; `_2`, `_3`, … name further pool accounts
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Environment variable for the default model
const ANTHROPIC_DEFAULT_MODEL_ENV: &str = "ANTHROPIC_DEFAULT_MODEL";

/// Environment variable for the API base URL
const ANTHROPIC_BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";

/// Environment variable for the default `max_tokens`
const ANTHROPIC_MAX_TOKENS_ENV: &str = "ANTHROPIC_MAX_TOKENS";

/// Environment variable for `output_config.effort`
const ANTHROPIC_EFFORT_ENV: &str = "ANTHROPIC_EFFORT";

/// Environment variable for the `thinking` mode
const ANTHROPIC_THINKING_ENV: &str = "ANTHROPIC_THINKING";

/// Prefix of the retry-tuning environment variables
const ANTHROPIC_ENV_PREFIX: &str = "ANTHROPIC";

/// Default model: the current Haiku
const DEFAULT_MODEL: &str = "claude-haiku-5-5";

/// Default API base URL
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Default `max_tokens` when the request names none. Thinking counts toward
/// it, so it leaves room for reasoning before the answer.
const DEFAULT_MAX_TOKENS: u32 = 16_000;

/// The `anthropic-version` header every request carries
const API_VERSION: &str = "2023-06-01";

/// Available Claude models
const AVAILABLE_MODELS: &[&str] = &["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"];

/// Model-id prefixes that accept `temperature` / `top_p`. Every other model —
/// Haiku 5.5, Sonnet 5 and 5.5, Opus 4.7 onward, Fable, Mythos — answers a
/// non-default sampling parameter with a 400.
const SAMPLING_MODEL_PREFIXES: &[&str] = &[
    "claude-3",
    "claude-haiku-4",
    "claude-sonnet-4",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2",
    "claude-opus-4-5",
    "claude-opus-4-6",
];

/// Model-id prefixes that reject a forced `tool_choice` (`any` / `tool`) with
/// a 400; `auto` and `none` are unaffected.
const FORCED_TOOL_CHOICE_REJECTED: &[&str] = &[
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-mythos-5-1",
];

/// Model-id prefixes that predate `output_config.effort` and adaptive
/// thinking: they think only on a manual `budget_tokens`, and answer `effort`
/// or `{"type": "adaptive"}` with a 400. (`claude-sonnet-4-2` is the dated
/// `claude-sonnet-4-20250514` id.)
const PRE_EFFORT_MODEL_PREFIXES: &[&str] = &[
    "claude-3",
    "claude-haiku-4",
    "claude-sonnet-4-0",
    "claude-sonnet-4-2",
    "claude-sonnet-4-5",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2",
];

/// Model-id prefixes that take `effort` but not adaptive thinking
const PRE_ADAPTIVE_MODEL_PREFIXES: &[&str] = &["claude-opus-4-5"];

/// Model-id prefixes that answer `{"type": "disabled"}` with a 400 at every
/// effort level
const THINKING_ALWAYS_ON: &[&str] = &["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5"];

/// Model-id prefixes that accept `{"type": "disabled"}` only at effort `high`
/// or below
const THINKING_OFF_UP_TO_HIGH: &[&str] = &["claude-haiku-5-5", "claude-opus-5"];

/// The phrase a 400 carries when the prompt exceeds the context window
const PROMPT_TOO_LONG: &str = "prompt is too long";

// ============================================================================
// Configuration
// ============================================================================

/// How much the model thinks, sent as `output_config.effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicEffort {
    /// Cheapest and fastest: chat, short tool tasks, high-volume requests
    Low,
    /// The default on Haiku 5.5 and Opus 5.5
    Medium,
    /// Knowledge work, longer agent tasks, strict instruction following
    High,
    /// Extra-high: longer thinking and replies
    XHigh,
    /// The most thinking the model does
    Max,
}

impl AnthropicEffort {
    /// The wire value
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// Parse a wire value, case-insensitively
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// The `thinking` mode sent with every request, when one is configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinking {
    /// `{"type": "adaptive"}`: the model decides how much to think
    Adaptive,
    /// `{"type": "disabled"}`: accepted by Haiku 5.5 at effort `high` or
    /// below; rejected by Opus 5.5 and Sonnet 5.5
    Disabled,
}

impl AnthropicThinking {
    /// Parse a wire value, case-insensitively
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adaptive" => Some(Self::Adaptive),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }
}

/// Configuration for the Anthropic provider.
#[derive(Clone)]
pub struct AnthropicConfig {
    /// API key for one Anthropic organization. Redacted in `Debug`.
    pub api_key: String,
    /// Model used when the request names none
    pub model: String,
    /// API base URL, without the `/v1` path
    pub base_url: String,
    /// `max_tokens` used when the request names none
    pub max_tokens: u32,
    /// `output_config.effort`; `None` leaves the model's default
    pub effort: Option<AnthropicEffort>,
    /// `thinking`; `None` sends no field, so the model's default applies
    pub thinking: Option<AnthropicThinking>,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
    /// Retry policy for transient failures (502, 503, 529, network errors); a
    /// 429 or 402 leaves the tier at once, never retried in place
    pub retry: HttpRetryConfig,
}

impl AnthropicConfig {
    /// A configuration with the given key and every other field at its default.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            max_tokens: DEFAULT_MAX_TOKENS,
            effort: None,
            thinking: None,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            retry: HttpRetryConfig::default(),
        }
    }

    /// Read the configuration from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when
    /// `ANTHROPIC_API_KEY` is unset or blank, or when `ANTHROPIC_MAX_TOKENS`,
    /// `ANTHROPIC_EFFORT` or `ANTHROPIC_THINKING` holds a value the API does
    /// not take (including `disabled` thinking at effort `xhigh` / `max`).
    pub fn from_env() -> Result<Self, RunnerError> {
        let api_key = env_value(ANTHROPIC_API_KEY_ENV).ok_or_else(missing_key)?;
        Self::shared_settings_from_env(api_key)
    }

    /// One configuration per numbered key in the environment, every other
    /// setting shared: `ANTHROPIC_API_KEY`, then `ANTHROPIC_API_KEY_2`,
    /// `ANTHROPIC_API_KEY_3`, … up to the first unset or blank one.
    ///
    /// Each key belongs to its own Anthropic organization with its own
    /// prepaid credit; handed to [`provider_pool`](crate::pool::provider_pool)
    /// in this order, account 1 serves until its credit or rate limit
    /// refuses, then account 2, and so on. A quota refusal is never retried
    /// in place, so the move costs one request. A 529 overload is
    /// Anthropic-wide and is retried in place on every account: before the
    /// chain leaves the pool it costs N × (`max_retries` + 1) requests and N
    /// × the backoff, so size `ANTHROPIC_MAX_RETRIES` with the pool.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::from_env`]; `ANTHROPIC_API_KEY` itself must be set.
    pub fn pool_from_env() -> Result<Vec<Self>, RunnerError> {
        let keys = numbered_api_keys(env_value);
        let Some(first) = keys.first() else {
            return Err(missing_key());
        };
        let shared = Self::shared_settings_from_env(first.clone())?;
        Ok(keys
            .into_iter()
            .map(|api_key| Self {
                api_key,
                ..shared.clone()
            })
            .collect())
    }

    /// Every setting but the key, read from the environment.
    fn shared_settings_from_env(api_key: String) -> Result<Self, RunnerError> {
        let defaults = Self::new(api_key);
        let max_tokens = match env_value(ANTHROPIC_MAX_TOKENS_ENV) {
            Some(raw) => raw
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| {
                    RunnerError::config(format!(
                        "{ANTHROPIC_MAX_TOKENS_ENV} must be a positive integer, got {raw:?}"
                    ))
                })?,
            None => defaults.max_tokens,
        };
        let effort = env_value(ANTHROPIC_EFFORT_ENV)
            .map(|raw| {
                AnthropicEffort::parse(&raw).ok_or_else(|| {
                    RunnerError::config(format!(
                        "{ANTHROPIC_EFFORT_ENV} must be one of low, medium, high, xhigh, max; got {raw:?}"
                    ))
                })
            })
            .transpose()?;
        let thinking = env_value(ANTHROPIC_THINKING_ENV)
            .map(|raw| {
                AnthropicThinking::parse(&raw).ok_or_else(|| {
                    RunnerError::config(format!(
                        "{ANTHROPIC_THINKING_ENV} must be adaptive or disabled; got {raw:?}"
                    ))
                })
            })
            .transpose()?;
        if thinking == Some(AnthropicThinking::Disabled)
            && matches!(effort, Some(AnthropicEffort::XHigh | AnthropicEffort::Max))
        {
            return Err(RunnerError::config(format!(
                "{ANTHROPIC_THINKING_ENV}=disabled is refused at effort xhigh or max; lower {ANTHROPIC_EFFORT_ENV} or use adaptive thinking"
            )));
        }
        Ok(Self {
            model: env_value(ANTHROPIC_DEFAULT_MODEL_ENV).unwrap_or(defaults.model),
            base_url: env_value(ANTHROPIC_BASE_URL_ENV).unwrap_or(defaults.base_url),
            max_tokens,
            effort,
            thinking,
            retry: HttpRetryConfig::from_env(ANTHROPIC_ENV_PREFIX),
            api_key: defaults.api_key,
            timeout: defaults.timeout,
        })
    }

    /// Set the default model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the API base URL (without the `/v1` path)
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the default `max_tokens`
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set `output_config.effort`
    #[must_use]
    pub const fn with_effort(mut self, effort: AnthropicEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Set the `thinking` mode
    #[must_use]
    pub const fn with_thinking(mut self, thinking: AnthropicThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set the request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the retry policy
    #[must_use]
    pub fn with_retry(mut self, retry: HttpRetryConfig) -> Self {
        self.retry = retry;
        self
    }
}

impl Debug for AnthropicConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("AnthropicConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("max_tokens", &self.max_tokens)
            .field("effort", &self.effort)
            .field("thinking", &self.thinking)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .finish()
    }
}

/// A set, non-blank environment variable
fn env_value(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The error for a missing `ANTHROPIC_API_KEY`
fn missing_key() -> RunnerError {
    RunnerError::config(format!(
        "{ANTHROPIC_API_KEY_ENV} environment variable not set"
    ))
}

/// The numbered keys `lookup` yields — `ANTHROPIC_API_KEY`,
/// `ANTHROPIC_API_KEY_2`, `ANTHROPIC_API_KEY_3`, … — up to the first unset
/// or blank one.
fn numbered_api_keys(lookup: impl Fn(&str) -> Option<String>) -> Vec<String> {
    let mut keys = Vec::new();
    for account in 1_usize.. {
        let name = if account == 1 {
            ANTHROPIC_API_KEY_ENV.to_owned()
        } else {
            format!("{ANTHROPIC_API_KEY_ENV}_{account}")
        };
        match lookup(&name).filter(|v| !v.trim().is_empty()) {
            Some(key) => keys.push(key),
            None => break,
        }
    }
    keys
}

// ============================================================================
// Wire types: request
// ============================================================================

/// A `cache_control` marker. Only the 5-minute `ephemeral` kind is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

/// The 5-minute ephemeral cache breakpoint
const EPHEMERAL: CacheControl = CacheControl { kind: "ephemeral" };

/// `POST /v1/messages` body. Field order is the serialization order.
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemBlock>,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<WireThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
    /// Automatic caching: a breakpoint on the last cacheable block
    cache_control: CacheControl,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

/// One block of the top-level `system` array
#[derive(Debug, Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// One turn of `messages`
#[derive(Debug, Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<WireBlock>,
}

/// A content block of a request turn
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "String::is_empty")]
        content: String,
    },
}

/// A base64 image source
#[derive(Debug, Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    kind: &'static str,
    media_type: String,
    data: String,
}

/// A tool definition
#[derive(Debug, Serialize)]
struct WireTool {
    name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    description: String,
    input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// `tool_choice`
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireToolChoice {
    Auto,
    None,
    Any,
    Tool { name: String },
}

/// `thinking`
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireThinking {
    Adaptive,
    Disabled,
}

/// `output_config`
#[derive(Debug, Serialize)]
struct OutputConfig {
    effort: &'static str,
}

// ============================================================================
// Wire types: response, stream, error
// ============================================================================

/// A non-streaming `/v1/messages` response
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ResponseBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

/// A content block of a response. `thinking`, `redacted_thinking` and any
/// future kind are read past.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

/// Why a refusal stopped, when the API says
#[derive(Debug, Deserialize)]
struct StopDetails {
    #[serde(default)]
    category: Option<String>,
}

/// Usage as Anthropic reports it: `input_tokens` is the fresh remainder only
#[derive(Debug, Default, Clone, Copy, Deserialize)]
struct WireUsage {
    #[serde(rename = "input_tokens", default)]
    input: Option<u32>,
    #[serde(rename = "output_tokens", default)]
    output: Option<u32>,
    #[serde(rename = "cache_read_input_tokens", default)]
    cache_read: Option<u32>,
    #[serde(rename = "cache_creation_input_tokens", default)]
    cache_creation: Option<u32>,
}

/// A streaming event, dispatched on its JSON `type`. Every field is
/// defaulted, so an event of a known type with an unexpected shape is read
/// past rather than failing the stream.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        #[serde(default)]
        message: MessageStartBody,
    },
    ContentBlockStart {
        #[serde(default)]
        index: usize,
        #[serde(default)]
        content_block: Option<ResponseBlock>,
    },
    ContentBlockDelta {
        #[serde(default)]
        index: usize,
        delta: StreamDelta,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<WireUsage>,
    },
    MessageStop,
    Error {
        error: ErrorBody,
    },
    /// `content_block_stop`, `ping` and any future event
    #[serde(other)]
    Other,
}

/// The message a `message_start` event opens: its usage carries the prompt
/// side (fresh input and both cache counts)
#[derive(Debug, Default, Deserialize)]
struct MessageStartBody {
    #[serde(default)]
    usage: Option<WireUsage>,
}

/// The delta of a `content_block_delta` event
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamDelta {
    TextDelta {
        text: String,
    },
    /// A fragment of a `tool_use` block's input, as raw JSON text
    InputJsonDelta {
        #[serde(default)]
        partial_json: String,
    },
    /// `thinking_delta`, `signature_delta`, …
    #[serde(other)]
    Other,
}

/// The delta of a `message_delta` event
#[derive(Debug, Deserialize)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
}

/// `{"type": "error", "error": {...}}`
#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

/// The vendor's own description of a failure
#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: String,
}

impl ErrorBody {
    /// `<type> - <message>`
    fn describe(&self) -> String {
        format!("{} - {}", self.kind, self.message)
    }
}

// ============================================================================
// Mapping
// ============================================================================

/// Whether `model` accepts `temperature` / `top_p`
fn accepts_sampling(model: &str) -> bool {
    has_prefix(model, SAMPLING_MODEL_PREFIXES)
}

/// Whether `model` refuses a forced `tool_choice`
fn rejects_forced_tool_choice(model: &str) -> bool {
    has_prefix(model, FORCED_TOOL_CHOICE_REJECTED)
}

/// Whether `model` starts with one of `prefixes`
fn has_prefix(model: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| model.starts_with(p))
}

/// The `output_config.effort` `model` takes from `config`: none for a model
/// that predates the parameter.
fn wire_effort(config: &AnthropicConfig, model: &str) -> Option<OutputConfig> {
    let effort = config.effort?;
    if has_prefix(model, PRE_EFFORT_MODEL_PREFIXES) {
        debug!(
            model,
            effort = effort.as_str(),
            "omitting output_config.effort: the model predates it"
        );
        return None;
    }
    Some(OutputConfig {
        effort: effort.as_str(),
    })
}

/// The `thinking` field `model` takes from `config`, gated on what the model
/// accepts so a configured mode never turns every request into a 400:
/// `adaptive` is omitted where the model predates it, and `disabled` where the
/// model always thinks, or thinks at the configured effort (`xhigh` / `max`).
/// An omitted field leaves the model's own default.
fn wire_thinking(config: &AnthropicConfig, model: &str) -> Option<WireThinking> {
    let refused = match config.thinking? {
        AnthropicThinking::Adaptive => {
            has_prefix(model, PRE_EFFORT_MODEL_PREFIXES)
                || has_prefix(model, PRE_ADAPTIVE_MODEL_PREFIXES)
        }
        AnthropicThinking::Disabled => {
            has_prefix(model, THINKING_ALWAYS_ON)
                || (has_prefix(model, THINKING_OFF_UP_TO_HIGH)
                    && matches!(
                        config.effort,
                        Some(AnthropicEffort::XHigh | AnthropicEffort::Max)
                    ))
        }
    };
    if refused {
        debug!(
            model,
            thinking = ?config.thinking,
            "omitting thinking: the model refuses this mode"
        );
        return None;
    }
    config.thinking.map(|thinking| match thinking {
        AnthropicThinking::Adaptive => WireThinking::Adaptive,
        AnthropicThinking::Disabled => WireThinking::Disabled,
    })
}

/// `{}`
fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// A tool call's arguments as the JSON object `tool_use.input` must be. A
/// JSON-encoded string is decoded; anything that is not an object is `{}`.
fn tool_input(arguments: &Value) -> Value {
    match arguments {
        Value::Object(_) => arguments.clone(),
        Value::String(encoded) => serde_json::from_str::<Value>(encoded)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(empty_object),
        _ => empty_object(),
    }
}

/// A text block, or nothing when the text is blank (the API rejects it)
fn text_block(text: &str) -> Option<WireBlock> {
    (!text.trim().is_empty()).then(|| WireBlock::Text {
        text: text.to_owned(),
    })
}

/// The messages of a conversation as Anthropic's `system` and `messages`.
fn convert_messages(messages: &[ChatMessage]) -> (Vec<SystemBlock>, Vec<WireMessage>) {
    let mut system: Vec<SystemBlock> = Vec::new();
    let mut turns: Vec<WireMessage> = Vec::new();
    // The tool calls of the most recent assistant turn: only these may be
    // answered by a tool_result in the turn that follows it.
    let mut open_calls: Vec<String> = Vec::new();

    for message in messages {
        match message.role {
            MessageRole::System => {
                if !message.content.trim().is_empty() {
                    system.push(SystemBlock {
                        kind: "text",
                        text: message.content.clone(),
                        cache_control: None,
                    });
                }
            }
            MessageRole::User => {
                let mut blocks: Vec<WireBlock> = message
                    .images
                    .iter()
                    .flatten()
                    .map(|image| WireBlock::Image {
                        source: ImageSource {
                            kind: "base64",
                            media_type: image.mime_type.clone(),
                            data: image.data.clone(),
                        },
                    })
                    .collect();
                blocks.extend(text_block(&message.content));
                push_turn(&mut turns, "user", blocks);
            }
            MessageRole::Assistant => {
                if turns.is_empty() {
                    debug!(
                        "dropping a leading assistant turn: the first message must be the user's"
                    );
                    continue;
                }
                let calls = message.tool_calls.as_deref().unwrap_or_default();
                let mut blocks: Vec<WireBlock> = text_block(&message.content).into_iter().collect();
                blocks.extend(calls.iter().map(|call| WireBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.function_name.clone(),
                    input: tool_input(&call.arguments),
                }));
                if blocks.is_empty() {
                    continue;
                }
                if turns.last().is_some_and(|t| t.role != "assistant") {
                    open_calls.clear();
                }
                open_calls.extend(calls.iter().map(|call| call.id.clone()));
                push_turn(&mut turns, "assistant", blocks);
            }
            MessageRole::Tool => {
                let block = match &message.tool_call_id {
                    Some(id) if open_calls.contains(id) => WireBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: message.content.clone(),
                    },
                    _ => WireBlock::Text {
                        text: format!(
                            "Result of tool {}: {}",
                            message.name.as_deref().unwrap_or("call"),
                            message.content
                        ),
                    },
                };
                push_turn(&mut turns, "user", vec![block]);
            }
        }
    }

    if let Some(last) = system.last_mut() {
        last.cache_control = Some(EPHEMERAL);
    }
    pair_tool_blocks(&mut turns);
    for turn in &mut turns {
        // Tool results lead the user turn that answers a tool call; a stable
        // sort keeps every other order intact.
        turn.content
            .sort_by_key(|block| !matches!(block, WireBlock::ToolResult { .. }));
    }
    (system, turns)
}

/// Make every `tool_use` and `tool_result` a matched pair, as the API
/// requires: each `tool_use` of an assistant turn is answered by a
/// `tool_result` in the user turn right after it, and each `tool_result`
/// answers a `tool_use` of the assistant turn right before it. A block on
/// either side without its partner becomes plain text, so a history with a
/// lost or mismatched result still reads as what happened instead of a 400.
fn pair_tool_blocks(turns: &mut [WireMessage]) {
    // Results first: what a user turn may answer is what the assistant turn
    // before it called.
    for index in 0..turns.len() {
        if turns[index].role != "user" {
            continue;
        }
        let called = match index.checked_sub(1).map(|prev| &turns[prev]) {
            Some(prev) if prev.role == "assistant" => block_ids(prev, ToolSide::Use),
            _ => Vec::new(),
        };
        for block in &mut turns[index].content {
            if let WireBlock::ToolResult {
                tool_use_id,
                content,
            } = block
            {
                if !called.contains(tool_use_id) {
                    *block = WireBlock::Text {
                        text: format!("Result of tool call {tool_use_id}: {content}"),
                    };
                }
            }
        }
    }
    // Then calls: what an assistant turn called must be answered by the user
    // turn after it.
    for index in 0..turns.len() {
        if turns[index].role != "assistant" {
            continue;
        }
        let answered = match turns.get(index + 1) {
            Some(next) if next.role == "user" => block_ids(next, ToolSide::Result),
            _ => Vec::new(),
        };
        for block in &mut turns[index].content {
            if let WireBlock::ToolUse { id, name, input } = block {
                if !answered.contains(id) {
                    *block = WireBlock::Text {
                        text: format!("Called tool {name} with {input}; no result was recorded."),
                    };
                }
            }
        }
    }
}

/// Which half of a tool exchange [`block_ids`] collects
#[derive(Clone, Copy)]
enum ToolSide {
    /// `tool_use` ids
    Use,
    /// `tool_result` `tool_use_id`s
    Result,
}

/// The tool-call ids a turn carries on one side of the exchange, in order
fn block_ids(turn: &WireMessage, side: ToolSide) -> Vec<String> {
    turn.content
        .iter()
        .filter_map(|block| match (side, block) {
            (ToolSide::Use, WireBlock::ToolUse { id, .. })
            | (
                ToolSide::Result,
                WireBlock::ToolResult {
                    tool_use_id: id, ..
                },
            ) => Some(id.clone()),
            _ => None,
        })
        .collect()
}

/// Append `blocks` as a `role` turn, merging into the previous turn when it
/// has the same role. Nothing is appended for an empty block list.
fn push_turn(turns: &mut Vec<WireMessage>, role: &'static str, blocks: Vec<WireBlock>) {
    if blocks.is_empty() {
        return;
    }
    match turns.last_mut() {
        Some(last) if last.role == role => last.content.extend(blocks),
        _ => turns.push(WireMessage {
            role,
            content: blocks,
        }),
    }
}

/// Tool definitions, the last one carrying the tools-prefix cache breakpoint
fn convert_tools(tools: &[ToolDefinition]) -> Vec<WireTool> {
    let mut wire: Vec<WireTool> = tools
        .iter()
        .map(|tool| WireTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            cache_control: None,
        })
        .collect();
    if let Some(last) = wire.last_mut() {
        last.cache_control = Some(EPHEMERAL);
    }
    wire
}

/// The request's tool choice in wire shape; `auto` when it names none
fn convert_tool_choice(choice: Option<&ToolChoice>, model: &str) -> WireToolChoice {
    match choice {
        None | Some(ToolChoice::Auto) => WireToolChoice::Auto,
        Some(ToolChoice::None) => WireToolChoice::None,
        Some(ToolChoice::Required | ToolChoice::Specific { .. })
            if rejects_forced_tool_choice(model) =>
        {
            WireToolChoice::Auto
        }
        Some(ToolChoice::Required) => WireToolChoice::Any,
        Some(ToolChoice::Specific { name }) => WireToolChoice::Tool { name: name.clone() },
    }
}

/// The request body for `request` under `config`
fn build_body(config: &AnthropicConfig, request: &ChatRequest, stream: bool) -> MessagesRequest {
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| config.model.clone());
    let (system, messages) = convert_messages(&request.messages);
    let tools = request
        .tools
        .as_deref()
        .map(convert_tools)
        .unwrap_or_default();
    let tool_choice =
        (!tools.is_empty()).then(|| convert_tool_choice(request.tool_choice.as_ref(), &model));
    let (temperature, top_p) = if accepts_sampling(&model) {
        // Claude 4 models refuse both together: temperature wins.
        (
            request.temperature,
            request.top_p.filter(|_| request.temperature.is_none()),
        )
    } else {
        (None, None)
    };
    let stop_sequences = request
        .stop
        .iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect();
    MessagesRequest {
        max_tokens: request.max_tokens.unwrap_or(config.max_tokens),
        system,
        messages,
        tools,
        tool_choice,
        temperature,
        top_p,
        stop_sequences,
        thinking: wire_thinking(config, &model),
        output_config: wire_effort(config, &model),
        cache_control: EPHEMERAL,
        stream,
        model,
    }
}

/// Anthropic's `stop_reason` in the crate's `OpenAI` vocabulary
fn map_stop_reason(reason: &str) -> String {
    match reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" | "model_context_window_exceeded" => "length",
        "tool_use" => "tool_calls",
        other => other,
    }
    .to_owned()
}

/// Usage normalised to the gross prompt convention, cache counts kept apart.
///
/// Anthropic's `input_tokens` is fresh-only: cache reads and writes are
/// separate and additive, so the real prompt is the sum of all three. The
/// same rule as the Claude Code runner, so pricing carves the cached share
/// back out of one convention whichever path served the turn.
fn gross_usage(usage: WireUsage) -> TokenUsage {
    let prompt = usage
        .input
        .unwrap_or(0)
        .saturating_add(usage.cache_read.unwrap_or(0))
        .saturating_add(usage.cache_creation.unwrap_or(0));
    let output = usage.output.unwrap_or(0);
    TokenUsage::new(prompt, output, prompt.saturating_add(output))
        .with_cache(usage.cache_read, usage.cache_creation)
}

/// A parsed response as a [`ChatResponse`]
fn into_chat_response(response: MessagesResponse, model: &str) -> ChatResponse {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    for block in response.content {
        match block {
            ResponseBlock::Text { text } => content.push_str(&text),
            ResponseBlock::ToolUse { id, name, input } => tool_calls.push(ToolCallRequest {
                id,
                function_name: name,
                arguments: input,
            }),
            ResponseBlock::Other => {}
        }
    }
    if response.stop_reason.as_deref() == Some("refusal") {
        // A decline can land mid-output: whatever came before it is a
        // fragment of an answer the model did not finish, never a reply.
        warn!(
            provider = PROVIDER_NAME,
            model,
            category = response
                .stop_details
                .as_ref()
                .and_then(|d| d.category.as_deref())
                .unwrap_or("unspecified"),
            discarded_text_len = content.len(),
            discarded_tool_calls = tool_calls.len(),
            "the model declined the request (stop_reason refusal); partial output discarded"
        );
        content.clear();
        tool_calls.clear();
    }
    ChatResponse {
        content,
        model: model.to_owned(),
        usage: response.usage.map(gross_usage),
        finish_reason: response.stop_reason.as_deref().map(map_stop_reason),
        warnings: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
    }
}

/// The status an error `type` stands for, so a mid-stream error event maps
/// the way the same refusal would as an HTTP status
fn status_for_error_type(kind: &str) -> StatusCode {
    match kind {
        "invalid_request_error" => StatusCode::BAD_REQUEST,
        "authentication_error" => StatusCode::UNAUTHORIZED,
        "billing_error" => StatusCode::PAYMENT_REQUIRED,
        "permission_error" => StatusCode::FORBIDDEN,
        "not_found_error" => StatusCode::NOT_FOUND,
        "request_too_large" => StatusCode::PAYLOAD_TOO_LARGE,
        "rate_limit_error" => StatusCode::TOO_MANY_REQUESTS,
        "overloaded_error" => overloaded_status(),
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Anthropic's 529 "overloaded"
fn overloaded_status() -> StatusCode {
    StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
}

/// Map a refusal to a [`RunnerError`]: the status, the vendor's error body
/// when it sent one, the `retry-after` wait, and the model asked for.
///
/// The message is the vendor's `<type> - <message>`, never the request: the
/// API key cannot reach it.
fn map_error(
    status: StatusCode,
    body: Option<&ErrorBody>,
    retry_after: Option<u64>,
    model: &str,
) -> RunnerError {
    let kind = body.map(|b| b.kind.as_str());
    let message = body.map_or_else(|| format!("HTTP {status}"), ErrorBody::describe);
    match (status.as_u16(), kind) {
        (413, _) | (_, Some("request_too_large")) => {
            RunnerError::context_length(format!("{PROVIDER_NAME}: {message}"))
        }
        (400, _) if message.to_lowercase().contains(PROMPT_TOO_LONG) => {
            RunnerError::context_length(format!("{PROVIDER_NAME}: {message}"))
        }
        (404, _) | (_, Some("not_found_error")) => RunnerError::model_unavailable(model),
        (402, _) | (_, Some("billing_error")) => {
            client::map_http_error(PROVIDER_NAME, StatusCode::PAYMENT_REQUIRED, &message)
        }
        (429, _) | (_, Some("rate_limit_error")) => {
            client::rate_limit_error(PROVIDER_NAME, &message, retry_after)
        }
        (529, _) | (_, Some("overloaded_error")) => {
            RunnerError::external_service(PROVIDER_NAME, format!("HTTP 529 overloaded: {message}"))
        }
        _ => client::map_http_error(PROVIDER_NAME, status, &message),
    }
}

/// The retry decision for a refused attempt. A quota refusal (`RateLimit`:
/// a 429, a spent credit, or a body typed `rate_limit_error` /
/// `billing_error` on any status) leaves the tier at once: retrying it in
/// place spends the backoff against an organization that has already said
/// no, while a [`FallbackProvider`](crate::fallback::FallbackProvider) holds
/// the tier in quota cooldown and asks the next account. Everything else
/// follows the shared status rule (502, 503 and 529 are retried in place).
fn attempt_error(status: StatusCode, error: RunnerError) -> AttemptError {
    if error.kind == ErrorKind::RateLimit {
        AttemptError::permanent(error)
    } else {
        AttemptError::from_status(status, error)
    }
}

/// Map a non-success HTTP response body
fn map_api_error(
    status: StatusCode,
    body: &str,
    retry_after: Option<u64>,
    model: &str,
) -> RunnerError {
    let parsed = serde_json::from_str::<ErrorEnvelope>(body)
        .ok()
        .map(|e| e.error);
    if parsed.is_none() {
        debug!(
            status = status.as_u16(),
            body_preview = %body.chars().take(200).collect::<String>(),
            "Anthropic API returned a non-JSON error response"
        );
    }
    map_error(status, parsed.as_ref(), retry_after, model)
}

/// The `retry-after` header in whole seconds, rounded up
fn retry_after_header(headers: &HeaderMap) -> Option<u64> {
    let seconds: f64 = headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(client::whole_seconds(seconds))
}

/// Parse one SSE `data:` payload of a stream opened for `model` into a
/// [`StreamChunk`]
fn parse_stream_data(json: &str, model: &str) -> Option<Result<StreamChunk, RunnerError>> {
    match serde_json::from_str::<StreamEvent>(json) {
        Ok(StreamEvent::ContentBlockDelta {
            delta: StreamDelta::TextDelta { text },
            ..
        }) => Some(Ok(StreamChunk {
            delta: text,
            is_final: false,
            finish_reason: None,
        })),
        Ok(StreamEvent::MessageDelta { delta, .. }) => {
            let reason = delta.stop_reason?;
            if reason == "refusal" {
                warn!(
                    provider = PROVIDER_NAME,
                    "the model declined the request mid-stream (stop_reason refusal)"
                );
            }
            Some(Ok(StreamChunk {
                delta: String::new(),
                is_final: true,
                finish_reason: Some(map_stop_reason(&reason)),
            }))
        }
        Ok(StreamEvent::Error { error }) => Some(Err(map_error(
            status_for_error_type(&error.kind),
            Some(&error),
            None,
            model,
        ))),
        Ok(
            StreamEvent::MessageStart { .. }
            | StreamEvent::ContentBlockStart { .. }
            | StreamEvent::ContentBlockDelta { .. }
            | StreamEvent::MessageStop
            | StreamEvent::Other,
        ) => None,
        Err(e) => Some(Err(RunnerError::external_service(
            PROVIDER_NAME,
            format!("SSE parse error: {e}"),
        ))),
    }
}

/// One content block of a streamed message, as its events assemble it
#[derive(Debug)]
enum StreamedBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        /// The input `content_block_start` carried: `{}` for a call whose
        /// arguments then stream as `input_json_delta`, and the whole input
        /// for a call that has none
        input: Value,
        /// The concatenated `input_json_delta` fragments
        partial_json: String,
    },
    /// A thinking, server-tool or future block: read past, as
    /// [`into_chat_response`] reads it past
    Other,
}

/// Assembles a `/v1/messages` event stream into the [`ToolStreamEvent`]s of
/// [`LlmProvider::complete_stream_with_tools`].
///
/// Text deltas are forwarded as they arrive. At `message_stop` the blocks,
/// stop reason and usage are rebuilt into the [`MessagesResponse`] the
/// non-streaming endpoint would have returned and mapped by the same
/// [`into_chat_response`], so the final response is `complete()`'s by
/// construction: refusal discard, stop-reason mapping and gross usage
/// included.
#[derive(Debug)]
struct ToolStreamAccumulator {
    /// The model the stream was opened for, reported as `complete()` does
    model: String,
    /// Content blocks keyed on their `index`, so they keep the wire order
    blocks: BTreeMap<usize, StreamedBlock>,
    stop_reason: Option<String>,
    stop_details: Option<StopDetails>,
    /// `message_start`'s usage, overlaid with each `message_delta`'s counts
    usage: Option<WireUsage>,
}

impl ToolStreamAccumulator {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            blocks: BTreeMap::new(),
            stop_reason: None,
            stop_details: None,
            usage: None,
        }
    }

    /// Read one SSE `data:` payload. Returns the event it produces, if any:
    /// a text delta, the final `Done` at `message_stop`, or an error.
    fn feed(&mut self, json: &str) -> Option<Result<ToolStreamEvent, RunnerError>> {
        let event = match serde_json::from_str::<StreamEvent>(json) {
            Ok(event) => event,
            Err(e) => {
                return Some(Err(RunnerError::external_service(
                    PROVIDER_NAME,
                    format!("SSE parse error: {e}"),
                )))
            }
        };
        match event {
            StreamEvent::MessageStart { message } => {
                if let Some(usage) = message.usage {
                    self.usage = Some(overlay_usage(self.usage, usage));
                }
                None
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let (block, text) = match content_block {
                    Some(ResponseBlock::Text { text }) => {
                        (StreamedBlock::Text(text.clone()), Some(text))
                    }
                    Some(ResponseBlock::ToolUse { id, name, input }) => (
                        StreamedBlock::ToolUse {
                            id,
                            name,
                            input,
                            partial_json: String::new(),
                        },
                        None,
                    ),
                    Some(ResponseBlock::Other) | None => (StreamedBlock::Other, None),
                };
                self.blocks.insert(index, block);
                text.filter(|t| !t.is_empty())
                    .map(|t| Ok(ToolStreamEvent::TextDelta(t)))
            }
            StreamEvent::ContentBlockDelta {
                index,
                delta: StreamDelta::TextDelta { text },
            } => {
                match self
                    .blocks
                    .entry(index)
                    .or_insert_with(|| StreamedBlock::Text(String::new()))
                {
                    StreamedBlock::Text(content) => content.push_str(&text),
                    StreamedBlock::ToolUse { .. } | StreamedBlock::Other => return None,
                }
                (!text.is_empty()).then_some(Ok(ToolStreamEvent::TextDelta(text)))
            }
            StreamEvent::ContentBlockDelta {
                index,
                delta: StreamDelta::InputJsonDelta { partial_json: part },
            } => {
                // A server tool's input streams the same way; like its block,
                // it is read past.
                if let Some(StreamedBlock::ToolUse { partial_json, .. }) =
                    self.blocks.get_mut(&index)
                {
                    partial_json.push_str(&part);
                }
                None
            }
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(reason);
                }
                if delta.stop_details.is_some() {
                    self.stop_details = delta.stop_details;
                }
                if let Some(usage) = usage {
                    self.usage = Some(overlay_usage(self.usage, usage));
                }
                None
            }
            StreamEvent::MessageStop => Some(self.finish().map(ToolStreamEvent::Done)),
            StreamEvent::Error { error } => Some(Err(map_error(
                status_for_error_type(&error.kind),
                Some(&error),
                None,
                &self.model,
            ))),
            StreamEvent::ContentBlockDelta {
                delta: StreamDelta::Other,
                ..
            }
            | StreamEvent::Other => None,
        }
    }

    /// The message as `complete()` would have returned it
    fn finish(&mut self) -> Result<ChatResponse, RunnerError> {
        let mut content = Vec::with_capacity(self.blocks.len());
        for block in mem::take(&mut self.blocks).into_values() {
            match block {
                StreamedBlock::Text(text) => content.push(ResponseBlock::Text { text }),
                StreamedBlock::ToolUse {
                    id,
                    name,
                    input,
                    partial_json,
                } => {
                    let input = if partial_json.is_empty() {
                        input
                    } else {
                        serde_json::from_str(&partial_json).map_err(|e| {
                            RunnerError::external_service(
                                PROVIDER_NAME,
                                format!("Failed to parse the streamed input of tool {name}: {e}"),
                            )
                        })?
                    };
                    content.push(ResponseBlock::ToolUse { id, name, input });
                }
                StreamedBlock::Other => {}
            }
        }
        let response = MessagesResponse {
            content,
            stop_reason: self.stop_reason.take(),
            stop_details: self.stop_details.take(),
            usage: self.usage.take(),
        };
        Ok(into_chat_response(response, &self.model))
    }
}

/// `base` with every count `update` reports replacing its own. Anthropic's
/// `message_delta` counts are cumulative, so the last report of each wins.
fn overlay_usage(base: Option<WireUsage>, update: WireUsage) -> WireUsage {
    let base = base.unwrap_or_default();
    WireUsage {
        input: update.input.or(base.input),
        output: update.output.or(base.output),
        cache_read: update.cache_read.or(base.cache_read),
        cache_creation: update.cache_creation.or(base.cache_creation),
    }
}

/// The [`ToolStream`] over a `/v1/messages` SSE byte stream opened for
/// `model`: pull-based, so the bytes are read only as the consumer asks.
fn create_tool_stream<S, B, E>(byte_stream: S, model: impl Into<String>) -> ToolStream
where
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]>,
    E: Display,
{
    Box::pin(ToolSseStream {
        bytes: Box::pin(byte_stream),
        parser: SseParser::new(),
        accumulator: ToolStreamAccumulator::new(model),
        pending: VecDeque::new(),
        terminated: false,
    })
}

/// The stream [`create_tool_stream`] returns
struct ToolSseStream<S> {
    bytes: Pin<Box<S>>,
    parser: SseParser,
    accumulator: ToolStreamAccumulator,
    pending: VecDeque<Result<ToolStreamEvent, RunnerError>>,
    /// Set once the `Done` or an error is queued: nothing follows either
    terminated: bool,
}

impl<S> ToolSseStream<S> {
    /// Queue what a batch of SSE events produces, up to the terminal item
    fn queue(&mut self, events: Vec<SseEvent>) {
        for event in events {
            if self.terminated {
                return;
            }
            // `[DONE]` is the OpenAI convention; Anthropic ends on `message_stop`.
            let SseEvent::Data(json) = event else {
                continue;
            };
            if let Some(item) = self.accumulator.feed(&json) {
                self.terminated = matches!(item, Err(_) | Ok(ToolStreamEvent::Done(_)));
                self.pending.push_back(item);
            }
        }
    }
}

impl<S, B, E> Stream for ToolSseStream<S>
where
    S: Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: Display,
{
    type Item = Result<ToolStreamEvent, RunnerError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(item) = this.pending.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.terminated {
                return Poll::Ready(None);
            }
            match this.bytes.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    let events = this.parser.feed(bytes.as_ref());
                    this.queue(events);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.terminated = true;
                    return Poll::Ready(Some(Err(RunnerError::external_service(
                        PROVIDER_NAME,
                        format!("Stream read error: {e}"),
                    ))));
                }
                Poll::Ready(None) => {
                    let events = this.parser.flush();
                    this.queue(events);
                    if !this.terminated {
                        // `complete()` fails on a truncated body; a stream that
                        // closes before `message_stop` has no response to give.
                        this.terminated = true;
                        this.pending.push_back(Err(RunnerError::external_service(
                            PROVIDER_NAME,
                            "the stream closed before message_stop",
                        )));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Provider
// ============================================================================

/// Anthropic Messages API provider
pub struct AnthropicProvider {
    config: AnthropicConfig,
    client: reqwest::Client,
    available_models: Vec<String>,
}

impl AnthropicProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: AnthropicConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: AnthropicConfig, client: reqwest::Client) -> Self {
        debug!(
            model = %config.model,
            max_tokens = config.max_tokens,
            effort = config.effort.map(AnthropicEffort::as_str),
            max_retries = config.retry.max_retries,
            "Anthropic provider initialized"
        );
        Self {
            config,
            client,
            available_models: AVAILABLE_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Override the default model after construction.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// `{base_url}/v1/{path}`
    fn api_url(&self, path: &str) -> String {
        format!("{}/v1/{path}", self.config.base_url.trim_end_matches('/'))
    }

    /// The authentication and version headers. The key is marked sensitive,
    /// so a header dump never prints it.
    fn auth_headers(&self) -> Result<HeaderMap, RunnerError> {
        let mut key = HeaderValue::from_str(&self.config.api_key).map_err(|_| {
            RunnerError::config(format!(
                "{ANTHROPIC_API_KEY_ENV} holds characters an HTTP header cannot carry"
            ))
        })?;
        key.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", key);
        headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        Ok(headers)
    }

    /// An authenticated `POST /v1/messages`
    fn post_messages(&self, body: &MessagesRequest) -> Result<RequestBuilder, RunnerError> {
        Ok(self
            .client
            .post(self.api_url("messages"))
            .headers(self.auth_headers()?)
            .header(CONTENT_TYPE, "application/json")
            .json(body))
    }

    /// One non-streaming attempt
    async fn attempt_complete(&self, body: &MessagesRequest) -> Result<ChatResponse, AttemptError> {
        let response = self
            .post_messages(body)
            .map_err(AttemptError::permanent)?
            .send()
            .await
            .map_err(|e| AttemptError::from_send(PROVIDER_NAME, e))?;

        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        let text = response.text().await.map_err(|e| {
            AttemptError::transient(RunnerError::external_service(
                PROVIDER_NAME,
                format!("Failed to read response: {e}"),
            ))
        })?;

        if !status.is_success() {
            return Err(attempt_error(
                status,
                map_api_error(status, &text, retry_after, &body.model),
            ));
        }

        let parsed: MessagesResponse = serde_json::from_str(&text).map_err(|e| {
            AttemptError::permanent(RunnerError::external_service(
                PROVIDER_NAME,
                format!("Failed to parse response: {e}"),
            ))
        })?;
        Ok(into_chat_response(parsed, &body.model))
    }

    /// One attempt at opening a stream: the response, once its status is a
    /// success
    async fn open_stream(&self, body: &MessagesRequest) -> Result<Response, AttemptError> {
        let response = self
            .post_messages(body)
            .map_err(AttemptError::permanent)?
            .send()
            .await
            .map_err(|e| AttemptError::from_send(PROVIDER_NAME, e))?;

        let status = response.status();
        if !status.is_success() {
            let retry_after = retry_after_header(response.headers());
            let text = response.text().await.unwrap_or_default();
            return Err(attempt_error(
                status,
                map_api_error(status, &text, retry_after, &body.model),
            ));
        }

        Ok(response)
    }

    /// One attempt at opening the text stream
    async fn attempt_stream(&self, body: &MessagesRequest) -> Result<ChatStream, AttemptError> {
        let response = self.open_stream(body).await?;
        let model = body.model.clone();
        Ok(create_sse_stream(
            response.bytes_stream(),
            move |json| parse_stream_data(json, &model),
            PROVIDER_NAME,
        ))
    }

    /// One attempt at opening the tool-use stream
    async fn attempt_tool_stream(
        &self,
        body: &MessagesRequest,
    ) -> Result<ToolStream, AttemptError> {
        let response = self.open_stream(body).await?;
        Ok(create_tool_stream(
            response.bytes_stream(),
            body.model.as_str(),
        ))
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> &str {
        "Anthropic API"
    }

    fn capabilities(&self) -> LlmCapabilities {
        let base = LlmCapabilities::STREAMING
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::FUNCTION_CALLING
            | LlmCapabilities::VISION
            | LlmCapabilities::MAX_TOKENS
            | LlmCapabilities::STOP_SEQUENCES;
        if accepts_sampling(&self.config.model) {
            base | LlmCapabilities::TEMPERATURE | LlmCapabilities::TOP_P
        } else {
            base
        }
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let body = build_body(&self.config, request, false);
        let body = &body;
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Anthropic request",
            |attempt| async move {
                debug!(attempt, "Sending request to the Anthropic Messages API");
                self.attempt_complete(body).await
            },
        )
        .await
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let body = build_body(&self.config, request, true);
        let body = &body;
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Anthropic streaming request",
            |attempt| async move {
                debug!(attempt, "Opening a stream on the Anthropic Messages API");
                self.attempt_stream(body).await
            },
        )
        .await
    }

    /// Streams the reply text and closes with the response `complete()`
    /// returns, tool calls and usage included. Opening the stream is retried
    /// as `complete()` is; a failure after it opened is the stream's `Err`
    /// item.
    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete_stream_with_tools(
        &self,
        request: &ChatRequest,
    ) -> Result<ToolStream, RunnerError> {
        let body = build_body(&self.config, request, true);
        let body = &body;
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Anthropic streaming request",
            |attempt| async move {
                debug!(
                    attempt,
                    "Opening a tool-use stream on the Anthropic Messages API"
                );
                self.attempt_tool_stream(body).await
            },
        )
        .await
    }

    /// `GET /v1/models/{model}`: verifies the key and the model without
    /// spending credit.
    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        let response = self
            .client
            .get(self.api_url(&format!("models/{}", self.config.model)))
            .headers(self.auth_headers()?)
            .send()
            .await
            .map_err(|e| client::map_send_error(PROVIDER_NAME, e))?;
        Ok(response.status().is_success())
    }
}

impl Debug for AnthropicProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("AnthropicProvider")
            .field("default_model", &self.config.model)
            .field("api_key", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;

    use serde_json::json;

    use super::*;
    use crate::fallback::is_empty_completion;
    use crate::types::ImagePart;

    fn body_json(config: &AnthropicConfig, request: &ChatRequest) -> Value {
        let body = build_body(config, request, false);
        serde_json::to_value(body).expect("serialises") // Safe: test assertion
    }

    fn call(id: &str, name: &str, arguments: Value) -> ToolCallRequest {
        ToolCallRequest {
            id: id.to_owned(),
            function_name: name.to_owned(),
            arguments,
        }
    }

    fn assistant_calling(calls: Vec<ToolCallRequest>) -> ChatMessage {
        ChatMessage {
            tool_calls: Some(calls),
            ..ChatMessage::assistant("")
        }
    }

    fn weather_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_owned(),
            description: format!("{name} tool"),
            parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
        }
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let config = AnthropicConfig::new("sk-ant-super-secret");
        assert!(!format!("{config:?}").contains("sk-ant-super-secret"));
        let provider = AnthropicProvider::with_client(config, reqwest::Client::new());
        let output = format!("{provider:?}");
        assert!(!output.contains("sk-ant-super-secret"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn the_key_header_is_sensitive_and_carries_the_version() {
        let provider = AnthropicProvider::with_client(
            AnthropicConfig::new("sk-ant-super-secret"),
            reqwest::Client::new(),
        );
        let headers = provider.auth_headers().expect("headers"); // Safe: test assertion
        let key = headers.get("x-api-key").expect("key header"); // Safe: test assertion
        assert!(key.is_sensitive());
        assert!(!format!("{headers:?}").contains("sk-ant-super-secret"));
        assert_eq!(
            headers.get("anthropic-version").map(HeaderValue::as_bytes),
            Some(&b"2023-06-01"[..])
        );
    }

    #[test]
    fn system_messages_are_hoisted_into_cached_blocks() {
        let request = ChatRequest::new(vec![
            ChatMessage::system("persona"),
            ChatMessage::user("hi"),
            ChatMessage::system("rules"),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["system"],
            json!([
                {"type": "text", "text": "persona"},
                {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
            ])
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        );
        assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(body["model"], "claude-haiku-5-5");
        assert_eq!(body["max_tokens"], 16_000);
        assert!(body.get("stream").is_none());
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn no_system_and_no_tools_send_neither_field() {
        let body = body_json(
            &AnthropicConfig::new("k"),
            &ChatRequest::new(vec![ChatMessage::user("hi")]),
        );
        assert!(body.get("system").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn tool_use_and_tool_result_pair_and_parallel_results_merge() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("weather in Paris and Lyon?"),
            ChatMessage {
                tool_calls: Some(vec![
                    call("toolu_1", "get_weather", json!({"city": "Paris"})),
                    call("toolu_2", "get_weather", json!("{\"city\":\"Lyon\"}")),
                ]),
                ..ChatMessage::assistant("Checking.")
            },
            ChatMessage::tool("get_weather", "toolu_1", "18C"),
            ChatMessage::tool("get_weather", "toolu_2", "21C"),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [{"type": "text", "text": "weather in Paris and Lyon?"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "Lyon"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "21C"}
                ]}
            ])
        );
    }

    #[test]
    fn a_tool_call_without_text_sends_only_tool_use_and_null_arguments_become_an_object() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("go"),
            assistant_calling(vec![call("toolu_1", "ping", Value::Null)]),
            ChatMessage::tool("ping", "toolu_1", ""),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"][1]["content"],
            json!([{"type": "tool_use", "id": "toolu_1", "name": "ping", "input": {}}])
        );
        assert_eq!(
            body["messages"][2]["content"],
            json!([{"type": "tool_result", "tool_use_id": "toolu_1"}])
        );
    }

    #[test]
    fn same_role_turns_merge_with_tool_results_first() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("go"),
            assistant_calling(vec![call("toolu_1", "ping", json!({}))]),
            ChatMessage::tool("ping", "toolu_1", "pong"),
            ChatMessage::user("and now?"),
            ChatMessage::assistant("first"),
            ChatMessage::assistant("second"),
            ChatMessage::user("ok"),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        let roles: Vec<&str> = body["messages"]
            .as_array()
            .expect("messages") // Safe: test assertion
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
        assert_eq!(
            body["messages"][2]["content"],
            json!([
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "pong"},
                {"type": "text", "text": "and now?"}
            ])
        );
        assert_eq!(
            body["messages"][3]["content"].as_array().map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn leading_assistant_turns_empty_turns_and_orphan_results_never_reach_the_wire() {
        let request = ChatRequest::new(vec![
            ChatMessage::assistant("Welcome back!"),
            ChatMessage::tool("lookup", "toolu_gone", "stale"),
            ChatMessage::user("   "),
            ChatMessage::user("hello"),
            ChatMessage::assistant(""),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [
                {"type": "text", "text": "Result of tool lookup: stale"},
                {"type": "text", "text": "hello"}
            ]}])
        );
    }

    #[test]
    fn images_travel_as_base64_blocks_before_the_text() {
        let image = ImagePart::new("aGVsbG8=", "image/png").expect("valid mime"); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user_with_images(
            "what is this?",
            vec![image],
        )]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                {"type": "text", "text": "what is this?"}
            ])
        );
    }

    #[test]
    fn the_last_tool_carries_the_cache_breakpoint_and_the_body_is_deterministic() {
        let request = ChatRequest::new(vec![
            ChatMessage::system("persona"),
            ChatMessage::user("hi"),
        ])
        .with_tools(vec![
            weather_tool("a"),
            weather_tool("b"),
            weather_tool("c"),
        ]);
        let config = AnthropicConfig::new("k");
        let body = body_json(&config, &request);
        let tools = body["tools"].as_array().expect("tools"); // Safe: test assertion
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert!(tools[0].get("cache_control").is_none());
        assert!(tools[1].get("cache_control").is_none());
        assert_eq!(tools[2]["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(
            tools[0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(body["tool_choice"], json!({"type": "auto"}));

        let breakpoints = serde_json::to_string(&body)
            .expect("serialises") // Safe: test assertion
            .matches("\"cache_control\"")
            .count();
        assert_eq!(
            breakpoints, 3,
            "tools, system, and the automatic tail: within the limit of 4"
        );

        let first = serde_json::to_vec(&build_body(&config, &request, false)).expect("serialises"); // Safe: test assertion
        let second = serde_json::to_vec(&build_body(&config, &request, false)).expect("serialises"); // Safe: test assertion
        assert_eq!(
            first, second,
            "the cached prefix must be byte-identical between calls"
        );
    }

    #[test]
    fn a_tool_without_parameters_gets_an_empty_object_schema() {
        let request =
            ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition {
                name: "now".to_owned(),
                description: String::new(),
                parameters: None,
            }]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["tools"][0]["input_schema"],
            json!({"type": "object", "properties": {}})
        );
        assert!(body["tools"][0].get("description").is_none());
    }

    #[test]
    fn forced_tool_choice_is_kept_where_accepted_and_downgraded_where_refused() {
        let tools = vec![weather_tool("a")];
        let base = ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(tools);
        let config = AnthropicConfig::new("k");
        let required = base.clone().with_tool_choice(ToolChoice::Required);
        assert_eq!(
            body_json(&config, &required)["tool_choice"],
            json!({"type": "any"})
        );
        let specific = base.clone().with_tool_choice(ToolChoice::Specific {
            name: "a".to_owned(),
        });
        assert_eq!(
            body_json(&config, &specific)["tool_choice"],
            json!({"type": "tool", "name": "a"})
        );
        let none = base.with_tool_choice(ToolChoice::None);
        assert_eq!(
            body_json(&config, &none)["tool_choice"],
            json!({"type": "none"})
        );
        for model in ["claude-opus-5-5", "claude-sonnet-5-5"] {
            let forced = required.clone().with_model(model);
            assert_eq!(
                body_json(&config, &forced)["tool_choice"],
                json!({"type": "auto"}),
                "{model}"
            );
        }
    }

    #[test]
    fn sampling_parameters_reach_only_models_that_accept_them() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_temperature(0.2)
            .with_top_p(0.5)
            .with_stop(vec!["END".to_owned(), String::new()]);
        let config = AnthropicConfig::new("k");
        let haiku = body_json(&config, &request);
        assert!(haiku.get("temperature").is_none());
        assert!(haiku.get("top_p").is_none());
        assert_eq!(haiku["stop_sequences"], json!(["END"]));

        let legacy = body_json(&config, &request.with_model("claude-haiku-4-5"));
        assert!((legacy["temperature"].as_f64().unwrap_or_default() - 0.2).abs() < 1e-6);
        assert!(
            legacy.get("top_p").is_none(),
            "Claude 4 refuses temperature and top_p together"
        );

        let top_p_only = ChatRequest::new(vec![ChatMessage::user("hi")])
            .with_top_p(0.5)
            .with_model("claude-sonnet-4-6");
        assert!(body_json(&config, &top_p_only).get("top_p").is_some());

        let provider = AnthropicProvider::with_client(config, reqwest::Client::new());
        assert!(!provider.capabilities().supports_temperature());
        let legacy_provider = provider.with_default_model("claude-sonnet-4-5");
        assert!(legacy_provider.capabilities().supports_temperature());
    }

    #[test]
    fn effort_thinking_and_max_tokens_come_from_config_or_request() {
        let config = AnthropicConfig::new("k")
            .with_effort(AnthropicEffort::Low)
            .with_thinking(AnthropicThinking::Disabled)
            .with_max_tokens(2048);
        let body = body_json(&config, &ChatRequest::new(vec![ChatMessage::user("hi")]));
        assert_eq!(body["output_config"], json!({"effort": "low"}));
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert_eq!(body["max_tokens"], 2048);
        let capped = body_json(
            &config,
            &ChatRequest::new(vec![ChatMessage::user("hi")]).with_max_tokens(64),
        );
        assert_eq!(capped["max_tokens"], 64);
        let streamed = serde_json::to_value(build_body(&config, &ChatRequest::new(vec![]), true))
            .expect("serialises"); // Safe: test assertion
        assert_eq!(streamed["stream"], true);
    }

    #[test]
    fn a_response_maps_text_tool_use_usage_and_stop_reason() {
        let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-5-5",
            "content":[
                {"type":"thinking","thinking":"","signature":"sig"},
                {"type":"text","text":"Let me "},
                {"type":"text","text":"check."},
                {"type":"tool_use","id":"toolu_9","name":"get_weather","input":{"city":"Paris"}}
            ],
            "stop_reason":"tool_use",
            "usage":{"input_tokens":12,"output_tokens":40,"cache_read_input_tokens":3000,"cache_creation_input_tokens":500}}"#;
        let parsed: MessagesResponse = serde_json::from_str(body).expect("parses"); // Safe: test assertion
        let response = into_chat_response(parsed, "claude-haiku-5-5");
        assert_eq!(response.content, "Let me check.");
        assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
        let calls = response.tool_calls.expect("tool calls"); // Safe: test assertion
        assert_eq!(calls[0].id, "toolu_9");
        assert_eq!(calls[0].function_name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Paris");
        let usage = response.usage.expect("usage"); // Safe: test assertion
        assert_eq!(
            usage.prompt_tokens, 3512,
            "gross: fresh + cache read + cache write"
        );
        assert_eq!(usage.completion_tokens, 40);
        assert_eq!(usage.total_tokens, 3552);
        assert_eq!(usage.cached_read_tokens, Some(3000));
        assert_eq!(usage.cached_write_tokens, Some(500));
        assert_eq!(usage.reasoning_tokens, None);
    }

    #[test]
    fn stop_reasons_map_to_the_crate_vocabulary() {
        let cases = [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("tool_use", "tool_calls"),
            ("refusal", "refusal"),
            ("pause_turn", "pause_turn"),
        ];
        for (wire, mapped) in cases {
            assert_eq!(map_stop_reason(wire), mapped, "{wire}");
        }
    }

    #[test]
    fn an_empty_refusal_is_an_empty_completion_marked_refusal() {
        let body = r#"{"content":[],"stop_reason":"refusal","stop_details":{"category":"cyber"},
            "usage":{"input_tokens":20,"output_tokens":0}}"#;
        let parsed: MessagesResponse = serde_json::from_str(body).expect("parses"); // Safe: test assertion
        let response = into_chat_response(parsed, "claude-haiku-5-5");
        assert_eq!(response.finish_reason.as_deref(), Some("refusal"));
        assert!(response.tool_calls.is_none());
        assert!(
            is_empty_completion(&response),
            "a strict chain moves past it"
        );
    }

    #[test]
    fn a_refusal_after_partial_output_discards_it() {
        let body = r#"{"content":[
                {"type":"text","text":"Here is how you would start: first"},
                {"type":"tool_use","id":"toolu_1","name":"search","input":{}}
            ],
            "stop_reason":"refusal","stop_details":{"category":"general_harms"},
            "usage":{"input_tokens":20,"output_tokens":9}}"#;
        let parsed: MessagesResponse = serde_json::from_str(body).expect("parses"); // Safe: test assertion
        let response = into_chat_response(parsed, "claude-haiku-5-5");
        assert_eq!(response.content, "", "the partial answer is discarded");
        assert!(response.tool_calls.is_none());
        assert_eq!(response.finish_reason.as_deref(), Some("refusal"));
        assert_eq!(
            response.usage.as_ref().map(|u| u.completion_tokens),
            Some(9),
            "the billed usage is kept"
        );
        assert!(
            is_empty_completion(&response),
            "a strict chain moves past it"
        );
    }

    #[test]
    fn a_result_for_another_call_leaves_neither_half_dangling() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("weather?"),
            assistant_calling(vec![call(
                "toolu_x",
                "get_weather",
                json!({"city": "Paris"}),
            )]),
            ChatMessage::tool("get_weather", "toolu_y", "sunny"),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [{"type": "text",
                    "text": "Called tool get_weather with {\"city\":\"Paris\"}; no result was recorded."}]},
                {"role": "user", "content": [{"type": "text", "text": "Result of tool get_weather: sunny"}]}
            ])
        );
    }

    #[test]
    fn an_unanswered_call_becomes_text_and_its_answered_sibling_stays_paired() {
        let request = ChatRequest::new(vec![
            ChatMessage::user("plan"),
            assistant_calling(vec![
                call("toolu_a", "load", json!({})),
                call("toolu_b", "fit", json!({})),
            ]),
            ChatMessage::tool("load", "toolu_a", "42"),
            ChatMessage::user("go on"),
        ]);
        let body = body_json(&AnthropicConfig::new("k"), &request);
        assert_eq!(
            body["messages"][1]["content"],
            json!([
                {"type": "tool_use", "id": "toolu_a", "name": "load", "input": {}},
                {"type": "text", "text": "Called tool fit with {}; no result was recorded."}
            ])
        );
        assert_eq!(
            body["messages"][2]["content"],
            json!([
                {"type": "tool_result", "tool_use_id": "toolu_a", "content": "42"},
                {"type": "text", "text": "go on"}
            ])
        );
        let trailing = body_json(
            &AnthropicConfig::new("k"),
            &ChatRequest::new(vec![
                ChatMessage::user("plan"),
                assistant_calling(vec![call("toolu_c", "load", json!({}))]),
            ]),
        );
        assert_eq!(
            trailing["messages"][1]["content"][0]["type"], "text",
            "a call with no turn after it has no result either"
        );
    }

    #[test]
    fn thinking_and_effort_reach_only_models_that_accept_them() {
        let body_for = |config: &AnthropicConfig, model: &str| {
            body_json(
                config,
                &ChatRequest::new(vec![ChatMessage::user("hi")]).with_model(model),
            )
        };
        let off_high = AnthropicConfig::new("k")
            .with_thinking(AnthropicThinking::Disabled)
            .with_effort(AnthropicEffort::High);
        let haiku = body_for(&off_high, "claude-haiku-5-5");
        assert_eq!(haiku["thinking"], json!({"type": "disabled"}));
        assert_eq!(haiku["output_config"], json!({"effort": "high"}));
        for always_on in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
            let body = body_for(&off_high, always_on);
            assert!(body.get("thinking").is_none(), "{always_on}");
            assert_eq!(
                body["output_config"],
                json!({"effort": "high"}),
                "{always_on}"
            );
        }

        let off_max = AnthropicConfig::new("k")
            .with_thinking(AnthropicThinking::Disabled)
            .with_effort(AnthropicEffort::Max);
        for model in ["claude-haiku-5-5", "claude-opus-5"] {
            let body = body_for(&off_max, model);
            assert!(body.get("thinking").is_none(), "{model}");
            assert_eq!(body["output_config"], json!({"effort": "max"}), "{model}");
        }

        let adaptive_low = AnthropicConfig::new("k")
            .with_thinking(AnthropicThinking::Adaptive)
            .with_effort(AnthropicEffort::Low);
        for legacy in [
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-latest",
        ] {
            let body = body_for(&adaptive_low, legacy);
            assert!(body.get("thinking").is_none(), "{legacy}");
            assert!(body.get("output_config").is_none(), "{legacy}");
        }
        let opus_4_5 = body_for(&adaptive_low, "claude-opus-4-5");
        assert!(opus_4_5.get("thinking").is_none());
        assert_eq!(opus_4_5["output_config"], json!({"effort": "low"}));
        let opus_4_6 = body_for(&adaptive_low, "claude-opus-4-6");
        assert_eq!(opus_4_6["thinking"], json!({"type": "adaptive"}));
        assert_eq!(opus_4_6["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn a_quota_refusal_leaves_the_tier_and_an_overload_is_retried() {
        let rate_limited = map_api_error(
            StatusCode::TOO_MANY_REQUESTS,
            &error_body("rate_limit_error", "slow down"),
            Some(3),
            "m",
        );
        assert!(!attempt_error(StatusCode::TOO_MANY_REQUESTS, rate_limited).retryable);
        let spent = map_api_error(
            StatusCode::PAYMENT_REQUIRED,
            &error_body("billing_error", "Your credit balance is too low"),
            None,
            "m",
        );
        assert!(!attempt_error(StatusCode::PAYMENT_REQUIRED, spent).retryable);
        // A quota type in the body decides it, whatever the status
        let typed = map_api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &error_body("rate_limit_error", "slow down"),
            None,
            "m",
        );
        assert_eq!(typed.kind, ErrorKind::RateLimit);
        assert!(!attempt_error(StatusCode::SERVICE_UNAVAILABLE, typed).retryable);
        let overloaded = map_api_error(
            overloaded_status(),
            &error_body("overloaded_error", "Overloaded"),
            None,
            "m",
        );
        assert!(attempt_error(overloaded_status(), overloaded).retryable);
    }

    fn error_body(kind: &str, message: &str) -> String {
        json!({"type": "error", "error": {"type": kind, "message": message}}).to_string()
    }

    #[test]
    fn a_402_spent_credit_is_a_quota_refusal_not_retried_in_place() {
        let err = map_api_error(
            StatusCode::PAYMENT_REQUIRED,
            &error_body("billing_error", "Your credit balance is too low"),
            None,
            "claude-haiku-5-5",
        );
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.starts_with("anthropic: "), "{}", err.message);
        assert!(
            err.message.contains("credit balance exhausted"),
            "{}",
            err.message
        );
        assert!(!client::is_retryable_status(402));
    }

    #[test]
    fn a_429_carries_the_retry_after_wait() {
        let err = map_api_error(
            StatusCode::TOO_MANY_REQUESTS,
            &error_body(
                "rate_limit_error",
                "Number of request tokens has exceeded your rate limit",
            ),
            Some(12),
            "claude-haiku-5-5",
        );
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(
            err.message.contains("try again in 12 seconds"),
            "{}",
            err.message
        );

        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        assert_eq!(retry_after_header(&headers), Some(7));
        headers.insert(RETRY_AFTER, HeaderValue::from_static("0.2"));
        assert_eq!(retry_after_header(&headers), Some(1));
        assert_eq!(retry_after_header(&HeaderMap::new()), None);
    }

    #[test]
    fn a_529_is_a_transient_external_service_failure() {
        let status = overloaded_status();
        assert_eq!(status.as_u16(), 529);
        let err = map_api_error(
            status,
            &error_body("overloaded_error", "Overloaded"),
            None,
            "m",
        );
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.kind.is_transient());
        assert!(client::is_retryable_status(529));
    }

    #[test]
    fn auth_validation_size_and_model_errors_map_to_their_kinds() {
        let auth = map_api_error(
            StatusCode::UNAUTHORIZED,
            &error_body("authentication_error", "invalid x-api-key"),
            None,
            "m",
        );
        assert_eq!(auth.kind, ErrorKind::AuthFailure);
        let forbidden = map_api_error(
            StatusCode::FORBIDDEN,
            &error_body("permission_error", "no"),
            None,
            "m",
        );
        assert_eq!(forbidden.kind, ErrorKind::AuthFailure);
        let invalid = map_api_error(
            StatusCode::BAD_REQUEST,
            &error_body("invalid_request_error", "messages: roles must alternate"),
            None,
            "m",
        );
        assert_eq!(invalid.kind, ErrorKind::InvalidRequest);
        assert!(invalid.message.contains("roles must alternate"));
        let too_long = map_api_error(
            StatusCode::BAD_REQUEST,
            &error_body(
                "invalid_request_error",
                "prompt is too long: 1200000 tokens > 1000000 maximum",
            ),
            None,
            "m",
        );
        assert_eq!(too_long.kind, ErrorKind::ContextLength);
        let too_large = map_api_error(StatusCode::PAYLOAD_TOO_LARGE, "<html>", None, "m");
        assert_eq!(too_large.kind, ErrorKind::ContextLength);
        let missing = map_api_error(
            StatusCode::NOT_FOUND,
            &error_body("not_found_error", "model: x"),
            None,
            "claude-x",
        );
        assert_eq!(missing.kind, ErrorKind::ModelUnavailable);
        let opaque = map_api_error(StatusCode::INTERNAL_SERVER_ERROR, "<html>", None, "m");
        assert_eq!(opaque.kind, ErrorKind::ExternalService);
        assert!(opaque.message.contains("HTTP 500"));
    }

    #[test]
    fn no_error_message_carries_the_api_key() {
        let key = "sk-ant-api03-very-secret";
        let provider =
            AnthropicProvider::with_client(AnthropicConfig::new(key), reqwest::Client::new());
        for status in [401_u16, 402, 403, 404, 413, 429, 500, 529] {
            let status = StatusCode::from_u16(status).expect("valid status"); // Safe: test assertion
            let err = map_api_error(
                status,
                &error_body("x", "refused"),
                Some(3),
                &provider.config.model,
            );
            assert!(!err.message.contains(key), "{status}: {}", err.message);
        }
        let invalid_header = AnthropicProvider::with_client(
            AnthropicConfig::new("bad\nkey-secret"),
            reqwest::Client::new(),
        );
        let err = invalid_header
            .auth_headers()
            .expect_err("a newline cannot ride in a header"); // Safe: test assertion
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(!err.message.contains("key-secret"));
    }

    #[test]
    fn a_canned_event_transcript_streams_text_then_the_stop() {
        let transcript = [
            r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}"#,
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}"#,
            r#"{"type":"content_block_stop","index":2}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let chunks: Vec<StreamChunk> = transcript
            .iter()
            .filter_map(|frame| parse_stream_data(frame, "claude-haiku-5-5"))
            .map(|chunk| chunk.expect("no error in the transcript")) // Safe: test assertion
            .collect();
        let deltas: Vec<&str> = chunks.iter().map(|c| c.delta.as_str()).collect();
        assert_eq!(deltas, ["Hello", " world", ""]);
        let last = chunks.last().expect("a final chunk"); // Safe: test assertion
        assert!(last.is_final);
        assert_eq!(last.finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn the_stream_parser_runs_over_raw_sse_bytes() {
        use tokio_stream::StreamExt;
        let raw = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n\
                   event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
                   event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\n\
                   event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let bytes = tokio_stream::iter(vec![Ok::<_, io::Error>(raw.as_bytes().to_vec())]);
        let chunks: Vec<StreamChunk> = create_sse_stream(
            bytes,
            |json| parse_stream_data(json, "claude-haiku-5-5"),
            PROVIDER_NAME,
        )
        .map(|c| c.expect("ok")) // Safe: test assertion
        .collect()
        .await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].delta, "Hi");
        assert_eq!(chunks[1].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn a_mid_stream_error_event_ends_the_stream_with_its_kind() {
        let overloaded = parse_stream_data(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            "claude-haiku-5-5",
        )
        .expect("an item") // Safe: test assertion
        .expect_err("an error"); // Safe: test assertion
        assert_eq!(overloaded.kind, ErrorKind::ExternalService);
        let billing = parse_stream_data(
            r#"{"type":"error","error":{"type":"billing_error","message":"credit"}}"#,
            "claude-haiku-5-5",
        )
        .expect("an item") // Safe: test assertion
        .expect_err("an error"); // Safe: test assertion
        assert_eq!(billing.kind, ErrorKind::RateLimit);
        let refusal = parse_stream_data(
            r#"{"type":"message_delta","delta":{"stop_reason":"refusal"}}"#,
            "claude-haiku-5-5",
        )
        .expect("an item") // Safe: test assertion
        .expect("a final chunk"); // Safe: test assertion
        assert_eq!(refusal.finish_reason.as_deref(), Some("refusal"));
        assert!(parse_stream_data("not json", "claude-haiku-5-5").is_some_and(|r| r.is_err()));
        let missing = parse_stream_data(
            r#"{"type":"error","error":{"type":"not_found_error","message":"model: claude-x"}}"#,
            "claude-x",
        )
        .expect("an item") // Safe: test assertion
        .expect_err("an error"); // Safe: test assertion
        assert_eq!(missing.kind, ErrorKind::ModelUnavailable);
        assert_eq!(
            missing.message, "Model \"claude-x\" is not available",
            "a mid-stream not-found names the model the stream was opened for"
        );
    }

    const STREAM_MODEL: &str = "claude-opus-5-5";

    /// Every event the accumulator produces for `frames`, in order
    fn accumulate(frames: &[&str]) -> Vec<Result<ToolStreamEvent, RunnerError>> {
        let mut accumulator = ToolStreamAccumulator::new(STREAM_MODEL);
        frames.iter().filter_map(|f| accumulator.feed(f)).collect()
    }

    /// The text deltas and the final response of a stream that succeeded
    fn deltas_and_done(
        events: Vec<Result<ToolStreamEvent, RunnerError>>,
    ) -> (Vec<String>, ChatResponse) {
        let mut deltas = Vec::new();
        let mut done = None;
        for event in events {
            match event.expect("no error in the transcript") {
                // Safe: test assertion
                ToolStreamEvent::TextDelta(text) => deltas.push(text),
                ToolStreamEvent::Done(response) => {
                    assert!(done.is_none(), "exactly one Done");
                    done = Some(response);
                }
            }
        }
        (deltas, done.expect("a Done at message_stop")) // Safe: test assertion
    }

    /// What `complete()` returns for the equivalent non-streaming body
    fn non_streaming(body: Value) -> ChatResponse {
        let parsed: MessagesResponse = serde_json::from_value(body).expect("a messages response"); // Safe: test assertion
        into_chat_response(parsed, STREAM_MODEL)
    }

    fn same(streamed: &ChatResponse, complete: &ChatResponse) {
        assert_eq!(
            serde_json::to_value(streamed).expect("serializes"), // Safe: test assertion
            serde_json::to_value(complete).expect("serializes"), // Safe: test assertion
            "the streamed Done is complete()'s response"
        );
    }

    const START: &str = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5","stop_reason":null,"usage":{"input_tokens":25,"cache_read_input_tokens":4000,"cache_creation_input_tokens":300,"output_tokens":1}}}"#;

    #[test]
    fn a_text_only_tool_stream_yields_its_deltas_then_completes_response() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"the athlete asks"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Easy spin"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" today."}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":40}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let (deltas, done) = deltas_and_done(events);
        assert_eq!(deltas, ["Easy spin", " today."]);
        assert!(done.tool_calls.is_none());
        same(
            &done,
            &non_streaming(json!({
                "content": [
                    {"type": "thinking", "thinking": "the athlete asks", "signature": "sig"},
                    {"type": "text", "text": "Easy spin today."}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 25, "cache_read_input_tokens": 4000,
                          "cache_creation_input_tokens": 300, "output_tokens": 40}
            })),
        );
    }

    #[test]
    fn text_then_tool_use_streams_the_text_and_closes_with_the_parsed_call() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me pull your rides."}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_activities","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"limit\": 2"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"0, \"sport\": \"ride\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":61}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let (deltas, done) = deltas_and_done(events);
        assert_eq!(deltas, ["Let me pull your rides."]);
        assert_eq!(done.finish_reason.as_deref(), Some("tool_calls"));
        let calls = done.tool_calls.as_ref().expect("a tool call"); // Safe: test assertion
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].arguments, json!({"limit": 20, "sport": "ride"}));
        same(
            &done,
            &non_streaming(json!({
                "content": [
                    {"type": "text", "text": "Let me pull your rides."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_activities",
                     "input": {"limit": 20, "sport": "ride"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 25, "cache_read_input_tokens": 4000,
                          "cache_creation_input_tokens": 300, "output_tokens": 61}
            })),
        );
    }

    #[test]
    fn parallel_tool_uses_without_text_keep_their_order_and_a_no_argument_call_gets_an_object() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_profile","input":{}}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_activity","input":{}}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"id\":"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"é-42\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let (deltas, done) = deltas_and_done(events);
        assert_eq!(deltas, Vec::<String>::new());
        assert_eq!(done.content, "");
        let calls = done.tool_calls.as_ref().expect("two calls"); // Safe: test assertion
        let names: Vec<&str> = calls.iter().map(|c| c.function_name.as_str()).collect();
        assert_eq!(names, ["get_profile", "get_activity"]);
        assert_eq!(calls[0].arguments, json!({}));
        assert_eq!(calls[1].arguments, json!({"id": "é-42"}));
        same(
            &done,
            &non_streaming(json!({
                "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "get_profile", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "get_activity", "input": {"id": "é-42"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 25, "cache_read_input_tokens": 4000,
                          "cache_creation_input_tokens": 300, "output_tokens": 30}
            })),
        );
    }

    #[test]
    fn streamed_usage_is_gross_with_both_cache_counts() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
            r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":9}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let usage = deltas_and_done(events).1.usage.expect("usage"); // Safe: test assertion
        assert_eq!(usage.prompt_tokens, 25 + 4000 + 300);
        assert_eq!(
            usage.completion_tokens, 9,
            "the last cumulative output count wins"
        );
        assert_eq!(usage.total_tokens, 4325 + 9);
        assert_eq!(usage.cached_read_tokens, Some(4000));
        assert_eq!(usage.cached_write_tokens, Some(300));

        let without_start = accumulate(&[
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":3,"output_tokens":2}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let usage = deltas_and_done(without_start).1.usage.expect("usage"); // Safe: test assertion
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 2));
        assert_eq!(usage.cached_read_tokens, None, "unreported is not zero");

        let no_usage = accumulate(&[r#"{"type":"message_stop"}"#]);
        assert!(deltas_and_done(no_usage).1.usage.is_none());
    }

    #[test]
    fn a_tool_stream_refusal_discards_the_streamed_output_in_its_done() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Sure, the first step"}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"category":"general_harms"}},"usage":{"output_tokens":6}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let (deltas, done) = deltas_and_done(events);
        assert_eq!(
            deltas,
            ["Sure, the first step"],
            "a delta cannot be taken back"
        );
        assert_eq!(done.finish_reason.as_deref(), Some("refusal"));
        assert_eq!(
            done.content, "",
            "the Done tells the consumer to discard it"
        );
        assert!(is_empty_completion(&done));
    }

    #[test]
    fn tool_stream_errors_end_it_without_a_done() {
        let events = accumulate(&[
            START,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Par"}}"#,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ]);
        assert!(matches!(events[0], Ok(ToolStreamEvent::TextDelta(_))));
        let err = events[1].as_ref().expect_err("the error event"); // Safe: test assertion
        assert_eq!(err.kind, ErrorKind::ExternalService);

        let bad_input = accumulate(&[
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_activities","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"limit\":"}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
            r#"{"type":"message_stop"}"#,
        ]);
        let err = bad_input[0].as_ref().expect_err("truncated tool input"); // Safe: test assertion
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("get_activities"), "{}", err.message);

        assert!(accumulate(&["not json"])[0].is_err());
    }

    /// Every item [`create_tool_stream`] yields over `chunks` of raw bytes
    async fn tool_stream_items(
        chunks: Vec<&'static [u8]>,
    ) -> Vec<Result<ToolStreamEvent, RunnerError>> {
        use tokio_stream::StreamExt;
        let bytes = tokio_stream::iter(chunks.into_iter().map(Ok::<_, io::Error>));
        create_tool_stream(bytes, STREAM_MODEL).collect().await
    }

    #[tokio::test]
    async fn the_tool_stream_runs_over_raw_sse_bytes_split_anywhere() {
        let raw: &'static str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n\
             event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
             event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Récup\"}}\n\n\
             event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_sleep\",\"input\":{}}}\n\n\
             event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"days\\\": 7}\"}}\n\n\
             event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n\
             event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let bytes = raw.as_bytes();
        // Split inside the two bytes of 'é' and mid-line elsewhere
        let accent = bytes.iter().position(|b| *b == 0xC3).expect("an é") + 1; // Safe: test assertion
        let items =
            tool_stream_items(vec![&bytes[..37], &bytes[37..accent], &bytes[accent..]]).await;
        assert_eq!(items.len(), 2, "one delta, then the Done, then the end");
        let (deltas, done) = deltas_and_done(items);
        assert_eq!(deltas, ["Récup"]);
        assert_eq!(done.content, "Récup");
        let call = &done.tool_calls.expect("a call")[0]; // Safe: test assertion
        assert_eq!(
            (call.function_name.as_str(), &call.arguments),
            ("get_sleep", &json!({"days": 7}))
        );
        assert_eq!(done.usage.map(|u| u.total_tokens), Some(16));
    }

    #[tokio::test]
    async fn the_tool_stream_ends_after_an_error_and_on_a_close_before_message_stop() {
        let errored = tool_stream_items(vec![
            b"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n\
              data: {\"type\":\"message_stop\"}\n\n",
        ])
        .await;
        assert_eq!(
            errored.len(),
            1,
            "nothing follows an error, not even a Done"
        );
        assert!(errored[0].is_err());

        let truncated = tool_stream_items(vec![
            b"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
              data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Half\"}}\n\n",
        ])
        .await;
        assert_eq!(truncated.len(), 2);
        assert!(matches!(&truncated[0], Ok(ToolStreamEvent::TextDelta(t)) if t == "Half"));
        let err = truncated[1].as_ref().expect_err("a truncated stream fails"); // Safe: test assertion
        assert!(err.message.contains("message_stop"), "{}", err.message);
    }

    #[test]
    fn numbered_keys_are_read_until_the_first_gap() {
        let env: HashMap<&str, &str> = HashMap::from([
            ("ANTHROPIC_API_KEY", "key-1"),
            ("ANTHROPIC_API_KEY_2", "key-2"),
            ("ANTHROPIC_API_KEY_3", "key-3"),
            ("ANTHROPIC_API_KEY_5", "key-5"),
        ]);
        let keys = numbered_api_keys(|name| env.get(name).map(|v| (*v).to_owned()));
        assert_eq!(keys, ["key-1", "key-2", "key-3"]);

        let blank: HashMap<&str, &str> = HashMap::from([
            ("ANTHROPIC_API_KEY", "key-1"),
            ("ANTHROPIC_API_KEY_2", "  "),
            ("ANTHROPIC_API_KEY_3", "key-3"),
        ]);
        let keys = numbered_api_keys(|name| blank.get(name).map(|v| (*v).to_owned()));
        assert_eq!(keys, ["key-1"]);
        assert_eq!(numbered_api_keys(|_| None), Vec::<String>::new());
    }

    #[test]
    fn effort_and_thinking_parse_their_wire_values() {
        assert_eq!(
            AnthropicEffort::parse("XHigh"),
            Some(AnthropicEffort::XHigh)
        );
        assert_eq!(
            AnthropicEffort::parse("medium"),
            Some(AnthropicEffort::Medium)
        );
        assert_eq!(AnthropicEffort::parse("extreme"), None);
        assert_eq!(
            AnthropicThinking::parse(" adaptive "),
            Some(AnthropicThinking::Adaptive)
        );
        assert_eq!(AnthropicThinking::parse("enabled"), None);
    }

    mod env_tests {
        use std::sync::{Mutex, PoisonError};

        use super::*;

        static ENV_MUTEX: Mutex<()> = Mutex::new(());

        const VARS: &[&str] = &[
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY_2",
            "ANTHROPIC_API_KEY_3",
            "ANTHROPIC_DEFAULT_MODEL",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MAX_TOKENS",
            "ANTHROPIC_EFFORT",
            "ANTHROPIC_THINKING",
            "ANTHROPIC_MAX_RETRIES",
        ];

        fn with_env<T>(values: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
            let guard = ENV_MUTEX.lock().unwrap_or_else(PoisonError::into_inner);
            let saved: Vec<(&str, Option<String>)> = VARS
                .iter()
                .map(|name| (*name, env::var(name).ok()))
                .collect();
            for name in VARS {
                env::remove_var(name);
            }
            for (name, value) in values {
                env::set_var(name, value);
            }
            let result = f();
            for (name, value) in saved {
                match value {
                    Some(v) => env::set_var(name, v),
                    None => env::remove_var(name),
                }
            }
            drop(guard);
            result
        }

        #[test]
        fn a_missing_key_is_a_config_error() {
            let err = with_env(&[], AnthropicConfig::from_env).expect_err("no key"); // Safe: test assertion
            assert_eq!(err.kind, ErrorKind::Config);
            let err = with_env(&[], AnthropicConfig::pool_from_env).expect_err("no key"); // Safe: test assertion
            assert_eq!(err.kind, ErrorKind::Config);
        }

        #[test]
        fn from_env_reads_every_setting() {
            let config = with_env(
                &[
                    ("ANTHROPIC_API_KEY", "key-1"),
                    ("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-5-5"),
                    ("ANTHROPIC_BASE_URL", "https://proxy.example"),
                    ("ANTHROPIC_MAX_TOKENS", "4096"),
                    ("ANTHROPIC_EFFORT", "high"),
                    ("ANTHROPIC_THINKING", "adaptive"),
                    ("ANTHROPIC_MAX_RETRIES", "1"),
                ],
                AnthropicConfig::from_env,
            )
            .expect("valid env"); // Safe: test assertion
            assert_eq!(config.api_key, "key-1");
            assert_eq!(config.model, "claude-sonnet-5-5");
            assert_eq!(config.base_url, "https://proxy.example");
            assert_eq!(config.max_tokens, 4096);
            assert_eq!(config.effort, Some(AnthropicEffort::High));
            assert_eq!(config.thinking, Some(AnthropicThinking::Adaptive));
            assert_eq!(config.retry.max_retries, 1);
        }

        #[test]
        fn invalid_values_are_config_errors() {
            for bad in [
                &[("ANTHROPIC_API_KEY", "k"), ("ANTHROPIC_EFFORT", "extreme")][..],
                &[
                    ("ANTHROPIC_API_KEY", "k"),
                    ("ANTHROPIC_THINKING", "enabled"),
                ][..],
                &[("ANTHROPIC_API_KEY", "k"), ("ANTHROPIC_MAX_TOKENS", "0")][..],
                &[
                    ("ANTHROPIC_API_KEY", "k"),
                    ("ANTHROPIC_THINKING", "disabled"),
                    ("ANTHROPIC_EFFORT", "max"),
                ][..],
            ] {
                let err = with_env(bad, AnthropicConfig::from_env).expect_err("invalid"); // Safe: test assertion
                assert_eq!(err.kind, ErrorKind::Config, "{bad:?}");
            }
        }

        #[test]
        fn pool_from_env_shares_every_setting_but_the_key() {
            let configs = with_env(
                &[
                    ("ANTHROPIC_API_KEY", "key-1"),
                    ("ANTHROPIC_API_KEY_2", "key-2"),
                    ("ANTHROPIC_EFFORT", "low"),
                ],
                AnthropicConfig::pool_from_env,
            )
            .expect("valid env"); // Safe: test assertion
            let keys: Vec<&str> = configs.iter().map(|c| c.api_key.as_str()).collect();
            assert_eq!(keys, ["key-1", "key-2"]);
            assert!(configs
                .iter()
                .all(|c| c.effort == Some(AnthropicEffort::Low)));
            assert!(configs.iter().all(|c| c.model == "claude-haiku-5-5"));
        }
    }
}
