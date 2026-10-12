// ABOUTME: Per-token price table for every provider this crate can run, keyed on LlmProvider::name()
// ABOUTME: Longest-prefix model lookup, cache read/write multipliers, reasoning at the output rate
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Model Pricing
//!
//! Compile-time pricing data for the providers this crate runs, keyed on the
//! exact string each one's [`LlmProvider::name`](crate::types::LlmProvider::name)
//! reports. [`MetricsProvider`](crate::metrics::MetricsProvider) prices every
//! call through it; a consumer that layers operator overrides on top builds
//! its own lookup from [`lookup_pricing`], [`cost_from_pricing`] and
//! [`zero_cost_for_unpriced`], which are public for that purpose.
//!
//! ## Cache accounting
//!
//! A prompt token can bill at three rates, so the pricing entry carries the
//! multipliers rather than a single global discount: fresh input at 1.0×, a
//! cache *read* at [`ModelPricing::cache_read_multiplier`] (Anthropic 0.10×,
//! `OpenAI` 0.50×, `Gemini` 0.25×), and a cache *write* at
//! [`ModelPricing::cache_write_multiplier`] — which is a **premium** on
//! Anthropic (1.25×), not a discount. Reasoning tokens bill at the output
//! rate and are additive to the completion count, because every provider that
//! reports them separately excludes them from it.
//!
//! Counts arrive on [`TokenUsage`](crate::types::TokenUsage), are carried
//! through [`TokenCounts`], and are priced by [`calculate_cost_for`].

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Default fraction of the input rate at which a context-cache *read* bills.
///
/// Matches `Gemini`'s advertised 25% cache-read discount, and applies to any
/// model that does not override it via [`ModelPricing::with_cache_rates`].
/// A provider with no cache at all never reports a read, so the multiplier
/// is inert rather than wrong for those entries.
pub const DEFAULT_CACHE_READ_RATE: f64 = 0.25;

/// Default fraction of the input rate at which a context-cache *write* bills.
///
/// `1.0` — a provider that does not price cache creation separately bills
/// those tokens as ordinary input. Anthropic is the exception and overrides
/// it; see [`ModelPricing::with_cache_rates`].
pub const DEFAULT_CACHE_WRITE_RATE: f64 = 1.0;

/// Per-model pricing rates in USD per million tokens, plus the two
/// cache multipliers that turn a raw token count into a billed one.
///
/// Cache economics are per-provider, not universal: Anthropic reads at
/// 0.10× and *charges a premium* to write at 1.25×, `OpenAI` reads at 0.50×
/// and does not bill writes separately, `Gemini` reads at 0.25×. A single
/// flat discount mispriced all three, so the rates travel with the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    /// USD per 1 million input (prompt) tokens
    pub input_per_million: f64,
    /// USD per 1 million output (completion) tokens
    pub output_per_million: f64,
    /// Fraction of `input_per_million` at which cache-read tokens bill.
    #[serde(default = "default_cache_read_rate")]
    pub cache_read_multiplier: f64,
    /// Fraction of `input_per_million` at which cache-write tokens bill.
    #[serde(default = "default_cache_write_rate")]
    pub cache_write_multiplier: f64,
    /// A second rate card for long prompts, when the model has one: a
    /// request whose gross prompt exceeds its threshold bills every token —
    /// input, cache and output — at the long-prompt rates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long_prompt: Option<LongPromptRates>,
}

/// The rate card a model switches to when a request's prompt is long.
///
/// Claude Haiku 5.5 bills \$0.10 / \$0.50 per million tokens for a prompt of
/// 100K tokens or fewer and \$0.50 / \$2.50 above that; the cache
/// multipliers apply to the long-prompt input rate the same way.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LongPromptRates {
    /// The largest gross prompt (fresh + cache read + cache write) that still
    /// bills on the base card
    pub above_prompt_tokens: i64,
    /// USD per 1 million input tokens above the threshold
    pub input_per_million: f64,
    /// USD per 1 million output tokens above the threshold
    pub output_per_million: f64,
}

/// Serde default for [`ModelPricing::cache_read_multiplier`], so an operator
/// override stored before the multipliers existed still deserializes.
fn default_cache_read_rate() -> f64 {
    DEFAULT_CACHE_READ_RATE
}

/// Serde default for [`ModelPricing::cache_write_multiplier`], so an operator
/// override stored before the multipliers existed still deserializes.
fn default_cache_write_rate() -> f64 {
    DEFAULT_CACHE_WRITE_RATE
}

impl ModelPricing {
    /// Rates for a model that uses the default cache multipliers.
    #[must_use]
    pub const fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            input_per_million,
            output_per_million,
            cache_read_multiplier: DEFAULT_CACHE_READ_RATE,
            cache_write_multiplier: DEFAULT_CACHE_WRITE_RATE,
            long_prompt: None,
        }
    }

    /// Bill a request whose gross prompt exceeds `above_prompt_tokens` at
    /// the given input / output rates instead of the base card.
    #[must_use]
    pub const fn with_long_prompt_rates(
        mut self,
        above_prompt_tokens: i64,
        input_per_million: f64,
        output_per_million: f64,
    ) -> Self {
        self.long_prompt = Some(LongPromptRates {
            above_prompt_tokens,
            input_per_million,
            output_per_million,
        });
        self
    }

    /// The (input, output) rates per million tokens for a request with a gross prompt
    /// of `prompt_tokens`: the long-prompt card above its threshold, the base
    /// card otherwise.
    #[must_use]
    pub fn rates_for_prompt(&self, prompt_tokens: i64) -> (f64, f64) {
        match self.long_prompt {
            Some(long) if prompt_tokens > long.above_prompt_tokens => {
                (long.input_per_million, long.output_per_million)
            }
            _ => (self.input_per_million, self.output_per_million),
        }
    }

    /// Override the cache read/write multipliers for a provider that prices
    /// them differently — Anthropic at `(0.10, 1.25)`, `OpenAI` at `(0.50, 1.0)`.
    #[must_use]
    pub const fn with_cache_rates(mut self, read: f64, write: f64) -> Self {
        self.cache_read_multiplier = read;
        self.cache_write_multiplier = write;
        self
    }
}

/// Providers whose per-token cost is genuinely \$0 by design, so a \$0 billed
/// cost is *correct* rather than a missing-price undercount.
///
/// Two classes share this property:
///
/// - **Self-hosted** runtimes (Ollama / vLLM / `LocalAI` / generic local
///   `OpenAI`-compatible endpoints): the operator pays for the hardware, not
///   per token, so the metered per-token cost is zero.
/// - **Flat-rate subscription** CLI/agent runners (GitHub Copilot, Cursor,
///   `OpenCode`, Codex, Goose, Cline, Continue, Warp, Kiro, Kilo): billed by
///   the upstream subscription, not per token.
///
/// Entries are the exact machine `name()` string each provider reports (see
/// [`LlmProvider::name`](crate::types::LlmProvider::name)) — hyphenated for
/// some runners and underscored for others. A model under one of these
/// prefixes resolves to \$0 *without* the missing-price warning, so cost
/// dashboards do not flag it as an undercount.
/// `copilot_headless`, `copilot_sdk` and `claude-code` are deliberately
/// absent here: they carry real per-token `PRICING_TABLE` entries because
/// their Anthropic pass-through usage is metered.
pub const NOT_PER_TOKEN_METERED_PROVIDERS: &[&str] = &[
    // Self-hosted OpenAI-compatible runtimes.
    "local",
    "ollama",
    "vllm",
    "localai",
    // Flat-rate subscription CLI / agent runners (machine names as reported
    // by each runner's name()).
    "cursor-agent",
    "opencode",
    "codex",
    "goose",
    "cline",
    "continue",
    "warp_cli",
    "kiro",
    "kilo",
    "copilot",
    // The browser-driven Claude.ai web UI: a subscription, not per-token.
    "claude_web",
];

/// `OpenRouter` exposes 200+ models with per-model pricing, so a static table
/// can never be exhaustive. An unknown `OpenRouter` model is therefore logged
/// as *model-specific price unset* (a deliberate, non-alarming message) rather
/// than counted as a silent undercount alongside genuinely-unexpected misses.
const OPENROUTER_PROVIDER: &str = "openrouter";

/// Compile-time pricing table: `(provider, model_prefix, pricing)`
///
/// The provider key is the exact `name()` string a provider reports
/// ([`LlmProvider::name`](crate::types::LlmProvider::name)), and
/// [`lookup_pricing`] matches it by equality — a key spelled any other way is
/// unreachable, and its models silently bill at \$0.
///
/// Model matching uses prefix comparison — a model name like "gemini-2.0-flash-exp"
/// matches the prefix "gemini-2.0-flash". When several of a provider's prefixes
/// match, [`lookup_pricing`] takes the longest, so the most specific row wins
/// wherever it sits in the table: `gpt-4o-mini` resolves to its own row even
/// though `gpt-4o` is listed first and also matches it.
///
/// A model is priced under every id a provider reports it as, and only those:
/// Claude Haiku 5.5 is `claude-haiku-5-5` on the Anthropic API and
/// `claude-haiku-5.5` on Copilot, and no `claude-haiku-5` family prefix exists,
/// so a later Haiku never inherits Haiku 5.5's card before it is priced.
/// Opus 5 and Sonnet 5 are the exception: their exact id (`claude-opus-5`,
/// `claude-sonnet-5`) is also the stem of every later 5.x id, so under prefix
/// matching an unpriced Opus or Sonnet 5.x bills its 5.0 card until it gets
/// a row of its own.
pub const PRICING_TABLE: &[(&str, &str, ModelPricing)] = &[
    // Gemini models (provider name matches GeminiProvider::name() = "gemini")
    // gemini-flash-lite-latest is a Google-maintained rolling alias to the
    // current GA flash-lite tier; prices track whatever that tier costs today.
    (
        "gemini",
        "gemini-flash-lite-latest",
        ModelPricing::new(0.10, 0.40),
    ),
    (
        "gemini",
        "gemini-2.5-flash-lite",
        ModelPricing::new(0.10, 0.40),
    ),
    ("gemini", "gemini-2.5-pro", ModelPricing::new(1.25, 10.0)),
    ("gemini", "gemini-2.5-flash", ModelPricing::new(0.15, 0.60)),
    ("gemini", "gemini-2.0-flash", ModelPricing::new(0.075, 0.30)),
    // Anthropic Messages API (AnthropicProvider::name() = "anthropic"; pool
    // accounts "anthropic#N" price through pool_family). Cache reads bill at
    // 0.10x input — 0.05x on Opus 5.5 ($0.20/MTok) — and 5-minute cache
    // writes at 1.25x. Haiku 5.5 has two rate cards: $0.10/$0.50 for a
    // prompt of 100K tokens or fewer, $0.50/$2.50 above that.
    (
        "anthropic",
        "claude-opus-5-5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "anthropic",
        "claude-sonnet-5-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "anthropic",
        "claude-haiku-5-5",
        ModelPricing::new(0.10, 0.50)
            .with_cache_rates(0.10, 1.25)
            .with_long_prompt_rates(100_000, 0.50, 2.50),
    ),
    // Groq models
    ("groq", "llama-3.3-70b", ModelPricing::new(0.59, 0.79)),
    ("groq", "mixtral", ModelPricing::new(0.24, 0.24)),
    ("groq", "llama-3.1-8b", ModelPricing::new(0.05, 0.08)),
    // Copilot headless (embacle) — proxies to Anthropic Claude models
    (
        "copilot_headless",
        // The Opus 4 family row, $15/$75. Opus 4.6 through 5.5 bill under
        // their own rows below, which the longest-prefix rule picks ahead of
        // this one; only Opus 4.5, 4.1 and 4.0 (`claude-opus-4.5`,
        // `claude-opus-4-5`, `claude-opus-4-1`, `claude-opus-4-0`) resolve
        // here.
        "claude-opus-4",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        // Version-agnostic Sonnet prefix for Sonnet 4.x (claude-sonnet-4,
        // -4.5, -4.6) at the $3/$15 Sonnet rate, so a bump keeps shadow-COGS
        // attributed instead of falling through to $0. Sonnet 5 ($2/$10) and
        // Sonnet 5.5 ($2/$10) bill under their own rows, which the
        // longest-prefix rule picks ahead of this one.
        "claude-sonnet",
        ModelPricing::new(3.0, 15.0).with_cache_rates(0.10, 1.25),
    ),
    // Claude Opus 4.6 through 5.5 and Sonnet 5, each under Copilot's dotted
    // id and Anthropic's hyphenated one (Opus 5 and Sonnet 5 have no minor
    // version, so one id serves both). Opus 5.5 is $4/$20 with cache reads at
    // 0.05x; Opus 5, 4.8, 4.7 and 4.6 are $5/$25; Sonnet 5 is $2/$10.
    // Copilot's `-fast` tiers keep their own rows below.
    (
        "copilot_headless",
        "claude-opus-5.5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-5-5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-5",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4.8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4-8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4.7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4-7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4.6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4-6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    // Copilot's `-fast` tiers of Opus 4.7 and 4.6. The reference gives no
    // list price for either (Anthropic retired both), and fast mode bills at
    // a premium over the base model ($10/$50 on Opus 4.8 and 5, 2x base), so
    // these bill at the $15/$75 of the `claude-opus-4` family row rather
    // than at the $5/$25 base rows they would otherwise prefix-match.
    (
        "copilot_headless",
        "claude-opus-4.7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4-7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4.6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-opus-4-6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-sonnet-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    // Claude Sonnet 5.5 and Haiku 5.5, each under Copilot's dotted id and
    // Anthropic's hyphenated one. Exact ids, never a `claude-haiku-5` family
    // prefix; the longest-prefix rule puts the Sonnet 5.5 rows ahead of the
    // version-agnostic `claude-sonnet` row above. Haiku 5.5 carries its base
    // card only ($0.10/$0.50, the card for a prompt of 100K tokens or fewer):
    // this runner reports one usage for a whole turn, summed over every model
    // request in it, so no single request's prompt length reaches the lookup
    // to choose the $0.50/$2.50 long-prompt card the way the `anthropic` rows do.
    (
        "copilot_headless",
        "claude-sonnet-5.5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-sonnet-5-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-haiku-5.5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        "claude-haiku-5-5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_headless",
        // Claude Haiku 4.5 at its $1/$5 list price.
        "claude-haiku-4",
        ModelPricing::new(1.0, 5.0).with_cache_rates(0.10, 1.25),
    ),
    // Copilot SDK (the Rust runtime) — the same Anthropic pass-through as
    // copilot_headless, keyed on the name CopilotSdkRunner reports.
    (
        "copilot_sdk",
        // Opus 4.5, 4.1 and 4.0 only, as on the copilot_headless row.
        "claude-opus-4",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-sonnet",
        ModelPricing::new(3.0, 15.0).with_cache_rates(0.10, 1.25),
    ),
    // Opus 4.6 through 5.5 and Sonnet 5 under both ids, for the reasons
    // given on the copilot_headless rows.
    (
        "copilot_sdk",
        "claude-opus-5.5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-5-5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-5",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4.8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4-8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4.7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4-7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4.6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4-6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    // Copilot's `-fast` tiers of Opus 4.7 and 4.6. The reference gives no
    // list price for either (Anthropic retired both), and fast mode bills at
    // a premium over the base model ($10/$50 on Opus 4.8 and 5, 2x base), so
    // these bill at the $15/$75 of the `claude-opus-4` family row rather
    // than at the $5/$25 base rows they would otherwise prefix-match.
    (
        "copilot_sdk",
        "claude-opus-4.7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4-7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4.6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-opus-4-6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-sonnet-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    // Sonnet 5.5 and Haiku 5.5 under both ids, base card only, for the
    // reasons given on the copilot_headless rows.
    (
        "copilot_sdk",
        "claude-sonnet-5.5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-sonnet-5-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-haiku-5.5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        "claude-haiku-5-5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "copilot_sdk",
        // Claude Haiku 4.5 at its $1/$5 list price.
        "claude-haiku-4",
        ModelPricing::new(1.0, 5.0).with_cache_rates(0.10, 1.25),
    ),
    // Claude Code CLI — same models as copilot_headless. Keyed "claude-code",
    // the name ClaudeCodeRunner reports; CliRunnerType's Display prints the
    // underscore spelling, but name() is what a caller's usage record carries.
    (
        "claude-code",
        // Opus 4.5, 4.1 and 4.0 only, as on the copilot_headless row.
        "claude-opus-4",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        // Version-agnostic Sonnet prefix for Sonnet 4.x at $3/$15, mirroring
        // the copilot_headless entry so a model bump keeps shadow-COGS
        // attributed instead of falling through to $0. Sonnet 5 and 5.5 have
        // their own rows, which the longest-prefix rule picks first.
        "claude-sonnet",
        ModelPricing::new(3.0, 15.0).with_cache_rates(0.10, 1.25),
    ),
    // Opus 4.6 through 5.5 and Sonnet 5 under both ids, for the reasons
    // given on the copilot_headless rows.
    (
        "claude-code",
        "claude-opus-5.5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-5-5",
        ModelPricing::new(4.0, 20.0).with_cache_rates(0.05, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-5",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4.8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4-8",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4.7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4-7",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4.6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4-6",
        ModelPricing::new(5.0, 25.0).with_cache_rates(0.10, 1.25),
    ),
    // Copilot's `-fast` tiers of Opus 4.7 and 4.6. The reference gives no
    // list price for either (Anthropic retired both), and fast mode bills at
    // a premium over the base model ($10/$50 on Opus 4.8 and 5, 2x base), so
    // these bill at the $15/$75 of the `claude-opus-4` family row rather
    // than at the $5/$25 base rows they would otherwise prefix-match.
    (
        "claude-code",
        "claude-opus-4.7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4-7-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4.6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-opus-4-6-fast",
        ModelPricing::new(15.0, 75.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-sonnet-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    // Sonnet 5.5 and Haiku 5.5 under both ids, base card only, for the
    // reasons given on the copilot_headless rows.
    (
        "claude-code",
        "claude-sonnet-5.5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-sonnet-5-5",
        ModelPricing::new(2.0, 10.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-haiku-5.5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        "claude-haiku-5-5",
        ModelPricing::new(0.10, 0.50).with_cache_rates(0.10, 1.25),
    ),
    (
        "claude-code",
        // Claude Haiku 4.5 at its $1/$5 list price.
        "claude-haiku-4",
        ModelPricing::new(1.0, 5.0).with_cache_rates(0.10, 1.25),
    ),
    // Cohere — Command A and Command R family.
    // The longest matching prefix wins, so `command-a-reasoning` and
    // `command-a-vision` resolve ahead of the bare `command-a` prefix and the
    // R-family entries don't accidentally swallow R+ / R7B.
    (
        "cohere",
        "command-a-reasoning",
        ModelPricing::new(2.50, 10.0),
    ),
    ("cohere", "command-a-vision", ModelPricing::new(2.50, 10.0)),
    ("cohere", "command-a", ModelPricing::new(2.50, 10.0)),
    ("cohere", "command-r-plus", ModelPricing::new(2.50, 10.0)),
    ("cohere", "command-r7b", ModelPricing::new(0.0375, 0.15)),
    ("cohere", "command-r", ModelPricing::new(0.15, 0.60)),
    // OpenAI API models
    (
        "openai_api",
        "gpt-4o",
        ModelPricing::new(2.50, 10.0).with_cache_rates(0.50, 1.00),
    ),
    (
        "openai_api",
        "gpt-4o-mini",
        ModelPricing::new(0.15, 0.60).with_cache_rates(0.50, 1.00),
    ),
    // OpenRouter — the gateway passes through the underlying model's published
    // price. These cover the curated default slugs the OpenRouter provider
    // ships (its default model + AVAILABLE_MODELS); any other slug is logged
    // as model-specific-unset rather than silently $0.
    // Prefixes carry the provider/model slug so prefix matching still resolves
    // dated or suffixed variants (e.g. `:free`, `-001`).
    (
        "openrouter",
        "meta-llama/llama-3.3-70b-instruct",
        ModelPricing::new(0.12, 0.30),
    ),
    (
        "openrouter",
        "meta-llama/llama-3.1-8b-instruct",
        ModelPricing::new(0.02, 0.03),
    ),
    (
        "openrouter",
        "anthropic/claude-3.5-sonnet",
        ModelPricing::new(3.0, 15.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "openrouter",
        "anthropic/claude-3.5-haiku",
        ModelPricing::new(0.80, 4.0).with_cache_rates(0.10, 1.25),
    ),
    (
        "openrouter",
        "openai/gpt-4o-mini",
        ModelPricing::new(0.15, 0.60).with_cache_rates(0.50, 1.00),
    ),
    (
        "openrouter",
        "openai/gpt-4o",
        ModelPricing::new(2.50, 10.0).with_cache_rates(0.50, 1.00),
    ),
    (
        "openrouter",
        "google/gemini-2.0-flash-001",
        ModelPricing::new(0.10, 0.40),
    ),
    (
        "openrouter",
        "google/gemini-pro-1.5",
        ModelPricing::new(1.25, 5.0),
    ),
    (
        "openrouter",
        "mistralai/mistral-large",
        ModelPricing::new(2.0, 6.0),
    ),
    (
        "openrouter",
        "mistralai/mistral-nemo",
        ModelPricing::new(0.03, 0.07),
    ),
    (
        "openrouter",
        "qwen/qwen-2.5-72b-instruct",
        ModelPricing::new(0.13, 0.40),
    ),
];

/// Look up pricing for a (provider, model) pair using prefix matching.
///
/// Of the provider's rows whose prefix the model starts with, the longest
/// prefix wins, so a broad row (`gpt-4o`, `claude-sonnet`) never shadows a
/// specific one (`gpt-4o-mini`, `claude-sonnet-5.5`) whatever the table order.
///
/// Public so a consumer's override layer can fall through to the table
/// after its own lookup misses.
#[must_use]
pub fn lookup_pricing(provider: &str, model: &str) -> Option<ModelPricing> {
    let provider = pool_family(provider);
    PRICING_TABLE
        .iter()
        .filter(|(p, prefix, _)| *p == provider && model.starts_with(prefix))
        .max_by_key(|(_, prefix, _)| prefix.len())
        .map(|(_, _, pricing)| *pricing)
}

/// The provider a pooled tier prices as: `claude-code#2` is a second Claude
/// account, billed like the first. See [`crate::pool`].
#[must_use]
pub fn pool_family(provider: &str) -> &str {
    provider
        .split_once('#')
        .map_or(provider, |(family, _)| family)
}

/// True when a provider bills via flat-rate subscription or self-hosting, so a
/// \$0 per-token cost is correct rather than a missing-price undercount.
#[must_use]
pub fn is_not_per_token_metered(provider: &str) -> bool {
    NOT_PER_TOKEN_METERED_PROVIDERS.contains(&pool_family(provider))
}

/// Log the resolution of a (provider, model) pair that has no `PRICING_TABLE`
/// or override entry, at a severity matching *why* it is unpriced, and return
/// the \$0 cost all three paths fall back to.
///
/// - Subscription / self-hosted providers: \$0 is correct — `debug!` only, no
///   undercount warning.
/// - `OpenRouter`: model-specific price genuinely unset for this slug —
///   `info!` with a distinct, non-alarming message.
/// - Anything else: a genuinely-unexpected miss that *does* undercount —
///   keep the `warn!`.
///
/// `scope` is whatever the caller keys its overrides by (an account, an
/// organisation) and only decorates the warning.
#[must_use]
pub fn zero_cost_for_unpriced(provider: &str, model: &str, scope: Option<&str>) -> f64 {
    if is_not_per_token_metered(provider) {
        debug!(
            provider = provider,
            model = model,
            "Provider is subscription/self-hosted; per-token cost is $0 by design (not an undercount)"
        );
    } else if provider == OPENROUTER_PROVIDER {
        info!(
            provider = provider,
            model = model,
            "OpenRouter model price is model-specific and unset for this slug; recording $0 (add a PRICING_TABLE or admin override to meter it)"
        );
    } else {
        warn!(
            provider = provider,
            model = model,
            scope = scope,
            "No pricing data for model, cost will be recorded as 0.0"
        );
    }
    0.0
}

/// The token counts a single LLM call reports, in the shape they bill.
///
/// `cached_read` and `cached_write` are both subsets of `prompt`: a provider
/// reports the gross prompt count and then breaks out how much of it was
/// served from cache and how much was written into it. `reasoning` is the
/// exception — providers that report it exclude it from `completion`, so it
/// is additive on the output side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCounts {
    /// Gross prompt tokens, cached portions included.
    pub prompt: i64,
    /// Prompt tokens served from the provider's context cache.
    pub cached_read: i64,
    /// Prompt tokens written into the provider's context cache this call.
    pub cached_write: i64,
    /// Completion tokens, excluding separately-reported reasoning tokens.
    pub completion: i64,
    /// Reasoning / "thought" tokens, when the provider reports them apart
    /// from `completion`. Billed at the output rate.
    pub reasoning: i64,
}

impl TokenCounts {
    /// The two counts every provider reports.
    #[must_use]
    pub const fn new(prompt: i64, completion: i64) -> Self {
        Self {
            prompt,
            cached_read: 0,
            cached_write: 0,
            completion,
            reasoning: 0,
        }
    }

    /// Attach the cache read/write split of `prompt`.
    #[must_use]
    pub const fn with_cache(mut self, read: i64, write: i64) -> Self {
        self.cached_read = read;
        self.cached_write = write;
        self
    }

    /// Attach a separately-reported reasoning-token count.
    #[must_use]
    pub const fn with_reasoning(mut self, reasoning: i64) -> Self {
        self.reasoning = reasoning;
        self
    }
}

/// Calculate the cost of an LLM request using compile-time pricing.
///
/// Returns the cost in USD. Returns 0.0 for unknown provider/model combinations
/// (with a warning log). Treats the whole prompt as uncached and assumes no
/// separately-reported reasoning tokens — callers holding either count must
/// use [`calculate_cost_with_cache`] or [`calculate_cost_for`].
#[must_use]
pub fn calculate_cost(
    provider: &str,
    model: &str,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> f64 {
    calculate_cost_for(
        provider,
        model,
        &TokenCounts::new(prompt_tokens, completion_tokens),
    )
}

/// Calculate the cost of an LLM request from a full [`TokenCounts`].
///
/// The complete entry point: every count the provider reported, priced with
/// that model's own cache multipliers. Returns 0.0 for unpriced pairs on the
/// same terms as [`calculate_cost_with_cache`].
#[must_use]
pub fn calculate_cost_for(provider: &str, model: &str, counts: &TokenCounts) -> f64 {
    let Some(pricing) = lookup_pricing(provider, model) else {
        return zero_cost_for_unpriced(provider, model, None);
    };

    cost_from_pricing(&pricing, counts)
}

/// Calculate the cost of an LLM request, breaking the prompt into cached + fresh tokens.
///
/// Cache-read tokens bill at the model's [`ModelPricing::cache_read_multiplier`]
/// of its input rate; the remaining `(prompt_tokens - cached_tokens)` bill at
/// the full input rate. Completion tokens always bill at the full output rate.
/// Reports no cache *writes* — use [`calculate_cost_for`] when the provider
/// broke those out. Returns
/// 0.0 for unpriced (provider, model) pairs — the $0 fallback is deliberate
/// so a missing pricing entry never blocks a chat turn. The log severity of
/// that fallback depends on *why* the pair is unpriced (see
/// [`zero_cost_for_unpriced`]): subscription/self-hosted providers log at
/// debug (their $0 is correct), `OpenRouter` logs an honest model-specific
/// notice, and only a genuinely-unexpected miss emits the undercount warning.
#[must_use]
pub fn calculate_cost_with_cache(
    provider: &str,
    model: &str,
    prompt_tokens: i64,
    cached_tokens: i64,
    completion_tokens: i64,
) -> f64 {
    calculate_cost_for(
        provider,
        model,
        &TokenCounts::new(prompt_tokens, completion_tokens).with_cache(cached_tokens, 0),
    )
}

/// Compute the USD cost from a resolved [`ModelPricing`] + token counts.
/// Shared by the compile-time lookup and any override path a consumer
/// layers on top, so both sources price a call identically.
///
/// Four terms, because a prompt token can bill at three different rates:
/// fresh input, a cache read at a discount, and a cache write at a premium.
/// Reasoning tokens join completion on the output side.
#[must_use]
pub fn cost_from_pricing(pricing: &ModelPricing, counts: &TokenCounts) -> f64 {
    let divisor = 1_000_000.0;
    let prompt = counts.prompt.max(0);
    // Reads and writes are carved out of the gross prompt in that order, so a
    // provider that over-reports one can never push the fresh remainder below
    // zero or bill a token twice.
    let cached_read = counts.cached_read.clamp(0, prompt);
    let cached_write = counts.cached_write.clamp(0, prompt - cached_read);
    let fresh_prompt = prompt - cached_read - cached_write;

    let (input, output) = pricing.rates_for_prompt(prompt);
    let fresh_prompt_cost = fresh_prompt as f64 * input / divisor;
    let cached_read_cost = cached_read as f64 * input * pricing.cache_read_multiplier / divisor;
    let cached_write_cost = cached_write as f64 * input * pricing.cache_write_multiplier / divisor;
    // Providers that report reasoning tokens exclude them from the completion
    // count, so they are added rather than carved out.
    let output_tokens = counts.completion.max(0) + counts.reasoning.max(0);
    let output_cost = output_tokens as f64 * output / divisor;

    fresh_prompt_cost + cached_read_cost + cached_write_cost + output_cost
}
