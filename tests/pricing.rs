// ABOUTME: The compile-time price table and its cost arithmetic
// ABOUTME: Known models, prefix matching, per-provider cache rates, reasoning tokens, clamping
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::collections::BTreeSet;

use embacle::pricing::{
    calculate_cost, calculate_cost_for, cost_from_pricing, lookup_pricing, ModelPricing,
    TokenCounts, PRICING_TABLE,
};

#[test]
fn test_known_model_cost() {
    // Gemini 2.0 Flash: $0.075/M input, $0.30/M output
    let cost = calculate_cost("gemini", "gemini-2.0-flash", 1000, 500);
    let expected = 1000.0_f64.mul_add(0.075, 500.0 * 0.30) / 1_000_000.0;
    assert!(
        (cost - expected).abs() < f64::EPSILON,
        "Expected {expected}, got {cost}"
    );
}

#[test]
fn test_unknown_model_returns_zero() {
    let cost = calculate_cost("unknown_provider", "unknown_model", 1000, 500);
    assert!(
        cost.abs() < f64::EPSILON,
        "Expected 0.0 for unknown model, got {cost}"
    );
}

#[test]
fn test_zero_tokens_returns_zero() {
    let cost = calculate_cost("gemini", "gemini-2.0-flash", 0, 0);
    assert!(
        cost.abs() < f64::EPSILON,
        "Expected 0.0 for zero tokens, got {cost}"
    );
}

#[test]
fn test_prefix_matching() {
    // "gemini-2.0-flash-exp" should match "gemini-2.0-flash" prefix
    let cost_exact = calculate_cost("gemini", "gemini-2.0-flash", 1000, 500);
    let cost_variant = calculate_cost("gemini", "gemini-2.0-flash-exp", 1000, 500);
    assert!(
        (cost_exact - cost_variant).abs() < f64::EPSILON,
        "Variant model should match base pricing"
    );
}

#[test]
fn test_gemini_25_pro_pricing() {
    // Gemini 2.5 Pro: $1.25/M input, $10.0/M output
    let cost = calculate_cost("gemini", "gemini-2.5-pro", 1_000_000, 500_000);
    let expected = 1.25 + 5.0; // $1.25 for 1M input + $5.0 for 500K output
    assert!(
        (cost - expected).abs() < 1e-10,
        "Expected {expected}, got {cost}"
    );
}

#[test]
fn test_groq_llama_pricing() {
    // Groq llama-3.3-70b: $0.59/M input, $0.79/M output
    let cost = calculate_cost("groq", "llama-3.3-70b-versatile", 2000, 1000);
    let expected = 2000.0_f64.mul_add(0.59, 1000.0 * 0.79) / 1_000_000.0;
    assert!(
        (cost - expected).abs() < f64::EPSILON,
        "Expected {expected}, got {cost}"
    );
}

#[test]
fn test_groq_mixtral_pricing() {
    // Groq mixtral: $0.24/M input, $0.24/M output
    let cost = calculate_cost("groq", "mixtral-8x7b-32768", 5000, 3000);
    let expected = 5000.0_f64.mul_add(0.24, 3000.0 * 0.24) / 1_000_000.0;
    assert!(
        (cost - expected).abs() < f64::EPSILON,
        "Expected {expected}, got {cost}"
    );
}

#[test]
fn test_gemini_flash_lite_latest_pricing() {
    // Gemini Flash-Lite Latest (alias to current GA flash-lite): $0.10/M input, $0.40/M output
    let cost = calculate_cost("gemini", "gemini-flash-lite-latest", 1_000_000, 100_000);
    let expected = 0.10 + 0.04; // $0.10 for 1M input + $0.04 for 100K output
    assert!(
        (cost - expected).abs() < 1e-10,
        "Expected {expected}, got {cost}"
    );
}

#[test]
fn test_every_default_model_has_pricing() {
    let production_models = [
        ("gemini", "gemini-flash-lite-latest"),
        ("gemini", "gemini-2.5-flash"),
        ("gemini", "gemini-2.0-flash"),
        ("groq", "llama-3.3-70b-versatile"),
    ];
    for (provider, model) in production_models {
        let cost = calculate_cost(provider, model, 1000, 1000);
        assert!(
            cost > 0.0,
            "Model {provider}/{model} has ZERO pricing — every turn on it bills $0.00"
        );
    }
}

// ============================================================================
// Cache and reasoning accounting
// ============================================================================

/// A cache *read* on an Anthropic-backed model bills at 0.10x input, not a
/// flat 0.25x applied to every provider.
#[test]
fn anthropic_cache_read_bills_at_ten_percent() {
    // 1M prompt tokens, all served from cache, no output.
    let counts = TokenCounts::new(1_000_000, 0).with_cache(1_000_000, 0);
    let cost = calculate_cost_for("copilot_headless", "claude-opus-4", &counts);

    // $15/M input x 0.10 = $1.50, NOT the $3.75 a flat 0.25x would charge.
    assert!(
        (cost - 1.50).abs() < 1e-9,
        "expected $1.50 for 1M Anthropic cache-read tokens, got {cost}"
    );
}

/// A cache *write* is a premium on Anthropic (1.25x), so it must cost MORE
/// than the same tokens billed as fresh input. Folding writes into the fresh
/// count understates the bill, which is the failure this asserts against.
#[test]
fn anthropic_cache_write_bills_above_fresh_input() {
    let written = TokenCounts::new(1_000_000, 0).with_cache(0, 1_000_000);
    let fresh = TokenCounts::new(1_000_000, 0);

    let write_cost = calculate_cost_for("copilot_headless", "claude-opus-4", &written);
    let fresh_cost = calculate_cost_for("copilot_headless", "claude-opus-4", &fresh);

    // $15/M x 1.25 = $18.75 against $15.00 fresh.
    assert!(
        (write_cost - 18.75).abs() < 1e-9,
        "expected $18.75 for 1M Anthropic cache-write tokens, got {write_cost}"
    );
    assert!(
        write_cost > fresh_cost,
        "a cache write is a premium, not a discount: write={write_cost} fresh={fresh_cost}"
    );
}

/// Reasoning tokens are excluded from `completion` by every provider that
/// reports them, so they are additive on the output side. Dropping them
/// charged nothing at all for that output.
#[test]
fn reasoning_tokens_bill_at_the_output_rate() {
    let without = TokenCounts::new(0, 1_000_000);
    let with = TokenCounts::new(0, 1_000_000).with_reasoning(1_000_000);

    let cost_without = calculate_cost_for("copilot_headless", "claude-opus-4", &without);
    let cost_with = calculate_cost_for("copilot_headless", "claude-opus-4", &with);

    // $75/M output: 1M completion = $75, plus 1M reasoning = $150 total.
    assert!(
        (cost_without - 75.0).abs() < 1e-9,
        "expected $75 for 1M completion tokens, got {cost_without}"
    );
    assert!(
        (cost_with - 150.0).abs() < 1e-9,
        "reasoning tokens must bill at the output rate; got {cost_with}"
    );
}

/// The three providers price cache reads differently, and the table carries
/// each rate rather than averaging them into one constant.
#[test]
fn cache_read_rate_is_per_provider() {
    let counts = TokenCounts::new(1_000_000, 0).with_cache(1_000_000, 0);

    // Anthropic 0.10x of $15/M = $1.50
    let anthropic = calculate_cost_for("copilot_headless", "claude-opus-4", &counts);
    // OpenAI 0.50x of $2.50/M = $1.25
    let openai = calculate_cost_for("openai_api", "gpt-4o", &counts);
    // Gemini 0.25x of $0.075/M = $0.01875
    let gemini = calculate_cost_for("gemini", "gemini-2.0-flash", &counts);

    assert!((anthropic - 1.50).abs() < 1e-9, "anthropic={anthropic}");
    assert!((openai - 1.25).abs() < 1e-9, "openai={openai}");
    assert!((gemini - 0.018_75).abs() < 1e-9, "gemini={gemini}");
}

/// A turn reporting reads AND writes bills three prompt segments at three
/// different rates. Shaped on a real Copilot ACP payload: 15,320 read +
/// 12,540 written out of 27,862 prompt tokens.
#[test]
fn real_acp_turn_splits_prompt_across_three_rates() {
    let counts = TokenCounts::new(27_862, 4).with_cache(15_320, 12_540);
    let cost = calculate_cost_for("copilot_headless", "claude-opus-4", &counts);

    let input = 15.0 / 1_000_000.0;
    let fresh = f64::from(27_862 - 15_320 - 12_540) * input;
    let read = 15_320.0 * input * 0.10;
    let write = 12_540.0 * input * 1.25;
    let output = 4.0 * 75.0 / 1_000_000.0;
    let expected = fresh + read + write + output;

    assert!(
        (cost - expected).abs() < 1e-12,
        "expected {expected}, got {cost}"
    );

    // The naive all-fresh imputation is a different number — this is the
    // whole point of carrying the counts.
    let naive = calculate_cost_for(
        "copilot_headless",
        "claude-opus-4",
        &TokenCounts::new(27_862, 4),
    );
    assert!(
        (cost - naive).abs() > 1e-9,
        "cache-aware and all-fresh imputation must differ; both were {cost}"
    );
}

/// Over-reported cache counts can never bill a prompt token twice or push the
/// fresh remainder negative.
#[test]
fn cache_counts_are_clamped_to_the_prompt() {
    let counts = TokenCounts::new(1_000, 0).with_cache(900, 900);
    let cost = calculate_cost_for("copilot_headless", "claude-opus-4", &counts);

    let input = 15.0 / 1_000_000.0;
    // 900 read, then only 100 left to count as written, then 0 fresh.
    let expected = 900.0f64.mul_add(input * 0.10, 100.0 * input * 1.25);
    assert!(
        (cost - expected).abs() < 1e-12,
        "expected {expected}, got {cost}"
    );
}

// ============================================================================
// Prefix resolution
// ============================================================================

/// The row a (provider, model) pair resolves to, or a failure naming the pair.
fn resolved(provider: &str, model: &str) -> ModelPricing {
    lookup_pricing(provider, model)
        .unwrap_or_else(|| panic!("{provider}/{model} resolves to no price row"))
}

/// Assert a resolved row's four rates.
fn assert_rates(provider: &str, model: &str, input: f64, output: f64, read: f64, write: f64) {
    let pricing = resolved(provider, model);
    let actual = (
        pricing.input_per_million,
        pricing.output_per_million,
        pricing.cache_read_multiplier,
        pricing.cache_write_multiplier,
    );
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(
        close(actual.0, input)
            && close(actual.1, output)
            && close(actual.2, read)
            && close(actual.3, write),
        "{provider}/{model} resolved to (input, output, read, write) = {actual:?}, \
         expected ({input}, {output}, {read}, {write})"
    );
}

/// A broad prefix listed before a specific one must not shadow it: `gpt-4o`
/// precedes `gpt-4o-mini` in the table and matches it too.
#[test]
fn the_longest_matching_prefix_wins_whatever_the_table_order() {
    assert_rates("openai_api", "gpt-4o-mini", 0.15, 0.60, 0.50, 1.0);
    assert_rates(
        "openai_api",
        "gpt-4o-mini-2024-07-18",
        0.15,
        0.60,
        0.50,
        1.0,
    );
    assert_rates("openai_api", "gpt-4o-2024-08-06", 2.50, 10.0, 0.50, 1.0);
}

/// Two rows with one (provider, prefix) would leave the winner to table order.
#[test]
fn no_provider_lists_the_same_prefix_twice() {
    let mut seen = BTreeSet::new();
    for (provider, prefix, _) in PRICING_TABLE {
        assert!(
            seen.insert((*provider, *prefix)),
            "PRICING_TABLE lists ({provider}, {prefix}) more than once"
        );
    }
}

// ============================================================================
// Claude Haiku on the runners that pass Anthropic usage through
// ============================================================================

/// Every runner with real Anthropic price rows, by the name it reports.
const ANTHROPIC_RUNNERS: [&str; 3] = ["copilot_headless", "copilot_sdk", "claude-code"];

/// Haiku 5.5 under Copilot's dotted id and Anthropic's hyphenated one.
const HAIKU_5_IDS: [&str; 2] = ["claude-haiku-5.5", "claude-haiku-5-5"];

/// Haiku 4.5 under Copilot's id, Anthropic's alias and its dated snapshot.
const HAIKU_4_IDS: [&str; 3] = [
    "claude-haiku-4.5",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
];

/// A cached, tool-using Copilot SDK turn on Haiku: two model calls summed,
/// 43,538 prompt tokens of which 21,678 were read from cache and 21,854
/// written to it, 163 completion and 113 reasoning tokens.
fn cached_tool_turn() -> TokenCounts {
    TokenCounts::new(43_538, 163)
        .with_cache(21_678, 21_854)
        .with_reasoning(113)
}

#[test]
fn haiku_5_resolves_to_its_own_row_on_every_anthropic_runner() {
    for provider in ANTHROPIC_RUNNERS {
        for model in HAIKU_5_IDS {
            // $0.10 / $0.50 list; a fall to the Haiku 4 row would read $1 / $5.
            assert_rates(provider, model, 0.10, 0.50, 0.10, 1.25);
        }
    }
    // A pooled second account prices like the first.
    assert_rates("claude-code#2", "claude-haiku-5-5", 0.10, 0.50, 0.10, 1.25);
}

#[test]
fn haiku_4_5_bills_at_list_price_on_every_anthropic_runner() {
    for provider in ANTHROPIC_RUNNERS {
        for model in HAIKU_4_IDS {
            assert_rates(provider, model, 1.0, 5.0, 0.10, 1.25);
        }
    }
}

#[test]
fn haiku_5_prices_a_cached_tool_turn() {
    let counts = cached_tool_turn();
    let cost = cost_from_pricing(&resolved("copilot_sdk", "claude-haiku-5.5"), &counts);

    // 6 fresh x $0.10/M + 21,678 read x $0.01/M + 21,854 written x $0.125/M
    // + (163 + 113) output x $0.50/M.
    let expected = 0.000_000_6 + 0.000_216_78 + 0.002_731_75 + 0.000_138;
    assert!(
        (cost - expected).abs() < 1e-12,
        "expected {expected}, got {cost}"
    );

    // The table entry point prices it identically.
    let via_table = calculate_cost_for("copilot_sdk", "claude-haiku-5.5", &counts);
    assert!(
        (via_table - cost).abs() < 1e-15,
        "calculate_cost_for={via_table} cost_from_pricing={cost}"
    );
}

#[test]
fn the_same_turn_on_haiku_4_5_costs_ten_times_haiku_5() {
    let counts = cached_tool_turn();
    for provider in ANTHROPIC_RUNNERS {
        let haiku_5 = calculate_cost_for(provider, "claude-haiku-5-5", &counts);
        let haiku_4 = calculate_cost_for(provider, "claude-haiku-4-5", &counts);
        // Every Haiku 4.5 rate is ten times Haiku 5.5's, and both share the
        // Anthropic cache multipliers, so the turn costs ten times as much.
        assert!(
            (haiku_5 - 0.003_087_13).abs() < 1e-12,
            "{provider}: haiku 5.5 cost {haiku_5}"
        );
        assert!(
            (haiku_4 - 0.030_871_3).abs() < 1e-12,
            "{provider}: haiku 4.5 cost {haiku_4}"
        );
    }
}
