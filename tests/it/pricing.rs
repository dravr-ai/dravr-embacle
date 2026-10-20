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
    calculate_cost, calculate_cost_for, lookup_pricing, ModelPricing, TokenCounts, PRICING_TABLE,
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
        ("anthropic", "claude-haiku-5-5"),
        ("anthropic", "claude-sonnet-5-5"),
        ("anthropic", "claude-opus-5-5"),
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
// Anthropic Messages API (the "anthropic" provider and its pool accounts)
// ============================================================================

/// Each 5.5 model's input, output, cache-read and cache-write rates, per
/// million tokens: (model, input, output, cache read, cache write).
const ANTHROPIC_5_5_RATES: &[(&str, f64, f64, f64, f64)] = &[
    ("claude-haiku-5-5", 0.10, 0.50, 0.01, 0.125),
    ("claude-sonnet-5-5", 2.0, 10.0, 0.20, 2.50),
    ("claude-opus-5-5", 4.0, 20.0, 0.20, 5.0),
];

/// Rates are checked on a 100K-token prompt, the largest that every model
/// bills on its base card; the expected cost is a tenth of the per-MTok rate.
#[test]
fn anthropic_5_5_models_price_every_rate() {
    for &(model, input, output, read, write) in ANTHROPIC_5_5_RATES {
        let tokens = 100_000;
        let tenth = |rate: f64| rate / 10.0;
        let fresh = calculate_cost("anthropic", model, tokens, 0);
        assert!(
            (fresh - tenth(input)).abs() < 1e-9,
            "{model} input: {fresh}"
        );
        let out = calculate_cost("anthropic", model, 0, tokens);
        assert!((out - tenth(output)).abs() < 1e-9, "{model} output: {out}");
        let cached = calculate_cost_for(
            "anthropic",
            model,
            &TokenCounts::new(tokens, 0).with_cache(tokens, 0),
        );
        assert!(
            (cached - tenth(read)).abs() < 1e-9,
            "{model} cache read: {cached}"
        );
        let written = calculate_cost_for(
            "anthropic",
            model,
            &TokenCounts::new(tokens, 0).with_cache(0, tokens),
        );
        assert!(
            (written - tenth(write)).abs() < 1e-9,
            "{model} cache write: {written}"
        );
    }
}

/// Haiku 5.5 bills a prompt above 100K tokens on its second card — input,
/// cache and output alike — and a prompt of exactly 100K on the first.
#[test]
fn haiku_5_5_bills_a_long_prompt_on_its_second_rate_card() {
    let at_threshold = calculate_cost("anthropic", "claude-haiku-5-5", 100_000, 1_000);
    // 100_000 × $0.10 + 1_000 × $0.50, per million tokens
    let base = 0.0105;
    assert!((at_threshold - base).abs() < 1e-12, "{at_threshold}");

    let above = calculate_cost("anthropic", "claude-haiku-5-5", 100_001, 1_000);
    // 100_001 × $0.50 + 1_000 × $2.50, per million tokens
    let long = 0.052_500_5;
    assert!((above - long).abs() < 1e-12, "{above}");

    // The threshold reads the gross prompt, cached tokens included, and the
    // cache multipliers apply to the long-card input rate.
    let cached = calculate_cost_for(
        "anthropic#2",
        "claude-haiku-5-5",
        &TokenCounts::new(150_000, 0).with_cache(140_000, 0),
    );
    // 10_000 fresh × $0.50 + 140_000 cache reads × $0.05, per million tokens
    let expected = 0.012;
    assert!((cached - expected).abs() < 1e-12, "{cached}");

    // Models with one card are unaffected by prompt length
    let sonnet = calculate_cost("anthropic", "claude-sonnet-5-5", 1_000_000, 0);
    assert!((sonnet - 2.0).abs() < 1e-9, "{sonnet}");
}

/// A second (or tenth) pool account bills exactly as the first.
#[test]
fn anthropic_pool_accounts_price_as_their_family() {
    let counts = TokenCounts::new(40_000, 900).with_cache(30_000, 5_000);
    let first = calculate_cost_for("anthropic", "claude-haiku-5-5", &counts);
    assert!(first > 0.0);
    for account in ["anthropic#2", "anthropic#10"] {
        let cost = calculate_cost_for(account, "claude-haiku-5-5", &counts);
        assert!((cost - first).abs() < 1e-12, "{account}: {cost} vs {first}");
    }
}

// ============================================================================
// Prefix resolution
// ============================================================================

/// The row a (provider, model) pair resolves to, or a failure naming the pair.
fn resolved(provider: &str, model: &str) -> ModelPricing {
    lookup_pricing(provider, model)
        .unwrap_or_else(|| panic!("{provider}/{model} resolves to no price row"))
}

/// Assert a resolved row's input, output, cache-read and cache-write rates,
/// and that it carries no long-prompt card unless `long` names one.
fn assert_rates(provider: &str, model: &str, rates: (f64, f64, f64, f64), long: bool) {
    let pricing = resolved(provider, model);
    let actual = (
        pricing.input_per_million,
        pricing.output_per_million,
        pricing.cache_read_multiplier,
        pricing.cache_write_multiplier,
    );
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(
        close(actual.0, rates.0)
            && close(actual.1, rates.1)
            && close(actual.2, rates.2)
            && close(actual.3, rates.3),
        "{provider}/{model} resolved to (input, output, read, write) = {actual:?}, \
         expected {rates:?}"
    );
    assert_eq!(
        pricing.long_prompt.is_some(),
        long,
        "{provider}/{model} long-prompt card: {:?}",
        pricing.long_prompt
    );
}

/// A broad prefix listed before a specific one must not shadow it: `gpt-4o`
/// precedes `gpt-4o-mini` in the table and matches it too, so a first-match
/// lookup prices the mini at the full model's $2.50.
#[test]
fn the_longest_matching_prefix_wins_whatever_the_table_order() {
    let mini = (0.15, 0.60, 0.50, 1.0);
    assert_rates("openai_api", "gpt-4o-mini", mini, false);
    assert_rates("openai_api", "gpt-4o-mini-2024-07-18", mini, false);
    assert_rates(
        "openai_api",
        "gpt-4o-2024-08-06",
        (2.50, 10.0, 0.50, 1.0),
        false,
    );
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
// Claude 5.5 and Haiku 4.5 on the runners that pass Anthropic usage through
// ============================================================================

/// Every runner with real Anthropic price rows, by the name it reports.
const PASS_THROUGH_RUNNERS: [&str; 3] = ["copilot_headless", "copilot_sdk", "claude-code"];

/// Haiku 5.5: $0.10 / $0.50, Anthropic cache multipliers.
const HAIKU_5_5: (f64, f64, f64, f64) = (0.10, 0.50, 0.10, 1.25);

/// Sonnet 5.5: $2 / $10, Anthropic cache multipliers.
const SONNET_5_5: (f64, f64, f64, f64) = (2.0, 10.0, 0.10, 1.25);

#[test]
fn haiku_5_5_is_priced_under_every_id_a_runner_reports() {
    for provider in PASS_THROUGH_RUNNERS {
        // Copilot's dotted id and Anthropic's hyphenated one. The base card
        // only: these runners report a whole turn's summed usage.
        for model in ["claude-haiku-5.5", "claude-haiku-5-5"] {
            assert_rates(provider, model, HAIKU_5_5, false);
        }
    }
    // The Anthropic API reports per request, so it carries the long card.
    assert_rates("anthropic", "claude-haiku-5-5", HAIKU_5_5, true);
    // A pooled second account prices like the first.
    assert_rates("claude-code#2", "claude-haiku-5-5", HAIKU_5_5, false);
}

#[test]
fn sonnet_5_5_wins_over_the_version_agnostic_sonnet_row() {
    for provider in PASS_THROUGH_RUNNERS {
        // `claude-sonnet` ($3/$15) is listed first and matches these too.
        for model in ["claude-sonnet-5.5", "claude-sonnet-5-5"] {
            assert_rates(provider, model, SONNET_5_5, false);
        }
        // Earlier Sonnets keep the version-agnostic row.
        assert_rates(
            provider,
            "claude-sonnet-4.5",
            (3.0, 15.0, 0.10, 1.25),
            false,
        );
    }
    assert_rates("anthropic", "claude-sonnet-5-5", SONNET_5_5, false);
}

/// Haiku 5.5 is priced by its exact version: no `claude-haiku-5` family row
/// lends its card to a Haiku that is not 5.5.
#[test]
fn no_other_haiku_5_inherits_the_haiku_5_5_card() {
    for provider in [
        "anthropic",
        "copilot_headless",
        "copilot_sdk",
        "claude-code",
    ] {
        for model in [
            "claude-haiku-5",
            "claude-haiku-5-1",
            "claude-haiku-5.1",
            "claude-haiku-5-0",
        ] {
            assert!(
                lookup_pricing(provider, model).is_none(),
                "{provider}/{model} resolved to {:?}; only Haiku 5.5 is priced",
                lookup_pricing(provider, model)
            );
        }
    }
}

#[test]
fn haiku_4_5_bills_at_its_list_price() {
    for provider in PASS_THROUGH_RUNNERS {
        for model in [
            "claude-haiku-4.5",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
        ] {
            assert_rates(provider, model, (1.0, 5.0, 0.10, 1.25), false);
        }
    }
}

/// A cached, tool-using Copilot SDK turn on Haiku 5.5: two model calls summed,
/// 43,538 prompt tokens of which 21,678 were read from cache and 21,854
/// written to it, 163 completion and 113 reasoning tokens.
#[test]
fn haiku_5_5_prices_a_cached_tool_turn() {
    let counts = TokenCounts::new(43_538, 163)
        .with_cache(21_678, 21_854)
        .with_reasoning(113);
    for provider in PASS_THROUGH_RUNNERS {
        let cost = calculate_cost_for(provider, "claude-haiku-5.5", &counts);
        // 6 fresh x $0.10/M + 21,678 read x $0.01/M + 21,854 written
        // x $0.125/M + (163 + 113) output x $0.50/M.
        let expected = 0.000_000_6 + 0.000_216_78 + 0.002_731_75 + 0.000_138;
        assert!(
            (cost - expected).abs() < 1e-12,
            "{provider}: expected {expected}, got {cost}"
        );
    }
}

/// A turn whose summed prompt passes 100K stays on the base card on a
/// pass-through runner; the Anthropic API, which reports one request, does not.
#[test]
fn a_long_summed_turn_stays_on_the_base_card_off_the_anthropic_api() {
    let turn = calculate_cost("copilot_sdk", "claude-haiku-5.5", 150_000, 0);
    assert!((turn - 0.015).abs() < 1e-12, "{turn}");
    let request = calculate_cost("anthropic", "claude-haiku-5-5", 150_000, 0);
    assert!((request - 0.075).abs() < 1e-12, "{request}");
}

// ============================================================================
// Claude Opus and Sonnet 5 on the pass-through runners
// ============================================================================

/// Opus 5.5: $4 / $20, cache reads at 0.05x.
const OPUS_5_5: (f64, f64, f64, f64) = (4.0, 20.0, 0.05, 1.25);

/// Opus 5, 4.8, 4.7 and 4.6: $5 / $25.
const OPUS_5_25: (f64, f64, f64, f64) = (5.0, 25.0, 0.10, 1.25);

/// The Opus 4 family row Opus 4.5, 4.1 and 4.0 resolve to.
const OPUS_4_FAMILY: (f64, f64, f64, f64) = (15.0, 75.0, 0.10, 1.25);

/// Sonnet 5: $2 / $10.
const SONNET_5: (f64, f64, f64, f64) = (2.0, 10.0, 0.10, 1.25);

/// Sonnet 4.x: the version-agnostic `claude-sonnet` row, $3 / $15.
const SONNET_4: (f64, f64, f64, f64) = (3.0, 15.0, 0.10, 1.25);

#[test]
fn opus_5_5_is_priced_under_both_ids() {
    for provider in PASS_THROUGH_RUNNERS {
        // `claude-opus-5` ($5/$25) is a prefix of both and must lose to them.
        for model in ["claude-opus-5.5", "claude-opus-5-5"] {
            assert_rates(provider, model, OPUS_5_5, false);
        }
    }
}

#[test]
fn opus_5_and_4_6_through_4_8_bill_five_and_twenty_five() {
    for provider in PASS_THROUGH_RUNNERS {
        for model in [
            "claude-opus-5",
            "claude-opus-4.8",
            "claude-opus-4-8",
            "claude-opus-4.7",
            "claude-opus-4-7",
            "claude-opus-4.6",
            "claude-opus-4-6",
        ] {
            assert_rates(provider, model, OPUS_5_25, false);
        }
    }
}

#[test]
fn opus_4_6_and_4_7_fast_tiers_do_not_fall_to_their_base_rows() {
    // The reference has no list price for these retired fast tiers and says
    // fast mode costs more than standard, so they bill at the $15/$75 Opus 4
    // family rate rather than prefix-matching the $5/$25 base rows.
    for provider in PASS_THROUGH_RUNNERS {
        for model in [
            "claude-opus-4.7-fast",
            "claude-opus-4-7-fast",
            "claude-opus-4.6-fast",
            "claude-opus-4-6-fast",
        ] {
            assert_rates(provider, model, OPUS_4_FAMILY, false);
        }
    }
}

#[test]
fn opus_4_5_and_older_keep_the_opus_4_family_row() {
    for provider in PASS_THROUGH_RUNNERS {
        for model in [
            "claude-opus-4",
            "claude-opus-4.5",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-1",
            "claude-opus-4-0",
        ] {
            assert_rates(provider, model, OPUS_4_FAMILY, false);
        }
    }
}

#[test]
fn sonnet_5_wins_over_the_version_agnostic_sonnet_row() {
    for provider in PASS_THROUGH_RUNNERS {
        assert_rates(provider, "claude-sonnet-5", SONNET_5, false);
        for model in [
            "claude-sonnet-4.6",
            "claude-sonnet-4-6",
            "claude-sonnet-4.5",
            "claude-sonnet-4",
            "claude-sonnet-4-0",
        ] {
            assert_rates(provider, model, SONNET_4, false);
        }
    }
}
