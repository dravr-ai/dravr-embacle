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

use embacle::pricing::{calculate_cost, calculate_cost_for, TokenCounts};

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
    let cost = calculate_cost_for("copilot_sdk", "claude-opus-4", &counts);

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

    let write_cost = calculate_cost_for("copilot_sdk", "claude-opus-4", &written);
    let fresh_cost = calculate_cost_for("copilot_sdk", "claude-opus-4", &fresh);

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

    let cost_without = calculate_cost_for("copilot_sdk", "claude-opus-4", &without);
    let cost_with = calculate_cost_for("copilot_sdk", "claude-opus-4", &with);

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
    let anthropic = calculate_cost_for("copilot_sdk", "claude-opus-4", &counts);
    // OpenAI 0.50x of $2.50/M = $1.25
    let openai = calculate_cost_for("openai_api", "gpt-4o", &counts);
    // Gemini 0.25x of $0.075/M = $0.01875
    let gemini = calculate_cost_for("gemini", "gemini-2.0-flash", &counts);

    assert!((anthropic - 1.50).abs() < 1e-9, "anthropic={anthropic}");
    assert!((openai - 1.25).abs() < 1e-9, "openai={openai}");
    assert!((gemini - 0.018_75).abs() < 1e-9, "gemini={gemini}");
}

/// A turn reporting reads AND writes bills three prompt segments at three
/// different rates. Shaped on a real Copilot runtime usage payload: 15,320
/// read + 12,540 written out of 27,862 prompt tokens.
#[test]
fn real_copilot_turn_splits_prompt_across_three_rates() {
    let counts = TokenCounts::new(27_862, 4).with_cache(15_320, 12_540);
    let cost = calculate_cost_for("copilot_sdk", "claude-opus-4", &counts);

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
    let naive = calculate_cost_for("copilot_sdk", "claude-opus-4", &TokenCounts::new(27_862, 4));
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
    let cost = calculate_cost_for("copilot_sdk", "claude-opus-4", &counts);

    let input = 15.0 / 1_000_000.0;
    // 900 read, then only 100 left to count as written, then 0 fresh.
    let expected = 900.0f64.mul_add(input * 0.10, 100.0 * input * 1.25);
    assert!(
        (cost - expected).abs() < 1e-12,
        "expected {expected}, got {cost}"
    );
}
