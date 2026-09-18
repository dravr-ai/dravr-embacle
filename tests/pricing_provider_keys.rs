// ABOUTME: Every pricing key and flat-rate entry must be a string a provider actually reports
// ABOUTME: A key spelled any other way is unreachable and its models silently bill at zero
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Usage records carry `LlmProvider::name()`. `lookup_pricing` matches that
//! string by equality, and `is_not_per_token_metered` by membership — so a
//! table keyed on any other spelling is not "slightly off", it is dead: every
//! model under it resolves to \$0.
//!
//! The spelling is not guessable from the selector — `CliRunnerType`'s
//! `Display` prints `claude_code` while the runner reports `claude-code`, and
//! only `name()` decides what a usage record carries. So the names here are
//! *derived*, not restated: each one comes from constructing the provider and
//! asking it. A release that renames a runner or adds one fails this file
//! rather than silently unpricing a provider.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
#![cfg(all(
    feature = "http-api",
    feature = "openai-api",
    feature = "copilot-headless",
    feature = "copilot-sdk",
    feature = "web-ui"
))]

use std::collections::BTreeSet;
use std::path::PathBuf;

use embacle::http_api::{
    CohereConfig, CohereProvider, GeminiConfig, GeminiProvider, GroqConfig, GroqProvider,
    OpenAiCompatibleConfig, OpenAiCompatibleProvider, OpenRouterConfig, OpenRouterProvider,
};
use embacle::pricing::{
    calculate_cost, is_not_per_token_metered, NOT_PER_TOKEN_METERED_PROVIDERS, PRICING_TABLE,
};
use embacle::types::LlmProvider;
use embacle::{
    ClaudeCodeRunner, CliRunnerType, ClineCliRunner, CodexCliRunner, ContinueCliRunner,
    CopilotHeadlessRunner, CopilotRunner, CopilotSdkRunner, CursorAgentRunner, GeminiCliRunner,
    GooseCliRunner, KiloCliRunner, KiroCliRunner, OpenAiApiConfig, OpenAiApiRunner, OpenCodeRunner,
    RunnerConfig, WarpCliRunner, WebProviderConfig, WebUiConfig, WebUiRunner,
};

/// Every CLI runner the factory can construct. This file requires every
/// runner feature, so the match below is exhaustive over the whole enum and a
/// release that adds a runner fails to compile here instead of escaping the
/// check.
const EVERY_CLI_RUNNER: &[CliRunnerType] = &[
    CliRunnerType::ClaudeCode,
    CliRunnerType::CursorAgent,
    CliRunnerType::OpenCode,
    CliRunnerType::Copilot,
    CliRunnerType::GeminiCli,
    CliRunnerType::CodexCli,
    CliRunnerType::GooseCli,
    CliRunnerType::ClineCli,
    CliRunnerType::ContinueCli,
    CliRunnerType::WarpCli,
    CliRunnerType::KiroCli,
    CliRunnerType::KiloCli,
    CliRunnerType::CopilotHeadless,
    CliRunnerType::CopilotSdk,
    CliRunnerType::ClaudeWeb,
];

/// The string this runner puts on a usage record.
fn cli_runner_name(kind: CliRunnerType) -> &'static str {
    let config = || RunnerConfig::new(PathBuf::from("/nonexistent"));
    match kind {
        CliRunnerType::ClaudeCode => ClaudeCodeRunner::new(config()).name(),
        CliRunnerType::CursorAgent => CursorAgentRunner::new(config()).name(),
        CliRunnerType::OpenCode => OpenCodeRunner::new(config()).name(),
        CliRunnerType::Copilot => CopilotRunner::new(config()).name(),
        CliRunnerType::GeminiCli => GeminiCliRunner::new(config()).name(),
        CliRunnerType::CodexCli => CodexCliRunner::new(config()).name(),
        CliRunnerType::GooseCli => GooseCliRunner::new(config()).name(),
        CliRunnerType::ClineCli => ClineCliRunner::new(config()).name(),
        CliRunnerType::ContinueCli => ContinueCliRunner::new(config()).name(),
        CliRunnerType::WarpCli => WarpCliRunner::new(config()).name(),
        CliRunnerType::KiroCli => KiroCliRunner::new(config()).name(),
        CliRunnerType::KiloCli => KiloCliRunner::new(config()).name(),
        CliRunnerType::CopilotHeadless => CopilotHeadlessRunner::from_env().name(),
        CliRunnerType::CopilotSdk => CopilotSdkRunner::from_env().name(),
        CliRunnerType::ClaudeWeb => WebUiRunner::new(
            WebUiConfig::default(),
            WebProviderConfig::claude_web_default().expect("the embedded provider config parses"),
        )
        .name(),
    }
}

/// The name a self-hosted endpoint reports under this `provider_name`.
fn local_name(provider_name: &str) -> &'static str {
    OpenAiCompatibleProvider::with_client(
        OpenAiCompatibleConfig {
            provider_name: provider_name.to_owned(),
            display_name: provider_name.to_owned(),
            ..OpenAiCompatibleConfig::default()
        },
        reqwest::Client::new(),
    )
    .name()
}

/// The providers that call a vendor API directly, asked for their own names.
async fn native_provider_names() -> Vec<&'static str> {
    let client = reqwest::Client::new();
    vec![
        GeminiProvider::with_client(GeminiConfig::new("test-key"), client.clone()).name(),
        GroqProvider::with_client(GroqConfig::new("test-key"), client.clone()).name(),
        CohereProvider::with_client(CohereConfig::new("test-key"), client.clone()).name(),
        OpenRouterProvider::with_client(OpenRouterConfig::new("test-key"), client.clone()).name(),
        local_name("ollama"),
        local_name("vllm"),
        local_name("localai"),
        // Anything the config does not name explicitly reports as "local".
        local_name("some-self-hosted-endpoint"),
        // Model discovery against a closed port fails fast and falls back to
        // the configured model; the name is what matters here.
        OpenAiApiRunner::with_client(OpenAiApiConfig::new("http://127.0.0.1:9"), client)
            .await
            .name(),
    ]
}

/// Every string a provider can put on a usage record.
async fn every_provider_name() -> BTreeSet<&'static str> {
    EVERY_CLI_RUNNER
        .iter()
        .copied()
        .map(cli_runner_name)
        .chain(native_provider_names().await)
        .collect()
}

#[tokio::test]
async fn every_pricing_key_is_a_name_a_provider_reports() {
    let reported = every_provider_name().await;
    for (provider, model_prefix, _) in PRICING_TABLE {
        assert!(
            reported.contains(provider),
            "PRICING_TABLE keys ({provider}, {model_prefix}) on a string no provider reports; \
             lookup_pricing matches by equality, so that row is unreachable and every model \
             under it bills $0. Known names: {reported:?}"
        );
    }
}

#[tokio::test]
async fn every_not_metered_entry_is_a_name_a_provider_reports() {
    let reported = every_provider_name().await;
    for provider in NOT_PER_TOKEN_METERED_PROVIDERS {
        assert!(
            reported.contains(provider),
            "NOT_PER_TOKEN_METERED_PROVIDERS lists {provider}, which no provider reports; \
             the suppression it is meant to apply never fires. Known names: {reported:?}"
        );
    }
}

#[test]
fn the_claude_code_rows_are_reachable_from_the_name_the_runner_reports() {
    let reported = cli_runner_name(CliRunnerType::ClaudeCode);
    assert_eq!(
        reported, "claude-code",
        "the pricing rows are keyed on this exact string"
    );

    // One million input tokens at the Sonnet rate is $3.00 exactly.
    let sonnet = calculate_cost(reported, "claude-sonnet-4.5", 1_000_000, 0);
    assert!(
        (sonnet - 3.0).abs() < 1e-9,
        "claude-sonnet must price at $3/M input; got {sonnet}"
    );
    let opus = calculate_cost(reported, "claude-opus-4-1", 0, 1_000_000);
    assert!(
        (opus - 75.0).abs() < 1e-9,
        "claude-opus-4 must price at $75/M output; got {opus}"
    );
    let haiku = calculate_cost(reported, "claude-haiku-4-5", 1_000_000, 0);
    assert!(
        (haiku - 0.80).abs() < 1e-9,
        "claude-haiku-4 must price at $0.80/M input; got {haiku}"
    );
}

#[test]
fn claude_code_is_priced_rather_than_suppressed() {
    assert!(
        !is_not_per_token_metered("claude-code"),
        "claude-code carries real PRICING_TABLE rows; suppressing it would zero its cost \
         and hide the miss behind the by-design $0 path"
    );
    assert!(
        !is_not_per_token_metered("copilot_headless"),
        "copilot_headless is priced for the same reason"
    );
}

#[test]
fn every_flat_rate_runner_the_table_does_not_price_is_suppressed() {
    // The complement of the rule above: a runner with no PRICING_TABLE rows
    // bills $0, and that $0 must be classified as correct rather than logged as
    // an undercount.
    let priced: BTreeSet<&str> = PRICING_TABLE.iter().map(|(p, _, _)| *p).collect();
    for kind in EVERY_CLI_RUNNER.iter().copied() {
        let name = cli_runner_name(kind);
        if priced.contains(name) {
            continue;
        }
        assert!(
            is_not_per_token_metered(name),
            "{name} has no pricing rows, so every one of its turns bills $0 — it must be \
             classified as subscription-billed or the cost path logs a false undercount"
        );
    }
}
