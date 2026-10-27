// ABOUTME: Integration tests for the embacle core library, linked as one test binary
// ABOUTME: One module per suite: runners, HTTP providers, fallback, pricing, quota and AG-UI
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

// The unix suites exec a script they just wrote; `support` retries the spawn
// through the ETXTBSY race a sibling test's fork opens.
#[cfg(unix)]
#[path = "../support/mod.rs"]
mod support;

mod agui;
mod anthropic_pool;
mod claude_code_result_document;
mod cli_argv_secrets;
mod credential_pool;
mod e2e;
mod e2e_cli_notice;
mod e2e_router;
mod fallback_observer;
mod fallback_policy;
mod function_call_serde;
mod github_rate_limit;
mod http_model_override;
mod http_provider_metadata;
mod http_sse_parser;
mod permission_policy_defaults;
mod pricing;
mod pricing_provider_keys;
mod process_kill_on_drop;
mod quota_store_redis;
mod request_model;
mod scenario_tests;
mod stateless_runners;
mod stream_deadline;
mod tool_stream;
