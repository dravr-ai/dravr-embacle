// ABOUTME: Standalone LLM runner library wrapping AI CLI tools and ACP as providers
// ABOUTME: Re-exports runners, agent loop, fallback chains, metrics, quality gates, MCP bridge, and structured output
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Embacle — LLM Runners
//!
//! Standalone library providing pluggable [`LlmProvider`](types::LlmProvider)
//! implementations that delegate to CLI tools (Claude Code, Copilot, Cursor Agent,
//! `OpenCode`, Gemini, Codex, Goose, Cline, Continue, Warp, Kiro, Kilo Code), an HTTP API client
//! (OpenAI-compatible), and ACP (Copilot Headless) for LLM completions.
//!
//! CLI runners wrap a binary, build prompts from [`ChatMessage`](types::ChatMessage)
//! sequences, parse JSON output, and manage session continuity. The Copilot Headless
//! runner communicates via NDJSON-framed JSON-RPC with `copilot --acp`.
//!
//! Two companion binary crates build on this library:
//! - **`embacle-server`** — OpenAI-compatible REST API + MCP Streamable HTTP on a single port
//! - **`embacle-mcp`** — standalone MCP server over stdio or HTTP
//!
//! ## Progress feedback (AG-UI)
//!
//! With the `agui` feature enabled, the [`agui`] module exposes the
//! [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) event vocabulary
//! plus an emitter trait so downstream runners, agent loops, and pipelines
//! can report run/step/tool/text progress to user-facing clients without
//! coupling to a specific transport.

/// Core types: traits, messages, requests, responses, and errors
pub mod types;

/// Configurable agent loop with multi-turn tool calling
pub mod agent;
/// AG-UI protocol event schema (feature-gated).
///
/// Canonical event types, filter configuration, and emitter trait for
/// streaming agent progress to user-facing clients. See module docs.
#[cfg(feature = "agui")]
pub mod agui;
/// Auth readiness checking for CLI runners
pub mod auth;
/// Response caching decorator
pub mod cache;
/// Request/provider capability validation
pub mod capability_guard;
/// Claude Code CLI runner
pub mod claude_code;
/// Shared base struct and macro for CLI runner boilerplate
pub mod cli_common;
/// Cline CLI runner
pub mod cline_cli;
/// Codex CLI runner
pub mod codex_cli;
/// Version compatibility and capability detection
pub mod compat;
/// Shared configuration types for CLI runners
pub mod config;
/// Container-based execution backend
pub mod container;
/// Continue CLI runner
pub mod continue_cli;
/// GitHub Copilot CLI runner
pub mod copilot;
/// Ranked catalog of Copilot-served models for intelligent default selection and self-heal
pub mod copilot_models;
/// Cursor Agent CLI runner
pub mod cursor_agent;
/// Binary auto-detection and discovery
pub mod discovery;
/// Runner factory, provider parsing, and provider enumeration
pub mod factory;
/// Provider fallback chains
pub mod fallback;
/// Gemini CLI runner
pub mod gemini_cli;
/// Goose CLI runner
pub mod goose_cli;
/// Pluggable guardrail middleware for request/response validation
pub mod guardrail;
/// Kilo Code CLI runner
pub mod kilo_cli;
/// Kiro CLI runner
pub mod kiro_cli;
/// MCP tool definition to text-tool-simulation bridge
pub mod mcp_tool_bridge;
/// Cost/latency normalization decorator
pub mod metrics;
/// `OpenCode` CLI runner
pub mod opencode;
/// Subprocess spawning with safety limits
pub mod process;
/// Prompt construction from `ChatMessage` sequences
pub mod prompt;
/// Response quality validation with retry
pub mod quality_gate;
/// Environment sandboxing and tool policy
pub mod sandbox;
/// Stream wrapper for child process lifecycle management
pub mod stream;
/// Schema-enforced JSON output from any provider
pub mod structured_output;
/// Text-based tool simulation for CLI runners without native function calling
pub mod tool_simulation;
/// Conversation-turn correlation identifier threaded through a user utterance
pub mod turn;
/// Warp terminal `oz` CLI runner
pub mod warp_cli;

// Config file module (behind feature flag)
/// TOML-based declarative configuration file loading
#[cfg(feature = "config-file")]
pub mod config_file;

// OpenAI API module (behind feature flag)
/// OpenAI-compatible HTTP API client runner
#[cfg(feature = "openai-api")]
pub mod openai_api;

// Copilot Headless modules (behind feature flag)
/// Configuration for the Copilot Headless (ACP) provider
#[cfg(feature = "copilot-headless")]
pub mod copilot_headless;
/// Configuration types for the Copilot Headless provider
#[cfg(feature = "copilot-headless")]
pub mod copilot_headless_config;

// C FFI bindings (behind feature flag)
#[cfg(feature = "ffi")]
#[allow(unsafe_code)]
mod ffi;

// Re-export the runner structs for ergonomic access
pub use agent::{AgentExecutor, AgentResult, OnTurnCallback, TurnInfo};
pub use auth::ProviderReadiness;
pub use cache::{CacheConfig, CacheProvider, CacheStats};
pub use capability_guard::validate_capabilities;
pub use claude_code::ClaudeCodeRunner;
pub use cli_common::CliRunnerBase;
pub use cline_cli::ClineCliRunner;
pub use codex_cli::CodexCliRunner;
pub use compat::{CliCapabilities, CliFeatureFlags};
pub use config::{CliRunnerType, RunnerConfig};
pub use container::{ContainerConfig, ContainerExecutor, NetworkMode};
pub use continue_cli::ContinueCliRunner;
pub use copilot::{copilot_fallback_models, CopilotRunner};
pub use cursor_agent::CursorAgentRunner;
pub use discovery::{discover_runner, resolve_binary};
pub use factory::{
    create_runner, create_runner_with_config, parse_runner_type, valid_provider_names,
    ALL_PROVIDERS,
};
pub use fallback::{FallbackProvider, RetryConfig};
pub use gemini_cli::GeminiCliRunner;
pub use goose_cli::GooseCliRunner;
pub use guardrail::{
    ContentLengthGuardrail, Guardrail, GuardrailProvider, GuardrailViolation, PiiScrubGuardrail,
    TopicFilterGuardrail,
};
pub use kilo_cli::KiloCliRunner;
pub use kiro_cli::KiroCliRunner;
pub use mcp_tool_bridge::{McpToolDefinition, McpToolExecutor};
pub use metrics::{
    default_pricing_table, MetricsProvider, MetricsReport, PricingTable, TokenPricing,
};
pub use opencode::OpenCodeRunner;
pub use quality_gate::{QualityGateProvider, QualityPolicy};
pub use structured_output::{
    extract_json_from_response, request_structured_output, StructuredOutputRequest,
};
pub use warp_cli::WarpCliRunner;

// Core tool calling type re-exports
pub use types::{ImagePart, ResponseFormat, ToolCallRequest, ToolChoice, ToolDefinition};

// Conversation-turn correlation re-export
pub use turn::ConversationTurnId;

// Tool simulation re-exports
pub use tool_simulation::{
    execute_with_text_tools, format_tool_results_as_text, generate_tool_catalog,
    inject_tool_catalog, parse_tool_call_blocks, strip_tool_call_blocks, FunctionCall,
    FunctionDeclaration, FunctionResponse, TextToolHandler, TextToolResponse,
};

// AG-UI re-exports (behind feature flag)
#[cfg(feature = "agui")]
pub use agui::{AgUiEmitter, AgUiEvent, AgUiEventFilter, AgUiEventKind, NoopEmitter};

// Config file re-exports (behind feature flag)
#[cfg(feature = "config-file")]
pub use config_file::{
    build_fallback_from_config, build_runner_config, load_config, load_config_from, resolve_alias,
    DefaultsConfig, EmbacleConfig, FallbackConfig, ProviderConfig,
};

// OpenAI API re-exports (behind feature flag)
#[cfg(feature = "openai-api")]
pub use openai_api::{OpenAiApiConfig, OpenAiApiRunner};

// Copilot Headless re-exports (behind feature flag)
#[cfg(feature = "copilot-headless")]
pub use copilot_headless::{CopilotHeadlessRunner, HeadlessToolResponse, ObservedToolCall};
#[cfg(feature = "copilot-headless")]
pub use copilot_headless_config::{
    CopilotHeadlessConfig, PermissionPolicy, DEFAULT_MAX_HISTORY_TURNS,
};
