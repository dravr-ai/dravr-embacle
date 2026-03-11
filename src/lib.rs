// ABOUTME: Standalone LLM runner library wrapping AI CLI tools and ACP as providers
// ABOUTME: Re-exports runners, agent loop, fallback chains, metrics, quality gates, MCP bridge, and structured output
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Embacle — LLM Runners
//!
//! Standalone library providing pluggable [`LlmProvider`](types::LlmProvider)
//! implementations that delegate to CLI tools (Claude Code, Copilot, Cursor Agent,
//! `OpenCode`, Gemini, Codex, Goose, Cline, Continue, Warp), an HTTP API client
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
//! ## Quick Start
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use embacle::{ClaudeCodeRunner, RunnerConfig};
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//!
//! # async fn example() -> Result<(), embacle::types::RunnerError> {
//! let config = RunnerConfig::new(PathBuf::from("claude"));
//! let runner = ClaudeCodeRunner::new(config);
//! let request = ChatRequest::new(vec![ChatMessage::user("Hello!")]);
//! let response = runner.complete(&request).await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```
//!
//! ## Fallback Chains
//!
//! Try multiple providers in order — first success wins:
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use embacle::{ClaudeCodeRunner, CopilotRunner, RunnerConfig};
//! use embacle::fallback::FallbackProvider;
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//!
//! # async fn example() -> Result<(), embacle::types::RunnerError> {
//! let claude = ClaudeCodeRunner::new(RunnerConfig::new(PathBuf::from("claude")));
//! let copilot = CopilotRunner::new(RunnerConfig::new(PathBuf::from("copilot"))).await;
//!
//! let provider = FallbackProvider::new(vec![
//!     Box::new(claude),
//!     Box::new(copilot),
//! ])?;
//!
//! // If claude fails, copilot handles it — same interface
//! let request = ChatRequest::new(vec![ChatMessage::user("Hello!")]);
//! let response = provider.complete(&request).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Structured Output
//!
//! Schema-validated JSON from any provider, with retry on validation failure:
//!
//! ```rust,no_run
//! use embacle::structured_output::{request_structured_output, StructuredOutputRequest};
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//! use serde_json::json;
//!
//! # async fn example(runner: &dyn LlmProvider) -> Result<(), embacle::types::RunnerError> {
//! let schema = json!({
//!     "type": "object",
//!     "properties": {
//!         "city": {"type": "string"},
//!         "temperature": {"type": "number"}
//!     },
//!     "required": ["city", "temperature"]
//! });
//!
//! let request = ChatRequest::new(vec![
//!     ChatMessage::user("What's the weather in Paris?"),
//! ]);
//!
//! let data = request_structured_output(
//!     runner,
//!     &StructuredOutputRequest { request, schema, max_retries: 2 },
//! ).await?;
//!
//! assert!(data["city"].is_string());
//! assert!(data["temperature"].is_number());
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! ### Core
//!
//! - [`types`] — `LlmProvider` trait, messages, requests, responses, errors
//! - [`config`] — `RunnerConfig`, `CliRunnerType` enum
//! - [`factory`] — Runner factory, provider parsing, `ALL_PROVIDERS` constant
//!
//! ### Higher-Level Features
//!
//! - [`agent`] — Multi-turn agent loop with configurable tool calling
//! - [`fallback`] — Ordered provider failover chains
//! - [`metrics`] — Latency, token, and error tracking decorator
//! - [`quality_gate`] — Response validation with retry on refusal
//! - [`structured_output`] — Schema-enforced JSON extraction from any provider
//! - [`tool_simulation`] — XML-based text tool calling for CLI runners without native function calling
//! - [`mcp_tool_bridge`] — MCP tool definitions to text-tool-simulation bridge
//! - [`capability_guard`] — Request/provider capability validation
//!
//! ### Runner Infrastructure
//!
//! - [`auth`] — Readiness and authentication checking
//! - [`discovery`] — Automatic binary detection on the host
//! - [`process`] — Subprocess spawning with timeout and output limits
//! - [`sandbox`] — Environment variable whitelisting and working directory control
//! - [`prompt`] — Prompt building from `ChatMessage` slices
//! - [`compat`] — Version compatibility and capability detection
//! - [`container`] — Container-based execution backend
//!
//! ### CLI Runners
//!
//! - [`claude_code`] — Claude Code CLI runner
//! - [`copilot`] — GitHub Copilot CLI runner
//! - [`cursor_agent`] — Cursor Agent CLI runner
//! - [`opencode`] — `OpenCode` CLI runner
//! - [`gemini_cli`] — Gemini CLI runner
//! - [`codex_cli`] — Codex CLI runner
//! - [`goose_cli`] — Goose CLI runner
//! - [`cline_cli`] — Cline CLI runner
//! - [`continue_cli`] — Continue CLI runner
//! - [`warp_cli`] — Warp terminal `oz` CLI runner
//!
//! ### Feature-Flagged Runners
//!
//! - `openai_api` — OpenAI-compatible HTTP API client (requires `openai-api` feature)
//! - `copilot_headless` — GitHub Copilot Headless ACP runner (requires `copilot-headless` feature)

/// Core types: traits, messages, requests, responses, and errors
pub mod types;

/// Configurable agent loop with multi-turn tool calling
pub mod agent;
/// Auth readiness checking for CLI runners
pub mod auth;
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
/// Warp terminal `oz` CLI runner
pub mod warp_cli;

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

// Re-export the runner structs for ergonomic access
pub use agent::{AgentExecutor, AgentResult, OnTurnCallback, TurnInfo};
pub use auth::ProviderReadiness;
pub use capability_guard::validate_capabilities;
pub use claude_code::ClaudeCodeRunner;
pub use cli_common::CliRunnerBase;
pub use cline_cli::ClineCliRunner;
pub use codex_cli::CodexCliRunner;
pub use compat::CliCapabilities;
pub use config::{CliRunnerType, RunnerConfig};
pub use container::{ContainerConfig, ContainerExecutor, NetworkMode};
pub use continue_cli::ContinueCliRunner;
pub use copilot::{copilot_fallback_models, discover_copilot_models, CopilotRunner};
pub use cursor_agent::CursorAgentRunner;
pub use discovery::{discover_runner, resolve_binary};
pub use factory::{create_runner, parse_runner_type, valid_provider_names, ALL_PROVIDERS};
pub use fallback::FallbackProvider;
pub use gemini_cli::GeminiCliRunner;
pub use goose_cli::GooseCliRunner;
pub use mcp_tool_bridge::{McpToolDefinition, McpToolExecutor};
pub use metrics::{MetricsProvider, MetricsReport};
pub use opencode::OpenCodeRunner;
pub use quality_gate::{QualityGateProvider, QualityPolicy};
pub use structured_output::{request_structured_output, StructuredOutputRequest};
pub use warp_cli::WarpCliRunner;

// Core tool calling type re-exports
pub use types::{ResponseFormat, ToolCallRequest, ToolChoice, ToolDefinition};

// Tool simulation re-exports
pub use tool_simulation::{
    execute_with_text_tools, format_tool_results_as_text, generate_tool_catalog,
    inject_tool_catalog, parse_tool_call_blocks, strip_tool_call_blocks, FunctionCall,
    FunctionDeclaration, FunctionResponse, TextToolHandler, TextToolResponse,
};

// OpenAI API re-exports (behind feature flag)
#[cfg(feature = "openai-api")]
pub use openai_api::{OpenAiApiConfig, OpenAiApiRunner};

// Copilot Headless re-exports (behind feature flag)
#[cfg(feature = "copilot-headless")]
pub use copilot_headless::{CopilotHeadlessRunner, HeadlessToolResponse, ObservedToolCall};
#[cfg(feature = "copilot-headless")]
pub use copilot_headless_config::{CopilotHeadlessConfig, PermissionPolicy};
