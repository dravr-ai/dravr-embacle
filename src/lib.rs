// ABOUTME: Standalone LLM runner library wrapping AI CLI tools and SDKs as providers
// ABOUTME: Re-exports runners for Claude Code, Copilot, Cursor Agent, OpenCode, Gemini, Codex, and Copilot SDK
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Embacle — LLM Runners
//!
//! Standalone library providing pluggable [`LlmProvider`](types::LlmProvider)
//! implementations that delegate to CLI tools (Claude Code, Copilot, Cursor Agent,
//! `OpenCode`, Gemini, Codex) and SDKs (Copilot SDK) for LLM completions.
//!
//! CLI runners wrap a binary, build prompts from [`ChatMessage`](types::ChatMessage)
//! sequences, parse JSON output, and manage session continuity. The Copilot SDK
//! runner maintains a persistent `copilot --headless` server via JSON-RPC.
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
//! ## Modules
//!
//! - [`types`] — Core types: `LlmProvider` trait, messages, requests, errors
//! - [`config`] — Runner types and configuration
//! - [`compat`] — Version compatibility and capability detection
//! - [`container`] — Container-based execution backend
//! - [`discovery`] — Automatic binary detection on the host
//! - [`auth`] — Readiness and authentication checking
//! - [`process`] — Subprocess spawning with timeout and output limits
//! - [`sandbox`] — Environment variable whitelisting and working directory control
//! - [`prompt`] — Prompt building from `ChatMessage` slices
//! - [`claude_code`] — Claude Code CLI runner
//! - [`copilot`] — GitHub Copilot CLI runner
//! - [`cursor_agent`] — Cursor Agent CLI runner
//! - [`opencode`] — `OpenCode` CLI runner
//! - [`gemini_cli`] — Gemini CLI runner
//! - [`codex_cli`] — Codex CLI runner
//! - [`copilot_sdk_runner`] — GitHub Copilot SDK runner (requires `copilot-sdk` feature)

/// Core types: traits, messages, requests, responses, and errors
pub mod types;

/// Auth readiness checking for CLI runners
pub mod auth;
/// Claude Code CLI runner
pub mod claude_code;
/// Codex CLI runner
pub mod codex_cli;
/// Version compatibility and capability detection
pub mod compat;
/// Shared configuration types for CLI runners
pub mod config;
/// Container-based execution backend
pub mod container;
/// GitHub Copilot CLI runner
pub mod copilot;
/// Cursor Agent CLI runner
pub mod cursor_agent;
/// Binary auto-detection and discovery
pub mod discovery;
/// Gemini CLI runner
pub mod gemini_cli;
/// `OpenCode` CLI runner
pub mod opencode;
/// Subprocess spawning with safety limits
pub mod process;
/// Prompt construction from `ChatMessage` sequences
pub mod prompt;
/// Environment sandboxing and tool policy
pub mod sandbox;
/// Stream wrapper for child process lifecycle management
pub mod stream;
/// Text-based tool simulation for CLI runners without native function calling
pub mod tool_simulation;

// Copilot SDK modules (behind feature flag)
/// Configuration for the Copilot SDK provider
#[cfg(feature = "copilot-sdk")]
pub mod copilot_sdk_config;
/// GitHub Copilot SDK runner (persistent JSON-RPC server)
#[cfg(feature = "copilot-sdk")]
pub mod copilot_sdk_runner;
/// Tool definition conversion for Copilot SDK native tool calling
#[cfg(feature = "copilot-sdk")]
pub mod tool_bridge;

// Re-export the runner structs for ergonomic access
pub use auth::ProviderReadiness;
pub use claude_code::ClaudeCodeRunner;
pub use codex_cli::CodexCliRunner;
pub use compat::CliCapabilities;
pub use config::{CliRunnerType, RunnerConfig};
pub use container::{ContainerConfig, ContainerExecutor, NetworkMode};
pub use copilot::CopilotRunner;
pub use cursor_agent::CursorAgentRunner;
pub use discovery::{discover_runner, resolve_binary};
pub use gemini_cli::GeminiCliRunner;
pub use opencode::OpenCodeRunner;

// Tool simulation re-exports
pub use tool_simulation::{
    execute_with_text_tools, format_tool_results_as_text, generate_tool_catalog,
    inject_tool_catalog, parse_tool_call_blocks, strip_tool_call_blocks, FunctionCall,
    FunctionDeclaration, FunctionResponse, TextToolHandler, TextToolResponse,
};

// Copilot SDK re-exports (behind feature flag)
#[cfg(feature = "copilot-sdk")]
pub use copilot_sdk_config::CopilotSdkConfig;
#[cfg(feature = "copilot-sdk")]
pub use copilot_sdk_runner::{CopilotSdkRunner, SdkToolResponse};
#[cfg(feature = "copilot-sdk")]
pub use tool_bridge::{convert_function_declarations, extract_declarations_from_tool_value};

// Re-export copilot-sdk types so consumers don't need a direct dependency
#[cfg(feature = "copilot-sdk")]
pub use copilot_sdk::{Tool as SdkTool, ToolHandler, ToolResultObject};
