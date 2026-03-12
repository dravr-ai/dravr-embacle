// ABOUTME: Library root re-exporting server modules for integration testing
// ABOUTME: Enables tests/ to access router, state, types, and handler modules
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # embacle-server
//!
//! Unified OpenAI-compatible REST API + MCP server for embacle LLM runners.
//!
//! ## Endpoints
//!
//! - `POST /v1/chat/completions` — chat completion (streaming and non-streaming)
//! - `GET /v1/models` — list available providers and models
//! - `GET /health` — per-provider readiness check
//! - `POST /mcp` — MCP Streamable HTTP (JSON-RPC 2.0, via embacle-mcp)
//!
//! ## Modules
//!
//! - [`completions`] — chat completion handler with multiplex fan-out
//! - [`models`] — model listing endpoint
//! - [`health`] — provider health checks
//! - [`auth`] — optional bearer token authentication
//! - [`streaming`] — SSE streaming for OpenAI-format responses
//! - [`provider_resolver`] — `provider:model` string routing
//! - [`router`] — Axum router wiring all endpoints (`OpenAI` + MCP)
//! - [`state`] — re-export of unified state from embacle-mcp
//! - [`runner`] — runner factory bridging to embacle core

pub mod auth;
pub mod completions;
pub mod health;
pub mod models;
pub mod openai_types;
pub mod provider_resolver;
pub mod router;
pub mod runner;
pub mod state;
pub mod streaming;
