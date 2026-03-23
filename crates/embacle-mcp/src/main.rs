// ABOUTME: CLI entry point for the embacle MCP server binary
// ABOUTME: Parses arguments, selects transport (stdio or HTTP), and starts serving
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use clap::Parser;
use dravr_tronc::server::cli::McpArgs;
use embacle::types::RunnerError;
use embacle_mcp::runner::{parse_runner_type, valid_provider_names};
use embacle_mcp::ServerState;
use tokio::sync::RwLock;

/// embacle-mcp — MCP server exposing embacle LLM runners via Model Context Protocol
#[derive(Parser)]
#[command(name = "embacle-mcp", version, about)]
struct Cli {
    #[command(flatten)]
    server: McpArgs,

    /// Default LLM provider
    ///
    /// Valid: claude, copilot, cursor-agent, opencode, gemini, codex, goose, cline, continue, warp, kiro, kilo
    #[arg(long, default_value = "copilot")]
    provider: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    dravr_tronc::server::tracing_init::init(&cli.server.transport);

    let provider = parse_runner_type(&cli.provider).ok_or_else(|| {
        RunnerError::config(format!(
            "Unknown provider: {}. Valid: {}",
            cli.provider,
            valid_provider_names()
        ))
    })?;

    let state = Arc::new(RwLock::new(ServerState::new(provider)));
    let registry = embacle_mcp::build_tool_registry();
    let server = Arc::new(dravr_tronc::McpServer::new(
        "embacle-mcp",
        env!("CARGO_PKG_VERSION"),
        registry,
        state,
    ));

    tracing::info!(
        transport = %cli.server.transport,
        provider = %provider,
        "Starting embacle MCP server"
    );

    match cli.server.transport.as_str() {
        "stdio" => dravr_tronc::mcp::transport::stdio::run(server).await?,
        "http" => {
            dravr_tronc::mcp::transport::http::serve(server, &cli.server.host, cli.server.port)
                .await?;
        }
        other => {
            return Err(RunnerError::config(format!(
                "Unknown transport: {other}. Valid: stdio, http"
            ))
            .into());
        }
    }

    Ok(())
}
