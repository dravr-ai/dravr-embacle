// ABOUTME: CLI entry point for the embacle MCP server binary
// ABOUTME: Parses arguments, selects transport (stdio or HTTP), and starts serving
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

mod protocol;
mod runner;
mod server;
mod state;
mod tools;
mod transport;

use std::sync::Arc;

use clap::Parser;
use embacle::types::RunnerError;
use tokio::sync::RwLock;

use runner::parse_runner_type;
use server::McpServer;
use state::ServerState;
use tools::build_tool_registry;
use transport::McpTransport;

/// embacle-mcp — MCP server exposing embacle LLM runners via Model Context Protocol
#[derive(Parser)]
#[command(name = "embacle-mcp", version, about)]
struct Cli {
    /// Transport mode: "stdio" for stdin/stdout or "http" for HTTP+SSE
    #[arg(long, default_value = "stdio")]
    transport: String,

    /// HTTP listen port (only used with --transport http)
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// HTTP listen host (only used with --transport http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Default LLM provider
    ///
    /// Valid: claude, copilot, cursor-agent, opencode, gemini, codex, goose, cline, continue, warp, kiro
    #[arg(long, default_value = "copilot")]
    provider: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Logs go to stderr to keep stdout clean for stdio transport
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    let provider = parse_runner_type(&cli.provider).ok_or_else(|| {
        RunnerError::config(format!(
            "Unknown provider: {}. Valid: {}",
            cli.provider,
            runner::valid_provider_names()
        ))
    })?;

    let state = Arc::new(RwLock::new(ServerState::new(provider)));
    let registry = build_tool_registry();
    let server = Arc::new(McpServer::new(state, registry));

    tracing::info!(
        transport = %cli.transport,
        provider = %provider,
        "Starting embacle MCP server"
    );

    match cli.transport.as_str() {
        "stdio" => {
            transport::stdio::StdioTransport.serve(server).await?;
        }
        "http" => {
            transport::http::HttpTransport::new(cli.host, cli.port)
                .serve(server)
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
