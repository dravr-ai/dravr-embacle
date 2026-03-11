// ABOUTME: CLI entry point for the embacle REST API server binary
// ABOUTME: Parses arguments, creates shared state, and starts the axum HTTP server
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use clap::Parser;
use embacle::types::RunnerError;

use embacle_server::router;
use embacle_server::runner::{self, parse_runner_type};
use embacle_server::state::ServerState;

/// embacle-server — OpenAI-compatible REST API + MCP server for embacle LLM runners
#[derive(Parser)]
#[command(name = "embacle-server", version, about)]
struct Cli {
    /// HTTP listen port
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// HTTP listen host
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let provider = parse_runner_type(&cli.provider).ok_or_else(|| {
        RunnerError::config(format!(
            "Unknown provider: {}. Valid: {}",
            cli.provider,
            runner::valid_provider_names()
        ))
    })?;

    let state = Arc::new(ServerState::new(provider));
    let app = router::build(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| RunnerError::internal(format!("Failed to bind {addr}: {e}")))?;

    tracing::info!(
        address = %addr,
        provider = %provider,
        "Starting embacle server (OpenAI API + MCP)"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| RunnerError::internal(format!("Server error: {e}")))?;

    Ok(())
}
