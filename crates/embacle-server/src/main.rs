// ABOUTME: CLI entry point for the unified embacle server binary
// ABOUTME: Serves OpenAI REST API + full MCP over HTTP or stdio with config-file and OTel support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use clap::Parser;
use embacle::types::RunnerError;
use embacle_mcp::ServerState;
use tokio::sync::RwLock;

use embacle_server::router;
use embacle_server::runner::{self, parse_runner_type};

/// embacle-server — OpenAI-compatible REST API + MCP server for embacle LLM runners
#[derive(Parser)]
#[command(name = "embacle-server", version, about)]
struct Cli {
    /// Transport mode: "http" for REST API + MCP, "stdio" for MCP-only stdin/stdout
    #[arg(long, default_value = "http")]
    transport: String,

    /// HTTP listen port (only used with --transport http)
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// HTTP listen host (only used with --transport http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Default LLM provider
    ///
    /// Valid: claude, copilot, cursor-agent, opencode, gemini, codex, goose, cline, continue, warp, kiro, kilo
    #[arg(long, default_value = "copilot")]
    provider: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    dravr_tronc::server::tracing_init::init(&cli.transport);

    // Initialize OpenTelemetry metrics (export pipeline configured via OTEL_EXPORTER_* env vars)
    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder().build();
    opentelemetry::global::set_meter_provider(meter_provider);

    // Determine effective provider: CLI arg takes precedence, then config file, then default
    let Some(effective_provider) = parse_runner_type(&cli.provider) else {
        return Err(RunnerError::config(format!(
            "Unknown provider: {}. Valid: {}",
            cli.provider,
            runner::valid_provider_names()
        ))
        .into());
    };

    // Try loading config file (best-effort — works without one)
    let effective_provider = match embacle::load_config() {
        Ok(Some(ref cfg)) => {
            // Only use config file default if CLI --provider was not explicitly set
            // clap sets the default to "copilot", so if the user didn't specify --provider
            // we check if there's a config file override
            if cli.provider == "copilot" {
                cfg.defaults
                    .model
                    .as_deref()
                    .and_then(parse_runner_type)
                    .unwrap_or(effective_provider)
            } else {
                effective_provider
            }
        }
        Ok(None) => effective_provider,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load config file, using CLI defaults");
            effective_provider
        }
    };

    let state = Arc::new(RwLock::new(ServerState::new(effective_provider)));

    tracing::info!(
        transport = %cli.transport,
        provider = %effective_provider,
        "Starting embacle server"
    );

    match cli.transport.as_str() {
        "stdio" => {
            let server = Arc::new(dravr_tronc::McpServer::new(
                "embacle-mcp",
                env!("CARGO_PKG_VERSION"),
                embacle_mcp::build_tool_registry(),
                Arc::clone(&state),
            ));
            dravr_tronc::mcp::transport::stdio::run(server).await?;
        }
        "http" => {
            let app = router::build(state);
            let addr = format!("{}:{}", cli.host, cli.port);
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| RunnerError::internal(format!("Failed to bind {addr}: {e}")))?;

            tracing::info!(address = %addr, "HTTP transport listening");

            axum::serve(listener, app)
                .await
                .map_err(|e| RunnerError::internal(format!("Server error: {e}")))?;
        }
        other => {
            return Err(RunnerError::config(format!(
                "Unknown transport: {other}. Valid: http, stdio"
            ))
            .into());
        }
    }

    Ok(())
}
