// ABOUTME: CLI entry point for the unified embacle server binary
// ABOUTME: Serves OpenAI REST API + full MCP over HTTP or stdio with config-file and OTel support
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#[cfg(feature = "mcp-tools")]
use std::env;
use std::error::Error;
use std::sync::Arc;

use clap::Parser;
use dravr_tronc::mcp::transport::stdio;
use dravr_tronc::server::tracing_init;
use dravr_tronc::McpServer;
use embacle::types::RunnerError;
use embacle_mcp::ServerState;
use opentelemetry::global;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

#[cfg(feature = "mcp-tools")]
use embacle_server::mcp_client::McpClientPool;
use embacle_server::router;
use embacle_server::runner::{self, parse_runner_type};
use embacle_server::state::{AppState, ServerTools};

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

/// Build server-side tools from configured MCP servers and the
/// `EMBACLE_MCP_SERVERS` environment variable.
///
/// Returns `None` when no servers are configured, the connection fails, or the
/// `mcp-tools` feature is disabled — in all cases the server still starts and
/// serves client-side tool calling.
#[cfg(feature = "mcp-tools")]
async fn build_server_tools(config: Option<&embacle::EmbacleConfig>) -> Option<ServerTools> {
    let mut servers: Vec<embacle::McpServerConfig> =
        config.map(|c| c.mcp_servers.clone()).unwrap_or_default();

    if let Ok(json) = env::var("EMBACLE_MCP_SERVERS") {
        match serde_json::from_str::<Vec<embacle::McpServerConfig>>(&json) {
            Ok(mut from_env) => servers.append(&mut from_env),
            Err(e) => tracing::warn!(error = %e, "Failed to parse EMBACLE_MCP_SERVERS as JSON"),
        }
    }

    if servers.is_empty() {
        return None;
    }

    match McpClientPool::connect(&servers).await {
        Ok(pool) if !pool.is_empty() => {
            let declarations = pool.declarations().to_vec();
            tracing::info!(
                tools = pool.tool_count(),
                "Server-side tool execution enabled"
            );
            Some(ServerTools {
                executor: Arc::new(pool),
                declarations,
            })
        }
        Ok(_) => {
            tracing::warn!("Configured MCP servers exposed no tools; server-side tools disabled");
            None
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to connect MCP tool servers; server-side tools disabled");
            None
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();
    tracing_init::init_with_notifications(&cli.transport);

    // Initialize OpenTelemetry metrics (export pipeline configured via OTEL_EXPORTER_* env vars)
    let meter_provider = SdkMeterProvider::builder().build();
    global::set_meter_provider(meter_provider);

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
    let config = match embacle::load_config() {
        Ok(config) => config,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load config file, using CLI defaults");
            None
        }
    };

    // Only use config file default if CLI --provider was not explicitly set.
    // clap sets the default to "copilot", so if the user didn't specify
    // --provider we check whether the config file supplies an override.
    let effective_provider = match &config {
        Some(cfg) if cli.provider == "copilot" => cfg
            .defaults
            .model
            .as_deref()
            .and_then(parse_runner_type)
            .unwrap_or(effective_provider),
        _ => effective_provider,
    };

    let state = Arc::new(RwLock::new(ServerState::new(effective_provider)));

    tracing::info!(
        transport = %cli.transport,
        provider = %effective_provider,
        "Starting embacle server"
    );

    match cli.transport.as_str() {
        "stdio" => {
            let server = Arc::new(McpServer::new(
                "embacle-mcp",
                env!("CARGO_PKG_VERSION"),
                embacle_mcp::build_tool_registry(),
                Arc::clone(&state),
            ));
            stdio::run(server).await?;
        }
        "http" => {
            #[cfg(feature = "mcp-tools")]
            let server_tools = build_server_tools(config.as_ref()).await;
            #[cfg(not(feature = "mcp-tools"))]
            let server_tools: Option<ServerTools> = None;

            let app_state = AppState::new(state).with_server_tools(server_tools);
            let app = router::build(app_state);
            let addr = format!("{}:{}", cli.host, cli.port);
            let listener = TcpListener::bind(&addr)
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
