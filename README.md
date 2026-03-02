# Embacle — LLM Runners

[![CI](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml/badge.svg)](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

Standalone Rust library that wraps AI CLI tools and SDKs as pluggable LLM providers.

Instead of integrating with LLM APIs directly (which require API keys, SDKs, and managing auth), **Embacle** delegates to CLI tools that users already have installed and authenticated — getting model upgrades, auth management, and protocol handling for free. For GitHub Copilot, an optional SDK mode maintains a persistent JSON-RPC connection for native tool calling.

## Supported Runners

### CLI Runners (subprocess-based)

| Runner | Binary | Features |
|--------|--------|----------|
| Claude Code | `claude` | JSON output, streaming, system prompts, session resume |
| GitHub Copilot | `copilot` | Text parsing, streaming |
| Cursor Agent | `cursor-agent` | JSON output, streaming, MCP approval |
| OpenCode | `opencode` | JSON events, session management |
| Gemini CLI | `gemini` | JSON/stream-JSON output, streaming, session resume |
| Codex CLI | `codex` | JSONL output, streaming, sandboxed exec mode |

### SDK Runners (persistent connection)

| Runner | Feature Flag | Features |
|--------|-------------|----------|
| GitHub Copilot SDK | `copilot-sdk` | Persistent JSON-RPC via `copilot --headless`, native tool calling, streaming |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
embacle = { git = "https://github.com/dravr-ai/dravr-embacle.git", branch = "main" }
```

Use a CLI runner:

```rust
use std::path::PathBuf;
use embacle::{ClaudeCodeRunner, RunnerConfig};
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};

#[tokio::main]
async fn main() -> Result<(), embacle::types::RunnerError> {
    let config = RunnerConfig::new(PathBuf::from("claude"));
    let runner = ClaudeCodeRunner::new(config);

    let request = ChatRequest::new(vec![
        ChatMessage::user("What is the capital of France?"),
    ]);

    let response = runner.complete(&request).await?;
    println!("{}", response.content);
    Ok(())
}
```

### Copilot SDK (feature flag)

Enable the `copilot-sdk` feature for persistent JSON-RPC instead of per-request subprocesses:

```toml
[dependencies]
embacle = { git = "https://github.com/dravr-ai/dravr-embacle.git", branch = "main", features = ["copilot-sdk"] }
```

```rust
use embacle::{CopilotSdkRunner, CopilotSdkConfig};
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};

#[tokio::main]
async fn main() -> Result<(), embacle::types::RunnerError> {
    // Reads COPILOT_SDK_MODEL, COPILOT_GITHUB_TOKEN, etc. from env
    let runner = CopilotSdkRunner::from_env();

    let request = ChatRequest::new(vec![
        ChatMessage::user("Explain Rust ownership"),
    ]);

    let response = runner.complete(&request).await?;
    println!("{}", response.content);
    Ok(())
}
```

The SDK runner starts `copilot --headless` once and reuses the connection across requests. Configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COPILOT_CLI_PATH` | auto-detect | Override path to copilot binary |
| `COPILOT_SDK_MODEL` | `claude-sonnet-4.6` | Default model for completions |
| `COPILOT_SDK_TRANSPORT` | `stdio` | Transport mode: `stdio` or `tcp` |
| `COPILOT_GITHUB_TOKEN` | stored OAuth | GitHub auth token (falls back to `GH_TOKEN`, `GITHUB_TOKEN`) |

## MCP Server (`embacle-mcp`)

A standalone MCP server binary that exposes embacle runners via the [Model Context Protocol](https://modelcontextprotocol.io/). Connect any MCP-compatible client (Claude Desktop, editors, custom agents) to use all embacle providers.

### Usage

```bash
# Stdio transport (default — for editor/client integration)
embacle-mcp --provider copilot

# HTTP transport (for network-accessible deployments)
embacle-mcp --transport http --host 0.0.0.0 --port 3000 --provider claude_code
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `get_provider` | Get active LLM provider and list available providers |
| `set_provider` | Switch the active provider (`claude_code`, `copilot`, `cursor_agent`, `opencode`, `gemini_cli`, `codex_cli`) |
| `get_model` | Get current model and list available models for the active provider |
| `set_model` | Set the model for subsequent requests (pass null to reset to default) |
| `get_multiplex_provider` | Get providers configured for multiplex dispatch |
| `set_multiplex_provider` | Configure providers for fan-out mode |
| `prompt` | Send chat messages to the active provider, or multiplex to all configured providers |

### Client Configuration

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "embacle": {
      "command": "embacle-mcp",
      "args": ["--provider", "copilot"]
    }
  }
}
```

## REST API Server (`embacle-server`)

An OpenAI-compatible HTTP server that proxies requests to embacle runners. Any client that speaks the OpenAI chat completions API can use it without modification.

### Usage

```bash
# Start with default provider (copilot) on localhost:3000
embacle-server

# Specify provider and port
embacle-server --provider claude_code --port 8080 --host 0.0.0.0
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `GET` | `/v1/models` | List available providers and models |
| `GET` | `/health` | Per-provider readiness check |

### Model Routing

The `model` field determines which provider handles the request. Use a `provider:model` prefix to target a specific runner, or pass a bare model name to use the server's default provider.

```bash
# Explicit provider
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude:opus", "messages": [{"role": "user", "content": "hello"}]}'

# Default provider
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}'
```

### Multiplex

Pass an array of models to fan out the same prompt to multiple providers concurrently. Each provider runs in its own task; failures in one don't affect others.

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": ["copilot:gpt-4o", "claude:opus"], "messages": [{"role": "user", "content": "hello"}]}'
```

The response uses `object: "chat.completion.multiplex"` with per-provider results and timing.

Streaming is not supported for multiplex requests.

### SSE Streaming

Set `"stream": true` for Server-Sent Events output in OpenAI streaming format (`data: {json}\n\n` with `data: [DONE]` terminator).

### Authentication

Optional. Set `EMBACLE_API_KEY` to require bearer token auth on all endpoints. When unset, all requests are allowed through (localhost development mode). The env var is read per-request, so key rotation doesn't require a restart.

```bash
EMBACLE_API_KEY=my-secret embacle-server
curl http://localhost:3000/v1/models -H "Authorization: Bearer my-secret"
```

## Architecture

```
Your Application
    └── embacle (this library)
            │
            ├── CLI Runners (subprocess per request)
            │   ├── ClaudeCodeRunner    → spawns `claude -p "prompt" --output-format json`
            │   ├── CopilotRunner       → spawns `copilot -p "prompt"`
            │   ├── CursorAgentRunner   → spawns `cursor-agent -p "prompt" --output-format json`
            │   ├── OpenCodeRunner      → spawns `opencode run "prompt" --format json`
            │   ├── GeminiCliRunner     → spawns `gemini -p "prompt" -o json -y`
            │   └── CodexCliRunner      → spawns `codex exec "prompt" --json --full-auto`
            │
            ├── SDK Runners (persistent connection, behind feature flag)
            │   └── CopilotSdkRunner    → JSON-RPC to `copilot --headless`
            │
            ├── MCP Server (separate binary crate)
            │   └── embacle-mcp         → JSON-RPC 2.0 over stdio or HTTP/SSE
            │
            ├── REST API Server (separate binary crate)
            │   └── embacle-server      → OpenAI-compatible HTTP, SSE streaming, multiplex
            │
            └── Tool Simulation (text-based tool calling for CLI runners)
                └── execute_with_text_tools()  → catalog injection, XML parsing, tool loop
```

All runners implement the same `LlmProvider` trait:
- **`complete()`** — single-shot completion
- **`complete_stream()`** — streaming completion
- **`health_check()`** — verify the runner is available and authenticated

The `CopilotSdkRunner` additionally provides:
- **`execute_with_tools()`** — native tool calling via the SDK's session and tool handler infrastructure

### Text-Based Tool Calling (CLI runners)

CLI runners don't have native tool calling, so Embacle provides a text-based simulation layer. It injects a tool catalog into the system prompt and parses `<tool_call>` XML blocks from the LLM response, looping until the model stops calling tools.

```rust
use embacle::tool_simulation::{
    FunctionDeclaration, FunctionCall, FunctionResponse,
    execute_with_text_tools,
};
use embacle::types::{ChatMessage, LlmProvider};
use std::sync::Arc;
use serde_json::json;

let declarations = vec![
    FunctionDeclaration {
        name: "get_weather".into(),
        description: "Get current weather for a city".into(),
        parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
    },
];

let handler = Arc::new(|name: &str, args: &serde_json::Value| -> FunctionResponse {
    FunctionResponse {
        name: name.to_owned(),
        response: json!({"temperature": 22, "conditions": "sunny"}),
    }
});

let mut messages = vec![ChatMessage::user("What's the weather in Paris?")];
let result = execute_with_text_tools(
    &runner,          // any LlmProvider (CopilotRunner, ClaudeCodeRunner, etc.)
    &mut messages,
    &declarations,
    handler,
    5,                // max iterations
).await?;

println!("{}", result.content);           // final response with tools stripped
println!("Tool calls: {}", result.tool_calls_count);
```

The pure functions are also available individually for custom loop implementations:

| Function | Purpose |
|----------|---------|
| `generate_tool_catalog()` | Converts declarations into a markdown catalog |
| `inject_tool_catalog()` | Appends catalog to the system message |
| `parse_tool_call_blocks()` | Parses `<tool_call>` XML blocks from response text |
| `strip_tool_call_blocks()` | Returns clean text with tool blocks removed |
| `format_tool_results_as_text()` | Formats results as `<tool_result>` XML blocks |

## Features

- **Zero API keys** — uses CLI tools' own auth (OAuth, API keys managed by the tool)
- **Auto-discovery** — finds installed CLI binaries via `which`
- **Auth readiness** — non-blocking checks, graceful degradation
- **Capability detection** — probes CLI version and supported features
- **Container isolation** — optional container-based execution for production
- **Subprocess safety** — timeout, output limits, environment sandboxing
- **Feature flags** — SDK integrations are opt-in to keep the default dependency footprint minimal

## Modules

| Module | Feature | Purpose |
|--------|---------|---------|
| `types` | default | Core types: `LlmProvider` trait, `ChatRequest`, `ChatResponse`, `RunnerError` |
| `config` | default | Runner types, execution modes, configuration |
| `discovery` | default | Auto-detect installed CLI binaries |
| `auth` | default | Readiness checking (is the CLI authenticated?) |
| `compat` | default | Version compatibility and capability detection |
| `process` | default | Subprocess spawning with timeout and output limits |
| `sandbox` | default | Environment variable whitelisting, working directory control |
| `container` | default | Container-based execution backend |
| `prompt` | default | Prompt building from chat messages |
| `tool_simulation` | default | Text-based tool calling for CLI runners (`<tool_call>` XML protocol) |
| `copilot_sdk_runner` | `copilot-sdk` | Copilot SDK runner (persistent JSON-RPC) |
| `copilot_sdk_config` | `copilot-sdk` | Copilot SDK configuration from environment |
| `tool_bridge` | `copilot-sdk` | Tool definition conversion for native tool calling |

## License

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>).
