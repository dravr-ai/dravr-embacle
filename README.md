# Embacle — LLM Runners

[![crates.io](https://img.shields.io/crates/v/embacle.svg)](https://crates.io/crates/embacle)
[![docs.rs](https://docs.rs/embacle/badge.svg)](https://docs.rs/embacle)
[![CI](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml/badge.svg)](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

Standalone Rust library that wraps AI CLI tools and SDKs as pluggable LLM providers.

Instead of integrating with LLM APIs directly (which require API keys, SDKs, and managing auth), **Embacle** delegates to CLI tools that users already have installed and authenticated — getting model upgrades, auth management, and protocol handling for free. For GitHub Copilot, an optional headless mode communicates via the ACP (Agent Client Protocol) for SDK-managed tool calling.

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
| Goose CLI | `goose` | JSON/stream-JSON output, streaming, no-session mode |
| Cline CLI | `cline` | NDJSON output, streaming, session resume via task IDs |
| Continue CLI | `cn` | JSON output, single-shot completions |
| Warp | `oz` | NDJSON output, conversation resume |

### ACP Runners (persistent connection)

| Runner | Feature Flag | Features |
|--------|-------------|----------|
| GitHub Copilot Headless | `copilot-headless` | NDJSON/JSON-RPC via `copilot --acp`, SDK-managed tool calling, streaming |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
embacle = "0.8"
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

### Copilot Headless (feature flag)

Enable the `copilot-headless` feature for ACP-based communication with SDK-managed tool calling:

```toml
[dependencies]
embacle = { version = "0.8", features = ["copilot-headless"] }
```

```rust
use embacle::{CopilotHeadlessRunner, CopilotHeadlessConfig};
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};

#[tokio::main]
async fn main() -> Result<(), embacle::types::RunnerError> {
    // Reads COPILOT_HEADLESS_MODEL, COPILOT_GITHUB_TOKEN, etc. from env
    let runner = CopilotHeadlessRunner::from_env().await;

    let request = ChatRequest::new(vec![
        ChatMessage::user("Explain Rust ownership"),
    ]);

    let response = runner.complete(&request).await?;
    println!("{}", response.content);
    Ok(())
}
```

The headless runner spawns `copilot --acp` per request and communicates via NDJSON-framed JSON-RPC. Configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COPILOT_CLI_PATH` | auto-detect | Override path to copilot binary |
| `COPILOT_HEADLESS_MODEL` | `claude-opus-4.6-fast` | Default model for completions |
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
| `set_provider` | Switch the active provider (`claude_code`, `copilot`, `cursor_agent`, `opencode`, `gemini_cli`, `codex_cli`, `goose_cli`, `cline_cli`, `continue_cli`, `warp_cli`) |
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
            │   ├── CodexCliRunner      → spawns `codex exec "prompt" --json --full-auto`
            │   ├── GooseCliRunner      → spawns `goose run --quiet --no-session`
            │   ├── ClineCliRunner      → spawns `cline task --json --act --yolo`
            │   ├── ContinueCliRunner   → spawns `cn -p --format json`
            │   └── WarpCliRunner       → spawns `oz agent run --prompt "..." --output-format json`
            │
            ├── ACP Runners (persistent connection, behind feature flag)
            │   └── CopilotHeadlessRunner → NDJSON/JSON-RPC to `copilot --acp`
            │
            ├── Provider Decorators (composable wrappers)
            │   ├── FallbackProvider    → ordered chain, first success wins
            │   ├── MetricsProvider     → latency, token, and error tracking
            │   └── QualityGateProvider → response validation with retry
            │
            ├── Agent Loop
            │   └── AgentExecutor       → multi-turn tool calling with configurable max turns
            │
            ├── Structured Output
            │   └── request_structured_output()  → schema-validated JSON extraction with retry
            │
            ├── MCP Tool Bridge
            │   └── McpToolBridge       → MCP tool definitions ↔ text-based tool loop
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

The `CopilotHeadlessRunner` (requires `copilot-headless` feature) additionally provides:
- **`converse()`** — returns `HeadlessToolResponse` with observed tool calls that copilot executed internally

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

### Native Tool Calling Types

The core library provides typed tool calling that flows through `ChatRequest` and `ChatResponse`, so callers can use native tool definitions without relying on the text-based XML simulation.

```rust
use embacle::types::{ChatMessage, ChatRequest};
use embacle::{ToolDefinition, ToolChoice};
use serde_json::json;

let tools = vec![ToolDefinition {
    name: "get_weather".into(),
    description: "Get current weather for a city".into(),
    parameters: Some(json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    })),
}];

let request = ChatRequest::new(vec![
    ChatMessage::user("What's the weather in Paris?"),
])
.with_tools(tools)
.with_tool_choice(ToolChoice::Auto);

// Providers that support function calling will use native tool calls;
// the server falls back to XML text simulation for CLI runners
```

Additional request fields: `top_p`, `stop` sequences, and `response_format` (text, JSON object, or JSON schema). The `capability_guard` module validates these against provider capabilities before dispatch.

### Agent Loop

`AgentExecutor` runs a multi-turn tool-calling loop: inject a tool catalog, send the prompt, parse `<tool_call>` blocks from the response, execute tools, feed results back, repeat until the model stops calling tools or `max_turns` is reached.

```rust
use embacle::agent::{AgentExecutor, AgentConfig};
use embacle::tool_simulation::FunctionDeclaration;
use embacle::types::{ChatMessage, ChatRequest};
use std::sync::Arc;
use serde_json::json;

let declarations = vec![FunctionDeclaration {
    name: "lookup".into(),
    description: "Look up a value".into(),
    parameters: Some(json!({"type": "object", "properties": {"key": {"type": "string"}}})),
}];

let handler = Arc::new(|name: &str, _args: &serde_json::Value| {
    embacle::tool_simulation::FunctionResponse {
        name: name.to_owned(),
        response: json!({"value": 42}),
    }
});

let agent = AgentExecutor::new(Arc::new(runner), declarations, handler)
    .with_max_turns(5);

let request = ChatRequest::new(vec![ChatMessage::user("Look up answer")]);
let result = agent.execute(&request).await?;
println!("Turns: {}, Tool calls: {}", result.total_turns, result.tool_calls);
```

### Fallback Chains

`FallbackProvider` wraps multiple providers and tries them in order. The first successful response wins; if all fail, the last error is returned.

```rust
use embacle::fallback::FallbackProvider;

let provider = FallbackProvider::new(vec![
    Box::new(primary_runner),
    Box::new(backup_runner),
])?;

// Uses primary_runner; falls back to backup_runner on error
let response = provider.complete(&request).await?;
```

Health checks pass if **any** provider is healthy. Capabilities are the union of all inner providers.

### Metrics

`MetricsProvider` wraps any provider to track latency, token usage, call counts, and errors.

```rust
use embacle::metrics::MetricsProvider;

let provider = MetricsProvider::new(Box::new(runner));
let response = provider.complete(&request).await?;

let report = provider.report();
println!("Calls: {}, Avg latency: {}ms", report.call_count, report.avg_latency_ms);
```

### Quality Gate

`QualityGateProvider` validates responses against a policy (minimum length, refusal detection) and retries with feedback if validation fails.

```rust
use embacle::quality_gate::{QualityGateProvider, QualityPolicy};

let policy = QualityPolicy {
    max_retries: 2,
    min_content_length: 10,
    ..QualityPolicy::default()
};
let provider = QualityGateProvider::new(Box::new(runner), policy);
let response = provider.complete(&request).await?;
```

### Structured Output

Forces any provider to return schema-valid JSON by injecting schema instructions and validating the response, retrying with error feedback on schema violations. Validation covers nested objects, array items, enum values, numeric bounds (`minimum`/`maximum`), and `additionalProperties: false`.

```rust
use embacle::structured_output::{request_structured_output, StructuredOutputRequest};
use serde_json::json;

let schema = json!({
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature": {"type": "number"}
    },
    "required": ["city", "temperature"]
});

let result = request_structured_output(
    &runner,
    &StructuredOutputRequest { request, schema, max_retries: 2 },
).await?;

let data: serde_json::Value = serde_json::from_str(&result.content)?;
```

### MCP Tool Bridge

Bridges MCP tool definitions to embacle's text-based tool loop, so CLI runners can use tools from any MCP-compatible tool server.

```rust
use embacle::mcp_tool_bridge::{McpToolBridge, McpToolDefinition};

let mcp_tools = vec![McpToolDefinition {
    name: "search".into(),
    description: Some("Search the web".into()),
    input_schema: serde_json::json!({"type": "object", "properties": {"query": {"type": "string"}}}),
}];

let declarations = McpToolBridge::to_declarations(&mcp_tools);
```

## Features

- **Zero API keys** — uses CLI tools' own auth (OAuth, API keys managed by the tool)
- **Auto-discovery** — finds installed CLI binaries via `which`
- **Auth readiness** — non-blocking checks with env var probes and graceful degradation
- **Capability detection** — probes CLI version and supported features
- **Capability guard** — validates request fields against provider capabilities before dispatch
- **Native tool calling types** — `ToolDefinition`, `ToolCallRequest`, `ToolChoice`, `ResponseFormat` in core
- **Container isolation** — optional container-based execution for production
- **Subprocess safety** — timeout, output limits, environment sandboxing
- **Agent loop** — multi-turn tool calling with configurable max turns and turn callbacks
- **Fallback chains** — ordered provider failover with automatic retry
- **Metrics** — latency, token, and error tracking as a provider decorator
- **Quality gate** — response validation with retry on refusal or insufficient content
- **Structured output** — schema-validated JSON extraction with recursive validation (nested objects, arrays, enums, numeric bounds)
- **MCP tool bridge** — connect MCP tool servers to CLI runners via text-based tool loop
- **Feature flags** — SDK integrations are opt-in to keep the default dependency footprint minimal

## Modules

| Module | Feature | Purpose |
|--------|---------|---------|
| `types` | default | Core types: `LlmProvider` trait, `ChatRequest`, `ChatResponse`, `RunnerError`, `ToolDefinition`, `ToolCallRequest`, `ToolChoice`, `ResponseFormat` |
| `config` | default | Runner types, execution modes, configuration |
| `discovery` | default | Auto-detect installed CLI binaries |
| `auth` | default | Readiness checking with env var probes (`ProviderReadiness`, `check_env_var_auth`) |
| `capability_guard` | default | Validates request fields against provider capabilities (tools, `top_p`, stop, response format) |
| `compat` | default | Version compatibility and capability detection |
| `process` | default | Subprocess spawning with timeout and output limits |
| `sandbox` | default | Environment variable whitelisting, working directory control |
| `container` | default | Container-based execution backend |
| `prompt` | default | Prompt building from chat messages |
| `tool_simulation` | default | Text-based tool calling for CLI runners (`<tool_call>` XML protocol) |
| `agent` | default | Multi-turn agent loop with tool calling and turn callbacks |
| `fallback` | default | Ordered provider chain with first-success-wins failover |
| `metrics` | default | Latency, token usage, and error tracking decorator |
| `quality_gate` | default | Response validation (refusal detection, length checks) with retry |
| `structured_output` | default | Schema-validated JSON extraction with recursive validation and retry |
| `mcp_tool_bridge` | default | MCP tool definitions ↔ text-based tool loop bridge |
| `copilot_headless` | `copilot-headless` | Copilot ACP runner (NDJSON/JSON-RPC via `copilot --acp`) |
| `copilot_headless_config` | `copilot-headless` | Copilot Headless configuration from environment (`PermissionPolicy`) |

## License

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>).
