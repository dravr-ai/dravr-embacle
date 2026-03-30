# Embacle — LLM Runners

[![crates.io](https://img.shields.io/crates/v/embacle.svg)](https://crates.io/crates/embacle)
[![docs.rs](https://docs.rs/embacle/badge.svg)](https://docs.rs/embacle)
[![CI](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml/badge.svg)](https://github.com/dravr-ai/dravr-embacle/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

Standalone Rust library that wraps 12 AI CLI tools and SDKs as pluggable LLM providers, with vision/image support.

Instead of integrating with LLM APIs directly (which require API keys, SDKs, and managing auth), **Embacle** delegates to CLI tools that users already have installed and authenticated — getting model upgrades, auth management, and protocol handling for free. For GitHub Copilot, an optional headless mode communicates via the ACP (Agent Client Protocol) for SDK-managed tool calling.

## Run It

Embacle ships as two ready-to-run servers — no code required:

**OpenAI-compatible HTTP server** — drop-in replacement for any OpenAI client:

```bash
embacle-server --provider copilot --port 3000
```
```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "copilot", "messages": [{"role": "user", "content": "hello"}]}'
```

**MCP server** — connect Claude Desktop, editors, or any MCP client directly:

```bash
# stdio (editor integration)
embacle-mcp --provider copilot

# HTTP (network-accessible)
embacle-mcp --transport http --port 3000 --provider copilot
```

Both modes support all 12 CLI runners, streaming, model routing, and vision. See [REST API Server](#rest-api-server-embacle-server) and [MCP Server](#mcp-server-embacle-mcp) for full details.

## Table of Contents

- [Install](#install)
- [Supported Runners](#supported-runners)
- [Quick Start](#quick-start)
- [REST API Server](#rest-api-server-embacle-server)
- [MCP Server](#mcp-server-embacle-mcp)
- [OpenAI API](#openai-api-feature-flag)
- [Copilot Headless](#copilot-headless-feature-flag)
- [Vision / Image Support](#vision--image-support)
- [Docker](#docker)
- [C FFI Static Library](#c-ffi-static-library)
- [Architecture](#architecture)
- [Tested With](#tested-with)
- [License](#license)

## Install

### Homebrew (macOS / Linux) — recommended

```bash
brew tap dravr-ai/tap
brew install embacle
```

This installs two binaries:

- **`embacle-server`** — OpenAI-compatible REST API + MCP server
- **`embacle-mcp`** — standalone MCP server for editor integration

### Docker

```bash
docker pull ghcr.io/dravr-ai/embacle:latest
docker run -p 3000:3000 ghcr.io/dravr-ai/embacle --provider copilot
```

### Cargo (library)

```toml
[dependencies]
embacle = "0.14"
```

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
| Kiro CLI | `kiro-cli` | ANSI-stripped text output, auto model selection |
| Kilo Code | `kilo` | NDJSON output, streaming, token tracking, 500+ models via Kilo Gateway |

### HTTP API Runners (feature-flagged)

| Runner | Feature Flag | Features |
|--------|-------------|----------|
| OpenAI API | `openai-api` | Any OpenAI-compatible endpoint (OpenAI, Groq, Gemini, Ollama, vLLM), streaming, tool calling, model discovery |

### ACP Runners (persistent connection)

| Runner | Feature Flag | Features |
|--------|-------------|----------|
| GitHub Copilot Headless | `copilot-headless` | NDJSON/JSON-RPC via `copilot --acp`, SDK-managed tool calling, streaming |

## Quick Start

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

## REST API Server (`embacle-server`)

A unified OpenAI-compatible HTTP server with built-in MCP support that proxies requests to embacle runners. Any client that speaks the OpenAI chat completions API or MCP protocol can use it without modification. Supports `--transport stdio` for MCP-only mode (editor integration).

### Usage

```bash
# Start with default provider (copilot) on localhost:3000
embacle-server

# Specify provider and port
embacle-server --provider claude_code --port 8080 --host 0.0.0.0

# MCP-only mode via stdio (for editor/client integration)
embacle-server --transport stdio --provider copilot
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `GET` | `/v1/models` | List available providers and models |
| `GET` | `/health` | Per-provider readiness check |
| `POST` | `/mcp` | MCP Streamable HTTP (JSON-RPC 2.0) |

### MCP Streamable HTTP

The server also speaks [MCP](https://modelcontextprotocol.io/) at `POST /mcp`, accepting JSON-RPC 2.0 requests. Any MCP-compatible client can connect over HTTP instead of stdio.

| Tool | Description |
|------|-------------|
| `prompt` | Send chat messages to an LLM provider, with optional `model` routing (e.g. `copilot:gpt-4o`) |
| `list_models` | List available providers and the server's default |

```bash
# MCP initialize handshake
curl http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl"}}}'

# Call the prompt tool
curl http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"prompt","arguments":{"messages":[{"role":"user","content":"hello"}]}}}'
```

Add `Accept: text/event-stream` to receive SSE-wrapped responses instead of plain JSON.

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
  -d '{"model": "gpt-5.4", "messages": [{"role": "user", "content": "hello"}]}'
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

## MCP Server (`embacle-mcp`)

A library and standalone binary that exposes embacle runners via the [Model Context Protocol](https://modelcontextprotocol.io/). Connect any MCP-compatible client (Claude Desktop, editors, custom agents) to use all embacle providers.

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
| `set_provider` | Switch the active provider (`claude_code`, `copilot`, `copilot_headless`, `cursor_agent`, `opencode`, `gemini_cli`, `codex_cli`, `goose_cli`, `cline_cli`, `continue_cli`, `warp_cli`, `kiro_cli`, `kilo_cli`) |
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

## OpenAI API (feature flag)

Enable the `openai-api` feature for HTTP-based communication with any OpenAI-compatible endpoint:

```toml
[dependencies]
embacle = { version = "0.14", features = ["openai-api"] }
```

```rust
use embacle::{OpenAiApiConfig, OpenAiApiRunner};
use embacle::types::{ChatMessage, ChatRequest, LlmProvider};

#[tokio::main]
async fn main() -> Result<(), embacle::types::RunnerError> {
    // Reads OPENAI_API_BASE_URL, OPENAI_API_KEY, OPENAI_API_MODEL from env
    let config = OpenAiApiConfig::from_env();
    let runner = OpenAiApiRunner::new(config).await;

    let request = ChatRequest::new(vec![
        ChatMessage::user("What is the capital of France?"),
    ]);

    let response = runner.complete(&request).await?;
    println!("{}", response.content);
    Ok(())
}
```

Works with any OpenAI-compatible endpoint — OpenAI, Groq, Google Gemini, Ollama, vLLM, and more. To inject a shared HTTP client (e.g. from a connection pool), use `OpenAiApiRunner::with_client(config, client)`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `OPENAI_API_KEY` | *(none)* | Bearer token for authentication |
| `OPENAI_API_MODEL` | `gpt-5.4` | Default model for completions |
| `OPENAI_API_TIMEOUT_SECS` | `300` | HTTP request timeout |

## Copilot Headless (feature flag)

Enable the `copilot-headless` feature for ACP-based communication with SDK-managed tool calling:

```toml
[dependencies]
embacle = { version = "0.14", features = ["copilot-headless"] }
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

The headless runner spawns `copilot --acp` per request and communicates via NDJSON-framed JSON-RPC. The system prompt is passed via ACP's `session/new` `systemPrompt` parameter. Conversation history from prior turns is serialized into a `<conversation-history>` block in the prompt text for multi-turn continuity. The `max_tokens` field from `ChatRequest` is forwarded to ACP's `session/prompt` as `maxTokens`.

Configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `COPILOT_CLI_PATH` | auto-detect | Override path to copilot binary |
| `COPILOT_HEADLESS_MODEL` | `claude-opus-4.6-fast` | Default model for completions |
| `COPILOT_GITHUB_TOKEN` | stored OAuth | GitHub auth token (falls back to `GH_TOKEN`, `GITHUB_TOKEN`) |
| `COPILOT_HEADLESS_MAX_HISTORY_TURNS` | `20` | Max conversation history turns in prompt (0 disables) |
| `COPILOT_HEADLESS_INJECT_SYSTEM_IN_PROMPT` | `false` | Re-inject system prompt as `<system-instructions>` in prompt text |

## Vision / Image Support

Embacle supports sending images alongside text prompts via the `ImagePart` type. Images are base64-encoded and tagged with a MIME type (PNG, JPEG, WebP, GIF).

### Which providers support vision?

| Provider | Vision | How |
|----------|--------|-----|
| Copilot Headless (ACP) | Native | Images sent as ACP `image` content blocks |
| OpenAI API | Native | Images sent as `image_url` parts with `data:` URIs |
| C FFI | Native | Images forwarded to copilot headless via `image_url` content |
| All 12 CLI runners | Tempfile | Images decoded to temp files, file paths injected into prompt |

CLI runners materialize base64 images to a temp directory and append `[Attached images]` with file paths to the user message. The temp directory is kept alive until the subprocess finishes.

### Library usage

```rust
use embacle::types::{ChatMessage, ChatRequest, ImagePart};

let image = ImagePart::new(base64_data, "image/png")?;
let request = ChatRequest::new(vec![
    ChatMessage::user_with_images("What do you see?", vec![image]),
]);
```

### Server usage (OpenAI multipart content)

Send images via the standard OpenAI multipart content format:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "copilot_headless",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
      ]
    }]
  }'
```

Plain string messages continue to work unchanged. All providers accept images — native providers send them directly, CLI runners materialize them to temp files.

## Docker

Pull the image from GitHub Container Registry:

```bash
docker pull ghcr.io/dravr-ai/embacle:latest
```

The image includes `embacle-server` and `embacle-mcp` with Node.js pre-installed for adding CLI backends.

### Adding a CLI Backend

The base image doesn't include CLI tools. Install them in a derived image:

```dockerfile
FROM ghcr.io/dravr-ai/embacle
USER root
RUN npm install -g @anthropic-ai/claude-code
USER embacle
```

Build and run:

```bash
docker build -t my-embacle .
docker run -p 3000:3000 my-embacle --provider claude_code
```

### Auth and Configuration

CLI tools store auth tokens in their config directories. Mount them from the host, or set provider-specific env vars:

```bash
# Mount Claude Code auth from host
docker run -p 3000:3000 \
  -v ~/.claude:/home/embacle/.claude:ro \
  my-embacle --provider claude_code

# Or pass env vars if the CLI supports them
docker run -p 3000:3000 \
  -e GITHUB_TOKEN=ghp_... \
  -e EMBACLE_API_KEY=my-secret \
  my-embacle --provider copilot
```

### Running embacle-mcp

Override the entrypoint to run the MCP server instead:

```bash
docker run --entrypoint embacle-mcp ghcr.io/dravr-ai/embacle --provider copilot
```

## C FFI Static Library

Embacle provides a C FFI static library (`libembacle.a`) that exposes copilot chat completion to any language that can call C functions — Swift, Objective-C, Python, Go, Ruby, and more. The FFI surface is 4 functions: init, chat completion, free string, and shutdown.

### Install via Homebrew

```bash
brew tap dravr-ai/tap
brew install embacle-ffi
```

This builds from source (requires Rust) and installs `libembacle.a` and `embacle.h` to Homebrew's prefix. The formula is published automatically with each release.

For CI environments:

```bash
brew tap dravr-ai/tap
brew install embacle-ffi
# libembacle.a and embacle.h are now available under $(brew --prefix)/lib and $(brew --prefix)/include
```

### Install via script

```bash
./scripts/install-ffi.sh                        # → /usr/local
./scripts/install-ffi.sh --prefix $HOME/.local  # → custom prefix
./scripts/install-ffi.sh --uninstall            # remove
```

### Build manually

```bash
cargo build --release --features ffi
# Output: target/release/libembacle.a
# Header: include/embacle.h
```

### Swift / SPM example

For Swift Package Manager, add a `systemLibrary` target in your `Package.swift` with a modulemap that links `embacle`:

```swift
.systemLibrary(name: "CEmbacle")
```

With a `module.modulemap`:
```
module CEmbacle {
    header "embacle.h"
    link "embacle"
    export *
}
```

The FFI accepts OpenAI-compatible JSON — the same format as the REST API:

```c
embacle_init();
char* response = embacle_chat_completion(
    "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
    60  /* timeout seconds */
);
/* use response JSON... */
embacle_free_string(response);
embacle_shutdown();
```

Vision payloads work via multipart content with `image_url` data URIs.

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
            │   ├── WarpCliRunner       → spawns `oz agent run --prompt "..." --output-format json`
            │   ├── KiroCliRunner       → spawns `kiro-cli send "prompt"`
            │   └── KiloCliRunner       → spawns `kilo run --auto --format json`
            │
            ├── HTTP API Runners (behind feature flag)
            │   └── OpenAiApiRunner       → reqwest to any OpenAI-compatible endpoint
            │
            ├── ACP Runners (persistent connection, behind feature flag)
            │   └── CopilotHeadlessRunner → NDJSON/JSON-RPC to `copilot --acp`
            │
            ├── Provider Decorators (composable wrappers)
            │   ├── FallbackProvider    → ordered chain with retry and exponential backoff
            │   ├── MetricsProvider     → latency, token, and cost tracking
            │   ├── QualityGateProvider → response validation with retry
            │   ├── GuardrailProvider   → pluggable pre/post request validation
            │   └── CacheProvider       → response caching with TTL and capacity
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
            ├── MCP Server (library + binary crate, powered by dravr-tronc)
            │   └── embacle-mcp         → JSON-RPC 2.0 over stdio or HTTP/SSE
            │
            ├── Unified REST API + MCP Server (binary crate, powered by dravr-tronc)
            │   └── embacle-server      → OpenAI-compatible HTTP, MCP Streamable HTTP, SSE streaming, multiplex
            │
            └── Tool Simulation (text-based tool calling for CLI runners)
                └── execute_with_text_tools()  → catalog injection, XML parsing, tool loop
```

All runners implement the same `LlmProvider` trait:
- **`complete()`** — single-shot completion
- **`complete_stream()`** — streaming completion
- **`health_check()`** — verify the runner is available and authenticated

For detailed API docs — fallback chains, structured output, agent loop, metrics, quality gates, tool simulation, and more — see [docs.rs/embacle](https://docs.rs/embacle).

## Tested With

Embacle has been tested with [mirroir.dev](https://github.com/jfarcand/mirroir-mcp), an MCP server for AI-powered iPhone automation.


## License

Licensed under the Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>).
