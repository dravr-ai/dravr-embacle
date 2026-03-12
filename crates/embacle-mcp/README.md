# embacle-mcp

MCP server library and binary that exposes [embacle](https://github.com/dravr-ai/dravr-embacle) LLM runners via the [Model Context Protocol](https://modelcontextprotocol.io/). Any MCP-compatible client (Claude Desktop, editors, custom agents) can connect to use all embacle providers.

## Install

```bash
cargo install embacle-mcp
```

## Usage

```bash
# Stdio transport (default — for editor/client integration)
embacle-mcp --provider copilot

# HTTP transport (for network-accessible deployments)
embacle-mcp --transport http --host 0.0.0.0 --port 3000 --provider claude_code
```

## Tools

| Tool | Description |
|------|-------------|
| `get_provider` | Get active LLM provider and list available providers |
| `set_provider` | Switch the active provider (`claude_code`, `copilot`, `cursor_agent`, `opencode`, `gemini_cli`, `codex_cli`, `goose_cli`, `cline_cli`, `continue_cli`, `warp_cli`, `kiro_cli`, `kilo_cli`) |
| `get_model` | Get current model and list available models for the active provider |
| `set_model` | Set the model for subsequent requests (pass null to reset to default) |
| `get_multiplex_provider` | Get providers configured for multiplex dispatch |
| `set_multiplex_provider` | Configure providers for fan-out mode |
| `prompt` | Send chat messages to the active provider, or multiplex to all configured providers |

## Client Configuration

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

## Requirements

At least one supported CLI tool must be installed and authenticated:
- `claude` (Claude Code)
- `copilot` (GitHub Copilot)
- `cursor-agent` (Cursor Agent)
- `opencode` (OpenCode)
- `gemini` (Gemini CLI)
- `codex` (Codex CLI)
- `goose` (Goose CLI)
- `cline` (Cline CLI)
- `cn` (Continue CLI)
- `oz` (Warp)
- `kiro-cli` (Kiro CLI)
- `kilo` (Kilo Code)

## License

Apache-2.0 — see [LICENSE-APACHE](../../LICENSE-APACHE).

Full project documentation: [github.com/dravr-ai/dravr-embacle](https://github.com/dravr-ai/dravr-embacle)
