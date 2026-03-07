# Changelog

## [0.7.0] — 2026-03-07

### Added

- feat: add InstalledAuthUnverified variant and env var auth checking
- feat: enhance JSON schema validation with recursion, enum, and bounds
- feat: add native tool calling types to core library

### Fixed

- fix: remove [Unreleased] section before v0.7.0 release
- fix: use npm prefix -g to locate copilot binary on CI
- fix: use npm bin -g to find copilot binary path on CI
- fix: symlink copilot-linux-x64 binary to copilot on CI PATH
- fix: use env var for secrets conditional in CI workflow
- fix: wire COPILOT_GITHUB_TOKEN secret into CI for headless e2e tests
- fix: CI now tests all workspace crates, fix workspace-wide clippy warnings
- fix: ACP permission response prefers AllowAlways, never falls back to reject
- fix: update AGENTS.md project overview to reflect full workspace scope
- fix: default to claude-opus-4.6-fast, update Copilot fallback model list
- fix: rewrite OpenCode NDJSON parser, add E2E test suite for all runners



## [0.6.0] — 2026-03-07

### Fixed

- fix: purge stale copilot-sdk references from docs, CI, and release workflow
- fix: complete headless tool calling + usage extraction Add SDK_TOOL_CALLING, ObservedToolCall tracking, converse() API, ACP usage parsing



## [0.5.0] — 2026-03-05

### Fixed

- fix: streaming downgrade for non-streaming providers, reduce lock scope Permissive mode now falls back to complete() + SSE instead of erroring

### Other

- refactor: deduplicate runners, remove as_any, async model discovery CliRunnerBase + macro (-528 lines), #[must_use] on RunnerError



## [0.4.1] — 2026-03-04

### Added

- feat: fix tool calling for coding-assistant LLMs, add index to ToolCall Code-generation framing for catalogs, ToolCall.index per OpenAI spec, live Copilot tests



## [0.3.0] — 2026-03-03

### Added

- feat: add agent loop, fallback chains, metrics, quality gate, MCP bridge, and structured output
- feat: add Goose, Cline, and Continue CLI runners Expand embacle from 6 to 9 CLI runners with JSON parsing, session resume, and streaming

### Fixed

- fix: restore trailing newlines stripped by LinesStream in SSE deltas



## [0.2.1] — 2026-03-02

### Added

- feat: add Gemini CLI and Codex CLI runners Add GeminiCliRunner and CodexCliRunner with streaming JSONL support, expanding embacle to 6 CLI tools

### Fixed

- fix: add GeminiCli and CodexCli to subcrate runner factories and provider lists



## [0.2.0] — 2026-03-02

### Added

- feat: add Gemini CLI and Codex CLI runners Add GeminiCliRunner and CodexCliRunner with streaming JSONL support, expanding embacle to 6 CLI tools



## [0.1.3] — 2026-03-01



## [0.1.2] — 2026-03-01



## [0.1.1] — 2026-03-01



## [0.1.0] — 2026-03-01

### Added

- feat: add embacle-server crate with OpenAI-compatible REST API Axum HTTP server with /v1/chat/completions, /v1/models, /health, SSE streaming, multiplex fan-out, bearer auth
- feat: add embacle-mcp binary crate with MCP server Stdio/HTTP transports, 7 tools, JSON-RPC 2.0, multiplex fan-out, README MCP section
- feat: add Timeout error kind, enhanced logging, and stdout capture on failure
- feat: add tool_simulation module for text-based tool calling Text-based tool loop for CLI runners with catalog generation, XML parsing, and async execution
- feat: address analysis weak spots, expand tests, add CI branch triggers Fix doc drift, propagate max_tokens, add gh-auth check, 44 new tests (18→62)
- feat: make available_models() configurable at runtime Change LlmProvider::available_models() return type from &'static [&'static str] to &[String] so models can be determined at runtime. Each runner stores its model list in a Vec<String>. CopilotRunner and CopilotSdkRunner discover models via `gh copilot models` at construction time, falling back to static defaults if the command fails.
- feat: default to claude-opus-4.6 for Copilot SDK, Copilot CLI, and Claude Code runners
- feat: add SDK_TOOL_CALLING capability and as_any() for downcasting Add flag for SDK-managed tool loops, as_any() on LlmProvider trait, re-export ToolHandler types
- feat: add Copilot SDK provider behind copilot-sdk feature flag

### Fixed

- fix: release workflow — use macos-14, build --workspace, reset versions to 0.0.1
- fix: rename remaining capitalized Embache references to Embacle
- fix: rename crate from embache to embacle
- fix: wire env keys to sandbox, guard streaming child lifecycle, remove dead ExecutionMode
- fix: use RunnerConfig::new in doc examples
- fix: correct rust-toolchain.toml format



## [0.1.0] — 2026-03-01

### Added

- feat: add embacle-server crate with OpenAI-compatible REST API Axum HTTP server with /v1/chat/completions, /v1/models, /health, SSE streaming, multiplex fan-out, bearer auth
- feat: add embacle-mcp binary crate with MCP server Stdio/HTTP transports, 7 tools, JSON-RPC 2.0, multiplex fan-out, README MCP section
- feat: add Timeout error kind, enhanced logging, and stdout capture on failure
- feat: add tool_simulation module for text-based tool calling Text-based tool loop for CLI runners with catalog generation, XML parsing, and async execution
- feat: address analysis weak spots, expand tests, add CI branch triggers Fix doc drift, propagate max_tokens, add gh-auth check, 44 new tests (18→62)
- feat: make available_models() configurable at runtime Change LlmProvider::available_models() return type from &'static [&'static str] to &[String] so models can be determined at runtime. Each runner stores its model list in a Vec<String>. CopilotRunner and CopilotSdkRunner discover models via `gh copilot models` at construction time, falling back to static defaults if the command fails.
- feat: default to claude-opus-4.6 for Copilot SDK, Copilot CLI, and Claude Code runners
- feat: add SDK_TOOL_CALLING capability and as_any() for downcasting Add flag for SDK-managed tool loops, as_any() on LlmProvider trait, re-export ToolHandler types
- feat: add Copilot SDK provider behind copilot-sdk feature flag

### Fixed

- fix: rename remaining capitalized Embache references to Embacle
- fix: rename crate from embache to embacle
- fix: wire env keys to sandbox, guard streaming child lifecycle, remove dead ExecutionMode
- fix: use RunnerConfig::new in doc examples
- fix: correct rust-toolchain.toml format


