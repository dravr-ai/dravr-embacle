# Changelog

## [0.15.1] — 2026-04-20



## [0.15.0] — 2026-04-17

### Added

- feat: ranked Copilot model catalog with self-heal + reasoning effort Replaces hardcoded default; adds ModelUnavailable kind and 1.95 clippy fixes.



## [0.14.11] — 2026-04-10

### Other

- build: reduce tokio feature footprint to minimal set



## [0.14.10] — 2026-04-03

### Fixed

- fix: use backticks for code identifiers in doc comment



## [0.14.9] — 2026-03-31



## [0.14.8] — 2026-03-31



## [0.14.7] — 2026-03-31

### Fixed

- fix: repair broken expect annotations in openai_types test code
- fix: add Safe annotation to test expect() calls in openai_types
- fix: resolve error handling violations found by dravr-build-config validation



## [0.14.6] — 2026-03-30



## [0.14.5] — 2026-03-30



## [0.14.4] — 2026-03-30

### Fixed

- fix: stop re-injecting system prompt into ACP prompt text, forward max_tokens

### Other

- deps: bump opentelemetry and opentelemetry_sdk from 0.28 to 0.31
- deps: bump toml from 0.8.23 to 1.1.0
- deps: bump agent-client-protocol-schema from 0.11.3 to 0.11.4
- deps: bump docker/build-push-action from 6 to 7
- deps: bump docker/login-action from 3 to 4
- deps: bump docker/setup-buildx-action from 3 to 4



## [0.14.3] — 2026-03-28



## [0.14.2] — 2026-03-26

### Other

- deps: bump dravr-tronc to 0.2 with error notification support



## [0.14.1] — 2026-03-23



## [0.13.5] — 2026-03-18



## [0.13.4] — 2026-03-18

### Added

- feat: make ACP session and prompt timeouts configurable via env vars



## [0.13.3] — 2026-03-18



## [0.13.2] — 2026-03-18



## [0.13.1] — 2026-03-18

### Fixed

- fix: inject system prompt into ACP prompt blocks



## [0.13.0] — 2026-03-13

### Added

- feat: add C FFI static library for Swift integration
- feat: materialize images to temp files for all 12 CLI runners



## [0.12.0] — 2026-03-13

### Added

- feat: add vision/image support and wire CopilotHeadless into provider system

### Fixed

- fix: resolve clippy and test count for copilot-headless wiring



## [0.11.0] — 2026-03-12

### Added

- feat: add Kilo Code CLI runner with NDJSON parsing and streaming
- feat: unify embacle-server with embacle-mcp into single binary
- feat: add guardrail middleware and cross-decorator scenario tests GuardrailProvider with 3 built-in guards, ScriptedProvider, 7 scenarios
- feat: add cost tracking, response cache, and OTel metrics export TokenPricing/PricingTable, CacheProvider, otel feature flag with instruments
- feat: add TOML config file loading behind config-file feature flag Declarative provider, fallback, alias setup via embacle.toml or ~/.config/embacle/
- feat: add retry with exponential backoff to FallbackProvider Add ErrorKind::is_transient(), RetryConfig, FallbackProvider::with_retry()
- feat: add Kiro CLI runner as 11th LLM provider

### Fixed

- fix: add kilo to --provider help text in server and mcp binaries
- fix: add permissions block to homebrew workflow
- fix: resolve clippy doc_markdown warnings for provider help strings
- fix: update health endpoint test to expect 11 providers



## [0.10.2] — 2026-03-11

### Added

- feat: add Kiro CLI runner as 11th LLM provider
- feat: add retry with exponential backoff to FallbackProvider
- feat: add TOML config file loading behind config-file feature flag
- feat: add cost tracking, response cache, and OTel metrics export
- feat: add guardrail middleware and cross-decorator scenario tests
- feat: unify embacle-server with embacle-mcp into single binary
- feat: add Kilo Code CLI runner with NDJSON parsing and streaming

### Fixed

- fix: resolve clippy doc_markdown warnings for provider help strings
- fix: add permissions block to homebrew workflow (code scanning alert #10)
- fix: filter artifact download to exclude Docker buildx cache

## [0.10.1] — 2026-03-11

### Fixed

- fix: Docker build uses -p flag for workspace binary targets

## [0.10.0] — 2026-03-11

### Added

- feat: add MCP Streamable HTTP to embacle-server, Homebrew tap with CI
- docs: add OpenAI API runner section to README



## [0.9.0] — 2026-03-11

### Added

- feat: add with_client() constructor for HTTP client injection
- feat: add OpenAI-compatible HTTP API client runner (openai-api feature) SSE streaming, tool calling, model discovery, 30 unit tests
- feat: add Docker image with ghcr.io release workflow

### Fixed

- fix: resolve 3 CodeQL uncontrolled-allocation-size alerts in stop field Use bounded to_bounded_vec() instead of clone+truncate to avoid copying unbounded user input
- fix: revert accidental version bump in Cargo.toml files
- fix: update OpenAI API default model to gpt-5.4
- fix: resolve 9 CodeQL security alerts (stop bounds, CI permissions)



## [0.8.1] — 2026-03-09



## [0.8.0] — 2026-03-09

### Added

- feat: add Warp terminal (oz) CLI runner as 10th provider

### Fixed

- fix: ACP prompt timeout, stdio error propagation, redact log output
- fix: timing-safe auth, dependabot config, remove Box::leak

### Other

- refactor: dedup factory/runner code, wire top_p/stop/response_format Move factory+parsing to core, add top_p/stop to OpenAI types, fix READMEs (0.6->0.7, 9 providers)
- Update Dependabot configuration version



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


