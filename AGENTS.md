## Git Workflow: NO Pull Requests

**CRITICAL: NEVER create Pull Requests. All merges happen locally via squash merge.**

### Rules
- **NEVER use `gh pr create`** or any PR creation command
- **NEVER suggest creating a PR**
- Feature branches are merged via **local squash merge**

### Workflow for Features
1. Create feature branch: `git checkout -b feature/my-feature`
2. Make commits, push to remote: `git push -u origin feature/my-feature`
3. When ready, squash merge locally (from main worktree):
   ```bash
   git checkout main
   git fetch origin
   git merge --squash origin/feature/my-feature
   git commit
   git push
   ```

### Bug Fixes
- Bug fixes go directly to `main` branch (no feature branch needed)
- Commit and push directly: `git push origin main`

## Project Overview

**embacle** is a Rust workspace providing pluggable LLM providers that delegate to AI CLI tools and the ACP protocol — plus an MCP server and an OpenAI-compatible REST API server.

### Workspace Crates

| Crate | Type | Purpose |
|-------|------|---------|
| `embacle` | library | Core: 12 CLI runners, HTTP API runner, ACP headless runner, agent loop, decorators, tool simulation |
| `embacle-mcp` | library + binary | MCP server (stdio/HTTP) exposing all runners via JSON-RPC 2.0 |
| `embacle-server` | binary | Unified OpenAI-compatible REST API + MCP server with SSE streaming, multiplex fan-out, bearer auth |

### CLI Runners (subprocess per request)
Claude Code, Copilot, Cursor Agent, OpenCode, Gemini, Codex, Goose, Cline, Continue, Warp, Kiro, Kilo Code

### ACP Runner (feature flag: `copilot-headless`)
`CopilotHeadlessRunner` — NDJSON/JSON-RPC via `copilot --acp` with SDK-managed tool calling

### Higher-Level Features
- **Agent loop** (`AgentExecutor`) — multi-turn tool calling with configurable max turns
- **Fallback chains** (`FallbackProvider`) — ordered failover with retry and exponential backoff
- **Metrics** (`MetricsProvider`) — latency, token, cost tracking, and OTel metrics export
- **Quality gate** (`QualityGateProvider`) — response validation with retry on refusal
- **Guardrails** (`GuardrailProvider`) — pluggable pre/post request validation middleware
- **Cache** (`CacheProvider`) — response caching with TTL and capacity limits
- **Structured output** — schema-validated JSON extraction from any provider
- **MCP tool bridge** — connect MCP tool servers to CLI runners via text-based tool loop
- **Text tool simulation** — XML-based `<tool_call>` protocol for CLI runners without native function calling
- **Config file** — TOML-based declarative configuration (feature flag: `config-file`)
- **Capability guard** — request/provider capability validation

### Architecture
```
src/
├── lib.rs                  # Re-exports all runners + shared types
├── types.rs                # LlmProvider trait, RunnerError, ChatRequest, ChatResponse, etc.
├── config.rs               # RunnerConfig, CliRunnerType enum
├── factory.rs              # Runner factory, provider parsing, ALL_PROVIDERS
├── cli_common.rs           # CliRunnerBase + macro for runner boilerplate
├── auth.rs                 # Auth checking + readiness state
├── discovery.rs            # Binary auto-detection via `which`
├── process.rs              # Subprocess spawning, timeout, output capture
├── sandbox.rs              # Env/cwd/tool policy enforcement
├── container.rs            # Optional container execution backend
├── prompt.rs               # Build prompts from ChatMessage history
├── compat.rs               # CLI compatibility detection
├── stream.rs               # GuardedStream for child process lifecycle
├── agent.rs                # Multi-turn agent loop with tool calling
├── fallback.rs             # Ordered provider failover with retry
├── metrics.rs              # Latency/token/cost tracking decorator
├── quality_gate.rs         # Response validation with retry
├── guardrail.rs            # Pluggable pre/post request validation
├── cache.rs                # Response caching with TTL and capacity
├── structured_output.rs    # Schema-enforced JSON extraction
├── tool_simulation.rs      # Text-based tool calling (XML protocol)
├── mcp_tool_bridge.rs      # MCP tool definitions ↔ text tool loop
├── capability_guard.rs     # Request/provider capability validation
├── config_file.rs          # TOML config loading (feature: config-file)
├── claude_code.rs          # ClaudeCodeRunner
├── copilot.rs              # CopilotRunner + model discovery
├── copilot_headless.rs     # CopilotHeadlessRunner (ACP/NDJSON)
├── copilot_headless_config.rs  # Headless config from env vars
├── cursor_agent.rs         # CursorAgentRunner
├── opencode.rs             # OpenCodeRunner (NDJSON)
├── openai_api.rs           # OpenAiApiRunner (feature: openai-api)
├── gemini_cli.rs           # GeminiCliRunner
├── codex_cli.rs            # CodexCliRunner (JSONL)
├── goose_cli.rs            # GooseCliRunner
├── cline_cli.rs            # ClineCliRunner (NDJSON)
├── continue_cli.rs         # ContinueCliRunner
├── warp_cli.rs             # WarpCliRunner (NDJSON)
├── kiro_cli.rs             # KiroCliRunner
└── kilo_cli.rs             # KiloCliRunner (NDJSON)

crates/
├── embacle-mcp/            # MCP server library + binary (stdio + HTTP/SSE)
└── embacle-server/         # Unified OpenAI-compatible REST API + MCP server

tests/
└── e2e.rs                  # E2E integration tests (env-gated per runner)
```

### Key Design Decisions
- **100% standalone** — zero dependency on dravr-platform or pierre-core
- **Types in `types.rs`** — `RunnerError`, `LlmProvider` trait, `ChatRequest`, `ChatResponse`, etc.
- **Subprocess-based** — wraps CLI tools via `tokio::process::Command`
- **ACP via feature flag** — `copilot-headless` adds NDJSON/JSON-RPC transport for `copilot --acp`
- **No HTTP dependencies in core** — only tokio (process), serde, tracing, which, bitflags
- **Workspace crates** — MCP server (library + binary) and unified REST API + MCP server (binary) are separate crates

## Git Hooks - MANDATORY for ALL AI Agents

**⚠️ MANDATORY - Run this at the START OF EVERY SESSION:**
```bash
git config core.hooksPath .githooks
```
This enables pre-commit, commit-msg, and pre-push hooks. Sessions get archived/revived, so this must run EVERY time you start working, not just once.

**NEVER use `--no-verify` when committing or pushing.** The hooks enforce:
- SPDX license headers on all source files
- Commit message format (max 2 lines, conventional commits)
- No AI-generated commit signatures (🤖, "Generated with", etc.)
- No unauthorized root markdown files

## Pre-Push Validation Workflow

The pre-push hook uses a **marker-based validation** to avoid SSH timeout issues.

### Workflow

1. **Make your changes and commit**
2. **Run validation before pushing:**
   ```bash
   ./scripts/pre-push-validate.sh
   ```
   On success, creates `.git/validation-passed` marker (valid for 15 minutes).

3. **Push:**
   ```bash
   git push
   ```

### Important Notes

- If validation expires or commit changes, re-run `./scripts/pre-push-validate.sh`
- To bypass (NOT RECOMMENDED): `git push --no-verify`

### NEVER

- Manually create `.git/validation-passed` marker
- Skip validation by creating a fake marker — CI will catch issues
- Claim "rustfmt isn't installed" or similar excuses to bypass validation

### CI Monitoring

Use the first available method. **NEVER ask the user for a GitHub token** — fall back instead.

| Priority | Method | When to use |
|----------|--------|-------------|
| 1 | `gh run list --branch main` / `gh run watch` | `gh` CLI is installed and authenticated |
| 2 | GitHub MCP tools (`mcp__github__*`) | `gh` unavailable but GitHub MCP server is configured |

# Writing code

- CRITICAL: NEVER USE --no-verify WHEN COMMITTING CODE
- We prefer simple, clean, maintainable solutions over clever or complex ones
- Make the smallest reasonable changes to get to the desired outcome
- When modifying code, match the style and formatting of surrounding code
- NEVER make code changes that aren't directly related to the task you're currently assigned
- NEVER remove code comments unless you can prove that they are actively false
- All code files should start with a brief 2 line comment explaining what the file does. Each line of the comment should start with the string "ABOUTME: " to make it easy to grep for.
- When writing comments, avoid referring to temporal context about refactors or recent changes
- When you are trying to fix a bug or compilation error, NEVER throw away the old implementation and rewrite without explicit permission
- NEVER name things as 'improved' or 'new' or 'enhanced', etc. Code naming should be evergreen.
- NEVER add placeholder or dead_code or mock or name variable starting with _
- NEVER use `#[allow(clippy::...)]` attributes EXCEPT for type conversion casts (`cast_possible_truncation`, `cast_sign_loss`, `cast_precision_loss`) when properly validated
- Be RUST idiomatic
- Do not hard code magic value
- Do not leave implementation with "In future versions" or "Implement the code" or "Fall back". Always implement the real thing.
- Commit without AI assistant-related commit messages. Do not reference AI assistance in git commits.
- Always create a branch when adding new features. Bug fixes go directly to main branch.
- Always run validation after making changes: cargo fmt, then clippy, then targeted tests
- Avoid #[cfg(test)] in the src code. Only in tests

## Security Engineering Rules

### Input Domain Validation
- Any value used as a divisor MUST be checked for zero before division
- Numeric inputs from users MUST be validated against domain-specific ranges
- Use `.max(1)` or equivalent guard before any division operation

### Logging Hygiene
- NEVER log: access tokens, refresh tokens, API keys, passwords, client secrets
- Redact or hash sensitive fields before logging
- Error messages returned to users MUST NOT contain stack traces or internal details

## Required Pre-Commit Validation

### Tiered Validation Approach

#### Tier 1: Quick Iteration (during development)
```bash
# 1. Format code
cargo fmt

# 2. Compile check only
cargo check --quiet

# 3. Run targeted tests
cargo test <test_name_pattern> -- --nocapture
```

#### Tier 2: Pre-Commit (before committing)
```bash
# 1. Format code
cargo fmt

# 2. Clippy with CI-matching strictness (warnings = errors)
RUSTFLAGS=-Dwarnings cargo clippy --all-targets -- -D warnings

# 3. Run targeted tests
cargo test <test_pattern> -- --nocapture
```

#### Tier 3: Full Validation (before merge)
```bash
cargo fmt
RUSTFLAGS=-Dwarnings cargo clippy --all-targets -- -D warnings
cargo test
```

### Test Output Verification - MANDATORY

**After running ANY test command, you MUST verify tests actually ran.**

**Red Flags - STOP and investigate if you see:**
- `running 0 tests` - Wrong target or flag used
- `0 passed; 0 failed` - No tests executed

**Never claim "tests pass" if 0 tests ran - that is a failure, not a success.**

## Error Handling Requirements

### Acceptable Error Handling
- `?` operator for error propagation
- `Result<T, E>` for all fallible operations
- `Option<T>` for values that may not exist
- Custom error types implementing `std::error::Error`

### Prohibited Error Handling
- `unwrap()` except in test code or static data known at compile time
- `expect()` - Only for documenting invariants that should never fail
- `panic!()` - Only in test assertions
- **`anyhow!()` macro** - ABSOLUTELY FORBIDDEN in all production code

### Structured Error Type Requirements
All errors MUST use `RunnerError` with appropriate `ErrorKind`:
```rust
// GOOD
return Err(RunnerError::internal("description"));
return Err(RunnerError::external_service("service", "description"));
return Err(RunnerError::binary_not_found("claude"));

// FORBIDDEN
return Err(anyhow!("something failed"));
```

## Mock Policy

### Real Implementation Preference
- PREFER real implementations over mocks in all production code
- NEVER implement mock modes for production features

### Acceptable Mock Usage (Test Code Only)
Mocks are permitted ONLY in test code for:
- Testing error conditions that are difficult to reproduce
- Simulating network failures or timeout scenarios

## Documentation Standards

### Code Documentation
- All public APIs MUST have comprehensive doc comments
- Use `/// ` for public API documentation
- Document error conditions and panic scenarios
- Include usage examples for complex APIs

## Task Completion Protocol - MANDATORY

### Before Claiming ANY Task Complete:

1. **Run Validation:**
   ```bash
   cargo fmt
   cargo clippy --all-targets
   cargo test <relevant_tests> -- --nocapture
   ```

2. **Verify tests ran** (N > 0 passed)

3. **Commit and push**
