## Git Workflow: NO Pull Requests

**CRITICAL: NEVER create Pull Requests. All merges happen locally via squash merge.**

- NEVER use `gh pr create` or any PR creation command
- NEVER suggest creating a PR
- Feature branches are merged via local squash merge

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

## Mandatory Session Startup Checklist

Before touching any code in a new session, run in this order:

```bash
# 1. Pull shared build config (provides .build/hooks, .build/validation, etc.)
git submodule update --init --recursive

# 2. Set canonical git hooks path — ALWAYS .build/hooks, NEVER .githooks
git config core.hooksPath .build/hooks

# 3. Scan recent history for context
git log --oneline -10

# 4. Check CI health on main
gh run list --branch main --limit 10 --json workflowName,conclusion

# 5. See uncommitted work
git status
```

**If any workflow on main has been red for 2+ runs, STOP and surface it to the user** before starting the requested task. Ask: "Should I investigate CI before doing X?"

The canonical hooks/validation live in the `.build/` git submodule from
https://github.com/dravr-ai/dravr-build-config — never use a local `.githooks/`.

## Mandatory Pre-Push Validation

**Before EVERY push, run:**

```bash
# 1. Format
cargo fmt --all

# 2. Clippy with warnings as errors
cargo clippy --workspace --all-targets -- -D warnings

# 3. Architectural validation (MUST exit 0)
.build/validation/validate.sh
```

**DO NOT push if `.build/validation/validate.sh` fails.** Fix all reported issues first.

The validation checks: placeholder code, forbidden anyhow usage, problematic unwraps/expects/panics, underscore-prefixed names, unauthorized clippy allows, dead code annotations, test integrity, and more.

**NEVER use `--no-verify` when committing or pushing.** Hooks enforce:
- SPDX license headers on all source files
- Commit message format (max 2 lines, conventional commits)
- No AI-generated commit signatures (🤖, "Generated with", etc.)
- No unauthorized root markdown files

### Tiered Validation (During Development)

| Tier | When | Commands |
|------|------|----------|
| Quick | During dev iteration | `cargo check --quiet && cargo test <pattern>` |
| Pre-commit | Before each commit | `cargo fmt --all && RUSTFLAGS=-Dwarnings cargo clippy --all-targets -- -D warnings` |
| Full | Before push (see above) | `cargo fmt + clippy + .build/validation/validate.sh` |

### Test Output Verification — MANDATORY

After running ANY test command, you MUST verify tests actually ran:
- `running 0 tests` = wrong target or flag used → STOP and investigate
- `0 passed; 0 failed` = no tests executed → STOP and investigate

Never claim "tests pass" if 0 tests ran — that is a failure, not a success.

### CI Monitoring

Use the first available method. **NEVER ask the user for a GitHub token** — fall back instead.

| Priority | Method | When to use |
|----------|--------|-------------|
| 1 | `gh run list --branch main` / `gh run watch` | `gh` CLI installed and authenticated |
| 2 | GitHub MCP tools (`mcp__github__*`) | `gh` unavailable but GitHub MCP server configured |

## Architectural Discipline

### Single Source of Truth (SSOT)
Before adding a new abstraction (registry, manager, factory, handler, schema module):
1. Grep for existing abstractions with similar purposes
2. If one exists, USE IT or DOCUMENT WHY it's being replaced + DELETE the old in the same commit
3. Never leave two systems doing the same job "for compat"

### No Orphan Migrations
If you introduce a "v2" of something:
- Migrate ALL callers in the same session, OR
- Record remaining work in memory (`type: project`) with explicit list of what's left
- NEVER leave "for compat" code without a tracked deletion date

### When Adding, Remove
Every commit that adds a new abstraction must identify what it replaces and delete that. If nothing is replaced, the commit message must justify why the new abstraction is needed.

### Complete Deletion, Not Deprecation
Don't mark code `// DEPRECATED` or `// TODO remove later`. Delete it. If deletion is blocked, file an issue and link it from the code.

## Pushback Triggers — When to Stop and Ask

STOP and ask the user before proceeding when you find:

1. **Duplication** — two systems/modules doing similar things
   → "Is this intentional? Should I consolidate before adding my feature?"
2. **Stale state** — `TODO`, `FIXME`, `for compat`, `temporary`, `v2` comments in code you're touching
   → "Is this still needed? Should I resolve it first?"
3. **Red CI** — workflows failing on main
   → "Should I fix CI first before doing the task?"
4. **Version drift** — two versions of the same dependency in Cargo.lock
   → "Is this intentional or should it be consolidated?"
5. **Request conflicts with architecture** — user asks you to add X but X exists differently
   → Surface the existing thing, ask which to use
6. **Half-finished migrations** — both old and new paths still live
   → "Finish migration first, or add feature on top?"

Default behavior is to complete the requested task. These triggers override that.

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
- **Metrics** (`MetricsProvider`) — latency, token, cost tracking, OTel metrics export
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
├── lib.rs           # Re-exports all runners + shared types
├── types.rs         # LlmProvider trait, RunnerError, ChatRequest, ChatResponse
├── config.rs        # RunnerConfig, CliRunnerType enum
├── factory.rs       # Runner factory, provider parsing, ALL_PROVIDERS
├── cli_common.rs    # CliRunnerBase + macro for runner boilerplate
├── {runner}.rs      # Per-runner implementations (claude_code, copilot, etc.)
├── agent.rs         # Multi-turn agent loop with tool calling
├── fallback.rs      # Ordered provider failover
├── metrics.rs       # Latency/token/cost tracking decorator
├── cache.rs         # Response caching
└── ...              # guardrail, quality_gate, structured_output, etc.

crates/
├── embacle-mcp/     # MCP server library + binary
└── embacle-server/  # Unified REST API + MCP server binary
```

### Key Design Decisions
- **100% standalone** — zero dependency on dravr-platform or pierre-core
- **Types in `types.rs`** — `RunnerError`, `LlmProvider` trait, `ChatRequest`, `ChatResponse`
- **Subprocess-based** — wraps CLI tools via `tokio::process::Command`
- **ACP via feature flag** — `copilot-headless` adds NDJSON/JSON-RPC transport for `copilot --acp`
- **No HTTP dependencies in core** — only tokio (process), serde, tracing, which, bitflags

# Writing code

- CRITICAL: NEVER USE `--no-verify` WHEN COMMITTING CODE
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
- NEVER introduce a new abstraction without deleting what it replaces in the same commit
- NEVER leave half-finished migrations — either complete or record state in memory
- ALWAYS grep for existing abstractions before adding a new one
- Be RUST idiomatic
- Do not hard code magic values
- Do not leave implementation with "In future versions" or "Implement the code" or "Fall back". Always implement the real thing.
- Commit without AI assistant-related commit messages. Do not reference AI assistance in git commits.
- Always create a branch when adding new features. Bug fixes go directly to main branch.
- Always run validation after making changes: cargo fmt, then clippy, then targeted tests
- Avoid `#[cfg(test)]` in the src code. Only in tests

## Security Engineering Rules

### Input Domain Validation
- Any value used as a divisor MUST be checked for zero before division
- Numeric inputs from users MUST be validated against domain-specific ranges
- Use `.max(1)` or equivalent guard before any division operation

### Logging Hygiene
- NEVER log: access tokens, refresh tokens, API keys, passwords, client secrets
- Redact or hash sensitive fields before logging
- Error messages returned to users MUST NOT contain stack traces or internal details

## Error Handling Requirements

### Acceptable
- `?` operator for error propagation
- `Result<T, E>` for all fallible operations
- `Option<T>` for values that may not exist
- Custom error types implementing `std::error::Error`

### Prohibited
- `unwrap()` except in test code or static data known at compile time
- `expect()` — only for documenting invariants that should never fail
- `panic!()` — only in test assertions
- **`anyhow!()` macro** — ABSOLUTELY FORBIDDEN in all production code

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

- PREFER real implementations over mocks in all production code
- NEVER implement mock modes for production features
- Mocks permitted ONLY in test code for error conditions difficult to reproduce or network failure simulation

## Documentation Standards

- All public APIs MUST have comprehensive doc comments
- Use `/// ` for public API documentation
- Document error conditions and panic scenarios
- Include usage examples for complex APIs
