# Changelog

## [0.44.0] — 2026-10-29

### Fixed

- fix(vision): send images through the Gemini API and Claude Code runners

### Other

- chore(deps): bump dravr-tronc 6.4.0 -> 6.5.0
- chore(deps): bump dravr-tronc 6.3.0 -> 6.4.0


## [0.43.0] — 2026-10-29

### Added

- feat(anthropic): load deferred tools on demand through tool search

### Other

- chore(deps): bump dravr-tronc 6.2.1 -> 6.3.0
- review: apply coding standards to the request-model fix


## [0.42.0] — 2026-10-27

### Added

- feat(anthropic): cache a system message's stable head as its own block
- feat(anthropic): 1-hour TTL for the tools and system cache breakpoints

### Other

- docs(anthropic): document ANTHROPIC_PREFIX_CACHE_TTL


## [0.41.0] — 2026-10-27

### Added

- feat(anthropic): count a request's input tokens with count_tokens


## [0.40.1] — 2026-10-27

### Fixed

- fix(runners): run every CLI runner on the model the request names

### Other

- build(cargo): declare rust-version, the fleet pin


## [0.40.0] — 2026-10-26

### Added

- feat(anthropic): stream tool-calling turns (complete_stream_with_tools)

### Fixed

- fix(tool-host): port the bearer mint to rand 0.10

### Other

- build(cargo): drop opentelemetry_sdk from the library's otel feature
- build(cargo): name the features of every optional dependency
- chore(deps): bump rand from 0.8.8 to 0.10.3


## [0.39.1] — 2026-10-23

### Fixed

- fix(copilot-sdk): take github-copilot-sdk 1.0.26 with `runtime`
- fix(usage): forward cache and reasoning tokens; refresh Copilot catalog

### Other

- chore(deps): bump dravr-tronc 6.2.0 -> 6.2.1
- chore(build): bump .build to 520f637, the fleet Rust pin check


## [0.39.0] — 2026-10-20

### Other

- chore(deps): bump dravr-tronc 6.1.0 -> 6.2.0
- chore(deps): bump dravr-tronc 6.0.0 -> 6.1.0
- chore(build): bump .build to 50dd908 and drop an unneeded clippy allow
- chore(toolchain): bump Rust 1.98.1 to 1.99.1
- chore(deps): bump dravr-tronc 5.2.0 -> 6.0.0
- chore(deps): bump dravr-tronc 5.1.0 -> 5.2.0
- chore(deps): bump dravr-tronc 5.0.0 -> 5.1.0


## [0.38.0] — 2026-10-15

### Other

- chore(deps): bump dravr-tronc to 5.0.0


## [0.37.0] — 2026-10-15

### Fixed

- fix(copilot): the runtime keeps its package cache and home variables
- fix(stream): a CLI stream honours RunnerConfig::timeout and caps lines
- fix(cli): the MCP bearer and the claude prompt stay off argv and logs
- fix(copilot): the ACP and SDK runtimes inherit only an allowlisted env

### Other

- chore(deps): bump dravr-tronc 4.3.0 -> 4.4.1


## [0.36.1] — 2026-10-14

### Fixed

- fix(router): a penalty whose Redis write failed survives the next read

### Other

- docs(agents): move the agent knowledge into the private dravr-savoir
- chore(deps): bump dravr-tronc 4.2.0 -> 4.3.0


## [0.36.0] — 2026-10-12

### Added

- feat(router): Redis quota store and ACP MCP-instructions opt-in


## [0.35.0] — 2026-10-12

### Added

- feat(fallback): shared quota cooldown book; claude OAuth beats API key
- feat(http-api): Anthropic Messages API provider and HTTP key pool

### Fixed

- fix(pricing): Claude Haiku 5.5 in the Copilot catalogue and pricing

### Other

- chore(deps): bump dravr-tronc 4.1.0 -> 4.2.0
- chore(deps): bump dravr-tronc 4.0.0 -> 4.1.0


## [Unreleased]

### Added

- feat(http-api): `AnthropicProvider` over the Anthropic Messages API — Claude Haiku / Sonnet / Opus 5.5 (default `claude-haiku-5-5`), streaming, tool use with merged parallel tool results, vision, prompt-cache breakpoints (tools, system, conversation tail), gross usage with cache read/write counts, `ANTHROPIC_EFFORT` / `ANTHROPIC_THINKING`
- feat(pool): `provider_pool` names any already-built providers as pool accounts (`<name>#N`); `AnthropicConfig::pool_from_env` reads `ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY_2`, … as one account per Anthropic organization
- feat(pricing): `anthropic` rows for `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`, with their cache read/write rates
- feat(fallback): `QuotaCooldowns`, a cheaply cloneable quota-cooldown book keyed by tier name (`is_cooling`, `remaining`, `start`, `start_for`, `end`, `duration`), and `FallbackProvider::with_cooldowns` so several chains — a full chain and the chains over its tails — share one book and agree on which tier is cooling; a caller asking a tier outside any chain reads and writes the same book. Without the call a chain keeps a private book of its own; `with_quota_cooldown` now overrides the book's duration for the refusals that chain records. `DEFAULT_QUOTA_COOLDOWN` (15 min) is public, and `QUOTA_COOLDOWN_SKIP` is the `FallthroughReason::Skipped` reason a chain gives for a tier it passes over inside its cooldown, so an observer can tell that skip from its own vetoes
- feat(pricing)!: `ModelPricing::with_long_prompt_rates` and the new `long_prompt` field — a second rate card above a prompt-length threshold; Haiku 5.5 bills a prompt over 100K tokens at $0.50 / $2.50
- feat(copilot): `claude-sonnet-5.5` and `claude-haiku-5.5` in the Copilot catalogue, so `copilot_sdk` and `copilot_headless` publish them; each heads its family, and `default_effort_for("claude-haiku-5.5")` is `medium` (Haiku 4.5 still takes no effort)
- feat(pricing): Claude Haiku 5.5 ($0.10 / $0.50) and Sonnet 5.5 ($2 / $10) on `copilot_headless`, `copilot_sdk` and `claude-code`, under both `claude-haiku-5.5` / `claude-haiku-5-5` and `claude-sonnet-5.5` / `claude-sonnet-5-5`; exact ids, no `claude-haiku-5` family prefix. These runners report a whole turn's summed usage, so Haiku 5.5 bills its base card there; only the `anthropic` row switches to the long-prompt card

### Changed

- fix(fallback): quota cooldowns are keyed by tier name, not tier position, in a chain's private book too — two tiers with the same name in one chain share a window (a refusal from either passes over both, an answer from either ends it), and a `RouterProvider` tier is keyed by its active backend, so it is passed over only while the backend active on the next call is the one cooling
- fix(http-api)!: HTTP 402 (credit balance exhausted) maps to `ErrorKind::RateLimit` for every HTTP provider, so a fallback chain cools the spent tier down instead of retrying it in place; it was `ExternalService`
- fix(http-api): HTTP 529 (overloaded) is retried in place like 502/503
- fix(pool)!: `PooledTier` no longer overwrites `ChatResponse::model` with the tier name; a response keeps the model the account ran, so a pooled per-token tier is priced instead of recorded at $0 (the account is still named by the tier's `name()` and in its errors)
- fix(http-api): the SSE parser buffers raw bytes and decodes whole lines, so a multi-byte UTF-8 character split across two network chunks is no longer replaced by U+FFFD
- fix(claude-code)!: `ChatResponse::model` names the model the turn ran on — the busiest `modelUsage` key of the CLI's result document (`claude-haiku-5-5`), else the value passed as `--model` — instead of the runner name `"claude-code"`, which no price row matches, so every Claude Code turn billed $0 whatever model served it
- fix(claude-code): `Credit balance is too low` is a quota refusal (`RateLimit`), the same policy as an HTTP 402

### Fixed

- fix(pricing): `lookup_pricing` takes the longest matching prefix instead of the first in table order, so `gpt-4o` no longer shadows `gpt-4o-mini` on `openai_api` and the `claude-sonnet` row no longer shadows `claude-sonnet-5.5`
- fix(pricing): Claude Haiku 4.5 bills at its $1 / $5 list price on `copilot_headless`, `copilot_sdk` and `claude-code`; it was $0.80 / $4
- fix(pricing): Claude Opus and Sonnet 5 list prices on `copilot_headless`, `copilot_sdk` and `claude-code`, under both the dotted and the hyphenated id — Opus 5.5 $4 / $20 with cache reads at 0.05x (it had no row, so it billed $0); Opus 5 $5 / $25 (no row before); Opus 4.8, 4.7 and 4.6 $5 / $25 (they billed $15 / $75 through the `claude-opus-4` row); Sonnet 5 $2 / $10 (it billed $3 / $15 through the `claude-sonnet` row). Opus 4.5, 4.1 and 4.0 stay on the `claude-opus-4` row and Sonnet 4.x on `claude-sonnet`. Copilot's `-fast` tiers of Opus 4.7 and 4.6 keep the $15 / $75 they billed before, on rows of their own: the reference lists no price for them, only that fast mode costs more than standard
- fix(tests): the fake-CLI runner tests retry a spawn that fails with `ETXTBSY` (`Text file busy`), which a concurrent test's fork holding the just-written script's write descriptor causes on Linux
- fix(claude-code): an OAuth token beats an API key — when `CLAUDE_CODE_OAUTH_TOKEN` reaches the `claude` child (a pool account's `with_env`, or the host's), `ANTHROPIC_API_KEY` is removed from its environment, so a service that sets the key for the Anthropic HTTP tier no longer moves every Claude Code tier off its subscription onto API credit; the key is forwarded only when no token exists
- fix(claude-code): when the sandbox policy cannot be built, the runner's explicit `with_env` credential is still set on the inherited-environment child


## [0.34.0] — 2026-10-03

### Other

- chore(deps): bump dravr-tronc 3.3.0 -> 4.0.0


## [0.33.2] — 2026-10-02

### Fixed

- fix(process): kill the CLI child when its call is dropped

### Other

- chore(deps): bump dravr-tronc 3.2.0 -> 3.3.0


## [0.33.1] — 2026-10-01

### Fixed

- fix(copilot-headless): withhold Autopilot continuation text after reply


## [0.33.0] — 2026-09-30

### Fixed

- fix(claude-code): no memory, CLAUDE.md or project settings from cwd
- fix(runners)!: never resume a CLI session across calls

### Other

- chore(deps): bump dravr-tronc 3.1.0 -> 3.2.0
- chore(deps): bump dravr-tronc 3.0.0 -> 3.1.0


## [0.32.0] — 2026-09-27

### Added

- feat(quota): GitHub rate limit as absolute remaining/limit/reset
- feat(tools): FunctionCall serializes as {name, args}



## [0.31.0] — 2026-09-26

### Fixed

- fix(uhp): configured harnesses run with their skills, MCP and tools

### Other

- chore(deps): bump dravr-tronc 2.0.0 -> 2.1.0



## [0.30.0] — 2026-09-25

### Other

- chore(deps): dravr-tronc 2.0.0; ungranted tools answer -32602
- refactor: bool::ok_or_else in the content-length guardrail (Rust 1.98)



## [0.29.1] — 2026-09-25

### Fixed

- fix(uhp): confine artifact downloads to a held session's own folder
- fix(acp): a streamed turn spawns under the model routing gate

### Other

- chore(deps): bump dravr-tronc 1.3.0 -> 1.4.0
- chore(deps): bump dravr-tronc 1.2.0 -> 1.3.0
- chore(deps): bump dravr-tronc 1.1.0 -> 1.2.0
- test(e2e): the streamed turn is asked for a verbatim reply, not a count



## [0.29.0] — 2026-09-22

### Added

- feat(pool): N accounts of one CLI runner as ordered chain tiers
- feat(ci): live Copilot tests run on a runner-local model, not an account

### Other

- test(delivery): the runner list is held to the enum, the SDK is named



## [0.28.2] — 2026-09-21



## [0.28.1] — 2026-09-21



## [0.28.0] — 2026-09-18



### Changed

- `openai-api` now implies `http-api` and `OpenAiApiRunner` maps statuses through the shared `http_api::client::map_http_error`: a 429 is `ErrorKind::RateLimit` (it was `ExternalService`, so it was retried in place and fell through a chain; now it propagates so a caller routes around the quota), and a 400/422 is the new `ErrorKind::InvalidRequest` (it was `ExternalService`). `embacle-server` answers `InvalidRequest` with HTTP 400 `invalid_request_error`.
- `ErrorKind` gains `InvalidRequest`; an exhaustive `match` on it needs a new arm.
- `MetricsProvider` prices every call through `embacle::pricing`, keyed on the wrapped provider's `name()`; `TokenPricing`, `PricingTable`, `default_pricing_table`, `with_pricing` and `with_default_pricing` are removed.
- `FallbackProvider::health_check` returns the last tier's outcome verbatim when no tier is healthy, instead of collapsing an `Err` into `Ok(false)`.
- The shared SSE stream always ends with exactly one final chunk, synthesising a `stop` when the byte stream closes without one.

## [0.27.0] — 2026-09-18

### Added

- feat(copilot-sdk): a provider over GitHub's Rust runtime, stdio only

### Fixed

- fix(copilot): list claude-sonnet-5 in the catalog; CI on features/*
- fix(ci): closed-stdin brew steps, and ignore the claude_docs symlink

### Other

- Revert "release: v0.27.0"
- perf(cache): O(1) eviction with a VecDeque insertion order



## [0.26.0] — 2026-09-10

### Added

- feat(uhp): serve the Unified Harness Protocol, fully conformant
- feat(router): hand back the ACP runner, but only while it is answering

### Other

- chore(deps): bump dravr-tronc 1.0.1 -> 1.1.0



## [0.25.3] — 2026-09-08



## [0.25.2] — 2026-09-08



## [0.25.1] — 2026-09-07

### Fixed

- fix(examples): acp_cache_boundary_probe declares the feature it imports
- fix(copilot): the headless runner works in a directory of its own

### Other

- test(copilot): the PATH-search assertion compares the stem
- chore(deps): bump dravr-tronc 1.0.0 -> 1.0.1
- chore(build): .build 87f8c6d -> 660d542



## [0.25.0] — 2026-09-04

### Other

- chore(deps): bump dravr-tronc 0.11.0 -> 1.0.0



## [0.24.0] — 2026-09-04



## [0.23.0] — 2026-09-02



## [0.22.6] — 2026-09-01



## [0.22.5] — 2026-09-01

### Fixed

- fix(acp): the model a turn runs on cannot be changed by a concurrent spawn

### Other

- chore(tronc): move to dravr-tronc 0.8.0



## [0.22.4] — 2026-08-30



## [0.22.3] — 2026-08-30

### Other

- perf(acp): the production chat path runs on the pool, and only reads its own session



## [0.22.2] — 2026-08-28

### Fixed

- fix(ci): clear the expect/panic gate, red on main since db6de60
- fix(acp): derive the session ceiling from measured memory, not a guess



## [0.22.1] — 2026-08-28



## [0.22.0] — 2026-08-27

### Added

- feat(usage): TokenUsage carries the cache counts providers already send
- feat(examples): a probe that asks the model whether it owns a host tool



## [0.21.3] — 2026-08-25

### Fixed

- fix(tests): assert the parser's output without .expect()
- fix(copilot-headless): an empty turn is retried, not returned



## [0.21.2] — 2026-08-22



## [0.21.1] — 2026-08-19



## [0.21.0] — 2026-08-19

### Added

- feat(tool-host): ask the caller's surface per call, not once per session
- feat(copilot): give the non-ACP runner real tools, via MCP
- feat(tool-host): host a caller's tools to an ACP agent over loopback MCP

### Fixed

- fix(tool-host): keep the error kind when adapting an McpToolExecutor
- fix(tool-host): survive a poisoned lock, and pin isolation and shutdown
- fix: repair the SessionStart bootstrap guard for an empty .build

### Other

- ci(release): bump and publish embacle-tool-host with the rest
- test(tool-host): prove streaming carries the caller's tools
- style(tests): rustfmt and SPDX header for the permission policy test



## [0.20.0] — 2026-08-07

### Fixed

- fix: clear the warnings the flag removal left behind



## [0.19.7] — 2026-07-28



## [0.19.6] — 2026-07-13



## [0.19.5] — 2026-07-10

### Fixed

- fix(claude-code): classify prompt-too-long as ContextLength (400), not 502
- fix(clippy): backtick doc addrs, is_ok_and, unwrap-free auth tests CI clippy --all-targets flagged doc_markdown (127.0.0.0/8, ::1) + map().unwrap_or -> is_ok_and in is_loopback_host; arch-validation counted 3 test unwrap()s — replaced with matches!. Fixes the 52d04d6 CI failure.

### Other

- test(copilot_headless): annotate safe test unwraps for arch-validation 15 test-only unwrap sites (tempdir/fs/serde setup) get same-line // Safe: comments; zero flagged remain



## [0.19.4] — 2026-06-29

### Added

- feat(ci): auto-validate Homebrew install after release release.yml now calls homebrew.yml (reusable workflow_call) as a final gated job once the tap formula is pushed, passing the released version for an exact match; drops the vestigial push:tags trigger (GITHUB_TOKEN tag pushes never fired it and would race the formula update). workflow_dispatch retained for manual runs.

### Fixed

- fix(copilot-headless): route ACP sessions to the configured model via settings.json copilot --acp routes by ~/.copilot/settings.json "model" and ignores both the session/new model field and the --model flag (cosmetic label only); write the model into settings.json before each spawn (preserving other keys), track the warm-process model and respawn on change, and log the real routing model — fixes ACP sessions silently falling through to the account GPT/Gemini default while logs claimed the requested model.
- fix(ci): trust dravr-ai tap for Homebrew 6.0 install (ubuntu) Homebrew 6.0 refuses untrusted third-party taps; brew trust dravr-ai/tap (|| true for older brew) before install. Last of 4 Homebrew jobs.
- fix(ci): make Homebrew test harness pass (llvm-nm, linuxbrew PATH, 503 health) macOS FFI symbols via toolchain llvm-nm (Apple nm can't read Rust 1.96 LLVM objects); add linuxbrew to PATH on ubuntu (brew not found); smoke test accepts degraded 503 /health (no provider CLIs in CI) by dropping curl -f. All validated locally.



## [0.19.3] — 2026-06-22

### Added

- feat(mcp): migrate to dravr-tronc 0.5.3; release v0.19.2 ActiveConfig->interior RwLock (provider/model/multiplex accessors &self async); init protocolVersion now 2025-11-25; -mcp/-server only.
- feat(release): adopt dravr-tronc 0.3 via coordinated republish (embacle-mcp/server 0.19.1) Republish embacle-mcp+embacle-server 0.19.1 on tronc 0.3.1 so the publish graph resolves a single tronc version; root embacle (no tronc dep) stays 0.19.0.

### Fixed

- fix(ci): combine publish dry-run into one workspace invocation Per-crate dry-run resolved internal embacle dep vs crates.io (only 0.19.0 there, not the unpublished 0.19.2) and failed; co-publishing all three resolves siblings locally. Verified cargo 1.96.0 exit 0.

### Other

- style(lints): enforce error-handling restriction lints; create_runner_with_config -> Result Sole prod fix: ClaudeWeb embedded-config parse propagated via ? (was expect); crate-root cfg_attr(test, allow) + tests/examples allow attrs; --all-targets clippy green.
- Revert "chore(deps): bump dravr-tronc 0.2 -> 0.3" (publish-dry-run version skew) embacle-server publishes against registry embacle-mcp 0.19.0 (tronc 0.2) -> two tronc versions clash; needs a coordinated member-crate release to adopt 0.3.
- chore(deps): bump dravr-tronc 0.2 -> 0.3 (now published to crates.io) Server/mcp crates compile against 0.3.1; picks up otel + NotifyLayer + error-handling fixes.



## [0.19.2] — 2026-06-19

### Changed

- deps: migrate `embacle-mcp` and `embacle-server` to dravr-tronc 0.5.3 (dual-era
  MCP engine). tronc 0.5 hands tools/handlers a shared `&Arc<ServerState>` with no
  outer `RwLock`, so the active provider/model/multiplex config moved to a
  per-field interior `RwLock<ActiveConfig>`; those accessors are now `&self` async
  (the runner cache was already interior `Mutex`). All three crates aligned to a
  single workspace version. The core `embacle` crate is unchanged.

## [0.19.0] — 2026-06-07



## [0.18.0] — 2026-06-06

### Added

- feat(acp): per-request mcp_servers for native Copilot ACP tool calling ChatRequest.mcp_servers flows into session/new; opt-in mcp_tool_calling re-enables SDK_TOOL_CALLING to route tool turns through converse().

### Fixed

- fix(example): Duration::from_mins in web_login for clippy 1.96



## [0.17.0] — 2026-06-05

### Added

- feat(web-ui): browser-driven Claude.ai provider (REST + MCP) via dravr-browser

### Other

- test(e2e): skip Copilot headless E2E when backend auth unavailable



## [0.16.0] — 2026-06-04



## [0.15.15] — 2026-06-02



## [0.15.14] — 2026-05-29

### Fixed

- fix(tool-sim): strip echoed <tool_result> scaffolding from CLI replies



## [0.15.13] — 2026-05-19



## [0.15.12] — 2026-05-19



## [0.15.11] — 2026-05-19



## [0.15.10] — 2026-05-19



## [0.15.9] — 2026-05-19



## [0.15.8] — 2026-05-19



## [0.15.7] — 2026-05-13

### Fixed

- fix(cli): surface stdout when stderr is empty claude-code on exit=1 writes the error to stdout while emitting zero bytes to stderr. Prefer stderr, fall back to stdout, truncate at 500 chars.

### Other

- build(deps): bump softprops/action-gh-release to v3
- build(deps): bump agent-client-protocol-schema to 0.12



## [0.15.6] — 2026-05-08



## [0.15.5] — 2026-05-01

### Other

- obs: payload logging across CLI runners, ACP path, and openai_api Process::run_cli_command emits info (lengths, latency) + trace (argv, full output) covering all 12 runners; copilot_headless::complete and openai_api gain matching pre/post call info+trace; agent::run logs each multi-turn iteration.



## [0.15.4] — 2026-04-23



## [0.15.3] — 2026-04-21



## [0.15.2] — 2026-04-21



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


