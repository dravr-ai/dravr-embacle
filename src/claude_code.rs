// ABOUTME: Claude Code CLI runner implementing the `LlmProvider` trait
// ABOUTME: Wraps the `claude` CLI with JSON output parsing and session management
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::io;
use std::process::Stdio;
use std::str;

use crate::cli_common::{CliRunnerBase, MAX_OUTPUT_BYTES};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, McpServerConfig,
    McpTransport, RunnerError, StreamChunk, TokenUsage,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::wrappers::LinesStream;
use tokio_stream::StreamExt;
use tracing::{debug, instrument, warn};

use crate::config::RunnerConfig;
use crate::process::{read_stderr_capped, run_cli_command};
use crate::prompt::{extract_system_message, prepare_user_prompt};
use crate::sandbox::{apply_sandbox, build_policy};
use crate::stream::{GuardedStream, MAX_STREAMING_STDERR_BYTES};

/// Serialize embacle MCP server configs into Claude Code's `--mcp-config` JSON.
///
/// Claude Code reads the same `{"mcpServers": {name: {...}}}` shape as a
/// `.mcp.json` file: HTTP and SSE entries are `type`-tagged and carry
/// `{url, headers}` as an object, stdio entries carry `{command, args, env}`.
/// embacle models headers and env as `Vec<McpHeader>`, so both collapse to a
/// map here.
fn mcp_servers_to_claude_json(servers: &[McpServerConfig]) -> String {
    let entries: serde_json::Map<String, Value> = servers
        .iter()
        .map(|s| {
            let cfg = match &s.transport {
                McpTransport::Http { url, headers } => json!({
                    "type": "http",
                    "url": url,
                    "headers": headers.iter().map(|h| (h.name.clone(), Value::String(h.value.clone()))).collect::<serde_json::Map<_, _>>(),
                }),
                McpTransport::Sse { url, headers } => json!({
                    "type": "sse",
                    "url": url,
                    "headers": headers.iter().map(|h| (h.name.clone(), Value::String(h.value.clone()))).collect::<serde_json::Map<_, _>>(),
                }),
                McpTransport::Stdio { command, args, env } => json!({
                    "command": command,
                    "args": args,
                    "env": env.iter().map(|e| (e.name.clone(), Value::String(e.value.clone()))).collect::<serde_json::Map<_, _>>(),
                }),
            };
            (s.name.clone(), cfg)
        })
        .collect();
    json!({ "mcpServers": entries }).to_string()
}

/// The `--allowed-tools` selector that admits every tool a server publishes.
///
/// Claude Code namespaces MCP tools `mcp__<server>__<tool>`, and accepts the
/// server-level prefix on its own. The caller does not know the tool names
/// ahead of the session, so naming the servers is the only selector that can
/// be built up front — and it is still narrower than allowing everything.
fn allowed_tools_for(servers: &[McpServerConfig]) -> String {
    servers
        .iter()
        .map(|s| format!("mcp__{}", s.name))
        .collect::<Vec<_>>()
        .join(",")
}

/// Normalise Claude's usage to the GROSS prompt convention every other runner reports.
///
/// Providers disagree about whether the prompt count includes its cached share, and the
/// disagreement is invisible at the field name. `OpenAI` and Codex report gross: the cached
/// figure is a subset of `prompt_tokens`. Anthropic reports **fresh-only** — cache reads and
/// creations are separate and *additive*, so the real prompt is
/// `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
///
/// Measured rather than assumed, on a real turn: `input_tokens` 2 against
/// `cache_read_input_tokens` 18483, which is only possible if the cached share is outside.
///
/// Consumers carve the cached share back out of the gross prompt to price it — a cache read at
/// a discount, a write at a premium. Handing them a fresh-only figure makes that subtraction
/// clamp the real counts to nothing: a ~38 000-token turn priced as 2. Normalising here keeps
/// one convention behind one field name, rather than pushing a per-provider special case into
/// every consumer that ever prices a token.
fn gross_usage(u: &ClaudeUsage) -> TokenUsage {
    let fresh = u.input.unwrap_or(0);
    let cache_read = u.cache_read.unwrap_or(0);
    let cache_creation = u.cache_creation.unwrap_or(0);
    let output = u.output.unwrap_or(0);
    let prompt = fresh
        .saturating_add(cache_read)
        .saturating_add(cache_creation);
    TokenUsage::new(prompt, output, prompt.saturating_add(output))
        .with_cache(u.cache_read, u.cache_creation)
}

/// Whether a failed Claude Code turn was refused for quota rather than fault.
///
/// The vocabulary is Claude Code's own. Its internal classifier keys on the
/// phrase `usage limit reached`, and the API error envelope carries
/// `rate_limit_error` (throttling) or `billing_error` (the account's own cap)
/// as the `type`. Matching those rather than inventing phrasing means this
/// keeps working when the prose around them changes.
///
/// `credit balance too low` is deliberately NOT here. It reads like a limit
/// but never resets on a clock, so routing away from it and waiting would wait
/// forever; it stays an ordinary external-service failure.
fn is_quota_refusal(terminal_reason: Option<&str>, message: &str) -> bool {
    if terminal_reason.is_some_and(|r| {
        r.eq_ignore_ascii_case("rate_limit") || r.eq_ignore_ascii_case("billing_error")
    }) {
        return true;
    }
    let lower = message.to_ascii_lowercase();
    lower.contains("usage limit reached")
        || lower.contains("rate_limit_error")
        || lower.contains("billing_error")
        || lower.contains("rate limited")
}

/// Default model for Claude Code
const DEFAULT_MODEL: &str = "opus";

/// Fallback model list when no runtime override is available
const FALLBACK_MODELS: &[&str] = &["sonnet", "opus", "haiku"];

/// Claude Code CLI response JSON structure
#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    result: Option<String>,
    #[serde(default)]
    is_error: bool,
    /// Why the CLI stopped. `"prompt_too_long"` marks a permanent context-window
    /// overflow (vs a transient upstream failure) so we can surface it as a
    /// non-retryable 400 instead of a retryable 502.
    #[serde(default)]
    terminal_reason: Option<String>,
    session_id: Option<String>,
    usage: Option<ClaudeUsage>,
}

/// Token usage from Claude Code CLI
#[derive(Debug, Deserialize)]
struct ClaudeUsage {
    #[serde(rename = "input_tokens")]
    input: Option<u32>,
    #[serde(rename = "output_tokens")]
    output: Option<u32>,
    /// Prompt tokens served from Anthropic's prompt cache. Present on every
    /// Claude Code turn and previously undeclared, so it deserialized away.
    #[serde(rename = "cache_read_input_tokens", default)]
    cache_read: Option<u32>,
    /// Prompt tokens written into the cache this turn.
    #[serde(rename = "cache_creation_input_tokens", default)]
    cache_creation: Option<u32>,
}

/// Claude Code CLI runner
///
/// Implements `LlmProvider` by delegating to the `claude` binary with
/// `--output-format json` for structured responses and optional session
/// resumption.
pub struct ClaudeCodeRunner {
    base: CliRunnerBase,
}

impl ClaudeCodeRunner {
    /// Create a new Claude Code runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            base: CliRunnerBase::new(config, DEFAULT_MODEL, FALLBACK_MODELS),
        }
    }

    /// Store a session ID for later resumption
    pub async fn set_session(&self, key: &str, session_id: &str) {
        self.base.set_session(key, session_id).await;
    }

    /// Build the base command with common arguments
    ///
    /// When `max_tokens` is `Some`, the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env var
    /// is injected after sandbox application so the CLI limits its output length.
    fn build_command(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
        output_format: &str,
        max_tokens: Option<u32>,
        mcp_servers: &[McpServerConfig],
    ) -> Command {
        let mut cmd = Command::new(&self.base.config.binary_path);
        cmd.args(["-p", prompt, "--output-format", output_format]);

        // stream-json requires --verbose flag in Claude Code CLI
        if output_format == "stream-json" {
            cmd.arg("--verbose");
        }

        if let Some(sys) = system_prompt {
            cmd.args(["--system-prompt", sys]);
        }

        let model = self
            .base
            .config
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        cmd.args(["--model", model]);

        // `--strict-mcp-config` is a boolean flag: it confines the session to
        // the servers named on this command line, so a developer's own
        // ~/.claude.json can never leak tools into a served turn. It is passed
        // whether or not we hand over servers of our own.
        //
        // With servers on the request, Claude Code connects and calls them
        // itself, the same arrangement `copilot --additional-mcp-config` uses;
        // the caller withholds the prose tool catalog in exactly that case, so
        // the model is never offered two tool surfaces at once. With none, the
        // flag alone leaves the turn tool-less and the prose catalog stands.
        if mcp_servers.is_empty() {
            cmd.arg("--strict-mcp-config");
        } else {
            cmd.args([
                "--mcp-config",
                &mcp_servers_to_claude_json(mcp_servers),
                "--strict-mcp-config",
                "--allowed-tools",
                &allowed_tools_for(mcp_servers),
            ]);
        }

        for arg in &self.base.config.extra_args {
            cmd.arg(arg);
        }

        if let Ok(policy) = build_policy(
            self.base.config.working_directory.as_deref(),
            &self.base.config.allowed_env_keys,
        ) {
            apply_sandbox(&mut cmd, &policy);
            debug!(
                allowed_keys = ?policy.allowed_env_keys,
                cwd = %policy.working_directory.display(),
                "Sandbox applied to claude command"
            );
        } else {
            warn!("Failed to build sandbox policy, running with inherited env");
        }

        // Inject max output tokens after sandbox (env_clear) so the value persists
        if let Some(tokens) = max_tokens {
            cmd.env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", tokens.to_string());
        }

        cmd
    }

    /// Parse a Claude Code JSON response into a `ChatResponse`
    fn parse_response(raw: &[u8]) -> Result<(ChatResponse, Option<String>), RunnerError> {
        let text = str::from_utf8(raw).map_err(|e| {
            RunnerError::internal(format!("Claude Code output is not valid UTF-8: {e}"))
        })?;

        let parsed: ClaudeResponse = serde_json::from_str(text).map_err(|e| {
            RunnerError::internal(format!("Failed to parse Claude Code JSON response: {e}"))
        })?;

        if parsed.is_error {
            let message = parsed
                .result
                .as_deref()
                .unwrap_or("Unknown error from Claude Code");
            // A context-window overflow is permanent for the identical prompt:
            // classify it as ContextLength (HTTP 400, non-transient) so the caller
            // shrinks the request instead of retrying the same oversized blob.
            // Detect via the CLI's structured `terminal_reason` first, falling back
            // to the human-readable message.
            let is_context_overflow = parsed
                .terminal_reason
                .as_deref()
                .is_some_and(|r| r.eq_ignore_ascii_case("prompt_too_long"))
                || message.to_ascii_lowercase().contains("prompt is too long");
            if is_context_overflow {
                // Phrase includes "context length" so OpenAI-compatible clients that
                // classify by message (not just HTTP status) recognise the overflow.
                return Err(RunnerError::context_length(format!(
                    "claude-code: context length exceeded — {message}"
                )));
            }
            // A quota refusal is not a fault: the provider is working and will
            // answer again once the window resets. Classifying it as
            // ExternalService made it indistinguishable from a flaky CLI, and
            // is_transient() then told every retry loop to keep hammering a
            // wall it cannot pass.
            if is_quota_refusal(parsed.terminal_reason.as_deref(), message) {
                return Err(RunnerError::rate_limit("claude-code", message));
            }
            return Err(RunnerError::external_service("claude-code", message));
        }

        let content = parsed.result.unwrap_or_default();
        let usage = parsed.usage.map(|u| gross_usage(&u));

        let response = ChatResponse {
            content,
            model: "claude-code".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        };

        Ok((response, parsed.session_id))
    }
}

#[async_trait]
impl LlmProvider for ClaudeCodeRunner {
    crate::delegate_provider_base!(
        "claude-code",
        "Claude Code CLI",
        LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS
    );

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        let mut cmd = self.build_command(
            prompt,
            system,
            "json",
            request.max_tokens,
            &request.mcp_servers,
        );

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
            }
        }

        let model_name = request
            .model
            .as_deref()
            .unwrap_or_else(|| self.base.default_model());
        debug!(
            binary = %self.base.config.binary_path.display(),
            model = model_name,
            has_system_prompt = system.is_some(),
            prompt_len = prompt.len(),
            "Spawning claude CLI"
        );

        let output = run_cli_command(&mut cmd, self.base.config.timeout, MAX_OUTPUT_BYTES).await?;
        self.base.check_exit_code(&output, "claude-code")?;

        let (response, session_id) = Self::parse_response(&output.stdout)?;

        if let Some(sid) = session_id {
            if let Some(model) = &request.model {
                self.base.set_session(model, &sid).await;
            }
        }

        Ok(response)
    }

    #[instrument(skip_all, fields(runner = "claude_code"))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let system = extract_system_message(&request.messages);
        let prepared = prepare_user_prompt(&request.messages)?;
        let prompt = &prepared.prompt;

        let mut cmd = self.build_command(
            prompt,
            system,
            "stream-json",
            request.max_tokens,
            &request.mcp_servers,
        );

        if let Some(model) = &request.model {
            if let Some(sid) = self.base.get_session(model).await {
                cmd.args(["--resume", &sid]);
            }
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            RunnerError::internal(format!("Failed to spawn claude for streaming: {e}"))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            RunnerError::internal("Failed to capture claude stdout for streaming")
        })?;

        let stderr_task = tokio::spawn(read_stderr_capped(
            child.stderr.take(),
            MAX_STREAMING_STDERR_BYTES,
        ));

        let reader = BufReader::new(stdout);
        let lines = LinesStream::new(reader.lines());

        let stream = lines.map(move |line_result: Result<String, io::Error>| {
            let line = line_result
                .map_err(|e| RunnerError::internal(format!("Error reading claude stream: {e}")))?;

            if line.trim().is_empty() {
                return Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                });
            }

            let value: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                RunnerError::internal(format!("Invalid JSON in claude stream: {e}"))
            })?;

            let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match chunk_type {
                "result" => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason: Some("stop".to_owned()),
                }),
                "assistant" => {
                    // Extract text from content array: message.content[].text where type == "text"
                    let text = value
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter(|item| {
                                    item.get("type").and_then(|t| t.as_str()) == Some("text")
                                })
                                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("")
                        })
                        .unwrap_or_default();
                    Ok(StreamChunk {
                        delta: text,
                        is_final: false,
                        finish_reason: None,
                    })
                }
                // system, rate_limit_event, and other event types are ignored
                _ => Ok(StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                }),
            }
        });

        Ok(Box::pin(GuardedStream::new(stream, child, stderr_task)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;
    use crate::types::McpHeader;

    #[test]
    fn test_parse_response_valid_json() {
        let json = br#"{"result":"Hello world","is_error":false,"session_id":"abc123","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion

        assert_eq!(response.content, "Hello world");
        assert_eq!(session_id, Some("abc123".to_owned()));
        assert_eq!(response.model, "claude-code");
        let usage = response.usage.unwrap(); // Safe: test assertion
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_response_error_flag() {
        // An ordinary fault: the CLI broke, and a retry may well succeed.
        // The fixture used to say "rate limited", which now classifies as a
        // quota refusal — see the tests below.
        let json = br#"{"result":"stream closed unexpectedly","is_error":true}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();

        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.kind.is_transient(), "a broken pipe is worth retrying");
        assert!(err.message.contains("stream closed unexpectedly"));
    }

    #[test]
    fn the_prompt_count_absorbs_the_cached_share() {
        // Measured on a real turn 2026-09-09: input 2, cache_read 18483,
        // cache_creation 19681. Anthropic's input_tokens excludes both, so a
        // consumer that carves the cached share back out of the prompt clamps
        // the real cost to nothing — a ~38k-token turn priced as 2.
        let json = br#"{"result":"ok","is_error":false,"usage":{
            "input_tokens":2,"output_tokens":7,
            "cache_read_input_tokens":18483,"cache_creation_input_tokens":19681}}"#;
        let (response, _) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion
        let usage = response.usage.expect("usage present"); // Safe: test assertion

        assert_eq!(
            usage.prompt_tokens, 38_166,
            "prompt must be gross — 2 + 18483 + 19681 — because that is the convention every \
             other runner reports and the one consumers carve the cached share back out of"
        );
        assert_eq!(
            usage.cached_read_tokens,
            Some(18_483),
            "the read stays separately priced"
        );
        assert_eq!(usage.cached_write_tokens, Some(19_681), "so does the write");
        assert_eq!(
            usage.total_tokens, 38_173,
            "total is the gross prompt plus output"
        );
    }

    #[test]
    fn a_turn_with_no_cache_is_unchanged() {
        // The normalisation must be invisible when there is nothing cached,
        // or it would inflate every uncached turn.
        let json =
            br#"{"result":"ok","is_error":false,"usage":{"input_tokens":500,"output_tokens":50}}"#;
        let (response, _) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion
        let usage = response.usage.expect("usage present"); // Safe: test assertion
        assert_eq!(usage.prompt_tokens, 500);
        assert_eq!(usage.total_tokens, 550);
        assert_eq!(
            usage.cached_read_tokens, None,
            "absent stays absent, never Some(0)"
        );
    }

    #[test]
    fn a_usage_limit_is_a_quota_refusal_not_a_fault() {
        // "usage limit reached" is Claude Code's own phrase — its internal
        // classifier keys on exactly this string.
        let json = br#"{"result":"Claude usage limit reached","is_error":true}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();

        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(
            !err.kind.is_transient(),
            "classifying a quota wall as transient is what made every retry loop hammer it; \
             the window resets on its own clock, not on a backoff schedule"
        );
    }

    #[test]
    fn the_api_error_envelope_types_are_recognised() {
        for body in [
            br#"{"result":"{\"type\":\"rate_limit_error\"}","is_error":true}"#.as_slice(),
            br#"{"result":"{\"type\":\"billing_error\"}","is_error":true}"#.as_slice(),
        ] {
            let err = ClaudeCodeRunner::parse_response(body).unwrap_err();
            assert_eq!(err.kind, ErrorKind::RateLimit, "body: {body:?}");
        }
    }

    #[test]
    fn a_low_credit_balance_is_not_a_quota_refusal() {
        // It reads like a limit but never resets on a clock. Routing away and
        // waiting for a reset would wait forever, so it stays an ordinary
        // failure the caller can see and act on.
        assert!(!is_quota_refusal(None, "credit balance is too low"));
    }

    #[test]
    fn a_structured_terminal_reason_is_enough_on_its_own() {
        assert!(is_quota_refusal(Some("rate_limit"), "something opaque"));
        assert!(is_quota_refusal(Some("billing_error"), "something opaque"));
        assert!(!is_quota_refusal(
            Some("turn_setup_failed"),
            "something opaque"
        ));
    }

    #[test]
    fn test_parse_response_prompt_too_long_is_context_length() {
        // A context-window overflow must classify as ContextLength (non-transient
        // 400), not ExternalService (transient 502) — otherwise the caller retries
        // the identical oversized prompt. Detected via structured terminal_reason.
        let json = br#"{"result":"Prompt is too long","is_error":true,"terminal_reason":"prompt_too_long"}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ContextLength);
        assert!(!err.kind.is_transient());
        assert!(err.message.contains("Prompt is too long"));
    }

    #[test]
    fn test_parse_response_prompt_too_long_via_message_fallback() {
        // Same classification when terminal_reason is absent but the message says so.
        let json = br#"{"result":"Prompt is too long","is_error":true}"#;
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ContextLength);
    }

    #[test]
    fn test_parse_response_missing_optional_fields() {
        let json = br#"{"result":"hi","is_error":false}"#;
        let (response, session_id) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion

        assert_eq!(response.content, "hi");
        assert!(session_id.is_none());
        assert!(response.usage.is_none());
    }

    #[test]
    fn test_parse_response_null_result() {
        let json = br#"{"is_error":false}"#;
        let (response, _) = ClaudeCodeRunner::parse_response(json).unwrap(); // Safe: test assertion
        assert_eq!(response.content, "");
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let json = b"not json at all";
        let err = ClaudeCodeRunner::parse_response(json).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Internal);
    }

    fn http_server() -> McpServerConfig {
        McpServerConfig {
            name: "dravr".to_owned(),
            transport: McpTransport::Http {
                url: "http://127.0.0.1:8081/mcp".to_owned(),
                headers: vec![McpHeader {
                    name: "Authorization".to_owned(),
                    value: "Bearer session-token".to_owned(),
                }],
            },
        }
    }

    #[test]
    fn an_http_server_carries_its_headers_as_an_object() {
        let cfg: Value =
            serde_json::from_str(&mcp_servers_to_claude_json(&[http_server()])).unwrap(); // Safe: serializer output
        let e = &cfg["mcpServers"]["dravr"];
        assert_eq!(e["type"], "http");
        assert_eq!(e["url"], "http://127.0.0.1:8081/mcp");
        assert_eq!(
            e["headers"]["Authorization"], "Bearer session-token",
            "the bearer must survive as a map entry — Claude Code reads headers as an object, \
             and a turn that loses it reaches an MCP server that will refuse every tool"
        );
    }

    #[test]
    fn a_stdio_server_carries_command_args_and_env() {
        let s = McpServerConfig {
            name: "local".to_owned(),
            transport: McpTransport::Stdio {
                command: "/usr/local/bin/pierre-mcp".to_owned(),
                args: vec!["--stdio".to_owned()],
                env: vec![McpHeader {
                    name: "TOKEN".to_owned(),
                    value: "abc".to_owned(),
                }],
            },
        };
        let cfg: Value = serde_json::from_str(&mcp_servers_to_claude_json(&[s])).unwrap(); // Safe: serializer output
        let e = &cfg["mcpServers"]["local"];
        assert_eq!(e["command"], "/usr/local/bin/pierre-mcp");
        assert_eq!(e["args"][0], "--stdio");
        assert_eq!(e["env"]["TOKEN"], "abc");
        assert!(
            e.get("type").is_none(),
            "stdio is the untagged shape; a type key here makes Claude Code read it as remote"
        );
    }

    #[test]
    fn every_server_reaches_the_config() {
        let second = McpServerConfig {
            name: "admin".to_owned(),
            transport: McpTransport::Sse {
                url: "http://127.0.0.1:8081/sse".to_owned(),
                headers: vec![],
            },
        };
        let cfg: Value =
            serde_json::from_str(&mcp_servers_to_claude_json(&[http_server(), second])).unwrap(); // Safe: serializer output
        let m = cfg["mcpServers"].as_object().unwrap(); // Safe: serializer output
        assert_eq!(m.len(), 2, "dropping a server silently removes its tools");
        assert_eq!(cfg["mcpServers"]["admin"]["type"], "sse");
    }

    #[test]
    fn the_allowlist_names_each_server_not_each_tool() {
        let second = McpServerConfig {
            name: "admin".to_owned(),
            transport: McpTransport::Sse {
                url: "http://127.0.0.1:8081/sse".to_owned(),
                headers: vec![],
            },
        };
        assert_eq!(
            allowed_tools_for(&[http_server(), second]),
            "mcp__dravr,mcp__admin",
            "tool names are not known before the session, so the server prefix is the only \
             selector that can be built up front"
        );
    }
}
