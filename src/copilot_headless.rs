// ABOUTME: CopilotHeadlessRunner wraps the copilot CLI via ACP (Agent Client Protocol) for LLM completions.
// ABOUTME: Keeps one copilot --acp subprocess alive across complete() calls so the GitHub→Copilot OAuth token exchange amortizes; per-call session/new is cheap on the warm transport.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio_stream::Stream;

use agent_client_protocol_schema as schema;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex as TokioMutex};
use tokio::time;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, field, info, instrument, trace, warn, Span};

use crate::copilot_headless_config::{CopilotHeadlessConfig, PermissionPolicy};
use crate::copilot_models::catalog_ids;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, McpHeader,
    McpServerConfig, McpTransport, MessageRole, RunnerError, StreamChunk, TokenUsage,
};

/// Default prompt timeout (5 minutes). Override with `EMBACLE_ACP_PROMPT_TIMEOUT_SECS`.
const DEFAULT_ACP_PROMPT_TIMEOUT_SECS: u64 = 300;

/// Read prompt timeout from env, falling back to [`DEFAULT_ACP_PROMPT_TIMEOUT_SECS`].
fn acp_prompt_timeout() -> Duration {
    let secs = env::var("EMBACLE_ACP_PROMPT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ACP_PROMPT_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Build the JSON params for an ACP `session/prompt` request.
///
/// Always includes `sessionId` and `prompt` blocks. When `max_tokens` is
/// specified, forwards it as `maxTokens` so the ACP provider can respect
/// the caller's output length limit.
fn build_prompt_params(session_id: &str, prompt: &[Value], max_tokens: Option<u32>) -> Value {
    let mut params = json!({
        "sessionId": session_id,
        "prompt": prompt,
    });
    if let Some(mt) = max_tokens {
        params["maxTokens"] = Value::from(mt);
    }
    params
}

/// Serialize embacle MCP server configs into the ACP `session/new`
/// `mcpServers` wire format.
///
/// Matches the Agent Client Protocol `McpServer` schema: HTTP/SSE are
/// `type`-tagged and carry `{name,url,headers:[{name,value}]}`; stdio is
/// untagged and carries `{name,command,args,env:[{name,value}]}`. The
/// `headers_to_json` shape pins the `{name,value}` pairs the schema's
/// `HttpHeader`/`EnvVariable` expect.
fn mcp_servers_to_acp_json(servers: &[McpServerConfig]) -> Vec<Value> {
    fn headers_to_json(headers: &[McpHeader]) -> Vec<Value> {
        headers
            .iter()
            .map(|h| json!({ "name": h.name, "value": h.value }))
            .collect()
    }

    servers
        .iter()
        .map(|server| match &server.transport {
            McpTransport::Http { url, headers } => json!({
                "type": "http",
                "name": server.name,
                "url": url,
                "headers": headers_to_json(headers),
            }),
            McpTransport::Sse { url, headers } => json!({
                "type": "sse",
                "name": server.name,
                "url": url,
                "headers": headers_to_json(headers),
            }),
            McpTransport::Stdio { command, args, env } => json!({
                "name": server.name,
                "command": command,
                "args": args,
                "env": headers_to_json(env),
            }),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// NDJSON transport
// ---------------------------------------------------------------------------

/// Async NDJSON transport for ACP JSON-RPC communication.
///
/// Handles reading/writing newline-delimited JSON messages over stdio pipes.
/// Each message is a single JSON line terminated by `\n`.
struct AcpTransport {
    writer: BufWriter<ChildStdin>,
    reader: BufReader<ChildStdout>,
    next_id: i64,
}

impl AcpTransport {
    fn new(stdin: ChildStdin, stdout: ChildStdout) -> Self {
        Self {
            writer: BufWriter::new(stdin),
            reader: BufReader::new(stdout),
            next_id: 1,
        }
    }

    /// Send a JSON-RPC request and return its id.
    async fn send_request(&mut self, method: &str, params: Value) -> Result<i64, RunnerError> {
        let id = self.next_id;
        self.next_id += 1;

        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        self.write_message(&msg).await?;
        Ok(id)
    }

    /// Send a JSON-RPC response (for server-to-client requests like permission).
    async fn send_response(&mut self, id: &Value, result: Value) -> Result<(), RunnerError> {
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        });
        self.write_message(&msg).await
    }

    /// Write a single NDJSON message.
    async fn write_message(&mut self, msg: &Value) -> Result<(), RunnerError> {
        let line = serde_json::to_string(msg)
            .map_err(|e| RunnerError::internal(format!("JSON serialization failed: {e}")))?;
        self.writer
            .write_all(line.as_bytes())
            .await
            .map_err(|e| RunnerError::internal(format!("Write failed: {e}")))?;
        self.writer
            .write_all(b"\n")
            .await
            .map_err(|e| RunnerError::internal(format!("Write newline failed: {e}")))?;
        self.writer
            .flush()
            .await
            .map_err(|e| RunnerError::internal(format!("Flush failed: {e}")))?;
        Ok(())
    }

    /// Default per-message read timeout (90 seconds).
    ///
    /// If the copilot process is alive but not sending any messages for this
    /// duration, the read is considered failed. This catches hung processes
    /// that don't produce output but haven't exited.
    /// Override with `EMBACLE_ACP_MESSAGE_TIMEOUT_SECS`.
    fn message_timeout() -> Duration {
        let secs = env::var("EMBACLE_ACP_MESSAGE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(90);
        Duration::from_secs(secs)
    }

    /// Read the next NDJSON message, skipping blank lines.
    ///
    /// Times out if no message arrives within [`Self::message_timeout`],
    /// detecting hung copilot processes that are alive but not responding.
    async fn read_message(&mut self) -> Result<Value, RunnerError> {
        let mut line = String::new();
        loop {
            line.clear();
            let read_result =
                time::timeout(Self::message_timeout(), self.reader.read_line(&mut line)).await;

            let n = match read_result {
                Ok(Ok(n)) => n,
                Ok(Err(e)) => {
                    return Err(RunnerError::internal(format!("Read failed: {e}")));
                }
                Err(_) => {
                    return Err(RunnerError::internal(format!(
                        "ACP message read timed out after {}s — copilot process may be hung",
                        Self::message_timeout().as_secs()
                    )));
                }
            };

            if n == 0 {
                return Err(RunnerError::internal("ACP connection closed unexpectedly"));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return serde_json::from_str(trimmed)
                .map_err(|e| RunnerError::internal(format!("JSON parse failed: {e}")));
        }
    }

    /// Read messages until we get the response matching the given request id.
    ///
    /// Non-matching messages (notifications, other responses) are skipped.
    async fn read_response(&mut self, expected_id: i64) -> Result<Value, RunnerError> {
        loop {
            let msg = self.read_message().await?;
            if msg.get("id").and_then(Value::as_i64) == Some(expected_id) {
                if let Some(error) = msg.get("error") {
                    return Err(RunnerError::external_service(
                        "copilot-acp",
                        format!("RPC error: {error}"),
                    ));
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ACP session lifecycle
// ---------------------------------------------------------------------------

/// Spawn the copilot --acp subprocess with piped stdio.
fn spawn_copilot(cli_path: &PathBuf, github_token: Option<&str>) -> Result<Child, RunnerError> {
    let mut cmd = Command::new(cli_path);
    cmd.arg("--acp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(token) = github_token {
        cmd.env("COPILOT_GITHUB_TOKEN", token);
    }

    info!(cli_path = %cli_path.display(), "Spawning copilot --acp subprocess");

    let child = cmd
        .spawn()
        .map_err(|e| RunnerError::internal(format!("Failed to spawn copilot --acp: {e}")))?;

    info!(
        pid = child.id().unwrap_or(0),
        "copilot --acp subprocess started"
    );
    Ok(child)
}

/// Default session setup timeout (60 seconds).
///
/// Copilot CLI may need 20–25s for first-run package extraction in containers,
/// plus time for auth handshake. Override with `EMBACLE_ACP_SESSION_TIMEOUT_SECS`.
const DEFAULT_ACP_SESSION_TIMEOUT_SECS: u64 = 60;

/// Read session setup timeout from `EMBACLE_ACP_SESSION_TIMEOUT_SECS` env var,
/// falling back to [`DEFAULT_ACP_SESSION_TIMEOUT_SECS`].
fn acp_session_timeout() -> Duration {
    let secs = env::var("EMBACLE_ACP_SESSION_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_ACP_SESSION_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Initialize ACP connection and create a session.
///
/// Returns the transport and session id ready for prompting.
/// Times out after [`acp_session_timeout`] to detect hung copilot processes early.
async fn setup_session(
    cli_path: &PathBuf,
    github_token: Option<&str>,
    model: &str,
    system_prompt: Option<&str>,
    mcp_servers: &[McpServerConfig],
) -> Result<(AcpTransport, Child, String), RunnerError> {
    let mut child = spawn_copilot(cli_path, github_token)?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| RunnerError::internal("Failed to capture copilot stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RunnerError::internal("Failed to capture copilot stdout"))?;

    let mut transport = AcpTransport::new(stdin, stdout);

    // Wrap handshake + session creation in a timeout to detect hung processes early
    let session_result = time::timeout(acp_session_timeout(), async {
        // Initialize handshake
        info!("ACP: sending initialize handshake");
        let init_id = transport
            .send_request(
                "initialize",
                json!({
                    "protocolVersion": 1,
                    "clientInfo": {
                        "name": "embacle",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                    "capabilities": {},
                }),
            )
            .await?;
        let init_resp = transport.read_response(init_id).await?;
        info!("ACP: initialize handshake complete");
        debug!(response = %init_resp, "ACP initialize response");

        // Create session with model, optional system prompt, and any MCP
        // servers the model should call tools from.
        let mut session_params = json!({
            "model": model,
            "cwd": env::current_dir()
                .map_err(|e| RunnerError::internal(format!("Failed to get cwd: {e}")))?,
            "mcpServers": mcp_servers_to_acp_json(mcp_servers),
        });
        if let Some(sys) = system_prompt {
            session_params["systemPrompt"] = Value::String(sys.to_owned());
        }

        info!(
            model = %model,
            has_system_prompt = system_prompt.is_some(),
            mcp_servers = mcp_servers.len(),
            "ACP: creating session"
        );
        let session_id_req = transport
            .send_request("session/new", session_params)
            .await?;
        let session_result = transport.read_response(session_id_req).await?;

        let session_id = session_result
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                RunnerError::external_service("copilot-acp", "Missing sessionId in response")
            })?
            .to_owned();

        info!(session_id = %session_id, model = %model, "ACP session created");
        Ok::<_, RunnerError>((session_id,))
    })
    .await;

    match session_result {
        Ok(Ok((session_id,))) => Ok((transport, child, session_id)),
        Ok(Err(e)) => {
            // Collect stderr for diagnostics before killing
            let stderr_output = collect_stderr(&mut child).await;
            warn!(
                error = %e,
                stderr = %stderr_output,
                "ACP session setup failed"
            );
            let _ = child.kill().await;
            Err(e)
        }
        Err(_elapsed) => {
            let stderr_output = collect_stderr(&mut child).await;
            warn!(
                stderr = %stderr_output,
                timeout_secs = acp_session_timeout().as_secs(),
                "ACP session setup timed out — copilot process may be hung (auth issue?)"
            );
            let _ = child.kill().await;
            Err(RunnerError::timeout(format!(
                "copilot-acp: session setup timed out after {}s (check copilot auth)",
                acp_session_timeout().as_secs()
            )))
        }
    }
}

/// A live `copilot --acp` subprocess that has completed the ACP `initialize`
/// handshake and is ready to accept `session/new` calls without re-running
/// the GitHub→Copilot OAuth token exchange.
///
/// Held inside [`CopilotHeadlessRunner::process`] so successful chat calls
/// amortize the auth handshake across the subprocess lifetime. The transport
/// is request/response by JSON-RPC id and cannot interleave concurrent
/// prompts, so the parent always wraps this in a `tokio::sync::Mutex`.
struct AcpProcess {
    child: Child,
    transport: AcpTransport,
}

impl AcpProcess {
    /// Spawn the subprocess and complete the ACP `initialize` handshake.
    /// On success the returned process is ready for [`Self::new_session`].
    ///
    /// On any handshake failure the subprocess is killed before the error
    /// is returned so callers never leak a half-initialized child.
    async fn spawn_and_initialize(
        cli_path: &PathBuf,
        github_token: Option<&str>,
    ) -> Result<Self, RunnerError> {
        let mut child = spawn_copilot(cli_path, github_token)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture copilot stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RunnerError::internal("Failed to capture copilot stdout"))?;
        let mut transport = AcpTransport::new(stdin, stdout);

        let init_outcome = time::timeout(acp_session_timeout(), async {
            info!("ACP: sending initialize handshake");
            let init_id = transport
                .send_request(
                    "initialize",
                    json!({
                        "protocolVersion": 1,
                        "clientInfo": {
                            "name": "embacle",
                            "version": env!("CARGO_PKG_VERSION"),
                        },
                        "capabilities": {},
                    }),
                )
                .await?;
            let init_resp = transport.read_response(init_id).await?;
            info!("ACP: initialize handshake complete");
            debug!(response = %init_resp, "ACP initialize response");
            Ok::<_, RunnerError>(())
        })
        .await;

        match init_outcome {
            Ok(Ok(())) => Ok(Self { child, transport }),
            Ok(Err(e)) => {
                let stderr_output = collect_stderr(&mut child).await;
                warn!(error = %e, stderr = %stderr_output, "ACP initialize failed");
                let _ = child.kill().await;
                Err(e)
            }
            Err(_elapsed) => {
                let stderr_output = collect_stderr(&mut child).await;
                warn!(
                    stderr = %stderr_output,
                    timeout_secs = acp_session_timeout().as_secs(),
                    "ACP initialize timed out — copilot process may be hung (auth issue?)"
                );
                let _ = child.kill().await;
                Err(RunnerError::timeout(format!(
                    "copilot-acp: initialize timed out after {}s (check copilot auth)",
                    acp_session_timeout().as_secs()
                )))
            }
        }
    }

    /// Create a fresh ACP session on the already-initialized subprocess.
    ///
    /// Cheap relative to [`Self::spawn_and_initialize`]: no new subprocess,
    /// no new GitHub→Copilot token exchange — copilot reuses the in-process
    /// token cache it built during `initialize`.
    async fn new_session(
        &mut self,
        model: &str,
        system_prompt: Option<&str>,
        mcp_servers: &[McpServerConfig],
    ) -> Result<String, RunnerError> {
        let outcome = time::timeout(acp_session_timeout(), async {
            let mut session_params = json!({
                "model": model,
                "cwd": env::current_dir()
                    .map_err(|e| RunnerError::internal(format!("Failed to get cwd: {e}")))?,
                "mcpServers": mcp_servers_to_acp_json(mcp_servers),
            });
            if let Some(sys) = system_prompt {
                session_params["systemPrompt"] = Value::String(sys.to_owned());
            }
            info!(
                model = %model,
                has_system_prompt = system_prompt.is_some(),
                mcp_servers = mcp_servers.len(),
                "ACP: creating session"
            );
            let req_id = self
                .transport
                .send_request("session/new", session_params)
                .await?;
            let resp = self.transport.read_response(req_id).await?;
            let session_id = resp
                .get("sessionId")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    RunnerError::external_service("copilot-acp", "Missing sessionId in response")
                })?
                .to_owned();
            info!(session_id = %session_id, model = %model, "ACP session created");
            Ok::<_, RunnerError>(session_id)
        })
        .await;

        match outcome {
            Ok(Ok(session_id)) => Ok(session_id),
            Ok(Err(e)) => Err(e),
            Err(_elapsed) => Err(RunnerError::timeout(format!(
                "copilot-acp: session/new timed out after {}s",
                acp_session_timeout().as_secs()
            ))),
        }
    }

    /// Returns true if the subprocess has not yet exited.
    ///
    /// Uses `try_wait` (non-blocking) so calling this on a healthy
    /// subprocess returns immediately without changing its state.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

/// Collect any available stderr output from the child process (non-blocking, best-effort).
async fn collect_stderr(child: &mut Child) -> String {
    let Some(stderr) = child.stderr.take() else {
        return "(no stderr captured)".to_owned();
    };
    let mut reader = BufReader::new(stderr);
    let mut output = String::new();
    // Read up to 4KB of stderr, with a short timeout
    let result = time::timeout(Duration::from_secs(1), async {
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => output.push_str(&buf),
            }
            if output.len() > 4096 {
                output.push_str("...(truncated)");
                break;
            }
        }
    })
    .await;
    if result.is_err() && output.is_empty() {
        return "(stderr read timed out — process may still be running)".to_owned();
    }
    if output.is_empty() {
        "(empty)".to_owned()
    } else {
        output
    }
}

// ---------------------------------------------------------------------------
// Notification and permission handling
// ---------------------------------------------------------------------------

/// Accumulated state from ACP session notifications during a prompt turn.
struct TurnAccumulator {
    content: String,
    tool_calls: Vec<ObservedToolCall>,
}

impl TurnAccumulator {
    const fn new() -> Self {
        Self {
            content: String::new(),
            tool_calls: Vec::new(),
        }
    }
}

/// Process a session/update notification, accumulating content and tool calls.
fn process_notification(params: &Value, acc: &mut TurnAccumulator) {
    process_notification_inner(params, acc, None);
}

/// Process a session/update notification with optional streaming sink.
///
/// When `event_tx` is `Some`, every observed text delta and tool-call
/// observation is also forwarded to the channel as a [`HeadlessStreamEvent`].
/// When `None`, behaves identically to [`process_notification`].
fn process_notification_streaming(
    params: &Value,
    acc: &mut TurnAccumulator,
    event_tx: &mpsc::UnboundedSender<Result<HeadlessStreamEvent, RunnerError>>,
) {
    process_notification_inner(params, acc, Some(event_tx));
}

fn process_notification_inner(
    params: &Value,
    acc: &mut TurnAccumulator,
    event_tx: Option<&mpsc::UnboundedSender<Result<HeadlessStreamEvent, RunnerError>>>,
) {
    let Some(params) = params.get("params").or(Some(params)) else {
        return;
    };

    let Ok(notif) = serde_json::from_value::<schema::SessionNotification>(params.clone()) else {
        return;
    };

    match &notif.update {
        schema::SessionUpdate::AgentMessageChunk(chunk) => {
            if let schema::ContentBlock::Text(text) = &chunk.content {
                acc.content.push_str(&text.text);
                if let Some(tx) = event_tx {
                    let _ = tx.send(Ok(HeadlessStreamEvent::TextDelta(text.text.clone())));
                }
            }
        }
        schema::SessionUpdate::ToolCall(tc) => {
            let observed = ObservedToolCall {
                id: tc.tool_call_id.0.to_string(),
                title: tc.title.clone(),
                status: format!("{:?}", tc.status),
            };
            acc.tool_calls.push(observed.clone());
            if let Some(tx) = event_tx {
                let _ = tx.send(Ok(HeadlessStreamEvent::ToolCall(observed)));
            }
        }
        schema::SessionUpdate::ToolCallUpdate(update) => {
            let update_id = update.tool_call_id.0.to_string();
            if let Some(existing) = acc.tool_calls.iter_mut().find(|t| t.id == update_id) {
                if let Some(ref title) = update.fields.title {
                    existing.title.clone_from(title);
                }
                if let Some(ref status) = update.fields.status {
                    existing.status = format!("{status:?}");
                }
                if let Some(tx) = event_tx {
                    let _ = tx.send(Ok(HeadlessStreamEvent::ToolCall(existing.clone())));
                }
            }
        }
        _ => {}
    }
}

/// Build a permission response based on the configured policy.
///
/// With `AutoApprove`: selects `AllowAlways` over `AllowOnce`. If no allow option
/// exists, cancels the request instead of falling back to a reject option.
/// With `DenyAll`: always cancels the request.
fn build_permission_response(params: &Value, policy: PermissionPolicy) -> Value {
    if policy == PermissionPolicy::DenyAll {
        debug!("Permission policy is DenyAll, cancelling");
        return json!({ "outcome": "cancelled" });
    }

    let Ok(req) = serde_json::from_value::<schema::RequestPermissionRequest>(params.clone()) else {
        warn!("Failed to parse permission request, cancelling");
        return json!({ "outcome": "cancelled" });
    };

    // Prefer AllowAlways over AllowOnce for fewer repeated prompts
    let option_id = req
        .options
        .iter()
        .find(|o| matches!(o.kind, schema::PermissionOptionKind::AllowAlways))
        .or_else(|| {
            req.options
                .iter()
                .find(|o| matches!(o.kind, schema::PermissionOptionKind::AllowOnce))
        })
        .map(|o| &o.option_id);

    option_id.map_or_else(
        || {
            warn!("Permission request had no allow options, cancelling");
            json!({ "outcome": "cancelled" })
        },
        |id| {
            debug!(?id, "Auto-approving permission request");
            json!({ "outcome": { "optionId": id.0 } })
        },
    )
}

/// Extract token usage from the prompt response JSON.
///
/// ACP returns usage at `/result/usage` with camelCase fields:
/// `totalTokens`, `inputTokens`, `outputTokens`.
fn extract_usage(result: &Value) -> Option<TokenUsage> {
    let usage = result
        .pointer("/result/usage")
        .or_else(|| result.get("usage"))?;

    let input = usage.get("inputTokens").and_then(Value::as_u64)?;
    let output = usage.get("outputTokens").and_then(Value::as_u64)?;
    let total = usage
        .get("totalTokens")
        .and_then(Value::as_u64)
        .unwrap_or(input + output);

    #[allow(clippy::cast_possible_truncation)]
    Some(TokenUsage {
        prompt_tokens: input as u32,
        completion_tokens: output as u32,
        total_tokens: total as u32,
    })
}

fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "max_tokens" => "length",
        "max_turn_requests" => "max_turns",
        "refusal" => "refusal",
        "cancelled" => "cancelled",
        _ => "stop",
    }
}

// ---------------------------------------------------------------------------
// Message collection loops
// ---------------------------------------------------------------------------

/// Read messages until prompt completes, collecting all content.
async fn collect_complete(
    transport: &mut AcpTransport,
    prompt_id: i64,
    model: String,
    policy: PermissionPolicy,
) -> Result<(ChatResponse, Vec<ObservedToolCall>), RunnerError> {
    let mut acc = TurnAccumulator::new();
    let mut message_count: u32 = 0;

    loop {
        let msg = transport.read_message().await?;
        message_count += 1;

        if message_count == 1 {
            info!("ACP: receiving first message from copilot");
        }
        // Log method notifications for visibility (every 10th to avoid spam)
        if let Some(method) = msg.get("method").and_then(Value::as_str) {
            if message_count <= 5 || message_count.is_multiple_of(10) {
                debug!(method, message_count, "ACP notification received");
            }
        }

        // Prompt response — the turn is complete
        if msg.get("id").and_then(Value::as_i64) == Some(prompt_id) {
            if let Some(error) = msg.get("error") {
                return Err(RunnerError::external_service(
                    "copilot-acp",
                    format!("Prompt failed: {error}"),
                ));
            }

            let stop_reason = msg
                .pointer("/result/stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn");

            let usage = extract_usage(&msg);

            debug!(
                content_len = acc.content.len(),
                tool_calls = acc.tool_calls.len(),
                model = %model,
                has_usage = usage.is_some(),
                "Copilot Headless complete() response"
            );

            let response = ChatResponse {
                content: acc.content,
                model,
                usage,
                finish_reason: Some(map_stop_reason(stop_reason).to_owned()),
                warnings: None,
                tool_calls: None,
            };

            return Ok((response, acc.tool_calls));
        }

        // Server requests and notifications
        handle_server_message(&msg, transport, &mut acc, policy).await?;
    }
}

/// Read messages until prompt completes, streaming chunks via channel.
async fn collect_streaming(
    transport: &mut AcpTransport,
    prompt_id: i64,
    chunk_tx: &mpsc::UnboundedSender<Result<StreamChunk, RunnerError>>,
    policy: PermissionPolicy,
) -> Result<(), RunnerError> {
    let mut acc = TurnAccumulator::new();

    loop {
        let msg = transport.read_message().await?;

        // Prompt response — the turn is complete
        if msg.get("id").and_then(Value::as_i64) == Some(prompt_id) {
            if let Some(error) = msg.get("error") {
                return Err(RunnerError::external_service(
                    "copilot-acp",
                    format!("Prompt failed: {error}"),
                ));
            }

            let stop_reason = msg
                .pointer("/result/stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn");

            let _ = chunk_tx.send(Ok(StreamChunk {
                delta: String::new(),
                is_final: true,
                finish_reason: Some(map_stop_reason(stop_reason).to_owned()),
            }));

            return Ok(());
        }

        // Server requests and notifications
        if let Some(method) = msg.get("method").and_then(Value::as_str) {
            match method {
                "session/update" => {
                    if let Some(params) = msg.get("params") {
                        // Try to extract text delta for streaming
                        if let Ok(notif) =
                            serde_json::from_value::<schema::SessionNotification>(params.clone())
                        {
                            if let schema::SessionUpdate::AgentMessageChunk(chunk) = &notif.update {
                                if let schema::ContentBlock::Text(text) = &chunk.content {
                                    let _ = chunk_tx.send(Ok(StreamChunk {
                                        delta: text.text.clone(),
                                        is_final: false,
                                        finish_reason: None,
                                    }));
                                }
                            }
                        }
                        // Also track tool calls for internal accounting
                        process_notification(params, &mut acc);
                    }
                }
                "session/request_permission" => {
                    if let (Some(id), Some(params)) = (msg.get("id"), msg.get("params")) {
                        let response = build_permission_response(params, policy);
                        transport.send_response(id, response).await?;
                    }
                }
                _ => {}
            }
        }
    }
}

/// Handle a server-to-client message (notification or request).
async fn handle_server_message(
    msg: &Value,
    transport: &mut AcpTransport,
    acc: &mut TurnAccumulator,
    policy: PermissionPolicy,
) -> Result<(), RunnerError> {
    if let Some(method) = msg.get("method").and_then(Value::as_str) {
        match method {
            "session/update" => {
                if let Some(params) = msg.get("params") {
                    process_notification(params, acc);
                }
            }
            "session/request_permission" => {
                if let (Some(id), Some(params)) = (msg.get("id"), msg.get("params")) {
                    let response = build_permission_response(params, policy);
                    transport.send_response(id, response).await?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Streaming variant of [`handle_server_message`].
///
/// Identical to the non-streaming form except that every observed
/// content/tool-call notification is also forwarded to `event_tx` as a
/// [`HeadlessStreamEvent`]. Permission-request handling is unchanged.
async fn handle_server_message_streaming(
    msg: &Value,
    transport: &mut AcpTransport,
    acc: &mut TurnAccumulator,
    policy: PermissionPolicy,
    event_tx: &mpsc::UnboundedSender<Result<HeadlessStreamEvent, RunnerError>>,
) -> Result<(), RunnerError> {
    if let Some(method) = msg.get("method").and_then(Value::as_str) {
        match method {
            "session/update" => {
                if let Some(params) = msg.get("params") {
                    process_notification_streaming(params, acc, event_tx);
                }
            }
            "session/request_permission" => {
                if let (Some(id), Some(params)) = (msg.get("id"), msg.get("params")) {
                    let response = build_permission_response(params, policy);
                    transport.send_response(id, response).await?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Read messages until prompt completes, emitting [`HeadlessStreamEvent`]s
/// as ACP notifications arrive while accumulating the same final state
/// that [`collect_complete`] produces.
///
/// On success returns the aggregated [`HeadlessToolResponse`] so the
/// caller can emit a final [`HeadlessStreamEvent::Done`] event.
async fn collect_streaming_with_tools(
    transport: &mut AcpTransport,
    prompt_id: i64,
    model: String,
    policy: PermissionPolicy,
    event_tx: &mpsc::UnboundedSender<Result<HeadlessStreamEvent, RunnerError>>,
) -> Result<HeadlessToolResponse, RunnerError> {
    let mut acc = TurnAccumulator::new();

    loop {
        let msg = transport.read_message().await?;

        // Prompt response — the turn is complete
        if msg.get("id").and_then(Value::as_i64) == Some(prompt_id) {
            if let Some(error) = msg.get("error") {
                return Err(RunnerError::external_service(
                    "copilot-acp",
                    format!("Prompt failed: {error}"),
                ));
            }

            let stop_reason = msg
                .pointer("/result/stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn");
            let usage = extract_usage(&msg);

            return Ok(HeadlessToolResponse {
                content: acc.content,
                model,
                tool_calls: acc.tool_calls,
                usage,
                finish_reason: Some(map_stop_reason(stop_reason).to_owned()),
            });
        }

        handle_server_message_streaming(&msg, transport, &mut acc, policy, event_tx).await?;
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A tool call observed during an ACP session turn.
#[derive(Debug, Clone)]
pub struct ObservedToolCall {
    /// Tool call ID from the ACP protocol.
    pub id: String,
    /// Human-readable title describing the tool action.
    pub title: String,
    /// Execution status (e.g., "Pending", "`InProgress`", "Completed", "Failed").
    pub status: String,
}

/// Response from a headless conversation turn including tool execution metadata.
#[derive(Debug, Clone)]
pub struct HeadlessToolResponse {
    /// Final assistant response content.
    pub content: String,
    /// Model that generated the response.
    pub model: String,
    /// Tool calls observed during the turn.
    pub tool_calls: Vec<ObservedToolCall>,
    /// Token usage for this turn.
    pub usage: Option<TokenUsage>,
    /// Finish reason.
    pub finish_reason: Option<String>,
}

/// Event emitted by [`CopilotHeadlessRunner::converse_stream`] as the ACP
/// turn progresses.
///
/// Unlike [`StreamChunk`] (which only carries text deltas), this enum
/// surfaces tool-call observations alongside text — keeping the rich
/// metadata that [`CopilotHeadlessRunner::converse`] returns at the end
/// of the turn while delivering it incrementally.
#[derive(Debug, Clone)]
pub enum HeadlessStreamEvent {
    /// Partial assistant text — the next chunk to append to the
    /// in-flight assistant message.
    TextDelta(String),
    /// A tool call was observed (start or status update). Each event
    /// is a snapshot of the tool call's latest known state, so a
    /// consumer can either accumulate updates or replace by id.
    ToolCall(ObservedToolCall),
    /// The turn has finished. Carries the aggregated
    /// [`HeadlessToolResponse`] — same shape that
    /// [`CopilotHeadlessRunner::converse`] would have returned.
    /// Always emitted as the last event before the stream closes
    /// successfully.
    Done(HeadlessToolResponse),
}

/// Stream of [`HeadlessStreamEvent`]s for a single converse turn.
pub type HeadlessEventStream =
    Pin<Box<dyn Stream<Item = Result<HeadlessStreamEvent, RunnerError>> + Send>>;

// ---------------------------------------------------------------------------
// Public runner
// ---------------------------------------------------------------------------

/// GitHub Copilot Headless (ACP) LLM provider.
///
/// Communicates with `copilot --acp` via the Agent Client Protocol (JSON-RPC over stdio).
/// Spawns a new copilot subprocess per request using NDJSON framing.
/// Uses types from `agent-client-protocol-schema` for protocol message deserialization.
///
/// Copilot manages its own tool execution internally (GitHub tools, code search),
/// but cannot execute external MCP tools. Tool calls are observed and reported
/// via [`HeadlessToolResponse`] from [`converse()`](Self::converse).
/// For custom tools, callers should use text-based tool calling (CLI tool loop).
pub struct CopilotHeadlessRunner {
    config: CopilotHeadlessConfig,
    available_models: Vec<String>,
    /// Long-lived `copilot --acp` subprocess + initialized transport.
    ///
    /// Lazily spawned on the first `complete()` call and kept warm across
    /// calls so the GitHub→Copilot OAuth token exchange amortizes across
    /// the subprocess lifetime instead of running per request. Cleared and
    /// respawned on subprocess death or any complete()-path error so the
    /// next call always starts from a known-good state.
    ///
    /// Wrapped in `tokio::sync::Mutex` because the NDJSON transport is
    /// request/response per JSON-RPC id and cannot interleave concurrent
    /// prompts without response routing — which the ACP wire format does
    /// not support.
    process: Arc<TokioMutex<Option<AcpProcess>>>,
}

impl CopilotHeadlessRunner {
    /// Create a new provider from environment configuration.
    ///
    /// The set of available models is taken from the ranked catalog in
    /// [`crate::copilot_models`]. Availability per account is resolved lazily
    /// by the Copilot CLI runner's self-heal loop, not at construction time.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            config: CopilotHeadlessConfig::from_env(),
            available_models: catalog_ids(),
            process: Arc::new(TokioMutex::new(None)),
        }
    }

    /// Create a new provider with explicit configuration.
    #[must_use]
    pub fn with_config(config: CopilotHeadlessConfig) -> Self {
        Self {
            config,
            available_models: catalog_ids(),
            process: Arc::new(TokioMutex::new(None)),
        }
    }

    /// Resolve the copilot CLI binary path.
    fn resolve_cli_path(&self) -> Result<PathBuf, RunnerError> {
        if let Some(ref path) = self.config.cli_path {
            return Ok(path.clone());
        }
        which::which("copilot").map_err(|_| RunnerError::binary_not_found("copilot"))
    }

    /// Resolve the model to use for a request.
    /// Provider-alias names (`copilot_headless`, `copilot`) are mapped to the
    /// configured default model because they are not valid Copilot model identifiers.
    /// Any other model name (e.g. `gpt-4.1`, `claude-opus-4.7`) is passed through.
    fn resolve_model(&self, requested: Option<&str>) -> String {
        match requested {
            Some(m) if m != "copilot_headless" && m != "copilot" => m.to_owned(),
            _ => self.config.model.clone(),
        }
    }

    /// Build ACP prompt content blocks from the conversation messages.
    ///
    /// ACP creates a fresh session per request with no built-in multi-turn memory.
    /// To provide conversation continuity, prior user/assistant exchanges are
    /// serialized into a `<conversation-history>` block prepended to the prompt.
    ///
    /// The system prompt is passed via ACP `session/new` `systemPrompt` and also
    /// prepended as plain text to the prompt so the model reliably sees it.
    /// Set `inject_system_in_prompt` to `false` to skip the prompt-text injection.
    ///
    /// The number of history messages is capped by `max_history_turns` from
    /// [`CopilotHeadlessConfig`]. Only the most recent turns are kept.
    ///
    /// Always includes a text block. When the last user message has images,
    /// appends image blocks with `type: "image"`, `data`, and `mimeType`.
    fn build_prompt_blocks(&self, request: &ChatRequest) -> Vec<Value> {
        let system = if self.config.inject_system_in_prompt {
            Self::extract_system_prompt(request)
        } else {
            None
        };
        let max_turns = self.config.max_history_turns;

        // Separate non-system messages into history (all but last user) + last user
        let non_system: Vec<&ChatMessage> = request
            .messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .collect();

        let (history, last_user) = if non_system.is_empty() {
            (Vec::new(), None)
        } else {
            let last_idx = non_system.iter().rposition(|m| m.role == MessageRole::User);
            match last_idx {
                Some(idx) => {
                    let hist = non_system[..idx].to_vec();
                    (hist, Some(non_system[idx]))
                }
                None => (non_system, None),
            }
        };

        let user_text = last_user.map(|m| m.content.as_str()).unwrap_or_default();

        // Apply max_history_turns limit — keep only the most recent turns
        let truncated_history = if max_turns == 0 || history.is_empty() {
            &[][..]
        } else if history.len() > max_turns {
            &history[history.len() - max_turns..]
        } else {
            &history
        };

        // Serialize prior turns into a conversation history block
        let history_block = if truncated_history.is_empty() {
            String::new()
        } else {
            let mut buf = String::from("<conversation-history>\n");
            for msg in truncated_history {
                let role_label = match msg.role {
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::Tool => "Tool",
                    MessageRole::System => continue,
                };
                buf.push_str(role_label);
                buf.push_str(": ");
                buf.push_str(&msg.content);
                buf.push('\n');
            }
            buf.push_str("</conversation-history>\n\n");
            buf
        };

        // Assemble: system prompt + conversation history + current user message
        let mut text = String::new();
        if let Some(sys) = system {
            text.push_str(sys);
            text.push_str("\n\n");
        }
        text.push_str(&history_block);
        text.push_str(user_text);

        let mut blocks = vec![json!({"type": "text", "text": text})];

        if let Some(images) = last_user.and_then(|m| m.images.as_ref()) {
            for img in images {
                blocks.push(json!({
                    "type": "image",
                    "data": img.data,
                    "mimeType": img.mime_type,
                }));
            }
        }

        blocks
    }

    /// Extract the system prompt if present.
    fn extract_system_prompt(request: &ChatRequest) -> Option<&str> {
        request
            .messages
            .iter()
            .find(|m| m.role == MessageRole::System)
            .map(|m| m.content.as_str())
    }

    /// Run a conversation turn and return detailed results including tool call metadata.
    ///
    /// Unlike [`complete()`](LlmProvider::complete), this returns an [`HeadlessToolResponse`]
    /// with observed tool calls that copilot executed internally during the turn.
    pub async fn converse(
        &self,
        request: &ChatRequest,
    ) -> Result<HeadlessToolResponse, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = self.resolve_model(request.model.as_deref());
        let system_prompt = Self::extract_system_prompt(request);
        let prompt_blocks = self.build_prompt_blocks(request);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            &model,
            system_prompt,
            &request.mcp_servers,
        )
        .await?;

        info!(session_id = %session_id, "ACP: sending prompt");
        let prompt_id = transport
            .send_request(
                "session/prompt",
                build_prompt_params(&session_id, &prompt_blocks, request.max_tokens),
            )
            .await?;

        let result = time::timeout(
            acp_prompt_timeout(),
            collect_complete(
                &mut transport,
                prompt_id,
                model,
                self.config.permission_policy,
            ),
        )
        .await;

        match &result {
            Ok(Ok((response, tool_calls))) => {
                info!(
                    content_len = response.content.len(),
                    tool_calls = tool_calls.len(),
                    "ACP converse completed successfully"
                );
            }
            Ok(Err(e)) => {
                let stderr_output = collect_stderr(&mut child).await;
                warn!(error = %e, stderr = %stderr_output, "ACP converse failed");
            }
            Err(_) => {
                let stderr_output = collect_stderr(&mut child).await;
                warn!(
                    stderr = %stderr_output,
                    timeout_secs = acp_prompt_timeout().as_secs(),
                    "ACP converse timed out"
                );
            }
        }

        let _ = child.kill().await;

        let result = result.map_err(|_| {
            RunnerError::timeout(format!(
                "copilot-acp: prompt timed out after {}s",
                acp_prompt_timeout().as_secs()
            ))
        })?;

        let (response, tool_calls) = result?;
        Ok(HeadlessToolResponse {
            content: response.content,
            model: response.model,
            tool_calls,
            usage: response.usage,
            finish_reason: response.finish_reason,
        })
    }

    /// Streaming variant of [`converse()`](Self::converse).
    ///
    /// Returns a [`HeadlessEventStream`] that yields [`HeadlessStreamEvent`]s
    /// as the ACP turn progresses:
    ///
    /// - [`HeadlessStreamEvent::TextDelta`] — partial assistant text as
    ///   `AgentMessageChunk` notifications arrive.
    /// - [`HeadlessStreamEvent::ToolCall`] — every observed tool call (and
    ///   subsequent status updates), letting the consumer surface "calling
    ///   tool X..." progress to end users while the turn is still running.
    /// - [`HeadlessStreamEvent::Done`] — the final aggregated response,
    ///   identical in shape to what [`converse()`](Self::converse) would
    ///   return. Always emitted last on success.
    ///
    /// The underlying ACP session runs in a background tokio task that
    /// owns the spawned `copilot --acp` child process; it is killed when
    /// the turn completes (success, error, or timeout). Dropping the
    /// returned stream early does **not** abort the in-flight turn — the
    /// task will still drain the session and the child will be cleaned up
    /// when the turn finishes.
    ///
    /// # Errors
    ///
    /// Returns a setup-time error before any events are emitted if the
    /// CLI cannot be located or the ACP handshake fails. After the stream
    /// starts, transport / protocol failures arrive as `Err` items in the
    /// stream itself, and the configured prompt timeout
    /// (`EMBACLE_ACP_PROMPT_TIMEOUT_SECS`, default 5 min) becomes a
    /// terminal `RunnerError::Timeout` event.
    #[instrument(skip(self, request), fields(model = field::Empty))]
    pub async fn converse_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<HeadlessEventStream, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = self.resolve_model(request.model.as_deref());
        Span::current().record("model", field::display(&model));
        let system_prompt = Self::extract_system_prompt(request);
        let prompt_blocks = self.build_prompt_blocks(request);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            &model,
            system_prompt,
            &request.mcp_servers,
        )
        .await?;

        info!(session_id = %session_id, "ACP: sending streaming prompt");
        let prompt_id = transport
            .send_request(
                "session/prompt",
                build_prompt_params(&session_id, &prompt_blocks, request.max_tokens),
            )
            .await?;

        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let policy = self.config.permission_policy;
        let timeout = acp_prompt_timeout();
        let model_for_task = model.clone();

        // Drive the ACP session and emit events on a background task so
        // the caller can consume the stream incrementally. The task owns
        // the transport and child, and kills the child when it finishes
        // — matching the lifecycle of `converse()`.
        tokio::spawn(async move {
            let result = time::timeout(
                timeout,
                collect_streaming_with_tools(
                    &mut transport,
                    prompt_id,
                    model_for_task,
                    policy,
                    &event_tx,
                ),
            )
            .await;

            match result {
                Ok(Ok(response)) => {
                    info!(
                        content_len = response.content.len(),
                        tool_calls = response.tool_calls.len(),
                        "ACP converse_stream completed successfully"
                    );
                    let _ = event_tx.send(Ok(HeadlessStreamEvent::Done(response)));
                }
                Ok(Err(e)) => {
                    let stderr_output = collect_stderr(&mut child).await;
                    warn!(error = %e, stderr = %stderr_output, "ACP converse_stream failed");
                    let _ = event_tx.send(Err(e));
                }
                Err(_) => {
                    let stderr_output = collect_stderr(&mut child).await;
                    warn!(
                        stderr = %stderr_output,
                        timeout_secs = timeout.as_secs(),
                        "ACP converse_stream timed out"
                    );
                    let _ = event_tx.send(Err(RunnerError::timeout(format!(
                        "copilot-acp: prompt timed out after {}s",
                        timeout.as_secs()
                    ))));
                }
            }

            let _ = child.kill().await;
        });

        let stream = UnboundedReceiverStream::new(event_rx);
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl LlmProvider for CopilotHeadlessRunner {
    fn name(&self) -> &'static str {
        "copilot_headless"
    }

    fn display_name(&self) -> &str {
        "GitHub Copilot (Headless)"
    }

    fn capabilities(&self) -> LlmCapabilities {
        let base =
            LlmCapabilities::STREAMING | LlmCapabilities::SYSTEM_MESSAGES | LlmCapabilities::VISION;
        // SDK_TOOL_CALLING is opt-in via `mcp_tool_calling`. When set, the
        // caller passes `mcp_servers` per request and Copilot calls those tools
        // natively over ACP — the CLI advertises `mcpCapabilities {http,sse}` at
        // initialize, so it CAN execute external MCP tools. When unset, callers
        // fall through to text-based tool calling, where the host parses
        // <tool_call> blocks and executes tools itself.
        if self.config.mcp_tool_calling {
            base | LlmCapabilities::SDK_TOOL_CALLING
        } else {
            base
        }
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip_all, fields(runner = "copilot_headless", model = field::Empty))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = self.resolve_model(request.model.as_deref());
        Span::current().record("model", field::display(&model));
        let system_prompt = Self::extract_system_prompt(request);
        let prompt_blocks = self.build_prompt_blocks(request);

        // Hold the process lock for the full RPC round-trip. ACP transport
        // is request/response by JSON-RPC id with no response-routing in
        // the wire format, so concurrent prompts would interleave reads
        // unsafely. The probe is every 5 minutes and chat traffic is
        // sequential per chat turn, so contention is in practice low.
        let mut guard = self.process.lock().await;

        // Detect a subprocess that exited between calls (crash, OOM kill,
        // upstream disconnect) and drop it so we respawn below.
        if let Some(p) = guard.as_mut() {
            if !p.is_alive() {
                warn!("ACP subprocess exited between calls; respawning");
                *guard = None;
            }
        }

        let process = if let Some(existing) = guard.as_mut() {
            existing
        } else {
            let fresh =
                AcpProcess::spawn_and_initialize(&cli_path, self.config.github_token.as_deref())
                    .await?;
            guard.insert(fresh)
        };

        let session_id = match process
            .new_session(&model, system_prompt, &request.mcp_servers)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                // session/new failed on a previously-healthy subprocess.
                // Could be the cached Copilot OAuth token expired and the
                // CLI didn't auto-refresh, or the process is wedged. Kill
                // and clear so the next call respawns and re-authenticates.
                let _ = process.child.kill().await;
                *guard = None;
                return Err(e);
            }
        };

        info!(
            session_id = %session_id,
            message_count = request.messages.len(),
            prompt_blocks = prompt_blocks.len(),
            "ACP complete: sending prompt"
        );
        if tracing::enabled!(tracing::Level::TRACE) {
            match serde_json::to_string(&prompt_blocks) {
                Ok(blocks_json) => trace!(prompt_blocks = %blocks_json, "ACP prompt blocks"),
                Err(e) => trace!(error = %e, "ACP prompt blocks serialization failed"),
            }
        }
        let prompt_id = match process
            .transport
            .send_request(
                "session/prompt",
                build_prompt_params(&session_id, &prompt_blocks, request.max_tokens),
            )
            .await
        {
            Ok(id) => id,
            Err(e) => {
                // Writing to stdin failed — pipe is broken. Discard.
                let _ = process.child.kill().await;
                *guard = None;
                return Err(e);
            }
        };

        let started = Instant::now();
        let result = time::timeout(
            acp_prompt_timeout(),
            collect_complete(
                &mut process.transport,
                prompt_id,
                model,
                self.config.permission_policy,
            ),
        )
        .await;

        match result {
            Ok(Ok((response, _tool_calls))) => {
                info!(
                    latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    content_len = response.content.len(),
                    tool_calls = response.tool_calls.as_ref().map_or(0, Vec::len),
                    finish_reason = response.finish_reason.as_deref().unwrap_or("none"),
                    "ACP complete: response received"
                );
                if tracing::enabled!(tracing::Level::TRACE) {
                    trace!(content = %response.content, "ACP response body");
                }
                // SUCCESS path: leave the subprocess alive so the next call
                // reuses it. This is the whole point of the pool.
                Ok(response)
            }
            Ok(Err(e)) => {
                // Prompt failed mid-stream. Transport may be desynced; the
                // conservative choice is to kill and respawn rather than
                // risk a corrupt session bleeding into the next call.
                let _ = process.child.kill().await;
                *guard = None;
                Err(e)
            }
            Err(_elapsed) => {
                let stderr_output = collect_stderr(&mut process.child).await;
                warn!(stderr = %stderr_output, "ACP complete timed out");
                let _ = process.child.kill().await;
                *guard = None;
                Err(RunnerError::timeout(format!(
                    "copilot-acp: prompt timed out after {}s",
                    acp_prompt_timeout().as_secs()
                )))
            }
        }
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = self.resolve_model(request.model.as_deref());
        let system_prompt = Self::extract_system_prompt(request).map(str::to_owned);
        let prompt_blocks = self.build_prompt_blocks(request);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            &model,
            system_prompt.as_deref(),
            &request.mcp_servers,
        )
        .await?;

        let prompt_id = transport
            .send_request(
                "session/prompt",
                build_prompt_params(&session_id, &prompt_blocks, request.max_tokens),
            )
            .await?;

        let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();
        let policy = self.config.permission_policy;

        tokio::spawn(async move {
            let result = time::timeout(
                acp_prompt_timeout(),
                collect_streaming(&mut transport, prompt_id, &chunk_tx, policy),
            )
            .await;
            match result {
                Ok(Err(e)) => {
                    let _ = chunk_tx.send(Err(e));
                }
                Err(_) => {
                    let _ = chunk_tx.send(Err(RunnerError::timeout(format!(
                        "copilot-acp: prompt timed out after {}s",
                        acp_prompt_timeout().as_secs()
                    ))));
                }
                Ok(Ok(())) => {}
            }
            let _ = child.kill().await;
        });

        let stream = UnboundedReceiverStream::new(chunk_rx);
        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.resolve_cli_path().map_or(Ok(false), |path| {
            tracing::info!(cli_path = %path.display(), "Copilot Headless health check: binary found");
            Ok(true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use serde_json::json;

    /// Build a valid ACP permission request JSON with the given option kinds.
    ///
    /// Uses camelCase field names matching the `agent-client-protocol-schema` serde config.
    /// `PermissionOptionKind` uses `snake_case`: `allow_once`, `allow_always`,
    /// `reject_once`, `reject_always`.
    fn make_permission_params(kinds: &[&str]) -> Value {
        let options: Vec<Value> = kinds
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                json!({
                    "optionId": format!("opt_{i}"),
                    "name": format!("Option {i}"),
                    "kind": kind
                })
            })
            .collect();
        json!({
            "sessionId": "test-session",
            "toolCall": {
                "toolCallId": "tc_1"
            },
            "options": options
        })
    }

    #[test]
    fn permission_only_reject_options_cancels() {
        let params = make_permission_params(&["reject_once", "reject_always"]);
        let result = build_permission_response(&params, PermissionPolicy::AutoApprove);
        assert_eq!(result["outcome"], "cancelled");
    }

    #[test]
    fn permission_prefers_allow_always_over_allow_once() {
        let params = make_permission_params(&["allow_once", "allow_always", "reject_once"]);
        let result = build_permission_response(&params, PermissionPolicy::AutoApprove);
        // AllowAlways is at index 1 → opt_1
        let selected_id = result["outcome"]["optionId"].as_str().unwrap(); // Safe: test assertion
        assert_eq!(selected_id, "opt_1");
    }

    #[test]
    fn permission_selects_allow_once_when_no_allow_always() {
        let params = make_permission_params(&["allow_once", "reject_once"]);
        let result = build_permission_response(&params, PermissionPolicy::AutoApprove);
        let selected_id = result["outcome"]["optionId"].as_str().unwrap(); // Safe: test assertion
        assert_eq!(selected_id, "opt_0");
    }

    #[test]
    fn permission_empty_options_cancels() {
        let params = json!({
            "sessionId": "test-session",
            "toolCall": {
                "toolCallId": "tc_1"
            },
            "options": []
        });
        let result = build_permission_response(&params, PermissionPolicy::AutoApprove);
        assert_eq!(result["outcome"], "cancelled");
    }

    #[test]
    fn permission_deny_all_policy_always_cancels() {
        let params = make_permission_params(&["allow_once", "allow_always"]);
        let result = build_permission_response(&params, PermissionPolicy::DenyAll);
        assert_eq!(result["outcome"], "cancelled");
    }

    /// Create a test runner with configurable `max_history_turns`.
    fn test_runner(max_history_turns: usize) -> CopilotHeadlessRunner {
        CopilotHeadlessRunner {
            config: CopilotHeadlessConfig {
                max_history_turns,
                ..CopilotHeadlessConfig::default()
            },
            available_models: vec![],
            process: Arc::new(TokioMutex::new(None)),
        }
    }

    /// Create a test runner with system prompt injection disabled.
    fn test_runner_no_system_injection(max_history_turns: usize) -> CopilotHeadlessRunner {
        CopilotHeadlessRunner {
            config: CopilotHeadlessConfig {
                max_history_turns,
                inject_system_in_prompt: false,
                ..CopilotHeadlessConfig::default()
            },
            available_models: vec![],
            process: Arc::new(TokioMutex::new(None)),
        }
    }

    #[test]
    fn build_prompt_blocks_text_only_no_system() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![ChatMessage::user("Hello")]);
        let blocks = runner.build_prompt_blocks(&request);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Hello");
    }

    #[test]
    fn build_prompt_blocks_injects_system_prompt_as_plain_text() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("You are a fitness assistant"),
            ChatMessage::user("Hello"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
                                                        // System prompt injected as plain text — no XML tags
        assert!(text.contains("You are a fitness assistant"));
        assert!(!text.contains("<system-instructions>"));
        assert!(text.contains("Hello"));
    }

    #[test]
    fn build_prompt_blocks_skips_system_when_injection_disabled() {
        let runner = test_runner_no_system_injection(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("You are a fitness assistant"),
            ChatMessage::user("Hello"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        assert_eq!(blocks.len(), 1);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
        assert!(!text.contains("You are a fitness assistant"));
        assert!(text.contains("Hello"));
    }

    #[test]
    fn build_prompt_blocks_with_images() {
        use crate::types::ImagePart;

        let runner = test_runner(20);
        let img = ImagePart::new("aGVsbG8=", "image/png").unwrap(); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user_with_images(
            "Describe this image",
            vec![img],
        )]);
        let blocks = runner.build_prompt_blocks(&request);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert!(blocks[0]["text"]
            .as_str()
            .unwrap() // Safe: test assertion
            .contains("Describe this image"));
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["data"], "aGVsbG8=");
        assert_eq!(blocks[1]["mimeType"], "image/png");
    }

    #[test]
    fn build_prompt_blocks_uses_last_user_message() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::user("first"),
            ChatMessage::assistant("response"),
            ChatMessage::user("second"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
        assert!(text.contains("second"));
        // The last user message should NOT be in the history section
        assert!(!text.ends_with("second\n</conversation-history>"));
    }

    #[test]
    fn build_prompt_blocks_includes_conversation_history() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("What is my pace?"),
            ChatMessage::assistant("Your average pace is 5:30/km"),
            ChatMessage::user("And my heart rate?"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // System prompt injected as plain text
        assert!(text.contains("You are helpful"));
        assert!(!text.contains("<system-instructions>"));

        // Conversation history block present with prior turns
        assert!(text.contains("<conversation-history>"));
        assert!(text.contains("User: What is my pace?"));
        assert!(text.contains("Assistant: Your average pace is 5:30/km"));
        assert!(text.contains("</conversation-history>"));

        // Current user message at the end (outside history block)
        assert!(text.contains("And my heart rate?"));
        // Current message should NOT be inside the history block
        assert!(!text.contains("User: And my heart rate?"));
    }

    #[test]
    fn build_prompt_blocks_no_history_for_single_turn() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("Be helpful"),
            ChatMessage::user("Hello"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
                                                        // No history block when there's only one user message
        assert!(!text.contains("<conversation-history>"));
        assert!(text.contains("Hello"));
    }

    #[test]
    fn build_prompt_blocks_truncates_history_to_max_turns() {
        let runner = test_runner(2); // Only keep 2 most recent history messages
        let request = ChatRequest::new(vec![
            ChatMessage::user("msg1"),
            ChatMessage::assistant("reply1"),
            ChatMessage::user("msg2"),
            ChatMessage::assistant("reply2"),
            ChatMessage::user("msg3"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // Only the 2 most recent history messages should be included
        assert!(!text.contains("User: msg1"));
        assert!(!text.contains("Assistant: reply1"));
        assert!(text.contains("User: msg2"));
        assert!(text.contains("Assistant: reply2"));
        assert!(text.contains("msg3")); // Current message
    }

    #[test]
    fn build_prompt_blocks_zero_max_turns_disables_history() {
        let runner = test_runner(0);
        let request = ChatRequest::new(vec![
            ChatMessage::user("first"),
            ChatMessage::assistant("response"),
            ChatMessage::user("second"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
        assert!(!text.contains("<conversation-history>"));
        assert!(text.contains("second"));
    }

    #[test]
    fn build_prompt_blocks_max_turns_one_keeps_last_history_message() {
        let runner = test_runner(1);
        let request = ChatRequest::new(vec![
            ChatMessage::user("msg1"),
            ChatMessage::assistant("reply1"),
            ChatMessage::user("msg2"),
            ChatMessage::assistant("reply2"),
            ChatMessage::user("current"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // Only the single most recent history message (reply2)
        assert!(!text.contains("User: msg1"));
        assert!(!text.contains("reply1"));
        assert!(!text.contains("User: msg2"));
        assert!(text.contains("Assistant: reply2"));
        assert!(text.contains("current"));
    }

    #[test]
    fn build_prompt_blocks_tool_messages_included_in_history() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::user("Check my activities"),
            ChatMessage::tool("get_activities", "call_1", "{\"activities\": []}"),
            ChatMessage::assistant("No activities found"),
            ChatMessage::user("Try again"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        assert!(text.contains("<conversation-history>"));
        assert!(text.contains("User: Check my activities"));
        assert!(text.contains("Tool: "));
        assert!(text.contains("Assistant: No activities found"));
        assert!(text.contains("Try again"));
    }

    #[test]
    fn build_prompt_blocks_empty_messages() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![]);
        let blocks = runner.build_prompt_blocks(&request);
        assert_eq!(blocks.len(), 1);
        // Empty prompt — no crash
        assert_eq!(blocks[0]["text"], "");
    }

    #[test]
    fn build_prompt_blocks_only_system_message() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![ChatMessage::system("Be helpful")]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
                                                        // System prompt as plain text, no XML tags, no history
        assert!(text.contains("Be helpful"));
        assert!(!text.contains("<system-instructions>"));
        assert!(!text.contains("<conversation-history>"));
    }

    #[test]
    fn build_prompt_blocks_long_conversation_keeps_most_recent() {
        let runner = test_runner(4);
        let mut messages = vec![ChatMessage::system("system")];
        for i in 1..=10 {
            messages.push(ChatMessage::user(format!("user_{i}")));
            messages.push(ChatMessage::assistant(format!("reply_{i}")));
        }
        messages.push(ChatMessage::user("current"));
        let request = ChatRequest::new(messages);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // Only last 4 history messages kept (user_9, reply_9, user_10, reply_10)
        assert!(!text.contains("user_8"));
        assert!(!text.contains("reply_8"));
        assert!(text.contains("User: user_9"));
        assert!(text.contains("Assistant: reply_9"));
        assert!(text.contains("User: user_10"));
        assert!(text.contains("Assistant: reply_10"));
        assert!(text.contains("current"));
    }

    #[test]
    fn build_prompt_blocks_preserves_section_ordering() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("sys prompt"),
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("q2"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // Verify ordering: system prompt < conversation-history < current message
        let sys_pos = text.find("sys prompt").unwrap(); // Safe: test assertion
        let hist_pos = text.find("<conversation-history>").unwrap(); // Safe: test assertion
        let current_pos = text.find("q2").unwrap(); // Safe: test assertion
        assert!(sys_pos < hist_pos, "system must come before history");
        assert!(
            hist_pos < current_pos,
            "history must come before current message"
        );
    }

    #[test]
    fn build_prompt_blocks_history_exact_at_max_turns() {
        let runner = test_runner(2);
        // Exactly 2 history messages — should include all, no truncation
        let request = ChatRequest::new(vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("current"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        assert!(text.contains("User: q1"));
        assert!(text.contains("Assistant: a1"));
        assert!(text.contains("current"));
    }

    #[test]
    fn build_prompt_blocks_multiple_system_messages_uses_first() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("first system"),
            ChatMessage::system("second system"),
            ChatMessage::user("hello"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // extract_system_prompt returns the first system message
        assert!(text.contains("first system"));
    }

    #[test]
    fn capabilities_omit_sdk_tool_calling_by_default() {
        let runner = CopilotHeadlessRunner::with_config(CopilotHeadlessConfig::default());
        let caps = runner.capabilities();
        assert!(caps.supports_vision());
        assert!(caps.supports_streaming());
        assert!(
            !caps.supports_sdk_tool_calling(),
            "default config falls through to text-based tool calling"
        );
    }

    #[test]
    fn capabilities_advertise_sdk_tool_calling_when_mcp_enabled() {
        let runner = CopilotHeadlessRunner::with_config(CopilotHeadlessConfig {
            mcp_tool_calling: true,
            ..CopilotHeadlessConfig::default()
        });
        assert!(
            runner.capabilities().supports_sdk_tool_calling(),
            "mcp_tool_calling=true routes tool turns through the ACP converse() loop"
        );
    }

    #[test]
    fn mcp_servers_to_acp_json_http_matches_wire_format() {
        let servers = vec![McpServerConfig {
            name: "dravr".to_owned(),
            transport: McpTransport::Http {
                url: "http://localhost:8081/mcp".to_owned(),
                headers: vec![McpHeader {
                    name: "Authorization".to_owned(),
                    value: "Bearer tok".to_owned(),
                }],
            },
        }];
        let json = mcp_servers_to_acp_json(&servers);
        assert_eq!(json.len(), 1);
        assert_eq!(json[0]["type"], "http");
        assert_eq!(json[0]["name"], "dravr");
        assert_eq!(json[0]["url"], "http://localhost:8081/mcp");
        assert_eq!(json[0]["headers"][0]["name"], "Authorization");
        assert_eq!(json[0]["headers"][0]["value"], "Bearer tok");
    }

    #[test]
    fn mcp_servers_to_acp_json_empty_is_empty_array() {
        assert!(mcp_servers_to_acp_json(&[]).is_empty());
    }

    #[test]
    fn build_prompt_params_without_max_tokens() {
        let blocks = vec![json!({"type": "text", "text": "hello"})];
        let params = build_prompt_params("sess-1", &blocks, None);
        assert_eq!(params["sessionId"], "sess-1");
        assert!(params["prompt"].is_array());
        assert!(params.get("maxTokens").is_none());
    }

    #[test]
    fn build_prompt_params_with_max_tokens() {
        let blocks = vec![json!({"type": "text", "text": "hello"})];
        let params = build_prompt_params("sess-2", &blocks, Some(1024));
        assert_eq!(params["sessionId"], "sess-2");
        assert_eq!(params["maxTokens"], 1024);
    }

    #[test]
    fn build_prompt_params_preserves_prompt_blocks() {
        let blocks = vec![
            json!({"type": "text", "text": "hello"}),
            json!({"type": "image", "data": "abc", "mimeType": "image/png"}),
        ];
        let params = build_prompt_params("s1", &blocks, Some(512));
        let prompt = params["prompt"].as_array().unwrap(); // Safe: test assertion
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0]["type"], "text");
        assert_eq!(prompt[1]["type"], "image");
    }

    #[test]
    fn default_mode_multi_turn_system_as_plain_text() {
        let runner = test_runner(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("Return JSON only"),
            ChatMessage::user("First question"),
            ChatMessage::assistant("{\"answer\": 1}"),
            ChatMessage::user("Second question"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // System prompt as plain text — no XML tags
        assert!(text.contains("Return JSON only"));
        assert!(!text.contains("<system-instructions>"));

        // Conversation history and current message also present
        assert!(text.contains("<conversation-history>"));
        assert!(text.contains("User: First question"));
        assert!(text.contains("Second question"));
    }

    #[test]
    fn disabled_injection_multi_turn_no_system_in_prompt() {
        let runner = test_runner_no_system_injection(20);
        let request = ChatRequest::new(vec![
            ChatMessage::system("Return JSON only"),
            ChatMessage::user("First question"),
            ChatMessage::assistant("{\"answer\": 1}"),
            ChatMessage::user("Second question"),
        ]);
        let blocks = runner.build_prompt_blocks(&request);
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion

        // System prompt NOT in text when injection is disabled
        assert!(!text.contains("Return JSON only"));

        // History and current message still present
        assert!(text.contains("<conversation-history>"));
        assert!(text.contains("User: First question"));
        assert!(text.contains("Second question"));
    }

    #[test]
    fn default_mode_with_images_includes_system() {
        use crate::types::ImagePart;

        let runner = test_runner(20);
        let img = ImagePart::new("aGVsbG8=", "image/png").unwrap(); // Safe: test assertion
        let request = ChatRequest::new(vec![
            ChatMessage::system("Analyze images precisely"),
            ChatMessage::user_with_images("What is this?", vec![img]),
        ]);
        let blocks = runner.build_prompt_blocks(&request);

        // System prompt present as plain text
        let text = blocks[0]["text"].as_str().unwrap(); // Safe: test assertion
        assert!(text.contains("Analyze images precisely"));
        assert!(!text.contains("<system-instructions>"));
        assert!(text.contains("What is this?"));

        // Image block still present
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1]["type"], "image");
    }

    #[test]
    fn resolve_model_uses_explicit_model() {
        let runner = test_runner(20);
        assert_eq!(runner.resolve_model(Some("gpt-4.1")), "gpt-4.1");
    }

    #[test]
    fn resolve_model_maps_copilot_headless_to_default() {
        let runner = test_runner(20);
        let result = runner.resolve_model(Some("copilot_headless"));
        assert_eq!(result, runner.config.model);
    }

    #[test]
    fn resolve_model_maps_copilot_to_default() {
        let runner = test_runner(20);
        let result = runner.resolve_model(Some("copilot"));
        assert_eq!(result, runner.config.model);
    }

    #[test]
    fn resolve_model_uses_default_when_none() {
        let runner = test_runner(20);
        let result = runner.resolve_model(None);
        assert_eq!(result, runner.config.model);
    }

    /// Build a `session/update` notification of an `AgentMessageChunk`
    /// carrying the given text. Mirrors the ACP wire format consumed by
    /// `process_notification`.
    fn make_text_chunk_notification(text: &str) -> Value {
        json!({
            "sessionId": "test-session",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": text
                }
            }
        })
    }

    /// Build a `session/update` notification of a `ToolCall` with the
    /// given id, title, and status.
    fn make_tool_call_notification(id: &str, title: &str, status: &str) -> Value {
        json!({
            "sessionId": "test-session",
            "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": id,
                "title": title,
                "status": status,
                "kind": "other",
                "content": []
            }
        })
    }

    /// Helper: pull the next event from `rx` and assert it is the expected
    /// `TextDelta`. Uses the existing test-assertion safety convention so
    /// the architectural validator (which counts bare panics / expects in
    /// src/) stays green.
    fn expect_text_delta(
        rx: &mut mpsc::UnboundedReceiver<Result<HeadlessStreamEvent, RunnerError>>,
        expected: &str,
    ) {
        let event = rx.try_recv().unwrap().unwrap(); // Safe: test assertion
        let HeadlessStreamEvent::TextDelta(s) = event else {
            unreachable!("expected TextDelta event variant"); // Safe: test assertion
        };
        assert_eq!(s, expected);
    }

    fn expect_tool_call(
        rx: &mut mpsc::UnboundedReceiver<Result<HeadlessStreamEvent, RunnerError>>,
    ) -> ObservedToolCall {
        let event = rx.try_recv().unwrap().unwrap(); // Safe: test assertion
        let HeadlessStreamEvent::ToolCall(tc) = event else {
            unreachable!("expected ToolCall event variant"); // Safe: test assertion
        };
        tc
    }

    #[test]
    fn streaming_notification_forwards_text_delta_and_accumulates() {
        let mut acc = TurnAccumulator::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        process_notification_streaming(&make_text_chunk_notification("Hello, "), &mut acc, &tx);
        process_notification_streaming(&make_text_chunk_notification("world!"), &mut acc, &tx);

        // Accumulator behaves like the non-streaming path
        assert_eq!(acc.content, "Hello, world!");
        assert_eq!(acc.tool_calls.len(), 0);

        // Both chunks are emitted on the channel in order
        expect_text_delta(&mut rx, "Hello, ");
        expect_text_delta(&mut rx, "world!");
        assert!(rx.try_recv().is_err(), "expected no more events");
    }

    #[test]
    fn streaming_notification_forwards_tool_call() {
        let mut acc = TurnAccumulator::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        process_notification_streaming(
            &make_tool_call_notification("tc_1", "Reading file", "in_progress"),
            &mut acc,
            &tx,
        );

        assert_eq!(acc.tool_calls.len(), 1);
        assert_eq!(acc.tool_calls[0].id, "tc_1");
        assert_eq!(acc.tool_calls[0].title, "Reading file");

        let tc = expect_tool_call(&mut rx);
        assert_eq!(tc.id, "tc_1");
        assert_eq!(tc.title, "Reading file");
    }

    #[test]
    fn process_notification_no_channel_matches_streaming_state() {
        // The non-streaming entry point must produce the *same* accumulator
        // state as the streaming one — we just lose the per-event channel.
        let mut acc_plain = TurnAccumulator::new();
        process_notification(&make_text_chunk_notification("Hi"), &mut acc_plain);
        process_notification(
            &make_tool_call_notification("tc_a", "Tool", "completed"),
            &mut acc_plain,
        );

        let mut acc_stream = TurnAccumulator::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        process_notification_streaming(&make_text_chunk_notification("Hi"), &mut acc_stream, &tx);
        process_notification_streaming(
            &make_tool_call_notification("tc_a", "Tool", "completed"),
            &mut acc_stream,
            &tx,
        );

        assert_eq!(acc_plain.content, acc_stream.content);
        assert_eq!(acc_plain.tool_calls.len(), acc_stream.tool_calls.len());
        assert_eq!(acc_plain.tool_calls[0].id, acc_stream.tool_calls[0].id);
        assert_eq!(
            acc_plain.tool_calls[0].title,
            acc_stream.tool_calls[0].title
        );
    }
}
