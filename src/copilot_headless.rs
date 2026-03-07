// ABOUTME: CopilotHeadlessRunner wraps the copilot CLI via ACP (Agent Client Protocol) for LLM completions.
// ABOUTME: Spawns copilot --acp per request and communicates via NDJSON-framed JSON-RPC over stdio.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::path::PathBuf;

use agent_client_protocol_schema as schema;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::copilot::{copilot_fallback_models, discover_copilot_models};
use crate::copilot_headless_config::CopilotHeadlessConfig;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, MessageRole, RunnerError,
    StreamChunk, TokenUsage,
};

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

    /// Read the next NDJSON message, skipping blank lines.
    async fn read_message(&mut self) -> Result<Value, RunnerError> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .reader
                .read_line(&mut line)
                .await
                .map_err(|e| RunnerError::internal(format!("Read failed: {e}")))?;
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
    let mut cmd = tokio::process::Command::new(cli_path);
    cmd.arg("--acp")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());

    if let Some(token) = github_token {
        cmd.env("COPILOT_GITHUB_TOKEN", token);
    }

    cmd.spawn()
        .map_err(|e| RunnerError::internal(format!("Failed to spawn copilot --acp: {e}")))
}

/// Initialize ACP connection and create a session.
///
/// Returns the transport and session id ready for prompting.
async fn setup_session(
    cli_path: &PathBuf,
    github_token: Option<&str>,
    model: &str,
    system_prompt: Option<&str>,
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

    // Initialize handshake
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
    transport.read_response(init_id).await?;

    // Create session with model and optional system prompt
    let mut session_params = json!({
        "model": model,
        "cwd": std::env::current_dir()
            .map_err(|e| RunnerError::internal(format!("Failed to get cwd: {e}")))?,
        "mcpServers": [],
    });
    if let Some(sys) = system_prompt {
        session_params["systemPrompt"] = Value::String(sys.to_owned());
    }

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

    debug!(session_id = %session_id, model = %model, "ACP session created");
    Ok((transport, child, session_id))
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
            }
        }
        schema::SessionUpdate::ToolCall(tc) => {
            acc.tool_calls.push(ObservedToolCall {
                id: tc.tool_call_id.0.to_string(),
                title: tc.title.clone(),
                status: format!("{:?}", tc.status),
            });
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
            }
        }
        _ => {}
    }
}

/// Auto-approve a permission request by selecting the first allow option.
fn build_permission_response(params: &Value) -> Value {
    let Ok(req) = serde_json::from_value::<schema::RequestPermissionRequest>(params.clone()) else {
        warn!("Failed to parse permission request, cancelling");
        return json!({ "outcome": "cancelled" });
    };

    let option_id = req
        .options
        .iter()
        .find(|o| {
            matches!(
                o.kind,
                schema::PermissionOptionKind::AllowOnce | schema::PermissionOptionKind::AllowAlways
            )
        })
        .or_else(|| req.options.first())
        .map(|o| &o.option_id);

    option_id.map_or_else(
        || {
            warn!("Permission request had no options, cancelling");
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
) -> Result<(ChatResponse, Vec<ObservedToolCall>), RunnerError> {
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
            };

            return Ok((response, acc.tool_calls));
        }

        // Server requests and notifications
        handle_server_message(&msg, transport, &mut acc).await?;
    }
}

/// Read messages until prompt completes, streaming chunks via channel.
async fn collect_streaming(
    transport: &mut AcpTransport,
    prompt_id: i64,
    chunk_tx: &mpsc::UnboundedSender<Result<StreamChunk, RunnerError>>,
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
                        let response = build_permission_response(params);
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
                    let response = build_permission_response(params);
                    transport.send_response(id, response).await?;
                }
            }
            _ => {}
        }
    }
    Ok(())
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

// ---------------------------------------------------------------------------
// Public runner
// ---------------------------------------------------------------------------

/// GitHub Copilot Headless (ACP) LLM provider.
///
/// Communicates with `copilot --acp` via the Agent Client Protocol (JSON-RPC over stdio).
/// Spawns a new copilot subprocess per request using NDJSON framing.
/// Uses types from `agent-client-protocol-schema` for protocol message deserialization.
///
/// Copilot manages its own tool execution loop internally (`SDK_TOOL_CALLING`).
/// Tool calls are observed and reported via [`HeadlessToolResponse`] from
/// [`converse()`](Self::converse), but the caller does not need to execute tools.
pub struct CopilotHeadlessRunner {
    config: CopilotHeadlessConfig,
    available_models: Vec<String>,
}

impl CopilotHeadlessRunner {
    /// Create a new provider from environment configuration.
    ///
    /// Attempts to discover available models via `gh copilot models`.
    /// Falls back to a static list if discovery fails.
    pub async fn from_env() -> Self {
        let available_models = discover_copilot_models()
            .await
            .unwrap_or_else(copilot_fallback_models);
        Self {
            config: CopilotHeadlessConfig::from_env(),
            available_models,
        }
    }

    /// Create a new provider with explicit configuration.
    pub async fn with_config(config: CopilotHeadlessConfig) -> Self {
        let available_models = discover_copilot_models()
            .await
            .unwrap_or_else(copilot_fallback_models);
        Self {
            config,
            available_models,
        }
    }

    /// Resolve the copilot CLI binary path.
    fn resolve_cli_path(&self) -> Result<PathBuf, RunnerError> {
        if let Some(ref path) = self.config.cli_path {
            return Ok(path.clone());
        }
        which::which("copilot").map_err(|_| RunnerError::binary_not_found("copilot"))
    }

    /// Extract the user prompt from the last user message.
    fn extract_user_prompt(request: &ChatRequest) -> &str {
        request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.content.as_str())
            .unwrap_or_default()
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
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.model)
            .to_owned();
        let user_prompt = Self::extract_user_prompt(request);
        let system_prompt = Self::extract_system_prompt(request);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            &model,
            system_prompt,
        )
        .await?;

        let prompt_id = transport
            .send_request(
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": user_prompt}],
                }),
            )
            .await?;

        let result = collect_complete(&mut transport, prompt_id, model).await;
        let _ = child.kill().await;

        let (response, tool_calls) = result?;
        Ok(HeadlessToolResponse {
            content: response.content,
            model: response.model,
            tool_calls,
            usage: response.usage,
            finish_reason: response.finish_reason,
        })
    }
}

#[async_trait]
impl LlmProvider for CopilotHeadlessRunner {
    fn name(&self) -> &'static str {
        "copilot_headless"
    }

    fn display_name(&self) -> &'static str {
        "GitHub Copilot (Headless)"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING
            | LlmCapabilities::SYSTEM_MESSAGES
            | LlmCapabilities::SDK_TOOL_CALLING
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = request
            .model
            .as_deref()
            .unwrap_or(&self.config.model)
            .to_owned();
        let user_prompt = Self::extract_user_prompt(request);
        let system_prompt = Self::extract_system_prompt(request);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            &model,
            system_prompt,
        )
        .await?;

        let prompt_id = transport
            .send_request(
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": user_prompt}],
                }),
            )
            .await?;

        let result = collect_complete(&mut transport, prompt_id, model).await;
        let _ = child.kill().await;
        result.map(|(response, _tool_calls)| response)
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let cli_path = self.resolve_cli_path()?;
        let model = request.model.as_deref().unwrap_or(&self.config.model);
        let user_prompt = Self::extract_user_prompt(request).to_owned();
        let system_prompt = Self::extract_system_prompt(request).map(str::to_owned);

        let (mut transport, mut child, session_id) = setup_session(
            &cli_path,
            self.config.github_token.as_deref(),
            model,
            system_prompt.as_deref(),
        )
        .await?;

        let prompt_id = transport
            .send_request(
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": user_prompt}],
                }),
            )
            .await?;

        let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();

        tokio::spawn(async move {
            let result = collect_streaming(&mut transport, prompt_id, &chunk_tx).await;
            if let Err(e) = result {
                let _ = chunk_tx.send(Err(e));
            }
            let _ = child.kill().await;
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(chunk_rx);
        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.resolve_cli_path().map_or(Ok(false), |path| {
            tracing::info!(cli_path = %path.display(), "Copilot Headless health check: binary found");
            Ok(true)
        })
    }
}
