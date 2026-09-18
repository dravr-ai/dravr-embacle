// ABOUTME: CopilotSdkRunner speaks to GitHub's Rust Copilot runtime through github-copilot-sdk over stdio.
// ABOUTME: One warm client per runner, one session per turn; served model, usage and tool events come from the runtime's own events.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! GitHub Copilot through its Rust SDK.
//!
//! [`CopilotSdkRunner`] drives the same Rust runtime that `copilot --acp`
//! wraps, minus the JS adapter in between. The SDK spawns the ~400 KB
//! `copilot-runtime` wrapper (`--server --stdio`), which loads
//! `runtime.node`; there is no Node process anywhere on this path.
//!
//! Why a session per turn: the host owns the conversation (history arrives
//! on every [`ChatRequest`]), so a session carries nothing worth keeping
//! between calls, and a fresh one guarantees a request never sees another
//! tenant's context. Prior turns are rendered into the prompt the same way
//! the ACP provider renders them ([`render_turn`]); the system prompt goes
//! through the runtime's own system slot (`system_message` in `replace`
//! mode), which the ACP adapter has no honoured field for.
//!
//! What this path reports that ACP cannot: the model that actually served
//! (`assistant.usage.model`), cache read/write counts per call, and tool
//! executions with their name, arguments and result.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use github_copilot_sdk::handler::{
    ApproveAllHandler, DenyAllHandler, PermissionHandler, PermissionResult,
};
use github_copilot_sdk::session_events::{
    AssistantMessageData, AssistantMessageDeltaData, AssistantUsageData, SessionErrorData,
    SessionEventType, ToolExecutionCompleteData, ToolExecutionStartData,
};
use github_copilot_sdk::subscription::RecvErrorKind;
use github_copilot_sdk::{
    Attachment, CliProgram, Client, ClientMode, ClientOptions, Error as SdkError,
    ErrorKind as SdkErrorKind, EventSubscription, IndexMap, LogLevel, McpHttpServerConfig,
    McpServerConfig, McpStdioServerConfig, MessageOptions, PermissionRequestData,
    PermissionRequestKind, ProtocolErrorKind, RequestId, SessionConfig, SessionErrorKind,
    SessionEvent, SessionId, SystemMessageConfig, Transport,
};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};
use tokio::time;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::{debug, info, warn};

use crate::copilot_common::{
    render_turn, system_prompt, HeadlessEventStream, HeadlessStreamEvent, HeadlessToolResponse,
    HeadlessTurnProvider, ObservedToolCall, PermissionPolicy,
};
use crate::copilot_models::catalog_ids;
use crate::copilot_sdk_config::CopilotSdkConfig;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, McpTransport,
    RunnerError, StreamChunk, TokenUsage,
};

/// Service name on every `RunnerError::external_service` this module raises.
const SERVICE: &str = "copilot-sdk";

/// Sender side of a streaming turn.
type EventSink<'a> = Option<&'a mpsc::UnboundedSender<Result<HeadlessStreamEvent, RunnerError>>>;

/// GitHub Copilot through `github-copilot-sdk` over stdio.
///
/// Cheap to clone the `Arc` inside; the runner is built once and shared.
#[derive(Debug)]
pub struct CopilotSdkRunner {
    /// Shared with the task that drives a streaming turn, which outlives the
    /// `converse_stream` call that started it.
    inner: Arc<Inner>,
    models: Vec<String>,
}

#[derive(Debug)]
struct Inner {
    config: CopilotSdkConfig,
    /// The warm client. `None` until the first turn, and again after a
    /// transport failure: the SDK says a client whose transport failed is to
    /// be discarded, and the next turn starts a fresh one.
    client: Mutex<Option<Arc<Client>>>,
}

impl CopilotSdkRunner {
    /// Build from environment variables (see [`CopilotSdkConfig::from_env`]).
    #[must_use]
    pub fn from_env() -> Self {
        Self::with_config(CopilotSdkConfig::from_env())
    }

    /// Build from an explicit configuration.
    #[must_use]
    pub fn with_config(config: CopilotSdkConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                client: Mutex::new(None),
            }),
            models: catalog_ids(),
        }
    }

    /// The configuration this runner was built with.
    #[must_use]
    pub fn config(&self) -> &CopilotSdkConfig {
        &self.inner.config
    }
}

impl Inner {
    /// The warm client, started on first use.
    async fn client(&self) -> Result<Arc<Client>, RunnerError> {
        let mut slot = self.client.lock().await;
        if let Some(client) = slot.as_ref() {
            return Ok(Arc::clone(client));
        }

        let mut options = ClientOptions::new()
            .with_transport(Transport::Stdio)
            .with_mode(ClientMode::Empty)
            .with_base_directory(self.config.base_directory.clone())
            .with_log_level(LogLevel::Warning);
        if let Some(path) = &self.config.runtime_path {
            options = options.with_program(CliProgram::Path(path.clone()));
        }
        // The token rides the child's environment as COPILOT_GITHUB_TOKEN —
        // the runtime's own precedence (then GH_TOKEN, GITHUB_TOKEN) — exactly
        // as the ACP adapter hands it over, so the runtime runs the same token
        // exchange and a personal access token works. The SDK's `github_token`
        // option is a different path: it presents the value as an SDK auth
        // token, and the endpoint behind that answers a PAT with 400
        // "Personal Access Tokens are not supported for this endpoint".
        // `--no-auto-login` would switch that resolution off along with the
        // interactive login, so the logged-in-user default stays on either way.
        if let Some(token) = &self.config.github_token {
            options = options.with_env([("COPILOT_GITHUB_TOKEN", token.as_str())]);
        }

        let started = Instant::now();
        let client = time::timeout(self.config.session_timeout, Client::start(options))
            .await
            .map_err(|_| {
                RunnerError::timeout(format!(
                    "{SERVICE}: runtime did not finish its handshake within {:?}",
                    self.config.session_timeout
                ))
            })?
            .map_err(|e| map_sdk_error(&e))?;
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            runtime = ?self.config.runtime_path,
            "{SERVICE}: runtime started"
        );

        let client = Arc::new(client);
        *slot = Some(Arc::clone(&client));
        Ok(client)
    }

    /// Forget a client whose transport failed. Stopping it is best effort:
    /// its child is already gone or unreachable.
    async fn discard(&self, failed: &Arc<Client>) {
        let mut slot = self.client.lock().await;
        if slot.as_ref().is_some_and(|c| Arc::ptr_eq(c, failed)) {
            *slot = None;
        }
        drop(slot);
        if let Err(e) = failed.stop().await {
            debug!(error = %e, "{SERVICE}: stopping a failed client");
        }
    }

    /// The model this turn runs on, checked against the runtime's catalogue.
    ///
    /// The runtime silently substitutes another model for an id it does not
    /// know, so a typo would serve the wrong model with no error anywhere —
    /// the failure that once put production on a model nobody chose. The
    /// catalogue is cached by the client after its first fetch.
    async fn validated_model(
        &self,
        client: &Client,
        requested: Option<&str>,
    ) -> Result<String, RunnerError> {
        let model = requested.map_or_else(|| self.config.model.clone(), str::to_owned);
        let catalogue = client.list_models().await.map_err(|e| map_sdk_error(&e))?;
        if catalogue.iter().any(|m| m.id == model) {
            Ok(model)
        } else {
            Err(RunnerError::model_unavailable(model))
        }
    }

    /// One turn: a fresh session, the prompt, every event until idle.
    async fn run_turn(
        &self,
        request: &ChatRequest,
        sink: EventSink<'_>,
    ) -> Result<HeadlessToolResponse, RunnerError> {
        let started = Instant::now();
        let client = self.client().await?;
        let model = self
            .validated_model(&client, request.model.as_deref())
            .await?;

        // A scratch directory per turn: the runtime reads instruction files
        // and MCP config from its working directory, and nothing of the
        // host's must be discoverable that way.
        let scratch = tempfile::Builder::new()
            .prefix("embacle-copilot-sdk-")
            .tempdir()
            .map_err(|e| RunnerError::internal(format!("{SERVICE}: scratch dir: {e}")))?;

        let rendered = render_turn(request, self.config.max_history_turns, false);
        let attachments = write_image_attachments(scratch.path(), rendered.last_user)?;
        let session_config = self.session_config(request, &model, scratch.path());

        let session = match time::timeout(
            self.config.session_timeout,
            client.create_session(session_config),
        )
        .await
        {
            Ok(Ok(session)) => session,
            Ok(Err(e)) => {
                if e.is_transport_failure() {
                    self.discard(&client).await;
                }
                return Err(map_sdk_error(&e));
            }
            Err(_) => {
                return Err(RunnerError::timeout(format!(
                    "{SERVICE}: session.create exceeded {:?}",
                    self.config.session_timeout
                )));
            }
        };
        let mut events = session.subscribe();

        let mut message = MessageOptions::from(rendered.text);
        if !attachments.is_empty() {
            message.attachments = Some(attachments);
        }
        if let Err(e) = session.send(message).await {
            if e.is_transport_failure() {
                self.discard(&client).await;
            }
            let _ = session.disconnect().await;
            return Err(map_sdk_error(&e));
        }

        let outcome =
            time::timeout(self.config.prompt_timeout, collect_turn(&mut events, sink)).await;
        let result = match outcome {
            Ok(Ok(acc)) => Ok(acc.finish(model)),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // The session stays usable after `abort`, but this one is
                // over; the disconnect below releases it.
                if let Err(e) = session.abort().await {
                    debug!(error = %e, "{SERVICE}: abort after prompt timeout");
                }
                Err(RunnerError::timeout(format!(
                    "{SERVICE}: turn exceeded {:?}",
                    self.config.prompt_timeout
                )))
            }
        };

        if let Err(e) = session.disconnect().await {
            debug!(error = %e, "{SERVICE}: session.disconnect");
        }
        drop(scratch);

        match &result {
            Ok(response) => info!(
                model = %response.model,
                tool_calls = response.tool_calls.len(),
                elapsed_ms = started.elapsed().as_millis(),
                "{SERVICE}: turn complete"
            ),
            Err(e) => {
                warn!(error = %e, elapsed_ms = started.elapsed().as_millis(), "{SERVICE}: turn failed");
            }
        }
        result
    }

    fn session_config(&self, request: &ChatRequest, model: &str, scratch: &Path) -> SessionConfig {
        let mut config = SessionConfig::default()
            .with_model(model)
            .with_streaming(true)
            .with_working_directory(scratch)
            .with_client_name(concat!("embacle/", env!("CARGO_PKG_VERSION")));

        if let Some(system) = system_prompt(request) {
            config = config.with_system_message(
                SystemMessageConfig::new()
                    .with_mode("replace")
                    .with_content(system),
            );
        }

        // `ClientMode::Empty` requires an explicit allowlist; with no MCP
        // servers on the request it is empty and the model has no tools.
        let mut available: Vec<String> = Vec::new();
        let mut registered: Vec<String> = Vec::new();
        if !request.mcp_servers.is_empty() {
            let mut servers = IndexMap::new();
            for server in &request.mcp_servers {
                servers.insert(server.name.clone(), mcp_server(&server.transport));
                registered.push(server.name.clone());
            }
            config = config.with_mcp_servers(servers);
            available.push("mcp:*".to_owned());
        }
        config
            .with_available_tools(available)
            .with_permission_handler(self.permission_handler(registered))
    }

    /// The permission handler for a turn that registered `servers`.
    ///
    /// Calls to those servers are approved: the loopback host on the request
    /// is the caller's own tool surface, and calling it is what the caller
    /// asked for. Everything else follows the configured policy, which
    /// denies unless the host opted into approval.
    fn permission_handler(&self, servers: Vec<String>) -> Arc<dyn PermissionHandler> {
        match self.config.permission_policy {
            PermissionPolicy::AutoApprove => Arc::new(ApproveAllHandler),
            PermissionPolicy::DenyAll => Arc::new(RegisteredMcpHandler { servers }),
        }
    }
}

/// Approves permission prompts for tool calls on the MCP servers the turn
/// registered, and rejects everything else.
///
/// The runtime asks before every MCP tool call. Under the deny policy the
/// answer is still yes for the caller's own servers — otherwise a request
/// that supplies `mcp_servers` could never have them called — and no for
/// anything the caller did not put on the request.
#[derive(Debug)]
struct RegisteredMcpHandler {
    servers: Vec<String>,
}

impl RegisteredMcpHandler {
    fn permits(&self, data: &PermissionRequestData) -> bool {
        if !matches!(data.kind, Some(PermissionRequestKind::Mcp))
            || data.managed_approval_required == Some(true)
        {
            return false;
        }
        let request = data.extra.get("permissionRequest").unwrap_or(&data.extra);
        request
            .get("serverName")
            .and_then(Value::as_str)
            .is_some_and(|server| self.servers.iter().any(|known| known == server))
    }
}

#[async_trait]
impl PermissionHandler for RegisteredMcpHandler {
    async fn handle(
        &self,
        session_id: SessionId,
        request_id: RequestId,
        data: PermissionRequestData,
    ) -> PermissionResult {
        if self.permits(&data) {
            PermissionResult::approve_once()
        } else {
            DenyAllHandler.handle(session_id, request_id, data).await
        }
    }
}

/// An MCP server on the request, in the SDK's shape.
fn mcp_server(transport: &McpTransport) -> McpServerConfig {
    match transport {
        McpTransport::Http { url, headers } => McpServerConfig::Http(McpHttpServerConfig {
            url: url.clone(),
            headers: headers
                .iter()
                .map(|h| (h.name.clone(), h.value.clone()))
                .collect(),
            ..McpHttpServerConfig::default()
        }),
        McpTransport::Sse { url, headers } => McpServerConfig::Sse(McpHttpServerConfig {
            url: url.clone(),
            headers: headers
                .iter()
                .map(|h| (h.name.clone(), h.value.clone()))
                .collect(),
            ..McpHttpServerConfig::default()
        }),
        McpTransport::Stdio { command, args, env } => {
            McpServerConfig::Stdio(McpStdioServerConfig {
                command: command.clone(),
                args: args.clone(),
                env: env
                    .iter()
                    .map(|h| (h.name.clone(), h.value.clone()))
                    .collect(),
                ..McpStdioServerConfig::default()
            })
        }
    }
}

/// The current user message's images, written into the scratch directory so
/// the runtime can attach them: the SDK attaches files, not bytes.
fn write_image_attachments(
    dir: &Path,
    last_user: Option<&ChatMessage>,
) -> Result<Vec<Attachment>, RunnerError> {
    let Some(images) = last_user.and_then(|m| m.images.as_ref()) else {
        return Ok(Vec::new());
    };
    let mut attachments = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        let bytes = BASE64.decode(&image.data).map_err(|e| {
            RunnerError::internal(format!("{SERVICE}: image {index} is not valid base64: {e}"))
        })?;
        let extension = match image.mime_type.as_str() {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/webp" => "webp",
            "image/gif" => "gif",
            _ => "bin",
        };
        let path: PathBuf = dir.join(format!("image-{index}.{extension}"));
        fs::write(&path, bytes)
            .map_err(|e| RunnerError::internal(format!("{SERVICE}: writing image {index}: {e}")))?;
        attachments.push(Attachment::File {
            path,
            display_name: None,
            line_range: None,
        });
    }
    Ok(attachments)
}

/// Everything a turn accumulates from the runtime's events.
#[derive(Debug, Default)]
struct TurnAccumulator {
    content: String,
    served_model: Option<String>,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_read_tokens: u32,
    cached_write_tokens: u32,
    reasoning_tokens: u32,
    saw_usage: bool,
    tool_calls: Vec<ObservedToolCall>,
    finish_reason: Option<String>,
}

impl TurnAccumulator {
    /// Fold one event in. `Ok(true)` when the session went idle — the turn
    /// is over; `Err` when the runtime reported the turn failed.
    fn absorb(&mut self, event: &SessionEvent, sink: EventSink<'_>) -> Result<bool, RunnerError> {
        match event.parsed_type() {
            SessionEventType::AssistantMessageDelta => {
                // A sub-agent's deltas carry its `agent_id`; only the main
                // conversation's text is the answer.
                if event.agent_id.is_none() {
                    if let Some(delta) = event.typed_data::<AssistantMessageDeltaData>() {
                        emit(sink, HeadlessStreamEvent::TextDelta(delta.delta_content));
                    }
                }
            }
            SessionEventType::AssistantMessage => {
                if let Some(message) = event.typed_data::<AssistantMessageData>() {
                    // A turn with tool calls carries several assistant
                    // messages; the last non-empty one is the answer.
                    if !message.content.is_empty() {
                        self.content = message.content;
                    }
                    if let Some(model) = message.model {
                        self.served_model = Some(model);
                    }
                }
            }
            SessionEventType::AssistantUsage => {
                if let Some(usage) = event.typed_data::<AssistantUsageData>() {
                    self.add_usage(&usage);
                }
            }
            SessionEventType::ToolExecutionStart => {
                if let Some(start) = event.typed_data::<ToolExecutionStartData>() {
                    let observed = ObservedToolCall {
                        id: start.tool_call_id,
                        title: start.tool_name.clone(),
                        status: "InProgress".to_owned(),
                        name: Some(start.tool_name),
                        arguments: start.arguments,
                        result: None,
                    };
                    self.tool_calls.push(observed.clone());
                    emit(sink, HeadlessStreamEvent::ToolCall(observed));
                }
            }
            SessionEventType::ToolExecutionComplete => {
                if let Some(done) = event.typed_data::<ToolExecutionCompleteData>() {
                    if let Some(existing) = self
                        .tool_calls
                        .iter_mut()
                        .find(|t| t.id == done.tool_call_id)
                    {
                        existing.status =
                            String::from(if done.success { "Completed" } else { "Failed" });
                        existing.result = done
                            .result
                            .map(|r| r.content)
                            .or_else(|| done.error.map(|e| e.message));
                        emit(sink, HeadlessStreamEvent::ToolCall(existing.clone()));
                    }
                }
            }
            SessionEventType::SessionError => {
                return Err(map_session_error(event));
            }
            SessionEventType::SessionIdle => return Ok(true),
            _ => {}
        }
        Ok(false)
    }

    fn add_usage(&mut self, usage: &AssistantUsageData) {
        self.saw_usage = true;
        self.prompt_tokens = self.prompt_tokens.saturating_add(count(usage.input_tokens));
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(count(usage.output_tokens));
        self.cached_read_tokens = self
            .cached_read_tokens
            .saturating_add(count(usage.cache_read_tokens));
        self.cached_write_tokens = self
            .cached_write_tokens
            .saturating_add(count(usage.cache_write_tokens));
        self.reasoning_tokens = self
            .reasoning_tokens
            .saturating_add(count(usage.reasoning_tokens));
        self.served_model = Some(usage.model.clone());
        if usage.finish_reason.is_some() {
            self.finish_reason.clone_from(&usage.finish_reason);
        }
    }

    fn finish(self, requested_model: String) -> HeadlessToolResponse {
        let usage = self.saw_usage.then(|| TokenUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.prompt_tokens.saturating_add(self.completion_tokens),
            cached_read_tokens: Some(self.cached_read_tokens),
            cached_write_tokens: Some(self.cached_write_tokens),
            reasoning_tokens: Some(self.reasoning_tokens),
        });
        HeadlessToolResponse {
            content: self.content,
            model: self.served_model.unwrap_or(requested_model),
            tool_calls: self.tool_calls,
            usage,
            finish_reason: self.finish_reason,
        }
    }
}

/// A token count from the wire, where a missing or negative value is zero.
fn count(value: Option<i64>) -> u32 {
    value.and_then(|v| u32::try_from(v).ok()).unwrap_or(0)
}

fn emit(sink: EventSink<'_>, event: HeadlessStreamEvent) {
    if let Some(tx) = sink {
        // A receiver that went away cancels nothing: the turn runs to its
        // end so the runtime's session is released cleanly.
        let _ = tx.send(Ok(event));
    }
}

/// Every event from the prompt until the session goes idle.
async fn collect_turn(
    events: &mut EventSubscription,
    sink: EventSink<'_>,
) -> Result<TurnAccumulator, RunnerError> {
    let mut acc = TurnAccumulator::default();
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(e) => match e.kind() {
                RecvErrorKind::Lagged(lag) => {
                    warn!(%lag, "{SERVICE}: event subscription lagged; continuing from the next event");
                    continue;
                }
                RecvErrorKind::Closed => {
                    return Err(RunnerError::external_service(
                        SERVICE,
                        "event stream closed before the turn went idle",
                    ));
                }
                // The kind is `#[non_exhaustive]`; anything the SDK adds
                // later is a failed subscription until proven otherwise.
                other => {
                    return Err(RunnerError::external_service(
                        SERVICE,
                        format!("event subscription failed: {other:?}"),
                    ));
                }
            },
        };
        if acc.absorb(&event, sink)? {
            return Ok(acc);
        }
    }
}

/// A `session.error` event as the error the host understands.
///
/// During a turn the runtime reports authentication, quota, rate-limit and
/// context failures as events with an `error_type`, not as RPC errors.
fn map_session_error(event: &SessionEvent) -> RunnerError {
    let Some(data) = event.typed_data::<SessionErrorData>() else {
        return RunnerError::external_service(SERVICE, format!("session.error: {}", event.data));
    };
    let mut message = data.message;
    if let Some(code) = &data.error_code {
        message.push_str(" (");
        message.push_str(code);
        message.push(')');
    }
    match data.error_type.as_str() {
        "authentication" | "authorization" => RunnerError::auth_failure(message),
        "quota" | "rate_limit" => RunnerError::rate_limit(SERVICE, message),
        "context_limit" => RunnerError::context_length(message),
        _ => RunnerError::external_service(SERVICE, message),
    }
}

/// An SDK error as the error the host understands.
fn map_sdk_error(error: &SdkError) -> RunnerError {
    let message = error.to_string();
    match error.kind() {
        SdkErrorKind::BinaryNotFound { .. } => RunnerError::binary_not_found("copilot-runtime"),
        SdkErrorKind::InvalidConfig => RunnerError::config(message),
        SdkErrorKind::Session(SessionErrorKind::Timeout(_)) => RunnerError::timeout(message),
        SdkErrorKind::Protocol(ProtocolErrorKind::VersionMismatch { .. }) => {
            RunnerError::config(format!("{SERVICE}: {message}"))
        }
        _ => RunnerError::external_service(SERVICE, message),
    }
}

#[async_trait]
impl HeadlessTurnProvider for CopilotSdkRunner {
    async fn converse(&self, request: &ChatRequest) -> Result<HeadlessToolResponse, RunnerError> {
        self.inner.run_turn(request, None).await
    }

    async fn converse_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<HeadlessEventStream, RunnerError> {
        let (tx, rx) = mpsc::unbounded_channel();
        let inner = Arc::clone(&self.inner);
        let request = request.clone();
        // The task outlives this call; its only untracked failure is a panic,
        // which drops `tx` and ends the stream without a `Done` — the
        // consumer sees a truncated turn, never a silent success.
        tokio::spawn(async move {
            let outcome = inner.run_turn(&request, Some(&tx)).await;
            let _ = match outcome {
                Ok(response) => tx.send(Ok(HeadlessStreamEvent::Done(response))),
                Err(e) => tx.send(Err(e)),
            };
        });
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}

#[async_trait]
impl LlmProvider for CopilotSdkRunner {
    fn name(&self) -> &'static str {
        "copilot_sdk"
    }

    fn display_name(&self) -> &str {
        "GitHub Copilot (SDK)"
    }

    fn capabilities(&self) -> LlmCapabilities {
        let base =
            LlmCapabilities::STREAMING | LlmCapabilities::SYSTEM_MESSAGES | LlmCapabilities::VISION;
        // SDK_TOOL_CALLING is opt-in via `mcp_tool_calling`: when set, the
        // caller passes `mcp_servers` per request and the runtime calls those
        // tools natively; when unset, callers fall through to text-based tool
        // calling, where the host parses `<tool_call>` blocks itself.
        if self.inner.config.mcp_tool_calling {
            base | LlmCapabilities::SDK_TOOL_CALLING
        } else {
            base
        }
    }

    fn default_model(&self) -> &str {
        &self.inner.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.models
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let turn = self.converse(request).await?;
        Ok(ChatResponse {
            content: turn.content,
            model: turn.model,
            usage: turn.usage,
            finish_reason: turn.finish_reason,
            warnings: None,
            tool_calls: None,
        })
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let events = self.converse_stream(request).await?;
        let chunks = events.map(|event| {
            event.map(|event| match event {
                HeadlessStreamEvent::TextDelta(delta) => StreamChunk {
                    delta,
                    is_final: false,
                    finish_reason: None,
                },
                HeadlessStreamEvent::ToolCall(_) => StreamChunk {
                    delta: String::new(),
                    is_final: false,
                    finish_reason: None,
                },
                HeadlessStreamEvent::Done(response) => StreamChunk {
                    delta: String::new(),
                    is_final: true,
                    finish_reason: response.finish_reason,
                },
            })
        });
        Ok(Box::pin(chunks))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let client = self.inner.client().await?;
        client.ping(None).await.map_err(|e| map_sdk_error(&e))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::McpHeader;
    use serde_json::json;

    /// `PermissionDecision` is not exported by the SDK, so the decision is
    /// read off its `Debug` form.
    fn approves_once(result: &PermissionResult) -> bool {
        format!("{result:?}").contains("ApproveOnce")
    }

    fn event(event_type: &str, data: serde_json::Value) -> SessionEvent {
        let mut envelope = json!({
            "id": "evt-1",
            "timestamp": "2026-09-18T00:00:00Z",
            "type": event_type,
        });
        envelope["data"] = data;
        serde_json::from_value(envelope).unwrap_or_else(|e| unreachable!("test event: {e}"))
    }

    #[test]
    fn usage_accumulates_across_calls_and_names_the_served_model() {
        let mut acc = TurnAccumulator::default();
        let first = event(
            "assistant.usage",
            json!({"model": "claude-haiku-4.5", "inputTokens": 21_678, "outputTokens": 154, "cacheReadTokens": 0, "cacheWriteTokens": 21_678}),
        );
        let second = event(
            "assistant.usage",
            json!({"model": "claude-haiku-4.5", "inputTokens": 21_860, "outputTokens": 9, "cacheReadTokens": 21_678, "cacheWriteTokens": 176, "reasoningTokens": 113, "finishReason": "stop"}),
        );
        assert_eq!(acc.absorb(&first, None).ok(), Some(false));
        assert_eq!(acc.absorb(&second, None).ok(), Some(false));
        let response = acc.finish("claude-sonnet-5".to_owned());
        assert_eq!(response.model, "claude-haiku-4.5");
        let usage = response
            .usage
            .unwrap_or_else(|| unreachable!("usage recorded"));
        assert_eq!(usage.prompt_tokens, 43_538);
        assert_eq!(usage.completion_tokens, 163);
        assert_eq!(usage.total_tokens, 43_701);
        assert_eq!(usage.cached_read_tokens, Some(21_678));
        assert_eq!(usage.cached_write_tokens, Some(21_854));
        assert_eq!(usage.reasoning_tokens, Some(113));
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn tool_events_carry_name_arguments_and_result() {
        let mut acc = TurnAccumulator::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let start = event(
            "tool.execution_start",
            json!({"toolCallId": "call-1", "toolName": "get_activities", "arguments": {"limit": 3}}),
        );
        let done = event(
            "tool.execution_complete",
            json!({"toolCallId": "call-1", "success": true, "result": {"content": "three rides"}}),
        );
        assert_eq!(acc.absorb(&start, Some(&tx)).ok(), Some(false));
        assert_eq!(acc.absorb(&done, Some(&tx)).ok(), Some(false));

        let Ok(Ok(HeadlessStreamEvent::ToolCall(started))) = rx.try_recv() else {
            unreachable!("first event is the tool start");
        };
        assert_eq!(started.status, "InProgress");
        assert_eq!(started.name.as_deref(), Some("get_activities"));
        assert_eq!(started.arguments, Some(json!({"limit": 3})));

        let Ok(Ok(HeadlessStreamEvent::ToolCall(finished))) = rx.try_recv() else {
            unreachable!("second event is the tool completion");
        };
        assert_eq!(finished.status, "Completed");
        assert_eq!(finished.result.as_deref(), Some("three rides"));

        let response = acc.finish("m".to_owned());
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(
            response.tool_calls[0].result.as_deref(),
            Some("three rides")
        );
    }

    #[test]
    fn the_last_non_empty_assistant_message_is_the_answer() {
        let mut acc = TurnAccumulator::default();
        let before_tool = event(
            "assistant.message",
            json!({"content": "", "messageId": "m1", "model": "claude-haiku-4.5"}),
        );
        let answer = event(
            "assistant.message",
            json!({"content": "PELICAN-7741", "messageId": "m2"}),
        );
        let idle = event("session.idle", json!({}));
        assert_eq!(acc.absorb(&before_tool, None).ok(), Some(false));
        assert_eq!(acc.absorb(&answer, None).ok(), Some(false));
        assert_eq!(acc.absorb(&idle, None).ok(), Some(true));
        assert_eq!(acc.finish("m".to_owned()).content, "PELICAN-7741");
    }

    #[test]
    fn session_error_types_map_onto_the_host_error_kinds() {
        let cases = [
            ("authentication", "auth"),
            ("authorization", "auth"),
            ("quota", "rate_limit"),
            ("rate_limit", "rate_limit"),
            ("context_limit", "context_length"),
            ("model_call", "external"),
        ];
        for (error_type, expected) in cases {
            let err = map_session_error(&event(
                "session.error",
                json!({"errorType": error_type, "message": "boom", "errorCode": "user_weekly_rate_limited"}),
            ));
            let text = err.to_string();
            assert!(text.contains("boom"), "{error_type}: message kept: {text}");
            assert!(
                text.contains("user_weekly_rate_limited"),
                "{error_type}: error code kept: {text}"
            );
            let kind = format!("{:?}", err.kind).to_lowercase();
            let matched = match expected {
                "auth" => kind.contains("auth"),
                "rate_limit" => kind.contains("ratelimit"),
                "context_length" => kind.contains("contextlength"),
                _ => kind.contains("externalservice"),
            };
            assert!(matched, "{error_type} -> {kind}");
        }
    }

    #[test]
    fn a_turn_ends_on_session_error() {
        let mut acc = TurnAccumulator::default();
        let err = event(
            "session.error",
            json!({"errorType": "quota", "message": "premium requests exhausted"}),
        );
        assert!(acc.absorb(&err, None).is_err());
    }

    #[test]
    fn mcp_servers_keep_their_headers() {
        let transport = McpTransport::Http {
            url: "http://127.0.0.1:1/mcp".to_owned(),
            headers: vec![McpHeader {
                name: "Authorization".to_owned(),
                value: "Bearer x".to_owned(),
            }],
        };
        let McpServerConfig::Http(http) = mcp_server(&transport) else {
            unreachable!("http stays http");
        };
        assert_eq!(http.url, "http://127.0.0.1:1/mcp");
        assert_eq!(
            http.headers.get("Authorization").map(String::as_str),
            Some("Bearer x")
        );
    }

    #[test]
    fn missing_or_negative_counts_are_zero() {
        assert_eq!(count(None), 0);
        assert_eq!(count(Some(-5)), 0);
        assert_eq!(count(Some(42)), 42);
    }

    fn mcp_permission(server: &str) -> PermissionRequestData {
        let mut data: PermissionRequestData =
            serde_json::from_value(json!({"kind": "mcp"})).unwrap_or_else(|e| unreachable!("{e}"));
        data.extra = json!({
            "requestId": "r-1",
            "permissionRequest": {"kind": "mcp", "serverName": server, "toolName": "get_secret_number"}
        });
        data
    }

    #[tokio::test]
    async fn registered_mcp_servers_are_approved_and_nothing_else_is() {
        let handler = RegisteredMcpHandler {
            servers: vec!["dravr".to_owned()],
        };
        assert!(handler.permits(&mcp_permission("dravr")));
        assert!(!handler.permits(&mcp_permission("github-mcp-server")));

        let mut shell: PermissionRequestData = serde_json::from_value(json!({"kind": "shell"}))
            .unwrap_or_else(|e| unreachable!("{e}"));
        shell.extra = json!({"permissionRequest": {"kind": "shell", "serverName": "dravr"}});
        assert!(
            !handler.permits(&shell),
            "a shell prompt is not an MCP call"
        );

        let mut managed = mcp_permission("dravr");
        managed.managed_approval_required = Some(true);
        assert!(!handler.permits(&managed), "managed policy wants a human");

        let approved = handler
            .handle(
                SessionId::from("s"),
                RequestId::new("1"),
                mcp_permission("dravr"),
            )
            .await;
        assert!(approves_once(&approved), "{approved:?}");
        let refused = handler
            .handle(
                SessionId::from("s"),
                RequestId::new("2"),
                mcp_permission("other"),
            )
            .await;
        assert!(!approves_once(&refused), "{refused:?}");
    }
}
