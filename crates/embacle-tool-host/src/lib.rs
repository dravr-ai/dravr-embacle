// ABOUTME: Loopback MCP endpoint serving a CALLER-SUPPLIED tool surface to an ACP agent
// ABOUTME: One listener per process, one revocable session per turn, bearer dies with the guard
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Host your own tools to an ACP agent.
//!
//! # Why this exists
//!
//! An ACP agent such as `copilot --acp` runs its own tool loop inside its own
//! subprocess. It never asks its caller to execute a tool; it executes them
//! itself and reports afterwards, and the report carries no tool name — ACP's
//! `session/update` notification has `toolCallId`, `title`, `kind` and `status`,
//! and nothing that identifies which tool ran.
//!
//! So a caller that wants the agent to use ITS tools has exactly one channel:
//! declare an MCP server in `session/new`. The agent then speaks MCP to that
//! server, and `tools/call` carries the name and the arguments in full fidelity.
//!
//! That channel cannot be an in-process callback. The agent forks the MCP
//! server itself when the transport is stdio, so the server is a grandchild
//! process in a different address space, and the ACP frame carries only
//! `command`/`args`/`env` — no socket, no file descriptor, no back-channel.
//! Reaching a caller's [`McpToolExecutor`] therefore requires a real listener,
//! and loopback HTTP is the smallest one that works.
//!
//! # Why it is not in the root crate
//!
//! `AGENTS.md` states "No HTTP dependencies in core" as a design decision, and
//! the root crate earns it: `ffi = ["copilot-headless"]` ships a `staticlib`
//! compiled `panic = "abort"`, where a panic inside a tool handler would abort
//! the host application. Consumers that enable `copilot-headless` without ever
//! hosting tools should not pay for a web stack. This crate is opt-in by
//! existing separately.
//!
//! # Lifetime
//!
//! One [`ToolHost`] per process binds one listener. Each turn opens a
//! [`ToolSession`] carrying its own bearer token and its own tool surface.
//! **Dropping the session revokes the bearer immediately** — an orphaned agent
//! subprocess that retries after the turn ends gets `401` instead of executing
//! an irreversible action for a user who has already gone.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock, Weak};

use async_trait::async_trait;
use dravr_tronc::mcp::auth::{AuthError, AuthHook};
use dravr_tronc::mcp::host::ToolDispatcher;
use dravr_tronc::mcp::protocol::JsonRpcRequest;
use dravr_tronc::mcp::schema::{Tool, ToolResponse};
use dravr_tronc::mcp::server::McpServer;
use dravr_tronc::mcp::tool::{ToolContext, ToolRegistry};
use dravr_tronc::mcp::transport::http::mcp_router;
use embacle::types::{McpHeader, McpServerConfig, McpTransport, RunnerError};
use embacle::{McpToolDefinition, McpToolExecutor};
use rand::RngCore;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tracing::{debug, warn};

/// Header the session bearer travels in, matching what ACP forwards verbatim.
const AUTHORIZATION: &str = "authorization";

/// How the endpoint binds.
#[derive(Debug, Clone)]
pub struct ToolHostConfig {
    /// Emitted as `mcpServers[].name`, so it namespaces the tools the model
    /// sees. A tool `get_activities` under server `dravr` is reported by the
    /// agent as `dravr-get_activities`.
    pub server_name: String,
    /// Interface to bind. Loopback by default — widening it publishes the
    /// caller's entire tool surface to the network.
    pub bind_addr: IpAddr,
    /// `0` lets the kernel assign, which is what you want: no port to
    /// configure, and no collision between concurrent stacks on one host.
    pub port: u16,
    /// Natural-language instructions served at `initialize`.
    ///
    /// An agent that opts in — Copilot's CLI does so with
    /// `--allow-all-mcp-server-instructions` — folds these into its SYSTEM
    /// prompt. That matters when the caller has a persona to impose: a CLI
    /// runner with no system-prompt flag can only put one in the prompt body,
    /// where the model reads it as the user talking and answers as itself.
    /// This is the one channel that reaches the system layer.
    pub instructions: Option<String>,
}

impl Default for ToolHostConfig {
    fn default() -> Self {
        Self {
            server_name: "tools".to_owned(),
            bind_addr: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            instructions: None,
        }
    }
}

/// One tool call's outcome, in MCP's own shape.
///
/// Three states, not two. `Result` can say "it worked" or "it failed", but a
/// tool that RAN and declined — a quota refusal, a guard saying no, a provider
/// that needs reconnecting — is neither: the model should read the reason and
/// adapt, exactly as it reads a success. Collapsing that into `Err` throws away
/// the text the model needed; collapsing it into `Ok` tells the model it
/// succeeded.
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    /// Text handed to the model.
    pub text: String,
    /// Machine-readable payload mirrored into MCP `structuredContent`.
    pub structured: Option<Value>,
    /// MCP `isError`. True for a refusal the model should adapt to.
    pub is_error: bool,
}

impl ToolOutcome {
    /// A successful call carrying JSON. The text is the compact encoding, which
    /// is what a model reads when a server sends no separate rendering.
    #[must_use]
    pub fn json(value: Value) -> Self {
        Self {
            text: value.to_string(),
            structured: Some(value),
            is_error: false,
        }
    }

    /// A successful call carrying prose.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            structured: None,
            is_error: false,
        }
    }

    /// The tool ran and declined. `reason` is for the model, not the operator.
    #[must_use]
    pub fn refused(reason: impl Into<String>) -> Self {
        Self {
            text: reason.into(),
            structured: None,
            is_error: true,
        }
    }

    /// Attach structured content to a refusal, so a caller can carry a machine
    /// -readable code alongside the prose.
    #[must_use]
    pub fn with_structured(mut self, value: Value) -> Self {
        self.structured = Some(value);
        self
    }
}

/// The caller's tool surface, consulted per request.
///
/// Both halves are asked every time, deliberately. A caller whose visible set
/// is fixed for a turn can answer from a `Vec` and pay nothing; a caller whose
/// set depends on state that can change — a role, a quota, an interview in
/// progress that must withhold a tool until it ends — can answer from that
/// state at the moment the agent asks. Fixing the list at session open would
/// make the second kind unrepresentable, and a gate that cannot be re-asked is
/// a gate that silently stops applying.
#[async_trait]
pub trait ToolSurface: Send + Sync {
    /// Tools visible to this session right now.
    async fn list_tools(&self) -> Vec<McpToolDefinition>;

    /// Run one call. A tool absent from `list_tools` is already refused by the
    /// host, so this is only reached for a tool the surface just advertised.
    async fn call(&self, tool_name: &str, arguments: &Value) -> ToolOutcome;
}

/// A surface whose tool list never changes, backed by an [`McpToolExecutor`].
///
/// The simple case, kept simple: callers with nothing dynamic to say hand over
/// a `Vec` and an executor and are done.
///
/// # Fidelity
///
/// [`McpToolExecutor`] returns `Result<Value, RunnerError>`, which has two
/// states where a tool call has three. A tool that RAN and declined can only
/// come back as `Err`, so this adapter reports it as a refusal — correct — but
/// the only machine-readable thing an `Err` carries is its
/// [`ErrorKind`](embacle::types::ErrorKind), which is preserved as
/// `structuredContent.error_kind`. A caller that needs to hand the model a
/// richer refusal — an error code, a pending id, a provider to reconnect —
/// should implement [`ToolSurface`] directly and build its own
/// [`ToolOutcome`]. That is not a limitation of the host; it is the shape of
/// the narrower trait.
pub struct StaticSurface {
    tools: Vec<McpToolDefinition>,
    executor: Arc<dyn McpToolExecutor>,
}

impl StaticSurface {
    /// Wrap a fixed tool list and its executor.
    #[must_use]
    pub const fn new(tools: Vec<McpToolDefinition>, executor: Arc<dyn McpToolExecutor>) -> Self {
        Self { tools, executor }
    }
}

#[async_trait]
impl ToolSurface for StaticSurface {
    async fn list_tools(&self) -> Vec<McpToolDefinition> {
        self.tools.clone()
    }

    async fn call(&self, tool_name: &str, arguments: &Value) -> ToolOutcome {
        match self.executor.execute(tool_name, arguments).await {
            Ok(value) => ToolOutcome::json(value),
            // The kind is the only machine-readable thing a `RunnerError`
            // carries; dropping it would leave the model nothing but prose to
            // branch on.
            Err(e) => ToolOutcome::refused(e.message.clone())
                .with_structured(json!({ "error_kind": format!("{:?}", e.kind) })),
        }
    }
}

/// What one live session may see and run.
struct SessionState {
    /// Stable correlation key, also what the auth hook hands the dispatcher.
    id: String,
    /// Constant-time compared, so a wrong bearer cannot be recovered by timing.
    bearer: String,
    surface: Arc<dyn ToolSurface>,
    calls_served: AtomicU64,
}

/// Everything shared between the listener task and the session guards.
struct Inner {
    server_name: String,
    addr: SocketAddr,
    sessions: RwLock<HashMap<String, Arc<SessionState>>>,
    shutdown: RwLock<Option<oneshot::Sender<()>>>,
}

impl Inner {
    // Every lock here recovers from poisoning rather than propagating it. The
    // map's invariant is "these sessions are open", which a panic in unrelated
    // code cannot break — and treating a poisoned lock as failure would take
    // one caller's panic and turn it into every other session silently
    // refusing, which is a worse outcome than the panic.

    /// Resolve a bearer to its session, in constant time across candidates.
    ///
    /// A revoked session is simply absent, which is the whole revocation
    /// mechanism: the guard's `Drop` removes the entry.
    fn session_for(&self, bearer: &str) -> Option<Arc<SessionState>> {
        let sessions = self.sessions.read().unwrap_or_else(PoisonError::into_inner);
        sessions
            .values()
            .find(|s| s.bearer.as_bytes().ct_eq(bearer.as_bytes()).into())
            .map(Arc::clone)
    }
}

/// A bound loopback MCP endpoint. Cheap to clone.
#[derive(Clone)]
pub struct ToolHost {
    inner: Arc<Inner>,
}

impl ToolHost {
    /// Bind and start serving.
    ///
    /// Returns only once the listener is accepting, so [`Self::local_addr`] is
    /// valid the instant this returns — there is no window where a session's
    /// `mcpServers` entry names a port nothing answers on.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] when the bind fails.
    pub async fn bind(config: ToolHostConfig) -> Result<Self, RunnerError> {
        let listener = TcpListener::bind(SocketAddr::new(config.bind_addr, config.port))
            .await
            .map_err(|e| RunnerError::config(format!("tool host could not bind: {e}")))?;
        let addr = listener
            .local_addr()
            .map_err(|e| RunnerError::config(format!("tool host bound but has no address: {e}")))?;

        let instructions = config.instructions;
        let (tx, rx) = oneshot::channel();
        let inner = Arc::new(Inner {
            server_name: config.server_name,
            addr,
            sessions: RwLock::new(HashMap::new()),
            shutdown: RwLock::new(Some(tx)),
        });

        let mut server = McpServer::new(
            "embacle-tool-host",
            env!("CARGO_PKG_VERSION"),
            ToolRegistry::new(),
            Arc::clone(&inner),
        )
        .with_tool_dispatcher(Arc::new(Forwarding))
        .with_auth_hook(Arc::new(BearerSessions));
        if let Some(text) = instructions {
            server = server.with_instructions(text);
        }
        let server = Arc::new(server);

        let router = mcp_router(server);
        tokio::spawn(async move {
            let outcome = axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await;
            if let Err(e) = outcome {
                warn!(error = %e, "tool host listener stopped");
            }
        });

        debug!(%addr, "tool host listening");
        Ok(Self { inner })
    }

    /// The bound address, with the kernel-assigned port resolved.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.inner.addr
    }

    /// Open a turn-scoped session served by `surface`.
    ///
    /// The returned guard owns the session's lifetime. Hold it for exactly as
    /// long as the turn may legitimately call tools.
    #[must_use]
    pub fn open_session(&self, surface: Arc<dyn ToolSurface>) -> ToolSession {
        let session_id = uuid::Uuid::new_v4().to_string();
        let bearer = mint_bearer();
        let state = Arc::new(SessionState {
            id: session_id.clone(),
            bearer: bearer.clone(),
            surface,
            calls_served: AtomicU64::new(0),
        });
        self.inner
            .sessions
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(session_id.clone(), Arc::clone(&state));
        ToolSession {
            session_id,
            bearer,
            server_name: self.inner.server_name.clone(),
            addr: self.inner.addr,
            state,
            host: Arc::downgrade(&self.inner),
        }
    }

    /// Live sessions. A floor that keeps rising is a leaked guard.
    #[must_use]
    pub fn open_sessions(&self) -> usize {
        self.inner
            .sessions
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Stop the listener. Idempotent, and safe to call from a signal handler.
    pub fn shutdown(&self) {
        // Taken and released before sending: holding the lock across the send
        // would let a concurrent shutdown block on a lock this one still owns.
        let signal = self
            .inner
            .shutdown
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(tx) = signal {
            let _ = tx.send(());
        }
    }
}

/// 256 bits from the OS, hex-encoded.
fn mint_bearer() -> String {
    let mut raw = [0_u8; 32];
    rand::thread_rng().fill_bytes(&mut raw);
    raw.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// A turn's credential and tool surface.
///
/// Dropping this revokes the bearer. That is deliberate and is the reason the
/// guard exists rather than a plain id: a turn that ends — normally, by error,
/// or because the caller went away — must not leave a live credential that an
/// orphaned agent subprocess can still spend.
pub struct ToolSession {
    session_id: String,
    bearer: String,
    server_name: String,
    addr: SocketAddr,
    state: Arc<SessionState>,
    host: Weak<Inner>,
}

impl ToolSession {
    /// The `mcpServers` entry to hand the agent.
    ///
    /// A `Vec` because that is the shape `ChatRequest::with_mcp_servers` takes
    /// and a caller may be composing several servers.
    #[must_use]
    pub fn mcp_servers(&self) -> Vec<McpServerConfig> {
        vec![McpServerConfig {
            name: self.server_name.clone(),
            transport: McpTransport::Http {
                url: format!("http://{}/mcp", self.addr),
                headers: vec![McpHeader {
                    name: "Authorization".to_owned(),
                    value: format!("Bearer {}", self.bearer),
                }],
            },
        }]
    }

    /// Non-secret id for log correlation. The bearer is never exposed.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Tool calls served on this session.
    ///
    /// Zero on a turn whose reply claimed to have consulted data is the signal
    /// that it did not.
    #[must_use]
    pub fn calls_served(&self) -> u64 {
        self.state.calls_served.load(Ordering::SeqCst)
    }
}

impl Drop for ToolSession {
    fn drop(&mut self) {
        if let Some(inner) = self.host.upgrade() {
            inner
                .sessions
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&self.session_id);
        }
    }
}

/// Resolves the session bearer, and refuses everything else.
struct BearerSessions;

#[async_trait]
impl AuthHook<Inner> for BearerSessions {
    async fn authenticate(
        &self,
        request: &JsonRpcRequest,
        state: &Arc<Inner>,
    ) -> Result<ToolContext, AuthError> {
        let bearer = request
            .auth_token
            .as_deref()
            .or_else(|| {
                request
                    .headers
                    .as_ref()
                    .and_then(|h| h.get(AUTHORIZATION))
                    .and_then(Value::as_str)
            })
            .map(|raw| raw.trim_start_matches("Bearer ").trim())
            .unwrap_or_default();

        // A revoked or unknown bearer is absent from the map, so both fail the
        // same way and neither says which.
        state.session_for(bearer).map_or_else(
            || {
                Err(AuthError::Unauthorized {
                    www_authenticate: "Bearer".to_owned(),
                })
            },
            |session| Ok(ToolContext::new().with_request_id(Value::from(session.id.clone()))),
        )
    }
}

/// Forwards `tools/list` and `tools/call` into the caller's executor.
struct Forwarding;

#[async_trait]
impl ToolDispatcher<Inner> for Forwarding {
    async fn list_tools(&self, state: &Arc<Inner>, ctx: &ToolContext) -> Vec<Tool> {
        let Some(session) = resolve(state, ctx) else {
            return Vec::new();
        };
        // Asked now, not at session open: a caller gating on state that moves —
        // a role, a quota, an interview that must withhold a tool until it ends
        // — gets to answer for the moment the agent is asking about.
        session
            .surface
            .list_tools()
            .await
            .into_iter()
            .map(|t| Tool {
                name: t.name,
                description: t.description,
                input_schema: t.input_schema,
                annotations: None,
                output_schema: None,
            })
            .collect()
    }

    async fn call_tool(
        &self,
        name: &str,
        state: &Arc<Inner>,
        ctx: &ToolContext,
        arguments: Value,
    ) -> ToolResponse {
        let Some(session) = resolve(state, ctx) else {
            return ToolResponse::error("session is no longer open".to_owned());
        };

        // Re-check visibility at call time against the SAME live answer the
        // listing came from. A tool withheld between the agent's list and its
        // call must not run just because it was visible a moment ago.
        if !session
            .surface
            .list_tools()
            .await
            .iter()
            .any(|t| t.name == name)
        {
            return ToolResponse::error(format!("unknown tool: {name}"));
        }

        session.calls_served.fetch_add(1, Ordering::SeqCst);
        let outcome = session.surface.call(name, &arguments).await;
        let mut response = if outcome.is_error {
            ToolResponse::error(outcome.text)
        } else {
            ToolResponse::text(outcome.text)
        };
        response.structured_content = outcome.structured;
        response
    }
}

/// Recover the session a request authenticated as.
fn resolve(state: &Arc<Inner>, ctx: &ToolContext) -> Option<Arc<SessionState>> {
    let id = ctx.request_id.as_ref()?.as_str()?;
    let sessions = state
        .sessions
        .read()
        .unwrap_or_else(PoisonError::into_inner);
    sessions.get(id).map(Arc::clone)
}
