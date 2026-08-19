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
use std::sync::{Arc, RwLock, Weak};

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
use serde_json::Value;
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
}

impl Default for ToolHostConfig {
    fn default() -> Self {
        Self {
            server_name: "tools".to_owned(),
            bind_addr: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
        }
    }
}

/// What one live session may see and run.
struct SessionState {
    /// Constant-time compared, so a wrong bearer cannot be recovered by timing.
    bearer: String,
    tools: Vec<McpToolDefinition>,
    executor: Arc<dyn McpToolExecutor>,
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
    /// Resolve a bearer to its session, in constant time across candidates.
    ///
    /// A revoked session is simply absent, which is the whole revocation
    /// mechanism: the guard's `Drop` removes the entry.
    fn session_for(&self, bearer: &str) -> Option<Arc<SessionState>> {
        let sessions = self.sessions.read().ok()?;
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

        let (tx, rx) = oneshot::channel();
        let inner = Arc::new(Inner {
            server_name: config.server_name,
            addr,
            sessions: RwLock::new(HashMap::new()),
            shutdown: RwLock::new(Some(tx)),
        });

        let server = Arc::new(
            McpServer::new(
                "embacle-tool-host",
                env!("CARGO_PKG_VERSION"),
                ToolRegistry::new(),
                Arc::clone(&inner),
            )
            .with_tool_dispatcher(Arc::new(Forwarding))
            .with_auth_hook(Arc::new(BearerSessions)),
        );

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

    /// Open a turn-scoped session over `tools`, run by `executor`.
    ///
    /// The returned guard owns the session's lifetime. Hold it for exactly as
    /// long as the turn may legitimately call tools.
    #[must_use]
    pub fn open_session(
        &self,
        tools: Vec<McpToolDefinition>,
        executor: Arc<dyn McpToolExecutor>,
    ) -> ToolSession {
        let session_id = uuid::Uuid::new_v4().to_string();
        let bearer = mint_bearer();
        let state = Arc::new(SessionState {
            bearer: bearer.clone(),
            tools,
            executor,
            calls_served: AtomicU64::new(0),
        });
        if let Ok(mut sessions) = self.inner.sessions.write() {
            sessions.insert(session_id.clone(), Arc::clone(&state));
        }
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
            .map_or(0, |sessions| sessions.len())
    }

    /// Stop the listener. Idempotent, and safe to call from a signal handler.
    pub fn shutdown(&self) {
        if let Ok(mut slot) = self.inner.shutdown.write() {
            if let Some(tx) = slot.take() {
                let _ = tx.send(());
            }
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
            if let Ok(mut sessions) = inner.sessions.write() {
                sessions.remove(&self.session_id);
            }
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
            |session| Ok(ToolContext::new().with_request_id(Value::from(session_tag(&session)))),
        )
    }
}

/// Stable non-secret tag for a session, used to route a call to its state.
fn session_tag(session: &Arc<SessionState>) -> String {
    format!("{:p}", Arc::as_ptr(session))
}

/// Forwards `tools/list` and `tools/call` into the caller's executor.
struct Forwarding;

#[async_trait]
impl ToolDispatcher<Inner> for Forwarding {
    async fn list_tools(&self, state: &Arc<Inner>, ctx: &ToolContext) -> Vec<Tool> {
        resolve(state, ctx).map_or_else(Vec::new, |session| {
            session
                .tools
                .iter()
                .map(|t| Tool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                    annotations: None,
                })
                .collect()
        })
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

        // Refuse a tool this session was never granted. The agent should not be
        // able to reach past the surface its caller published for this turn.
        if !session.tools.iter().any(|t| t.name == name) {
            return ToolResponse::error(format!("unknown tool: {name}"));
        }

        session.calls_served.fetch_add(1, Ordering::SeqCst);
        match session.executor.execute(name, &arguments).await {
            Ok(value) => ToolResponse::text(value.to_string()),
            // A refusal the model can adapt to, not a transport failure.
            Err(e) => ToolResponse::error(e.to_string()),
        }
    }
}

/// Recover the session a request authenticated as.
fn resolve(state: &Arc<Inner>, ctx: &ToolContext) -> Option<Arc<SessionState>> {
    let tag = ctx.request_id.as_ref()?.as_str()?;
    let sessions = state.sessions.read().ok()?;
    sessions
        .values()
        .find(|s| session_tag(s) == tag)
        .map(Arc::clone)
}
