// ABOUTME: AG-UI (Agent-User Interaction) protocol event schema and emitter trait
// ABOUTME: Canonical event vocabulary + sink trait; transport-agnostic, feature-gated behind `agui`
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! AG-UI protocol events.
//!
//! Implements the event vocabulary from the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui),
//! an open standard for streaming agent progress to user-facing applications
//! (web, mobile, messaging). This module provides only the canonical type
//! definitions plus an [`AgUiEmitter`] trait; HTTP routing, SSE framing, and
//! pipeline integration are deliberately left to downstream crates so the
//! core embacle library stays transport-agnostic.
//!
//! # Event lifecycle
//!
//! ```text
//! RUN_STARTED
//!   STEP_STARTED("prefetch")    STEP_FINISHED("prefetch")
//!   STEP_STARTED("tool_loop")
//!     TOOL_CALL_START  TOOL_CALL_ARGS  TOOL_CALL_END  TOOL_CALL_RESULT
//!     TEXT_MESSAGE_START  TEXT_MESSAGE_CONTENT...  TEXT_MESSAGE_END
//!   STEP_FINISHED("tool_loop")
//! RUN_FINISHED  |  RUN_ERROR
//! ```
//!
//! # Design
//!
//! - Events serialize as JSON objects tagged by a `type` field using the
//!   AG-UI screaming-snake-case convention, so the wire format is
//!   bit-for-bit compatible with any AG-UI client (CopilotKit, ag-ui-js,
//!   Pydantic AI, custom SSE consumers).
//! - Every event carries a monotonic `timestamp` (milliseconds since the
//!   Unix epoch) and a `run_id` that correlates every event within a
//!   single agent run.
//! - The [`AgUiEmitter`] trait is async and accepts `&AgUiEvent` by
//!   reference so fan-out sinks can inspect events without cloning.
//!
//! # Example
//!
//! ```
//! use embacle::agui::{AgUiEvent, AgUiEventKind, AgUiEventFilter, NoopEmitter, AgUiEmitter};
//!
//! # async fn demo() {
//! let filter = AgUiEventFilter::default().without(AgUiEventKind::TextMessageContent);
//! let sink = NoopEmitter::new(filter);
//! let event = AgUiEvent::run_started("run_abc", Some("thread_xyz"));
//! sink.emit(&event).await;
//! # }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────
// Event enum — AG-UI wire format
// ─────────────────────────────────────────────────────────────────────────

/// AG-UI event emitted by an agent run.
///
/// Events are tagged by the `type` field using the AG-UI protocol's
/// screaming-snake-case naming (e.g. `RUN_STARTED`, `TEXT_MESSAGE_CONTENT`).
/// All variants carry a `timestamp` (ms since epoch) and a `run_id` for
/// correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgUiEvent {
    /// A new agent run has started.
    #[serde(rename = "RUN_STARTED")]
    RunStarted {
        /// Unique identifier for this run (UUID or opaque token).
        run_id: String,
        /// Conversation/thread the run belongs to.
        #[serde(skip_serializing_if = "Option::is_none")]
        thread_id: Option<String>,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// A logical step inside a run has started (e.g. prefetch, `tool_loop`).
    #[serde(rename = "STEP_STARTED")]
    StepStarted {
        /// Run identifier.
        run_id: String,
        /// Human-readable step name, e.g. `"prefetch"`, `"compaction"`, `"tool_loop"`.
        step_name: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// A logical step inside a run has finished.
    #[serde(rename = "STEP_FINISHED")]
    StepFinished {
        /// Run identifier.
        run_id: String,
        /// Step name that matches a prior `STEP_STARTED`.
        step_name: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// The run has completed successfully.
    #[serde(rename = "RUN_FINISHED")]
    RunFinished {
        /// Run identifier.
        run_id: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// The run has terminated with an error.
    #[serde(rename = "RUN_ERROR")]
    RunError {
        /// Run identifier.
        run_id: String,
        /// Short machine-readable error code.
        code: String,
        /// Human-readable error message.
        message: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Start of a streamed assistant text message.
    #[serde(rename = "TEXT_MESSAGE_START")]
    TextMessageStart {
        /// Run identifier.
        run_id: String,
        /// Unique identifier for the streamed message (used to correlate `CONTENT` chunks).
        message_id: String,
        /// Author role (typically `"assistant"`).
        role: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Streamed text delta for an open `TEXT_MESSAGE_START`.
    #[serde(rename = "TEXT_MESSAGE_CONTENT")]
    TextMessageContent {
        /// Run identifier.
        run_id: String,
        /// Message identifier that matches a prior `TEXT_MESSAGE_START`.
        message_id: String,
        /// Delta text to append. Clients should concatenate deltas in order.
        delta: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// End of a streamed assistant text message.
    #[serde(rename = "TEXT_MESSAGE_END")]
    TextMessageEnd {
        /// Run identifier.
        run_id: String,
        /// Message identifier that matches a prior `TEXT_MESSAGE_START`.
        message_id: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// The agent has decided to invoke a tool.
    #[serde(rename = "TOOL_CALL_START")]
    ToolCallStart {
        /// Run identifier.
        run_id: String,
        /// Unique identifier for this tool call (used to correlate `ARGS`/`END`/`RESULT`).
        tool_call_id: String,
        /// Name of the tool being invoked.
        tool_name: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Streamed tool-call argument delta (typically used when arguments are
    /// generated token-by-token by the LLM).
    #[serde(rename = "TOOL_CALL_ARGS")]
    ToolCallArgs {
        /// Run identifier.
        run_id: String,
        /// Tool call identifier that matches a prior `TOOL_CALL_START`.
        tool_call_id: String,
        /// Delta of the serialized JSON arguments.
        delta: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Tool invocation has completed (arguments fully formed, execution dispatched).
    #[serde(rename = "TOOL_CALL_END")]
    ToolCallEnd {
        /// Run identifier.
        run_id: String,
        /// Tool call identifier that matches a prior `TOOL_CALL_START`.
        tool_call_id: String,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Tool execution produced a result.
    #[serde(rename = "TOOL_CALL_RESULT")]
    ToolCallResult {
        /// Run identifier.
        run_id: String,
        /// Tool call identifier that matches a prior `TOOL_CALL_START`.
        tool_call_id: String,
        /// Result payload as an opaque JSON value (provider-specific).
        result: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Full snapshot of the agent's shared state.
    #[serde(rename = "STATE_SNAPSHOT")]
    StateSnapshot {
        /// Run identifier.
        run_id: String,
        /// Full state document as an opaque JSON value.
        snapshot: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Incremental state update as a JSON Patch (RFC 6902) document.
    #[serde(rename = "STATE_DELTA")]
    StateDelta {
        /// Run identifier.
        run_id: String,
        /// JSON Patch array describing the delta.
        delta: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Full snapshot of the current message list.
    #[serde(rename = "MESSAGES_SNAPSHOT")]
    MessagesSnapshot {
        /// Run identifier.
        run_id: String,
        /// Messages as an opaque JSON array.
        messages: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Raw, provider-specific event passthrough. Clients that understand the
    /// `source` may consume `payload`; generic clients should ignore it.
    #[serde(rename = "RAW")]
    Raw {
        /// Run identifier.
        run_id: String,
        /// Provider label, e.g. `"gemini"`, `"groq"`, `"copilot_headless"`.
        source: String,
        /// Raw payload as an opaque JSON value.
        payload: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Custom application-defined event. Useful for channel-specific
    /// affordances (e.g. Telegram typing indicator hints, web UI tooltips).
    #[serde(rename = "CUSTOM")]
    Custom {
        /// Run identifier.
        run_id: String,
        /// Custom event name.
        name: String,
        /// Arbitrary payload as an opaque JSON value.
        payload: Value,
        /// Milliseconds since the Unix epoch.
        timestamp: u64,
    },

    /// Forward-compat catch-all for wire-format events whose `type`
    /// field does not match any variant above.
    ///
    /// When a newer AG-UI implementation adds an event kind the current
    /// embacle release does not know about, deserialization still
    /// succeeds and produces this variant. Producers never emit it;
    /// consumers typically ignore it. The original `type` string and
    /// payload are dropped because serde's `#[serde(other)]` mandates a
    /// unit variant — callers that need the raw JSON should deserialize
    /// into `serde_json::Value` first and hand-dispatch.
    #[serde(other)]
    Unknown,
}

impl AgUiEvent {
    /// Return the discriminant kind without cloning the event.
    #[must_use]
    pub const fn kind(&self) -> AgUiEventKind {
        match self {
            Self::RunStarted { .. } => AgUiEventKind::RunStarted,
            Self::StepStarted { .. } => AgUiEventKind::StepStarted,
            Self::StepFinished { .. } => AgUiEventKind::StepFinished,
            Self::RunFinished { .. } => AgUiEventKind::RunFinished,
            Self::RunError { .. } => AgUiEventKind::RunError,
            Self::TextMessageStart { .. } => AgUiEventKind::TextMessageStart,
            Self::TextMessageContent { .. } => AgUiEventKind::TextMessageContent,
            Self::TextMessageEnd { .. } => AgUiEventKind::TextMessageEnd,
            Self::ToolCallStart { .. } => AgUiEventKind::ToolCallStart,
            Self::ToolCallArgs { .. } => AgUiEventKind::ToolCallArgs,
            Self::ToolCallEnd { .. } => AgUiEventKind::ToolCallEnd,
            Self::ToolCallResult { .. } => AgUiEventKind::ToolCallResult,
            Self::StateSnapshot { .. } => AgUiEventKind::StateSnapshot,
            Self::StateDelta { .. } => AgUiEventKind::StateDelta,
            Self::MessagesSnapshot { .. } => AgUiEventKind::MessagesSnapshot,
            Self::Raw { .. } => AgUiEventKind::Raw,
            Self::Custom { .. } => AgUiEventKind::Custom,
            Self::Unknown => AgUiEventKind::Unknown,
        }
    }

    /// Return the run identifier this event belongs to, or `""` for
    /// the forward-compat [`Self::Unknown`] variant which has no
    /// recoverable fields.
    #[must_use]
    pub fn run_id(&self) -> &str {
        match self {
            Self::RunStarted { run_id, .. }
            | Self::StepStarted { run_id, .. }
            | Self::StepFinished { run_id, .. }
            | Self::RunFinished { run_id, .. }
            | Self::RunError { run_id, .. }
            | Self::TextMessageStart { run_id, .. }
            | Self::TextMessageContent { run_id, .. }
            | Self::TextMessageEnd { run_id, .. }
            | Self::ToolCallStart { run_id, .. }
            | Self::ToolCallArgs { run_id, .. }
            | Self::ToolCallEnd { run_id, .. }
            | Self::ToolCallResult { run_id, .. }
            | Self::StateSnapshot { run_id, .. }
            | Self::StateDelta { run_id, .. }
            | Self::MessagesSnapshot { run_id, .. }
            | Self::Raw { run_id, .. }
            | Self::Custom { run_id, .. } => run_id,
            Self::Unknown => "",
        }
    }

    /// Convenience constructor for `RUN_STARTED`.
    #[must_use]
    pub fn run_started(run_id: impl Into<String>, thread_id: Option<&str>) -> Self {
        Self::RunStarted {
            run_id: run_id.into(),
            thread_id: thread_id.map(String::from),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `STEP_STARTED`.
    #[must_use]
    pub fn step_started(run_id: impl Into<String>, step_name: impl Into<String>) -> Self {
        Self::StepStarted {
            run_id: run_id.into(),
            step_name: step_name.into(),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `STEP_FINISHED`.
    #[must_use]
    pub fn step_finished(run_id: impl Into<String>, step_name: impl Into<String>) -> Self {
        Self::StepFinished {
            run_id: run_id.into(),
            step_name: step_name.into(),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `RUN_FINISHED`.
    #[must_use]
    pub fn run_finished(run_id: impl Into<String>) -> Self {
        Self::RunFinished {
            run_id: run_id.into(),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `RUN_ERROR`.
    #[must_use]
    pub fn run_error(
        run_id: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::RunError {
            run_id: run_id.into(),
            code: code.into(),
            message: message.into(),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `TOOL_CALL_START`.
    #[must_use]
    pub fn tool_call_start(
        run_id: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self::ToolCallStart {
            run_id: run_id.into(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            timestamp: now_ms(),
        }
    }

    /// Convenience constructor for `TOOL_CALL_RESULT`.
    #[must_use]
    pub fn tool_call_result(
        run_id: impl Into<String>,
        tool_call_id: impl Into<String>,
        result: Value,
    ) -> Self {
        Self::ToolCallResult {
            run_id: run_id.into(),
            tool_call_id: tool_call_id.into(),
            result,
            timestamp: now_ms(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Kind discriminant + filter config
// ─────────────────────────────────────────────────────────────────────────

/// Discriminant for [`AgUiEvent`]. Used by [`AgUiEventFilter`] to allow or
/// deny classes of events at emission time without cloning the payload.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AgUiEventKind {
    /// Corresponds to `AgUiEvent::RunStarted`.
    RunStarted,
    /// Corresponds to `AgUiEvent::StepStarted`.
    StepStarted,
    /// Corresponds to `AgUiEvent::StepFinished`.
    StepFinished,
    /// Corresponds to `AgUiEvent::RunFinished`.
    RunFinished,
    /// Corresponds to `AgUiEvent::RunError`.
    RunError,
    /// Corresponds to `AgUiEvent::TextMessageStart`.
    TextMessageStart,
    /// Corresponds to `AgUiEvent::TextMessageContent`.
    TextMessageContent,
    /// Corresponds to `AgUiEvent::TextMessageEnd`.
    TextMessageEnd,
    /// Corresponds to `AgUiEvent::ToolCallStart`.
    ToolCallStart,
    /// Corresponds to `AgUiEvent::ToolCallArgs`.
    ToolCallArgs,
    /// Corresponds to `AgUiEvent::ToolCallEnd`.
    ToolCallEnd,
    /// Corresponds to `AgUiEvent::ToolCallResult`.
    ToolCallResult,
    /// Corresponds to `AgUiEvent::StateSnapshot`.
    StateSnapshot,
    /// Corresponds to `AgUiEvent::StateDelta`.
    StateDelta,
    /// Corresponds to `AgUiEvent::MessagesSnapshot`.
    MessagesSnapshot,
    /// Corresponds to `AgUiEvent::Raw`.
    Raw,
    /// Corresponds to `AgUiEvent::Custom`.
    Custom,
    /// Corresponds to `AgUiEvent::Unknown` — forward-compat fallback
    /// for wire-format events whose `type` field is unknown to this
    /// embacle version.
    Unknown,
}

impl AgUiEventKind {
    /// Every kind, in stable declaration order.
    pub const ALL: [Self; 18] = [
        Self::RunStarted,
        Self::StepStarted,
        Self::StepFinished,
        Self::RunFinished,
        Self::RunError,
        Self::TextMessageStart,
        Self::TextMessageContent,
        Self::TextMessageEnd,
        Self::ToolCallStart,
        Self::ToolCallArgs,
        Self::ToolCallEnd,
        Self::ToolCallResult,
        Self::StateSnapshot,
        Self::StateDelta,
        Self::MessagesSnapshot,
        Self::Raw,
        Self::Custom,
        Self::Unknown,
    ];
}

/// Allowlist of event kinds that the emitter is permitted to forward.
///
/// Emission is configurable so operators can opt out of high-volume streams
/// (e.g. `TEXT_MESSAGE_CONTENT` deltas on a Telegram channel where per-token
/// updates cost more than they inform) without changing the producer code.
///
/// By default every kind is enabled — downstream deployments narrow the
/// allowlist via [`Self::only`], [`Self::with`], and [`Self::without`].
#[derive(Debug, Clone)]
pub struct AgUiEventFilter {
    allowed: HashSet<AgUiEventKind>,
}

impl AgUiEventFilter {
    /// Filter that allows every event kind.
    #[must_use]
    pub fn allow_all() -> Self {
        Self {
            allowed: AgUiEventKind::ALL.iter().copied().collect(),
        }
    }

    /// Filter that allows no events. Useful as a starting point for
    /// narrowly-scoped sinks.
    #[must_use]
    pub fn deny_all() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Filter that allows only the provided kinds.
    #[must_use]
    pub fn only<I: IntoIterator<Item = AgUiEventKind>>(kinds: I) -> Self {
        Self {
            allowed: kinds.into_iter().collect(),
        }
    }

    /// Add a kind to the allowlist, returning the updated filter.
    #[must_use]
    pub fn with(mut self, kind: AgUiEventKind) -> Self {
        self.allowed.insert(kind);
        self
    }

    /// Remove a kind from the allowlist, returning the updated filter.
    #[must_use]
    pub fn without(mut self, kind: AgUiEventKind) -> Self {
        self.allowed.remove(&kind);
        self
    }

    /// `true` when `kind` is permitted by this filter.
    #[must_use]
    pub fn allows(&self, kind: AgUiEventKind) -> bool {
        self.allowed.contains(&kind)
    }
}

impl Default for AgUiEventFilter {
    fn default() -> Self {
        Self::allow_all()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Emitter trait + no-op default sink
// ─────────────────────────────────────────────────────────────────────────

/// Sink for [`AgUiEvent`]s.
///
/// Implementations are expected to be cheap and non-blocking — the pipeline
/// emits from hot paths and cannot tolerate back-pressure. Broadcast-based
/// implementations that drop on full channels are preferred; see
/// `pierre_mcp_server::agui` in the `dravr-platform` repository for a
/// production SSE-backed sink.
#[async_trait]
pub trait AgUiEmitter: Send + Sync {
    /// Filter the sink applies before forwarding.
    fn filter(&self) -> &AgUiEventFilter;

    /// Forward an event to the sink. Implementations MUST consult
    /// `self.filter()` and drop disallowed events.
    async fn emit(&self, event: &AgUiEvent);
}

/// Emitter that discards every event it receives.
///
/// Useful as a default when a caller does not care about progress feedback
/// (unit tests, one-off CLI usage), or as a fallback when configuration
/// declines to wire a real sink.
#[derive(Debug, Clone, Default)]
pub struct NoopEmitter {
    filter: AgUiEventFilter,
}

impl NoopEmitter {
    /// Construct a no-op emitter carrying the given filter.
    #[must_use]
    pub fn new(filter: AgUiEventFilter) -> Self {
        Self { filter }
    }
}

#[async_trait]
impl AgUiEmitter for NoopEmitter {
    fn filter(&self) -> &AgUiEventFilter {
        &self.filter
    }

    async fn emit(&self, _event: &AgUiEvent) {}
}

// ─────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────

/// Milliseconds since the Unix epoch. Saturating on overflow and on system
/// clocks that report a time before the epoch; neither should happen in
/// practice but both are safer than panicking inside an event constructor.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
