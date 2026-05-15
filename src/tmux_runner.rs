// ABOUTME: LlmProvider that drives an interactive TUI CLI in a persistent tmux session per chat key
// ABOUTME: Uses sentinel injection (start/done markers) so the agent's own scrollback is the memory
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Tmux Runner
//!
//! Implements [`LlmProvider`] by holding a long-lived [`TmuxSession`] per
//! conversation key. The first `complete()` call for a key spawns the
//! configured CLI; later calls reuse the same session, so the agent's own
//! scrollback acts as conversation memory — no local history persistence
//! needed.
//!
//! ## Turn protocol
//!
//! Each turn injects two uuid sentinels:
//!
//! ```text
//! Begin your response with the exact literal token <<EMBACLE_START_{a}>>
//! on its own line, and end with <<EMBACLE_DONE_{b}>> on its own line.
//! ```
//!
//! The prompt is delivered with [`TmuxSession::paste`] (bracketed paste) then
//! submitted with `Enter`. [`TmuxSession::wait_for_marker`] polls until the
//! done marker is visible in the ANSI-stripped pane history. The response is
//! the substring between the two markers; if the agent omits the start
//! marker, the runner falls back to the snapshot-diff boundary.
//!
//! ## Conversation keys
//!
//! Sessions are keyed by [`ChatRequest::model`] — same convention as the
//! existing one-shot runners. Callers running a multi-tenant bot can encode
//! their chat-id into the model field (e.g. `claude::chat-12345`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{sleep, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, instrument, warn};

use crate::tmux_session::{strip_ansi, TmuxSession};
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError, StreamChunk,
};

/// Default session key used when a request omits a model identifier.
const DEFAULT_SESSION_KEY: &str = "default";

/// Default polling interval while waiting for the done marker (200 ms).
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Default hard timeout for a single turn (10 minutes).
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(600);

/// Default quiescence window for spawn readiness (800 ms of unchanged pane).
const DEFAULT_SPAWN_QUIET: Duration = Duration::from_millis(800);

/// Default maximum spawn-ready wait (30 seconds).
const DEFAULT_SPAWN_MAX: Duration = Duration::from_secs(30);

/// Supported tmux-driven backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TmuxBackend {
    /// Claude Code CLI (`claude`) — full-screen TUI agent.
    ClaudeCode,
    /// GitHub Copilot CLI (`copilot`) — full-screen TUI agent.
    Copilot,
}

/// Compile-time metadata for a backend: trait identifiers, session naming.
struct BackendDef {
    name: &'static str,
    display_name: &'static str,
    session_prefix: &'static str,
}

impl TmuxBackend {
    const fn def(self) -> &'static BackendDef {
        match self {
            Self::ClaudeCode => &BackendDef {
                name: "claude_code_tmux",
                display_name: "Claude Code (tmux)",
                session_prefix: "embacle-claude",
            },
            Self::Copilot => &BackendDef {
                name: "copilot_tmux",
                display_name: "GitHub Copilot (tmux)",
                session_prefix: "embacle-copilot",
            },
        }
    }
}

/// Configuration for a [`TmuxRunner`] instance.
///
/// Construct via [`Self::claude_code`] or [`Self::copilot`] and adjust
/// fields as needed before passing to [`TmuxRunner::new`].
#[derive(Debug, Clone)]
pub struct TmuxRunnerConfig {
    /// Which backend's launch / framing rules to apply.
    pub backend: TmuxBackend,
    /// Path to the CLI binary (e.g. `/usr/local/bin/claude`).
    pub binary_path: PathBuf,
    /// Extra arguments appended to the binary on spawn.
    pub launch_args: Vec<String>,
    /// How long the pane content must remain unchanged before the runner
    /// considers the TUI ready for input.
    pub spawn_quiet: Duration,
    /// Hard cap on how long to wait for spawn readiness before giving up.
    pub spawn_max: Duration,
    /// Hard cap on a single turn (paste → done marker).
    pub turn_timeout: Duration,
    /// Polling interval for marker / quiescence detection.
    pub poll_interval: Duration,
}

impl TmuxRunnerConfig {
    /// Default Claude Code configuration. `binary_path` should be the
    /// resolved path to the `claude` binary.
    #[must_use]
    pub fn claude_code(binary_path: PathBuf) -> Self {
        Self {
            backend: TmuxBackend::ClaudeCode,
            binary_path,
            launch_args: Vec::new(),
            spawn_quiet: DEFAULT_SPAWN_QUIET,
            spawn_max: DEFAULT_SPAWN_MAX,
            turn_timeout: DEFAULT_TURN_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }

    /// Default GitHub Copilot CLI configuration. `binary_path` should be the
    /// resolved path to the `copilot` binary.
    #[must_use]
    pub fn copilot(binary_path: PathBuf) -> Self {
        Self {
            backend: TmuxBackend::Copilot,
            binary_path,
            launch_args: Vec::new(),
            spawn_quiet: DEFAULT_SPAWN_QUIET,
            spawn_max: DEFAULT_SPAWN_MAX,
            turn_timeout: DEFAULT_TURN_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }
}

/// Generic tmux-driven runner. One instance manages many persistent
/// sessions keyed by [`ChatRequest::model`].
pub struct TmuxRunner {
    config: TmuxRunnerConfig,
    sessions: Arc<Mutex<HashMap<String, Arc<TmuxSession>>>>,
    available_models: Vec<String>,
}

impl TmuxRunner {
    /// Construct a runner from explicit configuration.
    #[must_use]
    pub fn new(config: TmuxRunnerConfig) -> Self {
        // The "model" field is reused as a conversation key here, not a real
        // model identifier; we expose an empty available-models list to make
        // that explicit at the trait level.
        Self {
            config,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            available_models: Vec::new(),
        }
    }

    /// Conversation keys with an active tmux session.
    pub async fn live_session_keys(&self) -> Vec<String> {
        self.sessions.lock().await.keys().cloned().collect()
    }

    /// Tear down the tmux session for `key`, if any. Returns whether a
    /// session was killed.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the tmux kill command fails.
    pub async fn close_session(&self, key: &str) -> Result<bool, RunnerError> {
        let removed = self.sessions.lock().await.remove(key);
        match removed {
            Some(session) => {
                session.kill().await?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Resolve the session-key for a request, falling back to a default when
    /// the caller did not supply one.
    fn session_key(request: &ChatRequest) -> &str {
        request
            .model
            .as_deref()
            .filter(|m| !m.is_empty())
            .unwrap_or(DEFAULT_SESSION_KEY)
    }

    /// Compose the tmux session name from the backend prefix and a sanitised
    /// conversation key.
    fn session_name(&self, key: &str) -> String {
        let sanitised: String = key
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        format!("{}-{sanitised}", self.config.backend.def().session_prefix)
    }

    /// Obtain (or lazily spawn) the tmux session for the given conversation
    /// key. The lock is held only for the get-or-insert step; the spawn
    /// itself happens after the lock is released to avoid serialising
    /// concurrent first-turns for distinct keys.
    async fn session_for(&self, key: &str) -> Result<Arc<TmuxSession>, RunnerError> {
        {
            let map = self.sessions.lock().await;
            if let Some(existing) = map.get(key) {
                return Ok(existing.clone());
            }
        }

        let name = self.session_name(key);
        let binary = self
            .config
            .binary_path
            .to_str()
            .ok_or_else(|| RunnerError::config("tmux runner binary_path is not valid UTF-8"))?;
        let args: Vec<&str> = self.config.launch_args.iter().map(String::as_str).collect();

        debug!(
            backend = ?self.config.backend,
            session = %name,
            binary,
            ?args,
            "Spawning tmux-driven CLI session"
        );

        let session = TmuxSession::spawn(&name, binary, &args).await?;
        self.wait_until_ready(&session).await?;
        let shared = Arc::new(session);

        let mut map = self.sessions.lock().await;
        // A concurrent caller may have inserted while we were spawning. If
        // so, drop our session (Drop will kill the duplicate tmux) and
        // return the winner.
        if let Some(existing) = map.get(key) {
            return Ok(existing.clone());
        }
        map.insert(key.to_owned(), shared.clone());
        Ok(shared)
    }

    /// Block until the session's pane content has been unchanged for
    /// `config.spawn_quiet`, or `config.spawn_max` elapses.
    async fn wait_until_ready(&self, session: &TmuxSession) -> Result<(), RunnerError> {
        let deadline = Instant::now() + self.config.spawn_max;
        let mut previous = session.capture(false).await?;
        let mut stable_since = Instant::now();

        loop {
            sleep(self.config.poll_interval).await;
            let current = session.capture(false).await?;
            if current == previous {
                if stable_since.elapsed() >= self.config.spawn_quiet {
                    return Ok(());
                }
            } else {
                previous = current;
                stable_since = Instant::now();
            }
            if Instant::now() >= deadline {
                warn!(
                    backend = ?self.config.backend,
                    session = %session.name(),
                    "Spawn quiescence not reached within spawn_max; proceeding anyway"
                );
                return Ok(());
            }
        }
    }

    /// Run a single turn: paste the prompt, send Enter, wait for the done
    /// marker, slice out the response.
    async fn run_turn(&self, session: &TmuxSession, prompt: &str) -> Result<String, RunnerError> {
        let start_id = uuid::Uuid::new_v4().simple().to_string();
        let done_id = uuid::Uuid::new_v4().simple().to_string();
        let start_marker = format!("<<EMBACLE_START_{start_id}>>");
        let done_marker = format!("<<EMBACLE_DONE_{done_id}>>");

        let framed = format!(
            "{prompt}\n\n[Reply protocol — required for embacle to detect end of turn: \
             begin your response with the literal token {start_marker} on its own line, \
             and end with the literal token {done_marker} on its own line. \
             Do not place the tokens inside code blocks or paraphrase them.]"
        );

        let pre_snapshot_len = strip_ansi(&session.capture(true).await?).len();

        session.paste(&framed).await?;
        session.send_enter().await?;

        let full = session
            .wait_for_marker(
                &done_marker,
                self.config.turn_timeout,
                self.config.poll_interval,
            )
            .await?;

        Ok(extract_response(
            &full,
            &start_marker,
            &done_marker,
            pre_snapshot_len,
        ))
    }
}

/// Slice the agent's response out of a captured pane history.
///
/// Preference order:
/// 1. Substring strictly between the **last** `start_marker` and the
///    `done_marker` — covers the well-behaved case where the agent followed
///    the framing protocol.
/// 2. If no start marker is found, fall back to the slice from
///    `pre_snapshot_len` (the pane length just before paste) up to the done
///    marker — covers the agent ignoring the start marker.
///
/// In both cases the start marker, the framing instructions echoed in the
/// user input, and trailing whitespace are stripped.
fn extract_response(
    full: &str,
    start_marker: &str,
    done_marker: &str,
    pre_snapshot_len: usize,
) -> String {
    let Some(done_idx) = full.rfind(done_marker) else {
        return full.trim().to_owned();
    };

    // Use the last occurrence of the start marker before the done marker so
    // that any echo of the framing instructions in the user input does not
    // shadow the agent's actual opening marker.
    let head = &full[..done_idx];
    let body_start = head.rfind(start_marker).map_or_else(
        || pre_snapshot_len.min(head.len()),
        |idx| idx + start_marker.len(),
    );

    head[body_start..].trim().to_owned()
}

#[async_trait]
impl LlmProvider for TmuxRunner {
    fn name(&self) -> &'static str {
        self.config.backend.def().name
    }

    fn display_name(&self) -> &str {
        self.config.backend.def().display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        // Multi-turn is implicit via persistent session; no per-request
        // temperature / max_tokens hook on a TUI agent.
        LlmCapabilities::STREAMING
    }

    fn default_model(&self) -> &str {
        DEFAULT_SESSION_KEY
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip_all, fields(runner = self.name()))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let key = Self::session_key(request).to_owned();
        let session = self.session_for(&key).await?;

        let prompt = compose_prompt(request)?;
        let content = self.run_turn(&session, &prompt).await?;

        Ok(ChatResponse {
            content,
            model: self.name().to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }

    #[instrument(skip_all, fields(runner = self.name()))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        // TUI agents do not expose token-level deltas through tmux; emit the
        // full response as a single final chunk so the LlmProvider streaming
        // contract still holds for callers that want a Stream interface.
        let response = self.complete(request).await?;
        let (tx, rx) = mpsc::channel(2);
        tx.send(Ok(StreamChunk {
            delta: response.content,
            is_final: true,
            finish_reason: Some("stop".to_owned()),
        }))
        .await
        .map_err(|e| RunnerError::internal(format!("Failed to enqueue stream chunk: {e}")))?;
        drop(tx);
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        // A live runner needs only a working tmux binary; binary_path
        // validity is checked lazily on first complete() to avoid forcing
        // every dependent to install the CLI just to instantiate the
        // runner.
        Ok(true)
    }
}

/// Flatten request messages into a single prompt block. System messages are
/// rendered as a `[System]` preamble; user / assistant turns are tagged so
/// the agent has context the first time around. On follow-up turns the
/// agent already has the prior turns in its own scrollback, so callers
/// typically pass just the new user message.
fn compose_prompt(request: &ChatRequest) -> Result<String, RunnerError> {
    use crate::types::MessageRole;

    if request.messages.is_empty() {
        return Err(RunnerError::config(
            "tmux runner requires at least one message",
        ));
    }
    let mut parts: Vec<String> = Vec::with_capacity(request.messages.len());
    for msg in &request.messages {
        if msg.content.is_empty() {
            continue;
        }
        let text = msg.content.as_str();
        match msg.role {
            MessageRole::System => parts.push(format!("[System]\n{text}")),
            MessageRole::User => parts.push(text.to_owned()),
            MessageRole::Assistant => parts.push(format!("[Assistant]\n{text}")),
            MessageRole::Tool => parts.push(format!("[Tool]\n{text}")),
        }
    }
    if parts.is_empty() {
        return Err(RunnerError::config(
            "tmux runner received only empty messages",
        ));
    }
    Ok(parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_response_uses_markers_when_present() {
        let full = "preamble user input\n<<EMBACLE_START_aaa>>\nactual reply line one\nline two\n<<EMBACLE_DONE_bbb>>";
        let out = extract_response(full, "<<EMBACLE_START_aaa>>", "<<EMBACLE_DONE_bbb>>", 0);
        assert_eq!(out, "actual reply line one\nline two");
    }

    #[test]
    fn extract_response_falls_back_to_snapshot_when_start_missing() {
        let full = "old scrollback...user prompt body...assistant reply text\n<<EMBACLE_DONE_xyz>>";
        let pre_len = "old scrollback...user prompt body...".len();
        let out = extract_response(
            full,
            "<<EMBACLE_START_nope>>",
            "<<EMBACLE_DONE_xyz>>",
            pre_len,
        );
        assert_eq!(out, "assistant reply text");
    }

    #[test]
    fn extract_response_skips_echoed_framing_via_last_occurrence() {
        // The framing instructions echo the start marker in the captured
        // user input area. The agent's real response starts at the LAST
        // start marker before done.
        let full = concat!(
            "user input: begin with <<EMBACLE_START_dup>> ... end with <<EMBACLE_DONE_dup>>\n",
            "<<EMBACLE_START_dup>>\n",
            "real reply\n",
            "<<EMBACLE_DONE_dup>>",
        );
        let out = extract_response(full, "<<EMBACLE_START_dup>>", "<<EMBACLE_DONE_dup>>", 0);
        assert_eq!(out, "real reply");
    }

    #[test]
    fn extract_response_returns_trimmed_full_when_done_missing() {
        let full = "no markers at all\n";
        let out = extract_response(full, "<<EMBACLE_START_x>>", "<<EMBACLE_DONE_x>>", 0);
        assert_eq!(out, "no markers at all");
    }

    #[test]
    fn session_name_sanitises_chat_keys() {
        let cfg = TmuxRunnerConfig::claude_code(PathBuf::from("/usr/bin/true"));
        let runner = TmuxRunner::new(cfg);
        let name = runner.session_name("chat:123/user@example.com");
        assert_eq!(name, "embacle-claude-chat-123-user-example-com");
    }

    #[test]
    fn session_key_falls_back_to_default_when_model_missing() {
        let req = ChatRequest::new(Vec::new());
        assert_eq!(TmuxRunner::session_key(&req), DEFAULT_SESSION_KEY);
    }
}
