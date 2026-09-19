// ABOUTME: Types and helpers shared by every Copilot provider: the permission policy, the turn API
// ABOUTME: (tool observations, aggregated responses, the streaming event), history rendering, token lookup.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! The turn API the Copilot SDK provider implements.
//!
//! `CopilotSdkRunner` reaches GitHub's Copilot runtime through
//! `github-copilot-sdk`, and a host speaks to it through this turn API.
//! Everything here is transport-neutral and compiles without the
//! `copilot-sdk` feature, so a consumer can name the trait and its types
//! whether or not it links the provider.

use std::env;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::Value;
use tokio_stream::Stream;

use crate::types::{ChatMessage, ChatRequest, LlmProvider, MessageRole, RunnerError, TokenUsage};

/// Policy for handling permission requests from the Copilot runtime.
///
/// Controls whether tool-execution permission prompts are auto-approved or denied.
///
/// **Denies by default.** The runtime's own tools — shell, git, file editing —
/// run with the host's environment and credentials, in the session working
/// directory (a scratch directory unless the host configures one), so a host
/// that builds its prompt from untrusted input (an end user's chat text, say)
/// turns an approval into arbitrary execution next to its secrets. That is not a
/// hypothetical: a Dravr coaching turn was observed making five `shell`/`bash`/
/// `Grep`/`Glob` calls during a user's request, because the default approved them.
///
/// A host that genuinely wants the runtime to run tools must now say so
/// explicitly with [`PermissionPolicy::AutoApprove`]. Every current consumer
/// wants denial, and the safe value should not depend on each one remembering to
/// set an env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionPolicy {
    /// Deny all permission requests by cancelling them.
    #[default]
    DenyAll,
    /// Automatically approve permission requests by selecting the best allow option.
    ///
    /// Only safe when the prompt is fully trusted — never when it is assembled
    /// from end-user input.
    AutoApprove,
}

impl PermissionPolicy {
    /// Parse the value of a `*_PERMISSION_POLICY` environment variable.
    ///
    /// Approval is opt-in and must be spelled out. An unset — or misspelled —
    /// value denies, so a typo degrades to the safe side rather than silently
    /// handing the runtime a shell.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        match raw.to_lowercase().as_str() {
            "auto_approve" | "autoapprove" | "approve" => Self::AutoApprove,
            _ => Self::DenyAll,
        }
    }
}

/// Default number of conversation history turns rendered into a prompt.
/// Each "turn" is one user or assistant message.
pub const DEFAULT_MAX_HISTORY_TURNS: usize = 20;

/// The GitHub token the Copilot runtime authenticates with, from the
/// environment, in the runtime's own precedence order:
/// `COPILOT_GITHUB_TOKEN`, then `GH_TOKEN`, then `GITHUB_TOKEN`.
#[must_use]
pub fn resolve_github_token() -> Option<String> {
    env::var("COPILOT_GITHUB_TOKEN")
        .or_else(|_| env::var("GH_TOKEN"))
        .or_else(|_| env::var("GITHUB_TOKEN"))
        .ok()
}

/// A tool call observed during a turn.
///
/// `id`, `title` and `status` are on every observation. `name`, `arguments`
/// and `result` are filled from the runtime's `tool.execution_start` and
/// `tool.execution_complete` events, so a host can persist the observation
/// as a tool round; each stays `None` until the event that carries it arrives.
#[derive(Debug, Clone, Default)]
pub struct ObservedToolCall {
    /// Tool call ID as the runtime assigned it.
    pub id: String,
    /// Human-readable title describing the tool action.
    pub title: String,
    /// Execution status (e.g., "Pending", "`InProgress`", "Completed", "Failed").
    pub status: String,
    /// The tool's registered name, when the transport carries it.
    pub name: Option<String>,
    /// The arguments the model passed, when the transport carries them.
    pub arguments: Option<Value>,
    /// The text result handed back to the model, when the transport carries it.
    pub result: Option<String>,
}

/// Response from a conversation turn including tool execution metadata.
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

/// Event emitted by [`HeadlessTurnProvider::converse_stream`] as the turn
/// progresses.
///
/// Unlike [`StreamChunk`](crate::types::StreamChunk) (which only carries text
/// deltas), this enum surfaces tool-call observations alongside text —
/// keeping the rich metadata that [`HeadlessTurnProvider::converse`] returns
/// at the end of the turn while delivering it incrementally.
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
    /// [`HeadlessTurnProvider::converse`] would have returned.
    /// Always emitted as the last event before the stream closes
    /// successfully.
    Done(HeadlessToolResponse),
}

/// Stream of [`HeadlessStreamEvent`]s for a single converse turn.
pub type HeadlessEventStream =
    Pin<Box<dyn Stream<Item = Result<HeadlessStreamEvent, RunnerError>> + Send>>;

/// A provider whose runtime executes tools itself during a turn.
///
/// [`LlmProvider::complete`] hides the tool activity inside a plain response;
/// these two methods expose it, so a host that hands the runtime an MCP
/// surface on each [`ChatRequest`] can observe what was called.
#[async_trait]
pub trait HeadlessTurnProvider: LlmProvider {
    /// Run a conversation turn and return detailed results including tool call metadata.
    async fn converse(&self, request: &ChatRequest) -> Result<HeadlessToolResponse, RunnerError>;

    /// Run a conversation turn, streaming text deltas and tool-call
    /// observations as they happen, ending with
    /// [`HeadlessStreamEvent::Done`].
    async fn converse_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<HeadlessEventStream, RunnerError>;
}

/// The system prompt of a request, if it carries one.
///
/// Only the first `System` message counts; the runtime has one system slot.
#[must_use]
pub fn system_prompt(request: &ChatRequest) -> Option<&str> {
    request
        .messages
        .iter()
        .find(|m| m.role == MessageRole::System)
        .map(|m| m.content.as_str())
}

/// A turn rendered for a provider that opens a fresh session per call.
#[derive(Debug)]
pub struct RenderedTurn<'a> {
    /// The prompt text: the history block, then the current user message.
    pub text: String,
    /// The user message the turn answers, for attachments the text cannot
    /// carry (images).
    pub last_user: Option<&'a ChatMessage>,
}

/// Render a request into one prompt for a fresh session.
///
/// The host owns the conversation, so every request carries the whole
/// history; a session that starts empty gets the prior turns serialized into
/// a `<conversation-history>` block ahead of the current user message, capped
/// to the most recent `max_history_turns` messages (`0` disables history).
/// The system prompt is not part of the text: the runtime has a system slot of
/// its own, which [`system_prompt`] feeds, so it is never sent twice.
#[must_use]
pub fn render_turn(request: &ChatRequest, max_history_turns: usize) -> RenderedTurn<'_> {
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

    // Keep only the most recent turns.
    let truncated_history = if max_history_turns == 0 || history.is_empty() {
        &[][..]
    } else if history.len() > max_history_turns {
        &history[history.len() - max_history_turns..]
    } else {
        &history
    };

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

    let mut text = history_block;
    text.push_str(user_text);

    RenderedTurn { text, last_user }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest::new(messages)
    }

    #[test]
    fn permission_policy_parse_denies_unless_spelled_out() {
        assert_eq!(PermissionPolicy::parse(""), PermissionPolicy::DenyAll);
        assert_eq!(
            PermissionPolicy::parse("deny_all"),
            PermissionPolicy::DenyAll
        );
        assert_eq!(
            PermissionPolicy::parse("auto-approve"),
            PermissionPolicy::DenyAll
        );
        assert_eq!(
            PermissionPolicy::parse("auto_approve"),
            PermissionPolicy::AutoApprove
        );
        assert_eq!(
            PermissionPolicy::parse("APPROVE"),
            PermissionPolicy::AutoApprove
        );
    }

    #[test]
    fn render_puts_history_ahead_of_the_current_user_message() {
        let req = request(vec![
            ChatMessage::system("SYS"),
            ChatMessage::user("first"),
            ChatMessage::assistant("reply"),
            ChatMessage::user("second"),
        ]);
        let turn = render_turn(&req, DEFAULT_MAX_HISTORY_TURNS);
        assert_eq!(
            turn.text,
            "<conversation-history>\nUser: first\nAssistant: reply\n</conversation-history>\n\nsecond"
        );
        assert_eq!(turn.last_user.map(|m| m.content.as_str()), Some("second"));
    }

    #[test]
    fn render_leaves_the_system_slot_to_the_transport() {
        let req = request(vec![ChatMessage::system("SYS"), ChatMessage::user("hi")]);
        let turn = render_turn(&req, DEFAULT_MAX_HISTORY_TURNS);
        assert_eq!(turn.text, "hi");
        assert_eq!(system_prompt(&req), Some("SYS"));
    }

    #[test]
    fn render_caps_history_to_the_most_recent_turns() {
        let req = request(vec![
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
            ChatMessage::assistant("a2"),
            ChatMessage::user("u3"),
        ]);
        let turn = render_turn(&req, 2);
        assert_eq!(
            turn.text,
            "<conversation-history>\nUser: u2\nAssistant: a2\n</conversation-history>\n\nu3"
        );
        let none = render_turn(&req, 0);
        assert_eq!(none.text, "u3");
    }

    #[test]
    fn render_with_no_user_message_yields_history_only() {
        let req = request(vec![ChatMessage::assistant("a1")]);
        let turn = render_turn(&req, DEFAULT_MAX_HISTORY_TURNS);
        assert_eq!(
            turn.text,
            "<conversation-history>\nAssistant: a1\n</conversation-history>\n\n"
        );
        assert!(turn.last_user.is_none());
    }
}
