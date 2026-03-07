// ABOUTME: Prompt construction from ChatMessage sequences for CLI invocations
// ABOUTME: Extracts system messages and builds role-prefixed prompt strings
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use tracing::debug;

use crate::types::{ChatMessage, MessageRole};

/// Build a single prompt string from a slice of chat messages
///
/// Each message is prefixed with its role label (`[system]`, `[user]`,
/// `[assistant]`) followed by the content. Messages are separated by
/// double newlines.
#[must_use]
pub fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(messages.len());
    for msg in messages {
        let label = match msg.role {
            MessageRole::System => "[system]",
            MessageRole::User => "[user]",
            MessageRole::Assistant => "[assistant]",
            MessageRole::Tool => "[tool]",
        };
        parts.push(format!("{label}\n{}", msg.content));
    }
    let prompt = parts.join("\n\n");
    debug!(
        message_count = messages.len(),
        prompt_len = prompt.len(),
        has_system = messages.iter().any(|m| m.role == MessageRole::System),
        "Built prompt from messages"
    );
    prompt
}

/// Extract the content of the first system message, if any
#[must_use]
pub fn extract_system_message(messages: &[ChatMessage]) -> Option<&str> {
    let result = messages
        .iter()
        .find(|m| m.role == MessageRole::System)
        .map(|m| m.content.as_str());
    debug!(
        found = result.is_some(),
        len = result.map_or(0, str::len),
        "Extracting system message"
    );
    result
}

/// Build a prompt string from non-system messages only
///
/// Useful when the CLI tool accepts a separate `--system-prompt` flag
/// and the system message should not be mixed into the user prompt.
#[must_use]
pub fn build_user_prompt(messages: &[ChatMessage]) -> String {
    let non_system: Vec<&ChatMessage> = messages
        .iter()
        .filter(|m| m.role != MessageRole::System)
        .collect();

    let mut parts: Vec<String> = Vec::with_capacity(non_system.len());
    for msg in &non_system {
        let label = match msg.role {
            MessageRole::User => "[user]",
            MessageRole::Assistant => "[assistant]",
            MessageRole::Tool => "[tool]",
            MessageRole::System => unreachable!(),
        };
        parts.push(format!("{label}\n{}", msg.content));
    }
    let prompt = parts.join("\n\n");
    debug!(
        total_messages = messages.len(),
        user_messages = non_system.len(),
        prompt_len = prompt.len(),
        "Built user prompt (system messages excluded)"
    );
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_single_user_message() {
        let messages = vec![ChatMessage::user("Hello")];
        assert_eq!(build_prompt(&messages), "[user]\nHello");
    }

    #[test]
    fn test_build_prompt_multi_role_conversation() {
        let messages = vec![
            ChatMessage::system("Be concise"),
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("A systems language."),
        ];
        let result = build_prompt(&messages);
        assert_eq!(
            result,
            "[system]\nBe concise\n\n[user]\nWhat is Rust?\n\n[assistant]\nA systems language."
        );
    }

    #[test]
    fn test_build_prompt_empty_messages() {
        let messages: Vec<ChatMessage> = Vec::new();
        assert_eq!(build_prompt(&messages), "");
    }

    #[test]
    fn test_extract_system_message_present() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hi"),
        ];
        assert_eq!(extract_system_message(&messages), Some("You are helpful"));
    }

    #[test]
    fn test_extract_system_message_absent() {
        let messages = vec![ChatMessage::user("Hi")];
        assert_eq!(extract_system_message(&messages), None);
    }

    #[test]
    fn test_extract_system_message_returns_first() {
        let messages = vec![ChatMessage::system("First"), ChatMessage::system("Second")];
        assert_eq!(extract_system_message(&messages), Some("First"));
    }

    #[test]
    fn test_build_user_prompt_excludes_system() {
        let messages = vec![
            ChatMessage::system("System instructions"),
            ChatMessage::user("User question"),
            ChatMessage::assistant("Response"),
        ];
        let result = build_user_prompt(&messages);
        assert_eq!(result, "[user]\nUser question\n\n[assistant]\nResponse");
        assert!(!result.contains("[system]"));
    }

    #[test]
    fn test_build_user_prompt_only_system_messages() {
        let messages = vec![ChatMessage::system("Only system")];
        assert_eq!(build_user_prompt(&messages), "");
    }
}
