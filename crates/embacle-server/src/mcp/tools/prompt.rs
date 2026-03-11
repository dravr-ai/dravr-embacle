// ABOUTME: MCP tool that dispatches chat prompts via stateless provider routing
// ABOUTME: Uses the same model-string resolution as the OpenAI-compatible endpoints
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, MessageRole};
use serde_json::{json, Value};

use crate::mcp::protocol::{CallToolResult, ToolDefinition};
use crate::mcp::tools::McpTool;
use crate::provider_resolver;
use crate::state::SharedState;

/// Dispatches a chat prompt to a provider resolved from the model string
pub struct Prompt;

#[async_trait]
impl McpTool for Prompt {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "prompt".to_owned(),
            description:
                "Send a chat prompt to an LLM provider. Use the model field to route to a \
                 specific provider (e.g. \"copilot:gpt-4o\", \"claude:opus\")."
                    .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Chat messages to send to the provider",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"]
                                },
                                "content": {
                                    "type": "string"
                                }
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "model": {
                        "type": "string",
                        "description": "Provider and model (e.g. \"copilot:gpt-4o\", \"claude\"). Defaults to server's default provider."
                    }
                },
                "required": ["messages"]
            }),
        }
    }

    async fn execute(&self, state: &SharedState, arguments: Value) -> CallToolResult {
        let messages = match parse_messages(&arguments) {
            Ok(msgs) => msgs,
            Err(e) => return CallToolResult::error(e),
        };

        let resolved = arguments.get("model").and_then(Value::as_str).map_or_else(
            || provider_resolver::ResolvedProvider {
                runner_type: state.default_provider(),
                model: None,
            },
            |model_str| provider_resolver::resolve_model(model_str, state.default_provider()),
        );

        let runner = match state.get_runner(resolved.runner_type).await {
            Ok(r) => r,
            Err(e) => return CallToolResult::error(format!("Failed to create runner: {e}")),
        };

        let mut request = ChatRequest::new(messages);
        if let Some(m) = resolved.model {
            request = request.with_model(m);
        }

        match runner.complete(&request).await {
            Ok(response) => match serde_json::to_string_pretty(&response) {
                Ok(json) => CallToolResult::text(json),
                Err(e) => CallToolResult::error(format!("Response serialization failed: {e}")),
            },
            Err(e) => CallToolResult::error(format!("Completion error: {e}")),
        }
    }
}

/// Parse chat messages from the MCP tool arguments JSON
fn parse_messages(arguments: &Value) -> Result<Vec<ChatMessage>, String> {
    let arr = arguments
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "Missing or invalid 'messages' array".to_owned())?;

    let mut messages = Vec::with_capacity(arr.len());
    for (i, msg) in arr.iter().enumerate() {
        let role_str = msg
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Message {i}: missing 'role'"))?;

        let content = msg
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Message {i}: missing 'content'"))?;

        let role = match role_str {
            "system" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            other => return Err(format!("Message {i}: invalid role '{other}'")),
        };

        messages.push(ChatMessage::new(role, content));
    }

    if messages.is_empty() {
        return Err("Messages array must not be empty".to_owned());
    }

    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_messages() {
        let args = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
        });
        let msgs = parse_messages(&args).expect("should parse");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::System);
        assert_eq!(msgs[1].content, "Hello!");
    }

    #[test]
    fn parse_empty_messages_rejected() {
        let args = json!({"messages": []});
        assert!(parse_messages(&args).is_err());
    }

    #[test]
    fn parse_missing_role_rejected() {
        let args = json!({"messages": [{"content": "hi"}]});
        assert!(parse_messages(&args).is_err());
    }

    #[test]
    fn parse_invalid_role_rejected() {
        let args = json!({"messages": [{"role": "bot", "content": "hi"}]});
        let err = parse_messages(&args).unwrap_err();
        assert!(err.contains("invalid role"));
    }
}
