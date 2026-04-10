// ABOUTME: MCP tool that dispatches chat prompts to the active embacle LLM provider
// ABOUTME: Supports single-provider and multiplex modes for concurrent multi-provider queries
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, MessageRole};
use serde_json::{json, Value};

use dravr_tronc::mcp::protocol::{CallToolResult, ToolDefinition};
use dravr_tronc::McpTool;

use crate::runner::multiplex::MultiplexEngine;
use crate::state::{ServerState, SharedState};

/// Dispatches a chat prompt to the active provider or fans out via multiplex
pub struct Prompt;

#[async_trait]
impl McpTool<ServerState> for Prompt {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "prompt".to_owned(),
            description:
                "Send a chat prompt to the active LLM provider, or multiplex to all configured providers"
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
                                },
                                "images": {
                                    "type": "array",
                                    "description": "Optional images attached to the message (user role only)",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "data": {
                                                "type": "string",
                                                "description": "Base64-encoded image data"
                                            },
                                            "mime_type": {
                                                "type": "string",
                                                "description": "MIME type (image/png, image/jpeg, image/webp, image/gif)"
                                            }
                                        },
                                        "required": ["data", "mime_type"]
                                    }
                                }
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "multiplex": {
                        "type": "boolean",
                        "description": "If true, send to all multiplex providers instead of the active one",
                        "default": false
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

        let multiplex = arguments
            .get("multiplex")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        if multiplex {
            execute_multiplex(state, &messages).await
        } else {
            execute_single(state, &messages).await
        }
    }
}

/// Execute a prompt against the single active provider
async fn execute_single(state: &SharedState, messages: &[ChatMessage]) -> CallToolResult {
    let state_guard = state.read().await;
    let provider = state_guard.active_provider();
    let runner = match state_guard.get_runner(provider).await {
        Ok(r) => r,
        Err(e) => {
            return CallToolResult::error(format!("Failed to create runner: {e}"));
        }
    };
    let model = state_guard.active_model().map(ToOwned::to_owned);
    drop(state_guard);

    let mut request = ChatRequest::new(messages.to_vec());
    if let Some(m) = model {
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

/// Execute a prompt against all configured multiplex providers
async fn execute_multiplex(state: &SharedState, messages: &[ChatMessage]) -> CallToolResult {
    let providers = {
        let state_guard = state.read().await;
        state_guard.multiplex_providers().to_vec()
    };

    if providers.is_empty() {
        return CallToolResult::error(
            "No multiplex providers configured. Use set_multiplex_provider first.".to_owned(),
        );
    }

    let engine = MultiplexEngine::new(Arc::clone(state));
    match engine.execute(messages, &providers).await {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(json) => CallToolResult::text(json),
            Err(e) => CallToolResult::error(format!("Result serialization failed: {e}")),
        },
        Err(e) => CallToolResult::error(format!("Multiplex error: {e}")),
    }
}

/// Parse image objects from a message's "images" array
fn parse_images(msg: &Value, index: usize) -> Result<Option<Vec<embacle::ImagePart>>, String> {
    let Some(arr) = msg.get("images").and_then(Value::as_array) else {
        return Ok(None);
    };

    if arr.is_empty() {
        return Ok(None);
    }

    let mut images = Vec::with_capacity(arr.len());
    for (j, img_val) in arr.iter().enumerate() {
        let data = img_val
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Message {index}, image {j}: missing 'data'"))?;
        let mime_type = img_val
            .get("mime_type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Message {index}, image {j}: missing 'mime_type'"))?;

        let part = embacle::ImagePart::new(data, mime_type)
            .map_err(|e| format!("Message {index}, image {j}: {e}"))?;
        images.push(part);
    }

    Ok(Some(images))
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

        let images = parse_images(msg, i)?;
        let mut message = ChatMessage::new(role, content);
        message.images = images;
        messages.push(message);
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
        let msgs = parse_messages(&args).expect("should parse"); // Safe: test assertion
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

    #[test]
    fn parse_messages_with_images() {
        let args = json!({
            "messages": [{
                "role": "user",
                "content": "Describe this",
                "images": [{
                    "data": "aGVsbG8=",
                    "mime_type": "image/png"
                }]
            }]
        });
        let msgs = parse_messages(&args).expect("should parse"); // Safe: test assertion
        assert_eq!(msgs.len(), 1);
        let images = msgs[0].images.as_ref().expect("images present"); // Safe: test assertion
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/png");
        assert_eq!(images[0].data, "aGVsbG8=");
    }

    #[test]
    fn parse_messages_without_images() {
        let args = json!({
            "messages": [{"role": "user", "content": "Hello!"}]
        });
        let msgs = parse_messages(&args).expect("should parse"); // Safe: test assertion
        assert!(msgs[0].images.is_none());
    }

    #[test]
    fn parse_messages_invalid_mime_type() {
        let args = json!({
            "messages": [{
                "role": "user",
                "content": "Describe",
                "images": [{"data": "abc", "mime_type": "image/bmp"}]
            }]
        });
        let err = parse_messages(&args).unwrap_err();
        assert!(err.contains("image/bmp"));
    }
}
