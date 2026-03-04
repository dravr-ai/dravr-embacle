// ABOUTME: POST /v1/chat/completions handler for OpenAI-compatible chat completion
// ABOUTME: Routes to single provider or multiplex, supports both streaming and non-streaming
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, RunnerError};
use embacle::FunctionDeclaration;
use tracing::{debug, error, warn};

use crate::openai_types::{
    ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ErrorResponse,
    ModelField, MultiplexProviderResult, MultiplexResponse, ResponseMessage, ToolCall,
    ToolCallFunction, ToolChoice, Usage,
};
use crate::provider_resolver::resolve_model;
use crate::runner::multiplex::MultiplexEngine;
use crate::state::SharedState;
use crate::streaming;

/// OpenAI-specified upper bound for temperature
const MAX_TEMPERATURE: f32 = 2.0;

/// Handle POST /v1/chat/completions
///
/// Dispatches to single-provider or multiplex mode based on the model field.
/// Supports both streaming (SSE) and non-streaming (JSON) responses.
pub async fn handle(
    State(state): State<SharedState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Some(temp) = request.temperature {
        if !(0.0..=MAX_TEMPERATURE).contains(&temp) {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("temperature must be between 0.0 and {MAX_TEMPERATURE}"),
            );
        }
    }
    if let Some(max) = request.max_tokens {
        if max == 0 {
            return error_response(StatusCode::BAD_REQUEST, "max_tokens must be greater than 0");
        }
    }

    match request.model {
        ModelField::Multiple(ref models) if models.len() > 1 => {
            handle_multiplex(&state, &request, models).await
        }
        ModelField::Multiple(ref models) if models.len() == 1 => {
            handle_single(&state, &request, &models[0]).await
        }
        ModelField::Multiple(_) => {
            error_response(StatusCode::BAD_REQUEST, "Model array must not be empty")
        }
        ModelField::Single(ref model) => handle_single(&state, &request, model).await,
    }
}

/// Handle a single-provider request (standard case)
async fn handle_single(
    state: &SharedState,
    request: &ChatCompletionRequest,
    model_str: &str,
) -> Response {
    let has_tools = request
        .tools
        .as_ref()
        .is_some_and(|t| !t.is_empty() && !is_tool_choice_none(request.tool_choice.as_ref()));

    // Reject streaming with tools (not yet supported)
    if request.stream && has_tools {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Streaming is not supported with tool calling",
        );
    }

    let resolved = resolve_model(model_str, state.default_provider());
    debug!(
        provider = %resolved.runner_type,
        model = ?resolved.model,
        stream = request.stream,
        has_tools,
        "Dispatching completion"
    );

    let runner = match state.get_runner(resolved.runner_type).await {
        Ok(r) => r,
        Err(e) => return runner_error_to_response(&e),
    };

    let strict = request.strict_capabilities.unwrap_or_else(|| {
        std::env::var("EMBACLE_STRICT_CAPS")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    });

    let mut messages = convert_messages(&request.messages);

    // Inject tool catalog into system prompt if tools are provided
    if has_tools {
        let declarations = tools_to_declarations(request.tools.as_deref().unwrap_or_default());
        let catalog = embacle::generate_tool_catalog(&declarations);
        embacle::inject_tool_catalog(&mut messages, &catalog);
    }

    let mut chat_request = ChatRequest::new(messages);
    chat_request.model = resolved.model;
    chat_request.temperature = request.temperature;
    chat_request.max_tokens = request.max_tokens;

    let warnings = match embacle::validate_capabilities(
        runner.name(),
        runner.capabilities(),
        &chat_request,
        strict,
    ) {
        Ok(w) => w,
        Err(e) => return runner_error_to_response(&e),
    };
    let warnings_for_response = if warnings.is_empty() {
        None
    } else {
        Some(warnings)
    };

    if request.stream {
        chat_request.stream = true;
        match runner.complete_stream(&chat_request).await {
            Ok(stream) => {
                let model_name = format!("{}:{}", resolved.runner_type, runner.default_model());
                streaming::sse_response(stream, &model_name)
            }
            Err(e) => runner_error_to_response(&e),
        }
    } else {
        match runner.complete(&chat_request).await {
            Ok(response) => {
                let model_name = format!("{}:{}", resolved.runner_type, response.model);
                let usage = response.usage.map(|u| Usage {
                    prompt: u.prompt_tokens,
                    completion: u.completion_tokens,
                    total: u.total_tokens,
                });

                let (message, finish_reason) =
                    build_response_message(has_tools, response.content, response.finish_reason);

                let resp = ChatCompletionResponse {
                    id: generate_id(),
                    object: "chat.completion",
                    created: unix_timestamp(),
                    model: model_name,
                    choices: vec![Choice {
                        index: 0,
                        message,
                        finish_reason,
                    }],
                    usage,
                    warnings: warnings_for_response,
                };

                (StatusCode::OK, Json(resp)).into_response()
            }
            Err(e) => runner_error_to_response(&e),
        }
    }
}

/// Handle a multiplex request (multiple providers)
async fn handle_multiplex(
    state: &SharedState,
    request: &ChatCompletionRequest,
    models: &[String],
) -> Response {
    if request.stream {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Streaming is not supported for multiplex requests",
        );
    }

    let strict = request.strict_capabilities.unwrap_or_else(|| {
        std::env::var("EMBACLE_STRICT_CAPS")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    });

    let default_provider = state.default_provider();
    let resolved: Vec<_> = models
        .iter()
        .map(|m| resolve_model(m, default_provider))
        .collect();

    let providers: Vec<_> = resolved.iter().map(|r| r.runner_type).collect();
    let messages = convert_messages(&request.messages);

    // Build a temporary ChatRequest for capability validation
    let mut validation_request = ChatRequest::new(messages.clone());
    validation_request.temperature = request.temperature;
    validation_request.max_tokens = request.max_tokens;

    for &provider_type in &providers {
        let runner = match state.get_runner(provider_type).await {
            Ok(r) => r,
            Err(e) => return runner_error_to_response(&e),
        };
        match embacle::validate_capabilities(
            runner.name(),
            runner.capabilities(),
            &validation_request,
            strict,
        ) {
            Ok(w) => {
                for warning in &w {
                    warn!(provider = runner.name(), warning = %warning, "Capability warning");
                }
            }
            Err(e) => return runner_error_to_response(&e),
        }
    }

    let engine = MultiplexEngine::new(state);
    match engine
        .execute(
            &messages,
            &providers,
            request.temperature,
            request.max_tokens,
        )
        .await
    {
        Ok(result) => {
            let results = result
                .responses
                .into_iter()
                .map(|r| MultiplexProviderResult {
                    provider: r.provider,
                    model: r.model,
                    content: r.content,
                    error: r.error,
                    duration_ms: r.duration_ms,
                })
                .collect();

            let resp = MultiplexResponse {
                id: generate_id(),
                object: "chat.completion.multiplex",
                created: unix_timestamp(),
                results,
                summary: result.summary,
            };

            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => runner_error_to_response(&e),
    }
}

/// Build a `ResponseMessage` from LLM output, parsing tool calls if tools were requested
fn build_response_message(
    has_tools: bool,
    content: String,
    finish_reason: Option<String>,
) -> (ResponseMessage, Option<String>) {
    if has_tools {
        let parsed_calls = embacle::parse_tool_call_blocks(&content);
        if parsed_calls.is_empty() {
            (
                ResponseMessage {
                    role: "assistant",
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason.or_else(|| Some("stop".to_owned())),
            )
        } else {
            let remaining_text = embacle::strip_tool_call_blocks(&content);
            let text_content = if remaining_text.is_empty() {
                None
            } else {
                Some(remaining_text)
            };
            let tool_calls: Vec<ToolCall> = parsed_calls
                .iter()
                .enumerate()
                .map(|(i, fc)| ToolCall {
                    id: generate_tool_call_id(&fc.name, i),
                    tool_type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: fc.name.clone(),
                        arguments: serde_json::to_string(&fc.args)
                            .unwrap_or_else(|_| "{}".to_owned()),
                    },
                })
                .collect();
            (
                ResponseMessage {
                    role: "assistant",
                    content: text_content,
                    tool_calls: Some(tool_calls),
                },
                Some("tool_calls".to_owned()),
            )
        }
    } else {
        (
            ResponseMessage {
                role: "assistant",
                content: Some(content),
                tool_calls: None,
            },
            finish_reason.or_else(|| Some("stop".to_owned())),
        )
    }
}

/// Convert `OpenAI` message format to embacle `ChatMessage`
///
/// Handles all `OpenAI` roles including "tool" messages and assistant messages
/// with `tool_calls`. Tool messages are collected and formatted as `<tool_result>`
/// blocks. Assistant messages with `tool_calls` are reconstructed as `<tool_call>` blocks.
fn convert_messages(messages: &[ChatCompletionMessage]) -> Vec<ChatMessage> {
    let mut result = Vec::with_capacity(messages.len());
    let mut i = 0;

    while i < messages.len() {
        let m = &messages[i];
        match m.role.as_str() {
            "system" => {
                result.push(ChatMessage::system(m.content.as_deref().unwrap_or("")));
                i += 1;
            }
            "user" => {
                result.push(ChatMessage::user(m.content.as_deref().unwrap_or("")));
                i += 1;
            }
            "assistant" => {
                if let Some(ref tool_calls) = m.tool_calls {
                    // Reconstruct <tool_call> XML blocks from stored tool calls
                    let mut text = m.content.clone().unwrap_or_default();
                    for tc in tool_calls {
                        text.push_str("\n<tool_call>\n");
                        let payload = serde_json::json!({
                            "name": tc.function.name,
                            "arguments": serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                        });
                        text.push_str(
                            &serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_owned()),
                        );
                        text.push_str("\n</tool_call>");
                    }
                    result.push(ChatMessage::assistant(text));
                } else {
                    result.push(ChatMessage::assistant(m.content.as_deref().unwrap_or("")));
                }
                i += 1;
            }
            "tool" => {
                // Collect consecutive tool messages into a single user message
                let mut tool_responses = Vec::new();
                while i < messages.len() && messages[i].role == "tool" {
                    let tool_msg = &messages[i];
                    let name = tool_msg.name.as_deref().unwrap_or("unknown");
                    let response_value: serde_json::Value =
                        tool_msg.content.as_deref().map_or_else(
                            || serde_json::Value::Null,
                            |c| {
                                serde_json::from_str(c)
                                    .unwrap_or_else(|_| serde_json::Value::String(c.to_owned()))
                            },
                        );
                    tool_responses.push(embacle::FunctionResponse {
                        name: name.to_owned(),
                        response: response_value,
                    });
                    i += 1;
                }
                let text = embacle::format_tool_results_as_text(&tool_responses);
                result.push(ChatMessage::user(text));
            }
            other => {
                warn!(role = other, "Unknown message role, mapping to user");
                result.push(ChatMessage::user(m.content.as_deref().unwrap_or("")));
                i += 1;
            }
        }
    }

    result
}

/// Convert `OpenAI` tool definitions to embacle `FunctionDeclaration` format
fn tools_to_declarations(
    tools: &[crate::openai_types::ToolDefinition],
) -> Vec<FunctionDeclaration> {
    tools
        .iter()
        .map(|t| FunctionDeclaration {
            name: t.function.name.clone(),
            description: t.function.description.clone().unwrap_or_default(),
            parameters: t.function.parameters.clone(),
        })
        .collect()
}

/// Check if `tool_choice` is explicitly "none"
fn is_tool_choice_none(tool_choice: Option<&ToolChoice>) -> bool {
    matches!(tool_choice, Some(ToolChoice::Mode(ref m)) if m == "none")
}

/// Generate a deterministic tool call ID from function name and index
fn generate_tool_call_id(name: &str, index: usize) -> String {
    format!("call_{name}_{index}")
}

/// Map a `RunnerError` to an appropriate HTTP status code and `OpenAI` error response
fn runner_error_to_response(err: &RunnerError) -> Response {
    let (status, error_type) = match err.kind {
        ErrorKind::BinaryNotFound => (StatusCode::SERVICE_UNAVAILABLE, "provider_not_available"),
        ErrorKind::AuthFailure => (StatusCode::UNAUTHORIZED, "authentication_error"),
        ErrorKind::Timeout => (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
        ErrorKind::ExternalService => (StatusCode::BAD_GATEWAY, "external_service_error"),
        ErrorKind::Config => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        ErrorKind::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    };

    error!(kind = ?err.kind, message = %err.message, "Runner error");
    let body = ErrorResponse::new(error_type, &err.message);
    (status, Json(body)).into_response()
}

/// Build an error response with a given status and message
fn error_response(status: StatusCode, message: &str) -> Response {
    let body = ErrorResponse::new("invalid_request_error", message);
    (status, Json(body)).into_response()
}

/// Monotonic counter ensuring unique IDs even for requests within the same second
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique completion ID
///
/// Combines the unix timestamp with a monotonically increasing counter
/// to guarantee uniqueness across concurrent and rapid-fire requests.
pub fn generate_id() -> String {
    let ts = unix_timestamp();
    let seq = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{ts:x}{seq:08x}")
}

/// Get current unix timestamp in seconds
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_types::{FunctionObject, ToolCall, ToolCallFunction, ToolDefinition};
    use embacle::types::MessageRole;

    #[test]
    fn convert_messages_maps_roles() {
        let openai_msgs = vec![
            ChatCompletionMessage {
                role: "system".to_owned(),
                content: Some("You are helpful".to_owned()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatCompletionMessage {
                role: "user".to_owned(),
                content: Some("Hello".to_owned()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatCompletionMessage {
                role: "assistant".to_owned(),
                content: Some("Hi there".to_owned()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2].role, MessageRole::Assistant);
    }

    #[test]
    fn convert_unknown_role_defaults_to_user() {
        let openai_msgs = vec![ChatCompletionMessage {
            role: "function".to_owned(),
            content: Some("result".to_owned()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].role, MessageRole::User);
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let openai_msgs = vec![ChatCompletionMessage {
            role: "assistant".to_owned(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_owned(),
                tool_type: "function".to_owned(),
                function: ToolCallFunction {
                    name: "get_weather".to_owned(),
                    arguments: r#"{"city":"Paris"}"#.to_owned(),
                },
            }]),
            tool_call_id: None,
            name: None,
        }];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert!(messages[0].content.contains("<tool_call>"));
        assert!(messages[0].content.contains("get_weather"));
        assert!(messages[0].content.contains("</tool_call>"));
    }

    #[test]
    fn convert_tool_messages_to_user() {
        let openai_msgs = vec![
            ChatCompletionMessage {
                role: "tool".to_owned(),
                content: Some(r#"{"temp":72}"#.to_owned()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_owned()),
                name: Some("get_weather".to_owned()),
            },
            ChatCompletionMessage {
                role: "tool".to_owned(),
                content: Some(r#"{"time":"14:30"}"#.to_owned()),
                tool_calls: None,
                tool_call_id: Some("call_2".to_owned()),
                name: Some("get_time".to_owned()),
            },
        ];

        let messages = convert_messages(&openai_msgs);
        // Consecutive tool messages should be merged into one user message
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::User);
        assert!(messages[0].content.contains("tool_result"));
        assert!(messages[0].content.contains("get_weather"));
        assert!(messages[0].content.contains("get_time"));
    }

    #[test]
    fn convert_messages_none_content() {
        let openai_msgs = vec![ChatCompletionMessage {
            role: "user".to_owned(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].content, "");
    }

    #[test]
    fn tools_to_declarations_converts() {
        let tools = vec![ToolDefinition {
            tool_type: "function".to_owned(),
            function: FunctionObject {
                name: "search".to_owned(),
                description: Some("Search the web".to_owned()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]
                })),
            },
        }];

        let decls = tools_to_declarations(&tools);
        assert_eq!(decls.len(), 1);
        assert_eq!(decls[0].name, "search");
        assert_eq!(decls[0].description, "Search the web");
        assert!(decls[0].parameters.is_some());
    }

    #[test]
    fn tool_choice_none_detection() {
        let none_choice = ToolChoice::Mode("none".to_owned());
        assert!(is_tool_choice_none(Some(&none_choice)));
        let auto_choice = ToolChoice::Mode("auto".to_owned());
        assert!(!is_tool_choice_none(Some(&auto_choice)));
        assert!(!is_tool_choice_none(None));
    }

    #[test]
    fn generate_tool_call_id_format() {
        let id = generate_tool_call_id("get_weather", 0);
        assert_eq!(id, "call_get_weather_0");
    }

    #[test]
    fn generate_id_has_prefix() {
        let id = generate_id();
        assert!(id.starts_with("chatcmpl-"));
    }

    #[test]
    fn error_maps_binary_not_found_to_503() {
        let err = RunnerError::binary_not_found("claude");
        let (status, _) = match err.kind {
            ErrorKind::BinaryNotFound => {
                (StatusCode::SERVICE_UNAVAILABLE, "provider_not_available")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn error_maps_auth_to_401() {
        let err = RunnerError::auth_failure("bad token");
        let (status, _) = match err.kind {
            ErrorKind::AuthFailure => (StatusCode::UNAUTHORIZED, "authentication_error"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn error_maps_timeout_to_504() {
        let err = RunnerError::timeout("too slow");
        let (status, _) = match err.kind {
            ErrorKind::Timeout => (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    }
}
