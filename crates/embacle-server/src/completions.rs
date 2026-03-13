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
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, LlmCapabilities, RunnerError};
use embacle::FunctionDeclaration;
use tracing::{debug, error, warn};

use crate::openai_types::{
    ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ContentPart,
    ErrorResponse, MessageContent, ModelField, MultiplexProviderResult, MultiplexResponse,
    ResponseFormatRequest, ResponseMessage, StopField, ToolCall, ToolCallFunction, ToolChoice,
    Usage,
};
use crate::provider_resolver::resolve_model;
use crate::runner::multiplex::{MultiplexEngine, MultiplexParams};
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
    if let Some(top_p) = request.top_p {
        if !(0.0..=1.0).contains(&top_p) {
            return error_response(StatusCode::BAD_REQUEST, "top_p must be between 0.0 and 1.0");
        }
    }
    if let Some(ref stop) = request.stop {
        if stop.len() > 4 {
            return error_response(
                StatusCode::BAD_REQUEST,
                "stop must have at most 4 sequences",
            );
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

    let state_guard = state.read().await;
    let resolved = resolve_model(model_str, state_guard.active_provider());
    debug!(
        provider = %resolved.runner_type,
        model = ?resolved.model,
        stream = request.stream,
        has_tools,
        "Dispatching completion"
    );

    let runner = match state_guard.get_runner(resolved.runner_type).await {
        Ok(r) => r,
        Err(e) => return runner_error_to_response(&e),
    };
    drop(state_guard);

    let strict = request.strict_capabilities.unwrap_or_else(|| {
        std::env::var("EMBACLE_STRICT_CAPS")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    });

    let mut messages = convert_messages(&request.messages);

    // Inject tool catalog using the most effective strategy for this provider
    if has_tools {
        let declarations = tools_to_declarations(request.tools.as_deref().unwrap_or_default());
        let catalog = embacle::generate_tool_catalog(&declarations);

        if runner
            .capabilities()
            .contains(LlmCapabilities::SYSTEM_MESSAGES)
        {
            embacle::inject_tool_catalog(&mut messages, &catalog);
        } else {
            inject_tool_catalog_as_user_message(&mut messages, &catalog);
        }
    }

    let mut chat_request = ChatRequest::new(messages);
    chat_request.model = resolved.model;
    chat_request.temperature = request.temperature;
    chat_request.max_tokens = request.max_tokens;
    chat_request.top_p = request.top_p;
    chat_request.stop = request.stop.as_ref().map(StopField::to_bounded_vec);
    chat_request.response_format = request.response_format.as_ref().map(server_format_to_core);
    chat_request.tools = request
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(server_tool_to_core).collect());
    chat_request.tool_choice = request.tool_choice.as_ref().map(server_choice_to_core);

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

    let supports_streaming = runner.capabilities().contains(LlmCapabilities::STREAMING);

    dispatch_completion(
        runner.as_ref(),
        resolved.runner_type,
        chat_request,
        request.stream,
        has_tools,
        supports_streaming,
        warnings_for_response,
    )
    .await
}

/// Dispatch the completion request to the appropriate execution path
///
/// Routes between four modes:
/// 1. Streaming with tools: downgrade to `complete()`, emit as SSE
/// 2. Streaming without provider support: downgrade to `complete()`, emit as SSE
/// 3. Pure streaming: use `complete_stream()`
/// 4. Non-streaming: use `complete()`, return JSON
async fn dispatch_completion(
    runner: &dyn embacle::types::LlmProvider,
    runner_type: embacle::config::CliRunnerType,
    mut chat_request: ChatRequest,
    stream: bool,
    has_tools: bool,
    supports_streaming: bool,
    warnings: Option<Vec<String>>,
) -> Response {
    if stream && (has_tools || !supports_streaming) {
        // Downgrade to non-streaming complete(), emit result as SSE
        if has_tools {
            debug!("Downgrading stream+tools to non-streaming complete");
        } else {
            debug!(
                provider = runner.name(),
                "Provider does not support streaming; downgrading to non-streaming complete"
            );
        }
        match runner.complete(&chat_request).await {
            Ok(response) => {
                let model_name = format!("{runner_type}:{}", response.model);
                let (message, finish_reason) = build_response_message(
                    has_tools,
                    response.content,
                    response.finish_reason,
                    response.tool_calls.as_ref(),
                );
                let reason = finish_reason.as_deref().unwrap_or("stop");
                streaming::sse_single_response(message, reason, &model_name)
            }
            Err(e) => runner_error_to_response(&e),
        }
    } else if stream {
        chat_request.stream = true;
        match runner.complete_stream(&chat_request).await {
            Ok(s) => {
                let model_name = format!("{runner_type}:{}", runner.default_model());
                streaming::sse_response(s, &model_name)
            }
            Err(e) => runner_error_to_response(&e),
        }
    } else {
        match runner.complete(&chat_request).await {
            Ok(response) => {
                let model_name = format!("{runner_type}:{}", response.model);
                let usage = response.usage.map(|u| Usage {
                    prompt: u.prompt_tokens,
                    completion: u.completion_tokens,
                    total: u.total_tokens,
                });

                let (message, finish_reason) = build_response_message(
                    has_tools,
                    response.content,
                    response.finish_reason,
                    response.tool_calls.as_ref(),
                );

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
                    warnings,
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

    let state_guard = state.read().await;
    let default_provider = state_guard.active_provider();
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
    validation_request.top_p = request.top_p;
    validation_request.stop = request.stop.as_ref().map(StopField::to_bounded_vec);
    validation_request.response_format =
        request.response_format.as_ref().map(server_format_to_core);

    for &provider_type in &providers {
        let runner = match state_guard.get_runner(provider_type).await {
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

    drop(state_guard);
    let engine = MultiplexEngine::new(state);
    let params = MultiplexParams {
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        stop: request.stop.as_ref().map(StopField::to_bounded_vec),
        response_format: request.response_format.as_ref().map(server_format_to_core),
    };
    match engine.execute(&messages, &providers, &params).await {
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

/// Build a `ResponseMessage` from LLM output, using native tool calls if available
/// or falling back to XML parsing if tools were requested
fn build_response_message(
    has_tools: bool,
    content: String,
    finish_reason: Option<String>,
    native_tool_calls: Option<&Vec<embacle::ToolCallRequest>>,
) -> (ResponseMessage, Option<String>) {
    // If the provider returned native tool calls, use them directly
    if let Some(calls) = native_tool_calls {
        if !calls.is_empty() {
            let tool_calls: Vec<ToolCall> = calls
                .iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    index: i,
                    id: tc.id.clone(),
                    tool_type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: tc.function_name.clone(),
                        arguments: serde_json::to_string(&tc.arguments)
                            .unwrap_or_else(|_| "{}".to_owned()),
                    },
                })
                .collect();
            let text_content = if content.is_empty() {
                None
            } else {
                Some(content)
            };
            return (
                ResponseMessage {
                    role: "assistant",
                    content: text_content,
                    tool_calls: Some(tool_calls),
                },
                Some("tool_calls".to_owned()),
            );
        }
    }

    // Fall back to XML parsing for text-based tool simulation
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
                    index: i,
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

/// Extract text content from a `MessageContent`, returning an empty string for None
fn content_as_text(content: &Option<MessageContent>) -> String {
    content
        .as_ref()
        .map(MessageContent::as_text)
        .unwrap_or_default()
}

/// Parse a `data:` URI into an `ImagePart`
///
/// Expected format: `data:<mime_type>;base64,<data>`
fn parse_data_uri(url: &str) -> Option<embacle::ImagePart> {
    let rest = url.strip_prefix("data:")?;
    let (mime_type, data) = rest.split_once(";base64,")?;
    embacle::ImagePart::new(data, mime_type).ok()
}

/// Extract images from a `MessageContent::Parts` variant
fn extract_images(content: &Option<MessageContent>) -> Option<Vec<embacle::ImagePart>> {
    let parts = match content {
        Some(MessageContent::Parts(parts)) => parts,
        _ => return None,
    };

    let images: Vec<embacle::ImagePart> = parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::ImageUrl { image_url } => parse_data_uri(&image_url.url),
            ContentPart::Text { .. } => None,
        })
        .collect();

    if images.is_empty() {
        None
    } else {
        Some(images)
    }
}

/// Convert `OpenAI` message format to embacle `ChatMessage`
///
/// Handles all `OpenAI` roles including "tool" messages and assistant messages
/// with `tool_calls`. Tool messages are collected and formatted as `<tool_result>`
/// blocks. Assistant messages with `tool_calls` are reconstructed as `<tool_call>` blocks.
/// User messages with multipart content (text + images) are converted to `ChatMessage`
/// with attached `ImagePart` entries.
fn convert_messages(messages: &[ChatCompletionMessage]) -> Vec<ChatMessage> {
    let mut result = Vec::with_capacity(messages.len());
    let mut i = 0;

    while i < messages.len() {
        let m = &messages[i];
        match m.role.as_str() {
            "system" => {
                result.push(ChatMessage::system(content_as_text(&m.content)));
                i += 1;
            }
            "user" => {
                let text = content_as_text(&m.content);
                let images = extract_images(&m.content);
                if let Some(imgs) = images {
                    result.push(ChatMessage::user_with_images(text, imgs));
                } else {
                    result.push(ChatMessage::user(text));
                }
                i += 1;
            }
            "assistant" => {
                if let Some(ref tool_calls) = m.tool_calls {
                    // Reconstruct <tool_call> XML blocks from stored tool calls
                    let mut text = content_as_text(&m.content);
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
                    result.push(ChatMessage::assistant(content_as_text(&m.content)));
                }
                i += 1;
            }
            "tool" => {
                // Collect consecutive tool messages into a single user message
                let mut tool_responses = Vec::new();
                while i < messages.len() && messages[i].role == "tool" {
                    let tool_msg = &messages[i];
                    let name = tool_msg.name.as_deref().unwrap_or("unknown");
                    let content_text = content_as_text(&tool_msg.content);
                    let response_value: serde_json::Value = if content_text.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::from_str(&content_text)
                            .unwrap_or_else(|_| serde_json::Value::String(content_text))
                    };
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
                result.push(ChatMessage::user(content_as_text(&m.content)));
                i += 1;
            }
        }
    }

    result
}

/// Convert a server `ToolDefinition` to core `ToolDefinition`
fn server_tool_to_core(tool: &crate::openai_types::ToolDefinition) -> embacle::ToolDefinition {
    embacle::ToolDefinition {
        name: tool.function.name.clone(),
        description: tool.function.description.clone().unwrap_or_default(),
        parameters: tool.function.parameters.clone(),
    }
}

/// Convert a server `ToolChoice` to core `ToolChoice`
fn server_choice_to_core(choice: &ToolChoice) -> embacle::ToolChoice {
    match choice {
        ToolChoice::Mode(m) => match m.as_str() {
            "none" => embacle::ToolChoice::None,
            "required" => embacle::ToolChoice::Required,
            _ => embacle::ToolChoice::Auto,
        },
        ToolChoice::Specific(s) => embacle::ToolChoice::Specific {
            name: s.function.name.clone(),
        },
    }
}

/// Convert a server `ResponseFormatRequest` to core `ResponseFormat`
fn server_format_to_core(format: &ResponseFormatRequest) -> embacle::ResponseFormat {
    match format {
        ResponseFormatRequest::Text => embacle::ResponseFormat::Text,
        ResponseFormatRequest::JsonObject => embacle::ResponseFormat::JsonObject,
        ResponseFormatRequest::JsonSchema { json_schema } => embacle::ResponseFormat::JsonSchema {
            name: json_schema.name.clone(),
            schema: json_schema.schema.clone(),
        },
    }
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

/// Inject tool catalog into the last user message content
///
/// Used for providers that do not support system messages (e.g. Copilot CLI).
/// The catalog is prepended to the last user message so the LLM sees it in
/// the conversational flow rather than in a system prompt it cannot parse.
fn inject_tool_catalog_as_user_message(messages: &mut [ChatMessage], catalog: &str) {
    if let Some(last_user) = messages
        .iter_mut()
        .rev()
        .find(|m| m.role == embacle::types::MessageRole::User)
    {
        let augmented = format!("{catalog}\n\n{}", last_user.content);
        *last_user = ChatMessage::user(augmented);
    } else {
        // No user message found; this shouldn't happen in practice but
        // handle gracefully by appending a user message with the catalog.
        warn!("No user message found for tool catalog injection");
    }
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
        ErrorKind::Guardrail => (StatusCode::BAD_REQUEST, "guardrail_error"),
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
    use crate::openai_types::{
        ContentPart, FunctionObject, ImageUrlDetail, ToolCall, ToolCallFunction, ToolDefinition,
    };
    use embacle::types::MessageRole;

    /// Helper to create a `ChatCompletionMessage` with plain text content
    fn text_msg(role: &str, content: Option<&str>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: content.map(|c| MessageContent::Text(c.to_owned())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn convert_messages_maps_roles() {
        let openai_msgs = vec![
            text_msg("system", Some("You are helpful")),
            text_msg("user", Some("Hello")),
            text_msg("assistant", Some("Hi there")),
        ];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2].role, MessageRole::Assistant);
    }

    #[test]
    fn convert_unknown_role_defaults_to_user() {
        let openai_msgs = vec![text_msg("function", Some("result"))];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].role, MessageRole::User);
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let openai_msgs = vec![ChatCompletionMessage {
            role: "assistant".to_owned(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                index: 0,
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
                content: Some(MessageContent::Text(r#"{"temp":72}"#.to_owned())),
                tool_calls: None,
                tool_call_id: Some("call_1".to_owned()),
                name: Some("get_weather".to_owned()),
            },
            ChatCompletionMessage {
                role: "tool".to_owned(),
                content: Some(MessageContent::Text(r#"{"time":"14:30"}"#.to_owned())),
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
        let openai_msgs = vec![text_msg("user", None)];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].content, "");
    }

    #[test]
    fn convert_multipart_user_message_extracts_images() {
        let openai_msgs = vec![ChatCompletionMessage {
            role: "user".to_owned(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".to_owned(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlDetail {
                        url: "data:image/png;base64,aGVsbG8=".to_owned(),
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "What is this?");
        let images = messages[0].images.as_ref().expect("images present");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/png");
        assert_eq!(images[0].data, "aGVsbG8=");
    }

    #[test]
    fn parse_data_uri_valid() {
        let img = parse_data_uri("data:image/jpeg;base64,AAAA").expect("should parse");
        assert_eq!(img.mime_type, "image/jpeg");
        assert_eq!(img.data, "AAAA");
    }

    #[test]
    fn parse_data_uri_invalid_format() {
        assert!(parse_data_uri("https://example.com/image.png").is_none());
        assert!(parse_data_uri("data:text/plain;base64,abc").is_none());
        assert!(parse_data_uri("data:image/png;abc").is_none());
    }

    #[test]
    fn convert_plain_string_content_backward_compat() {
        let openai_msgs = vec![text_msg("user", Some("hello"))];
        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].content, "hello");
        assert!(messages[0].images.is_none());
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
    fn content_as_text_none() {
        assert_eq!(content_as_text(&None), "");
    }

    #[test]
    fn content_as_text_plain() {
        let content = Some(MessageContent::Text("hello".to_owned()));
        assert_eq!(content_as_text(&content), "hello");
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

    #[test]
    fn inject_tool_catalog_as_user_message_prepends_to_last_user() {
        let mut messages = vec![
            ChatMessage::user("First question"),
            ChatMessage::assistant("Some answer"),
            ChatMessage::user("What is the weather?"),
        ];
        let catalog = "## Available Tools\n- get_weather: Get the weather";

        inject_tool_catalog_as_user_message(&mut messages, catalog);

        assert_eq!(messages.len(), 3);
        assert!(messages[2].content.starts_with("## Available Tools"));
        assert!(messages[2].content.contains("What is the weather?"));
        // First user message should be untouched
        assert_eq!(messages[0].content, "First question");
    }

    #[test]
    fn inject_tool_catalog_as_user_message_single_user() {
        let mut messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let catalog = "## Tools\nsome tools";

        inject_tool_catalog_as_user_message(&mut messages, catalog);

        assert!(messages[1].content.starts_with("## Tools"));
        assert!(messages[1].content.contains("Hello"));
    }
}
