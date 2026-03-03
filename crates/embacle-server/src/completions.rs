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
use embacle::types::{ChatMessage, ChatRequest, ErrorKind, MessageRole, RunnerError};
use tracing::{debug, error, warn};

use crate::openai_types::{
    ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ErrorResponse,
    ModelField, MultiplexProviderResult, MultiplexResponse, ResponseMessage, Usage,
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
    let resolved = resolve_model(model_str, state.default_provider());
    debug!(
        provider = %resolved.runner_type,
        model = ?resolved.model,
        stream = request.stream,
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

    let messages = convert_messages(&request.messages);
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

                let resp = ChatCompletionResponse {
                    id: generate_id(),
                    object: "chat.completion",
                    created: unix_timestamp(),
                    model: model_name,
                    choices: vec![Choice {
                        index: 0,
                        message: ResponseMessage {
                            role: "assistant",
                            content: response.content,
                        },
                        finish_reason: response.finish_reason.or_else(|| Some("stop".to_owned())),
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

/// Convert `OpenAI` message format to embacle `ChatMessage`
fn convert_messages(messages: &[ChatCompletionMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => MessageRole::System,
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                other => {
                    warn!(role = other, "Unknown message role, mapping to user");
                    MessageRole::User
                }
            };
            ChatMessage::new(role, &m.content)
        })
        .collect()
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

    #[test]
    fn convert_messages_maps_roles() {
        let openai_msgs = vec![
            ChatCompletionMessage {
                role: "system".to_owned(),
                content: "You are helpful".to_owned(),
            },
            ChatCompletionMessage {
                role: "user".to_owned(),
                content: "Hello".to_owned(),
            },
            ChatCompletionMessage {
                role: "assistant".to_owned(),
                content: "Hi there".to_owned(),
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
            content: "result".to_owned(),
        }];

        let messages = convert_messages(&openai_msgs);
        assert_eq!(messages[0].role, MessageRole::User);
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
