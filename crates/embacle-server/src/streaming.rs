// ABOUTME: Bridges embacle ChatStream to OpenAI-compatible Server-Sent Events format
// ABOUTME: Converts StreamChunk items to "data: {json}\n\n" SSE with [DONE] terminator
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use embacle::types::ChatStream;
use futures::StreamExt;

use crate::completions::{generate_id, unix_timestamp};
use crate::openai_types::{ChatCompletionChunk, ChunkChoice, Delta, ResponseMessage};

/// Convert a `ChatStream` into an SSE response in `OpenAI` streaming format
///
/// Emits:
/// 1. An initial chunk with role="assistant" and empty content
/// 2. Content delta chunks as they arrive from the provider
/// 3. A final chunk with `finish_reason`
/// 4. `data: [DONE]` terminator
pub fn sse_response(stream: ChatStream, model: &str) -> Response {
    let completion_id = generate_id();
    let created = unix_timestamp();
    let model = model.to_owned();

    let sse_stream = {
        let mut sent_role = false;

        stream.map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let (role, content, finish_reason) = if !sent_role {
                        sent_role = true;
                        if chunk.delta.is_empty() && !chunk.is_final {
                            // First chunk: role announcement only
                            (Some("assistant"), None, None)
                        } else {
                            // First chunk has content: send role + content
                            (Some("assistant"), Some(chunk.delta), chunk.finish_reason)
                        }
                    } else if chunk.is_final {
                        (
                            None,
                            if chunk.delta.is_empty() {
                                None
                            } else {
                                Some(chunk.delta)
                            },
                            Some(chunk.finish_reason.unwrap_or_else(|| "stop".to_owned())),
                        )
                    } else {
                        (None, Some(chunk.delta), None)
                    };

                    // LinesStream strips trailing \n from each line. Restore it
                    // so concatenated SSE deltas preserve original line breaks.
                    let content = content.map(|c| {
                        if !c.is_empty() && !c.ends_with('\n') {
                            let mut normalized = c;
                            normalized.push('\n');
                            normalized
                        } else {
                            c
                        }
                    });

                    let data = ChatCompletionChunk {
                        id: completion_id.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role,
                                content,
                                tool_calls: None,
                            },
                            finish_reason,
                        }],
                    };

                    let json = serde_json::to_string(&data).unwrap_or_default();
                    Ok::<_, Infallible>(Event::default().data(json))
                }
                Err(e) => {
                    let error_json = serde_json::json!({
                        "error": {
                            "message": e.message,
                            "type": "stream_error"
                        }
                    });
                    Ok(Event::default().data(error_json.to_string()))
                }
            }
        })
    };

    // Append the [DONE] sentinel after the stream completes
    let done_stream =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = sse_stream.chain(done_stream);

    Sse::new(combined)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// Emit a complete non-streaming response as an SSE event sequence
///
/// Used when the caller requested `stream: true` but the backend performed a
/// non-streaming `complete()` (e.g. for tool-calling downgrade). Produces:
/// 1. Role announcement chunk with content and/or `tool_calls`
/// 2. Final chunk with `finish_reason`
/// 3. `[DONE]` sentinel
pub fn sse_single_response(message: ResponseMessage, finish_reason: &str, model: &str) -> Response {
    let completion_id = generate_id();
    let created = unix_timestamp();

    let content_chunk = ChatCompletionChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: message.content,
                tool_calls: message.tool_calls,
            },
            finish_reason: None,
        }],
    };

    let final_chunk = ChatCompletionChunk {
        id: completion_id,
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(finish_reason.to_owned()),
        }],
    };

    let events = vec![
        serde_json::to_string(&content_chunk).unwrap_or_default(),
        serde_json::to_string(&final_chunk).unwrap_or_default(),
    ];

    let event_stream = futures::stream::iter(
        events
            .into_iter()
            .map(|json| Ok::<_, Infallible>(Event::default().data(json))),
    );
    let done_stream =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = event_stream.chain(done_stream);

    Sse::new(combined)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}
