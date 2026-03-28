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

/// Convert a `ChatStream` into an SSE response, stripping markdown code fences
///
/// Used when `response_format` requests JSON. CLI runners often wrap JSON in
/// `` ```json ... ``` `` fences that arrive as separate stream chunks. This
/// variant filters those fence lines out so the client receives clean JSON.
pub fn sse_response_strip_fences(stream: ChatStream, model: &str) -> Response {
    let filtered = strip_fence_chunks(stream);
    sse_response(filtered, model)
}

/// Wrap a `ChatStream` to remove chunks that are markdown code fences
///
/// Fence-only chunks (`` ```json ``, `` ``` ``) are dropped entirely.
/// Final chunks with fence content have their delta cleared so the
/// finish signal still propagates.
fn strip_fence_chunks(stream: ChatStream) -> ChatStream {
    use embacle::types::StreamChunk;

    Box::pin(stream.filter_map(|result| async move {
        match result {
            Ok(chunk) => {
                if is_markdown_fence(&chunk.delta) {
                    if chunk.is_final {
                        // Preserve the final signal with empty content
                        Some(Ok(StreamChunk {
                            delta: String::new(),
                            is_final: true,
                            finish_reason: chunk.finish_reason,
                        }))
                    } else {
                        None
                    }
                } else {
                    Some(Ok(chunk))
                }
            }
            Err(e) => Some(Err(e)),
        }
    }))
}

/// Check if a stream chunk is a markdown code fence line (e.g. `` ```json `` or `` ``` ``)
fn is_markdown_fence(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with("```") && trimmed.bytes().skip(3).all(|b| b.is_ascii_alphanumeric())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_markdown_fence_detects_fences() {
        assert!(is_markdown_fence("```json\n"));
        assert!(is_markdown_fence("```\n"));
        assert!(is_markdown_fence("```json"));
        assert!(is_markdown_fence("```"));
        assert!(is_markdown_fence("  ```json  "));
    }

    #[test]
    fn is_markdown_fence_rejects_non_fences() {
        assert!(!is_markdown_fence("{\"key\": \"value\"}"));
        assert!(!is_markdown_fence("some text"));
        assert!(!is_markdown_fence(""));
        assert!(!is_markdown_fence("```json is cool```"));
        assert!(!is_markdown_fence("``` code here"));
    }

    #[tokio::test]
    async fn strip_fence_chunks_removes_fences() {
        use embacle::types::StreamChunk;

        let chunks = vec![
            Ok(StreamChunk {
                delta: "```json\n".to_owned(),
                is_final: false,
                finish_reason: None,
            }),
            Ok(StreamChunk {
                delta: "{\"key\":\"value\"}\n".to_owned(),
                is_final: false,
                finish_reason: None,
            }),
            Ok(StreamChunk {
                delta: "```\n".to_owned(),
                is_final: true,
                finish_reason: Some("stop".to_owned()),
            }),
        ];

        let stream: ChatStream = Box::pin(futures::stream::iter(chunks));
        let filtered = strip_fence_chunks(stream);

        let results: Vec<_> = filtered.collect().await;
        assert_eq!(results.len(), 2);

        // First result is the actual JSON content
        let first = results[0].as_ref().unwrap();
        assert_eq!(first.delta, "{\"key\":\"value\"}\n");
        assert!(!first.is_final);

        // Second result is the final signal with empty delta (fence stripped)
        let second = results[1].as_ref().unwrap();
        assert!(second.delta.is_empty());
        assert!(second.is_final);
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
    }
}
