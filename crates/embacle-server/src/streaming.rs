// ABOUTME: Bridges embacle ChatStream to OpenAI-compatible Server-Sent Events format
// ABOUTME: Converts StreamChunk items to "data: {json}\n\n" SSE with [DONE] terminator
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::convert::Infallible;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use embacle::types::{ChatStream, RunnerError, StreamChunk};
use futures::StreamExt;
use futures::{future, stream};

use crate::completions::{generate_id, generate_tool_call_id, unix_timestamp};
use crate::openai_types::{
    ChatCompletionChunk, ChunkChoice, Delta, ResponseMessage, ToolCall, ToolCallFunction,
};

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
    let done_stream = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = sse_stream.chain(done_stream);

    Sse::new(combined)
        .keep_alive(KeepAlive::default())
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

    let event_stream = stream::iter(
        events
            .into_iter()
            .map(|json| Ok::<_, Infallible>(Event::default().data(json))),
    );
    let done_stream = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    let combined = event_stream.chain(done_stream);

    Sse::new(combined)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Opening marker for a text-simulated tool call block.
const TOOL_OPEN: &str = "<tool_call>";
/// Closing marker for a text-simulated tool call block.
const TOOL_CLOSE: &str = "</tool_call>";

/// A single item produced by the incremental tool-call scanner.
enum Emit {
    /// Prose content to forward as a content delta
    Content(String),
    /// A completed tool call to forward as a `tool_calls` delta
    Tool(ToolCall),
}

/// Whether the scanner is reading prose or the body of a `<tool_call>` block.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    /// Reading prose outside any tool-call block
    Text,
    /// Reading the body of an open `<tool_call>` block
    Tool,
}

/// Incremental state machine that extracts `<tool_call>` blocks from a token
/// stream while forwarding surrounding prose.
///
/// Tool-call markers and bodies may be split arbitrarily across stream chunks;
/// the scanner buffers just enough to detect markers and never emits a partial
/// `<tool_call>` opening tag as prose.
struct ToolStreamState {
    /// Unprocessed tail of the stream
    buffer: String,
    /// Whether the scanner is reading prose or a tool-call body
    mode: ScanMode,
    /// Number of tool calls emitted so far (drives id + index)
    tool_index: usize,
    /// Whether any tool call has been emitted (drives the finish reason)
    emitted_tool: bool,
    /// Whether the role announcement has been sent on the first delta
    sent_role: bool,
    /// Whether the terminal finish chunk has been emitted
    finished: bool,
}

impl ToolStreamState {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: ScanMode::Text,
            tool_index: 0,
            emitted_tool: false,
            sent_role: false,
            finished: false,
        }
    }

    /// Feed an incoming delta and return the items ready to emit.
    fn process(&mut self, incoming: &str) -> Vec<Emit> {
        self.buffer.push_str(incoming);
        let mut out = Vec::new();

        loop {
            if self.mode == ScanMode::Tool {
                if let Some(idx) = self.buffer.find(TOOL_CLOSE) {
                    let inner = self.buffer[..idx].to_owned();
                    self.buffer.drain(..idx + TOOL_CLOSE.len());
                    self.mode = ScanMode::Text;
                    if let Some(tool) = self.parse_tool(&inner) {
                        out.push(Emit::Tool(tool));
                    }
                    continue;
                }
                break;
            }

            if let Some(idx) = self.buffer.find(TOOL_OPEN) {
                if idx > 0 {
                    out.push(Emit::Content(self.buffer[..idx].to_owned()));
                }
                self.buffer.drain(..idx + TOOL_OPEN.len());
                self.mode = ScanMode::Tool;
                continue;
            }

            // No complete opening marker — emit prose, holding back any suffix
            // that could be the start of a `<tool_call>` tag split across chunks.
            let safe = safe_prefix_len(&self.buffer);
            if safe > 0 {
                out.push(Emit::Content(self.buffer[..safe].to_owned()));
                self.buffer.drain(..safe);
            }
            break;
        }

        out
    }

    /// Flush any buffered remainder when the stream ends.
    ///
    /// Unterminated tool blocks and held-back partial markers are emitted as
    /// prose so no output is silently dropped.
    fn finalize(&mut self) -> Vec<Emit> {
        let mut out = Vec::new();
        if !self.buffer.is_empty() {
            let mut remainder = String::new();
            if self.mode == ScanMode::Tool {
                remainder.push_str(TOOL_OPEN);
            }
            remainder.push_str(&self.buffer);
            self.buffer.clear();
            out.push(Emit::Content(remainder));
        }
        self.mode = ScanMode::Text;
        out
    }

    /// Parse the inner body of a `<tool_call>` block into an `OpenAI` tool call.
    fn parse_tool(&mut self, inner: &str) -> Option<ToolCall> {
        let block = format!("{TOOL_OPEN}{inner}{TOOL_CLOSE}");
        let call = embacle::parse_tool_call_blocks(&block).into_iter().next()?;
        let index = self.tool_index;
        self.tool_index += 1;
        self.emitted_tool = true;
        Some(ToolCall {
            index,
            id: generate_tool_call_id(&call.name, index),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name: call.name,
                arguments: serde_json::to_string(&call.args).unwrap_or_else(|_| "{}".to_owned()),
            },
        })
    }

    /// Resolve the `OpenAI` finish reason for the terminal chunk.
    fn finish_reason(&self, provider: Option<String>) -> String {
        if self.emitted_tool {
            "tool_calls".to_owned()
        } else {
            provider.unwrap_or_else(|| "stop".to_owned())
        }
    }

    /// Take the role marker exactly once, for the first emitted delta.
    fn take_role(&mut self) -> Option<&'static str> {
        if self.sent_role {
            None
        } else {
            self.sent_role = true;
            Some("assistant")
        }
    }
}

/// Length of `buffer` that is safe to emit as prose without splitting a
/// `<tool_call>` opening tag that may continue in a later chunk.
///
/// Holds back the longest trailing substring of `buffer` that is also a proper
/// prefix of [`TOOL_OPEN`]. Because `TOOL_OPEN` is ASCII, the returned boundary
/// is always a valid char boundary.
fn safe_prefix_len(buffer: &str) -> usize {
    let max = (TOOL_OPEN.len() - 1).min(buffer.len());
    for k in (1..=max).rev() {
        if buffer.as_bytes().ends_with(&TOOL_OPEN.as_bytes()[..k]) {
            return buffer.len() - k;
        }
    }
    buffer.len()
}

/// Convert a `ChatStream` into an SSE response that streams prose as content
/// deltas and completed `<tool_call>` blocks as `tool_calls` deltas.
///
/// Unlike [`sse_single_response`], which buffers the whole completion, this
/// forwards tokens incrementally. Used when a tools-bearing request targets a
/// provider that supports streaming.
pub fn sse_response_with_tool_calls(stream: ChatStream, model: &str) -> Response {
    let id = generate_id();
    let created = unix_timestamp();
    let model = model.to_owned();

    let mapped = stream
        .scan(ToolStreamState::new(), move |state, chunk_result| {
            let events = handle_tool_chunk(state, chunk_result, &id, created, &model);
            future::ready(Some(stream::iter(events)))
        })
        .flatten();

    let done_stream = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let combined = mapped.chain(done_stream);

    Sse::new(combined)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Process one input chunk into zero or more SSE events.
fn handle_tool_chunk(
    state: &mut ToolStreamState,
    chunk_result: Result<StreamChunk, RunnerError>,
    id: &str,
    created: u64,
    model: &str,
) -> Vec<Result<Event, Infallible>> {
    match chunk_result {
        Ok(chunk) => {
            let mut emits = state.process(&chunk.delta);
            if chunk.is_final {
                emits.extend(state.finalize());
            }

            let mut events: Vec<Result<Event, Infallible>> = emits
                .into_iter()
                .map(|emit| Ok(emit_to_event(state, emit, id, created, model)))
                .collect();

            if chunk.is_final && !state.finished {
                state.finished = true;
                let reason = state.finish_reason(chunk.finish_reason);
                events.push(Ok(final_tool_event(id, created, model, &reason)));
            }

            events
        }
        Err(e) => {
            let error_json = serde_json::json!({
                "error": { "message": e.message, "type": "stream_error" }
            });
            vec![Ok(Event::default().data(error_json.to_string()))]
        }
    }
}

/// Build an SSE event for a single emitted item.
fn emit_to_event(
    state: &mut ToolStreamState,
    emit: Emit,
    id: &str,
    created: u64,
    model: &str,
) -> Event {
    let role = state.take_role();
    let delta = match emit {
        Emit::Content(text) => {
            // LinesStream strips trailing newlines; restore one so concatenated
            // deltas preserve line breaks, matching `sse_response`.
            let content = if !text.is_empty() && !text.ends_with('\n') {
                format!("{text}\n")
            } else {
                text
            };
            Delta {
                role,
                content: Some(content),
                tool_calls: None,
            }
        }
        Emit::Tool(tool_call) => Delta {
            role,
            content: None,
            tool_calls: Some(vec![tool_call]),
        },
    };

    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: None,
        }],
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
}

/// Build the terminal SSE event carrying the finish reason.
fn final_tool_event(id: &str, created: u64, model: &str, reason: &str) -> Event {
    let chunk = ChatCompletionChunk {
        id: id.to_owned(),
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
            finish_reason: Some(reason.to_owned()),
        }],
    };
    Event::default().data(serde_json::to_string(&chunk).unwrap_or_default())
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

    /// Collect prose content from a sequence of emits.
    fn collect_content(emits: &[Emit]) -> String {
        emits
            .iter()
            .filter_map(|e| match e {
                Emit::Content(c) => Some(c.as_str()),
                Emit::Tool(_) => None,
            })
            .collect()
    }

    /// Collect tool calls from a sequence of emits.
    fn collect_tools(emits: &[Emit]) -> Vec<&ToolCall> {
        emits
            .iter()
            .filter_map(|e| match e {
                Emit::Tool(t) => Some(t),
                Emit::Content(_) => None,
            })
            .collect()
    }

    #[test]
    fn safe_prefix_holds_back_partial_marker() {
        // "abc<tool" must hold back "<tool" (a prefix of <tool_call>)
        assert_eq!(safe_prefix_len("abc<tool"), 3);
        // Full prose with no partial marker is fully emittable
        assert_eq!(safe_prefix_len("hello world"), 11);
        // A lone "<" is held back (could begin the marker)
        assert_eq!(safe_prefix_len("done<"), 4);
        // Text containing "<" not at a marker-prefix position is safe
        assert_eq!(safe_prefix_len("a < b"), 5);
    }

    #[test]
    fn process_passes_through_prose() {
        let mut state = ToolStreamState::new();
        let emits = state.process("Hello, world!");
        assert_eq!(collect_content(&emits), "Hello, world!");
        assert!(collect_tools(&emits).is_empty());
    }

    #[test]
    fn process_extracts_text_then_tool_call() {
        let mut state = ToolStreamState::new();
        let input =
            "Let me check.<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";
        let emits = state.process(input);
        assert_eq!(collect_content(&emits), "Let me check.");
        let tools = collect_tools(&emits);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(tools[0].function.arguments.contains("Paris"));
        assert_eq!(tools[0].index, 0);
        assert_eq!(tools[0].id, "call_get_weather_0");
        assert!(state.emitted_tool);
    }

    #[test]
    fn process_handles_marker_split_across_chunks() {
        let mut state = ToolStreamState::new();
        // Opening marker split mid-tag
        let mut all = Vec::new();
        all.extend(state.process("answer<tool"));
        all.extend(state.process("_call>{\"name\":\"ping\","));
        all.extend(state.process("\"arguments\":{}}</tool_call> done"));
        // Prose before and after the tool call is preserved; the partial marker
        // is never emitted as prose.
        assert_eq!(collect_content(&all), "answer done");
        let tools = collect_tools(&all);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");
    }

    #[test]
    fn process_handles_multiple_tool_calls() {
        let mut state = ToolStreamState::new();
        let input = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let emits = state.process(input);
        let tools = collect_tools(&emits);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].index, 0);
        assert_eq!(tools[1].index, 1);
        assert_eq!(tools[1].id, "call_b_1");
    }

    #[test]
    fn finalize_flushes_held_back_partial_as_prose() {
        let mut state = ToolStreamState::new();
        // A trailing "<" that looked like a possible marker start but never completed
        let mut all = state.process("almost done<");
        all.extend(state.finalize());
        assert_eq!(collect_content(&all), "almost done<");
    }

    #[test]
    fn finalize_flushes_unterminated_tool_block_as_prose() {
        let mut state = ToolStreamState::new();
        let mut all = state.process("<tool_call>{\"name\":\"x\"");
        all.extend(state.finalize());
        // Unterminated block is surfaced rather than dropped
        assert!(collect_content(&all).contains("<tool_call>"));
        assert!(collect_content(&all).contains("\"name\":\"x\""));
    }

    #[test]
    fn finish_reason_reflects_tool_emission() {
        let mut state = ToolStreamState::new();
        assert_eq!(state.finish_reason(None), "stop");
        assert_eq!(state.finish_reason(Some("length".to_owned())), "length");
        state.emitted_tool = true;
        assert_eq!(state.finish_reason(None), "tool_calls");
    }

    #[test]
    fn take_role_emits_assistant_once() {
        let mut state = ToolStreamState::new();
        assert_eq!(state.take_role(), Some("assistant"));
        assert_eq!(state.take_role(), None);
    }

    #[tokio::test]
    async fn sse_with_tool_calls_emits_done_and_finish() {
        use axum::body::to_bytes;

        let chunks = vec![
            Ok(StreamChunk {
                delta: "Working<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>"
                    .to_owned(),
                is_final: false,
                finish_reason: None,
            }),
            Ok(StreamChunk {
                delta: String::new(),
                is_final: true,
                finish_reason: Some("stop".to_owned()),
            }),
        ];
        let input: ChatStream = Box::pin(stream::iter(chunks));
        let response = sse_response_with_tool_calls(input, "copilot:gpt-5.4");
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body"); // Safe: test assertion
        let text = String::from_utf8(body.to_vec()).expect("utf8"); // Safe: test assertion

        assert!(text.contains("\"tool_calls\""));
        assert!(text.contains("ping"));
        // Tool emission overrides the provider's "stop" with "tool_calls"
        assert!(text.contains("\"finish_reason\":\"tool_calls\""));
        assert!(text.contains("[DONE]"));
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

        let input: ChatStream = Box::pin(stream::iter(chunks));
        let filtered = strip_fence_chunks(input);

        let results: Vec<_> = filtered.collect().await;
        assert_eq!(results.len(), 2);

        // First result is the actual JSON content
        let first = results[0].as_ref().unwrap(); // Safe: test assertion
        assert_eq!(first.delta, "{\"key\":\"value\"}\n");
        assert!(!first.is_final);

        // Second result is the final signal with empty delta (fence stripped)
        let second = results[1].as_ref().unwrap(); // Safe: test assertion
        assert!(second.delta.is_empty());
        assert!(second.is_final);
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
    }
}
