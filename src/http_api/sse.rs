// ABOUTME: Shared Server-Sent Events line-buffering parser for the HTTP providers' streaming responses
// ABOUTME: Handles partial lines across TCP boundaries and multiple events per chunk, once, for every provider
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # SSE Stream Parser
//!
//! A line-buffering parser for Server-Sent Events shared by every streaming
//! HTTP provider. It solves two correctness problems once:
//!
//! 1. **Multiple events per TCP chunk**: when network buffers batch several
//!    SSE events into a single `bytes_stream()` chunk, every event is emitted,
//!    not just the first.
//! 2. **Partial JSON across TCP boundaries**: when a JSON payload is split
//!    across two chunks, the line buffer accumulates the partial data until a
//!    complete line arrives.
//!
//! Each provider supplies a `parse_data` closure that converts one raw
//! `data:` payload into a [`StreamChunk`]. The framing (line buffering,
//! `data:` prefix stripping, `[DONE]` detection) lives here.
//!
//! The returned stream always ends with exactly one final chunk: the
//! provider's own (a chunk whose `is_final` is set, or `[DONE]`), or a
//! synthesised `stop` when the byte stream closes without one. Empty
//! non-final deltas are dropped.

use std::collections::VecDeque;
use std::fmt::Display;
use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio_stream::Stream;

use crate::types::{ChatStream, RunnerError, StreamChunk};

/// A parsed SSE event from the stream
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    /// A `data:` payload with the JSON string (prefix stripped)
    Data(String),
    /// The `[DONE]` termination signal (OpenAI/Groq convention)
    Done,
}

/// The `[DONE]` sentinel `OpenAI`-style endpoints close a stream with.
const DONE_SENTINEL: &str = "[DONE]";

/// Line-buffering SSE parser that handles partial lines across TCP chunk boundaries
///
/// SSE streams are newline-delimited. TCP does not guarantee alignment between
/// network chunks and SSE event boundaries. This parser buffers incomplete lines
/// and emits complete events only when a full line (terminated by `\n`) is available.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Accumulated bytes not yet terminated by a newline
    buffer: String,
}

impl SseParser {
    /// Create a new empty line buffer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed raw bytes from a TCP chunk into the buffer, returning any complete SSE events
    ///
    /// Bytes are appended to the internal buffer. Complete lines (terminated by `\n`)
    /// are extracted, parsed as SSE events, and returned. Any trailing partial line
    /// remains in the buffer for the next `feed()` call.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.push_str(&String::from_utf8_lossy(bytes));

        let mut events = Vec::new();
        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].trim_end_matches('\r').to_owned();
            self.buffer.drain(..=newline_pos);
            events.extend(parse_line(&line));
        }
        events
    }

    /// Flush any remaining buffered content as a final event
    ///
    /// Called when the byte stream ends. If there is a partial line in the buffer
    /// (no trailing newline), attempt to parse it as an SSE event.
    pub fn flush(&mut self) -> Vec<SseEvent> {
        let remaining = mem::take(&mut self.buffer);
        parse_line(&remaining).into_iter().collect()
    }
}

/// One SSE line to at most one event. Non-`data` fields (`event:`, `id:`,
/// `retry:`, comments starting with `:`) and blank separators yield nothing.
/// The single optional space after `data:` is stripped, as the SSE grammar
/// says; `data:[DONE]` and `data: [DONE]` both terminate.
fn parse_line(line: &str) -> Option<SseEvent> {
    let trimmed = line.trim();
    let data = trimmed.strip_prefix("data:")?;
    let data = data.strip_prefix(' ').unwrap_or(data);
    if data.trim().is_empty() {
        return None;
    }
    if data.trim() == DONE_SENTINEL {
        return Some(SseEvent::Done);
    }
    Some(SseEvent::Data(data.to_owned()))
}

/// Create a properly-buffered SSE stream from a raw byte stream
///
/// Wraps a `reqwest` byte stream (or any stream of byte chunks) with SSE line
/// buffering. The `parse_data` closure converts provider-specific JSON strings
/// into `StreamChunk` values; it returns `None` to skip events that produce
/// no output (empty deltas, metadata-only frames).
///
/// # Arguments
///
/// * `byte_stream` - Raw bytes from `response.bytes_stream()`
/// * `parse_data` - Closure that parses a JSON string into an optional `StreamChunk`
/// * `provider` - Provider name for error messages (e.g., "groq", "gemini")
pub fn create_sse_stream<S, B, E, F>(
    byte_stream: S,
    parse_data: F,
    provider: &'static str,
) -> ChatStream
where
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]>,
    E: Display,
    F: Fn(&str) -> Option<Result<StreamChunk, RunnerError>> + Send + 'static,
{
    Box::pin(SseStream {
        bytes: Box::pin(byte_stream),
        parser: SseParser::new(),
        pending: VecDeque::new(),
        parse_data: Box::new(parse_data),
        provider,
        ended: false,
        final_emitted: false,
    })
}

/// A provider's `data:` payload parser, boxed so the stream stays `Unpin`.
type ParseData = Box<dyn Fn(&str) -> Option<Result<StreamChunk, RunnerError>> + Send>;

/// The stream [`create_sse_stream`] returns: pull-based, so the byte stream
/// is only read when a consumer asks for the next chunk.
struct SseStream<S> {
    bytes: Pin<Box<S>>,
    parser: SseParser,
    pending: VecDeque<Result<StreamChunk, RunnerError>>,
    parse_data: ParseData,
    provider: &'static str,
    ended: bool,
    final_emitted: bool,
}

impl<S> SseStream<S> {
    /// Queue what a batch of events produces, dropping empty non-final deltas
    /// and a `[DONE]` that follows the provider's own final chunk.
    fn queue(&mut self, events: Vec<SseEvent>) {
        for event in events {
            let item = match event {
                SseEvent::Data(json) => match (self.parse_data)(&json) {
                    Some(item) => item,
                    None => continue,
                },
                SseEvent::Done if self.final_emitted => continue,
                SseEvent::Done => Ok(final_chunk()),
            };
            if let Ok(chunk) = &item {
                if chunk.delta.is_empty() && !chunk.is_final {
                    continue;
                }
                self.final_emitted |= chunk.is_final;
            }
            self.pending.push_back(item);
        }
    }
}

/// The chunk that closes a stream when the provider sent `[DONE]` or nothing
/// at all.
fn final_chunk() -> StreamChunk {
    StreamChunk {
        delta: String::new(),
        is_final: true,
        finish_reason: Some("stop".to_owned()),
    }
}

impl<S, B, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: Display,
{
    type Item = Result<StreamChunk, RunnerError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(item) = this.pending.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.ended {
                return Poll::Ready(None);
            }
            match this.bytes.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    let events = this.parser.feed(bytes.as_ref());
                    this.queue(events);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(RunnerError::external_service(
                        this.provider,
                        format!("Stream read error: {e}"),
                    ))));
                }
                Poll::Ready(None) => {
                    this.ended = true;
                    let events = this.parser.flush();
                    this.queue(events);
                    if !this.final_emitted {
                        this.final_emitted = true;
                        this.pending.push_back(Ok(final_chunk()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_event() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: {\"choices\":[]}\n\n");
        assert_eq!(events, vec![SseEvent::Data("{\"choices\":[]}".to_owned())]);
    }

    #[test]
    fn multiple_events_per_chunk() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: first\n\ndata: second\n\n");
        assert_eq!(
            events,
            vec![
                SseEvent::Data("first".to_owned()),
                SseEvent::Data("second".to_owned())
            ]
        );
    }

    #[test]
    fn done_signal() {
        let mut parser = SseParser::new();
        assert_eq!(parser.feed(b"data: [DONE]\n\n"), vec![SseEvent::Done]);
        assert_eq!(parser.feed(b"data:[DONE]\n"), vec![SseEvent::Done]);
    }

    #[test]
    fn partial_line_stays_in_buffer_until_flushed() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: partial").is_empty());
        assert_eq!(parser.flush(), vec![SseEvent::Data("partial".to_owned())]);
        assert!(parser.flush().is_empty());
    }

    #[test]
    fn crlf_boundary() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: content\r\n\r\n");
        assert_eq!(events, vec![SseEvent::Data("content".to_owned())]);
    }

    #[test]
    fn comments_and_non_data_fields_are_ignored() {
        let mut parser = SseParser::new();
        let events =
            parser.feed(b": keepalive\n\nevent: message\nid: 123\nretry: 5000\ndata: real\n\n");
        assert_eq!(events, vec![SseEvent::Data("real".to_owned())]);
    }

    #[test]
    fn no_space_after_data_colon() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data:{\"ok\":true}\n\n");
        assert_eq!(events, vec![SseEvent::Data("{\"ok\":true}".to_owned())]);
    }

    #[test]
    fn a_data_line_with_no_payload_yields_nothing() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: \n\ndata:\n\n").is_empty());
    }
}
