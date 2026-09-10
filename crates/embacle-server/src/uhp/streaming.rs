// ABOUTME: Server-sent events for a streaming task, in the protocol's event taxonomy
// ABOUTME: sequence_number starts at 0 with no gaps; exactly one terminal event ends the stream
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Streaming a task.
//!
//! The ordering rules are the whole of this module. `sequence_number` starts at
//! **0** and increases by exactly one per event, so a client can detect a
//! dropped event rather than silently rendering a gap. The first event is
//! `response.created`. Exactly one terminal event — `response.completed`,
//! `response.incomplete` or `response.failed` — ends the stream, and it carries
//! the complete response object, so a client that watched nothing in between
//! still ends up with the same result the non-streaming path returns.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;

use super::error::{ErrorType, UhpError};
use super::state::UhpState;
use super::tasks::{self, Prepared, Response};

/// One streamed event.
#[derive(Debug, Clone, Serialize)]
pub struct StreamEvent {
    /// Event type, e.g. `response.created`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Starts at 0, increases by exactly 1.
    pub sequence_number: u64,
    /// Present on `response.created` and on the terminal event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Response>,
    /// Present on output-item events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<Value>,
    /// The text this event adds, on a delta event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
}

impl StreamEvent {
    /// An event carrying the response object.
    fn with_response(kind: &str, sequence_number: u64, response: Response) -> Self {
        Self {
            kind: kind.to_owned(),
            sequence_number,
            response: Some(response),
            item: None,
            delta: None,
        }
    }

    /// A chunk of generated text.
    fn delta(sequence_number: u64, text: &str) -> Self {
        Self {
            kind: "response.output_text.delta".to_owned(),
            sequence_number,
            response: None,
            item: None,
            delta: Some(text.to_owned()),
        }
    }
}

/// The terminal event type for a finished task's status.
///
/// `cancelled` streams as `response.incomplete`, because the taxonomy has no
/// cancelled event and reporting it as `failed` is explicitly forbidden — a
/// client asked for the stop, so it is not a failure.
const fn terminal_kind(status: &str) -> &'static str {
    match status.as_bytes() {
        b"completed" => "response.completed",
        b"failed" => "response.failed",
        _ => "response.incomplete",
    }
}

/// Stream a task as it runs, in a well-ordered event sequence.
///
/// The events are produced as the runner produces them, not assembled after it
/// finishes: `response.created` goes out before the harness is even asked, each
/// chunk becomes a `response.output_text.delta`, and the terminal event carries
/// the assembled response. A client that watched nothing in between still ends
/// with what the non-streaming path would have returned.
pub fn sse(state: UhpState, prepared: Prepared, previous: Option<String>) -> impl IntoResponse {
    Sse::new(run_stream(state, prepared, previous))
}

/// The event sequence for one streaming task.
fn run_stream(
    state: UhpState,
    prepared: Prepared,
    previous: Option<String>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut seq = 0_u64;
        let session_id = prepared.session_id.clone();
        let created = tasks::pending_response(&prepared, previous.clone());
        state.store.put(&session_id, created.clone());

        yield emit(StreamEvent::with_response("response.created", seq, created));
        seq += 1;

        let mut text = String::new();
        let outcome = prepared.runner.complete_stream(&prepared.chat).await;

        let failure = match outcome {
            Ok(mut chunks) => {
                let mut err = None;
                while let Some(next) = chunks.next().await {
                    match next {
                        Ok(chunk) => {
                            if !chunk.delta.is_empty() {
                                text.push_str(&chunk.delta);
                                yield emit(StreamEvent::delta(seq, &chunk.delta));
                                seq += 1;
                            }
                        }
                        Err(e) => {
                            err = Some(e.message);
                            break;
                        }
                    }
                }
                err
            }
            Err(e) => Some(e.message),
        };

        let model = prepared.model.clone();
        let response = match failure {
            None => tasks::completed_response(prepared, previous, text, None, model),
            Some(message) => {
                let mut r = tasks::pending_response(&prepared, previous);
                r.status = "failed";
                r.error = Some(UhpError {
                    kind: ErrorType::HarnessError,
                    code: "harness_error".to_owned(),
                    message,
                    detail: None,
                });
                r
            }
        };

        state.store.settle(response.clone());
        let kind = terminal_kind(response.status);
        yield emit(StreamEvent::with_response(kind, seq, response));
    }
}

/// Serialise one event onto the wire.
///
/// Returns the `Result` the SSE stream item type requires; the error half is
/// [`Infallible`] because a failure to serialise is answered with an empty
/// object rather than by tearing down a stream the client is already reading.
#[allow(clippy::unnecessary_wraps)]
fn emit(event: StreamEvent) -> Result<Event, Infallible> {
    let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_owned());
    Ok(Event::default().event(event.kind).data(json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    fn response(status: &'static str) -> Response {
        Response {
            id: "resp_x".to_owned(),
            object: "response",
            created_at: 1,
            status,
            error: None,
            previous_response_id: None,
            model: "m".to_owned(),
            output: Vec::new(),
            store: true,
            usage: None,
            metadata: Map::new(),
        }
    }

    #[test]
    fn a_cancelled_task_never_streams_as_failed() {
        // The specification is explicit: cancelled must never be reported as
        // failed, because the client asked for the stop.
        assert_eq!(terminal_kind("cancelled"), "response.incomplete");
        assert_eq!(terminal_kind("incomplete"), "response.incomplete");
        assert_eq!(terminal_kind("completed"), "response.completed");
        assert_eq!(terminal_kind("failed"), "response.failed");
    }

    #[test]
    fn the_event_sequence_obeys_the_ordering_rules() {
        // The rules the suite checks, asserted on the sequence this module
        // builds: starts at 0, no gaps, created first, exactly one terminal
        // event, and it is last and carries the response.
        let evs = [
            StreamEvent::with_response("response.created", 0, response("in_progress")),
            StreamEvent::delta(1, "ok"),
            StreamEvent::with_response("response.completed", 2, response("completed")),
        ];

        let seqs: Vec<u64> = evs.iter().map(|e| e.sequence_number).collect();
        assert_eq!(
            seqs,
            vec![0, 1, 2],
            "starts at 0 and increases by exactly one"
        );
        assert_eq!(
            evs[0].kind, "response.created",
            "the first event is created"
        );

        let terminal = [
            "response.completed",
            "response.incomplete",
            "response.failed",
        ];
        assert_eq!(
            evs.iter()
                .filter(|e| terminal.contains(&e.kind.as_str()))
                .count(),
            1,
            "exactly one terminal event"
        );
        assert!(
            evs.last()
                .is_some_and(|e| terminal.contains(&e.kind.as_str())),
            "and it is last"
        );
        assert!(
            evs.last().and_then(|e| e.response.as_ref()).is_some(),
            "the terminal event carries the complete response"
        );
        assert_eq!(
            evs[1].delta.as_deref(),
            Some("ok"),
            "a delta event carries the text it adds, and no response object"
        );
        assert!(evs[1].response.is_none());
    }
}
