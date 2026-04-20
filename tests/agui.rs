// ABOUTME: Integration tests for the AG-UI protocol event schema (feature: agui)
// ABOUTME: Exercises wire-format tagging, round-trip serde, and filter semantics
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![cfg(feature = "agui")]

use embacle::agui::{AgUiEmitter, AgUiEvent, AgUiEventFilter, AgUiEventKind, NoopEmitter};

/// Event constructors tag the JSON with the AG-UI spec's `type` field
/// (screaming-snake-case) so the wire format is interoperable with
/// off-the-shelf AG-UI clients.
#[test]
fn event_serializes_with_spec_tag() {
    let event = AgUiEvent::run_started("run_1", Some("thread_1"));
    let Ok(json) = serde_json::to_value(&event) else {
        panic!("AG-UI event must serialize to JSON without error");
    };
    assert_eq!(json["type"], "RUN_STARTED");
    assert_eq!(json["run_id"], "run_1");
    assert_eq!(json["thread_id"], "thread_1");
    assert!(json["timestamp"].is_u64());
}

/// Events round-trip through JSON so the wire format is stable for
/// downstream AG-UI consumers that re-serialize events.
#[test]
fn step_events_round_trip_via_json() {
    let event = AgUiEvent::step_started("run_1", "tool_loop");
    let Ok(json) = serde_json::to_string(&event) else {
        panic!("AG-UI event must serialize to JSON without error");
    };
    let Ok(decoded) = serde_json::from_str::<AgUiEvent>(&json) else {
        panic!("AG-UI event must deserialize from its own JSON");
    };
    assert!(matches!(decoded.kind(), AgUiEventKind::StepStarted));
    assert_eq!(decoded.run_id(), "run_1");
}

/// `without` narrows the allowlist by removing a single kind.
#[test]
fn filter_without_removes_kind() {
    let filter = AgUiEventFilter::allow_all().without(AgUiEventKind::TextMessageContent);
    assert!(filter.allows(AgUiEventKind::RunStarted));
    assert!(!filter.allows(AgUiEventKind::TextMessageContent));
}

/// `only` is exclusive — kinds not listed are denied.
#[test]
fn filter_only_is_exclusive() {
    let filter = AgUiEventFilter::only([AgUiEventKind::RunStarted, AgUiEventKind::RunFinished]);
    assert!(filter.allows(AgUiEventKind::RunStarted));
    assert!(!filter.allows(AgUiEventKind::StepStarted));
}

/// `NoopEmitter` compiles against `AgUiEmitter` and drops events silently.
#[tokio::test]
async fn noop_emitter_swallows_events() {
    let sink = NoopEmitter::default();
    sink.emit(&AgUiEvent::run_started("run_1", None)).await;
}

/// Forward-compat: a wire-format event whose `type` is unknown to this
/// embacle release MUST deserialize successfully into `AgUiEvent::Unknown`
/// so downstream consumers can skip it without the stream tearing down.
/// Without this, a newer producer (e.g. ag-ui-js or a future embacle)
/// adding a new event kind would break every existing canot / platform
/// build at the first frame carrying the new type.
#[test]
fn unknown_type_deserializes_to_unknown_variant() {
    let raw = r#"{"type":"FUTURE_EVENT_NOT_YET_DEFINED","run_id":"run_1","timestamp":1}"#;
    let decoded: AgUiEvent = serde_json::from_str(raw).expect("unknown type must decode");
    assert!(matches!(decoded, AgUiEvent::Unknown));
    assert_eq!(decoded.kind(), AgUiEventKind::Unknown);
    // `Unknown` is serde-lossy (no payload retained) — documenting the
    // contract so callers don't expect to recover the original type
    // string or run_id.
    assert_eq!(decoded.run_id(), "");
}

/// The default filter ships allow-all across every kind declared in
/// [`AgUiEventKind::ALL`]. If a new variant lands without being added
/// to `allow_all`'s set, the pipeline silently drops it on emit and the
/// regression is invisible until a consumer reports missing events.
/// This test fails loudly in that case so the author is forced to make
/// a conscious allow/deny decision.
#[test]
fn default_filter_permits_every_declared_kind() {
    let filter = AgUiEventFilter::default();
    for kind in AgUiEventKind::ALL {
        assert!(
            filter.allows(kind),
            "default filter must allow {kind:?} — either add it to allow_all or explicitly exclude"
        );
    }
}
