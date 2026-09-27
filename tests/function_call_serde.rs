// ABOUTME: The serde shape of the tool-calling FunctionCall a host persists or forwards
// ABOUTME: Pins the {"name", "args"} wire form and its round trip into ToolCallRequest
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::unwrap_used)]
use embacle::{FunctionCall, ToolCallRequest};
use serde_json::json;

#[test]
fn function_call_round_trips_through_json() {
    let call = FunctionCall {
        name: "get_activities".to_owned(),
        args: json!({"limit": 10, "provider": "strava", "filters": {"sport": ["run", "ride"]}}),
    };
    let wire = serde_json::to_value(&call).unwrap();
    assert_eq!(
        wire,
        json!({"name": "get_activities", "args": {"limit": 10, "provider": "strava", "filters": {"sport": ["run", "ride"]}}})
    );
    let back: FunctionCall = serde_json::from_value(wire).unwrap();
    assert_eq!(back.name, call.name);
    assert_eq!(back.args, call.args);

    // The call a provider returned survives the trip into the request type.
    let tc: ToolCallRequest = back.into();
    assert_eq!(tc.function_name, "get_activities");
    assert_eq!(tc.arguments["filters"]["sport"][1], "ride");
}
