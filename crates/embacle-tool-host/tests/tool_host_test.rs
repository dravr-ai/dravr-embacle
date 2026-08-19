// ABOUTME: The loopback tool host serves a caller's tools and revokes with the session guard
// ABOUTME: Drives real HTTP against the bound listener — no mock transport
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! What an ACP agent sees when it talks to the host.
//!
//! These drive real HTTP against a real bound listener, because every property
//! worth pinning here lives in the transport: that a bearer reaches the right
//! session, that a revoked one cannot, and that a tool outside the session's
//! surface is refused rather than forwarded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use embacle::types::{McpServerConfig, McpTransport, RunnerError};
use embacle::{McpToolDefinition, McpToolExecutor};
use embacle_tool_host::{ToolHost, ToolHostConfig};
use serde_json::{json, Value};

/// Records what it was asked to run, and answers with a marker the test can find.
struct RecordingExecutor {
    calls: AtomicUsize,
}

#[async_trait]
impl McpToolExecutor for RecordingExecutor {
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({ "ran": tool_name, "got": arguments }))
    }
}

fn one_tool() -> Vec<McpToolDefinition> {
    vec![McpToolDefinition {
        name: "get_activities".to_owned(),
        description: "List recent activities".to_owned(),
        input_schema: json!({ "type": "object", "properties": {} }),
    }]
}

/// Extract the bearer the session minted, as an agent would read it.
fn bearer_of(servers: &[McpServerConfig]) -> String {
    match &servers[0].transport {
        McpTransport::Http { headers, .. } => headers[0].value.clone(),
        _ => panic!("the host must publish an HTTP transport"),
    }
}

fn url_of(servers: &[McpServerConfig]) -> String {
    match &servers[0].transport {
        McpTransport::Http { url, .. } => url.clone(),
        _ => panic!("the host must publish an HTTP transport"),
    }
}

async fn post(url: &str, bearer: &str, body: Value) -> (u16, Value) {
    let response = reqwest::Client::new()
        .post(url)
        .header("authorization", bearer)
        .json(&body)
        .send()
        .await
        .expect("the host answers");
    let status = response.status().as_u16();
    let parsed = response.json::<Value>().await.unwrap_or(Value::Null);
    (status, parsed)
}

fn call(tool: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool, "arguments": {} }
    })
}

/// The whole point: an agent holding the session bearer can list and run the
/// caller's tools, and the caller's own executor is what runs them.
#[tokio::test]
async fn an_agent_lists_and_runs_the_callers_tools() {
    let host = ToolHost::bind(ToolHostConfig {
        server_name: "dravr".to_owned(),
        ..ToolHostConfig::default()
    })
    .await
    .expect("binds on loopback");

    let executor = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });
    let session = host.open_session(one_tool(), executor.clone());
    let servers = session.mcp_servers();
    let (url, bearer) = (url_of(&servers), bearer_of(&servers));

    // The server name namespaces the tools, which is how the agent reports them.
    assert_eq!(servers[0].name, "dravr");

    let (status, listed) = post(
        &url,
        &bearer,
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
    )
    .await;
    assert_eq!(
        status, 200,
        "an authenticated tools/list is served: {listed}"
    );
    let names = listed["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["get_activities"],
        "the agent sees exactly the caller's surface"
    );

    let (status, result) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(status, 200, "a tools/call is served: {result}");
    assert_eq!(
        executor.calls.load(Ordering::SeqCst),
        1,
        "the CALLER's executor ran it — not embacle, not the agent"
    );
    assert_eq!(session.calls_served(), 1);
}

/// Dropping the guard revokes the bearer. This is the property the guard exists
/// for: a turn that ended must not leave a credential an orphaned agent
/// subprocess can still spend on an irreversible action.
#[tokio::test]
async fn dropping_the_session_revokes_the_bearer() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let executor = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });

    let session = host.open_session(one_tool(), executor.clone());
    let servers = session.mcp_servers();
    let (url, bearer) = (url_of(&servers), bearer_of(&servers));

    let (status, _) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(status, 200, "the bearer works while the session is open");
    assert_eq!(host.open_sessions(), 1);

    drop(session);
    assert_eq!(host.open_sessions(), 0, "the guard removed its session");

    let (status, _) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(
        status, 401,
        "the same bearer must be refused once the turn is over"
    );
    assert_eq!(
        executor.calls.load(Ordering::SeqCst),
        1,
        "the revoked call must not have reached the caller's executor"
    );
}

/// A tool outside the session's surface is refused, not forwarded. The agent
/// must not be able to reach past what its caller published for this turn.
#[tokio::test]
async fn a_tool_outside_the_session_surface_is_refused() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let executor = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });
    let session = host.open_session(one_tool(), executor.clone());
    let servers = session.mcp_servers();

    let (status, result) = post(
        &url_of(&servers),
        &bearer_of(&servers),
        call("delete_everything"),
    )
    .await;

    assert_eq!(status, 200, "a refusal is in-band, not a transport error");
    assert!(
        result["result"]["isError"].as_bool().unwrap_or(false),
        "an ungranted tool must come back as an error result: {result}"
    );
    assert_eq!(
        executor.calls.load(Ordering::SeqCst),
        0,
        "an ungranted tool must never reach the caller's executor"
    );
}

/// No bearer at all is refused.
#[tokio::test]
async fn an_unauthenticated_call_is_refused() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let executor = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });
    let session = host.open_session(one_tool(), executor.clone());
    let servers = session.mcp_servers();

    let (status, _) = post(&url_of(&servers), "", call("get_activities")).await;
    assert_eq!(status, 401);
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
}
