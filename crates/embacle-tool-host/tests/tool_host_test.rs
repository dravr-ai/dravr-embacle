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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use embacle::types::{McpServerConfig, McpTransport, RunnerError};
use embacle::{McpToolDefinition, McpToolExecutor};
use embacle_tool_host::{StaticSurface, ToolHost, ToolHostConfig, ToolOutcome, ToolSurface};
use serde_json::{json, Value};
use tokio::time::sleep;

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
    let session = host.open_session(Arc::new(StaticSurface::new(one_tool(), executor.clone())));
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

    let session = host.open_session(Arc::new(StaticSurface::new(one_tool(), executor.clone())));
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
    let session = host.open_session(Arc::new(StaticSurface::new(one_tool(), executor.clone())));
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
    let session = host.open_session(Arc::new(StaticSurface::new(one_tool(), executor.clone())));
    let servers = session.mcp_servers();

    let (status, _) = post(&url_of(&servers), "", call("get_activities")).await;
    assert_eq!(status, 401);
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
}

/// A surface whose visible set changes. The whole reason listing is asked per
/// call rather than fixed at session open.
struct WithdrawingSurface {
    visible: AtomicBool,
    calls: AtomicUsize,
}

#[async_trait]
impl ToolSurface for WithdrawingSurface {
    async fn list_tools(&self) -> Vec<McpToolDefinition> {
        if self.visible.load(Ordering::SeqCst) {
            one_tool()
        } else {
            Vec::new()
        }
    }

    async fn call(&self, tool_name: &str, _arguments: &Value) -> ToolOutcome {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolOutcome::json(json!({ "ran": tool_name }))
    }
}

/// A tool withdrawn mid-session disappears from the listing AND stops being
/// callable — without reopening the session.
///
/// This is the guarantee a caller needs to gate on state that moves: a role
/// change, a quota crossing, an interview that must withhold a tool until it
/// finishes. Fixing the list at open would let the agent keep calling a tool
/// the caller has since withdrawn.
#[tokio::test]
async fn a_tool_withdrawn_mid_session_stops_being_listed_and_callable() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let surface = Arc::new(WithdrawingSurface {
        visible: AtomicBool::new(true),
        calls: AtomicUsize::new(0),
    });
    let session = host.open_session(surface.clone());
    let servers = session.mcp_servers();
    let (url, bearer) = (url_of(&servers), bearer_of(&servers));

    let (_, listed) = post(
        &url,
        &bearer,
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
    )
    .await;
    assert_eq!(
        listed["result"]["tools"].as_array().map(Vec::len),
        Some(1),
        "visible while the surface says so"
    );

    let (status, _) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(status, 200);
    assert_eq!(surface.calls.load(Ordering::SeqCst), 1);

    // The caller withdraws it. No session reopen, no new bearer.
    surface.visible.store(false, Ordering::SeqCst);

    let (_, listed) = post(
        &url,
        &bearer,
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
    )
    .await;
    assert_eq!(
        listed["result"]["tools"].as_array().map(Vec::len),
        Some(0),
        "the listing must reflect the withdrawal immediately"
    );

    let (status, result) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(status, 200, "a refusal is in-band");
    assert!(
        result["result"]["isError"].as_bool().unwrap_or(false),
        "a withdrawn tool must be refused: {result}"
    );
    assert_eq!(
        surface.calls.load(Ordering::SeqCst),
        1,
        "the withdrawn call must never reach the caller's surface"
    );
}

/// A refusal is neither success nor transport failure: the model must receive
/// the reason AND see it flagged.
#[tokio::test]
async fn a_refusal_carries_its_reason_and_is_flagged() {
    struct RefusingSurface;

    #[async_trait]
    impl ToolSurface for RefusingSurface {
        async fn list_tools(&self) -> Vec<McpToolDefinition> {
            one_tool()
        }
        async fn call(&self, _tool_name: &str, _arguments: &Value) -> ToolOutcome {
            ToolOutcome::refused("daily limit reached — try tomorrow")
                .with_structured(json!({ "code": "quota_exceeded" }))
        }
    }

    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let session = host.open_session(Arc::new(RefusingSurface));
    let servers = session.mcp_servers();

    let (status, result) = post(
        &url_of(&servers),
        &bearer_of(&servers),
        call("get_activities"),
    )
    .await;

    assert_eq!(status, 200, "a refusal is served, not a transport error");
    assert!(
        result["result"]["isError"].as_bool().unwrap_or(false),
        "must be flagged: {result}"
    );
    let text = result["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(
        text.contains("daily limit reached"),
        "the model must receive the reason, got {text:?}"
    );
    assert_eq!(
        result["result"]["structuredContent"]["code"].as_str(),
        Some("quota_exceeded"),
        "structured content must survive: {result}"
    );
}

/// Two turns in flight at once stay isolated: each sees only its own tools and
/// reaches only its own executor.
///
/// Every other test opens one session, which is the case that cannot fail. A
/// server is only useful if the normal case — several turns at once — holds.
#[tokio::test]
async fn concurrent_sessions_do_not_leak_into_each_other() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");

    let alice_exec = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });
    let bob_exec = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });

    let alice_tool = vec![McpToolDefinition {
        name: "alice_only".to_owned(),
        description: "Alice's tool".to_owned(),
        input_schema: json!({ "type": "object", "properties": {} }),
    }];

    let alice = host.open_session(Arc::new(StaticSurface::new(alice_tool, alice_exec.clone())));
    let bob = host.open_session(Arc::new(StaticSurface::new(one_tool(), bob_exec.clone())));
    assert_eq!(host.open_sessions(), 2);

    let a = alice.mcp_servers();
    let b = bob.mcp_servers();
    let (a_url, a_bearer) = (url_of(&a), bearer_of(&a));
    let (b_url, b_bearer) = (url_of(&b), bearer_of(&b));

    // Same listener, different credentials.
    assert_eq!(a_url, b_url);
    assert_ne!(a_bearer, b_bearer, "each session mints its own bearer");

    // Each sees only its own surface.
    let (_, a_list) = post(
        &a_url,
        &a_bearer,
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
    )
    .await;
    assert_eq!(
        a_list["result"]["tools"][0]["name"].as_str(),
        Some("alice_only")
    );

    // Bob's bearer must not reach Alice's tool.
    let (status, result) = post(&b_url, &b_bearer, call("alice_only")).await;
    assert_eq!(status, 200);
    assert!(
        result["result"]["isError"].as_bool().unwrap_or(false),
        "Bob must not be able to call Alice's tool: {result}"
    );
    assert_eq!(
        alice_exec.calls.load(Ordering::SeqCst),
        0,
        "Alice's executor must never be reached by Bob's session"
    );

    // Each still works on its own.
    let (status, _) = post(&a_url, &a_bearer, call("alice_only")).await;
    assert_eq!(status, 200);
    let (status, _) = post(&b_url, &b_bearer, call("get_activities")).await;
    assert_eq!(status, 200);
    assert_eq!(alice_exec.calls.load(Ordering::SeqCst), 1);
    assert_eq!(bob_exec.calls.load(Ordering::SeqCst), 1);

    // Revoking one leaves the other serving.
    drop(alice);
    assert_eq!(host.open_sessions(), 1);
    let (status, _) = post(&a_url, &a_bearer, call("alice_only")).await;
    assert_eq!(status, 401, "Alice's bearer is dead");
    let (status, _) = post(&b_url, &b_bearer, call("get_activities")).await;
    assert_eq!(status, 200, "Bob is unaffected by Alice's revocation");
}

/// `shutdown` actually stops the listener.
///
/// It binds a port in a long-lived process, so "does it let go" is not a
/// question to leave to inspection.
#[tokio::test]
async fn shutdown_stops_the_listener() {
    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let executor = Arc::new(RecordingExecutor {
        calls: AtomicUsize::new(0),
    });
    let session = host.open_session(Arc::new(StaticSurface::new(one_tool(), executor)));
    let servers = session.mcp_servers();
    let (url, bearer) = (url_of(&servers), bearer_of(&servers));

    let (status, _) = post(&url, &bearer, call("get_activities")).await;
    assert_eq!(status, 200, "serving before shutdown");

    host.shutdown();
    // Idempotent, and safe to call twice from separate teardown paths.
    host.shutdown();

    // Give the accept loop a moment to unwind.
    for _ in 0..40 {
        if reqwest::Client::new()
            .post(&url)
            .header("authorization", &bearer)
            .json(&call("get_activities"))
            .send()
            .await
            .is_err()
        {
            return; // refused — the listener is gone
        }
        sleep(Duration::from_millis(50)).await;
    }
    panic!("the listener was still accepting two seconds after shutdown");
}

/// The simple adapter still hands the model something to branch on.
///
/// `McpToolExecutor` returns `Result<Value, RunnerError>`, so a declining tool
/// can only come back as `Err`. Flattening that to prose would leave the model
/// with nothing machine-readable — the exact fidelity loss `ToolOutcome`
/// exists to prevent, reintroduced by the convenience wrapper.
#[tokio::test]
async fn the_static_adapter_preserves_the_error_kind() {
    struct FailingExecutor;

    #[async_trait]
    impl McpToolExecutor for FailingExecutor {
        async fn execute(&self, _tool: &str, _args: &Value) -> Result<Value, RunnerError> {
            Err(RunnerError::auth_failure("token expired"))
        }
    }

    let host = ToolHost::bind(ToolHostConfig::default())
        .await
        .expect("binds");
    let session = host.open_session(Arc::new(StaticSurface::new(
        one_tool(),
        Arc::new(FailingExecutor),
    )));
    let servers = session.mcp_servers();

    let (status, result) = post(
        &url_of(&servers),
        &bearer_of(&servers),
        call("get_activities"),
    )
    .await;

    assert_eq!(status, 200, "a failing tool is still an in-band result");
    assert!(
        result["result"]["isError"].as_bool().unwrap_or(false),
        "must be flagged as an error: {result}"
    );
    assert_eq!(
        result["result"]["structuredContent"]["error_kind"].as_str(),
        Some("AuthFailure"),
        "the kind must survive as a machine-readable discriminator: {result}"
    );
    let text = result["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(
        text.contains("token expired"),
        "the message must reach the model, got {text:?}"
    );
}
