// ABOUTME: Live proof that the Copilot SDK provider calls a caller-supplied tool over the loopback MCP host
// ABOUTME: Gated by EMBACLE_E2E_COPILOT_SDK=1; the observed tool call must carry its name, arguments and result
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The ACP adapter reports a tool call as `{id, title, status}` and nothing a
//! host could persist as a tool round. The SDK transport reports the runtime's
//! own `tool.execution_start` / `tool.execution_complete` events, so the
//! observation carries the tool's name, the arguments the model passed and
//! the result handed back. This test pins that over a real turn: a tool the
//! host serves, called by the runtime, reported by name.

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use embacle::types::{ChatMessage, ChatRequest, RunnerError};
use embacle::{
    CopilotSdkConfig, CopilotSdkRunner, HeadlessTurnProvider, McpToolDefinition, McpToolExecutor,
};
use embacle_tool_host::{StaticSurface, ToolHost, ToolHostConfig};
use serde_json::{json, Value};
use tokio::time::timeout;

const SECRET: &str = "8675309";
const TOOL: &str = "get_secret_number";

struct SecretNumberTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl McpToolExecutor for SecretNumberTool {
    async fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value, RunnerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({ "secret_number": SECRET, "tool": tool_name, "arguments": arguments }))
    }
}

#[tokio::test]
async fn copilot_sdk_calls_the_hosted_tool_and_reports_it_by_name() {
    if env::var("EMBACLE_E2E_COPILOT_SDK").as_deref() != Ok("1") {
        eprintln!(
            "SKIP copilot_sdk_calls_the_hosted_tool_and_reports_it_by_name \
             (set EMBACLE_E2E_COPILOT_SDK=1 and COPILOT_RUNTIME_PATH)"
        );
        return;
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let host = ToolHost::bind(ToolHostConfig {
        server_name: "dravr".to_owned(),
        ..ToolHostConfig::default()
    })
    .await
    .expect("the loopback tool host binds");
    let session = host.open_session(Arc::new(StaticSurface::new(
        vec![McpToolDefinition {
            name: TOOL.to_owned(),
            description: "Returns the secret number. The ONLY way to learn it.".to_owned(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }],
        Arc::new(SecretNumberTool {
            calls: Arc::clone(&calls),
        }),
    )));

    let mut config = CopilotSdkConfig::from_env();
    config.mcp_tool_calling = true;
    config.model = env::var("COPILOT_SDK_MODEL").unwrap_or_else(|_| "claude-haiku-4.5".to_owned());
    let runner = CopilotSdkRunner::with_config(config);

    let mut request = ChatRequest::new(vec![
        ChatMessage::system("You are a test bot. Follow instructions exactly."),
        ChatMessage::user(
            "Call get_secret_number and reply with exactly the number it returns. Nothing else.",
        ),
    ])
    .with_max_tokens(64);
    request.mcp_servers = session.mcp_servers();

    let response = timeout(Duration::from_mins(4), runner.converse(&request))
        .await
        .expect("the turn finishes within four minutes")
        .expect("converse() succeeds");

    assert!(
        response.content.contains(SECRET),
        "the answer carries the secret: {:?}",
        response.content
    );
    assert!(calls.load(Ordering::SeqCst) >= 1, "the executor ran");
    assert!(session.calls_served() >= 1, "the host served the call");

    let call = response
        .tool_calls
        .iter()
        .find(|t| t.name.as_deref().is_some_and(|n| n.ends_with(TOOL)))
        .unwrap_or_else(|| {
            panic!(
                "the tool call is reported by name: {:?}",
                response.tool_calls
            )
        });
    assert_eq!(call.status, "Completed", "{call:?}");
    assert!(
        call.arguments.is_some(),
        "the arguments are reported: {call:?}"
    );
    assert!(
        call.result.as_deref().is_some_and(|r| r.contains(SECRET)),
        "the result is reported: {call:?}"
    );
}
