// ABOUTME: Integration tests proving a configured UHP harness is runnable and its settings reach the CLI
// ABOUTME: A recording fake binary captures the argv and the staged skills each task is started with
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]
// The stand-in harness is a shell script, as in the workspace's other
// fake-binary tests.
#![cfg(unix)]

use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, LazyLock};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use embacle::config::CliRunnerType;
use embacle_mcp::ServerState;
use embacle_server::state::AppState;
use embacle_server::uhp;
use embacle_server::uhp::files::WORKDIR_ENV;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

/// The folders every test in this binary shares.
///
/// The binary overrides and `UHP_WORKDIR` are process-global, so they are set
/// exactly once, before any handler reads them; each test then tells its own
/// run apart by a prompt no other test sends.
struct Fixture {
    /// Where the fake harness writes one record per invocation.
    records: PathBuf,
}

/// Stands in for every harness CLI: records the arguments it was started with,
/// one per line, copies the plugin directory it was pointed at before the task
/// removes it, and answers as a successful Claude Code turn.
const FAKE_HARNESS: &str = r#"#!/bin/sh
rec="__RECORDS__/$$"
prev=""
for arg in "$@"; do
  printf '%s\n' "$arg" >> "$rec.args"
  if [ "$prev" = "--plugin-dir" ]; then /bin/cp -R "$arg" "$rec.plugin"; fi
  prev="$arg"
done
printf '%s' '{"type":"result","subtype":"success","is_error":false,"result":"ok","session_id":"fake"}'
"#;

static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("uhp-harness-settings-{}", process::id()));
    let records = root.join("records");
    fs::create_dir_all(&records).expect("create records folder");

    let fake = root.join("fake-harness");
    fs::write(
        &fake,
        FAKE_HARNESS.replace("__RECORDS__", records.to_str().expect("utf-8 path")),
    )
    .expect("write fake harness");
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).expect("make it executable");

    env::set_var(WORKDIR_ENV, root.join("work"));
    // Claude Code and the Copilot CLI apply every setting; Gemini CLI applies
    // none, so it is the base a refusal is proven on.
    env::set_var("CLAUDE_CODE_BINARY", &fake);
    env::set_var("COPILOT_BINARY", &fake);
    env::set_var("GEMINI_CLI_BINARY", &fake);
    env::remove_var("EMBACLE_API_KEY");
    Fixture { records }
});

/// The whole UHP router, over state of its own.
fn app() -> Router {
    LazyLock::force(&FIXTURE);
    uhp::router(AppState::new(Arc::new(ServerState::new(
        CliRunnerType::ClaudeCode,
    ))))
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .expect("build request");
    let response = app.clone().oneshot(request).await.expect("send request");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes();
    let json = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).expect("body is JSON")
    };
    (status, json)
}

/// Create a configured harness and return its id.
async fn configure(app: &Router, config: Value) -> String {
    let (status, body) = send(app, "POST", "/v1/harnesses", Some(config)).await;
    assert_eq!(status, StatusCode::OK, "harness is created: {body}");
    body["id"].as_str().expect("harness id").to_owned()
}

/// Run one task on `harness_id` with a prompt only this test sends.
async fn run_on(app: &Router, harness_id: &str, prompt: &str) -> (StatusCode, Value) {
    send(
        app,
        "POST",
        "/v1/responses",
        Some(json!({"input": prompt, "metadata": {"harness_id": harness_id}})),
    )
    .await
}

/// A prompt no other test sends, so its record can be found.
fn unique_prompt() -> String {
    format!("prompt-{}", Uuid::new_v4().simple())
}

/// The argv of the one invocation that carried `prompt`, and where its copy of
/// the plugin directory would be.
fn invocation_for(prompt: &str) -> Option<(Vec<String>, PathBuf)> {
    fs::read_dir(&FIXTURE.records)
        .expect("records folder")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "args"))
        .find_map(|p| {
            let args: Vec<String> = fs::read_to_string(&p)
                .expect("read record")
                .lines()
                .map(str::to_owned)
                .collect();
            args.iter()
                .any(|a| a.contains(prompt))
                .then(|| (args, p.with_extension("plugin")))
        })
}

/// The values that follow `flag` up to the next flag.
fn values_after<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let at = args
        .iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("{flag} is on the command line: {args:?}"));
    args[at + 1..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect()
}

/// A skill bundle with a text manifest and a binary member.
fn greeter_skill() -> Value {
    json!({
        "name": "greeter",
        "enabled": true,
        "files": [
            {"path": "SKILL.md", "content": "---\nname: greeter\ndescription: Greets\n---\nSay hi.\n"},
            {"path": "assets/blob.bin", "content_b64": "AAECAwQF"}
        ]
    })
}

/// Assert the staged plugin copy holds the greeter bundle and nothing disabled.
fn assert_greeter_staged(plugin: &Path) {
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(plugin.join(".claude-plugin/plugin.json")).expect("plugin manifest"),
    )
    .expect("manifest is JSON");
    assert_eq!(manifest["name"], "harness");
    assert_eq!(
        fs::read_to_string(plugin.join("skills/greeter/SKILL.md")).expect("skill manifest"),
        "---\nname: greeter\ndescription: Greets\n---\nSay hi.\n"
    );
    assert_eq!(
        fs::read(plugin.join("skills/greeter/assets/blob.bin")).expect("binary member"),
        vec![0_u8, 1, 2, 3, 4, 5],
        "a content_b64 member is written as its decoded bytes"
    );
    assert!(
        !plugin.join("skills/dormant").exists(),
        "a disabled skill is not handed to the harness"
    );
}

#[tokio::test]
async fn a_configured_harness_runs_without_offering_its_disabled_tools() {
    let app = app();
    let id = configure(
        &app,
        json!({"name": "reviewer", "base": "claude-code", "disabledTools": ["WebSearch", "Bash"]}),
    )
    .await;

    let (status, listed) = send(&app, "GET", "/v1/harnesses", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        listed["harnesses"]
            .as_array()
            .expect("harness list")
            .iter()
            .any(|h| h["id"] == id.as_str()),
        "a configured harness is listed beside the discovered ones: {listed}"
    );

    let (status, models) = send(&app, "GET", &format!("/v1/harnesses/{id}/models"), None).await;
    assert_eq!(status, StatusCode::OK, "{models}");
    assert_eq!(models["harness_id"], id.as_str());
    assert_eq!(models["backend"], "claude-code");

    let prompt = unique_prompt();
    let (status, response) = run_on(&app, &id, &prompt).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let (args, _) = invocation_for(&prompt).expect("the harness was started");
    assert_eq!(
        values_after(&args, "--disallowed-tools"),
        vec!["WebSearch", "Bash"],
        "Claude Code removes these from the tools the model is offered"
    );
}

#[tokio::test]
async fn a_configured_harness_hands_its_skills_and_mcp_servers_to_claude_code() {
    let app = app();
    let id = configure(
        &app,
        json!({
            "name": "summariser",
            "base": "claude-code",
            "skills": [
                greeter_skill(),
                {"name": "dormant", "enabled": false, "files": [{"path": "SKILL.md", "content": "x"}]}
            ],
            "mcpServers": [
                {"name": "docs", "url": "http://127.0.0.1:9/mcp", "transport": "http", "enabled": true},
                {"name": "offline", "url": "http://127.0.0.1:9/off", "enabled": false}
            ]
        }),
    )
    .await;

    let prompt = unique_prompt();
    let (status, response) = run_on(&app, &id, &prompt).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let (args, plugin) = invocation_for(&prompt).expect("the harness was started");
    let mcp: Value =
        serde_json::from_str(values_after(&args, "--mcp-config")[0]).expect("mcp config JSON");
    assert_eq!(mcp["mcpServers"]["docs"]["url"], "http://127.0.0.1:9/mcp");
    assert_eq!(mcp["mcpServers"]["docs"]["type"], "http");
    assert!(
        mcp["mcpServers"].get("offline").is_none(),
        "a disabled MCP server is never contacted: {mcp}"
    );

    let staged = values_after(&args, "--plugin-dir");
    assert_eq!(staged.len(), 1, "one plugin directory: {args:?}");
    assert_greeter_staged(&plugin);
    assert!(
        !Path::new(staged[0]).exists(),
        "the staged bundle is removed once its task is done"
    );

    let session = response["metadata"]["session_id"]
        .as_str()
        .expect("session id");
    let (status, files) = send(&app, "GET", &format!("/v1/sessions/{session}/files"), None).await;
    assert_eq!(status, StatusCode::OK, "{files}");
    assert_eq!(
        files["files"],
        json!([]),
        "a staged skill is not an artifact the harness produced"
    );
}

#[tokio::test]
async fn a_configured_harness_hands_all_three_settings_to_the_copilot_cli() {
    let app = app();
    let id = configure(
        &app,
        json!({
            "name": "copilot-reviewer",
            "base": "copilot",
            "skills": [greeter_skill()],
            "mcpServers": [{"name": "docs", "url": "http://127.0.0.1:9/sse", "transport": "sse", "enabled": true}],
            "disabledTools": ["web_fetch"]
        }),
    )
    .await;

    let prompt = unique_prompt();
    let (status, response) = run_on(&app, &id, &prompt).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let (args, plugin) = invocation_for(&prompt).expect("the harness was started");
    assert_eq!(
        values_after(&args, "--excluded-tools"),
        vec!["web_fetch"],
        "the Copilot CLI removes these from the tools the model is offered"
    );
    let mcp: Value = serde_json::from_str(values_after(&args, "--additional-mcp-config")[0])
        .expect("mcp config JSON");
    assert_eq!(mcp["mcpServers"]["docs"]["url"], "http://127.0.0.1:9/sse");
    assert_eq!(mcp["mcpServers"]["docs"]["type"], "sse");
    assert_greeter_staged(&plugin);
}

#[tokio::test]
async fn a_setting_the_base_cannot_apply_refuses_the_task_before_it_starts() {
    let app = app();
    let id = configure(
        &app,
        json!({
            "name": "gemini-reviewer",
            "base": "gemini",
            "skills": [greeter_skill()],
            "disabledTools": ["web_fetch"]
        }),
    )
    .await;

    let prompt = unique_prompt();
    let (status, body) = run_on(&app, &id, &prompt).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_harness_setting");
    assert_eq!(body["error"]["detail"]["base"], "gemini");
    assert_eq!(
        body["error"]["detail"]["unsupported"],
        json!(["skills", "disabledTools"])
    );
    assert!(
        invocation_for(&prompt).is_none(),
        "a refused task never starts the harness"
    );
}

#[tokio::test]
async fn a_skill_bundle_that_would_leave_its_folder_is_refused_at_config_time() {
    let app = app();
    for skill in [
        json!({"name": "../escape", "enabled": true, "files": [{"path": "SKILL.md", "content": "x"}]}),
        json!({"name": "ok", "enabled": true, "files": [
            {"path": "SKILL.md", "content": "x"},
            {"path": "../../escape.md", "content": "x"}
        ]}),
        json!({"name": "ok", "enabled": true, "files": [
            {"path": "SKILL.md", "content": "x"},
            {"path": "/etc/escape.md", "content": "x"}
        ]}),
        json!({"name": "ok", "enabled": true, "files": [
            {"path": "SKILL.md", "content": "x"},
            {"path": "blob.bin", "content_b64": "not base64!"}
        ]}),
    ] {
        let (status, body) = send(
            &app,
            "POST",
            "/v1/harnesses",
            Some(json!({"name": "n", "base": "claude-code", "skills": [skill]})),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{skill} is refused: {body}"
        );
        assert_eq!(body["error"]["code"], "invalid_input");
    }
}
