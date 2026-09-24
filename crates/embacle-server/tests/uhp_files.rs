// ABOUTME: Integration tests confining UHP artifact access to the session that produced the files
// ABOUTME: Traversal, unknown and symlinked container ids are refused; a held session still serves
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::sync::{Arc, LazyLock};

use axum::body::Body;
use axum::http::header::X_CONTENT_TYPE_OPTIONS;
use axum::http::{Request, StatusCode};
use axum::routing::get;
use axum::Router;
use embacle::config::CliRunnerType;
use embacle_mcp::ServerState;
use embacle_server::router;
use embacle_server::state::AppState;
use embacle_server::uhp::files::{self, WORKDIR_ENV};
use embacle_server::uhp::state::UhpState;
use embacle_server::uhp::tasks::Response;
use http_body_util::BodyExt;
use serde_json::{Map, Value};
use tower::ServiceExt;
use uuid::Uuid;

/// Bytes of the file that sits outside every session folder.
const OUTSIDE_BYTES: &[u8] = b"outside-secret";

/// Name of the file that sits outside every session folder.
const OUTSIDE_NAME: &str = "secret.txt";

/// The folders every test in this binary shares.
///
/// `UHP_WORKDIR` is process-global, so it is set exactly once, to one folder,
/// before any handler reads it; each test then works in sessions of its own.
struct Fixture {
    /// The parent of the workdir base, which is where a `..` container id lands.
    outside: PathBuf,
    /// The workdir base, `UHP_WORKDIR`.
    base: PathBuf,
}

static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    let outside =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("uhp-files-{}", process::id()));
    let base = outside.join("base");
    fs::create_dir_all(&base).expect("create workdir base");
    fs::write(outside.join(OUTSIDE_NAME), OUTSIDE_BYTES).expect("write outside file");
    env::set_var(WORKDIR_ENV, &base);
    Fixture { outside, base }
});

/// The two artifact routes, served over state the test can put sessions into.
fn app(state: UhpState) -> Router {
    Router::new()
        .route("/v1/sessions/{session_id}/files", get(files::session_files))
        .route(
            "/v1/containers/{container_id}/files/{file_id}/content",
            get(files::container_file),
        )
        .with_state(state)
}

fn uhp_state() -> UhpState {
    UhpState::new(Arc::new(ServerState::new(CliRunnerType::Copilot)))
}

/// A session id of the shape the server mints.
fn minted_session_id() -> String {
    format!("sess_{}", Uuid::new_v4().simple())
}

/// Record a session in the store, as a finished task would, and return its id.
fn hold_session(state: &UhpState) -> String {
    let session_id = minted_session_id();
    state.store.put(
        &session_id,
        Response {
            id: format!("resp_{}", Uuid::new_v4().simple()),
            object: "response",
            created_at: 1,
            status: "completed",
            error: None,
            previous_response_id: None,
            model: "m".to_owned(),
            output: Vec::new(),
            store: true,
            usage: None,
            metadata: Map::new(),
        },
    );
    session_id
}

/// Write `bytes` at `relative` inside a session's working folder.
fn write_artifact(session_id: &str, relative: &str, bytes: &[u8]) -> PathBuf {
    let path = FIXTURE.base.join(session_id).join(relative);
    fs::create_dir_all(path.parent().expect("artifact has a parent")).expect("create folder");
    fs::write(&path, bytes).expect("write artifact");
    path
}

/// Percent-encode every byte of `raw` outside the unreserved set, so a whole
/// path travels as one URL segment.
fn encode_segment(raw: &str) -> String {
    raw.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                char::from(b).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

async fn send(app: Router, uri: &str) -> (StatusCode, Vec<u8>, Option<String>) {
    let response = app
        .oneshot(
            Request::get(uri)
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("send request");
    let status = response.status();
    let nosniff = response
        .headers()
        .get(X_CONTENT_TYPE_OPTIONS)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec();
    (status, bytes, nosniff)
}

/// The artifacts a session lists, as `(filename, id)` pairs.
async fn listing(state: &UhpState, session_id: &str) -> Vec<(String, String)> {
    let (status, body, _) = send(
        app(state.clone()),
        &format!("/v1/sessions/{session_id}/files"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "a held session lists its files");
    let json: Value = serde_json::from_slice(&body).expect("listing is JSON");
    json["files"]
        .as_array()
        .expect("files is an array")
        .iter()
        .map(|f| {
            (
                f["filename"].as_str().expect("filename").to_owned(),
                f["id"].as_str().expect("id").to_owned(),
            )
        })
        .collect()
}

/// The file id a client learns for `relative` from a session of its own.
///
/// Ids are derived from the path inside a session, so the same relative name
/// carries the same id in every container — which is exactly what lets a
/// caller aim a download at a folder it does not own.
async fn id_learned_for(state: &UhpState, relative: &str) -> String {
    let own = hold_session(state);
    write_artifact(&own, relative, b"own-bytes");
    listing(state, &own)
        .await
        .into_iter()
        .find(|(name, _)| name == relative)
        .map(|(_, id)| id)
        .expect("the client's own session lists the file it wrote")
}

/// Assert a download was refused as not found, without leaking any bytes.
fn assert_refused(status: StatusCode, body: &[u8], probe: &str) {
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "container id {probe:?} must be refused as not found"
    );
    let json: Value = serde_json::from_slice(body).expect("a refusal is the error envelope");
    assert_eq!(
        json["error"]["code"], "file_not_found",
        "container id {probe:?} must answer exactly as a missing file does"
    );
    assert!(
        !String::from_utf8_lossy(body).contains("outside-secret"),
        "container id {probe:?} leaked a file outside the session folder"
    );
}

#[tokio::test]
async fn a_held_session_lists_and_serves_its_own_artifacts() {
    let state = uhp_state();
    let session = hold_session(&state);
    write_artifact(&session, "report.txt", b"own-artifact");
    write_artifact(&session, "notes/deep.md", b"nested");

    let listed = listing(&state, &session).await;
    let names: Vec<&str> = listed.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["notes/deep.md", "report.txt"]);

    let report = &listed[1].1;
    let (status, body, nosniff) = send(
        app(state.clone()),
        &format!("/v1/containers/{session}/files/{report}/content"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, b"own-artifact", "the artifact's own bytes, unwrapped");
    assert_eq!(nosniff.as_deref(), Some("nosniff"));
}

#[tokio::test]
async fn an_unknown_container_is_not_found_even_when_its_folder_exists() {
    let state = uhp_state();
    let id = id_learned_for(&state, "report.txt").await;

    // Well-formed, never held by this server, yet a folder by that name exists
    // on disk: the store is per-process, so a restart leaves exactly this.
    let orphan = minted_session_id();
    write_artifact(&orphan, "report.txt", b"orphaned-bytes");

    let (status, body, _) = send(
        app(state.clone()),
        &format!("/v1/containers/{orphan}/files/{id}/content"),
    )
    .await;
    assert_refused(status, &body, &orphan);
    assert!(!String::from_utf8_lossy(&body).contains("orphaned-bytes"));

    let (status, body, _) = send(app(state), &format!("/v1/sessions/{orphan}/files")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let json: Value = serde_json::from_slice(&body).expect("error envelope");
    assert_eq!(json["error"]["code"], "session_not_found");
}

#[tokio::test]
async fn a_traversal_container_id_is_refused() {
    let state = uhp_state();
    let id = id_learned_for(&state, OUTSIDE_NAME).await;
    // A held session with a folder on disk, so `{held}/..` resolves.
    let held = hold_session(&state);
    write_artifact(&held, "report.txt", b"own-artifact");
    let outside = FIXTURE.outside.to_string_lossy().into_owned();

    let probes = [
        // The parent of the workdir base, raw and percent-encoded.
        "..".to_owned(),
        "%2E%2E".to_owned(),
        // A held session's id as a prefix, then back out twice.
        format!("{held}%2F..%2F.."),
        encode_segment(&format!("{held}/../..")),
        // An absolute path, which `Path::join` takes in place of the base.
        encode_segment(&outside),
        // The base itself, and an id with a separator inside it.
        ".".to_owned(),
        encode_segment(&format!("{held}/x")),
    ];

    for probe in &probes {
        let (status, body, _) = send(
            app(state.clone()),
            &format!("/v1/containers/{probe}/files/{id}/content"),
        )
        .await;
        assert_refused(status, &body, probe);
    }
}

#[tokio::test]
async fn the_mounted_router_refuses_a_traversal_container_id() {
    let id = id_learned_for(&uhp_state(), OUTSIDE_NAME).await;
    let app = router::build(AppState::new(Arc::new(ServerState::new(
        CliRunnerType::Copilot,
    ))));

    for probe in [
        "..".to_owned(),
        encode_segment(&FIXTURE.outside.to_string_lossy()),
    ] {
        let (status, body, _) = send(
            app.clone(),
            &format!("/uhp/v1/containers/{probe}/files/{id}/content"),
        )
        .await;
        assert_refused(status, &body, &probe);
    }
}

#[cfg(unix)]
mod symlinks {
    use std::os::unix::fs::symlink;

    use super::*;

    #[tokio::test]
    async fn a_symlink_out_of_the_session_is_neither_listed_nor_served() {
        let state = uhp_state();
        let session = hold_session(&state);
        let report = write_artifact(&session, "report.txt", b"own-artifact");
        let folder = FIXTURE.base.join(&session);

        // A harness can write links as easily as files.
        symlink(&FIXTURE.outside, folder.join("escape")).expect("dir link");
        symlink(FIXTURE.outside.join(OUTSIDE_NAME), folder.join("leak.txt")).expect("file link");
        // A link that stays inside the session is still its artifact.
        symlink(&report, folder.join("alias.txt")).expect("inner link");

        let listed = listing(&state, &session).await;
        let names: Vec<&str> = listed.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["alias.txt", "report.txt"],
            "a link resolving outside the session folder is not its artifact"
        );

        let alias = &listed[0].1;
        let (status, body, _) = send(
            app(state.clone()),
            &format!("/v1/containers/{session}/files/{alias}/content"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"own-artifact");

        for relative in [format!("escape/{OUTSIDE_NAME}"), "leak.txt".to_owned()] {
            let id = id_learned_for(&state, &relative).await;
            let (status, body, _) = send(
                app(state.clone()),
                &format!("/v1/containers/{session}/files/{id}/content"),
            )
            .await;
            assert_refused(status, &body, &relative);
        }
    }

    #[tokio::test]
    async fn a_session_folder_that_is_itself_a_symlink_is_not_followed() {
        let state = uhp_state();
        let session = hold_session(&state);
        symlink(&FIXTURE.outside, FIXTURE.base.join(&session)).expect("folder link");

        assert!(
            listing(&state, &session).await.is_empty(),
            "a session folder pointing elsewhere holds none of this session's artifacts"
        );

        let id = id_learned_for(&state, OUTSIDE_NAME).await;
        let (status, body, _) = send(
            app(state),
            &format!("/v1/containers/{session}/files/{id}/content"),
        )
        .await;
        assert_refused(status, &body, &session);
    }
}
