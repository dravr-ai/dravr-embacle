// ABOUTME: Session artifact listing and container file download over the UHP contract
// ABOUTME: Downloads are raw bytes with nosniff, because an artifact is attacker-influenced
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Files a session produced.
//!
//! An artifact is content a harness wrote, which means it is influenced by
//! whatever the model was asked to do. Serving it inline would let a crafted
//! artifact run as a page on this origin, so a download is raw bytes with
//! `X-Content-Type-Options: nosniff` and a `Content-Disposition` of
//! `attachment` — the browser is told, twice and unambiguously, not to
//! interpret it.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path as FsPath, PathBuf};

use axum::extract::{Path, State};
use axum::http::header::{CONTENT_DISPOSITION, CONTENT_TYPE, X_CONTENT_TYPE_OPTIONS};
use axum::http::{HeaderMap, HeaderValue};
use axum::response::IntoResponse;
use axum::Json;
use serde::Serialize;

use super::error::UhpFailure;

/// Environment variable naming where session working folders live.
pub const WORKDIR_ENV: &str = "UHP_WORKDIR";
use super::state::UhpState;

/// One artifact a session produced.
#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    /// `file_`-prefixed id.
    pub id: String,
    /// Always `file`.
    pub object: &'static str,
    /// The container it lives in, which is how it is addressed for download.
    pub container_id: String,
    /// Its name as the harness wrote it.
    pub filename: String,
    /// Size in bytes.
    pub bytes: u64,
}

/// The artifacts of one session.
#[derive(Debug, Clone, Serialize)]
pub struct ArtifactList {
    /// The session these belong to.
    pub session_id: String,
    /// Its artifacts.
    pub files: Vec<Artifact>,
}

/// Where a session's tasks run, and therefore where its artifacts land.
///
/// One folder per session, because that is the unit the specification deletes:
/// §6 says deleting a session removes "the working folder the session's tasks
/// wrote".
#[must_use]
pub fn session_workdir(session_id: &str) -> PathBuf {
    let base =
        env::var(WORKDIR_ENV).map_or_else(|_| env::temp_dir().join("embacle-uhp"), PathBuf::from);
    base.join(session_id)
}

/// A stable id for an artifact, derived from its path inside the session.
///
/// Derived rather than random so the same file keeps the same id across two
/// listings — a client that fetched an id must be able to use it.
fn artifact_id(relative: &str) -> String {
    let mut hasher = DefaultHasher::new();
    relative.hash(&mut hasher);
    format!("file_{:016x}", hasher.finish())
}

/// Every file a session's tasks wrote, walked from its working folder.
fn walk(root: &FsPath, session_id: &str) -> Vec<Artifact> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            // Forward slashes on the wire, whatever the host separator. The
            // filename is served over HTTP and is how the artifact is named to
            // a client, so a Windows server emitting "notes\\deep.md" would
            // hand back a path that differs from the same session's listing on
            // Linux for the same file.
            let relative = relative
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let bytes = entry.metadata().map_or(0, |m| m.len());
            found.push(Artifact {
                id: artifact_id(&relative),
                object: "file",
                container_id: session_id.to_owned(),
                filename: relative,
                bytes,
            });
        }
    }
    found.sort_by(|a, b| a.filename.cmp(&b.filename));
    found
}

/// Remove a session's working folder, artifacts and all.
pub fn remove_workdir(session_id: &str) {
    let dir = session_workdir(session_id);
    if dir.exists() {
        let _ = fs::remove_dir_all(dir);
    }
}

/// Handle `GET /v1/sessions/{session_id}/files`.
///
/// # Errors
///
/// Answers `404 session_not_found` when this server holds no such session.
pub async fn session_files(
    State(state): State<UhpState>,
    Path(session_id): Path<String>,
) -> Result<Json<ArtifactList>, UhpFailure> {
    if state.store.session_turns(&session_id).is_none() {
        return Err(UhpFailure::not_found(
            "session_not_found",
            "no session with that id exists",
        ));
    }
    let root = session_workdir(&session_id);
    let files = if root.is_dir() {
        walk(&root, &session_id)
    } else {
        Vec::new()
    };
    Ok(Json(ArtifactList { session_id, files }))
}

/// Handle `GET /v1/containers/{container_id}/files/{file_id}/content`.
///
/// # Errors
///
/// Answers `404 file_not_found` when the container holds no artifact with that
/// id — including when the session was deleted, since its folder went with it.
pub async fn container_file(
    Path((container_id, file_id)): Path<(String, String)>,
) -> Result<impl IntoResponse, UhpFailure> {
    let root = session_workdir(&container_id);
    let artifact = walk(&root, &container_id)
        .into_iter()
        .find(|a| a.id == file_id)
        .ok_or_else(|| {
            UhpFailure::not_found("file_not_found", "no artifact with that id exists")
        })?;

    // The stored name is wire-shaped, so rebuild the host path from its
    // segments rather than joining a string that carries forward slashes.
    let mut path = root;
    for segment in artifact.filename.split('/') {
        path.push(segment);
    }
    let bytes = fs::read(&path)
        .map_err(|_| UhpFailure::not_found("file_not_found", "no artifact with that id exists"))?;

    Ok((download_headers(&artifact.filename), bytes))
}

/// The headers every artifact download carries.
///
/// Kept as a named function so the download path and its test assert the same
/// thing: an artifact is attacker-influenced content, and must never be
/// sniffed into an executable type or rendered inline on this origin.
#[must_use]
pub fn download_headers(filename: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    if let Ok(value) = format!("attachment; filename=\"{filename}\"").parse() {
        headers.insert(CONTENT_DISPOSITION, value);
    }
    headers
}

#[cfg(test)]
mod tests {
    use std::process;

    use super::*;

    #[test]
    fn a_walk_finds_nested_files_and_ids_them_stably() {
        let root = env::temp_dir().join(format!("uhp-walk-{}", process::id()));
        let nested = root.join("notes");
        fs::create_dir_all(&nested).expect("create"); // Safe: test assertion
        fs::write(root.join("out.txt"), b"artifact-ok").expect("write"); // Safe: test assertion
        fs::write(nested.join("deep.md"), b"nested").expect("write"); // Safe: test assertion

        let found = walk(&root, "sess_x");
        let names: Vec<&str> = found.iter().map(|a| a.filename.as_str()).collect();
        assert_eq!(
            names,
            vec!["notes/deep.md", "out.txt"],
            "a nested file is an artifact too, and its path is forward-slashed on every host \
             because the filename goes out over HTTP"
        );
        assert_eq!(found[1].bytes, 11, "size is the file's, not a guess");
        assert!(found.iter().all(|a| a.container_id == "sess_x"));

        // Ids must be stable across listings, or an id a client fetched stops
        // resolving on the next call.
        let again = walk(&root, "sess_x");
        assert_eq!(
            found.iter().map(|a| a.id.clone()).collect::<Vec<_>>(),
            again.iter().map(|a| a.id.clone()).collect::<Vec<_>>()
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_removed_workdir_takes_its_artifacts_with_it() {
        let session = format!("sess_rm{}", process::id());
        let root = session_workdir(&session);
        fs::create_dir_all(&root).expect("create"); // Safe: test assertion
        fs::write(root.join("a.txt"), b"x").expect("write"); // Safe: test assertion
        assert_eq!(walk(&root, &session).len(), 1);

        remove_workdir(&session);
        assert!(
            !root.exists(),
            "deleting a session must remove the folder its tasks wrote"
        );
    }

    #[test]
    fn an_artifact_download_is_never_sniffable_or_inline() {
        // X-07 reads x-content-type-options directly. The other two headers are
        // the same defence: an artifact is content a model was steered into
        // writing, so it must not render as a page on this origin.
        let h = download_headers("report.html");

        assert_eq!(
            h.get(X_CONTENT_TYPE_OPTIONS).and_then(|v| v.to_str().ok()),
            Some("nosniff")
        );
        assert_eq!(
            h.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            Some("application/octet-stream"),
            "never the artifact's own claimed type"
        );
        assert!(
            h.get(CONTENT_DISPOSITION)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.starts_with("attachment")),
            "inline would let a crafted artifact run on this origin"
        );
    }
}
