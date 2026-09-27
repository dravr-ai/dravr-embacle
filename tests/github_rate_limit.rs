// ABOUTME: The GitHub rate-limit checker's absolute counts: parsed remaining/limit/reset, and the live read path
// ABOUTME: Measured on the checker's own token, and consistent with the percentage window
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::unwrap_used)]
#![cfg(feature = "quota-http")]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use embacle::quota::LimitChecker;
use embacle::quota_http::{
    parse_github_rate_limit, parse_github_rate_limit_counts, GithubHeadroomChecker, GithubRateLimit,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// A `GET /rate_limit` body as api.github.com returns it (trimmed to the
/// resources a PAT sees most often, plus the deprecated top-level `rate`).
fn realistic_body() -> Value {
    json!({
        "resources": {
            "core": {"limit": 5000, "used": 4812, "remaining": 188, "reset": 1_790_000_000_u64, "resource": "core"},
            "search": {"limit": 30, "used": 0, "remaining": 30, "reset": 1_789_996_460_u64, "resource": "search"},
            "graphql": {"limit": 5000, "used": 7, "remaining": 4993, "reset": 1_789_998_000_u64, "resource": "graphql"}
        },
        "rate": {"limit": 5000, "used": 4812, "remaining": 188, "reset": 1_790_000_000_u64, "resource": "core"}
    })
}

#[test]
fn counts_are_read_from_the_core_pool() {
    let rl = parse_github_rate_limit_counts(&realistic_body()).unwrap();
    assert_eq!(
        rl,
        GithubRateLimit {
            remaining: 188,
            limit: 5000,
            used: 4812,
            resets_at: UNIX_EPOCH + Duration::from_secs(1_790_000_000),
        }
    );
}

#[test]
fn counts_and_percentage_describe_the_same_reading() {
    let body = realistic_body();
    let rl = parse_github_rate_limit_counts(&body).unwrap();
    let snaps = parse_github_rate_limit(&body, SystemTime::now()).unwrap();
    assert_eq!(snaps.len(), 1);
    assert!((snaps[0].percent - rl.percent_used()).abs() < f32::EPSILON);
    assert!((rl.percent_used() - 96.24).abs() < 0.01);
    assert_eq!(snaps[0].resets_at, rl.resets_at);
}

#[test]
fn a_missing_remaining_is_derived_never_infinite() {
    let body = json!({"resources":{"core":{"limit":5000,"used":4900,"reset":1_u64}}});
    let rl = parse_github_rate_limit_counts(&body).unwrap();
    assert_eq!(rl.remaining, 100);

    let empty = json!({"resources":{"core":{"reset":1_u64}}});
    let rl = parse_github_rate_limit_counts(&empty).unwrap();
    assert_eq!((rl.remaining, rl.limit), (0, 0));
    assert!((rl.percent_used() - 100.0).abs() < f32::EPSILON);
}

#[test]
fn a_shape_change_is_an_error() {
    assert!(parse_github_rate_limit_counts(&json!({})).is_err());
    let no_reset = json!({"resources":{"core":{"limit":5000,"used":1,"remaining":4999}}});
    assert!(parse_github_rate_limit_counts(&no_reset).is_err());
}

/// Serve one canned HTTP response and hand back the request it answered.
async fn serve_once(body: String) -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = vec![0_u8; 8192];
        let n = sock.read(&mut buf).await.unwrap();
        let request = String::from_utf8_lossy(&buf[..n]).into_owned();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(response.as_bytes()).await.unwrap();
        request
    });
    (format!("http://{addr}/rate_limit"), handle)
}

#[tokio::test]
async fn rate_limit_reads_counts_on_the_token_it_holds() {
    let (url, served) = serve_once(realistic_body().to_string()).await;
    let checker = GithubHeadroomChecker::new("ghp_copilot_token").with_url(url);

    let rl = checker.rate_limit().await.unwrap();
    assert_eq!((rl.remaining, rl.limit, rl.used), (188, 5000, 4812));
    assert_eq!(
        rl.resets_at,
        UNIX_EPOCH + Duration::from_secs(1_790_000_000)
    );

    let request = served.await.unwrap().to_lowercase();
    assert!(
        request.contains("authorization: token ghp_copilot_token"),
        "the read must be made with the checker's token"
    );
}

#[tokio::test]
async fn check_reports_the_same_reading_as_a_window() {
    let (url, _served) = serve_once(realistic_body().to_string()).await;
    let checker = GithubHeadroomChecker::new("t").with_url(url);
    let snaps = checker.check().await.unwrap();
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].key, "github_core");
    assert!((snaps[0].percent - 96.24).abs() < 0.01);
}
