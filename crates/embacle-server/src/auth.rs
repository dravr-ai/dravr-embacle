// ABOUTME: Optional bearer token authentication middleware for the REST API
// ABOUTME: Enforces EMBACLE_API_KEY when set; unauthenticated access is allowed only on a loopback bind
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::error::Error;
use std::fmt;
use std::net::IpAddr;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use subtle::ConstantTimeEq;

use crate::openai_types::ErrorResponse;

/// Environment variable name for the API key
const API_KEY_ENV: &str = "EMBACLE_API_KEY";

/// Middleware that validates the bearer token against `EMBACLE_API_KEY`
///
/// The env var is read on every request to allow runtime key rotation
/// without restarting the server. If the variable is not set, all requests
/// are allowed through (localhost development mode). If set, requests must
/// include a matching `Authorization: Bearer <key>` header.
pub async fn require_auth(request: Request, next: Next) -> Response {
    let expected_key = match env::var(API_KEY_ENV) {
        Ok(key) if !key.is_empty() => key,
        _ => return next.run(request).await,
    };

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header.as_bytes()["Bearer ".len()..];
            let expected = expected_key.as_bytes();
            if token.ct_eq(expected).into() {
                next.run(request).await
            } else {
                auth_error("Invalid API key")
            }
        }
        Some(_) => auth_error("Authorization header must use Bearer scheme"),
        None => auth_error("Missing Authorization header"),
    }
}

/// Build a 401 error response
fn auth_error(message: &str) -> Response {
    let body = ErrorResponse::new("authentication_error", message);
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

/// Whether `EMBACLE_API_KEY` is set to a non-empty value
pub fn api_key_configured() -> bool {
    matches!(env::var(API_KEY_ENV), Ok(key) if !key.is_empty())
}

/// Startup authentication posture resolved from the bind host and key presence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// `EMBACLE_API_KEY` is set — every request is authenticated
    Enforced,
    /// No key set, but the bind is loopback-only — unauthenticated dev access
    LoopbackDev,
}

/// Refusal to start unauthenticated on a non-loopback bind
///
/// embacle is the guardian gate, so it fails closed: exposing tool execution
/// on a reachable interface without an API key is never permitted.
#[derive(Debug, Clone)]
pub struct InsecureBindError {
    /// The non-loopback host the server was asked to bind
    pub host: String,
}

impl fmt::Display for InsecureBindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "refusing to start: no {API_KEY_ENV} set while binding non-loopback host '{}'. \
             Set {API_KEY_ENV} to require authentication, or bind 127.0.0.1 for local development.",
            self.host
        )
    }
}

impl Error for InsecureBindError {}

/// Whether `host` resolves to a loopback interface (127.0.0.0/8, ::1, localhost)
///
/// Anything that does not parse as a loopback address — including `0.0.0.0`,
/// `::`, and unknown names — is treated as non-loopback so the guardian fails
/// closed rather than open.
fn is_loopback_host(host: &str) -> bool {
    let trimmed = host.trim();
    if trimmed.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let stripped = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    stripped
        .parse::<IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Resolve the startup authentication posture for an HTTP bind.
///
/// - Key set → [`AuthMode::Enforced`] regardless of host.
/// - No key + loopback host → [`AuthMode::LoopbackDev`] (caller should warn).
/// - No key + non-loopback host → [`InsecureBindError`]; the server must not start.
pub fn resolve_startup_auth(host: &str, has_api_key: bool) -> Result<AuthMode, InsecureBindError> {
    if has_api_key {
        Ok(AuthMode::Enforced)
    } else if is_loopback_host(host) {
        Ok(AuthMode::LoopbackDev)
    } else {
        Err(InsecureBindError {
            host: host.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_env_is_correct() {
        assert_eq!(API_KEY_ENV, "EMBACLE_API_KEY");
    }

    #[test]
    fn loopback_hosts_are_recognized() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.1.2.3"));
        assert!(is_loopback_host("::1"));
        assert!(is_loopback_host("[::1]"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("LocalHost"));
    }

    #[test]
    fn non_loopback_hosts_are_rejected() {
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("::"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.com"));
        assert!(!is_loopback_host(""));
    }

    #[test]
    fn refuses_to_start_non_loopback_without_key() {
        // The core regression: guardian must fail closed, not open, when
        // exposed on a reachable interface with no API key configured.
        let err = resolve_startup_auth("0.0.0.0", false)
            .expect_err("non-loopback bind with no key must be refused");
        assert_eq!(err.host, "0.0.0.0");
        assert!(err.to_string().contains("refusing to start"));
        assert!(err.to_string().contains("EMBACLE_API_KEY"));
    }

    #[test]
    fn allows_loopback_dev_without_key() {
        assert_eq!(
            resolve_startup_auth("127.0.0.1", false).unwrap(),
            AuthMode::LoopbackDev
        );
    }

    #[test]
    fn enforces_when_key_present_on_any_host() {
        assert_eq!(
            resolve_startup_auth("0.0.0.0", true).unwrap(),
            AuthMode::Enforced
        );
        assert_eq!(
            resolve_startup_auth("127.0.0.1", true).unwrap(),
            AuthMode::Enforced
        );
    }
}
