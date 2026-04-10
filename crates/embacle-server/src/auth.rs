// ABOUTME: Optional bearer token authentication middleware for the REST API
// ABOUTME: Enforces EMBACLE_API_KEY when set, allows unauthenticated access otherwise
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_env_is_correct() {
        assert_eq!(API_KEY_ENV, "EMBACLE_API_KEY");
    }
}
