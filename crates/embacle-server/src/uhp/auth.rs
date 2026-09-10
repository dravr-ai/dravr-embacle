// ABOUTME: Bearer auth for UHP routes, answering refusals in the UHP error envelope
// ABOUTME: Separate from the OpenAI-surface middleware because the error shape differs
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Authentication for the UHP surface.
//!
//! The same `EMBACLE_API_KEY` as the rest of the server, but a refusal must be
//! the UHP envelope with `error.type` of `authentication_error` — the suite
//! reads that field, so reusing the OpenAI-surface middleware would fail A-02
//! while looking correct.
//!
//! Discovery is deliberately **not** behind this: a client has to be able to
//! learn whether it is talking to a UHP server before deciding what credential
//! to present.

use std::env;

use axum::extract::Request;
use axum::http::header::AUTHORIZATION;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

use super::error::UhpFailure;

/// The variable holding the key, shared with the rest of the server.
const API_KEY_ENV: &str = "EMBACLE_API_KEY";

/// Require a bearer token matching `EMBACLE_API_KEY`, when one is configured.
///
/// With no key set the server is in local development mode and every request
/// passes, matching the behaviour of the `OpenAI` surface beside it.
pub async fn require_auth(request: Request, next: Next) -> Response {
    let expected = match env::var(API_KEY_ENV) {
        Ok(key) if !key.is_empty() => key,
        _ => return next.run(request).await,
    };

    let presented = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default();

    let matches = expected.len() == presented.len()
        && bool::from(expected.as_bytes().ct_eq(presented.as_bytes()));

    if matches {
        next.run(request).await
    } else if presented.is_empty() {
        UhpFailure::unauthorized(
            "missing_credential",
            "this endpoint requires a bearer token",
        )
        .into_response()
    } else {
        UhpFailure::unauthorized("invalid_credential", "the bearer token was not accepted")
            .into_response()
    }
}
