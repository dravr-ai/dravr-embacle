// ABOUTME: UHP protocol version constants, negotiation, and the UHP-Version response header
// ABOUTME: An unsupported requested version is refused with 400, never silently substituted
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Version negotiation.
//!
//! Two rules, both checked by the suite. Every response carries `UHP-Version`
//! naming the contract it was served under, so a client never has to guess. And
//! a request naming a version this server does not speak is **refused with
//! 400**, not served under a different one — silently substituting hands the
//! client a body it may not be able to parse, which is worse than a clean
//! refusal.

use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use super::error::UhpFailure;

/// The only protocol version this server implements.
pub const VERSION: &str = "2026-08-11";

/// Every version this server can serve.
pub const SUPPORTED: &[&str] = &[VERSION];

/// The negotiation header, in both directions.
pub const HEADER: &str = "uhp-version";

/// Refuse a request for a version we do not speak; stamp every answer with the
/// version it was served under.
///
/// # Errors
///
/// Answers `400 unsupported_protocol_version` when the request names a version
/// outside [`SUPPORTED`], with the supported list in `error.detail.supported`
/// so the client can retry without a second round trip.
pub async fn negotiate(request: Request, next: Next) -> Response {
    if let Some(asked) = request.headers().get(HEADER) {
        let asked = asked.to_str().unwrap_or_default();
        if !asked.is_empty() && !SUPPORTED.contains(&asked) {
            return UhpFailure::new(
                super::error::ErrorType::InvalidRequestError,
                "unsupported_protocol_version",
                format!("this server does not speak protocol version '{asked}'"),
            )
            .with_detail(json!({ "supported": SUPPORTED }))
            .into_response();
        }
    }

    let mut response = next.run(request).await;
    response.headers_mut().insert(
        HeaderName::from_static(HEADER),
        HeaderValue::from_static(VERSION),
    );
    response
}
