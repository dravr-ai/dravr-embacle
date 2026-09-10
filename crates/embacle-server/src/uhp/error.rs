// ABOUTME: The UHP structured error envelope, its type/code taxonomy, and HTTP status mapping
// ABOUTME: Every UHP route answers failures with this shape; messages never carry internals
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Errors, in the shape the protocol requires.
//!
//! UHP does not accept a bare string or a foreign envelope: a failure is
//! `{"error": {"type", "code", "message"}}`, where `type` is one of six
//! categories and `code` is machine-readable. Clients branch on `code`, so an
//! unknown harness must say `harness_not_found` and nothing else.
//!
//! `message` is written for a human and MUST NOT leak internals — the
//! conformance suite greps it for `Traceback`, `File "/` and stack-frame
//! markers, and a server that formats a `Debug` error into it fails.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::Value;

/// The six error categories the specification enumerates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    /// The request itself was malformed or asked for something impossible.
    InvalidRequestError,
    /// No credential, or one the server does not accept.
    AuthenticationError,
    /// Authenticated, but not allowed to do this.
    PermissionError,
    /// Too many requests.
    RateLimitError,
    /// The harness ran and failed.
    HarnessError,
    /// Anything the server got wrong.
    ServerError,
}

impl ErrorType {
    /// The status this category answers with when nothing more specific applies.
    const fn status(self) -> StatusCode {
        match self {
            Self::InvalidRequestError => StatusCode::BAD_REQUEST,
            Self::AuthenticationError => StatusCode::UNAUTHORIZED,
            Self::PermissionError => StatusCode::FORBIDDEN,
            Self::RateLimitError => StatusCode::TOO_MANY_REQUESTS,
            Self::HarnessError | Self::ServerError => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// The body of a UHP failure.
#[derive(Debug, Clone, Serialize)]
pub struct UhpError {
    /// Broad category.
    #[serde(rename = "type")]
    pub kind: ErrorType,
    /// Machine-readable specific condition, e.g. `harness_not_found`.
    pub code: String,
    /// Human-readable, and free of stack traces or internal paths.
    pub message: String,
    /// Structured extra information, e.g. the versions a server does support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

/// The wire envelope: an error is always nested under `error`.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    /// The failure itself.
    pub error: UhpError,
}

/// A UHP failure together with the status it answers with.
#[derive(Debug, Clone)]
pub struct UhpFailure {
    status: StatusCode,
    error: UhpError,
}

impl UhpFailure {
    /// Build a failure, taking the status from the category.
    pub fn new(kind: ErrorType, code: &str, message: impl Into<String>) -> Self {
        Self {
            status: kind.status(),
            error: UhpError {
                kind,
                code: code.to_owned(),
                message: message.into(),
                detail: None,
            },
        }
    }

    /// Override the status where the specification names a different one.
    #[must_use]
    pub const fn with_status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }

    /// Attach structured detail, such as the list of supported versions.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        self.error.detail = Some(detail);
        self
    }

    /// No credential presented, or one the server does not accept.
    pub fn unauthorized(code: &str, message: impl Into<String>) -> Self {
        Self::new(ErrorType::AuthenticationError, code, message)
    }

    /// A harness id that names nothing this server runs.
    ///
    /// 404 rather than the category default, because the specification maps
    /// every `*_not_found` code to a not-found status.
    pub fn not_found(code: &str, message: impl Into<String>) -> Self {
        Self::new(ErrorType::InvalidRequestError, code, message).with_status(StatusCode::NOT_FOUND)
    }
}

impl UhpFailure {
    /// The error body, for embedding in a `failed` response object.
    #[must_use]
    pub fn into_error(self) -> UhpError {
        self.error
    }
}

impl IntoResponse for UhpFailure {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorEnvelope { error: self.error })).into_response()
    }
}
