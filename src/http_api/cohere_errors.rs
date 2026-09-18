// ABOUTME: Classifies a Cohere API error response into a RunnerError a fallback chain can act on
// ABOUTME: The distinction that matters — a provider that could not answer vs a request that is wrong
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Cohere error classification.
//!
//! One decision lives here, and it is load-bearing: whether a 4xx means
//! *this provider could not answer* (a provider fault, so a chain moves on)
//! or *this request is wrong* (an invalid request, so retrying elsewhere
//! would fail the same way and hide the real diagnostic).
//! [`ErrorKind::is_provider_fault`](crate::types::ErrorKind::is_provider_fault)
//! reads the kind this produces, so getting it wrong either strands a
//! recoverable turn or silently reroutes a genuine bad request.

use reqwest::StatusCode;
use serde::Deserialize;
use tracing::debug;

use super::client;
use crate::types::RunnerError;

/// The name the Cohere provider reports
pub(super) const PROVIDER_NAME: &str = "cohere";

/// Cohere's error envelope.
#[derive(Debug, Deserialize)]
pub struct CohereErrorResponse {
    /// Free-form error message. Some 4xx responses use `data.error.message`
    /// instead; the raw body is the fallback when neither shape parses.
    #[serde(default)]
    pub message: Option<String>,
}

/// Phrases Cohere 400/422s with when the model produced no answer.
///
/// Both mean "the provider could not generate", not "your request is
/// malformed", so both are classified as the provider's fault and cascade to
/// the next tier. The second phrasing was observed ending real turns in a
/// canned outage reply because a single-phrase match had sent it down the
/// invalid-request branch.
const EMPTY_COMPLETION_MARKERS: [&str; 2] =
    ["no tool calls or response", "no valid response generated"];

/// Parse an error response from the Cohere API into a [`RunnerError`].
///
/// Every status goes through the shared [`client::map_http_error`] except
/// the one Cohere-specific case: a 400/422 whose message says the model
/// produced nothing is an `ExternalService` error, not an `InvalidRequest`.
pub fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
    let parsed_message = serde_json::from_str::<CohereErrorResponse>(body)
        .ok()
        .and_then(|err| err.message)
        .filter(|m| !m.is_empty());

    let error_message = parsed_message.unwrap_or_else(|| {
        debug!(
            status = status.as_u16(),
            body_preview = %body.chars().take(200).collect::<String>(),
            "Cohere API returned non-JSON error response"
        );
        format!("HTTP {status}")
    });

    if matches!(status.as_u16(), 400 | 422) {
        // An empty-completion 400/422 is a "provider couldn't answer": the
        // provider's fault, so a fallback chain cascades to the next tier
        // instead of surfacing a failure. A genuine validation error stays an
        // invalid request.
        let lower = error_message.to_lowercase();
        if EMPTY_COMPLETION_MARKERS.iter().any(|m| lower.contains(m)) {
            return RunnerError::external_service(PROVIDER_NAME, error_message);
        }
    }
    client::map_http_error(PROVIDER_NAME, status, &error_message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn empty_completion_422_is_the_providers_fault() {
        // Cohere 422 "No tool calls or response was generated" is a
        // provider-side empty completion, not a malformed request — it must
        // classify as a provider fault so a fallback chain cascades.
        let body = r#"{"message":"No tool calls or response was generated. Try updating messages or tool definitions"}"#;
        let err = parse_error_response(StatusCode::UNPROCESSABLE_ENTITY, body);
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(
            err.kind.is_provider_fault(),
            "empty-completion 422 must cascade; got kind {:?}",
            err.kind
        );
    }

    #[test]
    fn the_other_empty_completion_phrasing_also_cascades() {
        // Cohere states the same inability two ways; the single-phrase match
        // once sent this one down the invalid-request branch.
        let body = r#"{"message":"No valid response generated. Try updating messages"}"#;
        let err = parse_error_response(StatusCode::BAD_REQUEST, body);
        assert!(
            err.kind.is_provider_fault(),
            "this phrasing must cascade too; got kind {:?}",
            err.kind
        );
    }

    #[test]
    fn genuine_validation_422_is_an_invalid_request() {
        // A real malformed-request 422 would fail on every provider, so it
        // must NOT cascade.
        let body = r#"{"message":"invalid request: messages must not be empty"}"#;
        let err = parse_error_response(StatusCode::UNPROCESSABLE_ENTITY, body);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(
            !err.kind.is_provider_fault(),
            "a genuine validation 422 must not cascade; got kind {:?}",
            err.kind
        );
    }

    #[test]
    fn maps_401_to_auth_failure() {
        let body = r#"{"message":"invalid api token"}"#;
        let err = parse_error_response(StatusCode::UNAUTHORIZED, body);
        assert_eq!(err.kind, ErrorKind::AuthFailure);
        assert!(err.message.contains("invalid api token"));
    }

    #[test]
    fn maps_429_to_rate_limit() {
        let body = r#"{"message":"too many requests"}"#;
        let err = parse_error_response(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
    }

    #[test]
    fn falls_back_to_the_status_on_an_unknown_body_shape() {
        let body = "<html>nginx 502</html>";
        let err = parse_error_response(StatusCode::BAD_GATEWAY, body);
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("502"));
    }
}
