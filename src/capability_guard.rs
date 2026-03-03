// ABOUTME: Pre-dispatch capability validation for request parameters vs provider support
// ABOUTME: Rejects unsupported params in strict mode, returns warnings in permissive mode
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Capability Guard
//!
//! Validates that a [`ChatRequest`] only uses parameters the target provider
//! actually supports, based on its [`LlmCapabilities`] flags.
//!
//! In **strict** mode, unsupported parameters cause an immediate
//! [`RunnerError::config`] rejection. In **permissive** mode (the default),
//! the function returns warning strings that callers can surface in
//! [`ChatResponse::warnings`].

use crate::types::{ChatRequest, LlmCapabilities, RunnerError};

/// Validate that a request's parameters are supported by the provider.
///
/// Returns a (possibly empty) list of warning strings when unsupported
/// parameters are present in permissive mode. In strict mode, returns
/// an `Err` on the first unsupported parameter.
///
/// # Errors
///
/// Returns [`RunnerError::config`] in strict mode when the request uses
/// a parameter the provider does not support.
pub fn validate_capabilities(
    provider_name: &str,
    capabilities: LlmCapabilities,
    request: &ChatRequest,
    strict: bool,
) -> Result<Vec<String>, RunnerError> {
    let mut warnings: Vec<String> = Vec::new();

    if request.temperature.is_some() && !capabilities.supports_temperature() {
        let msg = format!(
            "{provider_name} does not support temperature; requested value will be ignored"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.max_tokens.is_some() && !capabilities.supports_max_tokens() {
        let msg =
            format!("{provider_name} does not support max_tokens; requested value will be ignored");
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.stream && !capabilities.supports_streaming() {
        let msg =
            format!("{provider_name} does not support streaming; requested value will be ignored");
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ErrorKind};

    fn request_with_temperature() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello")]).with_temperature(0.7)
    }

    fn request_with_max_tokens() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello")]).with_max_tokens(1024)
    }

    fn request_with_streaming() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello")]).with_streaming()
    }

    fn request_with_all() -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello")])
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_streaming()
    }

    #[test]
    fn strict_rejects_unsupported_temperature() {
        let caps = LlmCapabilities::STREAMING;
        let err =
            validate_capabilities("test", caps, &request_with_temperature(), true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("temperature"));
    }

    #[test]
    fn strict_rejects_unsupported_max_tokens() {
        let caps = LlmCapabilities::STREAMING;
        let err =
            validate_capabilities("test", caps, &request_with_max_tokens(), true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("max_tokens"));
    }

    #[test]
    fn strict_rejects_unsupported_streaming() {
        let caps = LlmCapabilities::empty();
        let err = validate_capabilities("test", caps, &request_with_streaming(), true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("streaming"));
    }

    #[test]
    fn strict_allows_supported_capabilities() {
        let caps =
            LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS;
        let warnings = validate_capabilities("test", caps, &request_with_all(), true).unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn permissive_returns_warnings_for_temperature() {
        let caps = LlmCapabilities::STREAMING;
        let warnings =
            validate_capabilities("copilot", caps, &request_with_temperature(), false).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("copilot"));
        assert!(warnings[0].contains("temperature"));
    }

    #[test]
    fn permissive_returns_warnings_for_max_tokens() {
        let caps = LlmCapabilities::STREAMING;
        let warnings =
            validate_capabilities("copilot", caps, &request_with_max_tokens(), false).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("max_tokens"));
    }

    #[test]
    fn permissive_returns_empty_when_all_supported() {
        let caps =
            LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS;
        let warnings = validate_capabilities("test", caps, &request_with_all(), false).unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn permissive_returns_multiple_warnings() {
        let caps = LlmCapabilities::empty();
        let warnings = validate_capabilities("test", caps, &request_with_all(), false).unwrap();
        assert_eq!(warnings.len(), 3);
        assert!(warnings[0].contains("temperature"));
        assert!(warnings[1].contains("max_tokens"));
        assert!(warnings[2].contains("streaming"));
    }
}
