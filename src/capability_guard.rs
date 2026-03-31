// ABOUTME: Pre-dispatch capability validation for request parameters vs provider support
// ABOUTME: Rejects unsupported params in strict mode, returns warnings in permissive mode
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Capability Guard
//!
//! Validates that a [`ChatRequest`](crate::types::ChatRequest) only uses parameters the target provider
//! actually supports, based on its [`LlmCapabilities`](crate::types::LlmCapabilities) flags.
//!
//! In **strict** mode, unsupported parameters cause an immediate
//! [`RunnerError`](crate::types::RunnerError) rejection. In **permissive** mode (the default),
//! the function returns warning strings that callers can surface in
//! [`ChatResponse::warnings`](crate::types::ChatResponse::warnings).

use crate::types::{ChatRequest, LlmCapabilities, RunnerError};

/// Validate that a request's parameters are supported by the provider.
///
/// Returns a (possibly empty) list of warning strings when unsupported
/// parameters are present in permissive mode. In strict mode, returns
/// an `Err` on the first unsupported parameter.
///
/// # Errors
///
/// Returns [`RunnerError`] in strict mode when the request uses
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
        let msg = format!(
            "{provider_name} does not support streaming; response will be delivered as a single SSE event"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.tools.is_some() && !capabilities.supports_function_calling() {
        let msg = format!(
            "{provider_name} does not support native function calling; tools will use text simulation"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if matches!(
        request.tool_choice,
        Some(crate::types::ToolChoice::Required)
    ) && !capabilities.supports_function_calling()
    {
        let msg = format!(
            "{provider_name} does not support function calling; tool_choice=required is not available"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.top_p.is_some() && !capabilities.supports_top_p() {
        let msg =
            format!("{provider_name} does not support top_p; requested value will be ignored");
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.stop.is_some() && !capabilities.supports_stop_sequences() {
        let msg = format!(
            "{provider_name} does not support stop sequences; requested value will be ignored"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.response_format.is_some() && !capabilities.supports_response_format() {
        let msg = format!(
            "{provider_name} does not support response_format; requested value will be ignored"
        );
        if strict {
            return Err(RunnerError::config(msg));
        }
        warnings.push(msg);
    }

    if request.has_images() && !capabilities.supports_vision() {
        let msg = format!(
            "{provider_name} does not support vision/images; image content will be ignored"
        );
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
        let warnings = validate_capabilities("test", caps, &request_with_all(), true).unwrap(); // Safe: test assertion
        assert!(warnings.is_empty());
    }

    #[test]
    fn permissive_returns_warnings_for_temperature() {
        let caps = LlmCapabilities::STREAMING;
        let warnings =
            validate_capabilities("copilot", caps, &request_with_temperature(), false).unwrap(); // Safe: test assertion
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("copilot"));
        assert!(warnings[0].contains("temperature"));
    }

    #[test]
    fn permissive_returns_warnings_for_max_tokens() {
        let caps = LlmCapabilities::STREAMING;
        let warnings =
            validate_capabilities("copilot", caps, &request_with_max_tokens(), false).unwrap(); // Safe: test assertion
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("max_tokens"));
    }

    #[test]
    fn permissive_returns_empty_when_all_supported() {
        let caps =
            LlmCapabilities::STREAMING | LlmCapabilities::TEMPERATURE | LlmCapabilities::MAX_TOKENS;
        let warnings = validate_capabilities("test", caps, &request_with_all(), false).unwrap(); // Safe: test assertion
        assert!(warnings.is_empty());
    }

    #[test]
    fn permissive_returns_multiple_warnings() {
        let caps = LlmCapabilities::empty();
        let warnings = validate_capabilities("test", caps, &request_with_all(), false).unwrap(); // Safe: test assertion
        assert_eq!(warnings.len(), 3);
        assert!(warnings[0].contains("temperature"));
        assert!(warnings[1].contains("max_tokens"));
        assert!(warnings[2].contains("streaming"));
    }

    #[test]
    fn strict_rejects_tools_without_function_calling() {
        let caps = LlmCapabilities::STREAMING;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(vec![
            crate::types::ToolDefinition {
                name: "test".to_owned(),
                description: "test".to_owned(),
                parameters: None,
            },
        ]);
        let err = validate_capabilities("test", caps, &request, true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("function calling"));
    }

    #[test]
    fn strict_rejects_unsupported_top_p() {
        let caps = LlmCapabilities::STREAMING;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_top_p(0.9);
        let err = validate_capabilities("test", caps, &request, true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("top_p"));
    }

    #[test]
    fn strict_rejects_unsupported_stop() {
        let caps = LlmCapabilities::STREAMING;
        let request =
            ChatRequest::new(vec![ChatMessage::user("hello")]).with_stop(vec!["END".to_owned()]);
        let err = validate_capabilities("test", caps, &request, true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("stop sequences"));
    }

    #[test]
    fn strict_rejects_unsupported_response_format() {
        let caps = LlmCapabilities::STREAMING;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")])
            .with_response_format(crate::types::ResponseFormat::JsonObject);
        let err = validate_capabilities("test", caps, &request, true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("response_format"));
    }

    #[test]
    fn permissive_warns_for_tools_without_function_calling() {
        let caps = LlmCapabilities::STREAMING;
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(vec![
            crate::types::ToolDefinition {
                name: "test".to_owned(),
                description: "test".to_owned(),
                parameters: None,
            },
        ]);
        let warnings = validate_capabilities("test", caps, &request, false).unwrap(); // Safe: test assertion
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("function calling"));
    }

    #[test]
    fn strict_rejects_images_without_vision() {
        let caps = LlmCapabilities::STREAMING;
        let img = crate::types::ImagePart::new("data", "image/png").unwrap(); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user_with_images("describe", vec![img])]);
        let err = validate_capabilities("test", caps, &request, true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Config);
        assert!(err.message.contains("vision"));
    }

    #[test]
    fn permissive_warns_for_images_without_vision() {
        let caps = LlmCapabilities::STREAMING;
        let img = crate::types::ImagePart::new("data", "image/png").unwrap(); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user_with_images("describe", vec![img])]);
        let warnings = validate_capabilities("test", caps, &request, false).unwrap(); // Safe: test assertion
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("vision"));
    }

    #[test]
    fn allows_images_with_vision_capability() {
        let caps = LlmCapabilities::STREAMING | LlmCapabilities::VISION;
        let img = crate::types::ImagePart::new("data", "image/png").unwrap(); // Safe: test assertion
        let request = ChatRequest::new(vec![ChatMessage::user_with_images("describe", vec![img])]);
        let warnings = validate_capabilities("test", caps, &request, true).unwrap(); // Safe: test assertion
        assert!(warnings.is_empty());
    }
}
