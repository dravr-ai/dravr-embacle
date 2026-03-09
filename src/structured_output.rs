// ABOUTME: Standalone function forcing any LlmProvider to return schema-valid JSON
// ABOUTME: Includes lightweight JSON schema validation, markdown fence extraction, and retry loop
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Structured Output Enforcement
//!
//! Forces any [`LlmProvider`](crate::types::LlmProvider) to return JSON that validates against a provided
//! JSON Schema. The module injects schema instructions into the system message,
//! extracts JSON from the response (including markdown fences), validates against
//! the schema, and retries with validation feedback on failure.
//!
//! ## Schema Validation Coverage
//!
//! The built-in validator checks `type`, `required`, recursive `properties`,
//! array `items`, `enum` values, numeric `minimum`/`maximum`, and
//! `additionalProperties: false`. It does not cover the full JSON Schema
//! specification (e.g., `oneOf`, `anyOf`, `$ref`, `pattern`).

use serde_json::Value;
use tracing::{info, warn};

use crate::types::{ChatMessage, ChatRequest, LlmProvider, RunnerError};

/// Request configuration for structured JSON output
#[derive(Debug, Clone)]
pub struct StructuredOutputRequest {
    /// The original chat request
    pub request: ChatRequest,
    /// JSON Schema the response must conform to
    pub schema: Value,
    /// Maximum retry attempts on validation failure
    pub max_retries: u32,
}

/// A single schema validation error
#[derive(Debug, Clone)]
pub struct SchemaValidationError {
    /// Human-readable error description
    pub message: String,
    /// JSON path where the error occurred (e.g., "$.name")
    pub path: String,
}

impl std::fmt::Display for SchemaValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Request structured JSON output from any provider, with schema validation and retry.
///
/// # Flow
///
/// 1. Append schema instructions to the system message
/// 2. Call `provider.complete()` and extract JSON from the response
/// 3. Validate against the schema
/// 4. On failure, append errors as user feedback and retry up to `max_retries`
/// 5. After exhaustion, return [`RunnerError::external_service`]
///
/// # Errors
///
/// Returns [`RunnerError`] if the provider fails or validation is exhausted.
pub async fn request_structured_output(
    provider: &dyn LlmProvider,
    structured_request: &StructuredOutputRequest,
) -> Result<Value, RunnerError> {
    let schema_str = serde_json::to_string_pretty(&structured_request.schema)
        .map_err(|e| RunnerError::internal(format!("failed to serialize schema: {e}")))?;

    let schema_instruction = format!(
        "\n\nYou MUST respond with ONLY valid JSON that conforms to the following JSON Schema. \
         Do NOT include any explanatory text, markdown formatting, or anything other than the \
         JSON object.\n\nSchema:\n```json\n{schema_str}\n```"
    );

    let mut messages = structured_request.request.messages.clone();

    // Inject schema instruction into the system message
    inject_schema_instruction(&mut messages, &schema_instruction);

    let total_attempts = structured_request.max_retries + 1;
    for attempt in 0..total_attempts {
        let request = ChatRequest {
            messages: messages.clone(),
            model: structured_request.request.model.clone(),
            temperature: structured_request.request.temperature,
            max_tokens: structured_request.request.max_tokens,
            stream: false,
            tools: structured_request.request.tools.clone(),
            tool_choice: structured_request.request.tool_choice.clone(),
            top_p: structured_request.request.top_p,
            stop: structured_request.request.stop.clone(),
            response_format: structured_request.request.response_format.clone(),
        };

        let response = provider.complete(&request).await?;

        // Try to extract JSON from the response
        let json_str = extract_json_from_response(&response.content);

        let parsed: Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(parse_err) => {
                warn!(
                    attempt,
                    error = %parse_err,
                    "structured output: failed to parse JSON from response"
                );
                if attempt < structured_request.max_retries {
                    messages.push(ChatMessage::assistant(response.content.clone()));
                    messages.push(ChatMessage::user(format!(
                        "Your response was not valid JSON: {parse_err}. \
                         Please respond with ONLY a valid JSON object matching the schema."
                    )));
                }
                continue;
            }
        };

        let errors = validate_against_schema(&parsed, &structured_request.schema);

        if errors.is_empty() {
            info!(attempt, "structured output: validation passed");
            return Ok(parsed);
        }

        warn!(
            attempt,
            error_count = errors.len(),
            "structured output: schema validation failed"
        );

        if attempt < structured_request.max_retries {
            let error_feedback: Vec<String> = errors.iter().map(ToString::to_string).collect();
            messages.push(ChatMessage::assistant(response.content.clone()));
            messages.push(ChatMessage::user(format!(
                "Your JSON response had validation errors:\n- {}\n\
                 Please fix these and respond with ONLY a valid JSON object.",
                error_feedback.join("\n- ")
            )));
        }
    }

    Err(RunnerError::external_service(
        provider.name(),
        "structured output validation exhausted after all retries",
    ))
}

/// Inject schema instruction into the system message, or create one
fn inject_schema_instruction(messages: &mut Vec<ChatMessage>, instruction: &str) {
    if let Some(first) = messages.first_mut() {
        if first.role == crate::types::MessageRole::System {
            let augmented = format!("{}{instruction}", first.content);
            *first = ChatMessage::system(augmented);
            return;
        }
    }
    messages.insert(0, ChatMessage::system(instruction.to_owned()));
}

/// Extract JSON content from a response, handling markdown code fences.
///
/// Uses a brace-depth counter to find the outermost `{...}` block.
fn extract_json_from_response(content: &str) -> String {
    let trimmed = content.trim();

    // Fast path: already starts with `{`
    if trimmed.starts_with('{') {
        return extract_braced_json(trimmed);
    }

    // Try to find JSON inside markdown fences
    if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        // Skip optional language tag (e.g., "json")
        let content_start = after_fence.find('\n').map_or(0, |pos| pos + 1);
        let fence_content = &after_fence[content_start..];

        if let Some(end) = fence_content.find("```") {
            let inside = fence_content[..end].trim();
            if inside.starts_with('{') {
                return extract_braced_json(inside);
            }
        }
    }

    // Last resort: find the first `{` and extract from there
    if let Some(brace_pos) = trimmed.find('{') {
        return extract_braced_json(&trimmed[brace_pos..]);
    }

    trimmed.to_owned()
}

/// Extract a complete JSON object using brace-depth counting
fn extract_braced_json(text: &str) -> String {
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in text.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return text[..=i].to_owned();
                }
            }
            _ => {}
        }
    }

    text.to_owned()
}

/// Validate a JSON value against a schema.
///
/// Checks: `type`, `required`, recursive `properties`, array `items`,
/// `enum` values, numeric `minimum`/`maximum`, and `additionalProperties: false`.
pub fn validate_against_schema(value: &Value, schema: &Value) -> Vec<SchemaValidationError> {
    let mut errors = Vec::new();
    validate_value(value, schema, "$", &mut errors);
    errors
}

fn validate_value(
    value: &Value,
    schema: &Value,
    path: &str,
    errors: &mut Vec<SchemaValidationError>,
) {
    // Check type
    if let Some(expected_type) = schema.get("type").and_then(Value::as_str) {
        let actual_type = json_type_name(value);
        if actual_type != expected_type {
            errors.push(SchemaValidationError {
                message: format!("expected type \"{expected_type}\", got \"{actual_type}\""),
                path: path.to_owned(),
            });
            return;
        }
    }

    // Check enum constraint
    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
        if !enum_values.contains(value) {
            errors.push(SchemaValidationError {
                message: format!("value not in enum: expected one of {enum_values:?}, got {value}"),
                path: path.to_owned(),
            });
            return;
        }
    }

    // Numeric bounds (minimum, maximum)
    if let Some(num) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
            if num < min {
                errors.push(SchemaValidationError {
                    message: format!("value {num} is less than minimum {min}"),
                    path: path.to_owned(),
                });
            }
        }
        if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
            if num > max {
                errors.push(SchemaValidationError {
                    message: format!("value {num} exceeds maximum {max}"),
                    path: path.to_owned(),
                });
            }
        }
    }

    // For objects: check required fields, property types (recursive), additional properties
    if let Some(obj) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for req in required {
                if let Some(field_name) = req.as_str() {
                    if !obj.contains_key(field_name) {
                        errors.push(SchemaValidationError {
                            message: format!("missing required field \"{field_name}\""),
                            path: format!("{path}.{field_name}"),
                        });
                    }
                }
            }
        }

        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (prop_name, prop_schema) in properties {
                if let Some(prop_value) = obj.get(prop_name) {
                    let prop_path = format!("{path}.{prop_name}");
                    // Recurse into nested properties
                    validate_value(prop_value, prop_schema, &prop_path, errors);
                }
            }

            // Check additionalProperties: false
            if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                for key in obj.keys() {
                    if !properties.contains_key(key) {
                        errors.push(SchemaValidationError {
                            message: format!("unexpected additional property \"{key}\""),
                            path: format!("{path}.{key}"),
                        });
                    }
                }
            }
        }
    }

    // For arrays: validate items against the items schema
    if let Some(arr) = value.as_array() {
        if let Some(items_schema) = schema.get("items") {
            for (i, item) in arr.iter().enumerate() {
                let item_path = format!("{path}[{i}]");
                validate_value(item, items_schema, &item_path, errors);
            }
        }
    }
}

/// Map a JSON value to its JSON Schema type name
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
        RunnerError,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    struct TestProvider {
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
    }

    impl TestProvider {
        fn new(responses: Vec<Result<ChatResponse, RunnerError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            "test"
        }
        fn display_name(&self) -> &str {
            "Test Provider"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::text_only()
        }
        fn default_model(&self) -> &'static str {
            "test-model"
        }
        fn available_models(&self) -> &[String] {
            &[]
        }
        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock");
            if responses.is_empty() {
                Err(RunnerError::internal("no more test responses"))
            } else {
                responses.remove(0)
            }
        }
        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("not supported"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
        }
    }

    fn make_response(content: &str) -> ChatResponse {
        ChatResponse {
            content: content.to_owned(),
            model: "test-model".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        }
    }

    // --- validate_against_schema tests ---

    #[test]
    fn validate_valid_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let value = json!({"name": "Alice", "age": 30});
        let errors = validate_against_schema(&value, &schema);
        assert!(errors.is_empty());
    }

    #[test]
    fn validate_missing_required_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let value = json!({"name": "Alice"});
        let errors = validate_against_schema(&value, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("age"));
    }

    #[test]
    fn validate_wrong_types() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });

        let value = json!({"name": 42, "age": "not a number"});
        let errors = validate_against_schema(&value, &schema);
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn validate_wrong_root_type() {
        let schema = json!({"type": "object"});
        let value = json!("just a string");
        let errors = validate_against_schema(&value, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("expected type \"object\""));
    }

    // --- extract_json_from_response tests ---

    #[test]
    fn extract_raw_json() {
        let content = r#"{"name": "Alice", "age": 30}"#;
        let extracted = extract_json_from_response(content);
        let parsed: Value = serde_json::from_str(&extracted).expect("valid JSON");
        assert_eq!(parsed["name"], "Alice");
    }

    #[test]
    fn extract_json_from_markdown_fences() {
        let content = "Here is the result:\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```\nDone.";
        let extracted = extract_json_from_response(content);
        let parsed: Value = serde_json::from_str(&extracted).expect("valid JSON");
        assert_eq!(parsed["name"], "Bob");
    }

    #[test]
    fn extract_json_with_nested_braces() {
        let content = r#"{"outer": {"inner": "value"}, "list": [1, 2]}"#;
        let extracted = extract_json_from_response(content);
        let parsed: Value = serde_json::from_str(&extracted).expect("valid JSON");
        assert_eq!(parsed["outer"]["inner"], "value");
    }

    // --- full retry loop tests ---

    #[tokio::test]
    async fn full_retry_loop_eventual_success() {
        let provider = TestProvider::new(vec![
            Ok(make_response("not json at all")),
            Ok(make_response(r#"{"name": "Alice", "age": 30}"#)),
        ]);

        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let structured = StructuredOutputRequest {
            request: ChatRequest::new(vec![ChatMessage::user("give me data")]),
            schema,
            max_retries: 2,
        };

        let result = request_structured_output(&provider, &structured)
            .await
            .expect("should succeed on retry");
        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[tokio::test]
    async fn exhaustion_returns_error() {
        let provider = TestProvider::new(vec![
            Ok(make_response("garbage")),
            Ok(make_response("still garbage")),
            Ok(make_response("nope")),
        ]);

        let schema = json!({
            "type": "object",
            "required": ["name"]
        });

        let structured = StructuredOutputRequest {
            request: ChatRequest::new(vec![ChatMessage::user("give me data")]),
            schema,
            max_retries: 2,
        };

        let result = request_structured_output(&provider, &structured).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("exhausted"));
    }

    // --- enhanced validation tests ---

    #[test]
    fn validate_nested_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"}
                    },
                    "required": ["city"]
                }
            },
            "required": ["address"]
        });

        let valid = json!({"address": {"city": "Paris", "zip": "75001"}});
        assert!(validate_against_schema(&valid, &schema).is_empty());

        let missing_city = json!({"address": {"zip": "75001"}});
        let errors = validate_against_schema(&missing_city, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].path.contains("city"));

        let wrong_type = json!({"address": {"city": 42}});
        let errors = validate_against_schema(&wrong_type, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("expected type \"string\""));
    }

    #[test]
    fn validate_array_items() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"}
        });

        let valid = json!(["a", "b", "c"]);
        assert!(validate_against_schema(&valid, &schema).is_empty());

        let invalid = json!(["a", 42, "c"]);
        let errors = validate_against_schema(&invalid, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].path.contains("[1]"));
    }

    #[test]
    fn validate_enum_values() {
        let schema = json!({
            "type": "string",
            "enum": ["red", "green", "blue"]
        });

        let valid = json!("green");
        assert!(validate_against_schema(&valid, &schema).is_empty());

        let invalid = json!("yellow");
        let errors = validate_against_schema(&invalid, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("not in enum"));
    }

    #[test]
    fn validate_numeric_bounds() {
        let schema = json!({
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        });

        let valid = json!(50);
        assert!(validate_against_schema(&valid, &schema).is_empty());

        let too_low = json!(-1);
        let errors = validate_against_schema(&too_low, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("less than minimum"));

        let too_high = json!(101);
        let errors = validate_against_schema(&too_high, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("exceeds maximum"));
    }

    #[test]
    fn validate_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": false
        });

        let valid = json!({"name": "Alice"});
        assert!(validate_against_schema(&valid, &schema).is_empty());

        let with_extra = json!({"name": "Alice", "age": 30});
        let errors = validate_against_schema(&with_extra, &schema);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unexpected additional property"));
    }
}
