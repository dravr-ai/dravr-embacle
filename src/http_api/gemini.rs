// ABOUTME: Google Gemini provider over the Generative Language API, with streaming and function calling
// ABOUTME: Hoists system messages to system_instruction and retries thinking-only answers; the key rides in the URL, never in a log
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Gemini Provider
//!
//! [`LlmProvider`] over Google's Gemini models via the Generative Language
//! API (`generateContent` / `streamGenerateContent`).
//!
//! ## Configuration
//!
//! [`GeminiConfig::from_env`] reads:
//!
//! - `GEMINI_API_KEY`: required, from Google AI Studio
//! - `GEMINI_DEFAULT_MODEL`: default `gemini-flash-lite-latest`
//! - `GEMINI_MAX_RETRIES`, `GEMINI_INITIAL_RETRY_DELAY_MS`,
//!   `GEMINI_MAX_RETRY_DELAY_MS`: retry tuning (defaults 3 / 500 / 5000)
//!
//! ## Vendor quirks kept here
//!
//! - The API key is a `?key=` query parameter on every request. No request
//!   URL is ever written to a log, and `Debug` redacts the key.
//! - System messages are hoisted into `system_instruction` (every one of
//!   them, concatenated); a `Tool` role message is sent as `user`.
//! - Thinking-class models answer with a candidate that has no content parts.
//!   Such an answer is an error the provider retries in place, with the
//!   same backoff as a 429 or a 503.
//! - A 429 carries Google's `Please retry in Ns` wait in its message.
//! - `cachedContentTokenCount` is reported as the cache-read count on
//!   [`TokenUsage`]; Gemini reports no cache writes.
//!
//! ## Example
//!
//! ```rust,no_run
//! use embacle::http_api::gemini::{GeminiConfig, GeminiProvider};
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//!
//! # async fn example() -> Result<(), embacle::types::RunnerError> {
//! let provider = GeminiProvider::new(GeminiConfig::from_env()?)?;
//! let request = ChatRequest::new(vec![ChatMessage::user("What is machine learning?")]);
//! let response = provider.complete(&request).await?;
//! println!("{}", response.content);
//! # Ok(())
//! # }
//! ```

use std::env;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, instrument, warn};

use super::client::{
    self, map_send_error, with_retries, AttemptError, HttpRetryConfig, DEFAULT_TIMEOUT_SECS,
};
use super::sse::create_sse_stream;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, ErrorKind, LlmCapabilities, LlmProvider,
    MessageRole, RunnerError, StreamChunk, TokenUsage, ToolCallRequest, ToolDefinition,
};

/// The name this provider reports, and the price-table key its usage bills under
const PROVIDER_NAME: &str = "gemini";

/// Environment variable for the API key
const GEMINI_API_KEY_ENV: &str = "GEMINI_API_KEY";

/// Environment variable for the default model
const GEMINI_DEFAULT_MODEL_ENV: &str = "GEMINI_DEFAULT_MODEL";

/// Prefix of the retry-tuning environment variables
const GEMINI_ENV_PREFIX: &str = "GEMINI";

/// Default model: Google's rolling alias for the current GA flash-lite tier
const DEFAULT_MODEL: &str = "gemini-flash-lite-latest";

/// Available Gemini models
const AVAILABLE_MODELS: &[&str] = &[
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
];

/// Base URL for the Gemini API
const API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// ============================================================================
// API Request/Response Types
// ============================================================================

/// Gemini API request structure
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

/// Content structure for Gemini API
#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    /// Content parts - may be empty for thinking-only responses from flash-lite class models
    #[serde(default)]
    parts: Vec<ContentPart>,
}

/// Part of content (text, function call, or function response)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ContentPart {
    /// Text content
    Text { text: String },
    /// Function call from the model
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    /// Function response from the user
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

/// Function call made by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

/// Response to a function call
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

/// Function declaration inside a tool definition
#[derive(Debug, Clone, Serialize)]
struct FunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

/// Tool definition for the Gemini API
#[derive(Debug, Clone, Serialize)]
struct GeminiTool {
    function_declarations: Vec<FunctionDeclaration>,
}

/// Generation configuration
#[derive(Debug, Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_count: Option<u32>,
}

/// Gemini API response structure
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
    error: Option<GeminiError>,
}

/// Response candidate
#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

/// Usage metadata from Gemini API response.
///
/// `cachedContentTokenCount` is surfaced so the billing pipeline can
/// charge cache hits at the discounted rate. It is a subset of
/// `promptTokenCount`: the prompt count reported by the API is the
/// gross token count (cache hits + fresh tokens).
#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total: Option<u32>,
    #[serde(rename = "cachedContentTokenCount")]
    cached: Option<u32>,
}

/// API error response from Gemini
#[derive(Debug, Deserialize)]
struct GeminiError {
    message: String,
}

/// Streaming response chunk
#[derive(Debug, Deserialize)]
struct StreamingResponse {
    candidates: Option<Vec<Candidate>>,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Gemini provider.
#[derive(Clone)]
pub struct GeminiConfig {
    /// API key from Google AI Studio. Sent as a query parameter; redacted in `Debug`.
    pub api_key: String,
    /// Model used when the request names none
    pub model: String,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
    /// Retry policy for transient failures (429, 503, thinking-only answers)
    pub retry: HttpRetryConfig,
}

impl GeminiConfig {
    /// A configuration with the given key and every other field at its default.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_owned(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            retry: HttpRetryConfig::default(),
        }
    }

    /// Read the configuration from the environment.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when `GEMINI_API_KEY`
    /// is not set.
    pub fn from_env() -> Result<Self, RunnerError> {
        let api_key = env::var(GEMINI_API_KEY_ENV).map_err(|_| {
            RunnerError::config(format!("{GEMINI_API_KEY_ENV} environment variable not set"))
        })?;
        let model = env::var(GEMINI_DEFAULT_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        Ok(Self {
            model,
            retry: HttpRetryConfig::from_env(GEMINI_ENV_PREFIX),
            ..Self::new(api_key)
        })
    }

    /// Set the default model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the retry policy
    #[must_use]
    pub fn with_retry(mut self, retry: HttpRetryConfig) -> Self {
        self.retry = retry;
        self
    }
}

impl Debug for GeminiConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("GeminiConfig")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .finish()
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

/// Google Gemini LLM provider
pub struct GeminiProvider {
    config: GeminiConfig,
    client: reqwest::Client,
    available_models: Vec<String>,
}

impl GeminiProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: GeminiConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    #[must_use]
    pub fn with_client(config: GeminiConfig, client: reqwest::Client) -> Self {
        debug!(
            model = %config.model,
            max_retries = config.retry.max_retries,
            initial_delay_ms = config.retry.initial_delay_ms,
            max_delay_ms = config.retry.max_delay_ms,
            "Gemini provider initialized"
        );
        Self {
            config,
            client,
            available_models: AVAILABLE_MODELS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Override the default model after construction.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Convert our message role to Gemini's role format
    ///
    /// System messages are handled separately via `system_instruction`; if one
    /// appears here, map it to "user" for compatibility.
    const fn convert_role(role: MessageRole) -> &'static str {
        match role {
            MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
            MessageRole::Assistant => "model",
        }
    }

    /// Build the API URL for a model and method. Carries the API key: never log it.
    fn build_url(&self, model: &str, method: &str) -> String {
        format!(
            "{API_BASE_URL}/models/{model}:{method}?key={}",
            self.config.api_key
        )
    }

    /// Convert chat messages to Gemini format
    fn convert_messages(messages: &[ChatMessage]) -> (Vec<GeminiContent>, Option<GeminiContent>) {
        let mut contents = Vec::new();
        // Every system message is concatenated, not overwritten: assigning
        // would make the LAST system message win, and a mid-list one would
        // silently destroy whatever persona sits at index 0. A provider reached
        // as a fallback must not depend on an invariant held elsewhere.
        let mut system_texts: Vec<String> = Vec::new();

        for message in messages {
            if message.role == MessageRole::System {
                system_texts.push(message.content.clone());
            } else {
                contents.push(GeminiContent {
                    role: Some(Self::convert_role(message.role).to_owned()),
                    parts: vec![ContentPart::Text {
                        text: message.content.clone(),
                    }],
                });
            }
        }

        let system_instruction = (!system_texts.is_empty()).then(|| GeminiContent {
            role: None,
            parts: vec![ContentPart::Text {
                text: system_texts.join("\n\n"),
            }],
        });

        (contents, system_instruction)
    }

    /// Convert tool definitions into Gemini's `functionDeclarations` shape
    fn convert_tools(tools: &[ToolDefinition]) -> Vec<GeminiTool> {
        vec![GeminiTool {
            function_declarations: tools
                .iter()
                .map(|t| FunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                })
                .collect(),
        }]
    }

    /// Build a Gemini API request from a `ChatRequest`
    fn build_gemini_request(request: &ChatRequest) -> GeminiRequest {
        let (contents, system_instruction) = Self::convert_messages(&request.messages);

        let generation_config = if request.temperature.is_some() || request.max_tokens.is_some() {
            Some(GenerationConfig {
                temperature: request.temperature,
                max_output_tokens: request.max_tokens,
                candidate_count: Some(1),
            })
        } else {
            None
        };

        let tools = request
            .tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(Self::convert_tools);

        GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        }
    }

    /// Extract text content from Gemini response
    fn extract_content(response: &GeminiResponse) -> Result<String, RunnerError> {
        let first = response.candidates.as_ref().and_then(|c| c.first());
        let content = first.and_then(|c| c.content.as_ref());

        // Handle empty or missing content - can happen with thinking models like flash-lite
        let Some(content) = content else {
            if let Some(reason) = first.and_then(|c| c.finish_reason.as_deref()) {
                if reason == "SAFETY" || reason == "RECITATION" || reason == "OTHER" {
                    return Err(RunnerError::internal(format!(
                        "Response blocked by Gemini safety filter: {reason}"
                    )));
                }
            }
            return Err(RunnerError::internal(
                "No content in Gemini response - model may still be thinking",
            ));
        };

        // Handle empty parts - thinking models may return content with no parts
        let Some(part) = content.parts.first() else {
            return Err(RunnerError::internal(
                "Gemini response has no content parts - model may have returned thinking-only output",
            ));
        };

        match part {
            ContentPart::Text { text } => Ok(text.clone()),
            ContentPart::FunctionCall { function_call } => Ok(format!(
                "{{\"function_call\": {{\"name\": \"{}\", \"args\": {}}}}}",
                function_call.name, function_call.args
            )),
            ContentPart::FunctionResponse { .. } => Err(RunnerError::internal(
                "Unexpected function response in model output",
            )),
        }
    }

    /// Extract the function calls from a Gemini response, as tool-call
    /// requests. Gemini sends no call id, so one is derived from the name and
    /// the call's index within the candidate.
    fn extract_tool_calls(response: &GeminiResponse) -> Vec<ToolCallRequest> {
        response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.content.as_ref())
            .map(|c| {
                c.parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::FunctionCall { function_call } => Some(function_call),
                        _ => None,
                    })
                    .enumerate()
                    .map(|(index, call)| ToolCallRequest {
                        id: format!("{}-{index}", call.name),
                        function_name: call.name.clone(),
                        arguments: call.args.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Convert usage metadata to our token usage format.
    ///
    /// Gemini reports no cache-write count, so that half stays `None`:
    /// absent, not zero.
    fn convert_usage(metadata: &UsageMetadata) -> TokenUsage {
        TokenUsage::new(
            metadata.prompt.unwrap_or(0),
            metadata.candidates.unwrap_or(0),
            metadata.total.unwrap_or(0),
        )
        .with_cache(metadata.cached, None)
    }

    /// Map an API error status to a [`RunnerError`].
    ///
    /// The vendor's message is lifted from the JSON error envelope when there
    /// is one; a non-JSON body is previewed at debug level and never reaches
    /// the message.
    fn map_api_error(status: StatusCode, response_text: &str) -> RunnerError {
        let parsed_message = serde_json::from_str::<GeminiResponse>(response_text)
            .ok()
            .and_then(|r| r.error)
            .map(|e| e.message);
        if parsed_message.is_none() {
            debug!(
                status = status.as_u16(),
                body_preview = %response_text.chars().take(200).collect::<String>(),
                "Gemini API returned non-JSON error response"
            );
        }
        let message = parsed_message.unwrap_or_else(|| format!("HTTP {status}"));
        client::map_http_error(PROVIDER_NAME, status, &message)
    }

    /// Whether an error is worth retrying in place.
    ///
    /// A quota refusal backs off; a 503 / overloaded answer backs off; a
    /// thinking-only or part-less answer may succeed on the next attempt.
    fn is_retryable_error(error: &RunnerError) -> bool {
        if error.kind == ErrorKind::RateLimit {
            return true;
        }
        let message = &error.message;
        message.contains("429")
            || message.contains("quota exceeded")
            || message.contains("rate limit")
            || message.contains("503")
            || message.contains("overloaded")
            || message.contains("no content parts")
            || message.contains("thinking-only")
            || message.contains("still be thinking")
    }

    /// Attach the retry decision to an attempt's error.
    fn classify(error: RunnerError) -> AttemptError {
        let retryable = Self::is_retryable_error(&error);
        AttemptError { error, retryable }
    }

    /// Parse a Gemini SSE data payload into a `StreamChunk`
    ///
    /// Gemini's streaming response uses a different JSON structure than
    /// OpenAI-compatible providers, requiring provider-specific parsing.
    fn parse_stream_data(json_str: &str) -> Option<Result<StreamChunk, RunnerError>> {
        match serde_json::from_str::<StreamingResponse>(json_str) {
            Ok(response) => {
                let candidate = response.candidates?.into_iter().next()?;
                let content = candidate.content?;
                let part = content.parts.first()?;

                let is_final = candidate
                    .finish_reason
                    .as_ref()
                    .is_some_and(|r| r == "STOP");

                let delta = match part {
                    ContentPart::Text { text } => text.clone(),
                    ContentPart::FunctionCall { function_call } => {
                        format!(
                            "{{\"function_call\": {{\"name\": \"{}\", \"args\": {}}}}}",
                            function_call.name, function_call.args
                        )
                    }
                    ContentPart::FunctionResponse { .. } => return None,
                };

                Some(Ok(StreamChunk {
                    delta,
                    is_final,
                    finish_reason: candidate.finish_reason,
                }))
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse Gemini streaming chunk");
                None
            }
        }
    }

    /// One `generateContent` attempt: send, read, map the status, parse the
    /// body, and turn the candidate into a response.
    async fn attempt_complete(
        &self,
        url: &str,
        gemini_request: &GeminiRequest,
        model: &str,
    ) -> Result<ChatResponse, RunnerError> {
        let response = self
            .client
            .post(url)
            .json(gemini_request)
            .send()
            .await
            .map_err(|e| map_send_error(PROVIDER_NAME, e))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| {
            RunnerError::external_service(PROVIDER_NAME, format!("Failed to read response: {e}"))
        })?;

        if !status.is_success() {
            return Err(Self::map_api_error(status, &response_text));
        }

        let gemini_response: GeminiResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                error!(error = %e, response_len = response_text.len(), "Failed to parse Gemini response (body redacted)");
                RunnerError::internal(format!("Failed to parse Gemini response: {e}"))
            })?;

        if let Some(error) = gemini_response.error {
            return Err(RunnerError::internal(format!(
                "Gemini API error: {}",
                error.message
            )));
        }

        let usage = gemini_response
            .usage_metadata
            .as_ref()
            .map(Self::convert_usage);
        let finish_reason = gemini_response
            .candidates
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.finish_reason.clone());

        let tool_calls = Self::extract_tool_calls(&gemini_response);
        if !tool_calls.is_empty() {
            debug!(count = tool_calls.len(), "Extracted function calls");
            return Ok(ChatResponse {
                content: String::new(),
                model: model.to_owned(),
                usage,
                finish_reason,
                warnings: None,
                tool_calls: Some(tool_calls),
            });
        }

        // May fail on thinking-only responses; the caller retries those.
        let content = Self::extract_content(&gemini_response)?;
        Ok(ChatResponse {
            content,
            model: model.to_owned(),
            usage,
            finish_reason,
            warnings: None,
            tool_calls: None,
        })
    }

    /// One `streamGenerateContent` attempt: open the stream, or map the
    /// status that refused it.
    async fn attempt_stream(
        &self,
        url: &str,
        gemini_request: &GeminiRequest,
    ) -> Result<ChatStream, RunnerError> {
        let response = self
            .client
            .post(url)
            .query(&[("alt", "sse")])
            .json(gemini_request)
            .send()
            .await
            .map_err(|e| map_send_error(PROVIDER_NAME, e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_owned());
            return Err(Self::map_api_error(status, &error_text));
        }

        // Gemini does not send [DONE]; the stream ends with the HTTP response
        Ok(create_sse_stream(
            response.bytes_stream(),
            Self::parse_stream_data,
            PROVIDER_NAME,
        ))
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn display_name(&self) -> &str {
        "Google Gemini"
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::full_featured()
    }

    fn default_model(&self) -> &str {
        &self.config.model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let model = request.model.as_deref().unwrap_or(&self.config.model);
        let url = self.build_url(model, "generateContent");
        let gemini_request = Self::build_gemini_request(request);
        let (url, gemini_request) = (&url, &gemini_request);

        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Gemini request",
            |attempt| async move {
                debug!(attempt, "Sending request to Gemini API");
                self.attempt_complete(url, gemini_request, model)
                    .await
                    .map_err(Self::classify)
            },
        )
        .await
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let model = request.model.as_deref().unwrap_or(&self.config.model);
        let url = self.build_url(model, "streamGenerateContent");
        let gemini_request = Self::build_gemini_request(request);
        let (url, gemini_request) = (&url, &gemini_request);

        // Retry the initial HTTP request (consistent with non-streaming complete())
        with_retries(
            &self.config.retry,
            PROVIDER_NAME,
            "Gemini streaming request",
            |attempt| async move {
                debug!(attempt, "Starting streaming request to Gemini API");
                self.attempt_stream(url, gemini_request)
                    .await
                    .map_err(Self::classify)
            },
        )
        .await
    }

    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        // List models to verify the API key is valid
        let url = format!("{API_BASE_URL}/models?key={}", self.config.api_key);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| map_send_error(PROVIDER_NAME, e))?;
        Ok(response.status().is_success())
    }
}

impl Debug for GeminiProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("GeminiProvider")
            .field("default_model", &self.config.model)
            .field("api_key", &"[REDACTED]")
            // Omit `client` field as HTTP clients are not useful to debug
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> GeminiProvider {
        GeminiProvider::with_client(
            GeminiConfig::new("super-secret-key"),
            reqwest::Client::new(),
        )
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let output = format!("{:?}", provider());
        assert!(!output.contains("super-secret-key"));
        assert!(output.contains("[REDACTED]"));
        let config = format!("{:?}", GeminiConfig::new("super-secret-key"));
        assert!(!config.contains("super-secret-key"));
    }

    #[test]
    fn system_messages_are_hoisted_and_concatenated() {
        let messages = vec![
            ChatMessage::system("persona"),
            ChatMessage::user("hi"),
            ChatMessage::system("tools"),
            ChatMessage::assistant("hello"),
            ChatMessage::tool("get_x", "call_1", "result"),
        ];
        let (contents, system) = GeminiProvider::convert_messages(&messages);
        let roles: Vec<Option<&str>> = contents.iter().map(|c| c.role.as_deref()).collect();
        assert_eq!(roles, vec![Some("user"), Some("model"), Some("user")]);
        let system = system.expect("system instruction"); // Safe: test assertion
        assert!(
            matches!(&system.parts[0], ContentPart::Text { text } if text == "persona\n\ntools"),
            "unexpected part {:?}",
            system.parts[0]
        );
    }

    #[test]
    fn tools_ride_as_function_declarations() {
        let request =
            ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition {
                name: "get_weather".to_owned(),
                description: "Weather".to_owned(),
                parameters: None,
            }]);
        let body = GeminiProvider::build_gemini_request(&request);
        let json = serde_json::to_value(&body).expect("serialises"); // Safe: test assertion
        assert_eq!(
            json["tools"][0]["function_declarations"][0]["name"],
            "get_weather"
        );
        let bare =
            GeminiProvider::build_gemini_request(&ChatRequest::new(vec![]).with_tools(vec![]));
        assert!(
            bare.tools.is_none(),
            "an empty tool list sends no tools field"
        );
    }

    #[test]
    fn function_calls_become_tool_calls_with_derived_ids() {
        let body = r#"{"candidates":[{"content":{"role":"model","parts":[
            {"functionCall":{"name":"get_activities","args":{"limit":5}}},
            {"functionCall":{"name":"get_activities","args":{"limit":9}}}
        ]},"finishReason":"STOP"}]}"#;
        let response: GeminiResponse = serde_json::from_str(body).expect("parses"); // Safe: test assertion
        let calls = GeminiProvider::extract_tool_calls(&response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "get_activities-0");
        assert_eq!(calls[1].id, "get_activities-1");
        assert_eq!(calls[0].arguments["limit"], 5);
    }

    #[test]
    fn a_part_less_candidate_is_a_retryable_internal_error() {
        let body =
            r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}]}"#;
        let response: GeminiResponse = serde_json::from_str(body).expect("parses"); // Safe: test assertion
        let err = GeminiProvider::extract_content(&response).expect_err("no parts");
        assert_eq!(err.kind, ErrorKind::Internal);
        assert!(GeminiProvider::is_retryable_error(&err));
    }

    #[test]
    fn a_429_is_a_rate_limit_with_the_wait_and_is_retried_in_place() {
        let body = r#"{"error":{"message":"Quota exceeded for quota metric. Please retry in 6.406453963s."}}"#;
        let err = GeminiProvider::map_api_error(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("7 seconds"), "{}", err.message);
        assert!(GeminiProvider::is_retryable_error(&err));
    }

    #[test]
    fn a_503_is_external_service_and_retried_a_401_is_not() {
        let err = GeminiProvider::map_api_error(StatusCode::SERVICE_UNAVAILABLE, "overloaded");
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(GeminiProvider::is_retryable_error(&err));
        let err = GeminiProvider::map_api_error(StatusCode::UNAUTHORIZED, "<html>");
        assert_eq!(err.kind, ErrorKind::AuthFailure);
        assert!(!GeminiProvider::is_retryable_error(&err));
    }

    #[test]
    fn usage_carries_the_cache_read_count() {
        let usage = GeminiProvider::convert_usage(&UsageMetadata {
            prompt: Some(100),
            candidates: Some(10),
            total: Some(110),
            cached: Some(60),
        });
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.cached_read_tokens, Some(60));
        assert_eq!(usage.cached_write_tokens, None);
    }

    #[test]
    fn stream_chunk_parses_text_and_final_stop() {
        let frame =
            r#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}]}"#;
        let chunk = GeminiProvider::parse_stream_data(frame)
            .expect("a chunk") // Safe: test assertion
            .expect("ok"); // Safe: test assertion
        assert_eq!(chunk.delta, "Hi");
        assert!(chunk.is_final);
    }
}
