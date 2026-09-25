// ABOUTME: Generic OpenAI-compatible provider: the OpenAI API itself, and self-hosted Ollama, vLLM, LocalAI and others
// ABOUTME: Names itself after the endpoint it targets, so the OpenAI API bills as openai_api and a self-hosted model at $0
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # `OpenAI`-Compatible Provider
//!
//! [`LlmProvider`] over any endpoint that implements the `OpenAI` chat
//! completions API: the `OpenAI` API itself, and local LLM servers like
//! Ollama, vLLM and `LocalAI`. The body it sends and the answers it reads
//! are the shared [`chat_completions`](super::chat_completions) wire format.
//!
//! ## Self-hosted endpoints
//!
//! [`OpenAiCompatibleProvider::from_env`] reads:
//!
//! - `LOCAL_LLM_BASE_URL`: default <http://localhost:11434/v1> (Ollama)
//! - `LOCAL_LLM_MODEL`: default `qwen2.5:14b-instruct`
//! - `LOCAL_LLM_API_KEY`: optional, empty for local servers
//!
//! The provider's `name()` is `ollama`, `vllm` or `localai` when the base
//! URL's port identifies one of those, `local` otherwise — the names the
//! price table lists as not per-token metered. Errors carry hints for a
//! server that is not running or a model that was never pulled.
//!
//! - **Ollama**: <http://localhost:11434/v1>
//! - **vLLM**: <http://localhost:8000/v1>
//! - **`LocalAI`**: <http://localhost:8080/v1>
//! - **Any `OpenAI`-compatible endpoint**
//!
//! ## The `OpenAI` API
//!
//! [`OpenAiCompatibleConfig::openai_api_from_env`] reads:
//!
//! - `OPENAI_API_BASE_URL`: default <https://api.openai.com>; requests go to
//!   its `/v1` (`/v1/chat/completions`, `/v1/models`)
//! - `OPENAI_API_KEY`: bearer token
//! - `OPENAI_API_MODEL`: default `gpt-5.4`
//! - `OPENAI_API_TIMEOUT_SECS`: request timeout of a client the provider
//!   builds itself (default 120)
//!
//! It reports `openai_api`, the price-table key its usage bills under, and
//! advertises vision, `top_p`, stop sequences and `response_format`, which
//! therefore reach the wire. [`OpenAiCompatibleProvider::with_discovered_models`]
//! publishes the endpoint's own `GET /v1/models` list. A failure reads as the
//! endpoint's own message under the shared status mapping, and a health check
//! that cannot reach the endpoint reports it unhealthy rather than failing.
//!
//! The full request and response bodies are logged at `trace` level only
//! (`RUST_LOG=embacle::http_api::openai_compatible=trace`).

use std::env;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::{RequestBuilder, StatusCode};
use serde::Deserialize;
use tracing::{debug, enabled, info, instrument, trace, warn, Level};

use super::chat_completions::{self, CompletionRequest, Usage};
use super::client::{self, map_send_error, DEFAULT_TIMEOUT_SECS};
use super::sse::create_sse_stream;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

// ============================================================================
// Configuration Constants
// ============================================================================

/// Environment variable for local LLM base URL
const LOCAL_LLM_BASE_URL_ENV: &str = "LOCAL_LLM_BASE_URL";

/// Environment variable for local LLM default model
const LOCAL_LLM_MODEL_ENV: &str = "LOCAL_LLM_MODEL";

/// Environment variable for local LLM API key (optional)
const LOCAL_LLM_API_KEY_ENV: &str = "LOCAL_LLM_API_KEY";

/// Default base URL (Ollama)
const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// Default model for local inference
const DEFAULT_MODEL: &str = "qwen2.5:14b-instruct";

/// The name error messages and logs use for whichever endpoint is configured
const LOG_LABEL: &str = "LocalLLM";

/// Models commonly served by the local runtimes this provider targets.
const AVAILABLE_MODELS: &[&str] = &[
    "qwen2.5:14b-instruct",
    "qwen2.5:7b-instruct",
    "qwen2.5:32b-instruct",
    "llama3.1:8b-instruct",
    "llama3.1:70b-instruct",
    "llama3.3:70b-instruct",
    "mistral:7b-instruct",
    "hermes2pro:latest",
];

/// The name the `OpenAI` API reports, and the price-table key its usage bills under
const OPENAI_API_NAME: &str = "openai_api";

/// Display name of the `OpenAI` API
const OPENAI_API_DISPLAY_NAME: &str = "OpenAI API";

/// Environment variable for the `OpenAI` API base URL (without `/v1`)
const OPENAI_API_BASE_URL_ENV: &str = "OPENAI_API_BASE_URL";

/// Environment variable for the `OpenAI` API key
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

/// Environment variable for the `OpenAI` API default model
const OPENAI_API_MODEL_ENV: &str = "OPENAI_API_MODEL";

/// Environment variable for the `OpenAI` API request timeout, in seconds
const OPENAI_API_TIMEOUT_SECS_ENV: &str = "OPENAI_API_TIMEOUT_SECS";

/// Default `OpenAI` API base URL, without the version path
const OPENAI_API_DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// The version path the `OpenAI` API serves chat completions and models under
const OPENAI_API_VERSION_PATH: &str = "/v1";

/// Default `OpenAI` API model
const OPENAI_API_DEFAULT_MODEL: &str = "gpt-5.4";

/// Timeout of a model-discovery or health-check request, in seconds
const DISCOVERY_TIMEOUT_SECS: u64 = 5;

// ============================================================================
// Model discovery wire types
// ============================================================================

/// The `GET {base}/models` answer
#[derive(Debug, Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

/// One entry of the models list
#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

// ============================================================================
// Provider Configuration
// ============================================================================

/// Configuration for the `OpenAI`-compatible provider
#[derive(Clone)]
pub struct OpenAiCompatibleConfig {
    /// Base URL for the API (e.g., <http://localhost:11434/v1>)
    pub base_url: String,
    /// API key (optional for local servers). Redacted in `Debug`.
    pub api_key: Option<String>,
    /// Default model to use
    pub default_model: String,
    /// Provider name for logging; decides what `name()` reports
    pub provider_name: String,
    /// Provider display name
    pub display_name: String,
    /// Capabilities of this provider; they decide which optional request
    /// fields reach the wire
    pub capabilities: LlmCapabilities,
    /// HTTP request timeout for a client the provider builds itself
    pub timeout: Duration,
}

impl OpenAiCompatibleConfig {
    /// Create configuration for a local Ollama instance
    #[must_use]
    pub fn ollama(model: &str) -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "ollama".to_owned(),
            display_name: "Ollama (Local)".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Create configuration for a local vLLM instance
    #[must_use]
    pub fn vllm(model: &str) -> Self {
        Self {
            base_url: "http://localhost:8000/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "vllm".to_owned(),
            display_name: "vLLM (Local)".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES
                | LlmCapabilities::JSON_MODE,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Create configuration for `LocalAI`
    #[must_use]
    pub fn local_ai(model: &str) -> Self {
        Self {
            base_url: "http://localhost:8080/v1".to_owned(),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: "localai".to_owned(),
            display_name: "LocalAI".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Read the configuration from the environment; the endpoint's port
    /// decides the provider name (`11434` Ollama, `8000` vLLM, `8080`
    /// `LocalAI`, anything else `local`).
    #[must_use]
    pub fn from_env() -> Self {
        let base_url =
            env::var(LOCAL_LLM_BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let default_model =
            env::var(LOCAL_LLM_MODEL_ENV).unwrap_or_else(|_| DEFAULT_MODEL.to_owned());
        let api_key = env::var(LOCAL_LLM_API_KEY_ENV)
            .ok()
            .filter(|k| !k.is_empty());

        let (provider_name, display_name) = if base_url.contains(":11434") {
            ("ollama", "Ollama (Local)")
        } else if base_url.contains(":8000") {
            ("vllm", "vLLM (Local)")
        } else if base_url.contains(":8080") {
            ("localai", "LocalAI")
        } else {
            ("local", "Local LLM")
        };

        Self {
            base_url,
            api_key,
            default_model,
            provider_name: provider_name.to_owned(),
            display_name: display_name.to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Create configuration for the `OpenAI` API at
    /// <https://api.openai.com/v1>, reporting `openai_api`.
    ///
    /// Any endpoint that speaks the same API under its own `/v1` works by
    /// replacing `base_url`.
    #[must_use]
    pub fn openai_api(model: &str) -> Self {
        Self {
            base_url: format!("{OPENAI_API_DEFAULT_BASE_URL}{OPENAI_API_VERSION_PATH}"),
            api_key: None,
            default_model: model.to_owned(),
            provider_name: OPENAI_API_NAME.to_owned(),
            display_name: OPENAI_API_DISPLAY_NAME.to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::VISION
                | LlmCapabilities::SYSTEM_MESSAGES
                | LlmCapabilities::TEMPERATURE
                | LlmCapabilities::MAX_TOKENS
                | LlmCapabilities::TOP_P
                | LlmCapabilities::STOP_SEQUENCES
                | LlmCapabilities::RESPONSE_FORMAT,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }

    /// Read the `OpenAI` API configuration from the environment.
    ///
    /// `OPENAI_API_BASE_URL` (default <https://api.openai.com>) names the
    /// host without the version path: requests go to its `/v1`. An unset
    /// `OPENAI_API_MODEL` is `gpt-5.4`, an unset or unparseable
    /// `OPENAI_API_TIMEOUT_SECS` is 120, and an unset or empty
    /// `OPENAI_API_KEY` sends no `Authorization` header.
    #[must_use]
    pub fn openai_api_from_env() -> Self {
        let base_url = env::var(OPENAI_API_BASE_URL_ENV)
            .unwrap_or_else(|_| OPENAI_API_DEFAULT_BASE_URL.to_owned());
        let model =
            env::var(OPENAI_API_MODEL_ENV).unwrap_or_else(|_| OPENAI_API_DEFAULT_MODEL.to_owned());
        let timeout_secs = env::var(OPENAI_API_TIMEOUT_SECS_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        Self {
            base_url: format!(
                "{}{OPENAI_API_VERSION_PATH}",
                base_url.trim_end_matches('/')
            ),
            api_key: env::var(OPENAI_API_KEY_ENV).ok().filter(|k| !k.is_empty()),
            timeout: Duration::from_secs(timeout_secs),
            ..Self::openai_api(&model)
        }
    }

    /// Whether this is the `OpenAI` API rather than a self-hosted server:
    /// it decides the error wording, what an unreachable health check means,
    /// and the model list before discovery.
    fn is_openai_api(&self) -> bool {
        self.provider_name == OPENAI_API_NAME
    }
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            api_key: None,
            default_model: DEFAULT_MODEL.to_owned(),
            provider_name: "local".to_owned(),
            display_name: "Local LLM".to_owned(),
            capabilities: LlmCapabilities::STREAMING
                | LlmCapabilities::FUNCTION_CALLING
                | LlmCapabilities::SYSTEM_MESSAGES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }
}

impl Debug for OpenAiCompatibleConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenAiCompatibleConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("default_model", &self.default_model)
            .field("provider_name", &self.provider_name)
            .field("display_name", &self.display_name)
            .field("capabilities", &self.capabilities)
            .field("timeout", &self.timeout)
            .finish()
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

/// LLM provider for OpenAI-compatible APIs: the `OpenAI` API, Ollama, vLLM,
/// LM Studio, and others
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    config: OpenAiCompatibleConfig,
    available_models: Vec<String>,
}

impl OpenAiCompatibleProvider {
    /// Create a provider that builds its own HTTP client from `config.timeout`.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, RunnerError> {
        let client = client::build_client(config.timeout)?;
        Ok(Self::with_client(config, client))
    }

    /// Create a provider over a caller-owned HTTP client (a shared pool).
    ///
    /// A self-hosted endpoint publishes the models the local runtimes commonly
    /// serve; the `OpenAI` API publishes its default model until
    /// [`with_discovered_models`](Self::with_discovered_models) asks it.
    #[must_use]
    pub fn with_client(config: OpenAiCompatibleConfig, client: reqwest::Client) -> Self {
        let available_models = if config.is_openai_api() {
            vec![config.default_model.clone()]
        } else {
            AVAILABLE_MODELS.iter().map(|s| (*s).to_owned()).collect()
        };
        Self {
            client,
            config,
            available_models,
        }
    }

    /// Override the default model after construction.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Publish the endpoint's own model list: `GET {base_url}/models`, sorted.
    ///
    /// The list published so far is kept when the endpoint cannot be reached
    /// within five seconds, refuses, or lists nothing.
    pub async fn with_discovered_models(mut self) -> Self {
        let discovered = self.discover_models().await;
        if !discovered.is_empty() {
            self.available_models = discovered;
        }
        debug!(
            provider = %self.config.provider_name,
            base_url = %self.config.base_url,
            model = %self.config.default_model,
            published_models = self.available_models.len(),
            "OpenAI-compatible model list resolved"
        );
        self
    }

    /// Create a provider from environment variables (see
    /// [`OpenAiCompatibleConfig::from_env`]).
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with `ErrorKind::Config` when the HTTP client
    /// cannot be built.
    pub fn from_env() -> Result<Self, RunnerError> {
        let config = OpenAiCompatibleConfig::from_env();
        info!(
            provider = %config.display_name,
            base_url = %config.base_url,
            default_model = %config.default_model,
            "Initializing OpenAI-compatible provider"
        );
        Self::new(config)
    }

    /// Build the API URL for a given endpoint
    fn api_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint
        )
    }

    /// The name errors carry: `openai_api` for the `OpenAI` API, one label
    /// for whichever self-hosted server is configured.
    fn label(&self) -> &'static str {
        if self.config.is_openai_api() {
            OPENAI_API_NAME
        } else {
            LOG_LABEL
        }
    }

    /// Log message details for debugging LLM interactions
    fn log_messages_debug(messages: &[ChatMessage], provider_name: &str, has_tools: bool) {
        for (i, msg) in messages.iter().enumerate() {
            debug!(
                "Message[{i}] role={}, content_len={}",
                msg.role.as_str(),
                msg.content.len()
            );
            if msg.role.as_str() == "system" {
                debug!("System prompt present (content redacted for security)");
            }
        }
        debug!(
            "Sending chat completion request to {provider_name} with {} messages and tools={has_tools:?}",
            messages.len()
        );
    }

    /// Map a non-success response to an error, in the configured endpoint's
    /// terms.
    fn error_response(&self, status: StatusCode, body: &str) -> RunnerError {
        if self.config.is_openai_api() {
            Self::openai_api_error_response(status, body)
        } else {
            Self::parse_error_response(status, body)
        }
    }

    /// The `OpenAI` API's error: the vendor's message when the body is its
    /// envelope and the body as sent otherwise; the status decides the kind.
    fn openai_api_error_response(status: StatusCode, body: &str) -> RunnerError {
        let message = chat_completions::error_detail(body)
            .map_or_else(|| body.to_owned(), |detail| detail.message);
        client::map_http_error(OPENAI_API_NAME, status, &message)
    }

    /// Parse an error response from a self-hosted endpoint.
    ///
    /// Local servers answer with HTML or an empty body more often than a
    /// JSON envelope, so a 502–504 without JSON reads as "the local server is
    /// not responding". A 404 is an unavailable model (Ollama's answer to a
    /// model that was never pulled), a 503 says the server is starting.
    fn parse_error_response(status: StatusCode, body: &str) -> RunnerError {
        let Some(detail) = chat_completions::error_detail(body) else {
            if (502..=504).contains(&status.as_u16()) {
                return RunnerError::external_service(
                    LOG_LABEL,
                    "Local LLM server is not responding. Is Ollama/vLLM running?",
                );
            }
            debug!(
                status = status.as_u16(),
                body_preview = %body.chars().take(200).collect::<String>(),
                "Non-JSON error response body"
            );
            return client::map_http_error(LOG_LABEL, status, &format!("HTTP {status}"));
        };

        match status.as_u16() {
            404 => RunnerError::model_unavailable(format!("endpoint or model: {}", detail.message)),
            503 => RunnerError::external_service(
                LOG_LABEL,
                format!(
                    "Service unavailable (is the local server running?): {}",
                    detail.message
                ),
            ),
            _ => client::map_http_error(LOG_LABEL, status, &detail.typed_message()),
        }
    }

    /// Map a transport failure. A self-hosted endpoint that refused the
    /// connection is named, so the operator knows which server to start.
    fn map_send(&self, error: reqwest::Error) -> RunnerError {
        if error.is_connect() && !self.config.is_openai_api() {
            RunnerError::external_service(
                LOG_LABEL,
                format!(
                    "Cannot connect to {}. Is the server running at {}?",
                    self.config.display_name, self.config.base_url
                ),
            )
        } else {
            map_send_error(self.label(), error)
        }
    }

    /// Add authorization header if API key is configured
    fn add_auth_header(&self, request: RequestBuilder) -> RequestBuilder {
        match &self.config.api_key {
            Some(api_key) => request.bearer_auth(api_key),
            None => request,
        }
    }

    /// Build the request body for a `ChatRequest`
    fn build_body(&self, request: &ChatRequest, stream: bool) -> CompletionRequest {
        let body = CompletionRequest::new(
            request,
            &self.config.default_model,
            stream,
            self.config.capabilities,
        );
        Self::log_messages_debug(
            &request.messages,
            &self.config.provider_name,
            body.tools.is_some(),
        );
        body
    }

    /// Send a chat-completions request, logging its outcome and latency
    async fn send(&self, body: &CompletionRequest) -> Result<reqwest::Response, RunnerError> {
        if enabled!(Level::TRACE) {
            match serde_json::to_string(body) {
                Ok(json) => trace!(body_len = json.len(), body = %json, "request body"),
                Err(e) => trace!(error = %e, "request body serialization failed"),
            }
        }
        let http_request = self
            .client
            .post(self.api_url("chat/completions"))
            .header("Content-Type", "application/json")
            .json(body);
        let started = Instant::now();
        let response = self
            .add_auth_header(http_request)
            .send()
            .await
            .map_err(|e| self.map_send(e))?;

        let status = response.status();
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        if status.is_success() {
            info!(
                provider = self.name(),
                model = %body.model,
                stream = body.stream,
                status = status.as_u16(),
                latency_ms,
                "chat completion response received"
            );
        } else {
            warn!(
                provider = self.name(),
                model = %body.model,
                stream = body.stream,
                status = status.as_u16(),
                latency_ms,
                "chat completion non-success response"
            );
        }
        Ok(response)
    }

    /// Fetch the endpoint's model ids, sorted; empty when it cannot say.
    async fn discover_models(&self) -> Vec<String> {
        let request = self.add_auth_header(
            self.client
                .get(self.api_url("models"))
                .timeout(Duration::from_secs(DISCOVERY_TIMEOUT_SECS)),
        );
        let response = match request.send().await {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                debug!(status = %response.status(), "Model discovery returned non-200");
                return Vec::new();
            }
            Err(e) => {
                debug!(error = %e.without_url(), "Model discovery failed");
                return Vec::new();
            }
        };
        let body = match response.text().await {
            Ok(body) => body,
            Err(e) => {
                debug!(error = %e.without_url(), "Model discovery body read failed");
                return Vec::new();
            }
        };
        match serde_json::from_str::<ModelList>(&body) {
            Ok(list) => {
                let mut ids: Vec<String> = list.data.into_iter().map(|m| m.id).collect();
                ids.sort();
                ids
            }
            Err(e) => {
                debug!(error = %e, "Model discovery parse failed");
                Vec::new()
            }
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &'static str {
        // The trait wants a static string; these are the names the price
        // table knows: the OpenAI API, and four self-hosted ones.
        match self.config.provider_name.as_str() {
            OPENAI_API_NAME => OPENAI_API_NAME,
            "ollama" => "ollama",
            "vllm" => "vllm",
            "localai" => "localai",
            _ => "local",
        }
    }

    fn display_name(&self) -> &str {
        &self.config.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.config.capabilities
    }

    fn default_model(&self) -> &str {
        &self.config.default_model
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.default_model)))]
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let body = self.build_body(request, false);
        let response = self.send(&body).await?;

        let status = response.status();
        let text = response.text().await.map_err(|e| {
            RunnerError::external_service(self.label(), format!("Failed to read response: {e}"))
        })?;
        trace!(body_len = text.len(), body = %text, "response body");

        if !status.is_success() {
            return Err(self.error_response(status, &text));
        }

        let response = chat_completions::parse_response(self.label(), &text, Usage::with_details)?;
        if let Some(usage) = &response.usage {
            info!(
                provider = self.name(),
                model = %response.model,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                total_tokens = usage.total_tokens,
                "chat completion usage"
            );
        }
        Ok(response)
    }

    #[instrument(skip(self, request), fields(model = %request.model.as_deref().unwrap_or(&self.config.default_model)))]
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let body = self.build_body(request, true);
        let response = self.send(&body).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(self.error_response(status, &text));
        }

        let label = self.label();
        Ok(create_sse_stream(
            response.bytes_stream(),
            move |json| chat_completions::parse_stream_frame(label, json),
            label,
        ))
    }

    /// Probe `GET {base_url}/models` within five seconds.
    ///
    /// The `OpenAI` API that cannot be reached is unhealthy; a self-hosted
    /// endpoint that cannot be reached is an error naming the server to start.
    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RunnerError> {
        debug!(
            provider = %self.config.provider_name,
            base_url = %self.config.base_url,
            "Performing health check"
        );

        let http_request = self
            .client
            .get(self.api_url("models"))
            .timeout(Duration::from_secs(DISCOVERY_TIMEOUT_SECS));
        let response = match self.add_auth_header(http_request).send().await {
            Ok(response) => response,
            Err(e) if self.config.is_openai_api() => {
                warn!(
                    provider = OPENAI_API_NAME,
                    error = %e.without_url(),
                    "health check could not reach the endpoint"
                );
                return Ok(false);
            }
            Err(e) => return Err(self.map_send(e)),
        };

        let healthy = response.status().is_success();
        if healthy {
            debug!("{} health check passed", self.config.provider_name);
        } else {
            warn!(
                provider = %self.config.provider_name,
                status = response.status().as_u16(),
                "health check failed"
            );
        }
        Ok(healthy)
    }
}

impl Debug for OpenAiCompatibleProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OpenAiCompatibleProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn a_non_json_gateway_error_says_the_server_is_not_responding() {
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::BAD_GATEWAY, "<html>");
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("not responding"));
    }

    #[test]
    fn a_404_is_an_unavailable_model_a_400_an_invalid_request() {
        let body = r#"{"error":{"message":"model 'qwen3' not found","type":"api_error"}}"#;
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::NOT_FOUND, body);
        assert_eq!(err.kind, ErrorKind::ModelUnavailable);
        let body = r#"{"error":{"message":"invalid messages","type":"invalid_request_error"}}"#;
        let err = OpenAiCompatibleProvider::parse_error_response(StatusCode::BAD_REQUEST, body);
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
    }

    #[test]
    fn a_429_is_a_rate_limit_and_a_503_external_service() {
        let body = r#"{"error":{"message":"Rate limit reached. Please try again in 20s."}}"#;
        let err =
            OpenAiCompatibleProvider::parse_error_response(StatusCode::TOO_MANY_REQUESTS, body);
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("20 seconds"));
        let body = r#"{"error":{"message":"loading model"}}"#;
        let err =
            OpenAiCompatibleProvider::parse_error_response(StatusCode::SERVICE_UNAVAILABLE, body);
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("is the local server running"));
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let config = OpenAiCompatibleConfig {
            api_key: Some("super-secret".to_owned()),
            ..OpenAiCompatibleConfig::default()
        };
        let output = format!("{config:?}");
        assert!(!output.contains("super-secret"));
        assert!(output.contains("[REDACTED]"));
    }
}
