// ABOUTME: Multiplex engine that fans out prompts to multiple providers concurrently
// ABOUTME: Collects per-provider results with timing and produces an aggregated summary
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::sync::Arc;
use std::time::Instant;

use embacle::config::CliRunnerType;
use embacle::types::{ChatMessage, ChatRequest, RunnerError};
use serde::Serialize;

use crate::state::SharedState;

/// Aggregated result from dispatching a prompt to multiple providers
#[derive(Debug, Serialize)]
pub struct MultiplexResult {
    /// Individual per-provider responses
    pub responses: Vec<ProviderResponse>,
    /// Human-readable summary of the multiplex operation
    pub summary: String,
}

/// Response from a single provider in a multiplex operation
#[derive(Debug, Serialize)]
pub struct ProviderResponse {
    /// Provider identifier
    pub provider: String,
    /// Response content (None on failure)
    pub content: Option<String>,
    /// Model used by the provider
    pub model: Option<String>,
    /// Error message (None on success)
    pub error: Option<String>,
    /// Wall-clock time in milliseconds
    pub duration_ms: u64,
}

/// Engine that dispatches prompts to multiple embacle runners concurrently
pub struct MultiplexEngine {
    state: SharedState,
}

impl MultiplexEngine {
    /// Create a new multiplex engine backed by the shared server state
    pub const fn new(state: SharedState) -> Self {
        Self { state }
    }

    /// Execute a prompt against all specified providers concurrently
    ///
    /// Each provider runs in its own tokio task. Failures in one provider
    /// do not affect others — all results are collected and returned.
    pub async fn execute(
        &self,
        messages: &[ChatMessage],
        providers: &[CliRunnerType],
    ) -> Result<MultiplexResult, RunnerError> {
        let mut handles = Vec::with_capacity(providers.len());

        for &provider in providers {
            let state = Arc::clone(&self.state);
            let messages = messages.to_vec();

            handles.push(tokio::spawn(async move {
                dispatch_single(state, provider, messages).await
            }));
        }

        let mut responses = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(resp) => responses.push(resp),
                Err(e) => responses.push(ProviderResponse {
                    provider: "unknown".to_owned(),
                    content: None,
                    model: None,
                    error: Some(format!("Task join error: {e}")),
                    duration_ms: 0,
                }),
            }
        }

        let summary = build_summary(&responses);
        Ok(MultiplexResult { responses, summary })
    }
}

/// Dispatch a single prompt to one provider and capture the result
async fn dispatch_single(
    state: SharedState,
    provider: CliRunnerType,
    messages: Vec<ChatMessage>,
) -> ProviderResponse {
    let start = Instant::now();

    let runner = {
        let state_guard = state.read().await;
        state_guard.get_runner(provider).await
    };

    let runner = match runner {
        Ok(r) => r,
        Err(e) => {
            return ProviderResponse {
                provider: provider.to_string(),
                content: None,
                model: None,
                error: Some(e.to_string()),
                duration_ms: elapsed_ms(start),
            };
        }
    };

    let request = ChatRequest::new(messages);
    match runner.complete(&request).await {
        Ok(response) => ProviderResponse {
            provider: provider.to_string(),
            content: Some(response.content),
            model: Some(response.model),
            error: None,
            duration_ms: elapsed_ms(start),
        },
        Err(e) => ProviderResponse {
            provider: provider.to_string(),
            content: None,
            model: None,
            error: Some(e.to_string()),
            duration_ms: elapsed_ms(start),
        },
    }
}

/// Build a human-readable summary from multiplex responses
fn build_summary(responses: &[ProviderResponse]) -> String {
    let total = responses.len();
    let succeeded = responses.iter().filter(|r| r.content.is_some()).count();
    let failed = total - succeeded;
    format!("{succeeded} succeeded, {failed} failed out of {total} providers")
}

/// Convert elapsed time to milliseconds as u64
fn elapsed_ms(start: Instant) -> u64 {
    let millis = start.elapsed().as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}
