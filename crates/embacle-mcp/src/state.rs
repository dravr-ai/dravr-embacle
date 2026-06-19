// ABOUTME: Shared server state holding active provider, model, and multiplex configuration
// ABOUTME: Thread-safe via Arc<RwLock> with lazy runner creation on first use
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::collections::HashMap;
use std::sync::Arc;

use embacle::config::CliRunnerType;
use embacle::types::{LlmProvider, RunnerError};
use tokio::sync::{Mutex, RwLock};

use crate::runner::factory;

/// Type alias for the shared state handle used across the server.
///
/// `dravr-tronc` 0.5 shares state as `Arc<S>`, so the mutable provider/model
/// configuration lives behind a per-field interior `RwLock` rather than an
/// outer one.
pub type SharedState = Arc<ServerState>;

/// Interior-mutable provider configuration: the active provider, the optional
/// model override, and the multiplex fan-out list, co-locked so a provider
/// switch can atomically reset the model.
struct ActiveConfig {
    provider: CliRunnerType,
    model: Option<String>,
    multiplex: Vec<CliRunnerType>,
}

/// Central server state tracking provider configuration and cached runners
///
/// Runners are created lazily on first access and cached for reuse.
/// The active provider and model determine how prompt dispatch behaves.
pub struct ServerState {
    config: RwLock<ActiveConfig>,
    runners: Mutex<HashMap<CliRunnerType, Arc<dyn LlmProvider>>>,
}

impl ServerState {
    /// Create server state with the given default provider
    pub fn new(default_provider: CliRunnerType) -> Self {
        Self {
            config: RwLock::new(ActiveConfig {
                provider: default_provider,
                model: None,
                multiplex: Vec::new(),
            }),
            runners: Mutex::new(HashMap::new()),
        }
    }

    /// Get the currently active provider type
    pub async fn active_provider(&self) -> CliRunnerType {
        self.config.read().await.provider
    }

    /// Switch the active provider (resets the active model)
    pub async fn set_active_provider(&self, provider: CliRunnerType) {
        let mut config = self.config.write().await;
        config.provider = provider;
        config.model = None;
    }

    /// Get the currently selected model (None means use provider default)
    pub async fn active_model(&self) -> Option<String> {
        self.config.read().await.model.clone()
    }

    /// Set the model to use for subsequent requests
    pub async fn set_active_model(&self, model: Option<String>) {
        self.config.write().await.model = model;
    }

    /// Get the list of providers configured for multiplex dispatch
    pub async fn multiplex_providers(&self) -> Vec<CliRunnerType> {
        self.config.read().await.multiplex.clone()
    }

    /// Set the providers used when multiplexing prompts
    pub async fn set_multiplex_providers(&self, providers: Vec<CliRunnerType>) {
        self.config.write().await.multiplex = providers;
    }

    /// Get or lazily create a runner for the given provider type
    ///
    /// Created runners are cached for future calls. The runner cache uses
    /// interior mutability so callers only need `&self`.
    pub async fn get_runner(
        &self,
        provider: CliRunnerType,
    ) -> Result<Arc<dyn LlmProvider>, RunnerError> {
        // Fast path: check cache under lock
        {
            let runners = self.runners.lock().await;
            if let Some(runner) = runners.get(&provider) {
                return Ok(Arc::clone(runner));
            }
        }

        // Slow path: create runner without holding the lock
        let runner = factory::create_runner(provider).await?;
        let runner: Arc<dyn LlmProvider> = Arc::from(runner);

        let runner = self
            .runners
            .lock()
            .await
            .entry(provider)
            .or_insert_with(|| runner)
            .clone();
        Ok(runner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn default_state_uses_provided_provider() {
        let state = ServerState::new(CliRunnerType::Copilot);
        assert_eq!(state.active_provider().await, CliRunnerType::Copilot);
        assert!(state.active_model().await.is_none());
        assert!(state.multiplex_providers().await.is_empty());
    }

    #[tokio::test]
    async fn set_provider_resets_model() {
        let state = ServerState::new(CliRunnerType::Copilot);
        state.set_active_model(Some("gpt-4o".to_owned())).await;
        assert_eq!(state.active_model().await, Some("gpt-4o".to_owned()));

        state.set_active_provider(CliRunnerType::ClaudeCode).await;
        assert_eq!(state.active_provider().await, CliRunnerType::ClaudeCode);
        assert!(state.active_model().await.is_none());
    }

    #[tokio::test]
    async fn multiplex_providers_round_trip() {
        let state = ServerState::new(CliRunnerType::Copilot);
        let providers = vec![CliRunnerType::ClaudeCode, CliRunnerType::OpenCode];
        state.set_multiplex_providers(providers.clone()).await;
        assert_eq!(state.multiplex_providers().await, providers);
    }
}
