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

/// Type alias for the shared state handle used across the server
pub type SharedState = Arc<RwLock<ServerState>>;

/// Central server state tracking provider configuration and cached runners
///
/// Runners are created lazily on first access and cached for reuse.
/// The active provider and model determine how prompt dispatch behaves.
pub struct ServerState {
    active_provider: CliRunnerType,
    active_model: Option<String>,
    multiplex_providers: Vec<CliRunnerType>,
    runners: Mutex<HashMap<CliRunnerType, Arc<dyn LlmProvider>>>,
}

impl ServerState {
    /// Create server state with the given default provider
    pub fn new(default_provider: CliRunnerType) -> Self {
        Self {
            active_provider: default_provider,
            active_model: None,
            multiplex_providers: Vec::new(),
            runners: Mutex::new(HashMap::new()),
        }
    }

    /// Get the currently active provider type
    pub const fn active_provider(&self) -> CliRunnerType {
        self.active_provider
    }

    /// Switch the active provider (resets the active model)
    pub fn set_active_provider(&mut self, provider: CliRunnerType) {
        self.active_provider = provider;
        self.active_model = None;
    }

    /// Get the currently selected model (None means use provider default)
    pub fn active_model(&self) -> Option<&str> {
        self.active_model.as_deref()
    }

    /// Set the model to use for subsequent requests
    pub fn set_active_model(&mut self, model: Option<String>) {
        self.active_model = model;
    }

    /// Get the list of providers configured for multiplex dispatch
    pub fn multiplex_providers(&self) -> &[CliRunnerType] {
        &self.multiplex_providers
    }

    /// Set the providers used when multiplexing prompts
    pub fn set_multiplex_providers(&mut self, providers: Vec<CliRunnerType>) {
        self.multiplex_providers = providers;
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

        let mut runners = self.runners.lock().await;
        let runner = runners.entry(provider).or_insert_with(|| runner).clone();
        Ok(runner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_uses_provided_provider() {
        let state = ServerState::new(CliRunnerType::Copilot);
        assert_eq!(state.active_provider(), CliRunnerType::Copilot);
        assert!(state.active_model().is_none());
        assert!(state.multiplex_providers().is_empty());
    }

    #[test]
    fn set_provider_resets_model() {
        let mut state = ServerState::new(CliRunnerType::Copilot);
        state.set_active_model(Some("gpt-4o".to_owned()));
        assert_eq!(state.active_model(), Some("gpt-4o"));

        state.set_active_provider(CliRunnerType::ClaudeCode);
        assert_eq!(state.active_provider(), CliRunnerType::ClaudeCode);
        assert!(state.active_model().is_none());
    }

    #[test]
    fn multiplex_providers_round_trip() {
        let mut state = ServerState::new(CliRunnerType::Copilot);
        let providers = vec![CliRunnerType::ClaudeCode, CliRunnerType::OpenCode];
        state.set_multiplex_providers(providers.clone());
        assert_eq!(state.multiplex_providers(), &providers);
    }
}
