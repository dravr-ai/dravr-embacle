// ABOUTME: A pool of CLI runners of one type, one per credential, as ordered tiers of a chain
// ABOUTME: A spent account moves the turn to the next account before the chain leaves the runner type
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Credential pools
//!
//! One account's quota must never dead-end a turn. A pool is N runners of one
//! CLI type built from N credentials, each carrying its own token as explicit
//! child environment ([`RunnerConfig::with_env`]), so the sandbox's clearing of
//! the process environment never matters and every runner logs in as its own
//! account. Handed to a [`FallbackProvider`](crate::fallback::FallbackProvider)
//! in order, they are ordinary tiers: account 1 serves every turn until it
//! refuses on quota, the chain moves to account 2, and the refused account is
//! held in cooldown until it is worth asking again.
//!
//! Attribution stays per account. Tier N is named `<runner>#<N>` (the first
//! keeps the runner's own name), and its responses report that name as their
//! `model`, so a health snapshot, a quota alert label or a usage row says
//! which account served. Pricing keys on the family before the `#`
//! ([`crate::pricing::pool_family`]), so every account bills as its runner.
//!
//! There is no load spreading. Spreading turns across accounts would hide a
//! spent account until the last one dies, which is the failure this exists to
//! remove.

use async_trait::async_trait;
use tokio_stream::StreamExt;

use crate::claude_code::CREDENTIAL_ENV_KEYS as CLAUDE_KEYS;
use crate::codex_cli::CREDENTIAL_ENV_KEYS as CODEX_KEYS;
use crate::config::{CliRunnerType, RunnerConfig};
use crate::copilot::CREDENTIAL_ENV_KEYS as COPILOT_KEYS;
use crate::factory::create_runner_with_config;
use crate::gemini_cli::CREDENTIAL_ENV_KEYS as GEMINI_KEYS;
use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// The variable a runner type reads its credential from, when it reads one.
///
/// This is the first of that runner's declared credential keys — the one the
/// binary itself prefers — and it is where a pooled credential is set.
#[must_use]
pub const fn credential_env_key(runner_type: CliRunnerType) -> Option<&'static str> {
    match runner_type {
        CliRunnerType::ClaudeCode => Some(CLAUDE_KEYS[0]),
        CliRunnerType::Copilot => Some(COPILOT_KEYS[0]),
        CliRunnerType::GeminiCli => Some(GEMINI_KEYS[0]),
        CliRunnerType::CodexCli => Some(CODEX_KEYS[0]),
        _ => None,
    }
}

/// One account of a pool: a runner under a per-account name.
///
/// Every call is the inner runner's. Only the naming differs: [`name`] is the
/// pooled name, a completion's `model` reports it, and an error that names
/// the runner (`claude-code: usage limit reached`) names the account instead
/// (`claude-code#2: …`) — a quota alert reads the tier from that text, so a
/// spent second account must not be reported as the first.
///
/// [`name`]: LlmProvider::name
pub struct PooledTier {
    inner: Box<dyn LlmProvider>,
    name: &'static str,
    display_name: String,
}

impl PooledTier {
    /// Wrap `inner` as account `position` (zero-based) of its pool.
    ///
    /// The name is leaked once per tier: a provider lives as long as the
    /// process, and [`LlmProvider::name`] returns `&'static str`.
    #[must_use]
    pub fn new(inner: Box<dyn LlmProvider>, position: usize) -> Self {
        let name: &'static str = if position == 0 {
            inner.name()
        } else {
            Box::leak(format!("{}#{}", inner.name(), position + 1).into_boxed_str())
        };
        let display_name = if position == 0 {
            inner.display_name().to_owned()
        } else {
            format!("{} (account {})", inner.display_name(), position + 1)
        };
        Self {
            inner,
            name,
            display_name,
        }
    }
}

#[async_trait]
impl LlmProvider for PooledTier {
    fn name(&self) -> &'static str {
        self.name
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        self.inner.capabilities()
    }

    fn default_model(&self) -> &str {
        self.inner.default_model()
    }

    fn available_models(&self) -> &[String] {
        self.inner.available_models()
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let mut response = self
            .inner
            .complete(request)
            .await
            .map_err(|e| self.rename(e))?;
        self.name.clone_into(&mut response.model);
        Ok(response)
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let stream = self
            .inner
            .complete_stream(request)
            .await
            .map_err(|e| self.rename(e))?;
        let (runner, account) = (self.inner.name(), self.name);
        Ok(Box::pin(stream.map(move |chunk| {
            chunk.map_err(|e| renamed(e, runner, account))
        })))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.inner.health_check().await.map_err(|e| self.rename(e))
    }
}

impl PooledTier {
    fn rename(&self, error: RunnerError) -> RunnerError {
        renamed(error, self.inner.name(), self.name)
    }
}

/// `error` with a leading `<runner>: ` rewritten to `<account>: `; unchanged
/// when it does not start with the runner's name, or the account is the
/// runner itself.
fn renamed(mut error: RunnerError, runner: &str, account: &str) -> RunnerError {
    if runner != account {
        if let Some(rest) = error
            .message
            .strip_prefix(runner)
            .and_then(|r| r.strip_prefix(": "))
        {
            error.message = format!("{account}: {rest}");
        }
    }
    error
}

/// Build one tier per credential, in the order given.
///
/// Each runner is `base` with that credential set under the runner type's
/// [`credential_env_key`]. The list is what a
/// [`FallbackProvider`](crate::fallback::FallbackProvider) takes, optionally
/// followed by tiers of other types.
///
/// # Errors
///
/// [`ErrorKind::Config`](crate::types::ErrorKind::Config) when `credentials`
/// is empty, or when the runner type reads no credential from its environment
/// and so cannot be pooled by one.
pub async fn cli_pool(
    runner_type: CliRunnerType,
    base: RunnerConfig,
    credentials: &[String],
) -> Result<Vec<Box<dyn LlmProvider>>, RunnerError> {
    if credentials.is_empty() {
        return Err(RunnerError::config(
            "a credential pool needs at least one credential",
        ));
    }
    let Some(key) = credential_env_key(runner_type) else {
        return Err(RunnerError::config(format!(
            "{} reads no credential from its environment, so it cannot be pooled by one",
            runner_type.binary_name()
        )));
    };

    let mut tiers: Vec<Box<dyn LlmProvider>> = Vec::with_capacity(credentials.len());
    for (position, credential) in credentials.iter().enumerate() {
        let config = base.clone().with_env(key, credential.as_str());
        let runner = create_runner_with_config(runner_type, config).await?;
        tiers.push(Box::new(PooledTier::new(runner, position)));
    }
    Ok(tiers)
}
