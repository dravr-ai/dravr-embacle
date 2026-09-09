// ABOUTME: The two quota checkers that reach the network, and the cooldown that keeps them safe
// ABOUTME: One reads a real budget; the other is an honest proxy and says so
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

//! HTTP quota checkers.
//!
//! Two implementations of [`LimitChecker`], with very different standing:
//!
//! - [`AnthropicUsageChecker`] reads the **real** budget from the endpoint
//!   Claude Code itself uses.
//! - [`GithubHeadroomChecker`] is a **proxy**. GitHub exposes no Copilot
//!   premium-request quota — `/copilot_internal/user` returns entitlement only
//!   (plan, feature flags) and no usage figure, and `/copilot_internal/v2/token`
//!   is not reachable with a personal access token. What it does expose is core
//!   API rate-limit headroom, which Copilot's session-token exchange draws on.
//!   Low headroom therefore predicts the *next* Copilot call failing to
//!   authenticate. That is a useful signal and it is not the quota; the type
//!   says so in its own name and its doc, and nothing here should be read as a
//!   premium-request budget.
//!
//! ## Limitations
//!
//! Readings describe an account, not a process. Several processes sharing an
//! account share its budget, and each reads the same figure without knowing
//! what the others have spent.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::Value;

use crate::quota::{parse_anthropic_usage, Cooldown, LimitChecker, QuotaSnapshot};
use crate::types::RunnerError;

/// Where the Anthropic OAuth usage endpoint lives.
const ANTHROPIC_USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";

/// The beta opt-in the usage endpoint requires.
const ANTHROPIC_OAUTH_BETA: &str = "oauth-2025-04-20";

/// How long a reading stays usable once it can no longer be refreshed.
///
/// Past this, a stale figure is withheld rather than served as current: an
/// hours-old percentage is worse than no percentage, because a caller acts on
/// it as though it were now.
const CACHE_MAX_AGE: Duration = Duration::from_hours(6);

/// Upper bound on a cooldown, as a sanity rail rather than a clamp.
///
/// A stated `Retry-After` is honoured in full below this. The endpoint answers
/// a probe made *during* a penalty with a **longer** penalty — one account's
/// stated wait grew from 1746s to 2708s over a day of well-meaning retries —
/// so truncating a long wait into an early probe is precisely how a caller
/// talks itself into a deeper hole.
const COOLDOWN_MAX: Duration = Duration::from_hours(6);

/// Non-cryptographic identity for a credential, safe to log.
///
/// Equality is the only property needed — "is this the same token that earned
/// the penalty?" — so this is a hash for comparison, not a security primitive,
/// and it must never be treated as one. The token itself is never logged.
fn fingerprint(secret: &str) -> String {
    let mut h = DefaultHasher::new();
    secret.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Cached last-good reading plus whatever penalty is currently in force.
#[derive(Default)]
struct CheckerState {
    last_good: Option<(SystemTime, Vec<QuotaSnapshot>)>,
    cooldown: Option<Cooldown>,
}

/// Reads the real Claude subscription budget.
///
/// Uses the same endpoint and beta header Claude Code uses for its own usage
/// display, so the figures match what a human sees in the CLI.
pub struct AnthropicUsageChecker {
    client: reqwest::Client,
    token: String,
    url: String,
    state: Mutex<CheckerState>,
}

impl AnthropicUsageChecker {
    /// Build a checker for an OAuth access token.
    ///
    /// The token is the one Claude Code holds. In a container it arrives as
    /// `CLAUDE_CODE_OAUTH_TOKEN`; on a developer machine it lives in the
    /// keychain, and resolving it from there is the caller's business — this
    /// type takes the token it is given and does not go looking.
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            token: token.into(),
            url: ANTHROPIC_USAGE_URL.to_owned(),
            state: Mutex::new(CheckerState::default()),
        }
    }

    /// Point the checker at a different base URL, for tests.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Read the token from the environment, if it is there.
    ///
    /// Deliberately only the environment variable: it is the one source that
    /// works identically in a container and on a laptop.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        env::var("CLAUDE_CODE_OAUTH_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty())
            .map(Self::new)
    }
}

/// How long to stay out of the water after a refusal.
///
/// A `Retry-After` the server stated wins, uncapped up to the sanity rail.
/// `retry-after: 0` is a real value this endpoint sends, and it must not be
/// read as "probe immediately" — the floor keeps a zero from becoming a
/// hot loop.
fn cooldown_for(retry_after: Option<u64>) -> Duration {
    let stated = retry_after.map_or(Duration::from_mins(1), Duration::from_secs);
    stated.clamp(Duration::from_mins(1), COOLDOWN_MAX)
}

#[async_trait]
impl LimitChecker for AnthropicUsageChecker {
    fn name(&self) -> &str {
        "anthropic-usage"
    }

    async fn check(&self) -> Result<Vec<QuotaSnapshot>, RunnerError> {
        let fp = fingerprint(&self.token);
        let now = SystemTime::now();

        // A live penalty is a hard refusal to probe, not a hint.
        if let Ok(state) = self.state.lock() {
            if let Some(cd) = &state.cooldown {
                if cd.blocks(now, Some(&fp)) {
                    return serve_cache(&state, now).ok_or_else(|| {
                        RunnerError::rate_limit(
                            "anthropic-usage",
                            "usage endpoint is in cooldown and no fresh-enough reading is cached",
                        )
                    });
                }
            }
        }

        let response = self
            .client
            .get(&self.url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("anthropic-beta", ANTHROPIC_OAUTH_BETA)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| RunnerError::external_service("anthropic-usage", e.to_string()))?;

        let status = response.status();
        if status.as_u16() == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            if let Ok(mut state) = self.state.lock() {
                state.cooldown = Some(Cooldown::starting_at(
                    now,
                    cooldown_for(retry_after),
                    Some(fp),
                ));
            }
            // Not necessarily a rate limit at all: this endpoint answers an
            // EXPIRED token with 429 rather than 401. The cooldown carries the
            // token's fingerprint so refreshing it voids the penalty.
            return Err(RunnerError::rate_limit(
                "anthropic-usage",
                "usage endpoint refused (429) — this can also mean the OAuth token has expired",
            ));
        }
        if !status.is_success() {
            return Err(RunnerError::external_service(
                "anthropic-usage",
                format!("usage endpoint returned HTTP {status}"),
            ));
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| RunnerError::external_service("anthropic-usage", e.to_string()))?;
        let snapshots = parse_anthropic_usage(&body, now)?;

        if let Ok(mut state) = self.state.lock() {
            state.cooldown = None;
            state.last_good = Some((now, snapshots.clone()));
        }
        Ok(snapshots)
    }
}

/// Hand back the cached reading when it is still young enough to act on.
fn serve_cache(state: &CheckerState, now: SystemTime) -> Option<Vec<QuotaSnapshot>> {
    let (taken_at, snapshots) = state.last_good.as_ref()?;
    let age = now.duration_since(*taken_at).ok()?;
    (age <= CACHE_MAX_AGE).then(|| snapshots.clone())
}

/// GitHub core rate-limit headroom, used as a **proxy** for Copilot capacity.
///
/// This is not a Copilot quota and must not be documented or logged as one.
/// GitHub exposes no premium-request budget: `/copilot_internal/user` answers
/// with entitlement only — plan, feature flags, seat assignment — and carries
/// no usage figure at all. What it does expose is the core 5000/hour pool,
/// which Copilot's session-token exchange draws on, so exhausted headroom
/// predicts the next Copilot call failing to authenticate.
///
/// It therefore reports a single window keyed `github_core`. Treat it as
/// "Copilot is about to stop working", never as "Copilot has used N% of its
/// allowance".
pub struct GithubHeadroomChecker {
    client: reqwest::Client,
    token: String,
    url: String,
}

impl GithubHeadroomChecker {
    /// Build a checker for a GitHub token.
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            token: token.into(),
            url: "https://api.github.com/rate_limit".to_owned(),
        }
    }

    /// Point the checker at a different base URL, for tests.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Read the token from `GITHUB_TOKEN` or `GH_TOKEN`, if either is set.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        env::var("GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .ok()
            .filter(|t| !t.trim().is_empty())
            .map(Self::new)
    }
}

/// Map a `/rate_limit` payload to the single headroom window.
///
/// Split out so the mapping is testable without a network.
///
/// # Errors
///
/// Returns an error when `resources.core` is missing or malformed — a shape
/// change, which must not read as full headroom.
pub fn parse_github_rate_limit(
    body: &Value,
    observed_at: SystemTime,
) -> Result<Vec<QuotaSnapshot>, RunnerError> {
    let core = body
        .get("resources")
        .and_then(|r| r.get("core"))
        .ok_or_else(|| {
            RunnerError::external_service(
                "github-headroom",
                "rate_limit response carried no resources.core",
            )
        })?;

    let limit = core.get("limit").and_then(Value::as_u64).unwrap_or(0);
    let used = core.get("used").and_then(Value::as_u64).unwrap_or(0);
    let reset = core.get("reset").and_then(Value::as_u64).ok_or_else(|| {
        RunnerError::external_service("github-headroom", "rate_limit core carried no reset")
    })?;

    // A zero limit would divide to nothing; report it as fully consumed rather
    // than as infinite headroom, because a missing limit is not permission.
    #[allow(clippy::cast_precision_loss)]
    let percent = if limit == 0 {
        100.0
    } else {
        (used as f32 / limit as f32) * 100.0
    };

    Ok(vec![QuotaSnapshot {
        key: "github_core".to_owned(),
        label: "GitHub core rate limit (Copilot proxy)".to_owned(),
        percent,
        resets_at: UNIX_EPOCH + Duration::from_secs(reset),
        observed_at,
    }])
}

#[async_trait]
impl LimitChecker for GithubHeadroomChecker {
    fn name(&self) -> &str {
        "github-headroom"
    }

    async fn check(&self) -> Result<Vec<QuotaSnapshot>, RunnerError> {
        let response = self
            .client
            .get(&self.url)
            .header("Authorization", format!("token {}", self.token))
            .header("User-Agent", "embacle-quota")
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .map_err(|e| RunnerError::external_service("github-headroom", e.to_string()))?;

        if !response.status().is_success() {
            return Err(RunnerError::external_service(
                "github-headroom",
                format!("rate_limit returned HTTP {}", response.status()),
            ));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|e| RunnerError::external_service("github-headroom", e.to_string()))?;
        parse_github_rate_limit(&body, SystemTime::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(epoch: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(epoch)
    }

    #[test]
    fn headroom_reports_the_share_consumed() {
        let body = serde_json::json!({"resources":{"core":{
            "limit":5000,"used":1250,"remaining":3750,"reset":1_788_955_723_u64}}});
        let out = parse_github_rate_limit(&body, at(0)).unwrap(); // Safe: test assertion
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key, "github_core");
        assert!((out[0].percent - 25.0).abs() < f32::EPSILON);
        assert_eq!(out[0].resets_at, at(1_788_955_723));
    }

    #[test]
    fn the_proxy_says_so_in_its_own_label() {
        // Anyone reading a log or a dashboard must be able to tell this is not
        // a Copilot premium-request budget, because no such figure exists.
        let body = serde_json::json!({"resources":{"core":{
            "limit":5000,"used":0,"reset":1_u64}}});
        let out = parse_github_rate_limit(&body, at(0)).unwrap(); // Safe: test assertion
        assert!(
            out[0].label.contains("proxy"),
            "label was {:?}",
            out[0].label
        );
    }

    #[test]
    fn a_zero_limit_reads_as_exhausted_not_as_infinite() {
        // A missing limit is not permission.
        let body = serde_json::json!({"resources":{"core":{
            "limit":0,"used":0,"reset":1_u64}}});
        let out = parse_github_rate_limit(&body, at(0)).unwrap(); // Safe: test assertion
        assert!((out[0].percent - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_shape_change_is_an_error_not_full_headroom() {
        assert!(parse_github_rate_limit(&serde_json::json!({}), at(0)).is_err());
        let no_reset = serde_json::json!({"resources":{"core":{"limit":5000,"used":1}}});
        assert!(parse_github_rate_limit(&no_reset, at(0)).is_err());
    }

    #[test]
    fn a_stated_retry_after_is_honoured_in_full() {
        // Truncating a long wait into an early probe is how a caller talks
        // itself into a deeper penalty: the endpoint answers a probe made
        // during the penalty with a longer one.
        assert_eq!(cooldown_for(Some(2_708)), Duration::from_secs(2_708));
    }

    #[test]
    fn a_zero_retry_after_still_waits() {
        // `retry-after: 0` is a real value this endpoint sends. Taking it
        // literally would spin.
        assert_eq!(cooldown_for(Some(0)), Duration::from_mins(1));
    }

    #[test]
    fn an_absurd_retry_after_hits_the_sanity_rail() {
        assert_eq!(cooldown_for(Some(u64::MAX)), COOLDOWN_MAX);
    }

    #[test]
    fn a_fingerprint_is_stable_and_hides_the_secret() {
        let fp = fingerprint("sk-secret-value");
        assert_eq!(fp, fingerprint("sk-secret-value"));
        assert_ne!(fp, fingerprint("sk-other-value"));
        assert!(
            !fp.contains("secret"),
            "the token must not survive into a loggable string"
        );
    }
}
