// ABOUTME: What every HTTP provider shares on the wire: client construction, status-to-error mapping, retry policy
// ABOUTME: One map_http_error so a 401, a 429 and a 400 mean the same kind whichever vendor sent them
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! The transport half of the HTTP providers.
//!
//! [`map_http_error`] is the single site that turns an HTTP status into a
//! [`RunnerError`] kind. It matters for a fallback chain: a 401/403 is the
//! provider's credential, a 408/504 is its latency, a 429 is its quota, and a
//! 400/422 is *the request* — only the first three are the provider's fault.
//! A provider with a vendor-specific exception (Cohere's empty-completion
//! 422) decides that before delegating here.
//!
//! [`HttpRetryConfig`] is the in-place retry policy the providers apply to the
//! initial HTTP request: on 429/502/503 and on connect/timeout transport
//! errors, with exponential backoff and jitter. Once a stream's bytes flow,
//! the request is not retried.

use std::env;
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::StatusCode;
use tokio::time::sleep;
use tracing::warn;

use crate::types::RunnerError;

/// Timeout applied to a client a provider builds for itself.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Build a `reqwest::Client` with the given request timeout.
///
/// # Errors
///
/// Returns [`RunnerError`] with `ErrorKind::Config` when the client builder
/// refuses the configuration (a TLS backend that cannot initialise).
pub fn build_client(timeout: Duration) -> Result<reqwest::Client, RunnerError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| RunnerError::config(format!("HTTP client build failed: {e}")))
}

/// Map a non-success HTTP status and the vendor's parsed error message to a
/// [`RunnerError`].
///
/// - 401 / 403 → `AuthFailure`
/// - 408 / 504 → `Timeout`
/// - 429 → `RateLimit`, carrying the wait the vendor asked for when its
///   message states one (`Please retry in 6.4s`, `try again in 2s`)
/// - 400 / 422 → `InvalidRequest`
/// - anything else → `ExternalService`
///
/// `message` is what the provider extracted from the body — the vendor's
/// own text or, when the body did not parse, `HTTP <status>`.
pub fn map_http_error(provider: &'static str, status: StatusCode, message: &str) -> RunnerError {
    match status.as_u16() {
        401 | 403 => {
            RunnerError::auth_failure(format!("{provider} API authentication failed: {message}"))
        }
        408 | 504 => RunnerError::timeout(format!("{provider}: HTTP {status}: {message}")),
        429 => RunnerError::rate_limit(provider, rate_limit_message(message)),
        400 | 422 => {
            RunnerError::invalid_request(format!("{provider} API validation error: {message}"))
        }
        _ => RunnerError::external_service(provider, format!("HTTP {status}: {message}")),
    }
}

/// Map a transport failure from `reqwest` to a [`RunnerError`].
///
/// The error's `Display` includes the request URL, which for Gemini carries
/// the API key, so the URL is stripped before the text reaches a message.
pub fn map_send_error(provider: &'static str, error: reqwest::Error) -> RunnerError {
    let is_timeout = error.is_timeout();
    let is_connect = error.is_connect();
    let error = error.without_url();
    if is_timeout {
        RunnerError::timeout(format!("{provider}: request timed out: {error}"))
    } else if is_connect {
        RunnerError::external_service(provider, format!("Connection failed: {error}"))
    } else {
        RunnerError::external_service(provider, error.to_string())
    }
}

/// The wait a vendor's rate-limit message asks for, in whole seconds
/// (rounded up), when it states one.
///
/// Recognises the two phrasings in use — Gemini's `Please retry in 6.406s`
/// and the OpenAI-style `Please try again in 2.5s` / `try again in 20ms` —
/// case-insensitively. A value in milliseconds rounds up to one second.
#[must_use]
pub fn retry_after_seconds(message: &str) -> Option<u64> {
    let lower = message.to_lowercase();
    let start = ["retry in ", "try again in "]
        .iter()
        .find_map(|prefix| lower.find(prefix).map(|pos| pos + prefix.len()))?;
    let rest = &lower[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    let value: f64 = rest[..end].parse().ok()?;
    let unit = rest[end..].trim_start();
    let seconds = if unit.starts_with("ms") {
        value / 1000.0
    } else {
        value
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(seconds.ceil().max(1.0) as u64)
}

/// The message a 429 carries: the wait when the vendor states one, the
/// vendor's own text otherwise.
fn rate_limit_message(api_message: &str) -> String {
    match retry_after_seconds(api_message) {
        Some(seconds) => format!("Rate limit reached. Please try again in {seconds} seconds."),
        None if api_message.trim().is_empty() => {
            "Rate limit reached. Please wait a moment and try again.".to_owned()
        }
        None => format!("Rate limit reached: {api_message}"),
    }
}

/// Retry policy for a provider's initial HTTP request.
///
/// Streaming retries cover only the request that opens the stream. Once
/// bytes flow, the stream is not retried (the caller may already have
/// consumed partial output).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRetryConfig {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_retries: u32,
    /// Initial delay before first retry (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum delay cap for exponential backoff (milliseconds)
    pub max_delay_ms: u64,
}

impl Default for HttpRetryConfig {
    /// 3 retries, 500 ms initial, 5 s cap.
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 500,
            max_delay_ms: 5000,
        }
    }
}

impl HttpRetryConfig {
    /// Read `<PREFIX>_MAX_RETRIES`, `<PREFIX>_INITIAL_RETRY_DELAY_MS` and
    /// `<PREFIX>_MAX_RETRY_DELAY_MS`, each falling back to the default when
    /// unset or unparseable.
    #[must_use]
    pub fn from_env(prefix: &str) -> Self {
        let defaults = Self::default();
        let read = |suffix: &str, fallback: u64| {
            env::var(format!("{prefix}_{suffix}"))
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(fallback)
        };
        #[allow(clippy::cast_possible_truncation)]
        let max_retries = read("MAX_RETRIES", u64::from(defaults.max_retries)) as u32;
        Self {
            max_retries,
            initial_delay_ms: read("INITIAL_RETRY_DELAY_MS", defaults.initial_delay_ms),
            max_delay_ms: read("MAX_RETRY_DELAY_MS", defaults.max_delay_ms),
        }
    }

    /// Calculate exponential backoff delay with jitter for a given attempt
    ///
    /// `delay = min(initial_ms * 2^attempt, max_ms) + jitter(0..100ms)`
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay = self
            .initial_delay_ms
            .saturating_mul(1_u64 << attempt.min(63));
        let capped_delay = base_delay.min(self.max_delay_ms);
        // Small jitter (0-99ms) to avoid thundering herd
        let jitter = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::from(d.subsec_millis()))
            % 100;
        Duration::from_millis(capped_delay + jitter)
    }

    /// Sleep before retry number `attempt` (1-based), logging the wait.
    pub async fn wait_before(&self, provider: &'static str, attempt: u32, what: &'static str) {
        let delay = self.delay_for_attempt(attempt.saturating_sub(1));
        warn!(
            provider,
            attempt,
            delay_ms = delay.as_millis(),
            "retrying {what} after transient failure"
        );
        sleep(delay).await;
    }
}

/// One attempt's failure, with the retry decision the attempt made about
/// itself: a 429/502/503, a connect or timeout transport error, or an
/// unreadable body is worth another try; a 401 or a 400 is not.
#[derive(Debug)]
pub struct AttemptError {
    /// The error the caller sees if this attempt was the last
    pub error: RunnerError,
    /// Whether the retry policy may try again
    pub retryable: bool,
}

impl AttemptError {
    /// An error no retry can fix
    #[must_use]
    pub const fn permanent(error: RunnerError) -> Self {
        Self {
            error,
            retryable: false,
        }
    }

    /// An error worth another attempt
    #[must_use]
    pub const fn transient(error: RunnerError) -> Self {
        Self {
            error,
            retryable: true,
        }
    }

    /// A transport failure: retryable when it was a connect or timeout error
    #[must_use]
    pub fn from_send(provider: &'static str, error: reqwest::Error) -> Self {
        let retryable = is_retryable_request_error(&error);
        Self {
            error: map_send_error(provider, error),
            retryable,
        }
    }

    /// A non-success status the provider already mapped: retryable on 429/502/503
    #[must_use]
    pub const fn from_status(status: StatusCode, error: RunnerError) -> Self {
        Self {
            error,
            retryable: is_retryable_status(status.as_u16()),
        }
    }
}

/// Run `attempt` up to `retry.max_retries + 1` times, sleeping with backoff
/// between attempts while the failure says it is retryable.
///
/// `what` names the request in the retry log line.
pub async fn with_retries<T, F, Fut>(
    retry: &HttpRetryConfig,
    provider: &'static str,
    what: &'static str,
    mut attempt: F,
) -> Result<T, RunnerError>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T, AttemptError>>,
{
    let mut last_error: Option<RunnerError> = None;
    for number in 0..=retry.max_retries {
        if number > 0 {
            retry.wait_before(provider, number, what).await;
        }
        match attempt(number).await {
            Ok(value) => return Ok(value),
            Err(failed) if failed.retryable && number < retry.max_retries => {
                warn!(provider, attempt = number, error = %failed.error, "{what} failed, will retry");
                last_error = Some(failed.error);
            }
            Err(failed) => return Err(failed.error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        RunnerError::external_service(provider, format!("{what} failed after retries"))
    }))
}

/// Check if an HTTP error status code is retryable
///
/// Retryable errors are transient conditions that may resolve on retry:
/// - 429 Too Many Requests (rate limiting)
/// - 503 Service Unavailable (temporary overload)
/// - 502 Bad Gateway (upstream issues)
#[must_use]
pub const fn is_retryable_status(status: u16) -> bool {
    matches!(status, 429 | 502 | 503)
}

/// Check if a request error is retryable (connection/timeout errors)
#[must_use]
pub fn is_retryable_request_error(error: &reqwest::Error) -> bool {
    error.is_connect() || error.is_timeout()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ErrorKind;

    #[test]
    fn map_http_error_auth() {
        let err = map_http_error("acme", StatusCode::UNAUTHORIZED, "bad key");
        assert_eq!(err.kind, ErrorKind::AuthFailure);
        assert!(err.message.contains("bad key"));
        let err = map_http_error("acme", StatusCode::FORBIDDEN, "no");
        assert_eq!(err.kind, ErrorKind::AuthFailure);
    }

    #[test]
    fn map_http_error_timeout() {
        assert_eq!(
            map_http_error("acme", StatusCode::GATEWAY_TIMEOUT, "timeout").kind,
            ErrorKind::Timeout
        );
        assert_eq!(
            map_http_error("acme", StatusCode::REQUEST_TIMEOUT, "timeout").kind,
            ErrorKind::Timeout
        );
    }

    #[test]
    fn map_http_error_server_carries_the_api_message() {
        let err = map_http_error("acme", StatusCode::INTERNAL_SERVER_ERROR, "overloaded");
        assert_eq!(err.kind, ErrorKind::ExternalService);
        assert!(err.message.contains("overloaded"));
        assert!(err.message.contains("500"));
    }

    /// A 400 or 422 is the request's fault, not the provider's: a fallback
    /// chain must not spend a second tier on it.
    #[test]
    fn map_http_error_validation_is_invalid_request() {
        let err = map_http_error(
            "acme",
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
        );
        assert_eq!(err.kind, ErrorKind::InvalidRequest);
        assert!(!err.kind.is_provider_fault());
        assert!(err.message.contains("messages must not be empty"));
        assert_eq!(
            map_http_error("acme", StatusCode::UNPROCESSABLE_ENTITY, "x").kind,
            ErrorKind::InvalidRequest
        );
    }

    #[test]
    fn map_http_error_rate_limit_carries_the_wait() {
        let err = map_http_error(
            "acme",
            StatusCode::TOO_MANY_REQUESTS,
            "Quota exceeded. Please retry in 6.406453963s.",
        );
        assert_eq!(err.kind, ErrorKind::RateLimit);
        // The vendor's 429 is the vendor failing to serve: a chain moves on,
        // and nobody sleeps on the wait it names.
        assert!(err.kind.is_provider_fault());
        assert!(!err.kind.is_transient());
        assert!(
            err.message.contains("try again in 7 seconds"),
            "{}",
            err.message
        );
        assert!(err.message.starts_with("acme:"));
    }

    #[test]
    fn map_http_error_rate_limit_without_a_wait_carries_the_message() {
        let err = map_http_error("acme", StatusCode::TOO_MANY_REQUESTS, "too many requests");
        assert_eq!(err.kind, ErrorKind::RateLimit);
        assert!(err.message.contains("too many requests"));
        let bare = map_http_error("acme", StatusCode::TOO_MANY_REQUESTS, "");
        assert!(bare.message.contains("wait a moment"));
    }

    #[test]
    fn retry_after_seconds_recognises_both_phrasings() {
        assert_eq!(
            retry_after_seconds("Please retry in 6.406453963s."),
            Some(7)
        );
        assert_eq!(
            retry_after_seconds(
                "Rate limit reached for model x on tokens per minute (TPM): Limit 1, Used 2, Requested 3. Please try again in 2.5s."
            ),
            Some(3)
        );
        assert_eq!(retry_after_seconds("Try again in 20ms"), Some(1));
        assert_eq!(retry_after_seconds("try again in 45 seconds"), Some(45));
        assert_eq!(retry_after_seconds("too many requests"), None);
    }

    #[test]
    fn retry_config_delay_curve() {
        let config = HttpRetryConfig::default();
        let delay0 = config.delay_for_attempt(0);
        assert!(delay0.as_millis() >= 500 && delay0.as_millis() < 700);
        let delay1 = config.delay_for_attempt(1);
        assert!(delay1.as_millis() >= 1000 && delay1.as_millis() < 1200);
        let delay3 = config.delay_for_attempt(3);
        assert!(delay3.as_millis() >= 4000 && delay3.as_millis() <= 5100);
    }

    #[test]
    fn retryable_status_codes() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
        assert!(!is_retryable_status(500));
    }
}
