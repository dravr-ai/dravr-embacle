// ABOUTME: What "how much of this provider's budget is left" means, and who can answer it
// ABOUTME: Pure shapes and policy here; the checkers that perform I/O live behind features
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Provider quota.
//!
//! A subscription-backed provider does not fail like a service does. It works
//! perfectly until a window is exhausted, refuses everything until that window
//! resets, then works perfectly again. Routing around that needs one thing the
//! crate did not have: a reading of how much budget is left, taken *before* a
//! turn is spent rather than inferred from the corpse of a failed one.
//!
//! ## The one rule
//!
//! **Never compute a reset time.** Windows are rolling and anchored per
//! account, not aligned to a calendar. Two Claude accounts on one machine were
//! observed resetting their weekly windows three days apart. Every reset
//! instant here is one the provider stated; nothing in this module derives one
//! from a wall clock.
//!
//! ## Limitations
//!
//! A [`QuotaSnapshot`] describes an account, not a process. Two processes
//! sharing an account share its budget and each will read the same figure
//! without knowing about the other's spending.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

use crate::types::RunnerError;

/// One window's worth of quota, as the provider reported it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuotaSnapshot {
    /// Stable identifier for the window, e.g. `session`, `weekly_all`,
    /// `weekly_scoped:Fable`. Distinct windows for one provider have distinct
    /// keys, so a caller can track them independently.
    pub key: String,
    /// Human-readable name for logs.
    pub label: String,
    /// Percentage of the window consumed, 0-100.
    ///
    /// Deliberately not a 0-1 fraction: the Anthropic OAuth usage endpoint
    /// reports 0-100 while its response *headers* report 0-1, and normalising
    /// on the wrong one silently turns a 77% window into a 0.77% one.
    pub percent: f32,
    /// When the provider said this window resets. Never computed locally.
    pub resets_at: SystemTime,
    /// When this reading was taken, so a caller can judge staleness.
    pub observed_at: SystemTime,
}

impl QuotaSnapshot {
    /// Whether this window has consumed at least `threshold` percent.
    #[must_use]
    pub fn at_or_above(&self, threshold: f32) -> bool {
        self.percent >= threshold
    }

    /// Whether the stated reset instant has passed.
    #[must_use]
    pub fn has_reset(&self, now: SystemTime) -> bool {
        now >= self.resets_at
    }
}

/// Something that can report how much of a provider's budget remains.
///
/// Async, unlike [`Guardrail`](crate::guardrail::Guardrail) which is the
/// crate's other pluggable-policy trait: a quota reading costs I/O — an HTTP
/// call, a keychain read, a file — where a guardrail is an in-memory check.
///
/// An implementation returns every window it knows about. Returning an empty
/// vec means "I have no reading", which a caller must not treat as "there is
/// no limit".
#[async_trait]
pub trait LimitChecker: Send + Sync {
    /// Identifier for logs and for pairing a checker with its backend.
    fn name(&self) -> &str;

    /// Read the current windows.
    ///
    /// # Errors
    ///
    /// Returns an error when the reading cannot be taken. A caller should keep
    /// using the provider on an error rather than routing away from it: an
    /// unreadable quota is not an exhausted one.
    async fn check(&self) -> Result<Vec<QuotaSnapshot>, RunnerError>;
}

/// How long a failed probe stays out of the water, and why.
///
/// The Anthropic usage endpoint answers a probe made *during* a penalty with a
/// longer penalty — one account's stated wait grew 1746s to 2708s over a day
/// of well-meaning retries. So a cooldown here is a hard refusal to probe, and
/// a stated `Retry-After` is honoured in full rather than clamped to something
/// convenient.
#[derive(Debug, Clone)]
pub struct Cooldown {
    /// When probing may resume.
    pub until: SystemTime,
    /// Identity of the credential that earned the penalty.
    ///
    /// That endpoint answers an *expired* token with 429 rather than 401, so a
    /// "rate limit" is sometimes a stale credential. Without this, refreshing
    /// the token would leave the checker sulking for the full penalty over a
    /// problem that no longer exists.
    pub credential_fingerprint: Option<String>,
}

impl Cooldown {
    /// Build a cooldown that expires `after` from `now`.
    #[must_use]
    pub fn starting_at(now: SystemTime, after: Duration, fingerprint: Option<String>) -> Self {
        Self {
            until: now + after,
            credential_fingerprint: fingerprint,
        }
    }

    /// Whether probing is currently forbidden.
    ///
    /// A cooldown earned by a different credential is void: the caller has
    /// changed something, so the penalty no longer describes reality.
    #[must_use]
    pub fn blocks(&self, now: SystemTime, current_fingerprint: Option<&str>) -> bool {
        if self.credential_fingerprint.as_deref() != current_fingerprint {
            return false;
        }
        now < self.until
    }
}

/// Convert an ISO-8601 instant to [`SystemTime`].
///
/// Accepts the shapes the usage endpoint emits:
/// `2026-07-16T17:19:59.672659+00:00`, `...Z`, and offsets without a colon.
/// Hand-rolled rather than pulling a date crate into a library the whole
/// satellite fleet depends on; the civil-days arithmetic is exact and pinned
/// by tests against known epochs.
///
/// # Errors
///
/// Returns an error when the string is not a recognisable instant. The caller
/// must treat that as "no reading", never as "resets now" — a parse failure
/// that defaulted to the epoch would read as a window that reset in 1970 and
/// would send every turn back to an exhausted provider.
pub fn iso8601_to_system_time(raw: &str) -> Result<SystemTime, RunnerError> {
    let bad = || RunnerError::internal(format!("unparseable reset instant {raw:?}"));

    let (date, rest) = raw.split_once('T').ok_or_else(bad)?;
    let mut d = date.split('-');
    let year: i64 = d.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let month: i64 = d.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let day: i64 = d.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;

    // Split the offset off the time. Search from the end so the '-' of a
    // negative offset is not confused with anything earlier.
    let (time, offset_secs) = if let Some(t) = rest.strip_suffix('Z') {
        (t, 0_i64)
    } else if let Some(idx) = rest.rfind(['+', '-']) {
        let (t, off) = rest.split_at(idx);
        (t, parse_offset(off).ok_or_else(bad)?)
    } else {
        (rest, 0)
    };

    let time = time.split('.').next().unwrap_or(time);
    let mut t = time.split(':');
    let hour: i64 = t.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let minute: i64 = t.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
    let second: i64 = t.next().unwrap_or("0").parse().map_err(|_| bad())?;

    // Range-check before the arithmetic. days_from_civil is happy to convert
    // a 13th month or a 45th day into a real-looking instant, and a reset time
    // that is merely plausible is worse than an error: it would be believed.
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=60).contains(&second)
    {
        return Err(bad());
    }

    let days = days_from_civil(year, month, day);
    let epoch = days * 86_400 + hour * 3_600 + minute * 60 + second - offset_secs;
    if epoch < 0 {
        return Err(bad());
    }
    let secs = u64::try_from(epoch).map_err(|_| bad())?;
    Ok(UNIX_EPOCH + Duration::from_secs(secs))
}

/// Parse `+HH:MM`, `-HHMM` or `+HH` into seconds east of UTC.
fn parse_offset(raw: &str) -> Option<i64> {
    let (sign, rest) = raw.split_at(1);
    let sign: i64 = match sign {
        "+" => 1,
        "-" => -1,
        _ => return None,
    };
    let digits: String = rest.chars().filter(char::is_ascii_digit).collect();
    let (h, m) = match digits.len() {
        2 => (digits.parse::<i64>().ok()?, 0),
        4 => (
            digits.get(0..2)?.parse::<i64>().ok()?,
            digits.get(2..4)?.parse::<i64>().ok()?,
        ),
        _ => return None,
    };
    Some(sign * (h * 3_600 + m * 60))
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's `days_from_civil`).
///
/// Exact for the whole proleptic Gregorian calendar, leap years included.
const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Turn an Anthropic OAuth usage payload into snapshots.
///
/// The wire shape, from `GET /api/oauth/usage`:
///
/// ```json
/// {"limits":[{"kind":"session","percent":2,
///             "resets_at":"2026-07-16T17:19:59.672659+00:00",
///             "scope":{"model":{"display_name":"Fable"}}}]}
/// ```
///
/// `kind` is one of `session`, `weekly_all`, `weekly_scoped`; an unrecognised
/// one is passed through as its own key rather than dropped, so a window this
/// crate has never heard of still counts against the budget instead of
/// silently reading as headroom.
///
/// A limit whose `resets_at` cannot be read is **skipped**, not defaulted.
/// Without a reset instant there is no way to know when to come back, and a
/// fabricated one would send every turn at an exhausted provider.
///
/// Split from the HTTP call so the mapping is testable without a network.
///
/// # Errors
///
/// Returns an error when `limits` is absent or is not an array — that is a
/// shape change, not an empty budget, and must not read as "no limits".
pub fn parse_anthropic_usage(
    body: &serde_json::Value,
    observed_at: SystemTime,
) -> Result<Vec<QuotaSnapshot>, RunnerError> {
    let limits = body
        .get("limits")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            RunnerError::external_service(
                "anthropic-usage",
                "response carried no `limits` array; treating as a shape change, not as no limits",
            )
        })?;

    let mut out = Vec::with_capacity(limits.len());
    for limit in limits {
        let kind = limit
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let model = limit
            .get("scope")
            .and_then(|s| s.get("model"))
            .and_then(|m| m.get("display_name"))
            .and_then(serde_json::Value::as_str);

        let (key, label) = match (kind, model) {
            ("session", _) => ("session".to_owned(), "Session (5h)".to_owned()),
            ("weekly_all", _) => ("weekly_all".to_owned(), "Weekly (all models)".to_owned()),
            ("weekly_scoped", Some(m)) => (format!("weekly_scoped:{m}"), format!("Weekly ({m})")),
            ("weekly_scoped", None) => ("weekly_scoped".to_owned(), "Weekly (scoped)".to_owned()),
            (other, _) => (other.to_owned(), other.to_owned()),
        };

        let Some(resets_raw) = limit.get("resets_at").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Ok(resets_at) = iso8601_to_system_time(resets_raw) else {
            continue;
        };

        #[allow(clippy::cast_possible_truncation)]
        let percent = limit
            .get("percent")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32;

        out.push(QuotaSnapshot {
            key,
            label,
            percent,
            resets_at,
            observed_at,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(epoch: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(epoch)
    }

    #[test]
    fn the_unix_epoch_itself_round_trips() {
        let epoch = iso8601_to_system_time("1970-01-01T00:00:00Z").unwrap(); // Safe: test assertion
        assert_eq!(epoch, at(0));
    }

    #[test]
    fn a_known_instant_matches_its_epoch() {
        // 2026-07-16T17:19:59Z == 1784222399, checked independently.
        assert_eq!(
            iso8601_to_system_time("2026-07-16T17:19:59.672659+00:00").unwrap(), // Safe: test assertion
            at(1_784_222_399)
        );
    }

    #[test]
    fn a_leap_day_is_not_off_by_one() {
        // 2024-02-29T00:00:00Z == 1709164800. A naive 365-day year lands a day early.
        assert_eq!(
            iso8601_to_system_time("2024-02-29T00:00:00Z").unwrap(), // Safe: test assertion
            at(1_709_164_800)
        );
    }

    #[test]
    fn a_non_utc_offset_shifts_the_instant() {
        let utc = iso8601_to_system_time("2026-07-16T17:19:59Z").unwrap(); // Safe: test assertion
        let plus_two = iso8601_to_system_time("2026-07-16T19:19:59+02:00").unwrap(); // Safe: test assertion
        let minus_five = iso8601_to_system_time("2026-07-16T12:19:59-05:00").unwrap(); // Safe: test assertion
        assert_eq!(
            plus_two, utc,
            "+02:00 names the same instant two hours later on the clock"
        );
        assert_eq!(minus_five, utc);
    }

    #[test]
    fn a_colonless_offset_parses() {
        assert_eq!(
            iso8601_to_system_time("2026-07-16T19:19:59+0200").unwrap(), // Safe: test assertion
            iso8601_to_system_time("2026-07-16T17:19:59Z").unwrap()      // Safe: test assertion
        );
    }

    #[test]
    fn garbage_is_an_error_not_the_epoch() {
        // The failure that matters: defaulting to UNIX_EPOCH would read as a
        // window that reset in 1970 and send every turn at an exhausted provider.
        for raw in ["", "not a date", "2026-07-16", "T12:00:00Z", "2026-07-16T"] {
            assert!(
                iso8601_to_system_time(raw).is_err(),
                "{raw:?} is not an instant and must not become one"
            );
        }
    }

    #[test]
    fn an_out_of_range_field_is_rejected_rather_than_normalised() {
        // The civil-days arithmetic will happily turn a 13th month into a real
        // instant. A reset time that is merely plausible is worse than an
        // error, because it gets believed.
        for raw in [
            "2026-13-01T00:00:00Z",
            "2026-00-01T00:00:00Z",
            "2026-07-45T00:00:00Z",
            "2026-07-16T99:00:00Z",
            "2026-07-16T00:99:00Z",
        ] {
            assert!(
                iso8601_to_system_time(raw).is_err(),
                "{raw:?} must be rejected, not normalised into a believable instant"
            );
        }
    }

    #[test]
    fn a_leap_second_is_tolerated() {
        // Second 60 is legal in ISO-8601 and the endpoint is not ours to fix.
        assert!(iso8601_to_system_time("2016-12-31T23:59:60Z").is_ok());
    }

    fn snap(percent: f32, resets_at: u64) -> QuotaSnapshot {
        QuotaSnapshot {
            key: "session".to_owned(),
            label: "Session (5h)".to_owned(),
            percent,
            resets_at: at(resets_at),
            observed_at: at(0),
        }
    }

    #[test]
    fn a_threshold_is_inclusive_at_the_boundary() {
        assert!(
            snap(80.0, 100).at_or_above(80.0),
            "exactly at the threshold counts as reached"
        );
        assert!(!snap(79.9, 100).at_or_above(80.0));
    }

    #[test]
    fn a_window_resets_only_once_its_stated_instant_passes() {
        let s = snap(100.0, 1_000);
        assert!(!s.has_reset(at(999)));
        assert!(
            s.has_reset(at(1_000)),
            "the stated instant itself counts as reset"
        );
        assert!(s.has_reset(at(1_001)));
    }

    #[test]
    fn a_cooldown_blocks_until_it_expires() {
        let c = Cooldown::starting_at(at(100), Duration::from_mins(1), Some("fp".to_owned()));
        assert!(c.blocks(at(159), Some("fp")));
        assert!(
            !c.blocks(at(160), Some("fp")),
            "the expiry instant releases it"
        );
    }

    /// A real payload shape, trimmed to the fields the parser reads.
    fn usage_body() -> serde_json::Value {
        serde_json::json!({"limits":[
            {"kind":"session","percent":2,
             "resets_at":"2026-07-16T03:09:59+00:00"},
            {"kind":"weekly_all","percent":77,
             "resets_at":"2026-07-18T15:59:59+00:00"},
            {"kind":"weekly_scoped","percent":100,
             "resets_at":"2026-07-18T15:59:59+00:00",
             "scope":{"model":{"display_name":"Fable"}}}
        ]})
    }

    #[test]
    fn every_window_becomes_its_own_snapshot() {
        let out = parse_anthropic_usage(&usage_body(), at(0)).unwrap(); // Safe: test assertion
        let keys: Vec<&str> = out.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, ["session", "weekly_all", "weekly_scoped:Fable"]);
        assert!((out[1].percent - 77.0).abs() < f32::EPSILON);
        assert_eq!(out[2].label, "Weekly (Fable)");
    }

    #[test]
    fn a_scoped_window_saturates_independently_of_the_overall_one() {
        // Observed live: weekly_all at 77% while the Fable bucket was at 100%.
        // Collapsing them onto one key would hide an exhausted model behind
        // an overall figure that still looks healthy.
        let out = parse_anthropic_usage(&usage_body(), at(0)).unwrap(); // Safe: test assertion
        let all = out.iter().find(|s| s.key == "weekly_all").unwrap(); // Safe: test assertion
        let fable = out.iter().find(|s| s.key == "weekly_scoped:Fable").unwrap(); // Safe: test assertion
        assert!(all.percent < 100.0 && fable.percent >= 100.0);
    }

    #[test]
    fn an_unknown_kind_still_counts_against_the_budget() {
        // A window this crate has never heard of must not read as headroom.
        let body = serde_json::json!({"limits":[
            {"kind":"monthly_experimental","percent":95,"resets_at":"2026-07-18T15:59:59Z"}
        ]});
        let out = parse_anthropic_usage(&body, at(0)).unwrap(); // Safe: test assertion
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key, "monthly_experimental");
        assert!((out[0].percent - 95.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_window_without_a_readable_reset_is_skipped_not_defaulted() {
        // Without a reset instant there is no way to know when to come back.
        // A fabricated one would send every turn at an exhausted provider.
        let body = serde_json::json!({"limits":[
            {"kind":"session","percent":99},
            {"kind":"weekly_all","percent":50,"resets_at":"not-a-date"},
            {"kind":"weekly_scoped","percent":10,"resets_at":"2026-07-18T15:59:59Z",
             "scope":{"model":{"display_name":"Sonnet"}}}
        ]});
        let out = parse_anthropic_usage(&body, at(0)).unwrap(); // Safe: test assertion
        assert_eq!(out.len(), 1, "only the window with a usable reset survives");
        assert_eq!(out[0].key, "weekly_scoped:Sonnet");
    }

    #[test]
    fn a_missing_limits_array_is_an_error_not_an_empty_budget() {
        // A shape change must not read as "this account has no limits", which
        // would park every turn on a provider that may be exhausted.
        assert!(parse_anthropic_usage(&serde_json::json!({}), at(0)).is_err());
        assert!(parse_anthropic_usage(&serde_json::json!({"limits": 3}), at(0)).is_err());
    }

    #[test]
    fn percent_is_read_on_the_0_to_100_scale() {
        // The endpoint reports 0-100 while its response HEADERS report 0-1.
        // Normalising on the wrong one turns a 77% window into 0.77%.
        let out = parse_anthropic_usage(&usage_body(), at(0)).unwrap(); // Safe: test assertion
        assert!(
            out.iter().any(|s| s.percent > 1.0),
            "77 must stay 77, not become 0.77"
        );
    }

    #[test]
    fn a_cooldown_earned_by_another_credential_is_void() {
        // The endpoint answers an EXPIRED token with 429, so a penalty can be
        // about a stale credential. Once it is refreshed the penalty describes
        // nothing, and holding it would sulk through a fixed problem.
        let c = Cooldown::starting_at(at(100), Duration::from_hours(1), Some("old".to_owned()));
        assert!(c.blocks(at(200), Some("old")));
        assert!(
            !c.blocks(at(200), Some("new")),
            "a refreshed credential voids the penalty"
        );
        assert!(!c.blocks(at(200), None));
    }
}
