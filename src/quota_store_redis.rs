// ABOUTME: Routing state shared through Redis, so every instance spending one account sees one budget
// ABOUTME: Keys expire on their own: a reading after its TTL, a penalty at its reset instant
//
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 dravr.ai

//! A [`QuotaStore`] that several processes share.
//!
//! [`InMemoryQuotaStore`] is correct for one process. Under scale-out, N
//! instances serving one account each keep their own view of a budget they
//! jointly spend: each spends its share before noticing, the step-aside fires
//! unevenly, and a quota refusal is learned once per instance. This store puts
//! the view in Redis, keyed by **account** — the subscription whose budget is
//! shared, never the instance and never the tenant — so a reading or a penalty
//! one instance records steers every other instance's next turn.
//!
//! ## Layout
//!
//! Two keys per backend, both under `{prefix}:{account}:{backend}`:
//!
//! - `…:reading` — the latest snapshots and when they were taken, as JSON,
//!   written with a TTL. An absent key means nobody has read recently, which
//!   is exactly what `last_checked: None` already means to the router: the
//!   next instance to route refreshes it.
//! - `…:penalty` — the instant a backend that refused for quota may serve
//!   again, in epoch milliseconds, with a TTL running to that instant. It
//!   expires itself; no instance has to remember to clear it.
//!
//! No schema, no migration, nothing to sweep.
//!
//! ## When Redis is unreachable
//!
//! Every write also lands in a local [`InMemoryQuotaStore`], and a failed read
//! answers from it. An outage therefore degrades to exactly the per-process
//! behaviour that ships by default — never to "no state at all", which would
//! forget a penalty this instance learned the hard way.

use std::future::Future;
use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::{AsyncCommands, Client, ErrorKind as RedisErrorKind, RedisError, Script};
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tracing::warn;

use crate::quota::QuotaSnapshot;
use crate::quota_store::{BackendState, InMemoryQuotaStore, QuotaStore};
use crate::types::RunnerError;

/// Namespace every key starts with, so the store can share a Redis with
/// anything else without colliding.
pub const KEY_PREFIX: &str = "embacle:quota";

/// How long a reading outlives the write that stored it.
///
/// Longer than the router's default refresh interval on purpose: staleness is
/// judged from `last_checked`, not from the key's presence, so the TTL only
/// bounds how long a reading nobody refreshes keeps steering decisions — and
/// keeps the reset instants the reactive penalty is computed from. A router
/// configured to refresh less often than this re-reads once the key expires,
/// because an absent reading is `last_checked: None`.
pub const READING_TTL: Duration = Duration::from_mins(15);

/// The longest a turn waits on Redis before routing on its own view.
///
/// A Redis in the same network answers in about a millisecond. One that has
/// not answered in half a second is down or reconnecting, and the connection
/// manager's reconnect backoff would otherwise park the athlete's turn behind
/// it for seconds — the store exists to save turns, not to spend them.
pub const OPERATION_TIMEOUT: Duration = Duration::from_millis(500);

/// Records a reading and, in the same atomic step, drops a penalty whose
/// instant the reading has passed — the rule [`InMemoryQuotaStore`] applies.
///
/// Atomic so a penalty another instance sets between the compare and the
/// delete cannot be erased by a reading taken before it.
const RECORD_READING_LUA: &str = r"
redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
local until_ms = redis.call('GET', KEYS[2])
if until_ms and tonumber(until_ms) <= tonumber(ARGV[3]) then
  redis.call('DEL', KEYS[2])
end
return 1
";

/// One window, as stored. Times are epoch milliseconds so the value reads the
/// same from any language that inspects the key.
#[derive(Debug, Serialize, Deserialize)]
struct WireSnapshot {
    key: String,
    label: String,
    percent: f32,
    resets_at_ms: u64,
    observed_at_ms: u64,
}

/// One backend's reading, as stored under `…:reading`.
#[derive(Debug, Serialize, Deserialize)]
struct WireReading {
    observed_at_ms: u64,
    snapshots: Vec<WireSnapshot>,
}

/// Milliseconds since the epoch; an instant before it clamps to zero.
fn to_ms(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn from_ms(ms: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(ms)
}

/// Run one Redis operation, giving up after [`OPERATION_TIMEOUT`].
async fn bounded<T>(op: impl Future<Output = Result<T, RedisError>>) -> Result<T, RedisError> {
    timeout(OPERATION_TIMEOUT, op).await.unwrap_or_else(|_| {
        Err(RedisError::from(io::Error::new(
            io::ErrorKind::TimedOut,
            "Redis did not answer within the quota store's operation timeout",
        )))
    })
}

impl From<&QuotaSnapshot> for WireSnapshot {
    fn from(s: &QuotaSnapshot) -> Self {
        Self {
            key: s.key.clone(),
            label: s.label.clone(),
            percent: s.percent,
            resets_at_ms: to_ms(s.resets_at),
            observed_at_ms: to_ms(s.observed_at),
        }
    }
}

impl From<WireSnapshot> for QuotaSnapshot {
    fn from(w: WireSnapshot) -> Self {
        Self {
            key: w.key,
            label: w.label,
            percent: w.percent,
            resets_at: from_ms(w.resets_at_ms),
            observed_at: from_ms(w.observed_at_ms),
        }
    }
}

/// Routing state in Redis, shared by every instance serving one account.
pub struct RedisQuotaStore {
    conn: ConnectionManager,
    /// `{prefix}:{account}`, the part every key shares.
    namespace: String,
    /// Stable per-backend key segment, by router index. An index past the end
    /// uses its decimal position.
    backend_keys: Vec<String>,
    /// What this process has written, answered from when Redis cannot be read.
    local: InMemoryQuotaStore,
}

impl RedisQuotaStore {
    /// Connect to `redis_url` and share state for `account`.
    ///
    /// `account` names the subscription whose budget is being shared — every
    /// instance spending that budget must pass the same value, and nothing
    /// that spends a different budget may. It is a key segment, so pass an
    /// identifier, never the credential itself.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL does not parse or the first connection
    /// cannot be established. Later outages are absorbed — see the module docs.
    pub async fn connect(redis_url: &str, account: &str) -> Result<Self, RunnerError> {
        let client = Client::open(redis_url)
            .map_err(|e| RunnerError::config(format!("quota store: invalid Redis URL: {e}")))?;
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(OPERATION_TIMEOUT)
            .set_response_timeout(OPERATION_TIMEOUT);
        let conn = ConnectionManager::new_with_config(client, config)
            .await
            .map_err(|e| {
                RunnerError::external_service("redis", format!("quota store: cannot connect: {e}"))
            })?;
        Ok(Self::from_connection(conn, account))
    }

    /// Share state for `account` over a connection the caller already holds,
    /// so a process with one Redis pool does not open a second. Every
    /// operation is still bounded by [`OPERATION_TIMEOUT`], whatever the
    /// connection's own timeouts are.
    #[must_use]
    pub fn from_connection(conn: ConnectionManager, account: &str) -> Self {
        Self {
            conn,
            namespace: format!("{KEY_PREFIX}:{account}"),
            backend_keys: Vec::new(),
            local: InMemoryQuotaStore::new(),
        }
    }

    /// Name each backend by something stabler than its position.
    ///
    /// Pass the providers' names in router order. Without this, keys use the
    /// index, which is correct as long as every instance builds the same
    /// backend list in the same order — true of one binary, and not of a
    /// rollout that reorders backends while old and new instances overlap.
    #[must_use]
    pub fn with_backend_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.backend_keys = keys.into_iter().map(Into::into).collect();
        self
    }

    fn backend_key(&self, index: usize) -> String {
        self.backend_keys
            .get(index)
            .map_or_else(|| index.to_string(), Clone::clone)
    }

    fn reading_key(&self, index: usize) -> String {
        format!("{}:{}:reading", self.namespace, self.backend_key(index))
    }

    fn penalty_key(&self, index: usize) -> String {
        format!("{}:{}:penalty", self.namespace, self.backend_key(index))
    }

    /// Read every backend's two keys in one round trip.
    async fn load_shared(&self, len: usize) -> Result<Vec<BackendState>, RedisError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let keys: Vec<String> = (0..len)
            .flat_map(|i| [self.reading_key(i), self.penalty_key(i)])
            .collect();
        let mut conn = self.conn.clone();
        let values: Vec<Option<String>> = conn.mget(&keys).await?;
        let mut out = Vec::with_capacity(len);
        for (i, pair) in values.chunks(2).enumerate() {
            let mut state = BackendState::default();
            if let Some(raw) = pair.first().and_then(Option::as_ref) {
                match serde_json::from_str::<WireReading>(raw) {
                    Ok(reading) => {
                        state.last_checked = Some(from_ms(reading.observed_at_ms));
                        state.snapshots = reading.snapshots.into_iter().map(Into::into).collect();
                    }
                    // A value this version cannot read is no reading at all,
                    // which the router answers by refreshing it.
                    Err(e) => warn!(
                        key = %self.reading_key(i),
                        error = %e,
                        "quota store: unreadable reading; treating it as absent"
                    ),
                }
            }
            if let Some(raw) = pair.get(1).and_then(Option::as_ref) {
                match raw.parse::<u64>() {
                    Ok(ms) => state.penalty_until = Some(from_ms(ms)),
                    Err(e) => warn!(
                        key = %self.penalty_key(i),
                        error = %e,
                        "quota store: unreadable penalty; treating it as absent"
                    ),
                }
            }
            out.push(state);
        }
        out.resize(len, BackendState::default());
        Ok(out)
    }

    async fn record_shared(
        &self,
        index: usize,
        snapshots: &[QuotaSnapshot],
        observed_at: SystemTime,
    ) -> Result<(), RedisError> {
        let wire = WireReading {
            observed_at_ms: to_ms(observed_at),
            snapshots: snapshots.iter().map(WireSnapshot::from).collect(),
        };
        let body = serde_json::to_string(&wire).map_err(|e| {
            RedisError::from((
                RedisErrorKind::TypeError,
                "quota reading does not serialise",
                e.to_string(),
            ))
        })?;
        let ttl_ms = u64::try_from(READING_TTL.as_millis()).unwrap_or(u64::MAX);
        let mut conn = self.conn.clone();
        Script::new(RECORD_READING_LUA)
            .key(self.reading_key(index))
            .key(self.penalty_key(index))
            .arg(body)
            .arg(ttl_ms)
            .arg(to_ms(observed_at))
            .invoke_async::<i64>(&mut conn)
            .await?;
        Ok(())
    }

    async fn penalty_shared(
        &self,
        index: usize,
        until: Option<SystemTime>,
    ) -> Result<(), RedisError> {
        let key = self.penalty_key(index);
        let mut conn = self.conn.clone();
        // The key's lifetime runs to the penalty's own instant, so it clears
        // itself. One already past — or cleared — is simply removed.
        let remaining = until.and_then(|u| u.duration_since(SystemTime::now()).ok());
        match (until, remaining) {
            (Some(u), Some(left)) if !left.is_zero() => {
                let ttl_ms = u64::try_from(left.as_millis()).unwrap_or(u64::MAX).max(1);
                conn.pset_ex::<_, _, ()>(&key, to_ms(u), ttl_ms).await
            }
            _ => conn.del::<_, ()>(&key).await,
        }
    }
}

#[async_trait]
impl QuotaStore for RedisQuotaStore {
    fn name(&self) -> &str {
        "redis"
    }

    async fn load(&self, len: usize) -> Vec<BackendState> {
        match bounded(self.load_shared(len)).await {
            Ok(state) => state,
            Err(e) => {
                warn!(
                    error = %e,
                    "quota store: Redis unreadable; routing on this process's own view"
                );
                self.local.load(len).await
            }
        }
    }

    async fn record_reading(
        &self,
        index: usize,
        snapshots: Vec<QuotaSnapshot>,
        observed_at: SystemTime,
    ) {
        if let Err(e) = bounded(self.record_shared(index, &snapshots, observed_at)).await {
            warn!(
                error = %e,
                "quota store: reading not shared; other instances will read it themselves"
            );
        }
        self.local
            .record_reading(index, snapshots, observed_at)
            .await;
    }

    async fn set_penalty(&self, index: usize, until: Option<SystemTime>) {
        if let Err(e) = bounded(self.penalty_shared(index, until)).await {
            warn!(
                error = %e,
                "quota store: penalty not shared; other instances will each learn it"
            );
        }
        self.local.set_penalty(index, until).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_survives_the_wire_to_the_millisecond() {
        let snap = QuotaSnapshot {
            key: "weekly_scoped:Fable".to_owned(),
            label: "Weekly (Fable)".to_owned(),
            percent: 87.5,
            resets_at: from_ms(1_800_000_000_123),
            observed_at: from_ms(1_799_999_000_456),
        };
        let wire = WireReading {
            observed_at_ms: 1_799_999_000_456,
            snapshots: vec![WireSnapshot::from(&snap)],
        };
        let raw = serde_json::to_string(&wire).unwrap(); // Safe: test assertion
        let back: WireReading = serde_json::from_str(&raw).unwrap(); // Safe: test assertion
        let restored: Vec<QuotaSnapshot> = back.snapshots.into_iter().map(Into::into).collect();
        assert_eq!(restored, vec![snap]);
    }

    #[test]
    fn an_instant_before_the_epoch_clamps_rather_than_panicking() {
        assert_eq!(to_ms(UNIX_EPOCH - Duration::from_secs(1)), 0);
    }
}
