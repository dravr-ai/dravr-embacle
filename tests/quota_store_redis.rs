// ABOUTME: Proves RedisQuotaStore shares router state across instances, against a real redis-server
// ABOUTME: Each test starts its own throwaway redis-server on a free port and stops it on drop
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! `RedisQuotaStore` against a real Redis.
//!
//! Each test stands for two Cloud Run instances: two stores, two connections,
//! one account. What one records, the other must route on. A test account is
//! a fresh UUID, so tests never read each other's keys even on a shared
//! server.
//!
//! With `quota-redis` enabled these tests need a `redis-server` binary on
//! `PATH`; each test starts its own, so a test can stop it to prove the outage
//! path. A missing binary is a failure, not a skip — a store whose tests
//! silently stop running is how a shared budget quietly becomes N private ones
//! again.

#![cfg(feature = "quota-redis")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::str_to_string
)]

use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use embacle::quota::{LimitChecker, QuotaSnapshot};
use embacle::quota_store::{BackendState, QuotaStore};
use embacle::quota_store_redis::RedisQuotaStore;
use embacle::router::{Backend, PreferInOrder, RouterProvider};
use embacle::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
    TokenUsage,
};
use tokio::time::{sleep, timeout};
use uuid::Uuid;

/// A private Redis for one test, stopped when the test ends.
struct TestRedis {
    url: String,
    server: Option<Child>,
}

impl Drop for TestRedis {
    fn drop(&mut self) {
        if let Some(child) = self.server.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TestRedis {
    async fn start() -> Self {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .expect("a free local port")
            .port();
        let server = Command::new("redis-server")
            .args([
                "--port",
                &port.to_string(),
                "--bind",
                "127.0.0.1",
                "--save",
                "",
                "--appendonly",
                "no",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("quota-redis tests start a redis-server: install it and put it on PATH");
        let redis = Self {
            url: format!("redis://127.0.0.1:{port}"),
            server: Some(server),
        };
        for _ in 0..100 {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return redis;
            }
            sleep(Duration::from_millis(50)).await;
        }
        panic!("redis-server did not start listening on port {port}");
    }

    /// An instance's store: its own connection, the shared account.
    async fn store(&self, account: &str) -> RedisQuotaStore {
        RedisQuotaStore::connect(&self.url, account)
            .await
            .expect("connect to the test Redis")
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.server.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn account() -> String {
    format!("test-{}", Uuid::new_v4())
}

fn snap(percent: f32, resets_at: SystemTime) -> QuotaSnapshot {
    QuotaSnapshot {
        key: "weekly_all".to_owned(),
        label: "Weekly".to_owned(),
        percent,
        resets_at,
        observed_at: SystemTime::now(),
    }
}

/// Redis stores milliseconds; compare instants at that resolution.
fn same_ms(a: Option<SystemTime>, b: Option<SystemTime>) -> bool {
    let ms = |t: SystemTime| {
        t.duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    };
    a.map(ms) == b.map(ms)
}

/// Answers with its own name, scripted failures first, and counts every call.
struct TestProvider {
    provider_name: &'static str,
    models: Vec<String>,
    scripted: Mutex<Vec<RunnerError>>,
    calls: Arc<AtomicU32>,
}

impl TestProvider {
    fn ok(name: &'static str) -> Self {
        Self::counted(name, Arc::new(AtomicU32::new(0)))
    }

    fn counted(name: &'static str, calls: Arc<AtomicU32>) -> Self {
        Self {
            provider_name: name,
            models: vec![format!("{name}-model")],
            scripted: Mutex::new(Vec::new()),
            calls,
        }
    }

    fn refusing(name: &'static str) -> Self {
        let p = Self::ok(name);
        p.scripted
            .lock()
            .unwrap()
            .push(RunnerError::rate_limit(name, "usage limit reached"));
        p
    }
}

#[async_trait]
impl LlmProvider for TestProvider {
    fn name(&self) -> &'static str {
        self.provider_name
    }
    fn display_name(&self) -> &str {
        self.provider_name
    }
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING
    }
    fn default_model(&self) -> &str {
        &self.models[0]
    }
    fn available_models(&self) -> &[String] {
        &self.models
    }
    async fn complete(&self, _r: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let scripted = self.scripted.lock().unwrap().pop();
        if let Some(e) = scripted {
            return Err(e);
        }
        Ok(ChatResponse {
            content: format!("answered by {}", self.provider_name),
            model: self.models[0].clone(),
            usage: Some(TokenUsage::new(1, 1, 2)),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }
    async fn complete_stream(&self, _r: &ChatRequest) -> Result<ChatStream, RunnerError> {
        Err(RunnerError::internal("not used in these tests"))
    }
    async fn health_check(&self) -> Result<bool, RunnerError> {
        Ok(true)
    }
}

/// Reports a fixed window.
struct FixedChecker(QuotaSnapshot);

#[async_trait]
impl LimitChecker for FixedChecker {
    fn name(&self) -> &str {
        "fixed"
    }
    async fn check(&self) -> Result<Vec<QuotaSnapshot>, RunnerError> {
        Ok(vec![self.0.clone()])
    }
}

fn request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hello")])
}

#[tokio::test]
async fn a_refusal_one_instance_learns_steers_every_other_instance() {
    let redis = TestRedis::start().await;
    let account = account();
    let resets_at = SystemTime::now() + Duration::from_hours(2);

    // Instance A: its reading says there is room, the provider disagrees.
    let first = RouterProvider::new(
        vec![
            Backend::metered(
                Box::new(TestProvider::refusing("claude-code")),
                Box::new(FixedChecker(snap(5.0, resets_at))),
            ),
            Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
        ],
        Box::new(PreferInOrder),
    )
    .unwrap()
    .with_store(Box::new(redis.store(&account).await));
    let r = first.complete(&request()).await.unwrap();
    assert_eq!(r.content, "answered by copilot_headless");

    // Instance B: its own connection, never refused, no checker of its own.
    let untouched = Arc::new(AtomicU32::new(0));
    let second = RouterProvider::new(
        vec![
            Backend::unmetered(Box::new(TestProvider::counted(
                "claude-code",
                Arc::clone(&untouched),
            ))),
            Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
        ],
        Box::new(PreferInOrder),
    )
    .unwrap()
    .with_store(Box::new(redis.store(&account).await));
    let r = second.complete(&request()).await.unwrap();
    assert_eq!(r.content, "answered by copilot_headless");
    assert_eq!(
        untouched.load(Ordering::Relaxed),
        0,
        "a penalty is learned once per account, not once per instance"
    );
}

#[tokio::test]
async fn a_reading_one_instance_takes_is_what_another_routes_on() {
    let redis = TestRedis::start().await;
    let account = account();
    let resets_at = SystemTime::now() + Duration::from_hours(2);

    // A reads a full window; B has no checker and never read anything.
    let primary = Backend::metered(
        Box::new(TestProvider::ok("claude-code")),
        Box::new(FixedChecker(snap(95.0, resets_at))),
    );
    let first = RouterProvider::new(
        vec![
            primary,
            Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
        ],
        Box::new(PreferInOrder),
    )
    .unwrap()
    .with_store(Box::new(redis.store(&account).await));
    let r = first.complete(&request()).await.unwrap();
    assert_eq!(r.content, "answered by copilot_headless");

    let untouched = Arc::new(AtomicU32::new(0));
    let second = RouterProvider::new(
        vec![
            Backend::unmetered(Box::new(TestProvider::counted(
                "claude-code",
                Arc::clone(&untouched),
            ))),
            Backend::unmetered(Box::new(TestProvider::ok("copilot_headless"))),
        ],
        Box::new(PreferInOrder),
    )
    .unwrap()
    .with_store(Box::new(redis.store(&account).await));
    let r = second.complete(&request()).await.unwrap();
    assert_eq!(
        r.content, "answered by copilot_headless",
        "the step-aside fires on every instance at once, not on whichever read last"
    );
    assert_eq!(untouched.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn another_account_shares_nothing() {
    let redis = TestRedis::start().await;
    let ours = redis.store(&account()).await;
    let theirs = redis.store(&account()).await;
    ours.set_penalty(0, Some(SystemTime::now() + Duration::from_hours(1)))
        .await;
    ours.record_reading(0, vec![snap(50.0, SystemTime::now())], SystemTime::now())
        .await;
    assert_eq!(
        theirs.load(1).await,
        vec![BackendState::default()],
        "the key is the account; a different subscription's budget is not ours"
    );
}

#[tokio::test]
async fn every_backend_is_answered_for_even_when_nothing_is_stored() {
    let redis = TestRedis::start().await;
    let store = redis.store(&account()).await;
    assert_eq!(store.load(3).await, vec![BackendState::default(); 3]);
    assert!(store.load(0).await.is_empty());
}

#[tokio::test]
async fn a_reading_round_trips_between_instances() {
    let redis = TestRedis::start().await;
    let account = account();
    let writer = redis.store(&account).await;
    let reader = redis.store(&account).await;
    let observed = SystemTime::now();
    let window = snap(42.5, observed + Duration::from_hours(3));

    writer
        .record_reading(1, vec![window.clone()], observed)
        .await;
    let loaded = reader.load(2).await;
    assert_eq!(loaded[0], BackendState::default(), "untouched slot");
    assert_eq!(loaded[1].snapshots.len(), 1);
    assert_eq!(loaded[1].snapshots[0].key, window.key);
    assert!((loaded[1].snapshots[0].percent - 42.5).abs() < f32::EPSILON);
    assert!(same_ms(
        Some(loaded[1].snapshots[0].resets_at),
        Some(window.resets_at)
    ));
    assert!(same_ms(loaded[1].last_checked, Some(observed)));
}

#[tokio::test]
async fn a_reading_clears_a_penalty_whose_instant_has_passed_and_keeps_a_live_one() {
    let redis = TestRedis::start().await;
    let account = account();
    let a = redis.store(&account).await;
    let b = redis.store(&account).await;
    let now = SystemTime::now();
    let until = now + Duration::from_hours(1);

    a.set_penalty(0, Some(until)).await;
    assert!(same_ms(b.load(1).await[0].penalty_until, Some(until)));

    // A reading taken before the reset is not permission to serve.
    b.record_reading(0, vec![snap(1.0, until)], now).await;
    assert!(same_ms(a.load(1).await[0].penalty_until, Some(until)));

    // One taken at or after it is, on every instance.
    b.record_reading(0, vec![snap(1.0, until)], until).await;
    assert_eq!(a.load(1).await[0].penalty_until, None);
}

#[tokio::test]
async fn a_cleared_or_past_penalty_is_gone_for_everyone() {
    let redis = TestRedis::start().await;
    let account = account();
    let a = redis.store(&account).await;
    let b = redis.store(&account).await;

    a.set_penalty(0, Some(SystemTime::now() + Duration::from_hours(1)))
        .await;
    a.set_penalty(0, None).await;
    assert_eq!(b.load(1).await[0].penalty_until, None, "cleared");

    a.set_penalty(0, Some(SystemTime::now() - Duration::from_secs(1)))
        .await;
    assert_eq!(b.load(1).await[0].penalty_until, None, "already over");
}

#[tokio::test]
async fn a_penalty_expires_on_its_own_at_its_instant() {
    let redis = TestRedis::start().await;
    let account = account();
    let a = redis.store(&account).await;
    let b = redis.store(&account).await;

    a.set_penalty(0, Some(SystemTime::now() + Duration::from_millis(300)))
        .await;
    assert!(b.load(1).await[0].penalty_until.is_some());
    sleep(Duration::from_millis(600)).await;
    assert_eq!(
        b.load(1).await[0].penalty_until,
        None,
        "no instance has to remember to clear it"
    );
}

#[tokio::test]
async fn backend_keys_survive_instances_that_order_backends_differently() {
    let redis = TestRedis::start().await;
    let account = account();
    let old = redis
        .store(&account)
        .await
        .with_backend_keys(["claude_code", "copilot_headless"]);
    let reordered = redis
        .store(&account)
        .await
        .with_backend_keys(["copilot_headless", "claude_code"]);
    let until = SystemTime::now() + Duration::from_hours(1);

    old.set_penalty(0, Some(until)).await;
    let seen = reordered.load(2).await;
    assert_eq!(seen[0].penalty_until, None, "copilot was never penalised");
    assert!(same_ms(seen[1].penalty_until, Some(until)), "claude was");
}

#[tokio::test]
async fn an_outage_degrades_to_this_instances_own_view() {
    let mut redis = TestRedis::start().await;
    let store = redis.store(&account()).await;
    let until = SystemTime::now() + Duration::from_hours(1);
    store.set_penalty(0, Some(until)).await;

    redis.kill();

    let later = SystemTime::now() + Duration::from_hours(2);
    let state = timeout(Duration::from_secs(3), async {
        store.set_penalty(1, Some(later)).await;
        store.load(2).await
    })
    .await
    .expect("an unreachable Redis must not hang the turn");
    assert!(
        same_ms(state[0].penalty_until, Some(until)),
        "what this instance learned before the outage still holds"
    );
    assert!(
        same_ms(state[1].penalty_until, Some(later)),
        "and so does what it learns during it"
    );
}
