// ABOUTME: Response caching decorator for LlmProvider with TTL and capacity limits
// ABOUTME: Caches deterministic (temperature=0) non-streaming completions by request hash
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Response Caching
//!
//! [`CacheProvider`] wraps an inner `Box<dyn LlmProvider>` and caches
//! responses for identical requests. Cache keys are computed from the
//! serialized request parameters (messages, model, temperature, `max_tokens`).
//!
//! ## Behavior
//!
//! - Only `complete()` results are cached; `complete_stream()` always delegates.
//! - Requests with `temperature > Some(0.0)` bypass the cache by default
//!   (configurable via `cache_nonzero_temperature`).
//! - Entries expire after `ttl` and are evicted on access.
//! - When at capacity, the oldest entry is evicted to make room.

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::debug;

use crate::types::{
    ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, RunnerError,
};

/// Configuration for the response cache
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached entries
    pub max_entries: usize,
    /// Time-to-live for cached entries
    pub ttl: Duration,
    /// Whether to cache responses for requests with non-zero temperature
    pub cache_nonzero_temperature: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            ttl: Duration::from_secs(300),
            cache_nonzero_temperature: false,
        }
    }
}

/// Cache usage statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries evicted due to capacity or TTL
    pub evictions: u64,
    /// Current number of entries in the cache
    pub size: usize,
}

/// A cached response with its insertion time
#[derive(Debug, Clone)]
struct CacheEntry {
    response: ChatResponse,
    inserted_at: Instant,
}

/// Internal cache state
#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<u64, CacheEntry>,
    insertion_order: VecDeque<u64>,
    stats: CacheStats,
}

/// Caching decorator for any `LlmProvider`.
///
/// # Usage
///
/// ```rust,no_run
/// # use embacle::cache::{CacheProvider, CacheConfig};
/// # use embacle::types::LlmProvider;
/// # fn example(provider: Box<dyn LlmProvider>) {
/// let cached = CacheProvider::new(provider, CacheConfig::default());
/// // Identical requests will return cached responses
/// # }
/// ```
pub struct CacheProvider {
    inner: Box<dyn LlmProvider>,
    config: CacheConfig,
    state: Arc<Mutex<CacheState>>,
}

impl CacheProvider {
    /// Wrap a provider with response caching
    pub fn new(inner: Box<dyn LlmProvider>, config: CacheConfig) -> Self {
        Self {
            inner,
            config,
            state: Arc::new(Mutex::new(CacheState::default())),
        }
    }

    /// Return current cache statistics
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the internal mutex is poisoned.
    pub fn cache_stats(&self) -> Result<CacheStats, RunnerError> {
        let state = self
            .state
            .lock()
            .map_err(|_| RunnerError::internal("cache lock poisoned"))?;
        let mut snapshot = state.stats.clone();
        snapshot.size = state.entries.len();
        Ok(snapshot)
    }

    /// Compute a deterministic cache key from request parameters
    fn cache_key(request: &ChatRequest) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash the serialized form of the relevant fields
        serde_json::to_string(&(
            &request.messages,
            &request.model,
            &request.temperature,
            &request.max_tokens,
        ))
        .unwrap_or_default()
        .hash(&mut hasher);
        hasher.finish()
    }

    /// Whether caching should be bypassed for this request
    fn should_bypass(&self, request: &ChatRequest) -> bool {
        if self.config.cache_nonzero_temperature {
            return false;
        }
        matches!(request.temperature, Some(t) if t > 0.0)
    }
}

#[async_trait]
impl LlmProvider for CacheProvider {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn display_name(&self) -> &str {
        self.inner.display_name()
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
        if self.should_bypass(request) {
            return self.inner.complete(request).await;
        }

        let key = Self::cache_key(request);

        // Check cache
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| RunnerError::internal("cache lock poisoned"))?;
            let cached = state.entries.get(&key).and_then(|entry| {
                if entry.inserted_at.elapsed() < self.config.ttl {
                    Some(entry.response.clone())
                } else {
                    None
                }
            });
            if let Some(response) = cached {
                state.stats.hits += 1;
                debug!(key, "cache hit");
                return Ok(response);
            }
            // Remove expired entry if it exists
            if state.entries.remove(&key).is_some() {
                state.insertion_order.retain(|k| *k != key);
                state.stats.evictions += 1;
            }
            state.stats.misses += 1;
        }

        // Cache miss — delegate to inner provider
        let response = self.inner.complete(request).await?;

        // Store in cache
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| RunnerError::internal("cache lock poisoned"))?;

            // Evict oldest if at capacity — O(1) with VecDeque::pop_front
            while state.entries.len() >= self.config.max_entries {
                if let Some(oldest_key) = state.insertion_order.pop_front() {
                    state.entries.remove(&oldest_key);
                    state.stats.evictions += 1;
                } else {
                    break;
                }
            }

            state.entries.insert(
                key,
                CacheEntry {
                    response: response.clone(),
                    inserted_at: Instant::now(),
                },
            );
            state.insertion_order.push_back(key);
        }

        Ok(response)
    }

    /// Streaming responses are not cached; always delegates to the inner provider
    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        self.inner.complete_stream(request).await
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        self.inner.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
        RunnerError,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TestProvider {
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
    }

    impl TestProvider {
        fn new(responses: Vec<Result<ChatResponse, RunnerError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            "test"
        }
        fn display_name(&self) -> &str {
            "Test Provider"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::text_only()
        }
        fn default_model(&self) -> &'static str {
            "test-model"
        }
        fn available_models(&self) -> &[String] {
            &[]
        }
        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock"); // Safe: test assertion
            if responses.is_empty() {
                Ok(ChatResponse {
                    content: "default".to_owned(),
                    model: "test-model".to_owned(),
                    usage: None,
                    finish_reason: Some("stop".to_owned()),
                    warnings: None,
                    tool_calls: None,
                })
            } else {
                responses.remove(0)
            }
        }
        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("streaming not supported in test"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
        }
    }

    fn make_response(content: &str) -> ChatResponse {
        ChatResponse {
            content: content.to_owned(),
            model: "test-model".to_owned(),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        }
    }

    #[tokio::test]
    async fn cache_hit() {
        let provider = TestProvider::new(vec![Ok(make_response("cached"))]);
        let cached = CacheProvider::new(Box::new(provider), CacheConfig::default());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let r1 = cached.complete(&request).await.expect("first call"); // Safe: test assertion
        let r2 = cached.complete(&request).await.expect("second call"); // Safe: test assertion

        assert_eq!(r1.content, "cached");
        assert_eq!(r2.content, "cached");

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn cache_miss_different_request() {
        let provider = TestProvider::new(vec![
            Ok(make_response("first")),
            Ok(make_response("second")),
        ]);
        let cached = CacheProvider::new(Box::new(provider), CacheConfig::default());

        let r1 = cached
            .complete(&ChatRequest::new(vec![ChatMessage::user("hello")]))
            .await
            .expect("first"); // Safe: test assertion
        let r2 = cached
            .complete(&ChatRequest::new(vec![ChatMessage::user("goodbye")]))
            .await
            .expect("second"); // Safe: test assertion

        assert_eq!(r1.content, "first");
        assert_eq!(r2.content, "second");

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn bypass_nonzero_temp() {
        let provider = TestProvider::new(vec![Ok(make_response("r1")), Ok(make_response("r2"))]);
        let cached = CacheProvider::new(Box::new(provider), CacheConfig::default());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]).with_temperature(0.7);

        let r1 = cached.complete(&request).await.expect("first"); // Safe: test assertion
        let r2 = cached.complete(&request).await.expect("second"); // Safe: test assertion

        // Both calls go through (cache bypassed)
        assert_eq!(r1.content, "r1");
        assert_eq!(r2.content, "r2");
    }

    #[tokio::test]
    async fn cache_with_temp_configured() {
        let provider = TestProvider::new(vec![Ok(make_response("cached"))]);
        let config = CacheConfig {
            cache_nonzero_temperature: true,
            ..CacheConfig::default()
        };
        let cached = CacheProvider::new(Box::new(provider), config);
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]).with_temperature(0.7);

        let r1 = cached.complete(&request).await.expect("first"); // Safe: test assertion
        let r2 = cached.complete(&request).await.expect("second"); // Safe: test assertion

        assert_eq!(r1.content, "cached");
        assert_eq!(r2.content, "cached");

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.hits, 1);
    }

    #[tokio::test]
    async fn ttl_expiration() {
        let provider =
            TestProvider::new(vec![Ok(make_response("old")), Ok(make_response("fresh"))]);
        let config = CacheConfig {
            ttl: Duration::from_millis(10),
            ..CacheConfig::default()
        };
        let cached = CacheProvider::new(Box::new(provider), config);
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        let r1 = cached.complete(&request).await.expect("first"); // Safe: test assertion
        assert_eq!(r1.content, "old");

        tokio::time::sleep(Duration::from_millis(20)).await;

        let r2 = cached.complete(&request).await.expect("after expiry"); // Safe: test assertion
        assert_eq!(r2.content, "fresh");

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn eviction_at_capacity() {
        let provider = TestProvider::new(vec![
            Ok(make_response("a")),
            Ok(make_response("b")),
            Ok(make_response("c")),
        ]);
        let config = CacheConfig {
            max_entries: 2,
            ..CacheConfig::default()
        };
        let cached = CacheProvider::new(Box::new(provider), config);

        cached
            .complete(&ChatRequest::new(vec![ChatMessage::user("1")]))
            .await
            .expect("a"); // Safe: test assertion
        cached
            .complete(&ChatRequest::new(vec![ChatMessage::user("2")]))
            .await
            .expect("b"); // Safe: test assertion
        cached
            .complete(&ChatRequest::new(vec![ChatMessage::user("3")]))
            .await
            .expect("c"); // Safe: test assertion

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.size, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn stats_tracking() {
        let provider = TestProvider::new(vec![Ok(make_response("r1")), Ok(make_response("r2"))]);
        let cached = CacheProvider::new(Box::new(provider), CacheConfig::default());

        let req1 = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let req2 = ChatRequest::new(vec![ChatMessage::user("world")]);

        cached.complete(&req1).await.expect("miss"); // Safe: test assertion
        cached.complete(&req1).await.expect("hit"); // Safe: test assertion
        cached.complete(&req2).await.expect("miss"); // Safe: test assertion

        let stats = cached.cache_stats().unwrap(); // Safe: test assertion
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.size, 2);
    }

    #[tokio::test]
    async fn streaming_bypasses() {
        let provider = TestProvider::new(vec![]);
        let cached = CacheProvider::new(Box::new(provider), CacheConfig::default());
        let request = ChatRequest::new(vec![ChatMessage::user("hi")]);

        // complete_stream always delegates (and errors in our test provider)
        let result = cached.complete_stream(&request).await;
        assert!(result.is_err());
    }

    #[test]
    fn key_determinism() {
        let req1 = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let req2 = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let req3 = ChatRequest::new(vec![ChatMessage::user("different")]);

        assert_eq!(
            CacheProvider::cache_key(&req1),
            CacheProvider::cache_key(&req2)
        );
        assert_ne!(
            CacheProvider::cache_key(&req1),
            CacheProvider::cache_key(&req3)
        );
    }
}
