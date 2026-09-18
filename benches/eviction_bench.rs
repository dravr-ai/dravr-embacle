// ABOUTME: Criterion benchmarks comparing Vec::remove(0) vs VecDeque::pop_front() for FIFO eviction
// ABOUTME: Proves O(1) VecDeque::pop_front() outperforms O(n) Vec::remove(0) at all queue sizes
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Benchmarks proving that `VecDeque::pop_front()` is faster than
//! `Vec::remove(0)` for FIFO eviction order tracking.
//!
//! The cache stores `insertion_order` as a FIFO queue of hash keys. When the
//! cache is full, the oldest entry is evicted from the front. `Vec::remove(0)`
//! shifts all remaining elements left (O(n)), while `VecDeque::pop_front()`
//! adjusts a head pointer (O(1)).

#![allow(
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    missing_docs
)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::VecDeque;

/// Simulate the cache eviction pattern: fill to capacity, then evict + insert
/// for `eviction_count` cycles.
///
/// This mirrors the exact code path in `src/cache.rs`.
fn bench_eviction_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_eviction");
    let eviction_count: u64 = 100;

    for capacity in [16u64, 64, 128, 256] {
        group.throughput(Throughput::Elements(eviction_count));

        // -- Vec::remove(0) (the "before" pattern) --
        group.bench_with_input(
            BenchmarkId::new("Vec_remove_0", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    // Pre-fill
                    let mut order: Vec<u64> = (0..cap).collect();

                    // Evict oldest + push new, `eviction_count` times
                    for i in 0..eviction_count {
                        order.remove(0); // O(n) — shifts all elements
                        order.push(black_box(cap + i));
                    }

                    black_box(&order);
                });
            },
        );

        // -- VecDeque::pop_front() (the "after" pattern) --
        group.bench_with_input(
            BenchmarkId::new("VecDeque_pop_front", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    // Pre-fill
                    let mut order: VecDeque<u64> = (0..cap).collect();

                    // Evict oldest + push new, `eviction_count` times
                    for i in 0..eviction_count {
                        order.pop_front(); // O(1) — adjusts head pointer
                        order.push_back(black_box(cap + i));
                    }

                    black_box(&order);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark `retain()` — both `Vec` and `VecDeque` have O(n) retain, so this
/// confirms no regression.
fn bench_retain(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_retain");

    for capacity in [64u64, 256] {
        group.throughput(Throughput::Elements(capacity));

        group.bench_with_input(
            BenchmarkId::new("Vec_retain", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    let mut order: Vec<u64> = (0..cap).collect();
                    // Remove every other element (simulates selective eviction)
                    order.retain(|x| black_box(x % 2 == 0));
                    black_box(&order);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("VecDeque_retain", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    let mut order: VecDeque<u64> = (0..cap).collect();
                    order.retain(|x| black_box(x % 2 == 0));
                    black_box(&order);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_eviction_vec, bench_retain);
criterion_main!(benches);
