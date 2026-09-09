// ABOUTME: Live probe of the quota checkers against the real endpoints
// ABOUTME: Prints windows and reset times; never prints a credential
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: CLAUDE_CODE_OAUTH_TOKEN=... cargo run --example quota_probe --features quota-http

use std::time::{SystemTime, UNIX_EPOCH};

use embacle::quota::LimitChecker;
use embacle::quota_http::{AnthropicUsageChecker, GithubHeadroomChecker};

fn in_words(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let delta = i64::try_from(secs).unwrap_or(0) - i64::try_from(now).unwrap_or(0);
    if delta <= 0 {
        return "already reset".to_owned();
    }
    format!("in {}h{:02}m", delta / 3600, (delta % 3600) / 60)
}

async fn report(checker: &dyn LimitChecker) {
    println!("\n=== {} ===", checker.name());
    match checker.check().await {
        Ok(snapshots) if snapshots.is_empty() => println!("  (no windows reported)"),
        Ok(snapshots) => {
            for s in snapshots {
                println!(
                    "  {:<34} {:>6.1}%   resets {}",
                    s.label,
                    s.percent,
                    in_words(s.resets_at)
                );
            }
        }
        Err(e) => println!("  check failed: {e}"),
    }
}

#[tokio::main]
async fn main() {
    match AnthropicUsageChecker::from_env() {
        Some(c) => report(&c).await,
        None => println!("\n=== anthropic-usage ===\n  CLAUDE_CODE_OAUTH_TOKEN not set"),
    }
    match GithubHeadroomChecker::from_env() {
        Some(c) => report(&c).await,
        None => println!("\n=== github-headroom ===\n  GITHUB_TOKEN / GH_TOKEN not set"),
    }
}
