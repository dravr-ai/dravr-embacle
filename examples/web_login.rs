// ABOUTME: One-time interactive login for the browser-driven Claude.ai web provider
// ABOUTME: Opens a headed Chrome against the persistent profile; sign in once, cookies persist
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::error::Error;
use std::process;
use std::time::Duration;

use embacle::WebUiRunner;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("embacle::web_ui=info,dravr_browser=info")
            }),
        )
        .with_target(false)
        .init();

    let runner = WebUiRunner::from_env()?;
    eprintln!(
        "Opening a browser window for login. Sign in to Claude.ai; \
         this waits up to 5 minutes for an authenticated session..."
    );
    let timeout = Duration::from_secs(300);
    if runner.interactive_login(timeout).await? {
        println!("login: success (profile saved)");
        Ok(())
    } else {
        eprintln!("login: timed out before an authenticated session was detected");
        process::exit(1);
    }
}
