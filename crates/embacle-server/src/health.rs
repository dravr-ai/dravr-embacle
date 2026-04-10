// ABOUTME: GET /health handler checking provider availability and readiness
// ABOUTME: Returns per-provider status and HTTP 200 if any provider is ready, 503 if none
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::collections::HashMap;
use std::env;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use embacle::discovery::resolve_binary;
use tracing::debug;

use crate::openai_types::HealthResponse;
use crate::runner::ALL_PROVIDERS;
use crate::state::SharedState;

/// Handle GET /health
///
/// Checks which providers have their CLI binary available on the system.
/// For available providers, attempts a health check via the runner.
/// Returns HTTP 200 if at least one provider is ready, 503 if none.
pub async fn handle(State(state): State<SharedState>) -> impl IntoResponse {
    let mut providers = HashMap::new();
    let mut any_ready = false;
    let state_guard = state.read().await;

    for &provider in ALL_PROVIDERS {
        let binary_name = provider.binary_name();
        let env_key = provider.env_override_key();
        let env_override = env::var(env_key).ok();

        if resolve_binary(binary_name, env_override.as_deref()).is_err() {
            providers.insert(provider.to_string(), "not_found".to_owned());
            continue;
        }

        match state_guard.get_runner(provider).await {
            Ok(runner) => match runner.health_check().await {
                Ok(true) => {
                    providers.insert(provider.to_string(), "ready".to_owned());
                    any_ready = true;
                }
                Ok(false) => {
                    providers.insert(provider.to_string(), "not_ready".to_owned());
                }
                Err(e) => {
                    debug!(provider = %provider, error = %e, "Health check failed");
                    providers.insert(provider.to_string(), format!("error: {e}"));
                }
            },
            Err(e) => {
                debug!(provider = %provider, error = %e, "Failed to create runner");
                providers.insert(provider.to_string(), format!("error: {e}"));
            }
        }
    }

    let status_str = if any_ready { "ok" } else { "degraded" };
    let http_status = if any_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let resp = HealthResponse {
        status: status_str,
        providers,
    };

    (http_status, Json(resp))
}
