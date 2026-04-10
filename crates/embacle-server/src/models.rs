// ABOUTME: GET /v1/models handler listing available providers and their models
// ABOUTME: Probes installed CLI binaries and returns OpenAI-compatible model list
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use embacle::discovery::resolve_binary;
use tracing::debug;

use crate::openai_types::{ModelObject, ModelsResponse};
use crate::runner::ALL_PROVIDERS;
use crate::state::SharedState;

/// Handle GET /v1/models
///
/// Probes each known provider to check if its CLI binary is installed.
/// For installed providers, lists their available models in `OpenAI` format
/// with provider prefix (e.g., "copilot:gpt-4o").
pub async fn handle(State(state): State<SharedState>) -> impl IntoResponse {
    let mut data = Vec::new();
    let state_guard = state.read().await;

    for &provider in ALL_PROVIDERS {
        let binary_name = provider.binary_name();
        let env_key = provider.env_override_key();
        let env_override = env::var(env_key).ok();

        if resolve_binary(binary_name, env_override.as_deref()).is_err() {
            debug!(provider = %provider, "Binary not found, skipping");
            continue;
        }

        match state_guard.get_runner(provider).await {
            Ok(runner) => {
                let provider_name = runner.name();
                let models = runner.available_models();

                if models.is_empty() {
                    // Provider has no model list — expose just the provider name
                    data.push(ModelObject {
                        id: provider_name.to_owned(),
                        object: "model",
                        owned_by: provider_name.to_owned(),
                    });
                } else {
                    for model in models {
                        data.push(ModelObject {
                            id: format!("{provider_name}:{model}"),
                            object: "model",
                            owned_by: provider_name.to_owned(),
                        });
                    }
                }
            }
            Err(e) => {
                debug!(provider = %provider, error = %e, "Failed to create runner");
            }
        }
    }

    let resp = ModelsResponse {
        object: "list",
        data,
    };

    (StatusCode::OK, Json(resp))
}
