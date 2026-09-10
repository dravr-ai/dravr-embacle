// ABOUTME: Bearer auth and startup posture for the REST API, delegated to dravr-tronc
// ABOUTME: Only the EMBACLE_API_KEY name is embacle's; the mechanism is the fleet's
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Authentication for the OpenAI-compatible surface.
//!
//! This module used to carry its own copy of the middleware, the loopback test
//! and the startup posture check. That copy is gone: `dravr-tronc` now serves
//! all three to the whole fleet, and the version there is the one this file's
//! design was lifted into. Keeping a second implementation would mean a fix
//! landing in one and not the other, which is the failure the shared crate
//! exists to prevent.
//!
//! What remains is the only part that is genuinely embacle's: the name of the
//! variable holding the key.
//!
//! The wire shape is unchanged by the move. Both crates answer a refusal with
//! `{"error":{"type":"authentication_error","message":"…"}}` — embacle's
//! `OpenAI` envelope carries `param` and `code` as well, but both are `None`
//! for an auth error and both are skipped when serialising, so the bytes a
//! client sees are the same.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use dravr_tronc::server::auth;

pub use dravr_tronc::server::auth::{AuthMode, InsecureBindError};

/// Environment variable holding the API key.
const API_KEY_ENV: &str = "EMBACLE_API_KEY";

/// Validate the bearer token against `EMBACLE_API_KEY`.
///
/// The variable is read on every request, so a key can be rotated without a
/// restart. Unset means every request passes — development mode, and the
/// reason [`resolve_startup_auth`] exists to refuse that on a reachable bind.
pub async fn require_auth(request: Request, next: Next) -> Response {
    auth::require_auth(API_KEY_ENV, request, next).await
}

/// Whether a usable key is configured.
#[must_use]
pub fn api_key_configured() -> bool {
    auth::api_key_configured(API_KEY_ENV)
}

/// Resolve the startup posture for a bind, or refuse it.
///
/// embacle is the guardian gate, so it fails closed: exposing tool execution on
/// a reachable interface with nothing authenticating it is never permitted.
///
/// # Errors
///
/// Returns [`InsecureBindError`] when no key is set and `host` is not loopback.
pub fn resolve_startup_auth(host: &str, gated: bool) -> Result<AuthMode, InsecureBindError> {
    auth::resolve_startup_auth(host, gated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn the_key_variable_is_embacles_own() {
        // The one thing this module still owns. Everything else is tronc's.
        assert_eq!(API_KEY_ENV, "EMBACLE_API_KEY");
    }

    #[test]
    fn a_reachable_bind_with_nothing_gating_it_is_still_refused() {
        // The posture this crate had before the move, asserted through the
        // delegation so a tronc change that weakened it would fail here.
        env::remove_var(API_KEY_ENV);
        assert!(!api_key_configured());

        let err = resolve_startup_auth("0.0.0.0", api_key_configured())
            .expect_err("a reachable bind with no key must be refused"); // Safe: test assertion
        assert_eq!(err.host, "0.0.0.0");

        assert_eq!(
            resolve_startup_auth("127.0.0.1", api_key_configured()),
            Ok(AuthMode::LoopbackDev),
            "loopback without a key is development, not a refusal"
        );

        env::set_var(API_KEY_ENV, "k");
        assert!(api_key_configured());
        assert_eq!(
            resolve_startup_auth("0.0.0.0", api_key_configured()),
            Ok(AuthMode::Enforced),
            "a key enforces on every host"
        );
        env::remove_var(API_KEY_ENV);
    }
}
