// ABOUTME: The Copilot runtime permission policy must deny unless a host opts into approval
// ABOUTME: An approving default gave a Dravr coaching turn shell in the service container
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! The Copilot runtime's own tools (shell, git, file editing) run with the
//! host's environment and credentials, in the session working directory — a
//! scratch directory unless the host configures one. A host that assembles its
//! prompt from untrusted input therefore turns an auto-approval into arbitrary
//! execution beside its secrets.
//!
//! These tests pin the safe side: denial is what you get unless approval is
//! spelled out, and a misspelled value degrades to denial rather than silently
//! granting a shell. [`PermissionPolicy`] lives in `copilot_common`, which
//! compiles without any runner feature, so this file always runs.

use embacle::PermissionPolicy;

#[test]
fn derived_default_denies() {
    assert_eq!(
        PermissionPolicy::default(),
        PermissionPolicy::DenyAll,
        "the safe value must not depend on each consumer remembering to set an env var"
    );
}

#[test]
fn approval_requires_an_explicit_spelling() {
    // Guards the parser's fallback arm, which is the value that actually
    // reaches production through `CopilotSdkConfig::from_env` — the derived
    // default alone would not have caught the original bug, because the env
    // path had its own approving fallback.
    for raw in [
        "", "  ", "yes", "true", "1", "allow", "deny_all", "nonsense",
    ] {
        assert_eq!(
            PermissionPolicy::parse(raw),
            PermissionPolicy::DenyAll,
            "{raw:?} must not enable auto-approval"
        );
    }

    for raw in ["auto_approve", "autoapprove", "approve", "AUTO_APPROVE"] {
        assert_eq!(
            PermissionPolicy::parse(raw),
            PermissionPolicy::AutoApprove,
            "{raw:?} is an explicit opt-in and must approve"
        );
    }
}
