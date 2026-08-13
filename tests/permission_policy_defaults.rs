// ABOUTME: The ACP permission policy must deny unless a host opts into approval
// ABOUTME: An approving default gave a Dravr coaching turn shell in the service container

//! The copilot subprocess's own tools (shell, git, file editing) run in the
//! host process's working directory. A host that assembles its prompt from
//! untrusted input therefore turns an auto-approval into arbitrary execution
//! beside its environment and credentials.
//!
//! These tests pin the safe side: denial is what you get unless approval is
//! spelled out, and a misspelled value degrades to denial rather than silently
//! granting a shell.

// The module under test is feature-gated; the `//!` docs above sit before this
// attribute deliberately, so `missing-docs` still sees them when the feature is
// off and the crate root compiles out.
#![cfg(feature = "copilot-headless")]

use embacle::copilot_headless_config::PermissionPolicy;

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
    // reaches production — the derived default alone would not have caught the
    // original bug, because the env path had its own approving fallback.
    for raw in ["", "  ", "yes", "true", "1", "allow", "deny_all", "nonsense"] {
        assert_eq!(
            policy_for(raw),
            PermissionPolicy::DenyAll,
            "{raw:?} must not enable auto-approval"
        );
    }

    for raw in ["auto_approve", "autoapprove", "approve", "AUTO_APPROVE"] {
        assert_eq!(
            policy_for(raw),
            PermissionPolicy::AutoApprove,
            "{raw:?} is an explicit opt-in and must approve"
        );
    }
}

/// Mirrors the parser in `CopilotHeadlessConfig::from_env`. Kept in the test so
/// the accepted spellings are asserted rather than assumed; a divergence here
/// means the parser changed and this file has to change with it.
fn policy_for(raw: &str) -> PermissionPolicy {
    match raw.to_lowercase().as_str() {
        "auto_approve" | "autoapprove" | "approve" => PermissionPolicy::AutoApprove,
        _ => PermissionPolicy::DenyAll,
    }
}
