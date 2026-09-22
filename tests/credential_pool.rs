// ABOUTME: Tests the credential pool — N accounts of one CLI runner as named tiers — and the
// ABOUTME: explicit child environment each account's token rides on through the sandbox
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use embacle::config::{CliRunnerType, RunnerConfig};
use embacle::pool::{cli_pool, credential_env_key};
use embacle::types::ErrorKind;

#[tokio::test]
async fn accounts_are_named_in_order_and_the_first_keeps_the_runner_name() {
    let tiers = cli_pool(
        CliRunnerType::ClaudeCode,
        RunnerConfig::new(PathBuf::from("/bin/true")),
        &["tok-1".to_owned(), "tok-2".to_owned(), "tok-3".to_owned()],
    )
    .await
    .expect("a pool of three tokens builds");

    let names: Vec<&str> = tiers.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["claude-code", "claude-code#2", "claude-code#3"]);
    assert_eq!(tiers[0].display_name(), "Claude Code CLI");
    assert_eq!(tiers[1].display_name(), "Claude Code CLI (account 2)");
}

#[test]
fn the_credential_key_is_the_runner_family_s_first_key() {
    assert_eq!(
        credential_env_key(CliRunnerType::ClaudeCode),
        Some("CLAUDE_CODE_OAUTH_TOKEN")
    );
    assert_eq!(
        credential_env_key(CliRunnerType::Copilot),
        Some("COPILOT_GITHUB_TOKEN")
    );
    assert_eq!(
        credential_env_key(CliRunnerType::GeminiCli),
        Some("GEMINI_API_KEY")
    );
    assert_eq!(
        credential_env_key(CliRunnerType::CodexCli),
        Some("OPENAI_API_KEY")
    );
    assert_eq!(credential_env_key(CliRunnerType::WarpCli), None);
}

#[tokio::test]
async fn an_empty_pool_is_a_config_error() {
    let Err(err) = cli_pool(
        CliRunnerType::ClaudeCode,
        RunnerConfig::new(PathBuf::from("/bin/true")),
        &[],
    )
    .await
    else {
        panic!("an empty pool must be refused");
    };
    assert_eq!(err.kind, ErrorKind::Config);
}

#[tokio::test]
async fn a_runner_without_a_credential_variable_cannot_be_pooled() {
    let Err(err) = cli_pool(
        CliRunnerType::WarpCli,
        RunnerConfig::new(PathBuf::from("/bin/true")),
        &["x".to_owned()],
    )
    .await
    else {
        panic!("a runner that reads no credential must be refused");
    };
    assert_eq!(err.kind, ErrorKind::Config);
    assert!(err.message.contains("cannot be pooled"), "{}", err.message);
}

#[test]
fn with_env_keeps_one_value_per_key() {
    let config = RunnerConfig::new(PathBuf::from("/bin/true"))
        .with_env("CLAUDE_CODE_OAUTH_TOKEN", "first")
        .with_env("OTHER", "x")
        .with_env("CLAUDE_CODE_OAUTH_TOKEN", "second");
    assert_eq!(
        config.env,
        vec![
            ("OTHER".to_owned(), "x".to_owned()),
            ("CLAUDE_CODE_OAUTH_TOKEN".to_owned(), "second".to_owned())
        ]
    );
}

/// Each account's token reaches its own child, through the sandbox that
/// clears the process environment: a `claude` stand-in prints the token it
/// was started with as the result document, and each tier reports its own.
#[cfg(unix)]
#[tokio::test]
async fn each_account_s_child_sees_its_own_token() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use embacle::types::{ChatMessage, ChatRequest};

    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("claude");
    fs::write(
        &binary,
        "#!/bin/sh\nprintf '{\"is_error\":false,\"result\":\"%s\"}' \"$CLAUDE_CODE_OAUTH_TOKEN\"\n",
    )
    .unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();

    let tiers = cli_pool(
        CliRunnerType::ClaudeCode,
        RunnerConfig::new(binary),
        &[
            "token-of-account-1".to_owned(),
            "token-of-account-2".to_owned(),
        ],
    )
    .await
    .unwrap();
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);

    let first = tiers[0].complete(&request).await.unwrap();
    let second = tiers[1].complete(&request).await.unwrap();
    assert_eq!(first.content, "token-of-account-1");
    assert_eq!(second.content, "token-of-account-2");
    assert_eq!(
        second.model, "claude-code#2",
        "the response names its account"
    );
}

/// A spent second account is reported as the second account: the quota alert
/// reads the tier from the error text, and the runner writes its own name.
#[cfg(unix)]
#[tokio::test]
async fn a_spent_account_s_refusal_names_that_account() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use embacle::types::{ChatMessage, ChatRequest};

    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("claude");
    fs::write(
        &binary,
        "#!/bin/sh\nprintf '{\"is_error\":true,\"result\":\"Claude usage limit reached\"}'\nexit 1\n",
    )
    .unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();

    let tiers = cli_pool(
        CliRunnerType::ClaudeCode,
        RunnerConfig::new(binary),
        &["a".to_owned(), "b".to_owned()],
    )
    .await
    .unwrap();
    let request = ChatRequest::new(vec![ChatMessage::user("ping")]);

    let first = tiers[0].complete(&request).await.unwrap_err();
    let second = tiers[1].complete(&request).await.unwrap_err();
    assert_eq!(second.kind, ErrorKind::RateLimit);
    assert!(
        first.message.starts_with("claude-code: "),
        "{}",
        first.message
    );
    assert!(
        second.message.starts_with("claude-code#2: "),
        "{}",
        second.message
    );
}
