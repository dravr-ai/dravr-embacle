// ABOUTME: Drives the Claude Code and Copilot runners against shell stand-ins that dump their argv,
// ABOUTME: proving the MCP bearer and the prompt stay off the command line and the config file is 0600
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Anything on a child's command line is readable by every process on the
//! host in `/proc/<pid>/cmdline`. The tool host's per-turn bearer travels in
//! the MCP configuration, and the prompt carries the athlete's words, so the
//! configuration goes in a 0600 file and Claude Code's prompt goes on stdin.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use embacle::config::RunnerConfig;
use embacle::types::{
    ChatMessage, ChatRequest, LlmProvider, McpHeader, McpServerConfig, McpTransport,
};
use embacle::{ClaudeCodeRunner, CopilotRunner};
use tokio_stream::StreamExt;

// Each test execs a script it just wrote; see `support` for the ETXTBSY race
// that makes every call here retry its spawn.
use crate::support::retry_text_file_busy;

const BEARER: &str = "Bearer tok-secret-123";
const PROMPT: &str = "my resting HR is 41";

/// Write `body` as an executable `name` in `dir`.
fn stand_in(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, format!("#!/bin/sh\ndir=$(dirname \"$0\")\n{body}")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Shell that records the argv, one argument per line, and — for the
/// argument after `flag`, with any leading `@` removed — the file's
/// `ls -l` line and contents, while the child is running.
fn recording(flag: &str) -> String {
    format!(
        r#": > "$dir/argv.dump"
for a in "$@"; do printf '%s\n' "$a" >> "$dir/argv.dump"; done
prev=""
for a in "$@"; do
  if [ "$prev" = "{flag}" ]; then
    f="${{a#@}}"
    printf '%s\n' "$f" > "$dir/path.dump"
    ls -l "$f" > "$dir/mode.dump"
    cat "$f" > "$dir/config.dump"
  fi
  prev="$a"
done
"#
    )
}

fn request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user(PROMPT)]).with_mcp_servers(vec![McpServerConfig {
        name: "dravr".to_owned(),
        transport: McpTransport::Http {
            url: "http://127.0.0.1:9/mcp".to_owned(),
            headers: vec![McpHeader {
                name: "Authorization".to_owned(),
                value: BEARER.to_owned(),
            }],
        },
    }])
}

fn dump(dir: &Path, name: &str) -> String {
    fs::read_to_string(dir.join(name))
        .unwrap_or_else(|e| panic!("the stand-in did not write {name}: {e}"))
}

/// The argv carries neither the bearer nor the prompt; the configuration
/// file the child was pointed at was 0600, held the bearer, and is gone.
fn assert_secrets_stayed_off_the_command_line(dir: &Path) {
    let argv = dump(dir, "argv.dump");
    assert!(!argv.contains("tok-secret-123"), "bearer on argv: {argv}");
    assert!(!argv.contains(PROMPT), "prompt on argv: {argv}");
    assert!(
        dump(dir, "mode.dump").starts_with("-rw------- "),
        "the config file is readable by its owner only: {}",
        dump(dir, "mode.dump")
    );
    assert!(dump(dir, "config.dump").contains(BEARER));
    let path = dump(dir, "path.dump");
    assert!(
        !Path::new(path.trim_end()).exists(),
        "the config file is removed once the call is over: {path}"
    );
}

const RESULT_DOCUMENT: &str =
    r#"{"result":"ok","is_error":false,"usage":{"input_tokens":1,"output_tokens":1}}"#;

#[tokio::test]
async fn claude_reads_its_prompt_on_stdin_and_its_mcp_config_from_a_private_file() {
    let dir = tempfile::tempdir().unwrap();
    let body = format!(
        "{}cat > \"$dir/stdin.dump\"\nprintf '%s\\n' '{RESULT_DOCUMENT}'\n",
        recording("--mcp-config")
    );
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(stand_in(dir.path(), "claude", &body)));

    let request = request();
    let response = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap();

    assert_eq!(response.content, "ok");
    assert!(dump(dir.path(), "stdin.dump").contains(PROMPT));
    assert!(dump(dir.path(), "argv.dump")
        .lines()
        .any(|a| a == "--print"));
    assert_secrets_stayed_off_the_command_line(dir.path());
}

#[tokio::test]
async fn a_streamed_claude_call_keeps_its_config_file_while_the_child_runs() {
    let dir = tempfile::tempdir().unwrap();
    let body = format!(
        "{}cat > \"$dir/stdin.dump\"\n\
         printf '%s\\n' '{{\"type\":\"assistant\",\"message\":{{\"content\":[{{\"type\":\"text\",\"text\":\"ok\"}}]}}}}'\n\
         printf '%s\\n' '{{\"type\":\"result\",\"is_error\":false,\"result\":\"ok\"}}'\n",
        recording("--mcp-config")
    );
    let runner = ClaudeCodeRunner::new(RunnerConfig::new(stand_in(dir.path(), "claude", &body)));

    let request = request();
    let mut stream = retry_text_file_busy(|| runner.complete_stream(&request))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(&chunk.unwrap().delta);
    }
    drop(stream);

    assert_eq!(text, "ok");
    assert!(dump(dir.path(), "stdin.dump").contains(PROMPT));
    assert_secrets_stayed_off_the_command_line(dir.path());
}

#[tokio::test]
async fn copilot_reads_its_mcp_config_from_a_private_file() {
    let dir = tempfile::tempdir().unwrap();
    let body = format!("{}echo ok\n", recording("--additional-mcp-config"));
    let runner = CopilotRunner::new(RunnerConfig::new(stand_in(dir.path(), "copilot", &body)));

    let request = request();
    let response = retry_text_file_busy(|| runner.complete(&request))
        .await
        .unwrap();

    assert_eq!(response.content, "ok");
    let argv = dump(dir.path(), "argv.dump");
    let config_arg = argv
        .lines()
        .skip_while(|a| *a != "--additional-mcp-config")
        .nth(1)
        .unwrap_or_else(|| panic!("no --additional-mcp-config value: {argv}"));
    assert!(
        config_arg.starts_with('@'),
        "the CLI's file form is @<path>: {config_arg}"
    );
    // The -p runner still passes its prompt as `-p <prompt>`; only the
    // configuration, which carries the bearer, moved.
    assert!(!argv.contains("tok-secret-123"), "bearer on argv: {argv}");
    assert!(dump(dir.path(), "mode.dump").starts_with("-rw------- "));
    assert!(dump(dir.path(), "config.dump").contains(BEARER));
    let path = dump(dir.path(), "path.dump");
    assert!(!Path::new(path.trim_end()).exists(), "{path}");
}
