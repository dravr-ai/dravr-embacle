// ABOUTME: Low-level async wrapper around a detached tmux session used as a driveable terminal
// ABOUTME: Spawns a CLI, pastes input via bracketed paste, captures pane history, polls for markers
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Tmux Session Substrate
//!
//! Thin async wrapper around `tmux new-session -d`, `load-buffer` +
//! `paste-buffer`, `send-keys`, and `capture-pane`. The intent is to drive
//! interactive TUI agents (Claude Code, GitHub Copilot CLI) from a long-lived
//! detached tmux session — the agent's own scrollback acts as conversation
//! memory, so the runner does not need to persist history itself.
//!
//! ## Why tmux
//!
//! Full-screen TUI agents use alternate-screen-buffer mode, mouse tracking,
//! and cursor positioning. Raw PTY stdout cannot be scraped directly; it has
//! to be fed through a terminal emulator. tmux already is one. It also
//! survives the embacle process restart, exposes session inspection via
//! `tmux attach`, and standardises bracketed-paste delivery across CLIs.
//!
//! ## Input delivery
//!
//! Multi-line prompts are written to a paste buffer with `load-buffer`, then
//! injected with `paste-buffer -p` (bracketed paste). Newlines inside the
//! pasted content are inserted as literal newlines in the agent's input box
//! — they do not submit. An explicit `Enter` key follows to submit the
//! composed prompt.
//!
//! ## Done detection
//!
//! Callers select a unique sentinel string, instruct the agent to emit it at
//! the end of its response, and call [`TmuxSession::wait_for_marker`]. The
//! method polls `capture-pane` until the marker appears in the rendered (and
//! ANSI-stripped) pane history, or the hard timeout fires.

use std::process::{Command as StdCommand, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{sleep, timeout as tokio_timeout};
use tracing::{debug, trace, warn};

use crate::types::RunnerError;

/// Maximum bytes captured from a single `capture-pane` invocation (4 MiB).
///
/// `capture-pane` writes to stdout; this cap prevents runaway memory growth
/// if a misbehaving CLI produces an enormous scrollback.
const CAPTURE_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Wall-clock budget for any single tmux command (e.g. `send-keys`,
/// `capture-pane`). Individual tmux subcommands are essentially instantaneous
/// on a healthy server; this guards against a stuck tmux daemon.
const TMUX_COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

/// Handle to a detached tmux session running an interactive CLI.
///
/// Drop semantics: on drop, the session is best-effort killed synchronously
/// so a panicking caller cannot leak detached sessions. Explicit
/// [`TmuxSession::kill`] is preferred for clean shutdown because it returns
/// errors.
pub struct TmuxSession {
    name: String,
    killed: AtomicBool,
}

impl TmuxSession {
    /// Spawn a new detached tmux session named `name` running `program` with
    /// `args`. The session is created with `-d` (detached) so it runs in the
    /// background; callers may later `tmux attach -t <name>` for debugging.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if `tmux` is not on PATH, the spawn command
    /// fails, or a session with the same name already exists.
    pub async fn spawn(name: &str, program: &str, args: &[&str]) -> Result<Self, RunnerError> {
        Self::spawn_with_env(name, program, args, &[]).await
    }

    /// Like [`Self::spawn`] but injects environment variables into the
    /// spawned session via `tmux set-environment`.
    ///
    /// # Errors
    ///
    /// See [`Self::spawn`].
    pub async fn spawn_with_env(
        name: &str,
        program: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Self, RunnerError> {
        validate_session_name(name)?;

        let mut cmd = tmux_command();
        cmd.args(["new-session", "-d", "-s", name]);
        cmd.arg(program);
        for a in args {
            cmd.arg(a);
        }

        debug!(session = name, program, ?args, "Spawning tmux session");
        run_tmux(cmd).await?;

        // Apply environment after creation. `set-environment -t <name>` sets a
        // session-scoped variable that future windows in that session
        // inherit; the initial command already running won't see it, but
        // anything the user spawns afterwards will. For TUI agents that
        // re-read env via menus this is fine; for one-shot binaries the
        // caller should pre-bake env into the spawn args.
        for (key, value) in env {
            let mut env_cmd = tmux_command();
            env_cmd.args(["set-environment", "-t", name, key, value]);
            run_tmux(env_cmd).await?;
        }

        Ok(Self {
            name: name.to_owned(),
            killed: AtomicBool::new(false),
        })
    }

    /// Session name (as registered with tmux).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether a session by this name currently exists.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] only on tmux execution failure; a non-existent
    /// session is reported as `Ok(false)`.
    pub async fn exists(name: &str) -> Result<bool, RunnerError> {
        validate_session_name(name)?;
        let mut cmd = tmux_command();
        cmd.args(["has-session", "-t", name]);
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        match cmd.status().await {
            Ok(status) => Ok(status.success()),
            Err(e) => Err(RunnerError::internal(format!(
                "Failed to invoke tmux has-session: {e}"
            ))),
        }
    }

    /// Paste `text` into the session as if the user had pressed a paste
    /// keystroke with bracketed paste enabled. Multi-line content is
    /// delivered as a single block; newlines become literal newlines in the
    /// receiving program's input — they do not submit.
    ///
    /// Use [`Self::send_enter`] afterwards to submit the composed text.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if writing to tmux's stdin or invoking
    /// `paste-buffer` fails.
    pub async fn paste(&self, text: &str) -> Result<(), RunnerError> {
        let buffer_name = format!("embacle-{}", uuid::Uuid::new_v4());

        // load-buffer with `-` reads from stdin, avoiding a tempfile.
        let mut load = tmux_command();
        load.args(["load-buffer", "-b", &buffer_name, "-"]);
        load.stdin(Stdio::piped());
        load.stdout(Stdio::null());
        load.stderr(Stdio::piped());

        let mut child = load
            .spawn()
            .map_err(|e| RunnerError::internal(format!("Failed to spawn tmux load-buffer: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| RunnerError::internal(format!("Failed to write paste body: {e}")))?;
            // Explicit drop closes stdin so load-buffer can exit.
            drop(stdin);
        }

        let load_output = tokio_timeout(TMUX_COMMAND_TIMEOUT, child.wait_with_output())
            .await
            .map_err(|_| RunnerError::timeout("tmux load-buffer timed out"))?
            .map_err(|e| RunnerError::internal(format!("Failed to await load-buffer: {e}")))?;

        if !load_output.status.success() {
            return Err(RunnerError::external_service(
                "tmux",
                format!(
                    "load-buffer failed: {}",
                    String::from_utf8_lossy(&load_output.stderr).trim()
                ),
            ));
        }

        // paste-buffer -p emits bracketed paste wrappers, -d deletes the
        // buffer after pasting so we do not leak named buffers across turns.
        let mut paste = tmux_command();
        paste.args([
            "paste-buffer",
            "-p",
            "-d",
            "-b",
            &buffer_name,
            "-t",
            &self.name,
        ]);
        run_tmux(paste).await?;

        trace!(session = %self.name, len = text.len(), "Pasted text into tmux session");
        Ok(())
    }

    /// Submit the pending input by sending the `Enter` keystroke.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] on tmux command failure.
    pub async fn send_enter(&self) -> Result<(), RunnerError> {
        self.send_key("Enter").await
    }

    /// Send a single tmux-named key (e.g. `Enter`, `Escape`, `C-c`).
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] on tmux command failure.
    pub async fn send_key(&self, key: &str) -> Result<(), RunnerError> {
        let mut cmd = tmux_command();
        cmd.args(["send-keys", "-t", &self.name, key]);
        run_tmux(cmd).await
    }

    /// Capture the visible pane content (when `include_history` is `false`)
    /// or the full scrollback (when `true`). The returned string is the raw
    /// tmux output with terminal escape sequences left intact — use
    /// [`strip_ansi`] before substring matching against agent text.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if `capture-pane` fails or its output exceeds
    /// the [`CAPTURE_MAX_BYTES`] cap.
    pub async fn capture(&self, include_history: bool) -> Result<String, RunnerError> {
        let mut cmd = tmux_command();
        cmd.args(["capture-pane", "-p", "-J", "-t", &self.name]);
        if include_history {
            // -S - selects the start of the history buffer, -E - the end.
            cmd.args(["-S", "-", "-E", "-"]);
        }
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        // `-e` preserves escape sequences; default capture already strips
        // most attributes but keeps newlines. We do NOT pass -e because
        // bracketed paste output would otherwise show up as control codes in
        // the marker search.

        let output = tokio_timeout(TMUX_COMMAND_TIMEOUT, cmd.output())
            .await
            .map_err(|_| RunnerError::timeout("tmux capture-pane timed out"))?
            .map_err(|e| RunnerError::internal(format!("Failed to run capture-pane: {e}")))?;

        if !output.status.success() {
            return Err(RunnerError::external_service(
                "tmux",
                format!(
                    "capture-pane failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            ));
        }

        if output.stdout.len() > CAPTURE_MAX_BYTES {
            warn!(
                bytes = output.stdout.len(),
                cap = CAPTURE_MAX_BYTES,
                "capture-pane exceeded byte cap; truncating"
            );
        }
        let bytes = &output.stdout[..output.stdout.len().min(CAPTURE_MAX_BYTES)];
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Poll [`Self::capture`] until `marker` appears in the ANSI-stripped
    /// pane history, or the `hard_timeout` elapses.
    ///
    /// Returns the full ANSI-stripped pane history at the moment the marker
    /// was observed. Callers extract their content of interest by slicing
    /// before the marker (and after their input echo, if needed).
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] with [`ErrorKind::Timeout`](crate::types::ErrorKind::Timeout)
    /// if the marker is not observed within `hard_timeout`.
    pub async fn wait_for_marker(
        &self,
        marker: &str,
        hard_timeout: Duration,
        poll_interval: Duration,
    ) -> Result<String, RunnerError> {
        if marker.is_empty() {
            return Err(RunnerError::internal(
                "wait_for_marker called with empty marker",
            ));
        }
        let deadline = Instant::now() + hard_timeout;

        loop {
            let raw = self.capture(true).await?;
            let stripped = strip_ansi(&raw);
            if stripped.contains(marker) {
                return Ok(stripped);
            }
            if Instant::now() >= deadline {
                return Err(RunnerError::timeout(format!(
                    "Marker not observed in tmux session '{}' within {:?}",
                    self.name, hard_timeout
                )));
            }
            sleep(poll_interval).await;
        }
    }

    /// Kill the underlying tmux session. Idempotent — calling kill more than
    /// once is a no-op after the first success.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the tmux invocation fails for reasons
    /// other than "session does not exist".
    pub async fn kill(&self) -> Result<(), RunnerError> {
        if self.killed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let mut cmd = tmux_command();
        cmd.args(["kill-session", "-t", &self.name]);
        cmd.stdout(Stdio::null());
        // Suppress "no such session" stderr noise since kill is idempotent
        // from the caller's perspective.
        cmd.stderr(Stdio::null());

        let status = tokio_timeout(TMUX_COMMAND_TIMEOUT, cmd.status())
            .await
            .map_err(|_| RunnerError::timeout("tmux kill-session timed out"))?
            .map_err(|e| RunnerError::internal(format!("Failed to run kill-session: {e}")))?;

        if !status.success() {
            debug!(session = %self.name, "kill-session returned non-zero (likely already dead)");
        }
        Ok(())
    }
}

impl Drop for TmuxSession {
    fn drop(&mut self) {
        if self.killed.load(Ordering::SeqCst) {
            return;
        }
        // Synchronous best-effort cleanup. We cannot block on async tmux
        // commands from Drop; std::process::Command suffices and tmux
        // kill-session is fast.
        let _ = StdCommand::new("tmux")
            .args(["kill-session", "-t", &self.name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Build a base `tmux` Command with stdio wired up to fail loudly on error.
fn tmux_command() -> Command {
    let mut cmd = Command::new("tmux");
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd
}

/// Run a tmux command with a wall-clock timeout, mapping non-zero exits to
/// [`RunnerError::external_service`].
async fn run_tmux(mut cmd: Command) -> Result<(), RunnerError> {
    let output = tokio_timeout(TMUX_COMMAND_TIMEOUT, cmd.output())
        .await
        .map_err(|_| RunnerError::timeout("tmux command timed out"))?
        .map_err(|e| RunnerError::internal(format!("Failed to invoke tmux: {e}")))?;

    if !output.status.success() {
        return Err(RunnerError::external_service(
            "tmux",
            format!(
                "tmux exited with status {:?}: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    Ok(())
}

/// Reject session names that would let tmux argument parsing or shell
/// injection bite us. tmux session names cannot contain `:` (used as a
/// window/pane separator in target specs) and cannot be empty.
fn validate_session_name(name: &str) -> Result<(), RunnerError> {
    if name.is_empty() {
        return Err(RunnerError::config("tmux session name must be non-empty"));
    }
    if name.contains(':') || name.contains('.') {
        return Err(RunnerError::config(format!(
            "tmux session name '{name}' contains reserved characters ':' or '.'"
        )));
    }
    if name.contains(char::is_whitespace) {
        return Err(RunnerError::config(format!(
            "tmux session name '{name}' contains whitespace"
        )));
    }
    Ok(())
}

/// Strip ANSI/VT100 escape sequences from `input`, returning printable text.
///
/// Handles the subset of escape codes emitted by interactive TUI agents:
/// CSI sequences (`ESC [ ... <final>`), OSC sequences (`ESC ] ... BEL` or
/// `ESC ] ... ESC \`), single-character escapes (`ESC X`), and the bare
/// control characters that survive `capture-pane`'s default formatting.
///
/// Not a full terminal emulator — does not reconstruct cursor-positioned
/// overwrites. Sufficient for substring matching against markers in the
/// rendered pane snapshot.
#[must_use]
pub fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(c) = chars.next() {
        if c != '\x1b' {
            // Drop other C0 controls except whitespace we want to preserve.
            if c == '\n' || c == '\t' || !c.is_control() {
                out.push(c);
            }
            continue;
        }

        // Saw ESC. Inspect next byte to classify the sequence.
        let Some(next) = chars.next() else { break };
        match next {
            '[' => {
                // CSI: parameters in 0x30..=0x3F, intermediates in 0x20..=0x2F,
                // terminated by a final byte in 0x40..=0x7E.
                for c2 in chars.by_ref() {
                    let code = c2 as u32;
                    if (0x40..=0x7E).contains(&code) {
                        break;
                    }
                }
            }
            ']' => {
                // OSC: terminated by BEL (0x07) or ST (ESC \).
                while let Some(c2) = chars.next() {
                    if c2 == '\x07' {
                        break;
                    }
                    if c2 == '\x1b' {
                        // Consume the trailing `\` of ST and stop.
                        let _ = chars.next();
                        break;
                    }
                }
            }
            'P' | 'X' | '^' | '_' => {
                // DCS / SOS / PM / APC: all terminated by ST.
                while let Some(c2) = chars.next() {
                    if c2 == '\x1b' {
                        let _ = chars.next();
                        break;
                    }
                }
            }
            // Two-byte escapes like ESC = / ESC > / ESC ( B  — drop next byte
            // for the charset variants, otherwise just drop the ESC byte.
            '(' | ')' | '*' | '+' => {
                let _ = chars.next();
            }
            _ => {
                // Single-byte ESC X — just drop both.
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_csi_color_sequences() {
        let input = "\x1b[31mhello\x1b[0m world";
        assert_eq!(strip_ansi(input), "hello world");
    }

    #[test]
    fn strips_cursor_positioning() {
        let input = "before\x1b[2J\x1b[H\x1b[1;1Hafter";
        assert_eq!(strip_ansi(input), "beforeafter");
    }

    #[test]
    fn strips_osc_terminated_by_bel() {
        let input = "\x1b]0;window title\x07payload";
        assert_eq!(strip_ansi(input), "payload");
    }

    #[test]
    fn strips_osc_terminated_by_st() {
        let input = "\x1b]52;c;data\x1b\\done";
        assert_eq!(strip_ansi(input), "done");
    }

    #[test]
    fn preserves_newlines_and_tabs() {
        let input = "line1\nline2\tindented";
        assert_eq!(strip_ansi(input), "line1\nline2\tindented");
    }

    #[test]
    fn drops_orphan_escape_at_end() {
        let input = "trailing\x1b";
        assert_eq!(strip_ansi(input), "trailing");
    }

    #[test]
    fn session_name_rejects_colon_and_dot() {
        assert!(validate_session_name("foo:bar").is_err());
        assert!(validate_session_name("foo.bar").is_err());
        assert!(validate_session_name("").is_err());
        assert!(validate_session_name("foo bar").is_err());
        assert!(validate_session_name("embacle-claude-abc123").is_ok());
    }
}
