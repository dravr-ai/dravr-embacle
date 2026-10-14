// ABOUTME: Stream wrapper that owns a child process for proper lifecycle management
// ABOUTME: Prevents zombie processes, drains stderr, kills the child on drop or past its timeout; capped line reads
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::future::Future;
use std::io;
use std::mem;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tempfile::TempPath;
use tokio::io::AsyncBufRead;
use tokio::process::Child;
use tokio::task::JoinHandle;
use tokio::time::{self, Sleep};
use tokio_stream::Stream;
use tracing::{debug, warn};

use crate::types::{RunnerError, StreamChunk};

/// Maximum stderr to buffer during streaming (1 MiB)
pub(crate) const MAX_STREAMING_STDERR_BYTES: usize = 1024 * 1024;

/// Guards a child process for the lifetime of a streaming response.
///
/// When the stream is dropped (after natural completion or early
/// cancellation), the child process is killed and the stderr drain
/// task is aborted. This prevents zombie processes and resource leaks.
///
/// The stream is also bounded in time, as a non-streaming call is by
/// [`run_cli_command`](crate::process::run_cli_command): once the child has
/// run for its timeout, it is killed, the stream yields one
/// [`RunnerError::timeout`] and ends. A hung CLI no longer holds its
/// subprocess and its runner seat until the caller gives up.
///
/// All fields are `Unpin`, so `GuardedStream` is `Unpin` and the
/// `Stream` impl can safely access inner fields through `Pin<&mut Self>`.
pub struct GuardedStream {
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, RunnerError>> + Send>>,
    child: Option<Child>,
    stderr_task: Option<JoinHandle<Vec<u8>>>,
    /// A file the child reads its configuration from, removed with the stream.
    config_file: Option<TempPath>,
    /// Fires once the child has run for `timeout`.
    deadline: Pin<Box<Sleep>>,
    timeout: Duration,
    /// Set once the deadline fired: the stream has said so and is over.
    timed_out: bool,
}

impl GuardedStream {
    /// Create a guarded stream wrapping a child process, which may run for
    /// `timeout` (the runner's [`RunnerConfig::timeout`](crate::config::RunnerConfig::timeout))
    /// before it is killed.
    ///
    /// The `stderr_task` drains the child's stderr in the background
    /// to prevent buffer-full deadlocks where the child blocks on
    /// write to a full stderr pipe.
    pub fn new(
        inner: impl Stream<Item = Result<StreamChunk, RunnerError>> + Send + 'static,
        child: Child,
        stderr_task: JoinHandle<Vec<u8>>,
        timeout: Duration,
    ) -> Self {
        debug!(
            child_pid = child.id().unwrap_or(0),
            ?timeout,
            "GuardedStream created"
        );
        Self {
            inner: Box::pin(inner),
            child: Some(child),
            stderr_task: Some(stderr_task),
            config_file: None,
            deadline: Box::pin(time::sleep(timeout)),
            timeout,
            timed_out: false,
        }
    }

    /// Keep `config_file` — a file the child was pointed at on its command
    /// line — until the stream, and with it the child, is gone.
    pub(crate) fn keeping(mut self, config_file: Option<TempPath>) -> Self {
        self.config_file = config_file;
        self
    }
}

impl Stream for GuardedStream {
    type Item = Result<StreamChunk, RunnerError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.timed_out {
            return Poll::Ready(None);
        }
        if self.deadline.as_mut().poll(cx).is_ready() {
            self.timed_out = true;
            let timeout = self.timeout;
            warn!(
                ?timeout,
                "CLI stream ran past its timeout; killing the child"
            );
            if let Some(child) = self.child.as_mut() {
                if let Err(e) = child.start_kill() {
                    debug!(error = %e, "The timed-out CLI child had already exited");
                }
            }
            return Poll::Ready(Some(Err(RunnerError::timeout(format!(
                "CLI stream did not finish within {timeout:?}"
            )))));
        }
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for GuardedStream {
    fn drop(&mut self) {
        debug!("GuardedStream dropped, cleaning up child process");
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
        }
        if let Some(task) = self.stderr_task.take() {
            task.abort();
        }
    }
}

/// The lines of a child's output, each at most `max_line_bytes` long.
///
/// The line-by-line reader the streaming runners use: like
/// `AsyncBufReadExt::lines`, it strips the `\n` (and a `\r` before it) and
/// yields a last line that has no newline. Unlike it, a line still growing
/// past `max_line_bytes` is not buffered further: the stream yields an
/// [`io::ErrorKind::InvalidData`] error and ends, the bound a non-streaming
/// call's output already has.
pub(crate) struct CappedLines<R> {
    reader: R,
    line: Vec<u8>,
    max_line_bytes: usize,
    done: bool,
}

impl<R> CappedLines<R> {
    /// Read `reader` line by line, refusing a line longer than `max_line_bytes`.
    pub(crate) const fn new(reader: R, max_line_bytes: usize) -> Self {
        Self {
            reader,
            line: Vec::new(),
            max_line_bytes,
            done: false,
        }
    }

    /// The line read so far, without its line ending, as text.
    fn take_line(&mut self) -> io::Result<String> {
        let mut line = mem::take(&mut self.line);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        String::from_utf8(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl<R: AsyncBufRead + Unpin> Stream for CappedLines<R> {
    type Item = io::Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(None);
        }
        loop {
            let available = match Pin::new(&mut this.reader).poll_fill_buf(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(e)) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Ok(available)) => available,
            };
            if available.is_empty() {
                this.done = true;
                if this.line.is_empty() {
                    return Poll::Ready(None);
                }
                return Poll::Ready(Some(this.take_line()));
            }
            let newline = available.iter().position(|&b| b == b'\n');
            let content = newline.unwrap_or(available.len());
            if this.line.len().saturating_add(content) > this.max_line_bytes {
                this.done = true;
                return Poll::Ready(Some(Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("a line exceeded {} bytes", this.max_line_bytes),
                ))));
            }
            this.line.extend_from_slice(&available[..content]);
            let consumed = newline.map_or(content, |at| at + 1);
            Pin::new(&mut this.reader).consume(consumed);
            if newline.is_some() {
                return Poll::Ready(Some(this.take_line()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::BufReader;
    use tokio_stream::StreamExt;

    async fn lines_of(input: &[u8], max: usize) -> Vec<io::Result<String>> {
        CappedLines::new(BufReader::with_capacity(4, input), max)
            .collect()
            .await
    }

    #[tokio::test]
    async fn lines_are_split_as_buf_read_splits_them() {
        let lines: Vec<String> = lines_of(b"one\r\ntwo\n\nlast", 16)
            .await
            .into_iter()
            .map(Result::unwrap) // Safe: test assertion
            .collect();
        assert_eq!(lines, ["one", "two", "", "last"]);
    }

    #[tokio::test]
    async fn a_line_past_the_cap_is_an_error_and_ends_the_stream() {
        let lines = lines_of(b"short\nthis line is far too long\nnever read\n", 8).await;
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[0].as_deref().unwrap(), "short"); // Safe: test assertion
        let err = lines[1].as_ref().unwrap_err(); // Safe: test assertion
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(err.to_string(), "a line exceeded 8 bytes");
    }

    #[tokio::test]
    async fn a_line_of_exactly_the_cap_is_kept() {
        let lines = lines_of(b"12345678\n", 8).await;
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].as_deref().unwrap(), "12345678"); // Safe: test assertion
    }

    #[tokio::test]
    async fn invalid_utf8_is_an_error() {
        let lines = lines_of(b"\xff\xfe\n", 8).await;
        assert_eq!(
            lines[0].as_ref().unwrap_err().kind(), // Safe: test assertion
            io::ErrorKind::InvalidData
        );
    }
}
