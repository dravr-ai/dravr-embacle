/*
 * embacle — C FFI for Swift integration
 *
 * Static library: libembacle.a
 * Build: cargo build --release --features ffi
 *
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 dravr.ai
 */

#ifndef EMBACLE_H
#define EMBACLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the tokio runtime and copilot headless runner.
 *
 * Reads copilot auth tokens from ~/.config/github-copilot/ and env vars
 * (COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN).
 *
 * Returns:
 *   0  success
 *  -1  already initialized
 *  -2  runtime creation failed
 */
int embacle_init(void);

/*
 * Send a chat completion request.
 *
 * request_json: OpenAI-compatible chat completions JSON (null-terminated UTF-8).
 * timeout_seconds: max wait in seconds (0 = no timeout, runner default applies).
 *
 * Returns a malloc'd JSON string on success (free with embacle_free_string),
 * or NULL on error. Errors are logged to stderr.
 */
char* embacle_chat_completion(const char* request_json, int timeout_seconds);

/*
 * Free a string returned by embacle functions. NULL is a no-op.
 */
void embacle_free_string(char* ptr);

/*
 * Shutdown the tokio runtime and release resources.
 * Safe to call embacle_init() again after shutdown.
 */
void embacle_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* EMBACLE_H */
