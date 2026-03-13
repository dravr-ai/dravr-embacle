// ABOUTME: C FFI bindings for Swift integration, exposing copilot chat completion
// ABOUTME: Provides init/completion/shutdown lifecycle via extern "C" functions
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::types::{ChatMessage, ChatRequest, ImagePart, LlmProvider};
use crate::CopilotHeadlessRunner;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Holds the tokio runtime and copilot runner, created by `embacle_init`
struct FfiState {
    runtime: tokio::runtime::Runtime,
    runner: Box<dyn LlmProvider>,
}

// SAFETY: FfiState fields are Send+Sync (Runtime is Send+Sync, Box<dyn LlmProvider>
// requires Send+Sync from the trait bound). Arc allows concurrent completions
// without holding the lock during block_on.
static STATE: RwLock<Option<Arc<FfiState>>> = RwLock::new(None);

// ---------------------------------------------------------------------------
// OpenAI-compatible request types (deserialization)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct FfiRequest {
    model: Option<String>,
    messages: Vec<FfiMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct FfiMessage {
    role: String,
    #[serde(default)]
    content: Option<FfiContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FfiContent {
    Text(String),
    Parts(Vec<FfiContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum FfiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: FfiImageUrl },
}

#[derive(Deserialize)]
struct FfiImageUrl {
    url: String,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible response types (serialization)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FfiResponse {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<FfiChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<FfiTokenUsage>,
}

#[derive(Serialize)]
struct FfiChoice {
    index: u32,
    message: FfiResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct FfiResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct FfiTokenUsage {
    #[serde(rename = "prompt_tokens")]
    prompt: u32,
    #[serde(rename = "completion_tokens")]
    completion: u32,
    #[serde(rename = "total_tokens")]
    total: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a `data:` URI into its MIME type and base64 payload
fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let semi = rest.find(';')?;
    let mime = &rest[..semi];
    let after_semi = &rest[semi + 1..];
    let data = after_semi.strip_prefix("base64,")?;
    Some((mime, data))
}

/// Extract text content from message parts, ignoring non-text parts
fn extract_text(parts: &[FfiContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            FfiContentPart::Text { text } => Some(text.as_str()),
            FfiContentPart::ImageUrl { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert FFI messages to internal `ChatMessage` types
fn convert_ffi_messages(messages: &[FfiMessage]) -> Result<Vec<ChatMessage>, String> {
    let mut result = Vec::with_capacity(messages.len());

    for msg in messages {
        let content = msg.content.as_ref();
        match msg.role.as_str() {
            "system" => {
                let text = match content {
                    Some(FfiContent::Text(t)) => t.clone(),
                    Some(FfiContent::Parts(parts)) => extract_text(parts),
                    None => String::new(),
                };
                result.push(ChatMessage::system(text));
            }
            "user" => match content {
                Some(FfiContent::Text(t)) => {
                    result.push(ChatMessage::user(t.clone()));
                }
                Some(FfiContent::Parts(parts)) => {
                    let text = extract_text(parts);
                    let mut images = Vec::new();

                    for part in parts {
                        if let FfiContentPart::ImageUrl { image_url } = part {
                            let (mime, data) = parse_data_uri(&image_url.url).ok_or_else(|| {
                                format!(
                                    "unsupported image URL (expected data: URI): {}",
                                    &image_url.url[..image_url.url.len().min(60)]
                                )
                            })?;
                            images.push(
                                ImagePart::new(data, mime)
                                    .map_err(|e| format!("invalid image: {e}"))?,
                            );
                        }
                    }

                    if images.is_empty() {
                        result.push(ChatMessage::user(text));
                    } else {
                        result.push(ChatMessage::user_with_images(text, images));
                    }
                }
                None => {
                    result.push(ChatMessage::user(String::new()));
                }
            },
            "assistant" => {
                let text = match content {
                    Some(FfiContent::Text(t)) => t.clone(),
                    Some(FfiContent::Parts(parts)) => extract_text(parts),
                    None => String::new(),
                };
                result.push(ChatMessage::assistant(text));
            }
            other => {
                return Err(format!("unsupported message role: {other}"));
            }
        }
    }

    Ok(result)
}

/// Build an OpenAI-compatible response JSON from a `ChatResponse`
fn build_response_json(response: &crate::types::ChatResponse) -> String {
    let usage = response.usage.as_ref().map(|u| FfiTokenUsage {
        prompt: u.prompt_tokens,
        completion: u.completion_tokens,
        total: u.total_tokens,
    });

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let ffi_response = FfiResponse {
        id: format!("chatcmpl-{nanos:016x}"),
        object: "chat.completion",
        model: response.model.clone(),
        choices: vec![FfiChoice {
            index: 0,
            message: FfiResponseMessage {
                role: "assistant",
                content: response.content.clone(),
            },
            finish_reason: response
                .finish_reason
                .clone()
                .unwrap_or_else(|| "stop".to_owned()),
        }],
        usage,
    };

    // Serialization of these simple types cannot fail
    serde_json::to_string(&ffi_response)
        .unwrap_or_else(|e| format!("{{\"error\":{{\"message\":\"serialization failed: {e}\"}}}}"))
}

/// Convert a Rust string into a malloc'd C string (caller frees with `embacle_free_string`)
fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).map_or_else(
        |_| {
            eprintln!("embacle: response contains null bytes");
            std::ptr::null_mut()
        },
        CString::into_raw,
    )
}

// ---------------------------------------------------------------------------
// Public FFI API
// ---------------------------------------------------------------------------

/// Initialize the tokio runtime and create the copilot headless runner.
///
/// Reads copilot auth tokens from `~/.config/github-copilot/` and environment
/// variables (`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`).
///
/// Returns 0 on success, -1 if already initialized, -2 on runtime creation
/// failure, -3 on runner creation failure.
#[no_mangle]
pub extern "C" fn embacle_init() -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = match STATE.write() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("embacle_init: lock poisoned, recovering");
                e.into_inner()
            }
        };

        if guard.is_some() {
            eprintln!("embacle_init: already initialized");
            return -1;
        }

        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("embacle_init: failed to create tokio runtime: {e}");
                return -2;
            }
        };

        let runner: Box<dyn LlmProvider> = runtime.block_on(async {
            Box::new(CopilotHeadlessRunner::from_env().await) as Box<dyn LlmProvider>
        });

        *guard = Some(Arc::new(FfiState { runtime, runner }));
        0
    }));

    result.unwrap_or_else(|_| {
        eprintln!("embacle_init: panic during initialization");
        -2
    })
}

/// Send a chat completion request and return the response as a JSON string.
///
/// `request_json` must be a null-terminated UTF-8 string in `OpenAI` chat
/// completions format. `timeout_seconds` sets the maximum wait time (0 means
/// no timeout; the runner's internal timeout still applies).
///
/// Returns a malloc'd JSON string on success (free with [`embacle_free_string`]),
/// or `NULL` on error. Errors are logged to stderr.
///
/// # Safety
///
/// - `request_json` must point to a valid null-terminated UTF-8 C string.
/// - The returned pointer must be freed exactly once with [`embacle_free_string`].
#[no_mangle]
pub extern "C" fn embacle_chat_completion(
    request_json: *const c_char,
    timeout_seconds: i32,
) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if request_json.is_null() {
            eprintln!("embacle_chat_completion: request_json is NULL");
            return std::ptr::null_mut();
        }

        // SAFETY: caller guarantees valid null-terminated UTF-8
        let json_str = unsafe { CStr::from_ptr(request_json) };
        let json_str = match json_str.to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("embacle_chat_completion: invalid UTF-8: {e}");
                return std::ptr::null_mut();
            }
        };

        let ffi_request: FfiRequest = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("embacle_chat_completion: invalid request JSON: {e}");
                return std::ptr::null_mut();
            }
        };

        let messages = match convert_ffi_messages(&ffi_request.messages) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("embacle_chat_completion: {e}");
                return std::ptr::null_mut();
            }
        };

        let mut chat_request = ChatRequest::new(messages);
        chat_request.model = ffi_request.model;
        chat_request.temperature = ffi_request.temperature;
        chat_request.max_tokens = ffi_request.max_tokens;
        chat_request.top_p = ffi_request.top_p;

        let timeout = u64::try_from(timeout_seconds)
            .ok()
            .filter(|&s| s > 0)
            .map(Duration::from_secs);

        // Clone the Arc so the read lock is released before blocking
        let state = STATE
            .read()
            .unwrap_or_else(|e| {
                eprintln!("embacle_chat_completion: lock poisoned, recovering");
                e.into_inner()
            })
            .as_ref()
            .cloned();

        let Some(state) = state else {
            eprintln!("embacle_chat_completion: not initialized (call embacle_init first)");
            return std::ptr::null_mut();
        };

        let result = state.runtime.block_on(async {
            let completion = state.runner.complete(&chat_request);
            match timeout {
                Some(duration) => tokio::time::timeout(duration, completion)
                    .await
                    .unwrap_or_else(|_| {
                        Err(crate::types::RunnerError::timeout(format!(
                            "completion timed out after {timeout_seconds}s"
                        )))
                    }),
                None => completion.await,
            }
        });

        match result {
            Ok(response) => {
                let json = build_response_json(&response);
                to_c_string(&json)
            }
            Err(e) => {
                eprintln!("embacle_chat_completion: {:?}: {}", e.kind, e.message);
                std::ptr::null_mut()
            }
        }
    }));

    result.unwrap_or_else(|_| {
        eprintln!("embacle_chat_completion: panic during completion");
        std::ptr::null_mut()
    })
}

/// Free a string returned by embacle functions.
///
/// Passing `NULL` is a no-op.
///
/// # Safety
///
/// - `ptr` must have been returned by [`embacle_chat_completion`] or be `NULL`.
/// - `ptr` must not be freed more than once.
#[no_mangle]
pub extern "C" fn embacle_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: ptr was allocated by CString::into_raw in to_c_string
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }));
}

/// Shutdown the tokio runtime and release all resources.
///
/// After calling this, [`embacle_chat_completion`] returns `NULL` until
/// [`embacle_init`] is called again.
#[no_mangle]
pub extern "C" fn embacle_shutdown() {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = match STATE.write() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("embacle_shutdown: lock poisoned, recovering");
                e.into_inner()
            }
        };
        if let Some(arc) = guard.take() {
            // Unwrap the Arc — if other threads still hold references,
            // wait for them to finish first
            match Arc::try_unwrap(arc) {
                Ok(state) => {
                    state.runtime.shutdown_timeout(Duration::from_secs(5));
                }
                Err(still_shared) => {
                    // Other completions are still in flight; put it back and
                    // let the runtime shut down when the last Arc drops
                    eprintln!("embacle_shutdown: waiting for in-flight completions");
                    *guard = Some(still_shared);
                }
            }
        }
    }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_data_uri_valid_png() {
        let url = "data:image/png;base64,iVBORw0KGgo=";
        let (mime, data) = parse_data_uri(url).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "iVBORw0KGgo=");
    }

    #[test]
    fn parse_data_uri_valid_jpeg() {
        let url = "data:image/jpeg;base64,/9j/4AAQ";
        let (mime, data) = parse_data_uri(url).unwrap();
        assert_eq!(mime, "image/jpeg");
        assert_eq!(data, "/9j/4AAQ");
    }

    #[test]
    fn parse_data_uri_invalid_no_prefix() {
        assert!(parse_data_uri("https://example.com/img.png").is_none());
    }

    #[test]
    fn parse_data_uri_invalid_no_base64() {
        assert!(parse_data_uri("data:image/png;charset=utf-8,abc").is_none());
    }

    #[test]
    fn convert_simple_text_messages() {
        let messages = vec![
            FfiMessage {
                role: "system".to_owned(),
                content: Some(FfiContent::Text("Be concise".to_owned())),
            },
            FfiMessage {
                role: "user".to_owned(),
                content: Some(FfiContent::Text("Hello".to_owned())),
            },
        ];
        let result = convert_ffi_messages(&messages).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "Be concise");
        assert_eq!(result[1].content, "Hello");
    }

    #[test]
    fn convert_user_multipart_with_image() {
        let messages = vec![FfiMessage {
            role: "user".to_owned(),
            content: Some(FfiContent::Parts(vec![
                FfiContentPart::Text {
                    text: "What is this?".to_owned(),
                },
                FfiContentPart::ImageUrl {
                    image_url: FfiImageUrl {
                        url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_owned(),
                    },
                },
            ])),
        }];
        let result = convert_ffi_messages(&messages).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "What is this?");
        assert!(result[0].images.is_some());
        assert_eq!(result[0].images.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn convert_unsupported_role_fails() {
        let messages = vec![FfiMessage {
            role: "function".to_owned(),
            content: Some(FfiContent::Text("result".to_owned())),
        }];
        assert!(convert_ffi_messages(&messages).is_err());
    }

    #[test]
    fn convert_missing_content_produces_empty() {
        let messages = vec![FfiMessage {
            role: "user".to_owned(),
            content: None,
        }];
        let result = convert_ffi_messages(&messages).unwrap();
        assert_eq!(result[0].content, "");
    }

    #[test]
    fn convert_non_data_uri_fails() {
        let messages = vec![FfiMessage {
            role: "user".to_owned(),
            content: Some(FfiContent::Parts(vec![FfiContentPart::ImageUrl {
                image_url: FfiImageUrl {
                    url: "https://example.com/img.png".to_owned(),
                },
            }])),
        }];
        let err = convert_ffi_messages(&messages).unwrap_err();
        assert!(err.contains("unsupported image URL"));
    }

    #[test]
    fn build_response_json_basic() {
        let response = crate::types::ChatResponse {
            content: "Hello world".to_owned(),
            model: "test-model".to_owned(),
            usage: Some(crate::types::TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        };
        let json = build_response_json(&response);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "test-model");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(parsed["choices"][0]["message"]["role"], "assistant");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["prompt_tokens"], 10);
        assert_eq!(parsed["usage"]["total_tokens"], 15);
    }

    #[test]
    fn build_response_json_no_usage() {
        let response = crate::types::ChatResponse {
            content: "Hi".to_owned(),
            model: "m".to_owned(),
            usage: None,
            finish_reason: None,
            warnings: None,
            tool_calls: None,
        };
        let json = build_response_json(&response);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("usage").is_none());
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn request_json_round_trip() {
        let json = r#"{
            "model": "claude-opus-4.6-fast",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }"#;
        let req: FfiRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model.as_deref(), Some("claude-opus-4.6-fast"));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.max_tokens, Some(100));
    }

    #[test]
    fn request_json_multipart_content() {
        let json = r#"{
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}
                ]
            }]
        }"#;
        let req: FfiRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            Some(FfiContent::Parts(parts)) => assert_eq!(parts.len(), 2),
            _ => panic!("expected Parts variant"),
        }
    }

    #[test]
    fn request_json_minimal() {
        let json = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
        let req: FfiRequest = serde_json::from_str(json).unwrap();
        assert!(req.model.is_none());
        assert!(req.temperature.is_none());
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn to_c_string_and_free() {
        let ptr = to_c_string("hello");
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, "hello");
        embacle_free_string(ptr);
    }

    #[test]
    fn free_null_is_noop() {
        embacle_free_string(std::ptr::null_mut());
    }
}
