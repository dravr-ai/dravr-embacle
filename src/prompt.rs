// ABOUTME: Prompt construction from ChatMessage sequences for CLI invocations
// ABOUTME: Extracts system messages, builds role-prefixed prompt strings, materializes images to temp files
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::PathBuf;

use base64::Engine;
use tracing::{debug, warn};

use crate::types::{ChatMessage, ImagePart, MessageRole, RunnerError};

/// Prompt with materialized image files for CLI runners
///
/// Holds the built prompt string and an optional temp directory containing
/// image files decoded from base64. The temp directory is automatically
/// cleaned up when this struct is dropped, so it must be kept alive until
/// the CLI subprocess finishes reading the files.
pub struct PreparedPrompt {
    /// The prompt text, with image file path references injected for any attached images
    pub prompt: String,
    /// Temp directory holding decoded image files (cleaned up on drop)
    pub image_dir: Option<tempfile::TempDir>,
}

/// File extension for a given MIME type
fn mime_to_extension(mime_type: &str) -> &str {
    match mime_type {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "bin",
    }
}

/// Decode a base64-encoded image and write it to a file in the given directory
fn write_image_file(
    dir: &std::path::Path,
    image: &ImagePart,
    index: usize,
) -> Result<PathBuf, RunnerError> {
    let ext = mime_to_extension(&image.mime_type);
    let path = dir.join(format!("{index}.{ext}"));

    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&image.data)
        .map_err(|e| RunnerError::internal(format!("failed to decode base64 image data: {e}")))?;

    let mut file = std::fs::File::create(&path)
        .map_err(|e| RunnerError::internal(format!("failed to create temp image file: {e}")))?;
    file.write_all(&decoded)
        .map_err(|e| RunnerError::internal(format!("failed to write temp image file: {e}")))?;

    debug!(path = %path.display(), size = decoded.len(), mime = %image.mime_type, "Materialized image to temp file");
    Ok(path)
}

/// Materialize images from messages into temp files
///
/// Returns rewritten messages (with file path references appended to content)
/// and a `TempDir` handle that must be kept alive until the CLI subprocess finishes.
fn materialize_images(
    messages: &[ChatMessage],
) -> Result<(Vec<ChatMessage>, Option<tempfile::TempDir>), RunnerError> {
    let has_any_images = messages
        .iter()
        .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));

    if !has_any_images {
        return Ok((messages.to_vec(), None));
    }

    let temp_dir = tempfile::Builder::new()
        .prefix("embacle-images-")
        .tempdir()
        .map_err(|e| RunnerError::internal(format!("failed to create temp dir for images: {e}")))?;

    let mut rewritten = Vec::with_capacity(messages.len());
    let mut file_index: usize = 0;

    for msg in messages {
        let images = msg.images.as_ref().filter(|imgs| !imgs.is_empty());

        if msg.role != MessageRole::User || images.is_none() {
            rewritten.push(msg.clone());
            continue;
        }

        let images = images.expect("checked above");
        let mut file_refs = Vec::with_capacity(images.len());

        for image in images {
            let path = write_image_file(temp_dir.path(), image, file_index)?;
            file_refs.push(path);
            file_index += 1;
        }

        let mut content = msg.content.clone();
        content.push_str("\n\n[Attached images — read these files to view them]");
        for path in &file_refs {
            let _ = write!(content, "\n- {}", path.display());
        }

        let mut rewritten_msg = ChatMessage::user(content);
        // Preserve the original images field (downstream code may still want it)
        rewritten_msg.images.clone_from(&msg.images);
        rewritten.push(rewritten_msg);

        debug!(
            image_count = file_refs.len(),
            dir = %temp_dir.path().display(),
            "Materialized images for user message"
        );
    }

    Ok((rewritten, Some(temp_dir)))
}

/// Build a single prompt string from a slice of chat messages
///
/// Each message is prefixed with its role label (`[system]`, `[user]`,
/// `[assistant]`) followed by the content. Messages are separated by
/// double newlines.
#[must_use]
pub fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(messages.len());
    for msg in messages {
        let label = match msg.role {
            MessageRole::System => "[system]",
            MessageRole::User => "[user]",
            MessageRole::Assistant => "[assistant]",
            MessageRole::Tool => "[tool]",
        };
        parts.push(format!("{label}\n{}", msg.content));
    }
    let prompt = parts.join("\n\n");
    debug!(
        message_count = messages.len(),
        prompt_len = prompt.len(),
        has_system = messages.iter().any(|m| m.role == MessageRole::System),
        "Built prompt from messages"
    );
    prompt
}

/// Build a prompt with images materialized to temp files
///
/// If any user messages contain attached images, they are decoded from base64,
/// written to a temp directory, and file path references are appended to the
/// message content. The returned `PreparedPrompt` holds the temp directory
/// handle — keep it alive until the CLI subprocess finishes.
///
/// # Errors
///
/// Returns an error if image decoding or temp file creation fails.
pub fn prepare_prompt(messages: &[ChatMessage]) -> Result<PreparedPrompt, RunnerError> {
    let (rewritten, image_dir) = materialize_images(messages)?;
    let prompt = build_prompt(&rewritten);
    if image_dir.is_some() {
        debug!("Built prompt with materialized images");
    }
    Ok(PreparedPrompt { prompt, image_dir })
}

/// Build a prompt with images materialized, excluding system messages
///
/// Combines [`materialize_images`] with the system-message filtering of
/// [`build_user_prompt`]. Use when the CLI accepts a separate `--system-prompt` flag.
///
/// # Errors
///
/// Returns an error if image decoding or temp file creation fails.
pub fn prepare_user_prompt(messages: &[ChatMessage]) -> Result<PreparedPrompt, RunnerError> {
    let (rewritten, image_dir) = materialize_images(messages)?;
    let prompt = build_user_prompt(&rewritten);
    if image_dir.is_some() {
        debug!("Built user prompt with materialized images");
    }
    Ok(PreparedPrompt { prompt, image_dir })
}

/// Extract the content of the first system message, if any
#[must_use]
pub fn extract_system_message(messages: &[ChatMessage]) -> Option<&str> {
    let result = messages
        .iter()
        .find(|m| m.role == MessageRole::System)
        .map(|m| m.content.as_str());
    debug!(
        found = result.is_some(),
        len = result.map_or(0, str::len),
        "Extracting system message"
    );
    result
}

/// Build a prompt string from non-system messages only
///
/// Useful when the CLI tool accepts a separate `--system-prompt` flag
/// and the system message should not be mixed into the user prompt.
#[must_use]
pub fn build_user_prompt(messages: &[ChatMessage]) -> String {
    let non_system: Vec<&ChatMessage> = messages
        .iter()
        .filter(|m| m.role != MessageRole::System)
        .collect();

    let mut parts: Vec<String> = Vec::with_capacity(non_system.len());
    for msg in &non_system {
        let label = match msg.role {
            MessageRole::User => "[user]",
            MessageRole::Assistant => "[assistant]",
            MessageRole::Tool => "[tool]",
            MessageRole::System => unreachable!(),
        };
        parts.push(format!("{label}\n{}", msg.content));
    }
    let prompt = parts.join("\n\n");
    debug!(
        total_messages = messages.len(),
        user_messages = non_system.len(),
        prompt_len = prompt.len(),
        "Built user prompt (system messages excluded)"
    );
    prompt
}

/// Log a warning when images are present but the runner cannot process them
///
/// Called by CLI runners that do not have native vision support. The images
/// are materialized to temp files via [`prepare_prompt`] / [`prepare_user_prompt`],
/// but this warning helps with debugging when image analysis seems incomplete.
pub fn warn_images_via_tempfile(runner_name: &str, image_count: usize) {
    warn!(
        runner = runner_name,
        image_count, "Images materialized to temp files for CLI runner (no native vision support)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_single_user_message() {
        let messages = vec![ChatMessage::user("Hello")];
        assert_eq!(build_prompt(&messages), "[user]\nHello");
    }

    #[test]
    fn test_build_prompt_multi_role_conversation() {
        let messages = vec![
            ChatMessage::system("Be concise"),
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("A systems language."),
        ];
        let result = build_prompt(&messages);
        assert_eq!(
            result,
            "[system]\nBe concise\n\n[user]\nWhat is Rust?\n\n[assistant]\nA systems language."
        );
    }

    #[test]
    fn test_build_prompt_empty_messages() {
        let messages: Vec<ChatMessage> = Vec::new();
        assert_eq!(build_prompt(&messages), "");
    }

    #[test]
    fn test_extract_system_message_present() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hi"),
        ];
        assert_eq!(extract_system_message(&messages), Some("You are helpful"));
    }

    #[test]
    fn test_extract_system_message_absent() {
        let messages = vec![ChatMessage::user("Hi")];
        assert_eq!(extract_system_message(&messages), None);
    }

    #[test]
    fn test_extract_system_message_returns_first() {
        let messages = vec![ChatMessage::system("First"), ChatMessage::system("Second")];
        assert_eq!(extract_system_message(&messages), Some("First"));
    }

    #[test]
    fn test_build_user_prompt_excludes_system() {
        let messages = vec![
            ChatMessage::system("System instructions"),
            ChatMessage::user("User question"),
            ChatMessage::assistant("Response"),
        ];
        let result = build_user_prompt(&messages);
        assert_eq!(result, "[user]\nUser question\n\n[assistant]\nResponse");
        assert!(!result.contains("[system]"));
    }

    #[test]
    fn test_build_user_prompt_only_system_messages() {
        let messages = vec![ChatMessage::system("Only system")];
        assert_eq!(build_user_prompt(&messages), "");
    }

    #[test]
    fn test_prepare_prompt_no_images() {
        let messages = vec![ChatMessage::user("Hello")];
        let prepared = prepare_prompt(&messages).unwrap();
        assert_eq!(prepared.prompt, "[user]\nHello");
        assert!(prepared.image_dir.is_none());
    }

    #[test]
    fn test_prepare_prompt_with_images() {
        // 1x1 red PNG pixel as base64
        let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        let image = ImagePart::new(png_b64, "image/png").unwrap();
        let messages = vec![ChatMessage::user_with_images("Describe this", vec![image])];

        let prepared = prepare_prompt(&messages).unwrap();
        assert!(prepared.prompt.contains("Describe this"));
        assert!(prepared.prompt.contains("[Attached images"));
        assert!(prepared.prompt.contains(".png"));
        assert!(prepared.image_dir.is_some());

        // Verify the temp file exists and has valid PNG data
        let dir = prepared.image_dir.as_ref().unwrap();
        let image_file = dir.path().join("0.png");
        assert!(image_file.exists());
        let data = std::fs::read(&image_file).unwrap();
        assert_eq!(&data[..4], b"\x89PNG");
    }

    #[test]
    fn test_prepare_user_prompt_with_images_excludes_system() {
        let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        let image = ImagePart::new(png_b64, "image/png").unwrap();
        let messages = vec![
            ChatMessage::system("System prompt"),
            ChatMessage::user_with_images("Look at this", vec![image]),
        ];

        let prepared = prepare_user_prompt(&messages).unwrap();
        assert!(!prepared.prompt.contains("[system]"));
        assert!(prepared.prompt.contains("Look at this"));
        assert!(prepared.prompt.contains("[Attached images"));
    }

    #[test]
    fn test_prepare_prompt_multiple_images() {
        let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
        let img1 = ImagePart::new(png_b64, "image/png").unwrap();
        let img2 = ImagePart::new(png_b64, "image/jpeg").unwrap();
        let messages = vec![ChatMessage::user_with_images(
            "Two images",
            vec![img1, img2],
        )];

        let prepared = prepare_prompt(&messages).unwrap();
        assert!(prepared.prompt.contains("0.png"));
        assert!(prepared.prompt.contains("1.jpg"));
    }

    #[test]
    fn test_prepare_prompt_assistant_images_ignored() {
        // Only user messages should have images materialized
        let mut msg = ChatMessage::assistant("Response");
        msg.images = Some(vec![ImagePart::new(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "image/png",
        )
        .unwrap()]);
        let messages = vec![msg];

        let prepared = prepare_prompt(&messages).unwrap();
        assert!(!prepared.prompt.contains("[Attached images"));
        // TempDir is still created because has_any_images checks all messages
        // but no files are written for non-user messages
    }

    #[test]
    fn test_mime_to_extension() {
        assert_eq!(mime_to_extension("image/png"), "png");
        assert_eq!(mime_to_extension("image/jpeg"), "jpg");
        assert_eq!(mime_to_extension("image/webp"), "webp");
        assert_eq!(mime_to_extension("image/gif"), "gif");
        assert_eq!(mime_to_extension("image/bmp"), "bin");
    }
}
