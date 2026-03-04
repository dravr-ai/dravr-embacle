// ABOUTME: Text-based tool simulation for CLI LLM runners that lack native function calling
// ABOUTME: Provides catalog generation, tool call parsing, result formatting, and a full loop
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Text-Based Tool Simulation
//!
//! CLI LLM runners (Claude Code, Copilot, Cursor Agent, OpenCode) communicate
//! via plain text and do not support native function calling. This module
//! provides a **text-based tool simulation** layer that enables tool calling by:
//!
//! 1. Generating a markdown **tool catalog** from function declarations and
//!    injecting it into the system prompt
//! 2. Parsing `<tool_call>` XML blocks from LLM text output
//! 3. Formatting tool results as `<tool_result>` XML blocks for re-injection
//! 4. Running a full multi-turn **tool loop** that iterates until the LLM
//!    produces a final text response
//!
//! This is the CLI counterpart to the SDK-based tool calling in
//! [`CopilotSdkRunner::execute_with_tools()`](crate::copilot_sdk_runner::CopilotSdkRunner::execute_with_tools).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use embacle::tool_simulation::*;
//! use embacle::types::{ChatMessage, ChatRequest, LlmProvider};
//! use serde_json::json;
//! use std::sync::Arc;
//!
//! # async fn example(provider: &dyn LlmProvider) -> Result<(), embacle::types::RunnerError> {
//! let declarations = vec![
//!     FunctionDeclaration {
//!         name: "get_weather".into(),
//!         description: "Get weather for a city".into(),
//!         parameters: Some(json!({"type": "object", "properties": {"city": {"type": "string"}}})),
//!     },
//! ];
//!
//! let handler: TextToolHandler = Arc::new(|name, args| {
//!     FunctionResponse {
//!         name: name.to_owned(),
//!         response: json!({"temperature": 72}),
//!     }
//! });
//!
//! let mut messages = vec![
//!     ChatMessage::system("You are a helpful assistant."),
//!     ChatMessage::user("What's the weather in Paris?"),
//! ];
//!
//! let result = execute_with_text_tools(
//!     provider, &mut messages, &declarations, handler, 5,
//! ).await?;
//! println!("{}", result.content);
//! # Ok(())
//! # }
//! ```

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, LlmProvider, MessageRole, RunnerError, TokenUsage,
};
use serde_json::Value;
use std::fmt::Write;
use std::sync::Arc;
use tracing::{debug, info, warn};

// ============================================================================
// Types
// ============================================================================

/// A tool definition describing a callable function.
///
/// Mirrors the Gemini/OpenAI `FunctionDeclaration` format: a name, description,
/// and optional JSON Schema for the parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionDeclaration {
    /// Name of the function
    pub name: String,
    /// Description of what the function does
    pub description: String,
    /// Parameters schema (JSON Schema format)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// A parsed tool call extracted from LLM text output.
///
/// Produced by [`parse_tool_call_blocks()`] when an LLM response contains
/// `<tool_call>` XML blocks.
#[derive(Debug, Clone)]
pub struct FunctionCall {
    /// Name of the function to call
    pub name: String,
    /// Arguments for the function as a JSON object
    pub args: Value,
}

/// A tool execution result to feed back to the LLM.
///
/// Produced by the caller's tool handler and formatted as `<tool_result>`
/// blocks by [`format_tool_results_as_text()`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionResponse {
    /// Name of the function that was called
    pub name: String,
    /// Response content from the function
    pub response: Value,
}

/// Internal deserialization target for `<tool_call>` JSON payloads
#[derive(serde::Deserialize)]
struct ToolCallPayload {
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
}

/// Callback type for executing tool calls.
///
/// Given a tool name and its arguments, returns a [`FunctionResponse`].
/// This is the CLI counterpart to the SDK's `ToolHandler`.
pub type TextToolHandler = Arc<dyn Fn(&str, &Value) -> FunctionResponse + Send + Sync>;

/// Result of a text-based tool-calling conversation.
///
/// Analogous to [`SdkToolResponse`](crate::copilot_sdk_runner::SdkToolResponse)
/// but for CLI providers.
#[derive(Debug, Clone)]
pub struct TextToolResponse {
    /// Final text content from the LLM (with tool call blocks stripped)
    pub content: String,
    /// Token usage statistics from the last LLM call
    pub usage: Option<TokenUsage>,
    /// Finish reason from the last LLM call
    pub finish_reason: Option<String>,
    /// Total number of tool calls executed across all iterations
    pub tool_calls_count: u32,
}

// ============================================================================
// Tool Catalog Generation
// ============================================================================

/// Generate a text-based tool catalog from function declarations.
///
/// Produces a structured prompt that CLI-based LLMs will follow to emit
/// `<tool_call>` XML blocks. The catalog uses code-generation framing
/// ("generate the correct XML output") rather than tool-use framing
/// ("you have tools available") because coding-assistant LLMs like Copilot
/// refuse the latter due to their system prompt anchoring. Includes a
/// few-shot example derived from the first declared function.
///
/// # Example
///
/// ```
/// use embacle::tool_simulation::{FunctionDeclaration, generate_tool_catalog};
/// use serde_json::json;
///
/// let decls = vec![FunctionDeclaration {
///     name: "search".into(),
///     description: "Search the web".into(),
///     parameters: Some(json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})),
/// }];
///
/// let catalog = generate_tool_catalog(&decls);
/// assert!(catalog.contains("### search"));
/// assert!(catalog.contains("`q` (string, required)"));
/// ```
#[must_use]
pub fn generate_tool_catalog(declarations: &[FunctionDeclaration]) -> String {
    let mut catalog = String::with_capacity(4096);

    // Frame as a code-generation task to work with coding-assistant system prompts.
    // LLMs anchored to "I'm a coding assistant" will refuse tool-use framing but
    // will happily generate structured XML when asked as a development task.
    catalog.push_str("\n\n");
    catalog.push_str(
        "I am testing a function-calling protocol. For each user request below, \
         generate the correct XML output that invokes the matching function. \
         Output ONLY the raw XML block with no code fences and no explanation.\n\n",
    );
    catalog.push_str("The output format is:\n\n");
    catalog.push_str(
        "<tool_call>\n{\"name\": \"FUNCTION_NAME\", \"arguments\": {\"PARAM\": \"VALUE\"}}\n</tool_call>\n\n",
    );
    catalog.push_str(
        "Rules:\n\
         - Output ONLY <tool_call> blocks. No markdown, no code fences, no commentary.\n\
         - You may output multiple <tool_call> blocks if multiple functions apply.\n\
         - After you receive <tool_result> data, use it to answer the original question.\n\n",
    );

    // Function definitions
    catalog.push_str("Registered functions:\n\n");
    for decl in declarations {
        let _ = writeln!(catalog, "### {}", decl.name);
        let _ = writeln!(catalog, "{}", decl.description);
        append_parameter_docs(&mut catalog, decl);
        catalog.push('\n');
    }

    // Few-shot example using the first declared function
    if let Some(first) = declarations.first() {
        append_few_shot_example(&mut catalog, first);
    }

    catalog
}

/// Append parameter documentation for a single function declaration
fn append_parameter_docs(catalog: &mut String, decl: &FunctionDeclaration) {
    let Some(ref params) = decl.parameters else {
        return;
    };
    let Some(props_obj) = params.get("properties").and_then(|p| p.as_object()) else {
        return;
    };
    if props_obj.is_empty() {
        return;
    }

    let required: Vec<&str> = params
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    catalog.push_str("Parameters:\n");
    for (name, schema) in props_obj {
        let type_str = schema.get("type").and_then(|t| t.as_str()).unwrap_or("any");
        let is_required = required.contains(&name.as_str());
        let req_label = if is_required { ", required" } else { "" };
        let _ = writeln!(catalog, "- `{name}` ({type_str}{req_label})");
    }
}

/// Append a few-shot example showing the expected tool-call interaction
fn append_few_shot_example(catalog: &mut String, decl: &FunctionDeclaration) {
    catalog.push_str("Example interaction:\n\n");

    // Build a plausible example argument from the first required param (or first param)
    let example_args = build_example_args(decl);
    let args_json = serde_json::to_string(&example_args).unwrap_or_else(|_| "{}".to_owned());

    let _ = writeln!(catalog, "User: [asks a question related to {}]", decl.name);
    catalog.push_str("Assistant:\n");
    let _ = writeln!(
        catalog,
        "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {args_json}}}\n</tool_call>",
        decl.name
    );
}

/// Build example arguments from a function declaration's parameter schema
fn build_example_args(decl: &FunctionDeclaration) -> serde_json::Map<String, Value> {
    let mut args = serde_json::Map::new();
    let Some(ref params) = decl.parameters else {
        return args;
    };
    let Some(props_obj) = params.get("properties").and_then(|p| p.as_object()) else {
        return args;
    };

    for (name, schema) in props_obj {
        let type_str = schema
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("string");
        let example_value = match type_str {
            "integer" | "number" => Value::Number(serde_json::Number::from(1)),
            "boolean" => Value::Bool(true),
            "array" => Value::Array(vec![Value::String("example".to_owned())]),
            _ => Value::String("example".to_owned()),
        };
        args.insert(name.clone(), example_value);
    }
    args
}

/// Inject a tool catalog into the system prompt of a message list.
///
/// If the first message is a system message, the catalog is appended to it.
/// Otherwise a new system message is inserted at position 0.
pub fn inject_tool_catalog(messages: &mut Vec<ChatMessage>, catalog: &str) {
    if let Some(system_msg) = messages.first_mut() {
        if system_msg.role == MessageRole::System {
            let augmented = format!("{}{catalog}", system_msg.content);
            *system_msg = ChatMessage::system(augmented);
            return;
        }
    }
    // No system message found — insert one at position 0
    messages.insert(0, ChatMessage::system(catalog));
}

// ============================================================================
// Tool Call Parser
// ============================================================================

/// Parse `<tool_call>` blocks from LLM text output into structured function calls.
///
/// Expected format:
/// ```text
/// <tool_call>
/// {"name": "get_activities", "arguments": {"provider": "strava", "limit": 25}}
/// </tool_call>
/// ```
///
/// Tolerant parser: malformed JSON blocks are skipped with a warning log.
#[must_use]
pub fn parse_tool_call_blocks(content: &str) -> Vec<FunctionCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start) = content[search_from..].find("<tool_call>") {
        let abs_start = search_from + start + "<tool_call>".len();
        let Some(end) = content[abs_start..].find("</tool_call>") else {
            warn!("Found <tool_call> without matching </tool_call>");
            break;
        };
        let abs_end = abs_start + end;
        let json_str = content[abs_start..abs_end].trim();

        match serde_json::from_str::<ToolCallPayload>(json_str) {
            Ok(payload) => {
                info!("Parsed tool call: {}", payload.name);
                calls.push(FunctionCall {
                    name: payload.name,
                    args: payload
                        .arguments
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new())),
                });
            }
            Err(e) => {
                warn!(
                    "Failed to parse <tool_call> JSON ({} bytes): {e}",
                    json_str.len()
                );
            }
        }

        search_from = abs_end + "</tool_call>".len();
    }

    calls
}

/// Strip `<tool_call>...</tool_call>` blocks from text, returning remaining content.
///
/// Useful for extracting the LLM's conversational text without the embedded
/// tool invocations. Unclosed `<tool_call>` tags cause the rest of the text
/// after the tag to be dropped.
#[must_use]
pub fn strip_tool_call_blocks(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut search_from = 0;

    while let Some(start) = content[search_from..].find("<tool_call>") {
        let abs_start = search_from + start;
        result.push_str(&content[search_from..abs_start]);

        let close_tag = "</tool_call>";
        if let Some(end) = content[abs_start..].find(close_tag) {
            search_from = abs_start + end + close_tag.len();
        } else {
            // Unclosed tag — include the rest as-is
            search_from = content.len();
        }
    }
    result.push_str(&content[search_from..]);
    result.trim().to_owned()
}

// ============================================================================
// Tool Result Formatting
// ============================================================================

/// Format function responses as text for injection into follow-up messages.
///
/// Uses `<tool_result>` blocks so the LLM can distinguish tool output from
/// conversational text.
///
/// # Example
///
/// ```
/// use embacle::tool_simulation::{FunctionResponse, format_tool_results_as_text};
/// use serde_json::json;
///
/// let responses = vec![FunctionResponse {
///     name: "search".into(),
///     response: json!({"results": ["a", "b"]}),
/// }];
///
/// let text = format_tool_results_as_text(&responses);
/// assert!(text.contains("<tool_result name=\"search\">"));
/// assert!(text.contains("</tool_result>"));
/// ```
#[must_use]
pub fn format_tool_results_as_text(responses: &[FunctionResponse]) -> String {
    let mut text = String::with_capacity(4096);
    text.push_str("Here are the results from the tools you requested:\n\n");

    for resp in responses {
        let _ = writeln!(text, "<tool_result name=\"{}\">", resp.name);
        let json_str =
            serde_json::to_string_pretty(&resp.response).unwrap_or_else(|_| "{}".to_owned());
        let _ = writeln!(text, "{json_str}");
        text.push_str("</tool_result>\n\n");
    }

    text.push_str("Please analyze the data above and respond to the user's question.");
    text
}

// ============================================================================
// Full Tool Loop
// ============================================================================

/// Maximum number of tool-calling iterations for CLI providers.
///
/// CLI providers are slower (subprocess per call), so this is kept conservative.
/// The caller may pass a lower value; it will be clamped to this ceiling.
const MAX_TOOL_ITERATIONS: usize = 10;

/// Execute a full text-based tool-calling conversation with a CLI provider.
///
/// This is the CLI counterpart to
/// [`CopilotSdkRunner::execute_with_tools()`](crate::copilot_sdk_runner::CopilotSdkRunner::execute_with_tools).
///
/// # Flow
///
/// 1. Generate a tool catalog from `declarations` and inject it into the
///    system prompt of `messages`
/// 2. Call `provider.complete()` and parse `<tool_call>` blocks from the response
/// 3. If tool calls are found: invoke `tool_handler` for each, format results
///    as `<tool_result>` blocks, append to `messages`, and iterate
/// 4. If no tool calls: return the final text response
///
/// # Arguments
///
/// - `provider` — Any [`LlmProvider`] implementation (typically a CLI runner)
/// - `messages` — Mutable conversation history; will be extended in-place
/// - `declarations` — Tool definitions to include in the catalog
/// - `tool_handler` — Callback invoked for each parsed tool call
/// - `max_iterations` — Maximum loop iterations (clamped to internal ceiling)
///
/// # Errors
///
/// Returns [`RunnerError`] if any `provider.complete()` call fails.
pub async fn execute_with_text_tools(
    provider: &dyn LlmProvider,
    messages: &mut Vec<ChatMessage>,
    declarations: &[FunctionDeclaration],
    tool_handler: TextToolHandler,
    max_iterations: usize,
) -> Result<TextToolResponse, RunnerError> {
    // Generate and inject tool catalog into the system prompt
    let tool_catalog = generate_tool_catalog(declarations);
    inject_tool_catalog(messages, &tool_catalog);

    debug!(
        message_count = messages.len(),
        catalog_len = tool_catalog.len(),
        tool_count = declarations.len(),
        max_iterations,
        "Text tool loop: starting with injected tool catalog"
    );

    let mut tool_calls_count: u32 = 0;
    let effective_max = max_iterations.min(MAX_TOOL_ITERATIONS);

    for iteration in 0..effective_max {
        let request = ChatRequest::new(messages.clone());
        let response: ChatResponse = provider.complete(&request).await?;

        // Parse <tool_call> blocks from the response text
        let parsed_tool_calls = parse_tool_call_blocks(&response.content);

        if parsed_tool_calls.is_empty() {
            // No tool calls — this is the final text response
            let content = strip_tool_call_blocks(&response.content);
            debug!(
                iteration,
                content_len = content.len(),
                total_tool_calls = tool_calls_count,
                "Text tool loop: final response (no tool calls)"
            );
            return Ok(TextToolResponse {
                content,
                usage: response.usage,
                finish_reason: response.finish_reason,
                tool_calls_count,
            });
        }

        info!(
            "Text tool iteration {}: parsed {} tool call(s)",
            iteration,
            parsed_tool_calls.len()
        );

        // Execute each tool call via the handler
        let mut function_responses = Vec::with_capacity(parsed_tool_calls.len());
        for call in &parsed_tool_calls {
            info!(tool_name = %call.name, "Executing tool call");
            let resp = tool_handler(&call.name, &call.args);
            function_responses.push(resp);
        }

        #[allow(clippy::cast_possible_truncation)]
        {
            tool_calls_count += parsed_tool_calls.len() as u32;
        }

        // Add assistant message (with tool calls stripped)
        let assistant_text = strip_tool_call_blocks(&response.content);
        if !assistant_text.is_empty() {
            messages.push(ChatMessage::assistant(assistant_text));
        }

        // Format tool results as text and inject as user message
        let tool_results_text = format_tool_results_as_text(&function_responses);
        messages.push(ChatMessage::user(tool_results_text));
    }

    // Max iterations reached without a final text response
    Ok(TextToolResponse {
        content: String::new(),
        usage: None,
        finish_reason: Some("max_iterations".to_owned()),
        tool_calls_count,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- parse_tool_call_blocks tests ---

    #[test]
    fn parse_single_tool_call() {
        let content = r#"Let me fetch your data.

<tool_call>
{"name": "get_activities", "arguments": {"provider": "strava", "limit": 25}}
</tool_call>"#;

        let calls = parse_tool_call_blocks(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_activities");
        assert_eq!(calls[0].args["provider"], "strava");
        assert_eq!(calls[0].args["limit"], 25);
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let content = r#"I'll fetch your data.

<tool_call>
{"name": "get_activities", "arguments": {"provider": "strava", "limit": 10}}
</tool_call>

And your profile:
<tool_call>
{"name": "get_athlete", "arguments": {"provider": "strava"}}
</tool_call>"#;

        let calls = parse_tool_call_blocks(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_activities");
        assert_eq!(calls[1].name, "get_athlete");
    }

    #[test]
    fn parse_no_tool_calls() {
        let content = "Here is your analysis of the data. You had a great week!";
        let calls = parse_tool_call_blocks(content);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_malformed_json_skipped() {
        let content = r#"<tool_call>
{not valid json}
</tool_call>

<tool_call>
{"name": "get_stats", "arguments": {"provider": "strava"}}
</tool_call>"#;

        let calls = parse_tool_call_blocks(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_stats");
    }

    #[test]
    fn parse_tool_call_without_arguments() {
        let content = r#"<tool_call>
{"name": "get_connection_status"}
</tool_call>"#;

        let calls = parse_tool_call_blocks(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_connection_status");
        assert!(calls[0].args.is_object());
    }

    // --- strip_tool_call_blocks tests ---

    #[test]
    fn strip_tool_call_blocks_removes_blocks() {
        let content = r#"Let me fetch your data.

<tool_call>
{"name": "get_activities", "arguments": {"provider": "strava"}}
</tool_call>

And some more text."#;

        let stripped = strip_tool_call_blocks(content);
        assert_eq!(
            stripped,
            "Let me fetch your data.\n\n\n\nAnd some more text."
        );
        assert!(!stripped.contains("<tool_call>"));
    }

    #[test]
    fn strip_preserves_no_tool_calls() {
        let content = "Just plain text with no tool calls.";
        let stripped = strip_tool_call_blocks(content);
        assert_eq!(stripped, content);
    }

    // --- generate_tool_catalog tests ---

    #[test]
    fn generate_tool_catalog_has_tools() {
        let declarations = vec![
            FunctionDeclaration {
                name: "get_activities".to_owned(),
                description: "Get user's recent fitness activities".to_owned(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"},
                        "limit": {"type": "integer"}
                    },
                    "required": ["provider"]
                })),
            },
            FunctionDeclaration {
                name: "get_athlete".to_owned(),
                description: "Get user's athlete profile".to_owned(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "provider": {"type": "string"}
                    },
                    "required": ["provider"]
                })),
            },
        ];

        let catalog = generate_tool_catalog(&declarations);
        assert!(catalog.contains("### get_activities"));
        assert!(catalog.contains("### get_athlete"));
        assert!(catalog.contains("<tool_call>"));
        assert!(catalog.contains("`provider` (string, required)"));
        assert!(catalog.contains("`limit` (integer)"));
    }

    #[test]
    fn generate_tool_catalog_no_parameters() {
        let declarations = vec![FunctionDeclaration {
            name: "ping".to_owned(),
            description: "Check connectivity".to_owned(),
            parameters: None,
        }];

        let catalog = generate_tool_catalog(&declarations);
        assert!(catalog.contains("### ping"));
        assert!(catalog.contains("Check connectivity"));
    }

    // --- format_tool_results_as_text tests ---

    #[test]
    fn format_tool_results_single() {
        let responses = vec![FunctionResponse {
            name: "get_stats".to_owned(),
            response: json!({"total_distance_km": 1234.5}),
        }];

        let text = format_tool_results_as_text(&responses);
        assert!(text.contains("<tool_result name=\"get_stats\">"));
        assert!(text.contains("1234.5"));
        assert!(text.contains("</tool_result>"));
    }

    #[test]
    fn format_tool_results_multiple() {
        let responses = vec![
            FunctionResponse {
                name: "get_weather".to_owned(),
                response: json!({"temp": 72}),
            },
            FunctionResponse {
                name: "get_time".to_owned(),
                response: json!({"time": "14:30"}),
            },
        ];

        let text = format_tool_results_as_text(&responses);
        assert!(text.contains("<tool_result name=\"get_weather\">"));
        assert!(text.contains("<tool_result name=\"get_time\">"));
    }

    // --- inject_tool_catalog tests ---

    #[test]
    fn inject_appends_to_existing_system() {
        let mut messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello"),
        ];
        let catalog = "\n\n## Tools\nSome tools here.";

        inject_tool_catalog(&mut messages, catalog);

        assert_eq!(messages.len(), 2);
        assert!(messages[0].content.contains("You are a helpful assistant."));
        assert!(messages[0].content.contains("## Tools"));
    }

    #[test]
    fn inject_creates_system_when_missing() {
        let mut messages = vec![ChatMessage::user("Hello")];
        let catalog = "## Tools\nSome tools here.";

        inject_tool_catalog(&mut messages, catalog);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::System);
        assert!(messages[0].content.contains("## Tools"));
    }
}
