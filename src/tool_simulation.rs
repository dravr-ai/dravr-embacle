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
//! This is the CLI counterpart to the SDK-managed tool calling in
//! `CopilotHeadlessRunner` (requires `copilot-headless` feature).
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
    ToolCallRequest, ToolDefinition,
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
/// This is a type alias for [`ToolDefinition`] from core types, maintaining
/// backward compatibility with existing code that uses `FunctionDeclaration`.
pub type FunctionDeclaration = ToolDefinition;

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

impl From<ToolCallRequest> for FunctionCall {
    fn from(tc: ToolCallRequest) -> Self {
        Self {
            name: tc.function_name,
            args: tc.arguments,
        }
    }
}

impl From<FunctionCall> for ToolCallRequest {
    fn from(fc: FunctionCall) -> Self {
        Self {
            id: format!("call_{}", fc.name),
            function_name: fc.name,
            arguments: fc.args,
        }
    }
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
/// Analogous to [`HeadlessToolResponse`](crate::copilot_headless::HeadlessToolResponse)
/// (requires `copilot-headless` feature) but for CLI providers.
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

    // Present the tools as a natural capability the assistant already has, not as
    // an adversarial "I am testing a function-calling protocol / output ONLY XML"
    // instruction. Reasoning-tuned models (Copilot-served Claude Sonnet from
    // 2026-07 onward) read the old framing — plus the "Registered functions" label
    // — as an injected jailbreak and refuse the whole turn, breaking character
    // ("I'm GitHub Copilot, not <persona>"). Proven by A/B against live Copilot:
    // an identical poisoned conversation history refuses under the old framing and
    // stays fully in role under this one. The <tool_call> wire format is unchanged
    // so parse_tool_call_blocks / strip_simulation_artifacts keep working verbatim.
    catalog.push_str("\n\n");
    catalog.push_str(
        "You have access to the tools listed below to help with the user's \
         request. When a tool would help, call it by emitting a block in exactly \
         this format:\n\n",
    );
    catalog.push_str(
        "<tool_call>\n{\"name\": \"FUNCTION_NAME\", \"arguments\": {\"PARAM\": \"VALUE\"}}\n</tool_call>\n\n",
    );
    catalog.push_str(
        "Notes:\n\
         - Emit a <tool_call> block whenever you need data or an action a tool \
         provides; you may emit several blocks if more than one tool applies.\n\
         - Only call the tools listed under \"Available tools\" below. Other tools \
         (Glob, Grep, Read, Bash, Edit, Write, etc.) are not available in this environment.\n\
         - After each call you receive a <tool_result> block; use its data to \
         answer the user.\n\n",
    );

    // Function definitions
    catalog.push_str("Available tools:\n\n");
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

/// Deepest schema-node recursion when documenting a parameter.
///
/// A guard against a pathological or self-referential schema inflating the
/// system prompt without bound, not a working limit. Note it counts schema
/// nodes, and stepping from an array to its `items` costs a level, so the
/// deepest real schema — `weeks` → week → `days` → day → day fields — sits at
/// five, comfortably inside this.
const MAX_PARAM_DEPTH: usize = 8;

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

    catalog.push_str("Parameters:\n");
    append_property_lines(catalog, params, 0);
}

/// Render one object schema's `properties` as bullet lines, recursing into
/// nested objects and into the item schema of arrays of objects.
///
/// Without the recursion a nested parameter reached the model as nothing but
/// its top-level name and type — `` `outline` (object) `` — leaving every inner
/// field name invisible. On providers without native function calling the
/// catalog is the *only* schema the model ever sees, so it had to guess the
/// nested shape and guessed wrong every time: `save_training_plan` failed 24 of
/// 24 live calls (2026-07-12 → 2026-07-28) while every flat-schema tool
/// succeeded, the model sending `week_label`/`day`/`session` against a schema
/// wanting `week_start`/`date`/`sport`/`workout`.
///
/// A parameter expands only when it genuinely has nested fields, which keeps
/// the rendering of a purely scalar schema byte-identical to the pre-recursion
/// output — tools that already worked gain neither a token nor a risk.
fn append_property_lines(catalog: &mut String, schema: &Value, depth: usize) {
    let Some(props_obj) = schema.get("properties").and_then(|p| p.as_object()) else {
        return;
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let indent = "  ".repeat(depth);
    for (name, prop) in props_obj {
        let type_str = prop.get("type").and_then(|t| t.as_str()).unwrap_or("any");
        let is_required = required.contains(&name.as_str());
        let req_label = if is_required { ", required" } else { "" };

        let nested = nested_object_schema(prop).filter(|_| depth + 1 < MAX_PARAM_DEPTH);

        // Describe every field the recursion reveals — whether it expands
        // further or is a leaf. The format hints that make a nested field
        // fillable at all live on the leaves ("Race date, YYYY-MM-DD.", "One
        // of: rest | base | build | peak | taper."); a bare `date (string)`
        // is exactly what let the model send a French week label.
        //
        // A top-level scalar stays bare, which is what keeps a schema with no
        // nesting rendering byte-for-byte as it did before the recursion — the
        // tools already succeeding on every call gain neither token nor risk.
        let describe = depth > 0 || nested.is_some();
        let description = prop
            .get("description")
            .and_then(|d| d.as_str())
            .filter(|_| describe)
            .map_or_else(String::new, |d| format!(" — {d}"));
        let label = if nested.is_some() && type_str == "array" {
            "array of object"
        } else {
            type_str
        };
        let _ = writeln!(
            catalog,
            "{indent}- `{name}` ({label}{req_label}){description}"
        );

        if let Some(inner) = nested {
            append_property_lines(catalog, inner, depth + 1);
        }
    }
}

/// The object schema a parameter expands into — itself when it is an object
/// carrying `properties`, or its `items` when it is an array of such objects.
///
/// `None` for a scalar or an array of scalars, which have no inner field names
/// to reveal and so keep their single-line rendering.
fn nested_object_schema(prop: &Value) -> Option<&Value> {
    let has_fields = |v: &Value| {
        v.get("properties")
            .and_then(|p| p.as_object())
            .is_some_and(|p| !p.is_empty())
    };
    if has_fields(prop) {
        return Some(prop);
    }
    prop.get("items").filter(|items| has_fields(items))
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
    let Some(ref params) = decl.parameters else {
        return serde_json::Map::new();
    };
    match example_for_schema(params, 0) {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    }
}

/// Build a shape-correct example value for one schema node.
///
/// Recurses so a nested parameter is demonstrated as the object it actually is.
/// The pre-recursion generator emitted `"outline": "example"` for an
/// object-typed parameter — a worked example contradicting the very shape the
/// model is being asked to produce, at the moment it is learning the format.
fn example_for_schema(schema: &Value, depth: usize) -> Value {
    let type_str = schema
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("string");
    match type_str {
        "object" => {
            let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
                return Value::Object(serde_json::Map::new());
            };
            if depth + 1 >= MAX_PARAM_DEPTH {
                return Value::Object(serde_json::Map::new());
            }
            let mut map = serde_json::Map::new();
            for (name, prop) in props {
                map.insert(name.clone(), example_for_schema(prop, depth + 1));
            }
            Value::Object(map)
        }
        "array" => match schema.get("items") {
            Some(items) if depth + 1 < MAX_PARAM_DEPTH => {
                Value::Array(vec![example_for_schema(items, depth + 1)])
            }
            // No item schema to follow: the original placeholder element.
            _ => Value::Array(vec![Value::String("example".to_owned())]),
        },
        "integer" | "number" => Value::Number(serde_json::Number::from(1)),
        "boolean" => Value::Bool(true),
        _ => Value::String("example".to_owned()),
    }
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

/// Literal preamble [`format_tool_results_as_text`] prepends to the injected
/// tool-result turn. Defined as a constant so [`strip_tool_result_echo`] can
/// remove it verbatim — the formatter and the stripper MUST stay in lockstep,
/// otherwise an echoed preamble leaks to the user.
const TOOL_RESULTS_PREAMBLE: &str = "Here are the results from the tools you requested:";

/// Literal trailing instruction [`format_tool_results_as_text`] appends to the
/// injected tool-result turn. Stripped by [`strip_tool_result_echo`] for the
/// same lockstep reason as [`TOOL_RESULTS_PREAMBLE`].
const TOOL_RESULTS_FOOTER: &str =
    "Please analyze the data above and respond to the user's question.";

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
    text.push_str(TOOL_RESULTS_PREAMBLE);
    text.push_str("\n\n");

    for resp in responses {
        let _ = writeln!(text, "<tool_result name=\"{}\">", resp.name);
        let json_str =
            serde_json::to_string_pretty(&resp.response).unwrap_or_else(|_| "{}".to_owned());
        let _ = writeln!(text, "{json_str}");
        text.push_str("</tool_result>\n\n");
    }

    text.push_str(TOOL_RESULTS_FOOTER);
    text
}

/// Strip echoed tool-result scaffolding from a model's reply.
///
/// In CLI/simulation mode, tool outputs are fed back to the model as a
/// synthetic user turn built by [`format_tool_results_as_text`] — a
/// [`TOOL_RESULTS_PREAMBLE`] line, one or more `<tool_result>...</tool_result>`
/// blocks, and a [`TOOL_RESULTS_FOOTER`] line. Weaker CLI models sometimes copy
/// that injected turn verbatim into their own output, which leaks the raw tool
/// JSON to the end user. This removes the `<tool_result>` blocks plus the
/// preamble/footer literals so only the model's own prose survives.
///
/// An unclosed `<tool_result` tag drops the remainder of the text: a half-open
/// result block is always a leaked JSON dump, never prose. (This is the inverse
/// of [`strip_tool_call_blocks`], which keeps the remainder — an unclosed
/// `<tool_call>` is a malformed invocation the model may have prefaced with
/// real text, whereas an unclosed `<tool_result>` is pure scaffolding.)
#[must_use]
pub fn strip_tool_result_echo(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut search_from = 0;

    while let Some(start) = content[search_from..].find("<tool_result") {
        let abs_start = search_from + start;
        result.push_str(&content[search_from..abs_start]);

        let close_tag = "</tool_result>";
        if let Some(end) = content[abs_start..].find(close_tag) {
            search_from = abs_start + end + close_tag.len();
        } else {
            // Unclosed tag — drop the rest (it is a leaked JSON dump).
            search_from = content.len();
        }
    }
    result.push_str(&content[search_from..]);

    result
        .replace(TOOL_RESULTS_PREAMBLE, "")
        .replace(TOOL_RESULTS_FOOTER, "")
        .trim()
        .to_owned()
}

/// Strip all simulation scaffolding from a model's user-facing reply: both
/// `<tool_call>` blocks (via [`strip_tool_call_blocks`]) and echoed tool-result
/// scaffolding (via [`strip_tool_result_echo`]).
///
/// Apply this at every point where CLI/simulation output becomes the
/// user-visible answer, so neither the model's tool invocations nor any
/// parroted-back tool output reaches the end user.
#[must_use]
pub fn strip_simulation_artifacts(content: &str) -> String {
    strip_tool_result_echo(&strip_tool_call_blocks(content))
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
/// This is the CLI counterpart to the SDK-managed tool calling in
/// [`CopilotHeadlessRunner::converse()`](crate::copilot_headless::CopilotHeadlessRunner::converse).
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
            // No tool calls — this is the final text response. Strip both tool
            // calls and any echoed tool-result scaffolding so neither reaches
            // the user.
            let content = strip_simulation_artifacts(&response.content);
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

        // Add assistant message (with tool calls and any echoed tool-result
        // scaffolding stripped, so parroted output never accumulates).
        let assistant_text = strip_simulation_artifacts(&response.content);
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

    #[test]
    fn generate_tool_catalog_uses_natural_framing_not_injection_primer() {
        // Regression (2026-07 coach identity-break outage): the old preamble
        // ("I am testing a function-calling protocol / Output ONLY the raw XML /
        // Registered functions") primed reasoning-tuned models to read the turn
        // as a prompt injection and refuse in character. The catalog must present
        // tools naturally while keeping the <tool_call> wire format the parser
        // (parse_tool_call_blocks) depends on.
        let declarations = vec![FunctionDeclaration {
            name: "get_activities".to_owned(),
            description: "Get the user's recent activities".to_owned(),
            parameters: None,
        }];
        let catalog = generate_tool_catalog(&declarations);

        // Wire format + tool listing the parser depends on survive unchanged.
        assert!(catalog.contains("<tool_call>"));
        assert!(catalog.contains("### get_activities"));

        // The injection-priming framing is gone.
        assert!(!catalog.contains("I am testing"));
        assert!(!catalog.contains("function-calling protocol"));
        assert!(!catalog.contains("Registered functions"));
        assert!(!catalog.contains("Output ONLY the raw XML"));

        // Natural, in-role framing is present instead.
        assert!(catalog.contains("You have access to the tools"));
        assert!(catalog.contains("Available tools:"));
    }

    /// A `save_training_plan`-shaped declaration: an object parameter holding a
    /// nested object, and an array parameter whose items are objects that
    /// themselves hold an array of objects.
    fn nested_declaration() -> FunctionDeclaration {
        FunctionDeclaration {
            name: "save_training_plan".to_owned(),
            description: "Persist the training plan you agreed with the athlete".to_owned(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "coach_id": {"type": "string", "description": "Coach persona slug."},
                    "outline": {
                        "type": "object",
                        "description": "The plan outline.",
                        "required": ["goal_race"],
                        "properties": {
                            "goal_race": {
                                "type": "object",
                                "description": "The goal (A) race.",
                                "required": ["name", "date"],
                                "properties": {
                                    "name": {"type": "string", "description": "Race name."},
                                    "date": {"type": "string", "description": "Race date, YYYY-MM-DD."}
                                }
                            }
                        }
                    },
                    "weeks": {
                        "type": "array",
                        "description": "Day-by-day weeks to save.",
                        "items": {
                            "type": "object",
                            "required": ["week_start", "days"],
                            "properties": {
                                "week_start": {"type": "string", "description": "First day, YYYY-MM-DD."},
                                "days": {
                                    "type": "array",
                                    "description": "The day rows.",
                                    "items": {
                                        "type": "object",
                                        "required": ["date", "sport"],
                                        "properties": {
                                            "date": {"type": "string", "description": "Day date, YYYY-MM-DD."},
                                            "sport": {"type": "string", "description": "Sport or 'rest'."}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["outline"]
            })),
        }
    }

    #[test]
    fn catalog_reveals_nested_object_and_array_item_fields() {
        // Regression (2026-07-28): every inner field name was dropped, so on a
        // provider without native function calling the model never saw
        // `week_start` / `date` / `sport` and invented `week_label` / `day` /
        // `session` instead — 24 of 24 live save_training_plan calls rejected.
        let catalog = generate_tool_catalog(&[nested_declaration()]);

        // Fields one level down, through an object.
        assert!(catalog.contains("`goal_race` (object, required)"));
        assert!(catalog.contains("`name` (string, required)"));

        // Fields reached only through an array's item schema.
        assert!(catalog.contains("`week_start` (string, required)"));
        assert!(catalog.contains("`sport` (string, required)"));

        // An array of objects is labelled as such, not bare `array`.
        assert!(catalog.contains("`weeks` (array of object)"));
        assert!(catalog.contains("`days` (array of object, required)"));

        // Descriptions ride along on expanded fields: they carry the format
        // hints ("YYYY-MM-DD") without which a field name cannot be filled in.
        assert!(catalog.contains("Race date, YYYY-MM-DD."));
        assert!(catalog.contains("First day, YYYY-MM-DD."));

        // Nesting is legible as indentation.
        assert!(catalog.contains("  - `goal_race`"));
        assert!(catalog.contains("    - `date`"));
    }

    #[test]
    fn catalog_rendering_of_a_flat_schema_is_byte_identical() {
        // The recursion must be invisible to schemas that have no nesting: the
        // tools already succeeding on every call must gain neither a token nor
        // a behaviour change. Keys render in serde_json's sorted map order.
        let catalog = generate_tool_catalog(&[FunctionDeclaration {
            name: "get_activities".to_owned(),
            description: "Get the user's recent activities".to_owned(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "description": "Fitness provider to query."},
                    "limit": {"type": "integer", "description": "How many to return."}
                },
                "required": ["provider"]
            })),
        }]);

        assert!(
            catalog.contains("Parameters:\n- `limit` (integer)\n- `provider` (string, required)\n")
        );
        // Scalar parameters stay bare — no description, no indentation.
        assert!(!catalog.contains("Fitness provider to query."));
        assert!(!catalog.contains("  - `"));
    }

    #[test]
    fn few_shot_example_has_the_nested_shape_not_a_placeholder_string() {
        // The worked example is the model's most literal template. Emitting
        // `"outline": "example"` for an object parameter taught the exact
        // mistake the live failures made.
        let args = build_example_args(&nested_declaration());

        let outline = args.get("outline").expect("outline in example"); // Safe: test asserts on the declaration it just built
        assert!(
            outline.is_object(),
            "object parameter must render as an object, got {outline}"
        );
        assert!(outline
            .pointer("/goal_race/date")
            .is_some_and(Value::is_string));

        let weeks = args.get("weeks").expect("weeks in example"); // Safe: test asserts on the declaration it just built
        assert!(weeks.is_array(), "array parameter must render as an array");
        assert!(
            weeks
                .pointer("/0/days/0/sport")
                .is_some_and(Value::is_string),
            "array items must recurse into their object schema: {weeks}"
        );
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

    // --- strip_tool_result_echo tests ---

    #[test]
    fn strip_tool_result_echo_removes_full_echoed_turn() {
        // A weak CLI model parrots the entire injected turn back, then adds its
        // own analysis. Only the analysis must survive.
        let responses = vec![FunctionResponse {
            name: "get_activities".to_owned(),
            response: json!({"activities": [{"name": "Splish splash", "distance_km": 7.9}]}),
        }];
        let echoed = format!(
            "{}\n\nYour biggest ride this week was 7.9 km.",
            format_tool_results_as_text(&responses)
        );

        let stripped = strip_tool_result_echo(&echoed);
        assert_eq!(stripped, "Your biggest ride this week was 7.9 km.");
        assert!(!stripped.contains("<tool_result"));
        assert!(!stripped.contains("Here are the results"));
        assert!(!stripped.contains("Please analyze the data"));
    }

    #[test]
    fn strip_tool_result_echo_roundtrips_format_to_empty() {
        let responses = vec![
            FunctionResponse {
                name: "get_stats".to_owned(),
                response: json!({"total_distance_km": 1234.5}),
            },
            FunctionResponse {
                name: "get_athlete".to_owned(),
                response: json!({"name": "JF"}),
            },
        ];

        // Echoing the injected turn verbatim leaves nothing user-facing.
        let stripped = strip_tool_result_echo(&format_tool_results_as_text(&responses));
        assert_eq!(stripped, "");
    }

    #[test]
    fn strip_tool_result_echo_drops_unclosed_block() {
        let echoed = "Here is your data: <tool_result name=\"x\">\n{\"huge\": \"json dump";
        let stripped = strip_tool_result_echo(echoed);
        assert_eq!(stripped, "Here is your data:");
        assert!(!stripped.contains("json dump"));
    }

    #[test]
    fn strip_tool_result_echo_preserves_clean_prose() {
        let content = "Your easy run kept HR in Zone 2 — solid aerobic work.";
        assert_eq!(strip_tool_result_echo(content), content);
    }

    #[test]
    fn strip_simulation_artifacts_removes_both_scaffolds() {
        let content = "Fetching.\n\n<tool_call>\n{\"name\":\"get_activities\"}\n</tool_call>\n\n\
            Here are the results from the tools you requested:\n\n\
            <tool_result name=\"get_activities\">\n{\"x\":1}\n</tool_result>\n\n\
            Please analyze the data above and respond to the user's question.\n\n\
            You ran 5 km today.";

        let stripped = strip_simulation_artifacts(content);
        assert!(!stripped.contains("<tool_call>"));
        assert!(!stripped.contains("<tool_result"));
        assert!(!stripped.contains("Here are the results"));
        assert!(stripped.contains("Fetching."));
        assert!(stripped.contains("You ran 5 km today."));
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
