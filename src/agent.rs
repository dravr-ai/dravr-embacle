// ABOUTME: Configurable agent loop building on text-based tool simulation
// ABOUTME: Provides AgentExecutor with multi-turn tool calling, callbacks, and token accumulation
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! # Agent Loop
//!
//! [`AgentExecutor`] provides a configurable multi-turn agent loop that:
//!
//! 1. Injects a tool catalog into the conversation
//! 2. Calls `provider.complete()` and parses tool calls from the response
//! 3. Executes tools via the provided handler
//! 4. Feeds results back and repeats until no tool calls remain or
//!    `max_turns` is reached
//!
//! Builds on [`tool_simulation`](crate::tool_simulation) types and functions.
//!
//! ## Observability
//!
//! An optional [`OnTurnCallback`] is invoked after each turn with a
//! [`TurnInfo`] snapshot, enabling logging, metrics, or UI updates.

use std::sync::Arc;

use tracing::{debug, info};

use crate::tool_simulation::{
    format_tool_results_as_text, generate_tool_catalog, inject_tool_catalog,
    parse_tool_call_blocks, strip_tool_call_blocks, FunctionCall, FunctionDeclaration,
    TextToolHandler,
};
use crate::types::{ChatMessage, ChatRequest, LlmProvider, RunnerError, TokenUsage};

/// Default maximum turns for the agent loop
const DEFAULT_MAX_TURNS: u32 = 10;

/// Absolute ceiling for `max_turns` to prevent runaway loops
const MAX_TURNS_CEILING: u32 = 50;

/// Callback invoked after each agent turn for observability
pub type OnTurnCallback = Arc<dyn Fn(&TurnInfo) + Send + Sync>;

/// Information about a single agent turn, passed to [`OnTurnCallback`]
#[derive(Debug, Clone)]
pub struct TurnInfo {
    /// Turn number (1-based)
    pub turn: u32,
    /// Tool calls made during this turn
    pub tool_calls: Vec<FunctionCall>,
    /// Text content from the LLM (tool call blocks stripped)
    pub content: String,
    /// Token usage for this turn (if reported by the provider)
    pub usage: Option<TokenUsage>,
}

/// Result of an agent execution run
#[derive(Debug, Clone)]
pub struct AgentResult {
    /// Final text content from the LLM
    pub content: String,
    /// All tool calls made across all turns
    pub tool_calls: Vec<FunctionCall>,
    /// Total number of turns executed
    pub total_turns: u32,
    /// Accumulated token usage across all turns
    pub total_usage: TokenUsage,
    /// Finish reason (e.g., "stop", "`max_turns`")
    pub finish_reason: Option<String>,
}

/// Configurable agent loop with tool calling support.
///
/// # Usage
///
/// ```rust,no_run
/// # use embacle::agent::AgentExecutor;
/// # use embacle::tool_simulation::{FunctionDeclaration, FunctionResponse, TextToolHandler};
/// # use embacle::types::{ChatMessage, LlmProvider};
/// # use serde_json::json;
/// # use std::sync::Arc;
/// # async fn example(provider: &dyn LlmProvider) -> Result<(), embacle::types::RunnerError> {
/// let declarations = vec![FunctionDeclaration {
///     name: "search".into(),
///     description: "Search the web".into(),
///     parameters: Some(json!({"type": "object", "properties": {"q": {"type": "string"}}})),
/// }];
///
/// let handler: TextToolHandler = Arc::new(|name, _args| {
///     FunctionResponse { name: name.to_owned(), response: json!({"results": []}) }
/// });
///
/// let executor = AgentExecutor::new(provider, declarations, handler);
/// let messages = vec![ChatMessage::user("Search for Rust tutorials")];
/// let result = executor.run(messages).await?;
/// println!("{}", result.content);
/// # Ok(())
/// # }
/// ```
pub struct AgentExecutor<'a> {
    provider: &'a dyn LlmProvider,
    declarations: Vec<FunctionDeclaration>,
    tool_handler: TextToolHandler,
    max_turns: u32,
    on_turn: Option<OnTurnCallback>,
}

impl<'a> AgentExecutor<'a> {
    /// Create a new agent executor with default settings (`max_turns=10`)
    pub fn new(
        provider: &'a dyn LlmProvider,
        declarations: Vec<FunctionDeclaration>,
        tool_handler: TextToolHandler,
    ) -> Self {
        Self {
            provider,
            declarations,
            tool_handler,
            max_turns: DEFAULT_MAX_TURNS,
            on_turn: None,
        }
    }

    /// Set the maximum number of turns (clamped to ceiling of 50)
    pub fn with_max_turns(mut self, max_turns: u32) -> Self {
        self.max_turns = max_turns.min(MAX_TURNS_CEILING);
        self
    }

    /// Set an observability callback invoked after each turn
    pub fn with_on_turn(mut self, callback: OnTurnCallback) -> Self {
        self.on_turn = Some(callback);
        self
    }

    /// Run the agent loop with the given initial messages.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if any `provider.complete()` call fails.
    pub async fn run(
        &self,
        initial_messages: Vec<ChatMessage>,
    ) -> Result<AgentResult, RunnerError> {
        let mut messages = initial_messages;

        // Inject tool catalog into the conversation
        let catalog = generate_tool_catalog(&self.declarations);
        inject_tool_catalog(&mut messages, &catalog);

        debug!(
            tool_count = self.declarations.len(),
            max_turns = self.max_turns,
            "agent: starting loop"
        );

        let mut all_tool_calls: Vec<FunctionCall> = Vec::new();
        let mut total_usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };
        let mut turn: u32 = 0;

        loop {
            turn += 1;
            if turn > self.max_turns {
                info!(max_turns = self.max_turns, "agent: max turns reached");
                return Ok(AgentResult {
                    content: String::new(),
                    tool_calls: all_tool_calls,
                    total_turns: turn - 1,
                    total_usage,
                    finish_reason: Some("max_turns".to_owned()),
                });
            }

            let request = ChatRequest::new(messages.clone());
            let response = self.provider.complete(&request).await?;

            // Accumulate token usage
            if let Some(ref usage) = response.usage {
                total_usage.prompt_tokens += usage.prompt_tokens;
                total_usage.completion_tokens += usage.completion_tokens;
                total_usage.total_tokens += usage.total_tokens;
            }

            // Parse tool calls from the response
            let parsed_calls = parse_tool_call_blocks(&response.content);
            let content = strip_tool_call_blocks(&response.content);

            if parsed_calls.is_empty() {
                // No tool calls — final response
                let turn_info = TurnInfo {
                    turn,
                    tool_calls: vec![],
                    content: content.clone(),
                    usage: response.usage.clone(),
                };

                if let Some(ref callback) = self.on_turn {
                    callback(&turn_info);
                }

                debug!(turn, "agent: final response (no tool calls)");
                return Ok(AgentResult {
                    content,
                    tool_calls: all_tool_calls,
                    total_turns: turn,
                    total_usage,
                    finish_reason: response.finish_reason,
                });
            }

            info!(
                turn,
                call_count = parsed_calls.len(),
                "agent: executing tool calls"
            );

            // Execute tool calls
            let mut function_responses = Vec::with_capacity(parsed_calls.len());
            for call in &parsed_calls {
                let resp = (self.tool_handler)(&call.name, &call.args);
                function_responses.push(resp);
            }

            // Build turn info for callback
            let turn_info = TurnInfo {
                turn,
                tool_calls: parsed_calls.clone(),
                content: content.clone(),
                usage: response.usage,
            };

            if let Some(ref callback) = self.on_turn {
                callback(&turn_info);
            }

            all_tool_calls.extend(parsed_calls);

            // Append assistant response and tool results to conversation
            if !content.is_empty() {
                messages.push(ChatMessage::assistant(content));
            }

            let tool_results_text = format_tool_results_as_text(&function_responses);
            messages.push(ChatMessage::user(tool_results_text));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_simulation::FunctionResponse;
    use crate::types::{
        ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider,
        RunnerError, TokenUsage,
    };
    use async_trait::async_trait;
    use serde_json::json;
    use std::any::Any;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    struct TestProvider {
        responses: Mutex<Vec<Result<ChatResponse, RunnerError>>>,
        call_count: AtomicU32,
    }

    impl TestProvider {
        fn new(responses: Vec<Result<ChatResponse, RunnerError>>) -> Self {
            Self {
                responses: Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl LlmProvider for TestProvider {
        fn name(&self) -> &'static str {
            "test"
        }
        fn display_name(&self) -> &'static str {
            "Test Provider"
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::text_only()
        }
        fn default_model(&self) -> &'static str {
            "test-model"
        }
        fn available_models(&self) -> &[String] {
            &[]
        }
        async fn complete(&self, _request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let mut responses = self.responses.lock().expect("test lock");
            if responses.is_empty() {
                Err(RunnerError::internal("no more test responses"))
            } else {
                responses.remove(0)
            }
        }
        async fn complete_stream(&self, _request: &ChatRequest) -> Result<ChatStream, RunnerError> {
            Err(RunnerError::internal("not supported"))
        }
        async fn health_check(&self) -> Result<bool, RunnerError> {
            Ok(true)
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn make_response(content: &str, usage: Option<TokenUsage>) -> ChatResponse {
        ChatResponse {
            content: content.to_owned(),
            model: "test-model".to_owned(),
            usage,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
        }
    }

    fn noop_handler() -> TextToolHandler {
        Arc::new(|name: &str, _args: &serde_json::Value| FunctionResponse {
            name: name.to_owned(),
            response: json!({"status": "ok"}),
        })
    }

    #[tokio::test]
    async fn single_turn_no_tool_calls() {
        let provider = TestProvider::new(vec![Ok(make_response(
            "Here is a direct answer without tool calls.",
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 8,
                total_tokens: 18,
            }),
        ))]);

        let declarations = vec![FunctionDeclaration {
            name: "search".to_owned(),
            description: "Search the web".to_owned(),
            parameters: None,
        }];

        let executor = AgentExecutor::new(&provider, declarations, noop_handler());
        let messages = vec![ChatMessage::user("Hello")];
        let result = executor.run(messages).await.expect("should succeed");

        assert_eq!(
            result.content,
            "Here is a direct answer without tool calls."
        );
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.total_turns, 1);
        assert_eq!(result.total_usage.prompt_tokens, 10);
        assert_eq!(result.finish_reason, Some("stop".to_owned()));
    }

    #[tokio::test]
    async fn multi_turn_with_tool_calls() {
        let provider = TestProvider::new(vec![
            // Turn 1: LLM calls a tool
            Ok(make_response(
                "Let me search for that.\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"rust\"}}\n</tool_call>",
                Some(TokenUsage { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25 }),
            )),
            // Turn 2: LLM responds with the result
            Ok(make_response(
                "Based on the search results, Rust is a systems programming language.",
                Some(TokenUsage { prompt_tokens: 30, completion_tokens: 12, total_tokens: 42 }),
            )),
        ]);

        let declarations = vec![FunctionDeclaration {
            name: "search".to_owned(),
            description: "Search the web".to_owned(),
            parameters: Some(json!({"type": "object", "properties": {"q": {"type": "string"}}})),
        }];

        let executor = AgentExecutor::new(&provider, declarations, noop_handler());
        let messages = vec![ChatMessage::user("What is Rust?")];
        let result = executor.run(messages).await.expect("should succeed");

        assert!(result.content.contains("systems programming"));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.total_turns, 2);
        assert_eq!(result.total_usage.prompt_tokens, 40);
        assert_eq!(result.total_usage.completion_tokens, 27);
    }

    #[tokio::test]
    async fn on_turn_callback_invoked() {
        let provider = TestProvider::new(vec![
            Ok(make_response(
                "<tool_call>\n{\"name\": \"ping\", \"arguments\": {}}\n</tool_call>",
                None,
            )),
            Ok(make_response("Done!", None)),
        ]);

        let declarations = vec![FunctionDeclaration {
            name: "ping".to_owned(),
            description: "Ping".to_owned(),
            parameters: None,
        }];

        let turn_log: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let turn_log_clone = Arc::clone(&turn_log);

        let callback: OnTurnCallback = Arc::new(move |info: &TurnInfo| {
            turn_log_clone.lock().expect("lock").push(info.turn);
        });

        let executor =
            AgentExecutor::new(&provider, declarations, noop_handler()).with_on_turn(callback);
        let messages = vec![ChatMessage::user("ping")];
        executor.run(messages).await.expect("should succeed");

        let logged = turn_log.lock().expect("lock").clone();
        assert_eq!(logged, vec![1, 2]);
    }

    #[tokio::test]
    async fn max_turns_exhaustion() {
        // Provider always returns tool calls — should exhaust max_turns
        let mut responses = Vec::new();
        for _ in 0..5 {
            responses.push(Ok(make_response(
                "<tool_call>\n{\"name\": \"loop\", \"arguments\": {}}\n</tool_call>",
                None,
            )));
        }
        let provider = TestProvider::new(responses);

        let declarations = vec![FunctionDeclaration {
            name: "loop".to_owned(),
            description: "Loop forever".to_owned(),
            parameters: None,
        }];

        let executor =
            AgentExecutor::new(&provider, declarations, noop_handler()).with_max_turns(3);
        let messages = vec![ChatMessage::user("go")];
        let result = executor.run(messages).await.expect("should not error");

        assert_eq!(result.finish_reason, Some("max_turns".to_owned()));
        assert_eq!(result.total_turns, 3);
        assert_eq!(result.tool_calls.len(), 3);
    }

    #[tokio::test]
    async fn token_accumulation() {
        let provider = TestProvider::new(vec![
            Ok(make_response(
                "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>",
                Some(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                }),
            )),
            Ok(make_response(
                "final",
                Some(TokenUsage {
                    prompt_tokens: 20,
                    completion_tokens: 3,
                    total_tokens: 23,
                }),
            )),
        ]);

        let declarations = vec![FunctionDeclaration {
            name: "a".to_owned(),
            description: "Tool A".to_owned(),
            parameters: None,
        }];

        let executor = AgentExecutor::new(&provider, declarations, noop_handler());
        let messages = vec![ChatMessage::user("go")];
        let result = executor.run(messages).await.expect("should succeed");

        assert_eq!(result.total_usage.prompt_tokens, 30);
        assert_eq!(result.total_usage.completion_tokens, 8);
        assert_eq!(result.total_usage.total_tokens, 38);
    }
}
