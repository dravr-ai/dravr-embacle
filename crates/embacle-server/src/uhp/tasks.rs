// ABOUTME: POST /v1/responses and the stored Response objects it produces, plus read-back
// ABOUTME: Reserved fields are ignored observably; usage is reported honestly or as null
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! Running a task.
//!
//! `POST /v1/responses` is the core of the protocol: one unit of work, run on a
//! named harness, returned as a `Response` object. The surface is shaped like a
//! responses API rather than a chat-completions one, which is why it lives here
//! and not in the `OpenAI` handler beside it.
//!
//! Three requirements shape this module more than the happy path does.
//!
//! **Reserved fields are ignored, not rejected.** `tools` and `include` are
//! reserved by the specification, and a client that sends them for wire
//! compatibility with another API must still be served. But ignoring has to be
//! *observable*: dropping a field silently is indistinguishable from acting on
//! it, so what was dropped comes back in `metadata.ignored_fields` — and only
//! what this request actually carried, never a catalogue of what the server
//! would drop.
//!
//! **Usage is honest or absent.** `null` when the runner accounted for nothing.
//! A fabricated zero is worse than an honest absence, because a client cannot
//! tell it from a free task.
//!
//! **A terminal state never changes.** Reading a task back must return the same
//! status it finished with.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path as StdPath;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response as AxumResponse};
use axum::Json;
use embacle::config::{CliRunnerType, RunnerConfig};
use embacle::discovery::resolve_binary;
use embacle::factory::create_runner_with_config;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, RunnerError};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::error::{ErrorType, UhpFailure};
use super::files;
use super::harnesses;
use super::state::UhpState;

/// The fields this protocol reserves and a server must ignore.
const RESERVED: [&str; 2] = ["tools", "include"];

/// What a client sends to run a task.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateResponse {
    /// A bare string is shorthand for one user message.
    pub input: Value,
    /// Canonical model id; absent means the harness default.
    #[serde(default)]
    pub model: Option<String>,
    /// Client metadata. `harness_id` selects the harness.
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
    /// Whether to stream the result as server-sent events.
    #[serde(default)]
    pub stream: bool,
    /// Return immediately with an `in_progress` task and keep running it.
    #[serde(default)]
    pub background: bool,
    /// Continue the session this response belongs to.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// System instructions for the turn.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Everything the client sent that this struct does not name, kept so
    /// reserved fields can be reported rather than silently dropped.
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

impl CreateResponse {
    /// The prompt text, whether the client sent a string or an item array.
    pub fn prompt(&self) -> String {
        match &self.input {
            Value::String(s) => s.clone(),
            Value::Array(items) => items
                .iter()
                .filter_map(|item| {
                    item.get("content").map(|c| match c {
                        Value::String(s) => s.clone(),
                        other => other
                            .as_array()
                            .map(|parts| {
                                parts
                                    .iter()
                                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                                    .collect::<Vec<_>>()
                                    .join("")
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.as_str().unwrap_or_default().to_owned(),
        }
    }

    /// Which reserved fields this request actually carried, in spec order.
    ///
    /// Reports what was sent and dropped — not what the server would drop, and
    /// not what it happens to know about. A hardcoded list tells a client its
    /// request was altered when it was not.
    pub fn ignored_fields(&self) -> Vec<String> {
        RESERVED
            .iter()
            .filter(|f| self.rest.contains_key(**f))
            .map(|f| (*f).to_owned())
            .collect()
    }

    /// The harness the client asked for, if any.
    pub fn harness_id(&self) -> Option<&str> {
        self.metadata
            .as_ref()?
            .get("harness_id")
            .and_then(Value::as_str)
    }
}

/// One item of a task's output.
#[derive(Debug, Clone, Serialize)]
pub struct OutputItem {
    /// Item kind; `message` is the only one this server produces today.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Item id.
    pub id: String,
    /// Terminal status of the item.
    pub status: &'static str,
    /// Who produced it.
    pub role: &'static str,
    /// The parts of the message.
    pub content: Vec<ContentPart>,
}

/// One part of a message.
#[derive(Debug, Clone, Serialize)]
pub struct ContentPart {
    /// Part kind.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The text itself.
    pub text: String,
}

/// Token accounting, when the runner reported any.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub input_tokens: u32,
    /// Tokens generated.
    pub output_tokens: u32,
    /// Their sum.
    pub total_tokens: u32,
}

/// A task, in whatever state it reached.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    /// `resp_`-prefixed id.
    pub id: String,
    /// Always `response`.
    pub object: &'static str,
    /// Unix seconds.
    pub created_at: u64,
    /// One of the five lifecycle states.
    pub status: &'static str,
    /// Non-null only when `status` is `failed`.
    pub error: Option<super::error::UhpError>,
    /// The response this one continued, if any.
    pub previous_response_id: Option<String>,
    /// The model that actually ran.
    pub model: String,
    /// What the task produced.
    pub output: Vec<OutputItem>,
    /// Whether the server retained it.
    pub store: bool,
    /// `null` when the runner accounted for nothing.
    pub usage: Option<Usage>,
    /// Session id, ignored fields, and any model substitution.
    pub metadata: Map<String, Value>,
}

/// Responses this server has run and retained, and the sessions they belong to.
///
/// LIMITATION(registre#410): `ResponseStore` is per-process, so a task run on one
/// instance cannot be read back from another.
#[derive(Debug, Default)]
pub struct ResponseStore {
    responses: RwLock<HashMap<String, Response>>,
    sessions: RwLock<HashMap<String, Vec<String>>>,
    cancelled: RwLock<HashSet<String>>,
}

impl ResponseStore {
    /// Retain a finished response and record it against its session.
    pub fn put(&self, session_id: &str, response: Response) {
        if let Ok(mut sessions) = self.sessions.write() {
            sessions
                .entry(session_id.to_owned())
                .or_default()
                .push(response.id.clone());
        }
        if let Ok(mut responses) = self.responses.write() {
            responses.insert(response.id.clone(), response);
        }
    }

    /// Read a response back by id.
    pub fn get(&self, id: &str) -> Option<Response> {
        self.responses.read().ok()?.get(id).cloned()
    }

    /// Every session this server holds.
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions
            .read()
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// The responses making up one session, oldest first.
    pub fn session_turns(&self, session_id: &str) -> Option<Vec<Response>> {
        let ids = self.sessions.read().ok()?.get(session_id)?.clone();
        let responses = self.responses.read().ok()?;
        Some(
            ids.iter()
                .filter_map(|i| responses.get(i).cloned())
                .collect(),
        )
    }

    /// Ask a running task to stop.
    ///
    /// Returns whether the task was still running. A terminal task is left
    /// exactly as it is: a client retrying a cancel after a dropped connection
    /// must not be punished for having succeeded the first time, and a
    /// terminal status must never change afterwards.
    pub fn cancel(&self, id: &str) -> bool {
        let running = self.get(id).is_some_and(|r| r.status == "in_progress");
        if running {
            if let Ok(mut cancelled) = self.cancelled.write() {
                cancelled.insert(id.to_owned());
            }
            if let Ok(mut responses) = self.responses.write() {
                if let Some(r) = responses.get_mut(id) {
                    r.status = "cancelled";
                }
            }
        }
        running
    }

    /// Whether this task was asked to stop.
    pub fn is_cancelled(&self, id: &str) -> bool {
        self.cancelled.read().is_ok_and(|c| c.contains(id))
    }

    /// Replace a stored response, unless a cancel already settled it.
    ///
    /// Cancellation wins: a task that finished after the client asked it to
    /// stop must not report `completed`, or the cancel silently did nothing.
    pub fn settle(&self, response: Response) {
        if self.is_cancelled(&response.id) {
            return;
        }
        if let Ok(mut responses) = self.responses.write() {
            responses.insert(response.id.clone(), response);
        }
    }

    /// The ids of a session's responses that are still running.
    pub fn running_in(&self, session_id: &str) -> Vec<String> {
        self.session_turns(session_id)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.status == "in_progress")
            .map(|r| r.id)
            .collect()
    }

    /// Forget a session and every response in it.
    ///
    /// Returns whether it existed, so the caller can answer `session_not_found`
    /// rather than reporting a delete that removed nothing.
    pub fn delete_session(&self, session_id: &str) -> bool {
        let ids = self
            .sessions
            .write()
            .map_or(None, |mut sessions| sessions.remove(session_id));
        let Some(ids) = ids else { return false };
        if let Ok(mut responses) = self.responses.write() {
            for id in &ids {
                responses.remove(id);
            }
        }
        true
    }

    /// The session a response belongs to.
    pub fn session_of(&self, response_id: &str) -> Option<String> {
        self.get(response_id)?
            .metadata
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }
}

/// Unix seconds now.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// A prefixed id, as the object model requires.
fn id_with(prefix: &str) -> String {
    format!("{prefix}{}", Uuid::new_v4().simple())
}

/// Handle `POST /v1/responses`.
///
/// # Errors
///
/// Answers `404 harness_not_found` for an unknown harness and
/// `400 invalid_input` when the request names no runnable harness.
pub async fn create(
    State(state): State<UhpState>,
    Json(request): Json<CreateResponse>,
) -> Result<AxumResponse, UhpFailure> {
    if request.background {
        return Ok(Json(start_background(state, request)).into_response());
    }
    if request.stream {
        let prepared = prepare(&state, &request).await?;
        let previous = request.previous_response_id.clone();
        return Ok(super::streaming::sse(state, prepared, previous).into_response());
    }
    let outcome = run(&state, &request).await?;
    Ok(Json(outcome.response).into_response())
}

/// Start a task, retain it as `in_progress`, and answer straight away.
///
/// Background is what makes cancellation observable: a task the client is
/// still waiting on cannot be cancelled by that same client. The response is
/// stored before this returns, so the id it hands back is immediately readable.
///
/// The harness is resolved on the spawned task, so a resolution failure
/// surfaces as a `failed` response rather than a refusal to start — the client
/// already holds the id by then, and an id that never resolves is worse than
/// one that resolves to a failure.
fn start_background(state: UhpState, request: CreateResponse) -> Response {
    let id = id_with("resp_");
    let session_id = request
        .previous_response_id
        .as_ref()
        .and_then(|prev| state.store.session_of(prev))
        .unwrap_or_else(|| id_with("sess_"));

    let mut metadata = request.metadata.clone().unwrap_or_default();
    metadata.insert("session_id".to_owned(), Value::String(session_id.clone()));

    let pending = Response {
        id: id.clone(),
        object: "response",
        created_at: now_secs(),
        status: "in_progress",
        error: None,
        previous_response_id: request.previous_response_id.clone(),
        model: request.model.clone().unwrap_or_default(),
        output: Vec::new(),
        store: true,
        usage: None,
        metadata,
    };
    state.store.put(&session_id, pending.clone());

    let spawned = pending.clone();
    tokio::spawn(async move {
        let finished = match run(&state, &request).await {
            Ok(outcome) => {
                // The background task keeps the id the client already holds;
                // the run allocated its own, which nobody has seen.
                let mut r = outcome.response;
                r.id = id;
                r
            }
            Err(failure) => {
                let mut r = spawned;
                r.status = "failed";
                r.error = Some(failure.into_error());
                r
            }
        };
        state.store.settle(finished);
    });

    pending
}

/// Handle `POST /v1/responses/{response_id}/cancel`.
///
/// # Errors
///
/// Answers `404 response_not_found` when this server holds no such response.
pub async fn cancel(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<Response>, UhpFailure> {
    if state.store.get(&id).is_none() {
        return Err(UhpFailure::not_found(
            "response_not_found",
            "no response with that id exists",
        ));
    }
    state.store.cancel(&id);
    state.store.get(&id).map(Json).ok_or_else(|| {
        UhpFailure::not_found("response_not_found", "no response with that id exists")
    })
}

/// A finished task plus what the stream needs to describe it.
pub struct Outcome {
    /// The response object itself.
    pub response: Response,
}

/// Everything a task needs once the harness and session are settled.
pub struct Prepared {
    /// The runner that will answer.
    pub runner: Arc<dyn LlmProvider>,
    /// The request to hand it.
    pub chat: ChatRequest,
    /// The model that will run.
    pub model: String,
    /// The session this task belongs to.
    pub session_id: String,
    /// The response id, allocated before the work starts.
    pub id: String,
    /// Session id, ignored fields, and any model substitution.
    pub metadata: Map<String, Value>,
    /// What the client asked for, when it differs from what will run.
    pub requested_model: Option<String>,
}

/// Build a runner that executes inside `workdir`.
///
/// Deliberately not the shared pooled runner: pooling hands back one runner per
/// provider for the whole process, and every session needs its own directory or
/// their artifacts land in one pile with no way to tell them apart.
async fn runner_in(
    provider: CliRunnerType,
    workdir: &StdPath,
) -> Result<Arc<dyn LlmProvider>, RunnerError> {
    fs::create_dir_all(workdir)
        .map_err(|e| RunnerError::internal(format!("session working folder: {e}")))?;

    let env_override = env::var(provider.env_override_key()).ok();
    let binary = resolve_binary(provider.binary_name(), env_override.as_deref())?;
    let mut config = RunnerConfig::new(binary).with_working_directory(workdir.to_path_buf());
    config.extra_args = write_permission_args(provider);

    Ok(Arc::from(
        create_runner_with_config(provider, config).await?,
    ))
}

/// The arguments that let a harness write inside its own working folder.
///
/// A UHP server exists to run agent harnesses, and a harness that cannot write
/// a file cannot produce an artifact — the whole files half of the protocol is
/// unreachable without this. What makes it safe to grant is the confinement,
/// not trust: every task runs with its working directory set to that session's
/// folder alone, under embacle's sandbox policy, so an accepted edit lands
/// there and nowhere else.
///
/// Both flags are needed and neither is sufficient. `--permission-mode
/// acceptEdits` stops the interactive prompt, but with no MCP servers
/// configured the Write tool is still off the allow-list, so creating a file is
/// refused — and the refusal surfaces as prose from the model rather than an
/// error, which reads exactly like a model choosing not to act. Verified
/// against the CLI directly: with only `--permission-mode`, the run's
/// `permission_denials` carries the Write call.
///
/// Empty for a harness with no such flag: it either already writes freely or it
/// does not write at all, and inventing a flag name would fail at run time with
/// an unparseable argument rather than a missing file.
fn write_permission_args(provider: CliRunnerType) -> Vec<String> {
    match provider {
        CliRunnerType::ClaudeCode => vec![
            "--permission-mode".to_owned(),
            "acceptEdits".to_owned(),
            "--allowed-tools".to_owned(),
            "Write Edit".to_owned(),
        ],
        _ => Vec::new(),
    }
}

/// Resolve the harness, model and session for a task without running it.
///
/// Shared by the blocking and streaming paths so the two cannot drift on which
/// harness they picked or which session they reported.
///
/// # Errors
///
/// Answers `404 harness_not_found` when the requested harness is not installed.
pub async fn prepare(state: &UhpState, request: &CreateResponse) -> Result<Prepared, UhpFailure> {
    let harnesses = harnesses::discover(&state.shared).await;
    let chosen = match request.harness_id() {
        Some(id) => harnesses
            .iter()
            .find(|d| d.harness.id == id)
            .ok_or_else(|| {
                UhpFailure::not_found("harness_not_found", "no harness with that id is configured")
            })?,
        None => harnesses.first().ok_or_else(|| {
            UhpFailure::new(
                ErrorType::InvalidRequestError,
                "harness_not_found",
                "this server has no installed harness to run",
            )
        })?,
    };

    // The harness runs inside the session's own folder, which is what makes an
    // artifact attributable: a file that appears there was written by this
    // session's tasks and nothing else. Without it the harness writes into
    // whatever directory the server happened to start in, and the files it
    // produces belong to no one.
    let session_id = request
        .previous_response_id
        .as_ref()
        .and_then(|prev| state.store.session_of(prev))
        .unwrap_or_else(|| id_with("sess_"));
    let workdir = files::session_workdir(&session_id);
    let runner = runner_in(chosen.provider, &workdir).await.map_err(|_| {
        UhpFailure::new(
            ErrorType::HarnessError,
            "harness_unavailable",
            "the harness could not be started",
        )
    })?;

    let requested_model = request.model.clone();
    let model = requested_model
        .clone()
        .or_else(|| chosen.harness.default_model.clone())
        .unwrap_or_else(|| runner.default_model().to_owned());

    let mut chat = ChatRequest::new(vec![ChatMessage::user(request.prompt())]);
    chat.model = Some(model.clone());

    let mut metadata = request.metadata.clone().unwrap_or_default();
    metadata.insert("session_id".to_owned(), Value::String(session_id.clone()));
    let ignored = request.ignored_fields();
    if !ignored.is_empty() {
        metadata.insert(
            "ignored_fields".to_owned(),
            Value::Array(ignored.into_iter().map(Value::String).collect()),
        );
    }

    Ok(Prepared {
        runner,
        chat,
        model,
        session_id,
        id: id_with("resp_"),
        metadata,
        requested_model,
    })
}

/// The response object for a task that has not produced anything yet.
#[must_use]
pub fn pending_response(p: &Prepared, previous: Option<String>) -> Response {
    Response {
        id: p.id.clone(),
        object: "response",
        created_at: now_secs(),
        status: "in_progress",
        error: None,
        previous_response_id: previous,
        model: p.model.clone(),
        output: Vec::new(),
        store: true,
        usage: None,
        metadata: p.metadata.clone(),
    }
}

/// Wrap finished text as the terminal response object.
#[must_use]
pub fn completed_response(
    p: Prepared,
    previous: Option<String>,
    text: String,
    usage: Option<Usage>,
    ran_model: String,
) -> Response {
    let mut metadata = p.metadata;
    if p.requested_model.as_ref().is_some_and(|m| *m != ran_model) {
        metadata.insert(
            "requested_model".to_owned(),
            Value::String(p.requested_model.unwrap_or_default()),
        );
        metadata.insert("model_fallback".to_owned(), Value::Bool(true));
    }
    Response {
        id: p.id,
        object: "response",
        created_at: now_secs(),
        status: "completed",
        error: None,
        previous_response_id: previous,
        model: ran_model,
        output: vec![OutputItem {
            kind: "message",
            id: id_with("msg_"),
            status: "completed",
            role: "assistant",
            content: vec![ContentPart {
                kind: "output_text",
                text,
            }],
        }],
        store: true,
        usage,
        metadata,
    }
}

/// Run one task to a terminal state.
///
/// # Errors
///
/// Answers `404 harness_not_found` when the requested harness is not installed.
pub async fn run(state: &UhpState, request: &CreateResponse) -> Result<Outcome, UhpFailure> {
    let prepared = prepare(state, request).await?;
    let session_id = prepared.session_id.clone();
    let previous = request.previous_response_id.clone();
    let model = prepared.model.clone();

    let response = match prepared.runner.complete(&prepared.chat).await {
        Ok(reply) => {
            let ran = if reply.model.is_empty() {
                model
            } else {
                reply.model
            };
            // Honest or absent: a fabricated zero cannot be told from a free
            // task, which the specification calls the worse outcome.
            let usage = reply.usage.map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            });
            completed_response(prepared, previous, reply.content, usage, ran)
        }
        Err(e) => {
            let mut failed = pending_response(&prepared, previous);
            failed.status = "failed";
            failed.error = Some(super::error::UhpError {
                kind: ErrorType::HarnessError,
                code: "harness_error".to_owned(),
                message: e.message,
                detail: None,
            });
            failed
        }
    };

    state.store.put(&session_id, response.clone());
    Ok(Outcome { response })
}

/// Handle `GET /v1/responses/{response_id}`.
///
/// # Errors
///
/// Answers `404 response_not_found` when this server holds no such response.
pub async fn get(
    State(state): State<UhpState>,
    Path(id): Path<String>,
) -> Result<Json<Response>, UhpFailure> {
    state.store.get(&id).map(Json).ok_or_else(|| {
        UhpFailure::not_found("response_not_found", "no response with that id exists")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(body: Value) -> CreateResponse {
        serde_json::from_value(body).expect("request parses") // Safe: test assertion
    }

    #[test]
    fn a_bare_string_input_is_one_user_message() {
        assert_eq!(parse(json!({"input": "hello"})).prompt(), "hello");
    }

    #[test]
    fn an_item_array_input_is_flattened_to_its_text() {
        let r = parse(json!({"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {"role": "user", "content": "second"}
        ]}));
        assert_eq!(r.prompt(), "first\nsecond");
    }

    #[test]
    fn ignored_fields_reports_only_what_the_request_carried() {
        // T-09 and T-10 are two halves of one rule. Reporting a field the
        // client never sent tells it the request was altered when it was not.
        let both = parse(json!({"input": "x", "tools": [], "include": []}));
        assert_eq!(both.ignored_fields(), vec!["tools", "include"]);

        let one = parse(json!({"input": "x", "include": []}));
        assert_eq!(one.ignored_fields(), vec!["include"]);

        let neither = parse(json!({"input": "x"}));
        assert!(
            neither.ignored_fields().is_empty(),
            "a request that sent no reserved field must not be told one was ignored"
        );
    }

    #[test]
    fn a_reserved_field_does_not_make_the_request_unparseable() {
        // T-08: sending `tools` must be accepted and run, not rejected. If the
        // struct refused unknown fields this would fail at deserialisation.
        let r = parse(json!({
            "input": "x",
            "tools": [{"type": "function", "name": "probe"}],
            "metadata": {"harness_id": "chrn_copilot"}
        }));
        assert_eq!(r.harness_id(), Some("chrn_copilot"));
        assert_eq!(r.prompt(), "x");
    }

    #[test]
    fn ids_carry_the_prefix_the_object_model_requires() {
        assert!(id_with("resp_").starts_with("resp_"));
        assert!(id_with("sess_").starts_with("sess_"));
    }

    #[test]
    fn a_terminal_response_reads_back_unchanged() {
        let store = ResponseStore::default();
        let resp = Response {
            id: "resp_abc".to_owned(),
            object: "response",
            created_at: 1,
            status: "completed",
            error: None,
            previous_response_id: None,
            model: "m".to_owned(),
            output: Vec::new(),
            store: true,
            usage: None,
            metadata: Map::new(),
        };
        store.put("sess_1", resp);

        let got = store.get("resp_abc").expect("stored"); // Safe: test assertion
        assert_eq!(got.status, "completed", "a terminal state must not change");
        assert_eq!(store.session_turns("sess_1").map(|t| t.len()), Some(1));
        assert!(store.get("resp_missing").is_none());
    }
}
