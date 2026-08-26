// ABOUTME: Live A/B proving whether copilot --acp disowns a host tool under text tool-calling
// ABOUTME: Answers "does mcp_tool_calling=false actually make the model refuse?" with a run, not a comment
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai
//
// Run: cargo run --example copilot_tool_disowning --features copilot-headless
//
// Background. dravr-platform's `.envrc` asserts, in a comment dated 2026-08-19,
// that turning `mcp_tool_calling` off routes the turn to the text `<tool_call>`
// loop "which this model refuses outright — that tool isn't part of my real
// toolset". On 2026-08-26 the same model, with tools published over MCP
// instead, told a live athlete «je n'ai pas d'outil qui écrit vers
// intervals.icu» — a tool it had been given. Two opposite presentation modes,
// the same disowning.
//
// A comment is not evidence. This example re-runs the 08-19 experiment against
// the current CLI and model so the claim can be confirmed, dated, or dropped.
// It is deliberately NOT a #[test]: it spawns the real `copilot` binary and
// spends real Copilot quota.

use std::sync::Arc;
use std::time::Duration;

use embacle::tool_simulation::{FunctionDeclaration, FunctionResponse};
use embacle::types::ChatMessage;
use embacle::{execute_with_text_tools, CopilotHeadlessConfig, CopilotHeadlessRunner};
use serde_json::{json, Value};
use std::sync::Mutex;
use tokio::time::timeout;
use tracing::Level;
use tracing_subscriber::fmt;

/// The athlete's actual words, from the 2026-08-26 Telegram turn.
const ATHLETE_ASK: &str = "peux-tu sauvegarder mon plan dans intervals.icu?";

/// `TOOL_BOUNDARY` as dravr-platform shipped it (`prompt_builder.rs`). Reproduced
/// verbatim because it is a prime suspect: under MCP tool calling nothing is
/// "described elsewhere in this prompt", so the sentence reads to the model as
/// "you have no tools".
const TOOL_BOUNDARY: &str = "## Tool boundary\n\n\
     The tools available to you are the ones described elsewhere in this prompt \
     and nothing else. You cannot browse the web, scrape menus, look up prices, \
     use third-party services, or run arbitrary code. If a request needs a \
     capability you have not been given, say so honestly rather than inventing a \
     plan. Call tools with the parameters described in their schemas.";

const COACH_PROMPT: &str = "You are an endurance coach for an athlete named Phil. \
     Answer in French, in the athlete's register. Today is 2026-08-26.";

/// The three declarations that matter, carrying dravr-platform's real
/// descriptions so the model sees exactly what production publishes.
fn declarations() -> Vec<FunctionDeclaration> {
    vec![
        FunctionDeclaration {
            name: "prescribe_workout".into(),
            description: "Write one workout onto the athlete's Intervals.icu calendar for a \
                 given date, and record it in the prescribed_workouts audit trail. Requires a \
                 connected Intervals.icu account. Pass EITHER template_slug — one of the \
                 cornerstones (long_run_z2, threshold_4x8, vo2_5x3, recovery_30min, \
                 tempo_progression, sweet_spot_2x20) — OR session, a structured session you \
                 authored. Args: date (YYYY-MM-DD), template_slug or session, optional coach_id."
                .into(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "template_slug": {"type": "string"}
                },
                "required": ["date"]
            })),
        },
        FunctionDeclaration {
            name: "list_workout_templates".into(),
            description: "List the Endurance cornerstone workout templates with their \
                 structured steps and target zones."
                .into(),
            parameters: Some(json!({"type": "object", "properties": {}})),
        },
        FunctionDeclaration {
            name: "get_activities".into(),
            description: "Fetch the athlete's recent activities from their connected provider."
                .into(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"limit": {"type": "integer"}}
            })),
        },
    ]
}

/// Phrases that mean the model disowned a tool it was handed. Kept narrow: a
/// refusal to call is only interesting when it denies the capability EXISTS.
const DISOWN_MARKERS: &[&str] = &[
    "pas d'outil",
    "pas d outil",
    "aucun outil",
    "n'ai pas d'outil",
    "not part of my real toolset",
    "isn't part of my real toolset",
    "i don't have a tool",
    "i do not have a tool",
    "no tool that",
    "je ne peux pas pousser",
    "je ne peux pas écrire",
];

fn disowning_phrase(reply: &str) -> Option<&'static str> {
    let lower = reply.to_lowercase();
    DISOWN_MARKERS.iter().find(|m| lower.contains(**m)).copied()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    fmt().with_max_level(Level::INFO).with_target(false).init();

    // Force the flag OFF regardless of the ambient shell, so the run answers
    // the question it claims to. `from_env()` would inherit
    // COPILOT_HEADLESS_MCP_TOOL_CALLING and silently test the other mode.
    let mut config = CopilotHeadlessConfig::from_env();
    config.mcp_tool_calling = false;
    let model = config.model.clone();
    let runner = CopilotHeadlessRunner::with_config(config);

    println!("=== copilot --acp, text <tool_call> loop, mcp_tool_calling=false ===");
    println!("model: {model}");
    println!("ask:   {ATHLETE_ASK}\n");

    let called: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&called);
    let handler = Arc::new(move |name: &str, args: &Value| {
        // std Mutex, not tokio's: the handler is a sync callback invoked from
        // inside the runtime, where a tokio blocking_lock() panics.
        if let Ok(mut seen) = sink.lock() {
            seen.push(name.to_owned());
        }
        println!(">>> model called {name} with {args}");
        // Per-tool payloads. A blanket {"status":"pushed"} made a
        // list_workout_templates call read back as a successful calendar write,
        // and the model then reported the push — the probe manufacturing the
        // very false claim it exists to detect.
        let response = match name {
            "prescribe_workout" => json!({
                "status": "pushed",
                "provider_event_id": "evt_live_probe",
                "date": "2026-08-26"
            }),
            "list_workout_templates" => json!({
                "templates": [
                    {"slug": "vo2_5x3", "name": "VO2 5x3", "sport": "Ride"},
                    {"slug": "threshold_4x8", "name": "Threshold 4x8", "sport": "Ride"},
                    {"slug": "long_run_z2", "name": "Long Run Z2", "sport": "Run"}
                ]
            }),
            "get_activities" => json!({
                "activities": [
                    {"name": "Afternoon Trail Run", "date": "2026-08-25",
                     "distance_km": 8.73, "duration": "55:28"}
                ]
            }),
            other => json!({"error": format!("no such tool: {other}")}),
        };
        FunctionResponse {
            name: name.to_owned(),
            response,
        }
    });

    // Scenario B of the real turn: the plan already exists in the conversation,
    // exactly as it did on 2026-08-26 when the coach had just written it and
    // said «Ajusté et sauvegardé». With nothing to disambiguate, a model that
    // owns the tool has no reason left NOT to call it — which is what makes
    // this the scenario that actually decides the question.
    let mut messages = vec![
        ChatMessage::system(format!("{COACH_PROMPT}\n\n{TOOL_BOUNDARY}")),
        ChatMessage::user("je peux faire mes intervalles aujourd'hui a velo. ajuste mon plan"),
        ChatMessage::assistant(
            "Ajusté et sauvegardé. Aujourd'hui (26 août) devient une séance d'intervalles \
             40/20 à vélo (10-12 reps, 390-425W efforts / <190W récup) à la place du gravel \
             Z2 stable. Le reste ne bouge pas : vendredi 28 MTB Z1-Z2 1h45, samedi 29 ta \
             longue trail 105min avec ravito, dimanche 30 ta longue vélo progressive.",
        ),
        ChatMessage::user(ATHLETE_ASK),
    ];

    match timeout(
        Duration::from_mins(3),
        execute_with_text_tools(&runner, &mut messages, &declarations(), handler, 5),
    )
    .await
    {
        Ok(Ok(resp)) => {
            println!("\n--- reply ---\n{}", resp.content);
            println!("\n--- tool_calls_count: {} ---", resp.tool_calls_count);
            println!(
                "--- tools called: {:?} ---",
                called.lock().map(|g| g.clone()).unwrap_or_default()
            );
            match disowning_phrase(&resp.content) {
                Some(marker) if resp.tool_calls_count == 0 => println!(
                    "\nVERDICT: DISOWNED. Zero tool calls and the reply denies the \
                     capability (matched {marker:?}). The .envrc claim holds."
                ),
                Some(marker) => println!(
                    "\nVERDICT: MIXED. The model called {} tool(s) but the reply still \
                     denies the capability (matched {marker:?}).",
                    resp.tool_calls_count
                ),
                None if resp.tool_calls_count > 0 => println!(
                    "\nVERDICT: TOOL USED. The model called the tool over the text loop. \
                     The .envrc claim is STALE for this CLI/model."
                ),
                None => println!(
                    "\nVERDICT: NO CALL, NO DENIAL. It neither used the tool nor denied \
                     having it — read the reply above before concluding anything."
                ),
            }
        }
        Ok(Err(e)) => println!("execute_with_text_tools error: {e}"),
        Err(_) => println!("timed out after 180s"),
    }
}
