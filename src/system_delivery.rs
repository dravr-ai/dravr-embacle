// ABOUTME: Every runner must declare HOW it delivers the System message to its model
// ABOUTME: Makes "silently dropped the system prompt" a compile error instead of a runtime ghost
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

//! System-message delivery contract.
//!
//! A caller hands embacle a [`ChatRequest`](crate::types::ChatRequest) whose
//! first message is usually a `System` message carrying the entire persona,
//! tool discipline and safety scaffolding. Runners deliver that in one of two
//! ways, and until now the choice was made implicitly by which prompt builder
//! each runner happened to import:
//!
//! - [`prepare_prompt`](crate::prompt::prepare_prompt) / `build_prompt` —
//!   **inlines** the System message into the prompt body as `[system]`.
//! - [`prepare_user_prompt`](crate::prompt::prepare_user_prompt) /
//!   `build_user_prompt` — **filters the System message out**, on the
//!   assumption that the runner passes it through some other channel.
//!
//! The second is correct only when such a channel exists. `claude_code` has one
//! (`--system-prompt`). Seven runners did not — `codex_cli`, `goose_cli`,
//! `cursor_agent`, `gemini_cli`, `cline_cli`, `continue_cli`, `kiro_cli` — and
//! imported the excluding builder anyway. On those, the System message reached
//! the model **nowhere**: no error, no warning, and because each wraps a coding
//! assistant with its own built-in persona, the CLI still returned a plausible
//! answer. The failure was invisible by construction.
//!
//! Nothing in the type system tied "uses the excluding builder" to "has
//! somewhere else to put it", so the pairing came apart silently. This module
//! makes it explicit: every [`CliRunnerType`] declares its delivery mode, the
//! match is exhaustive, and a new runner cannot compile without choosing.
//!
//! The rule for choosing is mechanical, not a judgement call: **if the runner
//! has no dedicated channel, it must inline.** Dropping is never an option.

use crate::config::CliRunnerType;

/// How a runner gets the `System` message to its model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemDelivery {
    /// The runner passes the System message through a dedicated channel — a CLI
    /// flag such as `--system-prompt`, or a native API field the backend
    /// honours. The prompt body must NOT also carry it, or it is duplicated.
    SeparateChannel,
    /// The runner has no dedicated channel, so the System text must be inlined
    /// into the prompt body. This is the safe default: a duplicated system
    /// prompt is a cosmetic problem, a missing one is a silent outage.
    InlineInPrompt,
}

impl CliRunnerType {
    /// How this runner delivers the `System` message.
    ///
    /// Exhaustive by design — adding a runner without deciding this is a
    /// compile error, which is the entire point of the module.
    #[must_use]
    pub const fn system_delivery(self) -> SystemDelivery {
        match self {
            // Passes `--system-prompt` on the command line.
            Self::ClaudeCode => SystemDelivery::SeparateChannel,

            // No dedicated channel. Each of these previously used the excluding
            // builder and therefore delivered the System message nowhere.
            // Verified against their CLI surfaces: the only flags they accept
            // are shapes like --full-auto / --json / --quiet / --model /
            // --resume, none of which carries a system prompt.
            Self::CodexCli
            | Self::GooseCli
            | Self::CursorAgent
            | Self::GeminiCli
            | Self::ClineCli
            | Self::ContinueCli
            | Self::KiroCli => SystemDelivery::InlineInPrompt,

            // Already inlining, and correct to.
            Self::Copilot | Self::OpenCode | Self::WarpCli | Self::KiloCli => {
                SystemDelivery::InlineInPrompt
            }

            // ACP `session/new` accepts a `systemPrompt` field, but GitHub
            // Copilot CLI's request schema silently strips unknown keys, so the
            // field never reaches the model. Prompt-text inlining is the ONLY
            // delivery path — which is why the previous
            // `COPILOT_HEADLESS_INJECT_SYSTEM_IN_PROMPT` knob had no correct
            // `false` value.
            #[cfg(feature = "copilot-headless")]
            Self::CopilotHeadless => SystemDelivery::InlineInPrompt,

            // The SDK's `session.create` carries `systemMessage { mode:
            // "replace" }`, which the runtime honours; the prompt body must
            // not repeat it.
            #[cfg(feature = "copilot-sdk")]
            Self::CopilotSdk => SystemDelivery::SeparateChannel,

            // Browser-driven; the whole conversation is typed into the page.
            #[cfg(feature = "web-ui")]
            Self::ClaudeWeb => SystemDelivery::InlineInPrompt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{build_prompt, build_user_prompt};
    use crate::types::ChatMessage;

    const SYSTEM_TEXT: &str = "You are Dravr, the athlete's coach, and nothing else.";

    fn messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system(SYSTEM_TEXT),
            ChatMessage::user("what should I ride this weekend?"),
        ]
    }

    /// The contract, stated as a test: an inlining runner's prompt must contain
    /// the System text. This is the assertion whose absence let seven runners
    /// drop it silently for months.
    #[test]
    fn inlining_runners_carry_the_system_text_in_the_prompt() {
        let prompt = build_prompt(&messages());
        assert!(
            prompt.contains(SYSTEM_TEXT),
            "build_prompt must inline the System message, got: {prompt}"
        );
    }

    /// And the excluding builder genuinely excludes — so a runner declaring
    /// [`SystemDelivery::SeparateChannel`] must actually have one, or the text
    /// goes nowhere.
    #[test]
    fn the_excluding_builder_really_drops_the_system_text() {
        let prompt = build_user_prompt(&messages());
        assert!(
            !prompt.contains(SYSTEM_TEXT),
            "build_user_prompt is only safe for runners with a separate channel"
        );
    }

    /// Generates the runner list and an exhaustive `match` over the same
    /// tokens, so the list cannot fall behind the enum.
    ///
    /// These tests used to carry two hand-written arrays. `CopilotSdk` was added
    /// to the enum with a `SeparateChannel` declaration and to neither array, so
    /// the assertion that exactly one runner uses a separate channel kept
    /// passing while a second one shipped. A variant missing from the list below
    /// is now a non-exhaustive `match` in `is_listed`: it does not compile.
    macro_rules! runner_list {
        ($( $(#[$gate:meta])* $variant:ident ),* $(,)?) => {
            const ALL_RUNNERS: &[CliRunnerType] = &[
                $( $(#[$gate])* CliRunnerType::$variant ),*
            ];

            const fn is_listed(runner: CliRunnerType) -> bool {
                match runner {
                    $( $(#[$gate])* CliRunnerType::$variant => true ),*
                }
            }
        };
    }

    runner_list![
        ClaudeCode,
        CursorAgent,
        OpenCode,
        Copilot,
        GeminiCli,
        CodexCli,
        GooseCli,
        ClineCli,
        ContinueCli,
        WarpCli,
        KiroCli,
        KiloCli,
        #[cfg(feature = "copilot-headless")]
        CopilotHeadless,
        #[cfg(feature = "copilot-sdk")]
        CopilotSdk,
        #[cfg(feature = "web-ui")]
        ClaudeWeb,
    ];

    /// Every runner must have a declared delivery mode. The match in
    /// `system_delivery` is exhaustive, so this passing means no variant was
    /// added without a decision — and `ALL_RUNNERS` is held to the enum by
    /// `is_listed`, so no variant is skipped here either.
    #[test]
    fn every_runner_declares_a_delivery_mode() {
        for runner in ALL_RUNNERS {
            assert!(is_listed(*runner));
            let _ = runner.system_delivery();
        }
    }

    /// A runner may exclude the System message only because it has a dedicated
    /// slot to deliver it through. Two do: Claude Code's `--system-prompt`, and
    /// the Copilot SDK's `session.create` `systemMessage { mode: "replace" }`.
    /// If a future runner declares [`SystemDelivery::SeparateChannel`], this test
    /// forces the author to justify it here rather than in a silent import choice.
    #[test]
    fn only_runners_with_a_system_slot_use_a_separate_channel() {
        let separate: Vec<CliRunnerType> = ALL_RUNNERS
            .iter()
            .copied()
            .filter(|r| r.system_delivery() == SystemDelivery::SeparateChannel)
            .collect();

        let expected = vec![
            CliRunnerType::ClaudeCode,
            #[cfg(feature = "copilot-sdk")]
            CliRunnerType::CopilotSdk,
        ];
        assert_eq!(
            separate, expected,
            "a runner may only exclude the System message if it has a dedicated \
             channel to deliver it through; add the justification in \
             system_delivery() before changing this"
        );
    }
}
