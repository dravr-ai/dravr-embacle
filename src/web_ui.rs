// ABOUTME: Browser-driven LLM provider — drives the Claude.ai web UI via a persistent headless profile
// ABOUTME: Pastes the prompt into a fresh chat, captures the streamed SSE response, maps it to ChatResponse
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 dravr.ai

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use dravr_browser::{
    click_element, element_exists, fill_input_field, launch_browser, open_page_with_stealth,
    parse_sse_data, read_last_capture, read_visible_text, Browser, BrowserError,
    BrowserLaunchConfig, Page, StealthOptions,
};
use serde::Deserialize;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{sleep, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, info, warn};

use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, LlmCapabilities, LlmProvider, MessageRole,
    RunnerError, StreamChunk,
};

/// Default profile id (and on-disk profile directory name) for the Claude web session.
const DEFAULT_PROFILE_ID: &str = "claude-web";

/// Embedded default provider config (selectors + SSE extraction rules).
const CLAUDE_WEB_PROVIDER_TOML: &str = include_str!("../providers/claude_web.toml");

/// Runtime configuration for the browser-driven web runner.
#[derive(Debug, Clone)]
pub struct WebUiConfig {
    /// Persistent profile id — the authenticated session lives under this name.
    pub profile_id: String,
    /// Launch options for the underlying browser (headless, profile dir, proxy).
    pub launch: BrowserLaunchConfig,
    /// Optional path to a provider TOML overriding the embedded default.
    pub provider_config_path: Option<PathBuf>,
    /// Interval between capture-buffer polls while a response streams.
    pub poll_interval: Duration,
    /// Maximum time to wait for the composer to appear after navigation.
    pub composer_timeout: Duration,
    /// Maximum time to wait for the full response to arrive.
    pub response_timeout: Duration,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            profile_id: DEFAULT_PROFILE_ID.to_owned(),
            launch: BrowserLaunchConfig::default(),
            provider_config_path: None,
            poll_interval: Duration::from_millis(60),
            composer_timeout: Duration::from_secs(30),
            response_timeout: Duration::from_mins(3),
        }
    }
}

impl WebUiConfig {
    /// Build a config from environment variables, falling back to defaults.
    ///
    /// - `EMBACLE_WEB_PROFILE_ID` — persistent profile id (default `claude-web`)
    /// - `EMBACLE_WEB_HEADLESS` — `false`/`0` runs headed (for interactive login)
    /// - `EMBACLE_WEB_PROVIDER_CONFIG` — path to a provider TOML override
    /// - `EMBACLE_WEB_RESPONSE_TIMEOUT_SECS` — overall response timeout
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(id) = env::var("EMBACLE_WEB_PROFILE_ID") {
            if !id.is_empty() {
                config.profile_id = id;
            }
        }
        if let Ok(headless) = env::var("EMBACLE_WEB_HEADLESS") {
            config.launch.headless = !matches!(headless.as_str(), "false" | "0" | "no");
        }
        if let Ok(path) = env::var("EMBACLE_WEB_PROVIDER_CONFIG") {
            if !path.is_empty() {
                config.provider_config_path = Some(PathBuf::from(path));
            }
        }
        if let Some(secs) = env::var("EMBACLE_WEB_RESPONSE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            config.response_timeout = Duration::from_secs(secs);
        }
        config
    }
}

/// Provider identity and login-detection settings.
#[derive(Debug, Clone, Deserialize)]
pub struct WebProviderIdentity {
    /// Stable provider name (e.g. `claude_web`).
    pub name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// URL opened to start a fresh conversation per request.
    pub new_chat_url: String,
    /// URL used by the one-time interactive login flow.
    pub login_url: String,
    /// Current-URL substrings indicating an authenticated session.
    pub login_success_patterns: Vec<String>,
}

/// Composer (prompt input + send) selectors.
#[derive(Debug, Clone, Deserialize)]
pub struct ComposerConfig {
    /// Selector(s) for the prompt input element.
    pub input_selector: String,
    /// Selector(s) for the send button.
    pub send_button_selector: String,
}

/// Response capture + extraction settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseConfig {
    /// JS regex (source) matching the streaming completion request URL.
    pub stream_url_pattern: String,
    /// JSON Pointers tried in order against each SSE `data:` JSON payload.
    pub text_json_pointers: Vec<String>,
    /// SSE payloads equal to / containing any of these end the turn.
    pub done_markers: Vec<String>,
    /// DOM fallback selector for the assistant message if no stream was captured.
    pub assistant_message_selector: String,
}

/// Model advertising for the web provider.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    /// Default model name reported by the provider.
    pub default: String,
    /// Models advertised as available.
    pub available: Vec<String>,
}

/// Declarative provider configuration parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct WebProviderConfig {
    /// Provider identity / login detection.
    pub provider: WebProviderIdentity,
    /// Composer selectors.
    pub composer: ComposerConfig,
    /// Response capture + extraction rules.
    pub response: ResponseConfig,
    /// Advertised models.
    pub models: ModelsConfig,
}

impl WebProviderConfig {
    /// Parse a provider config from TOML text.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the TOML is malformed.
    pub fn from_toml(text: &str) -> Result<Self, RunnerError> {
        toml::from_str(text)
            .map_err(|e| RunnerError::config(format!("Invalid web provider config: {e}")))
    }

    /// Load a provider config from a file.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the file cannot be read or parsed.
    pub fn from_file(path: &PathBuf) -> Result<Self, RunnerError> {
        let text = fs::read_to_string(path)
            .map_err(|e| RunnerError::config(format!("Failed to read {}: {e}", path.display())))?;
        Self::from_toml(&text)
    }

    /// The embedded default Claude.ai web provider config.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] only if the compiled-in TOML is malformed
    /// (guarded by a unit test).
    pub fn claude_web_default() -> Result<Self, RunnerError> {
        Self::from_toml(CLAUDE_WEB_PROVIDER_TOML)
    }
}

/// Lazily-launched, request-serialized browser state.
struct BrowserState {
    browser: Option<Browser>,
}

/// Browser-driven LLM provider for the Claude.ai web UI.
///
/// Reuses a **persistent profile** (the user logs in once, headed, via the
/// login flow); subsequent headless requests inherit the session cookies.
/// Requests against the shared profile are serialized — Chrome locks the
/// profile directory — by holding the browser mutex for the whole turn.
pub struct WebUiRunner {
    config: WebUiConfig,
    provider: WebProviderConfig,
    available_models: Vec<String>,
    state: Arc<Mutex<BrowserState>>,
}

impl WebUiRunner {
    /// Construct a runner from explicit config + provider config.
    #[must_use]
    pub fn new(config: WebUiConfig, provider: WebProviderConfig) -> Self {
        let available_models = provider.models.available.clone();
        Self {
            config,
            provider,
            available_models,
            state: Arc::new(Mutex::new(BrowserState { browser: None })),
        }
    }

    /// Construct a Claude.ai web runner from environment configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the provider config (override path or the
    /// embedded default) cannot be loaded.
    pub fn from_env() -> Result<Self, RunnerError> {
        let config = WebUiConfig::from_env();
        let provider = match config.provider_config_path.as_ref() {
            Some(path) => WebProviderConfig::from_file(path)?,
            None => WebProviderConfig::claude_web_default()?,
        };
        Ok(Self::new(config, provider))
    }

    /// Drive a one-time interactive login in a **headed** browser.
    ///
    /// Launches Chrome (non-headless) against the persistent profile, opens the
    /// provider login page, and polls the current URL until it matches a
    /// `login_success_pattern` (the human has signed in) or `timeout` elapses.
    /// Cookies persist in the profile directory, so subsequent headless
    /// requests inherit the authenticated session.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] if the browser cannot be launched or navigated.
    pub async fn interactive_login(&self, timeout: Duration) -> Result<bool, RunnerError> {
        let mut launch = self.config.launch.clone();
        launch.headless = false;
        let mut browser = launch_browser(&launch, Some(&self.config.profile_id))
            .await
            .map_err(map_browser_error)?;

        let stealth = StealthOptions::stealth_only();
        let page = open_page_with_stealth(&browser, &self.provider.provider.login_url, &stealth)
            .await
            .map_err(map_browser_error)?;

        let deadline = Instant::now() + timeout;
        let patterns = &self.provider.provider.login_success_patterns;
        loop {
            let url = page.url().await.ok().flatten().unwrap_or_default();
            if patterns.iter().any(|p| url.contains(p)) {
                info!(%url, "web_ui: login detected — persisting session to profile");
                // Give Chrome a moment to write the auth cookies, then close
                // GRACEFULLY so the cookie DB is flushed to the profile dir.
                // Without this the browser is killed on drop and the headless
                // query later finds no session (lands on /login).
                sleep(Duration::from_secs(3)).await;
                let _ = page.close().await;
                let _ = browser.close().await;
                sleep(Duration::from_secs(1)).await;
                return Ok(true);
            }
            if Instant::now() >= deadline {
                let _ = page.close().await;
                let _ = browser.close().await;
                return Ok(false);
            }
            sleep(Duration::from_secs(2)).await;
        }
    }

    /// Render a chat request into a single prompt string for one fresh chat.
    ///
    /// A lone user message is sent verbatim; otherwise system content is
    /// prepended and each turn is labeled so the model sees the history.
    fn render_prompt(request: &ChatRequest) -> String {
        let system: Vec<&str> = request
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| m.content.as_str())
            .collect();
        let convo: Vec<&ChatMessage> = request
            .messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .collect();

        if system.is_empty() && convo.len() == 1 && convo[0].role == MessageRole::User {
            return convo[0].content.clone();
        }

        let mut out = String::new();
        if !system.is_empty() {
            out.push_str("[System]\n");
            out.push_str(&system.join("\n"));
            out.push_str("\n\n");
        }
        for m in convo {
            let label = match m.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::Tool => "Tool",
                MessageRole::System => continue,
            };
            out.push('[');
            out.push_str(label);
            out.push_str("]\n");
            out.push_str(&m.content);
            out.push_str("\n\n");
        }
        out.trim_end().to_owned()
    }

    /// Open a fresh chat, type the prompt, and submit it. Returns the live page
    /// whose streamed completion is being captured.
    async fn start_turn(&self, browser: &Browser, prompt: &str) -> Result<Page, BrowserError> {
        let stealth = StealthOptions::capture_stream(&self.provider.response.stream_url_pattern);
        info!(url = %self.provider.provider.new_chat_url, "web_ui: opening new chat");
        let page =
            open_page_with_stealth(browser, &self.provider.provider.new_chat_url, &stealth).await?;

        info!("web_ui: waiting for composer to appear");
        if let Err(e) = wait_for_selector(
            &page,
            &self.provider.composer.input_selector,
            self.config.composer_timeout,
            self.config.poll_interval,
        )
        .await
        {
            // The composer never appeared — enrich the error with the page we
            // actually landed on, which is almost always the login page (not
            // signed in) or a Cloudflare interstitial.
            let url = page.url().await.ok().flatten().unwrap_or_default();
            let _ = page.close().await;
            let hint = if self
                .provider
                .provider
                .login_success_patterns
                .iter()
                .any(|p| url.contains(p))
            {
                "page looks authenticated — the composer selector may be stale (update providers/claude_web.toml)"
            } else {
                "not authenticated — run scripts/web/claude-web-login.sh first (and confirm with EMBACLE_WEB_HEADLESS=false)"
            };
            return Err(BrowserError::timeout(format!(
                "{e}; landed on {url}; {hint}"
            )));
        }

        info!(
            chars = prompt.len(),
            "web_ui: composer ready, typing prompt"
        );
        // The composer can remount while the SPA hydrates: it passes the
        // visibility check, then querySelector momentarily misses it. Let it
        // settle, then retry the fill a few times before giving up.
        sleep(Duration::from_millis(800)).await;
        let mut fill_err = None;
        for attempt in 1..=10 {
            match fill_input_field(&page, &self.provider.composer.input_selector, prompt).await {
                Ok(()) => {
                    fill_err = None;
                    break;
                }
                Err(e) => {
                    debug!(attempt, error = %e, "web_ui: composer fill retry");
                    fill_err = Some(e);
                    sleep(Duration::from_millis(300)).await;
                }
            }
        }
        if let Some(e) = fill_err {
            let probe = probe_composer(&page).await;
            let _ = page.close().await;
            return Err(BrowserError::interaction(format!(
                "{e}; DOM probe: {probe}"
            )));
        }

        // The send button enables a beat after the composer registers text;
        // retry briefly so a not-yet-ready button isn't a hard failure.
        let mut send_err = None;
        for attempt in 1..=10 {
            match click_element(&page, &self.provider.composer.send_button_selector).await {
                Ok(()) => {
                    send_err = None;
                    break;
                }
                Err(e) => {
                    debug!(attempt, error = %e, "web_ui: send-button retry");
                    send_err = Some(e);
                    sleep(Duration::from_millis(300)).await;
                }
            }
        }
        if let Some(e) = send_err {
            let buttons = probe_buttons(&page).await;
            let _ = page.close().await;
            return Err(BrowserError::interaction(format!(
                "{e}; candidate buttons: {buttons}"
            )));
        }
        info!("web_ui: prompt submitted, awaiting streamed response");

        Ok(page)
    }
}

/// Probe the page for likely composer elements; returns a JSON summary used to
/// diagnose selector drift when the configured composer can't be filled.
async fn probe_composer(page: &Page) -> String {
    let js = r#"(function() {
        function desc(el) {
            if (!el) return null;
            var c = (el.getAttribute('class') || '').trim().split(/\s+/).filter(Boolean).slice(0, 4).join('.');
            return el.tagName.toLowerCase() + (el.id ? ('#' + el.id) : '') + (c ? ('.' + c) : '');
        }
        var ce = document.querySelectorAll('[contenteditable="true"]');
        var tb = document.querySelectorAll('[role="textbox"]');
        var ta = document.querySelectorAll('textarea');
        var pm = document.querySelectorAll('.ProseMirror');
        return JSON.stringify({
            url: location.href,
            title: document.title,
            contenteditable: ce.length,
            role_textbox: tb.length,
            textarea: ta.length,
            prosemirror: pm.length,
            first_contenteditable: desc(ce[0]),
            first_textbox: desc(tb[0]),
            first_prosemirror: desc(pm[0])
        });
    })()"#;
    match page.evaluate(js).await {
        Ok(v) => v
            .value()
            .and_then(|x| x.as_str().map(String::from))
            .unwrap_or_else(|| "<no probe value>".to_owned()),
        Err(e) => format!("<probe failed: {e}>"),
    }
}

/// List candidate buttons (aria-label / data-testid / type / text / disabled)
/// so the real send-button selector can be identified when the configured one
/// doesn't match.
async fn probe_buttons(page: &Page) -> String {
    let js = r#"(function() {
        var btns = Array.prototype.slice.call(document.querySelectorAll('button, [role="button"]'));
        var out = btns.map(function(b) {
            return {
                aria_label: b.getAttribute('aria-label'),
                testid: b.getAttribute('data-testid'),
                type: b.getAttribute('type'),
                disabled: !!b.disabled || b.getAttribute('aria-disabled') === 'true',
                text: (b.textContent || '').trim().slice(0, 24)
            };
        }).filter(function(b) {
            return b.aria_label || b.testid || b.type === 'submit'
                || /send|submit/i.test(b.text || '');
        });
        return JSON.stringify(out.slice(0, 25));
    })()"#;
    match page.evaluate(js).await {
        Ok(v) => v
            .value()
            .and_then(|x| x.as_str().map(String::from))
            .unwrap_or_else(|| "<no button probe value>".to_owned()),
        Err(e) => format!("<button probe failed: {e}>"),
    }
}

/// List the API-ish resource URLs the page has actually requested, via the
/// Performance API. Used to discover Claude.ai's real completion endpoint when
/// the configured `stream_url_pattern` captures nothing.
async fn probe_network(page: &Page) -> String {
    let js = r"(function() {
        try {
            var urls = performance.getEntriesByType('resource')
                .map(function(e) { return e.name; })
                .filter(function(u) { return /completion|chat_conversation|append_message|\/api\/|message|stream|sse|events/i.test(u); })
                .map(function(u) { try { var x = new URL(u); return x.pathname; } catch (e) { return u; } });
            // de-dupe, keep last ~15
            var seen = {}, out = [];
            for (var i = urls.length - 1; i >= 0 && out.length < 15; i--) {
                if (!seen[urls[i]]) { seen[urls[i]] = 1; out.push(urls[i]); }
            }
            return JSON.stringify({ url: location.href, api_paths: out });
        } catch (e) { return JSON.stringify({ error: String(e) }); }
    })()";
    match page.evaluate(js).await {
        Ok(v) => v
            .value()
            .and_then(|x| x.as_str().map(String::from))
            .unwrap_or_else(|| "<no network probe value>".to_owned()),
        Err(e) => format!("<network probe failed: {e}>"),
    }
}

/// Wait until a visible element matching `selector` exists, or time out.
async fn wait_for_selector(
    page: &Page,
    selector: &str,
    timeout: Duration,
    poll: Duration,
) -> Result<(), BrowserError> {
    let deadline = Instant::now() + timeout;
    loop {
        if element_exists(page, selector).await {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(BrowserError::timeout(format!(
                "selector '{selector}' did not appear within {timeout:?}"
            )));
        }
        sleep(poll).await;
    }
}

/// Extract ordered text deltas from a captured SSE body and whether a
/// done-marker was seen.
fn extract_text(body: &str, response: &ResponseConfig) -> (Vec<String>, bool) {
    let mut texts = Vec::new();
    let mut saw_done = false;
    for event in parse_sse_data(body) {
        if response.done_markers.iter().any(|m| event.contains(m)) {
            saw_done = true;
            // A control event may also carry no text — fall through to try
            // extraction, but most done markers have none.
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&event) {
            for pointer in &response.text_json_pointers {
                if let Some(s) = json.pointer(pointer).and_then(serde_json::Value::as_str) {
                    if !s.is_empty() {
                        texts.push(s.to_owned());
                    }
                    break;
                }
            }
        }
    }
    (texts, saw_done)
}

/// Map a [`BrowserError`] into the appropriate [`RunnerError`].
fn map_browser_error(e: BrowserError) -> RunnerError {
    match e {
        BrowserError::Auth { reason } => RunnerError::auth_failure(reason),
        BrowserError::Timeout { reason } => RunnerError::timeout(reason),
        BrowserError::Config { reason } => RunnerError::config(reason),
        BrowserError::Browser { reason }
        | BrowserError::Navigation { reason }
        | BrowserError::Interaction { reason } => {
            RunnerError::external_service("claude_web", reason)
        }
    }
}

#[async_trait]
impl LlmProvider for WebUiRunner {
    fn name(&self) -> &'static str {
        "claude_web"
    }

    fn display_name(&self) -> &str {
        &self.provider.provider.display_name
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::STREAMING | LlmCapabilities::SYSTEM_MESSAGES
    }

    fn default_model(&self) -> &str {
        &self.provider.models.default
    }

    fn available_models(&self) -> &[String] {
        &self.available_models
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, RunnerError> {
        let prompt = Self::render_prompt(request);
        let mut guard = self.state.clone().lock_owned().await;
        if guard.browser.is_none() {
            info!(
                headless = self.config.launch.headless,
                profile = %self.config.profile_id,
                "web_ui: launching browser"
            );
            let browser = launch_browser(&self.config.launch, Some(&self.config.profile_id))
                .await
                .map_err(map_browser_error)?;
            guard.browser = Some(browser);
        }
        let browser = guard
            .browser
            .as_ref()
            .ok_or_else(|| RunnerError::internal("browser not initialized"))?;

        let page = self
            .start_turn(browser, &prompt)
            .await
            .map_err(map_browser_error)?;

        let deadline = Instant::now() + self.config.response_timeout;
        let mut final_text = String::new();
        loop {
            if let Some(state) = read_last_capture(&page).await.map_err(map_browser_error)? {
                let (texts, saw_done) = extract_text(&state.body, &self.provider.response);
                if !texts.is_empty() {
                    final_text = texts.concat();
                }
                if state.done || saw_done {
                    break;
                }
            }
            if Instant::now() >= deadline {
                return Err(RunnerError::timeout(
                    "web response did not complete in time",
                ));
            }
            sleep(self.config.poll_interval).await;
        }

        // DOM fallback when no SSE stream was captured (URL pattern mismatch).
        if final_text.is_empty() {
            if let Some(text) =
                read_visible_text(&page, &self.provider.response.assistant_message_selector).await
            {
                final_text = text;
            }
        }

        let _ = page.close().await;

        if final_text.is_empty() {
            return Err(RunnerError::external_service(
                "claude_web",
                "empty response from web UI (check selectors / stream_url_pattern)",
            ));
        }

        Ok(ChatResponse {
            content: final_text,
            model: request
                .model
                .clone()
                .unwrap_or_else(|| self.provider.models.default.clone()),
            usage: None,
            finish_reason: Some("stop".to_owned()),
            warnings: None,
            tool_calls: None,
        })
    }

    async fn complete_stream(&self, request: &ChatRequest) -> Result<ChatStream, RunnerError> {
        let prompt = Self::render_prompt(request);
        let guard_owned = self.state.clone().lock_owned().await;
        let mut guard = guard_owned;
        if guard.browser.is_none() {
            info!(
                headless = self.config.launch.headless,
                profile = %self.config.profile_id,
                "web_ui: launching browser"
            );
            let browser = launch_browser(&self.config.launch, Some(&self.config.profile_id))
                .await
                .map_err(map_browser_error)?;
            guard.browser = Some(browser);
        }
        let browser = guard
            .browser
            .as_ref()
            .ok_or_else(|| RunnerError::internal("browser not initialized"))?;

        let page = self
            .start_turn(browser, &prompt)
            .await
            .map_err(map_browser_error)?;

        let (tx, rx) = mpsc::unbounded_channel::<Result<StreamChunk, RunnerError>>();
        let response_cfg = self.provider.response.clone();
        let poll = self.config.poll_interval;
        let deadline = Instant::now() + self.config.response_timeout;

        // Move the page + the owning guard into the task so the profile stays
        // locked (serialized) for the lifetime of the stream.
        tokio::spawn(async move {
            // Holds the profile lock for the whole turn; dropped explicitly below.
            let lock_guard = guard;
            let mut emitted = 0usize;
            let mut saw_capture = false;
            loop {
                match read_last_capture(&page).await {
                    Ok(Some(state)) => {
                        if !saw_capture {
                            info!(
                                status = state.status,
                                bytes = state.body.len(),
                                streaming = state.streaming,
                                "web_ui: completion response captured"
                            );
                            saw_capture = true;
                        }
                        let (texts, saw_done) = extract_text(&state.body, &response_cfg);
                        for delta in texts.iter().skip(emitted) {
                            let _ = tx.send(Ok(StreamChunk {
                                delta: delta.clone(),
                                is_final: false,
                                finish_reason: None,
                            }));
                        }
                        emitted = texts.len();
                        if state.done || saw_done {
                            info!(deltas = emitted, "web_ui: response complete");
                            let _ = tx.send(Ok(StreamChunk {
                                delta: String::new(),
                                is_final: true,
                                finish_reason: Some("stop".to_owned()),
                            }));
                            break;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        let _ = tx.send(Err(map_browser_error(e)));
                        break;
                    }
                }
                if Instant::now() >= deadline {
                    // Nothing matched stream_url_pattern — surface the URLs the
                    // page actually requested so the pattern can be corrected.
                    let err = if saw_capture {
                        RunnerError::timeout("web response captured but did not complete in time")
                    } else {
                        let net = probe_network(&page).await;
                        RunnerError::external_service(
                            "claude_web",
                            format!(
                                "no completion response captured in time; \
                                 stream_url_pattern may not match. API URLs seen: {net}"
                            ),
                        )
                    };
                    let _ = tx.send(Err(err));
                    break;
                }
                sleep(poll).await;
            }
            let _ = page.close().await;
            drop(lock_guard); // release the profile lock now the turn is done
            debug!("web_ui stream turn complete");
        });

        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }

    async fn health_check(&self) -> Result<bool, RunnerError> {
        let mut guard = self.state.clone().lock_owned().await;
        if guard.browser.is_none() {
            info!(
                headless = self.config.launch.headless,
                profile = %self.config.profile_id,
                "web_ui: launching browser"
            );
            let browser = launch_browser(&self.config.launch, Some(&self.config.profile_id))
                .await
                .map_err(map_browser_error)?;
            guard.browser = Some(browser);
        }
        let browser = guard
            .browser
            .as_ref()
            .ok_or_else(|| RunnerError::internal("browser not initialized"))?;

        let stealth = StealthOptions::stealth_only();
        let page =
            match open_page_with_stealth(browser, &self.provider.provider.new_chat_url, &stealth)
                .await
            {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = %e, "web_ui health check navigation failed");
                    return Ok(false);
                }
            };
        // Give the SPA a moment to settle on its post-auth URL.
        sleep(Duration::from_secs(2)).await;
        let url = page.url().await.ok().flatten().unwrap_or_default();
        let _ = page.close().await;

        let authenticated = self
            .provider
            .provider
            .login_success_patterns
            .iter()
            .any(|p| url.contains(p));
        Ok(authenticated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: MessageRole::User,
            content: content.to_owned(),
            images: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn embedded_provider_config_parses() {
        let cfg = WebProviderConfig::claude_web_default().expect("embedded config valid"); // Safe: embedded TOML, validated by this test
        assert_eq!(cfg.provider.name, "claude_web");
        assert!(!cfg.response.text_json_pointers.is_empty());
        assert!(cfg.provider.new_chat_url.contains("claude.ai"));
    }

    #[test]
    fn lone_user_message_sent_verbatim() {
        let req = ChatRequest {
            messages: vec![user_msg("hello world")],
            model: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            stop: None,
            response_format: None,
            turn_id: None,
            mcp_servers: Vec::new(),
        };
        assert_eq!(WebUiRunner::render_prompt(&req), "hello world");
    }

    #[test]
    fn system_and_history_are_labeled() {
        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: MessageRole::System,
                    content: "be terse".to_owned(),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                user_msg("hi"),
            ],
            model: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            tools: None,
            tool_choice: None,
            top_p: None,
            stop: None,
            response_format: None,
            turn_id: None,
            mcp_servers: Vec::new(),
        };
        let prompt = WebUiRunner::render_prompt(&req);
        assert!(prompt.contains("[System]"));
        assert!(prompt.contains("be terse"));
        assert!(prompt.contains("[User]"));
    }

    #[test]
    fn extract_text_handles_completion_pointer() {
        let cfg = WebProviderConfig::claude_web_default().unwrap().response; // Safe: embedded TOML, unit-tested
        let body = "data: {\"type\":\"completion\",\"completion\":\"Hel\"}\n\n\
                    data: {\"type\":\"completion\",\"completion\":\"lo\"}\n\n";
        let (texts, done) = extract_text(body, &cfg);
        assert_eq!(texts, vec!["Hel", "lo"]);
        assert!(!done);
    }

    #[test]
    fn extract_text_handles_delta_pointer_and_done() {
        let cfg = WebProviderConfig::claude_web_default().unwrap().response; // Safe: embedded TOML, unit-tested
        let body = "data: {\"delta\":{\"text\":\"hi\"}}\n\n\
                    data: {\"type\":\"message_stop\"}\n\n";
        let (texts, done) = extract_text(body, &cfg);
        assert_eq!(texts, vec!["hi"]);
        assert!(done);
    }

    #[test]
    fn map_browser_error_classifies() {
        assert!(matches!(
            map_browser_error(BrowserError::auth("x")),
            RunnerError { .. }
        ));
    }
}
