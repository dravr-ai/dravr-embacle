# Browser-driven web provider (`web-ui`)

The `web-ui` feature adds `WebUiRunner` — an `LlmProvider` that drives the
**Claude.ai web UI** through a headless Chrome session instead of a CLI binary
or HTTP API. It reuses a **persistent browser profile**, so you log in once
(interactively) and subsequent requests run headless against the saved session.

It is built on the standalone [`dravr-browser`](https://github.com/dravr-ai/dravr-browser)
crate (launch, stealth, CDP input, streaming network capture).

## ⚠️ Terms of Service

Automating a consumer web UI with stored credentials is generally **against the
provider's consumer Terms of Service**, and accounts driven this way may be
rate-limited or flagged. This provider is intended for **your own account** in
research / personal contexts. It is **off by default** (gated behind the
`web-ui` feature) and never automates credential entry — you sign in yourself.

## How it works

1. **Login (once):** a headed Chrome opens the Claude.ai login page; you sign
   in. Cookies persist in the profile directory.
2. **Query:** for each request a fresh `claude.ai/new` chat is opened, the
   prompt is typed into the composer, and the send button is clicked.
3. **Capture:** an injected hook tees the streaming completion response (SSE);
   `complete_stream` emits deltas as they arrive, `complete` returns the joined
   text. A DOM read is the fallback if the network pattern doesn't match.

Selectors and SSE-extraction rules live in `providers/claude_web.toml` — UI
changes are config, not code. Point `EMBACLE_WEB_PROVIDER_CONFIG` at your own
TOML to override.

## Quick start (scripts)

```bash
# 1. Log in once (opens a real browser window)
./scripts/web/claude-web-login.sh

# 2. Ask something (streams the reply)
./scripts/web/claude-web-query.sh "Explain the CAP theorem in two sentences."
```

## Configuration (env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBACLE_WEB_PROFILE_ID` | `claude-web` | Persistent profile name |
| `EMBACLE_WEB_HEADLESS` | `true` | `false` runs headed (login / debugging) |
| `EMBACLE_WEB_PROVIDER_CONFIG` | — | Path to a provider TOML override |
| `EMBACLE_WEB_RESPONSE_TIMEOUT_SECS` | `180` | Overall response timeout |
| `DRAVR_BROWSER_PROFILE_DIR` | `$TMPDIR/dravr-browser-profiles` | Profile base dir |
| `CHROME_PATH` | auto-detect | Chrome/Chromium binary |

## Library usage

```rust,no_run
use embacle::WebUiRunner;
use embacle::types::{ChatMessage, ChatRequest, LlmProvider, MessageRole};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let runner = WebUiRunner::from_env()?;
let request = ChatRequest {
    messages: vec![ChatMessage {
        role: MessageRole::User,
        content: "Hello!".to_owned(),
        images: None, tool_calls: None, tool_call_id: None, name: None,
    }],
    model: None, temperature: None, max_tokens: None, stream: false,
    tools: None, tool_choice: None, top_p: None, stop: None,
    response_format: None, turn_id: None,
};
let response = runner.complete(&request).await?;
println!("{}", response.content);
# Ok(()) }
```

## Server usage — REST + MCP (`embacle-server`)

`embacle-server` ships with the `web-ui` feature **on by default**, so the
`claude_web` provider is reachable over both the OpenAI-compatible REST API and
the MCP surface (both route through the same provider factory; no extra wiring).

**One-time host setup:** the server drives a real browser, so the host needs a
**pre-authenticated profile**. Run the interactive login once on the server
machine (a browser window opens — sign in to Claude.ai):

```bash
EMBACLE_WEB_HEADLESS=false cargo run --example web_login --features web-ui
# the `claude-web` profile under $TMPDIR/dravr-browser-profiles now holds the session
```

**REST** — select the provider with the `claude_web` model prefix:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "claude_web:claude-opus-4-8",
        "messages": [{"role":"user","content":"Explain the CAP theorem in two sentences."}],
        "stream": true
      }'
```

(`"model": "claude_web"` also works — it uses the account default model.)

**MCP** — the same `claude_web` provider/model routing is exposed through the
server's MCP transport (`dravr-tronc`); point an MCP client at the server and
request the `claude_web` model the same way.

**Operational notes for server use:**
- Requests against the single `claude-web` profile are **serialized** (Chrome
  locks the profile dir) — size capacity accordingly, or run multiple profiles.
- The session can expire; when it does, re-run the interactive login on the host.
- This pulls `chromiumoxide` into the server binary. To build a lean server
  without it, disable default features: `cargo build -p embacle-server --no-default-features --features mcp-tools`.

## Limitations

- **Serialized per profile** — Chrome locks the profile directory, so requests
  against one profile run one at a time.
- **Model selection is advisory** — the web UI picks the model by account/plan;
  `request.model` is recorded but not enforced.
- **Stateless per request** — each call opens a fresh chat; conversation history
  is replayed into the prompt, not resumed server-side.
- **Live selectors may drift** — the Claude.ai DOM/endpoints can change; update
  `providers/claude_web.toml` rather than the Rust code.
