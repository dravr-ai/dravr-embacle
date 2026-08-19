# embacle-tool-host

Host **your own tools** to an ACP agent over a loopback MCP endpoint.

## Why

An ACP agent such as `copilot --acp` runs its own tool loop inside its own
subprocess. It never asks its caller to execute a tool — it executes them itself
and reports afterwards, and that report carries no tool name: ACP's
`session/update` notification has `toolCallId`, `title`, `kind` and `status`, and
nothing identifying which tool ran.

So a caller that wants the agent to use *its* tools has one channel: declare an
MCP server in `session/new`. The agent then speaks MCP to it, and `tools/call`
carries the name and arguments in full fidelity.

That channel cannot be an in-process callback. With stdio the agent forks the
server itself, so it is a grandchild process in another address space, and the
ACP frame carries only `command`/`args`/`env` — no socket, no file descriptor,
no back-channel. Reaching a caller's `McpToolExecutor` needs a real listener.
Loopback HTTP is the smallest one that works.

This crate is separate from `embacle` because the root crate holds the line
"No HTTP dependencies in core", and its `ffi` feature ships a `staticlib`
compiled `panic = "abort"`. Consumers that enable `copilot-headless` without
hosting tools pay nothing for this.

## Use

```rust,ignore
let host = ToolHost::bind(ToolHostConfig {
    server_name: "dravr".to_owned(),
    ..ToolHostConfig::default()
}).await?;

// Per turn: publish a surface, get a revocable credential.
// StaticSurface for a fixed tool list; implement ToolSurface yourself when
// what is visible depends on state that can change mid-turn.
let surface = Arc::new(StaticSurface::new(my_tool_definitions(), my_executor));
let session = host.open_session(surface);
let request = ChatRequest::new(messages).with_mcp_servers(session.mcp_servers());
let response = runner.converse(&request).await?;

// `session` drops here — the bearer is revoked at the same instant.
```

## The session guard

`ToolSession` is a guard, not an id. **Dropping it revokes the bearer
immediately**, so a turn that ends — normally, by error, or because the caller
went away — leaves no live credential an orphaned agent subprocess can still
spend on an irreversible action.

`session.calls_served()` reports how many tool calls the turn actually made.
Zero, on a turn whose reply claimed to have consulted data, is the signal that
it did not.

## The surface is asked every time

`ToolSurface` has two halves and both are consulted per request:

```rust,ignore
async fn list_tools(&self) -> Vec<McpToolDefinition>;
async fn call(&self, tool_name: &str, arguments: &Value) -> ToolOutcome;
```

Fixing the tool list when the session opens would be cheaper and is wrong. A
caller that withholds a tool while an interview is running, or gates on a role
or a quota, needs the answer for *the moment the agent asks* — a gate that
cannot be re-asked is a gate that silently stops applying. A tool withdrawn
mid-session disappears from the listing and stops being callable, with no
session reopen.

`ToolOutcome` has three states rather than two. A tool that ran and declined —
a quota refusal, a guard saying no — is neither success nor transport failure:
the model should read the reason and adapt. `Result` cannot say that without
throwing away either the flag or the text.
