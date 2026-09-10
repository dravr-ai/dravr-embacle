# Unified Harness Protocol (`uhp`)

`embacle-server` implements the [Unified Harness Protocol](https://unifiedharnessprotocol.org)
version `2026-08-11` — an open HTTP contract for driving agent harnesses behind one API. It is on
by default; there is no feature flag.

Embacle already runs twelve CLI harnesses behind a single `LlmProvider` trait, which is exactly what
the specification calls a **runner**: a server that puts existing harnesses behind the contract and
advertises them as its catalog. UHP is the standard wire format for that catalogue, so a client can
drive Claude Code, Codex, Gemini CLI, Goose or Cursor through this server without knowing which one
answered.

## Conformance

```
64/64 passed · 0 failed · 0 skipped · 0 errored
CONFORMANT — UHP 2026-08-11 (full)
```

Class `full` is the highest of the three (`core`, `extended`, `full`) and is cumulative. The
capabilities reported at `GET /v1/uhp` are the truth about this build: a flag is set only once its
conformance checks pass, never to make them pass.

## It sits beside the OpenAI API

Both surfaces define `GET /v1/models`, and the bodies are incompatible:

| Surface | `GET /v1/models` returns |
|---|---|
| OpenAI-compatible | `{"object":"list","data":[{"id":"copilot:gpt-5",…}]}` |
| UHP | `{"backends":{"copilot":{"default":"gpt-5","models":[…]}}}` |

Serving both on one path would mean content negotiation or breaking every existing caller, so UHP
mounts under its own base path. **Nothing on `/v1` changed.**

The protocol anticipates this — its published `servers:` block lists a base of `/api/harness`, and
the conformance suite takes a `--base-url`.

```bash
embacle-server --port 3000                              # base is /uhp
UHP_BASE_PATH=/api/harness embacle-server --port 3000   # or anywhere
UHP_BASE_PATH=/ embacle-server --port 3000              # or the root
```

Set `UHP_BASE_PATH=/` only when you are not serving the OpenAI API from the same process — at the
root the two definitions of `/v1/models` collide and UHP's wins.

## Endpoints

Paths are relative to `UHP_BASE_PATH`.

| Method | Path | Notes |
|---|---|---|
| `GET` | `/v1/uhp` | Discovery. **Unauthenticated** by design |
| `GET` | `/v1/harnesses` | The installed harnesses |
| `POST` | `/v1/harnesses` | Configure a named harness over a base |
| `GET` `PUT` `DELETE` | `/v1/harnesses/{id}` | Read, update, remove one |
| `GET` | `/v1/harnesses/{id}/models` | That harness's models, flat |
| `GET` | `/v1/harnesses/{id}/skills/{name}/files` | A skill bundle's whole folder |
| `GET` | `/v1/models` | The catalogue, grouped by backend |
| `POST` | `/v1/responses` | Run a task. `stream: true` for SSE |
| `GET` `DELETE` | `/v1/responses/{id}` | Read a task back, or drop it |
| `POST` | `/v1/responses/{id}/cancel` | Stop a running task |
| `GET` | `/v1/sessions` | Paginated, with `has_more` and `next_cursor` |
| `GET` `DELETE` | `/v1/sessions/{id}` | Inspect or delete a session |
| `GET` | `/v1/sessions/{id}/turns` | Its turn history |
| `GET` | `/v1/sessions/{id}/files` | Artifacts the session produced |
| `POST` `GET` `DELETE` | `/v1/sessions/{id}/share` | Mint, read back, revoke a public view |
| `GET` | `/share/{share_id}` | The shared view. **Unauthenticated** |
| `GET` | `/v1/containers/{id}/files/{id}/content` | Download an artifact |
| `DELETE` | `/v1/traces/{id}` | Legacy alias for session delete |

## Authentication

The same `EMBACLE_API_KEY` as the rest of the server. Two endpoints are deliberately open:

- **`GET /v1/uhp`** — a client must be able to learn whether this is a UHP server, and which
  versions it speaks, *before* deciding what credential to present.
- **`GET /share/{share_id}`** — a shared view only its minter can open has not been published to
  anyone.

Everything else refuses an absent or unrecognised token with `401` and
`{"error":{"type":"authentication_error",…}}`.

A share id is **not** a credential. It lives in its own namespace, names nothing the API looks up,
and presenting it as a bearer token fails exactly as any unknown token does.

## Running a task

```bash
# Non-streaming
curl http://localhost:3000/uhp/v1/responses \
  -H "Authorization: Bearer $EMBACLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "Reply with exactly: ok", "stream": false}'

# Streaming — events arrive as the runner produces them
curl -N http://localhost:3000/uhp/v1/responses \
  -H "Authorization: Bearer $EMBACLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "Count to three", "stream": true}'

# Continue the session a previous task opened
curl http://localhost:3000/uhp/v1/responses \
  -H "Authorization: Bearer $EMBACLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "And the next one?", "previous_response_id": "resp_…"}'
```

The stream is progressive rather than assembled at the end: `response.created` goes out before the
harness is asked, each chunk becomes a `response.output_text.delta`, and exactly one terminal event
carries the complete response. `sequence_number` starts at 0 and increases by one, so a client can
detect a dropped event rather than silently rendering a gap.

## Artifacts

Every task runs with its working directory set to its own session's folder, so a file appearing
there was written by that session and nothing else. Those files are the session's artifacts.

```bash
curl http://localhost:3000/uhp/v1/sessions/$SID/files -H "Authorization: Bearer $KEY"
curl http://localhost:3000/uhp/v1/containers/$SID/files/$FID/content -H "Authorization: Bearer $KEY"
```

Downloads carry `X-Content-Type-Options: nosniff`, `Content-Type: application/octet-stream` and
`Content-Disposition: attachment`. An artifact is content a model was steered into producing, so it
must never render as a page on this origin.

Deleting a session removes its working folder, so an artifact never outlives what produced it. The
specification couples cancel and delete here deliberately: any in-flight task is cancelled first,
because the alternative is a running task writing into storage with no owner.

`UHP_WORKDIR` sets where session folders live (default: a subdirectory of the system temp dir).

## Harnesses you configure

Beyond the harnesses discovery finds by probing installed binaries, a client can register named
configurations over one of those bases — so "the reviewer" and "the summariser" can be distinct
harnesses running on the same CLI.

```bash
curl http://localhost:3000/uhp/v1/harnesses \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"name": "reviewer", "base": "claude-code", "disabled_tools": ["WebSearch"]}'
```

A base this server has no installed harness for is refused **when the harness is created**, with the
supported list in `error.detail.supported`. Accepting one it cannot run would only move the failure
to task time, after the client has committed. The base is immutable on update, because changing it
would silently repoint every session already running on that harness.

Skill bundles round-trip as whole folders — a member carries its bytes as `content`, or as
`content_b64` when it is not text. A bundle with no `SKILL.md` is refused at config time rather than
stored and silently ignored at run time.

## Verifying it yourself

The suite is Apache-2.0 and ships inside the reference implementation. It runs real agent tasks by
design, so a full pass spends real model calls and several minutes.

```bash
git clone --depth 1 https://github.com/HarnessRouter/harnessrouter.git
python3 -m venv venv && ./venv/bin/pip install -e harnessrouter/protocol/conformance

EMBACLE_API_KEY=secret embacle-server --port 3000 &
./venv/bin/uhp-conformance --base-url http://127.0.0.1:3000/uhp \
  --api-key secret --class full
```

`uhp-conformance` is not on PyPI; the documented `pip install -e protocol/conformance` is a path
install from that clone. Use `--class core` for a faster pass, or `--only <ID>` to rerun one check
while fixing it.
