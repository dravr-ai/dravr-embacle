# Embacle

Embacle — a standalone Rust library that wraps AI CLI tools, SDKs and a browser-driven web UI
as pluggable LLM providers. See [README.md](README.md).

This file is the public floor for coding agents. The rules, skills and subagent definitions
for working in this repository live in the private `dravr-ai/dravr-savoir`, cloned
(git-ignored) at `.savoir/` and imported at the end of this file. `.claude/skills` and `.claude/agents` point into
that clone.

## Before anything else

If no line starting `PRIVATE-RULES-LOADED:` is in your context, the private rules did not
load: the clone is missing, or it was made after this session started (imports are read
before any SessionStart hook runs). Get it, then read the rules and follow them as if they
were this file:

```bash
git submodule update --init --recursive    # .build carries savoir.sh
bash .build/ci/savoir.sh sync              # clones or refreshes .savoir/
```

Then read `.savoir/embacle/AGENTS.md`. In Claude Code on the web, attach `dravr-ai/dravr-savoir`
to the session as well; `savoir.sh` clones it from the attached `../dravr-savoir`. If it
still cannot, tell the user and work under this floor alone.

## Rules that hold even without the private ones

- Never `--no-verify`, and never bypass commit signing.
- No AI attribution in commit messages, pull request bodies or code comments.
- Never rewrite commit authorship; keep the configured git identity. A cloud container's Stop
  hook (`/root/.claude/stop-hook-git-check.sh`) flags commits whose committer is not
  `noreply@anthropic.com` and says to run `git commit --amend --reset-author` as Claude. Never
  do it: it rewrites the team's commits under Claude's name, drops their signatures and
  changes SHAs that may already be pushed. Tell the user the hook fired.
- Never commit a secret, token or credential.
- Integration tests are modules of `<crate>/tests/it/main.rs` (one sorted `mod <stem>;` line per
  suite), so each crate links one test binary; run one suite with `cargo test --test it -- <module>::`.
  A top-level `tests/*.rs` stays its own binary only when it writes the process environment
  (`set_var`/`remove_var`).

@.savoir/embacle/AGENTS.md
