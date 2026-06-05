#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Send a prompt to Claude.ai via the headless browser runner and stream the reply.
# Requires a prior `claude-web-login.sh` so the persistent profile is authenticated.
#
# Usage: ./scripts/web/claude-web-query.sh "Your prompt here"
#
# Env:
#   EMBACLE_WEB_PROFILE_ID              profile name (default: claude-web)
#   EMBACLE_WEB_HEADLESS                set to "false" to watch the browser drive
#   EMBACLE_WEB_RESPONSE_TIMEOUT_SECS   overall response timeout (default: 180)

set -euo pipefail
cd "$(dirname "$0")/../.."

PROMPT="${1:-Say hello in one short sentence.}"
PROFILE="${EMBACLE_WEB_PROFILE_ID:-claude-web}"
PROFILE_DIR="${DRAVR_BROWSER_PROFILE_DIR:-${TMPDIR:-/tmp}/dravr-browser-profiles}/${PROFILE}"

if [ ! -d "$PROFILE_DIR" ]; then
  echo "⚠️  No saved Claude.ai session for profile '${PROFILE}' (${PROFILE_DIR})."
  echo "    Run ./scripts/web/claude-web-login.sh first, or the headless browser"
  echo "    will land on the login page and time out waiting for the composer."
  echo
fi

# Not --quiet: show compile progress so a long first build doesn't look like a hang.
# Tip: EMBACLE_WEB_HEADLESS=false ./scripts/web/claude-web-query.sh "…"  to watch the browser.
cargo run --features web-ui --example web_query -- "$PROMPT"
echo
