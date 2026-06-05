#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-time interactive login for the browser-driven Claude.ai web runner.
# Opens a real Chrome window against the persistent profile; sign in once and
# the session cookies persist for subsequent (headless) queries.
#
# Usage: ./scripts/web/claude-web-login.sh
#
# Env:
#   EMBACLE_WEB_PROFILE_ID   profile name / dir (default: claude-web)
#   DRAVR_BROWSER_PROFILE_DIR base dir for profiles (default: $TMPDIR/dravr-browser-profiles)

set -euo pipefail
cd "$(dirname "$0")/../.."

export EMBACLE_WEB_HEADLESS=false
echo "Launching login browser (profile: ${EMBACLE_WEB_PROFILE_ID:-claude-web})."
echo "Sign in to Claude.ai in the window that opens; this exits once you're in."
echo "(first run compiles for a minute or two — that's normal, not a hang)"
cargo run --features web-ui --example web_login
