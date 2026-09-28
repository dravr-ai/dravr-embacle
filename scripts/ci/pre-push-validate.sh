#!/bin/bash
# ABOUTME: Pre-push gate for dravr-embacle: the shared satellite gate from dravr-build-config
# ABOUTME: Sets embacle's cargo features and execs .build/validation/satellite-pre-push-validate.sh
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 dravr.ai

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
GATE=.build/validation/satellite-pre-push-validate.sh
if [ ! -f "$GATE" ]; then
    echo "BLOCKED: $GATE is missing. Run: git submodule update --init --recursive .build"
    exit 1
fi
# The feature set CI's HTTP API test step runs: every runner feature the price-key
# check needs, plus agui and quota-http for the tests gated on them. ffi, otel and
# config-file stay out, as they do in that step.
SATELLITE_FEATURES="--features http-api,copilot-headless,copilot-sdk,web-ui,agui,quota-http" exec bash "$GATE" "$@"
