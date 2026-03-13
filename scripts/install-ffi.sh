#!/usr/bin/env bash
# ABOUTME: Build and install the embacle FFI static library for Swift/C integration
# ABOUTME: Installs libembacle.a and embacle.h to PREFIX (default /usr/local)
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 dravr.ai

set -euo pipefail

PREFIX="${PREFIX:-/usr/local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--prefix <path>] [--uninstall]

Options:
  --prefix <path>   Install prefix (default: /usr/local)
  --uninstall       Remove installed files
  -h, --help        Show this help

Examples:
  $(basename "$0")                           # Install to /usr/local
  $(basename "$0") --prefix \$HOME/.local    # Install to ~/.local
  sudo $(basename "$0")                      # Install to /usr/local (with sudo)
  $(basename "$0") --uninstall               # Remove installed files
EOF
    exit 0
}

UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if $UNINSTALL; then
    echo "Removing embacle FFI from ${PREFIX}..."
    rm -f "${PREFIX}/lib/libembacle.a"
    rm -f "${PREFIX}/include/embacle.h"
    echo "Done."
    exit 0
fi

# Verify we're in the repo root
if [[ ! -f "${REPO_ROOT}/Cargo.toml" ]]; then
    echo "Error: Cargo.toml not found in ${REPO_ROOT}" >&2
    echo "Run this script from the embacle repository." >&2
    exit 1
fi

# Check for cargo
if ! command -v cargo &>/dev/null; then
    echo "Error: cargo not found. Install Rust from https://rustup.rs" >&2
    exit 1
fi

echo "Building embacle FFI static library..."
cd "${REPO_ROOT}"
cargo build --release --features ffi

echo "Installing to ${PREFIX}..."
install -d "${PREFIX}/lib"
install -d "${PREFIX}/include"
install -m 644 target/release/libembacle.a "${PREFIX}/lib/"
install -m 644 include/embacle.h "${PREFIX}/include/"

# Verify
if [[ -f "${PREFIX}/lib/libembacle.a" ]] && [[ -f "${PREFIX}/include/embacle.h" ]]; then
    SIZE=$(du -h "${PREFIX}/lib/libembacle.a" | awk '{print $1}')
    echo ""
    echo "Installed:"
    echo "  ${PREFIX}/lib/libembacle.a (${SIZE})"
    echo "  ${PREFIX}/include/embacle.h"
    echo ""
    echo "Swift/SPM projects can now link against embacle via:"
    echo "  .systemLibrary(name: \"CEmbacle\")"
else
    echo "Error: installation verification failed" >&2
    exit 1
fi
