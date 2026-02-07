#!/usr/bin/env bash
# Launch the Dictate menu bar app.
# Add this script to System Settings → General → Login Items to start on boot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

exec python -m dictate.menubar_main
