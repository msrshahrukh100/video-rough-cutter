#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
[ -f "$PYTHON" ] || { echo "Run ./setup.sh first"; exit 1; }
"$PYTHON" "$SCRIPT_DIR/src/pipeline.py" "$@"
