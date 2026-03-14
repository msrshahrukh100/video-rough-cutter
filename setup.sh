#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate: source .venv/bin/activate"
