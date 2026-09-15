#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Always clear a previous result, including when setup is incomplete.
rm -f score.json score.json.tmp
exec .venv/bin/python evaluate.py "$@"
