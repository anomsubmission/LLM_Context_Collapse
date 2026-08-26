#!/usr/bin/env bash
# Re-derive the reported numbers and redraw the figures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 scripts/verify.py
