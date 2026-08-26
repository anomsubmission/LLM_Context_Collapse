#!/usr/bin/env bash
# Fast offline check: no GPU, no torch, no network. Seconds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hr() { printf '\n%s\n' "------------------------------------------------------------"; }

hr; echo "1/4  environment and what is runnable"
python3 scripts/check_env.py || true

hr; echo "2/4  all 94 reported numbers, plus the withdrawn-claim controls"
python3 -m cliff verify

hr; echo "3/4  figures from the archive"
python3 -m cliff figures

hr; echo "4/4  exercise the sweep pipeline with no model at all"
python3 -m cliff sweep --backend echo --model echo --tasks ag_news \
  --ks 0,1,4 --seeds 0 --n-test 20

hr; echo "SMOKE TEST PASSED"
