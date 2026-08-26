#!/usr/bin/env python3
"""Pre-flight check. Wraps `cliff doctor`, which already reports the environment and
exactly which subcommands are runnable, and adds the storage round-trip.

    python3 scripts/check_env.py
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

print("== cliff doctor")
rc = subprocess.run([sys.executable, "-m", "cliff", "doctor"], cwd=ROOT).returncode

results = os.environ.get("RESULTS_DIR", "./outputs")
print(f"\n== storage: RESULTS_DIR={results}")
rc |= subprocess.run([sys.executable, "runner/storage.py", "--check", results],
                     cwd=ROOT).returncode

print("\nenvironment ok" if rc == 0 else "\nproblems above", file=sys.stderr if rc else sys.stdout)
raise SystemExit(rc)
