#!/usr/bin/env python3
"""Verify every reported number, then check the figures redraw.

    python3 scripts/verify.py

A thin wrapper over the project's own `cliff` CLI, so this repository has the same entry
points as the others in this collection. Equivalent and better documented:

    cliff verify      # re-derive all 94 reported numbers from the archive
    cliff figures     # redraw Figures 1, 3 and 4

`cliff verify` needs no GPU, no torch and no network -- it re-derives the paper's numbers
from the archived result files in about a second. It also carries a `negative-controls`
group that confirms the claims the paper *withdrew* are genuinely unsupported by the
archive, which is a stronger discipline than checking only the surviving claims: it stops a
withdrawn claim from quietly reappearing.

Exit 0 if all claims verify and the figures render.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(args: list[str], what: str) -> bool:
    print("=" * 74)
    print(what)
    print("=" * 74)
    r = subprocess.run([sys.executable, "-m", "cliff", *args], cwd=ROOT)
    print()
    return r.returncode == 0


ok = run(["verify"], "94 reported numbers, re-derived from the archive")
ok = run(["figures"], "Figures 1, 3 and 4, redrawn from the archive") and ok

if not ok:
    print("FAILED: see above", file=sys.stderr)
    raise SystemExit(1)
print("VERIFIED: all reported numbers reproduce and every figure redraws from the archive.")
