"""Filesystem layout. No absolute paths anywhere else in the package."""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent


def results_dir() -> Path:
    """Directory holding the archived result files.

    Override with CLIFF_RESULTS=/path/to/results to point at a different archive
    (for example a freshly downloaded Zenodo record, or your own re-run output).
    """
    env = os.environ.get("CLIFF_RESULTS")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "results"


def figures_dir() -> Path:
    env = os.environ.get("CLIFF_FIGURES")
    p = Path(env).expanduser().resolve() if env else REPO_ROOT / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    """Where freshly-run experiments write. Never the archived results dir."""
    env = os.environ.get("CLIFF_OUTPUTS")
    p = Path(env).expanduser().resolve() if env else REPO_ROOT / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p
