"""Tier 0: re-derive every reported number from the archive and report PASS/FAIL.

No GPU, no network, no torch. Runs in about a second. This is the fastest way for
a reader to confirm that the manuscript's numbers are the numbers in the data.
"""
from __future__ import annotations

import json
import sys

from .claims import ALL_CLAIMS, ALL_GROUPS


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{v:+.4g}" if abs(v) < 1 else f"{v:.4g}"
    return str(v)


def _preflight() -> None:
    """Fail fast and actionably if the archive is absent, rather than reporting
    every claim as an individual failure."""
    from .paths import results_dir
    rd = results_dir()
    needed = ["hosted_sweeps", "interventions", "exp1a_complete.csv"]
    missing = [n for n in needed if not (rd / n).exists()]
    if not rd.exists() or missing:
        raise SystemExit(
            f"error: results archive not usable at {rd}\n"
            f"       missing: {missing or ['(directory itself)']}\n\n"
            f"The archive ships with this repository as results/. If yours is\n"
            f"elsewhere (for example a downloaded Zenodo record), point at it:\n"
            f"       CLIFF_RESULTS=/path/to/results cliff verify")


def run(groups: list[str] | None = None, verbose: bool = True,
        json_out: str | None = None) -> int:
    _preflight()
    selected = ALL_CLAIMS if not groups else [
        c for g in groups for c in ALL_GROUPS.get(g, [])]
    if groups:
        unknown = [g for g in groups if g not in ALL_GROUPS]
        if unknown:
            raise SystemExit(f"unknown group(s) {unknown}; "
                             f"choose from {sorted(ALL_GROUPS)}")

    results, n_pass, n_fail = [], 0, 0
    by_group = {g: cs for g, cs in ALL_GROUPS.items()
                if not groups or g in groups}

    for gname, claims in by_group.items():
        if verbose:
            print(f"\n\033[1m{gname}\033[0m  ({len(claims)} claims)")
        for c in claims:
            ok, got = c.check()
            n_pass += ok
            n_fail += not ok
            results.append({"id": c.id, "group": gname, "where": c.where,
                            "claim": c.text, "expected": c.expected,
                            "got": got if isinstance(got, (int, float, bool, str)) else str(got),
                            "pass": bool(ok)})
            if verbose:
                mark = "\033[32m PASS \033[0m" if ok else "\033[31m FAIL \033[0m"
                print(f" {mark} {c.id:34s} {c.text}")
                if not ok:
                    print(f"        expected {_fmt(c.expected)}  got {_fmt(got)}"
                          f"   [{c.where}]")

    total = n_pass + n_fail
    print(f"\n{'=' * 72}")
    print(f"  {n_pass}/{total} claims verified against the archive"
          + ("" if n_fail == 0 else f"   \033[31m{n_fail} FAILED\033[0m"))
    print(f"{'=' * 72}")
    if n_fail == 0:
        scope = "Every number reported in the manuscript" if not groups \
            else f"Every claim in {', '.join(groups)}"
        print(f"  {scope} is reproduced by the data.")
        if "negative-controls" in by_group:
            print("  The 'negative-controls' group additionally confirms that the claims")
            print("  the paper withdrew are indeed unsupported by the archive.")

    if json_out:
        from pathlib import Path
        Path(json_out).write_text(json.dumps(
            {"n_pass": n_pass, "n_fail": n_fail, "claims": results}, indent=2))
        print(f"  wrote {json_out}")
    return 1 if n_fail else 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cliff verify")
    ap.add_argument("--group", action="append", dest="groups",
                    choices=sorted(ALL_GROUPS), help="restrict to a claim group")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", dest="json_out", help="also write a machine-readable report")
    a = ap.parse_args(argv)
    return run(a.groups, not a.quiet, a.json_out)


if __name__ == "__main__":
    sys.exit(main())
