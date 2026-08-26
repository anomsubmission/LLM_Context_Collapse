"""Readers for the archived result files. Pure stdlib: no torch, no network.

Every loader raises a descriptive error if a cell is absent. Nothing in this module
substitutes a default for missing data -- that is the defect that put three
unmeasured values into an earlier draft of the paper's figures.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .paths import results_dir


class MissingCell(KeyError):
    """A requested (file, task, K) cell does not exist in the archive."""


@lru_cache(maxsize=None)
def _json(rel: str) -> dict:
    p = results_dir() / rel
    if not p.exists():
        raise FileNotFoundError(f"missing archive file: {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- hosted sweeps
@lru_cache(maxsize=None)
def hosted(fname: str) -> dict:
    """{(task, k): mean_accuracy} for one hosted-API behavioural sweep."""
    d = _json(f"nmi_hosted/{fname}")
    agg = defaultdict(list)
    for c in d["cells"]:
        agg[(c["task"], c["k"])].append(c["accuracy"])
    return {k: st.mean(v) for k, v in agg.items()}


def hosted_acc(fname: str, task: str, k: int) -> float:
    tbl = hosted(fname)
    if (task, k) not in tbl:
        raise MissingCell(f"{fname}: no cell {task} K={k}")
    return tbl[(task, k)]


def hosted_ratio(fname: str, task: str, num: int = 64, den: int = 0) -> float:
    return hosted_acc(fname, task, num) / hosted_acc(fname, task, den)


def hosted_peak(fname: str, task: str) -> tuple[int, float]:
    tbl = hosted(fname)
    ks = {k: v for (t, k), v in tbl.items() if t == task}
    if not ks:
        raise MissingCell(f"{fname}: task {task} absent")
    kbest = max(ks, key=lambda k: ks[k])
    return kbest, ks[kbest]


# ---------------------------------------------------------------- panel csv
@lru_cache(maxsize=None)
def panel_csv() -> list[dict]:
    p = results_dir() / "exp1a_complete.csv"
    with p.open() as fh:
        return list(csv.DictReader(fh))


def panel_acc(model: str, task: str, k: int) -> float:
    for r in panel_csv():
        if r["model"] == model and r["task"] == task and int(r["k"]) == k:
            return float(r["mean_acc"])
    raise MissingCell(f"exp1a_complete.csv: no cell {model}/{task} K={k}")


def zero_shot(model: str, task: str) -> float:
    d = _json("exp1e_results.json")
    if model not in d or task not in d[model]:
        raise MissingCell(f"exp1e_results.json: no zero-shot for {model}/{task}")
    return d[model][task]


# ---------------------------------------------------------------- interventions
@lru_cache(maxsize=None)
def _cells(fname: str, kind: str) -> dict:
    d = _json(f"nmi_new_causal/{fname}")
    out: dict = defaultdict(dict)
    for c in d[f"{kind}_cells"]:
        out[(c["task"], c["k"])][c["seed"]] = c
    return dict(out)


def per_seed_recovery(fname: str, task: str, k: int) -> list[float]:
    """Paired per-seed (after - before). Raises if the cell is absent."""
    B, A = _cells(fname, "before"), _cells(fname, "after")
    if (task, k) not in B or (task, k) not in A:
        raise MissingCell(f"{fname}: no cell {task} K={k}")
    seeds = sorted(set(B[(task, k)]) & set(A[(task, k)]))
    if not seeds:
        raise MissingCell(f"{fname}: no paired seeds for {task} K={k}")
    return [A[(task, k)][s]["accuracy"] - B[(task, k)][s]["accuracy"] for s in seeds]


def recovery(fname: str, task: str, k: int) -> float:
    return st.mean(per_seed_recovery(fname, task, k))


def is_decode_failure(fname: str, task: str, k: int, kind: str = "after") -> bool:
    cells = _cells(fname, kind)
    if (task, k) not in cells:
        raise MissingCell(f"{fname}: no {kind} cell {task} K={k}")
    return all(c["n_unparseable"] == c["n_test"] for c in cells[(task, k)].values())


def arm_acc(fname: str, task: str, k: int, kind: str) -> float:
    cells = _cells(fname, kind)
    if (task, k) not in cells:
        raise MissingCell(f"{fname}: no {kind} cell {task} K={k}")
    return st.mean(c["accuracy"] for c in cells[(task, k)].values())


def cell_exists(fname: str, task: str, k: int) -> bool:
    try:
        per_seed_recovery(fname, task, k)
        return True
    except (MissingCell, FileNotFoundError):
        return False


# ---------------------------------------------------------------- mechanism
def mechanism() -> dict:
    return _json("nmi_new_causal/t5_llama31_8b_ag_news_results.json")


def induction_band_means(lo: int = 8, hi: int = 16) -> tuple[float, float]:
    ih = mechanism()["induction_score_by_layer"]
    band = range(lo, hi)
    inside = [ih[i] for i in band]
    outside = [ih[i] for i in range(len(ih)) if i not in band]
    return st.mean(inside), st.mean(outside)


# ---------------------------------------------------------------- misc
def component_csv() -> list[dict]:
    p = results_dir() / "component_dissociation.csv"
    with p.open() as fh:
        return list(csv.DictReader(fh))


def backward_transfer() -> list[float]:
    return [c["icl_backward_transfer"] for c in _json("exp1b_results.json")]


def zero_accuracy_census() -> tuple[int, int, int]:
    """(n_zero_cells, n_total_cells, n_zero_that_are_fully_unparseable)."""
    total = zero = fully = 0
    for p in sorted((results_dir() / "nmi_new_causal").glob("*.json")):
        d = json.loads(p.read_text())
        for kind in ("before_cells", "after_cells"):
            for c in d.get(kind, []) or []:
                total += 1
                if c.get("accuracy") == 0.0:
                    zero += 1
                    if c.get("n_unparseable") == c.get("n_test"):
                        fully += 1
    return zero, total, fully


def welch(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Welch's unequal-variance t-test. Returns (t, df, two-sided p)."""
    import math
    nx, ny = len(x), len(y)
    vx, vy = st.variance(x), st.variance(y)
    se = math.sqrt(vx / nx + vy / ny)
    t = (st.mean(x) - st.mean(y)) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return t, df, 2.0 * _t_sf(abs(t), df)


def _t_sf(t: float, df: float) -> float:
    """Upper-tail Student-t via the regularized incomplete beta, stdlib only."""
    import math
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # continued-fraction betainc (Lentz); adequate for the precision we report
    def betacf(a, b, x, itmax=300, eps=3e-16):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - lbeta) \
            * betacf(b, a, 1.0 - x) / b
    return 0.5 * ib
