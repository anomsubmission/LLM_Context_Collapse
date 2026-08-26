"""Behavioural K-sweep: measure the in-context cliff for one model.

Backend-agnostic -- needs only text generation, so it runs against local weights,
a self-hosted server, or any vendor endpoint. This is the experiment that produces
the accuracy-versus-K curves and the a64/a0 regime ratio.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..data import class_balanced_sample, load_task
from ..prompting import Cell, parse_label, render_prompt
from ..paths import outputs_dir


def sweep(backend, tasks: list[str], ks: list[int], seeds: list[int],
          n_test: int = 200, progress: bool = True) -> dict:
    cells: list[Cell] = []
    for task_id in tasks:
        data = load_task(task_id, n_test=n_test)
        labels = data.label_names
        for k in ks:
            for seed in seeds:
                t0 = time.time()
                correct = unparseable = 0
                # One demonstration draw per cell, matching the published protocol.
                demos = class_balanced_sample(data.train, k, seed)
                for ex in data.test:
                    gen = backend.generate(render_prompt(task_id, demos, ex.text))
                    pred = parse_label(gen, labels)
                    if pred is None:
                        unparseable += 1
                    elif pred == ex.label:
                        correct += 1
                n = len(data.test)
                cell = Cell(task_id, k, seed, correct / max(1, n), n, unparseable)
                cells.append(cell)
                if progress:
                    flag = "  <-- DECODE FAILURE" if cell.is_decode_failure else ""
                    print(f"  {task_id:9s} K={k:<3d} seed={seed}  acc={cell.accuracy:.3f}"
                          f"  unparseable={unparseable}/{n}  ({time.time()-t0:.0f}s){flag}")
    return summarize(backend, tasks, ks, seeds, n_test, cells)


def summarize(backend, tasks, ks, seeds, n_test, cells: list[Cell]) -> dict:
    import statistics as st
    curves: dict = {}
    for task_id in tasks:
        by_k = {}
        for k in ks:
            vals = [c.accuracy for c in cells if c.task == task_id and c.k == k]
            if vals:
                by_k[k] = st.mean(vals)
        if not by_k:
            continue
        peak_k = max(by_k, key=lambda k: by_k[k])
        entry = {"accuracy_by_k": by_k, "peak_k": peak_k, "peak": by_k[peak_k]}
        if 0 in by_k and max(by_k) in by_k and by_k[0] > 0:
            kmax = max(by_k)
            entry["a_max_over_a0"] = by_k[kmax] / by_k[0]
            entry["regime"] = ("cliff-prone" if by_k[kmax] / by_k[0] < 0.9
                               else "cliff-immune")
        curves[task_id] = entry
    return {"experiment": "behavioural_sweep", "backend": backend.name,
            "model_id": getattr(backend, "model_id", "?"),
            "tasks": tasks, "k_values": ks, "seeds": seeds, "n_test": n_test,
            "curves": curves, "cells": [c.to_dict() for c in cells]}


def main(argv=None) -> int:
    import argparse
    from ..backends import get_backend
    ap = argparse.ArgumentParser(prog="cliff sweep")
    ap.add_argument("--backend", default="hf_local")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="ag_news")
    ap.add_argument("--ks", default="0,1,2,4,8,16,32,64")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--base-url"); ap.add_argument("--api-key")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    backend = get_backend(a.backend, a.model, device=a.device,
                          load_in_4bit=a.load_in_4bit or None,
                          base_url=a.base_url, api_key=a.api_key)
    res = sweep(backend, a.tasks.split(","), [int(x) for x in a.ks.split(",")],
                [int(x) for x in a.seeds.split(",")], a.n_test)
    out = Path(a.out) if a.out else outputs_dir() / \
        f"sweep_{a.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    for task, c in res["curves"].items():
        line = f"  {task}: peak {c['peak']:.3f} at K={c['peak_k']}"
        if "a_max_over_a0" in c:
            line += f"   a_max/a0 = {c['a_max_over_a0']:.3f}  -> {c['regime']}"
        print(line)
    return 0
