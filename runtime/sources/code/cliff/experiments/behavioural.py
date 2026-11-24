"""Behavioural K-sweep: measure the in-context cliff for one model.

Backend-agnostic -- needs only text generation, so it runs against local weights,
a self-hosted server, or any vendor endpoint. This is the experiment that produces
the accuracy-versus-K curves and the a64/a0 regime ratio.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..data import class_balanced_sample, exact_balanced_sample, exclude_examples, load_task
from ..prompting import Cell, parse_label, parse_label_strict, render_prompt
from ..paths import outputs_dir


def sweep(backend, tasks: list[str], ks: list[int], seeds: list[int],
          n_test: int = 200, progress: bool = True, *,
          sampling_protocol: str = "exact_k_v1", n_train: int | None = 512,
          dataset_revision: str | None = None, train_split: str | None = None,
          test_split: str | None = None, query_split_seed: int = 0,
          max_new_tokens: int = 10, run_id: str | None = None,
          condition: str = "baseline", training_run_id: str | None = None,
          record_sink=None, include_records: bool = True,
          label_scores: bool = False, parser: str = "strict",
          allow_unmeasured: bool = False) -> dict:
    """Run a measured sweep; exact-K and historical sampling are distinct protocols.

    A rejected query invalidates its complete cell. Failed generations remain in
    the denominator; rejected cells never enter scientific curves. K=0 is
    evaluated once per query under the exact-K protocol, irrespective of draw IDs.
    ``record_sink`` can stream each dictionary to an immutable run writer.
    """
    if sampling_protocol not in {"exact_k_v1", "legacy_floor"}:
        raise ValueError("unknown sampling_protocol")
    if parser not in {"strict", "legacy_prefix"}:
        raise ValueError("unknown label parser")
    if not tasks or not ks or not seeds or len(set(ks)) != len(ks) or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty tasks, unique K values and unique draw seeds are required")
    if min(ks) < 0:
        raise ValueError("K must be nonnegative")
    if sampling_protocol != "legacy_floor" and not hasattr(backend, "generate_record") and not allow_unmeasured:
        raise ValueError("scientific sweeps require generate_record measurements; "
                         "use a measured backend or explicitly allow_unmeasured for a technical pilot")
    parse = parse_label_strict if parser == "strict" else parse_label
    cells: list[Cell] = []
    records = []
    datasets = {}
    for task_id in tasks:
        kwargs = {"n_train": n_train, "n_test": n_test, "shard_seed": query_split_seed}
        if dataset_revision is not None:
            kwargs["revision"] = dataset_revision
        if train_split is not None:
            kwargs["train_split"] = train_split
        if test_split is not None:
            kwargs["test_split"] = test_split
        data = load_task(task_id, **kwargs)
        labels = data.label_names
        datasets[task_id] = {**data.metadata, "actual_train_count": len(data.train),
                             "actual_query_count": len(data.test), "chance_rate": data.chance_rate,
                             "query_majority_rate": data.majority_rate}
        # Exclude the entire evaluation set, preserving one shared draw per cell.
        pool = exclude_examples(data.train, data.test)
        for k in ks:
            draw_seeds = seeds[:1] if k == 0 and sampling_protocol == "exact_k_v1" else seeds
            for seed in draw_seeds:
                t0 = time.perf_counter()
                correct = unparseable = truncated = rejected = 0
                sampler = exact_balanced_sample if sampling_protocol == "exact_k_v1" else class_balanced_sample
                demos = sampler(pool, k, seed)
                for ex in data.test:
                    prompt = render_prompt(task_id, demos, ex.text)
                    t_query = time.perf_counter()
                    if hasattr(backend, "generate_record"):
                        generation = backend.generate_record(prompt, max_new_tokens)
                    else:
                        generation = {"raw_text": backend.generate(prompt, max_new_tokens),
                                      "measurement_status": "unmeasured_legacy_backend",
                                      "input_tokens_before": None, "input_tokens_after": None,
                                      "generation_tokens": None, "truncated": None,
                                      "rejected": False, "rejection_reason": None,
                                      "retained_instruction": None, "retained_query": None,
                                      "latency_seconds": time.perf_counter() - t_query}
                    pred = None if generation.get("rejected") else parse(generation["raw_text"], labels)
                    rejected += bool(generation.get("rejected"))
                    truncated += bool(generation.get("truncated"))
                    if not generation.get("rejected"):
                        unparseable += pred is None
                        correct += pred == ex.label
                    scores = None
                    if label_scores and not generation.get("rejected"):
                        if not hasattr(backend, "score_labels"):
                            raise ValueError("label_scores requires a backend with complete-label likelihoods")
                        scores = backend.score_labels(prompt, labels)
                    record = {**generation, "run_id": run_id, "task": task_id,
                              "query_id": ex.example_id, "query_hash": ex.text_hash,
                              "gold_label": ex.label, "gold_label_index": ex.label_index,
                              "demonstration_ids": [d.example_id for d in demos],
                              "demonstration_hashes": [d.text_hash for d in demos],
                              "demonstration_labels": [d.label for d in demos],
                              "requested_k": k, "realized_k": len(demos),
                              "retained_k": None if generation.get("truncated") else len(demos),
                              "draw_id": seed, "demo_seed": seed, "query_split_seed": query_split_seed,
                              "training_run_id": training_run_id, "condition": condition,
                              "sampling_protocol": sampling_protocol,
                              "model_id": getattr(backend, "model_id", "?"),
                              "prediction": pred, "parser": parser,
                              "parse_status": "rejected" if generation.get("rejected") else
                                              ("parseable" if pred is not None else "unparseable"),
                              "correct": pred == ex.label, "label_scores": scores}
                    if record_sink is not None:
                        record_sink(record)
                    if include_records:
                        records.append(record)
                n = len(data.test)
                cell = Cell(task_id, k, seed, None if rejected or n == 0 else correct / n,
                            n, unparseable, truncated, rejected, len(demos))
                cells.append(cell)
                if progress:
                    flag = "  <-- DECODE FAILURE" if cell.is_decode_failure else ""
                    accuracy = "invalid" if cell.accuracy is None else f"{cell.accuracy:.3f}"
                    print(f"  {task_id:9s} K={k:<3d} realized={len(demos)} draw={seed} acc={accuracy}"
                          f" unparseable={unparseable}/{n} rejected={rejected}"
                          f" ({time.perf_counter()-t0:.0f}s){flag}")
    result = summarize(backend, tasks, ks, seeds, n_test, cells,
                       regime_protocol="prospective_decline_v1" if sampling_protocol == "exact_k_v1"
                       else "legacy_endpoint_ratio")
    result.update({"sampling_protocol": sampling_protocol, "parser": parser,
                   "datasets": datasets, "predictions": records,
                   "query_split_seed": query_split_seed,
                   "scientific_evidence": (bool(records) and all(
                       r.get("measurement_status") == "measured" and not r.get("truncated")
                       and not r.get("rejected") for r in records)) if include_records else None,
                   "template": getattr(backend, "template", "backend_defined")})
    return result


def summarize(backend, tasks, ks, seeds, n_test, cells: list[Cell], *,
              regime_protocol: str = "legacy_endpoint_ratio") -> dict:
    import statistics as st
    curves: dict = {}
    for task_id in tasks:
        by_k = {}
        for k in ks:
            vals = [c.accuracy for c in cells if c.task == task_id and c.k == k
                    and c.accuracy is not None and c.n_rejected == 0
                    and (regime_protocol == "legacy_endpoint_ratio" or c.n_truncated == 0)]
            if vals:
                by_k[k] = st.mean(vals)
        if not by_k:
            continue
        peak_k = max(by_k, key=lambda k: by_k[k])
        entry = {"accuracy_by_k": by_k, "peak_k": peak_k, "peak": by_k[peak_k]}
        if 0 in by_k and max(by_k) in by_k and by_k[0] > 0:
            kmax = max(by_k)
            entry["a_max_over_a0"] = by_k[kmax] / by_k[0]
            if regime_protocol == "legacy_endpoint_ratio":
                entry["regime"] = ("cliff-prone" if by_k[kmax] / by_k[0] < 0.9
                                   else "cliff-immune")
        if regime_protocol == "prospective_decline_v1":
            crossings = []
            for k in sorted(by_k):
                earlier = [j for j in by_k if j < k]
                if k == 0 or not earlier:
                    continue
                reference = max(earlier, key=lambda j: by_k[j])
                decline = by_k[reference] - by_k[k]
                if by_k[reference] > 0 and decline >= 0.10 and decline / by_k[reference] >= 0.20:
                    crossings.append({"k_c": k, "k_ref": reference,
                                      "absolute_decline": decline,
                                      "relative_decline": decline / by_k[reference]})
            entry["development_candidate"] = crossings[0] if crossings else None
            entry["regime"] = "candidate_substantial_decline" if crossings else "no_substantial_decline_observed"
            entry["inference_status"] = "descriptive_development_only_requires_heldout_confirmation"
        curves[task_id] = entry
    return {"experiment": "behavioural_sweep", "backend": backend.name,
            "model_id": getattr(backend, "model_id", "?"),
            "tasks": tasks, "k_values": ks, "seeds": seeds, "n_test": n_test,
            "regime_protocol": regime_protocol,
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
    ap.add_argument("--template", choices=["native", "raw", "llama", "hermes"], default="native")
    ap.add_argument("--overflow-policy", choices=["reject", "legacy_left_truncate"], default="reject")
    ap.add_argument("--max-seq-len", type=int)
    ap.add_argument("--model-revision")
    ap.add_argument("--dataset-revision")
    ap.add_argument("--sampling-protocol", choices=["exact_k_v1", "legacy_floor"], default="exact_k_v1")
    ap.add_argument("--parser", choices=["strict", "legacy_prefix"], default="strict")
    ap.add_argument("--train-split")
    ap.add_argument("--test-split")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--query-split-seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--label-scores", action="store_true")
    ap.add_argument("--base-url"); ap.add_argument("--api-key")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else outputs_dir() / \
        f"sweep_{a.model.replace('/', '_')}.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite an existing sweep: {out}")

    backend_kwargs = {"base_url": a.base_url, "api_key": a.api_key}
    if a.backend == "hf_local":
        backend_kwargs.update(device=a.device, load_in_4bit=a.load_in_4bit,
                              template=a.template, overflow_policy=a.overflow_policy,
                              max_seq_len=a.max_seq_len, revision=a.model_revision)
    backend = get_backend(a.backend, a.model, **backend_kwargs)
    res = sweep(backend, a.tasks.split(","), [int(x) for x in a.ks.split(",")],
                [int(x) for x in a.seeds.split(",")], a.n_test,
                sampling_protocol=a.sampling_protocol, n_train=a.n_train,
                dataset_revision=a.dataset_revision, train_split=a.train_split,
                test_split=a.test_split, query_split_seed=a.query_split_seed,
                max_new_tokens=a.max_new_tokens, label_scores=a.label_scores, parser=a.parser)
    # Existing archives remain immutable; users must select a fresh output path.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(res, handle, indent=2, allow_nan=False)
    print(f"\nwrote {out}")
    for task, c in res["curves"].items():
        line = f"  {task}: peak {c['peak']:.3f} at K={c['peak_k']}"
        if "a_max_over_a0" in c:
            line += f"   a_max/a0 = {c['a_max_over_a0']:.3f}  -> {c['regime']}"
        print(line)
    return 0
