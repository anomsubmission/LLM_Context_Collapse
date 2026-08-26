"""Region-restricted LoRA consolidation: the paper's central causal experiment.

Two arms:
  remove  -- on a cliff-prone model, train toward the CORRECT label and measure
             cliff recovery. Early/mid relieves the cliff; the late readout does not.
  induce  -- on a cliff-immune model, train toward the model's OWN zero-shot label
             and measure the effect on context-free generalization.

Three properties of the published implementation are preserved deliberately, because
changing them would change the numbers, and are surfaced rather than hidden:

  1. Region targeting is ASSERTED here, not merely logged. The published script
     silently fell back to adapting every layer if no module name matched the
     requested band; `resolve_targets` raises instead. `--allow-fallback` restores
     the old behaviour if you need bit-comparability with the archive.
  2. The objective is next-token prediction over the WHOLE prompt, not the label
     span. `--label-only-loss` offers the arguably more correct alternative; it is
     off by default so the default reproduces the paper.
  3. The induce arm's target falls back to the first label when the model's
     zero-shot generation does not parse. The fallback count is recorded in the
     output (it was 0/500 in the published run).
"""
from __future__ import annotations

import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from ..backends import require_weight_access
from ..data import TASKS, class_balanced_sample, load_task
from ..paths import outputs_dir
from ..prompting import Cell, parse_label, render_prompt

REGIONS = {"early": "first n blocks", "mid": "centred n blocks", "late": "last n blocks"}


def region_window(n_layers: int, n_blocks: int, region: str,
                  block_start: int | None = None) -> range:
    n = max(1, min(n_blocks, n_layers))
    if block_start is not None:
        start = block_start
    elif region == "early":
        start = 0
    elif region == "mid":
        start = (n_layers - n) // 2
    elif region == "late":
        start = n_layers - n
    else:
        raise SystemExit(f"region must be one of {sorted(REGIONS)}")
    if not (0 <= start and start + n <= n_layers):
        raise SystemExit(f"window {start}..{start+n-1} outside a {n_layers}-block stack")
    return range(start, start + n)


def resolve_targets(model, window: range, component: str = "attn",
                    allow_fallback: bool = False) -> list[str]:
    """Full module names inside `window`. Raises on an empty match unless
    allow_fallback, which reproduces the published silent-fallback behaviour."""
    leaves = {n.rsplit(".", 1)[-1] for n, m in model.named_modules()
              if hasattr(m, "weight") and m.weight is not None
              and not list(m.children())}
    attn = {n for n in leaves if any(x in n.lower() for x in (
        "q_proj", "k_proj", "v_proj", "o_proj", "wqkv", "wq", "wk", "wv", "wo",
        "query", "key", "value", "out_proj", "c_attn"))}
    mlp = {n for n in leaves if any(x in n.lower() for x in (
        "gate", "up_proj", "down_proj", "w1", "w2", "w3", "ffn", "fc"))}
    wanted = mlp if component == "mlp" else (attn | mlp if component == "both" else attn)

    targets = []
    for name, _ in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) in window and name.rsplit(".", 1)[-1] in wanted:
            targets.append(name)
    if not targets:
        msg = (f"no {component} modules matched blocks {window.start}-{window.stop-1}. "
               f"Leaf names seen: {sorted(leaves)[:12]}")
        if not allow_fallback:
            raise RuntimeError(msg + "\nRefusing to fall back to all layers, which "
                               "would silently destroy the localization. "
                               "Pass --allow-fallback to reproduce the published behaviour.")
        print(f"[WARN] {msg}\n[WARN] falling back to ALL layers (localization void)")
        return sorted(wanted)
    return targets


def assert_targeting(peft_model, window: range) -> dict:
    """Post-hoc check that instantiated adapters lie only in `window`."""
    seen = set()
    for name, _ in peft_model.named_modules():
        if "lora_A" in name:
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                seen.add(int(m.group(1)))
    stray = sorted(seen - set(window))
    if stray:
        raise RuntimeError(f"adapters instantiated outside the requested band: {stray}")
    return {"adapted_blocks": sorted(seen), "requested": [window.start, window.stop - 1]}


def build_calibration(backend, task_id: str, arm: str, n: int, k: int,
                      seed: int = 42) -> tuple[list[dict], dict]:
    import random
    data = load_task(task_id)
    labels = TASKS[task_id]["label_names"]
    rng = random.Random(seed)
    rows, n_fallback = [], 0
    for _ in range(n):
        s = rng.randint(0, 1_000_000)
        demos = class_balanced_sample(data.train, k, s)
        q = rng.choice(data.train)
        if arm == "remove":
            target = q.label
        else:
            gen = backend.generate(render_prompt(task_id, [], q.text))
            parsed = parse_label(gen, labels)
            n_fallback += parsed is None
            target = parsed if parsed is not None else labels[0]
        rows.append({"prompt": render_prompt(task_id, demos, q.text), "target": target})
    stats = {"n": n, "n_fallback": n_fallback, "fallback_rate": n_fallback / max(1, n),
             "target_distribution": dict(Counter(r["target"] for r in rows))}
    return rows, stats


def consolidate(backend, rows, window, component, lora_r, lora_alpha, lr, epochs,
                allow_fallback, label_only_loss):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset

    model, tok = backend.torch_model(), backend.tokenizer()
    targets = resolve_targets(model, window, component, allow_fallback)
    print(f"[consolidate] {len(targets)} modules in blocks "
          f"{window.start}-{window.stop - 1} ({component})")
    peft_model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
        target_modules=targets, lora_dropout=0.05))
    info = assert_targeting(peft_model, window)
    print(f"[consolidate] verified: adapters only in blocks {info['adapted_blocks']}")

    texts, prompt_lens = [], []
    for r in rows:
        wrapped = backend._wrap(r["prompt"])
        texts.append(wrapped + f" {r['target']}{tok.eos_token}")
        prompt_lens.append(len(tok(wrapped, truncation=True,
                                   max_length=backend.max_seq_len)["input_ids"]))
    enc = tok(texts, truncation=True, max_length=backend.max_seq_len, padding="longest")
    labels_ids = [list(x) for x in enc["input_ids"]]
    for i, seq in enumerate(labels_ids):
        for j, tid in enumerate(seq):
            if tid == tok.pad_token_id:
                seq[j] = -100
        if label_only_loss:
            for j in range(min(prompt_lens[i], len(seq))):
                seq[j] = -100
    ds = Dataset.from_dict({"input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                            "labels": labels_ids})
    args = TrainingArguments(
        output_dir=str(outputs_dir() / "_trainer"), num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, save_strategy="no", logging_steps=25,
        report_to="none", max_grad_norm=1.0,
        bf16=(backend.device == "cuda"), use_cpu=(backend.device == "cpu"))
    Trainer(model=peft_model, args=args, train_dataset=ds).train()
    peft_model.eval()
    backend._model = peft_model
    return info


def evaluate(backend, tasks, ks, seeds, n_test) -> list[Cell]:
    cells = []
    for task_id in tasks:
        data = load_task(task_id, n_test=n_test)
        for k in ks:
            for seed in seeds:
                demos = class_balanced_sample(data.train, k, seed)
                correct = unpar = 0
                for ex in data.test:
                    pred = parse_label(backend.generate(
                        render_prompt(task_id, demos, ex.text)), data.label_names)
                    if pred is None:
                        unpar += 1
                    elif pred == ex.label:
                        correct += 1
                n = len(data.test)
                c = Cell(task_id, k, seed, correct / max(1, n), n, unpar)
                cells.append(c)
                print(f"  {task_id:9s} K={k:<3d} s{seed} acc={c.accuracy:.3f}"
                      f" unpar={unpar}/{n}{'  <-- DECODE FAILURE' if c.is_decode_failure else ''}")
    return cells


def main(argv=None) -> int:
    import argparse
    from ..backends import get_backend
    ap = argparse.ArgumentParser(prog="cliff intervene")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="hf_local",
                    help="must provide weight access; hf_local, or your own")
    ap.add_argument("--arm", choices=("remove", "induce"), default="remove")
    ap.add_argument("--region", choices=sorted(REGIONS), default="early")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--block-start", type=int, default=None,
                    help="override the region and place the window explicitly")
    ap.add_argument("--component", choices=("attn", "mlp", "both"), default="attn")
    ap.add_argument("--tasks", default="ag_news")
    ap.add_argument("--cal-task", default="ag_news")
    ap.add_argument("--ks", default="0,1,4,16")
    ap.add_argument("--cliff-k", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-calibration", type=int, default=500)
    ap.add_argument("--intervention-k", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--label-only-loss", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    tasks = a.tasks.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    backend = require_weight_access(
        get_backend(a.backend, a.model, device=a.device,
                    load_in_4bit=a.load_in_4bit or None), "the intervene command")
    n_layers = backend.n_layers()
    window = region_window(n_layers, a.n_blocks, a.region, a.block_start)
    print(f"[intervene] {a.model}: {n_layers} blocks; targeting "
          f"{window.start}-{window.stop-1} ({a.region}, {a.component}); arm={a.arm}")

    print("\n[phase 1] baseline sweep")
    before = evaluate(backend, tasks, ks, seeds, a.n_test)

    print("\n[phase 2] calibration set + region-restricted consolidation")
    rows, cal_stats = build_calibration(backend, a.cal_task, a.arm,
                                       a.n_calibration, a.intervention_k)
    print(f"[phase 2] calibration: fallback fired {cal_stats['n_fallback']}"
          f"/{cal_stats['n']}; targets {cal_stats['target_distribution']}")
    target_info = consolidate(backend, rows, window, a.component, a.lora_r,
                              a.lora_alpha, a.lr, a.epochs, a.allow_fallback,
                              a.label_only_loss)

    print("\n[phase 3] post-intervention sweep")
    after = evaluate(backend, tasks, ks, seeds, a.n_test)

    def paired(task, k):
        b = {c.seed: c for c in before if c.task == task and c.k == k}
        aa = {c.seed: c for c in after if c.task == task and c.k == k}
        s = sorted(set(b) & set(aa))
        if not s:
            return None
        deltas = [aa[i].accuracy - b[i].accuracy for i in s]
        return {"before": st.mean(b[i].accuracy for i in s),
                "after": st.mean(aa[i].accuracy for i in s),
                "recovery": st.mean(deltas), "per_seed": deltas,
                "after_is_decode_failure": all(aa[i].is_decode_failure for i in s)}

    summary = {t: {str(k): paired(t, k) for k in ks if paired(t, k)} for t in tasks}
    res = {"experiment": "region_restricted_intervention", "arm": a.arm,
           "model_id": a.model, "region": a.region, "window": [window.start, window.stop - 1],
           "component": a.component, "n_layers": n_layers,
           "objective": "label-span" if a.label_only_loss else "whole-prompt",
           "targeting": target_info, "calibration": cal_stats,
           "cliff_k": a.cliff_k, "hyperparameters": {
               "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "lr": a.lr,
               "epochs": a.epochs, "n_calibration": a.n_calibration,
               "intervention_k": a.intervention_k},
           "before_cells": [c.to_dict() for c in before],
           "after_cells": [c.to_dict() for c in after],
           "summary": summary}
    out = Path(a.out) if a.out else outputs_dir() / \
        f"{a.arm}_{a.model.replace('/', '_')}_{a.region}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\n{'=' * 60}\n  arm={a.arm} region={a.region} "
          f"blocks {window.start}-{window.stop-1}")
    for t, ent in summary.items():
        cell = ent.get(str(a.cliff_k))
        if cell:
            note = "  (DECODE FAILURE, not an accuracy)" if cell["after_is_decode_failure"] else ""
            print(f"  {t} K={a.cliff_k}: {cell['before']:.3f} -> {cell['after']:.3f}"
                  f"   recovery {cell['recovery']:+.3f}{note}")
    print(f"  wrote {out}")
    return 0
