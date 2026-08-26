"""Audit the induce-arm supervision target (Supplementary Note 5, Table S2).

The reverse-direction arm trains toward the label the model itself emits with no
demonstrations. That prediction falls back to the first label in the label set when
the generation does not parse. If the fallback fires often, the calibration set is
close to a constant label and the observed zero-shot decrement has an explanation
that does not implicate the targeted band.

This needs no fine-tuning: the calibration set is built from the base model with a
seeded draw, so it is exactly reproducible. Published result: 0 fallbacks in 500,
with targets agreeing with the gold label on 81.6% of examples.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..backends import get_backend
from ..data import TASKS
from ..experiments.intervention import build_calibration
from ..paths import outputs_dir


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cliff audit")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--intervention-k", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    labels = TASKS[a.task]["label_names"]
    backend = get_backend("hf_local", a.model, device=a.device)
    print(f"[audit] {a.model} on {backend.device}; {a.n} calibration examples")
    rows, stats = build_calibration(backend, a.task, "induce", a.n, a.intervention_k)

    tgt = Counter(r["target"] for r in rows)
    top_share = tgt.most_common(1)[0][1] / len(rows)
    res = {"experiment": "induce_arm_supervision_target_audit",
           "model_id": a.model, "task": a.task, "n_calibration": len(rows),
           "fallback_label": labels[0], **stats,
           "most_common_target_share": top_share}
    out = Path(a.out) if a.out else outputs_dir() / "induce_fallback_audit.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"\n{'=' * 62}")
    print(f"  fallback fired            {stats['n_fallback']}/{len(rows)}"
          f"  ({stats['fallback_rate']:.1%})")
    print(f"  most common target share  {top_share:.1%}  ({tgt.most_common(1)[0][0]})")
    print(f"  target distribution       {dict(tgt)}")
    print(f"{'=' * 62}")
    if stats["fallback_rate"] > 0.30:
        print("  HIGH: the calibration set is largely a constant label. The zero-shot")
        print("  decrement has a mundane explanation; withdraw or rerun the arm.")
    elif top_share > 0.60:
        print("  Targets are skewed toward one class even though the fallback is rare;")
        print("  report the skew alongside the result.")
    else:
        print("  LOW: the decrement is NOT explained by the fallback. This reproduces")
        print("  the published control (0/500).")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
