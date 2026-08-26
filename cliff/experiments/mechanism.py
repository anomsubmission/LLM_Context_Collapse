"""Layer-resolved mechanistic measures: function vector and induction-head score.

Reproduces Fig. 3. Requires weight access (hidden states), so it needs hf_local.

Note on interpretation, which the paper is explicit about: the function-vector
profile is consistent with the causal locus, but the induction-head profile is NOT
and is reported as a negative result. This script computes and prints the in-band
versus out-of-band means so that conclusion is checkable rather than asserted.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from ..backends import require_weight_access
from ..data import class_balanced_sample, load_task
from ..paths import outputs_dir
from ..prompting import parse_label, render_prompt


def induction_scores(backend, n_layers: int, seq_len: int = 256,
                     seed: int = 0) -> list[float]:
    """Attention mass on the token following a previous occurrence of the current
    token, averaged over heads, aggregated by layer, on a random repeated sequence."""
    import torch
    model, tok = backend.torch_model(), backend.tokenizer()
    g = torch.Generator().manual_seed(seed)
    vocab = getattr(model.config, "vocab_size", None) or tok.vocab_size
    half = torch.randint(5, min(vocab, 20000), (seq_len // 2,), generator=g)
    ids = torch.cat([half, half]).unsqueeze(0).to(model.device)

    # Attention weights are only materialized by the eager kernel; SDPA and
    # FlashAttention return None for `attentions`. Switch for this forward pass
    # only, then restore, so the rest of the run keeps the fast kernel.
    cfgs = [model.config] + ([model.config.text_config]
                             if hasattr(model.config, "text_config") else [])
    saved = [getattr(c, "_attn_implementation", None) for c in cfgs]
    for c in cfgs:
        c._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            out = model(ids, output_attentions=True)
    finally:
        for c, s in zip(cfgs, saved):
            if s is not None:
                c._attn_implementation = s
    if not getattr(out, "attentions", None):
        raise RuntimeError(
            "the model returned no attention weights, so the induction-head score "
            "cannot be computed. Load with attn_implementation='eager', or pass "
            "--skip-induction to compute only the function-vector profile.")
    scores = []
    for att in out.attentions:                      # (1, heads, T, T)
        a = att[0].float()
        T = a.shape[-1]
        n = T // 2
        # for query position n+i, the induction target is position i+1
        vals = [a[:, n + i, i + 1].mean().item() for i in range(n - 1)]
        scores.append(float(st.mean(vals)) if vals else 0.0)
    return scores[:n_layers]


def function_vector(backend, task_id: str, k: int, n_prompts: int,
                    n_test: int, seed: int = 0) -> tuple[list[float], float, float]:
    """Average each layer's residual state at the final position over in-context
    prompts, patch it into a zero-shot pass at the same layer, and score accuracy."""
    import torch
    model, tok = backend.torch_model(), backend.tokenizer()
    data = load_task(task_id, n_test=max(n_test, n_prompts))
    labels = data.label_names
    n_layers = backend.n_layers()

    def hidden(prompt):
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=backend.max_seq_len).to(model.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        return [h[0, -1].detach().clone() for h in out.hidden_states]

    demos = class_balanced_sample(data.train, k, seed)
    acc = [hidden(render_prompt(task_id, demos, ex.text))
           for ex in data.test[:n_prompts]]
    fv = [torch.stack([a[i] for a in acc]).mean(0) for i in range(len(acc[0]))]

    blocks = _blocks(model)

    def score(patch_layer: int | None) -> float:
        correct = 0
        handle = None
        if patch_layer is not None:
            vec = fv[patch_layer + 1]

            def hook(_m, _i, output):
                if isinstance(output, tuple):
                    h = output[0]
                    h[:, -1, :] = vec.to(h.dtype)
                    return (h,) + output[1:]
                output[:, -1, :] = vec.to(output.dtype)
                return output
            handle = blocks[patch_layer].register_forward_hook(hook)
        try:
            for ex in data.test[:n_test]:
                gen = backend.generate(render_prompt(task_id, [], ex.text))
                if parse_label(gen, labels) == ex.label:
                    correct += 1
        finally:
            if handle is not None:
                handle.remove()
        return correct / max(1, n_test)

    zero_shot = score(None)
    profile = [score(i) for i in range(n_layers)]
    icl_correct = sum(
        parse_label(backend.generate(render_prompt(task_id, demos, ex.text)),
                    labels) == ex.label for ex in data.test[:n_test])
    return profile, zero_shot, icl_correct / max(1, n_test)


def _blocks(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers",
                 "language_model.model.layers", "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot locate the transformer block list on this model")


def main(argv=None) -> int:
    import argparse
    from ..backends import get_backend
    ap = argparse.ArgumentParser(prog="cliff mechanism")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="hf_local",
                    help="must provide weight access; hf_local, or your own")
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--band", default="8,16", help="causal window, lo,hi (hi exclusive)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-fv", action="store_true",
                    help="compute only the induction-head profile (fast)")
    ap.add_argument("--skip-induction", action="store_true",
                    help="compute only the function-vector profile")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    backend = require_weight_access(
        get_backend(a.backend, a.model, device=a.device), "the mechanism command")
    n_layers = backend.n_layers()
    print(f"[mechanism] {a.model}: {n_layers} layers")

    lo, hi = (int(x) for x in a.band.split(","))
    if not (0 <= lo < hi <= n_layers):
        raise SystemExit(f"--band {lo},{hi} is not a valid window for a "
                         f"{n_layers}-block stack (need 0 <= lo < hi <= {n_layers}). "
                         f"The paper's band, blocks 8-15, applies to 32-block models.")
    ih = []
    if not a.skip_induction:
        print("[mechanism] induction-head scores")
        ih = induction_scores(backend, n_layers)
    if ih and len(ih) < hi:
        raise SystemExit(f"got {len(ih)} layer scores but --band needs {hi}")
    inside = st.mean(ih[lo:hi]) if ih else float("nan")
    outside = st.mean([v for i, v in enumerate(ih) if not lo <= i < hi]) if ih else float("nan")

    res = {"experiment": "mechanism", "model_id": a.model, "task": a.task,
           "k_icl": a.k, "n_layers": n_layers, "band": [lo, hi],
           "induction_score_by_layer": ih,
           "induction_mean_in_band": inside, "induction_mean_out_of_band": outside,
           # Localization requires the in-band mean to EXCEED the out-of-band mean by a
# margin. A direction-blind |difference| test would call a band with *less*
# signal "localized", which is the opposite of the claim being tested.
           "induction_localizes": (inside - outside > 0.005) if ih else None,
           "induction_band_excess": (inside - outside) if ih else None}
    if not a.skip_fv:
        print("[mechanism] function-vector causal effect (this is the slow part)")
        fv, zs, icl = function_vector(backend, a.task, a.k, a.n_prompts, a.n_test)
        res |= {"fv_effect_by_layer": fv, "zero_shot_acc": zs, "icl_acc": icl}

    out = Path(a.out) if a.out else outputs_dir() / \
        f"mechanism_{a.model.replace('/', '_')}.json"
    out.write_text(json.dumps(res, indent=2))
    if not ih:
        print(f"wrote {out}")
        return 0
    print(f"\n  induction mean in band [{lo},{hi}) = {inside:.4f}")
    print(f"  induction mean outside       = {outside:.4f}")
    print(f"  excess (in - out)            = {inside - outside:+.4f}")
    if res["induction_localizes"]:
        print("  -> the induction-head score DOES localize to this band")
    else:
        why = ("no excess signal in the band"
               if inside <= outside else "excess below the 0.005 margin")
        print(f"  -> the induction-head score does NOT localize to this band ({why})")
        print("     For Llama-3.1-8B on blocks 8-15 the paper reports exactly this")
        print("     negative result: 0.0327 in band vs 0.0313 outside.")
    print(f"wrote {out}")
    return 0
