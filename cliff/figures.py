"""Tier 1: regenerate the paper's main-text figures from the archive.

No GPU and no network. Reads only the result files, and has NO fallback values:
a missing cell raises rather than being filled in. Cells that were never run are
drawn as explicit `n.r.` markers, and cells that degenerated to unparseable output
are drawn as decode-failure markers, never as accuracy magnitudes.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .paths import figures_dir, results_dir

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
})

RES = results_dir()
OUT = figures_dir()

C_PRONE, C_IMM, C_HERMES = "#d62728", "#1f77b4", "#2ca02c"
C_EARLY, C_MID, C_LATE = "#1f4e79", "#6baed6", "#bdbdbd"


# ---------------------------------------------------------------- data loaders
def hosted(fname):
    """Hosted-API behavioural sweep: {(task,k): mean_acc}. Single pipeline."""
    d = json.load(open(results_dir() / "nmi_hosted" / fname))
    agg = defaultdict(list)
    for c in d["cells"]:
        agg[(c["task"], c["k"])].append(c["accuracy"])
    return {k: st.mean(v) for k, v in agg.items()}


def curve(tbl, task, ks):
    out = []
    for k in ks:
        if (task, k) not in tbl:
            raise KeyError(f"missing hosted cell {task} K={k}")
        out.append(tbl[(task, k)])
    return out


def exp1a(model, task):
    rows = [r for r in csv.DictReader(open(results_dir() / "exp1a_complete.csv"))
            if r["model"] == model and r["task"] == task]
    if not rows:
        raise KeyError(f"missing exp1a rows for {model}/{task}")
    return {int(r["k"]): float(r["mean_acc"]) for r in rows}


def cells(fname, kind):
    d = json.load(open(results_dir() / "nmi_new_causal" / fname))
    out = defaultdict(dict)
    for c in d[kind + "_cells"]:
        out[(c["task"], c["k"])][c["seed"]] = c
    return out


def recovery(fname, task, k):
    """Paired per-seed recovery. Raises if the cell is absent.
    Returns (mean, per_seed_list, all_unparseable_flag)."""
    B, A = cells(fname, "before"), cells(fname, "after")
    if (task, k) not in B or (task, k) not in A:
        raise KeyError(f"{fname}: no cell {task} K={k}")
    seeds = sorted(set(B[(task, k)]) & set(A[(task, k)]))
    if not seeds:
        raise KeyError(f"{fname}: no paired seeds for {task} K={k}")
    per = [A[(task, k)][s]["accuracy"] - B[(task, k)][s]["accuracy"] for s in seeds]
    degen = all(A[(task, k)][s]["n_unparseable"] == A[(task, k)][s]["n_test"] for s in seeds)
    return st.mean(per), per, degen


def before_acc(fname, task, k):
    B = cells(fname, "before")
    return st.mean([c["accuracy"] for c in B[(task, k)].values()])


def after_acc(fname, task, k):
    A = cells(fname, "after")
    return st.mean([c["accuracy"] for c in A[(task, k)].values()])


# =========================================================== FIGURE 1
def figure1():
    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    panel = {
        "Llama-3.1-8B-Instruct": ("recipe_difference_llama31_8b_instruct.json", C_PRONE, "-", "o"),
        "Gemma-3-4B-it":         ("recipe_difference_gemma3_4b_it.json", C_IMM, "-", "s"),
        "Gemma-3-12B-it":        ("recipe_difference_gemma3_12b_it.json", "#6baed6", "--", "^"),
        "Gemma-3-27B-it":        ("recipe_difference_gemma3_27b_it.json", "#9467bd", "--", "v"),
        "Qwen3-32B":             ("recipe_difference_qwen3_32b.json", "#8c564b", ":", "P"),
    }
    tabs = {n: hosted(f) for n, (f, *_ ) in panel.items()}
    herm = hosted("hermes3_full.json")

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.30)

    # (a) ag_news regime split, hosted pipeline only
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ks))
    for name, (f, col, ls, mk) in panel.items():
        ax.plot(x, curve(tabs[name], "ag_news", ks), ls, color=col, marker=mk,
                ms=4, lw=1.8, label=name)
    ax.axhline(0.25, ls=":", lw=1, color="#888")
    ax.text(7.4, 0.265, "chance = 0.25", fontsize=8, color="#666", ha="right")
    ax.axvline(0.5, ls=":", lw=0.9, color="#bbb")
    ax.text(0.02, 0.955, "$K{=}0$", fontsize=8, color="#666", transform=ax.transAxes)
    ax.annotate("cliff-prone\ncollapse", xy=(6.1, 0.34), fontsize=9, color=C_PRONE,
                ha="center", fontweight="bold")
    ax.annotate("cliff-immune", xy=(5.2, 0.895), fontsize=9, color=C_IMM,
                ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("$K$ (demonstrations)"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.20, 1.0)
    ax.set_title("a   AG News, hosted panel", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=7.2,
              bbox_to_anchor=(-0.02, 0.06))

    # (b) within-base-model recipe dissociation -- THE headline result
    ax = fig.add_subplot(gs[0, 1])
    L = tabs["Llama-3.1-8B-Instruct"]
    for task, mk, ls in [("ag_news", "o", "-"), ("sst2", "s", "--"), ("mnli", "^", ":")]:
        ax.plot(x, curve(L, task, ks), ls, color=C_PRONE, marker=mk, ms=4, lw=1.8)
        ax.plot(x, curve(herm, task, ks), ls, color=C_HERMES, marker=mk, ms=4, lw=1.8)
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("$K$ (demonstrations)"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.20, 1.0)
    ax.set_title("b   Same base checkpoint, two recipes", loc="left", fontweight="bold")
    hs = [plt.Line2D([], [], color=C_PRONE, lw=2, label="Meta recipe (Instruct)"),
          plt.Line2D([], [], color=C_HERMES, lw=2, label="NousResearch recipe (Hermes-3)"),
          plt.Line2D([], [], color="#555", lw=1.3, ls="-",  marker="o", ms=4, label="ag_news"),
          plt.Line2D([], [], color="#555", lw=1.3, ls="--", marker="s", ms=4, label="sst2"),
          plt.Line2D([], [], color="#555", lw=1.3, ls=":",  marker="^", ms=4, label="mnli")]
    ax.legend(handles=hs, loc="lower left", frameon=False, fontsize=7.5)

    # (c) the cliff at other scales / other pipelines, labelled as such
    ax = fig.add_subplot(gs[0, 2])
    ds = exp1a("deepseek_v3", "ag_news")
    dk = sorted(ds)
    ax.plot([ks.index(k) for k in dk], [ds[k] for k in dk], "-", color="#ff7f0e",
            marker="D", ms=4, lw=1.8, label="DeepSeek-V3.2, 671B (hosted, $K{\\geq}1$)")
    f3 = "remove_llama32_3b_early_t6_results.json"
    k3 = [0, 1, 4, 16]
    ax.plot([ks.index(k) for k in k3], [before_acc(f3, "ag_news", k) for k in k3],
            "-", color="#e377c2", marker="*", ms=7, lw=1.8,
            label="Llama-3.2-3B (self-hosted)")
    g4 = curve(tabs["Gemma-3-4B-it"], "ag_news", ks)
    ax.plot(x, g4, "-", color=C_IMM, marker="s", ms=4, lw=1.4, alpha=0.55,
            label="Gemma-3-4B, 4B (hosted)")
    ax.axhline(0.25, ls=":", lw=1, color="#888")
    ax.text(7.4, 0.265, "chance = 0.25", fontsize=8, color="#666", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("$K$ (demonstrations)"); ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.20, 1.0)
    ax.set_title("c   671B falls, 4B does not", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=7.2,
              bbox_to_anchor=(-0.02, 0.06))
    ax.text(0.99, 0.965, "pipelines not pooled", transform=ax.transAxes, ha="right",
            fontsize=7, color="#888", style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(figures_dir() / f"fig1_regime_taxonomy.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig1_regime_taxonomy")


# =========================================================== FIGURE 4
DEPTH_AG = {0: "remove_llama31_8b_blk0_n4_attn_results.json",
            8: "remove_llama31_8b_blk8_n4_attn_results.json",
            12: "remove_llama31_8b_blk12_n4_attn_results.json",
            20: "remove_llama31_8b_blk20_n4_attn_results.json",
            24: "remove_llama31_8b_blk24_n4_attn_results.json"}
NOT_RUN = [4, 16, 28]


def figure4():
    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)

    def nr(ax, x, y=0.0):
        """Mark a never-run cell at the baseline; never as a bar height."""
        ax.plot([x], [y], marker="_", ms=9, color="#bbb", mew=1.6)
        ax.text(x, y - 0.035, "n.r.", ha="center", va="top", fontsize=6.5, color="#999")

    # ---- (a) depth profile: measured windows only
    ax = fig.add_subplot(gs[0, 0])
    starts = [0, 4, 8, 12, 16, 20, 24, 28]
    for i, s in enumerate(starts):
        if s in NOT_RUN:
            nr(ax, i)
            continue
        m, _, _ = recovery(DEPTH_AG[s], "ag_news", 16)
        ax.bar(i, m, 0.68, color=C_EARLY if s in (8, 12) else "#7fa8cc")
        ax.text(i, m + 0.018, f"{m:+.2f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([f"{s}-{s+3}" for s in starts], rotation=45, fontsize=8)
    ax.set_xlabel("4-block window (blocks)"); ax.set_ylabel("cliff recovery")
    ax.set_ylim(-0.09, 0.74)
    ax.set_title("a   depth locus, ag_news $K{=}16$", loc="left", fontweight="bold")

    # ---- (b) region x task
    ax = fig.add_subplot(gs[0, 1])
    E = "remove_llama31_8b_early_mt_results.json"
    L = "remove_llama31_8b_late_mt_results.json"
    M = "remove_llama31_8b_mid_v2_results.json"
    tasks = [("ag_news", 16), ("sst2", 64), ("trec", 64)]
    w = 0.25
    for j, (task, k) in enumerate(tasks):
        me, pe, _ = recovery(E, task, k)
        ax.bar(j - w, me, w, color=C_EARLY)
        ax.errorbar(j - w, me, yerr=[[me - min(pe)], [max(pe) - me]], fmt="none",
                    ecolor="#222", lw=0.9, capsize=2)
        try:
            mm, _, _ = recovery(M, task, k)
            ax.bar(j, mm, w, color=C_MID)
        except KeyError:
            nr(ax, j)
        ml, pl, degen = recovery(L, task, k)
        if degen:
            ax.plot([j + w], [0.0], marker="x", ms=8, color="#c00", mew=2)
            ax.annotate("no parseable\noutput", xy=(j + w, 0.0), xytext=(j + w + 0.05, 0.30),
                        fontsize=7, color="#c00", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#c00", lw=0.9))
        else:
            ax.bar(j + w, ml, w, color=C_LATE)
            ax.errorbar(j + w, ml, yerr=[[ml - min(pl)], [max(pl) - ml]], fmt="none",
                        ecolor="#222", lw=0.9, capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{t}\n$K{{=}}{k}$" for t, k in tasks], fontsize=8.5)
    ax.set_ylabel("cliff recovery"); ax.set_ylim(-0.30, 0.78)
    ax.set_title("b   remove arm by region (5 seeds, range bars)", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=C_EARLY, label="early 0-7"),
                       Patch(color=C_MID, label="mid 12-19"),
                       Patch(color=C_LATE, label="late 24-31")],
              loc="upper right", frameon=False, fontsize=7.5, ncol=3,
              columnspacing=1.0, handlelength=1.2)

    # ---- (c) cross-model / cross-family
    ax = fig.add_subplot(gs[1, 0])
    fam = [
        ("Llama-3.1-8B", 16, E, M, L),
        ("Llama-3.2-3B", 16, "remove_llama32_3b_early_t6_results.json", None,
         "remove_llama32_3b_late_t6_results.json"),
        ("DeepSeek-7B", 16, "remove_deepseek_7b_remove_early_v2_results.json",
         "remove_deepseek_7b_remove_mid_v2_results.json",
         "remove_deepseek_7b_remove_late_v2_results.json"),
        ("Mistral-7B", 32, "remove_mistral7b_remove_early_k32_results.json",
         "remove_mistral7b_remove_mid_k32_results.json",
         "remove_mistral7b_remove_late_k32_results.json"),
        ("InternLM2.5", 64, "remove_internlm2_7b_remove_early_v8_results.json",
         "remove_internlm2_7b_remove_mid_v4_results.json",
         "remove_internlm2_7b_remove_late_v4_results.json"),
    ]
    for j, (name, k, fe, fm, fl) in enumerate(fam):
        for off, f, col in ((-w, fe, C_EARLY), (0.0, fm, C_MID), (w, fl, C_LATE)):
            if f is None:
                nr(ax, j + off)
                continue
            m, _, _ = recovery(f, "ag_news", k)
            ax.bar(j + off, m, w, color=col)
            ax.text(j + off, m + 0.022 if m >= 0 else m - 0.022, f"{m:+.2f}",
                    ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=6.5, rotation=90)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(fam)))
    ax.set_xticklabels([f"{n}\n$K{{=}}{k}$" for n, k, *_ in fam], fontsize=7.5)
    ax.set_ylabel("cliff recovery"); ax.set_ylim(-0.14, 0.76)
    ax.set_title("c   cross-model and cross-family, ag_news", loc="left", fontweight="bold")
    ax.text(0.5, -0.30, "Mistral-7B is the one family where late approaches the best band",
            transform=ax.transAxes, ha="center", fontsize=7, color="#888", style="italic")

    # ---- (d) induce arm, full sweep, correct K=0 values
    ax = fig.add_subplot(gs[1, 1])
    fe = "induce_gemma3_4b_induce_early_final_results.json"
    fl = "induce_gemma3_4b_induce_late_final_results.json"
    kk = [0, 1, 4, 8, 16, 32]
    xb = np.arange(len(kk))
    ax.plot(xb, [before_acc(fe, "ag_news", k) for k in kk], "-", color="#555",
            marker="o", ms=4.5, lw=1.8, label="before intervention")
    ax.plot(xb, [after_acc(fe, "ag_news", k) for k in kk], "-", color=C_PRONE,
            marker="s", ms=4.5, lw=1.8, label="after, early locus (0-7)")
    ax.plot(xb, [after_acc(fl, "ag_news", k) for k in kk], "--", color="#888",
            marker="^", ms=4.5, lw=1.8, label="after, late locus (24-31)")
    b0, a0 = before_acc(fe, "ag_news", 0), after_acc(fe, "ag_news", 0)
    ax.annotate(f"$K{{=}}0$: {b0:.2f} $\\rightarrow$ {a0:.2f}", xy=(0.05, a0),
                xytext=(1.15, 0.50), fontsize=8.5, color=C_PRONE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PRONE, lw=1.2))
    d16 = after_acc(fe, "ag_news", 16) - before_acc(fe, "ag_news", 16)
    ax.annotate(f"$K{{=}}16$: {d16:+.3f}", xy=(4, after_acc(fe, "ag_news", 16)),
                xytext=(3.0, 0.60), fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
    ax.set_xticks(xb); ax.set_xticklabels(kk)
    ax.set_xlabel("$K$ (demonstrations)"); ax.set_ylabel("accuracy (ag_news)")
    ax.set_ylim(0.38, 0.94)
    ax.set_title("d   reverse-direction arm, Gemma-3-4B", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    for ext in ("pdf", "png"):
        fig.savefig(figures_dir() / f"fig_causal_lesion_rescue.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_causal_lesion_rescue")


# =========================================================== FIGURE 3
def figure3():
    m = json.load(open(RES / "nmi_new_causal/t5_llama31_8b_ag_news_results.json"))
    fv, ih = m["fv_effect_by_layer"], m["induction_score_by_layer"]
    zs, icl = m["zero_shot_acc"], m["icl_acc"]
    band = range(8, 16)
    inb = st.mean([ih[i] for i in band])
    outb = st.mean([ih[i] for i in range(32) if i not in band])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                 gridspec_kw={"hspace": 0.13})
    a1.axvspan(7.5, 15.5, color="#dbe9f6", zorder=0)
    a1.plot(range(32), fv, "-o", color="#1f77b4", ms=3.5, lw=1.6)
    a1.axhline(zs, ls="--", lw=1, color="#777")
    a1.axhline(icl, ls="--", lw=1, color="#c00")
    a1.text(31.5, zs + 0.015, f"zero-shot ({zs:.2f})", fontsize=8, color="#777", ha="right")
    a1.text(31.5, icl + 0.015, f"cliff-level ICL ({icl:.2f})", fontsize=8, color="#c00", ha="right")
    a1.annotate(f"onset, layer 11\n({fv[11]:.2f})", xy=(11, fv[11]), xytext=(4.0, 0.52),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=1))
    a1.annotate(f"layer 14\n({fv[14]:.2f})", xy=(14, fv[14]), xytext=(17.5, 0.62),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=1))
    a1.text(11.5, 0.93, "blocks 8--15\n(causal window)", fontsize=8, color="#4a7ebb",
            style="italic", ha="center")
    a1.set_ylabel("FV causal effect\n(accuracy)")
    a1.set_ylim(0.15, 1.0)
    a1.set_title("Two correlational measures, unequal corroboration",
                 loc="left", fontweight="bold", fontsize=10.5)

    a2.axvspan(7.5, 15.5, color="#dbe9f6", zorder=0)
    a2.plot(range(32), ih, "-o", color="#ff7f0e", ms=3.5, lw=1.6)
    a2.axhline(inb, ls="-", lw=1.2, color="#4a7ebb")
    a2.axhline(outb, ls="--", lw=1.2, color="#999")
    a2.text(31.6, inb + 0.0035, f"mean in band = {inb:.3f}", fontsize=8,
            color="#4a7ebb", ha="right")
    a2.text(31.6, outb - 0.008, f"mean outside = {outb:.3f}", fontsize=8,
            color="#777", ha="right")
    for lay in (2, 15, 20):
        a2.annotate(f"L{lay}\n{ih[lay]:.3f}", xy=(lay, ih[lay]),
                    xytext=(lay - 0.4, ih[lay] + 0.014), fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7))
    a2.text(16.5, 0.092, "no localization: in-band mean $\\approx$ out-of-band mean",
            fontsize=8, color="#a33", style="italic", ha="center")
    a2.set_ylabel("Induction-head\nscore")
    a2.set_xlabel("Layer"); a2.set_ylim(0, 0.10); a2.set_xlim(-0.8, 31.8)
    a2.set_xticks(range(0, 32, 4))

    for ext in ("pdf", "png"):
        fig.savefig(figures_dir() / f"fig_fv_induction.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote fig_fv_induction")




def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="cliff figures")
    ap.add_argument("--only", choices=("1", "3", "4"), action="append")
    a = ap.parse_args(argv)
    want = a.only or ["1", "3", "4"]
    if "1" in want: figure1()
    if "4" in want: figure4()
    if "3" in want: figure3()
    print(f"\nfigures written to {figures_dir()}")
    print("Regenerated from the archive with no fallback values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
