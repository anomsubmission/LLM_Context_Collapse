"""Every quantitative claim in the paper, as executable assertions.

This is the reproduction contract. Each `Claim` names a value that appears in the
manuscript, where it appears, and a zero-argument function that recomputes it from
the archived data. `cliff verify` runs all of them and prints a PASS/FAIL table.

If you change an analysis, a claim fails and tells you which sentence of the paper
to update. If you re-run an experiment and overwrite the archive (CLIFF_RESULTS),
the claims tell you which reported numbers moved.

Claims are also the audit trail for claims the paper *withdrew*: see the
NEGATIVE_CONTROLS block, which asserts the properties of the data that made those
claims untenable, so a reader can confirm the withdrawals were warranted.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from typing import Callable

from . import loaders as L

LLAMA31 = "recipe_difference_llama31_8b_instruct.json"
HERMES = "hermes3_full.json"
GEMMA4B = "recipe_difference_gemma3_4b_it.json"
GEMMA12B = "recipe_difference_gemma3_12b_it.json"
GEMMA27B = "recipe_difference_gemma3_27b_it.json"
QWEN32B = "recipe_difference_qwen3_32b.json"

EARLY_MT = "remove_llama31_8b_early_mt_results.json"
LATE_MT = "remove_llama31_8b_late_mt_results.json"
MID_V2 = "remove_llama31_8b_mid_v2_results.json"
IND_EARLY = "induce_gemma3_4b_induce_early_final_results.json"
IND_LATE = "induce_gemma3_4b_induce_late_final_results.json"

DEPTH = {0: "remove_llama31_8b_blk0_n4_attn_results.json",
         8: "remove_llama31_8b_blk8_n4_attn_results.json",
         12: "remove_llama31_8b_blk12_n4_attn_results.json",
         20: "remove_llama31_8b_blk20_n4_attn_results.json",
         24: "remove_llama31_8b_blk24_n4_attn_results.json"}
DEPTH_NOT_RUN = (4, 16, 28)


@dataclass
class Claim:
    id: str
    where: str            # where in the paper the number appears
    text: str             # the claim in words
    expected: float | bool | str
    compute: Callable[[], float | bool | str]
    tol: float = 0.005
    tags: tuple[str, ...] = field(default_factory=tuple)

    def check(self):
        try:
            got = self.compute()
        except Exception as e:            # a missing cell is a failure, not a crash
            return False, f"{type(e).__name__}: {e}"
        if isinstance(self.expected, bool) or isinstance(self.expected, str):
            return got == self.expected, got
        return abs(float(got) - float(self.expected)) <= self.tol, got


def _c(*a, **kw) -> Claim:
    return Claim(*a, **kw)


# =====================================================================
# Results 1: the cliff and the recipe dissociation
# =====================================================================
BEHAVIOURAL = [
    _c("ag.k0", "Results 1, Fig 1a", "Llama-3.1-8B ag_news zero-shot = 0.810",
       0.810, lambda: L.hosted_acc(LLAMA31, "ag_news", 0)),
    _c("ag.peak", "Results 1", "peak is 0.818 at K=1, not at K=0",
       0.818, lambda: L.hosted_acc(LLAMA31, "ag_news", 1)),
    _c("ag.peak_k", "Results 1", "the argmax over K is K=1",
       1, lambda: L.hosted_peak(LLAMA31, "ag_news")[0], tol=0),
    _c("ag.k2", "Results 1", "already below baseline by K=2 (0.793)",
       0.793, lambda: L.hosted_acc(LLAMA31, "ag_news", 2)),
    _c("ag.k64", "Results 1, Fig 1a", "collapses to 0.325 by K=64",
       0.325, lambda: L.hosted_acc(LLAMA31, "ag_news", 64)),
    _c("ag.drop", "Results 1", "a drop of 49 accuracy points from peak",
       0.493, lambda: L.hosted_acc(LLAMA31, "ag_news", 1) - L.hosted_acc(LLAMA31, "ag_news", 64)),
    _c("ag.ratio", "Results 1, Methods", "a64/a0 = 0.40 for Llama-3.1-8B",
       0.401, lambda: L.hosted_ratio(LLAMA31, "ag_news")),
    _c("sst2.k0", "Results 1", "sst2 zero-shot 0.940", 0.940,
       lambda: L.hosted_acc(LLAMA31, "sst2", 0)),
    _c("sst2.k64", "Results 1", "sst2 falls to 0.530", 0.530,
       lambda: L.hosted_acc(LLAMA31, "sst2", 64)),
    _c("mnli.k0", "Results 1", "MNLI zero-shot 0.620", 0.620,
       lambda: L.hosted_acc(LLAMA31, "mnli", 0)),
    _c("mnli.peak", "Results 1, Fig S4", "MNLI peaks at K=1 (0.687)", 0.687,
       lambda: L.hosted_acc(LLAMA31, "mnli", 1)),
    _c("mnli.k64", "Results 1", "MNLI falls to 0.405", 0.405,
       lambda: L.hosted_acc(LLAMA31, "mnli", 64)),
    _c("immune.gemma4b.k0", "Results 1", "Gemma-3-4B starts BELOW Llama at 0.785",
       0.785, lambda: L.hosted_acc(GEMMA4B, "ag_news", 0)),
    _c("immune.qwen.k0", "Results 1", "Qwen3-32B starts BELOW Llama at 0.768",
       0.768, lambda: L.hosted_acc(QWEN32B, "ag_news", 0)),
    _c("immune.below_prone", "Results 1", "both immune models start below the prone one",
       True, lambda: (L.hosted_acc(GEMMA4B, "ag_news", 0) < L.hosted_acc(LLAMA31, "ag_news", 0)
                      and L.hosted_acc(QWEN32B, "ag_news", 0) < L.hosted_acc(LLAMA31, "ag_news", 0))),
    _c("immune.ratio.gemma4b", "Results 1", "immune a64/a0 = 1.00 (Gemma-3-4B)",
       0.998, lambda: L.hosted_ratio(GEMMA4B, "ag_news")),
    _c("immune.ratio.gemma12b", "Results 1", "immune a64/a0 = 1.06 (Gemma-3-12B)",
       1.060, lambda: L.hosted_ratio(GEMMA12B, "ag_news")),
    _c("immune.ratio.gemma27b", "Results 1", "immune a64/a0 = 1.04 (Gemma-3-27B)",
       1.043, lambda: L.hosted_ratio(GEMMA27B, "ag_news")),
    _c("immune.ratio.qwen", "Results 1", "immune a64/a0 = 1.13 (Qwen3-32B)",
       1.126, lambda: L.hosted_ratio(QWEN32B, "ag_news")),
    _c("recipe.hermes.ratio", "Results 1, Fig 1b", "Hermes-3 a64/a0 = 1.03 on the same base",
       1.031, lambda: L.hosted_ratio(HERMES, "ag_news")),
    _c("recipe.dissociation", "Results 1, Fig 1b",
       "same base checkpoint: Meta recipe collapses, NousResearch recipe does not",
       True, lambda: L.hosted_ratio(LLAMA31, "ag_news") < 0.5 < 0.9 < L.hosted_ratio(HERMES, "ag_news")),
    _c("recipe.hermes.mnli.k0", "Fig S4", "Hermes-3 MNLI 0.470 -> 0.575", 0.470,
       lambda: L.hosted_acc(HERMES, "mnli", 0)),
    _c("recipe.hermes.mnli.k64", "Fig S4", "Hermes-3 MNLI rises to 0.575", 0.575,
       lambda: L.hosted_acc(HERMES, "mnli", 64)),
    _c("scale.deepseek.peak", "Results 1, Fig 1c", "DeepSeek-V3.2 peaks at 0.828 at K=4",
       0.828, lambda: L.panel_acc("deepseek_v3", "ag_news", 4)),
    _c("scale.deepseek.k64", "Results 1, Fig 1c", "DeepSeek-V3.2 falls to 0.332",
       0.332, lambda: L.panel_acc("deepseek_v3", "ag_news", 64)),
    _c("scale.671b_falls_4b_not", "Intro, Fig 1c",
       "a 671B model falls off the cliff where a 4B model does not", True,
       lambda: (L.panel_acc("deepseek_v3", "ag_news", 64) < 0.4
                and L.hosted_ratio(GEMMA4B, "ag_news") > 0.95)),
    _c("pertask.llama_improves_imdb", "Results 1, Limitations",
       "Llama-3.1-8B improves monotonically on imdb (regime is per task)", True,
       lambda: L.panel_acc("llama3_1_8b", "imdb", 64) >= L.panel_acc("llama3_1_8b", "imdb", 1)),
    _c("pertask.llama_improves_dbpedia", "Results 1, Limitations",
       "Llama-3.1-8B improves monotonically on dbpedia", True,
       lambda: L.panel_acc("llama3_1_8b", "dbpedia", 64) >= L.panel_acc("llama3_1_8b", "dbpedia", 1)),
]

# =====================================================================
# Results 2: localization
# =====================================================================
LOCALIZATION = [
    _c("region.early.ag", "Table 2, Fig 4b", "early band recovers +0.58 on ag_news K=16",
       0.581, lambda: L.recovery(EARLY_MT, "ag_news", 16)),
    _c("region.mid.ag", "Table 2", "mid band recovers +0.57 on ag_news K=16",
       0.565, lambda: L.recovery(MID_V2, "ag_news", 16), tol=0.02),
    _c("region.late.ag", "Table 2, Fig 4b", "late band recovers only +0.16",
       0.163, lambda: L.recovery(LATE_MT, "ag_news", 16)),
    _c("region.early.sst2", "Table 2", "early recovers +0.32 on sst2 K=64",
       0.316, lambda: L.recovery(EARLY_MT, "sst2", 64)),
    _c("region.early.trec", "Table 2", "early recovers +0.21 on trec K=64",
       0.213, lambda: L.recovery(EARLY_MT, "trec", 64)),
    _c("region.late.trec", "Table 2", "late recovers -0.11 on trec K=64",
       -0.105, lambda: L.recovery(LATE_MT, "trec", 64)),
    _c("region.range.early", "Results 2", "early per-seed range 0.500-0.635 on ag_news",
       True, lambda: (abs(min(L.per_seed_recovery(EARLY_MT, "ag_news", 16)) - 0.500) < 0.005
                      and abs(max(L.per_seed_recovery(EARLY_MT, "ag_news", 16)) - 0.635) < 0.005)),
    _c("region.range.late", "Results 2", "late per-seed range 0.040-0.210 on ag_news",
       True, lambda: (abs(min(L.per_seed_recovery(LATE_MT, "ag_news", 16)) - 0.040) < 0.005
                      and abs(max(L.per_seed_recovery(LATE_MT, "ag_news", 16)) - 0.210) < 0.005)),
    _c("region.nonoverlap", "Results 2", "per-seed distributions do not overlap", True,
       lambda: min(L.per_seed_recovery(EARLY_MT, "ag_news", 16))
               > max(L.per_seed_recovery(LATE_MT, "ag_news", 16))),
    _c("stats.p.ag", "Results 2, Table 2", "Welch p = 9.3e-6 for ag_news early vs late",
       9.3e-6, lambda: L.welch(L.per_seed_recovery(EARLY_MT, "ag_news", 16),
                               L.per_seed_recovery(LATE_MT, "ag_news", 16))[2], tol=1e-6),
    _c("stats.p.sst2", "Results 2, Table 2", "Welch p = 1.8e-7 for sst2 early vs late",
       1.8e-7, lambda: L.welch(L.per_seed_recovery(EARLY_MT, "sst2", 64),
                               L.per_seed_recovery(LATE_MT, "sst2", 64))[2], tol=1e-7),
    _c("stats.p.trec", "Results 2", "Welch p = 2.4e-3 for trec early vs late",
       2.4e-3, lambda: L.welch(L.per_seed_recovery(EARLY_MT, "trec", 64),
                               L.per_seed_recovery(LATE_MT, "trec", 64))[2], tol=1e-4),
    _c("depth.0_3", "Table S7, Fig 4a", "window 0-3 recovers +0.41", 0.407,
       lambda: L.recovery(DEPTH[0], "ag_news", 16)),
    _c("depth.8_11", "Table S7, Fig 4a", "window 8-11 peaks at +0.61", 0.610,
       lambda: L.recovery(DEPTH[8], "ag_news", 16)),
    _c("depth.12_15", "Table S7, Fig 4a", "window 12-15 is +0.60", 0.600,
       lambda: L.recovery(DEPTH[12], "ag_news", 16)),
    _c("depth.20_23", "Table S7", "window 20-23 decays to +0.11", 0.105,
       lambda: L.recovery(DEPTH[20], "ag_news", 16)),
    _c("depth.24_27", "Table S7, Fig 4a", "window 24-27 decays to +0.08", 0.082,
       lambda: L.recovery(DEPTH[24], "ag_news", 16)),
    _c("depth.unimodal", "Results 2, Fig 4a",
       "the measured depth profile is unimodal, peaking in 8-15", True,
       lambda: (L.recovery(DEPTH[8], "ag_news", 16) > L.recovery(DEPTH[0], "ag_news", 16)
                and L.recovery(DEPTH[12], "ag_news", 16) > L.recovery(DEPTH[20], "ag_news", 16)
                > L.recovery(DEPTH[24], "ag_news", 16))),
    _c("component.early.attn", "Table S8", "early attention-only +0.41", 0.407,
       lambda: L.recovery("remove_llama31_8b_blk0_n4_attn_results.json", "ag_news", 16)),
    _c("component.early.mlp", "Table S8", "early MLP-only +0.59 (MLP > attention)", 0.592,
       lambda: L.recovery("remove_llama31_8b_blk0_n4_mlp_results.json", "ag_news", 16)),
    _c("component.peak.attn", "Table S8", "peak attention-only +0.60", 0.600,
       lambda: L.recovery("remove_llama31_8b_blk12_n4_attn_results.json", "ag_news", 16)),
    _c("component.peak.mlp", "Table S8", "peak MLP-only +0.60 (equal at the peak)", 0.595,
       lambda: L.recovery("remove_llama31_8b_blk12_n4_mlp_results.json", "ag_news", 16)),
]

# =====================================================================
# Results 3: cross-model, cross-family, reverse direction
# =====================================================================
CAUSAL = [
    _c("x.3b.before.k0", "Results 3", "Llama-3.2-3B ag_news 0.64 at K=0", 0.635,
       lambda: L.arm_acc("remove_llama32_3b_early_t6_results.json", "ag_news", 0, "before")),
    _c("x.3b.before.k16", "Results 3", "Llama-3.2-3B falls to 0.24 at K=16", 0.240,
       lambda: L.arm_acc("remove_llama32_3b_early_t6_results.json", "ag_news", 16, "before")),
    _c("x.3b.early", "Table 2, Fig 4c", "Llama-3.2-3B early recovers +0.58", 0.581,
       lambda: L.recovery("remove_llama32_3b_early_t6_results.json", "ag_news", 16)),
    _c("x.3b.late", "Table 2, Fig 4c", "Llama-3.2-3B late recovers +0.00", 0.000,
       lambda: L.recovery("remove_llama32_3b_late_t6_results.json", "ag_news", 16)),
    _c("x.ds7b.early", "Table 2", "DeepSeek-7B early +0.42", 0.423,
       lambda: L.recovery("remove_deepseek_7b_remove_early_v2_results.json", "ag_news", 16)),
    _c("x.ds7b.mid", "Table 2", "DeepSeek-7B mid +0.42", 0.423,
       lambda: L.recovery("remove_deepseek_7b_remove_mid_v2_results.json", "ag_news", 16)),
    _c("x.ds7b.late", "Table 2", "DeepSeek-7B late +0.18", 0.180,
       lambda: L.recovery("remove_deepseek_7b_remove_late_v2_results.json", "ag_news", 16)),
    _c("x.mistral.early", "Table 2", "Mistral-7B early +0.39", 0.390,
       lambda: L.recovery("remove_mistral7b_remove_early_k32_results.json", "ag_news", 32)),
    _c("x.mistral.late", "Table 2", "Mistral-7B late +0.33 (the weak family)", 0.330,
       lambda: L.recovery("remove_mistral7b_remove_late_k32_results.json", "ag_news", 32)),
    _c("x.mistral.weak", "Results 3",
       "on Mistral-7B late reaches >=80% of the best band", True,
       lambda: L.recovery("remove_mistral7b_remove_late_k32_results.json", "ag_news", 32)
               / L.recovery("remove_mistral7b_remove_early_k32_results.json", "ag_news", 32) >= 0.80),
    _c("x.internlm.early", "Table 2", "InternLM2.5 early is ineffective (-0.02)", -0.017,
       lambda: L.recovery("remove_internlm2_7b_remove_early_v8_results.json", "ag_news", 64)),
    _c("x.internlm.mid", "Table 2", "InternLM2.5 mid +0.16 (locus shifts to mid)", 0.157,
       lambda: L.recovery("remove_internlm2_7b_remove_mid_v4_results.json", "ag_news", 64)),
    _c("x.internlm.late", "Table 2", "InternLM2.5 late +0.07", 0.073,
       lambda: L.recovery("remove_internlm2_7b_remove_late_v4_results.json", "ag_news", 64)),
    _c("induce.k0.before", "Results 3, Fig 4d", "induce arm zero-shot before = 0.78", 0.780,
       lambda: L.arm_acc(IND_EARLY, "ag_news", 0, "before")),
    _c("induce.k0.after", "Results 3, Fig 4d", "early locus impairs zero-shot to 0.45", 0.450,
       lambda: L.arm_acc(IND_EARLY, "ag_news", 0, "after")),
    _c("induce.k16", "Results 3, Fig 4d", "many-shot unaffected at K=16 (-0.018)", -0.018,
       lambda: L.recovery(IND_EARLY, "ag_news", 16)),
    _c("induce.late.k0", "Results 3, Fig 4d", "late locus leaves zero-shot unchanged (+0.025)",
       0.025, lambda: L.recovery(IND_LATE, "ag_news", 0)),
    _c("induce.double_dissoc", "Results 3",
       "early hits K=0 and spares K=16; late does neither", True,
       lambda: (L.recovery(IND_EARLY, "ag_news", 0) < -0.30
                and abs(L.recovery(IND_EARLY, "ag_news", 16)) < 0.05
                and abs(L.recovery(IND_LATE, "ag_news", 0)) < 0.05)),
]

# =====================================================================
# Mechanistic measures (one supports the locus, one does not)
# =====================================================================
MECHANISM = [
    _c("mech.zeroshot", "Fig 3", "function-vector baseline = zero-shot 0.87", 0.87,
       lambda: L.mechanism()["zero_shot_acc"]),
    _c("mech.icl", "Fig 3", "in-context accuracy at the cliff point = 0.36", 0.36,
       lambda: L.mechanism()["icl_acc"]),
    _c("mech.fv.l10", "Fig 3", "FV tracks the baseline through layer 10", 0.85,
       lambda: L.mechanism()["fv_effect_by_layer"][10]),
    _c("mech.fv.onset", "Fig 3", "FV declines from layer 11 (0.78)", 0.78,
       lambda: L.mechanism()["fv_effect_by_layer"][11]),
    _c("mech.fv.l14", "Fig 3", "FV drops sharply at layer 14 to 0.39", 0.39,
       lambda: L.mechanism()["fv_effect_by_layer"][14]),
    _c("mech.fv.persists", "Fig 3",
       "FV stays low to the end of the stack (an onset, not a band)", True,
       lambda: st.mean(L.mechanism()["fv_effect_by_layer"][16:]) < 0.45),
    _c("mech.ih.l2", "Fig 3", "induction-head score peaks at layer 2 (0.081)", 0.081,
       lambda: L.mechanism()["induction_score_by_layer"][2], tol=0.001),
    _c("mech.ih.l15", "Fig 3", "second peak at layer 15 (0.079)", 0.079,
       lambda: L.mechanism()["induction_score_by_layer"][15], tol=0.001),
    _c("mech.ih.inband", "Fig 3", "induction mean inside blocks 8-15 = 0.033", 0.0327,
       lambda: L.induction_band_means()[0], tol=0.001),
    _c("mech.ih.outband", "Fig 3", "induction mean outside blocks 8-15 = 0.031", 0.0313,
       lambda: L.induction_band_means()[1], tol=0.001),
]

# =====================================================================
# The audited caveats -- properties of the data that the paper now discloses
# =====================================================================
CAVEATS = [
    _c("caveat.decode.census", "Supp Note 4", "265 of 3102 intervention cells are 0.000",
       True, lambda: L.zero_accuracy_census()[:2] == (265, 3102)),
    _c("caveat.decode.all_unparseable", "Supp Note 4",
       "every 0.000 cell is 100% unparseable, i.e. a decode failure not an accuracy",
       True, lambda: L.zero_accuracy_census()[0] == L.zero_accuracy_census()[2]),
    _c("caveat.sst2_late_degenerate", "Table 2, Supp Note 4",
       "the sst2 late cell is a decode failure, so no recovery value is reported",
       True, lambda: L.is_decode_failure(LATE_MT, "sst2", 64)),
    _c("caveat.backward_transfer", "Supp Note 2",
       "backward transfer is zero to within noise (|delta| <= 0.04)", True,
       lambda: max(abs(v) for v in L.backward_transfer()) <= 0.04),
    _c("caveat.induce_fallback", "Supp Note 5, Table S2",
       "the induce-arm default-label fallback fired 0 times in 500", 0,
       lambda: L._json("interventions/induce_fallback_audit.json")["n_fallback"], tol=0),
    _c("caveat.induce_targets_correct", "Supp Note 5",
       "induce-arm targets agree with gold on 81.6% of examples", 0.816,
       lambda: L._json("interventions/induce_fallback_audit.json")["target_gold_agreement_rate"]),
]

# =====================================================================
# NEGATIVE CONTROLS: the data properties that forced claims to be withdrawn.
# These SHOULD pass. Each one is a claim an earlier draft made that the archive
# does not support; asserting them here keeps the withdrawals auditable.
# =====================================================================
NEGATIVE_CONTROLS = [
    _c("neg.induction_does_not_localize", "Fig 3 (negative result)",
       "induction-head score does NOT localize: in-band mean ~ out-of-band mean",
       True, lambda: abs(L.induction_band_means()[0] - L.induction_band_means()[1]) < 0.005,
       tags=("withdrawn",)),
    _c("neg.layer2_outside_band", "Fig 3 (negative result)",
       "layer 2 is OUTSIDE blocks 8-15, so 'both peaks in band' was false",
       True, lambda: not (8 <= 2 < 16), tags=("withdrawn",)),
    _c("neg.ih_top8_mostly_outside", "Fig 3 (negative result)",
       "only 1 of the 8 highest induction layers lies inside blocks 8-15", 1,
       lambda: sum(1 for i in sorted(range(32),
                   key=lambda j: -L.mechanism()["induction_score_by_layer"][j])[:8]
                   if 8 <= i < 16), tol=0, tags=("withdrawn",)),
    _c("neg.synthetic_at_chance", "Supp Note 3, Table S1",
       "no model exceeds 0.58 on the synthetic rule probe at any K", True,
       lambda: max(L.panel_acc(m, "synthetic_rule_v1", k)
                   for m in ("llama3_1_8b", "deepseek_v3", "gemma3_4b", "gemma3_12b",
                             "gemma3_27b", "qwen3_32b", "llama3_3_70b")
                   for k in (1, 2, 4, 8, 16, 32, 64)) < 0.58, tags=("withdrawn",)),
    _c("neg.synthetic_prone_beats_immune", "Supp Note 3",
       "cliff-PRONE Llama scores above cliff-IMMUNE Gemma-3-4B on the probe", True,
       lambda: L.panel_acc("llama3_1_8b", "synthetic_rule_v1", 64)
               > L.panel_acc("gemma3_4b", "synthetic_rule_v1", 64), tags=("withdrawn",)),
    _c("neg.synthetic_zeroshot_equals_manyshot", "Supp Note 3",
       "zero-shot on the probe already equals many-shot: demonstrations add nothing",
       True, lambda: abs(L.zero_shot("gemma3_4b", "synthetic_rule_v1")
                         - L.panel_acc("gemma3_4b", "synthetic_rule_v1", 64)) < 0.05,
       tags=("withdrawn",)),
    _c("neg.projection_key_beats_value", "Supp Note 5, Table S9",
       "in the archive KEY recovery exceeds VALUE, reversing the withdrawn K-vs-V claim",
       True, lambda: (float([r for r in L.component_csv() if r["component"] == "k_only"][0]["recover_K32"])
                      > float([r for r in L.component_csv() if r["component"] == "v_only"][0]["recover_K32"])),
       tags=("withdrawn",)),
    _c("neg.projection_all_recover", "Table S9",
       "all six projections recover the cliff within seed noise (+0.49..+0.60)", True,
       lambda: all(0.45 <= float(r["recover_K32"]) <= 0.65 for r in L.component_csv()),
       tags=("withdrawn",)),
    _c("neg.depth_windows_not_run", "Table S7, Fig 4a",
       "depth windows 4-7, 16-19 and 28-31 have no data and must not be interpolated",
       True, lambda: not any(
           L.cell_exists(f"remove_llama31_8b_blk{s}_n4_attn_results.json", "ag_news", 16)
           for s in DEPTH_NOT_RUN), tags=("withdrawn",)),
    _c("neg.mid_cells_absent", "Table 2",
       "mid cells for sst2/trec and for Llama-3.2-3B were never run", True,
       lambda: not (L.cell_exists(MID_V2, "sst2", 64) or L.cell_exists(MID_V2, "trec", 64))),
]

ALL_GROUPS = {
    "behavioural": BEHAVIOURAL,
    "localization": LOCALIZATION,
    "causal": CAUSAL,
    "mechanism": MECHANISM,
    "caveats": CAVEATS,
    "negative-controls": NEGATIVE_CONTROLS,
}
ALL_CLAIMS = [c for g in ALL_GROUPS.values() for c in g]
