"""Redraw the original Figure 1 series with recorded evaluation-run variability.

No query-level uncertainty is inferred from summary records. This generator does
not change the original plotting implementation or any original result file.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import ast
import csv
import hashlib
import json
import math
import os
import statistics

PAPER = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = PAPER / 'code/results'
TARGET = PAPER / 'figures/revision'
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KS = [0, 1, 2, 4, 8, 16, 32, 64]
SOURCE = PAPER / 'code/cliff/figures.py'
PANEL = {
    'Llama-3.1-8B-Instruct': ('recipe_difference_llama31_8b_instruct.json', '#d62728', '-', 'o'),
    'Gemma-3-4B-it': ('recipe_difference_gemma3_4b_it.json', '#1f77b4', '-', 's'),
    'Gemma-3-12B-it': ('recipe_difference_gemma3_12b_it.json', '#6baed6', '--', '^'),
    'Gemma-3-27B-it': ('recipe_difference_gemma3_27b_it.json', '#9467bd', '--', 'v'),
    'Qwen3-32B': ('recipe_difference_qwen3_32b.json', '#8c564b', ':', 'P'),
}
LEDGER = []
SOURCES = set()


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run_rows(path, field, task, ks, n):
    SOURCES.add(path)
    obj = json.loads(path.read_text())
    rows = []
    for k in ks:
        records = [r for r in obj[field] if r['task'] == task and r['k'] == k]
        assert len(records) == n and len({r['seed'] for r in records}) == n
        assert all(not r.get('skipped', False) and 0 <= r['accuracy'] <= 1 for r in records)
        values = [r['accuracy'] for r in records]
        rows.append(dict(K=k, mean=statistics.mean(values), low=min(values), high=max(values),
                         n_evaluation_runs=n, uncertainty='recorded_min_max',
                         per_run_n_test=json.dumps(sorted({r['n_test'] for r in records})),
                         recorded_seeds=json.dumps(sorted(r['seed'] for r in records)),
                         recorded_values=json.dumps(values), source=str(path.relative_to(PAPER)),
                         source_field=field, task=task))
    return rows


def hosted(filename, task, ks=KS):
    return run_rows(RES / 'hosted_sweeps' / filename, 'cells', task, ks, 3)


def deepseek():
    path = RES / 'exp1a_complete.csv'; SOURCES.add(path)
    records = [r for r in csv.DictReader(path.open()) if r['model'] == 'deepseek_v3' and r['task'] == 'ag_news']
    assert {int(r['k']) for r in records} == set(KS[1:]) and len(records) == 7
    rows = []
    for r in sorted(records, key=lambda x: int(x['k'])):
        mean, sd = float(r['mean_acc']), float(r['std_acc'])
        assert int(r['n_seeds']) == 10
        rows.append(dict(K=int(r['k']), mean=mean, low=mean-sd, high=mean+sd,
                         n_evaluation_runs=10, uncertainty='reported_plus_minus_one_SD',
                         per_run_n_test='not_identified_by_CSV', recorded_seeds='not_in_summary_CSV',
                         recorded_values='not_in_summary_CSV', source=str(path.relative_to(PAPER)),
                         source_field='model=deepseek_v3', task='ag_news'))
    return rows


def plot(ax, data, panel, model, color, ls, marker, label=None, alpha=1, ms=4, lw=1.8):
    x = [KS.index(r['K']) for r in data]
    y = np.array([r['mean'] for r in data])
    lo, hi = np.array([r['low'] for r in data]), np.array([r['high'] for r in data])
    ax.errorbar(x, y, yerr=np.array([y-lo, hi-y]), fmt=marker, linestyle=ls,
                color=color, markersize=ms, linewidth=lw, elinewidth=.85,
                capsize=2, capthick=.85, alpha=alpha, label=label, zorder=3)
    for r in data:
        LEDGER.append({'panel': panel, 'model': model, **r})


def verify_against_original():
    # Execute only the original read-only loader definitions in an isolated
    # namespace. No original figure-generation or module-level code is run.
    tree = ast.parse(SOURCE.read_text())
    names = {'hosted', 'curve', 'exp1a', 'cells', 'before_acc'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {'json': json, 'csv': csv, 'st': statistics, 'defaultdict': defaultdict,
                 'results_dir': lambda: RES}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    checks = 0
    for row in LEDGER:
        p = PAPER / row['source']; k = row['K']; task = row['task']
        if row['source_field'] == 'cells':
            expected = namespace['hosted'](p.name)[task, k]
        elif row['source_field'] == 'before_cells':
            expected = namespace['before_acc'](p.name, task, k)
        else:
            expected = namespace['exp1a']('deepseek_v3', task)[k]
        assert math.isclose(expected, row['mean'], rel_tol=0, abs_tol=1e-15)
        checks += 1
        # Separate raw-record calculation verifies displayed ranges/SD; the
        # summary CSV is not silently used to invent per-query intervals.
        if row['uncertainty'] == 'recorded_min_max':
            original = json.loads(p.read_text())[row['source_field']]
            actual = [r['accuracy'] for r in original if r['task'] == task and r['k'] == k]
            assert row['low'] == min(actual) and row['high'] == max(actual)
            assert row['n_evaluation_runs'] == len(actual)
        else:
            raw = next(r for r in csv.DictReader(p.open()) if r['model'] == 'deepseek_v3' and r['task'] == task and int(r['k']) == k)
            assert row['low'] == float(raw['mean_acc']) - float(raw['std_acc'])
            assert row['high'] == float(raw['mean_acc']) + float(raw['std_acc'])
        checks += 3
    return checks


def main():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
                         'axes.titlesize': 10.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    TARGET.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=.30)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    x = np.arange(len(KS)); ax = axes[0]
    for name, (filename, color, ls, marker) in PANEL.items():
        plot(ax, hosted(filename, 'ag_news'), 'a', name, color, ls, marker, name)
    ax.axhline(.25, ls=':', lw=1, color='#888')
    ax.text(7.4, .265, 'uniform reference = 0.25', fontsize=7.5, color='#666', ha='right')
    ax.axvline(.5, ls=':', lw=.9, color='#bbb')
    ax.text(.02, .955, '$K{=}0$', fontsize=8, color='#666', transform=ax.transAxes)
    ax.text(5.1, .94, 'stable on AG News', color='#1f77b4', fontsize=8.5, ha='center')
    ax.set_title('a   AG News, hosted panel', loc='left', fontweight='bold')
    ax.legend(loc='lower left', frameon=False, fontsize=7.2, bbox_to_anchor=(-.02, .06))
    ax = axes[1]
    for task, marker, ls in [('ag_news', 'o', '-'), ('sst2', 's', '--'), ('mnli', '^', ':')]:
        plot(ax, hosted(PANEL['Llama-3.1-8B-Instruct'][0], task), 'b', 'Llama-3.1-8B-Instruct', '#d62728', ls, marker)
        plot(ax, hosted('hermes3_full.json', task), 'b', 'Hermes-3-Llama-3.1-8B', '#2ca02c', ls, marker)
    handles = [plt.Line2D([], [], color='#d62728', lw=2, label='Llama-3.1-8B-Instruct'),
               plt.Line2D([], [], color='#2ca02c', lw=2, label='Hermes-3-Llama-3.1-8B'),
               plt.Line2D([], [], color='#555', lw=1.3, ls='-', marker='o', ms=4, label='AG News'),
               plt.Line2D([], [], color='#555', lw=1.3, ls='--', marker='s', ms=4, label='SST2'),
               plt.Line2D([], [], color='#555', lw=1.3, ls=':', marker='^', ms=4, label='MNLI')]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=7.5)
    ax.set_title('b   Shared base, native formats', loc='left', fontweight='bold')
    ax = axes[2]
    plot(ax, deepseek(), 'c', 'DeepSeek-V3.2', '#ff7f0e', '-', 'D',
         'DeepSeek-V3.2, 671B (hosted, $K{\\geq}1$)')
    plot(ax, run_rows(RES / 'interventions/remove_llama32_3b_early_t6_results.json', 'before_cells', 'ag_news', [0, 1, 4, 16], 5),
         'c', 'Llama-3.2-3B', '#e377c2', '-', '*', 'Llama-3.2-3B (self-hosted)', ms=7)
    plot(ax, hosted(PANEL['Gemma-3-4B-it'][0], 'ag_news'), 'c', 'Gemma-3-4B-it', '#1f77b4', '-', 's',
         'Gemma-3-4B-it (hosted)', alpha=.65, lw=1.4)
    ax.axhline(.25, ls=':', lw=1, color='#888')
    ax.text(7.4, .265, 'uniform reference = 0.25', fontsize=7.5, color='#666', ha='right')
    ax.set_title('c   AG News across scales', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False, fontsize=7, bbox_to_anchor=(-.02, .94))
    ax.text(.99, .965, 'pipelines shown separately', transform=ax.transAxes, ha='right',
            fontsize=7, color='#777', style='italic')
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(KS)
        ax.set_xlabel('$K$ (demonstrations)'); ax.set_ylabel('Test accuracy')
        ax.set_ylim(.20, 1)
    fig.text(.5, -.025, 'Bars: min–max over 3 hosted / 5 self-hosted runs; ±1 SD over 10 DeepSeek runs. '
             'Evaluation-run variability, not query or training confidence intervals.', ha='center', fontsize=8)
    source_hashes = {str(p.relative_to(PAPER)): digest(p) for p in sorted(SOURCES | {SOURCE})}
    nchecks = verify_against_original()
    for ext in ['pdf', 'png']:
        fig.savefig(TARGET / f'final_behavioral_panel.{ext}', bbox_inches='tight', dpi=240,
                    metadata={'Creator': 'Recorded result summaries; reproducible revision generator'} if ext == 'pdf' else None)
    plt.close(fig)
    csvpath = HERE / 'final_behavioral_panel_plotted_cells.csv'
    with csvpath.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(LEDGER[0])); writer.writeheader(); writer.writerows(LEDGER)
    assert source_hashes == {str(p.relative_to(PAPER)): digest(p) for p in sorted(SOURCES | {SOURCE})}
    audit = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source_hashes_before_and_after_match': source_hashes,
             'generator_sha256': digest(__file__), 'plotted_rows': len(LEDGER),
             'independent_loader_mean_and_raw_range_checks': nchecks,
             'source_series_identical_to_original_figure1': True,
             'uncertainty': {'hosted_JSON': 'min–max of 3 recorded evaluation runs',
                             'self_hosted_Llama3_2_3B_JSON': 'min–max of 5 recorded evaluation runs',
                             'DeepSeek_CSV': 'mean ± 1 reported evaluation-run SD over 10 runs'},
             'no_inferred_query_training_or_bootstrap_intervals': True,
             'no_new_experiments_or_original_file_changes': True,
             'new_artifacts': {str(p.relative_to(PAPER)): digest(p) for p in [csvpath, TARGET / 'final_behavioral_panel.pdf', TARGET / 'final_behavioral_panel.png']}}
    (HERE / 'final_behavioral_panel_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'plotted_rows': len(LEDGER), 'checks': nchecks}))


if __name__ == '__main__':
    main()
