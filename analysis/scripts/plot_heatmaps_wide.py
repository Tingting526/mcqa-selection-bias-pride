#!/usr/bin/env python3
"""Side-by-side versions of the two cyclic-diagnostic heatmaps.

Same data, colour maps, and number formats as the stacked originals
(plot_phase4_thesis.py: benchmark_robustness; build_phase7_robustness_assets.py:
corrected_permutation_robustness). Only the panel arrangement changes.
The originals are left untouched because technical_audit/AUDIT_TESTS/verify_figures.py
parses their text. Outputs: Paper/figures/benchmark_robustness_wide.pdf and
Paper/figures/corrected_permutation_robustness_wide.pdf.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UNCORRECTED = ROOT / 'analysis/outputs/phase3b_20261018/verified/analysis_results.json'
CORRECTED = ROOT / 'analysis/outputs/phase6_robustness_20261018/analysis_results.json'
MODELS = ['llama-3.1-8b-instruct', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
          'mistral-7b-instruct-v0.3', 'qwen3-8b', 'glm-4-9b-chat-hf']
SHORT = ['Llama 3.1 8B', 'Qwen2.5 7B', 'Gemma 2 9B', 'Mistral 7B v0.3', 'Qwen3 8B', 'GLM 4 9B']
DATASETS = ['mmlu', 'arc', 'csqa', 'medqa', 'medmcqa']
LABELS = ['MMLU', 'ARC-C', 'CSQA', 'MedQA', 'MedMCQA']


def heat(fig, ax, values, cmap, vmin, vmax, fmt, title, unit, ylabels):
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(5), LABELS, rotation=35, ha='right')
    ax.set_yticks(range(6), SHORT if ylabels else [])
    ax.set_title(title, pad=7, fontsize=9)
    for i in range(6):
        for j in range(5):
            col = im.cmap(im.norm(values[i, j]))
            lum = .2126 * col[0] + .7152 * col[1] + .0722 * col[2]
            ax.text(j, i, fmt.format(values[i, j]), ha='center', va='center', fontsize=7,
                    color='black' if lum > .55 else 'white')
    cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.03)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main(preview_dir):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'xtick.labelsize': 7.5,
                         'ytick.labelsize': 8.5, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Uncorrected cyclic RStd and semantic consistency
    d = json.loads(UNCORRECTED.read_text())
    rows = {(r['model'], r['dataset']): r for r in d['permutation_metrics']}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    for k, (key, mult, title, unit, vmax) in enumerate([
            ('cyclic_rstd', 1, 'Cyclic recall imbalance', 'RStd (pp)', None),
            ('semantic_consistency', 100, 'Same content at every shift', 'Questions (%)', 100)]):
        values = np.array([[mult * rows[m, ds][key] for ds in DATASETS] for m in MODELS])
        heat(fig, axes[k], values, 'viridis', 0, vmax, '{:.1f}', title, unit, k == 0)
    fig.subplots_adjust(left=.17, right=.9, top=.9, bottom=.2, wspace=.38)
    out1 = ROOT / 'Paper/figures/benchmark_robustness_wide.pdf'
    fig.savefig(out1, facecolor='white')
    if preview_dir:
        fig.savefig(Path(preview_dir) / 'benchmark_robustness_wide.png', dpi=170, facecolor='white')
    plt.close(fig)
    # Fixed-prior PriDe minus Default
    c = json.loads(CORRECTED.read_text())['corrected_permutation']
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    for k, (key, title) in enumerate([
            ('delta_cyclic_rstd_pp', 'Change in cyclic RStd\n(lower is better)'),
            ('delta_semantic_consistency_pp', 'Change in consistency\n(higher is better)')]):
        values = np.array([[next(r[key] for r in c if r['model'] == m and r['benchmark'] == ds)
                            for ds in DATASETS] for m in MODELS])
        lim = np.abs(values).max()
        heat(fig, axes[k], values, 'RdBu', -lim, lim, '{:+.2f}', title, 'PriDe − Default (pp)', k == 0)
    fig.subplots_adjust(left=.17, right=.9, top=.86, bottom=.2, wspace=.38)
    out2 = ROOT / 'Paper/figures/corrected_permutation_robustness_wide.pdf'
    fig.savefig(out2, facecolor='white')
    if preview_dir:
        fig.savefig(Path(preview_dir) / 'corrected_permutation_robustness_wide.png', dpi=170, facecolor='white')
    plt.close(fig)
    print(out1, out2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview-dir', default=None)
    main(ap.parse_args().preview_dir)
