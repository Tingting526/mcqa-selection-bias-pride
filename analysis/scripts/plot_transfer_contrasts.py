#!/usr/bin/env python3
"""RQ3 accuracy contrasts (I-D, T-D, T-I) with paired bootstrap intervals.

Reads the approved Phase 4 bootstrap rows; no values are recomputed.
Writes Paper/figures/transfer_accuracy_contrasts.pdf (and a PNG preview
when --preview is given). The older transfer_accuracy_ci.pdf is left
untouched because technical_audit/AUDIT_TESTS/verify_figures.py parses it.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / 'analysis/outputs/phase4_20261018_20261018T091030Z/statistics_results.json'
MODELS = ['llama-3.1-8b-instruct', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
          'mistral-7b-instruct-v0.3', 'qwen3-8b', 'glm-4-9b-chat-hf']
SHORT = ['Llama 3.1 8B', 'Qwen2.5 7B', 'Gemma 2 9B', 'Mistral 7B v0.3', 'Qwen3 8B', 'GLM 4 9B']
CONTRASTS = [('in_domain_minus_default', r'$I-D$'),
             ('transfer_minus_default', r'$T-D$'),
             ('transfer_minus_in_domain', r'$T-I$')]
DIRECTIONS = [('medqa', 'MedQA → MedMCQA\n(n = 3,744)', '#256598'),
              ('medmcqa', 'MedMCQA → MedQA\n(n = 1,210)', '#b2671d')]


def main(preview):
    rows = json.loads(STATS.read_text())['bootstrap_rows']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 5.0), sharex=True, sharey=True)
    for r, (source, label, color) in enumerate(DIRECTIONS):
        for c, (key, title) in enumerate(CONTRASTS):
            ax = axes[r, c]
            sel = [next(x for x in rows if x['model'] == m and x['source'] == source and x['contrast'] == key)
                   for m in MODELS]
            p = np.array([x['difference_pp'] for x in sel])
            lo = np.array([x['ci_lower_pp'] for x in sel])
            hi = np.array([x['ci_upper_pp'] for x in sel])
            y = np.arange(6)
            excl = (lo > 0) | (hi < 0)
            ax.errorbar(p, y, xerr=np.array([p - lo, hi - p]), fmt='none', ecolor=color, capsize=2.5, lw=1)
            ax.scatter(p[~excl], y[~excl], s=22, facecolor='white', edgecolor=color, zorder=3, linewidths=1)
            ax.scatter(p[excl], y[excl], s=22, color=color, zorder=4)
            ax.axvline(0, color='#777777', lw=.7)
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.set_xlim(-4, 4.5)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_yticks(y, SHORT)
                ax.set_ylabel(label, fontsize=8.5)
            if r == 1:
                ax.set_xlabel('Accuracy difference (pp)', fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.subplots_adjust(left=.25, right=.98, top=.93, bottom=.11, wspace=.12, hspace=.18)
    out = ROOT / 'Paper/figures/transfer_accuracy_contrasts.pdf'
    fig.savefig(out, facecolor='white')
    if preview:
        fig.savefig(preview, dpi=170, facecolor='white')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', default=None)
    main(ap.parse_args().preview)
