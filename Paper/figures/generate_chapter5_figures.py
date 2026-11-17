#!/usr/bin/env python3
"""Reproduce the Chapter 5 figures without model inference.

Run: python Paper/figures/generate_chapter5_figures.py --output-dir FRESH_DIRECTORY
Requires matplotlib and numpy. The default uses the bundled full-precision
snapshot. --refresh-data reads the immutable project result JSONs instead.
The source manifest records their SHA-256 hashes and original selectors.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURE_OUTPUT = HERE
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/chapter5_mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

MODELS = ['llama-3.1-8b-instruct', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
          'mistral-7b-instruct-v0.3', 'qwen3-8b', 'glm-4-9b-chat-hf']
NAMES = ['Llama 3.1 8B', 'Qwen2.5 7B', 'Gemma 2 9B', 'Mistral 7B v0.3', 'Qwen3 8B', 'GLM 4 9B']
COLORS = dict(default='#8A9197', cpa='#2878A0', pride='#D17A25', transfer='#8866A5',
              single='#247D78', multi='#8866A5')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
    'axes.titlesize': 9, 'axes.labelsize': 8.2, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.spines.left': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.axisbelow': True})

SOURCES = {
 'primary': 'analysis/outputs/phase3b_20261018/verified/analysis_results.json',
 'statistics': 'analysis/outputs/phase4_20261018_20261018T091030Z/statistics_results.json',
 'robustness': 'analysis/outputs/phase6_robustness_20261018/analysis_results.json',
 'resampling': 'analysis/outputs/phase6_robustness_20261018/seed_resampling_results.json',
 'strengthening': 'analysis/outputs/thesis_strengthening_20261021/analysis_results.json',
 'templates': 'analysis/outputs/thesis_strengthening_20261021/gpu_sensitivity_results.json'}

def extract():
    src = {k: json.loads((ROOT / p).read_text()) for k, p in SOURCES.items()}
    medical = []
    for r in src['primary']['summary_rows']:
        if r['benchmark'] not in ('medqa', 'medmcqa'):
            continue
        d = next(x for x in src['statistics']['direction_rows']
                 if (x['model'], x['benchmark']) == (r['model'], r['benchmark']))
        medical.append({**r, 'exact_pride_delta_accuracy_pp':
            100 * d['exact_aggregate_numerator'] / d['exact_aggregate_denominator'],
            'default_correct': d['default_correct'],
            'hybrid_correct_by_draw': d['hybrid_correct_by_draw']})
    transfers = []
    for r in src['primary']['transfer_results']:
        switches = next(x for x in src['strengthening']['transfer']
                        if (x['model'], x['source']) == (r['model'], r['source']))
        counts = dict(gain=0, loss=0, both_wrong=0)
        for p in r['paired_predictions']:
            a, b, g = (p[k] for k in ('in_domain_prediction', 'transfer_prediction', 'gold_index'))
            if a != b:
                counts['gain' if b == g else 'loss' if a == g else 'both_wrong'] += 1
        assert counts == dict(gain=switches['correct_gained_n'], loss=switches['correct_lost_n'],
                              both_wrong=switches['changed_both_wrong_n'])
        transfers.append(dict(model=r['model'], source=r['source'], target=r['target'],
            metrics=r['metrics'], source_prior=r['source_prior'], target_prior=r['target_prior'],
            tv=switches['prior_total_variation'], counts=counts,
            changed_n=sum(counts.values()), n=switches['n']))
    # Per-draw rows are separate from the summary rows in the saved result.
    resample_key = next(k for k, v in src['resampling'].items()
        if isinstance(v, list) and len(v) == 60 and isinstance(v[0], dict) and 'seed' in v[0])
    return dict(medical=medical, transfers=transfers,
        subgroups=src['strengthening']['subgroups'],
        templates=[r for r in src['templates']['results'] if r['dataset'] in ('medqa', 'medmcqa')],
        equal_budget=src['robustness']['equal_budget'],
        resampling=src['resampling'][resample_key],
        intervals=src['statistics']['bootstrap_rows'],
        source_manifest={k: dict(path=p, sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
                         for k, p in SOURCES.items()})

def indexed(rows, key, value):
    return [next(r for r in rows if r['model'] == m and r[key] == value) for m in MODELS]

def style(ax, title, xlabel, xlim, names=True, signed=True):
    ax.set_title(title, loc='left', pad=9, fontweight='bold')
    ax.set_yticks(range(6), NAMES if names else ['']*6)
    ax.set_ylim(5.65, -.65)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.grid(axis='x', color='#E3E5E7', linewidth=.55)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    if signed:
        ax.axvline(0, color='#444A4E', linewidth=.8, zorder=2)

def save(fig, name):
    # Fixed physical width matches the thesis's 160 mm text block.
    for ext in ('pdf', 'svg'):
        fig.savefig(FIGURE_OUTPUT / f'{name}.{ext}', facecolor='white')
    preview = FIGURE_OUTPUT / 'previews'
    preview.mkdir(exist_ok=True)
    fig.savefig(preview / f'{name}.png', dpi=180, facecolor='white')
    plt.close(fig)

def medical(data, ds):
    rows = indexed(data['medical'], 'benchmark', ds)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
    fig.subplots_adjust(left=.22, right=.985, bottom=.17, top=.79, wspace=.24)
    for j, (title, label, lim) in enumerate([
        ('A  Accuracy', r'$\Delta$Accuracy (pp)', (-1, 4.6)),
        ('B  Original RStd', r'$\Delta$Original RStd (pp)', (-24, 5))]):
        for k, (method, color) in enumerate([('CPA', COLORS['cpa']), ('PriDe (5%)', COLORS['pride'])]):
            if j == 0:
                vals = [r['cpa_minus_default_accuracy_pp'] if k == 0 else r['exact_pride_delta_accuracy_pp'] for r in rows]
            else:
                vals = [r['cpa_minus_default_rstd_pp'] if k == 0 else r['pride_minus_default_rstd_pp'] for r in rows]
            axes[j].barh(np.arange(6) + (k-.5)*.32, vals, height=.29, color=color, label=method)
        style(axes[j], title, label, lim, names=j == 0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.59, .99))
    save(fig, f'medical_{ds}_effects')

def choice_types(data):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.85))
    fig.subplots_adjust(left=.22, right=.99, bottom=.17, top=.8, wspace=.35)
    for j, (field, title, label, lim) in enumerate([
        ('consistency_percent', 'A  Consistency', 'Semantic consistency (%)', (0, 62)),
        ('cyclic_rstd_pp', 'B  Cyclic RStd', 'Cyclic RStd (pp)', (0, 28)),
        ('hybrid_minus_default_accuracy_pp', 'C  Accuracy change', r'$\Delta$Accuracy (pp)', (-1.2, 2.7))]):
        for k, group in enumerate(('single', 'multi')):
            rows = indexed(data['subgroups'], 'group', group)
            vals = [r[field] for r in rows]
            if j == 2:
                vals = [100*(sum(r['hybrid_correct_n_by_draw'])-5*r['original_correct_default_n'])/(5*r['n']) for r in rows]
            axes[j].barh(np.arange(6)+(k-.5)*.32, vals, height=.29,
                         color=COLORS[group], label=group.capitalize())
        style(axes[j], title, label, lim, names=j == 0, signed=j == 2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.59, .99))
    save(fig, 'medmcqa_choice_types')

def native(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    fig.subplots_adjust(left=.22, right=.985, bottom=.18, top=.78, wspace=.25)
    for j, (field, title, label, lim) in enumerate([
        ('default_accuracy_change_pp', 'A  Default accuracy', r'$\Delta$Accuracy (pp)', (-12, 18)),
        ('consistency_percent_change_pp', 'B  Semantic consistency', r'$\Delta$Consistency (pp)', (-32, 48))]):
        for k, ds in enumerate(('medqa', 'medmcqa')):
            rows = indexed(data['templates'], 'dataset', ds)
            axes[j].barh(np.arange(6)+(k-.5)*.32, [r[field] for r in rows], height=.29,
                         color=(COLORS['single'], COLORS['multi'])[k], label=('MedQA', 'MedMCQA')[k])
        style(axes[j], title, label, lim, names=j == 0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.59, .99))
    save(fig, 'native_format_sensitivity')
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    fig.subplots_adjust(left=.22, right=.985, bottom=.18, top=.78, wspace=.25)
    for j, ds in enumerate(('medqa', 'medmcqa')):
        rows = indexed(data['templates'], 'dataset', ds)
        for k, (field, label, color) in enumerate([
            ('hybrid_minus_default_raw_pp', 'Raw text', COLORS['default']),
            ('hybrid_minus_default_new_pp', 'Native chat', COLORS['pride'])]):
            vals = [r[field] for r in rows]
            if k == 0:
                vals = [r['exact_pride_delta_accuracy_pp'] for r in indexed(data['medical'], 'benchmark', ds)]
            axes[j].barh(np.arange(6)+(k-.5)*.32, vals, height=.29, color=color, label=label)
        style(axes[j], ('A  MedQA', 'B  MedMCQA')[j], r'$\Delta$Accuracy within format (pp)', (-.9, 5.2), names=j == 0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.59, .99))
    save(fig, 'native_pride_effects')

def transfer_rstd(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    fig.subplots_adjust(left=.22, right=.985, bottom=.15, top=.77, wspace=.25)
    for j, source in enumerate(('medqa', 'medmcqa')):
        rows = indexed(data['transfers'], 'source', source)
        for k, (key, label, color) in enumerate([
            ('default', 'Default', COLORS['default']), ('cpa', 'CPA', COLORS['cpa']),
            ('in_domain', 'In-domain PriDe', COLORS['pride']), ('transfer', 'Transferred PriDe', COLORS['transfer'])]):
            axes[j].barh(np.arange(6)+(k-1.5)*.18, [r['metrics'][key]['rstd'] for r in rows],
                         height=.165, color=color, label=label)
        style(axes[j], ('A  MedQA → MedMCQA', 'B  MedMCQA → MedQA')[j],
              'Original RStd (pp)', (0, 28), names=j == 0, signed=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.59, 1.0))
    save(fig, 'transfer_original_rstd')

def switches(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
    fig.subplots_adjust(left=.22, right=.985, bottom=.17, top=.78, wspace=.25)
    for j, source in enumerate(('medqa', 'medmcqa')):
        rows = indexed(data['transfers'], 'source', source)
        left = np.zeros(6)
        for key, label, color in [('gain', 'Wrong → correct', COLORS['cpa']),
                                 ('loss', 'Correct → wrong', COLORS['pride']),
                                 ('both_wrong', 'Wrong → wrong', COLORS['default'])]:
            vals = np.array([r['counts'][key] for r in rows])
            axes[j].barh(range(6), vals, left=left, height=.58, color=color, label=label)
            for y, v, l in zip(range(6), vals, left):
                if v >= 9:
                    axes[j].text(l+v/2, y, str(v), ha='center', va='center', color='white', fontsize=7)
            left += vals
        style(axes[j], ('A  MedQA → MedMCQA', 'B  MedMCQA → MedQA')[j],
              'Changed predictions (count)', (0, 530 if j == 0 else 130), names=j == 0, signed=False)
        for y, n in enumerate(left):
            axes[j].text(n+6, y, str(int(n)), va='center', fontsize=7.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.54, .99), columnspacing=1.0, handlelength=1.3)
    save(fig, 'transfer_prediction_changes')

def calibration(data):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4))
    fig.subplots_adjust(left=.22, right=.985, bottom=.30, top=.795, wspace=.25)
    for j in range(2):
        for k, source in enumerate(('medqa', 'medmcqa')):
            color = (COLORS['single'], COLORS['multi'])[k]
            y = np.arange(6)+(k-.5)*.29
            if j == 0:
                rows = indexed(data['equal_budget'], 'source', source)
                a = [r['primary_transfer_minus_in_domain_accuracy_pp'] for r in rows]
                b = [r['transfer_minus_in_domain_accuracy_pp'] for r in rows]
                for yy, aa, bb in zip(y, a, b):
                    axes[j].plot([aa, bb], [yy, yy], color=color, linewidth=1)
                axes[j].scatter(a, y, marker='s', s=25, facecolor='white', edgecolor=color, zorder=4)
                axes[j].scatter(b, y, marker='o', s=15, color=color, zorder=5)
            else:
                vals = [[r['transfer_minus_in_domain_accuracy_pp'] for r in data['resampling']
                         if r['model'] == m and r['source'] == source] for m in MODELS]
                assert all(len(v) == 5 for v in vals)
                for yy, v in zip(y, vals):
                    axes[j].scatter(v, yy+np.linspace(-.045, .045, 5), s=8, color=color, alpha=.4, zorder=3)
                axes[j].errorbar(np.mean(vals, axis=1), y, xerr=np.std(vals, axis=1, ddof=1),
                                 fmt='o', color=color, markersize=3.5, capsize=2, linewidth=1, zorder=4)
        style(axes[j], ('A  Equal source budget', 'B  Calibration resampling')[j],
              'Transferred PriDe − In-domain PriDe\naccuracy difference (pp)', (-2.35, .95), names=j == 0)
    fig.legend(handles=[Patch(color=COLORS['single'], label='MedQA → MedMCQA'),
                        Patch(color=COLORS['multi'], label='MedMCQA → MedQA')],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.59, .99))
    fig.text(.25, .085, 'A: open square = primary; filled circle = source 63', fontsize=7.3)
    fig.text(.25, .04, 'B: mean ± sample standard deviation; small points = five draws', fontsize=7.3)
    save(fig, 'calibration_sensitivity')

def compact_existing_priors(data):
    """Reuse the original prior values and comparison in a compact 2×3 layout."""
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.0), sharex=True, sharey=True)
    for ax, row, name in zip(axes.ravel(), indexed(data['transfers'], 'source', 'medqa'), NAMES):
        x = np.arange(4)
        ax.bar(x-.18, 100*np.array(row['source_prior']), width=.34,
               color=COLORS['cpa'], label='MedQA (63)')
        ax.bar(x+.18, 100*np.array(row['target_prior']), width=.34,
               color=COLORS['pride'], label='MedMCQA (197)')
        ax.axhline(25, linestyle=':', color='#555555', linewidth=.8)
        ax.set_title(name, fontsize=8.2, pad=5)
        ax.set_xticks(x, list('ABCD'))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylim(0, 100)
        ax.grid(axis='y', color='#E3E5E7', linewidth=.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('Estimated prior (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(.55, .99))
    fig.subplots_adjust(left=.105, right=.99, bottom=.09, top=.81, wspace=.22, hspace=.48)
    save(fig, 'medical_priors')

def main():
    global FIGURE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=HERE,
                        help='Fresh figure directory; existing figure outputs are never overwritten.')
    args = parser.parse_args()
    FIGURE_OUTPUT = args.output_dir.resolve()
    names = ['medical_medqa_effects', 'medical_medmcqa_effects', 'medmcqa_choice_types',
             'native_format_sensitivity', 'native_pride_effects', 'transfer_original_rstd',
             'transfer_prediction_changes', 'calibration_sensitivity', 'medical_priors']
    for name in names:
        for ext in ('pdf', 'svg'):
            target = FIGURE_OUTPUT / f'{name}.{ext}'
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}; choose a fresh --output-dir.')
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = HERE / 'chapter5_data'
    folder.mkdir(exist_ok=True)
    path = folder / 'plot_data.json'
    if args.refresh_data or not path.exists():
        data = extract()
        path.write_text(json.dumps(data, indent=2) + '\n')
    else:
        data = json.loads(path.read_text())
    for ds in ('medqa', 'medmcqa'):
        medical(data, ds)
    choice_types(data)
    native(data)
    transfer_rstd(data)
    switches(data)
    calibration(data)
    compact_existing_priors(data)
    print(f'Generated nine vector figures from saved scores in {FIGURE_OUTPUT}.')

if __name__ == '__main__':
    main()
