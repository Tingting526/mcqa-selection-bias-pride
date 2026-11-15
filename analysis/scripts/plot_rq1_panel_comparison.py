#!/usr/bin/env python3
"""Render RQ1 panel comparisons from the retained, displayed LaTeX table cells.

No inference, endpoint calculation or uncertainty estimation is performed.
The source tables retain absolute means and their original two-decimal precision.
Changes are read directly, rather than subtracted from already rounded means.
Run from any directory with Python and Matplotlib installed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "Paper/figures"
REVIEW = ROOT / "output/chapter51_revision_20261031"
BENCHMARKS = ("MMLU", "ARC-Challenge", "CommonsenseQA")
PANELS = ("Zheng", "Current")
COLORS = {"Zheng": "#555555", "Current": "#276783"}
LABELS = {"Zheng": "Zheng et al. (2024), 20 models", "Current": "Current study, 6 models"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_rows(path: Path):
    for line in path.read_text().splitlines():
        if " & " not in line or line.startswith(("Benchmark", "Model")):
            continue
        yield [cell.strip().strip("$") for cell in line.rstrip("\\").split(" & ")]


def read_tables():
    baseline_path = ROOT / "Paper/generated/rq1_default_panel_comparison.tex"
    effects_path = ROOT / "Paper/generated/rq1_zheng_comparison.tex"
    baseline, effects = [], []
    benchmark = panel = None
    for cells in table_rows(baseline_path):
        if cells[0]:
            benchmark = cells[0]
        panel = cells[1].split()[0]
        baseline.append(dict(benchmark=benchmark, panel=panel,
            mean_accuracy_percent=cells[2], mean_rstd_pp=cells[3],
            min_rstd_pp=cells[4], max_rstd_pp=cells[5]))
    benchmark = panel = None
    for cells in table_rows(effects_path):
        if cells[0]:
            benchmark = cells[0]
        if cells[1]:
            panel = cells[1].split()[0]
        reduction = cells[6][1:] if cells[6].startswith("-") else str(-float(cells[6]))
        effects.append(dict(benchmark=benchmark, panel=panel, method=cells[2],
            mean_accuracy_percent=cells[3], accuracy_improvement_pp=cells[4],
            mean_rstd_pp=cells[5], original_rstd_change_pp=cells[6],
            rstd_reduction_pp=reduction))
    assert len(baseline) == 6 and len(effects) == 12
    assert {(r["benchmark"], r["panel"]) for r in baseline} == {
        (b, p) for b in BENCHMARKS for p in PANELS}
    assert all(float(r["accuracy_improvement_pp"]) > 0 and
               float(r["rstd_reduction_pp"]) > 0 for r in effects)
    return baseline, effects, [baseline_path, effects_path]


def lookup(rows, **where):
    found = [r for r in rows if all(r[k] == value for k, value in where.items())]
    assert len(found) == 1, where
    return found[0]


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.svg")
    fig.savefig(REVIEW / f"{name}.png", dpi=220)
    # Check text bounds in the exported figure, in addition to rendered-page QA.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, artist.get_text(), box)
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (
            name, artist.get_text(), box)
    plt.close(fig)


def range_plot(rows):
    fig, ax = plt.subplots(figsize=(6.30, 2.65))
    fig.subplots_adjust(left=.245, right=.97, bottom=.22, top=.79)
    for i, benchmark in enumerate(BENCHMARKS):
        center = 4 - 2 * i
        for panel, offset in (("Zheng", .23), ("Current", -.23)):
            row = lookup(rows, benchmark=benchmark, panel=panel)
            y = center + offset
            low, high, mean = (float(row[k]) for k in
                               ("min_rstd_pp", "max_rstd_pp", "mean_rstd_pp"))
            ax.hlines(y, low, high, color=COLORS[panel], linewidth=1.8,
                      linestyles="solid" if panel == "Zheng" else "dashed")
            ax.plot([low, high], [y, y], "|", color=COLORS[panel], markersize=6)
            ax.plot(mean, y, marker="D" if panel == "Zheng" else "o",
                    color=COLORS[panel], markerfacecolor="white" if panel == "Zheng"
                    else COLORS[panel], markersize=5.3, zorder=3)
    ax.set_yticks([4, 2, 0], BENCHMARKS)
    ax.set_ylim(-.65, 4.65)
    ax.set_xlim(0, 32)
    ax.set_xticks(range(0, 31, 5))
    ax.set_xlabel("Original RStd (pp)")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0, pad=8)
    fig.legend(handles=[Line2D([], [], color=COLORS[p],
        marker="D" if p == "Zheng" else "o", markerfacecolor="white" if p == "Zheng"
        else COLORS[p], linestyle="solid" if p == "Zheng" else "dashed", label=LABELS[p])
        for p in PANELS], loc="upper center", bbox_to_anchor=(.51, 1),
        ncol=2, frameon=False, columnspacing=1.3, handlelength=2.1)
    save(fig, "rq1_default_panel_ranges")


def effects_plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 4.30), sharey=True)
    fig.subplots_adjust(left=.145, right=.97, bottom=.13, top=.83, wspace=.25)
    positions = [7.1, 6.2, 4.1, 3.2, 1.1, .2]
    pairs = [(b, m) for b in BENCHMARKS for m in ("CPA", r"PriDe (5\%)")]
    for ax, field, title, label, xmax in zip(axes,
        ("accuracy_improvement_pp", "rstd_reduction_pp"),
        ("A  Accuracy improvement", "B  Original RStd reduction"),
        ("Accuracy improvement (pp)", "Original RStd reduction (pp)"), (9.6, 13.8)):
        for y, (benchmark, method) in zip(positions, pairs):
            for panel, offset in (("Zheng", .17), ("Current", -.17)):
                value = float(lookup(rows, benchmark=benchmark, panel=panel, method=method)[field])
                ax.barh(y + offset, value, height=.27,
                        facecolor="#f5f5f5" if panel == "Zheng" else COLORS[panel],
                        edgecolor=COLORS[panel], linewidth=.5,
                        hatch="///" if panel == "Zheng" else None, zorder=3)
                ax.text(value + .13, y + offset, f"{value:.2f}",
                        va="center", fontsize=8.3, color="#222222")
        for benchmark, y in zip(BENCHMARKS, (8.05, 5.05, 2.05)):
            ax.text(.01, y, benchmark, transform=ax.get_yaxis_transform(),
                    fontsize=9.2, fontweight="medium", va="center")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-.5, 8.5)
        ax.set_yticks(positions, ["CPA", "PriDe (5%)"] * 3)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_title(title, loc="left", fontsize=9.8, pad=8)
        ax.set_xlabel(label, fontsize=9.3)
        ax.grid(axis="x")
    axes[0].set_xticks([0, 2, 4, 6, 8])
    axes[1].set_xticks([0, 3, 6, 9, 12])
    fig.legend(handles=[Patch(facecolor="#f5f5f5" if p == "Zheng" else COLORS[p], edgecolor=COLORS[p],
        hatch="///" if p == "Zheng" else None, label=LABELS[p]) for p in PANELS],
        loc="upper center", bbox_to_anchor=(.51, .995), ncol=2,
        frameon=False, columnspacing=1.3, handlelength=1.8)
    save(fig, "rq1_debiasing_panel_effects")


def main():
    FIGURES.mkdir(exist_ok=True)
    REVIEW.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.6,
        "axes.labelsize": 10.2, "xtick.labelsize": 9.2, "ytick.labelsize": 9.3,
        "legend.fontsize": 8.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .6,
        "axes.edgecolor": "#555555", "grid.color": "#dedede",
        "grid.linewidth": .55, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .45,
    })
    baseline, effects, sources = read_tables()
    for name, rows in (("baseline_figure_data.csv", baseline), ("effects_figure_data.csv", effects)):
        with (REVIEW / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    range_plot(baseline)
    effects_plot(effects)
    (REVIEW / "figure_provenance.json").write_text(json.dumps({
        "script": str(Path(__file__).relative_to(ROOT)),
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "precision": "Two-decimal displayed cells from retained source tables",
        "rstd_sign": "Plotted reduction is the negative of the source Delta R cell",
        "averaging": "Existing unweighted model-panel means, not recalculated endpoints",
        "uncertainty": "Ranges are across models, not confidence intervals. No error bars.",
        "new_inference_or_experiments": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
