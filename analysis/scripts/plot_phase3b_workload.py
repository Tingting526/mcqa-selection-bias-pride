#!/usr/bin/env python3
"""Plot verified population-based nominal workload, without empirical placeholders."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.output/"analysis_results.json"
    data=json.loads(source.read_text())
    rows={row["benchmark"]:row for row in data["nominal_workload"]}
    order=["mmlu","arc","csqa","medqa","medmcqa"]
    labels={"mmlu":"MMLU","arc":"ARC-Challenge","csqa":"CommonsenseQA","medqa":"MedQA","medmcqa":"MedMCQA"}
    directory=args.output/"figures"; directory.mkdir(parents=True,exist_ok=True)
    files=[directory/("nominal_workload."+suffix) for suffix in ("png","pdf")]
    if any(path.exists() for path in files): raise FileExistsError("Choose a fresh versioned output")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(10,5.8))
    x=np.arange(len(order));width=.24
    default=np.ones(len(order));cpa=np.array([rows[d]["n_labels"] for d in order]);pride=np.array([rows[d]["pride_nominal_factor"] for d in order])
    for offset,values,color,label in [(-width,default,"#6b7280","Default"),(0,pride,"#007b87","PriDe hybrid"),(width,cpa,"#cf7e35","CPA")]:
        bars=ax.bar(x+offset,values,width,color=color,label=label)
        ax.bar_label(bars,labels=[f"{v:.2f}×" if v!=int(v) else f"{int(v)}×" for v in values],padding=4,fontsize=9)
    ax.set_xticks(x,[labels[d]+f"\nN={rows[d]['n']:,}" for d in order])
    ax.set_ylabel("Nominal scorings relative to Default")
    ax.set_ylim(0,5.85);ax.set_yticks(range(0,6));ax.set_axisbelow(True);ax.yaxis.grid(alpha=.18)
    ax.legend(frameon=False,ncols=3,loc="upper left")
    fig.suptitle("Nominal question-scoring workload for one application",x=.08,ha="left",fontsize=15,fontweight="bold",y=.96)
    fig.text(.08,.885,"Current raw protocol · six-model panel · full retained D · 5% prior estimation (floor)",fontsize=10,color="#404040")
    fig.text(.08,.075,"PriDe: N + (K−1) × floor(0.05N); CPA: KN. These are theoretical scoring counts.\nThey exclude total score collection, five-draw cumulative work, batching, downloads and retries; no runtime or energy claim.",fontsize=9,color="#404040",linespacing=1.5)
    fig.subplots_adjust(left=.08,right=.985,bottom=.235,top=.83)
    for path in files: fig.savefig(path,dpi=200,facecolor="white")
    plt.close(fig)
    metadata=dict(source_artifact=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol="original-open-clm-raw-prompt-1536",panel="all six current models; mathematical workload identical per benchmark",
        population_counts={d:rows[d]["n"] for d in order},denominator="full retained D",units="question-permutation scorings normalized to Default=1",
        calibration="one application; floor(.05N) estimation; five calibration draws are not cumulative runtime here",
        uncertainty="none: deterministic count formula, not an empirical estimate",matplotlib_version=matplotlib.__version__,plot_numpy_version=np.__version__,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    with (directory/"nominal_workload.metadata.json").open("x") as stream: json.dump(metadata,stream,indent=2);stream.write("\n")
    with (directory/"README.md").open("x") as stream:
        stream.write("# Phase 3B figures\n\nThe nominal-workload PNG/PDF uses verified frozen population counts and the documented one-application formula. Exact source, protocol, panel, denominators, units, calibration definition and hashes are in nominal_workload.metadata.json.\n\n")
        stream.write("Empirical RQ1/RQ2 bias figures and real RQ3 prior/transfer figures are not generated while their matching full raw observations remain unavailable. No missing empirical values are plotted as zeros, filled from summaries, or invented. The provenance-limited summary tables remain separate.\n")
    print(json.dumps(dict(files=[str(p) for p in files],empirical_result_placeholders=False)))


if __name__=="__main__": main()
