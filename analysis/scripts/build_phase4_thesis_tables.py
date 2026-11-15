#!/usr/bin/env python3
"""Generate traceable LaTeX tables from unchanged verified endpoints and new CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf']
SHORT=['Llama 3.1 8B','Qwen2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen3 8B','GLM 4 9B']
NAMES=dict(mmlu='MMLU',arc='ARC-Challenge',csqa='CommonsenseQA',medqa='MedQA',medmcqa='MedMCQA')

def main(output):
    verified=ROOT/'analysis/outputs/phase3b_20261018/verified';d=json.loads((verified/'analysis_results.json').read_text());s=json.loads((output/'statistics_results.json').read_text())
    directory=output/'tables/latex';directory.mkdir(parents=True,exist_ok=True);active=ROOT/'Paper/generated';active.mkdir(exist_ok=True)
    sources=[]
    def save(name,body,provenance):
        p=directory/name
        with p.open('x') as f:f.write('% Generated from verified numerical artifacts; edit the generator, not cells.\n'+body+'\n')
        destination=active/name
        if destination.exists():raise FileExistsError(destination)
        shutil.copy2(p,destination);sources.append(dict(path=str(destination),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),provenance=provenance))
    table=lambda caption,label,columns,head,rows:r'\begin{table}[htbp]'+'\n'+r'\centering\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+head+r'\\'+'\n'+r'\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'
    for ds in NAMES:
        subset=[r for r in d['summary_rows'] if r['benchmark']==ds];rows=[]
        for model,short in zip(MODELS,SHORT):
            r=next(r for r in subset if r['model']==model)
            vals=[r[k] for k in ['default_rstd_pp','default_accuracy_percent','cpa_rstd_pp','cpa_accuracy_percent','pride_hybrid_rstd_pp','pride_hybrid_accuracy_percent']]
            rows.append(short+' & '+' & '.join(f'{v:.1f}' for v in vals)+r'\\')
        caption=NAMES[ds]+f" ($N={subset[0]['n']:,}$), raw-text zero-shot protocol. Accuracy (A) in percent and original RStd (R) in pp. Default and CPA use full $D$; H is five-draw mean hybrid PriDe with 5\\% calibration."
        head=r'Model & \multicolumn{2}{c}{Default} & \multicolumn{2}{c}{CPA} & \multicolumn{2}{c}{H}\\'+'\n'+r' & R & A & R & A & R & A'
        save('results_'+ds+'.tex',table(caption,'tab:results-pride-'+ds,'lrrrrrr',head,rows),dict(source=str(verified/'analysis_results.json'),rows=[r['source_summary'] for r in subset]))
    # Long tables retain all models while keeping the main text readable.
    def long_table(caption,label,head,groups,columns='lrrrrr'):
        return r'\begingroup\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{longtable}{'+columns+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+head+r'\\\midrule'+'\n'+r'\endfirsthead'+'\n'+r'\toprule'+'\n'+head+r'\\\midrule\endhead'+'\n'+r'\bottomrule\endfoot'+'\n'+'\n'.join(groups)+'\n'+r'\end{longtable}\endgroup'
    groups=[]
    for ds in NAMES:
        groups.append(r'\multicolumn{6}{l}{\textbf{'+NAMES[ds]+r'}}\\*')
        for model,short in zip(MODELS,SHORT):
            r=next(r for r in d['permutation_metrics'] if (r['model'],r['dataset'])==(model,ds))
            vals=[r['mean_permutation_accuracy'],r['permutation_accuracy_sd'],r['permutation_accuracy_range'],r['cyclic_rstd'],100*r['semantic_consistency']]
            groups.append(short+' & '+' & '.join(f'{v:.2f}' for v in vals)+(r'\\*' if model!=MODELS[-1] else r'\\'))
    save('permutation_metrics.tex',long_table('Uncorrected complete-cycle diagnostics under the raw zero-shot protocol. Mean accuracy and content consistency (C) are percentages; accuracy SD/range and cyclic RStd are pp. Each benchmark uses its retained $N$ from Table~\\ref{tab:populations}; $K=5$ for CommonsenseQA and 4 otherwise.','tab:permutation-metrics',r'Model & Mean A & SD & Range & Cyclic R & C',groups),str(verified/'tables/verified_permutation_metrics.csv'))
    groups=[]
    for ds in NAMES:
        groups.append(r'\multicolumn{5}{l}{\textbf{'+NAMES[ds]+r'}}\\*')
        for model,short in zip(MODELS,SHORT):
            r=next(r for r in d['summary_rows'] if (r['model'],r['benchmark'])==(model,ds))
            groups.append(short+f" & {r['pride_accuracy_draw_sd_pp']:.2f} & [{r['pride_accuracy_min_percent']:.2f}, {r['pride_accuracy_max_percent']:.2f}] & {r['pride_rstd_draw_sd_pp']:.2f} & [{r['pride_rstd_min_pp']:.2f}, {r['pride_rstd_max_pp']:.2f}]"+(r'\\*' if model!=MODELS[-1] else r'\\'))
    save('calibration_variability.tex',long_table('Hybrid PriDe variability over five path-seeded calibration draws of fixed raw scores. SD uses ddof=0 and is in pp; accuracy extrema are percentages, RStd extrema pp. These are not confidence intervals.','tab:calibration-variability',r'Model & A SD & A [min,max] & R SD & R [min,max]',groups,'lrrrr'),str(verified/'tables/calibration_draws.csv'))
    for source in ['medqa','medmcqa']:
        subset=[r for r in d['transfer_results'] if r['source']==source];target=subset[0]['target'];n=subset[0]['metrics']['default']['n'];rows=[]
        for model,short in zip(MODELS,SHORT):
            r=next(r for r in subset if r['model']==model)
            vals=[r['metrics'][m][k] for m in ['default','in_domain','transfer','cpa'] for k in ['accuracy','rstd']]
            rows.append(short+' & '+' & '.join(f'{v:.2f}' for v in vals)+r'\\')
        caption=NAMES[source]+r' $\rightarrow$ '+NAMES[target]+f". Four methods on identical target $T_r$ ($n={n:,}$), raw-text protocol. A is accuracy (\\%), R original RStd (pp); I and T denote in-domain and transferred PriDe. One seed-42 split; source/target fit counts are 63/197 or 197/63."
        head=r'Model & \multicolumn{2}{c}{Default} & \multicolumn{2}{c}{I} & \multicolumn{2}{c}{T} & \multicolumn{2}{c}{CPA}\\'+'\n'+r' & A & R & A & R & A & R & A & R'
        save('transfer_'+source+'.tex',table(caption,'tab:transfer-'+source,'lrrrrrrrr',head,rows),[str(verified/'transfer'/r['model']/(source+'_to_'+target+'.json')) for r in subset])
        rows=[]
        for model,short in zip(MODELS,SHORT):
            cells=[]
            for contrast in ['in_domain_minus_default','transfer_minus_default','transfer_minus_in_domain']:
                r=next(r for r in s['bootstrap_rows'] if (r['model'],r['source'],r['contrast'])==(model,source,contrast))
                cells.append(f"${r['difference_pp']:+.2f}~[{r['ci_lower_pp']:.2f},{r['ci_upper_pp']:.2f}]$")
            rows.append(short+' & '+' & '.join(cells)+r'\\')
        caption=NAMES[source]+r' $\rightarrow$ '+NAMES[target]+f" paired accuracy differences (pp) and pointwise 95\\% percentile intervals, $n={n:,}$ target questions. I: in-domain, T: transfer, D: Default. 1,000 shared question-bootstrap draws, seed 20261018, fixed fitted priors; no prior-refit uncertainty or simultaneous-coverage adjustment."
        save('intervals_'+source+'.tex',table(caption,'tab:intervals-'+source,'lrrr',r'Model & I $-$ D & T $-$ D & T $-$ I',rows),str(output/'statistics_results.json'))
    supports=[]
    for ds in NAMES:
        r=[r for r in d['label_rows'] if r['benchmark']==ds and r['model']==MODELS[0] and r['method']=='Default']
        counts=[str(v['support_from_frozen_population']) for v in r]+(['--'] if len(r)==4 else [])
        supports.append(NAMES[ds]+' & '+' & '.join(counts)+r'\\')
    save('label_supports.tex',table('Original-order gold-label supports in the full retained populations. Supports are shared across models and methods on full $D$; RQ3 uses separately reported target-holdout supports.','tab:label-supports','lrrrrr','Benchmark & A & B & C & D & E',supports),str(verified/'tables/label_metrics.csv'))
    rows=[]
    for ds in NAMES:
        r=next(r for r in d['nominal_workload'] if r['benchmark']==ds)
        rows.append(NAMES[ds]+f" & {r['n_labels']} & {r['prior_estimation_n']} & {r['default_nominal_scorings']:,} & {r['pride_nominal_scorings']:,} & {r['cpa_nominal_scorings']:,} & {r['pride_nominal_factor']:.3f}"+r'\\')
    save('workload.tex',table('Nominal scored prompts for one full-population application. H uses CPA on the calibration subset and correction on the remainder. These are not measured runtime or total experimental collection costs.','tab:workload','lrrrrrr',r'Benchmark & $K$ & $|D_e|$ & Default & H & CPA & H/Default',rows),str(verified/'tables/nominal_workload.csv'))
    (output/'table_source_map.json').write_text(json.dumps(dict(files=sources,original_summary_cells_preserved=180,source_analysis_sha256=hashlib.sha256((verified/'analysis_results.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(latex_tables=len(sources),preserved_original_numeric_cells=180)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args().output.resolve())
