"""Final stage of the thesis table pipeline.

Run the generators first, in this order, then this script:
  1. build_phase4_thesis_tables.py --output analysis/outputs/phase4_20261018_20261018T091030Z
  2. restructure_rq1_presentation.py
  3. build_strengthening_tables.py
  4. build_phase7_robustness_assets.py
  5. finalize_thesis_tables.py --input <generator output dir> --output <final dir>

Stage 0 applies caption edits made before 3 November 2026. Stage A repeats the structural table revision of 3 November 2026
(analysis/outputs/revision_20261103T113140Z/revise_tables_appendix.py,
table part only) on the freshly generated files. Stage B applies the approved
caption and note revisions made afterwards (thesis_table_text_revisions.json).
Stage C copies hand-maintained tables without a generator
(analysis/thesis_tables_static). With --check DIR the result is compared byte
for byte with DIR, normally Paper/generated. No numerical cell is computed
here except those already computed by the 3 November revision from
Paper/figures/chapter5_data/plot_data.json and the published Zheng CSV.
"""
from pathlib import Path
import argparse,json,re,csv,shutil,sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--input',type=Path,required=True,help='directory with fresh generator output')
ap.add_argument('--output',type=Path,required=True,help='directory for the final tables')
ap.add_argument('--check',type=Path,help='compare the result with this directory')
ap.add_argument('--skip-text-revisions',action='store_true',help=argparse.SUPPRESS)
args=ap.parse_args()
SRC=args.input.resolve();DST=args.output.resolve()
if DST==SRC: sys.exit('--output must differ from --input')
DST.mkdir(parents=True,exist_ok=True)
for p in SRC.glob('*.tex'): shutil.copyfile(p,DST/p.name)
REVISIONS=json.loads((HERE/'thesis_table_text_revisions.json').read_text()) if not args.skip_text_revisions else []
def apply_revisions(stage):
    for r in REVISIONS:
        if r['stage']!=stage: continue
        p=DST/r['file'];t=p.read_text();n=t.count(r['old'])
        assert n==1,(r['file'],n,r['old'][:120])
        p.write_text(t.replace(r['old'],r['new']))

# ---- Stage 0: caption edits made before 3 November 2026 ----
apply_revisions('before_structural')

# ---- Stage A: structural revision of 3 November 2026 (verbatim logic) ----
DATA=json.loads((ROOT/'Paper/figures/chapter5_data/plot_data.json').read_text())
def original(name):
    assert name.startswith('generated/'),name
    return (DST/name.split('/',1)[1]).read_text()
def save(name,s):
    assert name.startswith('generated/'),name
    (DST/name.split('/',1)[1]).write_text(s)
def replace(s,old,new):
    assert old in s,old[:150]
    return s.replace(old,new)
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf']
NAMES=['Llama 3.1 8B','Qwen2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen3 8B','GLM 4 9B']

# Structural table edits are made from existing cells or saved full-precision values.
refs=[]
for benchmark in ['mmlu','arc','csqa']:
    rows=[line for line in original(f'generated/results_{benchmark}.tex').splitlines() if any(line.startswith(n+' &') for n in NAMES)]
    assert len(rows)==6
    refs.append([[cell.strip() for cell in row.rstrip('\\').split('&')[1:]] for row in rows])
table=r'''% Structural revision: existing result cells copied exactly; no analysis rerun.
\begin{table}[htbp]
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{Reference-benchmark accuracy (A, \%) and original RStd (R, pp). Each benchmark reports CPA and five-draw mean PriDe (5\%) on its full retained population; Default appears in Table~\ref{tab:rq1-default-overview}.}
\label{tab:results-reference-combined}
\begin{tabular}{l*{12}{r}}
\toprule
 & \multicolumn{4}{c}{MMLU} & \multicolumn{4}{c}{ARC-Challenge} & \multicolumn{4}{c}{CommonsenseQA}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
 & \multicolumn{2}{c}{CPA} & \multicolumn{2}{c}{PriDe (5\%)} & \multicolumn{2}{c}{CPA} & \multicolumn{2}{c}{PriDe (5\%)} & \multicolumn{2}{c}{CPA} & \multicolumn{2}{c}{PriDe (5\%)}\\
Model & A & R & A & R & A & R & A & R & A & R & A & R\\
\midrule
'''
for n,name in enumerate(NAMES):
    table+=name+' & '+' & '.join(v for benchmark in refs for v in benchmark[n])+r'\\'+'\n'
table+=r'''\bottomrule
\end{tabular}
\end{table}
'''
save('generated/results_reference_combined.tex',table)

for ds,display,n in [('medqa','MedQA','1,273'),('medmcqa','MedMCQA','3,941')]:
    table=r'''% Structural revision from plot_data.json medical full-precision endpoints.
\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{4pt}
'''+f'\\caption{{{display} ($N={n}$), raw-text zero-shot scoring. Accuracy (A) is percent and original RStd (R) is pp. All methods use full $D$; PriDe (5\\%) is the five-draw hybrid mean.}}\n'+f'\\label{{tab:results-pride-{ds}}}\n'+r'''\begin{tabular}{lrrrrrr}
\toprule
Model & \multicolumn{2}{c}{Default} & \multicolumn{2}{c}{CPA} & \multicolumn{2}{c}{PriDe (5\%)}\\
 & A & R & A & R & A & R\\
\midrule
'''
    for model,name in zip(MODELS,NAMES):
        row=next(x for x in DATA['medical'] if x['model']==model and x['benchmark']==ds)
        cols=['default_accuracy_percent','default_rstd_pp','cpa_accuracy_percent','cpa_rstd_pp','pride_hybrid_accuracy_percent','pride_hybrid_rstd_pp']
        table+=name+' & '+' & '.join(f'{row[k]:.2f}' for k in cols)+r'\\'+'\n'
    table+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    save(f'generated/results_{ds}.tex',table)

t=original('generated/strengthening_native_chat.tex')
t=replace(t, r'\centering\small\setlength{\tabcolsep}{4pt}',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}')
start=t.index('\\caption{')
end=t.index('\n\\label',start)
t=t[:start]+r'''\caption{Native-template medical scoring on the same 1,273 MedQA and 3,941 MedMCQA questions. A and C are native Default accuracy and semantic consistency (\%); $\Delta$ is native minus raw (pp). $R_{\rm cyc}$ is native Default cyclic RStd, and $R_D$/$R_H$ are native Default/hybrid original RStd (pp). H$_r$ and H$_n$ give PriDe (5\%) minus Default accuracy under raw and native input (pp), with identical calibration IDs and format-specific priors. Qwen3 uses its non-thinking template.}'''+t[end:]
t=t.replace(r'\begin{tabular}{lrrrrrrr}',r'\begin{tabular}{lrrrrrrrrr}')
t=t.replace(r'Model & A & $\Delta A$ & C & $\Delta C$ & R & H$_r$ & H$_n$',r'Model & A & $\Delta A$ & C & $\Delta C$ & $R_{\rm cyc}$ & $R_D$ & $R_H$ & H$_r$ & H$_n$')
t=t.replace(r'\multicolumn{8}',r'\multicolumn{10}')
lines=t.splitlines(); ds=None
native_values=[]
for k,line in enumerate(lines):
    if r'\textit{MedQA}' in line: ds='medqa'
    elif r'\textit{MedMCQA}' in line: ds='medmcqa'
    if ds and any(line.startswith(name+' &') for name in NAMES):
        model=MODELS[NAMES.index(line.split(' &')[0])]
        row=next(x for x in DATA['templates'] if x['model']==model and x['dataset']==ds)
        cells=line.rstrip('\\').split(' & ')
        cells[6:6]=[f"{row['default_rstd_new']:.2f}",f"{row['pride_rstd_new']:.2f}"]
        lines[k]=' & '.join(cells)+r'\\'
        native_values.append(dict(model=model,dataset=ds,default_rstd=row['default_rstd_new'],hybrid_rstd=row['pride_rstd_new']))
save('generated/strengthening_native_chat.tex','\n'.join(lines)+'\n')

t=original('generated/strengthening_transfer.tex')
t=replace(t,r'\centering\small\setlength{\tabcolsep}{4pt}',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}')
start=t.index('\\caption{');end=t.index('\n\\label',start)
t=t[:start]+r'''\caption{Transferred PriDe relative to In-domain PriDe on the primary holdouts. TV$_{ST}$ is the distance between priors; TV$_{Q,U}$ and TV$_{M,U}$ are their distances to uniform for MedQA and MedMCQA (probability units). Switches are percentages; G/L counts wrong-to-correct/correct-to-wrong changes. F: MedQA to MedMCQA, $n=3,744$; Rev.: MedMCQA to MedQA, $n=1,210$.}'''+t[end:]
t=t.replace(r'\begin{tabular}{lrrrrr}',r'\begin{tabular}{lrrrrrrr}')
t=t.replace('Model & TV & Switches F & G/L F & Switches R & G/L R',r'Model & TV$_{ST}$ & TV$_{Q,U}$ & TV$_{M,U}$ & Switches F & G/L F & Switches Rev. & G/L Rev.')
lines=t.splitlines()
for k,line in enumerate(lines):
    if any(line.startswith(name+' &') for name in NAMES):
        model=MODELS[NAMES.index(line.split(' &')[0])]
        row=next(x for x in DATA['transfers'] if x['model']==model and x['source']=='medqa')
        def tv(v): return .5*sum(abs(x/sum(v)-1/len(v)) for x in v)
        cells=line.rstrip('\\').split(' & ')
        cells[2:2]=[f"{tv(row['source_prior']):.3f}",f"{tv(row['target_prior']):.3f}"]
        lines[k]=' & '.join(cells)+r'\\'
save('generated/strengthening_transfer.tex','\n'.join(lines)+'\n')

# Published panel summaries: retain original twenty-model values and add the
# requested nineteen-open-model Default means in a table note.
rows=list(csv.DictReader((ROOT/'analysis/outputs/rq1_restructure_20261021/sources/published_table3_0shot.csv').open()))
open_means=[]
for ds in ['mmlu','arc','csqa']:
    group=[x for x in rows if x['benchmark']==ds and 'gpt' not in x['model'].lower()]
    assert len(group)==19
    open_means.append(dict(benchmark=ds,accuracy=sum(float(x['default_accuracy_percent']) for x in group)/19,rstd=sum(float(x['default_rstd_pp']) for x in group)/19))
note=r'''\par\vspace{3pt}\begin{minipage}{.98\linewidth}\footnotesize
The Zheng panel includes GPT-3.5, evaluated differently. Excluding it, the nineteen-model open panel has Default mean A/R of '''+', '.join(f"{r['accuracy']:.2f}/{r['rstd']:.2f}" for r in open_means)+r''' for MMLU, ARC-Challenge and CommonsenseQA, respectively; these are calculated from the retained published rounded cells.
\end{minipage}
'''
t=original('generated/rq1_default_panel_comparison.tex')
t=t.replace('\\end{tabular}\n','\\end{tabular}\n'+note)
save('generated/rq1_default_panel_comparison.tex',t)
t=original('generated/rq1_zheng_comparison.tex')
t=replace(t,r"Table~\ref{tab:rq1-published-changes} retains the separately reported $\Delta$ Average values.",r'The separately published $\Delta$ Average values are retained in the note below.')
note=r'''\par\vspace{3pt}\begin{minipage}{.98\linewidth}\footnotesize
Published $\Delta$ Average cells (CPA; PriDe (5\%)), each pair $\Delta A/\Delta R$ in pp: MMLU $+4.9/-8.7$; $+1.2/-7.6$; ARC-Challenge $+5.7/-8.6$; $+1.3/-5.6$; CommonsenseQA $+7.9/-11.2$; $+1.7/-6.9$. Their original one-decimal precision is retained; recalculated changes above can differ at rounding boundaries.
\end{minipage}
'''
t=t.replace('\\end{tabular}\n','\\end{tabular}\n'+note)
save('generated/rq1_zheng_comparison.tex',t)

# Supporting appendix: remove only the specifically redundant table/columns.
b=original('generated/robustness_appendix.tex')
b=replace(b,r'$D$, $I$, $T$ and $C$ denote Default, in-domain PriDe, transferred PriDe and CPA.',r'$D$, $I$ and $T$ denote Default, In-domain PriDe and Transferred PriDe; CPA is written in full.')
start=b.index('\\begin{table}[H]')
end=b.index('\\end{table}',start)+len('\\end{table}')
assert 'tab:eq-endpoints-medqa' in b[start:end]
b=b[:start]+r'Forward equal-budget endpoints are identical to Table~\ref{tab:transfer-medqa} by construction.'+b[end:]
start=b.index('\\begin{table}[H]')
end=b.index('\\end{table}',start)+len('\\end{table}')
block=b[start:end]
assert 'tab:eq-contrasts-medqa' in block
block=replace(block,'equal-budget contrasts and change from the primary source prior, all in pp.','equal-budget contrasts (pp). The omitted changes from the primary transferred endpoint, $\\Delta A_T$ and $\\Delta R_T$, are zero for all models.')
block=block.replace(r'\begin{tabular}{lrrrrrr}',r'\begin{tabular}{lrrrr}')
block=replace(block,r' & $\Delta A_T$ & $\Delta R_T$','')
block=block.replace(r' & 0.00 & 0.00\\',r'\\')
b=b[:start]+block+b[end:]
# $C$ remains semantic consistency everywhere except the old CPA method column.
b=b.replace(r'$D$ & $I$ & $T$ & $C$',r'$D$ & $I$ & $T$ & CPA')
b=replace(b,'C marks inclusion in both medical groups, B inclusion only in the broad group, and R exclusion from both.','Core marks inclusion in both medical groups, B inclusion only in the broad group, and Rest exclusion from both.')
start=b.index('\\label{tab:mmlu-subject-membership}')
end=b.index('\\end{longtable}',start)
block=b[start:end]
block=re.sub(r' & C(\\\\)',r' & Core\1',block)
block=re.sub(r' & R(\\\\)',r' & Rest\1',block)
b=b[:start]+block+b[end:]
b=b.replace('M denotes the medical group, R its remaining-subject complement.','M denotes the medical group, Rest its remaining-subject complement.')
b=b.replace(' & Core R &',' & Core Rest &').replace(' & Broad R &',' & Broad Rest &').replace(' & R & D &',' & Rest & D &').replace(' & R & P &',' & Rest & P &')
save('generated/robustness_appendix.tex',b)

# ---- Stage B: approved text revisions after 3 November 2026 ----
apply_revisions('after_structural')

# ---- Stage C: hand-maintained tables without a generator ----
for p in (ROOT/'analysis/thesis_tables_static').glob('*.tex'): shutil.copyfile(p,DST/p.name)

if args.check:
    ref=args.check.resolve();bad=[]
    names=sorted({p.name for p in DST.glob('*.tex')}|{p.name for p in ref.glob('*.tex')})
    for n in names:
        a,b=DST/n,ref/n
        if not (a.exists() and b.exists() and a.read_bytes()==b.read_bytes()): bad.append(n)
    print(json.dumps(dict(files=len(names),identical=len(names)-len(bad),different=bad)))
    sys.exit(1 if bad else 0)
