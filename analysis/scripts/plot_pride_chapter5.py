"""Export chapter-5 plots and standalone inline PGFPlots from validated summaries.

Adaptations of Zheng et al. (2024), Figures 3 and 4. No raw observations,
legacy results, interpolation, additional alpha values, or transfer inferred.
"""
from pathlib import Path
import json
import hashlib
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'analysis/outputs/lrz-results/pride-reproduction-20261016/results'
OUT=ROOT/'figures/chapter5'
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf']
NAMES=['Llama-3.1-8B','Qwen2.5-7B','Gemma-2-9B','Mistral-7B','Qwen3-8B','GLM-4-9B']
DS=['medqa','medmcqa','mmlu','arc','csqa']
LABELS=['MedQA','MedMCQA','MMLU','ARC','CSQA']
COLORS=['#4477AA','#EE6677','#228833','#CCBB44','#AA3377']
DATA={}
HASHES={}
for model in MODELS:
 for ds in DS:
  p=SOURCE/model/ds/'result.json'; d=json.loads(p.read_text())
  assert d['implementation']=='zheng-pride-0shot-v1' and len(d['trials'])==5
  for metric in ['accuracy','rstd']:
   assert abs(statistics.mean(t['full'][metric] for t in d['trials'])-d['pride'][metric])<1e-9
  assert np.isclose(np.std(d['default']['recalls']),d['default']['rstd'])
  DATA[model,ds]=d
  HASHES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def export(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
 plt.close(fig)

fig,axs=plt.subplots(3,2,figsize=(12,10),sharey=True)
width=.145; xmax=0
for ax,model,name in zip(axs.flat,MODELS,NAMES):
 for i in range(5):
  xy=[(j,DATA[model,ds]['default']['recalls'][i]-DATA[model,ds]['default']['accuracy'])
      for j,ds in enumerate(DS) if i<DATA[model,ds]['n_labels']]
  ax.bar([j+(i-2)*width for j,y in xy],[y for j,y in xy],width,color=COLORS[i],label=chr(65+i))
 ax.axhline(0,color='#444444',lw=.7)
 ax.set_xticks(range(5),LABELS); ax.set_title(name,loc='left',fontweight='bold')
 ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
 ax.set_ylabel('Recall − accuracy (pp)')
handles,labels=axs.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,title='Answer label (E: CSQA only)',loc='upper center',ncols=5,frameon=False)
fig.tight_layout(h_pad=2,rect=[0,0,1,.94])
export(fig,'zheng_normalized_recalls_0shot')

METHODS=['default','pride','cyclic_perm']
MCOL=['#666666','#CC3311','#0077BB']; MARKERS=['o','s','^']
AGG={}
fig,axs=plt.subplots(2,5,figsize=(15,6))
for j,(ds,title) in enumerate(zip(DS,LABELS)):
 k=DATA[MODELS[0],ds]['n_labels']
 cost=statistics.mean(t['theoretical_cost_factor'] for m in MODELS for t in DATA[m,ds]['trials'])
 AGG[ds]={}
 for method,label,color,marker,x in zip(METHODS,['Default','PriDe (5%)','CPA'],MCOL,MARKERS,[1,cost,k]):
  AGG[ds][method]={'cost':x}
  for row,metric in enumerate(['rstd','accuracy']):
   value=statistics.mean(DATA[m,ds][method][metric] for m in MODELS)
   AGG[ds][method][metric]=value
   axs[row,j].scatter(x,value,c=color,marker=marker,s=55,label=label,zorder=3)
 for row in range(2):
  ax=axs[row,j]; ax.grid(alpha=.2); ax.set_xlim(.85,k+.25); ax.set_xticks([1,2,3,4] if k==4 else [1,2,3,4,5])
  if row==0: ax.set_title(title,fontweight='bold'); ax.set_ylim(bottom=0)
  else: ax.set_xlabel('Relative scoring cost')
  if j==0: ax.set_ylabel('RStd (pp) ↓' if row==0 else 'Accuracy (%) ↑')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncols=3,frameon=False)
fig.tight_layout(rect=[0,0,1,.93],h_pad=2)
export(fig,'zheng_cost_comparison_0shot')

# Inline PGFPlots avoids external graphics dependencies in the document editor.
tex=r'''
\subsection{Selection Bias by Answer Label}
\label{sec:results-medical}
Figure~\ref{fig:results-normalized-recalls} adopts the normalized-recall
representation in Figure 3 of Zheng et al.~\cite{zheng2024robust}:
$r_l-\mathrm{Acc}$ for each answer label $l$. Positive bars indicate labels
with recall above the model's overall accuracy on that benchmark; negative
bars indicate lower recall. These are percentage-point differences, not
prediction frequencies. The same label colors are used across all panels.
Only the default arrangement is shown, without pooling cyclic arrangements.

\begin{figure}[!htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 3,horizontal sep=1.1cm,vertical sep=1cm},
 width=7.8cm,height=4.1cm,ybar,/pgf/bar width=3pt,
 ymin=-25,ymax=45,ytick={-20,0,20,40},
 xmin=.5,xmax=5.5,xtick={1,2,3,4,5},
 xticklabels={MedQA,MedMCQA,MMLU,ARC,CSQA},
 tick label style={font=\scriptsize},title style={font=\small},
 ylabel style={font=\scriptsize},ylabel={Recall minus Acc (pp)},
 ymajorgrids=true,grid style={gray!20},
 axis on top,axis lines*=left,
 legend style={font=\scriptsize,draw=none,legend columns=5},
]
'''
for ix,(m,name) in enumerate(zip(MODELS,NAMES)):
 tex+=r'\nextgroupplot[title={'+name+'}'
 if ix==0: tex+=',legend to name=chfiveanswerlabels'
 tex+=']\n'
 for i,color in enumerate(COLORS):
  rgb=','.join(str(int(color[a:a+2],16)) for a in [1,3,5])
  coords=' '.join(f'({j+1},{DATA[m,ds]["default"]["recalls"][i]-DATA[m,ds]["default"]["accuracy"]:.9f})'
   for j,ds in enumerate(DS) if i<DATA[m,ds]['n_labels'])
  tex+=r'\addplot+[draw=none,fill={rgb,255:red,'+rgb.split(',')[0]+';green,'+rgb.split(',')[1]+';blue,'+rgb.split(',')[2]+'},/pgf/bar shift='+str((i-2)*3.5)+r'pt] coordinates {'+coords+'};\n'
  if ix==0: tex+=r'\addlegendentry{'+chr(65+i)+'}\n'
tex+=r'''
\end{groupplot}
\end{tikzpicture}

\ref{chfiveanswerlabels}
\caption{Default answer-label recall minus overall accuracy, following the
visual measure in Zheng et al., Figure 3. Each panel is one of the six
completed models; bars are grouped by benchmark. MedQA and MedMCQA are
extensions, and MMLU is shown in aggregate rather than by its four domains.
These data do not isolate token bias from position bias.}
\label{fig:results-normalized-recalls}
\end{figure}
\clearpage
'''
(OUT/'normalized_recalls_inline.tex').write_text(tex)
tex=r'''
\subsection{Debiasing Effectiveness and Scoring Cost}
\label{sec:results-cpa}
Figure~\ref{fig:results-cost} adapts the cost--effectiveness presentation in
Zheng et al., Figure 4~\cite{zheng2024robust}, to the evaluated conditions.
Each point is an unweighted mean over the same six completed models.
Default has relative scoring cost 1, full CPA costs $K$, and PriDe costs
$1+(K-1)\lfloor0.05N\rfloor/N$. Only the measured settings are plotted:
there are no interpolated curves, additional calibration fractions, partial
CPA conditions, or full-permutation results. The horizontal axis expresses
nominal scoring workload, not measured GPU time. The nearby Default and
PriDe points have different costs even where their markers appear close.

\begin{figure}[!htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=5 by 2,horizontal sep=.48cm,vertical sep=1cm},
 width=3.35cm,height=4.1cm,
 tick label style={font=\tiny},title style={font=\scriptsize},
 ylabel style={font=\scriptsize},xlabel style={font=\tiny},
 grid=major,grid style={gray!20},axis lines*=left,
 legend style={font=\scriptsize,draw=none,legend columns=3},
]
'''
for row,metric in enumerate(['rstd','accuracy']):
 for j,(ds,name) in enumerate(zip(DS,LABELS)):
  k=DATA[MODELS[0],ds]['n_labels']
  opts=[f'xmin=.85,xmax={k+.3}',f'xtick={{1,2,3,4'+(',5' if k==5 else '')+'}']
  if row==0:opts+=['title={'+name+'}','ymin=0']
  else:opts+=['xlabel={Relative scoring cost}']
  if j==0:opts+=['ylabel={'+('RStd (pp)' if row==0 else 'Acc (\\%)')+'}']
  if row==0 and j==0:opts+=['legend to name=chfivemethods']
  tex+=r'\nextgroupplot['+','.join(opts)+']\n'
  for method,label,color,marker in zip(METHODS,['Default','PriDe (5\\%)','CPA'],['gray','red!75!black','blue!65!black'],['*','square*','triangle*']):
   a=AGG[ds][method]
   tex+=r'\addplot[only marks,mark='+marker+',color='+color+',mark size=2pt] coordinates {'+f'({a["cost"]:.9f},{a[metric]:.9f})'+'};\n'
   if row==0 and j==0:tex+=r'\addlegendentry{'+label+'}\n'
tex+=r'''
\end{groupplot}
\end{tikzpicture}

\ref{chfivemethods}
\caption{RStd (top; lower is better) and accuracy (bottom; higher is better)
versus relative scoring cost, in the spirit of Zheng et al., Figure 4.
Each point averages six models; PriDe uses 5\% calibration and five draws
per model. This is a three-condition comparison, not the paper's complete
cost sweep. The two medical benchmarks extend the reference evaluation.}
\label{fig:results-cost}
\end{figure}
\clearpage
'''
(OUT/'cost_comparison_inline.tex').write_text(tex)
(OUT/'plot_provenance.json').write_text(json.dumps({'source_hashes':HASHES,'n_pairs':30,'models':MODELS,'datasets':DS,'reference':'https://arxiv.org/html/2309.03882v4','figure_adaptations':[3,4],'cost_aggregates':AGG},indent=2))
print('Wrote PNG/PDF plots, inline PGFPlots and provenance:',OUT)
