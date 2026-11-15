#!/usr/bin/env python3
"""Publication-sized vector/PNG figures from independently verified results."""
import json
from pathlib import Path
import sys
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf']
SHORT=['Llama 3.1 8B','Qwen2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen3 8B','GLM 4 9B']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert json.loads((OUT/'independent_validation.json').read_text())['status']=='pass'
    d=json.loads((OUT/'analysis_results.json').read_text());s=json.loads((OUT/'seed_resampling_results.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    manifests=[]
    def save(fig,name,metadata):
        files=[]
        for suffix in ['svg','png']:
            p=OUT/'figures'/(name+'.'+suffix)
            if p.exists() and '--replace-after-visual-review' not in sys.argv:raise FileExistsError(p)
            fig.savefig(p,dpi=230,facecolor='white');files.append(dict(path=str(p),sha256=sha(p)))
        plt.close(fig)
        metadata.update(files=files,source_analysis_sha256=sha(OUT/'analysis_results.json'),seed_results_sha256=sha(OUT/'seed_resampling_results.json'),independent_validation_sha256=sha(OUT/'independent_validation.json'),source_version='phase6-robustness-v1',models=MODELS)
        with (OUT/'figures'/(name+'.metadata.json')).open('w' if '--replace-after-visual-review' in sys.argv else 'x') as f:json.dump(metadata,f,indent=2)
        manifests.append(metadata)
    fig,axes=plt.subplots(1,2,figsize=(6.3,4.2),sharex=True,sharey=True)
    for ax,source,title in zip(axes,['medqa','medmcqa'],['MedQA → MedMCQA','MedMCQA → MedQA']):
        rows=[next(r for r in d['equal_budget'] if r['model']==m and r['source']==source) for m in MODELS];y=np.arange(6)
        primary=np.array([r['primary_transfer_minus_in_domain_accuracy_pp'] for r in rows]);equal=np.array([r['transfer_minus_in_domain_accuracy_pp'] for r in rows])
        for i in range(6):ax.plot([primary[i],equal[i]],[y[i],y[i]],color='#ABB5BE',lw=1)
        ax.scatter(primary,y-.06,label='Primary (63 / 197)',marker='o',facecolor='white',edgecolor='#256598',s=28,zorder=3)
        ax.scatter(equal,y+.06,label='Source budget 63 / 63',marker='x',color='#B2671D',s=28,zorder=3)
        ax.axvline(0,color='#777777',lw=.7);ax.set_title(title);ax.set_yticks(y,SHORT);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_xlabel('Transfer − in-domain accuracy (pp)')
    axes[0].invert_yaxis();axes[0].set_xlim(-2.1,1);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.59,.005),fontsize=8)
    fig.subplots_adjust(left=.24,right=.98,top=.88,bottom=.28,wspace=.19)
    save(fig,'equal_budget_transfer',dict(title='Equal source calibration budgets',source_fit_n={'primary_medqa':63,'primary_medmcqa':197,'equal_medqa':63,'equal_medmcqa':63},target_fit_n={'medqa':63,'medmcqa':197},evaluation_n={'medqa':1210,'medmcqa':3744},evaluation_population='fixed primary seed42 target holdouts',units='accuracy difference pp',calibration_policy='Source-only nested63; original target priors unchanged. Forward condition unchanged by design.',uncertainty='Point estimates; no new intervals.'))
    fig,axes=plt.subplots(1,2,figsize=(6.3,4.7),sharex=True,sharey=True)
    for ax,source,title,color in zip(axes,['medqa','medmcqa'],['MedQA → MedMCQA','MedMCQA → MedQA'],['#256598','#B2671D']):
        rows=[next(r for r in s['summaries'] if r['model']==m and r['source']==source) for m in MODELS];key='transfer_minus_in_domain_accuracy_pp';y=np.arange(6)
        means=np.array([r[key+'_seed_mean'] for r in rows]);sd=np.array([r[key+'_seed_sample_sd'] for r in rows]);lo=np.array([r[key+'_seed_min'] for r in rows]);hi=np.array([r[key+'_seed_max'] for r in rows])
        for i in range(6):ax.plot([lo[i],hi[i]],[i,i],color=color,lw=1,alpha=.35)
        ax.errorbar(means,y,xerr=sd,fmt='o',capsize=3,color=color,ms=4,lw=1.4,label='Mean ± sample SD')
        for j,seed in enumerate([42,43,44,45,46]):
            values=[next(r[key] for r in s['rows'] if r['model']==m and r['source']==source and r['seed']==seed) for m in MODELS]
            ax.scatter(values,y+(j-2)*.055,s=12,marker='.',color=color,alpha=.65)
        primary=[next(r['primary_'+key] for r in s['rows'] if r['model']==m and r['source']==source) for m in MODELS]
        ax.scatter(primary,y,marker='x',color='#333333',s=27,label='Primary fixed prior',zorder=4)
        ax.axvline(0,color='#777777',lw=.7);ax.set_title(title);ax.set_yticks(y,SHORT);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Transfer − in-domain accuracy (pp)')
    axes[0].invert_yaxis();axes[0].set_xlim(-2.25,1);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.6,.045),fontsize=8)
    fig.text(.61,.018,'Five fixed-pool draws with replacement; SD is not a CI.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.24,right=.98,top=.89,bottom=.26,wspace=.19)
    save(fig,'calibration_resampling',dict(title='Conditional calibration selection sensitivity',seeds=[42,43,44,45,46],source_and_target_fit_n={'medqa':63,'medmcqa':197},evaluation_n={'medqa':1210,'medmcqa':3744},evaluation_population='fixed primary seed42 target holdouts',calibration_policy='Sample with replacement within original fit pools; both priors refitted; same draws across models',units='accuracy difference pp',uncertainty='Five-draw mean and sample SD ddof1, thin full range, individual draw dots; no confidence interval. Seed42 resample differs from primary.'))
    fig,axes=plt.subplots(2,1,figsize=(6.3,6.3));datasets=['mmlu','arc','csqa','medqa','medmcqa'];labels=['MMLU','ARC-C','CSQA','MedQA','MedMCQA']
    plotted={}
    for ax,key,title in [(axes[0],'delta_cyclic_rstd_pp','Change in cyclic RStd (lower is better)'),(axes[1],'delta_semantic_consistency_pp','Change in content consistency (higher is better)')]:
        values=np.array([[next(r[key] for r in d['corrected_permutation'] if r['model']==m and r['benchmark']==ds) for ds in datasets] for m in MODELS]);lim=np.abs(values).max();im=ax.imshow(values,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks(range(5),labels);ax.set_yticks(range(6),SHORT);ax.set_title(title,pad=8)
        for i in range(6):
            for j in range(5):
                color=im.cmap(im.norm(values[i,j]));lum=.2126*color[0]+.7152*color[1]+.0722*color[2]
                ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='black' if lum>.55 else 'white')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='PriDe − Default (pp)');plotted[key]=values.tolist()
    fig.subplots_adjust(left=.24,right=.91,top=.95,bottom=.065,hspace=.4)
    save(fig,'corrected_permutation_changes',dict(title='Frozen-prior correction on complete heldout cycles',evaluation_n={'mmlu':12913,'arc':1107,'csqa':1156,'medqa':1210,'medmcqa':3744},calibration_policy='First saved original path-seeded draw per model/benchmark; prior fitted only outside evaluation; same prior every shift',evaluation_population='complement of first original hybrid calibration draw, not full population and not RQ3 seed42 partitions',units='paired endpoint differences pp',cyclic_shifts={'csqa':5,'others':4},plotted_values=plotted,uncertainty='Descriptive point estimates; no interval.'))
    fig,axes=plt.subplots(1,2,figsize=(6.3,4.5),sharey=True)
    for ax,key,title in [(axes[0],'full_default_original_accuracy_percent','Original-order accuracy'),(axes[1],'full_default_semantic_consistency_percent','Content consistency')]:
        for scheme,marker,color,label in [('core5','o','#256598','Core five medical subjects'),('broad8','s','#B2671D','Broad eight medical subjects')]:
            values=[]
            for m in MODELS:
                a=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']=='medical');b=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']=='remaining_subjects');values.append(a[key]-b[key])
            ax.scatter(values,np.arange(6)+(0.06 if scheme=='core5' else -.06),marker=marker,color=color,s=24,label=label)
        ax.axvline(0,color='#777777',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_title(title);ax.set_yticks(range(6),SHORT);ax.set_xlabel('Medical − remaining MMLU (pp)',fontsize=8)
    axes[0].invert_yaxis();fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=1,frameon=False,bbox_to_anchor=(.59,.005),fontsize=8)
    fig.subplots_adjust(left=.24,right=.98,top=.88,bottom=.26,wspace=.22)
    save(fig,'mmlu_subject_context',dict(title='Descriptive MMLU grouping sensitivity',evaluation_n={'core5_medical':939,'core5_remaining':12653,'broad8_medical':1604,'broad8_remaining':11988},evaluation_population='full retained MMLU population',calibration_policy='Default only in figure; group PriDe heldout results provided separately',units='accuracy/consistency difference pp',caveat='Topic-name grouping, nonrandomized subjects; remaining group is not a pure nonmedical control; all conditions have four cyclic shifts.'))
    with (OUT/'figures/figure_manifest.json').open('w' if '--replace-after-visual-review' in sys.argv else 'x') as f:json.dump(dict(figures=manifests),f,indent=2)
    print(json.dumps(dict(figures=len(manifests),format='vector SVG + 230 dpi PNG',paper_modified=False)))

if __name__=='__main__':main()
