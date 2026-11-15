#!/usr/bin/env python3
"""Print-sized scientific figures from frozen Phase 3B evidence and approved CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf']
SHORT=['Llama 3.1 8B','Qwen2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen3 8B','GLM 4 9B']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(out):
    verified=ROOT/'analysis/outputs/phase3b_20261018/verified';source=verified/'analysis_results.json'
    d=json.loads(source.read_text());stat=json.loads((out/'statistics_results.json').read_text())
    directory=out/'figures';directory.mkdir(exist_ok=True);active=ROOT/'Paper/figures';active.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    manifests=[]
    def save(fig,name,metadata):
        files=[]
        for suffix in ('png','pdf'):
            p=directory/(name+'.'+suffix)
            if p.exists():raise FileExistsError(p)
            fig.savefig(p,dpi=210,facecolor='white');files.append(dict(path=str(p),sha256=sha(p)))
            if suffix=='pdf':shutil.copy2(p,active/p.name)
        plt.close(fig)
        metadata.update(dict(protocol='original-open-clm-raw-prompt-1536',models=MODELS,source_analysis=str(source),source_analysis_sha256=sha(source),statistics_source=str(out/'statistics_results.json'),statistics_sha256=sha(out/'statistics_results.json'),files=files))
        with (directory/(name+'.metadata.json')).open('x') as f:json.dump(metadata,f,indent=2)
        manifests.extend(files)
    fig,axes=plt.subplots(1,2,figsize=(6.3,4.35),sharex=True,sharey=True)
    for ax,source_ds,target,title,color in [(axes[0],'medqa','medmcqa','MedQA to MedMCQA','#256598'),(axes[1],'medmcqa','medqa','MedMCQA to MedQA','#b2671d')]:
        rows=[next(r for r in stat['bootstrap_rows'] if r['model']==m and r['source']==source_ds and r['contrast']=='transfer_minus_in_domain') for m in MODELS]
        p=np.array([r['difference_pp'] for r in rows]);lo=np.array([r['ci_lower_pp'] for r in rows]);hi=np.array([r['ci_upper_pp'] for r in rows])
        ax.errorbar(p,np.arange(6),xerr=np.array([p-lo,hi-p]),fmt='o',capsize=3,color=color,ms=4,lw=1)
        ax.axvline(0,color='#777777',lw=.7);ax.set_title(title);ax.set_yticks(np.arange(6),SHORT);ax.set_xlim(-3.5,1.6);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_xlabel('Accuracy difference (pp)')
    axes[0].invert_yaxis();fig.subplots_adjust(left=.22,right=.98,top=.86,bottom=.2,wspace=.18)
    save(fig,'transfer_accuracy_ci',dict(title='Transferred minus in-domain PriDe accuracy',denominators={'medqa':1210,'medmcqa':3744},units='Paired accuracy difference pp',uncertainty='Pointwise 95% percentile intervals; 1000 shared question-bootstrap draws, seed 20261018; fixed source and target priors; no simultaneous adjustment',calibration='63 MedQA and 197 MedMCQA questions; one seed-42 partition per dataset'))
    fig,axes=plt.subplots(2,1,figsize=(6.3,7.1));datasets=['mmlu','arc','csqa','medqa','medmcqa'];labels=['MMLU','ARC-C','CSQA','MedQA','MedMCQA']
    rows={(r['model'],r['dataset']):r for r in d['permutation_metrics']}
    for ax,key,mult,title,unit,vmax in [(axes[0],'cyclic_rstd',1,'Cyclic recall imbalance','RStd (pp)',None),(axes[1],'semantic_consistency',100,'Same selected content across every cyclic shift','Questions (%)',100)]:
        values=np.array([[mult*rows[m,ds][key] for ds in datasets] for m in MODELS]);im=ax.imshow(values,cmap='viridis',vmin=0,vmax=vmax,aspect='auto')
        ax.set_xticks(range(5),labels);ax.set_yticks(range(6),SHORT);ax.set_title(title,pad=9)
        for i in range(6):
            for j in range(5):
                col=im.cmap(im.norm(values[i,j]));lum=.2126*col[0]+.7152*col[1]+.0722*col[2]
                ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8.5,color='black' if lum>.55 else 'white')
        fig.colorbar(im,ax=ax,fraction=.045,pad=.025,label=unit)
    fig.subplots_adjust(left=.24,right=.91,top=.95,bottom=.065,hspace=.34)
    save(fig,'benchmark_robustness',dict(title='Cyclic recall and content consistency across five benchmarks',denominator='Full retained N: MMLU13592, ARC1165, CSQA1216, MedQA1273, MedMCQA3941; complete K cycles with K=5 for CSQA and4 otherwise',units='Cyclic RStd pp; semantic consistency %',uncertainty='Descriptive fixed-score endpoints; no intervals',calibration='Uncorrected scores; no fitted prior',caveats='Subject cue, option count, population/filter/budget differences and MMLU input conversion; no causal domain comparison'))
    fig,axes=plt.subplots(3,2,figsize=(6.3,7.25),sharex=True,sharey=True)
    for ax,m,short in zip(axes.ravel(),MODELS,SHORT):
        r=next(r for r in d['transfer_results'] if r['model']==m and r['source']=='medqa');x=np.arange(4)
        ax.bar(x-.17,100*np.array(r['source_prior']),width=.32,color='#256598',label='MedQA (63)')
        ax.bar(x+.17,100*np.array(r['target_prior']),width=.32,color='#b2671d',label='MedMCQA (197)')
        ax.axhline(25,ls=':',color='#777777',lw=.8);ax.set_title(short,pad=5);ax.set_xticks(x,list('ABCD'));ax.set_ylim(0,100)
    for ax in axes[:,0]:ax.set_ylabel('Estimated prior (%)')
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,.995),fontsize=9)
    fig.subplots_adjust(left=.13,right=.98,bottom=.06,top=.92,hspace=.44,wspace=.22)
    save(fig,'medical_priors',dict(title='Fitted medical option-ID priors',denominator='Mean of normalized per-question geometric priors;63 MedQA or197 MedMCQA fit questions',units='Estimated prior probability %',uncertainty='Single seed-42 split; no interval; no gold labels in fitting',calibration='Independent sorted namespaced source/target populations; floor5%; priors reused in reverse direction'))
    with (directory/'figure_manifest.json').open('x') as f:json.dump(dict(files=manifests),f,indent=2)
    print(json.dumps(dict(figures=3,files=6,active=str(active))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
