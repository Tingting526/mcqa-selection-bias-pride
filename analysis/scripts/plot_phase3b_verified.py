#!/usr/bin/env python3
"""Scientific figures from verified versioned results; no inference dependencies."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(output):
    source_path=output/'analysis_results.json';d=json.loads(source_path.read_text())
    assert len(d['validation'])==30 and all(r['status']=='verified_from_matching_raw_observations' for r in d['validation'])
    assert len(d['transfer_results'])==12
    directory=output/'figures';directory.mkdir(exist_ok=True)
    models=list(dict.fromkeys(r['model'] for r in d['summary_rows']))
    short=['Llama 3.1 8B','Qwen 2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen 3 8B','GLM 4 9B']
    datasets=['mmlu','arc','csqa','medqa','medmcqa']
    labels=['MMLU\nN=13,592','ARC-Challenge\nN=1,165','CommonsenseQA\nN=1,216','MedQA\nN=1,273','MedMCQA\nN=3,941']
    heat_labels=['MMLU\nN=13,592','ARC-C\nN=1,165','CSQA\nN=1,216','MedQA\nN=1,273','MedMCQA\nN=3,941']
    summary={(r['model'],r['benchmark']):r for r in d['summary_rows']}
    perm={(r['model'],r['dataset']):r for r in d['permutation_metrics']}
    files=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','pdf.fonttype':42})
    def matrix(rows,key): return np.array([[rows[m,ds][key] for ds in datasets] for m in models])
    def heat(ax,values,title,unit,cmap='viridis',vmin=None,vmax=None):
        im=ax.imshow(values,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
        ax.set_xticks(range(5),heat_labels,fontsize=9);ax.set_yticks(range(6),short)
        ax.set_title(title,pad=12)
        for i in range(6):
            for j in range(5):
                color=im.cmap(im.norm(values[i,j]));lum=.2126*color[0]+.7152*color[1]+.0722*color[2]
                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='black' if lum>.55 else 'white',fontsize=9)
        ax.figure.colorbar(im,ax=ax,pad=.025,fraction=.045,label=unit)
    def save(fig,name,title,denominator,units,note):
        fig.suptitle(title,fontsize=14,y=.985)
        if name in ('selection_bias','pride_effects','permutation_sensitivity'):
            note+='\nARC-C = ARC-Challenge; CSQA = CommonsenseQA.'
        fig.text(.02,.025,note,fontsize=9,va='bottom')
        fig.subplots_adjust(left=.13,right=.93,bottom=.21,top=.86,wspace=.65,hspace=.4)
        emitted=[]
        for suffix in ('png','pdf'):
            p=directory/(name+'.'+suffix)
            if p.exists(): raise FileExistsError(p)
            fig.savefig(p,dpi=190)
            emitted.append(dict(path=str(p),sha256=digest(p)))
        plt.close(fig)
        metadata=dict(title=title,protocol='original-open-clm-raw-prompt-1536',model_panel=models,populations=dict(mmlu=13592,arc=1165,csqa=1216,medqa=1273,medmcqa=3941),denominator=denominator,units=units,
            calibration='RQ1/RQ2: mean over five original path-seeded hybrid PriDe draws; floor 5%. RQ3: one seed-42 partition per dataset; MedQA 63 and MedMCQA 197 fit questions.',
            uncertainty='Point estimates; no confidence intervals or hypothesis tests. Five-draw variability is reported separately in tables.',
            source=str(source_path),source_sha256=digest(source_path),source_manifest=str(output/'source_manifest_final.json'),caveats='Descriptive benchmark contrasts. MMLU subject cue, CSQA five choices, preprocessing/populations/budgets differ; ten MMLU None options rendered nan. Medical truncation unknown.',files=emitted)
        with (directory/(name+'.metadata.json')).open('x') as stream:json.dump(metadata,stream,indent=2)
        files.extend(emitted)
    fig,axes=plt.subplots(1,2,figsize=(14,6.2))
    a=matrix(summary,'default_rstd_pp');b=matrix(perm,'cyclic_rstd');maximum=max(a.max(),b.max())
    heat(axes[0],a,'Original label-wise RStd','pp; lower = less recall imbalance',vmin=0,vmax=maximum)
    heat(axes[1],b,'Cyclic label-wise RStd','pp; lower = less recall imbalance',vmin=0,vmax=maximum)
    save(fig,'selection_bias','Selection bias across the five benchmarks','Original: N questions; cyclic: K×N displayed-label observations, K=5 for CSQA and 4 otherwise.','RStd pp','Raw-text zero-shot protocol; all six models. Cyclic RStd pools displayed-label recalls across complete shifts.\nRStd measures recall imbalance; semantic consistency is a separate endpoint. Both panels use the same color scale.')
    fig,axes=plt.subplots(1,2,figsize=(14,6.2))
    for ax,key,title in zip(axes,['pride_minus_default_accuracy_pp','pride_minus_default_rstd_pp'],['Accuracy change','RStd change']):
        v=matrix(summary,key);lim=np.max(abs(v));heat(ax,v,title,'PriDe − Default (pp)','RdBu_r',-lim,lim)
    save(fig,'pride_effects','PriDe effects across the five benchmarks','Full retained D; hybrid CPA on De plus correction on Dr; five-draw mean.','Differences pp','Raw-text zero-shot protocol. Full-D hybrid PriDe: five path-seeded draws, floor(5% N) calibration.\nPositive accuracy change is favorable; negative RStd change means less recall imbalance. No confidence intervals.')
    fig,axes=plt.subplots(1,2,figsize=(14,6.2))
    heat(axes[0],100*matrix(perm,'semantic_consistency'),'Item-level semantic consistency','% of questions',vmin=0,vmax=100)
    heat(axes[1],matrix(perm,'permutation_accuracy_range'),'Accuracy range across cyclic shifts','max − min accuracy (pp)',vmin=0)
    save(fig,'permutation_sensitivity','Sensitivity to cyclic option order','One question is one semantic-consistency unit; K complete shifts per question.','Consistency %; accuracy range pp','Raw-text zero-shot protocol; K=5 for CommonsenseQA and 4 for the other benchmarks.\nConsistency requires the same predicted answer content in every shift. CPA accuracy is not mean shift accuracy.')
    fig,axes=plt.subplots(1,2,figsize=(13,6.2))
    for ax,metric,title in zip(axes,['accuracy','rstd'],['Accuracy difference','RStd difference']):
        x=np.arange(6)
        for source,offset,color,label in [('medqa',-.18,'#24649b','MedQA → MedMCQA (Tr=3,744)'),('medmcqa',.18,'#d17b28','MedMCQA → MedQA (Tr=1,210)')]:
            vals=[next(r for r in d['transfer_results'] if r['model']==m and r['source']==source)['contrasts']['transfer_minus_in_domain'][metric] for m in models]
            ax.bar(x+offset,vals,width=.34,label=label,color=color)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(x,short,rotation=30,ha='right',fontsize=9)
        ax.set_ylabel('Transferred − in-domain PriDe (pp)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,1.03),fontsize=8,frameon=False)
    save(fig,'rq3_transfer_contrasts','Paired medical prior transfer','Identical target Tr for Default, in-domain PriDe, transfer and CPA. MedMCQA 3744; MedQA 1210.','Paired differences pp','Raw-text protocol; independent source/target calibration: MedQA 63 questions, MedMCQA 197; seed 42.\nPositive accuracy difference favors transfer; negative RStd difference means less recall imbalance. Point estimates; no CIs.')
    fig,axes=plt.subplots(2,3,figsize=(12,7.8))
    for ax,m,title in zip(axes.ravel(),models,short):
        r=next(r for r in d['transfer_results'] if r['model']==m and r['source']=='medqa')
        x=np.arange(4)
        ax.bar(x-.18,100*np.array(r['source_prior']),width=.35,color='#24649b',label='MedQA (63)')
        ax.bar(x+.18,100*np.array(r['target_prior']),width=.35,color='#d17b28',label='MedMCQA (197)')
        ax.axhline(25,color='#777777',ls=':',lw=.8);ax.set_xticks(x,list('ABCD'));ax.set_ylim(0,100);ax.set_title(title);ax.set_ylabel('Estimated prior (%)')
    axes[0,0].legend(frameon=False,fontsize=8)
    save(fig,'rq3_priors','Estimated option-ID priors for medical transfer','Mean per-question geometric prior over independent estimation subsets; no gold labels.','Prior probability %','Raw-text protocol; sorted namespaced IDs, seed 42, floor(5% N). One fitted prior per model and benchmark.\nPriors are reused in the reverse transfer direction; dotted line marks the uniform 25% reference.')
    fig,ax=plt.subplots(figsize=(10,5.8));x=np.arange(5)
    workload={r['benchmark']:r for r in d['nominal_workload']}
    for offset,key,label,color in [(-.25,'default_nominal_scorings','Default','#707070'),(0,'pride_nominal_scorings','PriDe hybrid','#24649b'),(.25,'cpa_nominal_scorings','CPA','#d17b28')]:
        vals=[workload[ds][key]/workload[ds]['n'] for ds in datasets]
        bars=ax.bar(x+offset,vals,width=.24,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylabel('Nominal scored prompts per retained question');ax.set_ylim(0,5.7);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'nominal_workload','Nominal workload for one application','Per retained question; identical formula for each of six models.','Scored prompts / question','Raw-text zero-shot protocol; PriDe uses K scores on De and one on Dr, with floor(5% N).\nNominal workload is not measured runtime, energy, total cache collection or cumulative cost of five analysis draws.')
    with (directory/'figure_manifest.json').open('x') as stream:json.dump(dict(files=files),stream,indent=2)
    with (directory/'README.md').open('x') as stream:stream.write('# Verified scientific figures\n\nSix figures are supplied as PNG and vector PDF. Each matching metadata JSON identifies protocol, populations, denominator, units, calibration, uncertainty and source SHA256. All 30 raw model–benchmark stores passed reconstruction. RQ3 uses all 12 paired medical directions. The MMLU executed-input caveat remains explicit; medical truncation is unknown. No thesis file was edited.\n')
    print(json.dumps(dict(figures=6,files=len(files))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
