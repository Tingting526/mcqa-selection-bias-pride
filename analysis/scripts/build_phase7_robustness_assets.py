#!/usr/bin/env python3
"""Generate new manuscript tables/one vector figure from frozen Phase 6 only."""
import json
from pathlib import Path
import sys
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path((ROOT/'tmp/phase7_active_output.txt').read_text().strip());P6=ROOT/'analysis/outputs/phase6_robustness_20261018'
MODELS=['llama-3.1-8b-instruct','qwen2.5-7b-instruct','gemma-2-9b-it','mistral-7b-instruct-v0.3','qwen3-8b','glm-4-9b-chat-hf'];SHORT=['Llama 3.1','Qwen2.5','Gemma 2','Mistral','Qwen3','GLM 4'];NAME=dict(zip(MODELS,SHORT))
DS=dict(mmlu='MMLU',arc='ARC-C',csqa='CSQA',medqa='MedQA',medmcqa='MedMCQA')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert json.loads((OUT/'methodology_verification.json').read_text())['status']=='pass'
    d=json.loads((P6/'analysis_results.json').read_text());s=json.loads((P6/'seed_resampling_results.json').read_text());grouping=json.loads((P6/'mmlu_subject_membership.json').read_text())
    generated=ROOT/'Paper/generated';trace=[];numeric=[]
    def number(row,field,source):
        value=row[field];numeric.append(dict(source=source,model=row.get('model'),benchmark=row.get('benchmark'),direction=row.get('source'),seed=row.get('seed'),group=row.get('group'),grouping=row.get('grouping'),field=field,value=value,display=f'{value:.2f}'));return f'{value:.2f}'
    def tab(label,caption,headers,rows,note='',long=False,colspec=None):
        cols=colspec or 'l'+'r'*(len(headers)-1)
        if long:
            text=r'{\small\setlength{\tabcolsep}{5pt}'+'\n'+r'\begin{longtable}{'+cols+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'
            head=r'\toprule'+'\n'+' & '.join(headers)+r'\\'+'\n'+r'\midrule'+'\n'
            text+=head+r'\endfirsthead'+'\n'+r'\multicolumn{'+str(len(headers))+r'}{l}{\small Table \thetable\ continued}\\'+'\n'+head+r'\endhead'+'\n'+r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{\small Continued on next page}\\\endfoot'+'\n'+r'\bottomrule\endlastfoot'+'\n'
            def endrow(row):
                keep=(label.startswith('tab:pool-') and int(row[1])<46) or (label.startswith('tab:mmlu-heldout-') and row[2]=='D')
                return r'\\*' if keep else r'\\'
            text+='\n'.join(' & '.join(map(str,row))+endrow(row) for row in rows)+'\n'+r'\end{longtable}}'+'\n'
        else:
            placement='htbp' if label=='tab:robustness-transfer-summary' else 'H'
            text=r'\begin{table}['+placement+']'+'\n'+r'\centering\small\setlength{\tabcolsep}{5pt}'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+cols+'}\n'+r'\toprule'+'\n'+' & '.join(headers)+r'\\'+'\n'+r'\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
            if note:text+=r'\par\vspace{3pt}\begin{minipage}{.98\linewidth}\footnotesize '+note+r'\end{minipage}'+'\n'
            text+=r'\end{table}'+'\n'
        trace.append(dict(label=label,headers=headers,display_rows=rows,caption=caption,note=note));return text
    rows=[]
    for source in ['medqa','medmcqa']:
        rows.append([r'\multicolumn{6}{l}{\textit{'+DS[source]+r' $\to$ '+DS['medmcqa' if source=='medqa' else 'medqa']+r'}}'])
        for m in MODELS:
            e=next(r for r in d['equal_budget'] if r['model']==m and r['source']==source);r=next(r for r in s['summaries'] if r['model']==m and r['source']==source);key='transfer_minus_in_domain_accuracy_pp'
            countkey=key+('_seed_positive_n' if source=='medqa' else '_seed_negative_n')
            rows.append([NAME[m],number(e,'primary_'+key,'analysis_results.json:equal_budget'),number(e,key,'analysis_results.json:equal_budget'),number(r,key+'_seed_mean','seed_resampling_results.json:summaries'),number(r,key+'_seed_sample_sd','seed_resampling_results.json:summaries'),f'{r[countkey]}/5'])
    text=tab('tab:robustness-transfer-summary',r'Sensitivity of transfer-minus-in-domain accuracy (pp). All methods retain the primary target holdouts: 3,744 MedMCQA or 1,210 MedQA questions.',
        ['Model','Primary','Source 63','Pool mean','Pool SD','Direction / 5'],rows,
        r'Source 63 changes only the source prior, using the nested randomized 63-question subset; target fits remain 63/197. Pool columns instead resample both original pools with replacement at 63/197 draws, seeds 42--46. SD uses $n-1$ and is not a CI. Direction / 5 counts positive forward or negative reverse contrasts among the five draws. These are separate supplementary conditions.')
    (generated/'robustness_transfer_summary.tex').write_text(text)
    appendix=[r'\FloatBarrier\subsection{Supplementary robustness protocols and results}\label{app:robustness}',
        r'The following tables supplement the primary endpoints rather than replacing them. Model abbreviations follow Table~\ref{tab:models}. $D$, $I$, $T$ and $C$ denote Default, in-domain PriDe, transferred PriDe and CPA. Accuracy and semantic consistency are percentages; RStd, SD, range and contrasts are in pp. Values are displayed to two decimals, while all direction counts use unrounded quantities. Full-precision vectors, question IDs and multiplicities remain in the electronic artifacts.']
    appendix += [r'\subsubsection{Medical transfer: exact fitting restrictions}\label{app:robustness-transfer}',
        r'The primary seed-42 fitting pools contain 63 MedQA and 197 MedMCQA questions, with fixed target complements of 1,210 and 3,744. Equal source budgets use \texttt{estimation\_order[:63]} from the saved source partition. Thus the MedMCQA source subset is nested in its original randomized pool, with no target test question added. Target in-domain fits remain unchanged. The forward prior averages the same 63 MedQA questions; its averaging order differs from the sorted primary list only by floating-point accumulation (at most $2.8\times10^{-16}$ per coordinate), with exactly identical predictions and endpoints. The reverse prior averages 63 rather than 197 MedMCQA questions.',
        r'For calibration-pool resampling, each benchmark and seed in $\{42,43,44,45,46\}$ initializes a separate Python \texttt{random.Random(seed)} generator. Starting from sorted original fit IDs, it draws 63 or 197 indices using \texttt{randrange(pool\_size)} with replacement. Each selected question contributes its normalized geometric prior once per occurrence. Both priors are refitted; the same dataset-specific draw is reused across models and transfer directions. Repeated draws are not collapsed or deduplicated. The resulting unique counts range from 36 to 43 for MedQA and 121 to 126 for MedMCQA.',
        r'The original seed-42 primary fit is a reference only; the with-replacement seed-42 draw differs from it. A full-pool, without-replacement control at all five seeds reproduces all original prior arrays and all four methods\textquotesingle{} accuracy/RStd endpoints exactly. Its zero variability follows from exhausting the available pool and is not evidence that calibration is generally stable. The informative condition measures variation conditional on one empirical fitting pool and a fixed test set. Its sample SD uses \texttt{ddof=1}; it neither estimates uncertainty over all dataset splits nor replaces the primary paired question-bootstrap intervals.']
    for source,target in [('medqa','medmcqa'),('medmcqa','medqa')]:
        eq=[next(r for r in d['equal_budget'] if r['model']==m and r['source']==source) for m in MODELS];direction=DS[source]+r' $\to$ '+DS[target]
        fields=[m+'_'+metric for m in ['default','in_domain','transfer'] for metric in ['accuracy_percent','rstd_pp']]
        appendix.append(tab('tab:eq-endpoints-'+source,direction+r': equal-source-budget endpoints. Source fit $n=63$; target fit $n='+str(eq[0]['target_calibration_n'])+r'$; target holdout $N='+str(eq[0]['evaluation_n'])+r'$.',
            ['Model','$A_D$','$R_D$','$A_I$','$R_I$','$A_T$','$R_T$'],[[NAME[r['model']],*[number(r,k,'analysis_results.json:equal_budget') for k in fields]] for r in eq],note=r'Accuracy $A$ is in percent; $R$ is original RStd in pp. The Default, in-domain and CPA endpoints are unchanged from the primary paired target comparison.'))
        fields=['transfer_minus_in_domain_accuracy_pp','transfer_minus_default_accuracy_pp','transfer_minus_in_domain_rstd_pp','transfer_minus_default_rstd_pp','change_vs_primary_transfer_accuracy_pp','change_vs_primary_transfer_rstd_pp']
        appendix.append(tab('tab:eq-contrasts-'+source,direction+r': equal-budget contrasts and change from the primary source prior, all in pp.',
            ['Model',r'$A_T-A_I$',r'$A_T-A_D$',r'$R_T-R_I$',r'$R_T-R_D$',r'$\Delta A_T$',r'$\Delta R_T$'],[[NAME[r['model']],*[number(r,k,'analysis_results.json:equal_budget') for k in fields]] for r in eq]))
        appendix.append(tab('tab:eq-prior-'+source,direction+r': option-ID prior from the 63-question source subset (probability units). Target priors remain those in Figure~\ref{fig:medical-priors}.',
            ['Model','A','B','C','D'],[[NAME[r['model']],*[f'{v:.5f}' for v in r['source_prior']]] for r in eq]))
        for metric,unit in [('accuracy','percent'),('rstd','pp')]:
            fields=[m+'_'+metric+'_'+unit for m in ['default','in_domain','transfer','cpa']]+['transfer_minus_in_domain_'+metric+'_pp','transfer_minus_default_'+metric+'_pp']
            seedrows=[next(r for r in s['rows'] if r['model']==m and r['source']==source and r['seed']==seed) for m in MODELS for seed in range(42,47)]
            appendix.append(tab('tab:pool-'+source+'-'+metric,direction+r': calibration-pool resampling '+('accuracy (percent)' if metric=='accuracy' else 'original RStd (pp)')+r'. Contrasts are pp; the fixed target holdout has $N='+str(eq[0]['evaluation_n'])+r'$.',
                ['Model','Draw',r'$D$',r'$I$',r'$T$',r'$C$',r'$T-I$',r'$T-D$'],[[NAME[r['model']],r['seed'],*[number(r,k,'seed_resampling_results.json:rows') for k in fields]] for r in seedrows],long=True,colspec='lrrrrrrr'))
    appendix += [r'\FloatBarrier\subsubsection{Frozen-prior corrected cyclic endpoints}\label{app:robustness-permutation}',
        r'Calibration uses the first existing path-seeded draw, with source-path bytes seeding the sequential shuffle as in the primary hybrid protocol. Only its complement is evaluated. A single prior is applied as $\log(p+10^{-10})-\log(\pi+10^{-10})$ in displayed-label space at every shift, followed by first-maximum selection and the permanent-content mapping. Cycles are complete: four per item, or five for CSQA. The calibration sets have no overlap with these evaluation sets. Membership may differ between models and from the medical seed-42 transfer holdouts.',
        r'In Tables~\ref{tab:corrected-mmlu}--\ref{tab:corrected-medmcqa}, $A_0$ is original-order accuracy, $R_0$ original RStd, $R_{\rm cyc}$ cyclic RStd, $\bar A$ mean per-shift accuracy, and $S$ and range describe per-shift accuracy. $C$ is all-shift content consistency. Rows $D$ and $P$ use identical heldout questions; $P$ is pure frozen-prior PriDe, not the full-population hybrid. Accuracy/consistency are percent and all dispersion quantities are pp. All models retain the previously declared first-maximum and additive-epsilon conventions. CPA is not a per-shift accuracy average; its deterministic complete-orbit invariance is an aggregation property.']
    for ds in ['mmlu','arc','csqa','medqa','medmcqa']:
        vals=[next(r for r in d['corrected_permutation'] if r['model']==m and r['benchmark']==ds) for m in MODELS];rows=[]
        fields=['original_accuracy_percent','original_rstd_pp','cyclic_rstd_pp','mean_permutation_accuracy_percent','permutation_accuracy_sd_pp','permutation_accuracy_range_pp','semantic_consistency_percent']
        for r in vals:
            for method,label in [('default','D'),('pride','P')]:rows.append([NAME[r['model']],label,*[number(r,method+'_'+k,'analysis_results.json:corrected_permutation') for k in fields]])
        appendix.append(tab('tab:corrected-'+ds,DS[ds]+r': complete heldout before/after cyclic metrics. Calibration $n='+str(vals[0]['calibration_n'])+r'$; evaluation $N='+str(vals[0]['evaluation_n'])+r'$.',
            ['Model','',r'$A_0$',r'$R_0$',r'$R_{\rm cyc}$',r'$\bar A$',r'$S$','Range',r'$C$'],rows,colspec='llrrrrrrr'))
    exceptions=json.loads((OUT/'methodology_verification.json').read_text())
    appendix.append(r'\FloatBarrier\paragraph{All exceptions to the dominant corrected pattern.} Cyclic RStd increases for Qwen2.5 on CSQA, Mistral on MedQA, and Qwen3 on ARC-C and CSQA. Consistency decreases for Mistral on MMLU, ARC-C and MedQA, and Qwen3 on ARC-C, CSQA and MedMCQA. There are no ties in either endpoint across the 30 pairs. The counts 26/30 and 24/30 use strict signs of unrounded values, not the displayed two-decimal cells; exact consistency gains minus losses give the same signs. The Qwen3 MedMCQA decrease is one question, approximately 0.027 pp. These counts carry no significance claim.')
    appendix += [r'\subsubsection{MMLU medical-subject sensitivity}\label{app:robustness-mmlu}',
        r'The core group includes exactly \path{anatomy}, \path{clinical_knowledge}, \path{college_medicine}, \path{medical_genetics} and \path{professional_medicine}. The broad group additionally includes \path{human_aging}, \path{nutrition} and \path{virology}. These topic-name definitions were stored before computing subgroup outcomes, not selected to maximize a performance contrast. They are transparent analytical classifications rather than a claim that all remaining subjects are nonmedical: related biology and psychology remain outside both definitions.',
        r'Table~\ref{tab:mmlu-subject-membership} lists all 57 names and full gold supports. C marks inclusion in both medical groups, B inclusion only in the broad group, and R exclusion from both. Thus it also specifies every excluded name. The core medical/full complement sizes are 939/12,653, with gold supports $(198,222,210,309)$ and $(2934,3167,3287,3265)$. Broad sizes are 1,604/11,988, with supports $(382,427,360,435)$ and $(2750,2962,3137,3139)$. Every printed group and subject has positive original-label support.']
    subjectrows=[r for r in d['mmlu_subjects'] if r['model']==MODELS[0] and r['grouping']=='individual_subject'];rows=[]
    for r in subjectrows:
        name=r['group'];code='C' if name in grouping['core5'] else 'B' if name in grouping['broad8'] else 'R'
        rows.append([r'\path{'+name+'}',r['group_population_n'],*r['full_default_gold_supports'],code])
    appendix.append(tab('tab:mmlu-subject-membership','All retained MMLU subjects: population size, original gold-label supports A--D and grouping membership.',
        ['Subject','$N$','A','B','C','D','Group'],rows,long=True,colspec=r'p{.49\textwidth}rrrrrl'))
    fullrows=[]
    for scheme in ['core5','broad8']:
        for m in MODELS:
            for group in ['medical','remaining_subjects']:
                r=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']==group)
                fullrows.append([NAME[m],('Core ' if scheme=='core5' else 'Broad ')+('M' if group=='medical' else 'R'),*[number(r,'full_default_'+k,'analysis_results.json:mmlu_subjects') for k in ['original_accuracy_percent','original_rstd_pp','cyclic_rstd_pp','semantic_consistency_percent']]])
    appendix.append(tab('tab:mmlu-group-default','Full-population Default by MMLU grouping. M denotes the medical group, R its remaining-subject complement. Accuracy and consistency are percent; RStd is pp.',
        ['Model','Group',r'$A_0$',r'$R_0$',r'$R_{\rm cyc}$',r'$C$'],fullrows,long=True,colspec='llrrrr'))
    appendix.append(r'The following group corrections reuse the benchmark-wide first-draw MMLU prior. They do not fit a new prior per subject or group. Their heldout counts differ across models, and their before/after denominators must not be mixed with the full-population Default table. The full electronic table contains every group support vector and all 342 model--individual-subject rows, including their full Default and disjoint-heldout corrected metrics. Small individual subjects are descriptive context rather than independent tests of a medical-domain effect.')
    for scheme in ['core5','broad8']:
        rows=[]
        for m in MODELS:
            for group in ['medical','remaining_subjects']:
                r=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']==group)
                for method,label in [('default','D'),('pride','P')]:rows.append([NAME[m],'M' if group=='medical' else 'R',label,r['group_evaluation_n'],*[number(r,method+'_'+k,'analysis_results.json:mmlu_subjects') for k in ['original_accuracy_percent','original_rstd_pp','cyclic_rstd_pp','semantic_consistency_percent']]])
        appendix.append(tab('tab:mmlu-heldout-'+scheme,('Core-five' if scheme=='core5' else 'Broad-eight')+r' MMLU grouping: identical heldout group questions before (D) and after (P) benchmark-wide prior correction. $N$ is the actual group holdout size.',
            ['Model','Group','','$N$',r'$A_0$',r'$R_0$',r'$R_{\rm cyc}$',r'$C$'],rows,long=True,colspec='lllrrrrr'))
    appendix.append(r'\FloatBarrier')
    (generated/'robustness_appendix.tex').write_text('\n\n'.join(appendix)+'\n')
    # This figure reproduces the verified two-panel Phase 6 plot, without editing it.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(6.3,6.3));datasets=['mmlu','arc','csqa','medqa','medmcqa'];plotvalues={}
    for ax,key,title in [(axes[0],'delta_cyclic_rstd_pp','Change in cyclic RStd (lower is better)'),(axes[1],'delta_semantic_consistency_pp','Change in content consistency (higher is better)')]:
        values=np.array([[next(r[key] for r in d['corrected_permutation'] if r['model']==m and r['benchmark']==ds) for ds in datasets] for m in MODELS]);lim=np.abs(values).max();im=ax.imshow(values,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(5),[DS[x] for x in datasets]);ax.set_yticks(range(6),['Llama 3.1 8B','Qwen2.5 7B','Gemma 2 9B','Mistral 7B v0.3','Qwen3 8B','GLM 4 9B']);ax.set_title(title,pad=8)
        for i in range(6):
            for j in range(5):
                col=im.cmap(im.norm(values[i,j]));lum=.2126*col[0]+.7152*col[1]+.0722*col[2];ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='black' if lum>.55 else 'white')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='PriDe - Default (pp)');plotvalues[key]=values.tolist()
    fig.subplots_adjust(left=.24,right=.91,top=.95,bottom=.065,hspace=.4)
    (OUT/'figures').mkdir(exist_ok=True);file=ROOT/'Paper/figures/corrected_permutation_robustness.pdf';fig.savefig(file,facecolor='white');fig.savefig(OUT/'figures/corrected_permutation_robustness.png',dpi=230,facecolor='white');plt.close(fig)
    metadata=dict(source_version='phase6-robustness-v1',source_path=str(P6/'analysis_results.json'),source_sha256=sha(P6/'analysis_results.json'),models=MODELS,datasets=datasets,plotted_values=plotvalues,output_path=str(file),output_sha256=sha(file))
    (OUT/'table_source_map.json').write_text(json.dumps(dict(tables=trace,numeric_cells=numeric,source_analysis_sha256=sha(P6/'analysis_results.json'),source_resampling_sha256=sha(P6/'seed_resampling_results.json')),indent=2)+'\n')
    (OUT/'figures/figure_source_map.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(tables=len(trace),numeric_cells=len(numeric),new_figures=1)))

if __name__=='__main__':main()
