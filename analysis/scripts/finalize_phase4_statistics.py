#!/usr/bin/env python3
"""Versioned exact-count reporting and approved paired bootstrap; CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ANALYSIS))
from src.offline_phase3b import load_raw,sha256,write_json_new
from src.phase4_statistics import (METHODS,CONTRASTS,BOOTSTRAP_SEED,BOOTSTRAP_RESAMPLES,count_direction,hybrid_correct_counts,question_indices,bootstrap_accuracy,unsupported_label_bound)

def write_corrected_reporting(output,verified,counts):
    original=verified/'rq1_rq2_interpretation.md'
    text=original.read_text();c=counts['RQ2']
    replacement=f"**RQ2:** PriDe accuracy increases in {c.get('increase',0)}/12, decreases in {c.get('decrease',0)}/12, and is unchanged in {c.get('equal',0)}/12; RStd decreases in 11/12. These directional counts do not measure effect size or statistical evidence."
    corrected,n=re.subn(r'^\*\*RQ2:\*\*.*$',replacement,text,flags=re.MULTILINE)
    assert n==1
    with (output/'rq1_rq2_interpretation_corrected.md').open('x') as stream:
        stream.write('> Phase 4 reporting correction: direction classified by exact aggregate correct counts. Original numerical summaries and the historical report remain unchanged.\n\n'+corrected)

def main(output):
    if (output/'statistics_results.json').exists():raise FileExistsError('Use a fresh versioned output directory')
    verified=ANALYSIS/'outputs/phase3b_20261018/verified'
    data=json.loads((verified/'analysis_results.json').read_text())
    assert len(data['validation'])==30 and len(data['transfer_results'])==12
    # Verify the immutable source artifacts before using their predictions.
    manifest=json.loads((verified/'output_manifest.json').read_text())
    assert all(sha256(Path(r['path']))==r['sha256'] for r in manifest['files'])
    directions=[]
    for v in data['validation']:
        summary=json.loads(Path(v['summary_path']).read_text())
        scores,_=load_raw(ANALYSIS,Path(v['manifest_path']),summary,records_path=Path(v['observation_path']))
        default,counts=hybrid_correct_counts(scores,summary['source_path_seed']);n=len(scores.ids)
        assert abs(100*default/n-summary['default']['accuracy'])<1e-9
        assert all(abs(100*c/n-t['full']['accuracy'])<1e-9 for c,t in zip(counts,summary['trials']))
        assert abs(100*sum(counts)/(5*n)-summary['pride']['accuracy'])<1e-9
        directions.append(dict(rq=v['rq'],model=v['model'],benchmark=v['dataset'],n=n,default_correct=default,hybrid_correct_by_draw=counts,
            exact_aggregate_numerator=sum(counts)-5*default,exact_aggregate_denominator=5*n,
            direction=count_direction(default,counts),preserved_float_difference_pp=summary['pride']['accuracy']-summary['default']['accuracy'],source_summary=v['summary_path'],source_sha256=v['summary_sha256']))
    counts={rq:dict(Counter(r['direction'] for r in directions if r['rq']==rq)) for rq in ('RQ1','RQ2')}
    assert counts['RQ2']==dict(increase=7,equal=1,decrease=4)
    indices={};ids={};supports={};risk=[];bootstrap_rows=[];sample_values={}
    for target in ('medqa','medmcqa'):
        r=next(r for r in data['transfer_results'] if r['target']==target)
        rows=r['paired_predictions'];ids[target]=[p['sample_id'] for p in rows];gold=np.array([p['gold_index'] for p in rows])
        indices[target]=question_indices(len(rows),target)
        supports[target]=np.bincount(gold,minlength=4).tolist()
        record=unsupported_label_bound(supports[target]);record['target']=target
        record['observed_unsupported_draws']=int(sum(len(np.unique(gold[ix]))<4 for ix in indices[target]))
        record['rstd_intervals_computed']=False;risk.append(record)
        np.savez_compressed(output/('bootstrap_indices_'+target+'.npz'),indices=indices[target],target_ids=np.array(ids[target]))
    for r in data['transfer_results']:
        rows=r['paired_predictions'];target=r['target']
        assert [p['sample_id'] for p in rows]==ids[target]
        gold=np.array([p['gold_index'] for p in rows])
        correct=np.column_stack([[p[m+'_prediction']==p['gold_index'] for p in rows] for m in METHODS])
        point,samples,intervals=bootstrap_accuracy(correct,indices[target])
        for left,right in CONTRASTS:
            contrast=left+'_minus_'+right
            assert abs(point[contrast]-r['contrasts'][contrast]['accuracy'])<1e-9
            lo,hi=intervals[contrast]
            key=r['model']+'__'+r['source']+'_to_'+target+'__'+contrast;sample_values[key]=samples[contrast]
            bootstrap_rows.append(dict(model=r['model'],source=r['source'],target=target,contrast=contrast,n=len(rows),difference_pp=r['contrasts'][contrast]['accuracy'],ci_lower_pp=lo,ci_upper_pp=hi,
                confidence_level=.95,resamples=BOOTSTRAP_RESAMPLES,bootstrap_seed=BOOTSTRAP_SEED,target_stream=0 if target=='medqa' else 1,
                method='paired original-question bootstrap; fixed priors; pointwise percentile, linear quantiles',
                target_ids_sha256=hashlib.sha256(json.dumps(ids[target],separators=(',',':')).encode()).hexdigest(),
                sampled_index_bytes_sha256=hashlib.sha256(indices[target].tobytes()).hexdigest(),
                source_result=str(verified/'transfer'/r['model']/(r['source']+'_to_'+target+'.json'))))
    np.savez_compressed(output/'bootstrap_accuracy_samples.npz',**sample_values)
    result=dict(version='phase4-statistics-v1',source=str(verified),source_analysis_sha256=sha256(verified/'analysis_results.json'),environment=dict(python=sys.version,numpy=np.__version__),direction_policy='Compare sum of five exact hybrid correct counts with five times the exact Default count. No general floating-point tolerance changes classifications.',direction_rows=directions,direction_counts=counts,
        bootstrap=dict(seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_RESAMPLES,random_generator='NumPy PCG64 with SeedSequence([20261018,target_stream]); medqa=0, medmcqa=1',unit='original target question',paired=True,fixed_priors=True,interval='95% pointwise percentile; numpy.quantile(method=linear)',shared_indices='Same target index matrix across all four methods and all six models; each target has a separate stream',limits='Conditional on fitted priors; no calibration-refit uncertainty, family-wise coverage adjustment, p-values or causal inference'),bootstrap_rows=bootstrap_rows,rstd_support_diagnostics=risk,original_quantitative_results_changed=False)
    write_json_new(output/'statistics_results.json',result)
    write_corrected_reporting(output,verified,counts)
    policy='The RQ2 count is 7 increases, 4 decreases and 1 equality. Llama/MedMCQA has 2334 Default successes and hybrid counts [2350,2330,2324,2325,2341]; sum=11670=5×2334. The saved ~7.1e-15 pp difference is preserved as floating-point output, but does not count as an improvement. No score, prediction or original summary was changed.'
    (output/'direction_reporting_correction.md').write_text('# Exact-count direction classification\n\n'+policy+'\n\nRQ1: '+str(counts['RQ1'])+'\nRQ2: '+str(counts['RQ2'])+'\n')
    (output/'rstd_uncertainty_recommendation.md').write_text('# RStd uncertainty recommendation\n\nBoth target holdouts have positive support for all four gold labels. The recorded log10 union bounds quantify the chance that an ordinary N-question bootstrap omits any label; the 1000-draw risk remains negligible. In the actual approved question-index matrices, no draw lacks a label. Thus unsupported-label resamples are not a realistic practical obstacle for these populations.\n\nFixed-prior RStd intervals are technically feasible, but would add a nonlinear imbalance estimand beyond the approved accuracy inference. They are not necessary to report the thesis\'s descriptive RStd findings and do not resolve benchmark-composition confounding or prior-fitting uncertainty. Recommendation: retain RStd point estimates and the existing five-draw variability, with approved accuracy intervals as the primary uncertainty analysis. No RStd interval is computed. If the author wants this optional extension, approve its estimand and policy first: undefined draws must be counted and reported, with the interval explicitly unavailable if any required draw is undefined; do not drop or redraw them.\n\nExact supports, log10 risk bounds and observed unsupported-draw counts are in statistics_results.json.\n')
    print(json.dumps(dict(direction_counts=counts,paired_accuracy_intervals=len(bootstrap_rows),rstd_diagnostics=risk),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);main(a.output.resolve())
