#!/usr/bin/env python3
"""Five-seed fixed-holdout calibration analysis; explicit policy argument."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'analysis'))
from src.phase6_robustness import *
from src.offline_phase3b import write_json_new
OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'

def main(policy):
    primary=json.loads((ROOT/'analysis/outputs/phase3b_20261018/verified/analysis_results.json').read_text())
    rows=[];summaries=[];partitions={}
    for p in primary['transfer_results']:
        model,source,target=p['model'],p['source'],p['target']
        s=np.load(OUT/'scores'/f'{model}__{source}.npz');t=np.load(OUT/'scores'/f'{model}__{target}.npz')
        ev=p['target_partition']['heldout'];li={i:j for j,i in enumerate(t['ids'])};ix=[li[i] for i in ev]
        pairrows=[]
        for seed in SEEDS:
            sf=calibration_draw(p['source_partition'],seed,policy);tf=calibration_draw(p['target_partition'],seed,policy)
            repeated=policy=='fixed_pool_with_replacement'
            validate_partition(s['ids'].tolist(),sf,p['source_partition']['heldout'],repeated=repeated)
            validate_partition(t['ids'].tolist(),tf,ev,repeated=repeated)
            assert not set(sf)&set(t['ids'])
            sp=average_prior(s['item_priors'],s['ids'],sf,repeated=repeated);tp=average_prior(t['item_priors'],t['ids'],tf,repeated=repeated)
            metrics,_=transfer_metrics(t['probabilities'][ix],t['gold'][ix],t['cpa'][ix],sp,tp)
            r=dict(model=model,source=source,target=target,benchmark=target,seed=seed,
                calibration_policy=policy,evaluation_population='fixed primary seed-42 target holdout',evaluation_n=len(ix),
                source_calibration_n=len(sf),target_calibration_n=len(tf),source_unique_fit_n=len(set(sf)),target_unique_fit_n=len(set(tf)),
                source_fit_ids=sf,target_fit_ids=tf,source_fit_ids_sha256=ids_hash(sf),target_fit_ids_sha256=ids_hash(tf),evaluation_ids_sha256=ids_hash(ev),
                source_prior=sp.tolist(),target_prior=tp.tolist(),source_prior_l1_from_primary=float(np.abs(sp-p['source_prior']).sum()),
                target_prior_l1_from_primary=float(np.abs(tp-p['target_prior']).sum()),
                units='accuracy percent; RStd/differences/seed SD pp; prior probability',source_version=VERSION,
                raw_protocol=p['signature']['protocol'],**metrics)
            for m in ('default','in_domain','transfer','cpa'):
                for metric,unit in (('accuracy','percent'),('rstd','pp')):
                    r[f'primary_{m}_{metric}_{unit}']=p['metrics'][m][metric]
            for contrast,value in p['contrasts'].items():
                for metric,v in value.items():r[f'primary_{contrast}_{metric}_pp']=v
            r['reproduces_primary_priors']=bool(np.allclose(sp,p['source_prior'],rtol=0,atol=1e-12) and np.allclose(tp,p['target_prior'],rtol=0,atol=1e-12))
            pairrows.append(r);partitions[f'{model}__{source}_to_{target}__{seed}']=dict(source_fit_ids=sf,target_fit_ids=tf,evaluation_ids=ev)
        summary=dict(model=model,source=source,target=target,**seed_statistics(pairrows));summaries.append(summary)
        for r in pairrows:rows.append({**r,**{k:v for k,v in summary.items() if k not in ('model','source','target')}})
    filename='seed_control_results.json' if policy=='full_pool_without_replacement_control' else 'seed_resampling_results.json'
    write_json_new(OUT/filename,dict(policy=policy,rows=rows,summaries=summaries,seed42_reproduces_primary=all(r['reproduces_primary_priors'] for r in rows if r['seed']==42),
        interpretation='Conditional fixed-pool calibration resampling; sample SD ddof1 across five draws, not a confidence interval; primary question-bootstrap intervals remain unchanged' if policy=='fixed_pool_with_replacement' else 'Degenerate no-variation control: fixed holdout exhausts all remaining calibration questions; zero SD is not evidence of general calibration stability'))
    write_json_new(OUT/(filename.replace('_results.json','_partitions.json')),partitions)
    print(json.dumps(dict(policy=policy,rows=len(rows),all_primary=all(r['reproduces_primary_priors'] for r in rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['full_pool_without_replacement_control','fixed_pool_with_replacement'],required=True);main(p.parse_args().policy)
