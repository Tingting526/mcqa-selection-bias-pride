#!/usr/bin/env python3
"""Separate real-data oracle check; no transfer/decomposition/metric helpers used."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ANALYSIS))
from src.offline_phase3b import DEBIAS_SHA256, MODELS, load_raw, sha256, write_json_new


def independent_metrics(gold,prediction):
    n=len(gold)
    supports=[sum(g==label for g in gold) for label in range(4)]
    recalls=[100*sum(g==p==label for g,p in zip(gold,prediction))/supports[label] if supports[label] else float('nan') for label in range(4)]
    mean=sum(recalls)/4
    return dict(n=n,accuracy=100*sum(g==p for g,p in zip(gold,prediction))/n,supports=supports,recalls=recalls,rstd=math.sqrt(sum((x-mean)**2 for x in recalls)/4))


def check(output):
    data=json.loads((output/'analysis_results.json').read_text())
    oracle_path=ANALYSIS/'data/reference/zheng2024/upstream/code/debias_utils.py'
    assert sha256(oracle_path)==DEBIAS_SHA256
    spec=importlib.util.spec_from_file_location('pinned_zheng_oracle',oracle_path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    checked=[]
    for model in MODELS:
        stores={};parts={};averages={};cpa={}
        for dataset in ('medqa','medmcqa'):
            v=next(v for v in data['validation'] if v['model']==model and v['dataset']==dataset)
            assert v['status']=='verified_from_matching_raw_observations'
            summary=json.loads(Path(v['summary_path']).read_text())
            scores,_=load_raw(ANALYSIS,Path(v['manifest_path']),summary,records_path=Path(v['observation_path']))
            stores[dataset]=scores
            selected=sorted(random.Random(42).sample(sorted(scores.ids),max(1,int(.05*len(scores.ids)))))
            heldout=sorted(set(scores.ids)-set(selected))
            assert not set(selected)&set(heldout) and set(selected)|set(heldout)==set(scores.ids)
            parts[dataset]=(selected,heldout)
            values={identifier:oracle.simple(p) for identifier,p in zip(scores.ids,scores.probabilities)}
            averages[dataset]=np.mean([values[i][2] for i in selected],axis=0)
            cpa[dataset]={i:values[i][1] for i in heldout}
        # Check both directions of the first model before moving to any other.
        for source,target in [('medqa','medmcqa'),('medmcqa','medqa')]:
            r=next(r for r in data['transfer_results'] if (r['model'],r['source'],r['target'])==(model,source,target))
            assert r['source_partition']['estimation']==parts[source][0]
            assert r['target_partition']['estimation']==parts[target][0]
            assert r['target_partition']['heldout']==parts[target][1]
            assert not set(parts[source][0])&set(stores[target].ids)
            assert np.allclose(r['source_prior'],averages[source],rtol=0,atol=1e-12)
            assert np.allclose(r['target_prior'],averages[target],rtol=0,atol=1e-12)
            lookup={i:(g,p[0]) for i,g,p in zip(stores[target].ids,stores[target].gold,stores[target].probabilities)}
            gold=[];predictions={m:[] for m in r['metrics']};ties={m:0 for m in r['metrics']}
            assert [p['sample_id'] for p in r['paired_predictions']]==parts[target][1]
            for i,saved in zip(parts[target][1],r['paired_predictions']):
                g,original=lookup[i];gold.append(int(g));assert saved['gold_index']==g
                values=dict(default=original,cpa=cpa[target][i],in_domain=np.log(original+1e-10)-np.log(averages[target]+1e-10),transfer=np.log(original+1e-10)-np.log(averages[source]+1e-10))
                for m,vector in values.items():
                    prediction=max(range(4),key=lambda j:vector[j])
                    assert saved[m+'_prediction']==prediction
                    predictions[m].append(prediction)
                    ties[m]+=sum(x==max(vector) for x in vector)>1
            assert ties==r['exact_ties']
            metrics={m:independent_metrics(gold,p) for m,p in predictions.items()}
            for m,record in metrics.items():
                for k,value in record.items():
                    assert np.allclose(value,r['metrics'][m][k],rtol=0,atol=1e-9,equal_nan=True),(model,source,m,k)
            for left,right in [('in_domain','default'),('transfer','default'),('transfer','in_domain')]:
                for k in ('accuracy','rstd'):
                    assert abs(metrics[left][k]-metrics[right][k]-r['contrasts'][left+'_minus_'+right][k])<1e-9
            checked.append(dict(model=model,source=source,target=target,paired_questions=len(gold),all_priors_predictions_ties_metrics_contrasts_match=True))
    report=dict(status='pass',oracle_path=str(oracle_path),oracle_sha256=sha256(oracle_path),checker_sha256=sha256(Path(__file__)),directions=checked,method='Pinned upstream simple() for each question and prior; separate stdlib partitions, first argmax, label arithmetic and contrasts. The raw loader is shared only for already-validated identity/IO.',uncertainty='Point estimates only')
    write_json_new(output/'independent_transfer_validation.json',report)
    print(json.dumps(dict(status='pass',directions=len(checked),paired_question_comparisons=sum(r['paired_questions'] for r in checked))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    check(parser.parse_args().output.resolve())
