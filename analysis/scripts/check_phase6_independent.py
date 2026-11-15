#!/usr/bin/env python3
"""Independent oracle decomposition and scalar/permutation arithmetic checks.

Does not import Phase 6 analysis functions, correction or metric helpers.
Shares only the strict loader's already validated read-only score snapshots.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'
EPS=1e-10

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ih(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
def close(a,b):
    if a is None:assert b is None or math.isnan(b)
    else:assert np.allclose(a,b,rtol=0,atol=1e-9,equal_nan=True),(a,b)

def scalar_metrics(gold,labels):
    n,k=labels.shape;supports=[sum(int(g)==l for g in gold) for l in range(k)]
    recalls=[100*sum(int(g)==l and int(p[0])==l for g,p in zip(gold,labels))/supports[l] if supports[l] else None for l in range(k)]
    origstd=statistics.pstdev(recalls) if all(x is not None for x in recalls) else None
    correct=[0]*k;bylabel=[0]*k;consistent=0
    for g,row in zip(gold,labels):
        contents=[]
        for s,label in enumerate(row):
            order=list(range(k))[s:]+list(range(k))[:s]
            content=order[int(label)];contents.append(content)
            if content==int(g):correct[s]+=1;bylabel[order.index(int(g))]+=1
        consistent+=len(set(contents))==1
    acc=[100*x/n for x in correct];cr=[100*x/n for x in bylabel]
    return dict(n=n,gold_supports=supports,original_recalls_percent=recalls,cyclic_supports=[n]*k,cyclic_recalls_percent=cr,
                original_accuracy_percent=acc[0],original_rstd_pp=origstd,cyclic_rstd_pp=statistics.pstdev(cr),accuracy_by_shift_percent=acc,
                mean_permutation_accuracy_percent=statistics.mean(acc),permutation_accuracy_sd_pp=statistics.pstdev(acc),
                permutation_accuracy_range_pp=max(acc)-min(acc),semantic_consistency_percent=100*consistent/n)

def predictions(p,prior=None):
    if prior is None:return np.array([[max(range(len(row)),key=lambda j:row[j]) for row in item] for item in p])
    return np.array([[max(range(len(row)),key=lambda j:math.log(float(row[j])+EPS)-math.log(float(prior[j])+EPS)) for row in item] for item in p])

def endpoints(p,g,cpa,sp,tp):
    vectors=dict(default=predictions(p)[:,0],in_domain=predictions(p[:,0:1],tp)[:,0],transfer=predictions(p[:,0:1],sp)[:,0],
                 cpa=np.array([max(range(4),key=lambda j:row[j]) for row in cpa]))
    result={}
    for m,v in vectors.items():
        # Original order metrics do not require a complete orbit.
        support=[sum(int(x)==j for x in g) for j in range(4)]
        recall=[100*sum(int(a)==j and int(b)==j for a,b in zip(g,v))/support[j] for j in range(4)]
        result[m+'_accuracy_percent']=100*sum(int(a)==int(b) for a,b in zip(g,v))/len(g)
        result[m+'_rstd_pp']=statistics.pstdev(recall)
    for a,b in [('in_domain','default'),('transfer','default'),('transfer','in_domain')]:
        for metric,unit in [('accuracy','percent'),('rstd','pp')]:result[a+'_minus_'+b+'_'+metric+'_pp']=result[a+'_'+metric+'_'+unit]-result[b+'_'+metric+'_'+unit]
    return result

def main(report_name='independent_validation.json'):
    data=json.loads((OUT/'analysis_results.json').read_text());primary=json.loads((ROOT/'analysis/outputs/phase3b_20261018/verified/analysis_results.json').read_text())
    path=ROOT/'analysis/data/reference/zheng2024/upstream/code/debias_utils.py'
    assert digest(path)=='f58d4d14074e13cf9f7dfc0a094c8cc6bb6ef578c41918a23e1bdfb456fc4151'
    spec=importlib.util.spec_from_file_location('phase6_zheng_oracle',path);oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    stores={};checked=[];scalar_cells=0
    for row in data['corrected_permutation']:
        model,ds=row['model'],row['benchmark'];z=np.load(OUT/'scores'/f'{model}__{ds}.npz')
        ids=z['ids'].tolist();gold=z['gold'];p=z['probabilities'];k=p.shape[-1]
        oracle_values=[oracle.simple(item) for item in p]
        cpa=np.array([v[1] for v in oracle_values]);priors=np.array([v[2] for v in oracle_values])
        np.testing.assert_allclose(cpa,z['cpa'],atol=1e-14,rtol=0);np.testing.assert_allclose(priors,z['item_priors'],atol=1e-14,rtol=0)
        part=json.loads((OUT/'partitions'/f'{model}__{ds}.json').read_text());fit=part['fit_ids'];ev=part['evaluation_ids']
        v=next(v for v in primary['validation'] if v['model']==model and v['dataset']==ds);summary=json.loads(Path(v['summary_path']).read_text())
        legacy=dict(zip(z['legacy_ids'],ids));expected_fit=[legacy[i] for i in summary['trials'][0]['estimation_ids']]
        assert fit==expected_fit and ev==sorted(set(ids)-set(fit)) and not set(ev)&set(fit)
        assert ih(fit)==row['fit_ids_sha256'] and ih(ev)==row['evaluation_ids_sha256']
        li={i:j for j,i in enumerate(ids)};ix=[li[i] for i in ev];prior=np.mean([priors[li[i]] for i in fit],axis=0);close(row['prior'],prior)
        a=predictions(p[ix]);b=predictions(p[ix],prior)
        saved=np.load(OUT/'scores'/f'{model}__{ds}__heldout_predictions.npz')
        np.testing.assert_array_equal(saved['default'],a);np.testing.assert_array_equal(saved['pride'],b);assert saved['ids'].tolist()==ev;np.testing.assert_array_equal(saved['gold'],gold[ix])
        am,bm=scalar_metrics(gold[ix],a),scalar_metrics(gold[ix],b)
        for prefix,metrics in [('default_',am),('pride_',bm)]:
            for key,value in metrics.items():close(row[prefix+key],value);scalar_cells+=1
        for key in am:
            delta='delta_'+key.replace('_percent','_pp')
            if delta in row:close(row[delta],bm[key]-am[key])
        ca=np.array([len(set((int(v)+s)%k for s,v in enumerate(item)))==1 for item in a]);cb=np.array([len(set((int(v)+s)%k for s,v in enumerate(item)))==1 for item in b])
        assert row['consistency_gained_n']==sum(~ca&cb) and row['consistency_lost_n']==sum(ca&~cb)
        assert row['original_correct_gained_n']==sum((a[:,0]!=gold[ix])&(b[:,0]==gold[ix]))
        assert row['original_correct_lost_n']==sum((a[:,0]==gold[ix])&(b[:,0]!=gold[ix]))
        stores[model,ds]=dict(ids=ids,gold=gold,p=p,cpa=cpa,priors=priors,lookup=li,legacy=z['legacy_ids'].tolist(),a=a,b=b,ix=ix)
        checked.append(dict(model=model,benchmark=ds,oracle_items=len(ids),heldout=len(ev),all_metrics_match=True))
        print(json.dumps(checked[-1]),flush=True)
    for r in data['equal_budget']:
        p=next(p for p in primary['transfer_results'] if (p['model'],p['source'],p['target'])==(r['model'],r['source'],r['target']))
        s=stores[r['model'],r['source']];t=stores[r['model'],r['target']]
        sf=p['source_partition']['estimation_order'][:63];tf=p['target_partition']['estimation'];ev=p['target_partition']['heldout']
        assert sf==r['source_fit_ids'] and tf==r['target_fit_ids'] and not set(sf)&set(t['ids'])
        assert not set(tf)&set(ev) and ih(ev)==r['evaluation_ids_sha256']
        sp=np.mean([s['priors'][s['lookup'][i]] for i in sf],axis=0);tp=np.mean([t['priors'][t['lookup'][i]] for i in tf],axis=0)
        close(r['source_prior'],sp);close(r['target_prior'],tp);close(tp,p['target_prior'])
        ix=[t['lookup'][i] for i in ev];m=endpoints(t['p'][ix],t['gold'][ix],t['cpa'][ix],sp,tp)
        for key,value in m.items():close(r[key],value);scalar_cells+=1
        for key,value in r.items():
            if key.startswith('change_vs_primary_'):
                primarykey=key.removeprefix('change_vs_primary_')
                if primarykey.endswith('_accuracy_pp') and '_minus_' not in primarykey:primarykey=primarykey.removesuffix('_pp')+'_percent'
                close(value,r[primarykey]-r['primary_'+primarykey])
    seed_reports=[]
    for filename in ['seed_control_results.json','seed_resampling_results.json']:
        if not (OUT/filename).exists():continue
        sd=json.loads((OUT/filename).read_text());policy=sd['policy']
        for r in sd['rows']:
            p=next(p for p in primary['transfer_results'] if (p['model'],p['source'],p['target'])==(r['model'],r['source'],r['target']))
            for role in ['source','target']:
                pool=sorted(p[role+'_partition']['estimation']);rng=random.Random(r['seed'])
                expected=sorted(rng.sample(pool,len(pool))) if policy=='full_pool_without_replacement_control' else [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
                assert r[role+'_fit_ids']==expected and ih(expected)==r[role+'_fit_ids_sha256']
                assert not set(expected)&set(p[role+'_partition']['heldout'])
            s=stores[r['model'],r['source']];t=stores[r['model'],r['target']]
            sp=np.mean([s['priors'][s['lookup'][i]] for i in r['source_fit_ids']],axis=0);tp=np.mean([t['priors'][t['lookup'][i]] for i in r['target_fit_ids']],axis=0)
            close(r['source_prior'],sp);close(r['target_prior'],tp)
            ev=p['target_partition']['heldout'];assert ih(ev)==r['evaluation_ids_sha256'];ix=[t['lookup'][i] for i in ev]
            for key,value in endpoints(t['p'][ix],t['gold'][ix],t['cpa'][ix],sp,tp).items():close(r[key],value);scalar_cells+=1
        for r in sd['summaries']:
            group=[v for v in sd['rows'] if (v['model'],v['source'],v['target'])==(r['model'],r['source'],r['target'])];assert sorted(v['seed'] for v in group)==[42,43,44,45,46]
            for key,value in r.items():
                for suffix,fn in [('_seed_mean',statistics.mean),('_seed_sample_sd',statistics.stdev),('_seed_min',min),('_seed_max',max)]:
                    if key.endswith(suffix):close(value,fn(v[key.removesuffix(suffix)] for v in group))
                for suffix,fn in [('_seed_positive_n',lambda x:x>1e-9),('_seed_negative_n',lambda x:x< -1e-9),('_seed_zero_n',lambda x:abs(x)<=1e-9)]:
                    if key.endswith(suffix):assert value==sum(fn(v[key.removesuffix(suffix)]) for v in group)
            for role in ['source','target']:
                close(r[role+'_prior_label_sample_sd_probability'],[statistics.stdev(v[role+'_prior'][j] for v in group) for j in range(4)])
        seed_reports.append(dict(policy=policy,rows=len(sd['rows']),all_metrics_and_sample_sd_match=True))
    grouping=json.loads((OUT/'mmlu_subject_membership.json').read_text())
    for r in data['mmlu_subjects']:
        t=stores[r['model'],'mmlu'];subjects=[i.rsplit(':',1)[0] for i in t['legacy']]
        if r['grouping']=='individual_subject':wanted={r['group']}
        else:
            medical=set(grouping[r['grouping']]);wanted=medical if r['group']=='medical' else set(subjects)-medical
        assert r['included_subjects']==sorted(wanted)
        fullix=[j for j,s in enumerate(subjects) if s in wanted];holdix=[j for j,i in enumerate(t['ix']) if subjects[i] in wanted]
        assert r['group_population_n']==len(fullix) and r['group_evaluation_n']==len(holdix)
        metrics=scalar_metrics(t['gold'][fullix],predictions(t['p'][fullix]))
        for key,value in metrics.items():close(r['full_default_'+key],value)
        groupmetrics={}
        for prefix,labels in [('default_',t['a']),('pride_',t['b'])]:
            groupmetrics[prefix]=scalar_metrics(t['gold'][t['ix']][holdix],labels[holdix])
            for key,value in groupmetrics[prefix].items():close(r[prefix+key],value);scalar_cells+=1
        for key,value in groupmetrics['default_'].items():
            delta='delta_'+key.replace('_percent','_pp')
            if delta in r:close(r[delta],groupmetrics['pride_'][key]-value)
        assert all(x>0 for x in metrics['gold_supports']) and all(x>0 for x in groupmetrics['default_']['gold_supports'])
    report=dict(status='pass',method='Pinned upstream simple() on every retained question; separate scalar label correction, explicit permutation orders, stdlib means/SD, independent saved-ID partition reconstruction. Raw-loader IO is shared through validated NPZ snapshots only.',oracle_sha256=digest(path),checks=checked,equal_budget_pairs=len(data['equal_budget']),seed_checks=seed_reports,mmlu_rows=len(data['mmlu_subjects']),metric_cells_checked=scalar_cells,
                forbidden_imports=[x for x in sys.modules if x.split('.')[0] in ('torch','transformers','vllm')])
    assert Path(report_name).name==report_name
    with (OUT/report_name).open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status='pass',metric_cells_checked=scalar_cells)))

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'independent_validation.json')
