#!/usr/bin/env python3
"""Read-only audit of Phase 6 definitions before manuscript integration."""
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path((ROOT/'tmp/phase7_active_output.txt').read_text().strip())
P6=ROOT/'analysis/outputs/phase6_robustness_20261018'
sys.path.insert(0,str(ROOT/'analysis'));sys.path.insert(0,str(ROOT/'analysis/scripts'))
from src.offline_phase3b import sha256,write_json_new
from check_phase6_independent import scalar_metrics, predictions

def ih(v):return hashlib.sha256(json.dumps(list(v),separators=(',',':')).encode()).hexdigest()

def main():
    manifest=json.loads((P6/'output_manifest.json').read_text());checked=0
    for r in manifest['files']+manifest['external_deliverables']:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha256(p)==r['sha256'],p;checked+=1
    sources=json.loads((P6/'source_manifest.json').read_text())
    for r in sources['inputs']:assert sha256(Path(r['path']))==r['sha256'],r['path']
    d=json.loads((P6/'analysis_results.json').read_text());s=json.loads((P6/'seed_resampling_results.json').read_text());c=json.loads((P6/'seed_control_results.json').read_text())
    primary=json.loads((ROOT/'analysis/outputs/phase3b_20261018/verified/analysis_results.json').read_text())
    equal=[];seed_checks=[];permutation=[];stores={}
    for r in d['equal_budget']:
        p=next(v for v in primary['transfer_results'] if (v['model'],v['source'],v['target'])==(r['model'],r['source'],r['target']))
        sf=p['source_partition']['estimation_order'][:63];tf=p['target_partition']['estimation'];ev=p['target_partition']['heldout']
        assert r['source_fit_ids']==sf and len(set(sf))==63 and set(sf)<=set(p['source_partition']['estimation'])
        assert r['target_fit_ids']==tf and r['evaluation_ids_sha256']==ih(ev)
        assert not set(tf)&set(ev) and not set(sf)&set(p['source_partition']['heldout'])
        assert ih(sf)==r['source_fit_ids_sha256'] and ih(tf)==r['target_fit_ids_sha256']
        assert r['target_prior']==p['target_prior']
        if r['source']=='medqa':
            np.testing.assert_allclose(r['source_prior'],p['source_prior'],rtol=0,atol=1e-15)
            assert all(abs(v)<1e-9 for k,v in r.items() if k.startswith('change_vs_primary_'))
            z=np.load(P6/'scores'/f'{r["model"]}__{r["target"]}.npz');li={i:j for j,i in enumerate(z['ids'])};ix=[li[i] for i in ev]
            np.testing.assert_array_equal(predictions(z['probabilities'][ix,0:1],r['source_prior']),predictions(z['probabilities'][ix,0:1],p['source_prior']))
        equal.append(dict(model=r['model'],source=r['source'],target=r['target'],source_fit_n=63,target_fit_n=len(tf),target_n=len(ev),exact_target_hash=ih(ev)))
    for condition in [s,c]:
        for r in condition['rows']:
            p=next(v for v in primary['transfer_results'] if (v['model'],v['source'],v['target'])==(r['model'],r['source'],r['target']))
            for role in ['source','target']:
                pool=sorted(p[role+'_partition']['estimation']);rng=random.Random(r['seed'])
                expected=[pool[rng.randrange(len(pool))] for _ in pool] if condition['policy']=='fixed_pool_with_replacement' else sorted(rng.sample(pool,len(pool)))
                actual=r[role+'_fit_ids'];assert expected==actual and ih(actual)==r[role+'_fit_ids_sha256']
                assert len(actual)==len(pool) and set(actual)<=set(pool) and not set(actual)&set(p[role+'_partition']['heldout'])
            assert r['evaluation_ids_sha256']==ih(p['target_partition']['heldout'])
        assert {r['seed'] for r in condition['rows']}=={42,43,44,45,46} and len(condition['rows'])==60
        seed_checks.append(dict(policy=condition['policy'],rows=60,exact_draws_and_multiplicities_verified=True))
    for r in d['corrected_permutation']:
        model,ds=r['model'],r['benchmark'];z=np.load(P6/'scores'/f'{model}__{ds}.npz');pred=np.load(P6/'scores'/f'{model}__{ds}__heldout_predictions.npz');part=json.loads((P6/'partitions'/f'{model}__{ds}.json').read_text());ids=z['ids'].tolist();li={i:j for j,i in enumerate(ids)}
        assert not set(part['fit_ids'])&set(part['evaluation_ids']) and set(part['fit_ids'])|set(part['evaluation_ids'])==set(ids)
        assert part['evaluation_ids']==pred['ids'].tolist() and ih(part['evaluation_ids'])==r['evaluation_ids_sha256']
        ix=[li[i] for i in part['evaluation_ids']];fitix=[li[i] for i in part['fit_ids']];prior=z['item_priors'][fitix].mean(axis=0);np.testing.assert_array_equal(prior,r['prior'])
        k=5 if ds=='csqa' else 4;assert z['probabilities'].shape==(len(ids),k,k)
        a=predictions(z['probabilities'][ix]);b=predictions(z['probabilities'][ix],prior)
        np.testing.assert_array_equal(a,pred['default']);np.testing.assert_array_equal(b,pred['pride']);np.testing.assert_array_equal(z['gold'][ix],pred['gold'])
        for prefix,values in [('default_',a),('pride_',b)]:
            for key,value in scalar_metrics(pred['gold'],values).items():np.testing.assert_allclose(r[prefix+key],value,rtol=0,atol=1e-9)
        permutation.append(dict(model=model,benchmark=ds,cyclic_rstd_delta_pp=r['pride_cyclic_rstd_pp']-r['default_cyclic_rstd_pp'],consistency_delta_pp=100*(r['consistency_gained_n']-r['consistency_lost_n'])/r['evaluation_n'],evaluation_n=r['evaluation_n']))
        stores[model,ds]=z
    cyclic_exceptions=[r for r in permutation if r['cyclic_rstd_delta_pp']>=0];consistency_exceptions=[r for r in permutation if r['consistency_delta_pp']<=0]
    assert len(cyclic_exceptions)==4 and len(consistency_exceptions)==6
    assert sum(r['cyclic_rstd_delta_pp']<0 for r in permutation)==26 and sum(r['consistency_delta_pp']>0 for r in permutation)==24
    policy=json.loads((P6/'mmlu_grouping_policy.json').read_text());groups=json.loads((P6/'mmlu_subject_membership.json').read_text())
    assert policy['core5']==['anatomy','clinical_knowledge','college_medicine','medical_genetics','professional_medicine']
    assert set(policy['broad8'])==set(policy['core5'])|{'human_aging','nutrition','virology'}
    for r in d['mmlu_subjects']:
        assert sum(r['full_default_gold_supports'])==r['group_population_n'] and sum(r['default_gold_supports'])==r['group_evaluation_n']
        if r['grouping'] in ['core5','broad8']:
            expected=939 if r['grouping']=='core5' else 1604
            assert r['group_population_n']==(expected if r['group']=='medical' else 13592-expected)
    code=(ROOT/'analysis/scripts/run_phase6_robustness.py').read_text();assert code.index("write_json_new(OUT/'mmlu_grouping_policy.json'")<code.index("for v in primary['validation']")
    result=dict(status='pass',phase6_outputs_verified=checked,phase6_sources_verified=len(sources['inputs']),equal_budget=equal,seed_conditions=seed_checks,
        resampling_definition=dict(unit='original calibration question with its complete cached cyclic orbit',draw_counts=dict(medqa=63,medmcqa=197),seeds=[42,43,44,45,46],replacement=True,duplicates='Repeated question priors receive their full multiplicity in arithmetic mean',both_priors_refitted=True,primary_seed42='reference only; not one of the five resampled fits',scope='Conditional empirical-pool sensitivity, not independent dataset partitions or all possible benchmark splits'),
        permutation_checks=permutation,cyclic_rstd_decreases=26,consistency_increases=24,count_policy='Strict signs of unrounded metrics; consistency also independently checked as exact integer gained-minus-lost count. No sign differs at1e-9pp tolerance. No significance test.',
        cyclic_rstd_exceptions=cyclic_exceptions,consistency_exceptions=consistency_exceptions,mmlu_grouping=groups,group_definitions_before_performance=True,
        interpretation_correction='Use calibration-pool resampling sensitivity. Earlier generic seed-sensitivity headings are too broad; no new independent partitions were evaluated. All-six reverse-negative means extend, rather than exactly reproduce, the primary five-negative/one-tie pattern.',
        integration_authorization='The Phase7 request explicitly asks to verify and integrate the executed with-replacement condition using accurate terminology; the prior Phase6 preference is no longer a blocker.',no_new_inference=True)
    write_json_new(OUT/'methodology_verification.json',result)
    print(json.dumps(dict(status='pass',outputs=checked,sources=len(sources['inputs']),equal_pairs=12,seed_rows=120,cyclic_exceptions=len(cyclic_exceptions),consistency_exceptions=len(consistency_exceptions))))

if __name__=='__main__':main()
