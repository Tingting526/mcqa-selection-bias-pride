#!/usr/bin/env python3
"""Recompute Phase 6 from read-only validated raw scores, entirely on CPU."""
import json
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from src.offline_phase3b import load_raw, decomposition_arrays, sha256, write_json_new, json_safe
from src.phase6_robustness import *
OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'
VERIFIED=ROOT/'analysis/outputs/phase3b_20261018/verified'

def main():
    primary=json.loads((VERIFIED/'analysis_results.json').read_text())
    subject_manifest=dict(policy="Declared before inspecting performance; topic-name classification, not clinical purity",
        core5=sorted(CORE_MEDICAL),broad8=sorted(CORE_MEDICAL|RELATED_MEDICAL),
        rationale="Core explicitly medical teaching/clinical subjects; broad adds human aging, nutrition, virology. All complements are remaining subjects, including medically related biology/psychology; no pure nonmedical control is claimed.")
    write_json_new(OUT/'mmlu_grouping_policy.json',subject_manifest)
    permutation=[];mmlu=[];validations=[];stores={};sources=[];partitions={}
    for v in primary['validation']:
        model,ds=v['model'],v['dataset']
        summary=json.loads(Path(v['summary_path']).read_text())
        scores,diagnostics=load_raw(ROOT/'analysis',Path(v['manifest_path']),summary,records_path=Path(v['observation_path']))
        cpa,priors=decomposition_arrays(scores)
        file=OUT/'scores'/f'{model}__{ds}.npz'
        np.savez_compressed(file,ids=np.array(scores.ids),legacy_ids=np.array(scores.legacy_ids),gold=scores.gold,
                            probabilities=scores.probabilities,cpa=cpa,item_priors=priors)
        sources.append(dict(model=model,benchmark=ds,signature=scores.signature,provenance=scores.provenance,
                            score_snapshot=str(file),score_snapshot_sha256=sha256(file)))
        bylegacy=dict(zip(scores.legacy_ids,scores.ids))
        fit=[bylegacy[i] for i in summary['trials'][0]['estimation_ids']]
        # Independently reconstruct the first sequential path-seeded shuffle.
        shuffled=list(range(len(scores.ids)));random.Random(summary['source_path_seed'].encode()).shuffle(shuffled)
        assert fit==[scores.ids[i] for i in shuffled[:int(.05*len(scores.ids))]]
        evaluation=sorted(set(scores.ids)-set(fit));validate_partition(scores.ids,fit,evaluation)
        prior=average_prior(priors,scores.ids,fit)
        assert np.allclose(prior,summary['trials'][0]['global_prior'],atol=1e-12,rtol=0)
        lookup={i:j for j,i in enumerate(scores.ids)};ix=[lookup[i] for i in evaluation]
        default,corrected=frozen_predictions(scores.probabilities[ix],prior);gold=scores.gold[ix];k=scores.probabilities.shape[-1]
        policy='original_hybrid_first_path_seeded_draw; frozen prior; heldout only'
        meta=dict(model=model,benchmark=ds,evaluation_population='retained population minus saved first hybrid calibration draw',
                  population_n=len(scores.ids),evaluation_n=len(ix),calibration_n=len(fit),n_labels=k,cyclic_shifts=k,
                  calibration_policy=policy,fit_ids_sha256=ids_hash(fit),evaluation_ids_sha256=ids_hash(evaluation),
                  prior=prior.tolist(),units='accuracy/recall/consistency percent; RStd, SD, range, differences pp',
                  source_version=VERSION,raw_protocol=scores.signature['protocol'],source_observation_sha256=v['actual_observation_sha256'])
        record={**meta,**before_after(gold,default,corrected,k)};permutation.append(record)
        assert abs(record['default_original_accuracy_percent']-summary['trials'][0]['heldout_default']['accuracy'])<1e-9
        assert abs(record['pride_original_accuracy_percent']-summary['trials'][0]['heldout_pride']['accuracy'])<1e-9
        part=dict(**meta,fit_ids=fit,evaluation_ids=evaluation)
        write_json_new(OUT/'partitions'/f'{model}__{ds}.json',part)
        np.savez_compressed(OUT/'scores'/f'{model}__{ds}__heldout_predictions.npz',ids=np.array(evaluation),gold=gold,default=default,pride=corrected)
        validations.append(dict(model=model,benchmark=ds,diagnostics=diagnostics,calibration_disjoint=True,
                                first_saved_draw_and_prior_reproduced=True,original_heldout_endpoints_reproduced=True))
        if ds=='mmlu':
            subjects=np.array([i.rsplit(':',1)[0] for i in scores.legacy_ids]);unique=sorted(set(subjects))
            assert len(unique)==57 and CORE_MEDICAL|RELATED_MEDICAL <= set(unique)
            subject_manifest['all_subjects']=unique
            subject_manifest['excluded_core5']=sorted(set(unique)-CORE_MEDICAL)
            subject_manifest['excluded_broad8']=sorted(set(unique)-(CORE_MEDICAL|RELATED_MEDICAL))
            full_default=scores.probabilities.argmax(axis=2)
            groups=[('individual_subject',s,subjects==s) for s in unique]
            for scheme in ('core5','broad8'):
                for group in ('medical','remaining_subjects'):
                    mask=np.array([subject_group(s,scheme)==group for s in subjects]);groups.append((scheme,group,mask))
            for scheme,group,mask in groups:
                holdmask=mask[ix]
                row={**meta,'grouping':scheme,'group':group,'included_subjects':sorted(set(subjects[mask])),
                     'group_population_n':int(mask.sum()),'group_evaluation_n':int(holdmask.sum()),
                     'prior_scope':'one benchmark-wide first-draw MMLU prior; not fitted separately by subject',
                     **{'full_default_'+key:value for key,value in cyclic_metrics(scores.gold[mask],full_default[mask],k).items()},
                     **before_after(gold[holdmask],default[holdmask],corrected[holdmask],k)}
                mmlu.append(row)
        if ds in ('medqa','medmcqa'):stores[model,ds]=(scores,cpa,priors)
        print(json.dumps(dict(loaded=model,benchmark=ds,n=len(scores.ids),heldout=len(ix))),flush=True)
    write_json_new(OUT/'mmlu_subject_membership.json',subject_manifest)
    equal=[]
    for p in primary['transfer_results']:
        model,source,target=p['model'],p['source'],p['target'];s,_,sp=stores[model,source];t,cpa,tp=stores[model,target]
        assert s.signature==t.signature
        sf=nested_budget(p['source_partition']);tf=p['target_partition']['estimation'];ev=p['target_partition']['heldout']
        validate_partition(s.ids,sf,p['source_partition']['heldout']);validate_partition(t.ids,tf,ev)
        assert not set(sf)&set(t.ids)
        sprior=average_prior(sp,s.ids,sf);tprior=average_prior(tp,t.ids,tf)
        np.testing.assert_allclose(tprior,p['target_prior'],rtol=0,atol=1e-12)
        li={i:j for j,i in enumerate(t.ids)};ix=[li[i] for i in ev]
        metrics,pred=transfer_metrics(t.probabilities[ix],t.gold[ix],cpa[ix],sprior,tprior)
        primary_metrics={}
        for m in ('default','in_domain','transfer','cpa'):
            primary_metrics[m+'_accuracy_percent']=p['metrics'][m]['accuracy'];primary_metrics[m+'_rstd_pp']=p['metrics'][m]['rstd']
        for contrast,v in p['contrasts'].items():
            for metric,value in v.items():primary_metrics[contrast+'_'+metric+'_pp']=value
        result=dict(model=model,source=source,target=target,benchmark=target,
                    evaluation_population='primary seed-42 target holdout',evaluation_n=len(ev),
                    source_calibration_n=len(sf),target_calibration_n=len(tf),source_prior=sprior.tolist(),target_prior=tprior.tolist(),
                    calibration_policy='source only: first 63 original randomized estimation_order; target prior unchanged',
                    changed_prior='source_medmcqa_197_to_nested63' if source=='medmcqa' else 'none; source_medqa_original63',
                    source_fit_ids=sf,target_fit_ids=tf,source_fit_ids_sha256=ids_hash(sf),target_fit_ids_sha256=ids_hash(tf),
                    evaluation_ids_sha256=ids_hash(ev),source_version=VERSION,units='accuracy percent; RStd and differences pp; priors probability',
                    raw_protocol=s.signature['protocol'],**metrics)
        for key,value in primary_metrics.items():
            result['primary_'+key]=value;result['change_vs_primary_'+key.replace('_percent','_pp')]=metrics[key]-value
        for method in ('default','in_domain','cpa'):
            assert abs(result['change_vs_primary_'+method+'_accuracy_pp'])<1e-9
        equal.append(result)
        partitions[model+'__'+source+'_to_'+target]=dict(source_fit_ids=sf,target_fit_ids=tf,evaluation_ids=ev)
    write_json_new(OUT/'equal_budget_partitions.json',partitions)
    write_json_new(OUT/'analysis_results.json',dict(version=VERSION,equal_budget=equal,corrected_permutation=permutation,mmlu_subjects=mmlu,raw_validation=validations))
    write_json_new(OUT/'raw_sources.json',dict(files=sources))
    write_json_new(OUT/'cpu_execution_audit.json',dict(imported_forbidden_modules=[x for x in sys.modules if x.split('.')[0] in ('torch','transformers','vllm')],new_inference=False,device='CPU',epsilon=EPSILON,tie_policy='first maximum'))
    print(json.dumps(dict(completed=True,equal_budget=len(equal),permutation=len(permutation),mmlu_rows=len(mmlu))))

if __name__=='__main__':main()
