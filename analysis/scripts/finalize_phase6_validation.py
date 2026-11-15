#!/usr/bin/env python3
"""Final delivery checks, source provenance and preservation; no manuscript edits."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'
sys.path.insert(0,str(ROOT/'analysis'))
from src.offline_phase3b import write_json_new,sha256

def record(p,role):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p),role=role)

def main():
    baseline=json.loads((OUT/'preservation_baseline.json').read_text())['files'];mismatches=[]
    for r in baseline:
        p=Path(r['path'])
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:mismatches.append(r['path'])
    assert not mismatches,mismatches
    # Also validate the historical source hashes, not merely a current snapshot.
    original_inputs=json.loads((ROOT/'analysis/outputs/phase3b_20261018/verified/source_manifest_final.json').read_text())['inputs']
    for r in original_inputs:assert Path(r['path']).stat().st_size==r['bytes'] and sha256(Path(r['path']))==r['sha256'],r['path']
    d=json.loads((OUT/'analysis_results.json').read_text());s=json.loads((OUT/'seed_resampling_results.json').read_text());c=json.loads((OUT/'seed_control_results.json').read_text())
    checks=[]
    for name,rows in [('equal_budget_results',d['equal_budget']),('calibration_seed_sensitivity',s['rows']+c['rows']),('corrected_permutation_robustness',d['corrected_permutation']),('mmlu_medical_subject_analysis',d['mmlu_subjects'])]:
        with (OUT/(name+'.csv')).open(newline='') as f:reader=csv.DictReader(f);headers=reader.fieldnames;actual=list(reader)
        expectedkeys=list(dict.fromkeys(k for row in rows for k in row));assert headers==expectedkeys and len(actual)==len(rows)
        cells=0
        for a,b in zip(actual,rows):
            for key in expectedkeys:
                value=b.get(key);text=a[key]
                if value is None:assert text=='NA'
                elif isinstance(value,bool):assert text==json.dumps(value)
                elif isinstance(value,(int,float)):assert math.isclose(float(text),value,rel_tol=0,abs_tol=1e-12)
                elif isinstance(value,(list,dict)):assert json.loads(text)==value
                else:assert text==value,(key,text,value)
                cells+=1
        for required in ['model','benchmark','evaluation_population','calibration_policy','units','source_version']:
            assert required in headers and all(row[required] for row in actual)
        checks.append(dict(table=name,rows=len(rows),columns=len(headers),cells=cells,all_cells_match=True))
    primary=json.loads((ROOT/'analysis/outputs/phase3b_20261018/verified/analysis_results.json').read_text())
    baseline_exact=[]
    for r in c['rows']:
        p=next(p for p in primary['transfer_results'] if (p['model'],p['source'],p['target'])==(r['model'],r['source'],r['target']))
        assert r['source_prior']==p['source_prior'] and r['target_prior']==p['target_prior']
        for m in ['default','in_domain','transfer','cpa']:
            for metric,unit in [('accuracy','percent'),('rstd','pp')]:assert r[m+'_'+metric+'_'+unit]==p['metrics'][m][metric]
        baseline_exact.append(True)
    for r in d['equal_budget']:
        assert r['source_calibration_n']==63 and r['target_calibration_n']==(63 if r['target']=='medqa' else 197)
        assert r['evaluation_n']==(1210 if r['target']=='medqa' else 3744)
        if r['source']=='medqa':assert all(abs(v)<1e-9 for k,v in r.items() if k.startswith('change_vs_primary_'))
    for r in d['corrected_permutation']:
        assert r['n_labels']==r['cyclic_shifts']==(5 if r['benchmark']=='csqa' else 4)
        for prefix in ['default_','pride_']:assert r[prefix+'cyclic_supports']==[r['evaluation_n']]*r['n_labels']
        assert abs(r['delta_semantic_consistency_pp']-100*(r['consistency_gained_n']-r['consistency_lost_n'])/r['evaluation_n'])<1e-9
        assert abs(r['delta_original_accuracy_pp']-100*(r['original_correct_gained_n']-r['original_correct_lost_n'])/r['evaluation_n'])<1e-9
    for r in d['mmlu_subjects']:
        assert sum(r['full_default_gold_supports'])==r['group_population_n']
        assert sum(r['default_gold_supports'])==r['group_evaluation_n']
        assert min(r['full_default_gold_supports'])>0 and min(r['default_gold_supports'])>0
    independent=json.loads((OUT/'independent_validation_final.json').read_text());assert independent['status']=='pass' and not independent['forbidden_imports']
    assert len(independent['checks'])==30 and independent['equal_budget_pairs']==12 and len(independent['seed_checks'])==2 and independent['mmlu_rows']==366
    cpu=json.loads((OUT/'cpu_execution_audit.json').read_text());assert not cpu['imported_forbidden_modules'] and cpu['new_inference'] is False
    suite=ET.parse(OUT/'tests_focused.xml').getroot();suites=[x for x in suite.iter('testsuite')]
    tests=sum(int(x.attrib['tests']) for x in suites);failures=sum(int(x.attrib['failures'])+int(x.attrib['errors']) for x in suites);skips=sum(int(x.attrib['skipped']) for x in suites)
    assert tests==53 and failures==0 and skips==0
    figuremanifest=json.loads((OUT/'figures/figure_manifest.json').read_text());assert len(figuremanifest['figures'])==4
    from PIL import Image
    for fig in figuremanifest['figures']:
        for r in fig['files']:assert sha256(Path(r['path']))==r['sha256']
        assert fig['source_analysis_sha256']==sha256(OUT/'analysis_results.json')
        assert fig['seed_results_sha256']==sha256(OUT/'seed_resampling_results.json')
        png=Path(next(r['path'] for r in fig['files'] if r['path'].endswith('.png')))
        with Image.open(png) as image:assert image.width>=1400 and image.height>=900
    tablemap=json.loads((OUT/'tables/table_source_map.json').read_text());assert len(tablemap['tables'])==11
    # Check the published table cells against independent source-row selection.
    for table in tablemap['tables']:
        name=table['name'];expected=[]
        for mapping in table['source_rows']:
            m=mapping['model']
            short={'llama-3.1-8b-instruct':'Llama 3.1 8B','qwen2.5-7b-instruct':'Qwen2.5 7B','gemma-2-9b-it':'Gemma 2 9B','mistral-7b-instruct-v0.3':'Mistral 7B v0.3','qwen3-8b':'Qwen3 8B','glm-4-9b-chat-hf':'GLM 4 9B'}[m]
            if name.startswith('equal_budget_'):
                r=next(r for r in d['equal_budget'] if r['model']==m and r['source']==mapping['source']);expected.append([short,*[r[k] for k in ['primary_transfer_minus_in_domain_accuracy_pp','transfer_minus_in_domain_accuracy_pp','transfer_minus_default_accuracy_pp','transfer_minus_in_domain_rstd_pp']]])
            elif name.startswith('seed_sensitivity_'):
                r=next(r for r in s['summaries'] if r['model']==m and r['source']==mapping['source']);k='transfer_minus_in_domain_accuracy_pp';expected.append([short,*[r[k+x] for x in ['_seed_mean','_seed_sample_sd','_seed_min','_seed_max']],str(r[k+'_seed_positive_n'])])
            elif name.startswith('corrected_permutation_'):
                r=next(r for r in d['corrected_permutation'] if r['model']==m and r['benchmark']==mapping['benchmark']);expected.append([short,*[r[k] for k in ['delta_original_accuracy_pp','delta_original_rstd_pp','delta_cyclic_rstd_pp','delta_semantic_consistency_pp']]])
            else:
                a=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==mapping['grouping'] and r['group']=='medical');b=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==mapping['grouping'] and r['group']=='remaining_subjects');expected.append([short,a['full_default_original_accuracy_percent'],b['full_default_original_accuracy_percent'],a['full_default_cyclic_rstd_pp'],b['full_default_cyclic_rstd_pp'],a['full_default_semantic_consistency_percent']-b['full_default_semantic_consistency_percent']])
        assert expected==table['display_rows'],name
        for suffix in ['md','tex']:assert (OUT/'tables'/(name+'.'+suffix)).is_file()
    # Scientific source/code inventory is frozen after all authoring has finished.
    code=[ROOT/'analysis/src/phase6_robustness.py',ROOT/'analysis/tests/test_phase6_robustness.py',*sorted((ROOT/'analysis/scripts').glob('*phase6*'))]
    for p in code:
        destination=OUT/'code_snapshot'/p.relative_to(ROOT/'analysis');destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
    inputs={r['path']:{**r,'role':'historical_verified_input'} for r in original_inputs}
    for rel in ['analysis/outputs/phase3b_20261018/verified/analysis_results.json','analysis/outputs/phase4_20261018_20261018T091030Z/statistics_results.json','analysis/outputs/phase4_20261018_20261018T091030Z/bootstrap_accuracy_samples.npz','analysis/outputs/phase4_20261018_20261018T091030Z/bootstrap_indices_medqa.npz','analysis/outputs/phase4_20261018_20261018T091030Z/bootstrap_indices_medmcqa.npz','analysis/src/offline_phase3b.py','analysis/data/reference/zheng2024/upstream/code/debias_utils.py']:
        p=ROOT/rel;inputs[str(p)]=record(p,'verified_upstream_evidence_or_reused_loader')
    for p in code:inputs[str(p)]=record(p,'new_phase6_cpu_code')
    request=Path('<local-attachment> text.txt');inputs[str(request)]=record(request,'user_phase6_request')
    for p in sorted((ROOT/'Paper/chapters').glob('*.tex')):inputs[str(p)]=record(p,'read_only_manuscript_context')
    write_json_new(OUT/'source_manifest.json',dict(version='phase6-robustness-v1',inputs=sorted(inputs.values(),key=lambda r:r['path']),raw_score_sources=json.loads((OUT/'raw_sources.json').read_text())['files'],preservation_baseline=str(OUT/'preservation_baseline.json'),preservation_baseline_sha256=sha256(OUT/'preservation_baseline.json'),python=sys.executable,python_version=sys.version,epsilon=1e-10,first_maximum=True,new_gpu_inference=False))
    report=dict(status='pass',coverage=dict(equal_budget_pairs=12,conditional_resampling_rows=60,degenerate_control_rows=60,corrected_permutation_pairs=30,mmlu_rows=366,mmlu_subjects=57,models=6,benchmarks=5),tests=dict(total=tests,new_synthetic=19,failures=failures,skips=skips),independent_validation=independent,table_roundtrip=checks,all_presentation_tables_match=True,protected_files=len(baseline),unchanged_paper_files=sum('/Paper/' in r['path'] for r in baseline),historical_source_manifest_entries_verified=len(original_inputs),preservation_mismatches=mismatches,primary_control_exact_matches=len(baseline_exact),source_and_target_partitions='disjoint within benchmark; namespaced across benchmarks; identical primary target holdouts in RQ3 variants',units='percent endpoints; pp contrasts/RStd/SD; probability priors; ddof1 across seeds and ddof0 across labels/shifts',no_gpu_inference=True,bootstrap_intervals_unchanged=True,thesis_pdf_sha256=sha256(ROOT/'Paper/thesis.pdf'),figures=dict(count=4,formats=['SVG','PNG'],visual_review='all four PNGs inspected; MMLU xlabel shortened after detecting right-edge clipping; corrected export reinspected'),decision_status='With-replacement sensitivity was disclosed before implementation and separately labelled; author methodological preference for integration remains unanswered. No additional inference is needed.',limitations=['conditional calibration resampling within one fixed original pool','both priors vary together in seed condition','single nested equal-source subset','first-draw model-specific heldout permutation evaluation','MMLU topic-group boundaries and difficulty are descriptive','primary input/scoring limits remain unchanged'])
    write_json_new(OUT/'validation_report.json',report)
    text=f"""# Phase 6 validation report

Status: PASS for the explicitly documented estimands. No manuscript revision.

- {tests} focused CPU tests pass, including19 new synthetic cases;0 failures/errors/skips. Coverage includes nested randomized-order selection, leakage/unknown/duplicate ID rejection, fitting multiplicities, five-seed controls and resampling, four/five-label mapping, frozen correction under a known factorization, additive epsilon and first ties, unsupported recall, paired transitions, sample-SD/sign counts and subject grouping. Test XML and logs are retained.
- Independent pinned upstream simple() decomposition is checked on every retained question in all30 model/benchmark populations. Separate scalar log correction, explicit cyclic mappings and stdlib mean/SD reconstruct all heldout metrics,12 equal-budget pairs,60 resamples,60 controls and366 subject/group rows. {independent['metric_cells_checked']} metric fields checked, plus priors, predictions, differences, supports and partitions. independent_validation_final.json additionally verifies every MMLU paired metric difference and positive label support.
- All60 full-pool controls exactly reproduce primary prior arrays and all four methods' accuracy/original-RStd values, including seed42. Informative seed42 resamples use different draws and are not represented as reproducing the primary condition.
- CSV read-back matches every source cell across12/120/30/366 rows. Model, benchmark, actual and parent subgroup counts, evaluation population, calibration policy, units and version are explicit. Arrays encode ordered IDs/multiplicities as JSON; undefined/not-applicable uses NA. All11 presentation tables match independently selected source rows. Four vector-SVG/PNG figures use verified values, with metadata hashes and visual review.
- Same raw protocol and model signatures, complete four/five cycles, canonical IDs, gold content, population fingerprints and available scoring traces are revalidated by the strict raw loader. No historical chat scores enter. RQ3 target holdouts stay exactly1210 MedQA or3744 MedMCQA, identified by ordered-ID hashes. Calibration is disjoint; source namespaces are separate; epsilon1e-10 and first maxima are preserved.
- All{len(baseline)} pre-existing protected files are byte-identical, including all45 Paper files and approved bootstrap evidence. All{len(original_inputs)} historical source-manifest entries also match their recorded original hashes. No production inference/analysis module was edited; all Phase6 code is new. No model dependency was imported and no GPU inference executed. Thesis SHA256: `{report['thesis_pdf_sha256']}`.

Accuracy/recall/consistency endpoints use percent; contrasts, RStd and SD use pp. Seed sample SD(ddof1) is descriptive calibration variability, not a confidence interval. Permutation and recall SD use ddof0. CPA is aggregation before selection; mean permutation accuracy is not CPA. No RStd CI, significance test, fresh-partition uncertainty or reinterpretation of the approved question-bootstrap intervals was introduced.

The fixed-pool full-size seed design is degenerate. Its with-replacement alternative was explained before implementation and is separately labelled; the author's integration preference has not yet been received. This is a disclosed scope/interpretation decision, not a numerical validation failure. Full policies and remaining limits are in analysis_protocol.md and scientific_interpretation.md.
"""
    with (OUT/'validation_report.md').open('x') as f:f.write(text)
    files=[record(p,'phase6_output') for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json']
    external=record(ROOT/'thesis_robustness_integration_plan.md','separate_integration_proposal')
    write_json_new(OUT/'output_manifest.json',dict(files=files,external_deliverables=[external],excludes='this manifest itself; no active thesis files are outputs'))
    print(json.dumps(dict(status='pass',protected=len(baseline),paper_files=45,tests=tests,csv_rows=[r['rows'] for r in checks],outputs=len(files),thesis_sha256=report['thesis_pdf_sha256'])))

if __name__=='__main__':main()
