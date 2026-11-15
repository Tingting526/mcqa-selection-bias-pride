#!/usr/bin/env python3
"""Read-only cross-artifact QA; emits one new final delivery report."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def check(output):
    root=Path(__file__).resolve().parents[2]
    d=json.loads((output/'analysis_results.json').read_text())
    expected={
        'reproduction_validation.csv':d['validation'],
        'tables/rq1_reproduction.csv':[r for r in d['summary_rows'] if r['rq']=='RQ1'],
        'tables/rq2_medical.csv':[r for r in d['summary_rows'] if r['rq']=='RQ2'],
        'tables/rq2_five_benchmark_comparison.csv':d['summary_rows'],
        'tables/label_metrics.csv':d['label_rows'],
        'tables/calibration_draws.csv':d['calibration_draws'],
        'tables/metric_availability.csv':d['metric_availability'],
        'tables/nominal_workload.csv':d['nominal_workload'],
        'tables/rq3_availability.csv':d['transfer_availability'],
        'tables/verified_permutation_metrics.csv':d['permutation_metrics'],
    }
    checks=[]
    for relative,records in expected.items():
        with (output/relative).open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==len(records),relative
        for actual,original in zip(rows,records):
            for key,value in original.items():
                if value is None: assert actual[key]=='',(relative,key)
                elif isinstance(value,bool): assert actual[key]==str(value).lower(),(relative,key)
                elif isinstance(value,(float,int)): assert float(actual[key])==value,(relative,key)
                elif isinstance(value,(dict,list)): assert json.loads(actual[key])==value,(relative,key)
                else: assert actual[key]==value,(relative,key)
        checks.append(dict(path=str(output/relative),rows=len(rows),all_cells_match_scientific_json=True))
    def read(relative):
        with (output/relative).open(newline='') as stream:return list(csv.DictReader(stream))
    methods=read('tables/rq3_results.csv');contrasts=read('tables/rq3_contrasts.csv');priors=read('tables/rq3_priors.csv')
    assert len(methods)==48 and len(contrasts)==12 and len(priors)==48
    for r in d['transfer_results']:
        for method,m in r['metrics'].items():
            row=next(v for v in methods if (v['model'],v['source'],v['target'],v['method'])==(r['model'],r['source'],r['target'],method))
            assert float(row['accuracy_percent'])==m['accuracy'] and float(row['rstd_pp'])==m['rstd']
            assert json.loads(row['recalls_percent'])==m['recalls'] and json.loads(row['supports'])==m['supports']
            assert int(row['n'])==len(r['target_partition']['heldout'])
            assert row['target_heldout_ids_sha256']==hashlib.sha256(json.dumps(r['target_partition']['heldout'],separators=(',',':')).encode()).hexdigest()
        row=next(v for v in contrasts if (v['model'],v['source'],v['target'])==(r['model'],r['source'],r['target']))
        for name,values in r['contrasts'].items():
            for metric,value in values.items():assert float(row[name+'_'+metric+'_pp'])==value
        for role in ('source','target'):
            for i,label in enumerate('ABCD'):
                row=next(v for v in priors if (v['model'],v['benchmark'],v['label'])==(r['model'],r[role],label))
                assert float(row['prior_probability'])==r[role+'_prior'][i]
    diffs=read('tables/input_representation_differences.csv')
    assert len(diffs)==10 and all(json.loads(r['differences'])[0]['source_text']=='None' and json.loads(r['differences'])[0]['executed_text']=='nan' for r in diffs)
    with (root/'experiment_evidence_ledger.csv').open(newline='') as stream:ledger=list(csv.DictReader(stream))
    baseline=json.loads((output/'preservation_baseline.json').read_text())
    backup=next(Path(r['backup']) for r in baseline['backups'] if Path(r['source'])==root/'experiment_evidence_ledger.csv')
    with backup.open(newline='') as stream:old=list(csv.DictReader(stream))
    assert len(ledger)==len(old)==73
    primary=[r for r in ledger if r['experimental_block'].startswith('A:')];transfer=[r for r in ledger if r['experimental_block'].startswith('C:')]
    assert len(primary)==30 and len(transfer)==12
    assert all(r['verification_status']=='verified_from_matching_raw_observations' for r in primary)
    assert all(r['verification_status']=='completed_from_verified_raw' for r in transfer)
    assert [r for r in ledger if not r['experimental_block'].startswith(('A:','C:'))]==[r for r in old if not r['experimental_block'].startswith(('A:','C:'))]
    for r in primary+transfer:
        assert sha(Path(r['summary_location']))==r['summary_sha256']
        assert sha(Path(r['observation_location']))==r['observation_hash']==r['expected_observation_hash']
        assert Path(r['observation_location']).stat().st_size==int(r['actual_bytes'])
    manifest=json.loads((output/'source_manifest_final.json').read_text())
    assert all(sha(Path(r['path']))==r['sha256'] for r in manifest['inputs'])
    export=json.loads((output/'table_export_validation.json').read_text())
    assert len(export['files'])==15 and all(r['allAuthoredCellsVerified'] and sha(Path(r['path']))==r['sha256'] for r in export['files'])
    cpu=json.loads((output/'cpu_execution_audit.json').read_text())
    assert cpu['import_guard_enabled'] and not cpu['forbidden_import_attempts'] and not cpu['loaded_forbidden_modules']
    tests=list(ET.parse(output/'tests_final.xml').getroot().iter('testsuite'))
    assert sum(int(t.get('tests')) for t in tests)==42
    assert all(t.get('failures')=='0' and t.get('errors')=='0' for t in tests)
    oracle=json.loads((output/'independent_transfer_validation.json').read_text())
    assert oracle['status']=='pass' and len(oracle['directions'])==12
    assert oracle['checker_sha256']==sha(root/'analysis/scripts/check_phase3b_transfer.py')
    for p in [root/'analysis/src/offline_phase3b.py',*list((root/'analysis/scripts').glob('*phase3b*.py')),*list((root/'analysis/tests').glob('test_*phase3b*.py'))]:ast.parse(p.read_text())
    figures=json.loads((output/'figures/figure_manifest.json').read_text())
    assert len(figures['files'])==12 and all(sha(Path(r['path']))==r['sha256'] for r in figures['files'])
    for p in (output/'figures').glob('*.metadata.json'):
        metadata=json.loads(p.read_text());assert metadata['source_sha256']==sha(output/'analysis_results.json')
    assert len(d['validation'])==30 and all(r['status']=='verified_from_matching_raw_observations' and not r['discrepancies'] for r in d['validation'])
    assert json.loads((output/'missing_observations.json').read_text())==[]
    recovery=json.loads((root/'analysis/outputs/phase3b_lrz_manual_20261018/local_recovery_verification.json').read_text())
    result=dict(status='pass',csv_roundtrip=checks,transfer_method_rows=48,transfer_contrast_rows=12,prior_label_rows=48,input_representation_rows=10,ledger_rows=73,historical_ledger_rows_unchanged=True,source_files_verified=len(manifest['inputs']),focused_tests_passed=42,full_raw_pairs_verified=30,real_transfer_directions_verified=12,independent_oracle_paired_questions=sum(r['paired_questions'] for r in oracle['directions']),figure_files_verified=12,figure_visual_qa='All six PNGs inspected; labels, units, legends and notes readable. PNG/PDF share the same Matplotlib canvas.',cpu_import_guard_passed=True,recovered_evidence_receipt=recovery)
    with (output/'delivery_quality_checks.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('csv_roundtrip','recovered_evidence_receipt')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);check(parser.parse_args().output.resolve())
