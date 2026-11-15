import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from src.pride.zheng import evaluate, original_debias
from src.pride.prior_estimation import estimate_sample_prior, estimate_global_prior
from src.pride.debias import apply_pride_correction, cyclic_content_average


def test_sharp_and_zero_probabilities_match_original_numerics_exactly():
    z = original_debias()
    p = np.array([[.9,.09,.009,.001],[1e-25,.5,.4,.1],
                  [.4,0,.5,.1],[.4,.1,1e-16,.5]])
    p /= p.sum(1,keepdims=True)
    observed,cpa,prior=z.simple(p)
    np.testing.assert_array_equal(estimate_sample_prior(p),prior)
    np.testing.assert_array_equal(estimate_global_prior([prior,prior]),np.mean([prior,prior],axis=0))
    expected=z.softmax(np.log(p[0]+1e-10)-np.log(prior+1e-10))
    np.testing.assert_array_equal(apply_pride_correction(p[0],prior),expected)
    mappings=[[(j+i)%4 for j in range(4)] for i in range(4)]
    np.testing.assert_array_equal(cyclic_content_average(p,mappings),cpa)
    with pytest.raises(ValueError,match='legacy'):
        estimate_sample_prior(p,epsilon=1e-12)
    with pytest.raises(ValueError,match='legacy'):
        apply_pride_correction(p[0],prior,epsilon=1e-12)


@pytest.mark.parametrize('shape',[(4,4),(24,4),(5,5)])
def test_five_repeat_results_match_literal_upstream_processing(shape):
    z=original_debias()
    rng=np.random.default_rng(20261015)
    n=100
    samples=[dict(id=f'q{i}',gold=i%shape[1],probs=rng.dirichlet(np.full(shape[1],.12),size=shape[0]).tolist())
             for i in range(n)]
    seed_path=f'results_test/0s_model/test_{shape[0]}'
    result=evaluate(samples,seed_path)
    source_rng=random.Random(seed_path.encode('utf-8'))
    shuffled=list(samples)
    expected_acc=[];expected_std=[]
    for index in range(5):
        source_rng.shuffle(shuffled)
        prefix,postfix=shuffled[:5],shuffled[5:]
        predictions=[];labels=[];priors=[]
        for s in prefix:
            observed,debiased,prior=z.simple(np.array(s['probs']))
            priors.append(prior);predictions.append(np.argmax(debiased));labels.append(s['gold'])
        prior=np.mean(priors,axis=0)
        for s in postfix:
            observed=np.array(s['probs'][0])
            predictions.append(np.argmax(np.log(observed+1e-10)-np.log(prior+1e-10)))
            labels.append(s['gold'])
        predictions=np.array(predictions);labels=np.array(labels)
        recalls=[np.mean(predictions[labels==i]==i)*100 for i in range(shape[1])]
        expected_acc.append(np.mean(predictions==labels)*100)
        expected_std.append(np.std(recalls))
        assert result['trials'][index]['estimation_ids']==[s['id'] for s in prefix]
        np.testing.assert_array_equal(result['trials'][index]['global_prior'],prior)
    assert result['pride']['accuracy']==np.mean(expected_acc)
    assert result['pride']['rstd']==pytest.approx(np.mean(expected_std),abs=1e-13)
    assert result['pride']['accuracy_std']==np.std(expected_acc)
    assert result['pride']['repeats']==5 and result['n_labels']==shape[1]


def test_labels_never_enter_estimated_prior():
    rng=np.random.default_rng(42)
    samples=[dict(id=str(i),gold=i%5,probs=rng.dirichlet(np.ones(5),size=5).tolist()) for i in range(100)]
    first=evaluate(samples,'fixed-path')
    second=evaluate([{**s,'gold':(s['gold']+1)%5} for s in samples],'fixed-path')
    for a,b in zip(first['trials'],second['trials']):
        assert a['global_prior']==b['global_prior']
        assert a['estimation_ids']==b['estimation_ids']


def test_sampling_rejects_empty_estimation_set_and_duplicate_ids():
    sample=dict(id='x',gold=0,probs=np.full((4,4),.25).tolist())
    with pytest.raises(ValueError,match='Too few'):
        evaluate([sample],'test')
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate([sample,sample],'test')


def test_medical_extension_uses_same_prompt_as_original_arc():
    import sys
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    from reproduce_pride import medical_inputs
    from src.experimental.zheng_selection_bias import load_original, prepared
    from src.data.base import MCQSample
    import pandas as pd
    original=load_original()
    file=Path(__file__).resolve().parents[1]/'data/reference/zheng2024/upstream/code/data_arc/test/arc_test.csv'
    row=pd.read_csv(file,names=['Question','A','B','C','D','Answer'],dtype=str).iloc[0]
    sample=MCQSample('test','medqa',row.Question,tuple(row[l] for l in 'ABCD'),'ABCD'.index(row.Answer))
    generated=medical_inputs([sample],original)[0]
    with prepared(original,'arc',0,'cyclic') as (_,_,sample_fn,_):
        expected=sample_fn('arc')[0]
    assert generated[1]==expected


def test_postprocessing_is_cpu_only_and_refuses_changed_source(tmp_path, monkeypatch):
    import json, sys
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    import reproduce_pride as runner
    source=tmp_path/'raw'
    source.mkdir()
    manifest=source/'manifest.json'
    manifest.write_text(json.dumps(dict(config=dict(settings=['cyclic']))))
    rng=np.random.default_rng(12)
    records=[]
    for i in range(20):
        records.append(dict(shots=0,setting='cyclic',subject='arc',data=dict(idx=i,
            options=['first','second','third','fourth'],ideal='ABCD'[i%4],
            probs=rng.dirichlet(np.ones(4),size=4).tolist())))
    raw=source/'records.jsonl'
    raw.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    monkeypatch.setitem(runner.COUNTS,'arc',20)
    monkeypatch.setattr(runner,'compatible_sources',lambda *a:[manifest])
    monkeypatch.setattr(runner,'infer_medical',lambda *a:pytest.fail('Unexpected GPU inference'))
    output=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',['reproduce_pride.py','--model','qwen3-8b',
        '--datasets','arc','--postprocess-only','--output-root',str(output)])
    runner.main()
    result=json.loads((output/'results/qwen3-8b/arc/result.json').read_text())
    assert result['pride']['repeats']==5 and result['default']['n']==20
    assert result['scoring_protocol']==runner.RAW_PROTOCOL
    assert (output/'results_0shot.tex').exists()
    runner.main()  # Completed observations are immutable and reused.
    raw.write_text(raw.read_text()+'\n')
    with pytest.raises(ValueError,match='observations changed'):
        runner.main()


def test_canonical_jobs_wait_for_previous_chain_and_do_not_duplicate(tmp_path,monkeypatch):
    import json,sys
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import start_pride_reproduction as launcher
    import start_gemma_all_datasets as campaign
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    old=tmp_path/'outputs/campaigns/recovery-20261015/serial/serial_jobs.json'
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps(dict(groups=[dict(job_id='5853458'),dict(job_id='5853462')])))
    monkeypatch.setattr(launcher,'queue',lambda:[['5853462','PENDING','Dependency','mcq-serial']])
    monkeypatch.setattr(launcher,'original_debias',lambda:None)
    monkeypatch.setattr(launcher,'load_original',lambda:None)
    monkeypatch.setattr(launcher,'source_digest',lambda d:None)
    monkeypatch.setattr(launcher,'compatible_sources',lambda *a:[])
    monkeypatch.setattr(campaign,'artifact',lambda d:tmp_path/d)
    cpu=[]
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:cpu.append(a))
    calls=[]
    def run(command,**kwargs):
        if command[0]=='sinfo':return SimpleNamespace(stdout='gpu-test*')
        assert command[0]=='sbatch'
        calls.append((command,kwargs['input']))
        return SimpleNamespace(stdout=str(1000+len(calls)))
    monkeypatch.setattr(launcher,'run',run)
    monkeypatch.setattr(sys,'argv',['start_pride_reproduction.py','--submit','--partition','gpu-test'])
    launcher.main()
    launcher.main()
    assert len(calls)==6 and len(cpu)==6
    for i,(command,body) in enumerate(calls):
        expected='5853462' if i==0 else str(1000+i)
        assert '--dependency=afterany:'+expected in command
        assert '--job-name=mcq-pride-repro' in body
        source=body.split("<<'CANONICAL_PRIDE_PY'\n",1)[1].split('\nCANONICAL_PRIDE_PY',1)[0]
        compile(source,'canonical-worker','exec')
    state=json.loads((tmp_path/'outputs/reproductions/zheng-pride-0shot-v1/jobs.json').read_text())
    assert len(state['groups'])==6 and all(g['job_id'] for g in state['groups'])
