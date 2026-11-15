import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery():
    path = Path(__file__).resolve().parents[1]/'scripts/recover_missing_runs.py'
    spec = importlib.util.spec_from_file_location('recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_covers_exactly_missing_full_pairs_and_reuses_pilots(recovery):
    full = {('gemma-2-9b-it', d) for d in recovery.DATASETS}
    full |= {(m, d) for m in recovery.MODELS[1:] for d in recovery.DATASETS[:3]}
    full.remove(('qwen2.5-7b-instruct', 'medqa'))
    pilots = {('qwen3-8b', d) for d in ('arc', 'csqa')}
    state = recovery.build_plan(full, pilots)
    assert len(state['entries']) == 26
    full_entries = [e for e in state['entries'] if e['stage'] == 'full']
    assert len(full_entries) == 14
    assert all(e['limit'] is None for e in full_entries)
    assert not any((e['model'], e['dataset']) in full for e in state['entries'])
    assert not any(e['stage'] == 'pilot' and e['model'] == 'qwen3-8b' for e in state['entries'])
    for i, entry in enumerate(state['entries']):
        assert all(dep < i for dep in entry['dependency_indices'])
        if entry['stage'] == 'full' and entry['model'] != 'qwen3-8b':
            pilot = state['entries'][entry['dependency_indices'][0]]
            assert pilot['stage'] == 'pilot'
            assert (pilot['model'], pilot['dataset']) == (entry['model'], entry['dataset'])


def test_reference_batch_explicitly_zero_shot_and_full_population(recovery, tmp_path):
    entry = dict(model='llama-3.1-8b-instruct', dataset='arc', stage='full', limit=None)
    body = recovery.script(entry, tmp_path)
    assert '--shots 0' in body
    assert '--limit' not in body
    assert 'MCQ_CONDA_ENV=mcq-analysis' in body
    assert 'HF_HUB_DISABLE_XET=1' in body


def test_medical_plan_keeps_pinned_population_and_no_question_limit(recovery, tmp_path):
    entry = dict(model='qwen2.5-7b-instruct', dataset='medqa', stage='full', limit=None)
    population = tmp_path/'population'
    body = recovery.script(entry, tmp_path, population)
    assert '--limit' not in body and '--scorer real' in body
    state = json.loads((Path(entry['pipeline'])/'pipeline.json').read_text())
    assert state['config']['preprocessed'] == str(population)
    assert state['config']['limit'] is None


def test_submission_retry_skips_existing_jobs_and_preserves_dependencies(recovery, tmp_path, monkeypatch):
    state = recovery.build_plan(set(), set())
    state['entries'] = state['entries'][:2]
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=f'{100 + len(calls)}\n')
    monkeypatch.setattr(recovery, 'run', run)
    recovery.submit(state, tmp_path, 'gpu-test', {'medqa': tmp_path/'population'})
    recovery.submit(state, tmp_path, 'gpu-test', {'medqa': tmp_path/'population'})
    assert len(calls) == 2
    assert '--dependency=afterok:101' in calls[1][0]
    assert state['entries'][1]['job_id'] == '102'
    assert not state['entries'][1]['submission_started']
    state['entries'][0]['submission_started'] = True
    with pytest.raises(RuntimeError, match='uncertain'):
        recovery.submit(state, tmp_path, 'gpu-test', {})
    assert len(calls) == 2


def test_cache_cleanup_requires_complete_gemma_and_keeps_results(recovery, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'completed', lambda: (set(), set()))
    with pytest.raises(RuntimeError, match='Gemma must'):
        recovery.cleanup()
    monkeypatch.setattr(recovery, 'completed', lambda: (
        {('gemma-2-9b-it', d) for d in recovery.DATASETS}, set()))
    base = tmp_path/'outputs/cache'
    for child in ('xdg/pip', 'huggingface/hub/models--google--gemma-2-9b-it',
                  'huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct'):
        path = base/child
        path.mkdir(parents=True)
        (path/'cache.bin').write_bytes(b'cache')
    result = tmp_path/'outputs/scores.json'
    result.write_text('saved results')
    recovery.cleanup()
    assert not (base/'xdg/pip').exists()
    assert not (base/'huggingface/hub/models--google--gemma-2-9b-it').exists()
    assert (base/'huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/cache.bin').exists()
    assert result.read_text() == 'saved results'


def test_active_job_guards_and_parenthesized_dependency_reason(recovery, monkeypatch):
    monkeypatch.setattr(recovery, 'run', lambda args: SimpleNamespace(
        stdout='5850222|PENDING|(DependencyNeverSatisfied)|mcq-analysis\n'))
    rows = recovery.queue()
    recovery.assert_idle(rows)
    with pytest.raises(RuntimeError, match='active MCQ job'):
        recovery.assert_idle([['123', 'RUNNING', 'node01', 'mcq-analysis']])


def test_unknown_sbatch_outcome_remains_blocked(recovery, tmp_path, monkeypatch):
    state = dict(entries=[dict(model='qwen3-8b', dataset='arc', stage='full',
                              limit=None, job_id=None, dependency_indices=[])])
    monkeypatch.setattr(recovery, 'run', lambda *a, **k: SimpleNamespace(stdout='unexpected response'))
    with pytest.raises(RuntimeError, match='Unknown sbatch'):
        recovery.submit(state, tmp_path, 'gpu-test', {})
    saved = json.loads((tmp_path/'recovery_jobs.json').read_text())
    assert saved['entries'][0]['submission_started']


def test_model_cache_eviction_keeps_current_model_and_saved_results(recovery, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    hub = tmp_path/'outputs/cache/huggingface/hub'
    current = hub/'models--meta-llama--Llama-3.1-8B-Instruct'
    other = hub/'models--THUDM--glm-4-9b-chat-hf'
    foreign = hub/'models--other--keep'
    for directory in (current, other, foreign):
        directory.mkdir(parents=True)
        (directory/'weights').write_text('cached weights')
    results = tmp_path/'outputs/results.json'
    results.write_text('real results')
    hashes, executions = [], []
    cache = SimpleNamespace(repos=[
        SimpleNamespace(repo_type='model', repo_id='meta-llama/Llama-3.1-8B-Instruct',
                        revisions=[SimpleNamespace(commit_hash='keep-revision')]),
        SimpleNamespace(repo_type='model', repo_id='THUDM/glm-4-9b-chat-hf',
                        revisions=[SimpleNamespace(commit_hash='remove-revision')])])
    def delete(*values):
        hashes.extend(values)
        return SimpleNamespace(expected_freed_size_str='18GB', execute=lambda: executions.append(True))
    cache.delete_revisions = delete
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(scan_cache_dir=lambda p: cache))
    monkeypatch.setenv('HF_HUB_CACHE', str(hub))
    monkeypatch.setattr(recovery.shutil, 'which', lambda name: None)
    recovery.evict_other_models('meta-llama/Llama-3.1-8B-Instruct', {
        'llama': dict(model_repo='meta-llama/Llama-3.1-8B-Instruct'),
        'glm': dict(model_repo='THUDM/glm-4-9b-chat-hf')})
    assert hashes == ['remove-revision'] and executions == [True]
    assert current.exists() and foreign.exists() and not other.exists()
    assert results.read_text() == 'real results'


def test_serial_worker_skips_full_after_failed_pilot_but_continues_next_dataset(recovery, tmp_path, monkeypatch):
    hub = tmp_path/'outputs/cache/huggingface/hub'
    hub.mkdir(parents=True)
    group = tmp_path/'group'
    group.mkdir()
    config = dict(model_repo='repo/model', tokenizer_repo='repo/model',
                  model_revision='revision', tokenizer_revision='revision')
    plan = dict(root=str(tmp_path), directory=str(group), model='llama-3.1-8b-instruct',
                config=config, repos=['repo/model'], populations={}, tasks=[
                    dict(dataset='arc', pilot_needed=True),
                    dict(dataset='csqa', pilot_needed=True)])
    path = group/'plan.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setitem(sys.modules, '_bootstrap', SimpleNamespace())
    downloads = []
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        scan_cache_dir=lambda p: SimpleNamespace(repos=[]),
        snapshot_download=lambda *a, **k: downloads.append((a,k))))
    monkeypatch.setenv('HF_HUB_CACHE', str(hub))
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(recovery.SERIAL_WORKER, namespace)
    monkeypatch.setattr(namespace['subprocess'], 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    calls=[]
    def task(plan, dataset, stage):
        calls.append((dataset, stage))
        if dataset=='arc':
            raise RuntimeError('pilot failed')
    namespace['execute_task'] = task
    with pytest.raises(SystemExit) as error:
        namespace['run_worker'](str(path))
    assert error.value.code == 1
    assert calls == [('arc','pilot'), ('csqa','pilot'), ('csqa','full')]
    status=json.loads((group/'worker_status.json').read_text())
    assert [t['status'] for t in status['tasks']] == ['failed','completed']
    assert len(downloads)==1 and downloads[0][1]['max_workers']==1


def test_medical_pipeline_ids_do_not_collide_between_models_or_datasets(recovery, tmp_path):
    paths=[]
    for model,dataset in [('qwen2.5-7b-instruct','medqa'), ('llama-3.1-8b-instruct','medqa'),
                          ('llama-3.1-8b-instruct','mmlu')]:
        entry=dict(model=model,dataset=dataset,stage='full',limit=None)
        recovery.script(entry,tmp_path,tmp_path/'population')
        paths.append(Path(entry['pipeline']).name)
    assert len(set(paths)) == 3


def test_serial_submission_queues_five_models_afterany_without_duplicates(recovery, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    full = {('gemma-2-9b-it', d) for d in recovery.DATASETS}
    full |= {(m,d) for m in recovery.MODELS[1:] for d in recovery.DATASETS[:3]}
    full.remove(('qwen2.5-7b-instruct','medqa'))
    pilots={('qwen3-8b',d) for d in ('arc','csqa')}
    monkeypatch.setattr(recovery,'queue',lambda:[])
    downloads=[]
    monkeypatch.setattr(recovery,'evict_other_models',lambda *a:None)
    monkeypatch.setattr(recovery,'download_model',lambda c:downloads.append(c['model_repo']))
    configs={m:dict(model_repo='repo/'+m, tokenizer_repo='repo/'+m,
                    model_revision='revision',tokenizer_revision='revision') for m in recovery.MODELS}
    monkeypatch.setitem(sys.modules,'src.utils.config',SimpleNamespace(named_config=lambda kind,m:configs[m]))
    monkeypatch.setitem(sys.modules,'start_gemma_all_datasets',SimpleNamespace(artifact=lambda d:tmp_path/d))
    monkeypatch.setitem(sys.modules,'start_medqa_campaign',SimpleNamespace(check_access=lambda m:None))
    monkeypatch.setitem(sys.modules,'src.experimental.zheng_selection_bias',SimpleNamespace(
        source_digest=lambda d:None,load_original=lambda:None))
    calls=[]
    def run(args, **kwargs):
        if args[0]=='sinfo':return SimpleNamespace(stdout='gpu-test*')
        assert args[0]=='sbatch'
        calls.append((args,kwargs['input']))
        return SimpleNamespace(stdout=str(100+len(calls)))
    monkeypatch.setattr(recovery,'run',run)
    args=SimpleNamespace(directory=tmp_path/'recovery',partition='gpu-test',apply=True)
    recovery.serial_main(args,full,pilots)
    recovery.serial_main(args,full,pilots)
    assert len(calls)==5
    assert downloads==['repo/llama-3.1-8b-instruct']
    for i,(command,body) in enumerate(calls):
        if i:assert f'--dependency=afterany:{100+i}' in command
        assert 'HF_HUB_DISABLE_XET=1' in body
        source=body.split("<<'SERIAL_RECOVERY_PY'\n",1)[1].split('\nSERIAL_RECOVERY_PY',1)[0]
        compile(source,'submitted-worker','exec')
    registry=json.loads((tmp_path/'recovery/serial/serial_jobs.json').read_text())
    assert all(g['job_id'] for g in registry['groups'])
