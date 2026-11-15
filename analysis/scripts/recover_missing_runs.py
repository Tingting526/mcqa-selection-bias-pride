"""Recover missing 0-shot runs on LRZ. Supports streaming via python -.

--apply frees only pip cache and the completed Gemma model download cache,
downloads pinned remaining model files without Xet, and submits missing runs.
Saved scores, datasets, results and credentials are never removed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap

ROOT = Path.cwd() if __file__ == '<stdin>' else Path(__file__).resolve().parents[1]
DATASETS = ('medqa', 'medmcqa', 'mmlu', 'arc', 'csqa')
COUNTS = dict(medqa=1273, medmcqa=3941, mmlu=13581, arc=1165, csqa=1216)
MODELS = ('llama-3.1-8b-instruct', 'qwen2.5-7b-instruct',
          'glm-4-9b-chat-hf', 'mistral-7b-instruct-v0.3', 'qwen3-8b')
BLOCKED_LLAMA = {'5850222', '5850224', '5850226', '5850228', '5850230'}


def run(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True,
                          cwd=ROOT, **kwargs)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed():
    full, pilots = set(), set()
    for path in (ROOT/'outputs/campaigns').glob('**/pipeline.json'):
        state = read(path)
        c = state.get('config', {})
        if state.get('status') != 'completed' or c.get('scorer') != 'real':
            continue
        comparison = path.with_name('comparison.csv')
        if not comparison.exists():
            continue
        rows = list(csv.DictReader(comparison.open()))
        pair = (c['model'], c['dataset'])
        if c.get('limit') is None and len(rows) == 1:
            if (rows[0]['scorer_backend'] == 'real_transformers'
                    and int(float(rows[0]['baseline_n'])) == COUNTS[c['dataset']]
                    and int(float(rows[0]['pride_pride_n'])) == COUNTS[c['dataset']]):
                full.add(pair)
        elif c.get('limit') and rows:
            pilots.add(pair)
    for path in (ROOT/'outputs/zheng_selection_bias').glob('*/manifest.json'):
        state = read(path)
        c = state.get('config', {})
        summary = path.with_name('summary.csv')
        if (state.get('status') != 'completed' or not summary.exists()
                or c.get('scorer_backend') != 'real_transformers' or 0 not in c.get('shots', [])):
            continue
        rows = [r for r in csv.DictReader(summary.open()) if r['shots'] == '0' and r['domain'] == 'Overall']
        expected = {'perm' if c['dataset'] != 'csqa' else 'cyclic', 'noid', 'shuffle_both'}
        if {r['setting'] for r in rows} != expected:
            continue
        pair = (c['model'], c['dataset'])
        if c.get('limit') is None and all(int(r['n']) == COUNTS[c['dataset']] for r in rows):
            full.add(pair)
        elif c.get('limit') and rows:
            pilots.add(pair)
    return full, pilots


def queue():
    output = run(['squeue', '--me', '--noheader', '--format=%i|%T|%R|%j']).stdout
    rows = [line.strip().split('|', 3) for line in output.splitlines() if line.strip()]
    for row in rows:
        row[2] = row[2].strip('()')
    return rows


def assert_idle(rows, registered=()):
    for job, status, reason, name in rows:
        if job in BLOCKED_LLAMA and status == 'PENDING' and reason == 'DependencyNeverSatisfied':
            continue
        if job in registered:
            continue
        if name.startswith('mcq-'):
            raise RuntimeError(f'Unregistered active MCQ job {job} ({status}); avoid duplicate work/cache removal.')


def cleanup():
    full, _ = completed()
    if not all(('gemma-2-9b-it', d) in full for d in DATASETS):
        raise RuntimeError('Gemma must have complete saved results on all five datasets before evicting its weights.')
    # Only exact, project-owned, reproducible cache subdirectories are allowed.
    base = ROOT/'outputs/cache'
    if base.resolve() != base or not base.is_dir():
        raise RuntimeError('Expected an ordinary project cache directory, without symlinked parents.')
    paths = [base/'xdg/pip', base/'huggingface/hub/models--google--gemma-2-9b-it']
    for path in paths:
        if path.is_symlink() or (path.exists() and path.resolve() != path):
            raise RuntimeError(f'Refusing symlinked cache deletion: {path}')
        if path.exists():
            print(f'Removing reproducible cache only: {path}', flush=True)
            shutil.rmtree(path)
    # Write and fsync real data: df alone does not measure a per-user quota.
    with tempfile.TemporaryFile(dir=base) as stream:
        chunk = b'\0' * (1024 * 1024)
        for _ in range(64):
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    print('64 MiB quota write check passed.', flush=True)


def print_failed_baselines():
    for path in (ROOT/'outputs/campaigns').glob('**/pipeline.json'):
        state = read(path)
        if state.get('status') != 'failed' or state.get('config', {}).get('model') not in MODELS[:2]:
            continue
        log = path.with_name('baseline.log')
        if log.exists():
            lines = log.read_text(errors='replace').replace('\r', '\n').splitlines()
            lines = [line for line in lines if line.strip() and not line.startswith('Loading weights:')]
            print(f'\n=== Previous baseline error: {log} ===\n' + '\n'.join(lines[-14:]), flush=True)


def build_plan(full, pilots):
    entries, first_pilots = [], {}
    for model in MODELS:
        datasets = DATASETS if model == MODELS[0] else (('medqa', 'arc', 'csqa') if model == MODELS[1] else ('arc', 'csqa'))
        for dataset in datasets:
            pair = (model, dataset)
            if pair in full:
                continue
            pilot_index = None
            if pair not in pilots:
                pilot_index = len(entries)
                dependencies = [] if model not in first_pilots else [first_pilots[model]]
                entries.append(dict(model=model, dataset=dataset, stage='pilot',
                    limit=20 if dataset in DATASETS[:3] else 2,
                    dependency_indices=dependencies, job_id=None))
                first_pilots.setdefault(model, pilot_index)
            entries.append(dict(model=model, dataset=dataset, stage='full', limit=None,
                dependency_indices=[] if pilot_index is None else [pilot_index], job_id=None))
    return dict(protocol='zero-shot-recovery-v1', entries=entries)


def script(entry, directory, population=None):
    model, dataset = entry['model'], entry['dataset']
    if dataset in DATASETS[:3]:
        pipeline = directory/'medical'/f'{model}_{dataset}_{entry["stage"]}'
        config = dict(model=model, dataset=dataset, scorer='real', limit=entry['limit'],
                      batch_size=1, preprocessed=str(population))
        status = pipeline/'pipeline.json'
        if not status.exists():
            save(status, dict(config=config, steps={}))
        elif read(status)['config'] != config:
            raise RuntimeError('Saved pipeline configuration differs')
        args = [sys.executable, 'scripts/run_pipeline.py', '--model', model,
                '--dataset', dataset, '--scorer', 'real', '--batch-size', '1',
                '--preprocessed', str(population), '--resume', str(pipeline)]
        entry['pipeline'] = str(pipeline)
    else:
        args = [sys.executable, 'scripts/run_zheng_selection_bias.py', '--model', model,
                '--dataset', dataset, '--shots', '0']
    if entry['limit'] is not None:
        args += ['--limit', str(entry['limit'])]
    return '\n'.join(['#!/bin/bash', '#SBATCH --job-name=mcq-recovery',
        '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=8', '#SBATCH --mem=96G',
        '#SBATCH --time=08:00:00', '#SBATCH --output=outputs/logs/slurm/%x-%j.out',
        '#SBATCH --error=outputs/logs/slurm/%x-%j.err', 'set -euo pipefail',
        'cd '+shlex.quote(str(ROOT)), 'export MCQ_CONDA_ENV=mcq-analysis',
        'export HF_HUB_DISABLE_XET=1', 'export HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
        'source scripts/slurm/environment.sh',
        'echo '+shlex.quote(f'{model} / {dataset} / {entry["stage"]} / 0-shot'),
        'srun '+shlex.join(args), ''])


def submit(state, directory, partition, populations):
    state_path = directory/'recovery_jobs.json'
    for entry in state['entries']:
        if entry.get('submission_started'):
            raise RuntimeError('Prior submission outcome uncertain; inspect Slurm before retrying.')
        if entry.get('job_id'):
            print(f'Already submitted {entry["job_id"]}: {entry["model"]} {entry["dataset"]} {entry["stage"]}', flush=True)
            continue
        body = script(entry, directory, populations.get(entry['dataset']))
        args = ['sbatch', '--parsable', '--export=ALL', '--partition='+partition]
        dependencies = [state['entries'][i]['job_id'] for i in entry['dependency_indices']]
        if dependencies:
            args.append('--dependency=afterok:'+':'.join(dependencies))
        for key in ('QOS', 'ACCOUNT'):
            if os.environ.get(key):
                args.append('--'+key.lower()+'='+os.environ[key])
        entry['submission_started'] = True
        save(state_path, state)
        try:
            output = run(args, input=body).stdout.strip().split(';')[0]
        except subprocess.CalledProcessError as error:
            entry['submission_started'] = False
            save(state_path, state)
            raise RuntimeError(error.stderr) from error
        if not output.isdigit():
            raise RuntimeError('Unknown sbatch response; inspect Slurm before retrying.')
        entry.update(job_id=output, submission_started=False)
        save(state_path, state)
        print(f'Submitted {output}: {entry["model"]} / {entry["dataset"]} / {entry["stage"]}', flush=True)


def evict_other_models(keep_repo, configs):
    """Use the HF deletion strategy so shared blobs are handled by the library."""
    from huggingface_hub import scan_cache_dir
    hub = ROOT/'outputs/cache/huggingface/hub'
    if hub.resolve() != hub or Path(os.environ.get('HF_HUB_CACHE', hub)).resolve() != hub:
        raise RuntimeError('Expected the ordinary project-owned Hugging Face cache.')
    allowed = {c['model_repo'] for c in configs.values()} | {'google/gemma-2-9b-it'}
    allowed.discard(keep_repo)
    cached = scan_cache_dir(hub)
    hashes = [r.commit_hash for repo in cached.repos
              if repo.repo_type == 'model' and repo.repo_id in allowed for r in repo.revisions]
    if hashes:
        strategy = cached.delete_revisions(*hashes)
        print('HF-managed eviction of other model downloads: '+strategy.expected_freed_size_str, flush=True)
        strategy.execute()
    # Entire known model download folders may also contain interrupted downloads.
    for repo in sorted(allowed):
        path = hub/('models--'+repo.replace('/', '--'))
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError('Refusing symlinked cache deletion: '+str(path))
        if path.exists():
            shutil.rmtree(path)
    # Current CLI also cleans shared blobs left unreferenced by earlier cache eviction.
    hf = shutil.which('hf')
    if hf:
        help_result = subprocess.run([hf, 'cache', 'prune', '--help'], text=True, capture_output=True)
        if help_result.returncode == 0:
            output = run([hf, 'cache', 'prune', '--yes', '--cache-dir', str(hub)])
            print(output.stdout, flush=True)


def download_model(config):
    from huggingface_hub import snapshot_download
    patterns = ['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja']
    snapshot_download(config['model_repo'], revision=config['model_revision'],
                      max_workers=1, allow_patterns=patterns)
    if (config['tokenizer_repo'], config['tokenizer_revision']) != (config['model_repo'], config['model_revision']):
        snapshot_download(config['tokenizer_repo'], revision=config['tokenizer_revision'],
                          max_workers=1, allow_patterns=patterns[1:])


# Embedded in sbatch so streaming this controller does not require uploading a file.
SERIAL_WORKER = r'''
import fcntl, json, os, subprocess, sys, traceback
from pathlib import Path

def write_status(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)

def execute_task(plan, dataset, stage):
    root, directory = Path(plan['root']), Path(plan['directory'])
    model = plan['model']
    limit = (20 if dataset in ('medqa', 'medmcqa', 'mmlu') else 2) if stage == 'pilot' else None
    log = directory/f'{dataset}-{stage}.log'
    if dataset in ('medqa', 'medmcqa', 'mmlu'):
        pipeline = directory/f'{model}_{dataset}_{stage}'
        pipeline.mkdir(parents=True, exist_ok=True)
        config = dict(model=model, dataset=dataset, scorer='real', limit=limit,
                      batch_size=1, preprocessed=plan['populations'][dataset])
        state = pipeline/'pipeline.json'
        if not state.exists():
            state.write_text(json.dumps(dict(config=config, steps={})))
        elif json.loads(state.read_text())['config'] != config:
            raise RuntimeError('Saved pipeline configuration differs')
        args = [sys.executable, 'scripts/run_pipeline.py', '--model', model, '--dataset', dataset,
                '--scorer', 'real', '--batch-size', '1', '--preprocessed', config['preprocessed'],
                '--resume', str(pipeline)]
    else:
        args = [sys.executable, 'scripts/run_zheng_selection_bias.py', '--model', model,
                '--dataset', dataset, '--shots', '0']
    if limit is not None:
        args += ['--limit', str(limit)]
    print(f'RUN {model} / {dataset} / {stage} / 0-shot; log={log}', flush=True)
    with log.open('a') as stream:
        result = subprocess.run(args, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]), flush=True)
        raise RuntimeError(f'{model}/{dataset}/{stage} failed, exit {result.returncode}')

def run_worker(path):
    plan = json.loads(Path(path).read_text())
    root, directory = Path(plan['root']), Path(plan['directory'])
    os.chdir(root)
    sys.path.insert(0, str(root/'scripts'))
    import _bootstrap
    from huggingface_hub import scan_cache_dir, snapshot_download
    lock_path = root/'outputs/cache/.serial-recovery.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # These five model jobs are chained afterany and share this lock. No
        # other active MCQ job is allowed to use this cache during eviction.
        result = subprocess.run(['squeue', '--me', '--noheader', '--format=%i|%T|%j'],
                                check=True, text=True, capture_output=True)
        for row in result.stdout.splitlines():
            job, state, name = row.strip().split('|', 2)
            if (state in ('RUNNING', 'COMPLETING') and name.startswith('mcq-')
                    and job != os.environ.get('SLURM_JOB_ID')):
                raise RuntimeError('Other MCQ job active; refusing model-cache eviction: '+job)
        hub = root/'outputs/cache/huggingface/hub'
        if hub.resolve() != hub or Path(os.environ['HF_HUB_CACHE']).resolve() != hub:
            raise RuntimeError('Unexpected model cache location')
        other = set(plan['repos']) - {plan['config']['model_repo']}
        cached = scan_cache_dir(hub)
        hashes = [r.commit_hash for repo in cached.repos
                  if repo.repo_type == 'model' and repo.repo_id in other for r in repo.revisions]
        if hashes:
            strategy = cached.delete_revisions(*hashes)
            print('Evicting previous model cache: '+strategy.expected_freed_size_str, flush=True)
            strategy.execute()
        import shutil
        for repo in other:
            candidate = hub/('models--'+repo.replace('/', '--'))
            if candidate.is_symlink() or candidate.resolve() != candidate:
                raise RuntimeError('Symlinked model cache')
            if candidate.exists():
                shutil.rmtree(candidate)
        c = plan['config']
        patterns = ['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja']
        print('Downloading only this model: '+plan['model'], flush=True)
        snapshot_download(c['model_repo'], revision=c['model_revision'], max_workers=1,
                          allow_patterns=patterns)
        if (c['tokenizer_repo'], c['tokenizer_revision']) != (c['model_repo'], c['model_revision']):
            snapshot_download(c['tokenizer_repo'], revision=c['tokenizer_revision'], max_workers=1,
                              allow_patterns=patterns[1:])
        status_path = directory/'worker_status.json'
        status = dict(model=plan['model'], job_id=os.environ.get('SLURM_JOB_ID'), tasks=[])
        failures = 0
        for task in plan['tasks']:
            entry = dict(dataset=task['dataset'], status='running')
            status['tasks'].append(entry)
            write_status(status_path, status)
            try:
                if task['pilot_needed']:
                    execute_task(plan, task['dataset'], 'pilot')
                    entry['pilot'] = 'completed'
                execute_task(plan, task['dataset'], 'full')
                entry['status'] = 'completed'
            except Exception as error:
                failures += 1
                entry.update(status='failed', error=str(error))
                print(traceback.format_exc(), flush=True)
            write_status(status_path, status)
        raise SystemExit(1 if failures else 0)
'''


def serial_main(args, full, pilots):
    directory = args.directory/'serial'
    registry = directory/'serial_jobs.json'
    pending = build_plan(full, pilots)['entries']
    models = [model for model in MODELS if any(e['model'] == model for e in pending)]
    state = read(registry) if registry.exists() else dict(protocol='serial-zero-shot-v2',
        groups=[dict(model=model, job_id=None) for model in models])
    if state.get('protocol') != 'serial-zero-shot-v2':
        raise RuntimeError('Serial registry differs')
    print('Serial storage plan: '+str(len(state['groups']))+' model jobs, one model cache at a time.', flush=True)
    if not args.apply:
        return
    if not state['groups']:
        return
    rows = queue()
    registered = {g['job_id'] for g in state['groups'] if g.get('job_id')}
    assert_idle(rows, registered)
    for group in state['groups']:
        if group.get('submission_started'):
            raise RuntimeError('Uncertain earlier submission; inspect Slurm before retry.')
    partitions = run(['sinfo', '-h', '-o', '%P']).stdout
    if args.partition not in [p.rstrip('*') for p in partitions.split()]:
        raise RuntimeError('Partition not available')
    sys.path.insert(0, str(ROOT/'scripts'))
    from src.utils.config import named_config
    from start_gemma_all_datasets import artifact
    from start_medqa_campaign import check_access
    from src.experimental.zheng_selection_bias import source_digest, load_original
    configs = {m: named_config('models', m) for m in MODELS}
    populations = {d: str(artifact(d)) for d in DATASETS[:3]}
    for d in DATASETS[3:]:
        source_digest(d)
    load_original()
    if not registered:
        if not all(('gemma-2-9b-it', d) in full for d in DATASETS):
            raise RuntimeError('Gemma completion must be confirmed before cache cleanup')
        evict_other_models(configs[state['groups'][0]['model']]['model_repo'], configs)
        # Full download of the first model must fit before GPUs are submitted.
        first = state['groups'][0]['model']
        check_access(first)
        download_model(configs[first])
    remaining = [g['model'] for g in state['groups'] if not g.get('job_id')]
    for model in remaining:
        check_access(model)
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT/'outputs/logs/slurm').mkdir(parents=True, exist_ok=True)
    save(registry, state)
    for job, status, reason, name in rows:
        if job in BLOCKED_LLAMA and status == 'PENDING' and reason == 'DependencyNeverSatisfied':
            run(['scancel', job])
    previous = None
    for group in state['groups']:
        model = group['model']
        if group.get('job_id'):
            previous = group['job_id']
            print(f'Already submitted {previous}: {model}', flush=True)
            continue
        tasks = [dict(dataset=e['dataset'], pilot_needed=(model, e['dataset']) not in pilots)
                 for e in pending if e['model'] == model and e['stage'] == 'full']
        group_dir = directory/model
        plan_path = group_dir/'plan.json'
        plan = dict(root=str(ROOT), directory=str(group_dir), model=model, tasks=tasks,
                    config=configs[model], populations=populations,
                    repos=[c['model_repo'] for c in configs.values()]+['google/gemma-2-9b-it'])
        save(plan_path, plan)
        body = '\n'.join(['#!/bin/bash', '#SBATCH --job-name=mcq-serial',
            '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=8', '#SBATCH --mem=96G',
            '#SBATCH --time=08:00:00', '#SBATCH --output=outputs/logs/slurm/%x-%j.out',
            '#SBATCH --error=outputs/logs/slurm/%x-%j.err', 'set -euo pipefail',
            'cd '+shlex.quote(str(ROOT)), 'export MCQ_CONDA_ENV=mcq-analysis',
            'export HF_HUB_DISABLE_XET=1', 'export HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
            'unset LIMIT RESUME PRIOR PREPROCESSED SBATCH_DEPENDENCY',
            'source scripts/slurm/environment.sh',
            'srun '+shlex.quote(sys.executable)+" - <<'SERIAL_RECOVERY_PY'",
            textwrap.dedent(SERIAL_WORKER), 'run_worker('+repr(str(plan_path))+')',
            'SERIAL_RECOVERY_PY', ''])
        command = ['sbatch', '--parsable', '--export=ALL', '--partition='+args.partition]
        if previous:
            command += ['--dependency=afterany:'+previous]
        for key in ('QOS', 'ACCOUNT'):
            if os.environ.get(key):
                command += ['--'+key.lower()+'='+os.environ[key]]
        group['submission_started'] = True
        save(registry, state)
        try:
            output = run(command, input=body).stdout.strip().split(';')[0]
        except subprocess.CalledProcessError as error:
            group['submission_started'] = False
            save(registry, state)
            raise RuntimeError(error.stderr) from error
        if not output.isdigit():
            raise RuntimeError('Unknown sbatch output; inspect Slurm before retry.')
        group.update(job_id=output, submission_started=False, previous_model_job=previous)
        previous = output
        save(registry, state)
        print(f'Submitted model job {output}: {model}, datasets='+
              ','.join(t['dataset'] for t in tasks), flush=True)
    print('Serial registry: '+str(registry), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--serial', action='store_true', help='One model cache and one model job at a time.')
    parser.add_argument('--partition', default='lrz-dgx-a100-80x8')
    parser.add_argument('--directory', type=Path, default=ROOT/'outputs/campaigns/recovery-20261015')
    args = parser.parse_args()
    os.chdir(ROOT)
    # Set before importing huggingface_hub, including helper imports.
    os.environ.update(MCQ_CONDA_ENV='mcq-analysis', HF_HUB_DISABLE_XET='1',
                      HF_XET_CHUNK_CACHE_SIZE_BYTES='0')
    for key in ('LIMIT', 'RESUME', 'PRIOR', 'PREPROCESSED', 'SBATCH_DEPENDENCY'):
        os.environ.pop(key, None)
    full, pilots = completed()
    print_failed_baselines()
    if args.serial:
        return serial_main(args, full, pilots)
    path = args.directory/'recovery_jobs.json'
    state = read(path) if path.exists() else build_plan(full, pilots)
    if state.get('protocol') != 'zero-shot-recovery-v1':
        raise RuntimeError('Recovery registry protocol differs')
    print(f'{len(full)} completed full pairs. Recovery plan: '+
          f'{sum(e["stage"]=="pilot" for e in state["entries"])} pilots and '+
          f'{sum(e["stage"]=="full" for e in state["entries"])} full runs.', flush=True)
    if not args.apply:
        print('Read-only check: nothing deleted or submitted. Use --apply on LRZ.')
        return
    if not state['entries']:
        print('All planned pairs are already complete; nothing to recover.')
        return
    for entry in state['entries']:
        if entry.get('submission_started'):
            raise RuntimeError('Uncertain prior submission; inspect Slurm before recovery.')
    rows = queue()
    registered = {e['job_id'] for e in state['entries'] if e.get('job_id')}
    assert_idle(rows, registered)
    partitions = run(['sinfo', '-h', '-o', '%P']).stdout
    if args.partition not in [p.rstrip('*') for p in partitions.split()]:
        raise RuntimeError('Partition not found')
    if not registered:
        cleanup()
    sys.path.insert(0, str(ROOT/'scripts'))
    from start_gemma_all_datasets import artifact
    from start_medqa_campaign import check_access
    from src.experimental.zheng_selection_bias import source_digest, load_original
    from src.utils.config import named_config
    populations = {d: artifact(d) for d in DATASETS[:3]}
    for dataset in DATASETS[3:]:
        source_digest(dataset)
    load_original()
    needed = sorted({e['model'] for e in state['entries'] if not e.get('job_id')})
    from huggingface_hub import snapshot_download
    for model in needed:
        check_access(model)
        c = named_config('models', model)
        print(f'Checking/downloading pinned weights sequentially: {model}', flush=True)
        snapshot_download(c['model_repo'], revision=c['model_revision'], max_workers=1,
            allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja'])
        if (c.get('tokenizer_repo', c['model_repo']), c.get('tokenizer_revision')) != (c['model_repo'], c['model_revision']):
            snapshot_download(c['tokenizer_repo'], revision=c['tokenizer_revision'], max_workers=1,
                allow_patterns=['*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja'])
    # No GPUs are queued until the weights fit in the user quota.
    args.directory.mkdir(parents=True, exist_ok=True)
    save(path, state)
    (ROOT/'outputs/logs/slurm').mkdir(parents=True, exist_ok=True)
    for job, status, reason, name in rows:
        if job in BLOCKED_LLAMA and status == 'PENDING' and reason == 'DependencyNeverSatisfied':
            run(['scancel', job])
            print(f'Cancelled obsolete blocked dependency job {job}', flush=True)
    submit(state, args.directory, args.partition, populations)
    print(f'Registry: {path}', flush=True)


if __name__ == '__main__':
    main()
