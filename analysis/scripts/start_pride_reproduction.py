"""Queue the canonical six-model/five-dataset 0-shot PriDe reproduction.

Compatible reference scores are reused; medical/MMLU chat observations are not.
Six serial GPU jobs keep only one model in the download cache at a time.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path.cwd() if __file__ == '<stdin>' else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import _bootstrap
from recover_missing_runs import run, save, read, queue, SERIAL_WORKER
from src.pride.zheng import VERSION, original_debias
from src.utils.config import named_config
from src.experimental.zheng_selection_bias import load_original, source_digest
from reproduce_pride import MODELS, COUNTS, compatible_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--partition', default='lrz-dgx-a100-80x8')
    args = p.parse_args()
    root = ROOT/'outputs/reproductions'/VERSION
    path = root/'jobs.json'
    state = read(path) if path.exists() else dict(implementation=VERSION,
        groups=[dict(model=m,job_id=None) for m in MODELS])
    if state.get('implementation') != VERSION:
        raise ValueError('Existing reproduction registry differs')
    original_debias()
    load_original()
    for dataset in ('mmlu','arc','csqa'):
        source_digest(dataset)
    from start_gemma_all_datasets import artifact
    populations = {d:str(artifact(d)) for d in ('medqa','medmcqa','mmlu')}
    for model in MODELS:
        reused = [d for d in COUNTS if compatible_sources(root,model,d)]
        print(model+': reusable raw-reference scores='+','.join(reused or ['none']),flush=True)
    if not args.submit:
        print('Plan only. No inference or submissions. Add --submit on LRZ.')
        return
    partitions = run(['sinfo','-h','-o','%P']).stdout
    if args.partition not in [p.rstrip('*') for p in partitions.split()]:
        raise ValueError('Selected partition unavailable')
    registered = {g['job_id'] for g in state['groups'] if g.get('job_id')}
    old = ROOT/'outputs/campaigns/recovery-20261015/serial/serial_jobs.json'
    old_ids = [g['job_id'] for g in read(old)['groups'] if g.get('job_id')] if old.exists() else []
    rows = queue()
    for job,status,reason,name in rows:
        if name.startswith('mcq-') and job not in registered and job not in old_ids:
            raise RuntimeError('Unknown active MCQ job; avoid shared-cache conflicts: '+job)
    active_old = {row[0] for row in rows if row[0] in old_ids}
    # The old chain already serializes all its predecessors; depend on its tail.
    predecessor = old_ids[-1] if active_old else None
    if active_old and predecessor not in active_old:
        raise RuntimeError('Old chain tail absent while predecessors remain active; inspect queue')
    configs = {m:named_config('models',m) for m in MODELS}
    root.mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/logs/slurm').mkdir(parents=True,exist_ok=True)
    if not registered:
        for model in MODELS:
            subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_pride.py'),
                '--model',model,'--postprocess-only','--output-root',str(root)],cwd=ROOT,check=True)
    save(path,state)
    override = '''
def execute_task(plan, dataset, stage):
    if stage != 'full':
        raise RuntimeError('Canonical reproduction has no limited-question pilot output')
    log = Path(plan['directory'])/(dataset+'-reproduction.log')
    command = [sys.executable, 'scripts/reproduce_pride.py', '--model', plan['model'],
               '--datasets', dataset, '--output-root', plan['output_root']]
    print('Reproducing '+plan['model']+'/'+dataset+'; log='+str(log), flush=True)
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=plan['root'], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\\n'.join(log.read_text(errors='replace').splitlines()[-25:]), flush=True)
        raise RuntimeError('Reproduction failed: '+dataset)
'''
    for group in state['groups']:
        if group.get('submission_started'):
            raise RuntimeError('Uncertain prior submission; inspect Slurm before retry')
        if group.get('job_id'):
            predecessor = group['job_id']
            print('Already submitted '+predecessor+': '+group['model'],flush=True)
            continue
        model=group['model']
        directory=root/'workers'/model
        plan_path=directory/'plan.json'
        plan=dict(root=str(ROOT),directory=str(directory),output_root=str(root),
            model=model,config=configs[model],populations=populations,
            repos=[c['model_repo'] for c in configs.values()],
            tasks=[dict(dataset=d,pilot_needed=False) for d in ('arc','csqa','medqa','medmcqa','mmlu')])
        save(plan_path,plan)
        body='\n'.join(['#!/bin/bash','#SBATCH --job-name=mcq-pride-repro',
            '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=8','#SBATCH --mem=96G',
            '#SBATCH --time=08:00:00','#SBATCH --output=outputs/logs/slurm/%x-%j.out',
            '#SBATCH --error=outputs/logs/slurm/%x-%j.err','set -euo pipefail',
            'cd '+shlex.quote(str(ROOT)), 'export MCQ_CONDA_ENV=mcq-analysis',
            'export HF_HUB_DISABLE_XET=1','export HF_XET_CHUNK_CACHE_SIZE_BYTES=0',
            'unset LIMIT RESUME PRIOR PREPROCESSED SBATCH_DEPENDENCY',
            'source scripts/slurm/environment.sh',
            'srun '+shlex.quote(sys.executable)+" - <<'CANONICAL_PRIDE_PY'",
            SERIAL_WORKER,override,'run_worker('+repr(str(plan_path))+')','CANONICAL_PRIDE_PY',''])
        command=['sbatch','--parsable','--export=ALL','--partition='+args.partition]
        if predecessor:
            command += ['--dependency=afterany:'+predecessor]
        for key in ('QOS','ACCOUNT'):
            if os.environ.get(key):command+=['--'+key.lower()+'='+os.environ[key]]
        group['submission_started']=True
        save(path,state)
        try:
            job=run(command,input=body).stdout.strip().split(';')[0]
        except subprocess.CalledProcessError as error:
            group['submission_started']=False
            save(path,state)
            raise RuntimeError(error.stderr) from error
        if not job.isdigit():raise RuntimeError('Unknown sbatch response; inspect Slurm before retry')
        group.update(job_id=job,submission_started=False,afterany=predecessor)
        predecessor=job
        save(path,state)
        print('Submitted '+job+': '+model+' / all five datasets / 0-shot / five calibration draws',flush=True)
    print('Registry: '+str(path),flush=True)


if __name__=='__main__':
    main()
