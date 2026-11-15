"""Submit Gemma or Llama pilots and dependent full runs across all five benchmarks.

Run on LRZ with --submit. Without --submit, validates inputs and prepares plans.
Previously registered MedMCQA/MMLU pilots can be reused. Full runs depend
on successful pilot exit codes. Repeated calls skip registered job IDs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from _bootstrap import PROJECT_ROOT
from run_campaign import plan, job_environment
from start_medqa_campaign import check_access
from src.data.preprocessing import load_preprocessed_artifact
from src.experimental.zheng_selection_bias import source_digest, load_original
from src.utils.config import named_config
from src.utils.io import write_json

MODEL = 'gemma-2-9b-it'
MODELS = (MODEL, 'llama-3.1-8b-instruct')
DATASETS = ('medqa', 'medmcqa', 'mmlu', 'arc', 'csqa')


def artifact(dataset):
    version = named_config('experiments', 'thesis')['prompt']['version']
    candidates = []
    for file in (PROJECT_ROOT/'outputs/preprocessed'/dataset).glob('*/manifest.json'):
        m = json.loads(file.read_text())
        if (m.get('prompt_template_version') == version and m.get('selection_limit') is None
                and m.get('selection_start_index') == 0):
            candidates.append(file.parent)
    if len(candidates) != 1:
        raise ValueError(f'Expected one complete {version} artifact for {dataset}')
    samples, _ = load_preprocessed_artifact(candidates[0])
    print(f'{dataset}: {len(samples)} validated questions')
    return candidates[0]


def medical_entry(directory, dataset, population, limit, model=MODEL):
    path = directory/'campaign.json'
    if path.exists():
        state = json.loads(path.read_text())
    else:
        state = plan(directory, [model], [dataset], limit=limit, batch_size=1,
                     preprocessed=population)
    if len(state['entries']) != 1:
        raise ValueError(f'Expected a single model entry in {directory}')
    entry = state['entries'][0]
    expected = dict(model=model, dataset=dataset, scorer='real', limit=limit,
                    batch_size=1, preprocessed=str(population.resolve()))
    if entry['config'] != expected:
        raise ValueError(f'Campaign configuration differs: {directory}')
    if entry.get('submission_started'):
        raise ValueError(f'Uncertain prior submission in {directory}; inspect Slurm first')
    return entry


def registry(directory, model):
    return Path(directory)/('gemma_jobs.json' if model == MODEL else 'llama_jobs.json')


def prepare(directory, reuse_pilots, model=MODEL):
    directory = Path(directory).resolve()
    populations = {d: artifact(d) for d in DATASETS[:3]}
    for d in DATASETS[3:]:
        source_digest(d)
    load_original()  # Verify hashes and required reference imports before GPU submission.
    state_path = registry(directory, model)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('model') != model or state.get('datasets') != list(DATASETS):
            raise ValueError('Saved model plan differs')
        return state
    state = dict(model=model, datasets=list(DATASETS), entries=[])
    for ds in DATASETS:
        if ds in populations:
            existing = Path(reuse_pilots)/ds
            pilot_dir = existing if (existing/'campaign.json').exists() else directory/'pilots'/ds
            pilot = medical_entry(pilot_dir, ds, populations[ds], 20, model)
            full = medical_entry(directory/'full'/ds, ds, populations[ds], None, model)
            for stage, entry in [('pilot', pilot), ('full', full)]:
                state['entries'].append(dict(dataset=ds, stage=stage, kind='analysis',
                    config=entry['config'], pipeline=entry['pipeline'], job_id=entry['job_id']))
        else:
            for stage, limit in [('pilot', 2), ('full', None)]:
                state['entries'].append(dict(dataset=ds, stage=stage, kind='zheng',
                    config=dict(model=model, dataset=ds, limit=limit), job_id=None))
    write_json(state_path, state)
    return state


def save_submission(directory, state, entry):
    """Keep both submission registries consistent to prevent duplicate jobs."""
    write_json(registry(directory, state['model']), state, overwrite=True)
    if entry['kind'] == 'analysis':
        campaign_path = Path(entry['pipeline']).parent/'campaign.json'
        campaign = json.loads(campaign_path.read_text())
        original = next(e for e in campaign['entries'] if e['pipeline'] == entry['pipeline'])
        for key in ('job_id', 'submission_started', 'partition', 'depends_on_pilot'):
            if key in entry:
                original[key] = entry[key]
        write_json(campaign_path, campaign, overwrite=True)


def submit(directory, state, partition):
    result = subprocess.run(['sinfo', '-h', '-o', '%P'], check=True,
                            text=True, capture_output=True)
    if partition not in [p.rstrip('*') for p in result.stdout.split()]:
        raise ValueError('Selected GPU partition is not present in sinfo')
    (PROJECT_ROOT/'outputs/logs/slurm').mkdir(parents=True, exist_ok=True)
    pilots = {}
    for entry in state['entries']:
        ds, stage = entry['dataset'], entry['stage']
        if entry.get('submission_started'):
            raise ValueError('Previous submission outcome uncertain; inspect squeue/sacct before retry')
        if not entry['job_id']:
            env = os.environ.copy()
            for key in ('LIMIT', 'RESUME', 'PRIOR', 'PREPROCESSED', 'SBATCH_DEPENDENCY'):
                env.pop(key, None)
            if entry['kind'] == 'analysis':
                env = job_environment(entry)
                env.pop('SBATCH_DEPENDENCY', None)
            else:
                env.update(MODEL=state['model'], DATASET=ds)
                if entry['config']['limit'] is not None:
                    env['LIMIT'] = str(entry['config']['limit'])
            command = ['sbatch', '--parsable', f'--partition={partition}']
            if stage == 'full':
                command += [f'--dependency=afterok:{pilots[ds]}']
            for key in ('QOS', 'ACCOUNT'):
                if env.get(key):
                    command += [f'--{key.lower()}={env[key]}']
            filename = 'run_analysis_step.sbatch' if entry['kind']=='analysis' else 'zheng_selection_bias.sbatch'
            command += [f'scripts/slurm/{filename}']
            entry['submission_started'] = True
            save_submission(directory, state, entry)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                    text=True, capture_output=True)
            if result.returncode:
                entry['submission_started'] = False
                save_submission(directory, state, entry)
                raise RuntimeError(result.stderr or result.stdout)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise ValueError('Unexpected sbatch response; inspect Slurm before retry')
            entry.update(job_id=job, submission_started=False, partition=partition)
            if stage == 'full':
                entry['depends_on_pilot'] = pilots[ds]
            save_submission(directory, state, entry)
        if stage == 'pilot':
            pilots[ds] = entry['job_id']
        print(f'{ds} {stage}: job {entry["job_id"]}' +
              (f' (after successful pilot {pilots[ds]})' if stage=='full' else ''))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default=MODEL)
    p.add_argument('--directory', type=Path)
    p.add_argument('--reuse-pilots', type=Path)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--partition', default='lrz-dgx-a100-80x8')
    args = p.parse_args()
    prefix = 'gemma' if args.model == MODEL else 'llama'
    args.directory = args.directory or PROJECT_ROOT/f'outputs/campaigns/{prefix}-all-datasets-20261013'
    args.reuse_pilots = args.reuse_pilots or PROJECT_ROOT/f'outputs/campaigns/{prefix}-pilot-20261013'
    if args.submit:
        check_access(args.model)
    state = prepare(args.directory, args.reuse_pilots, args.model)
    if args.submit:
        submit(args.directory, state, args.partition)
    else:
        print('Prepared five pilots and five full runs; no jobs submitted.')


if __name__ == '__main__':
    main()
