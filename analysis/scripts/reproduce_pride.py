"""Canonical 0-shot PriDe: original raw prompts/scoring, numerics, five draws.

Reuse compatible full reference observations; infer new scores only when needed.
Medical datasets extend the original protocol on the frozen retained population.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from _bootstrap import PROJECT_ROOT
from src.experimental.zheng_selection_bias import (
    load_original, option_diagnostic, RecordingModel, excluded, source_digest, probability_error,
)
from src.pride.zheng import evaluate, VERSION, DEBIAS_SHA256
from src.utils.config import named_config
from src.utils.io import write_json

COUNTS = dict(medqa=1273, medmcqa=3941, mmlu=13592, arc=1165, csqa=1216)
MODELS = ('llama-3.1-8b-instruct', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
          'mistral-7b-instruct-v0.3', 'qwen3-8b', 'glm-4-9b-chat-hf')
RAW_PROTOCOL = 'original-open-clm-raw-prompt-1536'


def compatible_sources(root, model, dataset):
    sources = []
    for path in (PROJECT_ROOT/'outputs/zheng_selection_bias').glob('*/manifest.json'):
        state = json.loads(path.read_text())
        c = state.get('config', {})
        current = named_config('models', model)
        if (state.get('status') == 'completed' and c.get('model') == model
                and c.get('dataset') == dataset and c.get('limit') is None
                and 0 in c.get('shots', []) and c.get('protocol') == RAW_PROTOCOL
                and c.get('scorer_backend') == 'real_transformers'
                and c.get('scoring_adapter_version') == 'last-token-fp32-v1'
                and c.get('model_config') == current
                and any(x in c.get('settings', []) for x in ('perm', 'cyclic'))):
            source_digest(dataset)
            if c.get('data_sha256') != source_digest(dataset):
                raise ValueError('Reference data fingerprint differs')
            sources.append(path)
    medical = root/'observations'/f'{model}_{dataset}'/'manifest.json'
    if dataset in ('medqa', 'medmcqa') and medical.exists():
        state = json.loads(medical.read_text())
        if state.get('status') == 'completed' and state['config']['implementation'] == VERSION:
            if state['config']['model_config'] != named_config('models', model):
                raise ValueError('Medical observation model config differs')
            from start_gemma_all_datasets import artifact
            current_population = artifact(dataset)
            if state['config']['preprocessing_manifest_sha256'] != hashlib.sha256(
                    (current_population/'manifest.json').read_bytes()).hexdigest():
                raise ValueError('Medical preprocessing fingerprint differs')
            sources.append(medical)
    return sorted(sources, key=lambda p: p.stat().st_mtime, reverse=True)


def load_observations(path, dataset):
    state = json.loads(path.read_text())
    setting = 'perm' if 'perm' in state['config']['settings'] else 'cyclic'
    original = load_original()
    samples = []
    with path.with_name('records.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['shots'] != 0 or record['setting'] != setting:
                continue
            data = record['data']
            if dataset in ('mmlu', 'arc', 'csqa') and excluded(data['options'], original):
                continue
            subject, index = record['subject'], int(data['idx'])
            samples.append(dict(id=f'{subject}:{index}', subject=subject, index=index,
                                probs=data['probs'], gold='ABCDE'.index(data['ideal'])))
    samples.sort(key=lambda s: (s['subject'], s['index']))
    if len(samples) != COUNTS[dataset]:
        raise ValueError(f'{dataset}: expected {COUNTS[dataset]} retained questions, got {len(samples)}')
    return samples, setting


def medical_inputs(samples, original):
    system = ('The following are multiple choice questions. '
              'You should directly answer the question by choosing the correct option.')
    output = []
    for i, sample in enumerate(samples):
        prompts = []
        for options in original.utils.cycle_options(list(sample.options)):
            prompt = f'Question: {sample.question.strip()}\nOptions:\n' + '\n'.join(
                f'{label}. {answer}'.strip() for label, answer in zip('ABCD', options)) + '\nAnswer:'
            prompts.append([system, prompt])
        output.append((i, (prompts, list(sample.options), sample.gold_label)))
    return output


def infer_medical(root, model_alias, dataset):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('New inference requires a Slurm GPU allocation')
    from start_gemma_all_datasets import artifact
    from src.data.preprocessing import load_preprocessed_artifact
    original = load_original()
    population = artifact(dataset)
    samples, manifest = load_preprocessed_artifact(population)
    if len(samples) != COUNTS[dataset]:
        raise ValueError('Medical population differs')
    config = dict(implementation=VERSION, model=model_alias, dataset=dataset, shots=[0],
                  settings=['cyclic'], limit=None, model_config=named_config('models', model_alias),
                  protocol=RAW_PROTOCOL, scorer_backend='real_transformers',
                  scoring_adapter_version='last-token-fp32-v1',
                  preprocessing_manifest_sha256=hashlib.sha256((population/'manifest.json').read_bytes()).hexdigest())
    directory = root/'observations'/f'{model_alias}_{dataset}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'manifest.json'
    state = json.loads(path.read_text()) if path.exists() else dict(config=config)
    if state['config'] != config:
        raise ValueError('Observation resume configuration differs')
    write_json(path, state, overwrite=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    try:
        c = config['model_config']
        tokenizer = AutoTokenizer.from_pretrained(c['tokenizer_repo'], revision=c['tokenizer_revision'],
            use_fast=False, add_bos_token=False, add_eos_token=False, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(c['model_repo'], revision=c['model_revision'],
            device_map='auto', use_safetensors=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model.eval()
        runtime = dict(model_revision=getattr(model.config, '_commit_hash', None),
            tokenizer_revision=tokenizer.init_kwargs.get('_commit_hash'), dtype=str(model.dtype),
            model_class=type(model).__name__, tokenizer_class=type(tokenizer).__name__,
            versions={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','pandas')})
        if state.get('runtime') and state['runtime'] != runtime:
            raise ValueError('Runtime changed; start a new observation version')
        diagnostic = option_diagnostic(tokenizer, 'ABCD')
        recorder = RecordingModel(model, diagnostic['slot_ids'])
        evaluator = original.evaluation.prepare_eval_fn_perm(recorder, tokenizer, [], 0, list('ABCD'))
        state.update(status='running', runtime=runtime, diagnostic=diagnostic,
                     job_id=os.environ['SLURM_JOB_ID'])
        write_json(path, state, overwrite=True)
        raw = directory/'records.jsonl'
        done = set()
        if raw.exists():
            for line in raw.read_text().splitlines():
                rec = json.loads(line)
                if rec['data']['idx'] in done:
                    raise ValueError('Duplicate partial observation')
                done.add(rec['data']['idx'])
        with raw.open('a') as stream:
            for item in medical_inputs(samples, original):
                if item[0] in done:
                    continue
                recorder.trace = []
                record = evaluator(item, random.Random(f'{item[0]}:20230101'.encode()))
                issue = probability_error(record['data']['probs'])
                if issue:
                    raise ValueError(issue)
                record.update(shots=0, setting='cyclic', subject=dataset)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                done.add(item[0])
                if len(done)%100 == 0:
                    print(f'{model_alias}/{dataset}: {len(done)}/{len(samples)}', flush=True)
        if done != set(range(len(samples))):
            raise ValueError('Incomplete medical observations')
        state.update(status='completed', selected_questions=len(samples))
        write_json(path, state, overwrite=True)
        return path
    except BaseException as error:
        state.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        write_json(path, state, overwrite=True)
        raise


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_tables(root):
    result_files = list((root/'results').glob('*/*/result.json'))
    results = {}
    for path in result_files:
        data = json.loads(path.read_text())
        if data.get('implementation') == VERSION:
            results[(data['model'],data['dataset'])] = data
    names = dict(zip(MODELS,['Llama-3.1-8B-Instruct','Qwen2.5-7B-Instruct','Gemma-2-9B-it',
                            'Mistral-7B-Instruct-v0.3','Qwen3-8B','GLM-4-9B-Chat-HF']))
    labels = dict(medqa='MedQA',medmcqa='MedMCQA',mmlu='MMLU',arc='ARC-Challenge',csqa='CommonsenseQA')
    lines = ['% Canonical 0-shot, original raw scoring and original PriDe numerics.',
             '% PriDe: mean over five path-seeded calibration draws; old chat results excluded.']
    for dataset in COUNTS:
        lines += [r'\begin{table}[!htbp]', r'\centering\small',
                  r'\caption{'+labels[dataset]+f' ($N={COUNTS[dataset]}$)'+r': 0-shot reference-protocol reproduction. '
                  r'Acc is in percent; RStd is in pp. PriDe uses 5\% calibration and averages five draws. '
                  r'Dashes denote missing results; $\Delta$ Average uses the completed models.}', r'\begin{tabular}{l|rr|rr|rr}',r'\hline',
                  r'Model & \multicolumn{2}{c|}{Default} & \multicolumn{2}{c|}{Cyclic Perm} & \multicolumn{2}{c}{PriDe (5\%)}\\',
                  r' & RStd & Acc & RStd & Acc & RStd & Acc\\',r'\hline']
        for model in MODELS:
            data = results.get((model,dataset))
            values = ([f'{data[method][key]:.1f}' for method in ('default','cyclic_perm','pride')
                       for key in ('rstd','accuracy')] if data else ['--']*6)
            lines.append(names[model]+' & '+' & '.join(values)+r'\\')
        available = [results[(m,dataset)] for m in MODELS if (m,dataset) in results]
        if available:
            delta = [sum(d[method][key]-d['default'][key] for d in available)/len(available)
                     for method in ('cyclic_perm','pride') for key in ('rstd','accuracy')]
            lines += [r'\hline',r'$\Delta$ Average & -- & -- & '+' & '.join(f'{v:+.1f}' for v in delta)+r'\\']
        lines += [r'\hline',r'\end{tabular}',r'\end{table}','']
    (root/'results_0shot.tex').write_text('\n'.join(lines))
    write_json(root/'inventory.json', dict(implementation=VERSION, complete_pairs=len(results),
               planned_pairs=30, missing=[dict(model=m,dataset=d) for m in MODELS for d in COUNTS
                                          if (m,d) not in results]), overwrite=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--datasets', nargs='+', choices=list(COUNTS), default=list(COUNTS))
    p.add_argument('--output-root', type=Path, default=PROJECT_ROOT/'outputs/reproductions'/VERSION)
    p.add_argument('--postprocess-only', action='store_true')
    args = p.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        destination = root/'results'/args.model/dataset/'result.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            if (existing.get('implementation') != VERSION
                    or existing.get('debias_source_sha256') != DEBIAS_SHA256
                    or existing.get('model') != args.model or existing.get('dataset') != dataset):
                raise ValueError('Conflicting prior implementation in destination')
            old_source = Path(existing['observation_manifest'])
            if (not old_source.exists() or digest(old_source) != existing['observation_manifest_sha256']
                    or digest(old_source.with_name('records.jsonl')) != existing['observations_sha256']):
                raise ValueError('Completed result observations changed; refusing stale results')
            print(f'Already completed: {args.model}/{dataset}', flush=True)
            continue
        sources = compatible_sources(root, args.model, dataset)
        if not sources:
            if args.postprocess_only:
                print(f'Missing compatible observations: {args.model}/{dataset}', flush=True)
                continue
            if dataset in ('medqa','medmcqa'):
                path = infer_medical(root,args.model,dataset)
            else:
                command = [sys.executable,str(PROJECT_ROOT/'scripts/run_zheng_selection_bias.py'),
                           '--model',args.model,'--dataset',dataset,'--shots','0','--settings','cyclic']
                subprocess.run(command,cwd=PROJECT_ROOT,check=True)
                sources = compatible_sources(root,args.model,dataset)
                if not sources:
                    raise ValueError('Reference inference produced no compatible full observations')
                path = sources[0]
        else:
            path = sources[0]
        samples,setting = load_observations(path,dataset)
        repo_name = named_config('models',args.model)['model_repo'].split('/')[-1]
        source_path = f'results_{dataset}/0s_{repo_name}/{dataset}_{setting}'
        result = evaluate(samples,source_path,alpha=0.05,repeats=5)
        result.update(model=args.model,dataset=dataset,scoring_protocol=RAW_PROTOCOL,
            scoring_adapter='last-token-fp32-v1', debias_source_sha256=DEBIAS_SHA256,
            observation_manifest=str(path),observation_manifest_sha256=digest(path),
            observations_sha256=digest(path.with_name('records.jsonl')),
            method_runtime_note='All cyclic observations were collected for CPA; reported cost factor is theoretical, not measured wall time.')
        write_json(destination,result)
        print(f'{args.model}/{dataset}: Default {result["default"]["accuracy"]:.3f}, '
              f'CPA {result["cyclic_perm"]["accuracy"]:.3f}, PriDe {result["pride"]["accuracy"]:.3f}',flush=True)
        write_tables(root)
    write_tables(root)


if __name__ == '__main__':
    main()
