"""Check access and submit selected models for full MedQA or MedMCQA on LRZ."""
import argparse
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
from _bootstrap import PROJECT_ROOT
from run_campaign import plan, submit
from src.utils.config import named_config

MODELS = ['llama-3.1-8b-instruct', 'mistral-7b-instruct-v0.3', 'qwen3-8b',
          'gemma-2-9b-it', 'glm-4-9b-chat-hf']


DATASETS = {'medqa': ('test', 1273), 'medmcqa': ('validation', 3941)}


def select_artifact(dataset='medqa'):
    split, count = DATASETS[dataset]
    version = named_config('experiments', 'thesis')['prompt']['version']
    candidates = []
    for path in (PROJECT_ROOT / 'outputs/preprocessed' / dataset).glob('*/manifest.json'):
        manifest = json.loads(path.read_text())
        if (manifest['dataset_alias'] == dataset and manifest['dataset_split'] == split
                and manifest['selected_samples'] == count
                and manifest['selection_start_index'] == 0
                and manifest['selection_limit'] is None
                and manifest['prompt_template_version'] == version):
            candidates.append((manifest['created_at_utc'], path.parent))
    if not candidates:
        raise ValueError(f'No full {dataset} artifact matching prompt {version}; prepare preprocessing first')
    return max(candidates)[1]


def check_access(model):
    config = named_config('models', model)
    revision = config['model_revision']
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Model revision must be a fixed commit')
    repo = config['model_repo']
    info = HfApi().model_info(repo, revision=revision)
    # Small files only; verifies access to gated repositories before queuing GPUs.
    hf_hub_download(repo, 'config.json', revision=revision)
    hf_hub_download(config['tokenizer_repo'], 'tokenizer_config.json',
                    revision=config['tokenizer_revision'])
    weights = sorted(f.rfilename for f in info.siblings if f.rfilename.endswith('.safetensors'))
    if not weights:
        raise ValueError('No safetensors model weights found')
    get_hf_file_metadata(hf_hub_url(repo, weights[0], revision=revision))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--dataset', choices=DATASETS, default='medqa')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--partition', default='lrz-dgx-a100-80x8')
    args = parser.parse_args()
    models = list(dict.fromkeys(args.models))
    artifact = select_artifact(args.dataset)
    print(f'{args.dataset} artifact: {artifact}', flush=True)
    failed = []
    for model in models:
        try:
            check_access(model)
            print(f'Access OK: {model}', flush=True)
        except Exception as exc:
            failed.append(model)
            # Do not print exception URLs or credential details.
            print(f'Access check failed: {model} ({type(exc).__name__})', flush=True)
    if failed:
        raise SystemExit('Nothing submitted. Resolve access/network for: ' + ', '.join(failed)
                         + '. For gated models, accept access on Hugging Face and run hf auth login on LRZ.')
    count = DATASETS[args.dataset][1]
    estimation = int(count * 0.05)
    print(f'Checks passed. Each model: {count} questions, {estimation} estimation / {count-estimation} remaining.', flush=True)
    if not args.submit:
        print('No jobs submitted; rerun with --submit to start.')
        return
    directory = (args.directory or PROJECT_ROOT / 'outputs/campaigns' /
                 f'{args.dataset}-five-models-20260930').resolve()
    path = directory / 'campaign.json'
    if path.exists():
        state = json.loads(path.read_text())
        expected = [dict(model=model, dataset=args.dataset, scorer='real', limit=None,
                         batch_size=1, preprocessed=str(artifact.resolve())) for model in models]
        if [entry['config'] for entry in state['entries']] != expected:
            raise ValueError('Existing campaign differs; use another directory')
    else:
        state = plan(directory, models, [args.dataset], batch_size=1, preprocessed=artifact)
    submit(directory, state, args.partition)


if __name__ == '__main__':
    main()
