"""Prepare explicit MedMCQA/MMLU plans for all configured models on this host.

Run on LRZ before submission so all saved paths point to the GPU host.
Default: 20-question pilots. --full prepares full populations after pilot review.
"""
import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from run_campaign import plan
from src.data.preprocessing import load_preprocessed_artifact
from src.utils.config import named_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    models = args.models or sorted(p.stem for p in (PROJECT_ROOT/'configs/models').glob('*.yaml'))
    for model in models:
        named_config('models', model)
    version = named_config('experiments', 'thesis')['prompt']['version']
    artifacts = {}
    for dataset in ('medmcqa', 'mmlu'):
        candidates = []
        for file in (PROJECT_ROOT/'outputs/preprocessed'/dataset).glob('*/manifest.json'):
            manifest = json.loads(file.read_text())
            if (manifest.get('prompt_template_version') == version
                    and manifest.get('selection_limit') is None
                    and manifest.get('selection_start_index') == 0):
                candidates.append(file.parent)
        if len(candidates) != 1:
            raise ValueError(f'Expected one complete {version} artifact for {dataset}, found {len(candidates)}')
        samples, _ = load_preprocessed_artifact(candidates[0], verify_hashes=True)
        artifacts[dataset] = candidates[0]
        print(f'{dataset}: {len(samples)} validated questions')
    directory = args.directory.resolve()
    if directory.exists():
        raise FileExistsError(f'Choose a new campaign directory: {directory}')
    for dataset, artifact in artifacts.items():
        state = plan(directory/dataset, models, [dataset],
                     limit=None if args.full else 20, batch_size=1, preprocessed=artifact)
        print(f'{dataset}: {len(state["entries"])} real-model jobs planned; none submitted')


if __name__ == '__main__':
    main()
