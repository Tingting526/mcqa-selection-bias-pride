"""CPU-only prior-selection sensitivity using saved cyclic probabilities."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _bootstrap import PROJECT_ROOT
from src.evaluation.saved_baseline import records_digest
from src.evaluation.pride_eval import estimate_prior_from_permutation_records, apply_prior_to_records, split_estimation_and_remaining
from src.evaluation.permutation_eval import cyclic_debiased_records
from src.pride.prior_estimation import select_estimation_ids, estimate_global_prior
from src.metrics.selection_bias import selection_bias_metrics
from src.metrics.paired_analysis import pair_predictions

SEEDS = tuple(range(42, 52))


def evaluate_seeds(baseline, permutations, reference, seeds=SEEDS):
    if len(set(seeds)) != len(seeds) or reference['seed'] not in seeds:
        raise ValueError('Unique seeds including the reference seed are required')
    ids = baseline.sample_id.astype(str).tolist()
    if len(set(ids)) != len(ids) or set(permutations.sample_id.astype(str)) != set(ids):
        raise ValueError('Unique baseline IDs and matching permutation population required')
    if reference['sampling_order'] != 'sorted_sample_id' or reference['estimation_sample_policy'] != 'cyclic':
        raise ValueError('Unsupported reference sampling protocol')
    rows = permutations.to_dict('records')
    sample_priors, _ = estimate_prior_from_permutation_records(rows, epsilon=reference['epsilon'])
    priors_by_id = {str(r['sample_id']): [r['prior_'+l] for l in 'ABCD'] for r in sample_priors}
    cyclic = cyclic_debiased_records(rows)
    selections = {seed: select_estimation_ids(ids, reference['alpha'], seed,
                  rounding=reference['alpha_rounding']) for seed in seeds}
    if selections[reference['seed']] != sorted(reference['estimation_sample_ids']):
        raise ValueError('Reference estimation IDs cannot be reproduced')
    reference_prior = estimate_global_prior([priors_by_id[i] for i in selections[reference['seed']]])
    # Separate forward passes can differ slightly. Abort on a material mismatch.
    if not np.allclose(reference_prior, reference['global_prior'], atol=1e-6, rtol=1e-5):
        raise ValueError('Cached permutations do not reproduce the original prior within tolerance')
    common = set(ids) - set().union(*(set(v) for v in selections.values()))
    if not common:
        raise ValueError('No common questions outside all estimation sets')
    results = []
    baseline_rows = baseline.to_dict('records')
    for seed, selected in selections.items():
        prior = estimate_global_prior([priors_by_id[i] for i in selected])
        corrected = apply_prior_to_records(baseline_rows, prior, epsilon=reference['epsilon'])
        corrected = split_estimation_and_remaining(corrected,
                    [r for r in cyclic if str(r['sample_id']) in set(selected)], selected)
        paired = pair_predictions(baseline, pd.DataFrame(corrected))
        for group in ('D_r', 'common_D_r', 'all'):
            subset = paired if group == 'all' else paired[
                paired.sample_id.isin(common) if group == 'common_D_r' else paired.sample_group == 'D_r']
            allowed = set(subset.sample_id)
            before = selection_bias_metrics([r for r in baseline_rows if str(r['sample_id']) in allowed])
            after = selection_bias_metrics([r for r in corrected if str(r['sample_id']) in allowed])
            results.append(dict(seed=seed, group=group, n=len(subset), estimation_n=len(selected),
                baseline_accuracy=before['accuracy'], pride_accuracy=after['accuracy'],
                delta_accuracy=after['accuracy']-before['accuracy'],
                baseline_rstd=before['rstd'], pride_rstd=after['rstd'],
                wrong_to_right=int((subset.transition == 'wrong_to_right').sum()),
                right_to_wrong=int((subset.transition == 'right_to_wrong').sum()),
                **{f'prior_{l}':float(prior[i]) for i,l in enumerate('ABCD')}))
    return pd.DataFrame(results), selections, sorted(common)


def read_parts(folder):
    files = sorted((folder/'parts').glob('part-*.parquet'))
    if not files:
        raise FileNotFoundError(f'No saved probabilities in {folder}; run scripts/fetch_medqa_results.sh on Mac')
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def analyze(campaign, output, project_root=PROJECT_ROOT):
    campaign, output, project_root = Path(campaign), Path(output), Path(project_root)
    entries = json.loads((campaign/'campaign.json').read_text())['entries']
    if len(entries) != 3 or {e['config']['model'] for e in entries} != {
            'qwen3-8b','mistral-7b-instruct-v0.3','glm-4-9b-chat-hf'}:
        raise ValueError('Expected the three-model MedQA campaign')
    results, audit, common_selections = [], [], None
    for entry in entries:
        if entry['config']['dataset'] != 'medqa' or entry['config']['scorer'] != 'real':
            raise ValueError('Expected real MedQA campaign')
        name = Path(entry['pipeline']).name
        pipeline = campaign/name
        state = json.loads((pipeline/'pipeline.json').read_text())
        if state.get('status') != 'completed':
            raise ValueError('Pipeline incomplete')
        prior = json.loads((pipeline/Path(state['prior']).name).read_text())
        root = project_root/'outputs/raw'
        baseline = read_parts(root/'baseline'/name)
        permutations = read_parts(root/'permutations'/name)
        if len(baseline) != 1273 or len(permutations) != 5092:
            raise ValueError('Expected 1273 questions and all four permutations')
        metadata = [json.loads((root/kind/name/'metadata.json').read_text())
                    for kind in ('baseline','permutations')]
        for key in ('scorer_backend','model_repo','model_revision_resolved','tokenizer_revision_resolved',
                    'prompt_template_version','chat_template_kwargs','option_token_ids','preprocessing'):
            if metadata[0].get(key) != metadata[1].get(key):
                raise ValueError(f'Incompatible saved runs: {key}')
        if metadata[0]['scorer_backend'] != 'real_transformers':
            raise ValueError('Not real model output')
        default = permutations[permutations.permutation_id == 0].set_index('sample_id').sort_index()
        base = baseline.set_index('sample_id').sort_index()
        if not default.index.equals(base.index) or not default.gold_label.equals(base.gold_label):
            raise ValueError('Permutation zero does not match baseline questions')
        cols = ['prob_'+l for l in 'ABCD']
        if not np.allclose(default[cols],base[cols],atol=1e-6,rtol=1e-5):
            raise ValueError('Permutation-zero probabilities differ from baseline')
        result, selections, common = evaluate_seeds(baseline, permutations, prior)
        if common_selections is not None and selections != common_selections:
            raise ValueError('Models do not use the same estimation samples')
        common_selections = selections
        result.insert(0,'model',entry['config']['model'])
        results.append(result)
        audit.append(dict(model=entry['config']['model'], pipeline=name,
            baseline_digest=records_digest(baseline.to_dict('records')),
            permutation_digest=records_digest(permutations.to_dict('records')),
            estimation_ids=selections, common_remaining_ids=common))
    table = pd.concat(results,ignore_index=True)
    overview = table.groupby(['model','group']).agg(seeds=('seed','count'),n=('n','first'),
        mean_delta=('delta_accuracy','mean'),sd_delta=('delta_accuracy','std'),
        min_delta=('delta_accuracy','min'),max_delta=('delta_accuracy','max'),
        mean_pride_accuracy=('pride_accuracy','mean'),mean_pride_rstd=('pride_rstd','mean')).reset_index()
    output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'seed_results.csv',index=False)
    overview.to_csv(output/'seed_summary.csv',index=False)
    (output/'provenance.json').write_text(json.dumps(dict(runs=audit,seeds=SEEDS,
        scope='Prior-selection sensitivity conditional on cached model scores. Seed SD is not a standard error or confidence interval; overlapping samples are not independent replications.'),indent=2))
    return table, overview


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=PROJECT_ROOT/'outputs/campaigns/medqa-three-models-20260930')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/analysis/medqa-seeds-20261006')
    args=parser.parse_args()
    _,overview=analyze(args.campaign,args.output)
    print(overview.to_string(index=False))
