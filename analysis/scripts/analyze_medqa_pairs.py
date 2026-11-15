"""Analyze the completed three-model MedQA campaign without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from _bootstrap import PROJECT_ROOT
from src.evaluation.saved_baseline import records_digest
from src.metrics.paired_analysis import pair_predictions, summarize_pairs, holm_adjust


def analyze(campaign, output, *, project_root=PROJECT_ROOT):
    campaign, output, project_root = Path(campaign), Path(output), Path(project_root)
    entries = json.loads((campaign / 'campaign.json').read_text())['entries']
    expected_models = {'qwen3-8b', 'mistral-7b-instruct-v0.3', 'glm-4-9b-chat-hf'}
    if len(entries) != 3 or {e['config']['model'] for e in entries} != expected_models:
        raise ValueError('Expected the three-model MedQA campaign')
    results, details, provenance = [], [], []
    common_ids = None
    common_estimation = None
    for entry in entries:
        model = entry['config']['model']
        if entry['config']['dataset'] != 'medqa' or entry['config']['scorer'] != 'real':
            raise ValueError('Only real MedQA predictions are accepted')
        pipeline_id = Path(entry['pipeline']).name
        pipeline = campaign / pipeline_id
        state = json.loads((pipeline / 'pipeline.json').read_text())
        if state.get('status') != 'completed':
            raise ValueError(f'Incomplete pipeline: {pipeline_id}')
        baseline_dir = project_root / 'outputs/raw/baseline' / pipeline_id
        pride_dir = project_root / 'outputs/raw/pride' / pipeline_id
        files = sorted((baseline_dir / 'parts').glob('part-*.parquet'))
        candidates = list(pride_dir.glob('pride_results*.parquet'))
        if not files or len(candidates) != 1:
            raise ValueError(f'Missing baseline or ambiguous PriDe result files: {pipeline_id}')
        baseline = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        pride = pd.read_parquet(candidates[0])
        meta = json.loads((pride_dir / 'metadata.json').read_text())
        if meta.get('scorer_backend') != 'real_transformers':
            raise ValueError('Mock or unknown backend in result metadata')
        if meta.get('baseline_source_sha256') != records_digest(baseline.to_dict('records')):
            raise ValueError('Saved baseline digest does not match PriDe provenance')
        paired = pair_predictions(baseline, pride)
        estimation = set(paired.loc[paired.sample_group == 'D_e', 'sample_id'])
        if estimation != set(map(str, meta['prior_estimation_sample_ids'])):
            raise ValueError('Estimation membership differs from metadata')
        if len(paired) != 1273 or len(estimation) != 63:
            raise ValueError('Expected 1273 total and 63 estimation questions')
        ids = set(paired.sample_id)
        if common_ids is not None and (ids != common_ids or estimation != common_estimation):
            raise ValueError('Models used different question populations or estimation sets')
        common_ids, common_estimation = ids, estimation
        comparison = pd.read_csv(pipeline / 'comparison.csv').iloc[0]
        for group in ('D_r', 'all', 'D_e'):
            subset = paired if group == 'all' else paired[paired.sample_group == group]
            result = summarize_pairs(subset)
            if group in ('D_r', 'all'):
                prefix = 'pride_' if group == 'all' else 'pride_D_r_'
                for metric, key in [('baseline_accuracy', prefix+'baseline_accuracy'),
                                    ('pride_accuracy', prefix+'pride_accuracy')]:
                    if abs(result[metric] - float(comparison[key])) > 1e-10:
                        raise ValueError(f'Individual answers disagree with summary: {model} {key}')
            results.append(dict(model=model, group=group, **result))
        paired.insert(0, 'model', model)
        details.append(paired)
        provenance.append(dict(model=model, pipeline_id=pipeline_id, baseline_digest=meta['baseline_source_sha256'],
                               pride_file=candidates[0].name,
                               pride_sha256=hashlib.sha256(candidates[0].read_bytes()).hexdigest()))
    summary = pd.DataFrame(results)
    summary['mcnemar_holm_p'] = float('nan')
    primary = summary.group == 'D_r'
    summary.loc[primary, 'mcnemar_holm_p'] = holm_adjust(summary.loc[primary, 'mcnemar_exact_p'])
    # No outputs are written until all models and their provenance have passed validation.
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / 'paired_summary.csv', index=False)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(output / 'answer_changes.csv', index=False)
    labels = []
    for (model, group), frame in detail.groupby(['model', 'sample_group']):
        for label in 'ABCD':
            gold = frame.gold_label_baseline == label
            labels.append(dict(model=model, group=group, label=label, gold_count=int(gold.sum()),
                               baseline_count=int((frame.predicted_label_baseline == label).sum()),
                               pride_count=int((frame.predicted_label_pride == label).sum()),
                               baseline_recall=float(frame.loc[gold, 'baseline_correct'].mean()),
                               pride_recall=float(frame.loc[gold, 'pride_correct'].mean())))
    pd.DataFrame(labels).to_csv(output / 'label_analysis.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(sources=provenance,
        method='Paired percentile bootstrap, 20000 draws, seed 42. Exact two-sided McNemar. Holm across three D_r tests.',
        scope='Conditional on one fixed estimated prior; excludes uncertainty from prior selection. All/D_e are descriptive.'), indent=2))
    return summary, detail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=PROJECT_ROOT/'outputs/campaigns/medqa-three-models-20260930')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'outputs/analysis/medqa-paired-20261006')
    args = parser.parse_args()
    summary, _ = analyze(args.campaign, args.output)
    print(summary[summary.group == 'D_r'].to_string(index=False))
