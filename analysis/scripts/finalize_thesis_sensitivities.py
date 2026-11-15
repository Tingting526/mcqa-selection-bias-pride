#!/usr/bin/env python3
"""Validate saved GPU sensitivities through a second, vectorized calculation.

Reads only completed observations. No model or production analysis helpers are
imported. Original calibration ID lists, rather than regenerated seeds, define
the five fits. The resulting CSVs include both absolute and paired differences.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis/outputs/thesis_strengthening_20261021'
P3 = ROOT / 'analysis/outputs/phase3b_20261018/verified'
P6 = ROOT / 'analysis/outputs/phase6_robustness_20261018'
MODELS = ('llama-3.1-8b-instruct', 'qwen2.5-7b-instruct', 'gemma-2-9b-it',
          'mistral-7b-instruct-v0.3', 'qwen3-8b', 'glm-4-9b-chat-hf')
EPS = 1e-10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(gold, prediction):
    recalls = [100*np.mean(prediction[gold == j] == j) for j in range(4)]
    return dict(accuracy=float(100*np.mean(gold == prediction)),
                rstd=float(np.std(recalls, ddof=0)))


def calculate(p, gold, legacy, trials):
    """Independent vectorized estimator; use saved, ordered calibration IDs."""
    n = len(gold)
    q = p/(p.sum(axis=2, keepdims=True)+EPS)
    aligned = np.stack([np.roll(q[:, s], s, axis=1) for s in range(4)], axis=1)
    cpa = aligned.mean(axis=1).argmax(axis=1)
    loggeom = np.log(q+EPS).mean(axis=1)
    item_priors = np.exp(loggeom-loggeom.max(axis=1, keepdims=True))
    item_priors /= item_priors.sum(axis=1, keepdims=True)+EPS
    index = {str(v): j for j, v in enumerate(legacy)}
    fits, predictions = [], []
    for trial in trials:
        estimation = [index[v] for v in trial['estimation_ids']]
        assert len(set(estimation)) == len(estimation) == int(.05*n)
        prior = item_priors[estimation].mean(axis=0)
        prediction = (np.log(p[:, 0]+EPS)-np.log(prior+EPS)).argmax(axis=1)
        prediction[estimation] = cpa[estimation]
        predictions.append(prediction)
        fits.append(dict(global_prior=prior.tolist(), **metrics(gold, prediction)))
    displayed = p.argmax(axis=2)
    contents = (displayed+np.arange(4)) % 4
    cyclic_gold = (gold[:, None]-np.arange(4)) % 4
    result = dict(default=metrics(gold, displayed[:, 0]),
                  cyclic_perm=metrics(gold, cpa),
                  pride={key:float(np.mean([row[key] for row in fits])) for key in ('accuracy', 'rstd')},
                  consistency_percent=float(100*np.all(contents == contents[:, :1], axis=1).mean()),
                  cyclic_rstd_pp=metrics(cyclic_gold.ravel(), displayed.ravel())['rstd'],
                  fits=fits)
    return result, dict(default=displayed[:, 0], cpa=cpa, hybrid=np.array(predictions))


def compare_saved(calculated, saved):
    for method in ('default', 'cyclic_perm', 'pride'):
        for key in ('accuracy', 'rstd'):
            np.testing.assert_allclose(calculated[method][key], saved[method][key], atol=1e-9, rtol=0)
    for fit, trial in zip(calculated['fits'], saved['trials']):
        np.testing.assert_allclose(fit['global_prior'], trial['global_prior'], atol=1e-12, rtol=0)
        for key in ('accuracy', 'rstd'):
            np.testing.assert_allclose(fit[key], trial['full'][key], atol=1e-9, rtol=0)


def read_run(model, phase, plan, run_name=None):
    directory = OUT/'gpu_results'/model/(run_name or phase)
    manifest = json.loads((directory/'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['inference_executed']
    assert manifest['model'] == model
    assert manifest['plan_sha256'] == digest(OUT/'inference_plan.json')
    assert manifest['records_sha256'] == digest(directory/'records.jsonl')
    assert manifest['config'] == plan['models'][model]
    assert manifest['model_revision'] == plan['models'][model]['model_revision']
    for name, value in manifest['tokenizer_hashes'].items():
        assert digest(directory/'tokenizer'/name) == value
    records = [json.loads(line) for line in (directory/'records.jsonl').open()]
    wanted = [row for row in plan['requests'] if (row['condition'] == 'medical_native_chat') == (phase == 'chat')]
    assert {(row['condition'], row['sample_id']) for row in records} == {(row['condition'], row['id']) for row in wanted}
    assert len(records) == len(wanted)
    lookup = {(row['condition'], row['id']):row for row in wanted}
    for row in records:
        expected = lookup[row['condition'], row['sample_id']]
        assert row['legacy_id'] == expected['legacy'] and row['gold_index'] == expected['gold']
        assert row['dataset'] == expected['dataset']
        values = np.array(row['probs'])
        assert values.shape == (4, 4) and np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-3, rtol=0)
        assert len(row['input_tokens_before_crop']) == len(row['prompt_sha256']) == 4
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available', action='store_true', help='Inspect complete models without writing final artifacts')
    args = parser.parse_args()
    plan = json.loads((OUT/'inference_plan.json').read_text())
    rows, checks, sources = [], [], []
    baseline = json.loads((OUT/'preservation_baseline.json').read_text())
    primary_files = [key for key in baseline if key.startswith('analysis/') or key == 'writing_style_guide.md']
    for name in primary_files:
        assert digest(ROOT/name) == baseline[name], name
    for model in MODELS:
        for phase in ('mmlu', 'chat'):
            run_name = 'chat_a100' if model == 'mistral-7b-instruct-v0.3' and phase == 'chat' else phase
            summary_path = OUT/'gpu_results'/model/(run_name+'_summary.json')
            if args.available and not summary_path.exists():
                continue
            records = read_run(model, phase, plan, run_name)
            summary = json.loads(summary_path.read_text())
            for dataset, record in summary['datasets'].items():
                cache_path = P6/f'scores/{model}__{dataset}.npz'
                cache = np.load(cache_path, allow_pickle=False)
                raw_sources = json.loads((P6/'raw_sources.json').read_text())['files']
                frozen = next(x for x in raw_sources if x['model'] == model and x['benchmark'] == dataset)
                assert digest(cache_path) == frozen['score_snapshot_sha256']
                original = json.loads((P3/f'recomputed/{model}/{dataset}/result.json').read_text())['result']
                assert record['original_primary'] == original
                assert record['prior_refitted_on_original_memberships']
                assert record['full_population_n'] == len(cache['gold'])
                assert [x['estimation_ids'] for x in original['trials']] == [x['estimation_ids'] for x in record['result']['trials']]
                index = {str(v):j for j, v in enumerate(cache['legacy_ids'])}
                p = cache['probabilities'].copy()
                condition = 'mmlu_literal' if phase == 'mmlu' else 'medical_native_chat'
                selected = [x for x in records if x['condition'] == condition and x['dataset'] == dataset]
                if phase == 'mmlu':
                    controls = [x for x in records if x['condition'] != 'mmlu_literal']
                    assert len(controls) == 12
                    max_error = max(float(np.max(np.abs(np.array(x['probs'])-p[index[x['legacy_id']]]))) for x in controls)
                    assert max_error <= 2e-5
                    np.testing.assert_equal(max_error, summary['mmlu_control_probability_max_abs_error'])
                    assert len(selected) == 10
                else:
                    max_error = None
                    assert len(selected) == len(p)
                for x in selected:
                    p[index[x['legacy_id']]] = x['probs']
                new, new_preds = calculate(p, cache['gold'], cache['legacy_ids'], original['trials'])
                old, old_preds = calculate(cache['probabilities'], cache['gold'], cache['legacy_ids'], original['trials'])
                compare_saved(new, record['result']); compare_saved(old, original)
                result = dict(model=model, dataset=dataset, n=len(p), condition=condition,
                              control_max_abs_error=max_error,
                              max_tokens=max(max(x['input_tokens_before_crop']) for x in selected),
                              truncated_questions=sum(max(x['input_tokens_before_crop']) > plan['context_cap'] for x in selected))
                for method in ('default', 'cyclic_perm', 'pride'):
                    for key in ('accuracy', 'rstd'):
                        result[method+'_'+key+'_raw'] = old[method][key]
                        result[method+'_'+key+'_new'] = new[method][key]
                        result[method+'_'+key+'_change_pp'] = new[method][key]-old[method][key]
                for key in ('consistency_percent', 'cyclic_rstd_pp'):
                    result[key+'_raw'], result[key+'_new'] = old[key], new[key]
                    result[key+'_change_pp'] = new[key]-old[key]
                for protocol in ('raw', 'new'):
                    result['hybrid_minus_default_'+protocol+'_pp'] = result['pride_accuracy_'+protocol]-result['default_accuracy_'+protocol]
                before, after, gold = old_preds['default'], new_preds['default'], cache['gold']
                result.update(default_predictions_changed_n=int((before != after).sum()),
                              default_correct_gained_n=int(((before != gold)&(after == gold)).sum()),
                              default_correct_lost_n=int(((before == gold)&(after != gold)).sum()),
                              default_correct_new_n=int((after == gold).sum()),
                              hybrid_correct_new_total_n=int((new_preds['hybrid'] == gold).sum()),
                              hybrid_accuracy_direction_new=int(np.sign((new_preds['hybrid'] == gold).sum()-5*(after == gold).sum())),
                              prior_max_abs_change=max(float(np.max(np.abs(np.array(a['global_prior'])-b['global_prior']))) for a,b in zip(new['fits'],old['fits'])))
                rows.append(result)
                checks.append(dict(model=model, dataset=dataset, observations=len(selected), validated=True))
            for file in (summary_path, summary_path.parent/run_name/'manifest.json', summary_path.parent/run_name/'records.jsonl'):
                sources.append(dict(path=str(file.relative_to(ROOT)), sha256=digest(file)))
    if args.available:
        print(json.dumps(rows, indent=2)); return
    assert len(rows) == 18
    # The H100 Mistral attempt is diagnostic evidence, excluded from the matched
    # template comparison because its raw controls differ from the frozen run.
    model = 'mistral-7b-instruct-v0.3'
    directory = OUT/'gpu_results'/model
    control_runs, control_manifests = {}, {}
    cache = np.load(P6/f'scores/{model}__mmlu.npz', allow_pickle=False)
    lookup = {str(v):j for j,v in enumerate(cache['legacy_ids'])}
    for name in ('native_hardware_raw_controls', 'native_a100_raw_controls'):
        run = directory/name
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['status'] == 'completed' and manifest['plan_sha256'] == digest(OUT/'inference_plan.json')
        assert manifest['records_sha256'] == digest(run/'records.jsonl')
        assert manifest['config'] == plan['models'][model]
        control_manifests[name] = manifest
        records = [json.loads(line) for line in (run/'records.jsonl').open()]
        assert len(records) == 12
        control_runs[name] = {(x['condition'],x['legacy_id']):x for x in records}
        for file in (run/'manifest.json',run/'records.jsonl'):
            sources.append(dict(path=str(file.relative_to(ROOT)), sha256=digest(file)))
    a, b = control_runs.values()
    ma, mb = control_manifests.values()
    assert ma['slot_ids'] == mb['slot_ids'] and ma['tokenizer_hashes'] == mb['tokenizer_hashes']
    assert ma['model_revision'] == mb['model_revision'] == plan['models'][model]['model_revision']
    assert a.keys() == b.keys()
    for key in a:
        assert a[key]['prompt_sha256'] == b[key]['prompt_sha256']
        assert a[key]['input_tokens_before_crop'] == b[key]['input_tokens_before_crop']
    errors = {name:max(float(np.max(np.abs(np.array(x['probs'])-cache['probabilities'][lookup[x['legacy_id']]]))) for x in run.values())
              for name,run in control_runs.items()}
    assert errors['native_a100_raw_controls'] <= 2e-5
    h100_path = directory/'chat/manifest.json'
    h100 = json.loads(h100_path.read_text())
    assert h100['status'] == 'completed' and h100['records_sha256'] == digest(directory/'chat/records.jsonl')
    assert sum(1 for _ in (directory/'chat/records.jsonl').open()) == 5214
    for file in (h100_path,directory/'chat/records.jsonl'):
        sources.append(dict(path=str(file.relative_to(ROOT)),sha256=digest(file),usage='excluded H100 diagnostic'))
    result = dict(version='review-sensitivities-v1', primary_files_unchanged=len(primary_files),
                  plan_sha256=digest(OUT/'inference_plan.json'), scored_prompts=plan['total_scored_prompts'],
                  additional_hardware_diagnostic_scorings=4*(5214+12+12),
                  total_completed_scored_prompts=plan['total_scored_prompts']+4*(5214+12+12),
                  mistral_hardware_controls_max_abs_error=errors,
                  mistral_template_comparison_run='chat_a100; H100 attempt excluded',
                  independent_vectorized_checks=checks, sources=sources, results=rows)
    (OUT/'gpu_sensitivity_results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for phase in ('mmlu', 'medical'):
        selected = [x for x in rows if (x['dataset'] == 'mmlu') == (phase == 'mmlu')]
        with (OUT/(phase+'_sensitivity.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    print(json.dumps(dict(conditions=len(rows), checked_primary_files=len(primary_files), scored_prompts=plan['total_scored_prompts'])))


if __name__ == '__main__':
    main()
