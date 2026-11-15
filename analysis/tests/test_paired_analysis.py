import pandas as pd
import pytest
from src.metrics.paired_analysis import pair_predictions, summarize_pairs, exact_mcnemar, holm_adjust


def fixture():
    baseline = pd.DataFrame(dict(sample_id=['a','b','c','d'],gold_label=['A']*4,
                                 predicted_label=['B','A','B','A']))
    pride = pd.DataFrame(dict(sample_id=['d','c','b','a'],gold_label=['A']*4,
                              predicted_label=['A','C','B','A'],sample_group=['D_r']*4))
    return baseline, pride


def test_zero_net_gain_can_include_three_answer_changes():
    pairs = pair_predictions(*fixture())
    result = summarize_pairs(pairs)
    assert result['delta_accuracy'] == 0
    assert result['wrong_to_right'] == result['right_to_wrong'] == 1
    assert result['both_wrong'] == result['both_right'] == 1
    assert result['changed_answers'] == 3
    assert result['ci95_low'] < 0 < result['ci95_high']
    assert summarize_pairs(pairs) == result


def test_invalid_pairing_is_rejected():
    base, pride = fixture()
    with pytest.raises(ValueError, match='identical'):
        pair_predictions(base, pride.iloc[:3])
    pride.loc[0, 'gold_label'] = 'B'
    with pytest.raises(ValueError, match='Gold labels'):
        pair_predictions(base, pride)
    with pytest.raises(ValueError, match='Duplicate'):
        pair_predictions(pd.concat([base, base.iloc[:1]]), pride)


def test_exact_test_and_multiplicity():
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(5, 0) == pytest.approx(.0625)
    assert exact_mcnemar(0, 5) == pytest.approx(.0625)
    assert holm_adjust([.04,.01,.03]).tolist() == pytest.approx([.06,.03,.06])


def test_campaign_analysis_and_summary_mismatch(tmp_path):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from analyze_medqa_pairs import analyze
    from src.evaluation.saved_baseline import records_digest
    campaign = tmp_path/'outputs/campaigns/test'
    campaign.mkdir(parents=True)
    entries = []
    # Synthetic fixture only: metadata exercises the real-result gate, no model is run.
    for model in ['qwen3-8b','mistral-7b-instruct-v0.3','glm-4-9b-chat-hf']:
        pipeline = campaign/model
        pipeline.mkdir()
        (pipeline/'pipeline.json').write_text(json.dumps({'status':'completed'}))
        entries.append({'config':{'model':model,'dataset':'medqa','scorer':'real'},'pipeline':str(pipeline)})
        base = pd.DataFrame({'sample_id':[str(i) for i in range(1273)],
                             'gold_label':['A']*1273,'predicted_label':['A']*1273})
        pride = base.copy()
        pride['sample_group'] = ['D_e']*63 + ['D_r']*1210
        for kind in ['baseline','pride']:
            folder = tmp_path/'outputs/raw'/kind/model
            folder.mkdir(parents=True)
            if kind == 'baseline':
                (folder/'parts').mkdir()
                base.to_parquet(folder/'parts/part-000000.parquet')
            else:
                pride.to_parquet(folder/'pride_results.parquet')
                (folder/'metadata.json').write_text(json.dumps({'scorer_backend':'real_transformers',
                    'baseline_source_sha256':records_digest(base.to_dict('records')),
                    'prior_estimation_sample_ids':[str(i) for i in range(63)]}))
        pd.DataFrame([{k:1.0 for k in ['pride_baseline_accuracy','pride_pride_accuracy',
                    'pride_D_r_baseline_accuracy','pride_D_r_pride_accuracy']}]).to_csv(pipeline/'comparison.csv', index=False)
    (campaign/'campaign.json').write_text(json.dumps({'entries':entries}))
    result, details = analyze(campaign, tmp_path/'result', project_root=tmp_path)
    assert len(result) == 9 and len(details) == 3819
    assert (result['mcnemar_exact_p'] == 1).all()
    assert (result['delta_accuracy'] == 0).all()
    path = campaign/'qwen3-8b/comparison.csv'
    summary = pd.read_csv(path)
    summary['pride_D_r_pride_accuracy'] = .9
    summary.to_csv(path,index=False)
    with pytest.raises(ValueError, match='disagree with summary'):
        analyze(campaign, tmp_path/'invalid', project_root=tmp_path)
    assert not (tmp_path/'invalid').exists()
