import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_medqa_seeds import evaluate_seeds
from src.pride.prior_estimation import select_estimation_ids


def fixture():
    baseline, permutations = [], []
    ids = [f'q{i:03}' for i in range(100)]
    for i in ids:
        row = dict(sample_id=i,dataset='test',model='synthetic',gold_label='A',predicted_label='A',
                   correct=True,**{'prob_'+label:.25 for label in 'ABCD'})
        baseline.append(row)
        for p in range(4):
            order = [(j+p)%4 for j in range(4)]
            permutations.append(dict(row,permutation_id=p,option_ordering=order,correct_content_index=0))
    prior = dict(seed=42,alpha=.05,alpha_rounding='floor',sampling_order='sorted_sample_id',
                 estimation_sample_policy='cyclic',epsilon=1e-10,global_prior=[.25]*4,
                 estimation_sample_ids=select_estimation_ids(ids,.05,42))
    return pd.DataFrame(baseline),pd.DataFrame(permutations),prior


def test_seed_population_and_common_holdout():
    table,selected,common=evaluate_seeds(*fixture())
    assert len(table)==30
    assert (table[table.group=='D_r'].n==95).all()
    assert all(len(ids)==5 for ids in selected.values())
    assert not set(common).intersection(set().union(*map(set,selected.values())))
    assert (table.delta_accuracy==0).all()
    again,_,_=evaluate_seeds(*fixture())
    pd.testing.assert_frame_equal(table,again)


def test_missing_permutation_and_incompatible_reference_fail():
    baseline,perms,prior=fixture()
    with pytest.raises(ValueError,match='four cyclic'):
        evaluate_seeds(baseline,perms.iloc[1:],prior)
    prior['global_prior']=[.1,.2,.3,.4]
    with pytest.raises(ValueError,match='reproduce'):
        evaluate_seeds(baseline,perms,prior)
