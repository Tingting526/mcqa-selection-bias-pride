"""Analytical fixtures for complete orderings and exact transfer accounting."""
from itertools import permutations
import numpy as np
import pytest
from src.thesis_strengthening import arc_full_comparison, permutation_summary, prior_transfer_decomposition
from src.thesis_strengthening import apply_score_patch, bridge_source_ids


def test_all_orderings_content_equivariance():
    orders = np.array(list(permutations(range(4))))
    content_scores = np.array([[.1, .6, .2, .1], [.7, .1, .1, .1]])
    probabilities = np.stack([content_scores[:, order] for order in orders], axis=1)
    result = arc_full_comparison(probabilities, [1, 0])
    assert result["cyclic_row_indices"] == [0, 9, 16, 18]
    assert result["full"]["consistency_percent"] == 100
    assert result["full"]["balanced_rstd_pp"] == 0
    assert result["full"]["aggregation_accuracy_percent"] == 100
    assert result["aggregation_predictions_changed_n"] == 0


def test_constant_label_changes_content_under_permutation():
    probabilities = np.tile([.7, .1, .1, .1], (4, 24, 1))
    result = arc_full_comparison(probabilities, [0, 1, 2, 3])
    assert result["full"]["consistency_percent"] == 0
    assert result["full"]["balanced_supports"] == [24] * 4
    assert result["full"]["balanced_recalls_percent"] == [100, 0, 0, 0]


def test_incomplete_ordering_and_imbalanced_exposure_rejected():
    with pytest.raises(ValueError, match="All 24"):
        arc_full_comparison(np.ones((2, 4, 4))/4, [0, 1])
    with pytest.raises(ValueError, match="Balanced"):
        permutation_summary(np.ones((2, 2, 4))/4, [0, 1], [[0, 1, 2, 3], [1, 0, 2, 3]])


def test_equal_priors_have_no_prediction_or_accuracy_effect():
    result = prior_transfer_decomposition([[.4, .3, .2, .1], [.1, .2, .3, .4]],
                                          [0, 3], [.25]*4, [.25]*4)
    assert result["prior_total_variation"] == 0
    assert result["predictions_changed_n"] == result["accuracy_change_pp"] == 0
    assert result["margin_eligible_n"] == 0


def test_transfer_gain_loss_and_same_distance_do_not_fix_direction():
    source, target = [.1, .3, .3, .3], [.4, .2, .2, .2]
    scores = np.array([[.35, .4, .15, .1], [.25, .45, .2, .1], [.01, .97, .01, .01]])
    result = prior_transfer_decomposition(scores, [0, 1, 1], source, target)
    assert result["predictions_changed_n"] == 2
    assert result["correct_gained_n"] == result["correct_lost_n"] == 1
    assert result["accuracy_change_pp"] == 0
    reverse = prior_transfer_decomposition(scores, [0, 0, 1], target, source)
    assert reverse["prior_total_variation"] == pytest.approx(result["prior_total_variation"])
    assert reverse["accuracy_change_pp"] < 0
    assert result["margin_eligible_n"] >= result["predictions_changed_n"]


def test_patch_preserves_source_and_requires_all_declared_repairs():
    original = np.ones((3, 4, 4))/4
    replacement = np.tile([.7, .1, .1, .1], (4, 1))
    result = apply_score_patch(["a", "b", "c"], original, {"b": replacement}, ["b"])
    assert np.array_equal(original, np.ones((3, 4, 4))/4)
    assert np.array_equal(result[[0, 2]], original[[0, 2]])
    assert np.array_equal(result[1], replacement)
    with pytest.raises(ValueError, match="required repair"):
        apply_score_patch(["a", "b", "c"], original, {}, ["b"])


def test_audit_and_cache_namespaces_join_by_unique_source_row():
    canonical = ["mmlu:zheng2024:mmlu:test:college_computer_science:10",
                 "mmlu:zheng2024:mmlu:test:abstract_algebra:0"]
    legacy = ["college_computer_science:10", "abstract_algebra:0"]
    bridge = bridge_source_ids(canonical, legacy)
    assert bridge[legacy[0]] == canonical[0]
    with pytest.raises(ValueError, match="one-to-one"):
        bridge_source_ids(canonical, [legacy[0], legacy[0]])
    with pytest.raises(ValueError, match="one-to-one"):
        bridge_source_ids(canonical[:1], legacy)
