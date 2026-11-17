"""Submission checks must detect changed scientific evidence."""
import copy
from pathlib import Path

import pytest

from src.release_validation import (
    PRIMARY, STATISTICS, archive_path, read_json, recompute_statistics,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence():
    return read_json(ROOT / PRIMARY), read_json(ROOT / STATISTICS)


def test_saved_predictions_reproduce_all_paired_intervals(evidence):
    primary, statistics = evidence
    result = recompute_statistics(primary, statistics)
    assert len(result["bootstrap_rows"]) == 36
    assert result["direction_counts"]["RQ2"] == {"increase": 7, "decrease": 4, "equal": 1}


def test_changed_prediction_is_rejected(evidence):
    primary, statistics = evidence
    changed = copy.deepcopy(primary)
    row = changed["transfer_results"][0]["paired_predictions"][0]
    row["default_prediction"] = (row["gold_index"] + 1) % 4 if row["default_prediction"] == row["gold_index"] else row["gold_index"]
    with pytest.raises(ValueError, match="default"):
        recompute_statistics(changed, statistics)


def test_changed_interval_is_rejected(evidence):
    primary, statistics = evidence
    changed = copy.deepcopy(statistics)
    changed["bootstrap_rows"][0]["ci_lower_pp"] += 0.1
    with pytest.raises(ValueError, match="Interval"):
        recompute_statistics(primary, changed)


def test_missing_condition_is_not_reported_as_a_complete_reproduction(evidence):
    primary, statistics = evidence
    changed = {**primary, "summary_rows": primary["summary_rows"][:-1]}
    with pytest.raises(ValueError, match="30"):
        recompute_statistics(changed, statistics)


def test_archive_can_be_relocated(tmp_path):
    assert archive_path("<project-root>/analysis/results.json", tmp_path) == tmp_path / "analysis/results.json"


@pytest.mark.parametrize("recorded", ["<project-root>/../outside", "/old/path/analysis/file", None])
def test_archive_resolution_cannot_fall_back_to_another_project(tmp_path, recorded):
    with pytest.raises(ValueError):
        archive_path(recorded, tmp_path)
