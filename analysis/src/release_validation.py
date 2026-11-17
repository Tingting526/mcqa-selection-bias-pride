"""Portable checks of the submitted results, with optional raw-score replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from src.offline_phase3b import (
    COUNTS, MODELS, compare_saved, label_metrics, load_raw, permutation_metrics,
    reproduce, sha256, transfer_pair,
)
from src.phase4_statistics import (
    METHODS, CONTRASTS, bootstrap_accuracy, count_direction, question_indices,
)

PRIMARY = "analysis/outputs/phase3b_20261018/verified/analysis_results.json"
STATISTICS = "analysis/outputs/phase4_20261018_20261018T091030Z/statistics_results.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_equal(expected, actual, context):
    differences = compare_saved(expected, actual)
    if differences:
        raise ValueError(f"{context}: {json.dumps(differences[:5])}")


def check_input_hashes(root):
    files = read_json(root / "provenance/results_manifest.json")["files"]
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Input outside repository: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Submitted result checksum differs: {relative}")
    return len(files)


def unique_rows(rows, keys):
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate rows for {keys}")
    return result


def recompute_statistics(primary, statistics):
    """Recompute endpoints and paired intervals from the saved predictions.

    This checks derived evidence. It cannot verify model logits or how a prior
    was estimated; raw-score replay below performs that separate check.
    """
    summaries = unique_rows(primary["summary_rows"], ("model", "benchmark"))
    expected_pairs = {(m, d) for m in MODELS for d in COUNTS}
    if set(summaries) != expected_pairs:
        raise ValueError("Expected all 30 model/benchmark summaries")
    direction_rows = unique_rows(statistics["direction_rows"], ("model", "benchmark"))
    if set(direction_rows) != expected_pairs:
        raise ValueError("Expected 30 exact-count direction rows")
    for pair, row in direction_rows.items():
        summary = summaries[pair]
        n = COUNTS[pair[1]]
        if row["n"] != n or len(row["hybrid_correct_by_draw"]) != 5:
            raise ValueError(f"Invalid population or draw count: {pair}")
        counts = [row["default_correct"], *row["hybrid_correct_by_draw"]]
        if any(type(x) is not int or not 0 <= x <= n for x in counts):
            raise ValueError(f"Invalid exact success counts: {pair}")
        direction = count_direction(row["default_correct"], row["hybrid_correct_by_draw"])
        check_equal(row["direction"], direction, f"Direction {pair}")
        check_equal(summary["default_accuracy_percent"], 100 * counts[0] / n, f"Default {pair}")
        check_equal(summary["pride_hybrid_accuracy_percent"], 100 * sum(counts[1:]) / (5 * n), f"Hybrid {pair}")
        draws = sorted((r for r in primary["calibration_draws"]
                        if (r["model"], r["benchmark"]) == pair), key=lambda r: r["draw"])
        if len(draws) != 5:
            raise ValueError(f"Expected five calibration draws: {pair}")
        for count, draw in zip(counts[1:], draws):
            check_equal(draw["hybrid_accuracy_percent"], 100 * count / n, f"Draw {pair}")
        check_equal(summary["pride_hybrid_rstd_pp"],
                    float(np.mean([r["hybrid_rstd_pp"] for r in draws])), f"Hybrid RStd {pair}")
    directions = {rq: dict(Counter(r["direction"] for r in direction_rows.values() if r["rq"] == rq))
                  for rq in ("RQ1", "RQ2")}
    check_equal(statistics["direction_counts"], directions, "Direction counts")

    transfers = unique_rows(primary["transfer_results"], ("model", "source", "target"))
    expected_transfers = {(m, s, t) for m in MODELS for s, t in
                          (("medqa", "medmcqa"), ("medmcqa", "medqa"))}
    if set(transfers) != expected_transfers:
        raise ValueError("Expected all 12 medical transfer directions")
    saved_rows = unique_rows(statistics["bootstrap_rows"], ("model", "source", "target", "contrast"))
    expected_rows = {(*key, left + "_minus_" + right) for key in expected_transfers for left, right in CONTRASTS}
    if set(saved_rows) != expected_rows:
        raise ValueError("Expected 36 paired bootstrap intervals")
    recomputed = []
    target_ids = {}
    for key, result in transfers.items():
        rows = result["paired_predictions"]
        ids = [r["sample_id"] for r in rows]
        part = result["target_partition"]
        if (ids != part["heldout"] or len(set(ids)) != len(ids)
                or set(ids) & set(part["estimation"]) or len(ids) + len(part["estimation"]) != part["n"]):
            raise ValueError(f"Invalid heldout membership: {key}")
        if result["target"] in target_ids and ids != target_ids[result["target"]]:
            raise ValueError("Target holdout differs across models")
        target_ids[result["target"]] = ids
        gold = [r["gold_index"] for r in rows]
        metrics = {}
        for method in METHODS:
            prediction = [r[method + "_prediction"] for r in rows]
            metrics[method] = label_metrics(gold, prediction, 4)
            check_equal(result["metrics"][method], metrics[method], f"{key} {method}")
        correctness = np.column_stack([[r[m + "_prediction"] == r["gold_index"] for r in rows]
                                      for m in METHODS])
        indices = question_indices(len(rows), result["target"])
        points, _, intervals = bootstrap_accuracy(correctness, indices)
        for left, right in CONTRASTS:
            contrast = left + "_minus_" + right
            saved = saved_rows[(*key, contrast)]
            check_equal(result["contrasts"][contrast],
                        {m: metrics[left][m] - metrics[right][m] for m in ("accuracy", "rstd")},
                        f"{key} {contrast}")
            lo, hi = intervals[contrast]
            values = dict(difference_pp=points[contrast], ci_lower_pp=lo, ci_upper_pp=hi)
            check_equal({k: saved[k] for k in values}, values, f"Interval {key} {contrast}")
            ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            if (ids_hash != saved["target_ids_sha256"]
                    or hashlib.sha256(indices.tobytes()).hexdigest() != saved["sampled_index_bytes_sha256"]):
                raise ValueError(f"Bootstrap membership or random stream differs: {key}")
            recomputed.append({**saved, **values})
    return {**statistics, "direction_counts": directions, "bootstrap_rows": recomputed}


def archive_path(recorded, root):
    """Resolve a published path inside a relocated, unchanged project archive."""
    prefix = "<project-root>/"
    if not isinstance(recorded, str) or not recorded.startswith(prefix):
        raise ValueError(f"Expected a project-relative archive path: {recorded}")
    result = (root / recorded[len(prefix):]).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes the selected root")
    return result


def replay_raw(primary, root, progress=None):
    """Read the original archive, verify hashes, and replay all 30 conditions."""
    analysis = root / "analysis"
    stores, checks = {}, []
    keys = ("implementation", "upstream_commit", "source_path_seed", "alpha",
            "n_labels", "default", "cyclic_perm", "pride", "trials")
    validations = unique_rows(primary["validation"], ("model", "dataset"))
    if set(validations) != {(m, d) for m in MODELS for d in COUNTS}:
        raise ValueError("Expected 30 raw input records")
    # Check every required path before beginning the more expensive replay.
    for row in validations.values():
        for field in ("summary_path", "manifest_path", "observation_path"):
            path = archive_path(row[field], root)
            if not path.is_file():
                raise FileNotFoundError(f"Archive is incomplete: {path}")
    permutations = unique_rows(primary["permutation_metrics"], ("model", "dataset"))
    for pair, row in validations.items():
        summary_path = archive_path(row["summary_path"], root)
        if sha256(summary_path) != row["summary_sha256"]:
            raise ValueError(f"Archived summary checksum differs: {pair}")
        summary = read_json(summary_path)
        scores, diagnostics = load_raw(
            analysis, archive_path(row["manifest_path"], root), summary,
            records_path=archive_path(row["observation_path"], root),
        )
        computed = reproduce(scores, summary["source_path_seed"])
        check_equal({k: summary[k] for k in keys}, computed, f"Raw summary {pair}")
        expected = {k: v for k, v in permutations[pair].items() if k not in ("model", "dataset", "protocol")}
        check_equal(expected, permutation_metrics(scores), f"Permutation metrics {pair}")
        if scores.dataset in ("medqa", "medmcqa"):
            stores[pair] = scores
        checks.append(dict(model=pair[0], benchmark=pair[1], n=len(scores.ids),
                           summary_sha256=row["summary_sha256"], diagnostics=diagnostics))
        if progress:
            progress(f"Verified raw scores: {pair[0]} / {pair[1]} ({len(scores.ids)} questions)")
    for saved in primary["transfer_results"]:
        computed = transfer_pair(stores[saved["model"], saved["source"]],
                                 stores[saved["model"], saved["target"]])
        check_equal({k: v for k, v in saved.items() if k != "provenance"}, computed,
                    f"Raw transfer {saved['model']} {saved['source']} to {saved['target']}")
    return dict(conditions=len(checks), transfer_directions=12, checks=checks)
