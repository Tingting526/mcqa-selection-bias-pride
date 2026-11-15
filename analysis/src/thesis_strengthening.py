"""Supplementary analyses on fixed scores; no model inference dependencies."""
from __future__ import annotations

from itertools import permutations
import numpy as np

from src.offline_phase3b import EPSILON, correction_scores, label_metrics


def permutation_summary(probabilities, gold, orders):
    """Content-aligned metrics for any complete, balanced set of orderings.

    orders[r, j] is the permanent content displayed at label j in ordering r.
    Consistency concerns agreement across this particular set, so comparing
    four with twenty-four orderings changes the criterion's strictness.
    """
    p, gold, orders = (np.asarray(probabilities, dtype=float),
                       np.asarray(gold, dtype=int), np.asarray(orders, dtype=int))
    n, m, k = p.shape
    if not n or gold.shape != (n,) or orders.shape != (m, k):
        raise ValueError("Aligned nonempty questions, probabilities and orderings required")
    if any(sorted(row) != list(range(k)) for row in orders.tolist()):
        raise ValueError("Each ordering must be a permutation of permanent contents")
    if not np.isfinite(p).all() or (p < 0).any() or ((gold < 0) | (gold >= k)).any():
        raise ValueError("Invalid scores or gold contents")
    if not np.allclose(p.sum(-1), 1, atol=1e-3, rtol=0):
        raise ValueError("Probability rows must sum to one")
    counts = np.stack([(orders == c).sum(0) for c in range(k)])
    if not (counts == m / k).all():
        raise ValueError("Balanced content/label exposure required")
    labels = p.argmax(-1)
    contents = np.take_along_axis(np.broadcast_to(orders, (n, m, k)),
                                 labels[..., None], axis=2)[..., 0]
    correct = contents == gold[:, None]
    displayed_gold = (orders[None, :, :] == gold[:, None, None]).argmax(-1)
    recall = label_metrics(displayed_gold.ravel(), labels.ravel(), k)
    accuracies = 100 * correct.mean(0)
    normalized = p / (p.sum(-1, keepdims=True) + EPSILON)
    aligned = np.zeros((n, k))
    for row, order in enumerate(orders):
        aligned[:, order] += normalized[:, row, :] / m
    averaged = aligned.argmax(-1)
    original_order = np.flatnonzero(np.all(orders == np.arange(k), axis=1))
    if len(original_order) != 1:
        raise ValueError("Exactly one original ordering required")
    original = label_metrics(gold, contents[:, original_order[0]], k)
    aggregation = label_metrics(gold, averaged, k)
    return dict(n=n, orderings=m, original_accuracy_percent=original["accuracy"],
                original_rstd_pp=original["rstd"],
                balanced_rstd_pp=recall["rstd"],
                balanced_recalls_percent=recall["recalls"],
                balanced_supports=recall["supports"],
                accuracy_by_order_percent=accuracies.tolist(),
                mean_accuracy_percent=float(accuracies.mean()),
                accuracy_sd_pp=float(accuracies.std(ddof=0)),
                accuracy_range_pp=float(np.ptp(accuracies)),
                consistency_percent=float(100 * np.all(contents == contents[:, :1], axis=1).mean()),
                aggregation_accuracy_percent=aggregation["accuracy"],
                aggregation_original_rstd_pp=aggregation["rstd"],
                aggregation_predictions=averaged.tolist())


def arc_full_comparison(probabilities, gold):
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[1:] != (24, 4):
        raise ValueError("All 24 four-option orderings required")
    orders = np.array(list(permutations(range(4))))
    cyclic_orders = np.array([np.roll(np.arange(4), -s) for s in range(4)])
    indices = [np.flatnonzero(np.all(orders == order, axis=1))[0] for order in cyclic_orders]
    cyclic = permutation_summary(p[:, indices], gold, cyclic_orders)
    full = permutation_summary(p, gold, orders)
    a, b = np.array(cyclic["aggregation_predictions"]), np.array(full["aggregation_predictions"])
    g = np.asarray(gold)
    gained, lost = int(((a != g) & (b == g)).sum()), int(((a == g) & (b != g)).sum())
    assert full["consistency_percent"] <= cyclic["consistency_percent"] + 1e-10
    assert full["accuracy_range_pp"] + 1e-10 >= cyclic["accuracy_range_pp"]
    return dict(cyclic=cyclic, full=full, cyclic_row_indices=[int(i) for i in indices],
                aggregation_predictions_changed_n=int((a != b).sum()),
                aggregation_correct_gained_n=gained, aggregation_correct_lost_n=lost,
                full_minus_cyclic_accuracy_pp=100 * (gained - lost) / len(g))


def subgroup_endpoints(gold, probabilities, cpa, legacy_ids, trials, groups):
    """Partition existing benchmark-wide predictions; never fit subgroup priors."""
    gold, p, cpa = np.asarray(gold), np.asarray(probabilities), np.asarray(cpa)
    n, k = p.shape[0], p.shape[-1]
    legacy_ids, groups = np.asarray(legacy_ids), np.asarray(groups)
    if any(len(x) != n for x in (gold, cpa, legacy_ids, groups)) or len(set(legacy_ids)) != n:
        raise ValueError("Unique aligned population required")
    if set(groups) != {"single", "multi"} or len(trials) != 5:
        raise ValueError("Two released groups and five original draws required")
    hybrid = []
    for trial in trials:
        estimate = np.isin(legacy_ids, trial["estimation_ids"])
        if int(estimate.sum()) != int(.05 * n):
            raise ValueError("Saved calibration memberships do not match population")
        prediction = correction_scores(p[:, 0], trial["global_prior"]).argmax(-1)
        prediction[estimate] = cpa[estimate].argmax(-1)
        hybrid.append(prediction)
    result = []
    orders = np.array([np.roll(np.arange(k), -s) for s in range(k)])
    for group in ("single", "multi"):
        mask = groups == group
        default = label_metrics(gold[mask], p[mask, 0].argmax(-1), k)
        averaged = label_metrics(gold[mask], cpa[mask].argmax(-1), k)
        h = [label_metrics(gold[mask], row[mask], k) for row in hybrid]
        cyclic = permutation_summary(p[mask], gold[mask], orders)
        ha, hr = float(np.mean([r["accuracy"] for r in h])), float(np.mean([r["rstd"] for r in h]))
        result.append(dict(group=group, n=int(mask.sum()), supports=default["supports"],
                           default_accuracy_percent=default["accuracy"], default_rstd_pp=default["rstd"],
                           cpa_accuracy_percent=averaged["accuracy"], cpa_rstd_pp=averaged["rstd"],
                           hybrid_accuracy_percent=ha, hybrid_rstd_pp=hr,
                           hybrid_minus_default_accuracy_pp=ha-default["accuracy"],
                           hybrid_minus_default_rstd_pp=hr-default["rstd"],
                           cyclic_rstd_pp=cyclic["balanced_rstd_pp"],
                           consistency_percent=cyclic["consistency_percent"],
                           original_correct_default_n=int((p[mask, 0].argmax(-1) == gold[mask]).sum()),
                           hybrid_correct_n_by_draw=[int((row[mask] == gold[mask]).sum()) for row in hybrid]))
    return result


def prior_transfer_decomposition(probabilities, gold, source_prior, target_prior):
    """Exact score-offset explanation of source- versus target-prior selections."""
    p, gold = np.asarray(probabilities), np.asarray(gold)
    source, target = np.asarray(source_prior), np.asarray(target_prior)
    if p.ndim != 2 or gold.shape != (len(p),) or not len(p):
        raise ValueError("Nonempty original-order scores and matching gold required")
    if source.shape != (p.shape[1],) or target.shape != source.shape or min(source.min(), target.min()) <= 0:
        raise ValueError("Positive compatible priors required")
    in_scores, transfer_scores = correction_scores(p, target), correction_scores(p, source)
    offset = np.log(target + EPSILON) - np.log(source + EPSILON)
    np.testing.assert_allclose(transfer_scores, in_scores + offset, atol=1e-12, rtol=0)
    before, after = in_scores.argmax(-1), transfer_scores.argmax(-1)
    ordered = np.sort(in_scores, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    width = float(np.ptp(offset))
    changed = before != after
    # A winner separated by more than the offset range cannot be overtaken.
    assert not np.any(changed & (margin > width + 1e-12))
    gained, lost = (before != gold) & (after == gold), (before == gold) & (after != gold)
    n = len(gold)
    return dict(n=n, prior_total_variation=float(.5 * np.abs(source/source.sum()-target/target.sum()).sum()),
                label_log_offsets=offset.tolist(), log_offset_range=width,
                margin_eligible_n=int((margin <= width + 1e-12).sum()),
                predictions_changed_n=int(changed.sum()), predictions_changed_percent=float(100 * changed.mean()),
                correct_gained_n=int(gained.sum()), correct_lost_n=int(lost.sum()),
                changed_both_wrong_n=int((changed & ~gained & ~lost).sum()),
                accuracy_change_pp=float(100 * (gained.sum()-lost.sum()) / n),
                in_domain_predictions=before.tolist(), transfer_predictions=after.tolist())


def apply_score_patch(ids, probabilities, replacements, required_ids):
    """Create an isolated sensitivity copy, requiring every declared repair."""
    ids = list(ids)
    if len(ids) != len(set(ids)) or set(replacements) != set(required_ids):
        raise ValueError("Unique population and exactly the required repair IDs expected")
    if not set(replacements) <= set(ids):
        raise ValueError("Repair IDs outside population")
    result = np.array(probabilities, dtype=float, copy=True)
    lookup = {sample_id: i for i, sample_id in enumerate(ids)}
    for sample_id, values in replacements.items():
        values = np.asarray(values, dtype=float)
        if values.shape != result.shape[1:] or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid replacement cycle")
        if not np.allclose(values.sum(-1), 1, atol=1e-3, rtol=0):
            raise ValueError("Replacement rows must sum to one")
        result[lookup[sample_id]] = values
    return result


def bridge_source_ids(canonical_ids, legacy_ids):
    """Match audit and frozen-cache namespaces through unique source-row IDs."""
    canonical_ids, legacy_ids = list(canonical_ids), list(legacy_ids)
    if (len(canonical_ids) != len(legacy_ids)
            or len(set(canonical_ids)) != len(canonical_ids)
            or len(set(legacy_ids)) != len(legacy_ids)):
        raise ValueError("A one-to-one source-row ID bridge is required")
    return dict(zip(map(str, legacy_ids), map(str, canonical_ids)))
