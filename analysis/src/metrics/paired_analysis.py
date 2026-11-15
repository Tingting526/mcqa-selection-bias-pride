"""Paired accuracy analysis conditional on one fitted prior and selected population."""
import math
import numpy as np
import pandas as pd


def pair_predictions(baseline, pride):
    required = {'sample_id', 'gold_label', 'predicted_label'}
    for frame in (baseline, pride):
        if not required.issubset(frame.columns) or frame['sample_id'].isna().any():
            raise ValueError('Missing prediction fields or sample IDs')
        if frame['sample_id'].astype(str).duplicated().any():
            raise ValueError('Duplicate sample IDs')
        for col in ('gold_label', 'predicted_label'):
            if not frame[col].isin(list('ABCD')).all():
                raise ValueError('Invalid answer label')
    if 'sample_group' not in pride or not pride.sample_group.isin(['D_e', 'D_r']).all():
        raise ValueError('Missing or invalid estimation groups')
    baseline, pride = baseline.copy(), pride.copy()
    baseline['sample_id'] = baseline.sample_id.astype(str)
    pride['sample_id'] = pride.sample_id.astype(str)
    if set(baseline.sample_id) != set(pride.sample_id):
        raise ValueError('Baseline and PriDe must have identical sample IDs')
    result = baseline[list(required)].merge(
        pride[list(required) + ['sample_group']], on='sample_id',
        suffixes=('_baseline', '_pride'), validate='one_to_one')
    if not result.gold_label_baseline.equals(result.gold_label_pride):
        raise ValueError('Gold labels differ between paired records')
    result['baseline_correct'] = result.predicted_label_baseline == result.gold_label_baseline
    result['pride_correct'] = result.predicted_label_pride == result.gold_label_baseline
    result['answer_changed'] = result.predicted_label_baseline != result.predicted_label_pride
    result['transition'] = np.select(
        [~result.baseline_correct & result.pride_correct,
         result.baseline_correct & ~result.pride_correct,
         result.baseline_correct & result.pride_correct],
        ['wrong_to_right', 'right_to_wrong', 'both_right'], default='both_wrong')
    return result.sort_values('sample_id').reset_index(drop=True)


def exact_mcnemar(improved, worsened):
    """Two-sided exact binomial test of discordant pairs under p=0.5."""
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2)
             for i in range(k+1)]
    maximum = max(terms)
    return min(1.0, 2 * math.exp(maximum) * sum(math.exp(t-maximum) for t in terms))


def summarize_pairs(frame, *, seed=42, resamples=20000):
    if frame.empty or resamples < 1000:
        raise ValueError('Need nonempty pairs and at least 1000 bootstrap resamples')
    counts = frame.transition.value_counts()
    improved, worsened = int(counts.get('wrong_to_right', 0)), int(counts.get('right_to_wrong', 0))
    n = len(frame)
    # Resampling counts of {-1,0,+1} equals resampling paired per-question differences.
    draws = np.random.default_rng(seed).multinomial(
        n, [worsened/n, (n-improved-worsened)/n, improved/n], size=resamples)
    bootstrap_delta = (draws[:, 2] - draws[:, 0]) / n
    lo, hi = np.quantile(bootstrap_delta, [.025, .975])
    return dict(n=n, baseline_accuracy=float(frame.baseline_correct.mean()),
                pride_accuracy=float(frame.pride_correct.mean()),
                delta_accuracy=(improved-worsened)/n,
                wrong_to_right=improved, right_to_wrong=worsened,
                both_right=int(counts.get('both_right', 0)), both_wrong=int(counts.get('both_wrong', 0)),
                changed_answers=int(frame.answer_changed.sum()),
                ci95_low=float(lo), ci95_high=float(hi),
                mcnemar_exact_p=exact_mcnemar(improved, worsened),
                bootstrap_seed=seed, bootstrap_resamples=resamples)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted
