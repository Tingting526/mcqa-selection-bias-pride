"""Pinned original PriDe numerics and five-repeat in-domain evaluation (4/5 IDs)."""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import random

import numpy as np

VERSION = 'zheng-pride-0shot-v1'
EPSILON = 1e-10
UPSTREAM_COMMIT = '2ae2f40c77006f7a00a4675bdfb30151c2406691'
DEBIAS_SHA256 = 'f58d4d14074e13cf9f7dfc0a094c8cc6bb6ef578c41918a23e1bdfb456fc4151'


@lru_cache(maxsize=1)
def original_debias():
    path = Path(__file__).resolve().parents[2]/'data/reference/zheng2024/upstream/code/debias_utils.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != DEBIAS_SHA256:
        raise ValueError('Pinned Zheng debias_utils.py checksum differs')
    spec = importlib.util.spec_from_file_location('_pinned_zheng_debias', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_metrics(gold, predictions, n_labels):
    gold, predictions = np.asarray(gold), np.asarray(predictions)
    recalls = [np.mean(predictions[gold == i] == i) if np.any(gold == i) else np.nan
               for i in range(n_labels)]
    return dict(n=len(gold), accuracy=float(np.mean(predictions == gold)*100),
                rstd=float(np.std(recalls)*100), recalls=[float(v*100) for v in recalls])


def evaluate(samples, source_path, alpha=0.05, repeats=5):
    """Match debias_pride.single_process on fixed, sorted source observations.

    ``samples``: ordered dicts with id, gold index, and displayed-ID probability
    matrices (4x4, 24x4 or 5x5). Gold labels enter metrics only, never the prior.
    The source-path byte seed and repeated in-place shuffle follow upstream.
    """
    if not samples or not 0 < alpha <= 1:
        raise ValueError('Nonempty observations and alpha in (0,1] required')
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate question IDs')
    z = original_debias()
    n = len(samples)
    n_labels = np.asarray(samples[0]['probs']).shape[1]
    if n_labels not in (4, 5):
        raise ValueError('Four or five labels required')
    k = int(n*alpha)
    if not k:
        raise ValueError('Too few samples for a nonempty original floor(alpha*N) estimation set')
    decomposed = []
    for sample in samples:
        p = np.asarray(sample['probs'], dtype=np.float64)
        if p.shape not in ((4,4), (24,4), (5,5)) or p.shape[1] != n_labels:
            raise ValueError('Invalid or mixed permutation matrix shapes')
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-3):
            raise ValueError('Invalid observed probabilities')
        if not 0 <= int(sample['gold']) < n_labels:
            raise ValueError('Invalid gold index')
        observed, cpa, prior = z.simple(p)
        decomposed.append(dict(id=sample['id'], gold=int(sample['gold']),
                               original=p[0], cpa=cpa, prior=prior,
                               estimation_cost=len(observed)))
    gold = [s['gold'] for s in decomposed]
    default = label_metrics(gold, [int(s['original'].argmax()) for s in decomposed], n_labels)
    cpa = label_metrics(gold, [int(s['cpa'].argmax()) for s in decomposed], n_labels)
    rng = random.Random(source_path.encode('utf-8'))
    shuffled = list(decomposed)
    trials = []
    for repeat in range(1 if alpha == 1 else repeats):
        rng.shuffle(shuffled)
        estimation, remaining = shuffled[:k], shuffled[k:]
        prior = np.mean([s['prior'] for s in estimation], axis=0)
        de_pred = [int(s['cpa'].argmax()) for s in estimation]
        dr_pred = [int(np.argmax(np.log(s['original']+EPSILON)-np.log(prior+EPSILON)))
                   for s in remaining]
        trial = dict(repeat=repeat, estimation_ids=[s['id'] for s in estimation],
            global_prior=prior.tolist(),
            full=label_metrics([s['gold'] for s in shuffled], de_pred+dr_pred, n_labels),
            theoretical_cost_factor=float((sum(s['estimation_cost'] for s in estimation)+len(remaining))/n))
        if remaining:
            trial['heldout_default'] = label_metrics([s['gold'] for s in remaining],
                [int(s['original'].argmax()) for s in remaining], n_labels)
            trial['heldout_pride'] = label_metrics([s['gold'] for s in remaining], dr_pred, n_labels)
        trials.append(trial)
    pride = dict(n=n, estimation_n=k, remaining_n=n-k, repeats=len(trials))
    for key in ('accuracy', 'rstd'):
        values = [t['full'][key] for t in trials]
        pride.update({key:float(np.mean(values)), key+'_std':float(np.std(values)),
                      key+'_min':float(np.min(values)), key+'_max':float(np.max(values))})
    return dict(implementation=VERSION, upstream_commit=UPSTREAM_COMMIT,
                source_path_seed=source_path, alpha=alpha, n_labels=n_labels,
                default=default, cyclic_perm=cpa, pride=pride, trials=trials)
