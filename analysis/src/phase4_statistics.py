"""Question-level, fixed-prior accuracy inference; no model dependencies."""
from __future__ import annotations
import math
import random
import numpy as np

METHODS=('default','in_domain','transfer','cpa')
CONTRASTS=(('in_domain','default'),('transfer','default'),('transfer','in_domain'))
BOOTSTRAP_SEED=20261018
BOOTSTRAP_RESAMPLES=1000

def count_direction(default_correct:int,hybrid_correct:list[int])->str:
    if not isinstance(default_correct,int) or not hybrid_correct or any(not isinstance(x,int) for x in hybrid_correct):
        raise ValueError('Exact integer correct-answer counts required')
    difference=sum(hybrid_correct)-len(hybrid_correct)*default_correct
    return 'increase' if difference>0 else 'decrease' if difference<0 else 'equal'

def hybrid_correct_counts(scores,source_path):
    from src.offline_phase3b import decomposition_arrays,correction_scores
    cpa,priors=decomposition_arrays(scores)
    original=scores.probabilities[:,0];gold=scores.gold
    default=int(np.count_nonzero(original.argmax(axis=1)==gold))
    indices=list(range(len(gold)));rng=random.Random(source_path.encode('utf-8'))
    nfit=int(.05*len(gold));counts=[]
    for _ in range(5):
        rng.shuffle(indices);de=indices[:nfit];dr=indices[nfit:]
        prior=priors[de].mean(axis=0)
        count=int(np.count_nonzero(cpa[de].argmax(axis=1)==gold[de]))
        count+=int(np.count_nonzero(correction_scores(original[dr],prior).argmax(axis=1)==gold[dr]))
        counts.append(count)
    return default,counts

def question_indices(n:int,target:str,resamples:int=BOOTSTRAP_RESAMPLES):
    if n<1 or target not in ('medqa','medmcqa') or resamples<1:raise ValueError('Valid target population required')
    # Each target uses its own stream, shared across models with the same IDs.
    sequence=np.random.SeedSequence([BOOTSTRAP_SEED,0 if target=='medqa' else 1])
    return np.random.Generator(np.random.PCG64(sequence)).integers(0,n,size=(resamples,n),dtype=np.int32)

def bootstrap_accuracy(correctness,indices):
    c=np.asarray(correctness)
    ix=np.asarray(indices)
    if c.ndim!=2 or c.shape[1]!=4 or not len(c) or not np.isin(c,[0,1]).all():raise ValueError('N original questions by four correctness columns required')
    if ix.ndim!=2 or ix.shape[1]!=len(c) or not np.issubdtype(ix.dtype,np.integer) or (ix<0).any() or (ix>=len(c)).any():raise ValueError('One paired question index matrix required')
    point={};samples={};intervals={}
    for left,right in CONTRASTS:
        name=left+'_minus_'+right
        differences=c[:,METHODS.index(left)].astype(np.int8)-c[:,METHODS.index(right)].astype(np.int8)
        point[name]=float(100*differences.mean())
        values=100*differences[ix].mean(axis=1)
        samples[name]=values
        lo,hi=np.quantile(values,[.025,.975],method='linear')
        intervals[name]=(float(lo),float(hi))
    return point,samples,intervals

def unsupported_label_bound(supports):
    counts=np.asarray(supports,dtype=int);n=int(counts.sum())
    if n<=0 or (counts<0).any():raise ValueError('Nonempty gold population required')
    logs=[n*math.log10(1-int(c)/n) if 0<c<n else -math.inf if c==n else 0. for c in counts]
    maximum=max(logs)
    log_bound=min(0.,maximum+math.log10(sum(10**(x-maximum) for x in logs))) if maximum!=-math.inf else -math.inf
    return dict(n=n,supports=counts.tolist(),label_absence_log10_probabilities=logs,
        any_label_absent_log10_union_bound=log_bound,
        any_undefined_in_1000_draws_log10_union_bound=min(0.,log_bound+3),
        definition='Union bound sum_l (1-n_l/N)^N; 1000-fold bound for any undefined draw. No independence of absence events assumed.')
