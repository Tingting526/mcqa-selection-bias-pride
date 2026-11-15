"""Synthetic scientific invariants for the new estimands and split policies."""
import random
import numpy as np
import pytest
from src.phase6_robustness import *

def part():
    order=[f'd:{i:03d}' for i in random.Random(42).sample(range(100),80)]
    return dict(estimation_order=order,estimation=sorted(order),heldout=sorted(set(f'd:{i:03d}' for i in range(100))-set(order)))

def test_nested_is_random_order_not_sorted():
    p=part();a=nested_budget(p,63)
    assert a==p['estimation_order'][:63] and a!=p['estimation'][:63]
    assert set(a)<=set(p['estimation']) and not set(a)&set(p['heldout'])

@pytest.mark.parametrize('change',['order','budget','duplicates'])
def test_invalid_nested_rejected(change):
    p=part();budget=63
    if change=='order':p.pop('estimation_order')
    if change=='budget':budget=81
    if change=='duplicates':p['estimation_order'][1]=p['estimation_order'][0]
    with pytest.raises((ValueError,KeyError)):nested_budget(p,budget)

@pytest.mark.parametrize('kind',['overlap','unknown','duplicate_fit','duplicate_eval'])
def test_partition_rejects_leakage_and_invalid_ids(kind):
    pop=['a','b','c'];fit=['a'];ev=['b','c']
    if kind=='overlap':ev.append('a')
    if kind=='unknown':fit=['x']
    if kind=='duplicate_fit':fit*=2
    if kind=='duplicate_eval':ev*=2
    with pytest.raises(ValueError):validate_partition(pop,fit,ev)

def test_repeated_fit_requires_explicit_policy_and_multiplicity_matters():
    priors=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1]])
    with pytest.raises(ValueError):average_prior(priors,['a','b'],['a','a','b'])
    np.testing.assert_allclose(average_prior(priors,['a','b'],['a','a','b'],repeated=True),[.5,.3,.1,.1])

def test_full_pool_seed_control_degenerate_and_resampling_reproducible():
    p=part()
    assert all(calibration_draw(p,s,'full_pool_without_replacement_control')==p['estimation'] for s in SEEDS)
    draws=[calibration_draw(p,s,'fixed_pool_with_replacement') for s in SEEDS]
    assert len({ids_hash(x) for x in draws})==5
    for seed,x in zip(SEEDS,draws):
        assert x==calibration_draw(p,seed,'fixed_pool_with_replacement')
        assert len(x)==80 and set(x)<=set(p['estimation']) and not set(x)&set(p['heldout'])
        assert len(set(x))<80

@pytest.mark.parametrize('k',[4,5])
def test_content_mapping_and_uniform_prior(k):
    n=20;gold=np.arange(n)%k;labels=(gold[:,None]-np.arange(k))%k
    p=np.full((n,k,k),.01)
    for i in range(n):
        for s in range(k):p[i,s,labels[i,s]]=1-.01*(k-1)
    a,b=frozen_predictions(p,np.full(k,1/k));np.testing.assert_array_equal(a,labels);np.testing.assert_array_equal(a,b)
    m=cyclic_metrics(gold,a,k)
    assert m['semantic_consistency_percent']==100 and m['mean_permutation_accuracy_percent']==100 and m['cyclic_rstd_pp']==0
    assert m['cyclic_supports']==[n]*k

@pytest.mark.parametrize('k',[4,5])
def test_correction_removes_constructed_label_preference(k):
    rng=np.random.default_rng(5);n=100;content=rng.dirichlet(np.ones(k),size=n);prior=np.arange(1,k+1,dtype=float)**5;prior/=prior.sum()
    p=np.array([[content[i,(np.arange(k)+s)%k]*prior for s in range(k)] for i in range(n)]);p/=p.sum(axis=2,keepdims=True)
    a,b=frozen_predictions(p,prior);m=before_after(content.argmax(axis=1),a,b,k)
    assert m['pride_semantic_consistency_percent']==100
    assert m['delta_semantic_consistency_pp']>0 and abs(m['pride_cyclic_rstd_pp'])<1e-10

def test_epsilon_zero_probabilities_and_first_maximum():
    p=np.zeros((1,4,4));p[:,:,0:2]=.5
    a,b=frozen_predictions(p,np.ones(4)/4)
    assert (a==0).all() and (b==0).all()
    assert cyclic_metrics(np.array([0]),a,4)['semantic_consistency_percent']==0
    assert np.isnan(cyclic_metrics(np.array([0]),a,4)['original_rstd_pp'])

def test_paired_gained_lost_and_missing_support():
    g=np.array([0,1]);a=np.array([[0,0,0,0],[0,0,0,0]]);b=np.array([[0,3,2,1],[1,0,3,2]])
    r=before_after(g,a,b,4)
    assert r['consistency_gained_n']==2 and r['consistency_lost_n']==0
    assert r['original_correct_gained_n']==1 and r['delta_original_accuracy_pp']==50
    assert np.isnan(r['default_original_rstd_pp']) and r['delta_semantic_consistency_pp']==100

def test_seed_summary_uses_sample_sd_and_exact_counts():
    rows=[dict(seed=s,transfer_minus_in_domain_accuracy_pp=float(x),default_accuracy_percent=50.,source_prior=[.25]*4,target_prior=[.25]*4,source_prior_l1_from_primary=0.,target_prior_l1_from_primary=0.) for s,x in zip(SEEDS,[-2,-1,0,1,2])]
    r=seed_statistics(rows)
    assert r['transfer_minus_in_domain_accuracy_pp_seed_sample_sd']==pytest.approx(np.sqrt(2.5))
    assert r['transfer_minus_in_domain_accuracy_pp_seed_positive_n']==2 and r['transfer_minus_in_domain_accuracy_pp_seed_zero_n']==1
    with pytest.raises(ValueError):seed_statistics(rows[:-1])

def test_predeclared_subject_groups_and_hash_order():
    assert subject_group('anatomy','core5')=='medical'
    assert subject_group('virology','core5')=='remaining_subjects'
    assert subject_group('virology','broad8')=='medical'
    assert subject_group('professional_psychology','broad8')=='remaining_subjects'
    assert ids_hash(['a','b'])!=ids_hash(['b','a'])

def test_cpa_is_not_mean_permutation_accuracy():
    from src.offline_phase3b import decompose
    p=np.array([[[.6,.4,0,0],[.8,.2,0,0],[0,0,1,0],[0,0,0,1]]])
    cpa=np.array([decompose(p[0])[1]])
    r,_=transfer_metrics(p,np.array([0]),cpa,np.ones(4)/4,np.ones(4)/4)
    assert r['cpa_accuracy_percent']!=cyclic_metrics(np.array([0]),p.argmax(axis=2),4)['mean_permutation_accuracy_percent']
