import numpy as np
import pytest
from src.phase4_statistics import count_direction,question_indices,bootstrap_accuracy,unsupported_label_bound

def test_exact_count_equality_avoids_llama_medmcqa_float_artifact():
    assert count_direction(2334,[2350,2330,2324,2325,2341])=='equal'
    assert count_direction(2334,[2351,2330,2324,2325,2341])=='increase'
    assert count_direction(2334,[2349,2330,2324,2325,2341])=='decrease'
    with pytest.raises(ValueError):count_direction(2334.,[2334])

def test_one_question_unit_and_pairing_not_independent_methods():
    c=np.array([[1,1,1,0],[0,0,0,1],[1,1,1,1],[0,0,0,0]])
    ix=question_indices(4,'medqa',1000)
    p,s,ci=bootstrap_accuracy(c,ix)
    assert ix.shape==(1000,4)
    assert all(v==0 for v in p.values())
    assert all(np.array_equal(v,np.zeros(1000)) for v in s.values())
    assert all(v==(0.,0.) for v in ci.values())

def test_known_draws_and_percentile_contrasts():
    c=np.array([[1,0,1,0],[0,1,1,0]])
    ix=np.array([[0,0],[1,1],[0,1],[1,0]])
    p,s,ci=bootstrap_accuracy(c,ix)
    assert p['in_domain_minus_default']==0
    assert np.array_equal(s['in_domain_minus_default'],[-100,100,0,0])
    assert ci['in_domain_minus_default']==pytest.approx((-92.5,92.5))
    assert p['transfer_minus_default']==50
    assert np.array_equal(s['transfer_minus_in_domain'],[100,0,50,50])

def test_indices_deterministic_and_target_streams_separate():
    assert np.array_equal(question_indices(40,'medqa'),question_indices(40,'medqa'))
    assert not np.array_equal(question_indices(40,'medqa'),question_indices(40,'medmcqa'))

def test_missing_label_risk_is_explicit_not_silently_filtered():
    assert unsupported_label_bound([0,10,10,10])['any_label_absent_log10_union_bound']==0
    balanced=unsupported_label_bound([250]*4)
    assert balanced['any_label_absent_log10_union_bound']<-100
    assert balanced['any_undefined_in_1000_draws_log10_union_bound']==pytest.approx(balanced['any_label_absent_log10_union_bound']+3)

@pytest.mark.parametrize('indices',[np.zeros((2,3),int),np.full((2,4),4),np.zeros((2,4),float)])
def test_invalid_resampling_rejected(indices):
    with pytest.raises(ValueError):bootstrap_accuracy(np.zeros((4,4),int),indices)
