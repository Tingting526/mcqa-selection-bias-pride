"""Scientific invariants and CPU-only end-to-end checks for Phase 3B."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.offline_phase3b import (
    DEBIAS_SHA256, EPSILON, RAW_PROTOCOL, Scores, compare_saved,
    correct_probabilities, cyclic_orders, decompose, fit_prior, label_metrics,
    load_raw, partition, permutation_metrics, reproduce, select_cycles, sha256,
    transfer_pair, write_json_new,
)

ANALYSIS=Path(__file__).resolve().parents[1]


def synthetic(dataset="medqa", n=80, k=4, seed=8):
    rng=np.random.default_rng(seed)
    probabilities=rng.dirichlet(np.arange(1,k+1),size=(n,k))
    return Scores("synthetic",dataset,tuple(f"{dataset}:item-{i:04d}" for i in range(n)),
        tuple(f"{dataset}:{i}" for i in range(n)),np.arange(n)%k,probabilities,
        dict(protocol=RAW_PROTOCOL,model_revision="test",slot_ids=list(range(2*k))))


def oracle():
    path=ANALYSIS/"data/reference/zheng2024/upstream/code/debias_utils.py"
    assert sha256(path)==DEBIAS_SHA256
    spec=importlib.util.spec_from_file_location("phase3b_test_oracle",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape",[(4,4),(5,5),(24,4)])
def test_independent_decomposition_matches_pinned_oracle(shape):
    rng=np.random.default_rng(80)
    for _ in range(30):
        p=rng.dirichlet(np.ones(shape[1]),size=shape[0])
        expected=oracle().simple(p)
        actual=decompose(p)
        for a,b in zip(expected,actual):
            np.testing.assert_allclose(a,b,atol=1e-14,rtol=0)


def test_arc_subset_is_cyclic_lexicographic_not_first_four():
    p=np.random.default_rng(12).dirichlet(np.ones(4),size=24)
    actual,indices=select_cycles(p)
    assert indices==(0,9,16,18)
    np.testing.assert_array_equal(actual,p[[0,9,16,18]])


@pytest.mark.parametrize("k",[4,5])
def test_all_five_path_seeded_draws_match_production_oracle(k):
    from src.pride.zheng import evaluate
    scores=synthetic(k=k)
    samples=[dict(id=identifier,gold=int(gold),probs=p) for identifier,gold,p in zip(scores.legacy_ids,scores.gold,scores.probabilities)]
    expected=evaluate(samples,"results/test/path",alpha=.05,repeats=5)
    actual=reproduce(scores,"results/test/path")
    assert compare_saved(expected,actual)==[]
    assert len(actual["trials"])==5
    assert all(len(t["estimation_ids"])==4 for t in actual["trials"])


def test_known_probability_correction_and_normalization():
    p=np.array([.1,.2,.3,.4]); q=np.array([.4,.3,.2,.1])
    ratios=(p+EPSILON)/(q+EPSILON)
    expected=ratios/ratios.max(); expected/=expected.sum()+EPSILON
    np.testing.assert_allclose(correct_probabilities(p,q),expected,atol=1e-14)
    assert correct_probabilities(p,q).argmax()==3


def test_prediction_uses_log_correction_without_an_extra_softmax(monkeypatch):
    import src.offline_phase3b as module
    monkeypatch.setattr(module,"correct_probabilities",lambda *a,**k:pytest.fail("A prediction should use direct log-score argmax"))
    reproduce(synthetic(),"a/path")
    transfer_pair(synthetic("medqa"),synthetic("medmcqa",seed=12))


def test_prior_matches_oracle_without_access_to_gold():
    scores=synthetic()
    selected=list(scores.ids[:7])
    expected=np.mean([oracle().simple(p)[2] for p in scores.probabilities[:7]],axis=0)
    np.testing.assert_allclose(fit_prior(scores.probabilities,scores.ids,selected),expected,atol=1e-14)
    assert "gold" not in __import__("inspect").signature(fit_prior).parameters


def test_cpa_mapping_consistency_and_mean_accuracy_are_distinct():
    scores=synthetic(n=4)
    content=np.array([.1,.2,.3,.4])
    scores.probabilities=np.array([[content[list(order)] for order in cyclic_orders(4)]]*4)
    _,cpa,_=decompose(scores.probabilities[0])
    np.testing.assert_allclose(cpa,content/(1+EPSILON),atol=1e-14)
    result=permutation_metrics(scores)
    assert result["semantic_consistency"]==1
    assert result["mean_permutation_accuracy"]==25
    assert result["cyclic_rstd"]==0
    # CPA accuracy is accuracy after averaging content scores, never an alias for
    # the average accuracy over individual permutations. A counterexample:
    scores.gold=np.array([0,0,0,0])
    scores.probabilities=np.array([[[.6,.4,0,0],[.8,.2,0,0],[0,0,1,0],[0,0,0,1]]]*4)
    cpa_predictions=np.array([decompose(p)[1].argmax() for p in scores.probabilities])
    assert label_metrics(scores.gold,cpa_predictions,4)["accuracy"] != permutation_metrics(scores)["mean_permutation_accuracy"]


def test_missing_support_is_nan_and_first_argmax_ties():
    scores=synthetic(n=4); scores.probabilities[:]=.25; scores.gold[:]=0
    assert np.all(scores.probabilities.argmax(axis=2)==0)
    result=label_metrics(scores.gold,np.zeros(4,dtype=int),4)
    assert result["accuracy"]==100 and np.isnan(result["rstd"])
    assert permutation_metrics(scores)["exact_ties_cyclic"]==16


def test_hybrid_ties_count_estimation_and_heldout_without_changing_argmax():
    scores=synthetic(n=40);scores.probabilities[:]=.25
    result=reproduce(scores,"ties/test")
    assert all(t["exact_ties"]==dict(cpa_estimation=2,pride_heldout=38,hybrid=40) for t in result["trials"])


@pytest.mark.parametrize("change",["nan","negative","sum","shape"])
def test_invalid_scores_rejected(change):
    p=np.full((4,4),.25)
    if change=="nan": p[0,0]=np.nan
    if change=="negative": p[0,0]=-.25
    if change=="sum": p[0]*=2
    if change=="shape": p=p[:3]
    with pytest.raises(ValueError): select_cycles(p)


def test_independent_partitions_and_namespace_collisions():
    source,target=synthetic("medqa"),synthetic("medmcqa",seed=9)
    result=transfer_pair(source,target)
    for name,scores in [("source_partition",source),("target_partition",target)]:
        p=result[name]
        assert not set(p["estimation"]) & set(p["heldout"])
        assert set(p["estimation"]) | set(p["heldout"])==set(scores.ids)
        assert len(p["estimation"])==4 and len(p["heldout"])==76
    assert set(result["source_partition"]["estimation"]).isdisjoint(target.ids)
    assert len(result["paired_predictions"])==76
    assert all(row["sample_id"] in result["target_partition"]["heldout"] for row in result["paired_predictions"])
    assert set(result["metrics"])=={"default","in_domain","transfer","cpa"}
    assert {m["n"] for m in result["metrics"].values()}=={76}
    malformed=copy.deepcopy(target); malformed.ids=source.ids
    with pytest.raises(ValueError,match="namespaced"): transfer_pair(source,malformed)


def test_partition_independent_of_source_ids_and_input_order():
    source,target=synthetic("medqa"),synthetic("medmcqa",seed=9)
    first=transfer_pair(source,target)
    source.ids=tuple("medqa:other-"+str(i) for i in range(80))
    second=transfer_pair(source,target)
    assert first["target_partition"]==second["target_partition"]
    assert partition(tuple(reversed(target.ids)),"medmcqa")==partition(target.ids,"medmcqa")


def test_no_test_gold_or_probability_leak_into_fitted_priors():
    source,target=synthetic("medqa"),synthetic("medmcqa",seed=9)
    first=transfer_pair(source,target)
    target.gold=(target.gold+1)%4
    changed_gold=transfer_pair(source,target)
    assert first["source_prior"]==changed_gold["source_prior"]
    assert first["target_prior"]==changed_gold["target_prior"]
    for a,b in zip(first["paired_predictions"],changed_gold["paired_predictions"]):
        assert {k:v for k,v in a.items() if k!="gold_index"}=={k:v for k,v in b.items() if k!="gold_index"}
    heldout=set(first["target_partition"]["heldout"])
    for i,identifier in enumerate(target.ids):
        if identifier in heldout:
            target.probabilities[i]=np.eye(4)
    changed_test=transfer_pair(source,target)
    assert first["source_prior"]==changed_test["source_prior"]
    assert first["target_prior"]==changed_test["target_prior"]


@pytest.mark.parametrize("direction",[("medqa","medmcqa"),("medmcqa","medqa")])
def test_bidirectional_contrasts_are_correctly_paired(direction):
    result=transfer_pair(synthetic(direction[0]),synthetic(direction[1],seed=32))
    for name,left,right in [("in_domain_minus_default","in_domain","default"),
                            ("transfer_minus_default","transfer","default"),
                            ("transfer_minus_in_domain","transfer","in_domain")]:
        for metric in ("accuracy","rstd"):
            assert result["contrasts"][name][metric]==result["metrics"][left][metric]-result["metrics"][right][metric]
    rows=result["paired_predictions"]
    for method in result["metrics"]:
        independent=label_metrics([r["gold_index"] for r in rows],[r[method+"_prediction"] for r in rows],4)
        assert compare_saved(independent,result["metrics"][method])==[]


def test_reject_raw_chat_and_model_mixing():
    source,target=synthetic("medqa"),synthetic("medmcqa")
    target.signature["protocol"]="thesis-mcq-v2"
    with pytest.raises(ValueError,match="Incompatible"): transfer_pair(source,target)
    target=synthetic("medmcqa"); target.model="other"
    with pytest.raises(ValueError,match="Incompatible"): transfer_pair(source,target)


def test_output_is_exclusive_and_nan_becomes_null(tmp_path):
    path=tmp_path/"new.json"
    write_json_new(path,dict(undefined=np.nan))
    assert json.loads(path.read_text())==dict(undefined=None)
    with pytest.raises(FileExistsError): write_json_new(path,{})


def test_cpu_module_import_and_transfer_do_not_load_models():
    program='''
import builtins, sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','datasets','accelerate'}:
        raise AssertionError('Forbidden model dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from src.offline_phase3b import Scores, transfer_pair, RAW_PROTOCOL
import numpy as np
def make(ds):
    return Scores('test',ds,tuple(ds+':'+str(i) for i in range(40)),tuple(str(i) for i in range(40)),np.arange(40)%4,np.full((40,4,4),.25),dict(protocol=RAW_PROTOCOL))
for source,target in [('medqa','medmcqa'),('medmcqa','medqa')]:
    assert len(transfer_pair(make(source),make(target))['paired_predictions'])==38
assert not ({'torch','transformers','datasets','accelerate'} & set(sys.modules))
'''
    subprocess.run([sys.executable,"-c",program],check=True,capture_output=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})


@pytest.mark.parametrize("dataset,run",[("arc","qwen3-8b_arc_0abc7b8b9606"),("csqa","qwen3-8b_csqa_8b67711108fe")])
def test_real_reference_pilot_is_checked_but_not_promoted(dataset,run):
    # Identify the completed fp32 pilot using metadata, not a guessed run name.
    parent=ANALYSIS/"outputs/lrz-results/zheng-selection-bias"
    candidates=[]
    for path in parent.glob("*/manifest.json"):
        m=json.loads(path.read_text()); c=m.get("config",{})
        if c.get("model")=="qwen3-8b" and c.get("dataset")==dataset and m.get("status")=="completed" and c.get("scoring_adapter_version")=="last-token-fp32-v1":
            candidates.append(path)
    assert len(candidates)==1
    scores,diagnostics=load_raw(ANALYSIS,candidates[0],allow_pilot=True)
    assert len(scores.ids)==2 and not diagnostics["full_population"]
    assert diagnostics["logits_reconstructed"]==2*(24 if dataset=="arc" else 5)
    assert diagnostics["truncation_status"]=="verified_none"
    with pytest.raises(ValueError,match="unrestricted"): load_raw(ANALYSIS,candidates[0])
