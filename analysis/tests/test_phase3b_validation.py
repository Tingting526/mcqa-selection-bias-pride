"""Regression checks for evidence rejection and summary-only limits."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from src.offline_phase3b import expected_population, load_raw, sha256

ANALYSIS=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("phase3b_analysis",ANALYSIS/"scripts/analyze_phase3b_offline.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.fixture
def pilot(tmp_path):
    paths=[]
    for path in (ANALYSIS/"outputs/lrz-results/zheng-selection-bias").glob("*/manifest.json"):
        manifest=json.loads(path.read_text()); c=manifest.get("config",{})
        if manifest.get("status")=="completed" and c.get("dataset")=="arc" and c.get("scoring_adapter_version")=="last-token-fp32-v1": paths.append(path)
    assert len(paths)==1
    manifest=tmp_path/"manifest.json"; records=tmp_path/"records.jsonl"
    manifest.write_bytes(paths[0].read_bytes()); records.write_bytes(paths[0].with_name("records.jsonl").read_bytes())
    return manifest,records


@pytest.mark.parametrize("change",["gold","options","prompt","duplicate","probabilities","trace"])
@pytest.mark.usefixtures("archive_data")
def test_reject_mutated_observations_even_with_other_metadata_valid(pilot,change):
    manifest,path=pilot
    rows=[json.loads(line) for line in path.read_text().split("\n") if line]
    row=next(r for r in rows if r["shots"]==0 and r["setting"]=="perm")
    if change=="gold": row["data"]["ideal"]="A" if row["data"]["ideal"]!="A" else "B"
    if change=="options": row["data"]["options"][0]+="altered"
    if change=="prompt": row["data"]["prompt"]+="altered"
    if change=="duplicate": rows.append(copy.deepcopy(row))
    if change=="probabilities": row["data"]["probs"][0]=[.5]*4
    if change=="trace": row["trace"][0]["option_slot_logits"]=[0]*8
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    with pytest.raises(ValueError): load_raw(ANALYSIS,manifest,allow_pilot=True)


@pytest.mark.usefixtures("archive_data")
def test_actual_hash_must_match_saved_summary_before_parsing(pilot):
    manifest,records=pilot
    summary=dict(observation_manifest_sha256=sha256(manifest),observations_sha256="0"*64)
    with pytest.raises(ValueError,match="SHA256"): load_raw(ANALYSIS,manifest,summary,allow_pilot=True)


@pytest.mark.usefixtures("reference_data")
def test_reference_reader_preserves_lexical_and_executed_na_representations():
    population,source=expected_population(ANALYSIS,"mmlu")
    row=population["college_computer_science:10"]
    assert row["options"][0]=="None"
    assert row["executed_options"][0]=="nan"
    assert row["gold"]==3 and row["retain"]
    differences=source["input_representation_differences"]
    assert len(differences)==10 and all(r["retained"] for r in differences)
    assert all(d["source_text"]=="None" and d["executed_text"]=="nan" for r in differences for d in r["differences"])


@pytest.mark.parametrize("change",["partition","aggregate","recall"])
@pytest.mark.usefixtures("archive_data")
def test_summary_checks_detect_partition_and_numeric_changes(change):
    summary=json.loads((ANALYSIS/"outputs/lrz-results/pride-reproduction-20261016/results/qwen3-8b/medqa/result.json").read_text())
    population,_=expected_population(ANALYSIS,"medqa")
    assert module.check_summary(summary,population)[0]==[]
    if change=="partition": summary["trials"][0]["estimation_ids"][0:2]=list(reversed(summary["trials"][0]["estimation_ids"][0:2]))
    if change=="aggregate": summary["pride"]["accuracy_std"]+=.1
    if change=="recall": summary["default"]["recalls"][0]+=.1
    assert module.check_summary(summary,population)[0]


def test_synthetic_medical_jsonl_roundtrip_and_bidirectional_transfer(tmp_path,monkeypatch):
    """Exercise the medical parser without representing fixtures as real results."""
    import numpy as np
    import yaml
    import src.offline_phase3b as offline
    model="qwen3-8b"
    model_config=yaml.safe_load((ANALYSIS/"configs/models"/(model+".yaml")).read_text())
    rng=np.random.default_rng(18)
    loaded={}
    for dataset in ("medqa","medmcqa"):
        population={f"{dataset}:{i}":dict(id=f"{dataset}:shared-{i:03d}",gold=i%4,
            options=["a","b","c","d"],question="Synthetic question "+str(i),retain=True) for i in range(40)}
        source=dict(manifest="synthetic_fixture_only",manifest_sha256="a"*64,retained_n=40)
        monkeypatch.setattr(offline,"expected_population",lambda *args,p=population,s=source:(p,s))
        directory=tmp_path/dataset; directory.mkdir()
        manifest=directory/"manifest.json"; records=directory/"records.jsonl"
        state=dict(status="completed",config=dict(model=model,dataset=dataset,limit=None,settings=["cyclic"],
            implementation=offline.IMPLEMENTATION,protocol=offline.RAW_PROTOCOL,scorer_backend="real_transformers",
            scoring_adapter_version="last-token-fp32-v1",model_config=model_config,preprocessing_manifest_sha256="a"*64),
            runtime=dict(model_revision=model_config["model_revision"],tokenizer_revision=None,dtype="synthetic-float64"),
            diagnostic=dict(labels=list("ABCD"),slot_ids=list(range(8)),whitespace_variants_pooled=True,duplicate_slots_preserved=True,prompt_adds_space=False))
        manifest.write_text(json.dumps(state))
        rows=[]
        for i,item in enumerate(population.values()):
            rows.append(dict(shots=0,setting="cyclic",subject=dataset,data=dict(idx=i,ideal="ABCD"[item["gold"]],options=item["options"],
                prompt=offline.raw_prompt(item["question"],item["options"],dataset,dataset),probs=rng.dirichlet(np.ones(4),size=4).tolist())))
        records.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        summary=dict(model=model,dataset=dataset,observation_manifest_sha256=sha256(manifest),observations_sha256=sha256(records))
        loaded[dataset],diagnostics=offline.load_raw(ANALYSIS,manifest,summary)
        assert diagnostics["full_population"] and diagnostics["records_without_logits"]==40
    for source,target in [("medqa","medmcqa"),("medmcqa","medqa")]:
        result=offline.transfer_pair(loaded[source],loaded[target])
        assert result["target_partition"]["n"]==40
        assert len(result["paired_predictions"])==38
        assert all(method["n"]==38 for method in result["metrics"].values())
