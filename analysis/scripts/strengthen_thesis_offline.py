#!/usr/bin/env python3
"""Close cached-score gaps identified in the thesis review, without inference."""
from __future__ import annotations
import csv
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from src.offline_phase3b import MODELS, raw_prompt
from src.thesis_strengthening import arc_full_comparison, subgroup_endpoints, prior_transfer_decomposition

OUT = ROOT / "analysis/outputs/thesis_strengthening_20261021"
P3 = ROOT / "analysis/outputs/phase3b_20261018/verified"
P6 = ROOT / "analysis/outputs/phase6_robustness_20261018"
AUDIT = ROOT / "technical_audit/evidence"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_rows(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":"))
                         if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def cache(model, dataset):
    path = P6 / "scores" / f"{model}__{dataset}.npz"
    source = next(row for row in json.loads((P6/"raw_sources.json").read_text())["files"]
                  if row["model"] == model and row["benchmark"] == dataset)
    assert sha(path) == source["score_snapshot_sha256"]
    return np.load(path, allow_pickle=False)


def run_analyses():
    population = json.loads((AUDIT / "canonical_populations.json").read_text())
    primary = list(csv.DictReader((P3 / "reproduction_validation.csv").open()))
    arc, subgroups, transfer, source_files = [], [], [], []
    source_frame = __import__("pandas").read_parquet(AUDIT / "source_medmcqa_validation.parquet")
    metadata = {}
    for row in population["medmcqa"]:
        raw = source_frame.iloc[row["source_index"]]
        assert str(raw["id"]) == row["id"].split(":", 1)[1]
        assert int(raw["cop"]) == row["gold"]
        assert raw["question"].strip() == row["question"]
        assert [raw[x].strip() for x in ("opa", "opb", "opc", "opd")] == row["options"]
        metadata[row["id"]] = raw["choice_type"]
    assert sum(value == "single" for value in metadata.values()) == 2801
    assert sum(value == "multi" for value in metadata.values()) == 1140

    for model in MODELS:
        v = next(row for row in primary if row["model"] == model and row["dataset"] == "arc")
        path = Path(v["observation_path"])
        assert sha(path) == v["expected_observation_sha256"]
        source_files.append(dict(path=str(path), sha256=sha(path)))
        wanted = {row["legacy"]: row for row in population["arc"]}
        stored = {}
        for line in path.open():
            row = json.loads(line)
            if row.get("shots") != 0 or row.get("setting") != "perm":
                continue
            key = f"{row['subject']}:{row['data']['idx']}"
            if key in wanted:
                assert key not in stored
                assert row["data"]["options"] == wanted[key]["options"]
                stored[key] = np.asarray(row["data"]["probs"], dtype=float)
        order = [row["legacy"] for row in population["arc"]]
        assert len(order) == 1165
        if stored:
            assert len(stored) == 1165
            result = arc_full_comparison(np.stack([stored[key] for key in order]),
                                         [wanted[key]["gold"] for key in order])
            expected = json.loads((P3/f"recomputed/{model}/arc/result.json").read_text())["result"]
            assert abs(result["cyclic"]["aggregation_accuracy_percent"] - expected["cyclic_perm"]["accuracy"]) < 1e-9
            arc.append(dict(model=model, status="complete_24_orderings", **result))
        else:
            manifest = json.loads(Path(v["manifest_path"]).read_text())
            assert "perm" not in manifest["config"]["settings"]
            arc.append(dict(model=model, status="only_four_cyclic_orderings_cached"))

        scores = cache(model, "medmcqa")
        summary = json.loads((P3/f"recomputed/{model}/medmcqa/result.json").read_text())["result"]
        groups = [metadata[str(i)] for i in scores["ids"]]
        group_results = subgroup_endpoints(scores["gold"], scores["probabilities"], scores["cpa"],
                                           scores["legacy_ids"], summary["trials"], groups)
        for row in group_results:
            subgroups.append(dict(model=model, **row))
        # Group partition recombines to the unchanged full-population accuracy.
        for method, key in [("default", "default"), ("cpa", "cyclic_perm"), ("hybrid", "pride")]:
            a = sum(row["n"] * row[method+"_accuracy_percent"] for row in group_results) / 3941
            assert abs(a - summary[key]["accuracy"]) < 1e-9

        for source, target in [("medqa", "medmcqa"), ("medmcqa", "medqa")]:
            record = json.loads((P3 / f"transfer/{model}/{source}_to_{target}.json").read_text())
            scores = cache(model, target)
            lookup = {str(i): j for j, i in enumerate(scores["ids"])}
            paired = record["paired_predictions"]
            ix = [lookup[row["sample_id"]] for row in paired]
            result = prior_transfer_decomposition(scores["probabilities"][ix, 0], scores["gold"][ix],
                                                  record["source_prior"], record["target_prior"])
            assert result["in_domain_predictions"] == [row["in_domain_prediction"] for row in paired]
            assert result["transfer_predictions"] == [row["transfer_prediction"] for row in paired]
            assert abs(result["accuracy_change_pp"] - record["contrasts"]["transfer_minus_in_domain"]["accuracy"]) < 1e-9
            transfer.append(dict(model=model, source=source, target=target, **result))
        print("Completed cached analyses:", model, flush=True)
    save("analysis_results.json", dict(arc=arc, subgroups=subgroups, transfer=transfer))
    csv_rows("arc_full_permutations.csv", [dict(model=r["model"],
        **{condition+"_"+key: value for condition in ("cyclic", "full")
           for key, value in r[condition].items() if key != "aggregation_predictions"},
        **{key: value for key, value in r.items() if key not in ("model", "cyclic", "full")})
        for r in arc if r["status"] == "complete_24_orderings"])
    csv_rows("medmcqa_subgroups.csv", subgroups)
    csv_rows("transfer_explanation.csv", [{key: value for key, value in r.items()
               if key not in ("in_domain_predictions", "transfer_predictions")} for r in transfer])
    save("analysis_sources.json", dict(files=source_files, canonical_population_sha256=sha(AUDIT/"canonical_populations.json"),
        subgroup_source_sha256=sha(AUDIT/"source_medmcqa_validation.parquet"), model_inference_executed=False))


def medical_token_lengths():
    """Retokenize exact executed medical serializations, with weights disabled."""
    os.environ["USE_TORCH"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    population = json.loads((AUDIT / "canonical_populations.json").read_text())
    old = json.loads((AUDIT / "tokenization_results.json").read_text())
    token_manifest = OUT / "local_tokenizer_manifest.json"
    entries = json.loads((token_manifest if token_manifest.exists() else AUDIT / "tokenizer_file_manifest.json").read_text())
    signatures = json.loads((P6 / "raw_sources.json").read_text())["files"]
    results, lengths = [], []
    for model in MODELS:
        files = [row for row in entries if row["model"] == model]
        if not files:
            results.append(dict(model=model, status="unavailable_gated_tokenizer", datasets=["medqa", "medmcqa"]))
            continue
        for row in files:
            assert sha(row["path"]) == row["sha256"]
        local = Path(files[0]["path"]).parent
        tok = AutoTokenizer.from_pretrained(local, use_fast=False, add_bos_token=False,
                                            add_eos_token=False, trust_remote_code=False, local_files_only=True)
        slots = [tok(": "+label).input_ids[-1] for label in "ABCD"] + [tok(":"+label).input_ids[-1] for label in "ABCD"]
        trailing = tok(": A").input_ids[-1] == tok(":A").input_ids[-1]
        for dataset in ("medqa", "medmcqa"):
            earlier = next((row for row in old if row["model"] == model and row.get("benchmark") == dataset), None)
            signature = next(row["signature"] for row in signatures if row["model"] == model and row["benchmark"] == dataset)
            assert slots == signature["slot_ids"]
            count, truncated, maximum = 0, [], 0
            for item in population[dataset]:
                options = item["options"]
                values = [len(tok(raw_prompt(item["question"], options[s:]+options[:s], dataset, dataset)
                                  + (" " if trailing else ""), truncation=False).input_ids) for s in range(4)]
                maximum = max(maximum, *values)
                count += sum(value > 1536 for value in values)
                if max(values) > 1536:
                    truncated.append(item["id"])
                lengths.append(dict(model=model, dataset=dataset, sample_id=item["id"],
                                    **{f"tokens_shift_{s}": v for s, v in enumerate(values)}))
            if earlier is not None:
                assert maximum == earlier["cyclic_length_max"] and len(truncated) == earlier["truncated_questions"]
            results.append(dict(model=model, dataset=dataset, status="recomputed_pinned_tokenizer",
                                tokenizer_revision=local.name, n=len(population[dataset]), scored_prompts=4*len(population[dataset]),
                                max_tokens=maximum, truncated_questions=len(truncated), truncated_scorings=count,
                                truncated_ids=truncated, candidate_slots_match=True,
                                historical_tokenizer_bytes_certified=False))
            print("Medical tokenizer lengths:", model, dataset, "max", maximum, "truncated", len(truncated), flush=True)
    save("medical_token_lengths.json", results)
    csv_rows("medical_token_lengths_by_question.csv", lengths)
    assert "torch" not in sys.modules


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if "--tokenizers-only" in sys.argv:
        medical_token_lengths()
    else:
        run_analyses()
