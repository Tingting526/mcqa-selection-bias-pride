#!/usr/bin/env python3
"""Plan, execute and summarize isolated MMLU/input-template sensitivities.

Planning and summarizing are CPU-only. Execution is opt-in, requires a Slurm
CUDA allocation, and never writes to the original evidence. Each model loads
once. Both candidate whitespace variants retain the reference pooling rule.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from src.offline_phase3b import MODELS, Scores, raw_prompt, reproduce
from src.thesis_strengthening import apply_score_patch, bridge_source_ids

OUT = ROOT / "analysis/outputs/thesis_strengthening_20261021"
P3 = ROOT / "analysis/outputs/phase3b_20261018/verified"
P6 = ROOT / "analysis/outputs/phase6_robustness_20261018"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write("\n")


def create_plan(path):
    pop_path = ROOT / "technical_audit/evidence/canonical_populations.json"
    pop = json.loads(pop_path.read_text())
    differences = json.loads((P3 / "input_representation_audit.json").read_text())["mmlu"]["differences"]
    repairs = {row["legacy_id"] for row in differences if row["retained"]}
    requests = []
    for item in pop["mmlu"]:
        if item["legacy"] not in repairs:
            continue
        subject, index = item["legacy"].rsplit(":", 1)
        source = ROOT / f"analysis/data/reference/zheng2024/upstream/code/data_mmlu/test/{subject}_test.csv"
        literal = list(csv.reader(source.open()))[int(index)]
        assert ord(literal[-1]) - ord("A") == item["gold"]
        assert literal[0].strip() == item["question"].strip()
        for condition, options in [("mmlu_literal", literal[1:5]), ("mmlu_cached_control", item["options"])]:
            requests.append({**item, "condition": condition, "dataset": "mmlu", "options": options})
    assert len(requests) == 20
    for item in [row for row in pop["mmlu"] if row["legacy"] not in repairs][:2]:
        requests.append(dict(condition="mmlu_unchanged_control", dataset="mmlu", **item))
    for dataset in ("medqa", "medmcqa"):
        requests.extend(dict(condition="medical_native_chat", dataset=dataset, **item) for item in pop[dataset])
    plan = dict(version="thesis-sensitivity-v1", models={model: yaml.safe_load(
        (ROOT / f"analysis/configs/models/{model}.yaml").read_text()) for model in MODELS},
        requests=requests, repaired_ids=sorted(row["id"] for row in requests if row["condition"] == "mmlu_literal"),
        requests_per_model=len(requests), scored_prompts_per_model=4*len(requests),
        total_scored_prompts=4*len(requests)*len(MODELS),
        mmlu_scored_prompts_all_models=22*4*len(MODELS),
        medical_scored_prompts_all_models=(1273+3941)*4*len(MODELS),
        context_cap=1536, candidate_pooling="reference_2K_slots_float32_softmax_duplicate_slots_retained",
        native_template="user message contains exact raw task text; add_generation_prompt=True; model YAML chat kwargs",
        calibration="reuse original model-specific five calibration memberships by original source-path seed",
        population_source_sha256=sha(pop_path), inference_executed=False)
    write_new(path, plan)
    print(json.dumps({key: value for key, value in plan.items() if key not in ("requests", "models")}, indent=2))


def execute(plan_path, model_name, directory, conditions):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real execution requires an explicit Slurm CUDA allocation")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU execution is disabled")
    plan = json.loads(plan_path.read_text()); config = plan["models"][model_name]
    requests = [row for row in plan["requests"] if row["condition"] in conditions]
    if not requests:
        raise ValueError("No selected requests")
    if directory.is_dir() and not (directory / "manifest.json").exists() and not (directory / "records.jsonl").exists():
        directory.rename(directory.with_name(directory.name+".initialization-failed-"+os.environ["SLURM_JOB_ID"]))
    directory.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_repo"], revision=config["tokenizer_revision"],
        use_fast=False, add_bos_token=False, add_eos_token=False, trust_remote_code=False)
    old_cache = Path(os.environ.get("HF_HUB_CACHE", Path.home()/".cache/huggingface/hub"))
    snapshot = old_cache / ("models--"+config["model_repo"].replace("/", "--")) / "snapshots" / config["model_revision"]
    cached_weights = snapshot.is_dir() and bool(list(snapshot.glob("*.safetensors")))
    weight_cache = old_cache if cached_weights else Path(os.environ.get("MCQ_REVIEW_WEIGHT_CACHE", ROOT / "work_cache")) / ("mcq-review-"+os.environ["SLURM_JOB_ID"])
    previous = directory.parent / "mmlu/manifest.json"
    if not cached_weights and previous.exists():
        candidate = Path(json.loads(previous.read_text()).get("weight_cache", "/nonexistent"))
        if not candidate.is_dir(): candidate = ROOT / "work_cache" / candidate.name
        if candidate.is_dir(): weight_cache = candidate
    weight_cache.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_repo"], revision=config["model_revision"], cache_dir=str(weight_cache),
        device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=False, use_safetensors=True)
    model.eval()
    slots = [tokenizer(": "+label).input_ids[-1] for label in "ABCD"] + [tokenizer(":"+label).input_ids[-1] for label in "ABCD"]
    trailing = tokenizer(": A").input_ids[-1] == tokenizer(":A").input_ids[-1]
    tokenizer_hashes = {}
    # Hash the actual tokenizer serialization as loaded, without model weights.
    tokenizer.save_pretrained(directory / "tokenizer")
    for file in (directory / "tokenizer").iterdir():
        if file.is_file(): tokenizer_hashes[file.name] = sha(file)
    manifest = dict(model=model_name, config=config, plan_sha256=sha(plan_path), slot_ids=slots,
        tokenizer_hashes=tokenizer_hashes, model_revision=getattr(model.config, "_commit_hash", None),
        condition_counts={condition: sum(row["condition"] == condition for row in requests) for condition in conditions},
        weight_cache=str(weight_cache), status="running", job_id=os.environ["SLURM_JOB_ID"], inference_executed=True)
    write_new(directory / "manifest.json", manifest)
    with (directory / "records.jsonl").open("x") as stream:
        for item in requests:
            probabilities, lengths, prompt_hashes = [], [], []
            for shift in range(4):
                options = item["options"][shift:] + item["options"][:shift]
                text = raw_prompt(item["question"], options, item["dataset"], item["subject"])
                if item["condition"] == "medical_native_chat":
                    rendered = tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=False,
                        add_generation_prompt=True, **config.get("chat_template_kwargs", {}))
                    tokens = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
                else:
                    rendered = text + (" " if trailing else "")
                    tokens = tokenizer(rendered, return_tensors="pt").input_ids
                lengths.append(int(tokens.shape[-1])); prompt_hashes.append(hashlib.sha256(rendered.encode()).hexdigest())
                tokens = tokens[..., -plan["context_cap"]:].to(model.device)
                with torch.inference_mode():
                    logits = model(input_ids=tokens).logits[0, -1, slots].float()
                    values = torch.softmax(logits, dim=-1).cpu().numpy().reshape(2, 4).sum(0)
                probabilities.append(values.tolist())
            stream.write(json.dumps(dict(sample_id=item["id"], legacy_id=item["legacy"], dataset=item["dataset"],
                condition=item["condition"], gold_index=item["gold"], probs=probabilities,
                input_tokens_before_crop=lengths, prompt_sha256=prompt_hashes)) + "\n")
            stream.flush()
    manifest.update(status="completed", records_sha256=sha(directory / "records.jsonl"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(plan_path, directory, output):
    plan, manifest = json.loads(plan_path.read_text()), json.loads((directory/"manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["plan_sha256"] == sha(plan_path)
    assert manifest["records_sha256"] == sha(directory/"records.jsonl")
    model = manifest["model"]; records = [json.loads(line) for line in (directory/"records.jsonl").open()]
    result = dict(model=model, condition="separate sensitivity; original primary results retained", datasets={})
    for dataset in ("mmlu", "medqa", "medmcqa"):
        condition = "mmlu_literal" if dataset == "mmlu" else "medical_native_chat"
        selected = [row for row in records if row["condition"] == condition and row["dataset"] == dataset]
        if not selected: continue
        if len({row["sample_id"] for row in selected}) != len(selected):
            raise ValueError("Duplicate sensitivity observations")
        cache = np.load(P6/f"scores/{model}__{dataset}.npz", allow_pickle=False)
        original = json.loads((P3/f"recomputed/{model}/{dataset}/result.json").read_text())["result"]
        # Reference audit IDs and frozen score IDs use different namespaces.
        # Stable source-row IDs provide the explicit one-to-one bridge.
        canonical = bridge_source_ids(cache["ids"], cache["legacy_ids"])
        mapping = {canonical[row["legacy_id"]]: row["probs"] for row in selected}
        required = [canonical[row["legacy"]] for row in plan["requests"] if row["condition"] == "mmlu_literal"] if dataset == "mmlu" else cache["ids"].tolist()
        patched = apply_score_patch(cache["ids"].tolist(), cache["probabilities"], mapping, required)
        lookup = {str(sample_id): j for j, sample_id in enumerate(cache["ids"])}
        if dataset == "mmlu":
            controls = [row for row in records if row["condition"] in ("mmlu_cached_control", "mmlu_unchanged_control")]
            if len(controls) != 12:
                raise ValueError("All ten old-input and two unchanged controls required")
            max_error = max(float(np.max(np.abs(np.array(row["probs"])-cache["probabilities"][lookup[canonical[row["legacy_id"]]]]))) for row in controls)
            result["mmlu_control_probability_max_abs_error"] = max_error
            if max_error > 2e-5:
                raise ValueError("Execution controls differ from cached scorer; do not attribute changes solely to repaired text")
        scores = Scores(model=model, dataset=dataset, ids=tuple(cache["ids"]), legacy_ids=tuple(cache["legacy_ids"]),
                        gold=cache["gold"], probabilities=patched, signature={"sensitivity_condition":condition})
        derived = reproduce(scores, original["source_path_seed"])
        assert [row["estimation_ids"] for row in derived["trials"]] == [row["estimation_ids"] for row in original["trials"]]
        result["datasets"][dataset] = dict(result=derived, original_primary=original,
            prior_refitted_on_original_memberships=True, full_population_n=len(cache["ids"]))
    write_new(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("execute"); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", choices=["mmlu_literal", "mmlu_cached_control", "mmlu_unchanged_control", "medical_native_chat"],
        default=["mmlu_literal", "mmlu_cached_control", "mmlu_unchanged_control", "medical_native_chat"])
    p = sub.add_parser("summarize"); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan": create_plan(args.output)
    elif args.command == "execute": execute(args.plan, args.model, args.output, args.conditions)
    else: summarize(args.plan, args.run, args.output)


if __name__ == "__main__":
    main()
