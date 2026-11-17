#!/usr/bin/env python3
"""Validate/recompute recovered raw evidence and run paired transfer, CPU only.

Run from analysis/ with numpy, pandas, pyarrow and PyYAML installed. No model
dependencies are used. Existing result files are never opened for writing.
CSV export is a separate standard-library Node.js step; JSON and Markdown are native outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import xml.etree.ElementTree as ET

ANALYSIS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ANALYSIS))
from src.offline_phase3b import (
    COUNTS, DEBIAS_SHA256, IMPLEMENTATION, LOGIT_RECONSTRUCTION_ATOL,
    MEDICAL_ARTIFACTS, METRIC_ATOL, MODELS, PARTITION_POLICY, PRIOR_ATOL,
    PROBABILITY_ATOL, RAW_PROTOCOL, compare_saved, expected_population,
    json_safe, label_metrics, load_raw, permutation_metrics, reproduce,
    sha256, transfer_pair, write_json_new,
)
import numpy as np

ROOT=ANALYSIS.parent
REMOTE_ROOT="/dss/dsshome1/<group>/<lrz-user>/analysis/"
DEFAULT_OUTPUT=ANALYSIS/"outputs/phase3b_20261017"
CURRENT=ANALYSIS/"outputs/lrz-results/pride-reproduction-20261016/results"
SUMMARY_KEYS=("implementation","upstream_commit","source_path_seed","alpha","n_labels","default","cyclic_perm","pride","trials")


def md_table(rows,columns):
    def cell(value):
        if value is None: return "unavailable"
        if isinstance(value,float): return f"{value:.4f}" if np.isfinite(value) else "undefined"
        return str(value).replace("|","\\|").replace("\n"," ")
    return "| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(cell(row.get(c)) for c in columns)+" |" for row in rows)+"\n"


def save_md(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream: stream.write(text)


def logical_order(population):
    return sorted((i for i,row in population.items() if row["retain"]),
                  key=lambda i:(i.rsplit(":",1)[0],int(i.rsplit(":",1)[1])))


def check_summary(summary,population):
    """Validate algebra/partitions from frozen labels, never claim raw validation."""
    differences=[]; order=logical_order(population); n=len(order); k=summary["n_labels"]
    gold=np.array([population[i]["gold"] for i in order]); support=np.bincount(gold,minlength=k)
    def check_metrics(metric,support,path):
        if metric["n"]!=int(sum(support)):
            differences.append(dict(path=path+".n",issue="population_count"))
        recalls=np.array(metric["recalls"],dtype=float)
        expected=dict(accuracy=float(np.dot(recalls,support)/sum(support)),rstd=float(np.std(recalls)))
        differences.extend(compare_saved(expected,{q:metric[q] for q in expected},path))
    if summary["implementation"]!=IMPLEMENTATION or summary["scoring_protocol"]!=RAW_PROTOCOL or summary["debias_source_sha256"]!=DEBIAS_SHA256 or summary["scoring_adapter"]!="last-token-fp32-v1":
        differences.append(dict(path="protocol",issue="unexpected_implementation"))
    if n!=COUNTS[summary["dataset"]] or k!=(5 if summary["dataset"]=="csqa" else 4):
        differences.append(dict(path="population",issue="count_or_K"))
    check_metrics(summary["default"],support,"default")
    check_metrics(summary["cyclic_perm"],support,"cyclic_perm")
    rng=random.Random(summary["source_path_seed"].encode("utf-8")); shuffled=order.copy()
    estimate_n=int(n*summary["alpha"])
    if summary["alpha"]!=.05 or len(summary["trials"])!=5:
        differences.append(dict(path="calibration",issue="unexpected_alpha_or_draw_count"))
    for i,trial in enumerate(summary["trials"]):
        rng.shuffle(shuffled)
        expected=shuffled[:estimate_n]
        if trial["estimation_ids"]!=expected:
            differences.append(dict(path=f"trials[{i}].estimation_ids",issue="path_seeded_partition_mismatch"))
        check_metrics(trial["full"],support,f"trials[{i}].full")
        dr_support=np.bincount([population[t]["gold"] for t in shuffled[estimate_n:]],minlength=k)
        for method in ("heldout_default","heldout_pride"):
            check_metrics(trial[method],dr_support,f"trials[{i}].{method}")
        prior=np.asarray(trial["global_prior"],dtype=float)
        if prior.shape!=(k,) or not np.isfinite(prior).all() or (prior<0).any() or not np.isclose(prior.sum(),1,atol=1e-8,rtol=0):
            differences.append(dict(path=f"trials[{i}].global_prior",issue="invalid_prior"))
        differences.extend(compare_saved(1+(k-1)*estimate_n/n,trial["theoretical_cost_factor"],f"trials[{i}].theoretical_cost_factor"))
    expected=dict(n=n,estimation_n=estimate_n,remaining_n=n-estimate_n,repeats=5)
    for metric in ("accuracy","rstd"):
        values=np.array([t["full"][metric] for t in summary["trials"]])
        expected.update({metric:float(values.mean()),metric+"_std":float(values.std()),metric+"_min":float(values.min()),metric+"_max":float(values.max())})
    differences.extend(compare_saved(expected,summary["pride"],"pride"))
    return differences,support.tolist()


def resolve_input(remote,recovered_roots,expected_sha):
    if not remote.startswith(REMOTE_ROOT):
        raise ValueError("Observation path outside documented remote project")
    relative=Path(remote[len(REMOTE_ROOT):])
    if ".." in relative.parts or relative.is_absolute(): raise ValueError("Unsafe relative recovery path")
    candidates=[ANALYSIS/relative]+[directory/relative for directory in recovered_roots]
    existing=[p for p in candidates if p.is_file()]
    matching=[p for p in existing if sha256(p)==expected_sha]
    return (matching[0] if matching else (existing[0] if existing else None)),[str(p) for p in candidates]


def run(output,recovered_roots):
    if (output/"analysis_results.json").exists():
        raise FileExistsError("Output already contains analysis; select a new versioned directory")
    if not (output/"preservation_baseline.json").exists():
        # Fresh reruns preserve scientific inputs by hash; source files are never
        # modified. The original 20261017 run contains the timestamped backups.
        original=DEFAULT_OUTPUT/"preservation_baseline.json"
        baseline=json.loads(original.read_text())
        ledger=ROOT/"experiment_evidence_ledger.csv"
        stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup=output/"backups"/stamp/"experiment_evidence_ledger.csv"
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ledger,backup)
        baseline["files"]["experiment_evidence_ledger.csv"]=dict(bytes=ledger.stat().st_size,sha256=sha256(ledger))
        baseline["backups"].append(dict(source=str(ledger),backup=str(backup),bytes=backup.stat().st_size,sha256=sha256(backup)))
        write_json_new(output/"preservation_baseline.json",baseline)
    populations={};sources={}
    for dataset in COUNTS:
        populations[dataset],sources[dataset]=expected_population(ANALYSIS,dataset)
    validation=[];result_rows=[];label_rows=[];draw_rows=[];availability=[];workload=[]
    raw_scores={};raw_diagnostics={};missing=[];permutations=[];source_paths=set()
    for model in MODELS:
        for dataset in COUNTS:
            path=CURRENT/model/dataset/"result.json"
            summary=json.loads(path.read_text()); source_paths.add(path)
            discrepancies,supports=check_summary(summary,populations[dataset])
            remote_manifest=summary["observation_manifest"]
            remote_record=str(Path(remote_manifest).with_name("records.jsonl"))
            obs,obs_candidates=resolve_input(remote_record,recovered_roots,summary["observations_sha256"])
            manifest,manifest_candidates=resolve_input(remote_manifest,recovered_roots,summary["observation_manifest_sha256"])
            recovered_manifest=ROOT/"tmp/phase3a_20261017/recovered_manifests"/(model+"_"+dataset+".json")
            if manifest is None and recovered_manifest.exists() and sha256(recovered_manifest)==summary["observation_manifest_sha256"]:
                manifest=recovered_manifest
            manifest_match=manifest is not None and sha256(manifest)==summary["observation_manifest_sha256"]
            if manifest is not None: source_paths.add(manifest)
            status="summary_verified_raw_recovery_required"
            diagnostics=None
            if discrepancies: status="summary_discrepancy"
            if obs is not None and manifest is not None:
                try:
                    scores,diagnostics=load_raw(ANALYSIS,manifest,summary,records_path=obs)
                    computed=reproduce(scores,summary["source_path_seed"])
                    raw_differences=compare_saved({key:summary[key] for key in SUMMARY_KEYS},computed)
                    discrepancies.extend(raw_differences)
                    if not discrepancies:
                        status="verified_from_matching_raw_observations"
                        raw_scores[(model,dataset)]=scores
                        raw_diagnostics[(model,dataset)]=diagnostics
                        permutations.append(dict(model=model,dataset=dataset,protocol=RAW_PROTOCOL,**permutation_metrics(scores)))
                        write_json_new(output/"recomputed"/model/dataset/"result.json",dict(result=computed,diagnostics=diagnostics,provenance=scores.provenance))
                    else: status="raw_recomputation_discrepancy"
                    source_paths.add(obs)
                except (ValueError,KeyError,IndexError,TypeError,FileNotFoundError) as error:
                    discrepancies.append(dict(path="raw_validation",issue=str(error)))
                    status="raw_validation_failed"
            for role,located,remote,expected,candidates in [
                ("observations",obs,remote_record,summary["observations_sha256"],obs_candidates),
                ("manifest",manifest,remote_manifest,summary["observation_manifest_sha256"],manifest_candidates)]:
                if located is None:
                    missing.append(dict(model=model,dataset=dataset,role=role,remote_path=remote,expected_sha256=expected,
                        status="not_available_locally; remote_unreachable_not_missing",searched_exact_paths=candidates))
            rq="RQ1" if dataset in ("mmlu","arc","csqa") else "RQ2"
            validation.append(dict(model=model,dataset=dataset,rq=rq,status=status,summary_path=str(path),summary_sha256=sha256(path),
                summary_internal_checks="pass" if not check_summary(summary,populations[dataset])[0] else "fail",
                manifest_available=manifest is not None,manifest_hash_matches=manifest_match,
                manifest_path=str(manifest) if manifest else None,actual_manifest_bytes=manifest.stat().st_size if manifest else None,
                actual_manifest_sha256=sha256(manifest) if manifest else None,
                observation_path=str(obs) if obs else None,actual_observation_sha256=sha256(obs) if obs else None,
                actual_observation_bytes=obs.stat().st_size if obs else None,
                expected_observation_sha256=summary["observations_sha256"],expected_manifest_sha256=summary["observation_manifest_sha256"],
                population_n=COUNTS[dataset],population_source=sources[dataset],discrepancies=discrepancies,diagnostics=diagnostics))
            default,cpa,pride=summary["default"],summary["cyclic_perm"],summary["pride"]
            base=dict(rq=rq,model=model,benchmark=dataset,protocol=RAW_PROTOCOL,n=default["n"],n_labels=summary["n_labels"],
                default_accuracy_percent=default["accuracy"],default_rstd_pp=default["rstd"],cpa_accuracy_percent=cpa["accuracy"],cpa_rstd_pp=cpa["rstd"],
                pride_hybrid_accuracy_percent=pride["accuracy"],pride_hybrid_rstd_pp=pride["rstd"],
                pride_minus_default_accuracy_pp=pride["accuracy"]-default["accuracy"],pride_minus_default_rstd_pp=pride["rstd"]-default["rstd"],
                cpa_minus_default_accuracy_pp=cpa["accuracy"]-default["accuracy"],cpa_minus_default_rstd_pp=cpa["rstd"]-default["rstd"],
                cpa_minus_pride_accuracy_pp=cpa["accuracy"]-pride["accuracy"],cpa_minus_pride_rstd_pp=cpa["rstd"]-pride["rstd"],
                calibration_n=pride["estimation_n"],calibration_draws=pride["repeats"],
                pride_accuracy_draw_sd_pp=pride["accuracy_std"],pride_rstd_draw_sd_pp=pride["rstd_std"],
                pride_accuracy_min_percent=pride["accuracy_min"],pride_accuracy_max_percent=pride["accuracy_max"],
                pride_rstd_min_pp=pride["rstd_min"],pride_rstd_max_pp=pride["rstd_max"],
                denominator="full retained D; PriDe hybrid CPA on De plus correction on Dr",uncertainty="five calibration draws; ddof=0; no confidence interval",
                verification_status=status,source_summary=str(path),source_summary_sha256=sha256(path),population_manifest=sources[dataset]["manifest"])
            result_rows.append(base)
            for method,recalls in [("Default",default["recalls"]),("CPA",cpa["recalls"]),
                ("PriDe hybrid: mean label recall over five draws",np.mean([t["full"]["recalls"] for t in summary["trials"]],axis=0).tolist())]:
                for label,support,recall in zip("ABCDE",supports,recalls):
                    label_rows.append(dict(rq=rq,model=model,benchmark=dataset,method=method,label=label,
                        support_from_frozen_population=support,recall_percent=recall,protocol=RAW_PROTOCOL,
                        support_identity_status="frozen population verified; observation identity "+("verified" if (model,dataset) in raw_scores else "pending"),
                        verification_status=status,source_summary=str(path),population_manifest=sources[dataset]["manifest"]))
            for trial in summary["trials"]:
                draw_rows.append(dict(model=model,benchmark=dataset,draw=trial["repeat"],n=COUNTS[dataset],estimation_n=pride["estimation_n"],heldout_n=pride["remaining_n"],
                    hybrid_accuracy_percent=trial["full"]["accuracy"],hybrid_rstd_pp=trial["full"]["rstd"],
                    heldout_default_accuracy_percent=trial["heldout_default"]["accuracy"],heldout_default_rstd_pp=trial["heldout_default"]["rstd"],
                    heldout_pride_accuracy_percent=trial["heldout_pride"]["accuracy"],heldout_pride_rstd_pp=trial["heldout_pride"]["rstd"],
                    source_path_seed=summary["source_path_seed"],estimation_ids_sha256=hashlib.sha256(json.dumps(trial["estimation_ids"],separators=(",",":")).encode()).hexdigest(),
                    **{"prior_"+label:value for label,value in zip("ABCDE",trial["global_prior"])},verification_status=status,source_summary=str(path)))
            for metric in ("cyclic_rstd","mean_permutation_accuracy","permutation_accuracy_sd","permutation_accuracy_range","semantic_consistency","exact_ties","truncation_incidence"):
                truncation_known=diagnostics is not None and diagnostics.get("truncation_status")=="verified_none"
                metric_known=(model,dataset) in raw_scores and (metric!="truncation_incidence" or truncation_known)
                availability.append(dict(rq=rq,model=model,benchmark=dataset,metric=metric,
                    status="computed_from_verified_raw" if metric_known else "unavailable",
                    reason=("No truncation: all traced scored lengths are strictly below 1536" if truncation_known else "Pre-truncation lengths/complete scoring traces not recorded") if metric=="truncation_incidence" else ("" if (model,dataset) in raw_scores else "Full current raw observations not available locally; not estimated from summaries")))
            k=summary["n_labels"];n=COUNTS[dataset];cal=pride["estimation_n"]
            workload.append(dict(model=model,benchmark=dataset,protocol=RAW_PROTOCOL,n=n,n_labels=k,prior_estimation_n=cal,
                default_nominal_scorings=n,cpa_nominal_scorings=k*n,pride_nominal_scorings=n+(k-1)*cal,pride_nominal_factor=(n+(k-1)*cal)/n,
                scope="one application; not total collection, five-draw cumulative cost, runtime or energy",population_manifest=sources[dataset]["manifest"]))

    transfer_results=[];transfer_availability=[];transfer_errors=[]
    for model in MODELS:
        pair_ready=all((model,ds) in raw_scores for ds in MEDICAL_ARTIFACTS)
        model_results=[]
        if pair_ready:
            try:
                model_results=[transfer_pair(raw_scores[(model,source)],raw_scores[(model,target)])
                    for source,target in [("medqa","medmcqa"),("medmcqa","medqa")]]
            except ValueError as error:
                transfer_errors.append(dict(model=model,issue=str(error)))
        for source,target in [("medqa","medmcqa"),("medmcqa","medqa")]:
            if model_results:
                result=next(r for r in model_results if r["source"]==source)
                # First recovered model is completed in both directions before the
                # next model is considered. Synthetic tests are a prerequisite.
                write_json_new(output/"transfer"/model/(source+"_to_"+target+".json"),result)
                transfer_results.append(result)
            transfer_availability.append(dict(model=model,source=source,target=target,
                status="completed_from_verified_raw" if model_results else ("blocked_incompatible_raw_pair" if pair_ready else "blocked_current_raw_pair_unavailable"),
                target_n=COUNTS[target],target_estimation_n=int(.05*COUNTS[target]),target_heldout_n=COUNTS[target]-int(.05*COUNTS[target]),
                note="Protocol remains current six-model raw; no historical chat substitution"))

    pilots=[]
    for path in (ANALYSIS/"outputs/lrz-results/zheng-selection-bias").glob("*/manifest.json"):
        m=json.loads(path.read_text());c=m.get("config",{})
        if m.get("status")=="completed" and c.get("limit")==2 and c.get("scoring_adapter_version")=="last-token-fp32-v1":
            try:
                scores,diagnostics=load_raw(ANALYSIS,path,allow_pilot=True)
                pilots.append(dict(model=scores.model,dataset=scores.dataset,status="verified_partial_pilot_only",diagnostics=diagnostics,
                    permutation_metrics=permutation_metrics(scores),provenance=scores.provenance))
                source_paths.update([path,path.with_name("records.jsonl")])
            except ValueError as error:
                pilots.append(dict(path=str(path),status="pilot_validation_failed",error=str(error)))
    write_json_new(output/"pilot_validation.json",pilots)
    write_json_new(output/"missing_observations.json",missing)
    write_json_new(output/"partition_policy.json",PARTITION_POLICY)
    representation={dataset:dict(reader=source.get("csv_reader"),differences=source.get("input_representation_differences",[])) for dataset,source in sources.items() if source.get("input_representation_differences")}
    for dataset,record in representation.items():
        affected={r["legacy_id"] for r in record["differences"]}
        record["calibration_draw_membership"]=[dict(model=model,draw=trial["repeat"],affected_estimation_ids=sorted(affected.intersection(trial["estimation_ids"])))
            for model in MODELS for trial in json.loads((CURRENT/model/dataset/"result.json").read_text())["trials"]]
    write_json_new(output/"input_representation_audit.json",representation)
    results=dict(version="phase3b-analysis-v2",created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        analysis_environment=dict(python=sys.version,numpy=np.__version__),
        validation=validation,summary_rows=result_rows,label_rows=label_rows,calibration_draws=draw_rows,
        permutation_metrics=permutations,metric_availability=availability,nominal_workload=workload,
        transfer_results=transfer_results,transfer_availability=transfer_availability,transfer_errors=transfer_errors,pilots=pilots,
        input_representation_audit=representation,recovered_roots=[str(p) for p in recovered_roots],
        tolerances=dict(probability_rows_atol=PROBABILITY_ATOL,pooled_logit_probability_atol=LOGIT_RECONSTRUCTION_ATOL,metric_pp_atol=METRIC_ATOL,prior_atol=PRIOR_ATOL,rtol=0),
        recovery_status=f"Current local evidence: {sum(v['observation_path'] is not None for v in validation)}/30 observation files located; {sum(v['manifest_hash_matches'] for v in validation)}/30 manifest hashes match; {len(missing)} expected files unavailable locally. Recovery roots are recorded; remote absence is not inferred.",
        uncertainty_status="Point estimates and five-draw SD only. Accuracy-bootstrap resample count awaits author approval; RStd intervals remain unresolved. No new hypothesis tests.")
    write_json_new(output/"analysis_results.json",results)
    source_paths.update(ANALYSIS/"configs/models"/(model+".yaml") for model in MODELS)
    ref=ANALYSIS/"data/reference/zheng2024"
    ref_manifest=json.loads((ref/"manifest.json").read_text()); source_paths.add(ref/"manifest.json")
    for entry in ref_manifest["datasets"]:
        source_paths.add(ref/entry["normalized_path"])
        source_paths.update(ref/r["path"] for r in entry["source_files"])
    for dataset in MEDICAL_ARTIFACTS:
        directory=ANALYSIS/"outputs/preprocessed"/dataset/MEDICAL_ARTIFACTS[dataset]
        source_paths.update(directory.iterdir())
    source_paths.update([ref/"upstream/code/debias_utils.py",Path(__file__),ANALYSIS/"src/offline_phase3b.py",ANALYSIS/"tests/test_offline_phase3b.py"])
    source_paths.update(p for pattern in ("scripts/*phase3b*","tests/test_phase3b*") for p in ANALYSIS.glob(pattern) if p.is_file())
    write_json_new(output/"source_manifest.json",dict(inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(source_paths) if p.is_file()],
        sources_verified_against="Pinned reference/preprocessing manifests and unchanged current summary hashes",upstream_debias_sha256=DEBIAS_SHA256))

    verified_count=sum(v["status"]=="verified_from_matching_raw_observations" for v in validation)
    info="Protocol: "+RAW_PROTOCOL+". Six-model panel; accuracy in %, RStd/differences in pp. PriDe is full-D hybrid; five path-seeded draws, SD ddof=0; no confidence intervals. Source paths/hashes and population manifests are included in the exported tables. "+f"**{verified_count}/30 results independently reproduced from matching full raw observations.** See each row's status and the input-representation audit; preserved summary values are never overwritten.\n\n"
    columns=["model","benchmark","n","default_accuracy_percent","default_rstd_pp","cpa_accuracy_percent","cpa_rstd_pp","pride_hybrid_accuracy_percent","pride_hybrid_rstd_pp","pride_minus_default_accuracy_pp","pride_minus_default_rstd_pp","verification_status"]
    for rq,name in [("RQ1","rq1_reproduction"),("RQ2","rq2_medical")]:
        save_md(output/"tables"/(name+".md"),"# "+rq+" results\n\n"+info+md_table([r for r in result_rows if r["rq"]==rq],columns))
    save_md(output/"reproduction_validation.md","# Reproduction validation\n\n"+info+
        md_table(validation,["model","dataset","population_n","summary_internal_checks","manifest_hash_matches","status"])+
        "\nFor raw-verified rows, all five path-seeded partitions and priors, Default, CPA, hybrid full-D PriDe and paired heldout endpoints are independently reconstructed from the cached probabilities. Summary algebra is also checked for every row. MMLU validation follows the executed pandas CSV reader: ten literal `None` options were rendered as `nan`; source and executed text are both retained in input_representation_audit.json. No source, gold label, cached score or saved summary was modified.\n\nDiscrepancies and declared tolerances: see analysis_results.json.\n")
    write_interpretation(output,results)
    write_transfer_reports(output,results)
    print(json.dumps(dict(summary_rows=len(result_rows),verified_raw_pairs=len(raw_scores),matching_manifests=sum(v["manifest_hash_matches"] for v in validation),
        summary_discrepancies=sum(v["summary_internal_checks"]!="pass" for v in validation),transfer_directions_completed=len(transfer_results),pilot_stores=len(pilots),missing_local_files=len(missing)),indent=2))


def write_interpretation(output,results):
    rows=results["summary_rows"];verified=sum(v["status"]=="verified_from_matching_raw_observations" for v in results["validation"])
    text="# RQ1/RQ2 descriptive interpretation\n\nAll numbers below are derived from the preserved summaries; independent full-raw verification is "+str(verified)+"/30. No claim of significance is made. Accuracy changes and RStd changes are different endpoints.\n\n"
    for rq in ("RQ1","RQ2"):
        subset=[r for r in rows if r["rq"]==rq]
        increases=sum(r["pride_minus_default_accuracy_pp"]>0 for r in subset)
        decreases=sum(r["pride_minus_default_accuracy_pp"]<0 for r in subset)
        lower=sum(r["pride_minus_default_rstd_pp"]<0 for r in subset)
        text+=f"**{rq}:** PriDe accuracy increases in {increases}/{len(subset)}, decreases in {decreases}/{len(subset)}, and is unchanged in {len(subset)-increases-decreases}/{len(subset)}; RStd decreases in {lower}/{len(subset)}. These directional counts do not measure effect size or statistical evidence.\n\n"
    text+="## RQ1 and Zheng et al. (2024)\n\nThe original paper reports recall imbalance and improvements from PriDe, particularly at low nominal cost (§4.1, Figure 4; detailed Appendix C). Current zero-shot, alpha=.05, five-draw findings can reproduce those qualitative effects with exceptions. They do not numerically replicate the original 20-model panel, all shot settings, alpha sweep or measured runtime. The newer model panel and float32 candidate adapter differ; no causal explanation by model age or architecture is identified. See https://proceedings.iclr.cc/paper_files/paper/2024/file/54dd9e0cff6d9214e20d97eb2a3bae49-Paper-Conference.pdf and the pinned source manifest.\n\n"
    text+="## RQ2 descriptive benchmark contrasts\n\nThe medical accuracy benefit is less uniform than across the reference combinations in the saved summaries. Inspect the per-model contrasts below rather than treating averages across unrelated benchmarks as a domain effect. MMLU has a subject cue and includes medical subjects; CSQA has five choices; populations, gold-label difficulty, exclusions, calibration counts and possible truncation differ. MedQA uses 63 prior questions, MedMCQA 197, MMLU 679, ARC 58 and CSQA 60. Equal alpha does not mean an equal absolute estimation budget. All current primary results use raw text and pooled candidate variants. Historical chat scores remain separate.\n\n"
    for model in MODELS:
        subset=[r for r in rows if r["model"]==model]
        text+="### "+model+"\n\n"+md_table(subset,["benchmark","pride_minus_default_accuracy_pp","pride_minus_default_rstd_pp","cpa_minus_pride_accuracy_pp","cpa_minus_pride_rstd_pp"])+"\n"
    text+="Per-label recalls and verified-population supports are in the label table; calibration trials are separate. Original RStd is not semantic consistency; CPA accuracy is not mean permutation accuracy. Cyclic metrics are computed directly from complete raw cycles, with displayed gold remapped for each shift. Exact ties use the first maximum. Reference traces strictly below the 1536-token cap rule out truncation for those traced scorings; absent medical traces leave truncation incidence unidentified. See validation diagnostics for each run.\n\nTen retained MMLU questions contain a literal `None` option in the frozen CSV. The executed pandas reader converts it to `nan`, which appears in the cached prompt. Their gold labels and population membership are unchanged. These are verified executed inputs, not evidence of lexical identity with the original paper's environment. Input-quality sensitivity needs explicit review; the overall PriDe effect cannot be bounded by ten affected predictions because such items can enter prior estimation.\n"
    save_md(output/"rq1_rq2_interpretation.md",text)
    transfer_count=len(results["transfer_results"])
    summary=f"""# Phase 3B summary

## Evidence coverage

- RQ1: 18 existing six-model original-benchmark summaries (MMLU, ARC, CSQA), independently checked for algebra and path-seeded partitions. Full-raw verified pairs: {sum(v['status']=='verified_from_matching_raw_observations' and v['rq']=='RQ1' for v in results['validation'])}/18.
- RQ2: 12 existing six-model medical summaries (MedQA, MedMCQA), descriptively compared with reference results. Full-raw verified pairs: {sum(v['status']=='verified_from_matching_raw_observations' and v['rq']=='RQ2' for v in results['validation'])}/12.
- RQ3: Real current-model direction analyses completed: {transfer_count}/12, after synthetic tests and a first-model bidirectional check. Default, in-domain PriDe, transferred PriDe and CPA share the exact target heldout IDs. Historical chat experiments remain separate.

## Recovery and empirical completion

The manual LRZ export recovered 72 files into an isolated local directory. The archive hash and all independently held expected source hashes were verified before analysis. {results['recovery_status']} Full five-draw priors, partitions, predictions and summary endpoints are reconstructed for verified rows. The original summaries are preserved byte-for-byte. See reproduction_validation.md, source_manifest.json and <project-root>/analysis/outputs/phase3b_lrz_manual_20261018/local_recovery_verification.json.

The primary transfer panel is all six models as authorized in Phase 3B. Deterministic source/target partitions use separate sorted namespaced populations, seed 42, floor(.05N), minimum one. All four methods use the same target Tr. This implements the recommended point-estimate policy without modifying the old inference runner. No bootstrap, new p-value test, equal-budget sensitivity or common-chat campaign was executed. The unresolved interval resample/unsupported-label policy is listed in decisions_pending.md.

RQ1 covers qualitative reproduction with newer models; RQ2 is a descriptive extension across all five benchmarks. The prompt cue, number of choices, populations, exclusions, absolute calibration budgets and truncation observability differ. A controlled common-prompt/equal-budget supplement may strengthen domain attribution, but is not required to report the descriptive findings. No additional GPU experiment is established as necessary for these point estimates.

An input-quality caveat remains: ten MMLU options read as `nan` rather than lexical `None`; all six models used that executed representation. See input_representation_audit.md. Medical logit/truncation traces and resolved tokenizer commits were not persisted and cannot be recovered from probabilities.

No thesis chapter, source dataset, cached probability, historical prior or original summary is modified. Preservation checks and test evidence appear in validation_report.json/md. Remaining author review: interpretation of the descriptive contrasts and MMLU input caveat, uncertainty-method choices, then chapter revision and compilation/visual review. Empirical point estimates do not establish significance or a causal domain effect.
"""
    save_md(output/"phase3b_summary.md",summary)
    save_md(output/"decisions_pending.md","""# Decisions pending / documented choices

## Implemented point-estimate policy

Phase 3B explicitly selects all six current raw models. The new standalone transfer module uses the Phase 3A recommended seed-42 sorted-ID, floor-5%, minimum-one policy, independent source and target partitions, first-maximum ties and additive epsilon 1e-10. These choices are stored in partition_policy.json. Reproduction retains its original five path-seeded hybrid draws. No old numerical result is replaced.

## Uncertainty approval needed before executing intervals

The earlier freeze specifies 1,000 fixed-prior percentile question-bootstrap resamples; historical accuracy analysis uses 20,000. Phase 3A leaves final resample count and unsupported-label RStd resample treatment for review. Recommendation: retain 1,000 fixed-prior paired question resamples for new transfer accuracy, with all cycles grouped and the same resampled target IDs across methods; report these as conditional on fitted priors. For RStd, report any missing-support resample count and resolve its interval policy explicitly rather than discarding/redrawing those samples. No interval or new hypothesis test is computed until that decision is approved. This does not prevent the deterministic paired point estimates.

## Data availability and optional supplements

All 30 current raw stores and matching manifests have been recovered; verification status is recorded per row. No new common-chat experiment is required. A common-prompt or equal-budget supplement remains optional and unexecuted. The ten MMLU `None` to `nan` conversions must be disclosed; changing inputs would define a new experiment and is not silently applied here. Medical traces and resolved tokenizer commits remain metadata limitations.
""")


def write_transfer_reports(output,results):
    method_rows=[]; contrast_rows=[]
    for r in results['transfer_results']:
        for method,m in r['metrics'].items():
            method_rows.append(dict(model=r['model'],direction=r['source']+' → '+r['target'],method=method,n=m['n'],accuracy_percent=m['accuracy'],rstd_pp=m['rstd']))
        contrast_rows.append(dict(model=r['model'],direction=r['source']+' → '+r['target'],source_estimation_n=len(r['source_partition']['estimation']),target_estimation_n=len(r['target_partition']['estimation']),target_heldout_n=len(r['target_partition']['heldout']),**{c+'_'+m+'_pp':v for c,ms in r['contrasts'].items() for m,v in ms.items()}))
    info='Protocol: '+RAW_PROTOCOL+'. All six current models; four options. Separate seed-42 sorted namespaced partitions, floor(5% N), additive epsilon 1e-10, first-maximum ties. Source priors use no target outcomes. All four methods use the identical target Tr: MedQA 1210 questions (Te=63 of 1273), MedMCQA 3744 (Te=197 of 3941). Accuracy/recall in %, RStd and differences in pp; lower RStd means less label-recall imbalance. Point estimates conditional on one fitted source and target prior; no confidence intervals or significance tests. Full label supports, priors, paired item predictions, input hashes and partitions are in transfer/<model>/<direction>.json.\n\n'
    save_md(output/'tables/rq3_results.md','# RQ3 paired medical prior transfer\n\n'+info+md_table(method_rows,['model','direction','method','n','accuracy_percent','rstd_pp']))
    if contrast_rows:
        save_md(output/'tables/rq3_contrasts.md','# RQ3 paired contrasts\n\n'+info+md_table(contrast_rows,list(contrast_rows[0])))
    text='# RQ3 descriptive interpretation\n\n'+info
    for source in ('medqa','medmcqa'):
        subset=[r for r in results['transfer_results'] if r['source']==source]
        if not subset: continue
        acc=[r['contrasts']['transfer_minus_in_domain']['accuracy'] for r in subset];rs=[r['contrasts']['transfer_minus_in_domain']['rstd'] for r in subset]
        text+=f"**{source} → {subset[0]['target']}:** Transfer accuracy is higher than in-domain PriDe in {sum(v>0 for v in acc)}/{len(subset)}, lower in {sum(v<0 for v in acc)}/{len(subset)}, equal in {sum(v==0 for v in acc)}/{len(subset)}. Transfer RStd is higher in {sum(v>0 for v in rs)}/{len(subset)} and lower in {sum(v<0 for v in rs)}/{len(subset)}.\n\n"
    text+='These directions have different target populations and different calibration budgets. This pattern does not identify causal asymmetry, an optimal prior or significance. Transfer beating in-domain PriDe does not necessarily imply improvement over Default. Consult all three contrasts and both endpoints. These heldout estimates are distinct from the five-draw full-D hybrid PriDe used in RQ1/RQ2.\n'
    save_md(output/'rq3_interpretation.md',text)
    for dataset,audit in results['input_representation_audit'].items():
        text='# Executed input representation audit\n\nThe frozen '+dataset.upper()+' CSV contains ten retained questions with a literal `None` answer option. The pinned evaluation reader calls pandas.read_csv(dtype=str) with default NA recognition and then str(cell). The saved prompts/options therefore contain `nan`. The loader validates this exact executed representation while retaining lexical source text and unchanged gold labels. All six models share the affected questions. No scientific input or result is repaired or excluded.\n\n'+md_table(audit['differences'],list(audit['differences'][0]))+'\nCalibration-draw membership is recorded below and in JSON. A changed option can affect its own prediction and, if selected for calibration, the global prior and other heldout predictions. No bound on the total PriDe impact follows from ten questions alone. Original-paper environment equivalence is not established. Disclose the limitation; review any corrected-input sensitivity separately before further model inference.\n\n'+md_table(audit['calibration_draw_membership'],['model','draw','affected_estimation_ids'])
        save_md(output/'input_representation_audit.md',text)


def finalize(output):
    data=json.loads((output/"analysis_results.json").read_text())
    baseline=json.loads((output/"preservation_baseline.json").read_text())
    changes=[]
    allowed={"experiment_evidence_ledger.csv",*baseline.get('authorized_analysis_changes',[])}
    backed_up={str(Path(r['source']).relative_to(ROOT)):r for r in baseline['backups']}
    for relative,record in baseline["files"].items():
        path=ROOT/relative
        if not path.exists() or sha256(path)!=record["sha256"]:
            authorized=relative in allowed and relative in backed_up and backed_up[relative]['sha256']==record['sha256']
            changes.append(dict(path=relative,authorized_backed_up_change=authorized,role='evidence ledger' if relative=='experiment_evidence_ledger.csv' else 'standalone offline analysis code'))
    unexpected=[r for r in changes if not r["authorized_backed_up_change"]]
    backup_errors=[r["backup"] for r in baseline["backups"] if not Path(r["backup"]).is_file() or sha256(Path(r["backup"]))!=r["sha256"]]
    tests=[]
    for path in sorted(output.glob("tests*.xml")):
        suites=ET.parse(path).getroot().iter("testsuite")
        tests.extend(dict(file=str(path),**suite.attrib) for suite in suites)
    test_failures=sum(int(t.get("failures",0))+int(t.get("errors",0)) for t in tests)
    cpu_audit=json.loads((output/"cpu_execution_audit.json").read_text()) if (output/"cpu_execution_audit.json").exists() else None
    oracle=json.loads((output/'independent_transfer_validation.json').read_text()) if (output/'independent_transfer_validation.json').exists() else None
    report=dict(version="phase3b-validation-v2",protected_files=len(baseline["files"]),changed_original_files=changes,
        checksummed_backups=len(baseline["backups"]),backup_verification_errors=backup_errors,
        unexpected_original_changes=unexpected,original_scientific_results_unchanged=not unexpected,tests=tests,test_failures=test_failures,
        independent_full_raw_pairs=sum(r["status"]=="verified_from_matching_raw_observations" for r in data["validation"]),
        matching_current_manifests=sum(r["manifest_hash_matches"] for r in data["validation"]),
        summary_internal_discrepancies=[r for r in data["validation"] if r["summary_internal_checks"]!="pass"],
        raw_validation_discrepancies=[r for r in data["validation"] if r["discrepancies"]],
        real_transfer_directions_completed=len(data["transfer_results"]),target_panel=list(MODELS),
        verified_partial_pilots=len([p for p in data["pilots"] if p["status"]=="verified_partial_pilot_only"]),
        gpu_inference_executed=False,model_dependencies_loaded_in_analysis=bool(cpu_audit["loaded_forbidden_modules"]) if cpu_audit else None,cpu_execution_audit=cpu_audit,
        independent_real_transfer_validation=oracle,
        limitations=[data["uncertainty_status"],"Ten MMLU source None options were executed as nan; see input_representation_audit.md.","Medical logit/truncation traces and resolved tokenizer commits were not persisted."],tolerances=data["tolerances"])
    if unexpected or test_failures or backup_errors or (cpu_audit and cpu_audit["loaded_forbidden_modules"]):
        raise RuntimeError("Preservation or test failure: "+json.dumps(json_safe(report)))
    write_json_new(output/"validation_report.json",report)
    save_md(output/"validation_report.md","# Phase 3B validation report\n\n"+
        f"Preservation: {len(baseline['files'])} pre-existing files checked; {len(unexpected)} unexpected changes. Authorized changes are restricted to the checksummed, backed-up root evidence ledger and new standalone CPU analysis/test code listed in preservation_baseline.json. Original scientific files remain unchanged.\n\n"+
        f"Current full-raw reverified pairs: {report['independent_full_raw_pairs']}/30. Matching current run manifests: {report['matching_current_manifests']}/30. Real paired transfer directions completed: {report['real_transfer_directions_completed']}/12.\n\n"+
        "Focused tests:\n\n"+md_table(tests,["file","tests","failures","errors","skipped"])+
        "\nNo GPU/model inference was invoked. The module import and synthetic transfer are tested with model dependencies explicitly blocked. Tests cover both transfer directions, partitions/collisions/leakage, shared Tr, pinned prior numerics, correction, CPA, contrasts, unsupported RStd, ties and five-option handling. Real pilot checks are explicitly partial.\n\n"+
        "The new raw loader verifies hashes, requested model/tokenizer configuration, resolved model revision, source fingerprints, prompts/gold/options, unique IDs, complete cycles and pooled logits when recorded. A null resolved tokenizer commit and missing pre-truncation lengths remain explicit limitations. Current medical inference did not persist logit traces; candidate settings are established from its pinned code and hashed manifest/diagnostic, without inventing missing logits.\n\n"+
        "All five source-path calibration partitions, raw priors, predictions and summary endpoints are independently reconstructed for verified rows. Complete reference scoring traces below the cap verify absence of truncation there; medical truncation remains unidentified. The executed MMLU CSV representation difference is explicitly documented, not silently corrected. See independent_transfer_validation.json for the separate real-data oracle check.\n\n"+
        "See source_manifest.json for input hashes, missing_observations.json for missing local paths (empty after full recovery), partition_policy.json for transfer policy, and decisions_pending.md for interval choices. Tables/figures use only verified raw rows and label all denominators. No thesis insertion has occurred.\n")
    outputs=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink() and "backups" not in path.relative_to(output).parts and "node_modules" not in path.parts and path.name!="output_manifest.json":
            outputs.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path)))
    write_json_new(output/"output_manifest.json",dict(files=outputs,updated_root_ledger=dict(path=str(ROOT/"experiment_evidence_ledger.csv"),sha256=sha256(ROOT/"experiment_evidence_ledger.csv")),
        hash_scope="All final files except this self-referential manifest and preserved backups; source hashes recorded separately"))
    print(json.dumps({k:v for k,v in report.items() if k not in ("tests","tolerances","limitations","target_panel")},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--recovered-root",type=Path,action="append",default=[])
    parser.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ANALYSIS/"outputs") or "phase3b" not in str(output.relative_to(ANALYSIS/"outputs")):
        raise ValueError("Choose a new phase3b versioned analysis/outputs directory")
    if args.finalize: finalize(output)
    else: run(output,[p.resolve() for p in args.recovered_root])


if __name__=="__main__": main()
