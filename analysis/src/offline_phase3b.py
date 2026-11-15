"""CPU-only, independent validation and paired medical transfer from frozen scores.

No inference code, torch, transformers, model downloads, or project inference
helpers are imported. Numerical reconstruction is checked against the pinned
Zheng oracle in tests, rather than calling the production evaluation wrapper.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EPSILON = 1e-10
RAW_PROTOCOL = "original-open-clm-raw-prompt-1536"
IMPLEMENTATION = "zheng-pride-0shot-v1"
TRANSFER_VERSION = "phase3b-medical-transfer-v1"
UPSTREAM_COMMIT = "2ae2f40c77006f7a00a4675bdfb30151c2406691"
DEBIAS_SHA256 = "f58d4d14074e13cf9f7dfc0a094c8cc6bb6ef578c41918a23e1bdfb456fc4151"
PROBABILITY_ATOL = 1e-3  # Same admissibility bound as executed raw wrapper.
LOGIT_RECONSTRUCTION_ATOL = 2e-6
METRIC_ATOL = 1e-9  # Accuracy/recall/RStd in percent/percentage points.
PRIOR_ATOL = 1e-12
COUNTS = dict(mmlu=13592, arc=1165, csqa=1216, medqa=1273, medmcqa=3941)
MODELS = ("llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-2-9b-it",
          "mistral-7b-instruct-v0.3", "qwen3-8b", "glm-4-9b-chat-hf")
MEDICAL_ARTIFACTS = dict(medqa="20260929T135658Z_test_mcq-preprocessing-v1",
                         medmcqa="20260929T135703Z_validation_mcq-preprocessing-v1")
PARTITION_POLICY = dict(version=TRANSFER_VERSION, alpha=0.05, rounding="floor",
                        minimum=1, seed=42, order="sorted_namespaced_canonical_id",
                        shared_dataset_partition_across_models=True,
                        tie_policy="first_argmax", epsilon=EPSILON,
                        uncertainty="point_estimates_only; CI policy awaits approval")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def cyclic_orders(k: int) -> tuple[tuple[int, ...], ...]:
    if k not in (4, 5):
        raise ValueError("Four or five labels required")
    return tuple(tuple((j + shift) % k for j in range(k)) for shift in range(k))


def select_cycles(probabilities: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape not in ((4, 4), (24, 4), (5, 5)):
        raise ValueError(f"Invalid probability matrix shape {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Non-finite or negative probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=PROBABILITY_ATOL, rtol=0):
        raise ValueError("Probability rows do not sum to one")
    if p.shape == (24, 4):
        permutations = list(itertools.permutations(range(4)))
        indices = tuple(permutations.index(order) for order in cyclic_orders(4))
    else:
        indices = tuple(range(p.shape[0]))
    return p[list(indices)], indices


def stable_softmax(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.exp(values - values.max(axis=-1, keepdims=True))
    return result / (result.sum(axis=-1, keepdims=True) + EPSILON)


def decompose(probabilities: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently reproduce pinned simple(): normalized cycles, CPA, prior."""
    p, _ = select_cycles(probabilities)
    p = p / (p.sum(axis=1, keepdims=True) + EPSILON)
    k = p.shape[1]
    content = np.empty_like(p)
    for shift, order in enumerate(cyclic_orders(k)):
        for displayed, permanent in enumerate(order):
            content[shift, permanent] = p[shift, displayed]
    return p, content.mean(axis=0), stable_softmax(np.log(p + EPSILON).mean(axis=0))


def correction_scores(original: Any, prior: Any) -> np.ndarray:
    p, q = np.asarray(original, dtype=np.float64), np.asarray(prior, dtype=np.float64)
    if p.shape[-1] != q.shape[-1] or q.ndim != 1:
        raise ValueError("Prior/candidate dimensions differ")
    for array in (p, q):
        if not np.isfinite(array).all() or (array < 0).any() or not np.allclose(array.sum(axis=-1), 1, atol=PROBABILITY_ATOL, rtol=0):
            raise ValueError("Invalid prior or original probabilities")
    return np.log(p + EPSILON) - np.log(q + EPSILON)


def correct_probabilities(original: Any, prior: Any) -> np.ndarray:
    return stable_softmax(correction_scores(original, prior))


def label_metrics(gold: Any, prediction: Any, k: int) -> dict:
    g, p = np.asarray(gold, dtype=int), np.asarray(prediction, dtype=int)
    if g.ndim != 1 or p.shape != g.shape or not len(g):
        raise ValueError("Nonempty, aligned gold/prediction vectors required")
    if ((g < 0) | (g >= k) | (p < 0) | (p >= k)).any():
        raise ValueError("Gold/prediction outside label range")
    supports = [int(np.sum(g == j)) for j in range(k)]
    recalls = [float(100 * np.mean(p[g == j] == j)) if supports[j] else float("nan") for j in range(k)]
    return dict(n=len(g), accuracy=float(100 * np.mean(g == p)), recalls=recalls,
                supports=supports, rstd=float(np.std(recalls, ddof=0)))


@dataclass
class Scores:
    model: str
    dataset: str
    ids: tuple[str, ...]
    legacy_ids: tuple[str, ...]
    gold: np.ndarray
    probabilities: np.ndarray  # N x K x K, cyclic display-label probabilities.
    signature: dict
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.ids)
        if not n or len(set(self.ids)) != n or len(set(self.legacy_ids)) != n or len(self.legacy_ids) != n:
            raise ValueError("Empty/duplicate/misaligned IDs")
        if any(not i.startswith(self.dataset + ":") for i in self.ids):
            raise ValueError("Canonical IDs must be benchmark-namespaced")
        if self.gold.shape != (n,) or self.probabilities.shape[0] != n:
            raise ValueError("Population shape mismatch")
        for p in self.probabilities:
            select_cycles(p)
        k = self.probabilities.shape[-1]
        if self.probabilities.shape[1:] != (k, k):
            raise ValueError("Scores must contain selected complete cycles")
        if ((self.gold < 0) | (self.gold >= k)).any():
            raise ValueError("Invalid gold label")


def decomposition_arrays(scores: Scores) -> tuple[np.ndarray, np.ndarray]:
    scores.validate()
    values = [decompose(p) for p in scores.probabilities]
    return np.array([v[1] for v in values]), np.array([v[2] for v in values])


def reproduce(scores: Scores, source_path: str, alpha: float = .05, repeats: int = 5) -> dict:
    scores.validate()
    n, k = len(scores.ids), scores.probabilities.shape[-1]
    estimation_n = int(n * alpha)
    if not 0 < alpha <= 1 or not estimation_n:
        raise ValueError("Nonempty floor(alpha*N) estimation set required")
    cpa, priors = decomposition_arrays(scores)
    original = scores.probabilities[:, 0]
    indices = list(range(n))
    rng, trials = random.Random(source_path.encode("utf-8")), []
    for repeat in range(1 if alpha == 1 else repeats):
        rng.shuffle(indices)
        estimation, remaining = indices[:estimation_n], indices[estimation_n:]
        prior = priors[estimation].mean(axis=0)
        estimates = cpa[estimation].argmax(axis=1)
        heldout_scores=correction_scores(original[remaining],prior) if remaining else np.empty((0,k))
        heldout = heldout_scores.argmax(axis=1) if remaining else []
        trial = dict(repeat=repeat, estimation_ids=[scores.legacy_ids[i] for i in estimation], global_prior=prior.tolist(),
                     full=label_metrics(scores.gold[indices], np.concatenate([estimates, heldout]).astype(int), k),
                     theoretical_cost_factor=float((k*len(estimation)+len(remaining))/n))
        de_ties=int(np.sum((cpa[estimation]==cpa[estimation].max(axis=1,keepdims=True)).sum(axis=1)>1))
        dr_ties=int(np.sum((heldout_scores==heldout_scores.max(axis=1,keepdims=True)).sum(axis=1)>1)) if remaining else 0
        trial["exact_ties"]=dict(cpa_estimation=de_ties,pride_heldout=dr_ties,hybrid=de_ties+dr_ties)
        if remaining:
            trial.update(heldout_default=label_metrics(scores.gold[remaining], original[remaining].argmax(axis=1), k),
                         heldout_pride=label_metrics(scores.gold[remaining], heldout, k))
        trials.append(trial)
    pride = dict(n=n, estimation_n=estimation_n, remaining_n=n-estimation_n, repeats=len(trials))
    for key in ("accuracy", "rstd"):
        vals = [trial["full"][key] for trial in trials]
        pride.update({key:float(np.mean(vals)), key+"_std":float(np.std(vals, ddof=0)),
                      key+"_min":float(np.min(vals)), key+"_max":float(np.max(vals))})
    return dict(implementation=IMPLEMENTATION, upstream_commit=UPSTREAM_COMMIT,
                source_path_seed=source_path, alpha=alpha, n_labels=k,
                default=label_metrics(scores.gold, original.argmax(axis=1), k),
                cyclic_perm=label_metrics(scores.gold, cpa.argmax(axis=1), k), pride=pride, trials=trials)


def permutation_metrics(scores: Scores) -> dict:
    scores.validate()
    n, k = len(scores.ids), scores.probabilities.shape[-1]
    predictions = scores.probabilities.argmax(axis=2)
    shifts = np.arange(k)[None, :]
    displayed_gold = (scores.gold[:, None] - shifts) % k
    permanent_prediction = (predictions + shifts) % k
    pooled = label_metrics(displayed_gold.ravel(), predictions.ravel(), k)
    accuracies = 100 * (predictions == displayed_gold).mean(axis=0)
    cpa, _ = decomposition_arrays(scores)
    ties = (scores.probabilities == scores.probabilities.max(axis=2, keepdims=True)).sum(axis=2) > 1
    return dict(n_questions=n, cyclic_recalls=pooled["recalls"], cyclic_supports=pooled["supports"], cyclic_rstd=pooled["rstd"],
                accuracy_by_shift=accuracies.tolist(), mean_permutation_accuracy=float(accuracies.mean()),
                permutation_accuracy_sd=float(accuracies.std(ddof=0)), permutation_accuracy_range=float(np.ptp(accuracies)),
                semantic_consistency=float(np.mean(np.all(permanent_prediction == permanent_prediction[:, :1], axis=1))),
                exact_ties_original=int(ties[:, 0].sum()), exact_ties_cyclic=int(ties.sum()),
                exact_ties_cpa=int(np.sum((cpa == cpa.max(axis=1, keepdims=True)).sum(axis=1) > 1)))


def compare_saved(expected: Any, actual: Any, path: str = "", atol: float = METRIC_ATOL) -> list[dict]:
    """Compare all saved keys, allowing explicitly added diagnostic keys."""
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [dict(path=path, expected=expected, actual=actual)]
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                differences.append(dict(path=child, issue="missing_key"))
            else:
                differences += compare_saved(value, actual[key], child, PRIOR_ATOL if key == "global_prior" else atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [dict(path=path, issue="list_length", expected=len(expected), actual=len(actual) if isinstance(actual,list) else None)]
        for i, (a, b) in enumerate(zip(expected, actual)):
            differences += compare_saved(a, b, f"{path}[{i}]", atol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(expected, actual, atol=atol, rtol=0, equal_nan=True):
            differences.append(dict(path=path, expected=expected, actual=actual, atol=atol))
    elif expected != actual:
        differences.append(dict(path=path, expected=expected, actual=actual))
    return differences


def partition(ids: tuple[str, ...], dataset: str, alpha: float = .05, seed: int = 42) -> dict:
    if not 0 < alpha < 1 or len(set(ids)) != len(ids) or len(ids) < 2 or any(not i.startswith(dataset+":") for i in ids):
        raise ValueError("Valid namespaced unique population and alpha required")
    population = sorted(ids)
    count = max(1, int(len(population)*alpha))
    estimation_order = random.Random(seed).sample(population, count)
    estimation = sorted(estimation_order)
    heldout = sorted(set(population)-set(estimation))
    return dict(dataset=dataset, alpha=alpha, seed=seed, n=len(population), estimation=estimation,
                estimation_order=estimation_order, heldout=heldout,
                population_ids_sha256=hashlib.sha256(json.dumps(population,separators=(",", ":")).encode()).hexdigest())


def fit_prior(probabilities: np.ndarray, ids: tuple[str, ...], estimation_ids: list[str]) -> np.ndarray:
    """This API deliberately accepts no gold values or outcomes."""
    if len(set(ids)) != len(ids) or not estimation_ids or len(set(estimation_ids)) != len(estimation_ids):
        raise ValueError("Unique population and estimation IDs required")
    lookup = dict(zip(ids, probabilities))
    if not set(estimation_ids) <= lookup.keys():
        raise ValueError("Estimation IDs outside the fitting population")
    return np.mean([decompose(lookup[i])[2] for i in estimation_ids], axis=0)


def transfer_pair(source: Scores, target: Scores, alpha: float = .05, seed: int = 42) -> dict:
    source.validate(); target.validate()
    if source.dataset not in MEDICAL_ARTIFACTS or target.dataset not in MEDICAL_ARTIFACTS or source.dataset == target.dataset:
        raise ValueError("Two distinct medical benchmarks required")
    if source.model != target.model or source.signature != target.signature or source.signature.get("protocol") != RAW_PROTOCOL:
        raise ValueError("Incompatible models or raw scoring signatures")
    if source.probabilities.shape[-1] != 4 or target.probabilities.shape[-1] != 4:
        raise ValueError("Medical transfer requires four choices")
    source_partition = partition(source.ids, source.dataset, alpha, seed)
    target_partition = partition(target.ids, target.dataset, alpha, seed)
    source_prior = fit_prior(source.probabilities, source.ids, source_partition["estimation"])
    target_prior = fit_prior(target.probabilities, target.ids, target_partition["estimation"])
    lookup = {identifier:i for i, identifier in enumerate(target.ids)}
    ids = target_partition["heldout"]
    index = [lookup[identifier] for identifier in ids]
    baseline = target.probabilities[index, 0]
    cpa, _ = decomposition_arrays(target)
    # Match executed argmax(log(p+eps)-log(prior+eps)) exactly. A second
    # softmax is unnecessary and can create extra ties through rounding.
    method_probs = dict(default=baseline, in_domain=correction_scores(baseline,target_prior),
                        transfer=correction_scores(baseline,source_prior), cpa=cpa[index])
    gold = target.gold[index]
    predictions = {method:values.argmax(axis=1) for method,values in method_probs.items()}
    metrics = {method:label_metrics(gold,values,4) for method,values in predictions.items()}
    contrasts = {}
    for left,right in [("in_domain","default"),("transfer","default"),("transfer","in_domain")]:
        contrasts[left+"_minus_"+right] = {metric:metrics[left][metric]-metrics[right][metric] for metric in ("accuracy","rstd")}
    rows = [dict(sample_id=identifier,gold_index=int(gold[j]),
                 **{method+"_prediction":int(values[j]) for method,values in predictions.items()}) for j,identifier in enumerate(ids)]
    ties = {method:int(np.sum((values == values.max(axis=1,keepdims=True)).sum(axis=1)>1)) for method,values in method_probs.items()}
    return dict(version=TRANSFER_VERSION,model=source.model,source=source.dataset,target=target.dataset,
                policy={**PARTITION_POLICY,"alpha":alpha,"seed":seed},signature=source.signature,
                source_partition=source_partition,target_partition=target_partition,
                source_prior=source_prior.tolist(),target_prior=target_prior.tolist(),metrics=metrics,
                contrasts=contrasts,exact_ties=ties,paired_predictions=rows,
                provenance=dict(source=source.provenance,target=target.provenance))


def expected_population(analysis: Path, dataset: str) -> tuple[dict, dict]:
    """Validate source files and provide legacy-row to canonical-item mapping."""
    if dataset in MEDICAL_ARTIFACTS:
        import pandas as pd  # Parquet IO only, not a model dependency.
        directory=analysis/"outputs/preprocessed"/dataset/MEDICAL_ARTIFACTS[dataset]
        manifest_path=directory/"manifest.json"
        manifest=json.loads(manifest_path.read_text())
        for name,record in manifest["artifact_files"].items():
            if sha256(directory/name) != record["sha256"]:
                raise ValueError("Preprocessed file hash mismatch: "+str(directory/name))
        frame=pd.read_parquet(directory/"samples.parquet")
        if len(frame) != COUNTS[dataset] or frame.sample_id.duplicated().any():
            raise ValueError("Medical population count or IDs differ")
        rows={}
        for index,row in enumerate(frame.to_dict("records")):
            rows[f"{dataset}:{index}"]=dict(id=dataset+":"+str(row["sample_id"]),gold=int(row["gold_index"]),
                options=[row["option_"+label] for label in "ABCD"],question=row["question"],retain=True)
        return rows,dict(manifest=str(manifest_path),manifest_sha256=sha256(manifest_path),
                         population_sha256=manifest["selected_sample_set_sha256"],retained_n=len(frame))
    reference=analysis/"data/reference/zheng2024"
    manifest=json.loads((reference/"manifest.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Reference repository revision differs")
    entries=[r for r in manifest["datasets"] if r["dataset"]==dataset]
    digest=hashlib.sha256()
    for entry in entries:
        for record in entry["source_files"]:
            if sha256(reference/record["path"]) != record["sha256"]:
                raise ValueError("Original reference CSV hash differs")
    for path in sorted((reference/f"upstream/code/data_{dataset}").glob("*/*.csv")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    entry=next(r for r in entries if r["repository_split"]=="test")
    path=reference/entry["normalized_path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError("Normalized reference hash differs")
    rows={}
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            index=int(row["sample_id"].rsplit(":",1)[1])
            rows[f'{row["subject"]}:{index}']=dict(id=dataset+":"+row["sample_id"],gold=row["gold_index"],options=row["options"],
                question=row["question"],retain=not row["excluded_by_upstream_filter"])
    # Validate the text actually produced by the pinned CSV reader as well as
    # the lexical source. pandas' default NA parsing turns literal "None" into
    # float NaN; upstream then stringifies option values to "nan". Preserve this
    # executed representation without changing the source, cache or gold label.
    import pandas as pd
    representation_differences=[]
    for source_path in sorted((reference/f"upstream/code/data_{dataset}/test").glob("*_test.csv")):
        subject=source_path.name.removesuffix("_test.csv")
        labels="ABCDE" if dataset=="csqa" else "ABCD"
        frame=pd.read_csv(source_path,names=["Question",*labels,"Answer"],dtype=str)
        for index,parsed in frame.iterrows():
            identifier=f"{subject}:{index}"
            item=rows[identifier]
            if "ABCDE".index(parsed["Answer"])!=item["gold"]:
                raise ValueError("CSV reader and canonical source gold differ")
            executed_options=[str(parsed[label]) for label in labels]
            executed_question=parsed["Question"]
            if not isinstance(executed_question,str):
                raise ValueError("Executed question text is not a string")
            item.update(executed_options=executed_options,executed_question=executed_question)
            differences=[dict(field="option_"+label,source_text=old,executed_text=new)
                         for label,old,new in zip(labels,item["options"],executed_options) if old!=new]
            if executed_question!=item["question"]:
                differences.append(dict(field="question",source_text=item["question"],executed_text=executed_question))
            if differences:
                representation_differences.append(dict(legacy_id=identifier,retained=item["retain"],differences=differences))
    if sum(r["retain"] for r in rows.values()) != COUNTS[dataset]:
        raise ValueError("Reference retained population differs")
    return rows,dict(manifest=str(reference/"manifest.json"),manifest_sha256=sha256(reference/"manifest.json"),
                     normalized_sha256=entry["sha256"],data_sha256=digest.hexdigest(),retained_n=COUNTS[dataset],
                     csv_reader=dict(library="pandas",validation_version=pd.__version__,dtype="str",keep_default_na=True),
                     input_representation_differences=representation_differences)


def raw_prompt(question: str, options: list[str], dataset: str, subject: str) -> str:
    prefix="The following are multiple choice questions"
    prefix+=(" about "+subject.replace("_"," ")) if dataset=="mmlu" else ""
    prefix+=". You should directly answer the question by choosing the correct option."
    choices="\n".join(f"{label}. {option}".strip() for label,option in zip("ABCDE",options))
    return prefix+"\n\nQuestion: "+question.strip()+"\nOptions:\n"+choices+"\nAnswer:"


def load_raw(analysis: Path, manifest_path: Path, summary: dict | None = None,
             *, allow_pilot: bool = False, records_path: Path | None = None) -> tuple[Scores, dict]:
    """Strict raw loader. Pilot opt-in never promotes partial coverage to full."""
    manifest_path=Path(manifest_path)
    record_path=Path(records_path) if records_path is not None else manifest_path.with_name("records.jsonl")
    if summary:
        if sha256(manifest_path)!=summary["observation_manifest_sha256"] or sha256(record_path)!=summary["observations_sha256"]:
            raise ValueError("Current observation/manifest SHA256 mismatch")
    manifest=json.loads(manifest_path.read_text()); config=manifest["config"]
    model,dataset=config["model"],config["dataset"]
    if summary and (model!=summary["model"] or dataset!=summary["dataset"]):
        raise ValueError("Summary model/dataset identifiers differ")
    if manifest.get("status")!="completed" or (config.get("limit") is not None and not allow_pilot):
        raise ValueError("Not a complete unrestricted run")
    if config.get("protocol")!=RAW_PROTOCOL or config.get("scorer_backend")!="real_transformers" or config.get("scoring_adapter_version")!="last-token-fp32-v1":
        raise ValueError("Incompatible raw scoring protocol")
    import yaml
    current=yaml.safe_load((analysis/"configs/models"/(model+".yaml")).read_text())
    if config["model_config"]!=current:
        raise ValueError("Recorded model/tokenizer configuration differs")
    runtime=manifest.get("runtime",{})
    if runtime.get("model_revision") != current["model_revision"]:
        raise ValueError("Resolved model revision missing or different")
    if runtime.get("tokenizer_revision") not in (None,current["tokenizer_revision"]):
        raise ValueError("Resolved tokenizer revision differs")
    population,source=expected_population(analysis,dataset)
    if dataset in MEDICAL_ARTIFACTS:
        if config.get("preprocessing_manifest_sha256")!=source["manifest_sha256"] or config.get("implementation")!=IMPLEMENTATION:
            raise ValueError("Medical population/implementation fingerprint differs")
    elif config.get("data_sha256")!=source["data_sha256"]:
        raise ValueError("Original benchmark fingerprint differs")
    if dataset not in MEDICAL_ARTIFACTS:
        upstream=analysis/"data/reference/zheng2024/upstream/code"
        for module in ("utils","mmlu_categories","eval_clm_utils"):
            expected=config.get("source_hashes",{}).get(module)
            if expected is None or sha256(upstream/(module+".py"))!=expected:
                raise ValueError("Pinned scoring source fingerprint differs")
    k=5 if dataset=="csqa" else 4
    diagnostic=manifest.get("diagnostic",{})
    slots=diagnostic.get("slot_ids",[])
    if (len(slots)!=2*k or diagnostic.get("labels")!=list("ABCDE"[:k])
            or not all(isinstance(slot,int) and slot>=0 for slot in slots)
            or diagnostic.get("whitespace_variants_pooled") is not True
            or diagnostic.get("duplicate_slots_preserved") is not True):
        raise ValueError("Candidate slot diagnostic missing or wrong")
    setting="perm" if "perm" in config["settings"] else "cyclic"
    selected={}; seen=set(); excluded_n=0; logits_checked=0; traces_missing=0; at_cap=0; trace_tokens=[]
    probability_sum_max_error=0.0;permutation_shapes={}
    with record_path.open() as stream:
        for number,line in enumerate(stream,1):
            record=json.loads(line)
            if record.get("shots")!=0 or record.get("setting")!=setting:
                continue
            data=record["data"]; legacy=f'{record["subject"]}:{int(data["idx"])}'
            if legacy in seen or legacy not in population:
                raise ValueError(f"Duplicate/unknown question ID at record {number}: {legacy}")
            seen.add(legacy); item=population[legacy]
            executed_options=item.get("executed_options",item["options"])
            executed_question=item.get("executed_question",item["question"])
            if data["options"]!=executed_options or "ABCDE".index(data["ideal"])!=item["gold"]:
                raise ValueError("Options or gold-content mapping differs: "+legacy)
            expected_prompt=raw_prompt(executed_question,executed_options,dataset,record["subject"])
            if data.get("prompt")!=expected_prompt+(" " if diagnostic.get("prompt_adds_space") else ""):
                raise ValueError("Stored original-order prompt differs: "+legacy)
            cycles,indices=select_cycles(data["probs"])
            raw=np.asarray(data["probs"],dtype=np.float64)
            probability_sum_max_error=max(probability_sum_max_error,float(np.max(np.abs(raw.sum(axis=1)-1))))
            shape="x".join(str(n) for n in raw.shape)
            permutation_shapes[shape]=permutation_shapes.get(shape,0)+1
            if cycles.shape[1]!=k or (np.asarray(data["probs"]).shape[0]==24 and setting!="perm"):
                raise ValueError("Permutation shape/setting differs")
            trace=record.get("trace")
            if trace is None:
                traces_missing+=1
            else:
                if len(trace)!=len(data["probs"]):
                    raise ValueError("Scoring trace/permutation length differs")
                for i,step in enumerate(trace):
                    logit=np.asarray(step.get("option_slot_logits"),dtype=np.float32)
                    if logit.shape!=(2*k,) or not np.isfinite(logit).all():
                        raise ValueError("Invalid candidate-slot logits")
                    exp=np.exp(logit-logit.max()); p=(exp/exp.sum()).reshape(2,k).sum(axis=0)
                    if not np.allclose(p,data["probs"][i],atol=LOGIT_RECONSTRUCTION_ATOL,rtol=0):
                        raise ValueError("Saved probabilities do not match pooled candidate logits")
                    tokens=step["input_tokens"]
                    if not 0<tokens<=1536:
                        raise ValueError("Context token count outside raw protocol")
                    trace_tokens.append(tokens); at_cap+=tokens==1536; logits_checked+=1
            if not item["retain"]:
                excluded_n+=1; continue
            selected[legacy]=(item,cycles)
    expected_ids={identifier for identifier,item in population.items() if item["retain"]}
    full=set(selected)==expected_ids
    if not full and not allow_pilot:
        raise ValueError(f"Incomplete retained population: {len(selected)}/{len(expected_ids)}")
    order=sorted(selected,key=lambda identifier:(identifier.rsplit(":",1)[0],int(identifier.rsplit(":",1)[1])))
    signature=dict(model_repo=current["model_repo"],model_revision=current["model_revision"],
        tokenizer_repo=current["tokenizer_repo"],tokenizer_revision_requested=current["tokenizer_revision"],
        tokenizer_revision_resolved=runtime.get("tokenizer_revision"),protocol=RAW_PROTOCOL,
        adapter="last-token-fp32-v1",slot_ids=slots,dtype=runtime.get("dtype"),n_labels=k,
        scoring_library_versions=runtime.get("versions",{}))
    provenance=dict(manifest=str(manifest_path),manifest_sha256=sha256(manifest_path),observations=str(record_path),
                    observations_sha256=sha256(record_path),source=source,full_population=full)
    scores=Scores(model,dataset,tuple(selected[i][0]["id"] for i in order),tuple(order),
                  np.array([selected[i][0]["gold"] for i in order]),np.array([selected[i][1] for i in order]),signature,provenance)
    scores.validate()
    truncation_status="verified_none" if trace_tokens and not traces_missing and at_cap==0 else "not_identifiable"
    diagnostics=dict(retained_n=len(order),excluded_n=excluded_n,full_population=full,
        logits_reconstructed=logits_checked,records_without_logits=traces_missing,
        probability_sum_max_abs_error=probability_sum_max_error,permutation_matrix_shapes=permutation_shapes,
        scoring_tokens_min=min(trace_tokens) if trace_tokens else None,scoring_tokens_max=max(trace_tokens) if trace_tokens else None,
        scorings_at_context_cap=int(at_cap) if trace_tokens else None,truncation_status=truncation_status,
        truncation_incidence="none: every recorded scored length is below the cap" if truncation_status=="verified_none" else "unknown: pre-truncation lengths not recorded",
        source_input_representation_differences=len(source.get("input_representation_differences",[])),
        tokenizer_resolution="resolved" if runtime.get("tokenizer_revision") else "requested pin recorded; resolved commit unavailable")
    return scores,diagnostics
