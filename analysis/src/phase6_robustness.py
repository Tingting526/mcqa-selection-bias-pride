"""Offline robustness estimands; no inference dependencies or mutable inputs."""
from __future__ import annotations
import hashlib
import json
import random
import numpy as np
from src.offline_phase3b import EPSILON, correction_scores, label_metrics

VERSION = "phase6-robustness-v1"
SEEDS = (42, 43, 44, 45, 46)
CORE_MEDICAL = frozenset(("anatomy", "clinical_knowledge", "college_medicine",
                         "medical_genetics", "professional_medicine"))
RELATED_MEDICAL = frozenset(("human_aging", "nutrition", "virology"))
SCALARS = ("original_accuracy_percent", "original_rstd_pp", "cyclic_rstd_pp",
           "mean_permutation_accuracy_percent", "permutation_accuracy_sd_pp",
           "permutation_accuracy_range_pp", "semantic_consistency_percent")

def ids_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()

def validate_partition(population, fitting, evaluation, *, repeated=False):
    if len(set(population)) != len(population) or not fitting or not evaluation:
        raise ValueError("Nonempty unique population and fitting/evaluation required")
    if len(set(evaluation)) != len(evaluation) or (not repeated and len(set(fitting)) != len(fitting)):
        raise ValueError("Unexpected duplicate IDs")
    if not (set(fitting) | set(evaluation)) <= set(population):
        raise ValueError("Unknown partition ID")
    if set(fitting) & set(evaluation):
        raise ValueError("Calibration/evaluation leakage")

def nested_budget(part, budget=63):
    order = part["estimation_order"]
    if len(order) != len(set(order)) or set(order) != set(part["estimation"]):
        raise ValueError("Original randomized estimation order unavailable or invalid")
    if not 0 < budget <= len(order):
        raise ValueError("Budget outside original pool")
    return order[:budget]

def calibration_draw(part, seed, policy):
    pool = sorted(part["estimation"])
    if policy == "full_pool_without_replacement_control":
        # Sorting the selected set reproduces the primary arithmetic order.
        return sorted(random.Random(seed).sample(pool, len(pool)))
    if policy == "fixed_pool_with_replacement":
        rng = random.Random(seed)
        return [pool[rng.randrange(len(pool))] for _ in pool]
    raise ValueError("Unknown calibration policy")

def average_prior(item_priors, population, fitting, *, repeated=False):
    if not fitting or (not repeated and len(set(fitting)) != len(fitting)):
        raise ValueError("Invalid fitting IDs or multiplicities")
    lookup = {i:j for j,i in enumerate(population)}
    if len(lookup) != len(population) or not set(fitting) <= set(lookup):
        raise ValueError("Unknown or duplicate population IDs")
    return np.asarray(item_priors)[[lookup[i] for i in fitting]].mean(axis=0)

def cyclic_metrics(gold, label_predictions, k):
    gold = np.asarray(gold, dtype=int)
    labels = np.asarray(label_predictions, dtype=int)
    if not len(gold) or labels.shape != (len(gold), k):
        raise ValueError("Complete aligned cyclic predictions required")
    shifts = np.arange(k)[None, :]
    content = (labels + shifts) % k
    displayed_gold = (gold[:, None] - shifts) % k
    original = label_metrics(gold, labels[:, 0], k)
    cyclic = label_metrics(displayed_gold.ravel(), labels.ravel(), k)
    accuracies = 100 * (content == gold[:, None]).mean(axis=0)
    return dict(n=len(gold), gold_supports=original["supports"], original_recalls_percent=original["recalls"],
                cyclic_supports=cyclic["supports"], cyclic_recalls_percent=cyclic["recalls"],
                original_accuracy_percent=original["accuracy"], original_rstd_pp=original["rstd"],
                cyclic_rstd_pp=cyclic["rstd"], accuracy_by_shift_percent=accuracies.tolist(),
                mean_permutation_accuracy_percent=float(accuracies.mean()),
                permutation_accuracy_sd_pp=float(accuracies.std(ddof=0)),
                permutation_accuracy_range_pp=float(np.ptp(accuracies)),
                semantic_consistency_percent=float(100*np.all(content == content[:, :1], axis=1).mean()))

def frozen_predictions(probabilities, prior):
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[1] != p.shape[2] or p.shape[2] not in (4, 5):
        raise ValueError("Four/five complete cycles required")
    corrected = correction_scores(p, prior)
    return p.argmax(axis=2), corrected.argmax(axis=2)

def before_after(gold, default, corrected, k):
    a, b = cyclic_metrics(gold, default, k), cyclic_metrics(gold, corrected, k)
    result = {**{f"default_{key}":v for key,v in a.items()},
              **{f"pride_{key}":v for key,v in b.items()}}
    for key in SCALARS:
        result["delta_"+key.replace("_percent", "_pp")] = b[key]-a[key]
    content_a = (default+np.arange(k)[None, :]) % k
    content_b = (corrected+np.arange(k)[None, :]) % k
    ca = np.all(content_a == content_a[:, :1], axis=1)
    cb = np.all(content_b == content_b[:, :1], axis=1)
    result.update(consistency_gained_n=int((~ca & cb).sum()), consistency_lost_n=int((ca & ~cb).sum()),
                  original_correct_gained_n=int(((default[:,0]!=gold)&(corrected[:,0]==gold)).sum()),
                  original_correct_lost_n=int(((default[:,0]==gold)&(corrected[:,0]!=gold)).sum()))
    return result

def transfer_metrics(probabilities, gold, cpa, source_prior, target_prior):
    original = probabilities[:, 0]
    vectors = dict(default=original, in_domain=correction_scores(original, target_prior),
                   transfer=correction_scores(original, source_prior), cpa=cpa)
    preds = {m:p.argmax(axis=1) for m,p in vectors.items()}
    result = {}
    for m,p in preds.items():
        q=label_metrics(gold,p,4)
        result.update({m+"_accuracy_percent":q["accuracy"],m+"_rstd_pp":q["rstd"]})
    for left,right in (("transfer","in_domain"),("transfer","default"),("in_domain","default")):
        for metric in ("accuracy", "rstd"):
            unit = "percent" if metric=="accuracy" else "pp"
            result[f"{left}_minus_{right}_{metric}_pp"] = result[f"{left}_{metric}_{unit}"]-result[f"{right}_{metric}_{unit}"]
    return result, preds

def seed_statistics(rows):
    if len(rows)!=5 or set(r["seed"] for r in rows)!=set(SEEDS):
        raise ValueError("Exactly five prespecified seeds required")
    keys=[k for k,v in rows[0].items() if (k.endswith("_accuracy_percent") or k.endswith("_rstd_pp") or "_minus_" in k) and isinstance(v,(int,float))]
    result={}
    for key in keys:
        x=np.array([r[key] for r in rows])
        result.update({key+"_seed_mean":float(x.mean()),key+"_seed_sample_sd":float(x.std(ddof=1)),
                       key+"_seed_min":float(x.min()),key+"_seed_max":float(x.max())})
        if "_minus_" in key:
            result[key+"_seed_positive_n"]=int((x>1e-9).sum())
            result[key+"_seed_zero_n"]=int((np.abs(x)<=1e-9).sum())
            result[key+"_seed_negative_n"]=int((x < -1e-9).sum())
    for role in ("source", "target"):
        priors=np.array([r[role+"_prior"] for r in rows])
        result[role+"_prior_label_sample_sd_probability"]=priors.std(axis=0,ddof=1).tolist()
        result[role+"_prior_max_l1_from_primary"]=max(r[role+"_prior_l1_from_primary"] for r in rows)
    return result

def subject_group(subject, scheme):
    medical = CORE_MEDICAL if scheme=="core5" else CORE_MEDICAL|RELATED_MEDICAL if scheme=="broad8" else None
    if medical is None: raise ValueError("Unknown subject grouping")
    return "medical" if subject in medical else "remaining_subjects"
