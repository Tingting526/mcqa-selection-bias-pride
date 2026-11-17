#!/usr/bin/env python3
"""Traceable scientific reports and compact booktabs tables; never edits Paper."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'analysis/outputs/phase6_robustness_20261018'
sys.path.insert(0,str(ROOT/'analysis'))
from src.offline_phase3b import write_json_new
from plot_phase6_robustness import MODELS,SHORT
NAMES=dict(zip(MODELS,SHORT));DS=dict(mmlu='MMLU',arc='ARC-Challenge',csqa='CommonsenseQA',medqa='MedQA',medmcqa='MedMCQA')

def write(name,text):
    with (OUT/name).open('x') as f:f.write(text.rstrip()+'\n')

def fmt(x):return 'NA' if x is None else f'{x:.2f}' if isinstance(x,(float,int)) else str(x)

def main():
    d=json.loads((OUT/'analysis_results.json').read_text());s=json.loads((OUT/'seed_resampling_results.json').read_text())
    assert json.loads((OUT/'independent_validation.json').read_text())['status']=='pass'
    trace=[]
    def table(name,title,headers,rows,note,mappings):
        md='# '+title+'\n\n'+note+'\n\n| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'
        md+='\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in rows)+'\n'
        write('tables/'+name+'.md',md)
        escape=lambda x:str(x).replace('−','-').replace('→',r'$\to$').replace('%',r'\%').replace('_',r'\_').replace('±',r'$\pm$')
        tex='% Phase 6 standalone fragment. Requires booktabs. No active thesis file includes this fragment.\n'
        tex+=r'\begin{table}[htbp]'+'\n'+r'\centering\small'+'\n'+r'\caption{'+escape(title)+r'}'+'\n'+r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n'+r'\toprule'+'\n'
        tex+=' & '.join(escape(x) for x in headers)+r'\\'+'\n'+r'\midrule'+'\n'
        tex+='\n'.join(' & '.join(escape(fmt(x)) for x in row)+r'\\' for row in rows)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\par\vspace{2pt}\begin{minipage}{.98\linewidth}\footnotesize '+escape(note)+r'\end{minipage}'+'\n'+r'\end{table}'+'\n'
        write('tables/'+name+'.tex',tex)
        trace.append(dict(name=name,headers=headers,display_rows=rows,note=note,source_rows=mappings,source_version='phase6-robustness-v1'))
    for source,target in [('medqa','medmcqa'),('medmcqa','medqa')]:
        eq=[next(r for r in d['equal_budget'] if r['model']==m and r['source']==source) for m in MODELS]
        table('equal_budget_'+source,DS[source]+' → '+DS[target]+': equal source budget',
              ['Model','Primary T−I','63-fit T−I','63-fit T−D','63-fit R(T)−R(I)'],
              [[NAMES[r['model']],r['primary_transfer_minus_in_domain_accuracy_pp'],r['transfer_minus_in_domain_accuracy_pp'],r['transfer_minus_default_accuracy_pp'],r['transfer_minus_in_domain_rstd_pp']] for r in eq],
              f"All numeric columns in pp. Target holdout N={eq[0]['evaluation_n']}; target fit n={eq[0]['target_calibration_n']} unchanged. Source fit n=63, nested original randomized order. T=transfer, I=in-domain, D=Default; R=original RStd. The MedQA-source condition is unchanged by construction. Source: phase6-robustness-v1, equal_budget_results.csv.",
              [dict(model=r['model'],source=source,target=target,source_array='analysis_results.json:equal_budget') for r in eq])
        seed=[next(r for r in s['summaries'] if r['model']==m and r['source']==source) for m in MODELS];k='transfer_minus_in_domain_accuracy_pp'
        table('seed_sensitivity_'+source,DS[source]+' → '+DS[target]+': calibration resampling',
              ['Model','Mean','Sample SD','Min','Max','Positive / 5'],
              [[NAMES[r['model']],r[k+'_seed_mean'],r[k+'_seed_sample_sd'],r[k+'_seed_min'],r[k+'_seed_max'],str(r[k+'_seed_positive_n'])] for r in seed],
              f"T−I accuracy in pp, five prespecified seeds42–46. Same primary target holdout N={eq[0]['evaluation_n']}. Source and target priors refitted by sampling with replacement within original pools (MedQA63, MedMCQA197). Sample SD uses ddof1 and is not a confidence interval. Source: phase6-robustness-v1, seed_resampling_results.json; the CSV also contains the degenerate full-pool control.",
              [dict(model=r['model'],source=source,target=target,source_array='seed_resampling_results.json:summaries') for r in seed])
    for ds in ['mmlu','arc','csqa','medqa','medmcqa']:
        rows=[next(r for r in d['corrected_permutation'] if r['model']==m and r['benchmark']==ds) for m in MODELS]
        table('corrected_permutation_'+ds,DS[ds]+': heldout frozen-prior correction',
              ['Model','Δ accuracy','Δ R(orig)','Δ R(cyc)','Δ consistency'],
              [[NAMES[r['model']],r['delta_original_accuracy_pp'],r['delta_original_rstd_pp'],r['delta_cyclic_rstd_pp'],r['delta_semantic_consistency_pp']] for r in rows],
              f"All differences are PriDe−Default in pp on identical holdout N={rows[0]['evaluation_n']}, calibration n={rows[0]['calibration_n']}. First saved path-seeded hybrid draw per model; one prior applied to all {rows[0]['cyclic_shifts']} shifts. R is recall population SD. Holdouts can differ across models; this is not the full-population hybrid endpoint. Source: phase6-robustness-v1, corrected_permutation_robustness.csv.",
              [dict(model=r['model'],benchmark=ds,source_array='analysis_results.json:corrected_permutation') for r in rows])
    for scheme in ['core5','broad8']:
        rows=[];mappings=[]
        for m in MODELS:
            a=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']=='medical');b=next(r for r in d['mmlu_subjects'] if r['model']==m and r['grouping']==scheme and r['group']=='remaining_subjects')
            rows.append([NAMES[m],a['full_default_original_accuracy_percent'],b['full_default_original_accuracy_percent'],a['full_default_cyclic_rstd_pp'],b['full_default_cyclic_rstd_pp'],a['full_default_semantic_consistency_percent']-b['full_default_semantic_consistency_percent']]);mappings.append(dict(model=m,grouping=scheme,source_array='analysis_results.json:mmlu_subjects'))
        table('mmlu_'+scheme,'MMLU medical-subject grouping: '+scheme,
              ['Model','Acc(M)','Acc(R)','Rcyc(M)','Rcyc(R)','C(M)−C(R)'],rows,
              f"Default on full retained population: medical N={a['group_population_n']}, remaining N={b['group_population_n']}. Accuracy and consistency C in percent; Rcyc and C difference in pp. Four cyclic shifts. No fitting required; heldout group PriDe effects, supports and individual subject sizes are in the CSV. Names declared before performance inspection; remaining subjects are not a pure nonmedical control. Source: phase6-robustness-v1, mmlu_medical_subject_analysis.csv.",mappings)
    write_json_new(OUT/'tables/table_source_map.json',dict(tables=trace))
    perm=d['corrected_permutation']
    counts={}
    for key in ['original_accuracy_pp','original_rstd_pp','cyclic_rstd_pp','semantic_consistency_pp','mean_permutation_accuracy_pp','permutation_accuracy_sd_pp','permutation_accuracy_range_pp']:
        vals=[r['delta_'+key] for r in perm];counts[key]=dict(increase=sum(v>1e-9 for v in vals),decrease=sum(v< -1e-9 for v in vals),equal=sum(abs(v)<=1e-9 for v in vals))
    write_json_new(OUT/'descriptive_direction_counts.json',counts)
    write('robustness_summary.md',"""# Phase 6 robustness summary

18 October 2026. All analyses use verified raw-text probability caches, six current models, CPU only. No file under Paper/, historical scientific output, cached score, prior, model configuration or approved Phase 4 bootstrap result was changed. The thesis remains the Phase 5 manuscript.

## Completed coverage

| Analysis | Coverage | Evaluation |
|---|---|---|
| Equal source calibration budget | 6 models × 2 directions = 12 | Original fixed RQ3 holdouts: MedQA1210; MedMCQA3744 |
| Five-seed conditional calibration resampling | 60 model/direction/seed results plus60 no-variation controls | Same fixed RQ3 holdouts; Default, in-domain PriDe, transfer PriDe, CPA |
| Frozen-prior corrected permutation robustness | 6 models × 5 benchmarks = 30 | Disjoint complements of first saved hybrid calibration draws |
| MMLU subject sensitivity | 57 subject rows +4 grouping rows per model =366 | Full Default endpoints and disjoint-heldout group PriDe endpoints |

The models are Llama3.1 8B Instruct, Qwen2.5 7B Instruct, Gemma2 9B IT, Mistral7B Instruct v0.3, Qwen3 8B and GLM4 9B Chat HF. Reference benchmarks remain MMLU, ARC-Challenge and CommonsenseQA; the extension remains MedQA and MedMCQA. RQs and reproduction → medical extension → paired medical transfer structure are preserved.

## Findings

- **Equal source budgets:** MedQA→MedMCQA T−I accuracy remains positive6/6 (the forward condition is unchanged by construction). With MedMCQA source reduced197→nested63 and the MedQA target prior unchanged, reverse T−I is negative6/6, versus five negative and one tie in primary results. Reverse changes in transfer accuracy range−0.33 to+0.25pp. Equal source budgets do not remove target, population, prompt or target-prior differences.
- **Calibration sensitivity:** Across seeds42–46, forward T−I is positive27/30 and reverse negative27/30, with2 positive and1 tie in reverse. All six per-model means retain the primary directional pattern, but Llama and GLM forward, and Qwen3 and GLM reverse, change sign across draws. Thus the pattern is **partially stable**, not uniformly stable.
- **Corrected permutations:** Cyclic RStd decreases26/30; content consistency increases24/30; original RStd decreases25/30. Original accuracy increases20/30, decreases6 and ties4. The consistency and imbalance endpoints disagree in meaningful cases; no universal robustness guarantee follows.
- **MMLU:** The core five medical subjects contain939 questions and have higher Default accuracy and consistency than the remaining12,653 in every model. Broadening the grouping to eight subjects gives1604 questions: accuracy remains higher6/6, but Gemma's consistency becomes lower than the remaining subjects. These are descriptive topic/difficulty comparisons.

## Calibration design clarification

The fixed RQ3 holdout exhausts all questions outside the original63/197 fit pools. Full-size selection without replacement from those pools cannot vary their membership: all60 five-seed controls reproduce the original priors and endpoints exactly. This was explained before implementing the alternative.

The informative extension samples with replacement within those pools, preserving nominal63/197 draw counts and fitting outside the fixed holdouts. It refits both priors, measures conditional calibration-resampling variability, and does **not** represent five independent population partitions. Its seed42 draw differs from the original seed42 fit; the original remains a separate reference. Sample SD across five draws uses ddof1 and is not a CI. No primary bootstrap interval was changed or extended.

The alternative was disclosed and offered as a methodological preference during execution; no author reply had been received at report generation. It is delivered as an explicitly labelled supplementary condition, not retroactive author approval. Review this choice before manuscript integration; any alternative policy would be a separate analysis.

## Delivery and verification

The four required CSVs contain full metrics, units, policy, sample sizes, version, priors and ID hashes;120 seed rows distinguish informative resampling and degenerate control. Exact fitting/evaluation memberships and duplicate multiplicities are in JSON. Four figures are supplied as editable vector SVG and230dpi PNG. Eleven booktabs table fragments and Markdown counterparts are in tables/; none is included in the active manuscript.

Independent upstream-oracle and separate scalar checks verify all30 populations,12 equal-budget pairs, both60-row seed conditions and366 MMLU rows. The focused CPU suite passes53 tests, including19 new synthetic tests. validation_report.json records numerical, export, provenance and preservation checks. No additional GPU inference is needed for these analyses. See scientific_interpretation.md and the separate thesis_robustness_integration_plan.md for recommendations.
""")
    write('scientific_interpretation.md',"""# Scientific interpretation of Phase 6

These analyses supplement the verified primary estimands. They do not replace RQ1/RQ2 full-population hybrid results, the paired RQ3 seed42 point estimates, or the approved fixed-prior question-bootstrap intervals. Model families are descriptive cases, not six independent hypothesis tests.

## 1. Equal source calibration budgets: the directional pattern is not removed

**Alternative explanation.** The primary comparison changes the number of source fit questions (63 MedQA versus197 MedMCQA), so absolute source budget could contribute to the direction pattern. This analysis changes only a MedMCQA source prior from197 to63 questions, using the first63 IDs in its saved randomized selection order. It leaves original target questions, target scores, target-prior fit and estimator unchanged. The MedQA source already has63 questions, so the entire forward condition is unchanged; it supplies the reference under equal *source* budgets rather than a second independent replication.

On MedQA targets (N1210; target fit63), reverse transfer-minus-in-domain accuracy becomes−1.74pp Llama,−0.33 Qwen2.5,−1.24 Gemma,−1.16 Mistral,−0.33 Qwen3 and−0.33 GLM. Transfer accuracy changes relative to the primary197-fit source by+0.08,+0.25,0.00,+0.08,−0.33 and0.00pp, respectively. Qwen3's primary tie becomes a loss. Forward T−I remains+0.03 to+0.72pp with target fit197. Thus a larger MedMCQA source budget is not necessary for the observed reverse disadvantage in this nested-subset comparison. One nested subset cannot establish budget-independent generality or a monotonic budget response.

Recall imbalance also retains its primary direction counts: forward transfer RStd exceeds in-domain RStd5/6; reverse4/6. Equal-budget reverse RStd contrasts range−0.47 to+2.35pp. PriDe can trade accuracy against balance; forward Gemma still has a7.71pp RStd disadvantage despite higher transfer accuracy. Transfer's comparison with Default remains essential: equal-budget Llama reverse is−1.49pp against Default, whereas GLM reverse remains+2.15pp. No cyclic item-robustness endpoint is estimated here: only original-order RQ3 metrics change with the prior.

Equalizing source counts does not equate targets, their gold supports, difficulty, prompt context or target-prior sample counts. It does not isolate a causal domain effect, prove MedQA's prior intrinsically superior, or supply a new significance claim. The original conditional CIs cannot be attached to these newly fitted priors.

## 2. Conditional calibration resampling: partial, model-specific stability

**Design limit and alternative.** Fixed primary target holdouts leave exactly the primary fit pools as their complements. Using all63/197 available questions without replacement yields the same membership for every seed; all60 controls reproduce primary priors and original accuracy/RStd exactly. Zero control SD is a mathematical consequence, not evidence that calibration uncertainty is absent.

After explaining this limitation, a separate condition resamples each original fit pool with replacement at its original nominal count for seeds42,43,44,45,46. IDs, repeated multiplicities and priors are retained. Both source and target priors are refitted, while the target population, raw scores and all four evaluation rules are fixed. Default and CPA consequently remain constant. The draws are shared across models; reverse conditions reuse dataset-specific draws. This couples comparisons deliberately and prevents interpreting60 observations as independent experiments. A nominal63 draws contains36–43 distinct MedQA questions, and197 draws121–126 distinct MedMCQA questions. Those smaller unique counts are disclosed, not presented as full unique budgets.

Across five draws, T−I mean accuracy is positive for every forward model and negative for every reverse model. Forward is positive27/30 model-seed cases; reverse is negative27/30, positive2 and equal1. Forward Qwen2.5, Gemma, Mistral and Qwen3 stay positive5/5; reverse Llama, Qwen2.5, Gemma and Mistral stay negative5/5. The smaller margins are sensitive: forward Llama spans−0.13 to+0.32pp and GLM−0.24 to+0.45; reverse Qwen3 spans−0.41 to+0.08 and GLM−0.58 to+0.08. Qwen3 reverse includes one zero. Per-model T−I sample SD ranges0.11–0.28pp forward and0.16–0.34pp reverse. Exact endpoints are in the seed tables.

The estimated priors do move: the maximum observed label-prior sample SD is0.0352 probability units for MedQA (3.52 percentage points of prior mass) and0.0248 for MedMCQA (2.48pp). Maximum L1 displacement from a primary prior ranges0.038–0.133 across MedQA model priors and0.030–0.076 across MedMCQA priors. These are descriptive distances, not accuracy effect sizes. A small prior shift can alter marginal rankings; the analysis does not identify a causal explanation for a particular changed prediction. Across seeds, T−I RStd is higher25/30 forward and20/30 reverse. T−Default accuracy is positive only16/30 forward and9/30 reverse (one reverse tie), so retaining a relative transfer advantage does not guarantee a net Default improvement. No item-consistency change is inferred from these original-order metrics.

The directional tendency is therefore **partially stable** under this conditional resampling policy, with sign-sensitive small contrasts. Five draws provide a limited sensitivity diagnostic; they are neither a precise estimate of the whole resampling distribution nor five new dataset partitions. The original pool itself remains one selected subset of each population. Both priors vary together, so this does not isolate source-only calibration variability. The new seed42 is not the original seed42 selection. The latter stays untouched as a reference.

Sample SD uses ddof1 and describes variation across the five fitted-prior conditions. The Phase4 intervals resample target questions conditional on one fixed pair of priors. These distinct variability sources must not be exchanged, combined into an unapproved interval, or used to claim that a sign change refutes the original fixed-prior estimate. The alternative was transparently disclosed before execution and is supplementary pending the author's methodological preference for integration.

## 3. Frozen-prior permutation robustness: aggregate balance and content stability diverge

**Question examined.** Does a prior that improves original label recalls also make selected answer content stable across rearrangements? For each of30 model/benchmark pairs, this analysis uses the first existing hybrid calibration draw, estimates one prior only on those questions, and corrects every cyclic score of every remaining question with that frozen prior. It applies correction in displayed-label space before mapping to permanent contents. Four shifts are used except five for CommonsenseQA. Fits, question scores, gold mappings and the heldout comparison set stay fixed between Default and corrected outcomes.

The complement sizes are MMLU12913, ARC1107, CommonsenseQA1156, MedQA1210 and MedMCQA3744; model-specific first draws may select different members. Medical sizes match RQ3 numerically, but their path-seeded memberships generally differ from its seed42 partitions. No fitting question contributes to its own reported corrected consistency. This condition is pure PriDe on heldout questions, whereas primary RQ1/RQ2 combine CPA on fitting questions and PriDe elsewhere and average five full-population endpoints. The new direction counts cannot overwrite those primary counts.

Original RStd decreases25/30, cyclic RStd26/30 and content consistency increases24/30. Original accuracy increases20, decreases6 and ties4; mean permutation accuracy rises25/30. Permutation-accuracy SD decreases17/30 and increases13/30; range decreases17, increases12 and ties1. Thus reducing within-question recall imbalance need not reduce variation in accuracy between shifts either. Different robust-selection endpoints capture different changes.

There are important discrepancies. Mistral MMLU improves accuracy by0.06pp and reduces original RStd by3.60pp and cyclic RStd by0.32pp, but content consistency falls51.85%→50.34% (−1.51pp). Mistral ARC and Qwen3 MedMCQA likewise reduce both RStd measures while losing consistency; the latter loss is one question (−0.027pp) and should not be overstated. Qwen3 ARC improves original accuracy by0.09pp and original RStd1.71→0.85pp, while cyclic RStd worsens0.32→1.42pp and consistency88.80%→87.53%. Qwen2.5 CommonsenseQA slightly reduces original RStd and raises consistency, but cyclic RStd increases0.69→1.65pp. These directly disprove any automatic inference from original RStd reduction to all robustness improvements.

Mistral MedQA is a stronger exception: accuracy rises0.25pp while original RStd increases2.65→5.72pp, cyclic RStd1.96→5.29pp and consistency50.91%→48.10%. Conversely, GLM CommonsenseQA improves consistency55.80%→64.19% (+8.39pp) and original accuracy+3.89pp, with both imbalance measures falling. Gemma MedMCQA has the largest cyclic-RStd reduction (−12.70pp). These are fixed-score point comparisons, not new significance tests. Consistency can credit a consistently wrong answer and must remain paired with correctness outcomes.

The new evidence materially strengthens RQ1/RQ2 measurement: PriDe often improves empirical content stability under the specified heldout/frozen-prior condition, but it does not guarantee invariance. No model was changed. CPA remains invariant under reindexing the same complete deterministic orbit by construction, using permanent-content first-maximum ties; that procedural fact is not evidence of an improved underlying model. CPA accuracy is also not mean per-shift accuracy.

## 4. MMLU subject sensitivity: useful context with classification dependence

The canonical retained MMLU IDs preserve all57 subject names, independently validated against original metadata. Groupings were declared from names before inspecting performance. The core group contains anatomy, clinical_knowledge, college_medicine, medical_genetics and professional_medicine:939 questions, gold supports[198,222,210,309]. The broad sensitivity adds human_aging, nutrition and virology:1604 questions, supports[382,427,360,435]. Complete included/excluded lists, individual subject sizes and every gold support are recorded in mmlu_subject_membership.json and the366-row CSV.

The remaining groups contain12653 and11988 questions respectively, and include related biology and psychology topics. They are complements of a reproducible grouping, not certified purely nonmedical populations. All labels have positive support in every full and heldout subject/group row, so original RStd is defined; the export convention would mark an unsupported endpoint NA rather than impute zero.

Without any prior fitting, Default original accuracy is higher for the medical group in all six models under both definitions. Core accuracy advantages span1.91–8.64pp; core consistency advantages2.52–13.59pp. Broad medical accuracy remains higher0.63–5.90pp, but Gemma's broad consistency is32.61% versus36.56% in the complement (−3.95pp), reversing its core-group advantage. Cyclic RStd is higher for core medical topics in Qwen2.5 and Gemma, and lower in the other four; broad grouping retains that count. There is no uniform medical-group imbalance pattern across models.

For PriDe effects, group subsets are evaluated only within the disjoint first-draw MMLU holdout, using the same benchmark-wide MMLU prior as Analysis3. The prior is not fitted separately on each group and uses calibration topics from both sides. Core-medical original accuracy decreases for Llama, Qwen2.5, Mistral and Qwen3, while increasing for Gemma and GLM; broad grouping changes this to two decreases, one tie and three increases. This sensitivity and the model-dependent imbalance pattern qualify any simple medical/nonmedical narrative. They do not establish that medical content causes better or worse performance. Full Default and heldout corrected populations must not be mixed in a single before/after comparison.

Subject difficulty, topic composition, the MMLU subject prompt cue and the chosen topic boundary all remain alternative explanations. Many individual subjects are small, with correspondingly coarse one-question increments; their rows are descriptive appendix material. This context supports treating MMLU as broad reference evidence in RQ2, not replacing the MedQA/MedMCQA medical extension or converting it into a causal or matched-prompt experiment.

## Integration judgment

The material additions are a concise corrected-permutation result and an RQ3 robustness paragraph explaining equal-source-budget persistence and partial calibration stability. The largest interpretive change is replacing an unmeasured consistency claim with actual measured benefits and exceptions. Most MMLU and seed-level details belong in the appendix. Preserve primary estimates and intervals, acknowledge the conditional resampling policy, and avoid a new chapter, causal domain claims, universal PriDe success or new GPU experiments. The separate integration plan identifies the specific manuscript statements that require later review; no thesis file was edited here.
""")
    write('analysis_protocol.md',"""# Phase 6 executable protocol and data dictionary

Source version: phase6-robustness-v1. Protocol: original-open-clm-raw-prompt-1536; candidate slot adapter last-token-fp32-v1. Same pinned model/tokenizer configurations and populations as verified Phase3B. No torch/transformers/vllm imports or model execution. Existing reference prompt cues, five-choice CommonsenseQA, retained preprocessing and the ten executed MMLU None→nan options remain unchanged. No corrected-input or matched-prompt inference is implied.

## Numerical and identity conventions

The validated loader checks canonical IDs, released gold content, executed question/options/prompt, model and tokenizer revisions, cached file hashes, complete cyclic orbits, probability sums and available logit traces. ARC24-row stores use cyclic rows0,9,16,18; CSQA uses all5 rows/labels. Per-question priors use pinned upstream normalization and geometric mean in label space, then arithmetic averaging across fit questions. Additive epsilon=1e-10; first exact maximum at every decision. No near-tie tolerance or extra softmax before prediction. Zero probabilities are admissible; unsupported original gold labels yield NA RStd, never zero or omitted labels. Cyclic gold support equals heldout question count at every displayed label.

All IDs are benchmark namespaced. ID SHA256 hashes serialize the ordered list as compact JSON separators(',',':'); duplicate fitting IDs are retained for resampling, so hashes encode order and multiplicity. Evaluation IDs are sorted canonical IDs. Full original score arrays in scores/ are read-only derived NPZ snapshots, with input-file hashes and model signatures in raw_sources.json. Fitting IDs/evaluation IDs are in partitions/ and the three analysis-specific partition JSON files. Subject metadata comes from validated legacy IDs, not inferred model outcomes.

Accuracy, label recall and semantic consistency use0–100 percent; RStd, permutation SD/range, metric contrasts and seed accuracy/RStd SD use percentage points. Prior coordinates and L1 distances use probability units; a0.01 prior change is1 percentage point of prior mass. Original RStd and cyclic RStd are population SD across all K recalls (ddof0). Per-shift accuracy SD uses ddof0 over K shifts; calibration sample SD uses ddof1 over exactly5 prespecified seeds. All shifts of a question stay together. CPA averages content-aligned probabilities before argmax; mean permutation accuracy averages correctness after per-shift argmax. No CI or significance calculation is introduced.

## Analysis policies

1. Equal source budgets: original seed42 source estimation_order[:63], unique IDs; target estimation set and holdout remain primary. MedQA-source prior unchanged, MedMCQA-source prior reduced197→63. The target MedMCQA prior remains197, target MedQA prior63. Difference-versus-primary columns compare the identical target set; neither target population nor target in-domain prior changes.
2. Seed control: random.Random(seed).sample(sorted(original_fit_pool),len(pool)), then sorted prior-averaging order. This is deliberately degenerate and exactly recovers the primary condition at all5 seeds.
3. Informative seed condition: for each benchmark and seed, initialize Python random.Random(seed), draw len(pool) integer positions with rng.randrange(len(pool)), with replacement from sorted original fit IDs. Reinitialize separately per benchmark. Use the draw order, including repeated multiplicities, in the prior mean. Same dataset draws across models and directions. Both priors refitted, original target holdouts fixed. Counts63/197 are draw counts, not unique counts. This conditional resampling alternative was explained before implementation and remains explicitly distinguished from fresh partitions; no reply to the author's preference question was received at report generation.
4. Corrected permutations: saved first path-seeded hybrid draw, independently reconstructed from source_path_seed bytes. All other questions form a heldout-only endpoint. Original and corrected predictions use identical questions; one frozen label prior is applied at every cyclic shift before mapping predicted labels to permanent content. The first path-seeded draw is model specific and differs from primary RQ3 seed42 membership. Five-draw hybrid aggregates remain historical references only.
5. MMLU: core5 and broad8 topic-name groups are in mmlu_grouping_policy.json, written before computing any performance. Full Default diagnostics use all retained questions in each group/subject. Group PriDe differences use only that group's portion of the first-draw MMLU holdout and the benchmark-wide prior; no group-specific or test-fitted prior.

## File semantics

- equal_budget_results.csv:12 rows, wide Default/In-domain/Transfer/CPA endpoints; priors, fit IDs/hash, heldout hash, primary values and new-minus-primary differences.
- calibration_seed_sensitivity.csv:120 rows, policy separates60 with-replacement rows and60 full-pool controls. Per-seed endpoints and contrasts; repeated per-pair mean/sample-SD/min/max/sign counts for convenient filtering. Original seed42 reference fields are prefixed primary_. Do not average the two policies together. Prior vectors and exact fitting draws are JSON-encoded cells.
- corrected_permutation_robustness.csv:30 rows; original/cyclic recall supports, before/after all metrics, endpoint differences and item transition counts. PriDe/Default share one holdout.
- mmlu_medical_subject_analysis.csv:366 rows. grouping=individual_subject for342 rows; core5/broad8 × medical/remaining for24 rows. group_population_n is the actual full-group denominator; group_evaluation_n is the subgroup holdout denominator. population_n/evaluation_n refer to the parent MMLU design. full_default_ fields use the full group; default_/pride_/delta_ fields use the heldout subgroup. These must not be mixed. group support vectors accompany both populations.
- Arrays are compact JSON strings in CSV. NA means undefined/not-applicable, not0. Mathematical NaN is serialized as null in JSON. Values retain computational precision; presentation tables use2 decimals and must not define sign counts from rounded cells. Zero direction uses numerical tolerance1e-9pp to avoid floating subtraction artifacts; original accuracy ties also match exact integer correctness counts.

## Reproduction

For the portable release, install the dependencies listed in requirements.txt and use analysis/scripts/reproduce_results.py with --archive-root to replay the original raw observations. The historical Phase 6 entry points are analysis/scripts/run_phase6_robustness.py, run_phase6_seed_sensitivity.py --policy full_pool_without_replacement_control and --policy fixed_pool_with_replacement, followed by check_phase6_independent.py. These scripts require the complete original archive and its intermediate files. They use exclusive-create for scientific outputs; preserve the finalized directory when preparing another run. CSV export uses the Node.js standard library; figures require Matplotlib. Scientific plots are vector SVG plus 230 dpi PNG. The historical source snapshot is recorded in code_snapshot/.

The root preservation baseline records671 pre-existing files, including all45 Paper files. Final validation compares every one of those bytes, not only the final manuscript PDF. The source manifest records source paths, hashes, script versions and model signatures. The output manifest is a delivery inventory and does not recursively hash itself. The manuscript is deliberately not recompiled during this analysis phase.
""")
    plan="""# Thesis robustness integration plan — Phase 6

18 October 2026. Proposal only. No Paper file was edited. Preserve all three research questions and reproduction → medical extension → medical prior transfer structure. Preserve the original30 full-population summaries,12 paired transfer results and36 approved Phase4 accuracy intervals.

## Main-text priority

1. **Chapter4: methodology.** Add a compact robustness protocol paragraph: source-only nested63 budgets on fixed seed42 targets; conditional with-replacement calibration resampling from original disjoint fit pools at seeds42–46; frozen-prior correction on every heldout cyclic shift using the first saved path-seeded calibration draw. Define sample SD(ddof1) separately from fixed-prior question-bootstrap intervals and permutation SD(ddof0). Give exact denominators and state that neither fresh population partitions nor new model runs were used.
2. **Chapter5: concise robustness subsection.** Present corrected cyclic RStd and consistency as genuinely new measured endpoints: improvements26/30 and24/30, respectively, with Mistral MMLU/MedQA and Qwen3 ARC exceptions. One two-panel corrected_permutation_changes figure is sufficient. Add an RQ3 paragraph/table noting that reverse transfer remains worse under nested equal source counts6/6, with forward unchanged by construction; the directional tendency is partially stable across conditional resamples but smaller comparisons change sign. Use either a compact12-row seed table or the calibration_resampling figure, not both if space is tight. Keep all primary tables and intervals intact.
3. **Chapter6: interpretation.** Replace the statement that corrected item consistency is unmeasured with measured, qualified results. Lower original RStd does not guarantee lower cyclic RStd or higher consistency, now supported directly by counterexamples. Replace the claim that no equal-budget/repeated-calibration analysis exists with the actual source-budget and fixed-pool resampling scope. Explain that source-count equality does not identify a causal medical-domain effect. The five-draw SD and prior L1 variation qualify small transfer margins; neither replaces the original conditional CIs. State that a transfer advantage over in-domain PriDe often fails to beat Default.
4. **MMLU context: one or two sentences only.** MMLU is not purely nonmedical. The core medical subgroup has939 questions and higher Default accuracy/consistency in all six models; the broad1604-question grouping reverses Gemma's consistency contrast. Topic/difficulty and grouping ambiguity prevent causal interpretation. Include this only if it strengthens the existing comparison limitation without distracting from the two medical benchmarks.

## Appendix and reproducibility

Place all equal-budget endpoint/prior tables, all five seed results and sign counts, both calibration policies, full before/after permutation metrics, supports and transition counts, and all57 MMLU subject rows in the appendix or linked supplementary evidence. Use the supplied booktabs fragments as reviewed inputs, not automatic thesis edits. Give exact core5/broad8 included and excluded subject lists, fit/evaluation hashes and duplicate draw multiplicities. The zero-variation full-pool control belongs in the methodological appendix, not as a robustness success claim. The MMLU figure and individual small-subject comparisons should remain appendix material. Keep source-version and metric-unit notes with every exported table.

Suggested compact evidence selection: corrected-permutation figure in main text; one RQ3 equal-budget/seed summary table or figure; all remaining figures in appendix. Do not add a new main chapter, extensive new related work, new hypothesis tests, RStd confidence intervals, a strictly matched-prompt RQ2, or a causal transfer explanation.

## Statements requiring a later edit, after review

- Chapter3 currently says cyclic diagnostics describe uncorrected scores only; retain that definition for primary results and introduce the supplementary frozen-prior condition explicitly.
- Chapter5 currently labels the existing permutation metrics as uncorrected; preserve that label and add a separate heldout analysis, never silently substitute these new denominators.
- Chapter6 says corrected consistency and equal-budget/refitting effects have not been measured. Update these statements to distinguish the new evidence from untested fresh partitions, corrected-input scoring and controlled prompts.
- The abstract and conclusion should be reconsidered only after the author reviews the integration; any change should be a concise qualified finding, with no claim of universal robustness or causal domain effects.

## Review decisions and limits

The with-replacement alternative was explained before implementation because the full original fit pool cannot vary under a fixed holdout. It is scientifically defined and separately verified, but the author's requested methodological preference has not yet been received. Before integrating it, confirm whether to retain this conditional-resampling estimand. It refits both priors and is not a source-only seed study; a source-only variant would answer another question. Rejecting it would leave Analysis2 as a degenerate-control limitation, with Analyses1/3/4 unaffected. No new GPU inference is needed for any completed analysis.

Remaining limits remain unchanged: raw first-token scoring; benchmark population and preprocessing differences; subject cues and four/five choices; only cyclic arrangements; selected calibration pools; ID-level rather than semantic independence; original MMLU None→nan conversions; missing medical truncation diagnostics; pointwise fixed-prior accuracy inference. Main findings should be framed as descriptive evidence within these cached conditions. The final integration is a separate authorized step; this document is not permission to rewrite the thesis automatically.

Evidence directory: analysis/outputs/phase6_robustness_20261018/. Start with robustness_summary.md, scientific_interpretation.md and validation_report.json; the full policies and units are in analysis_protocol.md.
"""
    with (ROOT/'thesis_robustness_integration_plan.md').open('x') as f:f.write(plan)
    write('README.md',"""# Phase 6 verified offline robustness evidence

Read robustness_summary.md first, then scientific_interpretation.md and analysis_protocol.md. validation_report.md/json record scientific, synthetic-test, independent-oracle, export and byte-preservation evidence. The root thesis_robustness_integration_plan.md is a separate review proposal; no active thesis asset was edited.

The four root CSVs are machine-readable full-precision exports. In calibration_seed_sensitivity.csv filter calibration_policy before aggregating: fixed_pool_with_replacement is the informative supplementary condition; full_pool_without_replacement_control is deliberately unchanged. The original seed42 result is preserved separately in primary_ columns. Do not confuse seed SD, permutation SD and historical question-bootstrap confidence intervals.

figures/ contains4 vector SVG/PNG pairs and numerical provenance. tables/ contains11 compact Markdown/booktabs pairs plus the exact source-row map and the CSV-authoring preview. scores/ contains derived validated raw-array snapshots and heldout label predictions; partitions/ and the separate partition JSON files retain all fitting/evaluation IDs and multiplicities. code_snapshot/ freezes only this phase's new analysis/report scripts. Every original source and production module remains unchanged.
""")
    print(json.dumps(dict(tables=len(trace),reports=5,root_integration_plan=True)))

if __name__=='__main__':main()
