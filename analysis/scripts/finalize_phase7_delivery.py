#!/usr/bin/env python3
"""Package the validated manuscript, report decisions and freeze delivery hashes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2];OUT=Path((ROOT/'tmp/phase7_active_output.txt').read_text().strip());PAPER=ROOT/'Paper';P6=ROOT/'analysis/outputs/phase6_robustness_20261018'
sys.path.insert(0,str(ROOT/'analysis'))
from src.offline_phase3b import sha256

def rec(p,role='output'):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p),role=role)
def format_report(text):
    prefixes='All|all|the|with|on|of|after|before|contains|its|one|across|including|under|by|at|and|from|among|primary|seeds|Seeds|seed|Tectonic|Phase|Chapter|Individual|first|randomized|remain|or|below|draws|additive|Sizes|give|decreases|has|checks|in|no|pre-edit|guard'
    text=re.sub(r'\b('+prefixes+r')(?=\d)',r'\1 ',text)
    text=re.sub(r'([:;,])(?=\d)',r'\1 ',text)
    text=re.sub(r'\bddof([01])\b',r'ddof=\1',text)
    return re.sub(r'(?<=\d)(?=pp\b)',' ',text)
def write(path,text):
    text=format_report(text)
    with path.open('x') as f:f.write(text.rstrip()+'\n')

def main():
    v=json.loads((OUT/'claims_and_structure_validation.json').read_text());m=json.loads((OUT/'methodology_verification.json').read_text());visual=json.loads((OUT/'visual_validation.json').read_text());baseline=json.loads((OUT/'preservation_baseline.json').read_text())
    assert v['status']==m['status']==visual['status']=='pass' and v['pdf_sha256']==visual['pdf_sha256']==sha256(PAPER/'build/thesis.pdf')
    assert not v['warnings'] and not v['example_overlap_matches'] and not v['model_dependencies_loaded']
    for r in baseline['protected']:assert sha256(Path(r['path']))==r['sha256']
    shutil.copy2(PAPER/'build/thesis.pdf',PAPER/'thesis.pdf');assert sha256(PAPER/'thesis.pdf')==v['pdf_sha256']
    logs=OUT/'build_logs';logs.mkdir(exist_ok=True)
    for name in ['thesis.log','thesis.blg','thesis.aux','thesis.toc']:shutil.copy2(PAPER/'build'/name,logs/name)
    code=[ROOT/'analysis/scripts'/x for x in ['verify_phase7_evidence.py','build_phase7_robustness_assets.py','validate_phase7_manuscript.py','finalize_phase7_delivery.py']]
    snapshot=OUT/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for p in code:shutil.copy2(p,snapshot/p.name)
    sourcefiles={Path(p) for p in v['active_input_paths']}|{PAPER/'bibliography.bib',PAPER/'sigillum.png',PAPER/'README.txt',PAPER/'reproducibility_manifest.json'}|set((PAPER/'figures').glob('*.pdf'))
    entries=[dict(path=str(p.relative_to(PAPER)),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(sourcefiles)]
    archive=OUT/'thesis_source_phase7_20261018.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sourcefiles):z.write(p,'Paper/'+str(p.relative_to(PAPER)))
        z.writestr('source_manifest.json',json.dumps(dict(files=entries,manuscript_pdf_sha256=v['pdf_sha256']),indent=2)+'\n')
        z.writestr('README.txt','Phase7 validated LaTeX source. Compile Paper/thesis.tex. The separate project evidence directories are referenced by path and are not duplicated in this source-only archive. Primary scientific evidence is unchanged.\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for e in entries:assert hashlib.sha256(z.read('Paper/'+e['path'])).hexdigest()==e['sha256']
    (OUT/'packaging_validation.json').write_text(json.dumps(dict(status='pass',archive=rec(archive),source_files=len(entries),all_zip_members_match=True),indent=2)+'\n')
    log=f"""# Robustness integration log — Phase 7

18 October 2026. Phase6 methods were verified before any manuscript edit. The complete previous Paper directory, including its53-page PDF and45 original files, was copied to `{baseline['backup']}`. Every backup byte matches the pre-edit baseline.

## Integration decisions

| Component | Final change and purpose |
|---|---|
| Chapter3 | Adds two short paragraphs defining frozen-prior correction across heldout cycles and separating calibration-pool sample SD from permutation SD and question-bootstrap intervals. Existing derivations and all21 equation blocks remain unchanged. |
| Chapter4 | Adds supplementary conditions: source-only nested63 budgets; with-replacement pool resampling at seeds42–46; first-draw disjoint corrected cyclic evaluation; predeclared core/broad MMLU groups. Exact counts, restrictions and populations are explicit. |
| Chapter5 | Adds **Robustness and Sensitivity Analyses** after RQ3 and before workload. One12-row transfer sensitivity table and one two-panel corrected-robustness figure present the material findings. The MMLU comparison remains a short qualifying paragraph. Primary tables and estimates are preserved. |
| Chapter6 | Replaces unexecuted-follow-up statements with verified evidence. Equal source counts do not remove the directional tendency; small margins can change sign within calibration pools. Measured consistency gains and exceptions qualify the relationship between balance, accuracy and robustness. No causal domain explanation or significance claim is added. |
| Conclusion | Adds only the qualified corrected-robustness and transfer-sensitivity implications. Repeated framing was shortened to keep the conclusion on one page. |
| Abstract | Adds one short qualitative robustness sentence, after revising the main chapters. It adds no robustness counts or extra catalogue of experiments. |
| Appendix | Adds19 supporting tables: equal-budget endpoints/contrasts/priors, all60 resampled fits' four-method accuracy and original RStd, all30 pairs' complete before/after cyclic metrics, all57 subject sizes/supports and both MMLU grouping comparisons. Detailed individual model–subject rows remain available in the electronic appendix, explicitly referenced. |
| Reproducibility | Updates source-version metadata and README. Frozen inputs, source maps, full memberships, duplicate multiplicities, validation reports and implementation locations are identified. |

## Methodology reconciliation

The earlier generic seed-sensitivity heading was broader than the actual experiment. The manuscript consistently calls the executed condition **calibration-pool resampling sensitivity**. Each of five predetermined generator seeds draws whole calibration questions with replacement at63/197 nominal counts, retaining duplicate multiplicity. Both priors are refitted within their original pools; target holdouts stay fixed. Original seed42 is a separate reference, not one of five new independent partitions. This discrepancy was reported before integration. The Phase7 instruction explicitly authorizes verifying and incorporating this condition with accurate terminology, resolving the earlier Phase6 integration preference.

Forward equal-budget transfer averages the same63 MedQA source questions and leaves both target quantities unchanged. Reordering the prior sum changes coordinates by at most2.8e-16 but changes no prediction or endpoint. Reverse transfer instead replaces197 MedMCQA source questions by the first63 in the retained randomized order. Source-count equality does not match targets or isolate a domain effect.

## Preservation and scope

All{v['protected_nonmanuscript_files']} protected nonmanuscript files retain their hashes, including Phase3B/4/6 source/output evidence, probabilities, priors, predictions and approved bootstrap arrays. The original generated numerical tables and three primary figure PDFs remain byte-identical. Introduction, Related Work, bibliography, master, metadata values and research questions are preserved. No new inference, experimental fit, confidence interval or model dependency was introduced in Phase7.

The added figure reproduces verified Phase6 values. Publication tables are generated from frozen JSON, with1620 metric cells linked to source fields; source-prior coordinates and all57 subject support rows receive additional direct checks. New outputs are the active LaTeX/PDF, the three requested reports and the timestamped evidence package. Older phase reports remain historical snapshots and were not rewritten.

## Writing and layout review

Both supplied example works were reread for concrete metric explanation, comparison populations and design-linked limitations. Their stronger causal interpretations were not adopted. No12-token prose overlap was found between added chapter text and the examples. The Results section reports observations; Discussion explains their consequences rather than repeating its full table or numerical catalogue.

The full project compiles with Tectonic0.17.0, BibTeX and all reference passes, with zero warnings. Visual review corrected a short conclusion continuation and the new appendix's floating-table order. Resampling model blocks and before/after group pairs stay together across longtable breaks. Final rendered pages, source diff, numerical claims and structure checks are retained under `{OUT.relative_to(ROOT)}`. The delivered PDF has{v['pages']} pages and SHA256 `{v['pdf_sha256']}`.
"""
    write(ROOT/'robustness_integration_log.md',log)
    exceptions='\n'.join('| '+r['model']+' | '+r['benchmark']+' | '+f"{r['cyclic_rstd_delta_pp']:+.6f}"+' |' for r in m['cyclic_rstd_exceptions'])
    consexceptions='\n'.join('| '+r['model']+' | '+r['benchmark']+' | '+f"{r['consistency_delta_pp']:+.6f}"+' |' for r in m['consistency_exceptions'])
    claims=f"""# Robustness claims validation — Phase 7

Status: PASS. Source version phase6-robustness-v1; primary Phase3B results and Phase4 fixed-prior intervals remain authoritative. The pre-edit audit verified all172 Phase6 delivery files and276 source records. Later manuscript edits intentionally change manuscript-context hashes; the653 protected nonmanuscript files remain unchanged.

## Experimental definitions checked

- **Equal source budget:**63 unique questions per source. The MedMCQA IDs are exactly the first63 of its retained randomized197-question selection order and lie wholly inside its original fitting pool. Target priors remain63 MedQA or197 MedMCQA. Every target method uses the identical primary1210/3744 IDs and ordered-ID hash. Forward membership/prior is mathematically unchanged, with machine accumulation differences below2.8e-16 and exactly identical predictions/endpoints; reverse changes only the MedMCQA source prior. This is not a causal domain contrast.
- **Calibration-pool resampling:** the unit is a whole question and complete cached orbit. Seeds42–46 initialize Python Random separately for each benchmark; randrange draws63/197 indices with replacement from sorted original fitting IDs. Repeated priors enter the mean with full multiplicity. Both priors vary; Default and CPA stay fixed. Original seed42 is a reference only. Full-pool controls reproduce all primary priors/endpoints exactly. The resampling condition estimates conditional pool variability, not independent splits or uncertainty over all possible partitions. Sample SD uses ddof1; no CI is replaced or added.
- **Corrected cycles:** every first-draw fitting set is disjoint from its evaluated complement. The frozen prior corrects displayed-label probabilities before permanent-content mapping, preserving additive1e-10, first exact maximum and complete four/five cycles. All30 Default/PriDe pairs use identical heldout IDs and gold content. Direct scalar checks reconstruct the outcomes independently of the Phase6 metric wrapper.
- **MMLU:** core5 names and broad8 additions match the pre-performance policy. Sizes939/12653 and1604/11988, all57 subject names and gold supports match the retained population. Full Default and disjoint-heldout corrections are kept separate. Both groupings and Gemma's classification-dependent consistency contrast are reported without a causal interpretation.

## Corrected robustness counts and every exception

Strict unrounded signs give26/30 cyclic-RStd decreases and24/30 consistency improvements. Integer question gains minus losses independently confirm all consistency signs. There are no ties in these endpoints; applying the Phase6 numerical guard1e-9pp does not change a sign. Two-decimal table rounding does not determine direction.

Cyclic RStd increases:

| Model | Benchmark | PriDe−Default (pp) |
|---|---|---:|
{exceptions}

Semantic consistency decreases:

| Model | Benchmark | PriDe−Default (pp) |
|---|---|---:|
{consexceptions}

Qwen3 MedMCQA loses exactly one consistent question. These counts are descriptive, dependent model–benchmark outcomes, not statistically significant findings. Original accuracy has20 improvements,6 losses and4 exact ties, while original RStd decreases25/30; neither is inferred from the cyclic/consistency counts.

## Numerical claim and presentation checks

All1620 mapped metric cells in20 new tables match the frozen source fields before rounding. Source-prior coordinates (five-decimal display), subject population sizes and all gold-support cells are directly rechecked as well. The source map identifies model, benchmark/direction, draw, group and field. The figure map records its60 plotted values and Phase6 source hash. Full exported data retain precision, IDs, multiplicities and probability units.

The executable claim audit checks{len(v['claim_checks'])} named arithmetic statements, including equal-budget reverse signs and extrema, resampling sign/unique-question counts, maximum prior-coordinate SD, Llama/GLM ranges, every main-text corrected-permutation example and both MMLU grouping directions. Added numerical source lines are retained for contextual review. Detailed evidence is in `claims_and_structure_validation.json`, `methodology_verification.json`, `table_source_map.json`, `figures/figure_source_map.json` and `manuscript_revision.diff` under `{OUT.relative_to(ROOT)}`.

## Primary preservation and references

All30 primary summaries,12 paired transfers and36 approved accuracy intervals are unchanged. Existing generated tables and three primary scientific figures retain their bytes. All21 equation blocks are unchanged; all16 active citation keys are defined, labels are unique, and internal references resolve across27 active LaTeX inputs. The final compiler log has zero warnings, undefined references/citations or overfull/underfull boxes. PDF text contains no broken-reference markers or placeholders. No GPU inference or new statistical method ran during integration.
"""
    write(ROOT/'robustness_claims_validation.md',claims)
    review=f"""# Final post-robustness review — Phase 7

18 October 2026. The active thesis integrates the scientifically verified Phase6 findings. The final PDF has **{v['pages']} pages** and was compiled from the complete multi-file source with Tectonic0.17.0. The conclusion fits one page; the seven main chapters retain their original structure.

## What entered the manuscript

Equal-source-budget transfer, calibration-pool resampling sensitivity and frozen-prior corrected permutation robustness appear in a concise Chapter5 subsection after RQ3 and before workload. One combined transfer table and one two-panel corrected-robustness figure present the main evidence. The short MMLU paragraph qualifies the broad-reference interpretation; it does not replace the medical extension.

Chapter3 adds only the new comparison definitions; Chapter4 records actual fitting policies, duplicate multiplicity and denominators. Discussion interprets partial stability, metric disagreements and unresolved explanations. Conclusion states only the material qualifications. Abstract adds one short qualitative consistency finding without further numbers. Introduction and Related Work remain unchanged.

The appendix contains19 additional supporting tables: complete equal-budget endpoints/contrasts/source priors; all60 conditional resampled fits' four-method accuracy and original RStd; complete before/after seven-metric cyclic results for all30 pairs; all57 subject sizes/supports; and both MMLU grouping comparisons on full Default and separate heldout corrections. Individual342 model–subject rows, full memberships, priors, multiplicities and machine-precision values remain in the explicitly referenced electronic appendix. No main-text catalogue repeats all these outputs.

## Scientific conclusions and terminology

The directional transfer tendency remains visible with equal source counts, although the forward comparison is unchanged by construction. It is partially stable under calibration-pool resampling: smaller contrasts sometimes change sign. This measures variability conditional on selected pools and fixed targets, refitting both priors; it does not represent five independent splits, all possible calibration samples or a new confidence interval. The original seed42 is a separate reference. The manuscript corrects the earlier generic seed-sensitivity framing throughout.

Fixed-prior PriDe reduces cyclic RStd in26/30 and improves content consistency in24/30 heldout pairs, with every exception disclosed. These measured benefits strengthen the robustness evidence, while disagreement with accuracy and original RStd limits a universal-success claim. The MMLU grouping sensitivity reinforces descriptive rather than causal comparison. The primary RQ1/RQ2/RQ3 results and all approved intervals remain unchanged.

## Validation and presentation

- Methods/IDs: all12 equal-budget comparisons,120 resampling/control rows and30 corrected-cycle pairs were audited; fitting/test separation, target hashes, complete K cycles, gold mappings, epsilon and ties match execution. MMLU groups were defined before subgroup outcome calculation.
- Numeric evidence:1620 table metric cells and{len(v['claim_checks'])} named arithmetic claims pass; prior coordinates and all57 subject supports are checked separately. Figure values come directly from verified JSON.
- Preservation:653 nonmanuscript evidence/source files remain unchanged. The complete pre-edit45-file Paper backup is verified. All old generated numerical assets, primary figures, bibliography, RQs, model identities and statistical methods are preserved.
- Compilation:27 active LaTeX inputs;16 resolved citation keys;21 unchanged equation blocks; unique labels and resolved references; **zero compiler warnings**, including undefined citations/references and overfull/underfull boxes. All four active scientific figures resolve.
- Visual review: the initial full rendering was inspected in overview, then every changed final page was reinspected. Identical-page comparisons carry forward unchanged reviewed pages. Enlarged review covers the main sensitivity table, corrected heatmaps, longtable continuations, subject supports, conclusion and declaration. Tables are readable and unclipped; table numbering follows source order and paired rows/model draw blocks stay together. The delivered PDF is byte-identical to the validated build.
- Style: both supplied examples informed comparison logic and design-linked limitations; no12-token overlap was found in the added chapter prose. Results and Discussion serve different functions, and old statements that the completed supplements were unmeasured have been removed.

## Remaining methodological review

There is no unresolved implementation discrepancy or additional methodological approval blocking this integration. The Phase7 request authorizes incorporating the executed resampling condition after verification with accurate terminology. Its conditional scope remains an explicit limitation, as do one nested MedMCQA subset, one first-draw corrected evaluation, nonrandomized subject groups, first-token/raw scoring, cyclic-only arrangements, ID-level independence, the MMLU input conversion and missing medical diagnostics. No claim resolves these limits through causal inference, statistical significance or clinical outcomes. No additional GPU inference is necessary for the completed work.

The author should review the scientific interpretation and authorship/tool disclosure before submission. This is not a new experiment authorization or a certification of institutional submission rules.

## Deliverables and provenance

- Active master: `Paper/thesis.tex`; final PDF: `Paper/thesis.pdf`.
- Requested reports: `robustness_integration_log.md`, `robustness_claims_validation.md` and this review.
- Source archive: `{archive.relative_to(ROOT)}` ({len(entries)} exact source/assets, plus internal manifest/README).
- Versioned validation, source maps, backups, compiler logs, rendered pages and output hashes: `{OUT.relative_to(ROOT)}`.

Final PDF SHA256: `{v['pdf_sha256']}`.
"""
    write(ROOT/'final_post_robustness_review.md',review)
    sources=[rec(Path(r['path']),'protected_prior_evidence') for r in baseline['protected']]+[rec(p,'phase7_implementation') for p in code]
    for rel in ['writing_style_guide.md','targeted_revision_log.md','final_quality_review.md','thesis_robustness_integration_plan.md','beispiel_arbeit/b-thesis_llm-consistency.pdf','beispiel_arbeit/Team4_vCommented.pdf']:
        sources.append(rec(ROOT/rel,'read_only_style_or_revision_context'))
    request=Path('<local-attachment> text.txt');sources.append(rec(request,'user_phase7_request'))
    (OUT/'source_manifest.json').write_text(json.dumps(dict(inputs=sources,source_version='phase6-robustness-v1',backup_manifest_sha256=sha256(OUT/'preservation_baseline.json')),indent=2)+'\n')
    delivery=dict(status='pass',pages=v['pages'],pdf_sha256=v['pdf_sha256'],primary_results_changed=False,bootstrap_intervals_changed=False,source_archive_verified=True,all_figures_resolve=True,all_citations_resolve=True,compiler_warnings=0,all_protected_nonmanuscript_files_unchanged=len(baseline['protected']),validation_report=str(OUT/'claims_and_structure_validation.json'),visual_validation=str(OUT/'visual_validation.json'),new_gpu_inference=False,unresolved_methodology_approval=False)
    (OUT/'delivery_validation.json').write_text(json.dumps(delivery,indent=2)+'\n')
    outputs=[rec(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json'];external=[rec(p,'active_manuscript_or_report') for p in sorted(sourcefiles|{PAPER/'thesis.pdf',ROOT/'robustness_integration_log.md',ROOT/'robustness_claims_validation.md',ROOT/'final_post_robustness_review.md'})]
    (OUT/'output_manifest.json').write_text(json.dumps(dict(files=outputs,external_deliverables=external,excludes='this manifest itself'),indent=2)+'\n')
    print(json.dumps(dict(status='pass',pages=v['pages'],source_files=len(entries),protected=len(baseline['protected']),reports=3,pdf_sha256=v['pdf_sha256'])))

if __name__=='__main__':main()
