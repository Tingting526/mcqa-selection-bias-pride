# Selection Bias and PriDe in Medical Multiple-Choice QA

Code for the Bachelor's thesis *Selection Bias in Open-Weight Large Language Models for Medical Multiple-Choice Question Answering: Reproducing and Extending PriDe on MedQA and MedMCQA* (Department of Statistics, LMU Munich, 2026).

The thesis has three research questions:

- **RQ1:** Does the selection-bias pattern reported by Zheng et al. (2024) reappear for six newer instruction-tuned open-weight models on MMLU, ARC-Challenge and CommonsenseQA, and how do CPA and PriDe change accuracy and recall imbalance?
- **RQ2:** How do these methods behave on the medical benchmarks MedQA and MedMCQA?
- **RQ3:** How does a PriDe option-ID prior estimated on one medical benchmark perform when applied to the other?

This repository contains the implementation that produced the thesis results. It is not a simplified re-implementation. See [Provenance](#provenance) for how this was checked.

## Contents

```text
analysis/
  src/                      implementation
    data/                   dataset loaders, normalization, preprocessing and validation
    prompts/                raw-text MCQ prompt construction
    models/                 model loading and first-token scoring
    permutations/           cyclic option permutations
    pride/                  PriDe prior estimation, correction and transfer;
                            zheng.py wraps the pinned original PriDe numerics
    experimental/           original Zheng et al. conditions for open models
    metrics/                accuracy, recall, RStd, paired comparisons
    evaluation/             baseline, permutation and PriDe evaluation
    offline_phase3b.py      primary RQ1-RQ3 analysis from saved scores
    phase4_statistics.py    paired question bootstrap with fixed priors
    phase6_robustness.py    calibration and seed robustness analyses
    thesis_strengthening.py prompt-length, ARC-order, choice-type, native-template analyses
  scripts/                  command-line entry points (inference, analysis, tables, figures)
    slurm/                  LRZ AI Systems job scripts
  configs/                  datasets, models (pinned revisions), preprocessing, experiment
  tests/                    pytest suite
  thesis_tables_static/     the two hand-maintained thesis tables
  outputs/                  derived result files needed to rebuild the thesis tables
Paper/
  generated/                the 28 thesis tables exactly as included in the manuscript
  figures/                  Chapter 5 figure generator and its full-precision data snapshot
  reproducibility_manifest.json  source versions of the thesis results (paths refer to the full project archive)
provenance/redactions.json  record of the path redactions described below
reproduce_tables.sh         rebuilds and checks the 28 tables
```

## Installation

Python 3.10 or newer. Inference was run with Python 3.11.16, PyTorch 2.14.0 (CUDA 13.0) and Transformers 5.17.0 on NVIDIA A100 80 GB GPUs (LRZ AI Systems), in bfloat16 without quantization.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers inference, analysis, figures and tests. Rebuilding tables and figures and running the tests needs no GPU. Inference needs a CUDA GPU.

## Data

Datasets are not redistributed here. They are downloaded from their original sources by the scripts:

| Benchmark | Source | Split and revision |
|---|---|---|
| MMLU, ARC-Challenge, CommonsenseQA | [chujiezheng/LLM-MCQ-Bias](https://github.com/chujiezheng/LLM-MCQ-Bias) | commit `2ae2f40c77006f7a00a4675bdfb30151c2406691`, archive SHA-256 `cc097f1a…ba810` |
| MedQA (4 options) | [openlifescienceai/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/openlifescienceai/MedQA-USMLE-4-options-hf) | test, revision `20a8f4d6b851f6391751f6e76c06bc3a26c83e0b` |
| MedMCQA | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | validation, revision `91c6572c454088bf71b679ad90aa8dffcd0d5868` |

```bash
cd analysis
python scripts/download_zheng2024.py                 # reference benchmarks and pinned PriDe code
python scripts/download_data.py --dataset medqa
python scripts/download_data.py --dataset medmcqa
python scripts/preprocess_data.py --dataset medqa
python scripts/preprocess_data.py --dataset medmcqa
```

`download_zheng2024.py` also fetches the original `debias_utils.py`. `src/pride/zheng.py` loads it at run time after checking its SHA-256, so the PriDe numerics are the original ones. The reference code is not copied into this repository.

## Models

All six models are open-weight instruction-tuned models from the Hugging Face Hub, pinned to these revisions (also in `analysis/configs/models/`):

| Model | Repository | Revision |
|---|---|---|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Gemma 2 9B IT | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Qwen3 8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| GLM-4 9B Chat | `zai-org/glm-4-9b-chat-hf` | `8599336fc6c125203efb2360bfaf4c80eef1d1bf` |

Llama and Gemma are gated: accept their licenses on the Hub and log in with `huggingface-cli login`. Never put access tokens into job scripts. Model weights are not part of this repository.

## Reproduction

All commands run from `analysis/`. Steps 1 and 4 need a GPU. Everything else runs on a CPU from saved scores.

### 1. Inference: first-token scores (GPU)

```bash
python scripts/reproduce_pride.py --model qwen2.5-7b-instruct            # all five benchmarks
python scripts/start_pride_reproduction.py --submit --partition <gpu-partition>   # all models via Slurm
```

`reproduce_pride.py` (implementation `zheng-pride-0shot-v1`) scores raw-text zero-shot prompts at the four cyclic shifts with first-token probabilities over the two candidate slots per label. It also computes Default, CPA and five-draw hybrid PriDe (5% calibration) with the pinned original numerics. Raw observations, priors, partitions and checksums are written under `outputs/reproductions/`. `scripts/slurm/` contains the job scripts and an LRZ setup guide (in German). LRZ user and project paths have been replaced by placeholders such as `<lrz-user>`, so set them for your own cluster.

### 2. Primary analysis: RQ1, RQ2 and RQ3 (CPU)

```bash
python scripts/analyze_phase3b_offline.py --finalize --output outputs/<new-dir>
python scripts/check_phase3b_transfer.py --output outputs/<new-dir>
```

This verifies the saved scores, recomputes the 30 model-benchmark summaries and computes the 12 direction-specific medical prior transfers. RQ3 uses fixed priors on the seed-42 holdouts (MedQA 63/1,210, MedMCQA 197/3,744).

### 3. Uncertainty and robustness (CPU)

```bash
python scripts/finalize_phase4_statistics.py --output outputs/<new-dir>   # paired question bootstrap, 1000 draws
python scripts/run_phase6_robustness.py                                  # calibration-pool resampling, equal budgets
python scripts/run_phase6_seed_sensitivity.py --policy fixed_pool_with_replacement
python scripts/run_phase6_seed_sensitivity.py --policy full_pool_without_replacement_control
python scripts/strengthen_thesis_offline.py                              # prompt lengths, ARC orders, choice types, switches
```

### 4. Supplementary sensitivity runs (GPU)

```bash
python scripts/run_thesis_sensitivity.py plan --output <plan.json>
python scripts/run_thesis_sensitivity.py execute --plan <plan.json> --model <model> --output <dir>
python scripts/finalize_thesis_sensitivities.py
python scripts/verify_thesis_prompt_serialization.py
```

These cover the literal MMLU input repair and native chat templates. Each new condition refits all priors on the original calibration memberships and does not replace primary observations.

### 5. Thesis tables (CPU, five steps)

The 28 tables in `Paper/generated/` are produced in five steps, and the order matters:

1. `build_phase4_thesis_tables.py --output analysis/outputs/phase4_20261018_20261018T091030Z` (needs an empty `tables/latex` in that directory)
2. `restructure_rq1_presentation.py`
3. `build_strengthening_tables.py`
4. `build_phase7_robustness_assets.py`
5. `finalize_thesis_tables.py --input <generator output> --output <final dir> --check Paper/generated`

Step 5 applies, in this order:

- the table restructuring of 3 November 2026
- all approved caption and note revisions, listed explicitly in `scripts/thesis_table_text_revisions.json` (each must match exactly once or the script stops)
- the two hand-maintained tables from `thesis_tables_static/`

The revisions change captions and notes only. No table cell changes.

Steps 1 to 4 write directly into `Paper/generated/` and would overwrite approved captions if run in place. Use the wrapper instead:

```bash
./reproduce_tables.sh
# {"files": 28, "identical": 28, "different": []}
```

It copies the repository to a temporary directory, runs all five steps there, and compares the result byte for byte with the committed `Paper/generated/`. The committed files are never written. The wrapper downloads the Zheng et al. (2024) paper PDF from the ICLR proceedings, because `restructure_rq1_presentation.py` records its SHA-256. The PDF is not redistributed.

**Rule for later caption edits:** add them as new `after_structural` entries to `thesis_table_text_revisions.json`. A caption edited only in `Paper/generated/` is lost at the next regeneration.

### 6. Figures (CPU)

```bash
python Paper/figures/generate_chapter5_figures.py --output-dir <fresh-dir>
python analysis/scripts/plot_transfer_contrasts.py      # paired transfer contrasts with bootstrap intervals
python analysis/scripts/plot_heatmaps_wide.py           # cyclic RStd and consistency heatmaps
```

The first command uses the full-precision snapshot `Paper/figures/chapter5_data/plot_data.json`. The other two read the bundled result files in `analysis/outputs/` and write their PDFs to `Paper/figures/`. None of them performs inference or refitting. With Matplotlib 3.10.7 the output matches the thesis figures. Other Matplotlib versions change only the SVG serialization, not the coordinates or labels.

### Tests

```bash
cd analysis && python -m pytest
```

Without raw scores, 175 tests pass. 12 further tests validate the saved raw LRZ observations (`outputs/lrz-results/`, `outputs/preprocessed/`, several hundred MB, not included). With those files present, all 187 tests pass. One integration test that loads a real model is deselected by default (`-m integration`, needs a GPU).

## What is included and what is not

| Included | Not included |
|---|---|
| Complete source code, scripts, configs and tests | Dataset copies, which are downloaded from their sources |
| Derived results needed to rebuild the 28 tables (`analysis/outputs/`) | Model weights |
| The 28 manuscript tables and the figure data snapshot | Raw per-question scores and priors (several hundred MB) |
| `Paper/reproducibility_manifest.json` | Editorial revision logs and backups named in that manifest |
| | The thesis PDF and manuscript sources |

Without the raw scores, you can rebuild tables, figures and statistics from the bundled derived results. Rebuilding the derived results needs either the raw-score archive or a new inference run (step 1).

## Provenance

The code was checked against the hashes recorded when the experiments ran:

- The inference release `pride-reproduction-20261015` matches 16 of 16 files by SHA-256.
- The Phase 3B analysis matches its execution snapshot, and the Phase 6 and Phase 7 source manifests match all files.
- `build_strengthening_tables.py` and `build_phase4_thesis_tables.py` differ from their recorded versions only by later caption changes. The tables they produce are verified by `reproduce_tables.sh`.

For publication, local user names, LRZ account and project identifiers, and absolute local paths were replaced by placeholders (`<project-root>`, `<lrz-user>`, `<lrz-project>`, `<local-user>`, `<local-attachment>`). `provenance/redactions.json` lists every affected file with its executed and published SHA-256. These replacements touch only paths in comments, documentation, defaults of Slurm scripts and provenance strings. Directory names of editorial revision logs were shortened in `finalize_thesis_tables.py` and in the manifest. The table check above runs on the redacted files.

The manuscript-delivery validator `validate_phase4_delivery.py` is omitted because it checks the thesis PDF and contains personal data. It is not part of the experiments.

## Reference

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou and Minlie Huang. 2024. Large Language Models Are Not Robust Multiple Choice Selectors. *ICLR 2024*.
