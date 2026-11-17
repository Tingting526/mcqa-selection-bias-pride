# Selection Bias and PriDe in Medical Multiple-Choice QA

Code and derived results for the Bachelor's thesis *Selection Bias in Open-Weight Large Language Models for Medical Multiple-Choice Question Answering: Reproducing and Extending PriDe on MedQA and MedMCQA* (Department of Statistics, LMU Munich, 2026).

## Version cited in the thesis

Appendix A.17 cites commit [`2fcfb67748413ae4d5059e42d18eced8837de686`](https://github.com/Tingting526/mcqa-selection-bias-pride/tree/2fcfb67748413ae4d5059e42d18eced8837de686). That exact snapshot remains available and is also tagged `thesis-cited-version`.

The current branch adds installation, test, CSV-export and reproducibility fixes to that snapshot. The submitted result values, all 28 manuscript tables and the original numerical functions are unchanged. Use the commands below to reproduce the results with these portability fixes. `provenance/technical_revision.json` records the differences from the cited commit. The corrected code release is tagged `thesis-submission-2026-11-17-v2`.

The thesis studies three questions:

- **RQ1:** Do the selection-bias patterns reported by Zheng et al. (2024) reappear for six newer instruction-tuned open-weight models on MMLU, ARC-Challenge and CommonsenseQA?
- **RQ2:** How do Default, cyclic permutation averaging (CPA) and PriDe behave on MedQA and MedMCQA?
- **RQ3:** How does an option-ID prior estimated on one medical benchmark perform on the other?

The experimental implementation is in `analysis/src/`. The portable submission check reuses its numerical functions and verifies the saved predictions. It does not run language models.

## Installation

Use Python 3.12 for the tested installation below. The package supports Python 3.10 or newer with compatible dependency versions, but the frozen validation environment targets 3.12. Node.js 18 or newer is needed only for CSV exports. Those exports use the Node.js standard library and need no npm packages.

All commands below start in the repository root unless a subshell explicitly changes directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-tested.txt
```

`constraints-tested.txt` records the versions used for the CPU release checks on macOS arm64. It is separate from the historical inference environment: Python 3.11.16, PyTorch 2.14.0 with CUDA 13.0, and Transformers 5.17.0 on NVIDIA A100 80 GB GPUs, using bfloat16 without quantization. Other platforms may require a platform-specific PyTorch installation. `requirements.txt` specifies the supported package ranges.

## Reproduce the submitted results without a GPU

The repository contains the derived evidence needed for these commands. No dataset or model download is needed for the first command, the figures or the CSV exports.

```bash
python analysis/scripts/reproduce_results.py --output analysis/outputs/reproduction-check
./reproduce_tables.sh
python Paper/figures/generate_chapter5_figures.py --output-dir submission-check/figures
python analysis/scripts/plot_transfer_contrasts.py
python analysis/scripts/plot_heatmaps_wide.py
```

The result check verifies the SHA-256 hashes of 42 scientific input files, the 30 model–benchmark conditions, all 12 medical transfer directions and the 36 paired bootstrap intervals. It recomputes accuracy and RStd from saved per-question transfer predictions and recomputes the 1,000-draw fixed-prior question bootstrap. It also checks the exact-count direction classifications against the saved five calibration draws. Outputs are `verification.json` and `statistics_results.json`. Choose a new `--output` directory for each run; existing results are never overwritten.

This check establishes consistency of the submitted derived evidence. It does not establish that raw model scores were produced correctly. Use the raw-archive replay below for the additional source checks.

The table wrapper runs all five generators in a temporary copy and compares the result with `Paper/generated/`. Expected result:

```text
{"files": 28, "identical": 28, "different": []}
```

It downloads and checks the SHA-256 of the Zheng et al. paper PDF from the ICLR proceedings. An internet connection is needed unless that PDF is already cached at `analysis/outputs/rq1_restructure_20261021/sources/zheng_iclr2024.pdf`. The temporary copy excludes virtual environments and raw-data directories. Set `PYTHON=/path/to/python` to select another interpreter; `--keep` retains the temporary files.

The figure scripts use the full-precision snapshot in `Paper/figures/chapter5_data/plot_data.json` and the bundled result JSON files. They do not perform inference or refit priors. Plot appearance and PDF/SVG serialization can vary with Matplotlib and fonts; the table comparison is byte-exact.

### CSV exports

Input and output directories are separate. The exporters preserve numerical precision, encode arrays as JSON, verify CSV round-trips and refuse to overwrite different CSV content. Re-exporting identical files is allowed.

```bash
node analysis/scripts/export_phase3b_tables.mjs analysis/outputs/phase3b_20261018/verified analysis/outputs/csv-export/phase3b
node analysis/scripts/export_phase4_statistics.mjs analysis/outputs/phase4_20261018_20261018T091030Z analysis/outputs/csv-export/phase4
node analysis/scripts/export_phase6_results.mjs analysis/outputs/phase6_robustness_20261018 analysis/outputs/csv-export/phase6
```

These produce 14 primary-analysis tables, three statistical tables and four robustness tables. The primary export also writes the 12 transfer JSON files referenced by `rq3_contrasts.csv` and records their hashes. Existing transfer files are retained if their parsed content matches the input; different results are never overwritten. The robustness export includes both the 60 calibration-resampling rows and the 60 full-pool control rows. Its missing-value marker is `NA`; the other exports use empty fields. JSON manifests record the exported hashes. CSV export verifies the conversion, not the underlying raw observations.

## Tests

```bash
(cd analysis && python -m pytest -ra)
node --test analysis/tests/csv.test.mjs
```

External fixtures are optional. Tests that need absent reference data or the raw LRZ archive report a skip with instructions. A present but invalid fixture still fails. The real-model integration test is excluded by default.

Download the pinned upstream reference code and datasets to enable the numerical oracle tests:

```bash
python analysis/scripts/download_zheng2024.py
(cd analysis && python -m pytest -ra --require-reference)
```

`--require-reference` turns missing reference data into an error. For the complete original archive tests, restore the archived `analysis/outputs/lrz-results/` and `analysis/outputs/preprocessed/` trees without changing their contents, then run:

```bash
(cd analysis && python -m pytest -ra --require-reference --require-archive)
```

The archive tests require the original pilot runs as well as the full-result summaries. They test the archived experiment, so newly generated runs do not replace those fixtures. GPU tests can be selected with `-m integration` on suitable hardware. The GitHub Actions workflow runs the portable checks; it does not have access to the raw archive or gated model weights.

## Replay the original raw-score archive

The raw archive is not included in Git. Keep its relative directory structure intact and provide the project root containing `analysis/`:

```bash
python analysis/scripts/reproduce_results.py --archive-root /path/to/original-project --output analysis/outputs/raw-replay
```

The archive must contain the summary, observation and manifest files named in the bundled `analysis/outputs/phase3b_20261018/verified/analysis_results.json`, the pinned reference data under `analysis/data/reference/zheng2024/`, the original medical preprocessing artifacts under `analysis/outputs/preprocessed/`, and the model configurations under `analysis/configs/models/`. `<project-root>` paths in the index are resolved against `--archive-root`; no historical username or absolute home path is required.

The replay checks source hashes, population identities, model configurations, complete permutations and recorded scoring traces. It then recomputes all 30 Default/CPA/five-draw PriDe conditions, the permutation metrics and all 12 paired medical transfers before checking the bootstrap intervals. A missing file or numerical discrepancy stops the run. The archive is read-only; only the selected output directory is created.

Medical logit/truncation traces and some resolved tokenizer commits were not recorded in the original runs. The validation reports those limitations; it cannot reconstruct missing provenance. The literal MMLU `None`-to-`nan` conversion is retained when replaying the original condition. Its separate sensitivity analysis remains in the bundled results.

## Data for new inference

Datasets and model weights are not redistributed. Medical dataset configurations are pinned to the revisions resolved in the original preprocessing manifests.

| Benchmark | Source | Split and revision |
|---|---|---|
| MMLU, ARC-Challenge, CommonsenseQA | [chujiezheng/LLM-MCQ-Bias](https://github.com/chujiezheng/LLM-MCQ-Bias) | commit `2ae2f40c77006f7a00a4675bdfb30151c2406691` |
| MedQA (4 options) | [openlifescienceai/MedQA-USMLE-4-options-hf](https://huggingface.co/datasets/openlifescienceai/MedQA-USMLE-4-options-hf) | test, `20a8f4d6b851f6391751f6e76c06bc3a26c83e0b` |
| MedMCQA | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | validation, `91c6572c454088bf71b679ad90aa8dffcd0d5868` |

```bash
python analysis/scripts/download_zheng2024.py
(cd analysis && python scripts/download_data.py --dataset medqa)
(cd analysis && python scripts/download_data.py --dataset medmcqa)
(cd analysis && python scripts/preprocess_data.py --dataset medqa)
(cd analysis && python scripts/preprocess_data.py --dataset medmcqa)
```

The reference downloader checks archive SHA-256 `cc097f1a1ffd9b4435c222fe2cff56b45eed2b9bf9a52947b5398162b22ba810`. `src/pride/zheng.py` separately verifies the upstream `debias_utils.py` before loading it. Canonical MMLU inference uses this reference dataset, not the legacy Hugging Face MMLU adapter.

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

Llama and Gemma are gated: accept their licenses on the Hub and log in with `hf auth login`. Never put access tokens into job scripts. Model weights are not part of this repository.

## New GPU runs

Run the canonical inference command inside an allocated Slurm GPU job, from `analysis/`:

```bash
python scripts/reproduce_pride.py --model qwen2.5-7b-instruct
```

To submit all six models, use `python scripts/start_pride_reproduction.py --submit --partition YOUR_GPU_PARTITION` after configuring your cluster environment. The setup scripts and LRZ-specific prerequisites are described in [the cluster guide](analysis/scripts/slurm/README.md). Replace cluster account/path placeholders before submitting jobs. The earlier generic pipeline in that guide is historical; the thesis uses `reproduce_pride.py` and protocol `zheng-pride-0shot-v1`.

The canonical runner uses raw-text zero-shot prompts, complete cyclic option shifts and first-token scores, with five-draw hybrid PriDe at 5% calibration. It writes observations, priors, calibration memberships and summary hashes under `analysis/outputs/reproductions/`. Its `--postprocess-only` option reuses compatible completed observations without inference. A new run is a separate experiment with its own runtime metadata; it must not overwrite the submitted results or be substituted into the historical archive without validation.

## Repository contents and historical scripts

| Path | Purpose |
|---|---|
| `analysis/src/` | Scientific implementation and release validation |
| `analysis/configs/` | Model, dataset, prompt and preprocessing configuration |
| `analysis/tests/` | Unit, reference, archive and integration tests |
| `analysis/outputs/` | Derived scientific results used by the submitted tables and plots |
| `Paper/generated/` | The 28 submitted LaTeX tables |
| `Paper/figures/` | Figure generator and full-precision plotting snapshot |
| `analysis/thesis_tables_static/` | Two hand-maintained tables |
| `provenance/` | Redaction, technical revision and scientific-input checksums |

The thesis PDF, manuscript sources, raw observations, model weights and dataset copies are not included. The generated tables match the `Paper` manuscript version. A manuscript edited separately must be checked against these tables before submission.

Earlier scripts such as `analyze_phase3b_offline.py`, `finalize_phase4_statistics.py`, `run_phase6_robustness.py`, `strengthen_thesis_offline.py` and the manuscript-delivery validators are retained as the executed workflow record. They depend on the full original archive, including intermediate caches, preservation baselines, editorial files and original paths. They are not fresh-checkout entry points. `reproduce_results.py` provides the portable submission and raw-replay commands above without those editorial dependencies. Supplementary sensitivity results are included as derived evidence; generating new sensitivity experiments requires the corresponding original plans and score caches.

## Provenance and table maintenance

`provenance/results_manifest.json` records the submitted scientific input hashes. `provenance/redactions.json` records executed and initially published hashes for files whose local account/path strings were anonymized. `provenance/technical_revision.json` records the subsequent portability fixes, including the baseline and current hashes; historical execution hashes are retained. The inference and numerical calculation functions were not changed by these release fixes.

`Paper/reproducibility_manifest.json` describes the original thesis project and its historical paths. Some referenced editorial files and PDF hashes belong to that archive and are not release inputs.

Table regeneration runs, in order: `build_phase4_thesis_tables.py`, `restructure_rq1_presentation.py`, `build_strengthening_tables.py`, `build_phase7_robustness_assets.py`, then `finalize_thesis_tables.py`. The last stage applies recorded structural and caption revisions plus the two static tables. Use the wrapper so that the generators cannot overwrite the submitted captions. Future caption edits must also be entered in `analysis/scripts/thesis_table_text_revisions.json` and pass `./reproduce_tables.sh`.

## Reference

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou and Minlie Huang. 2024. Large Language Models Are Not Robust Multiple Choice Selectors. *ICLR 2024*.
