#!/bin/bash
#SBATCH --job-name=mcq-review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with Slurm}"
: "${REVIEW_ROOT:?Set the isolated uploaded release directory}"
: "${MODEL:?Select one pinned model}"
base=/dss/dsshome1/<group>/<lrz-user>/analysis
export HF_HOME="$base/outputs/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MCQ_REVIEW_WEIGHT_CACHE="$REVIEW_ROOT/work_cache"
py="$base/.micromamba/envs/mcq-analysis/bin/python"
cd "$REVIEW_ROOT"
plan=analysis/outputs/thesis_strengthening_20261021/inference_plan.json
result=analysis/outputs/thesis_strengthening_20261021/gpu_results
mkdir -p "$result/$MODEL"
if [[ -d "$result/$MODEL/mmlu" && ! -f "$result/$MODEL/mmlu/manifest.json" ]]; then
    mv "$result/$MODEL/mmlu" "$result/$MODEL/mmlu_initialization_failed_before_$SLURM_JOB_ID"
fi
if [[ ! -f "$result/$MODEL/mmlu/manifest.json" ]]; then
"$py" analysis/scripts/run_thesis_sensitivity.py execute --plan "$plan" --model "$MODEL" \
    --output "$result/$MODEL/mmlu" --conditions mmlu_literal mmlu_cached_control mmlu_unchanged_control
fi
if [[ ! -f "$result/$MODEL/mmlu_summary.json" ]]; then
"$py" analysis/scripts/run_thesis_sensitivity.py summarize --plan "$plan" \
    --run "$result/$MODEL/mmlu" --output "$result/$MODEL/mmlu_summary.json"
fi
if [[ ! -f "$result/$MODEL/chat/manifest.json" ]]; then
"$py" analysis/scripts/run_thesis_sensitivity.py execute --plan "$plan" --model "$MODEL" \
    --output "$result/$MODEL/chat" --conditions medical_native_chat
fi
if [[ ! -f "$result/$MODEL/chat_summary.json" ]]; then
"$py" analysis/scripts/run_thesis_sensitivity.py summarize --plan "$plan" \
    --run "$result/$MODEL/chat" --output "$result/$MODEL/chat_summary.json"
fi
