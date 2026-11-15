#!/bin/bash
#SBATCH --job-name=mcq-a100-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
set -euo pipefail
: "${REVIEW_ROOT:?}"
base=/dss/dsshome1/<group>/<lrz-user>/analysis
export HF_HOME="$base/outputs/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MCQ_REVIEW_WEIGHT_CACHE="$REVIEW_ROOT/work_cache"
py="$base/.micromamba/envs/mcq-analysis/bin/python"
cd "$REVIEW_ROOT"
plan=analysis/outputs/thesis_strengthening_20261021/inference_plan.json
model=mistral-7b-instruct-v0.3
result=analysis/outputs/thesis_strengthening_20261021/gpu_results/$model
"$py" analysis/scripts/run_thesis_sensitivity.py execute --plan "$plan" --model "$model" \
 --output "$result/native_a100_raw_controls" --conditions mmlu_cached_control mmlu_unchanged_control
"$py" - <<'PY'
import json,numpy as np
from pathlib import Path
p=Path('analysis/outputs/thesis_strengthening_20261021/gpu_results/mistral-7b-instruct-v0.3/native_a100_raw_controls')
c=np.load('analysis/outputs/phase6_robustness_20261018/scores/mistral-7b-instruct-v0.3__mmlu.npz')
index={str(v):j for j,v in enumerate(c['legacy_ids'])}
rows=[json.loads(line) for line in (p/'records.jsonl').open()]
assert len(rows)==12
error=max(float(np.max(np.abs(np.array(row['probs'])-c['probabilities'][index[row['legacy_id']]]))) for row in rows)
assert error <= 2e-5, error
(p/'validation.json').write_text(json.dumps(dict(control_max_abs_error=error,controls=12,validated=True))+'\n')
print('Matched A100 controls:',error,flush=True)
PY
"$py" analysis/scripts/run_thesis_sensitivity.py execute --plan "$plan" --model "$model" \
 --output "$result/chat_a100" --conditions medical_native_chat
"$py" analysis/scripts/run_thesis_sensitivity.py summarize --plan "$plan" \
 --run "$result/chat_a100" --output "$result/chat_a100_summary.json"
