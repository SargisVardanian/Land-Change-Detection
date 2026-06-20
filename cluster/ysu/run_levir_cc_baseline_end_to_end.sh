#!/bin/bash
set -euo pipefail
set -x

if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
else
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/data/$USER/rs_change_project}"
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$RS_PROJECT_ROOT}"
export PROJECT_DIR="${PROJECT_DIR:-${PROJECT_ROOT}/code/project}"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/logs" "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

bash cluster/ysu/setup_rschange_env.sh

"$PYTHON" scripts/plan_change_retrieval_downloads.py \
  --project-root "$PROJECT_ROOT" \
  --phase baseline \
  --reserve-gb "${RESERVE_GB:-120}" \
  --max-download-gb "${MAX_DOWNLOAD_GB:-50}" \
  --output-json "$PROJECT_ROOT/reports/download_plan.json"

bash cluster/ysu/download_change_retrieval_datasets.sh
bash cluster/ysu/bootstrap_change_retrieval_assets.sh
bash cluster/ysu/verify_project_assets.sh

"$PYTHON" scripts/build_levir_cc_pair_manifest.py \
  --root "$PROJECT_ROOT/datasets/raw/LEVIR-CC" \
  --output "$PROJECT_ROOT/indexes/levir_cc_pair_manifest.jsonl"

bash cluster/ysu/submit_levir_cc_baseline.sh

cat <<EOF

Started LEVIR-CC end-to-end baseline bootstrap.

Next:
  PRESET=dinov2_signed_delta RUN_NAME=dinov2_signed_delta bash cluster/ysu/submit_levir_cc_baseline.sh
  PRESET=dinov2_change_fusion RUN_NAME=dinov2_change_fusion bash cluster/ysu/submit_levir_cc_baseline.sh

Watch:
  squeue -u $USER
  tail -f logs/*.out
  tail -f logs/*.err

EOF
