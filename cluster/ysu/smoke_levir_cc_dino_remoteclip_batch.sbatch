#!/bin/bash
#SBATCH --job-name=smoke_levir_cc_dino_remoteclip
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
export PROJECT_DIR="$RS_PROJECT_ROOT/code/project"
export PYTHON="/mnt/weka/shared-cache/miniforge3/bin/python"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="$RS_PROJECT_ROOT/.cache/huggingface"

cd "$PROJECT_DIR"
mkdir -p logs "$RS_PROJECT_ROOT/runs" "$RS_PROJECT_ROOT/indexes" "$HF_HOME"

"$PYTHON" scripts/smoke_levir_cc_dino_remoteclip_batch.py \
  --manifest "$RS_PROJECT_ROOT/indexes/levir_cc_pair_manifest.jsonl" \
  --output "$RS_PROJECT_ROOT/runs/levir_cc_dino_remoteclip_smoke_report.json" \
  --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
  --remoteclip-model-path "${REMOTECLIP_MODEL_PATH:-$RS_PROJECT_ROOT/models/remoteclip-rn50}" \
  --pair-feature-mode "${PAIR_FEATURE_MODE:-signed_delta}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --image-size "${IMAGE_SIZE:-224}" \
  --device auto
