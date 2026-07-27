#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v31-encoders
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail

ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
MANIFEST=${MANIFEST:-/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3_clean_058779b7/natural_validation_retrieval_manifest.jsonl}
CURRENT_CHECKPOINT=${CURRENT_CHECKPOINT:-/mnt/weka/svardanyan/rs_change_project/runs/qcpr_v3_grounding_smoke_9c8e91fc/global/last.pt}
REMOTECLIP_CHECKPOINT=${REMOTECLIP_CHECKPOINT:-/mnt/weka/svardanyan/rs_change_project/archive/removed_models/remoteclip_20260628_015151/RemoteCLIP-ViT-B-32.pt}
SIGLIP2_MODEL=${SIGLIP2_MODEL:-/mnt/weka/svardanyan/rs_change_project/models/siglip2-base-patch16-256}
GEORSCLIP_CHECKPOINT=${GEORSCLIP_CHECKPOINT:-/mnt/weka/svardanyan/rs_change_project/models/georsclip-vit-b-32/ckpt/RS5M_ViT-B-32.pt}

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"

ARGS=(
  --manifest "$MANIFEST"
  --current-checkpoint "$CURRENT_CHECKPOINT"
  --remoteclip-checkpoint "$REMOTECLIP_CHECKPOINT"
  --siglip2-model "$SIGLIP2_MODEL"
  --georsclip-checkpoint "$GEORSCLIP_CHECKPOINT"
  --georsclip-model-name "${GEORSCLIP_MODEL_NAME:-ViT-B-32}"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-16}"
)
"$PYTHON" scripts/evaluate_qcpr_v31_encoders.py "${ARGS[@]}"
