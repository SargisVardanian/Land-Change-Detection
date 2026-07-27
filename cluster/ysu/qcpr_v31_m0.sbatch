#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v31-m0
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
DERIVED_MANIFEST_DIR=${DERIVED_MANIFEST_DIR:?derived M0 split is required}
STEPS=${STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
GROUNDING_BACKBONE=${GROUNDING_BACKBONE:-universat}
SIGLIP2_MODEL=${SIGLIP2_MODEL:-/mnt/weka/svardanyan/rs_change_project/models/siglip2-base-patch16-256}
test "$STEPS" = 100
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"

"$PYTHON" scripts/train_qcpr_v3.py \
  --output-dir "$OUTPUT_DIR" \
  --phase mask_only_diagnostic \
  --mask-diagnostic-mode multi_pair \
  --initialization-mode clean_pretrained \
  --derived-manifest-dir "$DERIVED_MANIFEST_DIR" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --m0-eval-interval 10 \
  --direction-only-probe-captions \
  --grounding-backbone "$GROUNDING_BACKBONE" \
  --siglip2-model "$SIGLIP2_MODEL" \
  --seed 20260718

