#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-mask-micro
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?}
OUTPUT_DIR=${OUTPUT_DIR:?}
V3_CHECKPOINT=${V3_CHECKPOINT:?}
DERIVED_MANIFEST_DIR=${DERIVED_MANIFEST_DIR:-/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3}
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
STEPS=${STEPS:-30}
MICRO_TRAIN_SAMPLES=${MICRO_TRAIN_SAMPLES:-8}
VALIDATION_PROBE_SAMPLES=${VALIDATION_PROBE_SAMPLES:-16}
TRAIN_PAIR_ID=${TRAIN_PAIR_ID:-}
VALIDATION_PAIR_ID=${VALIDATION_PAIR_ID:-}
DIRECTION_ONLY_PROBE_CAPTIONS=${DIRECTION_ONLY_PROBE_CAPTIONS:-0}
GROUNDING_BACKBONE=${GROUNDING_BACKBONE:-universat}
SIGLIP2_MODEL=${SIGLIP2_MODEL:-/mnt/weka/svardanyan/rs_change_project/models/siglip2-base-patch16-256}
case "$STEPS" in 20|30|40|50) ;; *) echo "mask diagnostic steps must be 20, 30, 40, or 50" >&2; exit 2;; esac
PAIR_ARGS=()
if [[ -n "$TRAIN_PAIR_ID" ]]; then PAIR_ARGS+=(--train-probe-pair-id "$TRAIN_PAIR_ID"); fi
if [[ -n "$VALIDATION_PAIR_ID" ]]; then PAIR_ARGS+=(--validation-probe-pair-id "$VALIDATION_PAIR_ID"); fi
if [[ "$DIRECTION_ONLY_PROBE_CAPTIONS" = 1 ]]; then PAIR_ARGS+=(--direction-only-probe-captions); fi
"$PYTHON" scripts/train_qcpr_v3.py \
  --output-dir "$OUTPUT_DIR" \
  --phase mask_only_diagnostic \
  --mask-objective A \
  --initialization-mode clean_pretrained \
  --derived-manifest-dir "$DERIVED_MANIFEST_DIR" \
  --v3-checkpoint "$V3_CHECKPOINT" \
  --steps "$STEPS" \
  --micro-train-samples "$MICRO_TRAIN_SAMPLES" \
  --validation-probe-samples "$VALIDATION_PROBE_SAMPLES" \
  --probe-interval 5 \
  --batch-size 8 \
  --seed 20260716 \
  --grounding-backbone "$GROUNDING_BACKBONE" \
  --siglip2-model "$SIGLIP2_MODEL" \
  "${PAIR_ARGS[@]}"
