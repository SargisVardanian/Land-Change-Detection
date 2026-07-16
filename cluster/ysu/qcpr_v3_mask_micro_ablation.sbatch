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
case "$STEPS" in 20|30|40|50) ;; *) echo "mask diagnostic steps must be 20, 30, 40, or 50" >&2; exit 2;; esac
"$PYTHON" scripts/train_qcpr_v3.py \
  --output-dir "$OUTPUT_DIR" \
  --phase mask_only_diagnostic \
  --mask-objective A \
  --initialization-mode clean_pretrained \
  --derived-manifest-dir "$DERIVED_MANIFEST_DIR" \
  --v3-checkpoint "$V3_CHECKPOINT" \
  --steps "$STEPS" \
  --micro-train-samples 8 \
  --validation-probe-samples 16 \
  --probe-interval 5 \
  --batch-size 8 \
  --seed 20260716
