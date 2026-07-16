#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-rc1-memory
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
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
cd "$ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
PHASE=${PHASE:-global_bootstrap}
V3_CHECKPOINT=${V3_CHECKPOINT:-}
DERIVED_MANIFEST_DIR=${DERIVED_MANIFEST_DIR:-/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3}
for batch in 8 16 20 24; do
  ARGS=(--output-dir "$OUTPUT_ROOT/bs${batch}" --phase "$PHASE" --initialization-mode clean_pretrained --derived-manifest-dir "$DERIVED_MANIFEST_DIR" --steps 3 --batch-size "$batch")
  if [[ "$PHASE" != "global_bootstrap" ]]; then
    test -n "$V3_CHECKPOINT"
    ARGS+=(--v3-checkpoint "$V3_CHECKPOINT")
  fi
  "$PYTHON" scripts/train_qcpr_v3.py "${ARGS[@]}"
done
