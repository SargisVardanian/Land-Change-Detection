#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v31-m0-memory
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

ROOT=${ROOT:-/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-aligned}
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
DERIVED_MANIFEST_DIR=${DERIVED_MANIFEST_DIR:?derived M0 split is required}

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$OUTPUT_ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"

"$PYTHON" scripts/train_qcpr_v3.py \
  --output-dir "$OUTPUT_ROOT" \
  --phase mask_only_diagnostic \
  --mask-diagnostic-mode multi_pair \
  --initialization-mode clean_pretrained \
  --derived-manifest-dir "$DERIVED_MANIFEST_DIR" \
  --steps 3 \
  --batch-size 32 \
  --skip-development-evaluation \
  --direction-only-probe-captions \
  --grounding-backbone universat \
  --seed 20260718
