#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-controlled
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?}; OUTPUT_DIR=${OUTPUT_DIR:?}; PHASE=${PHASE:?}; STEPS=${STEPS:?}; BATCH_SIZE=${BATCH_SIZE:?}
case "$PHASE" in late_interaction|mask_grounding) ;; *) echo "invalid controlled phase" >&2; exit 2;; esac
case "$STEPS" in 100|500) ;; *) echo "controlled steps must be 100 or 500" >&2; exit 2;; esac
cd "$ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
FAST_PAIRS=${FAST_PAIRS:-64}
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
"$PYTHON" scripts/train_qcpr_v3.py --output-dir "$OUTPUT_DIR" --phase "$PHASE" --steps "$STEPS" --batch-size "$BATCH_SIZE"
"$PYTHON" scripts/evaluate_qcpr_v3_fast.py --checkpoint "$OUTPUT_DIR/last.pt" --phase "$PHASE" --training-report "$OUTPUT_DIR/smoke_report.json" --output "$OUTPUT_DIR/fast_validation.json" --batch-size "$BATCH_SIZE" --fast-pairs "$FAST_PAIRS"
