#!/usr/bin/env bash
#SBATCH --job-name=qcpr-s2-b1-full-eval
#SBATCH --partition=research
#SBATCH --account=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/runs/qcpr_stage2_b1_full_common_eval_e43cd4b_20260804-%j.out

set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project
WT=$ROOT/code/project-qcpr-dataset-v2-stage2
PY=$ROOT/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA to immutable code SHA}
RUN_ROOT=$ROOT/runs/qcpr_stage2_b1_full_common_eval_e43cd4b_20260804
MANIFEST=$ROOT/runs/qcpr_stage2_review_system_a772460_20260803/common_frozen_evaluation/common_exact_development.jsonl
CHECKPOINT=$ROOT/manifests/qcpr_dataset_v2_stage2_audit/architecture_screening/corrected-b1-63d9004a38df96a12a7e1c5159ff4dd35a2555c3-206222/B1_screen_checkpoint.pt
FEATURE_CACHE=$RUN_ROOT/common_b1_features.pt
EXPECTED_MANIFEST_SHA=89a69e144033957ce9c0756502b12c6e56a90f62801e7152d0001bfb339c30a9
EXPECTED_CHECKPOINT_SHA=3b6d89e6c37c25302f8af8f57e3f873564a301070ce47f96a8f611caeffc6e2b

cd "$WT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test -s "$MANIFEST"
test -s "$CHECKPOINT"
test ! -e "$RUN_ROOT"
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
CHECKPOINT_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
test "$MANIFEST_SHA" = "$EXPECTED_MANIFEST_SHA"
test "$CHECKPOINT_SHA" = "$EXPECTED_CHECKPOINT_SHA"
export PYTHONPATH="$WT/src:$WT/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
exec "$PY" "$WT/scripts/evaluate_qcpr_stage2_b1_common.py" \
  --development-manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT" \
  --feature-cache "$FEATURE_CACHE" \
  --universat-source "$ROOT/external/UniverSat" \
  --universat-checkpoint "$ROOT/models/universat-base" \
  --jina-model "$ROOT/models/jina-v5-text-small-retrieval" \
  --image-size 256 \
  --extract-batch-size 8 \
  --text-batch 64
