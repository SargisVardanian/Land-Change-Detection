#!/bin/bash
#SBATCH --job-name=qcpr-s2-arch-screen
#SBATCH --partition=research
#SBATCH --account=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_dataset_v2_stage2_audit/architecture_screening/slurm-%j.out

set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project
WT=$ROOT/code/project-qcpr-dataset-v2-stage2
PY=$ROOT/envs/rschange/bin/python
AUDIT=$ROOT/manifests/qcpr_dataset_v2_stage2_audit
CHECKPOINT=$ROOT/runs/qcpr_v3_grounding_smoke_9c8e91fc/global/last.pt
TRAIN=$ROOT/manifests/qcpr_dataset_v2_expanded_benchmark_157d292_20260728/manifests/retrieval_exact_train_v2.jsonl
DEV=$ROOT/manifests/qcpr_dataset_v2_expanded_benchmark_157d292_20260728/manifests/retrieval_exact_development_v2.jsonl
test -n "$EXPECTED_SHA"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$CHECKPOINT"
test -s "$TRAIN"
test -s "$DEV"
export PYTHONPATH="$WT/src:$WT/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$AUDIT/architecture_screening"
exec "$PY" "$WT/scripts/screen_qcpr_stage2_compatible_architectures.py" \
  --current-checkpoint "$CHECKPOINT" \
  --train-manifest "$TRAIN" \
  --development-manifest "$DEV" \
  --output-dir "$AUDIT/architecture_screening/compatible_screen-$EXPECTED_SHA-%j" \
  --image-size 256 \
  --extract-batch-size 8 \
  --microbatch 16 \
  --captions-per-pair 2 \
  --max-train-pairs 2048 \
  --max-development-pairs 512 \
  --steps 128 \
  --seed 20260802
