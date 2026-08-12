#!/usr/bin/env bash
# Submit the bounded 256-step GeoRSCLIP temporal-head baseline.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${GEORSCLIP_CHECKPOINT:?GEORSCLIP_CHECKPOINT is required}"
: "${GEORSCLIP_REVISION:?GEORSCLIP_REVISION is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"

STEPS=${STEPS:-256}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-32}
LOGICAL_PHYSICAL_BATCH_SIZE=${LOGICAL_PHYSICAL_BATCH_SIZE:-128}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
SEED=${SEED:-20260805}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-08:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
JOB_NAME=${JOB_NAME:-qcpr-georsclip-temporal-256}

test "$STEPS" -eq 256
test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -d "$DATA_RELEASE"
test -s "$TRAIN_MANIFEST"
test -s "$DEVELOPMENT_MANIFEST"
test -s "$GEORSCLIP_CHECKPOINT"
test "$((LOGICAL_PHYSICAL_BATCH_SIZE % PHYSICAL_BATCH_SIZE))" -eq 0

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT DATA_RELEASE TRAIN_MANIFEST DEVELOPMENT_MANIFEST
export GEORSCLIP_CHECKPOINT GEORSCLIP_REVISION WORKTREE PYTHON
export STEPS PHYSICAL_BATCH_SIZE LOGICAL_PHYSICAL_BATCH_SIZE CAPTIONS_PER_PAIR SEED

sbatch \
  --export=ALL \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --account="$ACCOUNT" \
  --gres="$GPU_GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name="$JOB_NAME" \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu
cd \"\$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"\$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
test -d \"\$DATA_RELEASE\"
test -s \"\$TRAIN_MANIFEST\"
test -s \"\$DEVELOPMENT_MANIFEST\"
test -s \"\$GEORSCLIP_CHECKPOINT\"
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/run_qcpr_georsclip_temporal_train.py\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --train-manifest \"\$TRAIN_MANIFEST\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --georsclip-checkpoint \"\$GEORSCLIP_CHECKPOINT\" \\
  --georsclip-revision \"\$GEORSCLIP_REVISION\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --worktree \"\$WORKTREE\" \\
  --steps \"\$STEPS\" \\
  --physical-batch-size \"\$PHYSICAL_BATCH_SIZE\" \\
  --logical-physical-batch-size \"\$LOGICAL_PHYSICAL_BATCH_SIZE\" \\
  --captions-per-pair \"\$CAPTIONS_PER_PAIR\" \\
  --seed \"\$SEED\" \\
  --authorize-256-step-run"
