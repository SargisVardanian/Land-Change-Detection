#!/usr/bin/env bash
# Guarded launcher for the direct TemporalSigLIP experiment.
# It only submits when the caller explicitly supplies all immutable inputs.

set -eu

: "${PHASE:?PHASE must be A or B}"
: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"

STEPS=${STEPS:-512}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-32}
LOGICAL_PHYSICAL_BATCH_SIZE=${LOGICAL_PHYSICAL_BATCH_SIZE:-128}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
SEED=${SEED:-20260807}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}

if test "$PHASE" = A; then
  test "$STEPS" -eq 512 || test "${QCPR_ALLOW_NONSTANDARD_STEPS:-0}" = 1
else
  test "$PHASE" = B
  test "$STEPS" -eq 1536 || test "${QCPR_ALLOW_NONSTANDARD_STEPS:-0}" = 1
fi
if test "$STEPS" -gt 32; then
  test "${AUTHORIZE_LONG_RUN:-0}" = 1
  test "${QCPR_ALLOW_LONG_TRAINING:-0}" = 1
fi

test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -d "$DATA_RELEASE"
test -s "$TRAIN_MANIFEST"
test -s "$DEVELOPMENT_MANIFEST"
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"
if test "$PHASE" = B; then
  : "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required for Stage B}"
  test -s "$CHECKPOINT_PATH"
else
  CHECKPOINT_PATH=none
fi

mkdir -p "$RUN_ROOT"
export PHASE EXPECTED_SHA RUN_ROOT WORKTREE PYTHON SIGLIP2_MODEL DATA_RELEASE
export TRAIN_MANIFEST DEVELOPMENT_MANIFEST CHECKPOINT_PATH STEPS
export PHYSICAL_BATCH_SIZE LOGICAL_PHYSICAL_BATCH_SIZE CAPTIONS_PER_PAIR SEED
export AUTHORIZE_LONG_RUN QCPR_ALLOW_LONG_TRAINING QCPR_ALLOW_NONSTANDARD_STEPS

sbatch \
  --export=ALL \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --account="$ACCOUNT" \
  --gres="$GPU_GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name="qcpr-temporal-siglip-${PHASE}" \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu
cd \"\$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"\$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
test -d \"\$DATA_RELEASE\"
test -s \"\$TRAIN_MANIFEST\"
test -s \"\$DEVELOPMENT_MANIFEST\"
test -d \"\$SIGLIP2_MODEL\"
test -s \"\$SIGLIP2_MODEL/model.safetensors\"
if test \"\$PHASE\" = B; then test -s \"\$CHECKPOINT_PATH\"; fi
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/run_temporal_siglip.py\" \\
  --phase \"\$PHASE\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --train-manifest \"\$TRAIN_MANIFEST\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --steps \"\$STEPS\" \\
  --physical-batch-size \"\$PHYSICAL_BATCH_SIZE\" \\
  --logical-physical-batch-size \"\$LOGICAL_PHYSICAL_BATCH_SIZE\" \\
  --captions-per-pair \"\$CAPTIONS_PER_PAIR\" \\
  --seed \"\$SEED\" \\
  --initial-checkpoint \"\$CHECKPOINT_PATH\" \\
  --authorize-long-run"
