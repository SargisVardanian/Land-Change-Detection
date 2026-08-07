#!/usr/bin/env bash
# Submit one explicitly authorized SigLIP-2 Phase-A or Phase-B run.
# This launcher is intentionally not invoked by the current audit task.

set -eu

: "${PHASE:?PHASE is required (A or B)}"
: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required; use none for Phase A}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"

STEPS=${STEPS:-256}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-32}
LOGICAL_PHYSICAL_BATCH_SIZE=${LOGICAL_PHYSICAL_BATCH_SIZE:-128}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
SEED=${SEED:-20260805}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
JOB_NAME=${JOB_NAME:-qcpr-siglip2-phase-${PHASE}}
EXCLUDE_NODES=${EXCLUDE_NODES:-}

case "$PHASE" in
  A) test "$STEPS" -eq 256 || test "${QCPR_ALLOW_NONSTANDARD_STEPS:-0}" = 1 ;;
  B) test "$STEPS" -eq 1536 || test "${QCPR_ALLOW_NONSTANDARD_STEPS:-0}" = 1 ;;
  *) echo "PHASE must be A or B" >&2; exit 2 ;;
esac

if test "$STEPS" -gt 32; then
  test "${AUTHORIZE_LONG_RUN:-0}" = 1
  test "${QCPR_ALLOW_LONG_TRAINING:-0}" = 1
fi

test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -s "$CONFIG_PATH"
test -d "$DATA_RELEASE"
test -s "$TRAIN_MANIFEST"
test -s "$DEVELOPMENT_MANIFEST"
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"
if test "$PHASE" = B; then
  test "$CHECKPOINT_PATH" != none
  test -s "$CHECKPOINT_PATH"
else
  test "$CHECKPOINT_PATH" = none
fi

mkdir -p "$RUN_ROOT"
export PHASE EXPECTED_SHA RUN_ROOT CONFIG_PATH DATA_RELEASE CHECKPOINT_PATH
export WORKTREE PYTHON SIGLIP2_MODEL TRAIN_MANIFEST DEVELOPMENT_MANIFEST
export STEPS PHYSICAL_BATCH_SIZE LOGICAL_PHYSICAL_BATCH_SIZE CAPTIONS_PER_PAIR SEED
export AUTHORIZE_LONG_RUN QCPR_ALLOW_LONG_TRAINING QCPR_ALLOW_NONSTANDARD_STEPS

NODE_ARGS=()
if test -n "$EXCLUDE_NODES"; then
  NODE_ARGS+=(--exclude="$EXCLUDE_NODES")
fi

sbatch \
  "${NODE_ARGS[@]}" \
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
test -s \"\$CONFIG_PATH\"
test -d \"\$DATA_RELEASE\"
test -s \"\$TRAIN_MANIFEST\"
test -s \"\$DEVELOPMENT_MANIFEST\"
test -d \"\$SIGLIP2_MODEL\"
if test \"\$PHASE\" = B; then test -s \"\$CHECKPOINT_PATH\"; fi
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/run_qcpr_siglip2_phase.py\" \\
  --phase \"\$PHASE\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --train-manifest \"\$TRAIN_MANIFEST\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --config-path \"\$CONFIG_PATH\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --steps \"\$STEPS\" \\
  --physical-batch-size \"\$PHYSICAL_BATCH_SIZE\" \\
  --logical-physical-batch-size \"\$LOGICAL_PHYSICAL_BATCH_SIZE\" \\
  --captions-per-pair \"\$CAPTIONS_PER_PAIR\" \\
  --seed \"\$SEED\" \\
  --initial-checkpoint \"\$CHECKPOINT_PATH\" \\
  --authorize-long-run"
