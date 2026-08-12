#!/usr/bin/env bash
# One bounded real H100 calibration allocation.  Total steps are capped at 32.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"

STEPS_PER_CANDIDATE=${STEPS_PER_CANDIDATE:-2}
BATCH_SIZES=${BATCH_SIZES:-"8 16 32"}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
SEED=${SEED:-20260805}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-00:30:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
JOB_NAME=${JOB_NAME:-qcpr-siglip2-batch-calibration}

set -- $BATCH_SIZES
test "$#" -gt 0
TOTAL_STEPS=$((STEPS_PER_CANDIDATE * $#))
test "$TOTAL_STEPS" -le 32
test "$STEPS_PER_CANDIDATE" -gt 0
test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -s "$CONFIG_PATH"
test -d "$DATA_RELEASE"
test -s "$TRAIN_MANIFEST"
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT CONFIG_PATH DATA_RELEASE WORKTREE PYTHON SIGLIP2_MODEL
export TRAIN_MANIFEST STEPS_PER_CANDIDATE CAPTIONS_PER_PAIR SEED BATCH_SIZES

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
test -s \"\$CONFIG_PATH\"
test -d \"\$DATA_RELEASE\"
test -s \"\$TRAIN_MANIFEST\"
test -d \"\$SIGLIP2_MODEL\"
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/run_qcpr_siglip2_batch_calibration.py\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --train-manifest \"\$TRAIN_MANIFEST\" \\
  --config-path \"\$CONFIG_PATH\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --batch-sizes \$BATCH_SIZES \\
  --steps-per-candidate \"\$STEPS_PER_CANDIDATE\" \\
  --captions-per-pair \"\$CAPTIONS_PER_PAIR\" \\
  --seed \"\$SEED\""
