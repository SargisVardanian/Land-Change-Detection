#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${PYTHON:?PYTHON is required}"

MODE=${MODE:-smoke}
STEPS=${STEPS:-16}
JOB_NAME=${JOB_NAME:-qcpr-v3-${MODE}}
SLURM_PARTITION=${SLURM_PARTITION:-defq}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEMORY=${MEMORY:-64G}
TIME_LIMIT=${TIME_LIMIT:-00:10:00}
mkdir -p "$RUN_ROOT"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -e "$DATA_RELEASE"
test -e "$CONFIG_PATH"
test -e "$CHECKPOINT_PATH"

sbatch --partition="$SLURM_PARTITION" --gres="$GPU_GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEMORY" --time="$TIME_LIMIT" --job-name="$JOB_NAME" --output="$RUN_ROOT/slurm-%j.out" --wrap="set -eu
cd \"$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
export PYTHONPATH=\"$WORKTREE/src:$WORKTREE/scripts\"
exec \"$PYTHON\" \"$WORKTREE/scripts/run_qcpr_v3_smoke.py\" --run-root \"$RUN_ROOT\" --expected-sha \"$EXPECTED_SHA\" --data-release \"$DATA_RELEASE\" --config-path \"$CONFIG_PATH\" --checkpoint-path \"$CHECKPOINT_PATH\" --steps \"$STEPS\""
