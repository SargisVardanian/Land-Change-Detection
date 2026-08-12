#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"

PYTHON_BIN="${QCPR_PYTHON:-/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python}"
ACCOUNT="${ACCOUNT:-research}"
QOS="${QOS:-researcher}"
PARTITION="${PARTITION:-research}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
MAX_PAIRS_PER_SOURCE="${MAX_PAIRS_PER_SOURCE:-32}"
BUDGETS="${BUDGETS:-256 576 1024}"
SEED="${SEED:-20260809}"

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT WORKTREE CONFIG_PATH DATA_RELEASE
export DEVELOPMENT_MANIFEST CHECKPOINT_PATH SIGLIP2_MODEL PYTHON_BIN
export MAX_PAIRS_PER_SOURCE BUDGETS SEED

JOB_ID="$(sbatch --parsable \
  --job-name=qcpr-src-audit \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres="${GRES:-gpu:h100:1}" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="${TIME_LIMIT:-00:20:00}" \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu; cd \"$WORKTREE\"; test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"; test -z \"\$(git status --porcelain)\"; test -f \"$DATA_RELEASE/SHA256SUMS\"; test -f \"$CHECKPOINT_PATH\"; export PYTHONPATH=src:scripts; exec \"$PYTHON_BIN\" scripts/audit_qcpr_siglip2_source_shortcut.py --siglip2-model \"$SIGLIP2_MODEL\" --data-release \"$DATA_RELEASE\" --development-manifest \"$DEVELOPMENT_MANIFEST\" --config-path \"$CONFIG_PATH\" --checkpoint \"$CHECKPOINT_PATH\" --output-dir \"$RUN_ROOT\" --expected-code-sha \"$EXPECTED_SHA\" --max-pairs-per-source \"$MAX_PAIRS_PER_SOURCE\" --budgets $BUDGETS --seed \"$SEED\"")"
printf '%s\n' "$JOB_ID"
