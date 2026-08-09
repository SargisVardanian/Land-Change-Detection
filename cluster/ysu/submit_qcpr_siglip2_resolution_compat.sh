#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${NAFLEX_MODEL:?NAFLEX_MODEL is required}"
: "${FIXRES_MODEL:?FIXRES_MODEL is required}"

PYTHON_BIN="${QCPR_PYTHON:-/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python}"
ACCOUNT="${ACCOUNT:-research}"
QOS="${QOS:-researcher}"
PARTITION="${PARTITION:-research}"
mkdir -p "$RUN_ROOT"

JOB_ID="$(sbatch --parsable \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres="${GRES:-gpu:h100:1}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEMORY:-96G}" \
  --time="${TIME_LIMIT:-02:00:00}" \
  --job-name=qcpr-res-compat \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu; cd \"$WORKTREE\"; test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"; test -z \"\$(git status --porcelain)\"; test -f \"$DATA_RELEASE/SHA256SUMS\"; test -f \"$CHECKPOINT_PATH\"; export PYTHONPATH=src:scripts; exec \"$PYTHON_BIN\" scripts/compare_qcpr_siglip2_resolution_compat.py --naflex-model \"$NAFLEX_MODEL\" --fixres-model \"$FIXRES_MODEL\" --data-release \"$DATA_RELEASE\" --development-manifest \"$DEVELOPMENT_MANIFEST\" --config-path \"$CONFIG_PATH\" --checkpoint \"$CHECKPOINT_PATH\" --output-dir \"$RUN_ROOT\" --expected-code-sha \"$EXPECTED_SHA\" --max-pairs \"${MAX_PAIRS:-0}\" --image-batch-size \"${IMAGE_BATCH_SIZE:-8}\" --text-batch-size \"${TEXT_BATCH_SIZE:-64}\"")"
printf '%s\n' "$JOB_ID"
