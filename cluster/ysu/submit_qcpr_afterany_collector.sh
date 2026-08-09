#!/usr/bin/env bash
set -eu

: "${UPSTREAM_JOB_ID:?UPSTREAM_JOB_ID is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${WORKTREE:?WORKTREE is required}"

PYTHON_BIN="${QCPR_PYTHON:-/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python}"
SACCT_BIN="${SACCT_BIN:-/opt/slurm/bin/sacct}"
ACCOUNT="${ACCOUNT:-research}"
QOS="${QOS:-researcher}"
PARTITION="${PARTITION:-research}"

test -d "$RUN_ROOT"
test -d "$WORKTREE"

sbatch --parsable \
  --dependency="afterany:$UPSTREAM_JOB_ID" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:05:00 \
  --job-name=qcpr-collect \
  --output="$RUN_ROOT/collector-%j.out" \
  --error="$RUN_ROOT/collector-%j.err" \
  --wrap="set -eu; cd \"$WORKTREE\"; test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"; test -z \"\$(git status --porcelain)\"; export PYTHONPATH=src:scripts; exec \"$PYTHON_BIN\" scripts/collect_qcpr_slurm_completion.py --job-id \"$UPSTREAM_JOB_ID\" --run-root \"$RUN_ROOT\" --expected-code-sha \"$EXPECTED_SHA\" --sacct-bin \"$SACCT_BIN\""
