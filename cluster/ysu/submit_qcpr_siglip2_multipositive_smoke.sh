#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?}"
: "${RUN_ROOT:?}"
: "${WORKTREE:?}"
: "${SIGLIP2_MODEL:?}"
: "${TRAIN_MANIFEST:?}"
: "${MULTIPOSITIVE_ARTIFACT:?}"

PYTHON_BIN=${QCPR_PYTHON:-/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python}
mkdir -p "$RUN_ROOT"
JOB_ID=$(sbatch --parsable \
  --account="${ACCOUNT:-research}" --partition="${PARTITION:-research}" \
  --qos="${QOS:-researcher}" --gres="${GRES:-gpu:h100:1}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" --mem="${MEMORY:-96G}" \
  --time="${TIME_LIMIT:-00:30:00}" --job-name=qcpr-multipos \
  --output="$RUN_ROOT/slurm-%j.out" --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu; cd \"$WORKTREE\"; test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"; test -z \"\$(git status --porcelain)\"; test -s \"$MULTIPOSITIVE_ARTIFACT\"; export PYTHONPATH=src:scripts HF_HUB_OFFLINE=1; exec \"$PYTHON_BIN\" scripts/run_qcpr_siglip2_real_smoke.py --siglip2-model \"$SIGLIP2_MODEL\" --train-manifest \"$TRAIN_MANIFEST\" --multipositive-artifact \"$MULTIPOSITIVE_ARTIFACT\" --output-dir \"$RUN_ROOT\" --expected-code-sha \"$EXPECTED_SHA\" --steps 1 --physical-batch-size 128 --captions-per-pair 2 --max-num-patches 256")
printf '%s\n' "$JOB_ID"
