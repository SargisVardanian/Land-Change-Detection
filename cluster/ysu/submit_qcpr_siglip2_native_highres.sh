#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?}"
: "${RUN_ROOT:?}"
: "${WORKTREE:?}"
: "${CONFIG_PATH:?}"
: "${DATA_RELEASE:?}"
: "${PHYSICAL_REGISTRY:?}"
: "${CHECKPOINT_PATH:?}"
: "${SIGLIP2_MODEL:?}"

PYTHON_BIN=${QCPR_PYTHON:-/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python}
mkdir -p "$RUN_ROOT"
JOB_ID=$(sbatch --parsable \
  --account="${ACCOUNT:-research}" --partition="${PARTITION:-research}" \
  --qos="${QOS:-researcher}" --gres="${GRES:-gpu:h100:1}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" --mem="${MEMORY:-96G}" \
  --time="${TIME_LIMIT:-00:30:00}" --job-name=qcpr-native-hires \
  --output="$RUN_ROOT/slurm-%j.out" --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu; cd \"$WORKTREE\"; test \"\$(git rev-parse HEAD)\" = \"$EXPECTED_SHA\"; test -z \"\$(git status --porcelain)\"; export PYTHONPATH=src:scripts; exec \"$PYTHON_BIN\" scripts/audit_qcpr_siglip2_native_highres.py --siglip2-model \"$SIGLIP2_MODEL\" --data-release \"$DATA_RELEASE\" --physical-registry \"$PHYSICAL_REGISTRY\" --config-path \"$CONFIG_PATH\" --checkpoint \"$CHECKPOINT_PATH\" --output-dir \"$RUN_ROOT\" --expected-code-sha \"$EXPECTED_SHA\" --source \"${SOURCE:-s2looking}\" --pair-count \"${PAIR_COUNT:-16}\" --budgets ${BUDGETS:-256 576 1024}")
printf '%s\n' "$JOB_ID"
