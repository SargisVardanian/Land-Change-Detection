#!/usr/bin/env bash
set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
: "${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${PYTHON:?PYTHON is required}"
: "${TRAIN_MANIFEST:?TRAIN_MANIFEST is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${UNISAT_SOURCE:?UNISAT_SOURCE is required}"
: "${UNISAT_CHECKPOINT:?UNISAT_CHECKPOINT is required}"
: "${JINA_MODEL:?JINA_MODEL is required}"

STEPS=${STEPS:-8}
PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-8}
CAPTIONS_PER_PAIR=${CAPTIONS_PER_PAIR:-2}
JOB_NAME=${JOB_NAME:-qcpr-v3-real-smoke}
SLURM_PARTITION=${SLURM_PARTITION:-defq}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEMORY=${MEMORY:-64G}
TIME_LIMIT=${TIME_LIMIT:-00:15:00}

test "${STEPS}" -ge 1
test "${STEPS}" -le 32
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -f "$DATA_RELEASE/dataset_v2_stage2_release.json"
test -f "$CONFIG_PATH"
test -f "$CHECKPOINT_PATH"
test -f "$TRAIN_MANIFEST"
test -f "$DEVELOPMENT_MANIFEST"
test -d "$UNISAT_SOURCE"
test -f "$UNISAT_CHECKPOINT/model.safetensors"
test -f "$JINA_MODEL/model.safetensors"
test "$(sha256sum "$CHECKPOINT_PATH" | awk '{print $1}')" = "$CHECKPOINT_SHA256"

mkdir -p "$RUN_ROOT"

sbatch \
  --partition="$SLURM_PARTITION" \
  --gres="$GPU_GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name="$JOB_NAME" \
  --output="$RUN_ROOT/slurm-%j.out" \
  --wrap="set -eu
cd \"\$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"\$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
test -f \"\$DATA_RELEASE/dataset_v2_stage2_release.json\"
test -f \"\$CONFIG_PATH\"
test -f \"\$CHECKPOINT_PATH\"
test \"\$(sha256sum \"\$CHECKPOINT_PATH\" | awk '{print \$1}')\" = \"\$CHECKPOINT_SHA256\"
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/run_qcpr_v3_real_integration_smoke.py\" \
  --data-release \"\$DATA_RELEASE\" \
  --train-manifest \"\$TRAIN_MANIFEST\" \
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \
  --unisat-source \"\$UNISAT_SOURCE\" \
  --unisat-checkpoint \"\$UNISAT_CHECKPOINT\" \
  --jina-model \"\$JINA_MODEL\" \
  --expected-code-sha \"\$EXPECTED_SHA\" \
  --output-dir \"\$RUN_ROOT\" \
  --checkpoint-path \"\$CHECKPOINT_PATH\" \
  --config-path \"\$CONFIG_PATH\" \
  --steps \"\$STEPS\" \
  --physical-batch-size \"\$PHYSICAL_BATCH_SIZE\" \
  --captions-per-pair \"\$CAPTIONS_PER_PAIR\""
