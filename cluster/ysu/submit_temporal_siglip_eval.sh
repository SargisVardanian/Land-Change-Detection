#!/usr/bin/env bash
# Direct full-gallery TemporalSigLIP evaluation; no reranking stage.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
FINAL_HANDOFF_PATH=${FINAL_HANDOFF_PATH:-}

PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-32}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-256}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}

test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -d "$DATA_RELEASE"
test -s "$DEVELOPMENT_MANIFEST"
test -s "$CHECKPOINT_PATH"
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"
test -s "$CONFIG_PATH"
if test -n "$FINAL_HANDOFF_PATH"; then test -s "$FINAL_HANDOFF_PATH"; fi

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT WORKTREE PYTHON SIGLIP2_MODEL CONFIG_PATH DATA_RELEASE DEVELOPMENT_MANIFEST CHECKPOINT_PATH FINAL_HANDOFF_PATH
export PAIR_BATCH_SIZE QUERY_BATCH_SIZE

sbatch \
  --export=ALL \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --account="$ACCOUNT" \
  --gres="$GPU_GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=qcpr-temporal-siglip-eval \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu
cd \"\$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"\$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
test -d \"\$DATA_RELEASE\"
test -s \"\$DEVELOPMENT_MANIFEST\"
test -s \"\$CONFIG_PATH\"
test -s \"\$CHECKPOINT_PATH\"
if test -n \"\$FINAL_HANDOFF_PATH\"; then test -s \"\$FINAL_HANDOFF_PATH\"; fi
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/evaluate_temporal_siglip.py\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --config-path \"\$CONFIG_PATH\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --checkpoint \"\$CHECKPOINT_PATH\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --pair-batch-size \"\$PAIR_BATCH_SIZE\" \\
  --query-batch-size \"\$QUERY_BATCH_SIZE\" \\
  \${FINAL_HANDOFF_PATH:+--final-handoff \"\$FINAL_HANDOFF_PATH\"}"
