#!/usr/bin/env bash
# Guarded evaluation-only source-shortcut audit for existing checkpoints.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${STAGE_A_CHECKPOINT:?STAGE_A_CHECKPOINT is required}"
: "${STAGE_B_1024_CHECKPOINT:?STAGE_B_1024_CHECKPOINT is required}"
: "${STAGE_B_2048_CHECKPOINT:?STAGE_B_2048_CHECKPOINT is required}"
: "${STAGE_A_RANKING:?STAGE_A_RANKING is required}"
: "${STAGE_B_1024_RANKING:?STAGE_B_1024_RANKING is required}"
: "${STAGE_B_2048_RANKING:?STAGE_B_2048_RANKING is required}"

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
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"
for required in \
  "$STAGE_A_CHECKPOINT" "$STAGE_B_1024_CHECKPOINT" "$STAGE_B_2048_CHECKPOINT" \
  "$STAGE_A_RANKING" "$STAGE_B_1024_RANKING" "$STAGE_B_2048_RANKING"; do
  test -s "$required"
done
test ! -e "$RUN_ROOT"

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT WORKTREE PYTHON SIGLIP2_MODEL DATA_RELEASE DEVELOPMENT_MANIFEST
export STAGE_A_CHECKPOINT STAGE_B_1024_CHECKPOINT STAGE_B_2048_CHECKPOINT
export STAGE_A_RANKING STAGE_B_1024_RANKING STAGE_B_2048_RANKING

sbatch \
  --export=ALL \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --account="$ACCOUNT" \
  --gres="$GPU_GRES" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --time="$TIME_LIMIT" \
  --job-name=qcpr-source-audit \
  --output="$RUN_ROOT/slurm-%j.out" \
  --error="$RUN_ROOT/slurm-%j.err" \
  --wrap="set -eu
cd \"\$WORKTREE\"
test \"\$(git rev-parse HEAD)\" = \"\$EXPECTED_SHA\"
test -z \"\$(git status --porcelain)\"
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/audit_temporal_siglip_source_dependence.py\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --checkpoint stage_a=\"\$STAGE_A_CHECKPOINT\" \\
  --checkpoint stage_b_step1024=\"\$STAGE_B_1024_CHECKPOINT\" \\
  --checkpoint stage_b_step2048=\"\$STAGE_B_2048_CHECKPOINT\" \\
  --ranking stage_a=\"\$STAGE_A_RANKING\" \\
  --ranking stage_b_step1024=\"\$STAGE_B_1024_RANKING\" \\
  --ranking stage_b_step2048=\"\$STAGE_B_2048_RANKING\""
