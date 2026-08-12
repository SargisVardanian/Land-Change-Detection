#!/usr/bin/env bash
# Frozen GeoRSCLIP step-zero common-gallery evaluation; no optimizer steps.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${GEORSCLIP_CHECKPOINT:?GEORSCLIP_CHECKPOINT is required}"
: "${GEORSCLIP_REVISION:?GEORSCLIP_REVISION is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"

GALLERY_BATCH_SIZE=${GALLERY_BATCH_SIZE:-16}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-128}
RERANK_QUERY_BATCH_SIZE=${RERANK_QUERY_BATCH_SIZE:-16}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-01:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
JOB_NAME=${JOB_NAME:-qcpr-georsclip-common-gallery}

test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -d "$DATA_RELEASE"
test -s "$DEVELOPMENT_MANIFEST"
test -s "$GEORSCLIP_CHECKPOINT"

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA RUN_ROOT DATA_RELEASE DEVELOPMENT_MANIFEST
export GEORSCLIP_CHECKPOINT GEORSCLIP_REVISION WORKTREE PYTHON
export TEMPORAL_CHECKPOINT="${TEMPORAL_CHECKPOINT:-}"
export GALLERY_BATCH_SIZE QUERY_BATCH_SIZE RERANK_QUERY_BATCH_SIZE

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
test -d \"\$DATA_RELEASE\"
test -s \"\$DEVELOPMENT_MANIFEST\"
test -s \"\$GEORSCLIP_CHECKPOINT\"
if test -n \"\$TEMPORAL_CHECKPOINT\"; then
  test -s \"\$TEMPORAL_CHECKPOINT\"
fi
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
set -- \"\$WORKTREE/scripts/evaluate_qcpr_georsclip_common_gallery.py\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --georsclip-checkpoint \"\$GEORSCLIP_CHECKPOINT\" \\
  --georsclip-revision \"\$GEORSCLIP_REVISION\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --worktree \"\$WORKTREE\" \\
  --gallery-batch-size \"\$GALLERY_BATCH_SIZE\" \\
  --query-batch-size \"\$QUERY_BATCH_SIZE\" \\
  --rerank-query-batch-size \"\$RERANK_QUERY_BATCH_SIZE\"
if test -n \"\$TEMPORAL_CHECKPOINT\"; then
  set -- \"\$@\" --temporal-checkpoint \"\$TEMPORAL_CHECKPOINT\"
fi
exec \"\$PYTHON\" \"\$@\""
