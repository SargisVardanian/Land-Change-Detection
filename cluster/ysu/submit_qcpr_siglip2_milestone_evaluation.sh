#!/usr/bin/env bash
# Submit the full-gallery evaluator after a completed SigLIP-2 phase job.
# This launcher does not train and cannot run before the phase dependency exits.

set -eu

: "${EXPECTED_SHA:?EXPECTED_SHA is required}"
: "${PHASE_JOB_ID:?PHASE_JOB_ID is required}"
: "${PHASE_RUN_ROOT:?PHASE_RUN_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${DATA_RELEASE:?DATA_RELEASE is required}"
: "${DEVELOPMENT_MANIFEST:?DEVELOPMENT_MANIFEST is required}"
: "${SIGLIP2_MODEL:?SIGLIP2_MODEL is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${PYTHON:?PYTHON is required}"

PHASE=${PHASE:?PHASE is required (A or B)}
PARTITION=${PARTITION:-research}
QOS=${QOS:-researcher}
ACCOUNT=${ACCOUNT:-research}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MEMORY=${MEMORY:-128G}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
GPU_GRES=${GPU_GRES:-gpu:h100:1}
JOB_NAME=${JOB_NAME:-qcpr-siglip2-milestone-eval-${PHASE}}
GALLERY_BATCH_SIZE=${GALLERY_BATCH_SIZE:-8}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-64}
RERANK_QUERY_BATCH_SIZE=${RERANK_QUERY_BATCH_SIZE:-4}

test -d "$WORKTREE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WORKTREE" status --porcelain)"
test -d "$PHASE_RUN_ROOT"
test -s "$PHASE_RUN_ROOT/milestone_evaluation_manifest.json"
test -d "$DATA_RELEASE"
test -s "$DEVELOPMENT_MANIFEST"
test -d "$SIGLIP2_MODEL"
test -s "$SIGLIP2_MODEL/model.safetensors"

mkdir -p "$RUN_ROOT"
export EXPECTED_SHA PHASE_JOB_ID PHASE PHASE_RUN_ROOT RUN_ROOT DATA_RELEASE
export DEVELOPMENT_MANIFEST SIGLIP2_MODEL WORKTREE PYTHON
export GALLERY_BATCH_SIZE QUERY_BATCH_SIZE RERANK_QUERY_BATCH_SIZE

sbatch \
  --dependency="afterok:${PHASE_JOB_ID}" \
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
test -s \"\$PHASE_RUN_ROOT/milestone_evaluation_manifest.json\"
export HF_HUB_OFFLINE=1
export PYTHONPATH=\"\$WORKTREE/src:\$WORKTREE/scripts\"
exec \"\$PYTHON\" \"\$WORKTREE/scripts/evaluate_qcpr_siglip2_milestones.py\" \\
  --phase-run-root \"\$PHASE_RUN_ROOT\" \\
  --siglip2-model \"\$SIGLIP2_MODEL\" \\
  --data-release \"\$DATA_RELEASE\" \\
  --development-manifest \"\$DEVELOPMENT_MANIFEST\" \\
  --output-dir \"\$RUN_ROOT\" \\
  --expected-code-sha \"\$EXPECTED_SHA\" \\
  --gallery-batch-size \"\$GALLERY_BATCH_SIZE\" \\
  --query-batch-size \"\$QUERY_BATCH_SIZE\" \\
  --rerank-query-batch-size \"\$RERANK_QUERY_BATCH_SIZE\""
