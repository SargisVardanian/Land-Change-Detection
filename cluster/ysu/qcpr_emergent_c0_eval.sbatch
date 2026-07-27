#!/usr/bin/env bash
#SBATCH --job-name=qcpr-c0-emergent-eval
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
source "$HOME/rschange_env.sh"
QCPR_WORKTREE=${QCPR_WORKTREE:?QCPR_WORKTREE is required}
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
CHAIN_ROOT=${CHAIN_ROOT:?CHAIN_ROOT is required}
export CODE_ROOT="$QCPR_WORKTREE"
cd "$QCPR_WORKTREE"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
export PYTHONPATH="$QCPR_WORKTREE/src:$CODE_ROOT/scripts"
exec "$RSCHANGE_PYTHON" scripts/evaluate_qcpr_c0_emergent.py \
  --checkpoint "$CHAIN_ROOT/b_long/best_feasible.pt" \
  --manifest "$S2LOOKING_MANIFEST_DIR/m0_dev_manifest.jsonl" \
  --output-dir "$CHAIN_ROOT/c0_development"
