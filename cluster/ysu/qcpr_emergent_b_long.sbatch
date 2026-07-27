#!/usr/bin/env bash
#SBATCH --job-name=qcpr-b-long
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=48:00:00
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
exec "$RSCHANGE_PYTHON" scripts/train_qcpr_emergent_epochs.py \
  --track B --output-dir "$CHAIN_ROOT/b_long" --manifest-dir "$MANIFEST_DIR" \
  --initialization-checkpoint "$CHAIN_ROOT/a0_long/best_feasible.pt" \
  --parent-checkpoint "$CHAIN_ROOT/b_smoke/last.pt" \
  --baseline-report "$CHAIN_ROOT/a0_long/best_feasible_development.json" \
  --prior-run "$CHAIN_ROOT/b_smoke" --start-epoch 2 --max-epochs 30 --min-epochs 10 --patience 6 \
  --batch-size 32
