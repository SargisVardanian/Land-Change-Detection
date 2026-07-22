#!/usr/bin/env bash
#SBATCH --job-name=qcpr-b-epoch-smoke
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_SHA:?}" "${CHAIN_ROOT:?}" "${MANIFEST_DIR:?}"
source "$HOME/rschange_env.sh"
cd "$CODE_ROOT"; test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git status --porcelain)"
test -f "$CHAIN_ROOT/a0_long/best_feasible.pt"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts"
exec "$RSCHANGE_PYTHON" scripts/train_qcpr_emergent_epochs.py \
  --track B --output-dir "$CHAIN_ROOT/b_smoke" --manifest-dir "$MANIFEST_DIR" \
  --parent-checkpoint "$CHAIN_ROOT/a0_long/best_feasible.pt" \
  --baseline-report "$CHAIN_ROOT/a0_long/best_feasible_development.json" \
  --max-epochs 1 --min-epochs 1 --patience 1 --batch-size 32 --engineering-smoke
