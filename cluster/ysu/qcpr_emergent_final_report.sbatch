#!/usr/bin/env bash
#SBATCH --job-name=qcpr-emergent-report
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
exec "$RSCHANGE_PYTHON" scripts/build_qcpr_emergent_report.py --chain-root "$CHAIN_ROOT"
