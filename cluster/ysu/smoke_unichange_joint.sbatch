#!/usr/bin/env bash
#SBATCH --job-name=unichange-smoke
#SBATCH --output=logs/unichange_smoke_%j.out
#SBATCH --error=logs/unichange_smoke_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$CODE_ROOT"
PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" -m pytest -q tests/test_unichange_contracts.py tests/test_unichange_joint_training.py
