#!/usr/bin/env bash
# Compatibility alias. Prefer overfit_unichange_mci_100.sbatch.
#SBATCH --job-name=unichange-joint-overfit100
#SBATCH --output=logs/unichange_joint_overfit100_%j.out
#SBATCH --error=logs/unichange_joint_overfit100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/overfit_unichange_mci_100.sbatch"
