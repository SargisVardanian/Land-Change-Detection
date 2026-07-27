#!/usr/bin/env bash
# Compatibility alias. Prefer eval_unichange_mci_100.sbatch.
#SBATCH --job-name=unichange-joint-eval100
#SBATCH --output=logs/unichange_joint_eval100_%j.out
#SBATCH --error=logs/unichange_joint_eval100_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/eval_unichange_mci_100.sbatch"
