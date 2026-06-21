#!/bin/bash
#SBATCH --job-name=cache_levir_cc_remoteclip
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

cd "$CODE_ROOT"
mkdir -p logs "$RS_PROJECT_ROOT/cache/levir_cc_remoteclip"

"$PYTHON" scripts/cache_levir_cc_remoteclip_text.py \
  --caption-manifest "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries.jsonl" \
  --output-dir "$RS_PROJECT_ROOT/cache/levir_cc_remoteclip" \
  --remoteclip-arch "${REMOTECLIP_ARCH:-ViT-B-32}" \
  --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --device auto
