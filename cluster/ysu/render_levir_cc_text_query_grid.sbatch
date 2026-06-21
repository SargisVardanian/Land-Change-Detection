#!/bin/bash
#SBATCH --job-name=render_levir_cc_text_query_grid
#SBATCH --partition=research
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
set -x

source "$(dirname "$0")/common_env.sh"

PRESET="${PRESET:-simple_patch_smoke}"
RUN_NAME="${RUN_NAME:-$PRESET}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train}"

cd "$CODE_ROOT"
mkdir -p logs "$RUN_DIR"

GRID_ARGS=(
  --preset "$PRESET"
  --manifest "$RS_PROJECT_ROOT/indexes/levir_cc_test.jsonl"
  --project-root "$RS_PROJECT_ROOT"
  --checkpoint "$RUN_DIR/best.pt"
  --output "$RUN_DIR/text_query_top5_grid.png"
  --report-json "$RUN_DIR/text_query_top5_grid.json"
  --top-k 5
  --num-queries 10
  --image-size "${IMAGE_SIZE:-224}"
  --device auto
)

if [ "$PRESET" != "simple_patch_smoke" ]; then
  GRID_ARGS+=(
    --dinov2-model-path "${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}"
    --remoteclip-arch "${REMOTECLIP_ARCH:-ViT-B-32}"
    --remoteclip-checkpoint "${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}"
    --local-files-only
  )
fi

"$PYTHON" scripts/render_levir_cc_text_query_grid.py "${GRID_ARGS[@]}"
