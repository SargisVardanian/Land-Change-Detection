#!/bin/bash
set -euo pipefail

PRESET="${PRESET:-simple_patch_smoke}"
RUN_NAME="${RUN_NAME:-$PRESET}"
PAIR_FEATURE_MODE="${PAIR_FEATURE_MODE:-signed_delta}"

if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/mnt/weka/$USER/rs_change_project}"
else
  export RS_PROJECT_ROOT="${RS_PROJECT_ROOT:-/data/$USER/rs_change_project}"
fi

mkdir -p "$RS_PROJECT_ROOT/runs"

echo "Submitting LEVIR-CC baseline for preset=$PRESET run_name=$RUN_NAME"

preprocess_job_id="$(sbatch --parsable cluster/ysu/preprocess_levir_cc.sbatch)"
echo "preprocess_levir_cc: $preprocess_job_id"

validate_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$preprocess_job_id \
    cluster/ysu/validate_levir_cc_pair_manifest.sbatch
)"
echo "validate_levir_cc_pair_manifest: $validate_job_id"

random_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$validate_job_id \
    cluster/ysu/eval_levir_cc_random_retrieval.sbatch
)"
echo "eval_levir_cc_random_retrieval: $random_job_id"

if [ "$PRESET" = "simple_patch_smoke" ]; then
  stage_dependency="$validate_job_id"
else
  validate_models_job_id="$(
    sbatch --parsable \
      --dependency=afterok:$validate_job_id \
      --export=ALL,DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}",REMOTECLIP_ARCH="${REMOTECLIP_ARCH:-ViT-B-32}" \
      cluster/ysu/validate_retrieval_model_assets.sbatch
  )"
  echo "validate_retrieval_model_assets: $validate_models_job_id"
  smoke_job_id="$(
    sbatch --parsable \
      --dependency=afterok:$validate_models_job_id \
      --export=ALL,PAIR_FEATURE_MODE="$PAIR_FEATURE_MODE",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
      cluster/ysu/smoke_levir_cc_dino_remoteclip_batch.sbatch
  )"
  echo "smoke_levir_cc_dino_remoteclip_batch: $smoke_job_id"
  cache_dino_job_id="$(
    sbatch --parsable \
      --dependency=afterok:$smoke_job_id \
      --export=ALL,DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}" \
      cluster/ysu/cache_levir_cc_dinov2_features.sbatch
  )"
  echo "cache_levir_cc_dinov2_features: $cache_dino_job_id"
  cache_text_job_id="$(
    sbatch --parsable \
      --dependency=afterok:$cache_dino_job_id \
      --export=ALL,REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
      cluster/ysu/cache_levir_cc_remoteclip_text.sbatch
  )"
  echo "cache_levir_cc_remoteclip_text: $cache_text_job_id"
  stage_dependency="$cache_text_job_id"
fi

overfit_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$stage_dependency \
    --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
    cluster/ysu/overfit_levir_cc_retrieval_100.sbatch
)"
echo "overfit_levir_cc_retrieval_100: $overfit_job_id"

train_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$overfit_job_id \
    --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
    cluster/ysu/train_levir_cc_retrieval.sbatch
)"
echo "train_levir_cc_retrieval: $train_job_id"

eval_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$train_job_id \
    --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train",LEVIR_PAIR_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_dinov2/index.json",LEVIR_TEXT_CACHE_INDEX="$RS_PROJECT_ROOT/cache/levir_cc_remoteclip/index.json",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
    cluster/ysu/eval_levir_cc_retrieval.sbatch
)"
echo "eval_levir_cc_retrieval: $eval_job_id"

grid_job_id="$(
  sbatch --parsable \
    --dependency=afterok:$eval_job_id \
    --export=ALL,PRESET="$PRESET",RUN_NAME="$RUN_NAME",RUN_DIR="$RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train",DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$RS_PROJECT_ROOT/models/dinov2-base}",REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-$RS_PROJECT_ROOT/models/remoteclip/RemoteCLIP-ViT-B-32.pt}" \
    cluster/ysu/render_levir_cc_text_query_grid.sbatch
)"
echo "render_levir_cc_text_query_grid: $grid_job_id"

cat <<EOF

Overfit directory:
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100

Full-train directory:
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train

Watch jobs:
  squeue -u $USER
  tail -f logs/*.out
  tail -f logs/*.err

Expected artifacts:
  $RS_PROJECT_ROOT/runs/levir_cc_manifest_validation.json
  $RS_PROJECT_ROOT/runs/levir_cc_random_retrieval_eval.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/best.pt
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/metrics_history.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/train_summary.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/overfit_report.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/eval_metrics.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/eval_summary.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/text_query_top5_grid.png
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/text_query_top5_grid.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_overfit100/environment_fingerprint.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/best.pt
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/last.pt
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/metrics_history.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/train_summary.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/eval_metrics.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/eval_summary.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/text_query_top5_grid.png
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/text_query_top5_grid.json
  $RS_PROJECT_ROOT/runs/levir_cc_${RUN_NAME}_train/environment_fingerprint.json

EOF
