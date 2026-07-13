#!/usr/bin/env bash
#SBATCH --job-name=ucv2-next-train
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
export CODE_ROOT="${CODE_ROOT_OVERRIDE:-$SLURM_SUBMIT_DIR}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT"

MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_stage1_next_train}"
SMOKE_REPORT="${SMOKE_REPORT:?Set SMOKE_REPORT to the successful Stage-1-next smoke report}"
MEMORY_REPORT="${MEMORY_REPORT:?Set MEMORY_REPORT to the successful Stage-1-next memory report}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
READINESS_COMMIT="${READINESS_COMMIT_OVERRIDE:-$CURRENT_COMMIT}"
TEMPORAL_DEPTH="${TEMPORAL_DEPTH:-6}"
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-256}"
MINIMUM_GLOBAL_BATCH="${MINIMUM_GLOBAL_BATCH:-32}"

GATE_ARGS=(
  "$SMOKE_REPORT" "$MEMORY_REPORT" "$READINESS_COMMIT" 1 "$MINIMUM_GLOBAL_BATCH" "$TEMPORAL_DEPTH"
  --expected-text-max-length "$TEXT_MAX_LENGTH"
)
if [[ "${ENABLE_PATCH_RERANKER:-0}" == "1" ]]; then
  GATE_ARGS+=(--expected-patch-reranker)
fi
append_gate_colon_list() {
  local flag="$1"
  local values="$2"
  local item
  if [[ -z "$values" ]]; then
    return 0
  fi
  IFS=':' read -r -a _gate_items <<< "$values"
  for item in "${_gate_items[@]}"; do
    if [[ -n "$item" ]]; then
      GATE_ARGS+=("$flag" "$item")
    fi
  done
}
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  GATE_ARGS+=(--expected-data-mode mixed --expected-dataset-config "$DATASET_CONFIG")
elif [[ -n "${TRAIN_MANIFESTS:-}" || -n "${VAL_MANIFESTS:-}" ]]; then
  GATE_ARGS+=(--expected-data-mode mixed)
  append_gate_colon_list --expected-train-manifest "${TRAIN_MANIFESTS:-}"
  append_gate_colon_list --expected-val-manifest "${VAL_MANIFESTS:-}"
  append_gate_colon_list --expected-dataset-weight "${DATASET_WEIGHTS:-levir_mci=0.55:second_cc=0.45}"
else
  GATE_ARGS+=(--expected-data-mode levir_only)
fi

BATCH_SIZE="$($PYTHON "$CODE_ROOT/scripts/ucv2_stage1_next_readiness_gate.py" "${GATE_ARGS[@]}")"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
if (( BATCH_SIZE < MINIMUM_GLOBAL_BATCH )); then
  echo "One-H100 Stage-1-next training requires actual batch size at least ${MINIMUM_GLOBAL_BATCH}" >&2
  exit 2
fi

mkdir -p "$RS_PROJECT_ROOT/logs" "$RUN_DIR"
append_colon_list() {
  local flag="$1"
  local values="$2"
  local item
  if [[ -z "$values" ]]; then
    return 0
  fi
  IFS=':' read -r -a _items <<< "$values"
  for item in "${_items[@]}"; do
    if [[ -n "$item" ]]; then
      ARGS+=("$flag" "$item")
    fi
  done
}

ARGS=(
  "$MCI_DATA"
  "$RUN_DIR"
  "$RS_PROJECT_ROOT/external/UniverSat"
  "$RS_PROJECT_ROOT/models/universat-base"
  "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval"
  "$BATCH_SIZE"
  "${EPOCHS:-25}"
  "${NUM_WORKERS:-8}"
  --temporal-depth "$TEMPORAL_DEPTH"
  --max-captions-per-pair "${MAX_CAPTIONS_PER_PAIR:-2}"
  --caption-frequency-power "${CAPTION_FREQUENCY_POWER:-0.5}"
  --train-eval-pairs "${TRAIN_EVAL_PAIRS:-1024}"
  --train-eval-interval "${TRAIN_EVAL_INTERVAL:-2}"
  --similarity-query-chunk-size "${SIMILARITY_QUERY_CHUNK_SIZE:-64}"
  --similarity-candidate-chunk-size "${SIMILARITY_CANDIDATE_CHUNK_SIZE:-256}"
  --text-max-length "$TEXT_MAX_LENGTH"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-4}"
  --early-stopping-min-improvement "${EARLY_STOPPING_MIN_IMPROVEMENT:-0.002}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ "${ENABLE_PATCH_RERANKER:-0}" == "1" ]]; then
  ARGS+=(
    --enable-patch-reranker
    --qcpr-architecture-version "${QCPR_ARCHITECTURE_VERSION:-v1}"
    --qcpr-alpha "${QCPR_ALPHA:-1.0}"
    --qcpr-beta "${QCPR_BETA:-0.25}"
    --qcpr-local-loss-weight "${QCPR_LOCAL_LOSS_WEIGHT:-1.0}"
    --query-segmentation-loss-weight "${QUERY_SEGMENTATION_LOSS_WEIGHT:-0.2}"
    --structured-fna-weight "${STRUCTURED_FNA_WEIGHT:-0.0}"
    --local-margin-loss-weight "${LOCAL_MARGIN_LOSS_WEIGHT:-0.1}"
    --local-margin "${LOCAL_MARGIN:-0.1}"
    --conditional-instance-loss-weight "${CONDITIONAL_INSTANCE_LOSS_WEIGHT:-0.01}"
    --structured-auxiliary-loss-weight "${STRUCTURED_AUXILIARY_LOSS_WEIGHT:-0.05}"
  )
fi
if [[ "${ENABLE_TEMPORAL_EXPLANATION_CHANNELS:-0}" == "1" ]]; then
  ARGS+=(--enable-temporal-explanation-channels)
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ -n "${INITIALIZE_FROM_V1:-}" ]]; then
  ARGS+=(--initialize-from-v1 "$INITIALIZE_FROM_V1")
fi
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  ARGS+=(--dataset-config "$DATASET_CONFIG")
else
  append_colon_list --train-manifest "${TRAIN_MANIFESTS:-}"
  append_colon_list --val-manifest "${VAL_MANIFESTS:-}"
  append_colon_list --dataset-weight "${DATASET_WEIGHTS:-}"
fi

PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" "$PYTHON" \
  "$CODE_ROOT/scripts/train_unichange_v2_stage1_next.py" "${ARGS[@]}"
