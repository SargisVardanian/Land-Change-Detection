#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"
cd "$CODE_ROOT"
export UNICHANGE_MCI_ROOT="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
export UNICHANGE_RUN_DIR="$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100"

cmd="${1:-}"
case "$cmd" in
  audit)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/audit_unichange_data.py"
    ;;
  probe-models)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/probe_jina_v5.py"
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/probe_universat.py"
    ;;
  preprocess-cc)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/build_levir_cc_pair_manifest.py" \
      --root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-CC" \
      --output "$RS_PROJECT_ROOT/indexes/levir_cc_pairs.jsonl" \
      --caption-output "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries.jsonl"
    ;;
  preprocess-mci)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/validate_levir_mci_dataset.py" \
      --manifest "$RS_PROJECT_ROOT/indexes/levir_mci_samples_fixed.jsonl"
    ;;
  smoke)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" -m pytest -q tests/test_unichange_contracts.py tests/test_unichange_joint_training.py
    ;;
  overfit)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/train_unichange_joint.py" \
      --data-root "$UNICHANGE_MCI_ROOT" \
      --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
      --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
      --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
      --output-dir "$UNICHANGE_RUN_DIR" \
      --subset-file "$UNICHANGE_RUN_DIR/subset_100.json" \
      --run-name joint_retrieval_grounding_overfit_100 \
      --max-pairs 100 \
      --batch-size "${BATCH_SIZE:-4}" \
      --epochs "${EPOCHS:-10}" \
      --device "${DEVICE:-cuda}"
    ;;
  train-retrieval)
    "$0" overfit
    ;;
  eval-retrieval)
    RUN_DIR="$UNICHANGE_RUN_DIR"
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/evaluate_unichange_joint.py" \
      --data-root "$UNICHANGE_MCI_ROOT" \
      --subset-file "$RUN_DIR/subset_100.json" \
      --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
      --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
      --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
      --checkpoint "$RUN_DIR/best_heads.pt" \
      --output "$RUN_DIR/eval_metrics.json" \
      --max-pairs 100 \
      --batch-size "${BATCH_SIZE:-4}" \
      --device "${DEVICE:-cuda}"
    ;;
  train-masks|eval-masks|pair-rerank)
    echo "Command '$cmd' is gated until joint_retrieval_grounding_overfit_100 produces real metrics and checkpoints." >&2
    exit 64
    ;;
  render)
    PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/render_unichange_results.py" \
      --data-root "$UNICHANGE_MCI_ROOT" \
      --subset-file "$UNICHANGE_RUN_DIR/subset_100.json" \
      --checkpoint "$UNICHANGE_RUN_DIR/best_heads.pt" \
      --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
      --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
      --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
      --output-dir "$UNICHANGE_RUN_DIR/visuals" \
      --device "${DEVICE:-cuda}"
    ;;
  *)
    cat >&2 <<'USAGE'
Usage: cluster/ysu/run_unichange.sh <command>

Commands:
  audit
  probe-models
  preprocess-cc
  preprocess-mci
  smoke
  overfit
  train-retrieval
  eval-retrieval
  train-masks
  eval-masks
  pair-rerank
  render
USAGE
    exit 2
    ;;
esac
