#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

cmd="${1:-}"
case "$cmd" in
  audit)
    PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/audit_unichange_data.py"
    ;;
  probe-models)
    PYTHONPATH="$RS_PROJECT_ROOT/src" "$PYTHON" "$RS_PROJECT_ROOT/scripts/probe_jina_v5.py"
    PYTHONPATH="$RS_PROJECT_ROOT/src" "$PYTHON" "$RS_PROJECT_ROOT/scripts/probe_universat.py"
    ;;
  preprocess-cc)
    PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/build_levir_cc_pair_manifest.py" \
      --root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-CC" \
      --output "$RS_PROJECT_ROOT/indexes/levir_cc_pairs.jsonl" \
      --caption-output "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries.jsonl"
    ;;
  preprocess-mci)
    PYTHONPATH="$RS_PROJECT_ROOT/src" "$PYTHON" "$RS_PROJECT_ROOT/scripts/validate_levir_mci_dataset.py" \
      --manifest "$RS_PROJECT_ROOT/indexes/levir_mci_samples_fixed.jsonl"
    ;;
  smoke)
    PYTHONPATH="$RS_PROJECT_ROOT/src" "$PYTHON" -m pytest -q tests/test_unichange_contracts.py
    ;;
  overfit)
    PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/train_unichange_joint.py" \
      --data-root "$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI" \
      --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
      --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
      --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
      --output-dir "$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100" \
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
    RUN_DIR="$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100"
    PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/eval_unichange_joint.py" \
      --data-root "$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI" \
      --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
      --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
      --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
      --checkpoint "$RUN_DIR/best.pt" \
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
    PYTHONPATH="$RS_PROJECT_ROOT/src:$RS_PROJECT_ROOT" "$PYTHON" "$RS_PROJECT_ROOT/scripts/render_unichange_joint.py" \
      --run-dir "$RS_PROJECT_ROOT/runs/joint_retrieval_grounding_overfit_100"
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
