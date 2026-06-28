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
    echo "Overfit is gated on successful offline Jina and UniverSat probes." >&2
    exit 64
    ;;
  train-retrieval|eval-retrieval|train-masks|eval-masks|pair-rerank|render)
    echo "Command '$cmd' is reserved for the next UniChange milestone implementation." >&2
    exit 64
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
