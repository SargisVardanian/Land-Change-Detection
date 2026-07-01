#!/usr/bin/env bash
#SBATCH --job-name=ucv2-ret-train
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --output=/mnt/weka/%u/rs_change_project/logs/%x_%j.out
#SBATCH --error=/mnt/weka/%u/rs_change_project/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=24:00:00

set -euo pipefail

source "$SLURM_SUBMIT_DIR/cluster/ysu/common_env.sh"
cd "$CODE_ROOT"

MCI_DATA="${MCI_ROOT:-$RS_PROJECT_ROOT/datasets/processed/LEVIR-MCI}"
RUN_DIR="${RUN_DIR:-$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_train}"
SMOKE_REPORT="${SMOKE_REPORT:-$RS_PROJECT_ROOT/runs/unichange_v2_retrieval_smoke/smoke_report.json}"

"$PYTHON" - "$SMOKE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing smoke report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
errors = []
if report.get("status") != "PASS":
    errors.append(f"status={report.get('status')!r}")
if report.get("fake_backbones") is not False:
    errors.append("fake_backbones must be false")
if report.get("frozen_grad_violations") != []:
    errors.append(f"frozen_grad_violations={report.get('frozen_grad_violations')!r}")
if report.get("missing_gradients") != []:
    errors.append(f"missing_gradients={report.get('missing_gradients')!r}")
if report.get("checkpoint_roundtrip_passed") is not True:
    errors.append("checkpoint_roundtrip_passed must be true")
if report.get("gradient_audit_passed") is not True:
    errors.append("gradient_audit_passed must be true")
if report.get("train_val_disjoint") is not True:
    errors.append("train_val_disjoint must be true")
if errors:
    raise SystemExit("UniChange v2 retrieval full training is gated on a successful real smoke: " + "; ".join(errors))
print(f"Smoke gate passed: {path}")
PY

PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT" "$PYTHON" "$CODE_ROOT/scripts/train_unichange_v2_retrieval.py" \
  --data-root "$MCI_DATA" \
  --output-dir "$RUN_DIR" \
  --universat-source "$RS_PROJECT_ROOT/external/UniverSat" \
  --universat-checkpoint "$RS_PROJECT_ROOT/models/universat-base" \
  --jina-model "$RS_PROJECT_ROOT/models/jina-v5-text-small-retrieval" \
  --device cuda \
  --image-size 256 \
  --output-grid 32 \
  --batch-size "${BATCH_SIZE:-16}" \
  --epochs "${EPOCHS:-25}" \
  --num-workers "${NUM_WORKERS:-8}"
