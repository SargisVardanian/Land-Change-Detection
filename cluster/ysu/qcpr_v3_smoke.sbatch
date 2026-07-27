#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-rc1-smoke
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
mkdir -p /mnt/weka/svardanyan/rs_change_project/logs
DERIVED_MANIFEST_DIR=${DERIVED_MANIFEST_DIR:-/mnt/weka/svardanyan/rs_change_project/manifests/qcpr_v3}
COMMON=(--initialization-mode clean_pretrained --derived-manifest-dir "$DERIVED_MANIFEST_DIR")
"$PYTHON" scripts/train_qcpr_v3.py --output-dir "$OUTPUT_DIR/global" --phase global_bootstrap "${COMMON[@]}" --steps 2 --batch-size 8 --required-datasets levir_mci,second_cc
"$PYTHON" scripts/train_qcpr_v3.py --output-dir "$OUTPUT_DIR/late" --phase late_interaction "${COMMON[@]}" --v3-checkpoint "$OUTPUT_DIR/global/last.pt" --steps 1 --batch-size 8 --required-datasets levir_mci,second_cc
"$PYTHON" scripts/train_qcpr_v3.py --output-dir "$OUTPUT_DIR/mask" --phase mask_grounding "${COMMON[@]}" --v3-checkpoint "$OUTPUT_DIR/late/last.pt" --steps 1 --batch-size 8 --required-datasets s2looking
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
reports={name:json.loads((root/name/'smoke_report.json').read_text()) for name in ('global','late','mask')}
status='PASS' if all(
    report['status']=='PASS'
    and report.get('all_finite') is True
    and report.get('gradient_audit', {}).get('expected_trainable_gradients_finite_nonzero') is True
    for report in reports.values()
) else 'FAIL'
(root/'smoke_report.json').write_text(json.dumps({'status':status,'components':reports},indent=2))
if status!='PASS': raise SystemExit(1)
PY
