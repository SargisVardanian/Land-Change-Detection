#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project
CODE_ROOT=${CODE_ROOT:-$ROOT/code/project-qcpr-emergent-grounding-long}
EXPECTED_SHA=${EXPECTED_SHA:?set the tested immutable SHA}
MANIFEST_DIR=${MANIFEST_DIR:-$ROOT/manifests/qcpr_v3_clean_058779b7}
S2LOOKING_MANIFEST_DIR=${S2LOOKING_MANIFEST_DIR:-$ROOT/manifests/qcpr_v31_m0_a936f116}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
CHAIN_ROOT=${CHAIN_ROOT:-$ROOT/runs/qcpr_emergent_grounding_${EXPECTED_SHA:0:8}_$STAMP}
cd "$CODE_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
export CODE_ROOT EXPECTED_SHA CHAIN_ROOT MANIFEST_DIR S2LOOKING_MANIFEST_DIR
J1=$(sbatch --parsable cluster/ysu/qcpr_emergent_a0_smoke.sbatch)
J2=$(sbatch --parsable --dependency="afterok:$J1" cluster/ysu/qcpr_emergent_a0_long.sbatch)
J3=$(sbatch --parsable --dependency="afterok:$J2" cluster/ysu/qcpr_emergent_b_smoke.sbatch)
J4=$(sbatch --parsable --dependency="afterok:$J3" cluster/ysu/qcpr_emergent_b_long.sbatch)
J5=$(sbatch --parsable --dependency="afterok:$J4" cluster/ysu/qcpr_emergent_c0_eval.sbatch)
J6=$(sbatch --parsable --dependency="afterok:$J5" cluster/ysu/qcpr_emergent_final_report.sbatch)
python - "$CHAIN_ROOT" "$EXPECTED_SHA" "$J1" "$J2" "$J3" "$J4" "$J5" "$J6" <<'PY'
import json, sys
from pathlib import Path
root, sha, *jobs = sys.argv[1:]
names = ("a0_smoke", "a0_long", "b_smoke", "b_long", "c0_evaluator", "final_report")
Path(root, "job_inventory.json").write_text(json.dumps({"git_sha": sha, "jobs": dict(zip(names, jobs))}, indent=2) + "\n")
print(json.dumps({"chain_root": root, "git_sha": sha, "jobs": dict(zip(names, jobs))}, indent=2))
PY
