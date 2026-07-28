#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-dataset-v2-audit-fixes
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
CONTROL_ROOT=${CONTROL_ROOT:?set CONTROL_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA}
BASELINE=${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT}
REPRO=${REPRODUCTION:?set REPRODUCTION}
IDENT=${IDENTIFIABILITY_AUDIT:?set IDENTIFIABILITY_AUDIT}
STEPS=${STEPS:-348}; SEED=${SEED:-20260728}
test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$ROOT" status --porcelain)"
test -f "$CONTROL_ROOT/c0_train.jsonl"; test -f "$CONTROL_ROOT/c1_train.jsonl"
test -f "$CONTROL_ROOT/c0_collision_audit.jsonl"; test -f "$CONTROL_ROOT/c1_collision_audit.jsonl"
COMMON=(--initial-checkpoint "$BASELINE" --seed "$SEED" --steps "$STEPS" --manifest-root "$CONTROL_ROOT" --baseline-reproduction "$REPRO" --identifiability-audit "$IDENT")
mkdir -p "$OUTPUT_ROOT"
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PYTHON" "$ROOT/scripts/test_qcpr_matched_mask_contract.py" --root "$CONTROL_ROOT" > "$OUTPUT_ROOT/matched_mask_contract_test.json"
submit() { local arm=$1; local dep=${2:-}; local depflag=(); [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep"); sbatch --parsable "${depflag[@]}" --job-name="qcpr-${arm}-matched" --account=research --partition=research --qos=researcher --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=08:00:00 --output="$OUTPUT_ROOT/${arm}-%j.out" --wrap="cd '$ROOT'; '$PYTHON' scripts/train_qcpr_p0_p1_control.py --arm '$arm' --train-manifest '$CONTROL_ROOT/${arm}_train.jsonl' --development-manifest '$CONTROL_ROOT/${arm}_development.jsonl' --relevance-manifest '$CONTROL_ROOT/${arm}_relevance.jsonl' --output-dir '$OUTPUT_ROOT/$arm' ${COMMON[*]}"
}
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN matched C0 -> C1; no sbatch submitted"; exit 0; fi
C0=$(submit c0); C1=$(submit c1 "$C0"); printf '{"c0_job":"%s","c1_job":"%s","dependency":"c1 afterok c0","logical_score_matrix":"256x128"}\n' "$C0" "$C1" | tee "$OUTPUT_ROOT/submission.json"
