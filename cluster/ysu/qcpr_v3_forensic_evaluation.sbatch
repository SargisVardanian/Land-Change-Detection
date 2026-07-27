#!/usr/bin/env bash
#SBATCH --job-name=qcpr-v3-forensic
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --qos=researcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.out
#SBATCH --error=/mnt/weka/svardanyan/rs_change_project/logs/%x-%j.err
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-v3-rc1
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
E0=/mnt/weka/svardanyan/rs_change_project/runs/qcpr_e0_20260711-015629/pilot/best_retrieval.pt
B_RUN=/mnt/weka/svardanyan/rs_change_project/runs/qcpr_v3_rc1_B100_75e6a7c_replacement_20260715-162718
cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
mkdir -p "$OUTPUT_ROOT"
"$PYTHON" scripts/audit_qcpr_e0_full.py --checkpoint "$E0" --output-dir "$OUTPUT_ROOT/e0_full_natural"
set +e
"$PYTHON" scripts/evaluate_qcpr_v3_fast.py --checkpoint "$B_RUN/last.pt" --training-report "$B_RUN/smoke_report.json" --phase late_interaction --output "$OUTPUT_ROOT/b_existing_checkpoint_full_natural.json" --fast-pairs 0 --batch-size 16
B_GATE_EXIT=$?
set -e
# Exit 3 is the evaluator's explicit FAIL_GATE result. It is evidence, not a
# runtime failure, and must not conceal the completed immutable evaluation.
if [[ "$B_GATE_EXIT" != 0 && "$B_GATE_EXIT" != 3 ]]; then exit "$B_GATE_EXIT"; fi
printf '{"existing_b_evaluation_exit": %s}\n' "$B_GATE_EXIT" > "$OUTPUT_ROOT/forensic_job_status.json"
