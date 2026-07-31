#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/weka/svardanyan/rs_change_project/code/project-qcpr-dataset-v2-audit-fixes
PYTHON=/mnt/weka/svardanyan/rs_change_project/envs/rschange/bin/python
CONTROL_ROOT=${CONTROL_ROOT:?set CONTROL_ROOT to immutable P0/P1 manifest root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to new run root}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA to the committed code SHA}
BASELINE=${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT to epoch-19 checkpoint}
REPRO=${REPRODUCTION:?set REPRODUCTION to immutable R0 reproduction.json}
IDENT=${IDENTIFIABILITY_AUDIT:?set IDENTIFIABILITY_AUDIT to immutable audit JSON}
STEPS=${STEPS:-348}
SEED=${SEED:-20260728}
test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$ROOT" status --porcelain)"
mkdir -p "$OUTPUT_ROOT"
COMMON=(--initial-checkpoint "$BASELINE" --seed "$SEED" --steps "$STEPS" --manifest-root "$CONTROL_ROOT" --baseline-reproduction "$REPRO" --identifiability-audit "$IDENT")
P0=$(sbatch --parsable --job-name=qcpr-p0-control --account=research --partition=research --qos=researcher --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=08:00:00 --output="$OUTPUT_ROOT/p0-%j.out" --wrap="cd '$ROOT'; '$PYTHON' scripts/train_qcpr_p0_p1_control.py --arm p0 --train-manifest '$CONTROL_ROOT/p0_train.jsonl' --development-manifest '$CONTROL_ROOT/p0_development.jsonl' --relevance-manifest '$CONTROL_ROOT/p0_relevance.jsonl' --output-dir '$OUTPUT_ROOT/p0' ${COMMON[*]}")
P1=$(sbatch --parsable --dependency=afterok:$P0 --job-name=qcpr-p1-control --account=research --partition=research --qos=researcher --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=08:00:00 --output="$OUTPUT_ROOT/p1-%j.out" --wrap="cd '$ROOT'; '$PYTHON' scripts/train_qcpr_p0_p1_control.py --arm p1 --train-manifest '$CONTROL_ROOT/p1_train.jsonl' --development-manifest '$CONTROL_ROOT/p1_development.jsonl' --relevance-manifest '$CONTROL_ROOT/p1_relevance.jsonl' --output-dir '$OUTPUT_ROOT/p1' ${COMMON[*]}")
printf '{"p0_job":"%s","p1_job":"%s","dependency":"p1 afterok p0","steps":%s,"seed":%s}\n' "$P0" "$P1" "$STEPS" "$SEED" | tee "$OUTPUT_ROOT/submission.json"
