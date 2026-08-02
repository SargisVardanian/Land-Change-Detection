#!/usr/bin/env bash
# Submit only an explicitly authorized fixed-contract P1 -> P2 chain.
# The default is a hard no-submit gate.
set -euo pipefail

ROOT=/mnt/weka/svardanyan/rs_change_project
WT=$ROOT/code/project-qcpr-dataset-v2-stage2
PY=$ROOT/envs/rschange/bin/python
RELEASE=${RELEASE_ROOT:-$ROOT/manifests/qcpr_dataset_v2_stage2_semantic_hold_63d9004_20260802}
INITIAL=${INITIAL_CHECKPOINT:-$ROOT/manifests/qcpr_dataset_v2_stage2_audit/architecture_screening/corrected-b1-63d9004a38df96a12a7e1c5159ff4dd35a2555c3-206222/B1_screen_checkpoint.pt}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA to the immutable Stage-2 code SHA}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new immutable run directory}
P2_TRAIN_MANIFEST=${P2_TRAIN_MANIFEST:?set P2_TRAIN_MANIFEST to reviewed RSCC/P2 data}
P2_DEV_MANIFEST=${P2_DEV_MANIFEST:?set P2_DEV_MANIFEST to reviewed RSCC/P2 development data}
STAGE2_AUTHORIZE=${STAGE2_AUTHORIZE:-NO}

if [[ "$STAGE2_AUTHORIZE" != YES ]]; then
  echo "NOT_SUBMITTED: set STAGE2_AUTHORIZE=YES only after Dataset-v2 Stage-2 gates are reviewed" >&2
  exit 2
fi

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$INITIAL"
test -s "$RELEASE/dataset_v2_stage2_release.json"
test "$("$PY" - "$RELEASE/dataset_v2_stage2_release.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)" = DATASET_V2_STAGE2_READY
test -s "$P2_TRAIN_MANIFEST"
test -s "$P2_DEV_MANIFEST"

mkdir -p "$RUN_ROOT"
COMMON="--initial-checkpoint '$INITIAL' --cache-root '$RUN_ROOT/feature_cache' --universat-source '$ROOT/external/UniverSat' --universat-checkpoint '$ROOT/models/universat-base' --jina-model '$ROOT/models/jina-v5-text-small-retrieval' --seed 20260802 --steps 348 --physical-microbatch 16 --logical-physical-batch 128 --captions-per-pair 2"

P1=$(sbatch --parsable   --job-name=qcpr-s2-p1-b1   --account=research --partition=research --qos=researcher   --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=12:00:00   --output="$RUN_ROOT/p1-%j.out"   --wrap="set -eu; cd '$WT'; test \"\$(git rev-parse HEAD)\" = '$EXPECTED_SHA'; test -z \"\$(git status --porcelain)\"; export PYTHONPATH='$WT/src:$WT/scripts'; exec '$PY' '$WT/scripts/train_qcpr_stage2_b1_control.py' --stage P1 --train-manifest '$RELEASE/manifests/retrieval_exact_train_v2.jsonl' --development-manifest '$RELEASE/manifests/retrieval_exact_development_v2.jsonl' --output-dir '$RUN_ROOT/p1' $COMMON")

P2=$(sbatch --parsable --dependency=afterok:$P1   --job-name=qcpr-s2-p2-real-b1   --account=research --partition=research --qos=researcher   --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=12:00:00   --output="$RUN_ROOT/p2-real-%j.out"   --wrap="set -eu; cd '$WT'; test \"\$(git rev-parse HEAD)\" = '$EXPECTED_SHA'; test -z \"\$(git status --porcelain)\"; export PYTHONPATH='$WT/src:$WT/scripts'; exec '$PY' '$WT/scripts/train_qcpr_stage2_b1_control.py' --stage P2-real --train-manifest '$P2_TRAIN_MANIFEST' --development-manifest '$P2_DEV_MANIFEST' --output-dir '$RUN_ROOT/p2-real' $COMMON")

printf '{"p1_job":"%s","p2_job":"%s","dependency":"p2 afterok p1","code_sha":"%s","steps":348,"logical_score_matrix":"256x128"}\n' "$P1" "$P2" "$EXPECTED_SHA" | tee "$RUN_ROOT/submission.json"
