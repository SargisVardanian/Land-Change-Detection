#!/usr/bin/env bash
set -euo pipefail

# One bounded P1 control allocation. This launcher intentionally submits no
# P2 job and does not attach any dependency to a later stage.
ROOT=/mnt/weka/svardanyan/rs_change_project
WT=$ROOT/code/project-qcpr-dataset-v2-stage2
PY=$ROOT/envs/rschange/bin/python
RELEASE=${RELEASE_ROOT:?set RELEASE_ROOT to the immutable Stage-2 release}
INITIAL=${INITIAL_CHECKPOINT:?set INITIAL_CHECKPOINT to the immutable B1 screen checkpoint}
EXPECTED_SHA=${EXPECTED_SHA:?set EXPECTED_SHA to the immutable Stage-2 code SHA}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a new immutable P1 run root}
AUTHORIZE=${P1_AUTHORIZE:-NO}
STEPS=${STEPS:-348}
SEED=${SEED:-20260802}
TRAIN_MANIFEST=$RELEASE/manifests/retrieval_exact_train_v2.jsonl
DEV_MANIFEST=$RELEASE/manifests/retrieval_exact_development_v2.jsonl

if [[ "$AUTHORIZE" != YES ]]; then
  echo "NOT_SUBMITTED: set P1_AUTHORIZE=YES after the P1 preflight is reviewed" >&2
  exit 2
fi
if [[ "$STEPS" != 348 ]]; then
  echo "P1 fixed control requires exactly 348 steps" >&2
  exit 2
fi

test -d "$WT"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WT" status --porcelain)"
test -s "$INITIAL"
test -s "$TRAIN_MANIFEST"
test -s "$DEV_MANIFEST"
test ! -e "$RUN_ROOT"

BASELINE_SHA=$("$PY" - "$INITIAL" <<'PY'
import hashlib
import sys
p=sys.argv[1]
h=hashlib.sha256()
with open(p,"rb") as f:
    for block in iter(lambda:f.read(8*1024*1024),b""):
        h.update(block)
print(h.hexdigest())
PY
)
test "$BASELINE_SHA" = 3b6d89e6c37c25302f8af8f57e3f873564a301070ce47f96a8f611caeffc6e2b

MANIFEST_AUDIT=$("$PY" - "$TRAIN_MANIFEST" "$DEV_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    rows=[json.loads(line) for line in Path(name).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"empty manifest: {name}")
    for row in rows:
        if row.get("query_scope") != "exact_pair":
            raise SystemExit(f"non-exact row in P1 manifest: {name}")
        if row.get("dataset_name") not in {"levir_mci","second_cc"}:
            raise SystemExit(f"unexpected source in P1 manifest: {row.get('dataset_name')}")
        if not row.get("canonical_pair_id") or not row.get("caption_id"):
            raise SystemExit(f"missing identity in {name}")
        if not row.get("t1_path") or not row.get("t2_path"):
            raise SystemExit(f"missing frame path in {name}")
        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    low=key.lower()
                    if any(token in low for token in ("mask","dense_label","official_label","semantic_map","label_path")):
                        raise SystemExit(f"forbidden mask/label field {key} in {name}")
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(row)
    print(f"{Path(name).name}:rows={len(rows)}:pairs={len({r['canonical_pair_id'] for r in rows})}")
PY
)

mkdir -p "$RUN_ROOT"
COMMON=(
  --initial-checkpoint "$INITIAL"
  --cache-root "$RUN_ROOT/feature_cache"
  --universat-source "$ROOT/external/UniverSat"
  --universat-checkpoint "$ROOT/models/universat-base"
  --jina-model "$ROOT/models/jina-v5-text-small-retrieval"
  --seed "$SEED"
  --steps "$STEPS"
  --physical-microbatch 16
  --logical-physical-batch 128
  --captions-per-pair 2
  --source-allowlist levir_mci,second_cc
)

JOB=$(sbatch --parsable \
  --job-name=qcpr-s2-p1-control \
  --account=research --partition=research --qos=researcher \
  --gres=gpu:1 --cpus-per-task=16 --mem=100G --time=12:00:00 \
  --output="$RUN_ROOT/p1-%j.out" \
  --wrap="set -eu; cd '$WT'; test \"\$(git rev-parse HEAD)\" = '$EXPECTED_SHA'; test -z \"\$(git status --porcelain)\"; test -s '$INITIAL'; test \"\$(sha256sum '$INITIAL' | awk '{print \$1}')\" = '$BASELINE_SHA'; export PYTHONPATH='$WT/src:$WT/scripts'; exec '$PY' '$WT/scripts/train_qcpr_stage2_b1_control.py' --stage P1 --train-manifest '$TRAIN_MANIFEST' --development-manifest '$DEV_MANIFEST' --output-dir '$RUN_ROOT/p1' ${COMMON[*]}")

printf '{"job_id":"%s","code_sha":"%s","baseline_sha256":"%s","steps":%s,"logical_score_matrix":"256x128","physical_microbatch":16,"captions_per_pair":2,"manifest_audit":%s,"p2_submitted":false}\n' \
  "$JOB" "$EXPECTED_SHA" "$BASELINE_SHA" "$STEPS" "$(printf '%s' "$MANIFEST_AUDIT" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')" \
  | tee "$RUN_ROOT/submission.json"
