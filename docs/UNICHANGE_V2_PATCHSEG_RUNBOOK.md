# UniChange v2 QCPR/PatchSeg runbook

This path adds query-conditioned patch reranking and mask supervision without
replacing Stage-1-next semantic retrieval. Global retrieval remains available
as `S_global`; QCPR adds `S_local` and ranks with
`S_final = alpha * S_global + beta * S_local`.

## Required order

Run every job from the exact commit that will be trained. For mixed LEVIR-MCI
and SECOND-CC experiments, export the same `TRAIN_MANIFESTS`, `VAL_MANIFESTS`
and `DATASET_WEIGHTS` for smoke, pilot and full training.

```bash
source "$HOME/rschange_env.sh"
cd /mnt/weka/svardanyan/rs_change_project/code/project
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD"

sbatch commands/stage1_next_v2_patchseg_smoke_premium.sh
sbatch commands/stage1_next_v2_patchseg_memory_premium.sh
```

The smoke report must contain:

- `patch_reranker_available=true`;
- `qcpr_score_mode="fused"`;
- `qcpr_gradient_audit_passed=true`;
- non-zero finite gradients for `patch_projector` and `query_mask_head`;
- ten finite BF16 H100 steps and a successful checkpoint round-trip.

The memory report must contain `patch_reranker_available=true`, fused scoring
and a successful physical batch of at least 32. Pass those exact report paths
to the pilot:

```bash
export SMOKE_REPORT=/path/to/patchseg-smoke/smoke_report.json
export MEMORY_REPORT=/path/to/patchseg-memory/memory_probe.json
sbatch commands/stage1_next_v2_patchseg_pilot_premium.sh
```

The pilot defaults to `MAX_STEPS=500` and `EPOCHS=2`. `MAX_STEPS` is forwarded
through both shell wrappers to Python and produces `BOUNDED_COMPLETED` in the
training report when reached.

After the pilot, run bounded evaluation and create the contact sheet:

```bash
export RUN_DIR=/path/to/pilot-run
sbatch commands/stage1_next_v2_patchseg_eval_micro_premium.sh
RUN_DIR="$RUN_DIR" bash commands/stage1_next_v2_patchseg_contact_sheet.sh
```

`evaluation/retrieval_results.jsonl` records top-level and per-result
`S_global`, `S_local`, `S_final`, their long-name aliases and `score_mode`.
Both evaluation JSON summaries must report `patch_reranker_available=true`
and `qcpr_score_mode="fused"`.

Full training is permitted only after the pilot has no NaN/Inf, bounded losses,
valid QCPR gradients and visually plausible non-trivial overlays:

```bash
sbatch commands/stage1_next_v2_patchseg_train.sh
```

The readiness gate rejects baseline smoke/memory reports when patch reranking
is enabled, even if they were produced at the same commit.

## S2Looking Phase-0 supervision

The official S2Looking contract has two direction-specific maps: `Label 1`
marks newly built regions and `Label 2` marks demolished regions. Build two
query-mask rows per base pair without enabling retrieval supervision:

```bash
python scripts/build_s2looking_qcpr_manifest.py \
  --root "$RS_PROJECT_ROOT/datasets/raw/S2Looking" \
  --output "$RS_PROJECT_ROOT/manifests/s2looking_qcpr.jsonl" \
  --audit-report "$RS_PROJECT_ROOT/reports/s2looking_qcpr_audit.json"
```

Directory names default to `Image1`, `Image2`, `label1` and `label2` below
each official `train`, `val` and `test` split and are configurable through CLI
flags. The builder fails on missing or misaligned stems. Rows use
`retrieval_supervision=false`, `seg_supervision_mode=query_specific` and
direction-specific mask paths, so S2Looking supervises the temporal/patch/mask
branches without being treated as a natural-language retrieval benchmark.

## Structured FNA ablation

RSICRC-inspired false-negative attraction is feature-gated and defaults off.
It preserves duplicate-caption positives, assigns graded attraction to
compatible object/direction/location constraints and gives temporal
contradictions zero relevance.

```bash
export STRUCTURED_FNA_WEIGHT=0.25
sbatch commands/stage1_next_v2_patchseg_pilot_premium.sh
```

Run this as a separate E2 ablation after the E0 QCPR control. Do not silently
change E0's objective.
