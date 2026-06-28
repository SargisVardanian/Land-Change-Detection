# YSU-HPC Actual Runbook

Run everything below inside the YSU Portal `Interactive -> Shell Terminal with Conda` session.

## Project Root

```bash
if [ -d "/mnt/weka/$USER" ] && [ -w "/mnt/weka/$USER" ]; then
    export RS_PROJECT_ROOT="/mnt/weka/$USER/rs_change_project"
else
    export RS_PROJECT_ROOT="/data/$USER/rs_change_project"
fi
```

## Clone Or Refresh The Branch

```bash
mkdir -p "$RS_PROJECT_ROOT/code"
cd "$RS_PROJECT_ROOT/code"

git clone -b codex/ysu-hpc-bootstrap-and-verification \
  https://github.com/SargisVardanian/Land-Change-Detection.git project \
  || true

cd project
git fetch origin
git checkout codex/ysu-hpc-bootstrap-and-verification
git pull origin codex/ysu-hpc-bootstrap-and-verification
```

## Environment And Dataset Bootstrap

```bash
bash cluster/ysu/setup_rschange_env.sh
PYTHONPATH=src /mnt/weka/shared-cache/miniforge3/bin/python \
  scripts/plan_change_retrieval_downloads.py \
  --project-root "$RS_PROJECT_ROOT" \
  --phase baseline \
  --reserve-gb 120 \
  --max-download-gb 50 \
  --output-json "$RS_PROJECT_ROOT/reports/download_plan.json"
bash cluster/ysu/download_change_retrieval_datasets.sh
bash cluster/ysu/bootstrap_change_retrieval_assets.sh
bash cluster/ysu/verify_project_assets.sh
```

The planner uses a single Hugging Face cache root at:

```bash
$RS_PROJECT_ROOT/.cache/huggingface
```

## Step Order

A. Run LEVIR-MCI validation, render, and overfit with the current simple baseline first.

```bash
sbatch cluster/ysu/validate_levir_mci_dataset.sbatch
sbatch cluster/ysu/render_levir_mci_samples.sbatch
sbatch cluster/ysu/overfit_levir_mci_100.sbatch
```

B. Keep retired visual baseline retrieval on `--visual-backbone simple_patch` first.

C. Build and verify the LEVIR-CC `simple_patch_smoke` retrieval path before touching retired visual baseline.

```bash
sbatch cluster/ysu/validate_levir_cc_pair_manifest.sbatch
bash cluster/ysu/submit_levir_cc_baseline.sh
```

This path writes the first LEVIR-CC retrieval artifacts under:

- `$RS_PROJECT_ROOT/runs/levir_cc_simple_patch_smoke_overfit100/`
- `$RS_PROJECT_ROOT/runs/levir_cc_simple_patch_smoke_train/`

D. Download `retired_visual_baseline-small` only after the `simple_patch` checks pass.

```bash
sbatch cluster/ysu/download_retired_visual_baseline_small.sbatch
```

E. Run the local-only retired visual baseline smoke test.

```bash
sbatch cluster/ysu/smoke_retired_visual_baseline_local.sbatch
```

F. Only then train with `--visual-backbone retired_visual_baseline --retired_visual_baseline-model-path "$RS_PROJECT_ROOT/models/retired_visual_baseline-small" --local-files-only`.

## LEVIR-CC Baseline Order

Fastest end-to-end bootstrap for the first `simple_patch_smoke` run:

```bash
bash cluster/ysu/run_levir_cc_baseline_end_to_end.sh all
```

That will:

1. preprocess the extracted LEVIR-CC data into coordinated pair and caption-query manifests
2. validate `levir_cc_pairs.jsonl`
3. run the deterministic random retrieval baseline
4. overfit on 100 unique pair IDs with all sibling captions
5. train the full baseline
6. evaluate retrieval metrics and render the qualitative top-5 text-query grid
7. run one final milestone audit that fails if the required cluster artifacts are incomplete

For the retired visual baseline presets, the dependent `submit_levir_cc_baseline.sh` runs add:

1. model-asset validation
2. one-batch raw-image GPU smoke
3. frozen retired visual baseline pair-token cache
4. frozen retired cross-modal baseline text cache

Build the coordinated manifests first if they are not already present:

```bash
PYTHONPATH=src /mnt/weka/shared-cache/miniforge3/bin/python \
  scripts/build_levir_cc_pair_manifest.py \
  --root "$RS_PROJECT_ROOT/datasets/raw/LEVIR-CC" \
  --output "$RS_PROJECT_ROOT/indexes/levir_cc_pairs.jsonl" \
  --caption-output "$RS_PROJECT_ROOT/indexes/levir_cc_caption_queries.jsonl"
```

Validate the manifest:

```bash
sbatch cluster/ysu/validate_levir_cc_pair_manifest.sbatch
```

Optional real retired visual baseline + retired cross-modal baseline one-batch smoke:

```bash
sbatch cluster/ysu/validate_retrieval_model_assets.sbatch
sbatch --export=ALL,PAIR_FEATURE_MODE=signed_delta,RETIRED_VISUAL_MODEL_PATH="$RS_PROJECT_ROOT/models/retired_visual_baseline-base",RETIRED_TEXT_CHECKPOINT="$RS_PROJECT_ROOT/models/retired_cross_modal_baseline/retired cross-modal baseline-ViT-B-32.pt" \
  cluster/ysu/smoke_levir_cc_retired_visual_baseline_retired_cross_modal_baseline_batch.sbatch
```

After model-asset validation and the raw-image GPU smoke pass, build frozen caches:

```bash
sbatch cluster/ysu/cache_levir_cc_retired_visual_baseline_features.sbatch
sbatch cluster/ysu/cache_levir_cc_retired_cross_modal_baseline_text.sbatch
```

Submit the full LEVIR-CC baseline chain with Slurm dependencies:

```bash
bash cluster/ysu/submit_levir_cc_baseline.sh
PRESET=retired_visual_baseline_t2_only RUN_NAME=retired_visual_baseline_t2_only bash cluster/ysu/submit_levir_cc_baseline.sh
PRESET=retired_visual_baseline_signed_delta RUN_NAME=retired_visual_baseline_signed_delta bash cluster/ysu/submit_levir_cc_baseline.sh
PRESET=retired_visual_baseline_change_fusion RUN_NAME=retired_visual_baseline_change_fusion bash cluster/ysu/submit_levir_cc_baseline.sh
```

For the retired visual baseline presets, also export model paths if they differ from defaults:

```bash
export RETIRED_VISUAL_MODEL_PATH="$RS_PROJECT_ROOT/models/retired_visual_baseline-base"
export RETIRED_TEXT_CHECKPOINT="$RS_PROJECT_ROOT/models/retired_cross_modal_baseline/retired cross-modal baseline-ViT-B-32.pt"
```

The retired visual baseline dependency chain is:

1. preprocess
2. validate manifests
3. random retrieval baseline
4. validate model assets
5. raw real-batch GPU smoke
6. frozen retired visual baseline cache
7. frozen retired cross-modal baseline text cache
8. overfit-100
9. full baseline training
10. evaluation
11. qualitative top-5 grid
12. milestone artifact audit

Each run writes:

- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/best.pt`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/last.pt`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/metrics_history.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/train_summary.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/overfit_report.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/eval_metrics.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/eval_summary.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/text_query_top5_grid.png`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/text_query_top5_grid.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/environment_fingerprint.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/best.pt`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/last.pt`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/metrics_history.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/train_summary.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/eval_metrics.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/eval_summary.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/text_query_top5_grid.png`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/text_query_top5_grid.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/environment_fingerprint.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_milestone_audit.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_manifest_validation.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_retired_visual_baseline_retired_cross_modal_baseline_smoke_report.json` for the explicit smoke step
- `$RS_PROJECT_ROOT/cache/levir_cc_retired_visual_baseline/index.json`
- `$RS_PROJECT_ROOT/cache/levir_cc_retired_cross_modal_baseline/index.json`

The final audit can also be run directly if you want to re-check an existing preset:

```bash
sbatch --export=ALL,PRESET=simple_patch_smoke,RUN_NAME=simple_patch_smoke \
  cluster/ysu/audit_levir_cc_milestone.sbatch
```

## Baseline Jobs

```bash
sbatch cluster/ysu/smoke_report.sbatch
sbatch cluster/ysu/train_levir_mci_binary_retrieval.sbatch
sbatch cluster/ysu/eval_levir_mci_binary_retrieval.sbatch
```

## Checks While Jobs Run

```bash
squeue -u $USER
tail -f logs/*.out
du -sh "$RS_PROJECT_ROOT/datasets/raw/"*
df -h /mnt/weka
df -h /data
```

## Success Criteria

1. `levir_cc_manifest_validation.json` finishes without pair leakage or missing-file errors.
2. `levir_cc_random_retrieval_eval.json` exists as the baseline floor.
3. The one-batch smoke finishes without tensor-shape or model-asset errors.
4. Frozen cache indices exist under `$RS_PROJECT_ROOT/cache/levir_cc_retired_visual_baseline/` and `$RS_PROJECT_ROOT/cache/levir_cc_retired_cross_modal_baseline/`.
5. `levir_cc_<preset>_overfit100/` contains `best.pt`, `last.pt`, `metrics_history.json`, `train_summary.json`, `overfit_report.json`, `eval_metrics.json`, `eval_summary.json`, `text_query_top5_grid.png`, and `environment_fingerprint.json`.
6. `levir_cc_<preset>_train/` contains the full-baseline `best.pt`, `last.pt`, `metrics_history.json`, `train_summary.json`, `eval_metrics.json`, `eval_summary.json`, `text_query_top5_grid.png`, and `environment_fingerprint.json`.
7. `levir_cc_<preset>_milestone_audit.json` exists and reports `"milestone_ready": true`.
8. Overfit-100 materially reduces loss and drives text-to-pair `Recall@1/5/10` above the random baseline.
9. Primary reported metrics are text-to-pair and pair-to-text retrieval metrics, with reversed-pair sanity checks only as auxiliary diagnostics.

Do not call the project complete until these cluster artifacts exist:

- `$RS_PROJECT_ROOT/runs/levir_cc_manifest_validation.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_random_retrieval_eval.json`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_overfit100/`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_train/`
- `$RS_PROJECT_ROOT/runs/levir_cc_<preset>_milestone_audit.json`
- `$RS_PROJECT_ROOT/logs/`
- `$RS_PROJECT_ROOT/indexes/`
- `$RS_PROJECT_ROOT/datasets/dataset_manifest.md`
- `$RS_PROJECT_ROOT/rs_change_project_verification.json`
- `$RS_PROJECT_ROOT/runs/cluster_smoke_report.json`
