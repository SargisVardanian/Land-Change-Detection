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
$RS_PROJECT_ROOT/cache/huggingface
```

## Step Order

A. Run LEVIR-MCI validation, render, and overfit with the current simple baseline first.

```bash
sbatch cluster/ysu/validate_levir_mci_dataset.sbatch
sbatch cluster/ysu/render_levir_mci_samples.sbatch
sbatch cluster/ysu/overfit_levir_mci_100.sbatch
```

B. Keep DINO retrieval on `--visual-backbone simple_patch` first.

C. Build and verify the simple pair-retrieval path before touching DINOv2.

```bash
sbatch cluster/ysu/train_dino_pair_retrieval.sbatch
sbatch cluster/ysu/eval_dino_pair_retrieval.sbatch
sbatch cluster/ysu/query_pair_to_pair_retrieval.sbatch
```

The eval job now writes both:

- `$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/eval_metrics.json`
- `$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/eval_summary.json`

The train job now also writes:

- `$RS_PROJECT_ROOT/runs/dino_pair_retrieval_simple_patch/train_summary.json`

D. Download `dinov2-small` only after the `simple_patch` checks pass.

```bash
sbatch cluster/ysu/download_dinov2_small.sbatch
```

E. Run the local-only DINO smoke test.

```bash
sbatch cluster/ysu/smoke_dinov2_local.sbatch
```

F. Only then train with `--visual-backbone dinov2 --dinov2-model-path "$RS_PROJECT_ROOT/models/dinov2-small" --local-files-only`.

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

1. LEVIR-MCI real files parse successfully.
2. `runs/levir_mci_grid.png` is generated from real data.
3. `runs/levir_mci_overfit/` exists and contains a checkpoint plus metrics history.
4. Overfit-100 reaches train Dice above `0.90`.
5. Full training reports Dice, IoU, Precision, Recall, Recall@1/5/10, and MRR.
6. Retrieval `Recall@5` is clearly above random.
7. Predicted masks visually match real ground-truth masks.

Do not call the project complete until these cluster artifacts exist:

- `$RS_PROJECT_ROOT/runs/levir_mci_grid.png`
- `$RS_PROJECT_ROOT/runs/levir_mci_overfit/`
- `$RS_PROJECT_ROOT/logs/`
- `$RS_PROJECT_ROOT/indexes/`
- `$RS_PROJECT_ROOT/datasets/dataset_manifest.md`
- `$RS_PROJECT_ROOT/rs_change_project_verification.json`
- `$RS_PROJECT_ROOT/runs/cluster_smoke_report.json`
