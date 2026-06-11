# YSU Cluster Runbook

This folder prepares `Land-Change-Detection` for training on the Yerevan State University cluster.

## What Is Included

- `cluster/ysu/prepare_workspace.sh`: creates the Weka-backed project layout and the local env export file.
- `cluster/ysu/activate_env.sh`: activates either the named Conda env or the repo `.venv`.
- `cluster/ysu/train_retrieval_head.sbatch`: Slurm job for retrieval-head training and evaluation.
- `cluster/ysu/train_qwen_lora.sbatch`: Slurm job for Qwen LoRA training.
- `cluster/ysu/install_terratorch.sh`: installs optional TerraTorch dependencies into the active environment.
- `cluster/ysu/write_prithvi_terratorch_template.sh`: writes a starter TerraTorch config template near the Prithvi checkpoint directory.
- `cluster/ysu/diagnose_terratorch_env.sbatch`: writes a readiness JSON for experimental Prithvi/TerraTorch runs.
- `cluster/ysu/inspect_prithvi_runtime.sbatch`: inspects whether checkpoint/config candidates for Prithvi/TerraTorch are present.
- `cluster/ysu/validate_prithvi_runtime_bundle.sbatch`: exports the unified Prithvi runtime bundle JSON.
- `cluster/ysu/inspect_prithvi_backend.sbatch`: exports the current backend runtime scaffold JSON.
- `cluster/ysu/load_prithvi_backend_runtime.sbatch`: exercises the backend loader lifecycle and writes the loaded runtime scaffold JSON.
- `cluster/ysu/setup_rschange_env.sh`: creates or reuses the `rschange` environment and installs the required Python packages.
- `cluster/ysu/push_and_bootstrap_from_mac.sh`: Mac-side helper that syncs the repo to YSU-HPC and launches dataset/model bootstrap remotely.
- `cluster/ysu/bootstrap_from_mac.sh`: Mac-side entrypoint that optionally writes `~/.ssh/config`, runs preflight, and then launches the remote bootstrap.
- `cluster/ysu/verify_project_assets.sh`: writes a JSON report on the cluster showing whether required datasets, models, indexes, and previews exist.
- `cluster/ysu/verify_project_assets_from_mac.sh`: runs the remote verifier and pulls the JSON report back to the MacBook.
- `cluster/ysu/smoke_report.sbatch`: runs a unified GPU smoke report and writes one JSON with Torch/CUDA, project assets, and Prithvi runtime status.
- `cluster/ysu/run_smoke_report_from_mac.sh`: submits the smoke-report job from the MacBook.
- `cluster/ysu/diagnose_access_from_mac.sh`: writes a Mac-side JSON report for DNS, HTTPS, SSH alias, SSH batch connectivity, and available WireGuard profiles.
- `cluster/ysu/private/wireguard/*.conf`: local-only WireGuard configs for off-campus and on-campus access.

## One-Time Setup

1. Install WireGuard on your laptop and import one of the local configs from `cluster/ysu/private/wireguard/`.
2. Connect to the VPN.
3. Open [cluster.ysu.am](https://cluster.ysu.am), sign in, change the cluster password there, and add your SSH key.
4. In the cluster Files page, open `Weka` once so `/mnt/weka/svardanyan/` is mounted for your account.
5. On the cluster terminal, clone this repo and run:

```bash
cd /path/to/Land-Change-Detection
bash cluster/ysu/prepare_workspace.sh
```

That creates:

- `/mnt/weka/svardanyan/land-change-detection/data`
- `/mnt/weka/svardanyan/land-change-detection/artifacts`
- `/mnt/weka/svardanyan/land-change-detection/slurm_logs`
- `/mnt/weka/svardanyan/land-change-detection/runs`

and links the repo-local `data/` and `artifacts/` paths to Weka.

## Environment Setup On The Cluster

Preferred Conda flow:

```bash
source /home/svardanyan/miniconda3/etc/profile.d/conda.sh
conda create -n lcd python=3.11 -y
conda activate lcd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,llm]"
```

If you prefer a repo-local virtualenv instead:

```bash
PYTHON_BIN=python3.11 ./scripts/create_env.sh
source .venv/bin/activate
python -m pip install -e ".[llm]"
```

## Storage Rules

- Keep code and small metadata in `/home/svardanyan/`.
- Keep large datasets, checkpoints, logs, and generated artifacts in `/mnt/weka/svardanyan/land-change-detection/`.
- The provided setup script makes the repo use Weka automatically through symlinks.

## Submitting Jobs

GPU LoRA training:

```bash
sbatch cluster/ysu/train_qwen_lora.sbatch
```

Retrieval-head scaffold run:

```bash
sbatch cluster/ysu/train_retrieval_head.sbatch
```

The Slurm templates default to the `research` partition and keep logs under `/mnt/weka/svardanyan/land-change-detection/slurm_logs`.

## Retrieval-First Order

1. Run LEVIR-MCI validation, render, and `overfit_100` with the current simple baseline.
2. Keep DINO retrieval on `--visual-backbone simple_patch` first.
3. Download `dinov2-small` only after the simple baseline passes.
4. Run `scripts/smoke_dinov2_local.py`.
5. Only then use `--visual-backbone dinov2`.

`DINOv2` is optional. It is not required for default tests, LEVIR-MCI bootstrap, or simple baseline training.

## MacBook One-Command Push

If your laptop can resolve and reach the cluster directly, you can push the repo and start the first dataset/model bootstrap with:

```bash
REMOTE_HOST=cluster.ysu.am REMOTE_USER=<your_ysu_user> \
  bash cluster/ysu/push_and_bootstrap_from_mac.sh
```

This flow now targets the portal's `Terminal + Conda` entrypoint by default through `your-user+conda@cluster.ysu.am`.
The remote helper prepares the `rschange` environment and bootstraps the first LEVIR-MCI dataset path unless `SETUP_ENV=0` or `RUN_BOOTSTRAP=0` is set.

If you want a single guarded launcher that checks DNS and SSH first, use:

```bash
REMOTE_USER=<your_ysu_user> WRITE_SSH_CONFIG=1 \
  bash cluster/ysu/bootstrap_from_mac.sh
```

That flow creates or updates both `ysu-hpc` and `ysu-hpc-conda` aliases in `~/.ssh/config`, runs DNS/HTTPS/SSH diagnostics, and then bootstraps the project through the Conda terminal target.
After bootstrap, it also pulls back a verification JSON from the cluster so you can confirm that datasets, indexes, and preview PNGs were actually created.
