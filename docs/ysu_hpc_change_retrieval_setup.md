# YSU HPC Change Retrieval Quickstart

This runbook adapts the repo to the YSU cluster for the Armenia-focused remote-sensing change workflow:

`T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable report`

For the retrieval/captioning branch, the first practical dataset order is:

1. `LEVIR-CC`
2. `LEVIR-MCI`
3. `SECOND-CC`
4. `BigEarthNet-S2` only if storage still looks healthy
5. `DynamicEarthNet` only with explicit approval and very large free space

## What To Use Where

- MacBook:
  - browser portal access
  - SSH / VS Code Remote SSH
  - code editing
  - pulling logs and results back locally
- YSU cluster login node:
  - cloning the repo
  - creating the Python environment
  - preparing folders
  - downloading datasets inside `tmux`
  - submitting Slurm jobs
- YSU cluster compute nodes through Slurm:
  - training
  - feature extraction
  - retrieval index building when it is non-trivial

Do not run full training on the login node.

## 1. First Login From macOS

Add this to `~/.ssh/config` on the MacBook:

```sshconfig
Host ysu-hpc
    HostName cluster.ysu.am
    User YOUR_YSU_USERNAME
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

Then connect:

```bash
ssh ysu-hpc
```

If the cluster requires VPN first, use the local WireGuard profiles under [cluster/ysu/private/wireguard](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/private/wireguard).

## 2. Recommended Directory Layout

The repo already supports a Weka-backed layout under `/mnt/weka/...`, but for the retrieval datasets described in this workflow the requested project root is:

```bash
/data/$USER/rs_change_project
```

Create it:

```bash
mkdir -p /data/$USER/rs_change_project
cd /data/$USER/rs_change_project

mkdir -p datasets/raw
mkdir -p datasets/processed
mkdir -p datasets/cache
mkdir -p code
mkdir -p logs
mkdir -p runs
mkdir -p checkpoints
mkdir -p indexes
```

## 3. Check Storage Before Any Large Download

```bash
df -h /
df -h /data
du -sh /data/$USER/rs_change_project 2>/dev/null || true
```

Current portal screenshot suggests `/data` has roughly `412 GB` free, which is enough for:

- `LEVIR-CC`
- `LEVIR-MCI`
- `SECOND-CC`

It is not enough for a comfortable `DynamicEarthNet` download.

## 4. Stable Download Session

```bash
tmux new -s rsdata
cd /data/$USER/rs_change_project
```

Useful `tmux` keys:

- detach: `Ctrl-b d`
- reattach: `tmux attach -t rsdata`
- list sessions: `tmux ls`

## 5. Python Environment On Cluster

Conda path:

```bash
source ~/.bashrc
conda create -n rschange python=3.11 -y
conda activate rschange
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e /data/$USER/rs_change_project/code/project[dev]
python -m pip install -U hf_xet datasets gdown zenodo-get tqdm pandas pillow opencv-python pyarrow fastparquet
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Virtualenv fallback:

```bash
python3.11 -m venv ~/venvs/rschange
source ~/venvs/rschange/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e /data/$USER/rs_change_project/code/project[dev]
python -m pip install -U hf_xet datasets gdown zenodo-get tqdm pandas pillow opencv-python pyarrow fastparquet
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## 6. Sync This Repo From MacBook To Cluster

From the MacBook, inside the local repo root:

```bash
rsync -avP \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".venv" \
  --exclude "datasets" \
  --exclude "checkpoints" \
  ./ ysu-hpc:/data/YOUR_YSU_USERNAME/rs_change_project/code/project/
```

Pull results back later with:

```bash
rsync -avP \
  ysu-hpc:/data/YOUR_YSU_USERNAME/rs_change_project/runs/ \
  ./runs_remote/
```

Keep large datasets on the cluster.

## 7. Download The First Three Datasets

This repo now also includes a scripted version of the dataset download stage:

```bash
cd /data/$USER/rs_change_project/code/project
bash cluster/ysu/download_change_retrieval_datasets.sh
```

That wrapper:

- creates the project layout if needed
- downloads `LEVIR-CC` and `LEVIR-MCI` with `huggingface-cli`
- prepares `SECOND-CC` with `zenodo_get`, or writes a fallback helper if `zenodo_get` is missing
- clones the reference repositories into `/data/$USER/rs_change_project/code`

### LEVIR-CC

```bash
cd /data/$USER/rs_change_project/datasets/raw

huggingface-cli download lcybuaa/LEVIR-CC \
    --repo-type dataset \
    --local-dir LEVIR-CC
```

Use:

- pair-to-text change captioning
- text-to-change retrieval
- retrieval embedding training

### LEVIR-MCI

```bash
cd /data/$USER/rs_change_project/datasets/raw

huggingface-cli download lcybuaa/LEVIR-MCI \
    --repo-type dataset \
    --local-dir LEVIR-MCI
```

Use:

- change-mask supervision
- prompt-conditioned segmentation experiments
- caption plus mask alignment

### SECOND-CC

Try Zenodo first:

```bash
cd /data/$USER/rs_change_project/datasets/raw
mkdir -p SECOND-CC
cd SECOND-CC

zenodo_get 10.5281/zenodo.16937571
```

If that fails, use the official repository reference and a Google Drive fallback with `gdown`.

Expected shape:

```text
/data/$USER/rs_change_project/datasets/raw/SECOND-CC/
    train/
    val/
    test/
    SECOND-CC-AUG.json
```

Use:

- semantic transition supervision
- captioned semantic change pairs
- transition-mask experiments

## 8. Clone Dataset Reference Repos

```bash
cd /data/$USER/rs_change_project/code

git clone https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset.git
git clone https://github.com/ChangeCapsInRS/SecondCC.git
git clone https://github.com/hanlinwu/ChangeChat.git
```

Treat `ChangeChat` as a format reference, not as a first training target.

## 9. Download Project Models

For the current repo:

```bash
cd /data/$USER/rs_change_project/code/project
source ~/.bashrc
conda activate rschange
python scripts/download_semantic_models.py \
  --output-root /data/$USER/rs_change_project/checkpoints/models/semantic
```

This fetches the current required semantic diagnostic model:

- `mfaytin/mask2former-satellite`

If you explicitly want the research checkpoints too:

```bash
python scripts/download_semantic_models.py \
  --include-research \
  --output-root /data/$USER/rs_change_project/checkpoints/models/semantic
```

That also downloads:

- `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`
- `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`

Important:

- `mask2former-satellite` is the current practical RGB fallback in this repo
- `Prithvi-EO-2.0-300M-TL` is the preferred next semantic implementation target
- `Prithvi-EO-2.0-600M-TL` is heavier and should wait until the 300M path is stable
- keep downloaded checkpoints under `/data/$USER/rs_change_project/checkpoints/models/semantic`

## 10. Build A Dataset Manifest

Create:

```bash
cat > /data/$USER/rs_change_project/datasets/dataset_manifest.md <<'EOF'
# Dataset Manifest

## LEVIR-CC
Path: /data/$USER/rs_change_project/datasets/raw/LEVIR-CC
Use:
- pair-to-text captioning
- text-to-change retrieval
- image-pair embedding training

## LEVIR-MCI
Path: /data/$USER/rs_change_project/datasets/raw/LEVIR-MCI
Use:
- change masks
- change captioning
- prompt-conditioned mask learning
- VLM instruction data base

## SECOND-CC
Path: /data/$USER/rs_change_project/datasets/raw/SECOND-CC
Use:
- semantic maps
- change captions
- transition segmentation
- prompt-to-mask training

## ChangeChat
Path: /data/$USER/rs_change_project/code/ChangeChat
Use:
- reference for VLM/instruction tuning
- not first-stage training

## BigEarthNet-v2 S2
Path: /data/$USER/rs_change_project/datasets/raw/BigEarthNet-v2
Use:
- static retrieval
- classification pretraining
- remote-sensing representation learning

## DynamicEarthNet
Path: not downloaded yet
Use:
- temporal/seasonal reasoning
- JEPA/temporal pretraining
EOF
```

## 11. Quick Dataset Validation

This repo now includes [scripts/check_datasets.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/check_datasets.py).

Run it on the cluster:

```bash
python /data/$USER/rs_change_project/code/project/scripts/check_datasets.py \
  --root /data/$USER/rs_change_project/datasets/raw
```

## 11.5. Bootstrap Indexed Retrieval Assets

After the raw datasets are present, use the new bootstrap helper:

```bash
cd /data/$USER/rs_change_project/code/project
bash cluster/ysu/bootstrap_change_retrieval_assets.sh
```

This will:

- create or refresh the project layout and dataset manifest
- scan `LEVIR-MCI` and `SECOND-CC` into JSONL sample indexes
- search for a `LEVIR-CC` captions JSON and build a small text retrieval manifest
- build a small `LEVIR-CC` text index prototype with `faiss` when available, otherwise with a numpy fallback
- write a preview manifest under `indexes/` so you know which sample to render first
- convert indexed pair datasets into training-ready retrieval manifests
- build semantic-task manifests for:
  - `LEVIR-MCI` change-mask segmentation
  - `SECOND-CC` semantic transition segmentation when semantic before/after labels are present

Expected outputs:

```text
/data/$USER/rs_change_project/indexes/levir_mci_samples.jsonl
/data/$USER/rs_change_project/indexes/second_cc_samples.jsonl
/data/$USER/rs_change_project/indexes/levir_cc_text_manifest.jsonl
/data/$USER/rs_change_project/indexes/levir_cc_text_index/
/data/$USER/rs_change_project/indexes/levir_mci_train_manifest.jsonl
/data/$USER/rs_change_project/indexes/second_cc_train_manifest.jsonl
/data/$USER/rs_change_project/indexes/semantic/
/data/$USER/rs_change_project/indexes/preview_samples.json
```

To render the first preview manually:

```bash
python scripts/render_change_retrieval_sample.py \
  --index /data/$USER/rs_change_project/indexes/levir_mci_samples.jsonl \
  --sample-id YOUR_SAMPLE_ID \
  --output /data/$USER/rs_change_project/runs/levir_mci_preview.png
```

## 12. First Safe Slurm Job

This repo now includes [cluster/ysu/slurm_test_gpu.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/slurm_test_gpu.sbatch).

Submit it after syncing the repo:

```bash
cd /data/$USER/rs_change_project/code/project
sbatch cluster/ysu/slurm_test_gpu.sbatch
```

Watch it:

```bash
squeue -u $USER
tail -f /data/$USER/rs_change_project/logs/rschange_test_*.out
```

Bootstrap as a batch job if you prefer not to do it on the login node:

```bash
sbatch cluster/ysu/bootstrap_change_retrieval_assets.sbatch
```

Then run the lightweight retrieval scaffold on a generated manifest:

```bash
sbatch cluster/ysu/train_change_retrieval_head.sbatch
```

Build semantic manifests as a separate batch step when needed:

```bash
sbatch cluster/ysu/build_semantic_manifests.sbatch
```

Train the semantic-first baseline on one of the generated semantic manifests:

```bash
sbatch cluster/ysu/train_semantic_change.sbatch
```

If you later enrich semantic manifests with multispectral raster paths in metadata
such as `before_ms_path` and `after_ms_path`, you can build a Prithvi-oriented
manifest:

```bash
sbatch cluster/ysu/build_prithvi_semantic_manifest.sbatch
```

Then run the current Prithvi-style 6-band baseline:

```bash
sbatch cluster/ysu/train_prithvi_semantic_change.sbatch
```

There is also an explicit experimental TerraTorch entrypoint that records whether
the run used a real TerraTorch path or the current fallback baseline:

```bash
sbatch cluster/ysu/train_prithvi_terratorch_experimental.sbatch
```

To prepare or audit the environment itself:

```bash
bash cluster/ysu/install_terratorch.sh
bash cluster/ysu/write_prithvi_terratorch_template.sh
sbatch cluster/ysu/diagnose_terratorch_env.sbatch
sbatch cluster/ysu/inspect_prithvi_runtime.sbatch
sbatch cluster/ysu/validate_prithvi_runtime_bundle.sbatch
sbatch cluster/ysu/inspect_prithvi_backend.sbatch
sbatch cluster/ysu/load_prithvi_backend_runtime.sbatch
```

## 13. What Counts As “Training” In This Repo Today

Two paths exist right now:

### Retrieval scaffold

Already implemented:

```bash
python scripts/train_retrieval_head.py \
  --manifest data/retrieval/train_manifest.jsonl \
  --output-dir artifacts/retrieval/train_run \
  --epochs 5
```

Or through Slurm after adapting paths:

```bash
sbatch cluster/ysu/train_retrieval_head.sbatch
```

This is a lightweight scaffold, not the final production retrieval trainer.

### Semantic segmentation direction

Canonical project direction remains:

`T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable change report`

That means:

1. use the downloaded datasets to build small dataset loaders
2. validate `T1`, `T2`, caption, and mask parsing
3. only then wire the semantic training loop
4. bring in `Prithvi-EO-2.0-300M-TL` before any 600M experiment

## 14. Do Not Download Yet

- `DynamicEarthNet` unless `/data` free space is safely above `700 GB`
- large optional model families that are not tied to the next experiment

## 15. Next Recommended Deliverables

Before full training, produce:

1. dataset manifest
2. dataset file counts
3. sample visualizations for `T1`, `T2`, mask, and caption
4. a small loader for `LEVIR-MCI`
5. a small loader for `SECOND-CC`
6. a tiny FAISS retrieval prototype for `LEVIR-CC`
