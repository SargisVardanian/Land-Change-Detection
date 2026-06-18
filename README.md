# Land Change Detection

Local Streamlit application for visual land-surface change analysis on paired satellite images.

The current runnable app uses:

- `mfaytin/mask2former-satellite` for RGB surface segmentation diagnostics.
- `gemma4:e4b` through Ollama for the final natural-language visual interpretation.
- OSCD sample imagery by default, with optional user-uploaded image pairs.

The normal user-facing report is VLM-first: Gemma receives only the before crop, the after crop, and a labeled `A1..D4` before/after contact sheet. It does not receive Mask2Former maps, class labels, transition tables, DINO features, or heatmaps.

## What Works Now

The Streamlit UI currently provides:

- before/after crop selection;
- Mask2Former surface segmentation diagnostics from `artifacts/models/semantic/mask2former-satellite`;
- a labeled 4x4 before/after comparison grid;
- Gemma visual interpretation with:
  - scene overview;
  - before summary;
  - after summary;
  - main visible changes;
  - per-cell observations for `A1..D4`.

## Runtime Models

Required for the current app:

| Role | Model | Runtime | Required? | Notes |
| --- | --- | --- | --- | --- |
| Surface segmentation diagnostics | `mfaytin/mask2former-satellite` | Hugging Face Transformers / PyTorch | Yes | RGB Mask2Former checkpoint trained for OpenEarthMap-style land-cover classes. |
| Visual explanation | `gemma4:e4b` | Ollama local server | Yes for VLM output | Local vision-language model used for the final English report. |

Research or debug only:

| Model | Status |
| --- | --- |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` | Planned multispectral semantic backend. Not required for the current RGB Streamlit app. |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL` | Heavier future benchmark. Not required for the current app. |
| `akshaydudhane/EarthDial_4B_RGB` | Relevant remote-sensing VLM candidate, but the local HF custom-code path is unstable with the current Transformers stack. Hidden behind experimental/debug UI. |
| AdaptLLM remote-sensing Qwen models | Research candidates. Hidden behind experimental/debug UI because they are heavy or unreliable on this MacBook runtime. |
| DINOv3 SAT models | Feature extractors only, not semantic segmenters. They are not used by the Streamlit app. |

## Requirements

Supported development targets:

- macOS, Linux, or Windows.
- Python `3.11` recommended.
- Git.
- For VLM output: Ollama installed and running.

Hardware notes:

- CPU works, but model inference is slower.
- Apple Silicon can use PyTorch MPS for Mask2Former.
- NVIDIA CUDA can be selected in the UI when PyTorch/CUDA is installed correctly.
- Ollama manages Gemma execution separately from PyTorch device selection.

## Setup On macOS / Linux

```bash
git clone https://github.com/SargisVardanian/Land-Change-Detection.git
cd Land-Change-Detection

chmod +x scripts/create_env.sh
./scripts/create_env.sh
source .venv/bin/activate
```

If your system does not have `python3.11`, install Python 3.11 first, or run:

```bash
PYTHON_BIN=python3 ./scripts/create_env.sh
```

## Setup On Windows

Use PowerShell:

```powershell
git clone https://github.com/SargisVardanian/Land-Change-Detection.git
cd Land-Change-Detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## Download The Required Segmentation Model

The current app needs only Mask2Former:

```bash
source .venv/bin/activate
python scripts/download_semantic_models.py
```

This downloads:

```text
artifacts/models/semantic/mask2former-satellite
```

Optional research checkpoints are not required for the app. Download them only if you need Prithvi experiments:

```bash
python scripts/download_semantic_models.py --include-research
```

## Install And Prepare Ollama

Gemma runs through Ollama, which is a separate local application/server.

Install Ollama:

- macOS: download from [ollama.com/download](https://ollama.com/download)
- Windows: download from [ollama.com/download/windows](https://ollama.com/download/windows)
- Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama if it is not already running:

```bash
ollama serve
```

In another terminal, pull the Gemma model used by the app:

```bash
ollama pull gemma4:e4b
```

Check that Ollama sees it:

```bash
ollama list
```

Expected entry:

```text
gemma4:e4b
```

If your Ollama registry does not provide `gemma4:e4b`, install an equivalent local vision-capable Gemma model and update `GEMMA4_E4B_OLLAMA` in `src/land_change_detection/remote_sensing_vlm.py`.

## Download Sample Data

The app can run with OSCD sample imagery:

```bash
source .venv/bin/activate
python scripts/download_oscd.py
```

You can also use your own before/after image pair from the UI.

## Run The App

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Recommended first run:

1. Keep `Image source` as `OSCD dataset`.
2. Keep `Semantic model` as `Mask2Former satellite / OpenEarthMap classes`.
3. Keep `Available model` as `gemma4:e4b (Ollama)`.
4. Keep `Reasoning budget` as `Full local analysis`.
5. Select or confirm a crop.
6. Wait for the Gemma visual interpretation.

## Normal UI Contract

Normal mode:

- shows selected before/after crop;
- shows Mask2Former segmentation diagnostics;
- shows the `A1..D4` visual comparison grid;
- sends only RGB before/after/contact-sheet images to Gemma;
- shows Gemma's final English report.

Debug mode:

- may show raw prompts, raw model output, runtime details, and semantic diagnostics;
- does not change the normal Gemma input contract.

Experimental/heavy VLMs:

- hidden by default;
- intended only for research/debug;
- not recommended for normal local use.

## Training On The YSU Cluster

The repo now includes a ready cluster bundle under `cluster/ysu/`.

Use it after you have VPN access and Weka mounted on the cluster:

```bash
cd ~/Land-Change-Detection
bash cluster/ysu/prepare_workspace.sh
```

Then create the cluster environment and submit jobs:

```bash
source /home/svardanyan/miniconda3/etc/profile.d/conda.sh
conda create -n lcd python=3.11 -y
conda activate lcd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,llm]"

sbatch cluster/ysu/train_qwen_lora.sbatch
sbatch cluster/ysu/train_retrieval_head.sbatch
```

Cluster-specific notes:

- `data/` and `artifacts/` are redirected to `/mnt/weka/svardanyan/land-change-detection/`.
- GPU jobs use the `research` partition by default.
- Keep large datasets and checkpoints on Weka, not in `/home/svardanyan/`.

For the remote-sensing change retrieval dataset workflow on YSU-HPC, the repo now also includes:

- [docs/ysu_hpc_change_retrieval_setup.md](/Users/sargisvardanyan/Land-Change-Detection/docs/ysu_hpc_change_retrieval_setup.md): end-to-end runbook for the Weka-first `RS_PROJECT_ROOT`
- [scripts/setup_rs_change_project.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/setup_rs_change_project.py): creates the YSU-HPC directory layout and optional dataset manifest
- [scripts/download_change_retrieval_datasets.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/download_change_retrieval_datasets.py): scripted download/prep stage for `LEVIR-CC`, `LEVIR-MCI`, `SECOND-CC`, and reference repos while preserving an existing unpacked LEVIR-MCI copy
- [scripts/check_datasets.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/check_datasets.py): quick dataset counts and size summary
- [scripts/bootstrap_change_retrieval_assets.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/bootstrap_change_retrieval_assets.py): builds indexed retrieval assets from downloaded raw datasets
- [scripts/index_change_retrieval_dataset.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/index_change_retrieval_dataset.py): indexes `LEVIR-MCI` or `SECOND-CC` style folders into JSONL samples
- [scripts/render_change_retrieval_sample.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/render_change_retrieval_sample.py): renders a `T1/T2/mask/caption` preview from an indexed sample
- [scripts/render_bootstrap_previews.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/render_bootstrap_previews.py): renders first automatic previews from bootstrap metadata
- [scripts/build_levir_cc_manifest.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/build_levir_cc_manifest.py): builds a lightweight text retrieval manifest from `LEVIR-CC` captions
- [scripts/build_change_retrieval_training_manifest.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/build_change_retrieval_training_manifest.py): converts indexed pair datasets into training-ready retrieval JSONL
- [scripts/build_semantic_change_task_manifests.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/build_semantic_change_task_manifests.py): converts indexed datasets into change-mask and semantic-transition manifests
- [scripts/train_semantic_change.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/train_semantic_change.py): baseline semantic-first training from semantic manifests
- [scripts/eval_semantic_change.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/eval_semantic_change.py): baseline semantic-first evaluation

## Retrieval-First Runbook

A. Existing LEVIR-MCI baseline:

- validate dataset with `scripts/validate_levir_mci_dataset.py`
- render debug gallery with `scripts/render_levir_mci_debug_gallery.py`
- run `scripts/overfit_levir_mci_100.py`
- summarize with `scripts/summarize_run_metrics.py`

B. Pair retrieval simple baseline:

- build SECOND/Hi-UCD manifests with `scripts/build_secondcc_pair_retrieval_manifest.py` and `scripts/build_hiucd_pair_retrieval_manifest.py`
- train `simple_patch` first with `scripts/train_dino_pair_retrieval.py --visual-backbone simple_patch`
- summarize training output with `scripts/summarize_pair_retrieval_train.py`
- evaluate with `scripts/eval_dino_pair_retrieval.py`
- summarize eval output with `scripts/summarize_pair_retrieval_eval.py`
- query top-k neighbors with `scripts/query_pair_to_pair_retrieval.py`

C. Optional DINOv2 download:

- download `dinov2-small` into `$RS_PROJECT_ROOT/models/dinov2-small` with `scripts/download_dinov2_small.py`

D. Optional local-only DINO smoke:

- run `scripts/smoke_dinov2_local.py`

E. Optional DINO pair retrieval:

- switch to `--visual-backbone dinov2 --dinov2-model-path "$RS_PROJECT_ROOT/models/dinov2-small" --local-files-only` only after the `simple_patch` path is stable

`DINOv2` is optional. It is not required for default tests, pair-retrieval manifest building, or `simple_patch` training.
- [scripts/build_prithvi_semantic_manifest.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/build_prithvi_semantic_manifest.py): converts semantic manifests into a Prithvi-style 6-band EO contract when multispectral paths are available
- [scripts/run_prithvi_semantic_train_eval.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/run_prithvi_semantic_train_eval.py): runs the current semantic baseline in Prithvi-style 6-band mode
- [scripts/run_prithvi_terratorch_experimental.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/run_prithvi_terratorch_experimental.py): explicit Prithvi/TerraTorch experimental runner with honest fallback diagnostics
- [scripts/diagnose_terratorch_env.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/diagnose_terratorch_env.py): writes environment readiness diagnostics for TerraTorch/Prithvi experiments
- [scripts/install_terratorch.sh](/Users/sargisvardanyan/Land-Change-Detection/scripts/install_terratorch.sh): installs optional TerraTorch dependencies into the active environment
- [scripts/inspect_prithvi_runtime.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/inspect_prithvi_runtime.py): inspects checkpoint/config/runtime readiness for Prithvi experiments
- [scripts/write_prithvi_terratorch_template.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/write_prithvi_terratorch_template.py): writes a minimal TerraTorch config template for Prithvi experiments
- [scripts/validate_prithvi_runtime_bundle.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/validate_prithvi_runtime_bundle.py): exports the unified Prithvi runtime bundle contract as JSON
- [scripts/inspect_prithvi_backend.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/inspect_prithvi_backend.py): exports the current `PrithviTerratorchBackend` runtime scaffold JSON
- [scripts/load_prithvi_backend_runtime.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/load_prithvi_backend_runtime.py): exercises the backend loader lifecycle and exports the loaded runtime scaffold JSON
- [scripts/build_levir_cc_text_index.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/build_levir_cc_text_index.py): builds a small `LEVIR-CC` text index with `faiss` or numpy fallback
- [scripts/run_change_retrieval_train_eval.py](/Users/sargisvardanyan/Land-Change-Detection/scripts/run_change_retrieval_train_eval.py): runs the current retrieval train/eval scaffold on generated manifests
- [cluster/ysu/download_change_retrieval_datasets.sh](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/download_change_retrieval_datasets.sh): one-command dataset download stage on the cluster
- [cluster/ysu/bootstrap_change_retrieval_assets.sh](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/bootstrap_change_retrieval_assets.sh): one-command asset bootstrap on the cluster
- [cluster/ysu/bootstrap_change_retrieval_assets.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/bootstrap_change_retrieval_assets.sbatch): batch bootstrap for generated indexes and previews
- [cluster/ysu/train_change_retrieval_head.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/train_change_retrieval_head.sbatch): batch retrieval train/eval on generated manifests
- [cluster/ysu/build_semantic_manifests.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/build_semantic_manifests.sbatch): batch semantic-manifest generation for mask/transition tasks
- [cluster/ysu/train_semantic_change.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/train_semantic_change.sbatch): batch semantic-first baseline training/eval
- [cluster/ysu/build_prithvi_semantic_manifest.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/build_prithvi_semantic_manifest.sbatch): batch Prithvi-style manifest generation
- [cluster/ysu/train_prithvi_semantic_change.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/train_prithvi_semantic_change.sbatch): batch Prithvi-style semantic baseline training/eval
- [cluster/ysu/train_prithvi_terratorch_experimental.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/train_prithvi_terratorch_experimental.sbatch): batch experimental Prithvi/TerraTorch runner with fallback reporting
- [cluster/ysu/install_terratorch.sh](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/install_terratorch.sh): cluster helper for TerraTorch installation
- [cluster/ysu/write_prithvi_terratorch_template.sh](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/write_prithvi_terratorch_template.sh): writes a starter TerraTorch config on the cluster
- [cluster/ysu/diagnose_terratorch_env.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/diagnose_terratorch_env.sbatch): batch TerraTorch readiness diagnostics
- [cluster/ysu/inspect_prithvi_runtime.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/inspect_prithvi_runtime.sbatch): batch runtime/checkpoint inspection for Prithvi experiments
- [cluster/ysu/validate_prithvi_runtime_bundle.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/validate_prithvi_runtime_bundle.sbatch): batch runtime bundle export for Prithvi experiments
- [cluster/ysu/inspect_prithvi_backend.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/inspect_prithvi_backend.sbatch): batch backend runtime scaffold inspection
- [cluster/ysu/load_prithvi_backend_runtime.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/load_prithvi_backend_runtime.sbatch): batch backend loader lifecycle inspection
- [cluster/ysu/slurm_test_gpu.sbatch](/Users/sargisvardanyan/Land-Change-Detection/cluster/ysu/slurm_test_gpu.sbatch): safe first GPU smoke test on the cluster

Typical YSU-HPC retrieval setup flow:

```bash
python scripts/setup_rs_change_project.py --root "$RS_PROJECT_ROOT" --write-manifest
bash cluster/ysu/download_change_retrieval_datasets.sh
bash cluster/ysu/bootstrap_change_retrieval_assets.sh
sbatch cluster/ysu/bootstrap_change_retrieval_assets.sbatch
sbatch cluster/ysu/train_change_retrieval_head.sbatch
sbatch cluster/ysu/build_semantic_manifests.sbatch
sbatch cluster/ysu/train_semantic_change.sbatch
sbatch cluster/ysu/build_prithvi_semantic_manifest.sbatch
sbatch cluster/ysu/train_prithvi_semantic_change.sbatch
sbatch cluster/ysu/train_prithvi_terratorch_experimental.sbatch
sbatch cluster/ysu/diagnose_terratorch_env.sbatch
sbatch cluster/ysu/inspect_prithvi_runtime.sbatch
sbatch cluster/ysu/validate_prithvi_runtime_bundle.sbatch
sbatch cluster/ysu/inspect_prithvi_backend.sbatch
sbatch cluster/ysu/load_prithvi_backend_runtime.sbatch
sbatch cluster/ysu/slurm_test_gpu.sbatch
```

## Why DINOv3 Is Not Used As Segmentation

DINOv3 SAT models such as `facebook/dinov3-vitl16-pretrain-sat493m` and `timm/vit_large_patch16_dinov3.sat493m` are feature-extraction backbones. They do not include a trained land-cover segmentation decoder/head in this project, so they cannot directly output classes such as road, building, water, or bare land.

For that reason, DINOv3 is not part of the Streamlit runtime.

## Architecture Direction

The longer-term research architecture remains semantic-first:

```text
T1 semantic segmentation -> T2 semantic segmentation -> transition matrix -> interpretable report
```

Planned next steps:

1. Keep Mask2Former as the current RGB baseline.
2. Integrate `Prithvi-EO-2.0-300M-TL` through a proper multispectral TerraTorch path.
3. Add `Prithvi-EO-2.0-600M-TL` only after the 300M path is stable.
4. Add CDMamba as a binary changed/unchanged validation baseline, not as the main semantic answer.
5. Keep VLMs as explanation/reporting tools, not as pixel-mask evidence generators.

## Verification

Run static and unit checks:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m compileall -q app.py src tests
PYTHONPATH=src python -m pytest -q
```

Expected current result:

```text
26 passed
```

## Sources

- `mfaytin/mask2former-satellite`: [Hugging Face](https://huggingface.co/mfaytin/mask2former-satellite)
- `Prithvi-EO-2.0-300M-TL`: [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)
- `Prithvi-EO-2.0-600M-TL`: [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)
- `Prithvi-EO-2.0` paper: [arXiv:2412.02732](https://arxiv.org/abs/2412.02732)
- `CDMamba`: [arXiv:2406.04207](https://arxiv.org/abs/2406.04207)
- `EarthDial`: [arXiv:2412.15190](https://arxiv.org/abs/2412.15190)
- Ollama installation: [ollama.com/download](https://ollama.com/download)
# Land-Change-Detection
