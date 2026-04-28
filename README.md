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
git clone https://github.com/<owner>/Land-Change-Detection.git
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
git clone https://github.com/<owner>/Land-Change-Detection.git
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
