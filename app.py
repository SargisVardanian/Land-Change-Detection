from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
try:
    from streamlit_cropper import st_cropper
except ModuleNotFoundError:
    st_cropper = None

# Keep MPS allocations within a tighter working set on Apple Silicon.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from land_change_detection.data import OSCDSceneRepository
from land_change_detection.change_interpretation.deterministic_reporting import (
    build_cell_report_rows_from_segmentation,
    build_pairwise_visual_context,
    build_scene_overview_from_rows,
)
from land_change_detection.legacy.legacy_visual_context import (
    analyze_change_cells,
    build_change_guide_image,
    build_change_zoom_strip,
    build_deterministic_breakdown,
)
from land_change_detection.pipeline_v2 import LandChangePipelineV2
from land_change_detection.semantic_hf import (
    DEFAULT_CLASS_COLORS,
    MASK2FORMER_SATELLITE_DIR,
    crop_rgb,
    draw_bbox,
)
from land_change_detection.remote_sensing_vlm import (
    GEMMA4_E4B_OLLAMA,
    QWEN3_5_VL_0_8B_MLX_4BIT,
    QWEN3_5_VL_0_8B_MLX_4BIT_DIR,
    QWEN3_VL_4B_THINKING,
    QWEN3_VL_4B_THINKING_MLX_3BIT,
    QWEN3_VL_4B_THINKING_MLX_3BIT_DIR,
    QWEN3_VL_4B_THINKING_DIR,
    QwenMlxVlmExplainer,
    OllamaSemanticChangeExplainer,
    REASONING_PROFILES,
    RemoteSensingQwen2VL2B,
    mlx_model_ready,
    preferred_mlx_qwen_model,
    ollama_model_available,
)
from land_change_detection.segmentation_runtime import SegmentationRuntime
from land_change_detection.semantic_surface import summarize_transitions


st.set_page_config(page_title="Land Surface Mapping", layout="wide")
st.title("Land Surface Mapping")
st.caption("Mouse-selected crop -> semantic segmentation per timestamp -> pairwise change interpretation")

if st_cropper is None:
    st.error("Missing dependency: `streamlit-cropper`.")
    st.code(
        "\n".join(
            [
                "cd /Users/sargisvardanyan/Land-Change-Detection",
                "./scripts/create_env.sh",
                "source .venv/bin/activate",
                "PYTHONPATH=src streamlit run app.py",
            ]
        ),
        language="bash",
    )
    st.stop()

data_root = Path("data/raw/oscd")
repo_ready = (data_root / "images").exists() and (data_root / "train_labels").exists()
VLM_RUNTIME_API_VERSION = "2026-04-16-neutral-contract-v1"


@st.cache_resource(show_spinner=False)
def load_segmentation_runtime(model_dir: str, device_name: str, backend_name: str = "mask2former_openearthmap") -> SegmentationRuntime:
    return SegmentationRuntime(backend_name=backend_name, model_dir=model_dir, device=device_name)


def load_vlm(model_name: str, device_name: str, reasoning_profile: str, runtime_backend: str, api_version: str = VLM_RUNTIME_API_VERSION):
    if runtime_backend == "mlx_vlm_qwen":
        return QwenMlxVlmExplainer(model_name_or_path=model_name, reasoning_profile=reasoning_profile)
    return RemoteSensingQwen2VL2B(model_name_or_path=model_name, device=device_name, reasoning_profile=reasoning_profile)


@st.cache_resource(show_spinner=False)
def load_ollama_explainer(model_name: str, reasoning_profile: str, api_version: str = VLM_RUNTIME_API_VERSION) -> OllamaSemanticChangeExplainer:
    return OllamaSemanticChangeExplainer(model_name=model_name, reasoning_profile=reasoning_profile)


def compact_summary(items: list[dict], min_percent: float = 0.5) -> list[dict]:
    return [item for item in items if item["percent"] >= min_percent]


def summarize_grid_cells(
    class_map: np.ndarray,
    id2label: dict[int, str],
    grid_size: int = 3,
    min_percent: float = 8.0,
) -> list[dict]:
    height, width = class_map.shape
    row_bounds = np.linspace(0, height, grid_size + 1, dtype=int)
    col_bounds = np.linspace(0, width, grid_size + 1, dtype=int)
    row_names = ["top", "middle", "bottom"]
    col_names = ["left", "center", "right"]
    result: list[dict] = []

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            tile = class_map[row_bounds[row_idx] : row_bounds[row_idx + 1], col_bounds[col_idx] : col_bounds[col_idx + 1]]
            if tile.size == 0:
                continue
            uniq, counts = np.unique(tile.astype(np.int32), return_counts=True)
            order = np.argsort(counts)[::-1]
            total = int(tile.size)
            labels = []
            for class_id, count in zip(uniq[order], counts[order], strict=False):
                percent = count * 100.0 / total
                if percent < min_percent:
                    continue
                labels.append(f"{id2label.get(int(class_id), str(int(class_id)))} {percent:.0f}%")
            result.append(
                {
                    "cell": f"{row_names[row_idx]}-{col_names[col_idx]}" if row_names[row_idx] != "middle" or col_names[col_idx] != "center" else "center",
                    "summary": ", ".join(labels[:3]) if labels else "unclear",
                }
            )
    return result


def build_vlm_semantic_context(
    before_map: np.ndarray,
    after_map: np.ndarray,
    id2label: dict[int, str],
    transitions: list[dict],
) -> str:
    before_cells = summarize_grid_cells(before_map, id2label=id2label)
    after_cells = summarize_grid_cells(after_map, id2label=id2label)
    lines = [
        "Semantic layout evidence for the same crop from a lightweight segmentation model.",
        "Treat this as weak secondary evidence only. It should not override clearly visible image content.",
        "BEFORE grid:",
    ]
    lines.extend(f"- {item['cell']}: {item['summary']}" for item in before_cells)
    lines.append("AFTER grid:")
    lines.extend(f"- {item['cell']}: {item['summary']}" for item in after_cells)
    if transitions:
        lines.append("Top semantic transitions:")
        lines.extend(f"- {item['before']} -> {item['after']} ({item['percent']:.1f}%)" for item in transitions[:8])
    return "\n".join(lines)


def trim_semantic_context(text: str, max_chars: int) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    total = 0
    for line in lines:
        extra = len(line) + 1
        if kept and total + extra > max_chars:
            break
        kept.append(line)
        total += extra
    return "\n".join(kept)


def create_trace_collector():
    phase_lines: list[str] = []
    thinking_text = ""
    content_text = ""

    def progress_cb(event: dict) -> None:
        nonlocal thinking_text, content_text
        stage = event.get("stage", "unknown")
        message = event.get("message")
        if message:
            phase_lines.append(f"- `{stage}`: {message}")
        if stage in {"ollama_thinking", "hf_reasoning", "mlx_reasoning"}:
            thinking_text = str(event.get("text", thinking_text))
        elif stage in {"hf_stream", "ollama_content", "mlx_final"}:
            content_text = str(event.get("text", content_text))

    progress_cb.debug_state = {
        "phase_lines": phase_lines,
        "thinking_text": lambda: thinking_text,
        "content_text": lambda: content_text,
    }

    return progress_cb


def uploaded_file_identity(uploaded_file) -> str:
    if uploaded_file is None:
        return "none"
    size = getattr(uploaded_file, "size", None)
    return f"{uploaded_file.name}:{size}"


def model_dir_complete(model_path: Path) -> bool:
    if not model_path.exists() or not model_path.is_dir():
        return False
    shard_files = list(model_path.glob("*.safetensors"))
    if shard_files:
        return True
    return (model_path / "pytorch_model.bin").exists()


def default_vlm_model_name() -> str:
    return preferred_mlx_qwen_model()


def vlm_ready(model_name: str) -> bool:
    model_path = Path(model_name)
    if model_path.exists():
        return model_dir_complete(model_path)
    return True


def has_complete_local_vlm() -> bool:
    return model_dir_complete(QWEN3_VL_4B_THINKING_DIR)


def has_complete_local_mlx_vlm() -> bool:
    return mlx_model_ready(str(QWEN3_VL_4B_THINKING_MLX_3BIT_DIR)) or mlx_model_ready(str(QWEN3_5_VL_0_8B_MLX_4BIT_DIR))


def display_model_name(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return model_path.name
    return model_name


def default_ollama_model_name() -> str:
    return GEMMA4_E4B_OLLAMA


def available_model_presets() -> dict[str, dict[str, str]]:
    presets: dict[str, dict[str, str]] = {}
    presets["Qwen MLX (Apple Silicon default)"] = {
        "backend": "MLX vision-language",
        "runtime_backend": "mlx_vlm_qwen",
        "model_name": preferred_mlx_qwen_model(),
    }
    if has_complete_local_vlm():
        presets["Qwen3-VL-4B-Thinking (HF legacy debug)"] = {
            "backend": "HF legacy debug",
            "runtime_backend": "hf_transformers_legacy",
            "model_name": str(QWEN3_VL_4B_THINKING_DIR),
        }
    presets["gemma4:e4b (Ollama)"] = {
        "backend": "Ollama text",
        "runtime_backend": "ollama_gemma",
        "model_name": GEMMA4_E4B_OLLAMA,
    }
    return presets


def cached_ollama_model_available(model_name: str) -> bool:
    return ollama_model_available(model_name)


def model_runtime_details(
    explanation_backend: str,
    runtime_backend: str,
    selected_model_name: str,
    vlm_device_name: str,
    reasoning_profile: str,
    use_semantic_hints: bool,
) -> dict[str, str]:
    model_path = Path(selected_model_name)
    profile = REASONING_PROFILES[reasoning_profile]
    if runtime_backend == "mlx_vlm_qwen":
        return {
            "family": "VLM",
            "backend": "MLX vision-language",
            "input_mode": "before/after crop images + secondary change-guide",
            "source": "local MLX directory" if model_path.exists() else "Hugging Face MLX repo id",
            "model": display_model_name(selected_model_name),
            "resolved_path": str(model_path.resolve()) if model_path.exists() else selected_model_name,
            "device": "Apple Silicon / MLX",
            "reasoning_profile": profile.label,
            "thinking": "separate reasoning stage",
            "context_window": "compact prompt; MLX manages the runtime context",
            "kv_cache": "managed by MLX; no app-side result reuse",
            "semantic_hints": "enabled" if use_semantic_hints else "disabled",
            "primary_images_sent": "2: before crop, after crop",
            "auxiliary_images_sent": "1: change-guide heatmap",
            "semantic_text_hints_sent": "yes" if use_semantic_hints else "no",
            "semantic_maps_sent_as_images": "no",
        }
    if runtime_backend == "hf_transformers_legacy":
        return {
            "family": "VLM",
            "backend": "HF legacy debug",
            "input_mode": "before/after crop images + secondary change-guide",
            "source": "local directory" if model_path.exists() else "Hugging Face repo id",
            "model": display_model_name(selected_model_name),
            "resolved_path": str(model_path.resolve()) if model_path.exists() else selected_model_name,
            "device": vlm_device_name,
            "reasoning_profile": profile.label,
            "thinking": "enabled" if profile.enable_thinking else "disabled",
            "context_window": "bounded prompt; HF legacy is debug-only on this Mac",
            "kv_cache": "off on mps" if not profile.use_cache_on_mps else "on on mps",
            "semantic_hints": "enabled" if use_semantic_hints else "disabled",
            "primary_images_sent": "2: before crop, after crop",
            "auxiliary_images_sent": "1: change-guide heatmap",
            "semantic_text_hints_sent": "yes" if use_semantic_hints else "no",
            "semantic_maps_sent_as_images": "no",
        }
    return {
        "family": "LLM",
        "backend": "Ollama text",
        "input_mode": "before/after crop images + secondary change-guide",
        "source": "Ollama local registry",
        "model": selected_model_name,
        "resolved_path": "ollama://" + selected_model_name,
        "device": "managed by Ollama",
        "reasoning_profile": profile.label,
        "thinking": str(profile.ollama_think),
        "context_window": f"num_ctx={profile.ollama_num_ctx}; output capped at {profile.ollama_num_predict} tokens",
        "kv_cache": f"keep_alive=0; use OLLAMA_FLASH_ATTENTION=1 + OLLAMA_KV_CACHE_TYPE=q8_0/q4_0 for lower KV memory",
        "semantic_hints": "enabled" if use_semantic_hints else "disabled",
        "primary_images_sent": "2: before crop, after crop",
        "auxiliary_images_sent": "1: change-guide heatmap",
        "semantic_text_hints_sent": "yes" if use_semantic_hints else "no",
        "semantic_maps_sent_as_images": "no",
    }


def green_ratio(rgb: np.ndarray) -> float:
    arr = np.asarray(np.clip(rgb, 0, 255), dtype=np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    greenish = (arr[..., 1] > arr[..., 0] * 1.08) & (arr[..., 1] > arr[..., 2] * 1.05) & (arr[..., 1] > 70)
    return float(greenish.mean())


def normalize_model_scene_overview(parsed: dict | None) -> str:
    if not parsed:
        return ""
    overview = str(parsed.get("scene_overview", "")).strip()
    if not overview:
        return ""
    lowered = overview.lower()
    bad_markers = ["here's a thinking process", "i will analyze", "since no images are provided", "step by step", "placeholder"]
    if any(marker in lowered for marker in bad_markers):
        return ""
    return overview


def render_semantic_color_legend(id2label: dict[int, str]) -> None:
    items = []
    for class_id, label in sorted(id2label.items()):
        color = DEFAULT_CLASS_COLORS.get(label, "#95a5a6")
        items.append(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:0.45rem;
                padding:0.35rem 0.5rem;
                border:1px solid rgba(255,255,255,0.12);
                border-radius:10px;
                background:rgba(255,255,255,0.03);
                min-width:150px;
            ">
                <span style="
                    width:18px;
                    height:18px;
                    border-radius:5px;
                    background:{html.escape(color)};
                    border:1px solid rgba(255,255,255,0.35);
                    display:inline-block;
                "></span>
                <span style="font-size:0.9rem;">{class_id}: {html.escape(label)}</span>
            </div>
            """
        )
    st.markdown(
        "<div style='display:flex; flex-wrap:wrap; gap:0.45rem; margin-top:0.35rem;'>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_full_grid_table(rows: list[dict[str, object]]) -> None:
    st.markdown("### Full A1..D4 technical table")
    visible_columns = ["cell", "likely_change", "technical_interpretation", "support", "score", "confidence"]
    visible_rows = [{key: row.get(key, "") for key in visible_columns} for row in rows]
    st.dataframe(
        visible_rows,
        width="stretch",
        height=460,
        hide_index=True,
        column_config={
            "cell": st.column_config.TextColumn("cell", width="small"),
            "likely_change": st.column_config.TextColumn("likely_change", width="medium"),
            "technical_interpretation": st.column_config.TextColumn("technical_interpretation", width="large"),
            "support": st.column_config.TextColumn("support", width="medium"),
            "score": st.column_config.NumberColumn("score", width="small", format="%.1f"),
            "confidence": st.column_config.TextColumn("confidence", width="small"),
        },
    )
    object_rows = [
        {
            "cell": row.get("cell", ""),
            "objects_before": row.get("objects_before", ""),
            "objects_after": row.get("objects_after", ""),
        }
        for row in rows
    ]
    with st.expander("Optional object context by cell", expanded=False):
        st.dataframe(
            object_rows,
            width="stretch",
            height=320,
            hide_index=True,
            column_config={
                "cell": st.column_config.TextColumn("cell", width="small"),
                "objects_before": st.column_config.TextColumn("objects_before", width="large"),
                "objects_after": st.column_config.TextColumn("objects_after", width="large"),
            },
        )


def render_final_answer(
    scene_overview: str,
    rows: list[dict[str, object]],
    change_zoom_strip: np.ndarray,
    reasoning_text: str = "",
) -> None:
    st.subheader("Final Answer")
    with st.container(border=True):
        st.markdown("### Scene overview")
        safe_overview = html.escape(scene_overview)
        st.markdown(
            f"""
            <div style="
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                background: linear-gradient(135deg, rgba(45,52,54,0.88), rgba(24,28,31,0.88));
                line-height: 1.55;
                font-size: 1rem;
            ">
                {safe_overview}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if reasoning_text.strip():
            with st.expander("Model reasoning notes", expanded=False):
                st.caption(
                    "Separate reasoning trace returned by the selected backend. "
                    "The final table below remains the normalized user-facing answer."
                )
                st.code(reasoning_text.strip()[-6000:])
        render_full_grid_table(rows)
        st.markdown("### Top changed cells")
        st.image(
            change_zoom_strip,
            caption="Before/after zooms for the strongest changed cells. The strip is visual reference only and is not sent to the model.",
            width="stretch",
        )


def pad_bbox(
    bbox: dict[str, int],
    height: int,
    width: int,
    padding: int,
) -> dict[str, int]:
    return {
        "left": max(0, bbox["left"] - padding),
        "top": max(0, bbox["top"] - padding),
        "right": min(width - 1, bbox["right"] + padding),
        "bottom": min(height - 1, bbox["bottom"] + padding),
    }


source_mode = st.sidebar.radio("Image source", ["OSCD dataset", "Upload your own pair"])
model_dir = Path(st.sidebar.text_input("Semantic model", value=str(MASK2FORMER_SATELLITE_DIR)))
device_name = st.sidebar.selectbox("Semantic device", ["cpu", "mps", "cuda"], index=1)
enable_vlm = st.sidebar.checkbox("Enable model explanation", value=True)
show_live_trace = st.sidebar.checkbox("Show debug trace/details", value=False)
use_semantic_hints = st.sidebar.checkbox("Pass semantic hints into model", value=False)
model_presets = available_model_presets()
model_preset_names = list(model_presets.keys())
default_preset = "Qwen MLX (Apple Silicon default)" if "Qwen MLX (Apple Silicon default)" in model_presets else "gemma4:e4b (Ollama)"
selected_preset = st.sidebar.selectbox("Available model", model_preset_names, index=model_preset_names.index(default_preset))
selected_model_config = model_presets[selected_preset]
explanation_backend = selected_model_config["backend"]
runtime_backend = selected_model_config["runtime_backend"]
vlm_device_name = st.sidebar.selectbox("VLM device", ["cpu", "mps", "cuda"], index=1)
reasoning_profile = st.sidebar.selectbox(
    "Reasoning budget",
    options=list(REASONING_PROFILES.keys()),
    index=list(REASONING_PROFILES.keys()).index("efficient" if vlm_device_name == "mps" else "balanced"),
    format_func=lambda key: REASONING_PROFILES[key].label,
)
if runtime_backend == "hf_transformers_legacy":
    st.sidebar.warning("HF legacy debug is the slowest and most memory-heavy path on this Mac.")
elif runtime_backend == "mlx_vlm_qwen" and reasoning_profile == "deep":
    st.sidebar.info("Deep reasoning on MLX is more stable than HF/MPS, but still materially slower.")
elif runtime_backend == "ollama_gemma" and reasoning_profile == "deep":
    st.sidebar.info("Deep reasoning increases Ollama context and response length, so memory use will rise.")
vlm_model_name = selected_model_config["model_name"] if runtime_backend in {"mlx_vlm_qwen", "hf_transformers_legacy"} else default_vlm_model_name()
ollama_model_name = selected_model_config["model_name"] if explanation_backend == "Ollama text" else default_ollama_model_name()

st.sidebar.markdown("### Models")
st.sidebar.write(f"`mask2former-satellite`: {'yes' if model_dir.exists() else 'no'}")
if runtime_backend == "mlx_vlm_qwen":
    st.sidebar.write(f"`mlx qwen runtime`: {'yes' if mlx_model_ready(vlm_model_name) else 'download on first use'}")
elif runtime_backend == "hf_transformers_legacy" and Path(vlm_model_name).exists():
    st.sidebar.write(f"`hf qwen legacy`: {'yes' if vlm_ready(vlm_model_name) else 'incomplete'}")
    if not vlm_ready(vlm_model_name):
        st.sidebar.warning("Selected HF legacy VLM directory is incomplete. Run: `python scripts/download_vlm_models.py`")
if explanation_backend == "Ollama text":
    st.sidebar.write(f"`ollama {ollama_model_name}`: {'yes' if cached_ollama_model_available(ollama_model_name) else 'no'}")

if source_mode == "OSCD dataset" and not repo_ready:
    st.warning("OSCD dataset is not ready. Run: `python scripts/download_oscd.py`.")
    st.stop()

if source_mode == "OSCD dataset":
    repo = OSCDSceneRepository(data_root)
    city = st.sidebar.selectbox("City", repo.split_cities("all"))
    scene = repo.load_scene(city)
    pre_rgb = scene.pre_rgb
    post_rgb = scene.post_rgb
    scene_key = f"oscd:{city}"
else:
    before_file = st.sidebar.file_uploader("Before image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="before")
    after_file = st.sidebar.file_uploader("After image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="after")
    if before_file is None or after_file is None:
        st.info("Upload both images to run semantic segmentation.")
        st.stop()
    before_img = Image.open(before_file).convert("RGB")
    after_img = Image.open(after_file).convert("RGB")
    if before_img.size != after_img.size:
        after_img = after_img.resize(before_img.size)
        st.sidebar.warning("After image was resized to match Before image.")
    pre_rgb = np.asarray(before_img, dtype=np.float32) / 255.0
    post_rgb = np.asarray(after_img, dtype=np.float32) / 255.0
    scene_key = f"upload:{uploaded_file_identity(before_file)}|{uploaded_file_identity(after_file)}"

previous_scene_key = st.session_state.get("active_scene_key")
if previous_scene_key != scene_key:
    st.session_state.active_scene_key = scene_key
    st.session_state.active_crop_bbox = None
    st.session_state.crop_selector_nonce = st.session_state.get("crop_selector_nonce", 0) + 1

if not model_dir.exists():
    st.error(f"Semantic model directory not found: {model_dir}")
    st.info("Run: `python scripts/download_semantic_models.py`")
    st.stop()

st.subheader("Select Crop")
pre_preview = np.asarray(np.clip(pre_rgb * 255.0, 0, 255), dtype=np.uint8)
post_preview = np.asarray(np.clip(post_rgb * 255.0, 0, 255), dtype=np.uint8)
intro_cols = st.columns(2)
intro_cols[0].image(pre_preview, caption="Before image", width="stretch")
intro_cols[1].image(post_preview, caption="After image", width="stretch")

selector_key = f"crop-selector::{scene_key}::{st.session_state.get('crop_selector_nonce', 0)}"

img_pil = Image.fromarray(pre_preview)
box = st_cropper(
    img_pil,
    realtime_update=True,
    box_color='blue',
    return_type='box',
    key=selector_key,
)
st.caption("Resize the blue outline on the image above to select a region of interest.")

selected_bbox = None
if box and isinstance(box, dict) and box.get("width", 0) > 0 and box.get("height", 0) > 0:
    left = int(box["left"])
    top = int(box["top"])
    right = left + int(box["width"])
    bottom = top + int(box["height"])
    selected_bbox = {"left": left, "top": top, "right": right, "bottom": bottom}

if selected_bbox is not None:
    active_bbox_candidate = selected_bbox
    st.image(draw_bbox(pre_preview, active_bbox_candidate), caption="Target exact area", width=min(pre_preview.shape[1], 400))
    if st.button("Confirm and process crop", type="primary"):
        st.session_state.active_crop_bbox = active_bbox_candidate
        st.rerun()

if st.button("Clear active crop and reset selector"):
    st.session_state.active_crop_bbox = None
    st.session_state.crop_selector_nonce = st.session_state.get("crop_selector_nonce", 0) + 1
    st.rerun()

active_bbox = st.session_state.get("active_crop_bbox")
if active_bbox is None:
    st.stop()

st.subheader("Active Crop")
crop_before = crop_rgb(pre_rgb, active_bbox)
crop_after = crop_rgb(post_rgb, active_bbox)
pipeline_mode = "v2"
segmentation_runtime = load_segmentation_runtime(str(model_dir), device_name=device_name, backend_name="mask2former_openearthmap")
pipeline_v2 = LandChangePipelineV2(segmentation_runtime=segmentation_runtime)
selected_model_name = vlm_model_name if runtime_backend in {"mlx_vlm_qwen", "hf_transformers_legacy"} else ollama_model_name
runtime_details = model_runtime_details(
    explanation_backend=explanation_backend,
    runtime_backend=runtime_backend,
    selected_model_name=selected_model_name,
    vlm_device_name=vlm_device_name,
    reasoning_profile=reasoning_profile,
    use_semantic_hints=use_semantic_hints,
)
semantic_ran_this_pass = False
explanation_ran_this_pass = False

with st.spinner("Running semantic segmentation on the selected crop..."):
    pipeline_result = pipeline_v2.run(crop_before, crop_after, rows=4, cols=4)
    crop_before_result = pipeline_result.before_segmentation
    crop_after_result = pipeline_result.after_segmentation
    cell_packs = pipeline_result.cell_packs
semantic_ran_this_pass = True

crop_transitions = [
    item
    for item in summarize_transitions(crop_before_result.class_map, crop_after_result.class_map, top_k=12, id2label=crop_before_result.legend)
    if item["before"] != item["after"]
]
vlm_result = None

st.subheader("Selected Crop")
selected_cols = st.columns(2)
selected_cols[0].image(crop_before, caption="Crop before", width="stretch")
selected_cols[1].image(crop_after, caption="Crop after", width="stretch")
with st.expander("Diagnostic semantic maps (not primary model input)", expanded=False):
    st.caption(
        "These maps come from the semantic segmentation model. They are diagnostic only. "
        "When `Pass semantic hints into model` is off, semantic labels are not sent to the VLM."
    )
    map_cols = st.columns(2)
    map_cols[0].image(crop_before_result.color_map, caption="T1 diagnostic semantic map", width="stretch")
    map_cols[1].image(crop_after_result.color_map, caption="T2 diagnostic semantic map", width="stretch")
    st.markdown("**Color legend**")
    render_semantic_color_legend(crop_before_result.legend)

change_evidence = analyze_change_cells(crop_before, crop_after, grid_size=4)
change_guide_image = build_change_guide_image(crop_before, crop_after, change_evidence)
change_zoom_strip = build_change_zoom_strip(crop_before, crop_after, change_evidence)
model_auxiliary_images = [change_guide_image]
base_cell_rows = build_cell_report_rows_from_segmentation(cell_packs)
base_scene_overview = build_scene_overview_from_rows(base_cell_rows)
visual_change_context = build_pairwise_visual_context(
    packs=cell_packs,
    before_segmentation=crop_before_result,
    after_segmentation=crop_after_result,
    rows=base_cell_rows,
)

progress_cb = None
if enable_vlm:
    progress_cb = create_trace_collector() if show_live_trace else None
    semantic_context = ""
    if use_semantic_hints:
        semantic_context = build_vlm_semantic_context(
            crop_before_result.class_map,
            crop_after_result.class_map,
            crop_before_result.legend,
            crop_transitions,
        )
        semantic_context = trim_semantic_context(semantic_context, REASONING_PROFILES[reasoning_profile].semantic_context_chars)
    if runtime_backend in {"mlx_vlm_qwen", "hf_transformers_legacy"}:
        model_ok = mlx_model_ready(vlm_model_name) if runtime_backend == "mlx_vlm_qwen" else vlm_ready(vlm_model_name)
        if not model_ok:
            st.warning("Selected Qwen runtime is not ready. Download the MLX model or use the local fallback preset.")
            vlm_result = None
        else:
            try:
                with st.spinner(f"Running {explanation_backend} on the selected crop..."):
                    vlm_model = load_vlm(vlm_model_name, vlm_device_name, reasoning_profile, runtime_backend, VLM_RUNTIME_API_VERSION)
                    try:
                        vlm_result = vlm_model.explain(
                            crop_before,
                            crop_after,
                            semantic_context=semantic_context,
                            visual_context=visual_change_context,
                            auxiliary_images=model_auxiliary_images,
                            response_language="English",
                            progress_cb=progress_cb,
                        )
                    finally:
                        vlm_model.unload()
                    explanation_ran_this_pass = True
            except Exception as exc:
                vlm_result = None
                st.error(f"{explanation_backend} backend failed: {type(exc).__name__}: {exc}")
    else:
        if not cached_ollama_model_available(ollama_model_name):
            vlm_result = None
            st.warning(f"Ollama model `{ollama_model_name}` is not installed. Run: `ollama pull {ollama_model_name}`")
        else:
            try:
                with st.spinner("Running Ollama semantic explainer on the selected crop..."):
                    ollama_explainer = load_ollama_explainer(ollama_model_name, reasoning_profile, VLM_RUNTIME_API_VERSION)
                    vlm_result = ollama_explainer.explain(
                        before_crop=crop_before,
                        after_crop=crop_after,
                        semantic_context=semantic_context,
                        visual_context=visual_change_context,
                        auxiliary_images=model_auxiliary_images,
                        response_language="English",
                        progress_cb=progress_cb,
                    )
                    explanation_ran_this_pass = True
            except Exception as exc:
                vlm_result = None
                st.error(f"Ollama backend failed: {type(exc).__name__}: {exc}")

    if vlm_result is not None:
        final_cell_rows = base_cell_rows
        final_scene_overview = normalize_model_scene_overview(vlm_result.parsed) or base_scene_overview

        render_final_answer(final_scene_overview, final_cell_rows, change_zoom_strip, vlm_result.reasoning_text)

    else:
        render_final_answer(base_scene_overview, base_cell_rows, change_zoom_strip)
else:
    vlm_result = None
    render_final_answer(base_scene_overview, base_cell_rows, change_zoom_strip)

if show_live_trace:
    stage_rows = [
        {"stage": "semantic segmentation", "status": "ran now" if semantic_ran_this_pass else "not run"},
        {
            "stage": "model explanation",
            "status": (
                "disabled"
                if not enable_vlm
                else "ran now"
                if explanation_ran_this_pass
                else "waiting or unavailable"
            ),
        },
    ]
    deterministic_rows = build_deterministic_breakdown(
        crop_before=crop_before,
        crop_after=crop_after,
        crop_before_result=crop_before_result,
        crop_after_result=crop_after_result,
        crop_transitions=crop_transitions,
    )
    with st.expander("Debug: Runtime, Evidence, and Semantic Diagnostics", expanded=False):
        debug_state = getattr(progress_cb, "debug_state", None) if progress_cb is not None else None
        if debug_state:
            st.markdown("**Live model trace**")
            phase_lines = debug_state["phase_lines"]
            st.code("\n".join(phase_lines[-40:]) or "No trace events recorded yet.")
            thinking_text = debug_state["thinking_text"]()
            content_text = debug_state["content_text"]()
            if thinking_text.strip():
                st.markdown("**Streaming reasoning/debug text**")
                st.code(thinking_text[-6000:])
            if content_text.strip():
                st.markdown("**Streaming final/debug text**")
                st.code(content_text[-6000:])
        if vlm_result is not None:
            parsed = vlm_result.parsed or {}
            debug_model_overview = normalize_model_scene_overview(parsed)
            st.markdown("**Model output acceptance**")
            st.dataframe(
                [
                    {"field": "scene overview accepted", "value": "yes" if debug_model_overview else "no"},
                    {"field": "reasoning complete", "value": "yes" if vlm_result.reasoning_complete else "no"},
                ],
                width="stretch",
                hide_index=True,
            )
            if (vlm_result.reasoning_text or "").strip():
                st.markdown("**Reasoning trace**")
                st.code(vlm_result.reasoning_text)
            st.markdown("**Raw final output**")
            st.code(vlm_result.final_text or vlm_result.raw_text)
            if vlm_result.phase_trace:
                st.markdown("**Backend phase trace**")
                st.code("\n".join(vlm_result.phase_trace))
        st.markdown("**Runtime breakdown**")
        st.dataframe(
            [
                {"field": "model family", "value": runtime_details["family"]},
                {"field": "backend", "value": runtime_details["backend"]},
                {"field": "input mode", "value": runtime_details["input_mode"]},
                {"field": "model", "value": runtime_details["model"]},
                {"field": "source", "value": runtime_details["source"]},
                {"field": "resolved path", "value": runtime_details["resolved_path"]},
                {"field": "device", "value": runtime_details["device"]},
                {"field": "reasoning profile", "value": runtime_details["reasoning_profile"]},
                {"field": "thinking mode", "value": runtime_details["thinking"]},
                {"field": "context window", "value": runtime_details["context_window"]},
                {"field": "kv cache mode", "value": runtime_details["kv_cache"]},
                {"field": "images sent to model", "value": "before crop, after crop, change-guide heatmap"},
                {"field": "not sent to model as images", "value": "semantic maps, top-cell zoom strip"},
                {"field": "primary images sent", "value": runtime_details["primary_images_sent"]},
                {"field": "auxiliary images sent", "value": runtime_details["auxiliary_images_sent"]},
                {"field": "semantic text hints sent", "value": runtime_details["semantic_text_hints_sent"]},
                {"field": "semantic maps sent as images", "value": runtime_details["semantic_maps_sent_as_images"]},
                {"field": "inference result cache", "value": "disabled; fresh run data is recomputed"},
                {"field": "pipeline mode", "value": pipeline_mode},
                {"field": "grid cell packs prepared", "value": str(len(cell_packs))},
                {"field": "active bbox", "value": json.dumps(active_bbox)},
            ],
            width="stretch",
            hide_index=True,
        )
        st.dataframe(stage_rows, width="stretch", hide_index=True)
        st.markdown("**Deterministic technical evidence**")
        st.dataframe(deterministic_rows, width="stretch", hide_index=True)
        st.image(change_guide_image, caption="Deterministic change-guide heatmap", width="stretch")
        st.image(change_zoom_strip, caption="Top changed cell zoom panels", width="stretch")
        st.markdown("**Pairwise context passed to model**")
        st.code(visual_change_context)
        st.markdown("**Semantic diagnostics**")
        st.dataframe(compact_summary(crop_before_result.label_summary), width="stretch", hide_index=True)
        st.dataframe(compact_summary(crop_after_result.label_summary), width="stretch", hide_index=True)
        st.dataframe(crop_transitions, width="stretch", hide_index=True)
        st.code(
            json.dumps(
                {
                    "before_summary": compact_summary(crop_before_result.label_summary),
                    "after_summary": compact_summary(crop_after_result.label_summary),
                    "transitions": crop_transitions,
                    "active_bbox": active_bbox,
                    "selected_crop_shape": list(crop_before.shape),
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )
