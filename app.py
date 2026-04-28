from __future__ import annotations

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
from land_change_detection.dinov3_features import (
    DINO_V3_VITL16_SAT,
    DINO_V3_VITL16_SAT_DIR,
    DINOv3FeatureEncoder,
    model_dir_complete as feature_model_dir_complete,
)
from land_change_detection.legacy.legacy_visual_context import (
    analyze_change_cells,
    build_change_guide_image,
    build_change_zoom_strip,
    build_deterministic_breakdown,
    build_visual_change_context,
)
from land_change_detection.legacy.heuristic_change_interpreter import (
    build_cell_report_rows as build_visual_cell_report_rows,
    build_scene_overview as build_visual_scene_overview,
)
from land_change_detection.pipeline_v2 import LandChangePipelineV2
from land_change_detection.semantic_hf import (
    DEFAULT_CLASS_COLORS,
    MASK2FORMER_SATELLITE_DIR,
    crop_rgb,
    draw_bbox,
)
from land_change_detection.remote_sensing_vlm import (
    EARTHDIAL_RGB,
    EARTHDIAL_RGB_DIR,
    GEMMA4_E4B_OLLAMA,
    QWEN3_5_VL_0_8B_MLX_4BIT,
    QWEN3_5_VL_0_8B_MLX_4BIT_DIR,
    QWEN3_VL_4B_THINKING,
    QWEN3_VL_4B_THINKING_MLX_3BIT,
    QWEN3_VL_4B_THINKING_MLX_3BIT_DIR,
    QWEN3_VL_4B_THINKING_DIR,
    REMOTE_SENSING_QWEN2_5_VL_3B,
    REMOTE_SENSING_QWEN2_5_VL_3B_DIR,
    REMOTE_SENSING_QWEN2_VL_2B,
    REMOTE_SENSING_QWEN2_VL_2B_DIR,
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
st.caption("Mouse-selected crop -> VLM visual comparison -> plain-English land-surface change report")

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
def load_dino_feature_encoder(model_name: str, device_name: str) -> DINOv3FeatureEncoder:
    return DINOv3FeatureEncoder(model_name_or_path=model_name, device=device_name)


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


def available_model_presets(show_debug: bool = False) -> dict[str, dict[str, str]]:
    presets: dict[str, dict[str, str]] = {}
    presets["gemma4:e4b (Ollama)"] = {
        "backend": "Ollama vision",
        "runtime_backend": "ollama_gemma",
        "model_name": GEMMA4_E4B_OLLAMA,
    }
    if model_dir_complete(EARTHDIAL_RGB_DIR):
        presets["EarthDial 4B RGB (HF local)"] = {
            "backend": "EarthDial VLM",
            "runtime_backend": "hf_transformers_legacy",
            "model_name": str(EARTHDIAL_RGB_DIR),
        }
    else:
        presets["EarthDial 4B RGB (download from HF)"] = {
            "backend": "EarthDial VLM",
            "runtime_backend": "hf_transformers_legacy",
            "model_name": EARTHDIAL_RGB,
        }
    if show_debug:
        presets["Qwen MLX (debug, heavy on Mac)"] = {
            "backend": "MLX vision-language",
            "runtime_backend": "mlx_vlm_qwen",
            "model_name": preferred_mlx_qwen_model(),
        }
        if has_complete_local_vlm():
            presets["Qwen3-VL-4B-Thinking (HF debug, very heavy)"] = {
                "backend": "HF legacy debug",
                "runtime_backend": "hf_transformers_legacy",
                "model_name": str(QWEN3_VL_4B_THINKING_DIR),
            }
        if model_dir_complete(REMOTE_SENSING_QWEN2_VL_2B_DIR):
            presets["Remote-sensing Qwen2-VL-2B (debug, memory-heavy)"] = {
                "backend": "HF remote-sensing VLM",
                "runtime_backend": "hf_transformers_legacy",
                "model_name": str(REMOTE_SENSING_QWEN2_VL_2B_DIR),
            }
        if model_dir_complete(REMOTE_SENSING_QWEN2_5_VL_3B_DIR):
            presets["Remote-sensing Qwen2.5-VL-3B (debug, very heavy)"] = {
                "backend": "HF remote-sensing VLM",
                "runtime_backend": "hf_transformers_legacy",
                "model_name": str(REMOTE_SENSING_QWEN2_5_VL_3B_DIR),
            }
    return presets


def available_semantic_model_presets() -> dict[str, dict[str, str]]:
    return {
        "Mask2Former satellite / OpenEarthMap classes": {
            "backend": "mask2former_openearthmap",
            "model_name": str(MASK2FORMER_SATELLITE_DIR),
            "kind": "semantic_classes",
        },
    }


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
            "input_mode": "before/after crop images + A1..D4 contact sheet + secondary change-guide",
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
            "auxiliary_images_sent": "2: A1..D4 contact sheet, change-guide heatmap",
            "semantic_text_hints_sent": "yes" if use_semantic_hints else "no",
            "semantic_maps_sent_as_images": "no",
        }
    if runtime_backend == "hf_transformers_legacy":
        return {
            "family": "VLM",
            "backend": explanation_backend,
            "input_mode": "before/after crop images + A1..D4 contact sheet + secondary change-guide",
            "source": "local directory" if model_path.exists() else "Hugging Face repo id",
            "model": display_model_name(selected_model_name),
            "resolved_path": str(model_path.resolve()) if model_path.exists() else selected_model_name,
            "device": vlm_device_name,
            "reasoning_profile": profile.label,
            "thinking": "enabled" if profile.enable_thinking else "disabled",
            "context_window": "bounded prompt; reduce image size/reasoning if MPS memory is tight",
            "kv_cache": "off on mps" if not profile.use_cache_on_mps else "on on mps",
            "semantic_hints": "enabled" if use_semantic_hints else "disabled",
            "primary_images_sent": "2: before crop, after crop",
            "auxiliary_images_sent": "2: A1..D4 contact sheet, change-guide heatmap",
            "semantic_text_hints_sent": "yes" if use_semantic_hints else "no",
            "semantic_maps_sent_as_images": "no",
        }
    return {
        "family": "VLM",
        "backend": "Ollama vision",
        "input_mode": "before/after crop images + A1..D4 contact sheet + secondary change-guide",
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
        "auxiliary_images_sent": "2: A1..D4 contact sheet, change-guide heatmap",
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


def build_cell_contact_sheet(before_rgb: np.ndarray, after_rgb: np.ndarray, grid_size: int = 4, panel_size: int = 150) -> np.ndarray:
    before_arr = np.asarray(np.clip(before_rgb, 0, 255), dtype=np.float32)
    after_arr = np.asarray(np.clip(after_rgb, 0, 255), dtype=np.float32)
    if before_arr.max() <= 1.5:
        before_arr = before_arr * 255.0
    if after_arr.max() <= 1.5:
        after_arr = after_arr * 255.0
    before_arr = before_arr.astype(np.uint8)
    after_arr = after_arr.astype(np.uint8)

    height, width = before_arr.shape[:2]
    row_bounds = np.linspace(0, height, grid_size + 1, dtype=int)
    col_bounds = np.linspace(0, width, grid_size + 1, dtype=int)
    pair_gap = 6
    cell_gap = 10
    label_h = 24
    pair_w = panel_size * 2 + pair_gap
    cell_h = panel_size + label_h
    canvas_w = grid_size * pair_w + (grid_size - 1) * cell_gap
    canvas_h = grid_size * cell_h + (grid_size - 1) * cell_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(10, 12, 16))
    draw = ImageDraw.Draw(canvas)

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            top = int(row_bounds[row_idx])
            bottom = int(row_bounds[row_idx + 1])
            left = int(col_bounds[col_idx])
            right = int(col_bounds[col_idx + 1])
            cell_id = f"{chr(ord('A') + row_idx)}{col_idx + 1}"
            x = col_idx * (pair_w + cell_gap)
            y = row_idx * (cell_h + cell_gap)
            before_tile = ImageOps.fit(Image.fromarray(before_arr[top:bottom, left:right]), (panel_size, panel_size), method=Image.Resampling.BICUBIC)
            after_tile = ImageOps.fit(Image.fromarray(after_arr[top:bottom, left:right]), (panel_size, panel_size), method=Image.Resampling.BICUBIC)
            draw.rectangle((x, y, x + pair_w - 1, y + cell_h - 1), outline=(70, 76, 88), width=1)
            draw.text((x + 6, y + 5), f"{cell_id} BEFORE", fill=(245, 245, 245))
            draw.text((x + panel_size + pair_gap + 6, y + 5), f"{cell_id} AFTER", fill=(245, 245, 245))
            canvas.paste(before_tile, (x, y + label_h))
            canvas.paste(after_tile, (x + panel_size + pair_gap, y + label_h))
    return np.asarray(canvas)


def render_debug_semantic_diagnostics(crop_before_result, crop_after_result, crop_transitions: list[dict]) -> None:
    st.markdown("**Mask2Former debug semantic maps**")
    st.caption(
        "Debug-only OpenEarthMap semantic segmentation from `artifacts/models/semantic/mask2former-satellite`. "
        "These labels are not used for the normal VLM answer."
    )
    map_cols = st.columns(2)
    map_cols[0].image(crop_before_result.color_map, caption="T1 debug semantic map", width="stretch")
    map_cols[1].image(crop_after_result.color_map, caption="T2 debug semantic map", width="stretch")
    st.dataframe(crop_transitions, width="stretch", hide_index=True)


def render_semantic_color_legend(id2label: dict[int, str], class_colors: dict[str, str] | None = None) -> None:
    colors = class_colors or DEFAULT_CLASS_COLORS
    cols = st.columns(3)
    for idx, (class_id, label) in enumerate(sorted(id2label.items())):
        color = colors.get(str(label), "#95a5a6")
        with cols[idx % len(cols)]:
            st.color_picker(f"{class_id}: {label}", value=color, disabled=True, key=f"legend-{class_id}-{label}")


def render_surface_segmentation(crop_before_result, crop_after_result, crop_transitions: list[dict], model_label: str) -> None:
    st.subheader("Mask2Former surface segmentation")
    st.caption(
        f"Semantic surface maps from `{model_label}`. These maps are segmentation evidence; "
        "the final human explanation below is produced separately by the VLM from the before/after images."
    )
    map_cols = st.columns(2)
    map_cols[0].image(crop_before_result.color_map, caption="T1 surface segmentation", width="stretch")
    map_cols[1].image(crop_after_result.color_map, caption="T2 surface segmentation", width="stretch")
    with st.expander("Surface class colors", expanded=False):
        render_semantic_color_legend(crop_before_result.legend, crop_before_result.class_colors)
    if crop_transitions:
        with st.expander("Surface segmentation transitions", expanded=False):
            st.dataframe(
                crop_transitions[:10],
                width="stretch",
                hide_index=True,
                column_config={
                    "before": st.column_config.TextColumn("before", width="medium"),
                    "after": st.column_config.TextColumn("after", width="medium"),
                    "pixels": st.column_config.NumberColumn("pixels", width="small"),
                    "percent": st.column_config.NumberColumn("percent", width="small", format="%.2f"),
                },
            )


def render_dino_feature_diagnostics(feature_rows: list[dict[str, object]], model_name: str) -> None:
    st.subheader("DINOv3 feature-change diagnostics")
    st.caption(
        f"Feature-based change ranking from `{model_name}`. This is not segmentation and not natural-language VLM output; "
        "it measures how much the visual semantics of each cell shift between BEFORE and AFTER."
    )
    st.dataframe(
        feature_rows,
        width="stretch",
        hide_index=True,
        column_config={
            "cell": st.column_config.TextColumn("cell", width="small"),
            "cosine_distance": st.column_config.NumberColumn("cosine distance", width="small", format="%.3f"),
            "l2_distance": st.column_config.NumberColumn("l2 distance", width="small", format="%.3f"),
            "interpretation": st.column_config.TextColumn("interpretation", width="large"),
        },
    )


def _render_main_changes(changes: list[object]) -> None:
    clean_changes = [str(item).strip() for item in changes if str(item).strip()]
    if not clean_changes:
        return
    st.markdown("**Main visible changes**")
    for item in clean_changes[:5]:
        st.markdown(f"- {item}")


def build_visual_fallback_summary(change_evidence) -> dict:
    rows = build_visual_cell_report_rows(change_evidence)
    sorted_rows = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    main_changes = []
    for row in sorted_rows:
        text = str(row.get("technical_interpretation") or row.get("likely_change") or "").strip()
        if text and text not in main_changes:
            main_changes.append(text)
        if len(main_changes) >= 4:
            break
    observations = [
        {
            "cell": str(row.get("cell", "")),
            "observation": (
                f"{row.get('technical_interpretation', '')} "
                f"Visible evidence: before {row.get('objects_before', '')}; after {row.get('objects_after', '')}. "
                f"Support: {row.get('support', '')}."
            ).strip(),
            "confidence": str(row.get("confidence", "uncertain")),
        }
        for row in rows
    ]
    return {
        "scene_overview": build_visual_scene_overview(change_evidence),
        "before_summary": "Before, the selected crop is summarized from visible roads, buildings, exposed ground, compact surfaces, and vegetation cues.",
        "after_summary": "After, the selected crop is compared cell by cell for new roads or tracks, grading, excavation, construction-like surfaces, building-like additions, and stable areas.",
        "main_changes": main_changes,
        "cell_observations": observations,
    }


def render_vlm_summary(parsed: dict | None, fallback_summary: dict | None = None) -> None:
    parsed = parsed or {}
    scene_overview = normalize_model_scene_overview(parsed)
    before_summary = str(parsed.get("before_summary", "")).strip()
    after_summary = str(parsed.get("after_summary", "")).strip()
    main_changes = parsed.get("main_changes") if isinstance(parsed.get("main_changes"), list) else []
    cell_observations = parsed.get("cell_observations") if isinstance(parsed.get("cell_observations"), list) else []

    used_fallback = False
    if (not scene_overview or not cell_observations) and fallback_summary:
        used_fallback = True
        scene_overview = normalize_model_scene_overview(fallback_summary)
        before_summary = str(fallback_summary.get("before_summary", "")).strip()
        after_summary = str(fallback_summary.get("after_summary", "")).strip()
        main_changes = fallback_summary.get("main_changes") if isinstance(fallback_summary.get("main_changes"), list) else []
        cell_observations = fallback_summary.get("cell_observations") if isinstance(fallback_summary.get("cell_observations"), list) else []

    st.subheader("Gemma visual interpretation" if parsed and not used_fallback else "Visual interpretation")
    if not parsed or not scene_overview:
        st.warning(
            "The model did not return a complete user-facing analysis. Try a smaller reasoning budget or another VLM preset."
        )
        return
    if used_fallback:
        st.info(
            "The selected VLM did not return a complete structured answer. This is a non-model heuristic fallback from "
            "deterministic before/after visual measurements, not a VLM conclusion and not a Mask2Former class-label answer."
        )
    st.write(scene_overview)

    if before_summary or after_summary:
        before_col, after_col = st.columns(2)
        before_col.markdown("**Before**")
        before_col.write(before_summary or "No separate before summary returned.")
        after_col.markdown("**After**")
        after_col.write(after_summary or "No separate after summary returned.")

    _render_main_changes(main_changes)

    if cell_observations:
        st.markdown("**Cell observations**")
        st.dataframe(
            cell_observations,
            width="stretch",
            hide_index=True,
            column_config={
                "cell": st.column_config.TextColumn("cell", width="small"),
                "observation": st.column_config.TextColumn("observation", width="large"),
                "confidence": st.column_config.TextColumn("confidence", width="small"),
            },
        )
    else:
        st.info("The VLM did not return complete A1..D4 cell observations.")


def render_final_answer(
    parsed: dict | None,
    change_zoom_strip: np.ndarray,
    fallback_summary: dict | None = None,
    reasoning_text: str = "",
    show_debug: bool = False,
) -> None:
    st.subheader("Analysis output")
    with st.container(border=True):
        render_vlm_summary(parsed, fallback_summary=fallback_summary)
        if show_debug and reasoning_text.strip():
            with st.expander("Model reasoning notes", expanded=False):
                st.caption(
                    "Separate reasoning trace returned by the selected backend. "
                    "This is hidden unless debug mode is enabled."
                )
                st.code(reasoning_text.strip()[-6000:])
        st.markdown("### Top changed cell zooms")
        st.image(
            change_zoom_strip,
            caption="Before/after zooms for the strongest pixel-change cells. The strip is visual reference only.",
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
semantic_model_presets = available_semantic_model_presets()
semantic_model_preset_names = list(semantic_model_presets.keys())
selected_semantic_model_preset = st.sidebar.selectbox("Semantic model", semantic_model_preset_names, index=0)
selected_semantic_model_config = semantic_model_presets[selected_semantic_model_preset]
semantic_backend_name = selected_semantic_model_config["backend"]
semantic_model_name = selected_semantic_model_config["model_name"]
model_dir = Path(semantic_model_name)
device_name = st.sidebar.selectbox("Semantic device", ["cpu", "mps", "cuda"], index=1)
enable_vlm = st.sidebar.checkbox("Enable model explanation", value=True)
show_live_trace = st.sidebar.checkbox("Show debug trace/details", value=False)
use_semantic_hints = False
model_presets = available_model_presets(show_debug=show_live_trace)
model_preset_names = list(model_presets.keys())
default_preset = "gemma4:e4b (Ollama)" if "gemma4:e4b (Ollama)" in model_presets else model_preset_names[0]
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
if explanation_backend == "EarthDial VLM" and reasoning_profile != "efficient":
    st.sidebar.info("EarthDial uses the Efficient profile in this app to reduce local memory pressure.")
    reasoning_profile = "efficient"
if runtime_backend == "hf_transformers_legacy":
    st.sidebar.warning("HF legacy debug is the slowest and most memory-heavy path on this Mac.")
elif runtime_backend == "mlx_vlm_qwen" and reasoning_profile == "deep":
    st.sidebar.info("Deep reasoning on MLX is more stable than HF/MPS, but still materially slower.")
elif runtime_backend == "ollama_gemma" and reasoning_profile == "deep":
    st.sidebar.info("Deep reasoning increases Ollama context and response length, so memory use will rise.")
vlm_model_name = selected_model_config["model_name"] if runtime_backend in {"mlx_vlm_qwen", "hf_transformers_legacy"} else default_vlm_model_name()
ollama_model_name = selected_model_config["model_name"] if explanation_backend == "Ollama vision" else default_ollama_model_name()

st.sidebar.markdown("### Models")
st.sidebar.write(f"`mask2former-satellite`: {'yes' if model_dir.exists() else 'no'}")
if show_live_trace:
    st.sidebar.write(f"`dinov3 SAT debug`: {'yes' if feature_model_dir_complete(DINO_V3_VITL16_SAT_DIR) else 'download on first use'}")
if runtime_backend == "mlx_vlm_qwen":
    st.sidebar.write(f"`mlx qwen runtime`: {'yes' if mlx_model_ready(vlm_model_name) else 'download on first use'}")
elif runtime_backend == "hf_transformers_legacy" and Path(vlm_model_name).exists():
    st.sidebar.write(f"`hf qwen legacy`: {'yes' if vlm_ready(vlm_model_name) else 'incomplete'}")
    if not vlm_ready(vlm_model_name):
        st.sidebar.warning("Selected HF legacy VLM directory is incomplete. Run: `python scripts/download_vlm_models.py`")
if explanation_backend == "Ollama vision":
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
        st.info("Upload both images to run VLM visual change analysis.")
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

if show_live_trace and semantic_backend_name == "mask2former_openearthmap" and not model_dir.exists():
    st.sidebar.warning(f"Debug semantic model directory not found: {model_dir}")

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
pipeline_mode = "vlm_first"
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
crop_before_result = None
crop_after_result = None
cell_packs = []
crop_transitions: list[dict] = []
dino_feature_rows: list[dict[str, object]] = []

if model_dir.exists():
    with st.spinner("Running Mask2Former surface segmentation on the selected crop..."):
        segmentation_runtime = load_segmentation_runtime(str(model_dir), device_name=device_name, backend_name="mask2former_openearthmap")
        pipeline_v2 = LandChangePipelineV2(segmentation_runtime=segmentation_runtime)
        pipeline_result = pipeline_v2.run(crop_before, crop_after, rows=4, cols=4)
        crop_before_result = pipeline_result.before_segmentation
        crop_after_result = pipeline_result.after_segmentation
        cell_packs = pipeline_result.cell_packs
        crop_transitions = [
            item
            for item in summarize_transitions(crop_before_result.class_map, crop_after_result.class_map, top_k=12, id2label=crop_before_result.legend)
            if item["before"] != item["after"]
        ]
        semantic_ran_this_pass = True
else:
    st.warning(f"Semantic model directory not found: `{model_dir}`. VLM analysis can run, but surface segmentation maps are unavailable.")

if show_live_trace:
    try:
        with st.spinner("Running DINOv3 SAT feature diagnostics on the selected crop..."):
            dino_encoder = load_dino_feature_encoder(DINO_V3_VITL16_SAT, device_name)
            dino_feature_rows = [
                {
                    "cell": row.cell,
                    "cosine_distance": row.cosine_distance,
                    "l2_distance": row.l2_distance,
                    "interpretation": row.interpretation,
                }
                for row in dino_encoder.analyze_grid(crop_before, crop_after, grid_size=4)
            ]
    except Exception as exc:
        dino_feature_rows = []
        st.warning(f"DINOv3 SAT feature diagnostics failed: {type(exc).__name__}: {exc}")
vlm_result = None

st.subheader("Selected Crop")
selected_cols = st.columns(2)
selected_cols[0].image(crop_before, caption="Crop before", width="stretch")
selected_cols[1].image(crop_after, caption="Crop after", width="stretch")
if crop_before_result is not None and crop_after_result is not None:
    render_surface_segmentation(crop_before_result, crop_after_result, crop_transitions, selected_semantic_model_preset)
if dino_feature_rows:
    render_dino_feature_diagnostics(dino_feature_rows, DINO_V3_VITL16_SAT)

change_evidence = analyze_change_cells(crop_before, crop_after, grid_size=4)
change_guide_image = build_change_guide_image(crop_before, crop_after, change_evidence)
change_zoom_strip = build_change_zoom_strip(crop_before, crop_after, change_evidence)
cell_contact_sheet = build_cell_contact_sheet(crop_before, crop_after, grid_size=4)
fallback_visual_summary = build_visual_fallback_summary(change_evidence)
st.subheader("4x4 visual comparison grid")
st.caption("Each panel shows the same cell before and after. This grid is sent to the VLM so it can describe A1..D4 directly.")
st.image(cell_contact_sheet, caption="A1..D4 before/after contact sheet", width="stretch")
model_auxiliary_images = [cell_contact_sheet] if explanation_backend == "EarthDial VLM" else [cell_contact_sheet, change_guide_image]
visual_change_context = build_visual_change_context(change_evidence)

progress_cb = None
if enable_vlm:
    progress_cb = create_trace_collector() if show_live_trace else None
    semantic_context = ""
    if show_live_trace and use_semantic_hints and crop_before_result is not None and crop_after_result is not None:
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
                with st.spinner("Running Ollama visual explainer on the selected crop..."):
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
        render_final_answer(
            parsed=vlm_result.parsed,
            change_zoom_strip=change_zoom_strip,
            fallback_summary=fallback_visual_summary,
            reasoning_text=vlm_result.reasoning_text,
            show_debug=show_live_trace,
        )

    else:
        render_final_answer(None, change_zoom_strip, fallback_summary=fallback_visual_summary, show_debug=show_live_trace)
else:
    vlm_result = None
    render_final_answer(None, change_zoom_strip, fallback_summary=fallback_visual_summary, show_debug=show_live_trace)

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
    deterministic_rows = []
    if crop_before_result is not None and crop_after_result is not None:
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
                {"field": "images sent to model", "value": "before crop, after crop, A1..D4 contact sheet, change-guide heatmap"},
                {"field": "not sent to model as images", "value": "semantic maps, top-cell zoom strip"},
                {"field": "primary images sent", "value": runtime_details["primary_images_sent"]},
                {"field": "auxiliary images sent", "value": runtime_details["auxiliary_images_sent"]},
                {"field": "semantic text hints sent", "value": runtime_details["semantic_text_hints_sent"]},
                {"field": "semantic maps sent as images", "value": runtime_details["semantic_maps_sent_as_images"]},
                {"field": "inference result cache", "value": "disabled; fresh run data is recomputed"},
                {"field": "pipeline mode", "value": pipeline_mode},
                {"field": "debug semantic cell packs prepared", "value": str(len(cell_packs))},
                {"field": "active bbox", "value": json.dumps(active_bbox)},
            ],
            width="stretch",
            hide_index=True,
        )
        st.dataframe(stage_rows, width="stretch", hide_index=True)
        if deterministic_rows:
            st.markdown("**Deterministic technical evidence**")
            st.dataframe(deterministic_rows, width="stretch", hide_index=True)
        st.image(change_guide_image, caption="Deterministic change-guide heatmap", width="stretch")
        st.image(change_zoom_strip, caption="Top changed cell zoom panels", width="stretch")
        st.markdown("**Visual context passed to model**")
        st.code(visual_change_context)
        if crop_before_result is not None and crop_after_result is not None:
            render_debug_semantic_diagnostics(crop_before_result, crop_after_result, crop_transitions)
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
        else:
            st.info("Mask2Former semantic diagnostics did not run. Enable debug with a ready semantic model to inspect them.")
