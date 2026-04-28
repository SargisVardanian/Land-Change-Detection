from __future__ import annotations

import json
import os
import queue
import re
import site
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms as T
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer


QWEN3_VL_4B_THINKING = "Qwen/Qwen3-VL-4B-Thinking"
QWEN3_VL_4B_THINKING_DIR = Path("artifacts/models/vlm/Qwen3-VL-4B-Thinking")
QWEN3_VL_4B_THINKING_MLX_3BIT = "mlx-community/Qwen3-VL-4B-Thinking-3bit"
QWEN3_VL_4B_THINKING_MLX_3BIT_DIR = Path("artifacts/models/mlx-community__Qwen3-VL-4B-Thinking-3bit")
QWEN3_5_VL_0_8B_MLX_4BIT = "mlx-community/Qwen3.5-0.8B-4bit"
QWEN3_5_VL_0_8B_MLX_4BIT_DIR = Path("artifacts/models/mlx-community__Qwen3.5-0.8B-4bit")
GEMMA4_E4B_OLLAMA = "gemma4:e4b"
REMOTE_SENSING_QWEN2_5_VL_3B = "AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct"
REMOTE_SENSING_QWEN2_5_VL_3B_DIR = Path("artifacts/models/vlm/remote-sensing-Qwen2.5-VL-3B-Instruct")
REMOTE_SENSING_QWEN2_VL_2B = "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
REMOTE_SENSING_QWEN2_VL_2B_DIR = Path("artifacts/models/vlm/remote-sensing-Qwen2-VL-2B-Instruct")
EARTHDIAL_RGB = "akshaydudhane/EarthDial_4B_RGB"
EARTHDIAL_RGB_DIR = Path("artifacts/models/vlm/EarthDial_4B_RGB")
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC RGB, got shape={arr.shape}")
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ensure_earthdial_import_path() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    site_src = Path(site.getsitepackages()[0]) / "src"
    if site_src.exists():
        site_src_str = str(site_src)
        if site_src_str not in sys.path:
            sys.path.insert(0, site_src_str)


def _patch_transformers_compat() -> None:
    try:
        import transformers
        from transformers import Cache
        from transformers import utils as transformers_utils
    except Exception:
        return
    if not hasattr(transformers, "EncoderDecoderCache"):
        transformers.EncoderDecoderCache = Cache
    if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False


def _composite_triptych(before: np.ndarray, after: np.ndarray, overlay: np.ndarray, panel_size: int = 448) -> Image.Image:
    panels = []
    for arr in (before, after, overlay):
        img = Image.fromarray(_to_uint8_rgb(arr))
        img = ImageOps.fit(img, (panel_size, panel_size), method=Image.Resampling.BICUBIC)
        panels.append(img)

    canvas = Image.new("RGB", (panel_size * 3 + 24, panel_size + 34), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(panels, ["Before", "After", "Overlay"], strict=True)):
        x = idx * panel_size + idx * 12
        canvas.paste(img, (x, 34))
        draw.text((x + 8, 8), label, fill=(255, 255, 255))
    return canvas


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _is_mps_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "insufficient memory" in text or "outofmemory" in text or "out of memory" in text or "kIOGPUCommandBufferCallbackErrorOutOfMemory".lower() in text


def _trim_json(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.splitlines()[:-1])
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        return clean[start : end + 1]
    return clean


def _strip_control_sequences(text: str) -> str:
    text = _ANSI_ESCAPE_RE.sub("", text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return "".join(char for char in text if char == "\n" or char == "\t" or char.isprintable())


def _sanitize_model_text(text: str) -> str:
    clean = _strip_control_sequences(text).strip()
    for marker in ("{", "Focus square:", "Scene classification:"):
        idx = clean.find(marker)
        if idx > 0:
            clean = clean[idx:]
            break
    return clean.strip()


def _sanitize_reasoning_text(text: str) -> str:
    clean = _strip_control_sequences(text).strip()
    clean = clean.replace("<think>", "").replace("</think>", "").strip()
    return clean


def _normalize_reasoning_notes(text: str) -> str:
    clean = _sanitize_reasoning_text(text)
    parsed = parse_reasoning_notes(clean)
    if not parsed:
        return clean
    ordered = [
        ("Visual evidence before", parsed.get("visual_evidence_before", "")),
        ("Visual evidence after", parsed.get("visual_evidence_after", "")),
        ("Observed differences", parsed.get("observed_differences", "")),
        ("Most affected area", parsed.get("most_affected_area", "")),
        ("Uncertainty notes", parsed.get("uncertainty_notes", "")),
    ]
    return "\n".join(f"{label}: {value}".strip() for label, value in ordered if str(value).strip())


def _looks_like_draft_output(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    draft_markers = [
        "got it",
        "let's compare",
        "first, look at",
        "wait,",
        "wait ",
        "maybe",
        "the problem says",
    ]
    if any(marker in lowered for marker in draft_markers):
        return True
    required_labels = [
        "scene classification:",
        "overall change summary:",
        "primary changed cells:",
        "cell observations:",
        "main difference:",
        "scene_overview",
        '"cells"',
        "cells:",
    ]
    return not any(label in lowered for label in required_labels)


def _looks_like_json(text: str) -> bool:
    trimmed = (text or "").strip()
    return trimmed.startswith("{") and trimmed.endswith("}")


def _parse_structured_text(text: str) -> dict | None:
    expected_keys = {
        "scene_classification",
        "overall_change_summary",
        "primary_changed_cells",
        "cell_observations",
        "main_difference",
        "surface_before",
        "surface_after",
        "confidence",
        "visible_evidence",
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed: dict[str, object] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        if key not in expected_keys:
            continue
        value = value.strip().strip("- ").strip()
        if not key:
            continue
        if key in {"visible_evidence", "primary_changed_cells", "cell_observations"}:
            items = [item.strip(" .;") for item in value.replace("•", ";").split(";") if item.strip(" .;")]
            parsed[key] = items
        else:
            parsed[key] = value
    return parsed or None


def _parse_partial_json_like(text: str) -> dict | None:
    expected_keys = [
        "scene_classification",
        "overall_change_summary",
        "primary_changed_cells",
        "cell_observations",
        "main_difference",
        "surface_before",
        "surface_after",
        "confidence",
    ]
    parsed: dict[str, object] = {}
    for key in expected_keys:
        if key in {"primary_changed_cells", "cell_observations"}:
            array_match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
            if array_match:
                items = re.findall(r'"([^"]+)"', array_match.group(1))
                if items:
                    parsed[key] = [item.strip() for item in items if item.strip()]
        else:
            match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
            if match:
                parsed[key] = match.group(1).strip()
    return parsed or None


def _parse_labeled_stream(text: str) -> dict | None:
    markers = [
        ("Scene classification", "scene_classification"),
        ("Overall change summary", "overall_change_summary"),
        ("Primary changed cells", "primary_changed_cells"),
        ("Cell observations", "cell_observations"),
        ("Main difference", "main_difference"),
        ("Surface before", "surface_before"),
        ("Surface after", "surface_after"),
        ("Confidence", "confidence"),
    ]
    pattern = re.compile(
        r"(?i)(?:^|[|]|\b)\s*("
        + "|".join(re.escape(label) for label, _ in markers)
        + r")\s*:\s*"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    label_to_key = {label.lower(): key for label, key in markers}
    parsed: dict[str, object] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).strip().lower()
        key = label_to_key.get(label)
        if not key:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[start:end].strip(" |.\n\t")
        if not value:
            continue
        if key in {"primary_changed_cells", "cell_observations"}:
            items = [item.strip(" .;") for item in value.replace("•", ";").split(";") if item.strip(" .;")]
            parsed[key] = items
        else:
            parsed[key] = value
    return parsed or None


def _normalize_focus_square(value: str | None) -> str | None:
    if not value:
        return None
    lowered_value = value.strip().lower()
    if "|" in lowered_value and "unclear" in lowered_value:
        return "unclear"
    for token in re.split(r"[|,;/]", value):
        token = token.strip().lower()
        if token in {"top-left", "top_right", "top left"}:
            return "top-left"
        if token in {"top-right", "top_right", "top right"}:
            return "top-right"
        if token in {"bottom-left", "bottom_left", "bottom left"}:
            return "bottom-left"
        if token in {"bottom-right", "bottom_right", "bottom right"}:
            return "bottom-right"
        if token in {"center", "centre"}:
            return "center"
        if token == "unclear":
            return "unclear"
    return value.strip()


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _extract_after_phrase(sentence: str, phrase: str) -> str | None:
    match = re.search(re.escape(phrase), sentence, flags=re.IGNORECASE)
    if not match:
        return None
    value = sentence[match.end() :].strip(" .,:;")
    return value or None


def _parse_prose_text(text: str) -> dict | None:
    if not text.strip():
        return None
    parsed: dict[str, object] = {}
    sentences = _split_sentences(text)

    scene = None
    overall_change_summary = None
    primary_changed_cells: list[str] | None = None
    cell_observations: list[str] | None = None
    main_difference = None
    before = None
    after = None
    confidence = None

    for sentence in sentences:
        lower = sentence.lower()
        if "the scene is" in lower and scene is None:
            scene = _extract_after_phrase(sentence, "the scene is")
        elif "scene classification" in lower and scene is None:
            scene = _extract_after_phrase(sentence, "scene classification:")
            if scene is None:
                scene = _extract_after_phrase(sentence, "scene classification is")
        if "overall change summary" in lower and overall_change_summary is None:
            overall_change_summary = _extract_after_phrase(sentence, "overall change summary:")
        if "primary changed cells" in lower and primary_changed_cells is None:
            tail = _extract_after_phrase(sentence, "primary changed cells:")
            if tail:
                primary_changed_cells = [bit.strip(" .,:;") for bit in re.split(r",|;|\band\b", tail, flags=re.IGNORECASE) if bit.strip(" .,:;")]
        if "cell observations" in lower and cell_observations is None:
            tail = _extract_after_phrase(sentence, "cell observations:")
            if tail:
                cell_observations = [bit.strip(" .,:;") for bit in re.split(r";|\n", tail, flags=re.IGNORECASE) if bit.strip(" .,:;")]
        if "main difference" in lower and main_difference is None:
            main_difference = _extract_after_phrase(sentence, "the main difference is")
            if main_difference is None:
                main_difference = _extract_after_phrase(sentence, "main difference:")
            if main_difference is None:
                main_difference = _extract_after_phrase(sentence, "main difference is")
        if "the change between the two images is" in lower and main_difference is None:
            main_difference = _extract_after_phrase(sentence, "the change between the two images is")
        if "surface before" in lower and before is None:
            before = _extract_after_phrase(sentence, "the surface before the change is")
            if before is None:
                before = _extract_after_phrase(sentence, "the surface before is")
            if before is None:
                before = _extract_after_phrase(sentence, "surface before:")
            if before is None:
                before = _extract_after_phrase(sentence, "surface before is")
            if before and "while" in before.lower():
                before = before.split("while", 1)[0].strip(" .,:;")
        if "surface after" in lower and after is None:
            after = _extract_after_phrase(sentence, "the surface after the change is")
            if after is None:
                after = _extract_after_phrase(sentence, "the surface after is")
            if after is None:
                after = _extract_after_phrase(sentence, "surface after:")
            if after is None:
                after = _extract_after_phrase(sentence, "surface after is")
        if "confidence" in lower and confidence is None:
            conf_match = re.search(r"(?i)\bconfidence(?: level(?: in the classification)?)?\b\s*(?:is|:)\s*([a-z]+)", sentence)
            if conf_match:
                confidence = conf_match.group(1).strip().lower()

    if scene:
        parsed["scene_classification"] = scene.strip(" .,:;")
    if overall_change_summary:
        parsed["overall_change_summary"] = overall_change_summary.strip(" .,:;")
    if primary_changed_cells:
        parsed["primary_changed_cells"] = primary_changed_cells
    if cell_observations:
        parsed["cell_observations"] = cell_observations
    if main_difference:
        parsed["main_difference"] = main_difference.strip(" .,:;")
    if before:
        parsed["surface_before"] = before.strip(" .,:;")
    if after:
        parsed["surface_after"] = after.strip(" .,:;")
    if confidence:
        parsed["confidence"] = confidence
    return parsed or None


@dataclass
class VLMResult:
    raw_text: str
    parsed: dict | None
    reasoning_text: str = ""
    reasoning_complete: bool = False
    final_text: str = ""
    phase_trace: list[str] | None = None
    backend_runtime: str = ""
    memory_mode: str = ""


@dataclass(frozen=True)
class ReasoningProfile:
    label: str
    enable_thinking: bool
    reasoning_tokens_mps: int
    reasoning_tokens_other: int
    max_new_tokens_mps: int
    max_new_tokens_other: int
    use_cache_on_mps: bool
    semantic_context_chars: int
    image_max_side_mps: int
    image_max_side_other: int
    ollama_think: bool | str
    ollama_num_predict: int
    ollama_num_ctx: int
    reasoning_rounds: int = 2
    mlx_prefill_step_size: int = 256


REASONING_PROFILES: dict[str, ReasoningProfile] = {
    "efficient": ReasoningProfile(
        label="Efficient",
        enable_thinking=True,
        reasoning_tokens_mps=192,
        reasoning_tokens_other=224,
        max_new_tokens_mps=192,
        max_new_tokens_other=256,
        use_cache_on_mps=True,
        semantic_context_chars=700,
        image_max_side_mps=256,
        image_max_side_other=640,
        ollama_think="low",
        ollama_num_predict=384,
        ollama_num_ctx=2048,
        reasoning_rounds=1,
        mlx_prefill_step_size=256,
    ),
    "balanced": ReasoningProfile(
        label="Balanced",
        enable_thinking=True,
        reasoning_tokens_mps=256,
        reasoning_tokens_other=320,
        max_new_tokens_mps=256,
        max_new_tokens_other=384,
        use_cache_on_mps=True,
        semantic_context_chars=900,
        image_max_side_mps=320,
        image_max_side_other=768,
        ollama_think="medium",
        ollama_num_predict=512,
        ollama_num_ctx=3072,
        reasoning_rounds=1,
        mlx_prefill_step_size=256,
    ),
    "deep": ReasoningProfile(
        label="Deep",
        enable_thinking=True,
        reasoning_tokens_mps=448,
        reasoning_tokens_other=512,
        max_new_tokens_mps=512,
        max_new_tokens_other=640,
        use_cache_on_mps=False,
        semantic_context_chars=1200,
        image_max_side_mps=384,
        image_max_side_other=896,
        ollama_think="high",
        ollama_num_predict=768,
        ollama_num_ctx=4096,
        reasoning_rounds=2,
        mlx_prefill_step_size=384,
    ),
}


def parse_model_response(raw_text: str) -> dict:
    raw_text = _sanitize_model_text(raw_text)
    parsed = None
    try:
        parsed = json.loads(_trim_json(raw_text))
    except Exception:
        parsed = _parse_partial_json_like(raw_text) or _parse_labeled_stream(raw_text) or _parse_structured_text(raw_text)

    if parsed and "focus_square" in parsed and "primary_changed_cells" not in parsed:
        focus = _normalize_focus_square(str(parsed["focus_square"]))
        if focus:
            parsed["primary_changed_cells"] = [focus]
    if parsed and "detailed_change_summary" in parsed and "overall_change_summary" not in parsed:
        parsed["overall_change_summary"] = parsed["detailed_change_summary"]
    if parsed and "changed_elements" in parsed and "cell_observations" not in parsed:
        parsed["cell_observations"] = parsed["changed_elements"]
    if parsed and "before_image_details" in parsed and "surface_before" not in parsed:
        parsed["surface_before"] = parsed["before_image_details"]
    if parsed and "after_image_details" in parsed and "surface_after" not in parsed:
        parsed["surface_after"] = parsed["after_image_details"]
    if parsed:
        parsed = normalize_visual_summary(parsed)
    return parsed or {}


def _clean_user_text(value: object) -> str:
    text = _sanitize_model_text(str(value or "")).strip()
    bad_markers = [
        "here's a thinking process",
        "i will analyze",
        "step by step",
        "placeholder",
        "the goal is",
        "analyze the request",
    ]
    lowered = text.lower()
    if any(marker in lowered for marker in bad_markers):
        return ""
    return text


def _normalize_string_list(value: object, limit: int = 5) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"\n+|;|(?:^|\s)[-*]\s+", value)
    else:
        return []
    result: list[str] = []
    for item in items:
        text = _clean_user_text(item)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


EXPECTED_GRID_CELLS = tuple(f"{row}{col}" for row in "ABCD" for col in range(1, 5))


def _is_bad_cell_observation(text: str) -> bool:
    lowered = text.lower()
    bad_phrases = [
        "possible expansion of low vegetation or cultivated surface",
        "clear transition from bare/prepared ground to low vegetation",
        "clear transition from compacted surface to low vegetation",
        "the cell gains vegetation-related classes",
        "semantic composition stays broadly similar",
        "weak semantic transition signal",
    ]
    if any(phrase in lowered for phrase in bad_phrases):
        return True
    semantic_only_markers = ["more grass", "less background", "less bareland", "more cropland", "more background"]
    return any(marker in lowered for marker in semantic_only_markers) and not any(
        cue in lowered for cue in ["road", "track", "building", "roof", "plot", "line", "graded", "cleared", "yard", "construction", "shadow", "stable", "unchanged"]
    )


def normalize_cell_observations(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for idx, item in enumerate(value, start=1):
        if isinstance(item, dict):
            cell = str(item.get("cell") or item.get("cell_id") or "").strip().upper()
            observation = _clean_user_text(item.get("observation") or item.get("summary") or item.get("change") or "")
            interpretation = _clean_user_text(item.get("interpretation") or item.get("meaning") or item.get("likely_process") or "")
            evidence = _clean_user_text(item.get("visual_evidence") or item.get("evidence") or "")
            if interpretation and interpretation.lower() not in observation.lower():
                observation = f"{observation} Likely meaning: {interpretation}" if observation else interpretation
            if evidence and evidence.lower() not in observation.lower():
                observation = f"{observation} Visible evidence: {evidence}" if observation else evidence
            confidence = str(item.get("confidence") or "uncertain").strip().lower()
        else:
            raw = _clean_user_text(item)
            match = re.match(r"^\s*([A-D][1-4])\s*[:\-]\s*(.+)$", raw, flags=re.IGNORECASE)
            cell = match.group(1).upper() if match else f"item_{idx}"
            observation = match.group(2).strip() if match else raw
            confidence = "uncertain"
        if not observation or _is_bad_cell_observation(observation):
            continue
        if not re.match(r"^[A-D][1-4]$", cell):
            cell = f"item_{idx}"
        if confidence not in {"high", "medium", "low", "uncertain"}:
            confidence = "uncertain"
        result.append({"cell": cell, "observation": observation, "confidence": confidence})
    return result


def cell_observations_quality_issues(rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    cells = [str(row.get("cell", "")).strip().upper() for row in rows]
    missing = [cell for cell in EXPECTED_GRID_CELLS if cell not in cells]
    if missing:
        issues.append("missing cells: " + ", ".join(missing))
    unexpected = [cell for cell in cells if cell not in EXPECTED_GRID_CELLS]
    if unexpected:
        issues.append("unexpected cells: " + ", ".join(unexpected[:6]))
    observations = [re.sub(r"\s+", " ", str(row.get("observation", "")).strip().lower()) for row in rows]
    repeated = {text for text in observations if text and observations.count(text) >= 4}
    if repeated:
        issues.append("repeated generic observations")
    too_short = [cell for cell, text in zip(cells, observations, strict=False) if len(text) < 18]
    if too_short:
        issues.append("too-short observations: " + ", ".join(too_short[:6]))
    return issues


def normalize_visual_summary(parsed: dict) -> dict:
    normalized = dict(parsed)
    for key in ("scene_overview", "before_summary", "after_summary"):
        normalized[key] = _clean_user_text(normalized.get(key, ""))
    if not normalized.get("before_summary"):
        normalized["before_summary"] = _clean_user_text(normalized.get("surface_before", ""))
    if not normalized.get("after_summary"):
        normalized["after_summary"] = _clean_user_text(normalized.get("surface_after", ""))
    normalized["main_changes"] = _normalize_string_list(normalized.get("main_changes") or normalized.get("primary_changed_cells"), limit=5)
    normalized["cell_observations"] = normalize_cell_observations(normalized.get("cell_observations"))
    return normalized


def has_complete_final_response(parsed: dict | None) -> bool:
    if not parsed:
        return False
    scene_overview = str(parsed.get("scene_overview", "")).strip()
    bad_markers = ["here's a thinking process", "i will analyze", "since no images are provided", "step by step", "placeholder"]
    observations = parsed.get("cell_observations")
    if not isinstance(observations, list) or cell_observations_quality_issues(observations):
        return False
    return (
        bool(scene_overview)
        and bool(str(parsed.get("before_summary", "")).strip())
        and bool(str(parsed.get("after_summary", "")).strip())
        and bool(parsed.get("main_changes"))
        and not any(marker in scene_overview.lower() for marker in bad_markers)
    )


def parse_reasoning_notes(text: str) -> dict[str, str]:
    markers = {
        "visual_evidence_before": "Visual evidence before",
        "visual_evidence_after": "Visual evidence after",
        "observed_differences": "Observed differences",
        "most_affected_area": "Most affected area",
        "uncertainty_notes": "Uncertainty notes",
    }
    parsed: dict[str, str] = {}
    for key, label in markers.items():
        match = re.search(rf"(?im)^{re.escape(label)}:\s*(.+)$", text or "")
        if match:
            parsed[key] = match.group(1).strip()
    area = parsed.get("most_affected_area")
    if area:
        parsed["most_affected_area"] = _normalize_focus_square(area) or area
    return parsed


def reasoning_notes_complete(text: str) -> bool:
    parsed = parse_reasoning_notes(_normalize_reasoning_notes(text))
    required = [
        "visual_evidence_before",
        "visual_evidence_after",
        "observed_differences",
        "most_affected_area",
        "uncertainty_notes",
    ]
    values = [parsed.get(key, "").strip() for key in required]
    if not all(values):
        return False
    if any(len(value) < 18 for value in values):
        return False
    bad_markers = [
        "<",
        ">",
        "concise evidence",
        "concrete visible differences",
        "ambiguities",
        "here's a thinking process",
        "step by step",
        "let's",
        "wait,",
        "wait ",
        "maybe",
        "the problem says",
    ]
    lowered = " ".join(values).lower()
    return not any(marker in lowered for marker in bad_markers)


def ollama_model_available(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def mlx_model_ready(model_name_or_path: str) -> bool:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return (model_path / "config.json").exists() and (model_path / "model.safetensors.index.json").exists()
    return model_name_or_path.startswith("mlx-community/")


def preferred_mlx_qwen_model() -> str:
    if mlx_model_ready(str(QWEN3_VL_4B_THINKING_MLX_3BIT_DIR)):
        return str(QWEN3_VL_4B_THINKING_MLX_3BIT_DIR)
    if mlx_model_ready(str(QWEN3_5_VL_0_8B_MLX_4BIT_DIR)):
        return str(QWEN3_5_VL_0_8B_MLX_4BIT_DIR)
    return QWEN3_VL_4B_THINKING_MLX_3BIT


class QwenMlxVlmExplainer:
    def __init__(self, model_name_or_path: str | None = None, reasoning_profile: str = "balanced"):
        self.model_name_or_path = model_name_or_path or preferred_mlx_qwen_model()
        self.reasoning_profile_name = reasoning_profile if reasoning_profile in REASONING_PROFILES else "balanced"
        self.reasoning_profile = REASONING_PROFILES[self.reasoning_profile_name]
        self.model = None
        self.processor = None
        self._mlx = None
        self._stream_generate = None
        self._load = None
        self._apply_chat_template = None

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        import mlx.core as mx

        self.model, self.processor = load(self.model_name_or_path)
        self._stream_generate = stream_generate
        self._mlx = mx
        self._apply_chat_template = apply_chat_template

    def _resize_if_needed(self, rgb: np.ndarray) -> np.ndarray:
        arr = _to_uint8_rgb(rgb)
        max_side = self.reasoning_profile.image_max_side_mps
        height, width = arr.shape[:2]
        side = max(height, width)
        if side <= max_side:
            return arr
        scale = max_side / float(side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return np.asarray(Image.fromarray(arr).resize(new_size, Image.BICUBIC))

    def _reasoning_prompt(self, response_language: str, visual_context: str, semantic_context: str, has_change_guide: bool) -> str:
        prompt = "".join(
            [
                "You are comparing aligned satellite crops of the same place.\n",
                "Image 1 is BEFORE. Image 2 is AFTER.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use BEFORE and AFTER as primary evidence. Use the contact sheet to inspect each grid cell; treat the heatmap only as a secondary inspection aid.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Do not infer object type from heatmap color alone.\n",
                "Inspect overall scene layout first, then inspect highlighted cells for local texture, line, road, plot, compact-surface, and ground-change cues.\n",
                "Use cautious hypotheses unless multiple cues agree. Roof/building-specific claims require strong rectilinear built-surface evidence.\n",
                "Return only five labeled fields and nothing else.\n",
                "No introduction. No numbering. No bullets. No 'step by step'. No explanation of your process.\n",
                "Fill every field with concrete visual observations from the images.\n",
                f"Write in {response_language} using exactly this format:\n",
                "Visual evidence before: concrete observations from BEFORE\n",
                "Visual evidence after: concrete observations from AFTER\n",
                "Observed differences: specific visible changes\n",
                "Most affected area: one or more grid cells like A1, A2, B3, C4, or unclear\n",
                "Uncertainty notes: what remains ambiguous\n",
            ]
        )
        if visual_context.strip():
            prompt += "\nDeterministic change-tool hints:\n" + visual_context.strip() + "\n"
        if semantic_context.strip():
            prompt += "\nOptional semantic cross-check:\n" + semantic_context.strip() + "\n"
        return prompt

    def _reasoning_continue_prompt(self, response_language: str, visual_context: str, semantic_context: str, prior_reasoning: str, has_change_guide: bool) -> str:
        return (
            self._reasoning_prompt(response_language, visual_context, semantic_context, has_change_guide)
            + "\nRewrite all five labeled fields from scratch.\n"
            + "The previous attempt was incomplete or contained meta-commentary. Do not continue prose. Do not mention the task.\n"
            + "Previous attempt:\n"
            + _normalize_reasoning_notes(prior_reasoning).strip()
        )

    def _final_prompt(self, response_language: str, reasoning_notes: str, visual_context: str, semantic_context: str, has_change_guide: bool) -> str:
        prompt = "".join(
            [
                "You are comparing aligned satellite crops of the same place.\n",
                "Image 1 is BEFORE. Image 2 is AFTER.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use BEFORE and AFTER as primary evidence. Use the contact sheet to inspect each grid cell; treat the heatmap only as a secondary inspection aid.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Do not infer object type from heatmap color alone. Reasoning notes below are scratch work and must not be exposed.\n",
                "Return strict JSON only with these keys: scene_overview, before_summary, after_summary, main_changes, cell_observations.\n",
                "scene_overview must be one short paragraph explaining the overall physical/geographic change pattern.\n",
                "before_summary and after_summary must describe visible surface state in plain language.\n",
                "main_changes must be 2 to 5 short plain-English strings.\n",
                "cell_observations must contain exactly 16 objects, one for every cell A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3, D4.\n",
                "Each cell object must be {\"cell\":\"A1\", \"observation\":\"specific visible before/after comparison plus likely real-world meaning\", \"confidence\":\"high|medium|low|uncertain\"}.\n",
                "For each changed cell, explicitly connect evidence to meaning, for example: 'new pale rectangular pads and access tracks suggest active construction or site preparation', 'a continuous dark linear feature suggests a new road or service track', 'rougher exposed soil suggests excavation or recently disturbed ground'.\n",
                "Prefer physical/geographic interpretation over raw low-level cues. Use 'possible' for uncertain object hypotheses. Do not use 'tone change' as likely_change when structural cues exist.\n",
                "Do not overuse building-specific language; only use it when rectilinear bright-surface and line-structure cues agree strongly.\n",
                "Do not include chain-of-thought, draft commentary, self-correction, markdown, or extra text outside the JSON.\n",
            ]
        )
        if visual_context.strip():
            prompt += "\nDeterministic change-tool hints:\n" + visual_context.strip() + "\n"
        if semantic_context.strip():
            prompt += "\nOptional semantic notes for cross-check only:\n" + semantic_context.strip() + "\n"
        if reasoning_notes.strip():
            prompt += "\nReasoning notes:\n" + _normalize_reasoning_notes(reasoning_notes).strip() + "\n"
        return prompt

    def _write_temp_images(self, images: list[tuple[str, np.ndarray]]) -> tuple[list[str], list[str]]:
        temp_paths: list[str] = []
        dirs: list[str] = []
        for prefix, arr in images:
            temp_dir = tempfile.mkdtemp(prefix=f"land-change-{prefix}-")
            dirs.append(temp_dir)
            path = Path(temp_dir) / f"{prefix}.png"
            Image.fromarray(arr).save(path)
            temp_paths.append(str(path))
        return temp_paths, dirs

    def _cleanup_temp_dirs(self, temp_dirs: list[str]) -> None:
        for path in temp_dirs:
            try:
                for child in Path(path).glob("*"):
                    child.unlink(missing_ok=True)
                Path(path).rmdir()
            except Exception:
                pass

    def _stream_text(
        self,
        prompt: str,
        image_paths: list[str],
        max_tokens: int,
        stage: str,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._ensure_loaded()
        prompt_with_images = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            add_generation_prompt=True,
            num_images=len(image_paths),
        )
        text_parts: list[str] = []
        stats: dict[str, Any] = {}
        iterator = self._stream_generate(
            self.model,
            self.processor,
            prompt=prompt_with_images,
            image=image_paths,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            verbose=False,
            prefill_step_size=self.reasoning_profile.mlx_prefill_step_size,
        )
        for response in iterator:
            segment = response.text or ""
            if segment:
                text_parts.append(segment)
                if progress_cb is not None:
                    progress_cb({"stage": stage, "delta": segment, "text": "".join(text_parts)})
            stats = {
                "prompt_tokens": getattr(response, "prompt_tokens", 0),
                "generation_tokens": getattr(response, "generation_tokens", 0),
                "generation_tps": getattr(response, "generation_tps", 0.0),
                "peak_memory": getattr(response, "peak_memory", 0.0),
            }
        return _sanitize_reasoning_text("".join(text_parts)), stats

    def explain(
        self,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        semantic_context: str = "",
        visual_context: str = "",
        auxiliary_images: list[np.ndarray] | None = None,
        response_language: str = "English",
        progress_cb: Callable[[dict], None] | None = None,
    ) -> VLMResult:
        before_img = self._resize_if_needed(before_crop)
        after_img = self._resize_if_needed(after_crop)
        aux_images = [self._resize_if_needed(image) for image in (auxiliary_images or [])]
        has_change_guide = bool(aux_images)
        phase_trace: list[str] = []
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "mlx_prepare",
                    "message": (
                        f"Prepared {2 + len(aux_images)} images for MLX, max side {max(before_img.shape[:2])}, "
                        f"profile {self.reasoning_profile.label}, reasoning_rounds={self.reasoning_profile.reasoning_rounds}"
                    ),
                }
            )
        image_paths, temp_dirs = self._write_temp_images(
            [("before", before_img), ("after", after_img), *[(f"aux-{idx + 1}", image) for idx, image in enumerate(aux_images)]]
        )
        reasoning_text = ""
        reasoning_complete = False
        memory_mode = "mlx-vlm"
        try:
            if self.reasoning_profile.enable_thinking:
                for idx in range(self.reasoning_profile.reasoning_rounds):
                    prompt = (
                        self._reasoning_prompt(response_language, visual_context, semantic_context, has_change_guide)
                        if not reasoning_text.strip()
                        else self._reasoning_continue_prompt(response_language, visual_context, semantic_context, reasoning_text, has_change_guide)
                    )
                    if progress_cb is not None:
                        progress_cb(
                            {
                                "stage": "mlx_reasoning_start",
                                "message": f"Reasoning round {idx + 1}/{self.reasoning_profile.reasoning_rounds}, up to {self.reasoning_profile.reasoning_tokens_mps} tokens",
                            }
                        )
                    round_text, stats = self._stream_text(
                        prompt,
                        image_paths,
                        max_tokens=self.reasoning_profile.reasoning_tokens_mps,
                        stage="mlx_reasoning",
                        progress_cb=progress_cb,
                    )
                    phase_trace.append(f"reasoning_round_{idx + 1}: {stats}")
                    if round_text.strip():
                        reasoning_text = _normalize_reasoning_notes((reasoning_text + "\n" + round_text).strip())
                    reasoning_complete = reasoning_notes_complete(reasoning_text)
                    memory_mode = f"mlx peak {stats.get('peak_memory', 0.0):.2f} GB"
                    if reasoning_complete:
                        if progress_cb is not None:
                            progress_cb({"stage": "mlx_reasoning_ready", "message": "Reasoning notes complete"})
                        break
                    if progress_cb is not None:
                        progress_cb({"stage": "mlx_reasoning_incomplete", "message": "Reasoning notes incomplete, continuing same stage"})

            if progress_cb is not None:
                progress_cb({"stage": "mlx_final_start", "message": f"Generating strict final answer up to {self.reasoning_profile.max_new_tokens_mps} tokens"})
            final_text, stats = self._stream_text(
                self._final_prompt(response_language, reasoning_text, visual_context, semantic_context, has_change_guide),
                image_paths,
                max_tokens=self.reasoning_profile.max_new_tokens_mps,
                stage="mlx_final",
                progress_cb=progress_cb,
            )
            phase_trace.append(f"final: {stats}")
            parsed = parse_model_response(final_text)
            if not has_complete_final_response(parsed):
                if progress_cb is not None:
                    progress_cb({"stage": "mlx_final_retry", "message": "Final answer incomplete, retrying one strict final-only pass"})
                retry_text, retry_stats = self._stream_text(
                    self._final_prompt(response_language, reasoning_text, visual_context, semantic_context, has_change_guide)
                    + "\nReturn only strict JSON. Include exactly 16 cell_observations, one for every A1..D4 cell, with specific non-repeated before/after visual evidence.",
                    image_paths,
                    max_tokens=max(72, self.reasoning_profile.max_new_tokens_mps),
                    stage="mlx_final",
                    progress_cb=progress_cb,
                )
                phase_trace.append(f"final_retry: {retry_stats}")
                final_text = retry_text
                parsed = parse_model_response(final_text)
                memory_mode = f"mlx peak {retry_stats.get('peak_memory', 0.0):.2f} GB"
            return VLMResult(
                raw_text=final_text,
                parsed=parsed,
                reasoning_text=reasoning_text,
                reasoning_complete=reasoning_complete,
                final_text=final_text,
                phase_trace=phase_trace,
                backend_runtime="mlx_vlm_qwen",
                memory_mode=memory_mode,
            )
        finally:
            self._cleanup_temp_dirs(temp_dirs)
            self.unload()

    def unload(self) -> None:
        if self._mlx is not None:
            try:
                self._mlx.clear_cache()
            except Exception:
                pass
        self.model = None
        self.processor = None


class RemoteSensingQwen2VL2B:
    def __init__(
        self,
        model_name_or_path: str = REMOTE_SENSING_QWEN2_5_VL_3B,
        device: str | torch.device = "cpu",
        reasoning_profile: str = "balanced",
    ):
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type in {"mps", "cuda"} else torch.float32
        self.reasoning_profile_name = reasoning_profile if reasoning_profile in REASONING_PROFILES else "balanced"
        self.reasoning_profile = REASONING_PROFILES[self.reasoning_profile_name]
        lowered_name = str(model_name_or_path).lower()
        self.backend = "earthdial" if "earthdial" in lowered_name else "hf_vlm"
        self.processor = None
        self.tokenizer = None
        self.transform = None
        self.model = None
        if self.backend == "earthdial":
            self._init_earthdial(model_name_or_path)
        else:
            self._init_hf_vlm(model_name_or_path)

    def _init_earthdial(self, model_name_or_path: str) -> None:
        _ensure_earthdial_import_path()
        _patch_transformers_compat()
        from earthdial.model.internvl_chat import InternVLChatModel
        from transformers import LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        self.model = InternVLChatModel.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
        )
        self.model = self.model.to(self.device, dtype=self.dtype) if self.device.type != "cpu" else self.model.to(self.device)
        self.model.eval()
        image_size = int(getattr(self.model.config, "force_image_size", None) or getattr(self.model.config.vision_config, "image_size", 448))
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img if img.mode == "L" else img.convert("RGB")),
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _init_hf_vlm(self, model_name_or_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        load_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device.type == "mps":
            # MPS is more stable with eager attention for large VL checkpoints.
            load_kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _resize_if_needed(self, rgb: np.ndarray, max_side: int | None = None) -> np.ndarray:
        arr = _to_uint8_rgb(rgb)
        if max_side is None:
            if self.device.type == "mps":
                max_side = self.reasoning_profile.image_max_side_mps
            else:
                max_side = self.reasoning_profile.image_max_side_other
        height, width = arr.shape[:2]
        side = max(height, width)
        if side <= max_side:
            return arr
        scale = max_side / float(side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return np.asarray(Image.fromarray(arr).resize(new_size, Image.BICUBIC))

    def _single_image_prompt(self, when: str, response_language: str) -> str:
        return (
            f"You are analyzing the {when} remote-sensing crop from one location.\n"
            "Describe only clearly visible objects and surfaces.\n"
            "Focus on: buildings, roads, bare ground, cleared areas, construction traces, vegetation, water.\n"
            "Do not invent details that are not clearly visible.\n"
            f"Return exactly 4 lines in {response_language} and nothing else:\n"
            "Scene classification: <short label>\n"
            "Visible structures: <comma-separated list of clearly visible man-made structures or 'none'>\n"
            "Visible surfaces: <comma-separated list of clearly visible surfaces>\n"
            "Confidence: low|medium|high\n"
        )

    def _reasoning_prompt(self, response_language: str, visual_context: str = "", semantic_context: str = "", has_change_guide: bool = False) -> str:
        return "".join(
            [
                "You are comparing aligned BEFORE and AFTER remote-sensing crop images from the same location.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use BEFORE and AFTER as primary evidence. Use the contact sheet to inspect each grid cell; treat the heatmap only as a secondary inspection aid.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Do not infer object type from heatmap color alone.\n",
                "Inspect highlighted changed cells carefully for roads, plot/service lines, compact surfaces, ground clearing, rectilinear patches, and local texture changes.\n",
                "Use cautious wording unless several cues agree; roof/building-specific claims need strong rectilinear built-surface evidence.\n",
                "Return only five labeled fields and nothing else.\n",
                "No introduction. No self-correction. No 'wait', 'maybe', or 'let us compare'.\n",
                f"Write in {response_language} using exactly these labels:\n",
                "Visual evidence before: what is clearly visible in BEFORE\n",
                "Visual evidence after: what is clearly visible in AFTER\n",
                "Observed differences: specific visible differences\n",
                "Most affected area: one or more grid cells like A1, A2, B3, C4, or unclear\n",
                "Uncertainty notes: what remains ambiguous\n",
                ("\nDeterministic change-tool hints:\n" + visual_context.strip() + "\n") if visual_context.strip() else "",
                ("\nOptional semantic cross-check:\n" + semantic_context.strip() + "\n") if semantic_context.strip() else "",
            ]
        )

    def _final_prompt(self, response_language: str, reasoning_notes: str = "", visual_context: str = "", semantic_context: str = "", has_change_guide: bool = False) -> str:
        prompt = "".join(
            [
                "You are comparing aligned BEFORE and AFTER remote-sensing crop images from the same location.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use BEFORE and AFTER as primary evidence. Use the contact sheet to inspect each grid cell; treat the heatmap only as a secondary inspection aid.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Do not infer object type from heatmap color alone.\n",
                "If reasoning notes are provided below, treat them as scratch analysis and distill them into a concise user-facing answer.\n",
                "Return strict JSON only with these keys: scene_overview, before_summary, after_summary, main_changes, cell_observations.\n",
                "scene_overview must be one short paragraph explaining the overall physical/geographic change pattern.\n",
                "before_summary and after_summary must describe visible surface state in plain language.\n",
                "main_changes must be 2 to 5 short plain-English strings.\n",
                "cell_observations must contain exactly 16 objects, one for every cell A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3, D4.\n",
                "Each cell object must be {\"cell\":\"A1\", \"observation\":\"specific visible before/after comparison plus likely real-world meaning\", \"confidence\":\"high|medium|low|uncertain\"}.\n",
                "For each changed cell, explicitly connect evidence to meaning, for example: 'new pale rectangular pads and access tracks suggest active construction or site preparation', 'a continuous dark linear feature suggests a new road or service track', 'rougher exposed soil suggests excavation or recently disturbed ground'.\n",
                "Describe visible physical/geographic changes cautiously: possible construction activity, new access road or track, new building or roof, compacted pad or yard, plot or service-line trace, road-edge rework, grading, clearing, excavation, field preparation, ambiguous local reworking, or stable area.\n",
                "Only call something roof/building-specific when multiple strong visual cues agree. Do not invent small objects or causes. Do not use 'tone change' as likely_change when structural cues exist.\n",
                f"Write field values in {response_language}. Do not output chain-of-thought, markdown, draft notes, or introductory phrases.\n",
            ]
        )
        if visual_context.strip():
            prompt += "\nDeterministic change-tool hints:\n" + visual_context.strip() + "\n"
        if semantic_context.strip():
            prompt += "\nOptional semantic cross-check:\n" + semantic_context.strip() + "\n"
        if reasoning_notes.strip():
            prompt += "\nReasoning notes:\n" + _normalize_reasoning_notes(reasoning_notes).strip()
        return prompt
    def _earthdial_chat(self, before_img: np.ndarray, after_img: np.ndarray, overlay_img: np.ndarray, prompt: str) -> str:
        from earthdial.conversation import get_conv_template

        composite = _composite_triptych(before_img, after_img, overlay_img)
        pixel_values = self.transform(composite).unsqueeze(0)
        if self.device.type != "cpu":
            pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt

        img_start_token = "<img>"
        img_end_token = "</img>"
        img_context_token = "<IMG_CONTEXT>"
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)
        self.model.img_context_token_id = img_context_token_id
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        num_patches = pixel_values.shape[0]
        image_tokens = img_start_token + img_context_token * self.model.num_image_token * num_patches + img_end_token
        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        query_text = template.get_prompt().replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=256,
            min_new_tokens=1,
            eos_token_id=eos_token_id,
        )
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        return response

    def _generate_text(
        self,
        images: list[np.ndarray],
        prompt: str,
        max_new_tokens: int,
        progress_cb: Callable[[dict], None] | None = None,
        force_disable_thinking: bool = False,
    ) -> str:
        content = []
        pil_images = []
        for img in images:
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        think_suffix = "<|im_start|>assistant\n<think>\n"
        if text.endswith(think_suffix) and (force_disable_thinking or not self.reasoning_profile.enable_thinking):
            text = text[: -len(think_suffix)] + "<|im_start|>assistant\n"
        inputs = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device).to(torch.int64 if key == "input_ids" else value.dtype) if key != "images" else value.to(self.device) for key, value in inputs.items()}
        # Fix the tensor conversion for Qwen processor outputs if any
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        
        use_streamer = (
            progress_cb is not None
            and self.reasoning_profile.enable_thinking
            and self.reasoning_profile_name == "deep"
            and self.device.type != "mps"
        )

        if progress_cb is not None and not use_streamer:
            progress_cb(
                {
                    "stage": "hf_generate_start",
                    "message": f"Generating final answer up to {max_new_tokens} tokens (fast mode, token streaming disabled)",
                }
            )

        if use_streamer:
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=None,
            )
            stream_stage = "hf_reasoning" if (self.reasoning_profile.enable_thinking and not force_disable_thinking) else "hf_stream"
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": self.reasoning_profile.use_cache_on_mps if self.device.type == "mps" else True,
                "streamer": streamer,
            }
            progress_cb({"stage": "hf_generate_start", "message": f"Generating up to {max_new_tokens} new tokens"})
            error_holder: dict[str, Exception] = {}

            def _run_generate() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except Exception as exc:
                    error_holder["error"] = exc

            thread = Thread(target=_run_generate)
            thread.start()
            chunks: list[str] = []
            last_reported_len = 0
            try:
                for chunk in streamer:
                    chunks.append(chunk)
                    current_text = "".join(chunks)
                    if len(current_text) - last_reported_len >= 32 or "\n" in chunk:
                        last_reported_len = len(current_text)
                        progress_cb({"stage": stream_stage, "delta": chunk, "text": current_text})
            except queue.Empty:
                progress_cb({"stage": "hf_stream_wait", "message": "Waiting for next decoded chunk..."})
            thread.join()
            if "error" in error_holder:
                raise error_holder["error"]
            return _sanitize_model_text("".join(chunks).strip())

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=self.reasoning_profile.use_cache_on_mps if self.device.type == "mps" else True,
        )
        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        raw_ids = generated[:, prompt_len:] if prompt_len > 0 else generated
        return _sanitize_model_text(self.processor.batch_decode(raw_ids, skip_special_tokens=True)[0].strip())

    @torch.no_grad()
    def explain(
        self,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        semantic_context: str = "",
        visual_context: str = "",
        auxiliary_images: list[np.ndarray] | None = None,
        response_language: str = "English",
        progress_cb: Callable[[dict], None] | None = None,
    ) -> VLMResult:
        before_img = self._resize_if_needed(before_crop)
        after_img = self._resize_if_needed(after_crop)
        extra_images = [self._resize_if_needed(image) for image in (auxiliary_images or [])]
        all_images = [before_img, after_img, *extra_images]
        has_change_guide = bool(extra_images)
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "hf_prepare",
                    "message": (
                        f"Prepared {len(all_images)} images, max side {max(before_img.shape[:2])}, "
                        f"profile {self.reasoning_profile.label}, thinking={self.reasoning_profile.enable_thinking}"
                    ),
                }
            )

        last_raw_text = ""
        last_parsed: dict | None = None
        reasoning_text = ""
        reasoning_complete = False
        run_device = self.device
        for attempt in range(1):
            max_new_tokens = (
                self.reasoning_profile.max_new_tokens_mps
                if run_device.type == "mps"
                else self.reasoning_profile.max_new_tokens_other
            )
            try:
                reasoning_prompt = self._reasoning_prompt(response_language, visual_context, semantic_context, has_change_guide)
                if self.backend == "earthdial":
                    earthdial_aux_img = extra_images[0] if extra_images else after_img
                    if self.reasoning_profile.enable_thinking:
                        reasoning_text = _normalize_reasoning_notes(self._earthdial_chat(before_img, after_img, earthdial_aux_img, reasoning_prompt))
                        if progress_cb is not None and reasoning_text.strip():
                            progress_cb({"stage": "hf_reasoning_ready", "message": "Reasoning pass completed"})
                    raw_text = self._earthdial_chat(before_img, after_img, earthdial_aux_img, self._final_prompt(response_language, reasoning_text, visual_context, semantic_context, has_change_guide))
                else:
                    if self.reasoning_profile.enable_thinking:
                        reasoning_budget = self.reasoning_profile.reasoning_tokens_mps if run_device.type == "mps" else self.reasoning_profile.reasoning_tokens_other
                        hf_reasoning_rounds = 1
                        for round_idx in range(hf_reasoning_rounds):
                            if progress_cb is not None:
                                progress_cb({"stage": "hf_reasoning_start", "message": f"Generating private reasoning notes round {round_idx + 1}/{hf_reasoning_rounds} up to {reasoning_budget} tokens"})
                            prompt = reasoning_prompt
                            if reasoning_text.strip():
                                prompt += "\nRewrite all five labeled fields from scratch. Previous attempt was incomplete:\n" + _normalize_reasoning_notes(reasoning_text)
                            round_text = self._generate_text(
                                all_images,
                                prompt,
                                max_new_tokens=reasoning_budget,
                                progress_cb=progress_cb,
                            )
                            reasoning_text = _normalize_reasoning_notes((reasoning_text + "\n" + round_text).strip())
                            reasoning_complete = reasoning_notes_complete(reasoning_text)
                            if reasoning_complete:
                                break
                    raw_text = self._generate_text(
                        all_images,
                        self._final_prompt(response_language, reasoning_text, visual_context, semantic_context, has_change_guide),
                        max_new_tokens=max_new_tokens,
                        progress_cb=progress_cb,
                        force_disable_thinking=True,
                    )
            except Exception as exc:
                if run_device.type == "mps" and _is_mps_oom_error(exc):
                    if progress_cb is not None:
                        progress_cb({"stage": "hf_oom_fallback", "message": "MPS memory pressure detected, retrying on CPU"})
                    _clear_device_cache(run_device)
                    self.model.to("cpu")
                    run_device = torch.device("cpu")
                    self.device = run_device
                    self.dtype = torch.float32
                    continue
                raw_text = f"[vlm-error] {type(exc).__name__}: {exc}"
            
            raw_text = _sanitize_model_text(raw_text)
            last_raw_text = raw_text
            parsed = parse_model_response(raw_text)
            last_parsed = parsed
            if _looks_like_draft_output(raw_text):
                if progress_cb is not None:
                    progress_cb({"stage": "hf_finalize", "message": "Final answer looked like draft text, rerunning strict finalization pass"})
                raw_text = self._generate_text(
                    all_images,
                    self._final_prompt(response_language, reasoning_text, visual_context, semantic_context, has_change_guide) + "\nReturn only strict JSON with scene_overview.",
                    max_new_tokens=max(72, max_new_tokens // 2),
                    progress_cb=progress_cb,
                    force_disable_thinking=True,
                )
                raw_text = _sanitize_model_text(raw_text)
                last_raw_text = raw_text
                parsed = parse_model_response(raw_text)
                last_parsed = parsed
            if parsed and any(str(value).strip() for value in parsed.values()):
                break

        fallback_parsed = last_parsed or _parse_labeled_stream(last_raw_text) or _parse_structured_text(last_raw_text) or _parse_prose_text(last_raw_text) or {}

        # Ultimate fallback: if we got absolutely nothing structured, show the raw text under the detailed summary
        if not any(str(v).strip() for v in fallback_parsed.values()) and last_raw_text.strip():
            if not last_raw_text.startswith("[vlm-error]"):
                fallback_parsed["detailed_change_summary"] = last_raw_text.strip()
        _clear_device_cache(run_device)
        return VLMResult(
            raw_text=last_raw_text,
            parsed=fallback_parsed,
            reasoning_text=_normalize_reasoning_notes(reasoning_text),
            reasoning_complete=reasoning_complete or reasoning_notes_complete(reasoning_text),
            final_text=last_raw_text,
            phase_trace=["hf_transformers_legacy"],
            backend_runtime="hf_transformers_legacy",
            memory_mode="torch+mps" if run_device.type == "mps" else run_device.type,
        )

    def unload(self) -> None:
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        _clear_device_cache(self.device)


class OllamaSemanticChangeExplainer:
    def __init__(self, model_name: str = GEMMA4_E4B_OLLAMA, reasoning_profile: str = "balanced"):
        self.model_name = model_name
        self.reasoning_profile_name = reasoning_profile if reasoning_profile in REASONING_PROFILES else "balanced"
        self.reasoning_profile = REASONING_PROFILES[self.reasoning_profile_name]

    def _reasoning_prompt(self, semantic_context: str, visual_context: str, response_language: str, has_change_guide: bool) -> str:
        return "".join(
            [
                "You are analyzing land-surface change for one aligned crop.\n",
                "You will see BEFORE and AFTER images. Use them as the primary evidence.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use the contact sheet to inspect each grid cell. Treat any heatmap only as a secondary inspection aid. Do not infer object type from heatmap color alone.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Semantic-segmentation notes are weak secondary evidence and may be noisy or partially wrong.\n",
                "Use the highlighted cells to inspect local road, plot/service-line, rectilinear patch, compact-surface, and ground-texture changes.\n",
                "Use cautious hypotheses unless multiple cues agree; roof/building-specific claims require strong rectilinear built-surface evidence.\n",
            "Fill every field with actual content. Do not repeat placeholders or say 'here is a thinking process'.\n",
            f"Return only these five labeled fields in {response_language}:\n",
            "Visual evidence before: \n",
            "Visual evidence after: \n",
            "Observed differences: \n",
            "Most affected area: one or more grid cells like A1, A2, B3, C4, or unclear\n",
                "Uncertainty notes: \n",
                "No intro. No numbering. No extra text.\n\n",
                "Deterministic change-tool hints:\n",
                f"{visual_context}\n\n",
                "Secondary semantic notes:\n",
                f"{semantic_context}\n",
            ]
        )

    def _reasoning_continue_prompt(self, semantic_context: str, visual_context: str, response_language: str, reasoning_text: str, has_change_guide: bool) -> str:
        return (
            self._reasoning_prompt(semantic_context=semantic_context, visual_context=visual_context, response_language=response_language, has_change_guide=has_change_guide)
            + "\nRewrite all five labeled fields from scratch.\n"
            + "Previous attempt was incomplete or contained meta-commentary.\nPrevious attempt:\n"
            + _normalize_reasoning_notes(reasoning_text).strip()
        )

    def _final_prompt(self, semantic_context: str, visual_context: str, response_language: str, reasoning_text: str, has_change_guide: bool) -> str:
        return "".join(
            [
                "You are analyzing land-surface change for one aligned crop.\n",
                "You will see BEFORE and AFTER images. Use them as the primary evidence.\n",
                "Additional images may include a labeled A1..D4 before/after contact sheet and a change-guide heatmap.\n" if has_change_guide else "",
                "Use the contact sheet to inspect each grid cell. Treat any heatmap only as a secondary inspection aid. Do not infer object type from heatmap color alone.\n",
                "Translate visual cues into meaningful land-use processes: construction activity, new access road or track, graded or cleared land, excavation, new building or roof, yard expansion, field preparation, or stable surface.\n",
                "Do not stop at low-level phrases like 'new lines', 'color changed', or 'tone changed'; explain what those cues most likely mean and state uncertainty.\n",
                "Use the reasoning notes only as scratch analysis. Do not reveal chain-of-thought or analysis steps.\n",
                "Return strict JSON only with exactly these keys: scene_overview, before_summary, after_summary, main_changes, cell_observations.\n",
                "scene_overview must be one short paragraph explaining the overall physical/geographic change pattern.\n",
                "before_summary must describe the visible surface state in BEFORE.\n",
                "after_summary must describe the visible surface state in AFTER.\n",
                "main_changes must be 2 to 5 short plain-English strings.\n",
                "cell_observations must contain exactly 16 objects, one for every cell A1, A2, A3, A4, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3, D4.\n",
                "Each cell object must have keys cell, observation, confidence. Confidence must be high, medium, low, or uncertain.\n",
                "For each changed cell, explicitly connect evidence to meaning, for example: 'new pale rectangular pads and access tracks suggest active construction or site preparation', 'a continuous dark linear feature suggests a new road or service track', 'rougher exposed soil suggests excavation or recently disturbed ground'.\n",
                "Describe visible physical/geographic changes cautiously: possible construction activity, new access road or track, new building or roof, compacted pad or yard, plot or service-line trace, road-edge rework, grading, clearing, excavation, field preparation, ambiguous local reworking, or stable area.\n",
                "Only call something roof/building-specific when multiple strong visual cues agree. Prefer physical interpretation over raw low-level cues. Do not use 'tone change' as likely_change when structural cues exist.\n",
                f"Write the answer in {response_language}.\n\n",
                "Reasoning notes:\n",
                f"{reasoning_text}\n\n",
                "Deterministic change-tool hints:\n",
                f"{visual_context}\n\n",
                "Secondary semantic notes:\n",
                f"{semantic_context}\n",
            ]
        )

    def _chat_request(
        self,
        prompt: str,
        images: list[np.ndarray],
        think_mode: bool | str,
        num_predict: int,
        num_ctx: int,
        response_format: dict | None,
        progress_cb: Callable[[dict], None] | None,
        stage_prefix: str,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "stream": progress_cb is not None,
            "think": think_mode,
            "keep_alive": "0",
            "options": {
                "temperature": 0.0,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "num_thread": max(1, (os.cpu_count() or 4) - 2),
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if response_format is not None:
            payload["format"] = response_format

        import base64
        from io import BytesIO

        def _encode(img: Image.Image) -> str:
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        payload["messages"][0]["images"] = [_encode(Image.fromarray(image)) for image in images]
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": f"{stage_prefix}_prepare",
                    "message": f"Prepared request for {self.model_name}, think={think_mode}, num_predict={num_predict}, num_ctx={num_ctx}",
                }
            )
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as response:
            if progress_cb is None:
                return json.loads(response.read().decode("utf-8"))
            thinking_chunks: list[str] = []
            content_chunks: list[str] = []
            body = {}
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                body = event
                message = event.get("message", {}) if isinstance(event, dict) else {}
                thinking_delta = message.get("thinking")
                content_delta = message.get("content")
                if thinking_delta:
                    thinking_chunks.append(str(thinking_delta))
                    progress_cb({"stage": "ollama_thinking", "delta": str(thinking_delta), "text": "".join(thinking_chunks)})
                if content_delta:
                    content_chunks.append(str(content_delta))
                    progress_cb({"stage": "ollama_content", "delta": str(content_delta), "text": "".join(content_chunks)})
                if event.get("done"):
                    return {
                        **event,
                        "message": {
                            **message,
                            "thinking": "".join(thinking_chunks),
                            "content": "".join(content_chunks),
                        },
                    }
            return body

    def explain(
        self,
        before_crop: np.ndarray,
        after_crop: np.ndarray,
        semantic_context: str,
        visual_context: str = "",
        auxiliary_images: list[np.ndarray] | None = None,
        response_language: str = "English",
        progress_cb: Callable[[dict], None] | None = None,
    ) -> VLMResult:
        before_img = _to_uint8_rgb(before_crop)
        after_img = _to_uint8_rgb(after_crop)
        extra_images = [_to_uint8_rgb(image) for image in (auxiliary_images or [])]
        all_images = [before_img, after_img, *extra_images]
        has_change_guide = bool(extra_images)
        schema = {
            "type": "object",
            "properties": {
                "scene_overview": {"type": "string"},
                "before_summary": {"type": "string"},
                "after_summary": {"type": "string"},
                "main_changes": {"type": "array", "items": {"type": "string"}},
                "cell_observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cell": {"type": "string"},
                            "observation": {"type": "string"},
                            "confidence": {"type": "string"},
                        },
                        "required": ["cell", "observation", "confidence"],
                    },
                },
            },
            "required": ["scene_overview", "before_summary", "after_summary", "main_changes", "cell_observations"],
        }
        try:
            reasoning_text = ""
            for round_idx in range(self.reasoning_profile.reasoning_rounds):
                reasoning_prompt = (
                    self._reasoning_prompt(semantic_context=semantic_context, visual_context=visual_context, response_language=response_language, has_change_guide=has_change_guide)
                    if not reasoning_text.strip()
                    else self._reasoning_continue_prompt(
                        semantic_context=semantic_context,
                        visual_context=visual_context,
                        response_language=response_language,
                        reasoning_text=reasoning_text,
                        has_change_guide=has_change_guide,
                    )
                )
                reasoning_body = self._chat_request(
                    prompt=reasoning_prompt,
                    images=all_images,
                    think_mode=self.reasoning_profile.ollama_think,
                    num_predict=max(128, self.reasoning_profile.ollama_num_predict // 2),
                    num_ctx=self.reasoning_profile.ollama_num_ctx,
                    response_format=None,
                    progress_cb=progress_cb,
                    stage_prefix="ollama_reasoning",
                )
                reasoning_message = reasoning_body.get("message", {}) if isinstance(reasoning_body, dict) else {}
                visible_reasoning = _sanitize_reasoning_text(str(reasoning_message.get("content", "")).strip())
                private_reasoning = _sanitize_reasoning_text(str(reasoning_message.get("thinking", "")).strip())
                candidate = visible_reasoning
                if not reasoning_notes_complete(candidate):
                    candidate = private_reasoning
                if candidate.strip():
                    reasoning_text = _normalize_reasoning_notes(candidate)
                if reasoning_notes_complete(reasoning_text):
                    break
            final_body = self._chat_request(
                prompt=self._final_prompt(semantic_context=semantic_context, visual_context=visual_context, response_language=response_language, reasoning_text=reasoning_text, has_change_guide=has_change_guide),
                images=all_images,
                think_mode=False,
                num_predict=self.reasoning_profile.ollama_num_predict,
                num_ctx=self.reasoning_profile.ollama_num_ctx,
                response_format="json",
                progress_cb=progress_cb,
                stage_prefix="ollama_final",
            )
        except Exception as exc:
            raw_text = f"[ollama-error] {type(exc).__name__}: {exc}"
            return VLMResult(raw_text=raw_text, parsed={}, reasoning_text="", backend_runtime="ollama_gemma", memory_mode=f"keep_alive=0 ctx={self.reasoning_profile.ollama_num_ctx}")
        message = final_body.get("message", {}) if isinstance(final_body, dict) else {}
        raw_text = _sanitize_model_text(str(message.get("content", "")).strip())
        parsed = parse_model_response(raw_text)
        if not has_complete_final_response(parsed):
            try:
                retry_body = self._chat_request(
                    prompt=self._final_prompt(
                        semantic_context=semantic_context,
                        visual_context=visual_context,
                        response_language=response_language,
                        reasoning_text=reasoning_text,
                        has_change_guide=has_change_guide,
                    )
                    + "\nReturn only strict JSON. Include exactly 16 cell_observations, one for every A1..D4 cell. Do not repeat the same observation across cells. Do not use semantic class labels such as grass, cropland, bareland, or background as evidence.",
                    images=all_images,
                    think_mode=False,
                    num_predict=self.reasoning_profile.ollama_num_predict,
                    num_ctx=self.reasoning_profile.ollama_num_ctx,
                    response_format="json",
                    progress_cb=progress_cb,
                    stage_prefix="ollama_final_retry",
                )
                retry_message = retry_body.get("message", {}) if isinstance(retry_body, dict) else {}
                retry_text = _sanitize_model_text(str(retry_message.get("content", "")).strip())
                retry_parsed = parse_model_response(retry_text)
                if has_complete_final_response(retry_parsed):
                    raw_text = retry_text
                    parsed = retry_parsed
            except Exception:
                pass
        return VLMResult(
            raw_text=raw_text,
            parsed=parsed,
            reasoning_text=reasoning_text,
            reasoning_complete=reasoning_notes_complete(reasoning_text) or bool(reasoning_text.strip()),
            final_text=raw_text,
            phase_trace=[f"ollama_reasoning think={self.reasoning_profile.ollama_think}", f"ollama_final ctx={self.reasoning_profile.ollama_num_ctx}"],
            backend_runtime="ollama_gemma",
            memory_mode=f"keep_alive=0 ctx={self.reasoning_profile.ollama_num_ctx}",
        )
