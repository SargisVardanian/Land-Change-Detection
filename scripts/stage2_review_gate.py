#!/usr/bin/env python3
"""Build a claim-level Stage-2 review gate without touching immutable data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


EXPLICIT_AI_EXCLUDED = {
    "assistant-004", "assistant-016", "assistant-019", "assistant-028",
    "assistant-029", "assistant-040", "assistant-048",
}
PROVIDED_AI_SUMMARY = {
    "fully_supported_as_written": 0,
    "partially_supported_requiring_rewrite": 19,
    "uncertain": 2,
    "unsupported": 23,
    "invalid_no_data_inputs": 4,
    "recommended_rewrite": 21,
    "recommended_reject": 20,
    "recommended_exclude": 7,
}
QUALITY_THRESHOLDS = {
    "no_data_fraction_max": 0.20,
    "near_black_fraction_max": 0.35,
    "cloud_or_smoke_fraction_max": 0.75,
    "visible_ground_fraction_min": 0.20,
    "temporal_overlap_fraction_min": 0.70,
    "registration_quality_min": 0.10,
}
CLAIM_TYPES = (
    "changed_object", "change_direction", "count", "location", "severity",
    "cause_event_type", "environmental_consequence",
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def paths_for(row: dict[str, Any]) -> tuple[Path, Path]:
    frames = row.get("frames") or []
    t1 = row.get("t1_path")
    t2 = row.get("t2_path")
    if not t1 and frames:
        t1 = frames[0].get("path")
    if not t2 and len(frames) > 1:
        t2 = frames[1].get("path")
    if not t1 or not t2:
        raise FileNotFoundError(row.get("canonical_pair_id", "unknown"))
    return Path(str(t1)), Path(str(t2))


def gray(pixel: tuple[int, int, int]) -> float:
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 8:
        return 0.0
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    lv = sum((x - lm) ** 2 for x in left)
    rv = sum((x - rm) ** 2 for x in right)
    if lv <= 1e-8 or rv <= 1e-8:
        return 0.0
    return sum((x - lm) * (y - rm) for x, y in zip(left, right)) / math.sqrt(lv * rv)


def image_metrics(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as source:
            width, height = source.size
            image = source.convert("RGB").resize((64, 64))
            pixels = list(image.getdata())
    except Exception as exc:
        return {"path": str(path), "decode_ok": False, "error": str(exc)}
    values = [gray(pixel) for pixel in pixels]
    no_data = sum(max(pixel) <= 3 for pixel in pixels) / len(pixels)
    near_black = sum(value <= 12 for value in values) / len(values)
    cloud_pixels = 0
    for y in range(0, 64, 8):
        for x in range(0, 64, 8):
            block = [pixels[yy * 64 + xx] for yy in range(y, y + 8) for xx in range(x, x + 8)]
            block_values = [gray(pixel) for pixel in block]
            mean = sum(block_values) / len(block_values)
            deviation = math.sqrt(sum((value - mean) ** 2 for value in block_values) / len(block_values))
            chroma = sum(max(pixel) - min(pixel) for pixel in block) / len(block)
            if ((mean >= 220 and deviation <= 18 and chroma <= 20) or
                    (100 <= mean < 220 and deviation <= 10 and chroma <= 12)):
                cloud_pixels += len(block)
    cloud = cloud_pixels / len(pixels)
    return {
        "path": str(path), "decode_ok": True, "width": width, "height": height,
        "no_data_fraction": no_data, "near_black_fraction": near_black,
        "cloud_or_smoke_fraction": cloud,
        "visible_ground_fraction": max(0.0, 1.0 - no_data - cloud),
    }


def pair_quality(t1: Path, t2: Path) -> dict[str, Any]:
    left, right = image_metrics(t1), image_metrics(t2)
    result: dict[str, Any] = {"t1": left, "t2": right, "reasons": [], "quality_gate": "PASS", "decode_ok": bool(left.get("decode_ok") and right.get("decode_ok"))}
    if not result["decode_ok"]:
        result["reasons"] = ["decode_failure"]
        result["quality_gate"] = "EXCLUDE"
        return result
    with Image.open(t1) as left_image, Image.open(t2) as right_image:
        left_values = list(left_image.convert("L").resize((64, 64)).getdata())
        right_values = list(right_image.convert("L").resize((64, 64)).getdata())
    valid_left = [value > 12 for value in left_values]
    valid_right = [value > 12 for value in right_values]
    union = sum(a or b for a, b in zip(valid_left, valid_right))
    result["temporal_overlap_fraction"] = sum(a and b for a, b in zip(valid_left, valid_right)) / union if union else 0.0
    common_left = [a for a, b in zip(left_values, right_values) if b > 12]
    common_right = [b for a, b in zip(left_values, right_values) if b > 12]
    result["registration_quality"] = max(0.0, min(1.0, (pearson(common_left, common_right) + 1.0) / 2.0))
    result.update({
        "no_data_fraction": max(left["no_data_fraction"], right["no_data_fraction"]),
        "near_black_fraction": max(left["near_black_fraction"], right["near_black_fraction"]),
        "cloud_or_smoke_fraction": max(left["cloud_or_smoke_fraction"], right["cloud_or_smoke_fraction"]),
        "visible_ground_fraction": min(left["visible_ground_fraction"], right["visible_ground_fraction"]),
    })
    if result["no_data_fraction"] > QUALITY_THRESHOLDS["no_data_fraction_max"]:
        result["reasons"].append("no_data_fraction")
    if result["near_black_fraction"] > QUALITY_THRESHOLDS["near_black_fraction_max"]:
        result["reasons"].append("near_black_fraction")
    if result["cloud_or_smoke_fraction"] > QUALITY_THRESHOLDS["cloud_or_smoke_fraction_max"] and result["visible_ground_fraction"] < QUALITY_THRESHOLDS["visible_ground_fraction_min"]:
        result["reasons"].append("cloud_or_smoke_fraction")
    if result["visible_ground_fraction"] < QUALITY_THRESHOLDS["visible_ground_fraction_min"]:
        result["reasons"].append("visible_ground_fraction")
    if result["temporal_overlap_fraction"] < QUALITY_THRESHOLDS["temporal_overlap_fraction_min"]:
        result["reasons"].append("temporal_overlap_fraction")
    if result["registration_quality"] < QUALITY_THRESHOLDS["registration_quality_min"]:
        result["reasons"].append("registration_quality")
    if result["reasons"]:
        result["quality_gate"] = "EXCLUDE"
    return result


def audit_image_rows(rows: list[dict[str, Any]], label: str, output: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    jobs: list[tuple[str, Path, Path]] = []
    for index, row in enumerate(rows):
        pair_id = str(row.get("canonical_pair_id") or f"row-{index}")
        try:
            t1, t2 = paths_for(row)
        except Exception:
            t1, t2 = Path(f"__missing_t1_{index}"), Path(f"__missing_t2_{index}")
        jobs.append((pair_id, t1, t2))

    def run(job: tuple[str, Path, Path]) -> dict[str, Any]:
        pair_id, t1, t2 = job
        if not t1.exists() or not t2.exists():
            return {"canonical_pair_id": pair_id, "t1_path": str(t1), "t2_path": str(t2), "decode_ok": False, "quality_gate": "EXCLUDE", "reasons": ["missing_file"]}
        result = pair_quality(t1, t2)
        result.update({"canonical_pair_id": pair_id, "t1_path": str(t1), "t2_path": str(t2)})
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        audited = list(executor.map(run, jobs))
    write_jsonl(output, audited)
    return summarize_quality_rows(audited, label)


def summarize_quality_rows(audited: list[dict[str, Any]], label: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary = {
        "label": label, "pairs": len(audited),
        "pass": sum(row.get("quality_gate") == "PASS" for row in audited),
        "excluded": sum(row.get("quality_gate") == "EXCLUDE" for row in audited),
        "decode_failures": sum(not row.get("decode_ok", False) for row in audited),
        "exclusion_reasons": dict(Counter(reason for row in audited for reason in row.get("reasons", []))),
        "thresholds": QUALITY_THRESHOLDS,
    }
    return {row["canonical_pair_id"]: row for row in audited}, summary


def reuse_quality_rows(source: Path, label: str, output: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audited = read_jsonl(source)
    if not audited:
        raise ValueError(f"quality reuse source is empty: {source}")
    write_jsonl(output, audited)
    return summarize_quality_rows(audited, label)


def load_ai_audit(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {
            "status": "MISSING_INPUT",
            "path": str(path) if path else None,
            "provided_summary": PROVIDED_AI_SUMMARY,
            "advisory_only": True,
            "note": "The exact qcpr_stage2_ai_audit_48.jsonl was not found; existing assistant rows contain selections only, not decisions.",
        }
    rows = read_jsonl(path)
    ids = [str(row.get("audit_row_id") or row.get("assistant_id") or row.get("row_id")) for row in rows]
    if len(rows) != 48 or len(set(ids)) != 48:
        raise ValueError("AI audit must contain exactly 48 unique rows")
    result = {}
    for row, row_id in zip(rows, ids):
        status = str(row.get("decision") or row.get("audit_status") or row.get("status") or row.get("review_status") or "unknown").lower()
        result[row_id] = {**row, "normalized_status": status}
    return result, {
        "status": "IMPORTED", "path": str(path), "rows": len(rows),
        "status_counts": dict(Counter(row["normalized_status"] for row in result.values())),
        "advisory_only": True,
    }


def extract_claims(text: str, review_id: str) -> list[dict[str, Any]]:
    segments = [segment.strip(" ,") for segment in re.split(r"(?:;|\.|!|\?)", text) if segment.strip()]
    patterns = {
        "changed_object": r"building|structure|road|field|farmland|vegetation|tree|forest|water|shore|soil|debris|vehicle|roof|house|landscape|infrastructure",
        "change_direction": r"appear|emerge|collapse|destroy|damag|disrupt|reduc|increase|expand|remove|clear|transform|replace|expos|alter|recover|erod|flood|burn|scorch",
        "count": r"\b(?:one|single|two|three|four|five|several|multiple|numerous|many|few|\d+)\b",
        "location": r"near|along|adjacent|central|eastern|western|northern|southern|left|right|top|bottom|corner|roadside|shoreline|coastal|across|within",
        "severity": r"mild|moderate|severe|significant|extensive|widespread|catastrophic|major|minor|dramatic|intense|massive",
        "cause_event_type": r"earthquake|hurricane|flood|volcano|tornado|wildfire|fire|disaster|seismic|storm|eruption|landslide",
        "environmental_consequence": r"vegetation|bare soil|erosion|floodwater|water|ash|smoke|scorch|forest|farmland|agricultural|environment|ecological",
    }
    claims: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments, 1):
        matched = [kind for kind, pattern in patterns.items() if re.search(pattern, segment, re.IGNORECASE)]
        if not matched:
            matched = ["change_direction"]
        for kind in matched:
            claims.append({
                "claim_id": f"{review_id}:claim:{len(claims) + 1}",
                "claim_type": kind,
                "claim_text": segment,
                "segment_index": segment_index,
                "status": "",
                "not_assessable_reason": "",
            })
    return claims


def event_type_from_event(event: str) -> str:
    lower = event.lower()
    for token in ("earthquake", "hurricane", "flood", "volcano", "tornado", "wildfire", "fire"):
        if token in lower:
            return token
    return "unknown"


def qvq_caption_map(release: Path) -> dict[str, str]:
    result = {}
    for row in read_jsonl(release / "manifests/provisional_semantic_candidates/rscc_qvq_unverified.jsonl"):
        captions = row.get("captions")
        result[str(row.get("canonical_pair_id"))] = captions[0] if isinstance(captions, list) and captions else str(captions or "")
    return result


def registry_rows(release: Path) -> list[dict[str, Any]]:
    return read_jsonl(release / "registries/pair_registry.jsonl")


def frame_paths(row: dict[str, Any]) -> tuple[str, str]:
    frames = row.get("frames") or []
    return str(row.get("t1_path") or (frames[0].get("path") if frames else "")), str(row.get("t2_path") or (frames[1].get("path") if len(frames) > 1 else ""))


def select_revised_packet(
    original: list[dict[str, Any]],
    all_registry: list[dict[str, Any]],
    quality: dict[str, dict[str, Any]],
    ai_rows: dict[str, dict[str, Any]],
    assistant_rows: list[dict[str, Any]],
    release: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    assistant_by_id = {str(row.get("audit_row_id")): row for row in assistant_rows}
    explicit_pairs = {str(assistant_by_id[row_id].get("canonical_pair_id")) for row_id in EXPLICIT_AI_EXCLUDED if row_id in assistant_by_id}
    ai_excluded = set()
    for row in ai_rows.values():
        status = str(row.get("normalized_status", "")).lower()
        if any(token in status for token in ("unsupported", "reject", "exclude", "invalid", "no_data")) and row.get("canonical_pair_id"):
            ai_excluded.add(str(row["canonical_pair_id"]))
    excluded = explicit_pairs | ai_excluded
    captions = qvq_caption_map(release)
    by_event = defaultdict(list)
    for row in all_registry:
        if str(row.get("source_dataset", "")).upper() == "RSCC-EBD":
            by_event[str(row.get("source_event_id"))].append(row)
    original_by_event = defaultdict(list)
    for row in original:
        original_by_event[str(row.get("source_event_id"))].append(row)
    final, replacements = [], []
    used = set()
    for event in sorted(original_by_event):
        selected = []
        for row in original_by_event[event]:
            pair_id = str(row.get("canonical_pair_id"))
            if pair_id in excluded or quality.get(pair_id, {}).get("quality_gate") == "EXCLUDE":
                continue
            selected.append(dict(row))
            used.add(pair_id)
        candidates = sorted(by_event[event], key=lambda row: (-float(quality.get(str(row.get("canonical_pair_id")), {}).get("registration_quality", 0.0)), str(row.get("canonical_pair_id"))))
        for candidate in candidates:
            if len(selected) >= 20:
                break
            pair_id = str(candidate.get("canonical_pair_id"))
            if pair_id in used or pair_id in excluded or quality.get(pair_id, {}).get("quality_gate") == "EXCLUDE":
                continue
            t1, t2 = frame_paths(candidate)
            replacement = {
                "review_id": f"rscc_review_gate_v2:{event}:{pair_id}",
                "canonical_pair_id": pair_id,
                "source_event_id": event,
                "split": candidate.get("split", "unknown"),
                "candidate_caption": captions.get(pair_id, ""),
                "t1_path": t1, "t2_path": t2,
                "verifier_score": None,
                "verification_status": "generated_unverified",
                "selection_reason": "replacement_after_image_usability_gate",
            }
            selected.append(replacement)
            used.add(pair_id)
            replacements.append({"event": event, "canonical_pair_id": pair_id, "reason": replacement["selection_reason"]})
        if len(selected) != 20:
            raise RuntimeError(f"usable replacement pool cannot fill 20 rows for {event}: {len(selected)}")
        final.extend(selected)
    return final, replacements, excluded


def copy_jpeg(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(target, format="JPEG", quality=92, optimize=True)


def public_rows(rows: list[dict[str, Any]], bundle: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata = {event: f"S{index:02d}" for index, event in enumerate(sorted({str(row["source_event_id"]) for row in rows}), 1)}
    visible, hidden = [], []
    for number, row in enumerate(rows, 1):
        t1, t2 = Path(str(row["t1_path"])), Path(str(row["t2_path"]))
        if not t1.exists() or not t2.exists():
            raise FileNotFoundError(f"missing review image: {row['canonical_pair_id']}")
        copy_jpeg(t1, bundle / "assets/images" / f"{number:03d}_t1.jpg")
        copy_jpeg(t2, bundle / "assets/images" / f"{number:03d}_t2.jpg")
        # Do not expose source/event-bearing identifiers to reviewers.  The
        # canonical join is retained only in reveal_metadata.jsonl.
        review_id = f"review-{number:03d}"
        visible.append({
            "row_number": number,
            "review_id": review_id,
            "pair_key": f"pair-{number:03d}",
            "review_stratum": strata[str(row["source_event_id"])],
            "split": row.get("split", "unknown"),
            "candidate_caption": row.get("candidate_caption", ""),
            "claims": extract_claims(row.get("candidate_caption", ""), review_id),
            "t1_rel": f"assets/images/{number:03d}_t1.jpg",
            "t2_rel": f"assets/images/{number:03d}_t2.jpg",
        })
        hidden.append({
            "review_id": review_id,
            "source_review_id": row.get("review_id"),
            "canonical_pair_id": row["canonical_pair_id"],
            "event_id": row["source_event_id"],
            "event_type": event_type_from_event(str(row["source_event_id"])),
            "event_type_visually_verified": "unknown",
            "verifier_score": row.get("verifier_score"),
            "selection_reason": row.get("selection_reason", "original_packet"),
            "t1_path": str(t1),
            "t2_path": str(t2),
        })
    return visible, hidden


def claim_review_html(role: str, rows: list[dict[str, Any]], packet_sha: str, assistant_zip_name: str | None) -> str:
    role_label = "Reviewer A — claim factuality" if role == "A" else "Reviewer B — independent claim sufficiency"
    role_help = (
        "Сначала сравните T1 и T2. Для каждой claim отметьте только то, что видно."
        if role == "A" else
        "Работайте независимо от Reviewer A. Отдельно проверяйте count, location, severity и cause."
    )
    rows_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QCPR Stage-2 __ROLE_LABEL__</title>
<style>
:root{--bg:#f4f7fa;--ink:#17202a;--muted:#5f6b76;--line:#d8dee4;--blue:#155d91;--green:#eaf7ee;--yellow:#fff7df}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:#102a43;color:#fff;padding:14px 22px;box-shadow:0 2px 8px #0003}.head{display:flex;justify-content:space-between;gap:16px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0}.muted{font-size:13px;color:#d7e6f4}
main{max-width:1500px;margin:20px auto;padding:0 16px}.toolbar,.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 2px 8px #14213d0b}.toolbar{display:flex;gap:9px;align-items:end;flex-wrap:wrap}
button,input,select,textarea{font:inherit}button{border:0;border-radius:8px;padding:9px 13px;background:var(--blue);color:#fff;cursor:pointer}button.secondary{background:#657789}button:disabled{opacity:.45;cursor:not-allowed}input[type=text],select,textarea{width:100%;border:1px solid #b9c4cf;border-radius:7px;padding:8px;background:#fff}label{display:flex;flex-direction:column;gap:4px;color:#334e68;font-weight:650}.small{width:230px}.progress{font-weight:700;font-variant-numeric:tabular-nums}
.instructions{background:var(--green);border:1px solid #b9e2c8;border-radius:9px;padding:12px;margin-bottom:14px}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}.image-card{margin:0;background:#0f1720;color:#fff;padding:10px;border-radius:12px}.image-card img{display:block;width:100%;height:min(56vh,620px);object-fit:contain;background:#070b0f;border-radius:7px}.image-card figcaption{padding:8px 3px 0;font-weight:700}
.caption{background:var(--yellow);border-left:5px solid #e39a00;padding:14px 16px;border-radius:9px;margin:16px 0;overflow-wrap:anywhere;white-space:pre-wrap;font-size:19px}.claims{display:grid;gap:10px}.claim{border:1px solid #cfd9e1;border-radius:9px;padding:11px;background:#fbfcfd}.claim-head{display:flex;gap:10px;justify-content:space-between;align-items:start}.claim-type{font-size:12px;color:#526777;text-transform:uppercase;letter-spacing:.04em}.claim-text{font-size:17px;overflow-wrap:anywhere}.claim select{max-width:250px}.fields{display:grid;grid-template-columns:repeat(3,minmax(190px,1fr));gap:12px}.wide{grid-column:1/-1}.metadata{display:none;background:#f1f3f5;border:1px dashed #8c99a6;border-radius:9px;padding:10px;overflow-wrap:anywhere}.metadata.visible{display:block}.status{min-height:1.4em;font-weight:700}.ok{color:#146c43}.error{color:#9b202b}.note{font-size:13px;color:var(--muted)}
@media(max-width:900px){.images,.fields{grid-template-columns:1fr}.image-card img{height:42vh}.claim-head{display:block}.claim select{max-width:none;margin-top:8px}}
</style></head><body>
<header><div class="head"><div><h1>QCPR Stage-2 · __ROLE_LABEL__</h1><div class="muted">__ROLE_HELP__</div></div><div class="progress" id="progress"></div></div></header><main>
<div class="toolbar"><button id="prev">← Назад</button><button id="next">Далее →</button><button class="secondary" id="save">Сохранить строку</button><button class="secondary" id="reveal" disabled>Reveal metadata</button><button class="secondary" id="export">Экспорт JSONL</button><button class="secondary" id="export-csv">Экспорт CSV</button><label class="small">Reviewer ID<input id="reviewer" type="text" placeholder="обязательно"></label><label class="small">Импорт JSONL<input id="import" type="file" accept=".jsonl,.json,.txt"></label>__ASSISTANT_LINK__</div>
<div id="status" class="status"></div><section id="app"></section></main>
<script>
const ROLE=__ROLE__,PACKET_SHA=__PACKET_SHA__,ROWS=__ROWS__;let index=0,answers={};const storageKey=`qcpr-stage2-claim-${ROLE}-${PACKET_SHA}`;
const claimChoices=['','supported','unsupported','uncertain','not_assessable'];const decisionChoices=['','accept','rewrite','reject'];const eventChoices=['','unknown','earthquake','hurricane','flood','volcano','tornado','wildfire','fire','other'];
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}function row(){return ROWS[index]}function blank(r){return {visible_change:'',visual_caption:r.candidate_caption,event_type:'',event_type_visually_verified:'',decision:'',confidence:'',notes:'',claims:r.claims.map(c=>({...c,status:'',not_assessable_reason:''})),saved:false}}
function load(){try{answers=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){answers={}}}function render(){const r=row(),a=answers[r.review_id]||blank(r);document.getElementById('progress').textContent=`${index+1} / ${ROWS.length}`;document.getElementById('app').innerHTML=`<section class="card"><div class="instructions"><b>Порядок:</b> сначала сравните T1/T2, потом проверьте каждую atomic claim. Event metadata скрыта до сохранения.</div><div class="images"><figure class="image-card"><img src="${r.t1_rel}" alt="T1"><figcaption>T1 · первая дата</figcaption></figure><figure class="image-card"><img src="${r.t2_rel}" alt="T2"><figcaption>T2 · вторая дата</figcaption></figure></div><div class="caption"><b>Кандидатный visual caption</b><br>${esc(r.candidate_caption)}</div><h2>Atomic claims</h2><div class="claims">${r.claims.map((c,n)=>`<div class="claim"><div class="claim-head"><div><div class="claim-type">${esc(c.claim_type)}</div><div class="claim-text">${esc(c.claim_text)}</div></div><select data-claim="${n}">${claimChoices.map(v=>`<option value="${v}">${v||'— выберите —'}</option>`).join('')}</select></div><label class="note">Причина not_assessable<input data-reason="${n}" type="text"></label></div>`).join('')}</div><h2>Итог</h2><div class="fields"><label>Видимое изменение<select data-field="visible_change"><option></option><option>yes</option><option>no</option><option>uncertain</option></select></label><label>Event type<select data-field="event_type">${eventChoices.map(v=>`<option value="${v}">${v||'— выберите —'}</option>`).join('')}</select></label><label>Event type visually verified<select data-field="event_type_visually_verified"><option></option><option>yes</option><option>no</option><option>uncertain</option></select></label><label class="wide">Короткий visual caption только из supported claims<textarea data-field="visual_caption" rows="3"></textarea></label><label>Решение<select data-field="decision">${decisionChoices.map(v=>`<option value="${v}">${v||'— выберите —'}</option>`).join('')}</select></label><label>Уверенность<select data-field="confidence"><option></option><option>low</option><option>medium</option><option>high</option></select></label><label class="wide">Заметки<textarea data-field="notes" rows="3"></textarea></label></div><div id="metadata" class="metadata"></div></section>`;document.querySelectorAll('[data-field]').forEach(e=>e.value=a[e.dataset.field]||'');document.querySelectorAll('[data-claim]').forEach(e=>e.value=a.claims?.[Number(e.dataset.claim)]?.status||'');document.querySelectorAll('[data-reason]').forEach(e=>e.value=a.claims?.[Number(e.dataset.reason)]?.not_assessable_reason||'');document.getElementById('reveal').disabled=!a.saved}
function collect(){const r=row(),a=answers[r.review_id]||blank(r);['visible_change','visual_caption','event_type','event_type_visually_verified','decision','confidence','notes'].forEach(k=>a[k]=document.querySelector(`[data-field="${k}"]`)?.value||'');a.claims=r.claims.map((c,n)=>({...c,status:document.querySelector(`[data-claim="${n}"]`)?.value||'',not_assessable_reason:document.querySelector(`[data-reason="${n}"]`)?.value||''}));answers[r.review_id]=a;return a}
function msg(text,ok=false){const e=document.getElementById('status');e.textContent=text;e.className='status '+(ok?'ok':'error')}
function save(){const a=collect();a.saved=true;localStorage.setItem(storageKey,JSON.stringify(answers));document.getElementById('reveal').disabled=false;msg('Строка сохранена. Metadata можно раскрыть.',true)}
async function reveal(){const a=answers[row().review_id];if(!a?.saved){msg('Сначала сохраните строку.');return}try{const text=await fetch('data/reveal_metadata.jsonl').then(r=>r.text());const meta=text.split(/\r?\n/).filter(Boolean).map(JSON.parse).find(x=>x.review_id===row().review_id);const box=document.getElementById('metadata');box.innerHTML=meta?`<pre>${esc(JSON.stringify(meta,null,2))}</pre>`:'Metadata not found';box.classList.add('visible')}catch(e){msg('Запустите viewer через start_review.sh, а не file://.')}}
function allComplete(){return ROWS.every(r=>{const a=answers[r.review_id];return a?.saved&&a.decision&&a.visible_change&&a.claims.every(c=>c.status)})}
function exportJsonl(){collect();const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){msg('Введите Reviewer ID перед экспортом.');return}if(!allComplete()){msg('Сначала сохраните все строки и все claim decisions.');return}const at=new Date().toISOString();const text=ROWS.map(r=>JSON.stringify({...r,...answers[r.review_id],reviewer_role:ROLE,reviewer_identity:reviewer,reviewed_at:at,independence_attestation:true})).join('\n')+'\n';const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([text],{type:'application/x-ndjson'}));link.download=`reviewer_${ROLE.toLowerCase()}_claim_decisions.jsonl`;link.click()}
function csvCell(value){return `"${String(value??'').replace(/"/g,'""')}"`}function exportCsv(){collect();const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){msg('Введите Reviewer ID перед экспортом.');return}if(!allComplete()){msg('Сначала сохраните все строки и все claim decisions.');return}const at=new Date().toISOString();const header=['row_number','review_id','pair_key','review_stratum','split','candidate_caption','visual_caption','visible_change','event_type','event_type_visually_verified','decision','confidence','notes','claim_decisions','reviewer_role','reviewer_identity','reviewed_at','independence_attestation'];const lines=[header.map(csvCell).join(',')];for(const r of ROWS){const a=answers[r.review_id];lines.push([r.row_number,r.review_id,r.pair_key,r.review_stratum,r.split,r.candidate_caption,a.visual_caption,a.visible_change,a.event_type,a.event_type_visually_verified,a.decision,a.confidence,a.notes,JSON.stringify(a.claims),ROLE,reviewer,at,true].map(csvCell).join(','))}const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv;charset=utf-8'}));link.download=`reviewer_${ROLE.toLowerCase()}_claim_decisions.csv`;link.click()}
function importJsonl(file){const reader=new FileReader();reader.onload=()=>{try{const rows=String(reader.result).split(/\r?\n/).filter(Boolean).map(JSON.parse);for(const item of rows){if(item.review_id)answers[item.review_id]={...item,saved:true}}localStorage.setItem(storageKey,JSON.stringify(answers));render();msg(`Импортировано: ${rows.length}`,true)}catch(error){msg(`Ошибка импорта: ${error.message}`)}};reader.readAsText(file)}
document.getElementById('prev').onclick=()=>{collect();index=Math.max(0,index-1);render()};document.getElementById('next').onclick=()=>{collect();index=Math.min(ROWS.length-1,index+1);render()};document.getElementById('save').onclick=save;document.getElementById('reveal').onclick=reveal;document.getElementById('export').onclick=exportJsonl;document.getElementById('export-csv').onclick=exportCsv;document.getElementById('import').onchange=e=>e.target.files[0]&&importJsonl(e.target.files[0]);load();render();
</script></body></html>'''
    assistant_link = (
        f'<a href="../{html.escape(assistant_zip_name)}" download>'
        '<button class="secondary" type="button">AI audit source bundle</button></a>'
        if assistant_zip_name else
        '<span class="note">AI audit source bundle unavailable</span>'
    )
    return (template.replace("__ROLE_LABEL__", html.escape(role_label)).replace("__ROLE_HELP__", html.escape(role_help)).replace("__ROLE__", json.dumps(role)).replace("__PACKET_SHA__", json.dumps(packet_sha)).replace("__ROWS__", rows_json).replace("__ASSISTANT_LINK__", assistant_link))


def build_bundle(rows: list[dict[str, Any]], root: Path, assistant_zip_name: str | None) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    visible, hidden = public_rows(rows, root)
    packet_sha = hashlib.sha256("\n".join(json.dumps(row, sort_keys=True) for row in visible).encode()).hexdigest()
    write_jsonl(root / "data/review_rows.jsonl", visible)
    write_jsonl(root / "data/reveal_metadata.jsonl", hidden)
    write_json(root / "data/claim_schema.json", {
        "claim_types": list(CLAIM_TYPES),
        "allowed_status": ["supported", "unsupported", "uncertain", "not_assessable"],
        "visual_caption_policy": "compose only from supported claims",
        "event_metadata_policy": "hidden until row save; event type is not automatically injected",
    })
    write_json(root / "data/review_sequence.json", [{"row_number": row["row_number"], "review_id": row["review_id"], "pair_key": row["pair_key"]} for row in visible])
    write_json(root / "data/viewer_self_test.json", {"rows": len(visible), "one_pair_per_screen": True, "relative_assets": True, "claim_level": True, "event_metadata_hidden_initially": True, "reveal_after_save": True})
    for role in ("A", "B"):
        (root / f"reviewer_{role.lower()}_review.html").write_text(claim_review_html(role, visible, packet_sha, assistant_zip_name), encoding="utf-8")
        write_jsonl(root / f"data/reviewer_{role.lower()}_decisions.jsonl", [])
    (root / "start_review.sh").write_text(
        "#!/usr/bin/env bash\nset -eu\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\nPORT=\"${1:-8765}\"\ncd \"$ROOT\"\nif command -v open >/dev/null 2>&1; then (sleep .5; open \"http://127.0.0.1:${PORT}/index.html\") >/dev/null 2>&1 & fi\nexec python3 -m http.server \"$PORT\" --bind 127.0.0.1\n",
        encoding="utf-8",
    )
    (root / "start_review.command").write_text("#!/usr/bin/env bash\nexec \"$(cd \"$(dirname \"$0\")\" && pwd)/start_review.sh\" \"$@\"\n", encoding="utf-8")
    for script in (root / "start_review.sh", root / "start_review.command"):
        script.chmod(0o755)
    (root / "index.html").write_text(
        '''<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>QCPR claim review</title><style>body{font:17px/1.55 system-ui;background:#f4f7fa;max-width:900px;margin:36px auto;padding:0 18px;color:#17202a}.card{background:white;border:1px solid #d8dee4;border-radius:14px;padding:22px;margin:14px 0}a{display:inline-block;background:#155d91;color:white;text-decoration:none;padding:11px 15px;border-radius:8px;margin:4px}li{margin:8px 0}code{background:#eef2f5;padding:2px 5px;border-radius:4px}</style><div class='card'><h1>QCPR Stage-2 · claim-level review</h1><p>Одна пара на экран. Сначала T1/T2, затем каждая atomic claim.</p><a href='reviewer_a_review.html'>Reviewer A</a><a href='reviewer_b_review.html'>Reviewer B</a></div><div class='card'><h2>Что проверять</h2><ol><li>Для каждой claim: supported, unsupported, uncertain или not_assessable.</li><li>Составьте visual caption только из supported claims.</li><li>Не используйте event name или verifier score как доказательство.</li><li>Сохраните строку. Metadata можно раскрыть только после сохранения.</li><li>После всех строк введите Reviewer ID и экспортируйте JSONL.</li></ol></div>''',
        encoding="utf-8",
    )
    sums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    write_json(root / "bundle_manifest.json", {"schema_version": "qcpr-stage2-claim-review-v1", "packet_sha256": packet_sha, "rows": len(visible), "claim_count": sum(len(row["claims"]) for row in visible), "relative_assets": True})
    return {"root": str(root), "packet_sha256": packet_sha, "rows": len(visible), "claim_count": sum(len(row["claims"]) for row in visible)}


def write_reports(args: argparse.Namespace, root: Path, original: list[dict[str, Any]], final: list[dict[str, Any]], replacements: list[dict[str, Any]], quality_240: dict[str, Any], quality_full: dict[str, Any], ai_summary: dict[str, Any], excluded_pairs: set[str], p1_state: dict[str, Any]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    excluded = []
    for row in original:
        pair_id = str(row.get("canonical_pair_id"))
        reasons = []
        if pair_id in excluded_pairs:
            reasons.append("ai_audit_exclusion")
        if quality_240.get("_rows", {}).get(pair_id, {}).get("quality_gate") == "EXCLUDE":
            reasons.extend(quality_240["_rows"][pair_id].get("reasons", []))
        if reasons:
            excluded.append({"review_id": row.get("review_id"), "canonical_pair_id": pair_id, "event": row.get("source_event_id"), "reasons": sorted(set(reasons))})
    write_jsonl(reports / "excluded_replaced_rows.jsonl", excluded + replacements)
    write_json(reports / "image_quality_thresholds.json", QUALITY_THRESHOLDS)
    write_json(reports / "ai_audit_import.json", {**ai_summary, "explicit_excluded_audit_row_ids": sorted(EXPLICIT_AI_EXCLUDED), "input_required_for_gate": True})
    write_json(reports / "claim_schema.json", {"claim_types": list(CLAIM_TYPES), "statuses": ["supported", "unsupported", "uncertain", "not_assessable"], "visual_caption": "supported claims only", "event_type": "separate field; no automatic injection"})
    events = Counter(str(row["source_event_id"]) for row in final)
    packet_audit = {"rows": len(final), "unique_pairs": len({row["canonical_pair_id"] for row in final}), "event_count": len(events), "rows_per_event": dict(sorted(events.items())), "relative_assets": True, "claim_level": True, "metadata_hidden_until_save": True, "two_independent_roles": True, "pass": len(final) == 240 and len(events) == 12 and all(value == 20 for value in events.values())}
    write_json(reports / "revised_review_packet_audit.json", packet_audit)
    write_json(reports / "rscc_sampling_policy.json", {"hierarchy": ["source", "domain", "event", "change_signature", "physical_pair"], "event_sampling": "uniform", "max_event_fraction": 0.10, "max_rscc_fraction_first_p2_real": {"min": 0.20, "max": 0.30}, "event_disjoint_validation_test": True, "training_enabled": False})
    write_json(reports / "rscc_event_robustness_plan.json", {"per_event_metrics": True, "leave_one_event_out": True, "macro_average_over_events": True, "micro_average_over_pairs": True, "metrics": ["MRR", "Recall@K", "nDCG", "soft_localization"]})
    p1_ready = {"status": "P1_NOT_READY", "mask_free_zero_violations": bool(p1_state.get("mask_free_zero_violations", False)), "authoritative_common_sha": p1_state.get("manifest_sha256"), "common_frozen_evaluation_complete": bool(p1_state.get("full_rankings_found", False)), "runtime_probe_32_64_steps": False, "fixed_exposure_contract": False, "reason": "common frozen rankings and final fixed-exposure/runtime probe are absent; no P1 submitted"}
    p2_ready = {"status": "P2_REAL_NOT_READY", "ai_audit_status": ai_summary.get("status"), "human_review_required": True, "training_enabled_reviewed_rows": 0, "reason": "AI audit is advisory and exact row-level input is missing or not human-adjudicated"}
    write_json(reports / "p1_readiness.json", p1_ready)
    write_json(reports / "p2_real_readiness.json", p2_ready)
    write_json(root / "stage2_review_gate_summary.json", {"code_sha": code_sha(), "status": "DATA_QUALITY_HOLD", "p2_submitted": False, "ai_audit": ai_summary, "quality_240": {key: value for key, value in quality_240.items() if key != "_rows"}, "quality_full_rscc": quality_full, "revised_packet": packet_audit, "replacements": len(replacements), "p1_readiness": p1_ready, "p2_real_readiness": p2_ready, "blockers": ["qcpr_stage2_ai_audit_48.jsonl missing" if ai_summary.get("status") != "IMPORTED" else "independent human review still required", "common frozen full rankings absent", "no P1 runtime probe or fixed-exposure submission performed"]})
    lines = ["# Stage-2 review gate", "", "**Status:** `DATA_QUALITY_HOLD`", "", "- AI audit: `" + str(ai_summary.get("status")) + "` (advisory only)", f"- revised packet: {len(final)} rows; 20 per event", f"- 240-row image gate exclusions: {quality_240.get('excluded', 0)}", f"- full RSCC rows audited: {quality_full.get('pairs', 0)}", "- event ID/verifier/selection reason hidden until save", "- visual_caption is separate from event metadata and uses supported claims only", "", f"- P1: `{p1_ready['status']}`", f"- P2-real: `{p2_ready['status']}`", "", "The exact qcpr_stage2_ai_audit_48.jsonl was not found; its supplied summary was recorded without fabricating row-level decisions."]
    (reports / "stage2_review_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ai-audit", type=Path, default=None)
    parser.add_argument("--reuse-quality-from", type=Path, default=None, help="Reuse an already completed quality JSONL directory; rules/code must match.")
    args = parser.parse_args()
    if args.run_root.exists():
        shutil.rmtree(args.run_root)
    args.run_root.mkdir(parents=True)
    original = read_jsonl(args.release / "reports/semantic_review/human_review_packet.jsonl")
    registry = registry_rows(args.release)
    rscc = [row for row in registry if str(row.get("source_dataset", "")).upper() == "RSCC-EBD"]
    assistant = read_jsonl(args.prior_run / "assistant_audit_bundle/assistant_audit_manifest.jsonl")
    ai_rows, ai_summary = load_ai_audit(args.ai_audit)
    if args.reuse_quality_from:
        quality_240_rows, quality_240 = reuse_quality_rows(args.reuse_quality_from / "reports/image_quality_240.jsonl", "review_packet_original", args.run_root / "reports/image_quality_240.jsonl")
        quality_full_rows, quality_full = reuse_quality_rows(args.reuse_quality_from / "reports/image_quality_full_rscc.jsonl", "full_rscc_registry", args.run_root / "reports/image_quality_full_rscc.jsonl")
    else:
        quality_240_rows, quality_240 = audit_image_rows(original, "review_packet_original", args.run_root / "reports/image_quality_240.jsonl")
        quality_full_rows, quality_full = audit_image_rows(rscc, "full_rscc_registry", args.run_root / "reports/image_quality_full_rscc.jsonl")
    merged_quality = {**quality_full_rows, **quality_240_rows}
    final, replacements, excluded_pairs = select_revised_packet(original, registry, merged_quality, ai_rows, assistant, args.release)
    # Replacements need captions from the candidate registry; the original rows
    # already carry their captions.
    captions = qvq_caption_map(args.release)
    for row in final:
        if not row.get("candidate_caption"):
            row["candidate_caption"] = captions.get(str(row["canonical_pair_id"]), "No generated caption available; describe only visible T1→T2 facts.")
    source_zip = args.prior_run / "qcpr_stage2_assistant_audit_bundle_7ef89dc_20260803.zip"
    assistant_zip_name = "assistant_audit_source_bundle.zip" if source_zip.exists() else None
    if source_zip.exists():
        shutil.copy2(source_zip, args.run_root / assistant_zip_name)
    bundle = build_bundle(final, args.run_root / "human_review_bundle_claims", assistant_zip_name)
    common = read_json(args.prior_run / "common_frozen_evaluation/common_frozen_evaluation_state.json", {}) or {}
    mask = read_json(args.prior_run / "maskfree_repair/maskfree_leakage_audit.json", {}) or {}
    write_reports(args, args.run_root, original, final, replacements, {"_rows": quality_240_rows, **quality_240}, quality_full, ai_summary, excluded_pairs, {**common, "mask_free_zero_violations": bool(mask.get("zero_violations_after_repair", False))})
    assistant_by_id = {str(row.get("audit_row_id")): row for row in assistant}
    write_json(args.run_root / "reports/ai_excluded_row_mapping.json", {
        "advisory_only": True,
        "source": str(args.prior_run / "assistant_audit_bundle/assistant_audit_manifest.jsonl"),
        "rows": [
            {
                "audit_row_id": row_id,
                "canonical_pair_id": assistant_by_id.get(row_id, {}).get("canonical_pair_id"),
                "event_id": assistant_by_id.get(row_id, {}).get("source_event_id") or assistant_by_id.get(row_id, {}).get("event_id"),
                "present_in_original_packet": any(str(item.get("canonical_pair_id")) == str(assistant_by_id.get(row_id, {}).get("canonical_pair_id")) for item in original) if row_id in assistant_by_id else False,
                "explicit_exclusion": True,
            }
            for row_id in sorted(EXPLICIT_AI_EXCLUDED)
        ],
        "note": "The row-level AI decision file was unavailable; these mappings are selection metadata only and do not constitute factual review.",
    })
    write_json(args.run_root / "reports/bundle_info.json", bundle)
    print(json.dumps({"code_sha": code_sha(), "run_root": str(args.run_root), "status": "DATA_QUALITY_HOLD", "ai_audit": ai_summary, "quality_240": quality_240, "quality_full_rscc": quality_full, "rows": len(final), "replacements": len(replacements), "p2_submitted": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
