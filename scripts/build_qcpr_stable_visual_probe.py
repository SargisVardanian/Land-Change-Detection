#!/usr/bin/env python3
"""Create independently probed stable-scene candidates for no-change pairs.

This is intentionally a candidate generator, not a human-verification step.
T1 and T2 are probed separately with a frozen image-text encoder.  Only the
intersection of independently supported anchors is carried into the stable
query.  Masks are not read and cannot influence the query text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


ANCHORS = (
    "roads",
    "buildings",
    "fields",
    "trees",
    "forest",
    "water",
    "bare ground",
    "houses",
    "urban area",
    "farmland",
    "industrial structures",
    "parking areas",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-registry", type=Path, required=True)
    parser.add_argument("--pair-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def _stable_text(anchors: list[str]) -> str:
    names = anchors[:3]
    if len(names) == 2:
        return f"The same {names[0]} and {names[1]} remain visible in both observations."
    return f"The same {', '.join(names[:-1])} and {names[-1]} remain visible in both observations."


def _image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _probe_scores(model: Any, processor: Any, images: list[Image.Image], text_embeds: torch.Tensor) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    outputs = model.vision_model(**inputs)
    image_embeds = outputs.pooler_output
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds @ text_embeds.T


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    captions = read_jsonl(args.caption_registry)
    pairs = {str(row.get("canonical_pair_id")): row for row in read_jsonl(args.pair_registry)}
    stable_pair_ids = sorted({
        str(row.get("canonical_pair_id"))
        for row in captions
        if str(row.get("query_scope") or "") == "generic_no_change"
        and str(row.get("canonical_pair_id") or "") in pairs
    })
    model = AutoModel.from_pretrained(args.model).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    text_inputs = processor(
        text=[f"a satellite image containing {anchor}" for anchor in ANCHORS],
        padding="max_length",
        return_tensors="pt",
    )
    with torch.inference_mode():
        text_outputs = model.text_model(**text_inputs)
        text_embeds = text_outputs.pooler_output
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    rows: list[dict[str, Any]] = []
    pending: list[tuple[str, str, str]] = []
    for pair_id in stable_pair_ids:
        pair = pairs[pair_id]
        frames = pair.get("frames") or []
        by_time = {str(frame.get("timestamp")): str(frame.get("path") or "") for frame in frames}
        t1 = by_time.get("t1") or str(pair.get("t1_path") or "")
        t2 = by_time.get("t2") or str(pair.get("t2_path") or "")
        pending.append((pair_id, t1, t2))

    valid = [row for row in pending if Path(row[1]).exists() and Path(row[2]).exists()]
    for begin in range(0, len(valid), max(1, args.batch_size)):
        batch = valid[begin : begin + max(1, args.batch_size)]
        images = [_image(path) for row in batch for path in (row[1], row[2])]
        with torch.inference_mode():
            score_tensor = _probe_scores(model, processor, images, text_embeds)
        for index, (pair_id, t1_path, t2_path) in enumerate(batch):
            t1_scores = [float(value) for value in score_tensor[2 * index].tolist()]
            t2_scores = [float(value) for value in score_tensor[2 * index + 1].tolist()]
            t1_order = sorted(range(len(ANCHORS)), key=lambda item: (-t1_scores[item], item))
            t2_order = sorted(range(len(ANCHORS)), key=lambda item: (-t2_scores[item], item))
            t1_claims = [
                {"anchor": ANCHORS[item], "score": round(t1_scores[item], 6), "supported": t1_scores[item] >= args.min_score}
                for item in t1_order[: max(1, args.top_k)]
            ]
            t2_claims = [
                {"anchor": ANCHORS[item], "score": round(t2_scores[item], 6), "supported": t2_scores[item] >= args.min_score}
                for item in t2_order[: max(1, args.top_k)]
            ]
            t1_map = {claim["anchor"]: claim for claim in t1_claims if claim["supported"]}
            t2_map = {claim["anchor"]: claim for claim in t2_claims if claim["supported"]}
            common = sorted(
                set(t1_map) & set(t2_map),
                key=lambda anchor: (-(t1_map[anchor]["score"] + t2_map[anchor]["score"]), anchor),
            )
            common = common[: max(0, args.top_k)]
            min_score = min(
                [min(t1_map[anchor]["score"], t2_map[anchor]["score"]) for anchor in common],
                default=0.0,
            )
            mean_score = sum(
                (t1_map[anchor]["score"] + t2_map[anchor]["score"]) / 2.0 for anchor in common
            ) / max(len(common), 1)
            identifiability = min(1.0, 0.45 * min(len(common), 3) / 3.0 + 0.55 * max(0.0, min(1.0, (mean_score - args.min_score) / 0.12)))
            rows.append({
                "candidate_id": "stable_probe:" + hashlib.sha256(pair_id.encode()).hexdigest()[:20],
                "canonical_pair_id": pair_id,
                "t1_path": t1_path,
                "t2_path": t2_path,
                "independent_t1_claims": t1_claims,
                "independent_t2_claims": t2_claims,
                "common_atomic_anchors": common,
                "query_text": _stable_text(common) if len(common) >= 2 else "",
                "anchor_count": len(common),
                "min_common_anchor_score": round(min_score, 6),
                "mean_common_anchor_score": round(mean_score, 6),
                "identifiability_score": round(identifiability, 6),
                "visual_evidence": {
                    "method": "siglip2_zero_shot_anchor_probe",
                    "model": str(args.model),
                    "masks_used": False,
                    "independent_frame_passes": 2,
                    "threshold": args.min_score,
                    "candidate_not_human_verified": True,
                },
                "verification": "visual_probe_candidate",
                "training_enabled": False,
                "review_decision": None,
            })
        if begin and begin % (max(1, args.batch_size) * 20) == 0:
            print(json.dumps({"processed": begin, "total": len(valid)}), flush=True)

    missing = [row for row in pending if row not in valid]
    for pair_id, t1_path, t2_path in missing:
        rows.append({
            "candidate_id": "stable_probe:" + hashlib.sha256(pair_id.encode()).hexdigest()[:20],
            "canonical_pair_id": pair_id,
            "t1_path": t1_path,
            "t2_path": t2_path,
            "independent_t1_claims": [],
            "independent_t2_claims": [],
            "common_atomic_anchors": [],
            "query_text": "",
            "anchor_count": 0,
            "min_common_anchor_score": 0.0,
            "mean_common_anchor_score": 0.0,
            "identifiability_score": 0.0,
            "visual_evidence": {"method": "not_available", "masks_used": False},
            "verification": "visual_probe_unavailable",
            "training_enabled": False,
            "review_decision": None,
        })
    rows.sort(key=lambda row: row["candidate_id"])
    write_jsonl(args.output, rows)
    summary = {
        "schema_version": "qcpr-stable-visual-probe-v1",
        "pair_count": len(stable_pair_ids),
        "probe_rows": len(rows),
        "visual_probe_rows": len(valid),
        "missing_visual_assets": len(missing),
        "rows_with_two_or_more_common_anchors": sum(row["anchor_count"] >= 2 for row in rows),
        "rows_with_unique_candidate_not_yet_known": 0,
        "training_enabled": False,
        "human_review_required": True,
        "model": str(args.model),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

