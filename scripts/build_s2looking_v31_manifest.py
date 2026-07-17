#!/usr/bin/env python3
"""Build immutable pair-level S2Looking records for directional grounding."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def _mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    labels, count = ndimage.label(mask)
    output: list[tuple[int, int, int, int, int]] = []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        area = int((labels[box] == label).sum())
        if area >= 16:
            output.append((area, box[1].start, box[0].start, box[1].stop, box[0].stop))
    return output


def _location(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return ""
    vertical = "top" if ys.mean() < mask.shape[0] / 3 else "bottom" if ys.mean() > 2 * mask.shape[0] / 3 else "middle"
    horizontal = "left" if xs.mean() < mask.shape[1] / 3 else "right" if xs.mean() > 2 * mask.shape[1] / 3 else "center"
    return f"{vertical} {horizontal}"


def _caption(direction: str, count: int, location: str) -> str:
    noun = "one building" if count == 1 else f"{count} buildings" if 1 < count <= 5 else "several buildings"
    action = "appeared" if direction == "appeared" else "disappeared"
    return f"{noun} {action}" + (f" near the {location}" if location else "")


def _image_shift(first: str, second: str) -> float:
    def read(path: str) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return float(np.abs(read(first) - read(second)).mean())


def s2looking_chronological_paths(image1_path: str, image2_path: str) -> tuple[str, str]:
    """Return (before, after) for the official S2Looking file convention.

    The official paper's Figure 2 identifies Image1 as the newer acquisition
    and Image2 as the older acquisition; Label1 is newly built and Label2 is
    demolished. Therefore chronological inference is Image2 -> Image1.
    """
    return image2_path, image1_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], str(row["source_metadata"]["base_pair_id"]))].append(row)
    prepared: list[dict] = []
    started = time.monotonic()
    ordered_groups = sorted(grouped.items())
    for group_index, ((split, base), members) in enumerate(ordered_groups, start=1):
        by_direction = {row["source_metadata"]["change_type"]: row for row in members}
        if set(by_direction) != {"appeared", "disappeared"}:
            raise RuntimeError(f"incomplete directional pair {split}:{base}: {sorted(by_direction)}")
        targets = []
        qualities = []
        directional_masks: list[np.ndarray] = []
        for direction in ("appeared", "disappeared"):
            source = by_direction[direction]
            mask = _mask(source["mask_path"])
            directional_masks.append(mask)
            components = _components(mask)
            area = float(mask.mean())
            cells = area * 32 * 32
            border = np.zeros_like(mask); border[[0, -1], :] = True; border[:, [0, -1]] = True
            edge_fraction = float((mask & border).sum() / max(mask.sum(), 1))
            caption = _caption(direction, len(components), _location(mask))
            ys, xs = np.nonzero(mask)
            bbox = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            targets.append({
                "direction": direction, "caption": caption, "mask_path": source["mask_path"],
                "direction_caption": "new buildings appeared" if direction == "appeared" else "buildings were demolished",
                "foreground_area": area, "canonical_cell_equivalents": cells,
                "component_count": len(components), "edge_fraction": edge_fraction,
                "raw_geometry": {"height": int(mask.shape[0]), "width": int(mask.shape[1]), "bbox_xyxy": bbox},
                "caption_provenance": "deterministic_verified_mask_attributes",
                "caption_confidence": 1.0, "audit_status": "derived_not_human_reviewed",
            })
            if not mask.any() or cells < 4 or edge_fraction > 0.5:
                qualities.append("stress")
            elif cells < 8 or edge_fraction > 0.2:
                qualities.append("hard")
            else:
                qualities.append("core")
        before_path, after_path = s2looking_chronological_paths(
            members[0]["t1_path"], members[0]["t2_path"],
        )
        shift = _image_shift(before_path, after_path)
        overlap_union = np.logical_or(directional_masks[0], directional_masks[1]).sum()
        directional_overlap_iou = float(
            np.logical_and(directional_masks[0], directional_masks[1]).sum() / max(overlap_union, 1)
        )
        tier = (
            "stress" if qualities.count("stress") == 2 or directional_overlap_iou > 0.30 else
            "hard" if "stress" in qualities or "hard" in qualities or shift > 0.18 or directional_overlap_iou > 0.10 else
            "core"
        )
        prepared.append({
            "schema_version": members[0]["schema_version"], "dataset_name": "s2looking",
            "split": split, "pair_id": f"s2looking:{split}:{base}", "original_id": base,
            "t1_path": before_path, "t2_path": after_path,
            "captions": [target["caption"] for target in targets],
            "normalized_caption_groups": [target["caption"].casefold() for target in targets],
            "caption_source": "deterministic_verified_mask_attributes",
            "directional_targets": targets, "quality_tier": tier, "radiometric_shift": shift,
            "directional_overlap_iou": directional_overlap_iou,
            "retrieval_supervision": False, "seg_supervision_mode": "query_specific",
            "image_height": 1024, "image_width": 1024, "sensor": "side_looking_vhr",
            "time_order": ["Image2:before", "Image1:after"],
            "source_metadata": {"base_pair_id": base, "paired_directional_record": True,
                                "retrieval_supervision": False, "seg_supervision_mode": "query_specific",
                                "official_source_order": "Image2_to_Image1",
                                "official_label_mapping": {"label1": "appeared", "label2": "disappeared"}},
        })
        if group_index == 1 or group_index % 100 == 0 or group_index == len(ordered_groups):
            elapsed = time.monotonic() - started
            fraction = group_index / len(ordered_groups)
            (output / "progress.json").write_text(json.dumps({
                "status": "RUNNING" if group_index < len(ordered_groups) else "FINALIZING",
                "completed_pairs": group_index, "total_pairs": len(ordered_groups),
                "completed_fraction": fraction, "elapsed_seconds": elapsed,
                "eta_seconds": max(elapsed / fraction - elapsed, 0.0),
            }, indent=2, sort_keys=True) + "\n")
    manifest = output / "s2looking_pair_level.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared)
    manifest.write_text(payload)
    for split, filename in (("train", "natural_train_manifest.jsonl"), ("val", "natural_validation_manifest.jsonl")):
        split_payload = "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in prepared if row["split"] == split and row["quality_tier"] != "stress"
        )
        (output / filename).write_text(split_payload)
    stress_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in prepared if row["quality_tier"] == "stress"
    )
    (output / "stress_evaluation_manifest.jsonl").write_text(stress_payload)
    counts = {tier: sum(row["quality_tier"] == tier for row in prepared) for tier in ("core", "hard", "stress")}
    overlaps = np.asarray([row["directional_overlap_iou"] for row in prepared], dtype=np.float64)
    audit = {"input": str(Path(args.input).resolve()), "pair_count": len(prepared), "tier_counts": counts,
             "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
             "directional_mapping": "Image2(before)->Image1(after); appeared=label1; disappeared=label2",
             "directional_overlap_iou_quantiles": {
                 str(q): float(np.quantile(overlaps, q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
             },
             "directional_overlap_policy": {"core_max": 0.10, "stress_above": 0.30}}
    (output / "s2looking_pair_level_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output / "progress.json").write_text(json.dumps({
        "status": "COMPLETED", "completed_pairs": len(prepared), "total_pairs": len(prepared),
        "completed_fraction": 1.0, "elapsed_seconds": time.monotonic() - started, "eta_seconds": 0.0,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
