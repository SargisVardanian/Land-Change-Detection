from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict, key: str) -> float | None:
    branch = payload.get("retrieval_branch_metrics", {}).get("fused", {})
    corpus = payload.get("corpus_metrics", {})
    value = branch.get(key, corpus.get(key))
    return None if value is None else float(value)


def compare(
    v1_dir: Path,
    v2_dir: Path,
    output_dir: Path,
    v1_localization_dir: Path | None = None,
    v2_localization_dir: Path | None = None,
) -> dict:
    v1 = _load(v1_dir / "evaluation_summary.json")
    v2 = _load(v2_dir / "evaluation_summary.json")
    v1_localization = _load((v1_localization_dir or v1_dir) / "evaluation_summary.json")
    v2_localization = _load((v2_localization_dir or v2_dir) / "evaluation_summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        "semantic_recall@1", "semantic_recall@5", "semantic_recall@10", "semantic_nDCG@10",
        "object_match_R@5", "direction_match_R@5", "location_match_R@5", "count_match_R@5", "relation_match_R@5",
        "best_positive_minus_best_negative_mean", "hard_negative_score_mean", "positive_score_mean", "ECE", "Brier",
        "mask_target_kind_query_specific_precision", "mask_target_kind_query_specific_recall",
    ]
    rows = []
    for key in keys:
        use_localization = key.startswith("mask_")
        before = _metric(v1_localization if use_localization else v1, key)
        after = _metric(v2_localization if use_localization else v2, key)
        rows.append({"metric": key, "qcpr_v1": before, "qcpr_v2": after, "delta": None if before is None or after is None else after - before})
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "qcpr_v1", "qcpr_v2", "delta"))
        writer.writeheader()
        writer.writerows(rows)

    v2_branches = v2.get("retrieval_branch_metrics", {})
    fused_r1 = v2_branches.get("fused", {}).get("semantic_recall@1")
    global_r1 = v2_branches.get("global", {}).get("semantic_recall@1")
    fused_outperforms_global = fused_r1 is not None and global_r1 is not None and float(fused_r1) > float(global_r1)
    v1_margin = _metric(v1, "best_positive_minus_best_negative_mean")
    v2_margin = _metric(v2, "best_positive_minus_best_negative_mean")
    local_negative_p90 = v2_branches.get("local", {}).get("hard_negative_score_p90")
    v1_mask_precision = _metric(v1_localization, "mask_target_kind_query_specific_precision")
    v2_mask_precision = _metric(v2_localization, "mask_target_kind_query_specific_precision")
    v1_mask_recall = _metric(v1_localization, "mask_target_kind_query_specific_recall")
    v2_mask_recall = _metric(v2_localization, "mask_target_kind_query_specific_recall")
    v1_direction, v2_direction = _metric(v1, "direction_match_R@5"), _metric(v2, "direction_match_R@5")
    v1_location, v2_location = _metric(v1, "location_match_R@5"), _metric(v2, "location_match_R@5")
    gates = {
        "semantic_R@1_drop_within_0.02": (_metric(v2, "semantic_recall@1") or 0.0) >= (_metric(v1, "semantic_recall@1") or 0.0) - 0.02,
        "semantic_R@5_drop_within_0.02": (_metric(v2, "semantic_recall@5") or 0.0) >= (_metric(v1, "semantic_recall@5") or 0.0) - 0.02,
        "structured_metrics_populated": all(_metric(v2, key) is not None for key in ("object_match_R@5", "direction_match_R@5", "location_match_R@5", "count_match_R@5")),
        "local_margin_improved": v1_margin is not None and v2_margin is not None and v2_margin > v1_margin,
        "local_negatives_not_saturated": local_negative_p90 is not None and float(local_negative_p90) < 0.95,
        "fused_vs_global_documented": fused_r1 is not None and global_r1 is not None,
        "direction_match_not_lower": v1_direction is not None and v2_direction is not None and v2_direction >= v1_direction,
        "location_match_not_lower": v1_location is not None and v2_location is not None and v2_location >= v1_location,
        "query_mask_precision_improved": v1_mask_precision is not None and v2_mask_precision is not None and v2_mask_precision > v1_mask_precision,
        "query_mask_recall_preserved": v1_mask_recall is not None and v2_mask_recall is not None and v2_mask_recall >= v1_mask_recall - 0.05,
    }
    report = {
        "v1_dir": str(v1_dir), "v2_dir": str(v2_dir), "rows": rows, "gates": gates,
        "v1_localization_dir": str(v1_localization_dir or v1_dir),
        "v2_localization_dir": str(v2_localization_dir or v2_dir),
        "fused_outperforms_global": fused_outperforms_global,
        "fused_vs_global_note": (
            "Fused semantic R@1 exceeds global-only."
            if fused_outperforms_global
            else "Fused did not exceed global-only semantic R@1; retain branch diagnostics and inspect calibration before changing weights."
        ),
        "passed": all(gates.values()),
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    images = []
    for label, directory in (("QCPR v1", v1_dir), ("QCPR v2", v2_dir)):
        for path in sorted((directory / "visuals").glob("*.png"))[:4]:
            with Image.open(path) as image:
                tile = image.convert("RGB").copy()
            tile.thumbnail((600, 400))
            images.append((label, tile))
    if images:
        width = max(image.width for _, image in images)
        height = max(image.height for _, image in images) + 28
        sheet = Image.new("RGB", (2 * width, ((len(images) + 1) // 2) * height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(images):
            x, y = (index % 2) * width, (index // 2) * height
            sheet.paste(image, (x, y + 28))
            draw.text((x + 8, y + 6), label, fill="black")
        sheet.save(output_dir / "comparison_contact_sheet.png")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v1-localization-dir", type=Path, default=None)
    parser.add_argument("--v2-localization-dir", type=Path, default=None)
    args = parser.parse_args()
    report = compare(args.v1_dir, args.v2_dir, args.output_dir, args.v1_localization_dir, args.v2_localization_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
