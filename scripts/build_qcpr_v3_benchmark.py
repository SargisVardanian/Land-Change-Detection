from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stratum(audit: dict) -> tuple:
    area = audit.get("foreground_area")
    area_bin = "unknown" if area is None else "empty" if area == 0 else "small" if area < 0.01 else "medium" if area < 0.1 else "large"
    return (
        audit["dataset"], audit["change_status"], audit["temporal_direction"],
        bool(audit["object_family_audit_tags"]), bool(audit["location_audit_tags"]),
        bool(audit["count_audit_tags"]), "replacement" in audit["relation_audit_tags"], area_bin,
    )


def build_benchmark(derived_dir: Path, output_dir: Path, *, max_queries: int = 500, candidates_per_query: int = 10) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _read_jsonl(derived_dir / "natural_validation_manifest.jsonl")
    audits = _read_jsonl(derived_dir / "caption_quality_audit.jsonl")
    validation_ids = {str(row["pair_id"]) for row in validation}
    audits = [audit for audit in audits if audit["pair_id"] in validation_ids]
    by_stratum: defaultdict[tuple, list[dict]] = defaultdict(list)
    for audit in audits:
        by_stratum[_stratum(audit)].append(audit)
    selected: list[dict] = []
    ordered_strata = sorted(by_stratum, key=repr)
    while len(selected) < max_queries:
        changed = False
        for key in ordered_strata:
            values = by_stratum[key]
            index = sum(1 for item in selected if tuple(item["stratum"]) == key)
            if index < len(values) and len(selected) < max_queries:
                audit = sorted(values, key=lambda item: item["caption_id"])[index]
                selected.append(audit | {"stratum": list(key)})
                changed = True
        if not changed:
            break
    rows_by_pair = {str(row["pair_id"]): row for row in validation}
    candidate_ids = sorted(rows_by_pair)
    review_rows: list[dict] = []
    for query in selected:
        digest = int(hashlib.sha256(query["caption_id"].encode()).hexdigest()[:16], 16)
        candidates = [query["pair_id"]]
        cursor = digest % max(len(candidate_ids), 1)
        while len(candidates) < min(candidates_per_query, len(candidate_ids)):
            candidate = candidate_ids[cursor % len(candidate_ids)]
            cursor += 7919
            if candidate not in candidates:
                candidates.append(candidate)
        for rank, candidate in enumerate(candidates):
            review_rows.append({
                "query_id": query["caption_id"], "query": query["caption"],
                "candidate_pair": candidate, "candidate_seed_rank": rank,
                "relevance": "unreviewed", "temporal_correctness": "unreviewed",
                "spatial_correctness": "unreviewed", "quantity_correctness": "unreviewed",
                "relation_correctness": "unreviewed",
                "query_specific_mask_available": bool(rows_by_pair[candidate].get("query_mask_path") or rows_by_pair[candidate].get("mask_path")),
                "reviewer_confidence": None, "annotator_id": None, "adjudication_status": "unreviewed",
                "stratum": query["stratum"],
            })
    (output_dir / "human_review_schema.json").write_text(json.dumps({
        "schema_version": "qcpr-v3-human-review-v1",
        "labels": ["relevance", "temporal_correctness", "spatial_correctness", "quantity_correctness", "relation_correctness"],
        "allowed_status": ["yes", "no", "uncertain", "not_applicable", "unreviewed"],
        "required_reviewers": 2,
    }, indent=2), encoding="utf-8")
    (output_dir / "review_items.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in review_rows), encoding="utf-8")
    (output_dir / "query_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected), encoding="utf-8")

    preview = selected[: min(24, len(selected))]
    cell_w, cell_h = 768, 300
    sheet = Image.new("RGB", (cell_w * 2, cell_h * ((len(preview) + 1) // 2)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, query in enumerate(preview):
        row = rows_by_pair[query["pair_id"]]
        x0, y0 = (index % 2) * cell_w, (index // 2) * cell_h
        for image_index, key in enumerate(("t1_path", "t2_path")):
            path = Path(row[key])
            if path.exists():
                with Image.open(path) as image:
                    tile = image.convert("RGB").resize((256, 256))
                sheet.paste(tile, (x0 + image_index * 260, y0))
        draw.text((x0 + 525, y0 + 5), query["caption"][:38], fill="black")
        draw.text((x0 + 525, y0 + 70), "UNREVIEWED", fill="red")
        draw.text((x0 + 525, y0 + 100), query["pair_id"][:34], fill="black")
    sheet.save(output_dir / "contact_sheet_unreviewed.png")
    report = {
        "schema_version": "qcpr-v3-benchmark-package-v1",
        "query_count": len(selected), "review_item_count": len(review_rows),
        "all_labels_unreviewed": all(row["relevance"] == "unreviewed" for row in review_rows),
        "stratum_count": len({_stratum(query) for query in selected}),
    }
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--candidates-per-query", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(build_benchmark(args.derived_dir, args.output_dir, max_queries=args.max_queries, candidates_per_query=args.candidates_per_query), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
