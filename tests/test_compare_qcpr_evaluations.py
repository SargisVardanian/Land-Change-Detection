from __future__ import annotations

import csv
import json

from PIL import Image

from compare_qcpr_evaluations import compare


def _evaluation(root, r1, r5, structured=True, margin=0.1, precision=0.4, recall=0.5):
    root.mkdir()
    metrics = {"semantic_recall@1": r1, "semantic_recall@5": r5, "semantic_recall@10": 0.9, "semantic_nDCG@10": 0.8}
    if structured:
        metrics.update({name: 0.5 for name in ("object_match_R@5", "direction_match_R@5", "location_match_R@5", "count_match_R@5", "relation_match_R@5")})
    metrics["best_positive_minus_best_negative_mean"] = margin
    (root / "evaluation_summary.json").write_text(json.dumps({"retrieval_branch_metrics": {
        "fused": metrics,
        "global": {"semantic_recall@1": r1 - 0.01},
        "local": {"hard_negative_score_p90": 0.7},
    }, "corpus_metrics": {
        "mask_target_kind_query_specific_precision": precision,
        "mask_target_kind_query_specific_recall": recall,
    }}))
    visuals = root / "visuals"
    visuals.mkdir()
    Image.new("RGB", (32, 32), "red").save(visuals / "one.png")


def test_comparison_writes_acceptance_csv_json_and_contact_sheet(tmp_path):
    v1, v2, output = tmp_path / "v1", tmp_path / "v2", tmp_path / "comparison"
    _evaluation(v1, 0.80, 0.90, margin=0.1, precision=0.4, recall=0.5)
    _evaluation(v2, 0.79, 0.89, margin=0.2, precision=0.5, recall=0.48)
    report = compare(v1, v2, output)
    assert report["passed"] is True
    assert (output / "comparison.json").exists()
    assert (output / "comparison_contact_sheet.png").exists()
    with (output / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "semantic_recall@1" for row in rows)
