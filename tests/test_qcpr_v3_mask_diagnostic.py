from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from land_change_detection.models.qcpr_v3_mask_diagnostic import (
    append_jsonl,
    exact_s2looking_query_indices,
    fixed_probe_subset,
    query_swap_metrics,
    resolve_mask_objective,
    swapped_direction_indices,
)


def _row(pair_id: str, *, dataset: str = "s2looking", mode: str = "query_specific", retrieval: bool = False):
    return {
        "pair_id": pair_id, "dataset_name": dataset, "seg_supervision_mode": mode,
        "retrieval_supervision": retrieval,
    }


def test_exact_s2looking_filter_rejects_nominal_presence_only() -> None:
    dataset = SimpleNamespace(samples=[
        _row("s2looking:train:1:appeared"),
        _row("levir:train:1", dataset="levir_mci", mode="binary_generic", retrieval=True),
        _row("s2looking:train:2:appeared", retrieval=True),
    ])
    assert exact_s2looking_query_indices(dataset) == [0]


def test_fixed_probe_skips_all_empty_base_pairs_and_keeps_direction_pair(tmp_path: Path) -> None:
    def mask(name: str, nonempty: bool) -> str:
        path = tmp_path / name
        image = Image.new("L", (4, 4), 0)
        if nonempty:
            image.putpixel((1, 1), 255)
        image.save(path)
        return str(path)
    rows = []
    for base, areas in (("1", (False, False)), ("2", (True, False))):
        for change, nonempty in zip(("appeared", "disappeared"), areas, strict=True):
            row = _row(f"s2looking:train:{base}:{change}")
            row["mask_path"] = mask(f"{base}-{change}.png", nonempty)
            row["source_metadata"] = {"change_type": change}
            rows.append(row)
    dataset = SimpleNamespace(samples=rows)
    subset = fixed_probe_subset(dataset, count=2)
    assert list(subset.indices) == [2, 3]


def test_explicit_probe_ids_are_selected_and_history_append_is_independent(tmp_path: Path) -> None:
    rows = [_row("s2looking:train:9:appeared"), _row("s2looking:train:7:disappeared")]
    subset = fixed_probe_subset(SimpleNamespace(samples=rows), count=2, pair_ids=(rows[0]["pair_id"], rows[1]["pair_id"]))
    assert list(subset.indices) == [0, 1]
    path = tmp_path / "history.jsonl"
    append_jsonl(path, {"step": 0})
    assert path.read_text().strip() == '{"step": 0}'


def test_pair_level_probe_allows_one_row_with_both_directions(tmp_path: Path) -> None:
    paths = []
    for direction in ("appeared", "disappeared"):
        path = tmp_path / f"{direction}.png"
        image = Image.new("L", (4, 4), 0); image.putpixel((1, 1), 255); image.save(path)
        paths.append(str(path))
    row = _row("s2looking:train:1506")
    row["directional_targets"] = [
        {"direction": direction, "mask_path": path}
        for direction, path in zip(("appeared", "disappeared"), paths, strict=True)
    ]
    subset = fixed_probe_subset(SimpleNamespace(samples=[row]), count=1, pair_ids=(row["pair_id"],))
    assert list(subset.indices) == [0]


def test_objective_ablation_contract_is_exact() -> None:
    assert resolve_mask_objective("A").tversky_weight == 0.0
    with pytest.raises(ValueError, match="unknown mask objective"):
        resolve_mask_objective("B")


def test_query_swap_gap_uses_same_target_without_smoothing() -> None:
    target = torch.zeros(1, 2, 2); target[0, 0, 0] = 1
    correct = torch.tensor([[[8.0, -8.0], [-8.0, -8.0]]])
    swapped = -correct
    metrics = query_swap_metrics(correct, swapped, target)
    assert metrics["correct_query_iou"] == 1.0
    assert metrics["swapped_query_iou"] == 0.0
    assert metrics["query_swap_gap"] == 1.0
    assert metrics["soft_query_swap_iou_gap"] > 0.9
