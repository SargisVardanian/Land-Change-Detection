from __future__ import annotations

import json
from pathlib import Path

import torch

from train_qcpr_stage2_b1_control import (
    grouped_rows,
    make_schedule,
    make_supervision,
    stable_sha,
)


def _write_manifest(path: Path, count: int = 128) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "canonical_pair_id": f"pair:{index}",
                "caption_id": f"caption:{index}",
                "caption": f"building change {index}",
                "dataset_name": "RSCC-EBD" if index % 2 else "levir_mci",
                "query_scope": "exact_pair",
                "t1_path": f"/tmp/{index}_a.png",
                "t2_path": f"/tmp/{index}_b.png",
                "ignored_pair_ids": [f"pair:{(index + 1) % count}"] if index == 0 else [],
                "training_enabled": True,
            }
        )
        rows.append(
            {
                "canonical_pair_id": f"pair:{index}",
                "caption_id": f"generic:{index}",
                "caption": "no change occurred",
                "dataset_name": "levir_mci",
                "query_scope": "generic_no_change",
                "t1_path": f"/tmp/{index}_a.png",
                "t2_path": f"/tmp/{index}_b.png",
                "training_enabled": True,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_grouped_rows_excludes_generic_and_disabled_records(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    rows = grouped_rows(
        manifest,
        allowed_scopes={"exact_pair"},
        allowed_sources=None,
        allow_generic_no_change=False,
    )
    assert len(rows) == 128
    assert all(row["query_scopes"] == ["exact_pair"] for row in rows)


def test_fixed_schedule_is_deterministic_and_has_one_common_logical_batch() -> None:
    first = make_schedule(257, 128, 7, 20260802)
    second = make_schedule(257, 128, 7, 20260802)
    assert first == second
    assert all(len(batch) == 128 and len(set(batch)) == 128 for batch in first)
    assert stable_sha(first) == stable_sha(second)


def test_matched_supervision_has_256_by_128_shape_and_ignored_collision() -> None:
    pair_ids = [f"pair:{index}" for index in range(128)]
    query_pair_ids = [pair_id for pair_id in pair_ids for _ in range(2)]
    ignored = [set() for _ in query_pair_ids]
    ignored[0] = {"pair:1"}
    positive, excluded = make_supervision(
        pair_ids,
        query_pair_ids,
        ignored,
        torch.device("cpu"),
    )
    assert positive.shape == (256, 128)
    assert excluded.shape == (256, 128)
    assert positive[0, 0] and not positive[0, 1]
    assert excluded[0, 1]
    assert not excluded[0, 0]


def test_stage2_trainer_has_no_mask_loader_contract() -> None:
    source = Path(__file__).parents[1] / "scripts/train_qcpr_stage2_b1_control.py"
    text = source.read_text(encoding="utf-8")
    assert "dense_label" not in text
    assert "mask_path" not in text
    assert "logical_score_matrix" not in text or "score_matrix_shape" in text
