from __future__ import annotations

from pathlib import Path

from PIL import Image

from build_s2looking_qcpr_manifest import build_rows


def _png(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L" if "label" in path.parent.name.casefold() else "RGB", (12, 10), value).save(path)


def test_s2looking_builder_emits_direction_specific_query_masks(tmp_path: Path):
    for split in ("train", "val", "test"):
        for directory in ("Image1", "Image2", "label1", "label2"):
            _png(tmp_path / split / directory / "0001.png", 255 if directory == "label1" else 0)

    rows, audit = build_rows(tmp_path)

    assert len(rows) == 6
    train = [row for row in rows if row["split"] == "train"]
    appeared = next(row for row in train if row["source_metadata"]["change_type"] == "appeared")
    demolished = next(row for row in train if row["source_metadata"]["change_type"] == "disappeared")
    assert appeared["captions"] == ["new buildings appeared"]
    assert appeared["mask_path"].endswith("label1/0001.png")
    assert demolished["captions"] == ["buildings were demolished"]
    assert demolished["mask_path"].endswith("label2/0001.png")
    assert appeared["retrieval_supervision"] is False
    assert appeared["seg_supervision_mode"] == "query_specific"
    assert appeared["pair_id"] != demolished["pair_id"]
    assert audit["total_base_pairs"] == 3


def test_s2looking_builder_rejects_unaligned_labels(tmp_path: Path):
    for directory in ("Image1", "Image2", "label1", "label2"):
        _png(tmp_path / "train" / directory / "0001.png")
    (tmp_path / "train" / "label2" / "0001.png").unlink()

    try:
        build_rows(tmp_path, splits=("train",))
    except ValueError as exc:
        assert "alignment mismatch" in str(exc)
    else:
        raise AssertionError("Expected missing S2Looking label to fail manifest generation")
