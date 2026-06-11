from __future__ import annotations

import json
from pathlib import Path

from land_change_detection.levir_mci import discover_levir_mci_samples


def _write_rgb(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)


def _write_mask(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=255).save(path)


def test_official_caption_parser_uses_raw_sentence_strings(tmp_path: Path):
    root = tmp_path / "LEVIR-MCI-dataset"
    _write_rgb(root / "images" / "train" / "A" / "000001.png")
    _write_rgb(root / "images" / "train" / "B" / "000001.png")
    _write_mask(root / "images" / "train" / "label" / "000001.png")
    (root / "LevirCCcaptions.json").write_text(
        json.dumps(
            [
                {
                    "filename": "000001.png",
                    "filepath": "images/train/A/000001.png",
                    "changeflag": 1,
                    "imgid": "img-1",
                    "sentences": [
                        {"raw": "there", "tokens": ["there"], "sentid": 100},
                        {"raw": "A new building appears near the road.", "tokens": ["A", "new", "building"], "sentid": 101},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = discover_levir_mci_samples(root)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.caption == "A new building appears near the road."
    assert len(sample.caption.split()) >= 3
    assert sample.captions == ("there", "A new building appears near the road.")
    assert sample.metadata["captions"] == ["there", "A new building appears near the road."]
    assert sample.metadata["imgid"] == "img-1"
    assert sample.metadata["changeflag"] == 1
    assert sample.metadata["sentids"] == [100, 101]


def test_official_caption_parser_does_not_flatten_token_lists(tmp_path: Path):
    root = tmp_path / "LEVIR-MCI-dataset"
    _write_rgb(root / "images" / "train" / "A" / "000002.png")
    _write_rgb(root / "images" / "train" / "B" / "000002.png")
    _write_mask(root / "images" / "train" / "label" / "000002.png")
    (root / "LevirCCcaptions.json").write_text(
        json.dumps(
            [
                {
                    "filename": "000002.png",
                    "sentences": [
                        {"raw": "A large road extension is visible.", "tokens": ["A", "large", "road", "extension"]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = discover_levir_mci_samples(root)[0]
    assert sample.captions == ("A large road extension is visible.",)
    assert "large" in sample.captions[0]
    assert sample.metadata["captions"] == ["A large road extension is visible."]
