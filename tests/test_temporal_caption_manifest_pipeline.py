from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from land_change_detection.temporal_caption_manifest import (
    annotation_rows_to_manifest,
    audit_manifest_rows,
    make_manifest_row,
    write_jsonl,
)
from land_change_detection.training.temporal_caption_dataset import (
    DeterministicWeightedDatasetSampler,
    TemporalCaptionManifestDataset,
)
from prepare_levir_mci_manifest import build_rows as build_levir_rows
from prepare_second_cc_manifest import build_karpathy_rows, discover_raw_rows
from ucv2_retrieval_metrics import RetrievalCorpus, compute_retrieval_metrics


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 5), color).save(path)


def _mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 5), value).save(path)


def test_levir_mci_manifest_preserves_splits_and_namespaces(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _image(root / "images/train/A/a.png", (1, 2, 3))
    _image(root / "images/train/B/a.png", (3, 2, 1))
    _mask(root / "images/train/label/a.png")
    (root / "LevirCCcaptions.json").write_text(
        json.dumps({"images": [{"filename": "a.png", "split": "train", "sentences": [{"raw": "No change has occurred."}]}]}),
        encoding="utf-8",
    )
    rows = build_levir_rows(root, "all")
    assert rows[0]["pair_id"] == "levir_mci:train:a"
    assert rows[0]["split"] == "train"
    assert rows[0]["caption_source"] == "human"
    assert rows[0]["normalized_caption_groups"] == ["no change has occurred"]
    assert audit_manifest_rows(rows)["valid"]


def test_second_cc_raw_adapter_does_not_need_hdf5(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _image(root / "train/A/s1.png", (1, 0, 0))
    _image(root / "train/B/s1.png", (0, 1, 0))
    _mask(root / "train/label/s1.png")
    _mask(root / "train/semantic_A/s1.png", 1)
    _mask(root / "train/semantic_B/s1.png", 2)
    (root / "captions.json").write_text(json.dumps({"s1": ["A building appeared."]}), encoding="utf-8")
    rows = discover_raw_rows(root, "all")
    assert rows[0]["pair_id"] == "second_cc:train:s1"
    assert rows[0]["mask_path"].endswith("label/s1.png")
    assert rows[0]["semantic_t1_path"].endswith("semantic_A/s1.png")
    assert audit_manifest_rows(rows)["valid"]


def test_second_cc_karpathy_manifest_uses_official_layout_and_semantics(tmp_path: Path) -> None:
    root = tmp_path / "second"
    filename = "000001.png"
    _image(root / "train/rgb/A" / filename, (1, 0, 0))
    _image(root / "train/rgb/B" / filename, (0, 1, 0))
    _mask(root / "train/sem/A" / filename, 1)
    _mask(root / "train/sem/B" / filename, 2)
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "filename": filename,
                        "split": "train",
                        "sentences": [
                            {"tokens": ["a", "road", "appeared"], "raw": "A road appeared."},
                            {"tokens": ["new", "construction"], "sentid": 42},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = build_karpathy_rows(root, annotations, "all", "train")
    assert rows[0]["pair_id"] == "second_cc:train:000001"
    assert rows[0]["t1_path"].endswith("train/rgb/A/000001.png")
    assert rows[0]["t2_path"].endswith("train/rgb/B/000001.png")
    assert rows[0]["mask_path"] is None
    assert rows[0]["semantic_t1_path"].endswith("train/sem/A/000001.png")
    assert rows[0]["semantic_t2_path"].endswith("train/sem/B/000001.png")
    assert rows[0]["captions"] == ["A road appeared.", "new construction"]
    assert rows[0]["source_metadata"]["official_split"] == "train"
    assert audit_manifest_rows(rows)["valid"]


def test_second_cc_restval_policy_and_expected_count_validation(tmp_path: Path) -> None:
    root = tmp_path / "second"
    filename = "rest.png"
    _image(root / "train/rgb/A" / filename, (1, 0, 0))
    _image(root / "train/rgb/B" / filename, (0, 1, 0))
    _mask(root / "train/sem/A" / filename, 1)
    _mask(root / "train/sem/B" / filename, 2)
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps({"images": [{"filename": filename, "split": "restval", "sentences": [{"tokens": ["rest", "caption"]}]}]}),
        encoding="utf-8",
    )
    rows = build_karpathy_rows(root, annotations, "all", "train")
    assert rows[0]["split"] == "train"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_second_cc_manifest.py",
            "--root",
            str(root),
            "--annotations",
            str(annotations),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--expected-pairs",
            "6041",
            "--expected-captions",
            "30205",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Expected 6041 SECOND-CC pairs" in result.stderr


def test_rscc_adapter_excludes_model_generated_by_default_and_preserves_license(tmp_path: Path) -> None:
    root = tmp_path / "rscc"
    _image(root / "a1.png", (1, 1, 1))
    _image(root / "a2.png", (2, 2, 2))
    annotations = root / "ann.json"
    annotations.write_text(
        json.dumps(
            [
                {"id": "h", "split": "train", "t1_path": "a1.png", "t2_path": "a2.png", "caption": "Damage appeared.", "license": "xBD", "caption_source": "human"},
                {"id": "m", "split": "train", "t1_path": "a1.png", "t2_path": "a2.png", "caption": "Generated.", "license": "xBD", "caption_source": "model_generated"},
            ]
        ),
        encoding="utf-8",
    )
    rows = annotation_rows_to_manifest(dataset_name="rscc", root=root, annotations=annotations, include_model_generated=False)
    assert [row["pair_id"] for row in rows] == ["rscc:train:h"]
    assert rows[0]["source_metadata"]["license"] == "xBD"


def test_audit_detects_missing_files_duplicates_and_split_leakage(tmp_path: Path) -> None:
    _image(tmp_path / "a.png", (1, 1, 1))
    _image(tmp_path / "b.png", (2, 2, 2))
    train = make_manifest_row(dataset_name="levir_mci", split="train", original_id="same", t1_path=tmp_path / "a.png", t2_path=tmp_path / "b.png", captions=["changed"])
    val = make_manifest_row(dataset_name="levir_mci", split="val", original_id="same", t1_path=tmp_path / "a.png", t2_path=tmp_path / "b.png", captions=["changed"])
    missing = make_manifest_row(dataset_name="second_cc", split="train", original_id="missing", t1_path=tmp_path / "missing.png", t2_path=tmp_path / "b.png", captions=[])
    report = audit_manifest_rows([train, val, missing])
    kinds = {error["kind"] for error in report["errors"]}
    assert "missing_file" in kinds
    assert "empty_captions" in kinds
    assert "duplicated_t1_t2_pairs_across_splits" in kinds
    assert "train_validation_test_leakage" in kinds
    assert report["exact_duplicate_images"]


def test_manifest_dataset_sampler_is_deterministic_and_rotates(tmp_path: Path) -> None:
    rows = []
    for dataset in ("levir_mci", "second_cc"):
        for index in range(3):
            t1 = tmp_path / dataset / f"{index}_a.png"
            t2 = tmp_path / dataset / f"{index}_b.png"
            _image(t1, (index, 0, 0))
            _image(t2, (0, index, 0))
            rows.append(make_manifest_row(dataset_name=dataset, split="train", original_id=str(index), t1_path=t1, t2_path=t2, captions=[f"{dataset} caption {index}"]))
    manifest = tmp_path / "mixed.jsonl"
    write_jsonl(manifest, rows)
    dataset = TemporalCaptionManifestDataset(manifest, split="train", image_size=4)
    weights = {"levir_mci": 0.55, "second_cc": 0.45}
    first = list(DeterministicWeightedDatasetSampler(dataset, weights=weights, seed=7, epoch=0, num_samples=8))
    again = list(DeterministicWeightedDatasetSampler(dataset, weights=weights, seed=7, epoch=0, num_samples=8))
    next_epoch = list(DeterministicWeightedDatasetSampler(dataset, weights=weights, seed=7, epoch=1, num_samples=8))
    assert first == again
    assert first != next_epoch
    item = dataset[first[0]]
    assert item.dataset_name in {"levir_mci", "second_cc"}
    assert item.metadata["caption_source"] == "human"


def test_manifest_inspection_command_reports_counts_and_fingerprint(tmp_path: Path) -> None:
    t1 = tmp_path / "a.png"
    t2 = tmp_path / "b.png"
    _image(t1, (1, 0, 0))
    _image(t2, (0, 1, 0))
    manifest = tmp_path / "inspect.jsonl"
    write_jsonl(
        manifest,
        [make_manifest_row(dataset_name="levir_mci", split="train", original_id="x", t1_path=t1, t2_path=t2, captions=["No change."])],
    )
    result = subprocess.run(
        [sys.executable, "scripts/inspect_temporal_caption_manifest.py", str(manifest)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["pairs_by_dataset_split"] == {"levir_mci:train": 1}
    assert payload["captions_by_dataset_split"] == {"levir_mci:train": 1}
    assert payload["captions_per_pair_distribution"] == {"1": 1}
    assert payload["fingerprint"]


def test_per_dataset_metrics_are_query_masks_over_same_rank_tensor() -> None:
    corpus = RetrievalCorpus(
        pair_embeddings=torch.eye(4, dtype=torch.float32),
        text_embeddings=torch.eye(4, dtype=torch.float32),
        caption_to_pair=torch.tensor([0, 1, 2, 3]),
        caption_group_ids=torch.tensor([0, 1, 2, 3]),
        pair_ids=["levir_mci:val:a", "levir_mci:val:b", "second_cc:val:c", "second_cc:val:d"],
        captions=["a", "b", "c", "d"],
        pair_mask_fractions=torch.zeros(4),
        encode_seconds=0.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
        dataset_names=["levir_mci", "levir_mci", "second_cc", "second_cc"],
    )
    metrics, _ = compute_retrieval_metrics(corpus, query_chunk_size=1, candidate_chunk_size=3)
    assert metrics["text_to_pair_R@1"] == 1.0
    assert metrics["levir_mci_R@1"] == 1.0
    assert metrics["second_cc_R@1"] == 1.0
    assert metrics["levir_mci_cross_R@1"] == 1.0
    assert metrics["second_cc_cross_R@1"] == 1.0
    assert metrics["levir_mci_within_R@1"] == 1.0
    assert metrics["second_cc_within_R@1"] == 1.0
    assert metrics["levir_mci_num_queries"] == 2
    assert metrics["second_cc_num_candidates"] == 2
