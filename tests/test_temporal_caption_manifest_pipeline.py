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
from prepare_second_cc_manifest import build_karpathy_rows, build_karpathy_rows_with_audit, discover_raw_rows
from render_unichange_v2_retrieval import _build_eval_dataset, _checkpoint_config, _eval_metadata
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


def _levir_sample(root: Path, split: str, name: str, before_color: tuple[int, int, int], after_color: tuple[int, int, int]) -> None:
    _image(root / f"images/{split}/A/{name}.png", before_color)
    _image(root / f"images/{split}/B/{name}.png", after_color)
    _mask(root / f"images/{split}/label/{name}.png")


def _write_levir_captions(root: Path, names: list[str]) -> None:
    root.joinpath("LevirCCcaptions.json").write_text(
        json.dumps(
            {
                "images": [
                    {"filename": f"{name}.png", "split": "train", "sentences": [{"raw": f"Caption for {name}."}]}
                    for name in names
                ]
            }
        ),
        encoding="utf-8",
    )


def test_levir_default_errors_on_train_test_duplicate_without_replacing_output(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _levir_sample(root, "train", "dup_train", (9, 9, 9), (8, 8, 8))
    _levir_sample(root, "test", "dup_test", (9, 9, 9), (8, 8, 8))
    _write_levir_captions(root, ["dup_train", "dup_test"])
    output = tmp_path / "levir.jsonl"
    output.write_text("old manifest\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_levir_mci_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--audit-report",
            str(audit),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "cross-split duplicate leakage" in result.stderr
    assert output.read_text(encoding="utf-8") == "old manifest\n"


def test_levir_drop_train_retains_test_and_reports_removed_pair(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _levir_sample(root, "train", "dup_train", (9, 9, 9), (8, 8, 8))
    _levir_sample(root, "test", "dup_test", (9, 9, 9), (8, 8, 8))
    _levir_sample(root, "train", "unique_train", (1, 2, 3), (3, 2, 1))
    _write_levir_captions(root, ["dup_train", "dup_test", "unique_train"])
    output = tmp_path / "levir.jsonl"
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_levir_mci_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--audit-report",
            str(audit),
            "--cross-split-duplicate-policy",
            "drop_train",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    pair_ids = {row["pair_id"] for row in rows}
    assert "levir_mci:test:dup_test" in pair_ids
    assert "levir_mci:train:dup_train" not in pair_ids
    report = json.loads(audit.read_text(encoding="utf-8"))
    removed = report["levir_mci_adapter"]["removed_duplicates"]
    assert removed == [
        {
            "removed_pair_id": "levir_mci:train:dup_train",
            "removed_split": "train",
            "retained_pair_id": "levir_mci:test:dup_test",
            "retained_split": "test",
            "t1_sha": removed[0]["t1_sha"],
            "t2_sha": removed[0]["t2_sha"],
        }
    ]


def test_levir_drop_train_retains_val_over_train(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _levir_sample(root, "train", "dup_train", (9, 1, 9), (8, 1, 8))
    _levir_sample(root, "val", "dup_val", (9, 1, 9), (8, 1, 8))
    _write_levir_captions(root, ["dup_train", "dup_val"])
    output = tmp_path / "levir.jsonl"
    audit = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_levir_mci_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
            "--audit-report",
            str(audit),
            "--cross-split-duplicate-policy",
            "drop_train",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    pair_ids = {json.loads(line)["pair_id"] for line in output.read_text(encoding="utf-8").splitlines()}
    assert pair_ids == {"levir_mci:val:dup_val"}


def test_levir_no_invalid_output_file_after_audit_failure(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _image(root / "images/train/A/bad.png", (1, 1, 1))
    _image(root / "images/train/B/bad.png", (2, 2, 2))
    _mask(root / "images/train/label/bad.png")
    output = tmp_path / "levir.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_levir_mci_manifest.py",
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "empty_captions" in result.stderr
    assert not output.exists()


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


def _second_aug_fixture(root: Path, split: str, filenames: list[str]) -> None:
    for index, filename in enumerate(filenames):
        _image(root / split / "rgb" / "A" / filename, (index + 1, 0, 0))
        _image(root / split / "rgb" / "B" / filename, (0, index + 1, 0))
        _mask(root / split / "sem" / "A" / filename, index + 1)
        _mask(root / split / "sem" / "B" / filename, index + 2)


def test_second_cc_canonical_only_strips_random_augment_and_prefers_canonical(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "train", ["000001.png", "000001_random_augment.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "000001_random_augment.png", "split": "train", "sentences": [{"raw": "Augmented view."}]},
                    {"filename": "000001.png", "split": "train", "sentences": [{"raw": "Canonical view."}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows, report = build_karpathy_rows_with_audit(root, annotations, "all", "train", "canonical_only")
    assert len(rows) == 1
    assert rows[0]["original_id"] == "000001"
    assert rows[0]["captions"] == ["Canonical view."]
    assert rows[0]["source_metadata"]["base_pair_id"] == "000001"
    assert rows[0]["source_metadata"]["is_augmented"] is False
    assert report["pre_filter"]["row_count"] == 2
    assert report["post_filter"]["selected_row_count"] == 1


def test_second_cc_train_views_keeps_train_augments_but_excludes_val_test_augments(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "train", ["000001.png", "000001_random_augment_flip.png"])
    _second_aug_fixture(root, "val", ["000002.png", "000002_random_augment.png"])
    _second_aug_fixture(root, "test", ["000003.png", "000003_random_augment_7.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "000001.png", "split": "train", "sentences": [{"raw": "Train canonical."}]},
                    {"filename": "000001_random_augment_flip.png", "split": "train", "sentences": [{"raw": "Train augmented."}]},
                    {"filename": "000002.png", "split": "val", "sentences": [{"raw": "Val canonical."}]},
                    {"filename": "000002_random_augment.png", "split": "val", "sentences": [{"raw": "Val augmented."}]},
                    {"filename": "000003_random_augment_7.png", "split": "test", "sentences": [{"raw": "Test augmented."}]},
                    {"filename": "000003.png", "split": "test", "sentences": [{"raw": "Test canonical."}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows, report = build_karpathy_rows_with_audit(root, annotations, "all", "train", "train_views")
    assert len(rows) == 4
    augmented = [row for row in rows if row["source_metadata"]["is_augmented"]]
    assert len(augmented) == 1
    assert augmented[0]["split"] == "train"
    assert augmented[0]["source_metadata"]["base_pair_id"] == "000001"
    assert augmented[0]["source_metadata"]["augmentation_kind"] == "random_augment_flip"
    assert augmented[0]["source_metadata"]["view_id"] == "000001_random_augment_flip"
    assert {row["captions"][0] for row in rows if row["split"] in {"val", "test"}} == {"Val canonical.", "Test canonical."}
    assert report["post_filter"]["selected_augmented_rows_by_split"] == {"train": 1}


def test_second_cc_all_rows_is_diagnostic_and_keeps_augmented_val_test_warning(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "val", ["000002.png", "000002_random_augment.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "000002.png", "split": "val", "sentences": [{"raw": "Val canonical."}]},
                    {"filename": "000002_random_augment.png", "split": "val", "sentences": [{"raw": "Val augmented."}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows, report = build_karpathy_rows_with_audit(root, annotations, "all", "train", "all_rows")
    assert len(rows) == 2
    assert any(row["source_metadata"]["is_augmented"] and row["split"] == "val" for row in rows)
    assert "diagnostic-only" in report["post_filter"]["warning"]


def test_second_cc_base_pair_leakage_across_splits_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "train", ["000001.png"])
    _second_aug_fixture(root, "test", ["000001.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "000001.png", "split": "train", "sentences": [{"raw": "Train."}]},
                    {"filename": "000001.png", "split": "test", "sentences": [{"raw": "Test leak."}]},
                ]
            }
        ),
        encoding="utf-8",
    )
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
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "base-pair leakage" in result.stderr
    assert not (tmp_path / "out.jsonl").exists()


def test_second_cc_deterministic_fallback_when_only_augmented_view_exists(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "train", ["000001_random_augment_z.png", "000001_random_augment_a.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "000001_random_augment_z.png", "split": "train", "sentences": [{"raw": "Z view."}]},
                    {"filename": "000001_random_augment_a.png", "split": "train", "sentences": [{"raw": "A view."}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    first, _ = build_karpathy_rows_with_audit(root, annotations, "all", "train", "canonical_only")
    second, _ = build_karpathy_rows_with_audit(root, annotations, "all", "train", "canonical_only")
    assert first == second
    assert first[0]["source_metadata"]["view_id"] == "000001_random_augment_a"


def test_second_cc_expected_count_mismatch_is_detailed_and_atomic(tmp_path: Path) -> None:
    root = tmp_path / "second"
    _second_aug_fixture(root, "train", ["000001.png"])
    annotations = root / "SECOND-CC-AUG.json"
    annotations.write_text(json.dumps({"images": [{"filename": "000001.png", "split": "train", "sentences": [{"raw": "One caption."}]}]}), encoding="utf-8")
    output = tmp_path / "out.jsonl"
    output.write_text("old manifest\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_second_cc_manifest.py",
            "--root",
            str(root),
            "--annotations",
            str(annotations),
            "--output",
            str(output),
            "--expected-pairs",
            "6041",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "SECOND-CC expected pair count mismatch" in result.stderr
    assert "found_pairs" in result.stderr
    assert output.read_text(encoding="utf-8") == "old manifest\n"


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
    assert "SECOND-CC expected pair count mismatch" in result.stderr


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
    assert "levir_mci_cross_MRR" in metrics
    assert "second_cc_within_exact_R@10" in metrics
    assert metrics["levir_mci_num_queries"] == 2
    assert metrics["second_cc_num_candidates"] == 2


def test_mixed_evaluation_loads_canonical_manifests_and_reports_metadata(tmp_path: Path) -> None:
    rows = []
    for dataset in ("levir_mci", "second_cc"):
        t1 = tmp_path / dataset / "a.png"
        t2 = tmp_path / dataset / "b.png"
        _image(t1, (1, 0, 0))
        _image(t2, (0, 1, 0))
        rows.append(make_manifest_row(dataset_name=dataset, split="val", original_id="x", t1_path=t1, t2_path=t2, captions=[f"{dataset} caption"]))
    manifest = tmp_path / "val.jsonl"
    write_jsonl(manifest, rows)
    args = type(
        "Args",
        (),
        {
            "data_root": tmp_path,
            "output_dir": tmp_path,
            "universat_source": tmp_path,
            "universat_checkpoint": tmp_path,
            "jina_model": tmp_path,
            "split": "val",
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "val_manifest": [manifest],
            "dataset_config": None,
            "dataset_weight": ["levir_mci=0.55", "second_cc=0.45"],
        },
    )()
    config = _checkpoint_config(args, {"config": {"image_size": 4, "output_grid": 4, "temporal_depth": 6}})
    dataset, data_mode = _build_eval_dataset(args, config)
    metadata = _eval_metadata(args, dataset, data_mode)
    assert data_mode == "mixed"
    assert metadata["dataset_names"] == ["levir_mci", "second_cc"]
    assert metadata["validation_row_counts"] == {"levir_mci": 1, "second_cc": 1}
    assert metadata["manifest_fingerprints"]["validation"][str(manifest)]


def test_levir_only_evaluation_backward_compatibility_and_depth6_config(tmp_path: Path) -> None:
    root = tmp_path / "levir"
    _image(root / "images/train/A/t.png", (1, 1, 1))
    _image(root / "images/train/B/t.png", (2, 2, 2))
    _mask(root / "images/train/label/t.png")
    _image(root / "images/val/A/a.png", (1, 2, 3))
    _image(root / "images/val/B/a.png", (3, 2, 1))
    _mask(root / "images/val/label/a.png")
    (root / "LevirCCcaptions.json").write_text(
        json.dumps({"images": [{"filename": "a.png", "split": "val", "sentences": [{"raw": "A change occurred."}]}]}),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "data_root": root,
            "output_dir": tmp_path,
            "universat_source": tmp_path,
            "universat_checkpoint": tmp_path,
            "jina_model": tmp_path,
            "split": "val",
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "val_manifest": [],
            "dataset_config": None,
            "dataset_weight": None,
        },
    )()
    config = _checkpoint_config(args, {"config": {"image_size": 4, "output_grid": 4, "temporal_depth": 6}})
    dataset, data_mode = _build_eval_dataset(args, config)
    assert data_mode == "levir_only"
    assert config.temporal_depth == 6
    assert len(dataset) == 1
