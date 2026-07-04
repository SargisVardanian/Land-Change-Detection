from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ucv2_stage1_next_core import (
    Stage1NextConfig,
    _mixed_subset_coverage_details,
    audit_dataset_conflicts,
    composite_score,
    make_optimizer,
    save_checkpoint,
    _selection_scores,
)
from land_change_detection.temporal_caption_manifest import SCHEMA_VERSION


def _config() -> Stage1NextConfig:
    return Stage1NextConfig(
        data_root=".",
        output_dir=".",
        universat_source=".",
        universat_checkpoint=".",
        jina_model=".",
        batch_size=2,
        num_workers=0,
    )


def test_optimizer_groups_cover_trainable_params_and_no_decay_specials():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_encoder = nn.Module()
            self.temporal_encoder.direction_embeddings = nn.Parameter(torch.zeros(2, 4))
            self.temporal_encoder.linear = nn.Linear(4, 4)
            self.retrieval_head = nn.Module()
            self.retrieval_head.logit_scale = nn.Parameter(torch.tensor(1.0))
            self.retrieval_head.linear = nn.Linear(4, 4)
            self.text_adapter = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))

    model = Model()
    optimizer = make_optimizer(model, _config())
    grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert sorted(grouped) == sorted(trainable)
    assert len(grouped) == len(set(grouped))
    no_decay_names = {
        name
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for name in group.get("param_names", [])
    }
    assert "temporal_encoder.direction_embeddings" in no_decay_names
    assert "retrieval_head.logit_scale" in no_decay_names
    assert any(group.get("name") == "text_adapter_no_decay" for group in optimizer.param_groups)


def test_conflict_audit_rules_and_optional_filter_indices():
    samples = [
        SimpleNamespace(captions=["No change has occurred."], mask=torch.ones(4, 4), metadata={"changeflag": 0}),
        SimpleNamespace(captions=["A new road appeared."], mask=torch.zeros(4, 4), metadata={"changeflag": 1}),
        SimpleNamespace(captions=["A house appeared.", "A house disappeared."], mask=torch.ones(4, 4), metadata={"changeflag": 1}),
    ]
    dataset = SimpleNamespace(samples=samples)
    counts, indices = audit_dataset_conflicts(dataset, _config())
    assert set(indices) == {0, 1, 2}
    assert counts["all_no_change_captions_mask_changed"] == 1
    assert counts["all_changed_captions_empty_mask"] == 1
    assert counts["appeared_disappeared_caption_contradiction"] == 1


def test_composite_score_uses_documented_weights():
    metrics = {
        "macro_semantic_recall@1": 1.0,
        "macro_semantic_recall@5": 0.8,
        "macro_semantic_recall@10": 0.6,
        "macro_semantic_nDCG@10": 0.5,
        "detailed_query_R@5": 0.4,
        "exact_pair_R@10": 1.0,
    }
    assert composite_score(metrics) == 0.30 + 0.25 * 0.8 + 0.20 * 0.6 + 0.15 * 0.5 + 0.10 * 0.4
    assert "exact_r10" not in _selection_scores(metrics)


def test_checkpoint_save_roundtrip_records_selection_metadata(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_encoder = nn.Linear(2, 2)
            self.retrieval_head = nn.Linear(2, 2)

    model = Model()
    optimizer = make_optimizer(model, _config())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "best_composite.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        _config(),
        epoch_index=3,
        next_batch_index=0,
        step=17,
        best_scores={"composite": 0.7},
        metrics={"text_to_pair_R@1": 0.5},
        selection_metric="composite",
        selection_value=0.7,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["stage1_next"] is True
    assert payload["selection_metric"] == "composite"
    assert payload["selection_value"] == 0.7
    model.load_state_dict(payload["model"])


def test_stage1_next_does_not_introduce_pair_to_pair_or_exact_selection_modules():
    root = Path(__file__).resolve().parents[1]
    stage1_text = (root / "scripts" / "ucv2_stage1_next_core.py").read_text(encoding="utf-8")
    retrieval_head_text = (root / "src" / "land_change_detection" / "models" / "retrieval_heads.py").read_text(encoding="utf-8")
    assert "pair_to_pair" not in stage1_text
    assert "pair-to-pair" not in stage1_text
    assert "best_exact_r10" not in stage1_text
    assert "exact_r10" not in stage1_text
    assert "pair_to_pair" not in retrieval_head_text


def test_stage1_next_slurm_scripts_pin_pythonpath_after_code_root_override():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "cluster" / "ysu" / "train_unichange_v2_stage1_next.sbatch",
        root / "cluster" / "ysu" / "smoke_unichange_v2_stage1_next.sbatch",
        root / "cluster" / "ysu" / "probe_unichange_v2_stage1_next_memory.sbatch",
        root / "cluster" / "ysu" / "evaluate_unichange_v2_stage1_next.sbatch",
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        code_root_index = text.index("export CODE_ROOT=")
        cd_index = text.index("cd ", code_root_index)
        pythonpath_index = text.index("export PYTHONPATH=", cd_index)
        assert pythonpath_index > cd_index
        assert "$CODE_ROOT/src:$CODE_ROOT/scripts:$CODE_ROOT" in text or "${CODE_ROOT}/src:${CODE_ROOT}/scripts:${CODE_ROOT}" in text

    train_text = scripts[0].read_text(encoding="utf-8")
    gate_index = train_text.index("ucv2_stage1_next_readiness_gate.py")
    pythonpath_index = train_text.index("export PYTHONPATH=", train_text.index("cd "))
    assert pythonpath_index < gate_index
    assert 'BATCH_SIZE="$($PYTHON "$CODE_ROOT/scripts/ucv2_stage1_next_readiness_gate.py"' in train_text


def _write_readiness_reports(tmp_path: Path, *, temporal_depth: int) -> tuple[Path, Path]:
    smoke = {
        "real_cluster_smoke_passed": True,
        "git_commit": "abc123",
        "slurm_job_id": "12345",
        "slurm_job_name": "ucv2-next-smoke",
        "steps_completed": 10,
        "finite_loss": True,
        "device_type": "cuda",
        "bf16_active": True,
        "fake_backbones": False,
        "train_val_disjoint": True,
        "frozen_grad_violations": [],
        "missing_gradients": [],
        "checkpoint_roundtrip_passed": True,
        "image_size": 256,
        "output_grid": 32,
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": temporal_depth,
        "text_adapter_enabled": True,
        "text_max_length": 256,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "data_mode": "levir_only",
        "mixed_smoke": False,
    }
    memory = {
        "status": "PASS",
        "git_commit": "abc123",
        "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": temporal_depth,
        "text_adapter_enabled": True,
        "text_max_length": 256,
        "recommended_batch_size": 32,
        "memory_data_mode": "shape_probe",
    }
    smoke_path = tmp_path / "smoke.json"
    memory_path = tmp_path / "memory.json"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    memory_path.write_text(json.dumps(memory), encoding="utf-8")
    return smoke_path, memory_path


def test_stage1_next_readiness_gate_requires_depth6_reports(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=6)
    result = subprocess.run(
        [sys.executable, "scripts/ucv2_stage1_next_readiness_gate.py", str(smoke_path), str(memory_path), "abc123", "1", "32", "6"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "32"


def test_stage1_next_readiness_gate_rejects_depth4_reports_for_depth6_training(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=4)
    result = subprocess.run(
        [sys.executable, "scripts/ucv2_stage1_next_readiness_gate.py", str(smoke_path), str(memory_path), "abc123", "1", "32", "6"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "temporal_depth" in result.stderr


def test_stage1_next_readiness_gate_rejects_text_max_length_mismatch(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=6)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["text_max_length"] = 96
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ucv2_stage1_next_readiness_gate.py",
            str(smoke_path),
            str(memory_path),
            "abc123",
            "1",
            "32",
            "6",
            "--expected-text-max-length",
            "256",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "text_max_length" in result.stderr


def test_mixed_subset_coverage_details_passes_when_smoke_subset_keeps_weighted_datasets():
    dataset = SimpleNamespace(
        indices_by_dataset={"levir_mci": list(range(10)), "second_cc": list(range(10, 20))},
        selection_metadata={"available_counts_by_dataset": {"levir_mci": 100, "second_cc": 100}, "omitted_datasets": []},
    )
    details = _mixed_subset_coverage_details(
        dataset,
        configured_weights={"levir_mci": 0.55, "second_cc": 0.45},
        max_pairs=20,
        subset_name="train",
    )
    assert details["sample_counts_by_dataset"] == {"levir_mci": 10, "second_cc": 10}
    assert details["configured_positive_sampling_weights"] == {"levir_mci": 0.55, "second_cc": 0.45}
    assert details["mixed_subset_coverage_passed"] is True


def test_mixed_subset_coverage_details_fails_early_when_positive_weight_dataset_is_omitted():
    dataset = SimpleNamespace(
        indices_by_dataset={"levir_mci": [0]},
        selection_metadata={
            "available_counts_by_dataset": {"levir_mci": 100, "second_cc": 100},
            "omitted_datasets": ["second_cc"],
        },
    )
    with pytest.raises(ValueError, match="Mixed-smoke coverage error"):
        _mixed_subset_coverage_details(
            dataset,
            configured_weights={"levir_mci": 0.55, "second_cc": 0.45},
            max_pairs=1,
            subset_name="train",
        )


def _manifest(path: Path, dataset: str, count: int, *, splits: tuple[str, ...] = ("train", "val")) -> None:
    rows = []
    for split in splits:
        for index in range(count):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": dataset,
                    "pair_id": f"{dataset}:{split}:{index}",
                    "captions": ["caption"],
                    "caption_source": "human",
                    "split": split,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _mixed_reports(tmp_path: Path, *, temporal_depth: int = 6):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=temporal_depth)
    levir = tmp_path / "levir.jsonl"
    second = tmp_path / "second.jsonl"
    _manifest(levir, "levir_mci", 2)
    _manifest(second, "second_cc", 3)
    import hashlib

    def fp(path: Path) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(path.read_bytes())
        return digest.hexdigest()

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.update(
        {
            "data_mode": "mixed",
            "mixed_smoke": True,
            "manifest_fingerprints": {"train": {str(levir): fp(levir), str(second): fp(second)}, "validation": {str(levir): fp(levir), str(second): fp(second)}},
            "dataset_names": ["levir_mci", "second_cc"],
            "dataset_weights": {"levir_mci": 0.55, "second_cc": 0.45},
            "train_row_count": 5,
            "validation_row_count": 5,
            "full_train_row_count": 5,
            "full_validation_row_count": 5,
            "selected_train_row_count": 5,
            "selected_validation_row_count": 5,
            "mixed_subset_coverage_passed": True,
            "sample_counts_by_dataset": {"levir_mci": 2, "second_cc": 3},
            "validation_sample_counts_by_dataset": {"levir_mci": 2, "second_cc": 3},
        }
    )
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    return smoke_path, memory_path, levir, second


def _mixed_gate_args(smoke_path: Path, memory_path: Path, levir: Path, second: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/ucv2_stage1_next_readiness_gate.py",
        str(smoke_path),
        str(memory_path),
        "abc123",
        "1",
        "32",
        "6",
        "--expected-data-mode",
        "mixed",
        "--expected-train-manifest",
        str(levir),
        "--expected-train-manifest",
        str(second),
        "--expected-val-manifest",
        str(levir),
        "--expected-val-manifest",
        str(second),
        "--expected-dataset-weight",
        "levir_mci=0.55",
        "--expected-dataset-weight",
        "second_cc=0.45",
    ]


def _write_mixed_split_manifest(path: Path, dataset: str, *, train: int, val: int, test: int) -> None:
    rows = []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for index in range(count):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": dataset,
                    "pair_id": f"{dataset}:{split}:{index:04d}",
                    "captions": [f"{dataset} {split} caption {index}"],
                    "caption_source": "human",
                    "split": split,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fingerprint(path: Path) -> str:
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_mixed_readiness_gate_uses_split_filtered_full_counts_and_smoke_caps(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=6)
    levir = tmp_path / "levir_mixed.jsonl"
    second = tmp_path / "second_mixed.jsonl"
    _write_mixed_split_manifest(levir, "levir_mci", train=14, val=10, test=3)
    _write_mixed_split_manifest(second, "second_cc", train=14, val=10, test=3)

    raw_total = sum(1 for path in (levir, second) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert raw_total == 54
    assert raw_total != 28
    assert raw_total != 20

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.update(
        {
            "data_mode": "mixed",
            "mixed_smoke": True,
            "manifest_fingerprints": {
                "train": {str(levir): _fingerprint(levir), str(second): _fingerprint(second)},
                "validation": {str(levir): _fingerprint(levir), str(second): _fingerprint(second)},
            },
            "dataset_names": ["levir_mci", "second_cc"],
            "dataset_weights": {"levir_mci": 0.55, "second_cc": 0.45},
            "train_row_count": 28,
            "validation_row_count": 20,
            "full_train_row_count": 28,
            "full_validation_row_count": 20,
            "selected_train_row_count": 20,
            "selected_validation_row_count": 16,
            "mixed_subset_coverage_passed": True,
            "sample_counts_by_dataset": {"levir_mci": 10, "second_cc": 10},
            "validation_sample_counts_by_dataset": {"levir_mci": 8, "second_cc": 8},
        }
    )
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")

    valid = subprocess.run(
        _mixed_gate_args(smoke_path, memory_path, levir, second),
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.strip() == "32"

    wrong_split = dict(smoke)
    wrong_split["train_row_count"] = raw_total
    smoke_path.write_text(json.dumps(wrong_split), encoding="utf-8")
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "train_row_count" in result.stderr

    missing_metadata = dict(smoke)
    missing_metadata.pop("gpu_name")
    smoke_path.write_text(json.dumps(missing_metadata), encoding="utf-8")
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "gpu_name" in result.stderr

    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    result = subprocess.run(
        _mixed_gate_args(smoke_path, memory_path, levir, second)[:4]
        + ["different-commit"]
        + _mixed_gate_args(smoke_path, memory_path, levir, second)[5:],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "git_commit" in result.stderr


def test_mixed_training_rejects_levir_only_smoke(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=6)
    levir = tmp_path / "levir.jsonl"
    second = tmp_path / "second.jsonl"
    _manifest(levir, "levir_mci", 1)
    _manifest(second, "second_cc", 1)
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "data_mode" in result.stderr


def test_mixed_readiness_rejects_fingerprint_weight_and_row_count_mismatches(tmp_path):
    smoke_path, memory_path, levir, second = _mixed_reports(tmp_path)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["dataset_weights"] = {"levir_mci": 1.0, "second_cc": 0.0}
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "dataset_weights" in result.stderr

    smoke_path, memory_path, levir, second = _mixed_reports(tmp_path)
    levir.write_text(
        levir.read_text(encoding="utf-8")
        + json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": "levir_mci",
                "pair_id": "extra",
                "split": "val",
                "caption_source": "human",
                "captions": ["x"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "manifest_fingerprints" in result.stderr or "row_count" in result.stderr

    smoke_path, memory_path, levir, second = _mixed_reports(tmp_path)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["validation_row_count"] = 4
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    result = subprocess.run(_mixed_gate_args(smoke_path, memory_path, levir, second), cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validation_row_count" in result.stderr


def test_three_dataset_mixed_readiness_accepts_matching_reports(tmp_path):
    smoke_path, memory_path = _write_readiness_reports(tmp_path, temporal_depth=6)
    manifests = [
        (tmp_path / "levir.jsonl", "levir_mci", 2, 0.4),
        (tmp_path / "second.jsonl", "second_cc", 3, 0.35),
        (tmp_path / "rscc.jsonl", "rscc", 4, 0.25),
    ]
    import hashlib

    def fp(path: Path) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(path.read_bytes())
        return digest.hexdigest()

    for path, dataset, count, _ in manifests:
        _manifest(path, dataset, count)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.update(
        {
            "data_mode": "mixed",
            "mixed_smoke": True,
            "manifest_fingerprints": {
                "train": {str(path): fp(path) for path, _, _, _ in manifests},
                "validation": {str(path): fp(path) for path, _, _, _ in manifests},
            },
            "dataset_names": ["levir_mci", "rscc", "second_cc"],
            "dataset_weights": {dataset: weight for _, dataset, _, weight in manifests},
            "train_row_count": 9,
            "validation_row_count": 9,
            "full_train_row_count": 9,
            "full_validation_row_count": 9,
            "selected_train_row_count": 9,
            "selected_validation_row_count": 9,
            "mixed_subset_coverage_passed": True,
            "sample_counts_by_dataset": {"levir_mci": 2, "rscc": 4, "second_cc": 3},
            "validation_sample_counts_by_dataset": {"levir_mci": 2, "rscc": 4, "second_cc": 3},
        }
    )
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    config = tmp_path / "dataset_config.json"
    config.write_text(
        json.dumps(
            {
                "train_manifests": [str(path) for path, _, _, _ in manifests],
                "val_manifests": [str(path) for path, _, _, _ in manifests],
                "dataset_sampling_weights": {dataset: weight for _, dataset, _, weight in manifests},
                "semantic_soft_target_weight": 0.25,
                "semantic_teacher_top_k": 8,
                "semantic_teacher_temperature": 0.05,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ucv2_stage1_next_readiness_gate.py",
            str(smoke_path),
            str(memory_path),
            "abc123",
            "1",
            "32",
            "6",
            "--expected-data-mode",
            "mixed",
            "--expected-dataset-config",
            str(config),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
