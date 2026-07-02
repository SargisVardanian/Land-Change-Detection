from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ucv2_stage1_next_core import (
    Stage1NextConfig,
    audit_dataset_conflicts,
    composite_score,
    make_optimizer,
    save_checkpoint,
)


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
    metrics = {"text_to_pair_R@1": 1.0, "text_to_pair_R@5": 0.8, "text_to_pair_R@10": 0.6, "MRR": 0.5}
    assert composite_score(metrics) == 0.35 + 0.25 * 0.8 + 0.20 * 0.6 + 0.20 * 0.5


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


def _write_readiness_reports(tmp_path: Path, *, temporal_depth: int) -> tuple[Path, Path]:
    smoke = {
        "real_cluster_smoke_passed": True,
        "git_commit": "abc123",
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
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": temporal_depth,
        "text_adapter_enabled": True,
        "gpu_name": "NVIDIA H100 80GB HBM3",
    }
    memory = {
        "status": "PASS",
        "git_commit": "abc123",
        "stage1_next": True,
        "loss": "multi_positive_set_info_nce",
        "stable_caption_groups": True,
        "use_direction_embeddings": True,
        "use_explicit_change_fusion": True,
        "trainable_temperature": True,
        "temporal_depth": temporal_depth,
        "text_adapter_enabled": True,
        "recommended_batch_size": 32,
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
