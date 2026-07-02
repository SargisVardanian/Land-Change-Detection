from __future__ import annotations

from dataclasses import replace
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
