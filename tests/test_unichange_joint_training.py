from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from land_change_detection.data.event_targets import ComponentTargets, TemporalContext, build_component_targets, connected_components_8
from land_change_detection.data.unichange_mci import UniChangeMciItem, collate_unichange_mci
from land_change_detection.losses.unichange_losses import masked_multi_positive_sigmoid_loss
from land_change_detection.models.unichange_model import UniChangeOutput
from land_change_detection.training.hungarian_event_matcher import hungarian_match_events
from land_change_detection.training.safe_negative_miner import mine_safe_negative_mask
from land_change_detection.training.unichange_joint_trainer import (
    UniChangeJointTrainer,
    UniChangeJointTrainerConfig,
    scheduled_joint_weights,
)
from scripts.render_unichange_joint import render_training_panel


def _components(mask: Tensor) -> ComponentTargets:
    return build_component_targets(mask, grid_size=(36, 36), min_area=1)


def test_pair_centric_batch_preserves_all_sibling_captions() -> None:
    mask = torch.ones(8, 8)
    items = [
        UniChangeMciItem("p0", torch.zeros(3, 8, 8), torch.ones(3, 8, 8), ["a", "b"], mask, _components(mask), TemporalContext(), {}),
        UniChangeMciItem("p1", torch.zeros(3, 8, 8), torch.ones(3, 8, 8), ["c"], mask, _components(mask), TemporalContext(), {}),
    ]
    batch = collate_unichange_mci(items)
    assert batch["captions"] == ["a", "b", "c"]
    assert batch["caption_to_pair"].tolist() == [0, 0, 1]
    pairs = torch.nn.functional.normalize(torch.eye(2, 4), dim=-1)
    texts = torch.nn.functional.normalize(torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]]), dim=-1)
    loss, stats = masked_multi_positive_sigmoid_loss(pairs, texts, batch["caption_to_pair"], torch.zeros(3, 2, dtype=torch.bool))
    assert torch.isfinite(loss)
    assert stats["positive_ratio"] == 0.5


def test_safe_negative_mining_never_marks_all_off_diagonal() -> None:
    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]]), dim=-1)
    caption_to_pair = torch.tensor([0, 0, 1, 2])
    mask = mine_safe_negative_mask(text, caption_to_pair, num_pairs=3, bottom_quantile=0.35)
    positives = torch.nn.functional.one_hot(caption_to_pair, num_classes=3).bool()
    assert not torch.any(mask & positives)
    assert mask.sum().item() > 0
    assert mask.sum().item() < (~positives).sum().item()


def test_connected_components_are_8_connected_and_resized() -> None:
    mask = torch.zeros(8, 8)
    mask[1, 1] = 1
    mask[2, 2] = 1
    mask[6, 6] = 1
    assert len(connected_components_8(mask, min_area=1)) == 2
    targets = build_component_targets(mask, grid_size=(36, 36), min_area=1)
    assert targets.masks.shape == (2, 1296)
    assert targets.full_resolution_mask.shape == (8, 8)


def test_hungarian_matching_aligns_events_to_components() -> None:
    components = torch.zeros(2, 4)
    components[0, :2] = 1
    components[1, 2:] = 1
    logits = torch.tensor([[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]])
    match = hungarian_match_events(logits, components)
    assert match.event_indices.tolist() == [0, 1]
    assert match.component_indices.tolist() == [0, 1]


def test_temporal_context_marks_levir_time_as_ordinal_unknown_duration() -> None:
    context = TemporalContext()
    payload = context.to_dict()
    assert payload["order"] == ["before", "after"]
    assert payload["delta_days"] is None
    assert payload["duration_known"] is False
    assert payload["time_semantics"] == "ordinal_not_calendar"


def test_scheduled_joint_weights_ramp_in_one_run() -> None:
    assert scheduled_joint_weights(0, 100).local == 0.0
    assert scheduled_joint_weights(15, 100).local > 0.0
    assert scheduled_joint_weights(35, 100).local == 0.25
    assert scheduled_joint_weights(15, 100).mask == 0.0
    assert scheduled_joint_weights(30, 100).mask > 0.0
    assert scheduled_joint_weights(45, 100).mask == 1.0


class TinyTextFeatures:
    def __init__(self, global_embedding: Tensor, token_embeddings: Tensor) -> None:
        self.global_embedding = global_embedding
        self.token_embeddings = token_embeddings
        self.attention_mask = torch.ones(token_embeddings.shape[:2], dtype=torch.long, device=token_embeddings.device)


class TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pair_table = nn.Parameter(torch.randn(4, 512) * 0.02)
        self.text_table = nn.Parameter(torch.randn(8, 512) * 0.02)
        self.local = nn.Parameter(torch.randn(4, 1296, 512) * 0.02)
        self.events = nn.Parameter(torch.randn(4, 4, 512) * 0.02)
        self.mask_logits = nn.Parameter(torch.randn(4, 4, 1296) * 0.02)
        self.presence = nn.Parameter(torch.zeros(4, 4))
        self.semantic = nn.Linear(512, 512)

    def encode_text(self, captions: list[str], role: str = "query") -> TinyTextFeatures:
        indices = torch.arange(len(captions), device=self.text_table.device) % self.text_table.shape[0]
        global_embedding = torch.nn.functional.normalize(self.text_table[indices], dim=-1)
        token_embeddings = global_embedding.unsqueeze(1).expand(-1, 3, -1)
        return TinyTextFeatures(global_embedding, token_embeddings)

    def forward(self, t1: Tensor, t2: Tensor, captions: list[str], text_role: str = "query") -> UniChangeOutput:
        batch_size = t1.shape[0]
        text = self.encode_text(captions, role=text_role)
        pair = torch.nn.functional.normalize(self.pair_table[:batch_size], dim=-1)
        local = torch.nn.functional.normalize(self.local[:batch_size], dim=-1)
        events = torch.nn.functional.normalize(self.events[:batch_size], dim=-1)
        event_masks = torch.sigmoid(self.mask_logits[:batch_size])
        weights = torch.softmax(torch.einsum("nd,bkd->nbk", text.global_embedding, events), dim=-1)
        text_conditioned = torch.einsum("nbk,bkp->nbp", weights, event_masks)
        return UniChangeOutput(
            global_pair_embedding=pair,
            local_change_tokens=local,
            text_global_embedding=text.global_embedding,
            text_token_embeddings=text.token_embeddings,
            event_embeddings=events,
            event_presence=self.presence[:batch_size],
            event_mask_logits=self.mask_logits[:batch_size],
            event_masks=event_masks,
            text_conditioned_mask=text_conditioned,
            semantic_prediction=torch.nn.functional.normalize(self.semantic(pair), dim=-1),
        )


def _synthetic_batch() -> dict:
    mask_a = torch.zeros(36, 36)
    mask_a[:10, :10] = 1
    mask_b = torch.zeros(36, 36)
    mask_b[20:, 20:] = 1
    return {
        "pair_ids": ["p0", "p1"],
        "t1": torch.rand(2, 3, 32, 32),
        "t2": torch.rand(2, 3, 32, 32),
        "captions": ["new buildings", "construction", "vegetation changed"],
        "caption_to_pair": torch.tensor([0, 0, 1]),
        "masks": torch.stack([mask_a, mask_b], dim=0),
        "components": [_components(mask_a), _components(mask_b)],
        "temporal_context": [TemporalContext().to_dict(), TemporalContext().to_dict()],
    }


def test_synthetic_joint_training_step_and_checkpoint_resume(tmp_path: Path) -> None:
    model = TinyJointModel()
    trainer = UniChangeJointTrainer(
        model,
        UniChangeJointTrainerConfig(output_dir=tmp_path / "run", epochs=1, total_steps=2, device="cpu"),
    )
    metrics = trainer.train_step(_synthetic_batch(), step=0, total_steps=2)
    assert metrics["finite_loss"] == 1.0
    assert metrics["finite_gradients"] == 1.0
    path = tmp_path / "run" / "last.pt"
    trainer.save_checkpoint(path, 0, metrics)
    assert trainer.load_checkpoint(path) == 0


def test_mask_rendering_writes_panel(tmp_path: Path) -> None:
    out = tmp_path / "panel.png"
    render_training_panel(
        out,
        torch.rand(3, 16, 16),
        torch.rand(3, 16, 16),
        torch.rand(16, 16),
        torch.rand(16, 16),
        torch.rand(2, 16, 16),
        torch.rand(16, 16),
        "new buildings near road",
        ["top1 p0 score=0.9"],
    )
    assert out.exists() and out.stat().st_size > 0
