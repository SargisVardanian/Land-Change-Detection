from __future__ import annotations

import io
import json

import pytest
import torch
from PIL import Image

from qcpr_siglip2.config.schema import Siglip2TemporalConfig
from qcpr_siglip2.data.naflex import validate_patch_budget_sequence
from qcpr_siglip2.data.runtime import processor_image_inputs
from qcpr_siglip2.data.manifest import load_exact_pair_rows
from qcpr_siglip2.models.model import Siglip2TemporalRetrievalModel
from qcpr_siglip2.models.temporal import TemporalTransformerAdapter


def _config(**overrides) -> Siglip2TemporalConfig:
    values = {
        "hidden_size": 32,
        "attention_heads": 4,
        "mlp_size": 128,
        "evidence_query_chunk_size": 2,
        "evidence_pair_chunk_size": 2,
        "large_scene_latents": 8,
    }
    values.update(overrides)
    return Siglip2TemporalConfig(**values).validate()


def _features(n: int, *, batch: int = 2, queries: int = 3):
    torch.manual_seed(41)
    return (
        torch.randn(batch, 2, n, 32),
        torch.randn(batch, 2, 32),
        torch.randn(queries, 5, 32),
        torch.randn(queries, 32),
        torch.ones(queries, 5, dtype=torch.bool),
    )


@pytest.mark.parametrize("patch_count", [256, 576, 1024])
def test_direct_naflex_budgets_return_one_pair_vector(patch_count: int):
    model = Siglip2TemporalRetrievalModel(None, _config())
    output = model.forward_from_features(*_features(patch_count))
    assert output.pair_cls.shape == (2, 32)
    assert output.temporal.temporal_patch_tokens.shape == (2, 2 * patch_count, 32)
    assert output.temporal.reduced is False
    assert output.evidence.evidence_map_valid_mask.shape[:2] == (2, 2)


def test_rectangular_grid_and_padding_mask_are_preserved():
    model = Siglip2TemporalRetrievalModel(None, _config())
    frame_tokens, frame_embeddings, text_tokens, text_embeddings, text_mask = _features(
        16, batch=2
    )
    shapes = torch.tensor([[[3, 5], [3, 5]], [[2, 4], [2, 4]]])
    valid = torch.zeros(2, 2, 16, dtype=torch.bool)
    valid[0, :, :15] = True
    valid[1, :, :8] = True
    output = model.forward_from_features(
        frame_tokens,
        frame_embeddings,
        text_tokens,
        text_embeddings,
        text_mask,
        patch_valid_mask=valid,
        spatial_shapes=shapes,
    )
    assert output.evidence.evidence_map.shape[-2:] == (3, 5)
    assert torch.equal(output.temporal.patch_valid_mask, valid)
    assert torch.equal(output.temporal.spatial_shapes, shapes)


def test_padded_token_values_cannot_change_embedding_or_evidence():
    model = Siglip2TemporalRetrievalModel(None, _config()).eval()
    frame_tokens, frame_embeddings, text_tokens, text_embeddings, text_mask = _features(
        16, batch=1, queries=2
    )
    shapes = torch.tensor([[[3, 4], [3, 4]]])
    valid = torch.zeros(1, 2, 16, dtype=torch.bool)
    valid[:, :, :12] = True
    first = model.forward_from_features(
        frame_tokens,
        frame_embeddings,
        text_tokens,
        text_embeddings,
        text_mask,
        patch_valid_mask=valid,
        spatial_shapes=shapes,
    )
    changed = frame_tokens.clone()
    changed[:, :, 12:] = torch.randn_like(changed[:, :, 12:]) * 1000
    second = model.forward_from_features(
        changed,
        frame_embeddings,
        text_tokens,
        text_embeddings,
        text_mask,
        patch_valid_mask=valid,
        spatial_shapes=shapes,
    )
    assert torch.equal(first.score_matrix, second.score_matrix)
    assert torch.equal(first.evidence.evidence_map, second.evidence.evidence_map)


def test_large_native_scene_uses_bounded_query_independent_reducer():
    model = Siglip2TemporalRetrievalModel(
        None, _config(direct_patch_token_budget=64, large_scene_latents=8)
    )
    output = model.forward_from_features(*_features(128, batch=1, queries=2))
    assert output.temporal.reduced is True
    assert output.temporal.patch_count == 8
    assert output.temporal.temporal_patch_tokens.shape == (1, 16, 32)
    assert output.pair_cls.shape == (1, 32)
    assert output.temporal.region_assignment is not None
    assert output.temporal.region_assignment.shape == (1, 2, 8, 128)
    assert output.evidence.evidence_map.shape[-2:] == (1, 128)
    assert output.evidence.processed_evidence_map is not None


def test_metadata_and_native_geometry_propagate_through_temporal_adapter():
    model = Siglip2TemporalRetrievalModel(None, _config()).eval()
    features = _features(16, batch=1, queries=1)
    native_size = torch.tensor([[[512, 768], [512, 768]]])
    frame_ids = torch.tensor([[10.0, 20.0]])
    sensor_ids = torch.tensor([[2, 2]])
    gsd = torch.tensor([[0.5, 0.5]])
    output = model.forward_from_features(
        *features,
        timestamps=torch.tensor([[100.0, 130.0]]),
        spatial_shapes=torch.tensor([[[3, 5], [3, 5]]]),
        native_image_size=native_size,
        processed_patch_grid=torch.tensor([[[3, 5], [3, 5]]]),
        transform_hash="transform-a",
        frame_ids=frame_ids,
        sensor_ids=sensor_ids,
        gsd=gsd,
        metadata_missing=torch.zeros(1, 2, 4, dtype=torch.bool),
    )
    temporal = output.temporal
    assert torch.equal(temporal.native_image_size, native_size)
    assert temporal.transform_hash == "transform-a"
    assert torch.equal(temporal.frame_ids, frame_ids)
    assert torch.equal(temporal.sensor_ids, sensor_ids)
    assert torch.equal(temporal.gsd, gsd)
    assert temporal.native_token_coordinates is not None


def test_incompatible_temporal_patch_grids_are_rejected():
    model = Siglip2TemporalRetrievalModel(None, _config())
    features = _features(16, batch=1, queries=1)
    shapes = torch.tensor([[[3, 5], [2, 8]]])
    valid = torch.zeros(1, 2, 16, dtype=torch.bool)
    valid[0, 0, :15] = True
    valid[0, 1, :16] = True
    with pytest.raises(ValueError, match="compatible patch grid"):
        model.forward_from_features(
            *features,
            patch_valid_mask=valid,
            spatial_shapes=shapes,
        )


def test_patch_budget_contract_is_explicit_and_deterministic():
    first = validate_patch_budget_sequence([256, 576, 1024], supported=(256, 576, 1024))
    second = validate_patch_budget_sequence([256, 576, 1024], supported=(256, 576, 1024))
    assert first == second
    with pytest.raises(ValueError):
        validate_patch_budget_sequence([256, 512], supported=(256, 576, 1024))


class _FakeNaflexProcessor:
    class _ImageProcessor:
        max_num_patches = 256

    image_processor = _ImageProcessor()

    def __init__(self):
        self.seen_budget = None

    def __call__(self, *, images, return_tensors, max_num_patches=None):
        self.seen_budget = max_num_patches
        count = int(max_num_patches or 256)
        return {
            "pixel_values": torch.zeros(len(images), count, 3),
            "pixel_attention_mask": torch.ones(len(images), count, dtype=torch.bool),
            "spatial_shapes": torch.tensor([[int(count**0.5), int(count**0.5)]] * len(images)),
        }


class _MismatchedGridProcessor(_FakeNaflexProcessor):
    def __call__(self, *, images, return_tensors, max_num_patches=None):
        self.seen_budget = max_num_patches
        count = int(max_num_patches or 256)
        shapes = [[3, 5], [5, 3]]
        mask = torch.zeros(len(images), count, dtype=torch.bool)
        mask[:, :15] = True
        return {
            "pixel_values": torch.zeros(len(images), count, 3),
            "pixel_attention_mask": mask,
            "spatial_shapes": torch.tensor(shapes[: len(images)]),
        }


def test_naflex_processor_refuses_implicit_256_and_accepts_explicit_budget(tmp_path):
    for name in ("t1.png", "t2.png"):
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(tmp_path / name)
    rows = [{"t1_path": str(tmp_path / "t1.png"), "t2_path": str(tmp_path / "t2.png")}]
    processor = _FakeNaflexProcessor()
    with pytest.raises(ValueError, match="explicit max_num_patches"):
        processor_image_inputs(processor, rows, torch.device("cpu"))
    result = processor_image_inputs(
        processor, rows, torch.device("cpu"), max_num_patches=256
    )
    assert processor.seen_budget == 256
    assert result["pixel_values"].shape == (1, 2, 256, 3)


def test_processor_rejects_mismatched_t1_t2_grid(tmp_path):
    for name in ("t1.png", "t2.png"):
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(tmp_path / name)
    rows = [{"t1_path": str(tmp_path / "t1.png"), "t2_path": str(tmp_path / "t2.png")}]
    with pytest.raises(ValueError, match="compatible processed NaFlex patch grid"):
        processor_image_inputs(
            _MismatchedGridProcessor(), rows, torch.device("cpu"), max_num_patches=256
        )


def test_processor_accepts_variable_length_temporal_items(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (64, 64), color=(10 + index, 20, 30)).save(path)
        paths.append(str(path))
    rows = [{"frames": paths}]
    processor = _FakeNaflexProcessor()
    result = processor_image_inputs(
        processor,
        rows,
        torch.device("cpu"),
        max_num_patches=256,
        synchronized_sequence_transform=lambda frames: frames,
    )
    assert result["pixel_values"].shape == (1, 3, 256, 3)
    assert result["pixel_attention_mask"].shape == (1, 3, 256)
    assert result["spatial_shapes"].shape == (1, 3, 2)


def test_r19g_query_and_physical_registry_schema_is_normalized(tmp_path):
    root = tmp_path
    (root / "registries").mkdir()
    frame_a = root / "a.png"
    frame_b = root / "b.png"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(frame_a)
    Image.new("RGB", (32, 32), color=(3, 2, 1)).save(frame_b)
    item_id = "levir_mci:train:item-1"
    registry_row = {
        "item_id": item_id,
        "frames": [
            {"native_path": str(frame_a), "timestamp": "t1"},
            {"native_path": str(frame_b), "timestamp": "t2"},
        ],
    }
    (root / "registries" / "physical_items.jsonl").write_text(
        json.dumps(registry_row) + "\n", encoding="utf-8"
    )
    manifest = root / "exact_core_train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "query_id": "q-1",
                "text": "a building appeared",
                "positive_item_ids": [item_id],
                "query_scope": "exact",
                "split": "train",
                "verification": "human",
                "provenance": {"source_dataset": "levir_mci"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = load_exact_pair_rows(manifest, split="train")[0]
    assert row["dataset_name"] == "levir_mci"
    assert row["caption"] == "a building appeared"
    assert row["canonical_pair_id"] == item_id
    assert row["caption_id"] == "q-1"
    assert row["t1_path"] == str(frame_a)
    assert row["t2_path"] == str(frame_b)
    assert row["frames"] == [str(frame_a), str(frame_b)]


def test_processor_rejects_mixed_sequence_lengths(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path)
        paths.append(str(path))
    rows = [{"frames": paths[:2]}, {"frames": paths}]
    with pytest.raises(ValueError, match="same frame count"):
        processor_image_inputs(
            _FakeNaflexProcessor(),
            rows,
            torch.device("cpu"),
            max_num_patches=256,
        )


def test_temporal_reversal_is_not_silently_ignored_and_double_reversal_restores():
    adapter = TemporalTransformerAdapter(_config(max_frames=4))
    tokens, embeddings, *_ = _features(256, batch=1, queries=1)
    forward = adapter(tokens, embeddings)
    reverse = adapter(tokens.flip(1), embeddings.flip(1))
    restored = adapter(tokens.flip(1).flip(1), embeddings.flip(1).flip(1))
    assert not torch.allclose(forward.pair_cls, reverse.pair_cls)
    assert torch.allclose(forward.pair_cls, restored.pair_cls)


def test_evidence_path_has_nonzero_gradient_and_checkpoint_roundtrip():
    torch.manual_seed(52)
    config = _config()
    model = Siglip2TemporalRetrievalModel(None, config).eval()
    inputs = _features(16, batch=2, queries=3)
    first = model.forward_from_features(*inputs)
    first.score_matrix.sum().backward()
    assert model.evidence_bottleneck.raw_gate.grad is not None
    assert model.temporal_adapter.blocks[0].attn_scale.grad is not None
    assert model.temporal_adapter.direction_scale.grad is not None

    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    restored = Siglip2TemporalRetrievalModel(None, config).eval()
    restored.load_state_dict(torch.load(io.BytesIO(payload.getvalue()), weights_only=True))
    with torch.no_grad():
        second = restored.forward_from_features(*inputs)
    assert torch.equal(first.score_matrix.detach(), second.score_matrix)


def test_pair_embedding_is_one_deterministic_ann_vector_per_item():
    torch.manual_seed(61)
    model = Siglip2TemporalRetrievalModel(None, _config()).eval()
    frame_tokens, frame_embeddings, *_ = _features(16, batch=4, queries=1)
    with torch.no_grad():
        first = model.temporal_adapter(frame_tokens, frame_embeddings).pair_cls
        second = model.temporal_adapter(frame_tokens, frame_embeddings).pair_cls
    assert first.shape == (4, 32)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(4))
