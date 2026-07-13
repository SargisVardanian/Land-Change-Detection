from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch
from PIL import Image

from land_change_detection.models.qcpr import segmentation_loss_components
from land_change_detection.temporal_caption_manifest import make_manifest_row
from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset, _semantic_change_mask
from ucv2_retrieval_metrics import RetrievalCorpus, supervised_mask_metrics, temporal_channel_mask_metrics
from ucv2_stage1_next_core import FrequencyBalancedCaptionCollator, retrieval_supervision_selection


def _rgb(path: Path, values: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2, 1))
    image.putdata(values)
    image.save(path)


def _palette(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("P", (2, 1))
    palette = [0] * 768
    palette[3:6] = [255, 0, 0]
    palette[6:9] = [0, 130, 0]  # Both palette entries quantize to similar luminance.
    image.putpalette(palette)
    image.putdata(indices)
    image.save(path)


def test_palette_labels_compare_indices_not_palette_luminance(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _palette(first, [1, 1])
    _palette(second, [2, 1])

    assert _semantic_change_mask(first, second, None).tolist() == [[1.0, 0.0]]


def test_rgb_labels_compare_class_tuples_not_luminance(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _rgb(first, [(255, 0, 0), (0, 0, 0)])
    _rgb(second, [(0, 130, 0), (0, 0, 0)])

    # PIL L conversion maps both first pixels to roughly the same luminance;
    # raw RGB labels must nevertheless be distinct semantic classes.
    assert _semantic_change_mask(first, second, None).tolist() == [[1.0, 0.0]]


def test_target_taxonomy_assigns_explicit_source_weights(tmp_path: Path):
    rgb = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    _rgb(rgb, [(0, 0, 0), (0, 0, 0)])
    Image.new("L", (2, 1), 255).save(mask)
    rows = [
        make_manifest_row(dataset_name="s2looking", split="train", original_id="s2", t1_path=rgb, t2_path=rgb, captions=["New buildings appeared"], mask_path=mask, source_metadata={"seg_supervision_mode": "query_specific"}),
        make_manifest_row(dataset_name="levir_mci", split="train", original_id="levir", t1_path=rgb, t2_path=rgb, captions=["Change"], mask_path=mask),
        make_manifest_row(dataset_name="example", split="train", original_id="none", t1_path=rgb, t2_path=rgb, captions=["Change"]),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = TemporalCaptionManifestDataset(manifest, split="train", image_size=None, output_grid=1)
    items = [dataset[index] for index in range(len(dataset))]
    by_dataset = {item.dataset_name: item.metadata for item in items}
    assert by_dataset["s2looking"]["segmentation_target_kind"] == "query_specific"
    assert by_dataset["s2looking"]["segmentation_supervision_weight"] == 1.0
    assert by_dataset["levir_mci"]["segmentation_target_kind"] == "binary_generic"
    assert by_dataset["levir_mci"]["segmentation_supervision_weight"] == 0.5
    assert by_dataset["example"]["segmentation_target_kind"] == "none"
    assert by_dataset["example"]["segmentation_supervision_weight"] == 0.0


def test_semantic_transition_union_uses_low_generic_weight(tmp_path: Path):
    rgb = tmp_path / "image.png"
    sem1 = tmp_path / "sem1.png"
    sem2 = tmp_path / "sem2.png"
    _rgb(rgb, [(0, 0, 0), (0, 0, 0)])
    Image.new("L", (2, 1), 1).save(sem1)
    Image.new("L", (2, 1), 2).save(sem2)
    row = make_manifest_row(
        dataset_name="second_cc",
        split="train",
        original_id="second",
        t1_path=rgb,
        t2_path=rgb,
        captions=["Buildings changed"],
        semantic_t1_path=sem1,
        semantic_t2_path=sem2,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    item = TemporalCaptionManifestDataset(manifest, split="train", image_size=None, output_grid=1)[0]
    assert item.metadata["segmentation_target_kind"] == "semantic_transition_union"
    assert item.metadata["segmentation_supervision_weight"] == 0.25


def test_generic_targets_have_reduced_loss_weight_and_none_is_excluded():
    logits = torch.zeros(3, 3, 1)
    masks = torch.ones(3, 1, 1)
    components = segmentation_loss_components(
        logits,
        torch.tensor([0, 1, 2]),
        masks,
        ["query_specific", "binary_generic", "none"],
        torch.tensor([1.0, 0.5, 0.0]),
    )
    assert components["query_specific_supervised_pairs"] == 1
    assert components["generic_supervised_pairs"] == 1
    assert components["segmentation_supervised_pairs"] == 2
    assert components["mean_segmentation_weight"] == 0.75
    assert components["generic_change_segmentation_loss"] < components["query_specific_segmentation_loss"]


def test_shared_retrieval_filter_excludes_s2looking_but_collator_keeps_its_mask_supervision():
    class Item:
        def __init__(self, pair_id: str, retrieval: bool, kind: str, weight: float):
            self.pair_id = pair_id
            self.dataset_name = "s2looking" if not retrieval else "levir_mci"
            self.t1 = torch.zeros(3, 2, 2)
            self.t2 = torch.zeros(3, 2, 2)
            self.captions = ["new buildings appeared"]
            self.mask = torch.ones(2, 2)
            self.metadata = {
                "retrieval_supervision": retrieval,
                "segmentation_supervision": weight > 0,
                "segmentation_supervision_weight": weight,
                "segmentation_target_kind": kind,
                "change_type": "appeared",
            }

    batch = FrequencyBalancedCaptionCollator({}, max_captions_per_pair=None, frequency_power=0.0, seed=0, epoch=0, training=False)(
        [Item("retrieval", True, "binary_generic", 0.5), Item("s2", False, "query_specific", 1.0)]
    )
    selected = retrieval_supervision_selection(batch, torch.device("cpu"))
    assert batch["segmentation_target_kinds"] == ["binary_generic", "query_specific"]
    assert batch["segmentation_weights"].tolist() == [0.5, 1.0]
    assert selected["selected_pairs"].tolist() == [0]
    assert selected["selected_queries"].tolist() == [0]


def test_mask_metrics_exclude_unsupervised_rows_from_denominator():
    corpus = RetrievalCorpus(
        pair_embeddings=torch.tensor([[1.0], [1.0]]),
        text_embeddings=torch.tensor([[1.0], [1.0]]),
        caption_to_pair=torch.tensor([0, 1]),
        caption_group_ids=torch.tensor([0, 1]),
        pair_ids=["supervised", "unsupervised"],
        captions=["q1", "q2"],
        pair_mask_fractions=torch.tensor([1.0, 0.0]),
        encode_seconds=0.0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
        dataset_names=["levir_mci", "unknown"],
        patch_tokens=torch.tensor([[[1.0]], [[1.0]]]),
        mask_query_embeddings=torch.tensor([[1.0], [1.0]]),
        pair_masks=torch.tensor([[[1.0]], [[0.0]]]),
        segmentation_target_kinds=["binary_generic", "none"],
        segmentation_weights=torch.tensor([0.5, 0.0]),
        change_types=[None, None],
    )
    metrics = supervised_mask_metrics(corpus)
    assert metrics["mask_count"] == 1
    assert metrics["mask_Dice"] == 1.0
    assert metrics["mask_IoU"] == 1.0
    assert metrics["predicted_mask_area_mean"] == 1.0
    assert metrics["target_mask_area_mean"] == 1.0


def test_temporal_channel_metrics_report_changed_appeared_and_disappeared_separately():
    logits = torch.full((2, 4, 3), -10.0)
    logits[0, :, 0] = 10.0
    logits[0, :, 1] = 10.0
    logits[1, :, 0] = 10.0
    logits[1, :, 2] = 10.0
    corpus = RetrievalCorpus(
        pair_embeddings=torch.ones(2, 1), text_embeddings=torch.ones(2, 1),
        caption_to_pair=torch.tensor([0, 1]), caption_group_ids=torch.tensor([0, 1]),
        pair_ids=["appeared", "disappeared"], captions=["appeared", "disappeared"],
        pair_mask_fractions=torch.ones(2), encode_seconds=0.0,
        peak_allocated_vram_bytes=0, peak_reserved_vram_bytes=0,
        pair_masks=torch.ones(2, 2, 2), changed_masks=torch.ones(2, 2, 2),
        segmentation_weights=torch.ones(2), segmentation_target_kinds=["query_specific", "query_specific"],
        change_types=["appeared", "disappeared"], temporal_explanation_logits=logits,
    )
    metrics = temporal_channel_mask_metrics(corpus)
    assert metrics["mask_temporal_changed_Dice"] == 1.0
    assert metrics["mask_temporal_appeared_Dice"] == 1.0
    assert metrics["mask_temporal_disappeared_Dice"] == 1.0


def test_patchseg_readiness_requires_mixed_supervision_smoke_evidence(tmp_path: Path):
    commit = "a" * 40
    smoke = {
        "real_cluster_smoke_passed": True, "git_commit": commit, "steps_completed": 10,
        "finite_loss": True, "device_type": "cuda", "bf16_active": True, "fake_backbones": False,
        "train_val_disjoint": True, "frozen_grad_violations": [], "missing_gradients": [],
        "checkpoint_roundtrip_passed": True, "image_size": 256, "output_grid": 32,
        "stage1_next": True, "loss": "semantic_soft_target_text_to_pair", "stable_caption_groups": True,
        "use_direction_embeddings": True, "use_explicit_change_fusion": True, "trainable_temperature": True,
        "temporal_depth": 6, "text_adapter_enabled": True, "text_max_length": 256,
        "slurm_job_id": "1", "slurm_job_name": "smoke", "gpu_name": "H100",
        "data_mode": "levir_only", "patch_reranker_available": True, "qcpr_score_mode": "fused",
        "qcpr_gradient_audit_passed": True, "retrieval_supervised_queries": 5,
        "segmentation_supervised_pairs": 4, "supervision_evidence_passed": True,
        "target_kind_counts": {"binary_generic": 4}, "target_source_counts": {"binary_generic": 4},
        "query_specific_segmentation_losses": [0.0], "generic_change_segmentation_losses": [0.5],
    }
    memory = {
        "status": "PASS", "git_commit": commit, "stage1_next": True,
        "loss": "semantic_soft_target_text_to_pair", "stable_caption_groups": True,
        "use_direction_embeddings": True, "use_explicit_change_fusion": True,
        "trainable_temperature": True, "temporal_depth": 6, "text_adapter_enabled": True,
        "text_max_length": 256, "memory_data_mode": "shape_probe", "recommended_batch_size": 32,
        "patch_reranker_available": True, "qcpr_score_mode": "fused",
    }
    smoke_path, memory_path = tmp_path / "smoke.json", tmp_path / "memory.json"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    memory_path.write_text(json.dumps(memory), encoding="utf-8")
    command = [
        sys.executable, "scripts/ucv2_stage1_next_readiness_gate.py", str(smoke_path), str(memory_path),
        commit, "1", "32", "6", "--expected-patch-reranker",
    ]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, env={"PYTHONPATH": f"{root}/src:{root}/scripts:{root}"})
    assert result.returncode == 0, result.stderr
    smoke["retrieval_supervised_queries"] = 0
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True, env={"PYTHONPATH": f"{root}/src:{root}/scripts:{root}"})
    assert failed.returncode != 0
    assert "retrieval-supervised queries" in failed.stderr
