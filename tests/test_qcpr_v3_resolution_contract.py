import torch
import json
import numpy as np
from PIL import Image

from land_change_detection.training.qcpr_v3_resolution_contract import (
    apply_spatial_contract,
    foreground_preserving_target,
    reverse_directional_sample,
    target_aware_crop_box,
)
from land_change_detection.training.temporal_caption_dataset import TemporalCaptionManifestDataset


def test_foreground_survives_canonical_downsampling():
    mask = torch.zeros(256, 256); mask[17, 201] = 1
    target = foreground_preserving_target(mask, 32)
    assert target.sum() == 1


def test_crop_uses_identical_geometry_and_nearest_mask():
    mask = torch.zeros(1024, 1024); mask[100:108, 900:908] = 1
    t1 = torch.zeros(3, 1024, 1024); t2 = torch.ones_like(t1)
    box = target_aware_crop_box(mask, output_size=256, context=2)
    a, b, target = apply_spatial_contract(t1, t2, mask, box)
    assert a.shape == b.shape == (3, 256, 256) and target.shape == (256, 256)
    assert set(target.unique().tolist()) <= {0.0, 1.0}
    assert target.sum() > 0 and a.max() == 0 and b.min() == 1


def test_jittered_crop_is_deterministic_and_never_drops_target():
    mask = torch.zeros(1024, 1024); mask[400:440, 600:650] = 1
    first = target_aware_crop_box(mask, output_size=256, context=4, jitter_seed=17)
    second = target_aware_crop_box(mask, output_size=256, context=4, jitter_seed=17)
    assert first == second
    y0, y1, x0, x1 = first
    assert bool(mask[y0:y1, x0:x1].sum() == mask.sum())


def test_temporal_reversal_is_atomic_and_involutive():
    t1, t2 = torch.tensor([1]), torch.tensor([2])
    appeared, disappeared = torch.tensor([3]), torch.tensor([4])
    first = reverse_directional_sample(t1, t2, appeared, disappeared, "new buildings appeared")
    second = reverse_directional_sample(*first[:4], first[4])
    for actual, expected in zip(second[:4], (t1, t2, appeared, disappeared), strict=True):
        torch.testing.assert_close(actual, expected)
    assert "appeared" in second[4]


def test_manifest_dataset_applies_native_target_aware_crop(tmp_path):
    first = np.zeros((1024, 1024, 3), dtype=np.uint8)
    second = first.copy(); second[900:920, 900:920] = 255
    mask = np.zeros((1024, 1024), dtype=np.uint8); mask[900:920, 900:920] = 255
    for name, array in (("t1.png", first), ("t2.png", second), ("mask.png", mask)):
        Image.fromarray(array).save(tmp_path / name)
    row = {"schema_version":"temporal-caption-manifest-v1","pair_id":"s2:test:1:appeared","dataset_name":"s2looking","split":"train","caption_source":"semantic_template","captions":["new buildings appeared"],"t1_path":str(tmp_path/"t1.png"),"t2_path":str(tmp_path/"t2.png"),"mask_path":str(tmp_path/"mask.png"),"query_mask_path":str(tmp_path/"mask.png"),"seg_supervision_mode":"query_specific"}
    manifest=tmp_path/"manifest.jsonl"; manifest.write_text(json.dumps(row)+"\n")
    item=TemporalCaptionManifestDataset(manifest,image_size=256,target_aware_mask_crop=True)[0]
    assert item.t1.shape == item.t2.shape == (3,256,256)
    assert item.mask.shape == (256,256) and item.mask.sum() > 4
    assert item.metadata["target_aware_crop_box"] is not None
