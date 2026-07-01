from __future__ import annotations

from land_change_detection.data.temporal_dataset_catalog import (
    DATASETS,
    DatasetRole,
    datasets_for_role,
    joint_text_mask_datasets,
)


def test_dataset_catalog_names_are_unique() -> None:
    names = [item.name for item in DATASETS]
    assert len(names) == len(set(names))


def test_joint_datasets_have_text_and_pixel_supervision() -> None:
    joint = joint_text_mask_datasets()
    assert {item.name for item in joint} >= {"LEVIR-MCI", "SECOND-CC"}
    assert all(item.has_text for item in joint)
    assert all(item.has_binary_masks or item.has_semantic_masks for item in joint)


def test_priority_filter_preserves_only_requested_tier() -> None:
    ssl = datasets_for_role(DatasetRole.TEMPORAL_SSL, max_priority=1)
    assert ssl
    assert all(item.priority <= 1 for item in ssl)


def test_large_language_alignment_sources_are_not_marked_temporal() -> None:
    general_text = datasets_for_role(DatasetRole.EO_IMAGE_TEXT)
    assert general_text
    assert all(item.temporal_depth.startswith("1") for item in general_text)
