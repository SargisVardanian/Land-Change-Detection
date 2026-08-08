"""Schema-validated exact-core data helpers for the SigLIP-2 track."""

from .loader import ExactBatch, make_exact_batches, select_epoch_rows
from .manifest import (
    ALLOWED_DATASETS,
    group_rows_by_pair,
    load_exact_core_rows,
    load_exact_pair_rows,
    ordered_id_sha256,
    paired_caption_rows,
    read_jsonl,
)
from .naflex import (
    assert_synchronized_pair_views,
    patch_mask_from_spatial_shapes,
    synchronized_transform_hash,
    validate_patch_budget_sequence,
)
from .runtime import (
    RawFeatureBatch,
    build_relevance_masks,
    encode_real_features,
    encode_real_images,
    encode_real_text,
    processor_image_inputs,
    processor_text_inputs,
)
from .transforms import SynchronizedResize, assert_pair_geometry

__all__ = [
    "ALLOWED_DATASETS",
    "ExactBatch",
    "RawFeatureBatch",
    "SynchronizedResize",
    "assert_pair_geometry",
    "assert_synchronized_pair_views",
    "build_relevance_masks",
    "encode_real_features",
    "encode_real_images",
    "encode_real_text",
    "group_rows_by_pair",
    "load_exact_core_rows",
    "load_exact_pair_rows",
    "make_exact_batches",
    "ordered_id_sha256",
    "paired_caption_rows",
    "patch_mask_from_spatial_shapes",
    "processor_image_inputs",
    "processor_text_inputs",
    "read_jsonl",
    "select_epoch_rows",
    "synchronized_transform_hash",
    "validate_patch_budget_sequence",
]
